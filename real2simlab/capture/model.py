"""The ``Capture`` (``r2s-capture/1.0``): a complete observation record of one physical event.

A capture lives in a directory: ``capture.json`` plus the media files it references by
relative path. Multiple views are first-class; a monocular capture has a single view. Every
quantity that is not raw media carries its provenance (see ``Sourced``).
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from real2simlab._base import FINITE_CONFIG, StrictModel
from real2simlab.capture import paths
from real2simlab.capture.conventions import (
    DEFAULT_GRAVITY,
    LENGTH_UNIT,
    UP_AXIS,
    Extrinsics,
    Intrinsics,
    Pose,
    Vec3,
)
from real2simlab.capture.provenance import Provenance, Sourced
from real2simlab.spec import SpecId, spec_validator

OBJECT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
"""Object names: letters, digits, ``_`` and ``-``; used as keys in tracks."""


def check_relative_path(value: str) -> str:
    """Raise ``ValueError`` unless ``value`` is a relative POSIX path that stays inside its root."""
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"path must be relative and stay inside its directory: {value!r}")
    return value


def check_object_name(value: str) -> str:
    """Raise ``ValueError`` unless ``value`` is a valid object name."""
    if not OBJECT_NAME.match(value):
        raise ValueError(
            f"object name {value!r} must start with a letter or digit and contain only "
            "letters, digits, '_' and '-'"
        )
    return value


def _check_view_name(value: str) -> str:
    if not value or "." in value or paths.WILDCARD in value:
        raise ValueError(f"view name must be non-empty and contain no '.' or '*': {value!r}")
    return value


def _check_field_path(value: str) -> str:
    root = paths.parse(value)[0]
    if root not in Capture.model_fields:
        raise ValueError(f"field path {value!r} does not start with a Capture field")
    return value


RelPath = Annotated[str, AfterValidator(check_relative_path)]
ObjectName = Annotated[str, AfterValidator(check_object_name)]
ViewName = Annotated[str, AfterValidator(_check_view_name)]
FieldPath = Annotated[str, AfterValidator(_check_field_path)]
"""A dotted path into a Capture, e.g. ``views.*.depth`` (see ``paths``)."""


class VideoStream(StrictModel):
    """An RGB video. ``num_frames`` and ``timestamps`` describe the visible frames."""

    model_config = FINITE_CONFIG

    path: RelPath
    fps: PositiveFloat
    num_frames: PositiveInt
    resolution: tuple[PositiveInt, PositiveInt]
    """Width and height in pixels."""
    timestamps: tuple[float, ...] | None = None
    """Seconds per visible frame; when absent, frame times follow from ``fps``."""

    @model_validator(mode="after")
    def _check_timestamps(self) -> Self:
        stamps = self.timestamps
        if stamps is None:
            return self
        if len(stamps) != self.num_frames:
            raise ValueError(f"{len(stamps)} timestamps for {self.num_frames} frames")
        if any(later <= earlier for earlier, later in itertools.pairwise(stamps)):
            raise ValueError("timestamps must be strictly increasing")
        return self


class DepthStream(StrictModel):
    """Per-frame depth maps aligned with the view's RGB frames."""

    path: RelPath
    unit: Literal["m", "mm"]
    kind: Literal["metric", "relative"]
    provenance: Provenance
    source: str | None = None


class MaskStream(StrictModel):
    """Per-frame object masks aligned with the view's RGB frames."""

    path: RelPath
    format: str = Field(min_length=1)
    """How masks are stored, e.g. ``png_sequence`` or ``npz``."""
    labels: dict[int, str] = Field(default_factory=dict[int, str])
    provenance: Provenance
    source: str | None = None


class PerFrameExtrinsics(StrictModel):
    """Camera-to-world transforms of a moving camera, one per frame of the file.

    ``path`` names a ``.npy`` file of shape ``(F, 4, 4)``, where ``F`` is at least the end of
    the view's visible frame range.
    """

    path: RelPath
    provenance: Provenance
    source: str | None = None


class View(StrictModel):
    """One camera and the per-frame streams it recorded."""

    rgb: VideoStream
    intrinsics: Sourced[Intrinsics] | None = None
    extrinsics: Sourced[Extrinsics] | None = None
    """Camera-to-world transform of a static camera."""
    extrinsics_per_frame: PerFrameExtrinsics | None = None
    """Camera-to-world transforms of a moving camera; excludes ``extrinsics``."""
    depth: DepthStream | None = None
    masks: MaskStream | None = None
    frame_range: tuple[NonNegativeInt, NonNegativeInt] | None = None
    """Visible window ``[start, end)`` as frame indices into every per-frame file of this view."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.frame_range is not None:
            start, end = self.frame_range
            if end - start != self.rgb.num_frames:
                raise ValueError(
                    f"frame_range {self.frame_range} does not span {self.rgb.num_frames} frames"
                )
        if self.extrinsics is not None and self.extrinsics_per_frame is not None:
            raise ValueError("give either extrinsics (static camera) or extrinsics_per_frame")
        return self

    @property
    def calibrated(self) -> bool:
        """Whether intrinsics and some form of extrinsics are known."""
        has_pose = self.extrinsics is not None or self.extrinsics_per_frame is not None
        return self.intrinsics is not None and has_pose

    @property
    def first_frame(self) -> int:
        """File index of the first visible frame."""
        return self.frame_range[0] if self.frame_range else 0

    def frame_time(self, index: int) -> float:
        """Time in seconds of visible frame ``index``, measured from frame 0 of the file."""
        if not 0 <= index < self.rgb.num_frames:
            raise IndexError(f"frame {index} outside 0..{self.rgb.num_frames - 1}")
        if self.rgb.timestamps is not None:
            return self.rgb.timestamps[index]
        return (self.first_frame + index) / self.rgb.fps

    def windowed(self, start: int, end: int) -> View:
        """Restrict every per-frame stream to visible frames ``[start, end)``."""
        if not 0 <= start < end <= self.rgb.num_frames:
            raise ValueError(f"window [{start}, {end}) outside 0..{self.rgb.num_frames}")
        stamps = self.rgb.timestamps
        rgb = self.rgb.model_copy(
            update={"num_frames": end - start, "timestamps": stamps[start:end] if stamps else None}
        )
        first = self.first_frame
        return View.model_validate(
            {**self.to_json(), "rgb": rgb.to_json(), "frame_range": (first + start, first + end)}
        )


class WorldFrame(StrictModel):
    """The world frame the capture's geometry is expressed in."""

    length_unit: Literal["m"] = LENGTH_UNIT
    up_axis: Literal["+z"] = UP_AXIS
    gravity: Sourced[Vec3] = Sourced(value=DEFAULT_GRAVITY, provenance=Provenance.DEFAULT)
    metric_scale: Sourced[bool] | None = None
    """Whether geometry (extrinsics, depth, object sizes) is at true metric scale."""


class SceneObject(StrictModel):
    """An object in the scene and what is known about it."""

    name: ObjectName
    role: Literal["dynamic", "static"]
    shape: str | None = None
    """Free-text or primitive name, e.g. ``sphere``, ``box``, ``rod``."""
    size: Sourced[Vec3] | None = None
    """Extent along the object's local x, y, z axes, in metres (a sphere: its diameter)."""
    mass: Sourced[PositiveFloat] | None = None
    material: Sourced[str] | None = None
    color: str | None = None
    """How the object looks in the video, e.g. ``red``; helps methods find it."""
    initial_pose: Sourced[Pose] | None = None


class SceneInfo(StrictModel):
    """Known scene content beyond the media."""

    objects: tuple[SceneObject, ...] = ()
    environment: str | None = None
    """Free-text description of the static environment."""

    @model_validator(mode="after")
    def _check_unique(self) -> Self:
        names = [item.name for item in self.objects]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate object names: {', '.join(duplicates)}")
        return self


class RobotStream(StrictModel):
    """A robot recording. Trajectory keys follow the XPolicyLab naming convention."""

    model_config = FINITE_CONFIG

    urdf: RelPath
    action_type: Literal["joint", "ee"]
    control_freq: PositiveFloat
    trajectory: RelPath
    time_offset: Sourced[float]
    """Robot time minus video time at frame 0 of the files, in seconds."""


class TextInfo(StrictModel):
    """Optional text accompanying the capture."""

    task: str | None = None
    scene_description: str | None = None


class SourceInfo(StrictModel):
    """Which dataset the capture comes from."""

    dataset: str = Field(min_length=1)
    version: str = Field(min_length=1)
    license: str | None = None
    url: str | None = None


class Capture(StrictModel):
    """A complete observation record of one physical event."""

    spec: Annotated[str, spec_validator(SpecId.CAPTURE)] = SpecId.CAPTURE.value
    capture_id: str = Field(min_length=1)
    source: SourceInfo | None = None
    world: WorldFrame = WorldFrame()
    views: dict[ViewName, View] = Field(min_length=1)
    scene: SceneInfo | None = None
    robot: RobotStream | None = None
    text: TextInfo | None = None
    extensions: dict[str, Any] = Field(default_factory=dict)
    """Rare, method-specific inputs (e.g. joint torques, LiDAR) kept out of the core schema."""

    @model_validator(mode="after")
    def _check_views_aligned(self) -> Self:
        clocks = {(view.rgb.fps, view.rgb.num_frames) for view in self.views.values()}
        timed = all(view.rgb.timestamps is not None for view in self.views.values())
        if len(clocks) > 1 and not timed:
            raise ValueError("views with different fps or frame counts must all provide timestamps")
        return self

    def media_paths(self) -> list[str]:
        """Relative paths of every media file the capture references."""
        files: list[str] = []
        for view in self.views.values():
            files += [stream.path for stream in (view.rgb, view.depth, view.masks) if stream]
            if view.extrinsics_per_frame is not None:
                files.append(view.extrinsics_per_frame.path)
        if self.robot:
            files += [self.robot.urdf, self.robot.trajectory]
        return files

    def object_names(self, role: str | None = "dynamic") -> list[str]:
        """Names of the scene objects with ``role`` (``None``: every object)."""
        if self.scene is None:
            return []
        return [item.name for item in self.scene.objects if role is None or item.role == role]

    def restricted(self, field_paths: Iterable[str]) -> Capture:
        """A copy that keeps only ``field_paths`` (plus the always-visible fields)."""
        return Capture.model_validate(paths.select(self.to_json(), field_paths))

    def windowed(self, start: int, end: int) -> Capture:
        """A copy whose views all expose only visible frames ``[start, end)``."""
        views = {name: view.windowed(start, end).to_json() for name, view in self.views.items()}
        return Capture.model_validate({**self.to_json(), "views": views})
