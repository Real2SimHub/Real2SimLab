"""``r2s capture from-video``: a single-view capture from an ordinary video file."""

from __future__ import annotations

import contextlib
import math
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from real2simlab._base import validation_problems
from real2simlab.capture.conventions import check_rigid_array
from real2simlab.capture.io import CAPTURE_FILENAME, save_capture
from real2simlab.capture.model import (
    Capture,
    SceneInfo,
    SceneObject,
    SourceInfo,
    TextInfo,
    VideoStream,
    View,
    WorldFrame,
    check_object_name,
)
from real2simlab.capture.provenance import Provenance, Sourced
from real2simlab.errors import InputNotFoundError, OutputExistsError, UsageError
from real2simlab.media import VideoInfo, probe_video

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

VIEW = "cam"


def intrinsics_from_fov(width: int, height: int, horizontal_fov_deg: float) -> list[list[float]]:
    """A pinhole matrix with square pixels and the principal point at the image centre."""
    if not 0 < horizontal_fov_deg < 180:
        raise UsageError("the horizontal field of view must be between 0 and 180 degrees")
    focal = (width / 2) / math.tan(math.radians(horizontal_fov_deg) / 2)
    return [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]]


def calibration_problems(
    intrinsics: Sequence[Sequence[float]] | None,
    extrinsics: Sequence[Sequence[float]] | None,
    object_sizes: dict[str, float] | None,
) -> list[str]:
    """Why the given calibration cannot describe a real camera and real objects: intrinsics
    that are not a finite 3x3 pinhole matrix with positive focal lengths, extrinsics that are
    not a finite rigid 4x4 transform (a proper rotation, as a capture holds it), object names a
    capture cannot hold, object sizes that are not positive lengths."""
    import numpy as np

    def matrix(value: object) -> NDArray[np.float64] | None:
        try:  # whatever was given: nested lists of numbers, or anything else
            return np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return None

    problems: list[str] = []
    if intrinsics is not None:
        k = matrix(intrinsics)
        if k is None or k.shape != (3, 3) or not np.all(np.isfinite(k)):
            problems.append("intrinsics must be a finite 3x3 matrix")
        elif k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            problems.append(
                "intrinsics must have positive focal lengths fx, fy and a last row 0 0 1"
            )
    if extrinsics is not None:
        t = matrix(extrinsics)
        if t is None or t.shape != (4, 4) or not np.all(np.isfinite(t)):
            problems.append("extrinsics must be a finite 4x4 matrix (camera-to-world)")
        else:
            try:
                check_rigid_array(t)
            except ValueError:
                problems.append(
                    "extrinsics must be a rigid transform: a rotation (orthonormal, with "
                    "determinant +1: not a mirror) and a translation, last row 0 0 0 1"
                )
    for name, size in (object_sizes or {}).items():
        if not name:
            problems.append("an object size needs the object's name (NAME=SIZE_M)")
            continue
        try:
            check_object_name(name)
        except ValueError as error:
            problems.append(str(error))
        if not (math.isfinite(size) and size > 0):
            problems.append(f"the size of {name!r} must be a positive length in metres, not {size}")
    return problems


def capture_from_video(
    video: Path,
    out_dir: Path,
    *,
    capture_id: str,
    intrinsics: Sequence[Sequence[float]] | None = None,
    intrinsics_provenance: Provenance = Provenance.MEASURED,
    extrinsics: Sequence[Sequence[float]] | None = None,
    object_sizes: dict[str, float] | None = None,
    metric_scale: bool | None = None,
    task_text: str | None = None,
    scene_description: str | None = None,
    source: SourceInfo | None = None,
) -> Capture:
    """Write a capture directory for ``video`` at ``out_dir`` and return the capture.

    The video is copied to ``<out_dir>/cam/rgb.mp4`` (or with its own suffix). ``intrinsics``
    and ``extrinsics`` (camera-to-world) are optional; ``object_sizes`` maps object names to a
    known size in metres, which gives a scale reference. ``metric_scale`` states whether the
    geometry is metric; it defaults to true exactly when extrinsics or object sizes are given.

    Everything is checked before anything is written, and a write that fails leaves nothing
    behind: ``out_dir`` is as it was (missing, or empty), so the same command can run again.
    """
    problems = calibration_problems(intrinsics, extrinsics, object_sizes)
    if not capture_id:
        problems.append("the capture id must not be empty")
    if problems:  # before anything is written: a capture is never certified with them
        raise UsageError("; ".join(problems))
    video, out_dir = Path(video), Path(out_dir)
    if not video.is_file():
        what = "is not a file" if video.exists() else "not found"
        raise InputNotFoundError(f"video {what}: {video}", fix="give the path of a video file")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise OutputExistsError(
            f"output directory is not empty: {out_dir}", fix="choose a new output directory"
        )
    info = probe_video(video)
    relative = f"{VIEW}/rgb{video.suffix.lower() or '.mp4'}"
    try:
        capture = _capture(
            info, relative, capture_id=capture_id, intrinsics=intrinsics,
            intrinsics_provenance=intrinsics_provenance, extrinsics=extrinsics,
            object_sizes=object_sizes, metric_scale=metric_scale, task_text=task_text,
            scene_description=scene_description, source=source,
        )  # fmt: skip
    except ValidationError as error:  # what the checks above do not cover, e.g. a 0 fps video
        raise UsageError(
            f"a capture cannot hold what was given: {'; '.join(validation_problems(error))}"
        ) from None
    existed = out_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:  # this command's: another one writing the same directory meanwhile is refused here
        (out_dir / VIEW).mkdir()
    except FileExistsError:
        raise OutputExistsError(
            f"output directory is not empty: {out_dir} (another command is writing it)",
            fix="choose a new output directory",
        ) from None
    try:
        shutil.copyfile(video, out_dir / relative)
        save_capture(capture, out_dir)
    except BaseException:  # nothing left behind, so the same command can run again
        shutil.rmtree(out_dir / VIEW, ignore_errors=True)
        (out_dir / CAPTURE_FILENAME).unlink(missing_ok=True)
        if not existed:
            with contextlib.suppress(OSError):
                out_dir.rmdir()
        raise
    return capture


def _capture(
    info: VideoInfo,
    relative: str,
    *,
    capture_id: str,
    intrinsics: Sequence[Sequence[float]] | None,
    intrinsics_provenance: Provenance,
    extrinsics: Sequence[Sequence[float]] | None,
    object_sizes: dict[str, float] | None,
    metric_scale: bool | None,
    task_text: str | None,
    scene_description: str | None,
    source: SourceInfo | None,
) -> Capture:
    """The capture of a video probed as ``info`` at ``relative``; nothing is written."""
    view = View(
        rgb=VideoStream(
            path=relative,
            fps=info.fps,
            num_frames=info.num_frames,
            resolution=(info.width, info.height),
        ),
        intrinsics=None
        if intrinsics is None
        else Sourced(value=[list(row) for row in intrinsics], provenance=intrinsics_provenance),
        extrinsics=None
        if extrinsics is None
        else Sourced(value=[list(row) for row in extrinsics], provenance=Provenance.MEASURED),
    )
    if metric_scale is None:
        metric_scale = extrinsics is not None or bool(object_sizes)
    objects = tuple(
        SceneObject(
            name=name,
            role="dynamic",
            size=Sourced(value=(size, size, size), provenance=Provenance.MEASURED),
        )
        for name, size in sorted((object_sizes or {}).items())
    )
    capture = Capture(
        capture_id=capture_id,
        source=source,
        world=WorldFrame(
            metric_scale=Sourced(value=metric_scale, provenance=Provenance.HUMAN_EDITED)
        ),
        views={VIEW: view},
        scene=SceneInfo(objects=objects) if objects else None,
        text=TextInfo(task=task_text, scene_description=scene_description)
        if task_text or scene_description
        else None,
    )
    return capture
