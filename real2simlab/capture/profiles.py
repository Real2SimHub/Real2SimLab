"""Observation profiles and input requirements.

A profile names what an observation offers (video, calibration, depth...). Profiles are not a
linear ladder, so a capture has a set of them, detected from which fields it carries.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from real2simlab._base import StrictModel
from real2simlab.capture import paths
from real2simlab.capture.model import Capture, FieldPath


class Profile(StrEnum):
    """What an observation offers."""

    RGB = "rgb"
    """Every view has RGB video."""
    RGB_CALIB = "rgb_calib"
    """Every view is calibrated (intrinsics and extrinsics) and geometry is metric."""
    RGBD = "rgbd"
    """Calibrated, and at least one view has depth."""
    MULTIVIEW = "multiview"
    """Calibrated, with two or more views."""
    MOVING_CAMERA = "moving_camera"
    """At least one view has per-frame extrinsics."""
    ROBOT = "robot"
    """A time-synchronised robot recording is present."""


_CALIBRATED = frozenset(
    {
        "views.*.rgb",
        "views.*.intrinsics",
        "views.*.extrinsics",
        "views.*.extrinsics_per_frame",
        "world.metric_scale",
    }
)

PROFILE_FIELDS: Mapping[Profile, frozenset[str]] = {
    Profile.RGB: frozenset({"views.*.rgb"}),
    Profile.RGB_CALIB: _CALIBRATED,
    Profile.RGBD: _CALIBRATED | {"views.*.depth"},
    Profile.MULTIVIEW: _CALIBRATED,
    Profile.MOVING_CAMERA: _CALIBRATED,
    Profile.ROBOT: frozenset({"robot"}),
}
"""The fields a method may read when it relies on each profile."""


def detect_profiles(capture: Capture) -> frozenset[Profile]:
    """The profiles a capture satisfies."""
    views = list(capture.views.values())
    found = {Profile.RGB}
    metric = capture.world.metric_scale is not None and capture.world.metric_scale.value
    if metric and all(view.calibrated for view in views):
        found.add(Profile.RGB_CALIB)
        if any(view.depth for view in views):
            found.add(Profile.RGBD)
        if len(views) >= 2:
            found.add(Profile.MULTIVIEW)
    if any(view.extrinsics_per_frame is not None for view in views):
        found.add(Profile.MOVING_CAMERA)
    if capture.robot is not None:
        found.add(Profile.ROBOT)
    return frozenset(found)


class InputSpec(StrictModel):
    """A set of inputs, given as profiles and individual field paths."""

    profiles: frozenset[Profile] = frozenset()
    fields: frozenset[FieldPath] = frozenset()

    def field_paths(self) -> frozenset[str]:
        """Every field path this spec covers, including those implied by its profiles."""
        implied = [PROFILE_FIELDS[profile] for profile in self.profiles]
        return self.fields.union(*implied)

    def missing(self, capture: Capture) -> list[str]:
        """Inputs in this spec that ``capture`` lacks, as ``profile:<name>`` or a field path."""
        absent_profiles = self.profiles - detect_profiles(capture)
        data = capture.to_json()
        absent_fields = [path for path in self.fields if not paths.is_present(data, path)]
        return [f"profile:{profile}" for profile in sorted(absent_profiles)] + sorted(absent_fields)
