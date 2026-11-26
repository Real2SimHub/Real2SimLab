"""Units, frames and pose conventions shared by all Real2SimLab data.

* Lengths are in metres, time in seconds, angles in radians. The world frame is right-handed
  with +Z up; gravity points along -Z.
* Cameras follow the OpenCV convention: +x right, +y down, +z forward. ``intrinsics`` is a
  3x3 matrix in pixels; the centre of pixel (i, j) is at (i + 0.5, j + 0.5).
  ``extrinsics`` is a 4x4 camera-to-world rigid transform.
* Poses are ``[x, y, z, qw, qx, qy, qz]``; quaternions are ``[qw, qx, qy, qz]``.
* Images are RGB uint8. Frames are indexed from 0 within their media file; frame ``i`` is at
  ``i / fps`` seconds unless timestamps say otherwise.

The annotated types below validate these conventions when data enters the system.
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from numpy.typing import NDArray
from pydantic import AfterValidator

LENGTH_UNIT = "m"
UP_AXIS = "+z"
STANDARD_GRAVITY = 9.80665
DEFAULT_GRAVITY = (0.0, 0.0, -9.81)

TOLERANCE = 1e-4


def finite_array(value: Any, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    """``value`` as a float array of ``shape`` with finite entries; ``ValueError`` otherwise."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def check_intrinsics_array(k: NDArray[np.float64], name: str = "intrinsics") -> None:
    """Raise ``ValueError`` unless ``k`` is a valid pinhole intrinsics matrix."""
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError(f"{name} focal lengths must be positive")
    if not np.allclose(k[2], (0.0, 0.0, 1.0), atol=TOLERANCE):
        raise ValueError(f"{name} last row must be [0, 0, 1]")


def check_rigid_array(t: NDArray[np.float64], name: str = "extrinsics") -> None:
    """Raise ``ValueError`` unless ``t`` is a 4x4 rigid transform (proper rotation)."""
    if not np.allclose(t[3], (0.0, 0.0, 0.0, 1.0), atol=TOLERANCE):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=TOLERANCE) or np.linalg.det(r) <= 0:
        raise ValueError(f"{name} rotation block must be a proper rotation matrix")


def _check_intrinsics(value: list[list[float]]) -> list[list[float]]:
    check_intrinsics_array(finite_array(value, (3, 3), "intrinsics"))
    return value


def _check_extrinsics(value: list[list[float]]) -> list[list[float]]:
    check_rigid_array(finite_array(value, (4, 4), "extrinsics"))
    return value


def _check_pose(value: list[float]) -> list[float]:
    p = finite_array(value, (7,), "pose")
    if abs(float(np.linalg.norm(p[3:])) - 1.0) > TOLERANCE:
        raise ValueError("pose quaternion [qw, qx, qy, qz] must have unit length")
    return value


Intrinsics = Annotated[list[list[float]], AfterValidator(_check_intrinsics)]
"""A 3x3 pinhole intrinsics matrix in pixels."""
Extrinsics = Annotated[list[list[float]], AfterValidator(_check_extrinsics)]
"""A 4x4 camera-to-world rigid transform in metres."""
Pose = Annotated[list[float], AfterValidator(_check_pose)]
"""``[x, y, z, qw, qx, qy, qz]`` in the world frame."""
Vec3 = tuple[float, float, float]
