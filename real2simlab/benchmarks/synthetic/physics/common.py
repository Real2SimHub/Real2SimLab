"""Shared helpers of the synthetic truth generators. NumPy only; no Real2SimLab imports.

The ``physics`` package is copied verbatim into generator twins, so it must run inside the
``r2s-python`` image without Real2SimLab installed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

FPS = 30.0
WIDTH, HEIGHT = 320, 240


def frames_for(duration: float, fps: float = FPS) -> int:
    """Frames of a span of ``duration`` seconds."""
    return round(duration * fps)


def times(num_frames: int, fps: float = FPS) -> np.ndarray:
    return np.arange(num_frames, dtype=np.float64) / fps


def object_value(setup: dict[str, Any], name: str, obj: str, default: Any) -> Any:
    """The value of the per-object setting ``name`` for ``obj``, or ``default``."""
    value = setup.get(name)
    if isinstance(value, dict) and obj in value:
        return value[obj]
    return default


def scene_value(setup: dict[str, Any], name: str, default: Any) -> Any:
    return setup.get(name, default)


def quat_from_axis_angle(axis: Any, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2
    return np.array([math.cos(half), *(math.sin(half) * axis)])


def quat_to_matrix(q: Any) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def look_at(
    eye: Any, target: Any, fov_deg: float, width: int = WIDTH, height: int = HEIGHT
) -> dict[str, Any]:
    """A camera as written to ``camera.json``: OpenCV axes, camera-to-world extrinsics."""
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, down, forward, eye
    focal = (width / 2) / math.tan(math.radians(fov_deg) / 2)
    k = [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]]
    return {"intrinsics": k, "width": width, "height": height, "extrinsics": pose.tolist()}
