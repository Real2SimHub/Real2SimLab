"""A ball released from rest above a flat floor, bouncing with a coefficient of restitution.

Exact, event by event: free flight is a parabola; each impact multiplies the vertical speed
by the restitution; the ball comes to rest once its rebound speed falls below ``rest_speed``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import render
from .common import FPS, look_at, object_value, scene_value, times

NAME = "bouncing_ball"
TITLE = "Bouncing ball"
OBJECTS = ("ball",)
DURATION_S = 2.0
RADIUS = 0.05
TRUE_PARAMS = {"restitution": 0.763, "rest_speed": 0.05}
GUESS_PARAMS = {"restitution": 0.70, "rest_speed": 0.05}
SETUP = {"release_height": {"ball": 0.5}, "gravity": 9.81}
SCENE_TEXT = (
    "A red ball (object `ball`, diameter 0.1 m) is released from rest above a flat, checkered "
    "floor and bounces until it settles."
)
GOAL_TEXT = "Reproduce the drop: the ball's fall and every bounce."
CAMERA = look_at(eye=(0.0, -1.6, 0.55), target=(0.0, 0.0, 0.45), fov_deg=60.0)
COLOR = (0.85, 0.16, 0.10)
METRICS_EXTRA = ("event_first_contact", "event_bounce_count", "event_bounce_times")


def heights(height: float, gravity: float, params: dict[str, Any], t: np.ndarray) -> np.ndarray:
    """Centroid height at times ``t`` for a ball released at rest with its centre at ``height``."""
    e, rest_speed = float(params["restitution"]), float(params["rest_speed"])
    z = np.full(len(t), RADIUS)
    start, z0, v0 = 0.0, height, 0.0
    while True:
        # Solve z0 + v0 s - g s^2 / 2 = RADIUS for the next impact time s > 0.
        drop = z0 - RADIUS
        s = (v0 + math.sqrt(v0 * v0 + 2 * gravity * max(drop, 0.0))) / gravity
        mask = (t >= start) & (t < start + s)
        local = t[mask] - start
        z[mask] = z0 + v0 * local - 0.5 * gravity * local**2
        impact_speed = abs(v0 - gravity * s)
        start += s
        rebound = e * impact_speed
        if rebound < rest_speed or start > t[-1] or s < 1e-9:
            z[t >= start] = RADIUS
            return z
        z0, v0 = RADIUS, rebound


def simulate(
    setup: dict[str, Any],
    params: dict[str, Any],
    fps: float = FPS,
    num_frames: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    height = float(object_value(setup, "release_height", "ball", SETUP["release_height"]["ball"]))
    gravity = float(scene_value(setup, "gravity", SETUP["gravity"]))
    frames = num_frames if num_frames is not None else round(DURATION_S * fps)
    z = heights(height, gravity, params, times(frames, fps))
    position = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    return {"ball": {"position": position}}


def static_scene(setup: dict[str, Any]) -> list[render.Primitive]:
    return []


def dynamic_scene(tracks: dict[str, dict[str, np.ndarray]], frame: int) -> list[render.Primitive]:
    return [render.Sphere(tracks["ball"]["position"][frame], RADIUS, COLOR)]


def scene_objects(setup: dict[str, Any]) -> list[dict[str, Any]]:
    height = float(object_value(setup, "release_height", "ball", 0.5))
    return [
        {
            "name": "ball",
            "role": "dynamic",
            "shape": "sphere",
            "size": [2 * RADIUS] * 3,
            "color": "red",
            "initial_pose": [0.0, 0.0, height, 1.0, 0.0, 0.0, 0.0],
        },
        {"name": "floor", "role": "static", "shape": "plane z = 0"},
    ]
