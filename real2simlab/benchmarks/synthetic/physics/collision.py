"""Two balls on a floor: ``ball_a`` rolls into ``ball_b`` at rest and they collide head-on.

Exact and one-dimensional along x: both balls decelerate at a constant rolling resistance
while they move; the collision exchanges momentum with a coefficient of restitution.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import render
from .common import FPS, look_at, object_value, times

NAME = "two_ball_collision"
TITLE = "Two-ball collision"
OBJECTS = ("ball_a", "ball_b")
DURATION_S = 2.5
RADIUS = 0.05
START_A = -0.6
START_B = 0.2
TRUE_PARAMS = {"restitution": 0.827, "rolling": 0.233}
GUESS_PARAMS = {"restitution": 1.0, "rolling": 0.25}
SETUP = {
    "initial_velocity": {"ball_a": [1.2, 0.0, 0.0]},
    "object_mass": {"ball_a": 0.17, "ball_b": 0.17},
}
SCENE_TEXT = (
    "A red ball (object `ball_a`) rolls along +x into a blue ball (object `ball_b`) at rest; "
    "both have a diameter of 0.1 m and a mass of 0.17 kg."
)
GOAL_TEXT = "Reproduce the collision."
CAMERA = look_at(eye=(0.3, -1.9, 0.75), target=(0.3, 0.0, 0.05), fov_deg=60.0)
COLORS = {"ball_a": (0.85, 0.16, 0.10), "ball_b": (0.12, 0.35, 0.85)}
METRICS_EXTRA: tuple[str, ...] = ()


def _motion(x0: float, v0: float, a: float, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity under constant deceleration ``a`` until rest."""
    if v0 == 0:
        return np.full(len(t), x0), np.zeros(len(t))
    stop = abs(v0) / a if a > 0 else math.inf
    tau = np.minimum(t, stop)
    sign = math.copysign(1.0, v0)
    x = x0 + v0 * tau - 0.5 * sign * a * tau * tau
    v = np.where(t < stop, v0 - sign * a * t, 0.0)
    return x, v


def simulate(
    setup: dict[str, Any],
    params: dict[str, Any],
    fps: float = FPS,
    num_frames: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    velocity = object_value(
        setup, "initial_velocity", "ball_a", SETUP["initial_velocity"]["ball_a"]
    )
    v_a = float(velocity[0])
    m_a = float(object_value(setup, "object_mass", "ball_a", SETUP["object_mass"]["ball_a"]))
    m_b = float(object_value(setup, "object_mass", "ball_b", SETUP["object_mass"]["ball_b"]))
    e, a = float(params["restitution"]), float(params["rolling"])
    frames = num_frames if num_frames is not None else round(DURATION_S * fps)
    t = times(frames, fps)
    gap = START_B - START_A - 2 * RADIUS
    # Time for ball_a to cover the gap: gap = v t - a t^2 / 2.
    disc = v_a * v_a - 2 * a * gap
    if disc < 0:
        x_a, _ = _motion(START_A, v_a, a, t)
        x_b = np.full(len(t), START_B)
    else:
        t_hit = (v_a - math.sqrt(disc)) / a if a > 0 else gap / v_a
        u_a = v_a - a * t_hit
        new_a = (m_a * u_a - m_b * e * u_a) / (m_a + m_b)
        new_b = (m_a * u_a + m_a * e * u_a) / (m_a + m_b)
        before = t < t_hit
        x_a = np.where(before, START_A + v_a * t - 0.5 * a * t * t, 0.0)
        after = np.clip(t - t_hit, 0.0, None)
        xa_after, _ = _motion(START_B - 2 * RADIUS, new_a, a, after)
        xb_after, _ = _motion(START_B, new_b, a, after)
        x_a = np.where(before, x_a, xa_after)
        x_b = np.where(before, START_B, xb_after)
    z = np.full(len(t), RADIUS)
    zeros = np.zeros(len(t))
    return {
        "ball_a": {"position": np.stack([x_a, zeros, z], axis=1)},
        "ball_b": {"position": np.stack([x_b, zeros, z], axis=1)},
    }


def static_scene(setup: dict[str, Any]) -> list[render.Primitive]:
    return []


def dynamic_scene(tracks: dict[str, dict[str, np.ndarray]], frame: int) -> list[render.Primitive]:
    return [
        render.Sphere(tracks[name]["position"][frame], RADIUS, COLORS[name]) for name in OBJECTS
    ]


def scene_objects(setup: dict[str, Any]) -> list[dict[str, Any]]:
    masses = {
        name: float(object_value(setup, "object_mass", name, SETUP["object_mass"][name]))
        for name in OBJECTS
    }
    return [
        {
            "name": name,
            "role": "dynamic",
            "shape": "sphere",
            "size": [2 * RADIUS] * 3,
            "mass": masses[name],
            "color": "red" if name == "ball_a" else "blue",
            "initial_pose": [x, 0.0, RADIUS, 1.0, 0.0, 0.0, 0.0],
        }
        for name, x in (("ball_a", START_A), ("ball_b", START_B))
    ] + [{"name": "floor", "role": "static", "shape": "plane z = 0"}]
