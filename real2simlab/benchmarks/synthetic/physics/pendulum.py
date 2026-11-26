"""A damped pendulum: a bob on a rigid, massless rod, released at rest from an angle.

Integrates theta'' = -(g / L) sin(theta) - c theta' with an 8th-order Runge-Kutta method at a
relative tolerance of 1e-11, so large release angles (far outside the small-angle regime) are
exact to well below a pixel.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from . import render
from .common import FPS, look_at, object_value, times

NAME = "pendulum"
TITLE = "Pendulum"
OBJECTS = ("bob",)
DURATION_S = 3.0
PIVOT = np.array([0.0, 0.0, 1.0])
LENGTH = 0.6
RADIUS = 0.05
GRAVITY = 9.81
TRUE_PARAMS = {"damping": 0.137}
GUESS_PARAMS = {"damping": 0.0}
SETUP = {"initial_angle": {"bob": 0.35}}
SCENE_TEXT = (
    "An orange bob (object `bob`, diameter 0.1 m) on a thin rigid rod of length 0.6 m hangs "
    "from a pivot 1.0 m above the floor; it is released at rest and swings in the x-z plane."
)
GOAL_TEXT = "Reproduce the swing."
CAMERA = look_at(eye=(0.0, -2.1, 0.75), target=(0.0, 0.0, 0.72), fov_deg=60.0)
COLOR = (0.95, 0.55, 0.10)
ROD_COLOR = (0.25, 0.25, 0.28)
METRICS_EXTRA: tuple[str, ...] = ()


def angles(theta0: float, params: dict[str, Any], t: np.ndarray) -> np.ndarray:
    damping = float(params["damping"])

    def rhs(_: float, y: np.ndarray) -> list[float]:
        return [y[1], -(GRAVITY / LENGTH) * math.sin(y[0]) - damping * y[1]]

    solution = solve_ivp(
        rhs, (0.0, float(t[-1])), [theta0, 0.0], t_eval=t, method="DOP853", rtol=1e-11, atol=1e-12
    )
    return np.asarray(solution.y[0])


def simulate(
    setup: dict[str, Any],
    params: dict[str, Any],
    fps: float = FPS,
    num_frames: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    theta0 = float(object_value(setup, "initial_angle", "bob", SETUP["initial_angle"]["bob"]))
    frames = num_frames if num_frames is not None else round(DURATION_S * fps)
    theta = angles(theta0, params, times(frames, fps))
    x = PIVOT[0] + LENGTH * np.sin(theta)
    z = PIVOT[2] - LENGTH * np.cos(theta)
    position = np.stack([x, np.zeros_like(x), z], axis=1)
    return {"bob": {"position": position}}


def static_scene(setup: dict[str, Any]) -> list[render.Primitive]:
    mount = render.Box(
        PIVOT + np.array([0, 0, 0.02]), np.eye(3), np.array([0.08, 0.03, 0.02]), ROD_COLOR
    )
    return [mount]


def dynamic_scene(tracks: dict[str, dict[str, np.ndarray]], frame: int) -> list[render.Primitive]:
    bob = tracks["bob"]["position"][frame]
    return [render.Capsule(PIVOT, bob, 0.008, ROD_COLOR), render.Sphere(bob, RADIUS, COLOR)]


def scene_objects(setup: dict[str, Any]) -> list[dict[str, Any]]:
    theta0 = float(object_value(setup, "initial_angle", "bob", 0.35))
    start = [PIVOT[0] + LENGTH * math.sin(theta0), 0.0, PIVOT[2] - LENGTH * math.cos(theta0)]
    return [
        {
            "name": "bob",
            "role": "dynamic",
            "shape": "sphere on a rigid rod",
            "size": [2 * RADIUS] * 3,
            "color": "orange",
            "initial_pose": [*start, 1.0, 0.0, 0.0, 0.0],
        },
        {
            "name": "pivot",
            "role": "static",
            "shape": f"pivot at {PIVOT.tolist()}, rod length {LENGTH} m",
        },
        {"name": "floor", "role": "static", "shape": "plane z = 0"},
    ]
