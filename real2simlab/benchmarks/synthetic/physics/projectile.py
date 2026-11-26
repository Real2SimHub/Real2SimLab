"""A ball thrown with quadratic air drag that bounces on the floor and rolls to a stop.

In flight, dv/dt = -g z - k |v| v, integrated with an 8th-order Runge-Kutta method (relative
tolerance 1e-11) and exact impact detection. An impact reverses the vertical speed times the
restitution and keeps a fraction of the horizontal speed; once the rebound is slower than
``rest_speed`` the ball rolls with constant deceleration until it stops.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from . import render
from .common import FPS, look_at, object_value, times

NAME = "projectile"
TITLE = "Thrown ball with air drag"
OBJECTS = ("ball",)
DURATION_S = 2.0
RADIUS = 0.06
START = np.array([-1.0, 0.0, 0.3])
GRAVITY = 9.81
TRUE_PARAMS = {
    "drag": 0.113,
    "restitution": 0.583,
    "tangential": 0.837,
    "rolling": 0.371,
    "rest_speed": 0.15,
}
GUESS_PARAMS = {
    "drag": 0.0,
    "restitution": 0.6,
    "tangential": 1.0,
    "rolling": 0.4,
    "rest_speed": 0.15,
}
SETUP = {"initial_velocity": {"ball": [2.0, 0.0, 2.5]}}
SCENE_TEXT = (
    "A red ball (object `ball`, diameter 0.12 m) is thrown from 0.3 m above the floor at "
    "x = -1 m, flies with noticeable air drag, bounces on the checkered floor and rolls."
)
GOAL_TEXT = "Reproduce the throw."
CAMERA = look_at(eye=(0.15, -2.6, 0.7), target=(0.15, 0.0, 0.45), fov_deg=70.0)
COLOR = (0.85, 0.16, 0.10)
METRICS_EXTRA = ("event_first_contact", "event_bounce_count")


def trajectory(velocity: np.ndarray, params: dict[str, Any], t: np.ndarray) -> np.ndarray:
    k = float(params["drag"])
    e, tangential = float(params["restitution"]), float(params["tangential"])
    rolling, rest_speed = float(params["rolling"]), float(params["rest_speed"])
    end = float(t[-1])
    out = np.zeros((len(t), 3))

    def rhs(_: float, y: np.ndarray) -> np.ndarray:
        v = y[3:]
        speed = math.sqrt(float(v @ v))
        return np.concatenate(
            [v, [-k * speed * v[0], -k * speed * v[1], -GRAVITY - k * speed * v[2]]]
        )

    def ground(_: float, y: np.ndarray) -> float:
        return float(y[2] - RADIUS)

    ground.terminal = True  # type: ignore[attr-defined]
    ground.direction = -1  # type: ignore[attr-defined]

    state = np.concatenate([START, velocity]).astype(np.float64)
    clock = 0.0
    while clock < end:
        sol = solve_ivp(
            rhs, (clock, end + 1.0), state, method="DOP853", rtol=1e-11, atol=1e-12,
            events=ground, dense_output=True,
        )  # fmt: skip
        stop = float(sol.t_events[0][0]) if sol.t_events[0].size else end + 1.0
        mask = (t >= clock) & (t < stop)
        if mask.any():
            out[mask] = sol.sol(t[mask])[:3].T
        if stop > end:
            return out
        state = np.asarray(sol.y_events[0][0], dtype=np.float64)
        clock = stop
        vz = -e * state[5]
        state[3:5] *= tangential
        state[5] = vz
        state[2] = RADIUS
        if vz < rest_speed:
            break
    # Roll on the floor with constant deceleration.
    position = state[:3].copy()
    horizontal = state[3:5].copy()
    speed = float(np.linalg.norm(horizontal))
    direction = horizontal / speed if speed > 0 else np.zeros(2)
    stop_time = speed / rolling if rolling > 0 else math.inf
    mask = t >= clock
    tau = np.minimum(t[mask] - clock, stop_time)
    distance = speed * tau - 0.5 * rolling * tau * tau
    out[mask, 0] = position[0] + direction[0] * distance
    out[mask, 1] = position[1] + direction[1] * distance
    out[mask, 2] = RADIUS
    return out


def simulate(
    setup: dict[str, Any],
    params: dict[str, Any],
    fps: float = FPS,
    num_frames: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    velocity = np.asarray(
        object_value(setup, "initial_velocity", "ball", SETUP["initial_velocity"]["ball"]),
        dtype=np.float64,
    )
    frames = num_frames if num_frames is not None else round(DURATION_S * fps)
    return {"ball": {"position": trajectory(velocity, params, times(frames, fps))}}


def static_scene(setup: dict[str, Any]) -> list[render.Primitive]:
    return []


def dynamic_scene(tracks: dict[str, dict[str, np.ndarray]], frame: int) -> list[render.Primitive]:
    return [render.Sphere(tracks["ball"]["position"][frame], RADIUS, COLOR)]


def scene_objects(setup: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": "ball",
            "role": "dynamic",
            "shape": "sphere",
            "size": [2 * RADIUS] * 3,
            "color": "red",
            "initial_pose": [*START.tolist(), 1.0, 0.0, 0.0, 0.0],
        },
        {"name": "floor", "role": "static", "shape": "plane z = 0"},
    ]
