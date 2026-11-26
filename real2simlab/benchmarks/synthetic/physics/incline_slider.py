"""A block sliding down an incline with Coulomb friction, then onto the floor until it stops.

Exact: on the slope the block accelerates at g (sin a - mu cos a) if tan a > mu, else it stays
put; when its centroid passes the incline's lowest edge it continues on the floor, keeping its
speed, and decelerates at mu g until it stops. Changing ``incline_angle`` rotates the incline
about its lowest edge; the block keeps its distance along the slope from that edge.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import render
from .common import (
    FPS,
    look_at,
    object_value,
    quat_from_axis_angle,
    quat_to_matrix,
    scene_value,
    times,
)

NAME = "incline_slider"
TITLE = "Block sliding down an incline"
OBJECTS = ("block",)
DURATION_S = 2.0
HALF = np.array([0.05, 0.04, 0.025])
START_DISTANCE = 0.8
INCLINE_LENGTH = 1.1
INCLINE_THICKNESS = 0.02
GRAVITY = 9.81
TRUE_PARAMS: dict[str, float] = {}
GUESS_PARAMS: dict[str, float] = {"friction_bias": 0.05}
SETUP = {"incline_angle": 0.5, "surface_friction": {"block": 0.287}}
SCENE_TEXT = (
    "A blue block (object `block`, 0.10 x 0.08 x 0.05 m) is released at rest on an incline "
    "whose lowest edge lies on the floor along the y axis at x = 0; the block starts 0.8 m up "
    "the slope from that edge, slides down, and continues on the floor."
)
GOAL_TEXT = "Reproduce the slide, down the incline and along the floor."
CAMERA = look_at(eye=(0.0, -2.2, 0.75), target=(0.0, 0.0, 0.2), fov_deg=60.0)
COLOR = (0.15, 0.35, 0.85)
INCLINE_COLOR = (0.62, 0.48, 0.32)
METRICS_EXTRA: tuple[str, ...] = ()


def _friction(setup: dict[str, Any], params: dict[str, Any]) -> float:
    mu = float(object_value(setup, "surface_friction", "block", SETUP["surface_friction"]["block"]))
    return max(0.0, mu + float(params.get("friction_bias", 0.0)))


def simulate(
    setup: dict[str, Any],
    params: dict[str, Any],
    fps: float = FPS,
    num_frames: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    angle = float(scene_value(setup, "incline_angle", SETUP["incline_angle"]))
    mu = _friction(setup, params)
    frames = num_frames if num_frames is not None else round(DURATION_S * fps)
    t = times(frames, fps)
    down = np.array([math.cos(angle), 0.0, -math.sin(angle)])
    normal = np.array([math.sin(angle), 0.0, math.cos(angle)])
    edge = np.zeros(3)
    on_slope = quat_from_axis_angle((0.0, 1.0, 0.0), angle)
    flat = np.array([1.0, 0.0, 0.0, 0.0])
    position = np.zeros((frames, 3))
    quaternion = np.zeros((frames, 4))
    accel = GRAVITY * (math.sin(angle) - mu * math.cos(angle))
    if accel <= 0:
        start = edge - START_DISTANCE * down + HALF[2] * normal
        position[:] = start
        quaternion[:] = on_slope
        return {"block": {"position": position, "quaternion": quaternion}}
    t_edge = math.sqrt(2 * START_DISTANCE / accel)
    v_edge = accel * t_edge
    x_floor = HALF[2] * math.sin(angle)
    stop = v_edge / (mu * GRAVITY) if mu > 0 else math.inf
    for i, time in enumerate(t):
        if time < t_edge:
            s = START_DISTANCE - 0.5 * accel * time * time
            position[i] = edge - s * down + HALF[2] * normal
            quaternion[i] = on_slope
        else:
            tau = min(time - t_edge, stop)
            x = x_floor + v_edge * tau - 0.5 * mu * GRAVITY * tau * tau
            position[i] = (x, 0.0, HALF[2])
            quaternion[i] = flat
    return {"block": {"position": position, "quaternion": quaternion}}


def static_scene(setup: dict[str, Any]) -> list[render.Primitive]:
    angle = float(scene_value(setup, "incline_angle", SETUP["incline_angle"]))
    down = np.array([math.cos(angle), 0.0, -math.sin(angle)])
    normal = np.array([math.sin(angle), 0.0, math.cos(angle)])
    slab_center = -0.5 * INCLINE_LENGTH * down - 0.5 * INCLINE_THICKNESS * normal
    slab = render.Box(
        slab_center,
        quat_to_matrix(quat_from_axis_angle((0.0, 1.0, 0.0), angle)),
        np.array([INCLINE_LENGTH / 2, 0.25, INCLINE_THICKNESS / 2]),
        INCLINE_COLOR,
    )
    return [slab]


def dynamic_scene(tracks: dict[str, dict[str, np.ndarray]], frame: int) -> list[render.Primitive]:
    block = tracks["block"]
    rotation = quat_to_matrix(block["quaternion"][frame])
    return [render.Box(block["position"][frame], rotation, HALF, COLOR)]


def scene_objects(setup: dict[str, Any]) -> list[dict[str, Any]]:
    angle = float(scene_value(setup, "incline_angle", 0.5))
    down = np.array([math.cos(angle), 0.0, -math.sin(angle)])
    normal = np.array([math.sin(angle), 0.0, math.cos(angle)])
    start = -START_DISTANCE * down + HALF[2] * normal
    q = quat_from_axis_angle((0.0, 1.0, 0.0), angle)
    return [
        {
            "name": "block",
            "role": "dynamic",
            "shape": "box",
            "size": (2 * HALF).tolist(),
            "color": "blue",
            "initial_pose": [*start.tolist(), *q.tolist()],
        },
        {
            "name": "incline",
            "role": "static",
            "shape": f"plank {INCLINE_LENGTH} m long, lowest edge along y at x = 0, "
            "rising towards -x",
        },
        {"name": "floor", "role": "static", "shape": "plane z = 0"},
    ]
