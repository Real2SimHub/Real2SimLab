"""Training samples of the synthetic suite: the same phenomena with sampled parameters.

Each phenomenon's evaluation task uses one fixed set of hidden parameters and one initial
state. A training sample draws them uniformly from ranges around those values (``SPACES``) and
is kept only if

* its objects stay on screen at least as well as in the evaluation task (projected with the
  same camera, they never leave the frame by more than the evaluation task's objects do), and
* it is not a near copy of the evaluation task: at least one sampled value differs from the
  evaluation task's by more than ``MARGIN`` (relative).

Sampling is deterministic: sample ``i`` of a phenomenon and seed is the ``i``-th accepted draw,
whatever the count asked for. When the tasks are built, a sample whose video the overlap check
cannot tell from the evaluation task's (:mod:`real2simlab.task.overlap`) is skipped as well, so
a training set never holds a task that ``r2s train export --against`` would leave out.
"""

from __future__ import annotations

import copy
import itertools
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

MARGIN = 0.05
"""A sample is a near copy of the evaluation task if every sampled value is within 5 %."""
MAX_DRAWS = 1000
"""Draws per accepted sample before giving up (the ranges would be wrong)."""


@dataclass(frozen=True)
class Space:
    """What is sampled for one phenomenon: hidden parameters and the initial state.

    Keys of ``setup`` are paths into the phenomenon's ``SETUP``: ``release_height.ball`` or
    ``initial_velocity.ball.0`` (a list index)."""

    params: dict[str, tuple[float, float]]
    setup: dict[str, tuple[float, float]]

    def keys(self) -> list[str]:
        return [*(f"params.{k}" for k in self.params), *(f"setup.{k}" for k in self.setup)]


SPACES: dict[str, Space] = {
    "bouncing_ball": Space(
        params={"restitution": (0.45, 0.92)},
        setup={"release_height.ball": (0.3, 0.9)},
    ),
    "pendulum": Space(
        params={"damping": (0.0, 0.5)},
        setup={"initial_angle.bob": (0.15, 0.6)},
    ),
    "incline_slider": Space(
        params={},
        setup={"incline_angle": (0.42, 0.6), "surface_friction.block": (0.18, 0.42)},
    ),
    "projectile": Space(
        params={
            "drag": (0.05, 0.2),
            "restitution": (0.45, 0.75),
            "tangential": (0.7, 0.95),
            "rolling": (0.25, 0.55),
        },
        setup={"initial_velocity.ball.0": (1.6, 2.3), "initial_velocity.ball.2": (2.1, 2.8)},
    ),
    "two_ball_collision": Space(
        params={"restitution": (0.6, 1.0), "rolling": (0.15, 0.4)},
        setup={"initial_velocity.ball_a.0": (0.95, 1.35)},
    ),
}


@dataclass(frozen=True)
class Sample:
    phenomenon: str
    seed: int
    index: int
    params: dict[str, Any]
    setup: dict[str, Any]

    @property
    def name(self) -> str:
        """``s<seed>-<index>``: the sample's segment in task ids."""
        return f"s{self.seed}-{self.index:05d}"


def _get(data: Any, path: str) -> Any:
    for part in path.split("."):
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def _set(data: Any, path: str, value: float) -> None:
    *head, last = path.split(".")
    for part in head:
        data = data[int(part)] if isinstance(data, list) else data[part]
    if isinstance(data, list):
        data[int(last)] = value
    else:
        data[last] = value


def values(module: ModuleType, params: dict[str, Any], setup: dict[str, Any]) -> dict[str, float]:
    """The sampled quantities of a phenomenon (``params.<name>``, ``setup.<path>``)."""
    space = SPACES[module.NAME]
    out = {f"params.{k}": float(params[k]) for k in space.params}
    out.update({f"setup.{k}": float(_get(setup, k)) for k in space.setup})
    return out


def _on_screen(module: ModuleType, tracks: dict[str, dict[str, np.ndarray]]) -> tuple[float, ...]:
    """The extent (u min, u max, v min, v max) of every object's projected centroid."""
    camera = module.CAMERA
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    pose = np.asarray(camera["extrinsics"], dtype=np.float64)
    points = np.concatenate([np.asarray(d["position"]) for d in tracks.values()])
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    uv = local @ intrinsics.T
    uv = uv[:, :2] / uv[:, 2:3]
    return (float(uv[:, 0].min()), float(uv[:, 0].max()), float(uv[:, 1].min()),
            float(uv[:, 1].max()))  # fmt: skip


def _allowed_extent(module: ModuleType) -> tuple[float, ...]:
    """The frame, widened to wherever the evaluation task's objects go."""
    width, height = float(module.CAMERA["width"]), float(module.CAMERA["height"])
    u0, u1, v0, v1 = _on_screen(module, module.simulate(module.SETUP, module.TRUE_PARAMS))
    return min(0.0, u0), max(width, u1), min(0.0, v0), max(height, v1)


def _accepted(module: ModuleType, params: dict[str, Any], setup: dict[str, Any]) -> bool:
    reference = values(module, module.TRUE_PARAMS, module.SETUP)
    drawn = values(module, params, setup)
    if all(abs(drawn[k] - v) <= MARGIN * abs(v) for k, v in reference.items()):
        return False  # a near copy of the evaluation task
    slides = module.NAME != "incline_slider" or (  # the block must slide: friction < tan(angle)
        _get(setup, "surface_friction.block") <= np.tan(setup["incline_angle"]) - 0.08
    )
    if not slides:
        return False
    u0, u1, v0, v1 = _on_screen(module, module.simulate(setup, params))
    a0, a1, b0, b1 = _allowed_extent(module)
    return u0 >= a0 and u1 <= a1 and v0 >= b0 and v1 <= b1


def iter_samples(module: ModuleType, seed: int) -> Iterator[Sample]:
    """The training samples of ``module`` for ``seed``, in order, without end."""
    space = SPACES[module.NAME]
    entropy = [seed, zlib.crc32(module.NAME.encode())]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    index = 0
    while True:
        for _ in range(MAX_DRAWS):
            params = dict(module.TRUE_PARAMS)
            setup = copy.deepcopy(module.SETUP)
            for name, (low, high) in space.params.items():
                params[name] = round(float(rng.uniform(low, high)), 4)
            for path, (low, high) in space.setup.items():
                _set(setup, path, round(float(rng.uniform(low, high)), 4))
            if _accepted(module, params, setup):
                yield Sample(module.NAME, seed, index, params, setup)
                index += 1
                break
        else:
            raise RuntimeError(f"no acceptable {module.NAME} sample in {MAX_DRAWS} draws")


def samples(module: ModuleType, seed: int, count: int) -> list[Sample]:
    """The first ``count`` training samples of ``module`` for ``seed``."""
    return list(itertools.islice(iter_samples(module, seed), count))
