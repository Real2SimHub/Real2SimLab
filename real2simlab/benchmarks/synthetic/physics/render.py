"""A small CPU ray caster for the synthetic scenes: spheres, boxes, capsules, a checkered floor.

NumPy only. Lambertian shading with one directional light, hard shadows, and 2x2
supersampling. Deterministic: the same scene always gives the same pixels.

A video is rendered with ``Renderer``: the static part of the scene is traced once, and each
frame re-traces only the rays that may see a moving object or its shadow (found with bounding
spheres). The result is identical to tracing every ray of every frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Union

import numpy as np

LIGHT = np.array([0.35, -0.55, 0.76]) / np.linalg.norm([0.35, -0.55, 0.76])
AMBIENT = 0.35
SKY_TOP = np.array([0.58, 0.70, 0.86])
SKY_BOTTOM = np.array([0.88, 0.91, 0.95])
FLOOR_A = np.array([0.80, 0.80, 0.78])
FLOOR_B = np.array([0.55, 0.56, 0.58])
TILE = 0.1
EPS = 1e-4


@dataclass(frozen=True)
class Sphere:
    center: np.ndarray
    radius: float
    color: tuple[float, float, float]

    def bound(self) -> tuple[np.ndarray, float]:
        return np.asarray(self.center, dtype=np.float64), float(self.radius)


@dataclass(frozen=True)
class Box:
    center: np.ndarray
    rotation: np.ndarray
    """3x3, box-local to world."""
    half: np.ndarray
    color: tuple[float, float, float]

    def bound(self) -> tuple[np.ndarray, float]:
        return np.asarray(self.center, dtype=np.float64), float(np.linalg.norm(self.half))


@dataclass(frozen=True)
class Capsule:
    start: np.ndarray
    end: np.ndarray
    radius: float
    color: tuple[float, float, float]

    def bound(self) -> tuple[np.ndarray, float]:
        start = np.asarray(self.start, dtype=np.float64)
        end = np.asarray(self.end, dtype=np.float64)
        return (start + end) / 2, float(np.linalg.norm(end - start) / 2 + self.radius)


Primitive = Union[Sphere, Box, Capsule]  # noqa: UP007 - runs on the twin's Python too


def _rays(camera: dict[str, Any], scale: int) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(camera["intrinsics"], dtype=np.float64)
    pose = np.asarray(camera["extrinsics"], dtype=np.float64)
    if pose.ndim == 3:
        pose = pose[0]
    width, height = camera["width"] * scale, camera["height"] * scale
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    x = (u / scale - k[0, 2]) / k[0, 0]
    y = (v / scale - k[1, 2]) / k[1, 1]
    local = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
    directions = np.einsum("ij,nj->ni", pose[:3, :3], local)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origin = np.broadcast_to(pose[:3, 3], directions.shape).copy()
    return origin, directions


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def _hits_bound(o: np.ndarray, d: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Whether each ray (unit direction) passes within ``radius`` of ``center``, ahead of it."""
    oc = center - o
    along = _dot(oc, d)
    squared = _dot(oc, oc)
    inside = squared <= radius * radius
    return inside | ((along > 0) & (squared - along * along <= radius * radius))


def _hit_sphere(o: np.ndarray, d: np.ndarray, s: Sphere) -> tuple[np.ndarray, np.ndarray]:
    oc = o - s.center
    b = _dot(oc, d)
    c = _dot(oc, oc) - s.radius**2
    disc = b * b - c
    ok = disc > 0
    root = np.sqrt(np.where(ok, disc, 0.0))
    t1, t2 = -b - root, -b + root
    t = np.where(ok & (t1 > 1e-6), t1, np.where(ok & (t2 > 1e-6), t2, np.inf))
    normal = (o + np.where(np.isfinite(t), t, 0.0)[:, None] * d - s.center) / s.radius
    return t, normal


def _hit_box(o: np.ndarray, d: np.ndarray, b: Box) -> tuple[np.ndarray, np.ndarray]:
    r = b.rotation
    lo = (o - b.center) @ r
    ld = d @ r
    safe = np.where(np.abs(ld) < 1e-12, 1e-12, ld)
    t1 = (-b.half - lo) / safe
    t2 = (b.half - lo) / safe
    tmin, tmax = np.minimum(t1, t2), np.maximum(t1, t2)
    near, far = tmin.max(axis=1), tmax.min(axis=1)
    hit = (far >= near) & (far > 1e-6)
    t = np.where(hit, np.where(near > 1e-6, near, far), np.inf)
    axis = tmin.argmax(axis=1)
    local_normal = np.zeros_like(lo)
    rows = np.arange(len(o))
    local_normal[rows, axis] = -np.sign(ld[rows, axis])
    return t, local_normal @ r.T


def _hit_capsule(o: np.ndarray, d: np.ndarray, c: Capsule) -> tuple[np.ndarray, np.ndarray]:
    axis = c.end - c.start
    length = float(np.linalg.norm(axis))
    a = axis / length
    oc = o - c.start
    da, oa = d @ a, oc @ a
    dd = d - da[:, None] * a
    od = oc - oa[:, None] * a
    qa = _dot(dd, dd)
    qb = 2 * _dot(dd, od)
    qc = _dot(od, od) - c.radius**2
    disc = qb * qb - 4 * qa * qc
    ok = (disc > 0) & (qa > 1e-12)
    root = np.sqrt(np.where(ok, disc, 0.0))
    tc = (-qb - root) / np.where(qa > 1e-12, 2 * qa, 1.0)
    along = oa + tc * da
    side = ok & (tc > 1e-6) & (along >= 0) & (along <= length)
    t = np.where(side, tc, np.inf)
    point = o + np.where(np.isfinite(t), t, 0.0)[:, None] * d
    proj = c.start + np.clip((point - c.start) @ a, 0, length)[:, None] * a
    normal = point - proj
    for center in (c.start, c.end):
        ts, ns = _hit_sphere(o, d, Sphere(center, c.radius, c.color))
        closer = ts < t
        t = np.where(closer, ts, t)
        normal = np.where(closer[:, None], ns * c.radius, normal)
    norm = np.linalg.norm(normal, axis=1, keepdims=True)
    return t, normal / np.where(norm > 0, norm, 1.0)


def _hit(o: np.ndarray, d: np.ndarray, primitive: Primitive) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(primitive, Sphere):
        return _hit_sphere(o, d, primitive)
    if isinstance(primitive, Box):
        return _hit_box(o, d, primitive)
    return _hit_capsule(o, d, primitive)


def _intersect(
    o: np.ndarray, d: np.ndarray, primitives: Sequence[Primitive], floor: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.full(len(o), np.inf)
    normal = np.zeros_like(o)
    color = np.zeros_like(o)
    if floor:
        down = d[:, 2] < -1e-9
        tf = np.where(down, -o[:, 2] / np.where(down, d[:, 2], -1.0), np.inf)
        tf = np.where(tf > 1e-6, tf, np.inf)
        point = o + np.where(np.isfinite(tf), tf, 0.0)[:, None] * d
        checker = (np.floor(point[:, 0] / TILE) + np.floor(point[:, 1] / TILE)) % 2 == 0
        hit = tf < t
        t = np.where(hit, tf, t)
        normal[hit] = (0.0, 0.0, 1.0)
        color[hit] = np.where(checker[hit][:, None], FLOOR_A, FLOOR_B)
    for primitive in primitives:
        tp, np_ = _hit(o, d, primitive)
        hit = tp < t
        t = np.where(hit, tp, t)
        normal[hit] = np_[hit]
        color[hit] = primitive.color
    return t, normal, color


def _shade(
    o: np.ndarray, d: np.ndarray, primitives: Sequence[Primitive], floor: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shaded colours, shadow-ray origins and the hit mask of rays against a scene."""
    t, normal, color = _intersect(o, d, primitives, floor)
    hit = np.isfinite(t)
    point = o + np.where(hit, t, 0.0)[:, None] * d + normal * EPS
    light = np.broadcast_to(LIGHT, point.shape)
    blocked = np.zeros(len(o), dtype=bool)
    for primitive in primitives:
        ts, _ = _hit(point, light, primitive)
        blocked |= np.isfinite(ts)
    diffuse = np.clip(normal @ LIGHT, 0.0, 1.0) * ~blocked
    shaded = color * (AMBIENT + (1 - AMBIENT) * diffuse)[:, None]
    blend = np.clip(0.5 - d[:, 2], 0.0, 1.0)[:, None]
    sky = SKY_BOTTOM * (1 - blend) + SKY_TOP * blend
    return np.where(hit[:, None], shaded, sky), point, hit


class Renderer:
    """Renders frames of one camera over a fixed static scene."""

    def __init__(
        self,
        camera: dict[str, Any],
        static: Sequence[Primitive] = (),
        *,
        floor: bool = True,
        scale: int = 2,
    ) -> None:
        self.camera = camera
        self.static = list(static)
        self.floor = floor
        self.scale = scale
        self.origin, self.direction = _rays(camera, scale)
        self.background, self.points, self.hit = _shade(
            self.origin, self.direction, self.static, floor
        )

    def frame(self, dynamic: Sequence[Primitive]) -> np.ndarray:
        """One RGB uint8 frame with the ``dynamic`` primitives added to the static scene."""
        mask = np.zeros(len(self.origin), dtype=bool)
        light = np.broadcast_to(LIGHT, self.points.shape)
        for primitive in dynamic:
            center, radius = primitive.bound()
            mask |= _hits_bound(self.origin, self.direction, center, radius * 1.001)
            mask |= self.hit & _hits_bound(self.points, light, center, radius * 1.001)
        image = self.background.copy()
        if mask.any():
            shaded, _, _ = _shade(
                self.origin[mask], self.direction[mask], [*self.static, *dynamic], self.floor
            )
            image[mask] = shaded
        height, width = self.camera["height"], self.camera["width"]
        image = image.reshape(height, self.scale, width, self.scale, 3).mean(axis=(1, 3))
        return (np.clip(image, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)


def render(primitives: Sequence[Primitive], camera: dict[str, Any], scale: int = 2) -> np.ndarray:
    """One frame of ``primitives``, every ray traced."""
    return Renderer(camera, primitives, scale=scale).frame(())


def render_frames(
    camera: dict[str, Any],
    static: Sequence[Primitive],
    dynamic_frames: Sequence[Sequence[Primitive]],
    scale: int = 2,
) -> np.ndarray:
    """(F, H, W, 3) uint8 frames: one per list of dynamic primitives."""
    renderer = Renderer(camera, static, scale=scale)
    return np.stack([renderer.frame(dynamic) for dynamic in dynamic_frames])
