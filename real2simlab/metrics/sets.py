"""Trajectory errors that do not match objects by name: Chamfer distances between centroid sets.

A method that reconstructs a scene from video names its objects as it likes, and may split or
merge them. At every reference frame, these metrics take the set of predicted object centroids
and the set of reference centroids and average the symmetric Chamfer distance (the mean
distance from each point to the nearest point of the other set, both ways). 3D distances can
first be aligned by one similarity transform over all frames, fitted on the nearest-neighbour
pairs; this is what a method that sees only video can be expected to get right.

A frame where the reference has objects and the prediction has none cannot be measured; like
the named-object trajectory metrics, the metric is then an error (scored as a failure), so
leaving out the frames that are hard to predict never lowers the error. Frames where the
reference has no objects carry no information and are skipped.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from real2simlab.formats.tracks import Camera, Tracks
from real2simlab.metrics.base import (
    CANNOT_ALIGN,
    Metric,
    MetricValue,
    centred,
    downscale_exponent,
    fit_similarity,
    one_of,
    option_problems,
    robust_mean,
    robust_norm,
    window_camera,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class _Sets:
    predicted: FloatArray
    """(F, Np, 3) predicted centroids at the reference's frame times."""
    expected: FloatArray
    """(F, Nr, 3) reference centroids."""
    predicted_camera: Camera | None
    """The prediction's camera at the same times (frame ``k`` is the pose at that instant)."""
    expected_camera: Camera | None


def _frames(prediction: Tracks, reference: Tracks, frames: tuple[int, int] | None) -> _Sets | str:
    """The predicted and reference centroids, and their cameras, over the reference's frames."""
    total = reference.num_frames
    start, end = frames or (0, total)
    if not reference.objects:
        return "the reference has no objects"
    if not prediction.objects or prediction.num_frames == 0:
        return "the prediction has no objects"
    if not 0 <= start < end <= total:
        return f"frames [{start}, {end}) are outside the reference's {total} frames"
    times = reference.times()[start:end]
    if prediction.duration + 0.5 / prediction.fps < times[-1] - 1e-9:
        return (
            f"the prediction covers {prediction.duration:.3f} s, "
            f"the reference needs {times[-1]:.3f} s"
        )
    sampled = prediction.at_times(times)
    predicted = np.stack([sampled.positions(name) for name in sampled.objects], axis=1)
    expected = np.stack(
        [reference.positions(name)[start:end] for name in reference.objects], axis=1
    )
    return _Sets(
        predicted=predicted,
        expected=expected,
        predicted_camera=sampled.camera,
        expected_camera=window_camera(reference.camera, start, end),
    )


def _has_points(points: FloatArray) -> bool:
    return bool(np.isfinite(points).all(axis=1).any())


def _mean_over_frames(pairs: list[tuple[FloatArray, FloatArray]], what: str) -> MetricValue:
    """The mean Chamfer distance over frames where the reference has points; an error if the
    prediction has none in any of them. ``pairs`` holds (predicted, reference) point sets."""
    values: list[float] = []
    uncovered = 0
    for predicted, expected in pairs:
        if not _has_points(expected):
            continue
        value = _chamfer(predicted, expected)
        if value is None:
            uncovered += 1
        else:
            values.append(value)
    if uncovered:
        frames = uncovered + len(values)
        return MetricValue.error(
            f"no predicted {what} in {uncovered} of the {frames} frames where the reference "
            f"has objects (coverage {len(values) / frames:.0%})"
        )
    if not values:
        return MetricValue.error("the reference has no objects in the window")
    return MetricValue.computed(robust_mean(np.asarray(values)))


def _chamfer(a: FloatArray, b: FloatArray) -> float | None:
    a = a[np.isfinite(a).all(axis=1)]
    b = b[np.isfinite(b).all(axis=1)]
    if len(a) == 0 or len(b) == 0:
        return None
    with np.errstate(over="ignore"):  # a difference above the largest float: not finite
        distances = robust_norm(a[:, None, :] - b[None, :, :], axis=-1)
    # Halves first: two finite means can have a sum above the largest float.
    return robust_mean(distances.min(axis=1)) / 2 + robust_mean(distances.min(axis=0)) / 2


def _align(predicted: FloatArray, expected: FloatArray) -> FloatArray | str:
    """``predicted`` mapped by the similarity that best fits nearest-reference pairs, refined;
    or why it cannot be aligned (a metric error: an alignment that fails never leaves the
    prediction unaligned). With no frame where both have points there is nothing to align,
    and ``predicted`` comes back as it is: the comparison reports those frames.

    Starts from matching the overall centroid and spread of the two point sets, so the first
    nearest-neighbour pairs are already roughly right.
    """
    # Typed: numpy's stubs give a reduction over one axis as a scalar or an array.
    present: NDArray[np.bool_] = np.asarray(np.isfinite(predicted).all(axis=2))  # (F, Np)
    finite_e: NDArray[np.bool_] = np.asarray(np.isfinite(expected).all(axis=2))
    measured: NDArray[np.bool_] = np.asarray(finite_e.any(axis=1))  # frames compared
    if not (present.any(axis=1) & measured).any():
        return predicted
    if downscale_exponent(predicted) == 0 and downscale_exponent(expected) == 0:
        moved = _refined(predicted, expected)  # as it always was
    else:
        # Larger positions (their sums or squares may overflow, a diverging solver): the same
        # alignment of the sets about their centroids, each divided by its own power of two
        # (it is equivariant under both, a translation alone aside), then moved back into the
        # reference's frame.
        _, about_p, shrink_p = centred(predicted)
        mu_e, about_e, shrink_e = centred(expected)
        moved = _refined(about_p, about_e, relative=shrink_p - shrink_e)
        if not isinstance(moved, str):
            with np.errstate(over="ignore", invalid="ignore"):  # beyond the float range: below
                moved = np.ldexp(moved, 1 + shrink_e) + mu_e
    if isinstance(moved, str):
        return moved
    # A point moved beyond the float range would silently leave its frame's set.
    finite_moved: NDArray[np.bool_] = np.asarray(np.isfinite(moved).all(axis=2))
    if (present & ~finite_moved)[measured].any():
        return f"{CANNOT_ALIGN}: aligned, some of its positions are beyond the floating-point range"
    return moved


def _refined(predicted: FloatArray, expected: FloatArray, relative: int = 0) -> FloatArray | str:
    """The alignment of :func:`_align`, where some frame has points of both and every
    coordinate is below ``2**SAFE_EXPONENT``. ``predicted`` is ``2**relative`` times smaller
    than in the units of ``expected`` (they were divided by different powers of two)."""
    flat_p = predicted.reshape(-1, 3)
    flat_e = expected.reshape(-1, 3)
    flat_p = flat_p[np.isfinite(flat_p).all(axis=1)]
    flat_e = flat_e[np.isfinite(flat_e).all(axis=1)]
    mu_p, mu_e = flat_p.mean(axis=0), flat_e.mean(axis=0)
    spread_p = float(np.sqrt(((flat_p - mu_p) ** 2).sum(axis=1).mean()))
    spread_e = float(np.sqrt(((flat_e - mu_e) ** 2).sum(axis=1).mean()))
    ratio = spread_e / spread_p if spread_p > 0 else 1.0
    current = (predicted - mu_p) * ratio + mu_e
    for _ in range(5):
        sources: list[FloatArray] = []
        targets: list[FloatArray] = []
        for frame in range(len(current)):
            # The same points on both sides, so that each keeps its own pair.
            kept = np.isfinite(current[frame]).all(axis=1) & np.isfinite(predicted[frame]).all(
                axis=1
            )
            p, raw = current[frame][kept], predicted[frame][kept]
            e = expected[frame][np.isfinite(expected[frame]).all(axis=1)]
            if len(p) == 0 or len(e) == 0:
                continue
            offsets: FloatArray = p[:, None, :] - e[None, :, :]
            nearest = (offsets * offsets).sum(axis=-1).argmin(axis=1)  # nearest by squared distance
            sources.append(raw)
            targets.append(e[nearest])
        if not sources:  # the previous fit moved every point out of the float range
            return f"{CANNOT_ALIGN}: aligned, its positions are beyond the floating-point range"
        source, target = np.concatenate(sources), np.concatenate(targets)
        if len(source) < 3:
            if relative:  # a translation keeps the prediction's own size
                with np.errstate(over="ignore", invalid="ignore"):  # beyond: refused by _align
                    moved = np.ldexp(predicted - source.mean(axis=0), relative)
                    return moved + target.mean(axis=0)
            shift = target.mean(axis=0) - source.mean(axis=0)
            return predicted + shift
        mapping = fit_similarity(source, target)
        if isinstance(mapping, str):
            return mapping
        with np.errstate(over="ignore", invalid="ignore"):  # out of the float range: refused
            current = mapping(predicted)
    return current


class CentroidSetChamfer(Metric):
    name: ClassVar[str] = "centroid_set_chamfer"
    unit: ClassVar[str] = "m"
    description: ClassVar[str] = (
        "Mean over reference frames of the symmetric Chamfer distance between the sets of "
        "predicted and reference object centroids (objects are not matched by name)."
    )

    def option_problems(self, options: Mapping[str, Any]) -> list[str]:
        return option_problems(self.name, options, {"align": one_of(("none", "similarity"))})

    def compute(
        self,
        prediction: Tracks,
        reference: Tracks,
        *,
        frames: tuple[int, int] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> MetricValue:
        sets = _frames(prediction, reference, frames)
        if isinstance(sets, str):
            return MetricValue.error(sets)
        predicted, expected = sets.predicted, sets.expected
        mode = str((options or {}).get("align", "none"))
        if mode not in ("none", "similarity"):
            return MetricValue.error(f"align must be none or similarity, got {mode!r}")
        if mode == "similarity":
            moved = _align(predicted, expected)
            if isinstance(moved, str):
                return MetricValue.error(moved)
            predicted = moved
        return _mean_over_frames(list(zip(predicted, expected, strict=True)), "object")


class CentroidSetChamfer2d(Metric):
    name: ClassVar[str] = "centroid_set_chamfer_2d"
    unit: ClassVar[str] = "1"
    description: ClassVar[str] = (
        "centroid_set_chamfer in the image: centroids projected with each world's own camera, "
        "in reference pixels, divided by the reference image diagonal."
    )

    def option_problems(self, options: Mapping[str, Any]) -> list[str]:
        return option_problems(self.name, options, {})

    def compute(
        self,
        prediction: Tracks,
        reference: Tracks,
        *,
        frames: tuple[int, int] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> MetricValue:
        if reference.camera is None:
            return MetricValue.not_applicable("the reference has no camera")
        if prediction.camera is None:
            return MetricValue.error("the prediction has no camera to project with")
        sets = _frames(prediction, reference, frames)
        if isinstance(sets, str):
            return MetricValue.error(sets)
        # Both cameras over the reference's frames, so each predicted centroid is projected
        # with the predicted camera of the same instant.
        ref_cam, pred_cam = sets.expected_camera, sets.predicted_camera
        assert ref_cam is not None and pred_cam is not None
        scale = np.array([ref_cam.width / pred_cam.width, ref_cam.height / pred_cam.height])
        pairs: list[tuple[FloatArray, FloatArray]] = []
        for index in range(len(sets.expected)):
            ref_uv = _project(ref_cam, sets.expected[index], index)
            pred_uv = _project(pred_cam, sets.predicted[index], index) * scale
            pairs.append(
                (np.c_[pred_uv, np.zeros(len(pred_uv))], np.c_[ref_uv, np.zeros(len(ref_uv))])
            )
        value = _mean_over_frames(pairs, "object in front of the camera")
        if value.value is None:
            return value
        return MetricValue.computed(value.value / ref_cam.diagonal)


def _project(camera: Camera, points: FloatArray, frame: int) -> FloatArray:
    pose: FloatArray = camera.pose(frame)
    count = len(points)
    homogeneous: FloatArray = np.c_[points, np.ones(count)]
    local: FloatArray = np.einsum("ij,nj->ni", np.linalg.inv(pose), homogeneous)[:, :3]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pixels: FloatArray = np.einsum("ij,nj->ni", camera.intrinsics, local)
        uv: FloatArray = pixels[:, :2] / pixels[:, 2:3]
    uv[~(local[:, 2] > 1e-9)] = np.nan
    return uv
