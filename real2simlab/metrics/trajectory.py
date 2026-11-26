"""Trajectory errors in 3D (metres) and in the image (pixels, and normalised by the diagonal)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from real2simlab.formats.tracks import Tracks
from real2simlab.metrics.base import (
    Aligned,
    Metric,
    MetricValue,
    align,
    downscale_exponent,
    fit_similarity,
    one_of,
    option_problems,
    robust_centroid,
    robust_mean,
    robust_norm,
    robust_rms,
)

ALIGN_MODES = ("none", "translation", "similarity")
MISSING_FRAMES = "the prediction is missing frames where the reference is visible"
TOO_FAR = "the prediction is too far from the reference to measure (a distance above 1.8e308)"


def _distances(aligned: Aligned, mode: str) -> NDArray[np.float64] | str:
    predicted = aligned.predicted
    if mode != "none":
        both = [(predicted[n], aligned.expected[n]) for n in aligned.expected]
        source = np.concatenate([p for p, _ in both])
        target = np.concatenate([e for _, e in both])
        valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
        if valid.sum() < 3:
            return "fewer than 3 frames to align"
        if mode == "translation":
            predicted = _translated(predicted, source[valid], target[valid])
        else:
            mapping = fit_similarity(source[valid], target[valid])
            if isinstance(mapping, str):
                return mapping
            # Frames outside the fit (the reference not visible there) may leave the float
            # range; no metric uses them.
            with np.errstate(over="ignore", invalid="ignore"):
                predicted = {n: mapping(p) for n, p in predicted.items()}
    gaps: list[NDArray[np.float64]] = []
    for name in aligned.expected:
        with np.errstate(over="ignore"):  # a difference above the largest float: not finite
            difference = predicted[name] - aligned.expected[name]
        gaps.append(robust_norm(difference, axis=1))
    return np.stack(gaps, axis=1)  # (frames, objects)


def _translated(
    predicted: dict[str, NDArray[np.float64]],
    source: NDArray[np.float64],
    target: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """``predicted`` moved so that ``source`` has the centroid of ``target``."""
    if downscale_exponent(source) == 0 and downscale_exponent(target) == 0:
        shift = target.mean(axis=0) - source.mean(axis=0)
        return {n: p + shift for n, p in predicted.items()}
    # Larger positions: their sums may overflow, and through a shift a far but compact
    # prediction would lose its shape to rounding. Each is moved about the centroids, as
    # (p - mu_s) + mu_t, in halves; beyond the float range it is not finite (too far).
    mu_s, mu_t = robust_centroid(source), robust_centroid(target)
    with np.errstate(over="ignore", invalid="ignore"):
        return {
            n: np.ldexp(np.ldexp(p, -1) - np.ldexp(mu_s, -1), 1) + mu_t
            for n, p in predicted.items()
        }


def _unmeasured(
    aligned: Aligned, distances: NDArray[np.float64], used: NDArray[np.bool_], missing: str
) -> MetricValue | None:
    """The error when a distance the metric uses is not finite: the prediction is missing
    there (``missing``), or its finite positions are too far from the reference to measure."""
    bad = used & ~np.isfinite(distances)
    if not bad.any():
        return None
    present = np.stack(
        [np.isfinite(aligned.predicted[name]).all(axis=1) for name in aligned.expected], axis=1
    )
    return MetricValue.error(TOO_FAR if present[bad].all() else missing)


class _Trajectory3D(Metric):
    formats: ClassVar[frozenset[str]] = frozenset({"tracks"})
    unit: ClassVar[str] = "m"

    def option_problems(self, options: Mapping[str, Any]) -> list[str]:
        return option_problems(self.name, options, {"align": one_of(ALIGN_MODES)})

    def compute(
        self,
        prediction: Tracks,
        reference: Tracks,
        *,
        frames: tuple[int, int] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> MetricValue:
        mode = str((options or {}).get("align", "none"))
        if mode not in ALIGN_MODES:
            return MetricValue.error(f"align must be one of {ALIGN_MODES}, got {mode!r}")
        aligned = align(prediction, reference, frames)
        if isinstance(aligned, str):
            return MetricValue.error(aligned)
        distances = _distances(aligned, mode)
        if isinstance(distances, str):
            return MetricValue.error(distances)
        return self.reduce(distances, aligned)

    def reduce(self, distances: NDArray[np.float64], aligned: Aligned) -> MetricValue:
        raise NotImplementedError


def _reference_visible(aligned: Aligned) -> NDArray[np.bool_]:
    return np.stack(
        [np.isfinite(aligned.expected[name]).all(axis=1) for name in aligned.expected], axis=1
    )


class Traj3dMean(_Trajectory3D):
    name: ClassVar[str] = "traj3d_mean"
    description: ClassVar[str] = "Mean 3D distance between predicted and reference centroids."

    def reduce(self, distances: NDArray[np.float64], aligned: Aligned) -> MetricValue:
        visible = _reference_visible(aligned)
        if not visible.any():
            return MetricValue.error("the reference has no visible frames in the window")
        problem = _unmeasured(aligned, distances, visible, MISSING_FRAMES)
        if problem is not None:
            return problem
        return MetricValue.computed(robust_mean(distances[visible]))


class Traj3dRmse(_Trajectory3D):
    name: ClassVar[str] = "traj3d_rmse"
    description: ClassVar[str] = "Root-mean-square 3D distance between centroids."

    def reduce(self, distances: NDArray[np.float64], aligned: Aligned) -> MetricValue:
        visible = _reference_visible(aligned)
        if not visible.any():
            return MetricValue.error("the reference has no visible frames in the window")
        problem = _unmeasured(aligned, distances, visible, MISSING_FRAMES)
        if problem is not None:
            return problem
        return MetricValue.computed(robust_rms(distances[visible]))


class Traj3dFinal(_Trajectory3D):
    name: ClassVar[str] = "traj3d_final"
    description: ClassVar[str] = (
        "3D centroid distance at the last reference-visible frame (endpoint error)."
    )

    def reduce(self, distances: NDArray[np.float64], aligned: Aligned) -> MetricValue:
        visible = _reference_visible(aligned)
        finals: list[float] = []
        for column, name in enumerate(aligned.expected):
            rows = np.flatnonzero(visible[:, column])
            if rows.size == 0:
                continue
            value = distances[rows[-1], column]
            if not np.isfinite(value):
                present = bool(np.isfinite(aligned.predicted[name][rows[-1]]).all())
                return MetricValue.error(
                    TOO_FAR if present else "the prediction is missing the final frame"
                )
            finals.append(float(value))
        if not finals:
            return MetricValue.error("the reference has no visible frames in the window")
        return MetricValue.computed(robust_mean(np.asarray(finals)))


class _Trajectory2D(Metric):
    formats: ClassVar[frozenset[str]] = frozenset({"tracks"})
    normalised: ClassVar[bool] = False

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
        aligned = align(prediction, reference, frames)
        if isinstance(aligned, str):
            return MetricValue.error(aligned)
        # Both cameras over the reference's frames: a moving predicted camera is taken at the
        # same instants as the predicted positions, whatever its own frame rate.
        ref_cam, pred_cam = aligned.expected_camera, aligned.predicted_camera
        assert ref_cam is not None and pred_cam is not None
        scale = np.array([ref_cam.width / pred_cam.width, ref_cam.height / pred_cam.height])
        errors: list[NDArray[np.float64]] = []
        for name, expected in aligned.expected.items():
            ref_uv = ref_cam.project(expected)
            pred_uv = pred_cam.project(aligned.predicted[name]) * scale
            visible = np.isfinite(ref_uv).all(axis=1)
            with np.errstate(over="ignore"):
                gap = robust_norm(pred_uv[visible] - ref_uv[visible], axis=1)
            if not np.isfinite(gap).all():
                return MetricValue.error(
                    f"{name}: the prediction cannot be projected in some frames"
                )
            errors.append(gap)
        joined = np.concatenate(errors) if errors else np.array([])
        if joined.size == 0:
            return MetricValue.error("no reference point projects into the image")
        value = robust_mean(joined)
        return MetricValue.computed(value / ref_cam.diagonal if self.normalised else value)


class Traj2dPxMean(_Trajectory2D):
    name: ClassVar[str] = "traj2d_px_mean"
    unit: ClassVar[str] = "px"
    description: ClassVar[str] = (
        "Mean image distance between projected centroids, each projected with its own camera, "
        "in reference pixels."
    )


class Traj2dNormMean(_Trajectory2D):
    name: ClassVar[str] = "traj2d_norm_mean"
    unit: ClassVar[str] = "1"
    normalised: ClassVar[bool] = True
    description: ClassVar[str] = "traj2d_px_mean divided by the reference image diagonal."
