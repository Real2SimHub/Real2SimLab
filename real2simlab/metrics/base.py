"""Metrics compare a prediction with a reference; each declares the formats it supports."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from real2simlab import plugins
from real2simlab.formats.tracks import Camera, Tracks


class MetricStatus(StrEnum):
    COMPUTED = "computed"
    NOT_APPLICABLE = "not_applicable"
    """The metric does not apply, e.g. bounce timing when the reference never bounces."""
    ERROR = "error"
    """The prediction cannot be compared, e.g. an object is missing; scored as a failure."""


@dataclass(frozen=True)
class MetricValue:
    """One metric value, or why there is none."""

    status: MetricStatus
    value: float | None = None
    detail: str | None = None

    @classmethod
    def computed(cls, value: float) -> MetricValue:
        """A computed value; one that is not a finite number (an overflow) is an error, so
        that no record holds ``NaN`` or ``Infinity``."""
        number = float(value)
        if not math.isfinite(number):
            return cls.error(
                f"the comparison gives {number!r}, which is not a finite number (for example, "
                "what it compares is too large to measure)"
            )
        return cls(MetricStatus.COMPUTED, number)

    @classmethod
    def error(cls, detail: str) -> MetricValue:
        return cls(MetricStatus.ERROR, None, detail)

    @classmethod
    def not_applicable(cls, detail: str) -> MetricValue:
        return cls(MetricStatus.NOT_APPLICABLE, None, detail)

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"status": self.status.value}
        if self.value is not None:
            data["value"] = self.value
        if self.detail is not None:
            data["detail"] = self.detail
        return data


class Metric(ABC):
    """A comparison of a prediction with a reference. Lower values are better."""

    name: ClassVar[str]
    formats: ClassVar[frozenset[str]] = frozenset({"tracks"})
    """Output formats the metric reads. ``tracks`` also covers formats convertible to tracks."""
    unit: ClassVar[str]
    description: ClassVar[str]

    @abstractmethod
    def compute(
        self,
        prediction: Tracks,
        reference: Tracks,
        *,
        frames: tuple[int, int] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> MetricValue:
        """Compare over reference frames ``[start, end)`` (default: all)."""

    def option_problems(self, options: Mapping[str, Any]) -> list[str]:
        """What is wrong with ``options`` (the task's ``scoring.metric_options`` for this
        metric): a name the metric does not take, or a value it cannot use. Empty when they
        are usable. Task validation reports them, so a misspelt option is a broken task, not
        a score of 0. This default accepts anything; a metric that reads options overrides it
        (see :func:`option_problems`)."""
        return []


def option_problems(
    metric: str, options: Mapping[str, Any], takes: Mapping[str, Callable[[Any], str | None]]
) -> list[str]:
    """Problems of ``options`` for ``metric``, which takes the options named in ``takes``,
    each checked by its function (a problem, or ``None`` for a usable value)."""
    problems = [
        f"metric {metric!r} takes no option {name!r}; it takes: {sorted(takes) or 'none'}"
        for name in options
        if name not in takes
    ]
    for name, check in takes.items():
        if name in options:
            problem = check(options[name])
            if problem is not None:
                problems.append(f"metric {metric!r}: option {name} {problem}")
    return problems


def one_of(choices: Sequence[str]) -> Callable[[Any], str | None]:
    """A check that a value is one of ``choices``."""
    return lambda value: (
        None if value in choices else f"must be one of {list(choices)}, got {value!r}"
    )


def positive_number(value: Any) -> str | None:
    """``None`` for a finite number above 0, else what is wrong."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return f"must be a number, got {value!r}"
    if not (np.isfinite(value) and value > 0):
        return f"must be a finite number above 0, got {value!r}"
    return None


# Finite inputs whose squares or sums overflow (above about 1e154 and 1e308) still have a
# finite norm, mean or root mean square: these compute it, and give the plain result, bit for
# bit, whenever it does not overflow.


def robust_norm(vectors: NDArray[np.float64], axis: int = -1) -> NDArray[np.float64]:
    """``np.linalg.norm(vectors, axis=axis)``, finite wherever the norm is: where the squares
    overflow, the vector is first divided by its largest component. A norm above the largest
    float, or of a vector that is not finite, stays ``inf`` or ``nan``."""
    with np.errstate(over="ignore"):
        norm: NDArray[np.float64] = np.asarray(np.linalg.norm(vectors, axis=axis), np.float64)
    overflowed = np.isinf(norm) & np.isfinite(vectors).all(axis=axis)
    if overflowed.any():
        big = np.moveaxis(vectors, axis, -1)[overflowed]
        scale = np.abs(big).max(axis=-1, keepdims=True)
        with np.errstate(over="ignore"):
            norm[overflowed] = scale[:, 0] * np.linalg.norm(big / scale, axis=-1)
    return norm


def robust_mean(values: NDArray[np.float64]) -> float:
    """The mean of ``values`` (not empty), finite whenever they are: where their sum
    overflows (to infinity, or to NaN when sums of both signs overflow), they are first
    divided by their largest magnitude."""
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.mean(values))
    if not math.isfinite(mean) and bool(np.isfinite(values).all()):
        scale = float(np.abs(values).max())
        mean = scale * float(np.mean(values / scale))
    return mean


def robust_rms(values: NDArray[np.float64]) -> float:
    """The root mean square of ``values`` (not empty), finite whenever they are (it is at
    most their largest magnitude)."""
    with np.errstate(over="ignore"):
        rms = float(np.sqrt((values**2).mean()))
    if math.isinf(rms) and bool(np.isfinite(values).all()):
        scale = float(np.abs(values).max())
        rms = scale * float(np.sqrt(((values / scale) ** 2).mean()))
    return rms


# Alignment of positions too large for their sums and squares (a diverging solver): they are
# taken about their centroids and divided by powers of two (exact), which a fit is equivariant
# under. Positions below 2**SAFE_EXPONENT are aligned exactly as they always were.

SAFE_EXPONENT = 200
"""Coordinates below ``2**SAFE_EXPONENT`` (about 1.6e60) can be summed, squared and multiplied
in pairs without overflowing, for any number of points a comparison holds."""


def downscale_exponent(points: NDArray[np.float64]) -> int:
    """The power of two to divide ``points`` by so that every finite coordinate is below
    ``2**SAFE_EXPONENT``; 0 when they already are (dividing by a power of two is exact)."""
    finite = points[np.isfinite(points)]
    if finite.size == 0:
        return 0
    return max(0, math.frexp(float(np.abs(finite).max()))[1] - SAFE_EXPONENT)


def robust_centroid(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """The mean of the finite rows of ``points`` (..., D; at least one), which does not
    overflow: each axis is averaged divided by its largest magnitude (exact for an axis that
    does not vary)."""
    flat = points.reshape(-1, points.shape[-1])
    finite = flat[np.isfinite(flat).all(axis=1)]
    largest = np.abs(finite).max(axis=0)
    largest[largest == 0] = 1.0
    return largest * (finite / largest).mean(axis=0)


def centred(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """The :func:`robust_centroid` of ``points``, the points about it divided by
    ``2**(1 + shrink)`` (below ``2**SAFE_EXPONENT``, so that their sums and squares do not
    overflow), and ``shrink``. Halves are subtracted, so nothing overflows; scaling to its own
    spread, not to the positions' size, keeps a small spread far from the origin exact."""
    centroid = robust_centroid(points)
    half = np.ldexp(points, -1) - np.ldexp(centroid, -1)
    shrink = downscale_exponent(half)
    return centroid, np.ldexp(half, -shrink), shrink


def get_metric(name: str) -> Metric:
    """The registered metric called ``name``."""
    factory = plugins.load("metric", name)
    metric = factory()
    if not isinstance(metric, Metric):
        raise TypeError(f"metric plugin {name!r} does not create a Metric")
    return metric


def metric_names() -> list[str]:
    return plugins.names("metric")


@dataclass(frozen=True)
class Aligned:
    """Prediction and reference positions per object over the same reference times."""

    predicted: dict[str, NDArray[np.float64]]
    expected: dict[str, NDArray[np.float64]]
    times: NDArray[np.float64]
    predicted_camera: Camera | None = None
    """The prediction's camera at ``times``: frame ``k`` of a moving camera is its pose at
    ``times[k]``, whatever the prediction's own frame rate."""
    expected_camera: Camera | None = None
    """The reference's camera over the same frames."""


def align(prediction: Tracks, reference: Tracks, frames: tuple[int, int] | None) -> Aligned | str:
    """Sample the prediction at the reference's frame times; a string explains a failure."""
    if not reference.objects:
        return "the reference has no objects"
    total = reference.num_frames
    start, end = frames or (0, total)
    if not 0 <= start < end <= total:
        return f"frames [{start}, {end}) are outside the reference's {total} frames"
    missing = sorted(set(reference.objects) - set(prediction.objects))
    if missing:
        return f"objects missing from the prediction: {', '.join(missing)}"
    if prediction.num_frames == 0:
        return "the prediction has no frames, or objects with different frame counts"
    times = reference.times()[start:end]
    if prediction.duration + 0.5 / prediction.fps < times[-1] - 1e-9:
        return (
            f"the prediction covers {prediction.duration:.3f} s, the reference needs "
            f"{times[-1]:.3f} s"
        )
    same_clock = abs(prediction.fps - reference.fps) < 1e-9 and prediction.num_frames >= end
    if same_clock:
        predicted = {name: prediction.positions(name)[start:end] for name in reference.objects}
        predicted_camera = window_camera(prediction.camera, start, end)
    else:
        sampled = prediction.at_times(times)
        predicted = {name: sampled.positions(name) for name in reference.objects}
        predicted_camera = sampled.camera
    expected = {name: reference.positions(name)[start:end] for name in reference.objects}
    return Aligned(
        predicted=predicted,
        expected=expected,
        times=times,
        predicted_camera=predicted_camera,
        expected_camera=window_camera(reference.camera, start, end),
    )


def window_camera(camera: Camera | None, start: int, end: int) -> Camera | None:
    """``camera`` over frames ``[start, end)``: a moving camera's poses from ``start``."""
    if camera is None or camera.extrinsics.ndim == 2:
        return camera
    return replace(camera, extrinsics=camera.extrinsics[start:end])


PositionMap = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _umeyama(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> tuple[float | None, NDArray[np.float64]]:
    """Scale and rotation of the similarity that best maps the centred positions ``a`` onto
    the centred positions ``b``; no scale (``None``) when ``a`` does not spread."""
    covariance = b.T @ a / len(a)
    u, d, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rotation = u @ sign @ vt
    variance = (a**2).sum() / len(a)
    scale = float(np.trace(np.diag(d) @ sign) / variance) if variance > 0 else None
    return scale, rotation


def _similarity(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], PositionMap]:
    """Scale, rotation and translation of the best similarity, and the map of positions by
    it. Positions below ``2**SAFE_EXPONENT`` are fitted and mapped as they always were."""
    if downscale_exponent(source) == 0 and downscale_exponent(target) == 0:
        mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
        found, rotation = _umeyama(source - mu_s, target - mu_t)
        scale = 1.0 if found is None else found
        translation = mu_t - scale * rotation @ mu_s

        def plain(points: NDArray[np.float64]) -> NDArray[np.float64]:
            return scale * points @ rotation.T + translation

        return scale, rotation, translation, plain
    # Larger positions (their sums or squares may overflow): fitted about their centroids,
    # each set divided by its own power of two, and mapped about the centroids too, as
    # s R (p - mu_s) + mu_t: through s R p + t, a far but compact prediction would lose its
    # shape to rounding (t cancels s R p).
    mu_s, a, shrink_s = centred(source)
    mu_t, b, shrink_t = centred(target)
    found, rotation = _umeyama(a, b)
    fitted = math.ldexp(1.0, shrink_s - shrink_t) if found is None else found  # scale 1 if none
    with np.errstate(over="ignore", invalid="ignore"):  # beyond the float range: inf (or 0)
        scale = float(np.ldexp(fitted, shrink_t - shrink_s))
        translation = mu_t - scale * rotation @ mu_s

    def about_centroids(points: NDArray[np.float64]) -> NDArray[np.float64]:
        about = np.ldexp(np.ldexp(points, -1) - np.ldexp(mu_s, -1), -shrink_s)
        return np.ldexp(fitted * about @ rotation.T, 1 + shrink_t) + mu_t

    return scale, rotation, translation, about_centroids


def similarity_transform(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Scale, rotation and translation minimising ``|s R source + t - target|`` (Umeyama).

    Any finite positions are fitted: those too large for their sums and squares (above
    ``2**SAFE_EXPONENT``) about their centroids and divided by powers of two, the others
    exactly as before. A scale or translation beyond the floating-point range comes back as
    infinity or 0; :func:`fit_similarity` maps positions without going through them."""
    scale, rotation, translation, _ = _similarity(source, target)
    return scale, rotation, translation


CANNOT_ALIGN = "the prediction cannot be aligned with the reference"


def fit_similarity(source: NDArray[np.float64], target: NDArray[np.float64]) -> PositionMap | str:
    """The map of positions by the similarity that best fits the finite positions ``source``
    onto ``target``, or why there is none. A prediction that cannot be aligned is a metric
    error (a failure of that prediction), never an exception: no twin's numbers may stop an
    experiment."""
    try:
        *_, mapping = _similarity(source, target)
        with np.errstate(over="ignore", invalid="ignore"):
            fitted = mapping(source)
    except np.linalg.LinAlgError as error:
        return f"{CANNOT_ALIGN}: {error}"
    if not np.isfinite(fitted).all():
        return f"{CANNOT_ALIGN}: aligned, its positions are beyond the floating-point range"
    return mapping
