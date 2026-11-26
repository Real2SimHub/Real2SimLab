"""Event timing errors: first contact, number of bounces, and bounce times.

A bounce is a frame where an object's height reaches a local minimum while moving down faster
than ``min_speed`` before it and up faster than ``min_speed`` after it. The first contact is
the first bounce. A missing predicted event counts as an error equal to the time from the
reference event to the end of the window.
"""

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
    option_problems,
    positive_number,
)

DEFAULT_MIN_SPEED = 0.05
"""m/s; slower vertical reversals are not counted as bounces."""


def bounce_times(
    z: NDArray[np.float64], times: NDArray[np.float64], min_speed: float
) -> list[float]:
    """Times of the bounces in a height series."""
    if len(z) < 3:
        return []
    dt = np.diff(times)
    # Between huge but finite heights a speed can overflow: infinity, still faster than
    # min_speed. A missing height (NaN) is a gap, where no bounce is counted.
    with np.errstate(over="ignore"):
        down = -(np.diff(z)) / dt  # positive when moving down
    found: list[float] = []
    for i in range(1, len(z) - 1):
        if not np.isfinite(z[i - 1 : i + 2]).all():
            continue
        before, after = down[i - 1], -down[i]
        if z[i] <= z[i - 1] and z[i] < z[i + 1] and before > min_speed and after > min_speed:
            found.append(float(times[i]))
    return found


class _EventMetric(Metric):
    formats: ClassVar[frozenset[str]] = frozenset({"tracks"})
    unit: ClassVar[str] = "s"

    def option_problems(self, options: Mapping[str, Any]) -> list[str]:
        return option_problems(self.name, options, {"min_speed": positive_number})

    def compute(
        self,
        prediction: Tracks,
        reference: Tracks,
        *,
        frames: tuple[int, int] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> MetricValue:
        min_speed = float((options or {}).get("min_speed", DEFAULT_MIN_SPEED))
        aligned = align(prediction, reference, frames)
        if isinstance(aligned, str):
            return MetricValue.error(aligned)
        events = {
            name: (
                bounce_times(aligned.expected[name][:, 2], aligned.times, min_speed),
                bounce_times(aligned.predicted[name][:, 2], aligned.times, min_speed),
            )
            for name in aligned.expected
        }
        if not any(expected for expected, _ in events.values()):
            return MetricValue.not_applicable("no bounce in the reference")
        return self.score(events, aligned)

    def score(
        self, events: dict[str, tuple[list[float], list[float]]], aligned: Aligned
    ) -> MetricValue:
        raise NotImplementedError


class FirstContactTime(_EventMetric):
    name: ClassVar[str] = "event_first_contact"
    description: ClassVar[str] = "Absolute error of the first contact (first bounce) time."

    def score(
        self, events: dict[str, tuple[list[float], list[float]]], aligned: Aligned
    ) -> MetricValue:
        end = float(aligned.times[-1])
        errors = [
            abs(predicted[0] - expected[0]) if predicted else end - expected[0]
            for expected, predicted in events.values()
            if expected
        ]
        return MetricValue.computed(float(np.mean(errors)))


class BounceCount(_EventMetric):
    name: ClassVar[str] = "event_bounce_count"
    unit: ClassVar[str] = "1"
    description: ClassVar[str] = (
        "Absolute difference in the number of bounces, averaged over objects."
    )

    def score(
        self, events: dict[str, tuple[list[float], list[float]]], aligned: Aligned
    ) -> MetricValue:
        errors = [abs(len(predicted) - len(expected)) for expected, predicted in events.values()]
        return MetricValue.computed(float(np.mean(errors)))


class BounceTimes(_EventMetric):
    name: ClassVar[str] = "event_bounce_times"
    description: ClassVar[str] = (
        "Mean absolute error of each reference bounce time, matched in order; an unmatched "
        "bounce counts as the time left until the end of the window."
    )

    def score(
        self, events: dict[str, tuple[list[float], list[float]]], aligned: Aligned
    ) -> MetricValue:
        end = float(aligned.times[-1])
        errors: list[float] = []
        for expected, predicted in events.values():
            for index, time in enumerate(expected):
                errors.append(
                    abs(predicted[index] - time) if index < len(predicted) else end - time
                )
        return MetricValue.computed(float(np.mean(errors)))
