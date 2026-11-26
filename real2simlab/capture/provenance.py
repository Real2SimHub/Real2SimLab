"""Where a quantity came from."""

from __future__ import annotations

from enum import StrEnum
from typing import Generic, TypeVar

from real2simlab._base import FINITE_CONFIG, StrictModel

T = TypeVar("T")


class Provenance(StrEnum):
    """How a value was obtained."""

    MEASURED = "measured"
    """Measured directly, e.g. checkerboard calibration or a scale reading."""
    ESTIMATED = "estimated"
    """Estimated by a model or an algorithm, e.g. monocular intrinsics or a guessed mass."""
    DEFAULT = "default"
    """A default used for lack of information, e.g. standard gravity."""
    HUMAN_EDITED = "human_edited"
    """Set or corrected by a person."""
    SYNTHETIC = "synthetic"
    """Exact, because the data was generated: the value used by the generator."""


class Sourced(StrictModel, Generic[T]):
    """A value together with its provenance and an optional free-text source."""

    model_config = FINITE_CONFIG

    value: T
    provenance: Provenance
    source: str | None = None
