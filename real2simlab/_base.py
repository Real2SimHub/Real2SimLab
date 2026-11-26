"""Base class for every Real2SimLab data model."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

FINITE_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
"""The ``model_config`` of the parts of an input that hold numbers (``task.toml``,
``capture.json``, an experiment, a method's options): ``StrictModel``'s, and ``inf`` and
``nan`` are refused. ``PositiveFloat`` alone takes ``inf`` (it is above 0), which then fails
far from where it was given (``cpus = inf`` made a whole number of threads). Records keep
``StrictModel``'s configuration: a result may be non-finite."""


class StrictModel(BaseModel):
    """An immutable model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_json(self) -> dict[str, Any]:
        """The JSON-compatible form, omitting unset optional fields."""
        return self.model_dump(mode="json", exclude_none=True)

    def dumps(self) -> str:
        """The JSON text written to disk: indented, UTF-8, ending in a newline."""
        return json.dumps(self.to_json(), indent=2, ensure_ascii=False) + "\n"


def validation_problems(error: ValidationError, prefix: str = "") -> list[str]:
    """One line per problem of a pydantic ``ValidationError``: ``<location>: <message>``."""
    lines: list[str] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "(root)"
        lines.append(f"{prefix}{location}: {item['msg']}")
    return lines
