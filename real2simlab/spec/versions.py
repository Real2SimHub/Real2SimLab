"""Version identifiers of every Real2SimLab specification object.

A document names its specification as ``r2s-<object>/<major>.<minor>``. A reader accepts the
same major version with a minor version up to the one it implements. Newer minor versions may
add optional fields, which a strict older reader would otherwise reject without explanation.
The specification set as a whole has the status ``SPEC_STATUS``; the status is not part of the
version strings, so documents do not change when it does.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from enum import StrEnum

from pydantic import AfterValidator

SPEC_STATUS = "1.0"
"""Status of the specification set implemented by this version of Real2SimLab."""

_SPEC = re.compile(r"^(?P<name>[a-z0-9-]+)/(?P<major>\d+)\.(?P<minor>\d+)$")


class SpecId(StrEnum):
    """Every specification object and the version this Lab writes."""

    TASK = "r2s-task/1.0"
    CAPTURE = "r2s-capture/1.0"
    TWIN = "r2s-twin/1.0"
    TRACKS = "r2s-tracks/1.0"
    RUN = "r2s-run/1.0"
    BUILD = "r2s-build/1.0"
    AGENT_RUN = "r2s-agent-run/1.0"
    TASK_RESULT = "r2s-task-result/1.0"
    RESULTS = "r2s-results/1.0"
    EXPERIMENT = "r2s-experiment/1.0"
    TRAINING = "r2s-training/1.0"

    @property
    def object_name(self) -> str:
        """The object part, e.g. ``r2s-task``."""
        return self.value.split("/")[0]


def check_spec(value: str, expected: SpecId) -> str:
    """Return ``value`` if a reader of ``expected`` can read it; raise ``ValueError`` otherwise."""
    match = _SPEC.match(value)
    want = _SPEC.match(expected.value)
    if want is None:  # pragma: no cover - SpecId values are constants
        raise AssertionError(expected)
    if match is None or match["name"] != want["name"]:
        raise ValueError(f"expected spec {expected.value!r}, got {value!r}")
    if match["major"] != want["major"]:
        raise ValueError(
            f"{value!r} has major version {match['major']}; this Real2SimLab reads "
            f"{expected.value!r}. Use a Real2SimLab release that supports it."
        )
    if int(match["minor"]) > int(want["minor"]):
        raise ValueError(
            f"{value!r} is newer than {expected.value!r}, the newest this Real2SimLab reads; "
            "upgrade Real2SimLab (git pull in your clone, then pip install -e . there)"
        )
    return value


def spec_validator(expected: SpecId) -> AfterValidator:
    """The validator for a model's ``spec`` field: ``Annotated[str, spec_validator(...)]``."""

    def check(value: str) -> str:
        return check_spec(value, expected)

    checker: Callable[[str], str] = check
    return AfterValidator(checker)
