"""``run_record.json`` (``r2s-run/1.0``): one run of a frozen twin."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import AwareDatetime, Field

from real2simlab._base import StrictModel
from real2simlab.spec import SpecId, spec_validator


class RunStatus(StrEnum):
    """How one run ended."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    """Non-zero exit, or the program changed during the run on a backend that does not keep
    the program from writing it (``local``)."""
    TIMED_OUT = "timed_out"
    INVALID_OUTPUT = "invalid_output"
    """Exit 0, but the output violates the declared format or the task's requirements."""
    INFRA_ERROR = "infra_error"
    """The sandbox failed before or around the program (e.g. a container did not start, or it
    was killed from outside), or something outside the program explains how it ended (the
    frozen twin changed from outside its sandbox, this machine's storage stopped a write)."""

    @property
    def retriable(self) -> bool:
        return self is RunStatus.INFRA_ERROR


class ImageInfo(StrictModel):
    reference: str
    digest: str | None = None


class RunLimits(StrictModel):
    timeout_s: float = Field(gt=0)
    cpus: float | None = None
    memory_mb: int | None = None
    gpu: str | None = None


class RunRecord(StrictModel):
    """One run of a frozen twin. Holds no host paths, so a run directory can be moved."""

    spec: Annotated[str, spec_validator(SpecId.RUN)] = SpecId.RUN.value
    twin_id: str
    twin_fingerprint: str
    program_fingerprint: str
    image: ImageInfo
    backend: dict[str, Any]
    label: str = "run"
    """Which run of the twin this is: ``run``, or ``rerun`` (the reproducibility check)."""
    seed: int
    limits: RunLimits
    network: str = "none"
    started_at: AwareDatetime
    duration_s: float
    exit_code: int | None = None
    """``None`` when the run timed out or never started."""
    status: RunStatus
    problems: tuple[str, ...] = ()
    output_format: str
    output_fingerprint: str | None = None
    """SHA-256 of the output directory when the run succeeded."""
    lab_version: str
    attempt: int = 1
