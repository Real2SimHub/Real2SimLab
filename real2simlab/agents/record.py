"""``agent_record.json`` (``r2s-agent-run/1.0``) and ``build_record.json`` (``r2s-build/1.0``)."""

from __future__ import annotations

import math
import stat
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

from pydantic import AwareDatetime, Field, ValidatorFunctionWrapHandler, field_validator

from real2simlab._base import StrictModel
from real2simlab.spec import SpecId, spec_validator


def cost_amount(value: object) -> float | None:
    """``value`` as what an agent run cost in US dollars (``cost_usd``), if it is an amount of
    money: a finite number, not negative (nor a boolean, nor a whole number too large for a
    float). ``None`` otherwise: what the run cost is then unknown, never taken as $0, as money
    given back or as infinite, and never written as ``Infinity`` or ``NaN``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        amount = float(value)
    except OverflowError:
        return None
    return amount if math.isfinite(amount) and amount >= 0 else None


class Usage(StrictModel):
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None


class AgentRecord(StrictModel):
    """One agent run that built (or failed to build) a twin."""

    spec: Annotated[str, spec_validator(SpecId.AGENT_RUN)] = SpecId.AGENT_RUN.value
    method: str
    agent: str
    agent_version: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    instruction_template: str
    instruction_sha256: str
    task_id: str
    task_hash: str
    trial: int
    image: dict[str, Any]
    backend: dict[str, Any]
    network: dict[str, Any]
    started_at: AwareDatetime
    duration_s: float
    time_budget_s: float
    exit_code: int | None = None
    exit_status: str
    """``completed``, ``failed``, ``timed_out`` or ``infra_error``."""
    trajectory: str | None = None
    """Path, relative to the build's work directory (``agent/trajectory.jsonl``), of the
    redacted trajectory; named once it holds no injected credential. Readers find it here
    (:func:`recorded_file`): mini-swe-agent's is one JSON document, ``agent/trajectory.json``."""
    stderr: str | None = None
    last_message: str | None = None
    usage: Usage | None = None
    cost_usd: float | None = None
    redactions: int = 0
    delivered: bool = False
    failure: str | None = None

    @field_validator("cost_usd", mode="wrap")
    @classmethod
    def _cost_is_an_amount(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        """A number that is not an amount of money (``Infinity``, ``NaN``, negative, a boolean,
        too large for a float: :func:`cost_amount`) says nothing of what the run cost: it is
        unknown (``None``), also written as text (``"Infinity"``, ``"-3"``), which is read as
        the number it holds. A record never holds one, so it is never written with one, and one
        an earlier Lab wrote (or an agent left) reads as unknown."""
        if isinstance(value, int | float):  # a boolean or a huge whole number included
            return cost_amount(value)
        return cost_amount(handler(value))


class BuildRecord(StrictModel):
    """One build of one method on one task."""

    spec: Annotated[str, spec_validator(SpecId.BUILD)] = SpecId.BUILD.value
    method: str
    method_kind: str
    method_digest: str
    privileged: bool = False
    task_id: str
    task_hash: str
    trial: int
    seed: int
    delivered: bool
    failure: str | None = None
    twin_fingerprint: str | None = None
    agent_record: str | None = None
    session_output: str | None = None
    session_output_fingerprint: str | None = None
    """SHA-256 of the session output when the build ended; a resumed experiment reuses the
    build only while its session output is unchanged."""
    network: str | None = None
    started_at: AwareDatetime
    duration_s: float
    lab_version: str
    options: dict[str, Any] = Field(default_factory=dict)
    infra_error: str | None = None
    """Why the build failed for a reason outside the method (a sandbox that did not start, a
    host that could not copy the delivery). Such a build is not a result of the method: it is
    redone when the experiment runs again."""


def recorded_file(work_dir: Path, name: str | None) -> Path | None:
    """The log an agent record names (``trajectory``, ``stderr``, ``last_message``: a file in
    ``agent/``, relative to the build's work directory ``work_dir``) when it is there as one: a
    regular file right in the agent's log directory. ``None`` for anything else: no name, a
    path out of that directory, a directory (Harbor's ``agent/trial``) or a link (the agent
    could write there, and a link could lead anywhere)."""
    parts = PurePosixPath(name).parts if name else ()
    if len(parts) != 2 or parts[0] != "agent" or parts[1] in {".", ".."}:
        return None
    path = work_dir / "agent" / parts[1]
    try:
        mode = path.lstat().st_mode
    except OSError:
        return None
    return path if stat.S_ISREG(mode) else None
