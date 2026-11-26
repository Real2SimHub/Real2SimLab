"""Result records: ``r2s-task-result/1.0`` (one task, method, trial and seed) and
``r2s-results/1.0`` (an experiment's aggregate)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, Field, JsonValue

from real2simlab._base import StrictModel
from real2simlab.spec import SpecId, spec_validator


class MetricRecord(StrictModel):
    status: Literal["computed", "not_applicable", "error"]
    value: float | None = None
    detail: str | None = None


class ReproductionStatus(StrEnum):
    SCORED = "scored"
    RUN_FAILED = "run_failed"
    NOT_DELIVERED = "not_delivered"


class WindowScore(StrictModel):
    """Errors over a span of frames, and the skill against a static baseline."""

    frames: tuple[int, int]
    errors: dict[str, MetricRecord]
    static_errors: dict[str, MetricRecord]
    skill_raw: float | None = None
    """1 - error / static error, for the primary metric."""
    skill: float = 0.0
    """``skill_raw`` clipped to [0, 1]; 0 when it cannot be computed."""


class Reproduction(StrictModel):
    """The frozen twin's output against the task's reference."""

    status: ReproductionStatus
    run_status: str | None = None
    score: WindowScore | None = None
    """Errors over every frame, and the skill against a static scene."""
    noise_floor: float | None = None
    """Mean error of the task's other recordings of the event against the reference."""
    error_to_noise: float | None = None
    repeat_skill: float | None = None
    """The skill another recording of the same event scores: the spread between recordings. A
    twin above it reproduces this recording more closely than repeating the experiment does."""
    reproducible: bool | None = None
    """Two runs with the same seed gave the same output (when checked)."""
    session_consistent: bool | None = None
    """The agent's own output in its session matches the frozen re-run (when available)."""
    detail: str | None = None


class TaskSummary(StrictModel):
    reproduction_skill: float | None = None
    """1 - error / static error, clipped to [0, 1] (reproduction protocol)."""
    reproduction_error: float | None = None


class Headline(StrictModel):
    """The one number a task reports, as its benchmark defines it, for cross-benchmark tables.

    Under the reproduction protocol it is the reproduction skill; under the native protocol
    the benchmark's own headline metric, read from its official report by the scorer plugin.
    """

    metric: str
    value: float | None = None
    """``None`` (left out of the record) when the delivery could not be scored."""
    unit: str | None = None
    higher_is_better: bool = True
    range: tuple[float, float] | None = None
    references: dict[str, float] = Field(default_factory=dict)
    """The same metric for reference points on this task, when the benchmark or the task
    defines them: ``static`` (a scene where nothing moves), ``another recording`` (a second
    recording of the same event) or ``reference solution`` (the benchmark's own solution)."""
    source: Literal["lab", "official"]
    scorer: str | None = None
    """The native scorer plugin, whose ``aggregate`` combines tasks the benchmark's way."""


class TaskResult(StrictModel):
    """One method's result on one task, for one trial and seed."""

    spec: Annotated[str, spec_validator(SpecId.TASK_RESULT)] = SpecId.TASK_RESULT.value
    task_id: str
    task_version: str
    task_hash: str
    benchmark: str
    """The benchmark the task comes from: the first segment of its id."""
    method: str
    method_kind: str
    privileged: bool = False
    trial: int
    seed: int
    protocol: str
    protocol_version: str
    primary_metric: str | None = None
    metric_unit: str | None = None
    delivered: bool
    delivery_failure: str | None = None
    twin_fingerprint: str | None = None
    reproduction: Reproduction
    summary: TaskSummary
    native: dict[str, JsonValue] | None = None
    """The official scorer's complete report (native protocol), unchanged."""
    headline: Headline | None = None
    created_at: AwareDatetime
    lab_version: str
    infra_error: str | None = None
    """Why the Lab could not score this result for a reason outside the method: the sandbox
    did not start, the host could not copy the delivery, the official scorer failed. Such a
    result is not a score: every statistic, scorecard and reward leaves it out, and running
    the experiment again redoes it. ``None`` for every result that is a score."""


class Estimate(StrictModel):
    """A mean with a bootstrap confidence interval."""

    mean: float | None = None
    """``None`` (left out of the record) when nothing could be scored."""
    ci_low: float | None = None
    ci_high: float | None = None
    n: int = 0


class BenchmarkScore(StrictModel):
    """One method on one benchmark: the headline metric over the benchmark's tasks."""

    benchmark: str
    metric: str
    unit: str | None = None
    higher_is_better: bool = True
    range: tuple[float, float] | None = None
    value: Estimate
    """Over the results that could be scored."""
    scored: int
    results: int
    references: dict[str, float] = Field(default_factory=dict)
    """The mean of each reference point over the tasks that define it."""
    reproducible_rate: float | None = None
    """Of the deliveries re-run twice, the share whose two runs gave the same output."""
    session_consistent_rate: float | None = None
    """Of the deliveries with a session output, the share the frozen re-run reproduced."""
    rerun_compared: int | None = None
    """Results whose frozen re-run was compared: with the agent's own output in its session,
    or, without one, with a second re-run. A delivery that did not run is not compared."""
    rerun_agreed: int | None = None
    """Of ``rerun_compared``, those that agreed."""
    infra_errors: int = 0
    """Of ``results``, those the Lab could not score (``TaskResult.infra_error``): left out of
    every number above and redone when the experiment runs again."""
    delivered: int | None = None
    """Of the results the Lab could score, those whose method delivered a twin."""


class MethodSummary(StrictModel):
    method: str
    method_kind: str
    privileged: bool = False
    tasks: int
    results: int
    reproduction_skill: Estimate
    """Over the reproduction-protocol results."""
    delivered_rate: float
    benchmarks: tuple[BenchmarkScore, ...] = ()
    """The scorecard: one entry per benchmark the method was evaluated on."""
    network: str | None = None
    """How the agent's build was networked, e.g. ``allowlist`` or ``open``."""
    infra_errors: int = 0
    """Of ``results``, those the Lab could not score (``TaskResult.infra_error``): left out of
    ``reproduction_skill``, ``delivered_rate`` and the scorecard, and redone when the
    experiment runs again."""


class Results(StrictModel):
    """An experiment's aggregate over tasks, seeds and trials."""

    spec: Annotated[str, spec_validator(SpecId.RESULTS)] = SpecId.RESULTS.value
    experiment: str
    created_at: AwareDatetime
    lab_version: str
    protocol_version: str
    confidence: float = 0.95
    bootstrap_resamples: int
    methods: tuple[MethodSummary, ...]
    tasks: tuple[str, ...]
    rows: tuple[dict[str, Any], ...] = ()
    """One row per task result: the summary fields, for tables."""
    skipped: tuple[str, ...] = ()
    """The steps the experiment skipped and why: a budget spent, a method not applicable to a
    task, a training task not held out."""


def method_label(name: str, privileged: bool, network: str | None) -> str:
    """A method's name as every table shows it: marked when it read hidden data (privileged)
    or was built with an open network, so it is never read as comparable to the others."""
    return (
        name
        + (" (privileged)" if privileged else "")
        + (" (open network)" if network == "open" else "")
    )
