"""``task.toml`` (``r2s-task/1.0``)."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Self

from pydantic import AfterValidator, Field, JsonValue, PositiveFloat, PositiveInt, model_validator

from real2simlab._base import FINITE_CONFIG, StrictModel
from real2simlab.capture.model import ObjectName
from real2simlab.spec import SpecId, spec_validator
from real2simlab.twin.conventions import DeliveryConvention
from real2simlab.twin.model import ProgramRules

TASK_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*(/[a-z0-9][a-z0-9_.-]*)*$")


def _check_task_id(value: str) -> str:
    if not TASK_ID.match(value):
        raise ValueError(
            f"task id {value!r} must be lowercase path segments of [a-z0-9_.-] separated by '/'"
        )
    return value


TaskId = Annotated[str, AfterValidator(_check_task_id)]


class Split(StrEnum):
    """What a task is for."""

    EVAL = "eval"
    """Evaluation: its results are reported; never training data."""
    TRAIN = "train"
    """Training: generated to learn from (``r2s train``); never in an evaluation's results."""


class Protocol(StrEnum):
    """How a task is scored."""

    REPRODUCTION = "reproduction"
    """The output of the frozen delivery against the task's reference, by the Lab's metrics."""
    NATIVE = "native"
    """The benchmark's own official scorer, run on the method's delivery."""


class SourceInfo(StrictModel):
    """Where the task's data comes from."""

    dataset: str = Field(min_length=1)
    version: str = Field(min_length=1)
    url: str | None = None
    citation: str | None = None
    notes: str | None = None


class Resources(StrictModel):
    """Resources one run of the twin may use."""

    model_config = FINITE_CONFIG

    gpu: bool = False
    cpus: PositiveFloat = 2.0
    memory_mb: PositiveInt = 4096


class OutputRequirements(StrictModel):
    """What the delivery's output must contain, told to the method."""

    model_config = FINITE_CONFIG

    fps: PositiveFloat | None = None
    num_frames: PositiveInt | None = None
    objects: tuple[ObjectName, ...] = ()
    """Objects the output must contain, under these names."""
    width: PositiveInt | None = None
    height: PositiveInt | None = None
    """Resolution of rendered video, when the format requires one."""


class Delivery(StrictModel):
    """What a method must deliver and how it is run."""

    environment: str = Field(min_length=1)
    """The runtime image: a registered environment name (``r2s-python``) or an image reference."""
    output_format: str = Field(min_length=1)
    convention: DeliveryConvention = DeliveryConvention.R2S
    agent_time_budget_s: PositiveInt = 1200
    """Wall-clock budget of an agent building the twin."""
    run_timeout_s: PositiveInt = 300
    """Wall-clock limit of one run of the frozen twin."""
    resources: Resources = Resources()
    program: ProgramRules = ProgramRules()
    output: OutputRequirements = OutputRequirements()


class NativeScoring(StrictModel):
    """The official scorer of the benchmark, for ``protocol = "native"``."""

    scorer: str = Field(min_length=1)
    """A registered scorer plugin."""
    options: dict[str, JsonValue] = Field(default_factory=dict)


class ScoringConfig(StrictModel):
    """How the task is scored."""

    protocol: Protocol
    metrics: tuple[str, ...] = ()
    primary: str | None = None
    """The metric the reproduction skill is computed from."""
    native: NativeScoring | None = None
    metric_options: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    """Options per metric name, e.g. ``{traj3d_mean = {align = "similarity"}}``."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.protocol is Protocol.NATIVE:
            if self.native is None:
                raise ValueError('protocol "native" needs [scoring.native]')
        else:
            if not self.metrics:
                raise ValueError(f"protocol {self.protocol.value!r} needs at least one metric")
            if self.primary is None:
                raise ValueError("primary must name one of the metrics")
        if self.primary is not None and self.metrics and self.primary not in self.metrics:
            raise ValueError(f"primary {self.primary!r} is not in metrics {list(self.metrics)}")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must be distinct")
        unknown = sorted(set(self.metric_options) - set(self.metrics))
        if unknown:
            raise ValueError(f"metric_options for metrics not listed: {unknown}")
        return self


class TaskConfig(StrictModel):
    """The content of ``task.toml``."""

    spec: Annotated[str, spec_validator(SpecId.TASK)] = SpecId.TASK.value
    id: TaskId
    version: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str | None = None
    source: SourceInfo
    license: str = Field(min_length=1)
    """SPDX identifier or a short statement, e.g. ``CC-BY-NC-4.0`` or ``none (fetched at run
    time)``."""
    redistributable: bool
    """Whether the task directory may be shared; data that is not must be fetched, not copied."""
    tags: tuple[str, ...] = ()
    visibility: str | None = None
    """Name of the visibility tier the visible directory was made with, e.g. ``calibrated``."""
    split: Split = Split.EVAL
    """``eval`` (the default) or ``train``. Training data is exported only from ``train`` tasks
    unless asked otherwise, so evaluation tasks stay out of training sets."""
    delivery: Delivery
    scoring: ScoringConfig
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Free-form data for the benchmark adapter; never shown to methods."""
