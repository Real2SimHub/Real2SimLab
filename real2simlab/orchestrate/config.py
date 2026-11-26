"""Experiment configuration files (``r2s-experiment/1.0``, TOML).

::

    spec = "r2s-experiment/1.0"
    name = "synthetic-demo"

    [tasks]
    paths = ["tasks/"]                       # task directories, searched for task.toml
    benchmarks = [{ name = "synthetic", select = ["bouncing_ball"], tiers = ["calibrated"] }]
    include = ["synthetic/*"]                # glob filters on task ids
    exclude = []

    [[methods]]
    name = "oracle"

    [[methods]]
    name = "codex"
    label = "codex-medium"                   # optional, unique
    model = "gpt-5-codex"                    # every other key is a method option
    reasoning_effort = "medium"

    [run]
    backend = "docker"
    seeds = [0]
    trials = 1
    concurrency = 4
    agent_concurrency = 2
    retries = 2
    check_reproducibility = false

    [budget]
    max_agent_runs = 15
    max_wall_time_s = 7200
    # max_cost_usd = 20.0                    # only for agents whose runs report their cost

Relative paths are resolved against the configuration file's directory. ``max_cost_usd``
limits agents by what their runs say they cost: an experiment with it refuses an agent whose
runs never say it (``codex``, ``gemini_cli``, ``harbor``).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Annotated, Any, Self, cast

from pydantic import (
    AfterValidator,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    model_validator,
)

from real2simlab._base import FINITE_CONFIG, StrictModel, validation_problems
from real2simlab.errors import SpecError
from real2simlab.spec import SpecId, spec_validator

LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]*$")
MAX_LABEL = 100
"""The longest method label: it names a directory of every build and run (file systems allow
255 bytes, and names built on it, ``.partial-...``, must fit too)."""


class BenchmarkSelection(StrictModel):
    name: str = Field(min_length=1)
    select: tuple[str, ...] | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    tiers: tuple[str, ...] | None = None


class TaskSelection(StrictModel):
    paths: tuple[str, ...] = ()
    benchmarks: tuple[BenchmarkSelection, ...] = ()
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.paths and not self.benchmarks:
            raise ValueError("[tasks] needs paths or benchmarks")
        return self


class MethodEntry(StrictModel):
    """One method of the experiment: its registered name, a label, and its options."""

    name: str = Field(min_length=1)
    label: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.label or self.name


def _a_seed(value: int) -> int:
    from real2simlab.twin.contract import MAX_SEED

    if value >= MAX_SEED:  # what a twin is given (R2S_SEED): refused here, not after the builds
        raise ValueError(f"a seed must be below {MAX_SEED}")
    return value


Seed = Annotated[NonNegativeInt, AfterValidator(_a_seed)]
"""A seed a twin can be given: a whole number from 0 to ``2**32 - 1``."""


class RunSettings(StrictModel):
    backend: str = "docker"
    seeds: tuple[Seed, ...] = (0,)
    trials: PositiveInt = 1
    concurrency: PositiveInt = 4
    agent_concurrency: PositiveInt = 2
    retries: NonNegativeInt = 2
    """Retries of infrastructure failures (container start, image pull), never of a method's."""
    check_reproducibility: bool = False
    """Run every twin twice and compare the outputs (the reproducibility check)."""
    backend_options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty list of distinct integers")
        return self


class Budget(StrictModel):
    model_config = FINITE_CONFIG

    max_agent_runs: NonNegativeInt | None = None
    max_wall_time_s: PositiveFloat | None = None
    max_cost_usd: PositiveFloat | None = None


class ExperimentConfig(StrictModel):
    spec: Annotated[str, spec_validator(SpecId.EXPERIMENT)] = SpecId.EXPERIMENT.value
    name: str = Field(min_length=1)
    description: str | None = None
    tasks: TaskSelection
    methods: tuple[MethodEntry, ...] = Field(min_length=1)
    run: RunSettings = RunSettings()
    budget: Budget = Budget()

    @model_validator(mode="after")
    def _check(self) -> Self:
        keys = [entry.key for entry in self.methods]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"methods need distinct labels; repeated: {duplicates}")
        bad = [key for key in keys if not LABEL.match(key)]
        if bad:
            raise ValueError(f"invalid method labels: {bad}")
        long = [key for key in keys if len(key) > MAX_LABEL]
        if long:
            raise ValueError(
                f"method label {long[0][:40]}... has {len(long[0])} characters; labels name "
                f"directories, so they have at most {MAX_LABEL} (give a shorter label=)"
            )
        return self


def parse_experiment(data: dict[str, Any], origin: str = "<config>") -> ExperimentConfig:
    """Build a config from its TOML form; method tables may hold options inline."""
    data = dict(data)
    methods: list[dict[str, Any]] = []
    given = data.get("methods", [])
    if not isinstance(given, list):
        raise SpecError(
            f"invalid experiment {origin}",
            ["methods: must be a list of tables ([[methods]] entries), not "
             f"{type(given).__name__}"],
        )  # fmt: skip
    for index, entry in enumerate(cast(list[Any], given)):
        if not isinstance(entry, dict):
            raise SpecError(f"invalid experiment {origin}", ["methods: each entry must be a table"])
        entry = dict(cast(dict[str, Any], entry))
        name, label = entry.pop("name", None), entry.pop("label", None)
        inline = entry.pop("options", {})
        if not isinstance(inline, dict):
            raise SpecError(
                f"invalid experiment {origin}",
                [f"methods[{index}].options: must be a table of option = value, not "
                 f"{type(inline).__name__}"],
            )  # fmt: skip
        options = dict(cast(dict[str, Any], inline))
        options.update(entry)
        methods.append({"name": name, "label": label, "options": options})
    data["methods"] = methods
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as error:
        raise SpecError(f"invalid experiment {origin}", validation_problems(error)) from error


PATH_OPTIONS = ("path", "checkpoint")
"""Method options that name a file or a directory (``program``'s ``path``, a trained method's
``checkpoint``): in an experiment file, relative ones start from the file's directory."""


def load_experiment(path: Path) -> tuple[ExperimentConfig, Path]:
    """Read an experiment file; return it and the directory relative paths start from.

    Relative task paths and method options that name files (``PATH_OPTIONS``) start from the
    file's directory."""
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SpecError(f"cannot read {path}", [str(error)]) from error
    config, base = parse_experiment(data, str(path)), path.resolve().parent

    def located(key: str, value: Any) -> Any:
        if key not in PATH_OPTIONS or not isinstance(value, str) or not value:
            return value  # an empty path stays empty: the method refuses it (it is not ".")
        given = Path(value).expanduser()
        return value if given.is_absolute() else str(base / given)

    methods = tuple(
        entry.model_copy(
            update={"options": {key: located(key, value) for key, value in entry.options.items()}}
        )
        for entry in config.methods
    )
    return config.model_copy(update={"methods": methods}), base
