"""Native scorers: a benchmark's own official scoring, as a plugin.

A task with ``protocol = "native"`` names a scorer in ``[scoring.native]``. The scorer gets
the task and the method's delivery (the twin package and, if the method produced one, its own
output) and returns JSON-compatible results, which are stored as ``TaskResult.native``. Its
``headline`` reads the benchmark's headline metric out of such a report, so results from every
benchmark line up in one scorecard.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

from real2simlab import plugins
from real2simlab.scoring.model import Headline
from real2simlab.task.io import Task

SCORER_FAILED = "scorer_failed"
"""The ``status`` of a report whose scorer itself failed (not the delivery)."""


LAB_OPTIONS = ("rerun", "scorer")
"""Native options the Lab itself reads: ``rerun`` (score the frozen program's run again) and
``scorer`` (the scorer's own arguments)."""


class NativeScorer(ABC):
    name: ClassVar[str]
    summary: ClassVar[str]
    task_options: ClassVar[tuple[str, ...] | None] = None
    """The names a task's ``scoring.native.options`` may hold for this scorer (besides the
    Lab's ``LAB_OPTIONS``): task validation reports any other name (a typo). ``None`` (the
    default): not declared, not checked."""
    required_task_options: ClassVar[tuple[str, ...]] = ()
    """Those of ``task_options`` every task must give (validation reports a missing one)."""

    @abstractmethod
    def score(
        self,
        task: Task,
        *,
        delivery_dir: Path,
        output_dir: Path | None,
        work_dir: Path,
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Official scores of one delivery. ``output_dir`` is the method's own output."""

    def problems(self) -> list[str]:
        """What prevents this scorer from running here (e.g. missing GPU or weights)."""
        return []

    def failure(self, report: Mapping[str, Any]) -> str | None:
        """Why the scorer itself failed to produce ``report``, or ``None`` when the report is a
        score (including a score of a delivery that failed). A failed scorer says nothing about
        the delivery: the result is left out of every statistic and reward, and redone when
        the experiment runs again. By default: a report whose ``status`` is ``scorer_failed``,
        with its ``detail``, ``exit_code`` and ``log`` when present."""
        return default_failure(self.name, report)

    def cache_identity(self) -> str:
        """What a score depends on beyond the delivery, the task, the options and this Lab (e.g.
        the content of a scorer image, the checkpoints present): an earlier score is reused only
        while this is unchanged. Nothing by default."""
        return ""

    def headline(self, report: Mapping[str, Any]) -> Headline | None:
        """The benchmark's headline metric in one of this scorer's reports (``value`` is
        ``None`` when the delivery could not be scored); ``None`` if the benchmark has none."""
        return None

    @classmethod
    def aggregate(cls, reports: Sequence[Mapping[str, Any]]) -> dict[str, float | None] | None:
        """The headline over several tasks the way the benchmark combines them, with the same
        for each reference point (``{"value": ..., "<reference>": ...}``). ``None`` means the
        mean of the tasks' headlines, which is what most benchmarks report."""
        return None


def task_option_problems(factory: Any, options: Mapping[str, Any]) -> list[str]:
    """What is wrong with a task's native ``options`` for the scorer ``factory`` creates: the
    names it declares (``task_options``) and the Lab's own."""
    problems: list[str] = []
    rerun = options.get("rerun")
    if rerun is not None and not isinstance(rerun, bool):
        problems.append(f"rerun must be true or false, not {rerun!r}")
    declared = getattr(factory, "task_options", None)
    if not isinstance(declared, tuple):
        return problems
    names = cast(tuple[str, ...], declared)
    unknown = sorted(set(options) - set(LAB_OPTIONS) - set(names))
    if unknown:
        problems.append(
            f"unknown options {unknown}; the scorer reads {sorted({*LAB_OPTIONS, *names})}"
        )
    required = cast(tuple[str, ...], getattr(factory, "required_task_options", ()))
    missing = [name for name in required if name not in options]
    if missing:
        problems.append(f"the scorer needs {missing}")
    return problems


def default_failure(name: str, report: Mapping[str, Any]) -> str | None:
    """:meth:`NativeScorer.failure` by default: a report whose ``status`` is ``scorer_failed``."""
    if report.get("status") != SCORER_FAILED:
        return None
    facts = [
        f"{key} {report[key]}"
        for key in ("detail", "exit_code", "log")
        if report.get(key) is not None
    ]
    return f"the official scorer {name!r} failed" + (f" ({'; '.join(facts)})" if facts else "")


def scorer_failure(scorer: Any, report: Mapping[str, Any]) -> str | None:
    """Why ``scorer`` failed to produce ``report`` (its ``failure`` method, or the default for
    scorer objects without one), or ``None`` when the report is a score."""
    method = getattr(scorer, "failure", None)
    if callable(method):
        result = method(report)
        return None if result is None else str(result)
    return default_failure(str(getattr(scorer, "name", type(scorer).__name__)), report)


def scorer_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """The ``scorer`` table of a task's native options (``[scoring.native.options.scorer]``):
    arguments of the scorer plugin. ``UsageError`` if it is not a table."""
    from real2simlab.errors import UsageError

    value = options.get("scorer")
    if value is None:
        return {}
    if not isinstance(value, Mapping):  # a scalar or a list where a table belongs
        raise UsageError(
            "scoring.native.options.scorer must be a table of the scorer's options "
            f"([scoring.native.options.scorer]), not {type(value).__name__} {value!r}"
        )
    table = cast(Mapping[object, Any], value)
    return {str(k): v for k, v in table.items()}


def get_scorer(name: str, **options: Any) -> NativeScorer:
    """Create the registered scorer ``name`` with ``options``, checked against what it takes
    (``UsageError`` names an option it does not take and the ones it does)."""
    from real2simlab.options import checked_options

    factory = plugins.load("scorer", name)
    scorer = factory(**checked_options(factory, options, f"scorer {name!r}"))
    if not isinstance(scorer, NativeScorer):
        raise TypeError(f"scorer plugin {name!r} does not create a NativeScorer")
    return scorer
