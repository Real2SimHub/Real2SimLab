"""Running an experiment: build twins, re-run them frozen, score, aggregate, report.

Builds run in parallel (agents in their own, smaller pool); each finished build's runs and
scoring follow in the main pool. Every step is keyed by the content hash of its inputs, so an
interrupted experiment resumes where it stopped and gives the same results. Infrastructure
failures (a container that does not start) are retried; a method's own failures are results.
Agent runs, wall time and cost are budgeted; when a budget is spent, the remaining agent
builds are skipped and reported as such.
"""

from __future__ import annotations

import contextlib
import fcntl
import fnmatch
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, TextIO, TypeGuard, cast

from pydantic import ValidationError

from real2simlab import __version__
from real2simlab._base import validation_problems
from real2simlab._fs import atomic_write_text, clashing_input, lab_record, read_regular
from real2simlab.agents.base import AgentMethod, agent_started
from real2simlab.agents.record import AgentRecord, BuildRecord, cost_amount
from real2simlab.agents.redact import environment_secrets, redact
from real2simlab.benchmarks.base import get_benchmark
from real2simlab.envs import image_reference
from real2simlab.errors import (
    InfrastructureError,
    InputNotFoundError,
    OutputExistsError,
    ProblemsError,
    R2SError,
    SetupError,
    UsageError,
)
from real2simlab.formats.base import Requirements, tree_problems
from real2simlab.methods.base import BuildContext, BuildOutcome, Method, get_method
from real2simlab.options import CredentialOptionError, checked_options
from real2simlab.orchestrate.config import PATH_OPTIONS, ExperimentConfig, MethodEntry
from real2simlab.orchestrate.layout import (
    COMPLETE,
    RUNNING,
    RunLayout,
    fresh,
    read_key,
    write_json,
    write_key,
)
from real2simlab.runner import (
    FrozenTwinModifiedError,
    RunStatus,
    limits_from_resources,
    read_record,
    run_twin,
)
from real2simlab.runner.run import twin_changed
from real2simlab.sandbox.base import Backend, ensure_chosen_ready, get_backend
from real2simlab.sandbox.process import cancel_all, reset_cancel
from real2simlab.scoring import MethodInfo, Results, RunOutput, Scorer, TaskResult, aggregate
from real2simlab.scoring.native import get_scorer, scorer_failure, scorer_options
from real2simlab.task import Task, find_tasks, load_task, validate_task
from real2simlab.task.instruction import statement_from
from real2simlab.task.io import INSTRUCTION_FILE, TASK_FILE, VISIBLE_DIR, copy_exactly
from real2simlab.twin import InvalidTwinError, LoadedTwin, load_twin
from real2simlab.twin.conventions import convention_spec
from real2simlab.twin.fingerprint import fingerprint, text_digest

BUILD_RECORD = "build_record.json"
RUN, RERUN = "run", "rerun"


@dataclass
class Unit:
    task: Task
    entry: MethodEntry
    method: Method
    trial: int

    @property
    def label(self) -> str:
        return f"{self.entry.key} on {self.task.id} (trial {self.trial})"


@dataclass
class Built:
    unit: Unit
    key: str
    record: BuildRecord | None
    twin: LoadedTwin | None
    skipped: str | None = None
    session_output: Path | None = None


@dataclass(frozen=True)
class _Validated:
    """What the builds of a task give a method, as the task was validated: the fingerprint of
    its ``visible/`` and the bytes of its ``instruction.md`` (``None``: it has none), which
    replaces the generated statement. The two are read apart, and the statement is kept as
    bytes that a build decodes only once it has compared them: whatever ``instruction.md``
    holds, the record of ``visible/`` is kept. ``unread`` is what could not be read, and why
    (``visible`` is ``None`` only then): the builds of such a task give the method nothing,
    never the task as it is by then in place of the record."""

    visible: str | None
    instruction: bytes | None
    unread: str | None = None

    @classmethod
    def read(cls, root: Path) -> _Validated:
        """What the task at ``root`` holds now. A symbolic link is never read (validation
        reports it, as it does an unreadable file; a build refuses a link as the statement)."""
        visible: str | None = None
        instruction: bytes | None = None
        unread: list[str] = []
        try:
            visible = fingerprint(root / VISIBLE_DIR)
        except (OSError, ValueError) as error:
            unread.append(f"{VISIBLE_DIR}/: {error}")
        statement = root / INSTRUCTION_FILE
        try:
            if not statement.is_symlink() and statement.is_file():
                instruction = statement.read_bytes()
        except OSError as error:
            unread.append(f"{INSTRUCTION_FILE}: {error}")
        return cls(visible, instruction, "; ".join(unread) or None)


@dataclass
class Progress:
    builds: int = 0
    runs: int = 0
    reused_builds: int = 0
    reused_runs: int = 0
    scores: int = 0
    reused_scores: int = 0
    skipped: list[str] = field(default_factory=lambda: [])


@dataclass(frozen=True)
class _Inputs:
    """What a build is a build of, as read when it was keyed: the task's content hash, the
    method's configuration digest (with its program or checkpoint), and the key they make."""

    task_hash: str
    method_digest: str
    key: str


@dataclass(frozen=True)
class _Raised(BuildOutcome):
    """The outcome of a build whose method raised an error it should have reported
    (:meth:`Experiment._attempt_build`): what the build did is not known."""


LOCK_FILE = "lock"
"""``state/lock``: held (``flock``) by the command using the run directory."""
UNKNOWN_COST = "cost_unknown_runs"
"""In ``state/budget.json``, once there is one: the agent runs that started and did not say what
they cost (``cost_usd`` is what the others did). Never counted as $0: with a cost budget, no
agent starts once there is one."""
BUILT_TASKS = "built_tasks.json"
"""The tasks a benchmark selection's build wrote, relative to its directory."""
NEW_OUT = "choose a new --out (the Lab writes into, and resumes, only its own run directories)"


def _declared_id(root: Path) -> str | None:
    """The id ``root``'s ``task.toml`` declares, read without validating the task (``None``
    when it cannot be read)."""
    import tomllib

    try:
        data = tomllib.loads((root / TASK_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None
    declared = data.get("id")
    return declared if isinstance(declared, str) and declared else None


class Experiment:
    """One experiment configuration bound to a run directory."""

    def __init__(
        self,
        config: ExperimentConfig,
        out_dir: Path,
        *,
        base_dir: Path | None = None,
        backend: Backend | None = None,
        resume: bool = True,
        log: Callable[[str], None] | None = None,
        report: bool = True,
    ) -> None:
        self.config = config
        self.layout = RunLayout(Path(out_dir))
        self.base_dir = Path(base_dir or Path.cwd())
        self.backend = backend
        self.resume = resume
        self.make_report = report
        self._log_lock = threading.Lock()
        self._budget_lock = threading.Lock()
        self._running = 0  # agent runs this command reserved and has not settled yet
        self._external_log = log
        self.progress = Progress()
        self._built: dict[tuple[str, str, int], _Inputs] = {}
        """What each unit this run built or reused (task id, method label, trial) was built from,
        as its build's key holds it: checked again before the run is complete."""
        self._fatal: BaseException | None = None
        self._started = time.monotonic()
        self._time_before = 0.0  # the wall time this run's earlier commands spent
        self._lock: TextIO | None = None
        self._backend: Backend | None = None
        self._image_ids: dict[str, str] = {}
        self._validated: dict[Path, _Validated] = {}
        """Each task's ``visible/`` and statement as they were validated, by task directory:
        exactly what its builds give a method."""

    # --- logging and budget --------------------------------------------------------------

    def log(self, message: str) -> None:
        line = f"[r2s {datetime.now().strftime('%H:%M:%S')}] {message}"
        with self._log_lock:
            self.layout.root.mkdir(parents=True, exist_ok=True)
            try:
                with self.layout.log_file.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
            except OSError as error:  # a full disk, `ulimit -f`: the error says which file
                if error.filename is not None or error.errno is None:
                    raise
                path = str(self.layout.log_file)
                raise OSError(error.errno, error.strerror, path) from error
            # A closed terminal, or a reader that went away (`r2s demo | head`): the log file
            # has the line, and the experiment goes on.
            with contextlib.suppress(OSError):
                if self._external_log is not None:
                    self._external_log(line)
                else:
                    print(line, file=sys.stderr, flush=True)

    def _budget_file(self) -> Path:
        return self.layout.state_dir / "budget.json"

    def _budget(self) -> dict[str, float]:
        """What this run has spent (``state/budget.json``): its agent runs, what they cost, how
        many of them did not say what they cost (:data:`UNKNOWN_COST`), and the wall time of its
        commands so far (``wall_time_s``, absent in a record written before it was kept). A
        missing record is counted from the agent records the run holds; a damaged one
        (unreadable, or a field the Lab writes missing or out of range) is an error: it is
        never taken for nothing spent."""
        path = self._budget_file()
        if not path.exists():
            return self._budget_from_records()
        fix = (
            "restore it from a copy, or remove it: the Lab then counts the agent runs and their "
            "cost again from the agent records in the run directory (a build done again holds "
            "the record of its last run only)"
        )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise SetupError(
                f"the budget record {path} cannot be read ({error}): what this run has spent "
                "is unknown",
                fix=fix,
            ) from error
        fields = cast(dict[str, Any], data) if isinstance(data, dict) else {}
        runs = fields.get("agent_runs")
        cost = fields.get("cost_usd")  # always written: a record without it is damaged, not $0
        unknown = fields.get(UNKNOWN_COST, 0)  # written once there is one
        spent = fields.get("wall_time_s", 0.0)  # absent in a record written before it was kept
        if not (_count(runs) and _amount(cost) and _count(unknown) and _amount(spent)):
            raise SetupError(
                f"the budget record {path} is damaged: what this run has spent is unknown",
                fix=fix,
            )
        return {**_spent(runs, cost, unknown), "wall_time_s": float(spent)}

    def _budget_from_records(self) -> dict[str, float]:
        """What the run's agent records show it spent (nothing in a new run): see
        :meth:`_count_records`."""
        return self._count_records()[0]

    def _count_records(self) -> tuple[dict[str, float], list[str]]:
        """What the run's agent records show it spent (nothing in a new run), each run counted
        as it was when it ran: a run of which nothing ran (exit status ``infra_error``) was
        given back; one that had started counts, whether its build is a result or not, at the
        cost its record says, or as a run of unknown cost when it says none. These are the
        records the builds hold now (a build done again holds its last run's).

        A record that cannot be read (damaged, cut, not a regular file, or missing while its
        build record names it) held what a run spent: it counts as a run of unknown cost,
        never as one that cost nothing or did not run. Also returned: why each such record
        cannot be read."""
        runs, cost, unknown = 0, 0.0, 0
        unreadable: list[str] = []
        builds = self.layout.root / "builds"
        for build_dir in sorted(path for path in builds.glob("*/*/trial-*") if path.is_dir()):
            try:
                record = _agent_spending(build_dir)
            except ValueError as error:
                unreadable.append(str(error))
                runs, unknown = runs + 1, unknown + 1
                continue
            if record is None:
                continue
            ran = record.exit_status != "infra_error"
            runs += ran
            amount = cost_amount(record.cost_usd)
            if amount is not None and math.isfinite(cost + amount):
                cost += amount
            elif ran:  # it says none, or a total too large to be a number: unknown, never $0
                unknown += 1
        return _spent(runs, cost, unknown), unreadable

    def _write_budget(self, state: dict[str, float]) -> None:
        state["wall_time_s"] = round(self._wall_time(), 3)
        # The runs of unknown cost are written once there is one: until then the record is
        # the one every run directory holds.
        written = {key: value for key, value in state.items() if key != UNKNOWN_COST or value}
        # Strict JSON: what the Lab writes, its reader takes (never Infinity or NaN).
        atomic_write_text(
            self._budget_file(), json.dumps(written, indent=2, allow_nan=False) + "\n"
        )

    def _check_budget(self) -> None:
        """Before anything runs: a damaged budget record stops the run; a missing one of a
        run that has agent records is counted from them, and said, with the records that
        could not be read (runs of unknown cost) and what such a count cannot hold: the runs
        of builds done again since."""
        with self._budget_lock:
            if self._budget_file().exists():
                self._time_before = self._budget()["wall_time_s"]
                return
            state, unreadable = self._count_records()
            if any(state.values()):
                self._write_budget(state)
                unknown = int(state[UNKNOWN_COST])
                more = f" (and {len(unreadable) - 1} more)" if len(unreadable) > 1 else ""
                self.log(
                    f"the budget record was missing: counted at least {int(state['agent_runs'])} "
                    f"agent run(s) and ${state['cost_usd']:g}"
                    + (f" ({unknown} of unknown cost)" if unknown else "")
                    + " from the agent records the builds hold (a build done again holds the "
                    "record of its last run only)"
                    + (
                        f"; a run whose record cannot be read is of unknown cost: the agent "
                        f"record {unreadable[0]}{more}"
                        if unreadable
                        else ""
                    )
                )

    def _check_cost_budget(self, methods: dict[str, Method]) -> None:
        """A cost budget limits agents by what their runs say they cost: it cannot limit an
        agent whose runs never say it (:meth:`AgentMethod.cost_reported`), whose every run
        would start, counted as $0. An experiment with one refuses such an agent before it
        starts."""
        limit = self.config.budget.max_cost_usd
        if limit is None:
            return
        for key, method in methods.items():
            if isinstance(method, AgentMethod) and not method.cost_reported():
                raise SetupError(
                    f"the cost budget (max_cost_usd = {limit:g}) cannot limit method {key!r}: "
                    f"{method.name} does not report what its runs cost",
                    fix="limit its runs with max_agent_runs or max_wall_time_s in [budget] and "
                    "remove max_cost_usd, or run it in an experiment without max_cost_usd; a "
                    "cost budget limits only agents that report what their runs cost "
                    "(claude_code, opencode, mini_swe)",
                )

    def _reserve_agent_run(self) -> str | None:
        """Count one agent run against the budget; return why not, if the budget is spent, or
        the cost budget cannot be shown not to be (a run's cost is unknown)."""
        budget = self.config.budget
        with self._budget_lock:
            state = self._budget()
            if budget.max_agent_runs is not None and state["agent_runs"] >= budget.max_agent_runs:
                return f"the agent-run budget ({budget.max_agent_runs}) is spent"
            if budget.max_cost_usd is not None:
                if state["cost_usd"] >= budget.max_cost_usd:
                    return f"the cost budget (${budget.max_cost_usd:g}) is spent"
                unknown = int(state[UNKNOWN_COST]) - self._running  # not those still running
                if unknown > 0:
                    return (
                        f"the cost budget (${budget.max_cost_usd:g}) cannot be checked: "
                        f"{unknown} agent run(s) did not say what they cost"
                    )
            # Of unknown cost until it says what it cost (:meth:`_settle_agent_run`), written
            # before the agent starts: a run stopped meanwhile (Ctrl-C, kill -9) never counts
            # as one that cost nothing.
            state["agent_runs"] = state["agent_runs"] + 1
            state[UNKNOWN_COST] = state[UNKNOWN_COST] + 1
            self._running += 1
            self._write_budget(state)
        return None

    def _settle_agent_run(self, *, ran: bool, cost: float | None = None) -> None:
        """Settle an agent run :meth:`_reserve_agent_run` counted, as one of unknown cost: one of
        which nothing ran is given back; what one that ran says it cost (an amount of money)
        replaces its unknown cost; one that says nothing (or whose cost would make the total
        too large to be a number) stays one of unknown cost, never $0."""
        amount = cost_amount(cost)
        with self._budget_lock:
            self._running = max(0, self._running - 1)
            state = self._budget()
            if not ran:
                state["agent_runs"] = max(0, state["agent_runs"] - 1)
            elif amount is not None and math.isfinite(state["cost_usd"] + amount):
                state["cost_usd"] = state["cost_usd"] + amount
            else:
                return
            state[UNKNOWN_COST] = max(0, state[UNKNOWN_COST] - 1)
            self._write_budget(state)

    def _add_cost(self, cost: float | None) -> None:
        """Count what an agent run of which nothing ran says it cost all the same (an amount of
        money: anything else, a negative cost too, is nothing given back)."""
        amount = cost_amount(cost)
        if not amount:
            return
        with self._budget_lock:
            state = self._budget()
            total = state["cost_usd"] + amount
            if math.isfinite(total):
                state["cost_usd"] = total
            else:  # a total too large to be a number: unknown, never Infinity
                state[UNKNOWN_COST] = state[UNKNOWN_COST] + 1
            self._write_budget(state)

    def _wall_time(self) -> float:
        """The wall time this run has spent: its earlier commands' (``wall_time_s`` in
        ``state/budget.json``) and this command's so far."""
        return self._time_before + (time.monotonic() - self._started)

    def _record_wall_time(self) -> None:
        """Save the wall time spent so far: the wall-time budget counts the time of every
        command on the run, as the other budgets count every command's agent runs and cost."""
        with self._budget_lock:
            if self._budget_file().exists():
                self._write_budget(self._budget())
            elif self.config.budget.max_wall_time_s is not None:
                self._write_budget(self._budget_from_records())

    def _out_of_time(self) -> bool:
        limit = self.config.budget.max_wall_time_s
        return limit is not None and self._wall_time() > limit

    # --- setup -----------------------------------------------------------------------------

    def resolve_tasks(self) -> list[Task]:
        selection = self.config.tasks
        roots: list[Path] = []
        for entry in selection.paths:
            given = Path(entry).expanduser()
            path = given if given.is_absolute() else self.base_dir / given
            if not path.exists():
                raise InputNotFoundError(
                    f"task path does not exist: {path}",
                    fix=None
                    if given.is_absolute()
                    else f"relative paths start from {self.base_dir} (an experiment file's "
                    "directory); write the path relative to it, or give an absolute path",
                )
            roots.extend(find_tasks(path))
        for bench in selection.benchmarks:
            roots.extend(
                self._build_benchmark(bench.name, bench.select, bench.tiers, bench.options)
            )

        def chosen(task_id: str) -> bool:
            if selection.include and not any(
                fnmatch.fnmatchcase(task_id, p) for p in selection.include
            ):
                return False
            return not any(fnmatch.fnmatchcase(task_id, p) for p in selection.exclude)

        tasks: dict[str, Task] = {}
        for root in roots:
            # include and exclude choose tasks by id: a task they leave out (one still being
            # written, say) is not part of the run, so it is not validated either. A task
            # whose id cannot be read cannot be left out: it is validated, and refused.
            declared = _declared_id(root)
            if (selection.include or selection.exclude) and declared and not chosen(declared):
                continue
            # visible/ and the statement as they are validated (taken first: a change while it
            # is validated counts): its builds give a method exactly these, or nothing.
            validated = _Validated.read(root)
            report = validate_task(root)
            if not report.ok:
                raise ProblemsError(
                    f"task {root} is invalid",
                    [str(i) for i in report.errors],
                    fix=f"correct what `r2s task validate {root}` lists, or move the task out "
                    "of the directories the run reads"
                    + (
                        "; or leave it out by its id: --include, or include and exclude in an "
                        "experiment file"
                        if declared
                        else ""
                    ),
                )
            task = load_task(root)
            if not chosen(task.id):
                continue
            known = tasks.get(task.id)
            if known is not None and known.root.resolve() != root.resolve():
                # Two selections may both pick a task: the same task, built twice, is one.
                if known.content_hash() == task.content_hash():
                    continue
                raise SetupError(
                    f"two task directories have the id {task.id!r}: {known.root} and {root}"
                )
            tasks[task.id] = task
            self._validated[task.root] = validated
        if not tasks:
            raise SetupError(
                "the experiment selects no tasks", fix="check [tasks] paths, benchmarks and include"
            )
        return [tasks[key] for key in sorted(tasks)]

    def _build_benchmark(
        self,
        name: str,
        select: tuple[str, ...] | None,
        tiers: tuple[str, ...] | None,
        options: dict[str, Any],
    ) -> list[Path]:
        adapter = get_benchmark(name)
        settings = {
            "select": select,
            "tiers": tiers,
            "options": options,
            "lab": __version__,
            "source": adapter.source_digest(),
        }
        key = text_digest(name, json.dumps(settings, sort_keys=True, default=str))
        # Each selection of a benchmark is built into its own directory: two entries for one
        # benchmark (one selecting bouncing_ball, the other pendulum) must not replace each
        # other's tasks.
        target = self.layout.tasks_dir / name / key[:16]
        manifest = target / BUILT_TASKS
        if self.resume and read_key(target) == key:
            # Reused only while every task the build wrote is still there as it wrote it: a
            # selection with a task removed, changed or damaged (by hand, a sync client, a disk)
            # is built again, never evaluated on what is there now nor refused as invalid.
            built, changed = _built_tasks(target)
            if changed is None:
                return built
            self.log(f"benchmark {name}: {changed}; building again")
        self.log(f"building benchmark {name} into {target}")
        fresh(target)
        extra = dict(options)
        if tiers is not None:
            extra["tiers"] = tiers
        extra = checked_options(
            adapter.build, extra, f"benchmark {name!r}", reserved=("out_dir", "select")
        )
        paths = adapter.build(target, select=select, **extra)
        hashes = {
            Path(path).resolve().relative_to(target.resolve()).as_posix(): _content_hash(Path(path))
            for path in paths
        }
        manifest.write_text(json.dumps({"tasks": hashes}, indent=1) + "\n", encoding="utf-8")
        write_key(target, key)
        return paths

    def _check_images(
        self, tasks: list[Task], methods: dict[str, Method], backend: Backend
    ) -> None:
        for reference in sorted({image_reference(t.config.delivery.environment) for t in tasks}):
            backend.resolve_image(reference)
        if backend.name == "local":
            from real2simlab.sandbox.local import image_only

            for task in tasks:
                missing = image_only(convention_spec(task.config.delivery.convention).entrypoint)
                if missing:
                    raise SetupError(
                        f"task {task.id}: {missing}", fix="run with --backend docker or bwrap"
                    )
        for method in methods.values():
            problems = method.problems(backend)
            if problems:
                raise SetupError(f"method {method.name!r} cannot run: {problems[0]}")
            if isinstance(method, AgentMethod) and method.agent_image:
                applicable = [t for t in tasks if method.applicable(t) is None] or tasks
                for reference in sorted(
                    {method.image(_fake_context(t, backend)) for t in applicable}
                ):
                    backend.resolve_image(reference)
        for task in tasks:
            native = task.config.scoring.native
            if task.config.scoring.protocol.value == "native" and native is not None:
                try:
                    scorer = get_scorer(native.scorer, **scorer_options(native.options))
                except (UsageError, CredentialOptionError) as error:
                    said = error.message if isinstance(error, CredentialOptionError) else error
                    raise SetupError(
                        f"task {task.id!r} ({task.root / TASK_FILE}): {said}",
                        fix="correct [scoring.native.options.scorer] in that file, then check "
                        f"it with: r2s task validate {task.root}",
                    ) from None
                problems = scorer.problems()
                if problems:
                    raise SetupError(
                        f"task {task.id!r} is scored by {native.scorer!r}, which cannot run here: "
                        + "; ".join(problems),
                        fix="do what the message says (an official scorer may need what the "
                        "backend does not, e.g. Docker), or leave these tasks out: for `r2s run`, "
                        "with exclude in the experiment file's [tasks]; for `r2s eval`, by not "
                        "giving their directories (--tasks) or their benchmark (--benchmark)",
                    )

    def _check_credentials(self, units: list[Unit], backend: Backend) -> None:
        """Every agent method that will build has its credentials, before any agent run is
        reserved: credentials it cannot have (no key, no login, a configuration it cannot use,
        an auth command that fails, a key for an endpoint the method does not name) are an
        ``error:`` with the adapter's fix here, so the same command, once the fix is done, runs
        the agent instead of finding the budget spent by a run that never started. They are
        fetched as a build fetches them (``credentials``, in a scratch context) and dropped. A
        method whose builds are all finished, and reused, needs none."""
        checked: set[str] = set()
        for unit in units:
            method = unit.method
            if not isinstance(method, AgentMethod) or unit.entry.key in checked:
                continue
            build_dir = self.layout.build_dir(unit.task.id, unit.entry.key, unit.trial)
            reused = read_key(build_dir) == self._build_key(unit)
            if self.resume and reused and (build_dir / BUILD_RECORD).is_file():
                continue
            checked.add(unit.entry.key)
            with tempfile.TemporaryDirectory(prefix="r2s-credentials-") as scratch:
                context = replace(
                    _fake_context(unit.task, backend),
                    work_dir=Path(scratch),
                    twin_dir=Path(scratch) / "twin",
                )
                method.credentials(context)  # raises SetupError with its fix

    # --- steps -------------------------------------------------------------------------------

    def _build(self, unit: Unit, backend: Backend) -> Built:
        if self._fatal is not None:
            return Built(unit, "", None, None, skipped="the experiment stopped")
        task, method, trial = unit.task, unit.method, unit.trial
        build_dir = self.layout.build_dir(task.id, unit.entry.key, trial)
        inputs = self._inputs(unit)  # read once: the key and the record say the same
        key = inputs.key
        if self.resume and read_key(build_dir) == key and (build_dir / BUILD_RECORD).is_file():
            record = _read_build(build_dir / BUILD_RECORD)
            reason = (
                "its build record is damaged" if record is None else _stale_build(build_dir, record)
            )
            twin = None
            if reason is None and record is not None and record.delivered:
                twin = self._load_twin(build_dir, task)
                if twin is None:
                    reason = "its twin is missing or no longer valid"
                elif not record.twin_fingerprint:
                    # The fingerprint of a delivered twin is always recorded: without it,
                    # whatever twin is there now cannot be shown to be the one that was built.
                    reason = (
                        "its build record has no twin_fingerprint (the record was edited, or "
                        "written by a pre-release build of the Lab)"
                    )
                elif twin.fingerprint != record.twin_fingerprint:
                    reason = "its twin changed since it was built"
            if reason is not None:
                self.log(f"build {unit.label}: {reason}; building again")
                # Its results were scored on what is no longer there: they never count, even
                # if a budget now skips the new build.
                self._discard_results(unit)
            elif record is not None:
                self.progress.reused_builds += 1
                self._built[_unit_id(unit)] = inputs
                session = build_dir / record.session_output if record.session_output else None
                return Built(unit, key, record, twin, session_output=session)
        if self._out_of_time():
            return Built(unit, key, None, None, skipped="the wall-time budget is spent")
        if method.kind == "agent":
            reason = self._reserve_agent_run()
            if reason is not None:
                return Built(unit, key, None, None, skipped=reason)
        attempts = 1 + self.config.run.retries
        outcome: BuildOutcome | None = None
        started = datetime.now(UTC)
        clock = time.monotonic()
        worked = False  # an agent ran (and may have spent money): it is never run again here
        said: float | None = None  # what the run that worked says it cost, if it says
        cleared = False  # the build directory holds only what this build wrote
        try:
            # A new build: whatever this unit's results were, they belong to an earlier one.
            self._discard_results(unit)
            for attempt in range(1, attempts + 1):
                fresh(build_dir)
                build_dir.mkdir(parents=True)
                cleared = True
                self.log(f"build {unit.label}" + (f" (attempt {attempt})" if attempt > 1 else ""))
                outcome = self._attempt_build(unit, backend, build_dir)
                # A build that failed (a full disk) names no record: there may be one where
                # agents write it all the same, written before the failure.
                written = outcome.agent_record
                if written is None and method.kind == "agent":
                    written = build_dir / AGENT_RECORD
                agent = _read_agent(written)
                # ``infra_error`` only when nothing of the agent ran (its sandbox, its command
                # or its model API failed first); a run stopped later has its own status.
                if isinstance(outcome, _Raised):
                    # How far its method got is not known: not run again by this command (the
                    # next run of the experiment builds it). An agent may have run, and spent:
                    # it counts, at what its record says it cost if it says, else unknown.
                    if method.kind == "agent":
                        worked, said = True, agent.cost_usd if agent is not None else None
                    break
                if agent is not None and agent.exit_status != "infra_error":
                    worked, said = True, agent.cost_usd
                elif agent is not None:
                    self._add_cost(agent.cost_usd)  # nothing of it ran: what it says it spent
                elif written is not None and (os.path.lexists(written) or _began(build_dir)):
                    # No record says what the run did: it cannot be read, or the build failed
                    # after the agent had started, before its record was written. What it spent
                    # is unknown, never nothing: it counts, and this command does not run it
                    # again.
                    worked = True
                if _infra_failure(outcome) is None or worked:
                    break
        except BaseException:
            # Stopped (Ctrl-C, SIGTERM) or failed on this machine meanwhile: a run whose agent
            # began stays one of unknown cost, as it was reserved; one that never began is
            # given back. (Killed outright, the reservation stays: unknown, never $0.)
            if method.kind == "agent":
                with contextlib.suppress(Exception):
                    self._settle_agent_run(ran=worked or (cleared and _began(build_dir)))
            raise
        assert outcome is not None
        duration = time.monotonic() - clock
        self._record_wall_time()  # a command killed later still counts the time spent here
        infra = _infra_failure(outcome)
        if method.kind == "agent":
            if worked:  # what it says it cost, or unknown: never $0
                self._settle_agent_run(ran=True, cost=said)
            elif infra is not None:  # the agent never ran: the run is not spent
                self._settle_agent_run(ran=False)
            else:  # it stopped before its agent began: counted, and nothing was spent
                self._settle_agent_run(ran=True, cost=0.0)
        record = BuildRecord(
            method=unit.entry.key,
            method_kind=method.kind,
            method_digest=inputs.method_digest,
            privileged=method.privileged,
            task_id=task.id,
            task_hash=inputs.task_hash,
            trial=trial,
            seed=self.config.run.seeds[0],
            delivered=outcome.delivered,
            failure=outcome.failure,
            twin_fingerprint=outcome.twin.fingerprint if outcome.twin else None,
            agent_record=_relative(outcome.agent_record, build_dir),
            session_output=_relative(outcome.session_output, build_dir),
            session_output_fingerprint=_output_fingerprint(outcome.session_output),
            network=outcome.network,
            started_at=started,
            duration_s=round(duration, 2),
            lab_version=__version__,
            options=dict(unit.entry.options),
            infra_error=infra,
        )
        write_json_text(build_dir / BUILD_RECORD, record.dumps())
        write_key(build_dir, key)
        self._built[_unit_id(unit)] = inputs
        self.progress.builds += 1
        if isinstance(outcome, _Raised):
            status = f"not a result (redone on the next run): {infra}"
        elif infra is not None:
            status = f"failed outside the method, not a result (redone on the next run): {infra}"
        elif outcome.delivered:
            status = "delivered a twin"
        else:
            status = f"delivered nothing: {outcome.failure}"
        self.log(f"build {unit.label}: {status} ({duration:.1f} s)")
        return Built(unit, key, record, outcome.twin, session_output=outcome.session_output)

    def _build_key(self, unit: Unit) -> str:
        """What a build is a build of: the task's content, the method's configuration, the
        trial, this Lab and, for an agent, the content of the image it works in (an image
        rebuilt under the same tag is another environment). A unit's results count only while
        its build has this key."""
        return self._inputs(unit).key

    def _inputs(self, unit: Unit) -> _Inputs:
        """The task's content hash and the method's digest as they are now, and the key of a
        build of ``unit`` they make (:meth:`_build_key`)."""
        task_hash, digest = unit.task.content_hash(), unit.method.config_digest()
        key = text_digest(
            task_hash, digest, str(unit.trial), __version__, *self._build_images(unit)
        )
        return _Inputs(task_hash, digest, key)

    def _changed_inputs(self, units: list[Unit]) -> list[str]:
        """What this run built from that is no longer what its builds were keyed with: a task,
        or a method's program or checkpoint, changed while the experiment ran (an editor, a sync
        client, another command). What was built and scored from it is of neither version."""
        changed: list[str] = []
        tasks: set[tuple[Path, str]] = set()
        methods: set[tuple[str, str]] = set()
        for unit in units:
            inputs = self._built.get(_unit_id(unit))
            if inputs is None:  # not built by this run (a budget spent): ``collect`` decides
                continue
            task, entry = unit.task, unit.entry
            if (task.root, inputs.task_hash) not in tasks:
                tasks.add((task.root, inputs.task_hash))
                if _task_changed(task, inputs.task_hash):
                    changed.append(f"the task {task.id} ({task.root})")
            if (entry.key, inputs.method_digest) not in methods:
                methods.add((entry.key, inputs.method_digest))
                if _method_digest(unit.method) != inputs.method_digest:
                    named = [
                        f"{option} {entry.options[option]}"
                        for option in PATH_OPTIONS
                        if isinstance(entry.options.get(option), str)
                    ]
                    changed.append(
                        f"the method {entry.key!r}" + (f" ({', '.join(named)})" if named else "")
                    )
        return list(dict.fromkeys(changed))

    def _check_unchanged(self, units: list[Unit]) -> None:
        """Before the run is complete: raise ``SetupError`` naming what changed while the
        experiment ran (:meth:`_changed_inputs`). The run stays ``running`` (its results are not
        one complete set), and the same command builds and scores again what changed."""
        changed = self._changed_inputs(units)
        if not changed:
            return
        for what in changed:
            self.log(
                f"{what} changed while the experiment ran: what was built from it is not counted"
            )
        listed = "; ".join(changed[:3]) + (
            f"; and {len(changed) - 3} more (r2s.log names them)" if len(changed) > 3 else ""
        )
        raise SetupError(
            f"{listed}: changed while the experiment ran (an editor, a sync client, another "
            "command?), so what was built and scored from it is of neither version: it is not "
            "counted, and this run is not complete",
            fix="run the same command again once nothing is changing it: it reuses what did not "
            "change and builds and scores again only what did",
        )

    def _build_images(self, unit: Unit) -> list[str]:
        if not isinstance(unit.method, AgentMethod) or self._backend is None:
            return []
        reference = unit.method.image(_fake_context(unit.task, self._backend))
        if reference not in self._image_ids:
            resolved = self._backend.resolve_image(reference)
            self._image_ids[reference] = f"{resolved.reference}@{resolved.identity}"
        return [self._image_ids[reference]]

    def _result_paths(self, unit: Unit) -> list[Path]:
        return [
            self.layout.result_file(unit.task.id, unit.entry.key, unit.trial, seed)
            for seed in self.config.run.seeds
        ]

    def _discard_results(self, unit: Unit) -> None:
        trial_dir = self.layout.result_file(unit.task.id, unit.entry.key, unit.trial, 0).parent
        for path in trial_dir.glob("seed-*.json"):
            path.unlink()

    def _attempt_build(self, unit: Unit, backend: Backend, build_dir: Path) -> BuildOutcome:
        task, method = unit.task, unit.method
        visible = build_dir / "visible"
        # The method gets the task as it was validated: visible/ copied without following a link
        # and compared with it, and the statement of exactly that (instruction.md as validated,
        # or the statement generated from the copy). A task changed since (an editor, a sync,
        # another process), or while it is copied, gives the method nothing: not a result, and
        # built again next time. So does a task without a record of what was validated: the
        # task as it is now never takes the place of the record.
        validated = self._validated.get(task.root)
        if validated is None:
            validated = _Validated(None, None, "it is not one of the tasks validated for this run")
        if validated.visible is None or validated.unread is not None:
            problem = f"task {task.id} was not recorded as it was validated ({validated.unread})"
            return BuildOutcome(None, problem, infra_error=problem)
        instruction, changed = _statement_since(task.root, validated.instruction)
        if changed is None:
            changed = copy_exactly(task.visible_dir, visible, expected=validated.visible)
        if changed is not None:
            problem = f"task {task.id} changed since it was validated: {changed}"
            return BuildOutcome(None, problem, infra_error=problem)
        work = build_dir / "work"
        work.mkdir()
        context = BuildContext(
            task_id=task.id,
            task_hash=task.content_hash(),
            config=task.config,
            statement=statement_from(task.config, instruction, visible),
            visible_dir=visible,
            work_dir=work,
            twin_dir=build_dir / "twin",
            backend=backend,
            seed=self.config.run.seeds[0],
            trial=unit.trial,
            hidden_dir=task.hidden_dir if method.privileged else None,
            log=self.log,
        )
        try:
            return method.build(context)
        except SetupError:
            raise
        except InfrastructureError as error:
            return BuildOutcome(None, str(error), infra_error=str(error))
        except (R2SError, InvalidTwinError) as error:
            return BuildOutcome(None, f"{type(error).__name__}: {error}")
        except OSError as error:  # the host (disk, permissions), not the method
            problem = f"the host failed during the build: {type(error).__name__}: {error}"
            return BuildOutcome(None, problem, infra_error=problem)
        except subprocess.TimeoutExpired:  # a tool every build waits for stopped answering
            raise
        except Exception as error:
            # An error the method should have reported (a bug in its code, a record of its tool
            # it could not read): what its build did is not known, so it is not a result, and
            # no reason to stop the other builds. Its traceback is kept for whoever debugs it.
            self._log_traceback(unit, error)
            problem = redact(
                f"{method.name} raised {type(error).__name__} during the build: {error} (what "
                "the build did is not known; the traceback is in r2s.log)",
                environment_secrets(),  # a credential the error quotes, whatever its shape
            )
            return _Raised(None, problem, infra_error=problem)

    def _log_traceback(self, unit: Unit, error: BaseException) -> None:
        """Write ``error``'s traceback to ``r2s.log`` only (key-shaped strings and this
        process's credentials redacted, as in every record): the terminal gets the build's
        line."""
        text = redact(
            "".join(traceback.format_exception(error)).rstrip("\n"), environment_secrets()
        )
        line = f"[r2s {datetime.now().strftime('%H:%M:%S')}] build {unit.label} raised:\n{text}\n"
        with self._log_lock, contextlib.suppress(OSError):  # a full disk: the line says enough
            self.layout.root.mkdir(parents=True, exist_ok=True)
            with self.layout.log_file.open("a", encoding="utf-8") as stream:
                stream.write(line)

    def _load_twin(self, build_dir: Path, task: Task) -> LoadedTwin | None:
        try:
            return load_twin(build_dir / "twin", rules=task.config.delivery.program)
        except InvalidTwinError:
            return None

    def _run(
        self,
        unit: Unit,
        twin: LoadedTwin,
        backend: Backend,
        seed: int,
        label: str,
    ) -> RunOutput:
        task = unit.task
        delivery = task.config.delivery
        run_dir = self.layout.run_dir(task.id, unit.entry.key, unit.trial, seed, label)
        limits = limits_from_resources(delivery.resources, delivery.run_timeout_s)
        # The image's content id, not only its tag: a tag moved to another image is another run.
        image = backend.resolve_image(image_reference(twin.manifest.environment.image))
        key = text_digest(
            twin.fingerprint,
            label,
            str(seed),
            image.reference,
            image.identity,
            backend.name,
            limits.dumps(),
            json.dumps(delivery.output.to_json(), sort_keys=True),
        )
        if self.resume and read_key(run_dir) == key:
            try:
                record = read_record(run_dir)
            except (OSError, ValueError):  # damaged: the run is done again
                record = None
                self.log(f"{label} of {unit.label}: its run record is damaged; running again")
            missing = None
            if record is not None and record.output_fingerprint is not None:
                missing = _changed_output(run_dir / "output", record.output_fingerprint)
            elif record is not None and record.status is RunStatus.SUCCEEDED:
                # The fingerprint of an output that succeeded is always recorded: without it,
                # the output there now cannot be shown to be the one the run wrote.
                missing = (
                    "has no output_fingerprint in its run record (the record was edited, or "
                    "written by a pre-release build of the Lab)"
                )
            if missing:
                self.log(f"{label} of {unit.label}: its output {missing}; running again")
            elif record is not None and record.status is not RunStatus.INFRA_ERROR:
                self.progress.reused_runs += 1
                return RunOutput(record.status.value, run_dir / "output")
        requirements = Requirements.from_task(delivery.output)
        status = RunStatus.INFRA_ERROR
        problem: str | None = None
        for attempt in range(1, 2 + self.config.run.retries):
            fresh(run_dir)
            try:
                result = run_twin(
                    twin,
                    run_dir,
                    backend=backend,
                    label=label,
                    seed=seed,
                    limits=limits,
                    requirements=requirements,
                    attempt=attempt,
                )
            except FrozenTwinModifiedError:
                # Changed from outside since it was built (an editor, a sync client; also during
                # an earlier run of it): not run and not a result, not a reason to stop the
                # experiment. The next run of the experiment builds it again.
                self.log(
                    f"{label} of {unit.label}: its frozen twin changed since it was built; not "
                    "run (the next run builds it again)"
                )
                return RunOutput(
                    RunStatus.INFRA_ERROR.value,
                    run_dir / "output",
                    "its frozen twin changed since it was built, so it was not run; the next run "
                    "of the experiment builds it again",
                )
            status = result.record.status
            if not status.retriable:
                problem = None
                break
            # Why it failed outside the program, for the result when no attempt is left.
            problem = "; ".join(result.record.problems) or None
            if twin_changed(twin):
                # Changed from outside while it ran: no attempt can run it any more (run_twin
                # refuses a changed twin, which would stop the experiment). Not a result, and
                # its record says why; the next run of the experiment builds it again.
                self.log(
                    f"{label} of {unit.label}: its frozen twin changed while it ran; not retried "
                    "(the next run builds it again)"
                )
                break
            self.log(f"{label} of {unit.label}: infrastructure error, retrying")
        write_key(run_dir, key)
        self.progress.runs += 1
        return RunOutput(status.value, run_dir / "output", problem)

    def _evaluate(self, built: Built, backend: Backend) -> list[Path]:
        unit = built.unit
        task, trial = unit.task, unit.trial
        record, twin = built.record, built.twin
        if built.skipped is not None:
            self.progress.skipped.append(f"{unit.label}: {built.skipped}")
            self.log(f"skip {unit.label}: {built.skipped}")
            return []
        assert record is not None
        scorer = Scorer(task)
        info = MethodInfo(
            name=unit.entry.key, kind=unit.method.kind, privileged=unit.method.privileged
        )
        written: list[Path] = []
        for seed in self.config.run.seeds:
            path = self.layout.result_file(task.id, unit.entry.key, trial, seed)
            if twin is None:
                infra = record.infra_error or _legacy_infra_error(
                    self.layout.build_dir(task.id, unit.entry.key, trial), record
                )
                result = scorer.score(
                    method=info,
                    trial=trial,
                    seed=seed,
                    delivered=False,
                    delivery_failure=record.failure,
                    infra_error=infra,
                )
                native = task.config.scoring.native
                if task.config.scoring.protocol.value == "native" and native is not None:
                    result = self._score_nothing(unit, seed, result, infra)
            elif task.config.scoring.protocol.value == "native":
                result = self._score_native(unit, twin, backend, scorer, info, seed, built)
            else:
                result = self._score_twin(
                    unit, twin, backend, scorer, info, seed, built.session_output
                )
            write_json(path, result.to_json())
            written.append(path)
            if result.infra_error is not None:
                self.log(f"not scored {unit.label} (seed {seed}): {result.infra_error}")
        summary = _summary_text(task, built)
        self.log(f"scored {unit.label}: {summary}")
        return written

    def _score_native(
        self,
        unit: Unit,
        twin: LoadedTwin,
        backend: Backend,
        scorer: Scorer,
        info: MethodInfo,
        seed: int,
        built: Built,
    ) -> TaskResult:
        """The benchmark's official scorer on the delivery: by default the world the method
        submitted; with the native option ``rerun = true``, the world of the frozen re-run."""
        task = unit.task
        native = task.config.scoring.native
        assert native is not None
        options = dict(native.options)
        output = None if options.get("rerun") else self._session_output(unit, built)
        run: RunOutput | None = None
        reproducible: bool | None = None
        infra: str | None = None
        if options.get("rerun"):
            if not convention_spec(task.config.delivery.convention).runnable:
                raise SetupError(
                    f"task {task.id}: native option rerun = true, but "
                    f"{task.config.delivery.convention.value} deliveries cannot be run",
                    fix="remove rerun from the task's [scoring.native] options",
                )
            run = self._run(unit, twin, backend, seed, RUN)
            output = run.output_dir if run.succeeded else None
            if run.infra_failed:
                infra = run.infra_reason()
            if self.config.run.check_reproducibility and run.succeeded:
                again = self._run(unit, twin, backend, seed, RERUN)
                reproducible = _same_output(run, again, scorer)
        work = self.layout.run_dir(task.id, unit.entry.key, unit.trial, seed, "native")
        official = get_scorer(native.scorer, **scorer_options(options))
        report: dict[str, Any] = {}
        if infra is None:
            # What the score is a score of: the output scored (by content), the delivery, the
            # task, the scorer and its options, and this Lab. Reused only while all are equal.
            key = text_digest(
                "native",
                task.content_hash(),
                twin.fingerprint,
                _output_fingerprint(output) or "no output",
                native.scorer,
                json.dumps(options, sort_keys=True, default=str),
                str(getattr(official, "cache_identity", lambda: "")()),
                __version__,
            )
            report, infra = self._official_report(
                task, official, work, key, delivery_dir=twin.program_dir, output=output
            )
        result = scorer.score(
            method=info,
            trial=unit.trial,
            seed=seed,
            delivered=True,
            twin_fingerprint=twin.fingerprint,
            run=run or RunOutput("not_run"),
            reproducible=reproducible,
            session_output=built.session_output if run is not None else None,
            infra_error=infra,
        )
        headline = _headline(official, report)
        if infra is not None and headline is not None:
            headline = headline.model_copy(update={"value": None})
        return result.model_copy(update={"native": report or None, "headline": headline})

    def _official_report(
        self,
        task: Task,
        official: Any,
        work: Path,
        key: str,
        *,
        delivery_dir: Path | None,
        output: Path | None,
    ) -> tuple[dict[str, Any], str | None]:
        """The official scorer's report in ``work`` and, if it failed rather than scoring,
        why (an infrastructure error). Reused while ``key`` is unchanged and the scorer still
        reads it as a score. ``delivery_dir`` ``None`` scores a method that delivered nothing:
        an empty delivery, no output."""
        native = task.config.scoring.native
        assert native is not None
        cached = _cached_report(work, key) if self.resume else None
        # A report kept by a Lab that judged it a score, which this scorer now calls its own
        # failure (a row its evaluator could not measure), is scored again, not reused.
        if cached is not None and scorer_failure(official, cached) is None:
            self.progress.reused_scores += 1
            return cached, None
        fresh(work)  # the scorer's files of an earlier attempt are not this one's
        work.mkdir(parents=True)
        if delivery_dir is None:
            delivery_dir = work / "nothing-delivered"
            delivery_dir.mkdir()
        report: dict[str, Any] = {}
        try:
            report = official.score(
                task,
                delivery_dir=delivery_dir,
                output_dir=output,
                work_dir=work,
                options=dict(native.options),
            )
        except SetupError:
            raise
        except Exception as error:  # the scorer's failure is not a score of the delivery
            infra: str | None = (
                f"the official scorer {native.scorer!r} raised {type(error).__name__}: {error}"
            )
        else:
            infra = scorer_failure(official, report)
        self.progress.scores += 1
        if infra is None:  # only a score is kept for reuse, with the report's own digest
            text = json.dumps(report, indent=2) + "\n"
            write_json_text(work / NATIVE_REPORT, text)
            write_key(work, _report_key(key, text))
        return report, infra

    def _score_nothing(
        self, unit: Unit, seed: int, result: TaskResult, infra: str | None
    ) -> TaskResult:
        """A native task the method delivered nothing for: the official score of nothing (the
        benchmark's lowest), so that a task it failed counts against it, as on the Lab's own
        protocols; not left out of the benchmark's score."""
        task = unit.task
        native = task.config.scoring.native
        assert native is not None
        options = dict(native.options)
        official = get_scorer(native.scorer, **scorer_options(options))
        report: dict[str, Any] = {}
        if infra is None:
            work = self.layout.run_dir(task.id, unit.entry.key, unit.trial, seed, "native")
            key = text_digest(
                "native, nothing delivered",
                task.content_hash(),
                native.scorer,
                json.dumps(options, sort_keys=True, default=str),
                str(getattr(official, "cache_identity", lambda: "")()),
                __version__,
            )
            report, infra = self._official_report(
                task, official, work, key, delivery_dir=None, output=None
            )
            if infra is not None:
                result = result.model_copy(update={"infra_error": infra})
        headline = _headline(official, report or {"status": "not_delivered"})
        if infra is not None and headline is not None:
            headline = headline.model_copy(update={"value": None})
        return result.model_copy(update={"native": report or None, "headline": headline})

    def _session_output(self, unit: Unit, built: Built) -> Path | None:
        """The output the method left in its session, checked again where the official scorer
        is given it (the scorer reads it as it is): ``None`` unless it is a directory the Lab
        may read as it is, with no symbolic link, pipe, socket or device and nothing it cannot
        read (:func:`~real2simlab.formats.base.tree_problems`). The method's fault: the task is
        then scored as one with no output."""
        output = built.session_output
        problems = tree_problems(output, unreadable=True) if output is not None else []
        if output is None or not problems:
            return output
        self.log(f"{unit.label}: its session output is not scored: {'; '.join(problems[:3])}")
        return None

    def _score_twin(
        self,
        unit: Unit,
        twin: LoadedTwin,
        backend: Backend,
        scorer: Scorer,
        info: MethodInfo,
        seed: int,
        session_output: Path | None,
    ) -> TaskResult:
        run = self._run(unit, twin, backend, seed, RUN)
        reproducible = None
        if self.config.run.check_reproducibility and run.succeeded:
            again = self._run(unit, twin, backend, seed, RERUN)
            reproducible = _same_output(run, again, scorer)
        return scorer.score(
            method=info,
            trial=unit.trial,
            seed=seed,
            delivered=True,
            twin_fingerprint=twin.fingerprint,
            run=run,
            reproducible=reproducible,
            session_output=session_output,
        )

    # --- the whole experiment ----------------------------------------------------------------

    def run(self) -> Results:
        self._outside_inputs()
        created = self._claim()
        try:
            return self._run_claimed(created)
        finally:
            self._release()

    def _outside_inputs(self) -> None:
        """Refuse a run directory that is, holds or lies inside an input (a task directory, a
        program's ``path``, a ``checkpoint``): writing it would change what is evaluated while
        it is evaluated (a program copied into itself, a task whose hash changes, a program
        delivered by an earlier run removed when its build is redone)."""
        inputs: list[Path] = []
        for entry in self.config.tasks.paths:
            given = Path(entry).expanduser()
            inputs.append(given if given.is_absolute() else self.base_dir / given)
        for method in self.config.methods:
            for key in PATH_OPTIONS:
                value = method.options.get(key)
                if isinstance(value, str) and value:
                    inputs.append(Path(value).expanduser())
        clash = clashing_input(self.layout.root, inputs)
        if clash is not None:
            raise SetupError(
                f"the run directory {self.layout.root} is, holds or lies inside the input "
                f"{clash}: the run would rewrite what it evaluates",
                fix="choose a run directory (--out) apart from the tasks, programs and "
                "checkpoints it evaluates (copy an input out of a run directory to use it)",
            )

    def _claim(self) -> bool:
        """Make the run directory this command's: refuse a directory of other files (the Lab
        writes into, and resumes, only its own run directories) and one another command is
        using. Return whether the directory was created here."""
        root = self.layout.root
        if root.exists() and not root.is_dir():
            raise OutputExistsError(f"{root} is a file, not a run directory", fix=NEW_OUT)
        created = not root.exists()
        empty = created or not any(root.iterdir())
        if not empty and not _lab_run_dir(self.layout):
            raise OutputExistsError(
                f"{root} holds other files and is not a Real2SimLab run directory (it has no "
                "experiment.json or state/status.json the Lab wrote)",
                fix=NEW_OUT,
            )
        made_state = not self.layout.state_dir.exists()
        self.layout.state_dir.mkdir(parents=True, exist_ok=True)
        handle = (self.layout.state_dir / LOCK_FILE).open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.seek(0)
            holder = handle.read().strip()
            handle.close()
            raise SetupError(
                f"another r2s command is using {root}" + (f" ({holder})" if holder else ""),
                fix="wait until it finishes (then run this command again to resume), or choose "
                "another --out",
            ) from None
        self._lock = handle
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid {os.getpid()}, since {_now()}\n")
            handle.flush()
            if empty:  # the directory is the Lab's from now on, even if this run stops early
                write_json(
                    self.layout.status_file,
                    {"status": RUNNING, "lab_version": __version__, "started_at": _now()},
                )
        except BaseException:  # a full disk: undo the claim, so the same command can run again
            self._release()
            if created:
                shutil.rmtree(root, ignore_errors=True)
            elif made_state and empty:
                shutil.rmtree(self.layout.state_dir, ignore_errors=True)
            raise
        return created

    def _release(self) -> None:
        if self._lock is not None:
            self._lock.close()  # closing the file releases the lock
            self._lock = None

    def _run_claimed(self, created: bool) -> Results:
        # Everything that can fail before the first step runs is checked before this directory
        # is written to: a run that cannot start leaves an earlier, finished run as it was, and
        # removes a directory it created (_remove_created()).
        try:
            units, excluded, tasks, methods, backend = self._setup()
        except BaseException:
            if created:
                self._remove_created()
            raise
        return self._execute(units, excluded, tasks, methods, backend)

    def _remove_created(self) -> None:
        """Remove the run directory this command created and cannot start in. It is moved aside
        (``.<name>.trash-<8 hex digits>``, beside it) while its lock is still held: another
        command given the same ``--out`` can claim it only once it is gone, and loses nothing
        to this removal. Then the lock is closed, as in ``_claim`` (on NFS a file still open is
        not removed but renamed, ``.nfsXXXX``, until it is closed), and what was moved aside is
        removed; if that fails, it stays under that hidden name, in no command's way."""
        root = self.layout.root
        trash = root.with_name(f".{root.name}.trash-{uuid.uuid4().hex[:8]}")
        try:
            root.rename(trash)
        except OSError:  # it cannot be moved: removed where it is, as before
            trash = root
        self._release()
        shutil.rmtree(trash, ignore_errors=True)

    def _setup(self) -> tuple[list[Unit], list[Unit], list[Task], dict[str, Method], Backend]:
        config = self.config
        # The user's own mistakes (a method option out of range, an edited checkpoint) first:
        # they are reported even on a machine that is not ready yet, and need nothing of it.
        methods = {entry.key: get_method(entry.name, **entry.options) for entry in config.methods}
        self._check_cost_budget(methods)
        backend = self.backend or get_backend(config.run.backend, **config.run.backend_options)
        ensure_chosen_ready(backend)
        self._backend = backend
        tasks = self.resolve_tasks()
        units: list[Unit] = []
        excluded: list[Unit] = []
        for task in tasks:
            for entry in config.methods:
                why, reason = "not applicable", methods[entry.key].applicable(task)
                if reason is None:  # a trained method is never evaluated on its training data
                    why, reason = "not held out", methods[entry.key].held_out(task)
                trials = [
                    Unit(task, entry, methods[entry.key], trial)
                    for trial in range(config.run.trials)
                ]
                if reason is not None:
                    self.progress.skipped.append(f"{entry.key} on {task.id}: {why}: {reason}")
                    excluded += trials
                else:
                    units += trials
        # Check images and scorers only for what will run: a task no method applies to does not
        # need its scorer here, nor a method that applies to no task its image.
        runnable_tasks = [t for t in tasks if any(u.task.id == t.id for u in units)]
        runnable_methods = {
            k: m for k, m in methods.items() if any(u.entry.key == k for u in units)
        }
        if runnable_tasks:
            self._check_images(runnable_tasks, runnable_methods, backend)
            self._check_credentials(units, backend)
        self._check_budget()
        return units, excluded, tasks, methods, backend

    def _execute(
        self,
        units: list[Unit],
        excluded: list[Unit],
        tasks: list[Task],
        methods: dict[str, Method],
        backend: Backend,
    ) -> Results:
        config = self.config
        # Until this run finishes, the directory's results are not a complete set: readers
        # (r2s train export, r2s report) check this marker and the results' digests.
        write_json(
            self.layout.status_file,
            {"status": RUNNING, "lab_version": __version__, "started_at": _now()},
        )
        write_json(
            self.layout.experiment_file,
            {
                "config": config.to_json(),
                "lab_version": __version__,
                "backend": backend.name,
                "tasks": {
                    task.id: os.path.relpath(task.root.resolve(), self.layout.root.resolve())
                    for task in tasks
                },
            },
        )
        for unit in excluded:  # nothing scored earlier counts for a unit that is not run
            self._discard_results(unit)
        self.log(
            f"experiment {config.name!r}: {len(tasks)} tasks x {len(methods)} methods x "
            f"{config.run.trials} trials x {len(config.run.seeds)} seeds on {backend.name}"
        )
        reset_cancel()
        main = ThreadPoolExecutor(max_workers=config.run.concurrency, thread_name_prefix="r2s")
        agents = ThreadPoolExecutor(
            max_workers=config.run.agent_concurrency, thread_name_prefix="r2s-agent"
        )
        evaluations: list[Future[list[Path]]] = []
        lock = threading.Lock()
        done = threading.Event()
        pending = [len(units)]

        def evaluate(built: Built) -> list[Path]:
            try:
                return self._evaluate(built, backend)
            except BaseException as error:
                self._stop(error)
                raise

        def on_built(future: Future[Built]) -> None:
            try:
                built = future.result()
                with lock:
                    if self._fatal is None:
                        evaluations.append(main.submit(evaluate, built))
            except BaseException as error:  # the build failed, or its evaluation cannot start
                self._stop(error)
            finally:  # whatever failed: a build not counted here would be waited for for ever
                with lock:
                    pending[0] -= 1
                    if pending[0] == 0:
                        done.set()

        try:
            for unit in units:
                pool = agents if unit.method.kind == "agent" else main
                future = pool.submit(self._build, unit, backend)
                future.add_done_callback(on_built)
            if units:
                done.wait()
            for future in list(evaluations):
                with contextlib.suppress(Exception):  # recorded by evaluate; never Ctrl-C
                    future.result()
        except KeyboardInterrupt as interrupt:
            # Stop what is running (its processes are in their own session, so the terminal's
            # Ctrl-C does not reach them) instead of waiting for it; no result is written for a
            # step that was stopped, so the same command resumes it.
            cancel_all()  # first: saying so may fail (a terminal that was closed)
            with contextlib.suppress(OSError):
                self._stop(interrupt)
            with contextlib.suppress(OSError, R2SError):  # its time counts when resumed
                self._record_wall_time()
            raise
        finally:
            agents.shutdown(wait=True, cancel_futures=True)
            main.shutdown(wait=True, cancel_futures=True)
            reset_cancel()
        if self._fatal is not None:
            with contextlib.suppress(OSError, R2SError):  # the error that stopped it is reported
                self._record_wall_time()
            raise self._fatal
        self._record_wall_time()  # before the check that may stop it: its time counts
        self._check_unchanged(units)  # what it built from, checked again before it is complete
        results = self.collect(units)
        unscored = sum(method.infra_errors for method in results.methods)
        write_json(
            self.layout.status_file,
            {
                "status": COMPLETE,
                "lab_version": __version__,
                "finished_at": _now(),
                "results": len(results.rows),
                "infra_errors": unscored,
            },
        )
        if self.make_report:
            from real2simlab.report import write_report

            write_report(self.layout.root)
        self.log(
            f"done: {self.progress.builds} builds ({self.progress.reused_builds} reused), "
            f"{self.progress.runs} runs ({self.progress.reused_runs} reused), "
            + (
                f"{self.progress.scores} official scores ({self.progress.reused_scores} reused), "
                if self.progress.scores or self.progress.reused_scores
                else ""
            )
            + f"{len(self.progress.skipped)} skipped; results in {self.layout.results_file}"
        )
        return results

    def _stop(self, error: BaseException) -> None:
        """Stop the experiment for ``error`` (the first one is what it ends with). Never
        raises: saying so may fail for the same reason (a log on a disk that is full)."""
        if self._fatal is None:
            self._fatal = error
            with contextlib.suppress(Exception):
                said = "interrupted" if isinstance(error, KeyboardInterrupt) else str(error)
                self.log(f"stopping: {redact(said, environment_secrets())}")

    def collect(self, units: list[Unit]) -> Results:
        """The results of ``units`` under the current configuration.

        A unit's result files count only if its build is a build of the current task and
        method configuration: after a configuration change whose new build was skipped (a
        budget spent, the experiment stopped), the files left by the earlier configuration
        are not its results, and the unit is reported as not run. A unit this run built or
        reused counts under the key it was built with: what it was built from was checked
        unchanged since (:meth:`_check_unchanged`).
        """
        results: list[TaskResult] = []
        digests: dict[tuple[str, str, int, int], str] = {}
        network: dict[str, str] = {}
        for unit in units:
            build_dir = self.layout.build_dir(unit.task.id, unit.entry.key, unit.trial)
            paths = [p for p in self._result_paths(unit) if p.is_file()]
            built = self._built.get(_unit_id(unit))
            if read_key(build_dir) != (self._build_key(unit) if built is None else built.key):
                note = f"{unit.label}: results of an earlier configuration are not counted"
                if paths and note not in self.progress.skipped:
                    self.progress.skipped.append(note)
                continue
            for path in paths:
                data = path.read_bytes()
                result = TaskResult.model_validate_json(data)
                results.append(result)
                key = (result.method, result.task_id, result.trial, result.seed)
                digests[key] = hashlib.sha256(data).hexdigest()
            record_path = (
                self.layout.build_dir(unit.task.id, unit.entry.key, unit.trial) / BUILD_RECORD
            )
            record = _read_build(record_path)
            if record is not None and record.network:
                network[unit.entry.key] = record.network
        aggregated = aggregate(self.config.name, results, network=network)
        data = aggregated.to_json()
        for row in data.get("rows", []):
            # What each row was computed from: readers check that the file is still that one.
            row["result_sha256"] = digests.get(
                (row["method"], row["task"], row["trial"], row["seed"])
            )
        data["skipped"] = list(self.progress.skipped)
        write_json(self.layout.results_file, data)
        return aggregated


def _statement_since(root: Path, validated: bytes | None) -> tuple[str | None, str | None]:
    """The text of the ``instruction.md`` of the task at ``root`` (``None``: it has none), and
    why it is not the one that was validated (``None``: it is), whose bytes were ``validated``
    (``None``: it had none). The file is compared byte for byte before anything is decoded,
    and a symbolic link is never followed. A statement that is not UTF-8 raises
    ``UnicodeDecodeError``, as reading it does."""
    path = root / INSTRUCTION_FILE
    if path.is_symlink():
        return None, f"{path} is a symbolic link"
    now = path.read_bytes() if path.is_file() else None
    if now != validated:
        return None, f"{path} changed since it was checked"
    return (None if now is None else _text(now)), None


def _text(data: bytes) -> str:
    """``data`` as ``Path.read_text(encoding="utf-8")`` reads a file that holds it: UTF-8, with
    universal newlines (``\\r\\n`` and ``\\r`` are read as ``\\n``)."""
    with io.TextIOWrapper(io.BytesIO(data), encoding="utf-8") as text:
        return text.read()


def _infra_failure(outcome: BuildOutcome) -> str | None:
    """Why ``outcome`` failed outside the method (the method's sandbox did not start, the host
    failed), or ``None`` when it is a result of the method."""
    if outcome.infra_error is not None:
        return outcome.infra_error
    record = _read_agent(outcome.agent_record)
    if record is not None and record.exit_status == "infra_error":
        return "the agent's sandbox failed (exit status infra_error)" + (
            f": {outcome.failure}" if outcome.failure else ""
        )
    return None


def _legacy_infra_error(build_dir: Path, record: BuildRecord) -> str | None:
    """The infrastructure failure of a build recorded before ``BuildRecord.infra_error``
    existed: an agent run whose exit status is ``infra_error``."""
    if record.infra_error is not None or record.agent_record is None:
        return record.infra_error
    agent = _read_agent(build_dir / record.agent_record)
    if agent is not None and agent.exit_status == "infra_error":
        return "the agent's sandbox failed (exit status infra_error)"
    return None


def _stale_build(build_dir: Path, record: BuildRecord) -> str | None:
    """Why the recorded build in ``build_dir`` may not be reused, or ``None``: it failed
    outside the method (to be retried), or its session output is gone or changed."""
    infra = _legacy_infra_error(build_dir, record)
    if infra is not None:
        return f"it failed outside the method ({infra})"
    session = build_dir / record.session_output if record.session_output else None
    missing = _changed_output(session, record.session_output_fingerprint)
    if missing:
        return f"its session output {missing}"
    return None


def _unit_id(unit: Unit) -> tuple[str, str, int]:
    return unit.task.id, unit.entry.key, unit.trial


def _task_changed(task: Task, task_hash: str) -> bool:
    """Whether ``task`` is no longer the version ``task_hash`` names, or its ``task.toml`` no
    longer says what was loaded; also when it can no longer be read (removed, a link added)."""
    try:
        return task.changed_since(task_hash)
    except (OSError, ValueError):
        return True


def _method_digest(method: Method) -> str | None:
    """The method's configuration digest as it is now (with its program's content); ``None``
    when it can no longer be taken (the program removed, a link added to it)."""
    try:
        return method.config_digest()
    except (OSError, ValueError, R2SError):
        return None


def _content_hash(task_dir: Path) -> str | None:
    """A task directory's content hash (:meth:`Task.content_hash`); ``None`` when it cannot be
    taken (a symbolic link in it, a file that cannot be read: its validation says which)."""
    try:
        return fingerprint(task_dir)
    except (OSError, ValueError):
        return None


def _built_tasks(target: Path) -> tuple[list[Path], str | None]:
    """The tasks a benchmark selection's build wrote in ``target`` (``built_tasks.json``: each
    one's directory and content hash), and why they may not be reused, or ``None`` when every
    one is still there with the content its build recorded."""
    listed = f"its list of the tasks it built ({BUILT_TASKS})"
    try:
        data: Any = json.loads((target / BUILT_TASKS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], f"{listed} cannot be read"
    hashes: Any = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(hashes, dict):
        return [], (
            f"{listed} holds no content hashes (it was edited, or written by a pre-release build "
            "of the Lab)"
        )
    tasks: list[Path] = []
    for relative, recorded in cast(dict[str, Any], hashes).items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            return [], f"{listed} names {relative!r}, which is not in the selection"
        path = target / relative
        if not (path / TASK_FILE).is_file():
            return [], f"the task {relative} it built is missing"
        if not isinstance(recorded, str) or _content_hash(path) != recorded:
            return [], f"the task {relative} it built has changed since (edited, or damaged)"
        tasks.append(path)
    return tasks, None


def write_json_text(path: Path, text: str) -> None:
    """Write a record atomically: an interrupted write leaves the old file or the new one."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _lab_run_dir(layout: RunLayout) -> bool:
    """Whether the directory is a run directory of the Lab, by what its records say (their
    names alone prove nothing: a user's own ``experiment.json`` or ``state/``): its
    ``experiment.json`` or ``state/status.json`` is a Lab record, or it holds only the lock of
    a run stopped before it wrote them (``state/lock`` as a run writes it, and that run's
    ``r2s.log``)."""
    if lab_record(layout.experiment_file) or lab_record(layout.status_file):
        return True
    names = {entry.name for entry in layout.root.iterdir()}
    state = layout.state_dir
    return (
        names <= {state.name, layout.log_file.name}
        and state.is_dir()
        # and the status that run was writing when it stopped (a full disk, kill -9)
        and {LOCK_FILE}
        <= {entry.name for entry in state.iterdir()}
        <= {LOCK_FILE, layout.status_file.name + ".tmp"}
        and _run_lock(state / LOCK_FILE)
    )


def _run_lock(path: Path) -> bool:
    """Whether ``path`` is a lock as :meth:`Experiment._claim` writes it (``pid <n>, since
    <time>``), or an empty one (the run was stopped before it wrote that line): another
    program's ``state/lock`` is not."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return False
    return text == "" or re.fullmatch(r"pid \d+, since \S+\n", text) is not None


def _read_build(path: Path) -> BuildRecord | None:
    """The build record at ``path``, or ``None`` when it is missing or damaged (the build is
    then done again, never reused)."""
    try:
        return BuildRecord.model_validate_json(path.read_bytes())
    except (OSError, ValueError):
        return None


AGENT_RECORD = "work/agent/agent_record.json"
"""Where, in a build's directory, agents write their record (``r2s-agent-run/1.0``)."""


def _agent_spending(build_dir: Path) -> AgentRecord | None:
    """The record of the agent run the build in ``build_dir`` holds, which says what that run
    spent; ``None`` when it holds none (a method that is not an agent, an agent that failed
    before it wrote one). Read where the build record names it, else where agents write it,
    and only as a regular file (an agent could leave a link in its log directory).

    Raises ``ValueError`` naming the record and why it cannot be read: it is damaged, cut or
    not a regular file, or it is missing while its build record names it (its run is then
    one of unknown cost)."""
    build = _read_build(build_dir / BUILD_RECORD)
    named = build.agent_record if build is not None else None
    if named is not None and (
        PurePosixPath(named).is_absolute() or ".." in PurePosixPath(named).parts
    ):
        raise ValueError(f"{named!r} named by {build_dir / BUILD_RECORD} is not in its build")
    path = build_dir / (named or AGENT_RECORD)
    try:
        info = path.lstat()
    except FileNotFoundError:
        if named is None:
            return None
        raise ValueError(f"{path} is missing (its build record names it)") from None
    except OSError as error:
        raise ValueError(f"{path} cannot be read ({error.strerror or error})") from None
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{path} is not a regular file")
    try:
        data = read_regular(path)  # not through a link swapped in meanwhile, nor a pipe
    except OSError as error:
        raise ValueError(f"{path} cannot be read ({error.strerror or error})") from None
    if data is None:
        raise ValueError(f"{path} is not a regular file this user can read")
    try:
        return AgentRecord.model_validate_json(data)
    except ValidationError as error:
        raise ValueError(f"{path} is damaged ({validation_problems(error)[0]})") from None


def _began(build_dir: Path) -> bool:
    """Whether the agent of the build in ``build_dir`` began (and may have spent money): it
    left a record where agents write it, or signs of having started (:func:`agent_started`)."""
    return os.path.lexists(build_dir / AGENT_RECORD) or agent_started(build_dir / "work" / "agent")


def _count(value: object) -> TypeGuard[int]:
    """Whether ``value`` counts something: a whole number, not negative (nor a boolean)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _amount(value: object) -> TypeGuard[float]:
    """Whether ``value`` is an amount of money: a finite number, not negative (nor a whole
    number too large for a float: :func:`~real2simlab.agents.record.cost_amount`)."""
    return cost_amount(value) is not None


def _spent(runs: int, cost: float, unknown: int) -> dict[str, float]:
    """What a run has spent, as ``state/budget.json`` holds it."""
    return {"agent_runs": runs, "cost_usd": float(cost), UNKNOWN_COST: unknown}


def _read_agent(path: Path | None) -> AgentRecord | None:
    """The agent record at ``path``; ``None`` when there is none, it is not a regular file (in
    the agent's log directory: never followed or waited on) or it is damaged."""
    data = read_regular(path) if path is not None else None
    if data is None:
        return None
    try:
        return AgentRecord.model_validate_json(data)
    except ValueError:
        return None


NATIVE_REPORT = "native_report.json"
"""The official scorer's report, kept beside its files so that an unchanged score is reused."""


def _report_key(key: str, text: str) -> str:
    """The key a report is kept under: the step's key and the report's own content, so a
    report changed since it was written (by hand, by a damaged disk) is scored again."""
    return text_digest(key, hashlib.sha256(text.encode("utf-8")).hexdigest())


def _cached_report(work: Path, key: str) -> dict[str, Any] | None:
    """The report scored earlier in ``work`` under ``key``, if it is still there as written."""
    try:
        text = (work / NATIVE_REPORT).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    if read_key(work) != _report_key(key, text):
        return None
    try:
        report = json.loads(text)
    except ValueError:
        return None
    return cast(dict[str, Any], report) if isinstance(report, dict) else None


def _output_fingerprint(directory: Path | None) -> str | None:
    if directory is None or Path(directory).is_symlink() or not Path(directory).is_dir():
        return None
    try:
        return fingerprint(Path(directory))
    except (OSError, ValueError):
        return None


def _changed_output(directory: Path | None, expected: str | None) -> str | None:
    """Why the output a step recorded can no longer be used (``None`` if it can): the
    directory is gone, or no longer has the recorded fingerprint.

    The fingerprint of every output that can be fingerprinted is recorded, so a record holds
    none (``expected`` ``None``) only for an output that could not be (a symbolic link in it):
    such an output that can be fingerprinted now cannot be shown to be the one recorded."""
    if directory is None:
        return None
    if not directory.is_dir():
        return "is gone"
    current = _output_fingerprint(directory)
    if expected is not None:
        return "changed since it was recorded" if current != expected else None
    if current is not None:
        return (
            "has no fingerprint in its record, though it can be fingerprinted now (it changed "
            "since, or the record was edited or written by a pre-release build of the Lab)"
        )
    return None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _relative(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _same_output(first: RunOutput, second: RunOutput, scorer: Scorer) -> bool | None:
    """Whether two runs of a twin gave the same output; ``None`` when they cannot be compared:
    one of them failed outside the program (``infra_error``), or neither succeeded."""
    from real2simlab.scoring.reproduction import outputs_agree

    if first.infra_failed or second.infra_failed:
        return None
    if not (first.succeeded and second.succeeded):
        return False if first.succeeded != second.succeeded else None
    assert first.output_dir is not None and second.output_dir is not None
    return outputs_agree(scorer.format, first.output_dir, second.output_dir)


def _headline(scorer: Any, report: dict[str, Any]) -> Any:
    """The scorer's headline of ``report``; ``None`` for scorers that do not define one."""
    method = getattr(scorer, "headline", None)
    return method(report) if callable(method) else None


def _summary_text(task: Task, built: Built) -> str:
    if built.twin is None:
        return "no twin"
    return f"twin {built.twin.fingerprint[:12]}"


def _fake_context(task: Task, backend: Backend) -> BuildContext:
    """A minimal context, only to ask a method which image it will use."""
    return BuildContext(
        task_id=task.id,
        task_hash="",
        config=task.config,
        statement="",
        visible_dir=task.visible_dir,
        work_dir=task.visible_dir,
        twin_dir=task.visible_dir,
        backend=backend,
    )


def run_experiment(
    config: ExperimentConfig,
    out_dir: Path,
    *,
    base_dir: Path | None = None,
    backend: Backend | None = None,
    resume: bool = True,
    log: Callable[[str], None] | None = None,
    report: bool = True,
) -> Results:
    """Run ``config`` into ``out_dir``, resuming a previous run there unless ``resume`` is false."""
    return Experiment(
        config, out_dir, base_dir=base_dir, backend=backend, resume=resume, log=log, report=report
    ).run()
