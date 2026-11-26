"""The evaluation's scorer as a reward: one solution on one task, scored as an evaluation would.

:func:`score_solution` packages a solution directory as a twin (the task's program rules
apply), runs it once, frozen and offline, in a sandbox backend, and scores the output with the
task's reproduction protocol: the same code as ``r2s eval``, so a reward and an evaluation of
the same program agree. The Harbor verifier (``r2s export harbor``) uses it too.

A run that fails for a reason outside the solution (the sandbox did not start: ``infra_error``)
gives no reward: :func:`score_solution` raises :class:`RewardUnavailable` instead of returning
0, so a trainer never penalises a program for a broken machine. The same holds when this
machine cannot copy or package the solution (a full disk, a file-size limit): that raises
:class:`~real2simlab.errors.InfrastructureError`. Only the solution's own faults (no entry
point, a special file, a program that breaks the task's rules) score 0. A solution or a task
that changed while it was scored (an editor, a sync tool, another command) gives no reward
either: it raises :class:`~real2simlab.errors.SetupError`, as the reward would be for what is
no longer there.

Use an isolating backend (``docker`` or ``bwrap``) for programs you did not write: the
``local`` backend runs them as you, where they could read the task's hidden reference.
"""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from real2simlab._fs import (
    clashing_input,
    exclusive_use,
    force_remove,
    has_mark,
    os_error_text,
    write_mark,
)
from real2simlab.agents.base import clean_solution, directory_problem, solution_digest
from real2simlab.errors import (
    DeliveryError,
    InfrastructureError,
    ProblemsError,
    R2SError,
    SetupError,
)
from real2simlab.formats.base import Requirements
from real2simlab.methods.base import BuildContext, package_twin
from real2simlab.runner import limits_from_resources, run_twin
from real2simlab.runner.record import RunStatus
from real2simlab.sandbox.base import Backend, ensure_chosen_ready, get_backend
from real2simlab.scoring import MethodInfo, RunOutput, Scorer, TaskResult
from real2simlab.task import Protocol, Task, load_task
from real2simlab.task.instruction import task_statement
from real2simlab.task.validate import scoring_problems
from real2simlab.twin import InvalidTwinError, program_of
from real2simlab.twin.conventions import convention_spec
from real2simlab.twin.fingerprint import is_os_litter
from real2simlab.twin.io import STAGING_SUFFIX, has_entry_point

SOLUTION = MethodInfo(name="solution", kind="method")
INFRA_ERROR = RunStatus.INFRA_ERROR.value
WORK_MARKER = ".r2s-reward"
WORK_MARK = "score_solution work directory\n"
"""Marks a work directory as score_solution's: it may clear it before scoring again."""
WORK_ENTRIES = frozenset({WORK_MARKER, "program", "twin", "twin" + STAGING_SUFFIX, "runs"})
"""What scoring writes into a work directory (:func:`_score`, and ``twin.partial`` when it was
killed while it saved the twin): all it clears there."""


class RewardUnavailable(InfrastructureError):
    """The solution could not be scored for a reason outside it (``infra_error``)."""

    def __init__(self, message: str, result: TaskResult, fix: str | None = None) -> None:
        super().__init__(message, fix)
        self.result = result


@dataclass(frozen=True)
class Scored:
    """A solution's reward and the full task result it comes from."""

    reward: float
    result: TaskResult


def reward_of(result: TaskResult) -> float:
    """A task result as a reward in [0, 1]: the reproduction skill under the reproduction
    protocol; under the native protocol, the headline metric scaled by its range (reversed when
    lower is better). 0 when nothing could be scored (no delivery, a failed run)."""
    if result.protocol == Protocol.REPRODUCTION.value:
        return float(result.summary.reproduction_skill or 0.0)
    headline = result.headline
    if headline is None or headline.value is None or headline.range is None:
        return 0.0
    low, high = headline.range
    scaled = (headline.value - low) / (high - low) if high > low else 0.0
    scaled = scaled if headline.higher_is_better else 1.0 - scaled
    return float(min(1.0, max(0.0, scaled)))


def score_solution(
    task: Task | Path,
    solution: Path,
    *,
    backend: Backend | str = "docker",
    work_dir: Path | None = None,
    seed: int = 0,
    session_output: Path | None = None,
    method: MethodInfo = SOLUTION,
    statement_root: str = "/workspace",
) -> Scored:
    """Score the solution in ``solution`` on ``task`` (a task or its directory).

    ``solution`` holds what a method delivers for the task's convention (``run.py`` and its
    files), or is a twin package whose ``program/`` holds it; either is taken as it is, never
    through a symbolic link in its place (that scores 0, as a solution that cannot be
    delivered). ``session_output``, when given, is the output the solution's author saw,
    compared with the frozen run's (``session_consistent``); one the Lab may not read as it is
    (a symbolic link, a pipe, a file it cannot read) is not compared, and one that does not
    read as its kind does not match. The run's files are kept in ``work_dir`` (a temporary
    directory otherwise); a work directory used before is cleared first, so a scoring can be
    retried in the same place. Raises :class:`RewardUnavailable` when the run
    failed for a reason outside the solution."""
    task = task if isinstance(task, Task) else load_task(Path(task))
    # The version scored: the reward is refused if the task is not this version when the
    # scoring ends (an edit while the program ran would pair one configuration's reward with
    # another's hash).
    version = task.content_hash()
    solution = program_of(Path(solution))  # a twin package (twin.json, program/) as well
    if task.config.scoring.protocol is not Protocol.REPRODUCTION:
        raise SetupError(
            f"task {task.id} is scored by its benchmark's own scorer (native protocol), which "
            "score_solution does not run",
            fix=f"evaluate it with r2s eval --tasks {task.root}",
        )
    problems = scoring_problems(task)
    if problems:  # checked before the program runs: a broken task gives no reward, not 0
        raise ProblemsError(
            f"task {task.id} cannot be scored (r2s task validate {task.root} checks it)",
            problems,
        )
    if isinstance(backend, str):
        backend = get_backend(backend)
        ensure_chosen_ready(backend)
    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="r2s-reward-") as tmp, _scoring(task, version):
            scored = _score(task, version, Path(solution), backend, Path(tmp), seed,
                            session_output, method, statement_root)  # fmt: skip
        return _unchanged(task, version, scored)
    clash = clashing_input(
        Path(work_dir),
        [Path(solution), task.root, *([Path(session_output)] if session_output else [])],
    )
    if clash is not None:
        raise SetupError(
            f"the work directory {work_dir} overlaps an input, {clash}: it is cleared before "
            "scoring",
            fix="give a work directory outside the solution, the task and the session output",
        )
    work = Path(work_dir)
    _check_work(work)
    work.mkdir(parents=True, exist_ok=True)
    with exclusive_use(work, fix="wait until it finishes, or give another --work directory"):
        _prepare_work(work)
        with _scoring(task, version):
            scored = _score(task, version, Path(solution), backend, work, seed, session_output,
                            method, statement_root)  # fmt: skip
    return _unchanged(task, version, scored)


def _unchanged(task: Task, version: str, scored: Scored) -> Scored:
    """``scored`` if the task is still the version it was loaded as; ``SetupError`` if it was
    edited while the solution was scored (the reward may mix two versions of it)."""
    if scored.result.task_hash != version or _task_changed(task, version):
        raise _changed_while_scored(task)
    return scored


@contextlib.contextmanager
def _scoring(task: Task, version: str) -> Iterator[None]:
    """Scoring ``task``, the version ``version`` names: a failure while the task was edited or
    replaced (a reference gone, a file half-written) is that edit, not a task that cannot be
    scored, whatever it raised. A failure of the machine (``InfrastructureError``) stays one."""
    try:
        yield
    except InfrastructureError:
        raise
    except Exception as error:
        if _task_changed(task, version):
            raise _changed_while_scored(task) from error
        raise


def _task_changed(task: Task, version: str) -> bool:
    """Whether ``task`` is no longer the version ``version`` names, also when it can no longer
    be read (removed, or given a symbolic link)."""
    try:
        return task.changed_since(version)
    except (OSError, ValueError):
        return True


def _changed_while_scored(task: Task) -> SetupError:
    return SetupError(
        f"task {task.id} changed while the solution was scored ({task.root}): the reward may "
        "mix two versions of it, so none is given",
        fix="score it again once the task is no longer being edited",
    )


def _check_work(work: Path) -> None:
    """Refuse a work directory that is neither new, empty nor one used to score before, and one
    used before that now holds files scoring did not write (the user's, added since)."""
    if work.exists() and not work.is_dir():
        raise SetupError(f"the work directory {work} is a file", fix="give a directory")
    if not work.is_dir() or not any(work.iterdir()):
        return
    if not _is_work(work):
        raise SetupError(
            f"the work directory {work} is not empty",
            fix="give an empty or new directory (one used to score before is reused)",
        )
    other = sorted(entry.name for entry in work.iterdir() if not _written_by_scoring(entry.name))
    if other:
        raise SetupError(
            f"the work directory {work} holds files scoring did not write "
            f"({', '.join(other[:5])}); using it again clears it, so it is refused",
            fix="move them out of it, or give a new directory",
        )


def _written_by_scoring(name: str) -> bool:
    """Whether the entry ``name`` of a work directory is one scoring writes (or a file manager's
    litter, such as ``.DS_Store``): what using the directory again clears."""
    return name in WORK_ENTRIES or is_os_litter(name)


def _is_work(work: Path) -> bool:
    """Whether ``work`` was prepared by :func:`_prepare_work`, by its marker's content (a file
    of the user's that bears its name does not count; see :func:`~real2simlab._fs.has_mark`)."""
    return has_mark(work / WORK_MARKER, WORK_MARK)


def _prepare_work(work: Path) -> None:
    """A work directory holding only its marker: new, or one this function prepared before,
    cleared of what scoring wrote there (it holds nothing else: :func:`_check_work`). The marker
    is written first and kept while the rest is removed, so scoring stopped at any point (a
    full disk, Ctrl-C, kill -9) leaves a directory the same command takes as its own."""
    _check_work(work)
    work.mkdir(parents=True, exist_ok=True)
    if not _is_work(work):
        write_mark(work / WORK_MARKER, WORK_MARK)
    for child in list(work.iterdir()):
        if child.name == WORK_MARKER or not _written_by_scoring(child.name):
            continue
        problem = force_remove(child)  # sandboxed runs may leave odd permissions
        if problem is not None:
            raise SetupError(problem, fix=f"remove {work} yourself, or give a new directory")


def _score(
    task: Task,
    version: str,
    solution: Path,
    backend: Backend,
    work: Path,
    seed: int,
    session_output: Path | None,
    method: MethodInfo,
    statement_root: str,
) -> Scored:
    scorer = Scorer(task)
    delivery = task.config.delivery
    entry = convention_spec(delivery.convention).entry_file

    def undelivered(reason: str) -> Scored:
        result = scorer.score(
            method=method, trial=0, seed=seed, delivered=False, delivery_failure=reason
        )
        return Scored(reward_of(result), result)

    linked = directory_problem(solution)  # e.g. a Harbor agent's /app/solution: never followed
    if linked is not None:
        return undelivered(linked)
    if not has_entry_point(solution, (entry,)):  # as r2s eval --method program:path= decides
        return undelivered(f"no {entry} in {solution.absolute().name}/")
    delivered = solution_digest(solution)  # what is scored: compared once copied and once run
    try:  # a special file is the solution's fault (scored 0); a failed copy is this machine's
        program = clean_solution(solution, work / "program")
    except DeliveryError as error:
        return undelivered(str(error))
    _same_solution(solution, delivered, "copied")
    context = BuildContext(
        task_id=task.id,
        task_hash=version,
        config=task.config,
        statement=task_statement(task, statement_root),
        visible_dir=task.visible_dir,
        work_dir=work,
        twin_dir=work / "twin",
        backend=backend,
        seed=seed,
    )
    try:
        twin = package_twin(context, program, method=method.name)
    except InfrastructureError:
        raise
    except (InvalidTwinError, R2SError, ValueError) as error:
        return undelivered(f"the solution is not a valid twin: {error}")
    except OSError as error:
        raise InfrastructureError(
            f"no reward: this machine could not package the solution: {os_error_text(error)}"
        ) from error
    run = run_twin(
        twin,
        work / "runs" / "run",
        backend=backend,
        seed=seed,
        limits=limits_from_resources(delivery.resources, delivery.run_timeout_s),
        requirements=Requirements.from_task(delivery.output),
    )
    infra = run.record.status is RunStatus.INFRA_ERROR
    problems = "; ".join(run.record.problems) or "the sandbox failed"
    result = scorer.score(
        method=method,
        trial=0,
        seed=seed,
        delivered=True,
        twin_fingerprint=twin.fingerprint,
        run=RunOutput(run.record.status.value, run.output_dir, problems if infra else None),
        session_output=session_output if session_output and session_output.is_dir() else None,
    )
    if infra:
        raise RewardUnavailable(
            f"no reward: the run failed for a reason outside the solution (infra_error): "
            f"{problems}",
            result,
            fix="this machine's failure, not the solution's: do what the reason says (a sandbox "
            "or an image to repair, a disk to free), then score the same solution again; a "
            "trainer drops the sample instead of counting it as 0",
        )
    _same_solution(solution, delivered, "scored")
    return Scored(reward_of(result), result)


def _same_solution(solution: Path, delivered: str, step: str) -> None:
    """Raise ``SetupError`` unless ``solution`` still holds what it held when its copy was
    made (``delivered``, a :func:`~real2simlab.agents.base.solution_digest`): edited while it
    was copied, the copy may mix two versions of it; edited while it was scored, the reward
    would be paired with a program that is no longer the one it is the reward of."""
    if solution_digest(solution) == delivered:
        return
    effect = (
        "the copy scored may mix two versions of it"
        if step == "copied"
        else "the reward is of what it held before, not of what it holds now"
    )
    raise SetupError(
        f"the solution {solution} changed while it was {step}: {effect}, so no reward is given",
        fix="score it again once the solution is no longer being edited",
    )
