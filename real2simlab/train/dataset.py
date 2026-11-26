"""Datasets for training: tasks and the results of runs on them, in one directory.

::

    <dataset>/
      README.md            what it holds: counts, licenses, provenance, the command
      export.json          the inputs, the checks made and everything left out, with reasons
      tasks.jsonl          one row per task
      tasks/<task>/visible/     what a method sees (capture.json, media)
      tasks/<task>/hidden/      the labels: reference output, hidden parameters, truth program
      runs.jsonl           one row per delivered program of the exported runs (with runs)
      runs/<run>/<task>/<method>/trial-<k>/seed-<s>/program/          the delivered program
      runs/<run>/<task>/<method>/trial-<k>/seed-<s>/trajectory.jsonl  the agent's redacted events
                           (mini_swe: trajectory.json, mini-swe-agent's one JSON document)
      overlap-prints.npz   the overlap prints of the exported tasks

What goes in. Only tasks with ``split = "train"`` (and runs on them) are exported, unless
``allow_eval_tasks`` is set. Tasks of a registered benchmark that releases no training split
(``BenchmarkAdapter.splits``) and tasks of a benchmark whose terms forbid training on its data
(``BenchmarkAdapter.training_refusal``) never are. With ``against`` (directories of evaluation
tasks), a task that overlaps one of them (the same recording, :mod:`real2simlab.task.overlap`)
is left out too. The prints of the exported tasks go with the dataset (those of tasks with a
row in ``tasks.jsonl`` or ``runs.jsonl``, and no others), so a method trained on it is never
evaluated on a task they overlap. A task that would show a method hidden data (a symbolic link,
content of ``hidden/`` in ``visible/`` or as its statement, ``instruction.md``:
:func:`real2simlab.task.validate.hidden_data_problems`) is refused; its ``visible/`` and
``hidden/`` are copied without following a link and checked again as copied, with the
``instruction.md`` the row's statement is made from.

Rewards. A run directory is exported only when its results are one complete set: the run
finished, and each result file is the one ``results.json`` was computed from
(:func:`real2simlab.orchestrate.layout.incomplete`); the reward is read from the very bytes so
checked. A run's result is exported only with the
task it was scored on: the task whose content hash the result records, found where the run
found it or among ``tasks``; and only with the program it was scored on: the frozen twin whose
fingerprint the result records. A result whose task or program is missing or has changed since
is an error (``skip_unmatched`` leaves it out instead, and says so): its reward could belong to
another version. The error says which it is, with its fix: the task as it was, or the program
scored again by the same command on the run directory. A result the Lab could not score
(``infra_error``: the sandbox, the host or the official scorer failed) is not a reward and is
left out, with or without ``include_undelivered``. Results with no delivered program are left
out unless ``include_undelivered``.

The dataset is written next to ``out`` and moved into place when complete: a failed export
leaves no partial dataset, and an existing one (``overwrite``) is replaced only then. Before it
is moved into place, every input is checked again (the tasks it holds or prints, the
``against`` tasks it was compared with, the results of the runs): one that changed while the
export ran (an editor, a sync, another command) stops it, so a dataset never pairs two
versions of an input, nor records a comparison with evaluation tasks that are no longer those.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from real2simlab import __version__
from real2simlab._base import validation_problems
from real2simlab._fs import (
    check_replaceable,
    clashing_input,
    force_remove,
    lab_record,
    read_regular,
    replace_dir,
    staging_dir,
)
from real2simlab.agents.record import AgentRecord, recorded_file
from real2simlab.agents.redact import Redactor, environment_secrets
from real2simlab.errors import R2SError, SetupError
from real2simlab.methods.training import DATASET_PRINTS
from real2simlab.orchestrate.layout import (
    RunLayout,
    incomplete,
    read_experiment,
    read_result_file,
    read_results,
    slug,
)
from real2simlab.scoring import TaskResult
from real2simlab.task import Split, Task, find_tasks, load_task
from real2simlab.task.instruction import task_statement
from real2simlab.task.io import PARAMETERS_FILE
from real2simlab.task.overlap import (
    Overlap,
    OverlapIndex,
    TaskPrint,
    default_cache,
    recordings_changed,
    task_print,
)
from real2simlab.train.reward import INFRA_ERROR, reward_of
from real2simlab.twin.fingerprint import (
    SymbolicLinkError,
    fingerprint,
    is_os_litter,
    sha256_file,
)

UNRECORDED = frozenset({"export.json", "README.md"})
"""The files of a dataset its ``content`` record leaves out: the record itself and the card."""


def _entries(dataset: Path) -> Iterator[tuple[str, Path]]:
    """Every entry of ``dataset`` but ``UNRECORDED`` (and a file manager's litter, such as
    ``.DS_Store``), by relative path, in sorted order (a symbolic link is not followed)."""
    for path in sorted(Path(dataset).rglob("*")):
        relative = path.relative_to(dataset).as_posix()
        if is_os_litter(path.name) or relative in UNRECORDED:
            continue
        yield relative, path


def _regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def content_digests(dataset: Path) -> dict[str, str]:
    """SHA-256 of every file of ``dataset`` but ``UNRECORDED`` (and a file manager's litter,
    such as ``.DS_Store``), by relative path."""
    return {relative: sha256_file(path) for relative, path in _entries(dataset) if _regular(path)}


def executable_files(dataset: Path) -> list[str]:
    """The files of ``content_digests`` that someone may execute (an executable bit set), by
    relative path: a program that runs a helper behaves differently without the bit, so it is
    content (:func:`~real2simlab.twin.fingerprint.fingerprint` counts it too), though a digest
    of the bytes cannot see it. ``export.json`` records them as ``executable``."""
    return [
        relative
        for relative, path in _entries(dataset)
        if _regular(path) and path.stat().st_mode & 0o111
    ]


def _irregular(dataset: Path) -> dict[str, str]:
    """The entries of ``dataset`` that are neither a regular file nor a directory, each with
    what it is: an export writes none (it copies what a link points to), so each was put there
    since, and no content record can vouch for what it leads to."""
    found: dict[str, str] = {}
    for relative, path in _entries(dataset):
        if path.is_symlink():
            found[relative] = "a symbolic link"
        elif not path.is_file() and not path.is_dir():
            found[relative] = "not a regular file"
    return found


def dataset_changes(dataset: Path) -> list[str]:
    """How ``dataset`` differs from what its export wrote (``export.json``'s ``content``):
    files changed, removed or added since (a symbolic link or another entry that is not a
    regular file too), and, when ``export.json`` records them (``executable``; exports before
    this record existed do not), files whose executable bit was set or cleared. An
    ``export.json`` that is missing, is not an export record, or records no content (every
    export records it: one without it was edited, or written by a pre-release build of the
    Lab, and nothing in the dataset can be checked), is a change."""
    path = Path(dataset) / EXPORT_FILE
    if not path.is_file():
        return [f"{EXPORT_FILE}: removed"]
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"{EXPORT_FILE} cannot be read: {error}"]
    if not lab_record(path):
        return [f"{EXPORT_FILE}: it is not an export record"]
    if "content" not in record:
        return [
            f"{EXPORT_FILE}: it records no content (the digest of every file, which every "
            "export records): it was edited, or written by a pre-release build of the Lab, so "
            "no file can be checked"
        ]
    recorded = record["content"]
    if not isinstance(recorded, dict):
        return [f"{EXPORT_FILE}: its content record is not a mapping of files to digests"]
    executable = record.get("executable")
    if executable is not None and not (
        isinstance(executable, list) and all(isinstance(name, str) for name in executable)
    ):
        return [f"{EXPORT_FILE}: its executable record is not a list of files"]
    current = content_digests(dataset)
    irregular = _irregular(dataset)
    changes: list[str] = []
    for name, digest in sorted(recorded.items()):
        if current.get(name) == digest:
            continue
        if name in irregular:
            changes.append(f"{name}: replaced by {irregular[name]}")
        else:
            changes.append(f"{name}: {'removed' if name not in current else 'changed'}")
    changes += [f"{name}: added" for name in sorted(set(current) - set(recorded))]
    changes += [
        f"{name}: added ({kind})"
        for name, kind in sorted(irregular.items())
        if name not in recorded
    ]
    if executable is not None:
        unchanged = {name for name, digest in recorded.items() if current.get(name) == digest}
        was, now = set(cast("list[str]", executable)), set(executable_files(dataset))
        changes += [
            f"{name}: changed (its executable bit was {'cleared' if name in was else 'set'})"
            for name in sorted((was ^ now) & unchanged)
        ]
    return changes


TASKS_FILE = "tasks.jsonl"
RUNS_FILE = "runs.jsonl"
EXPORT_FILE = "export.json"


@dataclass
class ExportSummary:
    out: Path
    tasks: int = 0
    """Rows of ``tasks.jsonl``."""
    runs: int = 0
    """Rows of ``runs.jsonl``."""
    programs: int = 0
    """Rows of ``runs.jsonl`` with a delivered program."""
    skipped: list[str] = field(default_factory=lambda: [])
    """What was left out, and why."""
    checked_against: int | None = None
    """Evaluation tasks the training tasks were compared with; ``None``: not checked."""
    overlapping: list[str] = field(default_factory=lambda: [])
    """Tasks left out because they overlap an evaluation task."""


def training_refusal(task: Task, allow_eval_tasks: bool) -> str | None:
    """Why ``task`` may not go into a training set, or ``None``."""
    refusal = _refusal(task, allow_eval_tasks)
    return None if refusal is None else refusal[1]


def _refusal(task: Task, allow_eval_tasks: bool) -> tuple[str, str] | None:
    """:func:`training_refusal`, with which kind of refusal it is (a key of
    :data:`_NOTHING_FIXES`)."""
    from real2simlab.benchmarks.base import benchmark_names, get_benchmark

    name = task.id.split("/", 1)[0]
    adapter = get_benchmark(name) if name in benchmark_names() else None
    refusal = None if adapter is None else adapter.training_refusal(task)
    if refusal is not None:
        return "terms", f"{task.id}: {refusal}"
    if allow_eval_tasks:
        return None
    if task.config.split is not Split.TRAIN:
        return "evaluation", (
            f"{task.id} is an evaluation task (split = {task.config.split.value}); training on "
            "it would make its results meaningless (allow_eval_tasks to export it anyway)"
        )
    if adapter is not None and Split.TRAIN.value not in adapter.splits:
        return "no training split", (
            f"{task.id} is marked split = train, but {name} releases no training split: its "
            "tasks are evaluation tasks"
        )
    return None


_GIVE_TRAINING = 'give training tasks (split = "train"; r2s train tasks makes some) or runs on them'

_NOTHING_FIXES = {
    "evaluation": _GIVE_TRAINING,
    "no training split": "give training tasks (r2s train tasks makes some): a benchmark that "
    "releases no training split has evaluation tasks only (a task of your own needs an id that "
    "does not begin with a registered benchmark's name)",
    "terms": "give other training tasks (r2s train tasks makes some): the terms of these tasks' "
    "benchmark forbid training on its data",
    "overlap": "check that --against names the evaluation tasks (not a copy of the training "
    "tasks), or give training tasks that do not overlap them",
    "infra_error": "fix what failed (r2s doctor checks this machine), run the same r2s eval or "
    "r2s run command again with the run directory as --out (it runs again what was not "
    "scored), then export",
    "undelivered": "pass --include-undelivered to export results without a program (reward 0), "
    "or give runs in which programs were delivered",
    "min_reward": "lower --min-reward, or give runs whose programs scored higher",
}
"""The fix of an export that left everything out, by why it left things out: what was given
was refused, overlaps ``--against``, was not scored, delivered nothing or scored too low."""


class _Checker:
    """Prints of the tasks to export, and the evaluation tasks they must not overlap."""

    def __init__(self, against: Sequence[Path], cache: Path | None) -> None:
        self.cache = cache
        self.index: OverlapIndex | None = None
        self.against: list[tuple[Task, TaskPrint]] = []
        """The evaluation tasks compared with, each with its print: checked again before the
        dataset is published (:meth:`changes`)."""
        if against:
            self.index = OverlapIndex()
            for root in against:
                for path in _task_dirs(root, "--against"):
                    task = load_task(path)
                    item = self.print_of(task)
                    self.index.add(item)
                    self.against.append((task, item))
        self.prints: dict[str, TaskPrint] = {}
        """Prints of the tasks to export, by content hash (two versions of one task id are
        two prints)."""
        self.included: dict[str, Task] = {}
        """The task of each print, by content hash."""
        self.found: dict[str, Overlap | None] = {}
        self.computed: dict[str, TaskPrint] = {}
        self.left_out: list[dict[str, str]] = []

    def print_of(self, task: Task) -> TaskPrint:
        """``task``'s print, its thumbnails written to the cache. A cache inside the task
        (``R2S_CACHE_DIR`` there) is refused first, wherever the task was found (given, found by
        a run, an evaluation task): its thumbnails would be written into the task, which would
        no longer be the version read (nor the one a run scored) and would be exported with
        them. A cache elsewhere, beside the tasks of a directory given, is used."""
        if self.cache is not None and _inside(self.cache, [task.root]) is not None:
            raise SetupError(
                f"the thumbnail cache {self.cache} lies inside the task {task.id} "
                f"({task.root}): the export would write into it",
                fix="set R2S_CACHE_DIR to a directory outside the task directories, then export "
                "again",
            )
        return task_print(task, cache=self.cache)

    def overlap(self, task: Task, task_hash: str) -> Overlap | None:
        """The evaluation task that ``task`` overlaps, if any."""
        if task_hash not in self.found:
            item = self.print_of(task)
            found = None if self.index is None else self.index.find(item)
            self.found[task_hash] = found
            self.computed[task_hash] = item
            if found is not None:
                self.left_out.append(
                    {"task_id": task.id, "overlaps": found.task_id, "reason": found.reason}
                )
        return self.found[task_hash]

    def include(self, task: Task, task_hash: str) -> None:
        """``task`` goes into the dataset (a task row, or a run row on it): its print goes into
        the dataset's prints. Tasks checked but left out (a reward below the threshold, no
        delivery) do not: a method trained on the dataset did not learn from them."""
        item = self.computed.get(task_hash) or self.print_of(task)
        self.prints.setdefault(task_hash, item)
        self.included.setdefault(task_hash, task)

    def changes(self) -> list[tuple[str, str]]:
        """What changed since it was read, with the fix: a task whose print is in the dataset
        that is no longer the version the print was taken from (its content hash), and an
        evaluation task whose recordings are no longer those it was compared by."""
        changed: list[tuple[str, str]] = []
        for task_hash, task in self.included.items():
            if _changed(task, task_hash):
                changed.append(
                    (
                        f"task {task.id} changed while it was exported ({task.root})",
                        "export again once the task is no longer being edited",
                    )
                )
        for task, item in self.against:
            if recordings_changed(task, item):
                changed.append(
                    (
                        f"the evaluation task {task.id} ({task.root}) changed while the dataset "
                        "was checked against it (--against): the dataset would record a "
                        "comparison with recordings that are no longer those",
                        "export again once the evaluation tasks are no longer being edited",
                    )
                )
        return changed


def _changed(task: Task, task_hash: str) -> bool:
    """Whether ``task`` is no longer the version ``task_hash`` names, also when it can no longer
    be read (removed, or given a symbolic link)."""
    try:
        return task.changed_since(task_hash)
    except (OSError, ValueError):
        return True


def _task_dirs(root: Path, option: str) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise SetupError(f"{root} ({option}) is not a directory", fix="check the path")
    found = find_tasks(root)
    if not found:
        raise SetupError(f"{root} ({option}) holds no task (no task.toml)", fix="check the path")
    return found


class _Tasks:
    """Tasks by id and content hash: those given, and those the runs name."""

    def __init__(self) -> None:
        self.by_hash: dict[tuple[str, str], Task] = {}
        self.hashes: dict[Path, str] = {}

    def hash_of(self, task: Task) -> str:
        root = task.root.resolve()
        if root not in self.hashes:
            self.hashes[root] = task.content_hash()
        return self.hashes[root]

    def add(self, task: Task) -> str:
        task_hash = self.hash_of(task)
        self.by_hash.setdefault((task.id, task_hash), task)
        return task_hash

    def lookup(self, task_id: str, task_hash: str, where: Path | None) -> Task | str:
        """The task ``task_id`` with content hash ``task_hash``: at ``where`` (where the run
        found it) or among those added; otherwise why not."""
        changed = False
        if where is not None and where.is_dir():
            try:
                task = load_task(where)
            except R2SError:
                task = None
            if task is not None and task.id == task_id:
                if self.hash_of(task) == task_hash:
                    return task
                changed = True
        if (task_id, task_hash) in self.by_hash:
            return self.by_hash[(task_id, task_hash)]
        if changed or any(known == task_id for known, _ in self.by_hash):
            return "the task has changed since the run scored it (its content hash differs)"
        return "the task is not found where the run found it, nor among the given tasks"


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _program_changed(layout: RunLayout, result: TaskResult) -> str | None:
    """Why the frozen twin in the run directory is not the one ``result`` scored, or ``None``:
    the reward of a result goes only with the program it was computed from."""
    twin = layout.build_dir(result.task_id, result.method, result.trial) / "twin"
    if result.twin_fingerprint is None:  # always recorded for a delivered result
        return (
            "its result records no fingerprint of the program it scored (the result file was "
            "edited, or written by a pre-release build of the Lab), so the program in the run "
            "directory cannot be shown to be that one"
        )
    if not twin.is_dir():
        return f"the program it scored is missing ({twin})"
    try:
        current = fingerprint(twin)
    except (OSError, ValueError) as error:
        return f"the program it scored cannot be read: {error}"
    if current != result.twin_fingerprint:
        return (
            "the program in the run directory changed after it was scored (its fingerprint "
            "differs): the reward belongs to another program"
        )
    return None


def _refuse_hidden_data(task: Task, checked: Task) -> None:
    """Raise ``SetupError`` if ``checked`` (``task``, or its copy) would show a method hidden
    data: a symbolic link, content of ``hidden/`` in ``visible/`` or as the statement
    (``instruction.md``), as ``r2s task validate`` states them."""
    from real2simlab.task.validate import hidden_data_problems, invalid_task_error

    problems = hidden_data_problems(checked)
    if problems:
        raise invalid_task_error(task, problems, then="export again")


def _copy_checked(task: Task, target: Path) -> str | None:
    """``task``'s ``visible/`` and ``hidden/`` copied to ``target``, no symbolic link followed,
    and checked there with the task's ``instruction.md`` (beside them while they are checked):
    what a method trained on the dataset sees (``visible/``, the statement) holds no hidden
    data, whatever happens to ``task`` meanwhile. Returns the statement of what was checked
    (``None`` for a task without one: a convention that needs ``instruction.md``, without
    it); the row holds it, the dataset no ``instruction.md``."""
    from real2simlab.task.io import INSTRUCTION_FILE, copy_exactly

    for source in (task.visible_dir, task.hidden_dir):
        if source == task.hidden_dir and not source.is_dir():
            continue  # a task without labels
        refused = copy_exactly(source, target / source.name)
        if refused is not None:
            raise SetupError(
                f"task {task.id} cannot be exported ({task.root}): {refused}",
                fix=f"check it with r2s task validate {task.root}, and export again once "
                "nothing edits it",
            )
    copied = Task(root=target, config=task.config)
    instruction = task.root / INSTRUCTION_FILE
    held = target / INSTRUCTION_FILE
    try:
        if instruction.is_symlink() or instruction.is_file():
            # A link is copied as a link (never followed), which the check refuses.
            shutil.copyfile(instruction, held, follow_symlinks=False)
        _refuse_hidden_data(task, copied)
        try:
            return task_statement(copied)
        except ValueError:
            return None
    finally:
        held.unlink(missing_ok=True)


def _program_changed_error(layout: RunLayout, result: TaskResult, changed: str) -> SetupError:
    return SetupError(
        f"{layout.root}: {result.method} on {result.task_id} (trial {result.trial}): {changed}",
        fix="export again once nothing writes to the run directory",
    )


def _task_changed_error(task: Task) -> SetupError:
    return SetupError(
        f"task {task.id} changed while it was exported ({task.root})",
        fix="export again once the task is no longer being edited",
    )


def _exported_task_row(task: Task, task_hash: str, out: Path) -> dict[str, Any]:
    """:func:`_task_row`, where a copy or a read that failed because the task changed
    meanwhile (a file gone, the directory replaced) says so, not that a file is missing."""
    try:
        return _task_row(task, task_hash, out)
    except (OSError, shutil.Error, ValueError) as error:
        if _changed(task, task_hash):
            raise _task_changed_error(task) from error
        raise


def _task_row(task: Task, task_hash: str, out: Path) -> dict[str, Any]:
    target = out / "tasks" / slug(task.id)
    statement = _copy_checked(task, target)
    # The copies are the version task_hash names (the one results are matched by) only if
    # nothing edited the task since it was loaded.
    if _changed(task, task_hash):
        raise SetupError(
            f"task {task.id} changed while it was exported ({task.root})",
            fix="export again once the task is no longer being edited",
        )
    config = task.config
    row: dict[str, Any] = {
        "task_id": task.id,
        "split": config.split.value,
        "benchmark": task.id.split("/", 1)[0],
        "title": config.title,
        "license": config.license,
        "redistributable": config.redistributable,
        "source": config.source.to_json(),
        "visibility": config.visibility,
        "output_format": config.delivery.output_format,
        "task_hash": task_hash,
        "metadata": config.metadata,
        "visible": _relative(target / "visible", out),
        "hidden": _relative(target / "hidden", out) if task.hidden_dir.is_dir() else None,
        "statement": statement,
    }
    if task.has_capture():
        capture = task.capture()
        row["capture"] = _relative(target / "visible" / "capture.json", out)
        view = next(iter(capture.views.values()))
        row["video"] = _relative(target / "visible" / view.rgb.path, out)
        row["fps"], row["num_frames"] = view.rgb.fps, view.rgb.num_frames
    reference = target / "hidden" / "reference"
    if reference.is_dir():
        row["reference"] = _relative(reference, out)
    parameters = target / "hidden" / PARAMETERS_FILE
    if parameters.is_file():
        row["parameters"] = _labels(task, parameters)
    generator = target / "hidden" / "generator" / "program"
    if generator.is_dir():
        row["truth_program"] = _relative(generator, out)
    return row


def _labels(task: Task, copy: Path) -> dict[str, Any]:
    """The hidden parameters of ``task`` (its ``hidden/parameters.json``), read from their
    ``copy`` in the dataset: a JSON object; ``SetupError`` naming the task's file otherwise."""
    try:
        data: Any = json.loads(copy.read_bytes())
    except OSError as error:
        problem = f"cannot be read ({error.strerror or error})"
    except ValueError as error:  # cut short, not JSON, not text
        problem = f"is not valid JSON ({error})"
    else:
        if isinstance(data, dict):
            return data
        problem = "is not a JSON object"
    path = task.hidden_dir / PARAMETERS_FILE
    raise SetupError(
        f"task {task.id} cannot be exported: its hidden parameters {path} {problem}",
        fix=f"correct or restore that file (r2s task validate {task.root} checks it; a "
        "synthetic training task can be written again with r2s train tasks ... --overwrite), "
        "then export again",
    )


def _run_names(runs: Sequence[Path]) -> list[tuple[Path, str]]:
    """Each distinct run directory and its name in the dataset: its directory name, or, when
    two runs share one, that name with a digest of the run's location."""
    unique: list[Path] = []
    for run in runs:
        resolved = Path(run).resolve()
        if resolved not in unique:
            unique.append(resolved)
    names = [path.name for path in unique]
    return [
        (
            path,
            path.name
            if names.count(path.name) == 1
            else f"{path.name}~{hashlib.sha256(str(path).encode()).hexdigest()[:8]}",
        )
        for path in unique
    ]


@dataclass
class _Export:
    out: Path
    allow_eval_tasks: bool
    min_reward: float | None
    include_undelivered: bool
    skip_unmatched: bool
    checker: _Checker
    known: _Tasks
    summary: ExportSummary
    unmatched: list[str] = field(default_factory=lambda: [])
    """Results that cannot be matched with the task they were scored on."""
    unmatched_programs: list[tuple[Path, str]] = field(default_factory=lambda: [])
    """Delivered results that cannot be matched with the program they scored, each with its run
    directory: the same command on it scores them again."""
    runs_read: list[tuple[Path, dict[str, Any]]] = field(default_factory=lambda: [])
    """Each run directory exported and its ``results.json`` as read: the result files must
    still be the ones it names when the dataset is published (:meth:`changes`)."""
    kinds: dict[str, None] = field(default_factory=lambda: {})
    """Why things were left out (keys of :data:`_NOTHING_FIXES`), in the order first seen."""
    best_left_out: float | None = None
    """The best reward ``min_reward`` left out."""

    def skip(self, what: str, reason: str, kind: str) -> None:
        self.summary.skipped.append(f"{what}: {reason}")
        self.kinds.setdefault(kind, None)

    def nothing_fix(self) -> str:
        """The fix of an export that left everything out: that of each reason it did (the
        reason of a skipped result that cannot be matched included), not always the one for
        evaluation tasks."""
        fixes = [_NOTHING_FIXES[kind] for kind in self.kinds]
        if "min_reward" in self.kinds and self.best_left_out is not None:
            fixes[list(self.kinds).index("min_reward")] = (
                f"lower --min-reward (the best reward left out is {self.best_left_out:.4f}), "
                "or give runs whose programs scored higher"
            )
        if self.unmatched or self.unmatched_programs:
            fixes.append(_unmatched_fix(self.unmatched, self.unmatched_programs))
        return "; ".join(dict.fromkeys(fixes)) or _GIVE_TRAINING

    def changes(self) -> list[tuple[str, str]]:
        """What changed since the export read it, each with its fix: the tasks and the
        evaluation tasks (:meth:`_Checker.changes`), and the results of each run (a result
        file no longer the one its row names, a run started again on the directory)."""
        changed = self.checker.changes()
        for run_dir, results in self.runs_read:
            problems = incomplete(run_dir, results=results)
            if problems:
                changed.append(
                    (
                        f"the results in {run_dir} changed while they were exported: "
                        f"{'; '.join(problems)}",
                        "export again once nothing writes to the run directory",
                    )
                )
        return changed

    def run_rows(self, run_dir: Path, name: str) -> list[dict[str, Any]]:
        """One row per counted result of the run at ``run_dir`` that may be exported."""
        layout = RunLayout(run_dir)
        # Read once: the rows checked are the rows used, and each reward is read from the
        # bytes checked against its row (a result file replaced meanwhile is never read).
        results = read_results(layout)
        problems = incomplete(run_dir, results=results)
        if problems:
            raise SetupError(
                f"the results in {run_dir} are not one complete set: {'; '.join(problems)}",
                fix="finish the run: run the same r2s eval or r2s run command again with this "
                "--out (it resumes and rewrites results.json), then export",
            )
        self.runs_read.append((Path(run_dir), results))
        digests: dict[tuple[str, str, int, int], object] = {
            (r["method"], r["task"], r["trial"], r["seed"]): r.get("result_sha256")
            for r in results.get("rows", [])
        }
        experiment = read_experiment(layout)
        where: dict[str, Path] = {}
        for task_id, path in experiment.get("tasks", {}).items():
            where[task_id] = Path(path) if Path(path).is_absolute() else layout.root / path
        out_rows: list[dict[str, Any]] = []
        for path in layout.result_files():
            result, digest = read_result_file(path)
            key = (result.method, result.task_id, result.trial, result.seed)
            if key not in digests:
                continue
            if digest != digests[key]:  # every row names one, or incomplete() refused the run
                raise SetupError(
                    f"{path} changed while {run_dir} was exported: it is no longer the result "
                    "file results.json was computed from",
                    fix="export again once nothing writes to the run directory",
                )
            label = (
                f"{name}: {result.method} on {result.task_id} "
                f"(trial {result.trial}, seed {result.seed})"
            )
            found = self.known.lookup(result.task_id, result.task_hash, where.get(result.task_id))
            if isinstance(found, str):
                self.unmatched.append(f"{label}: {found}")
                continue
            task = found
            refusal = _refusal(task, self.allow_eval_tasks)
            if refusal is not None:
                self.skip(label, refusal[1], refusal[0])
                continue
            overlap = self.checker.overlap(task, result.task_hash)
            if overlap is not None:
                self.skip(label, f"{task.id} overlaps {overlap.task_id}", "overlap")
                continue
            if result.infra_error is not None or (
                result.delivered and result.reproduction.run_status == INFRA_ERROR
            ):
                reason = result.infra_error or "the run failed outside the program (infra_error)"
                self.skip(label, f"not scored, so no reward: {reason}", "infra_error")
                continue
            if not result.delivered and not self.include_undelivered:
                self.skip(label, f"nothing delivered ({result.delivery_failure})", "undelivered")
                continue
            if result.delivered:
                changed = _program_changed(layout, result)
                if changed is not None:
                    self.unmatched_programs.append((Path(run_dir), f"{label}: {changed}"))
                    continue
            reward = reward_of(result)
            if self.min_reward is not None and reward < self.min_reward:
                self.skip(label, f"reward {reward:.4f} below {self.min_reward}", "min_reward")
                self.best_left_out = max(reward, self.best_left_out or reward)
                continue
            out_rows.append(self.run_row(layout, name, result, task, reward))
            self.checker.include(task, result.task_hash)
        return out_rows

    def run_row(
        self, layout: RunLayout, name: str, result: TaskResult, task: Task, reward: float
    ) -> dict[str, Any]:
        build = layout.build_dir(result.task_id, result.method, result.trial)
        target = (
            self.out / "runs" / name / slug(result.task_id) / slug(result.method)
            / f"trial-{result.trial}" / f"seed-{result.seed}"
        )  # fmt: skip
        if target.exists():  # names are unique per run, task, method, trial and seed
            raise SetupError(f"two results would be exported to {target}")
        row: dict[str, Any] = {
            "run": name,
            "task_id": result.task_id,
            "task_hash": result.task_hash,
            "split": task.config.split.value,
            "benchmark": result.benchmark,
            "method": result.method,
            "method_kind": result.method_kind,
            "privileged": result.privileged,
            "trial": result.trial,
            "seed": result.seed,
            "delivered": result.delivered,
            "delivery_failure": result.delivery_failure,
            "twin_fingerprint": result.twin_fingerprint,
            "run_status": result.reproduction.run_status,
            "reward": reward,
            "reproduction_skill": result.summary.reproduction_skill,
            "headline": result.headline.to_json() if result.headline else None,
            "session_consistent": result.reproduction.session_consistent,
            "program": None,
        }
        program = build / "twin" / "program"
        if result.delivered:
            if not program.is_dir():
                raise SetupError(
                    f"{layout.root}: the program {result.method} delivered for {result.task_id} "
                    f"(trial {result.trial}) is missing: {program}",
                    fix="export the run as it was written, with its builds/ directory",
                )
            try:
                shutil.copytree(program, target / "program")
            except (OSError, shutil.Error) as error:
                # a copy that failed because the program changed meanwhile (a file gone) is that
                changed = _program_changed(layout, result)
                if changed is None:
                    raise
                raise _program_changed_error(layout, result, changed) from error
            # The copy is the program that earned the reward only if nothing changed it while
            # it was copied: the source is still the scored one, and the copy is the source.
            changed = _program_changed(layout, result)
            if changed is None and fingerprint(target / "program") != fingerprint(program):
                changed = "the program changed while it was copied"
            if changed is not None:
                raise _program_changed_error(layout, result, changed)
            row["program"] = _relative(target / "program", self.out)
        # TASK.md, the agent's record and its trajectory are in directories the agent could
        # write: each is read only as a regular file (never through a link it left there, never
        # waiting on a pipe), and what the agent wrote is redacted again on its way out.
        redact = Redactor(environment_secrets())
        statement = read_regular(build / "work" / "workspace" / "TASK.md")
        row["statement"] = None if statement is None else redact(_text(statement))
        agent_record = build / "work" / "agent" / "agent_record.json"
        data = read_regular(agent_record)
        if data is not None:
            try:
                record = AgentRecord.model_validate_json(data).to_json()
            except ValidationError as error:
                raise SetupError(
                    f"{agent_record} is damaged: {'; '.join(validation_problems(error)[:3])}",
                    fix="restore the run directory from a copy, or export without this run",
                ) from None
            agent: dict[str, Any] = {
                key: record.get(key)
                for key in ("agent", "agent_version", "model", "reasoning_effort",
                            "exit_status", "usage", "cost_usd", "redactions")
            }  # fmt: skip
            # The trajectory its record names, which it names once the Lab removed the
            # injected credentials from the agent's files (not when they are still in it):
            # the agent's events, or mini-swe-agent's one JSON document (trajectory.json).
            trajectory = recorded_file(build / "work", record.get("trajectory"))
            events = read_regular(trajectory) if trajectory is not None else None
            if trajectory is not None and events is not None:
                document = trajectory.suffix == ".json"
                name = "trajectory.json" if document else "trajectory.jsonl"
                target.mkdir(parents=True, exist_ok=True)
                (target / name).write_bytes(redact(_text(events)).encode())
                agent["trajectory"] = _relative(target / name, self.out)
                agent["trajectory_format"] = f"{record.get('agent')} " + (
                    "trajectory (JSON), redacted" if document else "events, redacted"
                )
            row["agent"] = agent
        return row


def _text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _card(summary: ExportSummary, tasks: list[dict[str, Any]], runs: list[dict[str, Any]],
          command: str | None) -> str:  # fmt: skip
    licenses = sorted({row["license"] for row in tasks})
    splits = sorted({row["split"] for row in tasks} | {row["split"] for row in runs})
    restricted = sorted({row["task_id"] for row in tasks if not row["redistributable"]})
    benchmarks = sorted({row["benchmark"] for row in tasks} | {row["benchmark"] for row in runs})
    lines = [
        "# Real2SimLab training data",
        "",
        f"Exported by Real2SimLab {__version__} on {datetime.now(UTC).date().isoformat()}"
        + (f" with `{command}`." if command else "."),
        "",
        f"- {summary.tasks} tasks in `{TASKS_FILE}` (benchmarks: {', '.join(benchmarks) or '—'})",
        f"- {summary.runs} results of runs in `{RUNS_FILE}`, {summary.programs} of them with "
        "a delivered program",
        f"- splits: {', '.join(splits) or '—'}",
        f"- licenses: {', '.join(licenses) or '—'}",
        f"- left out: {len(summary.skipped)} (each with its reason in `{EXPORT_FILE}`)",
        "",
        "Each task row points to `visible/` (what a method sees: the capture and its media) and",
        "`hidden/` (the labels: the reference output, `parameters.json` with the hidden",
        "parameters and initial state, and the truth program, when the task has them). Each run",
        "row holds one result: its program (when one was delivered), its reward (the",
        "evaluation's own score, 0 to 1), the task's content hash it was scored on and, for",
        "agents, their redacted trajectory.",
        "",
    ]
    if summary.checked_against is None:
        lines += [
            "The tasks were **not** compared with evaluation tasks (`--against`): nothing shows",
            "that none of them holds an evaluation task's recording.",
            "",
        ]
    else:
        lines += [
            f"The tasks were compared with {summary.checked_against} evaluation tasks: "
            f"{len(summary.overlapping)} that overlap",
            f"one of them (the same recording) were left out (`{EXPORT_FILE}`).",
            "",
        ]
    if "eval" in splits:
        lines += [
            "**This dataset contains evaluation tasks.** A model trained on it can no longer be",
            "evaluated on them: report results on these tasks as training results.",
            "",
        ]
    else:
        lines += [
            "It holds training tasks only (`split = train`): the evaluation tasks are not in it.",
            "",
        ]
    if restricted:
        lines += [
            f"{len(restricted)} tasks may not be redistributed (their `redistributable` is",
            "false): keep this dataset to yourself.",
            "",
        ]
    return "\n".join(lines)


def _inside(directory: Path, inputs: Sequence[Path]) -> Path | None:
    """The first of ``inputs`` that ``directory`` (where the export writes besides ``out``) is
    or lies inside, if any."""
    target = Path(directory).resolve()
    for given in inputs:
        if target.is_relative_to(Path(given).resolve()):
            return Path(given)
    return None


def _listed(lines: Sequence[str]) -> str:
    more = len(lines) - 3
    return "; ".join(lines[:3]) + (f" (and {more} more)" if more > 0 else "")


def _unmatched_fix(tasks: list[str], programs: list[tuple[Path, str]]) -> str:
    """What matches results with what they were scored on again: a task is given as it was when
    the run scored it; a program is scored again by the same command on its run directory
    (which builds again what changed)."""
    runs = list(dict.fromkeys(str(run) for run, _ in programs))
    again = (
        "run the same r2s eval or r2s run command again with the run directory as --out "
        f"({', '.join(runs[:3])}{', ...' if len(runs) > 3 else ''})"
    )
    given = "give the tasks as they were when the runs scored them (--tasks DIR)"
    if not programs:
        return given
    if not tasks:
        return f"score them again: {again}"
    return f"{given}, and score again the results whose program cannot be matched: {again}"


def _unmatched_error(tasks: list[str], programs: list[tuple[Path, str]]) -> SetupError:
    """The error for results that cannot be matched with what they were scored on, with the fix
    of each cause: a task is given as it was when the run scored it; a program is scored again
    by the same command on its run directory (which builds again what changed)."""
    problems: list[str] = []
    if tasks:
        problems.append(
            f"{len(tasks)} result(s) cannot be matched with the task they were scored on: "
            + _listed(tasks)
        )
    if programs:
        problems.append(
            f"{len(programs)} result(s) cannot be matched with the program they scored: "
            + _listed([line for _, line in programs])
        )
    return SetupError(
        "; ".join(problems),
        fix=f"{_unmatched_fix(tasks, programs)}, or pass --skip-unmatched to leave these results "
        "out",
    )


def export_dataset(
    out: Path,
    *,
    tasks: Sequence[Path] = (),
    runs: Sequence[Path] = (),
    against: Sequence[Path] = (),
    allow_eval_tasks: bool = False,
    min_reward: float | None = None,
    include_undelivered: bool = False,
    skip_unmatched: bool = False,
    overwrite: bool = False,
    command: str | None = None,
    cache: Path | None = None,
) -> ExportSummary:
    """Export the tasks under ``tasks`` and the counted results of the runs in ``runs`` (run
    directories) to ``out``, leaving out those that overlap a task under ``against``; see the
    module docstring for the layout and the rules. Video thumbnails are kept in ``cache``
    (default: the Lab's cache)."""
    out = Path(out)
    clash = clashing_input(out, [*tasks, *runs, *against])
    if clash is not None:
        raise SetupError(
            f"the dataset directory {out} overlaps an input, {clash}",
            fix="write the dataset to a directory outside the task and run directories",
        )
    check_replaceable(out, overwrite=overwrite, marker=EXPORT_FILE, what="an exported dataset")
    cache = default_cache() if cache is None else Path(cache)  # never inside a task read
    if not tasks and not runs:
        raise SetupError("nothing to export", fix="give task directories, run directories or both")
    for run in runs:
        if not (Path(run) / "results.json").is_file():
            raise SetupError(
                f"{run} is not a finished run (it has no results.json)",
                fix="give the directory of a finished run (r2s eval or r2s run --out)",
            )
    task_dirs = [path for root in tasks for path in _task_dirs(root, "--tasks")]
    checker = _Checker(against, cache)
    summary = ExportSummary(out=out)
    if checker.index is not None:
        summary.checked_against = len(checker.index)
    known = _Tasks()
    given: dict[str, tuple[Task, str]] = {}
    for path in task_dirs:
        task = load_task(path)
        try:
            task_hash = known.add(task)
        except SymbolicLinkError as error:  # which no task holds (fingerprints refuse it)
            _refuse_hidden_data(task, task)
            raise SetupError(
                f"task {task.id} changed while it was read ({task.root}): {error}",
                fix="export again once nothing edits the task",
            ) from None
        if task.id in given and given[task.id][1] != task_hash:
            raise SetupError(
                f"two different tasks are both {task.id}: {given[task.id][0].root} and {path}",
                fix="give one version of each task",
            )
        given.setdefault(task.id, (task, task_hash))
    staging = staging_dir(out)
    try:
        export = _Export(
            staging, allow_eval_tasks, min_reward, include_undelivered, skip_unmatched,
            checker, known, summary,
        )  # fmt: skip
        task_rows: list[dict[str, Any]] = []
        for task, task_hash in given.values():
            refusal = _refusal(task, allow_eval_tasks)
            if refusal is not None:
                export.skip(task.id, refusal[1], refusal[0])
                continue
            overlap = checker.overlap(task, task_hash)
            if overlap is not None:
                export.skip(task.id, f"overlaps {overlap.task_id}: {overlap.reason}", "overlap")
                continue
            task_rows.append(_exported_task_row(task, task_hash, staging))
            checker.include(task, task_hash)
        run_rows: list[dict[str, Any]] = []
        for run_dir, name in _run_names(runs):
            run_rows += export.run_rows(run_dir, name)
        unmatched = [*export.unmatched, *(line for _, line in export.unmatched_programs)]
        if unmatched and not skip_unmatched:
            raise _unmatched_error(export.unmatched, export.unmatched_programs)
        for line in unmatched:
            summary.skipped.append(line)
        summary.overlapping = [entry["task_id"] for entry in checker.left_out]
        summary.tasks, summary.runs = len(task_rows), len(run_rows)
        summary.programs = sum(1 for row in run_rows if row["program"] is not None)
        if not task_rows and not run_rows:
            reasons = "; ".join(summary.skipped[:3]) or "nothing matched"
            raise SetupError(f"nothing could be exported: {reasons}", fix=export.nothing_fix())
        with (staging / TASKS_FILE).open("w", encoding="utf-8") as stream:
            for row in task_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if runs:
            with (staging / RUNS_FILE).open("w", encoding="utf-8") as stream:
                for row in run_rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        OverlapIndex(checker.prints.values()).save(staging / DATASET_PRINTS)
        report = {
            "lab_version": __version__,
            "command": command,
            "tasks": [str(Path(path)) for path in tasks],
            "runs": [str(path) for path, _ in _run_names(runs)],
            "against": [str(Path(path)) for path in against],
            "checked_against": summary.checked_against,
            "against_digest": None if checker.index is None else checker.index.digest(),
            "counts": {"tasks": summary.tasks, "runs": summary.runs, "programs": summary.programs},
            "overlapping": checker.left_out,
            "skipped": summary.skipped,
            "content": content_digests(staging),
            "executable": executable_files(staging),
        }
        (staging / EXPORT_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (staging / "README.md").write_text(
            _card(summary, task_rows, run_rows, command), encoding="utf-8"
        )
        # Every input checked again, last: the dataset is published only if none changed while
        # it was read (an editor, a sync, another command on a run directory), so it never
        # pairs two versions of an input or records a comparison with what is no longer there.
        changed = export.changes()
        if changed:
            message, fix = changed[0]
            more = f" (and {len(changed) - 1} more input(s) changed)" if len(changed) > 1 else ""
            raise SetupError(message + more, fix=fix)
        replace_dir(
            staging, out, overwrite=overwrite, marker=EXPORT_FILE, what="an exported dataset"
        )
    except BaseException:
        force_remove(staging)
        raise
    return summary
