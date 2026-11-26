"""The fixed layout of a run directory, which makes experiments resumable.

::

    <run>/
      experiment.json                      the resolved configuration
      tasks/<benchmark>/<selection>/       tasks built from benchmarks, one directory per selection
      builds/<task>/<method>/trial-<k>/    build_record.json, twin/, agent/, workspace/, work/
      runs/<task>/<method>/trial-<k>/seed-<s>/<label>/   one twin run each: run, rerun, native
      results/<task>/<method>/trial-<k>/seed-<s>.json        TaskResult
      results.json                         the aggregate
      report/                              index.html, report.md, results.csv
      state/                               budget accounting; status.json (running or complete)
      r2s.log

Every step directory carries a ``.key`` file: the content hash of its inputs. A step whose key
matches is reused on resume; any other is redone from scratch.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from real2simlab._base import validation_problems
from real2simlab._fs import force_remove
from real2simlab.errors import InfrastructureError, SetupError
from real2simlab.scoring.model import Results, TaskResult

KEY_FILE = ".key"
STATUS_FILE = "status.json"
RUNNING, COMPLETE = "running", "complete"
"""``state/status.json``: ``running`` from the start of a run until it has written every result
and ``results.json``; then ``complete``. A run directory left ``running`` was interrupted (or
is still being written): its result files may be a mix of two runs."""


MAX_SLUG = 120
"""The longest directory name made from a task id, in bytes: file systems allow 255, and
names built on it (``.partial-...``, ``.trash-...``) must fit too."""


def slug(task_id: str) -> str:
    """A task id as one directory name, distinct for distinct ids.

    ``/`` becomes ``__``. Without ``__`` in the id this is one-to-one (segments start with a
    letter or digit, so a run of underscores tells separators from the id's own). An id that
    contains ``__`` itself (``a/b__c`` and ``a__b/c`` would meet) also gets a digest of the id,
    and so does one whose name would be longer than ``MAX_SLUG`` bytes, cut to fit.
    """
    name = task_id.replace("/", "__")
    if "__" in task_id:
        name += "~" + hashlib.sha256(task_id.encode()).hexdigest()[:12]
    if len(name.encode()) > MAX_SLUG:  # a file name has at most 255 bytes
        name = name[: MAX_SLUG - 17] + "~" + hashlib.sha256(task_id.encode()).hexdigest()[:16]
    return name


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @property
    def experiment_file(self) -> Path:
        return self.root / "experiment.json"

    @property
    def tasks_dir(self) -> Path:
        return self.root / "tasks"

    @property
    def results_file(self) -> Path:
        return self.root / "results.json"

    @property
    def report_dir(self) -> Path:
        return self.root / "report"

    @property
    def state_dir(self) -> Path:
        return self.root / "state"

    @property
    def log_file(self) -> Path:
        return self.root / "r2s.log"

    @property
    def status_file(self) -> Path:
        return self.state_dir / STATUS_FILE

    def build_dir(self, task_id: str, method: str, trial: int) -> Path:
        return self.root / "builds" / slug(task_id) / method / f"trial-{trial}"

    def run_dir(self, task_id: str, method: str, trial: int, seed: int, label: str) -> Path:
        return (
            self.root / "runs" / slug(task_id) / method / f"trial-{trial}" / f"seed-{seed}" / label
        )

    def result_file(self, task_id: str, method: str, trial: int, seed: int) -> Path:
        return (
            self.root / "results" / slug(task_id) / method / f"trial-{trial}" / f"seed-{seed}.json"
        )

    def result_files(self) -> list[Path]:
        return sorted((self.root / "results").rglob("seed-*.json"))


def read_key(directory: Path) -> str | None:
    """The key in ``directory``'s ``.key``; ``None`` when it has none, or none that can be read
    (bytes that are not text, a file this process may not read: damaged), so the step is done
    again, as for any key that does not match."""
    path = Path(directory) / KEY_FILE
    try:
        return path.read_text(encoding="utf-8").strip() if path.is_file() else None
    except (OSError, ValueError):
        return None


def write_key(directory: Path, key: str) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / KEY_FILE).write_text(key + "\n", encoding="utf-8")


def fresh(directory: Path) -> Path:
    """Remove ``directory`` if it exists; return it (not created).

    Run directories hold what sandboxed programs wrote (unreadable directories, very deep
    trees), so removal restores access first and does not recurse in Python.
    """
    problem = force_remove(Path(directory))
    if problem:
        raise InfrastructureError(problem)
    return Path(directory)


def write_json(path: Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    except BaseException:  # a full disk, Ctrl-C: no half-written file is left beside it
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


RESULTS_FIX = (
    "run the same r2s eval or r2s run command again with this --out: it rewrites results.json "
    "from the result files"
)
ROW_FIELDS: dict[str, type] = {"task": str, "method": str, "trial": int, "seed": int}
"""The fields of a ``results.json`` row that name its result file."""


def read_results(layout: RunLayout) -> dict[str, Any]:
    """The run's ``results.json``, checked against its model (``r2s-results/1.0``), with every
    row naming its result (``task``, ``method``, ``trial``, ``seed``).

    Raises:
        SetupError: the file is missing, unreadable or damaged; the message names it and says
            how to rewrite it.
    """
    path = layout.results_file
    try:
        data = json.loads(path.read_bytes())
    except FileNotFoundError:
        raise SetupError(
            f"{layout.root} has no results.json, so it is not a finished run directory",
            fix="give the --out directory of r2s eval or r2s run (run the experiment first)",
        ) from None
    except OSError as error:
        raise SetupError(f"cannot read {path}: {error.strerror}", fix=RESULTS_FIX) from None
    except ValueError as error:
        raise SetupError(f"{path} is not valid JSON ({error})", fix=RESULTS_FIX) from None
    try:
        Results.model_validate(data)
    except ValidationError as error:
        problems = validation_problems(error)
        raise SetupError(
            f"{path} is damaged: {'; '.join(problems[:3])}" + (" ..." if len(problems) > 3 else ""),
            fix=RESULTS_FIX,
        ) from None
    for index, row in enumerate(data.get("rows", [])):
        for name, kind in ROW_FIELDS.items():
            if not isinstance(row.get(name), kind):
                raise SetupError(
                    f"{path} is damaged: row {index} has no {kind.__name__} {name!r}",
                    fix=RESULTS_FIX,
                )
    return data


def read_experiment(layout: RunLayout) -> dict[str, Any]:
    """The run's ``experiment.json`` (``{}`` when it has none), checked: an object with the
    experiment's ``config``, the ``lab_version``, and ``tasks`` mapping each task id to its
    directory (none in a run stopped before it resolved its tasks).

    Raises:
        SetupError: the file is unreadable or damaged; the message names it and says how to
            rewrite it.
    """
    path = layout.experiment_file
    fix = (
        "run the same r2s eval or r2s run command again with this --out: it rewrites "
        "experiment.json"
    )
    try:
        data = json.loads(path.read_bytes())
    except FileNotFoundError:
        return {}
    except OSError as error:
        raise SetupError(f"cannot read {path}: {error.strerror}", fix=fix) from None
    except ValueError as error:
        raise SetupError(f"{path} is not valid JSON ({error})", fix=fix) from None
    tasks = data.get("tasks", {}) if isinstance(data, dict) else None
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("config"), dict)
        or not isinstance(data.get("lab_version"), str)
        or not isinstance(tasks, dict)
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in tasks.items())
    ):
        raise SetupError(
            f"{path} is damaged: it is not an object with the experiment's 'config', the "
            "'lab_version' and 'tasks' mapping task ids to directories",
            fix=fix,
        )
    return data


def read_result(path: Path) -> TaskResult:
    """One result file (``results/.../seed-<s>.json``), checked against ``r2s-task-result/1.0``.

    Raises:
        SetupError: the file is unreadable or damaged; the message names it and says how to
            score the step again.
    """
    return read_result_file(path)[0]


def read_result_file(path: Path) -> tuple[TaskResult, str]:
    """One result file, as :func:`read_result`, and the SHA-256 of the bytes it was read from
    (what a row of ``results.json`` names): the result checked is the result used."""
    fix = (
        "delete it and run the same r2s eval or r2s run command again with this --out: the "
        "step is scored again"
    )
    try:
        data = Path(path).read_bytes()
        return TaskResult.model_validate_json(data), hashlib.sha256(data).hexdigest()
    except OSError as error:
        raise SetupError(f"cannot read {path}: {error.strerror}", fix=fix) from None
    except ValidationError as error:
        problems = validation_problems(error)
        raise SetupError(
            f"{path} is damaged: {'; '.join(problems[:3])}" + (" ..." if len(problems) > 3 else ""),
            fix=fix,
        ) from None


def incomplete(run_dir: Path, *, results: dict[str, Any] | None = None) -> list[str]:
    """Why the results of the run directory ``run_dir`` are not one complete set, or ``[]``.

    The run must have finished (``state/status.json`` says ``complete``), and every row of
    ``results.json`` must have its result file, unchanged since the row was computed (its
    ``result_sha256``). This catches a run interrupted while it rewrote its results, results
    deleted or replaced by hand, and archives that lost files. The status and the digests are
    always written: a run directory without them (edited or partly copied, or written by a
    pre-release build of the Lab) cannot be checked, so it is not one complete set either.
    ``results``: its ``results.json`` as already read (:func:`read_results`), so that the rows
    checked are the rows the caller uses.
    """
    layout = RunLayout(Path(run_dir))
    problems: list[str] = []
    damaged: str | None = None
    if results is None and layout.results_file.is_file():
        try:
            results = read_results(layout)
        except SetupError as error:
            damaged = str(error).split("\n", 1)[0]
    if layout.status_file.is_file():
        try:
            status = json.loads(layout.status_file.read_text(encoding="utf-8")).get("status")
        except (OSError, ValueError, AttributeError):
            status = None
        if status != COMPLETE:
            problems.append(
                f"the run did not finish (state/{STATUS_FILE} says {status!r}): it was "
                "interrupted or is still running, so its results may mix two runs"
            )
    elif results is not None:
        problems.append(
            f"it has no state/{STATUS_FILE}, which every run writes (the run directory was "
            "edited or partly copied, or written by a pre-release build of the Lab), so whether "
            "the run finished is unknown"
        )
    if damaged is not None:
        return [*problems, damaged]
    if results is None:
        return [*problems, "it has no results.json"]
    missing: list[str] = []
    changed: list[str] = []
    unchecked: list[str] = []
    for row in results.get("rows", []):
        path = layout.result_file(row["task"], row["method"], row["trial"], row["seed"])
        label = f"{row['method']} on {row['task']} (trial {row['trial']}, seed {row['seed']})"
        if not path.is_file():
            missing.append(label)
            continue
        expected = row.get("result_sha256")
        if not expected:
            unchecked.append(label)
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(label)
    if missing:
        problems.append(
            f"{len(missing)} result file(s) of results.json are missing: {', '.join(missing[:3])}"
            + (" ..." if len(missing) > 3 else "")
        )
    if changed:
        problems.append(
            f"{len(changed)} result file(s) changed since results.json was written: "
            f"{', '.join(changed[:3])}" + (" ..." if len(changed) > 3 else "")
        )
    if unchecked:
        problems.append(
            f"{len(unchecked)} row(s) of results.json name no SHA-256 of their result file "
            "(result_sha256, which every row holds: results.json was edited, or written by a "
            "pre-release build of the Lab), so the file cannot be shown to be the one scored: "
            f"{', '.join(unchecked[:3])}" + (" ..." if len(unchecked) > 3 else "")
        )
    return problems
