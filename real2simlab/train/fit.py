"""Train a method on an exported dataset: ``r2s train fit``.

The method's ``train`` writes into a fresh directory next to the checkpoint's; the Lab then adds
the record (``training.json``) and the dataset's overlap prints, and moves the directory into
place, so a checkpoint is either complete or absent. An existing checkpoint (``overwrite``) is
replaced only when the new one is complete: a failed training keeps it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from real2simlab import __version__
from real2simlab._fs import (
    check_replaceable,
    clashing_input,
    force_remove,
    recover_dir,
    replace_dir,
    staging_dir,
)
from real2simlab.errors import SetupError
from real2simlab.methods.base import Method
from real2simlab.methods.training import (
    DATASET_PRINTS,
    PRINTS_FILE,
    TRAINING_FILE,
    TrainContext,
    TrainingRecord,
    checkpoint_digest,
    content_digest,
)
from real2simlab.orchestrate.layout import slug
from real2simlab.task.overlap import OverlapIndex, recording_files
from real2simlab.train.dataset import EXPORT_FILE, RUNS_FILE, TASKS_FILE, dataset_changes
from real2simlab.twin.fingerprint import is_os_litter, sha256_file


def dataset_digest(dataset: Path) -> str:
    """SHA-256 of a dataset's ``tasks.jsonl`` and ``runs.jsonl``."""
    digest = hashlib.sha256()
    for name in (TASKS_FILE, RUNS_FILE):
        path = dataset / name
        digest.update(name.encode())
        digest.update(path.read_bytes() if path.is_file() else b"")
    return digest.hexdigest()


def _version(dataset: Path) -> tuple[int, int, str]:
    """Which version of the dataset directory is at ``dataset``: the directory itself (an
    export replaces it as a whole) and its ``export.json``."""
    info = os.stat(dataset)
    record = dataset / EXPORT_FILE
    marker = hashlib.sha256(record.read_bytes()).hexdigest() if record.is_file() else ""
    return info.st_dev, info.st_ino, marker


def _replaced(
    dataset: Path, version: tuple[int, int, str], digest: str, *, trainer: str | None
) -> SetupError | None:
    """The error for a dataset that is no longer the version ``version`` and ``digest`` name
    (replaced as a whole, by an export with --overwrite to the same directory, or removed), or
    ``None``. With ``trainer`` (the method, once it trained), every file is also checked
    against ``export.json``: a file changed since the checks before training changed while it
    trained. A dataset that can no longer be read is no longer that version either."""
    changes: list[str] = []
    try:
        same = _version(dataset) == version and dataset_digest(dataset) == digest
        if same and trainer is not None:
            changes = dataset_changes(dataset)
    except OSError as error:
        same, changes = False, [f"it can no longer be read: {error}"]
    if same and not changes:
        return None
    if trainer is None:
        return SetupError(
            f"the dataset {dataset} was replaced while r2s train fit read it (an r2s train "
            "export to the same directory?)" + (f" ({changes[0]})" if changes else ""),
            fix="train again once the dataset is no longer being written",
        )
    return SetupError(
        f"the dataset {dataset} was replaced or changed while {trainer} trained on it (an r2s "
        "train export to the same directory, an edited file?): the model may have learned "
        "from files its held-out tasks do not describe" + (f" ({changes[0]})" if changes else ""),
        fix="train again once the dataset is no longer being written; if a file was edited, "
        + EXPORT_AGAIN,
    )


def _task_ids(
    dataset: Path, names: tuple[str, ...] = (TASKS_FILE, RUNS_FILE)
) -> tuple[list[str], int]:
    ids: list[str] = []
    runs = 0
    for name in names:
        path = dataset / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:  # damaged (a disk, an editor): named, not decoded
            number = data[: error.start].count(b"\n") + 1
            raise SetupError(
                f"{path}, line {number}: not UTF-8 text ({error.reason})", fix=EXPORT_AGAIN
            ) from None
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                row, problem = None, f"not JSON ({error})"
            else:
                problem = "a row without a task_id"
            task_id = row.get("task_id") if isinstance(row, dict) else None
            if not isinstance(task_id, str) or not task_id:
                raise SetupError(
                    f"{path}, line {number}: {problem}",
                    fix=EXPORT_AGAIN,
                )
            ids.append(task_id)
            runs += name == RUNS_FILE
    return sorted(set(ids)), runs


EXPORT_AGAIN = "export the dataset again: r2s train export ... --out DATASET --overwrite"


def _check_unchanged(dataset: Path) -> None:
    """Raise ``SetupError`` unless the dataset holds exactly the files its export wrote: its
    overlap prints, which keep a trained method off its training tasks, describe only those."""
    changes = dataset_changes(dataset)
    if changes:
        more = f" (and {len(changes) - 5} more)" if len(changes) > 5 else ""
        raise SetupError(
            f"the dataset {dataset} is not as r2s train export wrote it: "
            f"{'; '.join(changes[:5])}{more}",
            fix=EXPORT_AGAIN,
        )


PRINTS_FIX = "export the dataset again with r2s train export (it writes the prints)"


def _check_prints(dataset: Path, task_ids: list[str]) -> None:
    """Raise ``SetupError`` unless the dataset's overlap prints cover all its tasks: they are
    what keeps a trained method from being evaluated on its training tasks."""
    path = dataset / DATASET_PRINTS
    if not path.is_file():
        raise SetupError(
            f"{dataset} has no {DATASET_PRINTS}, so the tasks a method trained on it learned "
            "from could not be held out of evaluations",
            fix=PRINTS_FIX,
        )
    printed = {item.task_id for item in OverlapIndex.load(path, fix=PRINTS_FIX).prints}
    missing = sorted(set(task_ids) - printed)
    if missing:
        raise SetupError(
            f"{DATASET_PRINTS} of {dataset} has no print of {len(missing)} of its tasks "
            f"({', '.join(missing[:3])}{', ...' if len(missing) > 3 else ''})",
            fix=PRINTS_FIX,
        )


def _check_printed_recordings(dataset: Path) -> None:
    """Raise ``SetupError`` unless the prints of each task the dataset holds hold every
    recording of it, by content: a recording replaced since the export, in a dataset whose
    ``content`` record was rewritten to match, would be learned from but never held out."""
    held: dict[str, set[str]] = {}
    for item in OverlapIndex.load(dataset / DATASET_PRINTS, fix=PRINTS_FIX).prints:
        held.setdefault(item.task_id, set()).update(item.files)
    exported, _ = _task_ids(dataset, names=(TASKS_FILE,))
    for task_id in exported:
        for recording in recording_files(dataset / "tasks" / slug(task_id)):
            if is_os_litter(recording.name):
                continue
            if sha256_file(recording) not in held.get(task_id, set()):
                raise SetupError(
                    f"{DATASET_PRINTS} of {dataset} does not describe the recordings of its "
                    f"tasks: no print holds {recording.relative_to(dataset).as_posix()} (of "
                    f"{task_id}), so a method trained on it could be evaluated on that recording",
                    fix=PRINTS_FIX,
                )


def fit_method(
    method: Method,
    dataset: Path,
    out: Path,
    *,
    seed: int = 0,
    overwrite: bool = False,
    log: Callable[[str], None] = print,
) -> TrainingRecord:
    """Train ``method`` on the exported ``dataset`` and write its checkpoint to ``out``."""
    if not method.trainable:
        raise SetupError(
            f"method {method.name!r} does not learn from data",
            fix="choose a method that learns (its class sets trainable = True)",
        )
    if method.checkpoint is not None:
        raise SetupError(
            f"{method.name} was given a checkpoint ({method.checkpoint}) to train",
            fix="leave checkpoint out of the options: training writes a new one",
        )
    dataset = Path(dataset)
    recover_dir(dataset, marker=EXPORT_FILE, what="the dataset")
    if not (dataset / TASKS_FILE).is_file():
        raise SetupError(
            f"{dataset} is not an exported dataset: it has no {TASKS_FILE}",
            fix="export one first: r2s train export --tasks TRAIN_TASKS --out DATASET",
        )
    # What is trained on is this version of the dataset: it is taken before the dataset is
    # checked and read (the record and the prints), and the dataset must still be this version
    # when training ends (an export with --overwrite to the same directory in the meantime
    # swaps in another one). A check or a read that fails because it was swapped meanwhile
    # says so, not that a file is missing.
    version = _version(dataset)
    digest = dataset_digest(dataset)
    try:
        task_ids, runs = _task_ids(dataset)
        _check_prints(dataset, task_ids)
        _check_unchanged(dataset)
        _check_printed_recordings(dataset)
        rows = (dataset / TASKS_FILE).read_text(encoding="utf-8").splitlines()
        prints = (dataset / DATASET_PRINTS).read_bytes()
    except Exception as error:
        replaced = _replaced(dataset, version, digest, trainer=None)
        if replaced is not None:
            raise replaced from error
        raise
    out = Path(out)
    clash = clashing_input(out, [dataset])
    if clash is not None:
        raise SetupError(
            f"the checkpoint directory {out} overlaps the dataset {clash}",
            fix="write the checkpoint to a directory outside the dataset",
        )
    check_replaceable(out, overwrite=overwrite, marker=TRAINING_FILE, what="a checkpoint")
    staging = staging_dir(out)
    started, clock = datetime.now(UTC), time.monotonic()
    try:
        try:
            method.train(TrainContext(dataset=dataset.resolve(), checkpoint_dir=staging,
                                      seed=seed, log=log))  # fmt: skip
        except Exception as error:
            # A file the method could not read (gone, half-written) in a dataset replaced or
            # changed meanwhile is that, whatever the method raised.
            replaced = _replaced(dataset, version, digest, trainer=method.name)
            if replaced is not None:
                raise replaced from error
            raise
        for name in (TRAINING_FILE, PRINTS_FILE):
            if (staging / name).exists():
                raise SetupError(
                    f"{method.name} wrote {name}, which the Lab writes",
                    fix=f"rename the method's {name}",
                )
        checkpoint_digest(staging)  # what the method wrote can be fingerprinted (no links)
        # Every file the rows point to, as before training: a recording replaced meanwhile
        # (an editor, a sync) was learned from, but the prints describe the one it replaced.
        replaced = _replaced(dataset, version, digest, trainer=method.name)
        if replaced is not None:
            raise replaced
        (staging / PRINTS_FILE).write_bytes(prints)
        record = TrainingRecord(
            content_digest=content_digest(staging),
            method=method.name,
            method_version=method.method_version,
            options=method.options.model_dump(mode="json"),
            dataset=str(dataset),
            dataset_digest=digest,
            tasks=sum(1 for line in rows if line),
            runs=runs,
            task_ids=tuple(task_ids),
            prints=PRINTS_FILE,
            seed=seed,
            started_at=started,
            duration_s=round(time.monotonic() - clock, 3),
            lab_version=__version__,
        )
        (staging / TRAINING_FILE).write_text(record.dumps() + "\n", encoding="utf-8")
        replace_dir(staging, out, overwrite=overwrite, marker=TRAINING_FILE, what="a checkpoint")
    except BaseException:
        force_remove(staging)
        raise
    return record
