"""Training a method: what it learns from, where it writes what it learned, and the record the
Lab keeps of it.

``r2s train fit METHOD --data DATASET --out CHECKPOINT`` gives a trainable method (``trainable =
True``) a ``TrainContext``: an exported dataset (``r2s train export``) and an empty checkpoint
directory. The method writes what it learned there. The Lab adds ``training.json``
(``r2s-training/1.0``: the method, its options, the dataset and its digest) and copies the
dataset's overlap prints (``training-prints.npz``). The method is then evaluated like any
other, with ``checkpoint=CHECKPOINT`` among its options; an experiment does not run it on a
task that overlaps one of its training tasks (:mod:`real2simlab.task.overlap`), and says so.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Annotated

from pydantic import AwareDatetime, Field, JsonValue, NonNegativeInt, ValidationError

from real2simlab._base import StrictModel
from real2simlab.errors import SetupError
from real2simlab.spec import SpecId, spec_validator
from real2simlab.task.overlap import OverlapIndex
from real2simlab.twin.fingerprint import fingerprint

TRAINING_FILE = "training.json"
"""The Lab's record in a checkpoint directory."""
PRINTS_FILE = "training-prints.npz"
"""The overlap prints of the training tasks, in a checkpoint directory."""
DATASET_PRINTS = "overlap-prints.npz"
"""The overlap prints of the tasks of an exported dataset."""


@dataclass(frozen=True)
class TrainContext:
    """Everything a method may use to learn."""

    dataset: Path
    """An exported dataset (``tasks.jsonl``, ``runs.jsonl``, ``tasks/``, ``runs/``); read only."""
    checkpoint_dir: Path
    """An empty directory for what the method learns; the Lab adds its record after."""
    seed: int = 0
    log: Callable[[str], None] = field(default=lambda message: None)


class TrainingRecord(StrictModel):
    """What a checkpoint was learned from (``training.json``)."""

    spec: Annotated[str, spec_validator(SpecId.TRAINING)] = SpecId.TRAINING.value
    method: str
    """The registered method name."""
    method_version: str
    options: dict[str, JsonValue] = Field(default_factory=dict)
    """The method's options while training."""
    dataset: str
    """The dataset directory, as given."""
    dataset_digest: str
    """SHA-256 of the dataset's ``tasks.jsonl`` and ``runs.jsonl``."""
    tasks: NonNegativeInt
    runs: NonNegativeInt
    task_ids: tuple[str, ...] = ()
    """The training tasks (of ``tasks.jsonl`` and of the runs)."""
    prints: str = PRINTS_FILE
    """The overlap prints of the training tasks (``training-prints.npz``, in the checkpoint):
    an evaluation with the checkpoint skips every task they overlap."""
    seed: int = 0
    started_at: AwareDatetime
    duration_s: float
    lab_version: str
    content_digest: str | None = None
    """SHA-256 of the checkpoint's files as training wrote them (every file but this record,
    :func:`content_digest`): a checkpoint changed since is refused, not used. ``r2s train fit``
    always writes it: a checkpoint without it (its record edited, or written by a pre-release
    build of the Lab) is refused too."""


def content_digest(checkpoint: Path) -> str:
    """The content fingerprint of a checkpoint's files, its record (``training.json``) aside."""
    try:
        return fingerprint(Path(checkpoint), exclude=(TRAINING_FILE,))
    except (OSError, ValueError) as error:
        raise SetupError(
            f"cannot read the checkpoint {checkpoint}: {error}",
            fix="a checkpoint holds regular files only (no symbolic links); train it again",
        ) from error


def _train_again(checkpoint: Path, record: TrainingRecord) -> str:
    return (
        f"restore it, or train it again: r2s train fit {record.method} --data DATASET "
        f"--out {checkpoint} --overwrite"
    )


def check_content(checkpoint: Path, record: TrainingRecord) -> None:
    """Raise ``SetupError`` if the checkpoint's files are not the ones training wrote, or its
    record has no digest of them to show it."""
    if record.content_digest is None:
        raise SetupError(
            f"the checkpoint {checkpoint} records no content_digest in its {TRAINING_FILE} "
            "(r2s train fit writes one in every checkpoint): it was edited, or written by a "
            "pre-release build of the Lab, so its files cannot be shown to be what was trained",
            fix=_train_again(checkpoint, record),
        )
    if content_digest(checkpoint) != record.content_digest:
        raise SetupError(
            f"the checkpoint {checkpoint} has changed since r2s train fit wrote it (a file was "
            "edited, added or removed), so what it holds is not what was trained",
            fix=_train_again(checkpoint, record),
        )


def read_training_record(checkpoint: Path) -> TrainingRecord:
    """The record of the checkpoint at ``checkpoint``; ``SetupError`` if it has none."""
    from real2simlab._fs import recover_dir

    recover_dir(Path(checkpoint), marker=TRAINING_FILE, what="the checkpoint")
    path = Path(checkpoint) / TRAINING_FILE
    if not path.is_file():
        raise SetupError(
            f"{checkpoint} is not a checkpoint: it has no {TRAINING_FILE}",
            fix="train the method first: r2s train fit METHOD --data DATASET --out CHECKPOINT",
        )
    try:
        # Bytes: a record that is not UTF-8 text (damaged) is invalid JSON, named like any other.
        return TrainingRecord.model_validate_json(path.read_bytes())
    except OSError as error:
        raise SetupError(
            f"{path} cannot be read ({error.strerror or error})",
            fix="make it readable, or train the method again",
        ) from error
    except ValidationError as error:
        raise SetupError(
            f"{path} is not a valid training record: {error.errors()[0]['msg']}",
            fix="train the method again",
        ) from error


def checkpoint_digest(checkpoint: Path) -> str:
    """The content fingerprint of the checkpoint at ``checkpoint``.

    It names one training: caches of what a checkpoint holds are keyed on it, so a checkpoint
    trained again in the same directory is read again.
    """
    try:
        return fingerprint(Path(checkpoint))
    except (OSError, ValueError) as error:
        raise SetupError(
            f"cannot read the checkpoint {checkpoint}: {error}",
            fix="a checkpoint holds regular files only (no symbolic links); train it again",
        ) from error


@cache
def training_index(checkpoint: Path, digest: str) -> OverlapIndex:
    """The overlap index of the training tasks of the checkpoint at ``checkpoint`` whose
    content fingerprint is ``digest`` (:func:`checkpoint_digest`).

    The prints are a file in the checkpoint (``prints`` names one there, never a file outside
    it), and they hold a print of every training task the record names: an edited record does
    not leave tasks a method learned from out of the held-out check.
    """
    record = read_training_record(checkpoint)
    fix = _train_again(checkpoint, record)
    path = prints_path(checkpoint, record)
    index = OverlapIndex.load(path, fix=fix)
    printed = {item.task_id for item in index.prints}
    missing = sorted(set(record.task_ids) - printed)
    if missing:
        raise SetupError(
            f"{path} has no print of {len(missing)} of the checkpoint's training tasks "
            f"({', '.join(missing[:3])}{', ...' if len(missing) > 3 else ''}), so an evaluation "
            "could not hold them out",
            fix=fix,
        )
    return index


def prints_path(checkpoint: Path, record: TrainingRecord) -> Path:
    """The file in ``checkpoint`` that holds its prints (``record.prints``); ``SetupError`` if
    the record names anything else (an absolute path, a path out of the checkpoint)."""
    name = Path(record.prints)
    root = Path(checkpoint).resolve()
    path = root / name
    resolved = path.resolve()
    inside = (
        not name.is_absolute()
        and ".." not in name.parts
        and resolved != root
        and resolved.is_relative_to(root)
    )
    if not inside:
        raise SetupError(
            f"{Path(checkpoint) / TRAINING_FILE} names prints that are not a file in the "
            f"checkpoint ({record.prints!r}): the record was edited (r2s train fit writes "
            f"{PRINTS_FILE}, in the checkpoint)",
            fix=_train_again(checkpoint, record),
        )
    return path
