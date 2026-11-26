"""Task directories (``r2s-task/1.0``) on disk.

::

    <task>/
      task.toml        metadata, delivery, scoring, source and license
      instruction.md   optional: replaces the generated task statement
      visible/         everything a method may see: capture.json and its media; docs/; tools/
      hidden/          never shown to a method: reference/ (the true output of the recorded
                       event); repeats/<k>/ (other recordings of the same event, for the noise
                       floor); generator/ (a twin that produces the truth, for synthetic tasks);
                       anything a benchmark's official scorer needs
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import tomli_w
from pydantic import ValidationError

from real2simlab._base import validation_problems
from real2simlab._fs import force_remove
from real2simlab.capture.io import CAPTURE_FILENAME, load_capture
from real2simlab.capture.model import Capture
from real2simlab.errors import SpecError
from real2simlab.task.model import TaskConfig
from real2simlab.twin.fingerprint import SymbolicLinkError, fingerprint, is_os_litter

TASK_FILE = "task.toml"
INSTRUCTION_FILE = "instruction.md"
VISIBLE_DIR = "visible"
HIDDEN_DIR = "hidden"
REFERENCE_DIR = "reference"
REPEATS_DIR = "repeats"
GENERATOR_DIR = "generator"
DOCS_DIR = "docs"
TOOLS_DIR = "tools"
PARAMETERS_FILE = "parameters.json"
"""``hidden/parameters.json``, optional: the hidden parameters and initial state (synthetic
tasks), a JSON object; for a training task, labels that ``r2s train export`` puts in its row."""


def read_task_config(path: Path) -> TaskConfig:
    """Parse a ``task.toml``. Raises ``SpecError`` listing every problem."""
    try:
        data = Path(path).read_bytes()
    except OSError as error:
        raise SpecError(f"cannot read {path}", [str(error)]) from error
    return _task_config(data, Path(path))


def _task_config(data: bytes, path: Path) -> TaskConfig:
    """``data``, the bytes of the ``task.toml`` at ``path``, parsed as that file is read (UTF-8,
    universal newlines). Raises ``SpecError`` listing every problem."""
    try:
        text = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8").read()
        parsed = tomllib.loads(text)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise SpecError(f"cannot read {path}", [str(error)]) from error
    try:
        return TaskConfig.model_validate(parsed)
    except ValidationError as error:
        raise SpecError(f"{path} is not a valid task", validation_problems(error)) from error


def write_task_config(config: TaskConfig, path: Path) -> Path:
    """Write ``config`` as TOML to ``path``. The default ``split`` (``eval``) is left out, so
    evaluation tasks are written as they were before the field existed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_json()
    if data.get("split") == "eval":
        del data["split"]
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    return path


@dataclass(frozen=True)
class Task:
    """A task directory with its parsed ``task.toml``."""

    root: Path
    config: TaskConfig
    toml_digest: str | None = field(default=None, compare=False)
    """The SHA-256 of the ``task.toml`` bytes ``config`` was parsed from (:func:`load_task`):
    whether the file changed since is told by its bytes (a parsed ``nan``, in a free-form
    table, is never equal to itself)."""

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def visible_dir(self) -> Path:
        return self.root / VISIBLE_DIR

    @property
    def hidden_dir(self) -> Path:
        return self.root / HIDDEN_DIR

    def capture(self) -> Capture:
        """The visible capture."""
        return load_capture(self.visible_dir)

    def has_capture(self) -> bool:
        return (self.visible_dir / CAPTURE_FILENAME).is_file()

    def reference_dir(self) -> Path:
        """The true output of the recorded event, in the task's output format."""
        return self.hidden_dir / REFERENCE_DIR

    def repeat_dirs(self) -> list[Path]:
        """Other recordings of the same event, for the noise floor; may be empty."""
        root = self.hidden_dir / REPEATS_DIR
        if not root.is_dir():
            return []
        return sorted(path for path in root.iterdir() if path.is_dir())

    @property
    def generator_dir(self) -> Path | None:
        """The truth generator (a twin package), when the task has one."""
        path = self.hidden_dir / GENERATOR_DIR
        return path if path.is_dir() else None

    def instruction_override(self) -> str | None:
        path = self.root / INSTRUCTION_FILE
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def content_hash(self) -> str:
        """SHA-256 of the whole task directory: the cache key of everything built from it."""
        return fingerprint(self.root)

    def changed_since(self, task_hash: str) -> bool:
        """Whether the directory is no longer the version ``task_hash`` names, or its
        ``task.toml`` is no longer what ``config`` was parsed from (an edit after it was
        loaded): told by its bytes when they were kept (``toml_digest``), else by what the file
        says now."""
        if self.content_hash() != task_hash:
            return True
        path = self.root / TASK_FILE
        if self.toml_digest is not None:
            try:
                return hashlib.sha256(path.read_bytes()).hexdigest() != self.toml_digest
            except OSError:
                return True
        try:
            return read_task_config(path) != self.config
        except SpecError:
            return True

    def visible_hash(self) -> str:
        return fingerprint(self.visible_dir)

    def describe(self) -> dict[str, object]:
        """A JSON-compatible summary for ``r2s task show`` (no hidden values)."""
        return {
            "id": self.config.id,
            "version": self.config.version,
            "title": self.config.title,
            "path": str(self.root),
            "source": self.config.source.to_json(),
            "license": self.config.license,
            "redistributable": self.config.redistributable,
            "visibility": self.config.visibility,
            "environment": self.config.delivery.environment,
            "output_format": self.config.delivery.output_format,
            "protocol": self.config.scoring.protocol.value,
            "metrics": list(self.config.scoring.metrics),
            "primary": self.config.scoring.primary,
            "native_scorer": (
                self.config.scoring.native.scorer if self.config.scoring.native else None
            ),
            "has_reference": self.reference_dir().is_dir(),
        }


def load_task(root: Path) -> Task:
    """Load the task at ``root``; ``SpecError`` if ``task.toml`` is missing or invalid."""
    root = Path(root)
    path = root / TASK_FILE
    if not path.is_file():
        raise SpecError(f"{root} is not a task directory", [f"{TASK_FILE} is missing"])
    try:
        data = path.read_bytes()
    except OSError as error:
        raise SpecError(f"cannot read {path}", [str(error)]) from error
    digest = hashlib.sha256(data).hexdigest()
    return Task(root=root, config=_task_config(data, path), toml_digest=digest)


def symbolic_links(root: Path) -> list[str]:
    """The symbolic links under ``root`` (none of them followed), relative to it, sorted."""
    found: list[str] = []
    for directory, names, files in os.walk(root):
        for name in (*names, *files):
            path = Path(directory, name)
            if path.is_symlink():
                found.append(path.relative_to(root).as_posix())
    return sorted(found)


def _litter(directory: str, names: list[str]) -> set[str]:
    """What :func:`copy_exactly` leaves out of ``directory``: a file manager's litter
    (``.DS_Store``, ``._*``; :func:`~real2simlab.twin.fingerprint.is_os_litter`), which no
    fingerprint covers, and a link of that name. A directory of that name is copied: the files
    in it are covered."""
    left_out: set[str] = set()
    for name in names:
        path = Path(directory, name)
        if is_os_litter(name) and (path.is_symlink() or not path.is_dir()):
            left_out.add(name)
    return left_out


def copy_exactly(source: Path, target: Path, *, expected: str | None = None) -> str | None:
    """Copy the directory ``source`` to ``target`` (which must not exist) exactly as it is, and
    return why the copy is not the checked version of ``source``, or ``None`` if it is.

    Nothing from outside ``source`` is copied: a symbolic link is copied as a link, never
    followed (``shutil.copytree`` follows links by default, so a link in a task's ``visible/``
    would put what it points to, hidden data for instance, in the copy). A file manager's
    litter (``.DS_Store``, ``._*``) is not copied at all: no fingerprint covers it, so the
    comparison below could not tell such a file added since the check, whatever it holds. A
    copy that holds a link is not the checked version (a task holds none), nor is one whose
    fingerprint is not ``expected``: the fingerprint of ``source`` when it was checked, by
    default just before copying. ``source`` then changed since it was checked, or while it was
    copied. Such a copy is removed: nothing unchecked is left for anyone to use. A copy that
    fails (a full disk) raises, as ``shutil.copytree`` does.
    """
    source, target = Path(source), Path(target)
    if source.is_symlink():
        return f"{source} is a symbolic link"
    if not source.is_dir():
        return f"{source} is not a directory"
    if expected is None:
        with contextlib.suppress(SymbolicLinkError):  # the copy shows which link
            expected = fingerprint(source)
    shutil.copytree(source, target, symlinks=True, ignore=_litter)
    links = symbolic_links(target)
    problem = None
    if links:
        problem = f"{source / links[0]} is a symbolic link"
    elif expected is None or fingerprint(target) != expected:
        problem = f"{source} changed since it was checked, or while it was copied"
    if problem is not None:
        force_remove(target)
    return problem


def find_tasks(root: Path) -> list[Path]:
    """Every task directory at or below ``root`` (directories holding ``task.toml``)."""
    root = Path(root)
    if (root / TASK_FILE).is_file():
        return [root]
    found = sorted(path.parent for path in root.rglob(TASK_FILE))
    return [path for path in found if HIDDEN_DIR not in path.relative_to(root).parts]


def write_json(path: Path, data: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
