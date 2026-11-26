"""The base class of benchmark adapters.

An adapter turns a benchmark's data into Real2SimLab task directories, declares what methods
deliver and how they are scored, and documents the source and license. The part that imports
data is a set of plain functions in the adapter's module, so other projects
can assemble their own tasks from it.
"""

from __future__ import annotations

import hashlib
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from real2simlab import plugins

if TYPE_CHECKING:
    from real2simlab.task import Task

NOT_OPTIONS = ("out_dir", "select")
"""Parameters of ``build`` that are not options: a command sets them itself (``--out``,
``--select``; ``select = [...]`` in an experiment file)."""


@dataclass(frozen=True)
class BenchmarkInfo:
    name: str
    summary: str
    license: str
    source: str
    protocols: tuple[str, ...]
    status: str
    """``ready``, ``partial`` (a documented gap) or ``needs-data``."""
    splits: tuple[str, ...] = ("eval",)


class BenchmarkAdapter(ABC):
    """Builds task directories for one benchmark."""

    name: ClassVar[str]
    summary: ClassVar[str]
    license: ClassVar[str]
    source: ClassVar[str]
    protocols: ClassVar[tuple[str, ...]] = ("reproduction",)
    status: ClassVar[str] = "ready"
    training_terms: ClassVar[str | None] = None
    """The benchmark's own terms, quoted, when they forbid training on its data: its tasks are
    then never exported as training data, whatever the user asks (``r2s train export``)."""
    splits: ClassVar[tuple[str, ...]] = ("eval",)
    """The splits the adapter builds: ``eval``, and ``train`` when the benchmark releases
    training data that its terms allow training on. ``build`` then takes ``split="train"``
    and writes tasks with ``split = "train"``, which ``r2s train export`` accepts."""

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name=self.name,
            summary=self.summary,
            license=self.license,
            source=self.source,
            protocols=self.protocols,
            status=self.status,
            splits=self.splits,
        )

    def source_files(self) -> list[Path]:
        """Every file that can shape this adapter's tasks: all of its top-level package (code,
        data, pins) and the Lab's own code and data, except the CLI and the report."""
        top = sys.modules[type(self).__module__.split(".")[0]]
        lab = Path(__file__).resolve().parents[1]
        roots = [lab]
        if top.__file__ and Path(top.__file__).resolve().parent != lab:
            origin = Path(top.__file__).resolve()
            roots.insert(0, origin.parent if origin.name == "__init__.py" else origin)
        files: list[Path] = []
        for root in roots:
            if root.is_file():
                files.append(root)
                continue
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root).parts
                if not path.is_file() or "__pycache__" in relative or path.suffix == ".pyc":
                    continue
                if root == lab and relative[0] in {"cli", "report"}:
                    continue
                files.append(path)
        return files

    def source_digest(self) -> str:
        """A digest of :meth:`source_files`. Experiments rebuild the tasks they cached when it
        changes, even if the Lab's version string does not; a task whose content does not
        change keeps its hash, so its finished builds are still reused."""
        digest = hashlib.sha256()
        lab = Path(__file__).resolve().parents[1]
        for path in self.source_files():
            # Relative names: moving the checkout or the environment does not change it.
            name = path.relative_to(lab) if path.is_relative_to(lab) else Path(path.name)
            digest.update(os.fsencode(name.as_posix()))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @abstractmethod
    def task_names(self) -> list[str]:
        """The tasks (or task groups) this adapter can build."""

    @abstractmethod
    def build(
        self, out_dir: Path, *, select: Sequence[str] | None = None, **options: Any
    ) -> list[Path]:
        """Write task directories under ``out_dir``; return their paths."""

    def options_taken(self, *read: str) -> list[str]:
        """The options ``build`` takes, for the error that refuses one it does not
        (``UnknownOptionError.from_names``): its parameters but those a command sets itself
        (``NOT_OPTIONS``), and ``read``, those it reads from its ``**options``."""
        import inspect

        parameters = inspect.signature(self.build).parameters.values()
        named = {p.name for p in parameters if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        return sorted((named | set(read)) - set(NOT_OPTIONS))

    def training_refusal(self, task: Task) -> str | None:
        """Why ``task``, one of this benchmark's, may never be training data, or ``None``.

        By default the benchmark's ``training_terms``. Override it when only some tasks are
        restricted, e.g. clips taken from another benchmark whose terms forbid training.
        """
        if self.training_terms is None:
            return None
        return f"{self.name}'s terms forbid training on its data: {self.training_terms}"

    def build_split(
        self,
        out_dir: Path,
        split: str,
        /,
        *,
        select: Sequence[str] | None = None,
        **options: Any,
    ) -> list[Path]:
        """Build the tasks of ``split`` (``r2s bench build --split``). Refuses a split the
        adapter does not declare, and checks that every task it wrote is marked with it."""
        from real2simlab.errors import SetupError
        from real2simlab.options import checked_options
        from real2simlab.task import load_task

        if split not in self.splits:
            raise SetupError(
                f"{self.name} has no {split!r} split (it builds: {', '.join(self.splits)})",
                fix="a benchmark released for evaluation has no training split; train on "
                "training tasks of your own (see the Training guide)",
            )
        if "split" in options:
            raise SetupError(
                "the split is chosen with build_split's split (r2s bench build --split), not "
                "with an option",
                fix=f"r2s bench build {self.name} --split {options['split']}",
            )
        if split != "eval":
            options["split"] = split
        checked = checked_options(
            self.build, options, f"benchmark {self.name!r}", reserved=NOT_OPTIONS
        )
        paths = self.build(out_dir, select=select, **checked)
        for path in paths:
            found = load_task(path).config.split.value
            if found != split:
                raise SetupError(
                    f"{self.name} wrote {path} with split = {found!r} for the {split!r} split",
                    fix=f"make {type(self).__name__}.build mark its tasks with split = {split!r}",
                )
        return paths


def get_benchmark(name: str) -> BenchmarkAdapter:
    factory = plugins.load("benchmark", name)
    adapter = factory()
    if not isinstance(adapter, BenchmarkAdapter):
        raise TypeError(f"benchmark plugin {name!r} does not create a BenchmarkAdapter")
    return adapter


def benchmark_names() -> list[str]:
    return plugins.names("benchmark")
