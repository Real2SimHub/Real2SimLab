"""``program``: evaluate a twin program that already exists, e.g. written by hand or by a pipeline.

The program directory holds the entry point of the task's delivery convention (``run.py`` by
default); a twin package (``twin.json`` and ``program/``) is taken for its program. A ``path``
that holds the entry point is the program for every task, whatever its subdirectories hold, as
``r2s train reward`` and ``score_solution`` take the same directory
(:func:`~real2simlab.twin.io.has_entry_point` decides it for both), so a reward and an
evaluation of it are equal. A ``path`` without one is a directory of per-task programs: the
program for a task is looked up, most specific first, as ``<path>/<task id>/`` (its segments as
nested directories, e.g. ``synthetic/bouncing_ball/calibrated/``), ``<path>/<task id with '/'
written as '__'>/``, or ``<path>/<one segment of the task id>/`` (the last first).

The caller's mistakes are refused before anything runs: an empty ``path``, one that is not a
directory, and one that is both a program and a directory holding a program for one of the
tasks (which of the two is meant is not known). A directory without a program for a task is
that task's result (nothing delivered), and so is a program holding what no program may
deliver (a symbolic link, a file that cannot be read) or reached through a symbolic link under
``path`` (a per-task directory, a twin package's ``program/``): its build fails, as its reward
is 0, not the experiment. The program is copied, checked against the task's rules, and frozen like
any other twin; one that changed while it was copied is not a result (``infra_error``) and is
built again.
"""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from real2simlab.agents.base import clean_solution, solution_digest
from real2simlab.errors import (
    DeliveryError,
    InfrastructureError,
    InputNotFoundError,
    R2SError,
    SetupError,
)
from real2simlab.methods.base import BuildContext, BuildOutcome, Method, MethodOptions, package_twin
from real2simlab.twin import InvalidTwinError, program_of
from real2simlab.twin.conventions import convention_spec
from real2simlab.twin.io import has_entry_point

if TYPE_CHECKING:
    from real2simlab.task.io import Task

PATH_FIX = (
    "give program:path= the directory that holds the program (run.py), a twin package, or a "
    "directory of per-task programs"
)


class ProgramOptions(MethodOptions):
    path: str
    """A program directory (it holds the entry point, e.g. ``run.py``: the program for every
    task), or a directory of per-task program directories named after the task id (nested,
    ``synthetic/bouncing_ball/calibrated``, or ``synthetic__bouncing_ball__calibrated``, or one
    segment of the id); each may also be a twin package (``twin.json`` and ``program/``)."""
    entrypoint: tuple[str, ...] | None = None
    """Defaults to the task's delivery convention (e.g. ``python ./run.py``)."""
    label_model: str | None = None
    """Recorded as the twin's ``origin.model``, e.g. the pipeline's name and version."""


class ProgramMethod(Method):
    name: ClassVar[str] = "program"
    kind: ClassVar[str] = "method"
    summary: ClassVar[str] = "an existing twin program (hand-written or from a pipeline)"
    Options: ClassVar[type[MethodOptions]] = ProgramOptions

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        if not self.program_options.path:  # an unset variable, a blank template: not "."
            raise InputNotFoundError(
                "the program path (path=) is empty: it would be the current directory",
                fix=PATH_FIX,
            )
        root = Path(self.program_options.path).expanduser()
        if not root.is_dir():
            what = "is not a directory" if root.exists() else "does not exist"
            raise InputNotFoundError(
                f"the program path {self.program_options.path} {what}", fix=PATH_FIX
            )

    @property
    def program_options(self) -> ProgramOptions:
        assert isinstance(self.options, ProgramOptions)
        return self.options

    def config_digest(self) -> str:
        """Includes the program's content, so editing it invalidates cached builds. A program
        holding what no program may deliver (a symbolic link, a file that cannot be read) has
        one too, of what it holds (:func:`~real2simlab.agents.base.solution_digest`): the
        build of the task whose program it is fails as that task's result, not every build."""
        from real2simlab.twin.fingerprint import fingerprint, text_digest

        root = Path(self.program_options.path).expanduser()
        if not root.is_dir():
            content = "missing"
        else:
            try:
                content = fingerprint(root)
            except (OSError, ValueError):  # a link, an unreadable file: refused when it is built
                content = "undeliverable " + solution_digest(root)
        return text_digest(super().config_digest(), content)

    def applicable(self, task: Task) -> str | None:
        """``None``: a directory without a program for ``task`` is that task's result (nothing
        delivered). Its program is looked up here, before anything runs, so that a ``path``
        that is ambiguous for the task (:meth:`locate`) is refused before any build starts."""
        self.locate(task.id, [convention_spec(task.config.delivery.convention).entry_file])
        return None

    def locate(self, task_id: str, markers: Sequence[str] = ("run.py",)) -> Path | None:
        """The program for ``task_id``: ``path`` itself when it holds one of ``markers`` (the
        entry point: :func:`~real2simlab.twin.io.has_entry_point`, as for a reward), else a
        subdirectory for the task holding one, most specific first: the task id as nested
        directories (``synthetic/bouncing_ball/calibrated``), the id with ``/`` written as
        ``__``, then one segment of the id (the last first; a segment's directory that also
        holds other programs, nested, is no task's to share). Raises ``SetupError`` when
        ``path`` holds one of ``markers`` and also a program for the task: which is meant is
        not known."""
        root = Path(self.program_options.path).expanduser()

        def marked(directory: Path) -> bool:
            try:
                return has_entry_point(program_of(directory), markers)
            except OSError:  # a path this system cannot hold (too long): not there
                return False

        if program_of(root) != root:  # a twin package: its program, for every task
            return program_of(root) if marked(root) else None
        own = _own_directory(root, task_id, marked)
        if not marked(root):
            return None if own is None else program_of(own)
        if own is not None:
            entry = " or ".join(markers)
            inside = own.relative_to(root).as_posix()
            raise SetupError(
                f"program:path={self.program_options.path} is a program (it holds {entry}) "
                f"and also holds a program for task {task_id} ({inside}/): which of them is "
                "meant is not known",
                fix=f"to evaluate {root} as the program for every task, move {inside}/ out of it "
                f"(or rename it); to evaluate per-task programs, move its own {entry} into the "
                "directory of the task it is for",
            )
        return root

    def build(self, context: BuildContext) -> BuildOutcome:
        markers = [convention_spec(context.config.delivery.convention).entry_file]
        located = self.locate(context.task_id, markers)
        if located is None:
            return BuildOutcome(
                None,
                f"no program with {' or '.join(markers)} for {context.task_id} in "
                f"{self.program_options.path}",
            )
        root = Path(self.program_options.path).expanduser()
        linked = _linked(root, located)
        if linked is not None:  # never delivered: as a reward refuses it, and not in the key
            return BuildOutcome(
                None,
                f"{linked}, not a directory of its own; fix: put the program itself there (a "
                "copy of what the link points to), or give program:path= that directory",
            )
        source = located.resolve()  # the user's own directory (path=), through its links
        fix = (
            f"fix: keep in {source} only regular files you can read, no symbolic link, within "
            f"the task's rules (`r2s twin check {source} --task <the task's directory>` checks "
            "them), then run the same command again"
        )
        staging = context.work_dir / "program"
        copied = solution_digest(source)
        try:
            clean_solution(source, staging)
        except DeliveryError as error:  # what it holds (an unreadable file): this task's result
            return BuildOutcome(None, f"{error}; {fix}")
        try:
            if solution_digest(source) != copied:
                problem = (
                    f"the program in {source} changed while it was copied, so the copy may mix "
                    "two versions of it; run the same command again once it is no longer edited"
                )
                return BuildOutcome(None, problem, infra_error=problem)
            twin = package_twin(
                context,
                staging,
                method=self.name,
                entrypoint=self.program_options.entrypoint,
                model=self.program_options.label_model,
            )
        except InfrastructureError:  # this machine's failure: not a result (the build says so)
            raise
        except (InvalidTwinError, R2SError, ValueError) as error:
            return BuildOutcome(None, f"the program is not a valid twin: {error}; {fix}")
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return BuildOutcome(twin)


def _linked(root: Path, program: Path) -> str | None:
    """The directory on the way from ``root`` (``path=``, itself taken through its links, as
    ``r2s train reward --solution`` is) to ``program`` that is a symbolic link, ``program``
    included (a per-task directory, a twin package's ``program/``), as ``<relative path>/ is a
    symbolic link``; ``None`` when there is none. What is behind such a link is not ``path``'s
    own: ``score_solution`` refuses it (:func:`~real2simlab.agents.base.directory_problem`),
    and the build's key (:meth:`ProgramMethod.config_digest`) holds only where it points."""
    walked = root
    for part in program.relative_to(root).parts:
        walked = walked / part
        try:
            link = stat.S_ISLNK(os.lstat(walked).st_mode)
        except OSError:  # gone since it was found: the copy says what is missing
            return None
        if link:
            return f"{walked.relative_to(root).as_posix()}/ is a symbolic link"
    return None


def _own_directory(root: Path, task_id: str, marked: Callable[[Path], bool]) -> Path | None:
    """The directory in ``root`` holding the program of ``task_id`` (``marked``), most specific
    first: the task id as nested directories, the id with ``/`` written as ``__``, then one
    segment of the id (the last first) that holds no other program (a nested layout's
    directory of another task is not shared). ``None`` when there is none."""

    def fits(name: Path) -> bool:  # a name longer than a file name can be is not there
        return all(len(part.encode()) <= 255 for part in name.parts)

    def alone(directory: Path) -> bool:
        """A program, not also a directory of programs (another task's, nested)."""
        try:
            if program_of(directory) != directory:  # a twin package: program/ is no task's
                return marked(directory)
            children = [child for child in directory.iterdir() if child.is_dir()]
        except OSError:  # not there, or a path this system cannot hold
            return False
        return marked(directory) and not any(marked(child) for child in children)

    segments = task_id.split("/")
    for name in (Path(*segments), Path(task_id.replace("/", "__"))):
        if fits(name) and marked(root / name):
            return root / name
    for name in map(Path, reversed(segments)):  # shared by the tasks with that segment
        if fits(name) and alone(root / name):
            return root / name
    return None
