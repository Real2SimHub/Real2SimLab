"""The output format registry: what a twin run writes and how it is checked."""

from __future__ import annotations

import os
import stat
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from real2simlab import plugins
from real2simlab.errors import ProblemsError

if TYPE_CHECKING:
    from real2simlab.formats.tracks import Tracks
    from real2simlab.task.model import OutputRequirements


class InvalidOutputError(ProblemsError):
    """A run's output violates its declared format."""


@dataclass(frozen=True)
class Requirements:
    """What a particular output must satisfy beyond its format's own rules."""

    fps: float | None = None
    num_frames: int | None = None
    objects: tuple[str, ...] = ()
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_task(cls, output: OutputRequirements | None) -> Requirements:
        """The requirements of a task's output."""
        if output is None:
            return cls()
        return cls(
            fps=output.fps,
            num_frames=output.num_frames,
            objects=output.objects,
            width=output.width,
            height=output.height,
        )


class OutputFormat(ABC):
    """A named output format: its rules, its documentation and its conversion to tracks."""

    name: ClassVar[str]
    summary: ClassVar[str]
    provides_tracks: ClassVar[bool] = False
    """Whether ``to_tracks`` converts outputs of this format, so tracks metrics apply."""
    content: ClassVar[tuple[str, ...] | None] = None
    """Glob patterns, relative to an output's root, of the files that make up an output of
    this format: what is compared when two outputs are checked for agreement (a session
    output and its frozen re-run, or two re-runs). ``None``: every file."""

    @abstractmethod
    def validate(self, root: Path, requirements: Requirements | None = None) -> list[str]:
        """Every violation of the format in ``root``; an empty list means it is valid."""

    def to_tracks(self, root: Path) -> Tracks | None:
        """The output as per-object tracks, or ``None`` if the format has none."""
        return None

    def documentation(self) -> str:
        """The format's specification in Markdown, as given to methods."""
        return f"# Output format `{self.name}`\n\n{self.summary}\n"

    def check(self, root: Path, requirements: Requirements | None = None) -> None:
        """Raise ``InvalidOutputError`` listing every problem of ``root``."""
        problems = output_problems(self, root, requirements)
        if problems:
            raise InvalidOutputError(f"{root} is not a valid {self.name!r} output", problems)


NO_OUTPUT = "the output directory does not exist"


def tree_problems(root: Path, *, unreadable: bool = False) -> list[str]:
    """Why the output directory ``root``, which a sandboxed program wrote, cannot be read as it
    is: ``root`` is missing, a symbolic link or not a directory; it holds a symbolic link, which
    reading could follow out of the output (into a task's hidden data), or a pipe, socket or
    device, which reading could wait on forever. With ``unreadable``, also a directory this
    user cannot list and a file it cannot read. Nothing is followed or opened, and a very deep
    tree needs no recursion. An empty list: the directory may be read."""
    root = Path(root)
    try:
        mode = os.lstat(root).st_mode
    except (OSError, ValueError):
        return [NO_OUTPUT]
    if stat.S_ISLNK(mode):
        return ["the output directory is a symbolic link"]
    if not stat.S_ISDIR(mode):
        return [NO_OUTPUT]
    links: list[str] = []
    other: list[str] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        named = "the output directory" if directory == root else f"{_relative(directory, root)}/:"
        try:
            if unreadable and not os.access(directory, os.R_OK | os.X_OK):
                raise PermissionError
            with os.scandir(directory) as listing:
                entries = list(listing)
        except OSError:
            if unreadable:
                other.append(f"{named} cannot be listed")
            continue
        for entry in entries:
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError:  # removed meanwhile
                continue
            relative = _relative(Path(entry.path), root)
            if stat.S_ISLNK(mode):
                links.append(relative)
            elif stat.S_ISDIR(mode):
                stack.append(Path(entry.path))
            elif not stat.S_ISREG(mode):
                other.append(f"{relative}: not a regular file (a pipe, socket or device)")
            elif unreadable and not os.access(entry.path, os.R_OK):
                other.append(f"{relative}: cannot be read")
    if links:
        return [f"{link}: symbolic links are not allowed" for link in sorted(links)]
    return sorted(other)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def output_problems(
    fmt: OutputFormat, root: Path, requirements: Requirements | None = None
) -> list[str]:
    """Generic checks (:func:`tree_problems`: a directory, not a symbolic link to one, of
    regular files: no symbolic links, pipes, sockets or devices, which reading could follow out
    of the output or wait on forever) followed by the format's own."""
    problems = tree_problems(root)
    if problems:
        return problems
    return fmt.validate(Path(root), requirements)


def get_format(name: str) -> OutputFormat:
    """The registered output format called ``name``."""
    factory = plugins.load("format", name)
    fmt = factory()
    if not isinstance(fmt, OutputFormat):
        raise TypeError(f"format plugin {name!r} does not create an OutputFormat")
    return fmt


def format_names() -> list[str]:
    return plugins.names("format")
