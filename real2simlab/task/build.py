"""The Python API benchmark adapters use to write task directories.

::

    builder = TaskBuilder(out_dir, config)
    builder.capture(full_capture, media_dir, visibility=CALIBRATED)
    builder.reference(tracks=true_tracks)       # what the delivery is scored against
    builder.repeat(1, tracks=second_take)       # optional: other recordings of the event
    task = builder.finish()          # writes task.toml and validates the directory
"""

from __future__ import annotations

import shutil
from pathlib import Path

from real2simlab._fs import copy_tree, force_remove, has_mark, lab_record, write_mark
from real2simlab.capture.io import copy_media, save_capture
from real2simlab.capture.model import Capture
from real2simlab.capture.visibility import VisibilityPolicy
from real2simlab.errors import OutputExistsError, SetupError
from real2simlab.formats.tracks import Tracks, write_tracks
from real2simlab.task.io import (
    DOCS_DIR,
    GENERATOR_DIR,
    HIDDEN_DIR,
    INSTRUCTION_FILE,
    REFERENCE_DIR,
    REPEATS_DIR,
    TASK_FILE,
    TOOLS_DIR,
    VISIBLE_DIR,
    Task,
    load_task,
    write_task_config,
)
from real2simlab.task.model import TaskConfig
from real2simlab.task.validate import validate_task

AUDIT_FILE = "visibility_audit.json"

BUILD_MARKER = ".r2s-task-build"
BUILD_MARK = "a task being written: task.toml comes last\n"
"""Written into a task directory before anything else, and removed once ``task.toml`` is: a
directory a build stopped in (Ctrl-C, a crash) is known by it as the Lab's, to be replaced by
``--overwrite``. One that merely holds a task's names (``visible/``, ``hidden/``) is not: a
task the user is writing by hand looks the same."""


def _being_built(root: Path) -> bool:
    """Whether ``root`` holds :data:`BUILD_MARKER` as a build writes it (or as a build killed
    while it wrote it left it: :func:`~real2simlab._fs.has_mark`)."""
    return has_mark(root / BUILD_MARKER, BUILD_MARK)


def _check_task_dir(root: Path) -> None:
    """Raise unless ``root`` is a directory a task build may replace: a task (its ``task.toml``
    is a task file), one a build was writing when it stopped (:data:`BUILD_MARKER`), or empty.
    ``overwrite`` never removes anything else, whatever its files are named."""
    if root.is_symlink() or not root.is_dir():
        raise SetupError(
            f"{root} is not a task directory, so --overwrite does not replace it",
            fix="move it away, or build into another directory",
        )
    if _being_built(root) or not any(root.iterdir()):
        return
    if (root / TASK_FILE).exists():
        if lab_record(root / TASK_FILE):
            return
        raise SetupError(
            f'{root}/{TASK_FILE} is not a task file (it has no spec = "r2s-task/..."); '
            "--overwrite replaces only task directories",
            fix="move it away, or build into another directory",
        )
    names = ", ".join(sorted(entry.name for entry in root.iterdir())[:5])
    raise SetupError(
        f"{root} is not a task directory: it has no {TASK_FILE}, and its files ({names}) are "
        "not part of a task a build was writing; --overwrite replaces only task directories",
        fix="move them away, or build into another directory",
    )


class TaskBuilder:
    """Writes one task directory step by step; ``finish`` validates it."""

    def __init__(self, root: Path, config: TaskConfig, *, overwrite: bool = False) -> None:
        self.root = Path(root)
        self.config = config
        if self.root.exists() or self.root.is_symlink():
            if not overwrite:
                raise OutputExistsError(
                    f"task directory already exists: {self.root}",
                    fix="pass --overwrite to replace it, or choose another --out",
                )
            _check_task_dir(self.root)
            # The mark first: stopped while the old task is removed, the directory is still
            # known as a build's. A stopped build's mark is kept as it is (written again on a
            # full disk, it would be lost); a mark that cannot be written is not left behind.
            if not _being_built(self.root):
                write_mark(self.root / BUILD_MARKER, BUILD_MARK)
            for entry in list(self.root.iterdir()):
                if entry.name == BUILD_MARKER:
                    continue
                problem = force_remove(entry)
                if problem is not None:
                    raise SetupError(problem, fix=f"remove {self.root} yourself, then build again")
        else:
            self.root.mkdir(parents=True)
            write_mark(self.root / BUILD_MARKER, BUILD_MARK)
        (self.root / VISIBLE_DIR).mkdir()
        (self.root / HIDDEN_DIR).mkdir()

    @property
    def visible_dir(self) -> Path:
        return self.root / VISIBLE_DIR

    @property
    def hidden_dir(self) -> Path:
        return self.root / HIDDEN_DIR

    def capture(
        self, capture: Capture, media_dir: Path, *, visibility: VisibilityPolicy | None = None
    ) -> Capture:
        """Write the method-visible part of ``capture`` and only the media it still references.

        With ``visibility``, fields it does not expose are withheld, and what was withheld is
        recorded in ``hidden/visibility_audit.json``.
        """
        visible = capture
        if visibility is not None:
            visible, audit = visibility.apply(capture)
            (self.hidden_dir / AUDIT_FILE).write_text(audit.dumps(), encoding="utf-8")
        save_capture(visible, self.visible_dir)
        copy_media(visible, Path(media_dir), self.visible_dir)
        return visible

    def visible_doc(
        self, name: str, *, text: str | None = None, source: Path | None = None
    ) -> Path:
        """Add a reference document under ``visible/docs/``."""
        return self._put(self.visible_dir / DOCS_DIR / name, text=text, source=source)

    def visible_tool(self, name: str, source: Path) -> Path:
        """Add a tool (file or directory) under ``visible/tools/``."""
        target = self.visible_dir / TOOLS_DIR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(source).is_dir():
            copy_tree(source, target)
        else:
            shutil.copyfile(source, target)
        return target

    def reference(self, source: Path | None = None, *, tracks: Tracks | None = None) -> Path:
        """Store the true output of the recorded event, from a directory or tracks."""
        return self._output(self.hidden_dir / REFERENCE_DIR, source, tracks)

    def repeat(
        self, index: int, source: Path | None = None, *, tracks: Tracks | None = None
    ) -> Path:
        """Store another recording of the same event, used for the noise floor."""
        return self._output(self.hidden_dir / REPEATS_DIR / f"{index:03d}", source, tracks)

    def generator(self, twin_dir: Path) -> Path:
        """Store a twin package that generates the truth (synthetic tasks)."""
        target = self.hidden_dir / GENERATOR_DIR
        copy_tree(twin_dir, target)
        return target

    def hidden_file(
        self, relative: str, *, text: str | None = None, source: Path | None = None
    ) -> Path:
        """Store any other file scoring needs under ``hidden/``."""
        return self._put(self.hidden_dir / relative, text=text, source=source)

    def instruction(self, text: str) -> Path:
        """Replace the generated task statement with ``text`` (``{workspace}`` is substituted)."""
        return self._put(self.root / INSTRUCTION_FILE, text=text)

    def finish(self, *, validate: bool = True) -> Task:
        """Write ``task.toml`` (which marks the directory as a task from then on, instead of
        :data:`BUILD_MARKER`); validate the directory unless ``validate`` is false."""
        write_task_config(self.config, self.root / TASK_FILE)
        (self.root / BUILD_MARKER).unlink(missing_ok=True)
        if validate:
            validate_task(self.root).raise_for_errors()
        return load_task(self.root)

    @staticmethod
    def _put(target: Path, *, text: str | None = None, source: Path | None = None) -> Path:
        if (text is None) == (source is None):
            raise ValueError("give exactly one of text and source")
        target.parent.mkdir(parents=True, exist_ok=True)
        if text is not None:
            target.write_text(text, encoding="utf-8")
        else:
            assert source is not None
            shutil.copyfile(source, target)
        return target

    @staticmethod
    def _output(target: Path, source: Path | None, tracks: Tracks | None) -> Path:
        if (source is None) == (tracks is None):
            raise ValueError("give exactly one of source and tracks")
        if target.exists():
            shutil.rmtree(target)
        if tracks is not None:
            return write_tracks(tracks, target)
        assert source is not None
        copy_tree(source, target)
        return target
