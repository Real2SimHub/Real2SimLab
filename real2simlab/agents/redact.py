"""Redacting credentials from logs and agent trajectories before they are written.

Two layers: exact values of the secrets injected into the run, and patterns of common key
formats (OpenAI, Anthropic, Google, GitHub, AWS, Hugging Face, Slack, JWTs, bearer tokens,
private keys, ``"api_key": "..."`` assignments). The patterns are defined once, in
``bootstrap.py``, which applies them inside the sandbox so raw agent output never reaches
the disk; this module applies them on the host.

An agent can also write files itself: in its workspace (its solution, which is frozen and
exported) and straight into the log directory. :func:`redact_files` replaces the injected
values in those after the run, before anything is packaged, rendered or exported. Only the
exact values are replaced there: a program is never changed by a pattern. A file that cannot
be rewritten is removed instead, so no copy of a value is left behind.
"""

from __future__ import annotations

import contextlib
import errno
import os
import re
import stat
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from real2simlab.agents.bootstrap import (
    LONG_VALUE,
    MIN_SECRET,
    PATTERNS,
    REPLACEMENT,
    Redactor,
    credential_name,
    file_secrets,
)

CHUNK = 1 << 20
MACHINE_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT, errno.EFBIG, errno.EROFS, errno.EIO})
"""Why this machine, not an agent, fails to rewrite a file: a full disk or quota, the
file-size limit (``ulimit -f``), a read-only or failing file system. Anything else (a name too
long for the copy, a link where a file was) is the agent's doing."""
_NOT_FOLLOWED = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
_LISTED = stat.S_IRUSR | stat.S_IXUSR


def environment_secrets() -> list[str]:
    """The values of this process's environment variables named as credentials by a whole
    word of the name (``OPENAI_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``, ``HF_TOKEN``; not
    ``GIT_AUTHOR_NAME`` or ``SSH_AUTH_SOCK``) and long enough to be one (``LONG_VALUE``
    characters, not a number: not ``CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000``). What the Lab
    writes of text it did not make (what an agent wrote, in an export; an error a method
    raised, in a record or ``r2s.log``) has them redacted, as it has strings shaped like keys.
    A name or a count cut out of the data would change it, or leave a JSON line unreadable."""
    return [
        value
        for name, value in os.environ.items()
        if credential_name(name) and len(value.strip()) >= LONG_VALUE and not value.isdigit()
    ]


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """``text`` with secrets and key-shaped strings replaced."""
    return Redactor(secrets)(text)


@dataclass
class Redacted:
    """What :func:`redact_files` did. Paths are relative to the directory it went through."""

    count: int = 0
    """Values replaced with ``[REDACTED]``."""
    removed: list[str] = field(default_factory=list[str])
    """Files that could not be rewritten (or read), so may have held a value: removed, or
    emptied where they could not be removed."""
    left: list[str] = field(default_factory=list[str])
    """Files and directories (``name/``) that may still hold a value: they could be neither
    read nor removed."""
    errors: list[OSError] = field(default_factory=list[OSError])
    """Why, for each path in ``removed`` and ``left``."""

    @property
    def machine_error(self) -> OSError | None:
        """The first of ``errors`` that is this machine's (:data:`MACHINE_ERRNOS`), if any."""
        return next((error for error in self.errors if error.errno in MACHINE_ERRNOS), None)


def redact_files(root: Path, secrets: Iterable[str]) -> Redacted:
    """Replace every occurrence of ``secrets`` (exact values, from ``MIN_SECRET`` characters)
    in the regular files under ``root`` with ``[REDACTED]``; say what was done.

    Symbolic links are never followed and special files (pipes, sockets, devices) never
    opened. A file is rewritten only when it holds a secret, as a new file that replaces it
    (a hard link is never written through) with the same permissions. A file or directory the
    agent made unreadable or unlistable is the Lab's again: it is opened up for this and gets
    its permissions back after. A file that cannot be rewritten (a full disk, a name too long
    for the copy) or read is removed, which needs no space, or else emptied; what can be
    neither is ``left``. Nothing is raised: :class:`Redacted` says what failed, and why."""
    values = sorted(
        {s.encode("utf-8") for s in secrets if len(s) >= MIN_SECRET}, key=len, reverse=True
    )
    if not values:
        return Redacted()
    pattern = re.compile(b"|".join(re.escape(value) for value in values))
    return _Scan(Path(root), pattern, len(values[0])).run()


class _Scan:
    def __init__(self, root: Path, pattern: re.Pattern[bytes], longest: int) -> None:
        self.root = root
        self.pattern = pattern
        self.longest = longest
        self.result = Redacted()
        self.opened: dict[Path, int] = {}
        """Directories opened up for the scan, with their permissions before it (a parent
        always before its children)."""

    def run(self) -> Redacted:
        try:
            stack = [self.root]
            while stack:  # iterative: an agent's directories may be very deep
                directory = stack.pop()
                for entry in self._entries(directory):
                    path = Path(entry.path)
                    try:
                        mode = entry.stat(follow_symlinks=False).st_mode
                    except FileNotFoundError:
                        continue
                    except OSError as error:  # a name the Lab cannot even look at
                        self._discard(path, error)
                        continue
                    if stat.S_ISDIR(mode):
                        stack.append(path)
                    elif stat.S_ISREG(mode):
                        self._file(path, mode)
                    # A link is not followed; a pipe, socket or device holds nothing on disk.
        finally:
            for directory, mode in reversed(self.opened.items()):
                with contextlib.suppress(OSError):
                    os.chmod(directory, mode)
        return self.result

    def _entries(self, directory: Path) -> list[os.DirEntry[str]]:
        try:  # the root is the Lab's own; below it, nothing is followed
            mode = (os.stat if directory == self.root else os.lstat)(directory).st_mode
            if not stat.S_ISDIR(mode):
                return []
            if _LISTED & ~mode:
                # The agent made its directory unlistable (chmod 000, 0300): the Lab's again.
                with contextlib.suppress(OSError):  # not the Lab's: other permissions may do
                    self._open_up(directory, mode, _LISTED)
            with os.scandir(directory) as listing:
                return list(listing)
        except FileNotFoundError:
            return []
        except OSError as error:
            self.result.errors.append(error)
            self.result.left.append(f"{self._relative(directory)}/")
            return []

    def _open_up(self, directory: Path, mode: int, bits: int) -> None:
        """Give the Lab ``bits`` on a directory of its build for the scan."""
        self.opened.setdefault(directory, stat.S_IMODE(mode))
        os.chmod(directory, stat.S_IMODE(mode) | bits)

    def _file(self, path: Path, mode: int) -> None:
        try:
            self.result.count += self._rewrite(path, stat.S_IMODE(mode))
        except FileNotFoundError:
            return
        except OSError as error:
            self._discard(path, error)

    def _rewrite(self, path: Path, mode: int) -> int:
        """Replace the values in one regular file; return how many were replaced."""
        if not mode & stat.S_IRUSR:
            # The agent made its file unreadable: the Lab's again, for as long as it reads it.
            os.chmod(path, mode | stat.S_IRUSR)
        with os.fdopen(os.open(path, os.O_RDONLY | _NOT_FOLLOWED), "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                return 0
            if not _holds(source, self.pattern, self.longest):
                if not mode & stat.S_IRUSR:
                    os.fchmod(source.fileno(), mode)
                return 0
            source.seek(0)
            handle, name = self._temporary(path.parent)
            try:
                with os.fdopen(handle, "wb") as target:
                    count = _copy_redacted(source, target, self.pattern, self.longest)
                    target.flush()
                    os.fchmod(target.fileno(), mode)
                os.replace(name, path)
            except BaseException:
                _remove(name)
                raise
        return count

    def _temporary(self, directory: Path) -> tuple[int, str]:
        try:
            return tempfile.mkstemp(dir=directory, prefix=".r2s-redacting-")
        except PermissionError:
            # The agent made its own directory read-only: it is the Lab's build directory again.
            mode = os.lstat(directory).st_mode
            if not stat.S_ISDIR(mode):
                raise
            self._open_up(directory, mode, stat.S_IRWXU)
            return tempfile.mkstemp(dir=directory, prefix=".r2s-redacting-")

    def _discard(self, path: Path, error: OSError) -> None:
        """A file that could not be rewritten or read may still hold a value: remove it (which
        needs no disk space), or else empty it; what can be neither is left, and said."""
        self.result.errors.append(error)
        try:
            self._unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            if not _empty(path):
                self.result.left.append(self._relative(path))
                return
        self.result.removed.append(self._relative(path))

    def _unlink(self, path: Path) -> None:
        try:
            os.unlink(path)
        except PermissionError:
            mode = os.lstat(path.parent).st_mode
            if not stat.S_ISDIR(mode):
                raise
            self._open_up(path.parent, mode, stat.S_IRWXU)
            os.unlink(path)

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()


def _empty(path: Path) -> bool:
    """Empty a file that could not be removed, unless another name shares it: a hard link to
    a file elsewhere is never written through."""
    try:
        descriptor = os.open(path, os.O_WRONLY | _NOT_FOLLOWED)
    except OSError:
        return False
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            return False
        os.ftruncate(descriptor, 0)
        return True
    except OSError:
        return False
    finally:
        os.close(descriptor)


def _remove(name: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(name)


def _holds(source: BinaryIO, pattern: re.Pattern[bytes], longest: int) -> bool:
    tail = b""
    while chunk := source.read(CHUNK):
        data = tail + chunk
        if pattern.search(data):
            return True
        tail = data[max(0, len(data) - (longest - 1)) :]
    return False


def _copy_redacted(
    source: BinaryIO, target: BinaryIO, pattern: re.Pattern[bytes], longest: int
) -> int:
    """Copy ``source`` to ``target`` with every match of ``pattern`` replaced, chunk by chunk:
    a match that starts where a longer secret could still run past the chunk's end waits for
    the next chunk."""
    replacement = REPLACEMENT.encode("utf-8")
    count = 0
    pending = b""
    while True:
        chunk = source.read(CHUNK)
        data = pending + chunk
        safe = len(data) if not chunk else max(0, len(data) - (longest - 1))
        position = 0
        for match in pattern.finditer(data):
            if match.start() >= safe:
                break
            target.write(data[position : match.start()])
            target.write(replacement)
            position = match.end()
            count += 1
        cut = max(position, safe)
        target.write(data[position:cut])
        pending = data[cut:]
        if not chunk:
            return count


__all__ = [
    "MACHINE_ERRNOS",
    "PATTERNS",
    "REPLACEMENT",
    "Redacted",
    "Redactor",
    "file_secrets",
    "redact",
    "redact_files",
]
