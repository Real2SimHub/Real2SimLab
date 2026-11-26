"""File-system helpers: trees that sandboxed programs wrote, and durable cache entries.

A twin or an agent controls what it leaves behind: directories without write or read
permission, trees deeper than Python's recursion limit, symbolic links. Copying, collecting or
deleting such trees with ``shutil`` can raise (and abort a whole experiment), so these helpers
first give the owner (this user: sandboxes run with the host uid) full access to every
directory, then delete with the system's ``rm -rf``, which is iterative and never follows
symbolic links. A file such a program could have replaced (a symbolic link, a named pipe in
its place) is read only as a regular file, never through a link or by waiting on a pipe
(``read_regular``), and the Lab's own records beside its files are written as new files,
never through what it left under their names (``write_new``).

Caches (third-party code, downloaded data, indexes) must survive a crash or a power cut: a
rename can reach the disk before the data it names, leaving empty or truncated files under a
complete-looking name. Cache entries are therefore written to a temporary name, flushed
(``fsync``) and only then renamed (``atomic_write_*``, ``fsync_tree``), and readers check what
they reuse and fetch it again when it is damaged (``DamagedCacheWarning``).
"""

from __future__ import annotations

import contextlib
import errno
import functools
import json
import os
import re
import shutil
import stat
import subprocess
import uuid
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from errno import EDQUOT, EFBIG, ENOSPC
from pathlib import Path
from typing import Any, cast

from real2simlab.errors import SetupError


class DamagedCacheWarning(UserWarning):
    """A cache entry was damaged (empty, truncated, unreadable) and is being fetched again."""


def warn_damaged(path: Path, problem: str) -> None:
    """Say that the cache entry ``path`` is damaged and will be fetched or rebuilt again."""
    warnings.warn(
        f"the cached {path} is damaged ({problem}); fetching it again",
        DamagedCacheWarning,
        stacklevel=3,
    )


def cache_problem(path: Path, check: Callable[[Path], object] | None = None) -> str | None:
    """Why the cached file ``path`` cannot be reused, or ``None``: missing, empty, or (by its
    suffix) a JSON file that does not parse, a NumPy array (``.npy``) or archive (``.npz``)
    that does not read whole, a video that ffprobe cannot read (e.g. truncated before its
    index). What the entry holds (its fields, their types and shapes) only its reader knows:
    ``check``, when given, reads it as its reader does and raises ``ValueError`` saying what
    is wrong with it (written by another program, or by another version)."""
    path = Path(path)
    if not path.is_file():
        return "missing"
    if path.stat().st_size == 0:
        return "empty"
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif suffix in (".npz", ".npy"):
            import numpy as np

            # An archive for .npz, which holds the file open until closed; the array itself
            # for .npy, read whole (it is no context manager).
            loaded = np.load(path, allow_pickle=False)
            if hasattr(loaded, "files"):
                with loaded:
                    for name in loaded.files:
                        loaded[name]
                if suffix == ".npy":
                    return "an archive, not a NumPy array"
            elif suffix == ".npz":
                return "a single array, not a NumPy archive"
        elif suffix == ".mp4":
            from real2simlab.media import probe_video

            probe_video(path)
    except Exception as error:  # any failure to read it means it cannot be reused
        return f"{type(error).__name__}: {str(error)[:200]}"
    if check is not None:
        try:
            check(path)
        except ValueError as error:
            return str(error)[:200]
    return None


def remove_damaged(
    paths: Iterable[Path], check: Callable[[Path], object] | None = None
) -> list[Path]:
    """Remove each cached file that exists but cannot be reused (with a warning; ``check`` as
    for ``cache_problem``); return the files that need fetching (missing or removed)."""
    needed: list[Path] = []
    for path in paths:
        problem = cache_problem(path, check)
        if problem is None:
            continue
        if problem != "missing":
            warn_damaged(path, problem)
            Path(path).unlink(missing_ok=True)
        needed.append(Path(path))
    return needed


STORAGE_ERRNOS = frozenset({ENOSPC, EDQUOT, EFBIG})
"""A disk that is full, a quota reached, a file-size limit (``ulimit -f``): this machine's
storage, not the command's input."""

STORAGE_FIX = (
    "free space on that disk (or raise the file-size limit, `ulimit -f`), then run the same "
    "command again"
)
"""What to do when this machine's storage stopped a write (``storage_errno``)."""


def copy_failures(error: shutil.Error) -> list[tuple[str, str, str]]:
    """The files a directory copy (``shutil.copytree``) could not copy: (from, to, why)."""
    listed: object = error.args[0] if error.args else None
    if not isinstance(listed, list):
        return []
    return [
        (str(item[0]), str(item[1]), str(item[-1]))
        for item in cast(list[object], listed)
        if isinstance(item, tuple) and len(cast(tuple[object, ...], item)) >= 3
    ]


def os_error_text(error: OSError) -> str:
    """``error`` in one line; for a directory copy, the first file it could not copy and how
    many others, rather than its whole list of Python tuples."""
    failures = copy_failures(error) if isinstance(error, shutil.Error) else []
    if not failures:
        return str(error)
    source, target, why = failures[0]
    others = f" (and {len(failures) - 1} other file(s))" if len(failures) > 1 else ""
    return f"copying {source} to {target} failed: {why}{others}"


def storage_errno(error: BaseException) -> int | None:
    """The full disk, quota or file-size limit (an errno of ``STORAGE_ERRNOS``) that stopped
    ``error``: an ``OSError`` it is, was raised from or while handling, or a file a directory
    copy could not write for it; ``None`` when there is none."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, shutil.Error):
            for *_, why in copy_failures(current):
                found = re.match(r"\[Errno (\d+)\]", why)
                if found and int(found.group(1)) in STORAGE_ERRNOS:
                    return int(found.group(1))
        elif isinstance(current, OSError) and current.errno in STORAGE_ERRNOS:
            return current.errno
        hidden = current.__suppress_context__  # `raise ... from None`
        current = current.__cause__ or (None if hidden else current.__context__)
    return None


def fsync_file(path: Path) -> None:
    """Flush ``path``'s data to the disk."""
    with open(path, "rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    """Flush ``path``'s entries (names, renames) to the disk, where the platform allows it."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - e.g. Windows cannot open directories
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - some file systems refuse fsync on directories
        pass
    finally:
        os.close(descriptor)


def fsync_tree(root: Path) -> None:
    """Flush every regular file and directory under ``root`` (symbolic links not followed)."""
    for directory, _, files in os.walk(root):
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                fsync_file(path)
        fsync_directory(Path(directory))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so that a crash leaves either the old file or the new one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with open(partial, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        partial.replace(path)
    except BaseException:  # a full disk, Ctrl-C: no half-written file is left beside it
        with contextlib.suppress(OSError):
            partial.unlink()
        raise
    fsync_directory(path.parent)


def atomic_write_text(path: Path, text: str) -> None:
    """``atomic_write_bytes`` for UTF-8 text."""
    atomic_write_bytes(path, text.encode("utf-8"))


def durable_replace(partial: Path, target: Path) -> None:
    """Flush the finished file ``partial``, then rename it to ``target``."""
    fsync_file(partial)
    Path(partial).replace(target)
    fsync_directory(Path(target).parent)


def staging_dir(out: Path) -> Path:
    """A new, empty directory next to ``out`` (on the same file system) to build ``out`` in,
    held by this process (:func:`held_path`) until it is moved into place (:func:`replace_dir`)
    or removed (:func:`force_remove`). What a command killed outright left staged for the same
    ``out`` is removed first (:func:`remove_abandoned`)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    remove_abandoned(out.parent, re.escape(f".{out.name}.partial-"))
    return held_path(lambda part: out.with_name(f".{out.name}.partial-{part}"))


# A command builds a large result (a dataset, a checkpoint, an unpacked image, a download) under
# a temporary name beside it and removes that when it fails; a command killed outright (kill -9,
# the out-of-memory killer) runs no cleanup. So each such name is held, while its command runs,
# by a lock (``flock``) that the kernel releases when the process ends, however it ends, and
# bears a tag of the running system whose kernel holds it (its boot id): the next command
# removes what no process holds any more. Whether a process of another system (a disk that the
# nodes of a cluster share) still runs cannot be told from here, so what bears another tag is
# left alone; so is what this machine left before it last started (its boot id was another,
# which no other machine's can be told from).

_HELD: dict[str, int] = {}
"""What this process holds (:func:`held_path`): its path, and the descriptor whose lock holds it."""


@functools.cache
def machine_tag() -> str | None:
    """Twelve hexadecimal digits naming this running system (from its kernel's boot id), which
    every name :func:`held_path` gives bears; ``None`` where there is no boot id (names then
    bear none, and :func:`remove_abandoned` never removes anything)."""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii")
    except (OSError, ValueError):
        return None
    digits = boot.strip().replace("-", "").lower()
    return digits[:12] if re.fullmatch(r"[0-9a-f]{32}", digits) else None


def _lock(path: Path) -> tuple[int | None, bool]:
    """An exclusive lock (``flock``) on ``path``, a directory or a file, taken at once and never
    through a symbolic link: (its descriptor, ``True``); (``None``, ``True``) when another
    process holds it or it is not there any more under that name; (``None``, ``False``) when
    this system cannot lock it (then no process can, and nothing is taken for abandoned)."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        return None, False
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:  # gone (or a symbolic link now): not this command's to judge
        return None, True
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None, True
    except OSError:  # a file system that does not lock
        os.close(descriptor)
        return None, False
    try:
        same = os.path.samestat(os.fstat(descriptor), os.lstat(path))
    except OSError:
        same = False
    if not same:  # removed, or replaced, before the lock was taken
        os.close(descriptor)
        return None, True
    return descriptor, True


def held_path(make: Callable[[str], Path], *, directory: bool = True, mode: int = 0o777) -> Path:
    """Create a new directory (an empty file without ``directory``) at ``make(part)``, ``part``
    being this machine's tag and eight random hexadecimal digits, with ``mode`` (less the
    umask; ``0o700`` as ``tempfile.mkdtemp`` gives), and hold it for this process until
    :func:`release`, or until the process ends however it ends, so that
    :func:`remove_abandoned` never takes it while its command runs. A name the tag makes too
    long for the file system (a target's name of more than about 225 bytes) is made without it,
    as long as names were before tags: like every name without a tag (where there is no boot
    id), nothing takes it for abandoned, even once its command was killed."""
    tag = machine_tag()
    attempts = 0
    while True:
        random = uuid.uuid4().hex[:8]
        path = make(f"{tag}-{random}" if tag else random)
        try:
            if directory:
                path.mkdir(mode=mode)
            else:
                os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode & 0o666))
        except FileExistsError:  # the same name drawn again: draw another
            attempts += 1
            if attempts < 100:
                continue
            raise
        except OSError as error:
            if error.errno != errno.ENAMETOOLONG or tag is None:
                raise
            tag = None  # 13 bytes shorter: what fitted before tags fits again
            continue
        if tag is None:
            return path
        descriptor, lockable = _lock(path)
        if descriptor is not None:
            _HELD[os.fspath(path)] = descriptor
            return path
        if not lockable:
            return path
        # another command's cleanup took it in the instant before the lock: take another name
        attempts += 1
        if attempts == 100:
            raise SetupError(
                f"cannot keep a new working copy in {path.parent}: another command removes "
                "each one as it is made",
                fix="stop the other r2s commands that write there, then run the same command again",
            )


def release(path: Path) -> None:
    """Stop holding ``path`` (:func:`held_path`): it was moved into place, or removed."""
    descriptor = _HELD.pop(os.fspath(path), None)
    if descriptor is not None:
        os.close(descriptor)


def remove_abandoned(parent: Path, before: str, after: str = "") -> list[Path]:
    """Remove from ``parent`` what commands of this machine held (:func:`held_path`) and no
    process holds any more, left by a command killed outright: names ``<before><machine
    tag>-<8 hexadecimal digits><after>`` (``before`` and ``after`` are regular expressions).
    Says so on standard error; returns what it removed. Nothing else is touched: what a running
    command holds, what another machine's command left (whether it still runs cannot be told
    here), what this system cannot lock, a symbolic link, any other name."""
    import sys

    tag = machine_tag()
    if tag is None:
        return []
    name = re.compile(f"{before}{tag}-[0-9a-f]{{8}}{after}")
    try:
        with os.scandir(parent) as entries:
            found = [Path(entry.path) for entry in entries if name.fullmatch(entry.name)]
    except OSError:  # a directory that cannot be listed holds nothing to remove
        return []
    removed: list[Path] = []
    for path in found:
        descriptor, _ = _lock(path)
        if descriptor is None:
            continue
        try:
            problem = force_remove(path)
        finally:
            os.close(descriptor)
        if problem is None:
            removed.append(path)
            print(
                f"note: removed {path}, left by a command that was killed before it finished",
                file=sys.stderr,
            )
    return removed


def _spec_of(value: object, name: str) -> bool:
    """Whether ``value`` names the specification object ``name`` (``r2s-task``) at some version,
    ``r2s-task/1.0`` (another release of the Lab may write another version)."""
    if not isinstance(value, str):
        return False
    return re.fullmatch(rf"{re.escape(name)}/\d+\.\d+", value) is not None


def _by_the_lab(data: dict[str, Any]) -> bool:
    """Whether ``data`` names the version of the Lab that wrote it (``lab_version``: ``1.0.0``)."""
    version = data.get("lab_version")
    return isinstance(version, str) and re.match(r"\d+\.\d+", version) is not None


def _experiment(data: dict[str, Any]) -> bool:
    config = data.get("config")
    return (
        _by_the_lab(data)
        and isinstance(config, dict)
        and _spec_of(cast("dict[str, Any]", config).get("spec"), "r2s-experiment")
    )


def _run_status(data: dict[str, Any]) -> bool:
    # what orchestrate.layout writes: RUNNING with started_at, then COMPLETE with finished_at
    return (
        _by_the_lab(data)
        and data.get("status") in ("running", "complete")
        and any(isinstance(data.get(key), str) for key in ("started_at", "finished_at"))
    )


def _export(data: dict[str, Any]) -> bool:
    return (
        _by_the_lab(data)
        and isinstance(data.get("counts"), dict)
        and all(isinstance(data.get(key), list) for key in ("tasks", "runs", "against"))
    )


LAB_RECORDS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "task.toml": lambda data: _spec_of(data.get("spec"), "r2s-task"),
    "training.json": lambda data: _spec_of(data.get("spec"), "r2s-training") and _by_the_lab(data),
    "experiment.json": _experiment,
    "status.json": _run_status,
    "export.json": _export,
}
"""The records that make a directory the Lab's, by file name, and what each one the Lab writes
holds, in every version: a task's ``task.toml`` (``r2s-task``), a checkpoint's
``training.json`` (``r2s-training``), a run directory's ``experiment.json`` (the experiment,
``r2s-experiment``) and ``state/status.json``, an exported dataset's ``export.json``."""


def lab_record(path: Path) -> bool:
    """Whether ``path`` is a record the Lab wrote, judged by all of its content: a JSON object (a
    TOML table for ``.toml``) holding what every record of its kind holds (:data:`LAB_RECORDS`):
    the specification it follows, the version of the Lab that wrote it, the fields it always
    has. A file of the same name the Lab did not write (the user's own ``experiment.json``,
    another tool's record with some ``lab_version``, a record of another ``spec``) is not one,
    so the Lab never takes its directory for one of its own, to write into or replace."""
    import tomllib

    path = Path(path)
    kind = LAB_RECORDS.get(path.name)
    if kind is None:
        return False
    try:
        text = path.read_text(encoding="utf-8")
        data: object = tomllib.loads(text) if path.suffix == ".toml" else json.loads(text)
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and kind(cast("dict[str, Any]", data))


def has_mark(path: Path, mark: str) -> bool:
    """Whether the file ``path`` holds ``mark`` as :func:`write_mark` writes it, the text that
    makes a directory the Lab's while one of its commands works in it (``.r2s-check``): all of
    it, or its start, empty included, which is what a write stopped by kill -9 or a power cut
    leaves. Marks bear names only the Lab gives, so a cut-short one is still the Lab's; a file
    of that name with other content is not."""
    expected = mark.encode("utf-8")
    try:
        content = read_regular(path)  # never through a link, never waiting on a pipe
    except OSError:
        return False
    if content is None:
        return False
    return expected.startswith(content[: len(expected) + 1])


def write_mark(path: Path, mark: str) -> None:
    """Write ``mark`` to ``path`` (see :func:`has_mark`), a regular file: never through a
    symbolic link (which could name a file anywhere) or into a pipe. A write that fails (a full
    disk, a file-size limit, Ctrl-C) removes the file again: the directory is left as it was
    before, not with a mark cut short beside its files."""
    path = Path(path)
    if (path.is_symlink() or path.exists()) and not is_regular(path):
        raise SetupError(
            f"{path} is a link or not a regular file: the Lab writes its mark there only as a "
            "file of its own",
            fix=f"remove it (rm {path}), then run the same command again",
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _NOFOLLOW | _NONBLOCK | _CLOEXEC
    try:
        descriptor = os.open(path, flags, 0o666)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(mark.encode("utf-8"))
    except BaseException:
        if is_regular(path):
            with contextlib.suppress(OSError):
                path.unlink()
        raise


def check_replaceable(out: Path, *, overwrite: bool, marker: str, what: str) -> None:
    """Raise ``SetupError`` unless ``out`` is absent, or may be replaced: ``overwrite`` and it
    is a directory that is empty or holds ``marker`` as the Lab wrote it (the record that makes
    it ``what``, see :func:`lab_record`)."""
    out = Path(out)
    recover_dir(out, marker=marker, what=what)
    if not out.exists() and not out.is_symlink():
        return
    if not overwrite:
        raise SetupError(f"{out} exists", fix="choose a new directory or pass --overwrite")
    if out.is_symlink() or not out.is_dir():
        raise SetupError(
            f"{out} is not a directory, so it is not {what}", fix="choose another directory"
        )
    if lab_record(out / marker) or not any(out.iterdir()):
        return
    has = "its " + marker + " was not written by the Lab" if (out / marker).exists() else ""
    raise SetupError(
        f"{out} is not {what} ({has or 'it has no ' + marker}); --overwrite replaces only {what}",
        fix="choose a new directory, or empty this one yourself",
    )


def _exchange(first: Path, second: Path) -> bool:
    """Swap two existing paths atomically (Linux ``renameat2(RENAME_EXCHANGE)``, macOS
    ``renamex_np(RENAME_SWAP)``); ``False`` when this system or file system cannot."""
    import ctypes
    import ctypes.util
    import errno
    import sys

    a, b = os.fsencode(first), os.fsencode(second)
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    except OSError:
        return False
    if sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        arguments: tuple[object, ...] = (-100, a, -100, b, 2)  # AT_FDCWD, RENAME_EXCHANGE
    elif sys.platform == "darwin":
        function = getattr(libc, "renamex_np", None)
        arguments = (a, b, 0x2)  # RENAME_SWAP
    else:
        return False
    if function is None:
        return False
    if function(*arguments) == 0:
        return True
    code = ctypes.get_errno()
    if code in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV):
        return False
    raise OSError(code, os.strerror(code), str(first))


def _left_beside(out: Path, kind: str) -> list[Path]:
    """What :func:`replace_dir` left beside ``out``, by the exact name it gives:
    ``.<out>.old-<8 hex digits>`` (``kind`` ``old``, a previous version moved aside) or
    ``.<out>.trash-<8 hex digits>`` (``trash``, one being removed). Anything else there,
    whatever its name, is not the Lab's."""
    name = re.compile(rf"\.{re.escape(out.name)}\.{kind}-[0-9a-f]{{8}}")
    try:
        return [path for path in out.parent.iterdir() if name.fullmatch(path.name)]
    except OSError:  # a directory that cannot be listed holds nothing it can find
        return []


def _backups(out: Path) -> list[Path]:
    """Earlier versions of ``out`` that :func:`replace_dir` moved aside, newest first."""
    return sorted(_left_beside(out, "old"), key=lambda p: p.stat().st_mtime, reverse=True)


def recover_dir(out: Path, *, marker: str, what: str) -> None:
    """Put back the previous version of ``out`` if a replacement was interrupted between moving
    it aside and moving the new one in (only on systems without an atomic swap). Says so on
    standard error; does nothing when ``out`` exists or there is no complete backup (one whose
    ``marker`` is the Lab's record, see :func:`lab_record`)."""
    import sys

    out = Path(out)
    if not out.parent.is_dir():
        return
    for trash in _left_beside(out, "trash"):  # removals that were interrupted
        force_remove(trash)
    if out.exists() or out.is_symlink():
        return
    for backup in _backups(out):
        if lab_record(backup / marker):
            backup.replace(out)
            print(
                f"note: restored {what} {out} from {backup.name}: replacing it was interrupted",
                file=sys.stderr,
            )
            return


def replace_dir(staging: Path, out: Path, *, overwrite: bool, marker: str, what: str) -> None:
    """Move the finished directory ``staging`` to ``out``.

    An existing ``out`` is replaced (``overwrite``) only if it is empty or holds ``marker``
    (see :func:`check_replaceable`), and only now that the new one is complete. Where the system
    can, the two are swapped in one atomic step, so a reader always finds a complete version at
    ``out``, even if this process is killed. Elsewhere the old one is moved aside first; if the
    process dies before the new one is in place, :func:`recover_dir` (run by every reader and
    writer of ``out``) puts the old one back.
    """
    staging, out = Path(staging), Path(out)
    recover_dir(out, marker=marker, what=what)
    check_replaceable(out, overwrite=overwrite, marker=marker, what=what)
    if not out.exists() and not out.is_symlink():
        staging.replace(out)
        release(staging)  # what was staged is ``out`` now
        fsync_directory(out.parent)
        return
    if _exchange(staging, out):
        release(staging)
        fsync_directory(out.parent)
        force_remove(staging)  # now the old version
        return
    old = out.with_name(f".{out.name}.old-{uuid.uuid4().hex[:8]}")
    out.replace(old)
    try:
        staging.replace(out)
    except BaseException:
        old.replace(out)
        raise
    release(staging)
    fsync_directory(out.parent)
    _discard(old, out)


def _discard(old: Path, out: Path) -> None:
    """Remove a replaced version of ``out``. It is renamed first, so that an interrupted
    removal never leaves a half-deleted directory that :func:`recover_dir` would take for a
    complete backup."""
    trash = out.with_name(f".{out.name}.trash-{uuid.uuid4().hex[:8]}")
    old.replace(trash)
    force_remove(trash)


@contextmanager
def exclusive_use(directory: Path, fix: str) -> Iterator[None]:
    """Hold ``directory`` (which exists) for this command while the block runs: a second
    command given the same directory (the first still running, or still stopping its
    programs) is refused instead of clearing what the first is writing. The lock is on the
    directory itself and ends with the command, however it ends."""
    import fcntl

    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SetupError(f"another r2s command is using {directory}", fix=fix) from None
        yield
    finally:
        os.close(descriptor)


def clashing_input(out: Path, inputs: Iterable[Path]) -> Path | None:
    """The first of ``inputs`` that ``out`` is, contains or lies inside, if any.

    Writing ``out`` (and replacing it) must never touch what is being read.
    """
    target = Path(out).resolve()
    for given in inputs:
        source = Path(given).resolve()
        if target == source or target.is_relative_to(source) or source.is_relative_to(target):
            return Path(given)
    return None


def make_owner_accessible(root: Path) -> None:
    """Give the owner read, write and search permission on every directory under ``root``.

    Symbolic links are neither followed nor changed. Errors are ignored: whatever stays
    inaccessible is reported by the caller's next step.
    """
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        return
    chmod = shutil.which("chmod")
    if chmod is not None:
        # GNU and BSD chmod -R ignore symbolic links met during the traversal.
        subprocess.run(
            [chmod, "-R", "u+rwX", str(root)],
            capture_output=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        return
    stack = [root]
    while stack:  # iterative: no recursion limit
        directory = stack.pop()
        try:
            mode = os.stat(directory, follow_symlinks=False).st_mode
            os.chmod(directory, stat.S_IMODE(mode) | stat.S_IRWXU)
            with os.scandir(directory) as entries:
                stack += [Path(e.path) for e in entries if e.is_dir(follow_symlinks=False)]
        except OSError:
            continue


def copy_tree(source: Path, target: Path, *, ignore: Any = None, exist_ok: bool = False) -> Path:
    """Copy the tree ``source`` to ``target`` (``shutil.copytree``), the copy readable, writable
    and searchable by its owner: a tree that is read-only where it comes from (the Lab's own
    files in a read-only installation, such as a Nix store or a read-only share, or a user's
    read-only task) would otherwise be read-only in the user's output, where the next command,
    ``--overwrite`` or ``rm -rf`` fails on it. Executable bits (part of a twin's fingerprint) are
    kept."""
    shutil.copytree(source, target, ignore=ignore, dirs_exist_ok=exist_ok)
    make_owner_accessible(target)
    return target


def force_remove(path: Path) -> str | None:
    """Remove ``path`` whatever a sandboxed program did to it; return a problem, or ``None``.
    What this process held there (:func:`staging_dir`) is not held any more."""
    problem = _force_remove(Path(path))
    release(path)
    return problem


def _force_remove(path: Path) -> str | None:
    if not (path.exists() or path.is_symlink()):
        return None
    if path.is_symlink() or not path.is_dir():
        try:
            path.unlink()
        except OSError as error:
            return f"cannot remove {path}: {error}"
        return None
    make_owner_accessible(path)
    rm = shutil.which("rm")
    if rm is not None:
        subprocess.run(
            [rm, "-rf", str(path)], capture_output=True, check=False, stdin=subprocess.DEVNULL
        )
    else:  # pragma: no cover - every supported platform has rm
        _remove_tree(path)
    if path.exists() or path.is_symlink():
        return f"cannot remove {path}"
    return None


def _remove_tree(root: Path) -> None:
    """Delete a tree without recursion: files on the way down, directories children-first."""
    directories: list[Path] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        directories.append(directory)
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    else:
                        os.unlink(entry.path)
        except OSError:
            continue
    for directory in reversed(directories):
        try:
            os.rmdir(directory)
        except OSError:
            continue


# --- files a sandboxed program could have replaced ---------------------------------------------

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


def _errnos(*names: str) -> frozenset[int]:
    import errno

    return frozenset(getattr(errno, name) for name in names if hasattr(errno, name))


_NOT_READABLE = _errnos(
    "ENOENT", "ENOTDIR", "ELOOP", "EMLINK", "EACCES", "EPERM", "ENXIO", "ENODEV", "EOPNOTSUPP",
    "EAGAIN", "EWOULDBLOCK", "EISDIR", "ENAMETOOLONG",
)  # fmt: skip
"""What opening a name to read it fails with when the name is not a regular file this user may
read (missing, a symbolic link, a socket, a device, no permission), rather than because this
machine failed (an I/O error)."""


def is_regular(path: Path) -> bool:
    """Whether ``path`` itself is a regular file: not a symbolic link (to one or not), a pipe, a
    socket, a device or a directory. For names a sandboxed program could have replaced."""
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except (OSError, ValueError):
        return False


def read_regular(path: Path) -> bytes | None:
    """The bytes of ``path`` when it is a regular file; ``None`` when it is missing, a symbolic
    link, a pipe, a socket, a device or a directory, or this user may not read it.

    For a file in a directory a sandboxed program could write (an agent's workspace and log
    directory, once its sandbox has ended): it may have left anything under a name the Lab
    reads. Following a link would read, on the host, what the sandbox kept from it (a task's
    hidden data, this process's own environment through ``/proc/self``); a pipe would keep the
    Lab waiting forever. So the name is opened without following a link (``O_NOFOLLOW``) or
    waiting (``O_NONBLOCK``), and read only if what was opened is a regular file (``fstat``).
    A failure of this machine (an I/O error) is raised."""
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):  # also where there is no O_NOFOLLOW
            return None
        descriptor = os.open(path, os.O_RDONLY | _NOFOLLOW | _NONBLOCK | _CLOEXEC)
    except OSError as error:
        if error.errno in _NOT_READABLE:
            return None
        raise
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            return None
        return stream.read()


def links_and_special(directory: str, names: list[str]) -> set[str]:
    """For ``shutil.copytree(..., ignore=links_and_special)``: the entries of ``directory``
    that are neither regular files nor directories (symbolic links, pipes, sockets, devices),
    left out of a copy of a tree a sandboxed program wrote: copying them would follow the link
    on the host or wait on the pipe forever."""
    left: set[str] = set()
    for name in names:
        try:
            mode = os.lstat(os.path.join(directory, name)).st_mode
        except OSError:
            left.add(name)
            continue
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            left.add(name)
    return left


def write_new(path: Path, data: bytes) -> None:
    """Write ``data`` as a new regular file named ``path``, in a directory a sandboxed program
    could write, once that program has ended (the Lab's own records in an agent's log
    directory, at the places the specification gives them). Whatever the program left under
    that name (a symbolic link, a pipe, a directory) is removed first, never written through:
    the file is created anew (``O_CREAT | O_EXCL``, which never follows a link)."""
    path = Path(path)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISDIR(mode):
            problem = force_remove(path)
            if problem is not None:
                raise OSError(problem)
        else:
            path.unlink()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC
    with os.fdopen(os.open(path, flags, 0o666), "wb") as stream:
        stream.write(data)
