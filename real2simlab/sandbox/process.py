"""Running a host process with a timeout, stopping its whole process group when it expires,
when the user interrupts (Ctrl-C) or when the experiment running it is cancelled. What it
prints is kept in a log of bounded size."""

from __future__ import annotations

import contextlib
import errno
import os
import selectors
import signal
import stat
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import BinaryIO

POLL_S = 0.5
"""How often a waiting run checks whether it has been cancelled."""

PROBE_TIMEOUT_S = 120.0
"""A short probe of a tool (``docker image inspect``, ``git rev-parse``) that has not answered
after this long hangs (a stopped Docker daemon, a dead network mount)."""

LOG_LIMIT_BYTES = 16 * 1024 * 1024
"""The most of a process's output that ``run_logged`` keeps in its log: the beginning, and the
last ``LOG_TAIL_BYTES`` once the process has ended, with a line between them saying how much was
left out. A program that prints in a loop would otherwise fill the disk. A file-size limit
(``ulimit -f``) lowers it, so the limit never stops a program for what it printed."""

LOG_TAIL_BYTES = 1024 * 1024
"""How much of the end of a long output its log keeps (where a crash says why)."""

STOP_GRACE_S = 5.0
"""How long a process that is stopped (its time limit, Ctrl-C, a cancelled experiment) has to
end after ``SIGTERM`` before its process group is killed: the local backend's supervisor stops
everything the program started in that time."""

_RESERVED = 4096
"""Bytes kept under the file-size limit for the Lab's own lines in a log."""

NO_DOCKER_LOG = ("--log-driver", "none")
"""``docker run`` options that turn Docker's own log of a container off. What the container
prints reaches the Lab through the attached client, in a bounded log; Docker's default log
driver would also keep all of it, without a size limit, on the daemon's disk until the
container is removed."""

_cancelled = threading.Event()


def cancel_all() -> None:
    """Stop every process ``run_logged`` is waiting for, in every thread, and refuse new ones
    until ``reset_cancel``: each raises ``KeyboardInterrupt`` in its own thread, so no result
    is written for a run that was stopped. The processes run in their own session (a timeout
    kills their whole group), so the terminal's Ctrl-C does not reach them by itself."""
    _cancelled.set()


def reset_cancel() -> None:
    _cancelled.clear()


def cancelled() -> bool:
    return _cancelled.is_set()


def run_logged(
    command: Sequence[str],
    log_path: Path,
    timeout_s: float,
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    on_timeout: Callable[[], None] | None = None,
    user: tuple[int, int] | None = None,
) -> tuple[int | None, bool, float]:
    """Run ``command`` with stdout and stderr in ``log_path``.

    Returns ``(exit_code, timed_out, seconds)``; ``exit_code`` is ``None`` after a timeout.
    ``on_timeout`` runs before the process group is stopped, e.g. to stop a container.
    ``user`` is a ``(uid, gid)`` to run the command as, with no supplementary groups (needs
    root). The output reaches the log through a pipe, at most ``LOG_LIMIT_BYTES`` of it.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if _cancelled.is_set():
        raise KeyboardInterrupt
    start = time.monotonic()
    with log_path.open("ab") as stream, selectors.DefaultSelector() as selector:
        log = _BoundedLog(stream)
        reader, writer = os.pipe()
        try:
            try:
                process = subprocess.Popen(
                    list(command),
                    stdin=subprocess.DEVNULL,
                    stdout=writer,
                    stderr=subprocess.STDOUT,
                    env=None if env is None else dict(env),
                    cwd=cwd,
                    start_new_session=True,
                    user=None if user is None else user[0],
                    group=None if user is None else user[1],
                    extra_groups=None if user is None else [],
                )
            finally:
                os.close(writer)  # the process holds its own copy
            selector.register(reader, selectors.EVENT_READ)
            deadline = start + timeout_s
            try:
                ended = False  # the end of the output was read
                while True:
                    wait = max(0.0, min(POLL_S, deadline - time.monotonic()))
                    if ended:
                        with contextlib.suppress(subprocess.TimeoutExpired):
                            process.wait(timeout=wait)
                    elif wait > 0:  # at the deadline, the stop below reads what is left
                        ended = _copy(selector, reader, log, wait)
                    code = process.poll()
                    if code is not None:
                        if not ended:  # what is left (a process it started may hold the pipe)
                            _copy(selector, reader, log, 0.0)
                        log.close()
                        return code, False, time.monotonic() - start
                    if _cancelled.is_set():
                        raise KeyboardInterrupt
                    if time.monotonic() >= deadline:
                        raise subprocess.TimeoutExpired(list(command), timeout_s)
            except subprocess.TimeoutExpired:
                _stop(process, on_timeout)
                _copy(selector, reader, log, 0.0)
                log.close(f"\n[r2s] killed after {timeout_s:g} s\n")
                return None, True, time.monotonic() - start
            except BaseException:  # Ctrl-C, or the experiment was cancelled: stop it, then stop
                _stop(process, on_timeout)
                with contextlib.suppress(Exception):
                    _copy(selector, reader, log, 0.0)
                log.close("\n[r2s] interrupted\n")
                raise
        finally:
            os.close(reader)


class _BoundedLog:
    """A process's output in its log, within ``LOG_LIMIT_BYTES`` and under the file-size limit:
    the beginning as it arrives and, at ``close``, the end kept meanwhile, after a line saying
    how much was left out between them."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        room = LOG_LIMIT_BYTES
        size_limit = _file_size_limit()
        if size_limit is not None:
            room = min(room, size_limit)
        room = max(0, room - _RESERVED - stream.tell())  # the log may hold something already
        self.tail_room = min(LOG_TAIL_BYTES, room // 2)
        self.head_room = room - self.tail_room
        self.head_size = self.head_room
        self.tail: deque[bytes] = deque()
        self.tail_size = 0
        self.left_out = 0
        self.failed = False

    def write(self, data: bytes) -> None:
        if self.head_room > 0:
            head = data[: self.head_room]
            self.head_room -= len(head)
            self._put(head)
            data = data[len(head) :]
        if not data:
            return
        self.tail.append(data)
        self.tail_size += len(data)
        while self.tail and self.tail_size - len(self.tail[0]) >= self.tail_room:
            dropped = self.tail.popleft()
            self.tail_size -= len(dropped)
            self.left_out += len(dropped)

    def close(self, note: str = "") -> None:
        """Write the end that was kept, after a line about what was left out, then ``note``."""
        excess = self.tail_size - self.tail_room
        if excess > 0 and self.tail:
            self.tail[0] = self.tail[0][excess:]
            self.left_out += excess
        if self.left_out:
            self._put(
                f"\n[r2s] {self.left_out} bytes of output left out here (a log keeps the first "
                f"{self.head_size} and the last {self.tail_room} bytes)\n".encode()
            )
        for chunk in self.tail:
            self._put(chunk)
        self.tail.clear()
        if note:
            self._put(note.encode())

    def _put(self, data: bytes) -> None:
        if self.failed or not data:
            return
        try:
            self.stream.write(data)
            self.stream.flush()
        except OSError:  # a full disk: the process goes on, and its result says what it did
            self.failed = True


def _file_size_limit() -> int | None:
    """This process's file-size limit (``ulimit -f``) in bytes; ``None`` without one."""
    import resource

    limit = resource.getrlimit(resource.RLIMIT_FSIZE)[0]
    return None if limit == resource.RLIM_INFINITY else limit


def file_room(limit: int) -> int:
    """``limit``, lowered to fit this process's file-size limit (``ulimit -f``) with room for
    the Lab's own lines: the most a file of kept output may hold."""
    size_limit = _file_size_limit()
    return limit if size_limit is None else max(0, min(limit, size_limit - _RESERVED))


def _copy(selector: selectors.BaseSelector, reader: int, log: _BoundedLog, seconds: float) -> bool:
    """Copy into ``log`` what ``reader`` delivers within ``seconds`` (``0``: what it holds now,
    for at most a second); whether its end was reached."""
    end = time.monotonic() + (seconds or 1.0)
    while True:
        timeout = max(0.0, end - time.monotonic()) if seconds else 0.0
        if not selector.select(timeout):
            return False
        data = os.read(reader, 1 << 16)
        if not data:
            return True
        log.write(data)
        if time.monotonic() >= end:
            return False


_NOTE_ROOM = 512
"""Bytes kept under a limit for the line saying what was left out."""


class KeptLines:
    """A stream within ``limit`` bytes, in whole lines: its first lines as they arrive and its
    last lines (at most ``tail`` bytes), with a line between them saying how much was left out.
    A line is never cut: one that does not fit is left out whole (a key cut in two would escape
    redaction). No more than the limit is held in memory either."""

    def __init__(self, limit: int, tail: int = LOG_TAIL_BYTES) -> None:
        self.limit = limit
        room = max(0, limit - _NOTE_ROOM)
        self.tail_room = min(tail, room // 2)
        self.head_room = room - self.tail_room
        self.head = bytearray()
        self.head_open = True
        self.tail: deque[bytes] = deque()
        self.tail_size = 0
        self.total = 0
        self.before_tail = b""
        """The last byte left out before the tail."""

    def add(self, data: bytes) -> None:
        self.total += len(data)
        if self.head_open:
            room = self.head_room - len(self.head)
            self.head += data[:room]
            data = data[room:]
            if not data:
                return
            self.head_open = False
        self.tail.append(data)
        self.tail_size += len(data)
        while self.tail and self.tail_size - len(self.tail[0]) >= self.tail_room:
            first = self.tail.popleft()
            self.tail_size -= len(first)
            self.before_tail = first[-1:]

    def value(self) -> bytes:
        tail = b"".join(self.tail)
        if self.total <= self.head_room + self.tail_room:  # all of it: nothing was left out
            return bytes(self.head) + tail
        excess = max(0, len(tail) - self.tail_room)
        before = tail[excess - 1 : excess] if excess else self.before_tail
        return _whole_lines(bytes(self.head), tail[excess:], before, self.total, self.limit)


def _whole_lines(head: bytes, tail: bytes, before: bytes, total: int, limit: int) -> bytes:
    """The beginning ``head`` and the end ``tail`` of ``total`` bytes, each cut to whole lines
    (``before`` is the byte before ``tail``), with a line between them saying what was left
    out."""
    head = head[: head.rfind(b"\n") + 1]
    if before != b"\n":  # the tail begins inside a line
        newline = tail.find(b"\n")
        tail = tail[newline + 1 :] if newline >= 0 else b""
    note = (
        f"[r2s] {total - len(head) - len(tail)} bytes left out here (kept: the first and the "
        f"last whole lines, at most {limit} bytes)\n"
    )
    return head + note.encode() + tail


def read_kept(path: Path, limit: int, tail: int = LOG_TAIL_BYTES) -> bytes:
    """A regular file's content within ``limit`` bytes, kept as ``KeptLines`` keeps a stream:
    all of a file that fits; of a longer one, its first and last whole lines, with a line
    between them saying how much was left out. Only what is kept is read. A link, or anything
    else that is not a regular file (a named pipe would never end), raises ``OSError``."""

    def opener(name: str, flags: int) -> int:
        return os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK)

    with open(path, "rb", opener=opener) as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise OSError(errno.EINVAL, "not a regular file", str(path))
        kept = KeptLines(limit, tail)
        room = kept.head_room + kept.tail_room
        data = stream.read(room + 1)
        if len(data) <= room:
            return data
        size = max(os.fstat(stream.fileno()).st_size, len(data))
        stream.seek(size - kept.tail_room - 1)
        before = stream.read(1)
        end = stream.read(kept.tail_room)
    return _whole_lines(data[: kept.head_room], end, before, size, limit)


def run_captured(
    command: Sequence[str],
    *,
    started: Callable[[], None] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    timeout_s: float | None = None,
    grace_s: float = 0.0,
    keep: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """``subprocess.run(command, capture_output=True, text=True)`` that an experiment can stop:
    the process runs in its own session, and ``cancel_all`` (an interrupted experiment, from
    any thread) or Ctrl-C kills its whole group and raises ``KeyboardInterrupt`` here. After
    ``timeout_s`` the group is killed and ``subprocess.TimeoutExpired`` raised. With
    ``grace_s``, the group is first asked to stop (``SIGTERM``: a tool that started containers
    removes them) and killed if it has not stopped after that many seconds.

    With ``keep``, stdout and stderr each keep at most that many bytes, in whole lines (see
    ``KeptLines``), read as they arrive and decoded as UTF-8: a tool that prints without end
    then fills neither this process's memory nor the disk its output is written to.

    ``started``, when given, is called once the process exists (it was spawned), before it is
    waited for: never for a command that could not be spawned or was stopped before (an
    experiment already cancelled). If it raises, the process group is stopped as on any error
    here (within moments of its start) and the exception propagates."""
    if _cancelled.is_set():
        raise KeyboardInterrupt
    if keep is not None:
        return _run_kept(list(command), env, cwd, timeout_s, grace_s, keep, started)
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=None if env is None else dict(env),
        cwd=cwd,
        start_new_session=True,
    )
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    try:
        if started is not None:
            started()
        while True:
            try:
                out, err = process.communicate(timeout=POLL_S)
                return subprocess.CompletedProcess(list(command), process.returncode, out, err)
            except subprocess.TimeoutExpired:
                if _cancelled.is_set():
                    raise KeyboardInterrupt from None
                if deadline is not None and time.monotonic() >= deadline:
                    raise
    except BaseException:
        if grace_s > 0:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with contextlib.suppress(Exception):
                process.communicate(timeout=grace_s)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            process.communicate()
        raise


def _run_kept(
    command: list[str],
    env: Mapping[str, str] | None,
    cwd: Path | None,
    timeout_s: float | None,
    grace_s: float,
    keep: int,
    started: Callable[[], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """``run_captured`` with ``keep``: each stream read into a ``KeptLines``; ``started`` is
    called once the process exists, as ``run_captured`` calls it."""
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=None if env is None else dict(env),
        cwd=cwd,
        start_new_session=True,
    )
    assert process.stdout is not None and process.stderr is not None
    out, err = KeptLines(keep), KeptLines(keep)
    streams = {process.stdout.fileno(): out, process.stderr.fileno(): err}
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    with process, selectors.DefaultSelector() as selector:
        for descriptor in streams:
            selector.register(descriptor, selectors.EVENT_READ)

        def read(until: float) -> None:
            """Keep what arrives until both streams have ended, or until ``until``."""
            while selector.get_map() and time.monotonic() < until:
                for key, _ in selector.select(max(0.0, min(POLL_S, until - time.monotonic()))):
                    data = os.read(key.fd, 1 << 16)
                    if data:
                        streams[key.fd].add(data)
                    else:
                        selector.unregister(key.fd)

        try:
            if started is not None:  # once the process exists (see run_captured)
                started()
            while selector.get_map() or process.poll() is None:
                if _cancelled.is_set():
                    raise KeyboardInterrupt
                if deadline is not None and time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, timeout_s or 0.0)
                if selector.get_map():
                    read(time.monotonic() + POLL_S)
                else:  # both streams ended, the process has not
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(POLL_S)
        except BaseException as error:
            try:
                if grace_s > 0:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                    end = time.monotonic() + grace_s
                    while process.poll() is None and time.monotonic() < end:
                        if selector.get_map():
                            read(min(end, time.monotonic() + POLL_S))
                        else:
                            with contextlib.suppress(subprocess.TimeoutExpired):
                                process.wait(min(POLL_S, max(0.0, end - time.monotonic())))
            finally:  # also after a second Ctrl-C meanwhile
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            read(time.monotonic() + 1.0)  # what is left (a process it started may hold a stream)
            process.wait()
            if isinstance(error, subprocess.TimeoutExpired):
                raise subprocess.TimeoutExpired(
                    command, error.timeout, output=_text(out), stderr=_text(err)
                ) from None
            raise
    return subprocess.CompletedProcess(command, process.returncode, _text(out), _text(err))


def _text(kept: KeptLines) -> str:
    return kept.value().decode("utf-8", errors="replace")


def run_container(
    docker: str,
    run_args: Sequence[str],
    log_path: Path,
    timeout_s: float = float("inf"),
) -> tuple[int | None, bool, float]:
    """``docker run RUN_ARGS`` with its output in ``log_path``, like ``run_logged``: a timeout,
    ``cancel_all`` or Ctrl-C stops the container itself, not only the Docker client (which
    would leave it running). ``run_args`` are everything after ``docker run``, without
    ``--name``; the container is removed when it ends, and Docker keeps no log of its own of it
    (``NO_DOCKER_LOG``)."""
    import uuid

    name = f"r2s-scorer-{uuid.uuid4().hex[:12]}"

    def stop() -> None:
        subprocess.run(
            [docker, "kill", name],
            capture_output=True,
            timeout=60,
            check=False,
            stdin=subprocess.DEVNULL,
        )

    try:
        return run_logged(
            [docker, "run", "--name", name, *NO_DOCKER_LOG, *run_args],
            log_path,
            timeout_s,
            on_timeout=stop,
        )
    finally:
        subprocess.run(
            [docker, "rm", "-f", name],
            capture_output=True,
            timeout=120,
            check=False,
            stdin=subprocess.DEVNULL,
        )


def _stop(process: subprocess.Popen[bytes], on_timeout: Callable[[], None] | None) -> None:
    """Stop ``process`` and everything it started (a container, through ``on_timeout``): its
    process group is asked to stop (``SIGTERM``), then killed after ``STOP_GRACE_S``."""
    if on_timeout is not None:
        with contextlib.suppress(Exception):
            on_timeout()
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        deadline = time.monotonic() + STOP_GRACE_S
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:  # also after a second Ctrl-C meanwhile
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    process.wait()


MIN_FREE_BYTES = 16 * 1024 * 1024
"""Less free space than this on a disk a failed run wrote to: the disk was full."""

SCAN_ENTRIES = 10_000
"""The most entries (files, directories, anything else) of a run's trees that the storage check
looks at: a program that leaves a huge tree costs the check a bounded time."""


def storage_problem(log_path: Path, writable: Sequence[Path]) -> str | None:
    """Why a run that failed may have failed because of this machine's storage, or ``None``.

    A file that reached the file-size limit (``ulimit -f``), the log included, or a disk left
    (nearly) full: either stops a correct program, so its failure is not the program's. The
    trees in ``writable`` hold what the program left there: they are only looked at
    (:func:`_files`), so nothing in them makes this fail."""
    import resource

    limit = resource.getrlimit(resource.RLIMIT_FSIZE)[0]
    if limit != resource.RLIM_INFINITY:
        for path, size in _files(log_path, writable):
            if size >= limit:
                return (
                    f"{path} reached this machine's file-size limit ({limit} bytes, "
                    "`ulimit -f`); raise it and run again"
                )
    for directory in {Path(log_path).parent, *map(Path, writable)}:
        with contextlib.suppress(OSError):
            info = os.statvfs(directory)
            if info.f_bavail * info.f_frsize < MIN_FREE_BYTES:
                return f"the disk holding {directory} is full; free space and run again"
    return None


def _files(log_path: Path, writable: Sequence[Path]) -> list[tuple[Path, int]]:
    """The log and the regular files in the trees ``writable``, with their sizes: the trees in
    that order, each directory once (a tree inside an earlier one is not looked at again), and
    at most :data:`SCAN_ENTRIES` entries in all. A program made these trees, so links are not
    followed (nor is a tree that has become one), and whatever cannot be read is skipped: a
    directory without read or search permission, a path longer than the system allows, an
    entry removed meanwhile."""
    found: list[tuple[Path, int]] = []
    with contextlib.suppress(OSError):
        found.append((Path(log_path), os.lstat(log_path).st_size))
    listed: set[str] = set()
    looked = 0
    for tree in writable:
        if os.path.islink(tree):
            continue
        stack = [os.fspath(tree)]
        while stack:  # iterative: no recursion limit
            directory = stack.pop()
            if directory in listed:
                continue
            listed.add(directory)
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        looked += 1
                        if looked > SCAN_ENTRIES:
                            return found
                        with contextlib.suppress(OSError):
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(entry.path)
                            elif entry.is_file(follow_symlinks=False):
                                size = entry.stat(follow_symlinks=False).st_size
                                found.append((Path(entry.path), size))
            except OSError:
                continue
    return found
