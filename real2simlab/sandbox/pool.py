"""Worker processes that stop with the command that started them.

A ``concurrent.futures.ProcessPoolExecutor`` of its own does not. Ctrl-C reaches every
process of the terminal's job, so each worker raised ``KeyboardInterrupt`` wherever it was
(holding a lock of the pool's queues, say); leaving the pool waited for the tasks its workers
had begun (a whole download); a second Ctrl-C during that wait left the command waiting for
ever for workers that never ended; and a command killed with ``kill -9`` left its workers
waiting for tasks that never came (A3-12). ``map_in_processes`` is the Lab's only way to start
worker processes.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.process import BaseProcess
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

KILL_WAIT_S = 5.0
"""How long to wait for killed workers to end: one in uninterruptible sleep (a hung network
file system) ends only when its I/O does."""

WATCH_S = 0.5
"""How often a worker checks that its parent still runs, where the system cannot be asked to
end it with its parent."""


def map_in_processes(function: Callable[[T], R], items: Iterable[T], workers: int) -> list[R]:
    """``function`` of each item, in order, ``workers`` at a time, each in a worker process.

    Stopping is this process's: the workers ignore Ctrl-C and a closed terminal (which reach
    them too) and end when this process ends, however it ends (``kill -9`` included). Whatever
    stops the map (an item's error, Ctrl-C, ``kill``) is raised at once: the items not begun
    are dropped and the workers killed without waiting for what they were doing, so
    ``function`` must leave nothing half-written when killed.
    """
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_worker_started)
    try:
        results = list(pool.map(function, items))
        pool.shutdown()
    except BaseException:
        _kill(pool)
        raise
    return results


def _kill(pool: ProcessPoolExecutor) -> None:
    """Drop the items not begun and kill the workers now, as Python 3.14's ``kill_workers``
    does: the pool's own shutdown waits for what its workers are doing."""
    running: dict[int, BaseProcess] = getattr(pool, "_processes", None) or {}
    processes = list(running.values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        with contextlib.suppress(OSError, ValueError):  # it ended meanwhile
            process.kill()
    deadline = time.monotonic() + KILL_WAIT_S
    for process in processes:
        with contextlib.suppress(OSError, ValueError, AssertionError):
            process.join(max(0.0, deadline - time.monotonic()))


def _worker_started() -> None:
    """A worker's first step: leave stopping to the command, and end with it. Ctrl-C and a
    closed terminal, which reach it too, go to a handler that does nothing, not ``SIG_IGN``:
    an ignored signal stays ignored across ``exec``, so the programs the worker starts
    (ffprobe, ffmpeg) would not stop on them either, and would run on once it is killed; a
    handler is reset to the default in them. A signal the command was started ignoring
    (SIGHUP under ``nohup``, SIGINT in a background job of a non-interactive shell) stays
    ignored, in the worker and in those programs, as in the command itself: a handler would
    give it back its default in them, and closing the terminal would kill them under a build
    that ``nohup`` keeps running."""
    for name in ("SIGINT", "SIGHUP"):
        number = getattr(signal, name, None)  # no SIGHUP on Windows
        if number is not None and signal.getsignal(number) is not signal.SIG_IGN:
            signal.signal(number, _left_to_the_command)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)  # not the command's handler a fork copies
    parent = os.getppid()
    if not _ends_with_parent(parent):
        threading.Thread(target=_watch, args=(parent,), daemon=True).start()


def _left_to_the_command(_signum: int, _frame: object) -> None:
    """A signal that the command handles (it kills its workers): nothing to do here."""


def _ends_with_parent(parent: int) -> bool:
    """Ask the system to kill this process when its parent ends (Linux); whether it will."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        from real2simlab.sandbox.reaper import PR_SET_PDEATHSIG, prctl

        prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:  # e.g. no ctypes: it watches its parent instead
        return False
    if os.getppid() != parent:  # the parent ended before it could be told
        os._exit(1)
    return True


def _watch(parent: int) -> None:
    """End this process once its parent has ended."""
    while os.getppid() == parent:
        time.sleep(WATCH_S)
    os._exit(1)
