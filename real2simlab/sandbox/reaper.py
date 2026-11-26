"""Runs a program on the host so that nothing it starts outlives it. Standard library only;
never imports Real2SimLab. Linux.

Usage: ``python -I -S reaper.py PARENT_PID -- COMMAND...``

The ``local`` backend has no sandbox whose end stops everything in it: a process the program
starts in a session of its own, or a program whose Lab was killed with ``kill -9``, would run
on, writing into the run directory. So the local backend runs every command under this
process. It is a child subreaper (what the program leaves behind becomes its child, not
init's) and is told when the process that started it (``PARENT_PID``, the Lab) ends, by the
parent-death signal. When the program exits, or when this process is told to stop
(``SIGTERM``, ``SIGINT``, ``SIGHUP``, or the Lab's end), it kills every process left under it,
then exits as the program did: with its exit status, or killed by the same signal.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import time
from typing import NoReturn

PR_SET_PDEATHSIG = 1
PR_SET_CHILD_SUBREAPER = 36
STOP = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
KILL_FOR_S = 10.0
"""How long to go on killing what is left: a process in uninterruptible sleep (a hung network
file system) ends only when its I/O does."""


class Stopped(Exception):
    """This process was told to stop."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def prctl(option: int, value: int) -> None:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, value, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def say(message: str) -> None:
    """One line on stderr, which is the run's log."""
    try:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


def descendants(root: int) -> list[int]:
    """The processes under ``root`` that have not ended, by their parents in ``/proc``."""
    children: dict[int, list[int]] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as stream:
                stat = stream.read()
        except OSError:  # it ended meanwhile
            continue
        fields = stat[stat.rfind(b")") + 2 :].split()  # after the name, which may hold anything
        if len(fields) < 2 or fields[0] in (b"Z", b"X"):
            continue  # ended: its parent collects it
        children.setdefault(int(fields[1]), []).append(int(name))
    found: list[int] = []
    pending = [root]
    while pending:
        for child in children.get(pending.pop(), []):
            found.append(child)
            pending.append(child)
    return found


def reap() -> None:
    """Collect every child of this process that has ended."""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def kill_all() -> None:
    """Kill every process under this one. A process whose parent is killed becomes this
    process's child (a subreaper's), so the next round finds it. Where the processes cannot be
    listed (no ``/proc``), it says so: what the program started may then outlive it."""
    deadline = time.monotonic() + KILL_FOR_S
    while True:
        try:
            alive = descendants(os.getpid())
        except (OSError, ValueError) as error:
            say(f"r2s: cannot list the processes here ({error}); some may outlive this program")
            reap()
            return
        for pid in alive:
            with contextlib.suppress(OSError):  # it ended meanwhile
                os.kill(pid, signal.SIGKILL)
        reap()
        if not alive or time.monotonic() > deadline:
            return
        time.sleep(0.01)


def start(command: list[str]) -> int:
    """Start ``command`` as a child of this process; its pid."""
    me = os.getpid()
    signal.pthread_sigmask(signal.SIG_BLOCK, set(STOP))  # until the child has its own handling
    pid = os.fork()
    if pid == 0:
        try:
            for signum in (*STOP, signal.SIGPIPE, signal.SIGXFSZ):
                signal.signal(signum, signal.SIG_DFL)  # as a child of the Lab has them
            signal.pthread_sigmask(signal.SIG_UNBLOCK, set(STOP))
            with contextlib.suppress(Exception):  # killed with this process, at worst
                prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
            if os.getppid() != me:  # this process is gone already
                os._exit(128 + signal.SIGKILL)
            os.execvp(command[0], command)
        except FileNotFoundError as error:
            say(f"r2s: cannot run {command[0]}: {error.strerror}")
            os._exit(127)
        except BaseException as error:
            say(f"r2s: cannot run {command[0]}: {error}")
            os._exit(126)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, set(STOP))
    return pid


def wait(pid: int) -> int:
    """Wait for the child ``pid``, collecting the others that end meanwhile; its status."""
    while True:
        ended, status = os.waitpid(-1, 0)
        if ended == pid:
            return status


def die(signum: int) -> NoReturn:
    """End killed by ``signum``, as the program was (never with a core file of this process)."""
    import resource

    with contextlib.suppress(OSError, ValueError):
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with contextlib.suppress(OSError, ValueError):  # SIGKILL and SIGSTOP have no handler
        signal.signal(signum, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signum})
    os.kill(os.getpid(), signum)
    os._exit(128 + signum)  # a signal that does not end a process


def main(argv: list[str]) -> NoReturn:
    if len(argv) < 4 or not argv[1].isdigit() or argv[2] != "--":
        say("usage: reaper.py PARENT_PID -- COMMAND...")
        os._exit(2)
    parent, command = int(argv[1]), argv[3:]

    def stop(signum: int, _frame: object) -> None:
        for each in STOP:  # once
            signal.signal(each, signal.SIG_IGN)
        raise Stopped(signum)

    for signum in STOP:
        signal.signal(signum, stop)
    status: int | None = None
    stopped = signal.SIGTERM
    try:
        try:
            prctl(PR_SET_CHILD_SUBREAPER, 1)
            prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        except Exception as error:  # e.g. no ctypes: the program runs, without that guarantee
            say(f"r2s: cannot follow this program's processes here ({error}); some may outlive it")
        try:
            if os.getppid() == parent:  # otherwise the Lab ended before this could tell
                status = wait(start(command))
        finally:
            for signum in STOP:
                signal.signal(signum, signal.SIG_IGN)
    except Stopped as error:
        stopped = error.signum
    try:
        kill_all()
    except Exception as error:  # the program's status stands: this is not the program's doing
        say(f"r2s: cannot stop what this program started ({error!r})")
    if status is None:
        die(stopped)
    if os.WIFEXITED(status):
        os._exit(os.WEXITSTATUS(status))
    die(os.WTERMSIG(status) if os.WIFSIGNALED(status) else signal.SIGKILL)


if __name__ == "__main__":
    main(sys.argv)
