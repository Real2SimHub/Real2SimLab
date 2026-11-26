"""The ``r2s`` command. Subcommands are plugins in the ``real2simlab.cli`` entry point group.

A plugin is a function ``register(subparsers)`` that adds one subcommand and sets
``handler=<function(args) -> int>`` as its default.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import traceback
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from real2simlab import __version__, plugins
from real2simlab._fs import STORAGE_FIX, storage_errno
from real2simlab.cli.common import FORMATTER
from real2simlab.errors import (
    InfrastructureError,
    ProblemsError,
    R2SError,
    SetupError,
    UnknownNameError,
    UsageError,
)
from real2simlab.spec import SPEC_STATUS

DESCRIPTION = f"""\
Real2SimLab {__version__} (specification {SPEC_STATUS}): a unified platform for
real-to-simulation. Any method, any benchmark: true to each, together on one scorecard; and
training tasks, datasets and rewards from the same code (`r2s train`).

Start with `r2s doctor` and `r2s demo`, then `r2s bench list`, `r2s bench build NAME --out
tasks/` and `r2s eval --tasks tasks/ --method codex --out runs/codex`.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="r2s", description=DESCRIPTION, formatter_class=FORMATTER)
    parser.add_argument("--version", action="version", version=f"r2s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name in plugins.names("cli"):
        register = plugins.load("cli", name)
        register(subparsers)
    return parser


def _show_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    """The Lab's own warnings as one plain line (no file name, no source line)."""
    if not category.__module__.startswith("real2simlab"):
        _default_show_warning(message, category, filename, lineno, file, line)
        return
    print(f"\nWARNING: {message}\n", file=file or sys.stderr)


_default_show_warning = warnings.showwarning


def main(argv: Sequence[str] | None = None) -> int:
    previous = warnings.showwarning
    warnings.showwarning = _show_warning
    # `kill` (SIGTERM) and closing the terminal (SIGHUP) stop a command as Ctrl-C does: what
    # it was running is stopped (its sandboxes and containers too), and the same command
    # resumes. A SIGHUP that is ignored (`nohup r2s ...`) stays ignored.
    previous_handlers: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGTERM", "SIGHUP"):
            number = getattr(signal, name, None)
            if number is None:
                continue  # no SIGHUP on Windows
            if name == "SIGHUP" and signal.getsignal(number) is signal.SIG_IGN:
                continue
            previous_handlers[number] = signal.signal(number, _interrupt)
    try:
        code = _main(argv)
        # What is left in the buffer reaches the reader now; when the reader has stopped
        # reading (`r2s demo | head`), nowhere, quietly (Python's own flush at exit would fail
        # with two lines on stderr and the status 120).
        flushed = _flush_output()
        return OUTPUT_CLOSED if code == 0 and not flushed else code
    except SystemExit:  # --help, --version, a usage error: printed by argparse
        _flush_output()
        raise
    finally:
        warnings.showwarning = previous
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


OUTPUT_CLOSED = 141
"""The exit status of a command whose output the reader stopped reading (``r2s demo |
head``): 128 + ``SIGPIPE``, that of a program the signal ended, as other tools of a pipeline
report it."""


def _flush_output() -> bool:
    """Flush stdout; ``False`` when whoever reads it has stopped (what is left is dropped)."""
    try:
        sys.stdout.flush()
    except BrokenPipeError:
        _drop_output()
        return False
    except (AttributeError, OSError, ValueError):  # no stdout at all (`>&-`), or closed
        pass
    return True


def _output_closed(error: BaseException) -> bool:
    """Whether ``error`` is this command's output that its reader stopped reading (``r2s ... |
    head``): the reading end of the pipe is gone. A broken pipe to anything else (a tool the
    Lab feeds, a network connection) is not."""
    if not isinstance(error, BrokenPipeError):
        return False
    import select

    if not hasattr(select, "poll"):
        return False
    try:
        poller = select.poll()
        poller.register(sys.stdout.fileno(), select.POLLOUT)
        return any(events & (select.POLLERR | select.POLLHUP) for _, events in poller.poll(0))
    except (AttributeError, OSError, ValueError):  # no file behind stdout
        return False


def _drop_output() -> None:
    """Send what is left to print to ``/dev/null``: its reader has gone."""
    with contextlib.suppress(AttributeError, OSError, ValueError):
        descriptor = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(descriptor, sys.stdout.fileno())
        finally:
            os.close(descriptor)


def _interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _main(argv: Sequence[str] | None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    try:
        return int(handler(args) or 0)
    except KeyboardInterrupt:
        # What running it again does differs by command: each says so (``on_interrupt``).
        hint = getattr(args, "on_interrupt", None)
        with contextlib.suppress(OSError):  # the terminal may be gone (SIGHUP)
            print("interrupted" + (f"; {hint}" if hint else ""), file=sys.stderr)
        return 130
    except MemoryError as error:  # this machine's memory, or a limit on it: not a bug to report
        _out_of_memory(error)
        return 1
    except R2SError as error:
        if os.environ.get("R2S_DEBUG"):
            traceback.print_exc()
        if not getattr(error, "fix", None) and storage_errno(error) is not None:
            # This machine's storage (a full disk, a quota, `ulimit -f`) stopped what the error
            # says (ffmpeg's output, a copy): the machine's to fix, said with what to do.
            print(f"error: {error}\n  fix: {STORAGE_FIX}", file=sys.stderr)
            return 1
        print(f"error: {_describe(error)}", file=sys.stderr)
        return 1
    except Exception as error:
        if _output_closed(error):  # `r2s ... | head`: the reader has what it wanted
            _drop_output()
            return OUTPUT_CLOSED
        if isinstance(error, OSError) and error.errno == errno.ENOMEM:
            _out_of_memory(error)  # memory too (a fork, a map): said before any import
            return 1
        try:
            explained = _explain(error)
        except Exception:  # telling what it is failed too: shown as the bug it then is
            explained = None
        if explained is None:  # nothing a user or this machine did wrong
            traceback.print_exc()
            print(
                "\nThis is a bug in Real2SimLab, not something you did wrong. Please report it at "
                f"{ISSUES} with the command you ran and this output.",
                file=sys.stderr,
            )
            return 1
        if os.environ.get("R2S_DEBUG"):
            traceback.print_exc()
        print(f"error: {explained}", file=sys.stderr)
        return 1


ISSUES = "https://github.com/Real2SimHub/Real2SimLab/issues"


FULL = {errno.ENOSPC, errno.EDQUOT, errno.EFBIG}
"""A disk that is full, a quota reached, a file-size limit (``ulimit -f``): the machine's."""

OUT_OF_MEMORY = (
    "error: this command ran out of memory (this machine's, or the share a limit allows it: "
    "`ulimit -v`, a batch job's memory request)\n"
    "  fix: free memory or run it where more is allowed (raise the limit, request more memory "
    "for the job), then run the same command again"
)


def _out_of_memory(error: BaseException) -> None:
    """Say that memory ran out (a ``MemoryError``, or an ``OSError`` of ``ENOMEM``), in one
    error line with its fix. What held the memory (the frames the error went through) is let go
    first, and nothing is imported: there may be little memory left."""
    if os.environ.get("R2S_DEBUG"):
        with contextlib.suppress(Exception):
            traceback.print_exc()
    traceback.clear_frames(error.__traceback__)
    with contextlib.suppress(Exception):
        print(OUT_OF_MEMORY, file=sys.stderr)


PATH_ERRORS = (
    FileExistsError,
    FileNotFoundError,
    IsADirectoryError,
    NotADirectoryError,
    PermissionError,
)
"""File-system errors that a command's arguments cause (a wrong or existing path, no access)."""

SYMLINK_LOOP = "Symlink loop from "
"""How ``Path.resolve()`` before Python 3.13 says a path leads through a loop of symbolic links
(a ``RuntimeError``; Python 3.13 raises ``OSError`` ``ELOOP``)."""

LOOP_FIX = (
    "the path leads through a loop of symbolic links (a link that points back at itself, or at "
    "a directory above it): remove or repoint that link, or choose another path; then run the "
    "same command again"
)
AGAIN = "then run the same command again"
LINK_FIX = (
    "the Lab never follows a symbolic link in what it hashes (a task, a program, an output): "
    "put the file or directory it points to in its place, or remove it (`r2s task validate` "
    f"checks a task); {AGAIN}"
)


def _explain(error: Exception) -> SetupError | None:
    """``error``, which no command turned into an ``R2SError``, as one error line and the fix
    that suits it; ``None`` for a bug. What only a bug raises (a ``KeyError``, a ``TypeError``,
    ...) is one whatever it was raised while handling. A failed or refused download is told
    first, whatever wraps it (a download library's ``ValueError``, ``FileNotFoundError`` or
    ``OSError``); what ``raise ... from None`` hid explains nothing."""
    from real2simlab.vendor import download_failure, is_bug

    if is_bug(error):
        return None
    found = download_failure(error)
    if found is not None:
        return found
    if isinstance(error, RuntimeError) and str(error).startswith(SYMLINK_LOOP):
        path = _quoted(str(error)[len(SYMLINK_LOOP) :])  # said as Python 3.13 says it
        return SetupError(f"{os.strerror(errno.ELOOP)} ({path})", fix=LOOP_FIX)
    if isinstance(error, PATH_ERRORS):  # a path the user gave: say which, and what to do
        return _path_error(error)
    from real2simlab.twin.fingerprint import SymbolicLinkError

    if isinstance(error, SymbolicLinkError):  # a link where the Lab hashes content (a task, ...)
        return SetupError(str(error), fix=LINK_FIX)
    if isinstance(error, ValueError):  # an input a command could not use
        return _value_error(error)
    if isinstance(error, shutil.Error):  # a directory copy (copytree) that failed for some files
        return _copy_error(error)
    if isinstance(error, subprocess.TimeoutExpired):  # a tool the Lab asked stopped answering
        command = error.cmd if isinstance(error.cmd, str) else shlex.join(map(str, error.cmd))
        return SetupError(
            f"`{command}` did not finish in {error.timeout:g} s",
            fix="check that this tool works on this machine (for Docker: `docker info`; for a "
            f"download: the network and the proxy settings), {AGAIN}",
        )
    if isinstance(error, OSError):
        return _os_error(error)
    return None


def _quoted(text: str) -> str:
    """The path in ``'...'``, as ``%r`` wrote it."""
    import ast

    with contextlib.suppress(ValueError, SyntaxError):
        value = ast.literal_eval(text)
        if isinstance(value, str):
            return value
    return text


def _filename(error: OSError) -> str | None:
    name = error.filename
    if name is None or isinstance(name, int):  # none, or a file descriptor
        return None
    return os.fsdecode(name)


def _loops(path: str) -> bool:
    """Whether ``path`` leads through a loop of symbolic links (a link that points back at
    itself, or at a directory above it): what exists there is no directory one can make."""
    try:
        os.stat(path)
    except OSError as error:
        return error.errno == errno.ELOOP
    return False


def _path_error(error: OSError) -> SetupError:
    """A path that is missing, already there, of the other kind (a file or a directory), or not
    accessible; or a program that could not be started."""
    program = _unstarted_program(error)
    if program is not None:
        return program
    path = _filename(error)
    if path is None:
        return SetupError(str(error), fix=f"check the paths the command was given, {AGAIN}")
    if isinstance(error, PermissionError):
        cache = _cache_of(path)
        return SetupError(
            f"{path} cannot be accessed (permission denied)",
            fix=f"Real2SimLab keeps its cache in {cache} (R2S_CACHE_DIR, or else XDG_CACHE_HOME "
            "or ~/.cache): make it writable, or set R2S_CACHE_DIR to a directory you may write; "
            f"{AGAIN}"
            if cache is not None
            else "give your user access to it (or to the directory it is in), or use a path "
            f"you may write to; {AGAIN}",
        )
    if isinstance(error, FileExistsError):
        if _loops(path):  # Python 3.13 resolve() lets a loop of links through to mkdir
            return SetupError(f"{os.strerror(errno.ELOOP)} ({path})", fix=LOOP_FIX)
        return SetupError(
            f"{path} already exists",
            fix="if another r2s command is writing there, wait until it finishes; otherwise "
            f"choose a path that does not exist yet; {AGAIN}",
        )
    if isinstance(error, IsADirectoryError):
        return SetupError(f"{path} is a directory, not a file", fix=f"give a file, {AGAIN}")
    if isinstance(error, NotADirectoryError):
        return SetupError(
            f"{path} is not a directory",
            fix=f"give a directory, on a path that does not go through a file; {AGAIN}",
        )
    relative = "" if os.path.isabs(path) else f" (it is relative to {os.getcwd()})"
    return SetupError(f"{path} does not exist", fix=f"check the path{relative}, {AGAIN}")


def _unstarted_program(error: OSError) -> SetupError | None:
    """The program a ``subprocess`` call could not start (not found, or not executable), when
    ``error`` was raised where it executes one; a working directory (``cwd``) that does not
    exist is said as one."""
    last = error.__traceback__
    while last is not None and last.tb_next is not None:
        last = last.tb_next
    program = _filename(error)
    if last is None or program is None:
        return None
    code = last.tb_frame.f_code
    if code.co_name != "_execute_child" or Path(code.co_filename).name != "subprocess.py":
        return None
    directory = last.tb_frame.f_locals.get("cwd")  # entered before the program is started
    if isinstance(directory, str | bytes | os.PathLike) and os.fsdecode(directory) == program:
        if isinstance(error, FileNotFoundError):
            return SetupError(
                f"the working directory {program} does not exist",
                fix=f"check that path (or the setting that gives it), {AGAIN}",
            )
        return None  # said as the path error it is (no access, not a directory)
    if isinstance(error, FileNotFoundError):
        if os.sep in program:
            return SetupError(
                f"the program {program} was not found",
                fix=f"check that path (or the setting that gives it), {AGAIN}",
            )
        return SetupError(
            f"the program {program} was not found: it is not on PATH",
            fix=f"install it, or add the directory that holds it to PATH; {AGAIN}",
        )
    if isinstance(error, PermissionError):
        return SetupError(
            f"the program {program} cannot be run (permission denied)",
            fix=f"make it executable (chmod +x), or put one that is first on PATH; {AGAIN}",
        )
    return None


def _cache_of(path: str) -> Path | None:
    """The Lab's cache, when ``path`` is in it, or is a directory above it still to be made."""
    from real2simlab.vendor import cache_dir

    cache = Path(os.path.abspath(cache_dir()))
    given = Path(os.path.abspath(path))
    inside = given.is_relative_to(cache)
    if inside or (cache.is_relative_to(given) and not os.path.lexists(given)):
        return cache
    return None


def _value_error(error: ValueError) -> SetupError:
    """An input a command could not use: the message says which; a specification model's
    validation in one line rather than pydantic's report."""
    from pydantic import ValidationError

    if isinstance(error, ValidationError):
        from real2simlab._base import validation_problems

        problems = [p.replace("Value error, ", "") for p in validation_problems(error)]
        more = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
        return SetupError(
            f"{error.title} is not valid: {problems[0] if problems else error}{more}",
            fix="correct that value (in the command's arguments or in the file it read), "
            f"{AGAIN} (set R2S_DEBUG=1 for the traceback)",
        )
    return SetupError(
        str(error), fix=f"correct that input, {AGAIN} (set R2S_DEBUG=1 for the traceback)"
    )


def _copy_error(error: shutil.Error) -> SetupError:
    problems = [item for item in error.args[0] if isinstance(item, tuple)] if error.args else []
    first = problems[0] if problems else ("?", "?", str(error))
    full = any(f"[Errno {code}]" in str(item[-1]) for item in problems for code in FULL)
    others = f"\n  (and {len(problems) - 1} other file(s))" if len(problems) > 1 else ""
    return SetupError(
        f"copying {first[0]} to {first[1]} failed: {first[-1]}{others}",
        fix="free space on that disk (or raise the file-size limit, `ulimit -f`), then run the "
        "same command again"
        if full
        else "make the source readable and the destination writable, then run the same "
        "command again",
    )


def _os_error(error: OSError) -> SetupError:
    names = [str(name) for name in (error.filename, error.filename2) if name]  # a move: from, to
    where = f" ({' -> '.join(names)})" if names else ""
    message = f"{error.strerror or error}{where}"
    if error.errno in FULL:
        fix = (
            "free space on that disk (or raise the file-size limit, `ulimit -f`), then run the "
            "same command again; finished steps are reused"
        )
    elif error.errno == errno.ENAMETOOLONG:  # a path the user gave
        fix = (
            "a name in this path is longer than the file system allows (255 bytes on most), or "
            f"the whole path is: choose a shorter path; {AGAIN}"
        )
    elif error.errno == errno.ELOOP:
        fix = LOOP_FIX
    elif error.errno == errno.EROFS:
        fix = f"that file system is read-only: choose a path on a writable one; {AGAIN}"
    elif error.errno == errno.ENOTEMPTY:  # a directory moved onto, or removed, that has files
        fix = (
            "a directory there is not empty: another command may have written it; wait until "
            f"it finishes, or choose a new path; {AGAIN}"
        )
    else:  # the machine refused it (too many open files, ...)
        fix = (
            "this machine's system refused the operation; fix that, then run the same command "
            "again (set R2S_DEBUG=1 for the traceback)"
        )
    return SetupError(message, fix=fix)


def _describe(error: R2SError) -> str:
    if isinstance(error, SetupError | UnknownNameError | ProblemsError | UsageError):
        return str(error)
    # A machine's failure is said as what failed, with its fix; other errors keep their kind.
    said = (
        str(error) if isinstance(error, InfrastructureError) else f"{type(error).__name__}: {error}"
    )
    fix = getattr(error, "fix", None)  # e.g. a MediaError's, kept out of what records quote
    return f"{said}\n  fix: {fix}" if isinstance(fix, str) and fix else said


if __name__ == "__main__":
    sys.exit(main())
