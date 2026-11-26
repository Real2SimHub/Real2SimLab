"""The errors Real2SimLab raises on purpose.

Every error a user can cause or fix derives from ``R2SError``. Errors that list several
problems carry them in ``problems``; errors caused by the environment carry a ``fix``: a
command or step that resolves them.
"""

from __future__ import annotations

import difflib
import errno
from collections.abc import Iterable
from typing import Self


class R2SError(Exception):
    """Base class of every error Real2SimLab raises on purpose."""


class ProblemsError(R2SError, ValueError):
    """Something is invalid in one or more ways, all listed in ``problems``; ``fix``, when
    given, says what to do about it."""

    def __init__(self, subject: str, problems: Iterable[str], fix: str | None = None) -> None:
        self.subject = subject
        self.problems = list(problems)
        self.fix = fix
        listed = "\n".join(f"  - {problem}" for problem in self.problems)
        text = f"{subject}:\n{listed}" if self.problems else subject
        super().__init__(f"{text}\n  fix: {fix}" if fix else text)


class SpecError(ProblemsError):
    """A document violates its Real2SimLab specification."""


class UnknownNameError(R2SError, LookupError):
    """A name is not registered, e.g. an unknown benchmark, method, metric or format."""

    def __init__(self, kind: str, name: str, known: Iterable[str]) -> None:
        self.kind = kind
        self.name = name
        self.known = sorted(known)
        listed = ", ".join(self.known) or "none"
        super().__init__(f"unknown {kind} {name!r} (known: {listed})")

    def __str__(self) -> str:
        return str(self.args[0])


class SetupError(R2SError, RuntimeError):
    """The environment is not ready, e.g. Docker is missing or an image is not built."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        self.message = message
        self.fix = fix
        super().__init__(f"{message}\n  fix: {fix}" if fix else message)


class UsageError(R2SError, ValueError):
    """A command's or a configuration's input is invalid: an unknown name, an option out of
    range. The message says what is accepted; ``fix``, when given, what to write instead."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        self.message = message
        self.fix = fix
        super().__init__(f"{message}\n  fix: {fix}" if fix else message)


class UnknownOptionError(UsageError, TypeError):
    """Options a benchmark or a method does not take."""

    @classmethod
    def from_names(cls, what: str, unknown: Iterable[str], taken: Iterable[str]) -> Self:
        """The error for the options ``unknown`` that ``what`` does not take: it names those it
        takes, and its fix the one probably meant for each (``release`` for ``relase``)."""
        names, takes = sorted(unknown), sorted(taken)
        meant = {name: difflib.get_close_matches(name, takes, n=1) for name in names}
        instead = [f"{close[0]} instead of {name}" for name, close in meant.items() if close]
        left = [name for name, close in meant.items() if not close]
        fixes = [f"write {', '.join(instead)}"] if instead else []
        if left:
            fixes.append(f"leave out {', '.join(left)}, or give an option it takes")
        return cls(
            f"unknown options for {what}: {names}; it takes: {takes or 'none'}",
            fix="; ".join(fixes),
        )


class OutputExistsError(SetupError, FileExistsError):
    """An output path is already there; it is never replaced without being asked."""


class InputNotFoundError(SetupError, FileNotFoundError):
    """An input path given to a command does not exist."""


class InfrastructureError(R2SError, RuntimeError):
    """A sandbox or container failed for reasons outside the method, and may be retried.
    ``fix``, when given, says what to do about it; it is not part of the message, which
    records quote."""

    def __init__(self, message: str = "", fix: str | None = None) -> None:
        super().__init__(message)
        self.fix = fix


class DeliveryError(R2SError):
    """A method ran but did not deliver a valid twin. Scored as zero, not raised to the user."""


_HOST_ERRNO_NAMES = (
    "EIO", "ESTALE", "EMFILE", "ENFILE", "ENOMEM", "ENOBUFS", "ENOSPC", "EDQUOT", "EROFS",
    "ETIMEDOUT", "ENOTCONN", "ECONNRESET", "ECONNABORTED", "EHOSTDOWN", "EHOSTUNREACH",
    "ENETDOWN", "ENETUNREACH", "EREMOTEIO", "ENOLINK", "ECOMM", "EUCLEAN", "ENOMEDIUM",
)  # fmt: skip
HOST_ERRNOS = frozenset(getattr(errno, name) for name in _HOST_ERRNO_NAMES if hasattr(errno, name))
"""The ``errno`` of a file operation this machine failed (besides no permission): an I/O
error, a network file system gone or stale, too many open files, no memory or space."""


def host_failure(error: BaseException) -> bool:
    """Whether ``error``, raised while the Lab reads what a program wrote, is this machine's
    failure to read it, not a problem of what was written: no permission (the Lab gives itself
    access to a program's files before it reads them), or :data:`HOST_ERRNOS`. A run that
    fails so is ``infra_error``, never a score. What the files themselves cause (one missing,
    a name too long, content a reader rejects, an error without an ``errno``) is not."""
    if isinstance(error, PermissionError):
        return True
    return isinstance(error, OSError) and error.errno in HOST_ERRNOS
