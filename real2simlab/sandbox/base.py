"""Sandbox backends run a command from an image with given mounts, network and limits.

Three backends implement the same interface: ``docker`` (the main one), ``bwrap`` (Linux
without root or Docker, from an unpacked image) and ``local`` (development only, no
isolation). Backends are plugins in the ``real2simlab.backends`` entry point group.
"""

from __future__ import annotations

import os
import signal
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from real2simlab import plugins
from real2simlab.errors import SetupError


class NetworkMode(StrEnum):
    NONE = "none"
    """No network at all: every twin run, and agents' commands under mini_swe."""
    ALLOWLIST = "allowlist"
    """Only the listed domains, through an egress proxy that logs every connection."""
    OPEN = "open"
    """Unrestricted; reported prominently wherever it is used."""


@dataclass(frozen=True)
class NetworkPolicy:
    mode: NetworkMode = NetworkMode.NONE
    allowlist: tuple[str, ...] = ()
    """Domains reachable in ``allowlist`` mode; ``*.example.com`` matches subdomains."""
    log_path: Path | None = None
    """Where the egress proxy writes its connection log (JSON lines)."""
    proxy_url: str | None = None
    """The egress proxy, as seen from inside the sandbox (``allowlist`` mode)."""
    docker_network: str | None = None
    """The internal Docker network that reaches only the proxy (``allowlist`` mode)."""

    def to_json(self) -> dict[str, Any]:
        return {"mode": self.mode.value, "allowlist": list(self.allowlist)}

    def proxy_env(self) -> dict[str, str]:
        """Proxy variables for the sandboxed process; empty unless a proxy is set."""
        if self.proxy_url is None:
            return {}
        url = self.proxy_url
        return {
            "HTTPS_PROXY": url,
            "HTTP_PROXY": url,
            "https_proxy": url,
            "http_proxy": url,
            "ALL_PROXY": url,
            "NO_PROXY": "localhost,127.0.0.1",
            "no_proxy": "localhost,127.0.0.1",
        }


@dataclass(frozen=True)
class Mount:
    """A host path visible inside the sandbox at ``target``."""

    source: Path
    target: str
    read_only: bool = True


@dataclass(frozen=True)
class Limits:
    timeout_s: float
    cpus: float | None = None
    memory_mb: int | None = None
    gpus: str | None = None
    """``None`` for no GPU, ``"all"``, or comma-separated device indices."""
    pids: int = 4096


@dataclass(frozen=True)
class SandboxSpec:
    """One command to run in a sandbox."""

    image: str
    """An image reference, e.g. ``ghcr.io/real2simhub/r2s-python:1.0``; ignored by ``local``."""
    argv: tuple[str, ...]
    workdir: str
    mounts: tuple[Mount, ...] = ()
    env: Mapping[str, str] = field(default_factory=lambda: {})
    secret_env: Mapping[str, str] = field(default_factory=lambda: {})
    """Variables passed to the process but never written to logs or records."""
    network: NetworkPolicy = NetworkPolicy()
    limits: Limits = Limits(timeout_s=300)
    tmpfs: tuple[str, ...] = ("/tmp",)
    label: str = "r2s"
    """A short name for the container or process, for humans."""
    user: str | None = None
    """``uid:gid`` to run as; ``None``: the host user (``docker``, ``bwrap``; ignored by
    ``local``)."""


@dataclass(frozen=True)
class ExecResult:
    """How a sandboxed command ended."""

    exit_code: int | None
    """``None`` when it was killed at the timeout or never started."""
    timed_out: bool
    duration_s: float
    infrastructure_error: str | None = None
    """Set when the backend failed (e.g. the container did not start) or something outside the
    sandbox stopped it (its own process was killed); may be retried."""


@dataclass(frozen=True)
class ReadinessProblem:
    message: str
    fix: str | None = None


@dataclass(frozen=True)
class ResolvedImage:
    """An image reference and its content digest, recorded with every run."""

    reference: str
    digest: str | None
    content_id: str | None = None
    """The content id of the image itself (Docker's image id, the unpacked image's id): what
    a run actually used. A tag moved to another image keeps its reference but not this."""

    @property
    def identity(self) -> str:
        """What identifies the image a run used: its content id, else its digest."""
        return self.content_id or self.digest or self.reference


class Backend(ABC):
    """Runs sandboxed commands."""

    name: ClassVar[str]
    isolated: ClassVar[bool] = True
    supports_network_allowlist: ClassVar[bool] = False

    @abstractmethod
    def problems(self) -> list[ReadinessProblem]:
        """What prevents this backend from running on this machine; empty when ready."""

    def ensure_ready(self) -> None:
        """Raise ``SetupError`` with a fix if the backend cannot run here."""
        found = self.problems()
        if found:
            first = found[0]
            raise SetupError(f"backend {self.name!r} is not ready: {first.message}", fix=first.fix)

    def setup_problem(self, log: str) -> ReadinessProblem | None:
        """Why the sandbox of a probe the Lab ran (nothing of a program) could not be set up on
        this machine, as this backend's own messages in the probe's ``log`` say; ``None`` when
        they do not say so (it was set up, or this backend cannot tell). Never asked about a
        program's log, which the program writes."""
        return None

    @abstractmethod
    def resolve_image(self, reference: str) -> ResolvedImage:
        """Check that ``reference`` is available; raise ``SetupError`` with a fix otherwise."""

    @abstractmethod
    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        """Run ``spec`` with stdout and stderr written to ``log_path``."""

    def info(self) -> dict[str, Any]:
        """Facts recorded with every run, e.g. the backend's version."""
        return {"backend": self.name}

    def gpu_problems(self) -> list[ReadinessProblem]:
        """What prevents GPU runs; empty when GPUs work."""
        return [ReadinessProblem(f"backend {self.name!r} does not support GPUs")]

    @contextmanager
    def session(self, spec: SandboxSpec) -> Generator[Session]:
        """A sandbox that stays up while many commands run in it (``spec.argv`` is ignored).

        Each command runs as ``[*session.prefix, "<shell command>"]`` with ``session.env`` as
        its whole environment and ``session.cwd`` as its working directory.
        """
        raise NotImplementedError(f"backend {self.name!r} has no command sessions")
        yield  # pragma: no cover


@dataclass(frozen=True)
class Session:
    """How to run one shell command in a live sandbox."""

    prefix: tuple[str, ...]
    """Ends just before the command string, e.g. ``(..., "bash", "-c")``."""
    env: Mapping[str, str]
    """The command process's complete environment; it never holds secrets."""
    cwd: Path


def finished(
    code: int | None, timed_out: bool, seconds: float, log_path: Path, spec: SandboxSpec
) -> ExecResult:
    """The result of a process that ran in ``spec``'s sandbox. A failure that this machine's
    storage explains (a file at the file-size limit, a full disk) is an infrastructure error,
    not the program's failure."""
    from real2simlab.sandbox.process import storage_problem

    if code not in (0, None) and not timed_out:
        writable = [Path(mount.source) for mount in spec.mounts if not mount.read_only]
        problem = storage_problem(log_path, writable)
        if problem is not None:
            return ExecResult(None, False, seconds, problem)
    return ExecResult(exit_code=code, timed_out=timed_out, duration_s=seconds)


def killed_from_outside(code: int | None, process: str) -> str | None:
    """Why a sandbox whose own process ``process`` (bwrap, the Docker client) ended with
    ``code`` holds no result of its program, or ``None``. That process stays outside the
    sandbox: the program (in another PID namespace, in a container) cannot signal it, and the
    program's own death by a signal is an exit status (128 + N). So the process dying of a
    signal itself (a negative ``code``) was done from outside: an administrator's ``kill``,
    the host's out-of-memory killer."""
    if code is None or code >= 0:
        return None
    try:
        name = signal.Signals(-code).name
    except ValueError:
        name = f"signal {-code}"
    return (
        f"the sandbox process ({process}) was killed by {name} from outside the sandbox, so the "
        "result is not the program's; run it again"
    )


def get_backend(name: str, **options: Any) -> Backend:
    """Create the registered backend called ``name``; ``options`` are its ``backend_options``."""
    from real2simlab.options import checked_options

    factory = plugins.load("backend", name)
    backend = factory(**checked_options(factory, options, f"backend {name!r}"))
    if not isinstance(backend, Backend):
        raise TypeError(f"backend plugin {name!r} does not create a Backend")
    return backend


def backend_names() -> list[str]:
    return plugins.names("backend")


def ensure_chosen_ready(backend: Backend) -> None:
    """``backend.ensure_ready()`` for the backend a command runs on (``--backend``, or
    ``backend`` under ``[run]``; ``docker`` unless given another). When it cannot run here, the
    fix also names each other isolated backend that can, unless it names it already: on a
    machine without Docker whose bwrap was ready, an evaluation said only to install or start
    Docker, both of which need root. Never ``local``, which isolates nothing; and the command
    never moves to another backend by itself."""
    try:
        backend.ensure_ready()
    except SetupError as error:
        fix = error.fix or ""
        # A backend the fix names already is not asked again (a Docker that hangs took 40 s).
        others = _ready_instead_of(backend, skip=lambda name: f"--backend {name}" in fix)
        if not others:
            raise
        message = str(error).removesuffix(f"\n  fix: {fix}") if fix else str(error)
        instead = "; or ".join(_use_instead(name) for name in others)
        raise SetupError(message, fix=f"{fix}; or {instead}" if fix else instead) from None


def _use_instead(name: str) -> str:
    """How the fix names ``name``, a backend that can run here, as ``r2s doctor`` does: bwrap
    runs twins only in its unpacked runtime, so without it the fix says how to unpack it."""
    if name == "bwrap":
        from real2simlab.doctor import rootfs_check

        try:  # said while another error is reported: never in its place
            runtime = rootfs_check("r2s-python")
        except Exception:
            return "use --backend bwrap (r2s doctor says whether its runtime is unpacked)"
        if not runtime.ok:
            unpack = runtime.fix or "r2s env unpack r2s-python"
            return f"use --backend bwrap once its runtime is unpacked: {unpack}"
    return f"use --backend {name}, which can run here"


def _ready_instead_of(
    chosen: Backend, skip: Callable[[str], bool] = lambda name: False
) -> list[str]:
    """The other isolated backends that can run on this machine (``r2s doctor`` calls them
    ready), but for those ``skip`` names (they are not asked)."""
    ready: list[str] = []
    for name in backend_names():
        if name == chosen.name or skip(name):
            continue
        try:
            other = get_backend(name)
            if other.isolated and not other.problems():
                ready.append(name)
        except Exception:  # a plugin that needs options or cannot load: nothing to suggest
            continue
    return ready


def host_user() -> str:
    """``uid:gid`` of this process, so files written in a sandbox belong to the user."""
    return f"{os.getuid()}:{os.getgid()}"


def rewrite_paths(value: str, mounts: Sequence[Mount]) -> str:
    """Replace a sandbox path prefix in ``value`` with the host path it is mounted from."""
    for mount in sorted(mounts, key=lambda m: len(m.target), reverse=True):
        if value == mount.target or value.startswith(mount.target.rstrip("/") + "/"):
            return str(mount.source) + value[len(mount.target.rstrip("/")) :]
    return value
