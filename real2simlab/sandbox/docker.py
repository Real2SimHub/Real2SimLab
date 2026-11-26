"""The ``docker`` backend: one ``docker run`` per command, its container removed afterwards.

Every container runs as the host user, with all capabilities dropped, no new privileges, a
read-only root file system, ``/tmp`` as tmpfs, limited CPUs, memory and processes, and
``--network none`` unless the spec asks for the egress allowlist (agents only) or for an open
network. Secrets are passed through the docker client's environment (``-e NAME`` without a
value), so they never appear on a command line, in a log or in a run record.

What a container prints reaches the Lab through the attached client, which keeps a bounded log
of it (``run_logged``). Docker's own log of the container is turned off (``--log-driver
none``): its default driver keeps everything a container prints, without a size limit, on the
daemon's disk until the container is removed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, cast

from real2simlab.errors import SetupError
from real2simlab.sandbox.base import (
    Backend,
    ExecResult,
    NetworkMode,
    ReadinessProblem,
    ResolvedImage,
    SandboxSpec,
    Session,
    finished,
    host_user,
    killed_from_outside,
)
from real2simlab.sandbox.process import NO_DOCKER_LOG, run_logged

DOCKER_ENV = "R2S_DOCKER"
INFRA_EXIT_CODE = 125
"""``docker run`` exits with 125 when the daemon fails before the program starts, and also
when the program itself exits with 125: the backend tells them apart by asking Docker whether
the container ever started."""
NEVER_STARTED = "0001-01-01T00:00:00Z"
"""Docker's ``State.StartedAt`` of a container that was created but never started."""

INSTALL_FIX = (
    "install Docker Engine (https://docs.docker.com/engine/install/) and add yourself to the "
    "docker group: sudo usermod -aG docker $USER (then log in again)"
)
INFO_TIMEOUT_S = 30
"""How long ``docker info`` may take: a daemon that has not answered by then hangs."""


CLIENT_VARIABLES = (
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
    "DOCKER_API_VERSION",
    "XDG_RUNTIME_DIR",
)
"""Variables that choose which Docker daemon the client talks to (and how). Every docker
command the Lab runs, including those run with a reduced environment, gets them, so creating,
using and removing a container all talk to the same daemon. None of them is a secret."""


def bind_mount(source: Path | str, target: str, *, read_only: bool = False) -> str:
    """The value of a ``docker run --mount`` option binding ``source`` at ``target``.

    Docker reads the value as one CSV record, so a field with a comma or a double quote in it
    (a directory called ``Last, First``) is quoted, as CSV quotes it."""
    fields = ["type=bind", f"source={Path(source).resolve()}", f"target={target}"]
    if read_only:
        fields.append("readonly")
    return ",".join(_csv_field(field) for field in fields)


def _csv_field(text: str) -> str:
    if any(character in text for character in ',"\r\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


def docker_binary() -> str | None:
    return os.environ.get(DOCKER_ENV) or shutil.which("docker")


def docker_client_env() -> dict[str, str]:
    """The Docker client variables of this process (see ``CLIENT_VARIABLES``)."""
    return {name: os.environ[name] for name in CLIENT_VARIABLES if name in os.environ}


OWN_DAEMON = ("unix:///var/run/docker.sock", "unix:///run/docker.sock")
"""``DOCKER_HOST`` values that name this machine's own daemon, the one whose socket the docker
group may use."""
DEFAULT_CONTEXT = "default"
"""The docker context of this machine's own daemon."""
CONTEXT_TIMEOUT_S = 10
"""How long ``docker context show`` may take: it reads the client's configuration, no daemon."""


def _chosen_daemon() -> str:
    """The variables of this shell that choose another Docker daemon than this machine's own,
    by name, as the docker client reads them: ``DOCKER_HOST``, unless it names this machine's
    own socket (it then overrides ``DOCKER_CONTEXT`` too), and ``DOCKER_CONTEXT``, unless it is
    the default context; empty when none does."""
    host, context = os.environ.get("DOCKER_HOST"), os.environ.get("DOCKER_CONTEXT")
    if host in OWN_DAEMON:
        return ""
    chosen = ["DOCKER_HOST"] if host else []
    if context and context != DEFAULT_CONTEXT:
        chosen.append("DOCKER_CONTEXT")
    return " and ".join(chosen)


def docker_problem(docker: str | None = None) -> str | None:
    """Why Docker cannot be used here, with its fix, in one line; ``None`` when it can. For what
    runs in Docker whatever the backend (an official scorer's own image)."""
    found = DockerBackend(docker).problems()
    if not found:
        return None
    return found[0].message + (f" (fix: {found[0].fix})" if found[0].fix else "")


class DockerBackend(Backend):
    name: ClassVar[str] = "docker"
    isolated: ClassVar[bool] = True
    supports_network_allowlist: ClassVar[bool] = True

    def __init__(self, docker: str | None = None) -> None:
        self.docker = docker or docker_binary() or "docker"
        self._images: dict[str, ResolvedImage] = {}
        self.max_cpus: float | None = None
        """The daemon's CPU count, once known: CPU limits above it are lowered to it (Docker
        refuses ``--cpus`` larger than the machine, e.g. an agent's 4 on a 2-core runner)."""
        self._cpus_known = False

    def _learn_cpus(self) -> None:
        if self._cpus_known:
            return
        self._cpus_known = True
        try:
            result = self._call("info", "--format", "{{.NCPU}}", timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return
        if result.returncode == 0 and result.stdout.strip().isdigit():
            self.max_cpus = float(result.stdout.strip())

    def _call(self, *args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.docker, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )

    def problems(self) -> list[ReadinessProblem]:
        if docker_binary() is None and not Path(self.docker).is_file():
            return [ReadinessProblem("the docker command is not installed", INSTALL_FIX)]
        result = self._info("{{json .ServerVersion}}")
        return [result] if isinstance(result, ReadinessProblem) else []

    def _info(self, template: str) -> subprocess.CompletedProcess[str] | ReadinessProblem:
        """``docker info --format template`` when the daemon answered it, else why not."""
        try:
            result = self._call("info", "--format", template, timeout=INFO_TIMEOUT_S)
        except subprocess.TimeoutExpired:  # an endpoint that accepted the connection, then hung
            return ReadinessProblem(
                f"the Docker daemon did not answer `docker info` within {INFO_TIMEOUT_S:g} s",
                self._daemon_fix(
                    "check it with `docker info` and restart it (sudo systemctl restart docker)",
                    "check that daemon with `docker info` and restart it (rootless Docker: "
                    "systemctl --user restart docker)",
                ),
            )
        except OSError as error:  # R2S_DOCKER (or the docker given) cannot be run
            fix = (
                f"point {DOCKER_ENV} at the docker command, or unset it"
                if os.environ.get(DOCKER_ENV)
                else INSTALL_FIX
            )
            return ReadinessProblem(f"the docker command cannot be run ({error})", fix)
        if result.returncode == 0:
            return result
        message = result.stderr.strip() or result.stdout.strip()
        if "permission denied" in message.lower():
            return ReadinessProblem(
                f"this user may not use the Docker daemon: {message}",
                self._daemon_fix(
                    "sudo usermod -aG docker $USER, then log in again (or run: newgrp docker)",
                    "ask whoever runs that daemon for access to it (the docker group is for "
                    "this machine's own daemon)",
                ),
            )
        return ReadinessProblem(
            f"the Docker daemon is not reachable: {message}",
            self._daemon_fix(
                "start it: sudo systemctl start docker",
                "start that daemon (rootless Docker: systemctl --user start docker)",
            ),
        )

    def _daemon_fix(self, own: str, other: str) -> str:
        """``own``, the fix for this machine's own daemon; or, where this shell's docker talks
        to another one, what chooses it, ``other`` (the fix for that daemon), and how to use
        this machine's own: ``systemctl`` and the docker group do not reach another daemon."""
        chosen = _chosen_daemon()
        if chosen:
            return (
                f"this shell chooses the daemon with {chosen}: {other}, correct {chosen}, or "
                f"unset {'them' if ' and ' in chosen else 'it'} to use this machine's own daemon"
            )
        context = self._context()
        if context is not None:
            return (
                f"the docker context {context!r} (chosen with `docker context use`) chooses the "
                f"daemon: {other}, or use this machine's own daemon: docker context use default"
            )
        return own

    def _context(self) -> str | None:
        """The docker context the client's configuration chose (``docker context use``) where
        no variable overrides it, unless it is this machine's own daemon (the default context);
        ``None`` too where docker cannot say (an older client)."""
        if os.environ.get("DOCKER_HOST") or os.environ.get("DOCKER_CONTEXT"):
            return None
        try:
            result = self._call("context", "show", timeout=CONTEXT_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired):
            return None
        names = result.stdout.split() if result.returncode == 0 else []
        return names[0] if names and names[0] != DEFAULT_CONTEXT else None

    def gpu_problems(self) -> list[ReadinessProblem]:
        result = self._info("{{json .Runtimes}}")
        if isinstance(result, ReadinessProblem):  # Docker itself did not answer
            return [result]
        if "nvidia" not in result.stdout:
            return [
                ReadinessProblem(
                    "the NVIDIA container runtime is not configured",
                    "install the NVIDIA Container Toolkit, then: sudo nvidia-ctk runtime "
                    "configure --runtime=docker && sudo systemctl restart docker",
                )
            ]
        return []

    def resolve_image(self, reference: str) -> ResolvedImage:
        if reference in self._images:
            return self._images[reference]
        result = self._call("image", "inspect", "--format", "{{json .}}", reference)
        if result.returncode != 0 and self._pull(reference):
            result = self._call("image", "inspect", "--format", "{{json .}}", reference)
        if result.returncode != 0:
            raise SetupError(
                f"docker image {reference!r} is not available locally",
                fix=_image_fix(reference),
            )
        data = cast(dict[str, Any], json.loads(result.stdout))
        digests = cast(list[str], data.get("RepoDigests") or [])
        identity = str(data.get("Id")) if data.get("Id") else None
        resolved = ResolvedImage(
            reference=reference, digest=digests[0] if digests else identity, content_id=identity
        )
        self._images[reference] = resolved
        return resolved

    def _pull(self, reference: str) -> bool:
        """Pull a missing registered, published image; ``False`` if that is not possible."""
        from real2simlab.envs import ENVIRONMENTS, pull_environment, registered_environment_for
        from real2simlab.envs.registry import pulling_disabled

        name = registered_environment_for(reference)
        if name is None or pulling_disabled() or not ENVIRONMENTS[name].publishable:
            return False
        try:
            pull_environment(name, docker=self.docker)
        except SetupError:
            return False
        return True

    def command(self, spec: SandboxSpec, name: str) -> list[str]:
        """The ``docker run`` command line for ``spec``. Pure: nothing is executed."""
        command = [
            self.docker,
            "run",
            "--name",
            name,
            "--init",
            "--user",
            spec.user or host_user(),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--pids-limit",
            str(spec.limits.pids),
            "--label",
            "real2simlab=1",
            *NO_DOCKER_LOG,
        ]
        if spec.network.mode is NetworkMode.NONE:
            command += ["--network", "none"]
        elif spec.network.mode is NetworkMode.ALLOWLIST:
            network = spec.network.docker_network
            if not network or spec.network.proxy_url is None:
                raise SetupError(
                    "allowlist networking needs a running egress proxy",
                    fix="run the agent through real2simlab.sandbox.egress.EgressProxy",
                )
            command += ["--network", network]
        if spec.limits.cpus is not None:
            cpus = spec.limits.cpus
            if self.max_cpus is not None:
                cpus = min(cpus, self.max_cpus)
            command += ["--cpus", f"{cpus:g}"]
        if spec.limits.memory_mb is not None:
            memory = f"{spec.limits.memory_mb}m"
            command += ["--memory", memory, "--memory-swap", memory]
        if spec.limits.gpus is not None:
            gpus = "all" if spec.limits.gpus == "all" else f'"device={spec.limits.gpus}"'
            command += ["--gpus", gpus]
        for path in spec.tmpfs:
            # mode=1777: writable by the container user even when the image has the directory
            # (runc would give the tmpfs that directory's mode, e.g. root-only 0755)
            command += ["--tmpfs", f"{path}:rw,exec,nosuid,size=4g,mode=1777"]
        for mount in spec.mounts:
            command += [
                "--mount",
                bind_mount(mount.source, mount.target, read_only=mount.read_only),
            ]
        env = {"HOME": "/tmp", "TMPDIR": "/tmp", **spec.network.proxy_env(), **spec.env}
        for key, value in env.items():
            command += ["--env", f"{key}={value}"]
        for key in spec.secret_env:
            command += ["--env", key]
        # The image this backend resolved (and the records name) is run by its ID, not by its
        # tag: a tag moved to another image while a command runs (a rebuild, a pull) must not
        # change what later containers run.
        resolved = self._images.get(spec.image)
        image = resolved.content_id if resolved is not None and resolved.content_id else spec.image
        command += ["--workdir", spec.workdir, image, *spec.argv]
        return command

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        for mount in spec.mounts:
            if not Path(mount.source).exists():
                return ExecResult(None, False, 0.0, f"mount source does not exist: {mount.source}")
        name = f"{spec.label}-{uuid.uuid4().hex[:12]}"
        self._learn_cpus()
        command = self.command(spec, name)
        client_env = {**os.environ, **spec.secret_env}

        def stop() -> None:
            subprocess.run(
                [self.docker, "kill", name],
                capture_output=True,
                timeout=60,
                check=False,
                stdin=subprocess.DEVNULL,
            )

        # No --rm: after it exits, the container says whether it ever started, which tells a
        # program that exited with 125 from a container Docker could not start.
        try:
            code, timed_out, seconds = run_logged(
                command,
                log_path,
                spec.limits.timeout_s,
                env=client_env,
                on_timeout=stop,
            )
            started = code != INFRA_EXIT_CODE or self._started(name)
        finally:
            subprocess.run(
                [self.docker, "rm", "-f", name],
                capture_output=True,
                timeout=120,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        if not started:
            tail = Path(log_path).read_text(encoding="utf-8", errors="replace")[-1000:]
            return ExecResult(None, False, seconds, f"docker could not start the container: {tail}")
        # The container's death by a signal is an exit status (128 + N); the Docker client's own
        # (the container was removed above) came from outside it.
        killed = killed_from_outside(code, "the Docker client")
        if killed is not None:
            return ExecResult(None, False, seconds, killed)
        return finished(code, timed_out, seconds, log_path, spec)

    def _started(self, name: str) -> bool:
        """Whether the container ``name`` exists and was started (ran the program)."""
        try:
            result = self._call("inspect", "--format", "{{.State.StartedAt}}", name, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return False
        started = result.stdout.strip()
        return result.returncode == 0 and bool(started) and started != NEVER_STARTED

    @contextmanager
    def session(self, spec: SandboxSpec) -> Generator[Session]:
        name = f"{spec.label}-{uuid.uuid4().hex[:12]}"
        detached = replace(spec, argv=("sleep", "infinity"))
        self._learn_cpus()
        command = self.command(detached, name)
        command.insert(2, "--detach")
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False,
            env={**os.environ, **spec.secret_env}, stdin=subprocess.DEVNULL
        )  # fmt: skip
        if result.returncode != 0:
            raise SetupError(f"cannot start the session container: {result.stderr.strip()}")
        try:
            timeout = str(int(spec.limits.timeout_s))
            prefix = (self.docker, "exec", "--workdir", spec.workdir, name)
            prefix += ("timeout", "--kill-after", "5", timeout, "bash", "-c")
            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": os.environ.get("HOME", "/tmp"),
                # The same daemon as the container was created on (DOCKER_HOST, a context).
                **docker_client_env(),
            }
            yield Session(prefix=prefix, env=env, cwd=Path.cwd())
        finally:
            subprocess.run(
                [self.docker, "rm", "-f", name],
                capture_output=True,
                timeout=120,
                check=False,
                stdin=subprocess.DEVNULL,
            )

    def info(self) -> dict[str, Any]:
        try:
            result = self._call("version", "--format", "{{.Server.Version}}", timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            result = None
        version = result.stdout.strip() if result is not None and result.returncode == 0 else None
        return {"backend": self.name, "isolated": True, "docker_version": version}


def _image_fix(reference: str) -> str:
    from real2simlab.envs import BUILD_HINT, ENVIRONMENTS, registered_environment_for
    from real2simlab.envs.registry import MIRRORS_ENV

    environment = registered_environment_for(reference)
    if environment is None:
        return f"docker pull {reference}"
    if not ENVIRONMENTS[environment].publishable:
        return f"r2s env build {environment}"
    return (
        f"r2s env pull {environment} (set {MIRRORS_ENV} to a registry mirror if the default "
        f"registries are slow), or build it: r2s env build {environment} ({BUILD_HINT})"
    )
