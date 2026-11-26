"""Runtime environments: container images that twins and agents run in.

An environment has a short name (``r2s-python``) used in task and twin files, and an image
tag (``ghcr.io/real2simhub/r2s-python:1.0``). Dockerfiles live in the repository's ``docker/``
directory and ship inside the wheel as ``real2simlab/envs/docker/``. Tasks may also name any
other image reference directly.
"""

from __future__ import annotations

import collections
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from real2simlab._fs import copy_tree
from real2simlab._project import IMAGE_REGISTRY
from real2simlab.errors import SetupError
from real2simlab.sandbox.process import PROBE_TIMEOUT_S
from real2simlab.twin.sdk import sdk_path

IMAGE_VERSION = "1.0"
PREFIX_ENV = "R2S_IMAGE_PREFIX"
DEFAULT_PREFIX = IMAGE_REGISTRY


@dataclass(frozen=True)
class Environment:
    """A registered environment and how to build its image."""

    name: str
    description: str
    depends_on: tuple[str, ...] = ()
    gpu: bool = False
    with_sdk: bool = False
    extra_files: tuple[str, ...] = ()
    """Files from ``real2simlab/envs/`` copied into the build context."""
    build_args: dict[str, str] = field(default_factory=lambda: {})
    publishable: bool = True
    """Whether the image may be published (only redistributable software); others are always
    built locally."""
    dockerfile: str | None = None
    """The directory under ``docker/`` that holds the Dockerfile, when it is not the
    environment's name (one Dockerfile on another base)."""
    accept_terms: tuple[str, str] | None = None
    """License terms the user must accept before the image is built (third-party software that
    requires it): the environment variable that records the acceptance (set to ``1``) and the
    URL of the terms. The build passes ``ACCEPT_EULA=Y`` once they are accepted."""

    @property
    def image(self) -> str:
        prefix = os.environ.get(PREFIX_ENV, DEFAULT_PREFIX)
        return f"{prefix}{self.name}:{IMAGE_VERSION}"


ENVIRONMENTS: dict[str, Environment] = {
    env.name: env
    for env in (
        Environment(
            "r2s-python",
            "CPU runtime: Python 3.11, NumPy, SciPy, ffmpeg, the twin SDK",
            with_sdk=True,
        ),
        Environment(
            "r2s-agent-codex",
            "r2s-python + Node 22 + the Codex CLI",
            depends_on=("r2s-python",),
        ),
        Environment(
            "r2s-agent-claude-code",
            "r2s-python + Node 22 + Claude Code (proprietary: built locally, never published)",
            depends_on=("r2s-python",),
            publishable=False,
        ),
        Environment(
            "r2s-agent-gemini-cli",
            "r2s-python + Node 22 + Gemini CLI",
            depends_on=("r2s-python",),
        ),
        Environment(
            "r2s-agent-opencode",
            "r2s-python + Node 22 + OpenCode 1.19.5",
            depends_on=("r2s-python",),
        ),
        Environment(
            "r2s-verifier",
            "r2s-python + Real2SimLab's dependencies, for verifiers of Harbor exports",
            depends_on=("r2s-python",),
        ),
        Environment(
            "r2s-blender-4.0",
            "r2s-python + Blender 4.0.2 (PhysCodeGym)",
            depends_on=("r2s-python",),
            gpu=True,
            extra_files=("blender_scene.py",),
        ),
        Environment(
            "r2s-physimcode",
            "r2s-python + Matplotlib: the runtime of PhySimCode's 2D (SciPy) programs",
            depends_on=("r2s-python",),
        ),
        Environment(
            "r2s-video2world",
            "r2s-python + ManiSkill 3.0.1 / SAPIEN 3.0.3: Video2World's evaluation "
            "environment (built locally; SAPIEN needs the GPU's Vulkan)",
            depends_on=("r2s-python",),
            gpu=True,
        ),
        Environment(
            "r2s-video2world-isaac",
            "CUDA 12.8 + Isaac Sim 5.1 + RoboDojo's IsaacLab and cuRobo: the scorer image of "
            "Video2World's RoboDojo and in-house evaluators (NVIDIA's license: built locally "
            "after accepting it, never published)",
            gpu=True,
            publishable=False,
            accept_terms=(
                "R2S_ACCEPT_NVIDIA_EULA",
                "https://docs.isaacsim.omniverse.nvidia.com/latest/common/"
                "NVIDIA_Omniverse_License_Agreement.html",
            ),
        ),
        Environment(
            "r2s-agent-codex-video2world",
            "r2s-video2world + Node 22 + the Codex CLI: Codex that can validate and replay its "
            "Video2World package (built locally)",
            depends_on=("r2s-video2world",),
            dockerfile="r2s-agent-codex",
        ),
        Environment(
            "r2s-agent-codex-physimcode",
            "r2s-physimcode + Node 22 + the Codex CLI: Codex that can run PhySimCode's programs "
            "(built locally)",
            depends_on=("r2s-physimcode",),
            dockerfile="r2s-agent-codex",
        ),
        Environment(
            "r2s-agent-codex-blender",
            "r2s-blender-4.0 + Node 22 + the Codex CLI: Codex with Blender in its session "
            "(built locally)",
            depends_on=("r2s-blender-4.0",),
            gpu=True,
            dockerfile="r2s-agent-codex",
        ),
    )
}


def agent_environment(agent_image: str, task_environment: str) -> str:
    """The image an agent should work in for a task: its own image when the task runs in that
    image's base; a registered variant of it built on the task's runtime when there is one
    (e.g. ``r2s-agent-codex-blender`` for tasks that run in ``r2s-blender-4.0``); the task's
    image itself when it is not a Lab environment (a benchmark's own agent image, such as
    4DCodeBench's); otherwise the agent's own image."""
    if task_environment not in ENVIRONMENTS:
        return task_environment
    own = ENVIRONMENTS.get(agent_image)
    if own is None or task_environment in own.depends_on[:1]:
        return agent_image
    for env in ENVIRONMENTS.values():
        same_dockerfile = (env.dockerfile or env.name) == agent_image
        if same_dockerfile and env.depends_on[:1] == (task_environment,):
            return env.name
    return agent_image


def get_environment(name: str) -> Environment:
    from real2simlab.errors import UnknownNameError

    if name not in ENVIRONMENTS:
        raise UnknownNameError("environment", name, ENVIRONMENTS)
    return ENVIRONMENTS[name]


BUILD_HINT = (
    "behind a slow network add --pip-index-url https://<PyPI mirror>/simple/ "
    "--apt-mirror http://<Debian mirror> --npm-registry https://<npm mirror> --network host; "
    "behind a proxy add --network host --build-arg HTTPS_PROXY --build-arg HTTP_PROXY"
)


def pull_environment(name: str, *, docker: str | None = None, log: Any = None) -> str:
    """Pull an environment's published image (see ``real2simlab.envs.registry``); return its
    canonical name. Raises ``SetupError`` with the build command when no source has it."""
    from real2simlab.envs.registry import MIRRORS_ENV, pull

    environment = get_environment(name)
    if not environment.publishable:
        raise SetupError(
            f"{environment.image} is not published (it contains software that may not be "
            "redistributed)",
            fix=f"r2s env build {name}",
        )
    result = pull(
        environment.image,
        f"{environment.name}:{IMAGE_VERSION}",
        docker=docker or _docker(),
        log=log,
    )
    if result.source is None:
        raise SetupError(
            f"could not pull {environment.image} (tried {', '.join(result.tried)}: {result.error})",
            fix=f"set {MIRRORS_ENV} to a registry mirror that has it, or build it: "
            f"r2s env build {name}  ({BUILD_HINT})",
        )
    return result.image


def image_reference(name_or_reference: str) -> str:
    """The image for a task's ``environment``: a registered name's tag, or the reference itself."""
    if name_or_reference in ENVIRONMENTS:
        return ENVIRONMENTS[name_or_reference].image
    return name_or_reference


def registered_environment_for(reference: str) -> str | None:
    """The environment name whose image is ``reference``, if any."""
    for environment in ENVIRONMENTS.values():
        if reference in (environment.image, environment.name):
            return environment.name
    return None


def dockerfiles_root() -> Path:
    """Where the Dockerfiles are: inside the installed package, or the source checkout."""
    packaged = Path(__file__).with_name("docker")
    if packaged.is_dir():
        return packaged
    source = Path(__file__).resolve().parents[2] / "docker"
    if source.is_dir():
        return source
    raise SetupError(
        "the Dockerfiles are missing from this installation", fix="reinstall real2simlab"
    )


def build_context(name: str, target: Path) -> Path:
    """Assemble the build context of ``name`` in ``target``; return it."""
    environment = get_environment(name)
    source = dockerfiles_root() / (environment.dockerfile or name)
    if not (source / "Dockerfile").is_file():
        raise SetupError(f"no Dockerfile for {name} in {source}", fix="reinstall real2simlab")
    copy_tree(source, target, exist_ok=True)  # writable, also from a read-only installation
    if environment.with_sdk:
        shutil.copyfile(sdk_path(), target / "r2s_twin.py")
    for name in environment.extra_files:
        shutil.copyfile(Path(__file__).with_name(name), target / name)
    return target


def build_order(names: list[str]) -> list[str]:
    """``names`` and their dependencies, dependencies first."""
    ordered: list[str] = []

    def visit(name: str) -> None:
        for dependency in get_environment(name).depends_on:
            visit(dependency)
        if name not in ordered:
            ordered.append(name)

    for name in names:
        visit(name)
    return ordered


def terms_accepted(environment: Environment) -> bool:
    """Whether ``environment`` has no license terms to accept, or the user accepted them."""
    return environment.accept_terms is None or os.environ.get(environment.accept_terms[0]) == "1"


def build_hint(environment: Environment) -> str:
    """The command that builds ``environment``'s image."""
    if environment.accept_terms is None:
        return f"r2s env build {environment.name}"
    variable, url = environment.accept_terms
    return f"{variable}=1 r2s env build {environment.name}  (accepts {url})"


def accepted_terms(environment: Environment) -> dict[str, str]:
    """The build arguments that record the acceptance of ``environment``'s license terms
    (``{}`` when it has none). Raises ``SetupError`` until the user has accepted them."""
    if environment.accept_terms is None:
        return {}
    variable, url = environment.accept_terms
    if not terms_accepted(environment):
        raise SetupError(
            f"{environment.name} contains software whose license must be accepted before it is "
            f"downloaded: read {url}",
            fix=f"if you accept it: {variable}=1 r2s env build {environment.name}",
        )
    return {"ACCEPT_EULA": "Y"}


def build_command(
    name: str,
    context: Path,
    *,
    docker: str = "docker",
    pip_index_url: str | None = None,
    npm_registry: str | None = None,
    apt_mirror: str | None = None,
    no_cache: bool = False,
    network: str | None = None,
    build_args: Mapping[str, str | None] | None = None,
) -> list[str]:
    """The ``docker build`` command line. A build argument without a value (``None``) is passed
    by name only: Docker takes its value from this process's environment (a proxy), so the
    value is never on the command line."""
    environment = get_environment(name)
    command = [docker, "build", "--tag", environment.image]
    args: dict[str, str | None] = dict(environment.build_args)
    if environment.depends_on:
        args["BASE_IMAGE"] = get_environment(environment.depends_on[0]).image
    if pip_index_url:
        args["PIP_INDEX_URL"] = pip_index_url
    if npm_registry:
        args["NPM_REGISTRY"] = npm_registry
    if apt_mirror:
        args["APT_MIRROR"] = apt_mirror
    args.update(build_args or {})
    for key, value in sorted(args.items()):
        command += ["--build-arg", key if value is None else f"{key}={value}"]
    if no_cache:
        command.append("--no-cache")
    if network:
        command += ["--network", network]
    return [*command, str(context)]


def _docker() -> str:
    from real2simlab.sandbox.docker import docker_binary

    return docker_binary() or "docker"


def build(
    name: str,
    *,
    docker: str | None = None,
    pip_index_url: str | None = None,
    npm_registry: str | None = None,
    apt_mirror: str | None = None,
    no_cache: bool = False,
    network: str | None = None,
    build_args: Mapping[str, str | None] | None = None,
    log: Path | None = None,
) -> str:
    """Build the image of ``name`` (not its dependencies); return the image tag."""
    environment = get_environment(name)
    accepted = accepted_terms(environment)
    if accepted:
        build_args = {**accepted, **(build_args or {})}
    with tempfile.TemporaryDirectory(prefix=f"r2s-build-{name}-") as tmp:
        context = build_context(name, Path(tmp) / "context")
        command = build_command(
            name,
            context,
            docker=docker or _docker(),
            pip_index_url=pip_index_url,
            npm_registry=npm_registry,
            apt_mirror=apt_mirror,
            no_cache=no_cache,
            network=network,
            build_args=build_args,
        )
        returncode, tail = _run_build(command, log)
    if returncode != 0:
        hints = diagnose_build_failure(tail, network=network, build_args=build_args)
        raise SetupError(
            f"building {environment.image} failed (exit {returncode})",
            fix="; ".join(hints)
            if hints
            else "see the build output above; behind a slow network see `r2s env build --help`",
        )
    return environment.image


def _run_build(command: list[str], log: Path | None) -> tuple[int, str]:
    """Run a build, streaming its output (to the terminal or ``log``); keep the tail."""
    tail: collections.deque[str] = collections.deque(maxlen=400)
    with contextlib.ExitStack() as stack:
        out = (
            stack.enter_context(Path(log).open("w", encoding="utf-8"))
            if log is not None
            else sys.stdout
        )
        process = stack.enter_context(
            subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                stdin=subprocess.DEVNULL,
            )
        )
        assert process.stdout is not None
        for line in process.stdout:
            tail.append(line)
            out.write(line)
            out.flush()
    return process.returncode, "".join(tail)


NETWORK_ERRORS = (
    "temporary failure in name resolution", "could not resolve", "name or service not known",
    "read timed out", "readtimeouterror", "connection timed out", "timed out", "etimedout",
    "econnreset", "connection reset", "connection refused", "network is unreachable",
    "tls handshake timeout", "failed to fetch", "unable to connect", "err_socket_timeout",
    "npm error network", "npm err! network", "i/o timeout", "no route to host",
)  # fmt: skip
"""Lower-case fragments of build output that mean a download failed."""


DOCKER_KEYS = ("registry-1.docker.io", "docker.io/", "auth.docker.io")
PIP_HOSTS = ("pypi.org", "files.pythonhosted")
PIP_KEYS = ("pip._vendor", "retrying (retry(")
"""pip's own failure messages; a ``Downloading https://files.pythonhosted.org/...`` line of a
step that went on to something else is not one."""
APT_KEYS = ("deb.debian.org", "archive.ubuntu.com", "security.debian.org", "apt-get", "w: failed")
NPM_KEYS = ("registry.npmjs.org", "npm error", "npm err")
CONTEXT_LINES = 15
"""Lines before a network error read to tell what was being downloaded: the output of the
failing step, not an earlier step that succeeded."""


def diagnose_build_failure(
    output: str,
    *,
    network: str | None = None,
    build_args: Mapping[str, str | None] | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Fixes for the network failures recognisable in a build's output, most specific first.

    Each failure is judged by the lines around it, so a download that failed after pip
    installed everything (a model's assets, from their own host) is not taken for pip's. The
    last fix fits how the build ran (``network``, ``build_args``) and the proxy variables set
    in ``environ`` (this process's environment): never a flag the build already had."""
    import re

    lines = output.lower().splitlines()
    failing = [i for i, line in enumerate(lines) if any(e in line for e in NETWORK_ERRORS)]
    if not failing:
        return []
    found: dict[str, None] = {}
    other_hosts: dict[str, None] = {}
    for index in failing:
        line = lines[index]
        context = "\n".join(lines[max(0, index - CONTEXT_LINES) : index + 1])
        if any(key in line for key in DOCKER_KEYS) and (
            "failed to resolve source metadata" in line or "failed to do request" in line
        ):
            found["docker"] = None
        elif any(key in line for key in PIP_HOSTS) or any(key in context for key in PIP_KEYS):
            found["pip"] = None
        elif any(key in context for key in NPM_KEYS):
            found["npm"] = None
        elif any(key in context for key in APT_KEYS):
            found["apt"] = None
        else:
            known = (*PIP_HOSTS, *DOCKER_KEYS, *APT_KEYS, *NPM_KEYS)
            for host in re.findall(r"https?://([a-z0-9.-]+)", context):
                if not any(key in host for key in known):
                    other_hosts[host] = None
    hints: list[str] = []
    if "docker" in found:
        hints.append(
            "pulling a base image failed: configure a Docker registry mirror "
            '("registry-mirrors" in /etc/docker/daemon.json) or pull the base image by hand'
        )
    if "pip" in found:
        hints.append("pip downloads failed: add --pip-index-url https://<a PyPI mirror>/simple/")
    if "apt" in found:
        hints.append("apt downloads failed: add --apt-mirror http://<a Debian/Ubuntu mirror>")
    if "npm" in found:
        hints.append("npm downloads failed: add --npm-registry https://<an npm mirror>")
    if other_hosts:
        hints.append(
            f"a download from {', '.join(other_hosts)} failed: the build fetches it itself, not "
            "through pip, apt or npm (a mirror option does not help); the build must reach it"
        )
    hints.append(
        _network_hint(network, build_args or {}, os.environ if environ is None else environ)
    )
    return hints


PROXY_VARIABLES = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
)  # fmt: skip
"""Proxy settings a build can be given (Docker's predefined build arguments, which it keeps out
of the image's history)."""


def _network_hint(
    network: str | None, build_args: Mapping[str, str | None], environ: Mapping[str, str]
) -> str:
    """How to give a build that could not download the network this machine has: its proxy
    (by name; a value may hold a password) and the host's network, unless it had them."""
    host = network == "host"
    proxies = [name for name in PROXY_VARIABLES if environ.get(name)]
    missing = [name for name in proxies if name not in build_args]
    if missing:
        flags = " ".join(f"--build-arg {name}" for name in missing)
        return (
            f"this shell sets {', '.join(missing)}, which the build does not use: add {flags} "
            "(Docker takes the values from this shell)"
            + ("" if host else " and, for a proxy on this machine (127.0.0.1), --network host")
        )
    if not host:
        return (
            "if the build cannot reach the network at all (or only slowly), also add --network host"
        )
    if proxies:
        return (
            "the build already used this machine's network and its proxy "
            f"({', '.join(proxies)}): check that the proxy reaches the host above, or use a mirror"
        )
    return (
        "the build already used this machine's network (--network host), which cannot reach the "
        "host above either: set a proxy (HTTPS_PROXY, HTTP_PROXY) in this shell and add "
        "--build-arg HTTPS_PROXY --build-arg HTTP_PROXY, or use a mirror"
    )


def image_status(name: str, docker: str | None = None) -> str | None:
    """The image id of ``name`` if it is built, else ``None``."""
    result = subprocess.run(
        [
            docker or _docker(),
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            get_environment(name).image,
        ],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=PROBE_TIMEOUT_S,
    )
    return result.stdout.strip() if result.returncode == 0 else None
