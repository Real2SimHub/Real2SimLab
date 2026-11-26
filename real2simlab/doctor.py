"""``r2s doctor``: what this machine can and cannot run yet, with a fix for each problem.

Every check returns :class:`Check` records; ``required`` ones decide the exit status, the others
are advice (an agent without credentials, an adapter whose extra is not installed, little disk).
No check prints or stores a credential: agent checks only report whether one was found.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from real2simlab._project import install_command

GB = 1024**3
CACHE_MIN_GB = 20
DOCKER_MIN_GB = 30
ENCODER = "libx264"
"""The video encoder the Lab writes captures and renders with (``real2simlab.media``)."""

EXTRAS: dict[str, tuple[tuple[str, ...], str]] = {
    "iris": (("huggingface_hub",), "IRIS adapter (downloads from Hugging Face)"),
    "physcodegym": (("huggingface_hub", "cv2"), "PhysCodeGym adapter and its scorer"),
    "fourdcodebench": (("huggingface_hub", "h5py", "hdf5plugin"), "4DCodeBench adapter"),
    "physicsiq": (
        ("huggingface_hub", "cv2", "pandas", "tqdm"),
        "Physics-IQ adapter and its scorer",
    ),
    "video2world": (("huggingface_hub",), "Video2World adapter (downloads from Hugging Face)"),
    "physimcode": (("huggingface_hub",), "PhySimCode adapter (downloads from Hugging Face)"),
    "mini-swe": (("minisweagent",), "the mini_swe method"),
    "docs": (("mkdocs", "mkdocstrings"), "building the documentation site"),
}
"""Optional dependency group → (modules it provides, what needs it)."""

AGENTS = ("codex", "claude_code", "gemini_cli", "opencode")
"""Agent methods whose credentials can be checked without starting them."""


@dataclass(frozen=True)
class Check:
    check: str
    ok: bool
    detail: str
    fix: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        # One check is one line (``[ok ] name: detail``, its fix on the next): what a program
        # printed on several lines is kept, joined on that line, in the text and the JSON.
        object.__setattr__(self, "detail", one_line(self.detail))
        if self.fix is not None:
            object.__setattr__(self, "fix", one_line(self.fix))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def one_line(text: str) -> str:
    """``text`` on one line: its non-empty lines, stripped, joined with "; "."""
    return "; ".join(line.strip() for line in text.splitlines() if line.strip())


def gpu_summary(output: str) -> str:
    """What ``nvidia-smi`` (:data:`GPU_QUERY`) printed, one line per GPU (``name, driver``),
    said on one line: how many GPUs, each model once with its count, and the driver."""
    rows = [line.rsplit(",", 1) for line in output.splitlines() if line.strip()]
    names = [row[0].strip() for row in rows]
    drivers = sorted({row[1].strip() for row in rows if len(row) == 2})
    counts = {name: names.count(name) for name in names}
    models = ", ".join(name if n == 1 else f"{name} x{n}" for name, n in counts.items())
    driver = f", driver {' / '.join(drivers)}" if drivers else ""
    return f"{len(rows)} GPU{'s' if len(rows) != 1 else ''}: {models}{driver}"


def system_checks() -> list[Check]:
    from real2simlab import __version__

    checks = [
        Check(
            "real2simlab",
            True,
            f"{__version__} on Python {platform.python_version()} ({platform.system()})",
        )
    ]
    from real2simlab.media import FFMPEG_ENV

    ffmpeg = None
    for tool in ("ffmpeg", "ffprobe"):
        check, found = media_tool_check(tool)
        checks.append(check)
        ffmpeg = found if tool == "ffmpeg" else ffmpeg
    git = shutil.which("git")
    checks.append(
        Check(
            "git",
            git is not None,
            git or "not found",
            None if git else "install git (Debian/Ubuntu: sudo apt-get install git)",
            required=False,
        )
    )
    if ffmpeg:
        checks.append(encoder_check(ffmpeg))
    on_path = shutil.which("ffmpeg")
    if os.environ.get(FFMPEG_ENV) and (
        on_path is None or ffmpeg is None or os.path.realpath(on_path) != os.path.realpath(ffmpeg)
    ):
        checks.append(local_ffmpeg_check(on_path))
    return checks


def media_tool_check(tool: str) -> tuple[Check, str | None]:
    """``ffmpeg`` or ``ffprobe`` as the Lab calls them (``media.tool``: ``R2S_FFMPEG`` or
    ``R2S_FFPROBE`` when set, else the one on PATH), and its path if it can run."""
    from real2simlab.errors import SetupError
    from real2simlab.media import FFMPEG_ENV, FFPROBE_ENV
    from real2simlab.media import tool as media_tool

    variable = FFMPEG_ENV if tool == "ffmpeg" else FFPROBE_ENV
    try:
        path = media_tool(tool)
    except SetupError as error:
        return Check(tool, False, "not found", error.fix), None
    found = shutil.which(path)  # a variable may name anything
    if not os.environ.get(variable):
        return Check(tool, found is not None, found or path), found
    if found is None:
        fix = f"point {variable} at the {tool} to use, or unset it to use the one on PATH"
        return Check(tool, False, f"{variable}={path} is not an executable file", fix), None
    return Check(tool, True, f"{found} ({variable})"), found


def local_ffmpeg_check(on_path: str | None) -> Check:
    """The ffmpeg programs find on the ``local`` backend: the one on PATH (they run with this
    machine's PATH; ``R2S_FFMPEG`` is the Lab's own). Docker and bwrap use their image's."""
    return Check(
        "ffmpeg on PATH",
        on_path is not None,
        f"{on_path}, for programs on the local backend"
        if on_path
        else "not found: programs on the local backend find no ffmpeg (docker and bwrap use "
        "their image's)",
        None if on_path else "to run programs on the local backend, put ffmpeg on PATH",
        required=False,
    )


def encoder_check(ffmpeg: str) -> Check:
    """Some ffmpeg builds (e.g. minimal conda ones) lack the H.264 encoder the Lab writes with."""
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return Check(f"ffmpeg {ENCODER}", False, f"ffmpeg -encoders failed: {error}")
    found = any(
        len(parts) > 1 and parts[1] == ENCODER
        for parts in (line.split() for line in result.stdout.splitlines())
    )
    return Check(
        f"ffmpeg {ENCODER}",
        found,
        "available" if found else f"this ffmpeg has no {ENCODER} encoder",
        None if found else "install an ffmpeg built with libx264 (e.g. the distribution's ffmpeg)",
    )


def free_space_check(
    name: str,
    path: Path,
    minimum_gb: float,
    why: str,
    usage: Callable[[Path], Any] = shutil.disk_usage,
) -> Check:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = usage(probe).free / GB
    except OSError as error:
        return Check(name, False, f"{path}: {error}", required=False)
    ok = free >= minimum_gb
    return Check(
        name,
        ok,
        f"{free:.0f} GB free at {path}",
        None if ok else f"free space or move it ({why}); {minimum_gb} GB or more is advisable",
        required=False,
    )


def storage_checks(
    docker: str | None = None, usage: Callable[[Path], Any] = shutil.disk_usage
) -> list[Check]:
    from real2simlab.vendor import CACHE_ENV, cache_dir

    cache = cache_dir()
    checks: list[Check] = []
    try:
        cache.mkdir(parents=True, exist_ok=True)
        writable = os.access(cache, os.W_OK)
    except OSError:
        writable = False
    checks.append(
        Check(
            "cache directory",
            writable,
            f"{cache} " + ("(writable)" if writable else "is not writable"),
            None if writable else f"set {CACHE_ENV} to a writable directory with space",
        )
    )
    checks.append(
        free_space_check(
            "cache space", cache, CACHE_MIN_GB, f"set {CACHE_ENV} to a larger disk", usage
        )
    )
    if docker:
        root = docker_root(docker)
        if root is not None:
            checks.append(
                free_space_check(
                    "docker space",
                    root,
                    DOCKER_MIN_GB,
                    "images take 1-3 GB each; 4DCodeBench's take about 60 GB",
                    usage,
                )
            )
    return checks


def docker_root(docker: str) -> Path | None:
    try:
        result = subprocess.run(
            [docker, "info", "--format", "{{.DockerRootDir}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    root = result.stdout.strip()
    return Path(root) if result.returncode == 0 and root else None


def rootfs_check(name: str) -> Check:
    """Whether the bwrap backend has the image of the environment ``name`` unpacked. Advice:
    bwrap is one of the backends twins can run on."""
    from real2simlab.envs import image_reference
    from real2simlab.envs.rootfs import find_rootfs
    from real2simlab.errors import SetupError

    image = image_reference(name)
    check = f"image {name} (bwrap)"
    try:
        unpacked = find_rootfs(image) is not None
    except SetupError as error:  # a damaged record of an unpacked image
        return Check(check, False, str(error).split("\n  fix:")[0], error.fix, required=False)
    return Check(
        check,
        unpacked,
        image + (" unpacked" if unpacked else " not unpacked"),
        None
        if unpacked
        else f"r2s env unpack {name} (without Docker here: add --from-tar with a tarball of "
        f"`r2s env export {name}` written where Docker exists, or --from-url)",
        required=False,
    )


def extras_checks(find: Callable[[str], Any] = importlib.util.find_spec) -> list[Check]:
    checks: list[Check] = []
    for extra, (modules, needed_by) in EXTRAS.items():
        missing = [module for module in modules if find(module) is None]
        checks.append(
            Check(
                f"extra {extra}",
                not missing,
                f"installed, for {needed_by}" if not missing else f"not installed, for {needed_by}",
                None if not missing else install_command(extra),
                required=False,
            )
        )
    return checks


def agent_checks(names: Iterable[str] = AGENTS) -> list[Check]:
    """Whether each agent's credentials are configured. Nothing is fetched: no provider auth
    command is run and no login file or key is read (``AgentMethod.credential_status``)."""
    from real2simlab.agents.base import AgentMethod
    from real2simlab.errors import SetupError
    from real2simlab.methods import get_method

    checks: list[Check] = []
    for name in names:
        try:
            method = get_method(name)
            status = method.credential_status() if isinstance(method, AgentMethod) else None
        except SetupError as error:
            message = str(error).split("\n  fix:")[0]
            checks.append(Check(f"agent {name}", False, message, error.fix, required=False))
            continue
        except Exception as error:  # a broken configuration is reported, not raised
            message = f"{type(error).__name__}: {error}"
            checks.append(Check(f"agent {name}", False, message, required=False))
            continue
        if status is None:
            checks.append(Check(f"agent {name}", True, "credentials not checked", required=False))
            continue
        checks.append(
            Check(f"agent {name}", status.available, status.detail, status.fix, required=False)
        )
    return checks


GPU_QUERY = ("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader")


def docker_gpu_check(docker: str, images: Iterable[str]) -> Check:
    """Whether Docker runs GPU containers, found out through Docker itself: its GPUs are its
    daemon's (another machine with ``DOCKER_HOST``, or a host whose own ``nvidia-smi`` cannot
    run here), so what this host's driver says does not decide it. ``nvidia-smi`` is run in the
    first of ``images`` that is built; with none, the configured runtime is all that is known."""
    from real2simlab.sandbox.base import get_backend

    problems = get_backend("docker").gpu_problems()
    if problems:
        return Check("gpu via docker", False, problems[0].message, problems[0].fix, required=False)
    for image in images:
        try:
            built = subprocess.run(
                [docker, "image", "inspect", "--format", "{{.Id}}", image],
                capture_output=True, text=True, check=False, timeout=60,
                stdin=subprocess.DEVNULL,
            )  # fmt: skip
        except (OSError, subprocess.TimeoutExpired):
            continue
        if built.returncode != 0:
            continue
        command = [docker, "run", "--rm", "--network", "none", "--gpus", "all", image,
                   *GPU_QUERY]  # fmt: skip
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=180,
                stdin=subprocess.DEVNULL,
            )  # fmt: skip
        except (OSError, subprocess.TimeoutExpired) as error:
            return Check("gpu via docker", False, f"a GPU container did not finish: {error}",
                         "check the NVIDIA Container Toolkit and the daemon's driver",
                         required=False)  # fmt: skip
        output = (result.stdout.strip() or result.stderr.strip())[-300:]
        if result.returncode == 0 and output:
            # every GPU's line (the whole output, not its cut), else what it printed
            summary = gpu_summary(result.stdout) if result.stdout.strip() else output
            return Check("gpu via docker", True, f"ready ({summary})", required=False)
        return Check(
            "gpu via docker",
            False,
            f"nvidia-smi in {image} with --gpus all failed (status {result.returncode}): {output}",
            "install or repair the NVIDIA driver and Container Toolkit where the Docker daemon "
            "runs, then restart it",
            required=False,
        )
    return Check(
        "gpu via docker",
        True,
        "the NVIDIA container runtime is configured (not tried: no Lab image is built)",
        required=False,
    )


def format_checks(checks: Iterable[Check]) -> str:
    lines: list[str] = []
    for check in checks:
        mark = "ok " if check.ok else ("ERR" if check.required else "-- ")
        lines.append(f"[{mark}] {check.check}: {check.detail}")
        if not check.ok and check.fix:
            lines.append(f"       fix: {check.fix}")
    return "\n".join(lines)
