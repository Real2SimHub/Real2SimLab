"""Shared fixtures: synthetic tasks (built once and cached), backends, small helpers."""

from __future__ import annotations

import fcntl
import functools
import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from real2simlab.formats.tracks import Camera, ObjectTrack, Tracks
from real2simlab.sandbox.local import LocalBackend

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "real2simlab" / "benchmarks" / "synthetic"
PHENOMENA_FOR_TESTS = ("bouncing_ball", "pendulum", "incline_slider")


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(SYNTHETIC.rglob("*.py")):
        digest.update(path.read_bytes())
    for path in sorted((ROOT / "real2simlab" / "task").rglob("*.py")):
        digest.update(path.read_bytes())
    digest.update((ROOT / "real2simlab" / "twin" / "sdk" / "r2s_twin.py").read_bytes())
    return digest.hexdigest()[:16]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Live tests (they spend model calls) run only when ``-m`` names them: a ``-m`` on the
    command line replaces the default ``-m "not live"``, so ``pytest -m docker`` selected
    them too."""
    if "live" in (config.option.markexpr or "") and "not live" not in config.option.markexpr:
        return
    skip = pytest.mark.skip(reason="live: spends model calls; run with pytest -m live")
    for item in items:
        if item.get_closest_marker("live") is not None:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _bytecode_is_current() -> None:
    """Compile the package before any test: a test that starts Python under a small file-size
    limit (``ulimit -f``, with ``SIGXFSZ`` ignored) would otherwise write the first bytecode of
    a module it is the first to import cut at the limit, and every later import of that module
    would fail ("marshal data too short"). With bytecode off (``PYTHONDONTWRITEBYTECODE``, which
    the processes tests start inherit) nothing is written: the source tree stays as it is."""
    if sys.dont_write_bytecode:
        return
    import compileall

    compileall.compile_dir(ROOT / "real2simlab", quiet=1)


@pytest.fixture(scope="session")
def synthetic_tasks(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic tasks for three phenomena in both tiers, cached across test sessions."""
    cache = ROOT / ".pytest_cache" / f"r2s-synthetic-{_source_hash()}"
    cache.parent.mkdir(parents=True, exist_ok=True)
    # One session builds at a time (the Docker tests beside the rest): the others wait for it.
    with (cache.parent / "r2s-synthetic.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (cache / ".complete").is_file():
            from real2simlab.benchmarks.synthetic import build_suite

            shutil.rmtree(cache, ignore_errors=True)
            build_suite(cache, select=PHENOMENA_FOR_TESTS)
            (cache / ".complete").write_text("ok")
    target = tmp_path_factory.mktemp("tasks")
    shutil.copytree(cache, target / "synthetic", ignore=shutil.ignore_patterns(".complete"))
    return target / "synthetic"


@pytest.fixture(scope="session")
def bouncing_task(synthetic_tasks: Path) -> Path:
    return synthetic_tasks / "bouncing_ball" / "calibrated"


@pytest.fixture
def task_copy(bouncing_task: Path, tmp_path: Path) -> Path:
    """A private, modifiable copy of the calibrated bouncing-ball task."""
    target = tmp_path / "task"
    shutil.copytree(bouncing_task, target)
    return target


@pytest.fixture(autouse=True)
def _no_image_pulls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never pull images from registries (tests of pulling unset this themselves)."""
    monkeypatch.setenv("R2S_NO_PULL", "1")


HOST_ENDPOINTS = ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE", "ANTHROPIC_API_BASE")
"""A host's endpoint variables, which agent adapters refuse to leave out (``endpoint_set``)."""


@pytest.fixture(autouse=True)
def _no_host_endpoints(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests do not depend on the endpoints this host's own agents use (tests of them set these
    themselves); live runs keep them."""
    if request.node.get_closest_marker("live") is None:
        for name in HOST_ENDPOINTS:
            monkeypatch.delenv(name, raising=False)


@functools.cache
def _credential_variables() -> frozenset[str]:
    """Every variable an agent adapter reads a model key or token from: its module's
    ``KEY_ENV``, ``KEY_ENVS`` and ``PROVIDER_KEYS``, and its options' default ``api_key_env``."""
    from real2simlab import plugins
    from real2simlab.agents.base import AgentMethod

    names: set[str] = set()
    for name in plugins.names("method"):
        try:
            method = plugins.load("method", name)
        except Exception:  # an adapter that cannot load here reads nothing here either
            continue
        if not (isinstance(method, type) and issubclass(method, AgentMethod)):
            continue
        module = sys.modules[method.__module__]
        names.update(getattr(module, "KEY_ENVS", ()))
        names.update(getattr(module, "PROVIDER_KEYS", {}).values())
        if isinstance(key := getattr(module, "KEY_ENV", None), str):
            names.add(key)
        field = method.Options.model_fields.get("api_key_env")
        if field is not None and isinstance(field.default, str):
            names.add(field.default)
    return frozenset(names)


@pytest.fixture(autouse=True)
def _no_credentials_from_the_shell(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests never see a model key or token of the shell they run in: a developer's
    ``ANTHROPIC_AUTH_TOKEN`` or ``CLAUDE_CODE_OAUTH_TOKEN`` would change what an agent's
    credential checks find. A test that needs one sets it itself; ``live`` tests keep them."""
    if request.node.get_closest_marker("live") is not None:
        return
    for name in _credential_variables():
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def local() -> LocalBackend:
    return LocalBackend(quiet=True)


def docker_command() -> str | None:
    docker = os.environ.get("R2S_DOCKER") or shutil.which("docker")
    if docker is None:
        return None
    try:
        result = subprocess.run([docker, "info"], capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return docker if result.returncode == 0 else None


def docker_image_ready(image: str = "ghcr.io/real2simhub/r2s-python:1.0") -> bool:
    docker = docker_command()
    if docker is None:
        return False
    result = subprocess.run([docker, "image", "inspect", image], capture_output=True, check=False)
    return result.returncode == 0


@pytest.fixture
def docker_backend() -> Iterator[object]:
    if not docker_image_ready():
        pytest.skip("Docker or the r2s-python image is not available (r2s env build r2s-python)")
    from real2simlab.sandbox.docker import DockerBackend

    yield DockerBackend()


BWRAP_SANDBOX = (
    "--unshare-all", "--die-with-parent", "--new-session", "--ro-bind", "/", "/",
    "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
)  # fmt: skip
"""What every sandbox of the bwrap backend has (all namespaces, a new ``/proc``, ``/dev``, a
``/tmp`` tmpfs), written out here: the backend's own command line is what its tests test."""


def bwrap_unavailable() -> str | None:
    """Why bwrap cannot start a sandbox here (bwrap's own message), or ``None`` when it can.

    Asked of bwrap itself (``R2S_BWRAP``, else ``bwrap`` on ``PATH``), never of the backend under
    test: when its readiness check decided, a regression that broke every sandbox it makes
    skipped its integration tests instead of failing them."""
    bwrap = os.environ.get("R2S_BWRAP") or shutil.which("bwrap")
    if bwrap is None:
        return "bwrap is not installed"
    try:
        result = subprocess.run(
            [bwrap, *BWRAP_SANDBOX, "true"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"{bwrap} cannot be run ({error})"
    if result.returncode != 0:
        return result.stderr.strip() or f"{bwrap} exited with status {result.returncode}"
    return None


def bwrap_ready() -> bool:
    """Whether bwrap can start a sandbox here (see ``bwrap_unavailable``)."""
    return bwrap_unavailable() is None


def make_tracks(
    positions: dict[str, np.ndarray], fps: float = 30.0, *, camera: Camera | None = None
) -> Tracks:
    """Tracks with a default camera looking at the origin from -y."""
    if camera is None:
        pose = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -2.0],
                [0.0, -1.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        camera = Camera(
            intrinsics=np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]]),
            width=320,
            height=240,
            extrinsics=pose,
        )
    return Tracks(
        fps=fps,
        objects={
            name: ObjectTrack(position=np.asarray(value, dtype=float))
            for name, value in positions.items()
        },
        camera=camera,
    )


def falling(num_frames: int = 60, height: float = 0.5, fps: float = 30.0) -> np.ndarray:
    t = np.arange(num_frames) / fps
    z = np.maximum(height - 0.5 * 9.81 * t**2, 0.05)
    return np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
