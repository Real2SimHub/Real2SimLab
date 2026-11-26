"""``r2s doctor``: each check reports a problem with a fix, and never shows a credential."""

from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest

from real2simlab import doctor
from real2simlab._project import install_command
from real2simlab.cli.main import main

Usage = namedtuple("Usage", "total used free")


def fake_ffmpeg(tmp_path: Path, encoders: str) -> str:
    script = tmp_path / "ffmpeg"
    script.write_text(f"#!/bin/sh\ncat <<'EOF'\n{encoders}\nEOF\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def test_encoder_check(tmp_path: Path) -> None:
    good = fake_ffmpeg(tmp_path, " V....D libx264              libx264 H.264")
    assert doctor.encoder_check(good).ok
    (tmp_path / "ffmpeg").unlink()
    bad = fake_ffmpeg(tmp_path, " V....D mpeg4                MPEG-4 part 2")
    check = doctor.encoder_check(bad)
    assert not check.ok and check.required and check.fix and "libx264" in check.fix


def _media_tools(directory: Path, encoders: str) -> Path:
    """A fake ffmpeg (listing ``encoders``) and ffprobe in ``directory``; they need no PATH."""
    directory.mkdir(parents=True)
    for name in ("ffmpeg", "ffprobe"):
        (directory / name).write_text(f"#!/bin/sh\necho '{encoders}'\n")
        (directory / name).chmod(0o755)
    return directory


def test_doctor_checks_the_ffmpeg_the_lab_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Lab runs ``R2S_FFMPEG`` and ``R2S_FFPROBE`` when they are set (``media.tool``): the
    doctor checks those and says so, and the ffmpeg on PATH apart, as what programs on the
    local backend run (it is no longer reported as the Lab's)."""
    x264 = _media_tools(tmp_path / "x264", " V....D libx264              libx264 H.264")
    plain = _media_tools(tmp_path / "plain", " V....D mpeg4                MPEG-4 part 2")
    (tmp_path / "empty").mkdir()

    def checks(path: Path, lab: Path | None) -> dict[str, doctor.Check]:
        monkeypatch.setenv("PATH", str(path))
        for name, variable in (("ffmpeg", "R2S_FFMPEG"), ("ffprobe", "R2S_FFPROBE")):
            if lab is None:
                monkeypatch.delenv(variable, raising=False)
            else:
                monkeypatch.setenv(variable, str(lab / name))
        return {c.check: c for c in doctor.system_checks()}

    # PATH's ffmpeg has no libx264 and the Lab's has: the Lab can render.
    found = checks(plain, x264)
    assert found["ffmpeg"].ok and found["ffmpeg"].detail == f"{x264 / 'ffmpeg'} (R2S_FFMPEG)"
    assert found["ffprobe"].ok and found["ffprobe"].detail.endswith("(R2S_FFPROBE)")
    assert found["ffmpeg libx264"].ok
    local = found["ffmpeg on PATH"]
    assert local.ok and not local.required and str(plain / "ffmpeg") in local.detail
    # The other way round: the Lab's has none, whatever PATH's has.
    found = checks(x264, plain)
    assert not found["ffmpeg libx264"].ok and found["ffmpeg libx264"].required
    # Only the variables: the Lab can run; programs on the local backend find no ffmpeg.
    found = checks(tmp_path / "empty", x264)
    assert all(found[name].ok for name in ("ffmpeg", "ffprobe", "ffmpeg libx264"))
    local = found["ffmpeg on PATH"]
    assert not local.ok and not local.required and "PATH" in (local.fix or "")
    # A variable naming nothing that runs is a failed check that names it.
    monkeypatch.setenv("R2S_FFMPEG", str(tmp_path / "missing"))
    broken = {c.check: c for c in doctor.system_checks()}["ffmpeg"]
    assert not broken.ok and broken.required and "R2S_FFMPEG" in (broken.fix or "")
    # Without the variables, PATH's is the Lab's: checked once.
    found = checks(x264, None)
    assert found["ffmpeg"].detail == str(x264 / "ffmpeg") and "ffmpeg on PATH" not in found
    assert found["ffmpeg libx264"].ok


def test_free_space_is_advice(tmp_path: Path) -> None:
    low = doctor.free_space_check(
        "cache space", tmp_path / "not" / "yet", 20, "move it", usage=lambda p: Usage(0, 0, 2**30)
    )
    assert not low.ok and not low.required and "1 GB free" in low.detail and low.fix
    high = doctor.free_space_check(
        "cache space", tmp_path, 20, "move it", usage=lambda p: Usage(0, 0, 100 * 2**30)
    )
    assert high.ok and high.fix is None


def test_storage_checks_use_the_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    checks = {c.check: c for c in doctor.storage_checks(None)}
    assert checks["cache directory"].ok and (tmp_path / "cache").is_dir()
    assert "docker space" not in checks


def test_extras_checks_name_the_pip_extra() -> None:
    checks = {c.check: c for c in doctor.extras_checks(find=lambda name: None)}
    iris = checks["extra iris"]
    assert not iris.ok and not iris.required and iris.fix == install_command("iris")
    assert all(c.ok for c in doctor.extras_checks(find=lambda name: object()))


def test_pyright_passes_without_the_optional_extras() -> None:
    """pyright, a check every pull request passes, gives the same result in any install the
    guides describe (``.[dev,docs]``, CI's ``.[dev,mini-swe]``): each import of a module that
    only an optional extra provides (``doctor.EXTRAS``) is marked, as ``cv2``'s are, so a
    missing extra is not a missing import."""
    import ast

    optional = {module for modules, _ in doctor.EXTRAS.values() for module in modules}
    package = Path(doctor.__file__).parent
    unmarked: list[str] = []
    for path in sorted(package.rglob("*.py")):
        lines = path.read_text().splitlines()
        for node in ast.walk(ast.parse("\n".join(lines))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                modules = [node.module]
            else:
                continue
            marked = "# pyright: ignore[reportMissingImports]" in lines[node.lineno - 1]
            if any(m.split(".")[0] in optional for m in modules) and not marked:
                unmarked.append(f"{path.relative_to(package.parent)}:{node.lineno}")
    assert not unmarked


def test_agent_checks_never_show_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from real2simlab.agents.claude_code import KEY_ENVS

    for name in KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    (missing,) = doctor.agent_checks(["claude_code"])
    assert not missing.ok and not missing.required
    assert missing.fix == "export ANTHROPIC_API_KEY=..." and "fix:" not in missing.detail
    secret = "sk-ant-doctor-test-1234567890abcdef"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    (found,) = doctor.agent_checks(["claude_code"])
    assert found.ok and secret not in found.detail and "ANTHROPIC_API_KEY" in found.detail


def test_codex_check_never_runs_the_auth_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "codex"
    home.mkdir()
    marker = tmp_path / "ran"
    (home / "config.toml").write_text(
        'model_provider = "corp"\n'
        "[model_providers.corp]\n"
        'base_url = "https://llm.example.org/v1"\n'
        f'auth = {{ command = "touch", args = ["{marker}"] }}\n'
    )
    monkeypatch.setenv("CODEX_HOME", str(home))
    (check,) = doctor.agent_checks(["codex"])
    assert check.ok and "not run" in check.detail and not marker.exists()
    (home / "config.toml").write_text("")  # the default provider: a login file
    (check,) = doctor.agent_checks(["codex"])
    assert not check.ok and "codex login" in (check.fix or "")
    (home / "auth.json").write_text('{"tokens": {"access_token": "secret-value"}}')
    (check,) = doctor.agent_checks(["codex"])
    assert check.ok and "secret-value" not in check.detail


def test_format_marks_required_failures() -> None:
    text = doctor.format_checks(
        [
            doctor.Check("a", True, "fine"),
            doctor.Check("b", False, "broken", "do this"),
            doctor.Check("c", False, "optional", "maybe", required=False),
        ]
    )
    assert "[ok ] a: fine" in text and "[ERR] b: broken\n       fix: do this" in text
    assert "[-- ] c: optional" in text


def test_doctor_command_json(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))  # not the host's configuration
    secret = "sk-doctor-cli-test-1234567890abcdef"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    code = main(["doctor", "--json"])
    output = capsys.readouterr().out
    checks = json.loads(output)
    names = {c["check"] for c in checks}
    assert {"real2simlab", "cache directory", "extra iris", "agent opencode"} <= names
    assert secret not in output
    required_ok = all(c["ok"] for c in checks if c["required"])
    assert code == (0 if required_ok else 1)
    assert os.environ["OPENAI_API_KEY"] == secret


def test_install_hints_point_at_this_clone() -> None:
    """Real2SimLab is not on PyPI: every hint installs from the clone it was imported from."""
    root = Path(__file__).resolve().parents[1]
    assert shlex.split(install_command("iris")) == ["pip", "install", "-e", f"{root}[iris]"]
    assert shlex.split(install_command()) == ["pip", "install", "-e", str(root)]


def _script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _ldconfig_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, found: bool, on_path: bool = False
) -> Path:
    """PATH as Debian gives an ordinary user (no sbin, no ldconfig on it; with ``on_path``,
    ldconfig's directory too); the system directories where the Lab also looks hold an
    ``ldconfig`` that lists a fake libcuda (or, with ``found`` false, none). Returns PATH's
    first directory."""
    from real2simlab.sandbox import bwrap

    bin_dir, sbin = tmp_path / "bin", tmp_path / "sbin"
    bin_dir.mkdir()
    sbin.mkdir()
    if found:
        libraries = tmp_path / "lib"
        libraries.mkdir()
        (libraries / "libcuda.so.1").write_bytes(b"not really a driver")
        listed = f"\tlibcuda.so.1 (libc6,x86-64) => {libraries / 'libcuda.so.1'}"
        _script(sbin / "ldconfig", f"echo '1 libs found in cache'\necho '{listed}'\n")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{sbin}" if on_path else str(bin_dir))
    monkeypatch.setattr(bwrap, "LDCONFIG_DIRS", (str(sbin),), raising=False)
    monkeypatch.setattr(bwrap, "_driver_version", lambda: "1.0")
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    return bin_dir


def test_bwrap_finds_ldconfig_where_debian_keeps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Debian keeps ldconfig in /sbin, which an ordinary user's PATH leaves out: GPU runs on
    bwrap still find the host's NVIDIA driver libraries (they stopped at "ldconfig does not
    exist")."""
    from real2simlab.sandbox import bwrap

    _ldconfig_elsewhere(tmp_path, monkeypatch, found=True)
    libraries = bwrap.driver_library_dir()
    assert (libraries / "libcuda.so.1").read_bytes() == b"not really a driver"


def test_the_first_gpu_runs_at_once_share_one_copy_of_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first GPU runs on bwrap start together (an experiment's --concurrency, commands on
    one cache): one copies the driver libraries while the others wait for it. Each cleared
    the copy another was making, and the experiment stopped ("<version>.partial already
    exists", or a copy into a directory gone)."""
    import threading
    import time

    from real2simlab.sandbox import bwrap

    _ldconfig_elsewhere(tmp_path, monkeypatch, found=True, on_path=True)
    first, second_done = threading.Lock(), threading.Event()
    copy = bwrap.shutil.copy2

    def slow_first_copy(source: object, target: object, **options: object) -> object:
        if first.acquire(blocking=False):  # the first run's first file: let the second one go
            second_done.wait(timeout=3)
        return copy(source, target, **options)

    monkeypatch.setattr(bwrap.shutil, "copy2", slow_first_copy)
    outcomes: list[object] = []

    def run(done: threading.Event | None = None) -> None:
        try:
            outcomes.append(bwrap.driver_library_dir())
        except Exception as error:
            outcomes.append(error)
        finally:
            if done is not None:
                done.set()

    one = threading.Thread(target=run)
    one.start()
    while one.is_alive() and not first.locked():  # until the first run copies
        time.sleep(0.01)
    two = threading.Thread(target=run, args=(second_done,))
    two.start()
    one.join(timeout=60)
    two.join(timeout=60)
    assert len(outcomes) == 2 and all(isinstance(o, Path) for o in outcomes), outcomes
    assert (bwrap.driver_library_dir() / "libcuda.so.1").read_bytes() == b"not really a driver"


def test_doctor_gpu_reports_a_missing_ldconfig_and_checks_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No ldconfig anywhere: ``r2s doctor --gpu`` still prints every check, and "gpu via
    bwrap" says what is missing and how to fix it; the whole command stopped instead, without
    a single check, and ``--json`` printed nothing."""
    from real2simlab.sandbox import bwrap

    bin_dir = _ldconfig_elsewhere(tmp_path, monkeypatch, found=False)
    _script(bin_dir / "nvidia-smi", "echo 'NVIDIA Test GPU, 1.0'\n")
    for name in ("R2S_BWRAP", "R2S_DOCKER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(bwrap.BwrapBackend, "problems", lambda self: [])
    nodes = bwrap.glob.glob
    monkeypatch.setattr(
        bwrap.glob, "glob", lambda pattern, **kw: ["/dev/nvidia0"] if "nvidia" in pattern
        else nodes(pattern, **kw)
    )  # fmt: skip
    main(["doctor", "--gpu", "--json"])
    checks = {c["check"]: c for c in json.loads(capsys.readouterr().out)}
    assert {"real2simlab", "nvidia driver", "gpu via local"} <= set(checks), sorted(checks)
    gpu = checks["gpu via bwrap"]
    assert not gpu["ok"] and "ldconfig" in gpu["detail"] and "PATH" in (gpu["fix"] or ""), gpu
    assert "fix:" not in gpu["detail"], gpu  # the fix is given once, as the fix


FAKE_DOCKER = """case "$1" in
  image) echo sha256:0123 ;;
  run) echo "$R2S_FAKE_GPU" ; exit "${R2S_FAKE_GPU_STATUS:-0}" ;;
  info) echo '{"nvidia": {"path": "nvidia-container-runtime"}}' ;;
  *) echo '{}' ;;
esac
"""


@pytest.mark.parametrize(("status", "ok"), [(0, True), (1, False)])
def test_docker_gpus_are_judged_by_docker_not_by_this_host(
    status: int,
    ok: bool,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """This host's nvidia-smi fails (no driver here, or a sandbox without its devices) while
    Docker's daemon has GPUs: whether Docker runs GPU containers is found out through Docker."""
    from real2simlab.sandbox.docker import DockerBackend

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _script(bin_dir / "nvidia-smi", "echo 'NVIDIA-SMI has failed'\nexit 9\n")
    docker = _script(bin_dir / "docker", FAKE_DOCKER)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("R2S_DOCKER", str(docker))
    monkeypatch.setenv("R2S_FAKE_GPU", "NVIDIA Test GPU, 1.0")
    monkeypatch.setenv("R2S_FAKE_GPU_STATUS", str(status))
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(DockerBackend, "problems", lambda self: [])
    main(["doctor", "--gpu", "--json"])
    checks = {c["check"]: c for c in json.loads(capsys.readouterr().out)}
    assert not checks["nvidia driver"]["ok"] and "status 9" in checks["nvidia driver"]["detail"]
    docker_gpu = checks["gpu via docker"]
    assert docker_gpu["ok"] is ok and "on this host" not in docker_gpu["detail"], docker_gpu
    assert ("NVIDIA Test GPU" in docker_gpu["detail"]) is ok or "status 1" in docker_gpu["detail"]
    assert not checks["gpu via bwrap"]["ok"]  # bwrap uses this host's driver and devices


CANNOT_CONNECT = (
    "echo 'Cannot connect to the Docker daemon at unix:///nonexistent. Is the docker daemon "
    "running?' >&2\nexit 1\n"
)


@pytest.mark.parametrize("variable", ["DOCKER_HOST", "DOCKER_CONTEXT"])
def test_a_daemon_this_shell_chooses_is_what_the_fix_names(
    variable: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DOCKER_HOST (or DOCKER_CONTEXT) that points elsewhere: the fix was to start this
    machine's own daemon, which cannot help; it names the variable. Asked about GPUs, the same,
    not "install the NVIDIA Container Toolkit"."""
    from real2simlab.sandbox.docker import DockerBackend

    for name in ("DOCKER_HOST", "DOCKER_CONTEXT"):
        monkeypatch.delenv(name, raising=False)
    docker = str(_script(tmp_path / "docker", CANNOT_CONNECT))
    (plain,) = DockerBackend(docker=docker).problems()
    assert plain.fix == "start it: sudo systemctl start docker"
    monkeypatch.setenv(variable, "unix:///nonexistent" if variable == "DOCKER_HOST" else "other")
    backend = DockerBackend(docker=docker)
    for problem in (*backend.problems(), *backend.gpu_problems()):
        assert "not reachable" in problem.message and "NVIDIA" not in problem.message, problem
        assert variable in (problem.fix or "") and not (problem.fix or "").startswith("start it")


def test_a_daemon_that_does_not_answer_is_not_a_missing_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`docker info` timed out (an endpoint that took the connection and hung): Docker is
    installed, so the fix is not to install it; the GPU check says the same instead of
    raising."""
    from real2simlab.sandbox.docker import INSTALL_FIX, DockerBackend

    def hangs(self: DockerBackend, *args: str, timeout: float = 60) -> Any:
        raise subprocess.TimeoutExpired(["docker", *args], timeout)

    monkeypatch.setattr(DockerBackend, "_call", hangs)
    backend = DockerBackend(docker=str(_script(tmp_path / "docker", "exit 0\n")))
    for problem in (*backend.problems(), *backend.gpu_problems()):
        assert "did not answer" in problem.message and problem.fix != INSTALL_FIX, problem


ANOTHER_SOCKET = "unix:///run/user/1000/docker.sock"
DAEMON_FAILURES = {
    "down": f"Cannot connect to the Docker daemon at {ANOTHER_SOCKET}. Is the docker daemon "
    "running?",
    "no access": f"permission denied while trying to connect to the Docker daemon socket at "
    f"{ANOTHER_SOCKET}: dial unix /run/user/1000/docker.sock: connect: permission denied",
}
OWN_FIXES = (
    "start it: sudo systemctl start docker",
    "sudo usermod -aG docker $USER, then log in again (or run: newgrp docker)",
)


@pytest.mark.parametrize("failure", sorted(DAEMON_FAILURES))
@pytest.mark.parametrize(
    ("variables", "context", "named"),
    [
        ({}, "qa-remote", "the docker context 'qa-remote'"),  # `docker context use qa-remote`
        ({"DOCKER_HOST": ANOTHER_SOCKET}, "default", "with DOCKER_HOST"),
        ({}, "default", None),
        ({"DOCKER_CONTEXT": "default"}, "qa-remote", None),  # it overrides the configuration
        ({"DOCKER_HOST": "unix:///var/run/docker.sock"}, "qa-remote", None),
    ],
    ids=["context use", "DOCKER_HOST", "own", "DOCKER_CONTEXT=default", "DOCKER_HOST=own"],
)
def test_the_fix_is_for_the_daemon_docker_talks_to(
    failure: str,
    variables: dict[str, str],
    context: str,
    named: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daemon chosen with `docker context use` (written in the client's configuration, no
    variable: rootless Docker, Docker Desktop) was down or refused this user: the fix was to
    start this machine's own daemon or join the docker group, which governs this machine's own
    socket only. So was a daemon DOCKER_HOST chooses that refused this user. The fix names what
    chooses the daemon and how to use this machine's own; for this machine's own daemon (also
    when a variable names it) it stays systemctl or the docker group."""
    from real2simlab.sandbox.docker import DockerBackend

    for name in ("DOCKER_HOST", "DOCKER_CONTEXT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    docker = _script(
        tmp_path / "docker",
        f'if [ "$1 $2" = "context show" ]; then echo {context}; exit 0; fi\n'
        f"echo {shlex.quote(DAEMON_FAILURES[failure])} >&2\nexit 1\n",
    )
    backend = DockerBackend(docker=str(docker))
    for problem in (*backend.problems(), *backend.gpu_problems()):
        fix = problem.fix or ""
        if named is None:
            assert fix in OWN_FIXES, problem
            continue
        assert named in fix and fix not in OWN_FIXES, problem
        assert "sudo systemctl" not in fix and "usermod" not in fix, problem
        assert "docker context use default" in fix or "unset it" in fix, problem


def test_a_chosen_daemon_that_does_not_answer_is_named_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`docker info` timed out where DOCKER_HOST or a docker context chooses the daemon: the fix
    began with restarting this machine's own (sudo systemctl restart docker), not the one that
    hung. It names what chooses the daemon first."""
    from real2simlab.sandbox.docker import DockerBackend

    def hangs(self: DockerBackend, *args: str, timeout: float = 60) -> Any:
        if args[:2] == ("context", "show"):  # the client's configuration: no daemon is asked
            return subprocess.CompletedProcess(["docker", *args], 0, "qa-remote\n", "")
        raise subprocess.TimeoutExpired(["docker", *args], timeout)

    for name in ("DOCKER_HOST", "DOCKER_CONTEXT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(DockerBackend, "_call", hangs)
    backend = DockerBackend(docker=str(_script(tmp_path / "docker", "exit 0\n")))
    for variable, start in ((None, "the docker context 'qa-remote'"),
                            ("DOCKER_HOST", "this shell chooses the daemon with DOCKER_HOST")):  # fmt: skip
        if variable:
            monkeypatch.setenv(variable, "tcp://127.0.0.1:2375")
        (problem,) = backend.problems()
        assert "did not answer" in problem.message, problem
        assert (problem.fix or "").startswith(start) and "sudo" not in (problem.fix or ""), problem


def test_doctor_asks_each_backend_once_and_says_what_bwrap_needs(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A hung Docker daemon made `r2s doctor` wait for `docker info` twice. Where bwrap can
    start its sandbox, doctor did not say whether it has its runtime image (and Docker's fix
    said nothing of bwrap); where it cannot, it still reported GPUs ready through it."""
    from real2simlab.envs import rootfs
    from real2simlab.sandbox.base import ReadinessProblem
    from real2simlab.sandbox.bwrap import BwrapBackend

    calls = tmp_path / "calls"
    docker = _script(tmp_path / "docker", f'echo "$1" >> {calls}\n' + CANNOT_CONNECT)
    for name in ("DOCKER_HOST", "DOCKER_CONTEXT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("R2S_DOCKER", str(docker))
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(BwrapBackend, "problems", lambda self: [])
    main(["doctor", "--json"])
    checks = {c["check"]: c for c in json.loads(capsys.readouterr().out)}
    assert calls.read_text().split().count("info") == 1
    image = checks["image r2s-python (bwrap)"]
    assert not image["ok"] and not image["required"], image
    assert "r2s env unpack r2s-python" in image["fix"] and "--from-tar" in image["fix"], image
    assert "bwrap" not in checks["backend docker"]["fix"]
    unpacked = rootfs.Rootfs("ghcr.io/real2simhub/r2s-python:1.0", "sha256:x", tmp_path, {}, "/")
    monkeypatch.setattr(rootfs, "find_rootfs", lambda reference: unpacked)
    main(["doctor", "--json"])
    checks = {c["check"]: c for c in json.loads(capsys.readouterr().out)}
    assert checks["image r2s-python (bwrap)"]["ok"]
    assert "use --backend bwrap" in checks["backend docker"]["fix"], checks["backend docker"]
    # bwrap cannot start its sandbox: nor GPU runs in it, whatever its devices and driver
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _script(bin_dir / "nvidia-smi", "echo 'NVIDIA Test GPU, 1.0'\n")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    cannot = [ReadinessProblem("bwrap cannot mount a new /proc here", "use --backend docker")]
    monkeypatch.setattr(BwrapBackend, "problems", lambda self: cannot)
    monkeypatch.setattr(BwrapBackend, "gpu_problems", lambda self: [])
    main(["doctor", "--gpu", "--json"])
    checks = {c["check"]: c for c in json.loads(capsys.readouterr().out)}
    assert checks["nvidia driver"]["ok"]
    assert not checks["gpu via bwrap"]["ok"] and "not ready" in checks["gpu via bwrap"]["detail"]


@pytest.mark.parametrize("command", ["eval", "run", "twin check", "twin run", "train reward"])
def test_a_command_whose_backend_cannot_run_names_one_that_can(
    command: str,
    bouncing_task: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a machine without Docker whose bwrap was ready (`r2s doctor` said so), each command
    that runs on Docker unless given another backend (an evaluation, `r2s run` of an experiment
    that names none, a twin check or run, a reward) stopped before anything ran with Docker's
    own fix only: start or install it, both needing root. The fix also names `--backend bwrap`
    there, and `--backend docker` where only Docker can run, once; never a backend that cannot
    run here, nor `local`, which isolates nothing. The command does not change the backend."""
    from real2simlab.sandbox.base import ReadinessProblem
    from real2simlab.sandbox.bwrap import BwrapBackend
    from real2simlab.sandbox.docker import DockerBackend

    for name in ("DOCKER_HOST", "DOCKER_CONTEXT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("R2S_DOCKER", str(_script(tmp_path / "docker", CANNOT_CONNECT)))
    experiment = tmp_path / "experiment.toml"
    experiment.write_text(
        f'name = "default-backend"\n[tasks]\npaths = ["{bouncing_task}"]\n'
        '[[methods]]\nname = "static"\n'
    )
    twin = bouncing_task / "hidden" / "generator"
    argv = {
        "eval": ["eval", "--tasks", str(bouncing_task), "--method", "static"],
        "run": ["run", str(experiment)],
        "twin check": ["twin", "check", str(twin)],
        "twin run": ["twin", "run", str(twin)],
        "train reward": ["train", "reward", "--task", str(bouncing_task), "--solution", str(twin)],
    }[command] + ([] if command == "train reward" else ["--out", str(tmp_path / "out")])
    from real2simlab import doctor

    bwrap_cannot = [ReadinessProblem("bwrap cannot create namespaces", "allow them for bwrap")]
    unpacked = doctor.Check("image r2s-python (bwrap)", True, "unpacked", required=False)
    for bwrap_problems, bwrap_named in (([], True), (bwrap_cannot, False)):
        monkeypatch.setattr(BwrapBackend, "problems", lambda self, found=bwrap_problems: found)
        monkeypatch.setattr(doctor, "rootfs_check", lambda name: unpacked)
        assert main(argv) == 1
        err = capsys.readouterr().err
        assert "error: backend 'docker' is not ready" in err and "Traceback" not in err, err
        assert "fix: start it: sudo systemctl start docker" in err, err
        assert ("; or use --backend bwrap, which can run here" in err) is bwrap_named, err
        assert "--backend local" not in err, err
        assert not (tmp_path / "out").exists()
    # bwrap can start its sandbox but its runtime is not unpacked: said with how to unpack it,
    # as `r2s doctor` says it (it was "which is ready here", and then the runtime was missing).
    monkeypatch.setattr(BwrapBackend, "problems", lambda self: [])
    monkeypatch.setattr(doctor, "rootfs_check", lambda name: doctor.Check(
        "image r2s-python (bwrap)", False, "not unpacked", "r2s env unpack r2s-python",
        required=False))  # fmt: skip
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "; or use --backend bwrap once its runtime is unpacked: r2s env unpack r2s-python" in err
    # The other way round: bwrap chosen and unable to start, Docker ready.
    monkeypatch.setattr(BwrapBackend, "problems", lambda self: bwrap_cannot)
    monkeypatch.setattr(DockerBackend, "problems", lambda self: [])
    assert main([*argv, "--backend", "bwrap"]) == 1
    err = capsys.readouterr().err
    assert "error: backend 'bwrap' is not ready: bwrap cannot create namespaces" in err, err
    assert "fix: allow them for bwrap; or use --backend docker, which can run here" in err, err
    # bwrap's own fix names Docker already (it cannot mount a new /proc here): named once.
    no_proc = [ReadinessProblem("bwrap cannot mount a new /proc here", "use --backend docker")]
    monkeypatch.setattr(BwrapBackend, "problems", lambda self: no_proc)
    assert main([*argv, "--backend", "bwrap"]) == 1
    err = capsys.readouterr().err
    assert "fix: use --backend docker" in err and err.count("--backend docker") == 1, err
    # ... and Docker is not asked whether it can run (a Docker that hangs made this take 40 s)
    monkeypatch.setattr(DockerBackend, "problems", lambda self: pytest.fail("Docker was asked"))
    assert main([*argv, "--backend", "bwrap"]) == 1
    assert "fix: use --backend docker" in capsys.readouterr().err


def test_doctor_gpu_says_each_check_on_one_line_with_several_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``nvidia-smi`` prints a line for each GPU: with two, the second stood under
    ``[ok ] nvidia driver: ...`` on a line of its own, like no check, and the JSON detail held
    a newline (R-A1-01). The GPUs are said on the check's line; any detail is one line."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gpu = "echo 'NVIDIA A100-SXM4-80GB, 580.95.05'\n"
    _script(bin_dir / "nvidia-smi", gpu * 2)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    main(["doctor", "--gpu"])
    text = capsys.readouterr().out
    assert "[ok ] nvidia driver: 2 GPUs: NVIDIA A100-SXM4-80GB x2, driver 580.95.05\n" in text
    assert all(line.startswith(("[", " " * 7)) for line in text.splitlines() if line), text
    main(["doctor", "--gpu", "--json"])
    checks = json.loads(capsys.readouterr().out)
    assert not [c for c in checks if "\n" in c["detail"] + (c["fix"] or "")], checks


def test_a_check_is_one_line_whatever_its_program_printed() -> None:
    from real2simlab.doctor import Check, format_checks, gpu_summary

    check = Check("x", False, "first\n\n  second  \n", "do this\nthen that")
    assert (check.detail, check.fix) == ("first; second", "do this; then that")
    assert format_checks([check]) == "[ERR] x: first; second\n       fix: do this; then that"
    two = "NVIDIA A, 1.0\nNVIDIA B, 1.0\nNVIDIA A, 1.0\n"
    assert gpu_summary(two) == "3 GPUs: NVIDIA A x2, NVIDIA B, driver 1.0"
    assert gpu_summary("NVIDIA A, 1.0") == "1 GPU: NVIDIA A, driver 1.0"
