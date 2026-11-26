"""Running frozen twins: outcomes on the local backend, isolation on the Docker backend."""

from __future__ import annotations

import contextlib
import json
import socket
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from real2simlab.runner import (
    FrozenTwinModifiedError,
    RunLimits,
    RunStatus,
    read_record,
    run_twin,
)
from real2simlab.sandbox.base import (
    Limits,
    Mount,
    NetworkMode,
    NetworkPolicy,
    SandboxSpec,
    rewrite_paths,
)
from real2simlab.twin import TwinManifest, save_twin
from real2simlab.twin.sdk import sdk_path

GOOD = """
import r2s_twin, numpy as np
twin = r2s_twin.Twin()
z = np.full(10, 0.5 + twin.seed / 100)
cam = r2s_twin.Camera.look_at((0, -2, 0.5), (0, 0, 0.5), width=64, height=48)
twin.write_tracks(fps=10, positions={"ball": np.c_[0 * z, 0 * z, z]}, camera=cam)
"""


def make_twin(tmp_path: Path, source: str, name: str = "twin"):
    program = tmp_path / f"{name}-src"
    program.mkdir()
    (program / "run.py").write_text(source)
    (program / "r2s_twin.py").write_text(sdk_path().read_text())
    manifest = TwinManifest(
        twin_id=f"test/{name}",
        entrypoint=("python", "./run.py"),
        environment={"image": "r2s-python"},
        output_format="tracks",
        origin={"method": "test", "task_id": "t", "created_at": datetime.now(UTC)},
    )
    return save_twin(manifest, program, tmp_path / name)


def test_successful_runs_and_records(tmp_path: Path, local) -> None:
    twin = make_twin(tmp_path, GOOD)
    result = run_twin(twin, tmp_path / "run", backend=local, seed=3)
    assert result.succeeded and result.record.seed == 3 and result.record.output_fingerprint
    record = read_record(tmp_path / "run")
    assert record == result.record and record.label == "run"
    assert str(tmp_path) not in json.dumps(record.to_json())
    rerun = run_twin(twin, tmp_path / "rerun", backend=local, label="rerun", seed=3)
    assert rerun.record.label == "rerun"
    assert rerun.record.output_fingerprint == result.record.output_fingerprint
    other = run_twin(twin, tmp_path / "other", backend=local, seed=4)
    assert other.record.output_fingerprint != result.record.output_fingerprint
    with pytest.raises(FileExistsError):
        run_twin(twin, tmp_path / "run", backend=local)
    with pytest.raises(ValueError, match="seed"):
        run_twin(twin, tmp_path / "s", backend=local, seed=-1)


@pytest.mark.parametrize(
    ("source", "status"),
    [
        ("raise SystemExit(2)", RunStatus.FAILED),
        ("import time; time.sleep(30)", RunStatus.TIMED_OUT),
        (
            "import os, pathlib; pathlib.Path(os.environ['R2S_OUTPUT_DIR'], 'meta.json').write_text('{}')",
            RunStatus.INVALID_OUTPUT,
        ),
    ],
)
def test_failure_outcomes(tmp_path: Path, local, source: str, status: RunStatus) -> None:
    twin = make_twin(tmp_path, source)
    result = run_twin(twin, tmp_path / "run", backend=local, limits=RunLimits(timeout_s=2))
    assert result.record.status is status and result.record.problems


def test_modified_twin_is_refused(tmp_path: Path, local) -> None:
    twin = make_twin(tmp_path, GOOD)
    (twin.program_dir / "run.py").write_text("print('changed')")
    with pytest.raises(FrozenTwinModifiedError):
        run_twin(twin, tmp_path / "run", backend=local)


def test_a_twin_with_a_link_put_in_it_is_refused_as_changed(tmp_path: Path, local) -> None:
    """A symbolic link put into a loaded twin (by another program) is a change like any other:
    the twin is refused as changed, before or after its run, not a crash of the Lab."""
    twin = make_twin(tmp_path, GOOD)
    (twin.program_dir / "data.py").symlink_to(twin.program_dir / "run.py")
    with pytest.raises(FrozenTwinModifiedError):
        run_twin(twin, tmp_path / "run", backend=local)


def test_rewrite_paths() -> None:
    mounts = [
        Mount(Path("/host/out"), "/r2s/output", read_only=False),
        Mount(Path("/host/prog"), "/r2s/program"),
    ]
    assert rewrite_paths("/r2s/output/x.json", mounts) == "/host/out/x.json"
    assert rewrite_paths("/r2s/outputs", mounts) == "/r2s/outputs"


# --- Docker isolation -------------------------------------------------------------------------

PROBE = """
import json, os, pathlib, socket, sys
out = pathlib.Path(os.environ["R2S_OUTPUT_DIR"])
report = {}
try:
    socket.create_connection(("1.1.1.1", 53), timeout=3)
    report["network"] = "reachable"
except OSError as error:
    report["network"] = "unreachable"
try:
    pathlib.Path("written.txt").write_text("x")
    report["program"] = "writable"
except OSError:
    report["program"] = "read-only"
report["hidden"] = os.path.exists(sys.argv[1])
report["cwd"] = os.getcwd()
report["uid"] = os.getuid()
(out / "probe.json").write_text(json.dumps(report))
"""


@pytest.mark.docker
def test_docker_isolation(tmp_path: Path, docker_backend, task_copy: Path) -> None:
    hidden = task_copy / "hidden"
    program = tmp_path / "probe"
    program.mkdir()
    (program / "probe.py").write_text(PROBE)
    spec = SandboxSpec(
        image="ghcr.io/real2simhub/r2s-python:1.0",
        argv=("python", "probe.py", str(hidden.resolve())),
        workdir="/r2s/program",
        mounts=(
            Mount(program, "/r2s/program"),
            Mount(tmp_path / "out", "/r2s/output", read_only=False),
        ),
        env={"R2S_OUTPUT_DIR": "/r2s/output"},
        limits=Limits(timeout_s=120, cpus=1, memory_mb=512),
    )
    (tmp_path / "out").mkdir()
    result = docker_backend.run(spec, tmp_path / "log.txt")
    assert result.exit_code == 0, (tmp_path / "log.txt").read_text()
    report = json.loads((tmp_path / "out" / "probe.json").read_text())
    import os

    assert report == {
        "network": "unreachable",
        "program": "read-only",
        "hidden": False,
        "cwd": "/r2s/program",
        "uid": os.getuid(),
    }


@pytest.mark.docker
def test_docker_twin_runs_and_timeouts(tmp_path: Path, docker_backend) -> None:
    twin = make_twin(tmp_path, GOOD)
    result = run_twin(twin, tmp_path / "ok", backend=docker_backend, seed=1)
    assert result.succeeded, (tmp_path / "ok" / "log.txt").read_text()
    assert result.record.image.digest and result.record.backend["backend"] == "docker"
    slow = make_twin(tmp_path, "import time\ntime.sleep(60)", name="slow")
    started = time.monotonic()
    timed = run_twin(
        slow, tmp_path / "slow-run", backend=docker_backend, limits=RunLimits(timeout_s=3)
    )
    assert timed.record.status is RunStatus.TIMED_OUT and time.monotonic() - started < 40


@pytest.mark.docker
def test_docker_missing_image_has_a_fix(docker_backend, monkeypatch: pytest.MonkeyPatch) -> None:
    from real2simlab.envs import image_reference
    from real2simlab.errors import SetupError

    # A registry prefix with nothing built under it, whatever this machine has built.
    monkeypatch.setenv("R2S_IMAGE_PREFIX", "r2s-test-never-built")
    with pytest.raises(SetupError, match="r2s env build r2s-agent-gemini-cli"):
        docker_backend.resolve_image(image_reference("r2s-agent-gemini-cli"))
    with pytest.raises(SetupError, match="docker pull example/unknown:9"):
        docker_backend.resolve_image("example/unknown:9")


@pytest.mark.docker
def test_docker_tells_a_program_exiting_125_from_a_container_that_did_not_start(
    tmp_path: Path, docker_backend
) -> None:
    """Docker exits 125 when it cannot start a container, and when the program exits 125."""
    program = make_twin(tmp_path, "print('ran', flush=True)\nraise SystemExit(125)")
    run = run_twin(program, tmp_path / "run", backend=docker_backend)
    assert run.record.status is RunStatus.FAILED and run.record.exit_code == 125
    assert "ran" in (tmp_path / "run" / "log.txt").read_text()
    spec = SandboxSpec(
        image="ghcr.io/real2simhub/r2s-python:1.0",
        argv=("true",),
        workdir="/",
        limits=Limits(timeout_s=60, memory_mb=1),  # below Docker's minimum: never starts
    )
    result = docker_backend.run(spec, tmp_path / "log.txt")
    assert result.exit_code is None and "could not start" in str(result.infrastructure_error)


def test_docker_exit_125_is_an_infrastructure_error_only_if_nothing_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    from real2simlab.sandbox import docker

    backend = docker.DockerBackend(docker="docker")
    backend._cpus_known = True
    calls: list[list[str]] = []
    started = {"value": "2026-11-19T10:00:00.5Z"}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[1] == "inspect":
            code = 1 if started["value"] is None else 0
            return subprocess.CompletedProcess(command, code, started["value"] or "", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(docker.subprocess, "run", fake_run)

    def fake_run_logged(command: list[str], log: Path, *a: object, **k: object) -> tuple:
        log.write_text("docker: Error response from daemon\n")
        return 125, False, 0.1

    monkeypatch.setattr(docker, "run_logged", fake_run_logged)
    spec = SandboxSpec(image="img", argv=("x",), workdir="/")
    assert backend.run(spec, tmp_path / "log").exit_code == 125  # the program's own exit
    assert calls[-1][:3] == ["docker", "rm", "-f"]  # the container is always removed
    for never in ("0001-01-01T00:00:00Z", None):  # created but not started; not created
        started["value"] = never
        result = backend.run(spec, tmp_path / "log")
        assert result.exit_code is None and result.infrastructure_error is not None


def test_a_docker_client_killed_from_outside_is_not_the_programs_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Docker client killed by a signal (a negative code; nothing in its container can
    signal it) is not the program's result: an infrastructure error, and its container is
    removed. The program's own death by a signal is the container's exit status, 128 + N."""
    import subprocess

    from real2simlab.sandbox import docker

    backend = docker.DockerBackend(docker="docker")
    backend._cpus_known = True
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(docker.subprocess, "run", fake_run)
    codes = [-9, 128 + 9]

    def fake_run_logged(command: list[str], log: Path, *a: object, **k: object) -> tuple:
        log.write_text("")
        return codes.pop(0), False, 0.1

    monkeypatch.setattr(docker, "run_logged", fake_run_logged)
    spec = SandboxSpec(image="img", argv=("x",), workdir="/")
    killed = backend.run(spec, tmp_path / "log")
    assert killed.exit_code is None
    assert "killed by SIGKILL from outside the sandbox" in (killed.infrastructure_error or "")
    assert calls[-1][:3] == ["docker", "rm", "-f"]  # it does not keep running
    inside = backend.run(spec, tmp_path / "log")
    assert (inside.exit_code, inside.infrastructure_error) == (137, None)


def test_docker_command_line(tmp_path: Path) -> None:
    from real2simlab.sandbox.docker import DockerBackend

    backend = DockerBackend(docker="docker")
    spec = SandboxSpec(
        image="img",
        argv=("python", "x.py"),
        workdir="/w",
        mounts=(Mount(tmp_path, "/w"),),
        env={"A": "1"},
        secret_env={"KEY": "s3cret"},
        limits=Limits(timeout_s=1, cpus=2, memory_mb=100, gpus="0"),
    )
    command = backend.command(spec, "name")
    text = " ".join(command)
    assert "--network none" in text and "--read-only" in text and "--cap-drop ALL" in text
    assert "s3cret" not in text and "--env KEY" in text
    assert '"device=0"' in text and "readonly" in text and "--cpus 2" in text
    assert "--rm" not in command  # removed after it ends, once Docker said whether it started
    backend.max_cpus = 1.0  # a smaller machine: the limit is lowered to what it has
    assert "--cpus 1" in " ".join(backend.command(spec, "name"))
    with pytest.raises(Exception, match="egress proxy"):
        backend.command(
            SandboxSpec(
                image="i",
                argv=("x",),
                workdir="/",
                network=NetworkPolicy(mode=NetworkMode.ALLOWLIST),
            ),
            "n",
        )


def test_docker_keeps_no_log_of_its_own_of_a_container(tmp_path: Path) -> None:
    """What a container prints reaches the Lab through the attached client, which keeps a
    bounded log; Docker's default log driver also kept all of it, without a limit, on the
    daemon's disk until the container was removed (A3-05). The containers the Lab runs
    programs in (sandboxes and their sessions, official scorers) turn that log off. These are
    the command lines; that the attached output still reaches the log is for a Docker daemon
    to show (the docker tests)."""
    from real2simlab.sandbox.docker import DockerBackend
    from real2simlab.sandbox.process import run_container

    def log_driver(command: list[str], image: str) -> str:
        option = command.index("--log-driver")
        assert option < command.index(image), command  # an option of docker run, not of the program
        return command[option + 1]

    spec = SandboxSpec(image="img", argv=("python", "x.py"), workdir="/w")
    assert log_driver(DockerBackend(docker="docker").command(spec, "name"), "img") == "none"
    calls = tmp_path / "calls.jsonl"
    docker = tmp_path / "docker"  # records what it is asked, and prints as a container would
    docker.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"with open({str(calls)!r}, 'a') as calls:\n"
        "    calls.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "print('scored')\n"
    )
    docker.chmod(0o755)
    code, timed_out, _ = run_container(str(docker), ["img", "score"], tmp_path / "log.txt", 60)
    assert code == 0 and not timed_out and "scored" in (tmp_path / "log.txt").read_text()
    run = next(call for call in map(json.loads, calls.read_text().splitlines()) if call[0] == "run")
    assert log_driver(run, "img") == "none"


# --- egress proxy -------------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_egress_proxy_allows_and_logs(tmp_path: Path) -> None:
    import asyncio

    from real2simlab.sandbox.egress import domains
    from real2simlab.sandbox.egress_proxy import Policy, serve

    upstream = socket.socket()
    upstream.bind(("127.0.0.1", 0))
    upstream.listen()
    upstream_port = upstream.getsockname()[1]

    def echo() -> None:
        while True:
            connection, _ = upstream.accept()
            connection.sendall(b"hello")
            connection.close()

    threading.Thread(target=echo, daemon=True).start()
    port = _free_port()
    log = tmp_path / "egress.jsonl"
    loop = asyncio.new_event_loop()
    server = loop.create_task(
        serve(port, Policy(["localhost", "*.example.org"], False), str(log), "127.0.0.1")
    )

    def run_server() -> None:
        with contextlib.suppress(asyncio.CancelledError):
            loop.run_until_complete(server)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)

    def request(host: str) -> bytes:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            client.sendall(
                f"CONNECT {host}:{upstream_port} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode()
            )
            data = b""
            while chunk := client.recv(4096):
                data += chunk
            return data

    allowed = request("localhost")
    assert b"200 Connection Established" in allowed and allowed.endswith(b"hello")
    assert b"403" in request("blocked.test")
    time.sleep(0.2)
    counts = domains(log)
    assert counts["blocked.test"]["denied"] == 1 and counts["localhost"]["allowed"] >= 1
    assert Policy(["*.example.org"], False).allowed("api.example.org")
    assert not Policy(["*.example.org"], False).allowed("example.org.evil.com")
    assert Policy([], True).allowed("anything")
    loop.call_soon_threadsafe(server.cancel)
    thread.join(timeout=5)
    assert not thread.is_alive()


REPLACING = (
    """
import os, shutil
out = os.environ["R2S_OUTPUT_DIR"]
assert out.endswith("/world"), out
shutil.rmtree(out)                       # "replace anything already there", as 4DCodeBench asks
os.makedirs(out)
parent = os.path.dirname(out)
open(os.path.join(parent, "junk.txt"), "w").write("not kept")
try:
    open("x.txt", "w").write("x")
    print("SOLUTION-WRITABLE")
except OSError:
    print("solution is read-only")
"""
    + GOOD
)


def replacing_twin(tmp_path: Path, *, probe_solution: bool = True):
    """A build.sh-style twin that replaces its output directory; optionally it also tries to
    write next to itself (the local backend does not isolate, so it would succeed there)."""
    from real2simlab.twin import Runtime

    program = tmp_path / "replacing-src"
    program.mkdir()
    probe = 'open("x.txt", "w").write("x")'
    source = REPLACING if probe_solution else REPLACING.replace(probe, "raise OSError")
    (program / "run.py").write_text(source)
    (program / "r2s_twin.py").write_text(sdk_path().read_text())
    manifest = TwinManifest(
        twin_id="test/replacing",
        entrypoint=("python", "./run.py"),
        environment={"image": "r2s-python"},
        output_format="tracks",
        runtime=Runtime(
            program_path="/workspace/solution",
            output_path="/workspace/world",
            replaceable_output=True,
        ),
        origin={"method": "test", "task_id": "t", "created_at": datetime.now(UTC)},
    )
    return save_twin(manifest, program, tmp_path / "replacing")


def check_replaced_output(run_dir: Path) -> None:
    record = read_record(run_dir)
    assert record.status is RunStatus.SUCCEEDED, record.problems
    assert (run_dir / "output" / "tracks.npz").is_file()
    assert not (run_dir / "scratch").exists() and not (run_dir / "junk.txt").exists()
    log = (run_dir / "log.txt").read_text()
    assert "SOLUTION-WRITABLE" not in log


def test_replaceable_output_local(tmp_path: Path, local) -> None:
    twin = replacing_twin(tmp_path, probe_solution=False)
    run_twin(twin, tmp_path / "run", backend=local)
    record = read_record(tmp_path / "run")
    assert record.status is RunStatus.SUCCEEDED, record.problems
    assert (tmp_path / "run" / "output" / "tracks.npz").is_file()
    assert not (tmp_path / "run" / "scratch").exists()


def test_replaceable_output_needs_a_parent() -> None:
    from pydantic import ValidationError

    from real2simlab.twin import Runtime

    with pytest.raises(ValidationError):
        Runtime(program_path="/r2s/program", output_path="/world", replaceable_output=True)


@pytest.mark.docker
def test_replaceable_output_docker(tmp_path: Path, docker_backend) -> None:
    twin = replacing_twin(tmp_path)
    run_twin(twin, tmp_path / "run", backend=docker_backend)
    check_replaced_output(tmp_path / "run")


HOSTILE = {
    "readonly_world": "os.chmod(out, 0o555)",
    "deep_tree": ("os.chdir(parent)\nfor _ in range(1500):\n    os.mkdir('a'); os.chdir('a')\n"),
    "unreadable_dir": "os.mkdir(os.path.join(parent, 'x')); os.chmod(os.path.join(parent, 'x'), 0)",
    "symlinked_world": "shutil.rmtree(out); os.symlink('/etc', out)",
}


def hostile_twin(tmp_path: Path, action: str, *, replaceable: bool = True):
    """A twin that writes valid output, then leaves something hostile behind."""
    from real2simlab.twin import Runtime

    program = tmp_path / f"{action}-src"
    program.mkdir()
    source = GOOD + (
        "\nimport os, shutil\n"
        "out = os.environ['R2S_OUTPUT_DIR']\n"
        "parent = os.path.dirname(out) if "
        + str(replaceable)
        + " else out\n"
        + HOSTILE[action]
        + "\n"
    )
    (program / "run.py").write_text(source)
    (program / "r2s_twin.py").write_text(sdk_path().read_text())
    runtime = (
        Runtime(program_path="/workspace/solution", output_path="/workspace/world",
                replaceable_output=True)
        if replaceable else Runtime()
    )  # fmt: skip
    manifest = TwinManifest(
        twin_id=f"test/{action}",
        entrypoint=("python", "./run.py"),
        environment={"image": "r2s-python"},
        output_format="tracks",
        runtime=runtime,
        origin={"method": "test", "task_id": "t", "created_at": datetime.now(UTC)},
    )
    return save_twin(manifest, program, tmp_path / action)


def check_hostile(tmp_path: Path, backend, action: str, replaceable: bool) -> None:
    from real2simlab.orchestrate.layout import fresh

    twin = hostile_twin(tmp_path, action, replaceable=replaceable)
    run_dir = tmp_path / f"run-{action}"
    run_twin(twin, run_dir, backend=backend)
    record = read_record(run_dir)  # a record is always written
    assert not (run_dir / "scratch").exists()
    if action == "symlinked_world" and replaceable:
        assert record.status is RunStatus.INVALID_OUTPUT
        assert not (run_dir / "output").is_symlink()
    elif replaceable:
        assert record.status is RunStatus.SUCCEEDED, record.problems
    fresh(run_dir)  # resuming can always start over
    assert not run_dir.exists()


@pytest.mark.parametrize("replaceable", [True, False])
@pytest.mark.parametrize("action", sorted(HOSTILE))
def test_hostile_leftovers_local(tmp_path: Path, local, action: str, replaceable: bool) -> None:
    if action == "symlinked_world" and not replaceable:
        pytest.skip("only a replaceable output directory can be replaced")
    check_hostile(tmp_path, local, action, replaceable)


@pytest.mark.docker
@pytest.mark.parametrize("action", sorted(HOSTILE))
def test_hostile_leftovers_docker(tmp_path: Path, docker_backend, action: str) -> None:
    check_hostile(tmp_path, docker_backend, action, replaceable=True)


def test_local_backend_keeps_a_host_interpreter_under_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host interpreter may live under a path the sandbox also uses (CI: /tmp/clean)."""
    import real2simlab.sandbox.local as local_module

    seen: list[list[str]] = []
    monkeypatch.setattr(
        local_module,
        "run_logged",
        lambda argv, *args, **kwargs: (seen.append(list(argv)), (0, False, 0.0))[1],
    )
    (tmp_path / "prog").mkdir()
    interpreter = tmp_path / "interpreter" / "bin" / "python"  # under /tmp, the sandbox's too
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    spec = SandboxSpec(
        image="i", argv=("python", "./run.py"), workdir="/r2s/program",
        mounts=(Mount(tmp_path / "prog", "/r2s/program"),),
    )  # fmt: skip
    local_module.LocalBackend(python=str(interpreter), quiet=True).run(spec, tmp_path / "log.txt")
    command = seen[0]  # on Linux, the program's command follows the supervisor's "--"
    program = command[command.index("--") + 1 :] if "--" in command else command
    assert program[0] == str(interpreter), seen


@pytest.mark.docker
def test_docker_tmpfs_is_writable_even_when_the_image_has_the_directory(
    tmp_path: Path, docker_backend
) -> None:
    """runc gives a tmpfs the mode of the image's directory under it: an agent home that an
    image created as root-only 0755 would not be writable for the container's user."""
    import subprocess
    import uuid

    base = "ghcr.io/real2simhub/r2s-python:1.0"
    tag = f"r2s-test-home:{uuid.uuid4().hex[:12]}"
    context = tmp_path / "image"
    context.mkdir()
    (context / "Dockerfile").write_text(
        f"FROM {base}\nUSER root\nRUN mkdir -p /r2s/home && chmod 755 /r2s/home\n"
    )
    subprocess.run(["docker", "build", "-q", "-t", tag, str(context)], check=True,
                   capture_output=True)  # fmt: skip
    try:
        spec = SandboxSpec(
            image=tag,
            argv=("sh", "-c", "mkdir /r2s/home/.codex && touch /r2s/home/.codex/auth.json"),
            workdir="/",
            tmpfs=("/tmp", "/r2s/home"),
            limits=Limits(timeout_s=60, cpus=1, memory_mb=256),
        )
        result = docker_backend.run(spec, tmp_path / "log.txt")
        assert result.exit_code == 0, (tmp_path / "log.txt").read_text()
    finally:
        subprocess.run(["docker", "image", "rm", "-f", tag], capture_output=True)
