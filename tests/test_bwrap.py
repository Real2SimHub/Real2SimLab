"""The bwrap backend and image unpacking."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from real2simlab.envs import rootfs as rootfs_module
from real2simlab.envs.rootfs import find_rootfs, resolve_in_root, unpack_tarball
from real2simlab.sandbox.base import Limits, Mount, NetworkMode, NetworkPolicy, SandboxSpec

from . import conftest

IMAGE = "ghcr.io/real2simhub/r2s-python:1.0"


def layer(entries: list[tuple[str, Any]]) -> bytes:
    """A layer tarball; an entry is (name, bytes | 'dir' | ('sym', target) | ('link', target))."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in entries:
            info = tarfile.TarInfo(name)
            if content == "dir":
                info.type, info.mode = tarfile.DIRTYPE, 0o755
                tar.addfile(info)
            elif isinstance(content, tuple):
                info.type = tarfile.SYMTYPE if content[0] == "sym" else tarfile.LNKTYPE
                info.linkname = content[1]
                tar.addfile(info)
            elif content == "dev":
                info.type = tarfile.CHRTYPE
                tar.addfile(info)
            else:
                info.size, info.mode = len(content), 0o4755
                tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def saved_image(
    path: Path, layers: list[bytes], env: list[str], tags: list[str] | None = None
) -> Path:
    """An image as ``docker save`` writes it; ``tags`` are the names its ``manifest.json`` gives
    it (``RepoTags``), none when it was saved by image id. Its configuration lists the digests
    of its layers, which unpacking checks."""
    import hashlib

    config = {
        "config": {"Env": env, "WorkingDir": "/app"},
        "rootfs": {"diff_ids": ["sha256:" + hashlib.sha256(data).hexdigest() for data in layers]},
    }
    with tarfile.open(path, "w") as tar:

        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        add("config.json", json.dumps(config).encode())
        names = []
        for index, data in enumerate(layers):
            names.append(f"layer{index}.tar")
            add(names[-1], data)
        entry: dict[str, Any] = {"Config": "config.json", "Layers": names}
        if tags is not None:
            entry["RepoTags"] = tags
        add("manifest.json", json.dumps([entry]).encode())
    return path


def test_unpack_applies_layers_and_whiteouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    base = layer([
        ("usr", "dir"), ("usr/lib", "dir"), ("lib", ("sym", "usr/lib")),
        ("usr/lib/a.so", b"a"), ("etc", "dir"), ("etc/keep", b"k"), ("etc/gone", b"g"),
        ("opaque", "dir"), ("opaque/old", b"o"), ("dev/null", "dev"),
        ("escape", ("sym", "/etc")),
    ])  # fmt: skip
    top = layer([
        ("etc/.wh.gone", b""), ("opaque/.wh..wh..opq", b""), ("opaque/new", b"n"),
        ("lib/b.so", b"b"), ("usr/lib/hard", ("link", "usr/lib/a.so")), ("escape/inside", b"i"),
    ])  # fmt: skip
    image = saved_image(tmp_path / "image.tar", [base, top], ["PATH=/usr/bin", "A=1"])
    unpacked = unpack_tarball(image, "test/image:1")
    root = unpacked.rootfs
    assert (root / "usr/lib/a.so").read_bytes() == b"a" and (
        root / "usr/lib/b.so"
    ).read_bytes() == b"b"
    assert (root / "usr/lib/hard").read_bytes() == b"a"
    assert not (root / "etc/gone").exists() and (root / "etc/keep").exists()
    assert sorted(p.name for p in (root / "opaque").iterdir()) == ["new"]
    assert (root / "etc/inside").read_bytes() == b"i"  # through the absolute link, inside the root
    assert not Path("/etc/inside").exists()
    assert not (root / "usr/lib/a.so").stat().st_mode & 0o4000
    assert unpacked.env == {"PATH": "/usr/bin", "A": "1"} and unpacked.workdir == "/app"
    assert json.loads((unpacked.root / "image.json").read_text())["skipped_special_files"] == 1
    assert find_rootfs("test/image:1") == unpacked and find_rootfs("other") is None
    assert (root / "r2s").is_dir() and (root / "workspace").is_dir()


def test_env_unpack_force_repairs_a_damaged_rootfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unpacking again reuses what is there (and says so); ``--force`` extracts it again."""
    from real2simlab.cli.main import main

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    base = layer([("usr", "dir"), ("usr/bin", "dir"), ("usr/bin/python", b"py")])
    image = saved_image(tmp_path / "image.tar", [base], ["PATH=/usr/bin"])
    argv = ["env", "unpack", "--from-tar", str(image), "test/repair:1"]
    assert main(argv) == 0
    unpacked = find_rootfs("test/repair:1")
    assert unpacked is not None
    (unpacked.rootfs / "usr/bin/python").unlink()  # e.g. a cleaner removed files
    capsys.readouterr()
    assert main(argv) == 0
    assert "was not extracted again (--force" in capsys.readouterr().out
    assert main([*argv, "--force"]) == 0
    assert capsys.readouterr().out.startswith("unpacked test/repair:1")
    repaired = find_rootfs("test/repair:1")
    assert repaired is not None and (repaired.rootfs / "usr/bin/python").read_bytes() == b"py"


def test_resolve_in_root_never_escapes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    os.symlink("/", root / "a" / "up")
    os.symlink("../../../../..", root / "a" / "dots")
    assert resolve_in_root(root, PurePosixPath("a/up/etc/passwd")) == root / "etc" / "passwd"
    assert resolve_in_root(root, PurePosixPath("a/dots/etc")) == root / "etc"
    os.symlink("loop", root / "loop")
    with pytest.raises(Exception, match="too many"):
        resolve_in_root(root, PurePosixPath("loop/x"))


def test_unpack_requires_a_saved_image(tmp_path: Path) -> None:
    from real2simlab.errors import SetupError

    with pytest.raises(SetupError):
        unpack_tarball(tmp_path / "missing.tar", "x")
    with tarfile.open(tmp_path / "bad.tar", "w"):
        pass
    with pytest.raises(SetupError, match=r"no manifest\.json"):
        unpack_tarball(tmp_path / "bad.tar", "x")


def test_bwrap_command_line(tmp_path: Path) -> None:
    from real2simlab.envs.rootfs import Rootfs
    from real2simlab.errors import SetupError
    from real2simlab.sandbox.bwrap import BwrapBackend

    unpacked = Rootfs("img", "sha256:x", tmp_path, {"PATH": "/usr/bin"}, "/")
    backend = BwrapBackend(bwrap="bwrap", cgroup_controllers=frozenset())
    spec = SandboxSpec(
        image="img",
        argv=("python", "x.py"),
        workdir="/r2s/program",
        mounts=(Mount(tmp_path, "/r2s/program"), Mount(tmp_path, "/r2s/output", read_only=False)),
        env={"A": "1"},
        secret_env={"KEY": "s3cret"},
        limits=Limits(timeout_s=1, cpus=1),
    )
    command = backend.command(spec, unpacked)
    text = " ".join(command)
    assert "--unshare-all" in text and "--share-net" not in text and "s3cret" not in text
    assert "--tmpfs /r2s" in text and f"--ro-bind {tmp_path}/rootfs /" in text
    assert backend.environment(spec, unpacked)["KEY"] == "s3cret"
    assert backend.environment(spec, unpacked)["CUDA_VISIBLE_DEVICES"] == ""
    open_spec = SandboxSpec(
        image="img", argv=("x",), workdir="/", network=NetworkPolicy(mode=NetworkMode.OPEN)
    )
    assert "--share-net" in backend.command(open_spec, unpacked)
    with pytest.raises(SetupError, match="allowlist"):
        backend.command(
            SandboxSpec(
                image="img",
                argv=("x",),
                workdir="/",
                network=NetworkPolicy(mode=NetworkMode.ALLOWLIST),
            ),
            unpacked,
        )


FAKE_BWRAP = r"""#!/bin/sh
# A bwrap that records its arguments, then fails as on a machine that does not let it mount a
# new /proc (an unprivileged container), or, with R2S_FAKE_NO_USERNS, makes no namespaces.
printf '%s\n' "$*" >> "$R2S_FAKE_CALLS"
if [ -n "$R2S_FAKE_NO_USERNS" ]; then
  echo "bwrap: setting up uid map: Permission denied" >&2; exit 1
fi
case " $* " in
  *" --proc "*) echo "bwrap: Can't mount proc on /newroot/proc: Operation not permitted" >&2; exit 1 ;;
esac
"""


def _executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o755)
    return path


def test_readiness_starts_the_sandbox_every_run_gets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`r2s doctor` and the check every command makes before it starts said bwrap was ready
    where each run then failed: the check left out the new /proc every sandbox mounts. It
    starts the sandbox `command` makes and says which part this machine forbids, with the fix
    for that part (unpacking an image again fixes neither)."""
    from real2simlab.envs.rootfs import Rootfs
    from real2simlab.sandbox.bwrap import APPARMOR_FIX, BwrapBackend

    monkeypatch.setenv("R2S_FAKE_CALLS", str(tmp_path / "calls"))
    monkeypatch.delenv("R2S_FAKE_NO_USERNS", raising=False)
    fake = _executable(tmp_path / "bwrap", FAKE_BWRAP)
    backend = BwrapBackend(bwrap=str(fake), cgroup_controllers=frozenset())
    problems = backend.problems()
    assert len(problems) == 1, problems
    (problem,) = problems
    assert "/proc" in problem.message and "Can't mount proc" in problem.message, problem
    assert problem.fix and problem.fix != APPARMOR_FIX and "env unpack" not in problem.fix
    # What it started is a run's sandbox: `command` for a plain program, with the host's / as
    # its root.
    rootfs = Rootfs("img", "sha256:x", tmp_path, {}, "/")
    run = backend.command(SandboxSpec(image="img", argv=("true",), workdir="/"), rootfs)
    started = (tmp_path / "calls").read_text().splitlines()[0].split()
    assert started == ["/" if arg == str(rootfs.rootfs) else arg for arg in run[1:]]
    # No user namespaces at all: they are what fails, with their own fix.
    monkeypatch.setenv("R2S_FAKE_NO_USERNS", "1")
    (problem,) = backend.problems()
    assert problem.message.startswith("bwrap cannot create namespaces"), problem
    assert problem.fix == APPARMOR_FIX
    # A bwrap that cannot be run is a problem with its fix, not a crash of `r2s doctor`.
    monkeypatch.setenv("R2S_BWRAP", str(tmp_path / "not-there" / "bwrap"))
    (problem,) = BwrapBackend(cgroup_controllers=frozenset()).problems()
    assert "R2S_BWRAP" in (problem.fix or ""), problem


def test_the_demo_does_not_choose_a_bwrap_that_cannot_start_its_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the image unpacked, `r2s demo` chose bwrap where no sandbox of it starts, and its
    first report was all unscored: it falls back to local, and says why."""
    from real2simlab.cli.demo import choose_backend
    from real2simlab.envs.rootfs import Rootfs

    monkeypatch.setenv("R2S_FAKE_CALLS", str(tmp_path / "calls"))
    monkeypatch.delenv("R2S_FAKE_NO_USERNS", raising=False)
    monkeypatch.setenv("R2S_BWRAP", str(_executable(tmp_path / "bwrap", FAKE_BWRAP)))
    no_daemon = "#!/bin/sh\necho 'Cannot connect to the Docker daemon' >&2\nexit 1\n"
    monkeypatch.setenv("R2S_DOCKER", str(_executable(tmp_path / "docker", no_daemon)))
    unpacked = Rootfs(IMAGE, "sha256:x", tmp_path, {}, "/")
    monkeypatch.setattr(rootfs_module, "find_rootfs", lambda reference: unpacked)
    said: list[str] = []
    assert choose_backend(log=said.append) == "local"
    assert any("bwrap" in line and "/proc" in line for line in said), said


def test_the_demo_does_not_choose_a_damaged_unpacked_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unpacked image whose tree a cleaner removed is damaged (A9R1-06): the demo falls back
    to local and says why, as when bwrap cannot start its sandbox."""
    from real2simlab.cli.demo import choose_backend

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    no_daemon = "#!/bin/sh\necho 'Cannot connect to the Docker daemon' >&2\nexit 1\n"
    monkeypatch.setenv("R2S_DOCKER", str(_executable(tmp_path / "docker", no_daemon)))
    readiness_passes(monkeypatch)
    image = saved_image(tmp_path / "image.tar", [layer([("etc", "dir")])], ["A=1"], tags=[IMAGE])
    unpacked = unpack_tarball(image, IMAGE)
    assert choose_backend(log=lambda line: None) == "bwrap"
    shutil.rmtree(unpacked.rootfs)
    said: list[str] = []
    assert choose_backend(log=said.append) == "local"
    assert any(line.startswith("bwrap: the unpacked image") and "damaged" in line for line in said)


@pytest.fixture
def bwrap_backend() -> Any:
    unavailable = conftest.bwrap_unavailable()
    if unavailable is not None:
        pytest.skip(f"bwrap cannot start a sandbox here: {unavailable}")
    if find_rootfs(IMAGE) is None:
        pytest.skip("r2s env unpack r2s-python")
    from real2simlab.sandbox.bwrap import BwrapBackend

    return BwrapBackend()


def test_the_bwrap_tests_ask_bwrap_itself_whether_they_can_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whether the bwrap integration tests ran was decided by the backend they test (its own
    readiness check): a regression that broke every sandbox it makes skipped them, and the CI
    job stayed green. They run where bwrap itself (``R2S_BWRAP``, else ``PATH``) starts a sandbox
    with what every sandbox of the backend has, a new ``/proc`` included, whatever the backend
    says; where it cannot, bwrap's own message is the reason they skip."""
    from real2simlab.sandbox import bwrap as module
    from real2simlab.sandbox.base import ReadinessProblem

    calls = tmp_path / "calls"
    fake = _executable(tmp_path / "bwrap", f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{calls}'\n")
    monkeypatch.setenv("R2S_BWRAP", str(fake))
    broken = [ReadinessProblem("bwrap: Unknown option --a-regression", "see bwrap's message")]
    monkeypatch.setattr(module.BwrapBackend, "problems", lambda self: broken)
    assert conftest.bwrap_ready()  # bwrap starts its sandbox: the tests run, and fail
    started = calls.read_text().split() if calls.exists() else []
    for part in (
        ["--unshare-all"],
        ["--proc", "/proc"],
        ["--dev", "/dev"],
        ["--ro-bind", "/", "/"],
    ):
        assert any(started[i : i + len(part)] == part for i in range(len(started))), started
    _executable(
        fake, '#!/bin/sh\necho "bwrap: Can\'t mount proc on /newroot/proc: denied" >&2\nexit 1\n'
    )
    monkeypatch.setattr(module.BwrapBackend, "problems", lambda self: [])
    assert not conftest.bwrap_ready()  # whatever the backend says
    assert "Can't mount proc" in (conftest.bwrap_unavailable() or "")


@pytest.mark.bwrap
def test_bwrap_is_ready_wherever_its_sandbox_starts() -> None:
    """Where bwrap itself starts a sandbox like every sandbox of the backend, the backend's own
    readiness check (``r2s doctor``, and every command before it starts) agrees: a regression in
    the sandbox it starts is this test's failure, not the other bwrap tests' skip."""
    unavailable = conftest.bwrap_unavailable()
    if unavailable is not None:
        pytest.skip(f"bwrap cannot start a sandbox here: {unavailable}")
    from real2simlab.sandbox.bwrap import BwrapBackend

    assert BwrapBackend().problems() == []


PROBE = """
import json, os, pathlib, socket, sys
out = pathlib.Path(os.environ["R2S_OUTPUT_DIR"])
report = {}
try:
    socket.create_connection(("1.1.1.1", 53), timeout=3)
    report["network"] = "reachable"
except OSError:
    report["network"] = "unreachable"
try:
    pathlib.Path("written.txt").write_text("x")
    report["program"] = "writable"
except OSError:
    report["program"] = "read-only"
report["hidden"] = os.path.exists(sys.argv[1])
report["home"] = os.path.exists("/home")
import numpy, r2s_twin
report["python"] = sys.prefix
(out / "probe.json").write_text(json.dumps(report))
"""


@pytest.mark.bwrap
def test_bwrap_isolation(tmp_path: Path, bwrap_backend: Any, task_copy: Path) -> None:
    program = tmp_path / "probe"
    program.mkdir()
    (program / "probe.py").write_text(PROBE)
    (tmp_path / "out").mkdir()
    spec = SandboxSpec(
        image=IMAGE,
        argv=("python", "probe.py", str((task_copy / "hidden").resolve())),
        workdir="/r2s/program",
        mounts=(
            Mount(program, "/r2s/program"),
            Mount(tmp_path / "out", "/r2s/output", read_only=False),
        ),
        env={"R2S_OUTPUT_DIR": "/r2s/output"},
        limits=Limits(timeout_s=60, cpus=1),
    )
    result = bwrap_backend.run(spec, tmp_path / "log.txt")
    assert result.exit_code == 0, (tmp_path / "log.txt").read_text()
    report = json.loads((tmp_path / "out" / "probe.json").read_text())
    assert report["network"] == "unreachable" and report["program"] == "read-only"
    assert report["hidden"] is False and report["python"] == "/usr/local"


@pytest.mark.bwrap
@pytest.mark.slow
def test_pipeline_on_bwrap(synthetic_tasks: Path, tmp_path: Path, bwrap_backend: Any) -> None:
    from real2simlab.orchestrate import parse_experiment, run_experiment

    config = parse_experiment(
        {
            "name": "bwrap",
            "tasks": {
                "paths": [str(synthetic_tasks)],
                "include": ["synthetic/bouncing_ball/calibrated"],
            },
            "methods": [
                {"name": "oracle"},
                {"name": "fake_agent", "label": "physical", "mode": "physical"},
            ],
            "run": {"backend": "bwrap"},
        }
    )
    results = run_experiment(
        config, tmp_path / "run", backend=bwrap_backend, log=lambda line: None, report=False
    )
    methods = {m.method: m for m in results.methods}
    assert methods["oracle"].reproduction_skill.mean == pytest.approx(1.0)
    assert 0.5 < methods["physical"].reproduction_skill.mean < 1.0


GPU_PROBE = """
import ctypes, json, os, pathlib
cuda = ctypes.CDLL("libcuda.so.1")
assert cuda.cuInit(0) == 0
count = ctypes.c_int()
assert cuda.cuDeviceGetCount(ctypes.byref(count)) == 0
pathlib.Path(os.environ["R2S_OUTPUT_DIR"], "gpu.json").write_text(json.dumps({"devices": count.value}))
"""


@pytest.mark.bwrap
def test_bwrap_gpu(tmp_path: Path, bwrap_backend: Any) -> None:
    if bwrap_backend.gpu_problems():
        pytest.skip(bwrap_backend.gpu_problems()[0].message)
    program = tmp_path / "gpu"
    program.mkdir()
    (program / "gpu.py").write_text(GPU_PROBE)
    (tmp_path / "out").mkdir()
    spec = SandboxSpec(
        image=IMAGE,
        argv=("python", "gpu.py"),
        workdir="/r2s/program",
        mounts=(
            Mount(program, "/r2s/program"),
            Mount(tmp_path / "out", "/r2s/output", read_only=False),
        ),
        env={"R2S_OUTPUT_DIR": "/r2s/output"},
        limits=Limits(timeout_s=60, gpus="0"),
    )
    result = bwrap_backend.run(spec, tmp_path / "log.txt")
    assert result.exit_code == 0, (tmp_path / "log.txt").read_text()
    assert json.loads((tmp_path / "out" / "gpu.json").read_text())["devices"] >= 1


def test_module_constants() -> None:
    assert rootfs_module.OPAQUE == ".wh..wh..opq"


@pytest.mark.bwrap
def test_replaceable_output_bwrap(tmp_path: Path, bwrap_backend: Any) -> None:
    from real2simlab.runner import run_twin

    from .test_runner_sandbox import check_replaced_output, replacing_twin

    twin = replacing_twin(tmp_path)
    run_twin(twin, tmp_path / "run", backend=bwrap_backend)
    check_replaced_output(tmp_path / "run")


def _linux(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    """Build bwrap command lines as on Linux, on any machine (CI runs these on macOS too)."""
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)


def test_bwrap_limits_through_a_user_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With delegated memory and pids controllers, bwrap runs in a limited systemd user scope."""
    from real2simlab.envs.rootfs import Rootfs
    from real2simlab.sandbox import bwrap as module

    monkeypatch.setattr(module.shutil, "which", lambda name: f"/usr/bin/{name}")
    _linux(monkeypatch, module)
    unpacked = Rootfs("img", "sha256:x", tmp_path, {"PATH": "/usr/bin"}, "/")
    spec = SandboxSpec(
        image="img", argv=("x",), workdir="/", limits=Limits(timeout_s=1, memory_mb=512, pids=99)
    )
    monkeypatch.setattr(module, "user_scope_oom_continue", lambda: True)  # systemd >= 253
    limited = module.BwrapBackend(bwrap="bwrap", cgroup_controllers=frozenset({"memory", "pids"}))
    command = limited.command(spec, unpacked)
    assert command[:5] == ["/usr/bin/systemd-run", "--user", "--scope", "--quiet", "--collect"]
    assert "MemoryMax=512M" in command and "TasksMax=99" in command
    assert command.index("--") < command.index("bwrap")
    # Its program out of memory is killed in it; the scope (and bwrap) carries on.
    policy = command.index("OOMPolicy=continue")
    assert command[policy - 1] == "-p" and policy < command.index("--")
    assert limited.limits_enforced() == ["time", "cpus", "memory", "pids"]
    monkeypatch.setattr(module, "user_scope_oom_continue", lambda: False)  # older: not taken
    assert "OOMPolicy=continue" not in limited.command(spec, unpacked)
    plain = module.BwrapBackend(bwrap="bwrap", cgroup_controllers=frozenset())
    assert plain.command(spec, unpacked)[0] == "bwrap"
    assert plain.limits_enforced() == ["time", "cpus"]  # taskset is there (which, above)


def test_cpu_limit_is_recorded_only_where_taskset_applies_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without taskset, bwrap cannot limit CPUs: no affinity, a warning, not in the record."""
    from real2simlab.envs.rootfs import Rootfs
    from real2simlab.sandbox import bwrap as module

    unpacked = Rootfs("img", "sha256:x", tmp_path, {"PATH": "/usr/bin"}, "/")
    spec = SandboxSpec(image="img", argv=("x",), workdir="/", limits=Limits(timeout_s=1, cpus=1))
    backend = module.BwrapBackend(bwrap="bwrap", cgroup_controllers=frozenset())
    with_taskset = {"taskset": "/usr/bin/taskset", "bwrap": "/usr/bin/bwrap"}
    monkeypatch.setattr(module.shutil, "which", with_taskset.get)
    _linux(monkeypatch, module)
    assert "taskset" in backend.command(spec, unpacked) and "cpus" in backend.limits_enforced()
    monkeypatch.setattr(module.shutil, "which", {"bwrap": "/usr/bin/bwrap"}.get)
    with pytest.warns(module.CpuLimitWarning, match="taskset"):
        command = backend.command(spec, unpacked)
    assert "taskset" not in command and backend.limits_enforced() == ["time"]
    version = subprocess.CompletedProcess(["bwrap"], 0, "bubblewrap 0.11.0\n", "")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: version)
    assert backend.info()["limits_enforced"] == ["time"]  # what each run record copies


def test_user_scope_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from real2simlab.sandbox import bwrap as module

    module.user_scope_controllers.cache_clear()
    monkeypatch.setenv("R2S_BWRAP_CGROUP", "0")
    try:
        assert module.user_scope_controllers() == frozenset()
    finally:
        module.user_scope_controllers.cache_clear()


KILLED_FROM_OUTSIDE = """#!/bin/sh
# bwrap killed by a signal from outside its sandbox (an administrator's kill, the host's
# out-of-memory killer) while its program runs. The runtime works: the Lab's probe of it
# (python -I ...) succeeds.
case " $* " in *" --version "*) echo "bubblewrap 0.9.0"; exit 0 ;; esac
case " $* " in *" -I "*) exit 0 ;; esac
kill -KILL $$
"""
KILLED_INSIDE = """#!/bin/sh
# bwrap whose program died of SIGKILL in the sandbox (by its own doing, or its memory limit):
# bwrap reports 128 + 9.
case " $* " in *" --version "*) echo "bubblewrap 0.9.0"; exit 0 ;; esac
exit 137
"""
SCOPE = """#!/bin/sh
# systemd-run --user --scope [OPTION...] -- COMMAND: here, COMMAND in place of its scope.
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do shift; done
shift
exec "$@"
"""


def _bin_script(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / "bin" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return str(path)


def test_a_sandbox_killed_from_outside_is_not_its_programs_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bwrap killed by a signal ends with a negative code: its program, in another PID
    namespace, cannot signal it, so the run is an infrastructure error, not the program's
    failure. The program's own death by a signal is bwrap's exit status 128 + N: its result.
    So is a kill in a systemd scope, where the program running out of the scope's memory can
    end bwrap too: nothing there tells that from a kill from outside."""
    from real2simlab.sandbox import bwrap as module

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    unpack_tarball(_small_image(tmp_path), "test/killed:1")
    (tmp_path / "program").mkdir()
    spec = SandboxSpec(
        image="test/killed:1",
        argv=("python", "run.py"),
        workdir="/r2s/program",
        mounts=(Mount(tmp_path / "program", "/r2s/program"),),
        limits=Limits(timeout_s=30),
    )
    bwrap = _bin_script(tmp_path, "killed", KILLED_FROM_OUTSIDE)
    killed = module.BwrapBackend(bwrap, cgroup_controllers=frozenset()).run(spec, tmp_path / "a")
    assert killed.exit_code is None
    assert "killed by SIGKILL from outside the sandbox" in (killed.infrastructure_error or "")
    inside = module.BwrapBackend(
        _bin_script(tmp_path, "inside", KILLED_INSIDE), cgroup_controllers=frozenset()
    ).run(spec, tmp_path / "b")
    assert (inside.exit_code, inside.infrastructure_error) == (137, None)
    monkeypatch.setattr(
        module.shutil, "which", {"systemd-run": _bin_script(tmp_path, "scope", SCOPE)}.get
    )
    # A systemd that takes OOMPolicy=continue (253 and later), not asked of the stand-in: the
    # probe's answer is kept for the whole process.
    monkeypatch.setattr(module, "user_scope_oom_continue", lambda: True)
    scoped = module.BwrapBackend(bwrap, cgroup_controllers=frozenset({"memory", "pids"}))
    unpacked = find_rootfs("test/killed:1")
    assert unpacked is not None and scoped.command(spec, unpacked)[0].endswith("scope")
    assert "OOMPolicy=continue" in scoped.command(spec, unpacked)  # and the kill still stands
    in_scope = scoped.run(spec, tmp_path / "c")
    assert (in_scope.exit_code, in_scope.infrastructure_error) == (-9, None)


def test_no_reward_when_the_sandbox_is_killed_from_outside(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``r2s train reward`` whose bwrap is killed from outside while the solution runs: no
    reward (``RewardUnavailable``), not 0 for a failed run; the run is an infrastructure
    error, so an evaluation redoes it instead of reusing a 0."""
    from real2simlab.sandbox.bwrap import BwrapBackend
    from real2simlab.train import score_solution
    from real2simlab.train.reward import RewardUnavailable

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    unpack_tarball(_small_image(tmp_path), IMAGE)
    bwrap = _bin_script(tmp_path, "killed", KILLED_FROM_OUTSIDE)
    backend = BwrapBackend(bwrap, cgroup_controllers=frozenset())
    solution = shutil.copytree(bouncing_task / "hidden" / "generator" / "program", tmp_path / "s")
    with pytest.raises(RewardUnavailable, match="killed by SIGKILL from outside the sandbox"):
        score_solution(bouncing_task, solution, backend=backend, work_dir=tmp_path / "work")
    record = json.loads((tmp_path / "work" / "runs" / "run" / "run_record.json").read_text())
    assert record["status"] == "infra_error" and record.get("exit_code") is None


RUNS_ON = """#!/bin/sh
# bwrap whose program runs past its time limit: the Lab stops bwrap itself (SIGTERM to its
# process group, then SIGKILL), and it dies of that signal.
exec sleep 60
"""


@pytest.mark.skipif(sys.platform != "linux", reason="bwrap and its stand-ins run on Linux")
def test_a_sandbox_stopped_at_its_time_limit_timed_out_and_was_not_killed_from_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At its time limit the Lab stops bwrap itself, so bwrap dies of a signal as it does when
    something outside the sandbox kills it: the run timed out (the program's result), and its
    log says so. A sandbox is taken for killed from outside by the code its run ends with,
    which is ``None`` after a time limit."""
    from real2simlab.sandbox import bwrap as module

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    unpack_tarball(_small_image(tmp_path), "test/slow:1")
    (tmp_path / "program").mkdir()
    spec = SandboxSpec(
        image="test/slow:1",
        argv=("python", "run.py"),
        workdir="/r2s/program",
        mounts=(Mount(tmp_path / "program", "/r2s/program"),),
        limits=Limits(timeout_s=1),
    )
    bwrap = _bin_script(tmp_path, "runs-on", RUNS_ON)
    result = module.BwrapBackend(bwrap, cgroup_controllers=frozenset()).run(spec, tmp_path / "log")
    assert (result.exit_code, result.timed_out, result.infrastructure_error) == (None, True, None)
    assert (tmp_path / "log").read_text().endswith("[r2s] killed after 1 s\n")


@pytest.mark.bwrap
def test_bwrap_memory_limit_is_enforced(tmp_path: Path, bwrap_backend: Any) -> None:
    if "memory" not in bwrap_backend.cgroup_controllers:
        pytest.skip("no systemd user scope with the memory controller here")
    hog = "b = bytearray(400 * 1024 * 1024); print('allocated')"
    spec = SandboxSpec(
        image=IMAGE,
        argv=("python", "-c", hog),
        workdir="/",
        limits=Limits(timeout_s=60, memory_mb=100),
    )
    result = bwrap_backend.run(spec, tmp_path / "log.txt")
    assert result.exit_code != 0 and "allocated" not in (tmp_path / "log.txt").read_text()
    # Killed inside the sandbox (bwrap reports 128 + 9), the program's own failure: never bwrap
    # itself, stopped with its scope (systemd >= 253 does that unless OOMPolicy=continue).
    assert result.exit_code is not None and result.exit_code > 0
    small = SandboxSpec(
        image=IMAGE,
        argv=("python", "-c", "b = bytearray(10 * 1024 * 1024); print('allocated')"),
        workdir="/",
        limits=Limits(timeout_s=60, memory_mb=100),
    )
    assert bwrap_backend.run(small, tmp_path / "small.txt").exit_code == 0


@pytest.mark.bwrap
def test_scope_variables_do_not_reach_the_program(tmp_path: Path, bwrap_backend: Any) -> None:
    show = "import os; print(sorted(k for k in os.environ if 'DBUS' in k or 'XDG_RUNTIME' in k))"
    spec = SandboxSpec(
        image=IMAGE, argv=("python", "-c", show), workdir="/", limits=Limits(timeout_s=60)
    )
    assert bwrap_backend.run(spec, tmp_path / "log.txt").exit_code == 0
    assert (tmp_path / "log.txt").read_text().strip().endswith("[]")


@pytest.mark.bwrap
@pytest.mark.parametrize(
    "action", ["readonly_world", "deep_tree", "unreadable_dir", "symlinked_world"]
)
def test_hostile_leftovers_bwrap(tmp_path: Path, bwrap_backend: Any, action: str) -> None:
    from .test_runner_sandbox import check_hostile

    check_hostile(tmp_path, bwrap_backend, action, replaceable=True)


def test_unpack_stages_in_the_cache_even_before_it_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first unpack, when the cache's rootfs directory does not exist yet, must not stage
    in the system's temporary directory: a rename to another file system fails (EXDEV)."""
    import errno

    cache = tmp_path / "cache"
    monkeypatch.setenv("R2S_CACHE_DIR", str(cache))
    other = tmp_path / "other-file-system"
    other.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(other))
    rename = Path.rename

    def one_file_system(self: Path, target: str | Path) -> Path:
        if not self.resolve().is_relative_to(cache.resolve()):
            raise OSError(errno.EXDEV, "Invalid cross-device link", str(self))
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", one_file_system)
    image = saved_image(tmp_path / "image.tar", [layer([("etc", "dir")])], ["A=1"])
    unpacked = unpack_tarball(image, "test/first:1")
    assert unpacked.root.is_relative_to(cache) and (unpacked.rootfs / "etc").is_dir()


@pytest.mark.parametrize("damage", ["null", "{", '{"image_id": 3}'])
def test_damaged_unpack_records_are_named_and_repaired(
    damage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A damaged ``image.json`` is named with the repair; unpacking again (``--force`` or not)
    extracts the image again instead of failing on the old record."""
    from real2simlab.cli.main import main
    from real2simlab.errors import SetupError

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    base = layer([("usr", "dir"), ("usr/bin", "dir"), ("usr/bin/python", b"py")])
    image = saved_image(tmp_path / "image.tar", [base], ["PATH=/usr/bin"])
    unpacked = unpack_tarball(image, "test/damaged:1")
    (unpacked.root / "image.json").write_text(damage)
    with pytest.raises(SetupError, match=r"image\.json\) is damaged") as error:
        find_rootfs("test/damaged:1")
    assert "--force" in str(error.value)
    for argv in (["--force"], []):
        (unpacked.root / "image.json").write_text(damage)
        assert main(["env", "unpack", "--from-tar", str(image), "test/damaged:1", *argv]) == 0
        repaired = find_rootfs("test/damaged:1")
        assert repaired is not None and (repaired.rootfs / "usr/bin/python").read_bytes() == b"py"
    assert "Traceback" not in capsys.readouterr().err


def test_concurrent_unpacks_keep_every_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unpacks running at once (one per cluster job) each keep their registration in the
    shared index; a damaged index is rebuilt from the images' own records."""
    import subprocess
    import sys

    cache = tmp_path / "cache"
    monkeypatch.setenv("R2S_CACHE_DIR", str(cache))
    image = saved_image(tmp_path / "image.tar", [layer([("etc", "dir")])], ["A=1"])
    code = (
        "import sys\nfrom pathlib import Path\nfrom real2simlab.envs.rootfs import unpack_tarball\n"
        "unpack_tarball(Path(sys.argv[1]), sys.argv[2])\n"
    )
    names = [f"test/many-{index}:1" for index in range(12)]
    processes = [subprocess.Popen([sys.executable, "-c", code, str(image), name]) for name in names]
    assert all(process.wait(120) == 0 for process in processes)
    assert all(find_rootfs(name) is not None for name in names)
    (cache / "rootfs" / "index.json").write_text("[")
    assert all(find_rootfs(name) is not None for name in names)


def test_an_image_unpacked_without_a_mount_point_gets_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An image unpacked before ``/app`` was a mount point (Blender scenes write there): the
    backend makes it, since bubblewrap cannot in a read-only root."""
    from real2simlab.envs.rootfs import MOUNT_ROOTS, mount_points_problem

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    image = saved_image(tmp_path / "image.tar", [layer([("etc", "dir")])], ["A=1"])
    unpacked = unpack_tarball(image, "test/old:1")
    assert all((unpacked.rootfs / root.lstrip("/")).is_dir() for root in MOUNT_ROOTS)
    (unpacked.rootfs / "app").rmdir()
    assert mount_points_problem(unpacked) is None
    assert (unpacked.rootfs / "app").is_dir()


def _archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


CONFIG = json.dumps({"rootfs": {"diff_ids": ["sha256:" + "1" * 64]}, "config": {}}).encode()


def _cut_layer(tmp_path: Path) -> dict[str, bytes]:
    import gzip

    data = gzip.compress(layer([("usr", "dir"), ("usr/big", os.urandom(50_000))]))
    return {"config.json": CONFIG, "layer.tar.gz": data[: len(data) // 2],
            "manifest.json": json.dumps([{"Config": "config.json", "Layers": ["layer.tar.gz"]}]).encode()}  # fmt: skip


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (lambda _: {"manifest.json": json.dumps([{"Config": "config.json", "RepoTags": ["qa:1"],
                    "Layers": ["layer/layer.tar"]}]).encode(), "config.json": CONFIG},
         "is incomplete: it has no layer/layer.tar"),  # the audit's archive
        (lambda _: {"manifest.json": b"[]", "config.json": CONFIG}, "names no image"),
        (lambda _: {"manifest.json": b"{", "config.json": CONFIG}, "manifest.json is damaged"),
        (lambda _: {"manifest.json": json.dumps([{"Config": "config.json", "Layers": []}]).encode(),
                    "config.json": b"[1]"}, "is not an object"),
        (lambda _: {"manifest.json": json.dumps([{"Config": 3, "Layers": []}]).encode()},
         "does not name a configuration"),
        (_cut_layer, "layer layer.tar.gz is damaged or incomplete"),
    ],
)  # fmt: skip
def test_an_incomplete_or_damaged_image_archive_is_a_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    files: Any,
    expected: str,
) -> None:
    """A saved image cut short (a copy or download that stopped) or damaged is named with what
    is missing and how to get a whole one, before anything is extracted, never a traceback."""
    from real2simlab.cli.main import main

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    archive = _archive(tmp_path / "image.tar", files(tmp_path))
    assert main(["env", "unpack", "qa-image:local", "--from-tar", str(archive)]) == 1
    err = capsys.readouterr().err
    assert expected in err and str(archive) in err and "Traceback" not in err, err
    assert "r2s env export" in err  # how to get a whole one
    leftovers = list((tmp_path / "cache" / "rootfs").glob("qa-image*"))
    assert not [path for path in leftovers if path.is_dir()], leftovers


def test_a_cut_image_archive_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import gzip

    from real2simlab.cli.main import main

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    base = layer([("usr", "dir"), ("usr/bin", "dir"), ("usr/bin/python3", os.urandom(80_000))])
    whole = saved_image(tmp_path / "image.tar", [base], ["PATH=/usr/bin"]).read_bytes()
    cut = tmp_path / "cut.tar.gz"
    cut.write_bytes(gzip.compress(whole)[: len(gzip.compress(whole)) * 2 // 3])
    assert main(["env", "unpack", "qa-image:local", "--from-tar", str(cut)]) == 1
    err = capsys.readouterr().err
    assert str(cut) in err and "Traceback" not in err, err


def _small_image(tmp_path: Path) -> Path:
    base = layer([("usr", "dir"), ("usr/bin", "dir"), ("usr/bin/python3", b"py")])
    return saved_image(tmp_path / "image.tar", [base], ["PATH=/usr/bin"])


def test_an_image_a_sandbox_uses_is_not_replaced_under_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``r2s env unpack --force`` while a sandbox runs on the image would make that program
    fail (its files vanish), a machine failure scored as the program's: it is refused."""
    from real2simlab.cli.main import main
    from real2simlab.envs.rootfs import in_use

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    argv = ["env", "unpack", "test/busy:1", "--from-tar", str(_small_image(tmp_path)), "--force"]
    assert main(argv) == 0
    unpacked = find_rootfs("test/busy:1")
    assert unpacked is not None
    capsys.readouterr()
    with in_use(unpacked):
        assert main(argv) == 1
        err = capsys.readouterr().err
        assert "in use by a running sandbox" in err and "Traceback" not in err
        assert (unpacked.rootfs / "usr/bin/python3").read_bytes() == b"py"  # untouched
    assert main(argv) == 0  # once it is no longer used


def _another_image(unpacked: Any, how: str) -> None:
    """Put another image where ``unpacked`` is, as a person or a tool that takes no lock does."""
    if how == "removed":
        unpacked.rootfs.rename(unpacked.root / "gone")
    elif how == "tree":  # another tree under the same record (a copy put in its place)
        old = unpacked.rootfs.rename(unpacked.root / "old")
        shutil.copytree(old, unpacked.rootfs, symlinks=True)
    else:  # another image's record
        meta = json.loads((unpacked.root / "image.json").read_text())
        meta["image_id"] = "sha256:" + "c" * 64
        (unpacked.root / "image.json").write_text(json.dumps(meta))


@pytest.mark.parametrize("change", ["record", "tree", "removed", "under the first run"])
def test_a_command_runs_only_on_the_unpacked_image_it_resolved(
    change: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command resolves an unpacked image once, and its run keys and records name that
    image's id. An image replaced, removed or given another record afterwards, by hand
    (between two seeds, or under a program and so before its retry), makes every later run of
    the command an infrastructure error that runs nothing, never a result filed under the id
    the command resolved (and reused for it once that image is back)."""
    from real2simlab.sandbox import bwrap as bwrap_module
    from real2simlab.sandbox.bwrap import BwrapBackend

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    unpacked = unpack_tarball(_small_image(tmp_path), "test/pinned:1")
    started: list[str] = []

    def program(*_: Any, **__: Any) -> tuple[int | None, bool, float]:
        started.append("program")
        if change == "under the first run":
            _another_image(unpacked, "record")
        return 1, False, 0.1  # the program's own failure

    monkeypatch.setattr(bwrap_module, "run_logged", program)
    (tmp_path / "program").mkdir()
    spec = SandboxSpec(
        image="test/pinned:1",
        argv=("python", "run.py"),
        workdir="/r2s/program",
        mounts=(Mount(tmp_path / "program", "/r2s/program"),),
        limits=Limits(timeout_s=10),
    )
    backend = BwrapBackend("/bin/true", cgroup_controllers=frozenset())
    resolved = backend.resolve_image("test/pinned:1")  # what a command does first
    first = backend.run(spec, tmp_path / "first.log")
    if change == "under the first run":
        assert first.exit_code is None and first.infrastructure_error
    else:
        assert (first.exit_code, first.infrastructure_error) == (1, None)  # the program's own
        _another_image(unpacked, change)
    again = backend.run(spec, tmp_path / "again.log")  # the next seed, or the retry
    assert started == ["program"], "a program ran on an image the command did not resolve"
    assert again.exit_code is None and not again.timed_out, again
    assert f"resolved it as {resolved.content_id}" in (again.infrastructure_error or ""), again
    assert backend.resolve_image("test/pinned:1") == resolved  # what its records name


@pytest.mark.skipif(sys.platform != "linux", reason="bwrap and its stand-ins run on Linux")
def test_an_agent_session_starts_only_on_the_unpacked_image_it_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent's build is keyed by the content id of the image it works in: a session on an
    unpacked image that is no longer the one the command resolved is an infrastructure error
    (the build is redone by the next command), not a build filed under the earlier id."""
    from real2simlab.errors import InfrastructureError
    from real2simlab.sandbox.bwrap import BwrapBackend

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    unpacked = unpack_tarball(_small_image(tmp_path), "test/agent:1")
    backend = BwrapBackend("/bin/true", cgroup_controllers=frozenset())
    resolved = backend.resolve_image("test/agent:1")
    (tmp_path / "workspace").mkdir()
    spec = SandboxSpec(
        image="test/agent:1",
        argv=(),
        workdir="/workspace",
        mounts=(Mount(tmp_path / "workspace", "/workspace", read_only=False),),
    )
    with backend.session(spec) as session:
        assert session.prefix[-2:] == ("bash", "-c")
    _another_image(unpacked, "record")
    with (
        pytest.raises(InfrastructureError, match=f"resolved it as {resolved.content_id}"),
        backend.session(spec),
    ):
        pytest.fail("a session started on another image")


def test_a_program_whose_image_was_replaced_meanwhile_has_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While a program runs, the image stays locked (a forced unpack is refused); an image
    replaced anyway (by hand) makes the run an infrastructure error, never the program's."""
    from real2simlab.cli.main import main
    from real2simlab.sandbox import bwrap as bwrap_module
    from real2simlab.sandbox.bwrap import BwrapBackend

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    image = _small_image(tmp_path)
    unpacked = unpack_tarball(image, "test/live:1")
    forced: list[int] = []

    def program(replace: bool) -> Any:
        def run(*_: Any, **__: Any) -> tuple[int | None, bool, float]:
            forced.append(main(["env", "unpack", "test/live:1", "--from-tar", str(image),
                                "--force"]))  # fmt: skip
            if replace:  # what an unpack that takes no lock (or `rm -rf`, `cp -r`) does
                old = unpacked.rootfs.rename(unpacked.root / "old")
                shutil.copytree(old, unpacked.rootfs, symlinks=True)
            return 1, False, 0.1  # the program failed: its files were gone

        return run

    (tmp_path / "program").mkdir()
    spec = SandboxSpec(
        image="test/live:1",
        argv=("python", "run.py"),
        workdir="/r2s/program",
        mounts=(Mount(tmp_path / "program", "/r2s/program"),),
        limits=Limits(timeout_s=10),
    )
    backend = BwrapBackend("/bin/true", cgroup_controllers=frozenset())
    monkeypatch.setattr(bwrap_module, "run_logged", program(replace=False))
    plain = backend.run(spec, tmp_path / "plain.log")
    assert plain.exit_code == 1 and plain.infrastructure_error is None  # the program's own
    monkeypatch.setattr(bwrap_module, "run_logged", program(replace=True))
    replaced = backend.run(spec, tmp_path / "replaced.log")
    assert replaced.exit_code is None
    assert "replaced or removed while the program ran" in (replaced.infrastructure_error or "")
    assert forced == [1, 1]  # refused while the program ran


NO_PROC = """#!/bin/sh
# bwrap on a host that allows no new procfs (a container): every sandbox fails as bwrap sets
# it up, before its program starts, and bwrap exits 1 with its own error.
case " $* " in *" --version "*) echo "bubblewrap 0.9.0"; exit 0 ;; esac
echo "bwrap: Can't mount proc on /newroot/proc: Operation not permitted" >&2
exit 1
"""


def readiness_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """On that host bwrap passes the Lab's readiness check (it mounts no procfs), so doctor
    says ready and the demo picks bwrap."""
    from real2simlab.sandbox.bwrap import BwrapBackend

    monkeypatch.setattr(BwrapBackend, "problems", lambda self: [])


PASS_THROUGH = """#!/bin/sh
# A bwrap that works, for what these tests need: it runs the command after "--" on the host.
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do shift; done
shift
exec "$@"
"""


def _fake_bwrap(tmp_path: Path, name: str, script: str) -> str:
    path = tmp_path / "fakebin" / name / "bwrap"
    path.parent.mkdir(parents=True)
    path.write_text(script)
    path.chmod(0o755)
    return str(path)


@pytest.mark.skipif(sys.platform != "linux", reason="bwrap and its stand-ins run on Linux")
def test_a_session_whose_sandbox_cannot_start_is_an_infrastructure_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session that bwrap cannot set up fails each of its commands before it runs: it says
    so when it starts, so no agent works (and is scored) in a sandbox that does not exist."""
    from real2simlab.errors import InfrastructureError
    from real2simlab.sandbox.bwrap import BwrapBackend

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    unpack_tarball(_small_image(tmp_path), "test/noproc:1")
    (tmp_path / "w").mkdir()
    spec = SandboxSpec(
        image="test/noproc:1",
        argv=(),
        workdir="/w",
        mounts=(Mount(tmp_path / "w", "/w", read_only=False),),
        limits=Limits(timeout_s=10),
    )
    broken = BwrapBackend(_fake_bwrap(tmp_path, "noproc", NO_PROC), cgroup_controllers=frozenset())
    with pytest.raises(InfrastructureError, match="Can't mount proc"), broken.session(spec):
        pass
    working = BwrapBackend(
        _fake_bwrap(tmp_path, "works", PASS_THROUGH), cgroup_controllers=frozenset()
    )
    with working.session(spec) as session:
        ran = subprocess.run(
            [*session.prefix, "echo hi"],
            env=session.env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    assert ran.stdout == "hi\n"


def test_an_agent_whose_bwrap_sandbox_cannot_start_is_not_a_score(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bwrap exits 1 with its own error before the agent's bootstrap starts, as an agent that
    fails does; a sandbox like it cannot start Python either: an infrastructure error."""
    from real2simlab.methods import get_method
    from real2simlab.methods.base import BuildContext
    from real2simlab.sandbox.bwrap import BwrapBackend
    from real2simlab.task import load_task
    from real2simlab.task.instruction import task_statement

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    unpack_tarball(_small_image(tmp_path), IMAGE)
    task = load_task(bouncing_task)
    backend = BwrapBackend(_fake_bwrap(tmp_path, "noproc", NO_PROC), cgroup_controllers=frozenset())
    context = BuildContext(
        task_id=task.id, task_hash=task.content_hash(), config=task.config,
        statement=task_statement(task), visible_dir=task.visible_dir,
        work_dir=tmp_path / "work", twin_dir=tmp_path / "twin", backend=backend,
    )  # fmt: skip
    outcome = get_method("fake_agent", mode="physical").build(context)
    assert outcome.twin is None and "Can't mount proc" in (outcome.infra_error or "")
    record = json.loads((tmp_path / "work" / "agent" / "agent_record.json").read_text())
    assert record["exit_status"] == "infra_error" and not record["delivered"]


@pytest.mark.slow
def test_the_demo_does_not_score_agents_whose_sandbox_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``r2s demo`` on bwrap where it passes its readiness check but no sandbox starts: its
    scripted agents are not scored 0 (nothing of them ran), like its baselines. Nothing could be
    scored at all, so it says so and exits with status 1."""
    from real2simlab.cli import demo
    from real2simlab.cli.main import main

    readiness_passes(monkeypatch)
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("R2S_BWRAP", _fake_bwrap(tmp_path, "noproc", NO_PROC))
    unpack_tarball(_small_image(tmp_path), IMAGE)
    monkeypatch.setattr(demo, "DEMO_TASKS", ("bouncing_ball",))
    assert main(["demo", "--backend", "bwrap", "--out", str(tmp_path / "demo")]) == 1
    assert "error: nothing could be scored" in capsys.readouterr().err
    rows = json.loads((tmp_path / "demo" / "results.json").read_text())["rows"]
    agents = [row for row in rows if row["method"] in ("scripted-agent", "broken-agent")]
    assert len(agents) == 2
    for row in agents:
        assert row["infra_error"] and row["reproduction_skill"] is None, row


SYSTEMD_RUN = """#!/bin/sh
# systemd-run --user --scope as systemd VERSION runs it, for what these tests need. Scopes take
# OOMPolicy= from systemd 253 on, and their default then (DefaultOOMPolicy=stop) stops the scope
# when the kernel kills one of its processes for lack of memory; older scopes always continue.
echo "$*" >> "CALLS"
policy=continue
[ VERSION -ge 253 ] && policy=stop
while [ "$#" -gt 0 ]; do
  case "$1" in
    -p) case "$2" in
          OOMPolicy=*) [ VERSION -ge 253 ] || { echo "Unknown assignment: $2" >&2; exit 1; }
                       policy=${2#OOMPolicy=} ;;
        esac
        shift 2 ;;
    --) shift; break ;;
    -*) shift ;;
    *) break ;;
  esac
done
case "$*" in *cgroup.controllers*) echo "cpu memory pids"; exit 0 ;; esac
SCOPE_OOM_POLICY=$policy exec "$@"
"""

OUT_OF_MEMORY = """#!/bin/sh
# bwrap, for what these tests need: it runs the program after "--" on the host, and reports its
# death by a signal as 128 + N. A scope that stops when the kernel kills one of its processes
# for lack of memory (SCOPE_OOM_POLICY=stop, from SYSTEMD_RUN) then stops bwrap too: SIGTERM.
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do shift; done
shift
"$@" &
wait $!
code=$?
if [ "$code" -eq 137 ] && [ "$SCOPE_OOM_POLICY" = stop ]; then kill -TERM $$; fi
exit $code
"""


@pytest.mark.skipif(sys.platform != "linux", reason="bwrap and its stand-ins run on Linux")
@pytest.mark.parametrize("systemd", [255, 252])
def test_a_program_out_of_its_memory_in_a_systemd_scope_is_its_own_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, systemd: int
) -> None:
    """bwrap's memory limit is a systemd user scope. A program that runs out of that memory is
    killed by the kernel inside the sandbox, which bwrap reports (128 + 9): the program's own
    result, never a sandbox killed from outside. A scope of systemd 253 or later would stop
    instead (its default OOMPolicy), killing bwrap itself, so the Lab has it continue where
    systemd takes the setting; an older scope does not take it and always continues.
    ``systemd-run`` is a stand-in that does what systemd does."""
    from real2simlab.agents.base import sandbox_failure
    from real2simlab.sandbox import bwrap as module

    calls = tmp_path / "systemd-run.log"
    stub = tmp_path / "bin" / "systemd-run"
    stub.parent.mkdir()
    stub.write_text(SYSTEMD_RUN.replace("VERSION", str(systemd)).replace("CALLS", str(calls)))
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.delenv("R2S_BWRAP_CGROUP", raising=False)
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    unpack_tarball(_small_image(tmp_path), IMAGE)
    spec = SandboxSpec(
        image=IMAGE,
        argv=("sh", "-c", "kill -KILL $$"),  # what the kernel does to a program out of memory
        workdir="/",
        limits=Limits(timeout_s=60, memory_mb=256),
        label="r2s-agent-test",
    )
    module.user_scope_controllers.cache_clear()  # probed once per process: this systemd is new
    module.user_scope_oom_continue.cache_clear()
    try:
        backend = module.BwrapBackend(_fake_bwrap(tmp_path, "oom", OUT_OF_MEMORY))
        assert backend.cgroup_controllers == {"cpu", "memory", "pids"}
        result = backend.run(spec, tmp_path / "sandbox.log")
    finally:
        module.user_scope_controllers.cache_clear()
        module.user_scope_oom_continue.cache_clear()
    scope = calls.read_text().splitlines()[-1]  # the sandbox's own scope
    assert "MemoryMax=256M" in scope
    assert ("OOMPolicy=continue" in scope) is (systemd >= 253)
    assert result.exit_code == 128 + 9 and result.infrastructure_error is None
    agent_dir = tmp_path / "agent"  # as an agent's bootstrap leaves it once it started the agent
    agent_dir.mkdir()
    for name in ("trajectory.jsonl", "stderr.log"):
        (agent_dir / name).write_text("")
    assert sandbox_failure(backend, spec, result, agent_dir) is None


def test_sandboxes_running_at_once_get_cpus_of_their_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sandboxes that run at the same time are pinned to different CPUs while there are enough
    (the least used ones) and give them back when they end. Every sandbox was pinned to the
    first CPUs: with --concurrency 4, four twins shared one CPU, ran into their time limit and
    were scored 0 (A9R1-02)."""
    import threading

    from real2simlab.sandbox import bwrap as module

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    unpack_tarball(_small_image(tmp_path), "test/cpus:1")
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/usr/bin/{name}")  # taskset
    _linux(monkeypatch, module)  # CPUs 0 to 3
    # Locks of its own: the r2s commands running on this machine are not part of the test.
    monkeypatch.setattr(module, "CPU_LOCK_DIR", tmp_path / "locks", raising=False)
    pinned: list[str] = []
    together = [threading.Barrier(3, timeout=30)]

    def run_logged(command: list[str], *_: Any, **__: Any) -> tuple[int | None, bool, float]:
        pinned.append(command[command.index("taskset") + 2])
        together[0].wait()  # each sandbox runs until all of them have started
        return 0, False, 0.1

    monkeypatch.setattr(module, "run_logged", run_logged)
    (tmp_path / "program").mkdir()
    backend = module.BwrapBackend("/bin/true", cgroup_controllers=frozenset())

    def spec(cpus: float) -> SandboxSpec:
        return SandboxSpec(
            image="test/cpus:1",
            argv=("python", "run.py"),
            workdir="/r2s/program",
            mounts=(Mount(tmp_path / "program", "/r2s/program"),),
            limits=Limits(timeout_s=10, cpus=cpus),
        )

    runs = [
        threading.Thread(target=backend.run, args=(spec(cpus), tmp_path / f"{index}.log"))
        for index, cpus in enumerate([1, 2, 1])
    ]
    for run in runs:
        run.start()
    for run in runs:
        run.join(30)
    assert sorted(int(cpu) for cpus in pinned for cpu in cpus.split(",")) == [0, 1, 2, 3], pinned
    pinned.clear()
    together[0] = threading.Barrier(1)
    backend.run(spec(4), tmp_path / "all.log")  # all of them were given back
    assert pinned == ["0,1,2,3"]


CPU_HOLDER = '''
"""An r2s command with one bwrap sandbox (limited to 1 CPU), which says the CPUs it was pinned
to in the file SAID and runs until the file GO exists. Usage: hold.py LOCKS PROGRAM SAID GO"""
import sys
import time
from pathlib import Path

from real2simlab.sandbox import bwrap
from real2simlab.sandbox.base import Limits, Mount, SandboxSpec

locks, program, said, go = map(Path, sys.argv[1:])
bwrap.CPU_LOCK_DIR = locks  # the test's own: the r2s commands running here are not part of it
bwrap.cpu_affinity = lambda: True  # nothing is run: taskset need not be installed
bwrap.os.sched_getaffinity = lambda pid: {0, 1, 2, 3}


def run_logged(command, *args, **kwargs):
    told = said.with_suffix(".partial")
    told.write_text(command[command.index("taskset") + 2])
    told.rename(said)
    while not go.exists():
        time.sleep(0.02)
    return 0, False, 0.1


bwrap.run_logged = run_logged
spec = SandboxSpec(
    image="test/cpus:1",
    argv=("python", "run.py"),
    workdir="/r2s/program",
    mounts=(Mount(program, "/r2s/program"),),
    limits=Limits(timeout_s=10, cpus=1.0),
)
bwrap.BwrapBackend("/bin/true", cgroup_controllers=frozenset()).run(spec, said.with_suffix(".log"))
'''


@pytest.mark.skipif(sys.platform != "linux", reason="CPU affinity is Linux's")
def test_commands_running_at_once_get_cpus_of_their_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two r2s commands whose bwrap sandboxes run at the same time pin them to different CPUs,
    and the CPU of a command that is killed (``kill -9``) is free again. Every command pinned
    its sandboxes to the first CPUs: two twins that each finished in time alone shared a CPU,
    ran into their time limit and were scored 0, with no warning (A9R1-02)."""
    import time

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    unpack_tarball(_small_image(tmp_path), "test/cpus:1")
    (tmp_path / "program").mkdir()
    script = tmp_path / "hold.py"
    script.write_text(CPU_HOLDER)
    go = tmp_path / "go"
    started: list[subprocess.Popen[bytes]] = []

    def command(name: str) -> tuple[subprocess.Popen[bytes], str]:
        """Start a command; the CPUs its sandbox was pinned to, once it runs."""
        said = tmp_path / f"{name}.cpus"
        arguments = [tmp_path / "locks", tmp_path / "program", said, go]
        process = subprocess.Popen(
            [sys.executable, str(script), *map(str, arguments)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        started.append(process)
        deadline = time.monotonic() + 60
        while not said.exists():
            if process.poll() is not None:
                assert process.stderr is not None
                pytest.fail(f"{name} ended: {process.stderr.read().decode()}")
            assert time.monotonic() < deadline, f"{name} did not start its sandbox"
            time.sleep(0.05)
        return process, said.read_text()

    try:
        first, one = command("first")
        second, two = command("second")
        assert not set(one.split(",")) & set(two.split(",")), (one, two)
        first.kill()  # kill -9 while its sandbox runs
        first.wait(30)
        third, three = command("third")
        assert three == one, (one, two, three)
    finally:
        go.touch()
        for process in started:
            try:
                process.wait(30)
            except subprocess.TimeoutExpired:
                process.kill()
    assert second.returncode == 0 and third.returncode == 0


def test_a_session_holds_cpus_of_its_own_and_checks_its_sandbox_on_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bwrap session (each command of a mini_swe agent starts its sandbox) holds CPUs of its
    own while it is open, from the same pool as runs: a run meanwhile gets other CPUs, and the
    session's are free again once it ends, also when its sandbox does not start. The check
    that it starts runs ``true`` in exactly the session's sandbox, on those CPUs."""
    from real2simlab.errors import InfrastructureError
    from real2simlab.sandbox import bwrap as module

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    unpack_tarball(_small_image(tmp_path), "test/cpus:1")
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/usr/bin/{name}")  # taskset
    _linux(monkeypatch, module)  # CPUs 0 to 3
    # Locks of its own: the r2s commands running on this machine are not part of the test.
    monkeypatch.setattr(module, "CPU_LOCK_DIR", tmp_path / "locks")
    checked: list[list[str]] = []
    starts = [0]  # the exit status of `true` in the session's sandbox

    def run_captured(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        checked.append(command)
        return subprocess.CompletedProcess(command, starts[0], "", "bwrap: no" * starts[0])

    pinned: list[str] = []

    def run_logged(command: list[str], *_: Any, **__: Any) -> tuple[int | None, bool, float]:
        pinned.append(command[command.index("taskset") + 2])
        return 0, False, 0.1

    monkeypatch.setattr(module, "run_captured", run_captured)
    monkeypatch.setattr(module, "run_logged", run_logged)
    (tmp_path / "w").mkdir()
    limits = Limits(timeout_s=10, cpus=2)
    mounts = (Mount(tmp_path / "w", "/w", read_only=False),)
    session_spec = SandboxSpec(image="test/cpus:1", argv=(), workdir="/w", mounts=mounts,
                               limits=limits)  # fmt: skip
    run_spec = SandboxSpec(image="test/cpus:1", argv=("python", "run.py"), workdir="/w",
                           mounts=mounts, limits=limits)  # fmt: skip
    backend = module.BwrapBackend("/bin/true", cgroup_controllers=frozenset())
    with backend.session(session_spec) as session:
        assert session.prefix[:3] == ("taskset", "-c", "0,1"), session.prefix
        assert checked == [[*session.prefix, "true"]]
        backend.run(run_spec, tmp_path / "meanwhile.log")
    backend.run(run_spec, tmp_path / "after.log")
    assert pinned == ["2,3", "0,1"]
    starts[0] = 1
    with pytest.raises(InfrastructureError, match="does not start"), backend.session(session_spec):
        pass
    backend.run(run_spec, tmp_path / "given-back.log")
    assert pinned[-1] == "0,1" and checked[-1][:3] == ["taskset", "-c", "0,1"], (pinned, checked)


def test_a_command_that_cannot_agree_on_cpus_with_others_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the CPU locks cannot be used (a file, or a directory that other users may write
    in), the sandboxes of one command still get CPUs of their own, and a warning says that
    another command's may be pinned to the same ones."""
    from real2simlab.sandbox import bwrap as module

    monkeypatch.setattr(module.shutil, "which", lambda name: f"/usr/bin/{name}")  # taskset
    _linux(monkeypatch, module)  # CPUs 0 to 3
    (tmp_path / "file").write_text("")
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    for locks, reason in ((tmp_path / "file", "not a directory"), (shared, "other users")):
        pool = module.CpuPool(locks)
        warned = pytest.warns(module.CpuLimitWarning, match=reason)
        with warned, pool.take(1) as first, pool.take(2) as second:
            assert first == [0] and second == [1, 2], (first, second)
