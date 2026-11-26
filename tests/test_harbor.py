"""Harbor export: the task layout, and the verifier's score equals the Lab's own."""

from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from real2simlab.bridges.harbor import export_task
from real2simlab.bridges.harbor_verify import main as verify_main
from real2simlab.orchestrate import parse_experiment, run_experiment

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_export_layout(bouncing_task: Path, tmp_path: Path) -> None:
    dest = export_task(bouncing_task, tmp_path / "harbor", solution="oracle")
    config = tomllib.loads((dest / "task.toml").read_text())
    assert config["schema_version"] == "1.4" and config["artifacts"] == [
        "/app/solution",
        "/app/output",
    ]
    assert config["verifier"]["environment_mode"] == "separate"
    assert config["verifier"]["environment"]["network_mode"] == "no-network"
    assert config["environment"]["network_mode"] == "allowlist"
    workspace = dest / "environment" / "workspace"
    assert (workspace / "visible" / "capture.json").is_file() and (
        workspace / "tools" / "r2s_twin.py"
    ).is_file()
    assert not any("hidden" in p.parts for p in (dest / "environment").rglob("*"))
    assert (dest / "tests" / "task" / "hidden" / "reference" / "tracks.npz").is_file()
    assert "/app/solution/run.py" in (dest / "instruction.md").read_text()
    assert list((dest / "tests" / "lab").glob("real2simlab-*.dist-info/entry_points.txt"))
    assert (dest / "solution" / "solve.sh").stat().st_mode & 0o111
    with pytest.raises(FileExistsError):
        export_task(bouncing_task, dest)
    contained = export_task(bouncing_task, tmp_path / "contained", self_contained=True)
    assert "python:3.11-slim" in (contained / "tests" / "Dockerfile").read_text()
    assert "pydantic==" in (contained / "tests" / "Dockerfile").read_text()


def run_solve(dest: Path, app: Path) -> Path:
    """Run solve.sh with /app pointing to ``app`` (a stand-in for Harbor's oracle agent)."""
    solve = dest / "solution" / "solve.sh"
    lines = solve.read_text(encoding="utf-8", errors="surrogateescape").splitlines(keepends=True)
    # The embedded files (base64, which may hold "/app" by chance) are left as they are.
    script = "".join(
        line if line.startswith("echo '") else line.replace("/app", str(app)) for line in lines
    )
    env = {**os.environ, "PYTHONPATH": str(app / "solution")}
    subprocess.run(
        ["bash", "-c", script.replace("python run.py", f"{sys.executable} run.py")],
        env=env,
        check=True,
    )
    return app / "solution"


def lab_score(task_root: Path, program: Path, tmp_path: Path) -> dict:
    config = parse_experiment(
        {
            "name": "lab",
            "tasks": {"paths": [str(task_root)]},
            "methods": [{"name": "program", "path": str(program)}],
            "run": {"backend": "local"},
        }
    )
    run_experiment(config, tmp_path / "lab-run", log=lambda line: None, report=False)
    return json.loads(next((tmp_path / "lab-run" / "results").rglob("seed-0.json")).read_text())


@pytest.mark.parametrize("solution", ["oracle", "physical"])
def test_verifier_reward_equals_the_lab_score(
    bouncing_task: Path, tmp_path: Path, solution: str
) -> None:
    if solution == "oracle":
        dest = export_task(bouncing_task, tmp_path / "harbor", solution="oracle")
    else:
        from real2simlab.benchmarks.synthetic.programs import simulation_program

        program = simulation_program(
            tmp_path / "physical", phenomenon="bouncing_ball", title="t",
            params={"restitution": 0.7, "rest_speed": 0.05},
        )  # fmt: skip
        dest = export_task(bouncing_task, tmp_path / "harbor", solution=program)
    app = tmp_path / "app"
    solved = run_solve(dest, app)
    logs = tmp_path / "logs"
    verify_main(["--task", str(dest / "tests" / "task"), "--solution", str(solved),
                 "--session-output", str(app / "output"), "--logs", str(logs),
                 "--no-isolation"])  # fmt: skip
    reward = json.loads((logs / "reward.json").read_text())
    harbor_result = json.loads((logs / "result.json").read_text())
    lab = lab_score(bouncing_task, solved, tmp_path)
    assert reward["reward"] == pytest.approx(lab["summary"]["reproduction_skill"], abs=1e-12)
    assert harbor_result["summary"] == lab["summary"]
    if solution == "oracle":
        assert reward["reward"] == pytest.approx(1.0)
    else:
        assert 0 < reward["reward"] < 1
    assert harbor_result["reproduction"]["session_consistent"] is True


def test_verifier_scores_missing_solutions_as_zero(bouncing_task: Path, tmp_path: Path) -> None:
    dest = export_task(bouncing_task, tmp_path / "harbor")
    (tmp_path / "empty").mkdir()
    logs = tmp_path / "logs"
    verify_main(
        [
            "--task",
            str(dest / "tests" / "task"),
            "--solution",
            str(tmp_path / "empty"),
            "--logs",
            str(logs),
            "--no-isolation",
        ]
    )
    reward = json.loads((logs / "reward.json").read_text())
    assert reward["reward"] == 0.0 and reward["delivered"] == 0.0


CHEAT = """\
import os, shutil
from pathlib import Path

# Copy the answer instead of simulating: the hidden reference is in the verifier's container.
for path in Path("/tests/task/hidden/reference").iterdir():
    shutil.copy(path, Path(os.environ["R2S_OUTPUT_DIR"]) / path.name)
"""


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a non-root user")
def test_verifier_refuses_to_run_solutions_it_cannot_isolate(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """Without root, the verifier cannot run the solution as another user, so it would share
    the hidden reference with it: it refuses with an error, and writes no reward (a reward of
    0 would penalise the solution for the verifier's problem)."""
    dest = export_task(bouncing_task, tmp_path / "harbor", solution="oracle")
    solved = run_solve(dest, tmp_path / "app")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "reward.json").write_text('{"reward": 1.0}')  # left by an earlier attempt
    status = verify_main(["--task", str(dest / "tests" / "task"), "--solution", str(solved),
                          "--logs", str(logs)])  # fmt: skip
    assert status == 1
    assert not (logs / "reward.json").exists()
    assert "root" in (logs / "error.txt").read_text()


def test_verifier_writes_no_reward_when_the_run_fails_outside_the_solution(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sandbox that does not start (infra_error) is not the solution's fault: no reward."""
    from real2simlab.sandbox.base import ExecResult
    from real2simlab.sandbox.local import LocalBackend

    dest = export_task(bouncing_task, tmp_path / "harbor", solution="oracle")
    solved = run_solve(dest, tmp_path / "app")
    monkeypatch.setattr(
        LocalBackend, "run", lambda self, spec, log: ExecResult(None, False, 0.0, "mount failed")
    )
    logs = tmp_path / "logs"
    status = verify_main(["--task", str(dest / "tests" / "task"), "--solution", str(solved),
                          "--logs", str(logs), "--no-isolation"])  # fmt: skip
    assert status == 1 and not (logs / "reward.json").exists()
    assert "infra_error" in (logs / "error.txt").read_text()
    assert json.loads((logs / "result.json").read_text())["reproduction"]["run_status"] == (
        "infra_error"
    )


@pytest.mark.docker
@pytest.mark.slow
def test_verifier_container_keeps_the_hidden_reference_from_the_solution(
    bouncing_task: Path, tmp_path: Path, docker_backend: object
) -> None:
    """In the exported verifier image, as Harbor runs it: a solution that copies the hidden
    reference gets nothing, and an honest one is scored as the Lab scores it."""
    import shutil
    import uuid

    from real2simlab.benchmarks.synthetic.programs import simulation_program
    from real2simlab.envs import image_reference

    base = image_reference("r2s-verifier")
    if subprocess.run(["docker", "image", "inspect", base], capture_output=True).returncode:
        pytest.skip(f"{base} is not built (r2s env build r2s-verifier)")
    dest = export_task(bouncing_task, tmp_path / "harbor")
    tag = f"r2s-test-verifier:{uuid.uuid4().hex[:12]}"
    subprocess.run(["docker", "build", "-q", "-t", tag, str(dest / "tests")], check=True,
                   capture_output=True)  # fmt: skip
    cheat = tmp_path / "cheat"
    cheat.mkdir()
    (cheat / "run.py").write_text(CHEAT)
    honest = simulation_program(
        tmp_path / "honest", phenomenon="bouncing_ball", title="t",
        params={"restitution": 0.7, "rest_speed": 0.05},
    )  # fmt: skip
    rewards: dict[str, float] = {}
    try:
        for name, program in (("cheat", cheat), ("honest", honest)):
            app, logs = tmp_path / name / "app", tmp_path / name / "logs"
            shutil.copytree(program, app / "solution")
            (app / "output").mkdir()
            logs.mkdir()
            subprocess.run(
                ["docker", "run", "--rm", "--network", "none",
                 "-v", f"{app}:/app", "-v", f"{logs}:/logs/verifier", tag,
                 "bash", "-c", "bash /tests/test.sh; chmod -R a+rwX /logs/verifier"],
                check=True, capture_output=True,
            )  # fmt: skip
            rewards[name] = json.loads((logs / "reward.json").read_text())["reward"]
    finally:
        subprocess.run(["docker", "image", "rm", "-f", tag], capture_output=True)
    assert rewards["cheat"] == 0.0
    assert rewards["honest"] == pytest.approx(
        lab_score(bouncing_task, honest, tmp_path)["summary"]["reproduction_skill"], abs=1e-12
    )
    assert 0 < rewards["honest"] < 1


def harbor_command() -> str | None:
    import shutil

    candidate = os.environ.get("R2S_HARBOR") or str(
        Path(__file__).resolve().parents[1] / ".venv-harbor" / "bin" / "harbor"
    )
    return candidate if Path(candidate).is_file() else shutil.which("harbor")


def test_harbor_method_writes_a_task_without_hidden_data(
    bouncing_task: Path, tmp_path: Path, local
) -> None:
    from real2simlab.methods import get_method
    from real2simlab.methods.base import BuildContext
    from real2simlab.task import load_task
    from real2simlab.task.instruction import task_statement

    task = load_task(bouncing_task)
    method = get_method("harbor", agent="codex", model="gpt-5")
    context = BuildContext(
        task_id=task.id, task_hash="h", config=task.config,
        statement=task_statement(task), visible_dir=task.visible_dir, work_dir=tmp_path / "work",
        twin_dir=tmp_path / "twin", backend=local,
    )  # fmt: skip
    written = method._task(context, tmp_path / "harbor")  # type: ignore[attr-defined]
    assert not any("hidden" in p.parts or p.name == "reference" for p in written.rglob("*"))
    config = tomllib.loads((written / "task.toml").read_text())
    assert config["artifacts"] == ["/workspace/solution", "/workspace/output"]
    assert config["environment"]["network_mode"] == "allowlist"
    assert method.label == "harbor:codex:gpt-5"


FAKE_HARBOR = '''#!{python}
"""A stand-in for `harbor`: `--version`, and a `run` that ends as {mode!r}."""
import json, os, pathlib, signal, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("0.27.0")
    sys.exit(0)
if {mode!r} == "killed":  # by something outside it (an administrator, the host's OOM killer)
    os.kill(os.getpid(), signal.SIGKILL)
if {mode!r} == "no docker":
    print("Docker daemon is not running. Please start Docker and try again.", file=sys.stderr)
    sys.exit(1)
# What the agent is given (--ae KEY=VALUE, a $NAME read from the environment), which Harbor's
# own errors may repeat.
given = [os.path.expandvars(args[i + 1].partition("=")[2]) for i, a in enumerate(args) if a == "--ae"]
trial = pathlib.Path(args[args.index("-o") + 1]) / args[args.index("--job-name") + 1] / "t__1"
(trial / "artifacts" / "workspace" / "solution").mkdir(parents=True)
ran = {{"started_at": "2026-11-23T08:00:00Z", "finished_at": "2026-11-23T08:01:00Z"}}
if {mode!r} == "environment failed":
    kind, valid, agent = "EnvironmentStartTimeoutError", False, None
else:
    kind, valid, agent = "NonZeroAgentExitCodeError", True, ran
policy = {{"should_retry": not valid, "should_grade": valid, "has_valid_reward": valid}}
info = {{"exception_type": kind, "exception_message": " ".join(["it stopped", *given]),
        "exception_traceback": "", "occurred_at": "2026-11-23T08:01:00Z", "policy": policy}}
result = {{"exception_info": info, "environment_setup": ran, "agent_execution": agent}}
(trial / "result.json").write_text(json.dumps(result))
sys.exit(1)
'''


def _harbor_build(task_dir: Path, tmp_path: Path, backend: Any, mode: str, **options: Any) -> Any:
    from real2simlab.methods import get_method
    from real2simlab.methods.base import BuildContext
    from real2simlab.task import load_task
    from real2simlab.task.instruction import task_statement

    harbor = tmp_path / "harbor"
    harbor.write_text(FAKE_HARBOR.format(python=sys.executable, mode=mode))
    harbor.chmod(0o755)
    task = load_task(task_dir)
    context = BuildContext(
        task_id=task.id, task_hash="h", config=task.config,
        statement=task_statement(task), visible_dir=task.visible_dir, work_dir=tmp_path / "work",
        twin_dir=tmp_path / "twin", backend=backend,
    )  # fmt: skip
    return get_method("harbor", agent="oracle", harbor=str(harbor), **options).build(context)


@pytest.mark.parametrize(
    ("mode", "outside", "status"),
    [
        ("no docker", "Docker daemon is not running", "infra_error"),
        ("environment failed", "EnvironmentStartTimeoutError", "infra_error"),
        ("killed", "killed by signal 9", "failed"),
        ("agent failed", None, "failed"),
    ],
)
def test_harbor_that_never_started_the_agent_is_not_a_score(
    bouncing_task: Path, tmp_path: Path, local, mode: str, outside: str | None, status: str
) -> None:
    """Harbor that could not run the agent at all (no Docker for its environment, an
    environment that did not start) did nothing of the method's: not a 0. Nor is a run that
    something outside Harbor killed; but it may have run the agent (``failed``: its run is
    spent). An agent that ran and failed is its result."""
    outcome = _harbor_build(bouncing_task, tmp_path, local, mode)
    record = json.loads((tmp_path / "work" / "agent" / "agent_record.json").read_text())
    assert outcome.twin is None and record["exit_status"] == status
    if outside is None:
        assert outcome.infra_error is None
    else:
        assert outside in (outcome.infra_error or "")


@pytest.mark.parametrize("budget", [{}, {"max_agent_runs": 1}], ids=["no budget", "one run"])
def test_harbor_that_ran_the_agent_before_this_machine_failed_is_not_run_again(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: dict[str, Any]
) -> None:
    """Harbor runs the agent from the host and writes its logs only once it ends: a build that
    this machine then fails (a full disk) before the agent's record is written ran the agent.
    It is counted as a run of unknown cost and not run again by the same command (its build
    is an ``infra_error``, redone by the next one), never given back and run again at once."""
    import errno

    from real2simlab.orchestrate.experiment import Experiment

    harbor, runs = tmp_path / "harbor", tmp_path / "harbor-runs"
    count = f'if args != ["--version"]:\n    open({str(runs)!r}, "a").write("run\\n")\n'
    script = FAKE_HARBOR.format(python=sys.executable, mode="agent failed")
    harbor.write_text(script.replace("args = sys.argv[1:]\n", "args = sys.argv[1:]\n" + count, 1))
    harbor.chmod(0o755)

    def full_disk(*args: Any, **kwargs: Any) -> Any:  # after Harbor ran the agent
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), "agent/trial_result.json")

    monkeypatch.setattr("real2simlab.agents.harbor.harbor_failure", full_disk)
    config = parse_experiment(
        {
            "name": "harbor-host-failure",
            "tasks": {"paths": [str(bouncing_task)]},
            "methods": [{"name": "harbor", "agent": "oracle", "harbor": str(harbor)}],
            "run": {"backend": "local", "retries": 2},
            "budget": budget,
        }
    )
    run = tmp_path / "run"
    Experiment(config, run, log=lambda line: None, report=False).run()
    assert runs.read_text().count("run") == 1
    spent = json.loads((run / "state" / "budget.json").read_text())
    spent.pop("wall_time_s", None)  # the wall-time budget, which its own tests check
    assert spent == {"agent_runs": 1, "cost_usd": 0.0, "cost_unknown_runs": 1}


@pytest.mark.parametrize("how", ["cannot be spawned", "stopped just before"])
def test_harbor_that_never_started_gives_its_run_back(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """Harbor whose process never started, because it cannot be spawned or because the
    experiment was stopped (Ctrl-C) just before, ran nothing of the agent: its reserved run is
    given back, never counted as one of unknown cost (with which a cost budget would start no
    further agent); one that cannot be spawned is tried again."""
    import errno

    from real2simlab.agents.harbor import HarborAgent
    from real2simlab.orchestrate.experiment import Experiment
    from real2simlab.sandbox import process

    harbor, runs = tmp_path / "harbor", tmp_path / "harbor-runs"
    count = f'if args != ["--version"]:\n    open({str(runs)!r}, "a").write("run\\n")\n'
    script = FAKE_HARBOR.format(python=sys.executable, mode="agent failed")
    harbor.write_text(script.replace("args = sys.argv[1:]\n", "args = sys.argv[1:]\n" + count, 1))
    harbor.chmod(0o755)
    spawned: list[str] = []
    if how == "cannot be spawned":
        popen = subprocess.Popen

        def cannot_spawn(args: Any, *more: Any, **options: Any) -> Any:
            if list(args)[:1] == [str(harbor)] and "--version" not in args:
                spawned.append("refused")  # Popen fails: the executable cannot run
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(harbor))
            return popen(args, *more, **options)

        monkeypatch.setattr(subprocess, "Popen", cannot_spawn)
    else:
        prepare = HarborAgent._task  # pyright: ignore[reportPrivateUsage]

        def then_stopped(self: HarborAgent, *args: Any, **options: Any) -> Path:
            task = prepare(self, *args, **options)
            process.cancel_all()  # Ctrl-C while Harbor's task was prepared
            return task

        monkeypatch.setattr(HarborAgent, "_task", then_stopped)
    config = parse_experiment(
        {
            "name": "harbor-never-started",
            "tasks": {"paths": [str(bouncing_task)]},
            "methods": [{"name": "harbor", "agent": "oracle", "harbor": str(harbor)}],
            "run": {"backend": "local", "retries": 2},
        }
    )
    run = tmp_path / "run"
    experiment = Experiment(config, run, log=lambda line: None, report=False)
    try:
        if how == "cannot be spawned":
            experiment.run()
        else:
            with pytest.raises(KeyboardInterrupt):
                experiment.run()
    finally:
        process.reset_cancel()
    assert not runs.exists()  # Harbor never ran
    spent = json.loads((run / "state" / "budget.json").read_text())
    spent.pop("wall_time_s", None)  # the wall-time budget, which its own tests check
    assert spent == {
        "agent_runs": 0,
        "cost_usd": 0.0,
    }
    assert len(spawned) == (3 if how == "cannot be spawned" else 0)  # given back, tried again


HARBOR_KEY = "qa-canary-harbor-" + "h" * 24  # no key's shape: only its value can redact it


def test_what_harbor_says_is_redacted_before_it_is_recorded(
    bouncing_task: Path, tmp_path: Path, local, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Harbor's error for an environment it could not start can repeat what the agent was
    given, its key included: the build's reason is redacted before it is recorded (the
    records, the results and the log all take it from there)."""
    monkeypatch.setenv("R2S_TEST_HARBOR_KEY", HARBOR_KEY)
    env = {"OPENAI_API_KEY": "$R2S_TEST_HARBOR_KEY"}
    outcome = _harbor_build(bouncing_task, tmp_path, local, "environment failed", agent_env=env)
    assert "EnvironmentStartTimeoutError: it stopped [REDACTED]" in (outcome.infra_error or "")
    assert HARBOR_KEY not in (outcome.failure or "") + (outcome.infra_error or "")
    for name in ("agent_record.json", "harbor.log", "trial_result.json"):
        assert HARBOR_KEY not in (tmp_path / "work" / "agent" / name).read_text(), name
    # Harbor's own job directory repeats it too: cleaned with the agent's files, although the
    # build is not a result.
    written = [path for path in (tmp_path / "work").rglob("*") if path.is_file()]
    assert written and not any(HARBOR_KEY.encode() in path.read_bytes() for path in written)


def test_harbor_gets_credentials_by_name_never_on_its_command_line(
    bouncing_task: Path, tmp_path: Path, local, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``agent_env = {KEY = "$NAME"}`` reaches ``harbor run`` as ``KEY=${NAME}``, which Harbor
    resolves from the environment it is given: the value never appears on a command line (where
    every user of the machine can read it), nor in what the Lab writes. An unset variable is an
    error with the fix, not an empty key. The key is a made-up canary."""
    from real2simlab.errors import SetupError
    from real2simlab.methods import get_method
    from real2simlab.methods.base import BuildContext
    from real2simlab.task import load_task
    from real2simlab.task.instruction import task_statement

    canary = "cnry-harbor-key-8d2a61"
    calls = tmp_path / "calls"
    calls.mkdir()
    harbor = tmp_path / "harbor"
    harbor.write_text(  # records its arguments only, never its environment
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('0.27.0')\n"
        "    sys.exit(0)\n"
        f"with open(os.path.join({str(calls)!r}, str(os.getpid()) + '.json'), 'w') as out:\n"
        "    json.dump(sys.argv[1:], out)\n"
        "sys.exit(1)\n"
    )
    harbor.chmod(0o755)
    task = load_task(bouncing_task)

    def build(name: str) -> object:
        method = get_method(
            "harbor", harbor=str(harbor), agent="terminus-2", model="openai/test",
            agent_env={"OPENAI_API_KEY": "$R2S_TEST_HARBOR_KEY", "MODE": "plain"},
        )  # fmt: skip
        context = BuildContext(
            task_id=task.id, task_hash="h", config=task.config, statement=task_statement(task),
            visible_dir=task.visible_dir, work_dir=tmp_path / name, twin_dir=tmp_path / "twin",
            backend=local,
        )  # fmt: skip
        return method.build(context)

    monkeypatch.setenv("R2S_TEST_HARBOR_KEY", canary)
    build("work")
    (call,) = calls.glob("*.json")
    argv = json.loads(call.read_text())
    assert not any(canary in argument for argument in argv)
    passed = [argv[i + 1] for i, argument in enumerate(argv) if argument == "--ae"]
    assert passed == ["OPENAI_API_KEY=${R2S_TEST_HARBOR_KEY}", "MODE=plain"]
    written = [p for p in (tmp_path / "work").rglob("*") if p.is_file()]
    assert written and not any(canary.encode() in path.read_bytes() for path in written)

    monkeypatch.delenv("R2S_TEST_HARBOR_KEY")
    with pytest.raises(SetupError, match="R2S_TEST_HARBOR_KEY") as unset:
        build("work-unset")
    assert "export R2S_TEST_HARBOR_KEY=" in (unset.value.fix or "")
    assert len(list(calls.glob("*.json"))) == 1  # Harbor was not run without its key


TRIAL_THAT_REPEATS_THE_KEY = '''#!{python}
"""A stand-in for `harbor`: a `run` whose agent started and kept a small file and a long log in
its trial (`agent/`), both repeating the key it was given, and which prints the key too."""
import json, os, pathlib, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("0.27.0")
    sys.exit(0)
key = os.path.expandvars(args[args.index("--ae") + 1].partition("=")[2])
trial = pathlib.Path(args[args.index("-o") + 1]) / args[args.index("--job-name") + 1] / "t__1"
(trial / "artifacts" / "workspace").mkdir(parents=True)
agent = trial / "agent"
agent.mkdir()
(agent / "a-key.txt").write_text(key + "\\n")  # copied first: the copy goes by name
with open(agent / "big.log", "w") as log:
    for i in range(20_000):
        log.write(f"{{i:05d}} the agent was given {{key}}\\n")
ran = {{"started_at": "2026-11-23T08:00:00Z", "finished_at": "2026-11-23T08:01:00Z"}}
(trial / "result.json").write_text(json.dumps({{"agent_execution": ran}}))
for i in range(20_000):
    print(f"{{i:05d}} Harbor ran the agent with {{key}}")
'''


def test_harbors_trial_is_copied_within_its_limit_then_cleaned_of_the_agents_key(
    bouncing_task: Path, tmp_path: Path, local, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The copy of Harbor's trial (``agent/trial``) keeps the first and the last whole lines of
    a long file within the limit of an agent's trajectory, and it is made before the agent's
    files are cleaned of the key it was given: what the copy kept holds the key replaced, and
    no file of the build holds it, even where a file of Harbor's own job directory could not be
    cleaned in place (it is left there, and said). The key is a made-up canary."""
    import errno
    import importlib

    from real2simlab.agents import harbor as module
    from real2simlab.methods import get_method
    from real2simlab.methods.base import BuildContext
    from real2simlab.task import load_task
    from real2simlab.task.instruction import task_statement

    redact = importlib.import_module("real2simlab.agents.redact")  # the module, not the function
    for name, value in (
        ("LOG_LIMIT_BYTES", 256 * 1024), ("OUTPUT_LIMIT", 512 * 1024), ("OUTPUT_TAIL", 64 * 1024)
    ):  # fmt: skip
        monkeypatch.setattr(module, name, value)  # the test writes little
    monkeypatch.setenv("R2S_TEST_HARBOR_KEY", HARBOR_KEY)
    harbor = tmp_path / "harbor"
    harbor.write_text(TRIAL_THAT_REPEATS_THE_KEY.format(python=sys.executable))
    harbor.chmod(0o755)
    task = load_task(bouncing_task)
    env = {"OPENAI_API_KEY": "$R2S_TEST_HARBOR_KEY"}

    def build(name: str) -> Any:
        context = BuildContext(
            task_id=task.id, task_hash="h", config=task.config, statement=task_statement(task),
            visible_dir=task.visible_dir, work_dir=tmp_path / name, twin_dir=tmp_path / "twin",
            backend=local,
        )  # fmt: skip
        return get_method("harbor", agent="oracle", harbor=str(harbor), agent_env=env).build(
            context
        )

    def holding(root: Path) -> list[str]:
        files = sorted(path for path in root.rglob("*") if path.is_file())
        assert files
        held = [path for path in files if HARBOR_KEY.encode() in path.read_bytes()]
        return [path.relative_to(root).as_posix() for path in held]

    build("work")
    assert holding(tmp_path / "work") == []
    agent = tmp_path / "work" / "agent"
    assert (agent / "trial" / "a-key.txt").read_text(encoding="utf-8") == "[REDACTED]\n"
    copied = (agent / "trial" / "big.log").read_text(encoding="utf-8")
    assert len(copied.encode()) <= 512 * 1024 and "left out here" in copied, len(copied)
    lines = copied.splitlines()
    assert lines[0] == "00000 the agent was given [REDACTED]", lines[0]
    assert lines[-1] == "19999 the agent was given [REDACTED]", lines[-1]
    log = (agent / "harbor.log").read_text(encoding="utf-8")
    assert "Harbor ran the agent with [REDACTED]" in log and "left out here" in log
    assert json.loads((agent / "agent_record.json").read_text())["redactions"] > 0

    # Harbor's job directory cannot be cleaned in place: its files can be neither rewritten,
    # removed nor emptied (as for a directory the Lab cannot change).
    def stuck(path: Path) -> bool:
        return "jobs" in Path(path).parts

    rewrite, unlink, empty = redact._Scan._rewrite, redact._Scan._unlink, redact._empty

    def no_rewrite(self: Any, path: Path, mode: int) -> int:
        if stuck(path):
            raise PermissionError(errno.EPERM, os.strerror(errno.EPERM), str(path))
        return rewrite(self, path, mode)

    def no_unlink(self: Any, path: Path) -> None:
        if stuck(path):
            raise PermissionError(errno.EPERM, os.strerror(errno.EPERM), str(path))
        unlink(self, path)

    monkeypatch.setattr(redact._Scan, "_rewrite", no_rewrite)
    monkeypatch.setattr(redact._Scan, "_unlink", no_unlink)
    monkeypatch.setattr(redact, "_empty", lambda path: not stuck(path) and empty(path))
    outcome = build("stuck")
    left = holding(tmp_path / "stuck")
    assert left and all(name.startswith("harbor/jobs/") for name in left), left  # said, below
    assert not outcome.delivered and "harbor/jobs/" in (outcome.failure or ""), outcome


TRIAL_WHOSE_AGENT_IS_A_LINK = '''#!{python}
"""A stand-in for `harbor`: a `run` whose trial's agent directory is a link to a directory
elsewhere on this machine."""
import json, os, pathlib, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("0.27.0")
    sys.exit(0)
trial = pathlib.Path(args[args.index("-o") + 1]) / args[args.index("--job-name") + 1] / "t__1"
(trial / "artifacts" / "workspace").mkdir(parents=True)
os.symlink({elsewhere!r}, trial / "agent")
ran = {{"started_at": "2026-11-23T08:00:00Z", "finished_at": "2026-11-23T08:01:00Z"}}
(trial / "result.json").write_text(json.dumps({{"agent_execution": ran}}))
'''


def test_harbors_trial_whose_agent_directory_is_a_link_is_not_copied(
    bouncing_task: Path, tmp_path: Path, local
) -> None:
    """Harbor's trial whose agent directory is itself a link is not followed: nothing behind it
    (a directory elsewhere on this machine, here the test's own) is read into the build, and
    the record names no trajectory. Below that directory, the bounded copy leaves out links."""
    from real2simlab.methods import get_method
    from real2simlab.methods.base import BuildContext
    from real2simlab.task import load_task
    from real2simlab.task.instruction import task_statement

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "notes.txt").write_text("not the agent's\n")
    harbor = tmp_path / "harbor"
    harbor.write_text(
        TRIAL_WHOSE_AGENT_IS_A_LINK.format(python=sys.executable, elsewhere=str(elsewhere))
    )
    harbor.chmod(0o755)
    task = load_task(bouncing_task)
    context = BuildContext(
        task_id=task.id, task_hash="h", config=task.config,
        statement=task_statement(task), visible_dir=task.visible_dir, work_dir=tmp_path / "work",
        twin_dir=tmp_path / "twin", backend=local,
    )  # fmt: skip
    get_method("harbor", agent="oracle", harbor=str(harbor)).build(context)
    agent = tmp_path / "work" / "agent"
    assert not (agent / "trial").exists()
    held = [path for path in agent.rglob("*") if path.is_file()]
    assert held and not any(b"not the agent's" in path.read_bytes() for path in held)
    assert json.loads((agent / "agent_record.json").read_text()).get("trajectory") is None
    assert (elsewhere / "notes.txt").read_text() == "not the agent's\n"


DAMAGED_RESULTS: dict[str, bytes | None] = {
    "cut short": b'{"trial_name": "t__1", "exception_info": nu',
    "not text": b"\xff\xfe{}",
    "not an object": b"[]",
    "a link": None,  # to a valid record elsewhere: never followed
}
HARBOR_DAMAGED = '''#!{python}
"""A stand-in for `harbor`: `--version`, and a `run` whose trial holds the agent's solution but
a damaged result.json (Harbor stopped while it wrote it; a disk that failed), or a link."""
import os, pathlib, shutil, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("0.27.0")
    sys.exit(0)
with open({calls!r}, "a") as calls:
    calls.write("run\\n")
trial = pathlib.Path(args[args.index("-o") + 1]) / args[args.index("--job-name") + 1] / "t__1"
shutil.copytree({solution!r}, trial / "artifacts" / "workspace" / "solution")
result = {result!r}
if result is None:
    pathlib.Path({elsewhere!r}).write_text('{{"trial_name": "t__1"}}')
    os.symlink({elsewhere!r}, trial / "result.json")
else:
    (trial / "result.json").write_bytes(result)
'''


@pytest.mark.parametrize("damage", sorted(DAMAGED_RESULTS))
def test_a_damaged_harbor_trial_result_is_not_a_score_and_stops_nothing_else(
    bouncing_task: Path, tmp_path: Path, damage: str
) -> None:
    """Harbor's record of its trial (``result.json``) cut short, not text, not a JSON object or
    not a regular file (a link, never followed, even to a valid record): whether and how the
    agent ran is not known, so the build is not a result (``infra_error``, naming the file) and
    its run is spent (``failed``: counted, and not run again by this command; the next run
    builds it again). The other methods are scored. A damaged record used to stop the whole
    experiment with the decoder's words alone; a link was followed."""
    calls = tmp_path / "calls"
    harbor = tmp_path / "harbor"
    solution = str(bouncing_task / "hidden" / "generator" / "program")
    script = HARBOR_DAMAGED.format(
        python=sys.executable, calls=str(calls), result=DAMAGED_RESULTS[damage], solution=solution,
        elsewhere=str(tmp_path / "elsewhere.json"),
    )  # fmt: skip
    harbor.write_text(script)
    harbor.chmod(0o755)
    config = parse_experiment(
        {
            "name": "harbor",
            "tasks": {"paths": [str(bouncing_task)]},
            "methods": [{"name": "harbor", "agent": "oracle", "harbor": str(harbor)},
                        {"name": "static"}],
            "run": {"backend": "local", "retries": 2},
        }
    )  # fmt: skip
    out = tmp_path / "run"
    results = run_experiment(config, out, log=lambda line: None, report=False)
    rows = {row["method"]: row for row in results.to_json()["rows"]}
    assert rows["static"]["infra_error"] is None and rows["static"]["reproduction_skill"] == 0.0
    assert "result.json" in rows["harbor"]["infra_error"]
    assert "cannot be read" in rows["harbor"]["infra_error"]
    (record,) = out.glob("builds/*/harbor/trial-0/work/agent/agent_record.json")
    assert json.loads(record.read_text())["exit_status"] == "failed"
    assert calls.read_text() == "run\n"  # not run again by this command
    assert json.loads((out / "state" / "budget.json").read_text())["agent_runs"] == 1


@pytest.mark.docker
@pytest.mark.slow
def test_harbor_method_end_to_end(bouncing_task: Path, tmp_path: Path, docker_backend) -> None:
    harbor = harbor_command()
    if harbor is None:
        pytest.skip("Harbor is not installed (set R2S_HARBOR)")
    from real2simlab.benchmarks.synthetic.programs import simulation_program

    program = simulation_program(
        tmp_path / "physical", phenomenon="bouncing_ball", title="t",
        params={"restitution": 0.7, "rest_speed": 0.05},
    )  # fmt: skip
    config = parse_experiment(
        {
            "name": "harbor",
            "tasks": {"paths": [str(bouncing_task)]},
            "methods": [
                {
                    "name": "harbor",
                    "label": "via-harbor",
                    "agent": "oracle",
                    "harbor": harbor,
                    "solution": str(program),
                },
                {"name": "program", "label": "direct", "path": str(program)},
            ],
            "run": {"backend": "docker"},
        }
    )
    results = run_experiment(
        config, tmp_path / "run", backend=docker_backend, log=lambda line: None, report=False
    )
    scores = {m.method: m.reproduction_skill.mean for m in results.methods}
    assert scores["via-harbor"] == scores["direct"] and 0 < scores["direct"] < 1


def test_only_tasks_with_a_run_py_solution_are_exported(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """Other conventions need the benchmark's own runtime: refused with the reason and a fix."""
    import shutil

    from real2simlab.errors import SetupError
    from real2simlab.task import TaskConfig
    from real2simlab.task.io import read_task_config, write_task_config

    task = tmp_path / "task"
    shutil.copytree(bouncing_task, task)
    data = read_task_config(task / "task.toml").to_json()
    data["delivery"]["convention"] = "build_sh_world"
    write_task_config(TaskConfig.model_validate(data), task / "task.toml")
    with pytest.raises(SetupError, match="build_sh_world delivery convention") as raised:
        export_task(task, tmp_path / "harbor", solution="oracle")
    assert "r2s eval --tasks" in (raised.value.fix or "")
    assert not (tmp_path / "harbor").exists()


DECOY = "R2S-TEST-DECOY harbor export"


@pytest.mark.parametrize(
    "mistake", ["link to hidden", "link outside", "copy of hidden", "statement copying hidden"]
)
def test_a_task_that_shows_hidden_data_is_not_exported(
    mistake: str, bouncing_task: Path, tmp_path: Path
) -> None:
    """``r2s export harbor`` validates the task as ``r2s eval`` does: a task that shows what a
    method must not see (in ``visible/``, a link into ``hidden/`` or out of the task, a copy of
    a hidden file; a copy of a hidden file as the ``instruction.md`` the agent is given) is
    refused with the fix, and nothing of it reaches the agent's environment, the verifier or
    the destination."""
    from real2simlab.errors import SetupError

    task = tmp_path / "task"
    shutil.copytree(bouncing_task, task)
    (task / "hidden" / "decoy.txt").write_text(DECOY)
    (tmp_path / "decoy.txt").write_text(DECOY)  # outside the task
    notes = task / "visible" / "notes.txt"
    if mistake == "link to hidden":
        notes.symlink_to(Path("..") / "hidden" / "decoy.txt")
    elif mistake == "link outside":
        notes.symlink_to(tmp_path / "decoy.txt")
    elif mistake == "copy of hidden":
        shutil.copyfile(task / "hidden" / "decoy.txt", notes)
    else:
        shutil.copyfile(task / "hidden" / "decoy.txt", task / "instruction.md")
    with pytest.raises(SetupError, match="is invalid") as raised:
        export_task(task, tmp_path / "harbor", solution="oracle")
    shown = "instruction.md" if mistake.startswith("statement") else "visible/notes.txt"
    assert shown in str(raised.value)
    assert f"r2s task validate {task}" in (raised.value.fix or "")
    assert not (tmp_path / "harbor").exists()
    leftovers = [p for p in tmp_path.iterdir() if p.name not in ("task", "decoy.txt")]
    assert not leftovers, leftovers  # no partial export beside the destination either


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
def test_a_solution_that_cannot_be_copied_scores_zero(bouncing_task: Path, tmp_path: Path) -> None:
    """Its own fault: a reward of 0, not a verifier error (which gives no reward)."""
    dest = export_task(bouncing_task, tmp_path / "harbor", solution="oracle")
    solved = run_solve(dest, tmp_path / "app")
    os.mkfifo(solved / "pipe")
    logs = tmp_path / "logs"
    status = verify_main(["--task", str(dest / "tests" / "task"), "--solution", str(solved),
                          "--logs", str(logs), "--no-isolation"])  # fmt: skip
    assert status == 0
    reward = json.loads((logs / "reward.json").read_text())
    assert reward["reward"] == 0.0 and reward["delivered"] == 0.0


def test_a_solution_is_recreated_whatever_its_file_names_hold(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """Quotes, spaces, ``$``, a leading dash, a name that is not UTF-8 (GBK bytes) and bytes
    whose base64 holds ``/app``: the exported ``solve.sh`` (and the harbor method's, rooted
    elsewhere) recreates every file exactly."""
    from real2simlab.bridges.harbor import (
        _solve_sh,  # pyright: ignore[reportPrivateUsage]
    )

    program = tmp_path / "program"
    shutil.copytree(bouncing_task / "hidden" / "generator" / "program", program)
    files = {
        "user's data.txt": b"hello",
        "$HOME and `cmd`.txt": b"x",
        "-starts with a dash": b"y",
        os.fsdecode(b"\xd6\xd0\xce\xc4.txt"): b"gbk",
        'dir with "quotes"/inner.bin': bytes([0xFE, 0x9A, 0xA9]) * 4,  # base64: /app/app...
    }
    for name, data in list(files.items()):
        (program / name).parent.mkdir(parents=True, exist_ok=True)
        try:
            (program / name).write_bytes(data)
        except OSError as error:  # APFS (macOS) refuses names that are not UTF-8: the others
            if error.errno != errno.EILSEQ:
                raise
            del files[name]
    dest = export_task(bouncing_task, tmp_path / "export", solution=program)
    solve = dest / "solution" / "solve.sh"
    assert subprocess.run(["bash", "-n", str(solve)], check=False).returncode == 0
    recreated = run_solve(dest, tmp_path / "app")
    elsewhere = tmp_path / "workspace"
    script = _solve_sh(program, app=str(elsewhere)).replace("python run.py", "true")
    subprocess.run(["bash", "-c", script], check=True, stdin=subprocess.DEVNULL)
    for root in (recreated, elsewhere / "solution"):
        for name, data in files.items():
            assert (root / name).read_bytes() == data, (root, name)


def _helper_program(reference: Path, target: Path) -> Path:
    """A correct program whose ``run.py`` runs an executable helper (``./helper.sh``)."""
    target.mkdir(parents=True)
    for path in reference.iterdir():
        shutil.copyfile(path, target / path.name)
    (target / "run.py").write_text(
        "import subprocess\nsubprocess.run(['./helper.sh'], check=True)\n"
    )
    (target / "helper.sh").write_text(
        '#!/bin/sh\ncp meta.json camera.json tracks.npz "$R2S_OUTPUT_DIR/"\n'
    )
    (target / "helper.sh").chmod(0o755)
    return target


def test_an_exported_solution_keeps_its_executable_files_executable(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """``solve.sh`` recreates an executable helper as executable: Harbor's verifier then scores
    the program the Lab scored, not one whose ``./helper.sh`` is refused."""
    program = _helper_program(bouncing_task / "hidden" / "reference", tmp_path / "program")
    dest = export_task(bouncing_task, tmp_path / "export", solution=program)
    (tmp_path / "app" / "output").mkdir(parents=True)  # Harbor's environment has it
    recreated = run_solve(dest, tmp_path / "app")
    assert os.access(recreated / "helper.sh", os.X_OK)
    assert not os.access(recreated / "meta.json", os.X_OK)
    assert (tmp_path / "app" / "output" / "tracks.npz").is_file()  # the helper ran


def test_a_twin_package_is_exported_as_its_program(bouncing_task: Path, tmp_path: Path) -> None:
    """A twin package (``twin.json`` and ``program/``) is a solution, as everywhere a program
    is: ``solve.sh`` recreates its program, whose ``run.py`` then runs (it embedded the
    package, ran no ``run.py`` and said nothing)."""
    package = Path(shutil.copytree(bouncing_task / "hidden" / "generator", tmp_path / "twin"))
    dest = export_task(bouncing_task, tmp_path / "export", solution=package)
    recreated = run_solve(dest, tmp_path / "app")
    assert (recreated / "run.py").is_file() and not (recreated / "twin.json").exists()
    assert (tmp_path / "app" / "output" / "tracks.npz").is_file()


def test_solve_sh_says_when_its_program_fails(bouncing_task: Path, tmp_path: Path) -> None:
    """``python run.py || true`` hid a program that failed (and, with no ``run.py``, a
    solution that was not one). ``solve.sh`` still ends normally, since the verifier scores
    the program (its failure is a reward of 0), but says that it failed."""
    program = tmp_path / "program"
    program.mkdir()
    (program / "run.py").write_text("raise SystemExit('broken on purpose')\n")
    dest = export_task(bouncing_task, tmp_path / "export", solution=program)
    app = tmp_path / "app"
    script = "".join(
        line if line.startswith("echo '") else line.replace("/app", str(app))
        for line in (dest / "solution" / "solve.sh").read_text().splitlines(keepends=True)
    )
    solved = subprocess.run(
        ["bash", "-c", script.replace("python run.py", f"{sys.executable} run.py")],
        capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL,
    )  # fmt: skip
    assert solved.returncode == 0, solved.stderr
    assert "broken on purpose" in solved.stderr and "run.py failed" in solved.stderr
