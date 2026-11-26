"""What a method leaves in the directories its sandbox could write is data it controls.

An agent's workspace (``solution/``, its session output, ``TASK.md``) and its log directory
(``/r2s/logs``: ``bootstrap.json``, ``trajectory.jsonl``, ``stderr.log``, where the Lab also
writes ``agent_record.json``) are writable in its sandbox, and the Lab reads them on the host
once the sandbox has ended. Anything there may be a symbolic link, which the host would resolve
(reaching what the sandbox kept from the agent), or a named pipe, which would keep the Lab
waiting forever. These tests leave such entries where an agent could, every link pointing at a
decoy of the test's own in its temporary directory, and check that the Lab neither reads nor
writes through them: a decoy is never changed or read, nothing is delivered, scored or exported
through a link, a pipe never blocks, and a session output that cannot be read as its kind is a
mismatch, not a crash."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import stat
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest

from real2simlab.agents.base import clean_solution, read_events
from real2simlab.benchmarks.synthetic.programs import simulation_program
from real2simlab.errors import DeliveryError
from real2simlab.methods import get_method
from real2simlab.methods.base import BuildContext, BuildOutcome
from real2simlab.orchestrate import Experiment, parse_experiment
from real2simlab.sandbox.base import ExecResult, SandboxSpec
from real2simlab.sandbox.local import LocalBackend
from real2simlab.scoring import MethodInfo, RunOutput, Scorer
from real2simlab.scoring.model import Headline
from real2simlab.task import TaskConfig, load_task
from real2simlab.task.instruction import task_statement
from real2simlab.task.io import read_task_config, write_task_config
from real2simlab.train import export_dataset, score_solution

MARKER = "decoy-7d41: a file of the test's own, outside every sandbox"
"""What every decoy holds: it must never reach a record, a score or a dataset."""


def _decoy(tmp_path: Path, name: str = "decoy.txt", text: str = MARKER) -> Path:
    decoy = tmp_path / "decoys" / name
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_text(text + "\n", encoding="utf-8")
    return decoy


def _program(target: Path) -> Path:
    """A valid program (it simulates the ball): what a link could make the Lab take."""
    return simulation_program(
        target, phenomenon="bouncing_ball", title="t",
        params={"restitution": 0.7, "rest_speed": 0.05},
    )  # fmt: skip


def _in_time(function: Callable[[], Any], pipes: Sequence[Path] = (), seconds: float = 20) -> Any:
    """``function()``, which must return within ``seconds``: waiting on a named pipe fails the
    test. Each of ``pipes`` still a pipe then has both its ends opened and closed, so a reader
    or a writer waiting on it goes on and nothing is left blocked."""
    outcome: dict[str, Any] = {}

    def call() -> None:
        try:
            outcome["value"] = function()
        except BaseException as error:  # raised again in the test's own thread
            outcome["error"] = error

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(seconds)
    waited = thread.is_alive()
    for pipe in pipes:
        with contextlib.suppress(OSError):
            if not stat.S_ISFIFO(os.lstat(pipe).st_mode):
                continue
            reader = os.open(pipe, os.O_RDONLY | os.O_NONBLOCK)  # a waiting writer goes on
            with contextlib.suppress(OSError):
                os.close(os.open(pipe, os.O_WRONLY | os.O_NONBLOCK))  # a waiting reader too
            os.close(reader)
    thread.join(10)
    assert not waited, f"the Lab waited on a named pipe for {seconds:.0f} s"
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


class LeavesBehind(LocalBackend):
    """The local backend, whose agent sandbox runs nothing: it leaves in the agent's workspace
    and log directory what ``leave`` puts there (a solution, a link to a decoy, a named pipe),
    as an agent could before its sandbox ends. Twins run as usual."""

    def __init__(self, leave: Callable[[Path, Path], None]) -> None:
        super().__init__(quiet=True)
        self.leave = leave

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        if not spec.label.startswith("r2s-agent"):
            return super().run(spec, log_path)
        mounts = {mount.target: Path(mount.source) for mount in spec.mounts}
        Path(log_path).write_text("", encoding="utf-8")
        self.leave(mounts["/workspace"], mounts["/r2s/logs"])
        return ExecResult(exit_code=0, timed_out=False, duration_s=0.0)


def _build(
    task_dir: Path, tmp_path: Path, leave: Callable[[Path, Path], None], pipes: Sequence[Path] = ()
) -> BuildOutcome:
    """The scripted agent's build, with ``leave`` as what its sandbox left."""
    task = load_task(task_dir)
    context = BuildContext(
        task_id=task.id, task_hash="h", config=task.config, statement=task_statement(task),
        visible_dir=task.visible_dir, work_dir=tmp_path / "work", twin_dir=tmp_path / "twin",
        backend=LeavesBehind(leave),
    )  # fmt: skip
    method = get_method("fake_agent", mode="physical")
    return _in_time(lambda: method.build(context), pipes)


# --- the agent's log directory: the Lab's record, and what it reads -----------------------------


@pytest.mark.parametrize("left", ["a link to a file outside", "a named pipe", "a directory"])
def test_the_agent_record_is_never_written_through_what_the_agent_left(
    bouncing_task: Path, tmp_path: Path, left: str
) -> None:
    decoy = _decoy(tmp_path)
    record = tmp_path / "work" / "agent" / "agent_record.json"

    def leave(workspace: Path, logs: Path) -> None:
        _program(workspace / "solution")
        if left == "a link to a file outside":
            record.symlink_to(decoy)
        elif left == "a named pipe":
            os.mkfifo(record)
        else:
            (record / "inside").mkdir(parents=True)

    outcome = _build(bouncing_task, tmp_path, leave, pipes=[record])
    assert outcome.delivered and outcome.agent_record == record
    assert decoy.read_text(encoding="utf-8") == MARKER + "\n"  # never written through
    assert not record.is_symlink() and record.is_file()
    assert json.loads(record.read_text(encoding="utf-8"))["spec"] == "r2s-agent-run/1.0"


@pytest.mark.parametrize(
    "name", ["trajectory.jsonl", "bootstrap.json", "stderr.log", "last_message.md"]
)
@pytest.mark.parametrize("left", ["a link to a file outside", "a named pipe"])
def test_the_agents_logs_are_read_only_as_regular_files(
    bouncing_task: Path, tmp_path: Path, name: str, left: str
) -> None:
    """Its usage, cost and redaction count come from its own logs, and its record names a log
    only when it is one: never a link the host would read through, never a pipe it waits on."""
    usage = {"input_tokens": 777_777, "output_tokens": 777_777}
    decoy = _decoy(tmp_path, text=json.dumps({"type": "turn.completed", "usage": usage,
                                              "redactions": 777_777, "note": MARKER}))  # fmt: skip
    log = tmp_path / "work" / "agent" / name

    def leave(workspace: Path, logs: Path) -> None:
        _program(workspace / "solution")
        log.unlink(missing_ok=True)
        if left == "a named pipe":
            os.mkfifo(log)
        else:
            log.symlink_to(decoy)

    outcome = _build(bouncing_task, tmp_path, leave, pipes=[log])
    assert outcome.agent_record is not None
    text = outcome.agent_record.read_text(encoding="utf-8")
    record = json.loads(text)
    assert MARKER not in text
    assert record.get("usage") is None and record.get("redactions", 0) == 0
    field = {"trajectory.jsonl": "trajectory", "stderr.log": "stderr",
             "last_message.md": "last_message"}.get(name)  # fmt: skip
    if field is not None:
        assert record.get(field) is None  # a record never names a link as its log
    assert decoy.read_text(encoding="utf-8").startswith("{")  # unchanged


def test_a_summary_the_agent_rewrote_never_stops_the_build(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """``bootstrap.json`` is in the agent's log directory too (an agent can leave its own there,
    read-only, so the bootstrap cannot replace it): a redaction count in it that is not one is
    no count, never an exception that stops the build."""

    def leave(workspace: Path, logs: Path) -> None:
        _program(workspace / "solution")
        (logs / "bootstrap.json").write_text('{"redactions": "all of them"}', encoding="utf-8")

    outcome = _build(bouncing_task, tmp_path, leave)
    assert outcome.delivered and outcome.agent_record is not None
    assert json.loads(outcome.agent_record.read_text(encoding="utf-8")).get("redactions", 0) == 0


# --- the workspace: the solution and the session output ----------------------------------------


def test_a_solution_that_is_a_link_is_not_delivered(bouncing_task: Path, tmp_path: Path) -> None:
    """The agent's ``solution/`` replaced by a link to a program elsewhere: nothing is
    delivered, and the program behind the link is never copied."""
    program = _program(tmp_path / "decoys" / "program")

    def leave(workspace: Path, logs: Path) -> None:
        (workspace / "solution").rmdir()
        (workspace / "solution").symlink_to(program, target_is_directory=True)

    outcome = _build(bouncing_task, tmp_path, leave)
    assert outcome.twin is None and not (tmp_path / "twin").exists()
    assert "symbolic link" in str(outcome.failure)
    target = tmp_path / "copy"
    with pytest.raises(DeliveryError, match="symbolic link"):
        clean_solution(tmp_path / "work" / "workspace" / "solution", target)
    assert not target.exists()


@pytest.mark.parametrize("left", ["the output a link", "a link in it", "a named pipe in it"])
def test_a_session_output_the_lab_may_not_read_as_it_is_is_not_used(
    bouncing_task: Path, tmp_path: Path, left: str
) -> None:
    """A session output that is a link, or holds a link or a pipe, is not the build's session
    output: nothing downstream (the session check, an official scorer) is given it."""
    decoys = _decoy(tmp_path, "tracks.npz").parent

    def leave(workspace: Path, logs: Path) -> None:
        _program(workspace / "solution")
        output = workspace / "output"
        if left == "the output a link":
            output.symlink_to(decoys, target_is_directory=True)
            return
        output.mkdir()
        if left == "a link in it":
            (output / "tracks.npz").symlink_to(decoys / "tracks.npz")
        else:
            os.mkfifo(output / "tracks.npz")

    outcome = _build(bouncing_task, tmp_path, leave)
    assert outcome.delivered and outcome.session_output is None


@pytest.mark.slow
@pytest.mark.parametrize("left", ["the output a link", "a link in it"])
def test_the_official_scorer_is_never_given_what_the_agent_linked(
    task_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, left: str
) -> None:
    """A native task scored on the session output (``rerun = false``): the official scorer
    reads that output as it is, so one with a link is never handed to it; the task is scored
    as one with no output (the method's fault), not through the link."""
    from real2simlab.orchestrate import experiment as experiment_module

    data = read_task_config(task_copy / "task.toml").to_json()
    data["scoring"] = {
        "protocol": "native",
        "native": {"scorer": "fourdcodebench", "options": {"kind": "synthetic", "case": "B_30"}},
    }
    write_task_config(TaskConfig.model_validate(data), task_copy / "task.toml")
    given: list[Path | None] = []

    class Recording:
        name: ClassVar[str] = "recording"

        def problems(self) -> list[str]:
            return []

        def score(self, task: Any, **kwargs: Any) -> dict[str, Any]:
            given.append(kwargs["output_dir"])
            return {"status": "scored", "value": 0.5}

        def headline(self, report: Any) -> Headline:
            return Headline(metric="m", value=report.get("value"), range=(0.0, 1.0),
                            source="official")  # fmt: skip

    monkeypatch.setattr(experiment_module, "get_scorer", lambda name, **options: Recording())
    decoys = _decoy(tmp_path, "render.mp4").parent

    def leave(workspace: Path, logs: Path) -> None:
        _program(workspace / "solution")
        if left == "the output a link":
            (workspace / "output").symlink_to(decoys, target_is_directory=True)
        else:
            (workspace / "output").mkdir()
            (workspace / "output" / "render.mp4").symlink_to(decoys / "render.mp4")

    config = parse_experiment(
        {
            "name": "leftovers",
            "tasks": {"paths": [str(task_copy)]},
            "methods": [{"name": "fake_agent"}],
            "run": {"backend": "local", "retries": 0},
        }
    )
    experiment = Experiment(
        config, tmp_path / "run", backend=LeavesBehind(leave), log=lambda line: None, report=False
    )
    results = experiment.run()
    assert given == [None]
    (row,) = results.to_json()["rows"]
    assert row["infra_error"] is None  # scored, as a delivery with no output


# --- the session check and the reward: a session output that cannot be read --------------------


@pytest.mark.parametrize(
    "damage",
    ["an empty archive", "JSON nested too deep", "an unreadable file", "a link in it",
     "a named pipe in it", "the output a link"],
)  # fmt: skip
def test_a_session_output_that_cannot_be_read_never_stops_the_scoring(
    bouncing_task: Path, tmp_path: Path, damage: str
) -> None:
    """The session output is the method's own: one that does not read as its kind does not
    match its frozen run (``session_consistent = false``); one the Lab may not read as it is (a
    link, a pipe, a file it cannot read) is not compared (``null``). Neither raises."""
    if damage == "an unreadable file" and os.geteuid() == 0:
        pytest.skip("root reads files without read permission")
    task = load_task(bouncing_task)
    run = tmp_path / "run"
    shutil.copytree(task.reference_dir(), run)  # an output of the task's format
    session = tmp_path / "session"
    shutil.copytree(run, session)
    tracks = session / "tracks.npz"
    if damage == "an empty archive":  # as an agent stopped while it wrote it
        tracks.write_bytes(b"")
    elif damage == "JSON nested too deep":
        (session / "meta.json").write_text("[" * 100_000 + "]" * 100_000, encoding="utf-8")
    elif damage == "an unreadable file":
        tracks.chmod(0)
    elif damage == "a link in it":
        tracks.unlink()
        tracks.symlink_to(run / "tracks.npz")
    elif damage == "a named pipe in it":
        tracks.unlink()
        os.mkfifo(tracks)
    else:
        shutil.rmtree(session)
        session.symlink_to(run, target_is_directory=True)
    result = _in_time(
        lambda: Scorer(task).score(
            method=MethodInfo(name="agent", kind="agent"),
            trial=0,
            seed=0,
            delivered=True,
            run=RunOutput("succeeded", run),
            session_output=session,
        ),
        pipes=[tracks],
    )
    expected = False if damage in ("an empty archive", "JSON nested too deep") else None
    assert result.reproduction.session_consistent is expected
    assert result.infra_error is None and result.summary.reproduction_skill is not None


@pytest.mark.slow
def test_the_harbor_verifier_gives_a_reward_whatever_the_session_output_holds(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """``r2s export harbor``'s verifier always passes the agent's ``/app/output``: an archive
    there cut short (empty) is a session that does not match, and the solution still gets its
    reward; it is never "the verifier could not score"."""
    from real2simlab.bridges import harbor_verify

    solution = _program(tmp_path / "solution")
    first = score_solution(
        bouncing_task, solution, backend=LocalBackend(quiet=True), work_dir=tmp_path / "w"
    )
    session = tmp_path / "session"
    shutil.copytree(tmp_path / "w" / "runs" / "run" / "output", session)
    (session / "tracks.npz").write_bytes(b"")
    logs = tmp_path / "logs"
    code = harbor_verify.main(
        ["--task", str(bouncing_task), "--solution", str(solution), "--session-output",
         str(session), "--logs", str(logs), "--no-isolation"]
    )  # fmt: skip
    assert code == 0, (logs / "error.txt").read_text() if (logs / "error.txt").exists() else ""
    reward = json.loads((logs / "reward.json").read_text())
    assert reward["reward"] == pytest.approx(first.reward)
    result = json.loads((logs / "result.json").read_text())
    assert result["reproduction"]["session_consistent"] is False


@pytest.mark.parametrize("linked", ["the solution", "the package's program"])
def test_a_reward_is_never_given_to_a_program_behind_a_link(
    bouncing_task: Path, tmp_path: Path, linked: str
) -> None:
    """``score_solution`` (``r2s train reward``, the Harbor verifier's ``/app/solution``) takes
    the solution as it is: a link in its place is not delivered and scores 0, and the program
    behind it never runs."""
    program = _program(tmp_path / "decoys" / "program")
    solution = tmp_path / "solution"
    if linked == "the solution":
        solution.symlink_to(program, target_is_directory=True)
    else:
        solution.mkdir()
        (solution / "twin.json").write_text("{}", encoding="utf-8")
        (solution / "program").symlink_to(program, target_is_directory=True)
    scored = score_solution(bouncing_task, solution, backend=LocalBackend(quiet=True))
    assert scored.reward == 0.0 and not scored.result.delivered
    assert "symbolic link" in str(scored.result.delivery_failure)


# --- exports and budgets ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def agent_run(bouncing_task: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A finished run of the scripted agent (no model) on the bouncing-ball task."""
    out = tmp_path_factory.mktemp("agent-run") / "run"
    config = parse_experiment(
        {
            "name": "leftovers",
            "tasks": {"paths": [str(bouncing_task)]},
            "methods": [{"name": "fake_agent"}],
            "run": {"backend": "local", "retries": 0},
        }
    )
    Experiment(config, out, backend=LocalBackend(quiet=True), log=lambda line: None,
               report=False).run()  # fmt: skip
    return out


def _agent_dirs(run: Path) -> tuple[Path, Path]:
    """The agent's log directory and workspace in the run directory ``run``."""
    (build,) = (run / "builds").glob("*/fake_agent/trial-0")
    return build / "work" / "agent", build / "work" / "workspace"


def _exported(dataset: Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(dataset.rglob("*")) if p.is_file())


@pytest.mark.slow
@pytest.mark.parametrize("left", ["a link to a file outside", "a named pipe"])
def test_an_export_reads_the_agents_files_only_as_regular_files(
    agent_run: Path, tmp_path: Path, left: str
) -> None:
    """``r2s train export`` copies the agent's statement and trajectory into the dataset: a link
    there (to a file outside every sandbox, or ``/proc/self/environ``, the exporting process's
    own environment) is never read, and a pipe never waited on."""
    run = tmp_path / "run"
    shutil.copytree(agent_run, run, symlinks=True)
    logs, workspace = _agent_dirs(run)
    decoy = _decoy(tmp_path, text=json.dumps({"type": "note", "text": MARKER}))
    names = [logs / "trajectory.jsonl", workspace / "TASK.md"]
    for name in names:
        name.unlink()
        if left == "a named pipe":
            os.mkfifo(name)
        else:
            name.symlink_to(decoy)
    data = tmp_path / "data"
    _in_time(lambda: export_dataset(data, runs=[run], allow_eval_tasks=True), pipes=names)
    (row,) = [json.loads(line) for line in (data / "runs.jsonl").read_text().splitlines()]
    assert row["statement"] is None and "trajectory" not in row["agent"]
    assert MARKER.encode() not in _exported(data)


@pytest.mark.slow
def test_an_export_redacts_what_the_agent_wrote_again(
    agent_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What reaches a dataset from the agent's files is redacted once more: strings shaped like
    keys, and the values of the exporting shell's credential variables (test values here)."""
    run = tmp_path / "run"
    shutil.copytree(agent_run, run, symlinks=True)
    logs, workspace = _agent_dirs(run)
    shaped = "sk-proj-" + "q" * 40
    value = "decoy-credential-" + "z" * 24
    monkeypatch.setenv("R2S_TEST_DECOY_API_KEY", value)
    with (logs / "trajectory.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "note", "text": f"{shaped} {value}"}) + "\n")
    with (workspace / "TASK.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\n{shaped} {value}\n")
    export_dataset(tmp_path / "data", runs=[run], allow_eval_tasks=True)
    exported = _exported(tmp_path / "data")
    assert shaped.encode() not in exported and value.encode() not in exported
    (row,) = [
        json.loads(line) for line in (tmp_path / "data" / "runs.jsonl").read_text().splitlines()
    ]
    assert "[REDACTED]" in row["statement"]
    trajectory = (tmp_path / "data" / row["agent"]["trajectory"]).read_text(encoding="utf-8")
    assert "[REDACTED]" in trajectory and "fake.done" in trajectory  # the rest as it was


def test_an_export_redacts_the_values_of_credential_variables_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The variables whose values an export redacts are named as credentials by a whole word
    and hold something long enough to be one: ``GIT_AUTHOR_NAME`` (AUTH in AUTHOR),
    ``SSH_AUTH_SOCK`` and ``CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`` no longer cut a name, a path
    or a count out of the data (a JSON line that no longer parsed); a token still goes."""
    from real2simlab.agents.redact import Redactor, environment_secrets

    key = "qa-gateway-token-" + "k" * 24
    socket = "/tmp/ssh-qa-agent/agent.1234"
    for name, value in [
        ("ANTHROPIC_AUTH_TOKEN", key),
        ("GIT_AUTHOR_NAME", "Ada Lovelace"),
        ("SSH_AUTH_SOCK", socket),
        ("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "128000"),
    ]:
        monkeypatch.setenv(name, value)  # fmt: skip
    line = json.dumps({"text": f"commit by Ada Lovelace ({key})", "socket": socket,
                       "usage": {"output_tokens": 1280004}})  # fmt: skip
    redacted = Redactor(environment_secrets())(line)
    assert key not in redacted
    record = json.loads(redacted)
    assert record["usage"] == {"output_tokens": 1280004} and record["socket"] == socket
    assert record["text"].startswith("commit by Ada Lovelace (")


@pytest.mark.slow
def test_the_budget_counts_only_the_records_the_lab_wrote(agent_run: Path, tmp_path: Path) -> None:
    """The budget counted again from a run's agent records (its ``budget.json`` gone) reads the
    Lab's records only: a file of that name the agent left in its workspace is not one, and a
    pipe where the Lab writes its record is a run of unknown cost, not a wait."""
    run = tmp_path / "run"
    shutil.copytree(agent_run, run, symlinks=True)
    logs, workspace = _agent_dirs(run)
    record = json.loads((logs / "agent_record.json").read_text(encoding="utf-8"))
    planted = workspace / "notes" / "agent_record.json"
    planted.parent.mkdir()
    planted.write_text(json.dumps({**record, "cost_usd": 1e6}), encoding="utf-8")
    (logs / "agent_record.json").unlink()
    os.mkfifo(logs / "agent_record.json")  # left by an agent stopped before the Lab wrote it
    config = parse_experiment(
        {"name": "b", "tasks": {"paths": ["."]}, "methods": [{"name": "fake_agent"}]}
    )
    experiment = Experiment(config, run, log=lambda line: None, report=False)
    spent = _in_time(experiment._budget_from_records, pipes=[logs / "agent_record.json"])
    assert spent == {"agent_runs": 1, "cost_usd": 0.0, "cost_unknown_runs": 1}


# --- the helpers --------------------------------------------------------------------------------


def test_files_an_agent_could_replace_are_read_and_written_as_they_are(tmp_path: Path) -> None:
    from real2simlab._fs import is_regular, read_regular, write_new
    from real2simlab.formats.base import tree_problems

    decoy = _decoy(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "own.json").write_text('{"spec": "mine"}\n', encoding="utf-8")
    (logs / "link").symlink_to(decoy)
    os.mkfifo(logs / "pipe")
    assert read_regular(logs / "own.json") == b'{"spec": "mine"}\n'
    assert read_regular(logs / "link") is None and read_regular(logs) is None
    assert read_regular(logs / "missing") is None and read_events(logs / "link") == []
    assert _in_time(lambda: read_regular(logs / "pipe"), [logs / "pipe"]) is None
    assert _in_time(lambda: read_events(logs / "pipe"), [logs / "pipe"]) == []
    (logs / "directory" / "inside").mkdir(parents=True)
    for name in ("link", "pipe", "directory", "new"):
        write_new(logs / name, b"the Lab's record\n")
        assert is_regular(logs / name)
        assert (logs / name).read_bytes() == b"the Lab's record\n"
    assert decoy.read_text(encoding="utf-8") == MARKER + "\n"
    outside = decoy.parent
    (tmp_path / "output").symlink_to(outside, target_is_directory=True)
    assert tree_problems(tmp_path / "output") == ["the output directory is a symbolic link"]
    (tmp_path / "real" / "deep").mkdir(parents=True)
    (tmp_path / "real" / "deep" / "x.npz").symlink_to(decoy)
    assert tree_problems(tmp_path / "real") == ["deep/x.npz: symbolic links are not allowed"]
