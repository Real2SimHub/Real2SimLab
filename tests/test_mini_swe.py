"""mini_swe: the host harness with a deterministic model; every command in a sandbox session."""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from real2simlab.orchestrate import parse_experiment, run_experiment
from real2simlab.sandbox.base import Limits, Mount, SandboxSpec, Session
from real2simlab.sandbox.local import LocalBackend

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("minisweagent") is None, reason="pip install mini-swe-agent"
)

SECRET = "sk-test-mini-swe-" + "s" * 24
PROGRAM = [
    "import r2s_twin, numpy as np",
    "twin = r2s_twin.Twin()",
    "g = 9.81",
    "t = np.arange(60) / 30",
    "z = np.maximum(0.5 - 0.5 * g * t**2, 0.05)",
    "cam = r2s_twin.Camera.look_at((0, -1.6, 0.55), (0, 0, 0.45), width=320, height=240)",
    'twin.write_tracks(fps=30, positions={"ball": np.c_[0*z, 0*z, z]}, camera=cam)',
]


def script(tmp_path: Path) -> Path:
    write = "printf '%s\\n' " + " ".join(f"'{line}'" for line in PROGRAM)
    commands = [
        "env > env.txt",
        "mkdir -p solution && cp tools/r2s_twin.py solution/",
        write + " > solution/run.py",
    ]
    path = tmp_path / "script.json"
    path.write_text(json.dumps(commands))
    return path


def run(tasks: Path, tmp_path: Path, backend: str, backend_object: Any = None) -> Any:
    config = parse_experiment(
        {
            "name": "mini-swe",
            "tasks": {"paths": [str(tasks)], "include": ["synthetic/bouncing_ball/calibrated"]},
            "methods": [
                {
                    "name": "mini_swe",
                    "scripted": str(script(tmp_path)),
                    "api_key_env": "R2S_TEST_KEY",
                }
            ],
            "run": {"backend": backend},
        }
    )
    return run_experiment(
        config, tmp_path / "run", backend=backend_object, log=lambda line: None, report=False
    )


def check(tmp_path: Path, results: Any) -> None:
    summary = results.methods[0]
    assert summary.delivered_rate == 1.0 and summary.reproduction_skill.mean > 0.0
    build = next((tmp_path / "run" / "builds").rglob("mini_swe/trial-0"))
    record = json.loads((build / "work" / "agent" / "agent_record.json").read_text())
    assert record["exit_status"] == "completed" and record["model"] == "scripted"
    trajectory = (build / "work" / "agent" / "trajectory.json").read_text()
    assert "Submitted" in trajectory and SECRET not in trajectory
    env = (build / "work" / "workspace" / "env.txt").read_text()
    assert SECRET not in env and "R2S_TEST_KEY" not in env


def test_mini_swe_on_local(
    synthetic_tasks: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("R2S_TEST_KEY", SECRET)
    check(tmp_path, run(synthetic_tasks, tmp_path, "local"))


@pytest.mark.docker
def test_mini_swe_on_docker(
    synthetic_tasks: Path, tmp_path: Path, docker_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("R2S_TEST_KEY", SECRET)
    check(tmp_path, run(synthetic_tasks, tmp_path, "docker", docker_backend))


@pytest.mark.docker
def test_docker_session_is_offline_and_cleaned_up(tmp_path: Path, docker_backend: Any) -> None:
    import subprocess

    spec = SandboxSpec(
        image="ghcr.io/real2simhub/r2s-python:1.0",
        argv=(),
        workdir="/w",
        mounts=(Mount(tmp_path, "/w", read_only=False),),
        limits=Limits(timeout_s=30),
        label="r2s-test-session",
    )
    with docker_backend.session(spec) as session:
        probe = "python -c \"import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)\""
        offline = subprocess.run(
            [*session.prefix, probe], env=session.env, capture_output=True, check=False
        )
        assert offline.returncode != 0
        made = subprocess.run([*session.prefix, "echo hi > made.txt"], env=session.env, check=False)
        assert made.returncode == 0 and (tmp_path / "made.txt").read_text() == "hi\n"
        name = session.prefix[session.prefix.index("--workdir") + 2]
    listing = subprocess.run(
        [docker_backend.docker, "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert name not in listing.stdout


Reply = Callable[[str], tuple[int, dict[str, Any]]]
"""How a model API answers a request, given the key it was sent: an HTTP status and a body."""


def _error(status: int, message: str, kind: str, code: str) -> tuple[int, dict[str, Any]]:
    return status, {"error": {"message": message, "type": kind, "code": code}}


def rejected_key(key: str) -> tuple[int, dict[str, Any]]:
    """401; its message repeats the key it was sent, as some gateways do."""
    return _error(
        401, f"Incorrect API key provided: {key}", "invalid_request_error", "invalid_api_key"
    )


def unavailable(key: str) -> tuple[int, dict[str, Any]]:
    """503: the provider is down."""
    return _error(503, "The service is temporarily unavailable", "server_error", "unavailable")


def too_long(key: str) -> tuple[int, dict[str, Any]]:
    """400: the prompt does not fit the model's context."""
    message = (
        "This model's maximum context length is 2048 tokens. However, your messages resulted "
        "in 3311 tokens. Please reduce the length of the messages."
    )
    return _error(400, message, "invalid_request_error", "context_length_exceeded")


def refused(key: str) -> tuple[int, dict[str, Any]]:
    """400: the provider's content policy refuses the prompt."""
    message = (
        "Invalid prompt: your prompt was flagged as potentially violating our usage policy. "
        "Please try again with a different prompt."
    )
    return _error(400, message, "invalid_request_error", "invalid_prompt")


def answer(command: str) -> Reply:
    """An answer in the harness's format, with one command."""
    content = f"THOUGHT: let me look.\n\n```mswea_bash_command\n{command}\n```"
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "qa",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return lambda key: (200, body)


@contextmanager
def endpoint(*replies: Reply) -> Iterator[str]:
    """A model API on 127.0.0.1: its n-th request gets ``replies[n]``, and every request after
    the last reply gets the last."""
    served: list[int] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            with lock:
                reply = replies[min(len(served), len(replies) - 1)]
                served.append(1)
            key = (self.headers.get("Authorization") or "").removeprefix("Bearer ")
            status, data = reply(key)
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def unreachable() -> str:
    """A model API address where nothing listens."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


def harness(tmp_path: Path) -> str:
    """This Python as the harness's, with mini-swe-agent trying a failed model request once
    rather than ten times (minutes of back-off): the run is otherwise the same."""
    path = tmp_path / "harness-python"
    path.write_text(
        f'#!/bin/sh\nMSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1 exec "{sys.executable}" "$@"\n'
    )
    path.chmod(0o755)
    return str(path)


def evaluate(
    tasks: Path,
    tmp_path: Path,
    method: dict[str, Any],
    backend: str = "local",
    obj: Any = None,
    *,
    retries: int = 0,
    budget: dict[str, Any] | None = None,
) -> Any:
    config = parse_experiment(
        {
            "name": "mini-swe",
            "tasks": {"paths": [str(tasks)], "include": ["synthetic/bouncing_ball/calibrated"]},
            "methods": [{"name": "mini_swe", "api_key_env": "R2S_TEST_KEY", **method}],
            "run": {"backend": backend, "retries": retries},
            "budget": budget or {},
        }
    )
    return run_experiment(
        config, tmp_path / "run", backend=obj, log=lambda line: None, report=False
    )


def records(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The build record of the run's one build, and its agent record (if one was written)."""
    (path,) = (tmp_path / "run" / "builds").rglob("build_record.json")
    build = json.loads(path.read_text())
    agent = path.parent / build["agent_record"] if build.get("agent_record") else None
    return build, json.loads(agent.read_text()) if agent else None


def spent(tmp_path: Path) -> int:
    return json.loads((tmp_path / "run" / "state" / "budget.json").read_text())["agent_runs"]


@pytest.mark.slow
def test_a_model_api_that_rejects_the_key_is_not_a_score(
    synthetic_tasks: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent never got an answer from its model (the key was rejected): nothing of the
    method worked, so the build is not its 0, the agent run is given back, and the same
    command builds it again."""
    monkeypatch.setenv("R2S_TEST_KEY", SECRET)
    with endpoint(rejected_key) as url:
        results = evaluate(synthetic_tasks, tmp_path, {"model": "openai/qa", "api_base": url})
    assert results.methods[0].infra_errors == 1
    build, agent = records(tmp_path)
    assert "AuthenticationError" in (build["infra_error"] or "")
    assert agent is not None and agent["exit_status"] == "infra_error"
    assert spent(tmp_path) == 0
    for path in (tmp_path / "run").rglob("*"):
        if path.suffix in (".json", ".log", ".jsonl", ".txt"):
            assert SECRET not in path.read_text(errors="replace"), path


@pytest.mark.slow
@pytest.mark.parametrize("api", ["down", "unreachable"])
def test_a_model_api_that_cannot_be_used_is_not_a_score(
    synthetic_tasks: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api: str
) -> None:
    """A provider that is down (503), or an address where nothing answers: the model never
    answered, nothing of the method worked, and the agent run is given back."""
    monkeypatch.setenv("R2S_TEST_KEY", SECRET)
    method = {"model": "openai/qa", "harness_python": harness(tmp_path)}
    if api == "down":
        with endpoint(unavailable) as url:
            results = evaluate(synthetic_tasks, tmp_path, {**method, "api_base": url})
    else:
        results = evaluate(synthetic_tasks, tmp_path, {**method, "api_base": unreachable()})
    assert results.methods[0].infra_errors == 1
    build, agent = records(tmp_path)
    assert "never got an answer from its model" in (build["infra_error"] or "")
    assert agent is not None and agent["exit_status"] == "infra_error"
    assert spent(tmp_path) == 0


@pytest.mark.slow
@pytest.mark.parametrize(
    ("prompt", "error"),
    [(too_long, "ContextWindowExceededError"), (refused, "ContentPolicyViolationError")],
)
def test_a_prompt_the_model_api_refuses_is_the_methods_result(
    synthetic_tasks: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt: Reply,
    error: str,
) -> None:
    """The model's API works but refuses this task's prompt (too long for the model's context,
    refused by the provider's content policy): a limit of the method, so its result (the
    benchmark's lowest), not a failure of the machine that would leave such tasks out of its
    score and redo them on every run."""
    monkeypatch.setenv("R2S_TEST_KEY", SECRET)
    method = {"model": "openai/qa", "harness_python": harness(tmp_path)}
    with endpoint(prompt) as url:
        results = evaluate(synthetic_tasks, tmp_path, {**method, "api_base": url})
    summary = results.methods[0]
    assert summary.infra_errors == 0 and summary.delivered_rate == 0.0
    assert summary.reproduction_skill.mean == 0.0
    build, agent = records(tmp_path)
    assert build.get("infra_error") is None and error in (build["failure"] or "")
    assert agent is not None and agent["exit_status"] == "completed"
    assert spent(tmp_path) == 1


@pytest.mark.slow
def test_a_model_that_answered_and_then_failed_is_the_methods_result(
    synthetic_tasks: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the model has answered, how the run ends is the method's result, also when its API
    then rejects the key; that error, which repeats the key, is recorded redacted."""
    key = "qa-canary-mini-swe-" + "c" * 24  # no key's shape: only its value can redact it
    monkeypatch.setenv("R2S_TEST_KEY", key)
    with endpoint(answer("ls"), rejected_key) as url:
        results = evaluate(synthetic_tasks, tmp_path, {"model": "openai/qa", "api_base": url})
    summary = results.methods[0]
    assert summary.infra_errors == 0 and summary.reproduction_skill.mean == 0.0
    build, agent = records(tmp_path)
    assert build.get("infra_error") is None
    assert "(harness: AuthenticationError" in build["failure"]
    assert agent is not None and agent["exit_status"] == "completed"
    assert spent(tmp_path) == 1
    for path in (tmp_path / "run").rglob("*"):
        if path.is_file():
            assert key not in path.read_text(errors="replace"), path


@pytest.mark.parametrize(
    ("types", "status", "outside"),
    [
        (["AuthenticationError"], 401, True),
        (["APIError"], 402, True),  # payment required: the account has no credit left
        (["PermissionDeniedError"], 403, True),
        (["NotFoundError"], 404, True),
        (["Timeout"], 408, True),
        (["RateLimitError"], 429, True),
        (["InternalServerError"], 500, True),
        (["BadGatewayError"], 502, True),
        (["ServiceUnavailableError"], 503, True),
        (["APIConnectionError"], None, True),
        (["ContextWindowExceededError", "BadRequestError"], 400, False),
        (["ContentPolicyViolationError", "BadRequestError"], 400, False),
        (["UnsupportedParamsError", "BadRequestError"], 400, False),
        (["UnprocessableEntityError"], 422, False),
        (["KeyError"], None, False),
    ],
)
def test_only_a_model_api_that_cannot_be_used_fails_outside_the_method(
    types: list[str], status: int | None, outside: bool
) -> None:
    """Before the model answered at all: an API that cannot be used (its key, the model, the
    endpoint, a rate limit, a server error) is not the method's result; an API that refused
    the request for what it holds is, as is any other error, and any error once it answered."""
    from real2simlab.agents.mini_swe import harness_failure

    summary = {
        "error": f"{types[0]}: it failed",
        "error_types": types,
        "error_status": status,
        "model_answers": 0,
    }
    assert (harness_failure(0, summary) is not None) is outside
    assert harness_failure(0, {**summary, "model_answers": 1}) is None


@pytest.mark.slow
def test_a_command_that_fails_in_a_working_sandbox_is_the_agents_to_see(
    synthetic_tasks: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command that fails on its own while its sandbox works is shown to the model, which
    goes on (and here delivers): only a sandbox that runs no command at all stops the agent."""
    monkeypatch.setenv("R2S_TEST_KEY", SECRET)
    path = script(tmp_path)
    path.write_text(json.dumps(["exit 3", *json.loads(path.read_text())]))
    results = evaluate(synthetic_tasks, tmp_path, {"scripted": str(path)})
    assert results.methods[0].infra_errors == 0 and results.methods[0].delivered_rate == 1.0
    build, agent = records(tmp_path)
    assert build.get("infra_error") is None and agent is not None
    trajectory = (tmp_path / "run" / "builds").rglob("agent/trajectory.json")
    assert "<returncode>3</returncode>" in next(trajectory).read_text()


class StoppingSession(LocalBackend):
    """A session whose sandbox stops running commands after the first ``works`` (its
    container killed, its Docker daemon gone): each later command fails before it runs."""

    def __init__(self, root: Path, works: int) -> None:
        super().__init__(quiet=True)
        self.root, self.works = root, works
        self.sessions = 0

    @contextmanager
    def session(self, spec: SandboxSpec) -> Generator[Session]:
        self.sessions += 1
        count, wrapper = self.root / "count", self.root / "sandbox"
        wrapper.write_text(
            "#!/bin/sh\n"
            f"n=$(($(cat {count} 2>/dev/null || echo 0) + 1)); echo $n > {count}\n"
            f'if [ "$n" -gt {self.works} ]; then\n'
            "  echo 'Error response from daemon: container r2s-mini-swe is not running' >&2\n"
            "  exit 1\n"
            "fi\n"
            'exec "$@"\n'
        )
        wrapper.chmod(0o755)
        with super().session(spec) as session:
            yield replace(session, prefix=(str(wrapper), *session.prefix))


@pytest.mark.slow
def test_a_sandbox_that_stops_running_commands_ends_the_agent_run(
    synthetic_tasks: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the session's sandbox runs no command at all (not even ``true``), the agent is
    stopped instead of being shown that error at every step, and the build is not a result.
    But the agent had worked (its model answered, its first commands ran): the run is spent,
    also when the budget is counted again from the run's records, and the same command does
    not run it again at once."""
    monkeypatch.setenv("R2S_TEST_KEY", SECRET)
    backend = StoppingSession(tmp_path, works=2)
    method = {"scripted": str(script(tmp_path))}
    results = evaluate(synthetic_tasks, tmp_path, method, obj=backend, retries=2)
    assert backend.sessions == 1
    assert results.methods[0].infra_errors == 1
    build, agent = records(tmp_path)
    assert "is not running" in (build["infra_error"] or "")
    assert agent is not None and agent["exit_status"] == "failed"
    assert spent(tmp_path) == 1
    (tmp_path / "run" / "state" / "budget.json").unlink()
    evaluate(
        synthetic_tasks, tmp_path, method, obj=backend, retries=2, budget={"max_agent_runs": 1}
    )
    skipped = json.loads((tmp_path / "run" / "results.json").read_text())["skipped"]
    assert backend.sessions == 1 and any("agent-run budget" in step for step in skipped)
    assert spent(tmp_path) == 1


@pytest.mark.slow
def test_a_bwrap_session_that_cannot_start_is_not_a_score(
    synthetic_tasks: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bwrap that passes its readiness check but cannot set up a sandbox here (no new procfs):
    the session fails as it starts, before the harness asks the model anything."""
    from real2simlab.envs.rootfs import unpack_tarball
    from real2simlab.sandbox.bwrap import BwrapBackend

    from .test_bwrap import IMAGE, NO_PROC, _fake_bwrap, _small_image, readiness_passes

    readiness_passes(monkeypatch)
    monkeypatch.setenv("R2S_TEST_KEY", SECRET)
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    unpack_tarball(_small_image(tmp_path), IMAGE)
    backend = BwrapBackend(_fake_bwrap(tmp_path, "noproc", NO_PROC), cgroup_controllers=frozenset())
    method = {"scripted": str(script(tmp_path))}
    results = evaluate(synthetic_tasks, tmp_path, method, "bwrap", backend)
    assert results.methods[0].infra_errors == 1
    build, agent = records(tmp_path)
    assert "Can't mount proc" in (build["infra_error"] or "") and agent is None
    assert spent(tmp_path) == 0


@pytest.mark.slow
def test_a_model_without_a_price_leaves_what_its_runs_cost_unknown(
    synthetic_tasks: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model LiteLLM has no price for (most self-hosted and gateway models): mini-swe-agent
    counts its calls as free, so the run does not say what it cost. Its record leaves the cost
    out and the budget counts it as a run of unknown cost, never $0: with a cost budget, no
    further agent starts, and the report says the cost is unknown."""
    from real2simlab.agents.mini_swe import SUBMIT
    from real2simlab.report import write_report

    monkeypatch.setenv("R2S_TEST_KEY", SECRET)
    with endpoint(answer(f"echo {SUBMIT}")) as url:
        config = parse_experiment(
            {
                "name": "unpriced",
                "tasks": {
                    "paths": [str(synthetic_tasks)],
                    "include": ["synthetic/bouncing_ball/calibrated"],
                },
                "methods": [
                    {
                        "name": "mini_swe",
                        "api_key_env": "R2S_TEST_KEY",
                        "model": "openai/qa-model-without-a-price",
                        "api_base": url,
                    }
                ],
                "run": {"backend": "local", "trials": 2, "agent_concurrency": 1, "retries": 0},
                "budget": {"max_cost_usd": 1.0},
            }
        )
        run_experiment(config, tmp_path / "run", log=lambda line: None, report=False)
    paths = sorted((tmp_path / "run" / "builds").rglob("agent_record.json"))
    assert len(paths) == 1, f"a second agent run started: {paths}"
    agent = json.loads(paths[0].read_text())
    assert agent["exit_status"] == "completed" and agent.get("cost_usd") is None, agent
    skipped = json.loads((tmp_path / "run" / "results.json").read_text())["skipped"]
    assert any("(trial 1): the cost budget ($1) cannot be checked" in step for step in skipped)
    budget = json.loads((tmp_path / "run" / "state" / "budget.json").read_text())
    budget.pop("wall_time_s", None)  # the wall-time budget, which its own tests check
    assert budget == {"agent_runs": 1, "cost_usd": 0.0, "cost_unknown_runs": 1}
    write_report(tmp_path / "run")
    pages = [page.read_text() for page in (tmp_path / "run" / "report").rglob("*.html")]
    assert any("agent trajectory (redacted)" in page and "cost unknown" in page for page in pages)


def test_a_harness_that_ran_before_this_machine_failed_is_not_run_again(
    synthetic_tasks: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The harness runs the agent from the host and its logs are written only once it ends: a
    build that this machine fails right after the harness ran (a full disk) ran the agent. It is
    counted as a run of unknown cost and not run again by the same command, never given back
    and run again at once."""
    import errno
    import os

    from real2simlab.agents import mini_swe

    real = mini_swe.run_captured
    ran: list[int] = []

    def then_full_disk(*args: Any, **kwargs: Any) -> Any:  # this machine fails once it ran
        ran.append(real(*args, **kwargs).returncode)
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), "agent/stderr.log")

    monkeypatch.setattr(mini_swe, "run_captured", then_full_disk)
    evaluate(synthetic_tasks, tmp_path, {"scripted": str(script(tmp_path))}, retries=2)
    assert len(ran) == 1, ran
    budget = json.loads((tmp_path / "run" / "state" / "budget.json").read_text())
    budget.pop("wall_time_s", None)  # the wall-time budget, which its own tests check
    assert budget == {"agent_runs": 1, "cost_usd": 0.0, "cost_unknown_runs": 1}


@pytest.mark.parametrize("how", ["cannot be spawned", "stopped just before"])
def test_a_harness_that_never_started_gives_its_run_back(
    synthetic_tasks: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """A harness whose process never started, because it cannot be spawned or because the
    experiment was stopped (Ctrl-C) just before, ran nothing of the agent: its reserved run is
    given back, never counted as one of unknown cost; one that cannot be spawned is tried
    again."""
    import errno
    import os
    import subprocess

    from real2simlab.agents import mini_swe
    from real2simlab.sandbox import process

    spawned: list[str] = []
    if how == "cannot be spawned":
        popen = subprocess.Popen

        def cannot_spawn(args: Any, *more: Any, **options: Any) -> Any:
            if str(mini_swe.DRIVER) in [str(arg) for arg in args]:
                spawned.append("refused")  # Popen fails: the harness cannot run
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(args[0]))
            return popen(args, *more, **options)

        monkeypatch.setattr(subprocess, "Popen", cannot_spawn)
    else:
        harness = mini_swe.MiniSweAgent.harness

        def then_stopped(self: mini_swe.MiniSweAgent) -> str:
            found = harness(self)
            process.cancel_all()  # Ctrl-C as the harness is about to start
            return found

        monkeypatch.setattr(mini_swe.MiniSweAgent, "harness", then_stopped)
    try:
        if how == "cannot be spawned":
            evaluate(synthetic_tasks, tmp_path, {"scripted": str(script(tmp_path))}, retries=2)
        else:
            with pytest.raises(KeyboardInterrupt):
                evaluate(synthetic_tasks, tmp_path, {"scripted": str(script(tmp_path))}, retries=2)
    finally:
        process.reset_cancel()
    budget = json.loads((tmp_path / "run" / "state" / "budget.json").read_text())
    budget.pop("wall_time_s", None)  # the wall-time budget, which its own tests check
    assert budget == {"agent_runs": 0, "cost_usd": 0.0}
    assert len(spawned) == (3 if how == "cannot be spawned" else 0)  # given back, tried again


@pytest.mark.slow
def test_mini_swe_shows_and_exports_the_trajectory_its_record_names(
    synthetic_tasks: Path, tmp_path: Path
) -> None:
    """mini-swe-agent's trajectory is one JSON document, ``agent/trajectory.json``, as its
    record says: the report shows it and a dataset exports it, as for any agent. The record
    counts what was redacted from it (strings shaped like keys) and says that a scripted run,
    which calls no model, cost nothing."""
    from real2simlab.report import write_report
    from real2simlab.train import export_dataset

    printed = "sk-proj-" + "Q7" * 20  # shaped like an OpenAI key; made up
    path = tmp_path / "script.json"
    path.write_text(json.dumps([f"echo {printed}", "echo done"]))
    evaluate(synthetic_tasks, tmp_path, {"scripted": str(path)})
    (record,) = (tmp_path / "run" / "builds").rglob("agent_record.json")
    agent = json.loads(record.read_text())
    assert agent["trajectory"] == "agent/trajectory.json", agent
    markers = sum(
        file.read_text(errors="replace").count("[REDACTED]")
        for file in record.parent.iterdir()
        if file.name != "agent_record.json"
    )
    assert agent["redactions"] == markers and markers > 0, (agent["redactions"], markers)
    write_report(tmp_path / "run")
    pages = [page.read_text() for page in (tmp_path / "run" / "report").rglob("*.html")]
    shown = [page for page in pages if "mini_swe, trial 0: agent trajectory (redacted)" in page]
    assert len(shown) == 1 and "[REDACTED]" in shown[0]
    export_dataset(
        tmp_path / "data", runs=[tmp_path / "run"], allow_eval_tasks=True, include_undelivered=True
    )
    (row,) = [json.loads(line) for line in (tmp_path / "data" / "runs.jsonl").open()]
    exported = tmp_path / "data" / row["agent"]["trajectory"]
    assert exported.name == "trajectory.json" and "[REDACTED]" in exported.read_text()
    assert row["agent"]["trajectory_format"] == "mini-swe-agent trajectory (JSON), redacted"
    for file in [*(tmp_path / "run").rglob("*"), *(tmp_path / "data").rglob("*")]:
        if file.is_file():
            assert printed not in file.read_text(errors="replace"), file
    assert agent.get("cost_usd") == 0.0 and "cost $0," in shown[0], agent  # no model was called
