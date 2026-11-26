"""``harbor``: run any agent Harbor supports, then freeze and score its twin with the Lab.

The build's workspace (visible data, documents, the SDK, the task statement) is written as a
throwaway Harbor task whose environment uses the task's runtime image and whose workspace is
``/workspace``. ``harbor run`` (from ``harbor`` — Harbor needs Python >= 3.12, so it usually
lives in its own environment) runs the chosen agent with verification disabled; Harbor
collects ``/workspace/solution`` and ``/workspace/output``. The Lab then packages the solution
and re-runs it frozen in its own sandbox, exactly as for its other agents. The hidden data
never enter the Harbor task.

Harbor's network policy applies to the agent (an allowlist with the inference host).

Harbor that produced no trial (no Docker for its environment, an environment it could not
build), or whose trial ended before the agent started, did nothing of the agent's: that build
is an ``infra_error``, not a result.

Credentials reach the agent through ``agent_env`` only, by name: ``{KEY = "$NAME"}`` is passed
as ``--ae KEY=${NAME}``, which Harbor resolves from the environment it runs in, so the value
never appears on a command line (which every user of the machine can read) or in a record.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import tomli_w
from pydantic import Field

from real2simlab._fs import read_regular
from real2simlab.agents.base import (
    WORKSPACE,
    AgentMethod,
    AgentOptions,
    Workspace,
    _read_json,  # pyright: ignore[reportPrivateUsage]
    budget_of,
    prepare_workspace,
    session_output_of,
)
from real2simlab.agents.bootstrap import OUTPUT_LIMIT, OUTPUT_TAIL
from real2simlab.agents.record import AgentRecord, Usage
from real2simlab.agents.redact import Redactor
from real2simlab.envs import image_reference
from real2simlab.errors import SetupError
from real2simlab.methods.base import BuildContext, BuildOutcome, MethodOptions
from real2simlab.sandbox.base import Backend
from real2simlab.sandbox.process import LOG_LIMIT_BYTES, file_room, read_kept, run_captured
from real2simlab.task.instruction import TEMPLATE_VERSION, with_time_budget

NO_VERIFIER = "#!/bin/bash\nmkdir -p /logs/verifier && echo 0 > /logs/verifier/reward.txt\n"
_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def variable(value: str) -> str | None:
    """The environment variable an ``agent_env`` value names (``$NAME`` or ``${NAME}``), or
    ``None`` for a literal value."""
    match = _REFERENCE.fullmatch(value)
    return (match.group(1) or match.group(2)) if match else None


def harbor_failure(
    exit_code: int | None, trial: Path | None, log: str, redact: Callable[[str], str]
) -> str | None:
    """Why Harbor's run failed outside the method (its result is then not the agent's), or
    ``None``: Harbor killed by a signal from outside it; no trial at all (Harbor could not
    start: no Docker for its environment, an environment it could not build); or a trial whose
    agent never started (Harbor's ``agent_execution`` is empty: its environment or the agent's
    setup failed). Once the agent has started, how its trial ends is its result. What Harbor
    said (its ``log``, its trial's exception) can repeat the agent's keys: it is ``redact``-ed
    before it is cut short and recorded."""
    if exit_code is not None and exit_code < 0:
        return f"Harbor was killed by signal {-exit_code} from outside it"
    if trial is None:
        lines = [line.strip() for line in redact(log).splitlines() if line.strip()]
        said = lines[-1][-300:] if lines else "it wrote nothing"
        return f"Harbor produced no trial, so the agent never ran: {said} (agent/harbor.log)"
    result = _read_json(trial / "result.json")
    exception = result.get("exception_info") if result is not None else None
    if result is None or not isinstance(exception, dict):
        return None
    execution = result.get("agent_execution")
    if isinstance(execution, dict) and execution.get("started_at"):
        return None
    kind = redact(str(exception.get("exception_type") or "an error"))[:100]
    message = redact(str(exception.get("exception_message") or ""))[:300]
    return f"Harbor could not set up the agent, so it never ran: {kind}: {message}"


MIN_KEPT_BYTES = 4096
"""Less room left than this: a file of Harbor's trial is left out rather than cut to a note."""
LEFT_OUT_LINES = 100
"""How many of the files left out of the copy of Harbor's trial ``harbor.log`` names."""


def copy_kept(source: Path, target: Path, limit: int | None = None) -> list[str]:
    """Copy the files under ``source`` (what Harbor kept of its trial's agent: its logs, its
    trajectory) to ``target`` within the limits of an agent's trajectory: at most ``limit``
    bytes in all (``OUTPUT_LIMIT``), each file under the file-size limit; a file longer than
    what is left keeps its first and last whole lines (``read_kept``), and once nothing is left
    the other files are left out. So are links and whatever is neither a regular file nor a
    directory: the agent made them, and they could point anywhere on this machine. Says what
    was left out, one line each."""
    limit = OUTPUT_LIMIT if limit is None else limit
    left_out: list[str] = []
    for directory, subdirectories, files in os.walk(source):
        here = Path(directory)
        relative = here.relative_to(source)
        (target / relative).mkdir(parents=True, exist_ok=True)
        for name in sorted(subdirectories):
            if (here / name).is_symlink():  # os.walk does not follow it
                left_out.append(f"{str(relative / name)!r}: a link, not copied")
        subdirectories.sort()
        for name in sorted(files):
            shown = repr(str(relative / name))
            try:
                info = os.lstat(here / name)
                if not stat.S_ISREG(info.st_mode):
                    left_out.append(f"{shown}: not a regular file, not copied")
                    continue
                room = min(limit, file_room(OUTPUT_LIMIT))
                if room < MIN_KEPT_BYTES:
                    left_out.append(f"{shown}: not copied, the trajectory's limit is spent")
                    continue
                data = read_kept(here / name, room, OUTPUT_TAIL)
            except OSError as error:
                left_out.append(f"{shown}: not copied ({error.strerror or error})")
                continue
            (target / relative / name).write_bytes(data)
            limit -= len(data)
    return left_out


def unreadable_result(trial: Path | None) -> str | None:
    """Why Harbor's record of its trial (``result.json``) cannot be read, or ``None`` (no
    trial, no record, or one that is a JSON object). A record cut short (Harbor stopped while it
    wrote it, a disk that failed), not text, not an object or not a regular file (a link or a
    pipe is neither followed nor waited on) leaves unknown how the agent's run went, and whether
    it ran at all: the build is not a result, and its run is spent."""
    path = trial / "result.json" if trial is not None else None
    if path is None or not os.path.lexists(path):
        return None
    try:
        data = read_regular(path)
    except OSError as error:  # this machine failed to read it
        why = error.strerror or str(error)
    else:
        try:
            record: Any = json.loads(data) if data is not None else None
        except (ValueError, RecursionError) as error:  # cut short, not JSON, not text
            why = str(error) or type(error).__name__
        else:
            if isinstance(record, dict):
                return None
            why = "it is not a regular file" if data is None else "it is not a JSON object"
    return (
        f"Harbor's record of its trial ({path.parent.name}/result.json) cannot be read ({why}): "
        "whether and how the agent ran is not known"
    )


def _text(path: Path) -> str:
    """``path``'s text as far as it is text (other bytes replaced); ``""`` when it is not a
    regular file (a link or a pipe is neither followed nor waited on) or cannot be read."""
    try:
        data = read_regular(path)
    except OSError:
        return ""
    return "" if data is None else data.decode("utf-8", errors="replace")


class HarborOptions(AgentOptions):
    agent: str = "terminus-2"
    """A Harbor agent name (``harbor agent list``) or an import path."""
    harbor: str = "harbor"
    """The ``harbor`` command (e.g. ``.venv-harbor/bin/harbor``)."""
    environment_type: str = "docker"
    agent_kwargs: dict[str, str] = Field(default_factory=dict[str, str])
    agent_env: dict[str, str] = Field(default_factory=dict[str, str])
    """Agent environment variables; a value ``$NAME`` (or ``${NAME}``) is read by Harbor from
    this process's environment, so only the name is passed and recorded."""
    solution: str | None = None
    """A program directory written by ``solution/solve.sh`` (for Harbor's ``oracle`` agent)."""
    reasoning_effort: str | None = None


class HarborAgent(AgentMethod):
    name: ClassVar[str] = "harbor"
    agent_name: ClassVar[str] = "harbor"
    summary: ClassVar[str] = "any agent Harbor supports, scored by the Lab"
    Options: ClassVar[type[MethodOptions]] = HarborOptions

    @property
    def harbor_options(self) -> HarborOptions:
        assert isinstance(self.options, HarborOptions)
        return self.options

    @property
    def label(self) -> str:
        options = self.harbor_options
        return f"harbor:{options.agent}" + (f":{options.model}" if options.model else "")

    def problems(self, backend: Backend) -> list[str]:
        self._require_set()  # before anything runs, as an error with its fix
        self._solution_program()
        try:
            result = subprocess.run(
                [self.harbor_options.harbor, "--version"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return [
                f"cannot run Harbor ({self.harbor_options.harbor}): {error}; set the option harbor"
            ]
        if result.returncode != 0:
            return [f"`{self.harbor_options.harbor} --version` failed: {result.stderr.strip()}"]
        return []

    def _require_set(self) -> None:
        """Raise ``SetupError`` when an ``agent_env`` entry reads a variable that is not set
        (or is empty) here: Harbor would give the agent an empty key."""
        for key, value in self.harbor_options.agent_env.items():
            name = variable(value)
            if name is not None and not os.environ.get(name):
                raise SetupError(
                    f"the harbor method's agent_env {key} reads {name}, which is not set",
                    fix=f"export {name}=... (the agent's key; only its name is passed to Harbor)",
                )

    def _solution_program(self) -> Path | None:
        """The program of the ``solution`` option, checked (``SetupError`` with its fix): a
        path that is not there, a file or a program without ``run.py`` gave Harbor's oracle
        agent a ``solve.sh`` that recreates nothing, and the method a 0."""
        from real2simlab.bridges.harbor import ENTRY, solution_program

        if not self.harbor_options.solution:
            return None
        return solution_program(
            Path(self.harbor_options.solution),
            given="the harbor method's solution",
            fix=f"give solution= a program directory with {ENTRY} (Harbor's oracle agent "
            "writes it with solution/solve.sh), or leave the option out",
        )

    def command(self, context: BuildContext, workspace: Workspace) -> list[str]:  # pragma: no cover
        raise NotImplementedError("the harbor method runs Harbor on the host")

    def version(self, backend: Backend, image: str) -> str | None:
        """Harbor's version, for the record (asked after the agent ran): ``None`` when Harbor
        cannot be asked or does not answer, which is no reason to discard the agent's run."""
        try:
            result = subprocess.run(
                [self.harbor_options.harbor, "--version"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return f"harbor {result.stdout.strip()}" if result.returncode == 0 else None

    def _task(self, context: BuildContext, root: Path) -> Path:
        options = self.harbor_options
        task = root / "task"
        budget = float(budget_of(options, context.config.delivery.agent_time_budget_s))
        statement = prepare_workspace(
            context, task / "environment" / "workspace", budget_s=budget
        ).statement
        (task / "environment" / "Dockerfile").write_text(
            f"FROM {image_reference(context.config.delivery.environment)}\n"
            f"WORKDIR {WORKSPACE}\n"
            f"COPY workspace/ {WORKSPACE}/\n"
            f"RUN mkdir -p {WORKSPACE}/solution {WORKSPACE}/output && chmod -R a+rwX {WORKSPACE}\n"
        )
        (task / "instruction.md").write_text(statement, encoding="utf-8")
        config: dict[str, Any] = {
            "schema_version": "1.4",
            "artifacts": [f"{WORKSPACE}/solution", f"{WORKSPACE}/output"],
            "task": {"name": "real2simlab/" + context.task_id.replace("/", "-").replace(".", "-")},
            "agent": {"timeout_sec": budget},
            "environment": {
                "build_timeout_sec": 1800.0,
                "cpus": max(1, int(options.cpus)),
                "memory_mb": options.memory_mb,
                "network_mode": "public" if options.network == "open" else "allowlist",
            },
        }
        (task / "task.toml").write_text(tomli_w.dumps(config), encoding="utf-8")
        tests = task / "tests"
        tests.mkdir()
        (tests / "test.sh").write_text(NO_VERIFIER)
        (tests / "test.sh").chmod(0o755)
        program = self._solution_program()
        if program is not None:
            from real2simlab.bridges.harbor import _solve_sh  # pyright: ignore[reportPrivateUsage]

            (task / "solution").mkdir()
            script = _solve_sh(program, app=WORKSPACE)
            (task / "solution" / "solve.sh").write_text(
                script, encoding="utf-8", errors="surrogateescape"
            )
            (task / "solution" / "solve.sh").chmod(0o755)
        return task

    def _environment(self, shim_dir: Path) -> tuple[dict[str, str], list[str]]:
        env = dict(os.environ)
        docker = os.environ.get("R2S_DOCKER")
        if docker:  # a docker wrapper: Harbor calls `docker`, so put it first on PATH as that
            wrapper = Path(shutil.which(docker) or docker).resolve()
            shim_dir.mkdir(parents=True, exist_ok=True)
            shim = shim_dir / "docker"
            shim.unlink(missing_ok=True)
            shim.symlink_to(wrapper)
            env["PATH"] = os.pathsep.join([str(shim_dir), env.get("PATH", "")])
        secrets: list[str] = []
        for value in self.harbor_options.agent_env.values():
            name = variable(value)
            if name is not None and os.environ.get(name):
                secrets.append(os.environ[name])
        return env, secrets

    def build(self, context: BuildContext) -> BuildOutcome:
        options = self.harbor_options
        self._require_set()
        agent_dir = context.work_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        task = self._task(context, context.work_dir / "harbor")
        jobs = context.work_dir / "harbor" / "jobs"
        command = [
            options.harbor, "run", "-p", str(task), "-a", options.agent, "-o", str(jobs),
            "--job-name", "build", "-e", options.environment_type,
            "--disable-verification", "-y", "-q",
        ]  # fmt: skip
        if options.model:
            command += ["-m", options.model]
        if options.reasoning_effort:
            command += ["--effort", options.reasoning_effort]
        for key, value in options.agent_kwargs.items():
            command += ["--ak", f"{key}={value}"]
        env, secrets = self._environment(context.work_dir / "harbor" / "bin")
        for key, value in options.agent_env.items():
            name = variable(value)
            # A reference stays one: Harbor resolves ${NAME} from the environment it is given.
            command += ["--ae", f"{key}={value}" if name is None else f"{key}=${{{name}}}"]
        budget = float(budget_of(options, context.config.delivery.agent_time_budget_s))
        started = datetime.now(UTC)
        try:
            # Stopped with the experiment (Ctrl-C): asked first, so Harbor removes its containers.
            # What it prints is kept within a sandbox log's limit (both streams, in harbor.log).
            # Once Harbor's process exists, the agent may run: said in agent/ (harness_starts).
            keep = file_room(LOG_LIMIT_BYTES) // 2
            result = run_captured(
                command,
                started=lambda: self.harness_starts(agent_dir),
                env=env,
                timeout_s=budget + 3600,
                grace_s=30,
                keep=keep,
            )
            status = "completed" if result.returncode == 0 else "failed"
            output = result.stdout + result.stderr
            exit_code: int | None = result.returncode
        except subprocess.TimeoutExpired as error:
            status, exit_code = "timed_out", None
            output = str(error.stdout or "")
        redactor = Redactor(secrets)
        said = redactor(output)  # redacted once: Harbor's log, and the build's reason below
        (agent_dir / "harbor.log").write_text(said, encoding="utf-8")
        trial = next(
            (p for p in sorted((jobs / "build").glob("*")) if (p / "artifacts").is_dir()), None
        )
        root = (
            (trial / "artifacts" / WORKSPACE.lstrip("/")) if trial else context.work_dir / "missing"
        )
        statement = with_time_budget(
            context.statement, context.config.delivery.agent_time_budget_s, budget
        )
        workspace = Workspace(root=root, statement_file=root / "TASK.md", statement=statement)
        if trial is not None:
            # The agent's own files (Harbor's job directory itself is Harbor's): links and pipes
            # are not copied, those below it by copy_kept(), and the directory itself here.
            logs = trial / "agent"
            if logs.is_dir() and not logs.is_symlink():
                left_out = copy_kept(logs, agent_dir / "trial")
                if len(left_out) > LEFT_OUT_LINES:
                    more = len(left_out) - LEFT_OUT_LINES
                    left_out[LEFT_OUT_LINES:] = [f"{more} more files left out"]
                with (agent_dir / "harbor.log").open("a", encoding="utf-8") as log:
                    for line in left_out:
                        log.write(redactor(f"[r2s] agent/trial: {line}") + "\n")
            result_file = trial / "result.json"
            if result_file.is_file():
                (agent_dir / "trial_result.json").write_text(
                    redactor(_text(result_file)), encoding="utf-8"
                )
                if (_read_json(result_file) or {}).get("exception_info"):
                    status = "failed"
        unreadable = unreadable_result(trial)
        outside = harbor_failure(exit_code, trial, said, redactor) or unreadable
        if outside is not None:  # not the agent's result: never delivered
            # No trial, or a trial whose agent never started, ran nothing of the agent: given
            # back and tried again. Harbor killed from outside, or whose record of its trial
            # cannot be read, may have run it: spent (counted, and not run again by this
            # command).
            killed = exit_code is not None and exit_code < 0
            status = "failed" if killed or unreadable else "infra_error"
        # Harbor's own job directory and the agent's logs and solution, before any is packaged
        # (also when Harbor failed around the agent: its job directory may hold the keys).
        twin, failure, infra, scrubbed = self.collect_clean(
            context, workspace, status, secrets, outside
        )
        record = AgentRecord(
            method=self.name,
            agent=f"harbor:{options.agent}",
            agent_version=self.version(context.backend, ""),
            model=options.model,
            reasoning_effort=options.reasoning_effort,
            instruction_template=TEMPLATE_VERSION,
            instruction_sha256=hashlib.sha256(statement.encode()).hexdigest(),
            task_id=context.task_id,
            task_hash=context.task_hash,
            trial=context.trial,
            image={
                "reference": image_reference(context.config.delivery.environment),
                "digest": None,
            },
            backend={"backend": f"harbor:{options.environment_type}"},
            network={"mode": options.network, "allowlist": ["(Harbor: inference host)"]},
            started_at=started,
            duration_s=round((datetime.now(UTC) - started).total_seconds(), 1),
            time_budget_s=budget,
            exit_code=exit_code,
            exit_status=status,
            trajectory="agent/trial" if (agent_dir / "trial").is_dir() else None,
            stderr="agent/harbor.log",
            usage=Usage(),
            redactions=redactor.count + scrubbed.redactions,
            delivered=twin is not None,
            failure=failure,
        )
        (agent_dir / "agent_record.json").write_text(record.dumps(), encoding="utf-8")
        session = session_output_of(context, workspace.output)  # the agent wrote it
        return BuildOutcome(
            twin, failure, agent_dir / "agent_record.json", session, options.network,
            infra_error=infra,
        )  # fmt: skip
