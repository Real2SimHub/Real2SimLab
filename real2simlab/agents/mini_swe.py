"""``mini_swe``: mini-swe-agent driving any OpenAI-compatible model, commands sandboxed.

The harness runs on the host (``harness_python``, an environment with mini-swe-agent) and is
the only process that talks to the model. Every command the agent issues runs in a sandbox
session of the task's runtime image with no network, seeing only the workspace (never the
hidden directory). The API key stays in the harness's environment.

For tests, ``scripted`` names a JSON list of command strings that a deterministic model
issues in order, so the whole path runs without any model.

The model's endpoint is ``api_base``, or the default of the model's LiteLLM provider. The
harness's environment is built from nothing but the key: the host's ``OPENAI_BASE_URL`` (or
``ANTHROPIC_BASE_URL``, and LiteLLM's ``*_API_BASE`` names) never reaches it, so a method without
``api_base`` for an ``openai/`` (``anthropic/``) model on a host that sets one is refused, as is a
model named without its provider (``gpt-4o``) while any of them is set, rather than its key sent
to the provider's own API.

A run that failed outside the agent is not its result (the build is an ``infra_error``): the
session's sandbox does not start or stops running commands, the harness is killed from outside,
or the model's API cannot be used before the model answered at all (a rejected key, a model that
does not exist, an endpoint that is down, unreachable or rate-limited). A prompt the API refuses
for what it holds (too long for the model's context, refused by its content policy) is the
method's result. Only a run in which the model never answered is given back to the budget and
tried again at once; one that got answers was spent.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveInt, field_validator

from real2simlab.agents.base import (
    WORKSPACE,
    AgentMethod,
    AgentOptions,
    Workspace,
    budget_of,
    effort_not_passed,
    endpoint_set,
    prepare_workspace,
    session_output_of,
)
from real2simlab.agents.bootstrap import OUTPUT_LIMIT, OUTPUT_TAIL
from real2simlab.agents.record import AgentRecord, cost_amount
from real2simlab.agents.redact import Redactor
from real2simlab.envs import image_reference
from real2simlab.errors import SetupError
from real2simlab.methods.base import BuildContext, BuildOutcome, MethodOptions
from real2simlab.sandbox.base import Backend, Limits, Mount, NetworkPolicy, SandboxSpec
from real2simlab.sandbox.process import (
    LOG_LIMIT_BYTES,
    PROBE_TIMEOUT_S,
    file_room,
    read_kept,
    run_captured,
)
from real2simlab.task.instruction import TEMPLATE_VERSION

DRIVER = Path(__file__).with_name("mini_swe_driver.py")
SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

RULES = f"""

## How to work

1. Every response contains exactly one bash code block with one command, in a
   ```mswea_bash_command fence.
2. Each command runs in a new shell in {WORKSPACE}: `cd` and variables do not persist.
   Commands have no network.
3. When your program works, finish with `echo {SUBMIT}` as a command on its own.

Example response:

THOUGHT: Let me look at the capture first.

```mswea_bash_command
cat {WORKSPACE}/visible/capture.json
```
"""


class MiniSweOptions(AgentOptions):
    model: str | None = None
    """LiteLLM model name, e.g. ``hosted_vllm/qwen3-coder`` or ``openai/gpt-5``."""
    api_base: str | None = None
    api_key_env: str | None = "OPENAI_API_KEY"
    """Variable holding the API key, read by the harness only; ``None`` for no key."""
    harness_python: str | None = None
    """Python with mini-swe-agent installed; defaults to this interpreter."""
    model_class: str = "litellm_textbased"
    step_limit: NonNegativeInt = 150
    cost_limit: NonNegativeFloat = 0.0
    command_timeout_s: PositiveInt = 120
    temperature: NonNegativeFloat | None = None
    max_tokens: PositiveInt | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict[str, Any])
    scripted: str | None = None
    """A JSON file with a list of commands, issued by a deterministic model (tests)."""
    network: Literal["allowlist", "open", "none"] = "none"
    reasoning_effort: str | None = None
    """Not passed to the model (give the model's own setting in ``model_kwargs``)."""

    @field_validator("reasoning_effort")
    @classmethod
    def _effort(cls, value: str | None) -> str | None:
        return effort_not_passed("mini-swe-agent", value, through="model_kwargs")


PROVIDER_ENDPOINTS = {
    "openai": ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
    "anthropic": ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_BASE"),
}
"""The host's variables that send a LiteLLM provider's requests elsewhere, which the harness
never sees."""


UNUSABLE_STATUSES = frozenset({401, 402, 403, 404, 408, 429})
"""HTTP statuses of a model API that cannot be used here, whatever it is asked: the key rejected
or not allowed (401, 403), an account without credit (402), no such model or endpoint (404), a
request that timed out (408), a rate limit (429). Every server error (5xx) is one too."""
UNREACHABLE = frozenset(
    {"APIConnectionError", "APITimeoutError", "Timeout", "ConnectionError", "TimeoutError"}
)
"""Errors of an API that could not be reached at all (no HTTP status)."""
REFUSED_REQUEST = frozenset({"BadRequestError", "UnprocessableEntityError"})
"""An API that refused this request for what it holds: a prompt too long for the model's
context (``ContextWindowExceededError``), one its content policy refuses
(``ContentPolicyViolationError``), a parameter it does not take. The method's own result."""


def model_unusable(summary: dict[str, Any]) -> bool:
    """Whether the error that ended the harness's run says that the model's API itself cannot be
    used (:data:`UNUSABLE_STATUSES`, a server error, :data:`UNREACHABLE`), not that it refused
    this request (:data:`REFUSED_REQUEST`), from the error's classes and HTTP status that the
    driver reports."""
    types = {str(name) for name in summary.get("error_types") or []}
    if types & REFUSED_REQUEST:
        return False
    status = summary.get("error_status")
    if isinstance(status, int) and not isinstance(status, bool):
        return status in UNUSABLE_STATUSES or 500 <= status <= 599
    return bool(types & UNREACHABLE)


def harness_failure(exit_code: int | None, summary: dict[str, Any]) -> str | None:
    """Why the harness's run failed outside the method (its result is then not the agent's), or
    ``None``: the harness killed by a signal from outside it; its sandbox stopped running
    commands; or, before the model answered at all, its API could not be used
    (:func:`model_unusable`: a rejected key, a model that does not exist, an endpoint that is
    down, unreachable or rate-limited). A prompt the API refused for what it holds (too long for
    the model's context, refused by its content policy), and how the run ends once the model has
    answered, are the agent's result."""
    if exit_code is not None and exit_code < 0:
        return f"the mini-swe-agent harness was killed by signal {-exit_code} from outside it"
    if summary.get("sandbox_error"):
        return f"the agent's sandbox stopped running commands: {summary['sandbox_error']}"
    if summary.get("error") and summary.get("model_answers") == 0 and model_unusable(summary):
        return f"the agent never got an answer from its model: {summary['error']}"
    return None


def harness_summary(lines: list[str]) -> tuple[dict[str, Any], str | None]:
    """The summary of the run that the driver prints last (whether the model answered, what it
    cost, whether the sandbox failed) from the lines a harness that exited normally printed;
    without one that can be read, how the agent's run went is not known, and the second value
    says why."""
    problem = "it printed nothing"
    if lines:
        try:
            data: Any = json.loads(lines[-1])
        except ValueError as error:
            problem = f"its last line is not JSON ({error})"
        else:
            cost = data.get("cost", 0.0) if isinstance(data, dict) else None
            if isinstance(cost, int | float) and not isinstance(cost, bool):
                return data, None
            problem = "its last line is not the driver's summary (a JSON object with the cost)"
    return {}, (
        f"the summary of the run the mini-swe-agent harness prints last cannot be read ({problem}):"
        " how the agent's run went is not known"
    )


class MiniSweAgent(AgentMethod):
    name: ClassVar[str] = "mini_swe"
    agent_name: ClassVar[str] = "mini-swe-agent"
    summary: ClassVar[str] = "mini-swe-agent on the host, every command in the sandbox"
    agent_image: ClassVar[str | None] = None
    reports_cost: ClassVar[bool | None] = True
    """For a model LiteLLM has a price for; a run of another says nothing of its cost."""
    Options: ClassVar[type[MethodOptions]] = MiniSweOptions

    @property
    def mini_options(self) -> MiniSweOptions:
        assert isinstance(self.options, MiniSweOptions)
        return self.options

    @property
    def label(self) -> str:
        model = self.mini_options.model or ("scripted" if self.mini_options.scripted else None)
        return f"{self.name}:{model}" if model else self.name

    def model_name(self) -> str | None:
        return "scripted" if self.mini_options.scripted else self.mini_options.model

    def harness(self) -> str:
        return self.mini_options.harness_python or sys.executable

    def problems(self, backend: Backend) -> list[str]:
        options = self.mini_options
        if options.model is None and options.scripted is None:
            return ["give the method option model (a LiteLLM model name) or scripted"]
        if options.network != "none":
            return ["mini_swe commands always run offline; the harness alone reaches the model"]
        self._check_endpoint()  # before anything runs, as an error with its fix
        result = subprocess.run(
            [self.harness(), "-c", "import minisweagent, yaml"],
            capture_output=True,
            text=True,
            env={**os.environ, "MSWEA_SILENT_STARTUP": "1"},
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=PROBE_TIMEOUT_S,
        )
        if result.returncode != 0:
            return [
                f"mini-swe-agent is not importable by {self.harness()}: "
                "pip install mini-swe-agent (or set harness_python)"
            ]
        return []

    def _check_endpoint(self) -> None:
        """Refuse (``SetupError``) a model without ``api_base`` whose provider the host sends
        elsewhere (:data:`PROVIDER_ENDPOINTS`): the harness would send it, and its key, to the
        provider's own API. A model without its provider part (``gpt-4o``,
        ``claude-sonnet-4-...``) goes to the provider LiteLLM finds for it, which depends on
        LiteLLM's version and model tables (and ``harness_python`` may hold another): such a
        name is refused while any of these variables is set, its fix naming the provider."""
        options = self.mini_options
        if options.api_base or options.scripted is not None or not options.model:
            return
        if "/" in options.model:
            provider = options.model.split("/", 1)[0].lower()
            variable = endpoint_set(PROVIDER_ENDPOINTS.get(provider, ()))
            where, model = f"{provider}'s own API", ""
        else:
            variable = endpoint_set(
                [name for names in PROVIDER_ENDPOINTS.values() for name in names]
            )
            provider = "anthropic" if (variable or "").startswith("ANTHROPIC") else "openai"
            where = "the own API of the provider LiteLLM finds for it (api.openai.com for gpt-4o)"
            model = f"model={provider}/{options.model},"
        if variable is None:
            return
        sent = options.api_key_env or "its requests"
        raise SetupError(
            f"{variable} is set on this host, but the method has no api_base: the mini-swe-agent "
            f"harness (whose environment holds only the key) would send {options.model!r} and "
            f"{sent} to {where}, not to that endpoint",
            fix=f"give the method that endpoint (--method mini_swe:{model}api_base=${variable}; "
            f'api_base = "..." in an experiment file), or unset {variable}',
        )

    def command(self, context: BuildContext, workspace: Workspace) -> list[str]:  # pragma: no cover
        raise NotImplementedError("mini_swe runs its harness on the host")

    def version(self, backend: Backend, image: str) -> str | None:
        """mini-swe-agent's version, for the record (asked after the agent ran): ``None`` when
        the harness cannot be asked or does not answer, which is no reason to discard the run."""
        try:
            result = subprocess.run(
                [
                    self.harness(),
                    "-c",
                    "import importlib.metadata as m; print(m.version('mini-swe-agent'))",
                ],
                capture_output=True,
                text=True,
                env={**os.environ, "MSWEA_SILENT_STARTUP": "1"},
                check=False,
                stdin=subprocess.DEVNULL,
                timeout=PROBE_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() or None if result.returncode == 0 else None

    def _cost(self, summary: dict[str, Any]) -> float | None:
        """What the run cost, as the harness counted it; ``None`` when that is not known: a
        model LiteLLM has no price for counts as free (``MSWEA_COST_TRACKING=ignore_errors``),
        and a harness that failed said nothing. A scripted run calls no model: it cost $0."""
        if self.mini_options.scripted is not None:
            return 0.0
        # An amount of money only (not NaN, Infinity, negative or too large); $0 is unpriced.
        return cost_amount(summary.get("cost")) or None

    def _model_settings(self) -> dict[str, Any]:
        options = self.mini_options
        if options.scripted is not None:
            commands = json.loads(Path(options.scripted).read_text(encoding="utf-8"))
            outputs = [
                {
                    "role": "assistant",
                    "content": f"THOUGHT: step {index}.\n\n```mswea_bash_command\n{command}\n```",
                    "extra": {"actions": [{"command": command}], "cost": 0.0},
                }
                for index, command in enumerate([*commands, f"echo {SUBMIT}"])
            ]
            return {
                "model_name": "deterministic",
                "model_class": "minisweagent.models.test_models.DeterministicModel",
                "model": {"outputs": outputs, "cost_per_call": 0.0},
                "model_kwargs": {},
            }
        kwargs = dict(options.model_kwargs)
        if options.api_base:
            kwargs["api_base"] = options.api_base
        if options.temperature is not None:
            kwargs["temperature"] = options.temperature
        if options.max_tokens is not None:
            kwargs["max_tokens"] = options.max_tokens
        return {
            "model_name": options.model,
            "model_class": options.model_class,
            "model": {},
            "model_kwargs": kwargs,
        }

    def build(self, context: BuildContext) -> BuildOutcome:
        options = self.mini_options
        agent_dir = context.work_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        budget = float(budget_of(options, context.config.delivery.agent_time_budget_s))
        workspace = prepare_workspace(context, context.work_dir / "workspace", budget_s=budget)
        image = image_reference(context.config.delivery.environment)
        resolved = context.backend.resolve_image(image)
        spec = SandboxSpec(
            image=resolved.reference,
            argv=(),
            workdir=WORKSPACE,
            mounts=(Mount(workspace.root.resolve(), WORKSPACE, read_only=False),),
            env={"HOME": "/tmp", "TMPDIR": "/tmp"},
            network=NetworkPolicy(),
            limits=Limits(
                timeout_s=options.command_timeout_s, cpus=options.cpus, memory_mb=options.memory_mb
            ),
            label="r2s-mini-swe",
        )
        secret = os.environ.get(options.api_key_env) if options.api_key_env else None
        redactor = Redactor([secret] if secret else [])  # for all the harness says
        harness_home = Path(tempfile.mkdtemp(prefix="r2s-mswea-"))
        started = datetime.now(UTC)
        summary: dict[str, Any] = {}
        unreadable: str | None = None  # why the harness's summary cannot be read
        exit_code: int | None = None
        status = "failed"
        stderr = ""
        try:
            with context.backend.session(spec) as session:
                config = {
                    **self._model_settings(),
                    "api_key_env": options.api_key_env,
                    "task": workspace.statement,
                    "instance_template": "{{task}}\n{% raw %}" + RULES + "{% endraw %}",
                    "sandbox_prefix": list(session.prefix),
                    "sandbox_env": dict(session.env),
                    "host_cwd": str(session.cwd),
                    "command_timeout_s": options.command_timeout_s,
                    "step_limit": options.step_limit,
                    "cost_limit": options.cost_limit,
                    "wall_time_limit_s": int(budget),
                    "trajectory": str(harness_home / "trajectory.json"),
                }
                (harness_home / "config.json").write_text(json.dumps(config), encoding="utf-8")
                env = {
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "HOME": str(harness_home),
                    "MSWEA_GLOBAL_CONFIG_DIR": str(harness_home),
                    "MSWEA_SILENT_STARTUP": "1",
                    "MSWEA_COST_TRACKING": "ignore_errors",
                    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                }
                if options.api_key_env and secret:
                    env[options.api_key_env] = secret
                for name in (
                    "HTTPS_PROXY",
                    "HTTP_PROXY",
                    "NO_PROXY",
                    "https_proxy",
                    "http_proxy",
                    "no_proxy",
                ):
                    if name in os.environ:
                        env[name] = os.environ[name]
                try:
                    result = run_captured(  # stopped with the experiment (Ctrl-C)
                        [self.harness(), str(DRIVER), str(harness_home / "config.json")],
                        env=env,
                        # once the harness's process exists: said in agent/ (harness_starts)
                        started=lambda: self.harness_starts(agent_dir),
                        timeout_s=budget + 300,
                        grace_s=30,
                        keep=file_room(LOG_LIMIT_BYTES),  # what it prints, as a sandbox's log
                    )
                    exit_code = result.returncode
                    lines = result.stdout.strip().splitlines()
                    if exit_code == 0:  # the driver always ends by printing its summary
                        summary, unreadable = harness_summary(lines)
                    status = "completed" if exit_code == 0 else "failed"
                    stderr = result.stderr
                except subprocess.TimeoutExpired:
                    status, stderr = "timed_out", "the harness did not stop"
            (agent_dir / "stderr.log").write_text(redactor(stderr), encoding="utf-8")
            trajectory = harness_home / "trajectory.json"
            if trajectory.is_file():  # within the limit of an agent's trajectory, whole lines
                kept = read_kept(trajectory, file_room(OUTPUT_LIMIT), OUTPUT_TAIL)
                (agent_dir / "trajectory.json").write_text(
                    redactor(kept.decode("utf-8", errors="replace")), encoding="utf-8"
                )
        except SetupError:
            raise
        finally:
            shutil.rmtree(harness_home, ignore_errors=True)
        duration = (datetime.now(UTC) - started).total_seconds()
        outside = harness_failure(exit_code, summary) or unreadable
        if outside is not None:  # nothing it left is the agent's result: never delivered
            # Nothing of the agent ran only if its model never answered: the run is given back
            # and tried again. One that got answers (or whose harness was killed, or whose
            # summary cannot be read: how far it got unknown) was spent: counted, and not run
            # again by this command.
            ran = (
                (exit_code is not None and exit_code < 0)
                or bool(summary.get("model_answers"))
                or unreadable is not None
            )
            status = "failed" if ran else "infra_error"
            outside = redactor(outside)
        twin, failure, infra, scrubbed = self.collect_clean(
            context, workspace, status, [secret] if secret else [], outside
        )
        harness_status = str(summary.get("exit_status", ""))
        if failure and harness_status and outside is None:
            # The harness's own words: an API's error can repeat the key it was sent.
            failure = f"{failure} (harness: {redactor(harness_status)})"

        def clean(name: str) -> str | None:
            """``agent/<name>`` for the record, if it is there and holds no injected value:
            the report and an exported dataset show the log the record names."""
            relative = f"agent/{name}"
            there = (agent_dir / name).is_file() and relative not in scrubbed.left
            return relative if there else None

        record = AgentRecord(
            method=self.name,
            agent=self.agent_name,
            agent_version=self.version(context.backend, resolved.reference),
            model=self.model_name(),
            instruction_template=TEMPLATE_VERSION,
            instruction_sha256=hashlib.sha256(workspace.statement.encode()).hexdigest(),
            task_id=context.task_id,
            task_hash=context.task_hash,
            trial=context.trial,
            image={"reference": resolved.reference, "digest": resolved.digest},
            backend=context.backend.info(),
            network={"mode": "none", "allowlist": [], "harness": "host"},
            started_at=started,
            duration_s=round(duration, 1),
            time_budget_s=budget,
            exit_code=exit_code,
            exit_status=status,
            trajectory=clean("trajectory.json"),  # one JSON document: mini-swe-agent's own
            stderr=clean("stderr.log"),
            usage=None,
            cost_usd=self._cost(summary),
            redactions=redactor.count + scrubbed.redactions,
            delivered=twin is not None,
            failure=failure,
        )
        (agent_dir / "agent_record.json").write_text(record.dumps(), encoding="utf-8")
        session_output = session_output_of(context, workspace.output)  # the agent wrote it
        return BuildOutcome(
            twin, failure, agent_dir / "agent_record.json", session_output, "none",
            infra_error=infra,
        )  # fmt: skip
