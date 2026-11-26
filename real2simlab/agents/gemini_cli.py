"""``gemini_cli``: Google's Gemini CLI, run non-interactively in a container.

Credentials: ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) from the environment, injected at run
time only. Tool calls are approved automatically inside the container (``--yolo``); the
network is the egress allowlist (``generativelanguage.googleapis.com`` by default). The task
statement is read from stdin. Token usage is the ``stats`` of the stream's ``result`` event.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, cast

from pydantic import field_validator

from real2simlab.agents.base import (
    AgentMethod,
    AgentOptions,
    Credentials,
    CredentialStatus,
    Workspace,
    effort_not_passed,
    env_status,
    read_events,
)
from real2simlab.agents.record import Usage
from real2simlab.errors import SetupError
from real2simlab.methods.base import BuildContext, MethodOptions

KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
DEFAULT_HOSTS = ("generativelanguage.googleapis.com", "oauth2.googleapis.com")


class GeminiCliOptions(AgentOptions):
    output_format: str = "stream-json"

    @field_validator("reasoning_effort")
    @classmethod
    def _effort(cls, value: str | None) -> str | None:
        return effort_not_passed("Gemini CLI", value)


class GeminiCliAgent(AgentMethod):
    name: ClassVar[str] = "gemini_cli"
    agent_name: ClassVar[str] = "gemini-cli"
    summary: ClassVar[str] = "Google Gemini CLI in a container"
    agent_image: ClassVar[str | None] = "r2s-agent-gemini-cli"
    version_command: ClassVar[tuple[str, ...] | None] = ("gemini", "--version")
    default_allowlist: ClassVar[tuple[str, ...]] = DEFAULT_HOSTS
    Options: ClassVar[type[MethodOptions]] = GeminiCliOptions

    def credential_status(self) -> CredentialStatus:
        return env_status(KEY_ENVS)

    def credentials(self, context: BuildContext) -> Credentials:
        for name in KEY_ENVS:
            if os.environ.get(name):
                return Credentials(
                    env={name: os.environ[name]},
                    public_env={"GEMINI_CLI_NO_RELAUNCH": "1"},
                    description=f"{name} from the environment",
                )
        raise SetupError("GEMINI_API_KEY is not set", fix="export GEMINI_API_KEY=...")

    def command(self, context: BuildContext, workspace: Workspace) -> list[str]:
        assert isinstance(self.options, GeminiCliOptions)
        argv = ["gemini", "--yolo", "--output-format", self.options.output_format]
        if self.options.model:
            argv += ["--model", self.options.model]
        return [*argv, *self.options.extra_args]

    def reasoning_effort(self) -> str | None:
        return None  # never passed to Gemini CLI

    def usage(self, trajectory: Path) -> Usage | None:
        """The ``stats`` of the last ``result`` event of the stream-json output: the run's
        ``input_tokens`` (``cached`` is the part of them read from the cache), ``output_tokens``
        and ``total_tokens``. ``None`` without one (another output format, a stopped run)."""
        for event in reversed(read_events(trajectory)):
            stats: Any = event.get("stats")
            if event.get("type") != "result" or not isinstance(stats, dict):
                continue
            counts = cast(dict[str, Any], stats)
            return Usage(
                input_tokens=_tokens(counts, "input_tokens"),
                cached_input_tokens=_tokens(counts, "cached"),
                output_tokens=_tokens(counts, "output_tokens"),
                total_tokens=_tokens(counts, "total_tokens"),
            )
        return None


def _tokens(stats: dict[str, Any], name: str) -> int | None:
    """A count of Gemini CLI's ``stats``: a non-negative integer, else unknown."""
    value = stats.get(name)
    valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
    return value if valid else None
