"""``claude_code``: Anthropic's Claude Code CLI, run non-interactively (``claude -p``).

Credentials, the first one set in the environment, injected at run time only:
``ANTHROPIC_API_KEY``; ``ANTHROPIC_AUTH_TOKEN`` (the token of an Anthropic-compatible endpoint,
given with ``base_url``); ``CLAUDE_CODE_OAUTH_TOKEN`` (a subscription token from
``claude setup-token``). The record names the variable used, never its value.

The endpoint is ``base_url`` (``ANTHROPIC_BASE_URL`` in the sandbox), or ``api.anthropic.com``.
The host's own ``ANTHROPIC_BASE_URL`` is never taken: set while the method has no ``base_url``,
or with ``ANTHROPIC_AUTH_TOKEN`` and no ``base_url``, the credential is the key of an endpoint
the method does not name, and the run is refused (``r2s doctor`` says so too) rather than its
credential sent to ``api.anthropic.com``.

Claude Code's own permission prompts are skipped inside the container (the container is the
sandbox, which ``IS_SANDBOX=1`` tells it, so it also runs when the agent's uid is 0); its
network is the egress allowlist (``api.anthropic.com`` by default). The event stream is
``--output-format stream-json``, whose ``result`` event carries token usage (in the Anthropic
API's terms: ``input_tokens`` leaves out what was read from or written to the prompt cache, which
the record counts as input) and cost.

Claude Code is proprietary software ("All rights reserved"): its image is built by each user
and never published.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import PositiveInt, field_validator

from real2simlab.agents.base import (
    AgentMethod,
    AgentOptions,
    Credentials,
    CredentialStatus,
    Workspace,
    effort_not_passed,
    endpoint_set,
    env_status,
    read_events,
)
from real2simlab.agents.record import Usage, cost_amount
from real2simlab.errors import SetupError
from real2simlab.methods.base import BuildContext, MethodOptions

KEY_ENV = "ANTHROPIC_API_KEY"
KEY_ENVS = (KEY_ENV, "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
"""Credential variables in the order they are tried; exactly one is passed to the sandbox."""
DEFAULT_HOSTS = ("api.anthropic.com", "statsig.anthropic.com", "sentry.io")
ENDPOINT_ENV = "ANTHROPIC_BASE_URL"
GATEWAY_TOKEN = "ANTHROPIC_AUTH_TOKEN"


class ClaudeCodeOptions(AgentOptions):
    max_turns: PositiveInt | None = None
    base_url: str | None = None
    """An Anthropic-compatible endpoint (``ANTHROPIC_BASE_URL``); its host joins the allowlist."""

    @field_validator("reasoning_effort")
    @classmethod
    def _effort(cls, value: str | None) -> str | None:
        return effort_not_passed("Claude Code", value)


class ClaudeCodeAgent(AgentMethod):
    name: ClassVar[str] = "claude_code"
    agent_name: ClassVar[str] = "claude-code"
    summary: ClassVar[str] = "Anthropic Claude Code (claude -p) in a container"
    agent_image: ClassVar[str | None] = "r2s-agent-claude-code"
    version_command: ClassVar[tuple[str, ...] | None] = ("claude", "--version")
    default_allowlist: ClassVar[tuple[str, ...]] = DEFAULT_HOSTS
    Options: ClassVar[type[MethodOptions]] = ClaudeCodeOptions

    @property
    def claude_options(self) -> ClaudeCodeOptions:
        assert isinstance(self.options, ClaudeCodeOptions)
        return self.options

    def allowlist(self) -> tuple[str, ...]:
        if self.claude_options.allowlist:
            return self.claude_options.allowlist
        if self.claude_options.base_url:
            from urllib.parse import urlsplit

            host = urlsplit(self.claude_options.base_url).hostname
            return (host,) if host else ()
        return DEFAULT_HOSTS

    def _endpoint_problem(self, name: str) -> tuple[str, str] | None:
        """Why the credential ``name`` would reach an endpoint the method does not name: it has
        no ``base_url`` while the host's ``ANTHROPIC_BASE_URL`` is set, or the credential is
        ``ANTHROPIC_AUTH_TOKEN`` (a compatible endpoint's token). The message and its fix."""
        if self.claude_options.base_url:
            return None
        if endpoint_set([ENDPOINT_ENV]) is not None:
            return (
                f"{ENDPOINT_ENV} is set on this host, but the method has no base_url: Claude Code "
                f"in its sandbox would send {name} to api.anthropic.com, not to that endpoint",
                f"give the method that endpoint (--method claude_code:base_url=${ENDPOINT_ENV}; "
                f'base_url = "..." in an experiment file), or unset {ENDPOINT_ENV} to use '
                "api.anthropic.com",
            )
        if name == GATEWAY_TOKEN:
            return (
                f"{GATEWAY_TOKEN} is the token of an Anthropic-compatible endpoint, but the method "
                "has no base_url: Claude Code would send it to api.anthropic.com",
                "give the method that endpoint (--method claude_code:base_url=https://...), or "
                f"export {KEY_ENV} for api.anthropic.com",
            )
        return None

    def credential_status(self) -> CredentialStatus:
        status = env_status(KEY_ENVS)
        name = next((n for n in KEY_ENVS if os.environ.get(n)), None)
        problem = self._endpoint_problem(name) if name is not None else None
        return status if problem is None else CredentialStatus(False, *problem)

    def credentials(self, context: BuildContext) -> Credentials:
        name = next((n for n in KEY_ENVS if os.environ.get(n)), None)
        if name is None:
            raise SetupError(
                f"none of {', '.join(KEY_ENVS)} is set",
                fix=f"export {KEY_ENV}=... (or ANTHROPIC_AUTH_TOKEN with base_url=..., or "
                "CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`)",
            )
        problem = self._endpoint_problem(name)
        if problem is not None:
            raise SetupError(problem[0], fix=problem[1])
        public = {
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "IS_SANDBOX": "1",
        }
        if self.claude_options.base_url:
            public["ANTHROPIC_BASE_URL"] = self.claude_options.base_url
        return Credentials(
            env={name: os.environ[name]},
            public_env=public,
            description=f"{name} from the environment",
        )

    def command(self, context: BuildContext, workspace: Workspace) -> list[str]:
        options = self.claude_options
        argv = [
            "claude", "-p", "--output-format", "stream-json", "--verbose",
            "--dangerously-skip-permissions",
        ]  # fmt: skip
        if options.model:
            argv += ["--model", options.model]
        if options.max_turns is not None:
            argv += ["--max-turns", str(options.max_turns)]
        return [*argv, *options.extra_args]

    def reasoning_effort(self) -> str | None:
        return None  # never passed to Claude Code

    def usage(self, trajectory: Path) -> Usage | None:
        """The ``result`` event's ``usage`` (the whole run's); without one (a run stopped before
        it), the ``usage`` of each assistant message (``message.usage``, repeated in every event
        of that message: counted once per message id)."""
        events = read_events(trajectory)
        usages = [e["usage"] for e in events if e.get("type") == "result" and "usage" in e]
        if not usages:
            messages: dict[str, Any] = {}
            for event in events:
                message = event.get("message")
                if event.get("type") == "assistant" and isinstance(message, dict):
                    key = message.get("id")
                    messages[str(key) if key is not None else f"#{len(messages)}"] = message.get(
                        "usage"
                    )
            usages = list(messages.values())
        return anthropic_usage(usages)

    def cost(self, trajectory: Path) -> float | None:
        for event in reversed(read_events(trajectory)):
            value: Any = event.get("total_cost_usd", event.get("cost_usd"))
            if isinstance(value, int | float):
                # The last cost said; one that is not an amount of money (NaN, Infinity,
                # negative, a boolean, too large) leaves it unknown, never taken from earlier.
                return cost_amount(value)
        return None


def anthropic_usage(usages: list[Any]) -> Usage | None:
    """The sum of usage objects in the Anthropic API's terms, in the record's: there,
    ``input_tokens`` leaves out the prompt cache's tokens (``cache_read_input_tokens``,
    ``cache_creation_input_tokens``); in the record, as for every agent, input counts them, and
    ``cached_input_tokens`` is the part read from the cache. ``None`` without any."""
    totals: dict[str, int] = {}
    found = False
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        found = True
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[key] = totals.get(key, 0) + value
    if not found:
        return None
    read = totals.get("cache_read_input_tokens", 0)
    inputs = totals.get("input_tokens", 0) + read + totals.get("cache_creation_input_tokens", 0)
    outputs = totals.get("output_tokens", 0)
    return Usage(
        input_tokens=inputs,
        cached_input_tokens=read,
        output_tokens=outputs,
        total_tokens=inputs + outputs,
    )
