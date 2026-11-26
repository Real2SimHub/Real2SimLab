"""``opencode``: SST OpenCode (``opencode run``), run non-interactively in a container.

The model is passed as ``provider/model``. Standard providers use their usual API-key
environment variable; a custom OpenAI-compatible endpoint can be selected with ``base_url``.
With ``base_url``, the model runs on that endpoint whatever its provider part says: it is
registered there as the provider ``custom``, and OpenCode is told to use ``custom/<id>``, so
``openai/<id>`` (the LiteLLM name ``mini_swe`` takes for the same endpoint) never selects
OpenCode's own ``openai`` provider, which would send the key to ``api.openai.com``. Its
configuration is written to the sandbox's tmpfs home and refers to the key by environment
variable, so the secret never enters the workspace or run records. Without ``base_url``, a host
whose ``OPENAI_BASE_URL`` (or ``ANTHROPIC_BASE_URL``) names another endpoint for the model's
provider is refused: its key would go to the provider's own API instead.

Experimental: tested without a model (command line, credentials, configuration). OpenCode may
also fetch model metadata (``models.dev``) and provider packages at start-up; the allowlist
includes ``models.dev``, and the egress log shows what else a first real run needs.
Written by Codex following ``.agents/skills/add-method/SKILL.md``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit

from pydantic import field_validator, model_validator

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

CONFIG_FILE = "/r2s/home/.config/opencode/opencode.json"
DEFAULT_HOSTS = ("api.openai.com", "models.dev")
PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_GENERATIVE_AI_API_KEY",
}
PROVIDER_HOSTS = {
    "openai": ("api.openai.com",),
    "anthropic": ("api.anthropic.com",),
    "google": ("generativelanguage.googleapis.com",),
}
PROVIDER_ENDPOINTS = {"openai": "OPENAI_BASE_URL", "anthropic": "ANTHROPIC_BASE_URL"}
"""The host's variable that sends a provider's requests elsewhere, which OpenCode in its sandbox
never sees: set, it means the key is not the provider's own API's."""
CUSTOM = "custom"
"""The provider that ``base_url`` is registered as."""
PROMPT = "Read /workspace/TASK.md and do what it says. Work only in /workspace."


class OpenCodeOptions(AgentOptions):
    base_url: str | None = None
    """An OpenAI-compatible endpoint; its host joins the network allowlist. The model (its id
    after the provider part, e.g. ``openai/<id>`` or ``custom/<id>``) runs there."""
    api_key_env: str = "OPENAI_API_KEY"
    """Environment variable containing the key for ``base_url``."""

    @field_validator("reasoning_effort")
    @classmethod
    def _effort(cls, value: str | None) -> str | None:
        return effort_not_passed("OpenCode", value)

    @model_validator(mode="after")
    def _model_for_endpoint(self) -> OpenCodeOptions:
        if self.base_url and not self.model:
            raise ValueError(
                "base_url needs model=<id>, the model's id on that endpoint (OpenCode would "
                "otherwise pick a model of a provider of its own)"
            )
        return self


class OpenCodeAgent(AgentMethod):
    name: ClassVar[str] = "opencode"
    agent_name: ClassVar[str] = "opencode"
    summary: ClassVar[str] = "SST OpenCode (opencode run) in a container"
    agent_image: ClassVar[str | None] = "r2s-agent-opencode"
    version_command: ClassVar[tuple[str, ...] | None] = ("opencode", "--version")
    default_allowlist: ClassVar[tuple[str, ...]] = DEFAULT_HOSTS
    Options: ClassVar[type[MethodOptions]] = OpenCodeOptions

    @property
    def opencode_options(self) -> OpenCodeOptions:
        assert isinstance(self.options, OpenCodeOptions)
        return self.options

    def _provider(self) -> str:
        return (self.opencode_options.model or "openai/").split("/", 1)[0].lower()

    def _model_id(self) -> str:
        """The model's id on the ``base_url`` endpoint: the model without its provider part."""
        model = self.opencode_options.model or "model"
        return model.split("/", 1)[1] if "/" in model else model

    def _key_env(self) -> str:
        options = self.opencode_options
        if options.base_url:
            return options.api_key_env
        return PROVIDER_KEYS.get(self._provider(), options.api_key_env)

    def _endpoint_problem(self, key_env: str) -> tuple[str, str] | None:
        """Why ``key_env`` would reach an API the host does not use for this provider: without
        ``base_url``, the host's endpoint variable for it (:data:`PROVIDER_ENDPOINTS`) is set."""
        if self.opencode_options.base_url:
            return None
        provider = self._provider()
        variable = endpoint_set([PROVIDER_ENDPOINTS.get(provider, "")])
        if variable is None:
            return None
        host = PROVIDER_HOSTS[provider][0]
        return (
            f"{variable} is set on this host, but the method has no base_url: OpenCode in its "
            f"sandbox would send {key_env} to {host}, not to that endpoint",
            f"give the method that endpoint, if it is OpenAI-compatible (--method "
            f'opencode:base_url=${variable}; base_url = "..." in an experiment file), or unset '
            f"{variable} to use {host}",
        )

    def allowlist(self) -> tuple[str, ...]:
        options = self.opencode_options
        if options.allowlist:
            return options.allowlist
        if options.base_url:
            host = urlsplit(options.base_url).hostname
            return (host, "models.dev") if host else ()
        provider = (options.model or "openai/").split("/", 1)[0].lower()
        return (*PROVIDER_HOSTS.get(provider, DEFAULT_HOSTS[:1]), "models.dev")

    def credential_status(self) -> CredentialStatus:
        key_env = self._key_env()
        status = env_status([key_env])
        problem = self._endpoint_problem(key_env) if status.available else None
        return status if problem is None else CredentialStatus(False, *problem)

    def credentials(self, context: BuildContext) -> Credentials:
        options = self.opencode_options
        key_env = self._key_env()
        key = os.environ.get(key_env)
        if not key:
            raise SetupError(f"{key_env} is not set", fix=f"export {key_env}=...")
        problem = self._endpoint_problem(key_env)
        if problem is not None:
            raise SetupError(problem[0], fix=problem[1])
        files: dict[str, bytes] = {}
        if options.base_url:
            model_id = self._model_id()
            config = {
                "$schema": "https://opencode.ai/config.json",
                "provider": {
                    CUSTOM: {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "Custom OpenAI-compatible",
                        "options": {
                            "baseURL": options.base_url,
                            "apiKey": f"{{env:{key_env}}}",
                        },
                        "models": {model_id: {"name": model_id}},
                    }
                },
            }
            files[CONFIG_FILE] = (json.dumps(config) + "\n").encode()
        return Credentials(
            env={key_env: key},
            public_env={},
            files=files,
            description=f"{key_env} from the environment",
        )

    def command(self, context: BuildContext, workspace: Workspace) -> list[str]:
        options = self.opencode_options
        argv = ["opencode", "run", "--format", "json", "--auto"]
        if options.base_url:  # the provider its configuration registers for that endpoint
            argv += ["-m", f"{CUSTOM}/{self._model_id()}"]
        elif options.model:
            argv += ["-m", options.model]
        return [*argv, *options.extra_args, PROMPT]

    def reasoning_effort(self) -> str | None:
        return None  # never passed to OpenCode

    def usage(self, trajectory: Path) -> Usage | None:
        """The tokens of OpenCode's ``step_finish`` events (``part.tokens``, one per model call),
        summed: ``input`` there leaves out what was read from or written to the provider's
        cache, which the record counts as input (``cached_input_tokens`` is what was read)."""
        totals = {"input": 0, "read": 0, "write": 0, "output": 0, "reasoning": 0, "total": 0}
        found = has_total = False
        for event in read_events(trajectory):
            part = event.get("part")
            tokens = part.get("tokens") if isinstance(part, dict) else None
            if event.get("type") != "step_finish" or not isinstance(tokens, dict):
                continue
            found = True
            cache = tokens.get("cache")
            cache = cache if isinstance(cache, dict) else {}
            for key, value in (
                ("input", tokens.get("input")),
                ("output", tokens.get("output")),
                ("reasoning", tokens.get("reasoning")),
                ("read", cache.get("read")),
                ("write", cache.get("write")),
            ):
                totals[key] += count(value)
            if isinstance(tokens.get("total"), int) and not isinstance(tokens.get("total"), bool):
                has_total = True
                totals["total"] += count(tokens.get("total"))
        if not found:
            return None
        inputs = totals["input"] + totals["read"] + totals["write"]
        return Usage(
            input_tokens=inputs,
            cached_input_tokens=totals["read"],
            output_tokens=totals["output"],
            reasoning_tokens=totals["reasoning"],
            total_tokens=totals["total"] if has_total else inputs + totals["output"],
        )

    def cost(self, trajectory: Path) -> float | None:
        for event in reversed(read_events(trajectory)):
            for key in ("cost", "cost_usd", "total_cost_usd"):
                value = event.get(key)
                if isinstance(value, int | float):
                    # The last cost said; one that is not an amount of money (NaN, Infinity,
                    # negative, a boolean, too large) leaves it unknown.
                    return cost_amount(value)
        return None


def count(value: object) -> int:
    """A token count as an agent CLI printed it: a non-negative integer, else 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
