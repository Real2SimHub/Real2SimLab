"""``codex``: OpenAI's Codex CLI, run non-interactively (``codex exec``) in a container.

Credentials follow the host's own Codex setup, injected only at run time:

* ``provider = "host"`` (default): read ``$CODEX_HOME/config.toml`` (default ``~/.codex``), with
  the profile it selects (``profile``) over its top level for the provider and the model. A
  custom ``model_provider`` is reproduced in the sandbox, its whole table (``query_params``,
  ``http_headers``, ``env_http_headers`` with the variables they name, retries...) but its key
  source: the key is passed through an environment variable (it comes from the provider's
  ``env_key`` variable or its ``auth.command``, run on the host). The default OpenAI provider
  uses the host's ChatGPT login (``auth.json``), copied into a tmpfs ``CODEX_HOME`` that
  disappears with the container. Other settings of the host's configuration (MCP servers,
  ``model_reasoning_summary``...) are not reproduced.
* ``provider = "api-key"``: ``OPENAI_API_KEY`` from the environment, for ``api.openai.com``.
  A host whose ``OPENAI_BASE_URL`` names another endpoint (which the sandbox's Codex never sees)
  is refused: the key would go to ``api.openai.com`` instead of there. A custom endpoint is a
  ``model_provider`` of the host's Codex configuration (``provider = "host"``).

Codex's own sandbox is bypassed inside the container (the container is the sandbox); its
network is the egress allowlist: the provider's host, plus what the logging proxy showed it
needs.
"""

from __future__ import annotations

import os
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit

import tomli_w

from real2simlab.agents.base import (
    AgentMethod,
    AgentOptions,
    Credentials,
    CredentialStatus,
    Workspace,
    endpoint_set,
    env_status,
    read_events,
    total_usage,
)
from real2simlab.agents.record import Usage
from real2simlab.errors import SetupError
from real2simlab.methods.base import BuildContext, MethodOptions

CODEX_HOME = "/r2s/home/.codex"
KEY_ENV = "R2S_CODEX_API_KEY"
OPENAI_HOSTS = ("api.openai.com", "chatgpt.com", "auth.openai.com", "ab.chatgpt.com")
ENV_KEY_NAME = re.compile(r"[A-Z_][A-Z0-9_]{0,63}")
"""How a provider's ``env_key`` names the variable that holds its key: in capitals, digits and
underscores (``OPENAI_API_KEY``), at most 64 characters. Any other value is never shown in a
message, which ``r2s eval`` also writes to the run's log: it may be the key itself, written
there by mistake (a key of letters, digits and underscores, mixed-case or hexadecimal, is a
valid variable name too, so that alone is not enough)."""
ENDPOINT_ENV = "OPENAI_BASE_URL"
"""Where the host's Codex sends the OpenAI provider's requests when it is set."""


class CodexOptions(AgentOptions):
    provider: Literal["host", "api-key"] = "host"
    codex_home: str | None = None
    """The host's Codex home; defaults to ``$CODEX_HOME`` or ``~/.codex``."""


def host_codex_home(option: str | None) -> Path:
    return Path(option or os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()


class CodexAgent(AgentMethod):
    name: ClassVar[str] = "codex"
    agent_name: ClassVar[str] = "codex"
    summary: ClassVar[str] = "OpenAI Codex CLI (codex exec) in a container"
    agent_image: ClassVar[str | None] = "r2s-agent-codex"
    version_command: ClassVar[tuple[str, ...] | None] = ("codex", "--version")
    Options: ClassVar[type[MethodOptions]] = CodexOptions

    @property
    def codex_options(self) -> CodexOptions:
        assert isinstance(self.options, CodexOptions)
        return self.options

    def _config_path(self) -> Path:
        """The host's Codex configuration: ``config.toml`` in ``codex_home``, ``$CODEX_HOME``
        or ``~/.codex``, the file every fix about the host's provider names."""
        return host_codex_home(self.codex_options.codex_home) / "config.toml"

    def _host_config(self) -> dict[str, Any]:
        path = self._config_path()
        if not path.is_file():
            return {}
        try:
            return tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise SetupError(
                f"cannot read {path}: {error}", fix="fix or remove the file"
            ) from error

    def _selected(self) -> dict[str, Any]:
        """The host's configuration as its Codex reads the provider and the model from it: the
        top level, with the profile it selects (``profile``) over it."""
        config = self._host_config()
        name = config.get("profile")
        if name is None or name == "":
            return config
        path = host_codex_home(self.codex_options.codex_home) / "config.toml"
        if not isinstance(name, str):
            raise SetupError(
                f"{path}: profile must name a table [profiles.NAME], not {name!r}",
                fix='write profile = "NAME" with a table [profiles.NAME] in that file, or remove '
                "profile",
            )
        profiles = config.get("profiles")
        chosen = profiles.get(name) if isinstance(profiles, dict) else None
        if not isinstance(chosen, dict):
            raise SetupError(
                f"{path} selects the profile {name!r}, which it does not define as a table",
                fix=f"define [profiles.{name}] there, or remove its profile setting",
            )
        keys = ("model_provider", "model")
        for key in keys:
            if key in chosen and not isinstance(chosen[key], str):
                raise SetupError(
                    f"{path}: {key} of the profile {name!r} is not a string: {chosen[key]!r}",
                    fix=f'write {key} = "..." in [profiles.{name}] there',
                )
        return {**config, **{key: chosen[key] for key in keys if key in chosen}}

    def _provider(self) -> tuple[str | None, dict[str, Any]]:
        """The host's custom provider id and table, or ``(None, {})`` for OpenAI."""
        if self.codex_options.provider != "host":
            return None, {}
        config = self._selected()
        provider = config.get("model_provider")
        if not provider or provider == "openai":
            return None, {}
        path = self._config_path()
        providers = config.get("model_providers", {})
        if not isinstance(provider, str) or not isinstance(providers, dict):
            raise SetupError(
                f"{path} does not define the Codex provider {provider!r} as Codex reads it: "
                "model_provider must name a table of model_providers",
                fix=f'write model_provider = "ID" and a table [model_providers.ID] (base_url, '
                f'env_key) in {path}, or use provider = "api-key"',
            )
        entry: Any = providers.get(provider, {})
        if not isinstance(entry, dict):
            raise SetupError(
                f"{path}: model_providers.{provider} is not a table",
                fix=f"write it as a table [model_providers.{provider}] (base_url, env_key) in "
                f'{path}, or use provider = "api-key"',
            )
        table: dict[str, Any] = dict(entry)
        if not table.get("base_url"):
            raise SetupError(
                f"the host's Codex provider {provider!r} has no base_url",
                fix=f'set it in [model_providers.{provider}] in {path} or use provider = "api-key"',
            )
        return provider, table

    def model_name(self) -> str | None:
        return self.codex_options.model or self._selected().get("model")

    def allowlist(self) -> tuple[str, ...]:
        if self.codex_options.allowlist:
            return self.codex_options.allowlist
        provider, table = self._provider()
        if provider is not None:
            host = urlsplit(str(table["base_url"])).hostname
            return (host,) if host else ()
        return OPENAI_HOSTS

    def credential_status(self) -> CredentialStatus:
        """Where the key would come from, without running the provider's auth command or
        reading the login file."""
        if self.codex_options.provider == "api-key":
            status = env_status(["OPENAI_API_KEY"])
            problem = _api_key_endpoint() if status.available else None
            return status if problem is None else CredentialStatus(False, *problem)
        provider, table = self._provider()
        if provider is not None:
            env_key = table.get("env_key")
            if env_key and os.environ.get(str(env_key)):
                return CredentialStatus(True, f"host provider {provider!r}: {env_key} is set")
            auth = table.get("auth")
            if isinstance(auth, dict) and auth.get("command"):
                return CredentialStatus(
                    True, f"host provider {provider!r}: key from its auth command (not run here)"
                )
            message, fix = self._no_key(provider, env_key)
            return CredentialStatus(False, f"host {message}", fix)
        if (host_codex_home(self.codex_options.codex_home) / "auth.json").is_file():
            return CredentialStatus(True, "host ChatGPT login (auth.json present, not read)")
        return CredentialStatus(
            False, "no Codex login found", 'run `codex login` on the host, or provider = "api-key"'
        )

    def credentials(self, context: BuildContext) -> Credentials:
        options = self.codex_options
        model = self.model_name()
        config: dict[str, Any] = {}
        if model:
            config["model"] = model
        if options.reasoning_effort:
            config["model_reasoning_effort"] = options.reasoning_effort
        secret_env: dict[str, str] = {}
        files: dict[str, bytes] = {}
        if options.provider == "api-key":
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise SetupError("OPENAI_API_KEY is not set", fix="export OPENAI_API_KEY=...")
            problem = _api_key_endpoint()
            if problem is not None:
                raise SetupError(problem[0], fix=problem[1])
            secret_env["OPENAI_API_KEY"] = key
            description = "OPENAI_API_KEY from the environment"
        else:
            provider, table = self._provider()
            if provider is not None:
                key = self._provider_key(provider, table)
                config["model_provider"] = provider
                config["model_providers"] = {provider: _sandbox_provider(provider, table)}
                secret_env[KEY_ENV] = key
                headers = table.get("env_http_headers")
                for variable in headers.values() if isinstance(headers, dict) else ():
                    if isinstance(variable, str) and os.environ.get(variable):
                        secret_env[variable] = os.environ[variable]  # a header's value
                description = f"host provider {provider!r}, key injected through {KEY_ENV}"
            else:
                auth = host_codex_home(options.codex_home) / "auth.json"
                if not auth.is_file():
                    raise SetupError(
                        f"no Codex login found at {auth}",
                        fix='run `codex login` on the host, or use provider = "api-key"',
                    )
                files[f"{CODEX_HOME}/auth.json"] = auth.read_bytes()
                description = "host ChatGPT login (auth.json) copied into a tmpfs CODEX_HOME"
        files[f"{CODEX_HOME}/config.toml"] = tomli_w.dumps(config).encode()
        return Credentials(
            env=secret_env,
            files=files,
            public_env={"CODEX_HOME": CODEX_HOME},
            description=description,
        )

    def _no_key(self, provider: str, env_key: object) -> tuple[str, str]:
        """Why the host's ``provider`` gives no key, and the fix: the variable its ``env_key``
        names is not set, or it names no source at all. (``KEY_ENV`` is only the variable the
        sandbox's Codex reads: setting it on the host changes nothing.) An ``env_key`` not
        written as a variable's name (``ENV_KEY_NAME``) is never shown: it may be the key."""
        config = self._config_path()
        if env_key and not (isinstance(env_key, str) and ENV_KEY_NAME.fullmatch(env_key)):
            return (
                f"provider {provider!r} has an env_key that is not the name of an environment "
                f"variable (not shown: it may be the key itself), in {config}",
                f"set env_key in [model_providers.{provider}] of {config} to the name of the "
                'variable that holds the key, in capitals (env_key = "MY_PROVIDER_API_KEY"), '
                'and export that variable, never the key itself; or use provider = "api-key"',
            )
        if env_key:
            return (
                f"provider {provider!r} reads its key from {env_key}, which is not set",
                f"export {env_key}=... (the variable env_key names in {config})",
            )
        return (
            f"provider {provider!r} has no key source (env_key or auth.command)",
            f'add env_key = "VARIABLE" to [model_providers.{provider}] in '
            f'{self._config_path()} and export VARIABLE, or use provider = "api-key"',
        )

    def _provider_key(self, provider: str, table: dict[str, Any]) -> str:
        env_key = table.get("env_key")
        if env_key and os.environ.get(str(env_key)):
            return os.environ[str(env_key)]
        auth = table.get("auth")
        if isinstance(auth, dict) and auth.get("command"):
            command = [str(auth["command"]), *[str(a) for a in auth.get("args", [])]]
            path = self._config_path()
            check = f"check the auth command of [model_providers.{provider}] in {path}"
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                    stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired:  # its message would quote the arguments
                raise SetupError(
                    f"the provider's auth command ({Path(command[0]).name}) timed out",
                    fix=f"{check}: it must print the key within 30 seconds",
                ) from None
            except OSError as error:
                raise SetupError(
                    f"the provider's auth command ({Path(command[0]).name}) failed: "
                    f"{error.strerror or type(error).__name__}",
                    fix=f"{check}: its command must be a program this user can run",
                ) from None
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            ended = f"exited with {result.returncode}" if result.returncode else "printed no key"
            raise SetupError(  # the command's arguments may hold a token: name it only
                f"the provider's auth command ({Path(command[0]).name}) {ended}",
                fix=check,
            )
        message, fix = self._no_key(provider, env_key)
        raise SetupError(f"the host's Codex {message}", fix=fix)

    def command(self, context: BuildContext, workspace: Workspace) -> list[str]:
        options = self.codex_options
        argv = [
            "codex", "exec", "--json", "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox", "--color", "never",
            "-C", "/workspace", "-o", "/tmp/codex-last-message.md",
        ]  # fmt: skip
        model = self.model_name()
        if model:
            argv += ["-m", model]
        if options.reasoning_effort:
            argv += ["-c", f'model_reasoning_effort="{options.reasoning_effort}"']
        return [*argv, *options.extra_args, "-"]

    def copy_after(self) -> dict[str, str]:
        return {"/tmp/codex-last-message.md": "last_message.md"}

    def usage(self, trajectory: Path) -> Usage | None:
        return total_usage(read_events(trajectory))


def _api_key_endpoint() -> tuple[str, str] | None:
    """Why ``OPENAI_API_KEY`` (``provider = "api-key"``) would reach an endpoint the host does
    not use for it: the host's ``OPENAI_BASE_URL`` is set. The message and its fix."""
    if endpoint_set([ENDPOINT_ENV]) is None:
        return None
    return (
        f'{ENDPOINT_ENV} is set on this host, but codex with provider = "api-key" sends '
        "OPENAI_API_KEY to api.openai.com (its sandbox never sees that variable), not to that "
        "endpoint",
        "make the endpoint a model_provider of ~/.codex/config.toml (base_url, env_key) and use "
        f'the default provider = "host", or unset {ENDPOINT_ENV} to use api.openai.com',
    )


NOT_COPIED = frozenset({"env_key", "env_key_instructions", "auth", "experimental_bearer_token"})
"""Keys of a host's provider table not reproduced in the sandbox: its key sources (the key is
passed through ``KEY_ENV``) and ``requires_openai_auth`` (the key, not a ChatGPT login, is what
the sandbox has)."""


def _sandbox_provider(provider: str, table: dict[str, Any]) -> dict[str, Any]:
    """The host's provider ``table`` as the sandbox's Codex gets it: every setting (its endpoint,
    ``query_params``, ``http_headers``, ``env_http_headers``, retries...) but its key source,
    which is ``KEY_ENV``."""
    copied = {
        key: value
        for key, value in table.items()
        if key not in NOT_COPIED and key != "requires_openai_auth"
    }
    return {
        "name": table.get("name", provider),
        "wire_api": table.get("wire_api", "responses"),
        **copied,
        "base_url": table["base_url"],
        "env_key": KEY_ENV,
    }
