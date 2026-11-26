"""Agent adapters: redaction, bootstrap, credentials, the Codex command line, the fake agent."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from real2simlab.agents.base import total_usage
from real2simlab.agents.codex import KEY_ENV, CodexAgent
from real2simlab.agents.redact import REPLACEMENT, Redactor, redact
from real2simlab.errors import SetupError

BOOTSTRAP = Path(__file__).resolve().parents[1] / "real2simlab" / "agents" / "bootstrap.py"


@pytest.mark.parametrize(
    "secret",
    [
        "sk-proj-" + "a" * 40,
        "sk-ant-api03-" + "b" * 30,
        "AIza" + "c" * 35,
        "ghp_" + "d" * 36,
        "github_pat_" + "e" * 30,
        "AKIA" + "F" * 16,
        "hf_" + "g" * 34,
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop",
    ],
)
def test_key_shaped_strings_are_redacted(secret: str) -> None:
    text = f"token is {secret} ok"
    assert secret not in redact(text) and REPLACEMENT in redact(text)


def test_redaction_of_assignments_and_known_values() -> None:
    text = 'Authorization: Bearer abcdefghijklmnopqrstuv {"api_key": "zzzzzzzzzzzz"} password=hunter2hunter2'
    cleaned = redact(text)
    assert "abcdefghijklmnopqrstuv" not in cleaned and "zzzzzzzzzzzz" not in cleaned
    assert "hunter2hunter2" not in cleaned and "Bearer [REDACTED]" in cleaned
    redactor = Redactor(["my-very-private-value"])
    assert redactor("x my-very-private-value y") == f"x {REPLACEMENT} y" and redactor.count == 1
    assert (
        redact("nothing secret here, just numbers 0.123456789")
        == "nothing secret here, just numbers 0.123456789"
    )


def test_bootstrap_injects_files_and_redacts(tmp_path: Path) -> None:
    secret = "sk-test-" + "x" * 30
    plan = {
        "argv": [
            sys.executable,
            "-c",
            (
                "import os, pathlib; print(pathlib.Path(os.environ['AUTH']).read_text()); "
                "print('key', os.environ['MY_KEY']); print('leftover', os.environ.get('R2S_SECRET_FILE_0'))"
            ),
        ],
        "stdout": str(tmp_path / "out.log"),
        "stderr": str(tmp_path / "err.log"),
        "summary": str(tmp_path / "summary.json"),
        "secret_env": ["MY_KEY"],
        "secret_files": {str(tmp_path / "home" / "auth.json"): "R2S_SECRET_FILE_0"},
        "unset_env": ["R2S_SECRET_FILE_0"],
        "copy_after": {str(tmp_path / "home" / "auth.json"): str(tmp_path / "copied.json")},
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    content = json.dumps({"tokens": {"access_token": "eyJ-access-token-value-123456"}}).encode()
    env = {
        **os.environ,
        "AUTH": str(tmp_path / "home" / "auth.json"),
        "MY_KEY": secret,
        "R2S_SECRET_FILE_0": base64.b64encode(content).decode(),
    }
    result = subprocess.run(
        [sys.executable, str(BOOTSTRAP), str(tmp_path / "plan.json")], env=env, check=False
    )
    assert result.returncode == 0
    out = (tmp_path / "out.log").read_text()
    assert secret not in out and "eyJ-access-token-value-123456" not in out
    assert "leftover None" in out
    assert (tmp_path / "home" / "auth.json").stat().st_mode & 0o777 == 0o600
    assert "eyJ-access-token-value-123456" not in (tmp_path / "copied.json").read_text()
    assert json.loads((tmp_path / "summary.json").read_text())["redactions"] >= 2


def test_usage_totals() -> None:
    events = [
        {"usage": {"input_tokens": 10, "output_tokens": 3}},
        {"type": "x"},
        {"usage": {"input_tokens": 5, "cached_input_tokens": 2}},
    ]
    usage = total_usage(events)
    assert usage is not None and usage.input_tokens == 15 and usage.total_tokens == 18
    assert total_usage([{"type": "x"}]) is None


def write_codex_home(root: Path, config: str, auth: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.toml").write_text(config)
    if auth is not None:
        (root / "auth.json").write_text(auth)
    return root


def test_codex_custom_provider_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "key"
    key_file.write_text("sk-provider-secret-1234567890\n")
    home = write_codex_home(
        tmp_path / "codex",
        f"""
model = "gpt-test"
model_provider = "gateway"
model_reasoning_effort = "xhigh"
[model_providers.gateway]
name = "Gateway"
base_url = "https://llm.example.com/v1"
wire_api = "responses"
[model_providers.gateway.auth]
command = "/bin/sh"
args = ["-c", "cat {key_file}"]
[mcp_servers.secret_tool]
command = "should-not-be-copied"
""",
    )
    agent = CodexAgent(codex_home=str(home), reasoning_effort="medium")
    assert agent.model_name() == "gpt-test"
    assert agent.allowlist() == ("llm.example.com",)
    credentials = agent.credentials(None)  # type: ignore[arg-type]
    assert credentials.env == {KEY_ENV: "sk-provider-secret-1234567890"}
    config = tomllib.loads(credentials.files["/r2s/home/.codex/config.toml"].decode())
    assert config["model_reasoning_effort"] == "medium" and config["model_provider"] == "gateway"
    assert config["model_providers"]["gateway"]["env_key"] == KEY_ENV
    assert "mcp_servers" not in config and "sk-provider" not in json.dumps(config)
    command = agent.command(None, None)  # type: ignore[arg-type]
    assert command[:2] == ["codex", "exec"] and "--json" in command and command[-1] == "-"
    assert "-m" in command and 'model_reasoning_effort="medium"' in command


def test_codex_chatgpt_login_and_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = write_codex_home(tmp_path / "codex", 'model = "gpt-x"\n', auth='{"tokens": {}}')
    agent = CodexAgent(codex_home=str(home))
    credentials = agent.credentials(None)  # type: ignore[arg-type]
    assert credentials.files["/r2s/home/.codex/auth.json"] == b'{"tokens": {}}'
    assert "api.openai.com" in agent.allowlist()
    empty = write_codex_home(tmp_path / "empty", "")
    with pytest.raises(SetupError, match="codex login"):
        CodexAgent(codex_home=str(empty)).credentials(None)  # type: ignore[arg-type]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SetupError, match="OPENAI_API_KEY"):
        CodexAgent(provider="api-key").credentials(None)  # type: ignore[arg-type]
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env-123456789012345")
    assert CodexAgent(provider="api-key").credentials(None).env == {
        "OPENAI_API_KEY": "sk-from-env-123456789012345"
    }  # type: ignore[arg-type]


def test_codex_reproduces_the_hosts_provider_and_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host's Codex provider reaches the sandbox whole (Azure's ``api-version`` in
    ``query_params``, headers, the variables ``env_http_headers`` names, retries), only its key
    source replaced; a provider and model chosen through ``profile`` are the ones used, not
    OpenAI's defaults with the ChatGPT login (A6-05)."""
    monkeypatch.setenv("QA_PROVIDER_KEY", "qa-not-a-key-provider-0123456789")
    monkeypatch.setenv("QA_TENANT_HEADER", "qa-tenant-7")
    home = write_codex_home(
        tmp_path / "codex",
        """
profile = "gw"
model = "top-level-model"
[profiles.gw]
model_provider = "qagw"
model = "profile-model"
[model_providers.qagw]
name = "QA gateway"
base_url = "https://gateway.example.invalid:8443/v1"
wire_api = "chat"
env_key = "QA_PROVIDER_KEY"
query_params = { api-version = "2025-04-01-preview" }
http_headers = { "X-QA-Team" = "a6" }
env_http_headers = { "X-QA-Tenant" = "QA_TENANT_HEADER" }
request_max_retries = 2
""",
        auth='{"tokens": {}}',  # a ChatGPT login too: not the one the profile chose
    )
    agent = CodexAgent(codex_home=str(home))
    assert agent.model_name() == "profile-model"
    assert agent.allowlist() == ("gateway.example.invalid",)
    assert agent.credential_status().available
    credentials = agent.credentials(None)  # type: ignore[arg-type]
    assert credentials.env == {
        KEY_ENV: "qa-not-a-key-provider-0123456789",
        "QA_TENANT_HEADER": "qa-tenant-7",
    }
    assert "/r2s/home/.codex/auth.json" not in credentials.files
    config = tomllib.loads(credentials.files["/r2s/home/.codex/config.toml"].decode())
    assert config["model"] == "profile-model" and config["model_provider"] == "qagw"
    provider = config["model_providers"]["qagw"]
    assert provider["env_key"] == KEY_ENV and provider["wire_api"] == "chat"
    assert provider["query_params"] == {"api-version": "2025-04-01-preview"}
    assert provider["http_headers"] == {"X-QA-Team": "a6"}
    assert provider["env_http_headers"] == {"X-QA-Tenant": "QA_TENANT_HEADER"}
    assert provider["request_max_retries"] == 2
    assert "QA_PROVIDER_KEY" not in json.dumps(config)
    missing = write_codex_home(tmp_path / "missing", 'profile = "nope"\n', auth="{}")
    with pytest.raises(SetupError, match="profile 'nope'"):
        CodexAgent(codex_home=str(missing)).credentials(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "config",
    [
        'profile = ["gw"]\n[profiles.gw]\nmodel_provider = "qagw"\n',
        'profile = "gw"\nprofiles = "gw"\n',
        'profile = "gw"\n[profiles]\ngw = "qagw"\n',
        'profile = "gw"\n[profiles.gw]\nmodel_provider = ["qagw"]\n',
        'profile = "gw"\n[profiles.gw]\nmodel = 4\n',
    ],
)
def test_a_codex_profile_codex_cannot_read_is_an_error_with_its_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str
) -> None:
    """A profile setting of the host's Codex configuration that is not what Codex reads (a
    profile that is not a name, profiles or a profile that is not a table, a provider or model
    that is not a string) is an ``error:`` with its fix wherever the configuration is read (the
    run's check of credentials, ``r2s doctor``), never a traceback."""
    from real2simlab.doctor import agent_checks

    home = write_codex_home(tmp_path / "codex", config, auth="{}")
    agent = CodexAgent(codex_home=str(home))
    for read in (agent.credential_status, lambda: agent.credentials(None), agent.model_name):  # type: ignore[arg-type]
        with pytest.raises(SetupError, match="profile") as refused:
            read()
        assert str(home / "config.toml") in str(refused.value) and refused.value.fix
    monkeypatch.setenv("CODEX_HOME", str(home))  # what r2s doctor reads
    (check,) = agent_checks(["codex"])
    assert not check.ok and "profile" in check.detail and check.fix


def test_method_options_are_strict() -> None:
    from real2simlab.errors import ProblemsError
    from real2simlab.methods import get_method

    with pytest.raises(ProblemsError, match="Extra inputs"):
        get_method("codex", modle="typo")
    assert get_method("fake_agent", mode="broken").label == "fake_agent:broken"
    with pytest.raises(ProblemsError):
        get_method("fake_agent", mode="animation")


def test_agent_needs_allowlist_capable_backend(local) -> None:
    from real2simlab.methods import get_method

    assert get_method("codex").problems(local)
    assert not get_method("codex", network="open").problems(local)


def test_a_key_in_the_developers_shell_never_reaches_a_test() -> None:
    """The gate's result does not depend on the shell it runs in: a gateway token
    (``ANTHROPIC_AUTH_TOKEN``), a subscription token (``CLAUDE_CODE_OAUTH_TOKEN``) or any other
    variable an adapter reads a key from is removed for every test (``conftest.py``), so no
    agent finds credentials a test did not give it."""
    from real2simlab.agents import claude_code, codex, gemini_cli, opencode
    from real2simlab.doctor import agent_checks
    from real2simlab.methods import get_method

    read = {*claude_code.KEY_ENVS, *gemini_cli.KEY_ENVS, *opencode.PROVIDER_KEYS.values()}
    assert not [name for name in sorted({*read, codex.KEY_ENV}) if name in os.environ]
    assert not [c for c in agent_checks(["claude_code", "gemini_cli", "opencode"]) if c.ok]
    with pytest.raises(SetupError, match="ANTHROPIC_API_KEY"):
        get_method("claude_code").credentials(None)  # type: ignore[attr-defined]


def test_claude_code_adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from real2simlab.agents.claude_code import KEY_ENVS
    from real2simlab.methods import get_method

    agent = get_method("claude_code", model="claude-sonnet-5-5", max_turns=50)
    for name in KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SetupError, match="ANTHROPIC_API_KEY"):
        agent.credentials(None)  # type: ignore[attr-defined]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-" + "k" * 20)
    credentials = agent.credentials(None)  # type: ignore[attr-defined]
    assert (
        list(credentials.env) == ["ANTHROPIC_API_KEY"] and "api.anthropic.com" in agent.allowlist()
    )  # type: ignore[attr-defined]
    command = agent.command(None, None)  # type: ignore[attr-defined]
    assert command[:2] == ["claude", "-p"] and "--dangerously-skip-permissions" in command
    assert command[command.index("--model") + 1] == "claude-sonnet-5-5" and "50" in command
    trajectory = tmp_path / "t.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "type": "result",
                "usage": {"input_tokens": 7, "output_tokens": 2},
                "total_cost_usd": 0.12,
            }
        )
        + "\n"
    )
    assert agent.usage(trajectory).input_tokens == 7 and agent.cost(trajectory) == 0.12  # type: ignore[attr-defined]
    proxied = get_method("claude_code", base_url="https://gw.example.org/anthropic")
    assert proxied.allowlist() == ("gw.example.org",)  # type: ignore[attr-defined]
    assert proxied.credentials(None).public_env["ANTHROPIC_BASE_URL"].startswith("https://gw")  # type: ignore[attr-defined]


def test_gemini_cli_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    from real2simlab.methods import get_method

    agent = get_method("gemini_cli", model="gemini-3-pro")
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SetupError, match="GEMINI_API_KEY"):
        agent.credentials(None)  # type: ignore[attr-defined]
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza" + "x" * 35)
    assert list(agent.credentials(None).env) == ["GOOGLE_API_KEY"]  # type: ignore[attr-defined]
    command = agent.command(None, None)  # type: ignore[attr-defined]
    assert command[:2] == ["gemini", "--yolo"] and "gemini-3-pro" in command
    assert "generativelanguage.googleapis.com" in agent.allowlist()  # type: ignore[attr-defined]


def test_opencode_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    from real2simlab.agents.opencode import PROMPT, OpenCodeAgent

    agent = OpenCodeAgent(model="custom/my-model", base_url="https://llm.example.org/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SetupError, match="OPENAI_API_KEY"):
        agent.credentials(None)  # type: ignore[arg-type]
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-12345678901234567890")
    credentials = agent.credentials(None)  # type: ignore[arg-type]
    assert credentials.env == {"OPENAI_API_KEY": "sk-test-12345678901234567890"}
    assert agent.allowlist() == ("llm.example.org", "models.dev")
    config = json.loads(credentials.files["/r2s/home/.config/opencode/opencode.json"])
    assert config["provider"]["custom"]["options"]["apiKey"] == "{env:OPENAI_API_KEY}"
    assert "sk-test" not in json.dumps(config) and not credentials.public_env
    command = agent.command(None, None)  # type: ignore[arg-type]
    assert command[:5] == ["opencode", "run", "--format", "json", "--auto"]
    assert command[command.index("-m") + 1] == "custom/my-model" and command[-1] == PROMPT
    assert all("sk-test" not in part for part in command)

    standard = OpenCodeAgent(model="anthropic/claude-sonnet")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-12345678901234567890")
    assert list(standard.credentials(None).env) == ["ANTHROPIC_API_KEY"]  # type: ignore[arg-type]
    assert "api.anthropic.com" in standard.allowlist()


def test_opencode_runs_the_model_on_its_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``base_url``, OpenCode is told the provider its configuration registers there
    (``custom``), whatever the model's provider part: ``openai/<id>`` (the LiteLLM name
    ``mini_swe`` takes for the same endpoint) would select OpenCode's own ``openai`` provider
    and send the endpoint's key to ``api.openai.com`` (A6Q-05)."""
    from real2simlab.agents.opencode import CONFIG_FILE, OpenCodeAgent
    from real2simlab.errors import ProblemsError
    from real2simlab.methods import get_method

    monkeypatch.setenv("OPENAI_API_KEY", "qa-not-a-key-0123456789")
    for model, model_id in [
        ("openai/gpt-4o-mini", "gpt-4o-mini"),
        ("custom/qa-model", "qa-model"),
        ("qa-model", "qa-model"),
        ("openai/org/qa-model", "org/qa-model"),
    ]:
        agent = OpenCodeAgent(model=model, base_url="https://gateway.example.invalid/v1")
        argv = agent.command(None, None)  # type: ignore[arg-type]
        provider, _, chosen = argv[argv.index("-m") + 1].partition("/")
        config = json.loads(agent.credentials(None).files[CONFIG_FILE])  # type: ignore[arg-type]
        assert set(config["provider"]) == {provider}, (model, argv)
        assert list(config["provider"][provider]["models"]) == [chosen] == [model_id]
        assert agent.allowlist()[0] == "gateway.example.invalid"
    with pytest.raises(ProblemsError, match="base_url needs model"):
        get_method("opencode", base_url="https://gateway.example.invalid/v1")
    standard = OpenCodeAgent(model="openai/gpt-4o-mini")  # no endpoint: OpenAI's own
    argv = standard.command(None, None)  # type: ignore[arg-type]
    assert argv[argv.index("-m") + 1] == "openai/gpt-4o-mini"


@pytest.mark.parametrize(
    ("method", "options", "present", "endpoint", "option"),
    [
        ("claude_code", {}, ["ANTHROPIC_AUTH_TOKEN"], "ANTHROPIC_BASE_URL", "base_url"),
        ("claude_code", {}, ["ANTHROPIC_API_KEY"], "ANTHROPIC_BASE_URL", "base_url"),
        ("claude_code", {}, ["ANTHROPIC_AUTH_TOKEN"], None, "base_url"),
        ("codex", {"provider": "api-key"}, ["OPENAI_API_KEY"], "OPENAI_BASE_URL", "model_provider"),
        ("opencode", {"model": "openai/qa-model"}, ["OPENAI_API_KEY"], "OPENAI_BASE_URL", "base_url"),
        ("opencode", {"model": "anthropic/qa"}, ["ANTHROPIC_API_KEY"], "ANTHROPIC_BASE_URL",
         "base_url"),
        ("mini_swe", {"model": "openai/qa-model"}, ["OPENAI_API_KEY"], "OPENAI_BASE_URL",
         "api_base"),
        # A model without its provider: LiteLLM takes it to that provider's own API all the same.
        ("mini_swe", {"model": "gpt-4o"}, ["OPENAI_API_KEY"], "OPENAI_BASE_URL", "api_base"),
        ("mini_swe", {"model": "claude-sonnet-4-20250514", "api_key_env": "ANTHROPIC_API_KEY"},
         ["ANTHROPIC_API_KEY"], "ANTHROPIC_BASE_URL", "api_base"),
    ],
)  # fmt: skip
def test_a_key_never_goes_past_the_endpoint_the_host_uses_it_for(
    monkeypatch: pytest.MonkeyPatch,
    local: Any,
    method: str,
    options: dict[str, Any],
    present: list[str],
    endpoint: str | None,
    option: str,
) -> None:
    """A host whose endpoint variable (``ANTHROPIC_BASE_URL``, ``OPENAI_BASE_URL``) sends the
    agent's provider elsewhere, or a gateway's token (``ANTHROPIC_AUTH_TOKEN``), while the method
    names no endpoint: the agent would send that key to the provider's own API. It is refused
    before anything runs, with the option that names the endpoint, and ``r2s doctor`` says so
    instead of ``ok`` (A9R2-04)."""
    from real2simlab.doctor import agent_checks
    from real2simlab.methods import get_method

    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
                 "OPENAI_API_KEY", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE",
                 "ANTHROPIC_API_BASE"):  # fmt: skip
        monkeypatch.delenv(name, raising=False)
    for name in present:
        monkeypatch.setenv(name, f"qa-not-a-key-{name}-0123456789")
    if endpoint is not None:
        monkeypatch.setenv(endpoint, "https://gateway.example.invalid/v1")
    network = {} if method == "mini_swe" else {"network": "open"}  # mini_swe's: always none
    agent: Any = get_method(method, **network, **options)
    with pytest.raises(SetupError) as refused:
        if method == "mini_swe":
            agent.problems(local)
        else:
            agent.credentials(None)
    said = str(refused.value)
    assert "api" in said and "qa-not-a-key" not in said and option in (refused.value.fix or "")
    if method != "mini_swe":  # what r2s doctor shows
        status = agent.credential_status()
        assert not status.available and option in (status.fix or ""), status
    if not options:
        (check,) = agent_checks([method])
        assert not check.ok and option in (check.fix or "")
    if option in {"base_url", "api_base"}:  # the endpoint named: what the host uses it for
        given = {option: "https://gateway.example.invalid/v1"}
        if method in {"opencode", "mini_swe"}:
            given["api_key_env"] = present[0]
        named: Any = get_method(method, **network, **{**options, **given})
        if method == "mini_swe":
            assert "api_base" not in str(named.problems(local))
        else:
            assert list(named.credentials(None).env) == present[:1]
            assert named.allowlist()[0] == "gateway.example.invalid"
            assert named.credential_status().available


def test_agents_that_pass_no_reasoning_effort_refuse_one_and_record_none() -> None:
    """``reasoning_effort`` reaches only the CLIs that take one (``codex``): another adapter
    refuses a value other than the default (it would change nothing that runs, and still be
    recorded and rebuild everything) and its record says null (A6-04)."""
    from real2simlab.errors import ProblemsError
    from real2simlab.methods import get_method

    for method, options in [
        ("claude_code", {}),
        ("gemini_cli", {}),
        ("opencode", {}),
        ("mini_swe", {"model": "openai/qa-model"}),
    ]:
        for effort in ("high", "bogus"):
            with pytest.raises(ProblemsError, match="reasoning_effort"):
                get_method(method, reasoning_effort=effort, **options)
        default: Any = get_method(method, **options)
        assert default.reasoning_effort() is None, method
        given: Any = get_method(
            method, reasoning_effort=default.options.reasoning_effort, **options
        )
        assert given.config_digest() == default.config_digest()  # the default, written out
    codex: Any = get_method("codex", reasoning_effort="high")
    assert codex.reasoning_effort() == "high"
    assert 'model_reasoning_effort="high"' in codex.command(None, None)


def test_token_usage_is_read_the_way_each_cli_prints_it(tmp_path: Path) -> None:
    """Each adapter reads its CLI's own events (as their pinned versions print them) into the
    record's terms: ``input_tokens`` is all the input, ``cached_input_tokens`` the part read from
    a cache (as Codex reports it) (A6-03)."""
    from real2simlab.methods import get_method

    def trajectory(*events: dict[str, Any]) -> Path:
        path = tmp_path / f"t{len(list(tmp_path.iterdir()))}.jsonl"
        path.write_text("".join(json.dumps(event) + "\n" for event in events))
        return path

    claude: Any = get_method("claude_code")
    usage = claude.usage(
        trajectory(
            {"type": "system", "subtype": "init"},
            {
                "type": "result",
                "subtype": "success",
                "total_cost_usd": 0.25,
                "usage": {
                    "input_tokens": 12,
                    "cache_creation_input_tokens": 3000,
                    "cache_read_input_tokens": 90000,
                    "output_tokens": 800,
                    "server_tool_use": {"web_search_requests": 0},
                    "service_tier": "standard",
                },
            },
        )
    )
    assert (usage.input_tokens, usage.cached_input_tokens) == (93012, 90000)
    assert (usage.output_tokens, usage.total_tokens) == (800, 93812)
    message = {"id": "msg_1", "usage": {"input_tokens": 4, "cache_read_input_tokens": 96,
                                        "output_tokens": 10}}  # fmt: skip
    stopped = claude.usage(  # stopped before its result: each message once, though repeated
        trajectory(
            {"type": "assistant", "message": message},
            {"type": "assistant", "message": message},
            {"type": "assistant", "message": {**message, "id": "msg_2"}},
        )
    )
    assert (stopped.input_tokens, stopped.cached_input_tokens, stopped.output_tokens) == (
        200,
        192,
        20,
    )
    gemini: Any = get_method("gemini_cli")
    stats = {"total_tokens": 1500, "input_tokens": 1200, "output_tokens": 300, "cached": 1000,
             "input": 200, "duration_ms": 5000, "tool_calls": 3, "models": {}}  # fmt: skip
    usage = gemini.usage(
        trajectory(
            {"type": "init", "timestamp": "t", "session_id": "s", "model": "gemini-x"},
            {"type": "message", "timestamp": "t", "role": "assistant", "content": "ok"},
            {"type": "result", "timestamp": "t", "status": "success", "stats": stats},
        )
    )
    assert (usage.input_tokens, usage.cached_input_tokens) == (1200, 1000)
    assert (usage.output_tokens, usage.total_tokens) == (300, 1500)
    assert gemini.usage(trajectory({"type": "message", "content": "no result"})) is None
    opencode: Any = get_method("opencode", model="openai/qa-model")

    def step(tokens: dict[str, Any]) -> dict[str, Any]:
        return {"type": "step_finish", "part": {"type": "step-finish", "tokens": tokens, "cost": 0}}

    usage = opencode.usage(
        trajectory(
            {"type": "step_start", "part": {"type": "step-start"}},
            step(
                {
                    "total": 110,
                    "input": 100,
                    "output": 10,
                    "reasoning": 0,
                    "cache": {"write": 0, "read": 0},
                }
            ),
            step(
                {
                    "total": 1260,
                    "input": 50,
                    "output": 10,
                    "reasoning": 5,
                    "cache": {"write": 200, "read": 1000},
                }
            ),
        )
    )
    assert (usage.input_tokens, usage.cached_input_tokens) == (1350, 1000)
    assert (usage.output_tokens, usage.reasoning_tokens, usage.total_tokens) == (20, 5, 1370)


def test_paths_in_an_agents_environment_are_rewritten_once_on_the_local_backend(
    bouncing_task: Path, tmp_path: Path, local, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a sandbox, a sandbox path in an agent's environment (Codex's ``CODEX_HOME``) is
    the host directory it is mounted from, where its credential files are, also when that lies
    under a sandbox mount point (the plan directory in ``/tmp``): rewritten twice, it pointed to
    nowhere, and the real Codex stopped at once (A6-02)."""
    import shutil
    import tempfile

    from real2simlab.agents import fake
    from real2simlab.agents.base import AGENT_HOME, Credentials
    from real2simlab.methods import get_method
    from real2simlab.methods.base import BuildContext
    from real2simlab.task import load_task
    from real2simlab.task.instruction import task_statement

    home = f"{AGENT_HOME}/.codex"
    injected = Credentials(
        files={f"{home}/config.toml": b'model = "qa"\n'},
        public_env={"CODEX_HOME": home},
        description="test",
    )
    monkeypatch.setattr(fake.FakeAgent, "credentials", lambda self, context: injected)
    probe = (
        "import json, os\n"
        "home = os.environ['CODEX_HOME']\n"
        "found = os.path.isfile(os.path.join(home, 'config.toml'))\n"
        "print(json.dumps({'type': 'probe', 'config': found}))\n"
    )
    monkeypatch.setattr(
        fake.FakeAgent, "command", lambda self, context, workspace: ["python", "-c", probe]
    )
    under_tmp = Path(tempfile.mkdtemp(prefix="r2s-test-", dir="/tmp"))
    monkeypatch.setattr(tempfile, "tempdir", str(under_tmp))  # the plan directory: in /tmp
    task = load_task(bouncing_task)
    context = BuildContext(
        task_id=task.id, task_hash="h", config=task.config, statement=task_statement(task),
        visible_dir=task.visible_dir, work_dir=tmp_path / "work", twin_dir=tmp_path / "twin",
        backend=local,
    )  # fmt: skip
    try:
        get_method("fake_agent", mode="nothing").build(context)
    finally:
        shutil.rmtree(under_tmp, ignore_errors=True)
    lines = (tmp_path / "work" / "agent" / "trajectory.jsonl").read_text().splitlines()
    assert {"type": "probe", "config": True} in [json.loads(line) for line in lines if line]


@pytest.mark.parametrize("gpu", [False, True])
def test_agent_sandbox_gets_the_task_gpu(
    gpu: bool, bouncing_task: Path, tmp_path: Path, local
) -> None:
    """An agent tests its program with the resources the frozen re-run will have."""
    from real2simlab.methods import get_method
    from real2simlab.methods.base import BuildContext
    from real2simlab.sandbox.base import ExecResult
    from real2simlab.task import load_task
    from real2simlab.task.instruction import task_statement

    task = load_task(bouncing_task)
    delivery = task.config.delivery.model_copy(
        update={"resources": task.config.delivery.resources.model_copy(update={"gpu": gpu})}
    )
    config = task.config.model_copy(update={"delivery": delivery})
    seen = []

    class Recording(type(local)):  # type: ignore[misc]
        def run(self, spec, log_path=None):  # type: ignore[no-untyped-def]
            seen.append(spec)
            return ExecResult(exit_code=1, timed_out=False, duration_s=0.0)

    method = get_method("fake_agent", mode="nothing", network="none", gpus="0")
    context = BuildContext(
        task_id=task.id, task_hash="h", config=config,
        statement=task_statement(task), visible_dir=task.visible_dir, work_dir=tmp_path / "work",
        twin_dir=tmp_path / "twin", backend=Recording(quiet=True),
    )  # fmt: skip
    outcome = method.build(context)
    assert outcome.twin is None
    assert seen and seen[0].limits.gpus == ("0" if gpu else None)


def test_a_solution_is_its_program(bouncing_task: Path, tmp_path: Path, local) -> None:
    """The agent hands in solution/run.py; the Lab writes twin.json itself."""
    from real2simlab.agents.base import Workspace
    from real2simlab.methods import get_method
    from real2simlab.methods.base import BuildContext
    from real2simlab.task import load_task

    task = load_task(bouncing_task)
    root = tmp_path / "workspace"
    (root / "solution").mkdir(parents=True)
    context = BuildContext(
        task_id=task.id, task_hash="h", config=task.config, statement="s",
        visible_dir=task.visible_dir, work_dir=tmp_path / "work", twin_dir=tmp_path / "twin",
        backend=local,
    )  # fmt: skip
    method = get_method("fake_agent", mode="physical")
    workspace = Workspace(root=root, statement_file=root / "TASK.md", statement="s")
    twin, failure = method._collect(context, workspace, "completed")  # type: ignore[attr-defined]
    assert twin is None and "no run.py" in str(failure)
    (root / "solution" / "run.py").write_text("print('twin')\n")
    (root / "solution" / "twin.json").write_text("{}")  # an agent's own manifest is ignored
    twin, failure = method._collect(context, workspace, "completed")  # type: ignore[attr-defined]
    assert failure is None and twin.manifest.origin.method == "fake_agent"


def test_the_workspace_never_follows_a_link_out_of_visible(
    bouncing_task: Path, tmp_path: Path, local
) -> None:
    """The agent's workspace gets the build's copy of ``visible/`` exactly: a symbolic link in
    it is never followed (what it points to stays out of the workspace); the workspace is not
    written, and the build fails outside the agent (not a result)."""
    import shutil

    from real2simlab.agents.base import prepare_workspace
    from real2simlab.errors import InfrastructureError
    from real2simlab.methods.base import BuildContext
    from real2simlab.task import load_task

    task = load_task(bouncing_task)
    visible = Path(shutil.copytree(task.visible_dir, tmp_path / "build" / "visible"))
    decoy = tmp_path / "decoy.txt"
    decoy.write_text("R2S-TEST-DECOY workspace")
    (visible / "notes.txt").symlink_to(decoy)
    context = BuildContext(
        task_id=task.id, task_hash="h", config=task.config, statement="s",
        visible_dir=visible, work_dir=tmp_path / "work", twin_dir=tmp_path / "twin",
        backend=local,
    )  # fmt: skip
    with pytest.raises(InfrastructureError, match="symbolic link"):
        prepare_workspace(context, tmp_path / "workspace")
    written = [p for p in (tmp_path / "workspace").rglob("*") if p.is_file() or p.is_symlink()]
    assert not [p for p in written if "visible" in p.parts], written


def test_agents_work_in_the_image_of_the_tasks_runtime() -> None:
    from real2simlab.envs import agent_environment

    assert agent_environment("r2s-agent-codex", "r2s-python") == "r2s-agent-codex"
    assert agent_environment("r2s-agent-codex", "r2s-blender-4.0") == "r2s-agent-codex-blender"
    assert agent_environment("r2s-agent-codex", "r2s-physimcode") == "r2s-agent-codex-physimcode"
    assert agent_environment("r2s-agent-codex", "4dcb/agent:dev") == "4dcb/agent:dev"
    assert agent_environment("r2s-agent-gemini-cli", "r2s-blender-4.0") == "r2s-agent-gemini-cli"


def test_the_readmes_say_which_agents_work_in_the_tasks_runtime() -> None:
    """Only the agents that get an image with the task's runtime are promised one (A8-01):
    Claude Code, Gemini CLI and OpenCode work in their own Python image on Blender tasks."""
    from real2simlab.envs import agent_environment
    from real2simlab.methods import get_method

    own = []
    for name in ("codex", "claude_code", "gemini_cli", "opencode"):
        image: Any = get_method(name).agent_image  # type: ignore[attr-defined]
        if agent_environment(image, "r2s-blender-4.0") == image:
            own.append(name)
    assert own == ["claude_code", "gemini_cli", "opencode"]
    root = Path(__file__).resolve().parents[1]
    for name in ("README.md", "README_zh.md", "CHANGELOG.md", "docs/tutorials/evaluate-agent.md"):
        text = (root / name).read_text(encoding="utf-8")
        assert "Each agent works in the image that matches" not in text, name
        assert "\u6bcf\u4e2a agent \u81ea\u52a8\u5728\u548c\u4efb\u52a1" not in text, name
        if name.startswith("README"):
            assert "Claude Code" in text and "Python" in text, name


def test_codex_auth_command_errors_never_show_its_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "codex"
    home.mkdir()
    token = "tok-1234567890abcdefghij"
    (home / "config.toml").write_text(
        'model_provider = "corp"\n[model_providers.corp]\nbase_url = "https://x.example/v1"\n'
        f'auth = {{ command = "false", args = ["--token", "{token}"] }}\n'
    )
    monkeypatch.setenv("CODEX_HOME", str(home))
    with pytest.raises(SetupError) as failed:
        CodexAgent().credentials(None)  # type: ignore[arg-type]
    assert token not in str(failed.value) and "(false) exited" in str(failed.value)
    (home / "config.toml").write_text(
        (home / "config.toml").read_text().replace('"false"', '"/nonexistent/helper"')
    )
    with pytest.raises(SetupError) as missing:
        CodexAgent().credentials(None)  # type: ignore[arg-type]
    assert token not in str(missing.value) and "(helper) failed" in str(missing.value)


@pytest.mark.parametrize("entry_point", ["credential_status", "credentials"])
def test_codex_names_the_variable_its_host_provider_reads(
    entry_point: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host provider whose ``env_key`` variable is not set was said to have "no key source",
    with the fix to set ``R2S_CODEX_API_KEY``, which the host never reads (only the sandbox's
    Codex does): following it changed nothing. Name the variable to export instead."""
    home = write_codex_home(
        tmp_path / "codex",
        'model_provider = "gw"\n[model_providers.gw]\nbase_url = "http://127.0.0.1:9/v1"\n'
        'env_key = "QA_GW_KEY_NOT_SET"\n',
    )
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.delenv("QA_GW_KEY_NOT_SET", raising=False)
    monkeypatch.setenv(KEY_ENV, "set-as-the-old-fix-said")
    agent = CodexAgent()
    if entry_point == "credential_status":
        status = agent.credential_status()
        assert not status.available
        message, fix = status.detail, status.fix or ""
    else:
        with pytest.raises(SetupError) as failed:
            agent.credentials(None)  # type: ignore[arg-type]
        message, fix = str(failed.value), failed.value.fix or ""
    assert "QA_GW_KEY_NOT_SET" in message and "no key source" not in message, message
    assert "export QA_GW_KEY_NOT_SET=" in fix and KEY_ENV not in fix + message, fix
    monkeypatch.setenv("QA_GW_KEY_NOT_SET", "sk-gw-fake-key-1234")
    assert agent.credential_status().available


ENV_KEY_MISTAKES = {
    "a key with dashes": "sk-qa-fake-000000000000",
    "a key of letters, digits and underscores": "qa_FAKE_key_000000000000",
    "a hexadecimal key": "a0b1c2d3e4f5061728394a5b6c7d8e9f",
}
"""Fake keys, as a user might write one where a Codex provider's ``env_key`` names a variable."""


@pytest.mark.parametrize("entry_point", ["credential_status", "credentials", "r2s eval"])
@pytest.mark.parametrize("value", list(ENV_KEY_MISTAKES.values()), ids=list(ENV_KEY_MISTAKES))
def test_codex_never_echoes_an_env_key_that_is_not_a_variable_name(
    value: str,
    entry_point: str,
    bouncing_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A host provider's ``env_key`` names the variable that holds its key. A key written
    there by mistake was echoed in the error and its fix ("reads its key from sk-...",
    "export sk-...=..."), which ``r2s eval`` also writes to the run's ``r2s.log``. Only a value
    written as such a variable's name is named; any other is not shown."""
    home = write_codex_home(
        tmp_path / "codex",
        'model_provider = "gw"\n[model_providers.gw]\nbase_url = "http://127.0.0.1:9/v1"\n'
        f'env_key = "{value}"\n',
    )
    monkeypatch.setenv("CODEX_HOME", str(home))
    run = tmp_path / "run"
    if entry_point == "credential_status":
        status = CodexAgent().credential_status()
        assert not status.available
        said = f"{status.detail}\n{status.fix}"
    elif entry_point == "credentials":
        with pytest.raises(SetupError) as failed:
            CodexAgent().credentials(None)  # type: ignore[arg-type]
        said = f"{failed.value}\n{failed.value.fix}"
    else:
        from real2simlab.cli.main import main

        argv = ["eval", "--tasks", str(bouncing_task), "--method", "codex:network=none",
                "--backend", "local", "--out", str(run)]  # fmt: skip
        assert main(argv) == 1
        captured = capsys.readouterr()
        said = captured.out + captured.err
    assert value not in said, said
    assert "not the name of an environment variable" in said, said
    assert str(home / "config.toml") in said, said
    kept = [p for p in run.rglob("*") if p.is_file() and value.encode() in p.read_bytes()]
    assert kept == [], kept  # the run's r2s.log among them


def test_codex_fixes_name_the_codex_home_in_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixes said ``~/.codex/config.toml`` whatever ``$CODEX_HOME`` (or ``codex_home=``)
    pointed to; a provider without any key source is told how to give one, and an auth
    command that cannot run gets a fix too."""
    home = write_codex_home(
        tmp_path / "elsewhere",
        'model_provider = "gw"\n[model_providers.gw]\nname = "no url"\n',
    )
    config = str(home / "config.toml")
    monkeypatch.setenv("CODEX_HOME", str(home))
    with pytest.raises(SetupError) as no_url:
        CodexAgent().credentials(None)  # type: ignore[arg-type]
    assert config in (no_url.value.fix or "") and "~/.codex" not in str(no_url.value)
    (home / "config.toml").write_text(
        'model_provider = "gw"\n[model_providers.gw]\nbase_url = "http://127.0.0.1:9/v1"\n'
    )
    status = CodexAgent().credential_status()
    assert not status.available and "env_key" in (status.fix or "") and config in (status.fix or "")
    assert KEY_ENV not in (status.fix or "")
    (home / "config.toml").write_text(
        'model_provider = "gw"\n[model_providers.gw]\nbase_url = "http://127.0.0.1:9/v1"\n'
        'auth = { command = "/nonexistent/helper" }\n'
    )
    with pytest.raises(SetupError) as helper:
        CodexAgent().credentials(None)  # type: ignore[arg-type]
    assert "(helper) failed" in str(helper.value) and config in (helper.value.fix or "")


@pytest.mark.parametrize(
    "config",
    ['model_provider = "gw"\nmodel_providers = "gw"\n',
     'model_provider = "gw"\n[model_providers]\ngw = "https://x.example/v1"\n'],
    ids=["model_providers-not-a-table", "provider-not-a-table"],
)  # fmt: skip
def test_codex_host_config_of_the_wrong_shape_is_an_error_with_a_fix(
    config: str,
    bouncing_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``model_providers`` (or the provider) that is not a table ended ``r2s eval`` in an
    ``AttributeError`` traceback and showed ``r2s doctor`` one without the file or a fix."""
    from real2simlab import doctor
    from real2simlab.cli.main import main

    home = write_codex_home(tmp_path / "codex", config)
    monkeypatch.setenv("CODEX_HOME", str(home))
    (check,) = doctor.agent_checks(["codex"])
    assert not check.ok and "AttributeError" not in check.detail, check.detail
    assert str(home / "config.toml") in (check.fix or ""), check.fix
    out = tmp_path / "run"
    code = main(["eval", "--tasks", str(bouncing_task), "--method", "codex:network=none",
                 "--backend", "local", "--out", str(out)])  # fmt: skip
    err = capsys.readouterr().err
    assert code == 1 and "Traceback" not in err and "AttributeError" not in err, err
    assert "error:" in err and str(home / "config.toml") in err, err


def test_agent_gpus_option() -> None:
    from real2simlab.methods import get_method

    assert get_method("fake_agent", gpus="none").agent_options.gpus is None  # type: ignore[attr-defined]
    assert get_method("fake_agent", gpus="0,1").agent_options.gpus == "0,1"  # type: ignore[attr-defined]
    with pytest.raises(Exception, match="gpus must be"):
        get_method("fake_agent", gpus="first")


@pytest.mark.parametrize(
    ("present", "chosen"),
    [
        (["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"], "ANTHROPIC_API_KEY"),
        (["ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"], "ANTHROPIC_AUTH_TOKEN"),
        (["CLAUDE_CODE_OAUTH_TOKEN"], "CLAUDE_CODE_OAUTH_TOKEN"),
    ],
)
def test_claude_code_credential_sources(
    monkeypatch: pytest.MonkeyPatch, present: list[str], chosen: str
) -> None:
    from real2simlab.agents.claude_code import KEY_ENVS, ClaudeCodeAgent

    for name in KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    for name in present:
        monkeypatch.setenv(name, f"value-of-{name}-0123456789")
    # A compatible endpoint's token goes only with its endpoint.
    gateway = {"base_url": "https://gw.example.org"} if chosen == "ANTHROPIC_AUTH_TOKEN" else {}
    agent = ClaudeCodeAgent(**gateway)
    credentials = agent.credentials(None)  # type: ignore[arg-type]
    assert list(credentials.env) == [chosen]  # exactly one reaches the sandbox
    assert credentials.description == f"{chosen} from the environment"
    assert "value-of" not in credentials.description
    assert credentials.public_env["IS_SANDBOX"] == "1"
    status = agent.credential_status()
    assert (status.available and chosen in status.detail) or present[0] in status.detail


def test_claude_code_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from real2simlab.agents.claude_code import KEY_ENVS, ClaudeCodeAgent

    for name in KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SetupError, match="CLAUDE_CODE_OAUTH_TOKEN") as missing:
        ClaudeCodeAgent().credentials(None)  # type: ignore[arg-type]
    assert "setup-token" in (missing.value.fix or "")
    assert not ClaudeCodeAgent().credential_status().available


def test_agents_run_unprivileged_when_the_host_user_is_root(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As root, the agent container runs as AGENT_USER; its directories go to that user for the
    run and back to their owners afterwards (checked against a fake docker)."""
    from real2simlab.agents import base
    from real2simlab.methods import get_method
    from real2simlab.methods.base import BuildContext
    from real2simlab.sandbox.base import ExecResult
    from real2simlab.sandbox.docker import DockerBackend
    from real2simlab.task import load_task
    from real2simlab.task.instruction import task_statement

    chowned: list[tuple[str, int, int]] = []
    monkeypatch.setattr(base.os, "getuid", lambda: 0)
    monkeypatch.setattr(
        base.os, "chown", lambda path, uid, gid, **_: chowned.append((str(path), uid, gid))
    )
    commands: list[list[str]] = []

    class FakeDocker(DockerBackend):
        def resolve_image(self, reference):  # type: ignore[no-untyped-def]
            from real2simlab.sandbox.base import ResolvedImage

            return ResolvedImage(reference=reference, digest=None)

        def run(self, spec, log_path):  # type: ignore[no-untyped-def]
            commands.append(self.command(spec, "test"))
            return ExecResult(exit_code=1, timed_out=False, duration_s=0.0)

    task = load_task(bouncing_task)
    context = BuildContext(
        task_id=task.id, task_hash="h", config=task.config,
        statement=task_statement(task), visible_dir=task.visible_dir, work_dir=tmp_path / "work",
        twin_dir=tmp_path / "twin", backend=FakeDocker(docker="docker"),
    )  # fmt: skip
    get_method("fake_agent", mode="nothing", network="none").build(context)
    command = commands[0]
    assert command[command.index("--user") + 1] == base.AGENT_USER
    workspace = str(tmp_path / "work" / "workspace")
    handed = [c for c in chowned if c[0] == workspace]
    assert handed[0][1:] == (10001, 10001)  # given to the agent first
    me = (Path(workspace).stat().st_uid, Path(workspace).stat().st_gid)
    assert handed[-1][1:] == me  # and given back afterwards


def test_agents_run_as_the_host_user_otherwise(monkeypatch: pytest.MonkeyPatch) -> None:
    from real2simlab.agents import base
    from real2simlab.sandbox.docker import DockerBackend
    from real2simlab.sandbox.local import LocalBackend

    monkeypatch.setattr(base.os, "getuid", lambda: 1000)
    assert base.agent_user(DockerBackend(docker="docker")) is None
    monkeypatch.setattr(base.os, "getuid", lambda: 0)
    assert base.agent_user(LocalBackend(quiet=True)) is None  # nothing to isolate


def test_bwrap_runs_as_the_requested_ids(tmp_path: Path) -> None:
    from real2simlab.envs.rootfs import Rootfs
    from real2simlab.sandbox.base import SandboxSpec
    from real2simlab.sandbox.bwrap import BwrapBackend

    unpacked = Rootfs("img", "sha256:x", tmp_path, {"PATH": "/usr/bin"}, "/")
    backend = BwrapBackend(bwrap="bwrap", cgroup_controllers=frozenset())
    command = backend.command(
        SandboxSpec(image="img", argv=("x",), workdir="/", user="10001:10001"), unpacked
    )
    text = " ".join(command)
    assert "--unshare-user --uid 10001 --gid 10001" in text
    assert "--uid" not in " ".join(
        backend.command(SandboxSpec(image="img", argv=("x",), workdir="/"), unpacked)
    )


def test_cli_version_takes_the_last_line(tmp_path: Path) -> None:
    from real2simlab.agents.base import cli_version

    script = tmp_path / "tool"
    script.write_text("#!/bin/sh\necho '== CUDA banner =='\necho\necho 'tool 1.2.3'\n")
    script.chmod(0o755)
    assert cli_version(None, "ignored", (str(script),)) == "tool 1.2.3"
    assert cli_version(None, "ignored", (str(tmp_path / "missing"),)) is None


def test_an_agent_version_is_the_one_in_the_image_it_runs_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On an isolating backend other than Docker (bwrap), the agent runs in its image: the
    version recorded with its runs (and in its twin's origin) is what ``version_command`` says
    there, run offline by the backend, never the host's CLI of the same name. An image without
    that CLI gives no version (unknown); on the ``local`` backend the agent is the host's."""
    from real2simlab.sandbox.base import ExecResult, NetworkMode, ResolvedImage, SandboxSpec
    from real2simlab.sandbox.local import LocalBackend

    host = tmp_path / "host-bin"
    host.mkdir()
    (host / "codex").write_text("#!/bin/sh\necho 'codex-cli 0.0.0-on-the-host'\n")
    (host / "codex").chmod(0o755)
    monkeypatch.setenv("PATH", f"{host}{os.pathsep}{os.environ['PATH']}")

    class InImage(LocalBackend):
        """An isolating backend whose sandboxes run in the image: what it prints is given."""

        isolated = True

        def __init__(self, printed: str, code: int) -> None:
            super().__init__(quiet=True)
            self.printed, self.code = printed, code
            self.specs: list[SandboxSpec] = []

        def resolve_image(self, reference: str) -> ResolvedImage:
            return ResolvedImage(reference, None, content_id=f"sha256:{reference}")

        def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
            self.specs.append(spec)
            Path(log_path).write_text(self.printed)
            return ExecResult(exit_code=self.code, timed_out=False, duration_s=0.1)

    image = str(tmp_path / "agent-image")  # its own name: versions are kept per image
    backend = InImage("codex-cli 9.9.9-in-the-image\n", 0)
    assert CodexAgent().version(backend, image) == "codex-cli 9.9.9-in-the-image"
    (spec,) = backend.specs
    assert spec.argv == ("codex", "--version") and spec.network.mode is NetworkMode.NONE
    assert not spec.mounts and not spec.secret_env
    without = InImage("sh: 1: codex: not found\n", 127)
    assert CodexAgent().version(without, image + "-without-codex") is None
    local = CodexAgent().version(LocalBackend(quiet=True), image + "-local")
    assert local == "codex-cli 0.0.0-on-the-host"


def test_usage_keeps_a_count_of_zero(tmp_path: Path) -> None:
    """Token counts of 0 (no cached input, no reasoning) are counts, not unknown."""
    usage = total_usage(
        [{"usage": {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 4,
                    "reasoning_output_tokens": 0}}]
    )  # fmt: skip
    assert usage is not None
    assert (usage.cached_input_tokens, usage.reasoning_tokens) == (0, 0)
    assert total_usage([{"usage": {"input_tokens": 3}}]).cached_input_tokens is None  # type: ignore[union-attr]


NOT_AN_AMOUNT = ["1e309", "-1e309", "NaN", "Infinity", "-2.0", "true", "1" + "0" * 400]
"""Costs a run can say in valid JSON for Python (1e309 reads as Infinity) that are no amount of
money: infinite, not a number, negative, a boolean, a whole number too large for a float."""


@pytest.mark.parametrize("said", NOT_AN_AMOUNT, ids=lambda said: said[:8])
def test_a_cost_a_run_says_that_is_no_amount_of_money_is_unknown(said: str, tmp_path: Path) -> None:
    """What an agent run cost is what its events (Claude Code, OpenCode) or its harness
    (mini_swe) say: an amount of money. Anything else leaves it unknown (``None``, counted as a
    run of unknown cost), never Infinity, money given back, $1 for ``true``, or a crash."""
    from real2simlab.agents.claude_code import ClaudeCodeAgent
    from real2simlab.agents.mini_swe import MiniSweAgent
    from real2simlab.agents.opencode import OpenCodeAgent

    events = tmp_path / "trajectory.jsonl"
    events.write_text(f'{{"cost_usd": 0.5}}\n{{"type": "result", "total_cost_usd": {said}}}\n')
    assert ClaudeCodeAgent().cost(events) is None
    events.write_text(f'{{"cost": 0.5}}\n{{"cost": {said}}}\n')
    assert OpenCodeAgent().cost(events) is None
    harness = MiniSweAgent(model="openai/priced-model")
    assert harness._cost(json.loads(f'{{"cost": {said}}}')) is None  # pyright: ignore[reportPrivateUsage]
    events.write_text('{"type": "result", "total_cost_usd": 0.25}\n')
    assert ClaudeCodeAgent().cost(events) == 0.25
    assert harness._cost({"cost": 2}) == 2.0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("said", NOT_AN_AMOUNT, ids=lambda said: said[:8])
def test_an_agent_record_never_holds_a_cost_that_is_no_amount_of_money(said: str) -> None:
    """An agent record's ``cost_usd`` is an amount of money or absent (unknown): whatever an
    agent's ``cost()`` returns, the record is written as strict JSON, never with Infinity or
    NaN, and a record an earlier Lab wrote with such a cost reads as one of unknown cost."""
    from real2simlab.agents.record import AgentRecord

    fields: dict[str, Any] = {
        "method": "claude_code", "agent": "claude", "instruction_template": "r2s-instruction/1.0",
        "instruction_sha256": "0" * 64, "task_id": "t", "task_hash": "h", "trial": 0,
        "image": {}, "backend": {}, "network": {}, "started_at": "2026-01-01T00:00:00Z",
        "duration_s": 1.0, "time_budget_s": 60.0, "exit_status": "completed",
    }  # fmt: skip

    def not_json(constant: str) -> None:
        raise AssertionError(f"{constant} is not JSON")

    record = AgentRecord.model_validate({**fields, "cost_usd": json.loads(said)})
    assert record.cost_usd is None
    assert "cost_usd" not in json.loads(record.dumps(), parse_constant=not_json)
    earlier = json.dumps(fields)[:-1] + f', "cost_usd": {said}}}'
    assert AgentRecord.model_validate_json(earlier).cost_usd is None
    assert AgentRecord.model_validate({**fields, "cost_usd": 1.5}).cost_usd == 1.5


@pytest.mark.parametrize("said", ['"Infinity"', '"inf"', '"nan"', '"-3"', '"1e309"'])
def test_an_agent_record_never_holds_a_cost_written_as_text_that_is_no_amount(said: str) -> None:
    """A record whose ``cost_usd`` is written as text (no Lab writes one: an edited record, or
    one an agent left in its log directory) is read as the number it holds, like any number:
    one that is no amount of money is unknown, never Infinity, NaN or a negative cost."""
    from real2simlab.agents.record import AgentRecord

    fields: dict[str, Any] = {
        "method": "claude_code", "agent": "claude", "instruction_template": "r2s-instruction/1.0",
        "instruction_sha256": "0" * 64, "task_id": "t", "task_hash": "h", "trial": 0,
        "image": {}, "backend": {}, "network": {}, "started_at": "2026-01-01T00:00:00Z",
        "duration_s": 1.0, "time_budget_s": 60.0, "exit_status": "completed",
    }  # fmt: skip
    record = AgentRecord.model_validate_json(json.dumps(fields)[:-1] + f', "cost_usd": {said}}}')
    assert record.cost_usd is None
    assert "cost_usd" not in json.loads(record.dumps())
    assert AgentRecord.model_validate({**fields, "cost_usd": json.loads(said)}).cost_usd is None
    assert AgentRecord.model_validate({**fields, "cost_usd": "0.5"}).cost_usd == 0.5


def test_a_budget_override_is_the_time_the_statement_states(bouncing_task: Path) -> None:
    """``time_budget_s`` changes the minutes a statement states, and only those."""
    from real2simlab.task import load_task
    from real2simlab.task.instruction import task_statement, with_time_budget

    statement = task_statement(load_task(bouncing_task))
    assert "You have about 20 minutes." in statement
    shorter = with_time_budget(statement, 1200, 300)
    assert "You have about 5 minutes." in shorter and "20 minutes" not in shorter
    assert shorter.replace("about 5 minutes", "about 20 minutes") == statement
    assert with_time_budget(statement, 1200, 1200) is statement
    video2world = "- you have 60 minutes; the package in `solution/` when you stop is scored"
    assert with_time_budget(video2world, 3600, 1200).startswith("- you have 20 minutes;")
    no_time = "Write a program. 20 minutes of video are shown."
    assert with_time_budget(no_time, 1200, 300) == no_time


def test_an_agent_is_told_its_own_budget(bouncing_task: Path, tmp_path: Path, local) -> None:
    """The workspace's TASK.md and the agent record state the budget the agent really has."""
    import hashlib

    from real2simlab.methods import get_method
    from real2simlab.methods.base import BuildContext
    from real2simlab.sandbox.base import ExecResult
    from real2simlab.task import load_task
    from real2simlab.task.instruction import task_statement

    task = load_task(bouncing_task)

    class Failing(type(local)):  # type: ignore[misc]
        def run(self, spec, log_path=None):  # type: ignore[no-untyped-def]
            return ExecResult(exit_code=1, timed_out=False, duration_s=0.0)

    method = get_method("fake_agent", mode="nothing", network="none", time_budget_s=300)
    context = BuildContext(
        task_id=task.id, task_hash="h", config=task.config,
        statement=task_statement(task), visible_dir=task.visible_dir, work_dir=tmp_path / "work",
        twin_dir=tmp_path / "twin", backend=Failing(quiet=True),
    )  # fmt: skip
    outcome = method.build(context)
    told = (tmp_path / "work" / "workspace" / "TASK.md").read_text(encoding="utf-8")
    assert "You have about 5 minutes." in told and "20 minutes" not in told
    assert outcome.agent_record is not None
    record = json.loads(outcome.agent_record.read_text(encoding="utf-8"))
    assert record["time_budget_s"] == 300
    assert record["instruction_sha256"] == hashlib.sha256(told.encode()).hexdigest()


# --- credentials never reach the disk ---------------------------------------------------------
# Every value below is a made-up canary, never a real credential.

SHORT_CANARY = "cnry-5f2e1d9a"
"""A short key (13 characters): no key format recognises it, only the injected value does."""


def _bootstrap_prints(tmp_path: Path, credential_file: str, printed: str) -> tuple[str, int]:
    """Inject ``credential_file`` as an agent's login file and let the agent print ``printed``
    on its output; return the output as written to disk and the redactions counted."""
    shown = tmp_path / "shown.txt"
    shown.write_text(printed)
    plan = {
        "argv": [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write(open(sys.argv[1]).read())",
            str(shown),
        ],
        "stdout": str(tmp_path / "out.log"),
        "stderr": str(tmp_path / "err.log"),
        "summary": str(tmp_path / "summary.json"),
        "secret_files": {str(tmp_path / "home" / "login"): "R2S_SECRET_FILE_0"},
        "unset_env": ["R2S_SECRET_FILE_0"],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    env = {**os.environ, "R2S_SECRET_FILE_0": base64.b64encode(credential_file.encode()).decode()}
    result = subprocess.run(
        [sys.executable, str(BOOTSTRAP), str(tmp_path / "plan.json")], env=env, check=False
    )
    assert result.returncode == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    return (tmp_path / "out.log").read_text(), int(summary["redactions"])


def _codex_auth(key: str) -> str:
    """A login file shaped like Codex's ``auth.json`` (``codex login`` indents it)."""
    tokens = {
        "id_token": "eyJ" + "a" * 40 + ".eyJ" + "b" * 40 + "." + "c" * 40,
        "refresh_token": "rt_" + "g" * 40,
    }
    return json.dumps(
        {"OPENAI_API_KEY": key, "tokens": tokens, "last_refresh": "2026-11-23T00:00:00Z"},
        indent=2,
    )


@pytest.mark.parametrize(
    ("credential_file", "printed"),
    [
        (
            json.dumps({"auth": {"password": SHORT_CANARY}, "other": "padding" * 35}),
            SHORT_CANARY + "\n",
        ),
        (_codex_auth(SHORT_CANARY), f'  "OPENAI_API_KEY": "{SHORT_CANARY}",\n'),
        (
            json.dumps({"user": "qa", "pin": SHORT_CANARY}, indent=2),
            json.dumps({"user": "qa", "pin": SHORT_CANARY}, indent=2) + "\n",
        ),
        (f"machine example.invalid login qa password {SHORT_CANARY}\n", SHORT_CANARY + "\n"),
        (f'[default]\nregion = "x"\napi_key = "{SHORT_CANARY}"\n', SHORT_CANARY + "\n"),
    ],
    ids=["json-password", "codex-auth-line", "small-file-in-full", "netrc", "toml-assignment"],
)
def test_bootstrap_redacts_short_values_of_injected_files(
    tmp_path: Path, credential_file: str, printed: str
) -> None:
    """A credential in an injected login file is redacted like an injected variable (from 6
    characters), whatever the file's size or format, also when the agent prints it line by line."""
    out, redactions = _bootstrap_prints(tmp_path, credential_file, printed)
    assert SHORT_CANARY not in out and REPLACEMENT in out
    assert redactions >= 1


def test_bootstrap_keeps_the_public_settings_of_injected_files(tmp_path: Path) -> None:
    """Settings injected through the same files (a model, an effort) are not credentials."""
    config = 'model = "gpt-test-codex"\nmodel_reasoning_effort = "medium"\n'
    out, redactions = _bootstrap_prints(tmp_path, config, "model gpt-test-codex, effort medium\n")
    assert out == "model gpt-test-codex, effort medium\n" and redactions == 0


def test_injected_credentials_never_reach_the_run_report_or_dataset(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent that writes its injected key into its solution and straight into the Lab's log
    directory, not only on its output (which bootstrap redacts): the run directory, the frozen
    twin, the HTML report and an exported dataset hold no copy of it, and the record counts it."""
    from real2simlab.agents import fake
    from real2simlab.agents.base import Credentials
    from real2simlab.orchestrate import parse_experiment, run_experiment
    from real2simlab.sandbox.local import LocalBackend
    from real2simlab.train import export_dataset

    canary = "cnry-agent-key-4f1c0b7e"
    leak = (
        "import json, os, subprocess, sys\n"
        "cli, workspace, logs = sys.argv[1:4]\n"
        "subprocess.run([sys.executable, cli, '--mode', 'physical', '--workspace', workspace],"
        " check=True)\n"
        "key = os.environ['R2S_TEST_AGENT_KEY']\n"
        "print('debug: the key is', key, flush=True)\n"
        "with open(os.path.join(workspace, 'solution', 'notes.txt'), 'w') as notes:\n"
        "    notes.write(key + '\\n')\n"
        "with open(os.path.join(logs, 'trajectory.jsonl'), 'a') as events:\n"
        "    events.write(json.dumps({'type': 'note', 'text': key}) + '\\n')\n"
    )
    command = ["python", "-c", leak, f"{fake.FAKE_DIR}/fake_agent_cli.py", "/workspace",
               "/r2s/logs"]  # fmt: skip
    monkeypatch.setattr(
        fake.FakeAgent,
        "credentials",
        lambda self, context: Credentials(env={"R2S_TEST_AGENT_KEY": canary}, description="test"),
    )
    monkeypatch.setattr(fake.FakeAgent, "command", lambda self, context, workspace: command)
    config = parse_experiment(
        {
            "name": "keys",
            "tasks": {"paths": [str(bouncing_task)]},
            "methods": [{"name": "fake_agent", "mode": "physical"}],
            "run": {"backend": "local"},
        }
    )
    run = tmp_path / "run"
    results = run_experiment(config, run, backend=LocalBackend(quiet=True), log=lambda line: None)
    assert results.methods[0].delivered_rate == 1.0  # the key's removal did not break the twin
    data = tmp_path / "data"
    export_dataset(data, runs=[run], allow_eval_tasks=True)
    assert list(run.glob("report/tasks/*.html")) and list(data.rglob("program/notes.txt"))
    holding = [
        str(path.relative_to(tmp_path))
        for path in sorted([*run.rglob("*"), *data.rglob("*")])
        if path.is_file() and canary.encode() in path.read_bytes()
    ]
    assert holding == []
    (record,) = run.glob("builds/*/fake_agent/trial-0/work/agent/agent_record.json")
    assert json.loads(record.read_text())["redactions"] >= 3  # output, solution and log
    (notes,) = run.glob("builds/*/fake_agent/trial-0/twin/program/notes.txt")
    assert notes.read_text() == REPLACEMENT + "\n"


def test_redacting_an_agents_files_stays_in_its_own_files(tmp_path: Path) -> None:
    """The host replaces injected values only in the agent's own regular files: a symbolic link
    to a file elsewhere is not followed, a hard link is not written through (the decoy it
    shares is unchanged), a named pipe is never opened, a program keeps its executable bit,
    and a value across the end of a read chunk is found. A file the agent made unreadable and
    a directory it made read-only or unlistable (000, 0300, 0400) are no hiding place, and get
    their permissions back. The value is a made-up canary; the decoy is this test's own file."""
    import stat

    from real2simlab.agents.redact import CHUNK, redact_files

    canary = "cnry-scan-0b4d2e"
    root = tmp_path / "work"
    solution = root / "solution"
    solution.mkdir(parents=True)
    decoy = tmp_path / "decoy.txt"
    decoy.write_text(f"decoy {canary}\n")
    (solution / "link.txt").symlink_to(decoy)
    (solution / "up").symlink_to(tmp_path, target_is_directory=True)
    os.link(decoy, solution / "hard.txt")
    os.mkfifo(solution / "pipe")
    program = solution / "run.sh"
    program.write_text(f"#!/bin/sh\necho {canary}\n")
    program.chmod(0o755)
    events = root / "agent" / "trajectory.jsonl"
    events.parent.mkdir()
    events.write_bytes(b"x" * (CHUNK - 5) + canary.encode() + b"\n")
    hidden = {"locked": 0o555, "sealed": 0o000, "blind": 0o300, "dark": 0o400}
    for name in hidden:
        (root / name).mkdir()
        (root / name / "notes.txt").write_text(canary)
    unreadable, plain = solution / "secret.txt", solution / "plain.txt"
    unreadable.write_text(canary)
    plain.write_text("nothing to replace")
    for path in (unreadable, plain):
        path.chmod(0)
    for name, mode in hidden.items():
        (root / name).chmod(mode)
    try:
        done = redact_files(root, [canary, "short"])
        modes = {name: stat.S_IMODE((root / name).stat().st_mode) for name in hidden}
        files = {stat.S_IMODE(path.stat().st_mode) for path in (unreadable, plain)}
    finally:
        for name in hidden:
            (root / name).chmod(0o755)
        for path in (unreadable, plain):
            path.chmod(0o644)
    assert (done.count, done.removed, done.left, done.errors) == (8, [], [], [])
    assert modes == hidden and files == {0}  # as the agent left them
    assert decoy.read_text() == f"decoy {canary}\n" and (solution / "link.txt").is_symlink()
    assert (solution / "hard.txt").read_text() == f"decoy {REPLACEMENT}\n"
    assert program.read_text() == f"#!/bin/sh\necho {REPLACEMENT}\n"
    assert os.access(program, os.X_OK)
    assert all((root / name / "notes.txt").read_text() == REPLACEMENT for name in hidden)
    assert unreadable.read_text() == REPLACEMENT and plain.read_text() == "nothing to replace"
    assert events.read_bytes() == b"x" * (CHUNK - 5) + REPLACEMENT.encode() + b"\n"
    assert redact_files(root, [canary]).count == 0  # nothing is left, and nothing is rewritten


def _deep_file(base: Path, content: str) -> Path:
    """A file with a short name whose path is just under PATH_MAX: it opens, but a longer name
    next to it (the copy that would replace it) does not fit."""
    limit = os.pathconf(str(base), "PC_PATH_MAX")
    directory = base
    while len(str(directory)) + 1 + 200 <= limit - 10:
        (directory / ("d" * 200)).mkdir()
        directory = directory / ("d" * 200)
    rest = limit - 10 - len(str(directory)) - 1
    if rest > 0:
        directory = directory / ("e" * rest)
        directory.mkdir()
    path = directory / "k"
    path.write_text(content)
    return path


def test_a_file_the_host_cannot_rewrite_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file holding an injected value that cannot be rewritten is removed (which needs no disk
    space), so no copy is left: when the disk fills while it is copied, the error is this
    machine's; when its path leaves no room for the copy's name, it is the agent's doing. A
    file that can be neither rewritten nor removed is said to be left. Made-up canary only."""
    import errno
    import importlib

    # the module (the package's ``redact`` is its function)
    redact = importlib.import_module("real2simlab.agents.redact")
    canary = "cnry-scan-full-61c7"
    root = tmp_path / "work"
    (root / "solution").mkdir(parents=True)
    (root / "agent").mkdir()
    (root / "solution" / "notes.txt").write_text("x" * 64 + canary + "\n")
    (root / "agent" / "trajectory.jsonl").write_text(json.dumps({"text": canary}) + "\n")
    copy = redact._copy_redacted

    class FullDisk:  # the disk fills after a few bytes of the copy
        def __init__(self, target: BinaryIO) -> None:
            self.target, self.written = target, 0

        def write(self, data: bytes) -> int:
            if self.written + len(data) > 8:
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
            self.written += len(data)
            return self.target.write(data)

    def full_disk(source: BinaryIO, target: BinaryIO, pattern: Any, longest: int) -> int:
        return copy(source, FullDisk(target), pattern, longest)  # type: ignore[arg-type]

    def denied(self: object, path: Path) -> None:
        raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))

    with monkeypatch.context() as patch:
        patch.setattr(redact, "_copy_redacted", full_disk)
        full = redact.redact_files(root, [canary])
    assert sorted(full.removed) == ["agent/trajectory.jsonl", "solution/notes.txt"]
    assert full.machine_error is not None and full.machine_error.errno == errno.ENOSPC
    assert full.left == [] and not any(root.rglob("*.txt")) and not any(root.rglob("*.jsonl"))
    assert not any(root.rglob(".r2s-redacting-*"))

    deep = _deep_file(root / "solution", canary + "\n")
    long_path = redact.redact_files(root, [canary])
    assert len(long_path.removed) == 1 and not deep.exists() and long_path.left == []
    assert long_path.machine_error is None  # ENAMETOOLONG: the agent's layout, not the machine
    assert long_path.errors[0].errno == errno.ENAMETOOLONG

    kept = root / "agent" / "kept.txt"
    kept.write_text(canary)
    os.link(kept, tmp_path / "other-name")  # emptying it would write through to that name
    with monkeypatch.context() as patch:
        patch.setattr(redact, "_copy_redacted", full_disk)
        patch.setattr(redact._Scan, "_unlink", denied)  # a directory the Lab cannot change
        left = redact.redact_files(root, [canary])
    assert left.left == ["agent/kept.txt"] and left.removed == []
    assert (tmp_path / "other-name").read_text() == canary  # never written through


def _bootstrap(
    tmp_path: Path,
    agent: str,
    *,
    stdout: str | None = None,
    limits: dict[str, int] | None = None,
    size_limit: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the bootstrap with the Python code ``agent`` as the agent's command, with smaller
    output ``limits`` or a file-size limit (``ulimit -f``); its files go to ``tmp_path``."""
    import resource

    plan = {
        "argv": [sys.executable, "-c", agent],
        "stdout": stdout or str(tmp_path / "out.log"),
        "stderr": str(tmp_path / "err.log"),
        "summary": str(tmp_path / "summary.json"),
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    runner = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('bootstrap', {str(BOOTSTRAP)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        + "".join(f"module.{name} = {value}\n" for name, value in (limits or {}).items())
        + "sys.exit(module.main(sys.argv[1]))\n"
    )

    def limited() -> None:
        if size_limit is not None:
            resource.setrlimit(resource.RLIMIT_FSIZE, (size_limit, size_limit))

    return subprocess.run(
        [sys.executable, "-c", runner, str(tmp_path / "plan.json")],
        capture_output=True, text=True, timeout=60, check=False, preexec_fn=limited,
    )  # fmt: skip


PRINTS = (
    "import sys\nfor i in range(4096):\n    sys.stdout.write(f'{i:05d} ' + 'x' * 1017 + '\\n')\n"
    "print('END')\n"
)
"""An agent printing 4 MiB of lines of 1 KiB."""


@pytest.mark.parametrize("bound", ["the bootstrap's limit", "ulimit -f"])
def test_the_bootstrap_keeps_a_bounded_part_of_an_agents_output(tmp_path: Path, bound: str) -> None:
    """An agent printing without end: its output file keeps the first and the last lines, whole
    and redacted, within the bootstrap's limit and within the machine's file-size limit, with a
    line saying what was left out; the agent goes on. Its trajectory grew until the disk or the
    file-size limit was full, and the bootstrap then stopped reading it (A3-05)."""
    most = 256 * 1024
    if bound == "ulimit -f":
        result = _bootstrap(tmp_path, PRINTS, size_limit=most)
    else:
        result = _bootstrap(
            tmp_path, PRINTS, limits={"OUTPUT_LIMIT": most, "OUTPUT_TAIL": most // 4}
        )
    assert result.returncode == 0, result.stderr
    out = (tmp_path / "out.log").read_bytes()
    assert len(out) <= most, len(out)
    lines = out.decode().splitlines()
    assert lines[0].startswith("00000 ") and lines[-1] == "END", (lines[0], lines[-1])
    assert all(len(line) == 1023 for line in lines if line[:1].isdigit())  # whole lines
    (left_out,) = [json.loads(line) for line in lines if line.startswith("{")]
    assert left_out["bytes"] > 0 and left_out["lines"] > 0
    assert json.loads((tmp_path / "summary.json").read_text())["exit_code"] == 0


def test_the_bootstrap_leaves_out_a_line_too_long_to_keep(tmp_path: Path) -> None:
    """A line longer than the bootstrap keeps is left out whole: never cut (a key across the
    cut would escape redaction) and never held in memory whole; what follows is kept (A3-05)."""
    agent = "import sys\nsys.stdout.write('z' * 3_000_000)\nsys.stdout.write('\\nEND\\n')\n"
    result = _bootstrap(tmp_path, agent, limits={"LINE_LIMIT": 64 * 1024})
    assert result.returncode == 0, result.stderr
    out = (tmp_path / "out.log").read_text()
    assert "zzzz" not in out and out.splitlines()[-1] == "END", out[-300:]
    (left_out,) = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    assert left_out["bytes"] == 3_000_001


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full (Linux)")
def test_the_bootstrap_reads_an_agents_output_to_the_end_when_it_cannot_write_it(
    tmp_path: Path,
) -> None:
    """The disk is full: the agent's output cannot be kept, but the bootstrap reads it to the
    end, so the agent finishes instead of blocking on a full pipe until its time limit, and the
    bootstrap says what it could not write (A3-05)."""
    agent = "for i in range(20000):\n    print('y' * 100)\nprint('done')\n"  # 2 MB: > a pipe
    result = _bootstrap(tmp_path, agent, stdout="/dev/full")
    assert result.returncode == 0, result.stderr
    assert "could not be written" in result.stderr, result.stderr
    assert json.loads((tmp_path / "summary.json").read_text())["exit_code"] == 0


def test_the_bootstrap_keeps_what_follows_a_line_it_left_out_after_the_note(
    tmp_path: Path,
) -> None:
    """A line that fits neither what is left of the beginning nor the end is left out, and the
    lines after it are part of the end, after the note saying what was left out: they were
    written back into the beginning, before the note (A, C, D, note instead of A, note, C, D)."""
    agent = "print('A' * 100)\nprint('B' * 60_000)\nprint('C' * 100)\nprint('D' * 100)\n"
    limits = {"OUTPUT_LIMIT": 64 * 1024, "OUTPUT_TAIL": 8 * 1024}
    result = _bootstrap(tmp_path, agent, limits=limits)
    assert result.returncode == 0, result.stderr
    lines = (tmp_path / "out.log").read_text().splitlines()
    assert [line[:1] for line in lines] == ["A", "{", "C", "D"], [line[:30] for line in lines]
    assert json.loads(lines[1])["bytes"] == 60_001


NUMBERED = "f'{{i:05d}} ' + {letter!r} * 94"
"""The Python expression of a numbered line (``i``) of 100 bytes made of ``letter``."""


def _whole_and_bounded(text: str, letter: str, count: int, most: int) -> None:
    """``text`` keeps the first and the last of ``count`` numbered lines of ``letter``, whole,
    within ``most`` bytes, with a line saying what was left out between them."""
    assert len(text.encode()) <= most, len(text.encode())
    lines = [line for line in text.splitlines() if line[:5].isdigit()]
    assert lines[0] == "00000 " + letter * 94 and lines[-1] == f"{count - 1:05d} " + letter * 94
    assert all(len(line) == 100 for line in lines), "a line was cut"
    assert "[r2s]" in text and "left out" in text, text[:200]


FAKE_HARBOR = '''#!{python}
"""Harbor, as the harbor method sees it: `--version`, and `run ... -o JOBS --job-name NAME`,
which leaves a trial whose agent kept a long log (and links), and prints a lot itself."""
import json, os, sys
from pathlib import Path

if sys.argv[1:] == ["--version"]:
    print("0.0-test")
    sys.exit(0)
args = sys.argv[1:]
trial = Path(args[args.index("-o") + 1]) / args[args.index("--job-name") + 1] / "trial-1"
(trial / "artifacts" / "workspace").mkdir(parents=True)
agent = trial / "agent"
agent.mkdir()
(agent / "a-small.txt").write_text("small\\n")
with open(agent / "big.log", "w") as log:
    for i in range(20_000):
        log.write({line_x} + "\\n")
os.symlink({decoy!r}, agent / "linked.txt")
os.symlink(os.path.dirname({decoy!r}), agent / "linked-dir")
(trial / "result.json").write_text(json.dumps({{}}))
for i in range(30_000):
    print({line_o})
    print({line_e}, file=sys.stderr)
'''


def test_what_harbor_prints_and_its_trial_are_kept_within_limits(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local
) -> None:
    """The harbor method runs Harbor on the host: what Harbor prints (``harbor.log``) and the
    copy of its trial (``agent/trial``) keep their first and last whole lines within the limits
    of an agent's output, and a link in the trial is not followed. Harbor's whole output was
    read into memory and written out, and its trial copied as it was, links followed (A3-05)."""
    from real2simlab.agents import harbor as module
    from real2simlab.methods import get_method
    from real2simlab.methods.base import BuildContext
    from real2simlab.task import load_task
    from real2simlab.task.instruction import task_statement

    for name, value in (
        ("LOG_LIMIT_BYTES", 256 * 1024), ("OUTPUT_LIMIT", 512 * 1024), ("OUTPUT_TAIL", 64 * 1024)
    ):  # fmt: skip
        monkeypatch.setattr(module, name, value, raising=False)  # the test writes little
    decoy = tmp_path / "elsewhere" / "decoy.txt"
    decoy.parent.mkdir()
    decoy.write_text("DECOY\n")
    harbor = tmp_path / "harbor"
    harbor.write_text(
        FAKE_HARBOR.format(
            python=sys.executable, decoy=str(decoy),
            **{f"line_{letter}": NUMBERED.format(letter=letter) for letter in "xoe"},
        )
    )  # fmt: skip
    harbor.chmod(0o755)
    task = load_task(bouncing_task)
    context = BuildContext(
        task_id=task.id, task_hash="h", config=task.config,
        statement=task_statement(task), visible_dir=task.visible_dir, work_dir=tmp_path / "work",
        twin_dir=tmp_path / "twin", backend=local,
    )  # fmt: skip
    get_method("harbor", harbor=str(harbor), agent="oracle").build(context)
    agent = tmp_path / "work" / "agent"
    log = (agent / "harbor.log").read_text()
    printed, _, notes = log.partition("[r2s] agent/trial:")  # then what the copy left out
    assert len(printed.encode()) <= 256 * 1024, len(printed.encode())
    stderr = printed.index("00000 " + "e" * 94)  # Harbor's stdout, then its stderr
    _whole_and_bounded(printed[:stderr], "o", 30_000, 128 * 1024)
    _whole_and_bounded(printed[stderr:], "e", 30_000, 128 * 1024)
    trial = agent / "trial"
    assert (trial / "a-small.txt").read_text() == "small\n"
    _whole_and_bounded((trial / "big.log").read_text(), "x", 20_000, 512 * 1024)
    assert sorted(path.name for path in trial.iterdir()) == ["a-small.txt", "big.log"]
    assert "linked.txt" in notes and "linked-dir" in notes, notes
    assert not any("DECOY" in path.read_text() for path in trial.iterdir())


FAKE_HARNESS = '''#!{python}
"""The mini-swe-agent harness, as the mini_swe method sees it: `-c ...` (its version), and
`DRIVER CONFIG`, which keeps a long trajectory, prints a lot on stderr and its summary last."""
import json, sys
from pathlib import Path

if sys.argv[1] == "-c":
    print("0.0-test")
    sys.exit(0)
config = json.loads(Path(sys.argv[2]).read_text())
with open(config["trajectory"], "w") as trajectory:
    trajectory.write("{{\\n")
    for i in range(20_000):
        trajectory.write("  " + json.dumps({line_t}) + ": 0,\\n")
    trajectory.write('  "exit_status": "Submitted"\\n}}\\n')
for i in range(30_000):
    print({line_e}, file=sys.stderr)
print(json.dumps({{"exit_status": "Submitted", "api_calls": 1, "cost": 0.0}}))
'''


def test_what_the_mini_swe_harness_prints_and_its_trajectory_are_kept_within_limits(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local
) -> None:
    """mini_swe runs its harness on the host: what it prints on stderr (``stderr.log``) and its
    ``trajectory.json`` keep their first and last whole lines within the limits of an agent's
    output, and the summary it prints last is still read. Both were read whole into memory and
    written out whole (A3-05)."""
    from real2simlab.agents import mini_swe as module
    from real2simlab.methods import get_method
    from real2simlab.methods.base import BuildContext
    from real2simlab.task import load_task
    from real2simlab.task.instruction import task_statement

    for name, value in (
        ("LOG_LIMIT_BYTES", 256 * 1024), ("OUTPUT_LIMIT", 512 * 1024), ("OUTPUT_TAIL", 64 * 1024)
    ):  # fmt: skip
        monkeypatch.setattr(module, name, value, raising=False)  # the test writes little
    harness = tmp_path / "harness"
    harness.write_text(
        FAKE_HARNESS.format(
            python=sys.executable,
            **{f"line_{letter}": NUMBERED.format(letter=letter) for letter in "te"},
        )
    )
    harness.chmod(0o755)
    task = load_task(bouncing_task)
    context = BuildContext(
        task_id=task.id, task_hash="h", config=task.config,
        statement=task_statement(task), visible_dir=task.visible_dir, work_dir=tmp_path / "work",
        twin_dir=tmp_path / "twin", backend=local,
    )  # fmt: skip
    method = get_method(
        "mini_swe", model="test/none", harness_python=str(harness), api_key_env="R2S_TEST_KEY"
    )
    method.build(context)
    agent = tmp_path / "work" / "agent"
    _whole_and_bounded((agent / "stderr.log").read_text(), "e", 30_000, 256 * 1024)
    trajectory = (agent / "trajectory.json").read_text()
    assert len(trajectory.encode()) <= 512 * 1024, len(trajectory.encode())
    lines = trajectory.splitlines()
    assert lines[0] == "{" and lines[-2:] == ['  "exit_status": "Submitted"', "}"], lines[-2:]
    kept = [line for line in lines if line.startswith('  "0') or line.startswith('  "1')]
    assert kept[0].startswith('  "00000 ') and kept[-1].startswith('  "19999 ')
    assert all(len(line) == len(kept[0]) for line in kept), "a line was cut"
    assert "[r2s]" in trajectory and "left out" in trajectory
    record = json.loads((agent / "agent_record.json").read_text())
    assert record["exit_status"] == "completed", record


def test_the_fake_agents_bundle_goes_when_the_process_ends(tmp_path: Path) -> None:
    """``r2s demo`` and every run of ``fake_agent`` left an ``r2s-fake-agent-*`` directory in
    the temporary directory that nothing removed (thousands on a shared machine): it is
    removed when the process that made it ends."""
    code = "from real2simlab.agents.fake import FakeAgent; print(FakeAgent().bundle())"
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
        env={**os.environ, "TMPDIR": str(tmp_path)}, check=False, stdin=subprocess.DEVNULL,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert Path(done.stdout.strip()).parent == tmp_path and not list(tmp_path.iterdir())
