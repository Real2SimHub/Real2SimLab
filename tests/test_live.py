"""Real agent runs. Marked ``live``: never run by default (``pytest -m live`` to run them).

They need Docker, the ``r2s-agent-codex`` image, and the host's Codex login or provider key,
and they spend model calls. One test that is not live checks how their settings reach each
agent, without a model.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from real2simlab.orchestrate import parse_experiment, run_experiment

from .conftest import docker_image_ready


@pytest.mark.live
@pytest.mark.docker
def test_codex_builds_a_twin_on_a_synthetic_task(
    synthetic_tasks: Path, tmp_path: Path, docker_backend: Any
) -> None:
    if not docker_image_ready("ghcr.io/real2simhub/r2s-agent-codex:1.0"):
        pytest.skip("r2s env build r2s-agent-codex")
    config = parse_experiment(
        {
            "name": "live-codex",
            "tasks": {
                "paths": [str(synthetic_tasks)],
                "include": ["synthetic/bouncing_ball/calibrated"],
            },
            "methods": [{"name": "codex", "reasoning_effort": "medium"}],
            "run": {"backend": "docker"},
            "budget": {"max_agent_runs": 1},
        }
    )
    results = run_experiment(config, tmp_path / "run", backend=docker_backend)
    record = next((tmp_path / "run" / "builds").rglob("agent_record.json"))
    assert record.is_file()
    assert results.methods[0].results == 1


LIVE_AGENTS: dict[str, tuple[str | None, tuple[str, ...], dict[str, Any]]] = {
    # method: (image, credential variables (any one), method options)
    "claude_code": (
        "ghcr.io/real2simhub/r2s-agent-claude-code:1.0",
        ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
        {},
    ),
    "gemini_cli": (
        "ghcr.io/real2simhub/r2s-agent-gemini-cli:1.0",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        {},
    ),
    "opencode": ("ghcr.io/real2simhub/r2s-agent-opencode:1.0", ("OPENAI_API_KEY",), {}),
    "mini_swe": (None, ("OPENAI_API_KEY",), {}),
}
"""Set R2S_LIVE_MODEL (and R2S_LIVE_BASE_URL for a compatible endpoint) to choose the model."""

ENDPOINTS: dict[str, tuple[str, str]] = {
    # method: (its endpoint option, the variable it is taken from)
    "mini_swe": ("api_base", "R2S_LIVE_BASE_URL"),
    "opencode": ("base_url", "R2S_LIVE_BASE_URL"),
    "claude_code": ("base_url", "ANTHROPIC_BASE_URL"),
}
"""R2S_LIVE_BASE_URL is an OpenAI-compatible endpoint: both ``mini_swe`` and ``opencode`` run
R2S_LIVE_MODEL, a LiteLLM name such as ``openai/<model>``, there (``opencode`` as
``custom/<model>``). ``claude_code`` speaks Anthropic's API: its endpoint is the host's
``ANTHROPIC_BASE_URL``, which it would otherwise refuse to leave out. ``gemini_cli`` takes none."""


def live_options(method: str, env: Mapping[str, str]) -> dict[str, Any] | str:
    """The options of ``method``'s smoke test in the environment ``env``, or why it is skipped
    (before its image is looked at)."""
    _, keys, options = LIVE_AGENTS[method]
    if not any(env.get(key) for key in keys):
        return f"set one of {', '.join(keys)}"
    options = dict(options)
    if env.get("R2S_LIVE_MODEL"):
        options["model"] = env["R2S_LIVE_MODEL"]
    elif method in {"mini_swe", "opencode"}:
        return "set R2S_LIVE_MODEL (e.g. openai/<model>)"
    option, variable = ENDPOINTS.get(method, ("", ""))
    if option and env.get(variable):
        options[option] = env[variable]
    return options


def test_live_settings_reach_each_agent_the_way_it_takes_them() -> None:
    """One R2S_LIVE_MODEL and R2S_LIVE_BASE_URL run ``mini_swe`` and ``opencode`` on the same
    endpoint, and are options every agent takes (no model is called)."""
    from real2simlab.methods import get_method

    env = {
        "R2S_LIVE_MODEL": "openai/qa-model",
        "R2S_LIVE_BASE_URL": "https://gateway.example.invalid/v1",
        "OPENAI_API_KEY": "x",
        "ANTHROPIC_AUTH_TOKEN": "x",
        "ANTHROPIC_BASE_URL": "https://gateway.example.invalid/anthropic",
        "GEMINI_API_KEY": "x",
    }
    chosen = {method: live_options(method, env) for method in LIVE_AGENTS}
    assert chosen["mini_swe"] == {"model": "openai/qa-model", "api_base": env["R2S_LIVE_BASE_URL"]}
    assert chosen["opencode"] == {"model": "openai/qa-model", "base_url": env["R2S_LIVE_BASE_URL"]}
    assert chosen["claude_code"]["base_url"] == env["ANTHROPIC_BASE_URL"]  # type: ignore[index]
    assert "base_url" not in chosen["gemini_cli"]
    for method, options in chosen.items():
        assert isinstance(options, dict)
        get_method(method, reasoning_effort="medium", **options)  # valid options
    opencode: Any = get_method("opencode", **chosen["opencode"])  # type: ignore[arg-type]
    argv = opencode.command(None, None)
    assert argv[argv.index("-m") + 1] == "custom/qa-model"  # the provider it registers there
    assert "set R2S_LIVE_MODEL" in str(live_options("opencode", {"OPENAI_API_KEY": "x"}))


@pytest.mark.live
@pytest.mark.docker
@pytest.mark.parametrize("method", sorted(LIVE_AGENTS))
def test_agent_smoke(
    method: str, synthetic_tasks: Path, tmp_path: Path, docker_backend: Any
) -> None:
    """One synthetic task with a real model: the run completes and is recorded. A missing key,
    image or model setting skips the test; whether a twin is delivered is up to the model."""
    import os

    image, _, _ = LIVE_AGENTS[method]
    options = live_options(method, os.environ)
    if isinstance(options, str):
        pytest.skip(options)
    if image is not None and not docker_image_ready(image):
        pytest.skip(f"r2s env build {image.split('/')[-1].split(':')[0]}")
    config = parse_experiment(
        {
            "name": f"live-{method}",
            "tasks": {
                "paths": [str(synthetic_tasks)],
                "include": ["synthetic/bouncing_ball/calibrated"],
            },
            "methods": [{"name": method, "reasoning_effort": "medium", **options}],
            "run": {"backend": "docker"},
            "budget": {"max_agent_runs": 1},
        }
    )
    run_experiment(config, tmp_path / "run", backend=docker_backend)
    record = next((tmp_path / "run" / "builds").rglob("agent_record.json"))
    data = record.read_text()
    assert '"exit_status"' in data and '"agent_version"' in data
