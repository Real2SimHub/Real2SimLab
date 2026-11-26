"""Every flag an agent adapter passes exists in that CLI's ``--help``, in the pinned image.

No model is called: each CLI only prints its help, offline. Skipped for images that are not
built (``r2s env build <image>``).
"""

from __future__ import annotations

import re
import subprocess
from typing import Any

import pytest

from real2simlab.methods import get_method

from .conftest import docker_command, docker_image_ready

pytestmark = pytest.mark.docker

CASES: dict[str, tuple[str, dict[str, Any], tuple[str, ...]]] = {
    "codex": (
        "ghcr.io/real2simhub/r2s-agent-codex:1.0",
        {"model": "some-model", "reasoning_effort": "medium"},
        ("codex", "exec", "--help"),
    ),
    "claude_code": (
        "ghcr.io/real2simhub/r2s-agent-claude-code:1.0",
        {"model": "some-model", "max_turns": 30},
        ("claude", "--help"),
    ),
    "gemini_cli": (
        "ghcr.io/real2simhub/r2s-agent-gemini-cli:1.0",
        {"model": "some-model"},
        ("gemini", "--help"),
    ),
    "opencode": (
        "ghcr.io/real2simhub/r2s-agent-opencode:1.0",
        {"model": "openai/some-model"},
        ("opencode", "run", "--help"),
    ),
}


HIDDEN: dict[str, dict[str, tuple[str, ...]]] = {
    # Flags the CLI accepts but leaves out of --help, with an offline probe that parses them.
    # The probe passes when the CLI gets past option parsing (it then fails for lack of a key).
    "claude_code": {
        "--max-turns": ("claude", "-p", "--max-turns", "3", "--output-format", "stream-json",
                        "--verbose"),
    },
}  # fmt: skip
UNKNOWN_OPTION = ("unknown option", "Unknown argument", "unrecognized argument")


def flags(argv: list[str]) -> list[str]:
    """The options in a command line: ``--name`` and ``-x`` (``=value`` removed)."""
    return sorted(
        {item.split("=")[0] for item in argv if re.fullmatch(r"--?[A-Za-z][\w-]*(=.*)?", item)}
    )


def documented(flag: str, help_text: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text) is not None


def test_hidden_probe_detects_unknown_options() -> None:
    assert any(m in "error: unknown option '--x'" for m in UNKNOWN_OPTION)


def test_flag_extraction() -> None:
    argv = ["codex", "exec", "--json", "-C", "/w", "-c", 'x="y"', "--color", "never", "-", "a-b"]
    assert flags(argv) == ["--color", "--json", "-C", "-c"]
    assert documented("-C", "  -C, --cd <DIR>") and not documented("--cd", "  --cdx")


@pytest.mark.parametrize("method", sorted(CASES))
def test_adapter_flags_exist_in_the_cli(method: str) -> None:
    image, options, help_argv = CASES[method]
    if not docker_image_ready(image):
        pytest.skip(f"{image} is not built")
    docker = docker_command()
    assert docker is not None
    agent: Any = get_method(method, **options)
    argv = agent.command(None, None)
    result = subprocess.run(
        [docker, "run", "--rm", "--network", "none", image, *help_argv],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    help_text = result.stdout + result.stderr
    assert result.returncode == 0, help_text[-2000:]
    hidden = HIDDEN.get(method, {})
    missing = [f for f in flags(argv) if not documented(f, help_text) and f not in hidden]
    assert not missing, f"{method}: {missing} not in `{' '.join(help_argv)}`"
    for flag, probe in hidden.items():
        if flag not in argv:
            continue
        parsed = subprocess.run(
            [docker, "run", "--rm", "-i", "--network", "none", "-e", "IS_SANDBOX=1", image, *probe],
            input="hi\n", capture_output=True, text=True, timeout=180, check=False,
        )  # fmt: skip
        output = parsed.stdout + parsed.stderr
        assert not any(marker in output for marker in UNKNOWN_OPTION), (flag, output[-500:])
        if documented(flag, help_text):
            pytest.fail(f"{method}: {flag} is documented now; remove it from HIDDEN")
