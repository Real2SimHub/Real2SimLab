"""Runs mini-swe-agent for the ``mini_swe`` method. Never imports Real2SimLab.

Usage: ``python mini_swe_driver.py CONFIG.json``, with the Python of an environment that has
mini-swe-agent (>= 2.0) installed. The harness talks to the model; every shell command the
agent issues runs in the sandbox, through ``sandbox_prefix``, with ``sandbox_env`` as its whole
environment, so the model's API key (in this process's environment) never reaches a command.

The last line on stdout is a JSON summary: exit status, model calls, cost, how many times the
model answered, the error that ended the run (``None`` when the agent ended it) with its class
and the classes it derives from and the HTTP status the model's API answered with (for the host
to tell an API that cannot be used from one that refused this request), and why the sandbox
stopped running commands (``None`` while it ran them).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from minisweagent.agents.default import DefaultAgent  # pyright: ignore[reportMissingImports]
from minisweagent.environments.local import (  # pyright: ignore[reportMissingImports]
    LocalEnvironment,
)
from minisweagent.exceptions import FormatError  # pyright: ignore[reportMissingImports]
from minisweagent.models import get_model  # pyright: ignore[reportMissingImports]

PROBE_TIMEOUT_S = 60


class SandboxStopped(Exception):
    """The sandbox runs no command any more, not even ``true``: the agent cannot go on."""


class SandboxedEnvironment(LocalEnvironment):
    """Runs each command as ``[*prefix, command]`` with a fixed, secret-free environment."""

    def __init__(
        self, prefix: list[str], sandbox_env: dict[str, str], host_cwd: str, **kwargs: Any
    ) -> None:
        super().__init__(cwd=host_cwd, **kwargs)
        self.prefix = prefix
        self.sandbox_env = sandbox_env
        self.host_cwd = host_cwd
        self.sandbox_error: str | None = None

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        limit = timeout or self.config.timeout
        try:
            result = subprocess.run(
                [*self.prefix, command],
                cwd=self.host_cwd,
                env=self.sandbox_env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=limit + 30,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
            )
            output = {
                "output": result.stdout + result.stderr,
                "returncode": result.returncode,
                "exception_info": "",
            }
        except subprocess.TimeoutExpired as error:
            output = {
                "output": str(error.output or ""),
                "returncode": -1,
                "exception_info": f"The command timed out after {limit} s.",
            }
        if output["returncode"] != 0:
            # The command failed, or its sandbox did: then every later command fails the same
            # way, so the agent is stopped instead of being shown that error at every step.
            problem = self.sandbox_problem()
            if problem is not None:
                self.sandbox_error = problem
                raise SandboxStopped(problem)
        self._check_finished(output)
        return output

    def sandbox_problem(self) -> str | None:
        """``None`` while the sandbox runs ``true``; otherwise why it does not."""
        try:
            result = subprocess.run(
                [*self.prefix, "true"],
                cwd=self.host_cwd,
                env=self.sandbox_env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=PROBE_TIMEOUT_S,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return f"`true` did not finish in {PROBE_TIMEOUT_S} s"
        except OSError as error:
            return f"`true` could not be started: {error}"
        if result.returncode == 0:
            return None
        said = (result.stdout + result.stderr).strip()[-300:]
        return said or f"`true` exited {result.returncode}"


class CountingAgent(DefaultAgent):
    """``DefaultAgent`` that counts the model's answers (one in the wrong format included):
    a run that ended without any answer did nothing of the agent's."""

    answers = 0

    def query(self) -> dict:
        try:
            message = super().query()
        except FormatError:
            self.answers += 1
            raise
        self.answers += 1
        return message


def describe(exception: BaseException) -> dict[str, Any]:
    """The error that ended the run: its text, its class and the classes it derives from (the
    model API's errors are classes such as ``AuthenticationError`` or
    ``ContextWindowExceededError``), and the HTTP status the API answered with, if any."""
    status = getattr(exception, "status_code", None)
    if not isinstance(status, int) or isinstance(status, bool):
        status = getattr(getattr(exception, "response", None), "status_code", None)
    return {
        "error": f"{type(exception).__name__}: {exception}",
        "error_types": [
            kind.__name__
            for kind in type(exception).__mro__
            if kind not in (BaseException, Exception, object)
        ],
        "error_status": status
        if isinstance(status, int) and not isinstance(status, bool)
        else None,
    }


def main(config_path: str) -> int:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    environment: SandboxedEnvironment | None = None
    agent: CountingAgent | None = None
    error: dict[str, Any] = {"error": None, "error_types": [], "error_status": None}
    try:
        defaults = yaml.safe_load(
            (
                Path(__import__("minisweagent").__file__).parent / "config" / "mini_textbased.yaml"
            ).read_text()
        )
        model_config: dict[str, Any] = {
            **defaults.get("model", {}),
            "model_name": config["model_name"],
            **config.get("model", {}),
        }
        kwargs = dict(model_config.get("model_kwargs", {}))
        kwargs.update(config.get("model_kwargs", {}))
        key_env = config.get("api_key_env")
        if key_env and os.environ.get(key_env):
            kwargs["api_key"] = os.environ[key_env]
        model_config["model_kwargs"] = kwargs
        if config.get("model_class"):
            model_config["model_class"] = config["model_class"]
        model = get_model(config["model_name"], model_config)
        environment = SandboxedEnvironment(
            config["sandbox_prefix"],
            config["sandbox_env"],
            config["host_cwd"],
            timeout=config["command_timeout_s"],
        )
        agent = CountingAgent(
            model,
            environment,
            system_template=defaults["agent"]["system_template"],
            instance_template=config["instance_template"],
            step_limit=config["step_limit"],
            cost_limit=config["cost_limit"],
            wall_time_limit_seconds=config["wall_time_limit_s"],
            output_path=Path(config["trajectory"]),
        )
        result = agent.run(config["task"])
        status = result.get("exit_status", "")
    except Exception as exception:  # the harness must always report
        error = describe(exception)
        status = error["error"]
    summary = {
        "exit_status": status,
        "api_calls": agent.n_calls if agent else 0,
        "cost": agent.cost if agent else 0.0,
        "model_answers": agent.answers if agent else 0,
        **error,
        "sandbox_error": environment.sandbox_error if environment else None,
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
