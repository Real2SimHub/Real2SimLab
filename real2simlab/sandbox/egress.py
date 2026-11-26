"""Running an egress proxy for one agent container (Docker backend).

``EgressProxy`` creates an internal Docker network (no route to the internet), starts the
proxy container on the default network, attaches it to the internal network under the alias
``r2s-proxy``, and yields the ``NetworkPolicy`` the agent container uses. Everything is removed
on exit. The proxy's connection log stays in ``log_dir``.
"""

from __future__ import annotations

import subprocess
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

from real2simlab.errors import InfrastructureError
from real2simlab.sandbox.base import NetworkMode, NetworkPolicy, host_user
from real2simlab.sandbox.docker import bind_mount

PROXY_SCRIPT = Path(__file__).with_name("egress_proxy.py")
PROXY_ALIAS = "r2s-proxy"
PROXY_PORT = 3128
LOG_NAME = "egress.jsonl"


class EgressProxy:
    """Context manager: an allowlisting proxy and the network policy that uses it."""

    def __init__(
        self,
        *,
        image: str,
        allowlist: Sequence[str],
        log_dir: Path,
        docker: str = "docker",
        log_only: bool = False,
    ) -> None:
        self.image = image
        self.allowlist = tuple(allowlist)
        self.log_dir = Path(log_dir)
        self.docker = docker
        self.log_only = log_only
        suffix = uuid.uuid4().hex[:10]
        self.network = f"r2s-egress-{suffix}"
        self.container = f"r2s-proxy-{suffix}"

    def _call(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [self.docker, *args],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        if check and result.returncode != 0:
            raise InfrastructureError(
                f"docker {' '.join(args[:2])} failed: {result.stderr.strip()}"
            )
        return result

    def __enter__(self) -> NetworkPolicy:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / LOG_NAME).touch()
        self._call("network", "create", "--internal", "--label", "real2simlab=1", self.network)
        command = [
            "run", "-d", "--rm", "--name", self.container, "--label", "real2simlab=1",
            "--user", host_user(), "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--read-only", "--memory", "256m", "--pids-limit", "256",
            "--mount", bind_mount(PROXY_SCRIPT, "/r2s/egress_proxy.py", read_only=True),
            "--mount", bind_mount(self.log_dir, "/r2s/logs"),
            self.image, "python", "/r2s/egress_proxy.py", "--port", str(PROXY_PORT),
            "--log", f"/r2s/logs/{LOG_NAME}",
        ]  # fmt: skip
        if self.log_only:
            command.append("--log-only")
        for host in self.allowlist:
            command += ["--allow", host]
        try:
            self._call(*command)
            self._call("network", "connect", "--alias", PROXY_ALIAS, self.network, self.container)
            self._wait_ready()
        except BaseException:
            self._cleanup()
            raise
        return NetworkPolicy(
            mode=NetworkMode.ALLOWLIST if not self.log_only else NetworkMode.OPEN,
            allowlist=self.allowlist,
            log_path=self.log_dir / LOG_NAME,
            proxy_url=f"http://{PROXY_ALIAS}:{PROXY_PORT}",
            docker_network=self.network,
        )

    def _wait_ready(self, timeout: float = 30) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            logs = self._call("logs", self.container, check=False)
            if "r2s-egress-proxy ready" in logs.stdout:
                return
            time.sleep(0.2)
        raise InfrastructureError(f"the egress proxy did not start within {timeout:g} s")

    def _cleanup(self) -> None:
        self._call("rm", "-f", self.container, check=False)
        self._call("network", "rm", self.network, check=False)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._cleanup()


def domains(log_path: Path) -> dict[str, dict[str, int]]:
    """Hosts in a proxy log: allowed and denied connection counts per host."""
    import json

    counts: dict[str, dict[str, int]] = {}
    path = Path(log_path)
    if not path.is_file():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        host = str(entry.get("host", ""))
        bucket = counts.setdefault(host, {"allowed": 0, "denied": 0})
        bucket["allowed" if entry.get("allowed") else "denied"] += 1
    return counts
