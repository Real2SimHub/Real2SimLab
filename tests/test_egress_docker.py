"""The agents' egress allowlist in Docker: an internal network whose only way out is the proxy."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from .conftest import docker_command, docker_image_ready

pytestmark = pytest.mark.docker

CLIENT = r"""
import json, socket, sys
proxy_host, proxy_port = sys.argv[1], int(sys.argv[2])
out = {}
try:  # no route out of the internal network except the proxy
    socket.create_connection(("1.1.1.1", 443), timeout=3).close()
    out["direct"] = "connected"
except OSError as error:
    out["direct"] = type(error).__name__
for host in ("allowed.r2s.invalid", "denied.r2s.invalid"):
    with socket.create_connection((proxy_host, proxy_port), timeout=10) as s:
        s.sendall(f"CONNECT {host}:443 HTTP/1.1\r\nHost: {host}\r\n\r\n".encode())
        out[host] = s.recv(200).decode(errors="replace").split("\r\n")[0]
print(json.dumps(out))
"""


def test_allowlist_proxy_in_docker(tmp_path: Path) -> None:
    if not docker_image_ready():
        pytest.skip("Docker or the r2s-python image is not available")
    docker = docker_command()
    assert docker is not None
    from real2simlab.sandbox.egress import PROXY_ALIAS, PROXY_PORT, EgressProxy, domains

    image = "ghcr.io/real2simhub/r2s-python:1.0"
    proxy = EgressProxy(
        image=image, allowlist=["allowed.r2s.invalid"], log_dir=tmp_path, docker=docker
    )
    with proxy as policy:
        assert policy.docker_network and policy.proxy_url == f"http://{PROXY_ALIAS}:{PROXY_PORT}"
        result = subprocess.run(
            [docker, "run", "--rm", "--network", policy.docker_network, image,
             "python", "-c", CLIENT, PROXY_ALIAS, str(PROXY_PORT)],
            capture_output=True, text=True, timeout=120, check=False,
        )  # fmt: skip
        network = policy.docker_network
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["direct"] != "connected"
    assert " 403" in out["denied.r2s.invalid"]
    # The allowed host does not resolve, so the proxy answers with a gateway error, not 403.
    assert " 403" not in out["allowed.r2s.invalid"]
    assert policy.log_path is not None
    counts = domains(policy.log_path)
    assert counts["denied.r2s.invalid"] == {"allowed": 0, "denied": 1}
    assert counts["allowed.r2s.invalid"]["allowed"] == 1
    # The network and the proxy container are removed afterwards.
    inspect = subprocess.run([docker, "network", "inspect", network], capture_output=True)
    assert inspect.returncode != 0
