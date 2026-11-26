"""An HTTP(S) egress proxy that allows only listed domains and logs every connection.

Standard library only; it runs in its own container (any image with Python 3.11), attached to
an internal Docker network that the agent's container is confined to, and to the default
network that reaches the internet.

Usage::

    python egress_proxy.py --port 3128 --log /logs/egress.jsonl --allow api.openai.com \\
        --allow '*.example.com'
    python egress_proxy.py --port 3128 --log /logs/egress.jsonl --log-only

HTTPS goes through ``CONNECT host:port`` (the proxy sees only the host name, never content);
plain HTTP requests in absolute form are forwarded. Each connection is logged as one JSON line
with the host, port, whether it was allowed, the bytes in each direction and the duration.
``--log-only`` allows everything and is used to discover which domains an agent needs.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fnmatch
import json
import sys
import time
from urllib.parse import urlsplit

READY = "r2s-egress-proxy ready"


class Policy:
    def __init__(self, allow: list[str], log_only: bool) -> None:
        self.allow = [pattern.lower().rstrip(".") for pattern in allow]
        self.log_only = log_only

    def allowed(self, host: str) -> bool:
        if self.log_only:
            return True
        host = host.lower().rstrip(".")
        for pattern in self.allow:
            if pattern.startswith("*."):
                if host == pattern[2:] or host.endswith(pattern[1:]):
                    return True
            elif fnmatch.fnmatchcase(host, pattern):
                return True
        return False


class Proxy:
    def __init__(self, policy: Policy, log_path: str) -> None:
        self.policy = policy
        self.log = open(log_path, "a", encoding="utf-8")  # noqa: SIM115

    def record(self, **entry: object) -> None:
        entry["time"] = round(time.time(), 3)
        self.log.write(json.dumps(entry) + "\n")
        self.log.flush()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        start = time.monotonic()
        peer = writer.get_extra_info("peername")
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=30)
        except (TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            writer.close()
            return
        lines = head.decode("latin-1").split("\r\n")
        parts = lines[0].split(" ")
        if len(parts) != 3:
            writer.close()
            return
        method, target, version = parts
        if method == "CONNECT":
            host, _, port_text = target.rpartition(":")
            port = int(port_text or 443)
        else:
            url = urlsplit(target)
            host, port = url.hostname or "", url.port or 80
        allowed = bool(host) and self.policy.allowed(host)
        if not allowed:
            writer.write(
                b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
            writer.close()
            self.record(client=str(peer), method=method, host=host, port=port, allowed=False)
            return
        try:
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=30
            )
        except (TimeoutError, OSError) as error:
            writer.write(
                b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
            writer.close()
            self.record(method=method, host=host, port=port, allowed=True, error=str(error))
            return
        if method == "CONNECT":
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
        else:
            url = urlsplit(target)
            path = (url.path or "/") + (f"?{url.query}" if url.query else "")
            headers = [line for line in lines[1:] if line and not line.lower().startswith("proxy-")]
            request = "\r\n".join([f"{method} {path} {version}", *headers, "", ""])
            upstream_writer.write(request.encode("latin-1"))
            await upstream_writer.drain()
        up, down = await asyncio.gather(
            self.pipe(reader, upstream_writer), self.pipe(upstream_reader, writer)
        )
        self.record(
            method=method,
            host=host,
            port=port,
            allowed=True,
            bytes_up=up,
            bytes_down=down,
            seconds=round(time.monotonic() - start, 3),
        )

    @staticmethod
    async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> int:
        total = 0
        try:
            while chunk := await reader.read(65536):
                total += len(chunk)
                writer.write(chunk)
                await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            with contextlib.suppress(ConnectionError, OSError):
                writer.close()
        return total


async def serve(port: int, policy: Policy, log_path: str, host: str = "0.0.0.0") -> None:
    proxy = Proxy(policy, log_path)
    server = await asyncio.start_server(proxy.handle, host, port)
    print(READY, flush=True)
    async with server:
        await server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--port", type=int, default=3128)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--log", required=True)
    parser.add_argument("--allow", action="append", default=[])
    parser.add_argument("--log-only", action="store_true")
    args = parser.parse_args(argv)
    asyncio.run(serve(args.port, Policy(args.allow, args.log_only), args.log, args.host))
    return 0


if __name__ == "__main__":
    sys.exit(main())
