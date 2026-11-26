"""Requests to Hugging Face that end.

The benchmark adapters download with huggingface_hub. Its HTTP client (httpx in 1.x, httpx2 in
2.x) has no time-out of its own, and some of its requests ask for none: listing a repository's
files (``list_repo_tree``, which ``snapshot_download`` and ``list_repo_files`` call) waited for
ever on a proxy or a network that accepts the connection and never answers, whatever
``HF_HUB_DOWNLOAD_TIMEOUT`` said (A3-13). In ``bounded``, every request without a time-out gets
that one (10 s unless set), which huggingface_hub retries as it retries any time-out, and a
request that never got its answer is raised as ``TimeoutError``: a download that failed, which
the command reports with its fix.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from collections.abc import Iterator
from typing import Any

from real2simlab.errors import R2SError

TIMEOUTS = (("httpx2", "TimeoutException"), ("httpx", "TimeoutException"), ("requests", "Timeout"))
"""What the HTTP libraries under huggingface_hub raise for a request that timed out, by module
and name (nothing is imported to find them)."""

WAITS = ("connect", "read", "write", "pool")
"""httpx's time-outs of a request: connecting, each read, each write, a free connection."""

_lock = threading.Lock()
_ready: list[bool] = []


@contextlib.contextmanager
def bounded() -> Iterator[None]:
    """Downloads from Hugging Face in this block end: each request has a time-out, and one
    that timed out (once huggingface_hub's own retries are spent) is raised as
    ``TimeoutError``. The Lab's own errors pass unchanged."""
    _bound_requests()
    try:
        yield
    except R2SError:
        raise
    except Exception as error:
        timed_out = _timed_out(error)
        if timed_out is None:
            raise
        raise TimeoutError(_said(timed_out)) from error


def _bound_requests() -> None:
    """Give each request huggingface_hub's shared client sends without a time-out the download
    time-out: from the Lab's first download on, in this process and in those forked from it
    (a process pool's workers), on top of whatever client the process configured. A
    huggingface_hub before 1.0 (requests, not httpx) is left as it is."""
    with _lock:
        if _ready:
            return
        _ready.append(True)
        try:
            import huggingface_hub  # pyright: ignore[reportMissingImports]
            from huggingface_hub.utils import _http  # pyright: ignore[reportMissingImports]
        except ImportError:  # each adapter says how to install it
            return
        make = getattr(_http, "_GLOBAL_CLIENT_FACTORY", None)
        install = getattr(huggingface_hub, "set_client_factory", None)
        if callable(make) and callable(install):
            install(lambda: _with_timeouts(make()))


def _with_timeouts(client: Any) -> Any:
    """``client`` (an httpx client) giving its requests the waits they lack."""
    hooks = getattr(client, "event_hooks", None)
    if isinstance(hooks, dict):  # otherwise not httpx's: left as the process configured it
        client.event_hooks = {**hooks, "request": [*hooks.get("request", []), _time_out]}
    return client


def _time_out(request: Any) -> None:
    """Each wait of ``request`` without a time-out (``WAITS``) gets the download time-out."""
    from huggingface_hub import constants  # pyright: ignore[reportMissingImports]

    limit = float(constants.HF_HUB_DOWNLOAD_TIMEOUT)
    waits = dict(request.extensions.get("timeout") or {})
    for wait in WAITS:
        if waits.get(wait) is None:
            waits[wait] = limit
    request.extensions["timeout"] = waits


def _timed_out(error: BaseException) -> BaseException | None:
    """The HTTP library's time-out behind ``error`` (it, or what led to it), or ``None``."""
    kinds = tuple(
        kind
        for module, name in TIMEOUTS
        if isinstance(kind := getattr(sys.modules.get(module), name, None), type)
    )
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if kinds and isinstance(current, kinds):
            return current
        seen.add(id(current))
        hidden = current.__suppress_context__  # `raise ... from None`
        current = current.__cause__ or (None if hidden else current.__context__)
    return None


def _said(timed_out: BaseException) -> str:
    """What timed out: the address asked (without a user, password or query, which a mirror's
    address may hold), and the library's words."""
    url = ""
    with contextlib.suppress(Exception):  # no request (httpx raises when asked for it)
        asked: Any = getattr(timed_out, "request", None)
        url = f" ({asked.url.scheme}://{asked.url.host}{asked.url.path})"
    detail = str(timed_out).strip() or type(timed_out).__name__
    return f"Hugging Face did not answer in time{url}: {detail}"
