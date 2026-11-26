"""Third-party code fetched at run time at a pinned commit, never bundled.

Some benchmarks publish code without a license that allows redistribution (4DCodeBench has
none). Real2SimLab therefore does not ship their code; it clones the official repository at a
fixed commit into the user's cache when a feature needs it.

A checkout is handed out only after it is checked, on every use: its completion marker names
the commit, git's ``HEAD`` is that commit and no tracked file differs from it. A checkout left
damaged by a crash (empty files under a complete-looking marker) is fetched again.

``download_failure`` explains a download of third-party code or data that failed, whatever
library raised it: a repository that refused this machine, or the network.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from real2simlab._fs import (
    fsync_directory,
    fsync_tree,
    held_path,
    release,
    remove_abandoned,
    warn_damaged,
)
from real2simlab.errors import SetupError
from real2simlab.sandbox.process import run_captured

MARKER = ".r2s-complete"
_verified: set[Path] = set()
"""The checkouts this process has checked or fetched: one found changed later was changed
while in use, and its warning says so."""
_verified_lock = threading.Lock()

CACHE_ENV = "R2S_CACHE_DIR"
# A fetch of a pinned commit that has not finished after this long is stopped: the network or
# the proxy stopped answering.
FETCH_TIMEOUT_S = 900.0


def cache_dir() -> Path:
    """Real2SimLab's cache: ``$R2S_CACHE_DIR`` or ``$XDG_CACHE_HOME/real2simlab``."""
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "real2simlab"


@dataclass(frozen=True)
class VendoredRepo:
    """A third-party repository pinned to a commit."""

    name: str
    url: str
    commit: str
    license: str
    """The repository's license, as checked when it was pinned."""
    override_env: str | None = None
    """An environment variable that points to an existing checkout instead."""

    def path(self) -> Path:
        return cache_dir() / "vendor" / f"{self.name}@{self.commit[:12]}"

    def ensure(self) -> Path:
        """The checkout, cloned on first use. Raises ``SetupError`` if it cannot be fetched."""
        if self.override_env and os.environ.get(self.override_env):
            root = Path(os.environ[self.override_env])
            if not root.is_dir():
                raise SetupError(
                    f"{self.override_env}={root} does not exist",
                    fix=f"unset {self.override_env} or point it to a {self.name} checkout",
                )
            return root
        target = self.path()
        git = shutil.which("git")
        if git is None:
            raise SetupError("git is needed to fetch " + self.name, fix="install git")
        # Checked on every use, not once per process: a checkout edited since this process
        # checked it (an editor, a sync tool, a `git checkout` in the cache) is fetched again,
        # never handed to a scorer as the pinned commit (two git calls, milliseconds).
        problem = self.problem(target, git)
        with _verified_lock:
            if problem is None:
                _verified.add(target)
                return target
            if target in _verified:  # it was intact when this process checked it before
                problem = f"changed since it was checked: {problem}"
        if (target / MARKER).exists():
            warn_damaged(target, problem)
        # A scratch clone per call: concurrent callers (parallel scoring) must not share one. A
        # clone of a command killed outright is removed (not one of a command still cloning).
        target.parent.mkdir(parents=True, exist_ok=True)
        remove_abandoned(target.parent, re.escape(f"{target.name}."), r"\.partial")
        partial = held_path(
            lambda part: target.parent / f"{target.name}.{part}.partial", mode=0o700
        )
        steps = [
            [git, "init", "-q", str(partial)],
            [git, "-C", str(partial), "remote", "add", "origin", self.url],
            [git, "-C", str(partial), "fetch", "-q", "--depth", "1", "origin", self.commit],
            [git, "-C", str(partial), "checkout", "-q", "FETCH_HEAD"],
        ]
        fix = (
            f"check the network and the proxy settings, or clone it yourself and set "
            f"{self.override_env}"
            if self.override_env
            else "check the network and the proxy settings"
        )
        try:
            for step in steps:
                # Stopped with the command (Ctrl-C, an interrupted experiment), and never waits
                # for ever on a network that stopped answering.
                try:
                    result = run_captured(step, timeout_s=FETCH_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    raise SetupError(
                        f"fetching {self.name} {self.commit} from {self.url} did not finish in "
                        f"{FETCH_TIMEOUT_S:g} s",
                        fix=fix,
                    ) from None
                if result.returncode != 0:
                    raise SetupError(
                        f"cannot fetch {self.name} {self.commit} from {self.url}: "
                        f"{result.stderr.strip()}",
                        fix=fix,
                    )
            # The files reach the disk before the marker that vouches for them.
            fsync_tree(partial)
            (partial / MARKER).write_text(self.commit + "\n", encoding="utf-8")
            fsync_tree(partial)
            if self.problem(target, git) is None:  # another caller finished first
                shutil.rmtree(partial, ignore_errors=True)
            else:
                shutil.rmtree(target, ignore_errors=True)
                try:
                    partial.rename(target)
                except OSError:
                    shutil.rmtree(partial, ignore_errors=True)
                    if self.problem(target, git) is not None:
                        raise
                fsync_directory(target.parent)
        except BaseException:  # nothing of a clone that failed is left beside the checkout
            shutil.rmtree(partial, ignore_errors=True)
            raise
        finally:
            release(partial)  # moved into place, or removed
        with _verified_lock:
            _verified.add(target)
        return target

    def problem(self, root: Path, git: str) -> str | None:
        """Why the checkout at ``root`` cannot be reused, or ``None`` if it is intact."""
        marker = root / MARKER
        if not marker.is_file():
            return "no checkout"
        if marker.read_text(encoding="utf-8", errors="replace").strip() != self.commit:
            return f"its {MARKER} marker does not name commit {self.commit[:12]}"
        head = run_captured([git, "-C", str(root), "rev-parse", "HEAD"], timeout_s=120)
        if head.returncode != 0 or head.stdout.strip() != self.commit:
            return f"git cannot read it at commit {self.commit[:12]}"
        status = run_captured(
            [git, "-C", str(root), "-c", "core.fileMode=false", "status", "--porcelain",
             "--untracked-files=no"],
            timeout_s=600,
        )  # fmt: skip
        if status.returncode != 0:
            return "git cannot read its files"
        changed = status.stdout.splitlines()
        if changed:
            return f"{len(changed)} files differ from the commit, e.g. {changed[0].strip()}"
        return None


FOURDCODEBENCH = VendoredRepo(
    name="4DCodeBench",
    url="https://github.com/4DCodeBench/4DCodeBench.git",
    commit="6c466be34107d7f388f30f97bf059b68ba1a21b0",
    license="none (no license file; not redistributed)",
    override_env="R2S_4DCODEBENCH",
)


# --- downloads that failed ----------------------------------------------------------------

HF_LOGIN = "`hf auth login` (`huggingface-cli login` before huggingface_hub 1.0), or set HF_TOKEN"
"""How this machine logs in to Hugging Face, whichever huggingface_hub the install has."""

NETWORK_ERRORS = (("requests", "RequestException"), ("httpx", "HTTPError"), ("httpx2", "HTTPError"))
"""The errors of the download libraries the Lab's dependencies use (huggingface_hub 0.x uses
requests, 1.x httpx, 2.x httpx2), by module and name."""


BUG_TYPES: tuple[type[BaseException], ...] = (
    LookupError, TypeError, AttributeError, NameError, AssertionError, ArithmeticError,
    NotImplementedError, RecursionError,
)  # fmt: skip
"""What a mistake in the code raises: a missing key or index, a wrong type or attribute, a
failed assertion, a division by zero, ... (see ``is_bug``)."""


def causes(error: BaseException) -> Iterator[BaseException]:
    """``error`` and what led to it, each once, as Python's traceback shows them: what it was
    raised ``from``, or else what it was raised while handling, unless ``from None`` hid that."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        hidden = current.__suppress_context__  # `raise ... from None` (or from a cause)
        current = current.__cause__ or (None if hidden else current.__context__)


def network_types() -> tuple[type[BaseException], ...]:
    """What a failed download raises: the standard library's errors and those of the download
    libraries already in use (imported by whatever downloaded: nothing is imported here)."""
    import http.client
    import socket
    import ssl
    import urllib.error

    types: list[type[BaseException]] = [
        urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError,
        socket.gaierror, ssl.SSLError,
    ]  # fmt: skip
    for module, name in NETWORK_ERRORS:
        found = getattr(sys.modules.get(module), name, None)
        if isinstance(found, type) and issubclass(found, BaseException):
            types.append(found)
    return tuple(types)


def is_bug(error: BaseException) -> bool:
    """Whether only a mistake in the code raises ``error``: one of ``BUG_TYPES`` that is not
    also the error of an input, of the system or of a download (numpy's ``AxisError`` is a
    ``ValueError`` too). It is a bug even when raised while a failed download was being
    handled: what it was handling never explains it."""
    return isinstance(error, BUG_TYPES) and not isinstance(
        error, (OSError, ValueError, *network_types())
    )


def network_failure(error: BaseException) -> BaseException | None:
    """The failed download behind ``error`` (it or what caused it), or ``None``."""
    types = network_types()
    return next((cause for cause in causes(error) if isinstance(cause, types)), None)


def download_failure(error: BaseException) -> SetupError | None:
    """The download that failed behind ``error`` (it, or what caused it), explained with the fix
    that suits it; ``None`` when no download failed, and for a bug (``is_bug``) raised while a
    failed download was being handled. A Hugging Face repository that refuses this machine
    (gated, private, gone, a pinned revision no longer there) is told apart from a network that
    failed (no connection, a name that does not resolve, a proxy, a time-out): logging in helps
    the first, never the second. Libraries are consulted only if already imported."""
    if is_bug(error):
        return None
    for cause in causes(error):
        refused = _hub_refusal(cause)
        if refused is not None:
            return refused
    network = network_failure(error)
    return None if network is None else _network_error(network)


def _network_error(error: BaseException) -> SetupError:
    reason: object = error
    url: object = None
    with contextlib.suppress(Exception):
        reason = getattr(error, "reason", None) or error
    with contextlib.suppress(Exception):  # httpx raises when an error has no request
        url = getattr(error, "url", None) or getattr(error, "filename", None)
        url = url or getattr(getattr(error, "request", None), "url", None)
    said = str(reason).strip().splitlines()
    proxies = sorted(
        name for name in os.environ if name.lower() in {"https_proxy", "http_proxy", "all_proxy"}
    )
    setting = f" (this shell sets {', '.join(proxies)})" if proxies else ""
    return SetupError(
        f"a download failed: {said[0] if said else type(reason).__name__}"
        + (f" ({url})" if url else ""),
        fix=f"check this machine's network connection and proxy settings{setting}, then run "
        "the same command again (set R2S_DEBUG=1 for the traceback)",
    )


def _hub_kinds(error: BaseException) -> set[str]:
    """The names of ``error``'s classes that are Hugging Face's library's own."""
    return {kind.__name__ for kind in type(error).__mro__ if kind.__module__ == "huggingface_hub"
            or kind.__module__.startswith("huggingface_hub.")}  # fmt: skip


def _hub_refusal(error: BaseException) -> SetupError | None:
    """What Hugging Face's library says in ``error`` when it is not the network's failure: a
    repository that refused this machine, or a download turned off (``HF_HUB_OFFLINE``)."""
    kinds = _hub_kinds(error)
    if not kinds:
        return None
    if kinds & {"LocalEntryNotFoundError", "OfflineModeIsEnabled"}:  # not downloaded, not cached
        network = None if "OfflineModeIsEnabled" in kinds else network_failure(error)
        if network is not None and "OfflineModeIsEnabled" not in _hub_kinds(network):
            return None  # the network failure behind it says more
        if os.environ.get("HF_HUB_OFFLINE", "").upper() in {"1", "ON", "YES", "TRUE"}:
            return SetupError(
                "a file must be downloaded from Hugging Face, which HF_HUB_OFFLINE turns off, "
                "and it is not in the local cache",
                fix="unset HF_HUB_OFFLINE, then run the same command again",
            )
        return SetupError(
            "a file could not be downloaded from Hugging Face, and it is not in the local cache",
            fix="check this machine's network connection and proxy settings, then run the same "
            "command again (set R2S_DEBUG=1 for the traceback)",
        )
    status: object = None
    url = ""
    with contextlib.suppress(Exception):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        url = str(getattr(response, "url", None) or "")
    code = f" (HTTP {status})" if isinstance(status, int) else ""
    repository, page = _repository(url)
    where = repository or "a repository"
    login = f"log in on this machine with that account: {HF_LOGIN}"
    if "GatedRepoError" in kinds:
        return SetupError(
            f"{where} on Hugging Face is gated, and this machine may not download it{code}",
            fix=f"accept its terms at {page or 'its Hugging Face page'} with your Hugging Face "
            f"account (and wait until access is granted), {login}; then run the same command "
            "again",
        )
    if "RepositoryNotFoundError" in kinds:
        return SetupError(
            f"{where} was not found on Hugging Face, or it is private and this machine may not "
            f"see it{code}",
            fix=f"check that {page or 'the repository'} opens with your Hugging Face account; "
            f"if it needs one, {login}; then run the same command again",
        )
    if "RevisionNotFoundError" in kinds or "EntryNotFoundError" in kinds:
        what = "The revision" if "RevisionNotFoundError" in kinds else "A file"
        return SetupError(
            f"{what} of {where} that this release of Real2SimLab downloads is not on Hugging "
            f"Face{code}",
            fix=f"check {page or 'the repository'}: if it was rewritten, a release of "
            "Real2SimLab that pins its current revision is needed (report it)",
        )
    if status in (401, 403):
        return SetupError(
            f"Hugging Face refused this machine a download from {where}{code}",
            fix=f"use an account that may download it and {login}; then run the same command again",
        )
    return None  # e.g. a server error: the network's kind of failure, run again


def _repository(url: str) -> tuple[str | None, str | None]:
    """The repository (``owner/name``) a Hugging Face URL is about, and the page of it:
    ``.../datasets/owner/name/resolve/...``, ``.../api/datasets/owner/name/...``."""
    parts = urllib.parse.urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return None, None
    segments = [segment for segment in parts.path.split("/") if segment]
    if segments[:1] == ["api"]:
        segments = segments[1:]
    kind = ""
    if segments[:1] in (["datasets"], ["spaces"], ["models"]):
        kind = "" if segments[0] == "models" else f"{segments[0]}/"
        segments = segments[1:]
    if len(segments) < 2:
        return None, None
    repository = "/".join(segments[:2])
    return repository, f"{parts.scheme}://{parts.netloc}/{kind}{repository}"
