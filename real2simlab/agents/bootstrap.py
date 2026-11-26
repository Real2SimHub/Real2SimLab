"""Starts an agent CLI inside its sandbox. Standard library only; never imports Real2SimLab.

Usage: ``python bootstrap.py PLAN.json``. The plan names the command, its stdin file, where to
write its stdout and stderr, and credential files to create from environment variables. The
bootstrap writes each credential file (e.g. Codex's ``auth.json``) into a temporary directory
inside the sandbox, removes the variables, runs the command, and writes its output only after
redacting the injected secret values and key-shaped strings. Raw output never reaches the disk,
and at most ``OUTPUT_LIMIT`` bytes of each stream do.
Files the agent writes itself (its workspace, or straight into the log directory) are not its
output: after the run, the host replaces the injected values in them as well (``file_secrets``
says which values of a credential file count).
"""

from __future__ import annotations

import base64
import collections
import contextlib
import json
import os
import re
import resource
import signal
import subprocess
import sys
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any

REPLACEMENT = "[REDACTED]"
START_FAILED = 3
"""The bootstrap's exit status when it could not start the agent's command (not in the image,
not executable, a credential file it could not write): nothing of the agent ran. Its summary
then says why (``start_error``)."""
MIN_SECRET = 6
"""The shortest value redacted as a secret: shorter ones would replace ordinary words."""
OUTPUT_LIMIT = 256 * 1024 * 1024
"""The most of each of the agent's output streams (stdout, stderr) that the bootstrap keeps,
in bytes: the beginning as it arrives and, when the stream ends, its last ``OUTPUT_TAIL``
bytes, in whole lines, with a line between them saying how much was left out. An agent that
prints in a loop would otherwise fill the disk. A file-size limit (``ulimit -f``) lowers it, so
the limit never stops the bootstrap writing."""
OUTPUT_TAIL = 16 * 1024 * 1024
"""How much of the end of a long output is kept (the agent's last events: its result, its
cost)."""
LINE_LIMIT = 8 * 1024 * 1024
"""A longer line is left out whole: never cut (a secret across the cut would escape
redaction), nor held in memory whole."""
RESERVED = 4096
"""Bytes kept under the limits for the bootstrap's own lines."""
_KEYS = r"api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|secret|password|authorization"

PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
            REPLACEMENT,
        ),
        (r"sk-ant-[A-Za-z0-9_\-]{16,}", REPLACEMENT),
        (r"sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}", REPLACEMENT),
        (r"AIza[0-9A-Za-z_\-]{35}", REPLACEMENT),
        (r"gh[pousr]_[A-Za-z0-9]{30,}", REPLACEMENT),
        (r"github_pat_[A-Za-z0-9_]{22,}", REPLACEMENT),
        (r"(?:AKIA|ASIA)[0-9A-Z]{16}", REPLACEMENT),
        (r"hf_[A-Za-z0-9]{30,}", REPLACEMENT),
        (r"xox[abposr]-[A-Za-z0-9\-]{10,}", REPLACEMENT),
        (r"eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}", REPLACEMENT),
        (r"(?i)(bearer\s+)[A-Za-z0-9._~+/=\-]{16,}", r"\1" + REPLACEMENT),
        (
            rf"(?i)([\"'](?:{_KEYS})[\"']\s*:\s*[\"'])[^\"']{{8,}}([\"'])",
            r"\1" + REPLACEMENT + r"\2",
        ),
        (rf"(?i)((?:{_KEYS}|token)=)[A-Za-z0-9._\-]{{12,}}", r"\1" + REPLACEMENT),
    )
)


class Redactor:
    """Replaces known secret values and key-shaped strings with ``[REDACTED]``."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self.secrets = sorted({s for s in secrets if len(s) >= MIN_SECRET}, key=len, reverse=True)
        self.count = 0

    def __call__(self, text: str) -> str:
        for secret in self.secrets:
            if secret in text:
                self.count += text.count(secret)
                text = text.replace(secret, REPLACEMENT)
        for pattern, replacement in PATTERNS:
            text, found = pattern.subn(replacement, text)
            self.count += found
        return text


LONG_VALUE = 16
"""A string this long in a credential file is redacted whatever it is named."""
_CREDENTIAL_WORDS = frozenset(
    {
        "key", "keys", "apikey", "token", "tokens", "secret", "secrets", "password",
        "passwords", "passwd", "pwd", "pass", "passphrase", "pin", "credential",
        "credentials", "cookie", "cookies", "authorization",
    }
)  # fmt: skip
_VARIABLE_NAME = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")
"""What an environment variable's name looks like (``OPENAI_API_KEY``): a name, not a key."""
_TOKEN = re.compile(r"\"[^\"]*\"|'[^']*'|[^\s=:,;\"']+")


def credential_name(name: str) -> bool:
    """Whether a name in a credential file says that its value is a credential (``password``,
    ``api_key``, ``accessToken``, ``OPENAI_API_KEY``, ``client_secret``, ...)."""
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()
    return any(word in _CREDENTIAL_WORDS for word in re.split(r"[^a-z0-9]+", snake))


def _strings(value: Any, named: bool = False) -> list[str]:
    """Candidate secrets inside a JSON value: every string of ``LONG_VALUE`` characters or
    more, and every string under a credential's name (at any depth below it)."""
    if isinstance(value, str):
        if named and not _VARIABLE_NAME.fullmatch(value):
            return [value]
        return [value] if len(value) >= LONG_VALUE else []
    if isinstance(value, dict):
        return [
            s
            for key, item in value.items()
            for s in _strings(item, named or credential_name(str(key)))
        ]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item, named)]
    return []


def _assigned(text: str) -> list[str]:
    """Values given to a credential's name in a file that is not JSON: ``password VALUE``
    (netrc), ``api_key = "VALUE"`` (TOML, INI), ``TOKEN=VALUE`` (.env), ``key: VALUE`` (YAML)."""
    found: list[str] = []
    for line in text.splitlines():
        tokens = [token.strip("\"'") for token in _TOKEN.findall(line)]
        for index in range(len(tokens) - 1):  # the images' Python may predate pairwise()
            name, value = tokens[index], tokens[index + 1]
            if credential_name(name) and not _VARIABLE_NAME.fullmatch(value):
                found.append(value)
    return found


def file_secrets(content: bytes) -> list[str]:
    """The values of an injected credential file to redact, as an injected variable's value is
    (from ``MIN_SECRET`` characters): a small file (a bare key, a short login) in full and line
    by line, since output is redacted line by line; every value under a credential's name; and
    every long string (JSON) or long line (other formats). Other short values (a model's name,
    an effort level) are settings, which adapters inject through the same files."""
    text = content.decode("utf-8", errors="replace")
    stripped = text.strip()
    found: list[str] = []
    if 0 < len(stripped) < 200:
        found.append(stripped)
        found.extend(line.strip() for line in stripped.splitlines())
    try:
        found.extend(_strings(json.loads(text)))
    except ValueError:
        found.extend(line.strip() for line in text.splitlines() if len(line.strip()) >= LONG_VALUE)
        found.extend(_assigned(text))
    return found


class _Kept:
    """One of the agent's output streams in its file, within ``OUTPUT_LIMIT`` (see there)."""

    def __init__(self, target: Path) -> None:
        self.target = target
        self.error: str | None = None
        self.out: IO[str] | None = None
        room = OUTPUT_LIMIT
        size_limit = resource.getrlimit(resource.RLIMIT_FSIZE)[0]
        if size_limit != resource.RLIM_INFINITY:
            room = min(room, size_limit)
        try:
            self.out = target.open("a", encoding="utf-8")
            room = max(0, room - RESERVED - target.stat().st_size)
        except OSError as error:
            self.error, room = str(error), 0
        self.tail_room = min(OUTPUT_TAIL, room // 2)
        self.head_room = room - self.tail_room
        self.tail: collections.deque[tuple[str, int]] = collections.deque()
        self.tail_size = 0
        self.left_out = 0
        self.lines_left_out = 0
        self.head_closed = False
        """Once a line did not fit the beginning, every later one belongs to the end: a line
        written after it would come before the note saying what was left out."""

    def add(self, line: str) -> None:
        size = len(line.encode("utf-8"))
        if not self.head_closed and size <= self.head_room:
            self.head_room -= size
            self._write(line)
            return
        self.head_closed = True
        self.tail.append((line, size))
        self.tail_size += size
        while self.tail and self.tail_size > self.tail_room:
            _, dropped = self.tail.popleft()
            self.tail_size -= dropped
            self.left_out += dropped
            self.lines_left_out += 1

    def close(self) -> None:
        if self.left_out:
            note = {
                "type": "r2s.output_left_out",
                "bytes": self.left_out,
                "lines": self.lines_left_out,
                "kept": f"the first and the last lines, at most {OUTPUT_LIMIT} bytes",
            }
            self._write(json.dumps(note) + "\n")
        for line, _ in self.tail:
            self._write(line)
        if self.out is not None:
            with contextlib.suppress(OSError):
                self.out.close()

    def _write(self, text: str) -> None:
        if self.out is None or self.error is not None:
            return
        try:
            self.out.write(text)
            self.out.flush()
        except OSError as error:  # a full disk: the agent goes on, its output is not kept
            self.error = str(error)


def _skip_line(source: IO[bytes]) -> int:
    """Read the rest of a line, keeping none of it; how many bytes it had."""
    skipped = 0
    while True:
        chunk = source.readline(1 << 20)
        skipped += len(chunk)
        if not chunk or chunk.endswith(b"\n"):
            return skipped


def _pump(source: IO[bytes], target: Path, redactor: Redactor, lock: threading.Lock) -> None:
    """Copy one of the agent's output streams into ``target``, redacted and bounded. It reads
    to the end whatever happens to the file (a full disk), so that the agent never blocks on a
    full pipe."""
    kept = _Kept(target)
    try:
        while True:
            raw = source.readline(LINE_LIMIT)
            if not raw:
                break
            if len(raw) == LINE_LIMIT and not raw.endswith(b"\n"):
                rest = _skip_line(source)
                if rest:  # longer than LINE_LIMIT: left out whole
                    note = {"type": "r2s.line_left_out", "bytes": len(raw) + rest}
                    with lock:
                        kept.add(json.dumps(note) + "\n")
                    continue
            line = raw.decode("utf-8", errors="replace")
            with lock:
                kept.add(redactor(line))
    finally:
        with lock:
            kept.close()
    if kept.error is not None:
        with contextlib.suppress(OSError):
            print(
                f"bootstrap: {target} could not be written ({kept.error}); the rest of that "
                "output was not kept",
                file=sys.stderr,
                flush=True,
            )


def _read_kept(path: Path) -> str:
    """A file the agent wrote, at most ``OUTPUT_LIMIT`` bytes of it, in whole lines."""
    with path.open("rb") as stream:
        data = stream.read(OUTPUT_LIMIT + 1)
    if len(data) > OUTPUT_LIMIT:  # cut after a whole line: a key is never cut in two
        data = data[: data.rfind(b"\n", 0, OUTPUT_LIMIT) + 1] + b"[r2s] the rest was left out\n"
    return data.decode("utf-8", errors="replace")


def _start(plan: dict[str, Any], secrets: list[str]) -> subprocess.Popen[bytes]:
    """Write the credential files and start the agent's command (its secrets go to
    ``secrets``)."""
    for name in plan.get("secret_env", []):
        if os.environ.get(name):
            secrets.append(os.environ[name])
    for target, variable in plan.get("secret_files", {}).items():
        encoded = os.environ.pop(variable, "")
        content = base64.b64decode(encoded) if encoded else b""
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(0o600)
        secrets.extend(file_secrets(content))
    stdin = open(plan["stdin"], "rb") if plan.get("stdin") else subprocess.DEVNULL  # noqa: SIM115
    env = {k: v for k, v in os.environ.items() if k not in set(plan.get("unset_env", []))}
    return subprocess.Popen(
        plan["argv"],
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=plan.get("cwd"),
        env=env,
        start_new_session=True,
    )


def main(plan_path: str) -> int:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    stdout_path, stderr_path = Path(plan["stdout"]), Path(plan["stderr"])
    # First, so that the Lab can tell a sandbox that never ran this from an agent that failed.
    for path in (stdout_path, stderr_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    secrets: list[str] = []
    try:
        process = _start(plan, secrets)
    except Exception as error:  # nothing of the agent ran: not the agent's failure
        start_error = Redactor(secrets)(f"{type(error).__name__}: {error}")
        summary = {"exit_code": None, "redactions": 0, "start_error": start_error}
        Path(plan["summary"]).write_text(json.dumps(summary) + "\n", encoding="utf-8")
        return START_FAILED
    redactor = Redactor(secrets)

    def forward(signum: int, _frame: object) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    lock = threading.Lock()
    assert process.stdout is not None and process.stderr is not None
    threads = [
        threading.Thread(target=_pump, args=(process.stdout, stdout_path, redactor, lock)),
        threading.Thread(target=_pump, args=(process.stderr, stderr_path, redactor, lock)),
    ]
    for thread in threads:
        thread.start()
    code = process.wait()
    for thread in threads:
        thread.join()
    for source, target in plan.get("copy_after", {}).items():
        path = Path(source)
        if path.is_file():
            text = _read_kept(path)
            Path(target).write_text(redactor(text), encoding="utf-8")
    summary = {"exit_code": code, "redactions": redactor.count}
    Path(plan["summary"]).write_text(json.dumps(summary) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
