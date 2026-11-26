"""The build flow shared by every agent adapter.

1. Prepare a workspace: the task's visible data, reference documents, the SDK, an empty
   ``solution/`` and the task statement (from the versioned template).
2. Run the agent's CLI in a sandbox that mounts only that workspace (never the hidden
   directory), with credentials injected at run time and its output redacted before it is
   written (``bootstrap.py``). The network is an allowlist through an egress proxy by default.
3. Collect ``solution/``, check it, and write ``twin.json`` (the agent never writes the
   environment, output format or origin).
4. An agent that delivers no valid twin is a result (scored as zero) with a recorded reason. A
   sandbox that failed around the agent (it did not start, it was killed from outside, it
   could not start the agent's command) is not: the build is an ``infra_error``.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

from pydantic import PositiveFloat, PositiveInt, field_validator

from real2simlab._fs import is_regular, os_error_text, read_regular, write_new
from real2simlab.agents.bootstrap import START_FAILED
from real2simlab.agents.record import AgentRecord, Usage
from real2simlab.agents.redact import REPLACEMENT, Redactor, file_secrets, redact_files
from real2simlab.capture.io import CAPTURE_FILENAME
from real2simlab.envs import image_reference
from real2simlab.errors import DeliveryError, InfrastructureError, R2SError, SetupError
from real2simlab.formats.base import get_format, tree_problems
from real2simlab.methods.base import BuildContext, BuildOutcome, Method, MethodOptions, package_twin
from real2simlab.sandbox.base import (
    Backend,
    ExecResult,
    Limits,
    Mount,
    NetworkMode,
    NetworkPolicy,
    SandboxSpec,
    rewrite_paths,
)
from real2simlab.sandbox.docker import DockerBackend
from real2simlab.task.instruction import TEMPLATE_VERSION, with_time_budget
from real2simlab.task.io import copy_exactly
from real2simlab.twin import AgentInfo, InvalidTwinError
from real2simlab.twin.conventions import convention_spec
from real2simlab.twin.fingerprint import is_os_litter
from real2simlab.twin.sdk import sdk_path

WORKSPACE = "/workspace"
LOGS = "/r2s/logs"
AGENT_DIR = "/r2s/agent"
AGENT_HOME = "/r2s/home"
"""The agent's HOME: a tmpfs in the sandbox, so credentials and caches never reach the host."""
BOOTSTRAP = Path(__file__).with_name("bootstrap.py")
GRACE_S = 120
"""Extra time after the agent's budget before its sandbox is killed."""


class AgentOptions(MethodOptions):
    model: str | None = None
    reasoning_effort: str | None = "medium"
    """Passed to the CLIs that take one (``codex``); adapters that pass none to their CLI refuse
    any other value than the default, and their records say null."""
    time_budget_s: PositiveInt | None = None
    """Seconds; defaults to the task's ``agent_time_budget_s``."""
    network: Literal["allowlist", "open", "none"] = "allowlist"
    allowlist: tuple[str, ...] | None = None
    """Domains the agent may reach; defaults to the adapter's list for its provider."""
    image: str | None = None
    """Overrides the agent image."""
    cpus: PositiveFloat = 4.0
    memory_mb: PositiveInt = 8192
    gpus: str | None = "all"
    """GPUs for tasks that grant one (``resources.gpu``): ``all``, device indices such as
    ``0,1``, or ``none`` (or null) to give the agent no GPU."""

    @field_validator("gpus")
    @classmethod
    def _check_gpus(cls, value: str | None) -> str | None:
        if value is None or value.strip().lower() == "none":
            return None
        if value != "all" and not all(part.strip().isdigit() for part in value.split(",")):
            raise ValueError(f"gpus must be 'all', device indices like '0,1', or 'none': {value!r}")
        return value

    extra_args: tuple[str, ...] = ()


def budget_of(options: AgentOptions, default: int) -> int:
    """The agent's time budget: ``time_budget_s`` when it is given, else the task's."""
    return options.time_budget_s if options.time_budget_s is not None else default


@dataclass(frozen=True)
class Credentials:
    """What an agent needs to authenticate, injected only at run time."""

    env: Mapping[str, str] = field(default_factory=lambda: {})
    """Secret environment variables (never logged or recorded)."""
    files: Mapping[str, bytes] = field(default_factory=lambda: {})
    """Files created inside the sandbox's tmpfs, keyed by their in-sandbox path."""
    public_env: Mapping[str, str] = field(default_factory=lambda: {})
    """Non-secret variables, e.g. ``CODEX_HOME``."""
    description: str = "none"
    """How the credentials were obtained, for the record (never the values)."""


@dataclass(frozen=True)
class CredentialStatus:
    """Whether an agent's credentials are configured, found without fetching them."""

    available: bool
    detail: str
    """Where they would come from (never a value)."""
    fix: str | None = None


def env_status(names: Sequence[str]) -> CredentialStatus:
    """Available if any of ``names`` is set in the environment."""
    for name in names:
        if os.environ.get(name):
            return CredentialStatus(True, f"{name} is set")
    return CredentialStatus(False, f"{names[0]} is not set", f"export {names[0]}=...")


def endpoint_set(names: Sequence[str]) -> str | None:
    """The first of ``names`` (a host's endpoint variables, such as ``ANTHROPIC_BASE_URL``) set
    in the environment. An agent never takes such a variable from the host: one that is set
    while the method names no endpoint means that its key is the key of that endpoint, which the
    agent would send to its provider's own API instead. Adapters refuse that, with the option
    that names the endpoint as the fix; the method's options, which are recorded, are the only
    place an agent's endpoint comes from."""
    return next((name for name in names if os.environ.get(name, "").strip()), None)


DEFAULT_EFFORT = "medium"
"""The default of the ``reasoning_effort`` option (:class:`AgentOptions`)."""


def effort_not_passed(cli: str, value: str | None, through: str = "extra_args") -> str | None:
    """``value``, the ``reasoning_effort`` option of an agent that does not pass one to ``cli``:
    only its default or none is taken (its record says null); another value would change
    nothing that runs, so it is refused before anything runs. ``through`` is the option that
    can carry the CLI's own setting."""
    if value is not None and value != DEFAULT_EFFORT:
        raise ValueError(
            f"this adapter does not pass a reasoning effort to {cli}, so {value!r} would change "
            f"nothing: remove reasoning_effort (the setting of {cli} itself, if it has one, can "
            f"be given through {through})"
        )
    return value


def injected_values(credentials: Credentials) -> list[str]:
    """The values injected into an agent's sandbox as credentials: the secret variables, and
    each credential file as it travels (base64) and the values in it that are secrets."""
    values = list(credentials.env.values())
    for content in credentials.files.values():
        values.append(base64.b64encode(content).decode())
        values.extend(file_secrets(content))
    return values


@dataclass(frozen=True)
class Scrubbed:
    """What :func:`scrub_injected` did to the files an agent wrote itself."""

    redactions: int = 0
    """Values replaced with ``[REDACTED]``, and files removed because they could not be."""
    infra: str | None = None
    """This machine could not rewrite them (a full disk, a read-only or failing file system):
    nothing is delivered, and the build is not a result."""
    failure: str | None = None
    """Files that may still hold a value could be neither rewritten nor removed for a reason
    of the agent's (a path too long): nothing is delivered, and that is its result."""
    note: str | None = None
    """Which files were removed, said with the reason the build failed (if it does)."""
    left: tuple[str, ...] = ()
    """Files (relative to the build's work directory) that may still hold a value."""


def scrub_injected(context: BuildContext, root: Path, values: Sequence[str]) -> Scrubbed:
    """Replace the injected ``values`` wherever the agent wrote them itself under ``root`` (its
    workspace and solution, its log directory), before anything is packaged, rendered or
    exported (:func:`redact_files`); a file that cannot be rewritten is removed. Only a failure
    of this machine (:data:`MACHINE_ERRNOS`) makes the build not a result: one the agent
    caused (a path too long for the copy) never turns its result into no result."""
    done = redact_files(root, values)
    if done.count:
        context.log(
            f"build of {context.task_id}: the agent wrote injected credentials into its files; "
            f"{done.count} replaced with {REPLACEMENT}"
        )
    if not done.errors:
        return Scrubbed(done.count)
    machine = done.machine_error
    error = machine or done.errors[0]
    said = f"({error.strerror or type(error).__name__})"
    if done.removed:
        said += f"; removed instead: {_listed(done.removed)}"
    if done.left:
        said += f"; these may still hold them, remove them: {_listed(done.left)}"
    said = Redactor(values)(said)  # recorded and shown: a file's name may hold a value too
    context.log(
        f"build of {context.task_id}: injected credentials could not be replaced in the "
        f"agent's files {said}"
    )
    redactions, left = done.count + len(done.removed), tuple(done.left)
    if machine is not None:
        infra = (
            f"this machine could not replace the injected credentials in the agent's files {said}"
        )
        return Scrubbed(redactions, infra=infra, left=left)
    if left:
        failure = f"the agent's files hold injected credentials that could not be removed {said}"
        return Scrubbed(redactions, failure=failure, left=left)
    note = f"injected credentials could not be replaced in some of its files {said}"
    return Scrubbed(redactions, note=note)


def _listed(paths: Sequence[str], most: int = 3) -> str:
    shown = ", ".join(paths[:most])
    return shown + (f" and {len(paths) - most} more" if len(paths) > most else "")


@dataclass(frozen=True)
class Workspace:
    root: Path
    statement_file: Path
    statement: str

    @property
    def solution(self) -> Path:
        return self.root / "solution"

    @property
    def output(self) -> Path:
        return self.root / "output"


DOCS: dict[str, str] = {}


def _reference_docs(context: BuildContext) -> dict[str, str]:
    """Reference documents given to every agent, keyed by file name."""
    from real2simlab.agents import docs

    fmt = get_format(context.config.delivery.output_format)
    return {
        "output-format.md": fmt.documentation(),
        "run-contract.md": docs.RUN_CONTRACT,
        "capture.md": docs.CAPTURE,
        "sdk.md": docs.sdk_doc(),
    }


def prepare_workspace(
    context: BuildContext, root: Path, *, budget_s: float | None = None
) -> Workspace:
    """Write the agent's workspace at ``root``. With ``budget_s`` (a method's ``time_budget_s``),
    the statement tells the agent that budget instead of the task's own. The build's copy of
    ``visible/`` is copied exactly: a symbolic link in it is never followed into the workspace
    (the build fails outside the agent, not a result)."""
    root.mkdir(parents=True)
    changed = copy_exactly(context.visible_dir, root / "visible")
    if changed is not None:
        raise InfrastructureError(f"the agent's workspace was not written: {changed}")
    if not (root / "visible" / CAPTURE_FILENAME).is_file():
        raise SetupError(f"task {context.task_id!r} has no visible capture")
    docs_dir = root / "docs"
    docs_dir.mkdir()
    for name, text in _reference_docs(context).items():
        (docs_dir / name).write_text(text, encoding="utf-8")
    tools = root / "tools"
    tools.mkdir()
    shutil.copyfile(sdk_path(), tools / "r2s_twin.py")
    (root / "solution").mkdir()
    statement = context.statement
    if budget_s is not None:
        statement = with_time_budget(
            statement, context.config.delivery.agent_time_budget_s, budget_s
        )
    statement_file = root / "TASK.md"
    statement_file.write_text(statement, encoding="utf-8")
    return Workspace(root=root, statement_file=statement_file, statement=statement)


def output_mounts(context: BuildContext, workspace: Workspace) -> list[Mount]:
    """A writable mount for conventions whose programs write to a fixed path (``/app/outputs``)."""
    spec = convention_spec(context.config.delivery.convention)
    if spec.agent_output_mount is None:
        return []
    directory = workspace.root / spec.session_output
    directory.mkdir(parents=True, exist_ok=True)
    return [Mount(directory.resolve(), spec.agent_output_mount, read_only=False)]


CACHE_PATTERNS = ("__pycache__", "*.pyc", ".pytest_cache", ".ipynb_checkpoints")
"""What a solution's own test runs leave behind; never copied with it."""


def directory_problem(directory: Path) -> str | None:
    """Why ``directory`` itself, which a method wrote (an agent's ``solution/``), cannot be
    taken as its own: it is a symbolic link, which reading would follow out of what the method
    wrote (into a task's hidden data, on the host), or not a directory. ``None`` otherwise, and
    when it is missing (the caller says what is missing)."""
    try:
        mode = os.lstat(directory).st_mode
    except (OSError, ValueError):
        return None
    if stat.S_ISLNK(mode):
        return f"{Path(directory).name}/ is a symbolic link, not a directory of its own"
    if not stat.S_ISDIR(mode):
        return f"{Path(directory).name}/ is not a directory"
    return None


def session_output_of(context: BuildContext, directory: Path) -> Path | None:
    """``directory``, the output an agent left in its session, when the Lab may read it as it
    is (:func:`~real2simlab.formats.base.tree_problems`); otherwise ``None``, as if it had left
    none, and the build's log says why. One that is a symbolic link or holds one, a pipe, a
    socket or a device, or what the Lab cannot read is never handed on: the official scorer,
    the session check and the run's records would read it as it is."""
    if not os.path.lexists(directory):
        return None
    problems = tree_problems(directory, unreadable=True)
    if not problems:
        return directory
    context.log(
        f"build of {context.task_id}: its session output {Path(directory).name}/ is not used: "
        + "; ".join(problems[:3])
        + (" ..." if len(problems) > 3 else "")
    )
    return None


def solution_problems(solution: Path) -> list[str]:
    """What in ``solution`` cannot be delivered, because of the solution itself: ``solution``
    itself a symbolic link or not a directory (:func:`directory_problem`), special files (named
    pipes, sockets, devices) and entries this user cannot read. Symbolic links in it are not
    followed (the program rules judge them)."""
    root = directory_problem(Path(solution))
    if root is not None:
        return [root]
    problems: list[str] = []
    ignored = shutil.ignore_patterns(*CACHE_PATTERNS)
    stack = [Path(solution)]
    while stack:  # iterative: a solution may be very deep
        directory = stack.pop()
        if not os.access(directory, os.R_OK | os.X_OK):
            problems.append(f"{directory.relative_to(solution)}/: cannot be listed")
            continue
        with os.scandir(directory) as entries:
            listed = list(entries)
        skip = ignored(str(directory), [entry.name for entry in listed])
        for entry in listed:
            if entry.name in skip:
                continue
            path = Path(entry.path)
            relative = path.relative_to(solution).as_posix()
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                continue
            if stat.S_ISDIR(mode):
                stack.append(path)
            elif not stat.S_ISREG(mode):
                problems.append(f"{relative}: not a regular file (a pipe, socket or device)")
            elif not os.access(path, os.R_OK):
                problems.append(f"{relative}: cannot be read")
    return problems


def clean_solution(solution: Path, target: Path) -> Path:
    """Copy ``solution`` to ``target`` without caches the agent's own test runs left behind.

    Raises ``DeliveryError`` when the solution itself cannot be delivered (see
    :func:`solution_problems`): the method's failure, scored as such. Any other failure to copy
    it is this machine's (a full disk, a file-size limit, a read-only directory) and raises
    ``InfrastructureError``: it is never scored."""
    problems = solution_problems(Path(solution))
    if problems:
        shown = "; ".join(problems[:5]) + (" ..." if len(problems) > 5 else "")
        raise DeliveryError(f"the solution cannot be delivered: {shown}")
    try:
        shutil.copytree(
            solution, target, symlinks=True, ignore=shutil.ignore_patterns(*CACHE_PATTERNS)
        )
    except OSError as error:  # shutil.Error too: the files it could not copy
        raise InfrastructureError(
            f"this machine could not copy the solution to {target}: {os_error_text(error)}"
        ) from error
    return target


def solution_digest(solution: Path) -> str:
    """A digest of what :func:`clean_solution` would copy from ``solution`` now: each entry's
    relative path and kind, a file's executable bit and bytes, a symbolic link's target, and
    the names of what cannot be read, leaving out the caches it leaves out and a file
    manager's litter (``.DS_Store``). It never follows a link, never waits on a pipe and never
    raises, so it covers a solution that cannot be delivered too: two digests that differ say
    that ``solution`` changed in between (an editor, a sync tool, another command), e.g. while
    it was copied or scored."""
    digest = hashlib.sha256()

    def add(*parts: str) -> None:
        for part in parts:
            data = os.fsencode(part)
            digest.update(len(data).to_bytes(8, "big") + data)

    root = Path(solution)
    try:
        mode = os.lstat(root).st_mode
    except (OSError, ValueError) as error:
        add("missing", str(getattr(error, "errno", "")))
        return digest.hexdigest()
    if not stat.S_ISDIR(mode):
        add("not a directory", _link_target(root) if stat.S_ISLNK(mode) else oct(mode))
        return digest.hexdigest()
    ignored = shutil.ignore_patterns(*CACHE_PATTERNS)
    stack = [root]
    while stack:  # iterative: a solution may be very deep
        directory = stack.pop()
        try:
            with os.scandir(directory) as found:
                entries = sorted(found, key=lambda entry: entry.name)
        except OSError:
            add("unlisted", directory.relative_to(root).as_posix())
            continue
        skip = ignored(str(directory), [entry.name for entry in entries])
        for entry in entries:
            if entry.name in skip:
                continue
            path = Path(entry.path)
            name = path.relative_to(root).as_posix()
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError:
                add("gone", name)
                continue
            if is_os_litter(entry.name) and not stat.S_ISDIR(mode):
                continue  # a file manager's file (as fingerprint skips it); not a directory's files
            if stat.S_ISLNK(mode):
                add("link", name, _link_target(path))
            elif stat.S_ISDIR(mode):
                add("directory", name)
                stack.append(path)
            elif stat.S_ISREG(mode):
                add("file", name, "x" if mode & 0o111 else "-", _file_digest(path))
            else:
                add("special", name)
    return digest.hexdigest()


def _link_target(path: Path) -> str:
    try:
        return os.readlink(path)
    except OSError:
        return ""


def _file_digest(path: Path) -> str:
    """SHA-256 of the regular file ``path``, opened without following a link or waiting on a
    pipe put in its place; ``"unreadable"`` when it cannot be read as one."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return "unreadable"
    try:
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return "unreadable"
            file_digest = hashlib.sha256()
            while chunk := stream.read(1 << 20):
                file_digest.update(chunk)
            return file_digest.hexdigest()
    except OSError:
        return "unreadable"


@functools.cache
def cli_version(docker: str | None, image: str, argv: tuple[str, ...]) -> str | None:
    """An agent CLI's version, from ``argv`` (e.g. ``codex --version``) run offline in ``image``
    (or on the host without Docker). The last output line: some images print a banner first."""
    command = [docker, "run", "--rm", "--network", "none", image, *argv] if docker else [*argv]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[-1] if result.returncode == 0 and lines else None


_IN_IMAGE: dict[tuple[str, str, str, tuple[str, ...]], str | None] = {}


def image_cli_version(backend: Backend, image: str, argv: tuple[str, ...]) -> str | None:
    """An agent CLI's version, from ``argv`` run by ``backend`` (one that isolates, other than
    Docker: bwrap) in ``image``, where the agent runs: offline, with nothing mounted and a time
    limit. The last line it prints; ``None`` when it does not run there (an image without that
    CLI) or the image cannot be found. Kept per image content, so an image unpacked again under
    the same reference is asked again."""
    try:
        resolved = backend.resolve_image(image)
    except R2SError:
        return None
    key = (backend.name, resolved.reference, resolved.identity, argv)
    if key in _IN_IMAGE:
        return _IN_IMAGE[key]
    with tempfile.TemporaryDirectory(prefix="r2s-version-") as scratch:
        log = Path(scratch) / "version.log"
        spec = SandboxSpec(
            image=resolved.reference,
            argv=argv,
            workdir="/tmp",
            env={"HOME": "/tmp", "TMPDIR": "/tmp"},
            limits=Limits(timeout_s=120),
            label="r2s-version",
        )
        try:
            result = backend.run(spec, log)
            printed = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        except (OSError, R2SError):
            return None
    if result.infrastructure_error is not None:  # the sandbox, not the image: asked again later
        return None
    lines = [line.strip() for line in printed.splitlines() if line.strip()]
    _IN_IMAGE[key] = lines[-1] if result.exit_code == 0 and lines else None
    return _IN_IMAGE[key]


AGENT_USER = "10001:10001"
"""The uid:gid agents run as when the host user is root: some agent CLIs refuse to run
unattended as root (Claude Code's ``--dangerously-skip-permissions``), and an unprivileged
user is the safer default anyway."""


def agent_user(backend: Backend) -> str | None:
    """``AGENT_USER`` when this process runs as root on an isolating backend, else ``None``
    (the backend's default: the host user)."""
    getuid = getattr(os, "getuid", None)
    if getuid is None or getuid() != 0 or not backend.isolated:
        return None
    return AGENT_USER


def unreadable_by_others(root: Path) -> list[str]:
    """Paths under ``root`` that a user other than the owner (and outside its group) cannot
    read: directories without ``o+rx``, files without ``o+r``. Symbolic links are not
    followed."""
    problems: list[str] = []
    stack = [Path(root)]
    while stack:
        path = stack.pop()
        try:
            info = path.lstat()
        except OSError as error:
            problems.append(f"{path}: {error}")
            continue
        if stat.S_ISLNK(info.st_mode):
            continue
        if stat.S_ISDIR(info.st_mode):
            if info.st_mode & 0o005 != 0o005:
                problems.append(f"{path}/ (mode {stat.S_IMODE(info.st_mode):o})")
                continue
            stack.extend(path.iterdir())
        elif info.st_mode & 0o004 == 0:
            problems.append(f"{path} (mode {stat.S_IMODE(info.st_mode):o})")
    return problems


def hand_over(directories: Sequence[Path], user: str) -> Callable[[], None]:
    """Give ``directories`` (recursively) to ``user``; return a function that gives every path
    back to its previous owner. Symbolic links are not followed."""
    uid, _, gid = user.partition(":")
    owners: dict[Path, tuple[int, int]] = {}
    for directory in directories:
        for path in [directory, *directory.rglob("*")]:
            info = path.lstat()
            owners[path] = (info.st_uid, info.st_gid)
            os.chown(path, int(uid), int(gid or uid), follow_symlinks=False)

    def restore() -> None:
        for directory in directories:
            # New files the agent wrote belong to it too: give them to the directory's owner.
            default = owners.get(directory)
            if default is None or not directory.exists():
                continue
            for path in [directory, *directory.rglob("*")]:
                old_uid, old_gid = owners.get(path, default)
                try:
                    os.chown(path, old_uid, old_gid, follow_symlinks=False)
                except OSError:
                    continue

    return restore


PROBE_LOG = "sandbox-probe.log"
"""Where the probe of an agent's sandbox that failed before its bootstrap began writes."""
PROBE = ("-I", "-c", "pass")
"""Starts the interpreter the bootstrap needs, and nothing else."""


def bootstrap_began(agent_dir: Path) -> bool:
    """Whether the bootstrap ran in the agent's sandbox: it first creates the agent's output
    files."""
    return (agent_dir / "trajectory.jsonl").exists() or (agent_dir / "stderr.log").exists()


HARNESS_STARTED = "harness-started"
"""Written in the agent's log directory as soon as the process of a harness that runs the agent
from the host (Harbor, mini-swe-agent) exists: the harness writes its own logs only once it
ends. Never for one that could not be spawned or was stopped before it started."""


def agent_started(agent_dir: Path) -> bool:
    """Whether the bootstrap started the agent's command (which may then have used its model and
    spent money): it began, and did not report that it could not start the command; or a
    harness on the host was started for it (:data:`HARNESS_STARTED`)."""
    if (agent_dir / HARNESS_STARTED).exists():
        return True
    if not bootstrap_began(agent_dir):
        return False
    summary = _read_json(agent_dir / "bootstrap.json")
    return not (summary is not None and summary.get("start_error"))


def sandbox_problem(backend: Backend, spec: SandboxSpec, log_path: Path) -> str | None:
    """Why the agent's sandbox ``spec`` cannot start anything here, or ``None`` when a sandbox
    like it starts Python (or this cannot be told). Asked when the agent's sandbox failed before
    its bootstrap began (bwrap that cannot set it up exits 1, as a failing agent does): when a
    sandbox like it, holding no secret and running only ``python -I -c pass`` in the Lab's
    read-only plan directory, fails too, the sandbox failed, not the agent. As for a twin's run
    (``runner.run.runtime_problem``), the probe is a diagnosis: when it cannot run, the result
    stands."""
    check = replace(
        spec,
        argv=("python", *PROBE),
        workdir=AGENT_DIR,  # the Lab's and read-only: whatever the agent did to its workspace
        secret_env={},
        label=f"{spec.label}-probe",
        limits=replace(spec.limits, timeout_s=min(60.0, spec.limits.timeout_s)),
    )
    try:
        result = backend.run(check, log_path)
    except Exception:
        return None
    if result.infrastructure_error is not None or result.timed_out:
        return None
    if result.exit_code is None or result.exit_code == 0:
        return None
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace").strip()[-300:]
    except OSError:
        tail = ""
    return (
        f"the agent's sandbox did not start ({backend.name}, {spec.image}): python "
        f"{' '.join(PROBE)} exited {result.exit_code} in a sandbox like it"
        f"{': ' + tail if tail else ''}; nothing of the agent ran"
    )


def sandbox_failure(
    backend: Backend, spec: SandboxSpec, result: ExecResult, agent_dir: Path
) -> str | None:
    """Why the agent's sandbox ``spec`` failed around the agent (``result`` is then not the
    agent's), or ``None``: the backend's own failure; no exit status at all; an isolating
    sandbox killed from outside (its process, bwrap or the Docker client, died of a signal: the
    agent's own processes cannot reach it, and an agent that runs out of its memory is killed
    inside it, as bwrap's systemd scope continues then); a bootstrap that could not start the
    agent's command; or a sandbox that failed before the bootstrap began, where a sandbox like
    it cannot start Python either (:func:`sandbox_problem`)."""
    if result.infrastructure_error is not None:
        return f"the agent's sandbox failed: {result.infrastructure_error}"
    code = result.exit_code
    if result.timed_out or code == 0:
        return None
    if code is None:
        return "the agent's sandbox did not start: the backend reported no exit status"
    if backend.isolated and code < 0:
        return (
            f"the agent's sandbox was killed by signal {-code} from outside it (an "
            "administrator, the host's out-of-memory killer): not the agent's result"
        )
    summary = _read_json(agent_dir / "bootstrap.json")
    if code == START_FAILED and summary is not None and summary.get("start_error"):
        return f"the agent's command could not be started in its sandbox: {summary['start_error']}"
    if not bootstrap_began(agent_dir):
        return sandbox_problem(backend, spec, agent_dir / PROBE_LOG)
    return None


class AgentMethod(Method):
    """A coding agent that builds a twin in a sandboxed workspace."""

    kind: ClassVar[str] = "agent"
    agent_name: ClassVar[str]
    agent_image: ClassVar[str | None] = None
    """The agent's environment; ``None`` runs it in the task's runtime image."""
    default_allowlist: ClassVar[tuple[str, ...]] = ()
    version_command: ClassVar[tuple[str, ...] | None] = None
    """The CLI's version command, e.g. ``("codex", "--version")``."""
    reports_cost: ClassVar[bool | None] = None
    """Whether its runs can say what they cost (the record's ``cost_usd``); ``None``: whether
    the agent implements :meth:`cost`. See :meth:`cost_reported`."""
    Options: ClassVar[type[MethodOptions]] = AgentOptions

    @property
    def agent_options(self) -> AgentOptions:
        assert isinstance(self.options, AgentOptions)
        return self.options

    @property
    def label(self) -> str:
        model = self.agent_options.model
        return f"{self.name}:{model}" if model else self.name

    # --- hooks --------------------------------------------------------------------------

    @abstractmethod
    def command(self, context: BuildContext, workspace: Workspace) -> list[str]:
        """The agent CLI's argv inside the sandbox; its stdin is the task statement."""

    def credentials(self, context: BuildContext) -> Credentials:
        """Credentials to inject; raise ``SetupError`` if they are missing."""
        return Credentials()

    def allowlist(self) -> tuple[str, ...]:
        return self.agent_options.allowlist or self.default_allowlist

    def credential_status(self) -> CredentialStatus | None:
        """Whether credentials are configured, without fetching them (no commands run, no
        secret read). ``None``: this agent cannot tell without fetching them."""
        return None

    def extra_mounts(self, context: BuildContext) -> list[Mount]:
        return []

    def version(self, backend: Backend, image: str) -> str | None:
        """The agent CLI's version, recorded with every run and in its twin's origin:
        ``version_command`` run offline in the image the agent runs in, by Docker or by the
        backend itself (bwrap); on the host only for a backend that does not isolate (the agent
        is the host's). ``None`` when it cannot be told there, never the host's CLI of that
        name for an agent that runs in an image."""
        if self.version_command is None:
            return None
        if isinstance(backend, DockerBackend):
            return cli_version(backend.docker, image, self.version_command)
        if not backend.isolated:
            return cli_version(None, image, self.version_command)
        return image_cli_version(backend, image, self.version_command)

    def usage(self, trajectory: Path) -> Usage | None:
        return None

    def cost(self, trajectory: Path) -> float | None:
        """The run's cost in US dollars, when the agent reports it."""
        return None

    @classmethod
    def cost_reported(cls) -> bool:
        """Whether this agent's runs can say what they cost: :attr:`reports_cost`, or by
        default whether it implements :meth:`cost`. A cost budget (``max_cost_usd``) limits
        agents by what their runs say they cost: an experiment with one refuses an agent whose
        runs never say it."""
        if cls.reports_cost is not None:
            return cls.reports_cost
        return cls.cost is not AgentMethod.cost

    @staticmethod
    def harness_starts(agent_dir: Path) -> None:
        """Say that a harness on the host started for the agent (:data:`HARNESS_STARTED`), as
        soon as its process exists (``run_captured(started=...)``): a build that then fails on
        this machine before the harness wrote its logs is one whose agent ran (its cost
        unknown), never given back and run again; one whose harness never started is."""
        (agent_dir / HARNESS_STARTED).touch()

    def model_name(self) -> str | None:
        return self.agent_options.model

    def copy_after(self) -> dict[str, str]:
        """In-sandbox files to keep (redacted) after the run: source path → log file name."""
        return {}

    def reasoning_effort(self) -> str | None:
        """The reasoning effort the agent's CLI is given, for its record: the
        ``reasoning_effort`` option, or ``None`` for an agent that passes none to its CLI."""
        return self.agent_options.reasoning_effort

    # --- the flow -------------------------------------------------------------------------

    def image(self, context: BuildContext) -> str:
        """The ``image`` option; otherwise the agent's image for the task's runtime
        (:func:`real2simlab.envs.agent_environment`), so one agent runs on every benchmark."""
        from real2simlab.envs import agent_environment

        environment = context.config.delivery.environment
        if self.agent_options.image:
            return image_reference(self.agent_options.image)
        if self.agent_image is None:
            return image_reference(environment)
        return image_reference(agent_environment(self.agent_image, environment))

    def problems(self, backend: Backend) -> list[str]:
        if self.agent_options.network == "allowlist" and not backend.supports_network_allowlist:
            return [
                f"backend {backend.name!r} cannot restrict the agent's network to an allowlist; "
                'use the docker backend, or set network = "open" (reported in the results)'
            ]
        return []

    def build(self, context: BuildContext) -> BuildOutcome:
        options = self.agent_options
        build_dir = context.work_dir
        agent_dir = build_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        budget = float(budget_of(options, context.config.delivery.agent_time_budget_s))
        workspace = prepare_workspace(context, build_dir / "workspace", budget_s=budget)
        image = self.image(context)
        resolved = context.backend.resolve_image(image)
        credentials = self.credentials(context)
        plan_dir = Path(tempfile.mkdtemp(prefix="r2s-agent-plan-"))
        restore: Callable[[], None] = lambda: None  # noqa: E731
        try:
            secret_env = dict(credentials.env)
            secret_files: dict[str, str] = {}
            for index, (path, content) in enumerate(sorted(credentials.files.items())):
                variable = f"R2S_SECRET_FILE_{index}"
                secret_env[variable] = base64.b64encode(content).decode()
                secret_files[path] = variable
            plan = {
                "argv": self.command(context, workspace),
                "stdin": f"{WORKSPACE}/TASK.md",
                "stdout": f"{LOGS}/trajectory.jsonl",
                "stderr": f"{LOGS}/stderr.log",
                "summary": f"{LOGS}/bootstrap.json",
                "cwd": WORKSPACE,
                "secret_env": sorted(credentials.env),
                "secret_files": secret_files,
                "unset_env": sorted(secret_files.values()),
                "copy_after": {
                    source: f"{LOGS}/{name}" for source, name in self.copy_after().items()
                },
            }
            shutil.copyfile(BOOTSTRAP, plan_dir / "bootstrap.py")
            mounts = [
                Mount(workspace.root.resolve(), WORKSPACE, read_only=False),
                Mount(agent_dir.resolve(), LOGS, read_only=False),
                Mount(plan_dir.resolve(), AGENT_DIR, read_only=True),
                *self.extra_mounts(context),
                *output_mounts(context, workspace),
            ]
            public_env = dict(credentials.public_env)
            if not context.backend.isolated:
                # Without a sandbox, in-sandbox paths in the plan must be host paths.
                for name, target in (("tmp", "/tmp"), ("home", AGENT_HOME)):
                    (plan_dir / name).mkdir()
                    mounts.append(Mount(plan_dir / name, target, read_only=False))
                plan = _rewrite(plan, mounts)
                # The backend rewrites the sandbox's environment (``public_env`` with ``HOME``
                # and ``TMPDIR``) itself: rewritten here too, a host path under a mount target
                # (``/tmp/r2s-agent-plan-...``) would be rewritten again, to nowhere.
            (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            run_as = agent_user(context.backend)
            if run_as is not None and isinstance(context.backend, DockerBackend):
                # The agent runs as an unprivileged uid; let it use its directories.
                restore = hand_over([workspace.root, agent_dir], run_as)
                for path in [plan_dir, *plan_dir.rglob("*")]:
                    path.chmod(0o755 if path.is_dir() else 0o644)
                unreadable = [
                    problem
                    for mount in mounts
                    if mount.read_only
                    for problem in unreadable_by_others(Path(mount.source))
                ]
                if unreadable:
                    raise SetupError(
                        f"the agent runs as uid {run_as} (this process is root), but it cannot "
                        f"read what is mounted for it: {'; '.join(unreadable[:3])}",
                        fix="make these files readable by other users (chmod -R o+rX), or run "
                        "the Lab as a regular user",
                    )
            started = datetime.now(UTC)
            with self._network(context, agent_dir) as network:
                spec = SandboxSpec(
                    image=resolved.reference,
                    argv=("python", f"{AGENT_DIR}/bootstrap.py", f"{AGENT_DIR}/plan.json"),
                    workdir=WORKSPACE,
                    mounts=tuple(mounts),
                    env={"HOME": AGENT_HOME, "TMPDIR": "/tmp", **public_env},
                    tmpfs=("/tmp", AGENT_HOME),
                    secret_env=secret_env,
                    network=network,
                    limits=Limits(
                        timeout_s=budget + GRACE_S,
                        cpus=options.cpus,
                        memory_mb=options.memory_mb,
                        # The agent tests its program where it will be re-run: with the GPU the
                        # task grants (e.g. Blender renders in 4DCodeBench and PhysCodeGym).
                        gpus=options.gpus if context.config.delivery.resources.gpu else None,
                    ),
                    label=f"r2s-agent-{self.name}",
                    user=run_as,
                )
                result = context.backend.run(spec, agent_dir / "sandbox.log")
                # Asked while the sandbox's plan, network and ownership are as the agent had them.
                outside = sandbox_failure(context.backend, spec, result, agent_dir)
        finally:
            shutil.rmtree(plan_dir, ignore_errors=True)
            restore()
        duration = (datetime.now(UTC) - started).total_seconds()
        if outside is not None:
            # Not a result either way. If nothing of the agent ran, its run is given back and
            # tried again (``infra_error``); one whose command had started was spent: counted,
            # and not run again by this command (its own status; the build says why).
            status = "failed" if agent_started(agent_dir) else "infra_error"
        elif result.timed_out:
            status = "timed_out"
        elif result.exit_code == 0:
            status = "completed"
        else:
            status = "failed"
        # Before anything is packaged: what the agent wrote itself, not through its output. A
        # sandbox that failed around the agent (``outside``) delivers nothing, and what an agent
        # that had started wrote is cleaned all the same.
        twin, failure, infra, scrubbed = self.collect_clean(
            context, workspace, status, injected_values(credentials), outside
        )
        if infra is None and failure is not None:
            # An agent that exits normally with no (or a cut) delivery because this machine's
            # storage stopped its writes (a file at the size limit, a full disk): not a result.
            from real2simlab.sandbox.process import storage_problem

            infra = storage_problem(agent_dir / "sandbox.log", [workspace.root])
        # The agent could write its log directory and workspace: what it left there is read only
        # as regular files (never through a link it left, never waiting on a pipe), and the
        # Lab's record is written as a new file, never through what it left under that name.
        summary = _read_json(agent_dir / "bootstrap.json")
        trajectory = agent_dir / "trajectory.jsonl"

        def clean(name: str) -> str | None:
            """``agent/<name>`` for the record, if it is there and holds no injected value."""
            relative = f"agent/{name}"
            there = is_regular(agent_dir / name) and relative not in scrubbed.left
            return relative if there else None

        record = AgentRecord(
            method=self.name,
            agent=self.agent_name,
            agent_version=self.version(context.backend, resolved.reference),
            model=self.model_name(),
            reasoning_effort=self.reasoning_effort(),
            instruction_template=TEMPLATE_VERSION,
            instruction_sha256=hashlib.sha256(workspace.statement.encode()).hexdigest(),
            task_id=context.task_id,
            task_hash=context.task_hash,
            trial=context.trial,
            image={"reference": resolved.reference, "digest": resolved.digest},
            backend=context.backend.info(),
            network={
                **network.to_json(),
                "log": "agent/egress.jsonl" if network.log_path else None,
            },
            started_at=started,
            duration_s=round(duration, 1),
            time_budget_s=budget,
            exit_code=result.exit_code,
            exit_status=status,
            # The record names a log only once it holds no injected value (and only when it is a
            # regular file): the report and an exported dataset show the trajectory it names.
            trajectory=clean("trajectory.jsonl"),
            stderr=clean("stderr.log"),
            last_message=clean("last_message.md"),
            usage=self.usage(trajectory) if is_regular(trajectory) else None,
            cost_usd=self.cost(trajectory) if is_regular(trajectory) else None,
            redactions=_redactions(summary) + scrubbed.redactions,
            delivered=twin is not None,
            failure=failure,
        )
        write_new(agent_dir / "agent_record.json", record.dumps().encode("utf-8"))
        session = session_output_of(
            context,
            workspace.root / convention_spec(context.config.delivery.convention).session_output,
        )
        return BuildOutcome(
            twin=twin,
            failure=failure,
            agent_record=agent_dir / "agent_record.json",
            session_output=session,
            network=network.mode.value,
            infra_error=infra,
        )

    def _network(self, context: BuildContext, agent_dir: Path) -> _NetworkScope:
        return _NetworkScope(self, context, agent_dir)

    def _collect(
        self, context: BuildContext, workspace: Workspace, status: str
    ) -> tuple[Any, str | None]:
        solution = workspace.solution
        convention = context.config.delivery.convention
        entry = convention_spec(convention).entry_file
        prefix = "" if status == "completed" else f"the agent {status.replace('_', ' ')}; "
        linked = directory_problem(solution)  # never looked through: the agent made the name
        if linked is not None:
            return None, f"{prefix}{linked}"
        if not (solution / entry).is_file():
            return None, f"{prefix}no {entry} in solution/"
        staging = Path(tempfile.mkdtemp(prefix="r2s-solution-")) / "program"
        try:
            clean_solution(solution, staging)
            twin = package_twin(
                context,
                staging,
                method=self.name,
                agent=AgentInfo(
                    name=self.agent_name, version=self.version(context.backend, self.image(context))
                ),
                model=self.model_name(),
            )
        except InfrastructureError:
            raise
        except (InvalidTwinError, R2SError, ValueError) as error:
            return None, f"{prefix}the solution is not a valid twin: {error}"
        except OSError as error:  # writing the twin failed on this machine
            raise InfrastructureError(
                f"this machine could not package the solution: {os_error_text(error)}"
            ) from error
        finally:
            shutil.rmtree(staging.parent, ignore_errors=True)
        return twin, None

    def collect(
        self, context: BuildContext, workspace: Workspace, status: str
    ) -> tuple[Any, str | None, str | None]:
        """:meth:`_collect`, with a failure of this machine (not the agent's) as the third
        value instead of an exception: the build is then not a result, and is retried."""
        try:
            twin, failure = self._collect(context, workspace, status)
        except InfrastructureError as error:
            return None, str(error), str(error)
        return twin, failure, None

    def collect_clean(
        self,
        context: BuildContext,
        workspace: Workspace,
        status: str,
        values: Sequence[str],
        outside: str | None = None,
    ) -> tuple[Any, str | None, str | None, Scrubbed]:
        """:meth:`collect`, once the injected ``values`` are out of the files the agent wrote
        itself under the build's work directory (:func:`scrub_injected`), and what that did.
        Nothing is delivered while a file may still hold one, nor when the run failed outside
        the agent (``outside`` says why: its sandbox, Harbor, the harness): that reason is then
        the failure and the infrastructure error, with what the cleaning could not do."""
        scrubbed = scrub_injected(context, context.work_dir, values)
        if outside is not None:  # not the agent's result, whatever its files hold
            said = scrubbed.infra or scrubbed.failure or scrubbed.note
            reason = outside if said is None else f"{outside}; {said}"
            return None, reason, reason, scrubbed
        if scrubbed.infra is not None:
            return None, scrubbed.infra, scrubbed.infra, scrubbed
        if scrubbed.failure is not None:
            return None, scrubbed.failure, None, scrubbed
        twin, failure, infra = self.collect(context, workspace, status)
        if failure is not None and scrubbed.note is not None:
            failure = f"{failure}; {scrubbed.note}"
        return twin, failure, infra, scrubbed


class _NetworkScope:
    """The agent's network for the duration of its run."""

    def __init__(self, method: AgentMethod, context: BuildContext, agent_dir: Path) -> None:
        self.method = method
        self.context = context
        self.agent_dir = agent_dir
        self._proxy: Any = None

    def __enter__(self) -> NetworkPolicy:
        mode = self.method.agent_options.network
        if mode == "none":
            return NetworkPolicy(mode=NetworkMode.NONE)
        if mode == "open":
            return NetworkPolicy(mode=NetworkMode.OPEN)
        backend = self.context.backend
        if not isinstance(backend, DockerBackend):
            raise SetupError(
                f"the {backend.name!r} backend cannot restrict an agent's network to an allowlist",
                fix='use --backend docker, or set the method option network = "open"',
            )
        from real2simlab.sandbox.egress import EgressProxy

        self._proxy = EgressProxy(
            image=image_reference("r2s-python"),
            allowlist=self.method.allowlist(),
            log_dir=self.agent_dir,
            docker=backend.docker,
        )
        return self._proxy.__enter__()

    def __exit__(self, *exc: object) -> None:
        if self._proxy is not None:
            self._proxy.__exit__(None, None, None)


def _rewrite(value: Any, mounts: Sequence[Mount]) -> Any:
    if isinstance(value, str):
        return rewrite_paths(value, mounts)
    if isinstance(value, list):
        return [_rewrite(item, mounts) for item in value]
    if isinstance(value, dict):
        return {_rewrite(key, mounts): _rewrite(item, mounts) for key, item in value.items()}
    return value


def _read_json(path: Path) -> dict[str, Any] | None:
    """The JSON object in ``path``, a file an agent's sandbox could write: ``None`` when it is
    not a regular file (:func:`~real2simlab._fs.read_regular`: a symbolic link or a pipe is
    neither followed nor waited on) or holds no JSON object."""
    try:
        data = json.loads(read_regular(path) or b"")
    except (OSError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _redactions(summary: Mapping[str, Any] | None) -> int:
    """How many values the bootstrap redacted, as its summary (``bootstrap.json``) says: 0 when
    it says no count (the agent could write that file too)."""
    value = summary.get("redactions") if summary else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def total_usage(events: Sequence[Mapping[str, Any]]) -> Usage | None:
    """Sum the ``usage`` objects found in a trajectory's events (Codex, Claude Code style)."""
    totals: dict[str, int] = {}
    found = False
    for event in events:
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue
        found = True
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    if not found:
        return None

    def first(*names: str) -> int | None:  # a count of 0 is a count, not one that is missing
        return next((totals[name] for name in names if name in totals), None)

    inputs = totals.get("input_tokens")
    outputs = totals.get("output_tokens")
    return Usage(
        input_tokens=inputs,
        cached_input_tokens=first("cached_input_tokens", "cache_read_input_tokens"),
        output_tokens=outputs,
        reasoning_tokens=first("reasoning_output_tokens", "reasoning_tokens"),
        total_tokens=(inputs or 0) + (outputs or 0)
        if inputs is not None or outputs is not None
        else None,
    )


def read_trajectory(path: Path) -> list[dict[str, Any]]:
    """The events of an agent's trajectory, the file its record names: the JSON objects of a
    JSON-lines file (``trajectory.jsonl``: the agent CLI's events), or the messages of one JSON
    document (``trajectory.json``: mini-swe-agent's); none of a document that is not JSON (one
    cut at its size limit)."""
    if path.suffix != ".json":
        return read_events(path)
    try:
        data: Any = json.loads((read_regular(path) or b"").decode("utf-8", errors="replace"))
    except (OSError, ValueError, RecursionError):
        return []
    items: Any = data.get("messages") if isinstance(data, dict) else data
    if isinstance(items, list):
        return [cast(dict[str, Any], item) for item in items if isinstance(item, dict)]
    return [cast(dict[str, Any], data)] if isinstance(data, dict) else []


def read_events(path: Path) -> list[dict[str, Any]]:
    """JSON objects of a JSON-lines file (an agent's trajectory, which its sandbox could
    write); other lines are skipped. A symbolic link, a pipe or anything but a regular file
    holds no events: it is neither followed nor waited on
    (:func:`~real2simlab._fs.read_regular`)."""
    events: list[dict[str, Any]] = []
    data = read_regular(path)
    if data is None:
        return events
    for line in data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(value, dict):
            events.append(value)
    return events
