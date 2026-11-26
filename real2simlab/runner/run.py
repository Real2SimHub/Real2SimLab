"""Run a frozen twin once, in a sandbox, with one seed.

A run directory holds::

    run_record.json   the record (r2s-run/1.0)
    output/           what the program wrote (R2S_OUTPUT_DIR)
    log.txt           the program's stdout and stderr (at most 16 MiB: the beginning and
                      the end, see ``sandbox.process.LOG_LIMIT_BYTES``)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from real2simlab import __version__
from real2simlab._fs import force_remove, make_owner_accessible
from real2simlab.envs import image_reference
from real2simlab.errors import (
    OutputExistsError,
    R2SError,
    SetupError,
    UsageError,
    host_failure,
)
from real2simlab.formats.base import Requirements, get_format, output_problems
from real2simlab.runner.record import ImageInfo, RunLimits, RunRecord, RunStatus
from real2simlab.sandbox.base import Backend, Limits, Mount, NetworkPolicy, SandboxSpec
from real2simlab.sandbox.process import storage_problem
from real2simlab.task.model import Resources
from real2simlab.twin.contract import (
    ASSETS_DIR_ENV,
    MAX_SEED,
    OUTPUT_DIR_ENV,
    SANDBOX_ASSETS,
    SEED_ENV,
)
from real2simlab.twin.fingerprint import fingerprint
from real2simlab.twin.io import LoadedTwin

RECORD_FILE = "run_record.json"
OUTPUT_DIR = "output"
LOG_FILE = "log.txt"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class FrozenTwinModifiedError(R2SError, RuntimeError):
    """The twin no longer matches the fingerprint taken when it was loaded."""


CHANGED_FROM_OUTSIDE = (
    "the frozen twin changed while it ran, from outside its sandbox (the program cannot write "
    "it there), so the result is not the program's; run it again once nothing else writes to it"
)
STORAGE_EXPLAINS = (RunStatus.FAILED, RunStatus.INVALID_OUTPUT, RunStatus.TIMED_OUT)
"""How a run ends that this machine's storage stopped: its program fails on the failed write,
exits 0 with a missing or cut output, or retries the write until its time limit."""


def twin_changed(twin: LoadedTwin) -> bool:
    """Whether ``twin`` no longer has the fingerprint it was loaded with: a file of its package
    was changed, added or removed (also while it was being read), or the package can no longer
    be read as one (a symbolic link was put in it). Any other error reading it (no permission,
    an I/O error) is raised: it does not show that the twin changed, and it is not a result of
    its program."""
    try:
        return twin.current_fingerprint() != twin.fingerprint
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError, ValueError):
        return True


@dataclass(frozen=True)
class RunResult:
    record: RunRecord
    run_dir: Path

    @property
    def output_dir(self) -> Path:
        return self.run_dir / OUTPUT_DIR

    @property
    def succeeded(self) -> bool:
        return self.record.status is RunStatus.SUCCEEDED


def limits_from_resources(
    resources: Resources, timeout_s: float, gpu: str | None = None
) -> RunLimits:
    return RunLimits(
        timeout_s=timeout_s,
        cpus=resources.cpus,
        memory_mb=resources.memory_mb,
        gpu=(gpu or "all") if resources.gpu else None,
    )


def run_twin(
    twin: LoadedTwin,
    run_dir: Path,
    *,
    backend: Backend,
    label: str = "run",
    seed: int = 0,
    limits: RunLimits | None = None,
    requirements: Requirements | None = None,
    attempt: int = 1,
) -> RunResult:
    """Run ``twin`` once and record the outcome in ``run_dir``, which must not exist.

    Caller errors raise before ``run_dir`` is created: a modified twin, an unavailable image,
    an out-of-range seed, an existing ``run_dir``. The program's own failures are recorded in
    the result instead.
    """
    limits = limits or RunLimits(timeout_s=300)
    run_dir = Path(run_dir)
    if twin_changed(twin):
        raise FrozenTwinModifiedError(f"twin {twin.manifest.twin_id!r} changed after it was loaded")
    manifest = twin.manifest
    if not 0 <= seed < MAX_SEED:
        raise UsageError(f"seed must be in [0, {MAX_SEED})")
    if run_dir.exists():
        raise OutputExistsError(
            f"run directory already exists: {run_dir}", fix="choose a new --out directory"
        )
    fmt = get_format(manifest.output_format)
    image = backend.resolve_image(image_reference(manifest.environment.image))
    try:  # the run's own: another run given the same directory meanwhile is refused here
        run_dir.mkdir(parents=True)
    except FileExistsError:
        raise OutputExistsError(
            f"run directory already exists: {run_dir} (another run made it just now)",
            fix="choose a new --out directory",
        ) from None

    output_dir = run_dir / OUTPUT_DIR
    runtime = manifest.runtime
    scratch: Path | None = None
    if runtime.replaceable_output:
        # The output's parent is a writable scratch directory, so the program may delete and
        # recreate the output directory; only that directory is kept afterwards.
        scratch = run_dir / "scratch"
        (scratch / PurePosixPath(runtime.output_path).name).mkdir(parents=True)
        mounts = [
            Mount(
                scratch.resolve(), str(PurePosixPath(runtime.output_path).parent), read_only=False
            ),
            Mount(twin.program_dir.resolve(), runtime.program_path, read_only=True),
        ]
    else:
        output_dir.mkdir(parents=True)
        mounts = [
            Mount(twin.program_dir.resolve(), runtime.program_path, read_only=True),
            Mount(output_dir.resolve(), runtime.output_path, read_only=False),
        ]
    env = {OUTPUT_DIR_ENV: manifest.runtime.output_path, SEED_ENV: str(seed)}
    if manifest.assets and twin.assets_dir.is_dir():
        mounts.append(Mount(twin.assets_dir.resolve(), SANDBOX_ASSETS, read_only=True))
        env[ASSETS_DIR_ENV] = SANDBOX_ASSETS
    if limits.cpus is not None:
        threads = str(max(1, int(limits.cpus)))
        env.update({name: threads for name in THREAD_VARIABLES})
    spec = SandboxSpec(
        image=image.reference,
        argv=manifest.entrypoint,
        workdir=manifest.runtime.program_path,
        mounts=tuple(mounts),
        env=env,
        network=NetworkPolicy(),
        limits=Limits(
            timeout_s=limits.timeout_s,
            cpus=limits.cpus,
            memory_mb=limits.memory_mb,
            gpus=limits.gpu,
        ),
        label="r2s-twin",
    )
    started_at = datetime.now(UTC)
    result = backend.run(spec, run_dir / LOG_FILE)
    collection_problems: list[str] = []
    # The twin controlled what it wrote (its output or scratch tree, and on the local backend
    # its temporary directory, also in the run directory): restore access first.
    make_owner_accessible(run_dir)
    # Whether this machine's storage can have stopped the program: a file it wrote that reached
    # the file-size limit anywhere in the run directory (its output or scratch tree first, then
    # the log and the local backend's temporary directory), or a full disk. Read before the
    # scratch tree is removed; it counts only for a run that did not succeed (below).
    storage = storage_problem(
        run_dir / LOG_FILE, [scratch if scratch is not None else output_dir, run_dir]
    )
    if scratch is not None:
        # Move only the output directory (a rename, never a copy) and delete the rest of the
        # scratch tree without recursion.
        produced = scratch / PurePosixPath(runtime.output_path).name
        if produced.is_dir() and not produced.is_symlink():
            try:
                os.rename(produced, output_dir)
            except OSError as error:
                collection_problems.append(f"the output directory could not be collected: {error}")
        if not output_dir.is_dir():
            output_dir.mkdir()
        cleanup = force_remove(scratch)
        if cleanup:
            collection_problems.append(cleanup)
    status, problems = _outcome(
        twin,
        result.exit_code,
        result.timed_out,
        result.infrastructure_error,
        isolated=backend.isolated,
    )
    if status is RunStatus.FAILED and result.exit_code not in (None, 0) and not result.timed_out:
        broken = runtime_problem(backend, spec, run_dir / PROBE_LOG)
        if broken is not None:  # nothing of the program ran: this machine's failure
            status, problems = RunStatus.INFRA_ERROR, [broken]
    output_fingerprint: str | None = None
    if status is RunStatus.SUCCEEDED:
        reqs = requirements or Requirements()
        # The output is the program's: whatever reading it raises (a truncated archive, a
        # value of the wrong kind) is a problem of the output, never a crash of the Lab; but
        # this machine failing to read it (no permission, an I/O error, a stale file handle:
        # :func:`host_failure`) is not the program's result.
        try:
            problems = [*collection_problems, *output_problems(fmt, output_dir, reqs)]
        except SetupError:
            raise  # this machine cannot check outputs of the format (e.g. a missing checker)
        except Exception as error:
            status, problems = _unread_output(error, "checked")
        if status is RunStatus.SUCCEEDED and problems:
            status = RunStatus.INVALID_OUTPUT
        elif status is RunStatus.SUCCEEDED:
            try:
                output_fingerprint = fingerprint(output_dir)
            except Exception as error:
                status, problems = _unread_output(error, "read")
    if storage is not None and status in STORAGE_EXPLAINS:
        # A program that catches its failed write (or ignores a writer's exit status) exits 0
        # with a missing or cut output, or retries it until its time limit: when this machine's
        # storage explains that (a file at the file-size limit, a full disk), it is not the
        # program's result.
        status, problems = RunStatus.INFRA_ERROR, [storage]
    record = RunRecord(
        twin_id=manifest.twin_id,
        twin_fingerprint=twin.fingerprint,
        program_fingerprint=twin.program_fingerprint,
        image=ImageInfo(reference=image.reference, digest=image.digest),
        backend=backend.info(),
        label=label,
        seed=seed,
        limits=limits,
        started_at=started_at,
        duration_s=round(result.duration_s, 3),
        exit_code=result.exit_code,
        status=status,
        problems=tuple(problems),
        output_format=manifest.output_format,
        output_fingerprint=output_fingerprint,
        lab_version=__version__,
        attempt=attempt,
    )
    write_record(run_dir / RECORD_FILE, record)
    return RunResult(record=record, run_dir=run_dir)


PROBE_LOG = "runtime-probe.log"
PROBES: dict[str, tuple[str, ...]] = {
    "python": ("-I", "-c", "pass"),  # isolated: nothing of the program directory is imported
    "python3": ("-I", "-c", "pass"),
    "bash": ("-c", "true"),
    "sh": ("-c", "true"),
}
"""Commands that start an entrypoint's interpreter and nothing else: they run in any working
runtime, whatever the program does."""
PROMISED = ("numpy", "scipy", "r2s_twin")
"""What every Lab runtime holds (``r2s-python`` and the images built on it, and the ``local``
backend: the Lab's own dependencies and the SDK), with ``ffmpeg``."""
MISSING = "r2s-runtime-missing:"
PROMISE_PROBE = (
    "-I",  # isolated: nothing of the program directory is imported, the SDK is added below
    "-B",  # and nothing is written next to what it imports (-I ignores PYTHONDONTWRITEBYTECODE)
    "-c",
    "import os, shutil, sys\n"
    "sys.path[:0] = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep)\n"
    "                if p and os.path.abspath(p) != os.getcwd()]\n"
    "missing = []\n"
    f"for name in {PROMISED!r}:\n"
    "    try:\n"
    "        __import__(name)\n"
    "    except Exception as error:\n"
    "        missing.append(f'{name} ({type(error).__name__}: {error})')\n"
    "if shutil.which('ffmpeg') is None:\n"
    "    missing.append('ffmpeg (not on PATH)')\n"
    f"if missing:\n    print({MISSING!r}, '; '.join(missing))\n    sys.exit(3)\n",
)
"""For a Python program in a Lab runtime: the interpreter starts and what the runtime promises
is there. A program that fails for lack of it failed for the runtime, not for itself."""


def runtime_problem(backend: Backend, spec: SandboxSpec, log_path: Path) -> str | None:
    """Why the runtime of ``spec`` cannot run a program (a damaged image or rootfs, a missing
    interpreter, a Lab runtime without what it promises: :data:`PROMISED` and ffmpeg), or
    ``None`` if it can or this cannot be told. Asked after a program failed: such a failure is
    not the program's. It is asked every time (a runtime that worked earlier in the same
    session may have been damaged since)."""
    from real2simlab.envs import registered_environment_for

    interpreter = spec.argv[0] if spec.argv else ""
    probe = PROBES.get(PurePosixPath(interpreter).name)
    if probe is None:
        return None
    lab_runtime = backend.name == "local" or registered_environment_for(spec.image) is not None
    if lab_runtime and probe == PROBES["python"]:
        probe = PROMISE_PROBE
    check = replace(
        spec,
        argv=(interpreter, *probe),
        label="r2s-probe",
        limits=replace(spec.limits, timeout_s=min(60.0, spec.limits.timeout_s)),
    )
    try:
        result = backend.run(check, log_path)
    except Exception:  # the probe is a diagnosis: when it cannot run, the result stands
        return None
    if result.exit_code == 0:
        return None
    if result.infrastructure_error is not None or result.timed_out or result.exit_code is None:
        return None
    try:
        log = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        log = ""
    sandbox = backend.setup_problem(log)
    if sandbox is not None:  # the backend's own failure: the runtime never started
        return (
            f"the {backend.name} sandbox could not be set up on this machine: {sandbox.message}. "
            "Nothing of the program ran, and the runtime is not at fault (unpacking or pulling "
            "it again does not help)" + (f". Fix: {sandbox.fix}" if sandbox.fix else "")
        )
    repair = _repair(backend, spec.image)
    found = [line for line in log.splitlines() if line.startswith(MISSING)]
    if found:
        missing = found[-1][len(MISSING) :].strip()[:400]
        return (
            f"the runtime {spec.image or backend.name} lacks what every Lab runtime holds: "
            f"{missing}. A program cannot be judged on it. {repair}"
        )
    tail = log.strip()[-300:]
    return (
        f"the runtime {spec.image} cannot start {interpreter} ({interpreter} "
        f"{' '.join(PROBES[PurePosixPath(interpreter).name])} exited {result.exit_code}"
        f"{': ' + tail if tail else ''}): nothing of the program ran. {repair}"
    )


def _repair(backend: Backend, image: str) -> str:
    """How to repair ``backend``'s runtime ``image``: its own command, named."""
    from real2simlab.envs import ENVIRONMENTS, registered_environment_for
    from real2simlab.envs.rootfs import unpack_again

    name = registered_environment_for(image)
    if backend.name == "docker":
        how = (
            f"docker pull {image}, or build it again"
            if name is None
            else f"r2s env pull {name} or r2s env build {name}"
            if ENVIRONMENTS[name].publishable
            else f"r2s env build {name}"
        )
    elif backend.name == "bwrap":
        how = unpack_again(image)
    elif backend.name == "local":
        how = "pip install -e . in your clone of Real2SimLab, and ffmpeg"
    else:
        how = (
            "Docker: r2s env pull or r2s env build; bwrap: r2s env unpack NAME --force; "
            "local: pip install -e . and ffmpeg"
        )
    return f"Repair the runtime ({how}) and run again"


def _outcome(
    twin: LoadedTwin,
    exit_code: int | None,
    timed_out: bool,
    infrastructure_error: str | None,
    *,
    isolated: bool,
) -> tuple[RunStatus, list[str]]:
    if infrastructure_error is not None:
        return RunStatus.INFRA_ERROR, [infrastructure_error]
    if twin_changed(twin):
        # An isolating backend mounts the program read-only: the change came from outside the
        # sandbox (an editor, a sync client). Without isolation the program may have made it.
        if isolated:
            return RunStatus.INFRA_ERROR, [CHANGED_FROM_OUTSIDE]
        return RunStatus.FAILED, ["the twin changed during the run"]
    if timed_out:
        return RunStatus.TIMED_OUT, ["killed at the time limit"]
    if exit_code is None:
        return RunStatus.INFRA_ERROR, ["the program did not start"]
    if exit_code != 0:
        return RunStatus.FAILED, [f"exit status {exit_code}; see {LOG_FILE}"]
    return RunStatus.SUCCEEDED, []


def _unread_output(error: Exception, step: str) -> tuple[RunStatus, list[str]]:
    """The outcome of a run whose output could not be ``step`` (checked, read) for ``error``:
    ``infra_error`` when this machine failed to read it (:func:`host_failure`: the run is done
    again, never scored), else ``invalid_output``, a problem of the output itself."""
    what = f"{type(error).__name__}: {error}"
    if host_failure(error):
        return RunStatus.INFRA_ERROR, [f"this machine could not read the output: {what}"]
    return RunStatus.INVALID_OUTPUT, [f"the output could not be {step}: {what}"]


def write_record(path: Path, record: RunRecord) -> None:
    """Write a record atomically, so an interrupted run never leaves a half-written file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(record.dumps(), encoding="utf-8")
    temporary.replace(path)


def read_record(run_dir: Path) -> RunRecord | None:
    path = Path(run_dir) / RECORD_FILE
    if not path.is_file():
        return None
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
