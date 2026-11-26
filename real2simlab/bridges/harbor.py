"""Export a Real2SimLab task as a Harbor task (https://harborframework.com).

Harbor then runs any of its agents on it, in its own sandboxes. The exported task has:

- ``instruction.md``: the Lab's task statement for a workspace at ``/app``;
- ``environment/``: the agent's image, ``FROM`` the task's runtime image, with the visible
  data, reference documents and the SDK in ``/app``, and an empty ``/app/solution``;
- ``tests/``: a separate, offline verifier ``FROM`` the ``r2s-verifier`` image, holding the
  whole task (visible and hidden) and a copy of this Real2SimLab. It freezes ``/app/solution``
  into a twin, re-runs it offline with the same code the Lab uses, and writes ``reward.json``;
  the reward is the reproduction skill;
- ``solution/solve.sh`` (optional): writes a known twin into ``/app/solution``, so Harbor's
  ``oracle`` agent checks the task end to end.

The hidden data exist only in the verifier image, which agents never see. The task is copied
without following a symbolic link and validated as ``r2s eval`` validates it, and everything is
written from that copy: a task that shows hidden data in ``visible/`` or its statement is not
exported.
Third-party code is not bundled: the 4DCodeBench checker, when a task needs it, is fetched in
the verifier image at its pinned commit.

With ``self_contained``, both Dockerfiles start from ``python:3.11-slim-bookworm`` and install
everything themselves, so the task builds anywhere, e.g. for a public registry.
"""

from __future__ import annotations

import base64
import difflib
import errno
import importlib.metadata
import shlex
import shutil
from pathlib import Path

import tomli_w

from real2simlab import __version__
from real2simlab._fs import clashing_input, copy_tree, force_remove, release, staging_dir
from real2simlab.agents.docs import CAPTURE, RUN_CONTRACT, sdk_doc
from real2simlab.envs import image_reference
from real2simlab.errors import InputNotFoundError, OutputExistsError, SetupError, UsageError
from real2simlab.formats import get_format
from real2simlab.task.instruction import task_statement
from real2simlab.task.io import Task, copy_exactly, load_task
from real2simlab.task.model import DeliveryConvention, Protocol
from real2simlab.task.validate import invalid_task_error, validate_task
from real2simlab.twin import program_of
from real2simlab.twin.conventions import convention_spec
from real2simlab.twin.sdk import sdk_path
from real2simlab.vendor import FOURDCODEBENCH

APP = "/app"
LAB_DIR = "lab"
TASK_DIR = "task"
PACKAGE = Path(__file__).resolve().parents[1]

FINISH = f"""

## Delivering in this environment

Write your program to `{APP}/solution/` as described above; it is collected when you finish.
Running it once with `R2S_OUTPUT_DIR={APP}/output` lets the evaluator compare your own output
with its frozen re-run.
"""

SELF_CONTAINED_RUNTIME = """\
FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \\
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3
COPY r2s_twin.py /opt/r2s/sdk/r2s_twin.py
ENV PYTHONPATH=/opt/r2s/sdk PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
"""

VERIFIER_EXTRAS = "pydantic=={pydantic} tomli-w=={tomli_w}"

TEST_SH = f"""\
#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
python -m real2simlab.bridges.harbor_verify \\
  --task /tests/{TASK_DIR} --solution {APP}/solution --session-output {APP}/output \\
  --logs /logs/verifier
"""


def _versions() -> dict[str, str]:
    return {
        "pydantic": importlib.metadata.version("pydantic"),
        "tomli_w": importlib.metadata.version("tomli-w"),
    }


def _entry_points() -> str:
    text = importlib.metadata.distribution("real2simlab").read_text("entry_points.txt")
    return text or ""


def copy_lab(target: Path) -> Path:
    """This Real2SimLab as an importable directory, with a dist-info so its plugins are found."""
    target.mkdir(parents=True)
    copy_tree(
        PACKAGE, target / "real2simlab", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    info = target / f"real2simlab-{__version__}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: real2simlab\nVersion: {__version__}\n", encoding="utf-8"
    )
    (info / "entry_points.txt").write_text(_entry_points(), encoding="utf-8")
    (info / "top_level.txt").write_text("real2simlab\n", encoding="utf-8")
    return target


def _workspace(task: Task, target: Path) -> None:
    changed = copy_exactly(task.visible_dir, target / "visible")
    if changed is not None:
        raise SetupError(
            f"task {task.id} changed while it was exported: {changed}",
            fix="export it again once nothing edits the task",
        )
    docs = target / "docs"
    docs.mkdir(parents=True)
    fmt = get_format(task.config.delivery.output_format)
    for name, text in {
        "output-format.md": fmt.documentation(),
        "run-contract.md": RUN_CONTRACT,
        "capture.md": CAPTURE,
        "sdk.md": sdk_doc(),
    }.items():
        (docs / name).write_text(text, encoding="utf-8")
    (target / "tools").mkdir()
    shutil.copyfile(sdk_path(), target / "tools" / "r2s_twin.py")


def _environment_dockerfile(task: Task, self_contained: bool) -> str:
    base = (
        SELF_CONTAINED_RUNTIME
        if self_contained
        else (f"FROM {image_reference(task.config.delivery.environment)}\n")
    )
    return base + (
        f"WORKDIR {APP}\n"
        f"COPY workspace/ {APP}/\n"
        f"RUN mkdir -p {APP}/solution {APP}/output && chmod -R a+rwX {APP}\n"
    )


def _verifier_dockerfile(task: Task, self_contained: bool) -> str:
    versions = _versions()
    if self_contained:
        lines = [
            SELF_CONTAINED_RUNTIME.replace(
                "COPY r2s_twin.py", "COPY lab/real2simlab/twin/sdk/r2s_twin.py"
            ),
            f"RUN pip install --no-cache-dir {VERIFIER_EXTRAS.format(**versions)}\n",
        ]
    else:
        lines = [f"FROM {image_reference('r2s-verifier')}\n"]
    if task.config.delivery.output_format == "4d_world":
        # Fetched, not bundled: 4DCodeBench has no license that allows redistributing it.
        repo = FOURDCODEBENCH
        lines.append(
            "RUN (command -v git >/dev/null || (apt-get update && apt-get install -y "
            "--no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*)) "
            "&& git init -q /opt/4DCodeBench "
            "&& for attempt in 1 2 3; do "
            f"git -C /opt/4DCodeBench fetch -q --depth 1 {repo.url} {repo.commit} && break; "
            "sleep 10; done "
            f"&& git -C /opt/4DCodeBench checkout -q {repo.commit}\n"
            f"ENV {repo.override_env}=/opt/4DCodeBench\n"
        )
    lines += [
        f"COPY {LAB_DIR}/ /opt/r2s/lab/\n",
        "ENV PYTHONPATH=/opt/r2s/lab:/opt/r2s/sdk\n",
        "COPY . /tests/\n",
    ]
    return "".join(lines)


def _task_toml(task: Task, timeout_s: float | None) -> dict[str, object]:
    config = task.config
    delivery = config.delivery
    name = "real2simlab/" + config.id.replace("/", "-").replace(".", "-")
    verifier_timeout = 120.0 + delivery.run_timeout_s * 2
    return {
        "schema_version": "1.4",
        "artifacts": [f"{APP}/solution", f"{APP}/output"],
        "task": {
            "name": name,
            "description": f"{config.title}: build a simulation of a recorded physical event.",
            "keywords": ["real2sim", "physics", "simulation", "video", "digital-twin"],
        },
        "metadata": {
            "real2simlab_version": __version__,
            "task_id": config.id,
            "task_version": config.version,
            "protocol": config.scoring.protocol.value,
            "source_dataset": config.source.dataset,
            "license": config.license,
        },
        "agent": {
            "timeout_sec": float(
                timeout_s if timeout_s is not None else delivery.agent_time_budget_s
            )
        },
        "environment": {
            "build_timeout_sec": 1800.0,
            "cpus": max(1, int(delivery.resources.cpus)),
            "memory_mb": max(2048, delivery.resources.memory_mb),
            "network_mode": "allowlist",
        },
        "verifier": {
            "environment_mode": "separate",
            "timeout_sec": verifier_timeout,
            "environment": {
                "build_timeout_sec": 1800.0,
                "cpus": max(1, int(delivery.resources.cpus)),
                "memory_mb": max(2048, delivery.resources.memory_mb),
                "network_mode": "no-network",
            },
        },
    }


ORACLE = "oracle"
"""``--solution oracle``: the task's truth generator."""
ENTRY = convention_spec(DeliveryConvention.R2S).entry_file
"""The file a solution's program must hold (``run.py``): the only convention exported."""


def solution_program(solution: Path, *, given: str, fix: str) -> Path:
    """The program in the solution directory ``solution``: the directory itself, or the
    ``program/`` of a twin package (``twin.json`` and ``program/``), as the verifier reads it.

    Checked before anything is written: a path that is not there (``oracel`` for ``oracle``),
    a file, or a program without ``run.py`` raise ``SetupError`` with ``fix``. Exported, each
    was a ``solve.sh`` that recreates nothing, and a reward of 0 for a working task. ``given``
    names the input in the message (``--solution``)."""
    solution = Path(solution)
    if not solution.exists():
        near = difflib.get_close_matches(str(solution).lower(), [ORACLE], n=1, cutoff=0.6)
        hint = f" (did you mean {ORACLE!r}?)" if near and given == "--solution" else ""
        raise InputNotFoundError(f"{given} {solution} does not exist{hint}", fix=fix)
    if not solution.is_dir():
        raise SetupError(f"{given} {solution} is not a directory", fix=fix)
    program = program_of(solution)
    if not (program / ENTRY).is_file():
        where = "" if program == solution else " in its program/ (a twin package)"
        raise SetupError(f"{given} {solution} has no {ENTRY}{where}", fix=fix)
    return program


def _solve_sh(program: Path, app: str = APP) -> str:
    """A script that recreates ``program`` in ``<app>/solution`` (files embedded in base64, an
    executable file executable again: ``run.py`` may run a helper) and runs it once, as an
    agent does. Every name is quoted for the shell: a file name may hold quotes, spaces or
    ``$``. ``program`` holds ``run.py`` (:func:`solution_program`); when it fails, the script
    says so and still ends normally: the verifier re-runs the program and scores it, so its
    failure is its result (a reward of 0), not an error of Harbor's oracle agent."""
    solution, output = shlex.quote(f"{app}/solution"), shlex.quote(f"{app}/output")
    lines = ["#!/bin/bash", "set -euo pipefail", f"mkdir -p -- {solution}", f"cd -- {solution}"]
    for path in sorted(
        p for p in program.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    ):
        relative = shlex.quote("./" + path.relative_to(program).as_posix())
        encoded = base64.b64encode(path.read_bytes()).decode()
        lines.append(f'mkdir -p -- "$(dirname -- {relative})"')
        lines.append(f"echo '{encoded}' | base64 -d > {relative}")
        if path.stat().st_mode & 0o111:
            lines.append(f"chmod +x {relative}")  # BSD chmod takes no -- after the mode
    lines.append(f"if ! R2S_OUTPUT_DIR={output} R2S_SEED=0 python run.py; then")
    lines.append('  echo "solve.sh: run.py failed; the verifier scores the program as it is" >&2')
    lines.append("fi")
    return "\n".join(lines) + "\n"


def oracle_program(task: Task, target: Path) -> Path:
    """The task's truth generator as a solution directory (synthetic tasks)."""
    from real2simlab.twin import load_twin

    generator = task.generator_dir
    if generator is None:
        raise UsageError(f"task {task.id!r} has no truth generator for an oracle solution")
    twin = load_twin(generator)
    copy_tree(twin.program_dir, target)
    return target


def export_task(
    task_dir: Path,
    dest: Path,
    *,
    solution: Path | str | None = None,
    self_contained: bool = False,
    timeout_s: float | None = None,
) -> Path:
    """Write the task at ``task_dir`` as a Harbor task directory at ``dest`` (must not exist).

    ``solution`` is a program directory with ``run.py`` (or a twin package whose ``program/``
    has one), or ``"oracle"`` for the task's truth generator; it becomes
    ``solution/solve.sh``. Anything else is refused before anything is written. The task must
    be valid, as ``r2s eval`` requires (``r2s task validate``): an invalid one raises
    ``SetupError`` listing its problems, and nothing is written.
    """
    task = load_task(task_dir)
    convention = task.config.delivery.convention
    if solution == "":  # an unset variable: not the current directory's program
        raise SetupError(
            "the solution is empty: it would be the current directory",
            fix="give 'oracle' (the task's truth generator) or a program directory with run.py, "
            "or no solution",
        )
    if convention is not DeliveryConvention.R2S:
        raise SetupError(
            f"{task.id} uses the {convention.value} delivery convention; Harbor export supports "
            "tasks whose solution is a run.py program (the r2s convention): the verifier runs "
            "the solution inside its own container, and the other conventions need the "
            "benchmark's own runtime (e.g. 4DCodeBench's fixed /workspace paths)",
            fix=f"evaluate it with r2s eval --tasks {task.root} (the Lab runs every convention)",
        )
    if task.config.scoring.protocol is Protocol.NATIVE:
        raise SetupError(
            f"{task.id} is scored by its benchmark's own scorer (the native protocol), which the "
            "Harbor verifier does not run",
            fix=f"evaluate it with r2s eval --tasks {task.root}",
        )
    if solution is not None and solution != ORACLE:
        solution = Path(solution)
        solution_program(
            solution,
            given="--solution",
            fix=f"give {ORACLE!r} (the task's truth generator) or a program directory with {ENTRY}",
        )
    final = Path(dest)
    if final.exists() or final.is_symlink():
        raise OutputExistsError(
            f"destination already exists: {final}", fix="choose a new --out directory"
        )
    inputs = [task.root, *([solution] if isinstance(solution, Path) else [])]
    clash = clashing_input(final, inputs)
    if clash is not None:
        raise SetupError(
            f"the destination {final} lies inside (or holds) the input {clash}: the export "
            "copies the input, so it would copy itself",
            fix="write the Harbor task to a directory outside the task and the solution",
        )
    # Written next to the destination and moved into place when complete: a failed export
    # leaves no partial task behind (and running the command again works; it removes what an
    # export killed outright left). The move claims the destination: another export given
    # the same one that finished first makes it fail, and this export is then refused as one
    # started after it is.
    dest = staging_dir(final)
    try:
        _write_task(task, dest, solution, self_contained, timeout_s)
        try:
            dest.rename(final)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            raise OutputExistsError(
                f"destination already exists: {final} (another export made it just now)",
                fix="choose a new --out directory",
            ) from None
        release(dest)
    except BaseException:
        force_remove(dest)
        raise
    return final


def _checked_copy(task: Task, target: Path) -> Task:
    """``task`` copied to ``target`` (no symbolic link followed) and validated there as
    ``r2s eval`` validates a task. The export is written from this copy, so the agent's
    environment and the verifier hold exactly a task that passed validation (no hidden data in
    ``visible/``, nothing from outside the task), whatever happens to ``task`` meanwhile."""
    changed = copy_exactly(task.root.resolve(), target)
    # A copy refused (a symbolic link in the task) is not there: the task says what is wrong.
    report = validate_task(target if changed is None else task.root)
    if not report.ok:
        problems = [str(issue) for issue in report.errors]
        raise invalid_task_error(task, problems, then="export it again")
    copied = load_task(target) if changed is None else None
    if copied is None or copied.config != task.config:
        raise SetupError(
            f"task {task.id} changed while it was exported ({task.root})"
            + (f": {changed}" if changed else ""),
            fix="export it again once nothing edits the task",
        )
    return copied


def _write_task(
    task: Task,
    dest: Path,
    solution: Path | str | None,
    self_contained: bool,
    timeout_s: float | None,
) -> None:
    tests = dest / "tests"
    # The verifier's copy of the task, validated; everything else is written from it.
    task = _checked_copy(task, tests / TASK_DIR)
    environment = dest / "environment"
    _workspace(task, environment / "workspace")
    if self_contained:
        shutil.copyfile(sdk_path(), environment / "r2s_twin.py")
    (environment / "Dockerfile").write_text(_environment_dockerfile(task, self_contained))
    statement = task_statement(task, APP) + FINISH
    (dest / "instruction.md").write_text(statement, encoding="utf-8")
    (dest / "task.toml").write_text(tomli_w.dumps(_task_toml(task, timeout_s)), encoding="utf-8")
    copy_lab(tests / LAB_DIR)
    (tests / "Dockerfile").write_text(_verifier_dockerfile(task, self_contained), encoding="utf-8")
    test_sh = tests / "test.sh"
    test_sh.write_text(TEST_SH, encoding="utf-8")
    test_sh.chmod(0o755)
    if solution is not None:
        staging = dest / ".solution-staging"
        program = (
            oracle_program(task, staging) if solution == ORACLE else program_of(Path(solution))
        )
        (dest / "solution").mkdir()
        solve = dest / "solution" / "solve.sh"
        # A name that is not UTF-8 is written as its own bytes, so the script recreates it.
        solve.write_text(_solve_sh(program), encoding="utf-8", errors="surrogateescape")
        solve.chmod(0o755)
        shutil.rmtree(staging, ignore_errors=True)
