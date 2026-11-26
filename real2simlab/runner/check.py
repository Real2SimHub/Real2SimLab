"""``r2s twin check``: run a twin the way an evaluation will, before handing it in.

Loads the package (structure, program rules) and runs it frozen twice: the first output must be
valid in the task's format, and the second must be identical (the same seed gives the same
output). Needs no reference data.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from real2simlab.errors import SetupError
from real2simlab.formats.base import Requirements, get_format
from real2simlab.runner.run import RunResult, run_twin
from real2simlab.sandbox.base import Backend
from real2simlab.scoring.reproduction import outputs_agree
from real2simlab.task.io import Task
from real2simlab.twin import load_twin


@dataclass(frozen=True)
class CheckedRun:
    name: str
    status: str
    seconds: float
    verdict: str
    ok: bool


def _changed(fmt: Any, first: RunResult, second: RunResult) -> bool | None:
    if not (first.succeeded and second.succeeded):
        return None
    return not outputs_agree(fmt, first.output_dir, second.output_dir)


def check_twin(
    twin_dir: Path,
    out_dir: Path,
    *,
    backend: Backend,
    task: Task | None = None,
    seed: int = 0,
    timeout_s: float = 300,
) -> list[CheckedRun]:
    """Run the checks; every run is kept under ``out_dir``."""
    from real2simlab.runner.run import limits_from_resources

    rules = task.config.delivery.program if task else None
    out_dir = Path(out_dir)
    twin_dir = Path(twin_dir)
    if not (twin_dir / "twin.json").is_file():
        twin_dir = freeze_program(twin_dir, out_dir / "twin", task)
    twin = load_twin(twin_dir, rules=rules)
    if task is not None:
        from real2simlab.twin.conventions import convention_spec

        if not convention_spec(task.config.delivery.convention).runnable:
            raise SetupError(
                f"{task.config.delivery.convention.value} deliveries are data, not programs: "
                "the package was frozen and passed the program rules, but only the benchmark's "
                "own evaluator executes it",
                fix=f"score it with r2s eval --tasks {task.root} --method program:path=...",
            )
    fmt = get_format(twin.manifest.output_format)
    limits = (
        limits_from_resources(task.config.delivery.resources, task.config.delivery.run_timeout_s)
        if task
        else None
    )
    if limits is None:
        from real2simlab.runner.record import RunLimits

        limits = RunLimits(timeout_s=timeout_s)
    requirements = Requirements.from_task(task.config.delivery.output) if task else Requirements()

    def run(name: str) -> RunResult:
        return run_twin(
            twin,
            out_dir / name,
            backend=backend,
            label=name,
            seed=seed,
            limits=limits,
            requirements=requirements,
        )

    first = run("run")
    checks = [
        CheckedRun(
            "run",
            first.record.status.value,
            first.record.duration_s,
            "valid output" if first.succeeded else "; ".join(first.record.problems),
            first.succeeded,
        )
    ]
    second = run("rerun")
    changed = _changed(fmt, first, second)
    checks.append(
        CheckedRun(
            "rerun",
            second.record.status.value,
            second.record.duration_s,
            {
                None: "not compared",
                False: "identical: deterministic",
                True: "DIFFERENT: not deterministic",
            }[changed],
            changed is False,
        )
    )
    (out_dir / "check.json").write_text(
        json.dumps([c.__dict__ for c in checks], indent=2, default=str) + "\n"
    )
    return checks


def freeze_program(program: Path, dest: Path, task: Task | None) -> Path:
    """A twin package at ``dest`` from a program directory.

    The environment, output format and delivery convention (entry point, runtime paths) come
    from ``task``; without one: ``r2s-python``, ``tracks`` and ``python ./run.py``.
    """
    from datetime import UTC, datetime

    from real2simlab.task.model import DeliveryConvention
    from real2simlab.twin import EnvironmentRef, Origin, TwinManifest, save_twin
    from real2simlab.twin.conventions import convention_spec

    delivery = task.config.delivery if task else None
    convention = convention_spec(delivery.convention if delivery else DeliveryConvention.R2S)
    manifest = TwinManifest(
        twin_id=f"check/{_id_part(program.resolve().name)}",
        entrypoint=convention.entrypoint,
        environment=EnvironmentRef(image=delivery.environment if delivery else "r2s-python"),
        output_format=delivery.output_format if delivery else "tracks",
        runtime=convention.runtime,
        origin=Origin(
            method="program",
            task_id=task.id if task else "unknown",
            created_at=datetime.now(UTC),
        ),
    )
    save_twin(manifest, program, dest, rules=delivery.program if delivery else None)
    return dest


def _id_part(name: str) -> str:
    """``name`` (a directory's, any characters) as a part of a twin id: what a twin id may not
    hold becomes ``-``; ``program`` when nothing is left."""
    import re

    part = re.sub(r"[^A-Za-z0-9_.:@+-]+", "-", name)
    return re.sub(r"^[^A-Za-z0-9]+", "", part).rstrip("-") or "program"


def format_checks(checks: Sequence[CheckedRun]) -> str:
    rows = ["| run | status | seconds | verdict |", "| --- | --- | --- | --- |"]
    for check in checks:
        mark = "" if check.ok else " ✗"
        rows.append(
            f"| {check.name} | {check.status} | {check.seconds:.1f} | {check.verdict}{mark} |"
        )
    return "\n".join(rows)
