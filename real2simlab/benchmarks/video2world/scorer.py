"""Video2World's own evaluator as a native scorer plugin (``video2world``).

Runs, with no network and the host user's uid, ``evaluate_package.py``: Video2World's
per-configuration evaluation (it executes the package in the evaluator's simulator and measures
geometry, dynamics and the task outcome) and its scoring, from the pinned checkout, on the
frozen package, against the task's hidden one-instance data release. The image follows the
instance's evaluator (:data:`~real2simlab.benchmarks.video2world.EVALUATORS`):
``r2s-video2world`` for ManiSkill / SAPIEN (with the GPU: SAPIEN needs a Vulkan device even
though the physics runs on the CPU) and MuJoCo, ``r2s-video2world-isaac`` for Isaac Sim. The
dataset directories the evaluator reads besides the sample are downloaded into the cache and
added to the release (copied, or mounted read-only when they are large). The report holds the
instance's row of the official score (build, F, G, D and S = build * (F + G + D) / 3, each
metric's error, tolerance and quality), ``score`` = 100 * S, the raw metric record, and the
same row for the shipped human-assisted reference, for comparison.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

from real2simlab.benchmarks.video2world import (
    CODE,
    ENVIRONMENT,
    EVALUATORS,
    ISAAC_ENVIRONMENT,
    ROBODOJO_ASSETS_TARGET,
    Evaluator,
    restore_public_files,
    robodojo_assets,
    shared_data,
)
from real2simlab.errors import SetupError
from real2simlab.sandbox.docker import bind_mount, docker_problem
from real2simlab.sandbox.process import PROBE_TIMEOUT_S, run_captured, run_container
from real2simlab.scoring.model import Headline
from real2simlab.scoring.native import NativeScorer
from real2simlab.task.io import Task

DRIVER = Path(__file__).with_name("evaluate_package.py")
LOG = "video2world.log"
"""The evaluator's output, in the scorer's work directory."""
EVALUATOR_ISSUES = (
    "evaluator failure:",
    "evaluator metric error:",
    "missing record",
    "missing metric record",
    "build must be boolean",
)
"""How the issues of Video2World's scoring (``v2w/scoring.py``) begin for a row its evaluator
could not measure: the evaluator crashed, timed out or lost its renderer or GPU, its metric code
failed, it wrote no metric record, or it wrote no build verdict. Video2World never blames the
package for these (``v2w/runner.py``: "an evaluator crash is never attributed to the method", "a
metric failure blocks the score; it is not a method failure"). The build verdict is the
evaluator's own: every evaluator the Lab runs (``v2w/tracks/*/evaluate.py``) writes a boolean,
except the ``fb`` evaluator's scene report (``v2w/tracks/furniture/evaluate.py``) and
Video2World's runner (``v2w/runner.py``) when the evaluator itself failed (``error_kind``
``evaluator``); a package that does not build is ``build_failed``, a score."""


def _docker() -> str:
    from real2simlab.sandbox.docker import docker_binary

    return docker_binary() or "docker"


def _image(environment: str = ENVIRONMENT) -> str:
    from real2simlab.envs import image_reference

    return image_reference(environment)


def _image_id(environment: str) -> str:
    try:
        found = run_captured(
            [_docker(), "image", "inspect", "--format", "{{.Id}}", _image(environment)],
            timeout_s=PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return found.stdout.strip() if found.returncode == 0 else "missing"


def runtime_of(task: Task) -> Evaluator:
    """How the evaluator of ``task`` runs (tasks built before the other evaluators were
    supported are ``fb`` tasks)."""
    meta = task.config.metadata.get("video2world")
    info = cast(dict[str, Any], meta) if isinstance(meta, dict) else {}
    evaluator = str(info.get("evaluator") or "fb")
    if evaluator not in EVALUATORS:
        raise SetupError(
            f"task {task.id} is scored by Video2World's {evaluator!r} evaluator, which this Lab "
            "does not run",
            fix="update Real2SimLab, or rebuild the task: r2s bench build video2world",
        )
    return EVALUATORS[evaluator]


class Video2WorldScorer(NativeScorer):
    name: ClassVar[str] = "video2world"
    task_options: ClassVar[tuple[str, ...] | None] = ()
    summary: ClassVar[str] = (
        "Video2World's evaluator: executes the package, scores V2WScore (build, functionality, "
        "geometry, dynamics)"
    )

    def cache_identity(self) -> str:
        """The content of the images the evaluators run in."""
        isaac = _image_id(ISAAC_ENVIRONMENT)
        main = _image_id(ENVIRONMENT)
        # Without the Isaac image, the identity is the one of the ManiSkill / MuJoCo image alone.
        return main if isaac in ("missing", "unknown") else f"{main}+{isaac}"

    def failure(self, report: Mapping[str, Any]) -> str | None:
        """The scorer's own failure (``scorer_failed``), and a row Video2World did not score for
        a reason that is not the package: its evaluator could not measure the package
        (``EVALUATOR_ISSUES``; an evaluation error, which Video2World's runner writes only when
        its evaluator failed; no metric record at all), or the row has no issue to block it, as
        for a sample Video2World does not score (``not_applicable``). None of these is a score
        of the package. A row blocked on what was measured of the package stays its result."""
        failed = super().failure(report)
        if failed is not None or report.get("status") != "blocked":
            return failed
        official = report.get("official_status")
        if official is None:
            return f"the official scorer {self.name!r} failed (no metric record; log {LOG})"
        issues = report.get("issues")
        reasons = [
            str(cast(Mapping[str, Any], issue).get("reason"))
            for issue in (cast(list[Any], issues) if isinstance(issues, list) else [])
            if isinstance(issue, Mapping)
        ]
        evaluator = [reason for reason in reasons if reason.startswith(EVALUATOR_ISSUES)]
        if evaluator:
            return f"the official scorer {self.name!r} failed ({'; '.join(evaluator)})"
        error = report.get("evaluation_error")
        if error:  # eval.json's error: written by Video2World's runner when its evaluator failed
            return f"the official scorer {self.name!r} failed (evaluator: {error}; log {LOG})"
        if not reasons:
            return (
                f"the official scorer {self.name!r} gave no score and no issue (Video2World's "
                f"row is {official!r}: a sample it does not score), not a result of the package"
            )
        return None

    def headline(self, report: Mapping[str, Any]) -> Headline:
        """The instance's V2WScore, on Video2World's scale: 100 · S, where S = build · (F + G +
        D) / 3 averages functionality, geometry and dynamics."""
        reference = report.get("reference")
        references: dict[str, float] = {}
        if isinstance(reference, Mapping) and isinstance(reference.get("S"), int | float):
            references["reference solution"] = 100.0 * float(reference["S"])
        value = report.get("S") if report.get("status") == "scored" else None
        return Headline(
            metric="V2WScore",
            value=None if value is None else 100.0 * float(value),
            unit="points",
            range=(0.0, 100.0),
            references=references,
            source="official",
            scorer=self.name,
        )

    @classmethod
    def aggregate(cls, reports: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
        """V2WScore over several tasks as Video2World combines instances: the mean within each
        task family, then the mean over families; the same for the reference solution."""
        scored = [
            r
            for r in reports
            if r.get("status") == "scored" and isinstance(r.get("S"), int | float)
        ]
        references = [
            (str(r.get("family") or ""), 100.0 * float(ref["S"]))
            for r in scored
            if isinstance(ref := r.get("reference"), Mapping)
            and isinstance(ref.get("S"), int | float)
        ]
        values = [(str(r.get("family") or ""), 100.0 * float(r["S"])) for r in scored]
        return {
            "value": _family_mean(values),
            "reference solution": (
                _family_mean(references) if len(references) == len(scored) else None
            ),
        }

    def problems(self, environment: str = ENVIRONMENT) -> list[str]:
        try:
            result = run_captured(
                [_docker(), "image", "inspect", _image(environment)], timeout_s=PROBE_TIMEOUT_S
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return [f"the Video2World evaluator runs in Docker, which is not available ({error})"]
        if result.returncode != 0:
            docker = docker_problem(_docker())
            if docker is not None:  # Docker did not answer: whether the image is built is unknown
                return [
                    "the Video2World evaluator runs in Docker whatever the backend, and Docker is "
                    f"not ready here: {docker}"
                ]
            from real2simlab.envs import build_hint, get_environment

            hint = build_hint(get_environment(environment))
            return [f"the image {_image(environment)} is not built: {hint}"]
        return []

    def score(
        self,
        task: Task,
        *,
        delivery_dir: Path,
        output_dir: Path | None,
        work_dir: Path,
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        release = task.hidden_dir / "v2w"
        meta = task.config.metadata.get("video2world")
        info = cast(dict[str, Any], meta) if isinstance(meta, dict) else {}
        sample, family = str(info.get("sample", "")), str(info.get("family", ""))
        if not (delivery_dir / "protocol.json").is_file():
            # Nothing to build: as Video2World scores a package that does not build (build 0,
            # so S = 0), not left out of the family means.
            return {
                "status": "scored",
                "family": family or None,
                "official_status": "no_package",
                "build": 0,
                "S": 0.0,
                "score": 0.0,
            }
        runtime = runtime_of(task)
        problems = self.problems(runtime.image)
        if problems:
            raise SetupError(
                "the Video2World evaluator cannot run: " + "; ".join(problems),
                fix="do what the message says (the evaluator runs in Docker whatever the "
                "backend), then run the same command again",
            )
        if not (release / "manifest.json").is_file() or not sample:
            raise SetupError(
                f"task {task.id} has no Video2World data release in hidden/v2w",
                fix="rebuild it: r2s bench build video2world",
            )
        work_dir = work_dir.resolve()
        shutil.rmtree(work_dir / "v2w", ignore_errors=True)
        (work_dir / "v2w" / "home").mkdir(parents=True)
        shutil.copytree(delivery_dir, work_dir / "v2w" / "pkg", symlinks=False)
        data = work_dir / "v2w" / "data"  # the task's release plus the shared data
        shutil.copytree(release, data)
        restore_public_files(data, task.visible_dir)

        def mount(source: Path, target: str, *, read_only: bool = True) -> list[str]:
            return ["--mount", bind_mount(source, target, read_only=read_only)]

        extra: list[str] = []
        for relative, source in shared_data(runtime).items():
            if runtime.mount_shared:
                # The mount point must exist: /data itself is mounted read-only.
                (data / relative).mkdir(parents=True, exist_ok=True)
                extra += mount(source, f"/data/{relative}")
            else:
                shutil.copytree(source, data / relative, dirs_exist_ok=True)
        if runtime.robodojo_assets:
            extra += mount(robodojo_assets(), ROBODOJO_ASSETS_TARGET)
        if runtime.scorer_gpu:
            extra += ["--gpus", "all"]
        if runtime.image == ISAAC_ENVIRONMENT:
            extra += ["--shm-size", "8g"]  # Isaac Sim's renderer and PhysX share memory

        run_args = [
            "--rm", "--network", "none",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "HOME=/work/home", "-e", "V2W_DATA=/data", "-e", "PYTHONPATH=/v2w",
            # The samples' records name files by their producer's paths under /root; the driver
            # links those paths to the release in this scratch /root (see evaluate_package.py).
            "--tmpfs", "/root:rw,mode=1777",
            *mount(CODE.ensure(), "/v2w"), *mount(data, "/data"),
            *mount(DRIVER, "/r2s/evaluate_package.py"),
            *mount(work_dir / "v2w", "/work", read_only=False),
            *extra,
            _image(runtime.image), runtime.python, "/r2s/evaluate_package.py", sample,
            "/work/pkg", "/work/run", "/work/result.json",
        ]  # fmt: skip
        log = work_dir / LOG
        log.unlink(missing_ok=True)
        # A timeout, or the experiment stopped (Ctrl-C), stops the container itself.
        code, _, _ = run_container(_docker(), run_args, log, runtime.timeout_s)
        output = work_dir / "v2w" / "result.json"
        if code != 0 or not output.is_file():
            return {
                "status": "scorer_failed",
                "exit_code": code,
                "log": LOG,
            }
        official = json.loads(output.read_text(encoding="utf-8"))
        (work_dir / "video2world_record.json").write_text(
            json.dumps(official.get("record"), indent=1) + "\n", encoding="utf-8"
        )
        row = official.get("score") or {}
        reference = official.get("reference_score") or {}
        s_value = row.get("S")
        return {
            "status": "scored" if row.get("status") in ("ok", "build_failed") else "blocked",
            "family": family or None,
            "official_status": row.get("status"),
            "build": row.get("build"),
            "task_success": (official.get("record") or {}).get("task_success"),
            "F": row.get("F"),
            "G": row.get("G"),
            "D": row.get("D"),
            "S": s_value,
            "score": None if s_value is None else 100.0 * s_value,
            "geometry": row.get("geometry"),
            "dynamics": row.get("dynamics"),
            "issues": row.get("issues"),
            "evaluation_error": official.get("evaluation_error"),
            # Video2World's shipped record of its human-assisted reference, scored the same way.
            "reference": {
                **{k: reference.get(k) for k in ("status", "build", "F", "G", "D", "S")},
                "score": None if reference.get("S") is None else 100.0 * reference["S"],
            },
            "record_file": "video2world_record.json",
        }


def _family_mean(values: Sequence[tuple[str, float]]) -> float | None:
    """The mean over families of each family's mean."""
    by_family: defaultdict[str, list[float]] = defaultdict(list)
    for family, value in values:
        by_family[family].append(value)
    if not by_family:
        return None
    means = [sum(v) / len(v) for v in by_family.values()]
    return sum(means) / len(means)
