"""The official 4DCodeBench scorer as a native scorer plugin (``fourdcodebench``).

It lays out the four root trees the scorer container expects (``/cases``, ``/data``, ``/runs``,
``/checkpoints``), writes a one-entry manifest, and runs ``python -m scorer`` (and, with the
model checkpoints, ``python -m scorer.prepare`` first) in the official scorer image
(``4dcb/scorer:dev``, built from the pinned checkout) with a GPU and no network. It returns
the official ``reward.json``.

Requirements, checked by ``problems()``: Docker with GPU support and the scorer image. The
model checkpoints (``scripts/download_checkpoints.sh`` of the official repository; the DINOv3
weights are gated on Hugging Face) go in the directory named by the ``checkpoints`` option or
``R2S_4DCB_CHECKPOINTS``. Without them, only the official metric types that need no model are
computed (``CHECKPOINT_FREE_TYPES``), and the result lists the types left out.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from real2simlab.errors import SetupError
from real2simlab.sandbox.docker import bind_mount, docker_problem
from real2simlab.sandbox.process import (
    PROBE_TIMEOUT_S,
    run_captured,
    run_container,
    storage_problem,
)
from real2simlab.scoring.model import Headline
from real2simlab.scoring.native import SCORER_FAILED, NativeScorer, default_failure
from real2simlab.task.io import Task

SCORER_IMAGE = "4dcb/scorer:dev"
CHECKPOINTS_ENV = "R2S_4DCB_CHECKPOINTS"
CHECKPOINT_DIRS = (
    "dinov3-vitl16-pretrain-lvd1689m",
    "tipsv2-l14",
    "video-depth-anything-large",
    "raft-large",
    "cotracker3",
    "moge-3-vitl",
    "uni3d-l",
)
CHECKPOINT_FREE_TYPES = (
    "dynamic_iou",
    "scene_3d",
    "trajectory",
    "dynamics",
    "occupancy",
    "interpenetration",
    "mesh",
)
"""Official metric types computed from the worlds and annotations alone."""
MODEL_TYPES = ("uni3d", "flow", "track2d", "semantic", "geophys")
"""Official metric types that need model checkpoints."""
RUN = "r2s"


def _docker() -> str:
    from real2simlab.sandbox.docker import docker_binary

    return docker_binary() or "docker"


def build_command(checkout: Path, image: str = SCORER_IMAGE) -> list[str]:
    """The command that builds the official scorer image from the pinned checkout."""
    return [_docker(), "build", "-t", image, "-f", "harness/scorer/Dockerfile", str(checkout)]


def official_status(exit_code: int | None, reward: Any) -> str:
    """The outcome of one official scoring, from the official scorer's exit code and its
    ``reward.json`` (``None``: there is none). Every world the scorer judges gets a
    ``dynamic_iou`` reading in ``reward.json`` (a number, ``error``, ``uncomputable``,
    ``excluded`` or ``not_applicable``) before the scorer exits, with 0 or 1
    (``scorer/score.py``, ``scorer/__main__.py``). With that reading: ``scored`` (exit 0), or
    ``rejected`` (exit 1: its gates refused the world or a metric failed on it, written to
    ``reward.json`` as errors: the delivery's result, unless ``reward.detail.json`` says it did
    not finish measuring ``dynamic_iou``, which ``score`` reads: ``unfinished``). Anything else
    is ``scorer_failed``, not a score: it judged nothing (no ``reward.json``), or it did not
    finish: stopped by an error of its own after its gate wrote ``reward.json`` (a worker it
    could not start, a full disk: exit 1 without the reading), or with an exit code it never
    returns itself (killed, as by the out-of-memory killer: 137)."""
    judged = isinstance(reward, Mapping) and "dynamic_iou" in reward
    if judged and exit_code == 0:
        return "scored"
    if judged and exit_code == 1:
        return "rejected"
    return SCORER_FAILED


NOTHING = ("no_world", "no_program")
"""Deliveries with nothing to grade; they score 0 (they are not left out of the score)."""
JUDGED = ("scored", "rejected")
"""Outcomes in which the official scorer judged the world (``official_status``)."""
ERROR = "error"
"""The official reading of a world its gates refused or a metric failed on; the official
aggregates count it as the metric's worst value (0 for ``dynamic_iou``)."""
DETAIL_FILE = "reward.detail.json"
"""Written by the official scorer with ``reward.json``, in the same step: why each reading that
is not a number is not (``causes``), and what each failed gate, preparation step or metric
reported (``<name>_failure``)."""
METRIC_FAILURE = "metric_failure"
"""The official cause of an ``error`` whose gate or metric call failed, rather than judged the
world (``scorer/outcome.py``): a defect of the world, or a measurement that did not finish."""
MEASUREMENT_FAILURES = re.compile(
    r"timeout after \S+s|worker exited without a result|worker is no longer running"
    r"|interrupted during \w+|model load failed: .*",
    re.DOTALL,
)
"""The whole text of a failure of the official scorer's own worker framework
(``scorer/execution.py``, ``scorer/score.py``): a call that timed out, a worker process lost
or interrupted, a model that did not load. An error raised in a worker reads
``<type>: <message>`` instead, so the world cannot write these."""
MACHINE_FAILURES = re.compile(
    r"OSError: \[Errno (?:5|12|23|24|28|122)\] "  # EIO, ENOMEM, ENFILE, EMFILE, ENOSPC, EDQUOT
    r"|BlockingIOError: \[Errno 11\] "  # EAGAIN: no more processes
    r"|RuntimeError: can't start new thread"
    r"|(?:RuntimeError|AcceleratorError|OutOfMemoryError): (?:CUDA error|CUDA out of memory"
    r"|CUDA driver error|Cuda error|cudaError|CUBLAS_STATUS_|CUDNN_STATUS_|cuDNN error)"
    r"|RuntimeError: rasterisation needs a CUDA device$"  # no GPU in the container
    r"|FileNotFoundError: \[Errno 2\] No such file or directory: '(?:ffmpeg|ffprobe|scorer-vdb)'$"
)
"""The start of an error a worker of the official scorer raised because of this machine or the
scorer image, not the world (``<type>: <message>``, ``scorer/execution.py``): the disk or the
quota full, the kernel's memory (ENOMEM), processes or open files exhausted, an I/O error, the
GPU failed or missing, a tool missing from the image. Matched at the start, where the
exception's own type and text are, never in a path or a name of the world further on.

Not ``MemoryError``, with or without NumPy's ``Unable to allocate <size>``: one allocation fails
so only when it asks for more than the machine can give at all, which the world decides (an
archive of a few hundred bytes whose header declares a 384 TiB array fails so in the official
world gate on any machine; a compression bomb). A machine short of memory kills the worker
instead (the out-of-memory killer: ``worker exited without a result``, or exit 137), which is
a scorer failure; a ``MemoryError`` is the world's result."""
CASE_VIDEO_FAILURE = re.compile(
    r"CalledProcessError: Command '\['ffprobe', (?:'[^'\\]*', )+'/cases/[^'/\\]+/[^'/\\]+/"
    r"reference\.mp4'\]' (?:returned non-zero exit status -?\d+|died with <Signals\.\w+: \d+>)\."
)
"""The scorer image could not read the case video (the task's own recording, ``/cases``) in the
video gate, after it read the delivered one (``scorer/gate.py``, ``scorer/world.py``)."""


def official_detail(path: Path) -> dict[str, Any]:
    """What the official ``reward.detail.json`` at ``path`` says about the readings that are not
    numbers: ``causes`` (the cause of each ``error``, ``uncomputable`` and ``excluded``
    reading) and ``failures`` (what each failed gate, preparation step or metric reported, the
    ``<name>_failure`` texts). Both ``None`` when the file is missing, unreadable or not an
    object."""
    try:
        detail = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # missing, unreadable, not JSON (UnicodeDecodeError too)
        detail = None
    if not isinstance(detail, Mapping):
        return {"causes": None, "failures": None}
    found: Mapping[str, Any] = detail
    causes = found.get("causes")
    return {
        "causes": dict(causes) if isinstance(causes, Mapping) else None,
        "failures": {
            str(key): str(value)[:2000]
            for key, value in found.items()
            if str(key).endswith("_failure") and value is not None
        },
    }


def _headline_failure(failures: Mapping[str, Any]) -> tuple[str, str] | None:
    """Where the failure that made ``dynamic_iou`` an error was reported, and its text: its own
    metric, or the preparation step it re-raised (``dynamic_iou_failure``); its worker, which
    did not start (``geometry_failure``); or a gate call (``gate_failure``, after the cause)."""
    for key in ("dynamic_iou_failure", "geometry_failure"):
        text = failures.get(key)
        if isinstance(text, str):
            return key, text
    text = failures.get("gate_failure")
    prefix = f"{METRIC_FAILURE}: "
    if isinstance(text, str) and text.startswith(prefix):
        return "gate_failure", text.removeprefix(prefix)
    return None


def unfinished(report: Mapping[str, Any]) -> str | None:
    """Why the official scorer did not finish measuring the headline of a world it rejected with
    ``dynamic_iou`` an ``error``, or ``None`` when the error is its judgement of the world.

    It did not finish (a machine failure: not a score, and never a 0) when it left no
    readable cause for the error (``reward.detail.json`` missing, unreadable, or older than
    the reading: it writes both files in one step, so it stopped in between), or when the cause
    is a failed call (``metric_failure``) whose text is its worker framework's
    (``MEASUREMENT_FAILURES``), this machine's or the scorer image's (``MACHINE_FAILURES``), the
    case video's (``CASE_VIDEO_FAILURE``), or that of the step loading the task's reference
    world. Anything else is the world's result, ``rejected``: a closed gate, an empty first
    frame, a non-finite reading, and a failed call this does not recognize. Only reports with
    the official causes recorded are read: one an earlier Lab kept without them is not judged
    again."""
    if report.get("status") != "rejected" or "causes" not in report:
        return None
    reward = report.get("reward")
    if not isinstance(reward, Mapping) or reward.get("dynamic_iou") != ERROR:
        return None  # dynamic_iou measured (a number), or left out by the official protocol
    causes, failures = report.get("causes"), report.get("failures")
    cause = causes.get("dynamic_iou") if isinstance(causes, Mapping) else None
    if cause is None or not isinstance(failures, Mapping):
        return (
            f"it stopped without recording in {DETAIL_FILE} why dynamic_iou is an error, which "
            "it writes with the reading"
        )
    if cause != METRIC_FAILURE:
        return None  # its gates' or a metric's judgement of the world
    found = _headline_failure(failures)
    if found is None:
        return None
    key, text = found
    if key == "dynamic_iou_failure" and failures.get("reference_failure") == text:
        return f"it could not load the task's reference world (reference_failure: {text})"
    if (
        MEASUREMENT_FAILURES.fullmatch(text)
        or MACHINE_FAILURES.match(text)
        or (key == "gate_failure" and CASE_VIDEO_FAILURE.fullmatch(text))
    ):
        return f"it did not finish measuring dynamic_iou ({key}: {text})"
    return None


def _failed_call(report: Mapping[str, Any]) -> bool:
    """Whether ``dynamic_iou`` is an error of a rejected world because a call failed
    (``metric_failure``), rather than a judgement of the world."""
    causes = report.get("causes")
    return (
        report.get("status") == "rejected"
        and isinstance(causes, Mapping)
        and causes.get("dynamic_iou") == METRIC_FAILURE
    )


class FourDCodeBenchScorer(NativeScorer):
    name: ClassVar[str] = "fourdcodebench"
    task_options: ClassVar[tuple[str, ...] | None] = ("kind", "case")
    required_task_options: ClassVar[tuple[str, ...]] = ("kind", "case")
    summary: ClassVar[str] = "the official 4DCodeBench scorer (GPU, scorer image, checkpoints)"

    def failure(self, report: Mapping[str, Any]) -> str | None:
        """The scorer's own failure (``scorer_failed``), and a report kept as a score or a
        rejection whose exit code and ``reward.json`` say the scorer judged nothing
        (``official_status``), or whose recorded causes say it did not finish measuring
        ``dynamic_iou`` (``unfinished``), as a Lab that attributed it otherwise kept it: scored
        again, not reused."""
        failed = super().failure(report)
        if failed is not None or report.get("status") not in JUDGED:
            return failed
        if (
            "exit_code" in report
            and "reward" in report
            and official_status(report["exit_code"], report["reward"]) == SCORER_FAILED
        ):
            return default_failure(self.name, {**report, "status": SCORER_FAILED})
        why = unfinished(report)
        if why is None:
            return None
        return default_failure(self.name, {**report, "status": SCORER_FAILED, "detail": why})

    def headline(self, report: Mapping[str, Any]) -> Headline:
        """``dynamic_iou`` from the official ``reward.json``. A world the scorer judged an
        ``error`` (its gates refused it, or the metric failed on it, and not because the
        measurement did not finish: ``unfinished``) counts 0, the metric's
        worst value, as in the official aggregates (``docs/spec.md``), as does a delivery with
        nothing to grade: delivering a world the scorer refuses does not raise a mean. Readings
        the official protocol leaves out (``uncomputable``, ``excluded``, ``not_applicable``)
        and the scorer's own failures have none."""
        reward = report.get("reward")
        reading = reward.get("dynamic_iou") if isinstance(reward, Mapping) else None
        value = None
        if isinstance(reading, int | float):
            value = float(reading)
        elif report.get("status") in NOTHING:
            value = 0.0  # nothing to grade: no world, as a world the grader rejects
        elif reading == ERROR and report.get("status") in JUDGED and self.failure(report) is None:
            value = 0.0
        return Headline(
            metric="dynamic_iou", value=value, range=(0.0, 1.0), source="official", scorer=self.name
        )

    def __init__(self, checkpoints: str | None = None, image: str = SCORER_IMAGE) -> None:
        location = checkpoints or os.environ.get(CHECKPOINTS_ENV)
        self.checkpoints = Path(location).expanduser() if location else None
        self.image = image

    def cache_identity(self) -> str:
        """The metric types (which checkpoints are present) and the scorer image's content."""
        try:
            found = run_captured(
                [_docker(), "image", "inspect", "--format", "{{.Id}}", self.image],
                timeout_s=PROBE_TIMEOUT_S,
            )
            image = found.stdout.strip() if found.returncode == 0 else "missing"
        except (OSError, subprocess.TimeoutExpired):
            image = "unknown"
        return f"{','.join(self.types())}|{image}"

    def complete_checkpoints(self) -> bool:
        """Whether every official checkpoint is present, so every metric type can run."""
        return self.checkpoints is not None and all(
            (self.checkpoints / d).is_dir() for d in CHECKPOINT_DIRS
        )

    def types(self) -> tuple[str, ...]:
        if self.complete_checkpoints():
            return (*CHECKPOINT_FREE_TYPES, *MODEL_TYPES)
        return CHECKPOINT_FREE_TYPES

    def problems(self) -> list[str]:
        try:
            result = run_captured(
                [_docker(), "image", "inspect", self.image], timeout_s=PROBE_TIMEOUT_S
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return [f"the official scorer runs in Docker, which is not available ({error})"]
        if result.returncode != 0:
            docker = docker_problem(_docker())
            if docker is not None:  # Docker did not answer: whether the image is built is unknown
                return [
                    "the official scorer runs in Docker whatever the backend, and Docker is not "
                    f"ready here: {docker}"
                ]
            return [
                f"the scorer image {self.image} is not built: in the 4DCodeBench checkout, "
                f"docker build -t {self.image} -f harness/scorer/Dockerfile ."
            ]
        return []

    def layout(
        self, task: Task, world: Path, solution: Path, roots: Path, kind: str, case: str
    ) -> tuple[Path, list[str]]:
        """The root trees and manifests for one delivery; returns the roots and the mounts.

        The official world gate requires the program (``solution/build.sh``) next to
        ``world/``, so the delivered program is placed there too.
        """
        shutil.rmtree(roots, ignore_errors=True)
        video = roots / "cases" / kind / case / "reference.mp4"
        video.parent.mkdir(parents=True)
        capture = task.capture()
        shutil.copyfile(task.visible_dir / next(iter(capture.views.values())).rgb.path, video)
        data = roots / "data" / kind / case
        data.mkdir(parents=True)
        if kind == "synthetic":
            shutil.copytree(task.reference_dir(), data / "world")
        else:
            (data / "annotation").mkdir()
            shutil.copyfile(
                task.hidden_dir / "annotation" / "dynamic_mask.npz",
                data / "annotation" / "dynamic_mask.npz",
            )
        shutil.copytree(world, roots / "runs" / RUN / "workspace" / "world")
        shutil.copytree(solution, roots / "runs" / RUN / "workspace" / "solution")
        entry = {
            "case": f"{kind}/{case}",
            "world": f"runs/{RUN}/workspace/world",
            "out": f"runs/{RUN}",
        }
        (roots / "manifest.json").write_text(json.dumps([entry]))
        (roots / "prepare.json").write_text(json.dumps([{"case": f"{kind}/{case}"}]))
        checkpoints = self.checkpoints if self.complete_checkpoints() else None
        if checkpoints is None:
            checkpoints = roots / "no-checkpoints"
            checkpoints.mkdir()
        mounts: list[str] = []
        for name in ("cases", "data", "runs"):
            mounts += ["--mount", bind_mount(roots / name, f"/{name}")]
        mounts += ["--mount", bind_mount(checkpoints, "/checkpoints", read_only=True)]
        for name in ("manifest.json", "prepare.json"):
            mounts += ["--mount", bind_mount(roots / name, f"/{name}", read_only=True)]
        return roots, mounts

    def score(
        self,
        task: Task,
        *,
        delivery_dir: Path,
        output_dir: Path | None,
        work_dir: Path,
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        problems = self.problems()
        if problems:
            raise SetupError(
                "the official 4DCodeBench scorer cannot run: " + "; ".join(problems),
                fix="do what the message says (the scorer runs in Docker whatever the backend), "
                "then run the same command again",
            )
        kind, case = str(options["kind"]), str(options["case"])
        types = self.types()
        if output_dir is None or not (output_dir / "camera.json").is_file():
            return {"status": "no_world", "types": list(types), "reward": None}
        if not (delivery_dir / "build.sh").is_file():
            return {"status": "no_program", "types": list(types), "reward": None}
        roots, mounts = self.layout(task, output_dir, delivery_dir, work_dir / "roots", kind, case)
        score = f"python -m scorer --manifest /manifest.json --type {','.join(types)}"
        if self.complete_checkpoints():
            script = f"python -m scorer.prepare --manifest /prepare.json && {score}"
        else:
            script = score
        run_args = [
            "--rm", "--gpus", "all", "--network", "none",
            "--user", f"{os.getuid()}:{os.getgid()}", "--entrypoint", "bash",
            *mounts, self.image, "-c", script,
        ]  # fmt: skip
        log = work_dir / "official_scorer.log"
        log.unlink(missing_ok=True)
        # The container is stopped with the experiment (Ctrl-C), not left running.
        exit_code, _, _ = run_container(_docker(), run_args, log)
        results = roots / "runs" / RUN / "results"
        reward_file = results / "reward.json"
        reward = json.loads(reward_file.read_text()) if reward_file.is_file() else None
        report: dict[str, Any] = {
            "status": official_status(exit_code, reward),
            "exit_code": exit_code,
            "types": list(types),
            "types_left_out": [t for t in MODEL_TYPES if t not in types],
            "reward": reward,
        }
        if report["status"] not in JUDGED:
            return report
        report.update(official_detail(results / DETAIL_FILE))
        why = unfinished(report)
        if why is None and _failed_call(report):
            # a failed call this does not recognize, while the disk holding the scorer's output
            # (``/runs``, beside the log) is full: that is why, more likely than the world
            why = storage_problem(log, [])
        if why is not None:
            report.update(status=SCORER_FAILED, detail=why)
        return report
