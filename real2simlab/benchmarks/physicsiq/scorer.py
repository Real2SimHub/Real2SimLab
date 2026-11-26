"""Physics-IQ's official metrics as a native scorer plugin (``physics_iq``).

Runs the pinned Physics-IQ code in a subprocess: its motion-mask generator on the predicted
video (named as the official pipeline names generated videos), then ``load_view`` and
``compute_view_metrics`` against take 1 and take 2 of the real continuation — exactly the
per-view computation of ``physiq/calculate_and_write_metrics_to_csv.py``.

The report holds the official per-view metrics, their means, the physical variance (the same
metrics between the two real takes, Physics-IQ's noise floor) and :func:`physics_iq_score` of
this one view, next to the same numbers for two references: ``static`` (the last observed frame
held for 5 seconds) and ``take_2`` (the second real take as the prediction: the upper bound).
The official benchmark score aggregates all views first; give :func:`physics_iq_score` the
reports of several tasks to compute it the same way. Needs OpenCV, pandas and tqdm
(the ``physicsiq`` extra).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from real2simlab._project import install_command
from real2simlab.benchmarks.physicsiq import CODE, FPS, TEST_FRAMES
from real2simlab.errors import SetupError
from real2simlab.media import MediaError, probe_video, tool
from real2simlab.sandbox.process import PROBE_TIMEOUT_S, run_captured
from real2simlab.scoring.model import Headline
from real2simlab.scoring.native import SCORER_FAILED, NativeScorer
from real2simlab.task.io import Task

SCRIPT = """\
import json, shutil, sys
from pathlib import Path
from physiq.binary_mask_generator import generate_mask
from physiq.calculate_and_write_metrics_to_csv import (
    ViewPaths, compute_view_metrics, load_view,
)
generated, testing_1, testing_2, mask_1, mask_2 = sys.argv[1:6]
work, number, perspective, scenario = sys.argv[6:10]
work = Path(work)
stem = f"{int(number):04d}_perspective-{perspective}_trimmed-{scenario}.mp4"
video = work / "generated" / stem
video.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(generated, video)
masks = work / "generated-masks"
masks.mkdir(exist_ok=True)
generate_mask(str(video), str(masks / stem), is_real=False)
mask = masks / (
    f"{int(number):04d}_video-masks_30FPS_perspective-{perspective}_take-1_trimmed-{scenario}.mp4"
)
consider = int(sys.argv[10])
# As run_physics_iq.py: generated videos are exactly 5 s and scored from their first frame.
paths = ViewPaths(testing_1, testing_2, str(video), mask_1, mask_2, str(mask))
frames = load_view(paths, 0, consider, consider)
metrics = compute_view_metrics(frames)
print(json.dumps({k: (v if isinstance(v, list) else float(v)) for k, v in metrics.items()}))
"""

MEANS = (
    ("spatiotemporal_iou", "spatiotemporal_iou_v1", "variance_spatiotemporal_iou"),
    ("spatial_iou", "spatial_iou_v1", "variance_spatial"),
    ("weighted_spatial_iou", "weighted_spatial_iou_v1", "variance_weighted_spatial"),
    ("mse", "v1_mse", "variance_mse"),
)
"""(summary name, official metric against take 1, official physical-variance metric)."""


def _mean(value: Any) -> float:
    values = value if isinstance(value, list) else [value]
    return float(sum(values) / len(values)) if values else float("nan")


def official_view_metrics(task: Task, video: Path, work_dir: Path) -> dict[str, Any] | str:
    """Physics-IQ's per-view metrics of ``video`` against the task's two takes, computed by the
    official code (a mask generated for the video exactly as the official pipeline does);
    a problem as a string when the official code fails on it."""
    reference = task.hidden_dir / "reference"
    case = json.loads((reference / "case.json").read_text(encoding="utf-8"))
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(CODE.ensure())}
    command = [
        sys.executable, "-c", SCRIPT, str(video.resolve()),
        *(str((reference / f"{name}.mp4").resolve())
          for name in ("testing_1", "testing_2", "mask_1", "mask_2")),
        str(work_dir), str(case["take_1"]), case["perspective"], case["scenario"],
        str(TEST_FRAMES),
    ]  # fmt: skip
    result = run_captured(command, env=env, cwd=work_dir)  # stopped with the experiment
    if result.returncode != 0:
        error = result.stderr.strip().splitlines()[-1:] or ["unknown error"]
        if "ModuleNotFoundError" in result.stderr:
            raise SetupError(
                f"Physics-IQ's metrics cannot run: {error[0]}",
                fix=install_command("physicsiq"),
            )
        return error[0]
    return json.loads(result.stdout.strip().splitlines()[-1])


def _opencv_problem(video: Path) -> str | None:
    """Why OpenCV, the reader Physics-IQ's code uses, cannot read every frame of ``video``."""
    import cv2  # pyright: ignore[reportMissingImports]  # an extra: physicsiq, physcodegym

    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            return "OpenCV (which Physics-IQ's code reads videos with) cannot open the video"
        frames = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            frames += 1
    finally:
        capture.release()
    if frames != TEST_FRAMES:
        return (
            f"OpenCV (which Physics-IQ's code reads videos with) reads {frames} of its "
            f"{TEST_FRAMES} frames; encode it as H.264 (e.g. ffmpeg -c:v libx264 -pix_fmt yuv420p)"
        )
    return None


def video_problem(video: Path | None) -> dict[str, str] | None:
    """Why ``video`` is not a valid Physics-IQ prediction (``status``, ``detail``), or ``None``."""
    if video is None or not video.is_file():
        return {"status": "no_video", "detail": "the delivery wrote no render.mp4"}
    try:
        info = probe_video(video)
    except MediaError as error:
        return {"status": "invalid_video", "detail": str(error)}
    if abs(info.fps - FPS) > 1e-3 or info.num_frames != TEST_FRAMES:
        return {
            "status": "invalid_video",
            "detail": f"{info.num_frames} frames at {info.fps:g} fps; Physics-IQ requires "
            f"exactly {TEST_FRAMES} at {FPS:g} (5 seconds)",
        }
    unreadable = _opencv_problem(video)
    if unreadable:  # the official code reads videos with OpenCV: the delivery's fault
        return {"status": "invalid_video", "detail": unreadable}
    return None


def view_summary(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Means of the per-frame metrics against take 1, and of the physical variance."""
    summary: dict[str, float] = {}
    for name, against_take_1, variance in MEANS:
        summary[name] = _mean(metrics[against_take_1])
        summary[f"physical_variance_{name}"] = _mean(metrics[variance])
    return summary


def static_prediction(task: Task, target: Path) -> Path:
    """A 5-second video of the last observed frame (the visible clip's last frame, held)."""
    clip = task.visible_dir / "cam" / "rgb.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    still = target.with_name("last.png")
    ffmpeg = tool("ffmpeg")
    # Every frame of the last second is written over the same image, so the one left is the
    # clip's last frame (a seek lands on a frame before it).
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-sseof", "-1", "-i", str(clip), "-update", "1",
         str(still)],
        check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=600
    )  # fmt: skip
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-loop", "1", "-framerate", f"{FPS:g}", "-i", str(still),
         "-frames:v", str(TEST_FRAMES), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
         str(target)],
        check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=600
    )  # fmt: skip
    return target


WORST = {"spatiotemporal_iou": 0.0, "spatial_iou": 0.0, "weighted_spatial_iou": 0.0, "mse": 1.0}
"""The worst value of each metric against take 1 (Physics-IQ's frames are scaled to [0, 1], so
MSE is at most 1): a view with no valid video counts with these."""


def blank_prediction(task: Task, target: Path) -> Path:
    """A 5-second black video at the camera's resolution, run through the official code for a
    view with no valid video, for the view's physical variance."""
    info = probe_video(task.visible_dir / "cam" / "rgb.mp4")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
         f"color=c=black:s={info.width}x{info.height}:r={FPS:g}", "-frames:v", str(TEST_FRAMES),
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(target)],
        check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=600
    )  # fmt: skip
    return target


def physics_iq_score(reports: Sequence[Mapping[str, Any]]) -> float | None:
    """The official Physics-IQ score of the scored views in ``reports`` (0 to 100).

    As ``physiq/calculate_iq_score.py``: means over views of each metric against take 1 and of
    its physical variance, then
    ``((stIoU / var + spatial IoU / var + weighted IoU / var) / 3 - (MSE - var MSE)) * 100``,
    clipped to [0, 100]. ``None`` if no view is scored or a physical variance is 0.
    """
    scored = [r for r in reports if r.get("status") == "scored"]
    if not scored:
        return None

    def mean(name: str) -> float:
        return sum(r[name] for r in scored) / len(scored)

    # The official code rounds the physical variances before using them.
    variance = {
        "spatiotemporal_iou": round(mean("physical_variance_spatiotemporal_iou"), 4),
        "spatial_iou": round(mean("physical_variance_spatial_iou"), 5),
        "weighted_spatial_iou": round(mean("physical_variance_weighted_spatial_iou"), 4),
        "mse": round(mean("physical_variance_mse"), 4),
    }
    if min(variance[k] for k in ("spatiotemporal_iou", "spatial_iou", "weighted_spatial_iou")) <= 0:
        return None
    ratios = [
        mean(name) / variance[name]
        for name in ("spatiotemporal_iou", "spatial_iou", "weighted_spatial_iou")
    ]
    score = (sum(ratios) / 3 - (mean("mse") - variance["mse"])) * 100
    return round(max(min(score, 100.0), 0.0), 2)


def _importable(*modules: str) -> bool:
    code = "import " + ", ".join(modules)
    return (
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=PROBE_TIMEOUT_S,
        ).returncode
        == 0
    )


REFERENCE_LABELS = {"static": "static", "take_2": "another recording"}
"""Report entries of the references, and their names in the scorecard."""


class PhysicsIQScorer(NativeScorer):
    name: ClassVar[str] = "physics_iq"
    task_options: ClassVar[tuple[str, ...] | None] = ()
    summary: ClassVar[str] = "Physics-IQ's motion-mask IoUs and MSE, with the physical variance"

    def problems(self) -> list[str]:
        if not _importable("cv2", "pandas", "tqdm"):
            return [
                "Physics-IQ's metrics need OpenCV, pandas and tqdm: " + install_command("physicsiq")
            ]
        return []

    def headline(self, report: Mapping[str, Any]) -> Headline:
        references = {
            label: float(entry["physics_iq_score"])
            for name, label in REFERENCE_LABELS.items()
            if isinstance(entry := report.get(name), Mapping)
            and entry.get("physics_iq_score") is not None
        }
        value = report.get("physics_iq_score") if report.get("status") == "scored" else None
        return Headline(
            metric="physics_iq_score",
            value=None if value is None else float(value),
            unit="points",
            range=(0.0, 100.0),
            references=references,
            source="official",
            scorer=self.name,
        )

    @classmethod
    def aggregate(cls, reports: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
        """The official score over every view together (Physics-IQ pools the metrics of all
        views before scoring), and the same for the references."""
        scored = [r for r in reports if r.get("status") == "scored"]
        combined: dict[str, float | None] = {"value": physics_iq_score(scored)}
        for name, label in REFERENCE_LABELS.items():
            entries = [
                {
                    **{k: v for k, v in r.items() if k.startswith("physical_variance_")},
                    **entry,
                    "status": "scored",
                }
                for r in scored
                if isinstance(entry := r.get(name), Mapping) and "spatiotemporal_iou" in entry
            ]
            combined[label] = (
                physics_iq_score(entries) if entries and len(entries) == len(scored) else None
            )
        return combined

    def score(
        self,
        task: Task,
        *,
        delivery_dir: Path,
        output_dir: Path | None,
        work_dir: Path,
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        video = None if output_dir is None else output_dir / "render.mp4"
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        problem = video_problem(video)
        if problem is not None:
            # No video, or one the protocol or the official reader refuses: the delivery's
            # fault. The official code still runs (on a black video) for the view's physical
            # variance, which the pooled score needs; the method's metrics are then the worst
            # possible, so the view counts against it instead of being left out.
            video = blank_prediction(task, work_dir / "blank" / "render.mp4")
        assert video is not None
        metrics = official_view_metrics(task, video, work_dir / "method")
        if isinstance(metrics, str):
            # The video passed the protocol's checks (150 frames at 30 fps, every one read by
            # OpenCV as the official code reads them), so a failure of the official code is
            # the scorer's, not a score of 0 for the delivery.
            return {"status": SCORER_FAILED, "detail": metrics}
        (work_dir / "physics_iq_metrics.json").write_text(json.dumps(metrics) + "\n")
        report: dict[str, Any] = {"status": "scored", **view_summary(metrics)}
        if problem is not None:
            report.update(WORST)
            report["delivery"] = {**problem, "scored_as": "the worst value of every metric"}
        report["physics_iq_score"] = physics_iq_score([report])
        report["metrics_file"] = "physics_iq_metrics.json"
        # References scored the same way: a static prediction (the last observed frame, held
        # for 5 s) and the upper bound, the second real take used as the prediction.
        references = {
            "static": static_prediction(task, work_dir / "static" / "render.mp4"),
            "take_2": task.hidden_dir / "reference" / "testing_2.mp4",
        }
        for name, prediction in references.items():
            baseline = official_view_metrics(task, prediction, work_dir / name)
            if isinstance(baseline, str):
                report[name] = {"status": "failed", "detail": baseline}
                continue
            summary = view_summary(baseline)
            entry: dict[str, Any] = {k: v for k, v in summary.items() if "variance" not in k}
            entry["physics_iq_score"] = physics_iq_score([{**summary, "status": "scored"}])
            report[name] = entry
        return report
