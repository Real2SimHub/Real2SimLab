"""Physics-IQ: case naming, task building (offline, from a fake cache), the score formula,
input checks, and agreement with the official pipeline on real data (when it is cached)."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from real2simlab.benchmarks.physicsiq import (
    _TESTING,  # pyright: ignore[reportPrivateUsage]
    CODE,
    TAKE_2_OFFSET,
    TEST_FRAMES,
    Case,
    PhysicsIQBenchmark,
    data_dir,
)
from real2simlab.benchmarks.physicsiq.scorer import PhysicsIQScorer, physics_iq_score
from real2simlab.media import write_video
from real2simlab.task import load_task, validate_task

CACHE = Path(os.environ.get("R2S_CACHE_DIR", Path.home() / ".cache" / "real2simlab"))
SCENARIO = "ball-and-block-fall"


def test_case_paths_follow_the_official_layout() -> None:
    case = Case("ball-and-block-fall", "center", 7)
    assert case.id2 == 7 + TAKE_2_OFFSET and case.name == "ball-and-block-fall/center"
    testing = case.file("testing", 1)
    assert testing == (
        "split-videos/testing/30FPS/0007_testing-videos_30FPS_perspective-center_take-1_"
        "trimmed-ball-and-block-fall.mp4"
    )
    match = _TESTING.fullmatch(testing)
    assert match and match.groups() == ("0007", "center", "ball-and-block-fall")
    assert case.file("mask", 2).startswith("video-masks/real/30FPS/0205_video-masks_30FPS_")
    assert case.file("conditioning", 1).startswith("split-videos/conditioning/30FPS/0007_")


def _fake_release(root: Path, case: Case, frames: int = TEST_FRAMES) -> None:
    """A cached 'original' release of one case: small videos and pins, no network."""
    data = data_dir("original")
    data.mkdir(parents=True, exist_ok=True)
    (data / "index.json").write_text(json.dumps([[case.id1, case.perspective, case.scenario]]))
    rng = np.random.default_rng(0)
    clips = {
        case.file("conditioning", 1): 90,
        case.file("testing", 1): frames,
        case.file("testing", 2): frames,
        case.file("mask", 1): frames,
        case.file("mask", 2): frames,
    }
    for path, count in clips.items():
        target = data / path
        target.parent.mkdir(parents=True, exist_ok=True)
        write_video(rng.integers(0, 255, (count, 64, 96, 3), dtype=np.uint8), target, fps=30.0)
        target.with_name(target.name + ".pin").write_text("fake pin\n")


@pytest.fixture
def fake_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    _fake_release(tmp_path, Case(SCENARIO, "center", 3))
    out = tmp_path / "tasks"
    built = PhysicsIQBenchmark().build(out, select=[SCENARIO])
    assert [p.relative_to(out).as_posix() for p in built] == [
        f"{SCENARIO}/center/native",
        f"{SCENARIO}/center/rerun",
    ]
    return out


def test_tasks_validate_and_keep_the_future_hidden(fake_tasks: Path) -> None:
    for variant in ("native", "rerun"):
        root = fake_tasks / SCENARIO / "center" / variant
        report = validate_task(root)
        assert not report.errors, report.errors
        task = load_task(root)
        assert task.config.id == f"physicsiq/{SCENARIO}/center/{variant}"
        native = task.config.scoring.native
        assert native is not None and native.scorer == "physics_iq"
        assert native.options == {"rerun": variant == "rerun"}
        output = task.config.delivery.output
        assert output is not None and output.num_frames == TEST_FRAMES and output.fps == 30.0
        assert sorted(p.name for p in (task.hidden_dir / "reference").iterdir()) == [
            "case.json", "mask_1.mp4", "mask_2.mp4", "testing_1.mp4", "testing_2.mp4",
        ]  # fmt: skip
        visible = sorted(p.relative_to(task.visible_dir).as_posix()
                         for p in task.visible_dir.rglob("*") if p.is_file())  # fmt: skip
        assert "cam/rgb.mp4" in visible and not any("testing" in v or "mask" in v for v in visible)
        assert "exactly 150 frames" in (root / "instruction.md").read_text()


def test_build_rejects_unknown_cases_and_options(fake_tasks: Path) -> None:
    with pytest.raises(ValueError, match="no Physics-IQ case"):
        PhysicsIQBenchmark().build(fake_tasks / "x", select=["no-such-scenario"])
    with pytest.raises(TypeError, match="unknown options"):
        PhysicsIQBenchmark().build(fake_tasks / "x", select=[SCENARIO], colour="red")
    with pytest.raises(ValueError, match="unknown variants"):
        PhysicsIQBenchmark().build(fake_tasks / "x", select=[SCENARIO], variants=["lab"])


@pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("cv2", "pandas", "tqdm")),
    reason="the official metrics need the [physicsiq] extra: pip install -e '.[physicsiq]'",
)
def test_scorer_checks_the_video_before_scoring(fake_tasks: Path, tmp_path: Path) -> None:
    task = load_task(fake_tasks / SCENARIO / "center" / "native")
    scorer = PhysicsIQScorer()
    options = {"rerun": False}
    empty = tmp_path / "empty"
    empty.mkdir()
    missing = scorer.score(
        task, delivery_dir=empty, output_dir=empty, work_dir=tmp_path / "w1", options=options
    )
    # No video, or an invalid one, is the delivery's fault: the worst value of every metric, so
    # the view counts against the method instead of being left out of the score.
    assert missing["status"] == "scored" and missing["delivery"]["status"] == "no_video"
    assert (missing["spatiotemporal_iou"], missing["mse"]) == (0.0, 1.0)
    assert missing["physics_iq_score"] == 0.0
    short = tmp_path / "short"
    write_video(np.zeros((100, 64, 96, 3), np.uint8), short / "render.mp4", fps=30.0)
    report = scorer.score(
        task, delivery_dir=short, output_dir=short, work_dir=tmp_path / "w2", options=options
    )
    assert report["status"] == "scored" and report["delivery"]["status"] == "invalid_video"
    assert "exactly 150" in report["delivery"]["detail"]
    assert report["physics_iq_score"] == missing["physics_iq_score"]


def _report(
    stiou: float, siou: float, wsiou: float, mse: float, **variance: float
) -> dict[str, Any]:
    return {
        "status": "scored",
        "spatiotemporal_iou": stiou,
        "spatial_iou": siou,
        "weighted_spatial_iou": wsiou,
        "mse": mse,
        "physical_variance_spatiotemporal_iou": variance.get("st", 0.5),
        "physical_variance_spatial_iou": variance.get("s", 0.8),
        "physical_variance_weighted_spatial_iou": variance.get("w", 0.4),
        "physical_variance_mse": variance.get("mse", 0.001),
    }


def test_physics_iq_score_formula() -> None:
    # ((0.25/0.5 + 0.4/0.8 + 0.2/0.4) / 3 - (0.011 - 0.001)) * 100 = 49
    assert physics_iq_score([_report(0.25, 0.4, 0.2, 0.011)]) == 49.0
    # Means over views first, as the official calculate_iq_score.py.
    views = [_report(0.5, 0.8, 0.4, 0.001), _report(0.0, 0.0, 0.0, 0.021)]
    assert physics_iq_score(views) == 49.0
    # Clipped to [0, 100]; nothing scored or a zero variance gives no score.
    assert physics_iq_score([_report(1.0, 1.0, 1.0, 0.0)]) == 100.0
    assert physics_iq_score([_report(0.0, 0.0, 0.0, 0.9)]) == 0.0
    assert physics_iq_score([{"status": "no_video"}]) is None
    assert physics_iq_score([_report(0.2, 0.2, 0.2, 0.0, st=0.0)]) is None
    # Physical variances are rounded as the official code rounds them (4 decimals for stIoU).
    assert physics_iq_score([_report(0.5, 0.8, 0.4, 0.001, st=0.50004)]) == 100.0


# Agreement with the official pipeline, on real data already in the cache.

VIEWS = ("left", "center", "right")


def _cached_views() -> bool:
    index = data_dir("original") / "index.json"
    if not index.is_file() or not (CODE.path() / ".r2s-complete").is_file():
        return False
    rows = {(p, s): n for n, p, s in json.loads(index.read_text())}
    for view in VIEWS:
        number = rows.get((view, SCENARIO))
        if number is None:
            return False
        case = Case(SCENARIO, view, number)
        for kind, take in (("conditioning", 1), ("testing", 1), ("testing", 2), ("mask", 1),
                           ("mask", 2)):  # fmt: skip
            path = data_dir("original") / case.file(kind, take)
            if not (path.is_file() and path.with_name(path.name + ".pin").is_file()):
                return False
    return all(importlib.util.find_spec(m) for m in ("cv2", "pandas", "tqdm"))


@pytest.mark.slow
@pytest.mark.skipif(
    not _cached_views(),
    reason=f"Physics-IQ {SCENARIO} (3 views), the pinned code and the [physicsiq] extra are not "
    f"in the cache: r2s bench build physicsiq --select {SCENARIO}/left ... (needs network)",
)
def test_scorer_matches_the_official_pipeline(tmp_path: Path) -> None:
    """The Lab's per-view metrics and score next to Physics-IQ's own end-to-end pipeline, on
    the same predictions (the second take and a static video), then one full report."""
    from real2simlab.benchmarks.physicsiq import case_files, case_index
    from real2simlab.benchmarks.physicsiq.official import official_score
    from real2simlab.benchmarks.physicsiq.scorer import (
        official_view_metrics,
        static_prediction,
        view_summary,
    )

    index = case_index("original")
    roots = PhysicsIQBenchmark().build(
        tmp_path / "tasks", select=[f"{SCENARIO}/{v}" for v in VIEWS], variants=["native"]
    )
    tasks = [load_task(root) for root in roots]
    for kind in ("take_2", "static"):
        reports: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        generated: dict[Case, Path] = {}
        cases: dict[Case, dict[str, Path]] = {}
        for view, task in zip(VIEWS, tasks, strict=True):
            case = index[f"{SCENARIO}/{view}"]
            cases[case] = case_files(case)[0]
            video = tmp_path / kind / view / "render.mp4"
            if kind == "take_2":
                video.parent.mkdir(parents=True)
                video.symlink_to(cases[case]["testing_2"])
            else:
                static_prediction(task, video)
            generated[case] = video
            values = official_view_metrics(task, video, tmp_path / kind / view / "work")
            assert isinstance(values, dict), values
            metrics.append(values)
            reports.append({"status": "scored", **view_summary(values)})
        official = official_score(cases, generated, tmp_path / kind / "official")
        assert physics_iq_score(reports) == pytest.approx(official["score"], abs=0.01)
        row = official["rows"][0]
        for view, values in zip(VIEWS, metrics, strict=True):
            for key in ("spatiotemporal_iou_v1", "v1_mse", "variance_mse"):
                theirs = ast.literal_eval(row[f"{key}_perspective-{view}"])
                assert values[key] == pytest.approx(theirs, abs=1e-9), (kind, view, key)
            for key in ("spatial_iou_v1", "weighted_spatial_iou_v1", "variance_spatial"):
                theirs = float(row[f"{key}_perspective-{view}"])
                assert values[key] == pytest.approx(theirs, abs=1e-9), (kind, view, key)
    # The full report: the method, the static reference and the upper bound (take 2).
    center = tasks[VIEWS.index("center")]
    output = tmp_path / "full" / "output"
    output.mkdir(parents=True)
    (output / "render.mp4").symlink_to(tmp_path / "take_2" / "center" / "render.mp4")
    report = PhysicsIQScorer().score(
        center, delivery_dir=output, output_dir=output, work_dir=tmp_path / "full" / "w", options={}
    )
    assert report["status"] == "scored"
    assert report["take_2"]["physics_iq_score"] == report["physics_iq_score"]
    assert report["static"]["physics_iq_score"] < report["physics_iq_score"]
