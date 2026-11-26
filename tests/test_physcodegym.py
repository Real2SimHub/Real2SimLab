"""PhysCodeGym: the blender_scene convention, task building, the pixel scorer, a Blender twin."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from real2simlab.task import DeliveryConvention
from real2simlab.twin.conventions import convention_spec

from .conftest import docker_image_ready

ROOT = Path(__file__).resolve().parents[1]
CASE = "physics_iq_0001_ball_and_block_fall"
CACHE = Path(os.environ.get("R2S_CACHE_DIR", Path.home() / ".cache" / "real2simlab"))
VIDEO = CACHE / "data" / "PhysCodeGym" / "videos" / f"{CASE}.mp4"
needs_video = pytest.mark.skipif(
    not VIDEO.is_file(), reason="PhysCodeGym reference video not downloaded"
)
needs_cv2 = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None, reason="pip install opencv-python-headless"
)


def test_conventions() -> None:
    r2s = convention_spec(DeliveryConvention.R2S)
    assert r2s.entry_file == "run.py" and r2s.runtime.output_path == "/r2s/output"
    build = convention_spec(DeliveryConvention.BUILD_SH_WORLD)
    assert build.entrypoint == ("bash", "./build.sh") and build.session_output == "world"
    blender = convention_spec(DeliveryConvention.BLENDER_SCENE)
    assert blender.entry_file == "scene.py" and blender.runtime.output_path == "/app/outputs"
    assert blender.agent_output_mount == "/app/outputs"


def test_blender_runner_reports_missing_scene(tmp_path: Path) -> None:
    runner = ROOT / "real2simlab" / "envs" / "blender_scene.py"
    result = subprocess.run(
        ["python3", str(runner), str(tmp_path / "scene.py"), str(tmp_path / "out")],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert result.returncode == 2 and "does not exist" in result.stderr


@pytest.fixture(scope="module")
def pcg_tasks(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not VIDEO.is_file():
        pytest.skip("PhysCodeGym reference video not downloaded")
    from real2simlab.benchmarks.physcodegym import build_case

    out = tmp_path_factory.mktemp("pcg")
    for variant in ("native", "rerun"):
        build_case(CASE, variant, out)
    return out


@needs_video
def test_tasks(pcg_tasks: Path) -> None:
    from real2simlab.task import load_task, validate_task
    from real2simlab.task.instruction import task_statement

    for variant in ("native", "rerun"):
        root = pcg_tasks / CASE / variant
        report = validate_task(root)
        assert report.ok and not report.warnings, report.issues
        task = load_task(root)
        assert task.config.delivery.convention is DeliveryConvention.BLENDER_SCENE
        assert task.config.redistributable is False and task.config.license == "CC-BY-4.0"
        assert task.config.scoring.native is not None
        assert task.config.scoring.native.options["rerun"] is (variant == "rerun")
        statement = task_statement(task, "/ws")
        assert "/ws/solution/scene.py" in statement and "{workspace}" not in statement
        assert "frame_0001.png" in statement


@needs_video
@needs_cv2
def test_pixel_scorer(pcg_tasks: Path, tmp_path: Path) -> None:
    from real2simlab.benchmarks.physcodegym.scorer import PhysCodeGymPixels
    from real2simlab.task import load_task

    task = load_task(pcg_tasks / CASE / "native")
    scorer = PhysCodeGymPixels(lpips=False)
    assert scorer.problems() == []
    frames = tmp_path / "out" / "frames"
    frames.mkdir(parents=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(task.hidden_dir / "reference" / "reference.mp4"),
         "-vf", "scale=640:360:flags=area", "-start_number", "1", str(frames / "frame_%04d.png")],
        check=True,
    )  # fmt: skip
    same = scorer.score(
        task, delivery_dir=tmp_path, output_dir=frames.parent, work_dir=tmp_path / "w", options={}
    )
    assert same["status"] == "scored" and same["S_SSIM"] > 0.95 and same["S_mot"] > 0.9
    written = (tmp_path / "w" / "physcodegym_metrics.json").read_text()
    assert str(tmp_path) not in written and str(pcg_tasks) not in written  # no host paths
    for frame in sorted(frames.glob("*.png"))[200:]:
        frame.unlink()
    short = scorer.score(
        task, delivery_dir=tmp_path, output_dir=frames.parent, work_dir=tmp_path / "w2", options={}
    )
    assert short["status"] == "invalid_frames" and short["S_SSIM"] == 0.0
    missing = scorer.score(
        task, delivery_dir=tmp_path, output_dir=None, work_dir=tmp_path / "w3", options={}
    )
    assert missing["status"] == "no_frames"


def _checkout_cached() -> bool:
    from real2simlab.benchmarks.physcodegym import CODE

    return (CODE.path() / ".r2s-complete").is_file()


def _clip(path: Path, fps: int, frames: int) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=640x360:rate={fps}",
         "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, stdin=subprocess.DEVNULL,
    )  # fmt: skip
    return path


@pytest.mark.slow
@needs_cv2
@pytest.mark.skipif(not _checkout_cached(), reason="the PhysCodeGym checkout is not cached")
def test_pixel_scorer_does_not_blame_the_delivery_for_the_reference(tmp_path: Path) -> None:
    """PhysCodeGym's own metrics (the pinned checkout) on 240 delivered frames that match a
    valid reference, with the reference replaced by one its reader rejects: the scorer failed
    (an infrastructure error), not the delivery's ``invalid_frames``. 239 frames are still the
    delivery's fault."""
    import shutil

    import cv2

    from real2simlab.benchmarks.physcodegym.scorer import PhysCodeGymPixels
    from real2simlab.scoring.native import scorer_failure
    from real2simlab.task import load_task

    good = _clip(tmp_path / "good.mp4", 30, 240)
    frames = tmp_path / "out" / "frames"
    frames.mkdir(parents=True)
    capture = cv2.VideoCapture(str(good))
    for index in range(1, 241):
        ok, image = capture.read()
        assert ok and cv2.imwrite(str(frames / f"frame_{index:04d}.png"), image)
    capture.release()
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(good.read_bytes()[:4096])
    references = {
        "valid": good,
        "25fps": _clip(tmp_path / "25fps.mp4", 25, 240),
        "120frames": _clip(tmp_path / "120frames.mp4", 30, 120),
        "truncated": cut,
    }
    scorer = PhysCodeGymPixels(lpips=False)

    def score(name: str) -> dict[str, Any]:
        task = tmp_path / "tasks" / name
        if not task.is_dir():
            shutil.copytree(ROOT / "examples" / "tasks" / "puck_slide", task)
            shutil.copyfile(references[name], task / "hidden" / "reference" / "reference.mp4")
        return scorer.score(
            load_task(task), delivery_dir=tmp_path, output_dir=frames.parent,
            work_dir=tmp_path / "w" / f"{name}-{len(list(frames.iterdir()))}", options={},
        )  # fmt: skip

    valid = score("valid")
    assert valid["status"] == "scored" and valid["S_mot"] == pytest.approx(1.0), valid
    for name in ("25fps", "120frames", "truncated"):
        report = score(name)
        assert report["status"] == "scorer_failed", (name, report)
        assert scorer_failure(scorer, report) is not None and "S_mot" not in report
    (frames / "frame_0240.png").unlink()
    short = score("valid")
    assert short["status"] == "invalid_frames" and short["S_mot"] == 0.0, short
    assert scorer_failure(scorer, short) is None


@pytest.mark.docker
@pytest.mark.slow
@needs_video
@needs_cv2
def test_blender_twin_end_to_end(pcg_tasks: Path, tmp_path: Path, docker_backend: Any) -> None:
    if not docker_image_ready("ghcr.io/real2simhub/r2s-blender-4.0:1.0"):
        pytest.skip("r2s env build r2s-blender-4.0")
    from real2simlab.orchestrate import parse_experiment, run_experiment

    twins = ROOT / "examples" / "twins" / "physcodegym"
    config = parse_experiment(
        {
            "name": "pcg",
            "tasks": {"paths": [str(pcg_tasks)]},
            "methods": [{"name": "program", "label": "hand", "path": str(twins)}],
            "run": {"backend": "docker"},
        }
    )
    run_experiment(
        config, tmp_path / "run", backend=docker_backend, log=lambda line: None, report=False
    )
    results = {
        json.loads(p.read_text())["task_id"].rsplit("/", 1)[1]: json.loads(p.read_text())
        for p in (tmp_path / "run" / "results").rglob("seed-0.json")
    }
    rerun = results["rerun"]
    assert rerun["native"]["status"] == "scored" and 0 < rerun["native"]["S_SSIM"] < 1
    assert rerun["reproduction"]["run_status"] == "succeeded"
    assert (
        results["native"]["native"]["status"] == "no_frames"
    )  # a program delivers no session frames
