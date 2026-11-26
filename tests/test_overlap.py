"""The overlap check: training data never holds an evaluation task's recording, and a method
trained on data is never evaluated on a task that overlaps it."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from real2simlab.benchmarks.base import get_benchmark
from real2simlab.benchmarks.synthetic import SyntheticBenchmark
from real2simlab.errors import SetupError
from real2simlab.media import tool
from real2simlab.methods.base import get_method
from real2simlab.orchestrate import Experiment, parse_experiment
from real2simlab.task import Split, TaskConfig, load_task
from real2simlab.task.io import read_task_config, write_task_config
from real2simlab.task.overlap import OverlapIndex, same_recording, task_print, thumbnails
from real2simlab.train import export_dataset, fit_method

VIDEO = Path("visible") / "cam" / "rgb.mp4"


def _copy(task: Path, target: Path, task_id: str, split: Split = Split.TRAIN) -> Path:
    """A copy of ``task`` under another id and split."""
    shutil.copytree(task, target)
    data = read_task_config(target / "task.toml").to_json()
    data["id"], data["split"] = task_id, split.value
    write_task_config(TaskConfig.model_validate(data), target / "task.toml")
    return target


def _transformed(task: Path, target: Path, task_id: str, *args: str) -> Path:
    """A copy of ``task`` whose video went through ffmpeg with ``args``, and without the
    reference (so only the video can show the overlap)."""
    _copy(task, target, task_id)
    shutil.rmtree(target / "hidden")
    source = task / VIDEO
    command = [tool("ffmpeg"), "-v", "error", "-y", "-i", str(source), *args]
    subprocess.run([*command, "-pix_fmt", "yuv420p", str(target / VIDEO)], check=True)
    return target


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("reencoded", ["-c:v", "libx264", "-crf", "30"]),
        ("resized", ["-vf", "scale=iw/2:-2", "-c:v", "libx264", "-crf", "23"]),
        ("trimmed", ["-vf", "trim=start_frame=20,setpts=PTS-STARTPTS", "-c:v", "libx264"]),
        ("fewer frames", ["-vf", "fps=15", "-c:v", "libx264"]),
        ("brighter", ["-vf", "eq=brightness=0.06", "-c:v", "libx264"]),
    ],
)
def test_a_copy_of_a_recording_overlaps_its_task(
    bouncing_task: Path, tmp_path: Path, name: str, args: list[str]
) -> None:
    copy = _transformed(bouncing_task, tmp_path / "copy", "mine/copy", *args)
    index = OverlapIndex([task_print(load_task(bouncing_task))])
    found = index.find(task_print(load_task(copy)))
    assert found is not None, name
    assert found.task_id == "synthetic/bouncing_ball/calibrated"
    assert "shows the same recording" in found.reason


def test_a_shared_file_overlaps(bouncing_task: Path, tmp_path: Path) -> None:
    copy = _copy(bouncing_task, tmp_path / "copy", "mine/copy")
    shutil.rmtree(copy / "visible")
    (copy / "visible").mkdir()
    found = OverlapIndex([task_print(load_task(bouncing_task))]).find(task_print(load_task(copy)))
    assert found is not None and "is the same file as" in found.reason
    assert "hidden/reference/tracks.npz" in found.reason


def test_other_events_do_not_overlap(synthetic_tasks: Path, tmp_path: Path) -> None:
    """Different phenomena, and training samples drawn far from the evaluation task."""
    tasks = [load_task(path) for path in sorted(synthetic_tasks.glob("*/calibrated"))]
    prints = [task_print(task) for task in tasks]
    for i, a in enumerate(prints):
        others = OverlapIndex(p for j, p in enumerate(prints) if j != i)
        assert others.find(a) is None, a.task_id
    (sample,) = SyntheticBenchmark().build(
        tmp_path, select=["bouncing_ball"], tiers=["calibrated"], split="train", count=1, seed=2
    )
    bouncing = next(p for p in prints if "bouncing" in p.task_id)
    assert OverlapIndex([bouncing]).find(task_print(load_task(sample))) is None


def test_a_video_matches_itself_at_the_right_offset(bouncing_task: Path, tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [tool("ffmpeg"), "-v", "error", "-i", str(bouncing_task / VIDEO), "-vf",
         "trim=start_frame=30:end_frame=60,setpts=PTS-STARTPTS", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )  # fmt: skip
    from real2simlab.task.overlap import VideoPrint

    whole = VideoPrint("whole", "a", *thumbnails(bouncing_task / VIDEO))
    part = VideoPrint("part", "b", *thumbnails(clip))
    match = same_recording(part, whole)
    assert match is not None and match.offset_s == pytest.approx(1.0, abs=0.04)


def test_the_index_survives_saving(bouncing_task: Path, tmp_path: Path) -> None:
    index = OverlapIndex([task_print(load_task(bouncing_task))])
    loaded = OverlapIndex.load(index.save(tmp_path / "prints.npz"))
    assert loaded.digest() == index.digest()
    assert loaded.find(task_print(load_task(bouncing_task))) is not None


def test_export_leaves_out_what_overlaps_the_evaluation_tasks(
    bouncing_task: Path, tmp_path: Path
) -> None:
    train = tmp_path / "train"
    _transformed(bouncing_task, train / "copy", "mine/copy", "-c:v", "libx264", "-crf", "28")
    SyntheticBenchmark().build(
        train, select=["bouncing_ball"], tiers=["calibrated"], split="train", count=1, seed=2
    )
    summary = export_dataset(tmp_path / "data", tasks=[train], against=[bouncing_task])
    assert summary.tasks == 1 and summary.checked_against == 1
    assert summary.overlapping == ["mine/copy"]
    report = json.loads((tmp_path / "data" / "export.json").read_text())
    (left_out,) = report["overlapping"]
    assert left_out["task_id"] == "mine/copy"
    assert left_out["overlaps"] == "synthetic/bouncing_ball/calibrated"
    assert report["checked_against"] == 1 and report["against_digest"]
    assert "left out" in (tmp_path / "data" / "README.md").read_text()
    prints = OverlapIndex.load(tmp_path / "data" / "overlap-prints.npz")
    assert [p.task_id for p in prints.prints] == [
        "synthetic-train/bouncing_ball/calibrated/s2-00000"
    ]
    unchecked = export_dataset(tmp_path / "data2", tasks=[train])
    assert unchecked.tasks == 2 and unchecked.checked_against is None
    assert "**not** compared" in (tmp_path / "data2" / "README.md").read_text()


def test_a_benchmark_without_a_training_split_gives_no_training_data(
    bouncing_task: Path, tmp_path: Path
) -> None:
    marked = _copy(bouncing_task, tmp_path / "iris", "iris/dropping_ball/marked")
    with pytest.raises(SetupError, match="iris releases no training split"):
        export_dataset(tmp_path / "data", tasks=[marked])
    with pytest.raises(SetupError, match="has no 'train' split"):
        get_benchmark("iris").build_split(tmp_path / "out", "train")
    (path,) = get_benchmark("synthetic").build_split(
        tmp_path / "out", "train", select=["pendulum"], tiers=["calibrated"], count=1
    )
    assert load_task(path).config.split is Split.TRAIN


@pytest.mark.slow
def test_a_trained_method_is_never_evaluated_on_its_training_data(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """The documented loop: training tasks, a dataset checked against the evaluation task, a
    trained method, an evaluation. Learning from the video beats the same method that only
    knows the training tasks' mean, and a training task given to the evaluation is skipped."""
    evaluation = tmp_path / "eval" / "bouncing_ball" / "calibrated"
    shutil.copytree(bouncing_task, evaluation)
    SyntheticBenchmark().build(
        tmp_path / "train", select=["bouncing_ball", "projectile"], tiers=["calibrated"],
        split="train", count=16,
    )  # fmt: skip
    export_dataset(tmp_path / "data", tasks=[tmp_path / "train"], against=[tmp_path / "eval"])
    record = fit_method(get_method("nearest_twin"), tmp_path / "data", tmp_path / "model")
    assert record.tasks == 32 and record.prints == "training-prints.npz"
    other = tmp_path / "other-model"
    shutil.copytree(tmp_path / "model", other)
    stored = json.loads((other / "training.json").read_text())
    (other / "training.json").write_text(json.dumps({**stored, "method": "someone_else"}))
    with pytest.raises(SetupError, match="was trained with method 'someone_else'"):
        get_method("nearest_twin", checkpoint=str(other))
    seen = tmp_path / "eval" / "seen"
    shutil.copytree(next((tmp_path / "train").rglob("s0-00003")), seen)
    config = parse_experiment(
        {
            "name": "learned",
            "tasks": {"paths": [str(tmp_path / "eval")]},
            "methods": [
                {"name": "nearest_twin", "checkpoint": str(tmp_path / "model")},
                {
                    "name": "nearest_twin",
                    "label": "prior",
                    "prior": True,
                    "checkpoint": str(tmp_path / "model"),
                },
            ],
            "run": {"backend": "local"},
        }
    )
    experiment = Experiment(config, tmp_path / "run", log=lambda line: None, report=False)
    results = experiment.run()
    skipped = [line for line in experiment.progress.skipped if "not held out" in line]
    assert len(skipped) == 2 and all("s0-00003" in line for line in skipped)
    skill = {
        row["method"]: row["reproduction_skill"]
        for row in results.rows
        if row["task"] == "synthetic/bouncing_ball/calibrated"
    }
    assert skill["nearest_twin"] > 0.9 and skill["nearest_twin"] > skill["prior"], skill
