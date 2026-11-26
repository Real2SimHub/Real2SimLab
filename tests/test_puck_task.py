"""The example task written with the write-tasks skill builds and validates."""

from pathlib import Path

from examples.make_puck_task import build_puck_slide
from real2simlab.task import load_task, validate_task


def test_puck_task_builds_and_validates(tmp_path: Path) -> None:
    root = build_puck_slide(tmp_path)
    report = validate_task(root)
    assert report.ok, report.issues
    assert not report.warnings, report.warnings
    task = load_task(root)
    assert task.config.scoring.protocol.value == "reproduction"
    assert task.reference_dir().is_dir() and len(task.repeat_dirs()) == 3


def test_committed_task_is_current(tmp_path: Path) -> None:
    committed = Path(__file__).resolve().parents[1] / "examples" / "tasks" / "puck_slide"
    assert validate_task(committed).ok
    rebuilt = build_puck_slide(tmp_path)
    assert (rebuilt / "task.toml").read_text() == (committed / "task.toml").read_text()
