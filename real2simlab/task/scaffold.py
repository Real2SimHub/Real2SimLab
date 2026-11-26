"""``r2s task new``: a task directory skeleton for authors to fill in."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from real2simlab.capture.from_video import capture_from_video
from real2simlab.errors import InputNotFoundError, OutputExistsError, UsageError
from real2simlab.task.io import HIDDEN_DIR, REFERENCE_DIR, TASK_FILE, VISIBLE_DIR
from real2simlab.task.model import TASK_ID

TEMPLATE = """\
# A Real2SimLab task (r2s-task/1.0). See "Write tasks" in the documentation:
# https://real2simhub.github.io/Real2SimLab/tutorials/write-tasks/
# Run `r2s task validate DIRECTORY` after every change.
spec = "r2s-task/1.0"
id = {task_id}
version = "1.0.0"
title = {title}
description = "TODO: one paragraph a method reads first."
license = "TODO: SPDX id of the data, e.g. CC-BY-4.0"
redistributable = false

[source]
dataset = "TODO: dataset name"
version = "TODO"
url = "TODO"

[delivery]
environment = "r2s-python"
output_format = "tracks"
agent_time_budget_s = 1200
run_timeout_s = 300

[delivery.output]
fps = {fps}
num_frames = {num_frames}
objects = ["TODO_object"]

[scoring]
protocol = "reproduction"
metrics = ["traj3d_mean", "traj3d_rmse", "traj3d_final", "traj2d_norm_mean"]
primary = "traj3d_mean"
"""

NEXT_STEPS = """\
Created {path}. Next:
  1. Edit {path}/task.toml (every TODO).
  2. Put the observation in {path}/visible/ (capture.json + media), e.g. with
     r2s capture from-video VIDEO {path}/visible --capture-id ...
  3. Put the true output of the recorded event in {path}/hidden/reference/ (in the task's
     output format), and other recordings of the same event, if any, in
     {path}/hidden/repeats/001/, 002/, ... (they give the noise floor).
  4. r2s task validate {path}
"""


def new_task(
    path: Path,
    *,
    task_id: str,
    title: str,
    video: Path | None = None,
) -> str:
    """Create a skeleton at ``path``; with ``video``, also its capture. Return next steps.

    Nothing is left behind when it fails: ``path`` is as it was (missing or empty)."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise OutputExistsError(
            f"{path} exists and is not an empty directory", fix="choose a new task directory"
        )
    if not TASK_ID.match(task_id):
        raise UsageError(
            f"task id {task_id!r} must be lowercase path segments of [a-z0-9_.-] separated by "
            "'/', e.g. mybench/drop_01"
        )
    if video is not None and not Path(video).is_file():
        raise InputNotFoundError(
            f"{video} is not a video file", fix="pass the path of the recording (an .mp4)"
        )
    existed = path.exists()
    try:  # this command's: another one making a task here meanwhile is refused, not undone
        (path / VISIBLE_DIR).mkdir(parents=True)
    except FileExistsError:
        raise OutputExistsError(
            f"{path} exists and is not an empty directory (another command is writing it)",
            fix="choose a new task directory",
        ) from None
    try:
        (path / HIDDEN_DIR / REFERENCE_DIR).mkdir(parents=True, exist_ok=True)
        fps, num_frames = 30, 60
        if video is not None:
            capture = capture_from_video(
                video, path / VISIBLE_DIR, capture_id=task_id, task_text=title
            )
            view = next(iter(capture.views.values()))
            fps, num_frames = view.rgb.fps, view.rgb.num_frames
        text = TEMPLATE.format(
            task_id=_toml_string(task_id),
            title=_toml_string(title),
            fps=fps,
            num_frames=num_frames,
        )
        (path / TASK_FILE).write_text(text, encoding="utf-8")
    except BaseException:
        # Only what this call made: ``path`` was missing or empty when it started.
        if not existed:
            shutil.rmtree(path, ignore_errors=True)
        else:
            for child in path.iterdir():
                shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink()
        raise
    steps = NEXT_STEPS.format(path=path)
    if video is not None:
        lines = steps.splitlines()
        steps = "\n".join(
            [
                lines[0],
                lines[1],
                f"  2. {path}/visible/ already holds the capture of {video}.",
                *lines[4:],
            ]
        )
    return steps


def _toml_string(value: str) -> str:
    """``value`` as a TOML basic string (a JSON string is one: same quotes and escapes)."""
    return json.dumps(value, ensure_ascii=False)
