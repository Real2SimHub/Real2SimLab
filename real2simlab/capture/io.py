"""Reading and writing capture directories."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from real2simlab._base import validation_problems
from real2simlab.capture.conventions import check_rigid_array
from real2simlab.capture.model import Capture
from real2simlab.errors import SpecError

CAPTURE_FILENAME = "capture.json"


def load_capture(capture_dir: Path, *, check_media: bool = True) -> Capture:
    """Load and validate ``<capture_dir>/capture.json``.

    With ``check_media``, every referenced media file must exist, and per-frame extrinsics
    must have the right shape and be rigid. Videos are not decoded. Raises ``SpecError``
    listing every problem.
    """
    directory = Path(capture_dir)
    path = directory / CAPTURE_FILENAME
    if not path.is_file():
        raise SpecError(f"{directory} is not a capture", [f"{CAPTURE_FILENAME} is missing"])
    try:
        capture = Capture.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise SpecError(f"{path} is not a valid capture", validation_problems(error)) from error
    if check_media:
        problems = media_problems(capture, directory)
        if problems:
            raise SpecError(f"capture {capture.capture_id!r} in {directory}", problems)
    return capture


def media_problems(capture: Capture, directory: Path) -> list[str]:
    """Every missing or malformed media file of ``capture`` stored in ``directory``."""
    problems = [
        f"missing media file: {relative}"
        for relative in capture.media_paths()
        if not (directory / relative).is_file()
    ]
    for name, view in capture.views.items():
        moving = view.extrinsics_per_frame
        if moving is None or not (directory / moving.path).is_file():
            continue
        field = f"views.{name}.extrinsics_per_frame ({moving.path})"
        try:
            loaded: object = np.load(directory / moving.path, allow_pickle=False)
        except Exception as error:  # not an array file numpy can read without pickles
            problems.append(f"{field}: cannot read: {type(error).__name__}: {error}")
            continue
        if not isinstance(loaded, np.ndarray):  # an .npz archive of arrays, not one array
            close = getattr(loaded, "close", None)
            if callable(close):
                close()
            problems.append(
                f"{field}: an NPZ archive, but the poses are one NPY array of shape (F, 4, 4): "
                "save them with numpy.save(path, poses), not numpy.savez"
            )
            continue
        poses = cast(NDArray[Any], loaded)
        end = view.first_frame + view.rgb.num_frames
        if poses.dtype.kind not in "iuf":
            problems.append(f"{field}: need numbers, got an array of {poses.dtype}")
            continue
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or poses.shape[0] < end:
            problems.append(f"{field}: need shape (>= {end}, 4, 4), got {poses.shape}")
            continue
        for index in range(view.first_frame, end):
            try:
                check_rigid_array(np.asarray(poses[index], dtype=np.float64))
            except ValueError as error:
                problems.append(f"views.{name}.extrinsics_per_frame[{index}]: {error}")
                break
    return problems


def save_capture(capture: Capture, capture_dir: Path) -> Path:
    """Write ``capture`` to ``<capture_dir>/capture.json`` and return that path."""
    path = Path(capture_dir) / CAPTURE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(capture.dumps(), encoding="utf-8")
    return path


def copy_media(capture: Capture, source_dir: Path, target_dir: Path) -> None:
    """Copy every media file ``capture`` references from ``source_dir`` into ``target_dir``."""
    for relative in capture.media_paths():
        target = Path(target_dir) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(source_dir) / relative, target)
