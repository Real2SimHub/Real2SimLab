"""PhysCodeGym's model-free official metrics as a native scorer plugin (``physcodegym_pixels``).

Runs ``physcodegym.metrics.evaluate`` from the pinned checkout, in a subprocess (its top-level
packages are ``physcodegym`` and ``experiment``), on the 4K reference video and the candidate's
240 PNG frames: ``S_SSIM``, ``S_mot`` and, when ``lpips`` and ``torch`` are importable,
``S_LPIPS``. Needs OpenCV (the ``physcodegym`` extra).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from real2simlab._project import install_command
from real2simlab.benchmarks.physcodegym import CODE, FRAMES
from real2simlab.errors import SetupError
from real2simlab.sandbox.process import PROBE_TIMEOUT_S, run_captured
from real2simlab.scoring.model import Headline
from real2simlab.scoring.native import SCORER_FAILED, NativeScorer, default_failure
from real2simlab.task.io import Task

SCRIPT = """\
import json, sys
from pathlib import Path
from physcodegym.metrics import evaluate
result = evaluate(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), lpips=sys.argv[4] == "1")
keys = ("S_SSIM", "S_LPIPS", "S_mot", "motion", "lpips_enabled", "frame_count", "schema_version")
print(json.dumps({key: result.get(key) for key in keys}))
"""
IMAGE_SUFFIXES = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp")
"""The files PhysCodeGym's reader takes from a directory of frames, every one as a frame
(``IMAGE_SUFFIXES`` in ``experiment/verifier/visual_metrics.py``)."""
FRAME_ERRORS = (
    "Failed to decode image:",
    "No supported images found in directory:",
    "Image shapes differ:",
    "Images must contain only finite values",
)
"""How the ``ValueError`` messages of PhysCodeGym's metrics that are about the candidate's frames
alone begin: the reference is a video, whose frames OpenCV decodes in colour."""
FRAME_COUNT_ERROR = "Both inputs must contain exactly 240 aligned frames"
"""About either input: the candidate's fault when it does not hold exactly 240 frames."""
REFERENCE_ERRORS = (
    "The benchmark requires 30 FPS",  # only a video has a frame rate: the reference
    "Failed to open video:",
    "Video has invalid timing metadata:",
    "Decoded ",
    FRAME_COUNT_ERROR,
)
"""How PhysCodeGym's messages about the reference video, as this machine's OpenCV reads it,
begin."""
UNREADABLE_FRAMES = "OpenCV cannot read "
"""How the report of delivered frames OpenCV cannot read begins (``_unreadable_frames``)."""


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


def _unreadable_frames(frames: Path) -> str | None:
    """Which frames OpenCV, the reader PhysCodeGym's metrics use, cannot read (none: ``None``)."""
    import cv2  # pyright: ignore[reportMissingImports]  # an extra: physicsiq, physcodegym

    bad = [p.name for p in sorted(frames.glob("frame_*.png")) if cv2.imread(str(p)) is None]
    if not bad:
        return None
    return f"{UNREADABLE_FRAMES}{len(bad)} frame(s): {', '.join(bad[:3])}"


def _opencv_png_problem(work_dir: Path) -> str | None:
    """Why this machine's OpenCV cannot read back a PNG it writes itself (``None`` when it can):
    then frames it cannot read are not the delivery's fault."""
    import cv2  # pyright: ignore[reportMissingImports]  # an extra: physicsiq, physcodegym
    import numpy as np

    probe = work_dir / "opencv_probe.png"
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(probe), np.zeros((4, 4, 3), np.uint8)):
            return "this machine's OpenCV cannot write a PNG"
        if cv2.imread(str(probe)) is None:
            return "this machine's OpenCV cannot read a PNG it wrote itself"
    except (OSError, cv2.error) as error:
        return f"this machine's OpenCV cannot write and read a PNG ({error})"
    finally:
        probe.unlink(missing_ok=True)
    return None


def _official_frames(frames: Path) -> int:
    """How many frames PhysCodeGym's reader takes from ``frames``: each image file in it."""
    return sum(1 for p in frames.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())


def _official_error(error: str, frames: Path) -> dict[str, Any]:
    """The report of PhysCodeGym's metrics failing with ``error`` (the last line of the error
    they printed). Their checks raise ``ValueError`` about either input: only one about the
    delivered ``frames`` is the delivery's result (``invalid_frames``, a score of 0). One about
    the reference video, as this machine's OpenCV reads it (not 30 fps, not 240 frames, not
    readable, as with an OpenCV without a video backend), or any other error, is the scorer
    failing, not a score of the delivery."""
    kind, _, message = error.partition(": ")
    if kind == "ValueError" and (
        message.startswith(FRAME_ERRORS)
        or (message.startswith(FRAME_COUNT_ERROR) and _official_frames(frames) != FRAMES)
    ):
        return {"status": "invalid_frames", "detail": error, "S_SSIM": 0.0, "S_mot": 0.0}
    if kind == "ValueError" and message.startswith(REFERENCE_ERRORS):
        error += "; about the task's reference video as this machine's OpenCV reads it"
    return {"status": SCORER_FAILED, "detail": error}


def _blames_the_frames(detail: str) -> bool:
    """Whether ``detail``, an ``invalid_frames`` report's, is one this scorer gives the delivery:
    frames OpenCV cannot read, or a ``ValueError`` of PhysCodeGym's checks of the frames
    (``FRAME_ERRORS``, and the frame count, which either input may fail: the report alone cannot
    tell which)."""
    kind, _, message = detail.partition(": ")
    if detail.startswith(UNREADABLE_FRAMES):
        return True
    return kind == "ValueError" and message.startswith((*FRAME_ERRORS, FRAME_COUNT_ERROR))


class PhysCodeGymPixels(NativeScorer):
    name: ClassVar[str] = "physcodegym_pixels"
    task_options: ClassVar[tuple[str, ...] | None] = ("case",)
    summary: ClassVar[str] = "PhysCodeGym's SSIM, motion-mask and (optional) LPIPS scores"

    def headline(self, report: Mapping[str, Any]) -> Headline:
        value = report.get("S_mot")
        return Headline(
            metric="S_mot",
            value=float(value) if isinstance(value, int | float) else None,
            range=(0.0, 1.0),
            source="official",
            scorer=self.name,
        )

    def __init__(self, lpips: bool | None = None) -> None:
        self.lpips = _importable("lpips", "torch") if lpips is None else lpips

    def problems(self) -> list[str]:
        if not _importable("cv2"):
            return ["OpenCV is not installed: " + install_command("physcodegym")]
        return []

    def failure(self, report: Mapping[str, Any]) -> str | None:
        """The scorer's own failure (``scorer_failed``), and a report kept as the delivery's
        ``invalid_frames`` for an error this scorer does not blame on the delivered frames, as a
        Lab that blamed the delivery for any ``ValueError`` of PhysCodeGym's metrics (the
        reference video's too) kept it: scored again, not reused."""
        failed = super().failure(report)
        if failed is not None or report.get("status") != "invalid_frames":
            return failed
        detail = report.get("detail")
        if not isinstance(detail, str) or _blames_the_frames(detail):
            return None
        return default_failure(self.name, {**report, "status": SCORER_FAILED})

    def score(
        self,
        task: Task,
        *,
        delivery_dir: Path,
        output_dir: Path | None,
        work_dir: Path,
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        frames = None if output_dir is None else output_dir / "frames"
        if frames is None or not frames.is_dir() or not any(frames.glob("frame_*.png")):
            return {"status": "no_frames", "S_SSIM": 0.0, "S_mot": 0.0}
        unreadable = _unreadable_frames(frames)
        if unreadable:  # the official metrics read the frames with OpenCV: the delivery's fault,
            machine = _opencv_png_problem(work_dir.resolve())
            if machine:  # unless this machine's OpenCV reads no PNG at all
                return {"status": SCORER_FAILED, "detail": f"{machine}; {unreadable}"}
            return {
                "status": "invalid_frames", "detail": unreadable, "S_SSIM": 0.0, "S_mot": 0.0
            }  # fmt: skip
        reference = (task.hidden_dir / "reference" / "reference.mp4").resolve()
        frames = frames.resolve()
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONPATH": str(CODE.ensure())}
        command = [  # paths relative to work_dir: the metrics file records them as given
            sys.executable, "-c", SCRIPT, os.path.relpath(reference, work_dir),
            os.path.relpath(frames, work_dir), "physcodegym_metrics.json",
            "1" if self.lpips else "0",
        ]  # fmt: skip
        result = run_captured(command, env=env, cwd=work_dir)  # stopped with the experiment
        if result.returncode != 0:
            error = result.stderr.strip().splitlines()[-1:] or ["unknown error"]
            if "ModuleNotFoundError" in result.stderr:
                raise SetupError(
                    f"PhysCodeGym's metrics cannot run: {error[0]}",
                    fix=install_command("physcodegym"),
                )
            # The official checks of the frames (count, decoding), or the official code failing
            # on something else (the reference, this machine), which is not a score.
            return _official_error(error[0], frames)
        scores = json.loads(result.stdout.strip().splitlines()[-1])
        return {"status": "scored", **scores}
