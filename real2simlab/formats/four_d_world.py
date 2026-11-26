"""The ``4d_world`` format: 4DCodeBench's world package, checked by 4DCodeBench's own checker.

4DCodeBench has no license, so its checker is not bundled: it is fetched at a pinned commit on
first use (``real2simlab.vendor``) and run in a subprocess with the checkout on ``PYTHONPATH``.
Its top-level packages are called ``config`` and ``checker``, which would collide with other
packages if imported into this process.

The conversion to ``tracks`` takes each ``dynamics/<name>.npz``'s ``pos`` array (F, N, 3) and
uses the mean of the points that exist at a frame as the object's centroid.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np

from real2simlab.errors import SetupError
from real2simlab.formats.base import OutputFormat, Requirements
from real2simlab.formats.tracks import Camera, ObjectTrack, Tracks
from real2simlab.media import MediaError, probe_video
from real2simlab.sandbox.process import run_captured
from real2simlab.vendor import FOURDCODEBENCH

_CHECK_SCRIPT = (
    "import json, sys\n"
    "from checker.check import check_world\n"
    "try:\n"
    "    problems = check_world(sys.argv[1])\n"
    "except ImportError:\n"
    "    raise\n"
    "except Exception as error:  # the world cannot be read: a problem of the output\n"
    "    problems = [f'the checker cannot read it: {type(error).__name__}: {error}']\n"
    "print(json.dumps([str(problem) for problem in problems]))\n"
)
"""Runs 4DCodeBench's checker; a missing dependency fails the script (this machine's problem),
a world the checker cannot read is a problem of the world."""


class FourDWorldFormat(OutputFormat):
    """4DCodeBench's world: camera, render, per-frame meshes and identity-keeping tracks."""

    name: ClassVar[str] = "4d_world"
    summary: ClassVar[str] = "4DCodeBench's world package, checked by its official checker."
    provides_tracks: ClassVar[bool] = True
    content: ClassVar[tuple[str, ...] | None] = (
        "camera.json",
        "render.mp4",
        "meshes/**/*",
        "dynamics/**/*",
        "solver/manifest.json",
    )
    """The world as 4DCodeBench defines it; a liquid solver's own cache is not compared."""

    def validate(self, root: Path, requirements: Requirements | None = None) -> list[str]:
        checkout = FOURDCODEBENCH.ensure()
        env = {**os.environ, "PYTHONPATH": str(checkout)}
        result = run_captured(  # stopped with the experiment
            [sys.executable, "-P", "-c", _CHECK_SCRIPT, str(Path(root).resolve())],
            env=env,
            cwd=checkout,
        )
        if result.returncode != 0:
            raise SetupError(
                f"the 4DCodeBench checker failed to run:\n{result.stderr.strip()[-2000:]}",
                fix="install the checker's dependencies: pip install trimesh pillow",
            )
        problems = [str(item) for item in cast(list[Any], json.loads(result.stdout))]
        render = Path(root) / "render.mp4"
        if requirements is not None and render.is_file():
            try:
                info = probe_video(render)
            except MediaError as error:
                return [*problems, f"render.mp4: {error}"]
            if requirements.num_frames is not None and info.num_frames != requirements.num_frames:
                problems.append(
                    f"render.mp4: {info.num_frames} frames, the reference has "
                    f"{requirements.num_frames}"
                )
            if requirements.fps is not None and abs(info.fps - requirements.fps) > 1e-3:
                problems.append(
                    f"render.mp4: {info.fps:g} fps, the reference has {requirements.fps:g}"
                )
        return problems

    def to_tracks(self, root: Path) -> Tracks:
        root = Path(root)
        objects: dict[str, ObjectTrack] = {}
        for path in sorted((root / "dynamics").glob("*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                pos = np.asarray(archive["pos"], dtype=np.float64)
            if pos.ndim == 2:
                pos = pos[:, None, :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN frames
                centroid = np.nanmean(pos, axis=1)
                # Points so far out that their sum overflows still have a finite mean: those
                # are averaged again, each coordinate divided by its largest magnitude first.
                redo = ~np.isfinite(centroid) & ~np.isinf(pos).any(axis=1)
                redo &= ~np.isnan(pos).all(axis=1)
                if redo.any():
                    largest = np.nanmax(np.abs(pos), axis=1)
                    rescaled = largest * np.nanmean(pos / largest[:, None, :], axis=1)
                    centroid[redo] = rescaled[redo]
            visible = np.isfinite(centroid).all(axis=1)
            objects[path.stem] = ObjectTrack(position=centroid, visible=visible)
        fps = 30.0
        camera = None
        camera_file = root / "camera.json"
        if camera_file.is_file():
            data = cast(dict[str, Any], json.loads(camera_file.read_text(encoding="utf-8")))
            fps = float(data.get("fps", fps))
            intrinsics = np.asarray(data.get("intrinsics", data.get("K")), dtype=np.float64)
            extrinsics = np.asarray(
                data.get("extrinsic", data.get("extrinsics", np.eye(4).tolist())), dtype=np.float64
            )
            width, height = int(data.get("width", 0)), int(data.get("height", 0))
            render = root / "render.mp4"
            if (not width or not height) and render.is_file():
                info = probe_video(render)
                width, height, fps = info.width, info.height, info.fps
            if (not width or not height) and intrinsics.shape == (3, 3):
                # 4DCodeBench's camera.json has no resolution: assume a centred principal point.
                width = round(2 * float(intrinsics[0, 2]))
                height = round(2 * float(intrinsics[1, 2]))
            camera = Camera(
                intrinsics=intrinsics, width=width, height=height, extrinsics=extrinsics
            )
        return Tracks(fps=fps, objects=objects, camera=camera)

    def documentation(self) -> str:
        return (
            "# Output format `4d_world`\n\n"
            "4DCodeBench's world package (`camera.json`, `render.mp4`, `meshes/`, "
            "`dynamics/`). See `world-format.md` from the 4DCodeBench repository, provided "
            "in `docs/`, and check your world with the provided checker.\n"
        )
