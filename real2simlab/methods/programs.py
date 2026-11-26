"""A generic twin program that writes fixed tracks, used by the reference baselines."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from real2simlab.formats.tracks import Tracks
from real2simlab.twin.sdk import sdk_path

FIXED = '''\
"""A Real2SimLab baseline: {summary}"""

import json
from pathlib import Path

import numpy as np
import r2s_twin

HERE = Path(__file__).parent
MODEL = json.loads((HERE / "model.json").read_text())
twin = r2s_twin.Twin()
with np.load(HERE / "trajectory.npz") as archive:
    positions = {{key[: -len(".position")]: archive[key] for key in archive.files}}
twin.write_tracks(fps=MODEL["fps"], positions=positions, camera=MODEL["camera"])
'''


def fixed_tracks_program(target: Path, tracks: Tracks, *, summary: str) -> Path:
    """A program that writes ``tracks`` (positions and camera)."""
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sdk_path(), target / "r2s_twin.py")
    if tracks.camera is None:
        raise ValueError("the tracks need a camera")
    model = {"fps": tracks.fps, "camera": tracks.camera.to_json()}
    (target / "model.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    arrays: dict[str, Any] = {
        f"{name}.position": tracks.objects[name].position for name in tracks.objects
    }
    np.savez(target / "trajectory.npz", **arrays)
    (target / "run.py").write_text(FIXED.format(summary=summary), encoding="utf-8")
    return target
