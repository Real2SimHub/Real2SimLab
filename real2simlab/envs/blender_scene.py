"""Runs a Blender scene program and turns its frames into ``render.mp4``. Standard library only.

The ``blender_scene`` delivery convention (PhysCodeGym's): ``scene.py`` builds a scene from an
empty file and renders frames ``frame_0001.png`` ... into ``/app/outputs/frames`` (and saves
``/app/outputs/model.blend``). This runner executes it headless with a time limit, then encodes
the frames as an H.264 ``render.mp4`` in the same directory, so the output is in the ``video``
format.

Usage: ``python3 blender_scene.py [SCENE] [OUTPUTS] [FPS]``
(defaults: ``$PWD/scene.py``, ``/app/outputs``, ``30``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    scene = Path(argv[1]) if len(argv) > 1 else Path.cwd() / "scene.py"
    outputs = Path(argv[2]) if len(argv) > 2 else Path("/app/outputs")
    fps = argv[3] if len(argv) > 3 else "30"
    blender = os.environ.get("BLENDER_BIN", "blender")
    if not scene.is_file():
        print(f"blender_scene: {scene} does not exist", file=sys.stderr)
        return 2
    (outputs / "frames").mkdir(parents=True, exist_ok=True)
    command = [
        blender,
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(scene),
    ]
    result = subprocess.run(command, check=False, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        print(f"blender_scene: Blender exited with {result.returncode}", file=sys.stderr)
        return result.returncode
    frames = sorted((outputs / "frames").glob("frame_*.png"))
    if not frames:
        print("blender_scene: no frames were rendered into frames/frame_*.png", file=sys.stderr)
        return 1
    encode = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-framerate",
        fps,
        "-start_number",
        "1",
        "-i",
        str(outputs / "frames" / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        str(outputs / "render.mp4"),
    ]
    return subprocess.run(encode, check=False, stdin=subprocess.DEVNULL).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv))
