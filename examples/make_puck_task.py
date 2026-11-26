"""Build the example task ``examples/puck_slide`` from the raw recording in ``raw/puck_slide``.

    python examples/make_puck_task.py [OUT_DIR] [RAW_DIR]

The recording (CC0, rendered by Real2SimLab) is a video with its calibrated camera, the
measurements an experimenter would note, the puck's measured trajectory, and three repeated
launches at the same setting (the noise floor). Written by Codex following
``.agents/skills/write-tasks/SKILL.md``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from real2simlab.capture import Provenance, SceneInfo, SceneObject, Sourced, TextInfo
from real2simlab.capture.from_video import capture_from_video
from real2simlab.capture.visibility import VisibilityPolicy
from real2simlab.formats.tracks import Camera, ObjectTrack, Tracks
from real2simlab.task import (
    Delivery,
    OutputRequirements,
    Protocol,
    ScoringConfig,
    SourceInfo,
    TaskBuilder,
    TaskConfig,
)

FPS = 30.0
FRAMES = 60
HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "puck_slide"


def _camera() -> Camera:
    data = json.loads((RAW / "camera.json").read_text())
    return Camera(
        intrinsics=np.asarray(data["intrinsics"]),
        width=data["width"],
        height=data["height"],
        extrinsics=np.asarray(data["extrinsics"]),
    )


def _tracks(path: Path) -> Tracks:
    with np.load(path) as data:
        position = np.asarray(data["position"], dtype=float)
    return Tracks(fps=FPS, objects={"puck": ObjectTrack(position=position)}, camera=_camera())


def _config() -> TaskConfig:
    return TaskConfig(
        id="examples/puck_slide",
        version="1.0.0",
        title="Puck launched across a floor",
        description=(
            "A red puck slides across a flat grey floor. The puck is launched from x = -0.6 m "
            "and its motion is recorded for 2 seconds."
        ),
        source=SourceInfo(
            dataset="Real2SimLab puck-slide recording",
            version="1.0",
            notes="Rendered by Real2SimLab (scripts/make_puck_recording.py).",
        ),
        license="CC0-1.0",
        redistributable=True,
        tags=("example", "puck", "sliding"),
        visibility="calibrated",
        delivery=Delivery(
            environment="r2s-python",
            output_format="tracks",
            run_timeout_s=120,
            output=OutputRequirements(fps=FPS, num_frames=FRAMES, objects=("puck",)),
        ),
        scoring=ScoringConfig(
            protocol=Protocol.REPRODUCTION,
            metrics=("traj3d_mean", "traj3d_final", "traj2d_norm_mean"),
            primary="traj3d_mean",
        ),
    )


def build_puck_slide(out_dir: Path) -> Path:
    """Build and validate the task, returning its directory."""
    config = _config()
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        capture = capture_from_video(
            RAW / "observed.mp4",
            work / "capture",
            capture_id=config.id,
            intrinsics=json.loads((RAW / "camera.json").read_text())["intrinsics"],
            extrinsics=json.loads((RAW / "camera.json").read_text())["extrinsics"],
            object_sizes={"puck": 0.08},
            scene_description=config.description,
        )
        measured = Provenance.MEASURED
        capture = capture.model_copy(
            update={
                "scene": SceneInfo(
                    objects=(
                        SceneObject(
                            name="puck",
                            role="dynamic",
                            shape="cylinder",
                            size=Sourced(value=(0.08, 0.08, 0.03), provenance=measured),
                            color="red",
                            initial_pose=Sourced(
                                value=[-0.6, 0.0, 0.015, 1.0, 0.0, 0.0, 0.0], provenance=measured
                            ),
                        ),
                        SceneObject(name="floor", role="static", shape="flat grey floor at z = 0"),
                    ),
                    environment="Flat grey floor, world-frame z is up.",
                ),
                "text": TextInfo(
                    task="Reproduce the puck's slide until it stops.",
                    scene_description=config.description,
                ),
            }
        )
        builder = TaskBuilder(out_dir / "puck_slide", config, overwrite=True)
        builder.capture(
            capture,
            work / "capture",
            visibility=VisibilityPolicy(
                name="calibrated",
                expose=frozenset(
                    {"views.*.intrinsics", "views.*.extrinsics", "world", "scene", "text"}
                ),
            ),
        )
        builder.reference(tracks=_tracks(RAW / "outcomes/observed.npz"))
        for index in range(1, 4):
            builder.repeat(index, tracks=_tracks(RAW / f"outcomes/observed_repeat{index}.npz"))
        return builder.finish().root


def main(argv: list[str]) -> int:
    global RAW
    out = Path(argv[1]) if len(argv) > 1 else HERE / "tasks"
    if len(argv) > 2:
        RAW = Path(argv[2])
    print(build_puck_slide(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
