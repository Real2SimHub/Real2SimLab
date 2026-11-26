"""Reference documents placed in every agent workspace (``docs/``)."""

from __future__ import annotations

from real2simlab.twin.sdk import sdk_source

RUN_CONTRACT = """\
# Run contract (r2s-twin/1.0)

Your program is frozen and re-run by the evaluator. Each run:

- starts in its own copy of `solution/`, mounted **read-only**;
- has **no network**;
- may write only to `$R2S_OUTPUT_DIR` (an empty directory) and `/tmp`;
- gets `R2S_SEED` (an integer; derive all randomness from it: the same seed must give the same
  output);
- must exit with status 0 on success.

The output of the run is what is scored: it must reproduce the recorded event.
"""

CAPTURE = """\
# The capture (r2s-capture/1.0)

`visible/capture.json` describes the observation. Its fields (all optional except `views`):

- `views.<name>.rgb`: the video file (`path`, `fps`, `num_frames`, `resolution` = [width, height]).
- `views.<name>.intrinsics.value`: 3x3 pinhole matrix in pixels (OpenCV convention).
- `views.<name>.extrinsics.value`: 4x4 camera-to-world transform (OpenCV axes: +x right,
  +y down, +z forward). A moving camera has `extrinsics_per_frame` (an .npy of F x 4 x 4).
- `world`: units (metres), up axis (+z), gravity, whether the geometry is metric.
- `scene.objects[]`: name, role (dynamic/static), shape, size (metres), mass, colour,
  `initial_pose` = [x, y, z, qw, qx, qy, qz] in the world frame.
- `text`: a description of the scene and the goal.

Every quantity carries a `provenance`: measured, estimated, default, human_edited or synthetic.
Conventions: metres, seconds, radians; world +z up; gravity along -z.

Decode frames with ffmpeg, e.g.
`ffmpeg -i visible/cam/rgb.mp4 -f rawvideo -pix_fmt rgb24 -` (then reshape to F x H x W x 3).
"""


def sdk_doc() -> str:
    """The SDK's own documentation (its module docstring) and where to find it."""
    source = sdk_source()
    docstring = source.split('"""')[1].strip() if source.startswith('"""') else ""
    return (
        "# The twin SDK: `r2s_twin`\n\n"
        "The file is `tools/r2s_twin.py`; it needs only NumPy and is importable as `r2s_twin` "
        "when your program is re-run (you may also copy it into `solution/`).\n\n"
        "```\n" + docstring + "\n```\n\n"
        "Main entry points: `r2s_twin.Twin()`, `twin.seed`, `twin.rng()`, "
        "`twin.write_tracks(fps=, positions=, camera=)`, "
        "`twin.write_video(frames, fps)`, `r2s_twin.Camera(...)` and `Camera.look_at(...)`, "
        "`python r2s_twin.py check OUTPUT_DIR`, `python r2s_twin.py selftest PROGRAM_DIR`.\n"
    )
