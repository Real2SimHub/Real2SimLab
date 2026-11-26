# Captures (`r2s-capture/1.0`)

A capture is the complete observation record of one physical event: `capture.json` plus the
media files it references by relative path. A task's `visible/` holds the part a method may
see; a visibility policy decides which fields that is.

```json
{
  "spec": "r2s-capture/1.0",
  "capture_id": "synthetic/bouncing_ball",
  "source": {"dataset": "real2simlab-synthetic", "version": "1.0", "license": "Apache-2.0"},
  "world": {"length_unit": "m", "up_axis": "+z",
            "gravity": {"value": [0, 0, -9.81], "provenance": "default"},
            "metric_scale": {"value": true, "provenance": "synthetic"}},
  "views": {
    "cam": {
      "rgb": {"path": "cam/rgb.mp4", "fps": 30, "num_frames": 60, "resolution": [320, 240]},
      "intrinsics": {"value": [[277.1, 0, 160], [0, 277.1, 120], [0, 0, 1]], "provenance": "measured"},
      "extrinsics": {"value": [[...4x4 camera-to-world...]], "provenance": "measured"}
    }
  },
  "scene": {"objects": [{"name": "ball", "role": "dynamic", "shape": "sphere",
                         "size": {"value": [0.1, 0.1, 0.1], "provenance": "measured"},
                         "color": "red",
                         "initial_pose": {"value": [0, 0, 0.5, 1, 0, 0, 0], "provenance": "measured"}}],
            "environment": "A flat floor at z = 0."},
  "text": {"task": "...", "scene_description": "..."},
  "extensions": {}
}
```

## Fields

| Field | Content |
|---|---|
| `views.<name>.rgb` | the video: `path`, `fps`, `num_frames`, `resolution` (width, height), optional per-frame `timestamps` (s) |
| `views.<name>.intrinsics` | sourced 3×3 pixel matrix |
| `views.<name>.extrinsics` | sourced 4×4 camera-to-world transform of a static camera |
| `views.<name>.extrinsics_per_frame` | a moving camera: `path` of an `.npy` (F, 4, 4), with `provenance`; excludes `extrinsics` |
| `views.<name>.depth` | per-frame depth maps: `path`, `unit` (m, mm), `kind` (metric, relative), provenance |
| `views.<name>.masks` | per-frame object masks: `path`, `format`, `labels`, provenance |
| `views.<name>.frame_range` | the visible window `[start, end)` into every per-frame file of the view |
| `world` | units (metres), up axis (+z), gravity vector, whether geometry is metric |
| `scene.objects[]` | `name`, `role` (dynamic, static), `shape`, `size` (metres along local axes), `mass`, `material`, `color`, `initial_pose` |
| `robot` | a time-synchronised robot recording: URDF, action type, control frequency, trajectory (XPolicyLab keys), time offset |
| `text` | `task` and `scene_description` |
| `source` | the dataset, version, license and URL |
| `extensions` | rare, method-specific inputs kept out of the core schema |

Multiple views are first-class. Views with different frame rates or frame counts must all
carry timestamps. Object names match `[A-Za-z0-9][A-Za-z0-9_-]*`; they are the keys of output
tracks.

## Observation profiles

Profiles name what a capture offers, detected from its fields: `rgb`, `rgb_calib` (every view
calibrated and metric), `rgbd`, `multiview`, `moving_camera`, `robot`. A method may declare
the profiles it requires.

## Visibility policies

A policy lists the field paths a method may see (`views.*.intrinsics`, `scene`, …), and
optionally a frame window (when a benchmark shows only the first part of a clip, as Physics-IQ
does). Applying it to the full capture yields
the visible capture and an audit of what was withheld; only the media the visible capture
still references are copied. Two policies ship with the Lab:

- `calibrated`: video, cameras, world frame, scene objects, text and source;
- `video_only`: video, text and source.

## From an ordinary video

```bash
r2s capture from-video drop.mp4 visible/ --capture-id lab/drop_01 \
    --fov 62 --object ball=0.067 --text "A tennis ball is dropped onto a table."
```

`--intrinsics fx,fy,cx,cy` gives measured intrinsics; `--fov` estimates them from a horizontal
field of view (provenance `estimated`); `--object NAME=SIZE` records a known size, which gives
the scene its scale; `--extrinsics FILE` reads a 4×4 camera-to-world matrix.
