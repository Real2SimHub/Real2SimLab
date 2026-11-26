# Output formats

A twin declares one output format; the task says which one it must use. Formats are plugins
(entry point group `real2simlab.formats`); each checks an output directory and, when it can,
converts it to `tracks` so trajectory metrics apply.

| Format | For | Converts to tracks |
|---|---|---|
| `tracks` | Lab-native, light: per-object trajectories and the camera | — |
| `4d_world` | 4DCodeBench's world package, checked by its official checker | yes (object centroids) |
| `video` | methods that only render, e.g. video world models | no |
| `video2sim_package` | Video2World's protocol package: a scene, a camera and actions | no |

## `tracks` (`r2s-tracks/1.0`)

```
meta.json     {"spec": "r2s-tracks/1.0", "fps": 30, "num_frames": 60, "objects": ["ball"],
               "frame": {"length_unit": "m", "up_axis": "+z"}}
camera.json   {"intrinsics": 3x3, "width": 320, "height": 240,
               "extrinsics": 4x4 camera-to-world, or a list of num_frames of them}
tracks.npz    <name>.position    float (F, 3)  the object's centroid, metres, world frame
              <name>.quaternion  float (F, 4)  optional, [qw, qx, qy, qz], unit norm
              <name>.visible     bool (F,)     optional; positions and quaternions may be
                                               NaN only where false
render.mp4    optional rendering
```

- Every object listed in `meta.json` has a position track; no other arrays are allowed.
- `meta.json` may also hold `notes` and `generator`; nothing else.
- The task's `[delivery.output]` adds requirements: frame rate, frame count and object names.
- Scoring resamples a prediction at the reference's frame times and errors if it does not
  cover them.

## `4d_world`

4DCodeBench's world (`camera.json`, `render.mp4`, `meshes/`, `dynamics/<name>.npz`). The
official checker is fetched at a pinned commit on first use (4DCodeBench has no license, so its
code is never bundled) and run in a subprocess. The conversion to tracks takes the mean of the
points of `dynamics/<name>.npz` (`pos`, shape (F, N, 3)) that exist at each frame.

## `video`

`render.mp4` (H.264), at the task's frame rate, frame count and resolution, and an optional
`meta.json` with `notes`.

## `video2sim_package`

Video2World's deliverable, which is data, not a program's output: `protocol.json` (the scene,
the camera and how the actions are to be read), an action stream (`actions.npy`, or the file
`protocol.json` names), meshes under `assets/` and, for most profiles, `expected/obj_poses.npy`.
The format checks only that `protocol.json` is a JSON object and that the action stream
exists; the profile's full contract is checked by Video2World's own validator and evaluator
([Video2World](../benchmarks/video2world.md)).

## Adding a format

Subclass `real2simlab.formats.OutputFormat` (`name`, `summary`, `validate`, optionally
`to_tracks` with `provides_tracks = True`, and `documentation`), and register it:

```toml
[project.entry-points."real2simlab.formats"]
my_format = "my_package.formats:MyFormat"
```
