# Write a twin with the SDK

A twin is a program that simulates a recorded event and writes its output in the task's format.
This tutorial writes one for `synthetic/bouncing_ball/calibrated` by hand with the SDK, checks
it the way the evaluation will, and scores it. A Real2Sim pipeline delivers its results the same
way.

## The SDK

`r2s_twin.py` is a single file that needs only NumPy. It is installed in every runtime image
(importable as `r2s_twin`) and given to agents as `tools/r2s_twin.py`. Copy it next to your
program to run it locally:

```bash
python -c "from real2simlab.twin.sdk import sdk_path; print(sdk_path())"
```

## The program

```python
# twin/run.py
import numpy as np
import r2s_twin

twin = r2s_twin.Twin()                       # reads R2S_OUTPUT_DIR and R2S_SEED
height, g = 0.5, 9.81                        # from the task's capture
restitution, radius, fps, frames = 0.76, 0.05, 30.0, 60

t = np.arange(frames) / fps
z = np.empty(frames)
z0, v0, start = height, 0.0, 0.0
for i, ti in enumerate(t):
    tau = ti - start
    zi = z0 + v0 * tau - 0.5 * g * tau * tau
    if zi < radius:                         # a bounce (coarse: at frame resolution)
        v0 = restitution * abs(v0 - g * tau)
        z0, start, zi = radius, ti, radius
    z[i] = zi

camera = r2s_twin.Camera.look_at((0, -1.6, 0.55), (0, 0, 0.45), width=320, height=240)
twin.write_tracks(fps=fps, positions={"ball": np.c_[0 * z, 0 * z, z]}, camera=camera)
```

What the SDK does for you:

- `Twin()` reads the output directory and the seed the evaluation passes; `twin.seed` and
  `twin.rng()` make randomness reproducible.
- `write_tracks(...)` writes `meta.json`, `camera.json` and `tracks.npz` in metres and checks
  them; `write_video(frames, fps)` writes `render.mp4` for tasks scored on video.
- `r2s_twin.Camera` holds a pinhole camera (`look_at`, `project`); use the capture's calibrated
  camera when the task gives one.

## Check it

```bash
python twin/r2s_twin.py selftest twin
```

`selftest` runs the program as the evaluation does, twice: it must write something, its tracks
must be valid, and every file of the two outputs must be identical (arrays exactly, JSON by
value, other files byte for byte); it names the files that differ. Videos are neither checked
nor compared here: `r2s twin check` does both. Then check it frozen, in the sandbox, with the task's rules (output format,
frame count, objects, size limits):

```bash
r2s twin check twin --task tasks/bouncing_ball/calibrated --backend docker
```

## Evaluate it

```bash
r2s eval --tasks tasks/bouncing_ball/calibrated --method mine=program:path=twin \
    --method oracle --method static --out runs/mine
```

`program` takes a program directory, or a directory of them, one per task. A directory that
holds the entry point (`run.py`) is the program for every task, whatever its subdirectories
hold, as `r2s train reward --solution` takes it (so the reward and the evaluation of it are
equal). A directory without one holds a program per task, named after the task id: as nested
directories (`twins/synthetic/bouncing_ball/calibrated/`), as one name with `/` written as `__`
(`twins/synthetic__bouncing_ball__calibrated/`), or after one segment of the id that tasks
share (`twins/calibrated/`); the task's own directory wins over a shared one. A directory that
is a program and also holds one for a task is ambiguous: the evaluation refuses it before
anything runs, with the fix. An empty `path=` is refused too (it is not the current directory),
and a program holding a symbolic link or a file it cannot read, or reached through a symbolic
link (a per-task directory, a twin package's `program/`), delivers nothing for its task, as its
reward is 0. A twin package (`twin.json` and `program/`) works wherever a program directory
does. Your reproduction skill should be close to the oracle's 1 and far above the
static scene's 0; the coarse bounce above loses a little at every impact.
