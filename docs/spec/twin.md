# Twins (`r2s-twin/1.0`)

A **twin** is the program a method delivers for a task: it simulates the recorded event and
writes its output in the task's format. The Lab packages it, freezes it and re-runs it offline.
A twin package is a directory:

```
<twin>/
  twin.json      the declaration (written by the Lab or a method adapter, never by an agent)
  program/       the program: any files, no symbolic links, within the task's size limits
  assets/        optional large files, verified by SHA-256
```

## `twin.json`

```json
{
  "spec": "r2s-twin/1.0",
  "twin_id": "codex/synthetic/bouncing_ball/calibrated/trial0",
  "entrypoint": ["python", "./run.py"],
  "environment": {"image": "r2s-python", "digest": null},
  "output_format": "tracks",
  "runtime": {"program_path": "/r2s/program", "output_path": "/r2s/output"},
  "assets": [{"path": "weights.bin", "sha256": "…", "size": 1048576, "url": null}],
  "origin": {"method": "codex", "agent": {"name": "codex", "version": "0.167.2"},
             "model": "…", "task_id": "synthetic/bouncing_ball/calibrated",
             "task_version": "1.0.0", "created_at": "2026-11-12T10:00:00Z", "privileged": false}
}
```

- `entrypoint` runs inside `program/`; items starting with `./` must name files in it.
- `environment` is the runtime image (a registered name such as `r2s-python`, or any image
  reference). `environment`, `output_format` and `origin` come from the task and the method
  adapter, not from the program's author.
- `runtime` places the program and output inside the sandbox; the default suits the `r2s`
  convention, while 4DCodeBench deliveries use `/workspace/solution` and `/workspace/world`.
  With `replaceable_output: true` (4DCodeBench's `build.sh` "replaces anything already in
  `world/`") the output's parent is a writable scratch directory, so the program may delete and
  recreate the output directory; only that directory is kept, and the program stays read-only.
- `origin.privileged` marks twins built with hidden data (the oracle and reference baselines).

## Delivery conventions

| Convention | The method hands in | The frozen run |
|---|---|---|
| `r2s` (also Physics-IQ, PhySimCode) | `solution/run.py` and anything it needs | `python run.py` under the run contract below |
| `build_sh_world` (4DCodeBench) | `solution/build.sh` | writes `/workspace/world` |
| `blender_scene` (PhysCodeGym) | `solution/scene.py` | Blender builds and renders it headless; `render.mp4` |
| `video2sim_package` (Video2World) | a protocol package | not run by the Lab: the benchmark's evaluator executes it |

## Program rules

- No symbolic links; only regular files and directories.
- Total size at most `delivery.program.max_size_mb` (default 50 MB).
- No video files unless `delivery.program.allow_video` (default false): a program may not copy
  the recording.
- No other suffix in `delivery.program.forbidden_suffixes`.

## Run contract

Each run of a frozen twin:

| | |
|---|---|
| working directory | the program directory, mounted **read-only** |
| `R2S_OUTPUT_DIR` | an empty directory; the program writes its complete output there |
| `R2S_SEED` | always set; a non-negative integer below 2³²; every source of randomness derives from it |
| `R2S_ASSETS_DIR` | set when the twin declares assets; read-only |
| network | none |
| writable | only `R2S_OUTPUT_DIR` and `/tmp` (also `HOME` and `TMPDIR`) |
| determinism | the same seed must give the same output |
| exit status | 0 on success |

The fingerprint of the whole package (and of `program/`) is taken when the twin is loaded
and checked before and after every run; a twin that changes is refused. It covers every
file's relative path, bytes and executable bit (a program that runs `./helper.sh` behaves
differently without it). A twin that changed while it ran on `docker` or `bwrap`, where its
program cannot write it, was changed from outside the sandbox (an editor, a sync client): the
run is an `infra_error`, not the program's result, and an experiment builds the twin again
on its next run. On `local`, where the program could have changed it, the run failed.

## The SDK (`r2s_twin.py`)

A single file that needs only NumPy and does not import Real2SimLab. It is installed in every
runtime image (importable as `r2s_twin`) and handed to agents as a tool.

```python
import r2s_twin

twin = r2s_twin.Twin()                                             # reads R2S_OUTPUT_DIR, R2S_SEED
rng = twin.rng()                                                   # seeded from R2S_SEED
camera = r2s_twin.Camera.look_at((0, -1.6, 0.55), (0, 0, 0.45), width=320, height=240)
twin.write_tracks(fps=30, positions={"ball": xyz}, camera=camera)  # writes and checks
twin.write_video(frames, fps=30)                                   # render.mp4, for video tasks
```

From the command line: `python r2s_twin.py check OUTPUT_DIR` checks an output;
`python r2s_twin.py selftest PROGRAM_DIR` runs a program twice as the evaluation does and
checks that it writes an output, that a tracks output (one with `tracks.npz`, or with both
`meta.json` and `camera.json`) is valid (the format's checks the Lab runs, but for its
optional `render.mp4`: videos, other formats and the task's own requirements are checked by
`r2s twin check`), and that every file but the videos is identical in the two runs.

`r2s twin check TWIN_OR_PROGRAM [--task TASK]` runs the same checks on a frozen twin package (or
a program directory, packaged with the task's convention) in a sandbox.
