# Tasks (`r2s-task/1.0`)

A task is a directory. Everything a method may see is in `visible/`; everything only the
evaluator may see is in `hidden/`, which is never mounted into a method's or agent's sandbox.

```
<task>/
  task.toml        metadata, delivery, scoring, source and license
  instruction.md   optional: replaces the generated task statement
  visible/         capture.json and the media it references; optional docs/ and tools/
  hidden/
    reference/     the true output of the recorded event, in the task's output format
    repeats/<k>/   optional: other recordings of the same event (001, 002, ...): the noise floor
    generator/     optional: a twin package that generates the truth (synthetic tasks)
    parameters.json  optional: the hidden parameters by name, a JSON object (training labels)
    ...            anything else scoring needs (e.g. what an official scorer reads)
```

## `task.toml`

```toml
spec = "r2s-task/1.0"
id = "synthetic/bouncing_ball/calibrated"    # lowercase path segments of [a-z0-9_.-]
version = "1.0.0"
title = "Bouncing ball (calibrated)"
description = "One paragraph a method reads first."
license = "Apache-2.0"                         # of the task's data
redistributable = true                         # false: the data must be fetched, not shared
tags = ["synthetic"]
visibility = "calibrated"                      # the visibility tier, informative
split = "eval"                                 # eval (default) | train: a training task

[source]                                       # where the data comes from
dataset = "real2simlab-synthetic"
version = "1.0"
url = "https://..."
citation = "..."                               # optional

[delivery]
environment = "r2s-python"       # a registered environment or any image reference
output_format = "tracks"         # tracks | 4d_world | video | video2sim_package | a plugin
convention = "r2s"               # r2s (solution/run.py) | build_sh_world (4DCodeBench) | blender_scene (PhysCodeGym) | video2sim_package (Video2World)
agent_time_budget_s = 1200
run_timeout_s = 120              # one frozen run

[delivery.resources]
gpu = false
cpus = 2.0
memory_mb = 4096

[delivery.program]               # rules for the delivered program/
max_size_mb = 50.0
allow_video = false              # no video files: a program may not copy the recording
forbidden_suffixes = []

[delivery.output]                # what the output must contain, told to the method
fps = 30.0
num_frames = 60
objects = ["ball"]

[scoring]
protocol = "reproduction"        # reproduction | native
metrics = ["traj3d_mean", "traj3d_rmse", "traj3d_final", "traj2d_norm_mean"]
primary = "traj3d_mean"          # the metric the reproduction skill is computed from

[scoring.metric_options.traj3d_mean]
align = "none"                   # none | translation | similarity; checked by r2s task validate

[scoring.native]                 # for protocol = "native"
scorer = "physics_iq"            # a registered scorer plugin
options = {}                     # rerun = true, scorer = { ... } (its arguments), the scorer's own

[metadata]                       # free-form, for the adapter; never shown to methods
```

The task's benchmark is the first segment of its id (`synthetic` above); the scorecard groups
results by it. Tasks of the same benchmark may differ in environment, output format and
metrics.

`split` says what a task is for. An evaluation task (`eval`) is never exported as training
data unless asked (`r2s train export --allow-eval-tasks`, and then the dataset says so). A
training task (`train`) is exported; it is evaluated like any other task, but its results are
training results. A task of a registered benchmark is a training task only if the benchmark
releases a training split (its adapter's `splits`; see [Training](../guides/training.md)).

## Validation

`r2s task validate` (and `real2simlab.task.validate_task`) reports errors and warnings:

| Code | Error when |
|---|---|
| `spec` | `task.toml`, the capture or a generator violates the specification (a number that is not finite, such as `cpus = inf`, included) |
| `hidden-leak` | a file in `visible/` (whatever `capture.json` lists it as) or the task's `instruction.md` (its statement) has the same content as a file in `hidden/`, or a `reference/`, `repeats/` or `generator/` directory appears in `visible/`. Only the observation itself is exempt: a hidden video identical to a view's `rgb` video (which an official scorer may read). Whole files are compared: hidden values written into a statement are not found |
| `missing-file` | `visible/capture.json`, its media, the reference, or a convention's required `instruction.md` is missing |
| `not-self-contained` | the task contains a symbolic link |
| `reference-invalid` | the reference or a repeat violates the output format or the task's output requirements |
| `scoring` | an unknown metric or scorer, or a metric that cannot read the output format |

A task's `hidden/parameters.json`, when it has one, holds the hidden parameters by name (a
synthetic task's truth), which `r2s train export` puts in the task's row as its labels: one that
cannot be read, is not JSON or is not a JSON object is a `spec` error.

Native tasks have no Lab reference: their official scorer reads what it needs from `hidden/`.

## The task statement

The statement given to agents is generated from a versioned template (`r2s-instruction/1.0`,
recorded with every agent run): the event, the workspace, what to deliver, the run contract and
the rules. A task's `instruction.md` replaces it (`{workspace}` is substituted); benchmark
adapters use it to give an agent the benchmark's own statement. Like `visible/`, it is checked
for hidden data (`hidden-leak`), and a build gives a method the `instruction.md` that was
validated.

## Building tasks in Python

```python
from real2simlab.task import TaskBuilder

builder = TaskBuilder(out_dir, config)
builder.capture(full_capture, media_dir, visibility=CALIBRATED)   # copies only visible media
builder.reference(tracks=true_tracks)                             # Tracks or a directory
builder.repeat(1, tracks=second_take)                             # optional, 1, 2, ...
builder.hidden_file("official/labels.json", source=Path("labels.json"))  # what a scorer needs
task = builder.finish()                                           # writes task.toml, validates
```

Until `finish` writes `task.toml`, the directory holds the build's mark (`.r2s-task-build`), so a
build that stopped (Ctrl-C, a crash) is replaced by the next one given `overwrite=True`
(`--overwrite`). `overwrite` replaces only such a directory, a task (its `task.toml` has
`spec = "r2s-task/..."`) or an empty directory, never one that merely holds a task's names.
