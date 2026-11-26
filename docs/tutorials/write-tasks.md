# Write tasks from your recordings

This tutorial turns a recording of a physical event into a Real2SimLab task, validates it, and
evaluates programs on it. It is for benchmark authors who bring their own data, or data from
other benchmarks.

A complete, runnable version of everything below is
[`examples/make_puck_task.py`](https://github.com/Real2SimHub/Real2SimLab/blob/main/examples/make_puck_task.py),
which builds the example task `examples/tasks/puck_slide` from the raw recording in
`examples/raw/puck_slide/` (a video, its calibrated camera, an experimenter's measurements, the
puck's measured trajectory and three repeated launches).

## 1. Decide what the task asks

A task answers three questions:

1. **What does a method see?** A capture: the video, and whatever you know about the cameras,
   the scene and its objects. You decide which of it is visible.
2. **What is the truth?** The *reference*: the true output of the recorded event in the task's
   output format (for `tracks`, each object's trajectory and the camera), and, if you have them,
   other recordings of the same event (the noise floor).
3. **How is it scored?** The protocol (`reproduction` with the Lab's metrics, or `native` with
   your benchmark's own scorer plugin), the metrics and the primary metric.

## 2. Start from a skeleton (optional)

```bash
r2s task new tasks/drop_01 --id mybench/drop_01 --title "Drop a ball" --video drop.mp4
```

This writes `task.toml` with `TODO`s and a capture of the video in `visible/`. Fill them in by
hand, or generate everything with the Python API as below, which is what an adapter should do.

## 3. Build the task in Python

```python
from pathlib import Path
from real2simlab.capture.from_video import capture_from_video
from real2simlab.capture.visibility import VisibilityPolicy
from real2simlab.task import (
    Delivery, OutputRequirements, Protocol, ScoringConfig, SourceInfo, TaskBuilder, TaskConfig,
)

# 3.1 The observation. Measured intrinsics/extrinsics make the capture metric.
capture = capture_from_video(
    Path("raw/observed.mp4"), Path("work/capture"),
    capture_id="mybench/puck_slide",
    intrinsics=K, extrinsics=camera_to_world,
    object_sizes={"puck": 0.08},
    scene_description="A red puck slides across a flat grey floor.",
)

# 3.2 The task: source and license, delivery, scoring.
config = TaskConfig(
    id="mybench/puck_slide", version="1.0.0", title="Puck launched across a floor",
    source=SourceInfo(dataset="MyBench", version="2.1", url="https://..."),
    license="CC-BY-4.0", redistributable=True,
    delivery=Delivery(
        environment="r2s-python", output_format="tracks",
        output=OutputRequirements(fps=30, num_frames=60, objects=("puck",)),
    ),
    scoring=ScoringConfig(
        protocol=Protocol.REPRODUCTION,
        metrics=("traj3d_mean", "traj3d_final", "traj2d_norm_mean"), primary="traj3d_mean",
    ),
)

# 3.3 Write the directory.
builder = TaskBuilder(Path("tasks/puck_slide"), config)
builder.capture(capture, Path("work/capture"), visibility=VisibilityPolicy(
    name="calibrated", expose=frozenset({"views.*.intrinsics", "views.*.extrinsics", "scene", "text"})))
builder.reference(tracks=measured_tracks)          # a Tracks object, or a directory
for k, take in enumerate(other_takes, start=1):    # optional: the same event, recorded again
    builder.repeat(k, tracks=take)
task = builder.finish()                            # writes task.toml and validates
```

Things to get right:

- **The reference uses the task's output format**, here `tracks`: one centroid track per object
  in metres, plus the camera (`real2simlab.formats.tracks.Tracks`, `write_tracks`). References
  of real recordings come from your annotations or a tracker; mark estimated quantities as such
  in the capture.
- **Repeats are recordings of the same setup**, not of different ones: their spread is the noise
  floor, and the skill they score is the reference point "another recording" in the scorecard.
- **Hidden means hidden.** Never copy the reference, a repeat or anything derived from them into
  `visible/` or `instruction.md`; the validator compares file contents and refuses leaks, also
  a copy that `capture.json` lists as a stream (masks, depth). To show only part of a
  video, use `VisibilityPolicy(frame_window=(0, K))` *and* cut the video file to K frames.
- **License and redistribution.** Set `license` to the data's license and `redistributable`
  honestly. Data that may not be redistributed must be fetched by a download step, never
  committed.

## 4. Validate

```bash
r2s task validate tasks/
r2s task show tasks/puck_slide              # metadata
r2s task show tasks/puck_slide --statement  # the full statement an agent receives
```

`validate` exits with status 1 on errors. The ones authors meet most: `hidden-leak` (hidden
content in `visible/`), `missing-file` (the reference, a media file) and `reference-invalid`
(a reference or a repeat that does not follow the output format or the task's requirements).

## 5. Check the task with baselines and a twin

```bash
r2s eval --tasks tasks/ --method static --method constant_velocity --backend docker --out runs/check
```

`static` must have reproduction skill 0. If you have a hand-written twin, evaluate it with the
`program` method:

```bash
r2s eval --tasks tasks/ --method mine=program:path=twins/ --out runs/check
```

`twins/` holds one program per task, named after the task id (`twins/mybench/puck_slide/`, or
`twins/mybench__puck_slide/`), or is itself a single program used for every task.

`examples/twins/puck_slide/` is such a twin for the example task: it guesses the floor's
friction, so its skill is high but below what another recording of the event scores, the
spread between repeated launches.

## 6. Share it

Commit the task directory (if redistributable) or the script that builds it. Every task records
its source and license, and tasks in one benchmark may differ in environment, output format and
metrics. To make a whole benchmark of them, write an adapter: [Add a benchmark](add-benchmark.md).
