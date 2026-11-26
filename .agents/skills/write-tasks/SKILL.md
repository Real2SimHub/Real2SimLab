---
name: write-tasks
description: Turn a recording (or another benchmark's data) into a Real2SimLab task directory that validates, with its reference, repeats and baselines checked. Use when asked to write, add or convert tasks for Real2SimLab.
---

# Write a Real2SimLab task

A task is a directory: `task.toml`, `visible/` (what a method sees) and `hidden/` (the
reference, repeats, anything scoring needs; never shown to a method). Build it with
`TaskBuilder`, never by hand-writing JSON. Full tutorial: `docs/tutorials/write-tasks.md`;
runnable example: `examples/make_puck_task.py` (it builds `examples/tasks/puck_slide`).

## Procedure

1. **Read the license** of the data first. Record `license`, `source` and an honest
   `redistributable`. Data that may not be redistributed is downloaded by a script at build
   time, never committed.
2. **Decide what a method sees**: the video, and which of the camera calibration, metric
   scale, the scene's objects and the text are visible (a `VisibilityPolicy`). Mark every
   quantity's provenance (`measured`, `estimated`, `default`, ...).
3. **Decide the protocol**: `reproduction` when you have the true outcome of the recorded event
   (the reference); `native` when the source benchmark has its own scorer (then see the
   `add-benchmark` skill).
4. **Build it with `TaskBuilder`**:
   - `TaskConfig`: id, title, description, source, license, delivery (environment, output
     format, convention, fps / frames / objects), scoring (metrics, primary metric).
   - `builder.capture(capture, media_dir, visibility=...)`, then
     `builder.reference(tracks=...)` (or a directory in the task's output format) and
     `builder.repeat(k, tracks=...)` for other recordings of the same event (the noise floor),
     then `builder.finish()`.
   - The reference uses the task's output format: for `tracks`, one centroid track per object
     in metres and the camera.
5. **Validate**: `r2s task validate PATH`. Fix every error and warning; never suppress:
   `hidden-leak` (hidden content copied into `visible/`), `missing-file`, `reference-invalid`.
6. **Read the statement** an agent will get: `r2s task show PATH --statement`. It must describe
   the event and the delivery and must not reveal the reference.
7. **Check with baselines**:
   `r2s eval --tasks PATH --method static --method constant_velocity --backend docker --out runs/check`.
   `static` has reproduction skill 0. If you can, write a twin with the SDK (see the
   `write-twin` tutorial) and evaluate it with `--method mine=program:path=DIR`: it should beat
   both baselines; the repeats' skill ("another recording") is the spread between runs of the
   same experiment, for comparison.
8. **Add a test** that builds the task (skipped when the source data are absent) and asserts
   `validate_task(root).ok` with no warnings.

## Details that are easy to miss

- **Repeats are the same setup recorded again**, not different setups: their spread is the
  noise floor.
- **The SDK** for a twin is `real2simlab/twin/sdk/r2s_twin.py` in the repository (agents get it
  as `tools/r2s_twin.py`); copy it next to `run.py` and run `python r2s_twin.py selftest DIR`.
- **Commit the inputs** of a task built from small, redistributable data (as
  `examples/raw/puck_slide/` is), so the build script and its test run anywhere; otherwise skip
  the test when the data are absent.
- **Deterministic builds**: the same inputs must give byte-identical task directories (no
  timestamps, temporary paths or unseeded randomness).
- **Training tasks** are the same, with `split=Split.TRAIN` in their `TaskConfig` and a
  benchmark name of their own (not a registered benchmark's). Before training on them, export
  them with `r2s train export --tasks DIR --against EVAL_TASKS`: it leaves out every task that
  holds the same file or the same recording as an evaluation task, and says which in
  `export.json`.

## Do not

- Put anything from `hidden/` (or computed from it) into `visible/` or the statement.
- Change units (degrees, cm): the specification uses metres, seconds, radians, kilograms.
- Commit data whose license forbids redistribution.
