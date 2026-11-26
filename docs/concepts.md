# Concepts

## Task

A **task** is a directory with `task.toml`, `visible/` and `hidden/`.

- `visible/` is everything a method may use: a **capture** (`capture.json`: the video, and, if
  the task allows, calibrated cameras, metric scale, the scene's objects, text) plus documents
  and tools.
- `hidden/` is never mounted into a method's sandbox: the **reference** (the true output of the
  recorded event, in the task's output format), other **recordings** of the same event (the
  noise floor), and anything the benchmark's official scorer needs.
- `task.toml` says what to deliver (runtime image, output format, delivery convention, time
  and resource limits), where the data comes from and under which license, and how to score
  (protocol, metrics, or the official scorer).

Adapters build tasks from a benchmark's own data (`r2s bench build NAME`); you can also write
tasks from your recordings. Details: [Tasks](spec/task.md), [Captures](spec/capture.md).

## Method and twin

A **method** builds a program from a task's visible data: a coding agent, a Real2Sim pipeline,
a hand-written program, or a reference baseline. Agents work in a sandboxed workspace and hand
in `solution/`; the Lab packages it as a **twin** (`twin.json` plus `program/`), written by the
Lab, never by the method. A method that hands in nothing valid scores zero, with the reason
recorded.

A twin follows a **delivery convention**: the Lab's own (`solution/run.py`, run under the
[run contract](spec/twin.md#run-contract)), or a benchmark's (4DCodeBench's `build.sh`,
PhysCodeGym's Blender `scene.py`, Video2World's protocol package).

## Frozen re-run

The twin is frozen (content-fingerprinted) and run offline: its program directory read-only,
no network, an empty output directory, a seed. Same seed, same output. What is scored is this
re-run's output, not what the method showed in its session; the Lab records whether the two
agree (`session_consistent`) and, when asked, whether two re-runs agree (`reproducible`).

## Scores

Two protocols:

- **native**: the benchmark's official scorer, as a plugin, scores the delivery (or its frozen
  re-run). Its report is stored unchanged.
- **reproduction**: the Lab's metrics compare the re-run's output with the reference. The
  reproduction skill is 1 − error / error of a static scene (nothing moves): 1 is perfect, 0 is
  no better than nothing moving. Where a task has other recordings of the event, their error is
  the noise floor, and the skill such a recording scores says how far apart two runs of the same
  experiment are.

Every result carries a **headline**: its benchmark's headline metric (the Physics-IQ score,
Video2World's V2WScore, the reproduction skill...) with its direction, range and reference points.
The **scorecard** puts methods against benchmarks, each column combined the benchmark's own
way, with bootstrap confidence intervals and the share of deliveries the re-run reproduced.
Details: [Scoring](spec/scoring.md).

## Experiment

An **experiment** (a TOML file, or `r2s eval` options) names tasks or benchmarks, methods,
seeds, trials, a backend, concurrency and budgets. Its run directory keeps every build, run,
record and result under a fixed layout keyed by content hashes, so it resumes after
interruptions and gives the same results. Details: [Records](spec/records.md).
