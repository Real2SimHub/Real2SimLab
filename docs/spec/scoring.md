# Scoring

Every result is scored by one of two protocols, carries a **headline** in its benchmark's own
metric, and is aggregated into a **scorecard**.

| Protocol | Used by | What is scored |
|---|---|---|
| `native` | Physics-IQ, Video2World, 4DCodeBench, PhysCodeGym, PhySimCode | the benchmark's official scorer, on the delivery or on its frozen re-run |
| `reproduction` (`reproduction/1.0`) | IRIS, the synthetic suite, your own recordings | the Lab's metrics, the frozen re-run's output against `hidden/reference/` |

## Native protocol

`protocol = "native"` names a scorer plugin in `[scoring.native]` (entry point group
`real2simlab.scorers`, class `real2simlab.scoring.NativeScorer`):

- `score(task, delivery_dir=, output_dir=, work_dir=, options=)` runs the official code and
  returns its report, stored **unchanged** in the task result's `native` field. `output_dir` is
  the method's own output, or the frozen re-run's when the task sets `options.rerun = true`.
- `headline(report)` returns the report's [headline](#headlines).
- `aggregate(reports)` (a classmethod, optional) combines several tasks the way the benchmark
  does, when that is not the mean (Physics-IQ pools the metrics of all views before scoring).
- `problems()` says what prevents the scorer from running here (a GPU, weights, an image).
- `task_options` (optional) lists the other names a task's `options` may hold for this scorer,
  and `required_task_options` those every task must give; `r2s task validate` then reports
  any other name (a typo) or a missing one. `options.scorer` is a table of the scorer's own
  arguments, checked against what its class takes.

Each adapter's page says how its scorer was checked against the official pipeline.

## Reproduction protocol

The twin runs frozen and offline; its output is compared with the reference by every metric of
the task. With the primary metric:

- **reproduction error**: the raw value;
- **reproduction skill** = 1 − error / error of a *static scene* (every object held at its first
  position), reported raw and clipped to [0, 1]. It is 0 when the twin was not delivered or its
  run failed. When that ratio is beyond the largest floating-point number (an error that far
  above the static one), the raw skill (`skill_raw`) is null and the skill 0;
- **noise floor**: when the task has other recordings of the event (`hidden/repeats/<k>/`), the
  mean primary-metric error of those recordings against the reference. Each error is also
  reported as a ratio to it (`error_to_noise`, null when the ratio is beyond the largest
  floating-point number), and **`repeat_skill`** is the skill a recording
  of the same event scores: the spread between recordings, on the skill's scale. A twin above
  it reproduces this recording more closely than repeating the experiment does.

## Re-run checks

For every protocol that runs the delivery:

- **`session_consistent`**: an agent's own output in its session (`/workspace/output`) matches
  the frozen re-run.
- **`reproducible`**: two runs with the same seed give the same output
  (`run.check_reproducibility = true`).

Outputs are compared over everything their format holds (`OutputFormat.content`: for
`tracks` the positions, rotations, visibility and camera; for `4d_world` every mesh and point,
not only the centroids its tracks keep; for `video` the video), never through a summary that
could hide a difference. They must have the same files; numbers in `.npz`, `.npy` and `.json`
files agree to 1 µm for the session check and exactly for two re-runs; videos agree frame by
frame up to rendering noise (after averaging 8×8-pixel blocks, no block of any frame may differ
by more than 10 of 255: Blender's sampling noise stays under 6, while a one-pixel shift of a
high-contrast edge exceeds it); any other file must be identical.

The session output is the agent's own, read as it left it. One the Lab may not read as it is
(a symbolic link, or a directory holding one, a pipe, a socket, a device or a file the Lab
cannot read) is not used at all, as if the agent had left none: `session_consistent` is `null`,
and a native task scored on the session output scores no output; a link is never followed. A
file that does not read as its kind (an archive cut short, JSON nested too deep) is compared
byte for byte, and one that cannot be read does not match: the session check never stops the
scoring.

## Headlines

Each task result has a `headline`: the one number its benchmark reports, so results from every
benchmark line up.

| Field | Meaning |
|---|---|
| `metric`, `unit` | the benchmark's headline metric, e.g. `physics_iq_score` in points |
| `value` | its value for this delivery; null when it could not be scored |
| `higher_is_better`, `range` | how to read it |
| `references` | the same metric for reference points on this task: `static` (a scene where nothing moves), `another recording` (a second recording of the event), `reference solution` (the benchmark's own solution) |
| `source`, `scorer` | `lab` (reproduction skill) or `official` (a native scorer, named) |

| Benchmark | Headline | References |
|---|---|---|
| Synthetic, IRIS, your recordings | `reproduction_skill` (0 to 1) | static 0; another recording, when the task has repeats |
| Physics-IQ | `physics_iq_score` (0 to 100), pooled over views | static (the last frame held), another recording (the second real take) |
| Video2World | `V2WScore` (0 to 100: 100 · build · mean of functionality, geometry and dynamics), the mean per family, then over families | reference solution |
| 4DCodeBench | `dynamic_iou` (0 to 1) | — |
| PhysCodeGym | `S_mot` (motion-mask IoU, 0 to 1) | — |
| PhySimCode | `parameters_within_20pct` (0 to 1: the share of all true parameters recovered within 20 %, PhySimCode's end-to-end recovery), pooled over tasks | — |

## Scorecard and aggregation

Per method, the results (`r2s-results/1.0`) hold the scorecard: one entry per benchmark
(`BenchmarkScore`) with its headline metric over the method's tasks, combined the benchmark's
way (its scorer's `aggregate`, otherwise the mean), the mean of each reference point, how many
results could be scored, and the share of deliveries whose re-run was session-consistent and
reproducible, with the counts behind it (`rerun_compared`, `rerun_agreed`: a delivery whose
first run failed has no re-run to compare and is not counted as agreeing). Tasks that share a
benchmark name but report different headline metrics (custom tasks) get one entry per metric;
none is dropped. Each method also gets its mean reproduction skill over reproduction tasks and
its delivered rate.

**A failure of the machine is not a score.** When the Lab cannot score a result for a reason
outside the method (the sandbox of the build or of the frozen re-run did not start or was
killed from outside it, the frozen twin was changed from outside its sandbox while it ran,
this machine's storage stopped the program's writes (a file at the file-size limit, a full
disk), the runtime lacks what every Lab runtime holds (NumPy, SciPy, the twin SDK, ffmpeg),
this machine could not copy or package the delivery, the benchmark's official scorer failed
rather than scoring the delivery), the task result records why in `infra_error` and has no
reproduction skill and no headline value. Every statistic, scorecard entry, rate and reward
leaves it out; the scorecard counts it in `results` and `infra_errors` (shown as "not scored:
infrastructure"), and running the same experiment again redoes exactly those steps. A
delivery the official scorer did score, even with 0, is a score. A second run for the
reproducibility check that did not start is not compared: only two runs that both ran can
agree or disagree.

**Only what the Lab can tell apart is outside the method.** A frozen re-run's sandbox was
killed from outside when its own process died of a signal: bwrap (outside a systemd user
scope) or the Docker client. Its twin was changed from outside when it changed while it ran on
`docker` or `bwrap`, which mount it read-only. This machine's storage stopped its program when
a file the Lab finds after the run (the log, the output or the scratch tree of a replaceable
output, the `local` backend's temporary directory) reached the file-size limit, or the disk
holding them is full, also when the program caught the failed write or retried it until its
time limit. What looks the same as the program's own failure stays its failure: a signal to
the program itself (inside its sandbox, `docker kill`, any kill on `local`) or to bwrap inside
a systemd scope (where its program running out of the scope's memory can end bwrap too), a
change of the twin on `local` (where its program could have made it), and a write stopped in
the sandbox's own `/tmp` (a tmpfs on `docker` and `bwrap`, which also holds `HOME`) or to an
anonymous temporary file, which leaves no file to find.

**A task a method failed is a score.** A method that delivers nothing for a task, or whose
program writes no valid output, gets the benchmark's lowest result for it, never a gap: on the
Lab's protocols a reproduction skill of 0; on a native protocol the official scorer's result
for an empty delivery with no output (`NativeScorer.score` is called with an empty
`delivery_dir` and `output_dir = None`). That is S_mot 0 on PhysCodeGym, `dynamic_iou` 0 on
4DCodeBench, build 0 (so V2WScore 0) on Video2World, the task's true parameters counted with
none recovered on PhySimCode, and on Physics-IQ the worst value of every metric against the
real continuation (the three IoUs 0, MSE 1, the largest possible on frames scaled to [0, 1]),
with the view's physical variance from the official code, so the view counts in the pooled
score. A video that shows nothing is not used for this: Physics-IQ counts an empty motion mask
against an empty one as a match, so a black video scores well on scenes that barely move. So a
method cannot raise its score by failing the hard tasks.

A method is scored only on tasks it can deliver for. The privileged baselines write `r2s`
programs from the reference trajectories (`static`, `constant_velocity`) or copy the task's
truth generator (`oracle`); on a task with another delivery convention, or without a reference
in tracks form or a generator, they are reported as not applicable, not as scoring zero.

Intervals are **95 % confidence intervals from a two-level bootstrap**: resample tasks with
replacement, then each sampled task's results (trials and seeds), 2000 times, with a fixed seed
so the same results always give the same intervals. For benchmarks with their own aggregate,
the same resampling is applied to that statistic. Privileged methods (which read hidden data)
and agents built with an open network are labelled in every table.

## Metrics

Metrics declare the formats they read (`tracks` covers every format that converts to tracks).
Lower is better. Metrics are plugins (`real2simlab.metrics`).

| Metric | Unit | Definition |
|---|---|---|
| `traj3d_mean` | m | mean 3D distance between predicted and reference centroids, over objects and frames where the reference is visible |
| `traj3d_rmse` | m | root-mean-square of the same distances |
| `traj3d_final` | m | distance at each object's last reference-visible frame (endpoint error) |
| `traj2d_px_mean` | px | mean image distance between centroids, each projected with its own camera, in reference pixels; a moving camera is taken at the same instant as the centroid, whatever the prediction's frame rate |
| `traj2d_norm_mean` | 1 | `traj2d_px_mean` / reference image diagonal |
| `event_first_contact` | s | error of the first contact (first bounce) time |
| `event_bounce_count` | 1 | difference in the number of bounces |
| `event_bounce_times` | s | mean error of each reference bounce time, matched in order |
| `centroid_set_chamfer`, `centroid_set_chamfer_2d` | m, 1 | object-name-free set distances for scene reconstructions (4DCodeBench's frozen re-run); a frame where the reference has objects and the prediction none (or, in 2D, none in front of its camera) makes the metric an error, with the coverage in its detail |

3D metrics accept `align = "translation"` or `"similarity"` (Umeyama) for methods that only
see video and cannot know the world frame or scale. A bounce is a local minimum of height with
vertical speed above 0.05 m/s before and after (`min_speed`). These are the only options
(`scoring.metric_options`), and their values are checked: an option a metric does not take,
or a value it does not know (`align = "similiarity"`), makes the task invalid
(`r2s task validate`), and it is not scored. A missing predicted event counts
as the time left until the end of the clip. A prediction that misses an object, or does not
cover the reference's time span, is an error and scores as a failure. Every computed value is a
finite number: a huge but finite distance (a diverging solver) is computed as such, without
overflowing, after an alignment too, and a comparison whose value is beyond the largest
floating-point number, or whose alignment cannot be computed, is an error (a failure of that
prediction), never `NaN` or `Infinity` in a record and never the end of an experiment.
