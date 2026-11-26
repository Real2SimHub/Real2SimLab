# Build a benchmark on Real2SimLab

Real2SimLab is the evaluation platform; a benchmark built on it owns the content: which data, which tasks,
which methods, which analysis. This page shows how a benchmark repository embeds the Lab,
assembles tasks from its own data and from other benchmarks' data, and runs evaluations.

## 1. Add the Lab as a submodule

```bash
git submodule add https://github.com/Real2SimHub/Real2SimLab.git third_party/Real2SimLab
git -C third_party/Real2SimLab checkout v1.0.0          # pin a tag
pip install -e third_party/Real2SimLab                   # Python >= 3.11
r2s doctor                                               # Docker, images, ffmpeg
r2s env build r2s-python                                 # the twin runtime image
```

Pin a tag and upgrade deliberately: task files carry `spec = "r2s-task/1.0"`, and a newer Lab
reads them unchanged within the major version.

## 2. Assemble tasks

Write one small module per data source that reads its data and calls the Lab's authoring API,
the same pattern as the Lab's own adapters:

```python
# mybench/sources/lab_recordings.py
from real2simlab.capture.from_video import capture_from_video
from real2simlab.task import TaskBuilder

def import_episode(raw_dir, out_dir) -> Path:
    capture = capture_from_video(raw_dir / "rgb.mp4", work / "capture", capture_id=..., intrinsics=...)
    builder = TaskBuilder(out_dir, config_for(raw_dir))
    builder.capture(capture, work / "capture", visibility=...)
    builder.reference(tracks=read_annotations(raw_dir / "trajectory"))
    for k, take in enumerate(other_takes(raw_dir), start=1):
        builder.repeat(k, tracks=take)
    return builder.finish().root
```

The Lab's benchmark adapters expose their import steps as plain functions you can reuse:

| Adapter | Reusable functions |
|---|---|
| `synthetic` | `full_capture`, `task_config`, `tracks_of`, `render_video`, `generator_twin`; the truth generators in `real2simlab.benchmarks.synthetic.physics` |
| `iris` | `download_video`, `extract_drops`, `observations`, `tracks_of`, `visible_capture`, `camera`, `task_config`, `build_tasks` |
| `fourdcodebench` | `download_synthetic_case`, `download_real_case`, `case_capture`, `task_config`, `build_case` |
| `physcodegym` | `case_names`, `case_record`, `download_reference`, `statement`, `task_config`, `build_case` |
| `physicsiq` | `case_index`, `case_files`, `download`, `case_capture`, `statement`, `task_config`, `build_case` |
| `video2world` | `manifest`, `instances`, `instance`, `lookup`, `EVALUATORS`, `download`, `hidden_release`, `shared_data`, `stage_handout`, `official_brief`, `statement`, `task_config`, `build_instance` |
| general | `capture_from_video`, `TaskBuilder`, `write_tracks`, `VisibilityPolicy`, `validate_task` |

Video2World's adapter, for example, reads the pinned manifest, fetches one sample and writes the
one-instance data release its evaluator reads, so your benchmark can build its own selection
(its page lists which families the evaluator image supports).

Tasks in one benchmark may be heterogeneous: different environments (`r2s-python`, a Blender
image, the 4DCodeBench image, the Video2World image), output formats (`tracks`, `4d_world`,
`video`, `video2sim_package`), protocols and metrics. Every task records its `source` and
`license`; set `redistributable = false` for data you may not share, and ship a download script
instead. If your benchmark has its own official scorer, register it as a scorer plugin with a
`headline` ([Add a benchmark](../tutorials/add-benchmark.md)), so its results join the scorecard.

See [Write tasks from your recordings](../tutorials/write-tasks.md) for the full walkthrough and
the example task.

## 3. Validate in CI

```bash
r2s task validate mybench/tasks/        # exit status 1 on any error
```

## 4. Run evaluations

Describe an experiment in TOML, saved as `mybench/experiments/pilot.toml`:

```toml
spec = "r2s-experiment/1.0"
name = "mybench-pilot"

[tasks]
paths = ["../tasks"]                    # relative to this file: mybench/tasks

[[methods]]
name = "static"                         # a privileged reference baseline

[[methods]]
name = "codex"                          # a coding agent
reasoning_effort = "medium"

[run]
backend = "docker"
seeds = [0]
trials = 1
agent_concurrency = 2
check_reproducibility = true

[budget]
max_agent_runs = 10
```

```bash
r2s run mybench/experiments/pilot.toml --out runs/pilot     # rerun the same command to resume
r2s report runs/pilot                                        # runs/pilot/report/index.html
```

`runs/pilot/results.json` (`r2s-results/1.0`, with the scorecard) and the per-task results
(`results/<task>/<method>/trial-<k>/seed-<s>.json`, `r2s-task-result/1.0`) are the stable
interface for your own analysis; their JSON Schemas are in `docs/spec/schemas/`.

## 5. What stays in the Lab, what goes in your benchmark

| Lab | Your benchmark |
|---|---|
| specification, validation | which data, which tasks |
| sandboxes, runtime images, agent adapters | which methods and agents to compare |
| scoring protocols, metrics, headlines, statistics | leaderboards, analysis, papers |
| adapters for public benchmarks | assembling them into your suite |

If you need a new metric, output format, delivery convention, scorer or backend, add it as a
plugin (entry points) in your own package, or propose it upstream; the Lab's core does not
change for it.
