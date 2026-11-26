# ![](../assets/icons/fourdcodebench.svg){ .r2s-title-icon } 4DCodeBench

| | |
|---|---|
| Adapter | `fourdcodebench` (`r2s bench build fourdcodebench --out tasks/`) |
| Status | partial: see [Gaps](#gaps) |
| Headline | `dynamic_iou` (0 to 1) from the official `reward.json`; 0 for a world the scorer judges an `error` (as the official aggregates count it) and for a task with no world or no `build.sh` delivered |
| Source | [github.com/4DCodeBench/4DCodeBench](https://github.com/4DCodeBench/4DCodeBench) at commit `6c466be3`; Hugging Face datasets `4DCodeBench/Dataset-Synthetic` (`6b634b2e`) and `4DCodeBench/Dataset-Real-World` (`2e6f267c`) |
| License | none declared, for code or data. Real2SimLab bundles neither: the code is cloned at the pinned commit and the data downloaded at the pinned revisions on first use, into your cache. Task directories are written with `redistributable = false`; do not share them. |
| Citation | Shen, Kovačič, Kulits et al. *4DCodeBench: Benchmarking Agents on Inverse Graphics of Dynamic Scenes.* arXiv:2610.03715 |

An agent watches one video and writes a program that reconstructs the dynamic scene as a 4D
world: a camera, a Blender render, per-frame meshes, and the trajectories of the moving
matter. 4DCodeBench has 100 synthetic cases (with ground-truth worlds) and 100 real-world
cases (with dynamic-object masks).

## Tasks

Each case gives one or two task directories:

| Variant | Protocol | What is scored |
|---|---|---|
| `native` (all cases) | `native`, scorer `fourdcodebench` | **The official protocol**: the official scorer scores the world the agent submitted in its session (`/workspace/world`); `build.sh` is not re-run. |
| `lab` (synthetic cases) | `reproduction` | **The Lab's frozen re-run**: `build.sh` is re-run offline in the task's image and its world is compared with the ground truth by Lab metrics that do not match objects by name: `centroid_set_chamfer` (3D, after a similarity alignment, since only video is given) and `centroid_set_chamfer_2d` (primary; projected, over the image diagonal). |

The native variant can also re-run first (`[scoring.native.options] rerun = true`), which gives
the official score of the *frozen re-run's* world: the reproducibility audit 4DCodeBench's
official protocol does not do.

Defaults: synthetic `B_30`, `B_12`, `G_12` (small worlds) and real
`phys101_01_solid_object_dropped`, `wisa_80k_04_marble_rolling_along`,
`physics_iq_benchmark_07_ball_rolls_down`. Choose others with `--select` (synthetic) and the
`real` option. 23 real-world videos are not redistributed by the dataset and must be rebuilt
from their public sources with its `scripts/prepare_videos.py`; the adapter reports this.

Delivery follows 4DCodeBench's own convention: `solution/build.sh` writes `world/` at fixed
paths (`/workspace/solution`, `/workspace/world`); the task statement is the official
`TASK.md` with its paths filled in, and `world-format.md` and `render-settings.md` are in
`visible/docs/`. Runs need the official agent image (`4dcb/agent:dev`, CUDA 13, Blender 5.2):

```bash
cd ~/.cache/real2simlab/vendor/4DCodeBench@6c466be34107
docker build -t 4dcb/sim-base:dev -f harness/base/Dockerfile .
docker build -t 4dcb/agent:dev    -f harness/agent/Dockerfile .
docker build -t 4dcb/scorer:dev   -f harness/scorer/Dockerfile .
```

The official agent image already contains the Codex and Claude Code CLIs, so agents run in it
with the Lab's sandbox, credentials and egress allowlist (and the task's GPU):

```bash
r2s eval --tasks tasks/synthetic --method codex:image=4dcb/agent:dev --out runs/4dcb
```

`examples/experiments/demo-4dcodebench.toml` runs Codex on three synthetic cases (official
protocol) and one frozen re-run.

## The official scorer

The scorer plugin lays out the four root trees the scorer container expects, writes a
one-entry manifest and runs the official `python -m scorer` in `4dcb/scorer:dev` with a GPU and
no network. Its `reward.json` is stored, unchanged, in the task result's `native` field. Every
world the scorer judges gets a `dynamic_iou` reading there (a number, or `error` and the like). A
world it rejects (it exits 1, with the reason in `reward.json`) is the delivery's result,
`rejected`. A world it judges an `error` (rejected, or a metric failed on it) counts 0 in the
benchmark's mean, the metric's worst value, as in the official aggregates; the readings the
official protocol leaves out (`uncomputable`, `excluded`, `not_applicable`) are left out of it.
A scorer that did not finish judged nothing: an infrastructure error, not a score.
That is a scorer killed (as by the out-of-memory killer); one stopped by an error of its own after
its gate wrote `reward.json` (a worker it could not start, a full disk), so that `reward.json`
holds no `dynamic_iou` reading; and one done without `reward.json`.

The scorer also writes `error` (and exits 1) for a world whose measurement did not finish, and
says why in `reward.detail.json`, whose `causes` and `<name>_failure` texts the result keeps
(`causes`, `failures`). The plugin reads them when `dynamic_iou` is an `error`: its gates'
verdicts (`no_video`, `undecodable_video`, `video_mismatch`, `no_world`, `invalid_world`), an
empty first frame, a non-finite reading and a failed call it does not recognize stay the
delivery's result, counted 0. A failed call (`metric_failure`) that is a measurement that did
not finish is an infrastructure error, redone the next time and never a reward: the scorer's
worker timed out, was lost or never became ready; this machine's disk, quota, processes or
open files ran out, its kernel refused memory (`ENOMEM`), or its GPU failed or is missing in
the container; the scorer image lacks `ffmpeg`, `ffprobe` or `scorer-vdb`; the task's own
case video or reference world could not be read. So is a call that failed while the disk
holding the scorer's output is full, and an `error` the scorer left no readable
`reward.detail.json` for (it writes it with `reward.json`, so it stopped in between). Only the
errors' own type and text count, not a name of the world quoted in them. A `MemoryError` stays
the delivery's result: an allocation fails so only when it asks for more than the machine can
give at all, which the world decides (an archive of a few hundred bytes whose header declares
a 384 TiB array), while a machine short of memory kills the worker, an infrastructure error.

Model checkpoints (about 6.7 GB) come from the official `scripts/download_checkpoints.sh`; the
DINOv3 weights are gated on Hugging Face. Point the plugin at them with
`R2S_4DCB_CHECKPOINTS=/path/to/checkpoints` (or the task option `scorer = {checkpoints = ...}`).
Without a complete set, only the official metric types that need no model are computed:
`dynamic_iou`, `scene_3d`, `trajectory`, `dynamics`, `occupancy`, `interpenetration`, `mesh`.
The result lists the types left out (`uni3d`, `flow`, `track2d`, `semantic`, `geophys`).

## Gaps

- The model-based official metric types need the checkpoints, one of which is gated: they are
  computed only where the checkpoints are installed.
- Alignment with officially published scores is not possible from public data: the
  `4DCodeBench/Results` dataset publishes only the systems' renders, not their worlds or scores.
  What is checked instead (on this machine, and by `tests/test_fourdcodebench.py`): through the
  plugin, the official scorer gives synthetic case `B_30`'s reference world full marks against
  itself — the official "sanity" use of the scorer — on every model-free metric
  (`dynamic_iou`, `scene_3d`, `trajectory_dtw`, `emd_step`, `occupancy_dtw`,
  `interpenetration` = 1.0; the mesh-quality checks describe the reference mesh itself).
- The official world gate requires the program (`solution/build.sh`) next to the world; the
  plugin places the delivered program there, and a delivery without it scores `no_program`.
- The `lab` variant and agent runs need the official agent image, which is large (CUDA 13,
  Blender 5.2, PyTorch; about 30 GB) and is not built by `r2s env build`: 4DCodeBench has no
  license, so the Lab neither ships nor publishes it.
