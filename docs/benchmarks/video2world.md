# ![](../assets/icons/video2world.svg){ .r2s-title-icon } Video2World

| | |
|---|---|
| Adapter | `video2world` (`r2s bench build video2world --out tasks/`) |
| Status | partial: all 37 families (222 instances, 12 evaluators) build and are scored by the official evaluators, but only the Push-T and DROID reference packages have been run end to end; native protocol only; see [Coverage and gaps](#coverage-and-gaps) |
| Headline | `V2WScore` (0 to 100): 100 · S, where S = build · (functionality + geometry + dynamics) / 3; over several tasks, the mean within each family, then over families, as Video2World combines them. A task with no package delivered scores 0 (build 0), as one that does not build. Reference: Video2World's own score of the reference solution it ships (from the metric record shipped with it) |
| Source | [github.com/AetherLabsAI/Video2World](https://github.com/AetherLabsAI/Video2World) at commit `df0e00b7` (code, run at score time); Hugging Face dataset [`AetherLabs-AI/Video2World`](https://huggingface.co/datasets/AetherLabs-AI/Video2World) at revision `9cac0842` (not gated) |
| License | code Apache-2.0. Data: each family keeps its source's terms, recorded in every task's `license` (below). Samples are downloaded into the cache when tasks are built and are not redistributed (`redistributable = false`). |
| Citation | Tang, Zhang, Yang et al. *Video2World: Benchmarking Coding Agents for Interactive World Modeling from Embodied Videos.* [arXiv:2610.04432](https://arxiv.org/abs/2610.04432) |

Video2World gives a method one monocular video of a robot or a human hand manipulating objects
and asks for an executable simulation of it: a **video2sim protocol package** P = (η, π), the
reconstructed scene η (`protocol.json`, meshes) and the behaviour π (`actions.npy`) that
reproduces the outcome. Video2World's evaluator for the instance executes the package in its
simulator and scores the result.

| Evaluator | Families | Instances | Simulator | Scorer image |
|---|---|---|---|---|
| `fb` | Push-T, DROID pick-and-place, 9 FurnitureBench families | 85 | ManiSkill 3 / SAPIEN | `r2s-video2world` |
| `ego_arm`, `ego_bimanual` | DexYCB, HOI4D, HOT3D, OakInk2 hand videos, each replayed by an arm and by two floating hands | 33 + 33 | ManiSkill 3 / SAPIEN | `r2s-video2world` |
| `twin`, `twin_toy` | rope routing, toy packing (reconstructed twins) | 5 + 2 | MuJoCo 3.11 | `r2s-video2world` |
| `cloth` | DROID towel folding | 2 | MuJoCo 3.11 | `r2s-video2world` |
| `robodojo_native_task` | RoboDojo tasks on X5 and xArm7 | 25 | Isaac Sim 5.1 | `r2s-video2world-isaac` |
| `inhouse`, `inhouse_point_robot`, `inhouse_reviewed`, `inhouse_wallet`, `inhouse_fryer` | household episodes of the AgiBot G1 in Genie Sim scenes | 13 + 17 + 4 + 2 + 1 | Isaac Sim 5.1 | `r2s-video2world-isaac` |

## Licenses of the data

The dataset card says Apache-2.0, but its samples are adapted from other datasets, whose terms
still apply. The adapter records each source's own license (see
[Add a benchmark](../tutorials/add-benchmark.md) for the rule):

| Families | Source | License in the tasks |
|---|---|---|
| `push-t`, `rope-route`, `toy-pack` | [real2sim-eval](https://github.com/kywind/real2sim-eval) / [PhysTwin](https://github.com/Jianghanxiao/PhysTwin) (reconstructed twins) | MIT |
| `droid-pick-place`, `droid-cloth-fold` | [DROID](https://droid-dataset.github.io) | CC-BY-4.0 |
| `bulb-insert`, `hood-place`, `cabinet-*`, `chair-*`, `drawer-insert`, `leg-insert`, `round-table-base` | [FurnitureBench](https://clvrai.github.io/furniture-bench) | none declared (its code is MIT; no license found for its data) |
| `dexycb-lift-*` | [DexYCB](https://dex-ycb.github.io) | CC-BY-NC-4.0 (non-commercial) |
| `hoi4d-pickplace-*` | [HOI4D](https://hoi4d.github.io) | CC-BY-NC-4.0 (non-commercial) |
| `hot3d-pickplace-*` | [HOT3D](https://facebookresearch.github.io/hot3d) | CC-BY-SA AND CC-BY-NC-SA (HOT3D Dataset License Agreement: sequences and object models CC BY-SA, hand annotations CC BY-NC-SA) |
| `oakink-rigid-pickplace-*` | [OakInk2](https://oakink.net/v2) | CC-BY-SA-4.0 |
| `robodojo-task-*` | [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) (episodes recorded in its simulation by Video2World) | Apache-2.0 (RoboDojo's assets; its code is MIT) |
| `inhouse-*` | Video2World's own recordings of the AgiBot G1 in [Genie Sim](https://github.com/AgibotTech/genie_sim) scenes | Apache-2.0 AND the AgiBot World Community License Agreement (Genie Sim's assets); integrated with the permission the maintainers obtained |

Three sources are non-commercial (DexYCB, HOI4D, HOT3D's hand annotations): their tasks may be
used for research only. Each task records its source's license, and the scorecard shows it.

The default selection is three Push-T and three DROID instances (see the check against the
official numbers below for how their references fare): `twin_pusht_demo_{0,1,2}`, `droid_guptalab_2023_05_19_11h_06m_39s_orange_hand`,
`droid_iris_2023_03_02_16h_02m_41s_bowl_hand`, `droid_tri_2023_08_09_08h_56m_36s_block_hand`.
`--select` takes instances (`family/sample`), samples (one task per family that has the
sample: the hand samples are in an arm family and a two-hand family) or family names; `all`
selects all 222 instances. Task directories are `<family>/<sample>`.

## Tasks

The agent's workspace holds the video (`visible/cam/rgb.mp4`), the `video2sim` library
(`visible/tools/video2sim`, from the pinned checkout, to validate and replay its package) and a
statement that is Video2World's own brief for the instance's profile, unchanged, after a short
header with the workspace's paths. `solution/` is the package (delivery convention
`video2sim_package`, output format `video2sim_package`).

Some profiles come with public material that Video2World's agent runner stages next to the
video; the adapter stages it with Video2World's own code (`v2w.agents.handouts`) into
`visible/tools/handout/` and appends it to the brief as the runner does:

- RoboDojo and in-house (`robodojo_candidate_task_v1`, `inhouse_provider_native_v2`): the public
  task and robot specification (`public_task.json`), the robot's assets, the task's public rule
  files and `native_contract.py`, the brief's validator;
- reconstructed twins (`twin_rope`, `twin_toy`): the public xArm7 robot contract, which the agent
  copies into its package; the MuJoCo interpreter the brief runs is `/opt/v2w-twin/bin/python`;
- cloth (`droid_cloth`): `visible/tools/cloth_contract/`, only `v2w.tracks.cloth.contract`, the
  brief's contract check, which reads no sample data.

Some files of the reference package are public material too (the twins' robot meshes, the
RoboDojo tasks' rule files). The task keeps one copy, in `visible/`: the hidden data release
lists them in `public_files.json`, and the scorer puts them back before the evaluator runs.

What the agent never sees: the ground truth, the human-assisted reference package (`gt_pkg/`),
the sample's metadata (DROID's task instruction, for example) and Video2World's `v2w` package,
whose metric rules are derived from the ground truth. All of them are public on the Hugging Face
dataset, so agents must not have open network access: the Lab's default allowlist reaches only
the model API. Models trained on data published after October 2026 may have seen them.

Runtime of every task: `r2s-video2world` (Python 3.11, ManiSkill 3.0.1, SAPIEN 3.0.3, torch
2.15.0 CPU build, the ManiSkill `xarm6`, `widowx250s` and `bridge_v2_real2sim` assets, and a
second interpreter at `/opt/v2w-twin/bin/python` with MuJoCo 3.11.0 and NumPy 1.26.4 for the
twin and cloth profiles), built locally and never published (`widowx250s` declares no license).
The physics runs on the CPU, but SAPIEN needs a Vulkan device to load meshes: the ManiSkill
tasks ask for an NVIDIA GPU through the NVIDIA Container Toolkit. Codex gets the same stack
through `r2s-agent-codex-video2world`.

```bash
r2s env build r2s-video2world r2s-agent-codex-video2world
r2s bench build video2world --out tasks/v2w            # the six default instances
r2s bench build video2world --select all --out tasks/v2w-all
r2s eval --tasks tasks/v2w --method codex=codex:image=r2s-agent-codex-video2world --out runs/v2w
```

The RoboDojo and in-house instances are scored in `r2s-video2world-isaac` (below), which you
build once, after reading and accepting NVIDIA's license for Isaac Sim:

```bash
R2S_ACCEPT_NVIDIA_EULA=1 r2s env build r2s-video2world-isaac
```

## Scoring

The scorer plugin `video2world` runs, with no network, Video2World's own per-configuration
evaluation (`v2w.runner.Experiment.evaluate` and `metrics`) on the frozen package and the task's
hidden one-instance data release, then its scoring (`v2w.scoring`) for that instance. The image
depends on the evaluator:

- `r2s-video2world` for the ManiSkill evaluators (with the GPU) and the MuJoCo ones (no GPU);
- `r2s-video2world-isaac` for the Isaac Sim evaluators, with the GPU: CUDA 12.8, Python 3.11,
  PyTorch 2.7.0, Isaac Sim 5.1.0 and RoboDojo's IsaacLab and cuRobo forks, built as RoboDojo's
  own Dockerfile does at the commit Video2World pins (`25691aa`). Isaac Sim is NVIDIA software:
  the image is built only with `R2S_ACCEPT_NVIDIA_EULA=1` (the [NVIDIA Isaac Sim
  license](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html)),
  is never published, and `r2s env build all` skips it.

The dataset directories an evaluator reads besides the sample are downloaded into the cache the
first time an instance of it is scored, at the pinned revision, and never put in tasks: the
robot assets (`assets/maniskill`, a few MB), the hand assets and HOI4D's object models (about
50 MB), and for the Isaac evaluators `assets/robots`, `assets/robodojo`, `assets/inhouse` and
`sources/robodojo`, `sources/inhouse` (several GB, mounted read-only). RoboDojo's own assets
(Hugging Face `RoboDojo-Benchmark/RoboDojo`, `Assets/`, about 41 GB, Apache-2.0) are downloaded
the same way for the RoboDojo instances and configured as RoboDojo's
`utils/update_embodiment_config_path.py` does. Each download resumes if interrupted; a partial
one is never used.

What the evaluator computes:

- `build`: the package loads and its actions run;
- F = 0.5 · task success + 0.5 · progress;
- G and D: the means of each geometry and dynamics metric's quality, max(0, 1 − error / τ);
- S = build · (F + G + D) / 3, and `score` = 100 · S: the instance's V2WScore, the task's headline.

The task result's `native` field holds these, each metric's error, tolerance and quality, the raw
metric record (`video2world_record.json`) and, as `reference`, the same instance score of the
human-assisted reference record that Video2World ships. Video2World's benchmark score, V2WScore,
averages the instances' scores within each family and then over families; the scorecard combines
tasks the same way (the scorer's `aggregate`).

A row Video2World's scoring blocks because its evaluator could not measure the package (the
evaluator crashed, timed out or lost its GPU, its metric code failed, it wrote no metric record
or no build verdict) is never the package's result, as Video2World itself says; nor is a row it
gives neither a score nor an issue (a sample it does not score, `not_applicable`). The task
result is an infrastructure error, left out of the scorecard and redone when the experiment runs
again. A row blocked on what was measured of the package (a measurement that is not finite,
outside its domain or over its cap) stays its result, without a score: the scorecard counts it
among the results but not among the scored ones and leaves it out of the family means (such a
row blocks Video2World's own benchmark score), and a training export gives it a reward of 0.

**Check against the official numbers.** Each sample ships a human-assisted reference
package (`gt_pkg/`) and the metric record of its execution (`har/`). The Lab's scorer, on the
reference package, against Video2World's scoring of the shipped record:

| Sample | S (Lab, released code) | S (shipped record) | F | G (Lab / record) |
|---|---|---|---|---|
| `twin_pusht_demo_0` | 0.991 | 0.990 | 1 / 1 | 0.991 / 0.991 |
| `twin_pusht_demo_1` | 0.990 | 0.989 | 1 / 1 | 0.990 / 0.990 |
| `twin_pusht_demo_2` | 0.991 | 0.990 | 1 / 1 | 0.990 / 0.990 |
| `droid_guptalab_..._orange_hand` | 0.580 | 0.936 | 0 / 1 | 0.989 / 0.989 |
| `droid_iris_..._bowl_hand` | 0.893 | 0.981 | 1 / 1 | 0.990 / 0.990 |
| `droid_tri_..._block_hand` | 0.585 | 0.927 | 0 / 1 | 0.986 / 0.986 |

Push-T reproduces. On DROID the geometry reproduces but the execution does not: with the
released evaluator the reference's gripper closes about 1.7 cm short of the object in two of
the three samples, and the dynamics errors are larger in all three. The shipped DROID records
say why they differ: they were not re-simulated with the released code
(`har_refresh.simulation_rerun` is false) but carried over from an earlier execution of an
earlier package (`har_refresh.original_execution`). `tests/test_video2world.py` checks S on
Push-T and G on every sample. Read DROID scores with this in mind: the released reference
itself scores 0.58 to 0.89 there.

## Use from another benchmark

Every import step is a module-level function in `real2simlab.benchmarks.video2world`:
`manifest()` and `instances()` read the pinned manifest, `instance(name)` and `lookup(name)`
find instances by `family/sample`, sample or family, `EVALUATORS` says how each evaluator
runs, `download(instance)` fetches one sample, `hidden_release()` writes the one-instance data
release the evaluator reads, and `official_brief()` composes Video2World's
brief for a profile. A benchmark that reuses Video2World's clips or adds its own instances can
call them directly; see [Build a benchmark on Real2SimLab](../guides/build-on.md).

## Coverage and gaps

- **What was validated**: the reference packages of Push-T and DROID, executed by the Lab's
  scorer against the shipped records (above). The other 216 instances (the FurnitureBench
  families, the hand, twin and cloth families, and the Isaac Sim families) build into tasks that
  pass the Lab's validation and leak checks, and their briefs match Video2World's runner
  exactly, but their evaluators have not yet been run end to end in the Lab's images. Treat
  their scores as unverified until a reference package of each evaluator has been checked
  against its shipped record, as above.
- **Isaac Sim**: `r2s-video2world-isaac` follows RoboDojo's Dockerfile but has not been built and
  run here. Isaac Sim 5.1 has been reported to crash on RTX 5090 drivers. The image takes tens
  of GB, and RoboDojo's assets add 41 GB to the cache.
- **Protocol**: native only. The package is data, not a program: there is no frozen re-run
  (Video2World's evaluator already executes the actions in the delivered scene).
- **Budget**: Video2World gives agents 180 minutes (and caps API cost); the Lab's default is 60.
  Scores at different budgets are not comparable.
- **Ground truth**: Push-T's object track is itself a PhysTwin simulation registered to the
  first frame, not a measurement (Video2World's grade C).
- **Release defects worked around**: the samples' hidden records name meshes by absolute paths
  on the producer's machine (under its `/root` directory), which some metric code opens without its
  own path mapping; the scorer recreates those paths inside its container as links into the
  data release (`evaluate_package.py`), leaving the data unchanged. The reference packages of
  DROID do not reproduce their records (above).
