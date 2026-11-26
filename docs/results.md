# Example run: one agent on every benchmark

What a complete evaluation with Real2SimLab produces: one coding agent run on all six
benchmarks and the synthetic suite, every delivery frozen, re-run and scored in its benchmark's
own terms, and the records combined into one scorecard. It shows the platform working end to end; it is not a
benchmark result, a leaderboard, or a comparison of agents (one run per task, a few tasks per
benchmark).

**Setup.**

| | |
|---|---|
| Agent | Codex CLI 0.167.2, in the image matching each task's runtime (`r2s-agent-codex`, `-blender`, `-physimcode`, `-video2world`); 0.168.0 as shipped in 4DCodeBench's own image (4DCodeBench) |
| Model | `gpt-6-astra`, called through an OpenAI-compatible endpoint; reasoning effort medium |
| Time budget | 20 minutes per task on synthetic, IRIS, Physics-IQ and 4DCodeBench; 30 on PhySimCode; 60 on PhysCodeGym and Video2World |
| Network | the model endpoint only, through the egress proxy |
| Runs | one per task; every delivery frozen and re-run offline, twice, and compared with the agent's own output |
| Code | synthetic, IRIS, PhySimCode and PhysCodeGym: this release, in one run. Physics-IQ, Video2World and 4DCodeBench: runs made with development versions of 1.0, with the same official scorers and protocols; their headlines are read from the stored official reports by this release's code. |

## Scorecard

| Benchmark | Headline metric | Codex | Reference points | Tasks | Re-run reproduced |
|---|---|---|---|---|---|
| Synthetic | reproduction skill (0–1) | **0.999** | static 0 | 1 | 1/1 |
| IRIS | reproduction skill (0–1) | **0.983** [0.966, 0.994] | static 0; another recording 0.956 | 3 | 3/3 |
| Physics-IQ | Physics-IQ score (0–100) | **52.2** [23.7, 83.6] | static 38.4; second take 100 | 6 views | 6/6 |
| Video2World | V2WScore (0–100) | **17.5** [17.1, 17.9] | reference solution 98.5 | 2 | — (a data package, executed by the evaluator) |
| 4DCodeBench | dynamic IoU (0–1) | **0.547** [0.250, 0.978] | — | 3 | — (scored on the session's world) |
| PhysCodeGym | motion IoU (0–1) | **0.472** | — | 1 | 1/1 |
| PhySimCode | parameters within 20 % (0–1) | **0.300** [0, 0.500] | — | 3 | 3/3 |

Intervals are 95 % bootstrap intervals over tasks (none with one task); with two or three
tasks they mostly show how few tasks there are. Each benchmark is combined its own way: the Physics-IQ score is computed over the six views together, as Physics-IQ
does over all of its views. *Re-run reproduced* counts deliveries whose frozen, offline re-run
gave the output the agent itself produced, and the same output on a second re-run.

How to read it: Codex simulates simple rigid-body events from video very well (synthetic, IRIS,
where its trajectories are closer to the recording than another take of the same drop is), is
mixed on scene-level video prediction (Physics-IQ: above the static reference, far from the second
take), and is weak where the benchmark asks for metric 3D scenes and actions (Video2World) or for
the physical parameters themselves (PhySimCode).

## Synthetic: bouncing ball

Codex's program reproduces the exact truth with a reproduction skill of 0.999: a mean 3D error
of 0.25 mm and the first contact at the right frame; only the count of the tiny last bounces,
when the ball is nearly at rest, is off by one. 119 s, 0.20M input tokens.

## IRIS: three real drops

| Task | Codex | Hand-written twin | Constant velocity | Static | Another recording | Codex error / noise floor |
|---|---|---|---|---|---|---|
| `drop_050` (0.61 m) | **0.966** | 0.952 | 0.312 | 0 | 0.922 | 0.44 |
| `drop_100` (1.11 m) | **0.990** | 0.973 | 0.192 | 0 | 0.972 | 0.35 |
| `drop_150` (1.61 m) | **0.994** | 0.978 | 0 | 0 | 0.973 | 0.23 |

Reproduction skill against take 01's extracted trajectory. *Another recording* is the skill the
other takes of the same height score: the spread between repetitions of the experiment. Codex's
error is a quarter to a half of that spread: its program, fitted to the take it sees, reproduces
it more closely than repeating the experiment does. The hand-written twin
(`examples/twins/iris/`, a rigid ball with a constant restitution read off the clip) is close
behind. Each run took about 2.5 minutes and 0.2M input tokens.

## PhySimCode: three 2D phenomena

| Sample | Video produced by the re-run | True parameters reported | Recovered within 20 % (of the true ones) |
|---|---|---|---|
| `scipy/projectile_bounce/006` | yes | 7 of 8 | 4 of 8 (0.500) |
| `scipy/simple_pendulum/014` | yes | 4 of 7 | 2 of 7 (0.286) |
| `scipy/single_spring/005` | yes | 3 of 5 | 0 of 5 (0.000) |

Every program ran frozen and offline and wrote its video; PhySimCode's own matching recovers 6 of
the 20 true parameters within 20 % (0.300, pooled as PhySimCode's Table 5 pools them; 6 of the 14
it reported). The spring is the hard case: Codex's mass,
stiffness and initial velocity are each 40 to 50 % below the truth. The oscillation's period fixes only
the ratio of stiffness to mass, and Codex got that right (0.95 against the true 0.97); the
video does not say how heavy the mass is. 2.5 to 3 minutes each.

## PhysCodeGym: one Blender scene

On `physics_iq_0001_ball_and_block_fall` (`rerun` variant: the scored frames are those of the
frozen scene re-rendered offline), Codex's Blender program scores a motion-mask IoU of 0.472 and
an SSIM of 0.826 against the 4K reference. It took 705 s and 1.2M input tokens. Its re-renders
differ from its own frames only by Blender's sampling noise.

## Physics-IQ: six real videos

Codex on six Physics-IQ scenarios, one or two per physics category, seen from the center. The
tasks use the `rerun` variant: the scored video is the one the frozen program wrote when the Lab
re-ran it offline, and in all six cases it was identical, pixel for pixel, to the video the
program wrote in Codex's session. All six were delivered, in 4 to 7 minutes each (0.39M to
0.84M input tokens). Every program draws the scene with NumPy and moves the objects in image
space (for example, a ballistic fall in pixels with a restitution factor at each bounce); none
used Blender, which was available in the session.

| Scenario | Category | Codex | Static | Second take | Codex: stIoU / sIoU / wsIoU / MSE |
|---|---|---|---|---|---|
| ball-and-block-fall | solid mechanics | 60.1 | 53.4 | 100 | 0.503 / 0.300 / 0.183 / 0.071 |
| block-domino | solid mechanics | 18.0 | 0 | 100 | 0.015 / 0.383 / 0.222 / 0.063 |
| fill-glass-red-drink | fluid dynamics | 0 | 2.3 | 100 | 0.007 / 0 / 0 / 0.040 |
| mirror-ball-fall | optics | 100 (raw 109) | 50.8 | 100 | 0.285 / 0.452 / 0.307 / 0.064 |
| lit-candle | thermodynamics | 98.6 | 100 | 100 | 1 / 1 / 1 / 0.014 |
| magnet-domino | magnetism | 31.6 | 4.1 | 100 | 0.087 / 0.489 / 0.387 / 0.005 |
| **these 6 views** (official formula) | | **52.2** | 38.4 | 100 | |

Scores are Physics-IQ's own (0 to 100), computed by its pinned code. *Static* holds the last
visible frame for 5 seconds; *second take* is the other real recording used as the prediction,
the upper bound. Reading the numbers:

- Compare with *static*, not with 0. On `lit-candle` neither real take has motion that the
  official masks detect, so every IoU is 1 for any prediction without motion and only the
  frame MSE separates the scores: the real last frame held still gets 100, Codex's drawn frames
  98.6.
- Single views are noisy and capped at 100. On `mirror-ball-fall` the spatiotemporal IoU of
  Codex's video with the reference take (0.285) is higher than the second take's (0.166); the
  raw score, 109, is clipped.
- Codex's program for a pouring liquid (`fill-glass-red-drink`) put no motion where the liquid
  moves: 0, below static.

**Video models.** Physics-IQ's leaderboard (the "Physics-IQ Original Leaderboard" in the README
of the pinned commit `d3f3de16`) gives one overall score per model, over all 198 views: the best
multi-frame (video-to-video) entry scores 64.5 (Magi-1 with GeoPhys, best of N) and the best
image-to-video entry 48.9; VideoPoet (multi-frame) scores 29.5 and Sora (image-to-video) 10.0.
The six views above are a **subset, not directly comparable** with those numbers: other
scenarios, one run, center views only. Like the multi-frame setting, the Lab's tasks give the
method the first 3 seconds of video. The leaderboard's generated videos are not published, so
the video models cannot be re-scored on this subset.

## Video2World: two robot videos

Codex, 60 minutes per task, in `r2s-agent-codex-video2world` (Codex with ManiSkill and SAPIEN, so
it can validate and replay its package). Video2World's own harness gives agents 180 minutes and
caps their API cost, so these scores are not comparable with its published ones. Each package
was frozen and scored by Video2World's own evaluator. The reference point is Video2World's own
score of the human-assisted reference it ships (from the metric record it ships with it).

| Instance | Codex V2WScore | Reference V2WScore | Build | Task success | G | D | Time | Tokens in / out |
|---|---|---|---|---|---|---|---|---|
| push-t `twin_pusht_demo_0` | 17.9 | 99.0 | yes | no | 0.536 | 0 | 502 s | 0.55M / 7.3k |
| droid-pick-place `droid_iris_..._bowl_hand` | 17.1 | 98.1 | yes | no | 0.514 | 0 | 673 s | 0.97M / 6.7k |

Both packages passed the validator and built in the simulator, and Codex stopped well inside its
budget, once its package validated. Its object shapes are close (shape Chamfer 0.8 and 1.1 cm),
but the scene and the object's trajectory are far from where things really are (scene Chamfer at
the 20 cm cap, trajectory error near 1 m): the metric depth of a monocular video is where it went
wrong. Its own actions then did not push or carry the object to the goal in its own scene. When the
Lab executes that reference itself, it reproduces its record on Push-T (99.1) but not fully on
this DROID instance (89.3, with task success); see
[the Video2World page](benchmarks/video2world.md).

## 4DCodeBench: three synthetic scenes

Codex ran in 4DCodeBench's own agent image (Blender 5.2, its world checker) with the task's GPU,
20 minutes per task (4DCodeBench's harness allows 60, so these scores are not comparable with its
published ones). The official scorer scores the world Codex left in its session. Only the
official metric types that need no model checkpoints were computed (the others need gated
weights); the reference world scores 1.0 on each of them.

| Case | Dynamic IoU | Scene 3D | Trajectory DTW | EMD step | Occupancy DTW | Interpenetration | Time | Tokens in / out |
|---|---|---|---|---|---|---|---|---|
| B_30 | 0.978 | 0.981 | 0.885 | 0.926 | 0.875 | 1.0 | 632 s | 1.03M / 10.1k |
| B_12 | 0.412 | 0.885 | 0.933 | 0.668 | 0.919 | 1.0 | 712 s | 1.00M / 10.8k |
| G_12 | 0.250 | 0.764 | 0.920 | 0.182 | 0.931 | 1.0 | 661 s | 1.17M / 9.0k |

The adapter's `lab` variant re-runs a delivered `build.sh` offline and compares the frozen
re-run's world with the ground truth: on `B_30`, Codex's program reproduces it with a
centroid-set Chamfer distance of 0.0058 m in 3D after a similarity alignment (static: 0.72) and
of 0.0012 image diagonals in the image (static: 0.24). The image distance is the variant's
primary metric, so the reproduction skill is 1 − 0.0012 / 0.24 = 0.995. Getting there took two fixes to the Lab, both now in place: the
statement says that the re-run mounts `solution/` read-only, and programs that delete and
recreate their output directory, as 4DCodeBench's statement asks, get a writable scratch parent
(`replaceable_output`).

## Agents' network

In the eight runs made with this release, the egress proxy allowed 114 connections, all to the
model API, and refused 64 (`chatgpt.com`, `ab.chatgpt.com`, `github.com`, `api.github.com`:
Codex's telemetry and update checks), without affecting the runs. In the six Physics-IQ runs it
allowed 274 connections, all to the model API, and refused 66 (the same hosts and `pypi.org`);
in the two Video2World runs, 69 allowed and 29 refused. Redaction found no credential in any
trajectory.

## Reproduce

```bash
r2s env build r2s-agent-codex r2s-agent-codex-blender r2s-agent-codex-physimcode
r2s run examples/experiments/unified.toml --out runs/unified         # the scorecard's tasks
r2s run examples/experiments/demo-iris.toml --out runs/demo-iris     # with the baselines
r2s run examples/experiments/demo-physicsiq.toml --out runs/demo-physicsiq
r2s env build r2s-video2world r2s-agent-codex-video2world
r2s run examples/experiments/demo-video2world.toml --out runs/demo-video2world
r2s run examples/experiments/demo-4dcodebench.toml --out runs/demo-4dcodebench
```

`unified.toml` runs the scorecard's 19 tasks in one experiment, each with its benchmark's own
time budget (60 minutes on Physics-IQ and 4DCodeBench, where the runs above had 20); the demo
files repeat the per-benchmark runs with the budgets above. Physics-IQ, PhysCodeGym, Video2World
and 4DCodeBench need a GPU, and 4DCodeBench its own images; see their pages. One run per task: expect different numbers from another run of the same agent.
