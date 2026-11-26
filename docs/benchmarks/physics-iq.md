# ![](../assets/icons/physics-iq.svg){ .r2s-title-icon } Physics-IQ

| | |
|---|---|
| Adapter | `physicsiq` (`r2s bench build physicsiq --out tasks/`) |
| Status | partial: native protocol only; see [Gaps](#gaps) |
| Headline | `physics_iq_score` (0 to 100), computed over all scored views together as Physics-IQ does; references: static (the last frame held), another recording (the second real take). A view with no valid video (none delivered, or one the protocol or OpenCV refuses) counts with the worst value of every metric (IoUs 0, MSE 1), so it counts against the method; its own score is 0 |
| Source | [github.com/google-deepmind/physics-IQ-benchmark](https://github.com/google-deepmind/physics-IQ-benchmark) at commit `d3f3de16` (code, run at score time); videos from the public Google Cloud Storage bucket [`physics-iq-benchmark`](https://console.cloud.google.com/storage/browser/physics-iq-benchmark) (default) or the gated Hugging Face dataset [Physics-IQ Verified](https://huggingface.co/datasets/Anates-Labs-Research/Physics-IQ-Verified) at revision `b65f567d` |
| License | code Apache-2.0; data CC-BY-4.0. Videos are downloaded into the cache when tasks are built and are not redistributed: not in this repository, its packages or its images. |
| Citation | Motamed, Culp, Swersky, Jaini, Geirhos. *Do generative video models understand physical principles?* [arXiv:2501.09038](https://arxiv.org/abs/2501.09038); project page: [physics-iq.github.io](https://physics-iq.github.io/) |

Physics-IQ films 66 real physical events (solid mechanics, fluid dynamics, optics,
thermodynamics, magnetism), each from three perspectives and twice. A model sees the first
3 seconds and predicts the next 5 as video. The benchmark compares *where things move*:
motion masks of the prediction against those of the real continuation.

## Tasks

One task per scenario and perspective (`scenario/perspective`; `--select ball-ramp` means the
center view). The agent gets take 1's first 3 seconds (30 fps) as `visible/cam/rgb.mp4` and
delivers a program, `solution/run.py`, that writes `render.mp4`: exactly 150 frames at 30 fps,
the 5 seconds after the last visible frame. The runtime is `r2s-blender-4.0` (Python with NumPy
and SciPy, and Blender 4.0.2 for programs that simulate and render a scene). The program may
not contain video or image files: frames cut from the visible clip would not show the future.

| Variant | What is scored |
|---|---|
| `native` | the video the agent's program wrote in its session, the official input |
| `rerun` | the video of the Lab's frozen, offline re-run of the program |

The 5 seconds to predict, take 2 and both takes' motion masks stay in `hidden/`; Physics-IQ asks
that its test data never be used to improve a submission, and no method ever sees them.

**Data.** The default release, `original`, is the public bucket, pinned by object generation
(objects written after 2025-02-21 are refused) and checked by MD5. Each scenario view is about
15–20 MB (the conditioning clip, two test clips, two masks); only the selected views are
downloaded. `--option release=verified` uses Physics-IQ Verified instead: accept its terms on
Hugging Face, then log in on this machine first with `hf auth login` (`huggingface-cli login`
before huggingface_hub 1.0), or set `HF_TOKEN`. Default selection: `ball-and-block-fall`,
`block-domino` and `balls-collide`, center view, both variants.

## Scoring

The scorer plugin `physics_iq` runs Physics-IQ's own code from the pinned checkout: its motion-mask
generator on the predicted video, then its per-view metrics against take 1 (spatiotemporal IoU,
spatial IoU, weighted spatial IoU, per-frame MSE) and the same metrics between take 1 and take 2,
the **physical variance**: how much two real recordings of the same event differ. This is
Physics-IQ's noise floor; a perfect prediction cannot beat it. The task result's `native` field
holds, for each view:

| Field | Meaning |
|---|---|
| `spatiotemporal_iou`, `spatial_iou`, `weighted_spatial_iou`, `mse` | the prediction against take 1 |
| `physical_variance_*` | the same metrics between the two real takes |
| `physics_iq_score` | the official score formula on this one view, 0–100 |
| `static` | the same numbers for the last visible frame held for 5 seconds |
| `take_2` | the same numbers for the second real take used as the prediction: the upper bound |

Read a score next to `static`: the official spatiotemporal IoU counts frames in which neither
video moves as a match, so a video in which nothing moves scores about 50 on a scene that is
mostly still (53.4 for `static` on `ball-and-block-fall`, center view).

The official benchmark score averages each metric over all views first and then applies the
formula: pass several reports to `real2simlab.benchmarks.physicsiq.scorer.physics_iq_score`.
`tests/test_physicsiq.py` checks, on real data, that the per-view metrics equal the official
`process_videos` output and the score equals `calculate_iq_score` for the same videos.

```bash
r2s env build r2s-blender-4.0 r2s-agent-codex-blender
r2s bench build physicsiq --out tasks/piq --select ball-and-block-fall
r2s eval --tasks tasks/piq --method codex=codex:image=r2s-agent-codex-blender --out runs/piq
```

`r2s-agent-codex-blender` is the Codex image on the Blender runtime, so the agent can run Blender
in its session; it is built locally.

## Comparing with video models

The official leaderboards report one overall score per model, over all 198 views (66 scenarios ×
3 perspectives), with generated videos not published. Scores of code twins on a subset of
scenarios are therefore **not directly comparable** with them: see the [example run](../results.md) for
the demonstration subset next to the static and second-take references on the same subset.

## Gaps

- Only the native protocol: Physics-IQ has no object tracks, and the Lab's own metrics are
  trajectory metrics.
- The official score is defined over all 198 views and four generation runs; the adapter scores
  any subset, one run per task.
