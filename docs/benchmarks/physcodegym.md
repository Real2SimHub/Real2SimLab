# ![](../assets/icons/physcodegym.svg){ .r2s-title-icon } PhysCodeGym

| | |
|---|---|
| Adapter | `physcodegym` (`r2s bench build physcodegym --out tasks/`) |
| Status | partial: model-free official metrics only; see [Gaps](#gaps) |
| Headline | `S_mot` (motion-mask IoU, 0 to 1); 0 for a task with no frames delivered or rendered |
| Source | [github.com/ShaoShuai0605/PhysCodeGym](https://github.com/ShaoShuai0605/PhysCodeGym) at commit `de1c8b75`; Hugging Face dataset `ShaoShuai0605/PhysCodeGym` (`d65acd3c`) |
| License | code MIT; every reference video keeps its source license (Pexels License, CC-BY-4.0, CC-BY-NC-4.0, ...), recorded in each task's `license`. Videos are downloaded when tasks are built and are not redistributed (`redistributable = false`). |
| Citation | Shao, He, Fan, Guo et al. *PhysCodeGym: Benchmarking Coding Agents for Physically Verifiable 4D Reconstruction.* [arXiv:2610.17288](https://arxiv.org/abs/2610.17288); project page: [shaoshuai0605.github.io/PhysCodeGym-Project-Page](https://shaoshuai0605.github.io/PhysCodeGym-Project-Page/) |

An agent reconstructs a reference video (8 s, 30 fps) as an executable Blender program whose
motion comes from simulation. PhysCodeGym has 156 tasks (rigid bodies, liquids, cloth, wind).

## Tasks

Delivery follows PhysCodeGym's contract, the `blender_scene` convention: `solution/scene.py`
builds the scene from an empty Blender file, saves `/app/outputs/model.blend` and renders
`/app/outputs/frames/frame_0001.png` … `frame_0240.png` at 640×360. The runtime image is
`r2s-blender-4.0` (Blender 4.0.2, as in PhysCodeGym; EEVEE renders on a GPU through EGL,
Cycles also on the CPU); its runner `/opt/r2s/blender_scene.py` runs the scene headless with a
time limit and encodes `render.mp4`, so the output is in the `video` format.

| Variant | What is scored |
|---|---|
| `native` | the frames the agent rendered in its session, as PhysCodeGym scores them |
| `rerun` | the frames of the Lab's frozen, offline re-run of `scene.py`: does the program reproduce its own submission? |

Both are scored by the plugin `physcodegym_pixels`, which runs PhysCodeGym's own
`physcodegym.metrics.evaluate` (from the pinned checkout) on the 4K reference and the 240
frames: `S_SSIM` (frame-aligned SSIM at 640×360), `S_mot` (motion-mask IoU) and, when `lpips`
and `torch` are installed, `S_LPIPS`. The result is stored unchanged in the task result's
`native` field. Only PhysCodeGym's checks of the delivered frames (not exactly 240 of them, one
it cannot decode) are the delivery's result (`invalid_frames`, a score of 0). When its metrics
reject the reference video as this machine's OpenCV reads it (an OpenCV without a video backend,
say), or this machine's OpenCV cannot read even a PNG it wrote itself, the scorer failed: an
infrastructure error, not a score.

Defaults: three CC-BY-4.0 rigid-body cases from Physics-IQ
(`physics_iq_0001_ball_and_block_fall`, `physics_iq_0019_ball_ramp`,
`physics_iq_0034_block_domino`). Choose others with `--select`.

`examples/twins/physcodegym/` holds a deliberately crude hand-written scene for the first case
(a ball and a block falling onto a table), to show the convention end to end. A program given
by path was not written in an agent session, so it has no session frames for `native` to
score (it would score 0 there, whatever the scene): build only the `rerun` variant for it.

```bash
r2s env build r2s-blender-4.0
r2s bench build physcodegym --out tasks/pcg --select physics_iq_0001_ball_and_block_fall \
  --option variants=rerun
r2s eval --tasks tasks/pcg --method hand=program:path=examples/twins/physcodegym --out runs/pcg
```

An agent (`--method codex`, ...) is evaluated on both variants: build without `--option`.

## Gaps

- The object, event and physical-validity scores (`S_obj`, `S_evt`, `S_val`) need LLM judges
  and a model API. They run through PhysCodeGym's own Harbor tasks and verifier (its
  `physcodegym build-tasks`; Harbor version pinned by its `harbor-version.txt`), with the
  provider configured as its `docs/evaluation.md` describes. This needs credentials and is
  left to users with model access: follow PhysCodeGym's own `docs/evaluation.md` at the
  pinned commit.
- PhysCodeGym's agent environment offers observation tools (`get_frame`, `run_blender`, ...)
  and an iteration budget. Lab tasks give the agent the reference video and Blender directly in
  its workspace; scores are therefore comparable to the official ones for the same program,
  not for the same agent run. The video a method gets is the 4K reference reduced to
  1280×720, the largest frames PhysCodeGym's observation tools show (`MAX_FRAME_WIDTH` in its
  `experiment/tools/observer_service.py`); the scorer still compares against the 4K
  reference.
