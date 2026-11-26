# ![](../assets/icons/physimcode.svg){ .r2s-title-icon } PhySimCode

| | |
|---|---|
| Adapter | `physimcode` (`r2s bench build physimcode --out tasks/`) |
| Status | partial: 2D (SciPy) phenomena, model-free measures only; see [Gaps](#gaps) |
| Headline | `parameters_within_20pct`: the share of the true parameters recovered within 20 % (0 to 1), PhySimCode's end-to-end recovery; over tasks, parameters are pooled, those of a task the method delivered nothing for included (none recovered) |
| Source | Hugging Face dataset [`SourajaKundu123/PhySimCode`](https://huggingface.co/datasets/SourajaKundu123/PhySimCode) at revision `1d03cd78`; evaluation code [github.com/SourajaKundu/PhySimCode](https://github.com/SourajaKundu/PhySimCode) at commit `e876cd20` |
| License | data MIT (declared on the dataset card), downloaded when tasks are built and not redistributed. The evaluation code declares **no license**: it is fetched from its authors at the pinned commit when tasks are scored, never copied into this repository, its packages or images (see [Add a benchmark](../tutorials/add-benchmark.md) for the rule). |
| Citation | Kundu, Gupta, Julin, Zhao, Xie, Jeni. *PhySimCode: A Benchmark and Evaluation Method for Physics Video to Code Generation.* arXiv:2610.15722. |

PhySimCode's evaluation set has 2,430 videos of 162 simulated phenomena (87 in 2D, simulated
with SciPy and drawn with Matplotlib; 75 in 3D, PyBullet and pyrender), 15 samples each with
randomly drawn parameters. Every sample comes with the program that rendered it, its parameter
values and a physics explanation. Given only the video, a model explains the physics (the law
and the parameter values) and writes a program that regenerates the video.

## Tasks

One task per sample, named `engine/experiment/sample` (for example
`scipy/simple_pendulum/014`). The agent sees the video (`visible/cam/rgb.mp4`, 360 × 180,
10 fps, 8 s) and delivers, in the Lab's run contract:

- `solution/cot.json`: PhySimCode's explanation format, with `simulation`,
  `physical_observation`, `physical_law` (name, statement, formula) and `parameters` (value,
  unit and description of each);
- `solution/run.py`: a self-contained program (NumPy, SciPy, Matplotlib) that simulates the
  phenomenon and writes the video at 30 fps to `$R2S_OUTPUT_DIR/render.mp4` (PhySimCode's own
  programs write `video.mp4` in their working directory; the content asked for is the same).

The runtime is `r2s-physimcode` (r2s-python and Matplotlib 3.11.2). The true program, its
parameters and its explanation stay in `hidden/reference/`. Default selection:
`scipy/simple_pendulum/014`, `scipy/projectile_bounce/006`, `scipy/single_spring/005`.

## Scoring

The scorer plugin `physimcode`, on the Lab's frozen, offline re-run of the delivery:

| Field | Meaning |
|---|---|
| `runnability.produces_video` | the frozen re-run wrote a readable `render.mp4` (with its frame count, frame rate and size) |
| `parameters.n_gt`, `n_matched`, `n_within_tolerance` | the true parameters (after PhySimCode's filter), those the delivered `cot.json` reports (matched by name as PhySimCode matches them), and those within 20 % of the true value (in SI units) |
| `parameters.fraction_reported` | `n_matched / n_gt` |
| `parameters.fraction_within_tolerance` | `n_within_tolerance / n_matched`: the accuracy among the *matched* parameters only, so a delivery that reports one parameter correctly gets 1 here |
| `parameters.cumulative_error`, `matched` | the sum of relative errors and each matched pair |

Parameter recovery runs PhySimCode's own `eval/eval_parameters.py` from the pinned checkout,
which leaves out gravity, colours, positions, camera settings and the like before matching.
The headline is `n_within_tolerance / n_gt`, the *end-to-end recovery* that PhySimCode's
paper reports (Table 5): the share of all the true parameters recovered, so reporting fewer
parameters never helps. Over several tasks the parameters are pooled as PhySimCode pools
them (the sum of `n_within_tolerance` over the sum of `n_gt`). A delivery without a readable
`cot.json` recovers nothing and still counts its true parameters.
The true programs, wrapped as deliveries, score 1 on both, with zero error, in
`r2s-physimcode` (checked on the three default samples).

## Gaps

- **3D phenomena**: their programs need PyBullet and pyrender with an EGL device; not yet.
- **The other official measures** are not run: physical-law validity and equivalence need
  three commercial LLM judges, the video similarity needs DINOv2 and X-CLIP weights (not
  pinned), CodeBLEU needs tree-sitter grammars. Human ratings have no code.
- **Videos are not compared**: the dataset's videos were converted to 10 fps and a low
  resolution by an unpublished step (the true programs render 12 s at 30 fps), so a delivered
  video cannot be compared with the clip frame for frame.
