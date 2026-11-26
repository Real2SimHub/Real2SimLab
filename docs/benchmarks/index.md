# Benchmarks

Six benchmarks and a synthetic suite of our own, behind one interface. Each keeps its own
protocol: an official scorer runs from a pinned checkout, or, for raw datasets, the Lab's
reproduction metrics. Every result carries the benchmark's headline metric, so all of them line
up in one scorecard.

| | Benchmark | Data | Protocol | Headline (references) | Fidelity check | License |
|:-:|---|---|---|---|---|---|
| ![](../assets/icons/physics-iq.svg){ .r2s-icon } | [Physics-IQ](physics-iq.md) | real videos of 66 physical events, 3 views, 2 takes | native | Physics-IQ score, 0–100 (static, second take) | matches the official pipeline to the last digit (tested) | Apache-2.0 code, CC-BY-4.0 data |
| ![](../assets/icons/video2world.svg){ .r2s-icon } | [Video2World](video2world.md) | robot and hand manipulation (DROID, Push-T, FurnitureBench, DexYCB, HOI4D, HOT3D, OakInk2, RoboDojo, in-house) | native | V2WScore, 0–100 (reference solution) | shipped reference packages: Push-T reproduces; DROID's do not (a release issue, documented) | Apache-2.0 code; data per source |
| ![](../assets/icons/fourdcodebench.svg){ .r2s-icon } | [4DCodeBench](fourdcodebench.md) | 100 synthetic and 100 real-world videos of dynamic scenes | native | dynamic IoU, 0–1 | the official scorer in its own image gives the reference world full marks | none declared |
| ![](../assets/icons/physcodegym.svg){ .r2s-icon } | [PhysCodeGym](physcodegym.md) | real and generated videos | native | motion IoU (`S_mot`), 0–1 | the official metric code from the pinned checkout; no public scores to align with | MIT code, per-video licenses |
| ![](../assets/icons/physimcode.svg){ .r2s-icon } | [PhySimCode](physimcode.md) | simulated 2D physics, with programs and parameters | native | parameters within 20 %, 0–1 | the ground-truth programs score 1 with zero error | data MIT; evaluation code none declared |
| ![](../assets/icons/iris.svg){ .r2s-icon } | [IRIS](iris.md) | real videos of a ball dropped from three heights, ten takes each | reproduction | reproduction skill, 0–1 (static, another recording) | extracted trajectories recover gravity to 1–2 % without assuming it | CC-BY-NC-4.0 |
| ![](../assets/icons/synthetic.svg){ .r2s-icon } | [Synthetic](synthetic.md) | 5 rendered phenomena, two visibility tiers | reproduction | reproduction skill, 0–1 (static) | exact truth; the generator scores 1 | Apache-2.0 |

`r2s bench list` shows the adapters; `r2s bench build NAME --out tasks/` builds a benchmark's
tasks. Adapters download third-party code and data at a pinned revision into the cache
(`~/.cache/real2simlab`, or `R2S_CACHE_DIR`) when tasks are built or scored; nothing
third-party is in this repository, its packages or its published images. Data without a
declared license is fetched only from its authors' own location and its tasks are marked
`redistributable = false`.

**Status.** *Ready*: every task builds and validates, and the protocol matches the source.
*Partial*: the tasks build and score, and the page says what is missing or not yet comparable
with the official numbers.

**Frozen re-runs.** Where a benchmark scores what the agent produced in its session, its
adapter also offers a variant that scores the frozen, offline re-run of the delivered program
instead: `rerun` for Physics-IQ and PhysCodeGym (their official scorers score the re-run's
output), `lab` for 4DCodeBench (the re-run's world, scored with the Lab's metrics); PhySimCode
always re-runs. Both variants record whether the two agree.

Your own benchmark: [Add a benchmark](../tutorials/add-benchmark.md), or
[write tasks](../tutorials/write-tasks.md) directly from your data.
