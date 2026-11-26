---
title: Real2SimLab
hide:
  - navigation
  - toc
---

<div class="r2s-hero" markdown>

![Real2SimLab](assets/logo-light.svg#only-light){ width="440" }
![Real2SimLab](assets/logo-dark.svg#only-dark){ width="440" }

# A unified platform for real-to-simulation { .r2s-tagline }

[Get started](quickstart.md){ .md-button .md-button--primary }
[Evaluate an agent](tutorials/evaluate-agent.md){ .md-button }

</div>

![Methods on the left, benchmarks on the right, each connected once to Real2SimLab](assets/teaser-light.svg#only-light)
![Methods on the left, benchmarks on the right, each connected once to Real2SimLab](assets/teaser-dark.svg#only-dark)

Real-to-simulation benchmarks ask a model to turn a recording of a physical event into a
simulation program that reproduces it, and each comes with its own data layout, delivery format,
agent setup and scorer. Real2SimLab connects each agent and each benchmark once: it runs every
method the same way (sandboxed, network-limited, recorded), freezes what it delivers, re-runs it
offline, and scores it with the benchmark's own official scorer, so results from every benchmark
land in one record format and one scorecard. The same task format and scorer serve training:
training tasks with exact truth or your own, datasets checked against the evaluation tasks,
methods trained on them and evaluated only on tasks that do not overlap them, and any program's
evaluation score as its reward.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } **Quickstart**

    ---

    Install from the repository and score baselines and two scripted agents in about 30 seconds.

    [:octicons-arrow-right-24: Quickstart](quickstart.md)

-   :material-robot-outline:{ .lg .middle } **Evaluate an agent**

    ---

    Codex, Claude Code, Gemini CLI, OpenCode, mini-swe-agent or any Harbor agent, sandboxed,
    with the network open only to the model API.

    [:octicons-arrow-right-24: Evaluate an agent](tutorials/evaluate-agent.md) ·
    [Evaluation guide](guides/evaluation.md)

-   :material-database-outline:{ .lg .middle } **Benchmarks**

    ---

    Physics-IQ, Video2World, 4DCodeBench, PhysCodeGym, PhySimCode, IRIS and a synthetic suite,
    each scored its own way.

    [:octicons-arrow-right-24: Benchmarks](benchmarks/index.md)

-   :material-chart-box-outline:{ .lg .middle } **Example run**

    ---

    What a full evaluation produces: one agent on every benchmark, its records and scorecard.

    [:octicons-arrow-right-24: Example run](results.md)

-   :material-school-outline:{ .lg .middle } **Training**

    ---

    Training tasks with exact truth or your own; datasets checked against the evaluation
    tasks; trained methods; and any program's evaluation score as its reward.

    [:octicons-arrow-right-24: Training](guides/training.md)

-   :material-puzzle-outline:{ .lg .middle } **Add a benchmark**

    ---

    An adapter, the official scorer as a plugin, a check against its reference solutions; or
    let your coding agent do it with the bundled skill.

    [:octicons-arrow-right-24: Add a benchmark](tutorials/add-benchmark.md)

-   :material-file-document-outline:{ .lg .middle } **Specification 1.0**

    ---

    Tasks, captures, twins, formats, scoring and records, with JSON Schemas; frozen within 1.x.

    [:octicons-arrow-right-24: Specification](spec/index.md)

-   :material-map-outline:{ .lg .middle } **Awesome-Real2Sim**

    ---

    A curated survey of real-to-simulation research, released with the Lab: methods,
    benchmarks, datasets and surveys, each placed by what it delivers.

    [:octicons-arrow-right-24: Awesome-Real2Sim](https://github.com/Real2SimHub/Awesome-Real2Sim)

</div>

## What every evaluation does

| Step | What happens |
|---|---|
| ![](assets/icons/step-build.svg){ .r2s-icon } **Build** | The method gets the task's visible data and works in a sandbox; an agent's network reaches only its model API, through a logging proxy. |
| ![](assets/icons/step-freeze.svg){ .r2s-icon } **Freeze** | The delivered program is packaged by the Lab and fingerprinted. |
| ![](assets/icons/step-rerun.svg){ .r2s-icon } **Re-run** | The frozen program runs offline and read-only, optionally twice; its output is checked against the agent's own. |
| ![](assets/icons/step-score.svg){ .r2s-icon } **Score** | The benchmark's official scorer, or the Lab's reproduction metrics for raw datasets. |
| ![](assets/icons/step-report.svg){ .r2s-icon } **Report** | One record per result with the benchmark's headline metric; one scorecard across benchmarks. |

More in [Concepts](concepts.md); how this relates to the benchmarks themselves and to Harbor in
[Related work](related.md).
