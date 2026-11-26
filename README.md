<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
  <img alt="Real2SimLab" src="docs/assets/logo-light.svg" height="64">
</picture>

<h3>A unified platform for real-to-simulation</h3>

<p><b>Any method, any benchmark: true to each, together on one scorecard.</b></p>

<p>
📄 <a href="https://arxiv.org/abs/2611.34849"><b>Paper</b></a> &nbsp;·&nbsp;
📖 <a href="https://real2simhub.github.io/Real2SimLab/"><b>Documentation</b></a> &nbsp;·&nbsp;
🚀 <a href="docs/quickstart.md"><b>Quickstart</b></a> &nbsp;·&nbsp;
📊 <a href="docs/benchmarks/index.md"><b>Benchmarks</b></a> &nbsp;·&nbsp;
🧪 <a href="docs/results.md"><b>Example run</b></a> &nbsp;·&nbsp;
🎯 <a href="docs/guides/evaluation.md"><b>Evaluation</b></a> &nbsp;·&nbsp;
🏋️ <a href="docs/guides/training.md"><b>Training</b></a> &nbsp;·&nbsp;
📐 <a href="docs/spec/index.md"><b>Specification</b></a> &nbsp;·&nbsp;
🌏 <a href="README_zh.md"><b>中文</b></a>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2611.34849-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2611.34849)
[![CI](https://github.com/Real2SimHub/Real2SimLab/actions/workflows/ci.yml/badge.svg)](https://github.com/Real2SimHub/Real2SimLab/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-4f46e5?logo=materialformkdocs&logoColor=white)](https://real2simhub.github.io/Real2SimLab/)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776ab?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-d22128?logo=apache&logoColor=white)](LICENSE)
[![Spec](https://img.shields.io/badge/spec-1.0-6d28d9?logo=json&logoColor=white)](docs/spec/index.md)
[![Sandbox](https://img.shields.io/badge/sandbox-Docker%20%7C%20bubblewrap-2496ed?logo=docker&logoColor=white)](docs/concepts.md)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-1f2328?logo=linux&logoColor=white)](docs/quickstart.md)
[![WeChat](https://img.shields.io/badge/WeChat-group-07c160?logo=wechat&logoColor=white)](https://real2simhub.github.io/assets/img/connect/wechat-group.jpg)

<!-- The Real2SimHub series: the hub, this repository and the research map. -->
<table align="center">
  <tr>
    <td align="center"><a href="https://real2simhub.github.io/"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/symbol-hub-dark.svg"><img src="docs/assets/symbol-hub.svg" width="26" height="22" align="absmiddle" alt=""></picture> <b>Real2SimHub</b></a></td>
    <td align="center"><a href="https://github.com/Real2SimHub/Real2SimLab"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/symbol-lab-dark.svg"><img src="docs/assets/symbol-lab.svg" width="22" height="22" align="absmiddle" alt=""></picture> <b>Real2SimLab</b></a></td>
    <td align="center"><a href="https://github.com/Real2SimHub/Awesome-Real2Sim"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/symbol-awesome-dark.svg"><img src="docs/assets/symbol-awesome.svg" width="22" height="22" align="absmiddle" alt=""></picture> <b>Awesome&#8209;Real2Sim</b></a></td>
  </tr>
</table>

</div>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/teaser-dark.svg">
  <img alt="Methods on the left, benchmarks on the right, each connected once to Real2SimLab, which builds, freezes, re-runs, scores and reports" src="docs/assets/teaser-light.svg" width="100%">
</picture>

Real-to-simulation benchmarks ask a model to turn a recording of a physical event into a
simulation program that reproduces it. Physics-IQ, Video2World, 4DCodeBench, PhysCodeGym,
PhySimCode and others each come with their own data layout, delivery format, agent setup and
scorer. Evaluating one agent on all of them means one integration per pair, and results that are
hard to compare or reproduce.
**Real2SimLab connects each agent and each benchmark once.** It runs every method the same way
(sandboxed, network-limited, recorded), freezes what it delivers, re-runs it offline, and scores
it with the benchmark's own official scorer, so results from every benchmark land in one record
format and one scorecard. The same task format and scorer serve training: the Lab generates
training tasks with exact truth or takes your own, exports them and the runs on them as datasets
checked against the evaluation tasks, trains methods that learn, and scores any program as a
reward, exactly as an evaluation would.

## ✨ Highlights

- 🎯 **Six benchmarks, each scored its own way.** Official scorers run from pinned checkouts and
  are checked against the originals: on Physics-IQ the Lab's numbers match the official pipeline
  to the last digit. A raw dataset (IRIS) and our own synthetic suite with exact truth get the
  Lab's reproduction metrics; where an event was recorded again (IRIS), the repeats give the
  noise floor.
- 🤖 **Any agent, the same way.** 47 coding agents: every agent of
  [Harbor](https://harborframework.com) (47 in Harbor 0.27: OpenHands, Aider, Cursor CLI, Goose,
  Qwen Code...), five of them (Codex, Claude Code, Gemini CLI, OpenCode, mini-swe-agent) also
  directly; or your own pipeline's programs. Agents run in a
  sandbox whose network reaches only the model API; keys never reach runs, logs or images.
  Codex, mini-swe-agent and Harbor's agents work in an image with the task's runtime (Blender,
  ManiSkill, Matplotlib...); Claude Code, Gemini CLI and OpenCode in their own Python image for
  those tasks.
- 🧊 **Frozen, offline re-runs.** What is scored is the delivered program re-run offline and
  read-only, not what the agent showed in its session; the Lab checks that both agree.
- 📋 **One record, one scorecard.** Every result carries its benchmark's headline metric with its
  reference points (a static scene, a second recording, the benchmark's reference solution);
  the scorecard combines tasks the way each benchmark does, with bootstrap intervals.
- 🏋️ **Training data and rewards from the same code.** Generated training tasks with exact
  truth, or any dataset you convert; datasets of inputs, labels, delivered programs, rewards and
  redacted agent trajectories, with every task that holds an evaluation recording left out; any
  program's reward is its evaluation score. A
  [worked example](docs/guides/training.md#example-a-method-that-learns) trains a small method
  on them and evaluates it on the benchmark, skipping every task that overlaps its training
  data.
- 🧩 **Built to be extended.** A frozen [specification](docs/spec/index.md) with JSON Schemas,
  plugins for benchmarks, methods, metrics and backends, and [agent skills](AGENTS.md#skills)
  that walk your coding agent through integrating a new benchmark.
- 🗺️ **A survey of the field.** [Awesome-Real2Sim](https://github.com/Real2SimHub/Awesome-Real2Sim), released with the Lab, is a curated
  survey of real-to-simulation research: methods, benchmarks, datasets and surveys, each placed
  by what it delivers and tagged with how it is built, the physics it covers and how it is
  evaluated.

## 🚀 Quickstart

```bash
git clone https://github.com/Real2SimHub/Real2SimLab.git && cd Real2SimLab
pip install -e .        # Python 3.11+ and ffmpeg
r2s demo --open         # scores baselines and two scripted agents on two tasks: about 30 s on a CPU
```

The demo runs its twins in Docker when it is there, in bubblewrap on Linux without root, and
otherwise unisolated with a warning. Evaluations and the other commands that run twins use
Docker unless given another backend: without Docker, unpack the runtime image and pass
`--backend bwrap` ([Run on a cluster without Docker](docs/tutorials/clusters.md)).
`r2s doctor` lists what is missing and how to fix it.

## 🤖 Evaluate your agent

```bash
r2s bench list                                         # the benchmarks: status, protocols, splits, license
r2s run examples/experiments/unified.toml              # Codex on all of them, one scorecard
r2s eval --benchmark synthetic --method claude_code --out runs/claude   # or one at a time
```

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Agent | `--method` | How it runs |
|:-:|---|---|---|
| <img src="docs/assets/icons/codex.svg" width="28" alt=""> | Codex&nbsp;CLI | `codex` | `codex exec` in the sandbox; your Codex login or `OPENAI_API_KEY` |
| <img src="docs/assets/icons/claude-code.svg" width="28" alt=""> | Claude&nbsp;Code | `claude_code` | `claude -p` in the sandbox; image built locally (Claude Code is proprietary) |
| <img src="docs/assets/icons/gemini-cli.svg" width="28" alt=""> | Gemini&nbsp;CLI | `gemini_cli` | in the sandbox; `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| <img src="docs/assets/icons/opencode.svg" width="28" alt=""> | OpenCode | `opencode` | `opencode run` in the sandbox, any provider it supports (experimental) |
| <img src="docs/assets/icons/mini-swe-agent.svg" width="28" alt=""> | mini&#8209;swe&#8209;agent | `mini_swe` | the agent loop on the host with any LiteLLM model; every command in the sandbox |
| <img src="docs/assets/icons/harbor.svg" width="28" alt=""> | Harbor&nbsp;agents | `harbor` | any of the 47 agents [Harbor](https://harborframework.com) 0.27 supports (`agent=openhands`, `aider`, `cursor-cli`, ...) |
| <img src="docs/assets/icons/program.svg" width="28" alt=""> | your&nbsp;pipeline | `program` | its programs, written with the [twin SDK](docs/tutorials/write-twin.md) |

Codex, mini-swe-agent and Harbor's agents work in an image with the task's runtime; Claude Code,
Gemini CLI and OpenCode work in their own Python image on Blender, PhySimCode and Video2World
tasks (`image=` names another). Credentials, budgets and network rules:
[Evaluate an agent](docs/tutorials/evaluate-agent.md). Choosing tasks, comparing methods
and reporting a score: [Evaluation](docs/guides/evaluation.md).

## 🏋️ Train

```bash
r2s train tasks --count 200 --tier calibrated --out train/tasks  # training tasks, exact truth
r2s bench build synthetic --tier calibrated --out eval/tasks     # the evaluation tasks
r2s train export --tasks train/tasks --against eval/tasks --out train/data   # a dataset
r2s train fit nearest_twin --data train/data --out train/model   # a method that learns
r2s eval --tasks eval/tasks --method nearest_twin:checkpoint=train/model --out eval/run
r2s train reward --task <task> --solution <program dir>           # a reward = its evaluation score
```

Training tasks are the synthetic phenomena with sampled hidden parameters and initial states,
labelled with their trajectories, parameters and truth program, or your own tasks marked
`split = "train"`. An export with `--against` leaves out every task that holds the same file
or the same recording as an evaluation task; evaluation tasks are not exported unless you ask,
and never those of a benchmark whose terms forbid it. An evaluation of a trained method skips
every task that overlaps one of its training tasks. Tasks with a `run.py` solution, training
tasks included, also export as Harbor tasks for Harbor's reinforcement-learning integrations.
The Lab ships no trainer and no weights: [Training](docs/guides/training.md).

## 📊 Benchmarks

Six benchmarks, plus a synthetic suite of our own. Every adapter runs the benchmark's own
protocol, and is checked against the original before it is trusted:

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Benchmark | Scored by (headline metric) | Links |
|:-:|---|---|---|
| <img src="docs/assets/icons/physics-iq.svg" width="28" alt=""> | [**Physics-IQ**](docs/benchmarks/physics-iq.md)<br><sub>real&nbsp;physical&nbsp;events</sub> | official motion-mask metrics (Physics-IQ score) | [![Paper](https://img.shields.io/badge/-paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2501.09038) [![Code](https://img.shields.io/badge/-code-181717?logo=github&logoColor=white)](https://github.com/google-deepmind/physics-IQ-benchmark) [![Data](https://img.shields.io/badge/-data-4285f4?logo=googlecloud&logoColor=white)](https://console.cloud.google.com/storage/browser/physics-iq-benchmark) |
| <img src="docs/assets/icons/video2world.svg" width="28" alt=""> | [**Video2World**](docs/benchmarks/video2world.md)<br><sub>robot&nbsp;and&nbsp;hand manipulation</sub> | official evaluator, executing the package (V2WScore: build × mean of functionality, geometry and dynamics) | [![Paper](https://img.shields.io/badge/-paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.04432) [![Code](https://img.shields.io/badge/-code-181717?logo=github&logoColor=white)](https://github.com/AetherLabsAI/Video2World) [![Data](https://img.shields.io/badge/-data-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/AetherLabs-AI/Video2World) |
| <img src="docs/assets/icons/fourdcodebench.svg" width="28" alt=""> | [**4DCodeBench**](docs/benchmarks/fourdcodebench.md)<br><sub>synthetic&nbsp;dynamic&nbsp;scenes</sub> | official scorer, in its own image (dynamic IoU) | [![Paper](https://img.shields.io/badge/-paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.03715) [![Code](https://img.shields.io/badge/-code-181717?logo=github&logoColor=white)](https://github.com/4DCodeBench/4DCodeBench) [![Data](https://img.shields.io/badge/-data-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/4DCodeBench/Dataset-Synthetic) |
| <img src="docs/assets/icons/physcodegym.svg" width="28" alt=""> | [**PhysCodeGym**](docs/benchmarks/physcodegym.md)<br><sub>real&nbsp;and&nbsp;generated</sub> | official pixel and motion metrics (motion IoU) | [![Paper](https://img.shields.io/badge/-paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.17288) [![Code](https://img.shields.io/badge/-code-181717?logo=github&logoColor=white)](https://github.com/ShaoShuai0605/PhysCodeGym) [![Data](https://img.shields.io/badge/-data-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/ShaoShuai0605/PhysCodeGym) |
| <img src="docs/assets/icons/physimcode.svg" width="28" alt=""> | [**PhySimCode**](docs/benchmarks/physimcode.md)<br><sub>simulated&nbsp;2D&nbsp;physics</sub> | official parameter recovery (parameters within 20 %) | [![Paper](https://img.shields.io/badge/-paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.15722) [![Code](https://img.shields.io/badge/-code-181717?logo=github&logoColor=white)](https://github.com/SourajaKundu/PhySimCode) [![Data](https://img.shields.io/badge/-data-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/SourajaKundu123/PhySimCode) |
| <img src="docs/assets/icons/iris.svg" width="28" alt=""> | [**IRIS**](docs/benchmarks/iris.md)<br><sub>real&nbsp;ball&nbsp;drops</sub> | Lab trajectory metrics (reproduction skill) | [![Paper](https://img.shields.io/badge/-paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.16432) [![Code](https://img.shields.io/badge/-code-181717?logo=github&logoColor=white)](https://github.com/KurbanIntelligenceLab/iris-bench) [![Data](https://img.shields.io/badge/-data-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/rasulkhanbayov/IRIS) |
| <img src="docs/assets/icons/synthetic.svg" width="28" alt=""> | [**Synthetic**](docs/benchmarks/synthetic.md)<br><sub>5&nbsp;rendered&nbsp;phenomena</sub> | Lab trajectory metrics (reproduction skill) | built in |

Third-party code and data are fetched at a pinned revision when you build or score tasks, never
redistributed; each page gives the license, the details of the check and what is not covered
yet. How this relates to the benchmarks themselves and to Harbor: [Related work](docs/related.md).
[An example run](docs/results.md) shows what a full evaluation produces: one agent on every
benchmark, its records and its scorecard.

## 🧩 Extend it

- 🛠️ **Your Real2Sim pipeline:** write its output with the [twin SDK](docs/tutorials/write-twin.md)
  and evaluate it with `--method program`.
- 🎥 **Your recordings:** [write tasks](docs/tutorials/write-tasks.md) or
  [add a benchmark](docs/tutorials/add-benchmark.md).
- 🤝 **Your coding agent does it:** open it in this repository and ask it to use the
  `add-benchmark` skill on your benchmark; it checks the license, implements the benchmark's
  own protocol, validates against its reference solutions and adds tests and a page.

A benchmark built on the Lab can include it as a git submodule ([guide](docs/guides/build-on.md)).

## 📝 Citation

If you use Real2SimLab or Awesome-Real2Sim, please cite our paper:

```bibtex
@article{real2simlab2026,
  title   = {Real2SimLab: A Unified Platform for Real-to-Simulation Transfer},
  author  = {Zhang, Yongchang and Huang, Jie and Ni, Yunhe and Zhou, Ningyu and Zhao, Guangyu and Liu, Ruixun and Huang, Rui and Wu, Yidi and Zhang, TianZe and Jiang, Zixiang and Li, Xiang and Shen, Kevin and Zhong, Yiwu},
  journal = {arXiv preprint arXiv:2611.34849},
  year    = {2026}
}
```

## 🙏 Acknowledgements

Real2SimLab builds on XPolicyLab, Harbor, mini-SWE-agent, Blender, ManiSkill and SAPIEN, and on the official
code and data of Physics-IQ, Video2World, 4DCodeBench, PhysCodeGym, PhySimCode and IRIS. We thank the
authors and maintainers for their open-source contributions to the community.

## 📄 License

Apache-2.0. Third-party code and data are never bundled; see [NOTICE](NOTICE). Product names
and logos are trademarks of their owners and only identify the agents and benchmarks the Lab
runs; no endorsement is implied.

## 📬 Contact us

**Homepage:** [real2simhub.github.io](https://real2simhub.github.io/)

<table>
  <tr><th>WeChat group</th><th>WeChat</th></tr>
  <tr>
    <td><img src="https://real2simhub.github.io/assets/img/connect/wechat-group.jpg" width="180" alt="WeChat group QR code"></td>
    <td><img src="https://real2simhub.github.io/assets/img/connect/wechat-personal.jpg" width="180" alt="WeChat QR code"></td>
  </tr>
</table>

**Contact us by email:** [jayhuang303hh@gmail.com](mailto:jayhuang303hh@gmail.com)
