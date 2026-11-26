# Related work

This page places the Lab among the benchmarks it runs and the platforms it resembles. For the
whole field (methods, benchmarks, datasets and surveys of real-to-simulation), see
[Awesome-Real2Sim](https://github.com/Real2SimHub/Awesome-Real2Sim), the curated survey released with the Lab.

## Real-to-simulation benchmarks

Real2SimLab is not another benchmark: it is the evaluation platform underneath them. Each benchmark below
asks for a program from a video and defines its own scorer; the Lab runs each under its own
protocol, with its official scorer, and adds what a cross-benchmark evaluation needs.

| Benchmark | Asks for | Its own score | Headline in the scorecard |
|---|---|---|---|
| [Physics-IQ](benchmarks/physics-iq.md) | the next 5 s of a real video | motion-mask IoUs and MSE against two real takes | Physics-IQ score |
| [Video2World](benchmarks/video2world.md) | a scene and the robot's actions | the executed package against the recording: functionality, geometry and dynamics, combined into V2WScore | V2WScore |
| [4DCodeBench](benchmarks/fourdcodebench.md) | a dynamic 3D scene | reproduction of the recorded scene | dynamic IoU |
| [PhysCodeGym](benchmarks/physcodegym.md) | a Blender scene program | pixel and motion metrics; LLM judges for physics | motion IoU |
| [PhySimCode](benchmarks/physimcode.md) | simulation code and an explanation | runnability, recovered parameters, LLM-judged physics | parameters within 20 % |

What the Lab adds to each of them:

- **One way to run any method.** The same sandbox, network allowlist, credential handling,
  redaction, budgets and records for every agent, instead of one agent setup per benchmark.
- **A frozen, offline re-run.** The delivered program is re-run offline and read-only, and what
  is scored is that re-run; the Lab records whether it reproduces what the agent showed.
- **A common record and scorecard.** Each benchmark's headline metric, with its reference
  points, in one result format, aggregated the benchmark's way.
- **Checked adapters.** Each adapter is validated against the benchmark's official pipeline or
  reference solutions; discrepancies in the benchmarks' own releases are reported on their
  pages.

## Harbor

[Harbor](https://harborframework.com) runs agents on tasks in general. Real2SimLab is specific to
real-to-simulation: its tasks carry captures, delivery conventions for scene and simulation
programs, official scorers as plugins, frozen re-runs and a cross-benchmark scorecard. The two
work together: `r2s export harbor` turns Lab tasks into Harbor tasks whose verifier gives the
Lab's score, and the `harbor` method runs any Harbor agent on Lab tasks. See the
[Harbor guide](guides/harbor.md).
