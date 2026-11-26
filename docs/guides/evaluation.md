# Evaluation

[Quickstart](../quickstart.md) and [Evaluate an agent](../tutorials/evaluate-agent.md) show how
to run an evaluation. This page is about running one whose numbers you can compare and report:
which tasks to pick, how to give every method the same conditions, how many runs to make, how
to read a score, and what to write down.

What the Lab guarantees for every evaluation: a method never sees a task's `hidden/` data; each
program is frozen and re-run offline before it is scored; each benchmark keeps its own scorer
and protocol; and every result goes into the same records. What it leaves to you: the choices
below.

## Choose the tasks

**A subset is not the benchmark's official number.** Each adapter covers part of its
benchmark, and its page lists what it leaves out. For example, Physics-IQ's official score is
defined over all 198 views and four generations; the adapter scores the views you build, with
one run per task. Report the task ids with the score, and compare methods only on the same
tasks. [`examples/experiments/unified.toml`](https://github.com/Real2SimHub/Real2SimLab/blob/main/examples/experiments/unified.toml)
selects a few tasks of every benchmark.

**Pick the variant that scores what you want to measure.** Some benchmarks score the output of
the agent's own session; a variant that scores the frozen re-run tells you what the program
does offline, without the agent:

| Benchmark | Variants (build option `variants`) |
|---|---|
| Physics-IQ, PhysCodeGym | `native`: the session's output, the official input; `rerun`: the frozen re-run's output; both with the official scorer |
| 4DCodeBench | `native`: the official scorer on the world the agent left in its session; `lab`: the frozen program re-run and scored with the Lab's metrics |
| Video2World | none: a package is data that Video2World's evaluator executes, so it is not re-run |

```toml
[tasks]
benchmarks = [{ name = "physicsiq", select = ["block-domino/center"], options = { variants = ["rerun"] } }]
```

**Synthetic tiers.** `calibrated` gives the camera, the metric scale and the scene's objects;
`video_only` gives the video and the text only, and its skill comes from the error in the
image. Report them separately (`--include "synthetic/*/calibrated"`): they are different
problems.

## Give every method the same conditions

- **The same budget.** An agent's time budget comes from each task's `task.toml` (the Lab's
  statement tells the agent "about N minutes"); the method option `time_budget_s` overrides it
  for every task. Give every agent the same one.
- **The same network.** By default an agent reaches only its model API (`network=allowlist`).
  A run with `network=open` is labelled in every table (`(open network)` after the method's
  name; the `network` column of `results.csv`); do not compare it with allowlisted runs
  as if they were the same condition.
- **An isolating backend.** Use `docker` or `bwrap`. `local` runs programs as you, where they
  could read the hidden reference; its run records say `isolated: false`.
- **Held-out tasks for trained methods.** A method given `checkpoint=DIR` is not run on a task
  that overlaps one of its training tasks; the step is listed as `not held out`
  ([Training a method](training.md#training-a-method)). Compare it with other methods on
  the tasks it was run on.
- **Reference baselines are not competitors.** `oracle`, `static` and `constant_velocity` read
  hidden data; they are labelled privileged (`(privileged)` after the name; the `privileged`
  column of `results.csv`) and serve as reference points.

## Repeat, and read the intervals

An evaluation has two kinds of repetition:

- **trials**: independent builds of the same task by the same method. For an agent, each trial
  is a new session, and agents vary from session to session.
- **seeds**: runs of one frozen program with different seeds. A deterministic program gives the
  same output for every seed; seeds matter for programs that sample.

```toml
[run]
trials = 3
seeds = [0]
check_reproducibility = true   # run each program twice with the same seed
```

Each headline comes with a **95 % interval from a two-level bootstrap**: tasks are resampled,
then each task's trials and seeds ([Scoring](../spec/scoring.md#headlines)). With few tasks the
interval is wide. In our [example run](../results.md), with one run per task, Physics-IQ over six
views is 52.2 [23.7, 83.6]. When two methods' intervals overlap widely, the run does not
separate them: add tasks or trials. `results.json` and `results/<task>/<method>/...` hold every
task's result, so you can also compare two methods task by task.

## Read a score

- **Next to its reference points.** A static scene scores 0 reproduction skill. Where a task
  has other recordings of the event (IRIS), `repeat_skill` is what another recording scores: a
  program above it reproduces this recording more closely than repeating the experiment does.
  On Physics-IQ the reference points are a static scene and the second real take; on
  Video2World, the score of the reference solution it ships.
- **Delivered, and not applicable.** A method that hands in nothing, or whose program fails,
  gets the benchmark's lowest result for that task, and the reason is recorded: reproduction
  skill 0 on the Lab's protocols, and on a benchmark with its own scorer that scorer's result
  for nothing (0 on PhysCodeGym, 4DCodeBench and Video2World; on Physics-IQ, the worst value
  of every metric; on PhySimCode, the task's true parameters counted as not recovered). Failing the
  hard tasks never raises a score. A benchmark's own rules still apply where they score more
  than the program: PhySimCode's parameter recovery scores the explanation a method hands in
  even when its program fails (its runnability is reported next to it). A method that cannot
  deliver for a task (a baseline on a Blender task) is listed as not applicable, not as 0. A
  failure of the machine (a container that did not start, an image that could not be pulled,
  a full disk, an output it could not read: no permission, an I/O error, a stale network file
  handle) is not a score at all: the result is marked `infra_error` and left out of the
  scorecard. For an agent this includes a sandbox that did not start, was killed from outside
  it or stopped running its commands, an agent CLI its image does not have, Harbor that could
  not set up the agent's environment, and a `mini_swe` run whose model API could not be used
  before it answered at all (a rejected key, a model that does not exist, a provider that is
  down, unreachable or rate-limiting). A prompt the model's API refuses (too long for the
  model's context, against its content policy) is the method's result, and so is a model API
  that fails a CLI agent (codex, claude_code, gemini_cli, opencode): the Lab does not tell that
  apart from the agent's own failure.
- **The re-run checks.** `session_consistent` says the frozen re-run gave the agent's own
  output; `reproducible` says two re-runs agreed. A delivery that only worked in the agent's
  session shows up here.
- **The cost.** The report gives each agent's tokens, time and cost (unknown for runs that
  do not say it: Codex, Gemini CLI and Harbor runs, a `mini_swe` model LiteLLM has no price
  for), and its redacted trajectory.

## Report it

Write down enough for someone else to run the same evaluation:

- the Lab's version and the task ids with their versions (`task_hash` in every result);
- each adapter's pinned code and data revisions (on its page) and the variant;
- each method's options: the agent and its version, the model, reasoning effort, budget,
  network;
- the backend, trials and seeds;
- the scorecard: each headline with its interval and reference points, the delivered rate, and
  the re-run counts.

The run directory holds all of it: `experiment.json` (the experiment as given), `results.json`,
every build and run, and the report (`report/index.html`, `report.md`, `results.csv`;
`r2s report RUN_DIR` writes them again). Records never hold credentials, and agent output is
redacted before it is written ([Security](../security.md)); `experiment.json` and the build
records keep the paths the command was given, which may be host paths
([Records](../spec/records.md)).
Every result carries the `task_hash` of the task it was scored on, and a run directory never
reuses a result of an earlier version of a task.
