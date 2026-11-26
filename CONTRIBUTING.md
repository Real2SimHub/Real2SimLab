# Contributing to Real2SimLab

Thank you for helping. Real2SimLab is a unified platform for real-to-simulation: it runs any
method on any benchmark and scores each the benchmark's own way, with one specification. Most
contributions add something *next to* the core — a benchmark adapter, a method or agent, a
metric, an output format, a backend, an official scorer — and need no change to it.

<!-- --8<-- [start:setup] -->
## Set up

```bash
git clone https://github.com/Real2SimHub/Real2SimLab.git && cd Real2SimLab
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev,docs]"
r2s doctor
r2s env build r2s-python          # needs Docker
```

## Checks (all must pass before a pull request)

```bash
ruff check real2simlab tests examples scripts && ruff format --check real2simlab tests examples scripts
pyright                            # the core packages are strict
pytest                             # Docker and bwrap tests run when available
mkdocs build --strict
python -m build --wheel
```

`pytest -m live` runs tests that call real agents and cost money; they never run by default.

Test what goes wrong as well as what goes right: a new command, cache or step adds its cases to
`tests/test_failure_modes.py` (machine failures, interruption, inputs changed since a cache)
and `tests/test_cli_mistakes.py` (wrong paths, the same command twice, outputs inside inputs).

<!-- --8<-- [end:setup] -->

## Where things go

How the package fits together: [Architecture](docs/development/architecture.md).

| You add | Do this | Docs |
|---|---|---|
| a benchmark | subclass `BenchmarkAdapter`, register `real2simlab.benchmarks`; keep its import steps as plain functions | [Add a benchmark](docs/tutorials/add-benchmark.md) |
| a method or agent | subclass `Method` or `AgentMethod`, register `real2simlab.methods` | [Evaluate an agent](docs/tutorials/evaluate-agent.md) |
| a metric | subclass `Metric`, register `real2simlab.metrics` | [Scoring](docs/spec/scoring.md) |
| an output format | subclass `OutputFormat`, register `real2simlab.formats` | [Output formats](docs/spec/formats.md) |
| a backend | subclass `Backend`, register `real2simlab.backends` | [Clusters](docs/tutorials/clusters.md) |
| an official scorer | subclass `NativeScorer`, register `real2simlab.scorers` | [Scoring](docs/spec/scoring.md) |
| a delivery convention | an entry in `real2simlab/twin/conventions.py` | [Twins](docs/spec/twin.md) |

Plugins in your own package work the same way: declare the entry points in its
`pyproject.toml`. If a change seems to need the core, open an issue first.

## Rewards for training

A training reward is an evaluation score. `reward_of` (`real2simlab/train/reward.py`) turns a
task result into a number from 0 to 1 with the code that scores evaluations: the reproduction
skill under the reproduction protocol, or, under the native protocol, the official headline
scaled by its `range` (reversed when lower is better). The training tasks that Real2SimLab
builds today (the synthetic phenomena) use the reproduction protocol, so their rewards compare
trajectories. To reward with something else, add a metric or an official scorer; there is no
separate reward function to write.

| To reward with | Do this | Where the reward appears |
|---|---|---|
| another comparison with the reference | subclass `Metric` (lower is better; it reads `tracks` or a format convertible to them), register `real2simlab.metrics`, and in the `[scoring]` table of the training tasks list it in `metrics` and name it `primary`: the reward is the reproduction skill under it | datasets (`r2s train export`) and online rewards (`r2s train reward`, `score_solution`, Harbor's verifier) |
| a benchmark's own score | subclass `NativeScorer` whose `headline` sets `range` and `higher_is_better`, register `real2simlab.scorers`, and name it in training tasks (`split = "train"`, `protocol = "native"`): tasks of your own, or a benchmark's training split (its adapter lists `train` in `splits` and sets no `training_terms`) | datasets only |
| a benchmark's own score, online | `score_solution` runs the reproduction protocol only; running a native scorer there changes the core, so open an issue first | |

Training tasks never come from evaluation data: `r2s train export` refuses a task that is not
`split = "train"` or whose benchmark releases no training split or forbids training on its
data, and, with `--against`, leaves out every task whose recording appears in an evaluation
task ([Training](docs/guides/training.md)).

<!-- --8<-- [start:rules] -->
## Rules the code follows

- Strict data models: unknown fields are errors.
- Explicit failure: no silent fallbacks. A method that delivers no twin is a result, not an
  exception.
- Hidden task data never enter a method's sandbox.
- No third-party code or data in the repository or the wheel unless its license allows it;
  every adapter documents its source and license, and fetches what may not be redistributed.
- Public text (code, comments, docs) is in English.
- Specification changes are backward compatible, or bump a version with a migration note in
  `CHANGELOG.md`.

## Pull requests

One topic per pull request, with tests and documentation; describe what changes for users and
add a line to `CHANGELOG.md` under the unreleased version.
<!-- --8<-- [end:rules] -->
