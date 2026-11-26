# AGENTS.md

Guidance for coding agents (Codex, Claude Code, Gemini CLI, OpenCode, ...) working in this
repository. Humans: see [CONTRIBUTING.md](CONTRIBUTING.md).

## What this is

Real2SimLab is a unified platform for real-to-simulation. A method (a coding agent, a
Real2Sim pipeline, a hand-written program) watches a recording of a physical event and delivers a
program that simulates it, a *twin*. The Lab runs every method the same way on every benchmark it
adapts (sandboxed, network-limited, recorded), freezes the delivery and re-runs it offline, and
scores it with the benchmark's own official scorer, or with the Lab's reproduction metrics for raw
datasets. Every result carries the benchmark's headline metric, so results from all benchmarks
line up in one scorecard. For training, it generates training tasks with exact truth (kept apart
from every evaluation task), exports datasets, and scores any program as a reward with the
evaluation's own code (`r2s train`, `real2simlab/train/`); it does not train models itself. It
is a measurement tool: it does not rank methods or host leaderboards.

## Environment and commands

Python ≥ 3.11 and ffmpeg. Docker for the `docker` backend (and for the Docker tests); bubblewrap
for the `bwrap` backend (Linux).

```bash
pip install -e ".[dev,docs]"                         # install for development
r2s doctor                                           # what this machine can run, with fixes
r2s demo                                             # end-to-end smoke test (about 30 s)

ruff check real2simlab tests examples scripts        # lint
ruff format --check real2simlab tests examples scripts
pyright                                              # types (core packages are strict)
pytest                                               # tests; Docker/bwrap ones skip when unavailable
pytest -m "not docker and not bwrap"                 # quick: no containers
mkdocs build --strict                                # documentation
python -m build                                      # sdist and wheel
```

Tests marked `live` call real models and are never run by default. Generated files have sync
tests: after changing the CLI, run
`python -c 'from pathlib import Path; from real2simlab.spec.docgen import write_cli_reference; write_cli_reference(Path("docs/reference/cli.md"))'`;
after changing a specification model, `r2s spec schema --out docs/spec/schemas`.

## Where things are

- Architecture (packages, dependency direction, data flow, isolation, extension points):
  [docs/development/architecture.md](docs/development/architecture.md).
- The specification (tasks, captures, twins, formats, scoring, records):
  [docs/spec/](docs/spec/index.md).
- Benchmark adapters: `real2simlab/benchmarks/<name>/`, documented in `docs/benchmarks/<name>.md`.
- Agents: `real2simlab/agents/`; runtime images: `docker/<image>/Dockerfile`.

## Rules

1. **Hidden data is never mounted** into an agent's or a twin's sandbox, and nothing derived
   from `hidden/` (the reference, repeats, what official scorers read) may reach `visible/`, a
   statement or a workspace. Task validation checks for leaks; keep it passing.
2. **Add, don't change.** New benchmarks, methods, agents, metrics, formats, backends, scorers
   and CLI commands are plugins (entry points in `pyproject.toml`); a new delivery convention is
   an entry in `real2simlab/twin/conventions.py`.
   Specification 1.0 is frozen: within 1.x only add optional fields and objects; never rename,
   remove, or change the meaning or unit of anything.
3. **Fail explicitly.** Missing inputs, wrong formats and broken environments raise clear errors
   with a fix (`SetupError(message, fix=...)`); never degrade silently.
4. **Licenses first.** Check the license of third-party code and data before using it. Download
   at a pinned revision at run time; never commit or package third-party benchmark data.
5. **Credentials** exist only in the environment at run time: never write them to files, logs,
   records or images. Agent output is redacted before it is written.
6. **Public text is English** (code, comments, docstrings, docs, README), except `README_zh.md`.
7. **Test the failure paths, not only the happy path.** For a new command, cache or step, add
   its cases to `tests/test_failure_modes.py` (a machine failure is `infra_error`, not a score;
   interrupted and resumed; an input changed since it was cached) and
   `tests/test_cli_mistakes.py` (a wrong or empty path, a file for a directory, the same command
   twice or at once, an output inside an input or in the user's own directory): each ends in
   one `error:` line with a fix, never a traceback, a score of 0, or a deleted user file.
8. **Before committing**, run ruff (check and format), pyright, pytest and
   `mkdocs build --strict`; regenerate generated files when needed. Add a CHANGELOG line for
   user-facing changes.

## Skills

Step-by-step procedures in `.agents/skills/`:

| Skill | Use it when |
|---|---|
| [add-benchmark](.agents/skills/add-benchmark/SKILL.md) | integrating an external benchmark or dataset: license check, survey, protocol, adapter, reference validation, tests, docs |
| [add-method](.agents/skills/add-method/SKILL.md) | adding a baseline, a Real2Sim pipeline or a coding-agent CLI as a method |
| [write-tasks](.agents/skills/write-tasks/SKILL.md) | turning a recording or another benchmark's data into validated task directories |

Each skill was tried by a coding agent on a real addition: the example task built from a
recording (`examples/tasks/puck_slide`), the `opencode` agent adapter (experimental), and a
survey of a second IRIS phenomenon (two pendulums), which found its reference trajectories
unreliable, so no tasks were built from it.
