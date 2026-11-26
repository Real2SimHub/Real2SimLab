# Evaluate a coding agent

This tutorial runs a coding agent on one benchmark, then on all of them: the agent builds
programs in a sandbox, the Lab freezes and re-runs them offline, scores them the way each
benchmark does, and puts everything in one scorecard.

## 1. Pick the agent and give it credentials

| | Method | Agent | Image | Credentials (environment, injected at run time only) |
|:-:|---|---|---|---|
| ![](../assets/icons/codex.svg){ .r2s-icon } | `codex` | OpenAI Codex CLI (`codex exec`) | `r2s-agent-codex` | your Codex login (`~/.codex`), a custom provider from `~/.codex/config.toml` (its whole provider table, also when a `profile` chooses it), or `OPENAI_API_KEY` (`provider=api-key`) |
| ![](../assets/icons/claude-code.svg){ .r2s-icon } | `claude_code` | Claude Code (`claude -p`) | `r2s-agent-claude-code` (build it yourself: Claude Code is proprietary) | the first set of `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` (only with `base_url=`, its compatible endpoint) and `CLAUDE_CODE_OAUTH_TOKEN` (`claude setup-token`) |
| ![](../assets/icons/gemini-cli.svg){ .r2s-icon } | `gemini_cli` | Gemini CLI | `r2s-agent-gemini-cli` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| ![](../assets/icons/opencode.svg){ .r2s-icon } | `opencode` (experimental) | OpenCode (`opencode run`) | `r2s-agent-opencode` | the provider's key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`); `base_url=` and `api_key_env=` for an OpenAI-compatible endpoint, where the model runs whatever its provider part (`openai/<id>` runs as `custom/<id>`) |
| ![](../assets/icons/mini-swe-agent.svg){ .r2s-icon } | `mini_swe` | mini-swe-agent on the host, any LiteLLM model | the task's runtime image | the key named by `api_key_env` (default `OPENAI_API_KEY`) |
| ![](../assets/icons/harbor.svg){ .r2s-icon } | `harbor` | any agent Harbor supports (47 in Harbor 0.27, e.g. `agent=openhands`) | Harbor's | as Harbor expects (`agent_env`) |

```bash
r2s env build r2s-agent-codex          # once
```

Keys never enter a task, run directory, log or image: they are passed to the sandbox at run
time and agent output is redacted before it is written. An agent's endpoint is only what its
options name (`base_url=`, `api_base=` for `mini_swe`, the host's Codex provider for `codex`): a
host whose `ANTHROPIC_BASE_URL` or `OPENAI_BASE_URL` points the agent's provider elsewhere while
the method names no endpoint is refused before anything runs, and `r2s doctor` says so, rather
than its key sent to the provider's own API. `reasoning_effort` is passed to `codex` (and
`harbor`); the other adapters pass none to their CLI, refuse any other value than the default and
record null. A key given as a method option (for
example in `model_kwargs`) is refused, since options are recorded: name its variable instead.
See [Security](../security.md).

### Check an agent with your key

Only `codex` has been run for real by the maintainers; the other adapters are covered by unit
tests and by a check of every flag they pass against the pinned CLI's `--help`. To check one
yourself, build its image and run its live smoke test (one synthetic task, a few minutes of
model time). The tests need the `dev` extra, which `pip install -e .` leaves out: install it in
your clone with `pip install -e ".[dev]"` (see [Contributing](../contributing.md)).

```bash
r2s env build r2s-agent-claude-code
export ANTHROPIC_API_KEY=...            # or the agent's own variable
R2S_LIVE_MODEL=<model> pytest -m live tests/test_live.py -k claude_code
```

`R2S_LIVE_MODEL` is required for `opencode` and `mini_swe`: a LiteLLM name such as
`openai/<model>`, which both run on `R2S_LIVE_BASE_URL` (an OpenAI-compatible endpoint) when it is
set. `claude_code` takes `ANTHROPIC_BASE_URL` as its `base_url` when that is set. When the host user is root, agents run as uid 10001 inside their
container (some CLIs refuse to run unattended as root); their files are handed back afterwards.

## 2. Run

```bash
r2s bench build synthetic --out tasks --overwrite    # tasks/ may hold the quickstart's already
r2s eval --tasks tasks --include "synthetic/*/calibrated" \
    --method codex:reasoning_effort=medium \
    --method oracle --method static \
    --max-agent-runs 5 --out runs/codex
```

Method options follow the name: `NAME:key=value,key=value`; a `LABEL=` prefix names the column
(`fast=codex:reasoning_effort=low`). A value is read as JSON when it is JSON, and one that
holds a comma is written as JSON, the whole method in single quotes:
`--method 'codex:gpus="0,1",allowlist=["a.example","b.example"]'`. For several
configurations, seeds or trials, write an [experiment file](../spec/records.md#experiments) and
use `r2s run`.

The agent works in `/workspace`: `visible/` (the capture), `docs/` (the output format, the run
contract, the capture format, the SDK), `tools/r2s_twin.py` (the SDK) and an empty
`solution/`. It reads the task statement (`r2s task show TASK --statement`): the Lab's own, or
the benchmark's when the task brings one.

### Every benchmark at once

```bash
r2s run examples/experiments/unified.toml --out runs/unified
```

`unified.toml` lists a few tasks of each benchmark and one method; the run directory gets one
scorecard with a column per benchmark. Codex works in the image that matches the task's
runtime: `r2s-agent-codex` for Python tasks, `r2s-agent-codex-blender` for Blender scenes,
`r2s-agent-codex-physimcode` and `r2s-agent-codex-video2world` for those benchmarks; every
agent gets the benchmark's own agent image for 4DCodeBench (set `image=` to override), and
`mini_swe` and `harbor` use the task's runtime image. `claude_code`, `gemini_cli` and `opencode`
work in their own Python image (no Blender, ManiSkill or Matplotlib) on the Blender, PhySimCode
and Video2World tasks unless `image=` names another. Build the images you
need first (`r2s env build ...`); `r2s doctor` says which benchmarks this machine can run.

By default the agent's network is an **allowlist** (only its model API, through a logging
egress proxy: `agent/egress.jsonl`). `network=open` lifts it and is marked in every report;
`mini_swe` commands never have a network.

## 3. Budgets and resumption

- `--max-agent-runs N` (or `[budget] max_agent_runs`) stops starting agents after N runs;
  `max_wall_time_s` and `max_cost_usd` work the same way. Each counts what every command on
  the run directory spent (`state/budget.json`; for the wall time, the time those commands
  ran): running the same command again, or resuming it, does not start a new budget. Skipped
  builds are listed in the results. A build that failed around the agent is not a result (`infra_error`; which
  failures these are: [Evaluation](../guides/evaluation.md#read-a-score)), and running the same
  command again builds it again. Its agent run is not counted against N, and is tried again at
  once, only if nothing of the agent ran (its sandbox did not start, its CLI is not in its
  image, or the model API of `mini_swe` could not be used before it answered); an agent that
  had started (its sandbox killed or stopped later, or this machine failing after it, a full
  disk cutting its record) counts, and the same command does not run it again.
- `max_cost_usd` counts what each agent run says it cost. An experiment with it refuses, before
  it starts, an agent whose runs never say it (`codex`, `gemini_cli`, `harbor`: limit those
  with `max_agent_runs` and `max_wall_time_s`), and starts no further agent once a run did not
  say it (a `mini_swe` model LiteLLM has no price for): that run's cost is unknown, never $0
  (`cost_unknown_runs`). To go on, put what it cost in `state/budget.json` (`cost_usd`, without
  `cost_unknown_runs`), or run without `max_cost_usd`. What a run directory has spent is in
  `state/budget.json`; keep it with the directory.
- Running the same command again reuses every finished step; an interrupted run resumes.
  Change a method's options and its tasks are built again; if a budget stops that, those tasks
  are reported as not run rather than keeping the earlier options' scores.
  Tasks an experiment builds from a benchmark are rebuilt when the adapter's code changes;
  a task whose content is unchanged keeps its hash, so its finished builds are reused.

## 4. Read the results

`runs/codex/report/index.html` opens with the scorecard (a column per benchmark, its headline
metric, the reference points), then the methods and the tasks; each task page shows the
recorded video with each twin's trajectory drawn on it, plots, the official scorer's report and
each agent's redacted trajectory with its tokens, time, cost and exit status. `results.json`
and `results/<task>/<method>/...` hold everything in machine-readable form.

What to look at:

- **delivered**: did the agent hand in a valid program at all? The reason is recorded when not.
- **the headline against its references**: a Physics-IQ score means little alone; next to the
  static scene's and the second real take's, it says how far the agent got.
- **re-run reproduced**: the agent's own output in its session matches the frozen, offline
  re-run (`session_consistent`), and two re-runs agree (`reproducible`,
  `--check-reproducibility`). A delivery that only worked in the agent's session is caught
  here.

To compare agents and report the numbers (the same conditions, repeats, intervals, what to
write down), see [Evaluation](../guides/evaluation.md).

## 5. Add your own agent

Subclass `real2simlab.agents.AgentMethod`: give the image, the CLI's command line, the
credentials to inject and, optionally, how to read usage and cost from its event stream
(`usage`, `cost`: an experiment with `max_cost_usd` refuses an agent whose runs cannot say what
they cost); register it in your package's `pyproject.toml` under `real2simlab.methods`. The
workspace, sandbox, network, redaction, packaging and records are shared.
`real2simlab/agents/claude_code.py` is a complete example in about 120 lines.
