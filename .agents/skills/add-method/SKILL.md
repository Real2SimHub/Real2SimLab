---
name: add-method
description: Add a twin-building method to Real2SimLab — a baseline, a Real2Sim pipeline, or a coding-agent CLI adapter that builds twins in a sandbox. Use when asked to add, integrate or wrap a method, model or coding agent for evaluation.
---

# Add a method or an agent

A method builds a twin for a task: `build(context) -> BuildOutcome`. The Lab then freezes the
twin, re-runs it offline and scores it; a method never sees `hidden/`. Methods are plugins
(`real2simlab.methods` entry points). Read the closest existing one first:

| Kind | Example | Base class |
|---|---|---|
| a program-based method or baseline | `real2simlab/methods/program.py`, `methods/baselines.py` | `Method` |
| a coding-agent CLI in a container | `real2simlab/agents/claude_code.py` (about 90 lines) | `AgentMethod` |
| an agent harness on the host, commands in the sandbox | `real2simlab/agents/mini_swe.py` | `AgentMethod` |

## A method (`Method`)

1. Subclass `real2simlab.methods.Method`; define `name`, an options model (subclass
   `MethodOptions`; unknown keys are rejected) and `build(context)`.
2. Write the program into `context.work_dir`, then `package_twin(context, program_dir,
   method=self.name)`: it applies the task's program rules and the delivery convention and
   writes `twin.json`.
3. A method that fails to deliver returns `BuildOutcome(None, "reason")`; it does not raise.
4. Mark privileged baselines (those that read hidden data) as such, like `static`.
5. **A method that learns** also sets `trainable = True`, derives its options from
   `TrainedOptions` (they add `checkpoint`) and implements `train(context)`: read the exported
   dataset at `context.dataset`, write what it learns in `context.checkpoint_dir` (never
   `training.json` or `training-prints.npz`, which the Lab writes). `build` reads it back from
   `self.checkpoint`. Train with `r2s train fit NAME --data DATASET --out CHECKPOINT`; evaluate
   with `NAME:checkpoint=CHECKPOINT`, which skips every task that overlaps a training task.
   `real2simlab/benchmarks/synthetic/learned.py` (`nearest_twin`) is a complete example; test
   that a trained method beats the same method without training and is not evaluated on its
   training tasks (`tests/test_overlap.py`).

## A coding agent (`AgentMethod`)

1. Subclass `real2simlab.agents.AgentMethod`; set the class variables `name`, `agent_name`,
   `summary`, `agent_image` (an environment in `real2simlab/envs/__init__.py` with a Dockerfile
   under `docker/` that pins the CLI's version; `None` runs the agent in the task's runtime
   image; variants of the image on other task runtimes, such as `r2s-agent-codex-blender`, are
   picked automatically by `real2simlab.envs.agent_environment`), `default_allowlist` (the model API hosts the egress proxy may reach) and `Options`
   (a subclass of `AgentOptions`). Implement:
   - `command(context, workspace)`: the CLI's non-interactive argv inside the sandbox; its stdin
     is the task statement, its working directory `/workspace`; ask for a JSON event stream if
     the CLI has one;
   - `credentials(context)`: a `Credentials` with secret `env`, secret `files` (created in the
     sandbox's tmpfs home) and non-secret `public_env`, plus a `description` for the record.
     Secrets go only to the sandbox at run time; never write them to the workspace, records,
     logs or an image. Raise `SetupError` with a `fix=` hint when they are missing. Read a key
     from an environment variable that an option names (like `api_key_env`), never from an
     option itself: options are recorded, and one named like a credential is refused;
   - optionally `allowlist()` (e.g. a custom endpoint's host), `usage(trajectory)` and
     `cost(trajectory)` to read tokens and cost from the event stream, `version(backend, image)`.
     Without `cost` (or `reports_cost = True` for an agent that records its cost another
     way), an experiment with a cost budget (`max_cost_usd`) refuses the agent: what its runs
     cost would be unknown, never $0.
2. Everything else is shared: workspace preparation, sandbox, egress proxy and its log,
   redaction of output (and of the injected values in files the agent writes itself),
   collecting `solution/`, packaging, and the agent record.
3. Register it in `pyproject.toml` under `[project.entry-points."real2simlab.methods"]`, then
   reinstall (`pip install -e .`): entry points come from the installed metadata, so
   `get_method("<name>")` and `r2s eval --method <name>` only see it after that. Tests can
   instantiate the class directly.
4. Things a test without a model cannot show — what else the CLI contacts at start-up, whether
   it reads the prompt from stdin — are checked on the first real run: read `agent/egress.jsonl`
   (denied hosts) and `agent/stderr.log`, and mark the adapter experimental until then.

## Tests (no real model calls by default)

- The command line and the credentials (`tests/test_agents.py` has the pattern): secrets
  appear only in secret env, never in the public env or the command.
- An end-to-end run with a fake CLI or a scripted model in the real image if possible
  (`mini_swe` uses a deterministic model), checking that a delivered `solution/` is frozen and
  scored, and that a missing one is a recorded failure.
- A real run, if any, is a `@pytest.mark.live` test; `pytest` skips those by default.

## Document

Add the method to the table in `docs/tutorials/evaluate-agent.md` (and the README list), with
the credentials it needs; add a CHANGELOG line, and a `@pytest.mark.live` smoke test that users
with a key can run.

## Checks

```bash
ruff check real2simlab tests && ruff format --check real2simlab tests
pyright
pytest tests/test_agents.py
r2s eval --tasks <a synthetic task> --method <name> --max-agent-runs 1 --out runs/try   # if a key is available
```
