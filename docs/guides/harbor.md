# Real2SimLab and Harbor

[Harbor](https://harborframework.com) is a framework for running agents on tasks in
sandboxes. Real2SimLab works with it in both directions:

| Direction | Command | Use it to |
|---|---|---|
| Lab task → Harbor task | `r2s export harbor` | run a Lab task with any agent Harbor supports, in Harbor's infrastructure, or publish it to a Harbor registry |
| Harbor agent → Lab method | method `harbor` | evaluate a Harbor agent with the Lab's frozen re-runs, scorecard and reports, next to other methods |

Harbor needs Python ≥ 3.12; Real2SimLab supports 3.11. Keep Harbor in its own environment and
point the Lab at its `harbor` command.

```bash
python3.12 -m venv .venv-harbor && .venv-harbor/bin/pip install harbor
.venv-harbor/bin/harbor --help                # check it; the commands below call it by this path
```

The commands on this page call `.venv-harbor/bin/harbor`; to type plain `harbor` instead,
activate that environment in another shell (`source .venv-harbor/bin/activate`).

## Export a task

`r2s export harbor` exports tasks whose solution is a `run.py` program (the `r2s` delivery
convention) and that the Lab scores itself (the reproduction protocol): the synthetic tasks,
IRIS, training tasks and tasks you write. The other benchmarks' tasks are refused, with the
reason: 4DCodeBench's use its `build.sh` convention, PhysCodeGym's a Blender scene, and
Physics-IQ, PhySimCode and Video2World are scored by their own scorers. Run them with
`r2s eval`. A task is exported only if it is valid as `r2s eval` requires (what
`r2s task validate` checks: no hidden data in `visible/` or `instruction.md`, no symbolic link,
a valid reference); the export is written from a copy of the task taken without following
links and validated, so the agent's environment holds exactly the validated `visible/` and
statement.

```bash
r2s env build r2s-python r2s-verifier
r2s export harbor tasks/bouncing_ball/calibrated --out harbor/bouncing_ball --solution oracle
.venv-harbor/bin/harbor lint harbor/bouncing_ball
.venv-harbor/bin/harbor run -p harbor/bouncing_ball -a oracle          # solve.sh writes the known twin
.venv-harbor/bin/harbor run -p harbor/bouncing_ball -a codex -m <model>
```

The exported task has:

- `instruction.md` — the Lab's task statement, for a workspace at `/app`;
- `environment/` — the agent's image (`FROM` the task's runtime image), with the visible data,
  reference documents and the SDK in `/app`; the network is Harbor's allowlist (the inference
  host only);
- `tests/` — a **separate, offline verifier** (`FROM ghcr.io/real2simhub/r2s-verifier`) holding the whole
  task and a copy of this Real2SimLab. It freezes `/app/solution`, re-runs it offline, and
  writes `reward.json`:

  | key | value |
  |---|---|
  | `reward` | the reproduction skill |
  | `reproduction_skill`, `delivered` | the main numbers |

  `result.json` next to it is the full `r2s-task-result/1.0`;
- `solution/solve.sh` — with `--solution oracle` (the task's truth generator) or
  `--solution DIR` (any program directory with `run.py`, or a twin package, whose `program/`
  is exported); anything else is refused before the task is written.

The hidden data exist only in the verifier image, and even there the solution cannot read
them: the verifier, which Harbor runs as root, makes the task's `hidden/` directory readable by
root only and runs the solution as an unprivileged user (`nobody`) who owns nothing but its
output directory. A solution that tries to copy the reference into its output fails and scores
0 (the tests check this in the verifier image). A missing, invalid or failing solution scores
0. When the verifier itself cannot score (run elsewhere without root, a dependency missing, a
run that fails outside the solution), it writes no reward, explains why in `error.txt` and
exits with status 1, so Harbor reports the trial as an error instead of a reward of 0;
`--no-isolation` scores a solution you trust as your own user. With `--self-contained`, both
Dockerfiles start from `python:3.11-slim-bookworm`, so the task builds without the `r2s` images (e.g. for a
public registry).

**The verifier's reward equals the Lab's own score.** It runs the same packaging, program rules,
run contract and scoring code. Checked on this machine: a physical twin of
`synthetic/bouncing_ball/calibrated` with a wrong restitution scored 0.7065343634894321 both as
Harbor's reward (`harbor run -a oracle`) and in `r2s eval` with the `program` method; the tests
check the same equality for every release.

## Evaluate a Harbor agent in the Lab

```bash
r2s eval --tasks tasks/ \
    --method "harbor:agent=terminus-2,model=<model>,harbor=.venv-harbor/bin/harbor" \
    --method oracle --method static --backend docker --out runs/harbor-agents
```

For each task the method writes a throwaway Harbor task with only the visible data (workspace
`/workspace`), runs `harbor run` with verification disabled, collects `/workspace/solution`,
and hands it to the Lab, which freezes it and re-runs it in its own sandbox like any other
agent's twin. Options: `agent`, `model`, `harbor`, `environment_type` (Harbor's, default
`docker`), `agent_kwargs`, `agent_env` (`{"KEY": "$ENV_NAME"}` is passed as
`KEY=${ENV_NAME}`, which Harbor reads from the environment, so the key never appears on a
command line; the variable must be set), `reasoning_effort`, `time_budget_s`, `solution` (a
program directory with `run.py` that Harbor's `oracle` agent writes with `solve.sh`, checked
before anything runs). Give keys
through `agent_env` references only: a literal value under a key's name in `agent_env` or
`agent_kwargs` (`OPENAI_API_KEY = "..."`, `api_key = "..."`) is refused, because options are
written into the run's records.
