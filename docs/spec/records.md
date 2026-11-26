# Records and run directories

Every record carries a `spec` and has a JSON Schema: [r2s-run](schemas/r2s-run.schema.json),
[r2s-build](schemas/r2s-build.schema.json), [r2s-agent-run](schemas/r2s-agent-run.schema.json),
[r2s-task-result](schemas/r2s-task-result.schema.json), [r2s-results](schemas/r2s-results.schema.json)
and [r2s-experiment](schemas/r2s-experiment.schema.json) (all schemas: the table in the
[specification](index.md)). Records never hold credentials. Paths inside a run directory are
relative to it, except the paths a record keeps as the command was given them, which may be
host paths. `experiment.json` is the experiment as given: its task paths as the experiment
file writes them (`r2s eval` makes `--tasks` absolute), and where each task was found,
relative to the run directory. It and each build record's `options` hold the method options
that name files (`path`, `checkpoint`) as given to `r2s eval`, or made absolute from the
experiment file's directory by `r2s run`.

## Run directory

```
<run>/
  experiment.json                      the resolved configuration and task locations
  tasks/<benchmark>/<selection>/       tasks built from benchmarks, one directory per selection
  builds/<task>/<method>/trial-<k>/
    build_record.json                  r2s-build/1.0
    visible/                           the copy of visible/ the method worked from
    work/agent/agent_record.json       r2s-agent-run/1.0 (agents)
    work/agent/trajectory.jsonl        the agent's redacted event stream (mini_swe:
                                       trajectory.json, mini-swe-agent's one JSON document)
    work/workspace/                    the agent's workspace (solution/, output/)
    twin/                              the frozen twin package
  runs/<task>/<method>/trial-<k>/seed-<s>/<run|rerun|native>/
    run_record.json                    r2s-run/1.0 (run, rerun)
    output/  log.txt                   (native: the official scorer's files and
                                        native_report.json, its report as the Lab read it)
  results/<task>/<method>/trial-<k>/seed-<s>.json     r2s-task-result/1.0
  results.json                                        r2s-results/1.0
  report/  index.html  report.md  results.csv  tasks/  media/
  local-media/                         videos of recordings that may not be redistributed
                                       (redistributable = false) and overlays drawn on them:
                                       the report plays them, but they are not in report/
  state/budget.json  state/status.json  state/lock  r2s.log
```

A run's `log.txt` (an agent's `sandbox.log` too) holds at most 16 MiB of what the program
printed: its beginning and its last MiB, with a line `[r2s] N bytes of output left out here`
between them. An agent's `trajectory.jsonl` and `stderr.log` keep at most 256 MiB each, in
whole lines, and say what they left out in a line `{"type": "r2s.output_left_out", ...}` (a
line longer than 8 MiB is replaced by `{"type": "r2s.line_left_out", "bytes": N}`); the usage
a record adds up from the events of a trajectory (Codex, OpenCode, Gemini CLI) then misses the
events left out. The agents whose harness runs on the host keep as much, in whole lines, with
a line `[r2s] N bytes left out here` where they left something out: what Harbor
(`harbor.log`) or the mini-swe-agent harness (`stderr.log`) printed, at most 16 MiB; Harbor's
trial, copied to `agent/trial`, and mini-swe-agent's `trajectory.json`, at most 256 MiB (a
longer `trajectory.json` is then no longer one JSON document; links in Harbor's trial are not
copied, and `harbor.log` names what was left out). Harbor's own job directory
(`work/harbor/jobs`) holds what Harbor wrote there. Each stays under the machine's file-size
limit (`ulimit -f`).

`<task>` is the task id with `/` written as `__`; an id that itself contains `__` also gets
`~` and a digest of the id, so two ids never share a directory. A name longer than 120 bytes
is cut and given the digest too (a file name holds at most 255 bytes).

Every step directory holds a `.key`: the content hash of the step's inputs (task, method and
options, trial, seed, twin fingerprint, the content id of the image actually used, backend,
limits, Lab version). Running the same command again reuses every step whose key matches and
whose files are intact (a build's frozen twin still has its recorded `twin_fingerprint` and its
session output its `session_output_fingerprint`, a run's output its `output_fingerprint`), and
redoes the others, so an interrupted experiment resumes where it stopped and gives the same
results. These fingerprints are always recorded (a session output's when it can be
fingerprinted, a run output's when the run succeeded): a step whose record lacks one it would
hold (edited, or written by a pre-release build of the Lab, whatever its `lab_version`) is
redone, not reused. A session output that could not be fingerprinted (a symbolic link in it) is
reused while it still cannot be; once it can be (a copy that followed its links, such as
`cp -rL`), its build is done again.
An official score (`native/`) is reused while the task, the frozen twin, the output
it scored, the scorer's options and the scorer itself (`NativeScorer.cache_identity`: for a
scorer that runs in an image, the image's content id) are unchanged, and while its
`native_report.json` is the file the Lab wrote (its digest is part of the stored key; an
edited or damaged report is computed again) and the scorer still reads it as a score (one it
now calls its own failure, `NativeScorer.failure`, is computed again). Steps that failed
outside the method (`infra_error`) are always redone. A `.key` that cannot be read (bytes that
are not text, a damaged disk) is no key: its step is redone. A benchmark selection is reused
only while every task it built still has the content hash its `built_tasks.json` records (a
task edited or damaged in the run directory has the selection built again; built again, the
same tasks reuse their builds and runs).

A run directory is only the Lab's: `r2s run` and `r2s eval` refuse an `--out` that holds
other files (it is new, empty or an earlier run directory, one whose `experiment.json` or
`state/status.json` the Lab wrote; files that only bear those names do not count) or that is,
contains or lies inside a task, program or checkpoint it evaluates, and only one command uses
it at a time (`state/lock`, held while the command runs; a second one stops with an error). A run that
cannot start (an unknown method, a missing image) leaves a directory it created as it was, and
an earlier run directory as it was. Each benchmark selection's directory lists the tasks it
built (`built_tasks.json`); a selection with a task removed is built again. Ctrl-C (or
`kill`) stops what is running, sandboxes included (on the `local` backend under Linux, also
what a program started in a session of its own), writes no result for a step it stopped, and
exits with status 130; the same command resumes.

`state/budget.json` holds what the run has spent against its budgets (`agent_runs`,
`cost_usd`, `cost_unknown_runs` once an agent run that started did not say what it cost,
and `wall_time_s`, the seconds its commands have run so far, counted across commands),
written atomically. A damaged one (also one without `cost_usd`) stops the next command with an
error naming it (what was spent is unknown, so it is never taken as nothing); a missing one is
counted again from the agent records the builds hold, each at the cost it says (unknown when it
says none); an agent record that cannot be read (damaged, cut, not a regular file), or is
missing while its build record names it, counts as a run of unknown cost too, and the Lab says
which. A build done again holds the record of its last run only, so such a count is at least
what was spent, and is said to be (after builds were done again, it is less): keep
`state/budget.json` with the run directory. An agent run is counted, as one of unknown cost,
before its agent starts, and settled once it says what it cost: one whose agent never began is
given back, and one stopped once it began (Ctrl-C, `kill -9`) or whose build failed on this
machine after the agent started (its record cut by a full disk, or never written; a harness
on the host, Harbor's or mini-swe-agent's, counts as started once its process exists, not when
it could not be spawned) stays one of unknown cost. A cost that is no amount of money (infinite, not a number, negative) is
unknown too, as is one that would make the total too large to be a number: `state/budget.json`
and agent records are strict JSON, never holding `Infinity` or `NaN`.

`state/status.json` says `running` from the start of a run until every result and
`results.json` are written, then `complete`. A task, program or checkpoint that changed while
the experiment ran (what was built and scored from it is of neither version) leaves it
`running`: the command ends with an error naming it, and the same command builds and scores
again what changed. Each row of `results.json` holds the SHA-256 of the
result file it was computed from (`result_sha256`). `r2s train export` refuses, and `r2s report`
warns about, a run directory that is not one complete set: still `running` (interrupted, or
being written), or with a result file missing or changed since `results.json` was written, or
without `state/status.json` or a row's `result_sha256` (edited or partly copied, or written by a
pre-release build of the Lab: the same command again finishes it). `r2s train export` reads each
reward from the very bytes it checked, and checks the result files again before it publishes
the dataset.

## `r2s-run/1.0` — one frozen run

`twin_id`, `twin_fingerprint`, `program_fingerprint`, `image` (reference and digest),
`backend` (name, version, whether isolated), `label` (`run`, or `rerun` for the
reproducibility check), `seed`, `limits` (timeout, CPUs, memory, GPU), `network` (`none`),
`started_at`, `duration_s`, `exit_code`, `status` (`succeeded`, `failed`, `timed_out`,
`invalid_output`, `infra_error`),
`problems`, `output_format`, `output_fingerprint`, `lab_version`, `attempt`.

## `r2s-build/1.0` — one build

`method` (label), `method_kind` (`agent`, `baseline`, `method`), `method_digest`,
`privileged`, `task_id`, `task_hash`, `trial`, `seed`, `delivered`, `failure`,
`twin_fingerprint`, `agent_record`, `session_output`, `session_output_fingerprint`, `network`,
`started_at`, `duration_s`, `lab_version`, `options`, `infra_error` (why the build failed outside
the method; such a build is redone on the next run).

## `r2s-agent-run/1.0` — one agent run

`agent`, `agent_version`, `model`, `reasoning_effort`, `instruction_template` (e.g.
`r2s-instruction/1.0`) and `instruction_sha256`, `task_id`, `task_hash`, `trial`, `image`,
`backend`, `network` (mode, allowlist, proxy log), `started_at`, `duration_s`,
`time_budget_s`, `exit_code`, `exit_status` (`completed`, `failed`, `timed_out`,
`infra_error`), `trajectory`, `stderr`, `last_message`, `usage` (tokens, when available),
`cost_usd` (when available), `redactions`, `delivered`, `failure`.

`agent_version` is what the agent's `--version` says where the agent ran: in its image (run
offline by Docker or bwrap), or on the host for the `local` backend and the agents whose
harness runs on the host (`mini_swe`, `harbor`); it is left out when it cannot be told there,
never taken from a CLI of the same name on the host. `trajectory` (like `stderr` and
`last_message`) names a file in `agent/` once it holds no injected credential; the report and
an exported dataset show the file it names: the agent CLI's events (`agent/trajectory.jsonl`),
or mini-swe-agent's trajectory, one JSON document (`agent/trajectory.json`). `cost_usd` is
there when the run says what it cost, as an amount of money (a finite number, not negative;
anything else a run says leaves it out: unknown): Claude Code and OpenCode report it,
`mini_swe` for a model LiteLLM has a price for, `fake_agent` costs $0; Codex, Gemini CLI and
Harbor runs never say it.

## `r2s-task-result/1.0` — one method on one task, trial and seed

`task_id`, `task_version`, `task_hash`, `benchmark`, `method`, `method_kind`, `privileged`,
`trial`, `seed`, `protocol`, `protocol_version`, `primary_metric`, `metric_unit`, `delivered`,
`delivery_failure`, `twin_fingerprint`, `reproduction` (status, run status, `score`: errors and
static errors per metric and the skill; `noise_floor`, `error_to_noise`, `repeat_skill`,
`reproducible`, `session_consistent`), `summary` (`reproduction_skill`, `reproduction_error`),
`native` (the official scorer's report, unchanged), `headline` (see
[Scoring](scoring.md#headlines)), `created_at`, `lab_version`, `infra_error` (why the Lab could
not score it; such a result has no skill or headline value and no statistic counts it).

## `r2s-results/1.0` — an experiment

`experiment`, `protocol_version`, `confidence`, `bootstrap_resamples`; per method the mean and
interval of the reproduction skill, the delivered rate, whether it is privileged, how its
agents were networked, and its scorecard: one `BenchmarkScore` per benchmark and headline
metric (headline metric, unit, direction, range, value with interval, scored and total results,
`infra_errors`: the results the Lab could not score, `delivered`: of the others, those whose
method delivered a twin, reference points, re-run rates, and
`rerun_compared` and `rerun_agreed`: the results whose frozen re-run could be compared and
those that agreed; see [Scoring](scoring.md#scorecard-and-aggregation)); the list of tasks;
one row per task result (with its `infra_error` and `result_sha256`); and the steps skipped (a
budget spent, a method not applicable to a task).

## Experiments

`r2s-experiment/1.0` is a TOML file:

```toml
spec = "r2s-experiment/1.0"
name = "synthetic-demo"       # r2s run without --out writes runs/<name>: one directory name

[tasks]
paths = ["tasks/"]           # searched for task.toml; relative to this file's directory
benchmarks = [{ name = "synthetic", select = ["bouncing_ball"], tiers = ["calibrated"] },
              { name = "physicsiq", select = ["mirror-ball-fall/center"], options = { variants = ["rerun"] } }]
include = ["synthetic/*"]                            # glob filters on task ids
exclude = []

[[methods]]
name = "oracle"

[[methods]]
name = "codex"
label = "codex-medium"            # unique; defaults to the name; letters, digits, _.:+- (≤ 100)
model = "gpt-5-codex"                                # any other key is a method option
reasoning_effort = "medium"

[run]
backend = "docker"            # docker | bwrap | local
seeds = [0]
trials = 1
concurrency = 4
agent_concurrency = 2
retries = 2                   # of infrastructure failures only
check_reproducibility = false

[budget]
max_agent_runs = 15
max_wall_time_s = 7200
# max_cost_usd = 20.0         # only for agents whose runs report their cost (not codex)
```

Relative paths (task paths, and the method options that name files: `path`, `checkpoint`) are
resolved against the file's directory and may contain `..`; absolute paths work too, and a task
path may also start with `~`.

`max_cost_usd` limits agents by what their runs say they cost (`cost_usd`): an experiment with
it refuses, before it starts, an agent whose runs never say it (`codex`, `gemini_cli`,
`harbor`), and once a run that started did not say it (`cost_unknown_runs`: a `mini_swe` model
LiteLLM has no price for, a run stopped before its agent reported), it starts no further agent,
since what was spent cannot be shown to be under the budget. `max_agent_runs` and
`max_wall_time_s` limit every agent.

## `r2s-training/1.0` — a checkpoint

`r2s train fit` writes a checkpoint directory: what the method learned, in its own files, and
the Lab's record of it, `training.json`:

`method` (the registered name), `method_version`, `options` (the method's options while
training), `dataset` (as given) and `dataset_digest` (SHA-256 of its `tasks.jsonl` and
`runs.jsonl`), `tasks` and `runs` (rows learned from), `task_ids`, `prints` (the file in the
checkpoint with the overlap prints of the training tasks, `training-prints.npz`), `seed`,
`started_at`, `duration_s`, `lab_version`, and `content_digest` (optional: the digest of every
file in the checkpoint but `training.json`; a checkpoint whose files no longer match it is
refused, since it is not what was trained). `r2s train fit` always writes `content_digest`: a
checkpoint without it (edited, or trained by a pre-release build of the Lab, whatever its
`lab_version`) is refused too, and must be trained again; so is one whose `prints` is not a
file in the checkpoint, or holds no print of one of its `task_ids`: its record was edited.

A method evaluated with `checkpoint=DIR` is not run on a task that overlaps one of these
training tasks (the step is listed among the skipped ones, `not held out`), and the build's
cache key includes the checkpoint's content.
