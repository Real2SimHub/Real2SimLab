# Training

Real2SimLab ships no trainer and no weights. It gives training what a trainer cannot easily
get elsewhere, from the same code that evaluates:

| | Command | What you get |
|---|---|---|
| **Training tasks** | `r2s train tasks`, `r2s bench build NAME --split train`, your own | Lab tasks marked `split = "train"`: the synthetic phenomena with sampled hidden parameters, a benchmark's own training split, or any dataset you convert |
| **Datasets** | `r2s train export` | training tasks and the runs on them as one directory: inputs, labels, delivered programs with their rewards, agents' redacted trajectories; checked against the evaluation tasks |
| **Training a method** | `r2s train fit` | a checkpoint of a method that learns, evaluated with `r2s eval` like any other method, on the tasks that do not overlap its training tasks |
| **Rewards** | `r2s train reward`, `score_solution` | one solution's reward: its evaluation score, computed by the evaluation's own code |
| **Environments** | `r2s export harbor` | a task (training tasks included) as a Harbor task whose verifier writes the same reward |

The rule behind all of them: a task is either for training or for evaluation, and nothing
that a method learns from may hold an evaluation task's recording.

The commands below use the README's names: `eval/tasks` for the evaluation tasks
(`r2s bench build synthetic --tier calibrated --out eval/tasks`), `train/tasks`, `train/data`
and `train/model`. None writes over tasks, a dataset or a checkpoint already there unless given
`--overwrite`: after the README's commands, give them another `--out`. The
[worked example](#example-a-method-that-learns) at the end needs nothing else and writes under
`example/`.

## Training tasks

A training task is an ordinary Lab task with `split = "train"` in its `task.toml`. Its
`hidden/` holds the labels (the reference output and, when the task has them, the hidden
parameters and a truth program); a method sees only `visible/`, as in an evaluation.

### Generated: the synthetic phenomena

```bash
r2s train tasks --count 200 --out train/tasks            # 200 samples of each phenomenon
r2s train tasks --count 50 --seed 1 --select pendulum --tier calibrated --out train/more
```

The same command and seed give the same tasks; a phenomenon or tier given twice (`--select`
and `--tier` repeat) is built once. A task is never written over unless you pass
`--overwrite`; after an interruption (Ctrl-C), run the same command with `--overwrite`: it
builds them all again (the tasks written before are not reused). `--overwrite` replaces only
the tasks the command writes: tasks an earlier command wrote to the same `--out` (another
`--count`, `--seed` or `--select`) stay, and are exported with it by `r2s train export
--tasks`; the command names them. It never replaces a directory of yours that merely holds a
task's names (`visible/`, `hidden/`) without a task's `task.toml`.

Each sample is one of the five [synthetic](../benchmarks/synthetic.md) phenomena with its
hidden parameters and initial state drawn uniformly from these ranges (the evaluation task's
value in brackets):

| Phenomenon | Hidden parameters | Initial state |
|---|---|---|
| `bouncing_ball` | restitution 0.45–0.92 [0.763] | release height 0.3–0.9 m [0.5] |
| `pendulum` | damping 0–0.5 s⁻¹ [0.137] | release angle 0.15–0.6 rad [0.35] |
| `incline_slider` | friction 0.18–0.42 [0.287] (kept under `setup` in `parameters.json`) | incline angle 0.42–0.6 rad [0.5] |
| `projectile` | drag 0.05–0.2 [0.113]; restitution 0.45–0.75 [0.583]; tangential 0.7–0.95 [0.837]; rolling 0.25–0.55 [0.371] | launch velocity x 1.6–2.3, z 2.1–2.8 m/s [2.0, 2.5] |
| `two_ball_collision` | restitution 0.6–1.0 [0.827]; rolling 0.15–0.4 [0.233] | launch speed 0.95–1.35 m/s [1.2] |

A draw is kept only if its objects stay on screen at least as well as in the evaluation task
(the same camera films it) and it is not a near copy of the evaluation task: at least one value
differs from the evaluation task's by more than 5 %. A kept draw whose video the
[overlap check](#keeping-training-data-apart-from-evaluation-tasks) cannot tell from the
evaluation task's is skipped too, and the command reports how many were. Sample *i* of a seed
is always the same, whatever the count (a skipped sample's index is missing). The tasks have
ids `synthetic-train/<phenomenon>/<tier>/s<seed>-<index>`, so their results never mix with the
suite's in a scorecard. Their `hidden/` holds the exact trajectories (`reference/`),
`parameters.json` (the hidden parameters and the initial state) and the truth program as a
frozen twin (`generator/`; its `program/` is a solution that scores 1 with `r2s train reward`).

### Your own training data

Any dataset can be turned into training tasks:

- **Tasks you write.** Build them with `TaskBuilder` as in [Write tasks](../tutorials/write-tasks.md)
  and set `split=Split.TRAIN` in their `TaskConfig` (`split = "train"` in `task.toml`). Give
  them their own benchmark name (the first segment of the id), not the name of a registered
  benchmark.
- **A benchmark adapter.** An adapter whose benchmark releases a training split, under terms
  that allow training on it, declares `splits = ("eval", "train")` and writes tasks with
  `split = "train"` when its `build` is called with `split="train"`. Then
  `r2s bench build NAME --split train --out train/tasks` builds them (and checks that every task
  it wrote is marked so). See [Add a benchmark](../tutorials/add-benchmark.md).

A task that claims `split = "train"` under the name of a registered benchmark that releases no
training split is refused by `r2s train export`.

**The six external benchmarks are evaluation sets.** They are released for evaluation, and the
Lab treats their tasks as evaluation tasks. Physics-IQ's terms forbid using its test data to
improve a score (the Lab never exports its tasks for training, even when asked, nor
PhysCodeGym's clips taken from it); IRIS's
recordings are CC BY-NC 4.0; and 29 of PhysCodeGym's 156 clips come from Physics-IQ (27) and
IRIS (2), so training on one benchmark's data can leak into another's results. The overlap
check below finds such clips.

## Keeping training data apart from evaluation tasks

`r2s train export --against EVAL_TASKS` compares every task to export with the evaluation
tasks and leaves out each one that overlaps them:

- **the same file**: a video, image or array of one task (in `visible/` or `hidden/`) that is
  also in the other;
- **the same recording**: a video that shows the same images as one of the other's, in the
  same order, at one time offset: a copy, a re-encoding, a resized copy, a clip cut from it, a
  copy at a lower frame rate or with another brightness. Frames are reduced to 32 × 32 colour
  thumbnails, and only frames where something moves count as evidence
  ([`real2simlab.task.overlap`](https://github.com/Real2SimHub/Real2SimLab/blob/main/real2simlab/task/overlap.py)
  documents the thresholds).

Measured on our machine. We made eight copies of each of the five synthetic evaluation videos
(320 × 240, 60 to 90 frames): re-encoded at CRF 28 and at CRF 40, at half size, without the
first 20 frames, a clip from the middle, at 15 fps, brighter, and in TV range. The check found
36 of the 40 copies, each matched to its own evaluation task only. The four misses are
re-encodings at CRF 40. On real data: PhysCodeGym's 27 clips taken from Physics-IQ, each
compared with the 81 Physics-IQ videos of those scenarios (the conditioning and testing videos
of take 1 and the testing video of take 2), matched both take-1 videos of their own scenario
and no video of another scenario; 7 also matched the testing video of take 2, a second
recording of the same event that the check could not tell apart from the first.

What it does not find: crops, mirrored or rotated copies, and heavily degraded copies; nor,
reliably, a second recording of the same event (7 of the 27 above, not the other 20). What it
finds besides copies: videos that a 32 × 32 thumbnail cannot tell apart, such as two renderings
of a pendulum released a few degrees apart, or a second take of an event that repeats closely;
it errs on the side of leaving data out. Check data from other sources yourself as well.

Each export writes `export.json` (what was compared, every task or result left out with its
reason; `content`: the SHA-256 of every file it wrote but `export.json` itself and the
dataset's card, `README.md`; and `executable`: which of those files are executable) and the
prints of the exported tasks (`overlap-prints.npz`). A method trained on the dataset carries
these prints in its checkpoint, so it is not evaluated on a task they overlap (below).

## Datasets

```bash
r2s eval --tasks train/tasks --method codex --method oracle --out train/runs   # optional
r2s train export --tasks train/tasks --runs train/runs --against eval/tasks --out train/data
r2s train export --runs train/runs --min-reward 0.9 --against eval/tasks --out train/best
```

```
train/data/
  README.md            counts, splits, licenses, the command that made it
  export.json          what was compared, and what was left out and why
  tasks.jsonl          one row per task
  tasks/<task>/visible/     the inputs: capture.json, the video
  tasks/<task>/hidden/      the labels: reference/, parameters.json, generator/
  runs.jsonl           one row per delivered program
  runs/<run>/<task>/<method>/trial-<k>/seed-<s>/program/          the program
  runs/<run>/<task>/<method>/trial-<k>/seed-<s>/trajectory.jsonl  the agent's events, redacted
                       (mini_swe: trajectory.json, mini-swe-agent's one JSON document)
  overlap-prints.npz   the overlap prints of the exported tasks
```

A task row holds the task id, split, benchmark, license, content hash, the statement an agent
gets, the paths of its video, capture and reference, `parameters` (inline) and
`truth_program`. A run row holds the method (and whether it is privileged, like `oracle`, which
reads the truth), the reward, the reproduction skill and headline, the run status, whether the
frozen re-run matched the agent's own output, the content hash of the task it was scored on,
the statement, the program and, for agents, the agent, model, token usage, cost and
trajectory. Use it for supervised learning (video → trajectories or parameters), for
fine-tuning on good programs (filter by `--min-reward`), or as rollouts with rewards. An
agent's statement and trajectory come from files it could write: each is exported only when it
is a regular file (a symbolic link or a pipe in its place is left out, never followed), and is
redacted once more on its way into the dataset (strings shaped like keys, and the values of the
exporting shell's credential variables).

What a dataset holds, exactly:

- **Training tasks only.** `--allow-eval-tasks` exports evaluation tasks too, and the dataset's
  README then says so; tasks of a benchmark whose terms forbid training on it (Physics-IQ) are
  refused even then.
- **Nothing hidden in the inputs.** A task whose `visible/` (what a method trained on the
  dataset sees) holds a symbolic link or the content of a `hidden/` file, also one that
  `capture.json` lists as a stream, or whose `instruction.md` (the row's statement) is a copy of
  one, is refused as `r2s task validate` refuses it, with the fix; `visible/` and `hidden/` are
  copied without following links and checked again as copied, with the `instruction.md` the
  statement is made from.
- **Each reward with the task version it was scored on.** A result is exported only with the
  task whose content hash it records, found where the run found it or among `--tasks` (so an
  archived run is exported by giving its tasks). A result whose task is missing or has changed
  since the run is an error; `--skip-unmatched` leaves such results out instead, and lists
  them in `export.json`.
- **Each reward with the program that earned it.** Runs whose directories share a name keep
  their own programs (each is then named with a digest of its location).
- **Each reward with the program that earned it, unchanged.** A result is exported only while
  the frozen program in the run directory still has the fingerprint the result records; one
  edited, synced over or replaced since, or a result that records no fingerprint, is an error
  that names the program: the same `r2s eval` or `r2s run` command again on the run directory
  scores it again (`--skip-unmatched` leaves it out).
- **Complete runs only.** A run directory is exported only when it finished (`state/status.json`
  says `complete`) and every result file is the one `results.json` was computed from (the
  SHA-256 its row holds; a row without one cannot show it, and is refused too); an
  interrupted run or an archive missing a result is refused, with the command that finishes it.
  Each reward is read from the very bytes so checked.
- **Delivered programs.** A result with no delivered program is left out, unless
  `--include-undelivered` (its `program` is then `null` and its reward 0). A result the Lab
  could not score (`infra_error`: the sandbox did not start, the host could not copy the
  delivery, the official scorer failed) is not a reward and is always left out, with or
  without `--include-undelivered`.
- **Labels as they are.** A task's `hidden/parameters.json` (the row's `parameters`) is
  exported only as a JSON object: one cut short, not text or not an object stops the export
  with an error that names the file (`r2s task validate` reports it too). `nearest_twin` names
  the `tasks.jsonl` line whose labels are not a truth it can learn from.
- **All or nothing.** The dataset is written next to `--out` and moved into place when it is
  complete; with `--overwrite`, an existing dataset is replaced only then. What an export
  killed outright (`kill -9`, the out-of-memory killer) left next to `--out` is removed by the
  next export to the same `--out` on that machine, once no process holds it. `--overwrite`
  replaces only an exported dataset (a directory with the `export.json` an export wrote, not
  any file of that name) or an empty directory, and `--out` may not be, contain or lie inside
  an input; nor may the thumbnail cache the overlap check writes (`$R2S_CACHE_DIR/thumbnails`)
  lie inside a task it reads, wherever the task is (a run's task too). Just before the dataset
  is moved into place, every input is checked again: a task, a result file or an `--against`
  evaluation task that changed while the export ran (an editor, a sync, another command) stops
  it, with nothing published (export again); so does a task or a program whose copy failed
  because it changed meanwhile, with the same error.
- **As exported.** `r2s train fit` checks every file against `export.json` before it trains,
  but the card `README.md`, which no command reads: a dataset edited, synced over or partly
  copied since is refused, naming the file (export it again), as is one in which a file's
  executable bit was cleared or set (a zip archive or a sync that keeps no file modes: a
  program that runs a helper behaves differently without it) or to which a symbolic link was
  added; a dataset whose `export.json` has no `executable` (exported before it was recorded)
  has its files checked, not their modes. A malformed `tasks.jsonl` row (or one that is not
  UTF-8 text) is named by its line. An `export.json` that records no content is refused too: it
  was edited, or written by a pre-release build of the Lab (a dataset exported by a development
  build before the 1.0.0 release must be exported again). Its overlap prints must also hold
  every recording of its tasks, by content.

## Training a method

A method that learns from data (its class sets `trainable = True`) is trained on a dataset
with `r2s train fit`, and evaluated with the checkpoint it wrote:

```bash
r2s train fit nearest_twin --data train/data --out train/model
r2s eval --tasks eval/tasks --method nearest_twin:checkpoint=train/model --backend bwrap --out eval/run
```

The checkpoint directory holds the method's own files, `training.json`
([`r2s-training/1.0`](../spec/records.md#r2s-training10-a-checkpoint): the method, its
options, the dataset and its digest, the training task ids) and `training-prints.npz`. An
evaluation with `checkpoint=DIR` skips every task that overlaps a training task, and lists it
among the skipped steps (`not held out`) with the reason. `r2s train fit` therefore needs a
dataset with the prints of all its tasks (every export by `r2s train export` has them). A
checkpoint is written next to `--out` and moved into place when training ends, so it is
complete or absent (what a training killed outright left there is removed by the next one
for the same `--out` on that machine); `--overwrite` replaces only a checkpoint (a directory
with the `training.json` `r2s train fit` wrote), and only then. A dataset replaced or changed
while `r2s train fit` checks it or the method trains on it (`r2s train export --overwrite` to
the same directory, an edited file) writes no checkpoint, and the error says so, also when
the method failed on a file that was no longer there: train again once nothing writes to it.
Training again in the same directory is noticed: the checkpoint's content fingerprint is part
of every build's cache key. A checkpoint whose files changed after `r2s train fit` wrote them
(`content_digest` in `training.json`), or
whose model cannot be read, is refused when the evaluation starts, before anything runs; so is
one whose record cannot show what was trained: no `content_digest` (`r2s train fit` always
writes one; a record without it was edited, or written by a pre-release build of the Lab: train
it again), prints that are not a file in the checkpoint, or prints that leave out one of its
training tasks.

To make your own method trainable, give it options that derive from `TrainedOptions` (they
add `checkpoint`), set `trainable = True`, and implement `train(context)`: read the dataset at
`context.dataset`, write what you learn in `context.checkpoint_dir`. In `build`, read it back
from `self.checkpoint`. [`nearest_twin`](https://github.com/Real2SimHub/Real2SimLab/blob/main/real2simlab/benchmarks/synthetic/learned.py)
is a complete example in about 300 lines.

## Rewards

```bash
r2s train reward --task train/tasks/synthetic-train/pendulum/calibrated/s0-00003 \
    --solution my_program/ --backend docker
```

```python
from real2simlab.train import score_solution

scored = score_solution(task_dir, solution_dir, backend="docker")  # or a twin package
scored.reward          # 0 to 1
scored.result          # the full r2s-task-result/1.0, as an evaluation records it
```

The solution is packaged as a twin (the task's program rules apply), run once, frozen and
offline, and scored with the task's protocol: the same code as `r2s eval`, so a reward and an
evaluation of the same program are equal (the tests check this). The reward is the
reproduction skill; nothing delivered, an invalid program, a failed run or an invalid output
scores 0. A failure outside the solution gives **no** reward: when the sandbox did not start
or was killed from outside it (bwrap outside a systemd scope, or the Docker client, ended by a
signal), the frozen copy of the solution was changed from outside its sandbox while it ran on
`docker` or `bwrap`, or this machine's storage stopped the program's writes (a file the Lab
finds after the run at the file-size limit, a full disk; also when the program caught the
failed write, or retried it until its time limit), `score_solution` raises
`RewardUnavailable`; when this machine could not copy or package the
solution (a full disk, a file-size limit), it raises `InfrastructureError`. In both cases
`r2s train reward` exits with status 1 and prints why, so a trainer never penalises a program
for a broken machine. `--work DIR` keeps the run's files; scoring again in the same directory
first clears what the earlier scoring wrote there (a directory with other files is refused,
and it may not hold the solution or the task). A task edited or written again while the
solution is scored gives no reward either, and the error says so (score again once nothing
edits it), also when the scoring failed on a file that was no longer there. A solution that
holds a file no
program may deliver (a named pipe, a device, a file it cannot read) is not delivered and scores
0. The solution directory is one program: its `run.py` is the entry point whatever its
subdirectories hold, and `r2s eval --method program:path=` takes the same directory the same
way. An empty `--solution` (an unset variable) is refused, as a path that is not there is: it
is not the current directory. A solution or a task changed while it is scored (an editor
saving, a sync tool) gives no reward either: the reward would be of what it held before, so
`score_solution` raises `SetupError` and `r2s train reward` exits with status 1; score it again
once it is no longer being edited. Use `docker` or `bwrap` for
programs a model wrote: with `local` they run as you and could read the hidden reference.
These rewards are for the reproduction protocol: a task scored by a benchmark's own scorer (the
native protocol) is refused, with the command that evaluates it. In an exported dataset, a run
on such a task carries the official headline scaled to [0, 1] as its reward. To reward with
another score, see [Rewards for training](../contributing.md#rewards-for-training).

What the Lab cannot tell from the solution's own failure still scores 0 (see
[Scoring](../spec/scoring.md)): a signal to the program itself (inside its sandbox,
`docker kill`, any kill on `local`) or to bwrap inside a systemd scope, a change of the frozen
copy on `local`, and a write the file-size limit stops in the sandbox's own `/tmp` (a tmpfs on
`docker` and `bwrap`, which also holds `HOME`) or to an anonymous temporary file, which leaves
no file to find. So score solutions on a machine without a file-size limit (`ulimit -f` prints
`unlimited`).

**Reinforcement learning.** `r2s export harbor` writes a task whose solution is a `run.py`
program, a training task included, as a [Harbor](harbor.md) task whose separate verifier
writes this reward (and no reward, with an error Harbor reports, when it cannot score).
Checked on our machine: a training task of the bouncing ball, exported with another sample's
truth program as its solution and run with `harbor run -a oracle`, got the reward 0.92010 from
Harbor's verifier, the same as `r2s train reward` gave the program. Harbor lists integrations
with reinforcement-learning frameworks that run Harbor tasks as environments
([SkyRL](https://github.com/NovaSky-AI/SkyRL/tree/main/examples/train_integrations/harbor),
[Tinker](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/harbor_rl)).
We have not trained with them ourselves.

### Using the rewards

How the pieces map onto common ways of training a model that writes programs. We have not
trained a model with any of them; this says what the Lab provides for each and what to watch.

- **Online reinforcement learning.** Score each new program when it is generated, with
  `score_solution` or a Harbor task's verifier; the rewards stored in an exported dataset are
  for offline use. A reward is terminal: one number for a whole program, or for a whole agent
  episode. Group-relative methods without a value model (GRPO, RLOO) fit this: sample several
  programs for one task and use the group's mean reward as the baseline. Methods with a learned
  value model (PPO) also work with a terminal reward, but need a critic, usually as large as the
  policy.
- **Rejection-sampling fine-tuning.** Generate programs, score them, keep the good ones
  (`r2s train export --runs ... --min-reward 0.9`), fine-tune on them, and repeat.
- **Preference learning.** Programs for the same task with different rewards make preference
  pairs (e.g. for DPO); `runs.jsonl` gives each program's task and reward.
- **Supervised learning.** The labels of a training task (`reference/`, `parameters.json`,
  the truth program) train a model from video to trajectories or parameters, as `nearest_twin`
  does. Fine-tuning on truth programs teaches one of many programs that reproduce the event;
  a reward accepts any of them.

What to watch:

- Drop a sample that raised `RewardUnavailable` instead of counting it as 0: it says nothing
  about the program, and a 0 would distort its group's statistics.
- Score programs a model wrote with `docker` or `bwrap`, never `local`.
- Reward workers scoring at the same time on one machine with `bwrap` each get CPUs of their
  own (the task's `cpus`) while there are enough, also across processes. Run no more at once
  than that: beyond it they share CPUs (a warning says so), and a program that then runs into
  its time limit scores 0 for the machine's load, not for itself.
- Near 1, the rewards of a group can differ by little (0.990 against 0.995); dividing
  advantages by the group's standard deviation then magnifies these differences.
- A synthetic training task has exact truth, so its reward has no measurement error. On a
  recording, the reference is itself measured: where a task has repeats, differences below its
  noise floor do not mean a better program.

## Example: a method that learns

`nearest_twin` is a small learned Real2Sim method for the synthetic phenomena that goes
through the whole loop. It sees what any method sees (the video and the scene's objects),
tracks each object by its colour, and predicts the hidden parameters and initial state from its
nearest training tasks; then it writes a program that simulates the phenomenon with them, and
the Lab evaluates the programs like any other method's. The example trains it with 5, 10, 20
and 40 training tasks per phenomenon, and writes everything under `example/`, apart from the
`train/` and `eval/` of the commands above and of the README.

```bash
r2s train tasks --count 40 --tier calibrated --out example/train-tasks
r2s bench build synthetic --tier calibrated --out example/eval-tasks
r2s train export --tasks example/train-tasks --against example/eval-tasks --out example/data
r2s train fit nearest_twin:count=5 --data example/data --out example/model-5
r2s train fit nearest_twin:count=10 --data example/data --out example/model-10
r2s train fit nearest_twin:count=20 --data example/data --out example/model-20
r2s train fit nearest_twin:count=40 --data example/data --out example/model-40
r2s eval --tasks example/eval-tasks \
    --method learned-5=nearest_twin:checkpoint=example/model-5 \
    --method learned-10=nearest_twin:checkpoint=example/model-10 \
    --method learned-20=nearest_twin:checkpoint=example/model-20 \
    --method learned-40=nearest_twin:checkpoint=example/model-40 \
    --method prior=nearest_twin:checkpoint=example/model-40,prior=true \
    --method fake_agent --method oracle --method static --backend bwrap --out example/run
```

Measured on our machine with the `bwrap` backend. There is one evaluation task per phenomenon,
so each mean is over five tasks; the interval is the scorecard's 95 % bootstrap interval.

| Method | Training tasks per phenomenon | Bouncing ball | Pendulum | Incline | Projectile | Two balls | Mean [95 % interval] |
|---|---|---|---|---|---|---|---|
| `static` (nothing moves) | — | 0 | 0 | 0 | 0 | 0 | 0 |
| `fake_agent` (fixed guesses) | — | 0.911 | 0.942 | 0.850 | 0.638 | 0.950 | 0.858 [0.743, 0.938] |
| `prior` (training mean) | 40 | 0.897 | 0.973 | 0.933 | 0.957 | 0.853 | 0.923 [0.880, 0.959] |
| `learned-5` | 5 | 0.854 | 0.954 | 0.829 | 0.977 | 0.934 | 0.910 [0.859, 0.959] |
| `learned-10` | 10 | 0.916 | 0.945 | 0.907 | 0.985 | 0.961 | 0.943 [0.918, 0.967] |
| `learned-20` | 20 | 0.949 | 0.945 | 0.913 | 0.981 | 0.973 | 0.952 [0.932, 0.971] |
| `learned-40` | 40 | 0.993 | 0.955 | 0.952 | 0.987 | 0.968 | 0.971 [0.957, 0.986] |
| `oracle` (the truth) | — | 1 | 1 | 1 | 1 | 1 | 1 |

**More training data gives better twins.** The mean skill rises from 0.910 with 5 training
tasks per phenomenon to 0.971 with 40. With 40 it is above the same method that ignores the
video and uses the training tasks' mean values (`prior`, 0.923), and above a simulator with
fixed guesses (`fake_agent`, 0.858).

**The prior is a strong baseline here.** The evaluation values lie near the middle of the
sampled ranges, so the mean is already close. With 5 training tasks the learned method is below
it, and on the pendulum it stays below it even with 40 (0.955 against 0.973).

**No training task reached the evaluation.** Generating the 200 training tasks skipped 9 draws
whose video looked like the evaluation task's (8 pendulums, 1 bouncing ball). The export then
compared the 200 tasks with the 5 evaluation tasks and found no overlap, and the evaluation
skipped no task.

It is a minimal example, not a strong method: it only interpolates between the training tasks
it has seen, filmed by the same cameras as the evaluation tasks, and synthetic scenes are not
real recordings.
