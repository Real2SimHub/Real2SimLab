# Command line reference

Generated from `r2s ... --help`. Every command also prints this with `--help`.

## `r2s`

```text
usage: r2s [-h] [--version] COMMAND ...

Real2SimLab 1.0.0 (specification 1.0): a unified platform for
real-to-simulation. Any method, any benchmark: true to each, together on one scorecard; and
training tasks, datasets and rewards from the same code (`r2s train`).

Start with `r2s doctor` and `r2s demo`, then `r2s bench list`, `r2s bench build NAME --out
tasks/` and `r2s eval --tasks tasks/ --method codex --out runs/codex`.

positional arguments:
  COMMAND
    bench               list benchmark adapters and build their tasks
    capture             create captures (r2s-capture/1.0)
    demo                see a first report: baselines and a scripted agent on two synthetic tasks
    doctor              check that this machine can run Real2SimLab, with fixes for what is
                        missing
    env                 build and list runtime images
    eval                evaluate methods on tasks without writing a configuration file
    export              export tasks to other frameworks
    report              write the Markdown, CSV and offline HTML report of a run directory
    run                 run an experiment from a configuration file
    spec                export the JSON Schemas of every specification object
    task                create, validate and inspect task directories (r2s-task/1.0)
    train               training tasks with exact truth, datasets, and the evaluation's scorer as
                        a reward
    twin                check and run twin packages (r2s-twin/1.0)

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
```

### `r2s bench`

```text
usage: r2s bench [-h] ACTION ...

list benchmark adapters and build their tasks

positional arguments:
  ACTION
    list                list the registered benchmark adapters
    build               build a benchmark's task directories

options:
  -h, --help            show this help message and exit
```

#### `r2s bench build`

```text
usage: r2s bench build [-h] --out OUT [--select SELECT] [--tier TIERS] [--option OPTION]
                       [--split SPLIT] [--overwrite]
                       name

Build a benchmark's task directories.

Example: r2s bench build synthetic --out tasks/ --select bouncing_ball --select pendulum --tier calibrated

positional arguments:
  name                  a benchmark from `r2s bench list`

options:
  -h, --help            show this help message and exit
  --out OUT             directory to write tasks into
  --select SELECT       build only these (repeatable)
  --tier TIERS          visibility tiers (synthetic)
  --option OPTION       adapter option key=value
  --split SPLIT         eval (default) or train, when the benchmark releases training data
  --overwrite           replace existing task directories
```

#### `r2s bench list`

```text
usage: r2s bench list [-h] [--json]

list the registered benchmark adapters

options:
  -h, --help            show this help message and exit
  --json
```

### `r2s capture`

```text
usage: r2s capture [-h] ACTION ...

create captures (r2s-capture/1.0)

positional arguments:
  ACTION
    from-video          make a single-view capture from a video file

options:
  -h, --help            show this help message and exit
```

#### `r2s capture from-video`

```text
usage: r2s capture from-video [-h] --capture-id CAPTURE_ID [--intrinsics INTRINSICS | --fov FOV]
                              [--extrinsics EXTRINSICS] [--object OBJECT] [--text TEXT]
                              video out

Make a single-view capture from an ordinary video file.

Example: r2s capture from-video drop.mp4 visible/ --capture-id lab/drop_01 \
           --fov 62 --object ball=0.067 --text 'A tennis ball is dropped.'

positional arguments:
  video
  out                   capture directory to create

options:
  -h, --help            show this help message and exit
  --capture-id CAPTURE_ID
  --intrinsics INTRINSICS
                        fx,fy,cx,cy in pixels (measured)
  --fov FOV             horizontal field of view in degrees, above 0 and below 180 (estimated
                        intrinsics)
  --extrinsics EXTRINSICS
                        JSON file with a 4x4 camera-to-world matrix
  --object OBJECT       NAME=SIZE_M: a known object size (scale)
  --text TEXT           what the video shows
```

### `r2s demo`

```text
usage: r2s demo [-h] [--out OUT] [--backend {auto,docker,bwrap,local}] [--open]

Build two synthetic tasks, evaluate the reference baselines and a scripted agent
(no model, no key) on them, and write the report. The backend is chosen
automatically: docker (image present or pullable), bwrap, then local.

Example: r2s demo --open

Exits with status 1 when nothing could be scored (every result failed because of the
infrastructure, not a method).

options:
  -h, --help            show this help message and exit
  --out OUT             run directory
  --backend {auto,docker,bwrap,local}
                        where twins run (default: the best available)
  --open                open the report in a browser
```

### `r2s doctor`

```text
usage: r2s doctor [-h] [--json] [--gpu]

check that this machine can run Real2SimLab, with fixes for what is missing

options:
  -h, --help            show this help message and exit
  --json
  --gpu                 also check GPU support
```

### `r2s env`

```text
usage: r2s env [-h] ACTION ...

build and list runtime images

positional arguments:
  ACTION
    list                list the registered environments and whether they are built
    build               build environment images with Docker (dependencies first)
    export              save an environment image to a tarball (to unpack it where there is no
                        Docker)
    unpack              unpack an environment image into a root file system for the bwrap backend
    pull                pull published environment images (instead of building them)

options:
  -h, --help            show this help message and exit
```

#### `r2s env build`

```text
usage: r2s env build [-h] [--pip-index-url PIP_INDEX_URL] [--npm-registry NPM_REGISTRY]
                     [--apt-mirror APT_MIRROR] [--no-cache] [--network NETWORK]
                     [--build-arg KEY[=VALUE]] [--only]
                     names [names ...]

Build environment images with Docker; dependencies are built first.

Example: r2s env build r2s-python r2s-agent-codex --pip-index-url https://mirror/simple

positional arguments:
  names                 environments from `r2s env list`, or 'all'

options:
  -h, --help            show this help message and exit
  --pip-index-url PIP_INDEX_URL
                        a PyPI mirror for the build
  --npm-registry NPM_REGISTRY
                        an npm mirror for the build
  --apt-mirror APT_MIRROR
                        a Debian mirror, e.g. http://mirrors.aliyun.com
  --no-cache
  --network NETWORK     network for the build's RUN steps, e.g. host (when downloads in the build
                        are slow)
  --build-arg KEY[=VALUE]
                        another Docker build argument, e.g.
                        BLENDER_URL=https://mirror/blender.tar.xz; a name alone (e.g. HTTPS_PROXY)
                        takes its value from this shell
  --only                do not build dependencies
```

#### `r2s env export`

```text
usage: r2s env export [-h] --out OUT name

save an environment image to a tarball (to unpack it where there is no Docker)

positional arguments:
  name                  an environment or image reference

options:
  -h, --help            show this help message and exit
  --out OUT             tarball to write
```

#### `r2s env list`

```text
usage: r2s env list [-h] [--json]

list the registered environments and whether they are built

options:
  -h, --help            show this help message and exit
  --json
```

#### `r2s env pull`

```text
usage: r2s env pull [-h] names [names ...]

Pull published images and tag them with their local names. Sources, in order: the
local name itself when R2S_IMAGE_PREFIX points at a registry of your own, each prefix
in R2S_IMAGE_MIRRORS (comma separated, e.g. a registry mirror closer to you), then
the project's registry (its GitHub Container Registry).

Example: R2S_IMAGE_MIRRORS=registry.example.com/real2simlab/ r2s env pull r2s-python

positional arguments:
  names                 environments from `r2s env list`, or 'all'

options:
  -h, --help            show this help message and exit
```

#### `r2s env unpack`

```text
usage: r2s env unpack [-h] [--from-tar FROM_TAR | --from-url [URL]] [--sha256 SHA256] [--force]
                      name

Unpack an image for the bwrap backend (Linux without root or Docker).

With Docker on this machine:  r2s env unpack r2s-python
Without Docker: on a machine with Docker run `r2s env export r2s-python --out r2s-python.tar`,
copy the file, then:  r2s env unpack r2s-python --from-tar r2s-python.tar

positional arguments:
  name                  an environment or image reference

options:
  -h, --help            show this help message and exit
  --from-tar FROM_TAR   a tarball written by `r2s env export`
  --from-url [URL]      download the image tarball (default: this version's GitHub release) and
                        check its SHA-256
  --sha256 SHA256       the expected SHA-256 of the --from-url download (default: <URL>.sha256)
  --force               extract it again even if it was unpacked before (to repair a damaged one)
```

### `r2s eval`

```text
usage: r2s eval [-h] [--tasks TASKS] [--benchmark BENCHMARK] [--select SELECT] [--tier TIERS]
                [--include INCLUDE] --method METHOD [--out OUT] [--backend BACKEND]
                [--seeds SEEDS [SEEDS ...]] [--trials TRIALS] [--concurrency CONCURRENCY]
                [--agent-concurrency AGENT_CONCURRENCY] [--max-agent-runs MAX_AGENT_RUNS]
                [--check-reproducibility] [--no-resume] [--name NAME]

Evaluate methods on tasks without writing a configuration file.

Examples:
  r2s eval --benchmark synthetic --method oracle --method static --out runs/demo
  r2s eval --tasks tasks/ --method codex:model=gpt-5-codex,reasoning_effort=medium
  r2s eval --tasks tasks/ --method physical=fake_agent:mode=physical --backend local

A method is NAME[:key=value,...], optionally prefixed by LABEL=. A value is read as
JSON when it is JSON; one that holds a comma must be (a text in double quotes, a list
or a table), in single quotes for the shell:
  --method 'codex:gpus="0,1",allowlist=["a.example","b.example"]'

Exits with status 1 when nothing could be scored (every result failed because of the
infrastructure, not a method).

options:
  -h, --help            show this help message and exit
  --tasks TASKS         task directories
  --benchmark BENCHMARK
                        build and use a benchmark's tasks
  --select SELECT       with --benchmark: only these tasks
  --tier TIERS          with --benchmark synthetic: tiers
  --include INCLUDE     glob filter on task ids
  --method METHOD       a method (repeatable)
  --out OUT             run directory
  --backend BACKEND     sandbox backend: docker, bwrap or local (default: docker; local has no
                        isolation)
  --seeds SEEDS [SEEDS ...]
  --trials TRIALS
  --concurrency CONCURRENCY
  --agent-concurrency AGENT_CONCURRENCY
  --max-agent-runs MAX_AGENT_RUNS
                        stop starting agents after this many
  --check-reproducibility
                        run each twin twice
  --no-resume
  --name NAME
```

### `r2s export`

```text
usage: r2s export [-h] TARGET ...

export tasks to other frameworks

positional arguments:
  TARGET
    harbor              write a task as a Harbor task (agent environment, offline verifier,
                        optional oracle)

options:
  -h, --help            show this help message and exit
```

#### `r2s export harbor`

```text
usage: r2s export harbor [-h] --out OUT [--solution SOLUTION] [--self-contained]
                         [--timeout TIMEOUT]
                         task

Write a task as a Harbor task directory. Harbor then runs any of its agents on it; the
separate, offline verifier re-runs the delivered twin and scores it with this Lab.

Example:
  r2s export harbor runs/tasks/bouncing_ball/calibrated --out harbor/bouncing_ball \
      --solution oracle
  r2s env build r2s-verifier
  harbor run -p harbor/bouncing_ball -a oracle

positional arguments:
  task                  a task directory

options:
  -h, --help            show this help message and exit
  --out OUT             Harbor task directory to create
  --solution SOLUTION   'oracle' (the task's truth generator) or a program directory with run.py
  --self-contained      Dockerfiles that install everything from python:3.11-slim (no r2s images
                        needed)
  --timeout TIMEOUT     agent timeout in seconds
```

### `r2s report`

```text
usage: r2s report [-h] run_dir

write the Markdown, CSV and offline HTML report of a run directory

positional arguments:
  run_dir

options:
  -h, --help            show this help message and exit
```

### `r2s run`

```text
usage: r2s run [-h] [--out OUT] [--backend BACKEND] [--no-resume] [--no-report] config

Run an experiment described by a TOML file (r2s-experiment/1.0): tasks, methods,
seeds, trials, backend, concurrency and budgets. Interrupted runs resume where they
stopped when the same command is run again.

Exits with status 1 when nothing could be scored (every result failed because of the
infrastructure, not a method).

positional arguments:
  config                experiment TOML file

options:
  -h, --help            show this help message and exit
  --out OUT             run directory (default: runs/<name>)
  --backend BACKEND     override the configured backend
  --no-resume           redo every step
  --no-report           do not write the HTML report
```

### `r2s spec`

```text
usage: r2s spec [-h] ACTION ...

export the JSON Schemas of every specification object

positional arguments:
  ACTION
    schema              write or print JSON Schemas

options:
  -h, --help            show this help message and exit
```

#### `r2s spec schema`

```text
usage: r2s spec schema [-h] [--out OUT] [name]

write or print JSON Schemas

positional arguments:
  name                  one object, e.g. r2s-task (default: every object; without --out, their
                        names)

options:
  -h, --help            show this help message and exit
  --out OUT             directory to write <name>.schema.json files into
```

### `r2s task`

```text
usage: r2s task [-h] ACTION ...

Create, validate and inspect task directories (r2s-task/1.0).

A task directory holds task.toml, visible/ (what a method sees) and hidden/ (the
reference and what scoring needs). See the tutorial 'Write tasks for your benchmark'.

positional arguments:
  ACTION
    new                 create a task skeleton to fill in
    validate            check task directories against r2s-task/1.0
    show                show a task's metadata or the task statement agents get
    list                list the tasks under directories

options:
  -h, --help            show this help message and exit
```

#### `r2s task list`

```text
usage: r2s task list [-h] [--json] [paths ...]

list the tasks under directories

positional arguments:
  paths

options:
  -h, --help            show this help message and exit
  --json
```

#### `r2s task new`

```text
usage: r2s task new [-h] --id ID [--title TITLE] [--video VIDEO] path

create a task skeleton to fill in

positional arguments:
  path                  directory to create (must not exist or be empty)

options:
  -h, --help            show this help message and exit
  --id ID               task id, e.g. mybench/drop_01
  --title TITLE         one-line title
  --video VIDEO         make visible/ from this video (r2s capture from-video)
```

#### `r2s task show`

```text
usage: r2s task show [-h] [--statement] [--json] path

show a task's metadata or the task statement agents get

positional arguments:
  path

options:
  -h, --help            show this help message and exit
  --statement           print the task statement agents get
  --json
```

#### `r2s task validate`

```text
usage: r2s task validate [-h] [--json] [--no-references] paths [paths ...]

Check task directories: spec, hidden-data leaks into visible/, missing files,
references valid in the output format, scoring.
Exit status 1 if any task has errors.

positional arguments:
  paths                 task directories, or directories of tasks

options:
  -h, --help            show this help message and exit
  --json                print the reports as JSON
  --no-references       skip checking reference outputs
```

### `r2s train`

```text
usage: r2s train [-h] COMMAND ...

Training with the Lab: training tasks (generated, or your own with split = train),
exported with the runs on them as a dataset that holds no evaluation task's recording,
a method trained on it, and any solution scored exactly as an evaluation would.

Example:
  r2s train tasks --count 200 --out train/tasks
  r2s bench build synthetic --out eval/tasks
  r2s train export --tasks train/tasks --against eval/tasks --out train/data
  r2s train fit nearest_twin --data train/data --out train/model
  r2s eval --tasks eval/tasks --method nearest_twin:checkpoint=train/model \
      --out eval/run
  r2s train reward --task train/tasks/synthetic-train/pendulum/calibrated/s0-00003 \
      --solution my_program/

positional arguments:
  COMMAND
    tasks               generate training tasks: the synthetic phenomena with sampled parameters
    export              export tasks and the results of runs on them as a training dataset
    fit                 train a method on an exported dataset: a checkpoint to evaluate it with
    reward              score one solution on one task as an evaluation would: its reward

options:
  -h, --help            show this help message and exit
```

#### `r2s train export`

```text
usage: r2s train export [-h] --out OUT [--tasks TASKS] [--against AGAINST] [--runs RUNS]
                        [--min-reward MIN_REWARD] [--include-undelivered] [--skip-unmatched]
                        [--allow-eval-tasks] [--overwrite]

Write a dataset: tasks.jsonl (inputs and labels of each task) and runs.jsonl (each
delivered program of the runs, with its reward, the content hash of the task it was
scored on and the agent's redacted trajectory). A result is exported only with the
task version it was scored on (found where the run found it, or among --tasks).
Only training tasks (split = train) are exported unless --allow-eval-tasks; tasks of
a benchmark whose terms forbid training on its data never are. With --against, a
task that holds the same file or the same recording as one of those evaluation tasks
(a copy, a re-encoding, a resized copy or a clip cut from it) is left out, and
export.json says why.

options:
  -h, --help            show this help message and exit
  --out OUT             dataset directory to create
  --tasks TASKS         task directories
  --against AGAINST     evaluation task directories the dataset must not overlap (repeatable)
  --runs RUNS           run directories (r2s eval/run)
  --min-reward MIN_REWARD
                        keep only programs with at least this reward
  --include-undelivered
                        also export results where nothing was delivered (no program, reward 0)
  --skip-unmatched      leave out results whose task or program is missing or has changed since
                        the run, instead of failing
  --allow-eval-tasks    also export evaluation tasks (their results then no longer count as held
                        out)
  --overwrite           replace an existing dataset
```

#### `r2s train fit`

```text
usage: r2s train fit [-h] --data DATA --out OUT [--seed SEED] [--overwrite] method

Give a method that learns (e.g. nearest_twin) an exported dataset; it writes what it
learned into the checkpoint directory, with the Lab's record (training.json) and the
dataset's overlap prints. Evaluate it with checkpoint=DIR among its options: it is not
run on tasks that overlap its training tasks.

Example:
  r2s train fit nearest_twin --data train/data --out train/model
  r2s eval --tasks eval/tasks --method nearest_twin:checkpoint=train/model --out eval/run

positional arguments:
  method                a method that learns, NAME[:key=value,...]

options:
  -h, --help            show this help message and exit
  --data DATA           an exported dataset
  --out OUT             checkpoint directory to create
  --seed SEED
  --overwrite           replace an existing checkpoint
```

#### `r2s train reward`

```text
usage: r2s train reward [-h] --task TASK --solution SOLUTION [--seed SEED] [--work WORK] [--json]
                        [--backend BACKEND]

Package a solution directory (run.py and its files) as a twin, run it once, frozen and
offline, and score it with the task's protocol. Prints the reward (0 to 1) and, with
--json, the full task result. A run that fails outside the solution (the sandbox did
not start) gives no reward: the command fails with status 1. Use docker or bwrap for
programs you did not write.

options:
  -h, --help            show this help message and exit
  --task TASK           a task directory
  --solution SOLUTION   a solution directory (run.py and its files), or a twin package (twin.json,
                        program/)
  --seed SEED
  --work WORK           keep the run's files here (cleared when used again)
  --json                print the full result as JSON
  --backend BACKEND     sandbox backend: docker, bwrap or local (default: docker; local has no
                        isolation)
```

#### `r2s train tasks`

```text
usage: r2s train tasks [-h] --out OUT [--count COUNT] [--seed SEED] [--select SELECT]
                       [--tier TIERS] [--overwrite]

Write training tasks (split = train): each synthetic phenomenon with its hidden
parameters and initial state drawn around the evaluation task's, rendered with the
same camera and scored the same way, with the exact truth as labels. The same seed
always gives the same tasks. A near copy of an evaluation task is never drawn, and a
sample whose video the overlap check cannot tell from the evaluation task's is
skipped (its index is then missing).

options:
  -h, --help            show this help message and exit
  --out OUT             directory to write tasks into
  --count COUNT         samples per phenomenon (20)
  --seed SEED           which samples (0)
  --select SELECT       only these phenomena (repeatable)
  --tier TIERS          visibility tiers (default: both)
  --overwrite           replace existing tasks
```

### `r2s twin`

```text
usage: r2s twin [-h] ACTION ...

check and run twin packages (r2s-twin/1.0)

positional arguments:
  ACTION
    check               run a twin the way the evaluation will, and report problems
    run                 run a frozen twin once

options:
  -h, --help            show this help message and exit
```

#### `r2s twin check`

```text
usage: r2s twin check [-h] [--task TASK] [--out OUT] [--seed SEED] [--backend BACKEND] twin

Run a twin package (or a program directory) the way the evaluation will: frozen,
offline, twice. The output must be valid and the two outputs identical. Exit status 1
on any problem.

positional arguments:
  twin                  twin package (twin.json + program/) or a program directory

options:
  -h, --help            show this help message and exit
  --task TASK           apply this task's format, rules and limits
  --out OUT             where runs are kept
  --seed SEED
  --backend BACKEND     sandbox backend: docker, bwrap or local (default: docker; local has no
                        isolation)
```

#### `r2s twin run`

```text
usage: r2s twin run [-h] --out OUT [--seed SEED] [--timeout TIMEOUT] [--backend BACKEND] twin

run a frozen twin once

positional arguments:
  twin

options:
  -h, --help            show this help message and exit
  --out OUT             run directory to create
  --seed SEED
  --timeout TIMEOUT
  --backend BACKEND     sandbox backend: docker, bwrap or local (default: docker; local has no
                        isolation)
```
