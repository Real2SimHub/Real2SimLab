# Architecture

For contributors: how the package is organised, which way dependencies point, what happens
during an experiment, how runs are isolated, and where to plug in new things. The objects that
pass between components are defined by the [specification](../spec/index.md).

## Packages

```
real2simlab/
  _base.py, errors.py   StrictModel (unknown fields rejected, immutable); errors (R2SError,
                        SetupError with a `fix`, ProblemsError with a list of problems)
  plugins.py            entry-point discovery: benchmarks, methods, metrics, formats,
                        backends, scorers, cli
  spec/                 version strings and compatibility rules; JSON Schema export;
                        generated documentation (CLI reference)
  capture/              captures (r2s-capture/1.0): conventions, provenance, visibility
                        policies, `capture_from_video`
  task/                 the TaskConfig model, task directories, validation, TaskBuilder, the
                        agent statement template, scaffolding; overlap.py (whether two tasks
                        hold the same file or the same recording)
  twin/                 TwinManifest, the run contract, delivery conventions,
                        fingerprints, saving and loading twins; sdk/r2s_twin.py (single-file SDK)
  formats/              output formats (plugins): tracks, video, 4d_world, video2sim_package
  media.py, vendor.py   ffmpeg and ffprobe; third-party code fetched at pinned commits
  sandbox/              the Backend interface; docker, bwrap, local; the egress proxy
  envs/                 runtime images: registry of environments (and which agent image fits
                        which task runtime), building, pulling, unpacking for bwrap; the
                        Dockerfiles are in docker/ and in the wheel
  runner/               run_twin (one frozen run and its r2s-run/1.0 record); twin check
  metrics/              trajectory, set and event metrics
  scoring/              the reproduction protocol, native scorers and their headlines,
                        TaskResult and Results, the scorecard, bootstrap aggregation
  methods/              Method, BuildContext, BuildOutcome, package_twin; the baselines;
                        training.py (TrainContext, the r2s-training/1.0 checkpoint record)
  agents/               AgentMethod (the shared agent pipeline), bootstrap.py (runs inside
                        the sandbox: credentials in, redaction out), records; the agents
  benchmarks/           BenchmarkAdapter; synthetic (truth generators and a renderer) and the
                        adapters of external benchmarks
  orchestrate/          experiment configuration, run directory layout, scheduling (parallel,
                        resumable, retries, budgets)
  report/               Markdown, CSV and the offline HTML report
  train/                training: score_solution (a solution's reward, the evaluation's own
                        scoring path, also the Harbor verifier's), export_dataset (checked
                        against evaluation tasks) and fit_method (a trainable method's
                        checkpoint)
  cli/                  the r2s command; each subcommand is a plugin
  doctor.py             the checks behind `r2s doctor`
```

## Dependency direction

```
_base, errors, spec
   ↑
capture → task (model) → twin → formats → metrics → scoring
                                        ↑                          ↑
                                     sandbox, envs → runner ───────┘
                                        ↑
                               methods → agents
                                        ↑
                     benchmarks, orchestrate, report, train, cli
```

- Lower layers never import upper ones. `twin` does not depend on `task` (`ProgramRules` lives
  in `twin.model`; tasks refer to it).
- Concrete plugins (baselines, agents, benchmarks, metrics, formats, backends, scorers) are found
  through entry points only; the core never imports a specific benchmark.
- Code that runs inside sandboxes or images — `twin/sdk/r2s_twin.py`, `agents/bootstrap.py`,
  `sandbox/egress_proxy.py`, `benchmarks/synthetic/physics/` — uses only the standard library,
  NumPy and SciPy, never `real2simlab`; so does `sandbox/reaper.py`, the supervisor every
  command of the `local` backend runs under.

## One experiment

```
experiment.toml ─→ resolve tasks (paths, or benchmarks built into the run) ─→ validate
      │
      ├─ for each (task, method, trial): build
      │     BuildContext (a copy of visible/, the statement, a work directory, the backend;
      │                   hidden/ only for privileged baselines)
      │     agents: prepare the workspace → sandbox (bootstrap.py → agent CLI) → collect
      │             solution/ → package_twin
      │     other methods: write the program → package_twin
      │     → builds/.../build_record.json and twin/
      │
      └─ evaluate, for each seed
            run_twin (run) [+ run_twin (rerun), the reproducibility check]
            reproduction protocol, or the native scorer on the delivery or the re-run
            → the benchmark's headline → results/.../seed-<s>.json
   collect → aggregate (bootstrap; per benchmark, the scorecard) → results.json → report/
```

## Isolation

| Situation | Backend | Network | Mounts |
|---|---|---|---|
| Frozen re-run of a twin | docker (`--network none --read-only --cap-drop ALL`, host uid) or bwrap | none | program read-only, output writable, `/tmp` tmpfs |
| Agent building a twin | docker | allowlist: an internal network whose only way out is the egress proxy, which logs every connection | workspace and logs writable, plan read-only; **`hidden/` is never mounted** |
| Debugging | local | not isolated (a warning; records say `isolated: false`) | paths rewritten |

Credentials are injected at run time only. Secret variables are passed as `docker run -e NAME`
(the value exists only in the Docker client's environment, never on a command line);
credential files travel base64-encoded in variables that `bootstrap.py` writes into the
sandbox's tmpfs home and then unsets. Agent output is redacted (injected values and common
key formats) before it reaches the disk, and the injected values are removed on the host from
the files the agent wrote itself (`redact_files`) before its solution is packaged. Method and
plugin options are recorded, so a credential among them is refused
(`options.refuse_credentials`). See [Security](../security.md).

## Caching and resumption

- A build's key: the task's content hash, the method's configuration digest, the trial and the
  Lab version.
- A run's key: the twin's fingerprint, the run's label, the seed, the image, the backend, the
  limits and the output requirements.
- Tasks built from a benchmark are keyed by the adapter's source digest as well, and reused
  only while each still has the content hash recorded when it was built (`built_tasks.json`).
- A trained method's configuration digest includes its checkpoint's content fingerprint: a new
  checkpoint in the same directory builds again.
- Keys live in each step directory's `.key`. A matching key means the step is reused; anything
  else (a `.key` that cannot be read included) redoes the step from scratch. Infrastructure
  failures are never cached.
- Results count only while their build's key is the current one: a new build discards the
  unit's earlier results first, and when the configuration changed but its new build was
  skipped (a budget spent), the earlier configuration's results are not collected and the unit
  is reported as not run. The report shows exactly the results `results.json` counts.
- What the run built from is checked again before it is `complete`: a task, program or
  checkpoint that changed while the experiment ran ends it with an error naming it (it stays
  `running`), and the same command builds and scores again what changed.
- A method whose code raises an error it should have reported (not a `SetupError`) stops only
  its own build: an `infra_error` saying what was raised, with the traceback in `r2s.log`; it
  is not tried again by the same command (an agent may have run), and the next run builds it.

## Extension points

| To add | Do |
|---|---|
| a benchmark | subclass `BenchmarkAdapter`, register under `real2simlab.benchmarks`; keep the import steps as module-level functions (other projects reuse them) |
| a method or agent | subclass `Method` or `AgentMethod`, register under `real2simlab.methods` |
| a method that learns | also set `trainable = True`, derive its options from `TrainedOptions` and implement `train(context)`; see [Training](../guides/training.md#training-a-method) |
| a training split | declare `splits = ("eval", "train")` on the adapter and build `split="train"` tasks (only under terms that allow training) |
| a metric | subclass `Metric`, register under `real2simlab.metrics` |
| an output format | subclass `OutputFormat`, register under `real2simlab.formats` |
| a backend | subclass `Backend`, register under `real2simlab.backends` |
| an official scorer | subclass `NativeScorer` (`score`, `headline`, and `aggregate` when the benchmark does not average), register under `real2simlab.scorers` |
| a delivery convention | add an entry to `real2simlab/twin/conventions.py` |
| a CLI command | a `register(subparsers)` function under `real2simlab.cli` |

Step-by-step guides: [Add a benchmark](../tutorials/add-benchmark.md),
[Evaluate an agent](../tutorials/evaluate-agent.md) (adding an agent is at the end).
