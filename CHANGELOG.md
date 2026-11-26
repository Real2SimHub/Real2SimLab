# Changelog

All notable changes are listed here. Versions follow [semantic versioning](https://semver.org);
the specification has its own version (`r2s-*/1.0`), stated in every object.

## 1.0.0 - UNRELEASED

The first public release: a unified platform for real-to-simulation.

### Specification 1.0

- Tasks (`task.toml`, a reference and repeated recordings in `hidden/`), captures, twins
  (`twin.json`), the run contract, output formats (`tracks`, `4d_world`, `video`,
  `video2sim_package`), delivery conventions (`r2s`, `build_sh_world`, `blender_scene`,
  `video2sim_package`), scoring protocols (`reproduction/1.0`, `native`), headlines, and records
  (run, build, agent run, task result, results with the scorecard, experiment, training
  checkpoint), with JSON
  Schemas in `docs/spec/schemas/`.
- Frozen for 1.x: changes within 1.x only add; a breaking change needs a new major version
  and a migration note.

### Evaluation

- Every delivery frozen and re-run offline and read-only; the re-run checked against the
  agent's own session output and, optionally, against a second run.
- Benchmarks' own official scorers as native protocols (scorer plugins with a headline and,
  where the benchmark needs it, its own aggregate), on the delivery or on its frozen re-run;
  the Lab's reproduction protocol for raw datasets, with a static baseline, a noise floor from
  repeated recordings and the skill such a recording scores.
- One scorecard across benchmarks: each benchmark's headline metric with its reference points,
  combined the benchmark's way, with two-level bootstrap intervals and re-run rates.
- Experiments: parallel, resumable (content-addressed steps), infrastructure retries, budgets
  on agent runs, wall time and cost; reports as Markdown, CSV and an offline HTML page with the
  scorecard, overlays and agent trajectories. `examples/experiments/unified.toml` runs one agent
  on every benchmark.
- `r2s demo`: the first report in one command. It and every run of `fake_agent` no longer
  leave an `r2s-fake-agent-*` directory in the temporary directory: it is removed when the
  command ends.
- Nothing is scored by leaving things out: the centroid-set metrics fail on frames
  where the reference has objects and the prediction none (instead of skipping them); image
  metrics project each predicted centroid with the predicted camera at the same instant
  (a moving camera at another frame rate than the reference was compared at the wrong time);
  the session and re-run checks compare everything an output's format holds (a 4D world's
  meshes and points, a track's camera and rotations), not only object centroids. A
  4DCodeBench world the official scorer judges an `error` (rejected by its gates, or a metric
  failed on it) counts `dynamic_iou` 0 in the benchmark's mean, as the official aggregates
  count it, instead of being left out (which ranked a method that delivered a world the scorer
  refuses above one that delivered a poor world).
- An experiment counts a unit's results only while they belong to its current configuration:
  after a configuration change whose new build a budget skips, the unit is reported as not run,
  not with the earlier configuration's score; the report shows exactly the counted results.
- The Harbor verifier runs the solution as an unprivileged user who cannot read the task's
  hidden reference (it needs root, which Harbor's verifier container has).
- Hidden data put where a method can see it is caught in more places. `r2s export harbor`
  validates the task as `r2s eval` does, and `r2s train export` refuses a task that shows hidden
  data (a symbolic link, a hidden file's content in `visible/` or as `instruction.md`; one error
  with the fix, nothing written); before, both copied such a task, link targets included, into
  the agent's environment or the dataset. `r2s task validate` also compares `instruction.md`
  (which replaces the statement) with `hidden/`, and no longer takes hidden data that
  `capture.json` lists as a stream (masks, depth, poses, a robot recording) or puts in place of
  the video for the observation: only a hidden video identical to a view's video is; it
  compares whole files, as before. A task's `visible/` is copied for a build, an agent's
  workspace or an export without following symbolic links or copying a file manager's litter
  (`.DS_Store`, `._*`), and checked against the version that was validated; a build's statement
  is the `instruction.md` that was validated, or is generated from that copy. A build whose
  task's `visible/` or `instruction.md` changed after validation is not given to the method
  (`infra_error`, built again next time); a symbolic link that appears in the task during a run
  stops the run with an error naming it (the task is no longer valid).
- Results are read back, kept and current: a delivery that could not be scored (no headline
  value) is a result without a score, not an error that stops the aggregation; a resumed
  experiment reuses a run only while its output has the recorded fingerprint, and a build only
  while its session output does; tasks of one benchmark name with different headline metrics
  each get a scorecard entry; re-run agreement counts only the deliveries that could be
  compared (`rerun_compared`, `rerun_agreed`); distinct task ids never share a directory; each
  selection of a benchmark is built into its own directory.
- A failure of the machine is not a score: a build or re-run whose sandbox did not start, a
  delivery this machine could not copy, an official scorer that failed on a valid delivery, or
  a second re-run that did not start is recorded as `infra_error` (build and task result),
  left out of every statistic, rate, scorecard entry and reward, counted as `infra_errors`
  ("not scored: infrastructure" in summaries and reports), and redone when the experiment runs
  again. Native scorers say what such a failure is (`NativeScorer.failure`).
- An agent whose sandbox or own infrastructure failed around it is not scored: an agent
  sandbox that bwrap could not set up (bwrap exits 1 before the agent starts, as a failing
  agent does; a sandbox like it that cannot start Python tells them apart), one killed from
  outside, an agent CLI its image does not have, a `mini_swe` session whose sandbox does not
  start or stops running commands or whose model API could not be used before it answered at
  all (a rejected key, an account without credit (402), a model that does not exist, an
  endpoint that is down, unreachable or rate-limiting; a prompt the API refuses, too long or
  against its policy, stays the method's
  result), and Harbor that produced no trial or could not set up the agent's environment are
  `infra_error`, no longer a "delivered nothing" 0 that was reused after the machine was fixed
  and exported as a reward of 0 by `r2s train export --include-undelivered`. An agent run of
  which nothing ran is given back to `max_agent_runs` and tried again at once; one that had
  started counts (also when the budget is counted again from the run's records) and is built
  again by the next run, not by the same command. bwrap's systemd scope continues when its
  program runs out of memory (`OOMPolicy=continue`, where systemd takes it), so that stays the
  program's own failure. Harbor's and mini-swe-agent's error messages are redacted before
  they are recorded.
- Resuming trusts content, not names: a build is reused only while its frozen twin still has
  the recorded fingerprint, and a run only while the image it used has the same content id (a
  tag moved to another image re-runs). An experiment writes `state/status.json` (running, then
  complete) and the SHA-256 of each result file in its `results.json` row; `r2s report` warns
  about an incomplete run directory and about tasks whose content changed since they were
  scored (their new media are not shown next to the old scores).
- A record without a fingerprint or digest the Lab always writes is not taken as unchanged,
  whatever its `lab_version`: a build record without its delivered twin's
  `twin_fingerprint`, or without the `session_output_fingerprint` of a session output that can
  be fingerprinted, and a run record of a successful run without its `output_fingerprint` make
  the step run again (it was reused with whatever program or output was there, and scored under
  the method's name); a run directory with a `results.json` row without `result_sha256`, or
  without `state/status.json`, is not one complete set (`r2s train export` refuses it,
  `r2s report` warns), and the report does not draw an output its run record cannot vouch for.
  Such a record was edited, or written by a pre-release build of the Lab: a run directory
  made by a development build before the 1.0.0 release lacks some of these fields, and must be
  finished by running the same command again (it redoes what its records cannot vouch for).
- A run is complete only with the inputs it was built from: a task, program or checkpoint that
  changes while the experiment runs (an editor, a sync client, another command) ends the
  command with an error naming it and leaves the run `running`; the same command builds and
  scores again what changed. It finished `complete` with exit status 0 and that unit's results
  silently left out of the scorecard, which counted only the tasks left. A task is told changed
  by the bytes of its `task.toml`: one holding `nan` (in `metadata` or a scorer's options,
  where `nan` is not equal to itself) is not always changed, for `r2s eval`, `r2s train
  export` or `r2s train reward`. A benchmark selection
  built into the run directory is reused only while each of its tasks has the content hash
  `built_tasks.json` now records: a task damaged or edited there is built again (it was refused
  as invalid by every resume, with no fix, or evaluated as edited); and a step whose `.key`
  cannot be read (bytes that are not text) is redone, where every resume stopped with the
  decoder's words alone.
- A method whose code raises an error it should have reported stops only its own build, which
  is an `infra_error` saying what was raised (the traceback is in `r2s.log`; both redacted of
  key-shaped strings and of the values of this process's credential variables, which a key of
  no known shape, such as a proxy's token, was not; so is the `stopping:` line), not
  tried again by the same command (an agent may have run, and counts) and built again by the
  next run; it stopped the whole experiment. Harbor's record of its trial (`result.json`) cut
  short, not text or not a JSON object, and a mini-swe-agent harness whose output does not end
  with its driver's summary, make such a build (whether and how the agent ran is not known),
  no longer an error with the decoder's words alone or a traceback; a Harbor or mini-swe-agent
  version that cannot be asked after the run is recorded as unknown instead of discarding the
  run.
- Datasets and checkpoints are replaced by an atomic swap
  where the system has one (Linux `renameat2`, macOS `renamex_np`); an interrupted replacement
  elsewhere is recovered on the next use, so the published path is never missing.
- Preflight checks only the images of tasks and methods that will run: a scorer image needed
  only by a task the method does not apply to no longer blocks a mixed experiment. Every check
  (backend, tasks, methods, images) runs before an experiment writes to its run directory, and
  a build that is no longer current takes its old results with it.
- Official scores are reused like builds and runs: a rerun of the same experiment reuses a
  native score (`native_report.json`) while the task, the frozen twin, its output, the scorer's
  options and the scorer itself (`NativeScorer.cache_identity`, e.g. its image's content id)
  are unchanged.
- An official scorer judges only deliveries its benchmark's own reader accepts: a Physics-IQ
  video or PhysCodeGym frames that OpenCV cannot read are the delivery's result
  (`invalid_video`, `invalid_frames`), and a failure of the official code after that is a
  scorer failure, not a 0. A 4DCodeBench world the official grader rejects with a reward is
  `rejected`, the delivery's result.
- An official scorer that fails on something other than the delivery does not score it:
  PhysCodeGym's metrics rejecting the task's reference video as this machine's OpenCV reads it
  (not 30 fps, not 240 frames, unreadable, as with an OpenCV without a video backend), and
  frames an OpenCV that cannot read even a PNG it wrote itself rejects, are a scorer failure
  (`infra_error`), no longer the delivery's `invalid_frames` and 0; a Video2World row its
  evaluator could not measure (the evaluator crashed, timed out or lost its GPU, its metric
  code failed, it wrote no metric record or no build verdict), and a row Video2World gives
  neither a score nor an issue (a sample it does not score), is an `infra_error`, no longer a
  result without a score that is reused on the next run and exported as a reward of 0; a
  4DCodeBench scorer that did not finish (killed, as by the out-of-memory killer, or stopped by
  an error of its own after its gate, leaving a `reward.json` without a `dynamic_iou` reading;
  or done without one) is an `infra_error`, no longer `rejected` or `scored` without a value. A
  kept official score that its scorer now reads as its own failure is scored again, not reused.
- A 4DCodeBench world whose measurement did not finish is not scored, though the official
  scorer writes `error` for it and exits 1 as for a world it rejects: the plugin reads why in
  `reward.detail.json` (kept in the result as `causes` and `failures`). Its worker timed out,
  was lost or never became ready; this machine's disk, quota, processes, open files or GPU
  failed it, or its kernel refused memory (`ENOMEM`); the scorer image lacks `ffmpeg`,
  `ffprobe` or `scorer-vdb`; the task's case video or reference world could not be read; a call
  failed while the disk holding its output was full; or it left no readable
  `reward.detail.json`: an `infra_error`, redone and never a reward, no longer `rejected` and
  counted 0 in the mean. Its gates' verdicts, an empty first frame, a non-finite reading, a
  `MemoryError` (an allocation larger than the machine can give at all, which the world
  decides, as with an archive whose header declares a 384 TiB array) and a failure it does not
  recognize stay the delivery's result.
- The `local` backend gives a program absolute paths (its `HOME`, `TMPDIR`, output) when the
  work directory was given as a relative one: the program runs in its own directory, where a
  relative path pointed elsewhere (a reward of 0 for a program that writes to `HOME`).
- A Lab runtime (`r2s-python`, the images built on it, the `local` backend) that lacks what it
  promises (NumPy, SciPy, the twin SDK, ffmpeg) is found after a program fails: an
  infrastructure error with the repair command, not the program's failure.
- A rotation in a twin's tracks must be a finite quaternion of unit norm wherever the object is
  visible: `NaN`, infinity or a zero quaternion is `invalid_output`, not a score.
- Huge but finite positions (a diverging solver, a reference in large units) no longer overflow
  into `Infinity`, `NaN` or wrong values: the trajectory and centroid-set metrics, aligned
  (`translation`, `similarity`) or not, the skill, the noise floor and `error_to_noise` are
  computed without overflow, and so are tracks sampled at another frame rate, the centroids of
  a 4D world, bounces and the report's plots (a skill halfway to such a reference is 0.5, not
  `NaN` clipped to 0; a prediction 1e160 times the reference aligns onto it, an error of 0
  rather than 0.1 m; results for positions below 1e60 are unchanged). A distance of 1e200 is
  that distance rather than "missing frames", and a value beyond the largest float is a metric
  error. The Lab's own errors and scores in result records are finite numbers (`skill_raw` and
  `error_to_noise` are null where their ratio is beyond the largest float), so the records
  parse as standard JSON.
- Huge positions no longer stop an experiment: a twin at x = 1e307 on a `video_only` task
  made the similarity alignment fail (`error: SVD did not converge`), and `r2s eval` stopped
  with no `results.json` (`r2s train reward` too). Such positions are aligned like any others,
  and an alignment that cannot be computed is that twin's metric error, scored as a failure.
- A valid task id longer than a file name may be (many path segments) is cut to 120
  characters in the Lab's directory names and told apart by a digest, instead of failing on
  the file system.
- A task holding a file whose name is not UTF-8 (unpacked from an archive made on Windows,
  its name in another encoding) is evaluated and exported like any other: fingerprints count a
  file's name as the bytes the file system holds (the same as before for every UTF-8 name, so
  existing fingerprints and caches still hold). Before, `r2s eval` stopped with `'utf-8' codec
  can't encode ...`, naming no file, though `r2s task validate` passed the task, and
  `r2s export harbor` and `r2s train export` said the task changed while it was exported, with
  the fix to export it again, which failed the same way every time. `solve.sh` (written by
  `r2s export harbor --solution` and the `harbor` method) recreates such a file of the program
  under its own name, instead of the export failing with the same encoding error.
- The `local` backend refuses, before running anything, a task whose launcher exists only in
  its runtime image (PhysCodeGym's Blender scene, 4DCodeBench's `build.sh`), with the fix
  (`--backend docker` or `bwrap`), instead of scoring the missing launcher as 0.
- Retries of infrastructure failures stop once the agent has worked (its cost counts on every
  attempt); an attempt where no agent ran is not charged to the budget.
- Each scorecard entry counts its own deliveries (`BenchmarkScore.delivered`), so a mixed
  experiment's delivery rate is per benchmark.
- A task a method failed counts against it on every benchmark: a native task it delivered
  nothing for, or whose program wrote no valid output, gets the official scorer's result for
  nothing (S_mot 0, `dynamic_iou` 0, V2WScore 0, PhySimCode's true parameters counted as not
  recovered, the worst value of every Physics-IQ metric) instead of being left out of the
  benchmark's score.
- A run directory is only the Lab's and one command's: `--out` holding other files is
  refused, a second command on a run directory in use stops with an error (`state/lock`), a
  run that cannot start removes the directory it created, and a benchmark selection with a
  task removed by hand is built again rather than evaluated on what is left
  (`built_tasks.json`).
- `--include`, and `include` and `exclude` in an experiment file, choose tasks by id before
  they are validated: a task they leave out (one that `r2s task new` has just begun, without
  its reference) is no longer validated and no longer stops the run. A task that is chosen, or
  whose id cannot be read, is still refused, now with a fix.
- What a run has spent (`state/budget.json`) is written atomically and never taken as nothing
  spent: a damaged record stops the run with the fix, and a missing one is counted from the
  run's agent records. The wall time is kept there too (`wall_time_s`): `max_wall_time_s`
  counts every command on the run, as the agent-run and cost budgets do, so running the same
  command again, or resuming it after Ctrl-C, no longer starts agents on a new time budget.
- What an agent run cost is never taken as less than it was: a `state/budget.json` without
  `cost_usd` is damaged (it was taken as $0 and the cost budget no longer held); a run that did
  not say what it cost is a run of unknown cost (`cost_unknown_runs`), no longer $0; so is a run
  whose agent record cannot be read or is missing while its build record names it, when a
  missing `state/budget.json` is counted again (it was counted as $0, or not at all), and such
  a count says it is at least what was spent (a build done again holds its last run's record
  only); and so is a run whose build failed on this machine after the agent started (its record
  cut by a full disk, or never written; also for Harbor and `mini_swe`, whose harness on the
  host writes its logs only once it ends), no longer given back and run again at once, and a
  run stopped once its agent began (Ctrl-C, SIGTERM, a closed terminal, `kill -9`): a run is
  counted, as one of unknown cost, before its agent starts, until it says what it cost (a run
  whose agent never began is given back, also one whose harness could not be spawned or was
  stopped before it started). A cost a run says that is no amount of money
  (`1e309`, read as Infinity; NaN; a negative cost; `true`; a whole number too large for a
  float) leaves it unknown too, as does a total too large to be a number: neither the agent
  record nor `state/budget.json` holds Infinity or NaN any more (which stopped every later
  command on the run directory as damaged, also once the file was removed as the fix said), a
  negative cost no longer gives back what was spent, and a huge one no longer crashes the
  experiment (which ran the agent again on every retry).
  `max_cost_usd`, which every Codex, Gemini CLI or Harbor run passed as $0, refuses those agents
  before the experiment starts, and starts no further agent once a run's cost is unknown (a
  `mini_swe` model LiteLLM has no price for). A scripted `mini_swe` run and `fake_agent` cost
  $0. The report gives each agent run's cost, or says it is unknown.
- The agent records say what happened where the agent ran: on bwrap, `agent_version` is what
  the CLI in the agent's image says, or left out when the image has none (it was the host's CLI
  of that name, also in the twin's origin); token counts of 0 stay 0 (no longer unknown);
  `mini_swe`'s trajectory (`agent/trajectory.json`, one JSON document) is shown in the report and
  exported with a dataset like any agent's, read from where its record names it (a report and a
  dataset left it out), and its record names a log only once it holds no injected credential.
- Ctrl-C and `kill` stop an experiment at once: running sandboxes and containers are stopped,
  nothing is scored for a step that was stopped, the exit status is 130, and the same command
  resumes. This includes downloads a step waits for (a scorer's pinned code, an image pull) and
  video decoding; a download that never finishes (a proxy that stopped answering) ends with an
  error after a time limit, and every other tool the Lab runs outside a sandbox has one too (a
  hung Docker daemon is an error that names the command, not a wait for ever).
- A run that failed because of this machine's storage (a file at the file-size limit,
  `ulimit -f`, or a full disk) is an infrastructure error, not the program's failure and not a
  reward of 0, also when the program (or an agent) caught the failed write and exited normally
  with a missing or cut output; the Lab's own writes to a full disk end in one error line with
  the fix.
- A frozen re-run that something outside its program stopped or changed is an
  `infra_error`, no longer a `failed` or `timed_out` run scored 0 (a reward of 0, reused by the
  next run and exported): its sandbox killed from outside (bwrap or the Docker client dying of
  a signal, which nothing in the sandbox can send; not inside a systemd scope, where its
  program's lack of memory can also end bwrap), its frozen twin changed from outside the sandbox
  while it ran on `docker` or `bwrap` (where the program cannot write it; the experiment
  neither retries it nor stops, and builds it again on its next run), or this machine's
  storage, now also for a program that retried its write until the time limit and for a file
  in the `local` backend's temporary directory or a twin's scratch tree (a write stopped in the
  sandbox's tmpfs `/tmp` or to an anonymous temporary file leaves no file to find, and stays
  the program's failure).
- What a program leaves in its run directory (a directory without read or search permission,
  a path longer than the system allows) no longer stops the check of this machine's storage
  when a file-size limit is set (`ulimit -f`): the run keeps its own result, where the command
  ended in an error and an evaluation stopped. The check follows no links and looks at 10,000
  entries at most.
- An agent's build is redone when the image it works in is rebuilt under the same tag (the
  build's key includes the image's content id).
- A method is scored only on tasks it can deliver for (`Method.applicable`): the reference
  baselines on a 4DCodeBench `build.sh` task, or `oracle` without a truth generator, are
  reported as not applicable, not as scoring 0.
- Reports keep the videos of recordings that may not be redistributed out of `report/` (in
  `local-media/` beside it), so sharing a report shares no restricted data.
- A damaged output is the program's result, never a crash of the Lab: a truncated or
  malformed archive, a number where a trajectory belongs, a named pipe or device in place of
  a file, or JSON of the wrong kind is `invalid_output`; reading a video has a time limit.
- What an agent leaves in its workspace and log directory is read as it is once its sandbox
  has ended, never through a symbolic link or by waiting on a named pipe: a `solution/` that is
  a link is not delivered (in `r2s eval`, `r2s train reward` and the Harbor verifier); a session
  output that is a link, or holds one, a pipe or a file the Lab cannot read, is not used (a
  native task scored on it scores no output, and the official scorer never reads through it);
  the Lab writes `agent_record.json` as a new file, never through what the agent left under
  that name; the agent's logs, `TASK.md` and trajectory are read, recorded, reported and
  exported only as regular files, and `r2s train export` redacts the statement and trajectory
  again (the values of the exporting shell's credential variables, named so by a whole word
  and long enough to be one: no longer a `GIT_AUTHOR_NAME` or a `*_MAX_OUTPUT_TOKENS` count,
  which left lines of the data unreadable). A session output whose files do not read as their kind (an archive cut short, JSON
  nested too deep) no longer stops `r2s eval` with a traceback or makes the Harbor verifier
  give no reward: it does not match its re-run.
- An output this machine cannot read after its program ran (no permission, an I/O error, a
  stale network file handle, too many open files) is an `infra_error`, no longer the
  program's `invalid_output`: no score or reward (`r2s train reward` exits 1), and the same
  command runs it again instead of reusing a 0. What the output's own files cause (a name too
  long, content a reader rejects) stays `invalid_output`, and `r2s task validate` reports a
  reference it cannot read as before.
- A run that failed because its runtime cannot start its interpreter (a damaged image or
  unpacked root file system) is an infrastructure error with the fix, not the program's
  failure: after a failed run, the Lab starts the interpreter alone in the same runtime, in
  isolated mode, so nothing of the program can make that check fail.
- A task that cannot be scored is refused before anything runs: scoring options a metric does
  not take or values it does not know (`align = "similiarity"`) are errors of
  `r2s task validate` and of the scorer, and `r2s train reward` refuses such a task, or one
  whose reference is missing, instead of giving a reward.
- A reused official score is checked against the report it was computed into: a
  `native_report.json` edited or damaged since is computed again.
- Reports write their media afresh on every build (a recording whose task may no longer be
  redistributed leaves `report/`), and an output changed since it was scored is not drawn
  next to that score; the task's page says so.
- `r2s eval` refuses a run directory inside a task, program or checkpoint it evaluates, and
  checks method options before the machine (a value out of range is reported even where
  Docker is not ready yet). A closed terminal (SIGHUP) stops an experiment like Ctrl-C, and
  `r2s train reward` stops its program the same way.
- `r2s eval`, `r2s run` and `r2s demo` exit with status 1 when nothing could be scored (every
  result an infrastructure error; before, 0), and say why: a run that failed outside the
  program carries its cause (its run record's problem) into the result's `infra_error`,
  `r2s.log`, the summary and the report, not only "the sandbox failed around the program",
  and running the same command again is advised for once the cause is fixed. A task whose
  official scorer cannot run here is refused with a fix (for `r2s run` and for `r2s eval`).
- An experiment whose `r2s.log` can no longer be written (a full disk, `ulimit -f`) ends with
  one error line naming the file and its fix instead of a traceback and a wait for ever; a
  terminal or pipe that stops taking its lines (a closed terminal, `r2s demo | head`) no longer
  stops or stalls it (the log file has every line). A command whose output its reader stops
  reading (`r2s demo | head`) ends quietly with the status 141 of a program `SIGPIPE` ended,
  instead of Python's error at exit or the advice to check the network.

### Training

- `r2s train tasks`: training tasks (`split = "train"`, ids `synthetic-train/...`): the
  synthetic phenomena with their hidden parameters and initial states sampled around the
  evaluation tasks' (deterministic per seed, never a near copy of an evaluation task and never
  a video the overlap check cannot tell from one), labelled with the exact trajectories,
  `parameters.json` and the truth program. Every synthetic task now has
  `hidden/parameters.json`.
- Your own training data: tasks marked `split = "train"`, or a benchmark adapter that declares
  a training split (`BenchmarkAdapter.splits`, `r2s bench build NAME --split train`). A task
  claiming the training split of a registered benchmark that releases none is refused.
- `r2s train export` (`export_dataset`): tasks and the runs on them as a dataset of inputs,
  labels, delivered programs with their rewards and agents' redacted trajectories.
  `--against EVAL_TASKS` leaves out every task that holds the same file or the same recording
  (a copy, re-encoding, resize, clip, frame-rate or brightness change) as an evaluation task
  (`real2simlab.task.overlap`), and records what was compared and left out in `export.json`.
  Each reward is exported with the task version it was scored on (matched by content hash) and
  the program that earned it (a program changed since it was scored is refused); only complete
  run directories are exported; results the Lab could not score (`infra_error`) are never
  rewards, with or without `--include-undelivered`; undelivered results only with
  `--include-undelivered`; `overlap-prints.npz` holds the prints of the exported tasks only
  (not of those `--min-reward` or another filter left out); the dataset is replaced only
  once the new one is complete, and `--overwrite` replaces only an exported dataset. Evaluation
  tasks are exported only when asked, and never those of a benchmark whose terms forbid
  training on its data (`BenchmarkAdapter.training_terms`, `training_refusal`: Physics-IQ, and
  PhysCodeGym's 27 clips taken from it).
- `r2s train fit` (`fit_method`): trains a method that learns (`Method.trainable`,
  `TrainedOptions`, `train(context)`) on a dataset and writes a checkpoint with its record
  (`r2s-training/1.0`) and the training tasks' overlap prints. A method given `checkpoint=DIR`
  is not evaluated on a task that overlaps one of its training tasks (the experiment lists it
  as not held out), and its build cache key includes the checkpoint's content. A dataset
  without the prints of all its tasks is refused, and `--overwrite` replaces only a checkpoint.
- `nearest_twin`: a small learned Real2Sim method for the synthetic phenomena, trained and
  evaluated through the whole loop in the Training guide, with its learning curve.
- `r2s train reward` (`score_solution`, `reward_of`): one solution's reward, computed by the
  evaluation's own code (the Harbor verifier uses it too). A run that failed outside the
  solution gives no reward (`RewardUnavailable`, exit status 1), nor does a solution this
  machine could not copy or package (`InfrastructureError`, exit status 1); `--work`
  directories can be reused.
- Task configurations gain the optional `split` (`eval` by default).
- Datasets and checkpoints are used only as written: `export.json` records the SHA-256 of
  every exported file but itself and the card `README.md` (`content`) and `r2s train fit`
  refuses a dataset changed since, naming
  the file (one whose `export.json` is missing, is not an export record or records no content
  is refused too), and a malformed `tasks.jsonl` row by its line; a checkpoint records the
  digest of its files (`content_digest`, optional in `r2s-training/1.0`) and one changed after
  `r2s train fit` wrote it is refused, as is a `nearest_twin` model that cannot be read, when
  the evaluation starts.
- A dataset or checkpoint record without its digest is refused, not taken as unchanged,
  whatever its `lab_version`: an `export.json` without `content`, a `training.json` without
  `content_digest` (edited, or written by a pre-release build of the Lab: a dataset or
  checkpoint made by a development build before the 1.0.0 release must be exported or trained
  again). A checkpoint whose `prints` names a file outside it, or whose prints leave out one of
  its `task_ids`, is refused when the evaluation starts (an edited record turned the held-out
  check off), and `r2s train fit` refuses a dataset whose overlap prints do not hold one of its
  recordings, by content (a recording replaced since the export would be learned from but
  never held out).
- A task's labels are checked: `r2s task validate` reports a `hidden/parameters.json` that
  cannot be read, is not JSON or is not a JSON object (it said the task was fine), and
  `r2s train export` stops on one with an error naming the file and the fix (it failed with the
  decoder's words alone, or exported the broken labels); `nearest_twin` names the `tasks.jsonl`
  line whose labels are not a truth it can learn from, and refuses a model holding such labels
  when the evaluation starts (a traceback, or an evaluation whose fix, training again, failed
  the same way). A dataset's `tasks.jsonl` or a checkpoint's `training.json` holding bytes
  that are not text is an error naming the file, with the fix.
- `r2s train export` reads each reward from the very bytes it checked against `results.json`,
  and checks every input again just before it publishes the dataset: a result file replaced, a
  task edited or an `--against` evaluation task changed while it ran stops it (it paired a
  reward with a result file another command rewrote, and recorded a comparison with evaluation
  tasks that were no longer those).
- `r2s train export` does not export a delivered result that names no fingerprint of its
  program (it was paired with whatever program was in the run directory), and reports a result
  whose program cannot be matched as such, with the fix that works: the same `r2s eval` or
  `r2s run` command again on the run directory, which scores it again (it was reported as a
  result whose task cannot be matched, with `--tasks` as the fix).
- `r2s train fit` writes no checkpoint when the dataset was replaced or changed while the
  method trained (`r2s train export --overwrite` to the same directory), `r2s train export`
  stops when a program or a task changes while it is copied, and `r2s train reward` gives no
  reward when the task was edited while the solution was scored: each would pair two versions.
- The same holds when the replacement made something fail first: `r2s train fit` whose method
  (or a check before training) could no longer find a file because `r2s train export
  --overwrite` replaced the dataset meanwhile says the dataset was replaced, with its fix (it
  said `error: .../capture.json does not exist`, without one); `r2s train reward` whose task
  was written again before the scorer read its reference, and `r2s train export` whose copy of
  a task or a program failed because it changed meanwhile, say the input changed (they named
  a missing reference or file).
- `export.json` records which files are executable (`executable`), and `r2s train fit`
  refuses a dataset in which a file's executable bit was cleared or set since, or to which a
  symbolic link was added: a zip archive or a sync that keeps no file modes changed how its
  programs run while their rewards stayed. A dataset exported before has its files checked,
  not their modes.
- `r2s train tasks`, `r2s bench build` and an experiment's benchmarks build a name given twice
  (`--select pendulum --select pendulum`, a tier or a variant twice, a Physics-IQ scenario and
  its center view, a Video2World family and one of its samples) once: in a new directory it
  failed with "task directory already exists", and with `--overwrite` it built everything
  twice and counted it twice.
- `r2s train export` that leaves out everything it was given gives the fix for why: every task
  overlaps `--against` (check that it names the evaluation tasks), every reward is below
  `--min-reward` (the best one is named), nothing was delivered (`--include-undelivered`), the
  results were not scored (fix the machine and run the same command again), or
  `--skip-unmatched` left them out (the tasks as they were, or the same command again), instead
  of always "give training tasks"; the help of `--skip-unmatched` names programs too.
- `r2s train export` refuses a thumbnail cache (`R2S_CACHE_DIR`) inside a task it reads,
  wherever the task is (one a run found too): the overlap check wrote thumbnails into the
  task, called the task edited, and on the retry exported them as its data. A cache beside the
  tasks of a directory given is used.
- `r2s train reward --work` and `r2s twin check --out` refuse a directory another r2s command
  is using.
- A reward is of the program its path holds: `r2s train reward` (and `score_solution`) gives no
  reward when the solution was edited while it was copied (the copy could mix two versions) or
  scored (the reward would be paired with a program it is not the reward of), with the fix,
  and an empty `--solution` (an unset variable), here and in `r2s export harbor` (and
  `export_task`), or an empty `checkpoint=` of a trained method, is refused instead of taken
  for the current directory.
- PhysCodeGym: a method gets the reference at 1280×720 (was 640×360), the largest frames
  PhysCodeGym's own observation tools show, encoded one frame at a time
  (`media.write_downscaled`); the statement states its rule that `scene.py` is one file that
  imports no module of its own.
- Content checks ignore what a file manager leaves behind (`.DS_Store`, `._*`, `Thumbs.db`):
  opening a run directory, dataset or checkpoint in Finder does not make its programs
  "changed", force a rebuild or make a dataset or checkpoint refused.
- CONTRIBUTING says how to reward with another score: a metric plugin named as the training
  tasks' primary metric (datasets and online rewards), or an official scorer on training tasks
  (datasets only).

### Backends

- `docker` (no network, read-only root, all capabilities dropped, the host user's uid), `bwrap`
  (Linux without root or Docker: images unpacked to a root file system, GPU passthrough, memory
  and process limits through a systemd user scope where available) and `local` (debugging only,
  marked not isolated).
- Runtime images are pulled when missing (from the project's GitHub Container Registry, or
  registry mirrors through `R2S_IMAGE_MIRRORS`) or built locally; `r2s env unpack --from-url`
  brings an image to machines without Docker.
- Runs survive hostile leftovers (directories without permissions, very deep trees, symbolic
  links in place of the output) and always write a run record.
- Docker backend: a program that exits with 125 is the program's failure with its exit code,
  no longer an infrastructure error retried as one (Docker's own start failures still are).
- bwrap backend: the CPU limit is recorded as enforced only where `taskset` applies it;
  otherwise a `CpuLimitWarning`, and `r2s doctor` says what to install.
- bwrap backend: GPU runs and `r2s doctor --gpu` find `ldconfig` in `/usr/sbin` or `/sbin`
  when PATH leaves them out (an ordinary user's PATH on Debian), where both stopped with
  "ldconfig does not exist"; the first GPU runs started together share one copy of the driver
  libraries instead of clearing each other's; and a problem with those libraries is a failed
  `gpu via bwrap` check with a fix, not the end of `r2s doctor`.
- Docker backend: every Docker call (agents' included, e.g. `mini_swe`) uses the daemon the
  host's client is configured for (`DOCKER_HOST`, `DOCKER_CONTEXT`, `DOCKER_CONFIG`, TLS
  variables), and a run's cache identity includes the image's content id.
- Run as root with Docker, the Lab refuses read-only mounts the container's unprivileged user
  cannot read, with the fix (`r2s demo` works as root: its scripted agent's files are readable).
- Docker backend: the agent's home (a tmpfs) is writable by the agent's user whatever the
  image's directory mode (runc gave the tmpfs the mode of the directory it covers).
- bwrap backend: images unpack when the cache and the temporary directory are on different
  file systems.
- bwrap backend: `r2s env unpack` can run several times at once (the index of unpacked images
  is locked, and a damaged index is rebuilt from the images); a damaged record of an unpacked
  image is named, and `--force` repairs it; an image unpacked without the mount points
  `/r2s`, `/workspace` or `/app` gets them (the Blender runtime could not start PhysCodeGym
  twins).
- Docker backend: a run uses the image by the content id it resolved (a tag moved between
  the check and the run cannot change what runs), and mount paths holding commas or quotes
  are quoted (Docker reads `--mount` as CSV).
- bwrap backend: a command runs only on the unpacked image it resolved. An image replaced,
  removed or given another `image.json` while the command runs (by hand between two seeds, or
  by `r2s env unpack --force` between two runs) makes its later runs, their retries and its
  agent sessions infrastructure errors that run nothing (redone when the command runs again),
  instead of results filed, and reused, under the content id the command resolved.
- `r2s env unpack --force` extracts an image again (repairs a damaged root file system);
  unpacking again without it says that it reused what was there.
- bwrap backend: the readiness check (`r2s doctor`, and every command before it starts) starts
  the sandbox every run gets, with its new `/proc`: where a machine forbids that (an
  unprivileged container that masks its `/proc`), bwrap is no longer reported ready while every
  run then fails; it says so with a fix of its own, as for user namespaces (AppArmor), and
  `r2s demo` falls back to local. A run whose sandbox bwrap could not set up is an
  infrastructure error that says so, not a runtime to unpack again, and the repair of a
  damaged unpacked image names `--from-tar` for machines without Docker.
- Docker backend: when Docker cannot be used, the fix names what chose the daemon
  (`DOCKER_HOST`, `DOCKER_CONTEXT`, or a context chosen with `docker context use`) instead of
  starting this machine's own or joining the docker group (which reaches only this machine's
  own daemon), a daemon that does not answer `docker info` is no longer taken for a missing
  installation, and the GPU check says that Docker is not ready instead of that the NVIDIA
  runtime is not configured. The 4DCodeBench and Video2World scorers, which run in Docker
  whatever the backend, say Docker is not ready instead of that their image is not built.
- `r2s doctor` asks each backend once (a hung Docker daemon cost two waits), says whether the
  bwrap runtime image is unpacked where bwrap can run (and names `--backend bwrap` in Docker's
  fix), and no longer reports GPUs ready through a backend that cannot start its sandbox.
- A command whose backend cannot run here (`r2s eval`, `r2s run`, `r2s twin check`, `r2s twin
  run`, `r2s train reward`; Docker unless given another) names each other isolated backend that
  can in its fix, e.g. `--backend bwrap` on a machine without Docker, where it said only to
  install or start Docker; it never moves to another backend by itself.
- `r2s env unpack --from-url`: a download that gets no answer (a stalled proxy, no network)
  names the network and the proxy variables set, no longer `--sha256`, which only moved the
  wait to the image's own download; a published checksum file that holds none, a `--sha256`
  that is not one, a `--sha256` the file does not match, and an image the server does not have
  (404, not the network) each say so; a path on this machine is refused (`--from-tar` takes
  it).
- What a program or an agent prints no longer fills the disk: a run's `log.txt` (and an
  agent's `sandbox.log`) keeps at most 16 MiB of it (the beginning and the last MiB, with a
  line saying how much was left out), an agent's `trajectory.jsonl` and `stderr.log` at most
  256 MiB each, in whole lines (a longer line is left out whole, never cut before redaction),
  all under the machine's file-size limit (`ulimit -f`). So do the agents whose harness runs
  on the host: what Harbor and the mini-swe-agent harness print (16 MiB, read as it comes, no
  longer all in memory), the copy of Harbor's trial (`agent/trial`, links left out) and
  mini-swe-agent's `trajectory.json` (256 MiB). The Docker backend and the official scorers'
  containers turn Docker's own log of a container off (`--log-driver none`), which kept a
  second copy without a limit on the daemon's disk. A program that prints a lot is no
  longer stopped at that limit and reported as an infrastructure error with the advice to
  raise it, one that prints in a loop is its own failure, and an agent whose output cannot be
  written (a full disk) no longer blocks until its time limit.
- bwrap backend: the sandboxes that run at the same time on one machine, in one command or in
  several of the same user's (lock files in `/tmp/r2s-cpus-<uid>`, released when a command
  ends, even with `kill -9`), are pinned to CPUs of their own while there are enough (all were
  pinned to the first CPUs: with `--concurrency`, or two commands at once, twins shared one
  CPU, ran into their time limit and were scored 0); beyond that they share the least used
  ones, with a warning.
- Local backend (Linux): every command runs under a supervisor, so what a program starts ends
  with its run (a process in a session of its own too, and an agent's CLI on Ctrl-C, `kill` or
  its time limit), and a program stops when `r2s` is killed with `kill -9`. A process that is
  stopped gets `SIGTERM`, then `SIGKILL` five seconds later.
- `r2s env unpack NAME --from-tar FILE` (and `--from-url`) unpacks the image FILE names as
  NAME (the names `docker save` gave it, read as Docker reads them, under any registry prefix)
  and refuses another one, saying what FILE holds and how to unpack that: any tarball was
  registered as any name (a tarball of r2s-python as r2s-blender-4.0 or as a misspelt name, so
  a twin calling Blender failed in r2s-python and was scored 0). A tarball saved by image id
  names no image and is unpacked as NAME, with a note. Unpacking writes each file of a layer as
  it reads it (it held a whole layer in memory: 1.4 GiB for r2s-blender-4.0, a `MemoryError`
  under a memory limit). An unpacked image whose tree was removed while its record stayed is
  damaged, not "not unpacked": its fix names `--force` and `--from-tar`, and unpacking it again
  extracts it again (the fixes of the run and of the unpack pointed at each other).
- `r2s env unpack` (`--from-tar`, `--from-url`, and with Docker) checks each layer as it reads
  it, as `docker load` does: its uncompressed contents, read to their end, have the digest its
  image lists (`rootfs.diff_ids`, which the image's id covers), and a file stored under its
  digest (`blobs/sha256/...`) has that one. A copy with one byte changed in a layer was unpacked
  under the id of the image saved (a correct program then failed, or scored 0, under that id),
  and a damaged entry header silently left the rest of its layer out; such a copy is refused,
  naming the file and the layer, and nothing is unpacked. `--sha256` without `--from-url` (with
  `--from-tar`, or from Docker), which nothing checked, is refused; a tarball whose only name is
  empty or `<none>:<none>` is taken as saved by image id.

### Methods and agents

- Baselines `oracle`, `static`, `constant_velocity`; `program` (a directory of programs);
  `fake_agent` (a scripted agent, no model).
- `program:path=` takes a directory the way a reward does: one that holds `run.py` is the
  program for every task, also when a subdirectory holds a `run.py` of its own (it was taken
  for a directory of per-task programs, so the task delivered nothing or got the
  subdirectory's program); one that is also a directory holding a program for a task is
  refused as ambiguous before anything runs, and so is an empty `path=` (in an experiment file
  too). A program holding a symbolic link or a file it cannot read, or reached through a
  symbolic link (a per-task directory, a twin package's `program/`), delivers nothing for its
  task, with the fix, as its reward is 0 (the whole evaluation stopped, for every task and
  method), and one edited while it was copied is not a result (`infra_error`) and is built
  again.
- Coding agents `codex`, `claude_code`, `gemini_cli`, `opencode` (experimental), `mini_swe` (any
  LiteLLM model) and `harbor` (every agent Harbor supports): credentials injected at run time
  only, output redacted, an egress allowlist through a logging proxy by default, an
  unprivileged uid when the host user is root, CLI versions pinned and recorded. Codex works
  in the image that matches the task's runtime (`r2s-agent-codex-blender`, `-physimcode`,
  `-video2world`), `mini_swe` and `harbor` in the task's runtime image, every agent in a
  benchmark's own agent image; Claude Code, Gemini CLI and OpenCode in their own Python image on
  the Lab's other runtimes. A method's
  `time_budget_s` is also the time its statement states (the Lab's statement and
  Video2World's), not only the time it is given.
- An agent's key goes only to the endpoint its options name: `opencode` with `base_url` runs the
  model there as `custom/<id>` (`openai/<id>`, the name `mini_swe` takes for the same endpoint,
  no longer selects OpenCode's own `openai` provider and `api.openai.com`; `base_url` needs a
  model); `ANTHROPIC_AUTH_TOKEN` without `base_url`, and a host's `ANTHROPIC_BASE_URL` or
  `OPENAI_BASE_URL` for the agent's provider while the method names no endpoint (`claude_code`,
  `codex` with `provider = "api-key"`, `opencode`, `mini_swe`, also for a model named without its
  provider, such as `gpt-4o`), are refused before anything runs with the option to set, and
  `r2s doctor` says so instead of `ok`.
- An agent method without its credentials (no key, no login, a configuration it cannot use) is
  refused before any agent run is reserved: the same command, once its `fix:` is done, is not
  refused for a spent `max_agent_runs`. On the local backend, the paths in an agent's environment (Codex's
  `CODEX_HOME`) are rewritten to the host once, not twice to a directory that does not exist.
- `codex` reproduces the host's custom provider whole (`query_params` such as Azure's
  `api-version`, `http_headers`, `env_http_headers` with the variables they name, retries), only
  its key source replaced, and the provider and model a `profile` of `~/.codex/config.toml`
  chooses (a profile it does not define is an error), instead of OpenAI's defaults.
- Agent records count tokens the CLIs' way: Gemini CLI's from its `result` event's `stats`,
  OpenCode's from its `step_finish` events, Claude Code's with the prompt cache's tokens as
  input (as Codex's are); `reasoning_effort` is refused, and recorded as null, for the adapters
  that pass none to their CLI (`claude_code`, `gemini_cli`, `opencode`, `mini_swe`).
- A credential reaches no record, command line or shared file: a method, benchmark, backend or
  scorer option that holds one (`model_kwargs.api_key`, a key in Harbor's `agent_kwargs` or
  `agent_env`, also under a name like `AWS_BEARER_TOKEN_BEDROCK`, a key with its flag in
  `extra_args`, or as `NAME=VALUE` there (`--env OPENAI_API_KEY=...`, codex's
  `-c ...bearer_token=...`) or as a header (`Authorization: Bearer ...`), a URL with a
  password) is refused before anything runs, with the variable to
  name instead (`r2s task validate` lists one in a task's scorer options); Harbor gets
  `agent_env = {KEY = "$NAME"}` as `KEY=${NAME}` and reads the value itself (an unset variable
  is an error with its fix, not an empty key); a short value of an injected login file (a
  password or key under its name, in JSON, netrc, TOML or `.env` form) is redacted like an
  injected variable, also when printed line by line; and injected values an agent writes
  itself, into its solution or straight into its logs, also in files it made unreadable, are
  replaced with `[REDACTED]` on the host before its twin is frozen, reported or exported (a
  file that cannot be rewritten is removed; only this machine's failure, such as a full disk,
  makes the build not a result), and counted in its record (`mini_swe` and `harbor` records
  count theirs too).

### Benchmarks

- Physics-IQ (predict 5 seconds of a real video; the official motion-mask metrics and score,
  matching the official pipeline exactly, with the static and second-take references). The
  static reference holds the clip's actual last frame.
- Video2World (robot and hand manipulation videos reconstructed as a scene and an action
  stream, scored by Video2World's own evaluator), with the delivery convention and output
  format `video2sim_package` and the `r2s-video2world` image. Its headline is V2WScore on
  Video2World's own scale (0 to 100), combined over tasks as Video2World does: the mean within
  each family, then over families.
- Video2World in full: all 222 instances of its 12 evaluators (the hand, MuJoCo twin and cloth
  families, and the RoboDojo and in-house Isaac Sim families), named `family/sample` (a hand
  sample is in an arm and a two-hand family), each with its source's license (DexYCB, HOI4D and
  HOT3D's hand data are non-commercial), the public handouts Video2World stages for its
  profiles, and the evaluator's own image: a MuJoCo interpreter in `r2s-video2world`, and the
  new `r2s-video2world-isaac` (Isaac Sim 5.1, built only after `R2S_ACCEPT_NVIDIA_EULA=1`,
  never published). The evaluators' shared data and RoboDojo's assets are downloaded at score
  time, resumably. Only Push-T and DROID have been run end to end so far.
- Environments can require accepting a license before they are built (`accept_terms`);
  `r2s env build all` skips those.
- 4DCodeBench (the official scorer in its own image, plus frozen re-runs scored by Lab metrics).
- PhysCodeGym (Blender scene programs scored with the official pixel and motion metrics).
- PhySimCode (2D simulated physics: the frozen program must regenerate the video; PhySimCode's
  own parameter recovery), with the `r2s-physimcode` image.
- IRIS (real dropping-ball videos; trajectories extracted from the videos; the other takes of
  each height as the noise floor).
- A synthetic suite: five phenomena with exact truth, two visibility tiers.
- One license rule for third-party benchmarks (licensed, none declared, forbidden) in the
  add-benchmark skill and tutorial.
- PhySimCode's headline is now its end-to-end recovery, the share of *all* true parameters
  recovered within 20 % (`n_within_tolerance / n_gt`, pooled over tasks as PhySimCode pools
  them), not the accuracy among the parameters a delivery happened to report; a delivery
  without a readable `cot.json` still counts its true parameters. The example run's PhySimCode
  number is 0.300.
- Benchmark caches (vendored checkouts, Physics-IQ's index and clips, IRIS, PhysCodeGym,
  PhySimCode, Video2World and 4DCodeBench data) are written durably and checked before reuse;
  an entry a crash left empty, truncated or unreadable is fetched again with a
  `DamagedCacheWarning` instead of failing on every retry.
- A cache entry that opens but is not what the Lab wrote there (IRIS's observations or the
  overlap check's thumbnails with a field missing, of another type or shape, or a single
  array where an archive belongs) is damaged like a truncated one: computed again with a
  `DamagedCacheWarning`, not a traceback (and no longer a take silently left out); an intact
  `.npy` entry is no longer taken for damaged. A vendored checkout is checked on every use, not
  once per command: one edited after a command first used it is fetched again, never handed to
  an official scorer as the pinned commit.
- PhysCodeGym's and IRIS's 4K recordings are decoded frame by frame (`media.iter_frames`,
  `read_downscaled`), with the same output bit for bit; a decoder killed for lack of memory
  is reported as such, not as an undecodable file.
- IRIS: selecting settings downloads and tracks only their takes, and each task's
  `extraction.json` holds its own setting's report.
- The READMEs and the docs count six benchmarks, as the paper does, and name the synthetic
  suite apart as our own.

### Tooling

- The `r2s` command line, with `r2s doctor` checking the machine (backends, images, ffmpeg,
  storage, extras, agent credentials) and naming a fix for each problem.
- Everyday mistakes end in one `error:` line with a fix, never a traceback or a score: a path
  that does not exist, an empty directory or a file where a directory belongs, a file that is
  not an image tarball, a command run again on its own output (`r2s bench build` and
  `r2s train tasks` say to pass `--overwrite`). `r2s train reward` refuses a `--solution` path
  that does not exist (an empty solution directory still scores 0). Commands never remove a
  directory they did not write: `r2s twin check --out` clears only an earlier check's
  directory, and `--overwrite` replaces only task directories. The Lab's warnings are printed
  once, as one line. Configuration files of the wrong shape (`methods = 123`, `options = 1`)
  and unknown names (`--select`, `--tier`, `--option`) are configuration errors; an error that
  is a bug in the Lab says so and where to report it.
- `r2s doctor --gpu`: an `nvidia-smi` that fails (no working driver) is a failed check, not a
  green one.
- The paper is on arXiv ([arXiv:2611.34849](https://arxiv.org/abs/2611.34849)): the READMEs link it and cite it with its
  authors, as do `CITATION.cff` and the package metadata.
- The project lives in the Real2SimHub organization: code at
  `github.com/Real2SimHub/Real2SimLab`, documentation at `real2simhub.github.io/Real2SimLab`,
  runtime images under `ghcr.io/real2simhub/`.
- More mistakes and machine failures end in one `error:` line with the fix that suits them: a
  path that cannot be accessed (an `--out` in a directory that is not writable; the cache,
  with `R2S_CACHE_DIR`, under a `HOME` that is not), that already exists or is of the other
  kind; a name longer than the file system allows (choose a shorter path) and a loop of
  symbolic links, which `r2s eval`, `r2s twin`, `r2s train` and `r2s export harbor` reported
  as a bug, with a traceback; a program the Lab runs that is not installed ("the program ...
  was not found: it is not on PATH", not "... does not exist"), or a working directory it is
  run in that is gone; a directory that is not empty where a finished one is moved (another
  command wrote it first), not "this machine's system refused the operation"; a video that
  ffmpeg cannot read; a specification model's validation, in one line instead of pydantic's
  report. A failed download is told by its cause, whatever wraps it: a Hugging Face repository
  that refuses this machine (gated, private, gone) says to accept its terms or to log in
  (`hf auth login`, or `HF_TOKEN`), and a network failure says to check the network and the
  proxy, also for IRIS's downloads in worker processes (Hugging Face's "Force download failed
  due to the above error" said neither), and a 4DCodeBench real-world case is "not
  redistributed" only when the dataset lacks its video. An error that only a bug raises (a
  `KeyError`, a `TypeError`, ...) is still reported as a bug, with its traceback, when it is
  raised while a failed download is being handled, and an error raised `from None` is told
  by its own message.
- `r2s twin check` takes a program directory of any name (`my twin`, `Twin(v2)`), as
  `--method program:path=` does. `r2s capture from-video` checks object names, the rotation of
  `--extrinsics` (a mirror is not one) and the capture id before it copies the video, and
  leaves nothing behind when it fails, so the corrected command works. Two `r2s twin run`
  (or `r2s capture from-video`, `r2s task new`, `r2s export harbor`) given the same output at
  once: one runs, the other is refused as one started after it is, with the fix.
- The Physics-IQ page names the Hugging Face login of the huggingface_hub the install has,
  `hf auth login` (there is no `huggingface-cli` from 1.0).
- Seeds, counts, trials and concurrency are checked on the command line (`--seed -1` and
  `--trials 0` are usage errors). Options given by name (`--option`, a benchmark's `options`,
  `backend_options`) are checked against what takes them: an unknown name or a value of the
  wrong type (`count=null`) is a usage error saying what is expected, and a single name for an
  option that takes several is a list of one (`variants=rerun`), not its letters.
- `--method NAME:key=value,...` (`r2s eval`, and `r2s train fit METHOD`) separates options
  only at the commas between them: a JSON value keeps its own, so several `agent_env` entries,
  an `allowlist`, `extra_args` or `entrypoint` of two items, `model_kwargs`, `gpus="0,1"` and a
  path with a comma can be given (they were cut apart and refused as
  `R2SError: expected key=value, got '"B":"2"}'`). A part that is not `key=value`
  (`--method codex:gpus=0,1`, `--option colour`) is a usage error naming it, with how to write
  the value as JSON (or to use an experiment file).
- `r2s run` without `--out` refuses an experiment `name` that is not one directory name (a path
  such as `../x`, `/abs/x` or `team/x`, `.` or `..`, blank, a control character, more than 255
  bytes) before writing anything, as a method label would be refused: the run went outside
  `runs/`, anywhere on the machine. `r2s eval` without `--tasks` or `--benchmark` ends in an
  `error:` line with its fix, not a bare line. A benchmark option it does not take
  (`--option relase=verified`, or in an experiment file's `options`) names the options the
  benchmark takes, and the fix the one meant (`release`).
- A run directory's records are read through one checked reader: a damaged or hand-edited
  `results.json`, `experiment.json`, result file or agent record stops `r2s report` and
  `r2s train export` with an error naming the file and how to rewrite it (a damaged agent
  record is noted in the report), and `r2s eval` and `r2s run` redo a step whose build, run or
  result record is damaged. The list of skipped steps `results.json` always held is now in
  `r2s-results/1.0` (optional `skipped`).
- Directories are known by what they hold, not by file names: `r2s eval` and `r2s run` write
  only into a new or empty `--out` or a run directory whose `experiment.json` or
  `state/status.json` the Lab wrote (not a directory of the user's with an `experiment.json`,
  a `state/` or an `r2s.log`); `r2s twin check --out`, `r2s train reward --work` and
  `--overwrite` (tasks, datasets, checkpoints) likewise check the content of their marker. A
  run directory may not hold a task, program or checkpoint it evaluates (an earlier run's
  delivered program would be removed when its build is redone).
- A directory is the Lab's only by a record or mark the Lab wrote there, never by names or a
  key any file may hold: a record counts only with what every record of its kind holds
  (`task.toml` its `spec = "r2s-task/..."`, `training.json` `r2s-training/...`,
  `experiment.json` the experiment's `r2s-experiment/...` and the `lab_version`,
  `state/status.json` and `export.json` the fields the Lab writes), so another tool's file of
  that name with some `lab_version`, or with an `r2s-` `spec` of another kind, no longer lets
  `r2s eval` write over the `results.json` and `report/` of the user's directory, nor
  `--overwrite` replace a dataset, checkpoint or task directory of the user's. `--overwrite`
  no longer removes a directory that merely holds a task's names (`visible/`, `hidden/`,
  `instruction.md`, as a task being written by hand does): a task build marks its directory
  until it has written `task.toml` (`.r2s-task-build`), so an interrupted build is still
  replaced. `r2s twin check` takes a directory without its marker for an earlier check's only
  when its `check.json` is one a check wrote (not a list of the user's checks with `ok`
  fields), and names only the files a check did not write; `r2s train reward --work`, used
  again, clears only what scoring wrote there and refuses a directory with other files, as
  `r2s twin check` does; a lone `state/lock` makes a directory a stopped run's only when a run
  wrote it; and only what a replacement left beside `--out` (`.<out>.old-<hex>` with the Lab's
  record, `.<out>.trash-<hex>`) is put back or removed. `r2s train tasks` and
  `r2s bench build` say when `--out` also holds tasks they did not write (`--overwrite`
  replaces only the tasks a command writes, and `--tasks` given that directory uses them all).
  What a command stopped midway left in its own directory is still its own: the twin
  `r2s train reward --work` or `r2s twin check --out` was saving when it was killed
  (`twin.partial`) is cleared with the rest, a mark (`.r2s-reward`, `.r2s-check`,
  `.r2s-task-build`) that a full disk kept from being written is removed again, and one cut
  short by a kill still counts, so the same command (for a task build, with the `--overwrite`
  it asks for) runs there again instead of refusing the directory. A mark is never read or
  written through a symbolic link left under its name (it could name any file): the command
  says to remove it.
- `--method program:path=` naming a path that does not exist, or a file, is an error before
  anything runs, not a score of 0; a directory without a program for a task is that task's
  result.
- Ctrl-C (or `kill`) stops the official scorers (Physics-IQ, PhysCodeGym, PhySimCode, the
  4DCodeBench checker and scorer, Video2World, their containers included) and Harbor's and
  mini-SWE-agent's runs at once, instead of waiting for them; a Video2World scorer that times
  out stops its container. After an interruption each command says what running it again
  does: `r2s eval` and `r2s run` resume, `r2s train tasks` and `r2s bench build` need
  `--overwrite`.
- What a command killed outright (`kill -9`, the out-of-memory killer) left beside its target
  under a temporary name, and nothing ever removed (an image half unpacked, a dataset, a
  checkpoint, a Harbor export, a download, a scorer's code, a benchmark's cache entry), is
  removed by the next command that writes there on the same machine, once no process holds it:
  each such name is held by a lock that ends with its process (what another machine's command
  left on a shared disk, or this machine's before it restarted, is left alone; a name that
  would be too long with the machine's tag, beside an output named with 225 to 237 bytes, is
  made without it, and left alone too). Running out of memory (`MemoryError`, or a fork or a
  mapping refused for it) ends a command in one `error:` line with a fix, not a traceback
  asking to report a bug.
- Building IRIS tasks (`r2s bench build iris`, `r2s eval --benchmark iris`, an experiment's or
  `r2s train`'s IRIS tasks) stops at once on Ctrl-C, `kill` or a closed terminal, killing its
  worker processes instead of waiting for the recordings they were downloading and decoding
  (a second Ctrl-C during that wait left it waiting for ever, with a traceback from every
  worker), while the programs they started (ffprobe, ffmpeg) still stop on Ctrl-C; an error
  stops it once the recordings before it are done (results come in order), and the rest are
  not begun. Its workers also end when the command is killed with `kill -9`. A request to Hugging Face that gets no answer (a proxy or network that accepts
  the connection and stays silent) ends after `HF_HUB_DOWNLOAD_TIMEOUT` seconds (10 unless
  set), retried as huggingface_hub retries, and the command reports a failed download with
  its fix: listing a dataset's files (PhySimCode, Video2World, Physics-IQ's verified release)
  waited for ever.
- A twin package (`twin.json` and `program/`, what `r2s twin check` takes) is accepted wherever
  a program directory is: `r2s train reward --solution` and `--method program:path=` score its
  program instead of reporting it undelivered (a reward of 0).
- Numbers are checked for their range where a wrong one would make every run fail or never
  end: `--timeout 0` or `nan`, `--min-reward inf`, an agent's `time_budget_s=0`, `cpus` or
  `memory_mb` at or below 0, mini-SWE's negative `step_limit` are usage errors.
- A number that is not finite is refused wherever a task, a capture, an experiment or a
  method's options give one: `cpus = inf` in a task's `[delivery.resources]` (above 0, so it
  passed as a positive number) or `--method NAME:cpus=1e309`, an output's `fps`,
  `max_size_mb`, a capture's frame rate, timestamps or gravity, a budget, mini-SWE's
  `temperature`. `r2s task validate` reports it and every command stops before anything runs,
  instead of `r2s eval`, `r2s run`, `r2s train reward`, `r2s twin check` and
  `r2s export harbor` ending in an `OverflowError` traceback (`r2s eval` leaving its run
  `running`).
- `r2s export harbor --solution` and the `harbor` method's `solution=` take only `oracle` or a
  program directory with `run.py` (or a twin package, whose `program/` is exported): a path
  that is not there (`oracel` is asked whether `oracle` was meant), a file or a directory
  without `run.py` is refused before anything is written, instead of an export (status 0)
  whose `solve.sh` recreated nothing, so that Harbor scored the task's own reference 0.
  `solve.sh` says when `run.py` fails; the verifier still scores the program.
- `r2s spec schema NAME --out DIR` writes only that object's schema (it wrote all of them,
  over the files of those names in DIR), and refuses an unknown name with `--out` too.
  `r2s eval --select` and `--tier` without `--benchmark` are refused: they were dropped
  without a word and every task ran (`--include` chooses task directories by id).
- `codex` with the host's provider names what is missing: a provider whose `env_key`
  variable is not set is told to export that variable (it was said to have no key source and
  told to set `R2S_CODEX_API_KEY`, which only the sandbox's Codex reads), one without any key
  source how to add one. Every fix names the `config.toml` in use (`codex_home=` or
  `$CODEX_HOME`, not always `~/.codex`), an auth command that cannot run gets a fix, and a
  `model_providers` that is not a table is an error naming the file instead of an
  `AttributeError` traceback. An `env_key` not written as a variable's name (in capitals,
  digits and underscores) is never shown in that error or its fix, which `r2s eval` also
  writes to the run's `r2s.log`: it may be the key itself, put there by mistake.
- `r2s twin run` and `r2s export harbor` refuse an output inside their input (the export
  would copy itself); a Harbor task is written next to `--out` and moved into place when
  complete.
- The twin SDK's `selftest` fails when the program writes nothing, checks the tracks whenever
  a tracks file exists, and compares every output file of its two runs (a camera that moves
  between runs, an extra file), naming those that differ; videos are left to
  `r2s twin check`.
- The twin SDK's `check` (and so `selftest` and `write_tracks`) judges a tracks output by the
  rules of the format the Lab checks, with the same messages: it said OK to outputs the Lab
  rejects as invalid, scored 0 (a NaN or infinite focal length, a camera that is not a rigid
  transform, a field the format does not have, an object listed twice or named with a space
  or a dot, a frame other than metres and `+z`, integer positions). `selftest` no longer
  calls an output it does not check (a video, a 4D world) valid: it says `r2s twin check`
  validates it. A tracks `fps` that is not finite is invalid for the Lab too (`Infinity`
  passed; an integer too large for a float stopped the check with an `OverflowError`).
- `r2s capture from-video` refuses an impossible calibration before writing anything: a
  `--fov` outside (0, 180) degrees, intrinsics with a focal length at or below 0 or a last row
  other than `0 0 1`, extrinsics that are not a finite rigid transform (or an `--extrinsics`
  file holding anything but a 4x4 matrix), an object size at or below 0 or without a name.
- A moving camera's poses saved as an NPZ archive (`numpy.savez`) or not as numbers are a
  validation problem naming the file and how to save them, not a crash.
- A directory copy that fails (a full disk, `ulimit -f`) ends in one error line naming what was
  copied where, with the fix.
- A full disk, a quota or the file-size limit (`ulimit -f`) that stops ffmpeg encoding a video
  (`r2s bench build`, `r2s train tasks`, a report's overlay) or the copy of a solution
  (`r2s train reward`, `r2s eval`) is said as such, naming the file, with the fix, not as an
  empty reason after an internal class name or a list of Python tuples; ffmpeg that the
  file-size limit killed (SIGXFSZ, which leaves no message) is recognised. A failed encode no
  longer destroys the video that was there or leaves part of one: videos are encoded beside
  their name and moved there once complete.
- `r2s eval` and `r2s run` that cannot start (a backend not ready, an image not unpacked)
  remove the run directory they created on NFS too, so the same command runs once the cause
  is fixed: the directory is moved aside while its lock is held, then its lock is closed (NFS
  kept the open lock as `.nfsXXXX`, leaving an empty `state/` the same command then refused
  as someone else's), then it is removed, so another command given the same `--out`
  meanwhile loses nothing.
- What the Lab copies into a user's directories is the user's to change, also when it is
  installed read-only (a Nix store, a read-only share): an image's build context takes the
  SDK (`r2s env build` failed on any image), and the physics code of synthetic tasks and
  twins, `r2s export harbor`'s copy of the Lab and the task, program and twin trees it copies
  no longer keep a read-only mode that stopped `--overwrite` and `rm -rf`. A path the user
  cannot access ends in an error with a fix.
- `r2s env build` judges each failing line of a build: a download after pip (a file the
  Dockerfile fetches itself) is no longer reported as a pip failure, and the hint names the
  host that failed.
- PhysCodeGym's page says to build only the `rerun` variant for a program given by path (it has
  no session frames for `native` to score).
- A download that fails (no network, a proxy that refuses, a name that does not resolve) and an
  operating-system refusal (a read-only file system, too many open files) end in one error
  line with the fix, naming the proxy variables that are set (never their values), instead of
  a traceback.
- A program's executable bits are part of its content: a helper that lost `chmod +x` makes a
  changed program (its build is redone, not its old score reused; a frozen program changed so
  is not exported with its old reward), and Harbor's `solve.sh` recreates executable files as
  executable, so the verifier scores the program the Lab scored.
- `r2s train fit` checks every file of the dataset (but `export.json` and the card
  `README.md`) again after training, as before it: a recording replaced while the model trained
  (an editor, a sync) publishes no checkpoint, so a task holding it cannot escape the
  checkpoint's held-out check.
- A run directory whose first record could not be written (a full disk) is left usable: the
  advised retry of the same command runs, and a run killed while writing it resumes. Records
  written atomically leave no half-written temporary file when a write fails.
- A task's native scorer options (`[scoring.native.options.scorer]`) are checked against what
  the scorer takes: a name it does not take (a typo) or a value of the wrong type is reported
  by `r2s task validate` and refused before an evaluation starts, naming the task file and the
  options it takes, instead of a `TypeError` traceback. The task's other native options are
  checked too: `rerun` must be true or false, and a scorer may declare the names it reads
  (`NativeScorer.task_options`, `required_task_options`; every built-in scorer does), so a typo
  is reported instead of ignored and a missing name instead of a `KeyError` mid-run.
- A method label is at most 100 characters (it names a directory of every build and run): a
  longer one is refused on the command line instead of `File name too long` mid-experiment,
  and the `program` method finds the program of a task whose id is longer than a file name.
- The `program` method finds a directory of per-task programs named after the task id as
  nested directories too (`programs/synthetic/bouncing_ball/calibrated/`, as the twin tutorial
  shows), besides `synthetic__bouncing_ball__calibrated/` and one segment of the id; a task's
  own directory wins, and a segment directory that also holds other tasks' programs is not
  taken for another task.
- A method built with an open network (`network=open`) or reading hidden data (privileged) is
  marked in every table: the scorecard the command prints, every table of `report.md` and of
  the HTML pages, and `results.csv` (new last columns `privileged` and `network`).
- `r2s env unpack --from-tar` checks a saved image before extracting anything: an archive cut
  short (a copy or a download that stopped), with a layer its `manifest.json` names missing, or
  with damaged data is one error naming the archive and what is missing, with how to get a
  whole one, instead of a `KeyError` traceback.
- An unpacked image a bubblewrap sandbox is using is not replaced under it: `r2s env unpack
  --force` refuses while a program runs on it, and a program whose image was replaced or
  removed anyway (by hand) is an infrastructure error (no score, no reward), not its failure.
- A failed image build's network fix fits how it ran: with proxy variables set in the shell it
  says to pass them (`--build-arg HTTPS_PROXY`, by name, never their values) and never repeats
  `--network host` when the build already had it; `--build-arg NAME` without a value takes the
  value from the shell.
- `r2s env export --out` naming a directory is refused before Docker is asked anything, with
  the file name to give; a `docker save` that fails while the image is there says to check the
  destination, and only a missing image is told to be built or pulled. Likewise 4DCodeBench's
  world unpacking says to install `h5py` only when it is missing, and otherwise that the
  download may be cut (running the command again downloads it again).
- No process the Lab starts reads its standard input: in a script piped to a shell
  (`... | bash`), the ffmpeg that decodes videos for a report no longer reads the commands after
  `r2s eval`.
- `r2s export harbor --solution` and the `harbor` method write a `solve.sh` that quotes every
  file name (an apostrophe, a space or `$` in a name broke the script) and is rooted at the
  method's workspace without rewriting the embedded files.
- `r2s report` writes the task pages afresh: a task whose results are no longer counted (an
  earlier configuration's) keeps no page with its old scores from an earlier build.
- `r2s doctor --gpu` finds out through Docker whether Docker runs GPU containers (`nvidia-smi`
  with `--gpus all` in a built Lab image), instead of concluding it from this host's driver;
  `bwrap` and `local`, which use this host's driver, are still judged by it.
- `r2s doctor` and `r2s demo` check the ffmpeg and ffprobe the Lab runs: `R2S_FFMPEG` and
  `R2S_FFPROBE` when they are set (the doctor names the variable and checks that ffmpeg for
  libx264), else the ones on PATH. When the variable names another ffmpeg, the one on PATH,
  which programs on the `local` backend run, is checked apart; `r2s demo` on the local backend
  asks for that one, instead of refusing to render (or rendering without one) for lack of
  ffmpeg on PATH.
- `r2s task new` writes any title as valid TOML, checks the task id before writing, and leaves
  nothing behind when it fails (a wrong `--video`), so the corrected command works. An
  experiment's missing task path says that relative paths start from the experiment file's
  directory; `~` is expanded.
- The task-writing toolkit (`TaskBuilder`, `r2s task new`, validation), the single-file twin SDK
  `r2s_twin.py` and `r2s twin check`.
- Harbor export and the `harbor` method. Tasks whose solution is not a `run.py` program
  (4DCodeBench's `build.sh`, PhysCodeGym's Blender scenes, native protocols) are refused with
  an explanation. The verifier writes no reward when it cannot score (an error Harbor reports),
  and its 4DCodeBench checker is fetched at its pinned commit with retries. A solution that
  cannot be copied (e.g. it holds a named pipe) scores 0 instead of failing the verifier.
- Experiment files: method options that name files (`path`, `checkpoint`) start from the
  file's directory, as task paths do (`demo-iris.toml`'s hand-written twins were not found
  from the repository root).
- Agent skills (`.agents/skills/`) for adding a benchmark, adding a method and writing tasks;
  the source distribution ships them with `AGENTS.md`, which links them (its own link test
  failed in the unpacked sdist).
- The checks every pull request passes give the same result on any machine set up as
  CONTRIBUTING.md says: `pytest` runs the test that needs 4DCodeBench's B_30 only when it is
  downloaded (it no longer downloads it, failing without the `fourdcodebench` extra or a
  network), tests never see a model key or token of the shell they run in
  (`ANTHROPIC_AUTH_TOKEN`, `CLAUDE_CODE_OAUTH_TOKEN`, ...), and `pyright` passes without the
  optional extras (their imports are marked, as `cv2`'s were).
- A documentation site (a test checks every relative link in it), with an Evaluation guide
  (choosing tasks and variants, the same conditions for every method, repeats and intervals,
  reading and reporting a score) and, in the Training guide, how the rewards map onto common
  training methods; the logo of the Real2SimHub family; and the icons (each institution by its
  own mark), teaser and social preview drawn by scripts in `scripts/` (`make_icons.py`,
  `make_teaser.py`, `make_social_preview.py`).
- The documentation says what the commands do: only the demo chooses its backend (the README
  and the quickstart promised a fallback to bubblewrap for every command; the rest takes
  `--backend bwrap` without Docker); the training guide's worked example trains every
  checkpoint it evaluates and writes under `example/`, apart from the README's and the guide's
  other commands; the pages read after the quickstart build its tasks again with `--overwrite`
  instead of stopping on them; the specification's general rules name the records that keep
  the paths a command was given (`experiment.json`, a build's method options, a checkpoint's
  dataset); 4DCodeBench's re-run variant is `lab`; the live agent tests say they need the `dev`
  extra. Tests check the pages' commands in the order a reader runs them. The fix of
  a task built twice names `--overwrite` alone, not the Python API's `overwrite=True`.
- Installed from the repository (`pip install -e .` in a clone, or as a git submodule); there is
  no PyPI package. Runtime images are published on the GitHub Container Registry.
- `r2s doctor --gpu` says each check on one line: with several GPUs, how many, each model once
  with its count and the driver (`2 GPUs: NVIDIA A100-SXM4-80GB x2, driver 580.95.05`), where
  the second GPU's line stood alone under the check; any detail or fix a program printed on
  several lines is joined on its line, in the text and in `--json`.
- A seed no twin can be given (from 2**32, what `R2S_SEED` holds) is refused when it is given:
  `--seed`/`--seeds` on the command line, `seeds` in an experiment file, `seed` of the training
  tasks' builder. An evaluation stopped on it after its builds, and `r2s train tasks` on a
  "File name too long".
- A symbolic link that appears in a task while `r2s eval` runs (it never reaches a method)
  stops the run with the fix: replace it with what it points to or remove it, and run the same
  command again; it said only `error: symbolic link in ...`.
- `r2s train reward` without a reward (`infra_error`) says why in one `error:` line, without
  the error's class, and the fix: the machine's failure to repair, then score the same solution
  again (a trainer drops the sample). `InfrastructureError` takes an optional `fix`.
- A command whose backend cannot run here names another that can only after the fix's own
  backends are left out unasked (a Docker that hangs added 40 s to bwrap's error), and names
  bwrap as `r2s doctor` does: with how to unpack its runtime when it is not unpacked, not as
  "ready here".
- A method option that takes text given a value that reads as JSON (`--method program:path=1`,
  a directory named `1`) is refused with how to give it as text
  (`--method 'program:path="1"'`).
- The twin SDK's `selftest` says what it checked: a tracks output with a `render.mp4` is "valid
  tracks", the video left to `r2s twin check` (it said "valid" of a video it cannot read); a
  video's optional `meta.json` or a 4D world's `camera.json` no longer makes an output tracks
  (it failed on a missing `tracks.npz`). A camera number beyond the float range
  (`10**400`) is a problem of `camera.json` in the SDK's checks and the Lab's, not an
  `OverflowError`.
- Video2World's scorer puts back the files of a task's `visible/` its data release needs only
  when they are files: a symbolic link there is refused, never followed into the evaluator.
- On Python 3.13, an output path through a loop of symbolic links is reported as such, with
  its fix, not as a path that "already exists" (3.13 resolves such a path without a word). An
  exported Harbor task's `solve.sh` calls `chmod +x ./file`, which BSD `chmod` accepts too.
  The tests run on macOS: what bwrap, a boot id or file names that are not UTF-8 need is
  skipped there, and a test under a file-size limit runs its command without coverage.
