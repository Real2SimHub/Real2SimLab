---
name: add-benchmark
description: Add a benchmark adapter to Real2SimLab (a plugin that downloads a benchmark at a pinned revision and builds task directories, reproducing its own protocol). Use when asked to integrate, adapt or import an external benchmark or dataset.
---

# Add a benchmark adapter

An adapter is a plugin under `real2simlab/benchmarks/<name>/` (or in another package) that turns
a benchmark's data into task directories. The core does not change. Tutorial:
`docs/tutorials/add-benchmark.md`. Read an existing adapter of the same shape first:
`physicsiq` (official scorer with its own aggregate and reference points, frozen re-run
variant), `video2world` (official evaluator that executes a data package), `physcodegym`
(official scorer, Blender delivery), `fourdcodebench` (official scorer in its own image),
`iris` (real videos, extracted references, reproduction protocol), `synthetic` (generated).

## Procedure

1. **License first.** Find the license of the code and of the data (repository `LICENSE`,
   dataset card; for data assembled from other datasets, each source's own terms). Write it
   down with the source URL and a pinned revision (commit or dataset revision). Then:
   - **a license that allows use** (redistribution or not): download at build or score time
     into `real2simlab.vendor.cache_dir()`; never commit or package third-party data;
   - **no license declared**: the same, and only from the authors' own public location; mark
     tasks `redistributable = false`, write `none declared` as the license in the adapter, its
     documentation page and `r2s bench list`, and never put it in a published image;
   - **a license that forbids the use, or terms that need an agreement or registration a
     script cannot accept for the user**: do not fetch it; leave it out, or let users point the
     adapter at files they obtained themselves. Stop and report if nothing usable is left.
2. **Survey before coding.** What is observed, what varies between cases, what the methods
   deliver, how the benchmark scores. Check the published parameters against the data (in
   IRIS the stated camera height contradicted the videos); record surprises.
3. **Choose the protocol**, the benchmark's own by default:
   - an official scorer → `native`, through a scorer plugin (`real2simlab.scorers` entry
     point, a `NativeScorer`; see `real2simlab/benchmarks/physicsiq/scorer.py`). It returns the
     official report unchanged and implements `headline(report)`: the benchmark's headline
     metric, its range and direction, and the reference points the benchmark or the task
     defines (`static`, `another recording`, `reference solution`). If the benchmark does not
     report the mean over tasks, implement the classmethod `aggregate(reports)` too;
   - a dataset of recordings with a measurable truth → `reproduction`: the reference in
     `hidden/reference/`, other recordings of the same event in `hidden/repeats/`.
   If the benchmark scores what the agent showed in its session, add a `rerun` variant that
   scores the frozen, offline re-run instead (as Physics-IQ and PhysCodeGym do).
4. **Delivery convention**: `r2s` (`run.py`, `tracks`) unless the benchmark's methods hand in
   something else; then reuse or add an entry in `real2simlab/twin/conventions.py`
   (`build_sh_world`, `blender_scene` exist). A new runtime goes in `docker/<image>/` and
   `real2simlab/envs/__init__.py`.
5. **Write the import steps as plain functions** (`download_*`, `extract_*`, `task_config`,
   `visible_capture`, `build_tasks`): other projects reuse them. Build every task with
   `TaskBuilder` (see the `write-tasks` skill). Subclass `BenchmarkAdapter` with `name`,
   `summary`, `license`, `source`, `protocols`, `status` (`ready` or `partial`), `task_names()`
   and `build()`.
   **A training split** only when the benchmark releases one under terms that allow training:
   declare `splits = ("eval", "train")` and, when `build` gets `split="train"`, write tasks with
   `split = "train"` (`r2s bench build NAME --split train`). Never mark evaluation data as
   training data; if its terms forbid training on it, set the adapter's `training_terms` (as
   Physics-IQ does): its tasks are then never exported for training. If its clips come from
   another benchmark, say which, so users pass that benchmark's tasks to
   `r2s train export --against`.
6. **Register** in `pyproject.toml` under `[project.entry-points."real2simlab.benchmarks"]`,
   add an optional-dependency group if it needs packages, reinstall (`pip install -e .`), and
   check `r2s bench list`.
7. **Validate the references.** Extracted or converted references need an independent check
   (IRIS: gravity from the fall's curvature, never used in the calibration; repeats agree to
   centimetres). Test every extraction or conversion step on synthetic data with a known
   answer.
8. **Decide whether the references are good enough — before building tasks.** Survey every
   take, state the acceptance criteria (e.g. the independent check within a few percent,
   repeats agreeing to a known spread, measured initial states matching the nominal ones) and
   apply them. If they fail, deliver the extraction, the survey and the documented reasons, and
   build no tasks: wrong references score methods against measurement errors. One file can
   hold several attempts, or two takes can be the same recording: check for both.
   Keep source data read-only and write derived caches elsewhere; stream long videos instead of
   decoding them whole.
9. **Build twice and compare**: the task directories must be byte-identical (no timestamps,
   temporary paths or unseeded randomness): tasks are identified by their content hash.
   Then **build and evaluate**: `r2s bench build <name> --out tasks/`, `r2s task validate tasks/`,
   then `r2s eval` with the benchmark's reference solutions (as a `program` method) and, for
   the reproduction protocol, `static` and `constant_velocity`.
10. **Fidelity check**: run the benchmark's reference solutions or ground-truth programs through
   the adapter and the scorer, and compare with the official numbers (or the official pipeline
   on the same input). Record the result on the page, including any discrepancy in the
   benchmark's own release; report such discrepancies, do not hide them.
11. **Tests**: unit tests of each step; a task-building test skipped when the data are not in
   the cache; for a scorer, a test that scores the reference against itself and a test of its
   headline.
12. **Document** `docs/benchmarks/<name>.md` (status, protocol, headline and references,
    source, license, citation, how references were made, the fidelity check, assumptions, and
    how scores compare with the official ones, or why they cannot), add it to `mkdocs.yml` and
    `docs/benchmarks/index.md`, add a CHANGELOG line, and explain non-obvious choices in the
    pull request.

## Checks before you finish

```bash
ruff check real2simlab tests && ruff format --check real2simlab tests
pyright
pytest tests/test_<name>.py -m ""
mkdocs build --strict
```
