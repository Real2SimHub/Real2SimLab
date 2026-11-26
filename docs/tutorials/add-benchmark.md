# Add a benchmark

An adapter turns a benchmark's data into task directories. It is a plugin: it lives in its own
module (in Real2SimLab or in your package) and the core does not change.

## 1. Check the license first

Read the license of the benchmark's code **and** data before writing anything. Then:

- record it in every task (`license`, `source`, `redistributable`);
- never commit or package third-party benchmark data, or code that may not be redistributed:
  download it at a pinned revision into the user's cache when tasks are built
  (`real2simlab.vendor.VendoredRepo` for code,
  `huggingface_hub.hf_hub_download(..., revision=...)` for data);
- when no license is declared, fetch only from the authors' own public location, mark the
  tasks `redistributable = false` and say "none declared" wherever the license is shown;
- when the license forbids the use, or the terms need an agreement or registration that a
  script cannot accept for the user, do not fetch it: leave it out, or let users point the
  adapter at files they obtained themselves;
- write the adapter's documentation page with the source, license and citation.

## 2. Decide the protocol

Following the Lab's design principles, reproduce the benchmark's **own** protocol by default:

| The benchmark | Use |
|---|---|
| has an official scorer | `native`: a scorer plugin runs it on the delivery or its frozen re-run |
| is a dataset of recordings with a measurable truth | `reproduction`: the reference in `hidden/reference/`, other recordings of the event in `hidden/repeats/` |

The Lab's own addition, scoring the frozen re-run instead of the agent's session output, comes
as a task variant (chosen with the build option `variants`): `rerun` in the
[Physics-IQ](../benchmarks/physics-iq.md) and [PhysCodeGym](../benchmarks/physcodegym.md)
adapters, `lab` in the [4DCodeBench](../benchmarks/fourdcodebench.md) adapter, which scores the
re-run with the Lab's metrics.

## 3. Write the import steps as functions

Keep every step that turns source data into captures and references a plain function
(`download_case`, `case_capture`, `build_case`, ...): other projects reuse them to assemble
their own tasks ([Build on Real2SimLab](../guides/build-on.md)). Build each task with `TaskBuilder` (see
[Write tasks](write-tasks.md)):

```python
class MyBenchmark(BenchmarkAdapter):
    name = "mybench"
    summary = "..."
    license = "CC-BY-4.0"
    source = "https://..."
    protocols = ("native",)

    def task_names(self) -> list[str]: ...

    def build(self, out_dir, *, select=None, overwrite=False, **options) -> list[Path]:
        return [build_case(case, out_dir, overwrite=overwrite) for case in select or DEFAULT]
```

```toml
# pyproject.toml of your package
[project.entry-points."real2simlab.benchmarks"]
mybench = "my_package.mybench:MyBenchmark"
```

`r2s bench list` shows it; `r2s bench build mybench --out tasks/` builds it.

## 4. Delivery and scoring

- Most tasks use the `r2s` convention (`solution/run.py`, `tracks` output). A benchmark whose
  methods hand in something else keeps its own convention: `build_sh_world` (4DCodeBench) and
  `blender_scene` (PhysCodeGym) exist; a new one is an entry in
  `real2simlab/twin/conventions.py`.
- A new output format, metric or official scorer is a plugin too: `real2simlab.formats`,
  `real2simlab.metrics`, `real2simlab.scorers`.

An official scorer subclasses `real2simlab.scoring.NativeScorer`:

```python
class MyScorer(NativeScorer):
    name = "mybench"
    summary = "MyBench's official metrics"

    def score(self, task, *, delivery_dir, output_dir, work_dir, options):
        ...                      # run the pinned official code; return its report unchanged

    def headline(self, report):  # the benchmark's headline metric, for the scorecard
        return Headline(metric="score", value=report.get("score"), range=(0.0, 1.0),
                        references={"reference solution": ...}, source="official",
                        scorer=self.name)
```

Give `aggregate(reports)` as well when the benchmark does not report the mean over tasks (see
the Physics-IQ scorer).

Make task building **deterministic**: the same inputs must give byte-identical task
directories (no timestamps, no temporary paths, fixed random seeds). Experiments identify tasks
by their content hash; a task that changes on every build would re-run every method, agents
included, each time it is rebuilt.

## 5. Test and document

- A test that builds a task (from a small cached case, skipped when the data are absent) and
  checks `validate_task(...)` has no errors or warnings.
- A test of each scorer or extraction step on data with a known answer, and of its headline.
- **Fidelity**: run the benchmark's reference solutions (or ground-truth programs) through the
  adapter and the scorer, and compare with the official numbers. Record the outcome, including
  any discrepancy in the benchmark's own release.
- `docs/benchmarks/<name>.md`: status, protocol, headline metric and references, source,
  license, citation, the fidelity check, and how its scores compare with the official ones, or
  why they cannot be compared yet.

The repository's `.agents/skills/add-benchmark/SKILL.md` is the same procedure written for a
coding agent.
