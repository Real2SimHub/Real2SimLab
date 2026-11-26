"""The whole pipeline: tasks → methods (baselines, scripted agents) → frozen re-runs → scores → report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from real2simlab.orchestrate import Experiment, parse_experiment, run_experiment
from real2simlab.scoring import TaskResult

METHODS = [
    {"name": "oracle"},
    {"name": "static"},
    {"name": "constant_velocity"},
    {"name": "fake_agent", "label": "physical", "mode": "physical"},
    {"name": "fake_agent", "label": "broken", "mode": "broken"},
    {"name": "fake_agent", "label": "invalid", "mode": "invalid"},
    {"name": "fake_agent", "label": "nothing", "mode": "nothing"},
]


def config(
    tasks: Path, backend: str = "local", methods: list[dict[str, Any]] | None = None, **run: Any
):
    return parse_experiment(
        {
            "name": "e2e",
            "tasks": {
                "paths": [str(tasks)],
                "include": ["synthetic/bouncing_ball/*", "synthetic/pendulum/calibrated"],
            },
            "methods": methods or METHODS,
            "run": {"backend": backend, "concurrency": 4, **run},
        }
    )


def by_method(results: Any) -> dict[str, Any]:
    return {m.method: m for m in results.methods}


def check_expected_scores(results: Any) -> None:
    methods = by_method(results)
    assert methods["oracle"].reproduction_skill.mean == pytest.approx(1.0)
    assert methods["static"].reproduction_skill.mean == 0.0
    assert 0.5 < methods["physical"].reproduction_skill.mean < 1.0
    for name in ("broken", "invalid", "nothing"):
        assert methods[name].reproduction_skill.mean == 0.0, name
    assert methods["nothing"].delivered_rate == 0.0 and methods["broken"].delivered_rate == 1.0
    assert methods["oracle"].privileged and not methods["physical"].privileged
    # The scorecard: one entry per benchmark, its headline over the method's tasks.
    (card,) = methods["physical"].benchmarks
    assert card.benchmark == "synthetic" and card.metric == "reproduction_skill"
    assert card.value.mean == pytest.approx(methods["physical"].reproduction_skill.mean)
    assert card.scored == card.results == 3 and card.references["static"] == 0.0
    (broken,) = methods["broken"].benchmarks
    assert broken.value.mean == 0.0


def load_results(root: Path) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    found = {}
    for path in sorted((root / "results").rglob("seed-*.json")):
        result = TaskResult.model_validate_json(path.read_text())
        data = result.to_json()
        data.pop("created_at")
        # Independent builds differ only in origin.created_at inside twin.json.
        data.pop("twin_fingerprint", None)
        found[(result.task_id, result.method, result.trial, result.seed)] = data
    return found


@pytest.mark.slow
def test_pipeline_on_local_backend(synthetic_tasks: Path, tmp_path: Path) -> None:
    results = run_experiment(config(synthetic_tasks), tmp_path / "run", log=lambda line: None)
    check_expected_scores(results)
    run = tmp_path / "run"
    index = (run / "report" / "index.html").read_text()
    assert "Scorecard" in index and "physical" in index and "reproduction_skill" in index
    task_page = (run / "report" / "tasks" / "synthetic__bouncing_ball__calibrated.html").read_text()
    assert (
        "agent trajectory (redacted)" in task_page and "<svg" in task_page and "<video" in task_page
    )
    assert (run / "report" / "results.csv").is_file() and (run / "report" / "report.md").is_file()
    broken = json.loads(next((run / "results").rglob("broken/trial-0/seed-0.json")).read_text())
    assert broken["delivered"] and broken["reproduction"]["status"] == "run_failed"
    nothing = json.loads(next((run / "results").rglob("nothing/trial-0/seed-0.json")).read_text())
    assert not nothing["delivered"] and "no run.py" in nothing["delivery_failure"]
    record = json.loads(
        next((run / "builds").rglob("physical/trial-0/work/agent/agent_record.json")).read_text()
    )
    assert (
        record["instruction_template"] == "r2s-instruction/1.0"
        and record["usage"]["input_tokens"] == 1000
    )
    for path in run.rglob("*.json"):
        assert str(Path.home()) not in path.read_text() or path.name in {"experiment.json"}, path


@pytest.mark.slow
def test_resume_after_interruption_gives_the_same_results(
    synthetic_tasks: Path, tmp_path: Path
) -> None:
    methods = METHODS[:5]
    reference = tmp_path / "reference"
    run_experiment(
        config(synthetic_tasks, methods=methods), reference, log=lambda line: None, report=False
    )

    interrupted = tmp_path / "interrupted"
    experiment = Experiment(
        config(synthetic_tasks, methods=methods, concurrency=1, agent_concurrency=1),
        interrupted,
        log=lambda line: None,
        report=False,
    )
    calls = {"n": 0}
    original = experiment._evaluate

    def flaky(built: Any, backend: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 4:
            raise KeyboardInterrupt("simulated interruption")
        return original(built, backend)

    experiment._evaluate = flaky  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        experiment.run()
    partial = load_results(interrupted)
    assert 0 < len(partial) < len(load_results(reference))

    resumed = Experiment(
        config(synthetic_tasks, methods=methods), interrupted, log=lambda line: None, report=False
    )
    resumed.run()
    assert resumed.progress.reused_builds > 0
    assert load_results(interrupted) == load_results(reference)

    again = Experiment(
        config(synthetic_tasks, methods=methods), interrupted, log=lambda line: None, report=False
    )
    again.run()
    assert again.progress.builds == 0 and again.progress.runs == 0


@pytest.mark.slow
def test_budget_skips_agent_builds(synthetic_tasks: Path, tmp_path: Path) -> None:
    data = config(synthetic_tasks, methods=[{"name": "fake_agent", "mode": "physical"}]).to_json()
    data["budget"] = {"max_agent_runs": 1}
    from real2simlab.orchestrate import ExperimentConfig

    experiment = Experiment(
        ExperimentConfig.model_validate(data), tmp_path / "run", log=lambda line: None, report=False
    )
    results = experiment.run()
    assert len(experiment.progress.skipped) == 2
    assert json.loads((tmp_path / "run" / "results.json").read_text())["skipped"]
    assert results.methods[0].results == 1


@pytest.mark.slow
def test_changed_configuration_never_inherits_earlier_results(
    synthetic_tasks: Path, tmp_path: Path
) -> None:
    """Resuming with a changed method whose new build the budget skips reports the unit as
    not run, instead of the earlier configuration's score; reverting resumes as before."""
    from real2simlab.orchestrate import ExperimentConfig
    from real2simlab.report import write_report

    def experiment(mode: str, budget: int | None) -> Experiment:
        data = parse_experiment(
            {
                "name": "e2e",
                "tasks": {"paths": [str(synthetic_tasks)], "include": ["synthetic/bouncing_ball/calibrated"]},
                "methods": [{"name": "fake_agent", "label": "agent", "mode": mode}],
                "run": {"backend": "local"},
            }
        ).to_json()  # fmt: skip
        data["budget"] = {"max_agent_runs": budget}
        config = ExperimentConfig.model_validate(data)
        return Experiment(config, tmp_path / "run", log=lambda line: None, report=False)

    first = experiment("physical", 1).run()
    assert first.methods[0].reproduction_skill.mean > 0.5

    changed = experiment("broken", 1)  # the new build would be a second agent run: skipped
    results = changed.run()
    assert results.methods == () and results.rows == ()
    skipped = json.loads((tmp_path / "run" / "results.json").read_text())["skipped"]
    assert any("earlier configuration" in line for line in skipped)
    write_report(tmp_path / "run")
    assert "0.9" not in (tmp_path / "run" / "report" / "report.md").read_text()

    reverted = experiment("physical", 1)  # the earlier build is still there: reused, rescored
    again = reverted.run()
    assert reverted.progress.reused_builds == 1 and reverted.progress.builds == 0
    assert again.methods[0].reproduction_skill.mean == first.methods[0].reproduction_skill.mean

    rebuilt = experiment("broken", None).run()  # with budget, the new configuration runs
    assert rebuilt.methods[0].reproduction_skill.mean == 0.0


@pytest.mark.docker
@pytest.mark.slow
def test_pipeline_on_docker_backend(
    synthetic_tasks: Path, tmp_path: Path, docker_backend: Any
) -> None:
    results = run_experiment(
        config(synthetic_tasks, backend="docker", check_reproducibility=True),
        tmp_path / "run",
        backend=docker_backend,
        log=lambda line: None,
    )
    check_expected_scores(results)
    result = json.loads(
        next((tmp_path / "run" / "results").rglob("physical/trial-0/seed-0.json")).read_text()
    )
    assert result["reproduction"]["reproducible"] is True
    record = json.loads(
        next(
            (tmp_path / "run" / "runs").rglob("physical/trial-0/seed-0/run/run_record.json")
        ).read_text()
    )
    assert record["backend"]["backend"] == "docker" and record["image"]["digest"]


def test_benchmark_tasks_are_rebuilt_when_the_adapter_changes(
    tmp_path: Path, local, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cached tasks follow the adapter's code, not just the Lab's version string."""
    from real2simlab.benchmarks.synthetic import SyntheticBenchmark

    config = parse_experiment(
        {
            "name": "rebuild",
            "tasks": {
                "benchmarks": [
                    {"name": "synthetic", "select": ["bouncing_ball"], "tiers": ["calibrated"]}
                ]
            },
            "methods": [{"name": "oracle"}],
            "run": {"backend": "local"},
        }
    )
    logs: list[str] = []

    def run() -> list[str]:
        logs.clear()
        run_experiment(config, tmp_path / "run", backend=local, log=logs.append, report=False)
        return logs

    assert any("building benchmark synthetic" in line for line in run())
    assert not any("building benchmark" in line for line in run())
    monkeypatch.setattr(SyntheticBenchmark, "source_digest", lambda self: "changed")
    lines = run()
    assert any("building benchmark synthetic" in line for line in lines)
    # Same task contents, so the builds (and any agent runs) are reused.
    done = next(line for line in lines if "done:" in line)
    assert "1 builds (1 reused)" in done or "(1 reused)" in done.split("builds")[1][:15], done


def test_adapter_source_files_cover_what_shapes_tasks() -> None:
    from real2simlab.benchmarks.fourdcodebench import FourDCodeBenchBenchmark
    from real2simlab.benchmarks.synthetic import SyntheticBenchmark

    files = {path.as_posix() for path in SyntheticBenchmark().source_files()}
    for needed in ("real2simlab/media.py", "real2simlab/vendor.py",
                   "real2simlab/twin/sdk/r2s_twin.py", "real2simlab/envs/registry.py",
                   "real2simlab/benchmarks/synthetic/physics/render.py"):  # fmt: skip
        assert any(name.endswith(needed) for name in files), needed
    assert not any("/real2simlab/cli/" in name or "__pycache__" in name for name in files)
    assert SyntheticBenchmark().source_digest() == FourDCodeBenchBenchmark().source_digest()
