"""What an experiment counts: every result read back, every metric kept, nothing inherited from
outputs that are gone, nothing scored that a method could not deliver."""

from __future__ import annotations

import json
import shutil
import typing
from pathlib import Path
from typing import Any, ClassVar

import pytest

from real2simlab.orchestrate import Experiment, parse_experiment
from real2simlab.scoring import TaskResult
from real2simlab.scoring.aggregate import aggregate, benchmark_scores
from real2simlab.scoring.model import Headline, Results
from real2simlab.task import TaskConfig, load_task
from real2simlab.task.io import read_task_config, write_task_config
from tests.test_metrics_scoring import result


def _models() -> list[type]:
    import importlib
    import pkgutil

    import real2simlab
    from real2simlab._base import StrictModel

    for module in pkgutil.walk_packages(real2simlab.__path__, "real2simlab."):
        if ".sdk" in module.name or module.name.endswith(("__main__", "bootstrap")):
            continue
        try:
            importlib.import_module(module.name)
        except ImportError:  # an optional dependency
            continue

    def below(cls: type) -> list[type]:
        return [c for s in cls.__subclasses__() for c in (s, *below(s))]

    return below(StrictModel)


def test_every_nullable_field_can_be_left_out() -> None:
    """Records are written without their ``None`` fields, so a field that may be ``None`` must
    also be optional, or a record could not be read back."""
    required = [
        f"{model.__module__}.{model.__name__}.{name}"
        for model in _models()
        for name, info in model.model_fields.items()  # type: ignore[attr-defined]
        if info.is_required() and type(None) in typing.get_args(info.annotation)
    ]
    assert required == []


def test_a_result_without_a_score_is_read_back_and_aggregated() -> None:
    from real2simlab.benchmarks.fourdcodebench.scorer import FourDCodeBenchScorer

    # A delivery with nothing to grade scores 0; a report without a score has no value.
    nothing_to_grade = FourDCodeBenchScorer().headline({"status": "no_world", "reward": None})
    assert nothing_to_grade.value == 0.0
    headline = FourDCodeBenchScorer().headline({"status": "scorer_failed", "reward": None})
    assert headline.value is None
    unscored = result("fourdcodebench/b_30", headline, native={"status": "scorer_failed"})
    scored = result(
        "fourdcodebench/b_31",
        FourDCodeBenchScorer().headline({"status": "scored", "reward": {"dynamic_iou": 0.5}}),
        native={"status": "scored"},
    )
    for item in (unscored, scored):
        assert TaskResult.model_validate_json(item.dumps()) == item
    results = aggregate("e", [unscored, scored], resamples=50)
    again = Results.model_validate_json(results.dumps())
    (card,) = again.methods[0].benchmarks
    assert (card.scored, card.results, card.value.mean) == (1, 2, 0.5)
    nothing = aggregate("e", [unscored], resamples=50)
    assert Results.model_validate_json(nothing.dumps()).methods[0].benchmarks[0].value.mean is None


def test_scorecard_keeps_every_headline_metric_of_a_benchmark() -> None:
    """Custom tasks may share a benchmark name and differ in headline metric: each metric gets
    its own entry, whatever order the results come in."""

    def headline(metric: str, value: float) -> Headline:
        return Headline(metric=metric, value=value, range=(0.0, 1.0), source="lab")

    items = [
        result("custom/a", headline("reproduction_skill", 0.8)),
        result("custom/b", headline("dynamic_iou", 0.4)),
        result("custom/c", headline("reproduction_skill", 0.6)),
    ]
    for ordered in (items, items[::-1]):
        scores = benchmark_scores(ordered, resamples=50)
        found = {(s.benchmark, s.metric): (s.value.mean, s.results) for s in scores}
        assert found == {
            ("custom", "dynamic_iou"): (0.4, 1),
            ("custom", "reproduction_skill"): (pytest.approx(0.7), 2),
        }


def test_rerun_agreement_counts_only_what_was_compared() -> None:
    headline = Headline(metric="reproduction_skill", value=1.0, range=(0.0, 1.0), source="lab")
    items = [
        result("synthetic/a", headline, reproducible=True),
        result("synthetic/b", headline, reproducible=None),  # its first run failed: no re-run
    ]
    (card,) = benchmark_scores(items, resamples=50)
    assert (card.rerun_compared, card.rerun_agreed, card.results) == (1, 1, 2)


def test_each_benchmark_counts_its_own_deliveries(capsys: pytest.CaptureFixture[str]) -> None:
    """A method that delivers on one benchmark and not on another: each scorecard entry (and
    the summary's `delivered` column) says so, not the method's rate over all benchmarks."""
    from real2simlab.cli.experiments import _print_summary

    headline = Headline(metric="reproduction_skill", value=0.5, range=(0.0, 1.0), source="lab")
    missing = Headline(metric="score", value=None, range=(0.0, 1.0), source="lab")
    items = [
        result("one/a", headline),
        result("one/b", headline),
        result("two/a", missing).model_copy(update={"delivered": False}),
        result("two/b", missing).model_copy(update={"delivered": False}),
    ]
    results = aggregate("e", items, resamples=50)
    cards = {card.benchmark: card for card in results.methods[0].benchmarks}
    assert (cards["one"].delivered, cards["two"].delivered) == (2, 0)
    _print_summary(results, Path("out"))
    rows = {line.split("|")[2].strip(): line for line in capsys.readouterr().out.splitlines()
            if line.startswith("| m ")}  # fmt: skip
    assert rows["one"].rstrip(" |").endswith("100%") and rows["two"].rstrip(" |").endswith("0%")


def test_task_ids_never_share_a_directory() -> None:
    from real2simlab.orchestrate.layout import slug

    assert slug("synthetic/bouncing_ball/calibrated") == "synthetic__bouncing_ball__calibrated"
    assert slug("synthetic/ball__calibrated") != slug("synthetic__ball/calibrated")
    ids = ["a/b_/c", "a/b/c", "a_/b", "a___b", "a__/b", "a/b__c", "a__b/c", "a_/b_", "a__b"]
    assert all(_valid_id(i) for i in ids)
    assert len({slug(i) for i in ids}) == len(ids)


def _valid_id(value: str) -> bool:
    from real2simlab.task.model import TASK_ID

    return bool(TASK_ID.match(value))


def _experiment(tmp_path: Path, tasks: dict[str, Any], methods: list[dict[str, Any]]) -> Any:
    return parse_experiment(
        {"name": "integrity", "tasks": tasks, "methods": methods, "run": {"backend": "local"}}
    )


@pytest.mark.slow
def test_two_selections_of_one_benchmark_are_both_built(tmp_path: Path) -> None:
    def select(name: str) -> dict[str, Any]:
        return {"name": "synthetic", "select": [name], "tiers": ["calibrated"]}

    config = _experiment(
        tmp_path,
        {"benchmarks": [select("bouncing_ball"), select("pendulum")]},
        [{"name": "oracle"}],
    )
    tasks = Experiment(config, tmp_path / "run", log=lambda line: None).resolve_tasks()
    assert [t.id for t in tasks] == [
        "synthetic/bouncing_ball/calibrated",
        "synthetic/pendulum/calibrated",
    ]


@pytest.mark.slow
def test_resume_runs_again_when_a_runs_output_is_gone(bouncing_task: Path, tmp_path: Path) -> None:
    config = _experiment(tmp_path, {"paths": [str(bouncing_task)]}, [{"name": "oracle"}])

    def run() -> tuple[Any, Experiment]:
        experiment = Experiment(config, tmp_path / "run", log=lambda line: None, report=False)
        return experiment.run(), experiment

    first, _ = run()
    assert first.methods[0].reproduction_skill.mean == pytest.approx(1.0)
    (output,) = (tmp_path / "run" / "runs").rglob("run/output")
    shutil.rmtree(output)
    again, experiment = run()
    assert experiment.progress.runs == 1 and experiment.progress.reused_runs == 0
    assert again.methods[0].reproduction_skill.mean == pytest.approx(1.0)
    _, experiment = run()  # intact now: reused
    assert experiment.progress.runs == 0 and experiment.progress.reused_runs == 1


@pytest.mark.slow
def test_resume_rebuilds_when_a_builds_session_output_is_gone(
    bouncing_task: Path, tmp_path: Path
) -> None:
    config = _experiment(
        tmp_path, {"paths": [str(bouncing_task)]}, [{"name": "fake_agent", "mode": "physical"}]
    )

    def run() -> tuple[Any, Experiment]:
        experiment = Experiment(config, tmp_path / "run", log=lambda line: None, report=False)
        return experiment.run(), experiment

    first, _ = run()
    record = json.loads(next((tmp_path / "run" / "builds").rglob("build_record.json")).read_text())
    session = (
        next((tmp_path / "run" / "builds").rglob("build_record.json")).parent
        / record["session_output"]
    )
    assert session.is_dir() and record["session_output_fingerprint"]
    shutil.rmtree(session)
    again, experiment = run()
    assert experiment.progress.builds == 1 and experiment.progress.reused_builds == 0
    rows = [(r["reproduction_skill"], r["session_consistent"]) for r in again.to_json()["rows"]]
    assert rows == [
        (r["reproduction_skill"], r["session_consistent"]) for r in first.to_json()["rows"]
    ]


class _NoWorld:
    """A native scorer whose method delivered a solution but no world."""

    name: ClassVar[str] = "fourdcodebench"

    def __init__(self, **options: Any) -> None:
        pass

    def problems(self) -> list[str]:
        return []

    def score(self, task: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "no_world", "types": [], "reward": None}

    def headline(self, report: Any) -> Headline:
        from real2simlab.benchmarks.fourdcodebench.scorer import FourDCodeBenchScorer

        return FourDCodeBenchScorer().headline(report)


@pytest.mark.slow
def test_a_delivery_with_nothing_to_grade_scores_0_and_does_not_stop_the_experiment(
    task_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from real2simlab.orchestrate import experiment as experiment_module
    from real2simlab.report import write_report

    data = read_task_config(task_copy / "task.toml").to_json()
    data["scoring"] = {
        "protocol": "native",
        "native": {"scorer": "fourdcodebench", "options": {"kind": "synthetic", "case": "B_30"}},
    }
    write_task_config(TaskConfig.model_validate(data), task_copy / "task.toml")
    monkeypatch.setattr(experiment_module, "get_scorer", lambda name, **options: _NoWorld())
    config = _experiment(
        tmp_path, {"paths": [str(task_copy)]}, [{"name": "fake_agent", "mode": "physical"}]
    )
    results = Experiment(config, tmp_path / "run", log=lambda line: None, report=False).run()
    (card,) = results.methods[0].benchmarks
    assert (card.scored, card.results, card.value.mean) == (1, 1, 0.0)
    write_report(tmp_path / "run")
    assert "dynamic_iou" in (tmp_path / "run" / "report" / "report.md").read_text()


def test_baselines_are_not_applicable_to_deliveries_they_cannot_make(task_copy: Path) -> None:
    from real2simlab.methods.base import get_method

    task = load_task(task_copy)
    for name in ("static", "constant_velocity", "oracle"):
        assert get_method(name).applicable(task) is None, name
    data = read_task_config(task_copy / "task.toml").to_json()
    data["delivery"]["convention"] = "build_sh_world"
    write_task_config(TaskConfig.model_validate(data), task_copy / "task.toml")
    world = load_task(task_copy)
    for name in ("static", "constant_velocity"):
        reason = get_method(name).applicable(world)
        assert reason is not None and "build_sh_world" in reason, name
    assert get_method("fake_agent").applicable(world) is None


@pytest.mark.slow
def test_inapplicable_methods_are_reported_as_such_not_scored(
    task_copy: Path, tmp_path: Path
) -> None:
    shutil.rmtree(task_copy / "hidden" / "generator", ignore_errors=True)
    config = _experiment(
        tmp_path, {"paths": [str(task_copy)]}, [{"name": "oracle"}, {"name": "static"}]
    )
    experiment = Experiment(config, tmp_path / "run", log=lambda line: None, report=False)
    results = experiment.run()
    assert [m.method for m in results.methods] == ["static"]
    assert any(
        "oracle" in line and "not applicable" in line for line in experiment.progress.skipped
    )


@pytest.mark.slow
@pytest.mark.parametrize("redistributable", [True, False])
def test_the_report_keeps_restricted_recordings_out_of_its_directory(
    task_copy: Path, tmp_path: Path, redistributable: bool
) -> None:
    from real2simlab.report import write_report

    data = read_task_config(task_copy / "task.toml").to_json()
    data["redistributable"] = redistributable
    write_task_config(TaskConfig.model_validate(data), task_copy / "task.toml")
    config = _experiment(tmp_path, {"paths": [str(task_copy)]}, [{"name": "oracle"}])
    Experiment(config, tmp_path / "run", log=lambda line: None).run()
    write_report(tmp_path / "run")
    report = tmp_path / "run" / "report"
    (page,) = (report / "tasks").glob("*.html")
    in_report = sorted(p.name for p in report.rglob("*.mp4"))
    kept_apart = sorted(p.name for p in (tmp_path / "run" / "local-media").rglob("*.mp4"))
    if redistributable:
        assert "observed.mp4" in in_report and not kept_apart
    else:
        assert not in_report and "observed.mp4" in kept_apart
        assert 'src="../../local-media/' in page.read_text()
        assert "may not be redistributed" in page.read_text()
