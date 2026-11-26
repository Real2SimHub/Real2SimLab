"""The example task and twins shipped in examples/ validate and score as documented."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from real2simlab.orchestrate import parse_experiment, run_experiment
from real2simlab.task import find_tasks, validate_task

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_example_tasks_validate() -> None:
    roots = find_tasks(EXAMPLES / "tasks")
    assert {root.name for root in roots} == {"puck_slide"}
    for root in roots:
        report = validate_task(root)
        assert report.ok and not report.warnings, report.issues


@pytest.mark.slow
def test_example_tasks_score(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    shutil.copytree(EXAMPLES / "tasks", tasks)
    config = parse_experiment(
        {
            "name": "examples",
            "tasks": {"paths": [str(tasks)]},
            "methods": [
                {"name": "program", "label": "hand-written", "path": str(EXAMPLES / "twins")},
                {"name": "static"},
                {"name": "constant_velocity"},
            ],
            "run": {"backend": "local", "check_reproducibility": True},
        }
    )
    results = run_experiment(config, tmp_path / "run", log=lambda line: None, report=False)
    methods = {m.method: m for m in results.methods}
    puck = next(r for r in results.rows if r["method"] == "hand-written")
    # The hand-written twin guesses the floor's friction (0.2; the recording's is 0.237).
    assert 0.7 < puck["reproduction_skill"] < 1.0 and puck["reproducible"] is True
    assert methods["static"].reproduction_skill.mean == 0.0
    assert methods["constant_velocity"].reproduction_skill.mean < puck["reproduction_skill"]
    (card,) = methods["hand-written"].benchmarks
    assert card.benchmark == "examples" and "another recording" in card.references
    assert card.references["another recording"] > puck["reproduction_skill"]


EXPERIMENTS = sorted((EXAMPLES / "experiments").glob("*.toml"))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda path: path.stem)
def test_example_experiments_name_what_exists(path: Path) -> None:
    """Every experiment file in examples/ names registered benchmarks with options their
    builders take, synthetic phenomena and tiers that exist, and methods whose options are
    valid; checked offline, without building anything."""
    import inspect

    from real2simlab.benchmarks.base import get_benchmark
    from real2simlab.benchmarks.synthetic import PHENOMENA, TIERS
    from real2simlab.methods.base import get_method
    from real2simlab.orchestrate.config import load_experiment

    config, _ = load_experiment(path)
    assert config.tasks.benchmarks and config.methods
    for entry in config.tasks.benchmarks:
        adapter = get_benchmark(entry.name)
        taken = inspect.signature(adapter.build).parameters
        assert set(entry.options) <= {name for name in taken if name != "options"}, entry
        if entry.name == "synthetic":
            assert set(entry.select or ()) <= set(PHENOMENA)
            assert set(entry.tiers or ()) <= set(TIERS)
    for method in config.methods:
        get_method(method.name, **method.options)
