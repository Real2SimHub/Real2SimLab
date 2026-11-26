"""Training with the Lab: generated training tasks, datasets and rewards (the learned method:
test_overlap.py)."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from real2simlab.benchmarks.synthetic import SyntheticBenchmark
from real2simlab.benchmarks.synthetic.physics import PHENOMENA
from real2simlab.benchmarks.synthetic.sampling import MARGIN, SPACES, samples, values
from real2simlab.errors import SetupError
from real2simlab.orchestrate import Experiment, parse_experiment
from real2simlab.scoring.model import (
    Headline,
    Reproduction,
    ReproductionStatus,
    TaskResult,
    TaskSummary,
)
from real2simlab.task import Split, TaskConfig, load_task, validate_task
from real2simlab.task.io import read_task_config, write_task_config
from real2simlab.train import export_dataset, reward_of, score_solution


def build_training(out: Path, names: list[str], count: int, seed: int = 0) -> list[Path]:
    return SyntheticBenchmark().build(
        out, select=names, tiers=["calibrated"], split="train", count=count, seed=seed
    )


def test_training_samples_are_reproducible_and_never_the_evaluation_task() -> None:
    for name, module in PHENOMENA.items():
        first = samples(module, 0, 12)
        assert [(s.params, s.setup) for s in samples(module, 0, 4)] == [
            (s.params, s.setup) for s in first[:4]
        ]
        assert [(s.params, s.setup) for s in samples(module, 1, 4)] != [
            (s.params, s.setup) for s in first[:4]
        ]
        evaluation = values(module, module.TRUE_PARAMS, module.SETUP)
        assert set(evaluation) == set(SPACES[name].keys())
        for sample in first:
            drawn = values(module, sample.params, sample.setup)
            assert any(abs(drawn[k] - v) > MARGIN * abs(v) for k, v in evaluation.items())


def test_training_tasks_are_marked_and_their_truth_reproduces(tmp_path: Path, local) -> None:
    (root,) = build_training(tmp_path, ["bouncing_ball"], 1, seed=3)
    report = validate_task(root)
    assert report.ok and not report.warnings, report.issues
    task = load_task(root)
    assert task.config.split is Split.TRAIN
    assert task.id == "synthetic-train/bouncing_ball/calibrated/s3-00000"
    truth = json.loads((task.hidden_dir / "parameters.json").read_text())
    (sample,) = samples(PHENOMENA["bouncing_ball"], 3, 1)
    assert truth == {"phenomenon": "bouncing_ball", "params": sample.params, "setup": sample.setup}
    capture = task.capture()
    assert capture.scene is not None
    (ball,) = [o for o in capture.scene.objects if o.name == "ball"]
    assert ball.initial_pose is not None
    assert ball.initial_pose.value[2] == sample.setup["release_height"]["ball"]
    scored = score_solution(task, task.hidden_dir / "generator" / "program", backend=local)
    assert scored.reward == pytest.approx(1.0) and scored.result.delivered


def test_evaluation_tasks_keep_their_truth_program_and_gain_their_parameters(
    bouncing_task: Path,
) -> None:
    task = load_task(bouncing_task)
    assert task.config.split is Split.EVAL
    module = PHENOMENA["bouncing_ball"]
    truth = json.loads((task.hidden_dir / "parameters.json").read_text())
    assert truth == {"phenomenon": module.NAME, "params": module.TRUE_PARAMS, "setup": module.SETUP}
    model = json.loads((task.hidden_dir / "generator" / "program" / "model.json").read_text())
    assert "setup" not in model  # the evaluation task's own initial state


@pytest.mark.slow
def test_a_reward_is_the_evaluation_score_of_the_same_program(tmp_path: Path, local) -> None:
    first, second = build_training(tmp_path / "tasks", ["pendulum"], 2)
    program = tmp_path / "programs" / load_task(first).id.replace("/", "__")
    shutil.copytree(load_task(second).hidden_dir / "generator" / "program", program)
    scored = score_solution(first, program, backend=local)
    assert 0.0 < scored.reward < 1.0
    config = parse_experiment(
        {
            "name": "reward",
            "tasks": {"paths": [str(first)]},
            "methods": [{"name": "program", "path": str(tmp_path / "programs")}],
            "run": {"backend": "local"},
        }
    )
    results = Experiment(config, tmp_path / "run", log=lambda line: None, report=False).run()
    assert results.rows[0]["reproduction_skill"] == pytest.approx(scored.reward, abs=1e-12)
    empty = tmp_path / "empty"
    empty.mkdir()
    nothing = score_solution(first, empty, backend=local)
    assert nothing.reward == 0.0 and not nothing.result.delivered


def test_a_solution_directory_is_one_program_for_its_reward_and_its_evaluation(
    bouncing_task: Path, tmp_path: Path, local, capsys: pytest.CaptureFixture[str]
) -> None:
    """A directory whose ``run.py`` is its program is that program for a reward and for
    ``r2s eval --method program:path=`` alike, whatever its subdirectories hold (a helper's own
    ``run.py``): the two are equal, not 0 for "nothing delivered". One that also holds a
    program for the task, under the task's id or a segment of it, is refused before anything
    runs, not evaluated as the other program."""
    from real2simlab.cli.main import main

    solution = tmp_path / "solution"
    shutil.copytree(bouncing_task / "hidden" / "generator" / "program", solution)
    (solution / "tools").mkdir()
    (solution / "tools" / "run.py").write_text("print('a helper')\n")
    scored = score_solution(bouncing_task, solution, backend=local)
    assert scored.reward > 0.99
    config = parse_experiment(
        {
            "name": "same-program",
            "tasks": {"paths": [str(bouncing_task)]},
            "methods": [{"name": "program", "path": str(solution)}],
            "run": {"backend": "local"},
        }
    )
    results = Experiment(config, tmp_path / "run", log=lambda line: None, report=False).run()
    assert results.rows[0]["reproduction_skill"] == pytest.approx(scored.reward, abs=1e-12)
    for inside in (
        "calibrated",
        "synthetic/bouncing_ball/calibrated",
        "synthetic__bouncing_ball__calibrated",
    ):
        ambiguous = tmp_path / ("ambiguous-" + inside.replace("/", "-"))
        shutil.copytree(solution, ambiguous)
        (ambiguous / inside).mkdir(parents=True)
        (ambiguous / inside / "run.py").write_text("raise SystemExit(1)\n")
        out = tmp_path / ("run-" + inside.replace("/", "-"))
        argv = ["eval", "--tasks", str(bouncing_task), "--method",
                f"prog=program:path={ambiguous}", "--backend", "local", "--out", str(out)]  # fmt: skip
        capsys.readouterr()
        assert main(argv) == 1, inside
        err = capsys.readouterr().err
        assert f"({inside}/)" in err and "fix:" in err and "Traceback" not in err, err
        assert not out.exists(), inside  # refused before anything ran


def _result(protocol: str, headline: Headline | None, skill: float | None = None) -> TaskResult:
    return TaskResult(
        task_id="b/t", task_version="1", task_hash="0" * 64, benchmark="b", method="m",
        method_kind="agent", trial=0, seed=0, protocol=protocol, protocol_version="1",
        delivered=True, reproduction=Reproduction(status=ReproductionStatus.SCORED),
        summary=TaskSummary(reproduction_skill=skill), headline=headline,
        created_at=datetime.now(UTC), lab_version="1",
    )  # fmt: skip


def test_reward_of_each_protocol() -> None:
    assert reward_of(_result("reproduction", None, 0.25)) == 0.25
    assert reward_of(_result("reproduction", None)) == 0.0

    def native(value: float | None, higher: bool = True) -> TaskResult:
        headline = Headline(
            metric="x", value=value, range=(0.0, 100.0), higher_is_better=higher, source="official"
        )
        return _result("native", headline)

    assert reward_of(native(75.0)) == 0.75
    assert reward_of(native(75.0, higher=False)) == 0.25
    assert reward_of(native(None)) == 0.0
    assert reward_of(native(150.0)) == 1.0


def test_rewards_are_for_the_reproduction_protocol(task_copy: Path, local) -> None:
    data = read_task_config(task_copy / "task.toml").to_json()
    data["scoring"] = {
        "protocol": "native",
        "native": {"scorer": "fourdcodebench", "options": {"kind": "synthetic", "case": "B_30"}},
    }
    write_task_config(TaskConfig.model_validate(data), task_copy / "task.toml")
    with pytest.raises(SetupError, match="native protocol"):
        score_solution(task_copy, task_copy, backend=local)


def _renamed(task: Path, target: Path, task_id: str) -> Path:
    shutil.copytree(task, target)
    data = read_task_config(target / "task.toml").to_json()
    data["id"] = task_id
    write_task_config(TaskConfig.model_validate(data), target / "task.toml")
    return target


def test_export_keeps_evaluation_tasks_out(bouncing_task: Path, tmp_path: Path) -> None:
    with pytest.raises(SetupError, match="evaluation task"):
        export_dataset(tmp_path / "data", tasks=[bouncing_task])
    assert not (tmp_path / "data").exists()
    summary = export_dataset(tmp_path / "data", tasks=[bouncing_task], allow_eval_tasks=True)
    assert summary.tasks == 1
    assert "contains evaluation tasks" in (tmp_path / "data" / "README.md").read_text()
    # A benchmark whose own terms forbid training on its data: never, even when asked.
    forbidden = _renamed(bouncing_task, tmp_path / "piq", "physicsiq/collision/center/lab")
    with pytest.raises(SetupError, match="physicsiq's terms forbid"):
        export_dataset(tmp_path / "data2", tasks=[forbidden], allow_eval_tasks=True)


@pytest.mark.slow
def test_export_holds_inputs_labels_programs_and_rewards(tmp_path: Path) -> None:
    roots = build_training(tmp_path / "tasks", ["bouncing_ball"], 2)
    config = parse_experiment(
        {
            "name": "export",
            "tasks": {"paths": [str(tmp_path / "tasks")]},
            "methods": [{"name": "oracle"}, {"name": "fake_agent", "mode": "physical"}],
            "run": {"backend": "local"},
        }
    )
    Experiment(config, tmp_path / "run", log=lambda line: None, report=False).run()
    out = tmp_path / "data"
    summary = export_dataset(out, tasks=[tmp_path / "tasks"], runs=[tmp_path / "run"])
    assert (summary.tasks, summary.runs) == (2, 4) and not summary.skipped
    tasks = [json.loads(line) for line in (out / "tasks.jsonl").read_text().splitlines()]
    assert {row["task_id"] for row in tasks} == {load_task(r).id for r in roots}
    for row in tasks:
        assert row["split"] == "train" and row["parameters"]["phenomenon"] == "bouncing_ball"
        assert (out / row["video"]).is_file() and (out / row["reference"]).is_dir()
        assert (out / row["truth_program"] / "run.py").is_file()
    runs = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    oracle = [r for r in runs if r["method"] == "oracle"]
    agent = [r for r in runs if r["method"] == "fake_agent"]
    assert all(r["reward"] == pytest.approx(1.0) and r["privileged"] for r in oracle)
    for row in agent:
        assert 0 < row["reward"] < 1 and not row["privileged"]
        assert (out / row["program"] / "run.py").is_file()
        assert (out / row["agent"]["trajectory"]).is_file() and row["statement"]
    best = export_dataset(
        tmp_path / "best", runs=[tmp_path / "run"], min_reward=0.99, command="r2s train export"
    )
    assert best.runs == 2
