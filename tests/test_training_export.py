"""Training data stay true to what was scored: each exported reward comes with the program and
the task version it was computed on, failures outside the program are not rewards, and a
failed export never costs an existing dataset."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from real2simlab.benchmarks.synthetic import SyntheticBenchmark
from real2simlab.benchmarks.synthetic.learned import NearestTwin
from real2simlab.benchmarks.synthetic.programs import simulation_program
from real2simlab.errors import SetupError
from real2simlab.methods.base import get_method
from real2simlab.methods.training import TrainContext
from real2simlab.orchestrate import Experiment, parse_experiment
from real2simlab.sandbox.base import ExecResult
from real2simlab.sandbox.local import LocalBackend
from real2simlab.train import dataset as dataset_module
from real2simlab.train import export_dataset, fit_method, score_solution
from real2simlab.train.reward import RewardUnavailable


@pytest.fixture(scope="module")
def training_task(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("training")
    (path,) = SyntheticBenchmark().build(
        out, select=["bouncing_ball"], tiers=["calibrated"], split="train", count=1, seed=4
    )
    return path


def _copy_task(task: Path, target: Path) -> Path:
    shutil.copytree(task, target)
    return target


def _truth(task: Path, target: Path) -> Path:
    shutil.copytree(task / "hidden" / "generator" / "program", target)
    return target


def _physical(target: Path) -> Path:
    return simulation_program(
        target, phenomenon="bouncing_ball", title="t",
        params={"restitution": 0.7, "rest_speed": 0.05},
    )  # fmt: skip


def _run(tasks: Path, program: Path, out: Path, label: str = "mine") -> Path:
    config = parse_experiment(
        {
            "name": "export",
            "tasks": {"paths": [str(tasks)]},
            "methods": [{"name": "program", "label": label, "path": str(program)}],
            "run": {"backend": "local"},
        }
    )
    Experiment(config, out, log=lambda line: None, report=False).run()
    return out


def _rows(dataset: Path, name: str = "runs.jsonl") -> list[dict[str, Any]]:
    return [json.loads(line) for line in (dataset / name).read_text().splitlines()]


def test_runs_with_the_same_directory_name_keep_their_own_programs(
    training_task: Path, tmp_path: Path
) -> None:
    tasks = _copy_task(training_task, tmp_path / "tasks" / "t")
    first = _run(tasks, _truth(training_task, tmp_path / "truth"), tmp_path / "a" / "run")
    second = _run(tasks, _physical(tmp_path / "physical"), tmp_path / "b" / "run")
    summary = export_dataset(tmp_path / "data", tasks=[tasks], runs=[first, second])
    assert summary.runs == summary.programs == 2
    rows = _rows(tmp_path / "data")
    assert len({row["program"] for row in rows}) == 2 and len({row["run"] for row in rows}) == 2
    rewards = sorted(row["reward"] for row in rows)
    assert 0 < rewards[0] < 0.99 and rewards[1] == pytest.approx(1.0)
    for row in rows:  # each reward comes with the program that earned it
        exported = (tmp_path / "data" / row["program"] / "model.json").read_text()
        source = first if row["reward"] == pytest.approx(1.0) else second
        (built,) = source.glob("builds/*/mine/trial-0/twin/program/model.json")
        assert exported == built.read_text()


def test_a_changed_task_does_not_take_its_old_rewards(training_task: Path, tmp_path: Path) -> None:
    tasks = _copy_task(training_task, tmp_path / "tasks" / "t")
    original = _copy_task(tasks, tmp_path / "original" / "t")
    run = _run(tasks, _physical(tmp_path / "physical"), tmp_path / "run")
    reference = next((tasks / "hidden" / "reference").glob("*.npz"))
    reference.write_bytes(reference.read_bytes() + b"\0")  # a new version, the same id
    with pytest.raises(SetupError, match="has changed since the run scored it"):
        export_dataset(tmp_path / "data", runs=[run])
    skipped = export_dataset(
        tmp_path / "data", tasks=[tasks], runs=[run], skip_unmatched=True
    )  # the task itself is still exported; its old reward is not
    assert (skipped.tasks, skipped.runs) == (1, 0)
    assert any("has changed" in line for line in skipped.skipped)
    matched = export_dataset(tmp_path / "data2", tasks=[original], runs=[run])
    (row,) = _rows(tmp_path / "data2")
    (task_row,) = _rows(tmp_path / "data2", "tasks.jsonl")
    assert matched.runs == 1 and row["task_hash"] == task_row["task_hash"]


def test_an_archived_run_finds_its_tasks_among_the_given_ones(
    training_task: Path, tmp_path: Path
) -> None:
    tasks = _copy_task(training_task, tmp_path / "work" / "tasks" / "t")
    run = _run(tasks, _physical(tmp_path / "physical"), tmp_path / "work" / "runs" / "r")
    archive = tmp_path / "archive"
    shutil.copytree(run, archive / "runs-2026" / "r")
    shutil.copytree(tasks, archive / "tasks-2026" / "t")
    shutil.rmtree(tmp_path / "work")
    with pytest.raises(SetupError, match="not found where the run found it"):
        export_dataset(tmp_path / "data", runs=[archive / "runs-2026" / "r"])
    summary = export_dataset(
        tmp_path / "data", tasks=[archive / "tasks-2026"], runs=[archive / "runs-2026" / "r"]
    )
    assert summary.runs == summary.programs == 1


def test_undelivered_results_are_not_programs(training_task: Path, tmp_path: Path) -> None:
    tasks = _copy_task(training_task, tmp_path / "tasks" / "t")
    (tmp_path / "nothing").mkdir()
    run = _run(tasks, tmp_path / "nothing", tmp_path / "run")
    with pytest.raises(SetupError, match="nothing delivered"):
        export_dataset(tmp_path / "data", runs=[run])
    summary = export_dataset(tmp_path / "data", runs=[run], include_undelivered=True)
    (row,) = _rows(tmp_path / "data")
    assert (summary.runs, summary.programs) == (1, 0)
    assert row["program"] is None and row["delivered"] is False and row["reward"] == 0.0


def test_results_of_runs_that_failed_outside_the_program_are_not_rewards(
    training_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = _copy_task(training_task, tmp_path / "tasks" / "t")
    real = LocalBackend.run

    def broken(self: LocalBackend, spec: Any, log: Path) -> ExecResult:
        if spec.label == "r2s-twin":
            return ExecResult(None, False, 0.0, "the container did not start")
        return real(self, spec, log)

    monkeypatch.setattr(LocalBackend, "run", broken)
    run = _run(tasks, _physical(tmp_path / "physical"), tmp_path / "run")
    summary = export_dataset(tmp_path / "data", tasks=[tasks], runs=[run])
    assert summary.runs == 0 and any("infra_error" in line for line in summary.skipped)


def test_a_result_file_changed_after_the_run_is_refused(
    training_task: Path, tmp_path: Path
) -> None:
    tasks = _copy_task(training_task, tmp_path / "tasks" / "t")
    run = _run(tasks, _physical(tmp_path / "physical"), tmp_path / "run")
    (result,) = run.glob("results/*/mine/trial-0/seed-0.json")
    data = json.loads(result.read_text())
    data["summary"]["reproduction_skill"] = 1.0
    result.write_text(json.dumps(data))
    with pytest.raises(SetupError, match=r"changed since results\.json was written"):
        export_dataset(tmp_path / "data", tasks=[tasks], runs=[run])


def test_a_reward_is_never_a_broken_sandbox(
    training_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    truth = _truth(training_task, tmp_path / "truth")
    monkeypatch.setattr(
        LocalBackend, "run", lambda self, spec, log: ExecResult(None, False, 0.0, "mount failed")
    )
    with pytest.raises(RewardUnavailable, match="infra_error") as raised:
        score_solution(training_task, truth, backend=LocalBackend(quiet=True))
    assert raised.value.result.reproduction.run_status == "infra_error"


def test_the_reward_command_fails_on_a_broken_sandbox(training_task: Path, tmp_path: Path) -> None:
    """The command's status is what batch scorers check: no reward, a non-zero status."""
    truth = _truth(training_task, tmp_path / "truth")
    script = (
        "import sys\n"
        "from real2simlab.sandbox.base import ExecResult\n"
        "from real2simlab.sandbox.local import LocalBackend\n"
        "LocalBackend.run = lambda self, spec, log: ExecResult(None, False, 0.0, 'mount failed')\n"
        "from real2simlab.cli.main import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, "train", "reward", "--task", str(training_task),
         "--solution", str(truth), "--backend", "local"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert result.returncode != 0 and "infra_error" in result.stderr
    # one error line without the error's class, and the fix on the next (it was
    # "error: RewardUnavailable: ..." with no fix: line)
    lines = result.stderr.strip().splitlines()
    assert lines[-2].startswith("error: no reward:") and "RewardUnavailable" not in lines[-2]
    assert lines[-1].startswith("  fix: this machine's failure, not the solution's"), lines
    assert "reward" not in result.stdout


def test_a_work_directory_can_be_scored_in_again(training_task: Path, tmp_path: Path) -> None:
    truth = _truth(training_task, tmp_path / "truth")
    work = tmp_path / "work"
    local = LocalBackend(quiet=True)
    first = score_solution(training_task, truth, backend=local, work_dir=work)
    (work / "runs" / "leftover").mkdir(parents=True)  # an interrupted attempt
    again = score_solution(training_task, truth, backend=local, work_dir=work)
    assert first.reward == again.reward == pytest.approx(1.0)
    foreign = tmp_path / "mine"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep me")
    with pytest.raises(SetupError, match="not empty"):
        score_solution(training_task, truth, backend=local, work_dir=foreign)
    assert (foreign / "notes.txt").read_text() == "keep me"


def test_a_failed_overwrite_keeps_the_dataset(
    training_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "data"
    export_dataset(out, tasks=[training_task])
    before = sorted(path.relative_to(out) for path in out.rglob("*"))
    with pytest.raises(SetupError, match="is not a directory"):
        export_dataset(out, tasks=[tmp_path / "missing"], overwrite=True)

    def broken(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise OSError("disk full")

    monkeypatch.setattr(dataset_module, "_task_row", broken)
    with pytest.raises(OSError, match="disk full"):
        export_dataset(out, tasks=[training_task], overwrite=True)
    assert sorted(path.relative_to(out) for path in out.rglob("*")) == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["data"]  # nothing partial left
    monkeypatch.undo()
    export_dataset(out, tasks=[training_task], overwrite=True)
    assert (out / "tasks.jsonl").is_file()


DECOY = b"R2S-TEST-DECOY dataset export"


@pytest.mark.parametrize("mistake", ["link to hidden", "copy of hidden", "copy listed as masks"])
def test_a_task_that_shows_hidden_data_is_not_exported(
    mistake: str, training_task: Path, tmp_path: Path
) -> None:
    """A dataset's ``visible/`` is what a method trained on it sees: a task whose ``visible/``
    shows hidden data (a link into ``hidden/``, a copy of a hidden file, also one that
    ``capture.json`` lists as the view's masks) is refused as ``r2s task validate`` refuses it,
    with the fix, and no dataset is written."""
    task = _copy_task(training_task, tmp_path / "tasks" / "t")
    marker = np.frombuffer(DECOY, dtype=np.uint8)
    np.savez(task / "hidden" / "decoy_labels.npz", marker=marker)
    visible = task / "visible" / "cam" / "masks.npz"
    if mistake == "link to hidden":
        visible.symlink_to(Path("..") / ".." / "hidden" / "decoy_labels.npz")
    else:
        shutil.copyfile(task / "hidden" / "decoy_labels.npz", visible)
    capture_file = task / "visible" / "capture.json"
    if mistake == "copy listed as masks":
        capture = json.loads(capture_file.read_text())
        masks = {"path": "cam/masks.npz", "format": "npz", "provenance": "estimated"}
        capture["views"]["cam"]["masks"] = masks
        capture_file.write_text(json.dumps(capture))
    with pytest.raises(SetupError, match="is invalid") as raised:
        export_dataset(tmp_path / "data", tasks=[task])
    assert "visible/cam/masks.npz" in str(raised.value)
    assert f"r2s task validate {task}" in (raised.value.fix or "")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["tasks"]  # nothing written
    visible.unlink()  # corrected: the task is exported, and nothing hidden is in its visible/
    shutil.copyfile(training_task / "visible" / "capture.json", capture_file)
    summary = export_dataset(tmp_path / "data", tasks=[task])
    assert summary.tasks == 1
    exported = [p for p in (tmp_path / "data" / "tasks").rglob("*") if p.is_file()]
    assert not [p for p in exported if "visible" in p.parts and DECOY in p.read_bytes()]


@pytest.mark.parametrize("mistake", ["copy of hidden", "link to hidden"])
def test_a_statement_that_shows_hidden_data_is_not_exported(
    mistake: str, training_task: Path, tmp_path: Path
) -> None:
    """A row's statement is what a method trained on the dataset is told: a task whose
    ``instruction.md`` (which replaces the statement) is a copy of a hidden file, or a link, is
    refused as ``r2s task validate`` refuses it, with the fix, and no dataset is written. A
    statement of its own becomes the row's statement, made from what was checked; the dataset
    holds no ``instruction.md``."""
    task = _copy_task(training_task, tmp_path / "tasks" / "t")
    (task / "hidden" / "decoy.txt").write_bytes(DECOY)
    statement = task / "instruction.md"
    if mistake == "copy of hidden":
        shutil.copyfile(task / "hidden" / "decoy.txt", statement)
    else:
        statement.symlink_to(Path("hidden") / "decoy.txt")
    with pytest.raises(SetupError, match="is invalid") as raised:
        export_dataset(tmp_path / "data", tasks=[task])
    assert "instruction.md" in str(raised.value)
    assert f"r2s task validate {task}" in (raised.value.fix or "")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["tasks"]  # nothing written
    statement.unlink()  # corrected: a statement of its own
    statement.write_text("Build a twin of the ball in {workspace}.\n")
    assert export_dataset(tmp_path / "data", tasks=[task]).tasks == 1
    rows = (tmp_path / "data" / "tasks.jsonl").read_text().splitlines()
    assert [json.loads(row)["statement"] for row in rows] == [
        "Build a twin of the ball in /workspace.\n"
    ]
    assert not list((tmp_path / "data").rglob("instruction.md"))


def test_the_output_never_overlaps_an_input(training_task: Path, tmp_path: Path) -> None:
    tasks = _copy_task(training_task, tmp_path / "tasks" / "t")
    for out in (tmp_path / "tasks", tmp_path / "tasks" / "t" / "data", tmp_path):
        with pytest.raises(SetupError, match="overlaps an input"):
            export_dataset(out, tasks=[tmp_path / "tasks"], overwrite=True)
    assert (tasks / "task.toml").is_file()
    data = tmp_path / "data"
    export_dataset(data, tasks=[tasks])
    with pytest.raises(SetupError, match="overlaps the dataset"):
        fit_method(get_method("nearest_twin"), data, data, overwrite=True, log=lambda line: None)
    assert (data / "tasks.jsonl").is_file()


def test_a_failed_training_keeps_the_checkpoint(
    training_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    export_dataset(data, tasks=[training_task])
    out = tmp_path / "model"
    quiet: Callable[[str], None] = lambda line: None  # noqa: E731
    fit_method(get_method("nearest_twin"), data, out, log=quiet)

    def files() -> dict[Path, bytes]:
        return {
            path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()
        }

    before = files()
    assert Path("training.json") in before
    with pytest.raises(SetupError, match="exists"):
        fit_method(get_method("nearest_twin"), data, out, log=quiet)

    def broken(self: NearestTwin, context: TrainContext) -> None:
        (context.checkpoint_dir / "half-written.npz").write_bytes(b"\0")
        raise RuntimeError("out of memory")

    monkeypatch.setattr(NearestTwin, "train", broken)
    with pytest.raises(RuntimeError, match="out of memory"):
        fit_method(get_method("nearest_twin"), data, out, overwrite=True, log=quiet)
    assert files() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["data", "model"]
    monkeypatch.undo()
    record = fit_method(get_method("nearest_twin"), data, out, overwrite=True, log=quiet)
    assert record.tasks == 1 and (out / "training.json").is_file()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["data", "model"]


def _renamed(task: Path, target: Path, task_id: str) -> Path:
    """A copy of ``task`` under another id: another task in a dataset."""
    from real2simlab.task.io import read_task_config, write_task_config

    shutil.copytree(task, target)
    config = read_task_config(target / "task.toml")
    write_task_config(config.model_copy(update={"id": task_id}), target / "task.toml")
    return target


def test_a_dataset_replaced_while_it_is_read_for_training_says_so(
    training_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``r2s train export --overwrite`` to the dataset while ``r2s train fit`` reads it (A7-08):
    a file the method then cannot find, or a check that then fails, is the dataset replaced,
    with its fix, not a file missing from the export (an error without a fix). No checkpoint
    is written."""
    from real2simlab.cli.main import main
    from real2simlab.train import fit as fit_module

    data, model = tmp_path / "data", tmp_path / "model"
    export_dataset(data, tasks=[training_task])
    other = _renamed(training_task, tmp_path / "other", "synthetic-train/bouncing_ball/other")

    def train_while_exported(self: NearestTwin, context: TrainContext) -> None:
        rows = (context.dataset / "tasks.jsonl").read_text().splitlines()
        export_dataset(data, tasks=[other], overwrite=True)  # another command, meanwhile
        (context.dataset / json.loads(rows[0])["capture"]).read_text()  # no longer there

    monkeypatch.setattr(NearestTwin, "train", train_while_exported)
    with pytest.raises(SetupError, match="was replaced or changed while nearest_twin") as raised:
        fit_method(get_method("nearest_twin"), data, model, log=lambda line: None)
    assert isinstance(raised.value.__cause__, FileNotFoundError)
    assert "train again once the dataset is no longer being written" in (raised.value.fix or "")
    assert not model.exists()
    export_dataset(data, tasks=[training_task], overwrite=True)
    capsys.readouterr()
    assert main(["train", "fit", "nearest_twin", "--data", str(data), "--out", str(model)]) == 1
    err = capsys.readouterr().err
    assert "was replaced or changed while" in err and "\n  fix: train again" in err, err
    assert "does not exist" not in err and not model.exists()
    monkeypatch.undo()
    export_dataset(data, tasks=[training_task], overwrite=True)
    check_prints = fit_module._check_prints

    def check_while_exported(dataset: Path, task_ids: list[str]) -> None:
        export_dataset(data, tasks=[other], overwrite=True)  # its prints are of other tasks
        check_prints(dataset, task_ids)

    monkeypatch.setattr(fit_module, "_check_prints", check_while_exported)
    with pytest.raises(SetupError, match="was replaced while r2s train fit read it") as checked:
        fit_method(get_method("nearest_twin"), data, model, log=lambda line: None)
    assert "train again" in (checked.value.fix or "") and not model.exists()
    monkeypatch.undo()
    assert fit_method(get_method("nearest_twin"), data, model, log=lambda line: None).tasks == 1


def test_a_task_replaced_while_a_solution_is_scored_gives_no_reward(
    training_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The task written again (``r2s train tasks --overwrite``) between the solution's run and
    its scoring: the reference the scorer looks for is not there yet. That is the task changed
    while it was scored (score again), not a task without a reference (restore it)."""
    from real2simlab.train import reward

    task = _copy_task(training_task, tmp_path / "task")
    truth = _truth(training_task, tmp_path / "truth")
    run_twin = reward.run_twin

    def run_while_replaced(*args: Any, **options: Any) -> Any:
        ran = run_twin(*args, **options)
        task.rename(tmp_path / "task.old")  # its new version: only visible/ written so far
        shutil.copytree(tmp_path / "task.old" / "visible", task / "visible")
        return ran

    monkeypatch.setattr(reward, "run_twin", run_while_replaced)
    with pytest.raises(SetupError, match="changed while the solution was scored") as raised:
        score_solution(task, truth, backend=LocalBackend(quiet=True))
    assert "score it again once the task is no longer being edited" in (raised.value.fix or "")


def test_inputs_replaced_while_they_are_copied_into_a_dataset_are_said_to_have_changed(
    training_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task or a program replaced while the export copies it (the copy fails: a file is
    gone) is an input that changed while it was exported, with that fix, not a missing file."""
    tasks = _copy_task(training_task, tmp_path / "tasks" / "t")
    run = _run(tasks, _physical(tmp_path / "physical"), tmp_path / "run")
    copytree = shutil.copytree
    moved: list[Path] = []

    def copy_while_replaced(source: Any, target: Any, *args: Any, **options: Any) -> Any:
        source = Path(source)
        replaced = tasks if source == tasks / "hidden" else None
        if source.name == "program" and "builds" in source.parts:
            replaced = source.parent  # the frozen twin, rebuilt by another command
        if replaced is not None and not moved:
            moved.append(replaced.rename(replaced.with_name(replaced.name + ".old")))
        return copytree(source, target, *args, **options)

    monkeypatch.setattr(shutil, "copytree", copy_while_replaced)
    with pytest.raises(SetupError, match="changed while it was exported") as raised:
        export_dataset(tmp_path / "data", tasks=[tasks])
    assert "export again" in (raised.value.fix or "")
    moved.pop().rename(tasks)
    with pytest.raises(SetupError, match="the program it scored is missing") as program:
        export_dataset(tmp_path / "data", runs=[run])
    assert "export again once nothing writes to the run directory" in (program.value.fix or "")
    assert not (tmp_path / "data").exists()


def test_a_dataset_whose_executable_bits_changed_is_refused(
    training_task: Path, tmp_path: Path
) -> None:
    """A copy or an archive that keeps no file modes (a zip file, an object store) clears a
    program's executable bits: a program that runs a helper then behaves differently, so the
    reward next to it is no longer its own (A7-03). ``export.json`` records the executable
    files, and ``r2s train fit`` refuses a dataset in which a bit was cleared or set, or to
    which a symbolic link was added, as it refuses an edited file."""
    task = _copy_task(training_task, tmp_path / "tasks" / "t")
    helper = task / "hidden" / "generator" / "program" / "helper.sh"
    helper.write_text("#!/bin/sh\nexit 0\n")
    helper.chmod(0o755)
    data = tmp_path / "data"
    export_dataset(data, tasks=[task])
    report = json.loads((data / "export.json").read_text())
    (executable,) = report["executable"]
    assert executable.endswith("program/helper.sh") and executable in report["content"]
    exported, run = data / executable, (data / executable).with_name("run.py")
    model = tmp_path / "model"

    def refused(match: str) -> None:
        with pytest.raises(SetupError, match=match) as raised:
            fit_method(get_method("nearest_twin"), data, model, log=lambda line: None)
        assert "r2s train export" in (raised.value.fix or "") and not model.exists()

    exported.chmod(0o644)
    refused(r"helper\.sh: changed \(its executable bit was cleared\)")
    exported.chmod(0o755)
    run.chmod(0o755)
    refused(r"run\.py: changed \(its executable bit was set\)")
    run.chmod(0o644)
    decoy = tmp_path / "decoy.txt"
    decoy.write_text("not in the dataset\n")
    (data / "tasks" / "extra").symlink_to(decoy)
    refused(r"tasks/extra: added \(a symbolic link\)")
    (data / "tasks" / "extra").unlink()
    fit_method(get_method("nearest_twin"), data, model, log=lambda line: None)  # as exported
    # a dataset exported before the record existed: its files are checked, not their modes
    exported.chmod(0o644)
    unrecorded = {key: value for key, value in report.items() if key != "executable"}
    (data / "export.json").write_text(json.dumps(unrecorded))
    fit_method(get_method("nearest_twin"), data, model, overwrite=True, log=lambda line: None)


def test_nothing_exported_is_told_the_fix_for_why(
    training_task: Path,
    bouncing_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An export that leaves out everything it was given says what would export something,
    for the reason it left things out (A12-robustness-13/14): not "give training tasks" when
    training tasks, or runs on them, were given."""
    tasks = _copy_task(training_task, tmp_path / "tasks" / "t")

    def fix_of(**given: Any) -> str:
        with pytest.raises(SetupError, match="nothing could be exported") as raised:
            export_dataset(tmp_path / "data", **given)
        return raised.value.fix or ""

    given_again = 'give training tasks (split = "train"'
    assert given_again in fix_of(tasks=[bouncing_task])  # evaluation tasks: as before
    overlapping = fix_of(tasks=[tasks], against=[_copy_task(tasks, tmp_path / "copy" / "t")])
    assert "--against" in overlapping and given_again not in overlapping
    run = _run(tasks, _physical(tmp_path / "physical"), tmp_path / "run")
    low = fix_of(runs=[run], min_reward=0.999)
    assert "lower --min-reward (the best reward left out is 0." in low and given_again not in low
    (tmp_path / "nothing").mkdir()
    undelivered = fix_of(runs=[_run(tasks, tmp_path / "nothing", tmp_path / "run-nothing")])
    assert "--include-undelivered" in undelivered and given_again not in undelivered
    reference = next((tasks / "hidden" / "reference").glob("*.npz"))
    reference.write_bytes(reference.read_bytes() + b"\0")  # every result's task changed since
    unmatched = fix_of(runs=[run], skip_unmatched=True)
    assert "give the tasks as they were when the runs scored them" in unmatched
    assert "--skip-unmatched" not in unmatched and given_again not in unmatched
    real = LocalBackend.run

    def broken(self: LocalBackend, spec: Any, log: Path) -> ExecResult:
        if spec.label == "r2s-twin":
            return ExecResult(None, False, 0.0, "the container did not start")
        return real(self, spec, log)

    monkeypatch.setattr(LocalBackend, "run", broken)
    infra = _run(tasks, _physical(tmp_path / "physical-2"), tmp_path / "run-infra")
    monkeypatch.undo()
    not_scored = fix_of(runs=[infra])
    assert "run the same r2s eval or r2s run command again" in not_scored
    assert given_again not in not_scored


def test_a_thumbnail_cache_inside_an_input_is_refused(
    training_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The export keeps video thumbnails in the Lab's cache: one inside a task would write them
    into the task (then "changed while it was exported", and exported as its data on the
    retry). It is refused before anything is written (A12-robustness-16)."""
    task = _copy_task(training_task, tmp_path / "tasks" / "t")
    before = sorted(path.relative_to(task) for path in task.rglob("*"))
    for cache in (task / "visible" / "inside-cache", task):
        with pytest.raises(SetupError, match="lies inside the task") as raised:
            export_dataset(tmp_path / "data", tasks=[task], cache=cache)
        assert "R2S_CACHE_DIR" in (raised.value.fix or "")
    monkeypatch.setenv("R2S_CACHE_DIR", str(task / "visible" / "inside-cache"))
    with pytest.raises(SetupError, match="lies inside the task"):
        export_dataset(tmp_path / "data", tasks=[tmp_path / "tasks"])
    assert sorted(path.relative_to(task) for path in task.rglob("*")) == before
    assert not (tmp_path / "data").exists()
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    assert export_dataset(tmp_path / "data", tasks=[task]).tasks == 1


def test_a_thumbnail_cache_beside_the_tasks_given_is_used(
    training_task: Path, tmp_path: Path
) -> None:
    """A thumbnail cache under a directory given to the export but in none of its tasks
    (``--tasks proj`` with ``R2S_CACHE_DIR=proj/.r2s-cache``) writes nothing into a task: the
    export works, as it did before the cache was checked."""
    project = tmp_path / "proj"
    task = _copy_task(training_task, project / "t")
    before = sorted(path.relative_to(task) for path in task.rglob("*"))
    cache = project / ".r2s-cache" / "thumbnails"
    assert export_dataset(tmp_path / "data", tasks=[project], cache=cache).tasks == 1
    assert any(path.is_file() for path in cache.rglob("*"))  # the thumbnails, there
    assert sorted(path.relative_to(task) for path in task.rglob("*")) == before


def test_a_thumbnail_cache_inside_a_task_a_run_found_is_refused(
    training_task: Path, tmp_path: Path
) -> None:
    """``--runs RUN`` reads each task where the run found it, which may be under no directory
    given to the export: a thumbnail cache inside that task is refused before a thumbnail is
    written into it (the task would no longer be the one the run scored), as one inside a task
    given with ``--tasks`` is."""
    task = _copy_task(training_task, tmp_path / "elsewhere" / "t")
    run = _run(task, _physical(tmp_path / "physical"), tmp_path / "run")
    before = sorted(path.relative_to(task) for path in task.rglob("*"))
    with pytest.raises(SetupError, match="lies inside the task") as raised:
        export_dataset(tmp_path / "data", runs=[run], cache=task / "visible" / "cache")
    assert "R2S_CACHE_DIR" in (raised.value.fix or "")
    assert sorted(path.relative_to(task) for path in task.rglob("*")) == before
    assert not (tmp_path / "data").exists()


DAMAGED_LABELS = {"cut short": b"{BROKEN", "not text": b"\xff\xfe{}", "not an object": b"[1, 2]\n"}


@pytest.mark.parametrize("damage", sorted(DAMAGED_LABELS))
def test_damaged_labels_are_named_by_validation_and_by_export(
    damage: str, training_task: Path, tmp_path: Path
) -> None:
    """A training task's hidden parameters (``hidden/parameters.json``, labels an export puts in
    its rows) that are not a JSON object are an error of the task: ``r2s task validate`` says
    so, and an export stops with an error naming the file and its fix, leaving no dataset. The
    task used to validate, and the export to fail with the decoder's words alone, or to export
    the broken labels."""
    from real2simlab.task import validate_task

    task = _copy_task(training_task, tmp_path / "tasks" / "t")
    labels = task / "hidden" / "parameters.json"
    labels.write_bytes(DAMAGED_LABELS[damage])
    report = validate_task(task)
    assert not report.ok
    assert any("hidden/parameters.json" in str(issue) for issue in report.errors), report.issues
    with pytest.raises(SetupError, match=r"parameters\.json") as raised:
        export_dataset(tmp_path / "data", tasks=[task])
    assert str(labels) in str(raised.value) and "r2s task validate" in (raised.value.fix or "")
    assert not (tmp_path / "data").exists()


def test_nearest_twin_names_a_dataset_row_whose_labels_are_not_a_truth(tmp_path: Path) -> None:
    """A dataset row whose ``parameters`` are not a JSON object (a dataset exported before the
    export checked them) stops nearest_twin's training with an error naming the row and the
    fix, not a traceback; a row of another kind of labels (no phenomenon it knows) is not
    learned from, and one of its phenomena without its parameters is an error too. A model
    holding such labels (trained before this check) is not a model it reads: its evaluation
    stops before anything runs, with the fix, instead of failing in each build."""
    import numpy as np

    from real2simlab.benchmarks.synthetic.learned import MODEL_FILE, Model

    data = tmp_path / "data"
    data.mkdir()

    def rows(*parameters: Any) -> None:
        lines = [
            json.dumps({"task_id": f"synthetic-train/x/s0-{i}", "parameters": labels,
                        "capture": "tasks/none/visible/capture.json", "metadata": {"index": i}})
            for i, labels in enumerate(parameters)
        ]  # fmt: skip
        (data / "tasks.jsonl").write_text("\n".join(lines) + "\n")

    def train() -> None:
        method = get_method("nearest_twin", count=1)
        method.train(TrainContext(dataset=data, checkpoint_dir=tmp_path / "model"))

    rows({"drop_height": 1.0}, [1, 2])
    with pytest.raises(SetupError, match=r"tasks\.jsonl, line 2") as raised:
        train()
    assert "not a JSON object" in str(raised.value) and "export" in (raised.value.fix or "")
    rows({"phenomenon": "bouncing_ball", "params": [0.7]})
    with pytest.raises(SetupError, match=r"tasks\.jsonl, line 1"):
        train()
    rows({"drop_height": 1.0})
    with pytest.raises(SetupError, match="no training task nearest_twin can learn from"):
        train()
    model = tmp_path / MODEL_FILE
    truth = {"phenomenon": "bouncing_ball", "params": [0.7], "setup": {}}
    Model(["ball@2"], [np.zeros((1, 2, 2))], [truth]).save(model)
    with pytest.raises(SetupError, match="is not a nearest_twin model") as raised:
        Model.load(model)
    assert "no params and setup objects" in str(raised.value) and raised.value.fix


def test_damaged_dataset_and_checkpoint_records_are_named(
    training_task: Path, tmp_path: Path
) -> None:
    """A dataset's ``tasks.jsonl`` or a checkpoint's ``training.json`` holding bytes that are not
    text: an error naming the file, with the fix, never the decoder's words alone."""
    from real2simlab.methods.training import TRAINING_FILE, read_training_record

    data = tmp_path / "data"
    export_dataset(data, tasks=[training_task])
    out = tmp_path / "model"
    fit_method(get_method("nearest_twin"), data, out, log=lambda line: None)
    with (data / "tasks.jsonl").open("ab") as stream:
        stream.write(b"\xff\n")
    with pytest.raises(SetupError, match=r"tasks\.jsonl") as raised:
        fit_method(get_method("nearest_twin"), data, tmp_path / "again", log=lambda line: None)
    assert "export the dataset again" in (raised.value.fix or "")
    (out / TRAINING_FILE).write_bytes(b'{"method": "\xff"}')
    with pytest.raises(SetupError, match=r"training\.json") as raised:
        read_training_record(out)
    assert raised.value.fix
