"""What keeps training data out of evaluations, and user data safe, when things go wrong: a
checkpoint trained again, two versions of one task, missing or damaged prints, outputs that
overlap inputs, solutions that cannot be copied, and evaluation clips of other benchmarks."""

from __future__ import annotations

import json
import os
import shutil
import warnings
from collections.abc import Callable
from pathlib import Path

import pytest

from real2simlab._fs import DamagedCacheWarning
from real2simlab.benchmarks.base import get_benchmark
from real2simlab.benchmarks.synthetic import SyntheticBenchmark
from real2simlab.benchmarks.synthetic.learned import NearestTwin
from real2simlab.errors import SetupError
from real2simlab.methods.base import get_method
from real2simlab.methods.training import TrainContext
from real2simlab.orchestrate import Experiment, parse_experiment
from real2simlab.sandbox.local import LocalBackend
from real2simlab.task import TaskConfig, load_task
from real2simlab.task.io import read_task_config, write_task_config
from real2simlab.task.overlap import OverlapIndex, task_print
from real2simlab.train import export_dataset, fit_method, score_solution

QUIET: Callable[[str], None] = lambda line: None  # noqa: E731


@pytest.fixture(scope="module")
def tasks(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("training")
    built: dict[str, Path] = {}
    for phenomenon in ("bouncing_ball", "pendulum"):
        (path,) = SyntheticBenchmark().build(
            out, select=[phenomenon], tiers=["calibrated"], split="train", count=1, seed=4
        )
        built[phenomenon] = path
    return built


def _edited(task: Path, target: Path, **changes: object) -> Path:
    shutil.copytree(task, target)
    data = read_task_config(target / "task.toml").to_json()
    for key, value in changes.items():
        if key == "source_dataset":
            data["source"]["dataset"] = value
        else:
            data[key] = value
    write_task_config(TaskConfig.model_validate(data), target / "task.toml")
    return target


def test_a_checkpoint_trained_again_is_read_again(tasks: dict[str, Path], tmp_path: Path) -> None:
    """The training data of a checkpoint trained again in the same directory, in the same
    process, are the new ones (nothing cached by path alone)."""
    first, second = tasks["bouncing_ball"], tasks["pendulum"]
    export_dataset(tmp_path / "d1", tasks=[first])
    export_dataset(tmp_path / "d2", tasks=[second])
    model = tmp_path / "model"
    fit_method(get_method("nearest_twin"), tmp_path / "d1", model, log=QUIET)
    method = get_method("nearest_twin", checkpoint=str(model))
    assert method.held_out(load_task(first)) is not None
    assert method.held_out(load_task(second)) is None
    fit_method(get_method("nearest_twin"), tmp_path / "d2", model, overwrite=True, log=QUIET)
    again = get_method("nearest_twin", checkpoint=str(model))
    assert again.held_out(load_task(second)) is not None
    assert again.config_digest() != method.config_digest()


def test_an_empty_checkpoint_is_a_mistake_not_the_current_directory(
    tasks: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``checkpoint=$UNSET`` on the command line, or ``checkpoint = ""`` in an experiment file,
    is a mistake in the command, not the current directory: a checkpoint that happens to be the
    current directory would be evaluated without a word."""
    from real2simlab.cli.main import main

    data = tmp_path / "data"
    export_dataset(data, tasks=[tasks["bouncing_ball"]])
    model = tmp_path / "model"
    fit_method(get_method("nearest_twin"), data, model, log=QUIET)
    monkeypatch.chdir(model)  # the current directory is a checkpoint: it must not be taken
    with pytest.raises(SetupError, match="is empty") as raised:
        get_method("nearest_twin", checkpoint="")
    assert raised.value.fix
    experiment = tmp_path / "experiment.toml"
    experiment.write_text(
        f'name = "empty"\n[tasks]\npaths = ["{tasks["pendulum"]}"]\n'
        '[[methods]]\nname = "nearest_twin"\ncheckpoint = ""\n[run]\nbackend = "local"\n'
    )
    commands = [
        ["eval", "--tasks", str(tasks["pendulum"]), "--method", "nt=nearest_twin:checkpoint=",
         "--backend", "local", "--out", str(tmp_path / "run")],
        ["run", str(experiment), "--out", str(tmp_path / "run-file")],
    ]  # fmt: skip
    for argv in commands:
        capsys.readouterr()
        assert main(argv) == 1, argv[0]
        err = capsys.readouterr().err
        assert "is empty" in err and "fix:" in err and "Traceback" not in err, err
    assert not (tmp_path / "run").exists() and not (tmp_path / "run-file").exists()


def test_every_version_of_a_task_is_printed(tasks: dict[str, Path], tmp_path: Path) -> None:
    """Two runs scored two versions of one task id: the prints of both are kept."""
    runs = []
    for version in ("v1", "v2"):
        task = _edited(tasks["bouncing_ball"], tmp_path / version / "t")
        if version == "v2":
            reference = next((task / "hidden" / "reference").glob("*.npz"))
            reference.write_bytes(reference.read_bytes() + b"\0")
        program = tmp_path / f"program-{version}"
        shutil.copytree(task / "hidden" / "generator" / "program", program)
        config = parse_experiment(
            {
                "name": version,
                "tasks": {"paths": [str(task)]},
                "methods": [{"name": "program", "label": "mine", "path": str(program)}],
                "run": {"backend": "local"},
            }
        )
        Experiment(config, tmp_path / f"run-{version}", log=QUIET, report=False).run()
        runs.append(tmp_path / f"run-{version}")
    export_dataset(tmp_path / "data", runs=runs)
    hashes = {json.loads(line)["task_hash"] for line in (tmp_path / "data" / "runs.jsonl").open()}
    assert len(hashes) == 2
    prints = OverlapIndex.load(tmp_path / "data" / "overlap-prints.npz").prints
    assert len(prints) == 2 and len({p.task_id for p in prints}) == 1


def test_a_checkpoint_without_prints_is_refused(tasks: dict[str, Path], tmp_path: Path) -> None:
    data = tmp_path / "data"
    export_dataset(data, tasks=[tasks["bouncing_ball"]])
    unprinted = tmp_path / "unprinted"
    shutil.copytree(data, unprinted)
    (unprinted / "overlap-prints.npz").unlink()
    with pytest.raises(SetupError, match=r"has no overlap-prints\.npz") as raised:
        fit_method(get_method("nearest_twin"), unprinted, tmp_path / "m0", log=QUIET)
    assert "r2s train export" in (raised.value.fix or "")
    model = tmp_path / "model"
    fit_method(get_method("nearest_twin"), data, model, log=QUIET)
    record = json.loads((model / "training.json").read_text())
    (model / "training.json").write_text(json.dumps({**record, "prints": None}))
    with pytest.raises(SetupError, match="not a valid training record"):
        get_method("nearest_twin", checkpoint=str(model))
    (model / "training.json").write_text(json.dumps(record))
    (model / "training-prints.npz").write_bytes(b"damaged")
    with pytest.raises(SetupError, match="has changed since r2s train fit") as changed:
        get_method("nearest_twin", checkpoint=str(model))
    assert "r2s train fit" in (changed.value.fix or "")
    # the digest made to match (a trainer that wrote damaged prints): they are still found
    from real2simlab.methods.training import content_digest

    rewritten = {**record, "content_digest": content_digest(model)}
    (model / "training.json").write_text(json.dumps(rewritten))
    with pytest.raises(SetupError, match="damaged") as damaged:
        get_method("nearest_twin", checkpoint=str(model))
    assert "r2s train fit" in (damaged.value.fix or "")


def test_an_edited_checkpoint_record_does_not_turn_the_held_out_check_off(
    tasks: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``training.json`` names the checkpoint's prints and the digest of its files. An edited
    record does not make an evaluation skip the held-out check: prints outside the checkpoint,
    prints that leave out its training tasks, or a record without its digest (edited, or written
    by a pre-release build of the Lab, whatever Lab version it names) are refused when the
    evaluation starts."""
    from real2simlab.cli.main import main
    from real2simlab.methods.training import content_digest

    data = tmp_path / "data"
    export_dataset(data, tasks=[tasks["bouncing_ball"]])
    model = tmp_path / "model"
    fit_method(get_method("nearest_twin"), data, model, log=QUIET)
    record = json.loads((model / "training.json").read_text())
    empty = OverlapIndex().save(tmp_path / "empty-prints.npz")

    def refused(edited: dict[str, object], match: str) -> None:
        (model / "training.json").write_text(json.dumps(edited))
        with pytest.raises(SetupError, match=match) as raised:
            get_method("nearest_twin", checkpoint=str(model))
        assert "r2s train fit" in (raised.value.fix or "")

    refused({**record, "prints": str(empty)}, "not a file in the checkpoint")
    refused({**record, "prints": "../empty-prints.npz"}, "not a file in the checkpoint")
    unrecorded = {key: value for key, value in record.items() if key != "content_digest"}
    refused(unrecorded, "records no content_digest .* pre-release build")
    refused({**unrecorded, "lab_version": "0.9.0"}, "records no content_digest")
    argv = ["eval", "--tasks", str(tasks["pendulum"]), "--method",
            f"nearest_twin:checkpoint={model}", "--backend", "local", "--out", str(tmp_path / "r")]  # fmt: skip
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "error:" in err and "fix:" in err and "Traceback" not in err, err
    # the prints replaced, and the digest made to match: they no longer cover the training task
    shutil.copyfile(empty, model / "training-prints.npz")
    refused({**record, "content_digest": content_digest(model)}, "no print of")
    shutil.copyfile(data / "overlap-prints.npz", model / "training-prints.npz")
    (model / "training.json").write_text(json.dumps(record))
    method = get_method("nearest_twin", checkpoint=str(model))  # as trained: usable
    assert method.held_out(load_task(tasks["bouncing_ball"])) is not None


def test_a_dataset_whose_prints_do_not_hold_its_recordings_is_refused(
    tasks: dict[str, Path], tmp_path: Path
) -> None:
    """The prints of a dataset are what keeps a method trained on it off its training tasks:
    a recording of the dataset that no print of its task holds (here another task's video in
    place of its own, in a dataset whose ``content`` record was rewritten to match) would be
    learned from but never held out, so ``r2s train fit`` refuses the dataset."""
    from real2simlab.train.dataset import content_digests

    data = tmp_path / "data"
    export_dataset(data, tasks=[tasks["bouncing_ball"]])
    row = json.loads((data / "tasks.jsonl").read_text().splitlines()[0])
    (other,) = (tasks["pendulum"] / "visible").rglob("*.mp4")
    shutil.copyfile(other, data / row["video"])
    report = json.loads((data / "export.json").read_text())
    (data / "export.json").write_text(json.dumps({**report, "content": content_digests(data)}))
    with pytest.raises(SetupError, match=r"overlap-prints\.npz .* no print holds") as raised:
        fit_method(get_method("nearest_twin"), data, tmp_path / "model", log=QUIET)
    assert "r2s train export" in (raised.value.fix or "")
    assert not (tmp_path / "model").exists()


def test_overwrite_replaces_only_what_the_command_wrote(
    tasks: dict[str, Path], tmp_path: Path
) -> None:
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "notes.txt").write_text("keep me")
    with pytest.raises(SetupError, match="not an exported dataset"):
        export_dataset(precious, tasks=[tasks["bouncing_ball"]], overwrite=True)
    data = tmp_path / "data"
    export_dataset(data, tasks=[tasks["bouncing_ball"]])
    with pytest.raises(SetupError, match="not a checkpoint"):
        fit_method(get_method("nearest_twin"), data, precious, overwrite=True, log=QUIET)
    with pytest.raises(SetupError, match="overlaps the dataset"):
        fit_method(get_method("nearest_twin"), data, data.parent, overwrite=True, log=QUIET)
    assert (precious / "notes.txt").read_text() == "keep me"
    empty = tmp_path / "empty"
    empty.mkdir()
    fit_method(get_method("nearest_twin"), data, empty, overwrite=True, log=QUIET)
    assert (empty / "training.json").is_file()


def test_a_checkpoint_holds_regular_files_only(
    tasks: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    export_dataset(data, tasks=[tasks["bouncing_ball"]])

    def linked(self: NearestTwin, context: TrainContext) -> None:
        (context.checkpoint_dir / "weights").symlink_to(data / "tasks.jsonl")

    monkeypatch.setattr(NearestTwin, "train", linked)
    with pytest.raises(SetupError, match="symbolic link"):
        fit_method(get_method("nearest_twin"), data, tmp_path / "model", log=QUIET)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["data"]


def test_damaged_thumbnails_are_computed_again(tasks: dict[str, Path], tmp_path: Path) -> None:
    task = load_task(tasks["bouncing_ball"])
    first = task_print(task, cache=tmp_path / "cache")
    (entry,) = (tmp_path / "cache").glob("*.npz")
    entry.write_bytes(b"")  # a crash, a full disk
    with pytest.warns(DamagedCacheWarning):
        again = task_print(task, cache=tmp_path / "cache")
    assert (again.videos[0].thumbs == first.videos[0].thumbs).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        task_print(task, cache=tmp_path / "cache")  # repaired


def test_the_work_directory_never_holds_the_solution(
    tasks: dict[str, Path], tmp_path: Path
) -> None:
    task = tasks["bouncing_ball"]
    work = tmp_path / "work"
    local = LocalBackend(quiet=True)
    score_solution(task, task / "hidden" / "generator" / "program", backend=local, work_dir=work)
    with pytest.raises(SetupError, match="overlaps an input"):
        score_solution(task, work / "program", backend=local, work_dir=work)
    assert (work / "program" / "run.py").is_file()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
def test_a_solution_that_cannot_be_copied_is_not_delivered(
    tasks: dict[str, Path], tmp_path: Path
) -> None:
    """Its own fault, so it scores 0: it cannot turn its failure into a verifier error."""
    task = tasks["bouncing_ball"]
    solution = tmp_path / "solution"
    shutil.copytree(task / "hidden" / "generator" / "program", solution)
    os.mkfifo(solution / "pipe")
    scored = score_solution(task, solution, backend=LocalBackend(quiet=True))
    assert scored.reward == 0.0 and not scored.result.delivered
    assert "cannot be delivered" in (scored.result.delivery_failure or "")


def test_physics_iq_clips_of_other_benchmarks_are_never_training_data(
    tasks: dict[str, Path], tmp_path: Path
) -> None:
    clip = _edited(
        tasks["bouncing_ball"], tmp_path / "clip", id="physcodegym/clip/frozen",
        split="eval", source_dataset="ShaoShuai0605/PhysCodeGym (Physics-IQ)",
    )  # fmt: skip
    other = _edited(
        tasks["pendulum"], tmp_path / "other", id="physcodegym/other/frozen",
        split="eval", source_dataset="ShaoShuai0605/PhysCodeGym (Pexels)",
    )  # fmt: skip
    summary = export_dataset(
        tmp_path / "data", tasks=[clip, other], allow_eval_tasks=True
    )  # asked for evaluation tasks: the Physics-IQ recording is still refused
    assert summary.tasks == 1
    assert any("Physics-IQ recording" in line for line in summary.skipped)


def test_the_split_is_not_an_option(tmp_path: Path) -> None:
    with pytest.raises(SetupError, match="--split"):
        get_benchmark("synthetic").build_split(tmp_path, "eval", split="train")
    assert not any(tmp_path.iterdir())


def test_paths_in_an_experiment_file_start_from_its_directory(tmp_path: Path) -> None:
    from real2simlab.orchestrate.config import load_experiment

    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "e.toml").write_text(
        'spec = "r2s-experiment/1.0"\nname = "e"\n[tasks]\npaths = ["tasks"]\n'
        '[[methods]]\nname = "program"\npath = "../twins"\n'
        '[[methods]]\nname = "nearest_twin"\ncheckpoint = "/abs/model"\n'
    )
    config, base = load_experiment(tmp_path / "experiments" / "e.toml")
    assert base == (tmp_path / "experiments").resolve()
    assert Path(config.methods[0].options["path"]).resolve() == (tmp_path / "twins").resolve()
    assert config.methods[1].options["checkpoint"] == "/abs/model"
