"""4DCodeBench: set metrics, the official-scorer plugin, the native protocol, task building."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from real2simlab.benchmarks.fourdcodebench import data_dir
from real2simlab.formats.tracks import ObjectTrack
from real2simlab.metrics import get_metric
from real2simlab.task import load_task, read_task_config, validate_task, write_task_config

from .conftest import docker_command, falling, make_tracks

CACHE_CASE = data_dir() / "synthetic" / "B_30"
"""Where the Lab keeps B_30 (``R2S_CACHE_DIR`` too): tests that need it skip without it,
rather than download it."""


def test_centroid_set_metrics_ignore_names_and_align() -> None:
    reference = make_tracks({"a": falling(), "b": falling() + [0.5, 0.0, 0.0]})
    renamed = replace(reference, objects={"x": reference.objects["b"], "y": reference.objects["a"]})
    chamfer, chamfer_2d = get_metric("centroid_set_chamfer"), get_metric("centroid_set_chamfer_2d")
    assert chamfer.compute(renamed, reference).value == pytest.approx(0.0)
    assert chamfer_2d.compute(renamed, reference).value == pytest.approx(0.0)
    scaled = replace(
        reference,
        objects={
            n: ObjectTrack(position=3.0 * t.position + [1.0, 2.0, 0.0])
            for n, t in reference.objects.items()
        },
    )
    assert chamfer.compute(scaled, reference).value > 0.5
    aligned = chamfer.compute(scaled, reference, options={"align": "similarity"})
    assert aligned.value == pytest.approx(0.0, abs=1e-9)
    merged = replace(reference, objects={"one": reference.objects["a"]})
    assert chamfer.compute(merged, reference).value == pytest.approx(0.125, rel=1e-6)
    assert chamfer.compute(reference, reference, options={"align": "bogus"}).status.value == "error"
    empty = replace(reference, objects={})
    assert chamfer.compute(empty, reference).status.value == "error"


def test_scorer_plugin_types(tmp_path: Path) -> None:
    from real2simlab.benchmarks.fourdcodebench.scorer import (
        CHECKPOINT_DIRS,
        CHECKPOINT_FREE_TYPES,
        MODEL_TYPES,
        FourDCodeBenchScorer,
    )

    assert FourDCodeBenchScorer(checkpoints=None).types() == CHECKPOINT_FREE_TYPES
    partial = tmp_path / "ckpt"
    (partial / CHECKPOINT_DIRS[0]).mkdir(parents=True)
    assert FourDCodeBenchScorer(checkpoints=str(partial)).types() == CHECKPOINT_FREE_TYPES
    for name in CHECKPOINT_DIRS:
        (partial / name).mkdir(exist_ok=True)
    full = FourDCodeBenchScorer(checkpoints=str(partial))
    assert full.complete_checkpoints() and full.types() == (*CHECKPOINT_FREE_TYPES, *MODEL_TYPES)
    missing = FourDCodeBenchScorer(image="r2s-test/does-not-exist:0").problems()
    assert missing
    # "not built" only where Docker answers: a docker command without its daemon is Docker
    # not being ready, whatever the image.
    assert "docker build" in missing[0] if docker_command() else "Docker" in missing[0]


@pytest.mark.parametrize(
    ("exit_code", "reading", "mean", "scored"),
    [
        (0, 0.1, 0.3, 2),  # a valid world
        (1, "error", 0.25, 2),  # a world the scorer's gates refused (rejected)
        (0, "error", 0.25, 2),  # judged, and the metric failed on the world
        (None, None, 0.25, 2),  # no world at all
        (0, "uncomputable", 0.5, 1),  # readings the official protocol leaves out
        (0, "excluded", 0.5, 1),
        (0, "not_applicable", 0.5, 1),
        (137, "error", 0.5, 1),  # a scorer killed judged nothing: never a 0
    ],
)
def test_a_world_the_official_scorer_refuses_counts_zero_in_the_mean(
    exit_code: int | None, reading: float | str | None, mean: float, scored: int
) -> None:
    """The official rule (4DCodeBench ``docs/spec.md``): in aggregates an ``error`` reading
    counts as the metric's worst value, 0. A method that delivered a world the scorer refuses
    is not ranked above one that delivered a poor world, or none (the mean left it out)."""
    from datetime import UTC, datetime

    from real2simlab.benchmarks.fourdcodebench.scorer import FourDCodeBenchScorer, official_status
    from real2simlab.scoring.aggregate import benchmark_scores
    from real2simlab.scoring.model import Reproduction, ReproductionStatus, TaskResult, TaskSummary

    scorer = FourDCodeBenchScorer()

    def report(exit_code: int | None, reading: float | str | None) -> dict[str, Any]:
        if exit_code is None:
            return {"status": "no_world", "reward": None}
        reward = {"dynamic_iou": reading, "gate_failure": "a reason"}
        return {
            "status": official_status(exit_code, reward),
            "exit_code": exit_code,
            "reward": reward,
        }

    def result(case: str, native: dict[str, Any]) -> TaskResult:
        return TaskResult(
            task_id=f"4dcodebench/synthetic/{case}", task_version="1", task_hash="0" * 64,
            benchmark="4dcodebench", method="m", method_kind="agent", trial=0, seed=0,
            protocol="native", protocol_version="1", delivered=native["status"] != "no_world",
            reproduction=Reproduction(status=ReproductionStatus.SCORED), summary=TaskSummary(),
            headline=scorer.headline(native), native=native, created_at=datetime.now(UTC),
            lab_version="1",
        )  # fmt: skip

    results = [result("b_30", report(0, 0.5)), result("b_31", report(exit_code, reading))]
    (card,) = benchmark_scores(results, resamples=10)
    assert (card.value.mean, card.scored, card.results) == (pytest.approx(mean), scored, 2)


def test_scorer_reports_missing_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    from real2simlab.benchmarks.fourdcodebench import scorer

    monkeypatch.setattr(scorer, "_docker", lambda: "/nonexistent/docker")
    problems = scorer.FourDCodeBenchScorer().problems()
    assert problems and "not available" in problems[0]


class FakeScorer:
    """A native scorer that records what it was given."""

    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **options: Any) -> None:
        self.options = options

    def problems(self) -> list[str]:
        return []

    def score(
        self,
        task: Any,
        *,
        delivery_dir: Path,
        output_dir: Path | None,
        work_dir: Path,
        options: Any,
    ) -> dict[str, Any]:
        FakeScorer.calls.append(
            {"output": output_dir, "options": dict(options), "init": self.options}
        )
        return {"status": "scored", "reward": {"fake": 0.5 if output_dir else 0.0}}


@pytest.mark.parametrize("rerun", [False, True])
def test_native_protocol(
    task_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rerun: bool
) -> None:
    from real2simlab.orchestrate import experiment as experiment_module
    from real2simlab.orchestrate import parse_experiment, run_experiment

    config = read_task_config(task_copy / "task.toml").to_json()
    config["scoring"] = {
        "protocol": "native",
        "native": {
            "scorer": "fourdcodebench",
            "options": {
                "kind": "synthetic",
                "case": "B_30",
                "rerun": rerun,
                "scorer": {"image": "x:1"},
            },
        },
    }
    from real2simlab.task import TaskConfig

    write_task_config(TaskConfig.model_validate(config), task_copy / "task.toml")
    monkeypatch.setattr(
        experiment_module, "get_scorer", lambda name, **options: FakeScorer(**options)
    )
    FakeScorer.calls = []
    experiment = parse_experiment(
        {
            "name": "native",
            "tasks": {"paths": [str(task_copy)]},
            "methods": [{"name": "fake_agent", "mode": "physical"}],
            "run": {"backend": "local"},
        }
    )
    run_experiment(experiment, tmp_path / "run", log=lambda line: None, report=False)
    result = json.loads(next((tmp_path / "run" / "results").rglob("seed-0.json")).read_text())
    assert result["native"]["reward"]["fake"] == 0.5 and result["protocol"] == "native"
    call = FakeScorer.calls[-1]
    assert call["init"] == {"image": "x:1"}
    if rerun:
        assert call["output"].parent.name == "run"
        assert result["reproduction"]["status"] == "scored"
        assert result["reproduction"]["session_consistent"] is True
    else:
        assert call["output"].name == "output" and "workspace" in call["output"].parts


@pytest.mark.skipif(
    not (CACHE_CASE / "world" / "camera.json").is_file(), reason="4DCodeBench B_30 not downloaded"
)
def test_build_tasks_from_the_cache(tmp_path: Path) -> None:
    from real2simlab.benchmarks.fourdcodebench import build_case

    for variant in ("native", "lab"):
        root = build_case("B_30", "synthetic", variant, tmp_path)
        report = validate_task(root)
        assert report.ok and not report.warnings, report.issues
        task = load_task(root)
        assert task.config.delivery.convention.value == "build_sh_world"
        assert task.config.redistributable is False
        statement = task.instruction_override() or ""
        assert "{{" not in statement and "/workspace/solution" in statement
        assert (task.visible_dir / "docs" / "world-format.md").is_file()
        assert not (task.visible_dir / "world").exists()
    assert (
        load_task(tmp_path / "synthetic" / "b_30" / "native").config.scoring.protocol.value
        == "native"
    )
    lab = load_task(tmp_path / "synthetic" / "b_30" / "lab")
    assert lab.config.scoring.primary == "centroid_set_chamfer_2d"
    # Only the re-run variant states the re-run's rules (read-only solution/).
    assert "**read-only**" in (lab.instruction_override() or "")
    native = load_task(tmp_path / "synthetic" / "b_30" / "native")
    assert "**read-only**" not in (native.instruction_override() or "")
    shutil.rmtree(tmp_path / "synthetic")


def test_vendored_code_is_pinned() -> None:
    from real2simlab.vendor import FOURDCODEBENCH

    assert len(FOURDCODEBENCH.commit) == 40 and "not redistributed" in FOURDCODEBENCH.license
    assert np.isfinite(1.0)


@pytest.mark.parametrize("missing", [False, True], ids=["no network", "not in the dataset"])
def test_a_real_case_that_cannot_be_downloaded_says_why(
    missing: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only a video the dataset does not hold is "not redistributed, choose another case"; a
    download that failed for lack of network is told as one, with the network's fix."""
    from types import SimpleNamespace

    import real2simlab.benchmarks.fourdcodebench as fourdcodebench
    import real2simlab.cli.tasks as tasks_cli
    from real2simlab.cli.main import main

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))

    def download(*args: object, **kwargs: object) -> str:
        if missing:  # what huggingface_hub raises for a file the repository does not hold
            raise pytest.importorskip("huggingface_hub.errors").EntryNotFoundError("404")
        try:
            raise ConnectionRefusedError(111, "Connection refused")
        except ConnectionRefusedError as error:
            raise ValueError("Force download failed due to the above error.") from error

    monkeypatch.setattr(fourdcodebench, "_hub", lambda: SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(tasks_cli, "_list", lambda args: fourdcodebench.download_real_case("c"))
    assert main(["task", "list"]) == 1  # as a command that builds the case says it
    err = capsys.readouterr().err
    assert ("not redistributed" in err) is missing and "\n  fix: " in err, err
    assert missing or "a download failed: [Errno 111] Connection refused" in err, err


def test_a_bug_while_a_missing_video_is_handled_is_not_called_not_redistributed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A mistake in the code (a ``KeyError``) raised while huggingface_hub's "no such file" is
    being handled is reported as the bug it is, not as a video the dataset does not hold."""
    from types import SimpleNamespace

    import real2simlab.benchmarks.fourdcodebench as fourdcodebench
    import real2simlab.cli.tasks as tasks_cli
    from real2simlab.cli.main import main

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    missing = pytest.importorskip("huggingface_hub.errors").EntryNotFoundError

    def download(*args: object, **kwargs: object) -> str:
        try:
            raise missing("404")
        except missing:
            raise KeyError("videos")  # noqa: B904 (no `from`: the 404 is its context)

    monkeypatch.setattr(fourdcodebench, "_hub", lambda: SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(tasks_cli, "_list", lambda args: fourdcodebench.download_real_case("c"))
    assert main(["task", "list"]) == 1
    err = capsys.readouterr().err
    assert "bug in Real2SimLab" in err and "not redistributed" not in err, err


@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.skipif(
    not (CACHE_CASE / "world" / "camera.json").is_file(), reason="4DCodeBench B_30 not downloaded"
)
def test_official_scorer_sanity(tmp_path: Path, docker_backend: Any) -> None:
    """The official scorer, through the plugin, gives a reference world full marks against itself."""
    from real2simlab.benchmarks.fourdcodebench import build_case
    from real2simlab.benchmarks.fourdcodebench.scorer import FourDCodeBenchScorer

    from .conftest import docker_image_ready

    if not docker_image_ready("4dcb/scorer:dev"):
        pytest.skip("the official scorer image 4dcb/scorer:dev is not built")
    task = load_task(build_case("B_30", "synthetic", "native", tmp_path / "tasks"))
    solution = tmp_path / "solution"
    solution.mkdir()
    (solution / "build.sh").write_text("#!/bin/bash\n")
    scorer = FourDCodeBenchScorer(checkpoints=None)
    result = scorer.score(
        task, delivery_dir=solution, output_dir=task.reference_dir(),
        work_dir=tmp_path / "work", options={"kind": "synthetic", "case": "B_30"},
    )  # fmt: skip
    assert result["status"] == "scored", result
    reward = result["reward"]
    model_free = ("dynamic_iou", "scene_3d", "trajectory_dtw", "emd_step", "occupancy_dtw",
                  "interpenetration")  # fmt: skip
    for key in model_free:
        assert reward[key] == pytest.approx(1.0), key
    assert "uni3d" in result["types_left_out"]
    no_program = scorer.score(
        task, delivery_dir=tmp_path, output_dir=task.reference_dir(),
        work_dir=tmp_path / "work2", options={"kind": "synthetic", "case": "B_30"},
    )  # fmt: skip
    assert no_program["status"] == "no_program"


@pytest.mark.parametrize(
    ("failure", "fix"),
    [
        ("import h5py_missing_for_the_test", "pip install h5py hdf5plugin"),
        ("raise OSError('truncated file')", "may be cut or damaged"),
    ],
)
def test_a_world_that_cannot_be_unpacked_gets_the_fix_for_its_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, fix: str
) -> None:
    """A missing dependency of the official unpacker is fixed by installing it; a cut download
    is not (it is downloaded again)."""
    from types import SimpleNamespace

    import real2simlab.benchmarks.fourdcodebench as fourdcodebench
    from real2simlab.errors import SetupError

    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "scripts" / "download_data.py").write_text(
        f"def unpack(packed, world):\n    {failure}\n"
    )
    monkeypatch.setattr(fourdcodebench, "FOURDCODEBENCH", SimpleNamespace(ensure=lambda: checkout))
    with pytest.raises(SetupError) as error:
        fourdcodebench._unpack(tmp_path / "B_30.h5", tmp_path / "world")  # pyright: ignore[reportPrivateUsage]
    assert fix in (error.value.fix or "")
