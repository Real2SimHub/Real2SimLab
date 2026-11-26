"""Failures injected where they happen in real use: the machine fails (a sandbox that does not
start, a host that cannot copy, an official scorer that crashes), what was cached changes (a
frozen program edited, an image tag moved, a task replaced by a new version), and processes
are interrupted. Each test checks what the Lab reports, what resuming does, and what reaches
training data: a failure of the machine is never a score, and a cached step is reused only
while what it used is unchanged."""

from __future__ import annotations

import errno
import importlib.util
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

from real2simlab._fs import recover_dir, replace_dir, staging_dir
from real2simlab.agents.fake import FakeAgent
from real2simlab.benchmarks.synthetic import SyntheticBenchmark
from real2simlab.errors import InfrastructureError, SetupError
from real2simlab.methods import get_method
from real2simlab.orchestrate import Experiment, parse_experiment
from real2simlab.orchestrate.layout import RunLayout, incomplete
from real2simlab.sandbox.base import ExecResult, ReadinessProblem, ResolvedImage, SandboxSpec
from real2simlab.sandbox.bwrap import BwrapBackend
from real2simlab.sandbox.local import LocalBackend
from real2simlab.scoring.model import Headline
from real2simlab.task import TaskConfig
from real2simlab.task.io import read_task_config, write_task_config
from real2simlab.train import export_dataset, score_solution


class FlakyBackend(LocalBackend):
    """The local backend, failing (infrastructure error) the runs ``fail`` selects, and
    reporting ``content_id`` as the image's content."""

    def __init__(self, fail: Callable[[SandboxSpec], bool] = lambda spec: False) -> None:
        super().__init__(quiet=True)
        self.fail = fail
        self.content_id = "sha256:one"
        self.started: list[str] = []

    def resolve_image(self, reference: str) -> ResolvedImage:
        return ResolvedImage(reference=reference, digest=None, content_id=self.content_id)

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        if self.fail(spec):
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("the container did not start\n")
            return ExecResult(None, False, 0.0, "injected: the container did not start")
        self.started.append(spec.label)
        return super().run(spec, log_path)


def twin_runs(spec: SandboxSpec) -> bool:
    return spec.label == "r2s-twin"


def agent_runs(spec: SandboxSpec) -> bool:
    return spec.label.startswith("r2s-agent")


def _config(tasks: list[Path], methods: list[dict[str, Any]], **run: Any) -> Any:
    return parse_experiment(
        {
            "name": "faults",
            "tasks": {"paths": [str(t) for t in tasks]},
            "methods": methods,
            "run": {"backend": "local", "retries": 0, **run},
        }
    )


def _run(config: Any, out: Path, backend: LocalBackend | None = None) -> tuple[Any, Experiment]:
    experiment = Experiment(
        config, out, backend=backend or FlakyBackend(), log=lambda line: None, report=False
    )
    return experiment.run(), experiment


@pytest.fixture(scope="module")
def train_task(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("train")
    (path,) = SyntheticBenchmark().build(
        out, select=["bouncing_ball"], tiers=["calibrated"], split="train", count=1, seed=4
    )
    return path


def _copy(task: Path, target: Path) -> Path:
    shutil.copytree(task, target)
    return target


# --- 01 a frozen program that changed ------------------------------------------------------


@pytest.mark.slow
def test_a_frozen_program_edited_in_the_run_directory_is_rebuilt_not_rerun(
    task_copy: Path, tmp_path: Path
) -> None:
    config = _config([task_copy], [{"name": "oracle"}])
    first, _ = _run(config, tmp_path / "run")
    assert first.methods[0].reproduction_skill.mean == pytest.approx(1.0)
    (program,) = (tmp_path / "run" / "builds").rglob("twin/program/run.py")
    original = program.read_text()
    program.write_text("raise SystemExit(3)\n")  # synced, edited or replaced by mistake
    again, experiment = _run(config, tmp_path / "run")
    assert experiment.progress.builds == 1 and experiment.progress.reused_builds == 0
    assert again.methods[0].reproduction_skill.mean == pytest.approx(1.0)
    assert program.read_text() == original  # the program that was scored is back


@pytest.mark.slow
def test_a_reward_is_never_paired_with_a_program_it_did_not_score(
    train_task: Path, tmp_path: Path
) -> None:
    """The error names the program, not the task, and its fix works: the same command again on
    the run directory builds and scores again what changed."""
    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    config = _config([tasks], [{"name": "oracle", "label": "mine"}])
    _run(config, tmp_path / "run")
    (program,) = (tmp_path / "run" / "builds").rglob("twin/program/run.py")
    program.write_text("raise SystemExit(3)\n")
    with pytest.raises(SetupError, match="changed after it was scored") as raised:
        export_dataset(tmp_path / "data", tasks=[tasks], runs=[tmp_path / "run"])
    assert "cannot be matched with the program they scored" in str(raised.value)
    assert "run the same r2s eval or r2s run command again" in (raised.value.fix or "")
    summary = export_dataset(
        tmp_path / "data", tasks=[tasks], runs=[tmp_path / "run"], skip_unmatched=True
    )
    assert summary.runs == 0 and any("changed after it was scored" in s for s in summary.skipped)
    _run(config, tmp_path / "run")
    assert export_dataset(tmp_path / "data2", tasks=[tasks], runs=[tmp_path / "run"]).runs == 1


@pytest.mark.parametrize("fault", ["a symbolic link", "an unreadable file"])
def test_a_program_that_cannot_be_delivered_fails_its_build_not_the_experiment(
    synthetic_tasks: Path, tmp_path: Path, fault: str
) -> None:
    """``program:path=`` a directory of per-task programs, one of which holds a symbolic link
    (``main.py -> run.py``) or a file it cannot read: that task's build delivers nothing, with
    the reason and a fix, as the program's reward is 0; the other task and the other method
    are evaluated, and nothing stops the experiment."""
    if fault == "an unreadable file" and os.geteuid() == 0:
        pytest.skip("root reads every file")
    tasks = [synthetic_tasks / name / "calibrated" for name in ("bouncing_ball", "pendulum")]
    programs = tmp_path / "programs"
    for task in tasks:
        shutil.copytree(task / "hidden" / "generator" / "program", programs / task.parent.name)
    broken = programs / "pendulum"
    if fault == "a symbolic link":
        (broken / "main.py").symlink_to("run.py")
    else:
        (broken / "notes.txt").write_text("notes\n")
        (broken / "notes.txt").chmod(0)
    reward = score_solution(tasks[1], broken, backend=LocalBackend(quiet=True))
    assert reward.reward == 0.0 and not reward.result.delivered
    config = _config(tasks, [{"name": "oracle"}, {"name": "program", "path": str(programs)}])
    results, _ = _run(config, tmp_path / "run", backend=LocalBackend(quiet=True))
    rows = {(row["task"].split("/")[1], row["method"]): row for row in results.rows}
    assert len(rows) == 4 and rows[("pendulum", "oracle")]["delivered"]
    assert rows[("bouncing_ball", "program")]["reproduction_skill"] > 0.99
    failure = rows[("pendulum", "program")]["failure"]
    said = "symbolic links are not allowed" if fault == "a symbolic link" else "cannot be read"
    assert said in failure and "fix:" in failure and said in str(reward.result.delivery_failure)


@pytest.mark.parametrize("step", ["copied", "scored"])
def test_a_solution_edited_while_it_is_scored_gets_no_reward(
    bouncing_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    step: str,
) -> None:
    """A solution edited while its copy was made (the copy may mix two versions), or while the
    copy ran (the reward would be paired with a program it is not the reward of), gets no
    reward: an error with its fix, as for a task edited meanwhile, not the old program's
    reward under the solution's name."""
    from real2simlab.cli.main import main
    from real2simlab.train import reward as reward_module

    solution = tmp_path / "solution"
    shutil.copytree(bouncing_task / "hidden" / "generator" / "program", solution)
    name = "clean_solution" if step == "copied" else "run_twin"
    original = getattr(reward_module, name)

    def then_edit(*args: Any, **kwargs: Any) -> Any:
        done = original(*args, **kwargs)
        (solution / "run.py").write_text("raise RuntimeError('edited')\n")
        return done

    monkeypatch.setattr(reward_module, name, then_edit)
    with pytest.raises(SetupError, match=f"changed while it was {step}") as raised:
        score_solution(bouncing_task, solution, backend=LocalBackend(quiet=True))
    assert "no reward" in str(raised.value) and raised.value.fix
    shutil.rmtree(solution)
    shutil.copytree(bouncing_task / "hidden" / "generator" / "program", solution)
    capsys.readouterr()
    argv = ["train", "reward", "--task", str(bouncing_task), "--solution", str(solution),
            "--backend", "local"]  # fmt: skip
    assert main(argv) == 1
    captured = capsys.readouterr()
    assert "changed while it was" in captured.err and "fix:" in captured.err
    assert "reward" not in captured.out


def test_a_program_edited_while_it_is_copied_is_not_a_result(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``program:path=`` edited while its copy was made: the copy may mix two versions, so the
    build is not a result (``infra_error``), never scored, and built again; the program changed
    while the experiment ran, so the run is not complete either, and the same command builds
    and scores it again."""
    from real2simlab.methods import program as program_module

    solution = tmp_path / "solution"
    shutil.copytree(bouncing_task / "hidden" / "generator" / "program", solution)
    original = program_module.clean_solution

    def then_edit(source: Path, target: Path) -> Path:
        copied = original(source, target)
        with (solution / "run.py").open("a") as stream:
            stream.write("# edited\n")
        return copied

    monkeypatch.setattr(program_module, "clean_solution", then_edit)
    config = _config([bouncing_task], [{"name": "program", "path": str(solution)}])
    with pytest.raises(SetupError, match="changed while the experiment ran"):
        _run(config, tmp_path / "run", backend=LocalBackend(quiet=True))
    (record,) = (tmp_path / "run" / "builds").rglob("trial-0/build_record.json")
    built = json.loads(record.read_text())
    assert "changed while it was copied" in (built.get("infra_error") or "")
    assert not built["delivered"]
    monkeypatch.undo()  # no longer edited: the same command builds it again, and scores it
    results, experiment = _run(config, tmp_path / "run", backend=LocalBackend(quiet=True))
    assert experiment.progress.builds == 1 and experiment.progress.reused_builds == 0
    assert results.methods[0].reproduction_skill.mean > 0.99


def test_a_program_reached_through_a_symbolic_link_is_never_delivered(
    synthetic_tasks: Path, tmp_path: Path
) -> None:
    """``program:path=`` whose per-task directory is a symbolic link to a program elsewhere, or
    a twin package whose ``program/`` is one: what is behind the link is never delivered (the
    build's key does not hold it, and a reward refuses it), so that task delivers nothing,
    with the fix, as its reward is 0; the other task is evaluated."""
    tasks = [synthetic_tasks / name / "calibrated" for name in ("bouncing_ball", "pendulum")]
    elsewhere = tmp_path / "elsewhere" / "pendulum"
    shutil.copytree(tasks[1] / "hidden" / "generator" / "program", elsewhere)
    programs = tmp_path / "programs"
    shutil.copytree(tasks[0] / "hidden" / "generator" / "program", programs / "bouncing_ball")
    (programs / "pendulum").symlink_to(elsewhere, target_is_directory=True)
    config = _config(tasks, [{"name": "program", "path": str(programs)}])
    results, _ = _run(config, tmp_path / "run", backend=LocalBackend(quiet=True))
    rows = {row["task"].split("/")[1]: row for row in results.rows}
    assert rows["bouncing_ball"]["reproduction_skill"] > 0.99
    failure = rows["pendulum"]["failure"] or ""
    assert not rows["pendulum"]["delivered"], rows["pendulum"]
    assert "pendulum/ is a symbolic link" in failure and "fix:" in failure, failure
    package = tmp_path / "package"
    package.mkdir()
    (package / "twin.json").write_text("{}\n")
    (package / "program").symlink_to(elsewhere, target_is_directory=True)
    reward = score_solution(tasks[1], package, backend=LocalBackend(quiet=True))
    assert reward.reward == 0.0 and not reward.result.delivered
    config = _config([tasks[1]], [{"name": "program", "path": str(package)}])
    results, _ = _run(config, tmp_path / "run-package", backend=LocalBackend(quiet=True))
    (row,) = results.rows
    assert not row["delivered"] and "program/ is a symbolic link" in (row["failure"] or ""), row


def test_a_solution_digest_sees_every_file_its_copy_holds(tmp_path: Path) -> None:
    """What ``clean_solution`` copies is what the digest of a solution covers: a file in a
    directory named like a file manager's (``._data/``) is the program's, and an edit of it
    is an edit; a file manager's own file (``.DS_Store``) is not."""
    from real2simlab.agents.base import solution_digest

    solution = tmp_path / "solution"
    (solution / "._data").mkdir(parents=True)
    (solution / "run.py").write_text("print('twin')\n")
    (solution / "._data" / "params.json").write_text("{}\n")
    before = solution_digest(solution)
    (solution / "._data" / "params.json").write_text('{"g": 9.81}\n')
    edited = solution_digest(solution)
    assert edited != before
    (solution / ".DS_Store").write_bytes(b"opened in Finder")
    assert solution_digest(solution) == edited


# --- 02 an image tag that moved ----------------------------------------------------------------


@pytest.mark.slow
def test_a_tag_moved_to_another_image_is_run_again(task_copy: Path, tmp_path: Path) -> None:
    config = _config([task_copy], [{"name": "oracle"}])
    backend = FlakyBackend()
    _run(config, tmp_path / "run", backend)
    _, experiment = _run(config, tmp_path / "run", backend)
    assert (experiment.progress.runs, experiment.progress.reused_runs) == (0, 1)
    backend.content_id = "sha256:two"  # the same tag, another image
    _, experiment = _run(config, tmp_path / "run", backend)
    assert (experiment.progress.runs, experiment.progress.reused_runs) == (1, 0)


@pytest.mark.slow
def test_an_agent_image_rebuilt_under_the_same_tag_redoes_the_build(
    train_task: Path, tmp_path: Path
) -> None:
    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    config = _config([tasks], [{"name": "fake_agent", "mode": "physical"}])
    backend = FlakyBackend()
    _run(config, tmp_path / "run", backend)
    _, experiment = _run(config, tmp_path / "run", backend)
    assert (experiment.progress.builds, experiment.progress.reused_builds) == (0, 1)
    backend.content_id = "sha256:two"  # the agent's environment, rebuilt
    _, experiment = _run(config, tmp_path / "run", backend)
    assert (experiment.progress.builds, experiment.progress.reused_builds) == (1, 0)


SWITCHED_IMAGE = """
import os
from pathlib import Path

import numpy as np

if os.environ.get("R2S_TEST_SWITCHED"):  # what it computes on the other image: no motion
    path = Path(os.environ["R2S_OUTPUT_DIR"]) / "tracks.npz"
    with np.load(path) as stored:
        arrays = {key: stored[key] for key in stored.files}
    for key, value in arrays.items():
        if key.endswith(".position"):
            arrays[key] = np.repeat(value[:1], len(value), axis=0)
    np.savez(path, **arrays)
"""


@pytest.mark.slow
def test_an_unpacked_image_changed_between_seeds_is_not_scored_as_the_first(
    task_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audit's case on bwrap: once seed 0 has run, the unpacked image is changed by hand
    (another image's record and files). Seed 1 then ran on it, its 0 was filed under the first
    image's id, and reused once that image was back (0.500 twice, 1.000 afresh). Now nothing
    runs on the other image (seed 1 and its retry are infrastructure errors), and the next
    command runs seed 1 on the image that is there."""
    from dataclasses import replace

    from real2simlab.envs.rootfs import unpack_tarball
    from real2simlab.sandbox import bwrap as bwrap_module
    from real2simlab.sandbox.bwrap import BwrapBackend

    from .test_bwrap import IMAGE, layer, saved_image

    class HostBwrap(BwrapBackend):
        """The bwrap backend (its unpacked image resolved once, checked around every run),
        with each program run on the host by the local backend in place of bubblewrap, which
        cannot run everywhere tests do (``on_the_host`` stands in for ``run_logged``). The
        program is told (``R2S_TEST_SWITCHED``) when the image's ``/etc/os-release`` is not
        the first one, as a program reading that file would find out."""

        def __init__(self) -> None:
            super().__init__(str(bwrap), cgroup_controllers=frozenset())
            self.spec: SandboxSpec | None = None
            self.after_first_run: Callable[[], None] | None = None

        def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
            self.spec = spec
            result = super().run(spec, log_path)
            if spec.label == "r2s-twin" and self.after_first_run is not None:
                self.after_first_run()
                self.after_first_run = None
            return result

        def on_the_host(
            self, command: list[str], log_path: Path, *_: Any, **__: Any
        ) -> tuple[int | None, bool, float]:
            assert self.spec is not None
            switched = "SWITCHED" in os_release.read_text()
            env = {**self.spec.env, **({"R2S_TEST_SWITCHED": "1"} if switched else {})}
            result = LocalBackend(quiet=True).run(replace(self.spec, env=env), log_path)
            return result.exit_code, result.timed_out, result.duration_s

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    files = layer([
        ("usr", "dir"), ("usr/bin", "dir"), ("usr/bin/python3", b""),
        ("etc", "dir"), ("etc/os-release", b"ID=first\n"),
    ])  # fmt: skip
    image = saved_image(tmp_path / "image.tar", [files], ["PATH=/usr/bin"])
    unpacked = unpack_tarball(image, IMAGE)
    record, os_release = unpacked.root / "image.json", unpacked.rootfs / "etc" / "os-release"
    first_image = record.read_bytes(), os_release.read_bytes()

    def switch() -> None:
        meta = json.loads(record.read_text())
        record.write_text(json.dumps({**meta, "image_id": "sha256:" + "c" * 64}))
        os_release.write_text("ID=second\nSWITCHED\n")

    bwrap = tmp_path / "bwrap"
    bwrap.write_text("#!/bin/sh\necho 'bubblewrap 0.9.0'\n")
    bwrap.chmod(0o755)
    program = _truth_program(task_copy, tmp_path / "program", SWITCHED_IMAGE)
    method = {"name": "program", "path": str(program)}
    config = _config([task_copy], [method], seeds=[0, 1], retries=1)

    def run(backend: HostBwrap) -> tuple[Any, Experiment]:
        monkeypatch.setattr(bwrap_module, "run_logged", backend.on_the_host)
        experiment = Experiment(
            config, tmp_path / "run", backend=backend, log=lambda line: None, report=False
        )
        return experiment.run(), experiment

    backend = HostBwrap()
    backend.after_first_run = switch
    results, _ = run(backend)
    (scored,) = results.methods
    assert scored.reproduction_skill.mean == pytest.approx(1.0), "a score of the other image"
    assert scored.infra_errors == 1
    (seed_1,) = (tmp_path / "run" / "runs").rglob("seed-1/run/run_record.json")
    run_record = json.loads(seed_1.read_text())
    assert run_record["status"] == "infra_error" and run_record["attempt"] == 2
    assert "changed after this command resolved it" in run_record["problems"][0]
    record.write_bytes(first_image[0])  # the first image is back
    os_release.write_bytes(first_image[1])
    results, experiment = run(HostBwrap())
    assert (experiment.progress.runs, experiment.progress.reused_runs) == (1, 1)
    assert results.methods[0].reproduction_skill.mean == pytest.approx(1.0)
    assert results.methods[0].infra_errors == 0


# --- 03 and 12 a sandbox that does not start ---------------------------------------------------


@pytest.mark.slow
def test_a_sandbox_that_did_not_start_is_not_a_score(task_copy: Path, tmp_path: Path) -> None:
    config = _config([task_copy], [{"name": "oracle"}])
    broken, _ = _run(config, tmp_path / "run", FlakyBackend(twin_runs))
    (method,) = broken.methods
    (card,) = method.benchmarks
    assert (card.scored, card.results, card.infra_errors) == (0, 1, 1)
    assert card.value.mean is None and method.reproduction_skill.mean is None
    assert method.infra_errors == 1
    (row,) = broken.to_json()["rows"]
    assert row["infra_error"] and row["reproduction_skill"] is None
    fixed, _ = _run(config, tmp_path / "run")  # the machine works again: the real score
    assert fixed.methods[0].reproduction_skill.mean == pytest.approx(1.0)
    assert fixed.methods[0].infra_errors == 0


@pytest.mark.slow
def test_a_second_run_that_did_not_start_is_not_a_reproduction_failure(
    task_copy: Path, tmp_path: Path
) -> None:
    calls: list[int] = []

    def second_twin_run(spec: SandboxSpec) -> bool:
        if spec.label != "r2s-twin":
            return False
        calls.append(1)
        return len(calls) == 2

    config = _config([task_copy], [{"name": "oracle"}], check_reproducibility=True)
    results, _ = _run(config, tmp_path / "run", FlakyBackend(second_twin_run))
    (card,) = results.methods[0].benchmarks
    assert card.value.mean == pytest.approx(1.0)
    assert (card.rerun_compared, card.rerun_agreed) == (0, 0)
    again, _ = _run(config, tmp_path / "run")  # both runs now: compared and agreeing
    (card,) = again.methods[0].benchmarks
    assert (card.rerun_compared, card.rerun_agreed) == (1, 1)


# --- 04 an agent whose sandbox did not start ----------------------------------------------------


@pytest.mark.slow
def test_an_agent_build_that_failed_outside_the_agent_is_redone(
    train_task: Path, tmp_path: Path
) -> None:
    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    config = _config([tasks], [{"name": "fake_agent", "mode": "physical"}])
    broken, _ = _run(config, tmp_path / "run", FlakyBackend(agent_runs))
    assert broken.methods[0].infra_errors == 1
    (record,) = (tmp_path / "run" / "builds").rglob("build_record.json")
    assert json.loads(record.read_text())["infra_error"]
    with pytest.raises(SetupError, match="nothing could be exported"):
        export_dataset(tmp_path / "data", runs=[tmp_path / "run"], include_undelivered=True)
    fixed, experiment = _run(config, tmp_path / "run")
    assert experiment.progress.builds == 1 and fixed.methods[0].delivered_rate == 1.0
    assert json.loads((tmp_path / "run" / "state" / "budget.json").read_text())["agent_runs"] == 1


CANNOT_MOUNT_PROC = "bwrap: Can't mount proc on /newroot/proc: Operation not permitted\n"


class _SandboxCannotStart(FlakyBackend):
    """bwrap on a host that creates namespaces but allows no new procfs (a container): every
    sandbox of the agent fails while bwrap sets it up, before anything runs in it, and bwrap
    exits 1 with its own error, as an agent that fails does. ``broken = False``: repaired."""

    broken = True

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        if self.broken and agent_runs(spec):
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(log_path).open("a", encoding="utf-8") as log:
                log.write(CANNOT_MOUNT_PROC)
            return ExecResult(exit_code=1, timed_out=False, duration_s=0.01)
        return super().run(spec, log_path)


def _agent_context(task_dir: Path, root: Path, backend: Any) -> Any:
    from real2simlab.methods.base import BuildContext
    from real2simlab.task import load_task
    from real2simlab.task.instruction import task_statement

    task = load_task(task_dir)
    return BuildContext(
        task_id=task.id, task_hash=task.content_hash(), config=task.config,
        statement=task_statement(task), visible_dir=task.visible_dir,
        work_dir=root / "work", twin_dir=root / "twin", backend=backend,
    )  # fmt: skip


def _agent_record(root: Path) -> dict[str, Any]:
    return json.loads((root / "work" / "agent" / "agent_record.json").read_text())


@pytest.mark.slow
def test_an_agent_whose_sandbox_cannot_start_is_not_a_score(
    train_task: Path, tmp_path: Path
) -> None:
    """bwrap that cannot set up the agent's sandbox exits 1 before the agent starts, as an agent
    that fails does: the build is an infrastructure error (not a 0, never exported, the agent
    run given back), and the same command builds it again once the machine works."""
    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    config = _config([tasks], [{"name": "fake_agent", "mode": "physical"}])
    backend = _SandboxCannotStart()
    broken, _ = _run(config, tmp_path / "run", backend)
    (card,) = broken.methods[0].benchmarks
    assert (card.scored, card.results, card.infra_errors) == (0, 1, 1)
    (path,) = (tmp_path / "run" / "builds").rglob("build_record.json")
    record = json.loads(path.read_text())
    assert "Can't mount proc" in (record["infra_error"] or "") and not record["delivered"]
    agent = json.loads((path.parent / record["agent_record"]).read_text())
    assert agent["exit_status"] == "infra_error"
    budget = tmp_path / "run" / "state" / "budget.json"
    assert json.loads(budget.read_text())["agent_runs"] == 0  # the agent never ran
    with pytest.raises(SetupError, match="nothing could be exported"):
        export_dataset(tmp_path / "data", runs=[tmp_path / "run"], include_undelivered=True)
    backend.broken = False  # the machine is repaired: the same command builds it again
    fixed, experiment = _run(config, tmp_path / "run", backend)
    assert (experiment.progress.builds, experiment.progress.reused_builds) == (1, 0)
    assert fixed.methods[0].delivered_rate == 1.0 and fixed.methods[0].infra_errors == 0
    assert json.loads(budget.read_text())["agent_runs"] == 1


def test_a_budget_counted_again_leaves_out_agents_that_never_ran(
    train_task: Path, tmp_path: Path
) -> None:
    """A lost budget record is counted again from the run's agent records: an agent whose
    sandbox failed was given back when it failed, and is not counted as a run spent."""
    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    config = _config([tasks], [{"name": "fake_agent", "mode": "physical"}])
    _run(config, tmp_path / "run", FlakyBackend(agent_runs))
    budget = tmp_path / "run" / "state" / "budget.json"
    assert json.loads(budget.read_text())["agent_runs"] == 0
    budget.unlink()  # deleted by hand, or a backup restored without it
    _run(config, tmp_path / "run", FlakyBackend(agent_runs))
    assert json.loads(budget.read_text())["agent_runs"] == 0


class _AgentsTakeAMinute(FlakyBackend):
    """Each agent run takes a minute of ``clock``, the experiment's ``time.monotonic``."""

    def __init__(self, clock: list[float]) -> None:
        super().__init__()
        self.clock = clock

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        if spec.label == "r2s-agent-fake_agent":
            self.clock[0] += 60.0
        return super().run(spec, log_path)


def test_the_wall_time_budget_counts_every_command_on_the_run(
    task_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``max_wall_time_s`` works as ``max_agent_runs`` does, across the commands on one run
    directory: once the time is spent, the same command run again (or resumed after Ctrl-C)
    starts no agent. Each command started a new budget, so N commands spent N budgets (A3-10).
    """
    from types import SimpleNamespace

    from real2simlab.orchestrate import experiment as module

    clock = [1000.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    config = parse_experiment(
        {
            "name": "faults",
            "tasks": {"paths": [str(task_copy)]},
            "methods": [{"name": "fake_agent", "mode": "physical"}],
            "run": {"backend": "local", "retries": 0, "trials": 3, "agent_concurrency": 1},
            "budget": {"max_wall_time_s": 100},
        }
    )
    for command in range(3):
        _, experiment = _run(config, tmp_path / "run", _AgentsTakeAMinute(clock))
        assert experiment.progress.builds == (2 if command == 0 else 0), command
        skipped = [s for s in experiment.progress.skipped if "wall-time budget is spent" in s]
        assert len(skipped) == 1, (command, experiment.progress.skipped)
    budget = json.loads((tmp_path / "run" / "state" / "budget.json").read_text())
    assert budget["agent_runs"] == 2 and budget["wall_time_s"] >= 120.0


class _ExitingAgent(FakeAgent):
    """An agent whose CLI starts and then fails on its own."""

    def command(self, context: Any, workspace: Any) -> list[str]:
        return ["python", "-c", 'import sys; print(\'{"type": "error"}\'); sys.exit(1)']


class _MissingCli(FakeAgent):
    """An agent whose CLI is not in its image (a broken build, an ``image`` without it)."""

    def command(self, context: Any, workspace: Any) -> list[str]:
        return ["r2s-no-such-agent-cli", "--json"]


class _BootstrapNeverRuns(LocalBackend):
    """The agent's sandbox exits 1 before its bootstrap writes anything, while a sandbox like
    it starts Python normally: whatever stopped it, the sandbox itself works."""

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        if spec.label == "r2s-agent-fake_agent":
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("Killed\n")
            return ExecResult(exit_code=1, timed_out=False, duration_s=0.01)
        return super().run(spec, log_path)


class _KilledFromOutside(LocalBackend):
    """An isolating backend whose sandbox process (bwrap, the Docker client) ends with
    ``code``: negative when it was killed by a signal from outside the sandbox, 128 + N when
    bwrap reports the agent's own process killed inside it (its memory limit). ``began``: the
    bootstrap had started the agent by then."""

    isolated: ClassVar[bool] = True
    code = -9
    began = True

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        logs = next(m.source for m in spec.mounts if m.target == "/r2s/logs")
        for name in ("trajectory.jsonl", "stderr.log") if self.began else ():
            (Path(logs) / name).write_text("")
        Path(log_path).write_text("")
        return ExecResult(exit_code=self.code, timed_out=False, duration_s=1.0)


def test_an_agent_that_fails_on_its_own_is_its_result(bouncing_task: Path, tmp_path: Path) -> None:
    """The other side: an agent whose sandbox started and that failed on its own is a result,
    scored as such, and its sandbox is not probed; nor is one whose sandbox starts Python."""
    outcome = _ExitingAgent(mode="nothing").build(
        _agent_context(bouncing_task, tmp_path / "a", LocalBackend(quiet=True))
    )
    assert outcome.twin is None and outcome.infra_error is None
    assert _agent_record(tmp_path / "a")["exit_status"] == "failed"
    assert not (tmp_path / "a" / "work" / "agent" / "sandbox-probe.log").exists()
    stopped = get_method("fake_agent", mode="physical").build(
        _agent_context(bouncing_task, tmp_path / "b", _BootstrapNeverRuns(quiet=True))
    )
    assert stopped.twin is None and stopped.infra_error is None
    assert _agent_record(tmp_path / "b")["exit_status"] == "failed"
    inside = _KilledFromOutside(quiet=True)
    inside.code = 128 + 9
    killed = get_method("fake_agent", mode="physical").build(
        _agent_context(bouncing_task, tmp_path / "c", inside)
    )
    assert killed.infra_error is None and _agent_record(tmp_path / "c")["exit_status"] == "failed"


def test_an_agent_whose_cli_cannot_start_is_not_a_score(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """Nothing of an agent whose CLI the bootstrap cannot start ran: not the agent's result."""
    outcome = _MissingCli(mode="nothing").build(
        _agent_context(bouncing_task, tmp_path, LocalBackend(quiet=True))
    )
    assert outcome.twin is None and "r2s-no-such-agent-cli" in (outcome.infra_error or "")
    assert _agent_record(tmp_path)["exit_status"] == "infra_error"


@pytest.mark.parametrize(("began", "status"), [(True, "failed"), (False, "infra_error")])
def test_an_agent_sandbox_killed_from_outside_is_not_a_score(
    bouncing_task: Path, tmp_path: Path, began: bool, status: str
) -> None:
    """bwrap or the Docker client killed by a signal from outside the sandbox (an
    administrator, the host's out-of-memory killer) stops the agent: not its result. The agent's
    exit status says whether it had started (its run was then spent: ``failed``) or nothing of
    it ran (``infra_error``)."""
    backend = _KilledFromOutside(quiet=True)
    backend.began = began
    outcome = get_method("fake_agent", mode="physical").build(
        _agent_context(bouncing_task, tmp_path, backend)
    )
    assert outcome.twin is None and "signal 9" in (outcome.infra_error or "")
    assert _agent_record(tmp_path)["exit_status"] == status


class _KilledOnceStarted(FlakyBackend):
    """An isolating sandbox (bwrap, the Docker client) that something outside it kills (an
    administrator, the host's out-of-memory killer) once the agent's bootstrap started the
    agent: it ran, and may have spent money on its model. Other sandboxes run normally."""

    isolated: ClassVar[bool] = True

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        if not agent_runs(spec):
            return super().run(spec, log_path)
        self.started.append(spec.label)
        logs = next(m.source for m in spec.mounts if m.target == "/r2s/logs")
        for name in ("trajectory.jsonl", "stderr.log"):
            (Path(logs) / name).write_text("")
        Path(log_path).write_text("")
        return ExecResult(exit_code=-9, timed_out=False, duration_s=1.0)


@pytest.mark.slow
def test_an_agent_stopped_after_it_started_is_spent_and_built_again_by_the_next_run(
    train_task: Path, tmp_path: Path
) -> None:
    """An agent whose sandbox was killed from outside after it had started is not a result, but
    it ran (and may have paid for its model): the run counts against the budget, also when the
    budget is counted again from the run's records, and the same command does not run it again
    at once; its next run builds it again. An agent of which nothing ran is retried at once and
    given back."""
    tasks = _copy(train_task, tmp_path / "tasks" / "t")

    def config(**budget: Any) -> Any:
        return parse_experiment(
            {
                "name": "faults",
                "tasks": {"paths": [str(tasks)]},
                "methods": [{"name": "fake_agent", "mode": "physical"}],
                "run": {"backend": "local", "retries": 2},
                "budget": budget,
            }
        )

    killed = _KilledOnceStarted()
    broken, _ = _run(config(), tmp_path / "run", killed)
    assert killed.started == ["r2s-agent-fake_agent"]  # once: not again by this command
    (card,) = broken.methods[0].benchmarks
    assert (card.scored, card.results, card.infra_errors) == (0, 1, 1)
    (path,) = (tmp_path / "run" / "builds").rglob("build_record.json")
    record = json.loads(path.read_text())
    assert "signal 9" in (record["infra_error"] or "") and not record["delivered"]
    assert json.loads((path.parent / record["agent_record"]).read_text())["exit_status"] == "failed"
    budget = tmp_path / "run" / "state" / "budget.json"
    assert json.loads(budget.read_text())["agent_runs"] == 1  # spent
    budget.unlink()  # counted again from the run's records: the agent that ran is in it
    again = _KilledOnceStarted()
    _, limited = _run(config(max_agent_runs=1), tmp_path / "run", again)
    assert not again.started
    assert any("agent-run budget" in step for step in limited.progress.skipped)
    assert json.loads(budget.read_text())["agent_runs"] == 1
    fixed, experiment = _run(config(), tmp_path / "run")  # the machine works: built again
    assert (experiment.progress.builds, experiment.progress.reused_builds) == (1, 0)
    assert fixed.methods[0].delivered_rate == 1.0 and fixed.methods[0].infra_errors == 0
    assert json.loads(budget.read_text())["agent_runs"] == 2
    tried: list[str] = []

    def never_starts(spec: SandboxSpec) -> bool:
        if agent_runs(spec):
            tried.append(spec.label)
        return agent_runs(spec)

    _run(config(), tmp_path / "never", FlakyBackend(never_starts))
    assert len(tried) == 3  # nothing of it ran: tried again at once (retries 2), given back
    assert json.loads((tmp_path / "never" / "state" / "budget.json").read_text())["agent_runs"] == 0


# --- 04b what agent runs spent is never taken as less than it was ---------------------------------


class _SaysNoCost(FakeAgent):
    """An agent whose runs can say what they cost, but these do not (``mini_swe`` with a model
    LiteLLM has no price for, a Claude Code run stopped before its result): what each cost is
    unknown."""

    reports_cost: ClassVar[bool | None] = True

    def cost(self, trajectory: Path) -> float | None:
        return None


def _budgeted(tasks: Path, trials: int = 1, **budget: Any) -> Any:
    return parse_experiment(
        {
            "name": "budget",
            "tasks": {"paths": [str(tasks)]},
            "methods": [{"name": "fake_agent", "mode": "physical"}],
            "run": {"backend": "local", "retries": 0, "trials": trials, "agent_concurrency": 1},
            "budget": budget,
        }
    )


def _agents_started(backend: FlakyBackend) -> list[str]:
    return [label for label in backend.started if label.startswith("r2s-agent")]


def _spent(run: Path) -> dict[str, Any]:
    """What ``state/budget.json`` says the run spent on agents: its ``wall_time_s`` (the
    wall-time budget's, which its own tests check) is left out."""
    spent = json.loads((run / "state" / "budget.json").read_text())
    spent.pop("wall_time_s", None)
    return spent


@pytest.mark.slow
@pytest.mark.parametrize("damage", ["broken", "cut", "a directory", "missing"])
def test_a_budget_counted_again_never_takes_an_agent_record_it_cannot_read_as_nothing(
    damage: str, train_task: Path, tmp_path: Path
) -> None:
    """A lost budget record is counted again from the run's agent records: one that is damaged,
    cut, not a regular file, or missing while its build record names it held what an agent run
    spent. It counts as a run of unknown cost, never as one that cost nothing or did not run:
    the agent-run budget counts it, a cost budget starts no further agent, and the Lab says
    which record it could not read."""
    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    run = tmp_path / "run"
    _run(_budgeted(tasks), run)
    assert _spent(run) == {"agent_runs": 1, "cost_usd": 0.0}  # the fake agent calls no model
    (record,) = run.glob("builds/*/*/trial-0/work/agent/agent_record.json")
    if damage == "broken":
        record.write_text("{broken")
    elif damage == "cut":
        record.write_bytes(record.read_bytes()[:40])
    else:
        record.unlink()  # its build record still names it
        if damage == "a directory":
            record.mkdir()
    (run / "state" / "budget.json").unlink()
    backend = FlakyBackend()
    said: list[str] = []
    config = _budgeted(tasks, trials=2, max_agent_runs=2, max_cost_usd=1.0)
    experiment = Experiment(config, run, backend=backend, log=said.append, report=False)
    experiment.run()
    assert _agents_started(backend) == []  # trial 1: what was spent may be more than $1
    (skipped,) = experiment.progress.skipped
    assert "the cost budget ($1) cannot be checked" in skipped, skipped
    assert _spent(run) == {"agent_runs": 1, "cost_usd": 0.0, "cost_unknown_runs": 1}
    (line,) = [line for line in said if "budget record was missing" in line]
    assert "1 of unknown cost" in line and f"agent record {record}" in line, line


@pytest.mark.parametrize("cost", [5.0, None], ids=["known", "unknown"])
def test_a_budget_counted_again_keeps_what_each_run_cost(
    cost: float | None, train_task: Path, tmp_path: Path
) -> None:
    """Counted again from the agent records, a run costs what its record says: $5 stays $5,
    and the cost budget stays spent; a run whose record does not say what it cost is one of
    unknown cost, never $0, so a cost budget starts no further agent either."""
    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    run = tmp_path / "run"
    _run(_budgeted(tasks), run)
    (record,) = run.glob("builds/*/*/trial-0/work/agent/agent_record.json")
    data = json.loads(record.read_text())
    data["cost_usd"] = cost  # as Claude Code or OpenCode record it, or a run that said nothing
    record.write_text(json.dumps(data))
    (run / "state" / "budget.json").unlink()
    backend = FlakyBackend()
    _, experiment = _run(_budgeted(tasks, trials=2, max_cost_usd=1.0), run, backend)
    assert _agents_started(backend) == []
    (skipped,) = experiment.progress.skipped
    if cost is not None:
        assert "the cost budget ($1) is spent" in skipped, skipped
        assert _spent(run) == {"agent_runs": 1, "cost_usd": 5.0}
    else:
        assert "the cost budget ($1) cannot be checked" in skipped, skipped
        assert _spent(run) == {"agent_runs": 1, "cost_usd": 0.0, "cost_unknown_runs": 1}
    assert "counted at least 1 agent run(s)" in (run / "r2s.log").read_text()


@pytest.mark.slow
def test_a_budget_counted_again_says_it_is_at_least_what_was_spent(
    train_task: Path, tmp_path: Path
) -> None:
    """A build done again (``--no-resume``, other options, a stale build) holds the record of
    its last agent run only: the Lab removed the record of the run before. A budget record
    counted again from the records the builds hold is at least what was spent, never more,
    and the Lab says so: ``state/budget.json`` is what holds all of it."""
    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    run = tmp_path / "run"
    config = _budgeted(tasks, max_agent_runs=2)
    for resume in (True, False):  # the second time as with --no-resume: built again
        Experiment(
            config, run, backend=FlakyBackend(), resume=resume, log=lambda line: None, report=False
        ).run()
    assert _spent(run)["agent_runs"] == 2
    assert len(list(run.glob("builds/*/*/trial-*/work/agent/agent_record.json"))) == 1
    (run / "state" / "budget.json").unlink()
    said: list[str] = []
    Experiment(config, run, backend=FlakyBackend(), log=said.append, report=False).run()
    (line,) = [line for line in said if "budget record was missing" in line]
    assert "counted at least 1 agent run(s)" in line and "build done again" in line, line
    assert _spent(run)["agent_runs"] == 1  # what the records show: the 2 runs are not in them


class _RecordCut(FakeAgent):
    """An agent run whose record this machine could not write in full (a full disk): the
    build failed after the agent ran, and names no record."""

    def build(self, context: Any) -> Any:
        outcome = super().build(context)
        record = Path(outcome.agent_record)
        record.write_bytes(record.read_bytes()[:40])
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(record))


class _FailsOnceStarted(FakeAgent):
    """The build fails after the agent had started and before its record is written (this
    machine could not read the agent's trajectory)."""

    def usage(self, trajectory: Path) -> Any:
        raise OSError(errno.EIO, os.strerror(errno.EIO), str(trajectory))


class _FailsBeforeItStarts(FakeAgent):
    """The build fails before the agent's sandbox starts (a full disk): nothing of it ran."""

    def extra_mounts(self, context: Any) -> Any:
        self.attempts = getattr(self, "attempts", 0) + 1
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), "visible/")


@pytest.mark.slow
@pytest.mark.parametrize(
    ("agent", "ran"),
    [(_RecordCut, True), (_FailsOnceStarted, True), (_FailsBeforeItStarts, False)],
    ids=["record-cut", "failed-once-started", "failed-before-it-started"],
)
def test_an_agent_run_without_a_record_it_can_read_is_never_taken_as_nothing(
    agent: type[FakeAgent],
    ran: bool,
    train_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A build that fails on this machine after its agent ran (its record cut by a full disk,
    or never written) is not a result (``infra_error``, built again by the next run), but the
    run may have spent: it counts, as a run of unknown cost, and this command does not run the
    agent again, as if nothing had run. One that failed before its agent started is tried again
    at once and given back."""
    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    made: list[FakeAgent] = []

    def make(name: str, **options: Any) -> FakeAgent:
        made.append(agent(**options))
        return made[-1]

    monkeypatch.setattr("real2simlab.orchestrate.experiment.get_method", make)
    config = parse_experiment(
        {
            "name": "budget",
            "tasks": {"paths": [str(tasks)]},
            "methods": [{"name": "fake_agent", "mode": "physical"}],
            "run": {"backend": "local", "retries": 2},
        }
    )
    backend = FlakyBackend()
    results, _ = _run(config, tmp_path / "run", backend)
    assert results.methods[0].infra_errors == 1
    (path,) = (tmp_path / "run" / "builds").rglob("build_record.json")
    assert "OSError" in (json.loads(path.read_text())["infra_error"] or "")
    if ran:
        assert _agents_started(backend) == ["r2s-agent-fake_agent"]  # once: not again
        assert _spent(tmp_path / "run") == {
            "agent_runs": 1, "cost_usd": 0.0, "cost_unknown_runs": 1
        }  # fmt: skip
    else:
        assert _agents_started(backend) == [] and getattr(made[0], "attempts", 0) == 3
        assert _spent(tmp_path / "run") == {"agent_runs": 0, "cost_usd": 0.0}


def test_an_agent_run_of_unknown_cost_is_never_taken_as_free(
    train_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent run that did not say what it cost is counted as one of unknown cost, never as
    $0 (``cost_unknown_runs`` in ``state/budget.json``): with a cost budget, no further agent
    starts, since what was spent cannot be shown to be under it; without one, every run
    starts, and each is counted so."""
    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    monkeypatch.setattr(
        "real2simlab.orchestrate.experiment.get_method",
        lambda name, **options: _SaysNoCost(**options),
    )
    backend = FlakyBackend()
    _, experiment = _run(_budgeted(tasks, trials=3, max_cost_usd=1.0), tmp_path / "a", backend)
    assert _agents_started(backend) == ["r2s-agent-fake_agent"]
    assert len(experiment.progress.skipped) == 2, experiment.progress.skipped
    assert all("cost budget ($1) cannot be checked" in s for s in experiment.progress.skipped)
    assert _spent(tmp_path / "a") == {"agent_runs": 1, "cost_usd": 0.0, "cost_unknown_runs": 1}
    _run(_budgeted(tasks, trials=3), tmp_path / "b")
    assert _spent(tmp_path / "b") == {"agent_runs": 3, "cost_usd": 0.0, "cost_unknown_runs": 3}


@pytest.mark.parametrize("name", ["codex", "gemini_cli", "harbor"])
def test_a_cost_budget_refuses_an_agent_that_never_reports_its_cost(
    name: str, train_task: Path, tmp_path: Path
) -> None:
    """``max_cost_usd`` limits agents by what their runs say they cost. Codex, Gemini CLI and
    Harbor runs never say it (each would be counted as $0, and every one would start): an
    experiment with a cost budget refuses them before it writes or runs anything, with the
    budgets that do limit them."""
    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    config = parse_experiment(
        {
            "name": "cost",
            "tasks": {"paths": [str(tasks)]},
            "methods": [{"name": name}],
            "run": {"backend": "local"},
            "budget": {"max_cost_usd": 20.0},
        }
    )
    experiment = Experiment(config, tmp_path / "run", log=lambda line: None, report=False)
    with pytest.raises(SetupError, match="max_cost_usd") as raised:
        experiment.run()
    assert "does not report what its runs cost" in str(raised.value), raised.value
    assert raised.value.fix is not None and "max_agent_runs" in raised.value.fix
    assert not (tmp_path / "run").exists()


def test_every_agent_says_whether_its_runs_report_their_cost() -> None:
    """Which agents a cost budget can limit: those whose runs say what they cost (Claude Code's
    and OpenCode's events, the mini-swe-agent harness for a model LiteLLM has a price for, the
    fake agent, which calls no model); not Codex, Gemini CLI or Harbor."""
    from real2simlab import plugins
    from real2simlab.agents.base import AgentMethod

    agents = {
        name: factory
        for name in plugins.names("method")
        if isinstance(factory := plugins.load("method", name), type)
        and issubclass(factory, AgentMethod)
    }
    reporting = {name for name, factory in agents.items() if factory.cost_reported()}
    assert reporting == {"claude_code", "opencode", "mini_swe", "fake_agent"}
    assert set(agents) - reporting == {"codex", "gemini_cli", "harbor"}


def test_an_agent_whose_setup_failed_is_not_spent_and_its_fix_is_not_refused(
    train_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent method without its credentials (no key: ``error:`` with ``fix: export ...``)
    is refused before any agent run is reserved: once its fix is done, the same command, with
    ``max_agent_runs = 1``, runs the agent instead of skipping it for a spent budget (A6-01)."""
    from real2simlab.agents import fake

    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    config = parse_experiment(
        {
            "name": "faults",
            "tasks": {"paths": [str(tasks)]},
            "methods": [{"name": "fake_agent", "mode": "physical"}],
            "run": {"backend": "local"},
            "budget": {"max_agent_runs": 1},
        }
    )

    def no_key(self: Any, context: Any) -> Any:
        raise SetupError("R2S_QA_KEY is not set", fix="export R2S_QA_KEY=...")

    with monkeypatch.context() as patch:
        patch.setattr(fake.FakeAgent, "credentials", no_key)
        with pytest.raises(SetupError, match="R2S_QA_KEY"):
            _run(config, tmp_path / "run")
    budget = tmp_path / "run" / "state" / "budget.json"
    spent = json.loads(budget.read_text())["agent_runs"] if budget.exists() else 0
    assert spent == 0  # refused before any agent run was reserved
    results, experiment = _run(config, tmp_path / "run")  # the fix done: the same command
    assert not experiment.progress.skipped and experiment.progress.builds == 1
    assert results.methods[0].delivered_rate == 1.0
    assert json.loads(budget.read_text())["agent_runs"] == 1


@pytest.mark.parametrize(
    ("method", "options", "said"),
    [
        ("claude_code", {}, "ANTHROPIC_API_KEY"),
        ("gemini_cli", {}, "GEMINI_API_KEY"),
        ("opencode", {"model": "openai/qa-model"}, "OPENAI_API_KEY"),
        ("codex", {"provider": "api-key"}, "OPENAI_API_KEY"),
        ("codex", {"codex_home": "{empty}"}, "codex login"),
        ("codex", {"codex_home": "{broken}"}, "cannot read"),
        ("codex", {"codex_home": "{no_base_url}"}, "base_url"),
        ("codex", {"codex_home": "{failing_auth}"}, "auth command"),
    ],
)
def test_agent_credentials_are_checked_before_any_agent_run_is_reserved(
    train_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    options: dict[str, Any],
    said: str,
) -> None:
    """Whatever keeps an agent's credentials from it (no key, no Codex login, a Codex
    configuration it cannot use, a provider's auth command that fails) stops the experiment
    with the adapter's ``error:`` and fix before any build starts or agent run is reserved
    (A6-01's class)."""
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
                 "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):  # fmt: skip
        monkeypatch.delenv(name, raising=False)
    homes = {
        "empty": "",
        "broken": "model = [unclosed\n",
        "no_base_url": 'model_provider = "gw"\n[model_providers.gw]\nname = "Gateway"\n',
        "failing_auth": 'model_provider = "gw"\n[model_providers.gw]\n'
        'base_url = "https://gateway.example.invalid/v1"\n'
        '[model_providers.gw.auth]\ncommand = "false"\n',
    }
    for name, config in homes.items():
        (tmp_path / name).mkdir()
        if config:
            (tmp_path / name / "config.toml").write_text(config)
    given = {
        key: str(tmp_path / value.strip("{}")) if key == "codex_home" else value
        for key, value in options.items()
    }
    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    config = parse_experiment(
        {
            "name": "faults",
            "tasks": {"paths": [str(tasks)]},
            "methods": [{"name": method, "network": "open", **given}],
            "run": {"backend": "local"},
            "budget": {"max_agent_runs": 1},
        }
    )
    with pytest.raises(SetupError, match=said) as refused:
        _run(config, tmp_path / "run")
    assert refused.value.fix
    assert not list((tmp_path / "run").glob("builds/*/*/trial-*"))  # no build started
    assert not (tmp_path / "run" / "state" / "budget.json").exists()  # no run reserved


class _InterruptedAgent(FlakyBackend):
    """The local backend, interrupted (Ctrl-C) at the agent's sandbox: before it starts, or
    once the agent ran. It keeps what ``state/budget.json`` said as the agent was starting
    (what a ``kill -9`` then leaves)."""

    def __init__(self, run_dir: Path, when: str) -> None:
        super().__init__()
        self.run_dir, self.when = run_dir, when
        self.spent_as_it_starts: dict[str, Any] | None = None

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        if not agent_runs(spec):
            return super().run(spec, log_path)
        self.spent_as_it_starts = _spent(self.run_dir)
        if self.when == "once it ran":
            super().run(spec, log_path)
        raise KeyboardInterrupt


@pytest.mark.parametrize(
    ("when", "spent"),
    [
        ("once it ran", {"agent_runs": 1, "cost_usd": 0.0, "cost_unknown_runs": 1}),
        ("before it began", {"agent_runs": 0, "cost_usd": 0.0}),
    ],
)
def test_an_agent_run_interrupted_once_it_began_is_never_taken_as_free(
    when: str, spent: dict[str, Any], train_task: Path, tmp_path: Path
) -> None:
    """An agent run is counted, as one of unknown cost, before its agent starts, and settled
    when it says what it cost. Stopped once its agent began (Ctrl-C, SIGTERM, or ``kill -9``,
    after which nothing more runs), what it spent is unknown, never $0: the same command with a
    cost budget starts no further agent. Stopped before its agent began, it is given back."""
    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    run = tmp_path / "run"
    stopped = _InterruptedAgent(run, when)
    with pytest.raises(KeyboardInterrupt):
        _run(_budgeted(tasks), run, stopped)
    assert stopped.spent_as_it_starts == {"agent_runs": 1, "cost_usd": 0.0, "cost_unknown_runs": 1}
    assert _spent(run) == spent
    again = FlakyBackend()
    _, experiment = _run(_budgeted(tasks, max_cost_usd=1.0), run, again)
    if when == "once it ran":
        assert _agents_started(again) == []
        (skipped,) = experiment.progress.skipped
        assert "the cost budget ($1) cannot be checked" in skipped, skipped
    else:  # the fake agent says it cost $0: settled, nothing unknown
        assert _agents_started(again) == ["r2s-agent-fake_agent"]
        assert _spent(run) == {"agent_runs": 1, "cost_usd": 0.0}


def _strict_json(path: Path) -> Any:
    """``path`` read as strict JSON: no ``Infinity`` or ``NaN``, which no other reader takes."""

    def refuse(constant: str) -> None:
        raise AssertionError(f"{path} holds {constant}: not JSON")

    return json.loads(path.read_text(), parse_constant=refuse)


def _strict_spent(run: Path) -> dict[str, Any]:
    """The run's ``state/budget.json`` as strict JSON, but its ``wall_time_s`` (the wall-time
    budget's, which its own tests check)."""
    spent = _strict_json(run / "state" / "budget.json")
    spent.pop("wall_time_s", None)
    return spent


@pytest.mark.slow
@pytest.mark.parametrize(
    ("costs", "spent"),
    [
        ([1e308, 1e308], {"agent_runs": 2, "cost_usd": 1e308, "cost_unknown_runs": 1}),
        ([float("inf")], {"agent_runs": 1, "cost_usd": 0.0, "cost_unknown_runs": 1}),
        ([float("nan")], {"agent_runs": 1, "cost_usd": 0.0, "cost_unknown_runs": 1}),
        ([3.0, -2.0], {"agent_runs": 2, "cost_usd": 3.0, "cost_unknown_runs": 1}),
    ],
    ids=["a total too large", "infinite", "not a number", "negative"],
)
def test_what_agent_runs_say_they_cost_never_damages_the_budget(
    costs: list[float],
    spent: dict[str, Any],
    train_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What an agent run says it cost is an amount of money, or unknown: Infinity (1e309 in its
    events), NaN or a negative cost is unknown, never money given back; a total too large to
    be a number leaves the run's cost unknown too. Neither its record nor
    ``state/budget.json`` is ever written with Infinity or NaN (which the Lab's own reader
    refuses: the run directory could never be resumed), the same command resumes, and the
    budget counted again from the agent records is the same."""

    class SaysWhatItCost(FakeAgent):
        def cost(self, trajectory: Path) -> float | None:
            return costs[int(trajectory.parents[2].name.removeprefix("trial-"))]

    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    monkeypatch.setattr(
        "real2simlab.orchestrate.experiment.get_method",
        lambda name, **options: SaysWhatItCost(**options),
    )
    config = _budgeted(tasks, trials=len(costs))
    run = tmp_path / "run"
    _run(config, run)
    assert _strict_spent(run) == spent
    for record in run.glob("builds/*/*/trial-*/work/agent/agent_record.json"):
        cost = _strict_json(record).get("cost_usd")
        assert cost is None or 0 <= cost < float("inf"), cost
    _run(config, run)  # the same command resumes, reusing every build
    assert _strict_spent(run) == spent
    (run / "state" / "budget.json").unlink()
    _run(config, run)  # counted again from the agent records
    assert _strict_spent(run) == spent


# --- 05 a host that cannot copy the solution -----------------------------------------------------


def test_a_solution_the_host_cannot_copy_gives_no_reward(
    train_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    from real2simlab.agents import base

    solution = shutil.copytree(train_task / "hidden" / "generator" / "program", tmp_path / "s")

    def full_disk(*args: Any, **kwargs: Any) -> None:
        raise OSError(errno.EFBIG, "File too large")

    monkeypatch.setattr(base.shutil, "copytree", full_disk)
    with pytest.raises(InfrastructureError, match="could not copy the solution"):
        score_solution(train_task, solution, backend=LocalBackend(quiet=True))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resource limits")
def test_the_reward_command_fails_when_the_host_cannot_write(
    train_task: Path, tmp_path: Path
) -> None:
    """A file-size limit on the host (RLIMIT_FSIZE): the command must fail, not print 0."""
    import resource

    solution = shutil.copytree(train_task / "hidden" / "generator" / "program", tmp_path / "s")
    (solution / "table.txt").write_text("0.0\n" * 50_000)  # 200 KB, over the limit below

    def limit() -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024, 64 * 1024))

    result = subprocess.run(
        [sys.executable, "-m", "real2simlab.cli.main", "train", "reward", "--task",
         str(train_task), "--solution", str(solution), "--backend", "local", "--json"],
        capture_output=True, text=True, preexec_fn=limit, check=False,
    )  # fmt: skip
    assert result.returncode != 0, result.stdout
    assert '"reward"' not in result.stdout and "error:" in result.stderr


# --- 06 an official scorer that fails ------------------------------------------------------------


class _BrokenScorer:
    name: ClassVar[str] = "broken"

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def problems(self) -> list[str]:
        return []

    def score(self, task: Any, **kwargs: Any) -> dict[str, Any]:
        if self.mode == "raises":
            raise RuntimeError("the official code crashed")
        if self.mode == "reports":
            return {"status": "scorer_failed", "exit_code": 1}
        return {"status": "scored", "value": 0.0}

    def headline(self, report: Any) -> Headline:
        value = report.get("value") if report.get("status") == "scored" else None
        return Headline(metric="m", value=value, range=(0.0, 1.0), source="official")


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["raises", "reports", "scores zero"])
def test_a_failed_official_scorer_is_not_a_zero(
    train_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from real2simlab.orchestrate import experiment as experiment_module

    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    data = read_task_config(tasks / "task.toml").to_json()
    data["scoring"] = {
        "protocol": "native",
        "native": {"scorer": "fourdcodebench", "options": {"kind": "synthetic", "case": "B_30"}},
    }
    write_task_config(TaskConfig.model_validate(data), tasks / "task.toml")
    monkeypatch.setattr(experiment_module, "get_scorer", lambda name, **o: _BrokenScorer(mode))
    results, _ = _run(_config([tasks], [{"name": "oracle"}]), tmp_path / "run")
    (card,) = results.methods[0].benchmarks
    if mode == "scores zero":  # a real score of 0 is a score, and a reward
        assert (card.scored, card.infra_errors, card.value.mean) == (1, 0, 0.0)
        return
    assert (card.scored, card.results, card.infra_errors) == (0, 1, 1)
    (row,) = results.to_json()["rows"]
    assert "official scorer" in row["infra_error"]
    with pytest.raises(SetupError, match="nothing could be exported"):
        export_dataset(tmp_path / "data", runs=[tmp_path / "run"], allow_eval_tasks=True)


@pytest.mark.slow
def test_an_official_score_is_reused_only_while_what_it_scored_is_unchanged(
    train_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from real2simlab.orchestrate import experiment as experiment_module

    calls: list[str] = []

    class Counting(_BrokenScorer):
        identity = "image-1"

        def score(self, task: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append(self.identity)
            return {"status": "scored", "value": 0.5}

        def cache_identity(self) -> str:
            return Counting.identity

    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    data = read_task_config(tasks / "task.toml").to_json()
    data["scoring"] = {
        "protocol": "native",
        "native": {"scorer": "fourdcodebench", "options": {"kind": "synthetic", "case": "B_30"}},
    }
    write_task_config(TaskConfig.model_validate(data), tasks / "task.toml")
    monkeypatch.setattr(experiment_module, "get_scorer", lambda name, **o: Counting("scores"))
    config = _config([tasks], [{"name": "oracle"}])
    for _ in range(2):
        results, _ = _run(config, tmp_path / "run")
        assert results.methods[0].benchmarks[0].value.mean == 0.5
    assert calls == ["image-1"]  # the second run reused the score
    report = next((tmp_path / "run").rglob("native_report.json"))
    report.write_text(report.read_text().replace("0.5", "0.0"))  # edited by hand
    results, _ = _run(config, tmp_path / "run")
    assert calls == ["image-1", "image-1"]  # scored again, not reused as edited
    assert results.methods[0].benchmarks[0].value.mean == 0.5
    Counting.identity = "image-2"  # e.g. the scorer's image changed
    _run(config, tmp_path / "run")
    assert calls == ["image-1", "image-1", "image-2"]


@pytest.mark.slow
def test_a_native_task_with_nothing_delivered_counts_against_the_method(
    train_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PhySimCode pools parameters over tasks: a task the method delivered nothing for adds its
    true parameters, none recovered (7 of 7 and 0 of 8 is 7/15, not 1)."""
    from real2simlab.orchestrate import experiment as experiment_module

    class Parameters(_BrokenScorer):
        def score(self, task: Any, **kwargs: Any) -> dict[str, Any]:
            delivered = (kwargs["delivery_dir"] / "run.py").is_file()
            counts = {"n_gt": 7, "n_within_tolerance": 7} if delivered else {"n_gt": 8,
                                                                             "n_within_tolerance": 0}  # fmt: skip
            return {"status": "scored", "parameters": counts}

        def headline(self, report: Any) -> Headline:
            counts = report.get("parameters") or {}
            value = counts["n_within_tolerance"] / counts["n_gt"] if counts else None
            return Headline(metric="parameters_within_20pct", value=value, range=(0.0, 1.0),
                            source="official", scorer="physimcode")  # fmt: skip

    tasks: list[Path] = []
    for name in ("a", "b"):
        task = _copy(train_task, tmp_path / "tasks" / name)
        data = read_task_config(task / "task.toml").to_json()
        data["id"] = f"psc/{name}"
        data["scoring"] = {"protocol": "native", "native": {"scorer": "physimcode"}}
        write_task_config(TaskConfig.model_validate(data), task / "task.toml")
        tasks.append(task)
    programs = tmp_path / "programs"
    shutil.copytree(train_task / "hidden" / "generator" / "program", programs / "a")  # b: none
    monkeypatch.setattr(experiment_module, "get_scorer", lambda name, **o: Parameters("scores"))
    config = _config(tasks, [{"name": "program", "path": str(programs)}])
    results, _ = _run(config, tmp_path / "run")
    (entry,) = results.methods[0].benchmarks
    assert entry.scored == 2 and entry.delivered == 1
    assert entry.value.mean == pytest.approx(7 / 15)


def test_a_world_the_official_scorer_rejects_is_the_deliverys_result() -> None:
    """4DCodeBench's scorer exits 1 when it rejects a world, with the world's ``dynamic_iou``
    reading (and why) in reward.json: the method's result, never retried as a broken scorer.
    Only a scorer that judged nothing is (no reward.json, or one without that reading, and an
    exit it never returns itself, below)."""
    from real2simlab.benchmarks.fourdcodebench.scorer import (
        FourDCodeBenchScorer,
        official_status,
    )
    from real2simlab.scoring.native import scorer_failure

    scorer = FourDCodeBenchScorer()
    reward = {"dynamic_iou": "error"}
    rejected = {"status": official_status(1, reward), "exit_code": 1, "reward": reward}
    broken = {"status": official_status(1, None), "exit_code": 1, "reward": None}
    assert official_status(0, {"dynamic_iou": 0.4}) == "scored"
    assert rejected["status"] == "rejected" and scorer_failure(scorer, rejected) is None
    assert scorer.headline(rejected).value == 0.0  # judged an error: the official rule's 0
    assert "failed" in (scorer_failure(scorer, broken) or "")


@pytest.mark.parametrize(
    ("exit_code", "reward"),
    [
        (137, {}),  # killed (the out-of-memory killer) after its gate wrote reward.json
        (-9, {"dynamic_iou": "error"}),  # the Docker client killed
        (2, {}),  # a usage error of the official command line
        (0, None),  # done, but no reward.json at all
        # stopped by an error of its own after its gate wrote reward.json ({}): a worker it
        # could not start (EAGAIN), the disk full when the dynamic_iou reading is written
        (1, {}),
        (1, {"scene_3d": 0.5}),  # readings, but none for dynamic_iou
        (0, {}),  # done, but without the dynamic_iou reading
    ],
)
def test_a_4dcodebench_scorer_that_did_not_finish_judged_nothing(
    exit_code: int, reward: dict[str, Any] | None
) -> None:
    """4DCodeBench's scorer writes ``reward.json`` as soon as its gate passes, before any metric,
    gives every world it judges a ``dynamic_iou`` reading there, and exits 0, or 1 for a world it
    rejects. Stopped with another exit code, done without ``reward.json``, or ended without that
    reading (an error of its own after its gate: a worker it could not start, a full disk), it
    judged nothing: a scorer failure, never the delivery's result or a score."""
    from real2simlab.benchmarks.fourdcodebench.scorer import (
        FourDCodeBenchScorer,
        official_status,
    )
    from real2simlab.scoring.native import scorer_failure

    scorer = FourDCodeBenchScorer()
    report = {"status": official_status(exit_code, reward), "exit_code": exit_code}
    assert report["status"] == "scorer_failed", report
    assert f"exit_code {exit_code}" in (scorer_failure(scorer, report) or "")


def _raised(error: BaseException) -> str:
    """An error raised in a worker of the official 4DCodeBench scorer as the scorer reports it
    (``scorer/execution.py``, ``serve``)."""
    return f"{type(error).__name__}: {error}"


_FFPROBE = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
            "stream=nb_read_frames,height,width,r_frame_rate", "-of", "json"]  # fmt: skip
"""How the official scorer reads a video's (F, H, W, fps) (``scorer/world.py``, ``video_stream``)."""


def _fdcb_outcome(where: str, text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """``reward.json`` and ``reward.detail.json`` as the pinned official 4DCodeBench scorer
    (``scorer/score.py``: ``gate``, ``geometry_pass``, ``failure``, ``rewards``) writes them for
    a world it rejects (exit 1) with ``dynamic_iou`` an error: ``text`` reported by a gate call
    (``gate``), by a gate's verdict (``verdict:<cause>``), by the dynamic_iou metric
    (``metric``), by its preparation step loading the delivered world (``geometry``) or the
    task's reference world (``reference``), by its worker that did not start (``worker``), or
    by a worker lost during the metric (``lost``); or a non-finite reading (``degenerate``)."""
    gate = {"dimensions": [240, 360, 640, 30.0], "reference": "/data/synthetic/B_30/world",
            "checkpoints": "/checkpoints", "video_gate": {"ok": True, "error": None, "detail": []},
            "world_gate": {"ok": True, "error": None, "detail": []}, "status": "ok"}  # fmt: skip
    blocked = {"dynamic_iou": "error", "scene_3d": "error"}
    both = {"dynamic_iou": "metric_failure", "scene_3d": "metric_failure"}
    if where == "gate" or where.startswith("verdict:"):
        cause = where.partition(":")[2] or "metric_failure"
        closed = {"ok": False, "error": cause, "detail": [text]}
        return blocked, {
            **gate, "dimensions": None, "video_gate": closed, "world_gate": None,
            "status": "gate_failure", "gate_failure": f"{cause}: {text}",
            "causes": {"dynamic_iou": cause, "scene_3d": cause},
        }  # fmt: skip
    if where == "metric":
        empty = "no covered pixel" in text or "no visible geometry" in text  # outcome.error_cause
        cause = "empty_first_frame" if empty else "metric_failure"
        return {"dynamic_iou": "error", "scene_3d": 0.5}, {
            **gate, "causes": {"dynamic_iou": cause}, "dynamic_iou_failure": text,
        }  # fmt: skip
    if where in ("geometry", "reference"):  # the step fails, every type needing it fails with it
        return blocked, {
            **gate, f"{where}_failure": text, "dynamic_iou_failure": text,
            "scene_3d_failure": text, "causes": both,
        }  # fmt: skip
    if where == "worker":
        return blocked, {**gate, "geometry_failure": text, "causes": both}
    if where == "lost":
        interrupted = "interrupted during dynamic_iou"
        return blocked, {
            **gate, "dynamic_iou_failure": text, "geometry_failure": interrupted,
            "scene_3d_failure": interrupted, "causes": both,
        }  # fmt: skip
    assert where == "degenerate", where  # the reading is not finite; a later metric failed
    return blocked, {
        **gate, "scene_3d_failure": text,
        "causes": {"dynamic_iou": "degenerate", "scene_3d": "metric_failure"},
    }  # fmt: skip


def _score_fdcb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
    reward: dict[str, Any] | None,
    detail: dict[str, Any] | str | None,
) -> tuple[Any, dict[str, Any]]:
    """The real ``FourDCodeBenchScorer.score`` of a delivery, the official scorer's container
    replaced by one that leaves its exit code, ``reward.json`` and ``reward.detail.json``
    (``None``: none; a string: written as it is)."""
    from real2simlab.benchmarks.fourdcodebench import scorer as module

    official = module.FourDCodeBenchScorer()
    monkeypatch.setattr(official, "problems", lambda: [])
    monkeypatch.setattr(
        official, "layout", lambda task, world, solution, roots, kind, case: (roots, [])
    )
    work = tmp_path / "work"

    def run_container(docker: str, run_args: Any, log: Path) -> tuple[int, bool, float]:
        results = work / "roots" / "runs" / "r2s" / "results"
        results.mkdir(parents=True)
        if reward is not None:
            (results / "reward.json").write_text(json.dumps(reward))
        if detail is not None:
            text = detail if isinstance(detail, str) else json.dumps(detail)
            (results / "reward.detail.json").write_text(text)
        log.write_text("")
        return exit_code, False, 1.0

    monkeypatch.setattr(module, "run_container", run_container)
    world, solution = tmp_path / "world", tmp_path / "solution"
    world.mkdir(parents=True)
    (world / "camera.json").write_text("{}")
    solution.mkdir()
    (solution / "build.sh").write_text("#!/bin/sh\n")
    report = official.score(
        cast(Any, None), delivery_dir=solution, output_dir=world, work_dir=work,
        options={"kind": "synthetic", "case": "B_30"},
    )  # fmt: skip
    return official, report


@pytest.mark.parametrize(
    ("where", "text"),
    [
        # the official worker framework: a call timed out, its worker was lost or never ready
        ("metric", "timeout after 1800s"),
        ("lost", "worker exited without a result"),
        ("metric", "worker is no longer running"),
        ("gate", "timeout after 600s"),
        ("gate", "worker exited without a result"),
        ("worker", "worker exited without a result"),
        ("worker", "timeout after 600s"),
        ("metric", "model load failed: OutOfMemoryError: CUDA out of memory."),
        # this machine, in the worker: disk, quota, the kernel's memory, processes, open files, I/O
        ("metric", _raised(OSError(28, os.strerror(28)))),  # ENOSPC writing its arrays
        ("metric", _raised(OSError(122, os.strerror(122), "/runs/r2s/results/.dynamic.npz.7.tmp"))),
        ("geometry", _raised(OSError(12, os.strerror(12)))),  # ENOMEM
        ("worker", _raised(BlockingIOError(11, os.strerror(11)))),  # EAGAIN: no more processes
        ("metric", _raised(RuntimeError("can't start new thread"))),
        ("gate", _raised(OSError(24, os.strerror(24), "/runs/r2s/workspace/world/render.mp4"))),
        ("metric", _raised(OSError(23, os.strerror(23)))),  # ENFILE
        ("metric", _raised(OSError(5, os.strerror(5)))),  # EIO
        # the GPU the metric rasterises on: missing, out of memory, failed
        ("metric", "RuntimeError: rasterisation needs a CUDA device"),
        ("metric", "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB."),
        ("metric", "RuntimeError: CUDA error: an illegal memory access was encountered"),
        ("metric", "AcceleratorError: CUDA error: out of memory"),
        ("metric", "RuntimeError: Cuda error: 2[cudaMalloc(&buffer, bytes);]"),
        # the scorer image without its tools
        ("gate", _raised(FileNotFoundError(2, os.strerror(2), "ffmpeg"))),
        ("gate", _raised(FileNotFoundError(2, os.strerror(2), "ffprobe"))),
        ("metric", _raised(FileNotFoundError(2, os.strerror(2), "scorer-vdb"))),
        # the task's own data the scorer could not read: the case video, the reference world
        ("gate", _raised(subprocess.CalledProcessError(
            1, [*_FFPROBE, "/cases/synthetic/B_30/reference.mp4"]))),
        ("gate", _raised(subprocess.CalledProcessError(
            -9, [*_FFPROBE, "/cases/real/internet_01/reference.mp4"]))),
        ("reference", _raised(subprocess.CalledProcessError(
            1, [*_FFPROBE, "/data/synthetic/B_30/world/render.mp4"]))),
    ],
)  # fmt: skip
def test_a_4dcodebench_measurement_that_did_not_finish_is_not_the_worlds_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str, text: str
) -> None:
    """4DCodeBench's scorer writes ``error`` (and exits 1) for a world whose measurement did not
    finish as for a world it rejects, saying why in ``reward.detail.json``: its worker timed
    out, was lost or never ready; this machine's disk, quota, processes, open files or GPU
    failed it, or its kernel refused memory (ENOMEM); its image lacks a tool; the task's own
    case video or reference world could not be read. Not a judgement of the world: a scorer
    failure (redone, no reward), never a rejection counted as 0."""
    from real2simlab.scoring.native import scorer_failure

    reward, detail = _fdcb_outcome(where, text)
    official, report = _score_fdcb(tmp_path, monkeypatch, 1, reward, detail)
    assert report["status"] == "scorer_failed", report
    failure = scorer_failure(official, report) or ""
    assert "the official scorer 'fourdcodebench' failed" in failure and text[:40] in failure
    assert official.headline(report).value is None
    assert report["causes"]["dynamic_iou"] == "metric_failure"  # the official record, kept


@pytest.mark.parametrize(
    ("where", "text"),
    [
        # its gates' verdicts on the delivered video and world
        ("verdict:no_video", "Missing render.mp4"),
        ("verdict:undecodable_video", "Video decoding failed: moov atom not found"),
        ("verdict:video_mismatch", "(F,H,W,fps) (120, 360, 640, Fraction(30, 1)) != reference "
                                   "(240, 360, 640, Fraction(30, 1))"),
        ("verdict:no_world", "Missing world/camera.json"),
        ("verdict:invalid_world", "camera.json: intrinsics must be 3x3"),
        # a metric's verdict, a non-finite reading, a defect of the world a call failed on
        ("metric", "ValueError: /runs/r2s/workspace/world: frame 0 rasterises to no covered pixel"),
        ("degenerate", "ValueError: scene_3d failed on the world"),
        ("metric", "ValueError: reference is (240, 360, 640) and the submission is (239, 360, 640)"),
        ("geometry", "ValueError: /runs/r2s/workspace/world/meshes/0.npz: faces must be int32 (T,3)"),
        ("worker", "ValueError: no frame of the submission rasterises"),
        ("gate", _raised(subprocess.CalledProcessError(
            1, [*_FFPROBE, "/runs/r2s/workspace/world/render.mp4"]))),  # the delivered video
        ("gate", _raised(ZeroDivisionError("Fraction(0, 0)"))),  # its frame rate is 0/0
        # what the world names, where an error quotes it, is not the error's own type and text
        ("metric", "ValueError: /runs/r2s/workspace/world/dynamics/OSError: [Errno 28] "
                   "CUDA error.npz: unknown members ['x']"),
        ("metric", "ValueError: /runs/r2s/workspace/world/cases/a/b/reference.mp4: no frames"),
        # a failed call this does not recognize stays the world's result
        ("gate", "KeyError: 'streams'"),
        # memory the world asked for: an array its archive's header declares larger than any
        # machine (what the pinned official world gate wrote for a 300-byte meshes/1.npz), a
        # decompression bomb; a machine short of memory kills the worker instead
        ("gate", "MemoryError: Unable to allocate 384. TiB for an array with shape "
                 "(105553116266496,) and data type float32"),
        ("metric", "_ArrayMemoryError: Unable to allocate 7.28 TiB for an array with shape "
                   "(1000000, 1000000) and data type float64"),
        ("metric", _raised(MemoryError())),
    ],
)  # fmt: skip
def test_a_world_4dcodebench_rejects_stays_the_deliverys_result_and_counts_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str, text: str
) -> None:
    """What 4DCodeBench's gates or metrics judge of the world (a closed gate, an empty first
    frame, a non-finite reading, a defect a metric failed on) stays the delivery's result,
    ``rejected``, counted 0 as the official aggregates count an ``error``: reading why does
    not make a world the scorer refuses a scorer failure."""
    from real2simlab.scoring.native import scorer_failure

    reward, detail = _fdcb_outcome(where, text)
    official, report = _score_fdcb(tmp_path, monkeypatch, 1, reward, detail)
    assert report["status"] == "rejected", report
    assert scorer_failure(official, report) is None
    assert official.headline(report).value == 0.0


def test_a_world_whose_archive_declares_an_impossible_array_counts_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delivered archive of a few hundred bytes whose header declares a 384 TiB array makes
    NumPy, reading it as 4DCodeBench's world gate does, fail to allocate on any machine: the
    world's defect, which the official scorer counts 0, never a scorer failure that is redone
    instead of scored."""
    import zipfile

    import numpy as np

    from real2simlab.scoring.native import scorer_failure

    member = tmp_path / "vertices_0000.npy"
    with member.open("wb") as handle:
        header = {"descr": "<f4", "fortran_order": False, "shape": (2**45, 3)}
        np.lib.format.write_array_header_1_0(handle, header)
        handle.write(bytes(12))
    archive = tmp_path / "1.npz"
    with zipfile.ZipFile(archive, "w") as packed:
        packed.write(member, member.name)
    assert archive.stat().st_size < 1000
    with pytest.raises(MemoryError) as raised, np.load(archive) as loaded:
        loaded["vertices_0000"]
    text = _raised(raised.value)  # as the official worker reports it
    assert text.startswith("MemoryError: Unable to allocate"), text
    reward, detail = _fdcb_outcome("gate", text)
    official, report = _score_fdcb(tmp_path / "score", monkeypatch, 1, reward, detail)
    assert report["status"] == "rejected", report
    assert scorer_failure(official, report) is None
    assert official.headline(report).value == 0.0


@pytest.mark.parametrize(
    ("exit_code", "reward", "detail", "status", "headline"),
    [
        # rejected, without a readable record of why dynamic_iou is an error: it writes that
        # record with the reading, so it stopped in between (a full disk), never a 0
        (1, {"dynamic_iou": "error"}, None, "scorer_failed", None),
        (1, {"dynamic_iou": "error"}, '{"causes": {', "scorer_failed", None),
        (1, {"dynamic_iou": "error"}, "[]", "scorer_failed", None),
        (1, {"dynamic_iou": "error"}, {"status": "ok", "causes": {}}, "scorer_failed", None),
        (1, {"dynamic_iou": "error"}, {"status": "ok"}, "scorer_failed", None),
        # what is measured, or finished, stands without it
        (1, {"dynamic_iou": 0.4, "scene_3d": "error"}, None, "rejected", 0.4),
        (0, {"dynamic_iou": "error"}, None, "scored", 0.0),  # exit 0: nothing failed
        (1, {"dynamic_iou": "uncomputable"}, None, "rejected", None),
    ],
)
def test_a_4dcodebench_rejection_without_its_record_of_why_is_not_a_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
    reward: dict[str, Any],
    detail: dict[str, Any] | str | None,
    status: str,
    headline: float | None,
) -> None:
    """A world 4DCodeBench's scorer rejected with ``dynamic_iou`` an error, whose
    ``reward.detail.json`` is missing, unreadable, or older than the reading (no cause for
    it), is not judged: the scorer stopped before it said why. A measured headline, and a
    scoring that ended without a failure (exit 0), stand without that record."""
    from real2simlab.scoring.native import scorer_failure

    official, report = _score_fdcb(tmp_path, monkeypatch, exit_code, reward, detail)
    assert report["status"] == status, report
    assert (scorer_failure(official, report) is not None) == (status == "scorer_failed")
    assert official.headline(report).value == headline


def test_a_4dcodebench_failed_call_on_a_full_disk_is_not_the_worlds_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call of 4DCodeBench's scorer failing with an error this does not recognize, while the
    disk holding the scorer's output is full, failed because of the disk; a gate's verdict on
    the world does not depend on it, and with space left the failure stays the world's."""
    import types

    from real2simlab.sandbox import process

    unknown = _fdcb_outcome("metric", "SystemError: error return without exception set")
    verdict = _fdcb_outcome("verdict:invalid_world", "camera.json: intrinsics must be 3x3")
    _, report = _score_fdcb(tmp_path / "space", monkeypatch, 1, *unknown)
    assert report["status"] == "rejected"
    full = types.SimpleNamespace(f_bavail=1, f_frsize=1024 * 1024)
    monkeypatch.setattr(process.os, "statvfs", lambda path: full)
    official, report = _score_fdcb(tmp_path / "full", monkeypatch, 1, *unknown)
    assert report["status"] == "scorer_failed" and "is full" in report["detail"], report
    assert official.headline(report).value is None
    _, report = _score_fdcb(tmp_path / "verdict", monkeypatch, 1, *verdict)
    assert report["status"] == "rejected"


@pytest.mark.parametrize(
    ("scorer", "report", "own_failure"),
    [
        # PhysCodeGym: a Lab that blamed the delivery for any ValueError of the official metrics
        # kept the reference video's (as this machine's OpenCV reads it) as invalid_frames, 0
        pytest.param("physcodegym_pixels", {"status": "invalid_frames", "S_mot": 0.0,
                     "detail": "ValueError: Failed to open video: ../task/hidden/reference/reference.mp4"},
                     True, id="physcodegym-reference-unreadable"),
        pytest.param("physcodegym_pixels", {"status": "invalid_frames", "S_mot": 0.0,
                     "detail": "ValueError: The benchmark requires 30 FPS"},
                     True, id="physcodegym-reference-fps"),
        pytest.param("physcodegym_pixels", {"status": "invalid_frames", "S_mot": 0.0,
                     "detail": "ValueError: LPIPS produced a non-finite distance"},
                     True, id="physcodegym-other-error"),
        # ... what PhysCodeGym blames on the delivered frames (the count: either input)
        pytest.param("physcodegym_pixels", {"status": "invalid_frames", "S_mot": 0.0,
                     "detail": "ValueError: Failed to decode image: frames/frame_0001.png"},
                     False, id="physcodegym-frame"),
        pytest.param("physcodegym_pixels", {"status": "invalid_frames", "S_mot": 0.0,
                     "detail": "ValueError: Both inputs must contain exactly 240 aligned frames"},
                     False, id="physcodegym-frame-count"),
        pytest.param("physcodegym_pixels", {"status": "invalid_frames", "S_mot": 0.0,
                     "detail": "OpenCV cannot read 1 frame(s): frame_0002.png"},
                     False, id="physcodegym-unreadable-frame"),
        pytest.param("physcodegym_pixels", {"status": "scored", "S_mot": 0.5},
                     False, id="physcodegym-scored"),
        # 4DCodeBench: a Lab that took any exit 1 with a reward.json for a rejection (and exit 0
        # for a score) kept a scorer that judged nothing as the delivery's result
        pytest.param("fourdcodebench", {"status": "rejected", "exit_code": 1, "reward": {}},
                     True, id="4dcodebench-no-reading"),
        pytest.param("fourdcodebench", {"status": "rejected", "exit_code": 137, "reward": {}},
                     True, id="4dcodebench-killed"),
        pytest.param("fourdcodebench", {"status": "scored", "exit_code": 0, "reward": None},
                     True, id="4dcodebench-no-reward"),
        # ... a measurement that did not finish, kept with the official record of why
        pytest.param("fourdcodebench", {"status": "rejected", "exit_code": 1,
                     "reward": {"dynamic_iou": "error"}, "causes": {"dynamic_iou": "metric_failure"},
                     "failures": {"dynamic_iou_failure": "OSError: [Errno 28] No space left on device"}},
                     True, id="4dcodebench-machine-failure"),
        pytest.param("fourdcodebench", {"status": "rejected", "exit_code": 1,
                     "reward": {"dynamic_iou": "error"}, "causes": None, "failures": None},
                     True, id="4dcodebench-no-record-of-why"),
        # ... what 4DCodeBench judged (with why), and a rejection an earlier Lab kept without why
        pytest.param("fourdcodebench", {"status": "rejected", "exit_code": 1,
                     "reward": {"dynamic_iou": "error"}, "causes": {"dynamic_iou": "invalid_world"},
                     "failures": {"gate_failure": "invalid_world: camera.json: intrinsics must be 3x3"}},
                     False, id="4dcodebench-rejected-why"),
        pytest.param("fourdcodebench", {"status": "rejected", "exit_code": 1,
                     "reward": {"dynamic_iou": "error"}}, False, id="4dcodebench-rejected"),
        pytest.param("fourdcodebench", {"status": "scored", "exit_code": 0,
                     "reward": {"dynamic_iou": 0.4}}, False, id="4dcodebench-scored"),
        pytest.param("fourdcodebench", {"status": "no_world", "reward": None},
                     False, id="4dcodebench-no-world"),
    ],
)  # fmt: skip
def test_a_report_an_earlier_lab_kept_is_read_as_the_scorer_attributes_now(
    scorer: str, report: dict[str, Any], own_failure: bool
) -> None:
    """A kept official report is reused only while its scorer reads it as a score. PhysCodeGym
    and 4DCodeBench attribute a failure while they score; their ``failure`` reads a report kept
    under an earlier attribution (the reference video's error blamed on the delivered frames, a
    scorer that judged nothing taken for a rejection or a score) as they attribute it now, and
    what they still blame on the delivery stays its result."""
    from real2simlab.benchmarks.fourdcodebench.scorer import FourDCodeBenchScorer
    from real2simlab.benchmarks.physcodegym.scorer import PhysCodeGymPixels
    from real2simlab.scoring.native import scorer_failure

    official = (
        PhysCodeGymPixels(lpips=False) if scorer == "physcodegym_pixels" else FourDCodeBenchScorer()
    )
    failure = scorer_failure(official, report)
    assert (failure is not None) == own_failure, failure
    if own_failure:
        assert f"the official scorer {scorer!r} failed" in (failure or "")


@pytest.mark.slow
def test_a_video2world_row_its_evaluator_could_not_measure_is_not_a_score(
    train_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Video2World blocks the row of a package its evaluator could not measure (the evaluator
    crashed or timed out, its renderer or GPU died, its metric code failed) and says it is never
    the package's result. Not a score: an infrastructure error, redone when the experiment runs
    again, and no reward."""
    from real2simlab.benchmarks.video2world.scorer import Video2WorldScorer
    from real2simlab.orchestrate import experiment as experiment_module

    calls: list[str] = []

    class Evaluator(Video2WorldScorer):  # the real scorer, its container replaced
        def problems(self) -> list[str]:
            return []

        def cache_identity(self) -> str:
            return "image"

        def score(self, task: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append(task.id)
            reason = "evaluator failure: evaluator exited with -9, see eval/eval.log"
            issue = {"family": "push-t", "sample": "s", "reason": reason}
            return {"status": "blocked", "official_status": "blocked", "S": None,
                    "score": None, "issues": [issue]}  # fmt: skip

    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    data = read_task_config(tasks / "task.toml").to_json()
    data["scoring"] = {"protocol": "native", "native": {"scorer": "video2world"}}
    write_task_config(TaskConfig.model_validate(data), tasks / "task.toml")
    monkeypatch.setattr(experiment_module, "get_scorer", lambda name, **o: Evaluator())
    config = _config([tasks], [{"name": "oracle"}])
    for _ in range(2):
        results, _ = _run(config, tmp_path / "run")
    assert len(calls) == 2  # redone by the second run, not reused
    (card,) = results.methods[0].benchmarks
    assert (card.scored, card.results, card.infra_errors) == (0, 1, 1)
    (row,) = results.to_json()["rows"]
    assert "evaluator failure" in row["infra_error"]
    with pytest.raises(SetupError, match="nothing could be exported"):
        export_dataset(tmp_path / "data", runs=[tmp_path / "run"], allow_eval_tasks=True)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("report", "said"),
    [
        # A12-review-07's report: an evaluation error (which Video2World's runner writes only
        # when its evaluator failed) and no issue
        pytest.param(
            {"status": "blocked", "official_status": "blocked", "S": None,
             "evaluation_error": "scorer could not initialize"},
            "scorer could not initialize", id="evaluation-error",
        ),
        # no build verdict: the evaluator, which writes it, could not read the sample
        pytest.param(
            {"status": "blocked", "official_status": "blocked", "S": None, "score": None,
             "issues": [{"family": "push-t", "sample": "s", "reason": "build must be boolean"}]},
            "build must be boolean", id="no-build-verdict",
        ),
        # no issue at all: a sample Video2World does not score
        pytest.param(
            {"status": "blocked", "official_status": "not_applicable", "S": None, "score": None,
             "issues": []},
            "'not_applicable'", id="not-applicable",
        ),
    ],
)  # fmt: skip
def test_a_video2world_row_that_says_nothing_of_the_package_is_not_a_score(
    train_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: dict[str, Any],
    said: str,
) -> None:  # fmt: skip
    """A blocked Video2World row whose cause is not the package (its evaluator's error, no build
    verdict from the evaluator, no issue at all) is an infrastructure error: redone when the
    experiment runs again, out of the scorecard, and no reward."""
    from real2simlab.benchmarks.video2world.scorer import Video2WorldScorer
    from real2simlab.orchestrate import experiment as experiment_module

    calls: list[str] = []

    class Evaluator(Video2WorldScorer):  # the real scorer, its container replaced
        def problems(self) -> list[str]:
            return []

        def cache_identity(self) -> str:
            return "image"

        def score(self, task: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append(task.id)
            return dict(report)

    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    data = read_task_config(tasks / "task.toml").to_json()
    data["scoring"] = {"protocol": "native", "native": {"scorer": "video2world"}}
    write_task_config(TaskConfig.model_validate(data), tasks / "task.toml")
    monkeypatch.setattr(experiment_module, "get_scorer", lambda name, **o: Evaluator())
    config = _config([tasks], [{"name": "oracle"}])
    for _ in range(2):
        results, _ = _run(config, tmp_path / "run")
    assert len(calls) == 2  # redone by the second run, not reused
    (card,) = results.methods[0].benchmarks
    assert (card.scored, card.results, card.infra_errors) == (0, 1, 1)
    (row,) = results.to_json()["rows"]
    assert said in row["infra_error"]
    with pytest.raises(SetupError, match="nothing could be exported"):
        export_dataset(tmp_path / "data", runs=[tmp_path / "run"], allow_eval_tasks=True)


@pytest.mark.slow
def test_a_kept_official_score_the_scorer_now_calls_its_own_failure_is_scored_again(
    train_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A report kept for reuse is read again by its scorer before it is reused: one a Lab that
    attributed it otherwise kept as the delivery's result (e.g. a Video2World row its evaluator
    could not measure), and which the scorer now calls its own failure, is scored again, not
    reused as a result."""
    from real2simlab.orchestrate import experiment as experiment_module

    calls: list[str] = []

    class Attributed(_BrokenScorer):
        own_failure: ClassVar[bool] = False

        def score(self, task: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append(task.id)
            return {"status": "blocked", "value": None}

        def failure(self, report: Any) -> str | None:
            return "the evaluator failed" if Attributed.own_failure else None

        def cache_identity(self) -> str:
            return "image"

    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    data = read_task_config(tasks / "task.toml").to_json()
    data["scoring"] = {
        "protocol": "native",
        "native": {"scorer": "fourdcodebench", "options": {"kind": "synthetic", "case": "B_30"}},
    }
    write_task_config(TaskConfig.model_validate(data), tasks / "task.toml")
    monkeypatch.setattr(experiment_module, "get_scorer", lambda name, **o: Attributed("scores"))
    config = _config([tasks], [{"name": "oracle"}])
    results, _ = _run(config, tmp_path / "run")
    assert results.methods[0].benchmarks[0].infra_errors == 0  # kept: a result, no score
    Attributed.own_failure = True  # the scorer's attribution changed (a newer Lab)
    results, _ = _run(config, tmp_path / "run")
    assert len(calls) == 2  # scored again, not reused
    (card,) = results.methods[0].benchmarks
    assert (card.scored, card.results, card.infra_errors) == (0, 1, 1)
    (row,) = results.to_json()["rows"]
    assert "the evaluator failed" in row["infra_error"]


@pytest.mark.slow
@pytest.mark.parametrize(
    ("scorer", "options", "kept", "fresh"),
    [
        # A12-review-06: this machine's OpenCV could not open the reference video, kept as the
        # delivery's invalid_frames and 0
        pytest.param(
            "physcodegym_pixels", {},
            {"status": "invalid_frames", "S_SSIM": 0.0, "S_mot": 0.0,
             "detail": "ValueError: Failed to open video: ../task/hidden/reference/reference.mp4"},
            {"status": "scored", "S_SSIM": 0.9, "S_mot": 0.75}, id="physcodegym",
        ),
        # the official scorer stopped by an error of its own after its gate, kept as a rejection
        pytest.param(
            "fourdcodebench", {"kind": "synthetic", "case": "B_30"},
            {"status": "rejected", "exit_code": 1, "types": ["dynamic_iou"], "reward": {}},
            {"status": "scored", "exit_code": 0, "types": ["dynamic_iou"],
             "reward": {"dynamic_iou": 0.75}}, id="4dcodebench",
        ),
    ],
)  # fmt: skip
def test_a_result_an_earlier_lab_kept_for_the_scorers_own_failure_is_scored_again(
    train_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scorer: str,
    options: dict[str, Any], kept: dict[str, Any], fresh: dict[str, Any],
) -> None:  # fmt: skip
    """A report a Lab that attributed it otherwise kept as the delivery's result (PhysCodeGym's
    error about the reference video blamed on the frames, a 4DCodeBench scorer that judged
    nothing taken for a rejection) is scored again, not reused: the real scorer's ``failure``
    reads it, though its attribution is made while scoring."""
    from real2simlab.benchmarks.fourdcodebench.scorer import FourDCodeBenchScorer
    from real2simlab.benchmarks.physcodegym.scorer import PhysCodeGymPixels
    from real2simlab.orchestrate import experiment as experiment_module
    from real2simlab.scoring.native import default_failure

    official = (
        PhysCodeGymPixels(lpips=False) if scorer == "physcodegym_pixels" else FourDCodeBenchScorer()
    )
    calls: list[str] = []
    reports = [kept]

    def score(task: Any, **kwargs: Any) -> dict[str, Any]:  # the official code, replaced
        calls.append(task.id)
        return dict(reports[-1])

    monkeypatch.setattr(official, "score", score)
    monkeypatch.setattr(official, "problems", lambda: [])
    monkeypatch.setattr(official, "cache_identity", lambda: "image")
    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    data = read_task_config(tasks / "task.toml").to_json()
    data["scoring"] = {"protocol": "native", "native": {"scorer": scorer, "options": options}}
    write_task_config(TaskConfig.model_validate(data), tasks / "task.toml")
    monkeypatch.setattr(experiment_module, "get_scorer", lambda name, **o: official)
    config = _config([tasks], [{"name": "oracle"}])
    with monkeypatch.context() as earlier:  # an earlier Lab: only scorer_failed is a failure
        earlier.setattr(official, "failure", lambda report: default_failure(scorer, report))
        results, _ = _run(config, tmp_path / "run")
    assert results.methods[0].benchmarks[0].infra_errors == 0  # kept: the delivery's result
    reports.append(fresh)  # this machine, and the scorer, as they are now
    results, _ = _run(config, tmp_path / "run")
    assert len(calls) == 2  # scored again, not reused
    (card,) = results.methods[0].benchmarks
    assert (card.scored, card.infra_errors, card.value.mean) == (1, 0, 0.75)


def _video(path: Path, frames: int, codec: list[str]) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc=size=64x48:rate=30:duration={frames / 30}", *codec, str(path)],
        check=True,
    )  # fmt: skip
    return path


def test_a_video_the_official_reader_cannot_read_is_the_deliverys_fault(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    from real2simlab.benchmarks.physicsiq.scorer import _opencv_problem

    good = _video(tmp_path / "good.mp4", 150, ["-c:v", "libx264", "-pix_fmt", "yuv420p"])
    assert _opencv_problem(good) is None
    short = _video(tmp_path / "short.mp4", 120, ["-c:v", "libx264", "-pix_fmt", "yuv420p"])
    assert "120 of its 150" in (_opencv_problem(short) or "")
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(good.read_bytes()[: len(good.read_bytes()) // 3])
    assert _opencv_problem(broken) is not None


def test_frames_the_official_reader_cannot_read_are_the_deliverys_fault(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    import cv2
    import numpy as np

    from real2simlab.benchmarks.physcodegym.scorer import _unreadable_frames

    frames = tmp_path / "frames"
    frames.mkdir()
    cv2.imwrite(str(frames / "frame_0001.png"), np.zeros((8, 8, 3), np.uint8))
    assert _unreadable_frames(frames) is None
    (frames / "frame_0002.png").write_bytes(b"not a png")
    assert "frame_0002.png" in (_unreadable_frames(frames) or "")


def test_frames_an_opencv_that_reads_no_png_cannot_read_are_not_the_deliverys_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OpenCV that cannot read even a PNG it wrote itself is this machine's problem: the
    delivered frames it cannot read are not ``invalid_frames`` and a score of 0; the scorer
    failed (not kept for reuse). A frame only the delivery got wrong still is the delivery's."""
    pytest.importorskip("cv2")
    import cv2
    import numpy as np

    from real2simlab.benchmarks.physcodegym.scorer import PhysCodeGymPixels
    from real2simlab.scoring.native import scorer_failure
    from real2simlab.task import load_task

    output = tmp_path / "output"
    (output / "frames").mkdir(parents=True)
    assert cv2.imwrite(str(output / "frames" / "frame_0001.png"), np.zeros((4, 4, 3), np.uint8))
    (output / "frames" / "frame_0002.png").write_bytes(b"not a png")
    task = load_task(Path(__file__).resolve().parents[1] / "examples" / "tasks" / "puck_slide")
    scorer = PhysCodeGymPixels(lpips=False)

    def score() -> dict[str, Any]:
        return scorer.score(
            task, delivery_dir=tmp_path, output_dir=output, work_dir=tmp_path / "w", options={}
        )

    report = score()  # this machine reads PNGs: the delivery's broken frame
    assert report["status"] == "invalid_frames" and scorer_failure(scorer, report) is None
    monkeypatch.setattr(cv2, "imread", lambda *args, **kwargs: None)  # it reads no PNG at all
    report = score()
    assert report["status"] == "scorer_failed", report
    assert "cannot read a PNG it wrote itself" in (scorer_failure(scorer, report) or "")
    assert scorer.headline(report).value is None
    assert not list((tmp_path / "w").iterdir())  # the probe is not left behind


_FAKE_PHYSCODEGYM_METRICS = '''\
import os
from pathlib import Path


def evaluate(reference, candidate, output, *, lpips=True):
    """Raises the ValueError PhysCodeGym's own metrics raise, naming the inputs as given."""
    frame = sorted(Path(candidate).iterdir())[0]
    message = os.environ["R2S_TEST_OFFICIAL_ERROR"]
    raise ValueError(message.format(reference=reference, frame=frame))
'''


@pytest.mark.parametrize(
    ("error", "frames", "delivery_fault"),
    [
        # PhysCodeGym's messages (de1c8b7) about the reference or how this machine reads it:
        ("The benchmark requires 30 FPS", 240, False),
        ("Failed to open video: {reference}", 240, False),
        ("Video has invalid timing metadata: {reference}", 240, False),
        ("Decoded 120 frames from {reference}, but metadata declares 240", 240, False),
        ("Both inputs must contain exactly 240 aligned frames", 240, False),
        ("LPIPS produced a non-finite distance", 240, False),
        # ... and about the delivered frames:
        ("Both inputs must contain exactly 240 aligned frames", 239, True),
        ("Failed to decode image: {frame}", 240, True),
        ("Image shapes differ: reference=(360, 640, 3), candidate=(360, 640)", 240, True),
    ],
)
def test_physcodegym_blames_the_delivery_only_for_its_own_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: str, frames: int, delivery_fault: bool
) -> None:
    """PhysCodeGym's metrics raise ``ValueError`` about either input. Only a fault of the
    delivered frames is the delivery's ``invalid_frames`` (a score of 0); a reference the
    official reader rejects (not 30 fps, not 240 frames, unreadable on this machine, e.g. an
    OpenCV without a video backend) is the scorer failing: not a score, and not kept for reuse."""
    pytest.importorskip("cv2")
    import cv2
    import numpy as np

    from real2simlab.benchmarks.physcodegym.scorer import PhysCodeGymPixels
    from real2simlab.scoring.native import scorer_failure
    from real2simlab.task import load_task

    checkout = tmp_path / "physcodegym-checkout"
    (checkout / "physcodegym").mkdir(parents=True)
    (checkout / "physcodegym" / "__init__.py").write_text("")
    (checkout / "physcodegym" / "metrics.py").write_text(_FAKE_PHYSCODEGYM_METRICS)
    monkeypatch.setenv("R2S_PHYSCODEGYM", str(checkout))
    monkeypatch.setenv("R2S_TEST_OFFICIAL_ERROR", error)
    output = tmp_path / "output"
    (output / "frames").mkdir(parents=True)
    for index in range(1, frames + 1):
        image = np.zeros((4, 4, 3), np.uint8)
        assert cv2.imwrite(str(output / "frames" / f"frame_{index:04d}.png"), image)
    task = load_task(Path(__file__).resolve().parents[1] / "examples" / "tasks" / "puck_slide")
    scorer = PhysCodeGymPixels(lpips=False)
    report = scorer.score(
        task, delivery_dir=tmp_path, output_dir=output, work_dir=tmp_path / "w", options={}
    )
    failure = scorer_failure(scorer, report)
    if delivery_fault:
        assert report["status"] == "invalid_frames" and failure is None, report
        assert scorer.headline(report).value == 0.0
    else:
        assert report["status"] == "scorer_failed" and failure is not None, report
        assert "S_mot" not in report and scorer.headline(report).value is None
        assert error.split("{")[0].split(":")[0] in failure  # what the official code said


# --- 07 a Docker daemon chosen by DOCKER_HOST ---------------------------------------------------


def test_every_docker_command_talks_to_the_same_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    from real2simlab.sandbox import docker

    monkeypatch.setenv("DOCKER_HOST", "unix:///run/custom/docker.sock")
    monkeypatch.setenv("DOCKER_CONTEXT", "custom")
    env = docker.docker_client_env()
    assert env["DOCKER_HOST"] == "unix:///run/custom/docker.sock"
    assert env["DOCKER_CONTEXT"] == "custom"


@pytest.mark.docker
def test_a_session_uses_the_daemon_its_container_was_created_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.conftest import docker_command, docker_image_ready

    if docker_command() is None or not docker_image_ready():
        pytest.skip("Docker or the r2s-python image is not available")
    from real2simlab.sandbox.base import Limits
    from real2simlab.sandbox.docker import DockerBackend

    socket = "/var/run/docker.sock"
    if not Path(socket).exists():
        pytest.skip("the default Docker socket is not at /var/run/docker.sock")
    monkeypatch.setenv("DOCKER_HOST", f"unix://{socket}")
    spec = SandboxSpec(
        image="ghcr.io/real2simhub/r2s-python:1.0", argv=(), workdir="/tmp",
        limits=Limits(timeout_s=60), label="r2s-test-session",
    )  # fmt: skip
    with DockerBackend().session(spec) as session:
        assert session.env["DOCKER_HOST"] == f"unix://{socket}"
        result = subprocess.run(
            [*session.prefix, "echo ok"], env=dict(session.env), capture_output=True,
            text=True, check=False,
        )  # fmt: skip
    assert result.returncode == 0 and result.stdout.strip() == "ok"


# --- 08 root: what the agent's user must read --------------------------------------------------


def test_what_a_fake_agent_mounts_is_readable_by_another_user() -> None:
    from real2simlab.agents.base import unreadable_by_others
    from real2simlab.agents.fake import FakeAgent

    assert unreadable_by_others(FakeAgent().bundle()) == []


def test_unreadable_mounts_are_found(tmp_path: Path) -> None:
    from real2simlab.agents.base import unreadable_by_others

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    (tmp_path / "secret.txt").write_text("x")
    (tmp_path / "secret.txt").chmod(0o600)
    tmp_path.chmod(0o755)
    found = unreadable_by_others(tmp_path)
    assert any("private" in p for p in found) and any("secret.txt" in p for p in found)


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a non-root user")
def test_what_the_lab_copies_from_a_read_only_installation_stays_the_users(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Installed read-only (a Nix store, a read-only share: directories 0555, files 0444), the
    Lab copied its own files into the user's output with those modes: an image's build context
    could not take the SDK, and task, run and export directories could not be replaced with
    ``--overwrite`` or removed with ``rm -rf`` (A11-04). A path the user cannot access ends in
    an error with a fix."""
    from real2simlab import envs
    from real2simlab.benchmarks.synthetic import programs
    from real2simlab.bridges import harbor
    from real2simlab.cli.main import main

    def installed(source: Path, name: str) -> Path:
        copy = tmp_path / "installed" / name
        shutil.copytree(source, copy, ignore=shutil.ignore_patterns("__pycache__"))
        for path in sorted([copy, *copy.rglob("*")], reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        return copy

    dockerfiles = installed(envs.dockerfiles_root(), "docker")
    monkeypatch.setattr(envs, "dockerfiles_root", lambda: dockerfiles)
    monkeypatch.setattr(programs, "PHYSICS_DIR", installed(programs.PHYSICS_DIR, "physics"))
    monkeypatch.setattr(harbor, "PACKAGE", installed(Path(envs.__file__).parent, "real2simlab"))
    out = tmp_path / "out"
    assert (envs.build_context("r2s-python", out / "context") / "r2s_twin.py").is_file()
    programs._common(out / "program", {"phenomenon": "pendulum"})
    harbor.copy_lab(out / "lab")
    for path in [out, *out.rglob("*")]:
        assert os.access(path, os.R_OK | os.W_OK | (os.X_OK if path.is_dir() else 0)), path
    shutil.rmtree(out)  # what rm -rf and --overwrite do

    locked = tmp_path / "locked"
    (locked / "task").mkdir(parents=True)
    locked.chmod(0)
    try:
        code = main(["task", "show", str(locked / "task")])
    finally:
        locked.chmod(0o755)
    error = capsys.readouterr().err
    assert code == 1 and "permission denied" in error and "\n  fix: " in error, error


# --- 09 results that are not one complete set ----------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("damage", ["file missing", "file replaced", "run interrupted"])
def test_an_incomplete_run_is_neither_exported_nor_reported_as_complete(
    train_task: Path, tmp_path: Path, damage: str
) -> None:
    from real2simlab.report.build import ReportBuilder

    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    run = tmp_path / "run"
    _run(_config([tasks], [{"name": "oracle"}, {"name": "static"}]), run)
    assert incomplete(run) == []
    first = sorted(RunLayout(run).result_files())[0]
    if damage == "file missing":
        first.unlink()
    elif damage == "file replaced":
        data = json.loads(first.read_text())
        data["summary"]["reproduction_skill"] = 0.0
        first.write_text(json.dumps(data))
    else:
        RunLayout(run).status_file.write_text(json.dumps({"status": "running"}))
    assert incomplete(run)
    with pytest.raises(SetupError, match="not one complete set"):
        export_dataset(tmp_path / "data", tasks=[tasks], runs=[run])
    builder = ReportBuilder(run)
    builder.build()
    assert any("not one complete set" in w for w in builder.warnings)
    assert "Warning" in (run / "report" / "report.md").read_text()


# --- 10 preflight checks only what will run -----------------------------------------------------


def _fourdcodebench_case_downloaded(case: str) -> bool:
    """Whether a 4DCodeBench synthetic case is in the Lab's cache (``R2S_CACHE_DIR`` too):
    tests never download it, like those of ``tests/test_fourdcodebench.py``."""
    from real2simlab.benchmarks.fourdcodebench import data_dir

    return (data_dir() / "synthetic" / case / "world" / "camera.json").is_file()


@pytest.mark.slow
@pytest.mark.skipif(
    not _fourdcodebench_case_downloaded("B_30"), reason="4DCodeBench B_30 not downloaded"
)
@pytest.mark.skipif(
    not all(importlib.util.find_spec(m) for m in ("huggingface_hub", "h5py", "hdf5plugin")),
    reason="the fourdcodebench extra is not installed: pip install -e '.[fourdcodebench]'",
)
def test_a_task_no_method_applies_to_does_not_block_the_others(
    task_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review's case: the static baseline on a Synthetic task and on 4DCodeBench's B_30,
    whose scorer image is missing. Static cannot deliver B_30's world, so B_30 is skipped as
    not applicable and its scorer is never needed."""
    from real2simlab.benchmarks.fourdcodebench import build_case
    from real2simlab.benchmarks.fourdcodebench.scorer import FourDCodeBenchScorer

    world = build_case("B_30", "synthetic", "native", tmp_path / "4d")
    monkeypatch.setattr(FourDCodeBenchScorer, "problems", lambda self: ["its image is missing"])
    config = _config([task_copy, world], [{"name": "static"}])
    results, experiment = _run(config, tmp_path / "run")
    assert [r["task"] for r in results.to_json()["rows"]] == [load_id(task_copy)]
    assert any("b_30" in s and "not applicable" in s for s in experiment.progress.skipped)
    with pytest.raises(SetupError, match="its image is missing"):  # B_30 run for real: checked
        _run(_config([world], [{"name": "fake_agent"}]), tmp_path / "run2")


def load_id(task: Path) -> str:
    return read_task_config(task / "task.toml").id


# --- 11 a report next to a task that changed -------------------------------------------------------


@pytest.mark.slow
def test_a_report_never_shows_a_new_task_version_next_to_old_scores(
    task_copy: Path, tmp_path: Path
) -> None:
    from real2simlab.report.build import ReportBuilder

    _run(_config([task_copy], [{"name": "oracle"}]), tmp_path / "run")
    video = next((task_copy / "visible").rglob("*.mp4"))
    video.write_bytes(video.read_bytes() + b"\0")  # a new version of the task, the same id
    builder = ReportBuilder(tmp_path / "run")
    builder.build()
    assert any("has changed since it was scored" in w for w in builder.warnings)
    (page,) = (tmp_path / "run" / "report" / "tasks").glob("*.html")
    assert "observed.mp4" not in page.read_text()
    assert "has changed since it was scored" in page.read_text()


# --- 13 the held-out prints of what was trained on, and nothing else ------------------------------


@pytest.mark.slow
def test_prints_hold_only_the_tasks_that_went_into_the_dataset(tmp_path: Path) -> None:
    from real2simlab.methods.training import DATASET_PRINTS
    from real2simlab.task.overlap import OverlapIndex

    a, b = SyntheticBenchmark().build(
        tmp_path / "tasks", select=["bouncing_ball"], tiers=["calibrated"], split="train",
        count=2, seed=4,
    )  # fmt: skip
    high = _run(_config([a], [{"name": "oracle", "label": "m"}]), tmp_path / "high")
    low = _run(_config([b], [{"name": "static", "label": "m"}]), tmp_path / "low")
    assert high and low
    summary = export_dataset(
        tmp_path / "data", runs=[tmp_path / "high", tmp_path / "low"], min_reward=0.5
    )
    assert summary.runs == 1
    printed = {p.task_id for p in OverlapIndex.load(tmp_path / "data" / DATASET_PRINTS).prints}
    assert printed == {load_id(a)}


# --- 14 replacing a published directory, interrupted ---------------------------------------------


def _export_record(text: str) -> str:
    """An ``export.json`` as ``r2s train export`` writes it, telling versions apart by
    ``command``."""
    counts = {"tasks": 0, "runs": 0, "programs": 0}
    return json.dumps(
        {"lab_version": "1.0.0", "command": text, "tasks": [], "runs": [], "against": [],
         "checked_against": 0, "against_digest": None, "counts": counts, "overlapping": [],
         "skipped": [], "content": {}}
    )  # fmt: skip


def _publish(out: Path, text: str) -> None:
    staging = staging_dir(out)
    (staging / "export.json").write_text(_export_record(text))
    replace_dir(staging, out, overwrite=True, marker="export.json", what="an exported dataset")


def _version(out: Path) -> str:
    return json.loads((out / "export.json").read_text())["command"]


def test_replacing_a_dataset_swaps_it_in_one_step(tmp_path: Path) -> None:
    out = tmp_path / "data"
    _publish(out, "one")
    _publish(out, "two")
    assert _version(out) == "two"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data"]  # nothing left behind


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("atomic", [True, False])
def test_a_replacement_killed_midway_leaves_a_complete_version(
    tmp_path: Path, atomic: bool
) -> None:
    """The process is killed (SIGKILL: no cleanup runs) at the worst moment. With an atomic
    swap, right after the swap: the new version is in place. Without one (the fallback),
    right after the old version was moved aside: the tools put it back on next use."""
    out = tmp_path / "data"
    _publish(out, "one")
    two = _export_record("two")
    script = f"""
import os, signal
from pathlib import Path
from real2simlab import _fs
out = Path({str(out)!r})
staging = _fs.staging_dir(out)
(staging / "export.json").write_text({two!r})
if {atomic}:
    if not _fs._exchange(staging, out):
        raise SystemExit(5)  # no atomic swap here
    os.kill(os.getpid(), signal.SIGKILL)
_fs._exchange = lambda a, b: False
real = Path.replace
def replace(self, target):
    result = real(self, target)
    os.kill(os.getpid(), signal.SIGKILL)
    return result
Path.replace = replace
_fs.replace_dir(staging, out, overwrite=True, marker="export.json", what="an exported dataset")
"""
    result = subprocess.run([sys.executable, "-c", script], check=False)
    if result.returncode == 5:
        pytest.skip("this system cannot swap two directories atomically")
    assert result.returncode == -signal.SIGKILL
    if atomic:
        assert _version(out) == "two"
        return
    assert not out.exists()  # killed between the two renames
    recover_dir(out, marker="export.json", what="an exported dataset")
    assert _version(out) == "one"


def test_only_what_a_replacement_left_beside_out_is_restored_or_removed(tmp_path: Path) -> None:
    """Directories beside ``--out`` are the user's unless they bear the exact name a
    replacement gives (``.<out>.old-<8 hex digits>``, ``.<out>.trash-<8 hex digits>``), and a
    backup is put back only when it holds the record the Lab wrote: a look-alike is neither
    moved into place (to be replaced later) nor deleted."""
    out = tmp_path / "data"
    mine = tmp_path / ".data.old-mine"  # not a name a replacement gives
    mine.mkdir()
    (mine / "export.json").write_text(_export_record("mine"))
    lookalike = tmp_path / ".data.old-0123abcd"  # the name, but not the Lab's record
    lookalike.mkdir()
    (lookalike / "export.json").write_text('{"lab_version": "some-other-tool"}\n')
    trash = tmp_path / ".data.trash-mine"
    trash.mkdir()
    (trash / "notes.txt").write_text("keep me")
    recover_dir(out, marker="export.json", what="an exported dataset")
    assert not out.exists()
    assert json.loads((mine / "export.json").read_text())["command"] == "mine"
    assert (lookalike / "export.json").is_file()
    assert (trash / "notes.txt").read_text() == "keep me"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_a_dataset_left_half_written_by_a_killed_export_is_removed_by_the_next(
    train_task: Path, tmp_path: Path
) -> None:
    """A3-09: ``r2s train export`` killed outright (kill -9, the out-of-memory killer: no
    cleanup runs) left its ``.<out>.partial-<hex>`` beside ``--out`` for good; running it again,
    with ``--overwrite`` too, left it there (``train fit`` and ``export harbor`` stage the same
    way). The next command that writes the same ``--out`` removes it, but never the directory
    of a command still writing one, nor a directory of the user's."""
    from real2simlab import _fs

    if _fs.machine_tag() is None:
        pytest.skip("no boot id on this system: nothing staged is ever removed")
    out = tmp_path / "data"
    killed_export = (
        "import os, signal, sys\nfrom pathlib import Path\nfrom real2simlab.train import dataset\n"
        "def killed(*args, **kwargs):\n    os.kill(os.getpid(), signal.SIGKILL)\n"
        "dataset.replace_dir = killed  # right before the dataset is moved into place\n"
        "dataset.export_dataset(Path(sys.argv[1]), tasks=[Path(sys.argv[2])])\n"
    )
    killed = subprocess.run(
        [sys.executable, "-c", killed_export, str(out), str(train_task.parent)],
        check=False, stdin=subprocess.DEVNULL, timeout=300,
    )  # fmt: skip
    assert killed.returncode == -signal.SIGKILL
    (left,) = tmp_path.iterdir()
    assert left.name.startswith(".data.partial-") and (left / "export.json").is_file()
    writing = subprocess.Popen(
        [sys.executable, "-c", "import sys\nfrom pathlib import Path\nfrom real2simlab._fs import "
         "staging_dir\nprint(staging_dir(Path(sys.argv[1])).name, flush=True)\nsys.stdin.read()\n",
         str(out)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )  # fmt: skip
    assert writing.stdin is not None and writing.stdout is not None
    try:
        held = writing.stdout.readline().strip()  # another command's, still being written
        mine = tmp_path / ".data.partial-mine"
        mine.mkdir()
        export_dataset(out, tasks=[train_task.parent])
        assert {path.name for path in tmp_path.iterdir()} == {"data", held, mine.name}
    finally:
        writing.stdin.close()
        writing.wait(60)


def _python_image(tmp_path: Path) -> Path:
    """A small image of r2s-python as ``r2s env export r2s-python`` writes it (its manifest
    names the image)."""
    from .test_bwrap import IMAGE, layer, saved_image

    base = layer([("usr", "dir"), ("usr/bin", "dir"), ("usr/bin/python", b"py")])
    return saved_image(tmp_path / "r2s-python.tar", [base], ["PATH=/usr/bin"], tags=[IMAGE])


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_an_unpack_killed_outright_leaves_nothing_the_next_unpack_keeps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A3-09: ``r2s env unpack`` killed outright (kill -9, the out-of-memory killer) left its
    half-extracted copy beside the cache's images for good (up to the image's size, each time)
    and neither the same command nor ``--force`` removed it. The next unpack removes it now,
    but never the copy of an unpack that is still running (one per job on a cluster)."""
    from real2simlab import _fs
    from real2simlab.cli.main import main
    from real2simlab.envs.rootfs import find_rootfs

    from .test_bwrap import IMAGE

    if _fs.machine_tag() is None:
        pytest.skip("no boot id on this system: nothing staged is ever removed")
    cache = tmp_path / "cache"
    monkeypatch.setenv("R2S_CACHE_DIR", str(cache))
    image = _python_image(tmp_path)
    unpacking = (
        "import os, signal, sys\nfrom pathlib import Path\nfrom real2simlab.envs import rootfs\n"
        "real = rootfs._apply_layer\n"
        "def stopped(*args, **kwargs):  # the first layer is being extracted\n"
        "    if sys.argv[2] == 'kill':\n"
        "        os.kill(os.getpid(), signal.SIGKILL)\n"
        "    print('extracting', flush=True)\n"
        "    sys.stdin.read()\n"
        "    return real(*args, **kwargs)\n"
        "rootfs._apply_layer = stopped\n"
        f"rootfs.unpack_tarball(Path(sys.argv[1]), {IMAGE!r})\n"
    )
    env = {**os.environ, "R2S_CACHE_DIR": str(cache)}

    def directories() -> set[str]:
        return {path.name for path in (cache / "rootfs").iterdir() if path.is_dir()}

    killed = subprocess.run(
        [sys.executable, "-c", unpacking, str(image), "kill"],
        env=env, check=False, stdin=subprocess.DEVNULL, timeout=120,
    )  # fmt: skip
    assert killed.returncode == -signal.SIGKILL
    left = directories()
    assert len(left) == 1, left  # its half-extracted copy
    running = subprocess.Popen(
        [sys.executable, "-c", unpacking, str(image), "wait"],
        env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )  # fmt: skip
    assert running.stdin is not None and running.stdout is not None
    try:
        assert running.stdout.readline() == "extracting\n"
        held = directories() - left  # the running unpack's copy
        assert len(held) == 1, held
        assert main(["env", "unpack", "r2s-python", "--from-tar", str(image)]) == 0
        unpacked = find_rootfs(IMAGE)
        assert unpacked is not None
        assert directories() == {unpacked.root.name, *held}
    finally:
        running.stdin.close()
        assert running.wait(120) == 0
    assert directories() == {unpacked.root.name}


def test_staging_left_on_another_machine_or_by_a_user_is_never_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whether the command that staged a directory still runs can be told only on the machine
    it runs on (the lock it holds is that kernel's): a copy staged by another machine (a cache
    or an output on a disk the nodes of a cluster share) is left alone, as is any directory
    whose name is not exactly one the Lab gives."""
    from real2simlab import _fs
    from real2simlab.envs.rootfs import rootfs_dir, unpack_tarball

    from .test_bwrap import IMAGE

    tag = _fs.machine_tag()
    if tag is None:
        pytest.skip("no boot id on this system: nothing staged is ever removed")
    other = "0" * 12 if tag != "0" * 12 else "f" * 12  # another machine's tag
    out = tmp_path / "data"
    elsewhere = tmp_path / f".data.partial-{other}-0123abcd"
    mine = [tmp_path / name for name in (".data.partial-0123abcd", f".data.partial-{tag}-mine")]
    for path in [elsewhere, *mine]:
        path.mkdir()
    staging = _fs.staging_dir(out)
    assert staging.name.startswith(f".data.partial-{tag}-")
    assert all(path.is_dir() for path in [elsewhere, *mine])
    _fs.force_remove(staging)
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    unpacked = unpack_tarball(_python_image(tmp_path), IMAGE)
    beside = rootfs_dir() / f"{unpacked.root.name}.{other}-0123abcd"
    beside.mkdir()
    unpack_tarball(_python_image(tmp_path), IMAGE, force=True)
    assert beside.is_dir()


def test_a_rebuilt_index_never_registers_a_copy_left_while_unpacking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A damaged index is rebuilt from the records in the cache: the copy an unpack killed
    right before its rename left (its record written, the copy not yet in place) is not an
    unpacked image."""
    from real2simlab.envs.rootfs import find_rootfs, rootfs_dir, unpack_tarball

    from .test_bwrap import IMAGE

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    unpacked = unpack_tarball(_python_image(tmp_path), IMAGE)
    left = rootfs_dir() / f"{unpacked.root.name}.0123abcd"
    shutil.copytree(unpacked.root, left, symlinks=True)
    (rootfs_dir() / "index.json").write_text("[")
    found = find_rootfs(IMAGE)
    assert found is not None and found.root == unpacked.root


def test_an_unpacked_image_whose_tree_is_gone_is_named_and_unpacked_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A9R1-06: the unpacked tree removed (a scratch cleaner, a removal stopped midway) while its
    record ``image.json`` stayed. The backend said "not unpacked", with a fix the unpack then
    refused ("was unpacked but is not there any more", fixed by the full reference without
    ``--from-tar``, which asks Docker): a loop only the unmentioned ``--force`` left. The image
    is named damaged now, with ``--force`` and ``--from-tar`` in the fix, and unpacking it again
    extracts it again, with ``--force`` or without."""
    from real2simlab.cli.main import main
    from real2simlab.envs import rootfs
    from real2simlab.envs.rootfs import find_rootfs, unpack_tarball

    from .test_bwrap import IMAGE

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    image = _python_image(tmp_path)
    argv = ["env", "unpack", "r2s-python", "--from-tar", str(image)]
    for again in ([], ["--force"]):
        assert main(argv) == 0
        unpacked = find_rootfs(IMAGE)
        assert unpacked is not None
        shutil.rmtree(unpacked.rootfs)
        with pytest.raises(SetupError, match="is damaged") as raised:
            BwrapBackend()._find(IMAGE)
        fix = raised.value.fix or ""
        assert "r2s env unpack r2s-python --force" in fix and "--from-tar" in fix, fix
        from real2simlab.doctor import rootfs_check

        check = rootfs_check("r2s-python")
        assert not check.ok and check.fix == fix, check
        capsys.readouterr()
        assert main([*argv, *again]) == 0
        assert capsys.readouterr().out.startswith(f"unpacked {IMAGE} ")
        repaired = find_rootfs(IMAGE)
        assert repaired is not None and (repaired.rootfs / "usr/bin/python").read_bytes() == b"py"
    # Removed by another command between the unpack and its check: the same command again
    # extracts it again (with the source the user gave, which only that command knows).
    monkeypatch.setattr(rootfs, "find_rootfs", lambda reference: None)
    with pytest.raises(SetupError, match="not there any more") as raised:
        unpack_tarball(image, IMAGE, force=True)
    assert raised.value.fix == "run the same command again: it extracts the image again"


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS and /proc/self/status")
def test_unpacking_an_image_holds_no_layer_in_memory(tmp_path: Path) -> None:
    """A9R1-04: every file of a layer was read into memory before any was written (unpacking
    r2s-blender-4.0 took 1.4 GiB), so with less memory (``ulimit -v``, a batch job's limit) the
    unpack ended in a MemoryError traceback. Here the unpack may use 48 MiB of address space
    more than a small image's unpack did, and the image's layer holds a file of 128 MiB."""
    import hashlib
    import io
    import tarfile

    big = tmp_path / "big.bin"
    with big.open("wb") as stream:
        stream.truncate(128 << 20)
    layer = tmp_path / "layer.tar"
    with tarfile.open(layer, "w") as tar:
        tar.add(big, arcname="big.bin")
    big.unlink()
    with layer.open("rb") as stream:
        diff_id = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    image = tmp_path / "big.tar"
    with tarfile.open(image, "w") as tar:
        for name, data in (
            ("config.json", json.dumps({"rootfs": {"diff_ids": [diff_id]}, "config": {}})),
            ("manifest.json", json.dumps([{"Config": "config.json", "Layers": ["layer.tar"],
                                           "RepoTags": ["example.org/qa/big:1"]}])),
        ):  # fmt: skip
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data.encode()))
        tar.add(layer, arcname="layer.tar")
    layer.unlink()
    unpacking = (
        "import resource, sys\nfrom real2simlab.cli.main import main\n"
        "small, big = sys.argv[1:]\n"
        "if main(['env', 'unpack', 'r2s-python', '--from-tar', small]) != 0:\n"
        "    sys.exit(5)\n"
        "size = int(open('/proc/self/status').read().split('VmSize:')[1].split()[0]) << 10\n"
        "resource.setrlimit(resource.RLIMIT_AS, (size + (48 << 20), resource.RLIM_INFINITY))\n"
        "sys.exit(main(['env', 'unpack', 'example.org/qa/big:1', '--from-tar', big]))\n"
    )
    cache = tmp_path / "cache"
    try:
        done = subprocess.run(
            [sys.executable, "-c", unpacking, str(_python_image(tmp_path)), str(image)],
            env={**os.environ, "R2S_CACHE_DIR": str(cache)}, capture_output=True, text=True,
            check=False, stdin=subprocess.DEVNULL, timeout=600,
        )  # fmt: skip
        assert done.returncode == 0, done.stderr[-3000:]
        (unpacked,) = (cache / "rootfs").glob("example.org_qa_big_1-*/rootfs/big.bin")
        assert unpacked.stat().st_size == 128 << 20
    finally:
        shutil.rmtree(cache, ignore_errors=True)
        image.unlink()


SDK = b'"""The twin SDK."""\n\nimport argparse\n' + b"FRAME = 1  # what a twin writes\n" * 400


def _layer_of(files: list[tuple[str, bytes]]) -> bytes:
    """A layer's contents: a tar archive (ustar, no PAX headers) of ``files``."""
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data in files:
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _saved_python(
    path: Path,
    layers: list[bytes],
    *,
    oci: bool = False,
    compress: Callable[[bytes], bytes] | None = None,
    listed: list[bytes] | None = None,
) -> Path:
    """An image of r2s-python as ``docker save`` writes it, of ``layers`` (their contents),
    each stored compressed by ``compress`` (or not), under its digest (``oci``: the layout of
    Docker 25 and later) or as ``layerN.tar``. Its configuration lists the digests of
    ``listed``: by default ``layers``; a changed layer of a copy lists the one that was saved."""
    import hashlib
    import io
    import tarfile

    from .test_bwrap import IMAGE

    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    stored = [compress(data) if compress else data for data in layers]
    diff_ids = ["sha256:" + digest(data) for data in (layers if listed is None else listed)]
    config = json.dumps(
        {"config": {"Env": ["PATH=/usr/bin"]}, "rootfs": {"type": "layers", "diff_ids": diff_ids}}
    ).encode()
    names = [f"blobs/sha256/{digest(d)}" if oci else f"layer{i}.tar" for i, d in enumerate(stored)]
    config_name = f"blobs/sha256/{digest(config)}" if oci else "config.json"
    manifest = [{"Config": config_name, "RepoTags": [IMAGE], "Layers": names}]
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data in [(config_name, config), *zip(names, stored, strict=True), ("manifest.json",
                           json.dumps(manifest).encode())]:  # fmt: skip
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def _change_byte(tarball: Path, member: str, offset: int, value: bytes) -> None:
    """Write ``value`` over the byte at ``offset`` of ``member``'s data in ``tarball``, as a
    copy that went wrong would."""
    import tarfile

    with tarfile.open(tarball) as tar:
        start = tar.getmember(member).offset_data
    with tarball.open("r+b") as stream:
        stream.seek(start + offset)
        stream.write(value)


def _python_layers() -> tuple[bytes, bytes]:
    base = _layer_of([("usr/bin/python", b"py"), ("usr/lib/libpython.so", b"lib" * 300)])
    top = _layer_of([("opt/r2s/sdk/r2s_twin.py", SDK), ("opt/r2s/sdk/units.py", b"M = 1\n" * 90),
                     ("usr/local/bin/pip", b"pip")])  # fmt: skip
    return base, top


def _damaged_python(path: Path, damage: str) -> Path:
    """A copy of an image of r2s-python whose second layer changed after it was saved: a byte of
    a file's data (``data``; ``oci``: stored under its digest), a byte of an entry's header
    (``header``: the entries after it are not read), or the data of a compressed layer
    (``compressed``: a whole archive, compressed as it was, of other contents)."""
    import gzip
    import io
    import tarfile

    base, top = _python_layers()
    if damage == "compressed":
        changed = top.replace(b"import argparse", b"Xmport argparse")
        return _saved_python(path, [base, changed], compress=gzip.compress, listed=[base, top])
    _saved_python(path, [base, top], oci=damage == "oci")
    with tarfile.open(path) as tar:
        manifest = tar.extractfile("manifest.json")
        assert manifest is not None
        member = json.loads(manifest.read())[0]["Layers"][1]
    if damage == "header":  # the name in the header of the layer's second entry
        with tarfile.open(fileobj=io.BytesIO(top)) as layer:
            offset = layer.getmembers()[1].offset
        _change_byte(path, member, offset, b"X")
    else:  # "import argparse" becomes "Xmport argparse"
        _change_byte(path, member, top.index(b"import argparse"), b"X")
    return path


def _nothing_registered(cache: Path) -> None:
    rootfs = cache / "rootfs"
    index = rootfs / "index.json"
    assert not index.exists() or json.loads(index.read_text()) == {}, index.read_text()
    left = [path.name for path in rootfs.iterdir() if path.is_dir()] if rootfs.exists() else []
    assert left == [], left  # neither an unpacked image nor its working copy


@pytest.mark.parametrize("damage", ["data", "oci", "header", "compressed"])
def test_a_damaged_copy_of_an_image_is_refused_by_its_layers_digests(
    damage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A10-01: ``r2s env unpack --from-tar`` never checked a layer against the digest its image
    lists: a copy with one byte changed in a layer (the twin SDK's ``import argparse`` became
    ``Xmport argparse``, or a coordinate frame's axis another) was unpacked, exit 0, under the
    id of the image saved, and a correct program then failed (``infra_error``) or scored 0.000
    under that id; a changed byte in an entry's header silently left the entries after it out.
    A copy whose layer does not match its digest is refused now, naming the layer, with how to
    get a whole one, and nothing is registered or left behind."""
    from real2simlab.cli.main import main
    from real2simlab.envs.rootfs import find_rootfs

    from .test_bwrap import IMAGE

    cache = tmp_path / "cache"
    monkeypatch.setenv("R2S_CACHE_DIR", str(cache))
    tarball = _damaged_python(tmp_path / "r2s-python.tar", damage)
    assert main(["env", "unpack", "r2s-python", "--from-tar", str(tarball)]) == 1
    err = capsys.readouterr().err
    assert f"error: {tarball} is damaged: layer 2 of 2 (" in err, err
    assert "does not match the digest its image lists" in err and "Traceback" not in err, err
    assert "fix: copy the file again" in err and "r2s env export" in err, err
    assert find_rootfs(IMAGE) is None
    _nothing_registered(cache)


def test_a_layer_whose_entries_are_read_only_in_part_is_never_unpacked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A10-01: an entry's header that tarfile cannot read (after the first) ended a layer's
    entries as quietly as the archive's end does, so the entries after it were left out of the
    unpacked image (1175 of one layer of r2s-python), exit 0. Even where the image lists that
    very layer (its digest matches), the layer is read to its end now: entries that could not
    be read are an error, not a smaller image."""
    import io
    import tarfile

    from real2simlab.cli.main import main

    cache = tmp_path / "cache"
    monkeypatch.setenv("R2S_CACHE_DIR", str(cache))
    base, top = _python_layers()
    with tarfile.open(fileobj=io.BytesIO(top)) as layer:
        offset = layer.getmembers()[1].offset
    broken = top[:offset] + b"X" + top[offset + 1 :]
    tarball = _saved_python(tmp_path / "r2s-python.tar", [base, broken])
    assert main(["env", "unpack", "r2s-python", "--from-tar", str(tarball)]) == 1
    err = capsys.readouterr().err
    assert f"layer 2 of 2 (layer1.tar) has entries after byte {offset} of its archive" in err, err
    assert "cannot be read" in err and "\n  fix: " in err and "Traceback" not in err, err
    _nothing_registered(cache)


@pytest.mark.parametrize("source", ["url", "docker"])
def test_every_way_to_unpack_an_image_checks_its_layers(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A10-01, the class: ``--from-url`` checks the download against a checksum, which says the
    file arrived as published, not that its layers are the image's (a checksum computed on a
    damaged file); ``r2s env unpack`` with Docker unpacks what ``docker save`` wrote. Both
    unpack through the same check as ``--from-tar``."""
    import functools
    import hashlib
    import http.server
    import threading

    from real2simlab.cli.main import main
    from real2simlab.envs.rootfs import unpack_image

    from .test_bwrap import IMAGE

    cache = tmp_path / "cache"
    monkeypatch.setenv("R2S_CACHE_DIR", str(cache))
    served = tmp_path / "served"
    served.mkdir()
    tarball = _damaged_python(served / "r2s-python.tar", "data")
    if source == "docker":  # a `docker` that saves the damaged tarball
        docker = tmp_path / "docker"
        docker.write_text(
            '#!/bin/sh\nwhile [ $# -gt 0 ]; do [ "$1" = --output ] && out=$2; shift; done\n'
            f'cp {shlex.quote(str(tarball))} "$out"\n'
        )
        docker.chmod(0o755)
        with pytest.raises(SetupError, match="does not match the digest its image lists"):
            unpack_image(IMAGE, str(docker))
        _nothing_registered(cache)
        return
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Quiet, directory=str(served))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/{tarball.name}"
        digest = hashlib.sha256(tarball.read_bytes()).hexdigest()  # the file as published
        argv = ["env", "unpack", "r2s-python", "--from-url", url, "--sha256", digest]
        assert main(argv) == 1
    finally:
        server.shutdown()
        server.server_close()
    err = capsys.readouterr().err
    assert "layer 2 of 2 (layer1.tar) does not match the digest its image lists" in err, err
    assert "Traceback" not in err, err
    rootfs = cache / "rootfs"
    assert not rootfs.exists() or not [p for p in rootfs.iterdir() if p.is_dir()]


@pytest.mark.parametrize("layout", ["docker save", "oci", "gzip", "bzip2", "xz"])
def test_an_intact_image_unpacks_in_every_layout(
    layout: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check of A10-01 takes every whole image: layers stored as ``docker save`` wrote them
    before Docker 25, under their digests (Docker 25 and later), or compressed (an image saved
    from a store that keeps them so): the digest an image lists is of a layer's uncompressed
    contents, read to their end."""
    import bz2
    import gzip
    import lzma

    from real2simlab.envs.rootfs import unpack_tarball

    from .test_bwrap import IMAGE

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    compress = {"gzip": gzip.compress, "bzip2": bz2.compress, "xz": lzma.compress}.get(layout)
    tarball = _saved_python(
        tmp_path / "r2s-python.tar", list(_python_layers()), oci=layout == "oci", compress=compress
    )
    unpacked = unpack_tarball(tarball, IMAGE)
    assert (unpacked.rootfs / "opt/r2s/sdk/r2s_twin.py").read_bytes() == SDK
    assert (unpacked.rootfs / "usr/local/bin/pip").read_bytes() == b"pip"
    assert (unpacked.rootfs / "usr/lib/libpython.so").read_bytes() == b"lib" * 300


@pytest.mark.parametrize("raised", ["MemoryError", "ENOMEM"])
def test_running_out_of_memory_is_an_error_with_its_fix_not_a_bug_report(
    raised: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A9R1-04: a MemoryError (this machine's memory, or a limit on it: ``ulimit -v``, a batch
    job's) ended any command in a traceback and "This is a bug in Real2SimLab"; an ``OSError``
    of ``ENOMEM`` (a fork or a mapping the limit refused) was a refusal of "this machine's
    system", with a fix that did not name memory."""
    import errno

    from real2simlab.cli import twins
    from real2simlab.cli.main import main

    def exhausted(args: object) -> int:
        if raised == "ENOMEM":
            raise OSError(errno.ENOMEM, os.strerror(errno.ENOMEM))
        raise MemoryError

    monkeypatch.setattr(twins, "_env_unpack", exhausted)
    assert main(["env", "unpack", "r2s-python", "--from-tar", "r2s-python.tar"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "out of memory" in err and "\n  fix: " in err, err
    assert "memory" in err.split("\n  fix: ")[1], err
    assert "Traceback" not in err and "This is a bug" not in err, err


@pytest.mark.parametrize("raised", ["MemoryError", "ENOMEM"])
def test_memory_running_out_is_said_before_anything_is_imported(
    raised: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Memory that ran out is said before ``_explain()``, the one exit for every other error,
    which imports what explains them: with no memory left, that import fails and turns the
    error into a bug report. With every import failing from the moment memory ran out, the
    command still ends in the one error line about memory, with its fix."""
    import builtins
    import errno

    from real2simlab.cli import twins
    from real2simlab.cli.main import main

    importing = builtins.__import__
    out_of_memory = False

    def no_memory_left(*args: Any, **kwargs: Any) -> Any:
        if out_of_memory:
            raise MemoryError
        return importing(*args, **kwargs)

    def exhausted(args: object) -> int:
        nonlocal out_of_memory
        out_of_memory = True
        if raised == "ENOMEM":
            raise OSError(errno.ENOMEM, os.strerror(errno.ENOMEM))
        raise MemoryError

    monkeypatch.setattr(twins, "_env_unpack", exhausted)
    monkeypatch.setattr(builtins, "__import__", no_memory_left)
    try:
        code = main(["env", "unpack", "r2s-python", "--from-tar", "r2s-python.tar"])
    finally:
        out_of_memory = False
    err = capsys.readouterr().err
    assert code == 1 and err.startswith("error: ") and "out of memory" in err, err
    assert "Traceback" not in err and "This is a bug" not in err, err


def test_what_a_command_builds_beside_its_target_is_held_for_the_next_one_to_judge() -> None:
    """A3-09, as a class: a command builds a large result beside its target under a temporary
    name (an unpacked image, a dataset, a checkpoint, a download, a clone, a cache entry) and
    removes it when it fails, but one killed outright (kill -9, the out-of-memory killer) removes
    nothing, and nothing removed what it left. Every such name now comes from ``_fs.held_path``
    (held by a lock that the kernel drops when the process ends, however it ends), so that the
    next command removes what no process holds: never ``tempfile.mkdtemp(dir=...)`` beside the
    target, nor a ``.partial`` name of the process's own making. ``atomic_write_bytes`` keeps its
    own: a record written in one call, whose directory a scan at every write would cost more
    than the bytes it can leave."""
    import ast

    import real2simlab

    allowed = {("_fs.py", "atomic_write_bytes")}
    package = Path(real2simlab.__file__).parent
    found: list[str] = []
    for path in sorted(package.rglob("*.py")):
        name = path.relative_to(package).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stack: list[tuple[ast.AST, str, bool]] = [(tree, "", False)]
        while stack:
            node, function, held = stack.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                function = node.name
            if isinstance(node, ast.Call):
                func = node.func
                callee = (
                    func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else ""
                )  # fmt: skip
                if callee == "mkdtemp" and any(keyword.arg == "dir" for keyword in node.keywords):
                    found.append(f"{name}:{node.lineno} ({function}): mkdtemp(dir=...)")
                held = held or callee in {"held_path", "remove_abandoned"}
            if isinstance(node, ast.JoinedStr) and not held and (name, function) not in allowed:
                text = "".join(
                    part.value
                    for part in node.values
                    if isinstance(part, ast.Constant) and isinstance(part.value, str)
                )
                if ".partial" in text:
                    found.append(f"{name}:{node.lineno} ({function}): {text!r}")
            stack.extend((child, function, held) for child in ast.iter_child_nodes(node))
    assert not found, "\n".join(sorted(found))


# --- 15 a task changed after it was validated ----------------------------------------------------


DECOY = "R2S-TEST-DECOY build copy"


def _build_after_validation(
    task_dir: Path, tmp_path: Path, change: Callable[[], object]
) -> tuple[Any, Path, list[Any], Experiment]:
    """Validate ``task_dir`` as every evaluation validates its tasks (``resolve_tasks``), make
    ``change`` to it, then build it once with a method that records the context it is given:
    the outcome, the build directory, the contexts and the experiment."""
    from real2simlab.methods.base import BuildContext, BuildOutcome, Method
    from real2simlab.orchestrate.experiment import Unit

    given: list[BuildContext] = []

    class Witness(Method):
        name = "witness"

        def build(self, context: BuildContext) -> BuildOutcome:
            given.append(context)
            return BuildOutcome(None, "delivered nothing")

    config = _config([task_dir], [{"name": "static"}])
    backend = LocalBackend(quiet=True)
    experiment = Experiment(config, tmp_path / "run", backend=backend, log=lambda line: None)
    (task,) = experiment.resolve_tasks()
    change()
    build = tmp_path / "build"
    build.mkdir()
    unit = Unit(task, config.methods[0], Witness(), 0)
    outcome = experiment._attempt_build(unit, backend, build)  # pyright: ignore[reportPrivateUsage]
    return outcome, build, given, experiment


def _paths_holding(directory: Path, text: str) -> list[Path]:
    return [
        p for p in directory.rglob("*") if p.is_file() and text in p.read_text(errors="replace")
    ]


@pytest.mark.parametrize(
    "change", ["link to hidden", "copy of hidden", "statement copying hidden", "statement link"]
)
def test_a_task_changed_after_validation_never_reaches_the_method(
    change: str, task_copy: Path, tmp_path: Path
) -> None:
    """A method gets the task as it was validated: its ``visible/`` and its statement. A link
    that appears later (an editor, a sync, another process) is not followed into ``hidden/``,
    and a hidden file copied in later is not given either, in ``visible/`` or as the
    ``instruction.md`` that replaces the statement: the build is not a result (``infra_error``,
    built again next time), and the method is never asked to build."""
    decoy = task_copy / "hidden" / "decoy.txt"
    notes = task_copy / "visible" / "notes.txt"
    statement = task_copy / "instruction.md"

    def make() -> None:
        decoy.write_text(DECOY)
        if change == "link to hidden":
            notes.symlink_to(Path("..") / "hidden" / "decoy.txt")
        elif change == "copy of hidden":
            shutil.copyfile(decoy, notes)
        elif change == "statement copying hidden":
            shutil.copyfile(decoy, statement)
        else:
            statement.symlink_to(Path("hidden") / "decoy.txt")

    outcome, build, given, _ = _build_after_validation(task_copy, tmp_path, make)
    assert given == []
    assert outcome.infra_error is not None
    assert "changed since it was validated" in outcome.infra_error
    assert not outcome.delivered and not (build / "visible").exists()
    assert not _paths_holding(build, DECOY)


def test_a_statement_edited_after_validation_is_not_given(task_copy: Path, tmp_path: Path) -> None:
    """A task with its own statement (``instruction.md``) edited after it was validated: the
    method is not given the new text (not validated) nor the old one (no longer the task's)."""
    statement = task_copy / "instruction.md"
    statement.write_text("Build a twin of the ball in {workspace}.\n")
    outcome, _, given, _ = _build_after_validation(
        task_copy, tmp_path, lambda: statement.write_text(DECOY)
    )
    assert given == [] and outcome.infra_error is not None
    assert f"{statement} changed since it was checked" in outcome.infra_error


@pytest.mark.parametrize("name", ["._notes.txt", ".DS_Store", "cam/Thumbs.db"])
def test_a_file_no_fingerprint_covers_never_reaches_the_method(
    name: str, task_copy: Path, tmp_path: Path
) -> None:
    """A file manager's litter (``.DS_Store``, ``._*``) is not part of the fingerprint a build's
    copy is compared with, so a copy could not tell such a file added after validation: copies
    leave it out. A hidden file copied into ``visible/`` under such a name after validation
    never reaches the method, which builds from the ``visible/`` that was validated."""
    from real2simlab.twin.fingerprint import fingerprint

    decoy = task_copy / "hidden" / "decoy.txt"
    validated = fingerprint(task_copy / "visible")

    def make() -> None:
        decoy.write_text(DECOY)
        shutil.copyfile(decoy, task_copy / "visible" / name)

    outcome, build, given, _ = _build_after_validation(task_copy, tmp_path, make)
    assert outcome.infra_error is None, outcome.infra_error
    (context,) = given
    assert not _paths_holding(context.visible_dir, DECOY) and not _paths_holding(build, DECOY)
    assert fingerprint(context.visible_dir) == validated


def test_the_statement_is_made_from_the_checked_copy(
    task_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The statement a method gets is generated from the copy of ``visible/`` that was checked
    (its ``capture.json``), never read again from the task directory: a change made to the task
    right after the copy was checked does not reach the statement."""
    import real2simlab.orchestrate.experiment as experiment_module

    checked_copy = experiment_module.copy_exactly
    capture_file = task_copy / "visible" / "capture.json"

    def copy_then_change(*args: Any, **kwargs: Any) -> str | None:
        refused = checked_copy(*args, **kwargs)
        capture = json.loads(capture_file.read_text())
        capture["text"]["scene_description"] = DECOY
        capture_file.write_text(json.dumps(capture))
        return refused

    monkeypatch.setattr(experiment_module, "copy_exactly", copy_then_change)
    outcome, _, given, _ = _build_after_validation(task_copy, tmp_path, lambda: None)
    assert outcome.infra_error is None, outcome.infra_error
    (context,) = given
    assert "A red ball" in context.statement and DECOY not in context.statement


LATIN1 = "Build a twin of the ball in {workspace}; the room was at 21 \xb0C.\n".encode("latin-1")
"""A statement saved as Latin-1, not UTF-8. Validation does not decode it (only a convention
that requires ``instruction.md`` reads it as text), so a task with it is validated."""


def test_statement_not_in_utf8_at_validation_then_visible_changed(
    task_copy: Path, tmp_path: Path
) -> None:
    """An ``instruction.md`` that is not UTF-8 when the task is validated does not cost the task
    its record of ``visible/``: re-saved as UTF-8 after validation, with a hidden file copied
    into ``visible/``, the build is compared with what was validated, and the method gets
    nothing (not the copy)."""
    decoy = task_copy / "hidden" / "decoy.txt"
    decoy.write_text(DECOY)
    statement = task_copy / "instruction.md"
    statement.write_bytes(LATIN1)

    def make() -> None:
        statement.write_text("Build a twin of the ball in {workspace}.\n")
        shutil.copyfile(decoy, task_copy / "visible" / "notes.txt")

    outcome, build, given, _ = _build_after_validation(task_copy, tmp_path, make)
    reached = [c.visible_dir for c in given if _paths_holding(c.visible_dir, DECOY)]
    assert not reached, f"method given hidden data: {reached}; infra_error={outcome.infra_error}"
    assert given == [] and outcome.infra_error is not None
    assert not _paths_holding(build, DECOY)


def test_statement_not_in_utf8_at_validation_then_statement_copies_hidden(
    task_copy: Path, tmp_path: Path
) -> None:
    """The same statement replaced after validation by a copy of a hidden file: compared with
    the bytes that were validated, it is a change, and the method is not given it as its
    statement."""
    decoy = task_copy / "hidden" / "decoy.txt"
    decoy.write_text(DECOY)
    statement = task_copy / "instruction.md"
    statement.write_bytes(LATIN1)
    outcome, _, given, _ = _build_after_validation(
        task_copy, tmp_path, lambda: shutil.copyfile(decoy, statement)
    )
    reached = [c.statement for c in given if DECOY in c.statement]
    assert not reached, f"method's statement is hidden data; infra_error={outcome.infra_error}"
    assert given == [] and outcome.infra_error is not None
    assert f"{statement} changed since it was checked" in outcome.infra_error


def test_the_visible_record_is_kept_whatever_the_statement_holds(
    task_copy: Path, tmp_path: Path
) -> None:
    """What a task holds is recorded as it is validated, ``visible/`` on its own: whatever its
    ``instruction.md`` holds (bytes that are not UTF-8), the record of ``visible/`` is kept, and
    a build of the task changed since is compared with it."""
    from real2simlab.twin.fingerprint import fingerprint

    statement = task_copy / "instruction.md"
    statement.write_bytes(LATIN1)
    validated = fingerprint(task_copy / "visible")
    outcome, _, given, experiment = _build_after_validation(
        task_copy, tmp_path, lambda: statement.write_text("x\n")
    )
    records = {
        Path(root).resolve(): record
        for root, record in experiment._validated.items()  # pyright: ignore[reportPrivateUsage]
    }
    record = records.get(task_copy.resolve())
    assert record is not None, f"no record of the validated task: {sorted(map(str, records))}"
    assert record.visible == validated
    assert given == [] and outcome.infra_error is not None


NOT_UTF8 = b"\xd6\xd0\xce\xc4.txt"
"""A Chinese file name as an archive made on a Chinese Windows writes it (GBK): not UTF-8, so Python
lists it with surrogate escapes (``\\udcd6...``), and only its bytes say what it is called."""


def _add_not_utf8(directory: Path, data: bytes = b"notes\n") -> None:
    """A file named ``NOT_UTF8`` in ``directory``, created by its bytes (as ``unzip`` does)."""
    try:
        with open(os.path.join(os.fsencode(directory), NOT_UTF8), "wb") as file:
            file.write(data)
    except OSError as error:
        if error.errno == errno.EILSEQ:  # APFS (macOS) refuses names that are not UTF-8
            pytest.skip("this file system refuses file names that are not UTF-8")
        raise


def _holds_not_utf8(directory: Path) -> bool:
    return NOT_UTF8 in os.listdir(os.fsencode(directory))


def _validates(task: Path) -> bool:
    from real2simlab.task import validate_task

    report = validate_task(task)
    return report.ok and not report.warnings


def test_a_file_name_that_is_not_utf8_is_content_like_any_other(
    task_copy: Path, tmp_path: Path
) -> None:
    """A file whose name is not UTF-8 is fingerprinted by the bytes of its name, like every
    other file: the task validates, has a content hash, and its exact copy is the checked
    version (not "changed since it was checked", which no number of retries could cure)."""
    from real2simlab.task.io import load_task
    from real2simlab.twin.fingerprint import fingerprint

    visible = task_copy / "visible"
    _add_not_utf8(visible)
    assert _validates(task_copy)
    checked = fingerprint(visible)
    assert load_task(task_copy).content_hash()
    from real2simlab.task.io import copy_exactly

    out = tmp_path / "copy"
    assert copy_exactly(visible, out) is None
    assert _holds_not_utf8(out) and fingerprint(out) == checked
    assert copy_exactly(visible, tmp_path / "checked", expected=checked) is None
    name = os.path.join(os.fsencode(out), NOT_UTF8)
    os.rename(name, name.replace(b"\xce\xc4", b""))
    assert fingerprint(out) != checked  # the name is content: renamed, it is another directory


FINGERPRINT_OF_UTF8_NAMES = "e5a4f5274324b6e2c625933125e8a2628623a6bb6d66969099cb3f81e1290c4e"
"""The fingerprint of the directory below as Real2SimLab 1.0.0 computed it."""


def test_fingerprints_of_utf8_names_are_unchanged(tmp_path: Path) -> None:
    """A name that is UTF-8 counts as the same bytes as before (its UTF-8 encoding): the
    fingerprints of existing tasks and programs, and every cache keyed on them, still hold."""
    from real2simlab.twin.fingerprint import fingerprint

    (tmp_path / "\u8bf4\u660e").mkdir()
    (tmp_path / "\u8bf4\u660e" / "\u4e2d\u6587.txt").write_bytes("\u4e2d\u6587\n".encode())
    (tmp_path / "run.py").write_text("print('hi')\n")
    (tmp_path / "helper.sh").write_text("#!/bin/sh\n")
    (tmp_path / "helper.sh").chmod(0o755)
    assert fingerprint(tmp_path) == FINGERPRINT_OF_UTF8_NAMES


@pytest.mark.slow
def test_a_task_with_a_file_name_that_is_not_utf8_is_evaluated(
    task_copy: Path, tmp_path: Path
) -> None:
    """``r2s task validate`` accepts such a task, and so does ``r2s eval``: the agent's
    workspace holds the file under its own name, and the task is scored (it does not stop the
    command with an encoding error that names no file)."""
    _add_not_utf8(task_copy / "visible")
    assert _validates(task_copy)
    config = _config([task_copy], [{"name": "fake_agent", "mode": "physical"}])
    results, experiment = _run(config, tmp_path / "run")
    (method,) = results.methods
    assert (method.infra_errors, experiment.progress.runs) == (0, 1)
    assert method.reproduction_skill.mean is not None
    (workspace,) = (tmp_path / "run" / "builds").rglob("workspace/visible")
    assert _holds_not_utf8(workspace)


@pytest.mark.parametrize("export", ["harbor", "dataset"])
def test_a_task_with_a_file_name_that_is_not_utf8_is_exported(
    task_copy: Path, train_task: Path, tmp_path: Path, export: str
) -> None:
    """``r2s export harbor`` and ``r2s train export`` copy such a task exactly, the file under
    its own name, instead of saying the task changed while it was exported (and to export it
    again, which failed the same way every time)."""
    if export == "harbor":
        from real2simlab.bridges.harbor import export_task

        _add_not_utf8(task_copy / "visible")
        (task_copy / "notes").mkdir()
        _add_not_utf8(task_copy / "notes")
        dest = export_task(task_copy, tmp_path / "harbor")
        task = dest / "tests" / "task"
        held = [dest / "environment" / "workspace" / "visible", task / "visible", task / "notes"]
    else:
        from real2simlab.train.dataset import dataset_changes

        task = _copy(train_task, tmp_path / "tasks" / "t")
        _add_not_utf8(task / "visible")
        _add_not_utf8(task / "hidden", b"labels\n")  # not the visible file's content: no leak
        assert export_dataset(tmp_path / "data", tasks=[task]).tasks == 1
        (capture,) = (tmp_path / "data").rglob("visible/capture.json")
        held = [capture.parent, capture.parent.parent / "hidden"]
        assert dataset_changes(tmp_path / "data") == []
    assert all(_holds_not_utf8(path) for path in held), held


def test_no_file_name_is_encoded_as_strict_utf8() -> None:
    """A path is hashed as the bytes the file system holds for it (``os.fsencode``), never as
    ``.as_posix().encode()``: that raises on a name that is not UTF-8, which stopped ``r2s
    eval`` with an error naming no file, and was taken for a change of the task it was in."""
    import ast

    import real2simlab

    found: list[str] = []
    for path in sorted(Path(real2simlab.__file__).parent.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "encode"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Attribute)
                and node.func.value.func.attr == "as_posix"
            ):
                found.append(f"{path.name}:{node.lineno}")
    assert not found, found


def test_a_task_not_recorded_as_validated_is_never_built_from_what_it_holds_later(
    task_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task whose ``visible/`` could not be read as it was validated (a read that failed
    then, though validation passed) has no record its builds can be compared with: they are
    failures of the machine (``infra_error``, built again next time), and the method is never
    given the task as it is by then instead."""
    import real2simlab.orchestrate.experiment as experiment_module

    real = experiment_module.fingerprint
    failed: list[Path] = []

    def fails_once(directory: Path, **options: Any) -> str:
        if not failed:
            failed.append(directory)
            raise OSError(5, "Input/output error (injected)")
        return real(directory, **options)

    monkeypatch.setattr(experiment_module, "fingerprint", fails_once)
    decoy = task_copy / "hidden" / "decoy.txt"

    def make() -> None:
        decoy.write_text(DECOY)
        shutil.copyfile(decoy, task_copy / "visible" / "notes.txt")

    outcome, build, given, _ = _build_after_validation(task_copy, tmp_path, make)
    assert [path.name for path in failed] == ["visible"]
    assert given == [] and outcome.infra_error is not None
    assert "visible/" in outcome.infra_error
    assert not _paths_holding(build, DECOY)


def test_tasks_chosen_by_id_are_given_to_a_method_as_they_were_validated(
    task_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """include and exclude choose tasks by their id before they are validated: a task left out
    (one still being written) is neither validated nor recorded. Each task chosen is recorded
    as it was before it was validated, so every task the run builds has its record, and a
    hidden file copied into ``visible/`` while the task is validated reaches no method."""
    import real2simlab.orchestrate.experiment as experiment_module
    from real2simlab.methods.base import BuildContext, BuildOutcome, Method
    from real2simlab.orchestrate.experiment import Unit

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    chosen = task_copy.rename(tasks / "bouncing")
    skeleton = tasks / "drop_01"  # a task still being written: its id, nothing else
    skeleton.mkdir()
    (skeleton / "task.toml").write_text('id = "mybench/drop_01"\n')
    decoy = chosen / "hidden" / "decoy.txt"
    decoy.write_text(DECOY)
    validated: list[Path] = []
    validate = experiment_module.validate_task

    def validate_then_change(root: Path) -> Any:
        report = validate(root)
        validated.append(root.resolve())
        if root.resolve() == chosen.resolve():  # while it is validated
            shutil.copyfile(decoy, root / "visible" / "notes.txt")
        return report

    monkeypatch.setattr(experiment_module, "validate_task", validate_then_change)
    config = parse_experiment(
        {
            "name": "chosen",
            "tasks": {"paths": [str(tasks)], "include": ["synthetic/*"]},
            "methods": [{"name": "static"}],
            "run": {"backend": "local", "retries": 0},
        }
    )
    backend = LocalBackend(quiet=True)
    experiment = Experiment(config, tmp_path / "run", backend=backend, log=lambda line: None)
    (task,) = experiment.resolve_tasks()
    assert validated == [chosen.resolve()]  # the task left out was not validated
    records = experiment._validated  # pyright: ignore[reportPrivateUsage]
    assert list(records) == [task.root]  # the task chosen has its record, and only it
    given: list[BuildContext] = []

    class Witness(Method):
        name = "witness"

        def build(self, context: BuildContext) -> BuildOutcome:
            given.append(context)
            return BuildOutcome(None, "delivered nothing")

    build = tmp_path / "build"
    build.mkdir()
    unit = Unit(task, config.methods[0], Witness(), 0)
    outcome = experiment._attempt_build(unit, backend, build)  # pyright: ignore[reportPrivateUsage]
    assert given == [] and outcome.infra_error is not None, outcome
    assert "changed since it was validated" in outcome.infra_error
    held = [p for p in build.rglob("*") if p.is_file() and DECOY in p.read_text(errors="replace")]
    assert held == []


# --- Nothing of the user's is ever deleted ---------------------------------------------------


def test_twin_check_clears_only_a_directory_a_check_wrote(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from real2simlab.benchmarks.synthetic.programs import simulation_program
    from real2simlab.cli.main import main

    program = simulation_program(
        tmp_path / "physical", phenomenon="bouncing_ball", title="t",
        params={"restitution": 0.7, "rest_speed": 0.05},
    )  # fmt: skip
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "notes.txt").write_text("keep me")
    args = ["twin", "check", str(program), "--task", str(bouncing_task), "--backend", "local"]
    assert main([*args, "--out", str(mine)]) == 1
    assert (mine / "notes.txt").read_text() == "keep me"
    assert "did not write" in capsys.readouterr().err
    # the twin itself, or a directory that holds it, is never the output
    assert main([*args, "--out", str(program)]) == 1
    assert main([*args, "--out", str(tmp_path)]) == 1
    assert (program / "run.py").is_file()
    # an earlier check's directory is cleared and written again
    out = tmp_path / "check"
    assert main([*args, "--out", str(out)]) == 0
    assert main([*args, "--out", str(out)]) == 0
    assert {p.name for p in out.iterdir()} <= {"twin", "run", "rerun", "check.json", ".r2s-check"}


def test_overwrite_replaces_only_task_directories(tmp_path: Path) -> None:
    from real2simlab.benchmarks.synthetic import build_suite as build

    out = tmp_path / "tasks"
    first = build(out, select=["bouncing_ball"], tiers=["calibrated"])
    root = first[0]
    assert build(out, select=["bouncing_ball"], tiers=["calibrated"], overwrite=True)
    (root / "task.toml").unlink()
    (root / "my-notes.txt").write_text("keep me")
    with pytest.raises(SetupError, match="not part of a task"):
        build(out, select=["bouncing_ball"], tiers=["calibrated"], overwrite=True)
    assert (root / "my-notes.txt").read_text() == "keep me"


def _files(root: Path) -> dict[str, bytes | None]:
    """Everything under ``root``: each file's content, ``None`` for a directory."""
    return {
        path.relative_to(root).as_posix(): None if path.is_dir() else path.read_bytes()
        for path in sorted(root.rglob("*"))
    }


@pytest.mark.parametrize(
    "layout",
    [
        ("visible/my-video-notes.txt", "hidden/"),
        ("instruction.md", "visible/cam/notes.txt", "hidden/reference/answer.csv"),
        ("hidden/measurements.csv",),
    ],
    ids=["visible-and-hidden", "written-by-hand", "hidden-only"],
)
def test_overwrite_never_removes_a_directory_no_build_wrote(
    layout: tuple[str, ...], tmp_path: Path
) -> None:
    """A directory that only bears a task's names (``visible/``, ``hidden/``,
    ``instruction.md``), with no ``task.toml``, is neither a task nor a build the Lab was
    interrupted in (a task being written by hand looks just like it): ``--overwrite`` refuses
    it and every file in it stays."""
    from real2simlab.benchmarks.synthetic import build_suite as build

    out = tmp_path / "tasks"
    root = out / "bouncing_ball" / "calibrated"
    for name in layout:
        if name.endswith("/"):
            (root / name).mkdir(parents=True)
        else:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(f"mine: {name}\n")
    before = _files(root)
    with pytest.raises(SetupError, match="--overwrite replaces only task directories"):
        build(out, select=["bouncing_ball"], tiers=["calibrated"], overwrite=True)
    assert _files(root) == before


def test_a_task_build_interrupted_midway_is_replaced_by_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What an interruption says to do: the same command with ``--overwrite`` replaces the task
    directory the build was writing (it has no ``task.toml`` yet)."""
    from real2simlab.benchmarks.synthetic import build_suite as build
    from real2simlab.task import TaskBuilder, load_task

    def interrupted(self: TaskBuilder, *, validate: bool = True) -> Any:
        raise KeyboardInterrupt

    out = tmp_path / "tasks"
    with monkeypatch.context() as patch:
        patch.setattr(TaskBuilder, "finish", interrupted)
        with pytest.raises(KeyboardInterrupt):
            build(out, select=["bouncing_ball"], tiers=["calibrated"])
    root = out / "bouncing_ball" / "calibrated"
    assert (root / "visible").is_dir() and not (root / "task.toml").exists()
    (path,) = build(out, select=["bouncing_ball"], tiers=["calibrated"], overwrite=True)
    assert load_task(path).id == "synthetic/bouncing_ball/calibrated"


def test_an_overwrite_interrupted_while_it_removes_the_old_task_can_run_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--overwrite`` stopped after it removed the old ``task.toml`` but not the rest: the
    directory is still marked as the build's, so the same command replaces it."""
    from real2simlab import _fs
    from real2simlab.benchmarks.synthetic import build_suite as build
    from real2simlab.task import build as task_build

    out = tmp_path / "tasks"
    (root,) = build(out, select=["bouncing_ball"], tiers=["calibrated"])

    def stopped(path: Path) -> str | None:
        problem = _fs.force_remove(path)
        if path.name == "task.toml":
            raise KeyboardInterrupt
        return problem

    with monkeypatch.context() as patch:
        patch.setattr(task_build, "force_remove", stopped)
        with pytest.raises(KeyboardInterrupt):
            build(out, select=["bouncing_ball"], tiers=["calibrated"], overwrite=True)
    assert not (root / "task.toml").exists()
    assert build(out, select=["bouncing_ball"], tiers=["calibrated"], overwrite=True) == [root]
    assert (root / "task.toml").is_file() and not (root / task_build.BUILD_MARKER).exists()


def test_the_local_backend_cannot_run_a_launcher_only_an_image_has(tmp_path: Path) -> None:
    """PhysCodeGym's scene runner lives in its image: on the local backend that is a failure
    of the machine (not a score of 0), and an experiment refuses it before it starts."""
    from real2simlab.sandbox.local import image_only

    spec = SandboxSpec(
        image="r2s-blender-4.0", argv=("python3", "/opt/r2s/blender_scene.py"),
        workdir=str(tmp_path),
    )  # fmt: skip
    result = LocalBackend(quiet=True).run(spec, tmp_path / "log")
    assert result.exit_code is None and "--backend docker" in (result.infrastructure_error or "")
    assert image_only(("python", "./run.py")) is None


def _as_typed_in_a_terminal() -> None:
    for name in ("SIGINT", "SIGHUP", "SIGTERM"):
        signal.signal(getattr(signal, name), signal.SIG_DFL)


@pytest.mark.parametrize("command", ["eval", "train reward"])
@pytest.mark.parametrize("stop", ["SIGINT", "SIGTERM", "SIGHUP"])
def test_an_interrupted_evaluation_stops_its_program_and_scores_nothing(
    stop: str, command: str, bouncing_task: Path, tmp_path: Path
) -> None:
    """Ctrl-C, ``kill`` or a closed terminal while a frozen re-run is running stops that run at
    once (it runs in its own session, out of the terminal's reach), writes no result for it,
    and exits 130."""
    import time

    program = tmp_path / "slow"
    shutil.copytree(bouncing_task / "hidden" / "generator" / "program", program)
    pid_file = tmp_path / "program.pid"
    run_py = program / "run.py"
    run_py.write_text(
        f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(120)\n" + run_py.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    out = tmp_path / "run"
    argv = [sys.executable, "-m", "real2simlab.cli.main", *{
        "eval": ["eval", "--tasks", str(bouncing_task), "--method", f"program:path={program}",
                 "--out", str(out)],
        "train reward": ["train", "reward", "--task", str(bouncing_task), "--solution",
                         str(program), "--work", str(out)],
    }[command], "--backend", "local"]  # fmt: skip
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # A command typed in a terminal: a shell's background job starts with SIGINT ignored,
        # and one started under nohup (as this test suite may be) with SIGHUP ignored.
        preexec_fn=_as_typed_in_a_terminal,
    )
    deadline = time.monotonic() + 120
    while not pid_file.exists() or not pid_file.read_text():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, "the program never started"
        time.sleep(0.2)
    program_pid = int(pid_file.read_text())
    started = time.monotonic()
    process.send_signal(getattr(signal, stop))
    _, err = process.communicate(timeout=60)
    assert time.monotonic() - started < 20, "the interrupt waited for the program"
    assert process.returncode == 130, err
    assert "interrupted" in err and "Traceback" not in err, err
    with pytest.raises(ProcessLookupError):
        os.kill(program_pid, 0)  # the program was stopped, not left running
    assert not list(out.rglob("seed-*.json")), "a stopped run was scored"
    assert "reward" not in err.replace("train reward", "")


@pytest.mark.parametrize("damage", ["its capture cut short", "its truth edited", "its key"])
def test_a_benchmark_task_changed_in_the_run_directory_is_built_again(
    damage: str, tmp_path: Path
) -> None:
    """The tasks a run builds from a benchmark are the Lab's own copies: one damaged in the run
    directory (a disk, a sync client) or edited there by hand is built again, neither refused
    as invalid with no way out nor evaluated as it is now; so is a selection whose key is not
    text. Built again, it is the same task, so its builds and runs are reused."""
    config = parse_experiment(
        {
            "name": "changed",
            "tasks": {"benchmarks": [{"name": "synthetic", "select": ["bouncing_ball"],
                                      "tiers": ["calibrated"]}]},
            "methods": [{"name": "oracle"}],
            "run": {"backend": "local"},
        }
    )  # fmt: skip
    out = tmp_path / "run"
    backend = LocalBackend(quiet=True)
    Experiment(config, out, backend=backend, log=lambda _: None, report=False).run()
    before = _scores(out)
    (key,) = (out / "tasks" / "synthetic").glob("*/.key")
    task = key.parent / "bouncing_ball" / "calibrated"
    if damage == "its capture cut short":
        (task / "visible" / "capture.json").write_text("{BROKEN")
    elif damage == "its truth edited":  # still valid: the oracle would deliver another program
        model = task / "hidden" / "generator" / "program" / "model.json"
        text = model.read_text()
        edited = text.replace('"restitution": 0.763', '"restitution": 0.5')
        assert edited != text
        model.write_text(edited)
    else:
        key.write_bytes(b"not a key {\x00\xff")
    experiment = Experiment(config, out, backend=backend, log=lambda _: None, report=False)
    experiment.run()
    assert _scores(out) == before
    assert (experiment.progress.builds, experiment.progress.reused_builds) == (0, 1)


def test_a_benchmark_task_removed_by_hand_is_built_again(tmp_path: Path) -> None:
    """Resuming must evaluate the same tasks: a cached selection missing a task is rebuilt,
    not evaluated on what is left."""
    config = parse_experiment(
        {
            "name": "lost",
            "tasks": {"benchmarks": [{"name": "synthetic", "select": ["bouncing_ball", "pendulum"],
                                      "tiers": ["calibrated"]}]},
            "methods": [{"name": "static"}],
            "run": {"backend": "local"},
        }
    )  # fmt: skip
    out = tmp_path / "run"
    first = Experiment(config, out, log=lambda _: None).run()
    assert len(first.rows) == 2
    (removed,) = [p for p in (out / "tasks").rglob("pendulum") if p.is_dir()]
    shutil.rmtree(removed)
    again = Experiment(config, out, log=lambda _: None).run()
    assert sorted(r["task"] for r in again.rows) == sorted(r["task"] for r in first.rows)


def test_a_program_printing_past_the_file_size_limit_gets_its_reward(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """A correct program that prints more than this machine's file-size limit (``ulimit -f``)
    allows in one file: the Lab keeps its log under the limit (the beginning and the end), so
    the limit never stops the program for what it printed, and it gets its reward. It used to
    be killed at the limit, with no reward and the advice to raise the limit, which a program
    printing in a loop then filled (A3-05)."""
    import resource

    program = tmp_path / "chatty"
    shutil.copytree(bouncing_task / "hidden" / "generator" / "program", program)
    with (program / "run.py").open("a", encoding="utf-8") as stream:
        stream.write('\nprint("x" * 200_000, flush=True)\nprint("done", flush=True)\n')
    argv = [sys.executable, "-m", "real2simlab.cli.main", "train", "reward", "--task",
            str(bouncing_task), "--solution", str(program), "--backend", "local",
            "--work", str(tmp_path / "work")]  # fmt: skip
    normal = subprocess.run(argv, capture_output=True, text=True, timeout=300, check=False)
    assert normal.returncode == 0 and "1.0000" in normal.stdout, normal.stdout + normal.stderr

    def limited() -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024, 64 * 1024))

    done = subprocess.run(
        [*argv[:-1], str(tmp_path / "limited")],
        capture_output=True, text=True, timeout=300, check=False, preexec_fn=limited,
    )  # fmt: skip
    assert done.returncode == 0 and "1.0000" in done.stdout, done.stdout + done.stderr
    log = (tmp_path / "limited" / "runs" / "run" / "log.txt").read_bytes()
    assert len(log) < 64 * 1024 and b"bytes of output left out" in log, len(log)
    assert log.endswith(b"done\n"), log[-200:]


def test_a_program_flooding_its_output_under_a_file_size_limit_fails_on_its_own(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """A program that prints in a loop until its time limit, on a machine with a file-size
    limit: the run timed out, the program's own result (a reward of 0), not a failure of the
    machine that is never scored and is retried and resumed for ever (A3-05)."""
    import resource

    task = _copy(bouncing_task, tmp_path / "task")
    data = read_task_config(task / "task.toml").to_json()
    data["delivery"]["run_timeout_s"] = 2
    write_task_config(TaskConfig.model_validate(data), task / "task.toml")
    program = tmp_path / "flood"
    shutil.copytree(task / "hidden" / "generator" / "program", program)
    with (program / "run.py").open("a", encoding="utf-8") as stream:
        stream.write("\nwhile True:\n    print('y' * 1000)\n")

    def limited() -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))

    done = subprocess.run(
        [sys.executable, "-m", "real2simlab.cli.main", "train", "reward", "--task", str(task),
         "--solution", str(program), "--backend", "local", "--work", str(tmp_path / "work")],
        capture_output=True, text=True, timeout=300, check=False, preexec_fn=limited,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    assert "0.0000" in done.stdout and "timed_out" in done.stdout, done.stdout
    log = tmp_path / "work" / "runs" / "run" / "log.txt"
    assert log.stat().st_size < 1024 * 1024


PRINTS_UNTIL_ITS_TIME_LIMIT = """
import json, sys
from pathlib import Path

from real2simlab.sandbox.process import run_logged, storage_problem

log, program = Path(sys.argv[1]), Path(sys.argv[2])
code, timed_out, _ = run_logged([sys.executable, str(program)], log, 1)
said = {"code": code, "timed_out": timed_out, "size": log.stat().st_size,
        "end": log.read_bytes()[-80:].decode("utf-8", "replace"),
        "storage": storage_problem(log, [log.parent])}
print(json.dumps(said))
"""
PRINTS_WITHOUT_END = """import os
while True:  # a write that fails (past a file-size limit) does not stop it
    try:
        os.write(1, b"y" * 999 + b"\\n")
    except OSError:
        pass
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resource limits")
def test_a_program_printing_until_its_time_limit_leaves_its_log_under_the_file_size_limit(
    tmp_path: Path,
) -> None:
    """Under a file-size limit (``ulimit -f``), a program that prints without end until its time
    limit leaves a log under the limit: what it printed first and last, then the Lab's own line
    saying it was stopped. The storage check, which a run that timed out gets too, then finds
    no file at the limit, so the run is the program's own. When the log reached the limit, the
    Lab's own line could not be written and stopped the command (``File too large``), and the
    check took the run for a failure of this machine."""
    limit = 256 * 1024
    log, program = tmp_path / "run" / "log.txt", tmp_path / "prints.py"
    log.parent.mkdir()
    program.write_text(PRINTS_WITHOUT_END)
    done = _in_a_new_process(["-c", PRINTS_UNTIL_ITS_TIME_LIMIT, str(log), str(program)], limit)
    assert done.returncode == 0 and "Traceback" not in done.stderr, done.stderr[-2000:]
    said = json.loads(done.stdout.splitlines()[-1])
    assert (said["code"], said["timed_out"], said["storage"]) == (None, True, None), said
    assert said["size"] < limit and said["end"].endswith("[r2s] killed after 1 s\n"), said
    assert "bytes of output left out here" in log.read_text(encoding="utf-8")


def test_a_write_the_program_survives_at_the_size_limit_is_not_a_reward_of_0(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """The program catches its failed write and exits 0 with no complete output: the
    file-size limit still explains it, so there is no reward (not 0 for invalid output)."""
    import resource

    program = tmp_path / "careful"
    shutil.copytree(bouncing_task / "hidden" / "generator" / "program", program)
    source = (program / "run.py").read_text()
    (program / "run.py").write_text(
        "import os, sys\nfrom pathlib import Path\ntry:\n"
        "    (Path(os.environ['R2S_OUTPUT_DIR']) / 'debug.txt').write_bytes(b'x' * 200_000)\n"
        + "\n".join("    " + line for line in source.splitlines())
        + "\nexcept OSError as error:\n    print('could not save output:', error)\n"
    )
    argv = [sys.executable, "-m", "real2simlab.cli.main", "train", "reward", "--task",
            str(bouncing_task), "--solution", str(program), "--backend", "local",
            "--work", str(tmp_path / "work")]  # fmt: skip
    normal = subprocess.run(argv, capture_output=True, text=True, timeout=300, check=False)
    assert normal.returncode == 0 and "1.0000" in normal.stdout, normal.stdout + normal.stderr

    def limited() -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024, 64 * 1024))

    done = subprocess.run(
        [*argv[:-1], str(tmp_path / "limited")],
        capture_output=True, text=True, timeout=300, check=False, preexec_fn=limited,
    )  # fmt: skip
    log = (tmp_path / "limited" / "runs" / "run" / "log.txt").read_text()
    assert "could not save output" in log and "File too large" in log, log
    assert done.returncode == 1, done.stdout + done.stderr
    assert "file-size limit" in done.stdout + done.stderr, done.stdout + done.stderr
    assert "0.0000" not in done.stdout


SCRATCH_FILES = {
    "output": "open(os.path.join(os.environ['R2S_OUTPUT_DIR'], 'scratch.bin'), 'wb')",
    "temporary": "tempfile.NamedTemporaryFile(delete=False)",  # in TMPDIR, kept
}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resource limits")
@pytest.mark.parametrize(
    ("where", "ending"), [("temporary", "fails"), ("output", "retries"), ("temporary", "retries")]
)
def test_a_write_the_file_size_limit_stops_is_never_a_reward_of_0(
    bouncing_task: Path, tmp_path: Path, where: str, ending: str
) -> None:
    """A correct program writes a scratch file this machine's file-size limit stops, wherever
    it writes it (its output, its temporary directory) and however it ends: it fails on the
    write, or retries it (waiting for room on the disk) until its time limit. The limit
    explains either, so there is no reward, rather than 0 for a failed or timed-out run."""
    import resource

    task = _copy(bouncing_task, tmp_path / "task")
    data = read_task_config(task / "task.toml").to_json()
    data["delivery"]["run_timeout_s"] = 5  # a short time limit, for the program that retries
    write_task_config(TaskConfig.model_validate(data), task / "task.toml")
    write = f"with {SCRATCH_FILES[where]} as scratch:\n    scratch.write(b'x' * 200_000)\n"
    if ending == "retries":
        retried = "".join(f"        {line}\n" for line in write.splitlines())
        write = f"while True:\n    try:\n{retried}        break\n"
        write += "    except OSError:\n        time.sleep(0.2)\n"
    program = _truth_program(bouncing_task, tmp_path / "program", "import os, tempfile, time\n"
                             + write)  # fmt: skip
    argv = [sys.executable, "-m", "real2simlab.cli.main", "train", "reward", "--task",
            str(task), "--solution", str(program), "--backend", "local",
            "--work", str(tmp_path / "work")]  # fmt: skip
    normal = subprocess.run(argv, capture_output=True, text=True, timeout=300, check=False)
    assert normal.returncode == 0 and "1.0000" in normal.stdout, normal.stdout + normal.stderr

    def limited() -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024, 64 * 1024))

    done = subprocess.run(
        [*argv[:-1], str(tmp_path / "limited")],
        capture_output=True, text=True, timeout=300, check=False, preexec_fn=limited,
    )  # fmt: skip
    record = json.loads((tmp_path / "limited" / "runs" / "run" / "run_record.json").read_text())
    assert record["status"] == "infra_error", record
    assert done.returncode == 1, done.stdout + done.stderr
    assert "file-size limit" in done.stdout + done.stderr, done.stdout + done.stderr
    assert "0.0000" not in done.stdout


class _EditedWhileItRuns(LocalBackend):
    """A backend that keeps the program from writing its frozen twin (docker and bwrap mount it
    read-only), here the local one: while each twin runs, something outside its sandbox (an
    editor saving, a sync client) changes the twin's program."""

    isolated: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(quiet=True)

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        result = super().run(spec, log_path)
        if spec.label == "r2s-twin":
            import stat

            program = next(Path(mount.source) for mount in spec.mounts if mount.read_only)
            source = program / "run.py"
            source.chmod(source.stat().st_mode | stat.S_IWUSR)
            with source.open("a", encoding="utf-8") as stream:
                stream.write("\n# saved by an editor while the twin ran\n")
        return result


class _EditedWhileItRunsLocally(_EditedWhileItRuns):
    """The same, where nothing keeps the program from writing its twin itself."""

    isolated: ClassVar[bool] = False


def test_a_frozen_twin_changed_from_outside_while_it_runs_gives_no_reward(
    train_task: Path, tmp_path: Path
) -> None:
    """Where the program cannot write its frozen twin, a change while it ran came from outside
    the sandbox: the run is not the program's, so there is no reward rather than 0. Where it
    can (``local``), the program may have changed it itself: that run failed."""
    from real2simlab.train.reward import RewardUnavailable

    solution = shutil.copytree(train_task / "hidden" / "generator" / "program", tmp_path / "s")
    with pytest.raises(RewardUnavailable, match="changed while it ran, from outside"):
        score_solution(
            train_task, solution, backend=_EditedWhileItRuns(), work_dir=tmp_path / "isolated"
        )
    run = tmp_path / "isolated" / "runs" / "run" / "run_record.json"
    record = json.loads(run.read_text())
    assert (record["status"], record["exit_code"]) == ("infra_error", 0)  # the program ran well
    local = score_solution(
        train_task, solution, backend=_EditedWhileItRunsLocally(), work_dir=tmp_path / "local"
    )
    run = tmp_path / "local" / "runs" / "run" / "run_record.json"
    assert local.reward == 0.0 and json.loads(run.read_text())["status"] == "failed"


@pytest.mark.slow
def test_an_experiment_whose_frozen_twin_changes_from_outside_builds_it_again(
    train_task: Path, tmp_path: Path
) -> None:
    """The run whose twin changed from outside is not scored and not retried, and the seeds
    after it do not run it: a changed twin cannot run, and that stops neither the experiment
    nor its other units. Each result says why it is not scored. The same command then builds
    the twin again and scores it."""
    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    config = _config([tasks], [{"name": "oracle"}], retries=2, seeds=[0, 1])
    changed, _ = _run(config, tmp_path / "run", _EditedWhileItRuns())
    (card,) = changed.methods[0].benchmarks
    assert (card.scored, card.results, card.infra_errors) == (0, 2, 2)
    ran, not_run = sorted(changed.to_json()["rows"], key=lambda row: row["seed"])
    assert "frozen twin changed while it ran, from outside its sandbox" in ran["infra_error"]
    assert "changed since it was built, so it was not run" in not_run["infra_error"]
    records = [
        json.loads(path.read_text())
        for path in sorted((tmp_path / "run" / "runs").rglob("run_record.json"))
    ]
    assert [(r["seed"], r["status"], r["attempt"]) for r in records] == [(0, "infra_error", 1)]
    status = json.loads((tmp_path / "run" / "state" / "status.json").read_text())
    assert (status["status"], status["infra_errors"]) == ("complete", 2)
    again, experiment = _run(config, tmp_path / "run")
    assert (experiment.progress.builds, experiment.progress.reused_builds) == (1, 0)
    assert again.methods[0].reproduction_skill.mean == pytest.approx(1.0)
    assert again.methods[0].infra_errors == 0


class _InputEditedWhileItRuns(LocalBackend):
    """While the first twin runs, something outside the Lab (an editor saving, a sync client)
    adds a line to ``path``: a file of a task, or of a program, the experiment evaluates."""

    def __init__(self, path: Path) -> None:
        super().__init__(quiet=True)
        self.path: Path | None = path

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        result = super().run(spec, log_path)
        if spec.label == "r2s-twin" and self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write("\n# saved by an editor while the experiment ran\n")
            self.path = None
        return result


@pytest.mark.parametrize("edited", ["task", "program"])
def test_an_input_edited_while_the_experiment_runs_leaves_the_run_unfinished(
    edited: str, train_task: Path, tmp_path: Path
) -> None:
    """A task or a program the experiment evaluates that changes while it runs was built and
    scored as neither version: the run ends with an error naming it and stays ``running``. It
    used to finish ``complete`` (exit status 0) with that unit's results silently left out of
    the scorecard. The same command then builds and scores it again, as it is now."""
    task = _copy(train_task, tmp_path / "tasks" / "t")
    program = shutil.copytree(train_task / "hidden" / "generator" / "program", tmp_path / "prog")
    config = _config([task], [{"name": "program", "path": str(program)}])
    out = tmp_path / "run"
    path = task / "task.toml" if edited == "task" else program / "run.py"
    with pytest.raises(SetupError, match="changed while the experiment ran") as raised:
        _run(config, out, _InputEditedWhileItRuns(path))
    assert str(task if edited == "task" else program) in str(raised.value)
    assert "run the same command again" in (raised.value.fix or "")
    assert json.loads((out / "state" / "status.json").read_text())["status"] == "running"
    again, experiment = _run(config, out)
    assert (experiment.progress.builds, experiment.progress.reused_builds) == (1, 0)
    assert again.methods[0].reproduction_skill.mean == pytest.approx(1.0)
    assert json.loads((out / "state" / "status.json").read_text())["status"] == "complete"


def _with_nan(task: Path) -> Path:
    """``task`` with ``nan`` in its free-form ``metadata`` table, written as TOML writes it."""
    import tomllib

    import tomli_w

    path = task / "task.toml"
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    data.setdefault("metadata", {})["qa_nan"] = float("nan")
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    return task


def test_a_task_holding_nan_is_not_taken_for_one_edited_while_it_runs(
    task_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``nan`` is not equal to itself: a ``task.toml`` with ``nan`` in a free-form table
    (``metadata``, a scorer's options), compared as parsed, was always "changed while the
    experiment ran", so ``r2s eval`` never finished, and its fix (run it again) did the same.
    A task is told changed by the bytes of its ``task.toml``."""
    from real2simlab.cli.main import main
    from real2simlab.task import load_task

    task = _with_nan(task_copy)
    loaded = load_task(task)
    assert not loaded.changed_since(loaded.content_hash())
    run = tmp_path / "run"
    argv = ["eval", "--tasks", str(task), "--method", "static", "--backend", "local",
            "--out", str(run)]  # fmt: skip
    assert main(argv) == 0, capsys.readouterr().err
    assert main(argv) == 0, capsys.readouterr().err
    said = "".join(capsys.readouterr())
    assert "0 builds (1 reused), 0 runs (1 reused)" in said, said
    (task / "task.toml").write_text(
        (task / "task.toml").read_text(encoding="utf-8") + "# edited\n", encoding="utf-8"
    )
    assert loaded.changed_since(loaded.content_hash()), "an edit is still an edit"


def test_a_training_task_holding_nan_is_exported(train_task: Path, tmp_path: Path) -> None:
    """``r2s train export`` asks the same question of each task it copies: a training task
    holding ``nan`` in ``metadata`` was "changed while it was exported", every time."""
    task = _with_nan(Path(shutil.copytree(train_task, tmp_path / "tasks" / "t")))
    assert export_dataset(tmp_path / "data", tasks=[task]).tasks == 1


class _UnreadableAfterItRuns(LocalBackend):
    """A backend that keeps the program from writing its frozen twin (docker and bwrap mount it
    read-only), here the local one, on a machine that fails to read the twin after it ran (an
    I/O error of its disk): nothing changed the twin, and its program did nothing wrong."""

    isolated: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(quiet=True)
        self.ran = False

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        result = super().run(spec, log_path)
        self.ran = self.ran or spec.label == "r2s-twin"
        return result


class _UnreadableAfterItRunsLocally(_UnreadableAfterItRuns):
    """The same, where nothing keeps the program from writing its twin itself."""

    isolated: ClassVar[bool] = False


@pytest.mark.parametrize(
    "machine", [_UnreadableAfterItRuns, _UnreadableAfterItRunsLocally], ids=["isolating", "local"]
)
def test_a_frozen_twin_this_machine_cannot_read_is_an_error_not_a_change(
    train_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    machine: type[_UnreadableAfterItRuns],
) -> None:
    """This machine fails to read the frozen twin after its run: that does not show the twin
    changed, from outside its sandbox or by its program. The command fails with the error,
    rather than giving a reward of 0 or blaming a change from outside."""
    import errno

    from real2simlab.twin.io import LoadedTwin

    backend = machine()
    fingerprint = LoadedTwin.current_fingerprint

    def read(twin: LoadedTwin) -> str:
        if backend.ran:
            raise OSError(errno.EIO, os.strerror(errno.EIO), str(twin.root / "program" / "run.py"))
        return fingerprint(twin)

    monkeypatch.setattr(LoadedTwin, "current_fingerprint", read)
    solution = shutil.copytree(train_task / "hidden" / "generator" / "program", tmp_path / "s")
    with pytest.raises(OSError, match=os.strerror(errno.EIO)):
        score_solution(train_task, solution, backend=backend, work_dir=tmp_path / "work")


# --- What a program leaves never stops the check of this machine's storage -----------------

LEFTOVERS = {
    # a directory the Lab may list but not enter (r--), holding a file
    "unsearchable": (
        "kept = os.path.join(where, 'kept')\n"
        "os.makedirs(kept, exist_ok=True)\n"
        "with open(os.path.join(kept, 'note.txt'), 'w') as note:\n"
        "    note.write('x')\n"
        "os.chmod(kept, 0o444)\n"
    ),
    # a file under a path longer than the system allows (PATH_MAX, 4096 bytes on Linux)
    "too_long": (
        "os.chdir(where)\n"
        "for _ in range(24):\n"
        "    os.mkdir('d' * 200)\n"
        "    os.chdir('d' * 200)\n"
        "with open('bottom.txt', 'w') as bottom:\n"
        "    bottom.write('x')\n"
    ),
}
LEFT_IN = {
    "temporary": "tempfile.gettempdir()",  # TMPDIR and HOME: the local backend's local_tmp
    "output": "os.environ['R2S_OUTPUT_DIR']",
}


def _leaving(leftover: str, where: str, then: str = "") -> str:
    """The end of a program that leaves ``leftover`` in ``where``, then runs ``then``."""
    return f"import os, sys, tempfile\nwhere = {LEFT_IN[where]}\n{LEFTOVERS[leftover]}{then}"


def _file_size_limited() -> None:
    """A file-size limit far above anything written here: with one, the Lab checks a run's
    files for one that reached it."""
    import resource

    resource.setrlimit(resource.RLIMIT_FSIZE, (32 << 20, 32 << 20))


@pytest.fixture
def leftovers(tmp_path: Path) -> Iterator[Path]:
    """``tmp_path``, for programs that leave what the Lab cannot read: made accessible again
    afterwards, whatever the code tested did, so that pytest can remove it."""
    from real2simlab._fs import make_owner_accessible

    yield tmp_path
    make_owner_accessible(tmp_path)


def _limited_r2s(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """``r2s ARGS`` (this checkout's) under :func:`_file_size_limited`, its temporary files in
    ``tmp_path``."""
    (tmp_path / "tmp").mkdir(exist_ok=True)
    here = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": here, "TMPDIR": str(tmp_path / "tmp")}
    return subprocess.run(
        [sys.executable, "-m", "real2simlab.cli.main", *args],
        env=env, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL,
        timeout=600, preexec_fn=_file_size_limited,
    )  # fmt: skip


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resource limits")
@pytest.mark.parametrize("leftover", sorted(LEFTOVERS))
@pytest.mark.parametrize(
    ("where", "then", "reward"),
    [("temporary", "", "1.0000"), ("output", "sys.exit(1)\n", "0.0000")],
    ids=["succeeds", "fails"],
)
def test_what_a_program_leaves_never_stops_its_reward(
    bouncing_task: Path, leftovers: Path, leftover: str, where: str, then: str, reward: str
) -> None:
    """With a file-size limit set, the Lab checks a run's files for one that reached it. A
    program that leaves a directory the Lab may list but not enter, or a path longer than the
    system allows, gets its own result all the same: 1 for a correct program that leaves it in
    its temporary directory, 0 for one that leaves it in its output and fails. The check skips
    what it cannot read; that is not an error of the command."""
    extra = _leaving(leftover, where, then)
    program = _truth_program(bouncing_task, leftovers / "program", extra)
    done = _limited_r2s(["train", "reward", "--task", str(bouncing_task), "--solution",
                         str(program), "--backend", "local"], leftovers)  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"reward {reward}" in done.stdout, done.stdout + done.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resource limits")
def test_an_evaluation_scores_a_program_that_leaves_what_the_lab_cannot_read(
    bouncing_task: Path, leftovers: Path
) -> None:
    """The same in an evaluation, where an error of the check would stop every method: the
    program that leaves both in its temporary directory is scored like the other one, and the
    evaluation completes."""
    truth = _truth_program(bouncing_task, leftovers / "truth")
    extra = _leaving("unsearchable", "temporary") + _leaving("too_long", "temporary")
    left = _truth_program(bouncing_task, leftovers / "left", extra)
    out = leftovers / "eval"
    done = _limited_r2s(["eval", "--tasks", str(bouncing_task),
                         "--method", f"truth=program:path={truth}",
                         "--method", f"left=program:path={left}",
                         "--backend", "local", "--out", str(out)], leftovers)  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads((out / "state" / "status.json").read_text())["status"] == "complete"
    methods = json.loads((out / "results.json").read_text())["methods"]
    scores = {m["method"]: (m["reproduction_skill"]["mean"], m["infra_errors"]) for m in methods}
    assert scores == {"truth": (pytest.approx(1.0), 0), "left": (pytest.approx(1.0), 0)}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions and named pipes")
def test_the_storage_check_only_looks_at_what_a_program_left(
    leftovers: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a program left in a run directory is only looked at: a directory the Lab may list
    but not enter, one it may not list, a path longer than the system allows, a named pipe
    and a link are skipped (the link is not followed) rather than failing the check, a file
    at the file-size limit elsewhere is still found, and a huge tree is looked at only up to a
    bound."""
    import resource

    from real2simlab.sandbox import process

    limit = 4096
    getrlimit = resource.getrlimit
    monkeypatch.setattr(
        resource,
        "getrlimit",
        lambda which: (
            (limit, resource.RLIM_INFINITY) if which == resource.RLIMIT_FSIZE else getrlimit(which)
        ),
    )
    run = leftovers / "run"
    left = run / "local_tmp"
    left.mkdir(parents=True)
    (run / "log.txt").write_text("log\n")
    for name, mode in (("unsearchable", 0o444), ("unreadable", 0o333)):
        (left / name).mkdir()
        (left / name / "note.txt").write_text("x")
        (left / name).chmod(mode)
    deep = os.open(left, os.O_RDONLY)
    try:
        for _ in range(24):  # 24 * 201 bytes: longer than PATH_MAX
            os.mkdir("d" * 200, dir_fd=deep)
            below = os.open("d" * 200, os.O_RDONLY, dir_fd=deep)
            os.close(deep)
            deep = below
        os.close(os.open("bottom.txt", os.O_WRONLY | os.O_CREAT, 0o644, dir_fd=deep))
    finally:
        os.close(deep)
    os.mkfifo(left / "pipe")
    (leftovers / "outside.bin").write_bytes(b"x" * limit)  # not the run's, though linked from it
    (left / "link").symlink_to(leftovers / "outside.bin")
    assert process.storage_problem(run / "log.txt", [run]) is None
    (run / "output").mkdir()
    (run / "output" / "frames.bin").write_bytes(b"x" * limit)
    found = process.storage_problem(run / "log.txt", [run / "output", run])
    assert found is not None and "frames.bin" in found, found

    chain = leftovers / "chain"  # 200 directories, one in the other
    (chain / "/".join(["a"] * 200)).mkdir(parents=True)
    scandir = os.scandir
    listed: list[object] = []

    def counted(path: Any) -> Any:
        listed.append(path)
        return scandir(path)

    monkeypatch.setattr(process, "SCAN_ENTRIES", 50)
    monkeypatch.setattr(os, "scandir", counted)
    problem = process.storage_problem(run / "log.txt", [chain])
    monkeypatch.setattr(os, "scandir", scandir)
    assert problem is None
    assert len(listed) <= 51, len(listed)  # the top, and at most 50 entries below it


# --- What a program leaves in its output is its own problem, never the Lab's crash -----------


def _truth_program(bouncing_task: Path, target: Path, extra: str = "") -> Path:
    """The task's truth program, with ``extra`` appended to its ``run.py``."""
    shutil.copytree(bouncing_task / "hidden" / "generator" / "program", target)
    if extra:
        with (target / "run.py").open("a", encoding="utf-8") as stream:
            stream.write("\n" + extra + "\n")
    return target


OUTPUT = "Path(os.environ['R2S_OUTPUT_DIR'])"


@pytest.mark.parametrize(
    "damage",
    [
        f"import os\np = {OUTPUT} / 'tracks.npz'\np.write_bytes(p.read_bytes()[:100])",
        f"import json, os\nimport numpy as np\nout = {OUTPUT}\n"
        "meta = json.loads((out / 'meta.json').read_text()); del meta['num_frames']\n"
        "(out / 'meta.json').write_text(json.dumps(meta))\n"
        "np.savez(out / 'tracks.npz', **{meta['objects'][0] + '.position': np.array(1.0)})",
        f"import os\nos.mkfifo({OUTPUT} / 'render.mp4')",
        f"import os\n({OUTPUT} / 'meta.json').write_text('[1, 2]')",
    ],
    ids=["truncated archive", "scalar array", "named pipe", "json of the wrong kind"],
)
def test_a_damaged_output_is_invalid_output_not_a_crash(
    damage: str, bouncing_task: Path, tmp_path: Path
) -> None:
    program = _truth_program(bouncing_task, tmp_path / "program", damage)
    scored = score_solution(bouncing_task, program, backend=LocalBackend(quiet=True))
    assert scored.reward == 0.0
    assert scored.result.reproduction.run_status == "invalid_output"


# --- An output this machine cannot read is not its program's result (A12-review-12) ------------

HOST_READ_FAULTS: dict[str, OSError] = {
    "no permission": PermissionError("host read failure"),
    "an I/O error": OSError(errno.EIO, os.strerror(errno.EIO)),
    "a stale file handle": OSError(errno.ESTALE, os.strerror(errno.ESTALE)),
    "too many open files": OSError(errno.EMFILE, os.strerror(errno.EMFILE)),
}


def _output_read_fails(monkeypatch: pytest.MonkeyPatch, step: str, error: Exception) -> None:
    """Reading a program's output raises ``error`` on the host, after the program ran: where
    the Lab checks it (``check``), where its format reads one of its files (``format``:
    ``meta.json``), or where the Lab fingerprints it (``fingerprint``)."""
    import real2simlab.runner.run as run_module

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error

    if step == "format":
        read_text = Path.read_text

        def read(self: Path, *args: Any, **kwargs: Any) -> str:
            if self.name == "meta.json" and self.parent.name == "output":
                raise error
            return read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read)
    else:
        monkeypatch.setattr(run_module, {"check": "output_problems"}.get(step, step), fail)


@pytest.mark.parametrize(
    ("step", "fault"),
    [
        ("check", "an I/O error"),
        ("check", "too many open files"),
        ("format", "an I/O error"),
        ("format", "no permission"),
        ("fingerprint", "no permission"),
        ("fingerprint", "an I/O error"),
        ("fingerprint", "a stale file handle"),
    ],
)
def test_an_output_this_machine_cannot_read_gives_no_reward(
    step: str, fault: str, bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A program ran and wrote its output, and this machine then fails to read it (no
    permission, an I/O error, a stale file handle, too many open files): not a problem of the
    output, so not ``invalid_output`` with a reward of 0, but ``infra_error``: no reward
    (``r2s train reward`` exits 1), no score, and the run is done again."""
    from real2simlab.train.reward import RewardUnavailable

    program = _truth_program(bouncing_task, tmp_path / "program")
    _output_read_fails(monkeypatch, step, HOST_READ_FAULTS[fault])
    work = tmp_path / "work"
    with pytest.raises(RewardUnavailable, match="this machine could not read the output") as no:
        score_solution(bouncing_task, program, backend=LocalBackend(quiet=True), work_dir=work)
    assert no.value.result.infra_error is not None
    assert no.value.result.summary.reproduction_skill is None
    record = json.loads((work / "runs" / "run" / "run_record.json").read_text())
    assert record["status"] == "infra_error", record["problems"]


@pytest.mark.parametrize(
    ("step", "error"),
    [
        ("check", ValueError("a value of the wrong kind")),
        ("check", OSError(errno.ENAMETOOLONG, os.strerror(errno.ENAMETOOLONG))),
        ("format", OSError("not a JSON document")),  # a reader's own error: no errno
        ("fingerprint", ValueError("symbolic link in output: link")),
    ],
    ids=["wrong kind", "name too long", "a reader's error", "symbolic link"],
)
def test_an_output_whose_own_files_cannot_be_read_is_invalid_output(
    step: str,
    error: Exception,
    bouncing_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the output's own files cause when they are read (a value of the wrong kind, a name
    too long, content a reader rejects, a symbolic link) stays the program's result:
    ``invalid_output``, a reward of 0."""
    program = _truth_program(bouncing_task, tmp_path / "program")
    _output_read_fails(monkeypatch, step, error)
    scored = score_solution(bouncing_task, program, backend=LocalBackend(quiet=True))
    assert scored.reward == 0.0
    assert scored.result.reproduction.run_status == "invalid_output"


@pytest.mark.slow
def test_an_output_this_machine_could_not_read_is_run_again_by_the_next_run(
    task_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In an experiment, this machine fails once to read a twin's output: that result is not
    scored (``infra_error``, not 0 in the scorecard), and the same command run again, the
    machine working, runs the twin again and scores it, never reusing the failed run."""
    import real2simlab.runner.run as run_module

    fingerprint = run_module.fingerprint
    failed: list[Path] = []

    def once(directory: Path, **options: Any) -> str:
        if not failed:
            failed.append(directory)
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(directory))
        return fingerprint(directory, **options)

    monkeypatch.setattr(run_module, "fingerprint", once)
    config = _config([task_copy], [{"name": "oracle"}])
    broken, _ = _run(config, tmp_path / "run")
    assert failed
    (card,) = broken.methods[0].benchmarks
    assert (card.scored, card.results, card.infra_errors) == (0, 1, 1)
    (row,) = broken.to_json()["rows"]
    assert "this machine could not read the output" in row["infra_error"]
    assert row["reproduction_skill"] is None
    fixed, experiment = _run(config, tmp_path / "run")
    assert fixed.methods[0].reproduction_skill.mean == pytest.approx(1.0)
    assert fixed.methods[0].infra_errors == 0
    assert (experiment.progress.runs, experiment.progress.reused_runs) == (1, 0)


def test_a_misspelt_scoring_option_is_a_broken_task_not_a_score(
    task_copy: Path, tmp_path: Path
) -> None:
    from real2simlab.errors import ProblemsError
    from real2simlab.scoring import Scorer
    from real2simlab.task import load_task
    from real2simlab.task.validate import validate_task

    data = read_task_config(task_copy / "task.toml").to_json()
    data["scoring"]["metric_options"] = {"traj3d_mean": {"align": "similiarity"}}
    write_task_config(TaskConfig.model_validate(data), task_copy / "task.toml")
    report = validate_task(task_copy)
    assert not report.ok and "similiarity" in str(report.errors[0])
    program = task_copy / "hidden" / "generator" / "program"
    with pytest.raises(ProblemsError, match="similiarity"):
        score_solution(task_copy, program, backend=LocalBackend(quiet=True))
    with pytest.raises(SetupError, match="cannot be scored"):
        Scorer(load_task(task_copy))
    data["scoring"]["metric_options"] = {"traj3d_mean": {"aling": "none"}}
    write_task_config(TaskConfig.model_validate(data), task_copy / "task.toml")
    assert "takes no option 'aling'" in str(validate_task(task_copy).errors[0])


def test_a_task_without_its_reference_gives_no_reward(task_copy: Path, tmp_path: Path) -> None:
    from real2simlab.errors import ProblemsError

    program = _truth_program(task_copy, tmp_path / "program")
    shutil.rmtree(task_copy / "hidden" / "reference")
    with pytest.raises(ProblemsError, match="reference output is missing"):
        score_solution(task_copy, program, backend=LocalBackend(quiet=True))


class _NoInterpreter(LocalBackend):
    """A runtime whose interpreter is gone: nothing starts."""

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("bwrap: execvp python: No such file or directory\n")
        return ExecResult(exit_code=1, timed_out=False, duration_s=0.01)


def test_a_runtime_that_cannot_start_python_gives_no_reward(
    bouncing_task: Path, tmp_path: Path
) -> None:
    program = _truth_program(bouncing_task, tmp_path / "program")
    # the same runtime worked a moment ago: a runtime damaged since is still found
    assert score_solution(bouncing_task, program, backend=LocalBackend(quiet=True)).reward > 0.99
    with pytest.raises(InfrastructureError, match="cannot start python"):
        score_solution(bouncing_task, program, backend=_NoInterpreter(quiet=True))
    # a program that fails on its own still fails: the runtime starts python
    failing = _truth_program(bouncing_task, tmp_path / "failing", "raise SystemExit(3)")
    scored = score_solution(bouncing_task, failing, backend=LocalBackend(quiet=True))
    assert (scored.reward, scored.result.reproduction.run_status) == (0.0, "failed")


class _DamagedNumpy(LocalBackend):
    """A runtime whose NumPy is damaged (an image rebuilt without it): every import fails."""

    damaged: Path

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        from dataclasses import replace

        from real2simlab.sandbox.base import Mount
        from real2simlab.twin.sdk import sdk_path

        # Mounted at a path of the sandbox, as an image's own files are (the environment's
        # paths are the sandbox's: a host path under /tmp would be taken for its tmpfs).
        shutil.copy2(sdk_path(), self.damaged)
        mounts = (*spec.mounts, Mount(self.damaged, "/opt/damaged", read_only=True))
        return super().run(
            replace(spec, mounts=mounts, env={**spec.env, "PYTHONPATH": "/opt/damaged"}), log_path
        )


def test_a_runtime_without_what_it_promises_gives_no_reward(
    bouncing_task: Path, tmp_path: Path
) -> None:
    program = _truth_program(bouncing_task, tmp_path / "program")
    backend = _DamagedNumpy(quiet=True)
    backend.damaged = tmp_path / "damaged"
    backend.damaged.mkdir()
    (backend.damaged / "numpy.py").write_text('raise ImportError("numpy is damaged")\n')
    with pytest.raises(InfrastructureError, match="lacks what every Lab runtime holds: numpy"):
        score_solution(bouncing_task, program, backend=backend)
    # a program that needs what no Lab runtime promises fails on its own: a result
    missing = _truth_program(bouncing_task, tmp_path / "missing", "import not_in_any_runtime")
    scored = score_solution(bouncing_task, missing, backend=LocalBackend(quiet=True))
    assert (scored.reward, scored.result.reproduction.run_status) == (0.0, "failed")


def test_the_runtime_probe_runs_nothing_of_the_program(tmp_path: Path) -> None:
    """The probe starts the interpreter in isolated mode: a program's ``sitecustomize.py``
    cannot make it fail (and so turn its own failure into one the Lab does not count)."""
    from real2simlab.runner.run import runtime_problem
    from real2simlab.twin.sdk import sdk_path

    (tmp_path / "sitecustomize.py").write_text("raise SystemExit(9)\n")
    spec = SandboxSpec(image="r2s-python", argv=("python", "run.py"), workdir=str(tmp_path))
    backend = LocalBackend(quiet=True)
    assert runtime_problem(backend, spec, tmp_path / "probe.log") is None
    # Nor does it write next to what it imports (isolated mode ignores PYTHONDONTWRITEBYTECODE;
    # the local backend's SDK is the installed package's).
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    shutil.copy2(sdk_path(), sdk)
    backend = _DamagedNumpy(quiet=True)
    backend.damaged = sdk  # mounted as the runtime's SDK; nothing in it is damaged
    assert runtime_problem(backend, spec, tmp_path / "probe2.log") is None
    assert not (sdk / "__pycache__").exists()


class _NoSandbox(BwrapBackend):
    """bwrap on a machine that forbids a part of its sandbox (a new /proc): nothing starts in
    it, and bwrap says why. With ``exec_failure``, the sandbox starts and its runtime has no
    python."""

    exec_failure: bool = False

    def problems(self) -> list[ReadinessProblem]:
        if self.exec_failure:
            return []
        return [ReadinessProblem("bwrap cannot mount a new /proc here", "use --backend docker")]

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "bwrap: execvp python: No such file or directory\n"
            if self.exec_failure
            else "bwrap: Can't mount proc on /newroot/proc: Operation not permitted\n"
        )
        return ExecResult(exit_code=1, timed_out=False, duration_s=0.01)


def test_a_sandbox_bwrap_cannot_set_up_is_not_blamed_on_the_runtime(tmp_path: Path) -> None:
    """A failed run whose probe never started (bwrap's own message: this machine forbids its
    sandbox) was blamed on the runtime, with the fix to unpack it again, which changes nothing.
    It is the backend's failure, with the backend's fix; a runtime without python stays the
    runtime's, with a repair that also works where there is no Docker (--from-tar)."""
    from real2simlab.runner.run import runtime_problem

    image = "ghcr.io/real2simhub/r2s-python:1.0"
    spec = SandboxSpec(image=image, argv=("python", "run.py"), workdir="/r2s/program")
    backend = _NoSandbox(bwrap="bwrap", cgroup_controllers=frozenset())
    problem = runtime_problem(backend, spec, tmp_path / "probe.log") or ""
    assert "cannot mount a new /proc" in problem and "use --backend docker" in problem, problem
    assert "env unpack" not in problem and "Repair the runtime" not in problem, problem
    backend.exec_failure = True
    problem = runtime_problem(backend, spec, tmp_path / "probe2.log") or ""
    assert "cannot start python" in problem, problem
    assert "r2s env unpack r2s-python --force" in problem and "--from-tar" in problem, problem


def _eval_command(command: str, task: Path, out: Path, methods: list[str]) -> list[str]:
    """``r2s eval`` of ``methods`` on ``task``, or ``r2s run`` of an experiment file saying so."""
    if command == "eval":
        flags = [flag for name in methods for flag in ("--method", name)]
        return ["eval", "--tasks", str(task), *flags, "--backend", "local", "--out", str(out)]
    config = out.with_suffix(".toml")
    listed = "".join(f'[[methods]]\nname = "{name}"\n' for name in methods)
    config.write_text(
        f'spec = "r2s-experiment/1.0"\nname = "pilot"\n[tasks]\npaths = [{json.dumps(str(task))}]\n'
        f'{listed}[run]\nbackend = "local"\nretries = 0\n'
    )
    return ["run", str(config), "--out", str(out), "--no-report"]


@pytest.mark.parametrize("command", ["eval", "run"])
def test_an_evaluation_that_scored_nothing_fails_and_says_why(
    command: str,
    task_copy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every twin's sandbox failed (a backend that cannot run on this machine): `r2s eval` and
    `r2s run` exited 0 and said to run the same command again, with the cause only in each run
    record. Nothing scored is a failed command, and it says why (so does r2s.log)."""
    from real2simlab.cli.main import main
    from real2simlab.orchestrate import experiment as module

    monkeypatch.setattr(module, "get_backend", lambda name, **options: FlakyBackend(twin_runs))
    out = tmp_path / "run"
    argv = _eval_command(command, task_copy, out, ["oracle"])
    capsys.readouterr()
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "error: nothing could be scored" in err and "\n  fix: " in err, err
    assert "injected: the container did not start" in err, err  # why, not only that
    assert "injected: the container did not start" in (out / "r2s.log").read_text()


def test_an_evaluation_with_results_left_unscored_says_why(
    task_copy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Some results scored and some not: the numbers stand (status 0), and the summary says why
    the others are missing and that the same command scores them once that is fixed."""
    from real2simlab.cli.main import main
    from real2simlab.orchestrate import experiment as module

    def static_twins(spec: SandboxSpec) -> bool:
        return twin_runs(spec) and any("/static/" in str(mount.source) for mount in spec.mounts)

    monkeypatch.setattr(module, "get_backend", lambda name, **options: FlakyBackend(static_twins))
    argv = _eval_command("eval", task_copy, tmp_path / "run", ["oracle", "static"])
    capsys.readouterr()
    assert main(argv) == 0
    err = capsys.readouterr().err
    assert "1 result(s) could not be scored" in err, err
    assert "injected: the container did not start" in err and "error:" not in err, err
    # The report's index said the cause was "listed below": it lists no results; the task's
    # page gives the cause, as the warning now says.
    report = tmp_path / "run" / "report"
    index = (report / "index.html").read_text()
    assert "could not be scored" in index and "listed below" not in index, index
    assert "task&#x27;s page" in index or "task's page" in index, index
    pages = [page.read_text() for page in (report / "tasks").glob("*.html")]
    assert any("injected: the container did not start" in page for page in pages)


DAEMON_DOWN = (
    "#!/bin/sh\necho 'Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the "
    "docker daemon running?' >&2\nexit 1\n"
)
IMAGE_MISSING = (
    '#!/bin/sh\ncase "$1" in\n  info) echo \'"28.1.0"\' ;;\n'
    '  *) echo "Error response from daemon: No such image: x" >&2; exit 1 ;;\nesac\n'
)


@pytest.mark.parametrize("name", ["fourdcodebench", "video2world"])
def test_a_scorer_in_docker_says_docker_is_down_not_that_its_image_is_unbuilt(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A docker command without its daemon (a login node, a DOCKER_HOST whose daemon stopped):
    the official scorers that run in Docker said their image was not built, and to build it,
    which needs that daemon. They say Docker is not ready, with its fix; "not built" only when
    Docker answers that it lacks the image."""
    from real2simlab.scoring.native import get_scorer

    for variable in ("DOCKER_HOST", "DOCKER_CONTEXT"):
        monkeypatch.delenv(variable, raising=False)
    docker = tmp_path / "docker"
    docker.write_text(DAEMON_DOWN)
    docker.chmod(0o755)
    monkeypatch.setenv("R2S_DOCKER", str(docker))
    (problem,) = get_scorer(name).problems()
    assert "not built" not in problem and "not ready" in problem, problem
    assert "sudo systemctl start docker" in problem, problem
    # Docker gone between the check before an evaluation and its scoring: the same cause, and
    # a fix line (there was none).
    from types import SimpleNamespace

    delivery = tmp_path / "delivery"
    delivery.mkdir()
    (delivery / "protocol.json").write_text("{}")  # a package for Video2World to score
    task: Any = SimpleNamespace(id="t", hidden_dir=tmp_path, config=SimpleNamespace(metadata={}))
    options = {"kind": "synthetic", "case": "c"}
    with pytest.raises(SetupError, match="cannot run") as raised:
        get_scorer(name).score(task, delivery_dir=delivery, output_dir=None,
                               work_dir=tmp_path / "work", options=options)  # fmt: skip
    assert "run the same command again" in (raised.value.fix or ""), raised.value
    docker.write_text(IMAGE_MISSING)
    (problem,) = get_scorer(name).problems()
    assert "not built" in problem, problem


@pytest.mark.parametrize("command", ["eval", "run"])
def test_a_task_whose_scorer_cannot_run_says_how_each_command_leaves_it_out(
    command: str,
    task_copy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A task whose official scorer cannot run here (Docker is down) is refused before
    anything runs: the fix said to leave it out with `[tasks] exclude`, which only the
    experiment file of `r2s run` has. It says how for `r2s run` and for `r2s eval`."""
    from real2simlab.cli.main import main
    from real2simlab.orchestrate import experiment as module

    class Unready(_BrokenScorer):
        def problems(self) -> list[str]:
            return ["the official scorer runs in Docker whatever the backend, and Docker is down"]

    data = read_task_config(task_copy / "task.toml").to_json()
    data["scoring"] = {
        "protocol": "native",
        "native": {"scorer": "fourdcodebench", "options": {"kind": "synthetic", "case": "B_30"}},
    }
    write_task_config(TaskConfig.model_validate(data), task_copy / "task.toml")
    monkeypatch.setattr(module, "get_scorer", lambda name, **options: Unready("scores"))
    argv = _eval_command(command, task_copy, tmp_path / "run", ["oracle"])
    capsys.readouterr()
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "cannot run here" in err and "\n  fix: " in err, err
    assert "`r2s eval`" in err and "--tasks" in err and "`r2s run`" in err, err


def test_an_edited_dataset_or_checkpoint_is_refused(train_task: Path, tmp_path: Path) -> None:
    from real2simlab.methods import get_method
    from real2simlab.train import fit_method

    data = tmp_path / "data"
    export_dataset(data, tasks=[train_task.parent])
    row = json.loads((data / "tasks.jsonl").read_text().splitlines()[0])
    video = data / row["video"]
    pristine = video.read_bytes()
    video.write_bytes(pristine[:-10])  # e.g. synced over by another version
    with pytest.raises(SetupError, match="not as r2s train export wrote it"):
        fit_method(get_method("nearest_twin"), data, tmp_path / "model")
    video.write_bytes(pristine)
    report = json.loads((data / "export.json").read_text())
    unrecorded = {key: value for key, value in report.items() if key != "content"}
    # every export records its content: one without it (edited, or written by a pre-release
    # build) cannot be checked, whatever Lab version it names
    for version in (report["lab_version"], "0.9.0"):
        (data / "export.json").write_text(json.dumps({**unrecorded, "lab_version": version}))
        with pytest.raises(SetupError, match=r"export\.json: it records no content") as raised:
            fit_method(get_method("nearest_twin"), data, tmp_path / "model")
        assert "pre-release build" in str(raised.value)
        assert "r2s train export" in (raised.value.fix or "")
    assert not (tmp_path / "model").exists()
    for damaged in ("null", "[]", "{}", json.dumps({**report, "content": None}), "x"):
        (data / "export.json").write_text(damaged)  # no longer a record of what was exported
        with pytest.raises(SetupError, match=r"export\.json"):
            fit_method(get_method("nearest_twin"), data, tmp_path / "model")
    (data / "export.json").unlink()
    with pytest.raises(SetupError, match=r"export\.json: removed"):
        fit_method(get_method("nearest_twin"), data, tmp_path / "model")
    (data / "export.json").write_text(json.dumps(report))
    with (data / "tasks.jsonl").open("a") as rows:
        rows.write("\n")  # still valid rows, but not the file the export wrote
    with pytest.raises(SetupError, match=r"tasks\.jsonl: changed"):
        fit_method(get_method("nearest_twin"), data, tmp_path / "model")
    shutil.rmtree(data)
    export_dataset(data, tasks=[train_task.parent])
    for litter in (data / ".DS_Store", video.parent / "._capture.json"):  # opened in Finder
        litter.write_bytes(b"\0")
    fit_method(get_method("nearest_twin"), data, tmp_path / "model", log=lambda _: None)
    (tmp_path / "model" / ".DS_Store").write_bytes(b"\0")
    get_method("nearest_twin", checkpoint=str(tmp_path / "model"))  # as trained: usable
    import numpy as np

    np.savez(tmp_path / "model" / "nearest_twin.npz", arbitrary=[1])
    with pytest.raises(SetupError, match="has changed since r2s train fit wrote it"):
        get_method("nearest_twin", checkpoint=str(tmp_path / "model"))


def test_a_malformed_dataset_row_names_its_file_and_line(tmp_path: Path) -> None:
    from real2simlab.train.fit import _task_ids

    (tmp_path / "tasks.jsonl").write_text('{"task_id": "a"}\n\n{}\n')
    with pytest.raises(SetupError, match=r"tasks.jsonl, line 3: a row without a task_id"):
        _task_ids(tmp_path)
    (tmp_path / "tasks.jsonl").write_text("not json\n")
    with pytest.raises(SetupError, match="line 1: not JSON"):
        _task_ids(tmp_path)


def test_a_broken_model_stops_before_anything_runs(train_task: Path, tmp_path: Path) -> None:
    """A model file that cannot be read is found when the evaluation starts (not in a worker,
    leaving the run marked running), even in a checkpoint whose digest vouches for it (a method
    that wrote a model it cannot read back)."""
    from real2simlab.methods import get_method
    from real2simlab.methods.training import TRAINING_FILE, content_digest
    from real2simlab.train import fit_method

    data = tmp_path / "data"
    export_dataset(data, tasks=[train_task.parent])
    model = tmp_path / "model"
    fit_method(get_method("nearest_twin"), data, model, log=lambda _: None)
    record = json.loads((model / TRAINING_FILE).read_text())
    import numpy as np

    np.savez(model / "nearest_twin.npz", arbitrary=[1])
    record["content_digest"] = content_digest(model)  # as if training had written it so
    (model / TRAINING_FILE).write_text(json.dumps(record))
    method = get_method("nearest_twin", checkpoint=str(model))
    (problem,) = method.problems(LocalBackend(quiet=True))
    assert "is not a nearest_twin model" in problem and "r2s train fit" in problem


def test_docker_runs_what_it_resolved_and_quotes_its_mounts(tmp_path: Path) -> None:
    from real2simlab.sandbox.base import Mount
    from real2simlab.sandbox.docker import DockerBackend, bind_mount

    assert bind_mount("/a/Last, First", "/x", read_only=True) == (
        'type=bind,"source=/a/Last, First",target=/x,readonly'
    )
    assert bind_mount('/a/say "hi"', "/x") == 'type=bind,"source=/a/say ""hi""",target=/x'
    backend = DockerBackend(docker="docker")
    backend._images["r2s-python:1.0"] = ResolvedImage(
        reference="r2s-python:1.0", digest="sha256:d", content_id="sha256:abc"
    )
    spec = SandboxSpec(image="r2s-python:1.0", argv=("python", "run.py"), workdir="/twin",
                       mounts=(Mount(tmp_path / "a, b", "/twin"),))  # fmt: skip
    command = backend.command(spec, "r2s-test")
    assert "sha256:abc" in command and "r2s-python:1.0" not in command
    assert f'type=bind,"source={tmp_path / "a, b"}",target=/twin,readonly' in command


# --- Two commands never share a directory; an output never lands inside its input -----------


def test_a_work_directory_in_use_is_refused(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from real2simlab._fs import exclusive_use
    from real2simlab.cli.main import main

    program = _truth_program(bouncing_task, tmp_path / "program")
    work, check = tmp_path / "work", tmp_path / "check"
    for directory in (work, check):
        directory.mkdir()
    reward = ["train", "reward", "--task", str(bouncing_task), "--solution", str(program),
              "--backend", "local", "--work", str(work)]  # fmt: skip
    twin = ["twin", "check", str(program), "--backend", "local", "--out", str(check)]
    for argv, directory in ((reward, work), (twin, check)):
        with exclusive_use(directory, fix="-"):  # another command holds it
            assert main(argv) == 1
            assert "another r2s command is using" in capsys.readouterr().err
            assert not any(directory.iterdir())  # nothing was cleared or written
    assert main(reward) == 0  # free again


# --- The report shows what was scored, and shares only what may be shared ---------------------


def _reported(task: Path, out: Path) -> Path:
    from real2simlab.report import write_report

    _run(_config([task], [{"name": "oracle"}]), out)
    return write_report(out).parent


def test_a_report_does_not_draw_an_output_changed_since_it_was_scored(
    task_copy: Path, tmp_path: Path
) -> None:
    report = _reported(task_copy, tmp_path / "run")
    (page,) = (report / "tasks").glob("*.html")
    assert "overlay_oracle" in page.read_text()
    (tracks,) = (tmp_path / "run").rglob("run/output/tracks.npz")
    import numpy as np

    with np.load(tracks) as data:
        arrays = {name: data[name] + 1.0 for name in data.files}
    np.savez(tracks, **arrays)  # e.g. a program run again by hand in the run directory
    from real2simlab.report import write_report

    write_report(tmp_path / "run")  # the report alone: resuming would run the program again
    text = page.read_text()
    assert "has changed since it was scored" in text and "overlay_oracle" not in text
    assert not list(report.rglob("overlay_*.mp4")), "an overlay of an earlier build was kept"


def test_a_report_does_not_draw_an_output_whose_record_cannot_vouch_for_it(
    task_copy: Path, tmp_path: Path
) -> None:
    """A run record holds the fingerprint of every output that succeeded: one without it
    (edited by hand, rewritten by a tool that drops optional fields, written by a pre-release
    build of the Lab), whatever Lab version it names, cannot show that the output is the one
    scored, so it is not drawn next to the score."""
    from real2simlab.report import write_report

    report = _reported(task_copy, tmp_path / "run")
    (page,) = (report / "tasks").glob("*.html")
    (record,) = (tmp_path / "run").rglob("run/run_record.json")
    data = json.loads(record.read_text())
    assert data.pop("output_fingerprint") and data["status"] == "succeeded"
    record.write_text(json.dumps({**data, "lab_version": "0.9.0"}))
    (tracks,) = (tmp_path / "run").rglob("run/output/tracks.npz")
    import numpy as np

    with np.load(tracks) as arrays:
        moved = {name: arrays[name] + 1.0 for name in arrays.files}
    np.savez(tracks, **moved)
    write_report(tmp_path / "run")
    text = page.read_text()
    assert "overlay_oracle" not in text and "cannot be checked" in text
    assert not list(report.rglob("overlay_*.mp4"))


def test_a_recording_that_may_no_longer_be_shared_leaves_the_report(
    task_copy: Path, tmp_path: Path
) -> None:
    report = _reported(task_copy, tmp_path / "run")
    assert list((report / "media").rglob("*.mp4"))
    data = read_task_config(task_copy / "task.toml").to_json()
    data["redistributable"] = False  # e.g. the license was found to forbid it
    write_task_config(TaskConfig.model_validate(data), task_copy / "task.toml")
    report = _reported(task_copy, tmp_path / "run")
    assert not list((report / "media").rglob("*.mp4")), "report/ still shares the recording"


# --- A run directory's records damaged or edited -----------------------------------------------


def _scores(out: Path) -> list[tuple[Any, ...]]:
    rows = json.loads((out / "results.json").read_text())["rows"]
    return sorted((r["method"], r["task"], r["trial"], r["reproduction_skill"]) for r in rows)


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tasks = tmp_path_factory.mktemp("tasks")
    (task,) = SyntheticBenchmark().build(tasks, select=["bouncing_ball"], tiers=["calibrated"])
    out = tmp_path_factory.mktemp("finished") / "run"
    config = _config([task], [{"name": "static"}, {"name": "oracle"}, {"name": "fake_agent"}])
    _run(config, out, backend=LocalBackend(quiet=True))
    return out


DAMAGED_JSON = {
    "null": "null",
    "list": "[]",
    "empty": "{}",
    "text": "not json",
}


EDITS: dict[str, Callable[[dict[str, Any]], None]] = {
    "rows=null": lambda data: data.update(rows=None),
    "row={}": lambda data: data["rows"].__setitem__(0, {}),
    "tasks=list": lambda data: data.update(tasks=[]),
    "task=null": lambda data: data["tasks"].update({next(iter(data["tasks"])): None}),
}
RECORD_DAMAGES = [
    *((name, damage) for name in ("results.json", "experiment.json") for damage in DAMAGED_JSON),
    ("results.json", "rows=null"),
    ("results.json", "row={}"),
    ("experiment.json", "tasks=list"),
    ("experiment.json", "task=null"),
]


@pytest.mark.parametrize(("name", "damage"), RECORD_DAMAGES, ids=lambda v: v)
def test_damaged_records_of_a_run_are_clear_errors_naming_them(
    name: str, damage: str, finished_run: Path, tmp_path: Path
) -> None:
    """``r2s report`` and ``r2s train export`` read a run's records: a damaged one (edited by
    hand, a copy cut short) is an error naming it and how to rewrite it, never a traceback."""
    from real2simlab.report import write_report

    out = Path(shutil.copytree(finished_run, tmp_path / "run"))
    path = out / name
    if damage in EDITS:
        data = json.loads(path.read_text())
        EDITS[damage](data)
        path.write_text(json.dumps(data))
    else:
        path.write_text(DAMAGED_JSON[damage])
    with pytest.raises(SetupError, match=name.replace(".", r"\.")) as error:
        write_report(out)
    assert "run the same r2s eval or r2s run command again" in str(error.value)
    with pytest.raises(SetupError, match=name.replace(".", r"\.")):
        export_dataset(tmp_path / "data", runs=[out], allow_eval_tasks=True)


def test_a_damaged_agent_record_is_said_not_crashed_on(finished_run: Path, tmp_path: Path) -> None:
    from real2simlab.report import write_report

    out = Path(shutil.copytree(finished_run, tmp_path / "run"))
    (record,) = out.glob("builds/*/fake_agent/trial-0/work/agent/agent_record.json")
    for damage in ("null", "[]", '{"network": "x"}', "not json"):
        record.write_text(damage)
        page = write_report(out)  # the report says so where the record would be shown
        text = "".join(p.read_text() for p in page.parent.rglob("*.html"))
        assert "agent record" in text and "is damaged" in text
        with pytest.raises(SetupError, match=r"agent_record\.json is damaged"):
            export_dataset(tmp_path / "data", runs=[out], allow_eval_tasks=True)


@pytest.mark.parametrize("damage", sorted(DAMAGED_JSON))
@pytest.mark.parametrize("record", ["build_record.json", "run_record.json", "seed-0.json"])
def test_a_damaged_step_record_is_redone_by_the_same_command(
    record: str, damage: str, finished_run: Path, tmp_path: Path
) -> None:
    """Running the same command again redoes a step whose record is damaged, and gives the
    same scores; it never stops on it."""
    out = Path(shutil.copytree(finished_run, tmp_path / "run"))
    before = _scores(out)
    for path in out.glob(f"**/static/**/{record}"):
        path.write_text(DAMAGED_JSON[damage])
    config = json.loads((out / "experiment.json").read_text())["config"]
    experiment = Experiment(
        parse_experiment(config), out, backend=LocalBackend(quiet=True), log=lambda line: None,
        report=False,
    )  # fmt: skip
    experiment.run()
    assert _scores(out) == before


@pytest.mark.parametrize("step", ["builds", "runs"])
def test_a_step_key_that_is_not_text_is_redone_by_the_same_command(
    step: str, finished_run: Path, tmp_path: Path
) -> None:
    """A step's ``.key`` holding bytes that are not text (a damaged disk, a sync client) is no
    key: the same command again redoes that step and gives the same scores. It used to stop
    every resume with the decoder's words alone (no file named, no fix), and left the finished
    run ``running``."""
    out = Path(shutil.copytree(finished_run, tmp_path / "run"))
    before = _scores(out)
    keys = sorted((out / step).glob("*/static/**/.key"))
    assert keys
    for key in keys:
        key.write_bytes(b"this is not a key {\x00\xff")
    experiment = _again(out)
    assert _scores(out) == before
    assert (experiment.progress.builds if step == "builds" else experiment.progress.runs) >= 1


def test_a_key_that_cannot_be_read_is_no_key(tmp_path: Path) -> None:
    """Every step reads its ``.key`` the same way: one that is not text or cannot be read is
    no key (the step is redone), also for a kept official score, which is then not reused."""
    from real2simlab.orchestrate.experiment import (
        NATIVE_REPORT,
        _cached_report,  # pyright: ignore[reportPrivateUsage]
        _report_key,  # pyright: ignore[reportPrivateUsage]
    )
    from real2simlab.orchestrate.layout import KEY_FILE, read_key, write_key

    write_key(tmp_path, "abc")
    assert read_key(tmp_path) == "abc"
    (tmp_path / KEY_FILE).write_bytes(b"\xff\xfe")
    assert read_key(tmp_path) is None
    (tmp_path / NATIVE_REPORT).write_text("{}\n")
    write_key(tmp_path, _report_key("k", "{}\n"))
    assert _cached_report(tmp_path, "k") == {}
    (tmp_path / KEY_FILE).write_bytes(b"\xff\xfe")
    assert _cached_report(tmp_path, "k") is None
    if os.geteuid() != 0:  # root reads it all the same
        write_key(tmp_path, "abc")
        (tmp_path / KEY_FILE).chmod(0)
        try:
            assert read_key(tmp_path) is None
        finally:
            (tmp_path / KEY_FILE).chmod(0o644)


# --- 15 a record without its integrity field is not taken as unchanged --------------------------


def _again(out: Path) -> Experiment:
    """The same command again on the run directory ``out``."""
    config = json.loads((out / "experiment.json").read_text())["config"]
    experiment = Experiment(
        parse_experiment(config), out, backend=LocalBackend(quiet=True), log=lambda line: None,
        report=False,
    )  # fmt: skip
    experiment.run()
    return experiment


def _outcomes(out: Path) -> list[tuple[Any, ...]]:
    rows = json.loads((out / "results.json").read_text())["rows"]
    return sorted(
        (r["method"], r["task"], r["trial"], r["seed"], r["reproduction_skill"],
         r.get("session_consistent"))
        for r in rows
    )  # fmt: skip


def _static_output(out: Path) -> Path:
    (tracks,) = out.glob("runs/*/static/trial-0/seed-0/run/output/tracks.npz")
    return tracks


def _static_program_as_oracles(out: Path) -> None:
    (static,) = out.glob("builds/*/static/trial-0/twin/program")
    (oracle,) = out.glob("builds/*/oracle/trial-0/twin/program")
    for name in ("run.py", "model.json", "trajectory.npz"):
        shutil.copy2(static / name, oracle / name)


def _static_output_as_oracles(out: Path) -> None:
    (oracle,) = out.glob("runs/*/oracle/trial-0/seed-0/run/output/tracks.npz")
    shutil.copyfile(_static_output(out), oracle)


def _static_output_as_the_agents_session(out: Path) -> None:
    (session,) = out.glob("builds/*/fake_agent/trial-0/work/workspace/output/tracks.npz")
    shutil.copyfile(_static_output(out), session)


INTEGRITY_FIELDS: dict[str, tuple[str, Callable[[Path], None]]] = {
    "twin_fingerprint": ("builds/*/oracle/trial-0/build_record.json", _static_program_as_oracles),
    "output_fingerprint": (
        "runs/*/oracle/trial-0/seed-0/run/run_record.json",
        _static_output_as_oracles,
    ),
    "session_output_fingerprint": (
        "builds/*/fake_agent/trial-0/build_record.json",
        _static_output_as_the_agents_session,
    ),
}
"""Each fingerprint a step record of this Lab always holds, where it is, and an edit of what it
vouches for (the static baseline's program or output in place of another method's)."""


@pytest.mark.slow
@pytest.mark.parametrize("field", sorted(INTEGRITY_FIELDS))
def test_a_step_record_without_its_fingerprint_is_redone_not_reused(
    field: str, finished_run: Path, tmp_path: Path
) -> None:
    """A build or run record holds the fingerprint of what the step left (the delivered twin,
    the session output, a successful run's output). One without it (edited, rewritten by a tool
    that drops optional fields, or written by a pre-release build of the Lab) is redone, not
    reused with whatever is there now, which would score another program or output under the
    method's name; whatever Lab version the record names."""
    from real2simlab import __version__

    out = Path(shutil.copytree(finished_run, tmp_path / "run"))
    before = _outcomes(out)
    pattern, edit = INTEGRITY_FIELDS[field]
    (record,) = out.glob(pattern)
    for version in (__version__, "0.9.0"):
        data = json.loads(record.read_text())  # (the second time, as the step redone wrote it)
        assert data.pop(field) and data["lab_version"] == __version__
        record.write_text(json.dumps({**data, "lab_version": version}))
        edit(out)
        experiment = _again(out)
        assert _outcomes(out) == before, "scored what was edited in, under the method's name"
        assert experiment.progress.builds + experiment.progress.runs > 0, version


@pytest.mark.slow
def test_a_session_output_the_build_could_not_fingerprint_is_reused(
    finished_run: Path, tmp_path: Path
) -> None:
    """A session output with a symbolic link in it has no fingerprint (the build record leaves
    it out): while it is as it was, the build is reused, not redone on every run."""
    out = Path(shutil.copytree(finished_run, tmp_path / "run"))
    (record,) = out.glob("builds/*/fake_agent/trial-0/build_record.json")
    data = json.loads(record.read_text())
    (record.parent / data["session_output"] / "latest.npz").symlink_to("tracks.npz")
    data.pop("session_output_fingerprint")  # what the build wrote for such an output
    record.write_text(json.dumps(data))
    experiment = _again(out)
    assert (experiment.progress.builds, experiment.progress.reused_builds) == (0, 3)


@pytest.mark.slow
@pytest.mark.parametrize("damage", ["a row without its hash", "no status"])
def test_results_that_cannot_be_checked_are_not_one_complete_set(
    damage: str, finished_run: Path, tmp_path: Path
) -> None:
    """Every row of a ``results.json`` holds the SHA-256 of its result file, and the run
    directory says whether the run finished (``state/status.json``). Without them (edited, or
    written by a pre-release build of the Lab), a result file edited since cannot be told from
    the one scored: the run is not exported, and the report says so, whatever Lab version
    ``results.json`` names. The same command again finishes the run directory."""
    from real2simlab.report.build import ReportBuilder

    out = Path(shutil.copytree(finished_run, tmp_path / "run"))
    before = _outcomes(out)
    summary = json.loads((out / "results.json").read_text())
    if damage == "no status":
        RunLayout(out).status_file.unlink()
    else:
        (row,) = [row for row in summary["rows"] if row["method"] == "oracle"]
        row.pop("result_sha256")
        (out / "results.json").write_text(json.dumps(summary))
        (result,) = out.glob("results/*/oracle/trial-0/seed-0.json")
        data = json.loads(result.read_text())
        data["summary"]["reproduction_skill"] = data["headline"]["value"] = 0.0
        result.write_text(json.dumps(data))
    assert incomplete(out)
    with pytest.raises(SetupError, match="not one complete set") as raised:
        export_dataset(tmp_path / "data", runs=[out], allow_eval_tasks=True)
    assert "pre-release build" in str(raised.value)
    assert "run the same r2s eval or r2s run command again" in (raised.value.fix or "")
    builder = ReportBuilder(out)
    builder.build()
    assert any("not one complete set" in warning for warning in builder.warnings)
    (out / "results.json").write_text(json.dumps({**summary, "lab_version": "0.9.0"}))
    assert incomplete(out)  # whatever Lab version it names
    _again(out)  # what the fix says
    assert incomplete(out) == [] and _outcomes(out) == before


def test_a_scorer_process_stops_with_the_experiment(tmp_path: Path) -> None:
    """An official scorer runs in a thread of the experiment: Ctrl-C (``cancel_all``) stops
    its process at once, not when it finishes."""
    import threading
    import time

    from real2simlab.sandbox.process import cancel_all, reset_cancel, run_captured

    raised: list[BaseException] = []
    marker = tmp_path / "still-running"

    def scorer() -> None:
        try:
            run_captured(
                [sys.executable, "-c", f"import time\ntime.sleep(30)\nopen({str(marker)!r}, 'w')"]
            )
        except BaseException as error:
            raised.append(error)

    thread = threading.Thread(target=scorer)
    started = time.monotonic()
    thread.start()
    time.sleep(0.5)
    try:
        cancel_all()
        thread.join(10)
    finally:
        reset_cancel()
    assert not thread.is_alive() and time.monotonic() - started < 5
    assert len(raised) == 1 and isinstance(raised[0], KeyboardInterrupt)
    time.sleep(0.2)
    assert not marker.exists()


def test_a_command_says_it_started_only_once_its_process_exists(tmp_path: Path) -> None:
    """``run_captured(started=...)`` is called once the command's process exists: never for
    one that cannot be spawned or that a stopped experiment never started (a harness on the
    host then ran nothing of its agent). If it raises, the process is stopped at once and the
    error propagates."""
    import time

    from real2simlab.sandbox.process import cancel_all, reset_cancel, run_captured

    calls: list[str] = []
    ran = run_captured([sys.executable, "-c", "print('ran')"], started=lambda: calls.append("ran"))
    assert ran.stdout.strip() == "ran" and calls == ["ran"]
    with pytest.raises(FileNotFoundError):
        run_captured([str(tmp_path / "missing")], started=lambda: calls.append("missing"))
    cancel_all()
    try:
        with pytest.raises(KeyboardInterrupt):
            run_captured([sys.executable, "-c", "pass"], started=lambda: calls.append("stopped"))
    finally:
        reset_cancel()
    assert calls == ["ran"]

    def cannot_say_so() -> None:
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    marker = tmp_path / "still-running"
    late = f"import time\ntime.sleep(1)\nopen({str(marker)!r}, 'w')"
    with pytest.raises(OSError, match=os.strerror(errno.ENOSPC)):
        run_captured([sys.executable, "-c", late], started=cannot_say_so)
    time.sleep(1.5)
    assert not marker.exists()  # stopped


def test_a_dataset_replaced_while_a_model_trains_is_refused(
    train_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``r2s train export --overwrite`` to the dataset while ``r2s train fit`` reads it: the
    model may mix two versions, so no checkpoint is written (with the new or the old record)."""
    from real2simlab.methods import get_method
    from real2simlab.train import fit_method

    data = tmp_path / "data"
    export_dataset(data, tasks=[train_task.parent])
    method = get_method("nearest_twin")
    train = method.train

    def train_while_exported(context: Any) -> None:
        train(context)
        export_dataset(data, tasks=[train_task.parent], overwrite=True)

    monkeypatch.setattr(method, "train", train_while_exported)
    with pytest.raises(SetupError, match="was replaced or changed while nearest_twin trained"):
        fit_method(method, data, tmp_path / "model", log=lambda _: None)
    assert not (tmp_path / "model").exists()


def test_a_recording_edited_while_a_model_trains_is_refused(
    train_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file a row points to, replaced while the model trains (an editor, a sync): the model
    learned from it, but the dataset's prints describe the one it replaced, so a task holding
    that recording would not be held out. No checkpoint is written."""
    from real2simlab.methods import get_method
    from real2simlab.train import fit_method

    data = tmp_path / "data"
    export_dataset(data, tasks=[train_task.parent])
    row = json.loads((data / "tasks.jsonl").read_text().splitlines()[0])
    capture = data / row["capture"]
    view = next(iter(json.loads(capture.read_text())["views"].values()))
    video = capture.parent / view["rgb"]["path"]
    method = get_method("nearest_twin")
    train = method.train

    def train_while_edited(context: Any) -> None:
        with video.open("ab") as file:  # the same name, other content
            file.write(b"\0")
        train(context)

    monkeypatch.setattr(method, "train", train_while_edited)
    with pytest.raises(SetupError, match="was replaced or changed while nearest_twin trained"):
        fit_method(method, data, tmp_path / "model", log=lambda _: None)
    assert not (tmp_path / "model").exists()


def test_a_program_changed_while_it_is_exported_is_refused(
    train_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A program that something writes to while the export copies it is not the one the reward
    belongs to: the export stops instead of pairing them."""
    from real2simlab.train import dataset

    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    run = tmp_path / "run"
    _run(_config([tasks], [{"name": "oracle"}]), run)
    copytree = shutil.copytree

    def copy_while_written(source: Any, target: Any, *args: Any, **options: Any) -> Any:
        copied = copytree(source, target, *args, **options)
        if Path(source).name == "program":  # the whole program, not one of its directories
            (Path(source) / "added.py").write_text("# written during the export\n")
        return copied

    monkeypatch.setattr(dataset.shutil, "copytree", copy_while_written)
    with pytest.raises(SetupError, match="changed"):
        export_dataset(tmp_path / "data", tasks=[tasks], runs=[run])
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("edit", ["configuration", "reference"])
def test_a_task_edited_while_a_solution_is_scored_gives_no_reward(
    edit: str, task_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reward is the score of one version of the task: an edit while the program runs (a
    configuration read before, a reference read after) gives an error, not a mixed reward."""
    from real2simlab.task.io import read_task_config, write_task_config
    from real2simlab.train import reward

    program = _truth_program(task_copy, tmp_path / "program")
    run_twin = reward.run_twin

    def run_while_edited(*args: Any, **options: Any) -> Any:
        ran = run_twin(*args, **options)
        if edit == "configuration":
            config = read_task_config(task_copy / "task.toml")
            write_task_config(
                config.model_copy(update={"title": "edited"}), task_copy / "task.toml"
            )
        else:
            (task_copy / "hidden" / "notes.txt").write_text("added while it was scored\n")
        return ran

    monkeypatch.setattr(reward, "run_twin", run_while_edited)
    with pytest.raises(SetupError, match="changed while the solution was scored"):
        score_solution(
            task_copy, program, backend=LocalBackend(quiet=True), work_dir=tmp_path / "w"
        )
    monkeypatch.undo()
    scored = score_solution(task_copy, program, backend=LocalBackend(quiet=True))  # as it now is
    assert scored.reward > 0.9


def test_a_task_edited_while_it_is_exported_is_refused(
    train_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The task's copy in a dataset is the version its rows name (the one results are matched
    by) only if nothing edits the task during the export."""
    from real2simlab.train import dataset

    task = _copy(train_task, tmp_path / "tasks" / "t")
    copytree = shutil.copytree

    def copy_while_edited(source: Any, target: Any, *args: Any, **options: Any) -> Any:
        copied = copytree(source, target, *args, **options)
        if Path(source) == task / "visible":
            (task / "hidden" / "notes.txt").write_text("added during the export\n")
        return copied

    monkeypatch.setattr(dataset.shutil, "copytree", copy_while_edited)
    with pytest.raises(SetupError, match="changed while it was exported"):
        export_dataset(tmp_path / "data", tasks=[task.parent])
    assert not (tmp_path / "data").exists()
    monkeypatch.undo()
    export_dataset(tmp_path / "data", tasks=[task.parent])  # the task as it now is


def test_a_result_changed_while_it_is_exported_is_refused(
    train_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A result file replaced while the export runs (another command on the run directory, a
    sync) is not the one ``results.json`` was computed from: the reward the export would read
    from it is not the run's, so the export stops instead of publishing it."""
    from real2simlab.train import dataset

    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    run = tmp_path / "run"
    _run(_config([tasks], [{"name": "oracle"}], seeds=[0, 1]), run)
    (second,) = run.glob("results/*/oracle/trial-0/seed-1.json")
    copytree = shutil.copytree
    replaced: list[Path] = []

    def copy_while_replaced(source: Any, target: Any, *args: Any, **options: Any) -> Any:
        copied = copytree(source, target, *args, **options)
        if Path(source).name == "program" and not replaced:  # seed 0's program is copied
            data = json.loads(second.read_text())
            data["summary"]["reproduction_skill"] = data["headline"]["value"] = 0.0
            replacement = second.with_name("seed-1.json.tmp")
            replacement.write_text(json.dumps(data))
            replaced.append(replacement.replace(second))  # as write_json or a sync writes
        return copied

    monkeypatch.setattr(dataset.shutil, "copytree", copy_while_replaced)
    with pytest.raises(SetupError, match="changed while") as raised:
        export_dataset(tmp_path / "data", runs=[run])
    assert replaced and "export again" in (raised.value.fix or "")
    assert not (tmp_path / "data").exists()


def test_evaluation_tasks_changed_while_a_dataset_is_checked_against_them_are_refused(
    train_task: Path, task_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dataset records that it was compared with the ``--against`` tasks: an evaluation task
    edited while the export runs (here it takes a training task's video) is no longer the one
    compared with, so the dataset is not published with that claim."""
    from real2simlab.train import dataset

    staging_dir = dataset.staging_dir

    def staged_while_edited(out: Path) -> Path:  # after the evaluation tasks were read
        (video,) = (task_copy / "visible").rglob("*.mp4")
        (training,) = (train_task / "visible").rglob("*.mp4")
        shutil.copyfile(training, video)
        return staging_dir(out)

    monkeypatch.setattr(dataset, "staging_dir", staged_while_edited)
    with pytest.raises(SetupError, match="changed while") as raised:
        export_dataset(tmp_path / "data", tasks=[train_task.parent], against=[task_copy])
    assert "export again" in (raised.value.fix or "")
    assert not (tmp_path / "data").exists()
    monkeypatch.undo()
    with pytest.raises(SetupError, match="nothing could be exported"):  # it overlaps them now
        export_dataset(tmp_path / "data", tasks=[train_task.parent], against=[task_copy])


def test_a_delivered_result_without_its_programs_fingerprint_is_not_exported(
    train_task: Path, tmp_path: Path
) -> None:
    """A delivered result names the fingerprint of the program it scored: without it (the
    result file edited, or written by a pre-release build of the Lab, whatever version it
    names), the program in the run directory cannot be shown to be that one, so its reward is
    not exported with it (here another program is in its place). The error is about the
    program, and its fix is the one that works: the same command on the run directory, which
    scores it again."""
    import hashlib

    tasks = _copy(train_task, tmp_path / "tasks" / "t")
    run = tmp_path / "run"
    config = _config([tasks], [{"name": "oracle"}])
    _run(config, run)
    (result,) = run.glob("results/*/oracle/trial-0/seed-0.json")
    data = json.loads(result.read_text())
    assert data.pop("twin_fingerprint") and data["delivered"]
    result.write_text(json.dumps({**data, "lab_version": "0.9.0"}))
    summary = json.loads((run / "results.json").read_text())  # its hash made to match
    summary["rows"][0]["result_sha256"] = hashlib.sha256(result.read_bytes()).hexdigest()
    (run / "results.json").write_text(json.dumps(summary))
    (program,) = run.glob("builds/*/oracle/trial-0/twin/program/run.py")
    program.write_text("raise SystemExit(3)\n")
    with pytest.raises(SetupError, match="cannot be matched with the program they") as raised:
        export_dataset(tmp_path / "data", tasks=[tasks], runs=[run])
    assert "no fingerprint of the program" in str(raised.value)
    fix = raised.value.fix or ""
    assert "run the same r2s eval or r2s run command again" in fix and str(run.resolve()) in fix
    assert "--skip-unmatched" in fix and "--tasks" not in fix
    summary = export_dataset(tmp_path / "data", tasks=[tasks], runs=[run], skip_unmatched=True)
    assert summary.runs == 0 and any("no fingerprint of the program" in s for s in summary.skipped)
    _run(config, run)  # what the fix says
    assert export_dataset(tmp_path / "data2", tasks=[tasks], runs=[run]).runs == 1


def test_results_unmatched_for_both_reasons_are_given_both_fixes(tmp_path: Path) -> None:
    """Results whose task cannot be matched need the task as it was; results whose program
    cannot be matched need the same command again on their run directory: the error names
    each, with its fix."""
    from real2simlab.train.dataset import _unmatched_error

    task = "run: oracle on t (trial 0, seed 0): the task has changed since the run scored it"
    program = "run: oracle on u (trial 0, seed 0): the program in the run directory changed"
    alone = _unmatched_error([task], [])
    assert "with the task they were scored on" in str(alone) and "program they" not in str(alone)
    assert alone.fix == (
        "give the tasks as they were when the runs scored them (--tasks DIR), or pass "
        "--skip-unmatched to leave these results out"
    )
    both = _unmatched_error([task], [(tmp_path / "run", program)])
    assert "with the task they were scored on" in str(both)
    assert "1 result(s) cannot be matched with the program they scored" in str(both)
    again = (
        "run the same r2s eval or r2s run command again with the run directory as --out "
        f"({tmp_path / 'run'})"
    )
    assert "--tasks DIR" in (both.fix or "") and again in (both.fix or "")
    assert "--skip-unmatched" in (both.fix or "")


def test_an_agent_stopped_by_this_machines_storage_has_no_result(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent that exits normally without a delivery is the method's result, unless this
    machine's storage explains it (its writes stopped at the size limit, a full disk)."""
    from real2simlab.methods import get_method
    from real2simlab.methods.base import BuildContext
    from real2simlab.sandbox import process
    from real2simlab.task import load_task
    from real2simlab.task.instruction import task_statement

    task = load_task(bouncing_task)
    method = get_method("fake_agent", mode="nothing")
    backend = LocalBackend(quiet=True)

    def context(name: str) -> BuildContext:
        return BuildContext(
            task_id=task.id, task_hash=task.content_hash(), config=task.config,
            statement=task_statement(task), visible_dir=task.visible_dir,
            work_dir=tmp_path / name / "work", twin_dir=tmp_path / name / "twin", backend=backend,
        )  # fmt: skip

    plain = method.build(context("plain"))
    assert plain.twin is None and plain.failure and plain.infra_error is None
    monkeypatch.setattr(
        process, "storage_problem", lambda *_: "solution/run.py reached this machine's file-size "
        "limit (65536 bytes, `ulimit -f`); raise it and run again"
    )  # fmt: skip
    limited = method.build(context("limited"))
    assert limited.twin is None and "file-size limit" in (limited.infra_error or "")


def _leaking_agent(monkeypatch: pytest.MonkeyPatch, key: str, then: str = "") -> None:
    """The fake agent, injected with ``key`` (a made-up canary), prints it (bootstrap redacts
    that), then writes it itself into its solution and straight into the Lab's log directory,
    then runs ``then`` (Python, with ``key`` and ``logs``)."""
    from real2simlab.agents import fake
    from real2simlab.agents.base import Credentials

    leak = (
        "import json, os, subprocess, sys\n"
        "cli, workspace, logs = sys.argv[1:4]\n"
        "subprocess.run([sys.executable, cli, '--mode', 'physical', '--workspace', workspace],"
        " check=True)\n"
        "key = os.environ['R2S_TEST_AGENT_KEY']\n"
        "print('debug: the key is', key, flush=True)\n"
        "with open(os.path.join(workspace, 'solution', 'notes.txt'), 'w') as notes:\n"
        "    notes.write(key + '\\n')\n"
        "with open(os.path.join(logs, 'trajectory.jsonl'), 'a') as events:\n"
        "    events.write(json.dumps({'type': 'note', 'text': key}) + '\\n')\n"
    ) + then
    command = ["python", "-c", leak, f"{fake.FAKE_DIR}/fake_agent_cli.py", "/workspace",
               "/r2s/logs"]  # fmt: skip
    monkeypatch.setattr(
        fake.FakeAgent,
        "credentials",
        lambda self, context: Credentials(env={"R2S_TEST_AGENT_KEY": key}, description="test"),
    )
    monkeypatch.setattr(fake.FakeAgent, "command", lambda self, context, workspace: command)


def _holding(value: str, *roots: Path) -> list[str]:
    return [
        path.relative_to(root).as_posix()
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and value.encode() in path.read_bytes()
    ]


def test_an_agent_whose_files_this_machine_cannot_redact_has_no_result(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Injected credentials are removed from what the agent wrote before its solution is
    packaged. When the disk fills while those files are rewritten, they are removed instead
    (which needs no space): neither the run directory nor its report holds the key, nothing is
    delivered, and the build is not a result (redone, not scored). Made-up canary only."""
    import errno
    import importlib

    from real2simlab.orchestrate import run_experiment

    redact = importlib.import_module("real2simlab.agents.redact")  # the module

    key = "cnry-full-disk-key-3e9b"
    _leaking_agent(monkeypatch, key)

    def full_disk(source: Any, target: Any, pattern: Any, longest: int) -> int:
        target.write(source.read(4))  # part of the copy is written, then the disk is full
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(redact, "_copy_redacted", full_disk, raising=False)
    run = tmp_path / "run"
    config = _config([bouncing_task], [{"name": "fake_agent", "mode": "physical"}])
    results = run_experiment(config, run, backend=LocalBackend(quiet=True), log=lambda line: None)
    assert _holding(key, run) == []  # the run directory, its records and its report
    assert list(run.glob("report/tasks/*.html")) and not list(run.rglob(".r2s-redacting-*"))
    assert results.methods[0].infra_errors == 1
    (row,) = results.to_json()["rows"]
    assert "could not replace the injected credentials" in row["infra_error"]
    assert "No space left on device" in row["infra_error"]
    assert "removed instead" in row["infra_error"]
    (record,) = run.glob("builds/*/fake_agent/trial-0/work/agent/agent_record.json")
    assert json.loads(record.read_text())["delivered"] is False


def test_an_agent_cannot_turn_its_result_into_no_result_with_a_long_path(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent that writes its key into a file whose path leaves no room for the rewritten
    copy's name (just under PATH_MAX) gets that file removed, and its build stays a result: a
    failure the agent caused is never this machine's, which would make it not scored."""
    from real2simlab.orchestrate import run_experiment

    key = "cnry-long-path-key-9a40"
    deep = (  # a short name in a directory whose path is just under PATH_MAX
        "limit = os.pathconf(logs, 'PC_PATH_MAX')\n"
        "os.chdir(logs)\n"
        "length = len(os.getcwd())\n"
        "while length + 201 <= limit - 10:\n"
        "    os.mkdir('d' * 200); os.chdir('d' * 200); length += 201\n"
        "rest = limit - 10 - length - 1\n"
        "if rest > 0:\n"
        "    os.mkdir('e' * rest); os.chdir('e' * rest)\n"
        "with open('k', 'w') as deep:\n"
        "    deep.write(key + '\\n')\n"
    )
    _leaking_agent(monkeypatch, key, then=deep)
    run = tmp_path / "run"
    config = _config([bouncing_task], [{"name": "fake_agent", "mode": "physical"}])
    results = run_experiment(config, run, backend=LocalBackend(quiet=True), log=lambda line: None)
    assert _holding(key, run) == []
    assert results.methods[0].infra_errors == 0 and results.methods[0].delivered_rate == 1.0
    (record,) = run.glob("builds/*/fake_agent/trial-0/work/agent/agent_record.json")
    assert json.loads(record.read_text())["redactions"] >= 4  # output, solution, log, deep file
    assert not [path for path in run.rglob("k") if path.is_file()]


class _WroteItsKeyThenKilled(_KilledFromOutside):
    """The agent started, wrote its injected key into its solution and straight into its log,
    then something outside its sandbox killed the sandbox."""

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        result = super().run(spec, log_path)
        mounts = {mount.target: Path(mount.source) for mount in spec.mounts}
        key = spec.secret_env["R2S_TEST_AGENT_KEY"]
        (mounts["/workspace"] / "solution" / "notes.txt").write_text(key + "\n")
        with (mounts["/r2s/logs"] / "trajectory.jsonl").open("a") as events:
            events.write(json.dumps({"type": "note", "text": key}) + "\n")
        return result


def test_an_agent_killed_after_it_wrote_its_key_leaves_neither_its_key_nor_a_result(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build whose sandbox failed around the agent is not a result, and what the agent wrote
    before is cleaned all the same: no file of the build holds its injected key, and its record
    names the cleaned trajectory. Made-up canary only."""
    key = "qa-canary-killed-agent-" + "k" * 20
    _leaking_agent(monkeypatch, key)
    outcome = get_method("fake_agent", mode="physical").build(
        _agent_context(bouncing_task, tmp_path, _WroteItsKeyThenKilled(quiet=True))
    )
    assert outcome.twin is None and "signal 9" in (outcome.infra_error or "")
    record = _agent_record(tmp_path)
    assert record["exit_status"] == "failed" and record["trajectory"] == "agent/trajectory.jsonl"
    assert record["redactions"] >= 2  # its solution and its log
    assert _holding(key, tmp_path / "work") == []


# --- 16 a method's own code fails, or a record its tool wrote is damaged ------------------------


@pytest.mark.parametrize("name", ["static", "fake_agent"])
def test_a_method_that_raises_what_it_should_not_stops_only_its_own_build(
    name: str, bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A method whose code raises an error it should have reported (a bug of its adapter, a
    record of its tool it could not read) leaves what its build did unknown: the build is not a
    result (``infra_error``, saying what was raised; the traceback goes to ``r2s.log``, not to
    the terminal; a key-shaped string in either is redacted, as in every record), and it stops
    no other build. It is not run again by this command (an agent may have run, and spent:
    counted), and the next run builds it again. Such an error used to stop the whole
    experiment, with no result for any method."""
    from real2simlab.methods.baselines import StaticBaseline

    calls: list[str] = []
    key = "sk-proj-" + "q" * 40  # shaped like a key: what an API client's error may repeat
    # ... and a credential of this process shaped like no key: only its value tells it (it
    # was in r2s.log, the build record and the results: stale-and-damaged K2)
    canary = "canary-7f3e-proxy-credential"
    monkeypatch.setenv("R2S_TEST_PROXY_TOKEN", canary)

    def broken(*args: Any) -> Any:
        calls.append(name)
        raise KeyError(f"a field its tool no longer writes (sent with {key} via {canary})")

    if name == "static":
        monkeypatch.setattr(StaticBaseline, "transform", broken)
    else:
        monkeypatch.setattr(FakeAgent, "command", broken)
    lines: list[str] = []
    config = _config([bouncing_task], [{"name": name}, {"name": "oracle"}], retries=2)
    out = tmp_path / "run"
    experiment = Experiment(
        config, out, backend=LocalBackend(quiet=True), log=lines.append, report=False
    )
    rows = {row["method"]: row for row in experiment.run().to_json()["rows"]}
    assert rows["oracle"]["reproduction_skill"] == pytest.approx(1.0)
    assert "KeyError" in rows[name]["infra_error"] and "r2s.log" in rows[name]["infra_error"]
    assert calls == [name]  # not run again by this command
    log = (out / "r2s.log").read_text()
    assert "Traceback" in log and "a field its tool no longer writes" in log
    assert not any("Traceback" in line for line in lines)
    (record,) = out.glob(f"builds/*/{name}/trial-0/build_record.json")
    results = [path.read_text() for path in out.glob(f"results/*/{name}/**/*.json")]
    said = [log, *lines, record.read_text(), (out / "results.json").read_text(), *results]
    assert not any(key in text or canary in text for text in said)
    if name == "fake_agent":
        budget = json.loads((out / "state" / "budget.json").read_text())
        assert budget["agent_runs"] == 1  # it may have run: spent, not given back
    monkeypatch.undo()
    again, experiment = _run(config, out, LocalBackend(quiet=True))
    assert (experiment.progress.builds, experiment.progress.reused_builds) == (1, 1)
    assert again.to_json()["rows"] and not any(r["infra_error"] for r in again.to_json()["rows"])


HARNESS = '''#!{python}
"""A stand-in for mini-swe-agent's interpreter: it can import, it has a version, and its run of
the driver ends without the summary the driver prints last ({said!r} instead)."""
import sys
if sys.argv[1:2] == ["-c"]:
    print("0.0.0" if "importlib.metadata" in sys.argv[2] else "", end="")
    sys.exit(0)
print({said!r})
'''


@pytest.mark.parametrize("said", ["Traceback (most recent call last):", "[1]", ""])
def test_a_harness_whose_summary_cannot_be_read_is_not_a_result(
    said: str, bouncing_task: Path, tmp_path: Path
) -> None:
    """mini-swe-agent's driver always ends by printing its summary of the run (whether the model
    answered, whether the sandbox failed): a harness that ends without one that can be read (a
    line that is not JSON, or not an object, or nothing) leaves how the agent's run went
    unknown. Its build is not a result and its run is spent (``failed``: not run again by this
    command). Such a summary used to stop the whole experiment, or end in a traceback."""
    harness = tmp_path / "harness"
    harness.write_text(HARNESS.format(python=sys.executable, said=said))
    harness.chmod(0o755)
    script = tmp_path / "script.json"
    script.write_text("[]")
    method = get_method("mini_swe", scripted=str(script), harness_python=str(harness))
    outcome = method.build(_agent_context(bouncing_task, tmp_path, LocalBackend(quiet=True)))
    assert outcome.twin is None
    assert "summary" in (outcome.infra_error or ""), outcome
    assert _agent_record(tmp_path)["exit_status"] == "failed"


@pytest.mark.parametrize("failure", ["missing", "hangs"])
def test_an_agent_version_that_cannot_be_asked_is_unknown(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Harbor's and mini-swe-agent's versions are asked for the record after the agent ran: a
    command that cannot run or does not answer leaves the version unknown (``None``) instead
    of throwing the agent's run away."""
    local = LocalBackend(quiet=True)
    command = str(tmp_path / "missing")
    if failure == "hangs":

        def hangs(argv: Any, **kwargs: Any) -> Any:
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout") or 120)

        monkeypatch.setattr(subprocess, "run", hangs)
    assert get_method("harbor", harbor=command).version(local, "") is None
    assert get_method("mini_swe", model="m", harness_python=command).version(local, "") is None


def test_a_trajectory_its_record_does_not_name_is_never_shown_or_exported(
    finished_run: Path, tmp_path: Path
) -> None:
    """An agent's record names its trajectory once the injected credentials are out of it. A
    trajectory it does not name (a build stopped before its record, a log the Lab could not
    clean) is not shown in the report nor copied into an exported dataset."""
    from real2simlab.report import write_report

    key = "cnry-unnamed-log-key-27d5"
    out = Path(shutil.copytree(finished_run, tmp_path / "run"))
    (record,) = out.glob("builds/*/fake_agent/trial-0/work/agent/agent_record.json")
    data = json.loads(record.read_text())
    data["trajectory"] = None
    record.write_text(json.dumps(data))
    with (record.parent / "trajectory.jsonl").open("a") as events:
        events.write(json.dumps({"type": "note", "text": key}) + "\n")
    write_report(out)
    export_dataset(tmp_path / "data", runs=[out], allow_eval_tasks=True)
    assert _holding(key, out / "report", tmp_path / "data") == []


def _stalling_tool(tmp_path: Path, name: str, stalls_on: str) -> Path:
    """A ``name`` on ``PATH`` that hangs on ``stalls_on`` (a network that stopped answering)
    and runs the real tool otherwise; ``<name>.started`` appears once it hangs."""
    real = shutil.which(name) or "/bin/false"
    started = tmp_path / f"{name}.started"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    tool = bin_dir / name
    tool.write_text(
        "#!/bin/sh\n"
        f'case " $* " in *" {stalls_on} "*) touch {shlex.quote(str(started))}; exec sleep 300;; '
        "esac\n"
        f'exec {shlex.quote(real)} "$@"\n'
    )
    tool.chmod(0o755)
    return started


def _interrupted(call: Callable[[], object], started: Path) -> BaseException | None:
    """Run ``call`` in a worker thread, stop the experiment once it hangs (``cancel_all``, as
    Ctrl-C or a failing worker does) and return what the worker raised."""
    import threading
    import time

    from real2simlab.sandbox import process

    raised: list[BaseException | None] = [None]

    def work() -> None:
        try:
            call()
        except BaseException as error:  # the test inspects it
            raised[0] = error

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 30
        while not started.exists():
            assert worker.is_alive() and time.monotonic() < deadline, raised[0]
            time.sleep(0.05)
        process.cancel_all()
        worker.join(10)
        assert not worker.is_alive(), "the worker is still waiting for the stalled command"
    finally:
        process.reset_cancel()
    return raised[0]


def test_an_interrupted_experiment_stops_a_stalled_download_of_scorer_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scorer fetching its pinned code over a network that stopped answering is stopped
    with the experiment (Ctrl-C), and leaves no partial checkout behind."""
    from real2simlab.vendor import VendoredRepo

    started = _stalling_tool(tmp_path, "git", "fetch")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    repo = VendoredRepo("demo", "https://example.invalid/demo.git", "0" * 40, "MIT")
    assert isinstance(_interrupted(repo.ensure, started), KeyboardInterrupt)
    assert not list((tmp_path / "cache" / "vendor").iterdir())


def test_a_download_of_scorer_code_that_never_finishes_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from real2simlab import vendor

    _stalling_tool(tmp_path, "git", "fetch")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(vendor, "FETCH_TIMEOUT_S", 1.0)
    repo = vendor.VendoredRepo("demo", "https://example.invalid/demo.git", "0" * 40, "MIT")
    with pytest.raises(SetupError, match="did not finish in 1 s") as error:
        repo.ensure()
    assert "proxy" in (error.value.fix or "")
    assert not list((tmp_path / "cache" / "vendor").iterdir())


def test_an_interrupted_experiment_stops_a_stalled_image_pull(tmp_path: Path) -> None:
    from real2simlab.envs.registry import pull

    started = _stalling_tool(tmp_path, "docker", "pull")
    docker = str(tmp_path / "bin" / "docker")
    image = "ghcr.io/example/r2s-demo:1.0"
    error = _interrupted(lambda: pull(image, image, docker=docker, log=lambda _: None), started)
    assert isinstance(error, KeyboardInterrupt)


def test_every_command_the_lab_waits_for_is_bounded() -> None:
    """A command the Lab runs outside a sandbox either has a timeout or goes through
    ``sandbox.process`` (stopped with the experiment): one that waits for ever (a stalled
    download, a hung Docker daemon) would keep an interrupted command alive and its output
    directory locked. The exceptions say why they cannot hang."""
    import ast

    import real2simlab

    exempt = {
        "sandbox/process.py": "the cancellable runners themselves",
        "twin/sdk/r2s_twin.py": "runs inside the twin's sandbox, which the Lab bounds",
        "envs/blender_scene.py": "runs inside the sandbox",
        "agents/fake_agent_cli.py": "runs inside the agent's sandbox",
        "agents/bootstrap.py": "runs inside the agent's sandbox and forwards signals",
        "_fs.py": "chmod and rm of a local tree",
        "media.py": "ffmpeg on a local file; streaming loops check for cancellation",
    }
    popen_ok = {"envs/__init__.py": "`r2s env build` streams to the terminal (main thread)"}
    package = Path(real2simlab.__file__).parent
    unbounded = []
    for path in sorted(package.rglob("*.py")):
        name = path.relative_to(package).as_posix()
        if name in exempt:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
            ):
                continue
            if node.func.attr == "Popen" and name not in popen_ok:
                unbounded.append(f"{name}:{node.lineno} (Popen)")
            elif node.func.attr in {"run", "check_output", "check_call", "call"} and not any(
                k.arg == "timeout" for k in node.keywords
            ):
                unbounded.append(f"{name}:{node.lineno}")
    assert not unbounded, f"give these a timeout, or use sandbox.process: {unbounded}"
    for name in ("vendor.py", "envs/registry.py"):  # downloads: stopped with the experiment
        text = (package / name).read_text(encoding="utf-8")
        assert "subprocess.run(" not in text and "subprocess.Popen(" not in text, name


def test_every_request_to_hugging_face_is_bounded() -> None:
    """Every download from Hugging Face (a file, a repository snapshot, a list of files) is
    made inside ``_hub.bounded()``. huggingface_hub's client has no time-out of its own and
    lists a repository's files without one: outside the block, a proxy or network that stopped
    answering keeps the command waiting for ever (A3-13); inside it, the request ends and a
    time-out is reported as a failed download."""
    import ast

    import real2simlab

    hub_calls = {
        "hf_hub_download", "snapshot_download", "list_repo_files", "list_repo_tree",
        "repo_info", "dataset_info", "model_info", "get_hf_file_metadata", "HfApi",
        "HfFileSystem",
    }  # fmt: skip
    package = Path(real2simlab.__file__).parent
    found: list[str] = []
    outside: list[str] = []

    def visit(node: ast.AST, where: str, bounded: bool) -> None:
        if isinstance(node, ast.With) and any(
            isinstance(item.context_expr, ast.Call)
            and getattr(item.context_expr.func, "id", getattr(item.context_expr.func, "attr", ""))
            == "bounded"
            for item in node.items
        ):
            bounded = True
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            of_hub = isinstance(func, ast.Attribute) and (
                (isinstance(func.value, ast.Name) and func.value.id in {"hub", "huggingface_hub"})
                or (
                    isinstance(func.value, ast.Call)
                    and getattr(func.value.func, "id", "") == "_hub"
                )
            )
            if name in hub_calls or of_hub:
                found.append(f"{where}:{node.lineno}")
                if not bounded:
                    outside.append(f"{where}:{node.lineno}")
        for child in ast.iter_child_nodes(node):
            visit(child, where, bounded)

    for path in sorted(package.rglob("*.py")):
        where = path.relative_to(package).as_posix()
        text = path.read_text(encoding="utf-8")
        if where != "_hub.py" and "huggingface_hub" in text:
            visit(ast.parse(text), where, False)
    adapters = {"iris", "physcodegym", "physicsiq", "physimcode", "video2world", "fourdcodebench"}
    seen = {where.split("/")[1] for where in found if where.startswith("benchmarks/")}
    assert adapters <= seen, f"the downloads of {sorted(adapters - seen)} were not found"
    assert not outside, f"make these inside `with bounded():` (real2simlab/_hub.py): {outside}"


SILENT_PROXY = (
    "import socket\n"
    "server = socket.socket()\n"
    "server.bind(('127.0.0.1', 0))\n"
    "server.listen(64)\n"
    "print(server.getsockname()[1], flush=True)\n"
    "held = []\n"
    "while True:\n"
    "    held.append(server.accept()[0])\n"
)
"""A proxy that accepts every connection and never answers (a network or proxy that stopped
answering); on this machine only."""


@pytest.fixture
def silent_proxy() -> Iterator[str]:
    """The URL of a ``SILENT_PROXY`` running for the test, in a process group of its own,
    which is killed afterwards."""
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", SILENT_PROXY], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, text=True, start_new_session=True,
    )  # fmt: skip
    try:
        assert process.stdout is not None
        yield f"http://127.0.0.1:{int(process.stdout.readline())}"
    finally:
        _stop_group(process)


def _stop_group(process: subprocess.Popen[str]) -> None:
    """Kill what is left of the process group ``process`` leads, then collect it."""
    import contextlib

    with contextlib.suppress(ProcessLookupError):  # nothing is left in it
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_a_download_from_hugging_face_that_never_answers_ends(
    silent_proxy: str, tmp_path: Path
) -> None:
    """A proxy (or a network) that accepts the connection and never answers: each way the
    Lab lists a Hugging Face repository (a PhySimCode sample's and Video2World's snapshots,
    Physics-IQ's verified release) ends with a ``TimeoutError`` once huggingface_hub's own
    retries are spent, and ``r2s bench build`` with one error line and its fix. They waited for
    ever: huggingface_hub lists files without a time-out, whatever ``HF_HUB_DOWNLOAD_TIMEOUT``
    says (A3-13). That variable now sets how long each request waits (1 s here)."""
    import time

    hub = pytest.importorskip("huggingface_hub")
    if not hasattr(hub, "set_client_factory"):
        pytest.skip("a huggingface_hub before 1.0 (requests) is left without time-outs")
    here = str(Path(__file__).resolve().parents[1])
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith(("HF_", "HUGGING_FACE"))
        and name.lower() not in {"no_proxy", "all_proxy"}
    }
    env.update(
        dict.fromkeys(("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"), silent_proxy)
    )
    env.update(
        PYTHONPATH=here, R2S_CACHE_DIR=str(tmp_path / "cache"), HF_HOME=str(tmp_path / "hf"),
        HF_HUB_DOWNLOAD_TIMEOUT="1", HF_HUB_DISABLE_TELEMETRY="1",
    )  # fmt: skip
    listings = {
        "a PhySimCode sample": "physimcode.download(physimcode.DEFAULT_SAMPLES[0])",
        "Video2World's assets": "video2world.shared_assets()",
        "Physics-IQ's verified release": "physicsiq.case_index('verified')",
    }
    commands = {
        what: [sys.executable, "-c", (
            "import sys\nfrom real2simlab.benchmarks import physicsiq, physimcode, video2world\n"
            f"try:\n    {call}\nexcept TimeoutError as error:\n"
            "    print('timed out:', error)\n    sys.exit(3)\n"
        )]
        for what, call in listings.items()
    }  # fmt: skip
    build = [sys.executable, "-m", "real2simlab.cli.main", "bench", "build", "physimcode",
             "--out", str(tmp_path / "tasks")]  # fmt: skip
    commands["r2s bench build physimcode"] = build
    pipes: dict[str, Any] = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True}
    processes = {
        what: subprocess.Popen(
            argv, env=env, stdin=subprocess.DEVNULL, start_new_session=True, **pipes
        )
        for what, argv in commands.items()
    }
    ended: dict[str, tuple[int, str, str]] = {}
    deadline = time.monotonic() + 150
    try:
        for what, process in processes.items():
            try:
                out, err = process.communicate(timeout=max(1.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                continue
            ended[what] = (process.returncode, out, err)
    finally:
        for process in processes.values():
            _stop_group(process)
    waiting = sorted(set(processes) - set(ended))
    assert not waiting, f"still waiting for Hugging Face after 150 s: {waiting}"
    for what in listings:
        code, out, err = ended[what]
        assert code == 3 and "timed out: Hugging Face did not answer" in out, (what, out, err)
    code, _, err = ended["r2s bench build physimcode"]
    assert code == 1 and "error: a download failed" in err, err
    assert "fix: check this machine's network connection and proxy settings" in err, err
    assert "Traceback" not in err, err


def test_a_hugging_face_time_out_says_where_without_what_the_address_hides() -> None:
    """A request that timed out is raised as ``TimeoutError`` naming the address it asked, never
    a user, password or query that a mirror's address (``HF_ENDPOINT``) may hold; the Lab's own
    errors and failures that are not time-outs pass unchanged."""
    httpx = pytest.importorskip("httpx")
    from real2simlab._hub import bounded

    asked = httpx.Request("GET", "https://reader@example.com/api/datasets/a/b?t=not-a-key")
    with pytest.raises(TimeoutError) as raised, bounded():
        raise httpx.ReadTimeout("timed out", request=asked)
    said = str(raised.value)
    assert said == (
        "Hugging Face did not answer in time (https://example.com/api/datasets/a/b): timed out"
    )
    with pytest.raises(SetupError, match="the Lab's own"), bounded():
        raise SetupError("the Lab's own") from httpx.ReadTimeout("timed out", request=asked)
    with pytest.raises(ValueError, match="not a time-out"), bounded():
        raise ValueError("not a time-out")


@pytest.mark.parametrize("limit", [64, 200])
def test_a_storage_failure_while_claiming_a_run_directory_leaves_it_usable(
    bouncing_task: Path, tmp_path: Path, limit: int
) -> None:
    """The first record of a run (its status) fails to write (a full disk; here a file-size
    limit): the error says to run the same command again once there is space, and doing so
    works; the directory is not left looking like someone else's."""
    import resource

    here = str(Path(__file__).resolve().parents[1])
    out = tmp_path / "run"
    argv = [sys.executable, "-m", "real2simlab.cli.main", "eval", "--tasks", str(bouncing_task),
            "--method", "oracle", "--backend", "local", "--out", str(out)]  # fmt: skip
    env = {**os.environ, "PYTHONPATH": here, "PYTHONDONTWRITEBYTECODE": "1"}  # see conftest

    def limited() -> None:
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))

    first = subprocess.run(argv, env=env, capture_output=True, text=True, check=False,
                           preexec_fn=limited, stdin=subprocess.DEVNULL, timeout=300)  # fmt: skip
    assert first.returncode == 1 and "run the same command again" in first.stderr, first.stderr
    again = subprocess.run(argv, env=env, capture_output=True, text=True, check=False,
                           stdin=subprocess.DEVNULL, timeout=300)  # fmt: skip
    assert again.returncode == 0, again.stderr
    assert json.loads((out / "state" / "status.json").read_text())["status"] == "complete"


def test_a_run_stopped_while_writing_its_first_status_is_still_the_labs(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """``kill -9`` between the lock and the first status: the lock and a half-written status
    are the Lab's, so the same command resumes rather than refusing the directory."""
    from real2simlab.cli.main import main

    out = tmp_path / "run"
    (out / "state").mkdir(parents=True)
    (out / "state" / "lock").write_text("pid 1, since 2026-01-01T00:00:00+00:00\n")
    (out / "state" / "status.json.tmp").write_text('{"status": "run')
    assert main(["eval", "--tasks", str(bouncing_task), "--method", "oracle", "--backend",
                 "local", "--out", str(out)]) == 0  # fmt: skip


# --- 15 Video2World's shared data and Isaac image -------------------------------------------------


class _InterruptedHub:
    """``huggingface_hub`` whose first snapshot download dies after one file (a dropped
    network), unless ``fail_first`` is false."""

    def __init__(self, files: dict[str, bytes], *, fail_first: bool = True) -> None:
        self.files, self.calls, self.fail_first = files, 0, fail_first

    def snapshot_download(self, *args: Any, local_dir: Path, **kwargs: Any) -> None:
        self.calls += 1
        for i, (name, data) in enumerate(self.files.items()):
            if self.fail_first and self.calls == 1 and i == 1:
                raise OSError("connection reset by peer")
            (Path(local_dir) / name).parent.mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / name).write_bytes(data)


def test_an_interrupted_download_of_shared_data_is_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from real2simlab.benchmarks import video2world as v2w

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    hub = _InterruptedHub({"assets/maniskill/a.urdf": b"a", "assets/hand/b.urdf": b"b",
                           "assets/datasets/c.obj": b"c"})  # fmt: skip
    monkeypatch.setattr(v2w, "_hub", lambda: hub)
    with pytest.raises(OSError, match="reset"):
        v2w.shared_data(v2w.HAND)
    # Half a download is never taken for a complete one: the next score downloads again.
    found = v2w.shared_data(v2w.HAND)
    assert hub.calls == 2 and sorted(found) == [
        "assets/datasets",
        "assets/hand",
        "assets/maniskill",
    ]
    assert v2w.shared_data(v2w.HAND) == found and hub.calls == 2  # then the cache is used


def test_incomplete_robodojo_assets_are_refused_then_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from real2simlab.benchmarks import video2world as v2w

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    files = {"Assets/Robots/x5/x5_tmp.yml": b"usd: ${ASSETS_PATH}/Assets/Robots/x5/x5.usd\n"}
    monkeypatch.setattr(v2w, "_hub", lambda: _InterruptedHub(dict(files), fail_first=False))
    with pytest.raises(SetupError, match="lack Object, Material, Eval_Layout") as error:
        v2w.robodojo_assets()
    assert "score again" in (error.value.fix or "")
    files.update({f"Assets/{d}/x": b"x" for d in ("Object", "Material", "Eval_Layout")})
    hub = _InterruptedHub(files)
    monkeypatch.setattr(v2w, "_hub", lambda: hub)
    with pytest.raises(OSError):
        v2w.robodojo_assets()
    assets = v2w.robodojo_assets()
    assert hub.calls == 2 and v2w.robodojo_assets() == assets and hub.calls == 2
    # RoboDojo's own configuration step: the templates name the assets where the image has them.
    assert (assets / "Robots" / "x5" / "x5.yml").read_text() == (
        "usd: /opt/RoboDojo/Assets/Robots/x5/x5.usd\n"
    )


def test_a_task_that_lost_a_public_file_says_to_rebuild(tmp_path: Path) -> None:
    from real2simlab.benchmarks import video2world as v2w

    release, visible = tmp_path / "release", tmp_path / "visible"
    release.mkdir()
    visible.mkdir()
    (release / v2w.PUBLIC_FILES).write_text(json.dumps({"samples/s/robot/a.stl": "tools/a.stl"}))
    with pytest.raises(SetupError, match=r"visible/tools/a\.stl is missing") as error:
        v2w.restore_public_files(release, visible)
    assert "r2s bench build video2world" in (error.value.fix or "")


def test_an_isaac_task_without_its_image_says_how_to_build_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scoring an Isaac Sim instance without ``r2s-video2world-isaac`` is a setup error that
    names the license to accept, never a score (and never the other image's problem)."""
    from real2simlab.benchmarks.video2world import scorer

    asked: list[str] = []

    def inspect(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        asked.append(argv[-1])
        missing = "isaac" in argv[-1]
        return subprocess.CompletedProcess(argv, 1 if missing else 0, "sha256:abc\n", "")

    monkeypatch.setattr(scorer, "run_captured", inspect)
    monkeypatch.setattr(scorer, "docker_problem", lambda docker: None)  # Docker answers here
    (problem,) = scorer.Video2WorldScorer().problems("r2s-video2world-isaac")
    assert "R2S_ACCEPT_NVIDIA_EULA=1 r2s env build r2s-video2world-isaac" in problem
    assert scorer.Video2WorldScorer().problems() == []
    # Without the Isaac image, scores of the other evaluators keep their cache identity.
    assert scorer.Video2WorldScorer().cache_identity() == "sha256:abc"


def test_an_image_whose_license_is_not_accepted_is_never_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from real2simlab import envs

    monkeypatch.delenv("R2S_ACCEPT_NVIDIA_EULA", raising=False)
    isaac = envs.get_environment("r2s-video2world-isaac")
    assert not isaac.publishable and not envs.terms_accepted(isaac)
    with pytest.raises(SetupError, match="license must be accepted") as error:
        envs.build("r2s-video2world-isaac", docker="/nonexistent/docker")
    assert "R2S_ACCEPT_NVIDIA_EULA=1 r2s env build" in (error.value.fix or "")
    monkeypatch.setenv("R2S_ACCEPT_NVIDIA_EULA", "yes")  # only "1" accepts
    assert not envs.terms_accepted(isaac)
    monkeypatch.setenv("R2S_ACCEPT_NVIDIA_EULA", "1")
    assert envs.accepted_terms(isaac) == {"ACCEPT_EULA": "Y"}
    assert envs.accepted_terms(envs.get_environment("r2s-python")) == {}


# --- What a command stopped in its own directory left there is still its own -------------------


def _in_a_new_process(
    argv: list[str], max_file_size: int | None = None
) -> subprocess.CompletedProcess[str]:
    """``python <argv>`` with this copy of the package; under a file-size limit if one is given
    (as ``ulimit -f``: a write past it fails, as it does on a full disk)."""
    import resource

    def limited() -> None:
        if max_file_size is not None:
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_size, max_file_size))

    here = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": here, "PYTHONDONTWRITEBYTECODE": "1"}  # see conftest
    if max_file_size is not None:
        # CI measures the processes tests start (coverage's patch = subprocess): under the
        # limit coverage cannot save its data file and prints a traceback at exit, so the child
        # runs without it.
        env = {name: value for name, value in env.items() if not name.startswith("COVERAGE_")}
    return subprocess.run([sys.executable, *argv], env=env,
                          capture_output=True, text=True, check=False, preexec_fn=limited,
                          stdin=subprocess.DEVNULL, timeout=300)  # fmt: skip


def _reward_or_check(command: str, task: Path, out: Path) -> list[str]:
    """``r2s train reward --work out`` or ``r2s twin check --out out`` of the task's program
    (a program directory: both save it as a twin in ``out``)."""
    program = str(task / "hidden" / "generator" / "program")
    if command == "train reward":
        return ["train", "reward", "--task", str(task), "--solution", program, "--backend",
                "local", "--work", str(out)]  # fmt: skip
    return ["twin", "check", program, "--task", str(task), "--backend", "local", "--out", str(out)]


KILLED_WHILE_THE_TWIN_IS_STAGED = """
import os, signal, sys
from real2simlab.twin import io
def killed(*args, **kwargs):  # save_twin loads the staged package before it renames it
    os.kill(os.getpid(), signal.SIGKILL)
io.load_twin = killed
from real2simlab.cli.main import main
sys.exit(main(sys.argv[1:]))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("command", ["train reward", "twin check"])
def test_a_command_killed_while_it_saves_the_twin_runs_again_in_its_directory(
    command: str, bouncing_task: Path, tmp_path: Path
) -> None:
    """``r2s train reward --work`` and ``r2s twin check --out`` save the program as a twin in
    their directory, staged as ``twin.partial`` and then renamed. Killed in between (kill -9,
    an OOM kill: no cleanup runs), the same command clears what it left there and runs, rather
    than refusing the directory for holding a file it did not write. The name alone, without
    the command's mark, makes no directory the command's."""
    from real2simlab.cli.main import main

    out = tmp_path / "out"
    argv = _reward_or_check(command, bouncing_task, out)
    killed = _in_a_new_process(["-c", KILLED_WHILE_THE_TWIN_IS_STAGED, *argv])
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    assert (out / "twin.partial").is_dir() and not (out / "twin").exists()
    assert main(argv) == 0
    assert (out / "twin" / "twin.json").is_file() and not (out / "twin.partial").exists()
    mine = tmp_path / "mine"
    (mine / "twin.partial").mkdir(parents=True)
    (mine / "twin.partial" / "notes.txt").write_text("keep me")
    assert main(_reward_or_check(command, bouncing_task, mine)) == 1
    assert (mine / "twin.partial" / "notes.txt").read_text() == "keep me"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resource limits")
@pytest.mark.parametrize("stop", ["full disk", "killed"])
@pytest.mark.parametrize("command", ["train reward", "twin check"])
def test_a_command_whose_mark_was_not_written_runs_again_in_its_directory(
    command: str, stop: str, bouncing_task: Path, tmp_path: Path
) -> None:
    """The first file ``r2s train reward --work`` and ``r2s twin check --out`` write into their
    directory is the mark that makes it theirs (``.r2s-reward``, ``.r2s-check``). When that
    write fails (a full disk; here a file-size limit of 0), the error says to run the same
    command again once there is space; killed between creating the mark and writing it, the
    command leaves it empty. Either way the same command runs, rather than refusing the
    directory as someone else's."""
    from real2simlab.cli.main import main

    out = tmp_path / "out"
    argv = _reward_or_check(command, bouncing_task, out)
    if stop == "killed":
        out.mkdir()
        (out / {"train reward": ".r2s-reward", "twin check": ".r2s-check"}[command]).touch()
    else:
        first = _in_a_new_process(["-m", "real2simlab.cli.main", *argv], max_file_size=0)
        assert first.returncode == 1, first.stderr
        assert "run the same command again" in first.stderr, first.stderr
        assert "Traceback" not in first.stderr, first.stderr
    assert main(argv) == 0


def test_a_mark_counts_whole_or_cut_short_and_nothing_else(tmp_path: Path) -> None:
    """A command's mark is its own whole, or cut short (kill -9 or a power cut while it was
    written, empty included); a file of that name with other content, or more, is not."""
    from real2simlab._fs import has_mark

    mark, path = "r2s twin check\n", tmp_path / ".r2s-check"
    for content, counts in [(mark, True), (mark[:5], True), ("", True), ("my notes\n", False),
                            (mark + "mine\n", False)]:  # fmt: skip
        path.write_text(content)
        assert has_mark(path, mark) is counts, content
    path.unlink()
    assert not has_mark(path, mark)
    path.mkdir()
    assert not has_mark(path, mark)


def test_a_mark_is_never_read_or_written_through_a_link(tmp_path: Path) -> None:
    """A mark's name holding a symbolic link (to a file elsewhere, or to nothing yet): the Lab
    neither takes what it points to for its mark nor writes its mark there, and says to
    remove the link; a regular mark is written and read as before."""
    from real2simlab._fs import has_mark, write_mark

    mark, out = "r2s twin check\n", tmp_path / "out"
    out.mkdir()
    bait = tmp_path / "elsewhere.txt"
    bait.write_text("")  # empty: what a mark cut short at its start looks like
    for target in (bait, tmp_path / "nothing-yet.txt"):
        link = out / ".r2s-check"
        link.unlink(missing_ok=True)
        link.symlink_to(target)
        assert not has_mark(link, mark)
        with pytest.raises(SetupError, match="link") as refused:
            write_mark(link, mark)
        assert "remove it" in (refused.value.fix or "") and link.is_symlink()
    assert bait.read_text() == "" and not (tmp_path / "nothing-yet.txt").exists()
    write_mark(out / ".r2s-task-build", mark)
    assert has_mark(out / ".r2s-task-build", mark)


STARTED_BUILD = """
import sys
from pathlib import Path
from real2simlab.task import TaskBuilder
from real2simlab.task.io import read_task_config
task, root, overwrite = sys.argv[1:]
TaskBuilder(Path(root), read_task_config(Path(task) / "task.toml"), overwrite=overwrite == "yes")
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resource limits")
@pytest.mark.parametrize("case", ["new directory", "a stopped build's directory", "killed"])
def test_a_task_build_whose_mark_was_not_written_is_replaced_by_overwrite(
    case: str, bouncing_task: Path, tmp_path: Path
) -> None:
    """A task build writes its mark (``.r2s-task-build``) before anything else. When that write
    fails (a full disk; here a file-size limit of 0), in a new directory or over the one a
    stopped build left (with ``--overwrite``), or the build is killed while it writes it (the
    mark is left empty), the retry the Lab asks for (the same command, which says to pass
    ``--overwrite``) builds the task: the directory is not refused as one no build wrote."""
    from real2simlab.benchmarks.synthetic import build_suite as build
    from real2simlab.errors import OutputExistsError
    from real2simlab.task import TaskBuilder, load_task

    out = tmp_path / "tasks"
    root = out / "bouncing_ball" / "calibrated"
    if case == "killed":
        root.mkdir(parents=True)
        (root / ".r2s-task-build").touch()
    else:
        overwrite = "no"
        if case == "a stopped build's directory":  # stopped before task.toml (Ctrl-C, kill -9)
            TaskBuilder(root, read_task_config(bouncing_task / "task.toml"))
            overwrite = "yes"
        started = _in_a_new_process(["-c", STARTED_BUILD, str(bouncing_task), str(root),
                                     overwrite], max_file_size=0)  # fmt: skip
        assert started.returncode == 0 or "File too large" in started.stderr, started.stderr
    with pytest.raises(OutputExistsError, match="--overwrite"):
        build(out, select=["bouncing_ball"], tiers=["calibrated"])
    assert build(out, select=["bouncing_ball"], tiers=["calibrated"], overwrite=True) == [root]
    assert load_task(root).id == "synthetic/bouncing_ball/calibrated"
    assert not (root / ".r2s-task-build").exists()


# --- What a program prints and what it leaves running stay within bounds --------------------


def _within(call: Callable[[], object], seconds: float = 120.0) -> tuple[bool, Any, Any]:
    """Run ``call`` in a worker thread for at most ``seconds``: whether it finished, what it
    returned and what it raised (a command that waits for ever must not hang the tests)."""
    import threading

    outcome: list[Any] = [None, None]

    def work() -> None:
        try:
            outcome[0] = call()
        except BaseException as error:  # the test inspects it
            outcome[1] = error

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(seconds)
    return not worker.is_alive(), outcome[0], outcome[1]


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full (Linux)")
def test_a_log_that_stops_being_writable_ends_the_command_with_an_error(
    bouncing_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The disk holding ``r2s.log`` fills up while builds run (or ``ulimit -f`` is reached
    there): the command ends with one error line and the fix. It printed a traceback from a
    build's callback and then waited for ever (A9R1-01)."""
    import pathlib
    import threading

    from real2simlab.cli.main import main

    full = [False]

    class Filling(pathlib.PosixPath):
        """``r2s.log``, on a disk that is full once a build has started writing to it."""

        def open(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            if threading.current_thread().name.startswith("r2s"):  # a build's worker
                full[0] = True
            if full[0]:
                return open("/dev/full", *args, **kwargs)
            return super().open(*args, **kwargs)

    monkeypatch.setattr(
        RunLayout, "log_file", property(lambda layout: Filling(layout.root / "r2s.log"))
    )
    argv = ["eval", "--tasks", str(bouncing_task), "--method", "oracle", "--method", "static",
            "--backend", "local", "--out", str(tmp_path / "run")]  # fmt: skip
    finished, code, raised = _within(lambda: main(argv))
    assert finished, "the command still waits after its log could not be written"
    err = capsys.readouterr().err
    assert raised is None and code == 1, (raised, err)
    assert "error: No space left on device" in err and "fix: free space" in err, err
    assert f"({tmp_path / 'run' / 'r2s.log'})" in err, err  # which file
    assert "Traceback" not in err and "exception calling callback" not in caplog.text, err


def test_an_echo_of_the_log_that_fails_neither_stops_nor_hangs_the_experiment(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """Where the experiment echoes its lines stops taking them (``r2s demo | head``, a closed
    terminal): ``r2s.log`` still gets every line and the experiment finishes. It waited for
    ever once the echo failed in a build's thread (A9R1-01)."""
    import errno

    from real2simlab.orchestrate import run_experiment

    def echo(line: str) -> None:
        if " experiment " not in line:  # the first line, before any build, gets through
            raise BrokenPipeError(errno.EPIPE, "Broken pipe")

    config = _config([bouncing_task], [{"name": "oracle"}])
    finished, results, raised = _within(
        lambda: run_experiment(config, tmp_path / "run", log=echo, report=False)
    )
    assert finished, "the experiment still waits after its echo failed"
    assert raised is None, raised
    assert len(results.rows) == 1
    assert "done:" in (tmp_path / "run" / "r2s.log").read_text()


def test_what_a_program_prints_is_kept_within_a_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process's output is kept up to a limit: its beginning and its end, with a line saying
    how much was left out between them, also when it runs into its time limit. A twin printing
    in a loop wrote gigabytes into its run directory, enough to fill the disk (A3-05)."""
    from real2simlab.sandbox import process

    limit = 256 * 1024
    monkeypatch.setattr(process, "LOG_LIMIT_BYTES", limit, raising=False)
    monkeypatch.setattr(process, "LOG_TAIL_BYTES", 64 * 1024, raising=False)
    prints = (
        "import sys\nfor i in range(4096):\n    sys.stdout.write(f'{i:05d}' + 'x' * 1018 + "
        "'\\n')\nprint('END')\n"
    )  # 4 MiB
    log = tmp_path / "log.txt"
    code, timed_out, _ = process.run_logged([sys.executable, "-c", prints], log, 60)
    kept = log.read_bytes()
    assert code == 0 and not timed_out
    assert len(kept) <= limit + 4096, len(kept)
    assert kept.startswith(b"00000xxx") and kept.endswith(b"END\n"), kept[-100:]
    assert b"bytes of output left out" in kept
    endless = tmp_path / "endless.txt"
    loop = "while True:\n    print('y' * 1000)\n"
    code, timed_out, _ = process.run_logged([sys.executable, "-c", loop], endless, 2)
    kept = endless.read_bytes()
    assert code is None and timed_out
    assert len(kept) <= limit + 4096 and kept.endswith(b"[r2s] killed after 2 s\n"), len(kept)


def test_what_a_tool_prints_is_kept_within_a_limit_as_it_comes() -> None:
    """``run_captured(keep=...)``, for what the agents whose harness runs on the host (Harbor,
    mini-swe-agent) print: each stream keeps its first and last whole lines within the limit,
    read as they come, also when the tool runs into its time limit; a line longer than what
    is kept is left out whole (a key cut in two would escape redaction). Their whole output
    was read into memory and written into the run directory (A3-05)."""
    from real2simlab.sandbox.process import run_captured

    most = 64 * 1024
    prints = (
        "import sys\nfor i in range(100):\n    print(f'{i:06d} ' + 'o' * 93)\n"
        "print('x' * 300_000)\nfor i in range(100, 100_000):\n    print(f'{i:06d} ' + 'o' * 93)\n"
        "    print(f'{i:06d} ' + 'e' * 93, file=sys.stderr)\nprint('END')\n"
    )
    result = run_captured([sys.executable, "-c", prints], keep=most, timeout_s=120)
    assert result.returncode == 0, result.stderr[-500:]
    for text, letter, first in ((result.stdout, "o", 0), (result.stderr, "e", 100)):
        assert len(text.encode()) <= most and "xxxx" not in text, len(text.encode())
        lines = text.splitlines()
        assert lines[0] == f"{first:06d} " + letter * 93, lines[0]
        assert lines[-1] == ("END" if letter == "o" else "099999 " + "e" * 93), lines[-1]
        assert all(len(line) == 100 for line in lines if line[:6].isdigit()), "a line was cut"
        assert sum("bytes left out here" in line for line in lines) == 1
    endless = "while True:\n    print('y' * 99)\n"
    with pytest.raises(subprocess.TimeoutExpired) as stopped:
        run_captured([sys.executable, "-c", endless], keep=most, timeout_s=2)
    kept = str(stopped.value.stdout)
    assert kept.startswith("y" * 99 + "\n") and len(kept.encode()) <= most, len(kept.encode())


def _alive(pid: int) -> bool:
    """Whether process ``pid`` still runs (a zombie has ended)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError:
        return False
    return stat[stat.rfind(b")") + 2 :][:1] not in (b"Z", b"X")


LINUX_ONLY = pytest.mark.skipif(
    sys.platform != "linux", reason="the local backend follows a program's processes on Linux"
)


def _late_writer(started: Path, late: Path, seconds: float) -> str:
    """Python code that says its pid in ``started``, then writes ``late`` after ``seconds``."""
    return (
        f"import os, time\nopen({str(started)!r}, 'w').write(str(os.getpid()))\n"
        f"time.sleep({seconds})\nopen({str(late)!r}, 'w').write('still running')\n"
    )


@LINUX_ONLY
@pytest.mark.slow
@pytest.mark.parametrize("ending", ["time limit", "exit"])
def test_what_a_local_program_starts_does_not_outlive_its_run(tmp_path: Path, ending: str) -> None:
    """A program on the local backend starts a process in a session of its own (out of reach of
    the run's process group), then runs into its time limit or exits: that process ends with
    the run. It ran on, writing into the run directory after the run was recorded
    (A12-robustness-20)."""
    import time

    from real2simlab.sandbox.base import Limits

    started, late = tmp_path / "started", tmp_path / "late"
    child = _late_writer(started, late, 3)
    program = (
        "import os, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}], start_new_session=True)\n"
        f"while not os.path.exists({str(started)!r}):\n    time.sleep(0.02)\n"
        + ("time.sleep(60)\n" if ending == "time limit" else "")
    )
    spec = SandboxSpec(
        image="local", argv=("python", "-c", program), workdir=str(tmp_path), tmpfs=(),
        limits=Limits(timeout_s=1.5 if ending == "time limit" else 60),
    )  # fmt: skip
    result = LocalBackend(quiet=True).run(spec, tmp_path / "log.txt")
    assert result.timed_out is (ending == "time limit"), result
    time.sleep(4)
    assert not late.exists(), "a process the program started ran on after its run"
    assert not _alive(int(started.read_text()))


@LINUX_ONLY
@pytest.mark.slow
def test_a_local_program_stops_when_the_lab_is_killed(tmp_path: Path) -> None:
    """``kill -9`` of the Lab while a program runs on the local backend: the program stops
    instead of running on with no time limit, writing into the run directory
    (A12-robustness-20)."""
    import time

    started, late = tmp_path / "started", tmp_path / "late"
    lab = (
        "import sys\nfrom pathlib import Path\n"
        "from real2simlab.sandbox.base import Limits, SandboxSpec\n"
        "from real2simlab.sandbox.local import LocalBackend\n"
        f"spec = SandboxSpec(image='local', argv=('python', '-c', {_late_writer(started, late, 3)!r}),"
        f" workdir={str(tmp_path)!r}, tmpfs=(), limits=Limits(timeout_s=60))\n"
        f"LocalBackend(quiet=True).run(spec, Path({str(tmp_path / 'log.txt')!r}))\n"
    )
    here = str(Path(__file__).resolve().parents[1])
    process = subprocess.Popen(
        [sys.executable, "-c", lab], env={**os.environ, "PYTHONPATH": here},
        stdin=subprocess.DEVNULL,
    )  # fmt: skip
    deadline = time.monotonic() + 60
    while not started.exists():
        assert process.poll() is None and time.monotonic() < deadline
        time.sleep(0.05)
    process.kill()
    process.wait()
    time.sleep(4)
    assert not late.exists(), "the program ran on after the Lab was killed"
    assert not _alive(int(started.read_text()))


IRIS_BUILD = (
    "import os, sys, time\n"
    "from pathlib import Path\n"
    "from real2simlab.benchmarks import iris\n"
    "def stalled(setting, take):  # a 4K recording that takes long to download and decode\n"
    "    (Path(sys.argv[1]) / f'{setting}_{take}.{os.getpid()}').touch()\n"
    "    time.sleep(300)\n"
    "iris.observations = stalled\n"
    "if sys.argv[3] == 'watching its parent':  # a system that cannot end a worker with it\n"
    "    from real2simlab.sandbox import pool\n"
    "    pool._ends_with_parent = lambda parent: False\n"
    "from real2simlab.cli.main import main\n"
    "sys.exit(main(['bench', 'build', 'iris', '--out', sys.argv[2]]))\n"
)
"""``r2s bench build iris`` whose takes never finish (arguments: a directory where each worker
says it began a take, the output directory, how its workers end with it)."""


def _iris_build(tmp_path: Path, how: str = "told by the system") -> subprocess.Popen[str]:
    """``IRIS_BUILD`` as typed in a terminal, in a process group of its own, once two worker
    processes are each busy with a take."""
    import multiprocessing
    import time

    if multiprocessing.get_start_method() != "fork":
        pytest.skip("workers that are not forked do not see the stalled takes")
    started = tmp_path / "started"
    started.mkdir()
    here = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": here, "R2S_CACHE_DIR": str(tmp_path / "cache"),
           "R2S_WORKERS": "2"}  # fmt: skip
    process = subprocess.Popen(
        [sys.executable, "-c", IRIS_BUILD, str(started), str(tmp_path / "tasks"), how],
        env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True, preexec_fn=_as_typed_in_a_terminal,
    )  # fmt: skip
    deadline = time.monotonic() + 120
    while len(list(started.iterdir())) < 2:
        if process.poll() is not None or time.monotonic() > deadline:
            _stop_group(process)
            pytest.fail("the build's workers did not begin their takes")
        time.sleep(0.05)
    return process


def _group(leader: int) -> list[int]:
    """The processes of the process group ``leader`` leads that have not ended."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_bytes()
        except OSError:  # it ended meanwhile
            continue
        fields = stat[stat.rfind(b")") + 2 :].split()  # state, parent, group, ...
        if fields[0] not in (b"Z", b"X") and int(fields[2]) == leader:
            found.append(int(entry.name))
    return found


def _left_in(leader: int, seconds: float = 10.0) -> list[int]:
    """What is still running in the process group ``leader`` led, ``seconds`` from now (or as
    soon as nothing is)."""
    import time

    deadline = time.monotonic() + seconds
    while (left := _group(leader)) and time.monotonic() < deadline:
        time.sleep(0.1)
    return left


@LINUX_ONLY
@pytest.mark.slow
@pytest.mark.timeout(240)
@pytest.mark.parametrize("stop", ["Ctrl-C", "Ctrl-C twice", "kill", "a closed terminal"])
def test_an_interrupted_iris_build_stops_its_workers_at_once(tmp_path: Path, stop: str) -> None:
    """IRIS downloads and decodes its recordings in worker processes. Ctrl-C (sent to the whole
    process group, as a terminal does), Ctrl-C twice, ``kill`` or a closed terminal stops the
    build at once: status 130, one line, no traceback, no worker left running. The build used
    to wait for the takes its workers had begun (a whole 4K download), and a second Ctrl-C
    during that wait left it waiting for ever, with every worker's traceback (A3-12)."""
    import contextlib
    import time

    process = _iris_build(tmp_path)
    try:
        started = time.monotonic()
        if stop == "kill":
            os.kill(process.pid, signal.SIGTERM)
        else:
            os.killpg(process.pid, signal.SIGHUP if stop == "a closed terminal" else signal.SIGINT)
        if stop == "Ctrl-C twice":
            time.sleep(0.3)
            with contextlib.suppress(ProcessLookupError):  # it had already stopped
                os.killpg(process.pid, signal.SIGINT)
        try:
            _, err = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            pytest.fail(f"the build was still running 30 s after {stop}")
        assert time.monotonic() - started < 15, f"{stop} waited for the workers' takes"
        assert process.returncode == 130, err
        assert "interrupted" in err and "Traceback" not in err, err
        assert not _left_in(process.pid), f"a worker outlived the build stopped by {stop}"
    finally:
        _stop_group(process)


@LINUX_ONLY
@pytest.mark.slow
@pytest.mark.timeout(240)
@pytest.mark.parametrize("how", ["told by the system", "watching its parent"])
def test_iris_workers_end_with_a_killed_build(tmp_path: Path, how: str) -> None:
    """``kill -9`` of ``r2s bench build iris`` while its workers download and decode: they end
    with it (on Linux the system tells them; elsewhere each watches its parent), instead of
    waiting for ever for takes that never come (A3-12)."""
    process = _iris_build(tmp_path, how)
    try:
        assert len(_group(process.pid)) >= 3  # the build and its two workers
        process.kill()
        process.wait()  # not its output: workers left running would hold it open
        assert not _left_in(process.pid), "workers outlived the killed build"
    finally:
        _stop_group(process)


def test_every_process_pool_stops_with_the_command() -> None:
    """Worker processes are started only through ``sandbox.pool.map_in_processes``, whose
    workers leave Ctrl-C to the command, are killed when it stops and end with it: a pool of
    its own would wait for its workers' tasks after Ctrl-C, for ever after a second one, and
    leave its workers running after ``kill -9`` (A3-12)."""
    import ast

    import real2simlab

    package = Path(real2simlab.__file__).parent
    pools = []
    for path in sorted(package.rglob("*.py")):
        where = path.relative_to(package).as_posix()
        if where == "sandbox/pool.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""] + [alias.name for alias in node.names]
            elif isinstance(node, ast.Name | ast.Attribute):
                modules = [getattr(node, "id", None) or getattr(node, "attr", "")]
            else:
                continue
            if any(
                name.split(".")[0] == "multiprocessing" or name == "ProcessPoolExecutor"
                for name in modules
            ):
                pools.append(f"{where}:{node.lineno}")
    assert not pools, f"start worker processes with sandbox.pool.map_in_processes: {pools}"


def _ignored_by_a_program_a_worker_starts(_item: int) -> str:
    """The signals a program started in a worker process ignores (``SigIgn`` in its
    ``/proc/<pid>/status``, a hexadecimal mask): what IRIS's ffprobe and ffmpeg get there."""
    status = subprocess.run(
        ["cat", "/proc/self/status"], capture_output=True, text=True, check=True, timeout=60
    ).stdout
    return next(line.split()[1] for line in status.splitlines() if line.startswith("SigIgn:"))


@LINUX_ONLY
def test_the_programs_a_worker_starts_still_stop_on_ctrl_c() -> None:
    """A worker of ``map_in_processes`` leaves Ctrl-C and a closed terminal to the command, but
    a program it starts (IRIS's ffprobe, ffmpeg) still stops on them: ``SIG_IGN`` lasts across
    ``exec``, so with its worker ignoring them that way, it ignored them too and, its worker
    killed, ran on as an orphan that no Ctrl-C stopped."""
    from real2simlab.sandbox.pool import map_in_processes

    (started,) = map_in_processes(_ignored_by_a_program_a_worker_starts, [0], workers=1)
    status = Path("/proc/self/status").read_text()
    own = next(line.split()[1] for line in status.splitlines() if line.startswith("SigIgn:"))
    signals = (1 << (signal.SIGINT - 1)) | (1 << (signal.SIGHUP - 1))
    # what this test's own process ignores (SIGHUP under nohup), the program may ignore too
    assert int(started, 16) & signals == int(own, 16) & signals, (started, own)


@pytest.fixture
def hangup_ignored() -> Iterator[None]:
    """This process ignores SIGHUP, as a command started with ``nohup`` does; restored after."""
    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGHUP, previous)


@LINUX_ONLY
@pytest.mark.usefixtures("hangup_ignored")
def test_a_signal_the_command_ignores_stays_ignored_in_what_its_workers_start() -> None:
    """``nohup r2s bench build iris ... &``: the command ignores SIGHUP, so closing the terminal
    leaves it running, and the programs its workers start (ffprobe, ffmpeg) must ignore it too.
    A worker that put its own handler on SIGHUP whatever the command did reset it to the
    default in them, so the terminal's hangup killed them under the running build."""
    from real2simlab.sandbox.pool import map_in_processes

    (started,) = map_in_processes(_ignored_by_a_program_a_worker_starts, [0], workers=1)
    status = Path("/proc/self/status").read_text()
    own = next(line.split()[1] for line in status.splitlines() if line.startswith("SigIgn:"))
    hangup, interrupt = 1 << (signal.SIGHUP - 1), 1 << (signal.SIGINT - 1)
    assert int(started, 16) & hangup, f"the program does not ignore SIGHUP: SigIgn {started}"
    assert int(started, 16) & interrupt == int(own, 16) & interrupt, (started, own)


@LINUX_ONLY
@pytest.mark.slow
@pytest.mark.parametrize("stop", ["interrupted", "time limit"])
def test_an_agent_cli_stops_with_its_sandbox_on_the_local_backend(
    tmp_path: Path, stop: str
) -> None:
    """Ctrl-C (``cancel_all``) or the time limit stops an agent's sandbox on the local backend:
    the agent's CLI, which the bootstrap starts in a session of its own, stops with it. It ran
    on after ``r2s`` had exited, writing into the run directory (A3-02)."""
    import time

    from real2simlab.agents.base import BOOTSTRAP
    from real2simlab.sandbox.base import Limits

    started, late = tmp_path / "started", tmp_path / "late"
    plan = {
        "argv": [sys.executable, "-c", _late_writer(started, late, 4)],
        "stdout": str(tmp_path / "trajectory.jsonl"),
        "stderr": str(tmp_path / "stderr.log"),
        "summary": str(tmp_path / "bootstrap.json"),
        "cwd": str(tmp_path),
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    spec = SandboxSpec(
        image="local", argv=("python", str(BOOTSTRAP), str(tmp_path / "plan.json")),
        workdir=str(tmp_path), tmpfs=(),
        limits=Limits(timeout_s=60 if stop == "interrupted" else 2),
    )  # fmt: skip

    def run() -> object:
        return LocalBackend(quiet=True).run(spec, tmp_path / "sandbox.log")

    if stop == "interrupted":
        assert isinstance(_interrupted(run, started), KeyboardInterrupt)
    else:
        assert getattr(run(), "timed_out", None) is True
    time.sleep(4.5)
    assert not late.exists(), "the agent's CLI ran on after its sandbox was stopped"
    assert not _alive(int(started.read_text()))


@LINUX_ONLY
def test_the_local_backend_reports_how_its_program_ended(tmp_path: Path) -> None:
    """Every local program runs under a supervisor that relays its end: its exit status, or
    the signal that killed it; a command that does not exist fails as before."""
    from real2simlab.sandbox.base import Limits

    local = LocalBackend(quiet=True)
    for index, (source, code) in enumerate(
        [("pass", 0), ("raise SystemExit(7)", 7), ("import os\nos.kill(os.getpid(), 15)", -15)]
    ):
        spec = SandboxSpec(
            image="local", argv=("python", "-c", source), workdir=str(tmp_path), tmpfs=(),
            limits=Limits(timeout_s=60),
        )  # fmt: skip
        assert local.run(spec, tmp_path / f"{index}.log").exit_code == code, source
    missing = SandboxSpec(
        image="local", argv=("r2s-no-such-command",), workdir=str(tmp_path), tmpfs=()
    )
    with pytest.raises(FileNotFoundError):
        local.run(missing, tmp_path / "missing.log")


@LINUX_ONLY
@pytest.mark.parametrize("code", [0, 3])
def test_the_supervisor_relays_the_programs_end_where_it_cannot_list_processes(
    tmp_path: Path, code: int
) -> None:
    """Where this machine's processes cannot be listed (no ``/proc``), the supervisor says so
    and still ends as its program did. It ended with a traceback and the status 1, so a
    program that succeeded was recorded as failed."""
    from real2simlab.sandbox import local

    supervisor = (
        "import importlib.util, os, sys\n"
        f"spec = importlib.util.spec_from_file_location('reaper', {str(local.REAPER)!r})\n"
        "reaper = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(reaper)\n"
        "listdir = os.listdir\n"
        "def unlisted(path='.'):\n"
        "    if str(path) == '/proc':\n"
        "        raise PermissionError(13, 'Permission denied', '/proc')\n"
        "    return listdir(path)\n"
        "reaper.os.listdir = unlisted\n"
        "reaper.main(['reaper.py', str(os.getppid()), '--', *sys.argv[1:]])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", supervisor, sys.executable, "-c", f"raise SystemExit({code})"],
        capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert result.returncode == code, result.stderr
    assert "Traceback" not in result.stderr and "cannot list the processes" in result.stderr


def test_what_a_process_prints_reaches_the_disk_only_through_a_bounded_log() -> None:
    """The output of a process the Lab starts goes to a pipe that it reads (``run_logged``
    keeps a bounded log of it, the agent's bootstrap bounded files), or nowhere: a file given
    to a process as its stdout or stderr has no size limit, and a program printing in a loop
    filled the disk (A3-05). The exceptions say why they are bounded. What it reads with
    ``run_captured`` stays in memory, except for the agents whose harness runs on the host:
    what that prints goes into the run directory, so they keep a bounded part (``keep``).
    Whatever else a module reads from a process and writes out is not checked here."""
    import ast

    import real2simlab

    exempt = {
        "sandbox/process.py": "run_logged: a pipe it reads into a bounded log",
        "media.py": "ffmpeg's own messages about a local file, read back for the error",
    }
    allowed = {"PIPE", "DEVNULL", "STDOUT"}
    package = Path(real2simlab.__file__).parent
    unbounded = []
    for path in sorted(package.rglob("*.py")):
        name = path.relative_to(package).as_posix()
        if name in exempt:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                name.startswith("agents/")
                and isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_captured"
                and "keep" not in {keyword.arg for keyword in node.keywords}
            ):
                unbounded.append(f"{name}:{node.lineno} (run_captured without keep)")
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
            ):
                continue
            for keyword in node.keywords:
                value = keyword.value
                if keyword.arg not in {"stdout", "stderr"}:
                    continue
                if isinstance(value, ast.Constant) and value.value is None:
                    continue
                if isinstance(value, ast.Attribute) and value.attr in allowed:
                    continue
                unbounded.append(f"{name}:{node.lineno} ({keyword.arg})")
    assert not unbounded, f"keep a bounded part, as sandbox.process does: {unbounded}"


# --- This machine's storage, stopped behind the Lab's own errors -----------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file-size limit")
@pytest.mark.parametrize("command", ["bench build", "train tasks"])
def test_a_file_size_limit_reached_while_encoding_a_task_video_says_so_with_the_fix(
    command: str, tmp_path: Path
) -> None:
    """``ulimit -f`` (as a full disk or a quota) reached while ffmpeg encodes a task's video:
    the kernel kills ffmpeg (SIGXFSZ), which says nothing, so the Lab says what stopped it,
    which file, and what to do; not an empty reason after an internal class name."""
    out = str(tmp_path / "tasks")
    argv = (
        ["bench", "build", "synthetic", "--select", "bouncing_ball", "--tier", "calibrated",
         "--out", out]
        if command == "bench build"
        else ["train", "tasks", "--count", "1", "--select", "pendulum", "--tier", "calibrated",
              "--out", out]
    )  # fmt: skip
    result = _in_a_new_process(["-m", "real2simlab.cli.main", *argv], max_file_size=4096)
    err = result.stderr
    assert result.returncode == 1, err
    assert "rgb.mp4: the file-size limit" in err, err
    assert "fix: free space on that disk (or raise the file-size limit" in err, err
    assert "MediaError" not in err and "Traceback" not in err, err


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file-size limit")
def test_a_file_size_limit_reached_while_copying_a_solution_says_so_with_the_fix(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """``r2s train reward`` whose copy of the solution reaches ``ulimit -f``: no reward (status
    1), one line naming the first file that could not be copied and how many others, and the
    fix; not the copy's whole list of failures as Python tuples."""
    argv = ["-m", "real2simlab.cli.main", *_reward_or_check("train reward", bouncing_task,
            tmp_path / "work")]  # fmt: skip
    result = _in_a_new_process(argv, max_file_size=4096)
    err = result.stderr
    assert result.returncode == 1, err
    assert "could not copy the solution" in err and "File too large" in err, err
    assert "other file(s)" in err, err
    assert "fix: free space on that disk (or raise the file-size limit" in err, err
    assert "InfrastructureError" not in err and "[(" not in err and "Traceback" not in err, err
    again = _in_a_new_process(argv)  # with room again, the same command scores it
    assert again.returncode == 0 and "reward 1.0000" in again.stdout + again.stderr, again.stderr


def test_a_seed_no_twin_can_be_given_is_refused_before_anything_runs(tmp_path: Path) -> None:
    """``seeds = [2**32]`` in an experiment, or ``seed=2**40`` for training tasks through the
    Python API: the experiment stopped after its builds, the training tasks on a file name
    too long. Both are refused when they are given."""
    from real2simlab.benchmarks.synthetic import SyntheticBenchmark
    from real2simlab.errors import SpecError, UsageError
    from real2simlab.orchestrate.config import ExperimentConfig

    config = {"name": "x", "tasks": {"paths": ["t"]}, "methods": [{"name": "static"}]}
    with pytest.raises((SpecError, ValueError), match="a seed must be below 4294967296"):
        ExperimentConfig.model_validate({**config, "run": {"seeds": [0, 2**32]}})
    assert ExperimentConfig.model_validate({**config, "run": {"seeds": [2**32 - 1]}})
    with pytest.raises(UsageError, match=r"seed must be in \[0, 4294967296\)"):
        SyntheticBenchmark().build(tmp_path / "t", split="train", seed=2**40)
    assert not (tmp_path / "t").exists()
