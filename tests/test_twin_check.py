"""``r2s twin check``: the pre-submission checks run a program the way an evaluation will."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from real2simlab.benchmarks.synthetic.programs import simulation_program
from real2simlab.runner.check import check_twin, format_checks, freeze_program
from real2simlab.sandbox.local import LocalBackend
from real2simlab.task import load_task
from real2simlab.task.model import DeliveryConvention
from real2simlab.twin import load_twin
from real2simlab.twin.conventions import convention_spec


def _physical(tmp_path: Path) -> Path:
    return simulation_program(
        tmp_path / "physical", phenomenon="bouncing_ball", title="t",
        params={"restitution": 0.7, "rest_speed": 0.05},
    )  # fmt: skip


def test_a_physical_twin_passes(bouncing_task: Path, tmp_path: Path, local: LocalBackend) -> None:
    task = load_task(bouncing_task)
    checks = check_twin(_physical(tmp_path), tmp_path / "check", backend=local, task=task)
    assert [c.name for c in checks] == ["run", "rerun"]
    assert all(c.ok for c in checks), format_checks(checks)
    assert "identical" in checks[1].verdict
    saved = json.loads((tmp_path / "check" / "check.json").read_text())
    assert [c["name"] for c in saved] == [c.name for c in checks]
    table = format_checks(checks)
    assert table.startswith("| run |") and "✗" not in table


def test_a_nondeterministic_program_is_caught(
    bouncing_task: Path, tmp_path: Path, local: LocalBackend
) -> None:
    program = _physical(tmp_path)
    run = program / "run.py"
    run.write_text(
        run.read_text().replace(
            'MODEL["params"]',
            '{**MODEL["params"], "restitution": 0.6 + 0.2 * __import__("random").random()}',
        )
    )
    checks = check_twin(program, tmp_path / "check", backend=local, task=load_task(bouncing_task))
    assert checks[0].ok and not checks[1].ok
    assert "DIFFERENT" in checks[1].verdict and "✗" in format_checks(checks)


def test_an_invalid_output_is_caught(
    bouncing_task: Path, tmp_path: Path, local: LocalBackend
) -> None:
    program = tmp_path / "invalid"
    program.mkdir()
    (program / "run.py").write_text(
        "import os, pathlib\n"
        'pathlib.Path(os.environ["R2S_OUTPUT_DIR"], "meta.json").write_text("{}")\n'
    )
    checks = check_twin(program, tmp_path / "check", backend=local, task=load_task(bouncing_task))
    assert checks[0].status == "invalid_output" and not checks[0].ok


def test_freeze_program_follows_the_task_convention(bouncing_task: Path, tmp_path: Path) -> None:
    program = tmp_path / "scene"
    program.mkdir()
    (program / "scene.py").write_text("print('scene')\n")
    (program / "run.py").write_text("print('run')\n")
    loaded = load_task(bouncing_task)
    delivery = loaded.config.delivery.model_copy(
        update={"convention": DeliveryConvention.BLENDER_SCENE, "output_format": "video"}
    )
    task = replace(loaded, config=loaded.config.model_copy(update={"delivery": delivery}))
    twin = load_twin(freeze_program(program, tmp_path / "twin", task))
    spec = convention_spec(DeliveryConvention.BLENDER_SCENE)
    assert twin.manifest.entrypoint == spec.entrypoint
    assert twin.manifest.runtime == spec.runtime
    plain = load_twin(freeze_program(program, tmp_path / "plain", None))
    assert plain.manifest.entrypoint == ("python", "./run.py")


@pytest.mark.parametrize("name", ["my twin", "\u5b6a\u751f", "Twin(v2)", "a,b", "-x"])
def test_a_program_directory_is_checked_whatever_its_name(
    name: str, bouncing_task: Path, tmp_path: Path, local: LocalBackend
) -> None:
    """``r2s twin check`` takes any program directory, as ``--method program:path=`` does: its
    name (a space, Chinese, brackets) is not a twin id the check may refuse."""
    program = _physical(tmp_path).rename(tmp_path / name)
    checks = check_twin(program, tmp_path / "check", backend=local, task=load_task(bouncing_task))
    assert all(c.ok for c in checks), format_checks(checks)
    twin_id = load_twin(tmp_path / "check" / "twin").manifest.twin_id
    assert twin_id.startswith("check/"), twin_id
