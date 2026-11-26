"""PhySimCode: sample naming, task building (offline, from a fake cache), the scorer's checks,
and PhySimCode's own parameter recovery on a real sample (when cached)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from real2simlab.benchmarks import physimcode as psc
from real2simlab.media import write_video
from real2simlab.task import load_task, validate_task

CACHE = Path(os.environ.get("R2S_CACHE_DIR", Path.home() / ".cache" / "real2simlab"))
SAMPLE = "scipy/simple_pendulum/014"
TRUTH = {
    "simulation": "a damped pendulum",
    "parameters": {
        "L": {"value": 1.2, "unit": "m", "description": "Rod length"},
        "m": {"value": 1.5, "unit": "kg", "description": "Bob mass"},
        "g": {"value": 9.8, "unit": "m/s^2", "description": "Gravitational acceleration"},
    },
}


def test_sample_names() -> None:
    assert psc.parse_sample(SAMPLE) == ("scipy", "simple_pendulum", "014")
    with pytest.raises(ValueError, match="engine/experiment/sample"):
        psc.parse_sample("simple_pendulum/014")
    with pytest.raises(ValueError, match="'kubric' engine is not supported"):
        psc.parse_sample("kubric/domino_chain/001")


@pytest.fixture
def fake_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    sample = psc.data_dir() / SAMPLE
    sample.mkdir(parents=True)
    write_video(np.zeros((20, 180, 360, 3), np.uint8), sample / "video.mp4", fps=10.0)
    (sample / "cot.json").write_text(json.dumps(TRUTH))
    (sample / "params.json").write_text(json.dumps({"L": 1.2}))
    (sample / "simulation_code.py").write_text("print('the true program')\n")
    return tmp_path


def test_tasks_validate_and_keep_the_truth_hidden(fake_cache: Path) -> None:
    (root,) = psc.PhySimCodeBenchmark().build(fake_cache / "tasks", select=[SAMPLE])
    assert root == fake_cache / "tasks" / "scipy" / "simple_pendulum" / "014"
    report = validate_task(root)
    assert not report.errors, report.errors
    task = load_task(root)
    assert task.config.id == "physimcode/scipy/simple_pendulum/014"
    assert task.config.license == "MIT" and not task.config.redistributable
    native = task.config.scoring.native
    assert (
        native is not None and native.scorer == "physimcode" and native.options == {"rerun": True}
    )
    visible = {p.relative_to(task.visible_dir).as_posix() for p in task.visible_dir.rglob("*")}
    assert "cam/rgb.mp4" in visible
    assert not any(n.endswith((".py", "cot.json", "params.json")) for n in visible)
    hidden = sorted(p.name for p in (task.hidden_dir / "reference").iterdir())
    assert hidden == ["cot.json", "params.json", "simulation_code.py"]
    text = (root / "instruction.md").read_text()
    assert "360 x 180, 10 fps" in text and "$R2S_OUTPUT_DIR/render.mp4" in text


# A stand-in for PhySimCode's eval/eval_parameters.py (the real one is tested below when its
# checkout is cached): gravity is left out, names match exactly, 20 % tolerance.
FAKE_EVAL = """
def gt_filtered_params(cot):
    return {k: v for k, v in cot.get("parameters", {}).items() if k != "g"}

def pred_params(rec):
    ok, cot = bool(rec.get("parse_ok_cot")), rec.get("parsed_cot")
    return ok, (cot.get("parameters", {}) if ok and isinstance(cot, dict) else {})

def evaluate_pair(gt, pred):
    pairs = [(k, k) for k in gt if k in pred]
    within = sum(abs(pred[p]["value"] - gt[g]["value"]) <= 0.2 * abs(gt[g]["value"])
                 for g, p in pairs)
    return {"n_pred_params": len(pred), "matched": [], "n_matched": len(pairs),
            "fraction_reported": len(pairs) / len(gt), "n_within_tolerance": within,
            "fraction_within_tolerance": within / len(pairs) if pairs else 0.0,
            "cumulative_error": 0.0}
"""


@pytest.fixture
def fake_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "PhySimCode"
    (root / "eval").mkdir(parents=True)
    (root / "eval" / "eval_parameters.py").write_text(FAKE_EVAL)
    monkeypatch.setenv("R2S_PHYSIMCODE", str(root))
    return root


def _score(root: Path, delivery: Path, work: Path) -> dict:
    from real2simlab.benchmarks.physimcode.scorer import PhySimCodeScorer

    return PhySimCodeScorer().score(
        load_task(root), delivery_dir=delivery, output_dir=None, work_dir=work, options={}
    )


def test_scorer_without_video_or_explanation(
    fake_cache: Path, fake_checkout: Path, tmp_path: Path
) -> None:
    """No cot.json, or an unreadable one: nothing recovered, every true parameter counted."""
    from real2simlab.benchmarks.physimcode.scorer import PhySimCodeScorer

    (root,) = psc.PhySimCodeBenchmark().build(fake_cache / "tasks", select=[SAMPLE])
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    report = _score(root, delivery, tmp_path / "w")
    assert report["runnability"] == {"produces_video": False}
    parameters = report["parameters"]
    assert parameters["parse_ok"] is False and parameters["detail"] == "no cot.json in the solution"
    assert parameters["n_gt"] == 2 and parameters["n_within_tolerance"] == 0
    assert PhySimCodeScorer().headline(report).value == 0.0
    (delivery / "cot.json").write_text("{not json")
    report = _score(root, delivery, tmp_path / "w")
    assert report["parameters"]["parse_ok"] is False and report["parameters"]["n_gt"] == 2


def test_headline_is_recovery_of_all_true_parameters(
    fake_cache: Path, fake_checkout: Path, tmp_path: Path
) -> None:
    """A delivery that reports one parameter, correctly, recovers one of the true parameters,
    not all of them; over tasks, parameters are pooled as PhySimCode's Table 5 pools them."""
    from real2simlab.benchmarks.physimcode.scorer import PhySimCodeScorer

    (root,) = psc.PhySimCodeBenchmark().build(fake_cache / "tasks", select=[SAMPLE])
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    (delivery / "cot.json").write_text(json.dumps({"parameters": {"L": {"value": 1.2}}}))
    one = _score(root, delivery, tmp_path / "w")
    assert one["parameters"]["fraction_within_tolerance"] == 1.0  # of the matched ones
    assert PhySimCodeScorer().headline(one).value == pytest.approx(0.5)  # of the true ones
    # Codex's case: one of ten reported and right; ten of ten reported, eight right.
    few = {
        "status": "scored",
        "parameters": {
            "parse_ok": True,
            "n_gt": 10,
            "n_matched": 1,
            "n_within_tolerance": 1,
            "fraction_within_tolerance": 1.0,
        },
    }
    most = {"status": "scored", "parameters": {"parse_ok": True, "n_gt": 10, "n_matched": 10,
                                               "n_within_tolerance": 8,
                                               "fraction_within_tolerance": 0.8}}  # fmt: skip
    scorer = PhySimCodeScorer()
    assert scorer.headline(few).value == pytest.approx(0.1)
    assert scorer.headline(most).value == pytest.approx(0.8)
    assert PhySimCodeScorer.aggregate([few, most])["value"] == pytest.approx(9 / 20)
    small = {"status": "scored", "parameters": {"parse_ok": True, "n_gt": 2,
                                                "n_within_tolerance": 2}}  # fmt: skip
    assert PhySimCodeScorer.aggregate([few, small])["value"] == pytest.approx(3 / 12)
    assert PhySimCodeScorer.aggregate([{"status": "not_delivered"}])["value"] is None


def _checkout_cached() -> bool:
    return (psc.CODE.path() / ".r2s-complete").is_file()


@pytest.mark.skipif(not _checkout_cached(), reason="the PhySimCode checkout is not cached")
def test_parameter_recovery_is_physimcodes(tmp_path: Path) -> None:
    """PhySimCode's matching: gravity is left out, names match by alias, 20 % tolerance."""
    from real2simlab.benchmarks.physimcode.scorer import parameter_recovery

    truth, delivered = tmp_path / "truth.json", tmp_path / "delivered.json"
    truth.write_text(json.dumps(TRUTH))
    delivered.write_text(json.dumps({"parameters": {
        "length": {"value": 1.3, "unit": "m"},           # within 20 %
        "mass": {"value": 3.0, "unit": "kg"},            # 100 % off
        "g": {"value": 9.81, "unit": "m/s^2"},           # not scored
    }}))  # fmt: skip
    result = parameter_recovery(truth, delivered, tmp_path)
    assert result["n_gt"] == 2 and result["n_matched"] == 2
    assert result["fraction_reported"] == 1.0 and result["fraction_within_tolerance"] == 0.5
    delivered.write_text(json.dumps({"parameters": {}}))
    assert parameter_recovery(truth, delivered, tmp_path)["fraction_reported"] == 0.0
