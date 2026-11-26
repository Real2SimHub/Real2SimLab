"""PhySimCode's model-free measures as a native scorer plugin (``physimcode``).

* **Runnability**: whether the Lab's frozen, offline re-run of ``solution/run.py`` produced a
  readable video (PhySimCode checks that the program runs and writes a non-empty video).
* **Parameter recovery**: PhySimCode's own ``eval/eval_parameters.py`` (pinned checkout, run in
  a subprocess): the true parameters, filtered as PhySimCode filters them (gravity, colours,
  positions, camera and the like left out), matched to the delivered ``cot.json`` by name
  aliases, compared in SI units; the report keeps its ``n_gt`` (true parameters),
  ``n_matched``, ``n_within_tolerance`` (relative error at most 20 %), ``fraction_reported``
  (``n_matched / n_gt``), ``fraction_within_tolerance`` (``n_within_tolerance / n_matched``:
  the accuracy among the *matched* parameters only) and ``cumulative_error``.

The headline is PhySimCode's **end-to-end recovery** (the measure its paper's Table 5
reports): the share of the true parameters recovered within 20 %, ``n_within_tolerance /
n_gt``. A delivery that reports one parameter correctly out of ten true ones recovers 0.1, not
1. Over several tasks, parameters are pooled as PhySimCode pools them: the sum of
``n_within_tolerance`` over the sum of ``n_gt``. A delivery without a readable ``cot.json``
recovers nothing but still counts its true parameters.

PhySimCode's other measures need commercial LLM judges (physical-law validity and
equivalence), model weights (DINOv2 and X-CLIP video similarity) or tree-sitter grammars
(CodeBLEU); they are not run.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from real2simlab.benchmarks.physimcode import CODE, read_reference
from real2simlab.media import MediaError, probe_video
from real2simlab.sandbox.process import run_captured
from real2simlab.scoring.model import Headline
from real2simlab.scoring.native import NativeScorer
from real2simlab.task.io import Task

SCRIPT = """\
import json, sys
from eval.eval_parameters import evaluate_pair, gt_filtered_params, pred_params
truth = json.load(open(sys.argv[1]))
delivered = json.load(open(sys.argv[2])) if sys.argv[2] else None
gt = gt_filtered_params(truth)
parse_ok, pred = pred_params({"parse_ok_cot": isinstance(delivered, dict), "parsed_cot": delivered})
print(json.dumps({"n_gt": len(gt), "gt": gt, "parse_ok": parse_ok, **evaluate_pair(gt, pred)}))
"""


def parameter_recovery(truth: Path, delivered: Path | None, work_dir: Path) -> dict[str, Any]:
    """PhySimCode's parameter matching of ``delivered`` (a ``cot.json``) against ``truth``;
    ``None`` for a delivery without a readable one (nothing matched, every true parameter
    counted)."""
    argument = "" if delivered is None else str(delivered.resolve())
    result = run_captured(  # stopped with the experiment
        [sys.executable, "-c", SCRIPT, str(truth.resolve()), argument],
        cwd=work_dir,
        env={**os.environ, "PYTHONPATH": str(CODE.ensure())},
    )
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        raise RuntimeError(f"PhySimCode's parameter evaluation failed: {lines[-1:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


class PhySimCodeScorer(NativeScorer):
    name: ClassVar[str] = "physimcode"
    task_options: ClassVar[tuple[str, ...] | None] = ()
    summary: ClassVar[str] = "PhySimCode's runnability and parameter recovery (within 20 %)"

    def headline(self, report: Mapping[str, Any]) -> Headline:
        """End-to-end recovery: the share of the true parameters within 20 %."""
        counts = _counts(report)
        value = None
        if counts is not None:
            within, truth = counts
            value = within / truth if truth else None
        elif report.get("status") == "scored":
            value = 0.0  # scored, but nothing readable was delivered: nothing recovered
        return Headline(
            metric="parameters_within_20pct",
            value=value,
            range=(0.0, 1.0),
            source="official",
            scorer=self.name,
        )

    @classmethod
    def aggregate(cls, reports: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
        """Recovery over several tasks as PhySimCode pools it: every task's parameters
        together, the sum of those within 20 % over the sum of true parameters."""
        counts = [c for c in (_counts(r) for r in reports) if c is not None]
        truth = sum(t for _, t in counts)
        return {"value": sum(w for w, _ in counts) / truth if truth else None}

    def score(
        self,
        task: Task,
        *,
        delivery_dir: Path,
        output_dir: Path | None,
        work_dir: Path,
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {"status": "scored"}
        video = None if output_dir is None else output_dir / "render.mp4"
        runnable: dict[str, Any] = {"produces_video": False}
        if video is not None and video.is_file():
            try:
                info = probe_video(video)
                runnable = {
                    "produces_video": True,
                    "num_frames": info.num_frames,
                    "fps": info.fps,
                    "width": info.width,
                    "height": info.height,
                }
            except MediaError as error:
                runnable = {"produces_video": False, "detail": str(error)}
        report["runnability"] = runnable
        delivered: Path | None = delivery_dir / "cot.json"
        detail = None
        try:
            json.loads((delivery_dir / "cot.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            delivered, detail = None, "no cot.json in the solution"
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            delivered, detail = None, f"cot.json: {error}"
        # Without a readable cot.json nothing is matched, but the true parameters still count.
        truth = work_dir / "truth_cot.json"
        truth.write_text(json.dumps(read_reference(task.hidden_dir)), encoding="utf-8")
        try:
            recovery = parameter_recovery(truth, delivered, work_dir)
        finally:
            truth.unlink(missing_ok=True)
        keep = (
            "parse_ok", "n_gt", "n_pred_params", "n_matched", "fraction_reported",
            "n_within_tolerance", "fraction_within_tolerance", "cumulative_error", "matched",
        )  # fmt: skip
        parameters = {k: recovery.get(k) for k in keep}
        if detail is not None:
            parameters["parse_ok"] = False
            parameters["detail"] = detail
        report["parameters"] = parameters
        return report


def _counts(report: Mapping[str, Any]) -> tuple[int, int] | None:
    """``(n_within_tolerance, n_gt)`` of a scored report, or ``None`` without them."""
    parameters = report.get("parameters")
    if report.get("status") != "scored" or not isinstance(parameters, Mapping):
        return None
    within, truth = parameters.get("n_within_tolerance"), parameters.get("n_gt")
    if isinstance(within, int) and isinstance(truth, int) and not isinstance(truth, bool):
        return within, truth
    return None
