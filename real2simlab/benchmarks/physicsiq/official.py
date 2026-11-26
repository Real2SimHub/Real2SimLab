"""Physics-IQ's own end-to-end scoring, for checking the ``physics_iq`` scorer against it.

:func:`official_score` lays out a scenario's real videos, their masks and a set of generated
videos the way the official pipeline expects them, generates the generated videos' masks with
the official code, runs the official ``process_videos`` (metrics CSV) and ``calculate_iq_score``
(the benchmark score), and returns both. The comparison test runs it on real data next to the
Lab's scorer on the same videos.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from real2simlab.benchmarks.physicsiq import CODE, Case
from real2simlab.sandbox.process import run_captured

SCRIPT = """\
import json, sys
from physiq.binary_mask_generator import generate_binary_masks
from physiq.calculate_and_write_metrics_to_csv import process_videos
from physiq.calculate_iq_score import calculate_iq_score
real, real_masks, generated, generated_masks, csv = sys.argv[1:6]
generate_binary_masks(generated, generated_masks, is_real=False)
process_videos(real, generated, real_masks, generated_masks, csv, 30, "first", n_processes=0)
score, variance = calculate_iq_score(csv)
print(json.dumps({"score": float(score), "physical_variance": float(variance)}))
"""


def official_score(
    cases: Mapping[Case, Mapping[str, Path]],
    generated: Mapping[Case, Path],
    work_dir: Path,
) -> dict[str, Any]:
    """Score ``generated[case]`` (a 5-second video per case) with the official pipeline.

    ``cases[case]`` holds the real files (``testing_1``, ``testing_2``, ``mask_1``, ``mask_2``)
    as returned by :func:`real2simlab.benchmarks.physicsiq.case_files`. Returns the official
    score, its physical variance and the official per-view metrics (``rows``, from its CSV).
    """
    work_dir = Path(work_dir).resolve()
    shutil.rmtree(work_dir, ignore_errors=True)
    folders = {name: work_dir / name for name in ("real", "real_masks", "generated", "gen_masks")}
    for folder in folders.values():
        folder.mkdir(parents=True)
    for case, files in cases.items():
        for take in (1, 2):
            shutil.copyfile(
                files[f"testing_{take}"], folders["real"] / Path(case.file("testing", take)).name
            )
            shutil.copyfile(
                files[f"mask_{take}"], folders["real_masks"] / Path(case.file("mask", take)).name
            )
        name = f"{case.id1:04d}_perspective-{case.perspective}_trimmed-{case.scenario}.mp4"
        shutil.copyfile(generated[case], folders["generated"] / name)
    csv = work_dir / "metrics.csv"
    result = run_captured(  # stopped with the experiment
        [sys.executable, "-c", SCRIPT, *(str(folders[k]) for k in folders), str(csv)],
        env={**os.environ, "PYTHONPATH": str(CODE.ensure())},
        cwd=work_dir,
    )
    if result.returncode != 0:
        raise RuntimeError(f"the official pipeline failed: {result.stderr[-2000:]}")
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    import csv as csv_module

    with csv.open(newline="") as handle:
        summary["rows"] = list(csv_module.DictReader(handle))
    return summary
