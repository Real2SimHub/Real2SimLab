"""Score one video2sim package with Video2World's own code (run inside ``r2s-video2world``).

    python evaluate_package.py SAMPLE PACKAGE WORK OUT_JSON

with ``V2W_DATA`` pointing at a Video2World data release and the pinned checkout on
``PYTHONPATH``. Runs the official per-configuration steps (``v2w.runner.Experiment.evaluate``,
then ``metrics``) on a copy of the package, scores the metric record with ``v2w.scoring``
(the instance's row of the benchmark score), and scores the shipped human-assisted reference
record the same way, for comparison. Standard library and Video2World only: no Real2SimLab.
"""

# pyright: reportMissingImports=false
# (runs inside r2s-video2world, where the pinned Video2World checkout provides `v2w`)

import json
import shutil
import sys
from pathlib import Path

from v2w import benchmark, paths, runner, scoring


def link_producer_paths():
    """Recreate, as symbolic links into the release, the producer's absolute paths that the
    samples' hidden records still name (``hidden/scene_gt.json`` gives meshes absolute paths
    under ``/root``), using Video2World's own table (``paths.json``, longest prefix
    first, as ``v2w.paths.resolve`` reads it). Some metric code opens those paths directly
    instead of resolving them; on the producer's machine they exist. Returns the links made."""
    hosts, table = paths._legacy()
    made = []
    for host in hosts:
        for old, new in table:
            target = paths.DATA / new
            link = Path(host) / old
            if not target.exists() or link.exists() or link.is_symlink():
                continue
            if any(parent.is_symlink() for parent in link.parents):
                continue
            try:
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(target)
            except OSError:  # a prefix outside the writable /root: not used by this release
                continue
            made.append(str(link))
    return made


def main() -> int:
    sample = sys.argv[1]
    package, work, out = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    links = link_producer_paths()
    config = benchmark.configurations(samples=[sample])[0]
    experiment = runner.Experiment(work)
    directory = experiment.dir(config)
    shutil.rmtree(directory, ignore_errors=True)
    shutil.copytree(package, directory / "pkg")
    status = {
        "eval": experiment.evaluate(config, gpu=0, video=False, retry=False),
        "metrics": experiment.metrics(config),
    }
    record = runner.load(directory / "metrics.json")
    evaluation = runner.load(directory / "eval" / "eval.json") or {}
    settings = scoring.load_config()
    scope = scoring.make_scope(benchmark.scope_rows([config]), settings)

    def row(metric_record):
        report = scoring.score({(config.family, config.sample): metric_record}, scope, settings)
        keys = ("status", "eligible", "build", "F", "G", "D", "S", "geometry", "dynamics")
        return {**{k: report["samples"][0].get(k) for k in keys}, "issues": report["issues"]}

    reference = paths.DATA / "har" / config.family / f"{sample}.json"
    result = {
        "family": config.family,
        "sample": sample,
        "steps": status,
        "producer_links": len(links),
        "evaluation_error": evaluation.get("error"),
        "score": row(record) if record is not None else None,
        "reference_score": row(json.loads(reference.read_text())) if reference.is_file() else None,
        "record": record,
    }
    out.write_text(json.dumps(result, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
