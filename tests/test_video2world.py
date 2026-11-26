"""Video2World: the package convention and format, task building (offline, from a fake data
release and checkout), the official brief, and the reference packages scored by Video2World's
own evaluator against its shipped records (when the data and the image are available)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from real2simlab.benchmarks import video2world as v2w
from real2simlab.formats import get_format
from real2simlab.media import write_video
from real2simlab.task import load_task, validate_task
from real2simlab.twin.conventions import DeliveryConvention, convention_spec

from .conftest import docker_image_ready

CACHE = Path(os.environ.get("R2S_CACHE_DIR", Path.home() / ".cache" / "real2simlab"))
SAMPLE = "twin_pusht_demo_0"
HAND = "hot3d_x"
"""A sample in two families (the arm and the two-hand settings), as in Video2World."""
TWIN = "twin_rope_x"


def test_package_convention_is_data_not_a_program() -> None:
    spec = convention_spec(DeliveryConvention.VIDEO2SIM_PACKAGE)
    assert spec.entry_file == "protocol.json" and spec.session_output == "solution"
    assert not spec.runnable
    assert convention_spec(DeliveryConvention.R2S).runnable


def test_package_format(tmp_path: Path) -> None:
    fmt = get_format("video2sim_package")
    assert fmt.validate(tmp_path) == ["protocol.json: file is required"]
    (tmp_path / "protocol.json").write_text("[1]")
    assert fmt.validate(tmp_path) == ["protocol.json: must be a JSON object"]
    (tmp_path / "protocol.json").write_text(json.dumps({"actions": {"path": "a.npy"}}))
    assert fmt.validate(tmp_path) == ["no action stream: expected one of a.npy"]
    np.save(tmp_path / "a.npy", np.zeros((2, 7)))
    assert fmt.validate(tmp_path) == []


def _fake_checkout(root: Path) -> Path:
    profiles = root / "v2w" / "agents" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "fb_pusht.md").write_text("Push the T in `{video}` ({W}x{H}, {T} frames).\n")
    (profiles / "common.md").write_text(
        '{{"robot": {{"uid": "widowx250s_bridge"}}}} // nominal: the robot is NOT part of this '
        "profile\nWrite `{out}`; validate with PYTHONPATH={ours} {py_eval}; name {name}.\n"
    )
    for profile in ("ego_arm", "ego_bimanual"):
        (profiles / f"{profile}.md").write_text("Replay the hand in `{video}`; name {name}.\n")
    (profiles / "twin_rope.md").write_text("Route the rope in `{video}` with {py}.\n")
    (root / "video2sim").mkdir()
    (root / "video2sim" / "__init__.py").write_text("")
    (root / "video2sim" / "cli.py").write_text("# the package validator\n")
    # Video2World's own handout stager, reduced to what the adapter relies on.
    agents = root / "v2w" / "agents"
    (root / "v2w" / "__init__.py").write_text("")
    (agents / "__init__.py").write_text("")
    (agents / "handouts.py").write_text(
        "import json, os, shutil\nfrom pathlib import Path\n"
        "def stage_twin(profile, destination):\n"
        "    dst = Path(destination)\n"
        "    shutil.copytree(Path(os.environ['V2W_DATA']) / 'assets/robots/twin_xarm7', dst / 'robot')\n"
        "    (dst / 'public_task.json').write_text(json.dumps({'profile': profile}))\n"
        "    return dst / 'public_task.json'\n"
    )
    return root


def _fake_release(data: Path) -> None:
    row = {"family": "push-t", "sample": SAMPLE,
           "metric_family": {"source": "real2sim-eval xarm7_pusht", "profile": "fb_pusht",
                             "evaluator": "fb"}}  # fmt: skip
    arm = {"family": "hot3d-pickplace-arm", "sample": HAND,
           "metric_family": {"source": "HOT3D", "profile": "ego_arm", "evaluator": "ego_arm"}}  # fmt: skip
    both = {"family": "hot3d-pickplace-bimanual", "sample": HAND,
            "metric_family": {"source": "HOT3D", "profile": "ego_bimanual",
                              "evaluator": "ego_bimanual"}}  # fmt: skip
    twin = {"family": "rope-route", "sample": TWIN,
            "metric_family": {"source": "real2sim-eval rope", "profile": "twin_rope",
                              "evaluator": "twin"}}  # fmt: skip
    odd = {"family": "odd", "sample": "odd_0",
           "metric_family": {"source": "HOT3D", "profile": "ego_arm", "evaluator": "new_sim"}}  # fmt: skip
    rows = [row, arm, both, twin, odd]
    (data).mkdir(parents=True)
    (data / "manifest.json").write_text(
        json.dumps({"version": "video2world/1", "configurations": len(rows), "samples": rows})
    )
    (data / "paths.json").write_text("{}")
    sample = data / "samples" / SAMPLE
    (sample / "hidden").mkdir(parents=True)
    (sample / "gt_pkg").mkdir()
    write_video(np.zeros((10, 32, 32, 3), np.uint8), sample / "video.mp4", fps=5.0)
    (sample / "meta.json").write_text(json.dumps({"T": 10, "resolution": [32, 32], "fps": 5.0}))
    (sample / "hidden" / "trajectory.npz").write_bytes(b"secret")
    (sample / "gt_pkg" / "protocol.json").write_text("{}")
    (data / "har" / "push-t").mkdir(parents=True)
    (data / "har" / "push-t" / f"{SAMPLE}.json").write_text(json.dumps({"build": True}))
    for family, name in (("hot3d-pickplace-arm", HAND), ("hot3d-pickplace-bimanual", HAND),
                         ("rope-route", TWIN)):  # fmt: skip
        sample = data / "samples" / name
        (sample / "hidden").mkdir(parents=True, exist_ok=True)
        write_video(np.zeros((10, 32, 32, 3), np.uint8), sample / "video.mp4", fps=5.0)
        (sample / "meta.json").write_text(json.dumps({"T": 10, "resolution": [32, 32]}))
        (sample / "hidden" / "truth.json").write_text('{"secret": 1}')
        (data / "har" / family).mkdir(parents=True, exist_ok=True)
        (data / "har" / family / f"{name}.json").write_text("{}")
    # The reference twin package carries the public robot's meshes, as the handout does.
    robot = data / "assets" / "robots" / "twin_xarm7" / "meshes"
    robot.mkdir(parents=True)
    (robot / "link1.stl").write_bytes(b"solid public link")
    (data / "samples" / TWIN / "gt_pkg" / "robot").mkdir(parents=True)
    shutil.copyfile(robot / "link1.stl", data / "samples" / TWIN / "gt_pkg" / "robot" / "link1.stl")


@pytest.fixture
def fake_v2w(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("R2S_VIDEO2WORLD", str(_fake_checkout(tmp_path / "checkout")))
    _fake_release(v2w.data_dir())
    marker = v2w._marker("handout:assets/robots/twin_xarm7/**")  # pyright: ignore[reportPrivateUsage]
    marker.parent.mkdir(parents=True)
    marker.write_text("complete\n")
    monkeypatch.setattr(v2w, "_hub", lambda: pytest.fail("no download expected"))
    return tmp_path


def test_tasks_validate_and_keep_the_truth_hidden(fake_v2w: Path) -> None:
    out = fake_v2w / "tasks"
    (root,) = v2w.Video2WorldBenchmark().build(out, select=[SAMPLE])
    assert root == out / "push-t" / SAMPLE
    report = validate_task(root)
    assert not report.errors, report.errors
    task = load_task(root)
    assert task.config.id == f"video2world/push-t/{SAMPLE}"
    assert task.config.license == "MIT" and not task.config.redistributable
    assert task.config.delivery.convention is DeliveryConvention.VIDEO2SIM_PACKAGE
    native = task.config.scoring.native
    assert native is not None and native.scorer == "video2world" and native.options == {}
    visible = {p.relative_to(task.visible_dir).as_posix() for p in task.visible_dir.rglob("*")}
    assert "cam/rgb.mp4" in visible and "tools/video2sim/cli.py" in visible
    assert not any(name.startswith(("hidden", "gt_pkg")) or "meta" in name for name in visible)
    release = task.hidden_dir / "v2w"
    manifest = json.loads((release / "manifest.json").read_text())
    assert [r["sample"] for r in manifest["samples"]] == [SAMPLE]
    assert (release / "samples" / SAMPLE / "hidden" / "trajectory.npz").is_file()
    assert (release / "har" / "push-t" / f"{SAMPLE}.json").is_file()
    text = (root / "instruction.md").read_text()
    assert "`{workspace}/visible/cam/rgb.mp4` (32x32, 10 frames)" in text
    assert '"uid": "xarm7_pusher"' in text and "// actual execution robot" in text
    assert "PYTHONPATH={workspace}/visible/tools python" in text
    assert "robot.uid=xarm7_pusher" in text


def test_instances_are_named_by_family_and_sample(fake_v2w: Path) -> None:
    assert sorted(v2w.instances()) == sorted(
        ["push-t/" + SAMPLE, f"hot3d-pickplace-arm/{HAND}", f"hot3d-pickplace-bimanual/{HAND}",
         f"rope-route/{TWIN}", "odd/odd_0"]
    )  # fmt: skip
    assert v2w.instance(SAMPLE).key == f"push-t/{SAMPLE}"
    assert v2w.instance(f"hot3d-pickplace-arm/{HAND}").profile == "ego_arm"
    with pytest.raises(ValueError, match="several families: name one of hot3d-pickplace-arm/"):
        v2w.instance(HAND)
    with pytest.raises(ValueError, match="is a family"):
        v2w.instance("rope-route")
    with pytest.raises(ValueError, match="'new_sim' evaluator, which this adapter does not know"):
        v2w.instance("odd_0")
    with pytest.raises(ValueError, match="no Video2World sample or family"):
        v2w.instance("nothing")
    with pytest.raises(TypeError, match="unknown options"):
        v2w.Video2WorldBenchmark().build(fake_v2w / "x", select=[SAMPLE], colour="red")


def test_a_sample_in_two_families_is_two_tasks(fake_v2w: Path) -> None:
    roots = v2w.Video2WorldBenchmark().build(fake_v2w / "tasks", select=[HAND])
    assert [r.relative_to(fake_v2w / "tasks").as_posix() for r in roots] == [
        f"hot3d-pickplace-arm/{HAND}",
        f"hot3d-pickplace-bimanual/{HAND}",
    ]
    task = load_task(roots[0])
    assert task.config.license == "CC-BY-SA AND CC-BY-NC-SA" and not task.config.redistributable
    assert task.config.delivery.resources.gpu  # SAPIEN replays need a Vulkan device
    text = (roots[0] / "instruction.md").read_text()
    assert f"name {HAND}__ego_arm." in text  # Video2World's name for hand agents
    assert "Execution robot: panda. This profile uses the benchmark hand adapter" in text
    assert "Write `{workspace}/solution`" in text  # common.md follows the hand profiles


def test_a_handout_file_the_truth_shares_stays_public_only(fake_v2w: Path) -> None:
    """The reference twin carries the public robot meshes the handout gives the agent: the task
    keeps one copy, in visible/, and the scorer puts it back into the data release."""
    (root,) = v2w.Video2WorldBenchmark().build(fake_v2w / "tasks", select=[TWIN])
    assert not validate_task(root).errors
    task = load_task(root)
    handout = task.visible_dir / "tools" / "handout"
    assert (handout / "robot" / "meshes" / "link1.stl").is_file()
    assert json.loads((handout / "public_task.json").read_text()) == {"profile": "twin_rope"}
    release = task.hidden_dir / "v2w"
    mesh = f"samples/{TWIN}/gt_pkg/robot/link1.stl"
    assert not (release / mesh).exists()
    assert json.loads((release / v2w.PUBLIC_FILES).read_text()) == {
        mesh: "tools/handout/robot/meshes/link1.stl"
    }
    assert (release / "samples" / TWIN / "hidden" / "truth.json").is_file()
    text = (root / "instruction.md").read_text()
    assert f"Route the rope in `{{workspace}}/visible/cam/rgb.mp4` with {v2w.TWIN_PYTHON}." in text
    assert "Public robot contract: {workspace}/visible/tools/handout/public_task.json" in text
    copy = fake_v2w / "copy"
    shutil.copytree(release, copy)
    assert v2w.restore_public_files(copy, task.visible_dir) == 1
    assert (copy / mesh).read_bytes() == b"solid public link"


def test_every_evaluator_has_a_runtime_and_every_source_a_license() -> None:
    assert set(v2w.EVALUATORS) == {
        "fb", "ego_arm", "ego_bimanual", "twin", "twin_toy", "cloth", "robodojo_native_task",
        "inhouse", "inhouse_point_robot", "inhouse_reviewed", "inhouse_wallet", "inhouse_fryer",
    }  # fmt: skip
    isaac = {name for name, e in v2w.EVALUATORS.items() if e.image == v2w.ISAAC_ENVIRONMENT}
    assert isaac == {"robodojo_native_task"} | {
        n for n in v2w.EVALUATORS if n.startswith("inhouse")
    }
    assert all(v2w.EVALUATORS[n].mount_shared and not v2w.EVALUATORS[n].agent_gpu for n in isaac)
    assert [n for n, e in v2w.EVALUATORS.items() if e.robodojo_assets] == ["robodojo_native_task"]
    noncommercial = {k for k, s in v2w.SOURCES.items() if "NC" in s.license}
    assert noncommercial == {"DexYCB", "HOI4D", "HOT3D"}


def test_scorer_without_a_package(fake_v2w: Path, tmp_path: Path) -> None:
    from real2simlab.benchmarks.video2world.scorer import Video2WorldScorer

    (root,) = v2w.Video2WorldBenchmark().build(fake_v2w / "tasks", select=[SAMPLE])
    empty = tmp_path / "empty"
    empty.mkdir()
    report = Video2WorldScorer().score(
        load_task(root), delivery_dir=empty, output_dir=None, work_dir=tmp_path / "w", options={}
    )
    # Nothing to build scores as a package that does not build: 0, not left out.
    assert report["status"] == "scored" and report["official_status"] == "no_package"
    assert (report["build"], report["S"]) == (0, 0.0)


def _blocked(*reasons: str) -> dict[str, Any]:
    """The Lab's report of a row Video2World's scoring blocked for ``reasons``."""
    issues = [{"family": "push-t", "sample": SAMPLE, "reason": reason} for reason in reasons]
    return {"status": "blocked", "official_status": "blocked", "S": None, "score": None,
            "issues": issues}  # fmt: skip


@pytest.mark.parametrize(
    ("report", "infra"),
    [
        # What Video2World's own runner and scoring (df0e00b) blame on its evaluator: a crash or
        # a timeout, a renderer or GPU that died, metric code that raised, no metric record.
        (_blocked("evaluator failure: evaluator exited with -9, see eval/eval.log"), True),
        (_blocked("evaluator failure: infrastructure error in the evaluator: DeviceLost"), True),
        (_blocked("evaluator failure: KeyError: 'rollout'"), True),
        (_blocked("evaluator metric error: ValueError: empty trajectory"), True),
        (_blocked("missing record"), True),
        (_blocked("missing metric record"), True),
        ({"status": "blocked", "official_status": None, "S": None, "issues": None}, True),
        # No build verdict from the evaluator, which writes it itself (it could not read the
        # sample): never the package's.
        pytest.param(_blocked("build must be boolean"), True, id="no-build-verdict"),
        # An evaluation error, which Video2World's runner writes only when its evaluator failed,
        # with no issue (A12-review-07's report).
        pytest.param(
            {
                "status": "blocked",
                "official_status": "blocked",
                "S": None,
                "evaluation_error": "scorer could not initialize",
            },
            True,
            id="evaluation-error",
        ),
        # No issue at all: a sample Video2World does not score.
        pytest.param(
            {
                "status": "blocked",
                "official_status": "not_applicable",
                "S": None,
                "score": None,
                "issues": [],
            },
            True,
            id="not-applicable",
        ),
        # Scores, a package that does not build included.
        ({"status": "scored", "official_status": "ok", "S": 0.4}, False),
        ({"status": "scored", "official_status": "build_failed", "S": 0.0}, False),
        # A measurement of this package that Video2World will not score stays its result.
        (_blocked("scene_chamfer_cm: expected finite number"), False),
    ],
)
def test_rows_the_evaluator_could_not_measure_are_not_scores(
    report: dict[str, Any], infra: bool
) -> None:
    from real2simlab.benchmarks.video2world.scorer import Video2WorldScorer
    from real2simlab.scoring.native import scorer_failure

    failure = scorer_failure(Video2WorldScorer(), report)
    assert (failure is not None) is infra, failure
    if infra:
        assert "video2world" in (failure or "")


CHECKOUT = v2w.CODE.path()
"""The pinned checkout in the cache the tests started with (the fixtures move the cache)."""
RECORDS: dict[str, Any] = {
    # what v2w.runner.Experiment.metrics writes when the evaluator died ...
    "crash": {"sample": SAMPLE, "build": None, "task_success": None, "error_kind": "evaluator",
              "error": "evaluator exited with -9, see eval/eval.log"},
    # ... or when the track's metric code raised
    "metric": {"sample": SAMPLE, "error": "RuntimeError: metric crashed",
               "error_kind": "evaluator"},
    # a record whose metric code reported its own error
    "metric_error": {"sample": SAMPLE, "build": True, "task_success": True, "error": "KeyError",
                     "progress": {"progress": 1.0}, "scene_chamfer_cm": 1.0,
                     "own_trajectory": {"ape": {"trans_cm": 1.0}}},
    # what rigid_record writes when the fb evaluator's scene report has no build verdict (it
    # could not read the sample's meta.json): no boolean build evidence
    "no_build_verdict": {"sample": SAMPLE, "schema_version": "eval2/4", "build": None,
                         "scene_chamfer_cm": None,
                         "metric_status": {"build": {"status": "error",
                                                     "reason": "missing or invalid build evidence"}}},
    # a measured package, and one that does not build (the method's failure)
    "ok": {"sample": SAMPLE, "build": True, "task_success": True, "progress": {"progress": 1.0},
           "scene_chamfer_cm": 1.0, "own_trajectory": {"ape": {"trans_cm": 1.0}}},
    "no_build": {"sample": SAMPLE, "build": False, "task_success": False,
                 "error_kind": "agent", "error": "agent produced no package"},
    # a measured package on a sample Video2World does not score (INELIGIBLE)
    "ineligible": {"sample": SAMPLE, "build": True, "task_success": True,
                   "progress": {"progress": 1.0}, "scene_chamfer_cm": 1.0,
                   "own_trajectory": {"ape": {"trans_cm": 1.0}}},
}  # fmt: skip
INELIGIBLE = ("ineligible",)
"""Records scored on a sample whose metric contract leaves nothing to measure, so that
Video2World's own scope (``make_scope``) makes it ineligible."""
OFFICIAL_ROWS = """\
import json, sys
from v2w import scoring
settings = scoring.load_config()
measured = ["progress.progress", "scene_chamfer_cm", "own_trajectory.ape.trans_cm"]
scope = {"config_sha256": "test", "rows": [{
    "family": "push-t", "sample": sys.argv[2], "eligible": True, "exclusion_reasons": [],
    "metric_family": {"metric_contract": {"applicable": measured}},
    "geometry": ["scene_chamfer_cm"], "dynamics": ["own_trajectory.ape.trans_cm"]}]}
nothing = scoring.make_scope(
    [{"family": "push-t", "sample": sys.argv[2], "metric_family": {}}], settings)
assert not nothing["rows"][0]["eligible"]
keys = ("status", "eligible", "build", "F", "G", "D", "S", "geometry", "dynamics")
rows = {}
for name, record in json.loads(sys.argv[1]).items():  # as evaluate_package.py's row()
    on = nothing if name in sys.argv[3:] else scope
    report = scoring.score({("push-t", sys.argv[2]): record}, on, settings)
    rows[name] = {**{k: report["samples"][0].get(k) for k in keys}, "issues": report["issues"]}
print(json.dumps(rows))
"""


@pytest.mark.skipif(
    not (CHECKOUT / ".r2s-complete").is_file(), reason="the Video2World checkout is not cached"
)
def test_the_scorer_attributes_failures_as_video2world_does(
    fake_v2w: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Video2World's own scoring of the records its runner writes, through the Lab's scorer with
    only the container replaced: what Video2World blames on its evaluator (and no record at
    all, no build verdict, a sample it does not score) is an infrastructure error; a score,
    even of a package that does not build, is one."""
    from real2simlab.benchmarks.video2world import scorer as module
    from real2simlab.scoring.native import scorer_failure

    done = subprocess.run(
        [sys.executable, "-c", OFFICIAL_ROWS, json.dumps(RECORDS), SAMPLE, *INELIGIBLE],
        capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(CHECKOUT)},
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-2000:]
    rows: dict[str, Any] = {**json.loads(done.stdout), "no_record": None}
    assert rows["no_build_verdict"]["issues"][0]["reason"] == "build must be boolean"
    assert (rows["ineligible"]["status"], rows["ineligible"]["issues"]) == ("not_applicable", [])
    (root,) = v2w.Video2WorldBenchmark().build(fake_v2w / "tasks", select=[SAMPLE])
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "protocol.json").write_text("{}")
    (tmp_path / "assets").mkdir()
    # The evaluator's shared data (the rigs' assets, for push-t), as if it were cached.
    monkeypatch.setattr(module, "shared_data", lambda runtime: {"assets": tmp_path / "assets"})
    monkeypatch.setattr(module.Video2WorldScorer, "problems", lambda self, environment="": [])
    scorer = module.Video2WorldScorer()
    infra = {"crash", "metric", "metric_error", "no_record", "no_build_verdict", "ineligible"}
    for name, row in rows.items():
        work = tmp_path / "w" / name
        record = RECORDS.get(name)

        def container(
            docker: str, args: Any, log: Path, timeout_s: float = 0.0, work: Path = work,
            row: Any = row, record: Any = record,
        ) -> tuple[int, bool, float]:  # fmt: skip
            result = {"family": "push-t", "sample": SAMPLE, "steps": {}, "producer_links": 0,
                      "evaluation_error": None, "score": row, "reference_score": None,
                      "record": record}  # fmt: skip
            (work.resolve() / "v2w" / "result.json").write_text(json.dumps(result))
            return 0, False, 0.0

        monkeypatch.setattr(module, "run_container", container)
        report = scorer.score(
            load_task(root), delivery_dir=package, output_dir=None, work_dir=work, options={}
        )
        failure = scorer_failure(scorer, report)
        assert (failure is not None) is (name in infra), (name, report)
        assert (report["status"] == "scored") is (name in {"ok", "no_build"}), (name, report)


def test_twin_check_refuses_to_run_a_package(fake_v2w: Path, tmp_path: Path) -> None:
    from real2simlab.errors import SetupError
    from real2simlab.runner.check import check_twin
    from real2simlab.sandbox.local import LocalBackend

    (root,) = v2w.Video2WorldBenchmark().build(fake_v2w / "tasks", select=[SAMPLE])
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "protocol.json").write_text(json.dumps({"actions": {"path": "actions.npy"}}))
    np.save(package / "actions.npy", np.zeros((2, 7)))
    with pytest.raises(SetupError, match="data, not programs"):
        check_twin(package, tmp_path / "out", backend=LocalBackend(), task=load_task(root))


# The real thing: Video2World's own brief and evaluator, when cached and built.

DATA = CACHE / "data" / "Video2World" / v2w.REVISION[:12]


def _cached() -> bool:
    if not (v2w.CODE.path() / ".r2s-complete").is_file():
        return False
    return all(
        (DATA / "samples" / s / "gt_pkg" / "protocol.json").is_file() for s in v2w.DEFAULT_SAMPLES
    )


@pytest.mark.skipif(not _cached(), reason="the Video2World checkout and samples are not cached")
def test_brief_matches_the_official_composition() -> None:
    """``official_brief`` against ``v2w.agents.run.brief`` from the pinned checkout."""
    fields: dict[str, Any] = {"video": "V", "out": "O", "name": "N", "W": 1, "H": 2, "T": 3,
                              "ours": "L", "repo": "R", "py_eval": "P", "py": "Q"}  # fmt: skip
    profiles = sorted(
        p.stem
        for p in (v2w.CODE.ensure() / "v2w/agents/profiles").glob("*.md")
        if p.stem != "common"
    )
    assert len(profiles) == 11
    code = (
        "import json, sys; from v2w.agents.run import brief; "
        "print(json.dumps({p: brief(p).format(**json.loads(sys.argv[1])) for p in "
        "json.loads(sys.argv[2])}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(fields), json.dumps(profiles)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(v2w.CODE.ensure()), "V2W_DATA": str(DATA)},
        check=False,
    )  # fmt: skip
    if result.returncode != 0 and "ModuleNotFoundError" in result.stderr:
        pytest.skip(f"the official agent module needs more packages: {result.stderr[-200:]}")
    assert result.returncode == 0, result.stderr[-2000:]
    official = json.loads(result.stdout)
    for profile, text in official.items():
        assert v2w.official_brief(v2w.CODE.ensure(), profile, fields) == text, profile


REPRODUCES = {"push-t"}
"""Families whose released reference package, executed by the released evaluator, reproduces
the shipped record. DROID's records were not re-simulated (``har_refresh.simulation_rerun`` is
false; they come from an earlier execution) and its released references miss the grasp by about
1.7 cm with the released evaluator: their geometry reproduces, their outcome does not (D-042)."""


@pytest.mark.slow
@pytest.mark.docker
@pytest.mark.skipif(not _cached(), reason="the Video2World checkout and samples are not cached")
def test_reference_packages_reproduce_the_shipped_records(tmp_path: Path) -> None:
    """Each sample's human-assisted reference package, scored by the Lab's scorer, against the
    score Video2World's scoring gives its shipped record: S where the record reproduces with
    the released code, the geometry G everywhere."""
    from real2simlab.envs import image_reference

    if not docker_image_ready(image_reference(v2w.ENVIRONMENT)):
        pytest.skip(f"r2s env build {v2w.ENVIRONMENT}")
    from real2simlab.benchmarks.video2world.scorer import Video2WorldScorer

    roots = v2w.Video2WorldBenchmark().build(tmp_path / "tasks")
    for root in roots:
        task = load_task(root)
        sample = root.name
        package = tmp_path / "pkg" / sample
        shutil.copytree(DATA / "samples" / sample / "gt_pkg", package)
        report = Video2WorldScorer().score(
            task, delivery_dir=package, output_dir=None, work_dir=tmp_path / "w" / sample,
            options={},
        )  # fmt: skip
        assert report["status"] == "scored", (sample, report)
        assert report["build"] is True, (sample, report)
        reference = report["reference"]
        assert report["G"] == pytest.approx(reference["G"], abs=0.002), (sample, report)
        if root.parent.name in REPRODUCES:
            assert report["S"] == pytest.approx(reference["S"], abs=0.002), (sample, report)
            assert report["F"] == reference["F"], (sample, report)
