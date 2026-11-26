"""The synthetic suite: five phenomena with exact truth, in two tiers.

Every task is generated from scratch by independent truth generators (``physics/``): analytic
solutions or high-precision integration, never a method's code. Videos are rendered on the CPU
by a NumPy ray caster and encoded with ffmpeg. Each phenomenon gives two tasks:

* ``synthetic/<phenomenon>/calibrated``: the video, calibrated camera, metric scale, scene
  objects (sizes, initial poses, masses where relevant) and text;
* ``synthetic/<phenomenon>/video_only``: the video and the text only.

Hidden: the reference output (the exact trajectories), the truth generator itself as a twin
package (used by the ``oracle`` baseline) and ``parameters.json`` (the hidden parameters and
the initial state). Scored by the reproduction protocol.

With the option ``split = "train"``, the adapter writes *training* tasks instead: ``count``
samples per phenomenon (option ``seed``) with parameters and initial states drawn around the
evaluation task's (``sampling``), as ``synthetic-train/<phenomenon>/<tier>/s<seed>-<index>``
with ``split = "train"``. They are never the evaluation tasks: a sample whose video the
overlap check cannot tell from the evaluation task's is skipped (its index is then missing),
and ``r2s train export`` turns them into datasets.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import numpy as np

from real2simlab import __version__
from real2simlab._project import REPOSITORY
from real2simlab.benchmarks.base import BenchmarkAdapter
from real2simlab.benchmarks.synthetic.physics import PHENOMENA, render_video_frames
from real2simlab.benchmarks.synthetic.programs import simulation_program
from real2simlab.benchmarks.synthetic.sampling import Sample, iter_samples
from real2simlab.capture import (
    CALIBRATED,
    VIDEO_ONLY,
    Capture,
    Provenance,
    SceneInfo,
    SceneObject,
    Sourced,
    SourceInfo,
    TextInfo,
    VideoStream,
    View,
    WorldFrame,
)
from real2simlab.errors import UnknownOptionError, UsageError
from real2simlab.formats.tracks import Camera, ObjectTrack, Tracks
from real2simlab.media import write_video
from real2simlab.task import (
    Delivery,
    OutputRequirements,
    Protocol,
    ScoringConfig,
    Split,
    TaskBuilder,
    TaskConfig,
)
from real2simlab.task.model import SourceInfo as TaskSource
from real2simlab.task.overlap import VideoPrint, same_recording, thumbnails
from real2simlab.twin import EnvironmentRef, Origin, TwinManifest, save_twin
from real2simlab.twin.fingerprint import sha256_file

TIERS = ("calibrated", "video_only")
VIEW = "cam"
ENVIRONMENT = "r2s-python"
DATASET = "real2simlab-synthetic"
DATASET_VERSION = "1.0"
TRAIN_BENCHMARK = "synthetic-train"
"""The first segment of training task ids: their results never mix with the suite's."""
PARAMETERS_FILE = "parameters.json"
BASE_METRICS = ("traj3d_mean", "traj3d_rmse", "traj3d_final", "traj2d_px_mean", "traj2d_norm_mean")


GENERATOR_DATE = datetime(2026, 11, 13, tzinfo=UTC)
"""The truth generators' ``created_at``: fixed, so rebuilding a task keeps its content hash."""


def tracks_of(
    simulated: dict[str, dict[str, np.ndarray]], fps: float, camera: dict[str, Any]
) -> Tracks:
    """Simulator output as ``Tracks`` with the phenomenon's camera."""
    return Tracks(
        fps=fps,
        objects={
            name: ObjectTrack(position=data["position"], quaternion=data.get("quaternion"))
            for name, data in simulated.items()
        },
        camera=Camera(
            intrinsics=np.asarray(camera["intrinsics"], dtype=np.float64),
            width=int(camera["width"]),
            height=int(camera["height"]),
            extrinsics=np.asarray(camera["extrinsics"], dtype=np.float64),
        ),
    )


def render_video(
    module: ModuleType,
    simulated: dict[str, dict[str, np.ndarray]],
    setup: dict[str, Any],
    path: Path,
    fps: float,
) -> Path:
    """Render a simulated phenomenon and encode it as H.264 at ``path``."""
    return write_video(render_video_frames(module, simulated, setup), path, fps)


def full_capture(
    module: ModuleType,
    num_frames: int,
    fps: float,
    setup: dict[str, Any] | None = None,
    capture_id: str | None = None,
) -> Capture:
    """The complete observation: video, calibrated camera, metric scale, scene, text."""
    setup = module.SETUP if setup is None else setup
    synthetic = Provenance.SYNTHETIC
    objects = tuple(
        SceneObject(
            name=item["name"],
            role=item["role"],
            shape=item.get("shape"),
            size=Sourced(value=tuple(item["size"]), provenance=synthetic)
            if "size" in item
            else None,
            mass=Sourced(value=item["mass"], provenance=synthetic) if "mass" in item else None,
            color=item.get("color"),
            initial_pose=Sourced(value=item["initial_pose"], provenance=synthetic)
            if "initial_pose" in item
            else None,
        )
        for item in module.scene_objects(setup)
    )
    camera = module.CAMERA
    return Capture(
        capture_id=capture_id or f"synthetic/{module.NAME}",
        source=SourceInfo(dataset=DATASET, version=DATASET_VERSION, license="Apache-2.0"),
        world=WorldFrame(metric_scale=Sourced(value=True, provenance=synthetic)),
        views={
            VIEW: View(
                rgb=VideoStream(
                    path=f"{VIEW}/rgb.mp4",
                    fps=fps,
                    num_frames=num_frames,
                    resolution=(int(camera["width"]), int(camera["height"])),
                ),
                intrinsics=Sourced(value=camera["intrinsics"], provenance=synthetic),
                extrinsics=Sourced(value=camera["extrinsics"], provenance=synthetic),
            )
        },
        scene=SceneInfo(
            objects=objects, environment="A flat, checkered floor at z = 0 (tiles of 0.1 m)."
        ),
        text=TextInfo(task=module.GOAL_TEXT, scene_description=module.SCENE_TEXT),
    )


def task_config(
    module: ModuleType, tier: str, num_frames: int, fps: float, sample: Sample | None = None
) -> TaskConfig:
    metrics = (*BASE_METRICS, *module.METRICS_EXTRA)
    calibrated = tier == "calibrated"
    options: dict[str, dict[str, Any]] = {}
    if not calibrated:
        options = {
            name: {"align": "similarity"} for name in ("traj3d_mean", "traj3d_rmse", "traj3d_final")
        }
    tier_text = (
        "You get the video, the calibrated camera, metric scale and the scene's objects."
        if calibrated
        else "You get only the video and a text description: no camera calibration, no scale."
    )
    task_id = f"synthetic/{module.NAME}/{tier}"
    title = f"{module.TITLE} ({tier.replace('_', ' ')})"
    metadata: dict[str, Any] = {
        "phenomenon": module.NAME,
        "generator": f"real2simlab {__version__}",
    }
    if sample is not None:
        task_id = f"{TRAIN_BENCHMARK}/{module.NAME}/{tier}/{sample.name}"
        title += f", training sample {sample.name}"
        metadata.update(seed=sample.seed, index=sample.index)
    return TaskConfig(
        id=task_id,
        version="1.0.0",
        title=title,
        description=f"{module.SCENE_TEXT} {tier_text}",
        source=TaskSource(
            dataset=DATASET,
            version=DATASET_VERSION,
            url=REPOSITORY,
            notes="Generated by real2simlab.benchmarks.synthetic; exact truth.",
        ),
        license="Apache-2.0",
        redistributable=True,
        tags=("synthetic", module.NAME, tier) + (("train",) if sample else ()),
        visibility=tier,
        split=Split.TRAIN if sample else Split.EVAL,
        delivery=Delivery(
            environment=ENVIRONMENT,
            output_format="tracks",
            agent_time_budget_s=1200,
            run_timeout_s=120,
            output=OutputRequirements(
                fps=fps, num_frames=num_frames, objects=tuple(module.OBJECTS)
            ),
        ),
        scoring=ScoringConfig(
            protocol=Protocol.REPRODUCTION,
            metrics=metrics,
            primary="traj3d_mean" if calibrated else "traj2d_norm_mean",
            metric_options=options,
        ),
        metadata=metadata,
    )


def generator_twin(
    module: ModuleType, target: Path, task_id: str, sample: Sample | None = None
) -> Path:
    """The truth generator as a twin package (privileged: it embeds the hidden parameters)."""
    with tempfile.TemporaryDirectory(prefix="r2s-generator-") as tmp:
        program = simulation_program(
            Path(tmp) / "program",
            phenomenon=module.NAME,
            title=module.TITLE,
            params=sample.params if sample else module.TRUE_PARAMS,
            setup=sample.setup if sample else None,
        )
        manifest = TwinManifest(
            twin_id=f"{task_id}/generator",
            entrypoint=("python", "./run.py"),
            environment=EnvironmentRef(image=ENVIRONMENT),
            output_format="tracks",
            origin=Origin(
                method="truth_generator",
                task_id=task_id,
                created_at=GENERATOR_DATE,  # fixed: rebuilt tasks keep their content hash
                privileged=True,
                notes="the synthetic truth generator with the hidden parameters",
            ),
        )
        save_twin(manifest, program, target)
    return target


class SyntheticBenchmark(BenchmarkAdapter):
    name: ClassVar[str] = "synthetic"
    summary: ClassVar[str] = (
        "Five synthetic phenomena (bouncing ball, pendulum, incline slider, projectile with drag, "
        "two-ball collision) with exact truth, in two visibility tiers."
    )
    license: ClassVar[str] = "Apache-2.0 (generated by Real2SimLab)"
    source: ClassVar[str] = "generated locally by real2simlab.benchmarks.synthetic"
    protocols: ClassVar[tuple[str, ...]] = ("reproduction",)
    splits: ClassVar[tuple[str, ...]] = ("eval", "train")
    look_alikes: dict[str, int]
    """Training samples the last build skipped because their video looks like the evaluation
    task's, per phenomenon."""

    def __init__(self) -> None:
        self.look_alikes = {}

    def task_names(self) -> list[str]:
        return [f"synthetic/{name}/{tier}" for name in PHENOMENA for tier in TIERS]

    def build(
        self,
        out_dir: Path,
        *,
        select: Sequence[str] | None = None,
        tiers: Sequence[str] = TIERS,
        overwrite: bool = False,
        split: str = "eval",
        count: int = 1,
        seed: int = 0,
        **options: Any,
    ) -> list[Path]:
        """Write the selected phenomena (default: all) in the given tiers under ``out_dir``:
        the evaluation tasks, or with ``split = "train"`` ``count`` training samples of each
        (``seed`` picks them)."""
        if options:
            raise UnknownOptionError.from_names(
                f"benchmark {self.name!r}", options, self.options_taken()
            )
        if split not in ("eval", "train"):
            raise UsageError(f"split must be 'eval' or 'train', not {split!r}")
        count, seed = int(count), int(seed)
        if split == "train" and count < 1:
            raise UsageError("count must be at least 1")
        from real2simlab.twin.contract import MAX_SEED

        if not 0 <= seed < MAX_SEED:  # it names the samples: a huge one was a name too long
            raise UsageError(f"seed must be in [0, {MAX_SEED})")
        # A name given twice (--select pendulum --select pendulum) is built once: a second
        # pass would find the tasks the first wrote and count them twice.
        names = list(dict.fromkeys(select)) if select else list(PHENOMENA)
        tiers = tuple(dict.fromkeys(tiers))
        unknown = sorted(set(names) - set(PHENOMENA))
        if unknown:
            raise UsageError(f"unknown synthetic phenomena {unknown}; known: {sorted(PHENOMENA)}")
        bad_tiers = sorted(set(tiers) - set(TIERS))
        if bad_tiers:
            raise UsageError(f"unknown tiers {bad_tiers}; known: {list(TIERS)}")
        built: list[Path] = []
        self.look_alikes: dict[str, int] = {}
        for name in names:
            module = PHENOMENA[name]
            if split == "eval":
                built.extend(self._build_phenomenon(module, Path(out_dir), tiers, overwrite))
                continue
            evaluation = _evaluation_video(module)
            kept = skipped = 0
            for sample in iter_samples(module, seed):
                if kept == count:
                    break
                paths = self._build_phenomenon(
                    module, Path(out_dir), tiers, overwrite, sample, evaluation
                )
                if paths:
                    built.extend(paths)
                    kept += 1
                    continue
                skipped += 1
                self.look_alikes[name] = skipped
                if skipped > MAX_SKIPPED + 10 * count:
                    raise RuntimeError(
                        f"{skipped} {name} samples look like the evaluation task: the sampled "
                        "ranges are too narrow"
                    )
        return built

    def _build_phenomenon(
        self,
        module: ModuleType,
        out_dir: Path,
        tiers: Sequence[str],
        overwrite: bool,
        sample: Sample | None = None,
        evaluation: VideoPrint | None = None,
    ) -> list[Path]:
        """Write the tiers of the evaluation task, or of a training sample; nothing when the
        sample's video cannot be told from the ``evaluation`` task's (the overlap check)."""
        fps = float(module.FPS)
        params = sample.params if sample else module.TRUE_PARAMS
        setup = sample.setup if sample else module.SETUP
        truth = module.simulate(setup, params, fps)
        num_frames = len(next(iter(truth.values()))["position"])
        truth_file = json.dumps(
            {"phenomenon": module.NAME, "params": params, "setup": setup}, indent=2
        )
        capture_id = None if sample is None else f"{TRAIN_BENCHMARK}/{sample.name}/{module.NAME}"
        paths: list[Path] = []
        with tempfile.TemporaryDirectory(prefix=f"r2s-synthetic-{module.NAME}-") as tmp:
            work = Path(tmp)
            video = render_video(module, truth, setup, work / "media" / VIEW / "rgb.mp4", fps)
            if evaluation is not None and same_recording(video_print(video), evaluation):
                return []
            capture = full_capture(module, num_frames, fps, setup, capture_id)
            for tier in tiers:
                config = task_config(module, tier, num_frames, fps, sample)
                root = out_dir / module.NAME / tier
                if sample is not None:
                    root = out_dir / TRAIN_BENCHMARK / module.NAME / tier / sample.name
                builder = TaskBuilder(root, config, overwrite=overwrite)
                builder.capture(
                    capture,
                    work / "media",
                    visibility=CALIBRATED if tier == "calibrated" else VIDEO_ONLY,
                )
                builder.reference(tracks=tracks_of(truth, fps, module.CAMERA))
                generator_twin(module, builder.hidden_dir / "generator", config.id, sample)
                builder.hidden_file(PARAMETERS_FILE, text=truth_file + "\n")
                builder.finish()
                paths.append(root)
        return paths


MAX_SKIPPED = 100
"""Samples skipped as look-alikes of the evaluation task before giving up (plus 10 per task)."""


def video_print(path: Path) -> VideoPrint:
    return VideoPrint(path.name, sha256_file(path), *thumbnails(path))


def _evaluation_video(module: ModuleType) -> VideoPrint:
    """The overlap print of the video of ``module``'s evaluation task."""
    fps = float(module.FPS)
    truth = module.simulate(module.SETUP, module.TRUE_PARAMS, fps)
    with tempfile.TemporaryDirectory(prefix=f"r2s-synthetic-{module.NAME}-") as tmp:
        return video_print(render_video(module, truth, module.SETUP, Path(tmp) / "rgb.mp4", fps))


def build_suite(
    out_dir: Path,
    *,
    select: Sequence[str] | None = None,
    tiers: Sequence[str] = TIERS,
    overwrite: bool = False,
) -> list[Path]:
    """Build the synthetic suite (all phenomena and tiers by default) under ``out_dir``."""
    return SyntheticBenchmark().build(out_dir, select=select, tiers=tiers, overwrite=overwrite)


def clean(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
