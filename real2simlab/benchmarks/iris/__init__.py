"""IRIS: real recordings of controlled physical events, as reproduction tasks.

Source: the Hugging Face dataset ``rasulkhanbayov/IRIS`` (revision pinned below), CC-BY-NC-4.0.
Real2SimLab does not redistribute it: the videos are downloaded into the user's cache when the
tasks are built, and the task directories are marked ``redistributable = false``.

Khanbayov, Barhdadi, Serpedin, Kurban. *IRIS: A Real-World Benchmark for Inverse Recovery and
Identification of Physical Dynamic Systems from Monocular Video.* ECCV 2026.

Phenomenon ``dropping_ball``: a football released from rest so that it falls 0.5, 1.0 or 1.5 m
(``drop_height``) onto a carpet, filmed from above, ten takes per height. Each task shows take
01 of one height; its reference is the trajectory extracted from that video (``track.py``), and
the other takes of the same height (nine; eight at 1.5 m, where take 05 repeats take 04) are
other recordings of the same event, which give the noise floor.

The camera's height is estimated per setting from the videos: the dataset states 1.94 m for
all three, but the ball's apparent size at floor contact differs by a factor of 2.5 between
settings, so the camera (or its zoom) was not the same. See ``docs/benchmarks/iris.md``.

Not adapted: ``sliding_cone``, whose published geometry is inconsistent (the ``cone_60``
catheti do not fit its hypotenuse, and the "angle" does not match the incline); and the
pendulum classes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from real2simlab._fs import (
    durable_replace,
    force_remove,
    fsync_file,
    held_path,
    remove_abandoned,
    remove_damaged,
)
from real2simlab._hub import bounded
from real2simlab._project import install_command
from real2simlab.benchmarks.base import BenchmarkAdapter
from real2simlab.benchmarks.iris.track import (
    BallObservations,
    Calibration,
    DropEvents,
    Trajectory,
    ball_trajectory,
    calibrate,
    drop_events,
    observe_ball,
)
from real2simlab.capture import (
    CALIBRATED,
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
from real2simlab.errors import R2SError, SetupError, UnknownOptionError, UsageError
from real2simlab.formats.tracks import Camera, ObjectTrack, Tracks
from real2simlab.media import read_downscaled, write_video
from real2simlab.sandbox.pool import map_in_processes
from real2simlab.task import (
    Delivery,
    OutputRequirements,
    Protocol,
    ScoringConfig,
    TaskBuilder,
    TaskConfig,
)
from real2simlab.task.model import SourceInfo as TaskSource
from real2simlab.vendor import cache_dir, download_failure

REPO = "rasulkhanbayov/IRIS"
REVISION = "0529a5e0cbbf45d43de2ee1ed7cb98e18de03782"
LICENSE = "CC-BY-NC-4.0"
CITATION = (
    "Khanbayov, Barhdadi, Serpedin, Kurban. IRIS: A Real-World Benchmark for Inverse Recovery "
    "and Identification of Physical Dynamic Systems from Monocular Video. ECCV 2026."
)
SETTINGS = {"drop_50": 0.5, "drop_100": 1.0, "drop_150": 1.5}
"""Setting → drop height in metres (``parameters.json``: ``drop_height``): how far the ball falls
before it touches the floor."""
BALL_RADIUS = 0.11
"""The ball's radius, assumed: a size-5 football (68-70 cm around). The videos do not fix it
(their focal length is unknown); it only offsets every height by the same amount."""
TAKES = tuple(range(1, 11))
FRAMES = 72
"""1.2 s at 60 fps from the release."""
DECODE_WIDTH = 960
METRICS = (
    "traj3d_mean",
    "traj3d_rmse",
    "traj3d_final",
    "event_first_contact",
    "event_bounce_count",
)


def release_height(setting: str) -> float:
    """The height of the ball's centre at the release in a setting."""
    return round(SETTINGS[setting] + BALL_RADIUS, 6)


def data_dir() -> Path:
    return cache_dir() / "data" / "IRIS"


def download_video(setting: str, take: int) -> Path:
    """One ``Dropping_ball`` video, downloaded at the pinned revision into the cache."""
    target = data_dir() / "Dropping_ball" / setting / f"{take:02d}.mp4"
    if not remove_damaged([target]):
        return target
    try:
        from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SetupError(
            "the IRIS adapter needs huggingface_hub", fix=install_command("iris")
        ) from error
    with bounded():
        path = hf_hub_download(
            REPO, f"Dropping_ball/{setting}/{take:02d}.mp4", repo_type="dataset",
            revision=REVISION, local_dir=data_dir(), force_download=True,
        )  # fmt: skip
    fsync_file(Path(path))
    return Path(path)


def observation_cache(setting: str, take: int) -> Path:
    return data_dir() / "observations" / f"{setting}_{take:02d}.npz"


NUMBERS = {"fps": "iuf", "w": "iu", "h": "iu", "scale": "iuf"}
"""The scalars of a cached take, by the NumPy kinds they may have (integers for the size)."""
SERIES = ("u", "v", "r")
"""The per-frame series of a cached take, one value per frame each."""


def _read_observations(cache: Path) -> BallObservations:
    """The observations kept in ``cache``, an archive that opens; ``ValueError`` saying why when
    it does not hold what ``observations`` writes there (written by another program, or by
    another version): a field missing, a number that is not one, series that are not three
    series of one length."""
    try:
        with np.load(cache, allow_pickle=False) as data:
            missing = [name for name in (*NUMBERS, *SERIES) if name not in data.files]
            if missing:
                raise ValueError(f"no {', '.join(missing)} in it")
            fields = {name: np.asarray(data[name]) for name in (*NUMBERS, *SERIES)}
    except (OSError, TypeError, AttributeError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(f"not an archive of observations: {error}") from None
    for name, kinds in NUMBERS.items():
        value = fields[name]
        if value.shape != () or value.dtype.kind not in kinds:
            raise ValueError(f"{name} is not a number")
        if not (np.isfinite(value) and value > 0):
            raise ValueError(f"{name} is {float(value)}, not a positive number")
    series = [fields[name] for name in SERIES]
    if (
        any(s.ndim != 1 or s.dtype.kind not in "iuf" for s in series)
        or len(set(map(len, series))) > 1
    ):
        raise ValueError(f"{', '.join(SERIES)} are not series of numbers of one length")
    return BallObservations(
        fps=float(fields["fps"]), width=int(fields["w"]), height=int(fields["h"]),
        scale=float(fields["scale"]), u=fields["u"], v=fields["v"], radius=fields["r"],
    )  # fmt: skip


def observations(setting: str, take: int) -> BallObservations:
    """The per-frame ball circles of one video, cached in the data directory (an entry that
    is damaged, or not what this function writes, is removed with a warning and computed
    again)."""
    cache = observation_cache(setting, take)
    if not remove_damaged([cache], check=_read_observations):
        return _read_observations(cache)
    obs = observe_ball(download_video(setting, take), width=DECODE_WIDTH)
    cache.parent.mkdir(parents=True, exist_ok=True)
    # what a build killed outright left is removed (not the file of one still writing it)
    remove_abandoned(cache.parent, re.escape(f".{cache.stem}."), r"\.partial\.npz")
    partial = held_path(
        lambda part: cache.with_name(f".{cache.stem}.{part}.partial.npz"), directory=False
    )
    try:
        np.savez(
            partial,
            u=obs.u,
            v=obs.v,
            r=obs.radius,
            fps=obs.fps,
            w=obs.width,
            h=obs.height,
            scale=obs.scale,
        )
        durable_replace(partial, cache)  # an interrupted build never leaves a truncated cache
    finally:
        force_remove(partial)  # moved into place, or a failed build's
    return obs


def _observe(item: tuple[str, int]) -> None:
    """``observations`` of one take, in a worker process. A download that failed is raised as
    the ``SetupError`` that explains it (the network, a repository that refuses), because a
    process pool hands an error back without what caused it, which is where that is told."""
    try:
        observations(*item)
    except R2SError:
        raise  # the Lab's own, which says what to do
    except Exception as error:
        explained = download_failure(error)
        if explained is None:
            raise
        raise explained from error


def prefetch_observations(items: Sequence[tuple[str, int]], workers: int | None = None) -> None:
    """Download and track the takes not cached yet, ``workers`` at a time (each decodes a 4K
    video in its own process; default: ``R2S_WORKERS`` or up to 4)."""
    missing = [
        item
        for item in items
        if remove_damaged([observation_cache(*item)], check=_read_observations)
    ]
    if workers is None:
        workers = int(os.environ.get("R2S_WORKERS", 0)) or min(4, os.cpu_count() or 1)
    if workers <= 1 or len(missing) <= 1:
        for item in missing:
            _observe(item)
        return
    map_in_processes(_observe, missing, min(workers, len(missing)))


@dataclass(frozen=True)
class Drop:
    setting: str
    take: int
    observations: BallObservations
    events: DropEvents
    calibration: Calibration
    trajectory: Trajectory


@dataclass(frozen=True)
class Extraction:
    """Every usable drop, and why the others were left out."""

    drops: list[Drop]
    skipped: dict[str, str]
    """``setting/take`` → reason."""


def _same_recording(a: BallObservations, b: BallObservations) -> bool:
    return a.radius.shape == b.radius.shape and bool(
        np.allclose(np.nan_to_num(a.radius), np.nan_to_num(b.radius))
    )


def extract_drops(
    settings: Sequence[str] = tuple(SETTINGS), takes: Sequence[int] = TAKES
) -> Extraction:
    """Every drop's events, camera and trajectory.

    Each take is calibrated on its own (the camera was moved between settings and may have
    moved between takes). A take whose frames are identical to an earlier take's is the same
    recording and is left out.
    """
    skipped: dict[str, str] = {}
    measured: dict[str, list[tuple[int, BallObservations, DropEvents]]] = {}
    prefetch_observations([(setting, take) for setting in settings for take in takes])
    for setting in settings:
        for take in takes:
            key = f"{setting}/{take:02d}"
            obs = observations(setting, take)
            twin = next(
                (t for t, o, _ in measured.get(setting, []) if _same_recording(o, obs)), None
            )
            if twin is not None:
                skipped[key] = f"the same recording as take {twin:02d}"
                continue
            try:
                events = drop_events(obs.radius)
            except ValueError as error:
                skipped[key] = str(error)
                continue
            measured.setdefault(setting, []).append((take, obs, events))
    drops: list[Drop] = []
    for setting, items in measured.items():
        for take, obs, events in items:
            try:
                calibration = calibrate(obs, events, radius=BALL_RADIUS, fall=SETTINGS[setting])
                trajectory = ball_trajectory(
                    obs, events, calibration, radius=BALL_RADIUS, fall=SETTINGS[setting],
                    frames=FRAMES,
                )  # fmt: skip
            except ValueError as error:
                skipped[f"{setting}/{take:02d}"] = str(error)
                continue
            drops.append(Drop(setting, take, obs, events, calibration, trajectory))
    return Extraction(drops=drops, skipped=skipped)


def camera(calibration: Calibration) -> Camera:
    """A take's (estimated) camera at the decoded resolution."""
    return Camera(
        intrinsics=calibration.intrinsics,
        width=calibration.width,
        height=calibration.height,
        extrinsics=calibration.pose,
    )


def tracks_of(drop: Drop) -> Tracks:
    """A drop's trajectory as ``tracks``."""
    trajectory = drop.trajectory
    return Tracks(
        fps=trajectory.fps,
        objects={
            "ball": ObjectTrack(position=trajectory.position.copy(), visible=trajectory.visible)
        },
        camera=camera(drop.calibration),
    )


def task_name(setting: str) -> str:
    return f"drop_{round(SETTINGS[setting] * 100):03d}"


def task_config(setting: str) -> TaskConfig:
    height = SETTINGS[setting]
    return TaskConfig(
        id=f"iris/dropping_ball/{task_name(setting)}",
        version="1.0.1",  # 1.0.1: the camera is not straight above (it is tilted 9 to 18 degrees)
        title=f"IRIS dropping ball, {height:g} m",
        description=(
            f"A real football (object `ball`) is released from rest {height:g} m above a carpet "
            "floor (its lowest point; its centre is one radius higher) and bounces; a camera "
            "films it from above."
        ),
        source=TaskSource(
            dataset=REPO,
            version=REVISION,
            url=f"https://huggingface.co/datasets/{REPO}",
            citation=CITATION,
        ),
        license=LICENSE,
        redistributable=False,
        tags=("iris", "real", "dropping_ball"),
        visibility="calibrated (estimated camera)",
        delivery=Delivery(
            environment="r2s-python",
            output_format="tracks",
            run_timeout_s=120,
            output=OutputRequirements(fps=60.0, num_frames=FRAMES, objects=("ball",)),
        ),
        scoring=ScoringConfig(
            protocol=Protocol.REPRODUCTION, metrics=METRICS, primary="traj3d_mean"
        ),
        metadata={"iris": {"class": "dropping_ball", "setting": setting, "take": 1}},
    )


def visible_capture(drop: Drop, media: Path, task_id: str) -> Capture:
    """The visible observation: the video from the release, at the decoded resolution."""
    obs, trajectory = drop.observations, drop.trajectory
    start = trajectory.release_frame
    video = download_video(drop.setting, drop.take)  # 4K: only the clip, never held whole
    scaled = read_downscaled(video, obs.width, obs.height, start=start, stop=start + FRAMES)
    write_video(scaled, media / "cam" / "rgb.mp4", fps=obs.fps)
    cam = camera(drop.calibration)
    estimated = Provenance.ESTIMATED
    calibration = "from the ball's apparent size at rest and at floor contact, and the drop height"
    start_xy = trajectory.position[0]
    drop_height = SETTINGS[drop.setting]
    return Capture(
        capture_id=task_id,
        source=SourceInfo(
            dataset=REPO,
            version=REVISION[:12],
            license=LICENSE,
            url=f"https://huggingface.co/datasets/{REPO}",
        ),
        world=WorldFrame(
            metric_scale=Sourced(
                value=True, provenance=Provenance.MEASURED, source="the drop height (dataset)"
            )
        ),
        views={
            "cam": View(
                rgb=VideoStream(
                    path="cam/rgb.mp4",
                    fps=obs.fps,
                    num_frames=FRAMES,
                    resolution=(obs.width, obs.height),
                ),
                intrinsics=Sourced(
                    value=cam.intrinsics.tolist(), provenance=estimated, source=calibration
                ),
                extrinsics=Sourced(
                    value=cam.extrinsics.tolist(), provenance=estimated, source=calibration
                ),
            )
        },
        scene=SceneInfo(
            objects=(
                SceneObject(
                    name="ball",
                    role="dynamic",
                    shape="sphere (football)",
                    size=Sourced(
                        value=(2 * BALL_RADIUS, 2 * BALL_RADIUS, 2 * BALL_RADIUS),
                        provenance=Provenance.DEFAULT,
                        source="assumed: a size-5 football",
                    ),
                    color="orange and black",
                    initial_pose=Sourced(
                        value=[
                            float(start_xy[0]),
                            float(start_xy[1]),
                            release_height(drop.setting),
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        provenance=Provenance.MEASURED,
                        source="drop height (dataset) plus the ball's radius; x, y estimated",
                    ),
                ),
                SceneObject(name="floor", role="static", shape="carpeted floor, plane z = 0"),
            ),
            environment="An indoor carpet floor; the camera looks straight down at it.",
        ),
        text=TextInfo(
            task="Reproduce the drop: the ball's fall and its bounces.",
            scene_description=(
                f"A football is released from rest with its lowest point {drop_height:g} m above "
                "the floor, and bounces. The video starts at the release."
            ),
        ),
    )


def extraction_report(extraction: Extraction, setting: str | None = None) -> dict[str, Any]:
    """What the references were extracted from: every take's events and camera, and skips;
    only ``setting``'s, if given (each task holds its own setting's, so that a task does not
    depend on which other settings were built with it)."""
    kept = [d for d in extraction.drops if setting is None or d.setting == setting]
    skipped = {
        key: reason
        for key, reason in extraction.skipped.items()
        if setting is None or key.split("/", 1)[0] == setting
    }
    settings: dict[str, Any] = {}
    for name in SETTINGS:
        drops = [d for d in kept if d.setting == name]
        if drops:
            settings[name] = {
                "drop_height_m": SETTINGS[name],
                "gravity_check_median": float(
                    np.median([d.trajectory.gravity_check for d in drops])
                ),
                "camera_height_median_m": float(
                    np.median([d.calibration.camera_height for d in drops])
                ),
                "tilt_median_deg": float(np.median([d.calibration.tilt_deg for d in drops])),
            }
    return {
        "ball_radius_m": BALL_RADIUS,
        "settings": settings,
        "drops": [
            {
                "setting": d.setting,
                "take": d.take,
                "release_frame": d.trajectory.release_frame,
                "contact_frame": d.trajectory.contact_frame,
                "r_release": d.events.r_release,
                "r_contact": d.events.r_contact,
                "focal_px": d.calibration.focal,
                "camera_height_m": d.calibration.camera_height,
                "tilt_deg": d.calibration.tilt_deg,
                "gravity_check": d.trajectory.gravity_check,
            }
            for d in kept
        ],
        "skipped": skipped,
    }


def build_tasks(
    out_dir: Path, *, settings: Sequence[str] = tuple(SETTINGS), overwrite: bool = False
) -> list[Path]:
    """One task per setting: take 01 is the observation and the reference, the other takes of
    the same setting are other recordings of the event. Only the selected settings' takes are
    downloaded and tracked (each take is calibrated on its own)."""
    unknown = sorted(set(settings) - set(SETTINGS))
    if unknown:
        raise UsageError(f"unknown IRIS settings {unknown}; known: {list(SETTINGS)}")
    extraction = extract_drops(settings=tuple(settings))
    by_setting: dict[str, dict[int, Drop]] = {}
    for drop in extraction.drops:
        by_setting.setdefault(drop.setting, {})[drop.take] = drop
    for setting in settings:
        if 1 not in by_setting.get(setting, {}):
            raise SetupError(
                f"IRIS {setting} take 01 could not be tracked; it is the task's reference"
            )
    paths: list[Path] = []
    for setting in settings:
        report = json.dumps(extraction_report(extraction, setting), indent=2) + "\n"
        config = task_config(setting)
        root = Path(out_dir) / "dropping_ball" / task_name(setting)
        builder = TaskBuilder(root, config, overwrite=overwrite)
        with tempfile.TemporaryDirectory(prefix="r2s-iris-") as tmp:
            capture = visible_capture(by_setting[setting][1], Path(tmp), config.id)
            builder.capture(capture, Path(tmp), visibility=CALIBRATED)
        takes = by_setting[setting]
        builder.reference(tracks=tracks_of(takes[1]))
        for index, take in enumerate(sorted(t for t in takes if t != 1), start=1):
            builder.repeat(index, tracks=tracks_of(takes[take]))
        builder.hidden_file("extraction.json", text=report)
        paths.append(builder.finish().root)
    return paths


class IrisBenchmark(BenchmarkAdapter):
    name: ClassVar[str] = "iris"
    summary: ClassVar[str] = (
        "IRIS (real videos): a ball dropped from 0.5, 1.0 and 1.5 m; reproduce the drop; the "
        "other takes of the same height (nine, or eight at 1.5 m) give the noise floor."
    )
    license: ClassVar[str] = f"{LICENSE} (downloaded at build time, not redistributed)"
    source: ClassVar[str] = f"https://huggingface.co/datasets/{REPO}"
    protocols: ClassVar[tuple[str, ...]] = ("reproduction",)
    status: ClassVar[str] = "partial"

    def task_names(self) -> list[str]:
        return [f"iris/dropping_ball/{task_name(s)}" for s in SETTINGS]

    def build(
        self,
        out_dir: Path,
        *,
        select: Sequence[str] | None = None,
        overwrite: bool = False,
        **options: Any,
    ) -> list[Path]:
        """``select`` names settings (``drop_50``, ``drop_100``, ``drop_150``)."""
        if options:
            raise UnknownOptionError.from_names(
                f"benchmark {self.name!r}", options, self.options_taken()
            )
        settings = list(dict.fromkeys(select)) if select else list(SETTINGS)  # each once
        return build_tasks(Path(out_dir), settings=settings, overwrite=overwrite)


def clear_cache() -> None:
    shutil.rmtree(data_dir() / "observations", ignore_errors=True)
