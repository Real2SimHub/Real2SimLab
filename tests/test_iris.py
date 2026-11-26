"""IRIS: trajectory extraction checked on a rendered drop with known physics; task building."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import numpy as np
import pytest

from real2simlab.benchmarks.iris.track import (
    ball_trajectory,
    calibrate,
    drop_events,
    fit_circle,
    observe_ball,
)
from real2simlab.benchmarks.synthetic.physics import render
from real2simlab.benchmarks.synthetic.physics.common import look_at
from real2simlab.media import write_video

CAMERA_HEIGHT = 1.94
RADIUS = 0.1
HELD = 20
FPS = 60.0
DROP = 0.9
EYE = np.array([0.0, -0.45, CAMERA_HEIGHT])
TARGET = np.array([0.02, 0.05, 0.0])


def heights(h0: float, frames: int, restitution: float = 0.6) -> np.ndarray:
    """Centre height of a ball held for HELD frames, then dropped from rest (event-exact)."""
    t = (np.arange(frames) - HELD) / FPS
    z = np.full(frames, h0)
    start, z0, v0 = 0.0, h0, 0.0
    while True:
        s = (v0 + np.sqrt(v0 * v0 + 2 * 9.81 * (z0 - RADIUS))) / 9.81
        mask = (t >= start) & (t < start + s)
        z[mask] = z0 + v0 * (t[mask] - start) - 4.905 * (t[mask] - start) ** 2
        start += s
        v0, z0 = restitution * 9.81 * s if v0 == 0 else restitution * abs(v0 - 9.81 * s), RADIUS
        if v0 < 0.3 or start > t[-1]:
            z[t >= start] = RADIUS
            return z


@pytest.fixture(scope="module")
def drop_video(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, np.ndarray]:
    """A ball falling DROP metres onto the floor, filmed by a camera tilted about 14 degrees."""
    camera = look_at(EYE, TARGET, fov_deg=60.0, width=640, height=360)
    z = heights(DROP + RADIUS, 120)
    frames = render.render_frames(
        camera,
        [],
        [[render.Sphere(np.array([0.02, -0.01, h]), RADIUS, (0.95, 0.45, 0.05))] for h in z],
    )
    path = write_video(frames, tmp_path_factory.mktemp("iris") / "drop.mp4", fps=FPS)
    return path, z


def test_circle_fit() -> None:
    angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    points = np.stack([3 + 2 * np.cos(angles), -1 + 2 * np.sin(angles)], axis=1)
    assert np.allclose(fit_circle(points), (3, -1, 2))


def test_drop_events_ignore_a_flickering_hold() -> None:
    """A hand covering the held ball for a few frames is not a release (IRIS drop_150/06)."""
    held = np.array([173.1, 184.0, 181.4, 184.4, 181.7, 182.4, 172.3, 172.4, 172.3, 185.4] * 3)
    fall = 180.0 / (1 + 0.5 * (np.arange(30) / 30) ** 2 * 3)
    bounce = fall[-1] * np.array([1.02, 1.05, 1.08, 1.1, 1.11, 1.1, 1.08, 1.05, 1.02, 1.0])
    radius = np.concatenate([held, fall, bounce])
    radius[5] = np.nan
    events = drop_events(radius)
    assert len(held) <= events.moving < len(held) + 8
    assert events.contact == len(held) + len(fall) - 1
    assert 178 < events.r_release < 183


def test_extraction_recovers_a_tilted_drop(drop_video: tuple[Path, np.ndarray]) -> None:
    path, truth = drop_video
    obs = observe_ball(path, width=640)
    assert np.isfinite(obs.radius).all()
    events = drop_events(obs.radius)
    calibration = calibrate(obs, events, radius=RADIUS, fall=DROP)
    forward = TARGET - EYE
    tilt = np.degrees(np.arccos(-forward[2] / np.linalg.norm(forward)))
    assert abs(calibration.tilt_deg - tilt) < 1.5
    assert abs(calibration.camera_height - CAMERA_HEIGHT) < 0.04
    trajectory = ball_trajectory(obs, events, calibration, radius=RADIUS, fall=DROP, frames=60)
    assert abs(trajectory.release_frame - HELD) <= 1
    z = trajectory.position[:, 2]
    expected = truth[trajectory.release_frame : trajectory.release_frame + 60]
    assert np.nanmean(np.abs(z - expected)) < 0.02
    assert abs(trajectory.gravity_check - 9.81) < 0.5
    horizontal = trajectory.position[:, :2]
    assert np.nanmax(np.abs(horizontal - np.nanmedian(horizontal, axis=0))) < 0.03


IRIS_CACHE = (
    Path(os.environ.get("R2S_CACHE_DIR", Path.home() / ".cache" / "real2simlab")) / "data" / "IRIS"
)


@pytest.mark.slow
@pytest.mark.skipif(
    not (IRIS_CACHE / "observations" / "drop_150_10.npz").is_file(), reason="IRIS not downloaded"
)
def test_iris_tasks(tmp_path: Path) -> None:
    from real2simlab.benchmarks.iris import build_tasks
    from real2simlab.task import load_task, validate_task

    (root,) = build_tasks(tmp_path, settings=["drop_100"])
    report = validate_task(root)
    assert report.ok and not report.warnings, report.issues
    task = load_task(root)
    assert task.id == "iris/dropping_ball/drop_100" and root.name == "drop_100"
    assert task.config.redistributable is False and task.config.license == "CC-BY-NC-4.0"
    assert task.config.scoring.protocol.value == "reproduction"
    assert task.reference_dir().is_dir() and len(task.repeat_dirs()) >= 8


def test_prefetch_tracks_missing_takes_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from real2simlab.benchmarks import iris
    from real2simlab.benchmarks.iris.track import BallObservations

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    calls = tmp_path / "calls"
    calls.mkdir()

    def fake_download(setting: str, take: int) -> Path:
        return tmp_path / f"{setting}_{take}.mp4"

    def fake_observe(video: Path, width: int) -> BallObservations:
        (calls / video.stem).touch()  # visible across worker processes
        series = np.arange(5, dtype=np.float64)
        return BallObservations(30.0, 64, 48, 0.5, series, series, series + 10)

    monkeypatch.setattr(iris, "download_video", fake_download)
    monkeypatch.setattr(iris, "observe_ball", fake_observe)
    items = [("drop_50", 1), ("drop_50", 2), ("drop_100", 1)]
    # Spawned workers (macOS, Windows) re-import the module and would not see the patches.
    workers = 2 if multiprocessing.get_start_method() == "fork" else 1
    iris.prefetch_observations(items, workers=workers)
    assert sorted(p.name for p in calls.iterdir()) == ["drop_100_1", "drop_50_1", "drop_50_2"]
    cached = sorted(
        p.name for p in (tmp_path / "cache" / "data" / "IRIS" / "observations").iterdir()
    )
    assert cached == ["drop_100_01.npz", "drop_50_01.npz", "drop_50_02.npz"]  # no partial files
    for path in calls.iterdir():
        path.unlink()
    iris.prefetch_observations(items, workers=workers)
    assert not any(calls.iterdir())
    assert iris.observations("drop_50", 2).radius[0] == 10


@pytest.mark.parametrize("workers", [1, 2])
def test_a_download_that_fails_in_a_worker_says_why(
    workers: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A take downloaded in a worker process fails for lack of network: the error that reaches
    the command still says it was the network, with its fix (a process pool hands back the
    error alone, without what caused it, which is where the network failure is)."""
    from real2simlab.benchmarks import iris
    from real2simlab.errors import SetupError

    if workers > 1 and multiprocessing.get_start_method() != "fork":
        pytest.skip("spawned workers re-import the module and would not see the patch")
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))

    def offline(setting: str, take: int) -> Path:
        try:
            raise ConnectionRefusedError(111, "Connection refused")
        except ConnectionRefusedError as error:  # what huggingface_hub 2 raises around it
            raise ValueError("Force download failed due to the above error.") from error

    monkeypatch.setattr(iris, "download_video", offline)
    with pytest.raises(SetupError) as raised:
        iris.prefetch_observations([("drop_50", 1), ("drop_50", 2)], workers=workers)
    assert "a download failed: [Errno 111] Connection refused" in str(raised.value)
    assert "fix: check this machine's network connection" in str(raised.value)


def test_a_bug_in_a_worker_is_not_told_as_the_download_it_was_handling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistake in the code (a ``KeyError``) raised while a failed download is being handled
    reaches the command as the bug it is, not as that download's error and its fix."""
    from real2simlab.benchmarks import iris

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))

    def broken(setting: str, take: int) -> Path:
        try:
            raise ConnectionRefusedError(111, "Connection refused")
        except ConnectionRefusedError:
            raise KeyError("tracks")  # noqa: B904 (no `from`: the failed download is its context)

    monkeypatch.setattr(iris, "download_video", broken)
    with pytest.raises(KeyError):
        iris.prefetch_observations([("drop_50", 1), ("drop_50", 2)], workers=1)


def test_selecting_a_setting_fetches_and_reports_only_that_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``select = ["drop_50"]`` downloads and tracks drop_50's ten takes, not the other twenty;
    and each task's extraction report holds its own setting only."""
    from types import SimpleNamespace

    from real2simlab.benchmarks import iris

    requested: list[tuple[str, int]] = []

    class Fetched(Exception):
        pass

    def fake_prefetch(items: list[tuple[str, int]], workers: int | None = None) -> None:
        requested.extend(items)
        raise Fetched

    monkeypatch.setattr(iris, "prefetch_observations", fake_prefetch)
    with pytest.raises(Fetched):
        iris.IrisBenchmark().build(tmp_path / "tasks", select=["drop_50"])
    assert sorted({setting for setting, _ in requested}) == ["drop_50"] and len(requested) == 10
    with pytest.raises(ValueError, match="unknown IRIS settings"):
        iris.build_tasks(tmp_path / "tasks", settings=["drop_70"])

    def drop(setting: str, take: int) -> SimpleNamespace:
        return SimpleNamespace(
            setting=setting,
            take=take,
            events=SimpleNamespace(r_release=10.0, r_contact=20.0),
            calibration=SimpleNamespace(focal=900.0, camera_height=1.9, tilt_deg=14.0),
            trajectory=SimpleNamespace(release_frame=3, contact_frame=30, gravity_check=9.8),
        )

    extraction = iris.Extraction(
        drops=[drop("drop_50", 1), drop("drop_100", 1)],  # type: ignore[list-item]
        skipped={"drop_50/04": "the same recording as take 03", "drop_100/02": "no release"},
    )
    report = iris.extraction_report(extraction, "drop_50")
    assert list(report["settings"]) == ["drop_50"]
    assert [d["setting"] for d in report["drops"]] == ["drop_50"]
    assert report["skipped"] == {"drop_50/04": "the same recording as take 03"}
    assert len(iris.extraction_report(extraction)["drops"]) == 2
