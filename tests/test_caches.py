"""Caches that a crash left damaged (empty, truncated, unparseable) are fetched again, not
trusted forever because a file exists."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from real2simlab._fs import DamagedCacheWarning, cache_problem, remove_damaged
from real2simlab.media import write_video


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


def _video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_video(np.zeros((4, 32, 48, 3), np.uint8), path, fps=10.0)
    return path


def test_cache_problem_recognises_damage(tmp_path: Path) -> None:
    good = _video(tmp_path / "good.mp4")
    truncated = tmp_path / "truncated.mp4"
    truncated.write_bytes(good.read_bytes()[: good.stat().st_size // 3])
    (tmp_path / "empty.json").write_bytes(b"")
    (tmp_path / "half.json").write_text('[[1, "center", "ball')
    np.savez(tmp_path / "arrays.npz", a=np.arange(1000))
    (tmp_path / "cut.npz").write_bytes((tmp_path / "arrays.npz").read_bytes()[:300])
    assert cache_problem(good) is None and cache_problem(tmp_path / "arrays.npz") is None
    assert cache_problem(tmp_path / "missing.mp4") == "missing"
    assert cache_problem(tmp_path / "empty.json") == "empty"
    for damaged in ("truncated.mp4", "half.json", "cut.npz"):
        assert cache_problem(tmp_path / damaged) not in (None, "missing", "empty"), damaged
    with pytest.warns(DamagedCacheWarning):
        needed = remove_damaged([good, tmp_path / "half.json", tmp_path / "missing.mp4"])
    assert needed == [tmp_path / "half.json", tmp_path / "missing.mp4"]
    assert good.is_file() and not (tmp_path / "half.json").exists()


@pytest.mark.parametrize("damage", [b"", b'[[7, "center", "ball-and-bl'])
def test_physics_iq_index_is_rebuilt_when_damaged(
    cache: Path, monkeypatch: pytest.MonkeyPatch, damage: bytes
) -> None:
    from real2simlab.benchmarks import physicsiq

    listing = [physicsiq.Case("ball-and-block-fall", "center", 7).file("testing", 1)]
    calls: list[str] = []

    def fake_list(release: str) -> list[str]:
        calls.append(release)
        return listing

    monkeypatch.setattr(physicsiq, "_list_testing_files", fake_list)
    index = physicsiq.data_dir("original") / "index.json"
    index.parent.mkdir(parents=True)
    index.write_bytes(damage)  # what a crash while writing it leaves
    with pytest.warns(DamagedCacheWarning):
        cases = physicsiq.case_index("original")
    assert list(cases) == ["ball-and-block-fall/center"] and calls == ["original"]
    assert json.loads(index.read_text()) == [[7, "center", "ball-and-block-fall"]]
    assert physicsiq.case_index("original") == cases and calls == ["original"]  # reused


def test_physics_iq_download_checks_its_pin(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from real2simlab.benchmarks import physicsiq

    content = b"the clip's bytes" * 100
    digest = base64.b64encode(hashlib.md5(content).digest()).decode()
    fetched: list[str] = []

    def fake_json(url: str) -> dict[str, Any]:
        return {"generation": "1", "md5Hash": digest}

    def fake_open(url: str, timeout: float = 0) -> io.BytesIO:
        fetched.append(url)
        return io.BytesIO(content)

    monkeypatch.setattr(physicsiq, "_gcs_json", fake_json)
    monkeypatch.setattr(physicsiq.urllib.request, "urlopen", fake_open)
    case = physicsiq.Case("ball-and-block-fall", "center", 7)
    path = case.file("testing", 1)
    target, pin = physicsiq.download(case, path)
    assert target.read_bytes() == content and f"md5={digest}" in pin and len(fetched) == 1
    assert physicsiq.download(case, path) == (target, pin) and len(fetched) == 1  # reused
    target.write_bytes(content[:50])  # truncated under its pin
    with pytest.warns(DamagedCacheWarning, match="MD5"):
        assert physicsiq.download(case, path) == (target, pin)
    assert target.read_bytes() == content and len(fetched) == 2


def test_iris_observations_are_recomputed_when_damaged(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from real2simlab.benchmarks import iris
    from real2simlab.benchmarks.iris.track import BallObservations

    observed: list[str] = []

    def fake_observe(video: Path, width: int) -> BallObservations:
        observed.append(video.name)
        series = np.arange(5, dtype=np.float64)
        return BallObservations(30.0, 64, 48, 0.5, series, series, series + 10)

    monkeypatch.setattr(iris, "download_video", lambda setting, take: Path(f"{setting}.mp4"))
    monkeypatch.setattr(iris, "observe_ball", fake_observe)
    first = iris.observations("drop_50", 1)
    cached = iris.observation_cache("drop_50", 1)
    cached.write_bytes(cached.read_bytes()[:100])
    with pytest.warns(DamagedCacheWarning):
        again = iris.observations("drop_50", 1)
    assert observed == ["drop_50.mp4", "drop_50.mp4"]
    assert np.array_equal(first.radius, again.radius) and cache_problem(cached) is None


def test_an_intact_npy_file_is_not_damaged(tmp_path: Path) -> None:
    """``np.load`` gives an array for a ``.npy`` file (an archive only for ``.npz``): an intact
    ``.npy`` is reused, not removed as damaged; a truncated one is damaged, and so is a file
    that holds an array where its name promises an archive, or the other way round."""
    good = tmp_path / "good.npy"
    np.save(good, np.arange(1000))
    (tmp_path / "cut.npy").write_bytes(good.read_bytes()[:300])
    (tmp_path / "array.npz").write_bytes(good.read_bytes())
    np.savez(tmp_path / "archive.npz", a=np.arange(10))
    (tmp_path / "archive.npy").write_bytes((tmp_path / "archive.npz").read_bytes())
    assert cache_problem(good) is None
    for damaged in ("cut.npy", "array.npz", "archive.npy"):
        assert cache_problem(tmp_path / damaged) not in (None, "missing", "empty"), damaged
    with warnings.catch_warnings():
        warnings.simplefilter("error", DamagedCacheWarning)
        assert remove_damaged([good]) == []
    assert good.is_file()


IRIS_FIELDS: dict[str, Any] = {
    "fps": 30.0, "w": 64, "h": 48, "scale": 0.5,
    "u": np.arange(5.0), "v": np.arange(5.0), "r": np.arange(5.0) + 10,
}  # fmt: skip


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param({"fps": 30.0}, id="arrays missing"),
        pytest.param({**IRIS_FIELDS, "fps": np.arange(3.0)}, id="fps not a number"),
        pytest.param({**IRIS_FIELDS, "w": "wide"}, id="width not a number"),
        pytest.param({**IRIS_FIELDS, "r": np.ones((5, 2))}, id="radii not a series"),
        pytest.param({**IRIS_FIELDS, "u": np.arange(4.0)}, id="series of other lengths"),
        pytest.param(None, id="an array, not an archive"),
    ],
)
def test_iris_observations_with_damaged_fields_are_recomputed(
    cache: Path, monkeypatch: pytest.MonkeyPatch, damage: dict[str, Any] | None
) -> None:
    """A cached archive that opens but lacks a field, or holds one of another kind or shape
    (written by another program, or another version), is damaged like a truncated one: it is
    computed again with a warning, never a traceback or observations of the wrong shape."""
    from real2simlab.benchmarks import iris
    from real2simlab.benchmarks.iris.track import BallObservations

    observed: list[str] = []

    def fake_observe(video: Path, width: int) -> BallObservations:
        observed.append(video.name)
        series = np.arange(5, dtype=np.float64)
        return BallObservations(30.0, 64, 48, 0.5, series, series, series + 10)

    monkeypatch.setattr(iris, "download_video", lambda setting, take: Path(f"{setting}.mp4"))
    monkeypatch.setattr(iris, "observe_ball", fake_observe)
    cached = iris.observation_cache("drop_50", 1)
    cached.parent.mkdir(parents=True)
    if damage is None:
        with cached.open("wb") as stream:
            np.save(stream, np.arange(5.0))
    else:
        np.savez(cached, **damage)
    with pytest.warns(DamagedCacheWarning):
        obs = iris.observations("drop_50", 1)
    assert observed == ["drop_50.mp4"]
    assert obs.width == 64 and np.array_equal(obs.radius, np.arange(5.0) + 10)
    again = iris.observations("drop_50", 1)  # the rebuilt entry is reused
    assert observed == ["drop_50.mp4"] and np.array_equal(again.u, obs.u)


@pytest.mark.parametrize("damage", ["an array, not an archive", "fps not a number"])
def test_cached_thumbnails_with_damaged_fields_are_computed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """The overlap check's thumbnails: an entry that opens but is not what was written there is
    damaged, computed again with a warning, never a traceback."""
    from real2simlab.task import overlap

    thumbs = np.zeros((3, overlap.SIZE, overlap.SIZE, 3), np.uint8)
    computed: list[Path] = []

    def fake_thumbnails(path: Path) -> tuple[float, Any]:
        computed.append(path)
        return 30.0, thumbs

    monkeypatch.setattr(overlap, "thumbnails", fake_thumbnails)
    stored = tmp_path / f"abc-v{overlap.VERSION}.npz"
    if damage == "an array, not an archive":
        with stored.open("wb") as stream:
            np.save(stream, thumbs)
    else:
        np.savez(stored, fps=np.arange(3.0), thumbs=thumbs)
    with pytest.warns(DamagedCacheWarning):
        fps, again = overlap._cached_thumbnails(tmp_path / "clip.mp4", "abc", tmp_path)
    assert fps == 30.0 and np.array_equal(again, thumbs) and len(computed) == 1


def test_a_vendored_checkout_changed_since_it_was_checked_is_not_handed_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkout of pinned code edited after this process checked it (an editor, a sync
    tool, a `git checkout` in the cache) is checked again on every use: the scorers that ask
    for it next get the pinned commit's code, not the edit."""
    import subprocess

    from real2simlab.vendor import VendoredRepo

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "score.py").write_text("score = 1\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=a", "-c",
                    "user.email=a@example.com", "commit", "-qm", "first"], check=True)  # fmt: skip
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()  # fmt: skip
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    repo = VendoredRepo("audit", f"file://{source}", commit, "MIT")
    checkout = repo.ensure()
    (checkout / "score.py").write_text("score = 0\n")  # in the same process
    with pytest.warns(DamagedCacheWarning, match="differ from the commit"):
        again = repo.ensure()
    assert (again / "score.py").read_text() == "score = 1\n"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an intact checkout is reused silently
        assert repo.ensure() == again


class FakeHub:
    """``hf_hub_download`` and ``snapshot_download`` writing local files, counted."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.calls: list[str] = []

    def hf_hub_download(self, repo: str, name: str, *, local_dir: str, **_: Any) -> str:
        self.calls.append(name)
        target = Path(local_dir) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[name])
        return str(target)


def test_video2world_manifest_is_fetched_again_when_damaged(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from real2simlab.benchmarks import video2world

    hub = FakeHub({"manifest.json": json.dumps({"samples": []}).encode()})
    monkeypatch.setattr(video2world, "_hub", lambda: hub)
    assert video2world.manifest() == {"samples": []} and hub.calls == ["manifest.json"]
    (video2world.data_dir() / "manifest.json").write_text('{"samp')
    with pytest.warns(DamagedCacheWarning):
        assert video2world.manifest() == {"samples": []}
    assert hub.calls == ["manifest.json", "manifest.json"]


def test_fourdcodebench_cases_are_replaced_whole(
    cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A case is unpacked beside the cache and moved in only when complete; a damaged one is
    fetched again."""
    from real2simlab.benchmarks import fourdcodebench as fdcb

    clip = _video(tmp_path / "clip.mp4").read_bytes()
    hub = FakeHub({"videos/c1.mp4": clip, "worlds/c1.h5": b"packed world"})
    monkeypatch.setattr(fdcb, "_hub", lambda: hub)
    unpacked: list[Path] = []

    def fake_unpack(packed: Path, world: Path) -> None:
        unpacked.append(world)
        if len(unpacked) == 1:
            raise RuntimeError("killed while unpacking")
        world.mkdir(parents=True)
        (world / "camera.json").write_text("{}")

    monkeypatch.setattr(fdcb, "_unpack", fake_unpack)
    with pytest.raises(RuntimeError, match="killed"):
        fdcb.download_synthetic_case("c1")
    parent = fdcb.data_dir() / "synthetic"
    assert list(parent.iterdir()) == []  # nothing half-written under the case's name
    video, world = fdcb.download_synthetic_case("c1")
    assert video.read_bytes() == clip and (world / "camera.json").is_file()
    assert fdcb.download_synthetic_case("c1") == (video, world) and len(unpacked) == 2
    video.write_bytes(clip[:40])
    with pytest.warns(DamagedCacheWarning):
        fdcb.download_synthetic_case("c1")
    assert video.read_bytes() == clip and len(unpacked) == 3
