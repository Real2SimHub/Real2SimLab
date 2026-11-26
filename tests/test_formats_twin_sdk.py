"""Output formats, twin packages and the single-file SDK."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from real2simlab.errors import UnknownNameError
from real2simlab.formats import Requirements, get_format, read_tracks, tracks_problems, write_tracks
from real2simlab.formats.base import InvalidOutputError, output_problems
from real2simlab.task.model import ProgramRules
from real2simlab.twin import (
    EnvironmentRef,
    InvalidTwinError,
    Origin,
    TwinManifest,
    fingerprint,
    load_twin,
    save_twin,
    sha256_file,
)
from real2simlab.twin.sdk import r2s_twin, sdk_path

from .conftest import falling, make_tracks


def manifest(**changes: object) -> TwinManifest:
    data = {
        "twin_id": "test/twin",
        "entrypoint": ["python", "./run.py"],
        "environment": {"image": "r2s-python"},
        "output_format": "tracks",
        "origin": {"method": "test", "task_id": "t", "created_at": datetime.now(UTC).isoformat()},
        **changes,
    }
    return TwinManifest.model_validate(data)


def test_tracks_round_trip_and_validate(tmp_path: Path) -> None:
    tracks = make_tracks({"ball": falling()})
    write_tracks(tracks, tmp_path)
    assert tracks_problems(tmp_path, Requirements(fps=30, num_frames=60, objects=("ball",))) == []
    loaded = read_tracks(tmp_path)
    assert loaded.num_frames == 60 and np.allclose(loaded.positions("ball"), falling())
    problems = tracks_problems(tmp_path, Requirements(fps=25, num_frames=61, objects=("cube",)))
    assert len(problems) == 3
    fmt = get_format("tracks")
    assert fmt.provides_tracks and "r2s-tracks/1.0" in fmt.documentation()
    with pytest.raises(UnknownNameError):
        get_format("nope")


def test_tracks_rejections(tmp_path: Path) -> None:
    write_tracks(make_tracks({"ball": falling()}), tmp_path)
    position = falling()
    position[3] = np.nan
    np.savez(tmp_path / "tracks.npz", **{"ball.position": position, "extra": np.zeros(2)})
    problems = tracks_problems(tmp_path, Requirements())
    assert any("non-finite" in p for p in problems) and any("unexpected" in p for p in problems)
    visible = np.ones(60, dtype=bool)
    visible[3] = False
    np.savez(tmp_path / "tracks.npz", **{"ball.position": position, "ball.visible": visible})
    assert tracks_problems(tmp_path, Requirements()) == []
    camera = json.loads((tmp_path / "camera.json").read_text())
    camera["extrinsics"][0][0] = 3.0
    (tmp_path / "camera.json").write_text(json.dumps(camera))
    assert any("rotation" in p for p in tracks_problems(tmp_path, Requirements()))
    (tmp_path / "meta.json").write_text(
        json.dumps({"spec": "x", "fps": 0, "num_frames": 0, "objects": []})
    )
    assert len(tracks_problems(tmp_path, Requirements())) >= 4
    (tmp_path / "link").symlink_to(tmp_path / "meta.json")
    fmt = get_format("tracks")
    assert output_problems(fmt, tmp_path) == ["link: symbolic links are not allowed"]
    with pytest.raises(InvalidOutputError):
        fmt.check(tmp_path)
    assert output_problems(fmt, tmp_path / "missing") == ["the output directory does not exist"]


def test_tracks_resampling_and_projection() -> None:
    tracks = make_tracks({"ball": falling()})
    times = np.arange(0, 1.9, 1 / 15)
    resampled = tracks.at_times(times)
    assert resampled.num_frames == len(times)
    assert np.allclose(resampled.positions("ball")[1], falling()[2])
    assert tracks.camera is not None
    uv = tracks.camera.project(np.array([[0.0, 0.0, 0.5], [0.0, -3.0, 0.5]]))
    assert np.allclose(uv[0], [160, 120]) and np.isnan(uv[1]).all()
    window = tracks.window(10, 20)
    assert window.num_frames == 10


def test_video_format(tmp_path: Path) -> None:
    from real2simlab.media import write_video

    fmt = get_format("video")
    assert fmt.validate(tmp_path) == ["render.mp4: file is required"]
    write_video(np.zeros((5, 32, 32, 3), dtype=np.uint8), tmp_path / "render.mp4", fps=10)
    assert fmt.validate(tmp_path, Requirements(fps=10, num_frames=5, width=32, height=32)) == []
    assert len(fmt.validate(tmp_path, Requirements(fps=12, num_frames=6, width=8))) == 3
    assert fmt.to_tracks(tmp_path) is None


def test_manifest_rules() -> None:
    assert manifest().program_files == ("run.py",)
    with pytest.raises(ValidationError, match="inside program"):
        manifest(entrypoint=["python", "./../x.py"])
    with pytest.raises(ValidationError, match="Extra inputs"):
        manifest(notes="unknown fields are rejected")
    with pytest.raises(ValidationError, match="absolute"):
        manifest(runtime={"program_path": "relative"})


def program(tmp_path: Path, files: dict[str, bytes | str]) -> Path:
    root = tmp_path / "program"
    root.mkdir(parents=True)
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)
    return root


def test_save_and_load_twin(tmp_path: Path) -> None:
    source = program(tmp_path, {"run.py": "print(1)", "lib/data.npy": b"\x00" * 10})
    twin = save_twin(manifest(), source, tmp_path / "twin")
    assert (
        twin.fingerprint == fingerprint(tmp_path / "twin")
        and twin.current_fingerprint() == twin.fingerprint
    )
    assert load_twin(tmp_path / "twin").program_fingerprint == fingerprint(source)
    with pytest.raises(FileExistsError):
        save_twin(manifest(), source, tmp_path / "twin")
    (tmp_path / "twin" / "stray.txt").write_text("x")
    with pytest.raises(InvalidTwinError, match="unexpected entry"):
        load_twin(tmp_path / "twin")


def test_program_rules(tmp_path: Path) -> None:
    source = program(tmp_path, {"run.py": "x", "replay.mp4": b"0", "big.bin": b"0" * 2_000_000})
    (source / "link.py").symlink_to(source / "run.py")
    with pytest.raises(InvalidTwinError) as error:
        save_twin(manifest(), source, tmp_path / "twin", rules=ProgramRules(max_size_mb=1))
    text = str(error.value)
    assert "video files" in text and "symbolic links" in text and "exceeds" in text
    assert not (tmp_path / "twin").exists()
    with pytest.raises(InvalidTwinError, match="not a file"):
        save_twin(
            manifest(entrypoint=["python", "./main.py"]),
            program(tmp_path / "b", {"run.py": ""}),
            tmp_path / "t2",
        )


def test_assets_are_verified(tmp_path: Path) -> None:
    source = program(tmp_path, {"run.py": ""})
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "w.bin").write_bytes(b"weights")
    digest = sha256_file(assets / "w.bin")
    good = manifest(assets=[{"path": "w.bin", "sha256": digest, "size": 7}])
    save_twin(good, source, tmp_path / "ok", assets_src=assets)
    bad = manifest(assets=[{"path": "w.bin", "sha256": "0" * 64, "size": 7}])
    with pytest.raises(InvalidTwinError, match="does not match"):
        save_twin(bad, source, tmp_path / "bad", assets_src=assets)


# --- the SDK ---------------------------------------------------------------------------------


def sdk_twin(tmp_path: Path) -> r2s_twin.Twin:
    return r2s_twin.Twin(output_dir=tmp_path / "out", seed=7)


def test_sdk_seed_and_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    twin = sdk_twin(tmp_path)
    assert twin.seed == 7 and twin.output_dir == tmp_path / "out" and twin.assets_dir is None
    assert np.isclose(twin.rng().random(), np.random.default_rng(7).random())
    monkeypatch.setenv("R2S_SEED", "3")
    monkeypatch.setenv("R2S_OUTPUT_DIR", str(tmp_path / "env-out"))
    from_env = r2s_twin.Twin()
    assert from_env.seed == 3 and from_env.output_dir == tmp_path / "env-out"
    monkeypatch.delenv("R2S_OUTPUT_DIR")
    with pytest.raises(RuntimeError, match="R2S_OUTPUT_DIR"):
        r2s_twin.Twin()


def test_sdk_writes_valid_tracks(tmp_path: Path) -> None:
    twin = sdk_twin(tmp_path)
    camera = r2s_twin.Camera.look_at((0, -2, 0.5), (0, 0, 0.5), width=320, height=240)
    out = twin.write_tracks(fps=30, positions={"ball": falling()}, camera=camera, notes="test")
    assert tracks_problems(out, Requirements(fps=30, num_frames=60, objects=("ball",))) == []
    assert np.allclose(camera.project([[0, 0, 0.5]]), [[160, 120]])
    with pytest.raises(ValueError, match="F, 3"):
        twin.write_tracks(fps=30, positions={"ball": np.zeros((5, 2))}, camera=camera)


def test_sdk_and_lab_checks_agree(tmp_path: Path) -> None:
    write_tracks(make_tracks({"ball": falling()}), tmp_path)
    assert r2s_twin.check_tracks(tmp_path) == []
    np.savez(tmp_path / "tracks.npz", **{"ball.position": np.full((60, 3), np.nan)})
    assert r2s_twin.check_tracks(tmp_path) and tracks_problems(tmp_path, Requirements())


def _sdk_output(root: Path) -> Path:
    """A valid tracks output written by the SDK itself: a ball, a static camera, a rotation."""
    twin = r2s_twin.Twin(output_dir=root, seed=0)
    camera = r2s_twin.Camera.look_at((0, -2, 0.5), (0, 0, 0.5), width=64, height=48)
    rotation = np.tile([1.0, 0.0, 0.0, 0.0], (30, 1))
    return twin.write_tracks(
        fps=30, positions={"ball": falling(30)}, camera=camera, quaternions={"ball": rotation}
    )


def _json_edit(name: str, change: Any) -> Any:
    def edit(root: Path) -> None:
        data = json.loads((root / name).read_text())
        change(data)
        (root / name).write_text(json.dumps(data))  # NaN and Infinity as Python writes them

    return edit


def _arrays_edit(change: Any) -> Any:
    def edit(root: Path) -> None:
        with np.load(root / "tracks.npz") as archive:
            arrays = {key: archive[key] for key in archive.files}
        change(arrays)
        np.savez(root / "tracks.npz", **arrays)

    return edit


def _camera(change: Any) -> Any:
    """Change the camera's extrinsics, as a (4, 4) array."""

    def edit(data: dict[str, Any]) -> None:
        matrix = np.asarray(data["extrinsics"], dtype=float)
        change(matrix)
        data["extrinsics"] = matrix.tolist()

    return _json_edit("camera.json", edit)


def _renamed(name: str) -> Any:
    def edit(root: Path) -> None:
        _json_edit("meta.json", lambda d: d.__setitem__("objects", [name]))(root)
        _arrays_edit(
            lambda a: a.update({k.replace("ball.", f"{name}.", 1): a.pop(k) for k in list(a)})
        )(root)

    return edit


BROKEN_OUTPUTS = {
    "nan-focal": _json_edit("camera.json", lambda d: d["intrinsics"][0].__setitem__(0, np.nan)),
    "inf-focal": _json_edit("camera.json", lambda d: d["intrinsics"][1].__setitem__(1, np.inf)),
    "intrinsics-last-row": _json_edit("camera.json", lambda d: d.__setitem__(
        "intrinsics", [*d["intrinsics"][:2], [0.0, 0.0, 2.0]])),
    "rotation-x2": _camera(lambda m: m[:3, :3].__imul__(2.0)),
    "rotation-x1.01": _camera(lambda m: m[:3, :3].__imul__(1.01)),
    "mirrored-axis": _camera(lambda m: m[:3, 0].__imul__(-1.0)),
    "nan-translation": _camera(lambda m: m.__setitem__((0, 3), np.nan)),
    "extrinsics-last-row": _camera(lambda m: m.__setitem__((3, 3), 2.0)),
    "camera-extra-field": _json_edit("camera.json", lambda d: d.__setitem__("distortion", [0] * 5)),
    "width-true": _json_edit("camera.json", lambda d: d.__setitem__("width", True)),
    "name-with-space": _renamed("red ball"),
    "name-with-dot": _renamed("ball.v2"),
    "name-with-underscore-first": _renamed("_ball"),
    "object-twice": _json_edit("meta.json", lambda d: d.__setitem__("objects", ["ball", "ball"])),
    "meta-extra-field": _json_edit("meta.json", lambda d: d.__setitem__("units", "m")),
    "frame-y-up": _json_edit("meta.json", lambda d: d.__setitem__(
        "frame", {"length_unit": "m", "up_axis": "+y"})),
    "fps-infinity": _json_edit("meta.json", lambda d: d.__setitem__("fps", np.inf)),
    "fps-huge-integer": _json_edit("meta.json", lambda d: d.__setitem__("fps", 10**400)),
    "position-int64": _arrays_edit(lambda a: a.__setitem__(
        "ball.position", a["ball.position"].astype(np.int64))),
    "visible-uint8": _arrays_edit(lambda a: a.__setitem__(
        "ball.visible", np.ones(30, dtype=np.uint8))),
    "quaternion-int": _arrays_edit(lambda a: a.__setitem__(
        "ball.quaternion", np.tile([1, 0, 0, 0], (30, 1)))),
    "position-0d": _arrays_edit(lambda a: a.__setitem__("ball.position", np.float64(1.0))),
}  # fmt: skip


@pytest.mark.parametrize("edit", list(BROKEN_OUTPUTS.values()), ids=list(BROKEN_OUTPUTS))
def test_sdk_and_lab_judge_every_output_alike(edit: Any, tmp_path: Path) -> None:
    """The SDK's ``check`` (and so ``write_tracks`` and ``selftest``) said OK to outputs the
    Lab rejects as invalid, scored 0: a NaN focal length, a camera that is not rigid, a field
    the format does not have, an object listed twice or named with a space... The SDK holds a
    copy of the Lab's rules (it cannot import the Lab): both find the same problems."""
    out = _sdk_output(tmp_path / "out")
    assert r2s_twin.check_tracks(out) == [] == tracks_problems(out, Requirements())
    edit(out)
    lab = tracks_problems(out, Requirements())
    assert lab, "the case breaks no rule of the format"
    assert r2s_twin.check_tracks(out) == lab


@pytest.mark.parametrize("case", ["nan-focal", "rotation-x1.01", "name-with-space"])
def test_sdk_write_tracks_refuses_what_the_lab_rejects(case: str, tmp_path: Path) -> None:
    """``write_tracks`` checks what it wrote: an output the Lab would reject is an error at
    once, not a twin that scores 0 after the agent has finished."""
    twin = r2s_twin.Twin(output_dir=tmp_path / "out", seed=0)
    camera = r2s_twin.Camera.look_at((0, -2, 0.5), (0, 0, 0.5), width=64, height=48)
    positions = {"ball": falling(30)}
    if case == "nan-focal":
        camera.intrinsics[0, 0] = np.nan
    elif case == "rotation-x1.01":
        camera.extrinsics[:3, :3] *= 1.01
    else:
        positions = {"red ball": falling(30)}
    with pytest.raises(ValueError, match="invalid tracks output"):
        twin.write_tracks(fps=30, positions=positions, camera=camera)


def test_sdk_copies_the_labs_tracks_rules() -> None:
    """The SDK cannot import the Lab, so it holds a copy of the format's rules: the same
    files, fields (those of the JSON Schemas), frame, object names and tolerance. A rule
    changed on one side only would make the SDK's check disagree with the Lab's again."""
    from real2simlab.capture.conventions import LENGTH_UNIT, TOLERANCE, UP_AXIS
    from real2simlab.capture.model import OBJECT_NAME
    from real2simlab.formats.tracks import CAMERA_FILE, META_FILE, TRACKS_FILE
    from real2simlab.spec import SpecId
    from real2simlab.spec.schemas import schema_for

    assert SpecId.TRACKS.value == r2s_twin.TRACKS_SPEC
    assert r2s_twin.TRACKS_FILES == (META_FILE, CAMERA_FILE, TRACKS_FILE)
    assert set(schema_for("r2s-tracks-meta")["properties"]) == r2s_twin.META_FIELDS
    assert set(schema_for("r2s-tracks-camera")["properties"]) == r2s_twin.CAMERA_FIELDS
    assert r2s_twin.FRAME == {"length_unit": LENGTH_UNIT, "up_axis": UP_AXIS}
    assert r2s_twin.OBJECT_NAME.pattern == OBJECT_NAME.pattern
    assert r2s_twin.TOLERANCE == TOLERANCE


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.longdouble])
def test_sdk_and_lab_agree_on_unit_norm_at_every_precision(dtype: type, tmp_path: Path) -> None:
    """The SDK's self-check judges a rotation's norm as the Lab does, at the precision the
    rotation is stored in: a long double whose norm is off by just over 1e-3 (by less than
    float64 resolves) is rejected by both, not accepted by the SDK alone."""
    write_tracks(make_tracks({"ball": falling()}), tmp_path)
    one, threshold = np.asarray(1, dtype=dtype), np.asarray(1e-3, dtype=dtype)
    norms = []
    for center in (one + threshold, one - threshold):
        above = below = center
        for _ in range(8):
            above, below = np.nextafter(above, 2 * one), np.nextafter(below, 0 * one)
            norms += [above, below]
        norms.append(center)
    verdicts: set[bool] = set()
    finer_than_float64 = 0
    for norm in norms:
        quaternion = np.zeros((60, 4), dtype=dtype)
        quaternion[:, 0] = norm
        arrays = {"ball.position": falling(), "ball.quaternion": quaternion}
        np.savez(tmp_path / "tracks.npz", **arrays)
        sdk = any("unit norm" in p for p in r2s_twin.check_tracks(tmp_path))
        lab = any("unit norm" in p for p in tracks_problems(tmp_path, Requirements()))
        assert sdk == lab, f"{np.dtype(dtype).name} norm {norm!r}: SDK {sdk}, Lab {lab}"
        verdicts.add(lab)
        off = np.max(np.abs(np.linalg.norm(quaternion, axis=1) - 1))
        if lab and float(off) <= 1e-3:
            finer_than_float64 += 1
    assert verdicts == {True, False}  # the norms straddle the threshold
    if np.finfo(dtype).eps < np.finfo(np.float64).eps:  # e.g. x86-64's 80-bit long double
        assert finer_than_float64, "no norm off by less than float64 resolves was tried"


@pytest.mark.parametrize("value", [np.nan, np.inf, 0.0, 2.0])
def test_a_rotation_is_a_finite_unit_quaternion(value: float, tmp_path: Path) -> None:
    """A NaN or infinite rotation where the object is visible is invalid, like a zero one,
    for the Lab and for the SDK's self-test alike; where it is hidden, it may be NaN."""
    write_tracks(make_tracks({"ball": falling()}), tmp_path)
    quaternion = np.tile([1.0, 0.0, 0.0, 0.0], (60, 1))
    quaternion[5] = value
    arrays = {"ball.position": falling(), "ball.quaternion": quaternion}
    np.savez(tmp_path / "tracks.npz", **arrays)
    assert any("quaternion" in p for p in tracks_problems(tmp_path, Requirements()))
    assert any("quaternion" in p for p in r2s_twin.check_tracks(tmp_path))
    visible = np.ones(60, dtype=bool)
    visible[5] = False
    position = falling()
    position[5] = np.nan
    np.savez(tmp_path / "tracks.npz", **{**arrays, "ball.position": position,
                                         "ball.visible": visible})  # fmt: skip
    assert tracks_problems(tmp_path, Requirements()) == []
    assert r2s_twin.check_tracks(tmp_path) == []


PROGRAM = """
import r2s_twin, numpy as np
twin = r2s_twin.Twin()
h = 0.5
t = np.arange(30) / 30
z = np.maximum(h - 4.9 * t**2, 0.05)
cam = r2s_twin.Camera.look_at((0, -2, 0.5), (0, 0, 0.5), width=64, height=48)
twin.write_tracks(fps=30, positions={"ball": np.c_[0 * z, 0 * z, z]}, camera=cam)
"""


def test_sdk_command_line(tmp_path: Path) -> None:
    source = program(tmp_path, {"run.py": PROGRAM})
    env = {**os.environ, "PYTHONPATH": str(sdk_path().parent)}
    result = subprocess.run(
        [sys.executable, str(sdk_path()), "selftest", str(source)],
        capture_output=True, text=True, env=env, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    assert "deterministic" in result.stdout
    out = tmp_path / "out"
    run = subprocess.run(
        [sys.executable, "run.py"], cwd=source, env={**env, "R2S_OUTPUT_DIR": str(out), "R2S_SEED": "0"},
        capture_output=True, check=False,
    )  # fmt: skip
    assert run.returncode == 0
    check = subprocess.run(
        [sys.executable, str(sdk_path()), "check", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0 and "OK" in check.stdout


def test_environment_ref_defaults() -> None:
    assert EnvironmentRef(image="r2s-python").digest is None
    assert Origin(method="m", task_id="t", created_at=datetime.now(UTC)).privileged is False


def test_4d_world_tracks_without_a_resolution(tmp_path: Path) -> None:
    """4DCodeBench's camera.json has no width/height; the principal point gives them."""
    import json

    from real2simlab.formats import get_format

    world = tmp_path / "world"
    (world / "dynamics").mkdir(parents=True)
    k = [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
    (world / "camera.json").write_text(
        json.dumps({"intrinsics": k, "extrinsic": np.eye(4).tolist()})
    )
    pos = np.zeros((5, 3, 3))
    pos[:, :, 2] = np.linspace(1.0, 0.5, 5)[:, None]
    pos[2, 1] = np.nan
    np.savez(world / "dynamics" / "ball.npz", pos=pos)
    tracks = get_format("4d_world").to_tracks(world)
    assert tracks.camera is not None
    assert (tracks.camera.width, tracks.camera.height) == (640, 480)
    assert np.allclose(tracks.positions("ball")[:, 2], np.linspace(1.0, 0.5, 5))


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_4d_world_centroids_of_huge_points(tmp_path: Path) -> None:
    """The centroid of points so far out that their sum overflows (to infinity, or to NaN with
    both signs) is still their mean, not a frame where the object is missing."""
    world = tmp_path / "world"
    (world / "dynamics").mkdir(parents=True)
    pos = np.zeros((4, 40, 3))
    pos[:, :, 0] = 1e307
    pos[:, ::2, 1], pos[:, 1::2, 1] = 1.5e308, -1.5e308
    pos[:, :, 2] = 0.5
    pos[1, 3] = np.nan  # a point that does not exist in frame 1
    np.savez(world / "dynamics" / "ball.npz", pos=pos)
    positions = get_format("4d_world").to_tracks(world).positions("ball")
    assert np.isfinite(positions).all()
    assert np.allclose(positions[:, [0, 2]], [1e307, 0.5], rtol=1e-12, atol=0)
    assert np.allclose(positions[:, 1], [0, 1.5e308 / 39, 0, 0], rtol=1e-12, atol=1e-300)


def test_sdk_selftest_does_not_call_an_output_it_cannot_check_valid(tmp_path: Path) -> None:
    """A program that writes a ``render.mp4`` (here not even a video) was "valid and
    deterministic" for ``selftest``, which checks only the tracks format."""
    source = program(
        tmp_path,
        {"run.py": "import os, pathlib\n"
                   "pathlib.Path(os.environ['R2S_OUTPUT_DIR'], 'render.mp4').write_text('x')\n"},
    )  # fmt: skip
    result = subprocess.run(
        [sys.executable, str(sdk_path()), "selftest", str(source)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    assert "valid and deterministic" not in result.stdout, result.stdout
    assert "not checked here" in result.stdout and "r2s twin check" in result.stdout


def test_sdk_selftest_catches_nondeterminism(tmp_path: Path) -> None:
    source = program(
        tmp_path,
        {"run.py": PROGRAM.replace("h = 0.5", "h = 0.5 + np.random.default_rng().random()")},
    )
    env = {**os.environ, "PYTHONPATH": str(sdk_path().parent)}
    result = subprocess.run(
        [sys.executable, str(sdk_path()), "selftest", str(source)],
        capture_output=True, text=True, env=env, check=False,
    )  # fmt: skip
    assert result.returncode == 1 and "different outputs" in result.stdout


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("print('done')\n", "wrote nothing"),
        (
            PROGRAM.replace("(0, -2, 0.5)", "(0, -2 - __import__('time').time_ns() % 997, 0.5)"),
            "camera.json",
        ),
        (
            PROGRAM
            + "(twin.output_path('notes.txt')).write_text(str(__import__('time').time_ns()))\n",
            "notes.txt",
        ),
    ],
    ids=["nothing written", "camera moves", "an extra file differs"],
)
def test_sdk_selftest_checks_every_output(source: str, expected: str, tmp_path: Path) -> None:
    program_dir = program(tmp_path, {"run.py": source})
    env = {**os.environ, "PYTHONPATH": str(sdk_path().parent)}
    result = subprocess.run(
        [sys.executable, str(sdk_path()), "selftest", str(program_dir)],
        capture_output=True, text=True, env=env, check=False,
    )  # fmt: skip
    assert result.returncode == 1 and expected in result.stdout, result.stdout


@pytest.mark.parametrize(
    ("extra", "said"),
    [
        # a tracks output with a render.mp4 the SDK cannot read: its tracks are said valid, the
        # video is not (the Lab probes it; it was "valid and deterministic", input-validation K2)
        ("pathlib.Path(out, 'render.mp4').write_text('not a video')\n", "valid tracks"),
        # a video's notes: not a tracks output, so not checked as one (it failed on a missing
        # tracks.npz, K3)
        ("[pathlib.Path(out, n).unlink() for n in ('camera.json', 'tracks.npz')]\n"
         "pathlib.Path(out, 'render.mp4').write_text('x')\n", "not a tracks output"),
    ],
)  # fmt: skip
def test_sdk_selftest_says_what_it_checked(extra: str, said: str, tmp_path: Path) -> None:
    run = PROGRAM + "import os, pathlib\nout = os.environ['R2S_OUTPUT_DIR']\n" + extra
    source = program(tmp_path, {"run.py": run})
    env = {**os.environ, "PYTHONPATH": str(sdk_path().parent)}
    result = subprocess.run(
        [sys.executable, str(sdk_path()), "selftest", str(source)],
        capture_output=True, text=True, env=env, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    assert said in result.stdout and "valid and deterministic" not in result.stdout, result.stdout


def test_a_camera_number_beyond_float_is_a_problem_not_a_traceback(tmp_path: Path) -> None:
    """``10**400`` in camera.json: the SDK's check ended in an ``OverflowError`` traceback and
    the Lab said "the output could not be checked" (input-validation K4)."""
    import json

    out = _sdk_output(tmp_path)
    camera = json.loads((out / "camera.json").read_text())
    camera["intrinsics"][0][0] = 10**400
    (out / "camera.json").write_text(json.dumps(camera))
    result = subprocess.run(
        [sys.executable, str(sdk_path()), "check", str(out)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert result.returncode == 1 and "Traceback" not in result.stderr, result.stderr
    assert "camera.json: " in result.stdout, result.stdout
    problems = get_format("tracks").validate(out)
    assert any(p.startswith("camera.json: ") for p in problems), problems
