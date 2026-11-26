from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from real2simlab.capture import (
    CALIBRATED,
    VIDEO_ONLY,
    Capture,
    InputSpec,
    Profile,
    Provenance,
    Sourced,
    VisibilityPolicy,
    detect_profiles,
    load_capture,
    save_capture,
)
from real2simlab.capture import paths as field_paths
from real2simlab.capture.from_video import capture_from_video, intrinsics_from_fov
from real2simlab.errors import SpecError
from real2simlab.media import probe_video, read_frames, write_video

K = [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]]
POSE = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def capture(**views: dict) -> Capture:
    data = {
        "capture_id": "test/drop",
        "views": views
        or {
            "cam": {
                "rgb": {
                    "path": "cam/rgb.mp4",
                    "fps": 30,
                    "num_frames": 10,
                    "resolution": [320, 240],
                }
            }
        },
    }
    return Capture.model_validate(data)


def calibrated() -> Capture:
    measured = {"provenance": "measured"}
    return Capture.model_validate(
        {
            "capture_id": "test/drop",
            "world": {"metric_scale": {"value": True, **measured}},
            "views": {
                "cam": {
                    "rgb": {
                        "path": "cam/rgb.mp4",
                        "fps": 30,
                        "num_frames": 10,
                        "resolution": [320, 240],
                    },
                    "intrinsics": {"value": K, **measured},
                    "extrinsics": {"value": POSE, **measured},
                    "depth": {"path": "cam/depth.npz", "unit": "m", "kind": "metric", **measured},
                }
            },
            "scene": {"objects": [{"name": "ball", "role": "dynamic"}]},
            "text": {"task": "drop"},
        }
    )


def test_minimal_capture_round_trips(tmp_path: Path) -> None:
    item = capture()
    assert item.spec == "r2s-capture/1.0"
    save_capture(item, tmp_path)
    with pytest.raises(SpecError, match="missing media"):
        load_capture(tmp_path)
    assert load_capture(tmp_path, check_media=False) == item


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"path": "/abs.mp4"}, "relative"),
        ({"path": "../up.mp4"}, "relative"),
        ({"timestamps": [0.0, 0.1]}, "timestamps"),
        ({"fps": -1}, "greater than 0"),
    ],
)
def test_invalid_video_streams(change: dict, message: str) -> None:
    rgb = {"path": "cam/rgb.mp4", "fps": 30, "num_frames": 10, "resolution": [320, 240], **change}
    with pytest.raises(ValidationError, match=message):
        capture(cam={"rgb": rgb})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        Capture.model_validate({"capture_id": "x", "views": {}, "colour": 1})


def test_conventions_are_checked() -> None:
    rgb = {"path": "a.mp4", "fps": 30, "num_frames": 10, "resolution": [4, 4]}
    bad_k = [[0.0, 0, 1], [0, 1, 1], [0, 0, 1]]
    with pytest.raises(ValidationError, match="focal"):
        capture(cam={"rgb": rgb, "intrinsics": {"value": bad_k, "provenance": "measured"}})
    skew = [[1.0, 1.0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    with pytest.raises(ValidationError, match="rotation"):
        capture(cam={"rgb": rgb, "extrinsics": {"value": skew, "provenance": "measured"}})
    both = {"rgb": rgb, "extrinsics": {"value": POSE, "provenance": "measured"},
            "extrinsics_per_frame": {"path": "poses.npy", "provenance": "estimated"}}  # fmt: skip
    with pytest.raises(ValidationError, match="either"):
        capture(cam=both)


def test_per_frame_extrinsics(tmp_path: Path) -> None:
    rgb = {"path": "rgb.mp4", "fps": 30, "num_frames": 3, "resolution": [4, 4]}
    item = capture(cam={"rgb": rgb, "intrinsics": {"value": K, "provenance": "measured"},
                        "extrinsics_per_frame": {"path": "poses.npy", "provenance": "estimated"}})  # fmt: skip
    save_capture(item, tmp_path)
    (tmp_path / "rgb.mp4").write_bytes(b"x")
    np.save(tmp_path / "poses.npy", np.stack([np.eye(4)] * 2))
    with pytest.raises(SpecError, match="need shape"):
        load_capture(tmp_path)
    poses = np.stack([np.eye(4)] * 3)
    poses[1, 0, 1] = 0.5
    np.save(tmp_path / "poses.npy", poses)
    with pytest.raises(SpecError, match=r"\[1\]"):
        load_capture(tmp_path)
    # Files a user may give instead: each is a validation problem naming the file, not a crash.
    poses = tmp_path / "poses.npy"
    for wrong, expected in (
        (lambda: np.savez(tmp_path / "a.npz", extrinsics=np.stack([np.eye(4)] * 3)), "savez"),
        (lambda: np.save(poses, np.array(["a"] * 48).reshape(3, 4, 4)), "need numbers"),
        (lambda: np.save(poses, np.array([{}], dtype=object)), "cannot read"),
        (lambda: poses.write_bytes(b"not numpy"), "cannot read"),
    ):
        wrong()
        if (tmp_path / "a.npz").exists():
            (tmp_path / "a.npz").replace(poses)  # an archive under the name the capture gives
        with pytest.raises(SpecError, match=rf"\(poses\.npy\): .*{expected}"):
            load_capture(tmp_path)
    np.save(poses, np.stack([np.eye(4)] * 3))
    assert load_capture(tmp_path).views["cam"].calibrated
    assert Profile.MOVING_CAMERA in detect_profiles(item)


def test_profiles_and_input_specs() -> None:
    full = calibrated()
    assert detect_profiles(full) == {Profile.RGB, Profile.RGB_CALIB, Profile.RGBD}
    assert InputSpec(profiles=frozenset({Profile.MULTIVIEW})).missing(full) == ["profile:multiview"]
    assert InputSpec(fields=frozenset({"robot"})).missing(full) == ["robot"]
    assert InputSpec(profiles=frozenset({Profile.RGBD})).missing(full) == []


def test_visibility_policies() -> None:
    full = calibrated()
    visible, audit = VIDEO_ONLY.apply(full)
    assert visible.views["cam"].intrinsics is None and visible.scene is None
    assert visible.text is not None
    assert "views.cam.intrinsics" in audit.hidden and "scene" in audit.hidden
    visible, audit = CALIBRATED.apply(full)
    assert visible.views["cam"].intrinsics is not None and visible.views["cam"].depth is None
    assert "views.cam.depth" in audit.hidden
    windowed, audit = VisibilityPolicy(name="w", frame_window=(2, 6)).apply(full)
    view = windowed.views["cam"]
    assert view.rgb.num_frames == 4 and view.frame_range == (2, 6)
    assert view.frame_time(0) == pytest.approx(2 / 30)
    with pytest.raises(ValueError):
        VisibilityPolicy(name="bad", frame_window=(3, 3))


def test_field_paths() -> None:
    data = calibrated().to_json()
    assert field_paths.is_present(data, "views.*.intrinsics")
    assert not field_paths.is_present(data, "robot")
    with pytest.raises(ValueError):
        field_paths.parse("a..b")
    kept = field_paths.select(data, ["text"])
    assert set(kept) == {"spec", "capture_id", "views", "text"}


def test_sourced_requires_provenance() -> None:
    assert Sourced[float](value=1.0, provenance=Provenance.ESTIMATED).value == 1.0
    with pytest.raises(ValidationError):
        Sourced[float].model_validate({"value": 1.0})


def test_video_io_and_from_video(tmp_path: Path) -> None:
    frames = np.zeros((12, 64, 80, 3), dtype=np.uint8)
    frames[:, 10:20, 10:20] = 255
    video = write_video(frames, tmp_path / "clip.mp4", fps=24)
    info = probe_video(video)
    assert (info.width, info.height, info.num_frames, info.fps) == (80, 64, 12, 24)
    assert read_frames(video).shape == (12, 64, 80, 3)
    k = intrinsics_from_fov(80, 64, 90.0)
    assert k[0][0] == pytest.approx(40.0)
    item = capture_from_video(
        video, tmp_path / "cap", capture_id="lab/clip", intrinsics=k,
        intrinsics_provenance=Provenance.ESTIMATED, object_sizes={"ball": 0.07}, task_text="drop",
    )  # fmt: skip
    loaded = load_capture(tmp_path / "cap")
    assert loaded == item
    assert loaded.world.metric_scale is not None and loaded.world.metric_scale.value
    assert loaded.views["cam"].intrinsics is not None
    assert loaded.object_names() == ["ball"]
    with pytest.raises(FileExistsError):
        capture_from_video(video, tmp_path / "cap", capture_id="again")
    with pytest.raises(ValueError):
        intrinsics_from_fov(10, 10, 190)
    json.loads((tmp_path / "cap" / "capture.json").read_text())
