"""The ``tracks`` format (``r2s-tracks/1.0``): per-object trajectories and the camera.

A ``tracks`` output directory holds:

* ``meta.json``: ``spec``, ``fps``, ``num_frames``, ``objects`` (names), ``frame``
  (``length_unit = "m"``, ``up_axis = "+z"``);
* ``camera.json``: ``intrinsics`` (3x3, pixels), ``width``, ``height`` and ``extrinsics``,
  the camera-to-world transform: one 4x4 matrix, or one per frame (F x 4 x 4);
* ``tracks.npz``: for every object ``<name>.position`` (F, 3) in metres, the object's centroid
  in the world frame; optionally ``<name>.quaternion`` (F, 4) ``[qw, qx, qy, qz]`` and
  ``<name>.visible`` (F,) booleans; positions and quaternions (finite unit quaternions) may
  be NaN only where ``visible`` is false;
* optionally ``render.mp4``, the rendered video.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, TypeGuard, cast

import numpy as np
from numpy.typing import NDArray

from real2simlab.capture.conventions import check_intrinsics_array, check_rigid_array
from real2simlab.capture.model import OBJECT_NAME
from real2simlab.errors import host_failure
from real2simlab.formats.base import OutputFormat, Requirements
from real2simlab.media import MediaError, probe_video
from real2simlab.spec import SpecId

META_FILE = "meta.json"
CAMERA_FILE = "camera.json"
TRACKS_FILE = "tracks.npz"
RENDER_FILE = "render.mp4"
FIELDS = ("position", "quaternion", "visible")
FPS_TOLERANCE = 1e-6

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Camera:
    """A pinhole camera: intrinsics, resolution, and camera-to-world extrinsics."""

    intrinsics: FloatArray
    width: int
    height: int
    extrinsics: FloatArray
    """(4, 4) for a static camera or (F, 4, 4) per frame."""

    def pose(self, frame: int) -> FloatArray:
        """The camera-to-world transform at ``frame``."""
        if self.extrinsics.ndim == 2:
            return self.extrinsics
        return self.extrinsics[min(frame, len(self.extrinsics) - 1)]

    def project(self, points: FloatArray) -> FloatArray:
        """Pixel coordinates (F, 2) of world points (F, 3), frame by frame; NaN behind it."""
        count = len(points)
        if self.extrinsics.ndim == 2:
            poses = np.broadcast_to(self.extrinsics, (count, 4, 4))
        else:
            index = np.minimum(np.arange(count), len(self.extrinsics) - 1)
            poses = self.extrinsics[index]
        world_to_camera = np.linalg.inv(poses)
        homogeneous = np.concatenate([points, np.ones((count, 1))], axis=1)
        local = np.einsum("fij,fj->fi", world_to_camera, homogeneous)[:, :3]
        depth = local[:, 2]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            pixels = (self.intrinsics @ local.T).T
            uv = pixels[:, :2] / pixels[:, 2:3]
        uv[~(depth > 1e-9)] = np.nan
        return uv

    def at_times(self, own: FloatArray, times: FloatArray) -> Camera:
        """This camera at ``times``, its poses being sampled at ``own``: a moving camera's pose
        interpolated between the neighbouring samples (translation linearly, rotation along
        the shortest arc) and held beyond the first and last; a static camera unchanged."""
        if self.extrinsics.ndim == 2:
            return self
        return replace(self, extrinsics=_poses_at(times, own, self.extrinsics))

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.width, self.height))

    def to_json(self) -> dict[str, Any]:
        return {
            "intrinsics": self.intrinsics.tolist(),
            "width": self.width,
            "height": self.height,
            "extrinsics": self.extrinsics.tolist(),
        }


@dataclass(frozen=True)
class ObjectTrack:
    """One object's trajectory."""

    position: FloatArray
    quaternion: FloatArray | None = None
    visible: BoolArray | None = None

    def __len__(self) -> int:
        return len(self.position)


@dataclass(frozen=True)
class Tracks:
    """Per-object trajectories sampled at ``fps``, with the camera that observes them."""

    fps: float
    objects: dict[str, ObjectTrack]
    camera: Camera | None = None
    extra_meta: dict[str, Any] = field(default_factory=lambda: {})

    @property
    def num_frames(self) -> int:
        lengths = {len(track) for track in self.objects.values()}
        return lengths.pop() if len(lengths) == 1 else 0

    def times(self) -> FloatArray:
        return np.arange(self.num_frames, dtype=np.float64) / self.fps

    @property
    def duration(self) -> float:
        """Time of the last frame, in seconds."""
        return (self.num_frames - 1) / self.fps if self.num_frames else 0.0

    def positions(self, name: str) -> FloatArray:
        """Positions with NaN where the object is not visible."""
        track = self.objects[name]
        position = track.position.astype(np.float64, copy=True)
        if track.visible is not None:
            position[~track.visible] = np.nan
        return position

    def at_times(self, times: FloatArray) -> Tracks:
        """These tracks linearly interpolated at ``times``; NaN outside the sampled span. A
        moving camera is resampled at the same times, so frame ``k`` of the result projects
        with the camera of that same instant."""
        own = self.times()
        objects: dict[str, ObjectTrack] = {}
        for name in self.objects:
            position = self.positions(name)
            resampled = np.stack(
                [_interp(times, own, position[:, axis]) for axis in range(3)], axis=1
            )
            visible: BoolArray = np.asarray(np.isfinite(resampled).all(axis=1), dtype=np.bool_)
            objects[name] = ObjectTrack(position=resampled, visible=visible)
        fps = 1.0 / float(times[1] - times[0]) if len(times) > 1 else self.fps
        camera = None if self.camera is None else self.camera.at_times(own, times)
        return replace(self, fps=fps, objects=objects, camera=camera)

    def window(self, start: int, end: int) -> Tracks:
        """Frames ``[start, end)``."""
        objects = {
            name: ObjectTrack(
                position=track.position[start:end],
                quaternion=None if track.quaternion is None else track.quaternion[start:end],
                visible=None if track.visible is None else track.visible[start:end],
            )
            for name, track in self.objects.items()
        }
        camera = self.camera
        if camera is not None and camera.extrinsics.ndim == 3:
            camera = replace(camera, extrinsics=camera.extrinsics[start:end])
        return replace(self, objects=objects, camera=camera)


def _quaternion(rotation: FloatArray) -> FloatArray:
    """The unit quaternion ``[w, x, y, z]`` of a rotation matrix (Shepperd's method)."""
    r = rotation
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [(r[2, 1] - r[1, 2]) / s, s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4]
    quaternion = np.asarray(q, dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _rotation(q: FloatArray) -> FloatArray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _slerp(a: FloatArray, b: FloatArray, t: float) -> FloatArray:
    dot = float(a @ b)
    if dot < 0:  # the shortest arc
        b, dot = -b, -dot
    if dot > 0.9995:
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    return (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)


def _poses_at(times: FloatArray, own: FloatArray, poses: FloatArray) -> FloatArray:
    """Rigid poses (N, 4, 4) at ``times`` from poses sampled at ``own``."""
    count = min(len(own), len(poses))
    own, poses = own[:count], poses[:count]
    result = np.empty((len(times), 4, 4), dtype=np.float64)
    for k in range(len(times)):
        time = float(times[k])
        index = int(np.searchsorted(own, time, side="right")) - 1
        if index < 0:
            result[k] = poses[0]
            continue
        if index >= count - 1 or np.isclose(time, own[index], atol=1e-9):
            result[k] = poses[min(index, count - 1)]
            continue
        t = float((time - own[index]) / (own[index + 1] - own[index]))
        if np.isclose(t, 1.0, atol=1e-9):
            result[k] = poses[index + 1]
            continue
        a, b = poses[index], poses[index + 1]
        pose = np.eye(4)
        pose[:3, :3] = _rotation(_slerp(_quaternion(a[:3, :3]), _quaternion(b[:3, :3]), t))
        pose[:3, 3] = (1 - t) * a[:3, 3] + t * b[:3, 3]
        result[k] = pose
    return result


def _interp(times: FloatArray, own: FloatArray, values: FloatArray) -> FloatArray:
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return np.full(len(times), np.nan)
    samples, known = own[finite], values[finite]
    result = np.interp(times, samples, known, left=np.nan, right=np.nan)
    # np.interp's slope (b - a) / dt overflows between huge but finite samples (|b - a| above
    # about 1.8e308 * dt): there, interpolate as (1 - w) a + w b, which does not.
    inside = (times >= samples[0]) & (times <= samples[-1])
    overflowed = np.flatnonzero(inside & ~np.isfinite(result))
    if overflowed.size and len(samples) > 1:
        after = np.clip(
            np.searchsorted(samples, times[overflowed], side="right"), 1, len(samples) - 1
        )
        before = after - 1
        weight = (times[overflowed] - samples[before]) / (samples[after] - samples[before])
        with np.errstate(over="ignore"):
            result[overflowed] = (1 - weight) * known[before] + weight * known[after]
    # Do not interpolate across invisible gaps: a time is valid only if both neighbours are.
    index = np.searchsorted(own, times, side="right") - 1
    lower = np.clip(index, 0, len(own) - 1)
    upper = np.clip(index + 1, 0, len(own) - 1)
    exact = np.isclose(times, own[lower], atol=1e-9)
    valid = finite[lower] & (finite[upper] | exact)
    result[~valid] = np.nan
    return result


def read_tracks(root: Path) -> Tracks:
    """Read a ``tracks`` directory. Raises ``ValueError`` if it is malformed; validate first."""
    root = Path(root)
    meta = cast(dict[str, Any], json.loads((root / META_FILE).read_text(encoding="utf-8")))
    camera = _read_camera(root / CAMERA_FILE) if (root / CAMERA_FILE).is_file() else None
    objects: dict[str, ObjectTrack] = {}
    with np.load(root / TRACKS_FILE, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for name in cast(list[str], meta["objects"]):
        quaternion = arrays.get(f"{name}.quaternion")
        visible = arrays.get(f"{name}.visible")
        objects[name] = ObjectTrack(
            position=np.asarray(arrays[f"{name}.position"], dtype=np.float64),
            quaternion=None if quaternion is None else np.asarray(quaternion, dtype=np.float64),
            visible=None if visible is None else np.asarray(visible, dtype=bool),
        )
    extra = {k: v for k, v in meta.items() if k not in {"spec", "fps", "num_frames", "objects"}}
    return Tracks(fps=float(meta["fps"]), objects=objects, camera=camera, extra_meta=extra)


def _read_camera(path: Path) -> Camera:
    data = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    return Camera(
        intrinsics=np.asarray(data["intrinsics"], dtype=np.float64),
        width=int(data["width"]),
        height=int(data["height"]),
        extrinsics=np.asarray(data["extrinsics"], dtype=np.float64),
    )


def write_tracks(tracks: Tracks, root: Path) -> Path:
    """Write ``tracks`` as a ``tracks`` directory at ``root`` (created if needed)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "spec": SpecId.TRACKS.value,
        "fps": tracks.fps,
        "num_frames": tracks.num_frames,
        "objects": list(tracks.objects),
        "frame": {"length_unit": "m", "up_axis": "+z"},
        **{k: v for k, v in tracks.extra_meta.items() if k != "frame"},
    }
    (root / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if tracks.camera is not None:
        text = json.dumps(tracks.camera.to_json(), indent=2) + "\n"
        (root / CAMERA_FILE).write_text(text, encoding="utf-8")
    arrays: dict[str, NDArray[Any]] = {}
    for name, track in tracks.objects.items():
        arrays[f"{name}.position"] = np.asarray(track.position, dtype=np.float64)
        if track.quaternion is not None:
            arrays[f"{name}.quaternion"] = np.asarray(track.quaternion, dtype=np.float64)
        if track.visible is not None:
            arrays[f"{name}.visible"] = np.asarray(track.visible, dtype=bool)
    np.savez(root / TRACKS_FILE, allow_pickle=False, **arrays)
    return root


class TracksFormat(OutputFormat):
    """Lab-native per-object trajectories."""

    name: ClassVar[str] = "tracks"
    summary: ClassVar[str] = "Per-object centroid trajectories in metres, plus the camera."
    content: ClassVar[tuple[str, ...] | None] = (META_FILE, CAMERA_FILE, TRACKS_FILE, RENDER_FILE)
    provides_tracks: ClassVar[bool] = True

    def validate(self, root: Path, requirements: Requirements | None = None) -> list[str]:
        return tracks_problems(Path(root), requirements or Requirements())

    def to_tracks(self, root: Path) -> Tracks:
        return read_tracks(root)

    def documentation(self) -> str:
        return TRACKS_DOC


def tracks_problems(root: Path, requirements: Requirements) -> list[str]:
    """Every violation of the ``tracks`` format (and of ``requirements``) in ``root``."""
    problems: list[str] = []
    meta = _read_json(root / META_FILE, problems)
    camera = _read_json(root / CAMERA_FILE, problems)
    if not (root / TRACKS_FILE).is_file():
        problems.append(f"{TRACKS_FILE}: file is required")
    if problems or meta is None:
        return problems
    objects, fps, num_frames = _meta_problems(meta, problems)
    if camera is not None:
        problems.extend(_camera_problems(camera, num_frames))
    try:
        with np.load(root / TRACKS_FILE, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except Exception as error:  # truncated, not a zip, a pickled object: not a valid archive
        if host_failure(error):
            raise  # this machine could not read it: not a problem of the archive
        return [*problems, f"{TRACKS_FILE}: cannot read: {type(error).__name__}: {error}"]
    problems.extend(_array_problems(arrays, objects, num_frames))
    wanted_fps = requirements.fps
    if (
        wanted_fps is not None
        and fps is not None
        and (abs(fps - wanted_fps) > FPS_TOLERANCE * max(1.0, wanted_fps))
    ):
        problems.append(f"{META_FILE}: fps is {fps:g}, the task requires {wanted_fps:g}")
    wanted_frames = requirements.num_frames
    if wanted_frames is not None and num_frames is not None and num_frames != wanted_frames:
        problems.append(
            f"{META_FILE}: num_frames is {num_frames}, the task requires {wanted_frames}"
        )
    missing = sorted(set(requirements.objects) - set(objects))
    if missing:
        problems.append(f"{META_FILE}: objects {missing} are required but missing")
    render = root / RENDER_FILE
    if render.exists():
        try:
            probe_video(render)
        except MediaError as error:
            problems.append(f"{RENDER_FILE}: {error}")
    return problems


def _read_json(path: Path, problems: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        problems.append(f"{path.name}: file is required")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if host_failure(error):
            raise  # this machine could not read it: not a problem of the file
        problems.append(f"{path.name}: cannot read JSON: {error}")
        return None
    if not isinstance(data, dict):
        problems.append(f"{path.name}: must be a JSON object")
        return None
    return cast(dict[str, Any], data)


def _meta_problems(
    meta: dict[str, Any], problems: list[str]
) -> tuple[list[str], float | None, int | None]:
    allowed = {"spec", "fps", "num_frames", "objects", "frame", "notes", "generator"}
    for key in sorted(set(meta) - allowed):
        problems.append(f"{META_FILE}: unknown field {key!r}")
    if meta.get("spec") != SpecId.TRACKS.value:
        problems.append(f"{META_FILE}: spec must be {SpecId.TRACKS.value!r}")
    fps = meta.get("fps")
    if not _finite(fps) or not fps > 0:  # Infinity is above 0, and no frame rate
        problems.append(f"{META_FILE}: fps must be a positive number")
        fps = None
    num_frames = meta.get("num_frames")
    if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames < 1:
        problems.append(f"{META_FILE}: num_frames must be a positive integer")
        num_frames = None
    raw = meta.get("objects")
    objects: list[str] = []
    if not isinstance(raw, list) or not raw:
        problems.append(f"{META_FILE}: objects must be a non-empty list of names")
    else:
        for item in cast(list[Any], raw):
            if not isinstance(item, str) or not OBJECT_NAME.match(item):
                problems.append(f"{META_FILE}: invalid object name {item!r}")
            elif item in objects:
                problems.append(f"{META_FILE}: object {item!r} is listed twice")
            else:
                objects.append(item)
    frame = meta.get("frame", {"length_unit": "m", "up_axis": "+z"})
    if frame != {"length_unit": "m", "up_axis": "+z"}:
        problems.append(f'{META_FILE}: frame must be {{"length_unit": "m", "up_axis": "+z"}}')
    return objects, float(fps) if fps is not None else None, num_frames


def _finite(value: Any) -> TypeGuard[int | float]:
    """A finite number (an integer too large for a float is not one)."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _camera_problems(camera: dict[str, Any], num_frames: int | None) -> list[str]:
    problems: list[str] = []
    for key in sorted(set(camera) - {"intrinsics", "width", "height", "extrinsics"}):
        problems.append(f"{CAMERA_FILE}: unknown field {key!r}")
    for key in ("width", "height"):
        value = camera.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            problems.append(f"{CAMERA_FILE}: {key} must be a positive integer")
    try:
        k = np.asarray(camera.get("intrinsics"), dtype=np.float64)
        if k.shape != (3, 3) or not np.all(np.isfinite(k)):
            raise ValueError(f"intrinsics must be a finite 3x3 matrix, got shape {k.shape}")
        check_intrinsics_array(k)
    except (ValueError, TypeError, OverflowError) as error:  # 10**400: beyond float
        problems.append(f"{CAMERA_FILE}: {error}")
    try:
        t = np.asarray(camera.get("extrinsics"), dtype=np.float64)
        if t.shape == (4, 4):
            t = t[None]
        if t.ndim != 3 or t.shape[1:] != (4, 4) or not np.all(np.isfinite(t)):
            raise ValueError("extrinsics must be a finite 4x4 matrix or a list of them")
        if num_frames is not None and len(t) not in (1, num_frames):
            raise ValueError(f"extrinsics has {len(t)} matrices for {num_frames} frames")
        for index, matrix in enumerate(t):
            try:
                check_rigid_array(matrix)
            except ValueError as error:
                raise ValueError(f"extrinsics[{index}]: {error}") from None
    except (ValueError, TypeError, OverflowError) as error:  # 10**400: beyond float
        problems.append(f"{CAMERA_FILE}: {error}")
    return problems


def _array_problems(
    arrays: dict[str, NDArray[Any]], objects: list[str], num_frames: int | None
) -> list[str]:
    problems: list[str] = []
    known = {f"{name}.{part}" for name in objects for part in FIELDS}
    for key in sorted(set(arrays) - known):
        problems.append(
            f"{TRACKS_FILE}: unexpected array {key!r} (objects are listed in meta.json)"
        )
    for name in objects:
        position = arrays.get(f"{name}.position")
        if position is None:
            problems.append(f"{TRACKS_FILE}: {name}.position is missing")
            continue
        frames = num_frames if num_frames is not None else (len(position) if position.ndim else 0)
        if position.shape != (frames, 3) or position.dtype.kind != "f":
            problems.append(
                f"{TRACKS_FILE}: {name}.position must be float ({frames}, 3), "
                f"got {position.dtype} {position.shape}"
            )
            continue
        visible = arrays.get(f"{name}.visible")
        mask = np.ones(frames, dtype=bool)
        if visible is not None:
            if visible.shape != (frames,) or visible.dtype != np.bool_:
                problems.append(f"{TRACKS_FILE}: {name}.visible must be bool ({frames},)")
            else:
                mask = visible
        if not np.all(np.isfinite(position[mask])):
            problems.append(
                f"{TRACKS_FILE}: {name}.position has non-finite values where the object is "
                "visible (mark those frames with <name>.visible = false)"
            )
        quaternion = arrays.get(f"{name}.quaternion")
        if quaternion is not None:
            if quaternion.shape != (frames, 4) or quaternion.dtype.kind != "f":
                problems.append(f"{TRACKS_FILE}: {name}.quaternion must be float ({frames}, 4)")
            elif not np.all(np.isfinite(quaternion[mask])):
                problems.append(
                    f"{TRACKS_FILE}: {name}.quaternion has non-finite values where the object "
                    "is visible (a rotation is a finite unit quaternion)"
                )
            else:
                norms = np.linalg.norm(quaternion[mask], axis=1)
                if norms.size and np.max(np.abs(norms - 1.0)) > 1e-3:
                    problems.append(f"{TRACKS_FILE}: {name}.quaternion must have unit norm")
    return problems


TRACKS_DOC = """\
# Output format `tracks` (r2s-tracks/1.0)

Write these files into the directory named by `R2S_OUTPUT_DIR`:

- `meta.json`:

      {"spec": "r2s-tracks/1.0", "fps": <frames per second>, "num_frames": <F>,
       "objects": ["<name>", ...], "frame": {"length_unit": "m", "up_axis": "+z"}}

- `camera.json`:

      {"intrinsics": <3x3, pixels>, "width": W, "height": H,
       "extrinsics": <4x4 camera-to-world, or a list of F of them>}

- `tracks.npz`, for every object: `<name>.position`, float (F, 3), the object's centroid in
  metres in the world frame; optionally `<name>.quaternion`, (F, 4) `[qw, qx, qy, qz]`, and
  `<name>.visible`, bool (F,).
- `render.mp4`, optionally: your rendering of the scene.

Conventions: metres, seconds, radians; world +z is up and gravity points along -z; frame `i` is
at time `i / fps`; cameras use OpenCV axes (+x right, +y down, +z forward). Positions and
quaternions (finite, unit norm) may be NaN only where `<name>.visible` is false. No other
arrays may appear in `tracks.npz`.

The SDK writes this format for you: `twin.write_tracks(...)` (see `r2s_twin.py`).
"""
