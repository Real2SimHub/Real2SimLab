"""r2s_twin: the Real2SimLab twin SDK. A single file that needs only NumPy.

A twin program uses it to read its seed and to write its output in the ``tracks`` format,
following the run contract of ``r2s-twin/1.0``::

    import numpy as np
    import r2s_twin

    twin = r2s_twin.Twin()
    rng = twin.rng()                    # every random number derives from R2S_SEED
    ...  # simulate
    twin.write_tracks(fps=30, positions={"ball": xyz}, camera=camera)

Check an output, or a whole program, from the command line::

    python r2s_twin.py check OUTPUT_DIR
    python r2s_twin.py selftest PROGRAM_DIR

This file does not depend on Real2SimLab, so frozen twins can be re-run without it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

SDK_VERSION = "1.0.0"
TRACKS_SPEC = "r2s-tracks/1.0"

OUTPUT_DIR_ENV = "R2S_OUTPUT_DIR"
SEED_ENV = "R2S_SEED"
ASSETS_DIR_ENV = "R2S_ASSETS_DIR"

# The Lab's rules for a tracks output (real2simlab.formats.tracks), which this file cannot
# import: tests/test_formats_twin_sdk.py checks that both judge the same outputs alike.
TRACKS_FILES = ("meta.json", "camera.json", "tracks.npz")
META_FIELDS = frozenset({"spec", "fps", "num_frames", "objects", "frame", "notes", "generator"})
CAMERA_FIELDS = frozenset({"intrinsics", "width", "height", "extrinsics"})
FRAME = {"length_unit": "m", "up_axis": "+z"}
OBJECT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
TOLERANCE = 1e-4
"""How far a camera's matrices may be from exact (the last rows, a rotation's orthonormality)."""


def _is_number(value: Any) -> bool:
    """A finite number (an integer too large for a float is not one)."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


class Camera:
    """A pinhole camera: 3x3 intrinsics in pixels, resolution, camera-to-world extrinsics.

    ``extrinsics`` is a 4x4 matrix for a static camera or an (F, 4, 4) array; OpenCV axes.
    """

    def __init__(self, intrinsics: Any, width: int, height: int, extrinsics: Any) -> None:
        self.intrinsics = np.asarray(intrinsics, dtype=np.float64)
        self.width = int(width)
        self.height = int(height)
        self.extrinsics = np.asarray(extrinsics, dtype=np.float64)

    @classmethod
    def look_at(
        cls,
        eye: Sequence[float],
        target: Sequence[float],
        *,
        width: int,
        height: int,
        horizontal_fov_deg: float = 60.0,
        up: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> Camera:
        """A static camera at ``eye`` looking at ``target`` with square pixels."""
        eye_v = np.asarray(eye, dtype=np.float64)
        forward = np.asarray(target, dtype=np.float64) - eye_v
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(up, dtype=np.float64))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        pose = np.eye(4)
        pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, down, forward, eye_v
        focal = (width / 2) / math.tan(math.radians(horizontal_fov_deg) / 2)
        k = np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])
        return cls(k, width, height, pose)

    def project(self, points: Any) -> np.ndarray:
        """Pixel coordinates (N, 2) of world points (N, 3) for a static camera."""
        pose = self.extrinsics if self.extrinsics.ndim == 2 else self.extrinsics[0]
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        local = (np.linalg.inv(pose) @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
        uvw = (self.intrinsics @ local.T).T
        return uvw[:, :2] / uvw[:, 2:3]

    def to_json(self) -> dict[str, Any]:
        return {
            "intrinsics": self.intrinsics.tolist(),
            "width": self.width,
            "height": self.height,
            "extrinsics": self.extrinsics.tolist(),
        }


class Twin:
    """The running twin: its seed, output directory and assets."""

    def __init__(
        self,
        *,
        output_dir: str | os.PathLike[str] | None = None,
        seed: int | None = None,
    ) -> None:
        out = output_dir if output_dir is not None else os.environ.get(OUTPUT_DIR_ENV)
        if not out:
            raise RuntimeError(f"{OUTPUT_DIR_ENV} is not set: where should the output go?")
        self.output_dir = Path(out)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_seed = seed if seed is not None else os.environ.get(SEED_ENV, "0")
        self.seed = int(raw_seed)
        assets = os.environ.get(ASSETS_DIR_ENV)
        self.assets_dir = Path(assets) if assets else None

    def rng(self) -> np.random.Generator:
        """A NumPy random generator seeded from ``R2S_SEED``."""
        return np.random.default_rng(self.seed)

    def output_path(self, *parts: str) -> Path:
        """A path inside the output directory, with parent directories created."""
        path = self.output_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_tracks(
        self,
        *,
        fps: float,
        positions: Mapping[str, Any],
        camera: Camera | Mapping[str, Any],
        quaternions: Mapping[str, Any] | None = None,
        visible: Mapping[str, Any] | None = None,
        notes: str | None = None,
    ) -> Path:
        """Write ``meta.json``, ``camera.json`` and ``tracks.npz``; check them; return the dir.

        ``positions`` maps object names to (F, 3) centroid positions in metres.
        """
        arrays: dict[str, np.ndarray] = {}
        frames = None
        for name, value in positions.items():
            array = np.asarray(value, dtype=np.float64)
            if array.ndim != 2 or array.shape[1] != 3:
                raise ValueError(f"positions[{name!r}] must be (F, 3), got {array.shape}")
            frames = len(array) if frames is None else frames
            if len(array) != frames:
                raise ValueError("every object needs the same number of frames")
            arrays[f"{name}.position"] = array
        if frames is None:
            raise ValueError("positions must name at least one object")
        for name, value in (quaternions or {}).items():
            arrays[f"{name}.quaternion"] = np.asarray(value, dtype=np.float64)
        for name, value in (visible or {}).items():
            arrays[f"{name}.visible"] = np.asarray(value, dtype=bool)
        meta: dict[str, Any] = {
            "spec": TRACKS_SPEC,
            "fps": float(fps),
            "num_frames": int(frames),
            "objects": list(positions),
            "frame": {"length_unit": "m", "up_axis": "+z"},
        }
        if notes:
            meta["notes"] = notes
        camera_json = camera.to_json() if isinstance(camera, Camera) else dict(camera)
        self.output_path("meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        self.output_path("camera.json").write_text(json.dumps(camera_json, indent=2) + "\n")
        np.savez(self.output_path("tracks.npz"), **cast(dict[str, Any], arrays))
        problems = check_tracks(self.output_dir)
        if problems:
            raise ValueError("invalid tracks output:\n" + "\n".join(f"  - {p}" for p in problems))
        return self.output_dir

    def write_video(self, frames: Any, fps: float, name: str = "render.mp4") -> Path:
        """Encode (F, H, W, 3) uint8 RGB frames as H.264 with ffmpeg; return the path."""
        return write_video(np.asarray(frames), self.output_path(name), fps)


def write_video(frames: np.ndarray, path: Path, fps: float) -> Path:
    """Encode (F, H, W, 3) uint8 RGB frames at ``path`` as H.264 MP4 (needs ffmpeg)."""
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"frames must be (F, H, W, 3) uint8, got {frames.shape} {frames.dtype}")
    _, height, width, _ = frames.shape
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not on PATH")
    command = [
        ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264",
        "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "1", str(path),
    ]  # fmt: skip
    result = subprocess.run(command, input=frames.tobytes(), capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')}")
    return Path(path)


def check_tracks(output_dir: str | os.PathLike[str]) -> list[str]:
    """Every problem of a ``tracks`` output directory: the checks of the format the Lab runs
    (``real2simlab.formats.tracks``, copied here because this file must not import the Lab;
    a test keeps the two in step). The task's own requirements (its fps, frame count and
    objects) and a ``render.mp4`` are checked by ``r2s twin check``, not here."""
    root = Path(output_dir)
    problems: list[str] = []
    meta = _read_json(root / "meta.json", problems)
    camera = _read_json(root / "camera.json", problems)
    if not (root / "tracks.npz").is_file():
        problems.append("tracks.npz: file is required")
    if problems or meta is None or camera is None:
        return problems
    objects, frames = _meta_problems(meta, problems)
    problems.extend(_camera_problems(camera, frames))
    try:
        with np.load(root / "tracks.npz", allow_pickle=False) as archive:
            arrays: dict[str, NDArray[Any]] = {key: archive[key] for key in archive.files}
    except Exception as error:  # truncated, not a zip, a pickled object: not a valid archive
        return [*problems, f"tracks.npz: cannot read: {type(error).__name__}: {error}"]
    problems.extend(_array_problems(arrays, objects, frames))
    return problems


def _meta_problems(meta: dict[str, Any], problems: list[str]) -> tuple[list[str], int | None]:
    for key in sorted(set(meta) - META_FIELDS):
        problems.append(f"meta.json: unknown field {key!r}")
    if meta.get("spec") != TRACKS_SPEC:
        problems.append(f"meta.json: spec must be {TRACKS_SPEC!r}")
    fps: Any = meta.get("fps")
    if not _is_number(fps) or fps <= 0:
        problems.append("meta.json: fps must be a positive number")
    raw_frames: Any = meta.get("num_frames")
    frames: int | None = None
    if isinstance(raw_frames, int) and not isinstance(raw_frames, bool) and raw_frames >= 1:
        frames = raw_frames
    else:
        problems.append("meta.json: num_frames must be a positive integer")
    raw_objects: Any = meta.get("objects")
    objects: list[str] = []
    if not isinstance(raw_objects, list) or not raw_objects:
        problems.append("meta.json: objects must be a non-empty list of names")
    else:
        for item in cast(list[Any], raw_objects):
            if not isinstance(item, str) or not OBJECT_NAME.match(item):
                problems.append(f"meta.json: invalid object name {item!r}")
            elif item in objects:
                problems.append(f"meta.json: object {item!r} is listed twice")
            else:
                objects.append(item)
    if meta.get("frame", FRAME) != FRAME:
        problems.append('meta.json: frame must be {"length_unit": "m", "up_axis": "+z"}')
    return objects, frames


def _camera_problems(camera: dict[str, Any], frames: int | None) -> list[str]:
    unknown = sorted(set(camera) - CAMERA_FIELDS)
    problems = [f"camera.json: unknown field {key!r}" for key in unknown]
    for key in ("width", "height"):
        value: Any = camera.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            problems.append(f"camera.json: {key} must be a positive integer")
    try:
        k = np.asarray(camera.get("intrinsics"), dtype=np.float64)
        if k.shape != (3, 3) or not np.all(np.isfinite(k)):
            raise ValueError(f"intrinsics must be a finite 3x3 matrix, got shape {k.shape}")
        if k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError("intrinsics focal lengths must be positive")
        if not np.allclose(k[2], (0.0, 0.0, 1.0), atol=TOLERANCE):
            raise ValueError("intrinsics last row must be [0, 0, 1]")
    except (ValueError, TypeError, OverflowError) as error:  # 10**400: beyond float
        problems.append(f"camera.json: {error}")
    try:
        t = np.asarray(camera.get("extrinsics"), dtype=np.float64)
        if t.shape == (4, 4):
            t = t[None]
        if t.ndim != 3 or t.shape[1:] != (4, 4) or not np.all(np.isfinite(t)):
            raise ValueError("extrinsics must be a finite 4x4 matrix or a list of them")
        if frames is not None and len(t) not in (1, frames):
            raise ValueError(f"extrinsics has {len(t)} matrices for {frames} frames")
        for index, matrix in enumerate(t):
            if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=TOLERANCE):
                raise ValueError(f"extrinsics[{index}]: extrinsics last row must be [0, 0, 0, 1]")
            rotation = matrix[:3, :3]
            if (
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=TOLERANCE)
                or np.linalg.det(rotation) <= 0
            ):
                raise ValueError(
                    f"extrinsics[{index}]: extrinsics rotation block must be a proper rotation "
                    "matrix"
                )
    except (ValueError, TypeError, OverflowError) as error:  # 10**400: beyond float
        problems.append(f"camera.json: {error}")
    return problems


def _array_problems(
    arrays: dict[str, NDArray[Any]], objects: list[str], num_frames: int | None
) -> list[str]:
    known = {f"{o}.{part}" for o in objects for part in ("position", "quaternion", "visible")}
    problems = [
        f"tracks.npz: unexpected array {key!r} (objects are listed in meta.json)"
        for key in sorted(set(arrays) - known)
    ]
    for name in objects:
        position = arrays.get(f"{name}.position")
        if position is None:
            problems.append(f"tracks.npz: {name}.position is missing")
            continue
        frames = num_frames if num_frames is not None else (len(position) if position.ndim else 0)
        if position.shape != (frames, 3) or position.dtype.kind != "f":
            problems.append(
                f"tracks.npz: {name}.position must be float ({frames}, 3), "
                f"got {position.dtype} {position.shape}"
            )
            continue
        visible = arrays.get(f"{name}.visible")
        mask: NDArray[np.bool_] = np.ones(frames, dtype=bool)
        if visible is not None:
            if visible.shape != (frames,) or visible.dtype != np.bool_:
                problems.append(f"tracks.npz: {name}.visible must be bool ({frames},)")
            else:
                mask = visible
        if not np.all(np.isfinite(position[mask])):
            problems.append(
                f"tracks.npz: {name}.position has non-finite values where the object is "
                "visible (mark those frames with <name>.visible = false)"
            )
        quaternion = arrays.get(f"{name}.quaternion")
        if quaternion is None:
            continue
        if quaternion.shape != (frames, 4) or quaternion.dtype.kind != "f":
            problems.append(f"tracks.npz: {name}.quaternion must be float ({frames}, 4)")
        elif not np.all(np.isfinite(quaternion[mask])):
            problems.append(
                f"tracks.npz: {name}.quaternion has non-finite values where the object is "
                "visible (a rotation is a finite unit quaternion)"
            )
        elif mask.any() and _not_unit(quaternion[mask]):
            problems.append(f"tracks.npz: {name}.quaternion must have unit norm")
    return problems


def _not_unit(quaternions: NDArray[Any]) -> bool:
    """Whether any quaternion's norm is more than 1e-3 from 1, compared at the array's own
    precision as the Lab compares (rounding to float64 first can accept a long double over it)."""
    norms: NDArray[Any] = np.linalg.norm(quaternions, axis=1)
    return bool(np.max(np.abs(norms - 1)) > 1e-3)


def _read_json(path: Path, problems: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        problems.append(f"{path.name}: file is required")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        problems.append(f"{path.name}: cannot read JSON: {error}")
        return None
    if not isinstance(data, dict):
        problems.append(f"{path.name}: must be a JSON object")
        return None
    return cast(dict[str, Any], data)


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _differences(a: Path, b: Path) -> list[str]:
    """The files in which two outputs differ: arrays compared exactly (NaN equal to NaN), JSON
    by value, videos not here (``r2s twin check`` compares their frames), the rest by bytes."""
    names_a, names_b = _files(a), _files(b)
    found = [f"{name}: written by one run only" for name in sorted(names_a ^ names_b)]
    for name in sorted(names_a & names_b):
        if name.endswith(".mp4"):
            continue
        if name.endswith(".npz"):
            with (
                np.load(a / name, allow_pickle=False) as x,
                np.load(b / name, allow_pickle=False) as y,
            ):
                same = set(x.files) == set(y.files) and all(
                    np.array_equal(x[k], y[k], equal_nan=x[k].dtype.kind in "fc") for k in x.files
                )
        elif name.endswith(".json"):
            same = json.loads((a / name).read_text()) == json.loads((b / name).read_text())
        else:
            same = (a / name).read_bytes() == (b / name).read_bytes()
        if not same:
            found.append(f"{name}: different")
    return found


def is_tracks_output(output_dir: Path) -> bool:
    """Whether ``output_dir`` is meant as a tracks output: it has ``tracks.npz``, or both
    ``meta.json`` and ``camera.json`` (a video's optional ``meta.json`` comes without a camera,
    and a 4D world's ``camera.json`` without ``meta.json``)."""
    return (output_dir / "tracks.npz").exists() or all(
        (output_dir / name).exists() for name in ("meta.json", "camera.json")
    )


def selftest(program_dir: Path, entry: Sequence[str]) -> int:
    """Run the program twice like the Lab does and report problems; return an exit status."""
    program_dir = Path(program_dir).resolve()
    with tempfile.TemporaryDirectory(prefix="r2s-selftest-") as tmp:
        base = Path(tmp)

        def run(label: str) -> Path | None:
            out = base / label / "output"
            out.mkdir(parents=True)
            env = {**os.environ, OUTPUT_DIR_ENV: str(out), SEED_ENV: "0"}
            result = subprocess.run(
                list(entry),
                cwd=program_dir,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                print(f"[{label}] exit status {result.returncode}\n{result.stderr[-2000:]}")
                return None
            if not _files(out):
                print(f"[{label}] the program wrote nothing into ${OUTPUT_DIR_ENV}")
                return None
            problems = check_tracks(out) if is_tracks_output(out) else []
            for problem in problems:
                print(f"[{label}] {problem}")
            return None if problems else out

        first, second = run("run"), run("rerun")
        if first is None or second is None:
            return 1
        try:
            differences = _differences(first, second)
        except (OSError, ValueError) as error:
            differences = [f"the outputs cannot be compared: {error}"]
        if differences:
            print("[rerun] two runs with the same seed gave different outputs:")
            for difference in differences:
                print(f"  {difference}")
            return 1
        videos = sorted(name for name in _files(first) if name.endswith(".mp4"))
        if videos:
            print(f"[rerun] {', '.join(videos)}: not compared here; r2s twin check compares them")
        if is_tracks_output(first) and videos:  # the video is the Lab's to read: not said valid
            print(
                "[rerun] ok: valid tracks, deterministic; "
                f"{', '.join(videos)} not checked here: r2s twin check validates it"
            )
        elif is_tracks_output(first):
            print("[rerun] ok: valid and deterministic")
        else:  # a video or a 4D world: this file checks only the tracks format
            print(
                "[rerun] ok: deterministic; not a tracks output, so not checked here: "
                "r2s twin check validates it"
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="r2s_twin", description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="check a tracks output directory")
    check.add_argument("output_dir")
    test = sub.add_parser("selftest", help="run a program as the Lab does and check it")
    test.add_argument("program_dir")
    test.add_argument(
        "--entry", default=f"{Path(sys.executable).name} run.py", help="the command to run"
    )
    args = parser.parse_args(argv)
    if args.command == "check":
        problems = check_tracks(args.output_dir)
        for problem in problems:
            print(problem)
        print("OK" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    if args.command == "selftest":
        entry = args.entry.split()
        if entry and entry[0] in {"python", "python3"}:
            entry[0] = sys.executable
        return selftest(Path(args.program_dir), entry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
