"""Whether two tasks hold the same recording: the check that keeps training data apart from
evaluation tasks.

Two tasks overlap when they share a media file, or when a video of one shows the same images
as a video of the other, in the same order: a copy, a re-encoding, a resized copy, or a clip
cut from the other (the 27 Physics-IQ recordings inside PhysCodeGym are such clips). Every
video in ``visible/`` and ``hidden/`` counts, so a training video that holds an evaluation
task's hidden reference overlaps it too.

How videos are compared. Each frame is reduced to a 32 x 32 colour thumbnail (the mean of
each cell, by ffmpeg). Two frames show the same image when no cell differs by more than
``CLOSE`` levels in any colour once the median difference (a change of brightness or colour
balance) is removed. Only *moving* frames count as evidence: frames that differ from the frame
0.1 s before them. Two events filmed by one camera share their still frames (a ball at rest,
an empty table), so still frames prove nothing. Two videos show the same recording when, at
one time offset, the moving frames of one that fall inside the other number at least
``MIN_FRAMES`` and at least ``COVER`` of the moving frames of the video with fewer, and at
least ``SHARE`` of them show the same image as the frame of the other at the same time. A
video with fewer moving frames than ``MIN_FRAMES`` is compared by all its frames, so two still
videos of the same view overlap.

What it does not find: crops, mirrored or rotated copies, heavily degraded copies (a 320 x 240
video re-encoded at CRF 40) and, in general, a second recording of the same event. What it
finds besides copies: renderings so alike that a 32 x 32 thumbnail cannot tell them apart (a
pendulum released 9 % higher), and a second take of a nearly still scene. Neither may be used
to train on a task that is evaluated.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import warnings
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from real2simlab._fs import (
    DamagedCacheWarning,
    durable_replace,
    force_remove,
    held_path,
    remove_abandoned,
)
from real2simlab.errors import SetupError
from real2simlab.media import READ_FIX, MediaError, is_video_file, probe_video, tool
from real2simlab.task.io import HIDDEN_DIR, VISIBLE_DIR, Task
from real2simlab.twin.fingerprint import sha256_file

SIZE = 32
"""Thumbnail side, in cells."""
CLOSE = 20
"""Largest difference of one cell (levels of one colour, 0 to 255) between two frames that show
the same image, after removing the median difference."""
MOTION_S = 0.1
"""A frame is moving when it differs from the frame this many seconds before it."""
MIN_FRAMES = 8
"""Moving frames two videos must share, at least."""
SHARE = 0.8
"""Share of the compared moving frames that must show the same image."""
COVER = 0.5
"""Share of the moving frames of the video with fewer of them that must be compared: two
events filmed by one camera can look alike for a moment (a ball at rest), not for long."""
MAX_EVIDENCE = 48
"""Moving frames of a video compared to find time offsets (evenly spaced among all of them)."""
CHUNK = 64
"""Frames of the other video compared at once."""
STEPS = (0,)
"""Frames of the other video compared with one frame, around the one at the same time."""
RECORDING_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".exr",
     ".npy", ".npz", ".h5", ".hdf5", ".ply", ".pcd"}
)  # fmt: skip
"""Files compared by content besides videos: images, arrays, point clouds. Programs and
configuration are not: unrelated tasks often share them (a truth generator's code)."""
VERSION = "1"
"""Version of the thumbnails; part of every cache key."""


@dataclass(frozen=True)
class VideoPrint:
    """One video, reduced: its file's digest, frame rate and per-frame thumbnails."""

    name: str
    """Path relative to the task directory."""
    digest: str
    fps: float
    thumbs: NDArray[np.uint8]
    """``(frames, SIZE, SIZE, 3)`` RGB thumbnails."""
    moving: NDArray[np.intp] = field(init=False, compare=False)
    """Indices of the moving frames (all frames when too few move)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "moving", _moving(self.thumbs, self.fps))


@dataclass(frozen=True)
class TaskPrint:
    """What the overlap check compares of one task."""

    task_id: str
    files: dict[str, str]
    """Digest → path relative to the task directory, of its videos and other recordings."""
    videos: tuple[VideoPrint, ...]


@dataclass(frozen=True)
class Match:
    """Two videos that show the same recording."""

    offset_s: float
    """Time of the second video minus time of the first, for the same image."""
    same: int
    compared: int


@dataclass(frozen=True)
class Overlap:
    """Why a task overlaps another one."""

    task_id: str
    """The other task."""
    reason: str

    def to_json(self) -> dict[str, str]:
        return {"task_id": self.task_id, "reason": self.reason}


# --- thumbnails ------------------------------------------------------------------------------


def thumbnails(path: Path) -> tuple[float, NDArray[np.uint8]]:
    """The frame rate and ``(frames, SIZE, SIZE, 3)`` RGB thumbnails of the video at ``path``."""
    info = probe_video(path)
    command = [
        tool("ffmpeg"), "-v", "error", "-i", str(path),
        "-vf", f"scale={SIZE}:{SIZE}:flags=area,format=rgb24",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]  # fmt: skip
    result = subprocess.run(
        command, capture_output=True, check=False, stdin=subprocess.DEVNULL, timeout=600
    )
    if result.returncode != 0:
        raise MediaError(
            f"ffmpeg cannot decode {path}: {result.stderr.decode().strip()}", fix=READ_FIX
        )
    cell = SIZE * SIZE * 3
    count = len(result.stdout) // cell
    if count == 0 or info.fps <= 0:
        raise MediaError(f"{path} has no frames", fix=READ_FIX)
    data = np.frombuffer(result.stdout[: count * cell], dtype=np.uint8)
    return info.fps, data.reshape(count, SIZE, SIZE, 3).copy()


def _cached_thumbnails(
    path: Path, digest: str, cache: Path | None
) -> tuple[float, NDArray[np.uint8]]:
    if cache is None:
        return thumbnails(path)
    stored = cache / f"{digest}-v{VERSION}.npz"
    if stored.is_file():
        try:  # what was written there, or damaged (another program's file, a single array)
            with np.load(stored, allow_pickle=False) as data:
                fps, thumbs = float(data["fps"]), data["thumbs"]
            if (
                np.isfinite(fps)
                and fps > 0
                and thumbs.dtype == np.uint8
                and thumbs.ndim == 4
                and thumbs.shape[1:] == (SIZE, SIZE, 3)
                and len(thumbs) > 0
            ):
                return fps, thumbs
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
            pass
        warnings.warn(
            f"the cached thumbnails {stored} are damaged; computing them again",
            DamagedCacheWarning,
            stacklevel=2,
        )
        stored.unlink(missing_ok=True)
    fps, thumbs = thumbnails(path)
    cache.mkdir(parents=True, exist_ok=True)
    # what a command killed outright left is removed (not the file of one still writing it)
    remove_abandoned(cache, re.escape(f".{stored.stem}."), r"\.partial\.npz")
    partial = held_path(lambda part: cache / f".{stored.stem}.{part}.partial.npz", directory=False)
    try:
        np.savez_compressed(partial, fps=np.array(fps), thumbs=thumbs)
        durable_replace(partial, stored)
    finally:
        force_remove(partial)  # moved into place, or a failed write's
    return fps, thumbs


def default_cache() -> Path:
    """Where thumbnails are kept, by file digest: the Lab's cache."""
    from real2simlab.vendor import cache_dir

    return cache_dir() / "thumbnails"


def _distance(a: NDArray[np.int16], b: NDArray[np.int16]) -> NDArray[np.int16]:
    """Largest cell difference after removing the median difference, of frames ``a[..., i]``
    against ``b[..., i]`` (broadcast over the leading axes)."""
    diff = (a - b).reshape(*np.broadcast_shapes(a.shape, b.shape)[:-3], SIZE * SIZE, 3)
    centre = np.median(diff, axis=-2, keepdims=True).astype(np.int16)
    return np.abs(diff - centre).max(axis=(-2, -1))


def _moving(thumbs: NDArray[np.uint8], fps: float) -> NDArray[np.intp]:
    lag = max(1, round(MOTION_S * fps))
    frames = thumbs.astype(np.int16)
    if len(frames) <= lag:
        return np.arange(len(frames))
    moving = np.flatnonzero(_distance(frames[lag:], frames[:-lag]) > CLOSE) + lag
    return moving if len(moving) >= MIN_FRAMES else np.arange(len(frames))


# --- comparing two videos --------------------------------------------------------------------


def _sample(moving: NDArray[np.intp]) -> NDArray[np.intp]:
    """At most ``MAX_EVIDENCE`` of ``moving``, evenly spaced."""
    count = min(MAX_EVIDENCE, len(moving))
    return moving[np.linspace(0, len(moving) - 1, count).astype(np.intp)]


def _aligned(a: VideoPrint, b: VideoPrint, offset: float) -> tuple[int, int]:
    """How many moving frames of ``a`` that fall inside ``b`` at ``offset`` show the same image
    as ``b``'s frame at that time (or one frame either side), and how many fall inside."""
    times = a.moving / a.fps + offset
    index = np.rint(times * b.fps).astype(np.intp)
    inside = (index >= 0) & (index < len(b.thumbs))
    if not inside.any():
        return 0, 0
    frames = a.thumbs[a.moving[inside]].astype(np.int16)
    best = np.full(len(frames), np.iinfo(np.int16).max, dtype=np.int16)
    for step in STEPS:
        near = np.clip(index[inside] + step, 0, len(b.thumbs) - 1)
        best = np.minimum(best, _distance(frames, b.thumbs[near].astype(np.int16)))
    return int((best <= CLOSE).sum()), int(inside.sum())


def _offsets(a: VideoPrint, b: VideoPrint) -> list[float]:
    """Candidate offsets: from pairs of a sample of ``a``'s moving frames and any frame of
    ``b`` that show the same image, the most frequent first."""
    sample = _sample(a.moving)
    frames = a.thumbs[sample].astype(np.int16)[:, None]
    pairs: list[NDArray[np.intp]] = []
    for start in range(0, len(b.thumbs), CHUNK):  # bounded memory, whatever the length
        other = b.thumbs[start : start + CHUNK].astype(np.int16)[None]
        found = np.argwhere(_distance(frames, other) <= CLOSE)
        found[:, 1] += start
        pairs.append(found)
    same = np.concatenate(pairs)
    if len(same) == 0:
        return []
    step = 0.5 / max(a.fps, b.fps)
    offsets = np.rint((same[:, 1] / b.fps - sample[same[:, 0]] / a.fps) / step).astype(np.int64)
    values, counts = np.unique(offsets, return_counts=True)
    return [float(v * step) for v in values[np.argsort(-counts, kind="stable")][:5]]


def _prefilter(a: VideoPrint, b: VideoPrint) -> bool:
    """A cheap test that is passed by every pair :func:`same_recording` can match: some moving
    frame of ``a`` is within twice ``CLOSE`` of some frame of ``b`` at 4 x 4 cells."""

    def coarse(thumbs: NDArray[np.uint8]) -> NDArray[np.int16]:
        shaped = thumbs.astype(np.float32).reshape(len(thumbs), 4, SIZE // 4, 4, SIZE // 4, 3)
        return shaped.mean(axis=(2, 4)).round().astype(np.int16).reshape(len(thumbs), 16, 3)

    evidence, other = coarse(a.thumbs[_sample(a.moving)]), coarse(b.thumbs)
    for start in range(0, len(other), CHUNK * 16):  # bounded memory for long videos
        diff = evidence[:, None] - other[None, start : start + CHUNK * 16]
        centre = np.median(diff, axis=-2, keepdims=True)
        if (np.abs(diff - centre).max(axis=(-2, -1)) <= 2 * CLOSE).any():
            return True
    return False


def _one_way(a: VideoPrint, b: VideoPrint) -> Match | None:
    if not _prefilter(a, b):
        return None
    for offset in _offsets(a, b):
        same, compared = _aligned(a, b, offset)
        enough = max(min(MIN_FRAMES, len(a.moving)), COVER * min(len(a.moving), len(b.moving)))
        if compared >= enough and same >= enough and same >= SHARE * compared:
            return Match(offset, same, compared)
    return None


def same_recording(a: VideoPrint, b: VideoPrint) -> Match | None:
    """How ``a`` and ``b`` show the same recording, or ``None``: either is (nearly) contained
    in the other, at one time offset (see the module docstring)."""
    if a.digest == b.digest:
        return Match(0.0, len(a.moving), len(a.moving))
    found = _one_way(a, b)
    if found is not None:
        return found
    back = _one_way(b, a)
    return None if back is None else Match(-back.offset_s, back.same, back.compared)


# --- tasks -----------------------------------------------------------------------------------


def recording_files(root: Path) -> Iterator[Path]:
    """The files a print holds of the task directory at ``root`` (or of a copy of its
    ``visible/`` and ``hidden/``, as in a dataset): its videos and other recordings
    (``RECORDING_SUFFIXES``), regular files only, in path order."""
    for part in (VISIBLE_DIR, HIDDEN_DIR):
        directory = Path(root) / part
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            if is_video_file(path) or path.suffix.lower() in RECORDING_SUFFIXES:
                yield path


def _media_files(task: Task) -> Iterator[Path]:
    return recording_files(task.root)


def recordings_changed(task: Task, item: TaskPrint) -> bool:
    """Whether the recordings of ``task`` are no longer those its print ``item`` was taken
    from (one changed, added or removed since), or can no longer be read."""
    try:
        return {sha256_file(path) for path in recording_files(task.root)} != set(item.files)
    except OSError:
        return True


def task_print(task: Task, *, cache: Path | None = None) -> TaskPrint:
    """The overlap print of ``task``: digests of its videos and other recordings, and
    thumbnails of its videos, kept in ``cache`` by file digest (decoding is the slow part)."""
    files: dict[str, str] = {}
    videos: list[VideoPrint] = []
    for path in _media_files(task):
        digest = sha256_file(path)
        name = path.relative_to(task.root).as_posix()
        files.setdefault(digest, name)
        if is_video_file(path):
            fps, thumbs = _cached_thumbnails(path, digest, cache)
            videos.append(VideoPrint(name, digest, fps, thumbs))
    return TaskPrint(task.id, files, tuple(videos))


class OverlapIndex:
    """Prints of a set of tasks, to find which of them another task overlaps."""

    def __init__(self, prints: Iterable[TaskPrint] = ()) -> None:
        self.prints: list[TaskPrint] = []
        self._files: dict[str, tuple[str, str]] = {}
        for item in prints:
            self.add(item)

    def __len__(self) -> int:
        return len(self.prints)

    def add(self, item: TaskPrint) -> None:
        self.prints.append(item)
        for digest, name in item.files.items():
            self._files.setdefault(digest, (item.task_id, name))

    def find(self, item: TaskPrint) -> Overlap | None:
        """The first task of the index that ``item`` overlaps, and why; ``None`` if none."""
        for digest, name in item.files.items():
            if digest in self._files:
                other, other_name = self._files[digest]
                return Overlap(other, f"{name} is the same file as {other}'s {other_name}")
        for video in item.videos:
            for other in self.prints:
                for candidate in other.videos:
                    match = same_recording(video, candidate)
                    if match is not None:
                        return Overlap(
                            other.task_id,
                            f"{video.name} shows the same recording as {other.task_id}'s "
                            f"{candidate.name} ({match.same} of {match.compared} moving frames "
                            f"match, {match.offset_s:+.2f} s apart)",
                        )
        return None

    # --- storage ---------------------------------------------------------------------------

    def save(self, path: Path) -> Path:
        """Write the index to ``path`` (one ``.npz`` file)."""
        arrays: dict[str, Any] = {"index": None}
        meta: list[dict[str, object]] = []
        for i, item in enumerate(self.prints):
            videos: list[dict[str, object]] = []
            for k, video in enumerate(item.videos):
                arrays[f"t{i}v{k}"] = video.thumbs
                videos.append({"name": video.name, "digest": video.digest, "fps": video.fps})
            meta.append({"task_id": item.task_id, "files": item.files, "videos": videos})
        index = json.dumps({"version": VERSION, "tasks": meta}).encode()
        arrays["index"] = np.frombuffer(index, dtype=np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: Path, *, fix: str = "export the dataset again") -> OverlapIndex:
        """The index saved at ``path``; ``SetupError`` with ``fix`` if it is missing, damaged or
        written by another version of the overlap check."""
        if not path.is_file():
            raise SetupError(f"{path} does not exist", fix=fix)
        try:
            with np.load(path) as data:
                index = json.loads(data["index"].tobytes())
                if index.get("version") != VERSION:
                    raise SetupError(
                        f"{path} was written by another version of the overlap check", fix=fix
                    )
                prints = [
                    TaskPrint(
                        entry["task_id"],
                        dict(entry["files"]),
                        tuple(
                            VideoPrint(v["name"], v["digest"], float(v["fps"]), data[f"t{i}v{k}"])
                            for k, v in enumerate(entry["videos"])
                        ),
                    )
                    for i, entry in enumerate(index["tasks"])
                ]
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
            raise SetupError(f"{path} is damaged: {error}", fix=fix) from error
        return cls(prints)

    def digest(self) -> str:
        """A digest of the tasks and files indexed."""
        digest = hashlib.sha256(f"r2s-overlap/{VERSION}\n".encode())
        for item in sorted(self.prints, key=lambda p: (p.task_id, sorted(p.files))):
            digest.update(json.dumps([item.task_id, sorted(item.files)]).encode() + b"\n")
        return digest.hexdigest()
