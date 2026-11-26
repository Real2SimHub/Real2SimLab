"""Video reading and writing through ffmpeg and ffprobe."""

from __future__ import annotations

import errno
import json
import os
import shutil
import signal
import subprocess
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from real2simlab.errors import R2SError, SetupError
from real2simlab.sandbox.process import cancelled

FFMPEG_ENV = "R2S_FFMPEG"
FFPROBE_ENV = "R2S_FFPROBE"
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"})
"""File suffixes treated as video files, e.g. by the twin program rules."""


PROBE_TIMEOUT_S = 600
"""How long ffprobe may take to count a video's frames before the video counts as unreadable."""


READ_FIX = (
    "check that the file is a video that ffmpeg can read (`ffprobe FILE` says what is wrong "
    "with it) and that your user may read it, then run the same command again"
)
"""What to do about a video that ffprobe or ffmpeg cannot read."""


class MediaError(R2SError, ValueError):
    """A media file cannot be read or written. ``fix``, when given, says what to do about it;
    the message (``str``) is the problem alone, as records and reports quote it."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.fix = fix


@dataclass(frozen=True)
class VideoInfo:
    """What ffprobe reports about a video stream."""

    fps: float
    num_frames: int
    width: int
    height: int
    codec: str


def tool(name: str) -> str:
    """Path of ``ffmpeg`` or ``ffprobe``; raise ``SetupError`` with a fix if it is missing."""
    override = os.environ.get(FFMPEG_ENV if name == "ffmpeg" else FFPROBE_ENV)
    if override:
        return override
    found = shutil.which(name)
    if found is None:
        raise SetupError(
            f"{name} is not installed or not on PATH",
            fix="install ffmpeg (Debian/Ubuntu: sudo apt-get install ffmpeg; "
            "macOS: brew install ffmpeg; conda: conda install -c conda-forge ffmpeg)",
        )
    return found


def probe_video(path: Path) -> VideoInfo:
    """Frame rate, exact frame count, resolution and codec of the first video stream."""
    command = [
        tool("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_read_frames,width,height,codec_name",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=PROBE_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        raise MediaError(
            f"ffprobe did not finish reading {path} in {PROBE_TIMEOUT_S} s", fix=READ_FIX
        ) from None
    if result.returncode != 0:
        raise MediaError(f"ffprobe cannot read {path}: {result.stderr.strip()}", fix=READ_FIX)
    streams = cast(list[dict[str, Any]], json.loads(result.stdout).get("streams", []))
    if not streams:
        raise MediaError(f"{path} has no video stream", fix=READ_FIX)
    stream = streams[0]
    rate = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
    fps = float(Fraction(rate)) if rate != "0/0" else 0.0
    if fps <= 0:
        rate = str(stream.get("r_frame_rate", "0/1"))
        fps = float(Fraction(rate)) if rate != "0/0" else 0.0
    return VideoInfo(
        fps=round(fps, 6),
        num_frames=int(stream.get("nb_read_frames", 0)),
        width=int(stream["width"]),
        height=int(stream["height"]),
        codec=str(stream.get("codec_name", "")),
    )


def _decode_command(path: Path) -> list[str]:
    return [
        tool("ffmpeg"),
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]


def _decode_error(path: Path, returncode: int, stderr: bytes) -> MediaError:
    """What a failed ffmpeg decode means: killed by a signal (on Linux usually the kernel's
    out-of-memory killer), or a file it cannot decode."""
    detail = stderr.decode(errors="replace").strip()
    if returncode < 0:
        return MediaError(
            f"ffmpeg was killed (signal {-returncode}) while decoding {path}, most likely for "
            "lack of memory; the file itself may be fine" + (f": {detail}" if detail else ""),
            fix="free memory on this machine (or run fewer commands at once), then run the same "
            "command again",
        )
    return MediaError(f"ffmpeg cannot decode {path}: {detail}", fix=READ_FIX)


def read_frames(path: Path) -> NDArray[np.uint8]:
    """Every frame of ``path`` as an ``(F, H, W, 3)`` RGB uint8 array.

    The whole video is held in memory (a 4K video of 240 frames is about 6 GB): for long or
    large videos use ``iter_frames``.
    """
    info = probe_video(path)
    result = subprocess.run(
        _decode_command(path), capture_output=True, check=False, stdin=subprocess.DEVNULL
    )
    if result.returncode != 0:
        raise _decode_error(path, result.returncode, result.stderr)
    frame_size = info.width * info.height * 3
    count = len(result.stdout) // frame_size
    data = np.frombuffer(result.stdout[: count * frame_size], dtype=np.uint8)
    return data.reshape(count, info.height, info.width, 3)


def iter_frames(
    path: Path, *, start: int = 0, stop: int | None = None
) -> Iterator[NDArray[np.uint8]]:
    """Frames ``start`` to ``stop`` (exclusive) of ``path``, one ``(H, W, 3)`` RGB uint8 array
    at a time, the same frames ``read_frames`` returns: memory for one frame, whatever the
    video's length or size. Decoding stops once ``stop`` is reached."""
    info = probe_video(path)
    frame_size = info.width * info.height * 3
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            _decode_command(path), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors
        )
        assert process.stdout is not None
        try:
            index = 0
            while stop is None or index < stop:
                if cancelled():  # an interrupted experiment: ffmpeg is stopped below
                    raise KeyboardInterrupt
                data = process.stdout.read(frame_size)
                if len(data) < frame_size:  # the end (a partial last frame is dropped)
                    break
                if index >= start:
                    yield np.frombuffer(data, dtype=np.uint8).reshape(info.height, info.width, 3)
                index += 1
            else:
                return  # stopped early: ffmpeg is stopped below, its exit status is moot
            process.stdout.close()
            if process.wait() != 0:
                errors.seek(0)
                raise _decode_error(path, process.returncode, errors.read())
        finally:
            if process.poll() is None:  # stopped early, or the caller stopped iterating
                process.kill()
            process.stdout.close()
            process.wait()


def downscale(frames: NDArray[np.uint8], width: int, height: int) -> NDArray[np.uint8]:
    """``(F, H, W, 3)`` frames reduced to ``width`` x ``height`` by averaging blocks of an
    integer size (the largest that fits; the right and bottom remainders are dropped)."""
    factor_x, factor_y = frames.shape[2] // width, frames.shape[1] // height
    trimmed = frames[:, : height * factor_y, : width * factor_x]
    shaped = trimmed.reshape(len(frames), height, factor_y, width, factor_x, 3)
    return shaped.mean(axis=(2, 4)).round().astype(np.uint8)


def read_downscaled(
    path: Path, width: int, height: int, *, start: int = 0, stop: int | None = None
) -> NDArray[np.uint8]:
    """Frames ``start`` to ``stop`` of ``path`` downscaled to ``width`` x ``height``
    (``downscale``), decoded one frame at a time: memory for the result and one frame, not
    for the whole video at full size (a 4K recording of 240 frames is about 6 GB)."""
    frames = [
        downscale(frame[None], width, height)[0]
        for frame in iter_frames(path, start=start, stop=stop)
    ]
    if not frames:
        raise MediaError(f"{path} has no frames between {start} and {stop}")
    return np.stack(frames)


def write_video(frames: NDArray[np.uint8], path: Path, fps: float, *, crf: int = 18) -> Path:
    """Encode ``(F, H, W, 3)`` RGB uint8 frames as H.264 MP4 at ``path``; return the path.

    Width and height must be even (a constraint of the yuv420p pixel format). An encode that
    fails leaves what ``path`` held as it was (``_encoding``).
    """
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise MediaError(f"frames must be (F, H, W, 3) uint8, got {frames.shape} {frames.dtype}")
    count, height, width, _ = frames.shape
    if count == 0:
        raise MediaError("no frames to write")
    if width % 2 or height % 2:
        raise MediaError(f"width and height must be even, got {width}x{height}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _encoding(path) as partial:
        command = _encode_command(partial, width, height, fps, crf)
        result = subprocess.run(command, input=frames.tobytes(), capture_output=True, check=False)
        if result.returncode != 0:
            raise _encode_error(path, result.returncode, result.stderr)
    return path


def write_downscaled(
    source: Path, target: Path, width: int, height: int, fps: float, *, crf: int = 18
) -> int:
    """``source`` reduced to ``width`` x ``height`` (``downscale``) and encoded as an H.264 MP4
    at ``target``, one frame at a time: memory for one frame at full size, whatever the video's
    length or resolution (a 4K recording of 240 frames is about 6 GB). Returns the number of
    frames written."""
    if width % 2 or height % 2:
        raise MediaError(f"width and height must be even, got {width}x{height}")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _encoding(target) as partial, tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            _encode_command(partial, width, height, fps, crf),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=errors,
        )
        assert process.stdin is not None
        try:
            for frame in iter_frames(source):
                process.stdin.write(downscale(frame[None], width, height)[0].tobytes())
                count += 1
            process.stdin.close()
        except BrokenPipeError:
            pass  # ffmpeg stopped: its exit status and message say why
        finally:
            if process.poll() is None and not process.stdin.closed:
                process.stdin.close()
            returncode = process.wait()
        if returncode != 0:
            errors.seek(0)
            raise _encode_error(target, returncode, errors.read())
        if count == 0:  # nothing to put in the target's place
            raise MediaError(f"{source} has no frames", fix=READ_FIX)
    return count


@contextmanager
def _encoding(path: Path) -> Iterator[Path]:
    """A new file beside ``path`` for ffmpeg to write, put in its place when the block ends
    without an error. An encode that fails or is stopped (a full disk, the file-size limit,
    Ctrl-C) leaves what ``path`` held as it was, and removes what it wrote; the suffix is
    kept: it tells ffmpeg the container."""
    partial = path.with_name(f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}{path.suffix}")
    try:
        yield partial
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


STORAGE_SAYS = {
    errno.ENOSPC: ("No space left on device",),
    errno.EDQUOT: ("Disk quota exceeded", "Disc quota exceeded", "Quota exceeded"),
    errno.EFBIG: ("File too large",),
}
"""How ffmpeg (through the C library's ``strerror``: glibc, musl, macOS) says that this
machine's storage refused its output."""

STORAGE_REASONS = {
    errno.ENOSPC: "the disk is full",
    errno.EDQUOT: "the disk quota is reached",
    errno.EFBIG: "the file-size limit (`ulimit -f`) is reached",
}


def _encode_error(path: Path, returncode: int, stderr: bytes) -> MediaError:
    """What a failed ffmpeg encode of ``path`` means. This machine's storage (a full disk, a
    quota, the file-size limit, at which the kernel kills ffmpeg with SIGXFSZ before it says
    anything) is said as such, with the ``OSError`` it amounts to as the cause, which the
    command line answers with what to do; anything else in ffmpeg's own words."""
    detail = stderr.decode(errors="replace").strip()
    code = next(
        (code for code, says in STORAGE_SAYS.items() if any(text in detail for text in says)),
        None,
    )
    limit = getattr(signal, "SIGXFSZ", None)  # POSIX
    if limit is not None and returncode == -limit:
        code = errno.EFBIG
    if code is None:
        killed = f"ffmpeg was killed (signal {-returncode})" if returncode < 0 else ""
        return MediaError(f"ffmpeg cannot write {path}: {detail or killed}")
    said = "; ".join(line.strip() for line in detail.splitlines() if line.strip())
    error = MediaError(
        f"ffmpeg cannot write {path}: {STORAGE_REASONS[code]}" + (f" ({said})" if said else "")
    )
    error.__cause__ = OSError(code, os.strerror(code), str(path))
    return error


def _encode_command(path: Path, width: int, height: int, fps: float, crf: int) -> list[str]:
    return [
        tool("ffmpeg"),
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-movflags",
        "+faststart",
        str(path),
    ]


def is_video_file(path: Path) -> bool:
    """Whether ``path`` has a video file suffix."""
    return path.suffix.lower() in VIDEO_SUFFIXES
