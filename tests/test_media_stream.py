"""Decoding long or large videos with bounded memory: ``iter_frames``, and the adapters that
use it (PhysCodeGym's and IRIS's 4K recordings)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from real2simlab import media
from real2simlab.media import (
    MediaError,
    downscale,
    iter_frames,
    read_downscaled,
    read_frames,
    write_downscaled,
    write_video,
)


@pytest.fixture
def video(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (30, 48, 64, 3), dtype=np.uint8)
    return write_video(frames, tmp_path / "clip.mp4", fps=10.0)


def test_streamed_frames_are_read_frames(video: Path) -> None:
    whole = read_frames(video)
    assert np.array_equal(np.stack(list(iter_frames(video))), whole)
    assert np.array_equal(np.stack(list(iter_frames(video, start=5, stop=12))), whole[5:12])
    assert np.array_equal(np.stack(list(iter_frames(video, start=25, stop=99))), whole[25:])


def test_streaming_stops_decoding_at_stop(video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the frames up to ``stop`` are read from ffmpeg, which is then stopped."""
    processes: list[subprocess.Popen[bytes]] = []
    original = subprocess.Popen

    def recording(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = original(*args, **kwargs)  # type: ignore[call-overload]
        if "rawvideo" in args[0]:  # the decoder (ffprobe runs through Popen too)
            processes.append(process)
        return process

    monkeypatch.setattr(media.subprocess, "Popen", recording)
    frames = iter_frames(video, start=0, stop=2)
    assert len(list(frames)) == 2
    (process,) = processes
    assert process.returncode is not None and process.stdout is not None and process.stdout.closed
    partial = iter_frames(video)
    next(partial)
    partial.close()  # the caller stops early: ffmpeg is stopped too
    assert processes[-1].returncode is not None


def test_a_killed_decoder_is_not_reported_as_a_bad_file(
    video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the system kills ffmpeg (out of memory), the error says so."""
    killer = tmp_path / "ffmpeg"
    killer.write_text("#!/bin/sh\nkill -9 $$\n")
    killer.chmod(0o755)
    monkeypatch.setenv("R2S_FFMPEG", str(killer))
    for decode in (lambda: list(iter_frames(video)), lambda: read_frames(video)):
        with pytest.raises(MediaError, match=r"killed .*lack of memory"):
            decode()


def test_downscaled_stream_equals_downscaling_the_whole_video(video: Path) -> None:
    """Bit for bit what the adapters computed before, from the whole decoded video."""
    whole = read_frames(video)
    assert np.array_equal(read_downscaled(video, 16, 12), downscale(whole, 16, 12))
    assert np.array_equal(
        read_downscaled(video, 21, 16, start=4, stop=19), downscale(whole[4:19], 21, 16)
    )
    with pytest.raises(MediaError, match="no frames"):
        read_downscaled(video, 16, 12, start=40)


def test_a_video_is_downscaled_and_encoded_frame_by_frame(video: Path, tmp_path: Path) -> None:
    """``write_downscaled`` writes what downscaling the whole video and encoding it writes,
    holding one frame at a time; a source it cannot read is an error, not a short video."""
    whole = downscale(read_frames(video), 16, 12)
    count = write_downscaled(video, tmp_path / "small.mp4", 16, 12, fps=10.0)
    expected = read_frames(write_video(whole, tmp_path / "whole.mp4", fps=10.0))
    assert count == 30 and np.array_equal(read_frames(tmp_path / "small.mp4"), expected)
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(video.read_bytes()[:200])
    with pytest.raises(MediaError):
        write_downscaled(broken, tmp_path / "out.mp4", 16, 12, fps=10.0)
    with pytest.raises(MediaError, match="even"):
        write_downscaled(video, tmp_path / "odd.mp4", 15, 12, fps=10.0)


def test_adapters_decode_4k_recordings_frame_by_frame(
    video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PhysCodeGym and IRIS never hold a whole recording; IRIS decodes only its clip."""
    from real2simlab.benchmarks import iris, physcodegym

    for module in (iris, physcodegym):
        assert not hasattr(module, "read_frames"), module.__name__
    assert not hasattr(physcodegym, "write_video")  # it streams: write_downscaled
    calls: list[tuple[int, int | None]] = []

    class Stop(Exception):
        pass

    def spy(path: Path, width: int, height: int, *, start: int = 0, stop: int | None = None):
        calls.append((start, stop))
        raise Stop

    monkeypatch.setattr(iris, "read_downscaled", spy)
    monkeypatch.setattr(iris, "download_video", lambda setting, take: video)
    drop = SimpleNamespace(
        setting="drop_50",
        take=1,
        observations=SimpleNamespace(width=16, height=12, fps=10.0),
        trajectory=SimpleNamespace(release_frame=7),
    )
    with pytest.raises(Stop):
        iris.visible_capture(drop, video.parent, "iris/dropping_ball/drop_050")  # type: ignore[arg-type]
    assert calls == [(7, 7 + iris.FRAMES)]


# --- encoding where this machine has no room ---------------------------------------------------


def _ffmpeg_under_a_file_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ffmpeg as it runs under ``ulimit -f``: once its output passes the limit the kernel kills
    it (SIGXFSZ), and it says nothing."""
    import shlex

    real = media.tool("ffmpeg")
    limited = tmp_path / "bin" / "ffmpeg"
    limited.parent.mkdir()
    limited.write_text(f'#!/bin/sh\nulimit -f 8\nexec {shlex.quote(real)} "$@"\n')
    limited.chmod(0o755)
    monkeypatch.setenv("R2S_FFMPEG", str(limited))


@pytest.mark.parametrize("writer", ["write_video", "write_downscaled"])
def test_an_encode_stopped_by_the_file_size_limit_says_so_and_keeps_the_old_video(
    video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
) -> None:
    """ffmpeg killed by the file-size limit fails with that reason (and this machine's storage
    error as its cause, which the command line turns into the fix); the video that was there
    is kept whole, and no partial file is left beside it."""
    import errno

    target = tmp_path / "out" / "clip.mp4"
    write_video(np.zeros((2, 16, 16, 3), np.uint8), target, fps=5.0)
    before = target.read_bytes()
    _ffmpeg_under_a_file_size_limit(tmp_path, monkeypatch)
    noise = np.random.default_rng(1).integers(0, 256, (30, 64, 64, 3), dtype=np.uint8)
    with pytest.raises(
        MediaError, match=r"cannot write .*clip\.mp4: the file-size limit"
    ) as raised:
        if writer == "write_video":
            write_video(noise, target, fps=10.0)
        else:
            write_downscaled(video, target, 64, 48, fps=10.0)
    cause = raised.value.__cause__
    assert isinstance(cause, OSError) and cause.errno == errno.EFBIG, repr(cause)
    assert target.read_bytes() == before
    assert [p.name for p in target.parent.iterdir()] == ["clip.mp4"]


@pytest.mark.parametrize(
    ("said", "reason", "code"),
    [
        ("No space left on device", "the disk is full", "ENOSPC"),
        ("Disk quota exceeded", "the disk quota is reached", "EDQUOT"),
    ],
)
def test_an_encode_on_a_full_disk_says_so_and_keeps_the_old_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: str, reason: str, code: str
) -> None:
    """ffmpeg that could not finish its output for lack of space says so in its own words: the
    error names the cause, and the earlier video is kept."""
    import errno

    fake = tmp_path / "bin" / "ffmpeg"
    fake.parent.mkdir()
    fake.write_text(
        '#!/bin/sh\ncat > /dev/null\nfor last; do :; done\nprintf partial > "$last"\n'
        f'echo "[out#0/mp4 @ 0x1] Error writing trailer: {said}" >&2\nexit 1\n'
    )
    fake.chmod(0o755)
    monkeypatch.setenv("R2S_FFMPEG", str(fake))
    target = tmp_path / "out" / "clip.mp4"
    target.parent.mkdir()
    target.write_bytes(b"the recording that was there")
    with pytest.raises(MediaError, match=reason) as raised:
        write_video(np.zeros((2, 16, 16, 3), np.uint8), target, fps=5.0)
    cause = raised.value.__cause__
    assert isinstance(cause, OSError) and cause.errno == getattr(errno, code), repr(cause)
    assert said in str(raised.value)  # ffmpeg's own words are kept
    assert target.read_bytes() == b"the recording that was there"
    assert [p.name for p in target.parent.iterdir()] == ["clip.mp4"]
