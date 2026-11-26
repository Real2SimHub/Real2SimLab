"""The ``video`` format: only a rendered video, for methods such as video models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, cast

from real2simlab.errors import host_failure
from real2simlab.formats.base import OutputFormat, Requirements
from real2simlab.media import MediaError, probe_video

RENDER_FILE = "render.mp4"
META_FILE = "meta.json"


class VideoFormat(OutputFormat):
    """``render.mp4`` and an optional ``meta.json`` with free-form notes."""

    name: ClassVar[str] = "video"
    summary: ClassVar[str] = "A rendered video only, e.g. from a video world model."
    content: ClassVar[tuple[str, ...] | None] = (RENDER_FILE,)
    """The video; ``meta.json`` holds free-form notes, and a scene renderer's working files
    (a ``.blend`` file, frame images) are not part of the output."""

    def validate(self, root: Path, requirements: Requirements | None = None) -> list[str]:
        root = Path(root)
        requirements = requirements or Requirements()
        render = root / RENDER_FILE
        if not render.is_file():
            return [f"{RENDER_FILE}: file is required"]
        problems: list[str] = []
        meta = root / META_FILE
        if meta.exists():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    problems.append(f"{META_FILE}: must be a JSON object")
                else:
                    unknown = sorted(set(cast(dict[str, Any], data)) - {"notes", "generator"})
                    problems.extend(f"{META_FILE}: unknown field {key!r}" for key in unknown)
            except (OSError, json.JSONDecodeError) as error:
                if host_failure(error):
                    raise  # this machine could not read it: not a problem of the file
                problems.append(f"{META_FILE}: cannot read JSON: {error}")
        try:
            info = probe_video(render)
        except MediaError as error:
            return [*problems, f"{RENDER_FILE}: {error}"]
        if requirements.fps is not None and abs(info.fps - requirements.fps) > 1e-3:
            problems.append(
                f"{RENDER_FILE}: {info.fps:g} fps, the task requires {requirements.fps:g}"
            )
        if requirements.num_frames is not None and info.num_frames != requirements.num_frames:
            problems.append(
                f"{RENDER_FILE}: {info.num_frames} frames, the task requires "
                f"{requirements.num_frames}"
            )
        for axis, want, got in (
            ("width", requirements.width, info.width),
            ("height", requirements.height, info.height),
        ):
            if want is not None and want != got:
                problems.append(f"{RENDER_FILE}: {axis} is {got}, the task requires {want}")
        return problems

    def documentation(self) -> str:
        return (
            "# Output format `video`\n\n"
            f"Write `{RENDER_FILE}` (H.264 MP4) into `R2S_OUTPUT_DIR`, at the frame rate, frame "
            "count and resolution the task states. An optional `meta.json` may hold "
            '`{"notes": "..."}`.\n'
        )
