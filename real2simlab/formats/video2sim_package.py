"""The ``video2sim_package`` format: a video2sim protocol package (Video2World's deliverable).

A package is data, not a program's output: ``protocol.json`` (the scene, the camera and how the
actions are to be read), an action stream (``actions.npy``, or ``actions.npz`` for the cloth
profile), meshes under ``assets/`` and, for most profiles, ``expected/obj_poses.npy``. This
format checks only that the package can be read; the profile's full contract is checked by
the benchmark's own validator and evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, cast

from real2simlab.errors import host_failure
from real2simlab.formats.base import OutputFormat, Requirements

PROTOCOL_FILE = "protocol.json"
ACTION_FILES = ("actions.npy", "actions.npz")


class Video2SimPackageFormat(OutputFormat):
    """``protocol.json`` and an action stream."""

    name: ClassVar[str] = "video2sim_package"
    summary: ClassVar[str] = "A video2sim protocol package (scene, camera, actions): Video2World."

    def validate(self, root: Path, requirements: Requirements | None = None) -> list[str]:
        root = Path(root)
        protocol = root / PROTOCOL_FILE
        if not protocol.is_file():
            return [f"{PROTOCOL_FILE}: file is required"]
        try:
            data = json.loads(protocol.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            if host_failure(error):
                raise  # this machine could not read it: not a problem of the file
            return [f"{PROTOCOL_FILE}: cannot read JSON: {error}"]
        if not isinstance(data, dict):
            return [f"{PROTOCOL_FILE}: must be a JSON object"]
        manifest = cast(dict[str, Any], data)
        problems: list[str] = []
        actions = manifest.get("actions")
        named = cast(dict[str, Any], actions).get("path") if isinstance(actions, dict) else None
        candidates = [str(named)] if isinstance(named, str) else list(ACTION_FILES)
        if not any((root / name).is_file() for name in candidates):
            problems.append(f"no action stream: expected one of {', '.join(candidates)}")
        return problems

    def documentation(self) -> str:
        return (
            "# Output format `video2sim_package`\n\n"
            f"A video2sim protocol package: `{PROTOCOL_FILE}`, the action stream it names "
            "(`actions.npy` for most profiles), meshes under `assets/` and, where the profile "
            "asks for it, `expected/obj_poses.npy`. The task statement gives the profile's full "
            "contract; the benchmark's own validator checks it.\n"
        )
