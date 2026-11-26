"""JSON Schemas of every specification object, generated from the pydantic models.

``r2s spec schema --out DIR`` writes ``<object>.schema.json`` files; the documentation ships
the same files in ``docs/spec/schemas/``, and a test checks they are up to date.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from real2simlab.errors import UnknownNameError


def _models() -> dict[str, type[BaseModel]]:
    from real2simlab.agents.record import AgentRecord, BuildRecord
    from real2simlab.capture.model import Capture
    from real2simlab.methods.training import TrainingRecord
    from real2simlab.orchestrate.config import ExperimentConfig
    from real2simlab.runner.record import RunRecord
    from real2simlab.scoring.model import Results, TaskResult
    from real2simlab.task.model import TaskConfig
    from real2simlab.twin.model import TwinManifest

    return {
        "r2s-task": TaskConfig,
        "r2s-capture": Capture,
        "r2s-twin": TwinManifest,
        "r2s-run": RunRecord,
        "r2s-build": BuildRecord,
        "r2s-agent-run": AgentRecord,
        "r2s-task-result": TaskResult,
        "r2s-results": Results,
        "r2s-experiment": ExperimentConfig,
        "r2s-training": TrainingRecord,
    }


def _tracks_meta() -> dict[str, Any]:
    return {
        "title": "r2s-tracks meta.json",
        "type": "object",
        "additionalProperties": False,
        "required": ["spec", "fps", "num_frames", "objects"],
        "properties": {
            "spec": {"const": "r2s-tracks/1.0"},
            "fps": {"type": "number", "exclusiveMinimum": 0},
            "num_frames": {"type": "integer", "minimum": 1},
            "objects": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_-]*$"},
            },
            "frame": {
                "type": "object",
                "properties": {"length_unit": {"const": "m"}, "up_axis": {"const": "+z"}},
                "additionalProperties": False,
            },
            "notes": {"type": "string"},
            "generator": {"type": "string"},
        },
    }


def _tracks_camera() -> dict[str, Any]:
    matrix4 = {
        "type": "array",
        "minItems": 4,
        "maxItems": 4,
        "items": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}},
    }
    return {
        "title": "r2s-tracks camera.json",
        "type": "object",
        "additionalProperties": False,
        "required": ["intrinsics", "width", "height", "extrinsics"],
        "properties": {
            "intrinsics": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "number"},
                },
            },
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "extrinsics": {"oneOf": [matrix4, {"type": "array", "minItems": 1, "items": matrix4}]},
        },
    }


EXTRA: dict[str, Callable[[], dict[str, Any]]] = {
    "r2s-tracks-meta": _tracks_meta,
    "r2s-tracks-camera": _tracks_camera,
}


def schema_names() -> list[str]:
    return sorted([*_models(), *EXTRA])


def schema_for(name: str) -> dict[str, Any]:
    """The JSON Schema of the specification object ``name`` (e.g. ``r2s-task``)."""
    models = _models()
    if name in models:
        schema = models[name].model_json_schema(mode="validation")
    elif name in EXTRA:
        schema = EXTRA[name]()
    else:
        raise UnknownNameError("specification object", name, [*models, *EXTRA])
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{name}.schema.json",
        **schema,
    }


def export_schemas(out_dir: Path, names: Sequence[str] | None = None) -> list[Path]:
    """Write the schemas of ``names`` (default: every object) to
    ``<out_dir>/<name>.schema.json``; return the paths. An unknown name raises
    ``UnknownNameError`` before anything is written."""
    out_dir = Path(out_dir)
    schemas = {name: schema_for(name) for name in (schema_names() if names is None else names)}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, schema in schemas.items():
        path = out_dir / f"{name}.schema.json"
        path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(path)
    return paths
