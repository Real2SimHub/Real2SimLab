from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from real2simlab import plugins
from real2simlab.errors import UnknownNameError
from real2simlab.spec import SpecId, check_spec
from real2simlab.spec.schemas import export_schemas, schema_for, schema_names

ROOT = Path(__file__).resolve().parents[1]


def test_check_spec_accepts_same_and_older_minor() -> None:
    assert check_spec("r2s-task/1.0", SpecId.TASK) == "r2s-task/1.0"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("r2s-twin/1.0", "expected spec"),
        ("r2s-task/2.0", "major version"),
        ("r2s-task/1.7", "newer"),
        ("task", "expected spec"),
    ],
)
def test_check_spec_rejects(value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_spec(value, SpecId.TASK)


def test_every_schema_exports(tmp_path: Path) -> None:
    paths = export_schemas(tmp_path)
    assert len(paths) == len(schema_names()) >= 10
    for path in paths:
        schema = json.loads(path.read_text())
        assert schema["$schema"].startswith("https://json-schema.org/")
    with pytest.raises(UnknownNameError):
        schema_for("r2s-nothing")


def _floats(
    node: Any,
    path: str,
    config: Mapping[str, Any] | None = None,
    definitions: Mapping[str, Any] | None = None,
    active: frozenset[str] = frozenset(),
) -> Iterator[tuple[str, bool]]:
    """Every number with a fraction in a pydantic core schema: its place (a shared definition
    is followed where it is used), and whether it takes ``inf`` and ``nan``: its own setting,
    else its model's (each model validates its fields with its own configuration)."""
    config, definitions = config or {}, definitions or {}
    if isinstance(node, list | tuple):
        for item in node:
            yield from _floats(item, path, config, definitions, active)
        return
    if not isinstance(node, dict):
        return
    kind = node.get("type")
    if kind == "definitions":
        definitions = {**definitions, **{item["ref"]: item for item in node["definitions"]}}
        yield from _floats(node["schema"], path, config, definitions, active)
        return
    if kind == "definition-ref":
        ref = node["schema_ref"]
        if ref not in active:  # a recursive definition (JsonValue) is walked once
            yield from _floats(definitions[ref], path, config, definitions, active | {ref})
        return
    if kind == "model":
        config = node.get("config") or {}
    if kind == "float":
        yield path, bool(node.get("allow_inf_nan", config.get("allow_inf_nan", True)))
    for key, value in node.items():
        if key in ("cls", "config", "metadata", "serialization"):
            continue
        if key == "fields" and kind == "model-fields":
            for name, field in value.items():
                yield from _floats(field, f"{path}.{name}", config, definitions, active)
        else:
            yield from _floats(value, path, config, definitions, active)


def _inputs() -> dict[str, Any]:
    from real2simlab.capture.model import Capture
    from real2simlab.orchestrate.config import ExperimentConfig
    from real2simlab.task.model import TaskConfig
    from real2simlab.twin.model import TwinManifest

    models: dict[str, Any] = {
        "task.toml": TaskConfig,
        "capture.json": Capture,
        "twin.json": TwinManifest,
        "experiment": ExperimentConfig,
    }
    for name, method in plugins.load_all("method").items():
        models[f"method {name}"] = method.Options
    return models


PLUGIN_OPTIONS = (
    "task.toml.scoring.native.options",
    "task.toml.scoring.metric_options",
    "task.toml.metadata",
)
"""What a task passes to its scorer, its metrics or its benchmark adapter: JSON values that
the plugin reads and checks (``metrics.base.positive_number`` refuses a number that is not
finite)."""


def test_no_input_takes_a_number_that_is_not_finite() -> None:
    """``inf`` passes ``PositiveFloat`` (it is above 0): ``cpus = inf`` in a task, or
    ``cpus=1e309`` for a method, passed every check and every run then ended in a traceback.
    Every number with a fraction in a task, a capture, a twin, an experiment and every
    method's options must refuse ``inf`` and ``nan``."""
    found = {
        place: takes
        for label, model in _inputs().items()
        for place, takes in _floats(model.__pydantic_core_schema__, label)
    }
    assert "task.toml.delivery.resources.cpus" in found  # the walk reaches what it guards
    assert "method fake_agent.cpus" in found
    assert any(place.startswith(PLUGIN_OPTIONS) for place in found)
    taking = sorted(p for p, takes in found.items() if takes and not p.startswith(PLUGIN_OPTIONS))
    assert not taking, f"these take inf and nan (give their model FINITE_CONFIG): {taking}"


def test_spec_schema_with_a_name_writes_only_that_object(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``r2s spec schema r2s-task --out DIR`` wrote every schema (over the files of those
    names already in DIR), and an unknown name with ``--out`` wrote them all, with status 0;
    without ``--out`` the name was honoured."""
    from real2simlab.cli.main import main

    out = tmp_path / "one"
    out.mkdir()
    (out / "r2s-experiment.schema.json").write_text('{"mine": true}\n')
    assert main(["spec", "schema", "r2s-task", "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "r2s-experiment.schema.json",
        "r2s-task.schema.json",
    ]
    assert json.loads((out / "r2s-experiment.schema.json").read_text()) == {"mine": True}
    written = json.loads((out / "r2s-task.schema.json").read_text())
    assert written == json.loads(json.dumps(schema_for("r2s-task")))
    capsys.readouterr()
    unknown = tmp_path / "unknown"
    assert main(["spec", "schema", "no-such-object", "--out", str(unknown)]) == 1
    err = capsys.readouterr().err
    assert "unknown specification object 'no-such-object'" in err, err
    assert not unknown.exists()


def test_published_schemas_are_up_to_date(tmp_path: Path) -> None:
    published = ROOT / "docs" / "spec" / "schemas"
    if not published.is_dir():
        pytest.skip("docs/spec/schemas not generated yet")
    export_schemas(tmp_path)
    for path in sorted(tmp_path.glob("*.json")):
        current = published / path.name
        assert current.is_file(), (
            f"run: r2s spec schema --out docs/spec/schemas ({path.name} missing)"
        )
        assert json.loads(current.read_text()) == json.loads(path.read_text()), (
            f"{path.name} is stale; run: r2s spec schema --out docs/spec/schemas"
        )


def test_cli_reference_is_up_to_date() -> None:
    from real2simlab.spec.docgen import cli_reference

    path = ROOT / "docs" / "reference" / "cli.md"
    assert path.read_text() == cli_reference(), (
        "docs/reference/cli.md is stale; run: python -c 'from pathlib import Path; "
        "from real2simlab.spec.docgen import write_cli_reference; "
        'write_cli_reference(Path("docs/reference/cli.md"))\''
    )
