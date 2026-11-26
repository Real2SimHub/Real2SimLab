"""Task directories: model, validation (hidden-data leaks, missing files, invalid references),
builder, task statement, scaffold."""

from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from real2simlab.errors import ProblemsError, SpecError
from real2simlab.task import (
    Code,
    TaskConfig,
    find_tasks,
    load_task,
    read_task_config,
    validate_task,
)
from real2simlab.task.instruction import TEMPLATE_VERSION, task_statement
from real2simlab.task.scaffold import new_task


def test_synthetic_tasks_validate(synthetic_tasks: Path) -> None:
    roots = find_tasks(synthetic_tasks)
    assert len(roots) == 6
    for root in roots:
        report = validate_task(root)
        assert report.ok, report.issues
        assert not report.warnings, report.warnings


def test_detects_hidden_data_leak(task_copy: Path) -> None:
    shutil.copyfile(
        task_copy / "hidden" / "reference" / "meta.json", task_copy / "visible" / "notes.json"
    )
    shutil.copytree(task_copy / "hidden" / "generator", task_copy / "visible" / "docs" / "copy")
    report = validate_task(task_copy)
    assert not report.ok
    leaks = [i.message for i in report.issues if i.code is Code.HIDDEN_LEAK]
    assert any("notes.json" in m for m in leaks)
    assert any("run.py" in m for m in leaks)


STREAMS: dict[str, dict[str, str]] = {
    "masks": {"path": "cam/masks.npz", "format": "npz", "provenance": "estimated"},
    "depth": {"path": "cam/depth.npz", "unit": "m", "kind": "metric", "provenance": "estimated"},
}


def _hidden_decoy(task: Path) -> str:
    """A label file of the test's own in ``hidden/`` (not one of the task's), as an archive."""
    marker = np.frombuffer(b"R2S-TEST-DECOY hidden labels", dtype=np.uint8)
    np.savez(task / "hidden" / "decoy_labels.npz", marker=marker)
    return "decoy_labels.npz"


def _leaks(task: Path) -> list[str]:
    return [i.message for i in validate_task(task).errors if i.code is Code.HIDDEN_LEAK]


@pytest.mark.parametrize("hidden", ["decoy", "reference/tracks.npz"])
@pytest.mark.parametrize("stream", sorted(STREAMS))
def test_hidden_data_listed_as_a_capture_stream_is_a_leak(
    task_copy: Path, stream: str, hidden: str
) -> None:
    """Listing a copy of hidden data in ``capture.json`` (as the view's masks or depth) does not
    make it part of the observation: it is a leak like any other copy."""
    if hidden == "decoy":
        hidden = _hidden_decoy(task_copy)
    entry = STREAMS[stream]
    shutil.copyfile(task_copy / "hidden" / hidden, task_copy / "visible" / entry["path"])
    capture_file = task_copy / "visible" / "capture.json"
    capture = json.loads(capture_file.read_text())
    capture["views"]["cam"][stream] = entry
    capture_file.write_text(json.dumps(capture, indent=2))
    assert f"visible/{entry['path']} has the same content as hidden/{hidden}" in _leaks(task_copy)


@pytest.mark.parametrize("hidden", ["reference/tracks.npz", "parameters.json", "decoy"])
def test_hidden_data_in_place_of_the_observed_video_is_a_leak(task_copy: Path, hidden: str) -> None:
    """Only the observed video itself may also be in ``hidden/`` (what an official scorer reads
    can include it): hidden data that is not that video, put where the capture names the video,
    is a leak."""
    if hidden == "decoy":
        hidden = _hidden_decoy(task_copy)
    shutil.copyfile(task_copy / "hidden" / hidden, task_copy / "visible" / "cam" / "rgb.mp4")
    assert f"visible/cam/rgb.mp4 has the same content as hidden/{hidden}" in _leaks(task_copy)


def test_a_hidden_copy_of_the_observed_video_is_no_leak(task_copy: Path) -> None:
    """What an official scorer reads may include the observed video itself (Video2World's sample
    video), and a reference world may render exactly it (4DCodeBench's synthetic cases): such a
    copy is the observation, which the method sees anyway, not a leak."""
    video = task_copy / "visible" / "cam" / "rgb.mp4"
    (task_copy / "hidden" / "scorer").mkdir()
    shutil.copyfile(video, task_copy / "hidden" / "scorer" / "video.mp4")
    shutil.copyfile(video, task_copy / "hidden" / "reference" / "render.mp4")
    report = validate_task(task_copy)
    assert report.ok, report.issues


@pytest.mark.parametrize("hidden", ["decoy.txt", "parameters.json"])
def test_a_statement_that_copies_hidden_data_is_a_leak(task_copy: Path, hidden: str) -> None:
    """``instruction.md`` replaces the statement every method and export is given, and nothing
    derived from ``hidden/`` may reach a statement (AGENTS.md rule 1): a copy of a hidden file
    as ``instruction.md`` is a leak like a copy in ``visible/``. A statement of its own is
    not."""
    if hidden == "decoy.txt":
        (task_copy / "hidden" / hidden).write_text("R2S-TEST-DECOY hidden statement\n")
    statement = task_copy / "instruction.md"
    shutil.copyfile(task_copy / "hidden" / hidden, statement)
    assert _leaks(task_copy) == [f"instruction.md has the same content as hidden/{hidden}"]
    statement.write_text("Build a twin of the bouncing ball in {workspace}.\n")
    assert validate_task(task_copy).ok


@pytest.mark.parametrize("name", ["._notes.txt", ".DS_Store", "cam/Thumbs.db"])
def test_a_checked_copy_holds_only_what_its_fingerprint_covers(
    task_copy: Path, tmp_path: Path, name: str
) -> None:
    """A copy is compared with the fingerprint of what was checked, which leaves out a file
    manager's litter (``.DS_Store``, ``._*``): a copy holding such a file could not be told from
    the checked version, so no copy holds one. A file added under such a name after the check
    (here a copy of a hidden file) is never copied, whatever it holds; a directory of such a
    name is copied, since the files in it are covered."""
    from real2simlab.task.io import copy_exactly
    from real2simlab.twin.fingerprint import fingerprint

    visible = task_copy / "visible"
    (visible / "._meta").mkdir()
    (visible / "._meta" / "notes.txt").write_text("a directory named like litter is content\n")
    checked = fingerprint(visible)
    decoy = task_copy / "hidden" / "decoy.txt"
    decoy.write_text("R2S-TEST-DECOY added after the check\n")
    shutil.copyfile(decoy, visible / name)
    out = tmp_path / "out"
    assert copy_exactly(visible, out, expected=checked) is None
    assert fingerprint(out) == checked
    assert not (out / name).exists()
    assert (out / "._meta" / "notes.txt").is_file()
    held = [p for p in out.rglob("*") if p.is_file() and b"R2S-TEST-DECOY" in p.read_bytes()]
    assert not held, held


COPIES = frozenset({"copytree", "copyfile", "copy", "copy2"})
"""What ``shutil`` copies a tree or a file with; each follows a symbolic link by default."""

LAB_COPIES = frozenset({"copy_tree"})
"""What ``real2simlab._fs`` copies a tree with (``shutil.copytree``, so following links)."""

TASK_DATA = re.compile(r"visible|hidden|\.root\b|reference_dir|repeat_dirs|generator_dir")
"""A source expression that names a task's ``visible/``, its ``hidden/`` (or a part of it) or
the whole task."""

BY_DESIGN = {
    "real2simlab/benchmarks/fourdcodebench/scorer.py:layout": "an official scorer's workspace",
    "real2simlab/benchmarks/video2world/scorer.py:score": "an official scorer's workspace",
    "real2simlab/benchmarks/video2world/__init__.py:restore_public_files": "the same, for "
    "video2world/scorer.py:score (its only caller); a link in visible/ is refused",
    "real2simlab/methods/baselines.py:build": "the oracle: a privileged method",
    "real2simlab/bridges/harbor.py:oracle_program": "Harbor's solution/: the agent never sees it",
    "real2simlab/report/build.py:_observed_video": "the observed video, for the report",
    "real2simlab/train/dataset.py:_copy_checked": "instruction.md: a link stays one, then checked",
}
"""The functions that copy task data otherwise, each for its reason: not for a method or an
export, or copied without following a link and checked as copied."""


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """The nodes of ``scope``, not those of the functions defined in it."""
    found: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        found.append(node)
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            stack.extend(ast.iter_child_nodes(node))
    return found


def _names(node: ast.AST) -> set[str]:
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}


def _copies_of_task_data(path: Path) -> list[tuple[str, int, str]]:
    """``(function, line, source)`` of each copy (:data:`COPIES`, however ``shutil`` or the
    function is imported) whose source names task data, directly or through a name assigned
    from it in the same function."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules, functions = {"shutil"}, set[str]()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {
                alias.asname or alias.name
                for alias in node.names
                if alias.name in ("shutil", "real2simlab._fs")
            }
        elif isinstance(node, ast.ImportFrom) and node.module == "shutil":
            functions |= {
                alias.asname or alias.name for alias in node.names if alias.name in COPIES
            }
        elif isinstance(node, ast.ImportFrom) and node.module == "real2simlab._fs":
            functions |= {
                alias.asname or alias.name for alias in node.names if alias.name in LAB_COPIES
            }
        elif isinstance(node, ast.ImportFrom) and node.module == "real2simlab":
            modules |= {alias.asname or alias.name for alias in node.names if alias.name == "_fs"}
    defined = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    found: list[tuple[str, int, str]] = []
    for scope in [tree, *defined]:
        nodes = _own_nodes(scope)
        derived: set[str] = set()

        def task_data(expression: ast.AST, derived: set[str] = derived) -> bool:
            return bool(TASK_DATA.search(ast.unparse(expression)) or _names(expression) & derived)

        grew = True
        while grew:  # names assigned from task data, or from such names
            grew = False
            for node in nodes:
                if isinstance(node, ast.Assign | ast.AnnAssign | ast.NamedExpr) and node.value:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    new = set[str]().union(*map(_names, targets)) - derived
                    if new and task_data(node.value):
                        derived |= new
                        grew = True
        for node in nodes:
            if not (isinstance(node, ast.Call) and node.args):
                continue
            func = node.func
            copying = (
                isinstance(func, ast.Attribute)
                and func.attr in COPIES | LAB_COPIES
                and isinstance(func.value, ast.Name)
                and func.value.id in modules
            ) or (isinstance(func, ast.Name) and func.id in functions)
            if copying and task_data(node.args[0]):
                name = (
                    scope.name if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef) else ""
                )
                found.append((name, node.lineno, ast.unparse(node.args[0])))
    return found


def test_the_guard_sees_the_usual_ways_of_writing_a_copy(tmp_path: Path) -> None:
    """The class guard below sees a copy with ``copytree`` or a file copy of ``shutil``, or
    with the Lab's ``copy_tree`` (``shutil.copytree`` underneath), also under another name
    (the module imported as another name, or the function itself), of task data named in the
    call or by a name assigned from it in the same function. It does not see a copy made
    otherwise (``cp`` in a subprocess, a copy in a loop over a directory)."""
    source = tmp_path / "module.py"
    source.write_text(
        "import shutil\nimport shutil as sh\nfrom shutil import copyfile as cp, copytree\n"
        "def a(task, out):\n    shutil.copytree(task.visible_dir, out)\n"
        "def b(task, out):\n    shutil.copyfile(task.visible_dir / 'x', out)\n"
        "def c(task, out):\n    sh.copy2(task.reference_dir() / 'x', out)\n"
        "def d(task, out):\n    cp(task.root / 'instruction.md', out)\n"
        "def e(task, out):\n    src = task.hidden_dir\n    path = src / 'x'\n    copytree(path, out)\n"
        "def f(work, out):\n    shutil.copytree(work, out)\n"
        "from real2simlab._fs import copy_tree as ct\nfrom real2simlab import _fs\n"
        "def g(task, out):\n    ct(task.root, out)\n"
        "def h(task, out):\n    _fs.copy_tree(task.visible_dir, out)\n"
        "def i(lab, out):\n    ct(lab, out)\n"
    )
    found = [name for name, _, _ in _copies_of_task_data(source)]
    assert found == ["a", "b", "c", "d", "e", "g", "h"]


def test_task_data_is_copied_only_by_the_checked_copy() -> None:
    """Class guard: a task's ``visible/``, ``hidden/`` (or a part of it) and a whole task are
    copied for a method or an export only by ``copy_exactly``, which never follows a symbolic
    link and checks the copy against what was validated. ``shutil``'s copies follow links by
    default, so a link in ``visible/`` would put what it points to (hidden data) into the copy;
    the few copies that are not for a method or are checked otherwise are listed with their
    reason (:data:`BY_DESIGN`)."""
    package = Path(__file__).resolve().parents[1] / "real2simlab"
    found: list[str] = []
    for path in sorted(package.rglob("*.py")):
        module = path.relative_to(package.parent).as_posix()
        for function, line, source in _copies_of_task_data(path):
            if f"{module}:{function}" not in BY_DESIGN:
                found.append(f"{module}:{line} ({function or 'module level'}): {source}")
    assert not found, "task data copied with shutil, not copy_exactly:\n" + "\n".join(found)


def test_detects_hidden_directory_names_in_visible(task_copy: Path) -> None:
    shutil.copytree(task_copy / "hidden" / "reference", task_copy / "visible" / "reference")
    (task_copy / "visible" / "reference" / "extra.txt").write_text("not hidden content")
    leaks = [i.message for i in validate_task(task_copy).issues if i.code is Code.HIDDEN_LEAK]
    assert any("reference/ belongs in hidden/" in m for m in leaks)


def test_detects_missing_files(task_copy: Path) -> None:
    (task_copy / "visible" / "cam" / "rgb.mp4").unlink()
    shutil.rmtree(task_copy / "hidden" / "reference")
    report = validate_task(task_copy)
    missing = [i.message for i in report.issues if i.code is Code.MISSING_FILE]
    assert any("rgb.mp4" in m for m in missing)
    assert any("hidden/reference/" in m for m in missing)


def test_detects_invalid_references_and_symlinks(task_copy: Path) -> None:
    meta = task_copy / "hidden" / "reference" / "meta.json"
    meta.write_text("{}")
    (task_copy / "visible" / "link").symlink_to(task_copy / "hidden")
    report = validate_task(task_copy)
    codes = report.codes()
    assert Code.REFERENCE_INVALID in codes and Code.NOT_SELF_CONTAINED in codes
    with pytest.raises(ProblemsError):
        report.raise_for_errors()
    assert report.to_json()["ok"] is False


def test_detects_invalid_repeats(task_copy: Path) -> None:
    repeat = task_copy / "hidden" / "repeats" / "001"
    shutil.copytree(task_copy / "hidden" / "reference", repeat)
    assert validate_task(task_copy).ok
    (repeat / "meta.json").write_text("{}")
    messages = [i.message for i in validate_task(task_copy).errors]
    assert any("repeats/001" in m for m in messages)


def test_spec_errors(tmp_path: Path, bouncing_task: Path) -> None:
    assert validate_task(tmp_path).codes() == {Code.SPEC}
    config = read_task_config(bouncing_task / "task.toml")
    data = config.to_json()
    with pytest.raises(ValidationError, match="primary"):
        TaskConfig.model_validate(
            {
                **data,
                "scoring": {"protocol": "reproduction", "metrics": ["traj3d_mean"], "primary": "x"},
            }
        )
    with pytest.raises(ValidationError, match="task id"):
        TaskConfig.model_validate({**data, "id": "Bad Id"})
    with pytest.raises(ValidationError, match="native"):
        TaskConfig.model_validate({**data, "scoring": {"protocol": "native"}})
    with pytest.raises(ValidationError, match="conditions"):
        TaskConfig.model_validate({**data, "conditions": {}})
    (tmp_path / "task.toml").write_text("not = [toml")
    with pytest.raises(SpecError):
        load_task(tmp_path)


def test_statement(bouncing_task: Path) -> None:
    task = load_task(bouncing_task)
    statement = task_statement(task, "/ws")
    assert "/ws/solution/run.py" in statement and "R2S_OUTPUT_DIR" in statement
    assert "selftest" in statement and "60 frames" in statement
    assert "hidden" not in statement and "hidden/" not in statement
    assert TEMPLATE_VERSION.startswith("r2s-instruction/")
    describe = task.describe()
    assert describe["protocol"] == "reproduction" and describe["redistributable"] is True
    assert describe["has_reference"] is True


def test_instruction_override(task_copy: Path) -> None:
    (task_copy / "instruction.md").write_text("Do it in {workspace}.")
    assert task_statement(load_task(task_copy), "/w") == "Do it in /w."


def test_scaffold(tmp_path: Path) -> None:
    text = new_task(tmp_path / "t", task_id="mine/drop_01", title="Drop")
    assert "Next:" in text and "hidden/reference/" in text
    report = validate_task(tmp_path / "t")
    assert not report.ok  # a skeleton needs a capture and a reference
    config = read_task_config(tmp_path / "t" / "task.toml")
    assert config.id == "mine/drop_01" and config.scoring.protocol.value == "reproduction"
    with pytest.raises(FileExistsError):
        new_task(tmp_path / "t", task_id="mine/x", title="x")
