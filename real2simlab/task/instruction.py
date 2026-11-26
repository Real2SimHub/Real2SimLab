"""The task statement given to agents.

The statement is generated from a versioned template, recorded with every agent run; a task's
``instruction.md`` replaces the generated statement.
"""

from __future__ import annotations

import re
from pathlib import Path

from real2simlab.capture.io import CAPTURE_FILENAME, load_capture
from real2simlab.errors import UsageError
from real2simlab.formats import get_format
from real2simlab.task.io import Task
from real2simlab.task.model import DeliveryConvention, TaskConfig

TEMPLATE_VERSION = "r2s-instruction/1.0"
"""Version of the generated task statement; bumped whenever the wording changes."""


def task_statement(task: Task, workspace: str = "/workspace") -> str:
    """The full task statement for an agent whose workspace is mounted at ``workspace``."""
    return statement_from(task.config, task.instruction_override(), task.visible_dir, workspace)


def statement_from(
    config: TaskConfig, instruction: str | None, visible_dir: Path, workspace: str = "/workspace"
) -> str:
    """The statement of a task with ``config`` whose ``instruction.md`` holds ``instruction``
    (``None``: it has none) and whose ``visible/`` is ``visible_dir``: ``instruction`` itself,
    which replaces the generated statement, or the statement generated from ``config`` and the
    capture in ``visible_dir``. Given the ``instruction.md`` and the copy of ``visible/`` that
    were checked, it is the statement of exactly what was checked, whatever the task directory
    holds by then."""
    if instruction is not None:
        return instruction.replace("{workspace}", workspace)
    if config.delivery.convention is not DeliveryConvention.R2S:
        raise UsageError(
            f"task {config.id!r} uses the {config.delivery.convention.value!r} convention and "
            "must provide instruction.md"
        )
    delivery = config.delivery
    output = delivery.output
    fmt = get_format(delivery.output_format)
    needs: list[str] = []
    if output.fps is not None:
        needs.append(f"`fps` = {output.fps:g}")
    if output.num_frames is not None:
        needs.append(f"{output.num_frames} frames")
    if output.objects:
        needs.append("objects named " + ", ".join(f"`{name}`" for name in output.objects))
    if output.width is not None and output.height is not None:
        needs.append(f"resolution {output.width}x{output.height}")
    requirement = ("Your output must have " + "; ".join(needs) + ".") if needs else ""
    w = workspace
    text = _capture_section(visible_dir)
    gpu = "a GPU is available" if delivery.resources.gpu else "there is no GPU"
    about = f"\n{config.description.strip()}\n" if config.description else ""
    return f"""\
# Task: {config.title}

Build a **twin** of the physical event recorded in `{w}/visible/`: a program that simulates
the event and reproduces what was recorded.
{about}{text}
## Your workspace (`{w}`)

- `{w}/visible/`: everything you may use about the event: `capture.json` (cameras, scene,
  text; see `docs/capture.md`) and the media it references; maybe `docs/` and `tools/`.
- `{w}/docs/`: reference documents: the output format, the run contract, the SDK.
- `{w}/tools/r2s_twin.py`: the twin SDK (NumPy only). It reads the seed and writes the
  output format for you. It is also importable as `r2s_twin` when your program is re-run; you
  may copy it into `solution/`.
- `{w}/solution/`: empty. Your deliverable goes here.

## What to deliver

1. `{w}/solution/run.py`, run as `python run.py` from inside `solution/`.
2. Output in the `{fmt.name}` format (`docs/output-format.md`). {requirement}
3. When you are done, run your program once with `R2S_OUTPUT_DIR={w}/output`, so the Lab can
   check that the frozen re-run produces the same output.

## How your program is run

- Working directory: its own copy of `solution/`, **read-only**. It cannot see `visible/` or
  anything else of the workspace: put every file it needs inside `solution/`.
- `R2S_OUTPUT_DIR`: an empty directory; write the complete output there.
- `R2S_SEED`: an integer; derive every random number from it. The same seed must give the
  same output.
- No network. Write only to `R2S_OUTPUT_DIR` and `/tmp`. Time limit {delivery.run_timeout_s} s;
  {gpu}.
- Environment `{delivery.environment}`: Python 3.11 with NumPy, SciPy and ffmpeg.

## Rules

- Do not put video files in `solution/` (copying the recording is not a simulation), and keep
  it under {delivery.program.max_size_mb:g} MB. Symbolic links are not allowed.
- You have about {delivery.agent_time_budget_s // 60} minutes.

## Check your work

    cd {w}/solution && python {w}/tools/r2s_twin.py selftest .

This runs your program the way it will be re-run, twice: the outputs must be valid and
identical.
"""


def with_time_budget(statement: str, task_budget_s: float, budget_s: float) -> str:
    """``statement`` with the agent's time stated as ``budget_s`` instead of the task's own
    budget, for a method option that overrides it (``time_budget_s``).

    It rewrites the sentence that states the task's budget in minutes ("You have about N
    minutes" in the Lab's statement, "you have N minutes" in Video2World's); a statement that
    states no time, as most benchmarks' own statements, is returned unchanged."""
    old, new = int(task_budget_s) // 60, int(budget_s) // 60
    if old == new:
        return statement
    pattern = re.compile(rf"\b([Yy]ou have (?:about )?){old}( minutes?\b)")
    return pattern.sub(lambda m: f"{m[1]}{new}{m[2]}", statement)


def capture_text(task: Task) -> str:
    """The capture's text fields, if any, as a short section."""
    return _capture_section(task.visible_dir)


def _capture_section(visible_dir: Path) -> str:
    """The text fields of the capture in ``visible_dir``, if any, as a short section."""
    if not (visible_dir / CAPTURE_FILENAME).is_file():
        return ""
    capture = load_capture(visible_dir)
    if capture.text is None:
        return ""
    lines = [
        f"- {label}: {value}"
        for label, value in (
            ("Scene", capture.text.scene_description),
            ("Goal", capture.text.task),
        )
        if value
    ]
    return ("\n## The event\n\n" + "\n".join(lines) + "\n") if lines else ""
