"""The verifier of an exported Harbor task: freeze ``/app/solution``, re-run it, score it.

Runs inside Harbor's separate, offline verifier container, which holds the whole task,
including its hidden reference. The container is the sandbox, so twins run with the ``local``
backend here, but never as the verifier's own user: the hidden directory is made readable by
root only, and the solution runs as an unprivileged user (``nobody``, uid 65534) who owns only
its output directory, so it cannot copy the answer into its output. This needs the verifier to
run as root, as Harbor's verifier container does. Everything else is
:func:`real2simlab.train.score_solution`, the code the Lab's own evaluation uses (packaging,
program rules, run contract, the scoring protocol), so the reward equals the Lab's score of
the same program.

Writes ``reward.json`` (``reward`` and the main numbers) and ``result.json`` (the full
``r2s-task-result/1.0``). A solution that is missing, invalid, fails or writes a wrong output
scores 0. When the verifier itself cannot score (it does not run as root, a dependency is
missing, the run failed outside the solution), it writes **no** reward, explains why in
``error.txt`` and exits with status 1: Harbor then reports the trial as an error, so a trainer
never takes a broken verifier for a reward of 0.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from real2simlab.errors import SetupError
from real2simlab.sandbox.local import LocalBackend
from real2simlab.scoring import MethodInfo, TaskResult
from real2simlab.task import load_task
from real2simlab.train.reward import RewardUnavailable, reward_of, score_solution

METHOD = MethodInfo(name="harbor-agent", kind="agent")
UNPRIVILEGED = (65534, 65534)
"""The user and group solutions run as: ``nobody`` and ``nogroup`` on Debian."""


def shield_hidden(task_dir: Path) -> tuple[int, int]:
    """Make the task's hidden directory readable by root only; the user to run solutions as.

    Raises ``SetupError`` unless this process runs as root (it could not run the solution as
    another user, and the solution could then read the hidden reference)."""
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise SetupError(
            "the verifier runs the solution as an unprivileged user, so that it cannot read "
            "the task's hidden reference; that needs the verifier to run as root",
            fix="run it in the exported task's verifier container (Harbor runs it as root); "
            "--no-isolation scores a solution you trust without this protection",
        )
    hidden = Path(task_dir) / "hidden"
    if hidden.exists():
        os.chown(hidden, 0, 0)
        hidden.chmod(0o700)
    return UNPRIVILEGED


def verify(
    task_dir: Path,
    solution: Path,
    session_output: Path | None,
    work: Path,
    *,
    isolate: bool = True,
) -> TaskResult:
    """Score the solution in ``solution`` on the task at ``task_dir``, working in ``work``.

    With ``isolate`` (the default, and always in Harbor), the solution runs as an unprivileged
    user who cannot read the task's hidden directory (see :func:`shield_hidden`)."""
    task = load_task(task_dir)
    run_as = shield_hidden(task.root) if isolate else None
    if run_as is not None:
        work.chmod(0o755)  # the solution's user reads its program here, and writes nowhere else
    return score_solution(
        task,
        solution,
        backend=LocalBackend(quiet=True, run_as=run_as),
        work_dir=work,
        session_output=session_output,
        method=METHOD,
        statement_root="/app",
    ).result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--session-output", type=Path)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="run the solution as this user, who can read the hidden reference (trusted "
        "solutions only, e.g. to debug a task without root)",
    )
    args = parser.parse_args(argv)
    args.logs.mkdir(parents=True, exist_ok=True)
    for name in ("reward.json", "result.json", "error.txt"):  # nothing left from an earlier try
        (args.logs / name).unlink(missing_ok=True)
    work = Path(tempfile.mkdtemp(prefix="r2s-verify-"))
    try:
        result = verify(
            args.task, args.solution, args.session_output, work, isolate=not args.no_isolation
        )
    except Exception as error:  # the verifier failed: no reward, an error Harbor reports
        if isinstance(error, RewardUnavailable):
            (args.logs / "result.json").write_text(error.result.dumps(), encoding="utf-8")
        message = f"{type(error).__name__}: {error}\n"
        (args.logs / "error.txt").write_text(message, encoding="utf-8")
        print(f"the verifier could not score the solution: {message}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    reward = {
        "reward": reward_of(result),
        "reproduction_skill": float(result.summary.reproduction_skill or 0.0),
        "delivered": 1.0 if result.delivered else 0.0,
    }
    (args.logs / "result.json").write_text(result.dumps(), encoding="utf-8")
    (args.logs / "reward.json").write_text(json.dumps(reward, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reward))
    return 0


if __name__ == "__main__":
    sys.exit(main())
