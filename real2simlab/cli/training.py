"""``r2s train``: training tasks, datasets and rewards."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

from real2simlab.cli.common import (
    TASKS_INTERRUPTED,
    add_backend,
    add_parser,
    backend_from,
    emit,
    finite_float,
    note_other_tasks,
    positive_int,
    seed_int,
)


def register_train(subparsers: Any) -> None:
    parser = add_parser(
        subparsers,
        "train",
        "training tasks with exact truth, datasets, and the evaluation's scorer as a reward",
        "Training with the Lab: training tasks (generated, or your own with split = train),\n"
        "exported with the runs on them as a dataset that holds no evaluation task's recording,\n"
        "a method trained on it, and any solution scored exactly as an evaluation would.\n\n"
        "Example:\n"
        "  r2s train tasks --count 200 --out train/tasks\n"
        "  r2s bench build synthetic --out eval/tasks\n"
        "  r2s train export --tasks train/tasks --against eval/tasks --out train/data\n"
        "  r2s train fit nearest_twin --data train/data --out train/model\n"
        "  r2s eval --tasks eval/tasks --method nearest_twin:checkpoint=train/model \\\n"
        "      --out eval/run\n"
        "  r2s train reward --task train/tasks/synthetic-train/pendulum/calibrated/s0-00003 \\\n"
        "      --solution my_program/",
    )
    sub = parser.add_subparsers(dest="train_command", metavar="COMMAND", required=True)

    tasks = add_parser(
        sub,
        "tasks",
        "generate training tasks: the synthetic phenomena with sampled parameters",
        "Write training tasks (split = train): each synthetic phenomenon with its hidden\n"
        "parameters and initial state drawn around the evaluation task's, rendered with the\n"
        "same camera and scored the same way, with the exact truth as labels. The same seed\n"
        "always gives the same tasks. A near copy of an evaluation task is never drawn, and a\n"
        "sample whose video the overlap check cannot tell from the evaluation task's is\n"
        "skipped (its index is then missing).",
    )
    tasks.add_argument("--out", type=Path, required=True, help="directory to write tasks into")
    tasks.add_argument("--count", type=positive_int, default=20, help="samples per phenomenon (20)")
    tasks.add_argument("--seed", type=seed_int, default=0, help="which samples (0)")
    tasks.add_argument("--select", action="append", help="only these phenomena (repeatable)")
    tasks.add_argument(
        "--tier", action="append", dest="tiers", help="visibility tiers (default: both)"
    )
    tasks.add_argument("--overwrite", action="store_true", help="replace existing tasks")
    tasks.set_defaults(
        handler=_tasks,
        on_interrupt=TASKS_INTERRUPTED,
    )

    export = add_parser(
        sub,
        "export",
        "export tasks and the results of runs on them as a training dataset",
        "Write a dataset: tasks.jsonl (inputs and labels of each task) and runs.jsonl (each\n"
        "delivered program of the runs, with its reward, the content hash of the task it was\n"
        "scored on and the agent's redacted trajectory). A result is exported only with the\n"
        "task version it was scored on (found where the run found it, or among --tasks).\n"
        "Only training tasks (split = train) are exported unless --allow-eval-tasks; tasks of\n"
        "a benchmark whose terms forbid training on its data never are. With --against, a\n"
        "task that holds the same file or the same recording as one of those evaluation tasks\n"
        "(a copy, a re-encoding, a resized copy or a clip cut from it) is left out, and\n"
        "export.json says why.",
    )
    export.add_argument("--out", type=Path, required=True, help="dataset directory to create")
    export.add_argument("--tasks", type=Path, action="append", default=[], help="task directories")
    export.add_argument(
        "--against",
        type=Path,
        action="append",
        default=[],
        help="evaluation task directories the dataset must not overlap (repeatable)",
    )
    export.add_argument(
        "--runs", type=Path, action="append", default=[], help="run directories (r2s eval/run)"
    )
    export.add_argument(
        "--min-reward", type=finite_float, help="keep only programs with at least this reward"
    )
    export.add_argument(
        "--include-undelivered",
        action="store_true",
        help="also export results where nothing was delivered (no program, reward 0)",
    )
    export.add_argument(
        "--skip-unmatched",
        action="store_true",
        help="leave out results whose task or program is missing or has changed since the run, "
        "instead of failing",
    )
    export.add_argument(
        "--allow-eval-tasks",
        action="store_true",
        help="also export evaluation tasks (their results then no longer count as held out)",
    )
    export.add_argument("--overwrite", action="store_true", help="replace an existing dataset")
    export.set_defaults(
        handler=_export, on_interrupt="--out was left as it was; run the same command again"
    )

    fit = add_parser(
        sub,
        "fit",
        "train a method on an exported dataset: a checkpoint to evaluate it with",
        "Give a method that learns (e.g. nearest_twin) an exported dataset; it writes what it\n"
        "learned into the checkpoint directory, with the Lab's record (training.json) and the\n"
        "dataset's overlap prints. Evaluate it with checkpoint=DIR among its options: it is not\n"
        "run on tasks that overlap its training tasks.\n\n"
        "Example:\n"
        "  r2s train fit nearest_twin --data train/data --out train/model\n"
        "  r2s eval --tasks eval/tasks --method nearest_twin:checkpoint=train/model --out eval/run",
    )
    fit.add_argument("method", help="a method that learns, NAME[:key=value,...]")
    fit.add_argument("--data", type=Path, required=True, help="an exported dataset")
    fit.add_argument("--out", type=Path, required=True, help="checkpoint directory to create")
    fit.add_argument("--seed", type=seed_int, default=0)
    fit.add_argument("--overwrite", action="store_true", help="replace an existing checkpoint")
    fit.set_defaults(
        handler=_fit, on_interrupt="--out was left as it was; run the same command again"
    )

    reward = add_parser(
        sub,
        "reward",
        "score one solution on one task as an evaluation would: its reward",
        "Package a solution directory (run.py and its files) as a twin, run it once, frozen and\n"
        "offline, and score it with the task's protocol. Prints the reward (0 to 1) and, with\n"
        "--json, the full task result. A run that fails outside the solution (the sandbox did\n"
        "not start) gives no reward: the command fails with status 1. Use docker or bwrap for\n"
        "programs you did not write.",
    )
    reward.add_argument("--task", type=Path, required=True, help="a task directory")
    reward.add_argument(
        "--solution",
        required=True,  # the text as given: an empty one is refused, not taken for "."
        help="a solution directory (run.py and its files), or a twin package (twin.json, program/)",
    )
    reward.add_argument("--seed", type=seed_int, default=0)
    reward.add_argument(
        "--work", type=Path, help="keep the run's files here (cleared when used again)"
    )
    reward.add_argument("--json", action="store_true", help="print the full result as JSON")
    add_backend(reward)
    reward.set_defaults(handler=_reward, on_interrupt="run the same command again")


def _tasks(args: argparse.Namespace) -> int:
    from real2simlab.benchmarks.synthetic import TIERS, SyntheticBenchmark

    adapter = SyntheticBenchmark()
    paths = adapter.build(
        args.out,
        select=args.select,
        tiers=args.tiers or TIERS,
        overwrite=args.overwrite,
        split="train",
        count=args.count,
        seed=args.seed,
    )
    print(f"wrote {len(paths)} training task(s) in {args.out}")
    for name, skipped in sorted(adapter.look_alikes.items()):
        print(f"{name}: skipped {skipped} sample(s) whose video looks like the evaluation task's")
    note_other_tasks(args.out, paths)
    return 0


def _export(args: argparse.Namespace) -> int:
    from real2simlab.train import export_dataset

    if not args.against:
        print(
            "note: not compared with evaluation tasks; pass --against EVAL_TASKS to leave out "
            "tasks that hold their recordings",
            file=sys.stderr,
        )
    summary = export_dataset(
        args.out,
        tasks=args.tasks,
        runs=args.runs,
        against=args.against,
        allow_eval_tasks=args.allow_eval_tasks,
        min_reward=args.min_reward,
        include_undelivered=args.include_undelivered,
        skip_unmatched=args.skip_unmatched,
        overwrite=args.overwrite,
        command=" ".join(["r2s", *map(shlex.quote, sys.argv[1:])]),
    )
    for line in summary.skipped:
        print(f"skipped: {line}", file=sys.stderr)
    print(
        f"exported {summary.tasks} task(s) and {summary.runs} result(s) of runs "
        f"({summary.programs} with a program) to {summary.out}"
    )
    if summary.skipped:
        print(f"left out {len(summary.skipped)} (reasons above and in export.json)")
    if summary.checked_against is not None:
        print(
            f"compared with {summary.checked_against} evaluation task(s): "
            f"{len(summary.overlapping)} overlapping task(s) left out"
        )
    return 0


def _fit(args: argparse.Namespace) -> int:
    from real2simlab.cli.common import parse_method
    from real2simlab.methods.base import get_method
    from real2simlab.train import fit_method

    entry = parse_method(args.method, given="r2s train fit")
    name = entry.pop("name")
    entry.pop("label", None)
    method = get_method(name, **entry)
    record = fit_method(
        method, args.data, args.out, seed=args.seed, overwrite=args.overwrite, log=print
    )
    print(
        f"trained {record.method} in {record.duration_s:.1f} s on {args.data} ({record.tasks} "
        f"task(s), {record.runs} result(s) of runs): checkpoint {args.out}"
    )
    return 0


def _reward(args: argparse.Namespace) -> int:
    from real2simlab.errors import SetupError
    from real2simlab.train import score_solution

    # a solution directory that is there but empty or wrong scores 0; a path that is not there,
    # or none at all (an unset variable: "" would be the current directory), is a mistake in
    # the command, not a program that failed
    fix = "give the directory that holds the solution's program (with its entry point, e.g. run.py)"
    if not args.solution:
        raise SetupError("--solution is empty (it would be the current directory)", fix=fix)
    given = Path(args.solution)
    if not given.is_dir():
        what = "is not a directory" if given.exists() else "does not exist"
        raise SetupError(f"--solution {args.solution} {what}", fix=fix)
    # The directory the user named, through the links in its path (score_solution follows none
    # in place of the solution: there, it is what an agent left).
    solution = given.resolve()
    scored = score_solution(  # RewardUnavailable (status 1) when the sandbox failed
        args.task, solution, backend=backend_from(args), work_dir=args.work, seed=args.seed
    )
    result = scored.result
    text = f"reward {scored.reward:.4f}" + (
        "" if result.delivered else f" (not delivered: {result.delivery_failure})"
    )
    if result.delivered and result.reproduction.run_status not in (None, "succeeded"):
        text += f" (run {result.reproduction.run_status})"
    emit({"reward": scored.reward, "result": json.loads(result.dumps())}, args.json, text)
    return 0
