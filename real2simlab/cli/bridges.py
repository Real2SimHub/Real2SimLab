"""``r2s export harbor``: write a task as a Harbor task."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from real2simlab.cli.common import add_parser, positive_float


def register_export(subparsers: Any) -> None:
    parser = add_parser(subparsers, "export", "export tasks to other frameworks")
    sub = parser.add_subparsers(dest="export_command", metavar="TARGET", required=True)
    harbor = add_parser(
        sub,
        "harbor",
        "write a task as a Harbor task (agent environment, offline verifier, optional oracle)",
        "Write a task as a Harbor task directory. Harbor then runs any of its agents on it; the\n"
        "separate, offline verifier re-runs the delivered twin and scores it with this Lab.\n\n"
        "Example:\n"
        "  r2s export harbor runs/tasks/bouncing_ball/calibrated --out harbor/bouncing_ball \\\n"
        "      --solution oracle\n"
        "  r2s env build r2s-verifier\n"
        "  harbor run -p harbor/bouncing_ball -a oracle",
    )
    harbor.add_argument("task", type=Path, help="a task directory")
    harbor.add_argument("--out", type=Path, required=True, help="Harbor task directory to create")
    harbor.add_argument(
        "--solution",
        help="'oracle' (the task's truth generator) or a program directory with run.py",
    )
    harbor.add_argument(
        "--self-contained",
        action="store_true",
        help="Dockerfiles that install everything from python:3.11-slim (no r2s images needed)",
    )
    harbor.add_argument("--timeout", type=positive_float, help="agent timeout in seconds")
    harbor.set_defaults(handler=_harbor)


def _harbor(args: argparse.Namespace) -> int:
    from real2simlab.bridges.harbor import export_task
    from real2simlab.errors import SetupError

    if args.solution == "":  # an unset variable: not the current directory's program
        raise SetupError(
            "--solution is empty (it would be the current directory)",
            fix="give 'oracle' (the task's truth generator) or a program directory with run.py, "
            "or leave --solution out",
        )
    solution = args.solution if args.solution in (None, "oracle") else Path(args.solution)
    print(
        export_task(
            args.task,
            args.out,
            solution=solution,
            self_contained=args.self_contained,
            timeout_s=args.timeout,
        )
    )
    return 0
