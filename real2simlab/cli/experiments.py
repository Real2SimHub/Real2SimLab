"""``r2s run``, ``r2s eval``, ``r2s report``: running experiments and reporting them."""

from __future__ import annotations

import argparse
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any

from real2simlab.cli.common import (
    UNSCORED_EXIT,
    add_backend,
    add_parser,
    non_negative_int,
    parse_method,
    positive_int,
    seed_int,
)
from real2simlab.errors import UsageError


def register_run(subparsers: Any) -> None:
    parser = add_parser(
        subparsers,
        "run",
        "run an experiment from a configuration file",
        "Run an experiment described by a TOML file (r2s-experiment/1.0): tasks, methods,\n"
        "seeds, trials, backend, concurrency and budgets. Interrupted runs resume where they\n"
        "stopped when the same command is run again.\n\n" + UNSCORED_EXIT,
    )
    parser.add_argument("config", type=Path, help="experiment TOML file")
    parser.add_argument("--out", type=Path, help="run directory (default: runs/<name>)")
    parser.add_argument("--backend", help="override the configured backend")
    parser.add_argument("--no-resume", action="store_true", help="redo every step")
    parser.add_argument("--no-report", action="store_true", help="do not write the HTML report")
    parser.set_defaults(
        handler=_run, on_interrupt="run the same command again to resume: finished steps are reused"
    )


def _print_summary(results: Any, out: Path) -> int:
    """Print the summary table; returns the exit status (``report_unscored``)."""
    from real2simlab.cli.common import report_unscored, table
    from real2simlab.scoring.model import method_label

    rows = []
    for method in results.methods:
        name = method_label(method.method, method.privileged, method.network)
        for entry in method.benchmarks:
            unit = f" {entry.unit}" if entry.unit else ""
            rows.append(
                [
                    name,
                    entry.benchmark,
                    f"{entry.metric}{unit}",
                    _num(entry.value.mean),
                    f"{entry.scored}/{entry.results}",
                    str(entry.infra_errors) if entry.infra_errors else "",
                    _delivered(entry),
                ]
            )
        if not method.benchmarks:
            pending = str(method.infra_errors) if method.infra_errors else ""
            scorable = method.results - method.infra_errors
            rate = f"{method.delivered_rate:.0%}" if scorable > 0 else "—"
            rows.append([name, "—", "—", "—", "—", pending, rate])
    header = ["method", "benchmark", "headline", "value", "scored", "not scored", "delivered"]
    print(table(rows, header))
    print(f"\nresults: {out / 'results.json'}\nreport:  {out / 'report' / 'index.html'}")
    return report_unscored(results)


def _delivered(entry: Any) -> str:
    """The share of this benchmark's scorable results whose method delivered a twin."""
    scorable = entry.results - entry.infra_errors
    if entry.delivered is None or scorable <= 0:
        return "—"
    return f"{entry.delivered / scorable:.0%}"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _run(args: argparse.Namespace) -> int:
    from real2simlab.orchestrate import load_experiment, run_experiment

    config, base = load_experiment(args.config)
    if args.backend:
        config = config.model_copy(
            update={"run": config.run.model_copy(update={"backend": args.backend})}
        )
    out = args.out or Path("runs") / _directory_name(config.name, args.config)
    results = run_experiment(
        config, out, base_dir=base, resume=not args.no_resume, report=not args.no_report
    )
    return _print_summary(results, out)


NAME_BYTES = 255
"""The longest file name most file systems allow, in bytes."""


def _directory_name(name: str, config: Path) -> str:
    """``name``, the experiment's, as the name of its run directory in ``runs/`` (``r2s run``
    without ``--out``). Refused, as a method label would be, when it is not one directory
    name: a path (``team/exp1``, ``../../elsewhere``, an absolute path), ``.`` or ``..``,
    blank, with a control character, or longer than a file name may be."""
    size = len(name.encode())
    problem = (
        "is blank" if not name.strip()
        else "is not a name of its own (. or ..)" if name in (".", "..")
        else "is a path (it holds /)" if "/" in name or (os.altsep and os.altsep in name)
        else "holds a control character" if any(unicodedata.category(c) == "Cc" for c in name)
        else f"has {size} bytes, more than a directory name may" if size > NAME_BYTES
        else None
    )  # fmt: skip
    if problem is None:
        return name
    shown = repr(name) if size <= NAME_BYTES else f"{name[:40]!r}..."
    raise UsageError(
        f"the experiment name {shown} in {config} {problem}, so it cannot name the run "
        "directory runs/<name>",
        fix="give the run directory with --out DIR, or make name in that file one directory "
        "name (letters, digits and _.:+- always are)",
    )


def register_eval(subparsers: Any) -> None:
    parser = add_parser(
        subparsers,
        "eval",
        "evaluate methods on tasks without writing a configuration file",
        "Evaluate methods on tasks without writing a configuration file.\n\n"
        "Examples:\n"
        "  r2s eval --benchmark synthetic --method oracle --method static --out runs/demo\n"
        "  r2s eval --tasks tasks/ --method codex:model=gpt-5-codex,reasoning_effort=medium\n"
        "  r2s eval --tasks tasks/ --method physical=fake_agent:mode=physical --backend local\n\n"
        "A method is NAME[:key=value,...], optionally prefixed by LABEL=. A value is read as\n"
        "JSON when it is JSON; one that holds a comma must be (a text in double quotes, a list\n"
        "or a table), in single quotes for the shell:\n"
        """  --method 'codex:gpus="0,1",allowlist=["a.example","b.example"]'\n\n""" + UNSCORED_EXIT,
    )
    parser.add_argument("--tasks", type=Path, action="append", default=[], help="task directories")
    parser.add_argument(
        "--benchmark", action="append", default=[], help="build and use a benchmark's tasks"
    )
    parser.add_argument("--select", action="append", help="with --benchmark: only these tasks")
    parser.add_argument(
        "--tier", action="append", dest="tiers", help="with --benchmark synthetic: tiers"
    )
    parser.add_argument("--include", action="append", default=[], help="glob filter on task ids")
    parser.add_argument("--method", action="append", required=True, help="a method (repeatable)")
    parser.add_argument("--out", type=Path, default=Path("runs/eval"), help="run directory")
    add_backend(parser)
    parser.add_argument("--seeds", type=seed_int, nargs="+", default=[0])
    parser.add_argument("--trials", type=positive_int, default=1)
    parser.add_argument("--concurrency", type=positive_int, default=4)
    parser.add_argument("--agent-concurrency", type=positive_int, default=2)
    parser.add_argument(
        "--max-agent-runs", type=non_negative_int, help="stop starting agents after this many"
    )
    parser.add_argument("--check-reproducibility", action="store_true", help="run each twin twice")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--name", default="eval")
    parser.set_defaults(
        handler=_eval,
        on_interrupt="run the same command again to resume: finished steps are reused",
    )


def _eval(args: argparse.Namespace) -> int:
    from real2simlab.orchestrate import parse_experiment, run_experiment

    if not args.tasks and not args.benchmark:
        raise UsageError(
            "r2s eval needs tasks to evaluate: give --tasks or --benchmark",
            fix="add --tasks DIR (task directories, e.g. written by `r2s bench build`) or "
            "--benchmark NAME (a benchmark of `r2s bench list`, built into the run directory)",
        )
    if (args.select or args.tiers) and not args.benchmark:
        from real2simlab.errors import SetupError

        raise SetupError(
            "--select and --tier choose among the tasks a --benchmark builds; "
            "with --tasks alone they would choose nothing, and every task would run",
            fix="choose task directories by id with --include GLOB "
            "(e.g. --include 'synthetic/bouncing_ball/*'), or give --benchmark",
        )
    data: dict[str, Any] = {
        "name": args.name,
        "tasks": {
            "paths": [str(path.resolve()) for path in args.tasks],
            "benchmarks": [
                {"name": name, "select": args.select, "tiers": args.tiers}
                for name in args.benchmark
            ],
            "include": args.include,
        },
        "methods": [parse_method(text) for text in args.method],
        "run": {
            "backend": args.backend,
            "seeds": args.seeds,
            "trials": args.trials,
            "concurrency": args.concurrency,
            "agent_concurrency": args.agent_concurrency,
            "check_reproducibility": args.check_reproducibility,
        },
        "budget": {"max_agent_runs": args.max_agent_runs},
    }
    config = parse_experiment(data, "command line")
    results = run_experiment(config, args.out, resume=not args.no_resume)
    return _print_summary(results, args.out)


def register_report(subparsers: Any) -> None:
    parser = add_parser(
        subparsers,
        "report",
        "write the Markdown, CSV and offline HTML report of a run directory",
    )
    parser.add_argument("run_dir", type=Path)
    parser.set_defaults(handler=_report)


def _report(args: argparse.Namespace) -> int:
    from real2simlab.report.build import ReportBuilder

    builder = ReportBuilder(args.run_dir)
    index = builder.build()
    for warning in builder.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(index)
    return 0
