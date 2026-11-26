"""Helpers shared by the ``r2s`` subcommands."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from real2simlab.errors import UsageError
from real2simlab.sandbox.base import Backend, ensure_chosen_ready, get_backend

HELP_COLUMN = 24


class Formatter(argparse.RawDescriptionHelpFormatter):
    """Help with option and subcommand descriptions always starting at column ``HELP_COLUMN``.

    argparse picks that column from the longest option or subcommand name, and Python 3.11 and
    3.13 measure subcommand names differently; a fixed column makes ``--help`` (and the
    generated CLI reference) the same on every Python version.
    """

    def __init__(
        self,
        prog: str,
        indent_increment: int = 2,
        max_help_position: int = HELP_COLUMN,
        width: int | None = None,
    ) -> None:
        super().__init__(prog, indent_increment, max_help_position, width)
        self._action_max_length = max_help_position


FORMATTER = Formatter


def add_parser(
    subparsers: Any, name: str, help_text: str, description: str | None = None
) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        name, help=help_text, description=description or help_text, formatter_class=FORMATTER
    )
    return parser


def add_backend(parser: argparse.ArgumentParser, default: str = "docker") -> None:
    parser.add_argument(
        "--backend",
        default=default,
        help=f"sandbox backend: docker, bwrap or local (default: {default}; "
        "local has no isolation)",
    )


def backend_from(args: argparse.Namespace) -> Backend:
    backend = get_backend(args.backend)
    ensure_chosen_ready(backend)
    return backend


def positive_int(text: str) -> int:
    """An argparse type: a whole number of at least 1 (``--count``, ``--trials``)."""
    return _whole(text, 1)


def non_negative_int(text: str) -> int:
    """An argparse type: a whole number of at least 0 (``--seed``, ``--max-agent-runs``)."""
    return _whole(text, 0)


def seed_int(text: str) -> int:
    """An argparse type: a seed, a whole number from 0 to ``2**32 - 1`` (what a twin is given,
    ``R2S_SEED``); a larger one was taken and stopped the command later (after the build of an
    evaluation, or as a file name too long for ``r2s train tasks``)."""
    from real2simlab.twin.contract import MAX_SEED

    value = _whole(text, 0)
    if value >= MAX_SEED:
        shown = text if len(text) <= 24 else f"{text[:12]}...({len(text)} digits)"
        raise argparse.ArgumentTypeError(f"must be at most {MAX_SEED - 1}, not {shown}")
    return value


def positive_float(text: str) -> float:
    """An argparse type: a finite number above 0 (``--timeout``)."""
    value = finite_float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be above 0, not {text}")
    return value


def finite_float(text: str) -> float:
    """An argparse type: a finite number (not ``nan`` or ``inf``)."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, not {text!r}") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"expected a finite number, not {text!r}")
    return value


def _whole(text: str, least: int) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, not {text!r}") from None
    if value < least:
        raise argparse.ArgumentTypeError(f"must be at least {least}, not {value}")
    return value


def parse_value(text: str) -> Any:
    """A JSON value, or the text itself if it is not JSON."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def parse_options(items: Sequence[str], flag: str = "--option") -> dict[str, Any]:
    """``key=value`` pairs, one per item (``--option``), values parsed as JSON when they
    are."""
    options: dict[str, Any] = {}
    for item in items:
        key, equals, value = item.partition("=")
        if not equals or not key.strip():
            raise UsageError(
                f"{flag} {item!r} is not key=value",
                fix=f"give each option as {flag} NAME=VALUE, one per {flag} (the value is read "
                "as JSON when it is JSON)",
            )
        options[key.strip()] = parse_value(value)
    return options


IN_A_FILE = "or give the method in an experiment file ([[methods]], `r2s run`)"
_JSON = json.JSONDecoder()


def parse_method(text: str, given: str = "--method") -> dict[str, Any]:
    """``name[:key=value,key=value]`` or ``label=name[:...]`` → a method table.

    Values are parsed as JSON when they are, and options are separated by the commas outside
    them: a JSON list, table or text keeps its own (``allowlist=["a.example","b.example"]``,
    ``gpus="0,1"``). ``given`` is where the method was given, for the error that names a part
    that is not an option: ``--method`` (which an experiment file can replace) or
    ``r2s train fit``."""
    label = None
    head, _, rest = text.partition(":")
    if "=" in head:
        label, _, head = head.partition("=")
    entry: dict[str, Any] = {"name": head}
    if label:
        entry["label"] = label
    entry.update(_method_options(text, len(text) - len(rest), given))
    return entry


def _method_options(text: str, start: int, given: str) -> dict[str, Any]:
    """The ``key=value,...`` of the method ``text``, from ``start``."""
    options: dict[str, Any] = {}
    last: tuple[str, int] | None = None  # the option before: its key, where its value starts
    position = start
    while position < len(text):
        comma = text.find(",", position)
        end = len(text) if comma < 0 else comma
        key, equals, _ = text[position:end].partition("=")
        if not equals or not key.strip():
            if text[position:end].strip():
                raise _not_an_option(text, position, end, last, given)
            position = end + 1  # nothing between two commas, or after the last one
            continue
        value = position + len(key) + 1
        found = _json_end(text, value)
        end = end if found is None else found
        options[key.strip()] = parse_value(text[value:end])
        last = (key.strip(), value)
        position = end + 1
    return options


def _json_end(text: str, start: int) -> int | None:
    """Where the value at ``start`` ends when it is a JSON list, table or text: at the comma
    after it, or at the end; ``None`` when it is not one, or when more follows it."""
    begin = len(text) - len(text[start:].lstrip())
    if begin == len(text) or text[begin] not in '[{"':
        return None
    try:
        _, end = _JSON.raw_decode(text, begin)
    except ValueError:
        return None
    end = len(text) - len(text[end:].lstrip())
    return end if end == len(text) or text[end] == "," else None


def _not_an_option(
    text: str, start: int, end: int, last: tuple[str, int] | None, given: str
) -> UsageError:
    """The error for ``text[start:end]``, a part of the method ``text`` that is not
    ``key=value``: after an option, most likely the rest of a value that holds a comma."""
    also = f"; {IN_A_FILE}" if given == "--method" else ""
    piece = text[start:end].strip()
    general = (
        "write options as NAME:key=value,key=value, and a value that holds a comma as JSON (a "
        "text in double quotes, a list or a table) with the whole method in single quotes: "
        f"""{given} 'NAME:gpus="0,1",allowlist=["a.example","b.example"]'{also}"""
    )
    if last is None or "=" in piece:  # the first part, or one with nothing before its "="
        why = "has no option name before =" if "=" in piece else "is not an option (key=value)"
        return UsageError(f"{given} {text!r}: {piece!r} {why}", fix=general)
    said = f"{given} {text!r}: {piece!r} is not an option (key=value)"
    key, value = last
    stop = end  # with the parts after it that are not options either: what the value held
    while stop < len(text):
        following = text.find(",", stop + 1)
        after = len(text) if following < 0 else following
        if "=" in text[stop + 1 : after]:
            break
        stop = after
    meant = text[value:stop]
    if meant.lstrip().startswith(("[", "{", '"')):  # meant as JSON
        try:
            json.loads(meant)
        except ValueError as error:
            return UsageError(
                f"{given} {text!r}: {key}={meant} is not valid JSON ({error})", general
            )
    written = text[:value] + json.dumps(meant) + text[stop:]
    return UsageError(
        f"{said}; options are separated by commas",
        fix=f"if the comma is part of the value of {key}, write that value as JSON text: "
        f"{given} {shlex.quote(written)}{also}",
    )


def table(rows: Sequence[Sequence[Any]], header: Sequence[str]) -> str:
    """A Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("—" if cell is None else str(cell) for cell in row) + " |")
    return "\n".join(lines)


def emit(data: Any, as_json: bool, text: str) -> None:
    if as_json:
        json.dump(data, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(text)


CAUSES_SHOWN = 3
UNSCORED_EXIT = (
    "Exits with status 1 when nothing could be scored (every result failed because of the\n"
    "infrastructure, not a method)."
)
"""What the experiment commands' help says about their exit status."""


def report_unscored(results: Any) -> int:
    """Say which results of an experiment the infrastructure kept from being scored, and why
    (on stderr): they are left out of every number and redone by the same command once the
    cause is fixed. Returns the command's exit status: 1 when nothing could be scored at all,
    which no number shows and running the same command again on this machine does not fix."""
    unscored = [row for row in results.rows if row.get("infra_error")]
    if not unscored:
        return 0
    causes = list(dict.fromkeys(str(row["infra_error"]) for row in unscored))
    print(
        f"\n{len(unscored)} result(s) could not be scored because the infrastructure failed; "
        "they are left out of every number above. Why:",
        file=sys.stderr,
    )
    for cause in causes[:CAUSES_SHOWN]:
        print(f"  - {cause}", file=sys.stderr)
    if len(causes) > CAUSES_SHOWN:
        print(f"  - and {len(causes) - CAUSES_SHOWN} other cause(s) in r2s.log", file=sys.stderr)
    if len(unscored) < len(results.rows):
        print(
            "Once that is fixed, run the same command again to score them (what was scored is "
            "reused).",
            file=sys.stderr,
        )
        return 0
    print(
        "error: nothing could be scored: every result failed because of the infrastructure "
        "(above), not because of a method",
        file=sys.stderr,
    )
    print("  fix: fix the cause above, then run the same command again", file=sys.stderr)
    return 1


TASKS_INTERRUPTED = (
    "the tasks written so far are complete, the one being written is not; run the same command "
    "with --overwrite (it builds them all again), or give a new --out"
)
"""What to do after a command that writes tasks was interrupted (its tasks are not reused)."""


def note_other_tasks(out: Path, written: Sequence[Path]) -> None:
    """Say (on standard error) that ``out`` also holds tasks this command did not write, such as
    an earlier command's with another count, seed or selection: ``--overwrite`` replaces only
    the tasks a command writes, and every command given ``--tasks out`` uses them all."""
    from real2simlab.task import find_tasks

    if not out.is_dir():
        return
    mine = {Path(path).resolve() for path in written}
    others = [path for path in find_tasks(out) if path.resolve() not in mine]
    if others:
        print(
            f"note: {out} also holds {len(others)} task(s) this command did not write (e.g. "
            f"{others[0]}); every command given --tasks {out} uses them too: remove them, or "
            "write to a new --out",
            file=sys.stderr,
        )
