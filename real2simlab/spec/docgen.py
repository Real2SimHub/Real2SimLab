"""Generated parts of the documentation, kept in sync with the code by tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast


def cli_reference() -> str:
    """Every ``r2s`` command's ``--help``, as Markdown."""
    import os

    from real2simlab.cli.main import build_parser

    previous = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = "100"  # argparse wraps help to the terminal width
    try:
        parser = build_parser()
        return _cli_sections(parser)
    finally:
        if previous is None:
            os.environ.pop("COLUMNS", None)
        else:
            os.environ["COLUMNS"] = previous


def _cli_sections(parser: Any) -> str:
    import argparse

    sections = [
        "# Command line reference",
        "",
        "Generated from `r2s ... --help`. Every command also prints this with `--help`.",
        "",
    ]

    def walk(current: argparse.ArgumentParser, name: str, level: int) -> None:
        sections.extend(
            [f"{'#' * level} `{name}`", "", "```text", current.format_help().rstrip(), "```", ""]
        )
        for action in current._actions:  # pyright: ignore[reportPrivateUsage]
            if isinstance(action, argparse._SubParsersAction):  # pyright: ignore[reportPrivateUsage]
                choices = cast("dict[str, argparse.ArgumentParser]", action.choices)  # pyright: ignore[reportUnknownMemberType]
                for child_name, child in sorted(choices.items()):
                    walk(child, f"{name} {child_name}", min(level + 1, 4))

    walk(parser, "r2s", 2)
    return "\n".join(sections)


def write_cli_reference(path: Path) -> bool:
    text = cli_reference()
    current = path.read_text(encoding="utf-8") if path.is_file() else ""
    if current != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return True
    return False
