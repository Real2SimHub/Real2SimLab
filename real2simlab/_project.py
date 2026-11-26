"""Where the project lives: the repository, its documentation and its published images; and how
it is installed (from a clone of the repository; it is not published on PyPI)."""

from __future__ import annotations

import shlex
from pathlib import Path

REPOSITORY = "https://github.com/Real2SimHub/Real2SimLab"
"""The source repository (also the base of release downloads)."""
DOCUMENTATION = "https://github.com/Real2SimHub/Real2SimLab/tree/main/docs"
IMAGE_REGISTRY = "ghcr.io/real2simhub/"
"""Where the release workflow publishes the runtime images: the GitHub Container Registry of
the repository's owner. It is also the prefix of the images' local names."""


def release_asset(version: str, name: str) -> str:
    """The download URL of a file attached to the GitHub release of ``version``."""
    return f"{REPOSITORY}/releases/download/v{version}/{name}"


def install_command(extra: str | None = None) -> str:
    """The command that installs Real2SimLab, or one of its optional extras, into the current
    environment: an editable install from the clone this package was imported from (the
    project is installed from its repository, not from PyPI)."""
    root = Path(__file__).resolve().parents[1]
    if (root / "pyproject.toml").is_file():
        target = str(root)
        suffix = ""
    else:  # installed some other way: point at a clone
        target = "."
        suffix = f"  # in your clone of {REPOSITORY}"
    spec = f"{target}[{extra}]" if extra else target
    return f"pip install -e {shlex.quote(spec)}{suffix}"
