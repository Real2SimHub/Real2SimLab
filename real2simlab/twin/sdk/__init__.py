"""The twin SDK, ``r2s_twin.py``: a single file that needs only NumPy.

It is copied into runtime images and agent workspaces; frozen twins never import Real2SimLab.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

SDK_FILE = "r2s_twin.py"


def sdk_path() -> Path:
    """Path of the SDK source file inside the installed package."""
    return Path(str(files("real2simlab.twin.sdk").joinpath(SDK_FILE)))


def sdk_source() -> str:
    return sdk_path().read_text(encoding="utf-8")
