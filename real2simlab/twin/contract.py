"""The run contract shared by twin programs and the runner (``r2s-twin/1.0``).

A twin program is run with its working directory set to its program directory, mounted
read-only, with no network, and with:

- ``R2S_OUTPUT_DIR``: an empty directory the program writes its complete output into;
- ``R2S_SEED``: always set, a non-negative integer every source of randomness derives from;
- ``R2S_ASSETS_DIR``: set when the twin declares assets, a read-only directory holding them.

The program may write only to ``R2S_OUTPUT_DIR`` and its scratch directory (``TMPDIR``, also
``HOME``). The same seed must give the same output.
"""

TWIN_FILE = "twin.json"
PROGRAM_DIR = "program"
ASSETS_DIR = "assets"

OUTPUT_DIR_ENV = "R2S_OUTPUT_DIR"
SEED_ENV = "R2S_SEED"
ASSETS_DIR_ENV = "R2S_ASSETS_DIR"

MAX_SEED = 2**32

PROGRAM_PATH_PREFIX = "./"
"""Entrypoint items starting with this prefix name files inside ``program/``."""

SANDBOX_PROGRAM = "/r2s/program"
SANDBOX_OUTPUT = "/r2s/output"
SANDBOX_ASSETS = "/r2s/assets"
SANDBOX_TMP = "/tmp"
"""Where each directory appears inside the sandbox for the ``r2s`` convention."""
