"""Twin packages (``r2s-twin/1.0``): what a method delivers and the rules its program follows."""

from real2simlab.twin.contract import (
    ASSETS_DIR,
    ASSETS_DIR_ENV,
    OUTPUT_DIR_ENV,
    PROGRAM_DIR,
    SEED_ENV,
    TWIN_FILE,
)
from real2simlab.twin.fingerprint import fingerprint, sha256_file, text_digest
from real2simlab.twin.io import (
    InvalidTwinError,
    LoadedTwin,
    load_twin,
    program_of,
    program_problems,
    save_twin,
)
from real2simlab.twin.model import (
    AgentInfo,
    Asset,
    EnvironmentRef,
    Origin,
    ProgramRules,
    Runtime,
    TwinManifest,
)

__all__ = [
    "ASSETS_DIR",
    "ASSETS_DIR_ENV",
    "OUTPUT_DIR_ENV",
    "PROGRAM_DIR",
    "SEED_ENV",
    "TWIN_FILE",
    "AgentInfo",
    "Asset",
    "EnvironmentRef",
    "InvalidTwinError",
    "LoadedTwin",
    "Origin",
    "ProgramRules",
    "Runtime",
    "TwinManifest",
    "fingerprint",
    "load_twin",
    "program_of",
    "program_problems",
    "save_twin",
    "sha256_file",
    "text_digest",
]
