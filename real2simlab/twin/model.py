"""``twin.json`` (``r2s-twin/1.0``): what a twin package declares."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Annotated, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    Field,
    NonNegativeInt,
    PositiveFloat,
    field_validator,
    model_validator,
)

from real2simlab._base import FINITE_CONFIG, StrictModel
from real2simlab.capture.model import RelPath
from real2simlab.spec import SpecId, spec_validator
from real2simlab.twin.contract import (
    PROGRAM_PATH_PREFIX,
    SANDBOX_OUTPUT,
    SANDBOX_PROGRAM,
)

TWIN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _check_twin_id(value: str) -> str:
    if not TWIN_ID.match(value):
        raise ValueError(f"twin id {value!r} must match {TWIN_ID.pattern}")
    return value


def _check_sha256(value: str) -> str:
    if not SHA256.match(value):
        raise ValueError("must be a lowercase hex SHA-256 digest")
    return value


TwinId = Annotated[str, AfterValidator(_check_twin_id)]
Sha256 = Annotated[str, AfterValidator(_check_sha256)]


class ProgramRules(StrictModel):
    """Limits on the delivered ``program/`` directory."""

    model_config = FINITE_CONFIG

    max_size_mb: PositiveFloat = 50.0
    allow_video: bool = False
    """Video files are forbidden by default, so a twin cannot copy the recording."""
    forbidden_suffixes: tuple[str, ...] = ()
    """More forbidden file suffixes, e.g. ``.npz``; lowercase, starting with ``.``."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        bad = [s for s in self.forbidden_suffixes if not s.startswith(".") or s != s.lower()]
        if bad:
            raise ValueError(f"suffixes must be lowercase and start with '.': {bad}")
        return self


class EnvironmentRef(StrictModel):
    """The runtime image the program runs in."""

    image: str = Field(min_length=1)
    """A registered environment name (``r2s-python``) or an image reference."""
    digest: str | None = None
    """Image id or registry digest recorded when the twin was frozen."""


class Asset(StrictModel):
    """A large file the program needs, stored in ``assets/`` and verified by its digest."""

    path: RelPath
    sha256: Sha256
    size: NonNegativeInt
    url: str | None = None
    """Where to fetch the file when it is not stored with the twin."""


class AgentInfo(StrictModel):
    name: str = Field(min_length=1)
    version: str | None = None


class Origin(StrictModel):
    """Who built the twin, for which task, and when."""

    method: str = Field(min_length=1)
    agent: AgentInfo | None = None
    model: str | None = None
    """The model behind an agent; absent for methods without one."""
    task_id: str = Field(min_length=1)
    task_version: str | None = None
    created_at: AwareDatetime
    privileged: bool = False
    """The method read hidden task data, e.g. the oracle baseline."""
    notes: str | None = None


class Runtime(StrictModel):
    """Where the program and its output appear inside the sandbox."""

    program_path: str = SANDBOX_PROGRAM
    output_path: str = SANDBOX_OUTPUT
    """For the ``build_sh_world`` convention, fixed paths such as ``/workspace/world``."""
    replaceable_output: bool = False
    """The program may delete and recreate its output directory (4DCodeBench's ``build.sh``
    replaces ``/workspace/world``): the output's parent is then a writable scratch directory,
    of which only the output directory is kept."""

    @field_validator("program_path", "output_path")
    @classmethod
    def _check_absolute(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts or path == PurePosixPath("/"):
            raise ValueError(f"must be an absolute path below /: {value!r}")
        if any(
            path == PurePosixPath(p) or PurePosixPath(p) in path.parents for p in ("/proc", "/dev")
        ):
            raise ValueError(f"cannot be inside /proc or /dev: {value!r}")
        return value

    @model_validator(mode="after")
    def _check_replaceable(self) -> Self:
        if self.replaceable_output and PurePosixPath(self.output_path).parent == PurePosixPath("/"):
            raise ValueError("a replaceable output needs a parent directory other than /")
        return self


class TwinManifest(StrictModel):
    """The content of ``twin.json``. Written by the Lab or a method adapter, never by an agent."""

    spec: Annotated[str, spec_validator(SpecId.TWIN)] = SpecId.TWIN.value
    twin_id: TwinId
    entrypoint: tuple[str, ...] = Field(min_length=1)
    """Command run inside ``program/``; items starting with ``./`` name files in ``program/``."""
    environment: EnvironmentRef
    output_format: str = Field(min_length=1)
    runtime: Runtime = Runtime()
    assets: tuple[Asset, ...] = ()
    origin: Origin

    @field_validator("entrypoint")
    @classmethod
    def _check_entrypoint(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            if not item:
                raise ValueError("items must be non-empty")
            if item.startswith(PROGRAM_PATH_PREFIX):
                path = PurePosixPath(item.removeprefix(PROGRAM_PATH_PREFIX))
                if not path.parts or ".." in path.parts or path.is_absolute():
                    raise ValueError(f"{item!r} must name a file inside program/")
        return value

    @model_validator(mode="after")
    def _check(self) -> Self:
        paths = [asset.path for asset in self.assets]
        if len(set(paths)) != len(paths):
            raise ValueError("asset paths must be distinct")
        return self

    @property
    def program_files(self) -> tuple[str, ...]:
        """Paths, relative to ``program/``, of the files the entrypoint names."""
        return tuple(
            item.removeprefix(PROGRAM_PATH_PREFIX)
            for item in self.entrypoint
            if item.startswith(PROGRAM_PATH_PREFIX)
        )
