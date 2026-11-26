"""Reading, checking and writing twin package directories.

A twin package is a directory with ``twin.json``, ``program/`` and optionally ``assets/``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from real2simlab._base import validation_problems
from real2simlab.errors import OutputExistsError, ProblemsError
from real2simlab.media import VIDEO_SUFFIXES
from real2simlab.twin.contract import ASSETS_DIR, PROGRAM_DIR, PROGRAM_PATH_PREFIX, TWIN_FILE
from real2simlab.twin.fingerprint import fingerprint, sha256_file
from real2simlab.twin.model import ProgramRules, TwinManifest

STAGING_SUFFIX = ".partial"
""":func:`save_twin` writes a package at ``<dest>.partial`` and renames it to ``dest`` once it
is complete: a save killed midway (kill -9, a crash) leaves ``<dest>.partial`` beside ``dest``,
which the next save there replaces. Commands that clear the directory they save a twin in
(``r2s train reward --work``, ``r2s twin check --out``) count it among what they wrote there."""


class InvalidTwinError(ProblemsError):
    """A twin directory is malformed or breaks the task's program rules."""


def program_of(directory: Path) -> Path:
    """The program in ``directory``: its ``program/`` when it is a twin package (``twin.json``
    and ``program/``), else ``directory`` itself. Everything that takes a program directory
    takes a twin package too, as ``r2s twin check`` does."""
    directory = Path(directory)
    if (directory / TWIN_FILE).is_file() and (directory / PROGRAM_DIR).is_dir():
        return directory / PROGRAM_DIR
    return directory


def has_entry_point(program: Path, entries: Iterable[str]) -> bool:
    """Whether the program directory ``program`` (a directory's :func:`program_of`) holds one of
    ``entries``, the entry point of a delivery convention (``run.py``): what makes a directory a
    program, whatever else it holds (a subdirectory with a ``run.py`` of its own is one of its
    files). A reward (``score_solution``) and ``r2s eval --method program:path=`` decide it
    alike, so that both take the same directory for the same program. A path this system
    cannot hold (a name too long) holds none."""
    try:
        return any((Path(program) / entry).is_file() for entry in entries)
    except OSError:
        return False


@dataclass(frozen=True)
class LoadedTwin:
    """A validated twin directory and its fingerprints at load time."""

    manifest: TwinManifest
    root: Path
    fingerprint: str
    """SHA-256 of the whole package: ``twin.json``, ``program/`` and ``assets/``."""
    program_fingerprint: str
    """SHA-256 of ``program/`` alone."""

    @property
    def program_dir(self) -> Path:
        return self.root / PROGRAM_DIR

    @property
    def assets_dir(self) -> Path:
        return self.root / ASSETS_DIR

    def current_fingerprint(self) -> str:
        return fingerprint(self.root)


def load_twin(
    root: Path,
    *,
    rules: ProgramRules | None = None,
) -> LoadedTwin:
    """Read and validate a twin directory. Raises ``InvalidTwinError`` listing every problem.

    ``rules`` are the task's program rules (defaults apply when ``None``).
    """
    root = Path(root)
    problems: list[str] = []
    manifest = _read_manifest(root / TWIN_FILE, problems)
    program_dir = root / PROGRAM_DIR
    if program_dir.is_symlink() or not program_dir.is_dir():
        problems.append(f"{PROGRAM_DIR}/: directory is required")
    else:
        problems.extend(program_problems(program_dir, manifest, rules or ProgramRules()))
    unexpected = (
        sorted(
            path.name
            for path in root.iterdir()
            if path.name not in {TWIN_FILE, PROGRAM_DIR, ASSETS_DIR}
        )
        if root.is_dir()
        else []
    )
    problems.extend(f"{name}: unexpected entry in the twin package" for name in unexpected)
    if manifest is not None:
        problems.extend(_asset_problems(root, manifest))
    if problems or manifest is None:
        raise InvalidTwinError(f"{root} is not a valid twin package", problems)
    return LoadedTwin(
        manifest=manifest,
        root=root,
        fingerprint=fingerprint(root),
        program_fingerprint=fingerprint(program_dir),
    )


def save_twin(
    manifest: TwinManifest,
    program_src: Path,
    dest: Path,
    *,
    rules: ProgramRules | None = None,
    assets_src: Path | None = None,
) -> LoadedTwin:
    """Copy ``program_src`` into a new twin directory at ``dest`` and write ``twin.json``.

    Refuses to overwrite an existing ``dest``. The program is checked before anything is
    written, so a refused save leaves nothing behind.
    """
    program_src, dest = Path(program_src), Path(dest)
    if dest.exists() or dest.is_symlink():
        raise OutputExistsError(
            f"twin destination already exists: {dest}", fix="choose a new destination"
        )
    if not program_src.is_dir():
        raise InvalidTwinError(f"cannot save twin {dest}", [f"{program_src} is not a directory"])
    problems = program_problems(program_src, manifest, rules or ProgramRules())
    if problems:
        raise InvalidTwinError(f"the program in {program_src} breaks the rules", problems)
    staging = dest.with_name(dest.name + STAGING_SUFFIX)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        shutil.copytree(program_src, staging / PROGRAM_DIR, symlinks=True)
        if assets_src is not None:
            shutil.copytree(assets_src, staging / ASSETS_DIR, symlinks=True)
        (staging / TWIN_FILE).write_text(manifest.dumps(), encoding="utf-8")
        loaded = load_twin(staging, rules=rules)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    staging.rename(dest)
    return LoadedTwin(
        manifest=loaded.manifest,
        root=dest,
        fingerprint=loaded.fingerprint,
        program_fingerprint=loaded.program_fingerprint,
    )


def program_problems(
    program_dir: Path, manifest: TwinManifest | None, rules: ProgramRules
) -> list[str]:
    """Every way ``program_dir`` breaks the twin rules or the task's program rules."""
    problems: list[str] = []
    total = 0
    forbidden = set(rules.forbidden_suffixes)
    if not rules.allow_video:
        forbidden |= VIDEO_SUFFIXES
    for path in sorted(program_dir.rglob("*")):
        relative = path.relative_to(program_dir).as_posix()
        if path.is_symlink():
            problems.append(f"{PROGRAM_DIR}/{relative}: symbolic links are not allowed")
            continue
        if path.is_file():
            total += path.stat().st_size
            if path.suffix.lower() in forbidden:
                kind = "video files" if path.suffix.lower() in VIDEO_SUFFIXES else "these files"
                problems.append(
                    f"{PROGRAM_DIR}/{relative}: {kind} ({path.suffix.lower()}) are not allowed "
                    "in the program"
                )
        elif not path.is_dir():
            problems.append(f"{PROGRAM_DIR}/{relative}: only regular files and directories")
    limit = rules.max_size_mb * 1024 * 1024
    if total > limit:
        problems.append(
            f"{PROGRAM_DIR}/: {total / 1048576:.1f} MB exceeds the limit of "
            f"{rules.max_size_mb:g} MB"
        )
    if manifest is not None:
        for name in manifest.program_files:
            if not (program_dir / name).is_file():
                problems.append(
                    f"entrypoint: {PROGRAM_PATH_PREFIX}{name} is not a file in {PROGRAM_DIR}/"
                )
    return problems


def _read_manifest(path: Path, problems: list[str]) -> TwinManifest | None:
    if not path.is_file():
        problems.append(f"{TWIN_FILE}: file is required")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        problems.append(f"{TWIN_FILE}: cannot read JSON: {error}")
        return None
    try:
        return TwinManifest.model_validate(data)
    except ValidationError as error:
        problems.extend(validation_problems(error, f"{TWIN_FILE}: "))
        return None


def _asset_problems(root: Path, manifest: TwinManifest) -> list[str]:
    problems: list[str] = []
    for asset in manifest.assets:
        path = root / ASSETS_DIR / asset.path
        if not path.is_file():
            if asset.url is None:
                problems.append(f"assets: {asset.path} is missing and has no url")
            continue
        if path.stat().st_size != asset.size or sha256_file(path) != asset.sha256:
            problems.append(f"assets: {asset.path} does not match its declared size and sha256")
    return problems
