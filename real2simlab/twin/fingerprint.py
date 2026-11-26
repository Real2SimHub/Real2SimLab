"""Content fingerprints that make frozen programs and directories verifiable."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from pathlib import Path

_CHUNK = 1 << 20

OS_LITTER = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})
"""Files a file manager leaves in a directory it opened (macOS Finder, Windows Explorer)."""


def is_os_litter(name: str) -> bool:
    """Whether a file named ``name`` is a file manager's, not anyone's content (``.DS_Store``,
    macOS's ``._`` companions, ``Thumbs.db``): content checks of what the Lab wrote skip it."""
    return name in OS_LITTER or name.startswith("._")


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class SymbolicLinkError(ValueError):
    """What :func:`fingerprint` raises for a symbolic link in the directory: a ``ValueError``
    of its own, so a caller that reports links later (once it has copied the directory, say)
    never takes another error for one."""


EXECUTABLE = 1 << 63
"""Set in a file's size field for a file someone may execute: a program that runs a helper
(``./helper.sh``) behaves differently without the bit, so it is content. Files without it
fingerprint as before."""


def fingerprint(directory: Path, *, exclude: Iterable[str] = ()) -> str:
    """SHA-256 over every file's relative path, executable bit and bytes, in sorted path order.

    Directories contribute only through the files they hold, so an empty directory does not
    change the fingerprint, nor does a file manager's litter (:func:`is_os_litter`): looking
    at a run directory in Finder does not make its programs "changed". ``exclude`` lists
    top-level names to skip. Raises :class:`SymbolicLinkError` (a ``ValueError``) on a symbolic
    link.

    A path counts as the bytes the file system holds for it (:func:`os.fsencode`): the same as
    its UTF-8 encoding for every name that is UTF-8, and a name that is not (a file unpacked
    from an archive made on Windows, its name in another encoding) still has a fingerprint.
    """
    directory = Path(directory)
    skipped = set(exclude)
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*"), key=lambda p: p.relative_to(directory).as_posix()):
        relative = path.relative_to(directory)
        if relative.parts[0] in skipped or is_os_litter(path.name):
            continue
        if path.is_symlink():
            raise SymbolicLinkError(f"symbolic link in {directory}: {relative.as_posix()}")
        if not path.is_file():
            continue
        name = os.fsencode(relative.as_posix())
        digest.update(len(name).to_bytes(8, "big") + name)
        info = path.stat()
        digest.update(
            (info.st_size | (EXECUTABLE if info.st_mode & 0o111 else 0)).to_bytes(8, "big")
        )
        with path.open("rb") as file:
            while chunk := file.read(_CHUNK):
                digest.update(chunk)
    return digest.hexdigest()


def text_digest(*parts: str) -> str:
    """SHA-256 of strings joined unambiguously; used for cache keys."""
    digest = hashlib.sha256()
    for part in parts:
        data = part.encode()
        digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()
