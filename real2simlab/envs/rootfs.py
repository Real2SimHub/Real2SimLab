"""Unpacking container images into root file systems for the ``bwrap`` backend.

An image saved with ``docker save`` (a tarball, also written by ``r2s env export``) is unpacked
layer by layer, applying OCI whiteouts, into ``<cache>/rootfs/<name>-<id>/rootfs``, next to
``image.json`` (the image's id, environment, working directory). Nothing needs root: device
nodes are skipped, ownership is the unpacking user's, set-id bits are cleared. A layer's files
are written as they are read: memory does not grow with the image.

A tarball names the image it holds (the names ``docker save`` gives it): it is unpacked as one
of those, never registered as another image (a misspelt name, the wrong file). One saved by
image id names none, and is unpacked as the name given, with a note that it was not checked.

Each layer is checked as it is read, as ``docker load`` checks it: its contents (uncompressed,
read to their end) must have the digest its image's configuration lists (``rootfs.diff_ids``,
which the image's id covers), and a file stored under its digest (``blobs/sha256/...``, the
layout of Docker 25 and later) that digest. A damaged copy is refused before anything of it is
registered: an image's id names what was unpacked.

Symbolic links inside the image are resolved as if the root file system were ``/``, so an
entry never escapes the root file system, even through absolute links.

On a machine without Docker (e.g. a cluster): ``r2s env export r2s-python --out r2s-python.tar``
where Docker exists, copy the file, then ``r2s env unpack r2s-python --from-tar r2s-python.tar``.
"""

from __future__ import annotations

import bz2
import functools
import gzip
import hashlib
import json
import lzma
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zlib
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Any, cast

from real2simlab.errors import SetupError
from real2simlab.sandbox.process import run_captured
from real2simlab.vendor import cache_dir

WHITEOUT = ".wh."
OPAQUE = ".wh..wh..opq"
INDEX = "index.json"
MOUNT_ROOTS = ("/r2s", "/workspace", "/app")
"""Where a sandbox mounts programs, workspaces and outputs (``/app`` for Blender scenes): every
unpacked image has these directories, since bubblewrap cannot create one in a read-only root.
A tmpfs goes there to hold the mount points beneath."""
MAX_LINKS = 40
UNPACKED = re.compile(r".+-[0-9a-f]{12}")
"""The name of an unpacked image's directory: its reference, then its configuration's digest."""
STAGED = r".+-[0-9a-f]{12}\."
"""How the name of an unpack's working copy starts: the directory it becomes and a dot, before
the tag that ``_fs.held_path`` gives it."""
CHUNK = 1 << 20
RECORD_LIMIT = 64 << 20
"""The largest ``manifest.json`` or configuration read (real ones have a few kilobytes)."""
BLOB = re.compile(r"blobs/sha256/([0-9a-f]{64})")
"""A file of an image stored under its own digest (the OCI layout ``docker save`` writes since
Docker 25)."""
DIFF_ID = re.compile(r"sha256:[0-9a-f]{64}")
"""A layer's digest as an image's configuration lists it (``rootfs.diff_ids``)."""


@dataclass(frozen=True)
class Rootfs:
    """An unpacked image."""

    reference: str
    image_id: str
    root: Path
    env: dict[str, str]
    workdir: str

    @property
    def rootfs(self) -> Path:
        return self.root / "rootfs"


def rootfs_dir() -> Path:
    return cache_dir() / "rootfs"


@contextmanager
def _locked() -> Iterator[None]:
    """Hold the cache's lock while the index is read and written and an unpacked image is put
    in place: unpacks that run at the same time (one per job on a cluster) each keep their
    registration, and none moves a directory another is replacing."""
    import fcntl

    rootfs_dir().mkdir(parents=True, exist_ok=True)
    with (rootfs_dir() / "index.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _use_lock(root: Path) -> Path:
    """The lock file of the unpacked image at ``root``: next to it, so it outlives a repair."""
    return root.with_name(root.name + ".lock")


@contextmanager
def in_use(rootfs: Rootfs) -> Iterator[None]:
    """Hold a shared lock on ``rootfs`` while a sandbox uses it: ``r2s env unpack --force``
    does not replace an image under a running program (which would then fail for a reason
    that is not its own). Without write access to the cache (a shared, read-only one), only
    an existing lock file can be used; nobody here can replace the image either."""
    import fcntl

    path = _use_lock(rootfs.root)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CREAT, 0o644)
    except OSError:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            yield
            return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        yield
    finally:
        os.close(descriptor)


@contextmanager
def _replacing(root: Path, reference: str) -> Iterator[None]:
    """Hold the unpacked image at ``root`` exclusively while it is removed or replaced;
    refuse at once while a sandbox uses it (``in_use``)."""
    import fcntl

    root.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(_use_lock(root), os.O_RDONLY | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SetupError(
                f"the unpacked image {reference} ({root}) is in use by a running sandbox (an "
                "evaluation, a reward or a twin check on this machine); replacing it would make "
                "that program fail",
                fix="wait until it finishes (or stop it), then run the same command again",
            ) from None
        yield
    finally:
        os.close(descriptor)


def identity(rootfs: Rootfs) -> tuple[int, int, bytes] | None:
    """What tells this unpacked image from one put in its place (``--force``, by hand): its
    tree's inode and its record; ``None`` when it is gone."""
    try:
        status = rootfs.rootfs.stat()
        return status.st_dev, status.st_ino, (rootfs.root / "image.json").read_bytes()
    except OSError:
        return None


def _meta(root: Path) -> dict[str, Any] | None:
    """The ``image.json`` of the unpacked image at ``root``, or ``None`` when it is missing or
    damaged (not an object with the image's id)."""
    try:
        meta = json.loads((root / "image.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or not isinstance(meta.get("image_id"), str):
        return None
    return cast(dict[str, Any], meta)


def _unpacked(root: Path) -> bool:
    """Whether ``root`` holds a whole unpacked image: its record and its tree (a cleaner may
    have removed the tree and left the record)."""
    return _meta(root) is not None and (root / "rootfs").is_dir()


def _index() -> dict[str, str]:
    """The registered images (reference to directory). A damaged index is rebuilt from the
    unpacked images' own records, which name their reference (not from the working copy of an
    unpack, which may have written its record when it was killed)."""
    path = rootfs_dir() / INDEX
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        index = None
    if isinstance(index, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in index.items()
    ):
        return cast(dict[str, str], index)
    rebuilt: dict[str, str] = {}
    for root in sorted(rootfs_dir().iterdir()):
        if UNPACKED.fullmatch(root.name) is None or not root.is_dir():
            continue
        meta = _meta(root)
        if meta is not None and isinstance(meta.get("reference"), str):
            rebuilt[meta["reference"]] = root.name
    return rebuilt


def _write_index(index: dict[str, str]) -> None:
    path = rootfs_dir() / INDEX
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{INDEX}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def mount_points_problem(rootfs: Rootfs) -> str | None:
    """Create the mount points an image unpacked by an earlier version lacks (``MOUNT_ROOTS``);
    why they cannot be created, or ``None``."""
    for root in MOUNT_ROOTS:
        path = rootfs.rootfs / root.lstrip("/")
        if path.is_dir():
            continue
        try:
            path.mkdir()
        except OSError as error:
            return (
                f"the unpacked image {rootfs.reference} has no {root}, and making it failed: "
                f"{error}"
            )
    return None


def find_rootfs(reference: str) -> Rootfs | None:
    """The unpacked root file system of ``reference``, if it exists.

    Raises:
        SetupError: it was unpacked, but is damaged: its record (``image.json``) is missing or
            damaged, or its tree was removed (e.g. by a cleaner) and its record left.
    """
    name = _index().get(reference)
    if name is None:
        return None
    root = rootfs_dir() / name
    if not root.is_dir():
        return None
    meta = _meta(root)
    if meta is None:
        record = root / "image.json"
        raise SetupError(
            f"the record of the unpacked image {reference} ({record}) is "
            + ("damaged" if record.exists() else "missing"),
            fix=unpack_again(reference),
        )
    if not (root / "rootfs").is_dir():
        raise SetupError(
            f"the unpacked image {reference} ({root}) is damaged: its root file system "
            f"{root / 'rootfs'} is gone (removed by hand, or by a cleaner?)",
            fix=unpack_again(reference),
        )
    env = meta.get("env")
    return Rootfs(
        reference=reference,
        image_id=meta["image_id"],
        root=root,
        env={str(k): str(v) for k, v in env.items()} if isinstance(env, dict) else {},
        workdir=str(meta.get("workdir") or "/"),
    )


def resolve_in_root(root: Path, path: PurePosixPath, *, follow_last: bool = False) -> Path:
    """The host path of ``path`` in ``root``, following symbolic links as if ``root`` were ``/``."""
    parts = [part for part in path.parts if part not in ("/", "", ".")]
    current = PurePosixPath("/")
    hops = 0
    while parts:
        part = parts.pop(0)
        if part == "..":
            current = current.parent
            continue
        candidate = current / part
        host = root / candidate.relative_to("/")
        last = not parts
        if host.is_symlink() and (follow_last or not last):
            hops += 1
            if hops > MAX_LINKS:
                raise SetupError(f"too many symbolic links resolving {path} in the image")
            target = PurePosixPath(os.readlink(host))
            base = PurePosixPath("/") if target.is_absolute() else current
            parts = [*(p for p in (base / target).parts if p != "/"), *parts]
            current = PurePosixPath("/")
            continue
        current = candidate
    return root / current.relative_to("/")


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _apply_layer(
    root: Path, layer: IO[bytes], scratch: Path, check: Callable[[int], None] | None = None
) -> int:
    """Extract one layer tarball (uncompressed) into ``root``; return the number of skipped
    special files.

    The layer is read once. Each file's data is written to ``scratch`` (a new directory on the
    file system of ``root``) as it is read, and moved into place once the layer's whiteouts are
    applied: memory does not grow with the layer (all of its files were held at once: 1.4 GiB
    for the Blender image's largest layer). Once the layer is read, and before anything of it
    is applied, ``check`` is given the offset where reading its entries ended (an archive's end,
    or a header that could not be read); it raises when the layer is not the one expected."""
    skipped = 0
    scratch.mkdir()
    members: list[tuple[tarfile.TarInfo, Path | None]] = []
    with tarfile.open(fileobj=layer, mode="r|") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            data: Path | None = None
            if member.isfile() and not name.name.startswith(WHITEOUT) and ".." not in name.parts:
                data = scratch / str(len(members))
                extracted = archive.extractfile(member)
                with data.open("wb") as stream:
                    if extracted is not None:
                        shutil.copyfileobj(extracted, stream, CHUNK)
            members.append((member, data))
        end = archive.offset
    if check is not None:
        check(end)
    # Whiteouts hide lower layers' entries; apply them before this layer's own entries.
    for member, _ in members:
        name = PurePosixPath(member.name)
        if name.name == OPAQUE:
            directory = resolve_in_root(root, name.parent, follow_last=True)
            if directory.is_dir():
                for child in directory.iterdir():
                    _remove(child)
        elif name.name.startswith(WHITEOUT):
            target = resolve_in_root(root, name.parent / name.name[len(WHITEOUT) :])
            if target.exists() or target.is_symlink():
                _remove(target)
    for member, data in members:
        name = PurePosixPath(member.name)
        if name.name.startswith(WHITEOUT) or ".." in name.parts:
            continue
        target = resolve_in_root(root, name)
        mode = member.mode & 0o0777
        if member.isdir():
            if target.exists() and not target.is_dir():
                _remove(target)
            target.mkdir(parents=True, exist_ok=True)
            target.chmod(mode | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        if member.issym():
            os.symlink(member.linkname, target)
        elif member.islnk():
            source = resolve_in_root(root, PurePosixPath(member.linkname), follow_last=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        elif data is not None:  # a regular file
            os.replace(data, target)
            target.chmod(mode | stat.S_IRUSR | stat.S_IWUSR)
        else:
            skipped += 1  # device nodes and fifos cannot be created without root
    scratch.rmdir()  # each file was moved into place
    return skipped


AGAIN = (
    "copy the file again, or export the image again where Docker has it: "
    "r2s env export NAME --out FILE.tar"
)
NOT_SAVED = "docker save IMAGE -o FILE.tar (or r2s env export NAME --out FILE.tar)"
ARCHIVE_ERRORS = (tarfile.TarError, EOFError, zlib.error, gzip.BadGzipFile, lzma.LZMAError)
"""What reading a damaged or cut archive raises (its data, not this machine's storage)."""
UNREAD_FIX = (
    "this image cannot be unpacked for bwrap as it is (Python's tarfile does not read that "
    "header): run it with the docker backend, or report it with the image's name"
)


def _short(digest: str) -> str:
    """``sha256:`` and the first 12 digits of ``digest`` (as Docker shows an image's id)."""
    return digest[: len("sha256:") + 12]


def _json_member(tar: tarfile.TarFile, name: str, tarball: Path) -> object:
    member = tar.getmember(name)
    if member.isfile() and member.size > RECORD_LIMIT:  # read whole: never one of any size
        raise SetupError(
            f"{tarball}: {name} has {member.size} bytes, more than an image's record ever has",
            fix=NOT_SAVED,
        )
    stream = tar.extractfile(member)
    if stream is None:
        raise SetupError(f"{tarball}: {name} is not a file", fix=AGAIN)
    data = stream.read()
    stored, read = BLOB.fullmatch(name), hashlib.sha256(data).hexdigest()
    if stored is not None and read != stored.group(1):
        raise SetupError(
            f"{tarball} is damaged: {name} does not match the digest it is stored under (it "
            f"reads as {_short('sha256:' + read)}); nothing was unpacked",
            fix=AGAIN,
        )
    try:
        return json.loads(data)
    except ValueError as error:  # JSON or UTF-8
        raise SetupError(f"{tarball}: {name} is damaged ({error})", fix=AGAIN) from None


class _Hashed:
    """A stream as it is read: the SHA-256 of what was read from it (``sha256``), how much
    (``size``), and where its last byte that is not zero ends (``end``). ``peek`` reads ahead,
    and ``drain`` reads what nobody read (what follows the end of a tar archive). Reading a
    decompressor (``decompressed``), a damaged stream is a ``tarfile.ReadError``."""

    def __init__(self, stream: IO[bytes], *, decompressed: bool = False) -> None:
        self._stream = stream
        self._decompressed = decompressed
        self._ahead = b""
        self.sha256 = hashlib.sha256()
        self.size = 0
        self.end = 0

    def _take(self, size: int) -> bytes:
        try:
            data = self._stream.read(size)
        except OSError as error:
            if not self._decompressed or error.errno is not None:  # this machine's storage
                raise
            raise tarfile.ReadError(f"invalid compressed data ({error})") from None
        if data:
            self.sha256.update(data)
            last = len(data) if data[-1] else len(data.rstrip(b"\0"))
            if last:
                self.end = self.size + last
            self.size += len(data)
        return data

    def peek(self, size: int) -> bytes:
        while len(self._ahead) < size and (data := self._take(size - len(self._ahead))):
            self._ahead += data
        return self._ahead[:size]

    def read(self, size: int = -1, /) -> bytes:
        if size < 0:
            data, self._ahead = self._ahead + self._take(-1), b""
            return data
        if self._ahead:
            data, self._ahead = self._ahead[:size], self._ahead[size:]
            return data
        return self._take(size)

    def drain(self) -> None:
        while self.read(CHUNK):
            pass


def _contents(stored: _Hashed, opened: ExitStack) -> _Hashed:
    """The uncompressed contents of the layer ``stored`` (a tar archive, compressed with gzip,
    bzip2 or xz as ``tarfile`` recognizes them, or not), hashed as they are read: an image
    lists the digests of its layers' contents. A decompressor is closed with ``opened``."""
    head = stored.peek(tarfile.BLOCKSIZE)
    stream = cast(Any, stored)
    if head.startswith(b"\x1f\x8b\x08"):
        reader = opened.enter_context(gzip.GzipFile(fileobj=stream, mode="rb"))
    elif head[:3] == b"BZh" and head[4:10] == b"1AY&SY":
        reader = opened.enter_context(bz2.BZ2File(stream, mode="rb"))
    elif head.startswith((b"\x5d\x00\x00\x80", b"\xfd7zXZ")):
        reader = opened.enter_context(lzma.LZMAFile(stream, mode="rb"))  # noqa: SIM115
    else:
        return stored
    return _Hashed(cast(IO[bytes], reader), decompressed=True)


def _check_layer(
    tarball: Path,
    layer: str,
    place: str,
    diff_id: str,
    stored: _Hashed,
    contents: _Hashed,
    end: int,
) -> None:
    """Raise unless the layer just read (``place``: which of the image's layers it is) is the
    one the image lists: its contents, read to their end, have the digest ``diff_id``; stored
    under a digest (``blobs/sha256/...``), it has that one; and no entry follows where reading
    its entries ended (``end``: a header that could not be read ends a tar archive's entries
    as quietly as its end does)."""
    contents.drain()
    stored.drain()
    read = "sha256:" + contents.sha256.hexdigest()
    if read != diff_id:
        raise SetupError(
            f"{tarball} is damaged: {place} ({layer}) does not match the digest its image "
            f"lists ({_short(diff_id)}; it reads as {_short(read)}); nothing was unpacked",
            fix=AGAIN,
        )
    blob = BLOB.fullmatch(layer)
    if blob is not None and stored.sha256.hexdigest() != blob.group(1):
        raise SetupError(
            f"{tarball} is damaged: {place} ({layer}) does not match the digest it is stored "
            f"under (it reads as {_short('sha256:' + stored.sha256.hexdigest())}); nothing was "
            "unpacked",
            fix=AGAIN,
        )
    if contents.end > end:
        raise SetupError(
            f"{tarball}: {place} ({layer}) has entries after byte {end} of its archive that "
            "cannot be read (an entry's header there is not one tarfile reads); nothing was "
            "unpacked",
            fix=UNREAD_FIX,
        )


@dataclass(frozen=True)
class _Saved:
    """An image in a tarball written by ``docker save``."""

    layers: list[str]
    config: dict[str, Any]
    names: tuple[str, ...]
    """The names it was saved under (``RepoTags``); none when it was saved by image id."""


def _read_saved(tar: tarfile.TarFile, tarball: Path) -> list[_Saved]:
    """The images saved by ``docker save`` in ``tar`` (usually one): their layers (in order),
    configurations and names, checked before anything is extracted: every file
    ``manifest.json`` names is there."""
    names = set(tar.getnames())
    if "manifest.json" not in names:
        raise SetupError(
            f"{tarball} is not an image saved by `docker save`: it has no manifest.json",
            fix=NOT_SAVED,
        )
    manifest = _json_member(tar, "manifest.json", tarball)
    entries = cast(list[object], manifest) if isinstance(manifest, list) else []
    if not entries or not all(isinstance(entry, dict) for entry in entries):
        raise SetupError(f"{tarball}: manifest.json names no image", fix=AGAIN)
    images: list[_Saved] = []
    for entry in entries:
        fields = cast(dict[str, Any], entry)
        config_name, layers = fields.get("Config"), fields.get("Layers")
        if not isinstance(config_name, str) or not isinstance(layers, list):
            raise SetupError(
                f"{tarball}: manifest.json does not name a configuration and layers", fix=AGAIN
            )
        named = [config_name, *cast(list[object], layers)]
        if not all(isinstance(name, str) for name in named):
            raise SetupError(
                f"{tarball}: manifest.json names a layer that is not a file name", fix=AGAIN
            )
        missing = [str(name) for name in named if name not in names]
        if missing:
            more = f" (and {len(missing) - 1} more)" if len(missing) > 1 else ""
            raise SetupError(
                f"{tarball} is incomplete: it has no {missing[0]}{more}, which its "
                "manifest.json names (a copy or a download that stopped early?)",
                fix=AGAIN,
            )
        config = _json_member(tar, config_name, tarball)
        if not isinstance(config, dict):
            raise SetupError(
                f"{tarball}: the configuration {config_name} is not an object", fix=AGAIN
            )
        tags = fields.get("RepoTags")
        listed = cast(list[object], tags) if isinstance(tags, list) else []
        images.append(
            _Saved(
                [str(layer) for layer in cast(list[object], layers)],
                cast(dict[str, Any], config),
                tuple(tag for tag in listed if isinstance(tag, str) and _a_name(tag)),
            )
        )
    if len(images) == 1 and not images[0].names and "repositories" in names:
        images[0] = _Saved(images[0].layers, images[0].config, _repositories(tar))
    return images


def _a_name(name: str) -> bool:
    """Whether ``name``, of the names a tarball gives its image, names one: not an empty one,
    nor the ``<none>:<none>`` that some tools write for an image saved by id."""
    return bool(name.strip()) and not name.startswith("<none>")


def _repositories(tar: tarfile.TarFile) -> tuple[str, ...]:
    """The names an older ``docker save`` gave its one image in ``repositories`` (``{repository:
    {tag: layer}}``); none when that file is damaged (Docker does not read it either when there
    is a ``manifest.json``)."""
    try:
        member = tar.getmember("repositories")
        stream = tar.extractfile(member) if member.size <= RECORD_LIMIT else None
        listed: object = json.loads(stream.read()) if stream is not None else None
    except (KeyError, ValueError):
        return ()
    if not isinstance(listed, dict):
        return ()
    return tuple(
        f"{repository}:{tag}"
        for repository, tags in cast(dict[object, object], listed).items()
        if isinstance(repository, str) and isinstance(tags, dict)
        for tag in cast(dict[object, object], tags)
        if isinstance(tag, str) and _a_name(f"{repository}:{tag}")
    )


def _docker_name(reference: str) -> tuple[str, str]:
    """``reference`` as Docker reads it: its repository, with Docker Hub's host and
    ``library/`` filled in (``python`` is ``docker.io/library/python``), and its tag (``latest``
    when it names none), or ``@`` and its digest."""
    name, at, digest = reference.partition("@")
    repository, tag = name, "latest"
    if ":" in name.rsplit("/", 1)[-1]:
        repository, tag = name.rsplit(":", 1)
    parts = repository.split("/")
    if len(parts) == 1 or not (parts[0] == "localhost" or "." in parts[0] or ":" in parts[0]):
        parts = ["docker.io", *parts]
    if parts[0] == "index.docker.io":
        parts[0] = "docker.io"
    if parts[0] == "docker.io" and len(parts) == 2:
        parts.insert(1, "library")
    return "/".join(parts), f"@{digest}" if at else tag


def _environment_of(reference: str) -> tuple[str, str] | None:
    """The Lab environment ``reference`` is an image of, whatever registry and namespace it is
    under (``R2S_IMAGE_PREFIX`` where it was built, a mirror), and its version; ``None`` when
    the last part of its name is no environment's."""
    from real2simlab.envs import ENVIRONMENTS

    repository, version = _docker_name(reference)
    environment = repository.rsplit("/", 1)[-1]
    return (environment, version) if environment in ENVIRONMENTS else None


def _same_image(name: str, reference: str) -> bool:
    """Whether the image a tarball saved as ``name`` is ``reference``: the same name as Docker
    reads it, or the same environment and version under another registry or namespace. A
    digest names a registry's manifest, which a saved image does not hold: for a reference by
    digest, the repository (or the environment) must be the same."""
    saved, wanted = _docker_name(name), _docker_name(reference)
    ours, theirs = _environment_of(reference), _environment_of(name)
    if wanted[1].startswith("@"):
        same_environment = ours is not None and theirs is not None and ours[0] == theirs[0]
        return saved[0] == wanted[0] or same_environment
    return saved == wanted or (ours is not None and theirs == ours)


def _listed(names: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``; of more, the first three and how many more."""
    if len(names) > 3:
        return f"{', '.join(names[:3])} and {len(names) - 3} more"
    return f"{', '.join(names[:-1])} and {names[-1]}" if len(names) > 1 else names[0]


def _choose(images: list[_Saved], tarball: Path, reference: str, *, quiet: bool) -> _Saved:
    """The image in ``images`` that is ``reference``, by the names the tarball gives it
    (:func:`_same_image`). A tarball that names none (saved by image id) is taken for
    ``reference``, with a note on standard error unless ``quiet``.

    Raises:
        SetupError: the tarball names its images, and none of them is ``reference``.
    """
    import sys

    for image in images:
        if any(_same_image(name, reference) for name in image.names):
            return image
    held = [name for image in images for name in image.names]
    if held:
        raise _another_image(tarball, reference, held)
    if not quiet:
        print(
            f"note: {tarball} names no image (it was saved by image id), so it cannot be "
            f"checked to be {reference}; it is unpacked as {reference}",
            file=sys.stderr,
        )
    return images[0]


def _another_image(tarball: Path, reference: str, held: list[str]) -> SetupError:
    """The error for a tarball that holds ``held`` (the names it gives), not ``reference``:
    with what to unpack it as, and where the image asked for comes from."""
    import shlex

    from real2simlab.envs import ENVIRONMENTS, IMAGE_VERSION, registered_environment_for

    message = f"{tarball} holds {_listed(held)}, not {reference}"
    wanted = registered_environment_for(reference)
    if wanted is None and not any(mark in reference for mark in "/:@"):  # a misspelt name?
        message += f" ({reference} is not a registered environment: {', '.join(ENVIRONMENTS)})"
    names = [
        environment[0] if environment is not None and environment[1] == IMAGE_VERSION else name
        for name, environment in ((name, _environment_of(name)) for name in held)
    ]
    suggestion = next((name for name in names if name in ENVIRONMENTS), names[0])
    source = (
        f"r2s env export {wanted} --out {wanted}.tar"
        if wanted is not None
        else f"docker save {reference} -o {tarball_name(reference)}"
    )
    return SetupError(
        message,
        fix=f"unpack it as what it is: r2s env unpack {suggestion} --from-tar "
        f"{shlex.quote(str(tarball))}; for {wanted or reference}, give a tarball of that image "
        f"(`{source}` where Docker is)",
    )


def _object(value: object) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def unpack_tarball(tarball: Path, reference: str, *, force: bool = False) -> Rootfs:
    """Unpack an image saved with ``docker save`` and register it as ``reference``: the image
    the tarball holds under that name (see the module's description). An image unpacked before
    is reused, unless ``force`` (extract it again: repairs a damaged one)."""
    return _unpack(tarball, reference, force=force, quiet=False)


def _unpack(tarball: Path, reference: str, *, force: bool, quiet: bool) -> Rootfs:
    tarball = Path(tarball)
    if not tarball.is_file():
        raise SetupError(
            f"image tarball not found: {tarball}",
            fix="docker save IMAGE -o FILE.tar on a machine "
            "with Docker, or r2s env unpack --from-url",
        )
    try:
        readable = tarfile.is_tarfile(tarball)
    except ARCHIVE_ERRORS:
        readable = False
    if not readable:
        raise SetupError(
            f"{tarball} is not an image saved by `docker save` (not a tar archive)", fix=NOT_SAVED
        )
    try:
        with tarfile.open(tarball) as tar:
            name = _unpack_saved(tar, tarball, reference, force=force, quiet=quiet)
    except ARCHIVE_ERRORS as error:  # cut or damaged data, found while reading it
        raise SetupError(f"{tarball} is damaged or incomplete ({error})", fix=AGAIN) from None
    with _locked():
        index = _index()
        index[reference] = name
        _write_index(index)
    found = find_rootfs(reference)
    if found is None:
        raise SetupError(
            f"{rootfs_dir() / name} was unpacked but is not there any more (removed by another "
            "command?)",
            fix="run the same command again: it extracts the image again",
        )
    return found


def _unpack_saved(
    tar: tarfile.TarFile, tarball: Path, reference: str, *, force: bool, quiet: bool
) -> str:
    """Extract the image in ``tar`` that is ``reference`` into the cache (unless it is there,
    whole); its directory's name."""
    from real2simlab._fs import force_remove, held_path, release, remove_abandoned

    image = _choose(_read_saved(tar, tarball), tarball, reference, quiet=quiet)
    layers, config = image.layers, image.config
    listed = _object(config.get("rootfs")).get("diff_ids")
    diff_ids = [str(d) for d in cast(list[object], listed)] if isinstance(listed, list) else []
    if len(diff_ids) != len(layers) or not all(DIFF_ID.fullmatch(d) for d in diff_ids):
        # checked before anything is extracted: without them, no layer can be checked
        raise SetupError(
            f"{tarball} is damaged: its image's configuration lists "
            + (
                f"{len(diff_ids)} layer digests (rootfs.diff_ids) for its {len(layers)} layers"
                if len(diff_ids) != len(layers)
                else "a layer digest (rootfs.diff_ids) that is not a SHA-256"
            )
            + ", so its layers cannot be checked; nothing was unpacked",
            fix=AGAIN,
        )
    last = diff_ids[-1] if diff_ids else "unknown"
    image_id = last.removeprefix("sha256:")[:24]
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    name = f"{reference.replace('/', '_').replace(':', '_')}-{config_digest[:12]}"
    target = rootfs_dir() / name
    if force and target.exists():
        with _replacing(target, reference):  # refuse at once, before extracting anything
            pass
    # Unpack next to the target, never in the system's temporary directory: a rename across
    # file systems fails (EXDEV), e.g. when /tmp is a tmpfs. What an unpack killed outright left
    # there (of any image) is removed first; the copy of an unpack still running is not.
    rootfs_dir().mkdir(parents=True, exist_ok=True)
    remove_abandoned(rootfs_dir(), STAGED)
    if force or not _unpacked(target):  # damaged (its record, or its tree): extracted again
        staging = held_path(lambda part: rootfs_dir() / f"{name}.{part}", mode=0o700)
        try:
            (staging / "rootfs").mkdir()
            skipped = 0
            for number, (layer, diff_id) in enumerate(zip(layers, diff_ids, strict=True), 1):
                stream = tar.extractfile(layer)
                if stream is None:
                    raise SetupError(f"{tarball}: layer {layer} is not a file", fix=AGAIN)
                stored = _Hashed(stream)
                try:
                    with ExitStack() as opened:
                        contents = _contents(stored, opened)
                        place = f"layer {number} of {len(layers)}"
                        check = functools.partial(
                            _check_layer, tarball, layer, place, diff_id, stored, contents
                        )
                        skipped += _apply_layer(
                            staging / "rootfs", cast(IO[bytes], contents), staging / "layer", check
                        )
                except ARCHIVE_ERRORS as error:
                    raise SetupError(
                        f"{tarball}: layer {layer} is damaged or incomplete ({error})",
                        fix=AGAIN,
                    ) from None
            for directory in (
                "tmp",
                "proc",
                "dev",
                *(root.lstrip("/") for root in MOUNT_ROOTS),
                "usr/local/nvidia/lib64",
            ):
                (staging / "rootfs" / directory).mkdir(parents=True, exist_ok=True)
            image_config = _object(config.get("config"))
            listed = image_config.get("Env")
            env = dict(
                item.split("=", 1)
                for item in cast(list[object], listed if isinstance(listed, list) else [])
                if isinstance(item, str) and "=" in item
            )
            workdir = image_config.get("WorkingDir")
            meta = {
                "reference": reference,
                "image_id": "sha256:" + config_digest,
                "layer": image_id,
                "env": env,
                "workdir": workdir if isinstance(workdir, str) and workdir else "/",
                "skipped_special_files": skipped,
            }
            (staging / "image.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            with _locked(), _replacing(target, reference):
                if target.exists() and (force or not _unpacked(target)):
                    # an earlier unpack, damaged or being replaced (--force)
                    problem = force_remove(target)  # its directories may be read-only
                    if problem is not None:
                        raise SetupError(problem, fix=f"remove {target} yourself, then unpack")
                if target.exists():  # another command unpacked the same image meanwhile
                    shutil.rmtree(staging, ignore_errors=True)
                else:
                    staging.rename(target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            release(staging)
    return name


def tarball_name(reference: str) -> str:
    """A file name for the tarball of ``reference``: ``r2s-python.tar``."""
    return reference.rsplit("/", 1)[-1].split(":", 1)[0].split("@", 1)[0] + ".tar"


def check_tarball_path(reference: str, out: Path) -> None:
    """Refuse an ``--out`` that cannot be the tarball of ``reference``: a directory."""
    if Path(out).is_dir():
        raise SetupError(
            f"--out {out} is a directory; it names the tarball to write",
            fix=f"give a file name, e.g. --out {Path(out) / tarball_name(reference)}",
        )


def export_image(reference: str, out: Path, docker: str = "docker") -> Path:
    """``docker save`` ``reference`` to the file ``out``. Raises ``SetupError`` with the fix
    for what failed: the path (a directory), the image (not on this machine) or the write."""
    out = Path(out)
    check_tarball_path(reference, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Stopped with the command (Ctrl-C, an interrupted experiment).
    result = run_captured([docker, "save", "--output", str(out), reference])
    if result.returncode == 0:
        return out
    from real2simlab.sandbox.process import PROBE_TIMEOUT_S

    try:
        present = (
            run_captured([docker, "image", "inspect", reference], timeout_s=PROBE_TIMEOUT_S)
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        present = False
    from real2simlab.envs import registered_environment_for

    name = registered_environment_for(reference)
    raise SetupError(
        f"docker save {reference} failed: {result.stderr.strip()}",
        fix=f"the image is here; check that {out.parent} is writable and has room for it, then "
        "run the same command again"
        if present
        else f"build or pull it first: r2s env build {name} (or r2s env pull {name})"
        if name
        else f"pull or build it first: docker pull {reference}",
    )


def unpack_image(reference: str, docker: str = "docker", *, force: bool = False) -> Rootfs:
    """Save ``reference`` from the local Docker daemon and unpack it (see ``unpack_tarball``)."""
    with tempfile.TemporaryDirectory(prefix="r2s-save-") as tmp:
        tarball = export_image(reference, Path(tmp) / "image.tar", docker)
        return _unpack(tarball, reference, force=force, quiet=True)  # saved by that name


def rootfs_asset_name(environment: str, image_version: str) -> str:
    """The release file of an environment's image (``docker save``, gzip-compressed)."""
    return f"{environment}-{image_version}.tar.gz"


TARBALL_ROUTE = (
    "on a machine with Docker, `r2s env export NAME --out FILE.tar`; copy the file here and "
    "`r2s env unpack NAME --from-tar FILE.tar`"
)
"""Bringing an image to a machine without a download: the fix every download failure offers."""
SHA256_DIGITS = 64


def unpack_again(reference: str) -> str:
    """How to extract the image ``reference`` again (a damaged unpacked image): with Docker
    here, or from a tarball where Docker is not (``r2s env unpack --force`` alone asks Docker)."""
    from real2simlab.envs import registered_environment_for

    name = registered_environment_for(reference) or reference
    return (
        f"unpack it again: r2s env unpack {name} --force (without Docker here, add --from-tar "
        f"with a tarball of `r2s env export {name}` written where Docker is, or --from-url)"
    )


def _network_fix() -> str:
    """The fix for a download that got no answer: this machine's network or proxy (named,
    never shown: a proxy's value may hold a password), which no option of the command changes."""
    proxies = sorted(
        name
        for name, value in os.environ.items()
        if name.lower() in {"https_proxy", "http_proxy", "all_proxy"} and value
    )
    setting = f" (this shell sets {', '.join(proxies)})" if proxies else ""
    return (
        f"check this machine's network connection and proxy settings{setting}, then run the "
        f"same command again; or {TARBALL_ROUTE}"
    )


def _is_sha256(text: str) -> bool:
    return len(text) == SHA256_DIGITS and all(c in "0123456789abcdef" for c in text)


NO_FILE = (404, 410)
"""HTTP answers that there is no file at a URL: the server answered, the network works."""
NO_FILE_FIX = (
    "the server has no file at that URL (a release's files may not be published yet): check "
    f"the URL, or give --from-url one that has the file (a mirror); or {TARBALL_ROUTE}"
)


def _url_problem(url: str) -> SetupError | None:
    """Why ``url`` is not a URL to download from (a path on this machine, which ``--from-tar``
    takes; a URL without a scheme ``urllib`` downloads, or without a host), or ``None``."""
    import shlex
    import urllib.parse

    try:
        parts = urllib.parse.urlsplit(url)
        _ = parts.port  # a port that is not a number
    except ValueError:  # e.g. a malformed IPv6 address
        return SetupError(
            f"--from-url takes an http(s) URL, and {url!r} is not a valid one",
            fix="give --from-url an http:// or https:// URL",
        )
    if parts.scheme == "file" or (parts.scheme in ("http", "https", "ftp") and parts.hostname):
        return None
    if Path(url).expanduser().is_file():
        fix = f"for a file on this machine, use --from-tar instead: --from-tar {shlex.quote(url)}"
    else:
        fix = (
            "give --from-url an http:// or https:// URL; for a file on this machine, use "
            "--from-tar FILE"
        )
    return SetupError(f"--from-url takes an http(s) URL, not {url!r}", fix=fix)


def download_verified(url: str, target: Path, *, expected_sha256: str | None = None) -> Path:
    """Download ``url`` to ``target`` and check its SHA-256: ``expected_sha256``, or the hash
    published next to it at ``<url>.sha256`` (``sha256sum`` format). Raises ``SetupError``
    with the fix for what failed: the URL (not one to download from), the network (no answer:
    a timeout, a proxy), the server (no file there), the published checksum (missing, or not
    one), the given one, the download's content."""
    import errno
    import urllib.error
    import urllib.request

    problem = _url_problem(url)
    if problem is not None:
        raise problem
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = (expected_sha256 or "").strip().lower()
    if expected and not _is_sha256(expected):
        raise SetupError(
            f"--sha256 {expected_sha256!r} is not a SHA-256 (64 hexadecimal digits)",
            fix="give the published checksum of the file",
        )
    if not expected:
        published = f"pass --sha256 with the published checksum, or {TARBALL_ROUTE}"
        try:
            with urllib.request.urlopen(url + ".sha256", timeout=60) as response:
                body: bytes = response.read()
        except urllib.error.HTTPError as error:  # an answer: no checksum there, or a refusal
            raise SetupError(
                f"cannot fetch the checksum {url}.sha256: {error}",
                fix=published if error.code in NO_FILE else _network_fix(),
            ) from error
        except OSError as error:  # no answer: the network or a proxy, which --sha256 cannot fix
            raise SetupError(
                f"cannot fetch the checksum {url}.sha256: {error}", fix=_network_fix()
            ) from error
        try:
            expected = body.decode().split()[0].strip().lower()
        except (IndexError, UnicodeDecodeError) as error:  # empty, or not text
            raise SetupError(f"{url}.sha256 holds no checksum", fix=published) from error
        if not _is_sha256(expected):
            raise SetupError(f"{url}.sha256 holds no SHA-256 ({expected[:80]!r})", fix=published)
    from real2simlab._fs import force_remove, held_path, release, remove_abandoned

    # What a download killed outright left beside the target is removed first (not the file of
    # a download still running); this one is held while it is written.
    remove_abandoned(target.parent, re.escape(f".{target.name}."), r"\.partial")
    partial = held_path(
        lambda part: target.with_name(f".{target.name}.{part}.partial"), directory=False
    )
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=600) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
    except urllib.error.HTTPError as error:  # an answer: no file there, or a refusal
        force_remove(partial)
        raise SetupError(
            f"cannot download {url}: {error}",
            fix=NO_FILE_FIX if error.code in NO_FILE else _network_fix(),
        ) from error
    except OSError as error:  # no answer (the network, a proxy), or this machine's disk
        force_remove(partial)
        if error.errno in (errno.ENOSPC, errno.EDQUOT, errno.EFBIG):  # this machine's disk
            raise
        raise SetupError(f"cannot download {url}: {error}", fix=_network_fix()) from error
    except BaseException:  # an answer cut short, Ctrl-C: nothing is left beside the target
        force_remove(partial)
        raise
    if digest.hexdigest() != expected:
        force_remove(partial)
        raise SetupError(
            f"{url}: SHA-256 {digest.hexdigest()} does not match {expected}",
            fix="check --sha256: it must be the published checksum of this file (if it is, the "
            "download was damaged: run the same command again)"
            if expected_sha256
            else "download it again; if it keeps failing, report it",
        )
    partial.replace(target)
    release(partial)
    return target
