"""The ``local`` backend: runs commands directly on the host. For development only.

It does not isolate anything: no read-only program directory, no network cut, hidden
directories are reachable by path. Every use prints a prominent warning, and run records say
``isolated: false``. Sandbox paths in the working directory and in environment variables are
rewritten to the host paths they are mounted from; programs that hard-code sandbox paths
(e.g. 4DCodeBench's ``/workspace/world``) cannot run locally.

With ``run_as``, programs run as another user (a ``(uid, gid)``; this needs root) who owns
only the directories they may write. The Harbor verifier uses it so that a solution cannot
read the hidden reference held in the same container.

There is no sandbox whose end stops everything in it, so on Linux every command runs under a
supervisor (``reaper.py``): what the program starts, even in a session of its own, ends with
its run, and a program whose Lab is killed (``kill -9``) ends too.
"""

from __future__ import annotations

import errno
import os
import shutil
import sys
import warnings
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

from real2simlab.sandbox.base import (
    Backend,
    ExecResult,
    Mount,
    ReadinessProblem,
    ResolvedImage,
    SandboxSpec,
    Session,
    finished,
    rewrite_paths,
)
from real2simlab.sandbox.process import run_logged
from real2simlab.twin.sdk import sdk_path

WARNING = (
    "the 'local' backend runs programs WITHOUT ISOLATION (no read-only program, no network "
    "cut, hidden data reachable by path). Use it only to debug your own code; use 'docker' "
    "or 'bwrap' for evaluation."
)


def image_only(argv: Sequence[str]) -> str | None:
    """Why ``argv`` cannot run on this host: it names an absolute file that only a runtime
    image has (``/opt/r2s/blender_scene.py``), or ``None``."""
    for item in argv[:2]:
        if item.startswith("/") and " " not in item and not Path(item).exists():
            return (
                f"{item} exists only in the task's runtime image, so the local backend cannot "
                "run this; use --backend docker or bwrap"
            )
    return None


class LocalIsolationWarning(UserWarning):
    """Raised as a warning whenever the local backend runs something."""


REAPER = Path(__file__).with_name("reaper.py")


def supervised(argv: Sequence[str], env: Mapping[str, str], cwd: Path) -> list[str]:
    """``argv`` under the supervisor (``reaper.py``, Linux): nothing the program starts
    outlives its run, and it ends when the Lab does (``kill -9`` included). A command that
    cannot be found raises here, as starting it directly does. Elsewhere ``argv`` itself."""
    if sys.platform != "linux" or not sys.executable:
        return list(argv)
    program = argv[0]
    if os.sep in program:
        found = (cwd / program).exists()
    else:
        found = shutil.which(program, path=env.get("PATH")) is not None
    if not found:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), program)
    return [sys.executable, "-I", "-S", str(REAPER), str(os.getpid()), "--", *argv]


def _absolute(mounts: Sequence[Mount]) -> list[Mount]:
    return [replace(mount, source=Path(mount.source).absolute()) for mount in mounts]


class LocalBackend(Backend):
    name: ClassVar[str] = "local"
    isolated: ClassVar[bool] = False

    def __init__(
        self,
        python: str | None = None,
        quiet: bool = False,
        run_as: tuple[int, int] | None = None,
    ) -> None:
        self.python = python or sys.executable
        self.quiet = quiet
        self.run_as = run_as
        self._warned = False

    def problems(self) -> list[ReadinessProblem]:
        return []

    def resolve_image(self, reference: str) -> ResolvedImage:
        return ResolvedImage(reference=f"local-host ({reference} not used)", digest=None)

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        self._warn()
        # The program runs in its own directory: every path it is given is absolute, or a
        # relative --work or --out would point somewhere else from there (HOME, the output).
        mounts = _absolute(spec.mounts)
        log_path = Path(log_path).absolute()
        for mount in mounts:
            if not mount.read_only:
                Path(mount.source).mkdir(parents=True, exist_ok=True)
        env = {
            "PATH": os.pathsep.join([str(Path(self.python).parent), os.environ.get("PATH", "")]),
            "PYTHONPATH": str(sdk_path().parent),
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        tmp = Path(log_path).parent / "local_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        env.update({"HOME": str(tmp), "TMPDIR": str(tmp)})
        for path in spec.tmpfs:
            if not any(m.target == path for m in mounts):
                directory = tmp / path.strip("/").replace("/", "_")
                directory.mkdir(parents=True, exist_ok=True)
                mounts.append(Mount(directory, path, read_only=False))
        for key, value in {**spec.env, **spec.secret_env}.items():
            env[key] = rewrite_paths(value, mounts) if key not in spec.secret_env else value
        # Rewrite sandbox paths first; the host interpreter is substituted afterwards, so a host
        # path that happens to lie under a sandbox mount point (e.g. /tmp/venv) stays intact.
        argv = [rewrite_paths(item, mounts) for item in spec.argv]
        # Checked before the host interpreter is substituted (it is the host's, not the image's).
        missing = image_only(argv)
        if missing:  # e.g. a convention's launcher that only its runtime image has
            return ExecResult(None, False, 0.0, missing)
        argv = [self.python if item in {"python", "python3"} else item for item in argv]
        cwd = Path(rewrite_paths(spec.workdir, mounts))
        if not cwd.is_dir():
            return ExecResult(None, False, 0.0, f"working directory {cwd} does not exist")
        if self.run_as is not None:
            for directory in [tmp, *(Path(m.source) for m in mounts if not m.read_only)]:
                _give(directory, self.run_as)
        code, timed_out, seconds = run_logged(
            supervised(argv, env, cwd),
            log_path,
            spec.limits.timeout_s,
            env=env,
            cwd=cwd,
            user=self.run_as,
        )
        return finished(code, timed_out, seconds, log_path, spec)

    @contextmanager
    def session(self, spec: SandboxSpec) -> Generator[Session]:
        self._warn()
        mounts = _absolute(spec.mounts)
        cwd = Path(rewrite_paths(spec.workdir, mounts))
        env = {
            "PATH": os.pathsep.join([str(Path(self.python).parent), os.environ.get("PATH", "")]),
            "PYTHONPATH": str(sdk_path().parent),
            "LANG": "C.UTF-8",
            "HOME": str(cwd),
            **{key: rewrite_paths(value, mounts) for key, value in spec.env.items()},
        }
        yield Session(prefix=("bash", "-c"), env=env, cwd=cwd)

    def info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"backend": self.name, "isolated": False}
        if self.run_as is not None:
            info["run_as"] = list(self.run_as)
        return {**info, "python": sys.version.split()[0]}

    def gpu_problems(self) -> list[ReadinessProblem]:
        return []

    def _warn(self) -> None:
        if self._warned or self.quiet:
            return
        self._warned = True
        warnings.warn(WARNING, LocalIsolationWarning, stacklevel=3)  # r2s prints it plainly


def _give(directory: Path, owner: tuple[int, int]) -> None:
    """Make ``owner`` the owner of ``directory`` and everything in it."""
    os.chown(directory, *owner)
    for path in directory.rglob("*"):
        os.chown(path, *owner, follow_symlinks=False)
