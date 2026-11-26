"""The ``bwrap`` backend: bubblewrap over an unpacked image, for Linux without root or Docker.

The image's root file system (see ``real2simlab.envs.rootfs``) is mounted read-only as ``/``;
the mounts of the spec go on top; ``/tmp`` is a fresh tmpfs. All namespaces are unshared (so
there is no network unless the spec asks for an open one), the process dies with its parent,
and nothing of the host is visible except what is mounted.

The command line never holds secrets: the environment is passed to bubblewrap's own process,
which hands it to the sandboxed program unchanged.

GPUs: every ``/dev/nvidia*`` node is bound (CUDA selects devices with ``CUDA_VISIBLE_DEVICES``,
it must see all nodes), and the host's NVIDIA driver libraries, copied once into the cache, are
mounted at ``/usr/local/nvidia/lib64`` and put on ``LD_LIBRARY_PATH``.

Limits: time is enforced; CPUs by affinity, where ``taskset`` exists (Linux, util-linux;
otherwise the CPU limit is not applied, and records say so): sandboxes that run at the same time
on this machine, in one r2s command or in several of the same user's, get CPUs of their own
while there are enough (``CPUS``; the commands hold their CPUs with lock files in
``/tmp/r2s-cpus-<uid>``). Memory and process counts are
enforced where a systemd user session delegates the cgroup v2 ``memory`` and ``pids``
controllers (most desktops; not all clusters): the sandbox then runs in a transient scope
(``systemd-run --user --scope``) with ``MemoryMax`` and ``TasksMax``, and, where systemd takes
it (253 and later), ``OOMPolicy=continue``: a program that runs out of its memory is killed in
the sandbox (bwrap reports 128 + 9), and never stops the scope with bwrap in it. ``info()``
records which limits are enforced; ``R2S_BWRAP_CGROUP=0`` turns the scope off.
"""

from __future__ import annotations

import contextlib
import functools
import glob
import math
import os
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import warnings
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

from real2simlab.envs.rootfs import (
    MOUNT_ROOTS,
    Rootfs,
    find_rootfs,
    identity,
    in_use,
    mount_points_problem,
    unpack_again,
)
from real2simlab.errors import InfrastructureError, SetupError
from real2simlab.sandbox.base import (
    Backend,
    ExecResult,
    NetworkMode,
    ReadinessProblem,
    ResolvedImage,
    SandboxSpec,
    Session,
    finished,
    killed_from_outside,
)
from real2simlab.sandbox.process import PROBE_TIMEOUT_S, run_captured, run_logged
from real2simlab.vendor import cache_dir

APPARMOR_FIX = (
    "allow unprivileged user namespaces for bwrap (Ubuntu 24.04: add an AppArmor profile "
    "'profile bwrap /usr/bin/bwrap flags=(unconfined) { userns, }' in /etc/apparmor.d/bwrap "
    "and run sudo apparmor_parser -r /etc/apparmor.d/bwrap)"
)
PROC_FIX = (
    "use --backend docker, or run where bwrap may mount /proc: on the host rather than in this "
    "container, or in a container whose /proc is not masked (e.g. Docker's --security-opt "
    "systempaths=unconfined, Kubernetes' procMount: Unmasked)"
)
INSTALL_FIX = "sudo apt-get install bubblewrap (or ask your admin)"
NVIDIA_LIB_PATTERNS = ("libcuda.so*", "libnvidia-*.so*", "libnvcuvid.so*", "libnvoptix.so*")
DRIVER_LIB_DIR = "/usr/local/nvidia/lib64"
LDCONFIG_DIRS = ("/usr/sbin", "/sbin", "/usr/bin", "/bin")
"""Where ``ldconfig`` is when PATH leaves it out: Debian keeps it in ``/sbin``, which is not on
an ordinary user's PATH there."""


CGROUP_ENV = "R2S_BWRAP_CGROUP"
"""Set to ``0`` to run bwrap without a systemd user scope (no memory or process limits)."""
SCOPE_ENV = ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
"""What ``systemd-run --user`` needs to reach the user's service manager."""
SCOPE_PROBE = 'cat "/sys/fs/cgroup$(sed -n "s/^0:://p" /proc/self/cgroup)/cgroup.controllers"'
OOM_POLICY = ("-p", "OOMPolicy=continue")
"""Keeps a scope running when the kernel kills one of its processes for lack of memory. From
systemd 253 on, a scope stops then by default (``DefaultOOMPolicy=stop``), and stopping it kills
bwrap itself (SIGTERM): a program that ran out of its own memory would look like a sandbox
killed from outside. Older scopes do not take the setting, and always continue."""


CPU_WARNING = (
    "the bwrap backend cannot apply the CPU limit here: taskset (util-linux) is not installed, "
    "so the program may use every CPU of the host; run records list only the limits that applied"
)


SHARED_CPUS_WARNING = (
    "the bwrap sandboxes running at the same time on this machine (this command's and those of "
    "your other r2s commands) need more CPUs than this command may use ({count}), so some "
    "share CPUs and their programs get less CPU time than their limit; run fewer at once (a "
    "lower --concurrency, fewer commands) to give each its own"
)

UNSHARED_CPUS_WARNING = (
    "the bwrap backend cannot agree on CPUs with your other r2s commands on this machine "
    "({reason}): the sandboxes of this command get CPUs of their own, but those of another "
    "command running at the same time may be pinned to the same ones; run one command at a "
    "time, or give each command CPUs of its own (taskset -c LIST r2s ...)"
)


class CpuLimitWarning(UserWarning):
    """A CPU limit was asked for, but the bwrap backend cannot apply it on this host."""


def cpu_affinity() -> bool:
    """Whether the bwrap backend can apply CPU limits here: by affinity, with ``taskset``."""
    return sys.platform == "linux" and shutil.which("taskset") is not None


CPU_LOCK_DIR: Path | None = None
"""Where r2s commands hold the CPUs their bwrap sandboxes are pinned to, one lock file per CPU;
``None``: ``/tmp/r2s-cpus-<uid>``, the same for every command of a user. A local disk, not the
cache: locks do not work on every network file system."""


def cpu_lock_dir() -> Path:
    """The directory of the CPU locks (see ``CPU_LOCK_DIR``)."""
    if CPU_LOCK_DIR is not None:
        return CPU_LOCK_DIR
    return Path("/tmp") / f"r2s-cpus-{os.getuid()}"


class CpuPool:
    """The CPUs bwrap sandboxes are pinned to, shared out among the sandboxes running at the
    same time on this machine: this process's, and those of the user's other r2s commands (two
    terminals, a sweep, the reward workers of a trainer). A sandbox takes CPUs that none of
    them uses and gives them back when it ends; every sandbox was pinned to the first CPUs, so
    four twins shared one, ran into their time limit and were scored 0. Another command's CPUs
    are known by their lock files (``fcntl.flock`` in ``cpu_lock_dir()``): a lock is held while
    a sandbox of this process uses its CPU, and the system releases it when the process ends,
    ``kill -9`` included. When there are not enough CPUs, sandboxes share the ones this process
    uses least, as Docker's ``--cpus`` shares the machine, and a warning says so once."""

    def __init__(self, locks: Path | None = None) -> None:
        self._locks = locks
        self._lock = threading.Lock()
        self._use: dict[int, int] = {}
        """How many sandboxes of this process use each CPU."""
        self._held: dict[int, int] = {}
        """The open lock file of each CPU this process holds."""
        self._unshared: str | None = None
        """Why the CPUs cannot be agreed on with other commands, once that is known."""
        self._warned: set[str] = set()
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._forget)

    @contextmanager
    def take(self, cpus: float | None) -> Generator[list[int] | None]:
        """The CPUs for a sandbox limited to ``cpus``, while it runs; ``None`` when no CPU
        limit applies."""
        if cpus is None or not cpu_affinity() or sys.platform != "linux":
            yield None
            return
        with self._lock:
            available = sorted(os.sched_getaffinity(0))
            count = max(1, min(len(available), math.ceil(cpus)))
            chosen: list[int] = []
            for cpu in available:  # what no sandbox uses, here or in another command
                if len(chosen) < count and not self._use.get(cpu) and self._hold(cpu):
                    chosen.append(cpu)
            shared = len(chosen) < count
            if shared:  # not enough: those this process uses least, shared
                rest = [cpu for cpu in available if cpu not in chosen]
                rest.sort(key=lambda cpu: (self._use.get(cpu, 0), cpu))
                chosen = sorted(chosen + rest[: count - len(chosen)])
            for cpu in chosen:
                self._use[cpu] = self._use.get(cpu, 0) + 1
            due = [
                message
                for message in (
                    SHARED_CPUS_WARNING.format(count=len(available)) if shared else None,
                    UNSHARED_CPUS_WARNING.format(reason=self._unshared) if self._unshared else None,
                )
                if message is not None and message not in self._warned
            ]
            self._warned.update(due)
        for message in due:
            warnings.warn(message, CpuLimitWarning, stacklevel=3)
        try:
            yield chosen
        finally:
            with self._lock:
                for cpu in chosen:
                    self._use[cpu] = self._use.get(cpu, 1) - 1
                    if self._use[cpu] <= 0:
                        del self._use[cpu]
                        self._release(cpu)

    def _hold(self, cpu: int) -> bool:
        """Take ``cpu``'s lock for this process; ``False`` while another command holds it. When
        locks cannot be used here, ``True``: the CPUs are shared out within this process."""
        if self._unshared is not None:
            return True
        directory = self._locks or cpu_lock_dir()
        try:
            with contextlib.suppress(FileExistsError):
                os.mkdir(directory, 0o700)
            info = os.lstat(directory)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                raise OSError(f"{directory} is not a directory of this user's")
            if info.st_mode & 0o022:  # anyone could replace a lock file in it
                raise OSError(f"{directory} may be written by other users")
            flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
            descriptor = os.open(directory / f"cpu{cpu}", flags, 0o600)
        except OSError as error:
            self._unshared = str(error)
            return True
        import fcntl

        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:  # another command's sandbox runs on it
            os.close(descriptor)
            return False
        except OSError as error:  # e.g. a file system without locks
            os.close(descriptor)
            self._unshared = f"{directory}: {error}"
            return True
        self._held[cpu] = descriptor
        return True

    def _release(self, cpu: int) -> None:
        descriptor = self._held.pop(cpu, None)
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)  # the lock goes with it

    def _forget(self) -> None:
        """In a child made by ``fork``: no sandbox of the parent's runs there, and its copies
        of the parent's lock files would keep the parent's CPUs held after the parent ends."""
        self._lock = threading.Lock()
        for descriptor in self._held.values():
            with contextlib.suppress(OSError):
                os.close(descriptor)
        self._held.clear()
        self._use.clear()


CPUS = CpuPool()
"""The pool every bwrap backend of this process takes its sandboxes' CPUs from."""


@functools.cache
def user_scope_oom_continue() -> bool:
    """Whether a transient systemd user scope takes :data:`OOM_POLICY` here (systemd 253 and
    later; probed once, like :func:`user_scope_controllers`). Where it does not, a scope always
    continues when the kernel kills one of its processes."""
    systemd_run = shutil.which("systemd-run")
    if os.environ.get(CGROUP_ENV) == "0" or systemd_run is None:
        return False
    probe = [systemd_run, "--user", "--scope", "--quiet", "--collect", *OOM_POLICY, "true"]
    try:
        result = subprocess.run(
            probe, capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


@functools.cache
def user_scope_controllers() -> frozenset[str]:
    """The cgroup v2 controllers a transient systemd user scope gets: ``memory``, ``pids``,
    ``cpu``... Empty without a systemd user session or delegation (e.g. many clusters)."""
    systemd_run = shutil.which("systemd-run")
    if os.environ.get(CGROUP_ENV) == "0" or systemd_run is None:
        return frozenset()
    probe = [systemd_run, "--user", "--scope", "--quiet", "--collect", "-p", "TasksMax=64"]
    try:
        result = subprocess.run(
            [*probe, "sh", "-c", SCOPE_PROBE],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return frozenset()
    if result.returncode != 0:
        return frozenset()
    return frozenset(result.stdout.split())


BWRAP_ENV = "R2S_BWRAP"
"""The bwrap command to use instead of the one on ``PATH``."""


def bwrap_binary() -> str | None:
    return os.environ.get(BWRAP_ENV) or shutil.which("bwrap")


def _failure(command: list[str]) -> str | None:
    """What went wrong starting the probe sandbox ``command``, or ``None`` if it ran."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"a sandbox running `true` did not finish in {PROBE_TIMEOUT_S:g} s"
    if result.returncode == 0:
        return None
    return result.stderr.strip() or f"exit status {result.returncode}"


class BwrapBackend(Backend):
    name: ClassVar[str] = "bwrap"
    isolated: ClassVar[bool] = True
    supports_network_allowlist: ClassVar[bool] = False

    def __init__(
        self, bwrap: str | None = None, *, cgroup_controllers: frozenset[str] | None = None
    ) -> None:
        self.bwrap = bwrap or bwrap_binary() or "bwrap"
        self._rootfs: dict[str, Rootfs] = {}
        self._identities: dict[str, tuple[int, int, bytes] | None] = {}
        """The ``identity`` of each unpacked image when it was resolved (see ``_changed``)."""
        self._cgroup = cgroup_controllers

    @property
    def cgroup_controllers(self) -> frozenset[str]:
        """The cgroup controllers a transient user scope can limit here (probed once)."""
        if self._cgroup is None:
            self._cgroup = user_scope_controllers()
        return self._cgroup

    def limits_enforced(self) -> list[str]:
        controllers = self.cgroup_controllers
        return ["time"] + (["cpus"] if cpu_affinity() else []) + [
            name for name, controller in (("memory", "memory"), ("pids", "pids"))
            if controller in controllers
        ]  # fmt: skip

    def problems(self) -> list[ReadinessProblem]:
        if bwrap_binary() is None and not Path(self.bwrap).is_file():
            return [ReadinessProblem("bubblewrap is not installed", INSTALL_FIX)]
        # The sandbox every run gets, started for real: a check of only a part of it (without
        # the new /proc) passed in containers where every run then failed. When it fails, the
        # parts tell why: the namespaces alone, then everything but the new /proc.
        sandbox = self.probe_command()
        proc = sandbox.index("--proc")
        try:
            failed = _failure(sandbox)
            if failed is None:
                return []
            bare = [self.bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--unshare-all", "true"]
            namespaces = _failure(bare)
            without_proc = None if namespaces else _failure([*sandbox[:proc], *sandbox[proc + 2 :]])
        except OSError as error:  # R2S_BWRAP (or the bwrap given) cannot be run
            fix = (
                f"point {BWRAP_ENV} at the bwrap command, or unset it"
                if os.environ.get(BWRAP_ENV)
                else INSTALL_FIX
            )
            return [ReadinessProblem(f"bwrap cannot be run ({error})", fix)]
        if namespaces is not None:
            return [ReadinessProblem(f"bwrap cannot create namespaces: {namespaces}", APPARMOR_FIX)]
        if without_proc is None:
            return [
                ReadinessProblem(
                    f"bwrap cannot mount a new /proc here ({failed}), and every sandbox of this "
                    "backend has one: this machine does not allow it (e.g. an unprivileged "
                    "container that masks its own /proc), and unpacking an image again does not "
                    "change that",
                    PROC_FIX,
                )
            ]
        return [
            ReadinessProblem(
                f"bwrap cannot set up the sandbox every run gets: {failed}",
                f"see bwrap's message: `{shlex.join(sandbox)}` must work on this machine; or use "
                "--backend docker",
            )
        ]

    def probe_command(self) -> list[str]:
        """The sandbox every run gets (``command``: its namespaces, a new ``/proc``, ``/dev``,
        a ``/tmp`` tmpfs), around ``true``, with this host's ``/`` as its root: what
        ``problems`` starts. The systemd scope and the CPU affinity a run may add are checked
        on their own (``user_scope_controllers``, ``cpu_affinity``)."""
        plain = BwrapBackend(self.bwrap, cgroup_controllers=frozenset())
        return plain._command(SandboxSpec(image="", argv=("true",), workdir="/"), Path("/"))

    def setup_problem(self, log: str) -> ReadinessProblem | None:
        """Why bwrap could not set up the sandbox of a probe, as its own message in the probe's
        log says (``bwrap: ...``); ``None`` without one. ``bwrap: execvp ...`` is not one: the
        sandbox was set up, and its runtime has no such command."""
        message = next((line for line in log.splitlines() if line.startswith("bwrap: ")), None)
        if message is None or message.startswith("bwrap: execvp "):
            return None
        found = self.problems()  # what keeps bwrap from making any sandbox here, with its fix
        if found:
            return found[0]
        return ReadinessProblem(
            f"bwrap could not set up this run's sandbox ({message})",
            "the sandbox every run gets works here, so what this run adds to it (a GPU device, "
            "a mount) failed: see bwrap's message; or use --backend docker",
        )

    def resolve_image(self, reference: str) -> ResolvedImage:
        rootfs = self._find(reference)
        return ResolvedImage(
            reference=reference, digest=rootfs.image_id, content_id=rootfs.image_id
        )

    def _changed(
        self, reference: str, rootfs: Rootfs, now: tuple[int, int, bytes] | None
    ) -> str | None:
        """Why the unpacked image of ``reference``, whose ``identity`` is ``now``, is not the
        one this backend resolved, or ``None``. A command resolves an image once, and its run
        keys and records name that image's id (``resolve_image``): an image replaced, removed or
        given another record since (by hand, or by a forced unpack between two runs, when no
        sandbox holds it) would have its results filed, and reused, as the first image's."""
        if now is not None and now == self._identities.get(reference):
            return None
        return (
            f"the unpacked image {reference} ({rootfs.root}) changed after this command resolved "
            f"it as {rootfs.image_id} (it was replaced, removed or given another image.json), so "
            "nothing was run on it: this command files its results under that id; run the "
            "command again"
        )

    def _find(self, reference: str) -> Rootfs:
        if reference in self._rootfs:
            return self._rootfs[reference]
        rootfs = find_rootfs(reference)
        if rootfs is not None:
            problem = mount_points_problem(rootfs)
            if problem is not None:
                raise SetupError(problem, fix=unpack_again(reference))
        if rootfs is None:
            from real2simlab.envs import registered_environment_for

            name = registered_environment_for(reference) or reference
            raise SetupError(
                f"image {reference!r} is not unpacked for the bwrap backend",
                fix=f"r2s env unpack {name}   (without Docker here: run `r2s env export {name} "
                f"--out {name}.tar` where Docker exists, copy it, then `r2s env unpack {name} "
                f"--from-tar {name}.tar`)",
            )
        python = rootfs.rootfs / "usr" / "local" / "bin" / "python"
        if not (
            python.exists()
            or python.is_symlink()
            or (rootfs.rootfs / "usr" / "bin" / "python3").exists()
        ):
            raise SetupError(
                f"the unpacked image {reference!r} has no Python (damaged, or not a Python image)",
                fix=unpack_again(reference),
            )
        self._identities[reference] = identity(rootfs)  # first: a cached image has its identity
        self._rootfs[reference] = rootfs
        return rootfs

    def _scope(self, spec: SandboxSpec) -> list[str]:
        """The start of ``spec``'s command line that runs its sandbox in a transient systemd
        user scope limiting memory and processes (with :data:`OOM_POLICY` where systemd takes
        it), or nothing (no delegated controllers)."""
        controllers = self.cgroup_controllers
        properties: list[str] = []
        if spec.limits.memory_mb is not None and "memory" in controllers:
            properties += ["-p", f"MemoryMax={spec.limits.memory_mb}M", "-p", "MemorySwapMax=0"]
        if "pids" in controllers:
            properties += ["-p", f"TasksMax={spec.limits.pids}"]
        if properties and user_scope_oom_continue():  # its program's lack of memory is its own
            properties += OOM_POLICY
        systemd_run = shutil.which("systemd-run")
        if not properties or not systemd_run:
            return []
        return [systemd_run, "--user", "--scope", "--quiet", "--collect", *properties, "--"]

    def command(
        self,
        spec: SandboxSpec,
        rootfs: Rootfs,
        *,
        driver_libs: Path | None = None,
        cpus: Sequence[int] | None = None,
    ) -> list[str]:
        """The bubblewrap command line for ``spec``. Pure: nothing is executed. ``cpus`` are
        the CPUs to pin it to (``run`` and ``session`` take them from ``CPUS``); without them,
        the first ones this process may use."""
        return self._command(spec, rootfs.rootfs, driver_libs=driver_libs, cpus=cpus)

    def _command(
        self,
        spec: SandboxSpec,
        top: Path,
        *,
        driver_libs: Path | None = None,
        cpus: Sequence[int] | None = None,
    ) -> list[str]:
        """``command``, with ``top`` as the sandbox's ``/`` (``probe_command``: the host's)."""
        command = self._scope(spec)
        if spec.limits.cpus is not None and not cpu_affinity():
            warnings.warn(CPU_WARNING, CpuLimitWarning, stacklevel=2)
        # cpu_affinity() implies Linux; saying so lets type checkers on macOS see it too.
        elif spec.limits.cpus is not None and sys.platform == "linux":
            if cpus is None:
                available = sorted(os.sched_getaffinity(0))
                cpus = available[: max(1, min(len(available), math.ceil(spec.limits.cpus)))]
            command += ["taskset", "-c", ",".join(str(cpu) for cpu in cpus)]
        command += [self.bwrap, "--unshare-all", "--die-with-parent", "--new-session"]
        for name in SCOPE_ENV:  # systemd-run needs them; the sandboxed program does not
            command += ["--unsetenv", name]
        if spec.user is not None:  # the ids the program sees (mapped to the host user)
            uid, _, gid = spec.user.partition(":")
            command += ["--unshare-user", "--uid", uid, "--gid", gid or uid]
        if spec.network.mode is NetworkMode.OPEN:
            command.append("--share-net")
        elif spec.network.mode is NetworkMode.ALLOWLIST:
            raise SetupError(
                "the bwrap backend cannot restrict the network to an allowlist",
                fix='use the docker backend, or set network = "open" for the agent',
            )
        command += ["--ro-bind", str(top), "/", "--proc", "/proc", "--dev", "/dev"]
        targets = [m.target.rstrip("/") for m in spec.mounts] + [p.rstrip("/") for p in spec.tmpfs]
        for root in MOUNT_ROOTS:
            if root not in targets and any(t.startswith(root + "/") for t in targets):
                command += ["--tmpfs", root]  # mount points beneath it are created there
        for path in spec.tmpfs:
            command += ["--tmpfs", path]
        if spec.network.mode is NetworkMode.OPEN:
            for name in ("/etc/resolv.conf", "/etc/hosts"):
                if Path(name).exists():
                    command += ["--ro-bind", name, name]
        if spec.limits.gpus is not None:
            for device in sorted(glob.glob("/dev/nvidia*")):
                command += ["--dev-bind", device, device]
            if driver_libs is not None:
                command += ["--ro-bind", str(driver_libs), DRIVER_LIB_DIR]
        for mount in spec.mounts:
            option = "--ro-bind" if mount.read_only else "--bind"
            command += [option, str(Path(mount.source).resolve()), mount.target]
        command += ["--chdir", spec.workdir, "--", *spec.argv]
        return command

    def environment(self, spec: SandboxSpec, rootfs: Rootfs) -> dict[str, str]:
        """The sandboxed program's environment: the image's, then the spec's and secrets."""
        env = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            # For systemd-run (the user scope); bwrap removes them before the program starts.
            **{name: os.environ[name] for name in SCOPE_ENV if name in os.environ},
            **rootfs.env,
            "HOME": "/tmp",
            "TMPDIR": "/tmp",
            **spec.network.proxy_env(),
            **spec.env,
            **spec.secret_env,
        }
        if spec.limits.gpus is not None:
            env["LD_LIBRARY_PATH"] = ":".join(
                p for p in (DRIVER_LIB_DIR, env.get("LD_LIBRARY_PATH", "")) if p
            )
            if spec.limits.gpus != "all":
                env["CUDA_VISIBLE_DEVICES"] = spec.limits.gpus
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    def run(self, spec: SandboxSpec, log_path: Path) -> ExecResult:
        try:
            rootfs = self._find(spec.image)
        except SetupError as error:
            return ExecResult(None, False, 0.0, str(error))
        for mount in spec.mounts:
            if not Path(mount.source).exists():
                return ExecResult(None, False, 0.0, f"mount source does not exist: {mount.source}")
        driver_libs = driver_library_dir() if spec.limits.gpus is not None else None
        with CPUS.take(spec.limits.cpus) as cpus:  # CPUs of its own while there are enough
            command = self.command(spec, rootfs, driver_libs=driver_libs, cpus=cpus)
            with in_use(rootfs):  # `r2s env unpack --force` refuses to replace it meanwhile
                before = identity(rootfs)
                changed = self._changed(spec.image, rootfs, before)
                if changed is not None:  # since this command resolved it (between runs, by hand)
                    return ExecResult(None, False, 0.0, changed)
                code, timed_out, seconds = run_logged(
                    command, log_path, spec.limits.timeout_s, env=self.environment(spec, rootfs)
                )
                replaced = before is None or identity(rootfs) != before
        if replaced:  # by hand, or by a command that does not take the lock: not the program's
            return ExecResult(
                None,
                False,
                seconds,
                f"the unpacked image {spec.image} ({rootfs.root}) was replaced or removed while "
                "the program ran, so its result is not the program's; run it again",
            )
        # bwrap reports its program's death by a signal as 128 + N; bwrap itself dying of one
        # was done from outside. Not so in a systemd scope: its program running out of the
        # scope's memory can end bwrap too (from systemd 253 on, a scope that did not take
        # OOM_POLICY is stopped whole), and nothing here tells that from a kill from outside,
        # so the code stands.
        killed = None if self._scope(spec) else killed_from_outside(code, "bwrap")
        if killed is not None:
            return ExecResult(None, False, seconds, killed)
        return finished(code, timed_out, seconds, log_path, spec)

    @contextmanager
    def session(self, spec: SandboxSpec) -> Generator[Session]:
        """Each command starts its own sandbox, so the session checks first that one starts
        (``true`` runs in it): bwrap that cannot set it up here fails every command before it
        runs, which an agent would take for its commands' own failures."""
        if spec.secret_env:
            raise SetupError("bwrap sessions cannot hold secrets: each command would see them")
        rootfs = self._find(spec.image)
        changed = self._changed(spec.image, rootfs, identity(rootfs))
        if changed is not None:  # an agent's build is keyed by the image it resolved
            raise InfrastructureError(changed)
        driver_libs = driver_library_dir() if spec.limits.gpus is not None else None
        with CPUS.take(spec.limits.cpus) as cpus:  # each command runs on the session's CPUs
            command = self.command(
                replace(spec, argv=("bash", "-c")), rootfs, driver_libs=driver_libs, cpus=cpus
            )
            prefix, env = tuple(command), self.environment(spec, rootfs)
            session = Session(prefix=prefix, env=env, cwd=Path.cwd())
            with in_use(rootfs):
                problem = _start_problem(session)
                if problem is not None:
                    raise InfrastructureError(
                        f"the bwrap sandbox of this session ({spec.image}) does not start here: "
                        f"{problem}"
                    )
                yield session

    def info(self) -> dict[str, Any]:
        result = subprocess.run(
            [self.bwrap, "--version"],
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=PROBE_TIMEOUT_S,
        )
        return {
            "backend": self.name,
            "isolated": True,
            "bwrap_version": result.stdout.strip().split()[-1] if result.returncode == 0 else None,
            "limits_enforced": self.limits_enforced(),
        }

    def gpu_problems(self) -> list[ReadinessProblem]:
        if not glob.glob("/dev/nvidia[0-9]*"):
            return [ReadinessProblem("no NVIDIA device nodes in /dev", "install the NVIDIA driver")]
        try:
            libraries = driver_library_dir()
        except SetupError as error:
            return [ReadinessProblem(error.message, error.fix)]
        except (OSError, subprocess.SubprocessError) as error:  # listing or copying them failed
            return [
                ReadinessProblem(
                    f"the host's NVIDIA driver libraries could not be copied into the cache: "
                    f"{error}",
                    f"check that `ldconfig -p` lists them and that {cache_dir()} is writable, "
                    "with room for them (R2S_CACHE_DIR chooses another place)",
                )
            ]
        if not any(libraries.glob("libcuda.so*")):
            return [
                ReadinessProblem("libcuda was not found on the host", "install the NVIDIA driver")
            ]
        return []


def _start_problem(session: Session) -> str | None:
    """Why ``true`` does not run in ``session``'s sandbox, or ``None`` when it does."""
    try:
        result = run_captured(
            [*session.prefix, "true"], env=session.env, cwd=session.cwd, timeout_s=PROBE_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return f"`true` did not finish in {PROBE_TIMEOUT_S:g} s"
    except OSError as error:
        return f"bwrap could not be started: {error}"
    if result.returncode == 0:
        return None
    said = (result.stdout + result.stderr).strip()[-300:]
    return f"`true` exited {result.returncode}" + (f": {said}" if said else "")


def _driver_version() -> str:
    path = Path("/proc/driver/nvidia/version")
    if path.is_file():
        for word in path.read_text().split():
            if word.count(".") >= 1 and word.replace(".", "").isdigit():
                return word
    return "unknown"


def ldconfig() -> str:
    """The ``ldconfig`` command: on PATH, or in the system directories PATH may leave out."""
    found = shutil.which("ldconfig") or shutil.which(
        "ldconfig", path=os.pathsep.join(LDCONFIG_DIRS)
    )
    if found is None:
        raise SetupError(
            f"ldconfig was not found (not on PATH, nor in {', '.join(LDCONFIG_DIRS)}): it lists "
            "the host's NVIDIA driver libraries, which GPU runs on bwrap need",
            fix="install it (Debian and Ubuntu: sudo apt-get install libc-bin), or add the "
            "directory that holds it to PATH",
        )
    return found


def driver_library_dir() -> Path:
    """A cached copy of the host's NVIDIA user-space driver libraries, made once per version.

    One run (of any command on this cache) makes it while the others wait for it: the first GPU
    runs of an experiment start together, and each would otherwise clear the copy another one
    is making."""
    target = cache_dir() / "nvidia" / _driver_version()
    if (target / ".complete").is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive(target.parent / ".lock"):
        if not (target / ".complete").is_file():  # not made while this run waited
            _copy_driver_libraries(target)
    return target


@contextmanager
def _exclusive(path: Path) -> Generator[None]:
    """Hold the lock file ``path`` while the block runs (released with the process, too)."""
    import fcntl

    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _copy_driver_libraries(target: Path) -> None:
    result = subprocess.run(
        [ldconfig(), "-p"],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=PROBE_TIMEOUT_S,
    )
    sources: set[Path] = set()
    for line in result.stdout.splitlines():
        if "=>" not in line or ("x86-64" not in line and "x86_64" not in line):
            continue
        path = Path(line.split("=>")[-1].strip())
        if any(path.match(pattern) for pattern in NVIDIA_LIB_PATTERNS):
            sources.add(path)
    if not sources:
        raise SetupError(
            "no NVIDIA driver libraries found with ldconfig", fix="install the NVIDIA driver"
        )
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    for source in sorted(sources):
        real = source.resolve()
        shutil.copy2(real, partial / real.name)
        if source.name != real.name and not (partial / source.name).exists():
            os.symlink(real.name, partial / source.name)
    for library in list(partial.iterdir()):
        stem = library.name.split(".so")[0] + ".so"
        for alias in (stem, stem + ".1"):
            if not (partial / alias).exists():
                os.symlink(library.name, partial / alias)
    (partial / ".complete").write_text("ok\n")
    shutil.rmtree(target, ignore_errors=True)
    partial.rename(target)
