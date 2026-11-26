"""``r2s twin check | run``, ``r2s env build | list``, ``r2s doctor``, ``r2s spec schema``."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from real2simlab.cli.common import (
    add_backend,
    add_parser,
    backend_from,
    emit,
    positive_float,
    seed_int,
    table,
)


def register_twin(subparsers: Any) -> None:
    parser = add_parser(subparsers, "twin", "check and run twin packages (r2s-twin/1.0)")
    sub = parser.add_subparsers(dest="twin_command", metavar="ACTION", required=True)
    check = add_parser(
        sub,
        "check",
        "run a twin the way the evaluation will, and report problems",
        "Run a twin package (or a program directory) the way the evaluation will: frozen,\n"
        "offline, twice. The output must be valid and the two outputs identical. Exit status 1\n"
        "on any problem.",
    )
    check.add_argument(
        "twin", type=Path, help="twin package (twin.json + program/) or a program directory"
    )
    check.add_argument("--task", type=Path, help="apply this task's format, rules and limits")
    check.add_argument(
        "--out", type=Path, default=Path("runs/twin-check"), help="where runs are kept"
    )
    check.add_argument("--seed", type=seed_int, default=0)
    add_backend(check)
    check.set_defaults(handler=_check, on_interrupt="run the same command again")

    run = add_parser(sub, "run", "run a frozen twin once")
    run.add_argument("twin", type=Path)
    run.add_argument("--out", type=Path, required=True, help="run directory to create")
    run.add_argument("--seed", type=seed_int, default=0)
    run.add_argument("--timeout", type=positive_float, default=300)
    add_backend(run)
    run.set_defaults(handler=_run)


CHECK_MARKER = ".r2s-check"
CHECK_MARK = "r2s twin check\n"
"""Written first into a check's --out: the directory is a check's, which a check may clear."""
CHECK_ENTRIES = frozenset({"twin", "run", "rerun", "check.json", CHECK_MARKER})
"""What a check writes into its --out (and ``twin.partial``, the twin it was saving if it was
killed then: :func:`_check_dir`)."""


def _is_check_dir(out: Path) -> bool:
    """Whether ``out`` was written by a twin check, by its content (a directory of the user's
    that merely holds a ``run/`` is not): its marker, or the ``check.json`` of an earlier
    version as checks write it, the two runs ``run`` and ``rerun`` with the fields of
    ``CheckedRun`` (a list of checks of the user's own with ``ok`` fields is not)."""
    import dataclasses
    import json

    from real2simlab._fs import has_mark
    from real2simlab.runner.check import CheckedRun

    if has_mark(out / CHECK_MARKER, CHECK_MARK):
        return True
    try:
        checks = json.loads((out / "check.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    fields = {field.name for field in dataclasses.fields(CheckedRun)}
    return (
        isinstance(checks, list)
        and [item.get("name") if isinstance(item, dict) else None for item in checks]
        == ["run", "rerun"]
        and all(set(item) == fields and isinstance(item["ok"], bool) for item in checks)
    )


def _outside_inputs(out: Path, inputs: list[Path]) -> None:
    """Refuse an ``--out`` that is, holds or lies inside an input: the command would write
    into what it reads (copying a program into itself, or changing it while it runs)."""
    from real2simlab._fs import clashing_input
    from real2simlab.errors import SetupError

    clash = clashing_input(out, inputs)
    if clash is not None:
        raise SetupError(
            f"--out {out} is, holds or lies inside the input {clash}",
            fix="choose an --out outside the twin and the task (the default is runs/twin-check)",
        )


def _check_dir(out: Path) -> None:
    """Refuse an ``--out`` a check may not clear: not a directory, or one with other files."""
    from real2simlab.errors import SetupError
    from real2simlab.twin.io import STAGING_SUFFIX

    if not out.exists() and not out.is_symlink():
        return
    if out.is_symlink() or not out.is_dir():
        raise SetupError(f"--out {out} is not a directory", fix="choose another --out")
    names = {entry.name for entry in out.iterdir()}
    # what a check writes, and the twin it was saving if it was killed then (kill -9)
    entries = CHECK_ENTRIES | {"twin" + STAGING_SUFFIX}
    if names and not (names <= entries and _is_check_dir(out)):
        other = sorted(names - entries)
        held = (
            f"files a twin check did not write ({', '.join(other[:5])})"
            if other
            else f"no record of a twin check ({CHECK_MARKER} or the check.json a check writes)"
        )
        raise SetupError(
            f"--out {out} holds {held}; a check clears its --out directory, so it refuses this one",
            fix="choose a new or empty --out (the default is runs/twin-check)",
        )


def _clear(out: Path) -> None:
    """Empty ``out`` (:func:`_check_dir` allowed it) but for the check's marker, which is
    written first and kept while the rest is removed: a check stopped at any point (a full
    disk, Ctrl-C, kill -9) leaves a directory the same command takes as a check's."""
    from real2simlab._fs import force_remove, has_mark, write_mark
    from real2simlab.errors import SetupError

    if not has_mark(out / CHECK_MARKER, CHECK_MARK):  # new, or an earlier version's
        write_mark(out / CHECK_MARKER, CHECK_MARK)
    for child in list(out.iterdir()):
        if child.name == CHECK_MARKER:
            continue
        problem = force_remove(child)
        if problem:
            raise SetupError(problem, fix=f"remove {out} yourself or choose another --out")


def _check(args: argparse.Namespace) -> int:
    from real2simlab._fs import exclusive_use
    from real2simlab.errors import InputNotFoundError
    from real2simlab.runner.check import check_twin, format_checks
    from real2simlab.task import load_task

    if not args.twin.is_dir():
        what = "is not a directory" if args.twin.exists() else "does not exist"
        raise InputNotFoundError(
            f"{args.twin} {what}", fix="give a twin package (twin.json) or a program directory"
        )
    task = load_task(args.task) if args.task else None
    _outside_inputs(args.out, [args.twin, *([task.root] if task is not None else [])])
    _check_dir(args.out)
    backend = backend_from(args)
    args.out.mkdir(parents=True, exist_ok=True)
    # One check per directory at a time: a second one would clear the runs of the first.
    with exclusive_use(args.out, fix="wait until it finishes, or choose another --out"):
        _check_dir(args.out)
        _clear(args.out)
        checks = check_twin(args.twin, args.out, backend=backend, task=task, seed=args.seed)
    print(format_checks(checks))
    return 0 if all(c.ok for c in checks) else 1


def _run(args: argparse.Namespace) -> int:
    from real2simlab.runner import RunLimits, run_twin
    from real2simlab.twin import load_twin

    twin = load_twin(args.twin)
    _outside_inputs(args.out, [args.twin])
    result = run_twin(
        twin,
        args.out,
        backend=backend_from(args),
        seed=args.seed,
        limits=RunLimits(timeout_s=args.timeout),
    )
    print(
        f"{result.record.status.value} in {result.record.duration_s:.1f} s; "
        f"output: {result.output_dir}"
    )
    for problem in result.record.problems:
        print(f"  {problem}")
    return 0 if result.succeeded else 1


def register_env(subparsers: Any) -> None:
    parser = add_parser(subparsers, "env", "build and list runtime images")
    sub = parser.add_subparsers(dest="env_command", metavar="ACTION", required=True)
    listing = add_parser(sub, "list", "list the registered environments and whether they are built")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(handler=_env_list)
    build = add_parser(
        sub,
        "build",
        "build environment images with Docker (dependencies first)",
        "Build environment images with Docker; dependencies are built first.\n\n"
        "Example: r2s env build r2s-python r2s-agent-codex --pip-index-url https://mirror/simple",
    )
    build.add_argument("names", nargs="+", help="environments from `r2s env list`, or 'all'")
    build.add_argument("--pip-index-url", help="a PyPI mirror for the build")
    build.add_argument("--npm-registry", help="an npm mirror for the build")
    build.add_argument("--apt-mirror", help="a Debian mirror, e.g. http://mirrors.aliyun.com")
    build.add_argument("--no-cache", action="store_true")
    build.add_argument(
        "--network",
        help="network for the build's RUN steps, e.g. host (when downloads in the build are slow)",
    )
    build.add_argument(
        "--build-arg",
        action="append",
        default=[],
        metavar="KEY[=VALUE]",
        help="another Docker build argument, e.g. BLENDER_URL=https://mirror/blender.tar.xz; a "
        "name alone (e.g. HTTPS_PROXY) takes its value from this shell",
    )
    build.add_argument("--only", action="store_true", help="do not build dependencies")
    build.set_defaults(handler=_env_build)
    export = add_parser(
        sub,
        "export",
        "save an environment image to a tarball (to unpack it where there is no Docker)",
    )
    export.add_argument("name", help="an environment or image reference")
    export.add_argument("--out", type=Path, required=True, help="tarball to write")
    export.set_defaults(handler=_env_export)
    unpack = add_parser(
        sub,
        "unpack",
        "unpack an environment image into a root file system for the bwrap backend",
        "Unpack an image for the bwrap backend (Linux without root or Docker).\n\n"
        "With Docker on this machine:  r2s env unpack r2s-python\n"
        "Without Docker: on a machine with Docker run `r2s env export r2s-python --out "
        "r2s-python.tar`,\ncopy the file, then:  "
        "r2s env unpack r2s-python --from-tar r2s-python.tar",
    )
    unpack.add_argument("name", help="an environment or image reference")
    source = unpack.add_mutually_exclusive_group()
    source.add_argument("--from-tar", type=Path, help="a tarball written by `r2s env export`")
    source.add_argument(
        "--from-url",
        nargs="?",
        const="release",
        metavar="URL",
        help="download the image tarball (default: this version's GitHub release) and check "
        "its SHA-256",
    )
    unpack.add_argument(
        "--sha256",
        help="the expected SHA-256 of the --from-url download (default: <URL>.sha256)",
    )
    unpack.add_argument(
        "--force",
        action="store_true",
        help="extract it again even if it was unpacked before (to repair a damaged one)",
    )
    unpack.set_defaults(handler=_env_unpack)
    pull = add_parser(
        sub,
        "pull",
        "pull published environment images (instead of building them)",
        "Pull published images and tag them with their local names. Sources, in order: the\n"
        "local name itself when R2S_IMAGE_PREFIX points at a registry of your own, each prefix\n"
        "in R2S_IMAGE_MIRRORS (comma separated, e.g. a registry mirror closer to you), then\n"
        "the project's registry (its GitHub Container Registry).\n\n"
        "Example: R2S_IMAGE_MIRRORS=registry.example.com/real2simlab/ r2s env pull r2s-python",
    )
    pull.add_argument("names", nargs="+", help="environments from `r2s env list`, or 'all'")
    pull.set_defaults(handler=_env_pull)


def _env_pull(args: argparse.Namespace) -> int:
    from real2simlab.envs import ENVIRONMENTS, pull_environment
    from real2simlab.sandbox.docker import DockerBackend

    backend = DockerBackend()
    backend.ensure_ready()
    names = (
        [n for n, e in ENVIRONMENTS.items() if e.publishable]
        if args.names == ["all"]
        else args.names
    )
    for name in names:
        print(f"pulled {pull_environment(name, docker=backend.docker)}")
    return 0


def _env_export(args: argparse.Namespace) -> int:
    from real2simlab.envs import image_reference
    from real2simlab.envs.rootfs import check_tarball_path, export_image
    from real2simlab.sandbox.docker import DockerBackend

    check_tarball_path(image_reference(args.name), args.out)  # before Docker is asked anything
    backend = DockerBackend()
    backend.ensure_ready()
    print(export_image(image_reference(args.name), args.out, backend.docker))
    return 0


def _env_unpack(args: argparse.Namespace) -> int:
    from real2simlab.envs import image_reference
    from real2simlab.envs.rootfs import find_rootfs, unpack_image, unpack_tarball
    from real2simlab.errors import SetupError

    if args.sha256 is not None and not args.from_url:  # nothing here would check it
        import shlex

        tarball = shlex.quote(str(args.from_tar)) if args.from_tar else None
        source = (
            f"--from-tar {tarball} unpacks a file already on this machine"
            if tarball
            else "without --from-url the image is saved from Docker here"
        )
        whole = (
            f" (to compare the whole file with a checksum: sha256sum {tarball})" if tarball else ""
        )
        raise SetupError(
            f"--sha256 is the checksum of a download (--from-url), and {source}: it would not "
            "be checked",
            fix="leave out --sha256: every layer is checked against the digest its image lists "
            f"as it is unpacked{whole}; or give the URL the checksum is of: --from-url URL "
            "--sha256 HASH",
        )
    reference = image_reference(args.name)
    try:
        before = find_rootfs(reference)
    except SetupError:  # a damaged record: the unpack below extracts the image again
        before = None
    if args.from_url:
        from real2simlab import __version__
        from real2simlab._project import release_asset
        from real2simlab.envs import IMAGE_VERSION, registered_environment_for
        from real2simlab.envs.rootfs import download_verified, rootfs_asset_name
        from real2simlab.errors import SetupError
        from real2simlab.vendor import cache_dir

        url = args.from_url
        if url == "release":
            name = registered_environment_for(reference)
            if name is None:
                raise SetupError(
                    f"{args.name!r} is not a registered environment", fix="pass --from-url URL"
                )
            url = release_asset(__version__, rootfs_asset_name(name, IMAGE_VERSION))
        target = cache_dir() / "downloads" / url.rsplit("/", 1)[-1]
        print(f"downloading {url} ...", flush=True)
        download_verified(url, target, expected_sha256=args.sha256)
        rootfs = unpack_tarball(target, reference, force=args.force)
    elif args.from_tar:
        rootfs = unpack_tarball(args.from_tar, reference, force=args.force)
    else:
        from real2simlab.sandbox.docker import DockerBackend

        backend = DockerBackend()
        backend.ensure_ready()
        rootfs = unpack_image(reference, backend.docker, force=args.force)
    if not args.force and before is not None and before.image_id == rootfs.image_id:
        print(
            f"{reference} ({rootfs.image_id[:19]}) was already unpacked into {rootfs.rootfs}; "
            "it was not extracted again (--force extracts it again, e.g. to repair it)"
        )
    else:
        print(f"unpacked {reference} ({rootfs.image_id[:19]}) into {rootfs.rootfs}")
    return 0


def _env_list(args: argparse.Namespace) -> int:
    from real2simlab.envs import ENVIRONMENTS, image_status
    from real2simlab.sandbox.docker import docker_binary

    docker = docker_binary()
    rows, data = [], []
    for env in ENVIRONMENTS.values():
        image_id = image_status(env.name, docker) if docker else None
        rows.append([env.name, env.image, "yes" if image_id else "no", env.description])
        data.append({"name": env.name, "image": env.image, "built": bool(image_id), "id": image_id})
    emit(data, args.json, table(rows, ["environment", "image", "built", "content"]))
    return 0


def _build_args(items: list[str]) -> dict[str, str | None]:
    """``--build-arg KEY=VALUE`` items; a name alone (``HTTPS_PROXY``) is passed by name, and
    Docker takes its value from this shell (so it is never on a command line)."""
    from real2simlab.errors import UsageError

    parsed: dict[str, str | None] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not key.strip() or key != key.strip():
            raise UsageError(f"--build-arg {item!r}: give KEY=VALUE, or a variable's name alone")
        parsed[key] = value if separator else None
    return parsed


def _env_build(args: argparse.Namespace) -> int:
    from real2simlab.envs import (
        ENVIRONMENTS,
        accepted_terms,
        build,
        build_order,
        get_environment,
        terms_accepted,
    )
    from real2simlab.sandbox.docker import DockerBackend

    names = list(ENVIRONMENTS) if args.names == ["all"] else args.names
    if args.names == ["all"]:
        # "all" never accepts a license on the user's behalf: name such an image to build it.
        for name in [n for n in names if not terms_accepted(ENVIRONMENTS[n])]:
            terms = ENVIRONMENTS[name].accept_terms
            assert terms is not None
            print(f"skipping {name}: its license must be accepted first ({terms[1]})")
            names.remove(name)
    order = names if args.only else build_order(names)
    for name in order:  # before anything is built: says which terms to accept
        accepted_terms(get_environment(name))
    DockerBackend().ensure_ready()
    for name in order:
        print(f"building {name} ...", flush=True)
        image = build(
            name,
            pip_index_url=args.pip_index_url,
            npm_registry=args.npm_registry,
            apt_mirror=args.apt_mirror,
            no_cache=args.no_cache,
            network=args.network,
            build_args=_build_args(args.build_arg),
        )
        print(f"built {image}")
    return 0


def register_doctor(subparsers: Any) -> None:
    parser = add_parser(
        subparsers,
        "doctor",
        "check that this machine can run Real2SimLab, with fixes for what is missing",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="also check GPU support")
    parser.set_defaults(handler=_doctor)


def _doctor(args: argparse.Namespace) -> int:
    from real2simlab import doctor
    from real2simlab.envs import ENVIRONMENTS, build_hint, image_status
    from real2simlab.sandbox.base import backend_names, get_backend
    from real2simlab.sandbox.bwrap import BwrapBackend
    from real2simlab.sandbox.docker import docker_binary

    checks = doctor.system_checks()
    backends = {name: get_backend(name) for name in backend_names()}
    found = {name: backend.problems() for name, backend in backends.items()}  # asked once each
    # bwrap can start its sandbox here: whether it has the runtime twins run in.
    runtime = doctor.rootfs_check("r2s-python") if found.get("bwrap") == [] else None
    for name, backend in backends.items():
        problems = found[name]
        ready = "ready"
        if not problems and isinstance(backend, BwrapBackend):
            enforced = backend.limits_enforced()
            ready += "; limits enforced: " + ", ".join(enforced)
            if "cpus" not in enforced:
                ready += " (not CPU limits: install taskset, from util-linux)"
        fix = problems[0].fix if problems else None
        if name == "docker" and problems and runtime is not None and runtime.ok:
            fix = (fix + "; or " if fix else "") + "use --backend bwrap, which can run here"
        checks.append(
            doctor.Check(
                f"backend {name}",
                not problems,
                ready if not problems else problems[0].message,
                fix,
                required=name == "docker",
            )
        )
        if name == "bwrap" and runtime is not None:
            checks.append(runtime)
    docker = docker_binary()
    docker_problems = found["docker"] if "docker" in found else get_backend("docker").problems()
    docker_ready = docker is not None and not docker_problems
    if docker and docker_ready:
        for env in ENVIRONMENTS.values():
            built = image_status(env.name, docker)
            checks.append(
                doctor.Check(
                    f"image {env.name}",
                    built is not None,
                    env.image + (" built" if built else " not built"),
                    None if built else build_hint(env),
                    required=env.name == "r2s-python",
                )
            )
    checks += doctor.storage_checks(docker if docker_ready else None)
    checks += doctor.extras_checks()
    checks += doctor.agent_checks()
    if args.gpu:
        nvidia = shutil.which("nvidia-smi")
        driver, failure = None, "nvidia-smi not found"
        if nvidia:
            try:
                result = subprocess.run(
                    [nvidia, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                    stdin=subprocess.DEVNULL,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                failure = f"nvidia-smi failed: {error}"
            else:  # nvidia-smi prints its failures on stdout, with a non-zero status
                output = (result.stdout or result.stderr).strip()
                if result.returncode == 0 and output:
                    driver = doctor.gpu_summary(output)
                else:
                    failure = f"nvidia-smi failed (status {result.returncode}): {output}"
        checks.append(
            doctor.Check(
                "nvidia driver",
                driver is not None,
                driver or failure,
                None if driver else "install the NVIDIA driver, or fix it until nvidia-smi works",
                required=False,
            )
        )
        for name in backend_names():
            if name == "docker":  # Docker's GPUs are its daemon's: asked through Docker
                if docker is None or not docker_ready:
                    checks.append(doctor.Check("gpu via docker", False, "Docker is not ready "
                                               "(above)", None, required=False))  # fmt: skip
                else:
                    images = [env.image for env in ENVIRONMENTS.values()]
                    checks.append(doctor.docker_gpu_check(docker, images))
                continue
            if found.get(name):  # its sandbox cannot start here: neither can GPU runs
                checks.append(
                    doctor.Check(
                        f"gpu via {name}",
                        False,
                        f"{name} is not ready (above)",
                        None,
                        required=False,
                    )
                )
                continue
            if driver is None:  # the other backends use this host's driver and devices
                checks.append(
                    doctor.Check(
                        f"gpu via {name}",
                        False,
                        "no working NVIDIA driver on this host (above)",
                        None,
                        required=False,
                    )
                )
                continue
            problems = get_backend(name).gpu_problems()
            checks.append(
                doctor.Check(
                    f"gpu via {name}",
                    not problems,
                    "ready" if not problems else problems[0].message,
                    problems[0].fix if problems else None,
                    required=False,
                )
            )
    if args.json:
        print(json.dumps([check.to_json() for check in checks], indent=2))
    else:
        print(doctor.format_checks(checks))
    return 0 if all(check.ok for check in checks if check.required) else 1


def register_spec(subparsers: Any) -> None:
    parser = add_parser(subparsers, "spec", "export the JSON Schemas of every specification object")
    sub = parser.add_subparsers(dest="spec_command", metavar="ACTION", required=True)
    schema = add_parser(sub, "schema", "write or print JSON Schemas")
    schema.add_argument(
        "name",
        nargs="?",
        help="one object, e.g. r2s-task (default: every object; without --out, their names)",
    )
    schema.add_argument("--out", type=Path, help="directory to write <name>.schema.json files into")
    schema.set_defaults(handler=_schema)


def _schema(args: argparse.Namespace) -> int:
    from real2simlab.spec.schemas import export_schemas, schema_for

    if args.out:  # only the named object when a name is given; an unknown name writes nothing
        for path in export_schemas(args.out, [args.name] if args.name else None):
            print(path)
        return 0
    if args.name:
        print(json.dumps(schema_for(args.name), indent=2))
        return 0
    from real2simlab.spec.schemas import schema_names

    for name in schema_names():
        print(name)
    return 0
