"""``r2s demo``: the first report in one command.

Builds two synthetic tasks, evaluates the reference baselines and the scripted fake agent on
them, writes the report and prints where it is. The backend is chosen for the machine: Docker
when the runtime image is present or can be pulled, otherwise bubblewrap when it can start its
sandbox here and its root file system is unpacked, otherwise the unisolated local backend (with
a warning).
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

from real2simlab.cli.common import UNSCORED_EXIT, add_parser
from real2simlab.errors import SetupError

DEMO_TASKS = ("bouncing_ball", "pendulum")
DEMO_METHODS = (
    {"name": "oracle"},
    {"name": "static"},
    {"name": "constant_velocity"},
    {"name": "fake_agent", "label": "scripted-agent", "mode": "physical", "network": "none"},
    {"name": "fake_agent", "label": "broken-agent", "mode": "broken", "network": "none"},
)
FFMPEG_INSTALL = {
    "Linux": "Debian/Ubuntu: sudo apt-get install ffmpeg; Fedora: sudo dnf install ffmpeg",
    "Darwin": "brew install ffmpeg",
    "Windows": "use WSL 2 (see the platforms page), then: sudo apt-get install ffmpeg",
}


def register_demo(subparsers: Any) -> None:
    parser = add_parser(
        subparsers,
        "demo",
        "see a first report: baselines and a scripted agent on two synthetic tasks",
        "Build two synthetic tasks, evaluate the reference baselines and a scripted agent\n"
        "(no model, no key) on them, and write the report. The backend is chosen\n"
        "automatically: docker (image present or pullable), bwrap, then local.\n\n"
        "Example: r2s demo --open\n\n" + UNSCORED_EXIT,
    )
    parser.add_argument("--out", type=Path, default=Path("r2s-demo"), help="run directory")
    parser.add_argument(
        "--backend",
        choices=("auto", "docker", "bwrap", "local"),
        default="auto",
        help="where twins run (default: the best available)",
    )
    parser.add_argument("--open", action="store_true", help="open the report in a browser")
    parser.set_defaults(
        handler=_demo,
        on_interrupt="run the same command again to resume: finished steps are reused",
    )


def choose_backend(requested: str = "auto", log: Any = print) -> str:
    """The backend for the demo, with the reason, falling back as described above."""
    from real2simlab.envs import image_reference
    from real2simlab.envs.rootfs import find_rootfs
    from real2simlab.sandbox.base import get_backend

    if requested != "auto":
        return requested
    image = image_reference("r2s-python")
    docker = get_backend("docker")
    if not docker.problems():
        try:
            docker.resolve_image(image)  # pulls it if it is published and missing
            return "docker"
        except SetupError as error:
            log(f"docker: {str(error).splitlines()[0]}; trying bwrap")
    bwrap = get_backend("bwrap")
    found = bwrap.problems()  # its sandbox, started for real
    if found:
        log(f"bwrap: {found[0].message}; using local")
        return "local"
    try:
        unpacked = find_rootfs(image)
    except SetupError as error:  # unpacked, but damaged: `r2s doctor` says how to repair it
        log(f"bwrap: {str(error).splitlines()[0]}; using local")
        return "local"
    return "bwrap" if unpacked is not None else "local"


def _demo(args: argparse.Namespace) -> int:
    from real2simlab.cli.common import report_unscored
    from real2simlab.media import FFMPEG_ENV, FFPROBE_ENV, tool
    from real2simlab.orchestrate import parse_experiment, run_experiment
    from real2simlab.scoring.model import method_label

    install = FFMPEG_INSTALL.get(platform.system(), "install ffmpeg from https://ffmpeg.org")
    for name, variable in (("ffmpeg", FFMPEG_ENV), ("ffprobe", FFPROBE_ENV)):
        try:
            path = tool(name)  # what the Lab renders with: the variable's, or the one on PATH
        except SetupError:
            raise SetupError(
                "ffmpeg and ffprobe are needed to render the demo's videos", fix=install
            ) from None
        if shutil.which(path) is None:
            raise SetupError(
                f"{variable}={path} is not an executable file",
                fix=f"point {variable} at the {name} to use, or unset it to use the one on PATH",
            )
    backend = choose_backend(args.backend)
    if backend == "local" and shutil.which("ffmpeg") is None:
        # Programs on the local backend run with this machine's PATH, never with R2S_FFMPEG.
        raise SetupError(
            "the demo's programs run on the local backend, with this machine's PATH, and no "
            f"ffmpeg is on it ({FFMPEG_ENV} is only what the Lab itself runs)",
            fix=f"put ffmpeg on PATH ({install})",
        )
    if backend == "local" and args.backend != "local":
        print(
            "\n  note: neither Docker nor bubblewrap with an unpacked root file system can run\n"
            "  sandboxes here, so the demo runs on the local backend, WITHOUT isolation. Fine\n"
            "  for the demo (its programs are generated by Real2SimLab); `r2s doctor` says how\n"
            "  to get docker or bwrap, which anything else needs.\n",
            file=sys.stderr,
        )
    print(f"backend: {backend}", flush=True)
    config = parse_experiment(
        {
            "name": "demo",
            "tasks": {
                "benchmarks": [
                    {"name": "synthetic", "select": list(DEMO_TASKS), "tiers": ["calibrated"]}
                ]
            },
            "methods": [dict(method) for method in DEMO_METHODS],
            "run": {"backend": backend, "concurrency": 4, "check_reproducibility": True},
        },
        "r2s demo",
    )
    started = time.monotonic()
    results = run_experiment(config, args.out, log=lambda line: print(line, flush=True))
    report = (args.out / "report" / "index.html").resolve()
    print()
    labels = [method_label(m.method, m.privileged, m.network) for m in results.methods]
    width = max(map(len, labels), default=0)
    for method, label in zip(results.methods, labels, strict=True):
        score = method.reproduction_skill.mean
        shown = f"{score:6.3f}" if score is not None else "     -"
        print(f"  {label:<{width}}  reproduction skill {shown}")
    print(f"\nreport: {report}  ({time.monotonic() - started:.0f} s)")
    print("Next: docs/tutorials/evaluate-agent.md to evaluate a real coding agent.")
    if args.open:
        webbrowser.open(report.as_uri())
    return report_unscored(results)
