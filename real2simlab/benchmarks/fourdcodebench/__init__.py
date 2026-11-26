"""4DCodeBench: agents reconstruct a dynamic scene from one video as a 4D world.

Source: https://github.com/4DCodeBench/4DCodeBench (code, no license file) and the Hugging Face
datasets ``4DCodeBench/Dataset-Synthetic`` and ``4DCodeBench/Dataset-Real-World`` (no license
declared). Nothing of either is bundled: the code is fetched at a pinned commit and the data at
pinned revisions, on first use, into the user's cache; the task directories this adapter writes
are marked ``redistributable = false``.

Two variants per case:

* ``native`` (default, every case) — the official protocol: the agent delivers
  ``solution/build.sh`` and ``world/`` (4DCodeBench's convention, in its own agent image), and
  the official scorer scores the world the agent submitted (scorer plugin ``fourdcodebench``).
  The official scorer needs a GPU, the scorer image and model checkpoints, one of them gated.
* ``lab`` (synthetic cases) — the Lab's frozen re-run: ``build.sh`` is re-run offline and its
  world compared with the ground-truth world by Lab metrics that do not match objects by name
  (centroid-set Chamfer distances, aligned by a similarity in 3D because only video is given).

The functions ``download_synthetic_case``, ``download_real_case``, ``agent_context`` and
``case_capture`` are the reusable import steps (e.g. for benchmarks built on the Lab).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

from real2simlab._fs import (
    cache_problem,
    fsync_directory,
    fsync_tree,
    held_path,
    release,
    remove_abandoned,
    warn_damaged,
)
from real2simlab._hub import bounded
from real2simlab._project import install_command
from real2simlab.benchmarks.base import BenchmarkAdapter
from real2simlab.capture import Capture, SourceInfo, TextInfo, VideoStream, View, WorldFrame
from real2simlab.capture.provenance import Provenance, Sourced
from real2simlab.errors import SetupError, UnknownOptionError, UsageError
from real2simlab.media import probe_video
from real2simlab.task import (
    Delivery,
    DeliveryConvention,
    NativeScoring,
    OutputRequirements,
    Protocol,
    Resources,
    ScoringConfig,
    TaskBuilder,
    TaskConfig,
)
from real2simlab.task.model import SourceInfo as TaskSource
from real2simlab.twin.model import ProgramRules
from real2simlab.vendor import FOURDCODEBENCH, cache_dir, causes, is_bug

SYNTHETIC_REPO = "4DCodeBench/Dataset-Synthetic"
SYNTHETIC_REVISION = "6b634b2ef9e63ffa6b78639c16b6859717423a59"
REAL_REPO = "4DCodeBench/Dataset-Real-World"
REAL_REVISION = "2e6f267c94b9a61afd2c82852765758afe838f33"
AGENT_IMAGE = "4dcb/agent:dev"
"""Built from the official checkout:
``docker build -t 4dcb/agent:dev -f harness/agent/Dockerfile .``."""
DEFAULT_SYNTHETIC = ("B_30", "B_12", "G_12")
DEFAULT_REAL = (
    "phys101_01_solid_object_dropped",
    "wisa_80k_04_marble_rolling_along",
    "physics_iq_benchmark_07_ball_rolls_down",
)
LICENSE = "none declared (4DCodeBench code and data); fetched at run time, not redistributed"
VARIANTS = ("native", "lab")

WORKSPACE = "/workspace"
VIDEO = "input.mp4"
LAB_METRICS = ("centroid_set_chamfer", "centroid_set_chamfer_2d")


def _hub() -> Any:
    try:
        import huggingface_hub  # pyright: ignore[reportMissingImports]
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SetupError(
            "the 4DCodeBench adapter needs huggingface_hub, h5py and hdf5plugin",
            fix=install_command("fourdcodebench"),
        ) from error
    return huggingface_hub


def data_dir() -> Path:
    return cache_dir() / "data" / "4DCodeBench"


def download_synthetic_case(case: str) -> tuple[Path, Path]:
    """The reference video and the unpacked ground-truth world of a synthetic case."""
    target = data_dir() / "synthetic" / case
    video, world = target / "reference.mp4", target / "world"
    if _cached(target, [video, world / "camera.json"]):
        return video, world
    hub = _hub()
    with (
        tempfile.TemporaryDirectory(prefix="r2s-4dcb-") as tmp,
        _entry(target) as partial,
        bounded(),
    ):
        downloaded = Path(
            hub.hf_hub_download(
                SYNTHETIC_REPO,
                f"videos/{case}.mp4",
                repo_type="dataset",
                revision=SYNTHETIC_REVISION,
                local_dir=tmp,
            )
        )
        shutil.copyfile(downloaded, partial / video.name)
        packed = hub.hf_hub_download(
            SYNTHETIC_REPO,
            f"worlds/{case}.h5",
            repo_type="dataset",
            revision=SYNTHETIC_REVISION,
            local_dir=tmp,
        )
        _unpack(Path(packed), partial / world.name)
    return video, world


def _cached(target: Path, needed: list[Path]) -> bool:
    """Whether the cache entry ``target`` holds ``needed``, readable; a damaged entry is
    removed (with a warning) so that it is fetched again."""
    if not target.exists():
        return False
    problems = [(path, cache_problem(path)) for path in needed]
    damaged = [(path, problem) for path, problem in problems if problem is not None]
    if not damaged:
        return True
    warn_damaged(damaged[0][0], str(damaged[0][1]))
    shutil.rmtree(target, ignore_errors=True)
    return False


@contextmanager
def _entry(target: Path) -> Generator[Path]:
    """A directory to fill in place of the cache entry ``target``; it becomes ``target`` (its
    files flushed to disk first) only if the block completes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    # what a command killed outright left is removed (not the copy of a command still running)
    remove_abandoned(target.parent, re.escape(f".{target.name}."))
    partial = held_path(lambda part: target.parent / f".{target.name}.{part}", mode=0o700)
    try:
        yield partial
        fsync_tree(partial)
        shutil.rmtree(target, ignore_errors=True)
        partial.rename(target)
        fsync_directory(target.parent)
    finally:
        shutil.rmtree(partial, ignore_errors=True)
        release(partial)


def _unpack(packed: Path, world: Path) -> None:
    """Unpack with the official ``scripts/download_data.py:unpack`` of the pinned checkout."""
    checkout = FOURDCODEBENCH.ensure()
    code = (
        "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
        "from download_data import unpack; unpack(Path(sys.argv[2]), Path(sys.argv[3]))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(checkout / "scripts"), str(packed), str(world)],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=3600,
    )
    if result.returncode != 0:
        missing = "No module named" in result.stderr  # the official unpacker's dependencies
        raise SetupError(
            f"cannot unpack {packed.name}: {result.stderr.strip()[-800:]}",
            fix="pip install h5py hdf5plugin"
            if missing
            else "the downloaded file may be cut or damaged: run the same command again (it "
            "downloads it again); if it fails the same way, report it with this output",
        )


def download_real_case(case: str) -> tuple[Path, Path]:
    """The reference video and the dynamic-object mask annotation of a real-world case."""
    target = data_dir() / "real" / case
    video, mask = target / "reference.mp4", target / "dynamic_mask.npz"
    if _cached(target, [video, mask]):
        return video, mask
    hub = _hub()
    with (
        tempfile.TemporaryDirectory(prefix="r2s-4dcb-") as tmp,
        _entry(target) as partial,
        bounded(),
    ):
        try:
            source = hub.hf_hub_download(
                REAL_REPO,
                f"videos/{case}.mp4",
                repo_type="dataset",
                revision=REAL_REVISION,
                local_dir=tmp,
            )
        except Exception as error:
            if not _not_in_dataset(error):
                raise  # the network, or a refusal: said as what it is, with its own fix
            raise SetupError(
                f"the video of {case} is not redistributed by the dataset; it must be rebuilt from "
                "its public source with the dataset's scripts/prepare_videos.py",
                fix="choose a case whose video is in the dataset (see its videos/ folder)",
            ) from error
        shutil.copyfile(source, partial / video.name)
        annotation = hub.hf_hub_download(
            REAL_REPO,
            f"annotations/{case}/dynamic_mask.npz",
            repo_type="dataset",
            revision=REAL_REVISION,
            local_dir=tmp,
        )
        shutil.copyfile(annotation, partial / mask.name)
    return video, mask


def _not_in_dataset(error: BaseException) -> bool:
    """Whether a download failed because the dataset has no such file (huggingface_hub's
    ``EntryNotFoundError``, HTTP 404), not because of the network or of this machine; never
    for a bug raised while such a failure was being handled."""
    if is_bug(error):
        return False
    for cause in causes(error):
        kinds = {kind.__name__ for kind in type(cause).__mro__}
        if "EntryNotFoundError" in kinds and "LocalEntryNotFoundError" not in kinds:
            return True
    return False


def agent_context(target: Path) -> Path:
    """The official agent context (TASK.md, world-format.md, render-settings.md) with its
    placeholders filled for the Lab's workspace; written to ``target``."""
    checkout = FOURDCODEBENCH.ensure()
    shutil.copytree(checkout / "harness" / "agent_context", target)
    paths = {
        "paths.task": f"{WORKSPACE}/visible/docs",
        "paths.video": f"{WORKSPACE}/visible/{VIDEO}",
        "paths.world": f"{WORKSPACE}/world",
        "paths.solution": f"{WORKSPACE}/solution",
    }
    for document in target.glob("*.md"):
        text = re.sub(
            r"{{([a-z_.]+)}}", lambda m: paths[m.group(1)], document.read_text(encoding="utf-8")
        )
        document.write_text(text, encoding="utf-8")
    return target


def case_capture(case: str, kind: str, video: Path) -> Capture:
    """A video-only capture of a case (the video is stored as ``input.mp4``)."""
    info = probe_video(video)
    return Capture(
        capture_id=f"fourdcodebench/{kind}/{case}",
        source=SourceInfo(
            dataset=SYNTHETIC_REPO if kind == "synthetic" else REAL_REPO,
            version=SYNTHETIC_REVISION[:12] if kind == "synthetic" else REAL_REVISION[:12],
            license=LICENSE,
        ),
        world=WorldFrame(metric_scale=Sourced(value=False, provenance=Provenance.DEFAULT)),
        views={
            "cam": View(
                rgb=VideoStream(
                    path=VIDEO,
                    fps=info.fps,
                    num_frames=info.num_frames,
                    resolution=(info.width, info.height),
                )
            )
        },
        text=TextInfo(task="Reconstruct the dynamic scene as a 4D world (see docs/TASK.md)."),
    )


def task_config(case: str, kind: str, variant: str, video: Path) -> TaskConfig:
    info = probe_video(video)
    native = variant == "native"
    protocol_name = "official protocol" if native else "Lab frozen re-run"
    scoring = (
        ScoringConfig(
            protocol=Protocol.NATIVE,
            native=NativeScoring(scorer="fourdcodebench", options={"kind": kind, "case": case}),
        )
        if native
        else ScoringConfig(
            protocol=Protocol.REPRODUCTION,
            metrics=LAB_METRICS,
            primary="centroid_set_chamfer_2d",
            metric_options={"centroid_set_chamfer": {"align": "similarity"}},
        )
    )
    return TaskConfig(
        id=f"fourdcodebench/{kind}/{case.lower()}/{variant}",
        version="1.0.0",
        title=f"4DCodeBench {kind} case {case} ({protocol_name})",
        description=(
            "Reconstruct the dynamic scene in the video as a 4D world (4DCodeBench). "
            + (
                "Scored by the official scorer on the world you submit."
                if native
                else "Your build.sh is re-run offline and its world compared with the ground truth."
            )
        ),
        source=TaskSource(
            dataset=SYNTHETIC_REPO if kind == "synthetic" else REAL_REPO,
            version=SYNTHETIC_REVISION if kind == "synthetic" else REAL_REVISION,
            url="https://github.com/4DCodeBench/4DCodeBench",
            citation=(
                "Shen, Kovačič, Kulits et al. 4DCodeBench: Benchmarking Agents on Inverse "
                "Graphics of Dynamic Scenes. arXiv:2610.03715"
            ),
        ),
        license=LICENSE,
        redistributable=False,
        tags=("fourdcodebench", kind, variant),
        visibility="video_only",
        delivery=Delivery(
            environment=AGENT_IMAGE,
            output_format="4d_world",
            convention=DeliveryConvention.BUILD_SH_WORLD,
            agent_time_budget_s=3600,
            run_timeout_s=1800,
            resources=Resources(gpu=True, cpus=8, memory_mb=32768),
            program=ProgramRules(max_size_mb=20),
            output=OutputRequirements(
                fps=info.fps, num_frames=info.num_frames, width=info.width, height=info.height
            ),
        ),
        scoring=scoring,
        metadata={
            "fourdcodebench": {"kind": kind, "case": case, "code_commit": FOURDCODEBENCH.commit}
        },
    )


def build_case(
    case: str, kind: str, variant: str, out_dir: Path, *, overwrite: bool = False
) -> Path:
    """Write one task directory and return it."""
    if kind == "synthetic":
        video, truth = download_synthetic_case(case)
    else:
        if variant == "lab":
            raise UsageError(
                "real cases have no ground-truth world: only the native variant exists"
            )
        video, truth = download_real_case(case)
    config = task_config(case, kind, variant, video)
    root = out_dir / kind / case.lower() / variant
    builder = TaskBuilder(root, config, overwrite=overwrite)
    with tempfile.TemporaryDirectory(prefix="r2s-4dcb-task-") as tmp:
        media = Path(tmp)
        shutil.copyfile(video, media / VIDEO)
        builder.capture(case_capture(case, kind, video), media)
        context = agent_context(Path(tmp) / "context")
        for document in sorted(context.glob("*.md")):
            builder.visible_doc(document.name, source=document)
        builder.instruction(_instruction(context, rerun=variant == "lab"))
    if kind == "synthetic":
        builder.reference(truth)
    else:
        builder.hidden_file("annotation/dynamic_mask.npz", source=truth)
    return builder.finish().root


RERUN_RULES = """
## How your program is re-run

Your `world/` is not scored as you leave it: `bash build.sh` is run again, offline, in a
fresh copy of this environment. There `{w}/solution/` is **read-only**; only `{w}/world/`
(empty) and `/tmp` are writable, and nothing else of this workspace exists. Generate any
intermediate file under `/tmp`, not next to `build.sh`, and run `bash build.sh` once yourself
that way before you finish.
"""


def _instruction(context: Path, *, rerun: bool = False) -> str:
    """The official TASK.md, plus where things are; for re-run variants, the re-run rules."""
    task = (context / "TASK.md").read_text(encoding="utf-8")
    text = task + (
        "\n\n## Where things are in this workspace\n\n"
        f"- The video: `{WORKSPACE}/visible/{VIDEO}` (also described in `visible/capture.json`).\n"
        f"- The format documents: `{WORKSPACE}/visible/docs/`.\n"
        f"- Deliver `{WORKSPACE}/solution/build.sh` (and what it runs) and `{WORKSPACE}/world/`.\n"
    )
    if rerun:
        text += RERUN_RULES.format(w=WORKSPACE)
    return text


class FourDCodeBenchBenchmark(BenchmarkAdapter):
    name: ClassVar[str] = "fourdcodebench"
    summary: ClassVar[str] = (
        "4DCodeBench: reconstruct a dynamic scene from one video as a 4D world; official protocol, "
        "plus a frozen re-run with Lab metrics on synthetic cases."
    )
    license: ClassVar[str] = LICENSE
    source: ClassVar[str] = (
        "https://github.com/4DCodeBench/4DCodeBench, Hugging Face datasets 4DCodeBench/*"
    )
    protocols: ClassVar[tuple[str, ...]] = ("native", "reproduction")
    status: ClassVar[str] = "partial"

    def task_names(self) -> list[str]:
        names = [
            f"fourdcodebench/synthetic/{c.lower()}/{v}" for c in DEFAULT_SYNTHETIC for v in VARIANTS
        ]
        return names + [f"fourdcodebench/real/{c.lower()}/native" for c in DEFAULT_REAL]

    def build(
        self,
        out_dir: Path,
        *,
        select: Sequence[str] | None = None,
        variants: Sequence[str] = VARIANTS,
        real: Sequence[str] | None = None,
        overwrite: bool = False,
        **options: Any,
    ) -> list[Path]:
        """Synthetic cases ``select`` (default: three small ones) and real cases ``real``."""
        if options:
            raise UnknownOptionError.from_names(
                f"benchmark {self.name!r}", options, self.options_taken()
            )
        bad = sorted(set(variants) - set(VARIANTS))
        if bad:
            raise UsageError(f"unknown variants {bad}; known: {list(VARIANTS)}")
        variants = tuple(dict.fromkeys(variants))  # a name given twice is built once
        paths: list[Path] = []
        for case in dict.fromkeys(select or DEFAULT_SYNTHETIC):
            for variant in variants:
                paths.append(
                    build_case(case, "synthetic", variant, Path(out_dir), overwrite=overwrite)
                )
        if "native" in variants:
            for case in dict.fromkeys(DEFAULT_REAL if real is None else real):
                paths.append(build_case(case, "real", "native", Path(out_dir), overwrite=overwrite))
        return paths


def write_manifest(entries: Sequence[dict[str, str]], path: Path) -> Path:
    path.write_text(json.dumps(list(entries), indent=2) + "\n", encoding="utf-8")
    return path
