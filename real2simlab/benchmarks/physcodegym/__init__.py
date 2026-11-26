"""PhysCodeGym: reconstruct a reference video as an executable, physics-driven Blender program.

Source: https://github.com/ShaoShuai0605/PhysCodeGym (MIT, commit pinned below) and the Hugging
Face dataset ``ShaoShuai0605/PhysCodeGym`` (revision pinned below). The reference videos keep
their source licenses (Pexels License, CC-BY-4.0, CC-BY-NC-4.0, ...), recorded per task; they
are downloaded when tasks are built and never redistributed (``redistributable = false``).

Shao, He, Fan, Guo et al. *PhysCodeGym: Benchmarking Coding Agents for Physically Verifiable 4D
Reconstruction.*

Delivery follows PhysCodeGym's contract (convention ``blender_scene``): ``solution/scene.py``
builds the scene from an empty Blender file and renders frames ``frame_0001.png`` ...
``frame_0240.png`` (640x360, 30 fps) into ``/app/outputs/frames``. Runtime image
``r2s-blender-4.0`` (Blender 4.0.2, as in PhysCodeGym).

Scoring here uses the official metrics that need no model API, computed by PhysCodeGym's own
code (scorer plugin ``physcodegym_pixels``): ``S_SSIM``, ``S_LPIPS`` (when ``lpips`` and
``torch`` are installed) and the motion-mask score ``S_mot``. Variants:

* ``native``: the official input, the frames the agent rendered in its session;
* ``rerun``: the frames of the Lab's frozen, offline re-run of ``scene.py`` — whether the
  program reproduces its own submission.

The object, event and physical-validity scores need LLM judges; they run through PhysCodeGym's
own Harbor tasks (see ``docs/benchmarks/physcodegym.md``).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from real2simlab._fs import fsync_file, remove_damaged
from real2simlab._hub import bounded
from real2simlab._project import install_command
from real2simlab.benchmarks.base import BenchmarkAdapter
from real2simlab.capture import Capture, SourceInfo, TextInfo, VideoStream, View, WorldFrame
from real2simlab.capture.provenance import Provenance, Sourced
from real2simlab.errors import SetupError, UnknownOptionError, UsageError
from real2simlab.media import write_downscaled
from real2simlab.task import (
    Delivery,
    DeliveryConvention,
    NativeScoring,
    OutputRequirements,
    Protocol,
    Resources,
    ScoringConfig,
    Task,
    TaskBuilder,
    TaskConfig,
)
from real2simlab.task.model import SourceInfo as TaskSource
from real2simlab.twin.model import ProgramRules
from real2simlab.vendor import VendoredRepo, cache_dir

CODE = VendoredRepo(
    name="PhysCodeGym",
    url="https://github.com/ShaoShuai0605/PhysCodeGym.git",
    commit="de1c8b75cddff0054f77b6322a979759e7a01f81",
    license="MIT (code); reference videos under their source licenses",
    override_env="R2S_PHYSCODEGYM",
)
DATASET = "ShaoShuai0605/PhysCodeGym"
REVISION = "d65acd3ccc4a4880034161dfd05281b165a33645"
DEFAULT_CASES = (
    "physics_iq_0001_ball_and_block_fall",
    "physics_iq_0019_ball_ramp",
    "physics_iq_0034_block_domino",
)
VARIANTS = ("native", "rerun")
WIDTH, HEIGHT, FPS, FRAMES = 640, 360, 30.0, 240
INPUT_WIDTH, INPUT_HEIGHT = 1280, 720
"""The reference a method is given: the 4K clip reduced to the largest frames PhysCodeGym's own
agent environment shows (its observation tools return frames at most 1280 pixels wide,
``MAX_FRAME_WIDTH`` in ``experiment/tools/observer_service.py``)."""
CITATION = (
    "Shao, He, Fan, Guo et al. PhysCodeGym: Benchmarking Coding Agents for Physically "
    "Verifiable 4D Reconstruction. https://github.com/ShaoShuai0605/PhysCodeGym"
)


def data_dir() -> Path:
    return cache_dir() / "data" / "PhysCodeGym"


def case_record(case: str) -> dict[str, Any]:
    """``benchmark/cases/<case>/case.json`` of the pinned code."""
    path = CODE.ensure() / "benchmark" / "cases" / case / "case.json"
    if not path.is_file():
        raise UsageError(f"no PhysCodeGym case {case!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def case_names(category: str | None = None) -> list[str]:
    root = CODE.ensure() / "benchmark" / "cases"
    names = sorted(p.parent.name for p in root.glob("*/case.json"))
    if category is None:
        return names
    return [n for n in names if case_record(n).get("category") == category]


def download_reference(case: str) -> Path:
    """The case's standardized 4K reference video, at the pinned revision."""
    target = data_dir() / "videos" / f"{case}.mp4"
    if not remove_damaged([target]):
        return target
    try:
        from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SetupError(
            "the PhysCodeGym adapter needs huggingface_hub",
            fix=install_command("physcodegym"),
        ) from error
    with bounded():
        path = hf_hub_download(
            DATASET, f"videos/{case}.mp4", repo_type="dataset", revision=REVISION,
            local_dir=data_dir(), force_download=True,
        )  # fmt: skip
    fsync_file(Path(path))
    return Path(path)


def statement(record: dict[str, Any]) -> str:
    """The task statement: PhysCodeGym's output contract and constraints, for this workspace."""
    return f"""\
# Task: build an executable physics world from a video

`{{workspace}}/visible/cam/rgb.mp4` (also described in `visible/capture.json`) shows:
{record.get("description", "a physical event")}

Reconstruct it as an executable Blender 4.0 world: a procedural Python script that builds the
geometry, materials, camera and lighting from an empty file, and reproduces the observed motion
**by simulating the physics that causes it**.

## Deliver

1. `{{workspace}}/solution/scene.py`, started as
   `blender --background --factory-startup --python scene.py`. It must
   - start from an empty scene and build everything procedurally;
   - save `/app/outputs/model.blend`;
   - render exactly 240 frames (frames 1 to 240, 30 fps, 640 x 360) as
     `/app/outputs/frames/frame_0001.png` ... `frame_0240.png`
     (set `scene.render.filepath = "/app/outputs/frames/frame_"`).

`/app/outputs` is writable in your workspace; run your scene there before you finish, so the
frames you rendered are the ones scored. Test with
`cd {{workspace}}/solution && python3 /opt/r2s/blender_scene.py` (renders, then encodes
`/app/outputs/render.mp4`).

## Constraints (PhysCodeGym's)

- Motion must come from simulation: physics solvers, forces, constraints and initial
  conditions. Keyframes, drivers, frame-change handlers or kinematic rigid bodies on the objects
  the scene is about count as replay.
- Bake the simulation before rendering (`bpy.ops.ptcache.bake_all(bake=True)`), or the frames
  show the initial pose.
- No hidden helpers (off-screen colliders, rails or props the video does not show).
- `scene.py` stays procedural and self-contained: one file under 256 KiB that imports no module
  of its own (no other `.py` next to it), reads no media and uses no network. When it is
  re-run frozen, it sees only its own directory.

## Scoring

Frame-aligned SSIM and the motion-mask score of your 240 frames against the reference video
(PhysCodeGym's own code), and LPIPS where available. No model judge is used here.
"""


def task_config(case: str, variant: str, record: dict[str, Any]) -> TaskConfig:
    source = record.get("source", {})
    rerun = variant == "rerun"
    return TaskConfig(
        id=f"physcodegym/{case.lower()}/{variant}",
        version="1.0.0",
        title=f"PhysCodeGym {case} ({'frozen re-run' if rerun else 'official input'})",
        description=record.get("description"),
        source=TaskSource(
            dataset=f"{DATASET} ({source.get('dataset', 'unknown source')})",
            version=REVISION,
            url=source.get("upstream_page_url")
            or source.get("source_url")
            or f"https://huggingface.co/datasets/{DATASET}",
            citation=CITATION,
            notes=f"original file {source.get('original_filename')}",
        ),
        license=str(source.get("license", "see the PhysCodeGym dataset card")),
        redistributable=False,
        tags=("physcodegym", str(record.get("category", "unknown")), variant),
        visibility="video_only",
        delivery=Delivery(
            environment="r2s-blender-4.0",
            output_format="video",
            convention=DeliveryConvention.BLENDER_SCENE,
            agent_time_budget_s=3600,
            run_timeout_s=1800,
            resources=Resources(gpu=True, cpus=8, memory_mb=16384),
            program=ProgramRules(
                max_size_mb=1.0, forbidden_suffixes=(".png", ".jpg", ".jpeg", ".npy", ".npz")
            ),
            output=OutputRequirements(fps=FPS, num_frames=FRAMES, width=WIDTH, height=HEIGHT),
        ),
        scoring=ScoringConfig(
            protocol=Protocol.NATIVE,
            native=NativeScoring(
                scorer="physcodegym_pixels", options={"case": case, "rerun": rerun}
            ),
        ),
        metadata={
            "physcodegym": {
                "case": case,
                "category": record.get("category"),
                "code_commit": CODE.commit,
            }
        },
    )


def build_case(case: str, variant: str, out_dir: Path, *, overwrite: bool = False) -> Path:
    record = case_record(case)
    video = download_reference(case)
    config = task_config(case, variant, record)
    root = Path(out_dir) / case.lower() / variant
    builder = TaskBuilder(root, config, overwrite=overwrite)
    with tempfile.TemporaryDirectory(prefix="r2s-pcg-") as tmp:
        media = Path(tmp)
        frames = write_downscaled(  # 4K: never held whole
            video, media / "cam" / "rgb.mp4", INPUT_WIDTH, INPUT_HEIGHT, FPS
        )
        capture = Capture(
            capture_id=f"physcodegym/{case}",
            source=SourceInfo(
                dataset=DATASET,
                version=REVISION[:12],
                license=config.license,
                url=config.source.url,
            ),
            world=WorldFrame(metric_scale=Sourced(value=False, provenance=Provenance.DEFAULT)),
            views={
                "cam": View(
                    rgb=VideoStream(
                        path="cam/rgb.mp4",
                        fps=FPS,
                        num_frames=frames,
                        resolution=(INPUT_WIDTH, INPUT_HEIGHT),
                    )
                )
            },
            text=TextInfo(
                task="Reconstruct the video as a physics-driven Blender world.",
                scene_description=record.get("description"),
            ),
        )
        builder.capture(capture, media)
    builder.hidden_file("reference/reference.mp4", source=video)
    builder.hidden_file("reference/case.json", text=json.dumps(record, indent=2) + "\n")
    builder.instruction(statement(record))
    return builder.finish().root


class PhysCodeGymBenchmark(BenchmarkAdapter):
    name: ClassVar[str] = "physcodegym"
    summary: ClassVar[str] = (
        "PhysCodeGym: reconstruct a video as a physics-driven Blender program; official "
        "pixel and motion metrics, on the agent's frames or on the frozen re-run's."
    )
    license: ClassVar[str] = "MIT code; per-video source licenses (downloaded, not redistributed)"
    source: ClassVar[str] = "https://github.com/ShaoShuai0605/PhysCodeGym"
    protocols: ClassVar[tuple[str, ...]] = ("native",)
    status: ClassVar[str] = "partial"

    def training_refusal(self, task: Task) -> str | None:
        """27 of the clips are Physics-IQ recordings, whose terms forbid training on them."""
        refusal = super().training_refusal(task)
        if refusal is None and "(Physics-IQ)" in task.config.source.dataset:
            from real2simlab.benchmarks.physicsiq import PhysicsIQBenchmark

            return (
                "the clip is a Physics-IQ recording, and Physics-IQ's terms forbid training on "
                f"its data: {PhysicsIQBenchmark.training_terms}"
            )
        return refusal

    def task_names(self) -> list[str]:
        return [f"physcodegym/{c}/{v}" for c in DEFAULT_CASES for v in VARIANTS]

    def build(
        self,
        out_dir: Path,
        *,
        select: Sequence[str] | None = None,
        variants: Sequence[str] = VARIANTS,
        overwrite: bool = False,
        **options: Any,
    ) -> list[Path]:
        if options:
            raise UnknownOptionError.from_names(
                f"benchmark {self.name!r}", options, self.options_taken()
            )
        bad = sorted(set(variants) - set(VARIANTS))
        if bad:
            raise UsageError(f"unknown variants {bad}; known: {list(VARIANTS)}")
        return [  # a name given twice is built once
            build_case(case, variant, Path(out_dir), overwrite=overwrite)
            for case in dict.fromkeys(select or DEFAULT_CASES)
            for variant in dict.fromkeys(variants)
        ]


def clear_cache() -> None:
    shutil.rmtree(data_dir(), ignore_errors=True)
