"""PhySimCode: watch a physics simulation video, explain it and write the code that regenerates it.

Source: the PhySimCode evaluation set (Kundu et al., arXiv:2610.15722): 2,430 videos of 162
simulated phenomena, each with the program that rendered it, its parameters and a physics
explanation. Data: Hugging Face ``SourajaKundu123/PhySimCode`` (MIT, declared on the dataset
card), pinned by revision, downloaded into the cache when tasks are built. Evaluation code:
GitHub ``SourajaKundu/PhySimCode``, which declares no license: fetched from its authors at a
pinned commit when tasks are scored, never redistributed (the Lab's rule for unlicensed code).

A task shows one video (360x180, 10 fps) of a 2D (SciPy) simulation. The method delivers what
PhySimCode asks for, in the Lab's run contract: ``solution/cot.json``, its explanation with the
physical law and the parameter values, and ``solution/run.py``, a self-contained program that
simulates the phenomenon and renders the video (``$R2S_OUTPUT_DIR/render.mp4``, 30 fps). The
scorer plugin ``physimcode`` reports whether the Lab's frozen, offline re-run produced the
video (PhySimCode's runnability) and PhySimCode's own parameter recovery: its matching of the
delivered parameters to the true ones and the share within 20 %.

Supported: the 2D (SciPy) phenomena, run in ``r2s-physimcode`` (r2s-python and Matplotlib).
Not yet: the 3D (PyBullet + pyrender) phenomena and the LLM-judged and model-based metrics.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from real2simlab._fs import fsync_tree, remove_damaged
from real2simlab._hub import bounded
from real2simlab._project import install_command
from real2simlab.benchmarks.base import BenchmarkAdapter
from real2simlab.capture import Capture, SourceInfo, TextInfo, VideoStream, View, WorldFrame
from real2simlab.capture.provenance import Provenance, Sourced
from real2simlab.errors import SetupError, UnknownOptionError, UsageError
from real2simlab.media import probe_video
from real2simlab.task import (
    Delivery,
    NativeScoring,
    Protocol,
    Resources,
    ScoringConfig,
    TaskBuilder,
    TaskConfig,
)
from real2simlab.task.model import SourceInfo as TaskSource
from real2simlab.twin.model import ProgramRules
from real2simlab.vendor import VendoredRepo, cache_dir

CODE = VendoredRepo(
    name="PhySimCode",
    url="https://github.com/SourajaKundu/PhySimCode.git",
    commit="e876cd202228b27491ea9e391926246bf15c8749",
    license="none declared",
    override_env="R2S_PHYSIMCODE",
)
DATASET = "SourajaKundu123/PhySimCode"
REVISION = "1d03cd78fe25261af9c1f39d64c5fc438985b8eb"
LICENSE = "MIT"
CITATION = (
    "Kundu, Gupta, Julin, Zhao, Xie, Jeni. PhySimCode: A Benchmark and Evaluation Method for "
    "Physics Video to Code Generation. arXiv:2610.15722."
)
ENVIRONMENT = "r2s-physimcode"
ENGINES = ("scipy",)
"""Engines whose programs ``r2s-physimcode`` runs."""
DEFAULT_SAMPLES = ("scipy/simple_pendulum/014", "scipy/projectile_bounce/006",
                   "scipy/single_spring/005")  # fmt: skip
SAMPLE_FILES = ("video.mp4", "cot.json", "params.json", "simulation_code.py")
FPS_OUT = 30


def data_dir() -> Path:
    return cache_dir() / "data" / "PhySimCode" / REVISION[:12]


def parse_sample(name: str) -> tuple[str, str, str]:
    """``engine/experiment/sample`` → its parts; raises ``ValueError``."""
    parts = name.strip("/").split("/")
    if len(parts) != 3 or not all(parts):
        raise UsageError(f"PhySimCode samples are named engine/experiment/sample, got {name!r}")
    engine, experiment, sample = parts
    if engine not in ENGINES:
        raise UsageError(
            f"{name}: the {engine!r} engine is not supported yet (supported: {list(ENGINES)})"
        )
    return engine, experiment, sample


def download(name: str) -> Path:
    """One sample's four files, into the cache (once)."""
    parse_sample(name)
    target = data_dir() / name
    if not remove_damaged(target / f for f in SAMPLE_FILES):
        return target
    try:
        from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SetupError(
            "the PhySimCode adapter needs huggingface_hub",
            fix=install_command("physimcode"),
        ) from error
    with bounded():
        snapshot_download(
            DATASET, repo_type="dataset", revision=REVISION, allow_patterns=[f"{name}/*"],
            local_dir=data_dir(), force_download=True,
        )  # fmt: skip
    fsync_tree(target)
    missing = [f for f in SAMPLE_FILES if not (target / f).is_file()]
    if missing:
        raise SetupError(f"PhySimCode sample {name} has no {', '.join(missing)} at {REVISION[:12]}")
    if not (target / "simulation_code.py").read_text(encoding="utf-8").strip():
        raise SetupError(f"PhySimCode sample {name}: its ground-truth program is empty")
    return target


def statement(experiment: str, info: Any) -> str:
    return f"""\
# Task: explain a physics simulation and write the code that regenerates it

`{{workspace}}/visible/cam/rgb.mp4` ({info.width} x {info.height}, {info.fps:g} fps,
{info.num_frames} frames) shows a 2D physics simulation, rendered with Matplotlib from a SciPy
model (PhySimCode, phenomenon `{experiment.replace("_", " ")}`). Work out the physics: what moves,
the governing law, and the value of every parameter. Then write a program that regenerates the
video from that model.

## Deliver

1. `{{workspace}}/solution/cot.json`, your explanation, as JSON with these keys:
   - `simulation`: a short description of what is simulated;
   - `physical_observation`: what you observe in the video;
   - `physical_law`: `{{"name": ..., "statement": ..., "formula": ...}}`;
   - `parameters`: one entry per parameter, `{{"<name>": {{"value": <number>, "unit": "<unit>",
     "description": "<what it is>"}}}}`, with SI-style units (`m`, `kg`, `s`, `rad`, `N/m`, ...).
2. `{{workspace}}/solution/run.py`, run as `python run.py` inside `solution/`: a self-contained
   program (NumPy, SciPy and Matplotlib only) that simulates the phenomenon with your parameters
   and renders it as a video at {FPS_OUT} fps to `$R2S_OUTPUT_DIR/render.mp4` (Matplotlib's
   `FFMpegWriter` works; `tools/r2s_twin.py` can also write frames).

Run your program once with `R2S_OUTPUT_DIR={{workspace}}/output` before you finish. It is
re-run frozen, offline, with only its own directory: put everything it needs in `solution/`,
and no video or frames cut from the clip.

## Scoring

PhySimCode's own measures: whether the frozen re-run produces the video, and how many of the
true parameters your `cot.json` reports, matched by name as PhySimCode matches them, and how many
of those lie within 20 % of the true value.
"""


def task_config(name: str) -> TaskConfig:
    engine, experiment, sample = parse_sample(name)
    return TaskConfig(
        id=f"physimcode/{engine}/{experiment}/{sample}",
        version="1.0.0",
        title=f"PhySimCode {experiment} ({engine}, sample {sample})",
        description="Explain a simulated physics video and write the code that regenerates it.",
        source=TaskSource(
            dataset=DATASET,
            version=REVISION,
            url=f"https://huggingface.co/datasets/{DATASET}",
            citation=CITATION,
            notes=f"evaluation code {CODE.url} at {CODE.commit[:12]} (no license declared)",
        ),
        license=LICENSE,
        redistributable=False,
        tags=("physimcode", engine, experiment),
        visibility="video_only",
        delivery=Delivery(
            environment=ENVIRONMENT,
            output_format="video",
            agent_time_budget_s=1800,
            run_timeout_s=300,
            resources=Resources(gpu=False, cpus=2, memory_mb=4096),
            program=ProgramRules(max_size_mb=5.0),
        ),
        scoring=ScoringConfig(
            protocol=Protocol.NATIVE,
            native=NativeScoring(scorer="physimcode", options={"rerun": True}),
        ),
        metadata={
            "physimcode": {
                "engine": engine,
                "experiment": experiment,
                "sample": sample,
                "dataset_revision": REVISION,
                "code_commit": CODE.commit,
            }
        },
    )


def build_case(name: str, out_dir: Path, *, overwrite: bool = False) -> Path:
    engine, experiment, sample = parse_sample(name)
    files = download(name)
    info = probe_video(files / "video.mp4")
    config = task_config(name)
    root = Path(out_dir) / engine / experiment / sample
    builder = TaskBuilder(root, config, overwrite=overwrite)
    with tempfile.TemporaryDirectory(prefix="r2s-physimcode-") as tmp:
        media = Path(tmp)
        (media / "cam").mkdir()
        shutil.copyfile(files / "video.mp4", media / "cam" / "rgb.mp4")
        capture = Capture(
            capture_id=config.id,
            source=SourceInfo(
                dataset=DATASET, version=REVISION[:12], license=LICENSE, url=config.source.url
            ),
            world=WorldFrame(metric_scale=Sourced(value=False, provenance=Provenance.DEFAULT)),
            views={
                "cam": View(
                    rgb=VideoStream(
                        path="cam/rgb.mp4",
                        fps=info.fps,
                        num_frames=info.num_frames,
                        resolution=(info.width, info.height),
                    )
                )
            },
            text=TextInfo(task="Explain the simulation and write the code that regenerates it."),
        )
        builder.capture(capture, media)
    for file in ("cot.json", "params.json", "simulation_code.py"):
        builder.hidden_file(f"reference/{file}", source=files / file)
    builder.instruction(statement(experiment, info))
    return builder.finish().root


class PhySimCodeBenchmark(BenchmarkAdapter):
    name: ClassVar[str] = "physimcode"
    summary: ClassVar[str] = (
        "PhySimCode: explain a simulated 2D physics video and write the code that regenerates "
        "it; runnability and PhySimCode's parameter recovery."
    )
    license: ClassVar[str] = (
        "data MIT (downloaded at build time); evaluation code none declared (fetched at score time)"
    )
    source: ClassVar[str] = "https://github.com/SourajaKundu/PhySimCode"
    protocols: ClassVar[tuple[str, ...]] = ("native",)
    status: ClassVar[str] = "partial"

    def task_names(self) -> list[str]:
        return [f"physimcode/{name}" for name in DEFAULT_SAMPLES]

    def build(
        self,
        out_dir: Path,
        *,
        select: Sequence[str] | None = None,
        overwrite: bool = False,
        **options: Any,
    ) -> list[Path]:
        """``select`` names samples ``engine/experiment/sample``."""
        if options:
            raise UnknownOptionError.from_names(
                f"benchmark {self.name!r}", options, self.options_taken()
            )
        return [  # a name given twice is built once
            build_case(n, Path(out_dir), overwrite=overwrite)
            for n in dict.fromkeys(select or DEFAULT_SAMPLES)
        ]


def read_reference(task_hidden: Path) -> dict[str, Any]:
    """The true explanation (``cot.json``) of a task."""
    return json.loads((task_hidden / "reference" / "cot.json").read_text(encoding="utf-8"))
