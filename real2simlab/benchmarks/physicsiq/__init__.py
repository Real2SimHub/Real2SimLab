"""Physics-IQ: watch 3 seconds of a real physical event, predict the next 5 seconds as video.

Source: the Physics-IQ benchmark (Google DeepMind; code Apache-2.0, pinned below; data
CC-BY-4.0). Two releases of the videos: ``original`` (default), the public Google Cloud Storage
bucket ``physics-iq-benchmark``, pinned by object generation and checked by MD5; and
``verified``, the "Physics-IQ Verified" Hugging Face dataset, which is gated (accept its terms,
then ``hf auth login``, or ``huggingface-cli login`` before huggingface_hub 1.0, or set
``HF_TOKEN``), pinned by revision. Videos are downloaded into the user's cache when tasks are
built.

Motamed, Culp, Swersky, Jaini, Geirhos. *Do generative video models understand physical
principles?* arXiv:2501.09038.

Each of the 66 scenarios is filmed from three perspectives (left, center, right), twice. A
task shows take 1's first 3 seconds of one perspective (30 fps) and asks for the next 5 seconds
(150 frames) as a video, ``render.mp4``. Scoring is Physics-IQ's own (scorer plugin
``physics_iq``, running the official code): spatial IoU, spatiotemporal IoU, weighted spatial
IoU and MSE of motion masks and frames against take 1, and the same metrics between take 1 and
take 2 — the physical variance a perfect prediction could not beat. Variants:

* ``native``: the official input, the video the method produced in its session;
* ``rerun``: the video of the Lab's frozen, offline re-run of the delivered program.

Physics-IQ asks that its test data never be used to improve a submission. The Lab keeps the 5
seconds to predict, take 2 and the masks in ``hidden/``, which no method ever sees.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from real2simlab._fs import atomic_write_text, durable_replace, fsync_file, warn_damaged
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
    OutputRequirements,
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
    name="physics-IQ-benchmark",
    url="https://github.com/google-deepmind/physics-IQ-benchmark.git",
    commit="d3f3de16a9c66b4d3bfd6bdb5518570e8b72ad48",
    license="Apache-2.0 (code); CC-BY-4.0 (data)",
    override_env="R2S_PHYSICSIQ",
)
RELEASES = ("original", "verified")
BUCKET = "physics-iq-benchmark"
GCS_API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
ORIGINAL_CUTOFF_US = 1_740_100_000_000_000
"""The pinned original release: objects written after 2025-02-21 are refused (generation
numbers are microseconds since the epoch)."""
DATASET = "Anates-Labs-Research/Physics-IQ-Verified"
REVISION = "b65f567dd8446a4076c7f72446335b3c217a8464"
LICENSE = "CC-BY-4.0"
CITATION = (
    "Motamed, Culp, Swersky, Jaini, Geirhos. Do generative video models understand physical "
    "principles? arXiv:2501.09038. Physics-IQ Verified: "
    "https://huggingface.co/datasets/Anates-Labs-Research/Physics-IQ-Verified"
)
PERSPECTIVES = ("left", "center", "right")
TAKE_2_OFFSET = 198
"""Take 2 of the recording with id ``n`` has id ``n + 198``."""
FPS = 30.0
CONDITIONING_SECONDS = 3
TEST_FRAMES = 150
"""5 seconds at 30 fps: what the official scorer compares."""
DEFAULT_CASES = ("ball-and-block-fall/center", "block-domino/center", "balls-collide/center")
VARIANTS = ("native", "rerun")
_TESTING = re.compile(
    r"split-videos/testing/30FPS/(\d{4})_testing-videos_30FPS_perspective-(\w+)_take-1_"
    r"trimmed-(.+)\.mp4"
)


def data_dir(release: str = "original") -> Path:
    return cache_dir() / "data" / "Physics-IQ" / release


@dataclass(frozen=True)
class Case:
    """One scenario seen from one perspective; ``id1`` and ``id2`` number its two takes."""

    scenario: str
    perspective: str
    id1: int

    @property
    def id2(self) -> int:
        return self.id1 + TAKE_2_OFFSET

    @property
    def name(self) -> str:
        return f"{self.scenario}/{self.perspective}"

    def file(self, kind: str, take: int) -> str:
        """The dataset path of the 30 fps ``conditioning``, ``testing`` or ``mask`` video."""
        number = self.id1 if take == 1 else self.id2
        stem = f"perspective-{self.perspective}_take-{take}_trimmed-{self.scenario}.mp4"
        if kind == "mask":
            return f"video-masks/real/30FPS/{number:04d}_video-masks_30FPS_{stem}"
        folder = {"conditioning": "conditioning", "testing": "testing"}[kind]
        return f"split-videos/{folder}/30FPS/{number:04d}_{folder}-videos_30FPS_{stem}"


def _gcs_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _list_testing_files(release: str) -> list[str]:
    if release == "verified":
        try:
            from huggingface_hub import HfApi  # pyright: ignore[reportMissingImports]
        except ImportError as error:  # pragma: no cover - environment dependent
            raise SetupError(
                "the Physics-IQ verified release needs huggingface_hub",
                fix=install_command("physicsiq"),
            ) from error
        with bounded():
            return HfApi().list_repo_files(DATASET, repo_type="dataset", revision=REVISION)
    names: list[str] = []
    token: str | None = None
    prefix = urllib.parse.quote("split-videos/testing/30FPS/", safe="")
    while True:
        url = f"{GCS_API}?prefix={prefix}&fields=items(name),nextPageToken&maxResults=1000"
        page = _gcs_json(url + (f"&pageToken={urllib.parse.quote(token)}" if token else ""))
        names += [item["name"] for item in page.get("items", [])]
        token = page.get("nextPageToken")
        if not token:
            return names


def case_index(release: str = "original") -> dict[str, Case]:
    """``scenario/perspective`` → case, from the release's file list (cached; a damaged cache,
    e.g. left empty or truncated by a crash, is rebuilt)."""
    cache = data_dir(release) / "index.json"
    rows = _read_index(cache) if cache.is_file() else None
    if rows is None:
        rows = sorted(
            (int(m.group(1)), m.group(2), m.group(3))
            for m in (_TESTING.fullmatch(name) for name in _list_testing_files(release))
            if m
        )
        if not rows:
            raise SetupError(f"no Physics-IQ cases found in the {release} release")
        atomic_write_text(cache, json.dumps(rows) + "\n")
    cases = [Case(scenario, perspective, number) for number, perspective, scenario in rows]
    return {case.name: case for case in cases}


def _read_index(cache: Path) -> list[tuple[int, str, str]] | None:
    """The cached index's rows, or ``None`` (with a warning) if it cannot be read."""
    try:
        data: Any = json.loads(cache.read_text("utf-8"))
        rows = [(int(n), str(p), str(s)) for n, p, s in data]
    except (OSError, UnicodeDecodeError, ValueError, TypeError) as error:
        warn_damaged(cache, f"{type(error).__name__}: {error}")
        return None
    if not rows:
        warn_damaged(cache, "no cases")
        return None
    return rows


def download(case: Case, path: str, release: str = "original") -> tuple[Path, str]:
    """One file of the release into the cache; returns it and its pin (generation and MD5,
    or the dataset revision)."""
    target = data_dir(release) / path
    pin_file = target.with_name(target.name + ".pin")
    if target.is_file() and pin_file.is_file():
        pin = pin_file.read_text(encoding="utf-8", errors="replace").strip()
        problem = _download_problem(target, pin)
        if problem is None:
            return target, pin
        warn_damaged(target, problem)
        target.unlink(missing_ok=True)
        pin_file.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if release == "verified":
        from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]

        with bounded():
            local = hf_hub_download(
                DATASET, path, repo_type="dataset", revision=REVISION,
                local_dir=data_dir(release), force_download=True,
            )  # fmt: skip
        pin = f"{DATASET}@{REVISION}"
        if Path(local) != target:
            shutil.copyfile(local, target)
        fsync_file(target)
    else:
        quoted = urllib.parse.quote(path, safe="")
        meta = _gcs_json(f"{GCS_API}/{quoted}?fields=generation,md5Hash")
        generation = int(meta["generation"])
        if generation > ORIGINAL_CUTOFF_US:
            raise SetupError(
                f"gs://{BUCKET}/{path} changed after the pinned release (generation {generation})",
                fix="update ORIGINAL_CUTOFF_US after checking the new release, or use "
                "--option release=verified",
            )
        from real2simlab import _fs

        # what a download killed outright left is removed (not the file of one still running)
        _fs.remove_abandoned(target.parent, re.escape(f".{target.name}."), r"\.partial")
        partial = _fs.held_path(
            lambda part: target.with_name(f".{target.name}.{part}.partial"), directory=False
        )
        url = f"{GCS_API}/{quoted}?alt=media&generation={generation}"
        md5 = hashlib.md5()
        try:
            with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as out:
                while chunk := response.read(1 << 20):  # never the whole file in memory
                    md5.update(chunk)
                    out.write(chunk)
            digest = base64.b64encode(md5.digest()).decode()
            if digest != meta["md5Hash"]:
                raise SetupError(f"gs://{BUCKET}/{path}: MD5 mismatch after download; retry")
            durable_replace(partial, target)
        finally:  # moved into place, or a failed download: nothing is left beside the target
            _fs.force_remove(partial)
        pin = f"gs://{BUCKET}/{path}#{generation} md5={digest}"
    atomic_write_text(pin_file, pin + "\n")  # last: a pin vouches for a complete file
    return target, pin


def _download_problem(target: Path, pin: str) -> str | None:
    """Why a downloaded file and its pin cannot be reused (its MD5, when the pin has one)."""
    if not pin:
        return "its pin is empty"
    if target.stat().st_size == 0:
        return "empty"
    expected = re.search(r"md5=(\S+)", pin)
    if expected:
        digest = base64.b64encode(hashlib.md5(target.read_bytes()).digest()).decode()
        if digest != expected.group(1):
            return "its MD5 does not match its pin"
    return None


def case_files(case: Case, release: str = "original") -> tuple[dict[str, Path], dict[str, str]]:
    """The conditioning clip, both takes' test clips and both takes' motion masks, with pins."""
    files: dict[str, Path] = {}
    pins: dict[str, str] = {}
    wanted = {"conditioning": case.file("conditioning", 1)}
    for take in (1, 2):
        wanted[f"testing_{take}"] = case.file("testing", take)
        wanted[f"mask_{take}"] = case.file("mask", take)
    for key, path in wanted.items():
        files[key], pins[key] = download(case, path, release)
    return files, pins


def statement(case: Case, width: int, height: int) -> str:
    return f"""\
# Task: predict the next 5 seconds of a real video

`{{workspace}}/visible/cam/rgb.mp4` is the first {CONDITIONING_SECONDS} seconds ({FPS:g} fps,
{width} x {height}) of a real recording of a physical event ("{case.scenario.replace("-", " ")}",
seen from the {case.perspective}). Predict what the camera sees **in the 5 seconds that follow**,
starting right after its last frame.

## Deliver

1. `{{workspace}}/solution/run.py`, run as `python run.py` inside `solution/`, which writes
   `$R2S_OUTPUT_DIR/render.mp4`: exactly {TEST_FRAMES} frames at {FPS:g} fps (any resolution;
   the scorer resizes; the same aspect ratio is best). `tools/r2s_twin.py` can write it
   (`Twin().write_video(frames, fps={FPS:g})`). Blender 4.0 is available (`blender`) if you
   want to simulate and render a scene.

Run your program once with `R2S_OUTPUT_DIR={{workspace}}/output` before you finish. When it is
re-run frozen, it sees only its own directory and has no network: put everything it needs in
`solution/` (but no frames or video cut from the visible clip: they would not show the future).

## Scoring

Physics-IQ's own metrics: where things move in your video (motion masks) compared with where
they move in the real continuation — spatial IoU, spatiotemporal IoU and weighted spatial IoU —
and the per-frame MSE. A second recording of the same event sets the bar: two real takes do
not match each other perfectly either.
"""


def release_source(release: str) -> tuple[str, str, str]:
    """(dataset name, version, URL) of a release."""
    if release == "verified":
        return DATASET, REVISION, f"https://huggingface.co/datasets/{DATASET}"
    return (
        f"Physics-IQ (gs://{BUCKET})",
        f"objects written before generation {ORIGINAL_CUTOFF_US}",
        f"https://console.cloud.google.com/storage/browser/{BUCKET}",
    )


def task_config(
    case: Case, variant: str, release: str = "original", pins: dict[str, str] | None = None
) -> TaskConfig:
    rerun = variant == "rerun"
    prefix = "physicsiq" if release == "original" else "physicsiq-verified"
    dataset, version, url = release_source(release)
    return TaskConfig(
        id=f"{prefix}/{case.scenario}/{case.perspective}/{variant}",
        version="1.0.0",
        title=f"Physics-IQ {case.scenario} ({case.perspective}, "
        f"{'frozen re-run' if rerun else 'official input'})",
        description=f"Predict 5 seconds of '{case.scenario}' from its first 3 seconds.",
        source=TaskSource(
            dataset=dataset,
            version=version,
            url=url,
            citation=CITATION,
            notes=f"{release} release; take 1 id {case.id1:04d}, take 2 id {case.id2:04d}",
        ),
        license=LICENSE,
        redistributable=True,
        tags=("physicsiq", release, "forecast", case.perspective, variant),
        visibility="video_only",
        delivery=Delivery(
            environment="r2s-blender-4.0",
            output_format="video",
            agent_time_budget_s=3600,
            run_timeout_s=1800,
            resources=Resources(gpu=True, cpus=8, memory_mb=16384),
            program=ProgramRules(
                max_size_mb=5.0, forbidden_suffixes=(".mp4", ".png", ".jpg", ".jpeg", ".npy")
            ),
            output=OutputRequirements(fps=FPS, num_frames=TEST_FRAMES),
        ),
        scoring=ScoringConfig(
            protocol=Protocol.NATIVE,
            native=NativeScoring(scorer="physics_iq", options={"rerun": rerun}),
        ),
        metadata={
            "physicsiq": {
                "release": release,
                "scenario": case.scenario,
                "perspective": case.perspective,
                "take_1": case.id1,
                "take_2": case.id2,
                "code_commit": CODE.commit,
                "files": dict(pins or {}),
            }
        },
    )


def case_capture(case: Case, config: TaskConfig, info: Any) -> Capture:
    """The visible observation: take 1's first 3 seconds, as recorded."""
    return Capture(
        capture_id=config.id.rsplit("/", 1)[0],
        source=SourceInfo(
            dataset=config.source.dataset,
            version=REVISION[:12] if "verified" in config.tags else f"<={ORIGINAL_CUTOFF_US}",
            license=LICENSE,
            url=config.source.url,
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
        text=TextInfo(
            task="Predict the next 5 seconds of the video.",
            scene_description=f"{case.scenario.replace('-', ' ')}, seen from the "
            f"{case.perspective}.",
        ),
    )


def build_case(
    case_name: str,
    variant: str,
    out_dir: Path,
    *,
    release: str = "original",
    overwrite: bool = False,
) -> Path:
    if release not in RELEASES:
        raise UsageError(f"unknown Physics-IQ release {release!r}; known: {list(RELEASES)}")
    index = case_index(release)
    if case_name not in index:
        raise UsageError(f"no Physics-IQ case {case_name!r}; e.g. {', '.join(DEFAULT_CASES)}")
    case = index[case_name]
    files, pins = case_files(case, release)
    info = probe_video(files["conditioning"])
    config = task_config(case, variant, release, pins)
    root = Path(out_dir) / case.scenario / case.perspective / variant
    builder = TaskBuilder(root, config, overwrite=overwrite)
    with tempfile.TemporaryDirectory(prefix="r2s-physicsiq-") as tmp:
        media = Path(tmp)
        (media / "cam").mkdir()
        shutil.copyfile(files["conditioning"], media / "cam" / "rgb.mp4")
        builder.capture(case_capture(case, config, info), media)
    for key in ("testing_1", "testing_2", "mask_1", "mask_2"):
        builder.hidden_file(f"reference/{key}.mp4", source=files[key])
    record = {
        "release": release,
        "scenario": case.scenario,
        "perspective": case.perspective,
        "take_1": case.id1,
        "take_2": case.id2,
    }
    builder.hidden_file("reference/case.json", text=json.dumps(record, indent=2) + "\n")
    builder.instruction(statement(case, info.width, info.height))
    return builder.finish().root


class PhysicsIQBenchmark(BenchmarkAdapter):
    name: ClassVar[str] = "physicsiq"
    summary: ClassVar[str] = (
        "Physics-IQ: from 3 seconds of a real physical event, predict the next 5 as video; "
        "the official motion-mask metrics, with a second take as the physical variance."
    )
    license: ClassVar[str] = "Apache-2.0 code, CC-BY-4.0 data (downloaded at build time)"
    source: ClassVar[str] = "https://github.com/google-deepmind/physics-IQ-benchmark"
    protocols: ClassVar[tuple[str, ...]] = ("native",)
    status: ClassVar[str] = "partial"
    training_terms: ClassVar[str | None] = (
        '"the test data must under no circumstances be used to improve a submission score - '
        'not for BoN selection, and not by a prompt rewriter" (README of the pinned checkout)'
    )

    def task_names(self) -> list[str]:
        return [f"physicsiq/{c}/{v}" for c in DEFAULT_CASES for v in VARIANTS]

    def build(
        self,
        out_dir: Path,
        *,
        select: Sequence[str] | None = None,
        variants: Sequence[str] = VARIANTS,
        overwrite: bool = False,
        **options: Any,
    ) -> list[Path]:
        """``select`` names cases ``scenario/perspective`` (``scenario`` alone: center); the
        option ``release`` is ``original`` (default, public) or ``verified`` (gated)."""
        release = str(options.pop("release", "original"))
        if options:
            raise UnknownOptionError.from_names(
                f"benchmark {self.name!r}", options, self.options_taken("release")
            )
        bad = sorted(set(variants) - set(VARIANTS))
        if bad:
            raise UsageError(f"unknown variants {bad}; known: {list(VARIANTS)}")
        # a case given twice (also as scenario and scenario/center) is built once
        cases = dict.fromkeys(c if "/" in c else f"{c}/center" for c in (select or DEFAULT_CASES))
        return [
            build_case(case, variant, Path(out_dir), release=release, overwrite=overwrite)
            for case in cases
            for variant in dict.fromkeys(variants)
        ]
