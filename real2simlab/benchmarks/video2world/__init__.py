"""Video2World: reconstruct a manipulation video as an executable simulation package.

Source: Video2World (AetherLabs). Code: Apache-2.0, pinned below and run at score time. Data:
the Hugging Face dataset ``AetherLabs-AI/Video2World``, pinned by revision, not gated. Its card
says Apache-2.0, but its samples come from other datasets under their own terms, so every task
records the license of its own source (:data:`SOURCES`), and nothing is redistributed: the
samples are downloaded into the user's cache when tasks are built.

Each instance is one monocular RGB video of a robot or a human hand manipulating objects. The
method delivers a video2sim protocol package in ``solution/``: the reconstructed scene and the
behaviour (an action stream) that should reproduce the outcome. Video2World's own evaluator
executes the package and scores the instance: ``S = build * (F + G + D) / 3``, with F the task
outcome (success and progress), G the geometry errors and D the dynamics errors, each error
turned into ``max(0, 1 - error / tau)`` (scorer plugin ``video2world``).

The agent gets the video, Video2World's own brief for the instance's profile, the ``video2sim``
library and the public handouts Video2World stages for some profiles (task and robot
specifications, robot assets, contract checkers). The ground truth, the reference package and
the per-sample metric rules stay in ``hidden/``; Video2World's ``v2w`` package, which holds
rules derived from the ground truth, is never given to the agent. Tasks are scored natively.

All twelve evaluators of the manifest (222 instances) are supported (:data:`EVALUATORS`):
ManiSkill / SAPIEN (``fb``, the hand tracks) and MuJoCo (the reconstructed twins, cloth) in the
``r2s-video2world`` image, Isaac Sim (RoboDojo, the in-house episodes) in
``r2s-video2world-isaac``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from real2simlab._fs import fsync_file, remove_damaged
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
from real2simlab.twin.conventions import DeliveryConvention
from real2simlab.twin.model import ProgramRules
from real2simlab.vendor import VendoredRepo, cache_dir

CODE = VendoredRepo(
    name="Video2World",
    url="https://github.com/AetherLabsAI/Video2World.git",
    commit="df0e00b7893b24968e87c5a51a67d824482e431b",
    license="Apache-2.0",
    override_env="R2S_VIDEO2WORLD",
)
DATASET = "AetherLabs-AI/Video2World"
REVISION = "9cac08429be06ad190b22d6e172af2f815024e3f"
CITATION = (
    "Tang, Zhang, Yang et al. Video2World: Benchmarking Coding Agents for Interactive World "
    "Modeling from Embodied Videos. arXiv:2610.04432. https://github.com/AetherLabsAI/Video2World"
)
ENVIRONMENT = "r2s-video2world"
"""The agents' runtime and the scorer's image for the ManiSkill / SAPIEN and MuJoCo evaluators."""
ISAAC_ENVIRONMENT = "r2s-video2world-isaac"
"""The scorer's image for the Isaac Sim evaluators (RoboDojo, in-house)."""
TWIN_PYTHON = "/opt/v2w-twin/bin/python"
"""The MuJoCo interpreter of ``r2s-video2world`` (Video2World's ``twin`` extra)."""
ISAAC_PYTHON = "/opt/robodojo/bin/python"
"""The Isaac Sim interpreter of ``r2s-video2world-isaac``."""
ROBODOJO_DATASET = "RoboDojo-Benchmark/RoboDojo"
ROBODOJO_REVISION = "1127bda94f172d757386945af3eb3a6754d1a4e6"
"""RoboDojo's assets (Hugging Face, Apache-2.0), at the revision the Lab pins."""
ROBODOJO_ASSETS_TARGET = "/opt/RoboDojo/Assets"
"""Where RoboDojo's checkout in ``r2s-video2world-isaac`` expects its assets."""
ROBODOJO_ASSET_DIRS = ("Robots", "Object", "Material", "Eval_Layout")
"""The directories RoboDojo's ``scripts/init_assets.sh`` requires."""


@dataclass(frozen=True)
class Evaluator:
    """How the Lab runs one of Video2World's evaluators (``v2w.benchmark.EVALUATORS``)."""

    image: str
    """The environment the scorer runs it in."""
    python: str
    """The interpreter that runs the scoring driver in that image."""
    shared: tuple[str, ...]
    """Dataset directories it reads besides the sample: downloaded into the cache at score
    time and mounted read-only, never put in tasks."""
    agent_gpu: bool
    """Whether the agent's own replay needs a GPU (SAPIEN needs a Vulkan device)."""
    scorer_gpu: bool
    timeout_s: int
    """Video2World's evaluator timeout (5400 s), plus time for its metrics and start-up."""
    mount_shared: bool = False
    """Mount the shared directories read-only instead of copying them (they are gigabytes;
    SAPIEN writes collision caches next to its meshes, so its evaluators get copies)."""
    robodojo_assets: bool = False
    """Whether it needs RoboDojo's own assets (:func:`robodojo_assets`)."""


SAPIEN_TIMEOUT_S = 5400 + 600
ISAAC_TIMEOUT_S = 5400 + 1800
"""Isaac Sim compiles its shaders on a fresh start, which takes minutes."""
FB = Evaluator(ENVIRONMENT, "python", ("assets/maniskill",), True, True, SAPIEN_TIMEOUT_S)
HAND = Evaluator(
    ENVIRONMENT,
    "python",
    ("assets/maniskill", "assets/hand", "assets/datasets"),
    agent_gpu=True,
    scorer_gpu=True,
    timeout_s=SAPIEN_TIMEOUT_S,
)
TWIN = Evaluator(ENVIRONMENT, "python", (), agent_gpu=False, scorer_gpu=False, timeout_s=6000)
"""The MuJoCo twins and cloth: the driver runs in ``python``, the evaluator in
``V2W_TWIN_PYTHON``."""
ROBODOJO = Evaluator(
    ISAAC_ENVIRONMENT,
    ISAAC_PYTHON,
    (
        "assets/robodojo",
        "assets/robots",
        "assets/twins/upstream/real2sim-eval/assets/robots",
        "sources/robodojo",
    ),
    agent_gpu=False,
    scorer_gpu=True,
    timeout_s=ISAAC_TIMEOUT_S,
    mount_shared=True,
    robodojo_assets=True,
)
INHOUSE = Evaluator(
    ISAAC_ENVIRONMENT,
    ISAAC_PYTHON,
    ("assets/inhouse", "assets/robots", "sources/inhouse"),
    agent_gpu=False,
    scorer_gpu=True,
    timeout_s=ISAAC_TIMEOUT_S,
    mount_shared=True,
)
EVALUATORS: dict[str, Evaluator] = {
    "fb": FB,
    "ego_arm": HAND,
    "ego_bimanual": HAND,
    "twin": TWIN,
    "twin_toy": TWIN,
    "cloth": TWIN,
    "robodojo_native_task": ROBODOJO,
    "inhouse": INHOUSE,
    "inhouse_point_robot": INHOUSE,
    "inhouse_reviewed": INHOUSE,
    "inhouse_wallet": INHOUSE,
    "inhouse_fryer": INHOUSE,
}
"""Every evaluator of Video2World's registry, with what the Lab needs to run it."""
SUPPORTED_EVALUATORS = tuple(EVALUATORS)
DEFAULT_SAMPLES = (
    "twin_pusht_demo_0",
    "twin_pusht_demo_1",
    "twin_pusht_demo_2",
    "droid_guptalab_2023_05_19_11h_06m_39s_orange_hand",
    "droid_iris_2023_03_02_16h_02m_41s_bowl_hand",
    "droid_tri_2023_08_09_08h_56m_36s_block_hand",
)
"""Push-T (MIT) and DROID (CC-BY-4.0): clean licenses, and the reference reproduces."""
SHARED_FILES = ("manifest.json", "paths.json")
SHARED_PATTERNS = ("assets/maniskill/**",)

# Video2World's profiles, as ``v2w.agents.checks`` groups them.
NATIVE_PROFILES = ("robodojo_candidate_task_v1", "inhouse_provider_native_v2")
TWIN_PROFILES = ("twin_rope", "twin_toy")
CLOTH_PROFILES = ("droid_cloth",)
NO_COMMON = NATIVE_PROFILES + TWIN_PROFILES + CLOTH_PROFILES
"""Profiles whose brief does not end with ``common.md`` (``v2w.agents.run.NO_COMMON``)."""


@dataclass(frozen=True)
class Source:
    """The terms of the dataset a family's samples come from (D-043)."""

    license: str
    attribution: str


SOURCES: dict[str, Source] = {
    "DROID": Source(
        "CC-BY-4.0",
        "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset (Khazatsky et al., 2024), "
        "https://droid-dataset.github.io; video and poses adapted by Video2World",
    ),
    "real2sim-eval": Source(
        "MIT",
        "real2sim-eval (https://github.com/kywind/real2sim-eval) and PhysTwin "
        "(https://github.com/Jianghanxiao/PhysTwin); reconstructed twin adapted by Video2World",
    ),
    "FurnitureBench": Source(
        "none declared",
        "FurnitureBench (Heo et al., 2023), https://clvrai.github.io/furniture-bench; the "
        "dataset declares no license (its code is MIT); adapted by Video2World",
    ),
    "DexYCB": Source(
        "CC-BY-NC-4.0",
        "DexYCB (Chao et al., 2021), https://dex-ycb.github.io, with the YCB object set; video, "
        "poses and meshes adapted by Video2World",
    ),
    "HOI4D": Source(
        "CC-BY-NC-4.0",
        "HOI4D (Liu et al., 2022), https://hoi4d.github.io; video, poses and CAD models adapted "
        "by Video2World",
    ),
    "HOT3D": Source(
        "CC-BY-SA AND CC-BY-NC-SA",
        "HOT3D (Banerjee et al., 2025), https://facebookresearch.github.io/hot3d; HOT3D Dataset "
        "License Agreement: sequences and object models CC BY-SA, hand annotations CC BY-NC-SA; "
        "adapted by Video2World",
    ),
    "OakInk2": Source(
        "CC-BY-SA-4.0",
        "OakInk2 (Zhan et al., 2024), https://oakink.net/v2 (dataset card kelvin34501/OakInk-v2); "
        "video, poses and meshes adapted by Video2World",
    ),
    "RoboDojo": Source(
        "Apache-2.0",
        "RoboDojo (https://github.com/RoboDojo-Benchmark/RoboDojo; code MIT, assets Apache-2.0 "
        "on Hugging Face RoboDojo-Benchmark/RoboDojo); episodes newly recorded in RoboDojo's "
        "simulation by Video2World",
    ),
    "in-house": Source(
        "Apache-2.0 AND LicenseRef-AgiBot-World-Community",
        "Video2World's own household recordings (the dataset's Apache-2.0) of the AgiBot G1 "
        "robot in Genie Sim scenes (https://github.com/AgibotTech/genie_sim; its assets are "
        "under the AgiBot World Community License Agreement); integrated with the permission "
        "the Real2SimLab maintainers obtained (2026-11-24)",
    ),
}
"""Keyed by the first word of the manifest's ``source``; the in-house evaluators' samples
(whose ``source`` describes their review) by ``in-house``."""


def data_dir() -> Path:
    return cache_dir() / "data" / "Video2World" / REVISION[:12]


def _hub() -> Any:
    try:
        import huggingface_hub  # pyright: ignore[reportMissingImports]
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SetupError(
            "the Video2World adapter needs huggingface_hub",
            fix=install_command("video2world"),
        ) from error
    return huggingface_hub


def manifest() -> dict[str, Any]:
    """Video2World's manifest at the pinned revision (downloaded on first use)."""
    path = data_dir() / "manifest.json"
    if remove_damaged([path]):
        with bounded():
            _hub().hf_hub_download(
                DATASET, "manifest.json", repo_type="dataset", revision=REVISION,
                local_dir=data_dir(), force_download=True,
            )  # fmt: skip
        fsync_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Instance:
    """One configuration of the manifest: a sample, its family and its scoring contract. A
    sample can appear in several families (the hand tracks' arm and two-hand settings)."""

    family: str
    sample: str
    spec: Mapping[str, Any]
    row: Mapping[str, Any]

    @property
    def key(self) -> str:
        return f"{self.family}/{self.sample}"

    @property
    def evaluator(self) -> str:
        return str(self.spec.get("evaluator") or self.spec.get("eval2"))

    @property
    def profile(self) -> str:
        """The agent profile, as ``v2w.benchmark.Configuration.profile`` defines it."""
        profile = self.spec.get("profile")
        if profile:
            return str(profile)
        return "droid_cloth" if self.evaluator == "cloth" else ""

    @property
    def runtime(self) -> Evaluator:
        return EVALUATORS[self.evaluator]

    @property
    def source(self) -> Source | None:
        if self.evaluator.startswith("inhouse"):
            return SOURCES["in-house"]
        return SOURCES.get(str(self.spec.get("source", "")).split(" ")[0])


def instances() -> dict[str, Instance]:
    """``family/sample`` → instance, for every configuration of the manifest."""
    out: dict[str, Instance] = {}
    for row in manifest()["samples"]:
        found = Instance(row["family"], row["sample"], row["metric_family"], row)
        out[found.key] = found
    return out


def lookup(name: str) -> list[Instance]:
    """The instances ``name`` selects: ``family/sample``, a sample (each family that has it)
    or a family (all its samples). Raises ``UsageError`` when it names none."""
    known = instances()
    if name in known:
        return [known[name]]
    found = [i for i in known.values() if name in (i.sample, i.family)]
    if not found:
        raise UsageError(
            f"no Video2World sample or family {name!r}; e.g. {', '.join(DEFAULT_SAMPLES[:2])}"
        )
    return found


def instance(name: str) -> Instance:
    """The instance ``name`` (``family/sample``, or a sample that is in one family only);
    raises ``UsageError`` if this adapter cannot build it."""
    found = [i for i in lookup(name) if name in (i.key, i.sample)]
    if not found:
        raise UsageError(f"{name!r} is a family: name one of its samples")
    if len(found) > 1:
        raise UsageError(
            f"{name} is in several families: name one of {', '.join(i.key for i in found)}"
        )
    return _supported(found[0])


def _supported(found: Instance) -> Instance:
    if found.evaluator not in EVALUATORS:
        raise UsageError(
            f"{found.key} is scored by Video2World's {found.evaluator!r} evaluator, which this "
            f"adapter does not know (known: {list(SUPPORTED_EVALUATORS)})"
        )
    if found.source is None:
        raise UsageError(
            f"{found.key}: no license recorded for source {found.spec.get('source')!r}"
        )
    return found


def public_task(checkout: Path, sample: str) -> dict[str, Any] | None:
    """Video2World's public task specification of a native-profile sample
    (``v2w/agents/tasks.json``), or ``None``."""
    tasks = json.loads((checkout / "v2w" / "agents" / "tasks.json").read_text(encoding="utf-8"))
    spec = tasks.get(sample)
    return dict(spec) if isinstance(spec, dict) else None


def handout_patterns(found: Instance, checkout: Path) -> list[str]:
    """The public robot assets Video2World stages for the agent of ``found``'s profile."""
    if found.profile in TWIN_PROFILES:
        return ["assets/robots/twin_xarm7/**"]
    if found.profile in NATIVE_PROFILES:
        spec = public_task(checkout, found.sample)
        if spec is None:
            raise SetupError(
                f"Video2World has no public task specification for {found.sample}",
                fix="the pinned checkout is incomplete: remove it from the cache and rebuild",
            )
        return [f"assets/robots/{spec['robot_asset']}/**"]
    return []


def download(found: Instance, extra: Sequence[str] = ()) -> Path:
    """The sample, its reference record, the shared files and ``extra`` patterns (public robot
    assets for the handout), into the cache (once)."""
    sample = data_dir() / "samples" / found.sample
    needed = [
        *(data_dir() / name for name in SHARED_FILES),
        sample / "video.mp4",
        sample / "meta.json",
        data_dir() / "har" / found.family / f"{found.sample}.json",
    ]
    marker = _marker("handout:" + ",".join(extra)) if extra else None
    if not remove_damaged(needed) and (marker is None or marker.is_file()):
        return sample
    patterns = [
        *SHARED_FILES,
        *SHARED_PATTERNS,
        *extra,
        f"samples/{found.sample}/**",
        f"har/{found.family}/{found.sample}.json",
    ]
    with bounded():
        _hub().snapshot_download(
            DATASET, repo_type="dataset", revision=REVISION, allow_patterns=patterns,
            local_dir=data_dir(),
        )  # fmt: skip
    for path in needed:
        if path.is_file():
            fsync_file(path)
    if not (sample / "video.mp4").is_file():
        raise SetupError(f"Video2World sample {found.sample} is incomplete in {data_dir()}")
    if marker is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("complete\n", encoding="utf-8")
    return sample


def _marker(what: str) -> Path:
    """A file that records a completed download of ``what`` (a partial download has none)."""
    digest = hashlib.sha256(what.encode()).hexdigest()[:16]
    return data_dir() / ".r2s-complete" / digest


# Video2World's agent brief, as v2w/agents/run.py composes it.
ROBOTS = {"fb_lamp": "panda", "fb_furniture": "panda", "fb_droid": "panda_robotiq",
          "fb_pusht": "xarm7_pusher", "ego_arm": "panda",
          "ego_bimanual": "wuji_bimanual_floating"}  # fmt: skip
"""``v2w.agents.checks.ROBOTS``: the execution robot of the rigid and hand profiles."""
ROBOT_BRIEF = (
    "\nExecution backend: video2sim native rigid robot scene, robot.uid={robot}. The actual robot "
    "base pose is required for native self-replay in the camera frame; do not substitute an "
    "identity base for a differently oriented robot. Task scoring is benchmark-side.\n"
)
HAND_BRIEF = (
    "\nExecution robot: {robot}. This profile uses the benchmark hand adapter; generic "
    "bridge/WidowX replay is not its execution interface. Use the hand profile validator and "
    "public hand control contract.\n"
)


def official_brief(checkout: Path, profile: str, fields: Mapping[str, Any]) -> str:
    """Video2World's brief for ``profile`` with ``fields`` filled in, as ``v2w.agents.run``
    composes it (the profile's template, ``common.md`` unless the profile has its own
    contract, the execution robot)."""
    profiles = checkout / "v2w" / "agents" / "profiles"
    path = profiles / f"{profile}.md"
    if profile == "common" or not path.is_file():
        known = sorted(p.stem for p in profiles.glob("*.md") if p.stem != "common")
        raise UsageError(f"profile {profile!r} is not one of {known}")
    text = path.read_text(encoding="utf-8")
    if profile not in NO_COMMON:
        text += (profiles / "common.md").read_text(encoding="utf-8")
    robot = ROBOTS.get(profile)
    if robot and profile.startswith("fb_"):
        text = text.replace('"uid": "widowx250s_bridge"', f'"uid": "{robot}"').replace(
            "// nominal: the robot is NOT part of this profile", "// actual execution robot"
        )
        text += ROBOT_BRIEF.format(robot=robot)
    elif robot:
        text += HAND_BRIEF.format(robot=robot)
    return text.format(**fields)


HANDOUT = "{workspace}/visible/tools/handout"
STAGE_TIMEOUT_S = 600
"""Staging copies a robot's assets (up to a few hundred MB)."""
CLOTH_CONTRACT = "{workspace}/visible/tools/cloth_contract"


def stage_handout(found: Instance, checkout: Path, target: Path) -> Path | None:
    """Video2World's public handout for ``found``'s profile, staged by its own code
    (``v2w.agents.handouts``) into ``target``; returns ``public_task.json`` (``None`` for
    profiles without one). Paths in the native specification are made relative to ``target``:
    the agent's workspace is elsewhere."""
    if found.profile in TWIN_PROFILES:
        code = (
            "from v2w.agents import handouts; print(handouts.stage_twin(sys.argv[1], sys.argv[2]))"
        )
        argument = found.profile
    elif found.profile in NATIVE_PROFILES:
        code = (
            "from v2w.agents import handouts; "
            "print(handouts.stage_native(Path(sys.argv[1]), Path(sys.argv[2])))"
        )
        argument = str(data_dir() / "samples" / found.sample)
    else:
        return None
    target.mkdir(parents=True)
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import sys; from pathlib import Path; {code}", argument,
             str(target)],
            env={**os.environ, "PYTHONPATH": str(checkout), "V2W_DATA": str(data_dir()),
                 "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL,
            timeout=STAGE_TIMEOUT_S,
        )  # fmt: skip
        failure = result.stderr.strip()[-600:] if result.returncode != 0 else ""
    except subprocess.TimeoutExpired:
        failure = f"it did not finish within {STAGE_TIMEOUT_S} s"
    public = target / "public_task.json"
    if failure or not public.is_file():
        raise SetupError(
            f"Video2World's handout for {found.key} could not be staged: {failure}",
            fix="check that the pinned checkout and the sample's robot assets are complete in "
            f"{CODE.path()} and {data_dir()}",
        )
    if found.profile in NATIVE_PROFILES:
        spec = json.loads(public.read_text(encoding="utf-8"))
        # native_contract.py falls back to the robot/ next to it.
        spec.pop("robot_asset_directory", None)
        prefixes = {str(target) + os.sep, str(target.resolve()) + os.sep}
        for key, value in list(spec.items()):
            for prefix in prefixes:
                if isinstance(value, str) and value.startswith(prefix):
                    spec[key] = value[len(prefix) :]
        public.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return public


def stage_cloth_contract(checkout: Path, target: Path) -> None:
    """``v2w.tracks.cloth.contract`` alone (it reads no sample data), as the package the cloth
    brief runs with ``python -m``: the rest of ``v2w`` holds rules derived from the truth."""
    module = target / "v2w" / "tracks" / "cloth"
    module.mkdir(parents=True)
    for directory in (target / "v2w", target / "v2w" / "tracks", module):
        (directory / "__init__.py").write_text("", encoding="utf-8")
    shutil.copyfile(checkout / "v2w" / "tracks" / "cloth" / "contract.py", module / "contract.py")


def statement(
    checkout: Path,
    found: Instance,
    meta: Mapping[str, Any],
    budget_s: int,
    public: Path | None = None,
) -> str:
    width, height = meta["resolution"]
    profile = found.profile
    name = found.sample if not profile.startswith("ego_") else f"{found.sample}__{profile}"
    brief = official_brief(
        checkout,
        profile,
        {
            "video": "{workspace}/visible/cam/rgb.mp4",
            "out": "{workspace}/solution",
            "name": name,
            "W": width,
            "H": height,
            "T": meta["T"],
            "ours": "{workspace}/visible/tools",
            "repo": CLOTH_CONTRACT if profile in CLOTH_PROFILES else "(not available)",
            "py_eval": "python",
            "py": TWIN_PYTHON if profile in TWIN_PROFILES else "python",
        },
    )
    if public is not None:
        content = public.read_text(encoding="utf-8")
        if profile in TWIN_PROFILES:
            brief += f"\nPublic robot contract: {HANDOUT}/public_task.json\n{content}"
        else:
            brief += f"\nPublic task/robot specification: {HANDOUT}/public_task.json\n{content}"
            if json.loads(content).get("wallet_dynamics") == "rigid_or_elastic":
                supplement = checkout / "v2w" / "agents" / "profiles" / "wallet_supplement.txt"
                brief += "\n" + supplement.read_text(encoding="utf-8")
    tools = [
        "- the video is `{workspace}/visible/cam/rgb.mp4`; the `video2sim` library is in\n"
        "  `{workspace}/visible/tools/`"
    ]
    if profile in TWIN_PROFILES:
        tools.append(
            f"- the public robot contract Video2World stages into the package is in\n"
            f"  `{HANDOUT}/` (`robot/`, `public_task.json`): copy both into\n"
            "  `{workspace}/solution/` before you start, as the brief expects them there;\n"
            f"  `{TWIN_PYTHON}` is the MuJoCo interpreter the brief runs"
        )
    elif profile in NATIVE_PROFILES:
        tools.append(
            f"- the public task and robot specification is in `{HANDOUT}/`\n"
            "  (`public_task.json`, `robot/`, `native_contract.py` and the task's rule files);\n"
            "  file names in `public_task.json` are relative to that directory"
        )
    elif profile in CLOTH_PROFILES:
        tools.append(
            f"- the brief's contract check is in `{CLOTH_CONTRACT}`\n"
            "  (only `v2w.tracks.cloth.contract`, which reads no sample data)"
        )
    else:
        tools[0] += (
            " (the brief's validator:\n"
            "  `PYTHONPATH={workspace}/visible/tools python -m video2sim.cli validate "
            "{workspace}/solution`;\n"
            "  `replay` works the same way)"
        )
    listing = ";\n".join(tools)
    return f"""\
# Task: Video2World instance `{found.sample}` ({found.family})

This is a task of the Video2World benchmark, with its own brief below. In this workspace:

{listing};
- `{{workspace}}/solution/` **is** the package: `protocol.json` at its top level. The package
  may contain `source/video.mp4`, a copy of the clip, as the brief's `protocol.json` suggests;
- you have {budget_s // 60} minutes; the package in `solution/` when you stop is scored, frozen,
  by Video2World's own evaluator. The network reaches only the model API.

---

{brief}"""


def task_config(found: Instance, *, budget_s: int = 3600) -> TaskConfig:
    source = found.source
    assert source is not None
    runtime = found.runtime
    return TaskConfig(
        id=f"video2world/{found.family}/{found.sample}",
        version="1.0.0",
        title=f"Video2World {found.family}: {found.sample}",
        description="Reconstruct a manipulation video as a scene and an action stream that "
        "reproduce its outcome (a video2sim protocol package).",
        source=TaskSource(
            dataset=DATASET,
            version=REVISION,
            url=f"https://huggingface.co/datasets/{DATASET}",
            citation=CITATION,
            notes=f"{source.attribution}. Video2World code {CODE.commit[:12]}.",
        ),
        license=source.license,
        redistributable=False,
        tags=("video2world", found.family, found.profile, "manipulation"),
        visibility="video_only",
        delivery=Delivery(
            environment=ENVIRONMENT,
            output_format="video2sim_package",
            convention=DeliveryConvention.VIDEO2SIM_PACKAGE,
            agent_time_budget_s=budget_s,
            run_timeout_s=runtime.timeout_s - 600,
            # SAPIEN needs a Vulkan device, for the agent's own replays too.
            resources=Resources(gpu=runtime.agent_gpu, cpus=4, memory_mb=8192),
            program=ProgramRules(max_size_mb=50.0, allow_video=True),
        ),
        scoring=ScoringConfig(
            protocol=Protocol.NATIVE,
            native=NativeScoring(scorer="video2world", options={}),
        ),
        metadata={
            "video2world": {
                "family": found.family,
                "sample": found.sample,
                "profile": found.profile,
                "evaluator": found.evaluator,
                "source": str(found.spec.get("source", "")),
                "code_commit": CODE.commit,
                "dataset_revision": REVISION,
            }
        },
    )


def shared_data(runtime: Evaluator) -> dict[str, Path]:
    """The dataset directories ``runtime`` reads besides the sample (``Evaluator.shared``), at
    the pinned revision, in the cache: relative path → local directory. The scorer mounts them
    into the evaluator's data release; tasks never carry them."""
    if not runtime.shared:
        return {}
    marker = _marker("shared:" + ",".join(runtime.shared))
    if not marker.is_file():
        with bounded():
            _hub().snapshot_download(
                DATASET, repo_type="dataset", revision=REVISION, local_dir=data_dir(),
                allow_patterns=[f"{directory}/**" for directory in runtime.shared],
            )  # fmt: skip
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("complete\n", encoding="utf-8")
    out = {directory: data_dir() / directory for directory in runtime.shared}
    for directory, path in out.items():
        if not path.is_dir():
            marker.unlink(missing_ok=True)
            raise SetupError(
                f"Video2World's {directory}/ is missing from {data_dir()} after the download",
                fix="check the network and score again (the download resumes)",
            )
    return out


def shared_assets() -> Path:
    """Video2World's public robot assets (``assets/maniskill``: the URDFs and meshes of the
    rigs), at the pinned revision, in the cache."""
    return shared_data(FB)["assets/maniskill"]


def robodojo_assets() -> Path:
    """RoboDojo's own assets (``Assets/``, about 41 GB, Apache-2.0) at the pinned revision, in
    the cache: RoboDojo's native tasks load their robots, objects and materials from there.
    As RoboDojo's ``utils/update_embodiment_config_path.py`` does, each robot configuration
    template (``*_tmp.yml``) is written out with ``$ASSETS_PATH`` set to RoboDojo's checkout in
    ``r2s-video2world-isaac``, where the scorer mounts this directory."""
    root = cache_dir() / "data" / "RoboDojo" / ROBODOJO_REVISION[:12]
    marker = root / ".r2s-complete"
    assets = root / "Assets"
    if not marker.is_file():
        with bounded():
            _hub().snapshot_download(
                ROBODOJO_DATASET, repo_type="dataset", revision=ROBODOJO_REVISION,
                local_dir=root, allow_patterns=["Assets/**"],
            )  # fmt: skip
        missing = [d for d in ROBODOJO_ASSET_DIRS if not (assets / d).is_dir()]
        if missing:
            raise SetupError(
                f"RoboDojo's assets in {assets} lack {', '.join(missing)}",
                fix="check the network and score again (the download resumes)",
            )
        checkout = str(Path(ROBODOJO_ASSETS_TARGET).parent)
        for template in sorted((assets / "Robots").rglob("*_tmp.yml")):
            text = template.read_text(encoding="utf-8")
            text = text.replace("${ASSETS_PATH}", checkout).replace("$ASSETS_PATH", checkout)
            template.with_name(template.name.replace("_tmp.yml", ".yml")).write_text(
                text, encoding="utf-8"
            )
        marker.write_text("complete\n", encoding="utf-8")
    return assets


PUBLIC_FILES = "public_files.json"
"""In a task's data release: files left out because the same content is public in the task's
``visible/`` (robot meshes the reference package shares with the ``video2sim`` library or the
handout), release path → visible path. The scorer puts them back."""


def public_digests(visible: Path) -> dict[str, str]:
    """sha256 → path under ``visible`` of every non-empty file there."""
    out: dict[str, str] = {}
    for path in sorted(visible.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.stat().st_size > 0:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            out.setdefault(digest, path.relative_to(visible).as_posix())
    return out


def hidden_release(
    found: Instance,
    sample_dir: Path,
    target: Path,
    *,
    public: Mapping[str, str] | None = None,
) -> None:
    """A one-instance Video2World data release at ``target``, without the shared data
    (:func:`shared_data`): the manifest row, ``paths.json``, the sample and its reference
    record. With the shared data added, it is what the evaluator reads as ``V2W_DATA``.
    ``public`` (:func:`public_digests` of the task's ``visible/``) leaves out files whose
    content is public there, recorded in ``public_files.json``."""
    data = data_dir()
    full = manifest()
    one = {**{k: v for k, v in full.items() if k != "samples"}, "configurations": 1,
           "samples": [dict(found.row)]}  # fmt: skip
    target.mkdir(parents=True)
    (target / "manifest.json").write_text(json.dumps(one, indent=1) + "\n", encoding="utf-8")
    shutil.copyfile(data / "paths.json", target / "paths.json")
    shutil.copytree(sample_dir, target / "samples" / found.sample)
    har = data / "har" / found.family / f"{found.sample}.json"
    (target / "har" / found.family).mkdir(parents=True)
    shutil.copyfile(har, target / "har" / found.family / har.name)
    if not public:
        return
    left_out: dict[str, str] = {}
    for path in sorted(target.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.stat().st_size > 0:
            seen = public.get(hashlib.sha256(path.read_bytes()).hexdigest())
            if seen is not None and path.name != "video.mp4":
                left_out[path.relative_to(target).as_posix()] = seen
                path.unlink()
    if left_out:
        (target / PUBLIC_FILES).write_text(json.dumps(left_out, indent=1) + "\n", encoding="utf-8")


def restore_public_files(release: Path, visible: Path) -> int:
    """Put back, in a copy of a task's data release, the files :func:`hidden_release` left out
    because they are public in the task's ``visible/``. Returns how many."""
    listing = release / PUBLIC_FILES
    if not listing.is_file():
        return 0
    entries = json.loads(listing.read_text(encoding="utf-8"))
    for relative, seen in entries.items():
        source = visible / seen
        if source.is_symlink() or not source.is_file():  # a link is never followed into it
            raise SetupError(
                f"the task's visible/{seen} is missing or a symbolic link (the data release "
                "needs the file)",
                fix="rebuild the task: r2s bench build video2world",
            )
        (release / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, release / relative)
    return len(entries)


def build_instance(name: str | Instance, out_dir: Path, *, overwrite: bool = False) -> Path:
    found = name if isinstance(name, Instance) else instance(name)
    _supported(found)
    checkout = CODE.ensure()
    sample_dir = download(found, handout_patterns(found, checkout))
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    video = sample_dir / "video.mp4"
    info = probe_video(video)
    if "resolution" not in meta or "T" not in meta:
        meta = {**meta, "resolution": [info.width, info.height], "T": info.num_frames}
    config = task_config(found)
    root = Path(out_dir) / found.family / found.sample
    builder = TaskBuilder(root, config, overwrite=overwrite)
    with tempfile.TemporaryDirectory(prefix="r2s-video2world-") as tmp:
        media = Path(tmp) / "media"
        (media / "cam").mkdir(parents=True)
        shutil.copyfile(video, media / "cam" / "rgb.mp4")
        capture = Capture(
            capture_id=config.id,
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
                        fps=info.fps,
                        num_frames=info.num_frames,
                        resolution=(info.width, info.height),
                    )
                )
            },
            text=TextInfo(task="Reconstruct the manipulation as a scene and an action stream."),
        )
        builder.capture(capture, media)
        builder.visible_tool("video2sim", checkout / "video2sim")
        staged = stage_handout(found, checkout, Path(tmp) / "handout")
        public = None
        if staged is not None:
            builder.visible_tool("handout", staged.parent)
            public = builder.visible_dir / "tools" / "handout" / "public_task.json"
        if found.profile in CLOTH_PROFILES:
            stage_cloth_contract(checkout, Path(tmp) / "cloth_contract")
            builder.visible_tool("cloth_contract", Path(tmp) / "cloth_contract")
        release = Path(tmp) / "v2w"
        hidden_release(found, sample_dir, release, public=public_digests(builder.visible_dir))
        for path in sorted(p for p in release.rglob("*") if p.is_file()):
            builder.hidden_file(f"v2w/{path.relative_to(release).as_posix()}", source=path)
        text = statement(checkout, found, meta, config.delivery.agent_time_budget_s, public)
    builder.instruction(text)
    return builder.finish().root


class Video2WorldBenchmark(BenchmarkAdapter):
    name: ClassVar[str] = "video2world"
    summary: ClassVar[str] = (
        "Video2World: reconstruct a robot or hand manipulation video as a scene and an action "
        "stream; Video2World's own evaluator executes the package and scores it."
    )
    license: ClassVar[str] = (
        "Apache-2.0 code; data under each source's terms (DROID CC-BY-4.0, real2sim-eval MIT, "
        "FurnitureBench none declared, DexYCB and HOI4D CC-BY-NC-4.0, HOT3D CC-BY-SA and "
        "CC-BY-NC-SA, OakInk2 CC-BY-SA-4.0, RoboDojo Apache-2.0, in-house Apache-2.0 with Genie "
        "Sim assets under the AgiBot World Community License), downloaded at build time"
    )
    source: ClassVar[str] = "https://github.com/AetherLabsAI/Video2World"
    protocols: ClassVar[tuple[str, ...]] = ("native",)
    status: ClassVar[str] = "partial"

    def task_names(self) -> list[str]:
        families = {s: ("push-t" if s.startswith("twin_pusht") else "droid-pick-place")
                    for s in DEFAULT_SAMPLES}  # fmt: skip
        return [f"video2world/{families[s]}/{s}" for s in DEFAULT_SAMPLES]

    def build(
        self,
        out_dir: Path,
        *,
        select: Sequence[str] | None = None,
        overwrite: bool = False,
        **options: Any,
    ) -> list[Path]:
        """``select`` names instances (``family/sample``), samples (in each family that has
        them) or families (all their samples); ``all`` selects all 222 instances."""
        if options:
            raise UnknownOptionError.from_names(
                f"benchmark {self.name!r}", options, self.options_taken()
            )
        chosen: dict[str, Instance] = {}
        if select:
            for name in select:
                found = list(instances().values()) if name == "all" else lookup(name)
                chosen.update((i.key, i) for i in found)
        else:
            for name in DEFAULT_SAMPLES:
                chosen.update((i.key, i) for i in lookup(name))
        return [build_instance(i, Path(out_dir), overwrite=overwrite) for i in chosen.values()]
