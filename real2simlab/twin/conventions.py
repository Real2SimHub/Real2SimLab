"""Delivery conventions: what a method hands in, how its program runs, where its output goes.

* ``r2s`` — Real2SimLab's own: ``solution/run.py``; run as
  ``python run.py`` with the ``r2s-twin/1.0`` run contract.
* ``build_sh_world`` — 4DCodeBench's: ``solution/build.sh`` writes ``/workspace/world``.
* ``blender_scene`` — PhysCodeGym's: ``solution/scene.py`` builds a Blender scene from an empty
  file and renders ``/app/outputs/frames/frame_0001.png`` ...; the image's runner
  (``/opt/r2s/blender_scene.py``) runs it headless and encodes ``render.mp4``.
* ``video2sim_package`` — Video2World's: ``solution/`` is a video2sim protocol package
  (``protocol.json``, ``actions.npy``, meshes, ``expected/obj_poses.npy``); it is data, not a
  program, and only the benchmark's own evaluator executes it (native protocol only).

Benchmark adapters pick a convention; agents and the runner take everything else from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from real2simlab.twin.model import Runtime


class DeliveryConvention(StrEnum):
    """What a method hands in and how its program is run."""

    R2S = "r2s"
    """``solution/run.py``; the run contract of ``r2s-twin/1.0``."""
    BUILD_SH_WORLD = "build_sh_world"
    """4DCodeBench's convention: ``solution/build.sh`` writes ``world/`` at fixed paths."""
    BLENDER_SCENE = "blender_scene"
    """PhysCodeGym's convention: ``solution/scene.py`` renders frames into ``/app/outputs``."""
    VIDEO2SIM_PACKAGE = "video2sim_package"
    """Video2World's convention: ``solution/`` is a video2sim protocol package; native only."""


@dataclass(frozen=True)
class ConventionSpec:
    entry_file: str
    """The file a solution must contain."""
    entrypoint: tuple[str, ...]
    runtime: Runtime
    session_output: str
    """Directory of the agent's workspace that holds its own output in its session."""
    agent_output_mount: str | None = None
    """An in-sandbox path the agent's session output is mounted at, when it is fixed."""
    runnable: bool = True
    """Whether the Lab can run the delivery as a program (frozen re-runs, ``r2s twin check``);
    a package of data is executed only by its benchmark's evaluator."""


CONVENTIONS: dict[DeliveryConvention, ConventionSpec] = {
    DeliveryConvention.R2S: ConventionSpec(
        entry_file="run.py",
        entrypoint=("python", "./run.py"),
        runtime=Runtime(),
        session_output="output",
    ),
    DeliveryConvention.BUILD_SH_WORLD: ConventionSpec(
        entry_file="build.sh",
        entrypoint=("bash", "./build.sh"),
        runtime=Runtime(
            program_path="/workspace/solution",
            output_path="/workspace/world",
            replaceable_output=True,  # "build.sh replaces anything already in world/"
        ),
        session_output="world",
    ),
    DeliveryConvention.BLENDER_SCENE: ConventionSpec(
        entry_file="scene.py",
        entrypoint=("python3", "/opt/r2s/blender_scene.py"),
        runtime=Runtime(program_path="/r2s/program", output_path="/app/outputs"),
        session_output="app_outputs",
        agent_output_mount="/app/outputs",
    ),
    DeliveryConvention.VIDEO2SIM_PACKAGE: ConventionSpec(
        entry_file="protocol.json",
        # What a run would do: the package's own validator. The Lab never runs it (runnable).
        entrypoint=("python", "-m", "video2sim.cli", "validate", "."),
        runtime=Runtime(),
        session_output="solution",
        runnable=False,
    ),
}


def convention_spec(convention: DeliveryConvention) -> ConventionSpec:
    return CONVENTIONS[convention]
