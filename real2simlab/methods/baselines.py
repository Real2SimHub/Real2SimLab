"""Reference baselines: ``static``, ``constant_velocity`` and ``oracle``.

They are *privileged*: they read the task's reference (or, for ``oracle``, its truth
generator) instead of estimating anything from the video, so they bracket what a twin can
score. Reports label them as privileged.

* ``static``: every object stays where it starts; its reproduction skill is 0.
* ``constant_velocity``: extrapolates each object's initial velocity in a straight line.
* ``oracle``: the task's truth generator (synthetic tasks only); its skill is 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from real2simlab._fs import copy_tree
from real2simlab.formats.base import get_format
from real2simlab.formats.tracks import ObjectTrack, Tracks
from real2simlab.methods.base import BuildContext, BuildOutcome, Method, MethodOptions, package_twin
from real2simlab.methods.programs import fixed_tracks_program
from real2simlab.task.io import GENERATOR_DIR, REFERENCE_DIR
from real2simlab.twin import InvalidTwinError, load_twin
from real2simlab.twin.conventions import DeliveryConvention

if TYPE_CHECKING:
    from real2simlab.task.io import Task


class _ReferenceBaseline(Method):
    kind: ClassVar[str] = "baseline"
    privileged: ClassVar[bool] = True

    def reference(self, context: BuildContext) -> Tracks | None:
        if context.hidden_dir is None:
            return None
        reference = context.hidden_dir / REFERENCE_DIR
        if not reference.is_dir():
            return None
        return get_format(context.config.delivery.output_format).to_tracks(reference)

    def transform(self, reference: Tracks, context: BuildContext) -> Tracks:
        raise NotImplementedError

    def applicable(self, task: Task) -> str | None:
        delivery = task.config.delivery
        if delivery.convention is not DeliveryConvention.R2S or delivery.output_format != "tracks":
            return (
                f"{self.name} writes r2s programs with tracks output; this task needs a "
                f"{delivery.convention.value} delivery with {delivery.output_format} output"
            )
        reference = task.hidden_dir / REFERENCE_DIR
        if not reference.is_dir():
            return f"{self.name} needs the task's reference; this task has none"
        return None

    def build(self, context: BuildContext) -> BuildOutcome:
        reference = self.reference(context)
        if reference is None or reference.camera is None:
            return BuildOutcome(None, f"{self.name} needs a reference in tracks form with a camera")
        program = fixed_tracks_program(
            context.work_dir / "program", self.transform(reference, context), summary=self.summary
        )
        try:
            twin = package_twin(
                context,
                program,
                method=self.name,
                privileged=True,
                notes=self.summary,
            )
        except InvalidTwinError as error:
            return BuildOutcome(None, str(error))
        return BuildOutcome(twin)


class StaticBaseline(_ReferenceBaseline):
    name: ClassVar[str] = "static"
    summary: ClassVar[str] = "every object stays at its first observed position"

    def transform(self, reference: Tracks, context: BuildContext) -> Tracks:
        from real2simlab.scoring.reproduction import static_tracks

        return static_tracks(reference)


class ConstantVelocityOptions(MethodOptions):
    frames: int = 3
    """Frames from the start used to estimate each object's velocity."""


class ConstantVelocityBaseline(_ReferenceBaseline):
    name: ClassVar[str] = "constant_velocity"
    summary: ClassVar[str] = "extrapolates each object's initial velocity in a straight line"
    Options: ClassVar[type[MethodOptions]] = ConstantVelocityOptions

    def transform(self, reference: Tracks, context: BuildContext) -> Tracks:
        assert isinstance(self.options, ConstantVelocityOptions)
        count = reference.num_frames
        fit = min(max(2, self.options.frames), count)
        times = reference.times()
        objects: dict[str, ObjectTrack] = {}
        for name in reference.objects:
            position = reference.positions(name)
            rows = np.flatnonzero(np.isfinite(position[:fit]).all(axis=1))
            if rows.size >= 2:
                slope = np.polyfit(times[rows], position[rows], 1)
                velocity, start = slope[0], slope[1]
            else:
                velocity, start = np.zeros(3), np.nan_to_num(position[0])
            objects[name] = ObjectTrack(position=start + times[:, None] * velocity)
        return Tracks(fps=reference.fps, objects=objects, camera=reference.camera)


class OracleBaseline(Method):
    name: ClassVar[str] = "oracle"
    kind: ClassVar[str] = "baseline"
    summary: ClassVar[str] = "the task's own truth generator (synthetic tasks only)"
    privileged: ClassVar[bool] = True

    def applicable(self, task: Task) -> str | None:
        if not (task.hidden_dir / GENERATOR_DIR).is_dir():
            return "the oracle copies the task's truth generator; this task has none"
        return None

    def build(self, context: BuildContext) -> BuildOutcome:
        generator = None if context.hidden_dir is None else context.hidden_dir / GENERATOR_DIR
        if generator is None or not generator.is_dir():
            return BuildOutcome(None, "the task has no truth generator")
        try:
            source = load_twin(generator)
        except InvalidTwinError as error:
            return BuildOutcome(None, f"the task's generator is invalid: {error}")
        program = context.work_dir / "program"
        copy_tree(source.program_dir, program)
        try:
            twin = package_twin(
                context,
                program,
                method=self.name,
                entrypoint=source.manifest.entrypoint,
                privileged=True,
                notes=self.summary,
            )
        except InvalidTwinError as error:
            return BuildOutcome(None, str(error))
        return BuildOutcome(twin)
