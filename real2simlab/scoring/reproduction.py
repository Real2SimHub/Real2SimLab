"""The reproduction protocol, version 1.0, and the task result every protocol writes.

The frozen twin is re-run offline and its output is compared with the task's reference by the
task's metrics. The skill of the primary metric is ``1 - error / static error``, where the
static baseline holds every object where it starts: 1 is a perfect reproduction, 0 is no
better than a scene where nothing moves. When the task has other recordings of the same event
(``hidden/repeats/``), their mean error is the noise floor, and the skill such a recording
scores is reported as a reference point.

Tasks under the native protocol are scored by their benchmark's official scorer; this module
still writes their result record, with the delivery's re-run checks.
"""

from __future__ import annotations

import filecmp
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from real2simlab import __version__
from real2simlab._fs import is_regular
from real2simlab.errors import SetupError
from real2simlab.formats.base import get_format, tree_problems
from real2simlab.formats.tracks import ObjectTrack, Tracks
from real2simlab.metrics.base import Metric, MetricStatus, MetricValue, get_metric, robust_mean
from real2simlab.scoring.model import (
    Headline,
    MetricRecord,
    Reproduction,
    ReproductionStatus,
    TaskResult,
    TaskSummary,
    WindowScore,
)
from real2simlab.task.io import Task

PROTOCOL_VERSION = "reproduction/1.0"
INFRA_ERROR = "infra_error"
"""The run status of a run that failed outside the program (``RunStatus.INFRA_ERROR``)."""
ZERO = 1e-12
VIDEO_BLOCK = 8
VIDEO_TOLERANCE = 10.0
"""Two videos agree when, after averaging 8 x 8-pixel blocks, no block of any frame differs by more
than 10 (of 255): a renderer's sampling noise stays well under it (Blender: about 6), while a
one-pixel shift of a high-contrast edge, or a frame of different motion, exceeds it."""


@dataclass(frozen=True)
class RunOutput:
    """A twin run as scoring sees it: its status and, when it succeeded, its output."""

    status: str
    output_dir: Path | None = None
    problem: str | None = None
    """Why it failed outside the program (its record's problem), when it did."""

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded" and self.output_dir is not None

    @property
    def infra_failed(self) -> bool:
        """The run failed outside the program: it says nothing about the program."""
        return self.status == INFRA_ERROR

    def infra_reason(self) -> str:
        """Why the result of a run that failed outside the program cannot be scored."""
        return "the sandbox failed around the program (run status infra_error)" + (
            f": {self.problem}" if self.problem else ""
        )


@dataclass(frozen=True)
class MethodInfo:
    name: str
    kind: str
    privileged: bool = False


def _record(value: MetricValue) -> MetricRecord:
    return MetricRecord(status=value.status.value, value=value.value, detail=value.detail)


def _failed(detail: str, metrics: Sequence[str]) -> dict[str, MetricRecord]:
    return {name: MetricRecord(status="error", detail=detail) for name in metrics}


def static_tracks(reference: Tracks) -> Tracks:
    """Every object held at its first visible reference position."""
    objects: dict[str, ObjectTrack] = {}
    for name in reference.objects:
        position = reference.positions(name)
        visible = np.flatnonzero(np.isfinite(position).all(axis=1))
        anchor = visible[0] if visible.size else 0
        held = np.repeat(position[anchor][None, :], reference.num_frames, axis=0)
        objects[name] = ObjectTrack(position=held)
    return Tracks(fps=reference.fps, objects=objects, camera=reference.camera)


def max_difference(a: Tracks, b: Tracks) -> float:
    """Largest position difference between two tracks; inf if they are not comparable."""
    if set(a.objects) != set(b.objects) or a.num_frames != b.num_frames:
        return float("inf")
    largest = 0.0
    for name in a.objects:
        pa, pb = a.positions(name), b.positions(name)
        if not np.array_equal(np.isnan(pa), np.isnan(pb)):
            return float("inf")
        if pa.size:
            largest = max(largest, float(np.nanmax(np.abs(pa - pb), initial=0.0)))
    return largest


def _block_means(frames: np.ndarray) -> np.ndarray:
    count, height, width, channels = frames.shape
    k = VIDEO_BLOCK
    cropped = frames[:, : height // k * k, : width // k * k].astype(np.float64)
    return cropped.reshape(count, height // k, k, width // k, k, channels).mean(axis=(2, 4))


def outputs_agree(fmt: Any, first: Path, second: Path, *, tolerance: float = 0.0) -> bool:
    """Whether two outputs of a twin agree, over everything the format holds.

    The outputs must have the same files among the format's content (``OutputFormat.content``;
    every file when a format does not restrict it, or when neither output has any of it).
    Numbers in ``.npz``, ``.npy`` and ``.json`` files agree to within ``tolerance`` (and NaN
    where the other is NaN); videos (``.mp4``) agree frame by frame up to rendering noise
    (:data:`VIDEO_TOLERANCE`); any other file must be identical. So a 4D world is compared
    point by point and mesh by mesh, not only through the object centroids its tracks keep,
    and a ``tracks`` output through its camera, rotations and visibility too.
    """
    files_a, files_b = _content(fmt, Path(first)), _content(fmt, Path(second))
    if not files_a and not files_b:  # nothing of the format in either: compare what is there
        files_a, files_b = _content(None, Path(first)), _content(None, Path(second))
    if files_a != files_b:
        return False
    return all(_files_agree(Path(first) / name, Path(second) / name, tolerance) for name in files_a)


def session_readable(directory: Path) -> bool:
    """Whether the Lab may read ``directory``, a method's own session output, as it is: a
    directory, not a symbolic link to one, of regular files it can read, without symbolic
    links, pipes, sockets or devices (:func:`~real2simlab.formats.base.tree_problems`). Any
    other is not used, as if the method had left none: following a link would read on the host
    what the method's sandbox kept from it (a task's hidden data), and a pipe would wait
    forever."""
    return not tree_problems(directory, unreadable=True)


def _session_agrees(fmt: Any, session: Path, output: Path) -> bool:
    """Whether a method's own session output agrees with its frozen run's ``output``. The
    session output is the method's: one that cannot be read as it is compared agrees with
    nothing; it never stops the scoring."""
    try:
        return outputs_agree(fmt, session, output, tolerance=1e-6)
    except SetupError:
        raise  # this machine cannot compare outputs of the kind: the fix, not an answer
    except Exception:
        return False


def _content(fmt: Any, root: Path) -> set[str]:
    patterns: tuple[str, ...] | None = getattr(fmt, "content", None)
    if root.is_symlink() or not root.is_dir():
        return set()
    found = root.rglob("*") if patterns is None else (p for g in patterns for p in root.glob(g))
    return {p.relative_to(root).as_posix() for p in found if is_regular(p)}


def _files_agree(a: Path, b: Path, tolerance: float) -> bool:
    """Whether two files agree (see :func:`outputs_agree`). Outputs are programs' (a session
    output is the agent's own): a file that cannot be read as its kind (an empty or truncated
    archive, JSON nested too deep) is compared byte for byte, and one that cannot be read at
    all agrees with nothing."""
    suffix = a.suffix.lower()
    try:
        if suffix in (".npz", ".npy"):
            return _arrays_agree(_load_arrays(a), _load_arrays(b), tolerance)
        if suffix == ".json":
            return _json_agree(
                json.loads(a.read_text(encoding="utf-8")),
                json.loads(b.read_text(encoding="utf-8")),
                tolerance,
            )
        if suffix == ".mp4":
            return _videos_agree(a, b)
    except SetupError:
        raise  # this machine cannot read the kind (no ffmpeg): no answer, the fix instead
    except Exception:
        pass  # not readable as its kind: compare the bytes
    try:
        return filecmp.cmp(a, b, shallow=False)
    except OSError:
        return False


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    loaded: Any = np.load(path, allow_pickle=False)
    if not hasattr(loaded, "files"):  # a .npy file: one array
        return {"": np.asarray(loaded)}
    try:
        return {str(key): np.asarray(loaded[key]) for key in loaded.files}
    finally:
        loaded.close()


def _arrays_agree(a: dict[str, np.ndarray], b: dict[str, np.ndarray], tolerance: float) -> bool:
    if set(a) != set(b):
        return False
    for key, x in a.items():
        y = b[key]
        if x.shape != y.shape:
            return False
        numeric = x.dtype.kind in "iuf" and y.dtype.kind in "iuf"
        if not numeric:
            if not np.array_equal(x, y):
                return False
            continue
        x, y = x.astype(np.float64), y.astype(np.float64)
        missing = np.isnan(x)
        if not np.array_equal(missing, np.isnan(y)):
            return False
        u, v = x[~missing], y[~missing]
        # inf - inf: equal infinities pass by ==; a difference beyond the largest float: inf
        with np.errstate(invalid="ignore", over="ignore"):
            if not np.all((u == v) | (np.abs(u - v) <= tolerance)):
                return False
    return True


def _json_agree(a: Any, b: Any, tolerance: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b or abs(a - b) <= tolerance
    if isinstance(a, dict) and isinstance(b, dict):
        da, db = cast(dict[str, Any], a), cast(dict[str, Any], b)
        return set(da) == set(db) and all(_json_agree(da[k], db[k], tolerance) for k in da)
    if isinstance(a, list) and isinstance(b, list):
        la, lb = cast(list[Any], a), cast(list[Any], b)
        if len(la) != len(lb):
            return False
        return all(_json_agree(x, y, tolerance) for x, y in zip(la, lb, strict=True))
    return a == b


def _videos_agree(a: Path, b: Path) -> bool:
    from real2simlab.media import MediaError, read_frames

    try:
        frames_a, frames_b = read_frames(a), read_frames(b)
    except MediaError:
        return filecmp.cmp(a, b, shallow=False)
    if frames_a.shape != frames_b.shape:
        return False
    if np.array_equal(frames_a, frames_b):
        return True
    difference = np.abs(_block_means(frames_a) - _block_means(frames_b))
    return bool(difference.max() <= VIDEO_TOLERANCE)


def benchmark_of(task_id: str) -> str:
    """The benchmark a task comes from: the first segment of its id."""
    return task_id.split("/", 1)[0]


class Scorer:
    """Scores one task: loads its reference once, then scores any number of twins."""

    def __init__(self, task: Task) -> None:
        self.task = task
        config = task.config
        self.scoring = config.scoring
        self.metric_names = tuple(self.scoring.metrics)
        self.metrics: dict[str, Metric] = {name: get_metric(name) for name in self.metric_names}
        problems = [
            problem
            for name, metric in self.metrics.items()
            for problem in metric.option_problems(self.scoring.metric_options.get(name) or {})
        ]
        if problems:  # a broken scoring configuration is not a score of 0
            raise SetupError(
                f"task {task.id} cannot be scored: {'; '.join(problems)}",
                fix=f"correct scoring.metric_options in {task.root / 'task.toml'} "
                f"(r2s task validate {task.root} lists every problem)",
            )
        self.primary = self.scoring.primary
        self.format = get_format(config.delivery.output_format)
        self._reference: Tracks | None = None
        self._loaded = False
        self._noise: float | None = None
        self._noise_done = False

    @property
    def reference(self) -> Tracks | None:
        if not self._loaded:
            self._reference = self.tracks(self.task.reference_dir())
            self._loaded = True
        return self._reference

    def tracks(self, directory: Path | None) -> Tracks | None:
        if directory is None or not directory.is_dir():
            return None
        return self.format.to_tracks(directory)

    def _evaluate(self, prediction: Tracks | None, reference: Tracks) -> dict[str, MetricRecord]:
        if prediction is None:
            return _failed("the format of the output has no tracks", self.metric_names)
        return {
            name: _record(
                metric.compute(prediction, reference, options=self.scoring.metric_options.get(name))
            )
            for name, metric in self.metrics.items()
        }

    def _primary_value(self, records: Mapping[str, MetricRecord]) -> float | None:
        if self.primary is None:
            return None
        record = records.get(self.primary)
        return record.value if record is not None and record.status == "computed" else None

    @staticmethod
    def _skill(error: float | None, baseline: float | None) -> tuple[float | None, float]:
        if error is None or baseline is None:
            return None, 0.0
        if not (math.isfinite(error) and math.isfinite(baseline)):  # not numbers to compare
            return None, 0.0
        if baseline <= ZERO:
            return None, 1.0 if error <= ZERO else 0.0
        ratio = error / baseline
        if not math.isfinite(ratio):  # worse than the baseline beyond what a float holds
            return None, 0.0
        raw = 1.0 - ratio
        return raw, float(min(1.0, max(0.0, raw)))

    def noise_floor(self) -> float | None:
        """Mean primary-metric error of the task's other recordings against the reference."""
        if self._noise_done:
            return self._noise
        reference = self.reference
        values: list[float] = []
        if reference is not None and self.primary is not None:
            metric = self.metrics[self.primary]
            for repeat in self.task.repeat_dirs():
                tracks = self.tracks(repeat)
                if tracks is None:
                    continue
                value = metric.compute(
                    tracks, reference, options=self.scoring.metric_options.get(self.primary)
                )
                if value.status is MetricStatus.COMPUTED and value.value is not None:
                    values.append(value.value)
        self._noise = robust_mean(np.asarray(values)) if values else None
        self._noise_done = True
        return self._noise

    def static_error(self) -> float | None:
        reference = self.reference
        if reference is None:
            return None
        return self._primary_value(self._evaluate(static_tracks(reference), reference))

    def reproduction(self, run: RunOutput) -> tuple[Reproduction, Tracks | None]:
        reference = self.reference
        if not run.succeeded:
            return Reproduction(status=ReproductionStatus.RUN_FAILED, run_status=run.status), None
        twin = self.tracks(run.output_dir)
        if reference is None and self.scoring.protocol.value == "reproduction":
            raise SetupError(  # a damaged task says nothing about the program
                f"task {self.task.id} has no reference output to score against",
                fix=f"restore {self.task.reference_dir()} (r2s task validate {self.task.root} "
                "lists what is missing)",
            )
        if reference is None:
            detail = "the task has no reference"
            return Reproduction(
                status=ReproductionStatus.SCORED, run_status=run.status, detail=detail
            ), twin
        errors = self._evaluate(twin, reference)
        static = self._evaluate(static_tracks(reference), reference)
        raw, clipped = self._skill(self._primary_value(errors), self._primary_value(static))
        score = WindowScore(
            frames=(0, reference.num_frames),
            errors=errors,
            static_errors=static,
            skill_raw=raw,
            skill=clipped,
        )
        noise = self.noise_floor()
        error = self._primary_value(errors)
        _, repeat_skill = self._skill(noise, self._primary_value(static))
        to_noise = error / noise if error is not None and noise else None
        return (
            Reproduction(
                status=ReproductionStatus.SCORED,
                run_status=run.status,
                score=score,
                noise_floor=noise,
                # Far above the noise floor beyond what a float holds: no ratio, not Infinity.
                error_to_noise=to_noise if to_noise is None or math.isfinite(to_noise) else None,
                repeat_skill=repeat_skill if noise is not None else None,
            ),
            twin,
        )

    def score(
        self,
        *,
        method: MethodInfo,
        trial: int,
        seed: int,
        delivered: bool,
        delivery_failure: str | None = None,
        twin_fingerprint: str | None = None,
        run: RunOutput | None = None,
        reproducible: bool | None = None,
        session_output: Path | None = None,
        infra_error: str | None = None,
    ) -> TaskResult:
        """The task result of one twin (or of a method that delivered none).

        ``infra_error`` (or a run whose status is ``infra_error``) makes the result one the Lab
        could not score: it carries no skill and no headline value, so no statistic or reward
        counts it."""
        protocol = self.scoring.protocol.value
        if infra_error is None and delivered and run is not None and run.infra_failed:
            infra_error = run.infra_reason()
        if not delivered or run is None:
            reproduction = Reproduction(
                status=ReproductionStatus.NOT_DELIVERED, detail=delivery_failure
            )
        else:
            reproduction, _ = self.reproduction(run)
            consistent = None
            if session_output is not None and run.succeeded and session_readable(session_output):
                assert run.output_dir is not None
                consistent = _session_agrees(self.format, session_output, run.output_dir)
            reproduction = reproduction.model_copy(
                update={"reproducible": reproducible, "session_consistent": consistent}
            )
        has_reference = protocol != "native" and self.reference is not None
        summary = summarize(reproduction, self.primary, has_reference=has_reference)
        if infra_error is not None:
            summary = TaskSummary(reproduction_skill=None, reproduction_error=None)
        unit = self.metrics[self.primary].unit if self.primary in self.metrics else None
        headline = None
        if has_reference:
            references = {"static": 0.0}
            if reproduction.repeat_skill is not None:
                references["another recording"] = reproduction.repeat_skill
            headline = Headline(
                metric="reproduction_skill",
                value=summary.reproduction_skill,
                range=(0.0, 1.0),
                references=references,
                source="lab",
            )
        return TaskResult(
            task_id=self.task.id,
            task_version=self.task.config.version,
            task_hash=self.task.content_hash(),
            benchmark=benchmark_of(self.task.id),
            method=method.name,
            method_kind=method.kind,
            privileged=method.privileged,
            trial=trial,
            seed=seed,
            protocol=protocol,
            protocol_version=PROTOCOL_VERSION,
            primary_metric=self.primary,
            metric_unit=unit,
            delivered=delivered,
            delivery_failure=delivery_failure,
            twin_fingerprint=twin_fingerprint,
            reproduction=reproduction,
            summary=summary,
            headline=headline,
            created_at=datetime.now(UTC),
            lab_version=__version__,
            infra_error=infra_error,
        )


def summarize(
    reproduction: Reproduction, primary: str | None, *, has_reference: bool = True
) -> TaskSummary:
    """The reproduction skill and error of one task result.

    The skill is 0 when the twin was not delivered or its run failed, and ``None`` when the task
    has no reference to compare with (as the native tasks of benchmarks that score only their
    own metrics)."""
    score = reproduction.score
    skill = score.skill if score is not None else (0.0 if has_reference else None)
    error = None
    if score is not None and primary is not None and primary in score.errors:
        record = score.errors[primary]
        error = record.value if record.status == "computed" else None
    return TaskSummary(reproduction_skill=skill, reproduction_error=error)
