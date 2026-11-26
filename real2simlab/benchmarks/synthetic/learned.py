"""``nearest_twin``: a learned baseline for the synthetic phenomena, trained with ``r2s train fit``.

It sees only what any method sees: the video and the scene's objects (names and colours), so it
needs tasks that list them (the ``calibrated`` tier). It finds each moving object by its colour
in every frame. Trained on an exported dataset of synthetic training tasks, it keeps their
pixel tracks and their truth (``parameters.json``). For a task, it takes the phenomenon of the
nearest training task and predicts the hidden parameters and initial state as the
distance-weighted mean of its ``k`` nearest training tasks of that phenomenon (the phenomena
are filmed with fixed cameras, so pixel tracks are comparable). It then writes a program that
simulates the phenomenon with what it predicted. With ``prior=true`` it predicts the mean of
the training tasks instead, whatever the video shows: the same method without learning from
the video.

A minimal method that goes through the whole loop, not a strong one: it can only interpolate
between the training tasks it has seen.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import PositiveInt

from real2simlab.benchmarks.synthetic.physics import PHENOMENA
from real2simlab.benchmarks.synthetic.programs import simulation_program
from real2simlab.errors import R2SError, SetupError
from real2simlab.media import read_frames
from real2simlab.methods.base import (
    BuildContext,
    BuildOutcome,
    Method,
    MethodOptions,
    TrainedOptions,
    package_twin,
)
from real2simlab.methods.training import TrainContext
from real2simlab.sandbox.base import Backend
from real2simlab.task.io import Task
from real2simlab.twin import InvalidTwinError
from real2simlab.twin.conventions import DeliveryConvention

MODEL_FILE = "nearest_twin.npz"
COLORS = {
    "red": (0.85, 0.16, 0.10),
    "blue": (0.13, 0.35, 0.85),
    "orange": (0.95, 0.55, 0.10),
}
"""Reference colours of the scene objects' colour names."""


# --- observation ----------------------------------------------------------------------------


def track(video: Path, objects: list[dict[str, Any]]) -> NDArray[np.float64]:
    """``(objects, frames, 2)`` pixel centroids of each named moving object, found by its
    colour; frames where it is not seen are interpolated."""
    frames = read_frames(video).astype(np.float64)
    chroma = frames / (frames.sum(axis=3, keepdims=True) + 1e-6)
    saturated = (frames.max(axis=3) - frames.min(axis=3)) > 50
    tracks: list[NDArray[np.float64]] = []
    for obj in objects:
        reference = np.asarray(COLORS[str(obj["color"])])
        mask = saturated & (np.linalg.norm(chroma - reference / reference.sum(), axis=3) < 0.08)
        centres = np.full((len(frames), 2), np.nan)
        for index, frame_mask in enumerate(mask):
            v, u = np.nonzero(frame_mask)
            if len(u) >= 3:
                centres[index] = (u.mean(), v.mean())
        tracks.append(_fill(centres))
    return np.stack(tracks)


def _fill(centres: NDArray[np.float64]) -> NDArray[np.float64]:
    seen = ~np.isnan(centres[:, 0])
    if not seen.any():
        return np.zeros_like(centres)
    index = np.arange(len(centres))
    return np.stack([np.interp(index, index[seen], centres[seen, i]) for i in range(2)], axis=1)


def observed(capture_file: Path) -> tuple[str, NDArray[np.float64]] | None:
    """A capture's key (its moving objects and frame count) and its pixel tracks; ``None`` if
    the capture does not list its objects or their colours."""
    capture = json.loads(capture_file.read_text(encoding="utf-8"))
    if "scene" not in capture:
        return None
    view = next(iter(capture["views"].values()))
    objects = sorted(
        (o for o in capture["scene"]["objects"] if o.get("role") == "dynamic"),
        key=lambda o: o["name"],
    )
    if not objects or any(o.get("color") not in COLORS for o in objects):
        return None
    tracks = track(capture_file.parent / view["rgb"]["path"], objects)
    key = "+".join(o["name"] for o in objects) + f"@{view['rgb']['num_frames']}"
    return key, tracks


# --- the numeric leaves of parameters.json ---------------------------------------------------


def leaves(data: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(data, dict):
        out: dict[str, float] = {}
        for key, value in data.items():  # pyright: ignore[reportUnknownVariableType]
            out.update(leaves(value, f"{prefix}{key}."))
        return out
    if isinstance(data, list):
        out = {}
        for index, value in enumerate(data):  # pyright: ignore[reportUnknownVariableType]
            out.update(leaves(value, f"{prefix}{index}."))
        return out
    return {prefix[:-1]: float(data)} if isinstance(data, int | float) else {}


def with_leaves(template: Any, values: dict[str, float], prefix: str = "") -> Any:
    if isinstance(template, dict):
        return {
            k: with_leaves(v, values, f"{prefix}{k}.")
            for k, v in template.items()  # pyright: ignore[reportUnknownVariableType]
        }
    if isinstance(template, list):
        return [
            with_leaves(v, values, f"{prefix}{i}.")
            for i, v in enumerate(template)  # pyright: ignore[reportUnknownVariableType]
        ]
    return values.get(prefix[:-1], template)


def simulated(labels: dict[str, Any]) -> bool:
    """Whether a training task's labels (its ``parameters``) are of a phenomenon nearest_twin
    simulates: labels of another kind (another benchmark's) are not learned from."""
    phenomenon = labels.get("phenomenon")
    return isinstance(phenomenon, str) and phenomenon in PHENOMENA


def truth_problem(labels: Any) -> str | None:
    """Why ``labels`` (a training task's ``parameters``) are not the truth of a phenomenon
    nearest_twin simulates, ``None`` when they are: a JSON object naming the phenomenon, with
    its ``params`` and its ``setup`` (what the program it writes is given)."""
    if not isinstance(labels, dict):
        return "they are not a JSON object"
    named = cast(dict[str, Any], labels)
    if not simulated(named):
        return f"they name no phenomenon it simulates ({named.get('phenomenon')!r})"
    if not (isinstance(named.get("params"), dict) and isinstance(named.get("setup"), dict)):
        return f"the labels of a {named['phenomenon']} task have no params and setup objects"
    return None


# --- the model -------------------------------------------------------------------------------


class Model:
    """The training tasks' keys, pixel tracks and truth."""

    def __init__(self, keys: list[str], tracks: list[NDArray[np.float64]],
                 truths: list[dict[str, Any]]) -> None:  # fmt: skip
        self.keys, self.tracks, self.truths = keys, tracks, truths

    def save(self, path: Path) -> None:
        arrays: dict[str, Any] = {f"tracks_{i}": t for i, t in enumerate(self.tracks)}
        arrays["keys"] = np.array(self.keys)
        arrays["truths"] = np.array([json.dumps(t, sort_keys=True) for t in self.truths])
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> Model:
        try:
            with np.load(path) as data:
                keys = [str(k) for k in data["keys"]]
                truths = [json.loads(str(t)) for t in data["truths"]]
                tracks = [data[f"tracks_{i}"] for i in range(len(keys))]
            problems = [problem for problem in map(truth_problem, truths) if problem]
            if problems:  # training learns only from truths it can simulate
                raise ValueError(f"a training truth is not one it can simulate: {problems[0]}")
        except Exception as error:  # whatever is wrong with the file, it is not this model
            raise SetupError(
                f"{path} is not a nearest_twin model: {type(error).__name__}: {error}",
                fix="train it again: r2s train fit nearest_twin --data DATASET --out "
                f"{path.parent} --overwrite",
            ) from error
        return cls(keys, tracks, truths)

    def predict(self, key: str, tracks: NDArray[np.float64], k: int, prior: bool) -> dict[str, Any]:
        """The predicted ``parameters.json`` of an observation: the phenomenon of the nearest
        training task, then the distance-weighted mean of the ``k`` nearest of that phenomenon
        (or, with ``prior``, the mean of all of them)."""
        rows = [i for i, other in enumerate(self.keys) if other == key]
        if not rows:
            raise SetupError(
                f"nearest_twin saw no training task like this one ({key})",
                fix="train it on training tasks of this phenomenon and tier",
            )
        truths = [self.truths[i] for i in rows]
        distances = np.array(
            [float(np.linalg.norm(self.tracks[i] - tracks, axis=2).mean()) for i in rows]
        )
        phenomenon = truths[int(np.argmin(distances))]["phenomenon"]
        same = [j for j, truth in enumerate(truths) if truth["phenomenon"] == phenomenon]
        if prior:
            chosen = [(1.0, truths[j]) for j in same]
        else:
            nearest = sorted(same, key=lambda j: distances[j])[:k]
            chosen = [(1.0 / (distances[j] + 1e-3), truths[j]) for j in nearest]
        total = sum(weight for weight, _ in chosen)
        names = set[str]().union(*(leaves(truth) for _, truth in chosen))
        mean = {
            name: sum(weight * leaves(truth).get(name, 0.0) for weight, truth in chosen) / total
            for name in names
        }
        return with_leaves(chosen[0][1], mean)


@cache
def _model(path: Path, digest: str) -> Model:
    """The model at ``path`` of the checkpoint whose content fingerprint is ``digest`` (a
    checkpoint trained again is read again)."""
    return Model.load(path)


# --- the method ------------------------------------------------------------------------------


class NearestTwinOptions(TrainedOptions):
    k: PositiveInt = 5
    """Neighbours averaged."""
    prior: bool = False
    """Predict the mean of the training tasks, ignoring the video."""
    count: PositiveInt | None = None
    """Training: learn only from the first ``count`` training tasks of each phenomenon and
    tier (by sample index)."""


class NearestTwin(Method):
    name: ClassVar[str] = "nearest_twin"
    kind: ClassVar[str] = "baseline"
    summary: ClassVar[str] = (
        "a learned baseline for the synthetic phenomena: the hidden parameters of the nearest "
        "training tasks (r2s train fit)"
    )
    trainable: ClassVar[bool] = True
    Options: ClassVar[type[MethodOptions]] = NearestTwinOptions

    @property
    def nearest_options(self) -> NearestTwinOptions:
        assert isinstance(self.options, NearestTwinOptions)
        return self.options

    def problems(self, backend: Backend) -> list[str]:
        if self.checkpoint is None:
            return [
                "nearest_twin needs checkpoint=DIR: train it first "
                "(r2s train fit nearest_twin --data DATASET --out DIR)"
            ]
        try:  # read before anything runs: a broken model stops the evaluation, not one build
            _model(self.checkpoint / MODEL_FILE, str(self.checkpoint_digest()))
        except SetupError as error:
            return [str(error)]  # its message ends with the fix
        return []

    def applicable(self, task: Task) -> str | None:
        delivery = task.config.delivery
        if delivery.convention is not DeliveryConvention.R2S or delivery.output_format != "tracks":
            return "nearest_twin writes r2s programs with tracks output"
        if not task.has_capture() or "scene" not in json.loads(
            (task.visible_dir / "capture.json").read_text(encoding="utf-8")
        ):
            return "nearest_twin needs a capture that lists the scene's objects (calibrated tier)"
        return None

    def train(self, context: TrainContext) -> None:
        count = self.nearest_options.count
        keys: list[str] = []
        tracks: list[NDArray[np.float64]] = []
        truths: list[dict[str, Any]] = []
        path = context.dataset / "tasks.jsonl"
        rows: list[dict[str, Any]] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            data: Any = json.loads(line) if line.strip() else None
            if not (isinstance(data, dict) and "parameters" in data and "capture" in data):
                continue
            row = cast(dict[str, Any], data)
            labels: Any = row["parameters"]
            if isinstance(labels, dict) and not simulated(cast(dict[str, Any], labels)):
                continue  # labels of another kind: nothing it can simulate
            problem = truth_problem(labels)
            if problem is not None:  # labels an export took as they were: say which
                raise SetupError(
                    f"{path}, line {number}: the task's labels (its hidden/parameters.json, "
                    f"exported as parameters) are not a truth nearest_twin can learn: {problem}",
                    fix="correct that task's hidden/parameters.json (r2s task validate checks "
                    "it), then export the dataset again: r2s train export ... --out DATASET "
                    "--overwrite",
                )
            rows.append(row)
        if count is not None:  # the first count tasks of each phenomenon and tier, by index
            groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for row in rows:
                group = (str(row["parameters"].get("phenomenon")), str(row.get("visibility")))
                groups.setdefault(group, []).append(row)
            rows = [
                row
                for group in groups.values()
                for row in sorted(group, key=lambda r: int(r["metadata"].get("index", 0)))[:count]
            ]
        for row in rows:
            seen = observed(context.dataset / row["capture"])
            if seen is None:
                continue
            keys.append(seen[0])
            tracks.append(seen[1])
            truths.append(row["parameters"])
        if not keys:
            raise SetupError(
                f"{context.dataset} has no training task nearest_twin can learn from",
                fix="export synthetic training tasks of the calibrated tier "
                "(r2s train tasks --tier calibrated)",
            )
        Model(keys, tracks, truths).save(context.checkpoint_dir / MODEL_FILE)
        context.log(f"nearest_twin learned from {len(keys)} training tasks")

    def build(self, context: BuildContext) -> BuildOutcome:
        checkpoint = self.checkpoint
        if checkpoint is None:
            return BuildOutcome(None, "nearest_twin was given no checkpoint (checkpoint=DIR)")
        seen = observed(context.visible_dir / "capture.json")
        if seen is None:
            return BuildOutcome(None, "the capture does not list the scene's objects")
        options = self.nearest_options
        model = _model(checkpoint / MODEL_FILE, str(self.checkpoint_digest()))
        try:  # no training task like this one: the method cannot answer, a result
            guess = model.predict(seen[0], seen[1], options.k, options.prior)
        except SetupError as error:
            return BuildOutcome(None, str(error))
        module = PHENOMENA[guess["phenomenon"]]
        program = simulation_program(
            context.work_dir / "program",
            phenomenon=module.NAME,
            title=module.TITLE,
            params=guess["params"],
            setup=guess["setup"],
        )
        try:
            twin = package_twin(context, program, method=self.name, notes=self.summary)
        except (InvalidTwinError, R2SError, ValueError) as error:
            return BuildOutcome(None, f"the program is not a valid twin: {error}")
        return BuildOutcome(twin)
