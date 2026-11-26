"""Metrics, the reproduction protocol, headlines, the scorecard and aggregation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from real2simlab.metrics import (
    MetricStatus,
    bounce_times,
    get_metric,
    metric_names,
    similarity_transform,
)
from real2simlab.scoring import (
    Headline,
    Reproduction,
    ReproductionStatus,
    TaskResult,
    TaskSummary,
    benchmark_scores,
    bootstrap,
    max_difference,
    static_tracks,
    summarize,
)
from real2simlab.scoring.aggregate import bootstrap_statistic

from .conftest import falling, make_tracks

ROOT = Path(__file__).resolve().parents[1]


def test_registered_metrics() -> None:
    assert set(metric_names()) >= {
        "traj3d_mean", "traj3d_rmse", "traj3d_final", "traj2d_px_mean", "traj2d_norm_mean",
        "event_first_contact", "event_bounce_count", "event_bounce_times",
    }  # fmt: skip


def test_trajectory_errors() -> None:
    reference = make_tracks({"ball": falling()})
    shifted = make_tracks({"ball": falling() + [0.0, 0.0, 0.1]})
    for name in ("traj3d_mean", "traj3d_rmse", "traj3d_final"):
        value = get_metric(name).compute(shifted, reference)
        assert value.status is MetricStatus.COMPUTED and value.value == pytest.approx(0.1)
    assert get_metric("traj3d_mean").compute(reference, reference).value == pytest.approx(0.0)
    aligned = get_metric("traj3d_mean").compute(
        shifted, reference, options={"align": "translation"}
    )
    assert aligned.value == pytest.approx(0.0, abs=1e-9)
    scaled = make_tracks({"ball": falling() * 2.0})
    assert get_metric("traj3d_mean").compute(
        scaled, reference, options={"align": "similarity"}
    ).value == pytest.approx(0.0, abs=1e-9)
    assert (
        get_metric("traj3d_mean").compute(shifted, reference, options={"align": "bogus"}).status
        is MetricStatus.ERROR
    )
    window = get_metric("traj3d_mean").compute(shifted, reference, frames=(10, 20))
    assert window.value == pytest.approx(0.1)


def test_trajectory_errors_explain_failures() -> None:
    reference = make_tracks({"ball": falling()})
    assert "missing" in (
        get_metric("traj3d_mean").compute(make_tracks({"cube": falling()}), reference).detail or ""
    )
    short = make_tracks({"ball": falling(20)})
    assert "covers" in (get_metric("traj3d_mean").compute(short, reference).detail or "")
    assert (
        get_metric("traj3d_mean").compute(reference, reference, frames=(0, 99)).status
        is MetricStatus.ERROR
    )
    resampled = make_tracks({"ball": falling(120, fps=60)}, fps=60)
    assert get_metric("traj3d_mean").compute(resampled, reference).value == pytest.approx(
        0.0, abs=1e-3
    )


def test_projected_errors() -> None:
    reference = make_tracks({"ball": falling()})
    moved = make_tracks({"ball": falling() + [0.1, 0.0, 0.0]})
    px = get_metric("traj2d_px_mean").compute(moved, reference)
    assert px.value == pytest.approx(300 * 0.1 / 2.0, rel=1e-6)
    norm = get_metric("traj2d_norm_mean").compute(moved, reference)
    assert norm.value == pytest.approx(px.value / np.hypot(320, 240))
    no_camera = make_tracks({"ball": falling()})
    object.__setattr__(no_camera, "camera", None)
    assert get_metric("traj2d_px_mean").compute(no_camera, reference).status is MetricStatus.ERROR
    assert (
        get_metric("traj2d_px_mean").compute(reference, no_camera).status
        is MetricStatus.NOT_APPLICABLE
    )


def panning_scene(fps: float, seconds: float = 1.0):
    """A ball moving at constant velocity, filmed by a camera that slides and turns at constant
    rates: sampled at any frame rate it is the same physical scene, exactly."""
    from real2simlab.formats.tracks import Camera

    t = np.arange(round(seconds * fps) + 1) / fps
    ball = np.c_[0.4 * t - 0.2, 0.3 * t, 0.5 + 0.2 * t]
    poses = []
    for time in t:
        yaw = 0.5 * time
        c, s = np.cos(yaw), np.sin(yaw)
        turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        look = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # along +y
        pose = np.eye(4)
        pose[:3, :3] = turn @ look
        pose[:3, 3] = [0.3 * time, -2.0, 0.5]
        poses.append(pose)
    camera = Camera(
        intrinsics=np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]]),
        width=320,
        height=240,
        extrinsics=np.array(poses),
    )
    return make_tracks({"ball": ball}, fps=fps, camera=camera)


@pytest.mark.parametrize(("predicted_fps", "reference_fps"), [(20, 10), (10, 20), (10, 10)])
@pytest.mark.parametrize("name", ["traj2d_px_mean", "centroid_set_chamfer_2d"])
def test_moving_camera_at_another_frame_rate(
    name: str, predicted_fps: float, reference_fps: float
) -> None:
    """A correct prediction sampled at another frame rate than the reference projects each
    centroid with its own camera at the same instant: no image error, in either direction."""
    reference = panning_scene(reference_fps)
    value = get_metric(name).compute(panning_scene(predicted_fps), reference)
    assert value.status is MetricStatus.COMPUTED and value.value == pytest.approx(0.0, abs=1e-6)
    window = get_metric(name).compute(panning_scene(predicted_fps), reference, frames=(3, 9))
    assert window.value == pytest.approx(0.0, abs=1e-6)
    still = make_tracks({"ball": falling(31, fps=30)}, fps=30)  # a static camera, for control
    assert get_metric(name).compute(
        make_tracks({"ball": falling(61, fps=60)}, fps=60), still
    ).value == pytest.approx(0.0, abs=1e-2)


def test_centroid_sets_fail_when_the_prediction_drops_objects() -> None:
    """Frames where the reference has objects and the prediction none are not skipped: such a
    prediction is an error, not a perfect reproduction."""
    from real2simlab.formats.tracks import ObjectTrack, Tracks

    reference = make_tracks({"ball": falling()})
    visible = np.zeros(60, dtype=bool)
    visible[0] = True
    gone = Tracks(
        fps=30.0,
        objects={"ball": ObjectTrack(position=falling(), visible=visible)},
        camera=reference.camera,
    )
    for name in ("centroid_set_chamfer", "centroid_set_chamfer_2d"):
        metric = get_metric(name)
        assert metric.compute(reference, reference).value == pytest.approx(0.0)
        dropped = metric.compute(gone, reference)
        assert dropped.status is MetricStatus.ERROR and "59 of the 60 frames" in (
            dropped.detail or ""
        )
        assert metric.compute(static_tracks(reference), reference).value > 0


HUGE = (1.2e154, 1e200)
"""Distances whose squares are beyond the largest float, while they themselves are not."""


@pytest.mark.parametrize("offset", HUGE)
@pytest.mark.parametrize(
    "name", ["traj3d_mean", "traj3d_rmse", "traj3d_final", "centroid_set_chamfer"]
)
def test_huge_but_finite_errors_are_computed(name: str, offset: float) -> None:
    """A prediction a huge but finite distance away has that error: computed without
    overflowing, never Infinity, NaN or an error about missing frames."""
    reference = make_tracks({"ball": falling()})
    far = make_tracks({"ball": falling() + [offset, 0.0, 0.0]})
    value = get_metric(name).compute(far, reference)
    assert value.status is MetricStatus.COMPUTED, value
    assert value.value == pytest.approx(offset, rel=1e-12)


@pytest.mark.parametrize("name", ["traj2d_px_mean", "traj2d_norm_mean", "centroid_set_chamfer_2d"])
def test_huge_but_finite_image_errors_are_computed(name: str) -> None:
    """In the image too: the error grows with the offset (linearly, for a sideways offset in
    front of the camera) instead of overflowing."""
    reference = make_tracks({"ball": falling()})
    values: list[float] = []
    for offset in (1e150, 1e200):
        far = make_tracks({"ball": falling() + [offset, 0.0, 0.0]})
        value = get_metric(name).compute(far, reference)
        assert value.status is MetricStatus.COMPUTED and value.value is not None, value
        values.append(value.value)
    assert values[1] / values[0] == pytest.approx(1e50, rel=1e-9)


@pytest.mark.parametrize(
    "name",
    ["traj3d_mean", "traj3d_rmse", "traj3d_final", "centroid_set_chamfer",
     "centroid_set_chamfer_2d", "traj2d_px_mean"],
)  # fmt: skip
def test_an_error_beyond_the_largest_float_is_an_error_not_infinity(name: str) -> None:
    reference = make_tracks({"ball": falling() + [-1.5e308, 0.0, 0.0]})
    far = make_tracks({"ball": falling() + [1.5e308, 0.0, 0.0]})
    value = get_metric(name).compute(far, reference)
    assert value.status is MetricStatus.ERROR and value.value is None, value
    assert value.to_json() == json.loads(json.dumps(value.to_json(), allow_nan=False))


def _strict_json(text: str) -> object:
    def refuse(constant: str) -> object:
        raise ValueError(f"not JSON: {constant}")

    return json.loads(text, parse_constant=refuse)


def _puck_slide(root: Path, metrics: tuple[str, ...], primary: str) -> Path:
    from real2simlab.task import load_task
    from real2simlab.task.io import write_task_config

    shutil.copytree(ROOT / "examples" / "tasks" / "puck_slide", root)
    config = load_task(root).config
    scoring = config.scoring.model_copy(update={"metrics": metrics, "primary": primary})
    write_task_config(config.model_copy(update={"scoring": scoring}), root / "task.toml")
    return root


def test_a_huge_but_valid_reference_scores_its_true_skill(tmp_path: Path) -> None:
    """A task whose reference is huge but finite, which validates: the error, the static error,
    the skill and the noise floor of the recorded repeats are computed without overflowing.
    Halfway between the static scene and the reference is a skill of 0.5, not NaN clipped
    to 0, and the record is JSON (no NaN, no Infinity)."""
    from real2simlab.scoring import MethodInfo, RunOutput, Scorer
    from real2simlab.task import load_task, validate_task

    root = _puck_slide(tmp_path / "task", ("traj3d_rmse",), "traj3d_rmse")
    reference = root / "hidden" / "reference" / "tracks.npz"
    with np.load(reference) as archive:
        positions = archive["puck.position"].copy()
    positions[:] = [1.2e154, 0.0, 0.0]
    positions[0] = [0.0, 0.0, 0.0]
    np.savez(reference, **{"puck.position": positions})
    assert validate_task(root).ok
    output = tmp_path / "output"
    shutil.copytree(root / "hidden" / "reference", output)
    halfway = positions.copy()
    halfway[1:] = [0.6e154, 0.0, 0.0]
    np.savez(output / "tracks.npz", **{"puck.position": halfway})
    result = Scorer(load_task(root)).score(
        method=MethodInfo("halfway", "method"), trial=0, seed=0, delivered=True,
        run=RunOutput("succeeded", output),
    )  # fmt: skip
    reproduction = result.reproduction
    assert reproduction.score is not None
    assert reproduction.score.skill_raw == pytest.approx(0.5, rel=1e-9)
    assert reproduction.score.skill == pytest.approx(0.5, rel=1e-9)
    assert reproduction.noise_floor is not None and np.isfinite(reproduction.noise_floor)
    assert reproduction.error_to_noise is not None and np.isfinite(reproduction.error_to_noise)
    _strict_json(result.dumps())


def test_a_twin_with_huge_positions_gets_a_json_record(tmp_path: Path) -> None:
    """The twin's side, on an unchanged task: a program whose positions are huge but finite (a
    diverging integrator) gets finite errors and a skill of 0, and a record every JSON reader
    can parse."""
    from real2simlab.scoring import MethodInfo, RunOutput, Scorer
    from real2simlab.task import load_task

    metrics = ("traj3d_mean", "traj3d_rmse", "traj3d_final", "traj2d_norm_mean",
               "centroid_set_chamfer", "centroid_set_chamfer_2d")  # fmt: skip
    root = _puck_slide(tmp_path / "task", metrics, "traj3d_rmse")
    output = tmp_path / "output"
    shutil.copytree(root / "hidden" / "reference", output)
    with np.load(output / "tracks.npz") as archive:
        positions = archive["puck.position"].copy()
    positions[1:] = [1.2e154, 0.0, 0.0]
    np.savez(output / "tracks.npz", **{"puck.position": positions})
    result = Scorer(load_task(root)).score(
        method=MethodInfo("diverging", "method"), trial=0, seed=0, delivered=True,
        run=RunOutput("succeeded", output),
    )  # fmt: skip
    score = result.reproduction.score
    assert score is not None and score.skill == 0.0
    for name in metrics:
        record = score.errors[name]
        assert record.status == "computed" and np.isfinite(record.value), (name, record)
    _strict_json(result.dumps())


# Alignment of huge but finite positions. The tests below turn numpy's RuntimeWarnings into
# errors: none of these comparisons may print an overflow warning on the terminal either.
FAR = 1e307
"""A position whose sum over 60 frames is beyond the largest float, while it itself is not."""
ALIGNED = ["traj3d_mean", "traj3d_rmse", "traj3d_final", "centroid_set_chamfer"]
SIMILARITY = {"align": "similarity"}


def _far(kind: str) -> np.ndarray:
    """The reference ball (x = y = 0), moved far along x: all of it, from frame 1 on, or from
    frame 1 on alternately to either side."""
    position = falling()
    if kind == "offset":
        position[:, 0] = FAR
    elif kind == "step":
        position[1:, 0] = FAR
    else:
        position[1:, 0] = 1.5e308
        position[1::2, 0] = -1.5e308
    return position


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("factor", [1e160, 1e300])
@pytest.mark.parametrize("name", ALIGNED)
def test_a_huge_multiple_of_the_reference_aligns_onto_it(name: str, factor: float) -> None:
    """Similarity alignment is for a prediction whose scale is unknown: 1e160 times the
    reference aligns onto it exactly, even where its squares are beyond the largest float (it
    was squeezed onto the reference's centroid, an error of 0.1 m)."""
    reference = make_tracks({"ball": falling()})
    far = make_tracks({"ball": falling() * factor})
    value = get_metric(name).compute(far, reference, options=SIMILARITY)
    assert value.status is MetricStatus.COMPUTED, value
    assert value.value == pytest.approx(0.0, abs=1e-9)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("kind", ["offset", "step", "alternating"])
@pytest.mark.parametrize("name", ALIGNED)
def test_a_far_prediction_is_aligned_by_similarity(name: str, kind: str) -> None:
    """Positions whose sums overflow (a solver that blew up) are aligned like any others: no
    exception (it stopped the whole experiment: "SVD did not converge"), and not the prediction
    left unaligned. The error is that of the same prediction divided by 2**1000 (exact), where
    nothing overflows: a similarity-aligned error does not depend on the prediction's scale. An
    offset prediction is the reference moved: 0."""
    reference = make_tracks({"ball": falling()})
    position = _far(kind)
    value = get_metric(name).compute(make_tracks({"ball": position}), reference, options=SIMILARITY)
    assert value.status is MetricStatus.COMPUTED and value.value is not None, value
    if kind == "offset":
        assert value.value == pytest.approx(0.0, abs=1e-9)
        return
    small = make_tracks({"ball": np.ldexp(position, -1000)})
    expected = get_metric(name).compute(small, reference, options=SIMILARITY).value
    assert expected is not None and value.value == pytest.approx(expected, rel=1e-9)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("kind", ["offset", "step", "alternating"])
@pytest.mark.parametrize("name", ALIGNED[:3])
def test_a_far_prediction_is_aligned_by_translation(name: str, kind: str) -> None:
    """The shift of a translation is computed without overflowing, and a distance after it is
    "too far to measure" only when it is beyond the largest float (it was said of predictions
    0 m away). Exact values: the frames' x minus their mean."""
    reference = make_tracks({"ball": falling()})
    value = get_metric(name).compute(
        make_tracks({"ball": _far(kind)}), reference, options={"align": "translation"}
    )
    assert value.status is MetricStatus.COMPUTED and value.value is not None, value
    if kind == "offset":
        assert value.value == pytest.approx(0.0, abs=1e-9)
        return
    # step: frame 0 at -59/60 X, the others at X/60; alternating (X = 1.5e308): frame 0 at
    # X/60, the 30 odd frames at -59/60 X, the 29 even ones at 61/60 X.
    x = FAR if kind == "step" else 1.5e308
    expected = {
        "step": {"traj3d_mean": x / 1800 * 59, "traj3d_rmse": x / 60 * 59**0.5,
                 "traj3d_final": x / 60},
        "alternating": {"traj3d_mean": x / 60 * 59, "traj3d_rmse": x / 60 * 3539**0.5,
                        "traj3d_final": x / 60 * 59},
    }  # fmt: skip
    assert value.value == pytest.approx(expected[kind][name], rel=1e-12)


def test_an_alignment_that_fails_is_a_metric_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever fails numerically in an alignment is that prediction's metric error (scored as
    a failure): no twin's numbers stop an experiment."""

    def no_convergence(*args: object, **kwargs: object) -> None:
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(np.linalg, "svd", no_convergence)
    reference = make_tracks({"ball": falling()})
    for name in ("traj3d_mean", "centroid_set_chamfer"):
        value = get_metric(name).compute(
            make_tracks({"ball": falling() * 2.0}), reference, options=SIMILARITY
        )
        assert value.status is MetricStatus.ERROR, value
        assert "cannot be aligned" in (value.detail or "") and "SVD" in (value.detail or "")


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_similarity_transform_of_huge_positions() -> None:
    """The fit itself, for source or target positions too large to square: the same rotation,
    and the scale and translation of the unscaled fit, scaled (dividing by 2**1000 is exact)."""
    rng = np.random.default_rng(0)
    source = rng.normal(size=(60, 3))
    angle = 0.3
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    target = 1.7 * source @ rotation.T + [1, 2, 3]
    scale, found, translation = similarity_transform(np.ldexp(source, 1000), target)
    assert scale == pytest.approx(np.ldexp(1.7, -1000), rel=1e-9) and np.allclose(found, rotation)
    assert np.allclose(translation, [1, 2, 3])
    scale, found, translation = similarity_transform(source, np.ldexp(target, 1000))
    assert scale == pytest.approx(np.ldexp(1.7, 1000), rel=1e-9) and np.allclose(found, rotation)
    assert np.allclose(np.ldexp(translation, -1000), [1, 2, 3])


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("phenomenon", ["bouncing_ball", "pendulum", "incline_slider"])
def test_a_far_twin_is_scored_on_video_only_tasks(
    phenomenon: str, synthetic_tasks: Path, tmp_path: Path
) -> None:
    """A twin of an unchanged built-in task whose solver blows up (x = 1e307 from frame 1 on)
    gets a scored record like any failing twin, a skill of 0 and finite similarity-aligned
    errors, where it stopped the experiment ("SVD did not converge", no results.json)."""
    from real2simlab.scoring import MethodInfo, RunOutput, Scorer
    from real2simlab.task import load_task

    task = load_task(synthetic_tasks / phenomenon / "video_only")
    output = tmp_path / "output"
    shutil.copytree(task.reference_dir(), output)
    with np.load(output / "tracks.npz") as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for key, values in arrays.items():
        if key.endswith(".position"):
            values[1:, 0] = FAR
    np.savez(output / "tracks.npz", **arrays)
    result = Scorer(task).score(
        method=MethodInfo("far", "program"), trial=0, seed=0, delivered=True,
        run=RunOutput("succeeded", output),
    )  # fmt: skip
    assert result.reproduction.status is ReproductionStatus.SCORED and result.infra_error is None
    score = result.reproduction.score
    assert score is not None and score.skill == 0.0
    for name in ("traj3d_mean", "traj3d_rmse", "traj3d_final"):
        record = score.errors[name]
        assert record.status == "computed" and np.isfinite(record.value), (name, record)
    _strict_json(result.dumps())


def test_the_mean_of_huge_values_of_both_signs() -> None:
    """Sums of both signs can overflow to NaN, not only to infinity: the mean is still taken."""
    from real2simlab.metrics.base import robust_mean

    assert robust_mean(np.array([1e308, -1e308, 0, 0, 0, 0, 0, 0] * 2)) == 0.0
    assert robust_mean(np.array([1.5e308] * 30 + [-1.5e308] * 29 + [0.0])) == pytest.approx(
        1.5e308 / 60
    )


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_huge_positions_sampled_at_another_frame_rate() -> None:
    """Between two huge but finite samples the slope of an interpolation can overflow: the
    positions between them are still interpolated (they were lost, and the prediction "missing
    frames"). At 20 fps, x = (k - 10) 1.5e307; at the reference's 30 fps, (2j/3 - 10) 1.5e307."""
    reference = make_tracks({"ball": np.tile([0.0, 0.0, 0.5], (31, 1))}, fps=30)
    k = np.arange(21)
    prediction = make_tracks({"ball": np.c_[(k - 10) * 1.5e307, 0 * k, 0 * k + 0.5]}, fps=20)
    sampled = prediction.at_times(reference.times()).positions("ball")
    assert np.allclose(sampled[:, 0], (2 * np.arange(31) / 3 - 10) * 1.5e307, rtol=1e-12, atol=0)
    value = get_metric("traj3d_mean").compute(prediction, reference)
    assert value.status is MetricStatus.COMPUTED, value
    assert value.value == pytest.approx(480 / 93 * 1.5e307, rel=1e-9)  # mean of |2j/3 - 10|


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_a_bounce_faster_than_the_largest_float_is_a_bounce() -> None:
    """Heights so far apart that the speed between them overflows still make a bounce (faster
    than any min_speed), not a gap in the track; a missing height is a gap."""
    times = np.arange(11) / 30
    z = np.abs(np.arange(11) - 5) * 1e307
    assert bounce_times(z, times, 0.05) == [pytest.approx(5 / 30)]
    z[4] = np.nan
    assert bounce_times(z, times, 0.05) == []


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_a_plot_of_a_twin_beyond_the_float_range() -> None:
    """The report draws a twin that diverges to +-1.5e308 next to the reference (every point
    of the plot was NaN), and a plot of one huge value still has a range."""
    from real2simlab.report.svg import Series, line_plot

    t = np.linspace(0, 2, 60)
    twin = np.full(60, 1.5e308)
    twin[1::2] = -1.5e308
    series = [Series("reference", t, np.sin(t), "#000"), Series("twin", t, twin, "#f00")]
    svg = line_plot(series, title="x", x_label="t", y_label="x (m)")
    assert "nan" not in svg and "inf" not in svg and svg.count("<polyline") == 2
    flat = line_plot([Series("twin", t, np.full(60, 1e300), "#f00")], title="x", x_label="t",
                     y_label="x (m)")  # fmt: skip
    assert "nan" not in flat and "inf" not in flat and flat.count("<polyline") == 1


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_outputs_further_apart_than_the_largest_float_disagree(tmp_path: Path) -> None:
    from real2simlab.scoring.reproduction import outputs_agree

    for name, x in (("a", 1.5e308), ("b", -1.5e308)):
        (tmp_path / name).mkdir()
        np.savez(tmp_path / name / "tracks.npz", x=np.array([x, 1.0]))
    assert not outputs_agree(None, tmp_path / "a", tmp_path / "b", tolerance=1e-6)


def bouncing(e: float = 0.7, frames: int = 90) -> np.ndarray:
    from real2simlab.benchmarks.synthetic.physics import bouncing_ball

    z = bouncing_ball.heights(
        0.6, 9.81, {"restitution": e, "rest_speed": 0.05}, np.arange(frames) / 30
    )
    return np.c_[0 * z, 0 * z, z]


def test_event_metrics() -> None:
    reference = make_tracks({"ball": bouncing(0.7)})
    times = bounce_times(bouncing(0.7)[:, 2], np.arange(90) / 30, 0.05)
    assert len(times) >= 3 and times[0] == pytest.approx(np.sqrt(2 * 0.55 / 9.81), abs=1 / 30)
    assert get_metric("event_first_contact").compute(reference, reference).value == 0.0
    different = make_tracks({"ball": bouncing(0.4)})
    assert get_metric("event_bounce_count").compute(different, reference).value >= 1
    assert get_metric("event_bounce_times").compute(different, reference).value > 0
    flat = make_tracks({"ball": np.zeros((90, 3))})
    assert (
        get_metric("event_first_contact").compute(flat, flat).status is MetricStatus.NOT_APPLICABLE
    )
    assert get_metric("event_first_contact").compute(flat, reference).value > 0


def test_similarity_transform_recovers_known_motion() -> None:
    rng = np.random.default_rng(0)
    source = rng.normal(size=(20, 3))
    angle = 0.3
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    target = 1.7 * source @ rotation.T + [1, 2, 3]
    scale, found, translation = similarity_transform(source, target)
    assert (
        scale == pytest.approx(1.7)
        and np.allclose(found, rotation)
        and np.allclose(translation, [1, 2, 3])
    )


def test_static_tracks_and_differences() -> None:
    reference = make_tracks({"ball": falling()})
    held = static_tracks(reference)
    assert np.allclose(held.positions("ball"), falling()[0])
    assert max_difference(reference, reference) == 0.0
    assert max_difference(reference, make_tracks({"ball": falling(30)})) == float("inf")


def test_bootstrap_is_deterministic_and_bounded() -> None:
    groups = [[0.1, 0.2], [0.5], [0.9, 1.0, 0.8]]
    first, second = bootstrap(groups, resamples=500), bootstrap(groups, resamples=500)
    assert first == second
    assert first.ci_low is not None and first.ci_high is not None
    assert first.ci_low <= first.mean <= first.ci_high
    assert first.mean == pytest.approx(np.mean([0.15, 0.5, 0.9]))
    assert bootstrap([]).mean is None
    single = bootstrap([[0.4]])
    assert single.ci_low == single.ci_high == 0.4


def test_bootstrap_statistic() -> None:
    groups = [[1.0, 3.0], [2.0], [10.0]]
    estimate = bootstrap_statistic(groups, lambda items: max(items), resamples=300)
    assert estimate.mean == 10.0 and estimate.n == 4
    assert estimate.ci_low is not None and estimate.ci_low <= 10.0 == estimate.ci_high
    assert bootstrap_statistic([], lambda items: 1.0).mean is None
    assert bootstrap_statistic([[1.0]], lambda items: None).mean is None


def test_reproduction_skill_without_a_reference_is_not_a_failure() -> None:
    no_reference = Reproduction(status=ReproductionStatus.SCORED, detail="no reference")
    assert summarize(no_reference, None, has_reference=False).reproduction_skill is None
    undelivered = Reproduction(status=ReproductionStatus.NOT_DELIVERED)
    assert summarize(undelivered, None, has_reference=False).reproduction_skill is None
    failed = Reproduction(status=ReproductionStatus.RUN_FAILED)
    assert summarize(failed, None).reproduction_skill == 0.0


def result(
    task: str,
    headline: Headline | None,
    *,
    native: dict | None = None,
    reproducible: bool | None = None,
    consistent: bool | None = None,
) -> TaskResult:
    from datetime import UTC, datetime

    return TaskResult(
        task_id=task,
        task_version="1",
        task_hash="0" * 64,
        benchmark=task.split("/")[0],
        method="m",
        method_kind="agent",
        trial=0,
        seed=0,
        protocol="native" if native is not None else "reproduction",
        protocol_version="reproduction/1.0",
        delivered=True,
        reproduction=Reproduction(
            status=ReproductionStatus.SCORED,
            reproducible=reproducible,
            session_consistent=consistent,
        ),
        summary=TaskSummary(),
        native=native,
        headline=headline,
        created_at=datetime.now(UTC),
        lab_version="1.0.0",
    )


def test_scorecard_averages_headlines_and_references() -> None:
    def skill(value: float | None, repeat: float) -> Headline:
        return Headline(
            metric="reproduction_skill", value=value, range=(0.0, 1.0),
            references={"static": 0.0, "another recording": repeat}, source="lab",
        )  # fmt: skip

    results = [
        result("iris/a", skill(0.9, 0.95), reproducible=True),
        result("iris/b", skill(0.7, 0.85), reproducible=False),
        result("iris/c", skill(None, 0.9)),
        result("synthetic/x", skill(1.0, 0.5)),
        result("other/y", None),
    ]
    scores = {score.benchmark: score for score in benchmark_scores(results, resamples=200)}
    assert set(scores) == {"iris", "synthetic"}
    iris = scores["iris"]
    assert iris.value.mean == pytest.approx(0.8) and (iris.scored, iris.results) == (2, 3)
    assert iris.references == {"another recording": pytest.approx(0.9), "static": 0.0}
    assert iris.reproducible_rate == 0.5 and iris.session_consistent_rate is None
    assert scores["synthetic"].value.mean == 1.0


def test_scorecard_uses_a_benchmarks_own_aggregate() -> None:
    """Physics-IQ pools the metrics of all views before scoring them: not the mean of views."""
    from real2simlab.benchmarks.physicsiq.scorer import PhysicsIQScorer, physics_iq_score

    def view(st: float, sp: float, w: float, mse: float, static: float) -> dict:
        variance = {
            "physical_variance_spatiotemporal_iou": 0.2,
            "physical_variance_spatial_iou": 0.5,
            "physical_variance_weighted_spatial_iou": 0.4,
            "physical_variance_mse": 0.002,
        }
        report = {
            "status": "scored", "spatiotemporal_iou": st, "spatial_iou": sp,
            "weighted_spatial_iou": w, "mse": mse, **variance,
        }  # fmt: skip
        report["physics_iq_score"] = physics_iq_score([report])
        report["static"] = {
            "spatiotemporal_iou": static, "spatial_iou": 0.0, "weighted_spatial_iou": 0.0,
            "mse": 0.003,
        }  # fmt: skip
        return report

    reports = [view(0.3, 0.5, 0.4, 0.05, 0.25), view(0.05, 0.1, 0.05, 0.01, 0.2)]
    scorer = PhysicsIQScorer()
    results = [
        result(f"physicsiq/v{i}", scorer.headline(r), native=r, consistent=True)
        for i, r in enumerate(reports)
    ]
    (score,) = benchmark_scores(results, resamples=200)
    official = physics_iq_score(reports)
    mean_of_views = np.mean([r["physics_iq_score"] for r in reports])
    assert score.value.mean == official and official != pytest.approx(mean_of_views)
    assert score.metric == "physics_iq_score" and score.unit == "points"
    assert "static" in score.references and score.session_consistent_rate == 1.0


def test_native_headlines() -> None:
    from real2simlab.benchmarks.fourdcodebench.scorer import FourDCodeBenchScorer
    from real2simlab.benchmarks.physcodegym.scorer import PhysCodeGymPixels
    from real2simlab.benchmarks.physimcode.scorer import PhySimCodeScorer
    from real2simlab.benchmarks.video2world.scorer import Video2WorldScorer

    missing = {"status": "not_delivered"}
    v2w = Video2WorldScorer.__new__(Video2WorldScorer)
    headline = v2w.headline({"status": "scored", "S": 0.17, "reference": {"S": 0.98}})
    assert headline.metric == "V2WScore" and headline.range == (0.0, 100.0)
    assert headline.value == pytest.approx(17.0)
    assert headline.references == {"reference solution": pytest.approx(98.0)}
    assert v2w.headline(missing).value is None
    # Video2World's benchmark score: the mean within each family, then over families.
    reports = [
        {"status": "scored", "family": "droid", "S": 0.2, "reference": {"S": 1.0}},
        {"status": "scored", "family": "droid", "S": 0.4, "reference": {"S": 0.8}},
        {"status": "scored", "family": "pusht", "S": 0.9, "reference": {"S": 1.0}},
        {"status": "no_package"},
    ]
    combined = Video2WorldScorer.aggregate(reports)
    assert combined["value"] == pytest.approx((30.0 + 90.0) / 2)
    assert combined["reference solution"] == pytest.approx((90.0 + 100.0) / 2)
    psc = PhySimCodeScorer()
    scored = {
        "status": "scored",
        "parameters": {"parse_ok": True, "n_gt": 4, "n_within_tolerance": 2},
    }
    assert psc.headline(scored).value == 0.5
    assert psc.headline({"status": "scored", "parameters": {"parse_ok": False}}).value == 0.0
    assert psc.headline(missing).value is None
    four = FourDCodeBenchScorer.__new__(FourDCodeBenchScorer)
    assert four.headline({"status": "scored", "reward": {"dynamic_iou": 0.41}}).value == 0.41
    # nothing delivered is a failed task: the floor of the headline, not "no score"
    assert four.headline({"status": "no_world", "reward": None}).value == 0.0
    pcg = PhysCodeGymPixels()
    assert pcg.headline({"status": "no_frames", "S_SSIM": 0.0, "S_mot": 0.0}).value == 0.0
    assert pcg.headline(missing).value is None
    for each in (headline, psc.headline(scored), pcg.headline(missing)):
        assert each.source == "official" and each.scorer is not None


def test_outputs_agree(tmp_path) -> None:
    """Session and re-run outputs are compared the same way everywhere: numbers to a tolerance,
    videos frame by frame, other files byte for byte."""
    from real2simlab.formats import get_format, write_tracks
    from real2simlab.media import write_video
    from real2simlab.scoring.reproduction import outputs_agree

    tracks = get_format("tracks")
    a, b = tmp_path / "a", tmp_path / "b"
    write_tracks(make_tracks({"ball": falling()}), a)
    write_tracks(make_tracks({"ball": falling() + 1e-9}), b)
    assert outputs_agree(tracks, a, b, tolerance=1e-6) and not outputs_agree(tracks, a, b)
    video = get_format("video")
    frames = np.zeros((6, 32, 48, 3), np.uint8)
    frames[:, 8:16, 8:16] = 200
    for name in ("c", "d", "e", "n"):
        (tmp_path / name).mkdir()
    write_video(frames, tmp_path / "c" / "render.mp4", fps=10.0)
    write_video(frames, tmp_path / "d" / "render.mp4", fps=10.0)
    noisy = np.clip(frames + np.random.default_rng(0).integers(-3, 4, frames.shape), 0, 255)
    write_video(noisy.astype(np.uint8), tmp_path / "n" / "render.mp4", fps=10.0)
    frames[3, 20:28, 30:40] = 255  # an object where there was none, in one frame
    write_video(frames, tmp_path / "e" / "render.mp4", fps=10.0)
    assert outputs_agree(video, tmp_path / "c", tmp_path / "d")
    assert outputs_agree(video, tmp_path / "c", tmp_path / "n")  # rendering noise
    assert not outputs_agree(video, tmp_path / "c", tmp_path / "e")
    package = get_format("video2sim_package")
    for name, text in (("f", "x"), ("g", "x"), ("h", "y")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "out.txt").write_text(text)
    assert outputs_agree(package, tmp_path / "f", tmp_path / "g")
    assert not outputs_agree(package, tmp_path / "f", tmp_path / "h")
    assert not outputs_agree(video, tmp_path / "f", tmp_path / "h")  # no video in either


def test_outputs_agree_on_everything_the_format_holds(tmp_path) -> None:
    """Two outputs with the same centroids but different geometry, or the same tracks seen by
    another camera, do not agree."""
    import json

    from real2simlab.formats import get_format, write_tracks
    from real2simlab.formats.tracks import Camera
    from real2simlab.scoring.reproduction import outputs_agree

    world = get_format("4d_world")
    rng = np.random.default_rng(0)
    points = rng.normal(size=(5, 20, 3))
    points -= points.mean(axis=1, keepdims=True)  # every frame's points centred on the origin
    for name, scale in (("small", 1.0), ("same", 1.0), ("large", 2.0)):
        root = tmp_path / name
        (root / "dynamics").mkdir(parents=True)
        (root / "camera.json").write_text(json.dumps({"intrinsics": np.eye(3).tolist()}))
        np.savez(root / "dynamics" / "body.npz", pos=(points * scale).astype(np.float32))
        (root / "solver").mkdir()
        (root / "solver" / "cache.bin").write_bytes(rng.bytes(64))  # a solver cache: not compared
    small, large = world.to_tracks(tmp_path / "small"), world.to_tracks(tmp_path / "large")
    assert max_difference(small, large) < 1e-6  # the centroids cannot tell them apart
    assert outputs_agree(world, tmp_path / "small", tmp_path / "same")
    assert not outputs_agree(world, tmp_path / "small", tmp_path / "large", tolerance=1e-3)

    tracks = get_format("tracks")
    base = make_tracks({"ball": falling()})
    assert base.camera is not None
    shifted = base.camera.extrinsics.copy()
    shifted[0, 3] += 0.5  # the same tracks, seen from half a metre to the side
    moved = Camera(
        intrinsics=base.camera.intrinsics,
        width=base.camera.width,
        height=base.camera.height,
        extrinsics=shifted,
    )
    write_tracks(base, tmp_path / "t1")
    write_tracks(make_tracks({"ball": falling()}, camera=moved), tmp_path / "t2")
    assert not outputs_agree(tracks, tmp_path / "t1", tmp_path / "t2", tolerance=1e-6)
