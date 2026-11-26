"""Reference trajectories from IRIS videos, by classical vision.

Dropping ball (``Dropping_ball``): the camera looks down at the floor, roughly but not exactly
straight. The ball is segmented by colour (saturated orange), holes from its black panels
filled, and its image circle ``(u, v, r)`` measured in every frame. A sphere's image radius is
``f R / Z`` to first order, ``Z`` its depth along the optical axis, so with ``k = f R`` each frame
back-projects to the camera-frame point ``(R (u - cx) / r, R (v - cy) / r, k / r)`` (principal
point at the image centre; the small radial stretch of off-axis spheres is corrected).

Two events fix the geometry of a take: at the release the ball is held at rest, and at its first
floor contact its centre is one radius above the floor; between them it falls the setting's
drop height along the vertical. The vertical is the direction of the fall (a line fitted to the
falling points), and ``k`` is solved so that the fall along it equals the drop height. Gravity is
not used for the geometry: the fall's curvature is reported as an independent check.

Everything here is an estimate (provenance ``estimated``); the adapter's report lists every
take's events, camera and gravity check.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from real2simlab.media import probe_video, tool

GRAVITY = 9.81
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BallObservations:
    """Per-frame apparent ball circles at the decoded resolution."""

    fps: float
    width: int
    height: int
    scale: float
    """Decoded pixels per original pixel."""
    u: FloatArray
    v: FloatArray
    radius: FloatArray
    """NaN where the ball is not found or touches the image border."""


def decode(video: Path, width: int) -> tuple[NDArray[np.uint8], float, float]:
    """Frames of ``video`` scaled to ``width`` pixels wide: (frames, fps, scale)."""
    info = probe_video(video)
    height = round(info.height * width / info.width / 2) * 2
    command = [
        tool("ffmpeg"), "-v", "error", "-i", str(video),
        "-vf", f"scale={width}:{height}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]  # fmt: skip
    result = subprocess.run(
        command, capture_output=True, check=True, stdin=subprocess.DEVNULL, timeout=600
    )
    frames = np.frombuffer(result.stdout, dtype=np.uint8)
    count = len(frames) // (width * height * 3)
    return (
        frames[: count * width * height * 3].reshape(count, height, width, 3),
        info.fps,
        width / info.width,
    )


def orange_mask(frame: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Saturated orange pixels (hue 8-38 degrees, saturation > 0.5, value > 0.3)."""
    rgb = frame.astype(np.float32) / 255.0
    high = rgb.max(axis=2)
    low = rgb.min(axis=2)
    saturation = np.where(high > 0, (high - low) / np.maximum(high, 1e-6), 0.0)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = np.degrees(np.arctan2(np.sqrt(3.0) * (green - blue), 2 * red - green - blue)) % 360
    return (hue > 8) & (hue < 38) & (saturation > 0.5) & (high > 0.3) & (red >= high - 1e-6)


def fit_circle(points: FloatArray) -> tuple[float, float, float]:
    """Least-squares (Kasa) circle through ``points`` (N, 2): centre x, centre y, radius."""
    x, y = points[:, 0], points[:, 1]
    design = np.stack([x, y, np.ones_like(x)], axis=1)
    (a, b, c), *_ = np.linalg.lstsq(design, x * x + y * y, rcond=None)
    cx, cy = a / 2, b / 2
    return float(cx), float(cy), float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))


def ball_circle(mask: NDArray[np.bool_]) -> tuple[float, float, float] | None:
    """The ball's image circle: a robust circle fit to the convex hull of the largest blob.

    The hull closes the gaps the black panels and highlights leave in the colour mask; points
    far outside the fitted circle (fingers holding the ball) are dropped and the fit repeated.
    ``None`` if there is no blob or it touches the image border.
    """
    from scipy.spatial import ConvexHull, QhullError

    labeled: Any = ndimage.label(mask)  # scipy stubs mistype the (labels, count) result
    labels, count = np.asarray(labeled[0]), int(labeled[1])
    if count == 0:
        return None
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    biggest = int(np.argmax(sizes)) + 1
    if sizes[biggest - 1] < 50:
        return None
    blob = labels == biggest
    eroded = np.asarray(ndimage.binary_erosion(blob), dtype=np.bool_)
    rows, cols = np.nonzero(blob & ~eroded)
    height, width = mask.shape
    if rows.min() <= 1 or cols.min() <= 1 or rows.max() >= height - 2 or cols.max() >= width - 2:
        return None
    points = np.stack([cols, rows], axis=1).astype(np.float64) + 0.5
    try:
        hull = points[ConvexHull(points).vertices]
    except QhullError:
        return None
    # Densify the hull so long straight edges weigh like the curved ones.
    closed = np.vstack([hull, hull[:1]])
    dense = [
        closed[i] + (closed[i + 1] - closed[i]) * t
        for i in range(len(hull))
        for t in np.linspace(0, 1, 8, endpoint=False)
    ]
    sample = np.asarray(dense)
    cx, cy, radius = fit_circle(sample)
    for _ in range(4):
        residual = np.hypot(sample[:, 0] - cx, sample[:, 1] - cy) - radius
        keep = residual < 2.5 * np.median(np.abs(residual)) + 0.5
        if keep.sum() < 10 or keep.all():
            break
        sample = sample[keep]
        cx, cy, radius = fit_circle(sample)
    return cx, cy, radius


def observe_ball(video: Path, width: int = 960) -> BallObservations:
    frames, fps, scale = decode(video, width)
    circles = [ball_circle(orange_mask(frame)) for frame in frames]
    u = np.array([c[0] if c else np.nan for c in circles])
    v = np.array([c[1] if c else np.nan for c in circles])
    r = np.array([c[2] if c else np.nan for c in circles])
    return BallObservations(fps, frames.shape[2], frames.shape[1], scale, u, v, r)


@dataclass(frozen=True)
class DropEvents:
    """When a drop's ball starts to fall and first touches the floor, and its size then."""

    moving: int
    """First frame at which the ball is clearly smaller than while held (a few frames after
    the release)."""
    contact: int
    """Frame of the first floor contact: the smallest apparent radius before the bounce."""
    r_release: float
    """Apparent radius while held (decoded pixels)."""
    r_contact: float
    """Apparent radius at the first floor contact (decoded pixels)."""


def smooth(values: FloatArray, size: int = 5) -> FloatArray:
    """A running median that ignores missing frames (NaN where a window has none)."""
    half = size // 2
    padded = np.pad(values.astype(np.float64), half, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    out = np.full(len(values), np.nan)
    has = np.isfinite(windows).any(axis=1)
    out[has] = np.nanmedian(windows[has], axis=1)
    return out


def drop_events(
    radius: FloatArray, *, hold: int = 10, drop: float = 0.97, sustain: int = 5, lookahead: int = 3
) -> DropEvents:
    """Find the release and the first floor contact in a series of apparent radii.

    The series is median-filtered first, so a hand briefly covering the held ball does not look
    like a fall. The ball is moving once the filtered radius stays below ``drop`` times its held
    value for ``sustain`` frames; the contact is the first local minimum after that.
    """
    filtered = smooth(radius)
    finite = filtered[np.isfinite(filtered)]
    if len(finite) < 2 * hold:
        raise ValueError("the ball is visible in too few frames")
    held = float(np.median(finite[:hold]))
    below = np.isfinite(filtered) & (filtered < drop * held)
    runs = np.convolve(below.astype(int), np.ones(sustain, dtype=int), mode="valid")
    starts = np.flatnonzero(runs == sustain)
    if len(starts) == 0:
        raise ValueError("the ball never moves away from the camera")
    moving = int(starts[0])
    before = radius[max(0, moving - 8) : moving]
    r_release = float(np.nanmedian(before)) if np.isfinite(before).any() else held
    for index in range(moving, len(filtered) - lookahead):
        after = filtered[index + 1 : index + 1 + lookahead]
        current = filtered[index]
        if np.isfinite(current) and np.all(np.isfinite(after)) and current < after.min():
            window = np.arange(max(index - 2, 0), min(index + 3, len(radius)))
            window = window[np.isfinite(radius[window])]
            contact = int(window[np.argmin(radius[window])])
            return DropEvents(moving, contact, r_release, float(radius[contact]))
    raise ValueError("no floor contact found")


@dataclass(frozen=True)
class Calibration:
    """A take's camera, from its release and contact and the drop height between them."""

    k: float
    """Focal length (decoded pixels) times the ball's radius (metres)."""
    focal: float
    """Focal length in decoded pixels: ``k`` over the ball's radius."""
    width: int
    height: int
    rotation: FloatArray
    """Rows: the world's x, y and z (up) axes in camera coordinates."""
    origin: FloatArray
    """The world origin in camera coordinates: the floor under the ball's first contact."""

    def to_world(self, camera_points: FloatArray) -> FloatArray:
        return (camera_points - self.origin) @ self.rotation.T

    @property
    def pose(self) -> FloatArray:
        """Camera to world, 4 x 4 (OpenCV camera axes: x right, y down, z forward)."""
        pose = np.eye(4)
        pose[:3, :3] = self.rotation
        pose[:3, 3] = -self.rotation @ self.origin
        return pose

    @property
    def intrinsics(self) -> FloatArray:
        return np.array(
            [[self.focal, 0, self.width / 2], [0, self.focal, self.height / 2], [0, 0, 1.0]]
        )

    @property
    def camera_height(self) -> float:
        return float(self.pose[2, 3])

    @property
    def tilt_deg(self) -> float:
        """Angle between the optical axis and straight down."""
        return float(np.degrees(np.arccos(np.clip(-self.rotation[2, 2], -1.0, 1.0))))


def camera_points(obs: BallObservations, k: float, radius: float) -> FloatArray:
    """Every frame's ball centre in camera coordinates (NaN where the ball was not found)."""
    focal = k / radius
    du, dv = obs.u - obs.width / 2, obs.v - obs.height / 2
    # An off-axis sphere's image is stretched radially by 1/cos(theta); the fitted circle
    # averages the two axes.
    cos_theta = 1.0 / np.sqrt(1.0 + (du * du + dv * dv) / (focal * focal))
    r = obs.radius / (1.0 + (1.0 / cos_theta - 1.0) / 2.0)
    return np.stack([radius * du / r, radius * dv / r, k / r], axis=1)


def _geometry(
    obs: BallObservations, events: DropEvents, k: float, radius: float
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """The release point, the contact point, the downward unit vector, and the fall length."""
    points = camera_points(obs, k, radius)
    held = points[max(0, events.moving - 8) : events.moving]
    held = held[np.isfinite(held).all(axis=1)]
    release = np.median(held, axis=0) if len(held) else points[events.moving]
    contact = points[events.contact]
    fall = points[events.moving : events.contact + 1]
    fall = np.concatenate([release[None], fall[np.isfinite(fall).all(axis=1)]])
    centred = fall - fall.mean(axis=0)
    down = np.linalg.svd(centred, full_matrices=False)[2][0]
    if down @ (contact - release) < 0:
        down = -down
    return release, contact, down, float((contact - release) @ down)


def calibrate(
    obs: BallObservations, events: DropEvents, *, radius: float, fall: float
) -> Calibration:
    """Solve ``k`` so that the ball falls ``fall`` metres along the vertical between its release
    and its first contact; the vertical is the direction of the fall."""
    if not events.r_release > events.r_contact > 0:
        raise ValueError("the ball must look smaller at contact than at rest")
    # Bracket around the straight-down estimate, where the fall length grows with k.
    straight = events.r_contact * events.r_release * fall / (events.r_release - events.r_contact)
    low, high = straight / 3, straight * 3
    if not _geometry(obs, events, low, radius)[3] < fall < _geometry(obs, events, high, radius)[3]:
        raise ValueError("no camera near the straight-down one makes the ball fall the drop height")
    for _ in range(50):
        mid = (low + high) / 2
        if _geometry(obs, events, mid, radius)[3] < fall:
            low = mid
        else:
            high = mid
    k = (low + high) / 2
    _, contact, down, _ = _geometry(obs, events, k, radius)
    up = -down
    x_axis = np.array([1.0, 0.0, 0.0]) - up[0] * up
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.asarray(np.cross(up, x_axis), dtype=np.float64)
    return Calibration(
        k=float(k),
        focal=float(k / radius),
        width=obs.width,
        height=obs.height,
        rotation=np.stack([x_axis, y_axis, up]),
        origin=contact + radius * down,
    )


@dataclass(frozen=True)
class Trajectory:
    """The ball's centre in the world frame from the release, at the video's frame rate."""

    fps: float
    position: FloatArray
    visible: NDArray[np.bool_]
    release_frame: int
    release_time: float
    """Seconds from the first frame of the file to the release."""
    gravity_check: float
    """Gravity implied by the fall's curvature under this calibration (expected about 9.8)."""
    contact_frame: int


def ball_trajectory(
    obs: BallObservations,
    events: DropEvents,
    calibration: Calibration,
    *,
    radius: float,
    fall: float,
    frames: int,
    gravity: float = GRAVITY,
) -> Trajectory:
    """``frames`` frames of the ball's 3D centre from the release (frame 0 = release).

    The floor is ``z = 0``; the ball's centre is ``radius`` above it at contact and about
    ``fall + radius`` at the release. The release time inverts free fall frame by frame
    (median over the fall); ``gravity`` is used only for that timing, not for the geometry.
    """
    times = np.arange(len(obs.radius)) / obs.fps
    world = calibration.to_world(camera_points(obs, calibration.k, radius))
    z = world[:, 2]
    top = radius + fall
    fall_frames = np.arange(events.moving, events.contact)
    fall_frames = fall_frames[np.isfinite(z[fall_frames]) & (top - z[fall_frames] > 0.03)]
    if len(fall_frames) < 4:
        raise ValueError("too few frames of free fall")
    t0 = float(np.median(times[fall_frames] - np.sqrt(2 * (top - z[fall_frames]) / gravity)))
    p2 = float(np.polyfit(times[fall_frames], z[fall_frames], 2)[0])
    release = int(np.ceil(t0 * obs.fps - 1e-6))
    if release < 0 or release + frames > len(z):
        raise ValueError(f"fewer than {frames} frames after the release")
    position = world[release : release + frames].copy()
    visible = np.asarray(np.isfinite(position).all(axis=1), dtype=np.bool_)
    position[~visible] = np.nan
    return Trajectory(
        fps=obs.fps,
        position=position,
        visible=visible,
        release_frame=release,
        release_time=t0,
        gravity_check=-2 * p2,
        contact_frame=events.contact,
    )
