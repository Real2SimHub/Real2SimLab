"""Small SDK-based twin for the puck-slide example task."""

import numpy as np
import r2s_twin

twin = r2s_twin.Twin()
velocity = [1.2, 0.0, 0.0]  # the measured launch (measurements.json)
friction = 0.2  # a guess: the recording does not publish it

fps, frames, g = 30.0, 60, 9.81
t = np.arange(frames) / fps
start = np.array([-0.6, 0.0, 0.015])
v = np.asarray(velocity, dtype=float)
speed = np.linalg.norm(v[:2])
direction = v[:2] / speed if speed > 1e-12 else np.zeros(2)
deceleration = max(float(friction) * g, 1e-12)
tau = np.minimum(t, speed / deceleration)
distance = speed * tau - 0.5 * deceleration * tau**2
position = np.column_stack(
    (
        start[0] + direction[0] * distance,
        start[1] + direction[1] * distance,
        np.full(frames, start[2]),
    )
)
twin.write_tracks(
    fps=fps,
    positions={"puck": position},
    camera={
        "intrinsics": [
            [461.03571047307986, 0.0, 240.0],
            [0.0, 461.03571047307986, 135.0],
            [0.0, 0.0, 1.0],
        ],
        "width": 480,
        "height": 270,
        "extrinsics": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -0.5407575913134988, 0.8411784753765535, -1.4],
            [0.0, -0.8411784753765535, -0.5407575913134988, 0.9],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
)
