"""A hand-written twin of an IRIS dropping-ball task.

A rigid ball released from rest falls under gravity and bounces off the floor with a constant
coefficient of restitution, read off the clip (the height of its first bounce). ``model.json``
holds the release height, the ball's position and radius, the restitution and the camera from
the task's capture.
"""

import json
from pathlib import Path

import numpy as np
import r2s_twin

MODEL = json.loads((Path(__file__).parent / "model.json").read_text())

twin = r2s_twin.Twin()
z, v = MODEL["release_height"], 0.0
radius, restitution, g = MODEL["radius"], MODEL["restitution"], 9.81
fps, frames, dt = MODEL["fps"], MODEL["frames"], 1e-4

heights = np.empty(frames)
time, next_frame = 0.0, 0
while next_frame < frames:
    if time >= next_frame / fps - 1e-12:
        heights[next_frame] = z
        next_frame += 1
    v -= g * dt
    z += v * dt
    if z < radius and v < 0:
        z, v = radius, -restitution * v
    time += dt

x, y = MODEL["xy"]
position = np.stack([np.full(frames, x), np.full(frames, y), heights], axis=1)
twin.write_tracks(fps=fps, positions={"ball": position}, camera=MODEL["camera"])
