# ![](../assets/icons/iris.svg){ .r2s-title-icon } IRIS

| | |
|---|---|
| Adapter | `iris` (`r2s bench build iris --out tasks/`) |
| Status | partial: one class (`dropping_ball`) of eight; see [Not adapted](#not-adapted) |
| Protocol | `reproduction`: the Lab's trajectory metrics against trajectories extracted from the videos |
| Headline | `reproduction_skill` (0 to 1); references: static scene 0, another recording (the other takes of the same height) |
| Source | [`rasulkhanbayov/IRIS`](https://huggingface.co/datasets/rasulkhanbayov/IRIS) on Hugging Face, revision `0529a5e0cbbf` |
| License | CC-BY-NC-4.0: **non-commercial**. Videos are downloaded into your cache when tasks are built; Real2SimLab never redistributes them, and the tasks are marked `redistributable = false`. |
| Citation | Khanbayov, Barhdadi, Serpedin, Kurban. *IRIS: A Real-World Benchmark for Inverse Recovery and Identification of Physical Dynamic Systems from Monocular Video.* ECCV 2026. [arXiv:2603.16432](https://arxiv.org/abs/2603.16432); code: [KurbanIntelligenceLab/iris-bench](https://github.com/KurbanIntelligenceLab/iris-bench); project page: [kurbanintelligencelab.github.io/iris-bench](https://kurbanintelligencelab.github.io/iris-bench/) |
| Install | `pip install -e ".[iris]"` in your clone of Real2SimLab |

```bash
r2s bench build iris --out tasks/       # downloads 30 videos (about 620 MB), extracts tracks
r2s eval --tasks tasks/dropping_ball --method static --method constant_velocity \
    --method hand-written=program:path=examples/twins/iris --out runs/iris
```

## Tasks

A real football is dropped onto a carpet from three heights (its lowest point 0.5, 1.0 and
1.5 m above the floor) and filmed at 60 fps (3840 × 2160, decoded at 960 × 540), ten takes per
height. IRIS publishes the videos and the per-setting parameters, but no trajectories; the Lab
extracts them and asks a method to simulate the drop it sees.

| Task | Release height (ball's centre) | Repeats |
|---|---|---|
| `iris/dropping_ball/drop_050` | 0.61 m | 9 takes |
| `iris/dropping_ball/drop_100` | 1.11 m | 9 takes |
| `iris/dropping_ball/drop_150` | 1.61 m | 8 takes (take 05 duplicates take 04, below) |

Each task shows take 01 of its height: 72 frames (1.2 s) from the release, with the estimated
camera (`calibrated` profile). Its reference is take 01's extracted trajectory; the other takes
of the same height are recordings of the same event and give the **noise floor** and the
"another recording" reference point. Metrics: 3D trajectory error (mean, RMS, final),
first-contact time and bounce count; primary `traj3d_mean`.

## How the references are extracted

IRIS publishes per-setting parameters (`drop_height`, `camera_to_horizontal = 1.94 m`) but no
trajectories, so the adapter extracts them from the videos (`real2simlab/benchmarks/iris/track.py`):

1. **Segment** the ball by colour (saturated orange) in every frame, fill its black panels,
   and fit a circle to the convex hull of the largest blob: image position `(u, v)` and radius
   `r`.
2. **Find the events** on a median-filtered radius series: the release (the ball shrinks and
   keeps shrinking; a hand briefly covering it is not a release) and the first floor contact
   (the first minimum of the radius).
3. **Back-project** each frame to the camera frame: depth `Z = k / r` and lateral offsets
   `R (u − cx) / r`, `R (v − cy) / r`, with `k` the focal length times the ball's radius `R`.
4. **Calibrate each take**: the vertical is the direction of the fall (a line fitted to the
   falling points), and `k` is solved so that the ball falls exactly the drop height along it
   between the release and the contact. The world frame has `z` up and its origin on the floor
   under the first contact.
5. **Check with gravity**, which steps 3–4 do not use: the curvature of the fall.

| Setting | Takes used | Gravity check (median, range) | Focal length | Camera tilt | Camera height |
|---|---|---|---|---|---|
| drop_50 | 10 | 9.97 m/s² (7.6–11.7) | ≈ 2040 px | 8.8° | 2.40 m |
| drop_100 | 10 | 9.69 m/s² (9.5–10.2) | ≈ 1170 px | 15.9° | 2.12 m |
| drop_150 | 9 | 9.83 m/s² (9.4–10.0) | ≈ 660 px | 17.8° | 2.00 m |

Gravity comes out within 1–2 % of 9.81 m/s² in every setting without being assumed (the
0.5 m drops are short, about 12 frames of fall, hence their spread). Repeated takes agree to
0.5–2.5 cm RMS. Each task's `hidden/extraction.json` holds the per-take report of its own
setting. Selecting settings (`select = ["drop_50"]`) downloads and tracks only their takes:
each take is calibrated on its own, so a task is the same whichever settings are built with it.

### What the data showed, and what we assume

- **The camera differs per setting.** The dataset states one camera height (1.94 m) for all
  three settings, but the ball's apparent size at floor contact differs by a factor of 2.5
  between them, and the focal lengths above differ by a factor of three (different zoom
  settings). The camera is not looking straight down either (9–18°). The adapter therefore
  calibrates every take from its own video and does not use the 1.94 m.
- **`drop_150` take 05 is take 04** (same frames; 47 bytes of metadata differ). It is left out,
  so that setting has nine takes.
- **The ball's radius is assumed**: 0.11 m, a size-5 football. The videos cannot fix it (the
  focal length is unknown); it offsets every height by the same amount, and the capture marks
  it `default`.
- The principal point is assumed at the image centre; lens distortion is ignored.

## Results

Reproduction skill against take 01 of each height (Codex: one 20-minute run per task,
`docs/results.md`; the hand-written twin is `examples/twins/iris/`, a rigid ball with a constant
restitution read off the clip):

| Task | Codex | Hand-written twin | Constant velocity | Static | Another recording |
|---|---|---|---|---|---|
| `drop_050` | 0.966 | 0.952 | 0.312 | 0 | 0.922 |
| `drop_100` | 0.990 | 0.973 | 0.192 | 0 | 0.972 |
| `drop_150` | 0.994 | 0.978 | 0 | 0 | 0.973 |

*Another recording* is the skill the other takes of the same height score: the spread between
repetitions of the experiment. Both twins reproduce the take they were given more closely than
repeating the experiment does (Codex's error is 0.23–0.44 of the noise floor, the hand-written
twin's 0.61–0.96).

## Comparison with IRIS's own protocol

IRIS scores recovered parameters (drop height, gravity, ...) and equation families per setting;
the Lab scores the simulated trajectory against the recording. The two are not comparable, and
this adapter does not reproduce IRIS's parameter-recovery numbers.

## Not adapted

- `sliding_cone`: its published geometry is inconsistent (the `cone_60` catheti do not fit its
  hypotenuse, and the angle does not match the incline).
- `two_moving_pendulums`: the two balls' releases are not synchronous and their strings give an
  unstable projective geometry; references extracted so far are not reliable enough to score.
- `pendulum`, `two_moving_pendulum_one_static`, `falling_ball`, `rotation`, `hitting_cones`:
  not yet surveyed.
