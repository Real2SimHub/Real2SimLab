# Quickstart

## 1. Install

```bash
git clone https://github.com/Real2SimHub/Real2SimLab.git && cd Real2SimLab
pip install -e .        # Python 3.11+ and ffmpeg
r2s doctor              # what this machine can run, and a fix for everything else
```

There is no PyPI package: optional extras for some benchmark adapters install the same way,
e.g. `pip install -e ".[iris]"`, and `r2s doctor` prints the exact command. To update, `git pull`
and `pip install -e .` again. `r2s doctor --gpu` adds the GPU checks; problems marked `ERR` stop
evaluations (and make the exit status 1), `--` marks advice. Evaluations use Docker unless given
another backend, so without Docker `backend docker` is `ERR`:
[Run on a cluster without Docker](tutorials/clusters.md) uses `--backend bwrap` instead.

## 2. Run the demo

```bash
r2s demo --open
```

It builds two synthetic tasks, scores the reference baselines and two scripted agents, and opens
the report: about 30 seconds on a laptop CPU. The demo chooses where its twins run: in Docker
when the `r2s-python` image is present or can be pulled, otherwise in bubblewrap (Linux, where it
can start its sandbox, image unpacked), otherwise unisolated with a warning (acceptable for the
demo's own programs only). The rest of this page does the same by hand and explains what you
see. Its commands run twins in Docker unless given `--backend`: only the demo chooses by itself.
Without Docker, follow the bubblewrap notes in steps 3 and 5. Slow network:
[Platforms](guides/platforms.md).

## 3. Get the runtime image

```bash
r2s env pull r2s-python       # or build it: r2s env build r2s-python
```

The CPU image twins run in: Python 3.11, NumPy, SciPy, ffmpeg and the twin SDK. Evaluations
pull it automatically when it is missing (`R2S_NO_PULL=1` turns that off). Behind a slow
network, list registry mirrors in `R2S_IMAGE_MIRRORS`, or build it with mirrors:
`--pip-index-url ... --apt-mirror ... --network host`. Without Docker, bubblewrap runs twins in
the same image unpacked: `r2s env unpack r2s-python --from-url` downloads the one published with
this release (other ways: [Run on a cluster without Docker](tutorials/clusters.md)).

## 4. Build tasks

```bash
r2s bench build synthetic --out tasks --select bouncing_ball --select pendulum
r2s task list tasks
r2s task show tasks/bouncing_ball/calibrated
```

Each task is a directory: `task.toml`, `visible/` (the video and what a method may know) and
`hidden/` (the reference and anything scoring needs, never shown to a method).

## 5. Evaluate

```bash
r2s eval --tasks tasks \
    --method oracle --method static --method constant_velocity \
    --method physical=fake_agent:mode=physical \
    --method broken=fake_agent:mode=broken \
    --out runs/quickstart
```

Without Docker, add `--backend bwrap` (with the image unpacked as in step 3).

`oracle`, `static` and `constant_velocity` are privileged reference baselines (they read the
truth generator or the reference). `fake_agent` goes through the whole agent pipeline (sandbox,
workspace, delivery, packaging) without calling a model: `physical` writes a simulator with
slightly wrong parameters, `broken` a program that crashes. Expected scorecard:

| method | synthetic (reproduction_skill) | delivered |
|---|---|---|
| oracle | 1.00 | 100 % |
| physical | ≈ 0.93 | 100 % |
| constant_velocity | ≈ 0.5 | 100 % |
| static | 0.00 | 100 % |
| broken | 0.00 | 100 % (its re-run fails) |

## 6. Read the report

Open `runs/quickstart/report/index.html` in a browser (no server needed): the scorecard, the
methods, and per task the recorded video with each twin's trajectory drawn on it and plots of
the reference and each twin's trajectories. `report.md` and `results.csv` hold the same tables.

Run the same command again: every build, run and official score is reused (only the results
and the report are written again from them). Interrupt it and run it again: it resumes where it
stopped.

## Next

- [Evaluate a real agent](tutorials/evaluate-agent.md) (Codex, Claude Code, Gemini CLI, OpenCode,
  mini-swe-agent, Harbor agents), on one benchmark or on all of them at once.
- [Evaluation](guides/evaluation.md): choose tasks, compare methods and report a score.
- [Write a twin with the SDK](tutorials/write-twin.md) and check it with `r2s twin check`.
- [Write tasks for your benchmark](tutorials/write-tasks.md).
