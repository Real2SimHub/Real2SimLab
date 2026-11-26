# Roadmap

What we plan to work on after 1.0, roughly in order of priority. Real2SimLab is a measurement
tool: it runs and scores methods, it does not rank them, host leaderboards or accept
submissions. Contributions toward any item are welcome; open an issue first for the larger ones.

## Benchmarks

- **Wider coverage of the adapted benchmarks**: end-to-end checks of Video2World's other
  evaluators (the hand, MuJoCo and Isaac families are integrated; only Push-T and DROID have
  been run against the shipped records), PhySimCode's 3D (PyBullet) phenomena, 4DCodeBench's
  real-world cases with the official, longer agent budget, more IRIS classes once their
  references can be extracted reliably.
- **More adapters**, each reproducing the benchmark's own protocol first: PhysTwin data,
  text-to-simulation benchmarks, video world-model benchmarks.
- **Model-based official metrics** with pinned weights: 4DCodeBench's DINOv3-based metrics,
  PhySimCode's video similarity, and the LLM-judged scores of PhysCodeGym and PhySimCode for
  users with model access.
- **Noise floors for the synthetic suite**: repeated, noisy renderings of each event.

## Agents and methods

- `mini_swe` with multimodal models: video frames sent as images.
- More agents through Harbor.
- An example adapter for a non-agent Real2Sim pipeline.

## Backends

- An Apptainer (Singularity) backend: many clusters offer only Apptainer.
- Batch execution on Slurm.

## Results and analysis

- Comparing runs of the same tasks (methods, model versions, budgets).
- Classifying failures automatically from the records (no delivery, run failed, not
  reproducible, scored).

## Training

- Training tasks for more phenomena, and from real recordings with measured truth.
- More methods that learn (`r2s train fit`), beside the `nearest_twin` example.

## Bridges

- Export to OpenEnv for reinforcement learning.
- MCP tools for agents: frame access, calibration, tracking.

## Documentation

- More tutorials and short demo animations.
