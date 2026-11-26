"""Real2SimLab: a unified platform for real-to-simulation.

A method (a coding agent, a Real2Sim pipeline, a hand-written program) watches a recording of a
physical event and delivers a program that simulates it: a *twin*. The Lab runs every method
the same way on every benchmark it adapts (sandboxed, with the network restricted, recorded),
freezes the delivery and re-runs it offline, and scores it with the benchmark's own official
scorer or the Lab's reproduction metrics, in one record and one scorecard. The same task format
and scorer serve training (``real2simlab.train``): generated training tasks with exact truth,
datasets, and any program's evaluation score as its reward.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
