"""Training with Real2SimLab: training tasks, datasets that hold no evaluation task's
recording, training methods, and the evaluation's scorer as a reward.

* ``r2s train tasks`` (``SyntheticBenchmark.build(split="train")``) and ``r2s bench build NAME
  --split train``: training tasks; or tasks of your own with ``split = "train"``;
* :func:`export_dataset` (``r2s train export``): tasks and the results of runs on them as a
  dataset: inputs, labels (reference trajectories, hidden parameters, truth programs) and, per
  delivered program, its reward and, for agents, their redacted trajectory; with ``against``,
  without the tasks that hold an evaluation task's recording;
* :func:`fit_method` (``r2s train fit``): a method that learns, trained on a dataset, gives a
  checkpoint to evaluate it with; it is never evaluated on a task that overlaps its training
  tasks;
* :func:`score_solution` (``r2s train reward``): one solution's reward, scored exactly as an
  evaluation scores it.
"""

from real2simlab.train.dataset import ExportSummary, export_dataset
from real2simlab.train.fit import fit_method
from real2simlab.train.reward import Scored, reward_of, score_solution

__all__ = ["ExportSummary", "Scored", "export_dataset", "fit_method", "reward_of", "score_solution"]
