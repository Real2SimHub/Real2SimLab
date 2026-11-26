# Python API

The command line is a thin layer over this API. Everything below is importable from the
package named in each heading; anything not listed here, or starting with an underscore, is
internal and may change between releases.

## Tasks

::: real2simlab.task.TaskBuilder

::: real2simlab.task.load_task

::: real2simlab.task.validate_task

::: real2simlab.task.find_tasks

::: real2simlab.task.Task

::: real2simlab.task.TaskConfig

## Captures

::: real2simlab.capture.Capture

::: real2simlab.capture.load_capture

## Twins

::: real2simlab.twin.TwinManifest


::: real2simlab.twin.load_twin

::: real2simlab.twin.fingerprint

::: real2simlab.runner.run_twin

## Experiments

::: real2simlab.orchestrate.parse_experiment

::: real2simlab.orchestrate.load_experiment

::: real2simlab.orchestrate.run_experiment

::: real2simlab.scoring.Results

## Training

::: real2simlab.train.score_solution

::: real2simlab.train.reward_of

::: real2simlab.train.Scored

::: real2simlab.train.export_dataset

::: real2simlab.train.ExportSummary

::: real2simlab.train.fit_method

::: real2simlab.train.reward.RewardUnavailable

::: real2simlab.methods.training.TrainContext

::: real2simlab.methods.training.TrainingRecord

::: real2simlab.task.overlap.same_recording

::: real2simlab.task.overlap.OverlapIndex

## Extending

::: real2simlab.benchmarks.base.BenchmarkAdapter

::: real2simlab.methods.Method

::: real2simlab.methods.package_twin

::: real2simlab.agents.AgentMethod

::: real2simlab.metrics.Metric

::: real2simlab.formats.OutputFormat

::: real2simlab.scoring.NativeScorer

::: real2simlab.sandbox.Backend
