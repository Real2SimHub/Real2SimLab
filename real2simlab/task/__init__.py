"""Task directories (``r2s-task/1.0``): model, building and validation."""

from real2simlab.task.build import TaskBuilder
from real2simlab.task.io import (
    HIDDEN_DIR,
    TASK_FILE,
    VISIBLE_DIR,
    Task,
    find_tasks,
    load_task,
    read_task_config,
    write_task_config,
)
from real2simlab.task.model import (
    Delivery,
    DeliveryConvention,
    NativeScoring,
    OutputRequirements,
    ProgramRules,
    Protocol,
    Resources,
    ScoringConfig,
    SourceInfo,
    Split,
    TaskConfig,
)
from real2simlab.task.validate import Code, Issue, Severity, ValidationReport, validate_task

__all__ = [
    "HIDDEN_DIR",
    "TASK_FILE",
    "VISIBLE_DIR",
    "Code",
    "Delivery",
    "DeliveryConvention",
    "Issue",
    "NativeScoring",
    "OutputRequirements",
    "ProgramRules",
    "Protocol",
    "Resources",
    "ScoringConfig",
    "Severity",
    "SourceInfo",
    "Split",
    "Task",
    "TaskBuilder",
    "TaskConfig",
    "ValidationReport",
    "find_tasks",
    "load_task",
    "read_task_config",
    "validate_task",
    "write_task_config",
]
