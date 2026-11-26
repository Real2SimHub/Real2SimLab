"""``r2s task validate``: every rule a task directory must follow.

Errors make a task unusable; warnings point at likely mistakes. The classes of error a task
author most needs to catch are hidden data leaking into ``visible/`` or the statement,
references that are not valid in the task's output format, and missing files.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from real2simlab.capture.io import CAPTURE_FILENAME, media_problems
from real2simlab.errors import (
    ProblemsError,
    R2SError,
    SetupError,
    SpecError,
    UsageError,
    host_failure,
)
from real2simlab.formats.base import Requirements, get_format, output_problems
from real2simlab.media import is_video_file
from real2simlab.task.io import (
    GENERATOR_DIR,
    HIDDEN_DIR,
    INSTRUCTION_FILE,
    PARAMETERS_FILE,
    REFERENCE_DIR,
    REPEATS_DIR,
    TASK_FILE,
    VISIBLE_DIR,
    Task,
    load_task,
)
from real2simlab.task.model import DeliveryConvention, NativeScoring, Protocol
from real2simlab.twin.fingerprint import sha256_file


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class Code(StrEnum):
    """Machine-readable problem classes."""

    SPEC = "spec"
    HIDDEN_LEAK = "hidden-leak"
    MISSING_FILE = "missing-file"
    NOT_SELF_CONTAINED = "not-self-contained"
    REFERENCE_INVALID = "reference-invalid"
    SCORING = "scoring"


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: Code
    message: str

    def __str__(self) -> str:
        return f"{self.severity.value}: [{self.code.value}] {self.message}"


@dataclass
class ValidationReport:
    """The outcome of validating one task directory."""

    root: Path
    task_id: str | None = None
    issues: list[Issue] = field(default_factory=lambda: [])

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self) -> set[Code]:
        return {issue.code for issue in self.issues}

    def add(self, code: Code, messages: Iterable[str], severity: Severity = Severity.ERROR) -> None:
        self.issues.extend(Issue(severity, code, message) for message in messages)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ProblemsError(
                f"task {self.task_id or self.root} is invalid", [str(i) for i in self.errors]
            )

    def to_json(self) -> dict[str, object]:
        return {
            "path": str(self.root),
            "task_id": self.task_id,
            "ok": self.ok,
            "issues": [
                {"severity": i.severity.value, "code": i.code.value, "message": i.message}
                for i in self.issues
            ],
        }


def validate_task(root: Path, *, check_references: bool = True) -> ValidationReport:
    """Check the task directory at ``root`` and return every issue found."""
    root = Path(root)
    report = ValidationReport(root=root)
    try:
        task = load_task(root)
    except SpecError as error:
        report.add(Code.SPEC, error.problems or [str(error)])
        return report
    report.task_id = task.id
    todo = (root / TASK_FILE).read_text(encoding="utf-8").count("TODO")
    if todo:
        report.add(
            Code.SPEC, [f"{TASK_FILE} still has {todo} TODO(s) to fill in"], Severity.WARNING
        )
    _check_layout(task, report)
    _check_visible(task, report)
    _check_scoring(task, report)
    _check_leaks(task, report)
    _check_parameters(task, report)
    if check_references:
        _check_references(task, report)
    return report


def scoring_problems(task: Task) -> list[str]:
    """What keeps ``task`` from being scored, the checks of :func:`validate_task` scoring
    depends on: its scoring configuration and its reference outputs. A task that fails them
    gives no score (not a score of 0)."""
    report = ValidationReport(root=task.root, task_id=task.id)
    _check_scoring(task, report)
    _check_references(task, report)
    return [str(issue) for issue in report.errors]


HIDDEN_DATA = frozenset({Code.HIDDEN_LEAK, Code.NOT_SELF_CONTAINED})
"""The errors by which a task would show a method its hidden data."""


def hidden_data_problems(task: Task) -> list[str]:
    """What in ``task`` would show a method hidden data, the checks of :func:`validate_task`
    that guard it: a symbolic link (a copy could follow it out of ``visible/``, or out of the
    task) and content of ``hidden/`` in ``visible/`` or as the statement (``instruction.md``).
    A task that fails them is never given to a method or exported, even where its other
    problems are tolerated (a training task without labels)."""
    report = ValidationReport(root=task.root, task_id=task.id)
    _check_layout(task, report)
    _check_leaks(task, report)
    return [str(issue) for issue in report.errors if issue.code in HIDDEN_DATA]


def invalid_task_error(task: Task, problems: Sequence[str], *, then: str) -> SetupError:
    """The error that refuses ``task`` for ``problems`` (as :func:`validate_task` states them):
    one error naming the task and each problem, and the fix, which ends with ``then``."""
    listed = "\n".join(f"  - {problem}" for problem in problems)
    return SetupError(
        f"task {task.id} ({task.root}) is invalid:\n{listed}",
        fix=f"correct it (r2s task validate {task.root} lists every problem), then {then}",
    )


def _check_layout(task: Task, report: ValidationReport) -> None:
    for name in (VISIBLE_DIR, HIDDEN_DIR):
        path = task.root / name
        if path.is_symlink() or not path.is_dir():
            report.add(Code.MISSING_FILE, [f"{name}/: directory is required"])
    links = sorted(
        path.relative_to(task.root).as_posix() for path in task.root.rglob("*") if path.is_symlink()
    )
    report.add(
        Code.NOT_SELF_CONTAINED, [f"{link}: symbolic links are not allowed" for link in links]
    )
    if task.config.delivery.convention is not DeliveryConvention.R2S and (
        task.instruction_override() is None
    ):
        report.add(
            Code.MISSING_FILE,
            [
                f"instruction.md: required for the "
                f"{task.config.delivery.convention.value!r} convention"
            ],
        )


def _check_visible(task: Task, report: ValidationReport) -> None:
    visible = task.visible_dir
    if not visible.is_dir():
        return
    if not (visible / CAPTURE_FILENAME).is_file():
        report.add(Code.MISSING_FILE, [f"{VISIBLE_DIR}/{CAPTURE_FILENAME}: file is required"])
        return
    try:
        capture = task.capture()
    except SpecError as error:
        problems = error.problems or [str(error)]
        code = Code.MISSING_FILE if any("missing" in p for p in problems) else Code.SPEC
        report.add(code, [f"{VISIBLE_DIR}/{CAPTURE_FILENAME}: {p}" for p in problems])
        return
    report.add(
        Code.MISSING_FILE, [f"{VISIBLE_DIR}/: {p}" for p in media_problems(capture, visible)]
    )


def _check_scoring(task: Task, report: ValidationReport) -> None:
    scoring = task.config.scoring
    try:
        fmt = get_format(task.config.delivery.output_format)
    except R2SError as error:
        report.add(Code.SPEC, [f"delivery.output_format: {error}"])
        return
    if scoring.protocol is Protocol.NATIVE:
        from real2simlab import plugins

        if scoring.native is not None and scoring.native.scorer not in plugins.names("scorer"):
            report.add(
                Code.SCORING,
                [f"scoring.native.scorer: no scorer {scoring.native.scorer!r} is registered"],
            )
        elif scoring.native is not None:
            report.add(Code.SCORING, _scorer_option_problems(scoring.native))
        return
    from real2simlab.metrics import get_metric

    for name in scoring.metrics:
        try:
            metric = get_metric(name)
        except R2SError as error:
            report.add(Code.SCORING, [f"scoring.metrics: {error}"])
            continue
        usable = fmt.name in metric.formats or ("tracks" in metric.formats and fmt.provides_tracks)
        if not usable:
            report.add(
                Code.SCORING,
                [f"metric {name!r} supports {sorted(metric.formats)}, not {fmt.name!r}"],
            )
        options = scoring.metric_options.get(name)
        if options:
            report.add(
                Code.SCORING,
                [f"scoring.metric_options: {p}" for p in metric.option_problems(options)],
            )


def _scorer_option_problems(native: NativeScoring) -> list[str]:
    """``[scoring.native.options.scorer]`` checked against what the scorer takes."""
    from real2simlab import plugins
    from real2simlab.options import CredentialOptionError, checked_options
    from real2simlab.scoring.native import scorer_options, task_option_problems

    try:
        factory = plugins.load("scorer", native.scorer)
    except (R2SError, ImportError):  # the scorer cannot load here (a missing extra): its own check
        return []
    problems = [
        f"scoring.native.options: {p}" for p in task_option_problems(factory, native.options)
    ]
    try:
        options = scorer_options(native.options)
        if options:
            checked_options(factory, options, f"scorer {native.scorer!r}")
    except CredentialOptionError as error:  # a problem of the task file, like any other
        problems.append(f"scoring.native.options.scorer: {error.message}; remove it")
    except UsageError as error:
        problems.append(f"scoring.native.options.scorer: {error}")
    return problems


def _observed_videos(task: Task) -> set[str]:
    """SHA-256 of each video the capture names as a view's ``rgb``: the observation itself."""
    try:
        capture = task.capture()
    except R2SError:
        return set()
    videos = [task.visible_dir / view.rgb.path for view in capture.views.values()]
    return {
        sha256_file(path)
        for path in videos
        if is_video_file(path) and path.is_file() and not path.is_symlink()
    }


def _check_leaks(task: Task, report: ValidationReport) -> None:
    """Hidden files whose content appears in ``visible/`` or is the statement
    (``instruction.md``), and hidden-only file names.

    Only the observed video itself is exempt: a hidden video with the content of a view's
    ``rgb`` video is that video, which the method sees anyway (what an official scorer reads
    can include it, as Video2World's sample video; a reference may hold exactly it, as a
    4DCodeBench world's render). Any other hidden content in ``visible/`` is a leak, whatever
    ``capture.json`` lists it as (depth, masks, poses, a robot's recording), and also in place
    of the video: listing a file in the capture does not make it the observation. The task's
    ``instruction.md`` replaces the statement a method is given, so it is compared too. Like
    the rest, the comparison is of whole files: hidden values written into the prose of a
    statement are not found.
    """
    if not task.visible_dir.is_dir() or not task.hidden_dir.is_dir():
        return
    observed = _observed_videos(task)
    hidden: dict[str, str] = {}
    for path in sorted(task.hidden_dir.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.stat().st_size > 0:
            digest = sha256_file(path)
            if digest in observed and is_video_file(path):
                continue  # the observed video itself, visible by definition
            hidden.setdefault(digest, path.relative_to(task.root).as_posix())
    leaks: list[str] = []
    statement = task.root / INSTRUCTION_FILE
    if statement.is_file() and not statement.is_symlink() and statement.stat().st_size > 0:
        copied = hidden.get(sha256_file(statement))
        if copied is not None:
            leaks.append(f"{INSTRUCTION_FILE} has the same content as {copied}")
    for path in sorted(task.visible_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(task.root).as_posix()
        digest = sha256_file(path) if path.stat().st_size > 0 else None
        if digest is not None and digest in hidden:
            leaks.append(f"{relative} has the same content as {hidden[digest]}")
        parts = path.relative_to(task.visible_dir).parts
        if parts and parts[0] in {REFERENCE_DIR, REPEATS_DIR, GENERATOR_DIR}:
            leaks.append(f"{relative}: {parts[0]}/ belongs in {HIDDEN_DIR}/")
    report.add(Code.HIDDEN_LEAK, leaks)


def _check_parameters(task: Task, report: ValidationReport) -> None:
    """``hidden/parameters.json``, when the task has one: the hidden parameters (the labels
    ``r2s train export`` puts in the task's row), a JSON object."""
    path = task.hidden_dir / PARAMETERS_FILE
    if path.is_symlink() or not path.is_file():  # a link is reported by the layout check
        return
    try:
        data = json.loads(path.read_bytes())
    except OSError as error:
        problem = f"cannot be read ({error.strerror or error})"
    except ValueError as error:  # cut short, not JSON, not text
        problem = f"is not valid JSON ({error})"
    else:
        if isinstance(data, dict):
            return
        problem = "is not a JSON object (the hidden parameters by name)"
    report.add(Code.SPEC, [f"{HIDDEN_DIR}/{PARAMETERS_FILE}: {problem}"])


def _unreadable(error: OSError) -> list[str]:
    """A reference this machine could not read (no permission, an I/O error): said as a
    problem of the task here, as the format's checks say a file they cannot read."""
    if not host_failure(error):
        raise error
    return [f"cannot be read: {error}"]


def _check_references(task: Task, report: ValidationReport) -> None:
    config = task.config
    if config.scoring.protocol is Protocol.NATIVE:
        return
    try:
        fmt = get_format(config.delivery.output_format)
    except R2SError:
        return
    reference = task.reference_dir()
    label = f"{HIDDEN_DIR}/{REFERENCE_DIR}/"
    if not reference.is_dir():
        report.add(Code.MISSING_FILE, [f"{label}: the reference output is missing"])
    else:
        try:
            problems = output_problems(
                fmt, reference, Requirements.from_task(config.delivery.output)
            )
        except R2SError as error:
            problems = [str(error)]
        except OSError as error:
            problems = _unreadable(error)
        report.add(Code.REFERENCE_INVALID, [f"{label} {p}" for p in problems])
    for repeat in task.repeat_dirs():
        try:
            problems = output_problems(fmt, repeat, Requirements())
        except R2SError as error:
            problems = [str(error)]
        except OSError as error:
            problems = _unreadable(error)
        rel = repeat.relative_to(task.root).as_posix()
        report.add(Code.REFERENCE_INVALID, [f"{rel}/: {p}" for p in problems])
    generator = task.generator_dir
    if generator is not None:
        from real2simlab.twin import InvalidTwinError, load_twin

        try:
            load_twin(generator)
        except InvalidTwinError as error:
            report.add(Code.SPEC, [f"{HIDDEN_DIR}/{GENERATOR_DIR}/: {p}" for p in error.problems])


__all__ = [
    "HIDDEN_DATA",
    "Code",
    "Issue",
    "Severity",
    "TASK_FILE",
    "ValidationReport",
    "hidden_data_problems",
    "invalid_task_error",
    "scoring_problems",
    "validate_task",
]
