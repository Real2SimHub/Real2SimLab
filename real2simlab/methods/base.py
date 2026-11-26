"""The base class of every twin-building method: agents, baselines, Real2Sim pipelines.

A method gets a ``BuildContext`` (a copy of the task's visible directory, the task statement,
a work directory and a sandbox backend) and returns a ``BuildOutcome``: a frozen twin package,
or the reason it delivered none. Failing to deliver is a result (scored as zero), not an
exception. Only ``privileged`` methods (the oracle and the reference baselines) are given the
task's hidden directory, and their results are labelled as such.

A ``trainable`` method also learns: ``train`` gets an exported dataset and writes a checkpoint
(``r2s train fit``), and the method is evaluated with ``checkpoint=DIR`` among its options
(:mod:`real2simlab.methods.training`).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ValidationError

from real2simlab import plugins
from real2simlab._base import FINITE_CONFIG, validation_problems
from real2simlab.errors import ProblemsError, SetupError
from real2simlab.methods.training import (
    TrainContext,
    check_content,
    checkpoint_digest,
    read_training_record,
    training_index,
)
from real2simlab.options import CREDENTIAL_FIX, refuse_credentials
from real2simlab.sandbox.base import Backend
from real2simlab.task.model import TaskConfig
from real2simlab.twin import (
    AgentInfo,
    EnvironmentRef,
    LoadedTwin,
    Origin,
    TwinManifest,
    save_twin,
)
from real2simlab.twin.conventions import convention_spec
from real2simlab.twin.fingerprint import text_digest

if TYPE_CHECKING:
    from real2simlab.task.io import Task


class MethodOptions(BaseModel):
    """Options of a method, from the experiment configuration. Unknown keys are rejected, and
    so are ``inf`` and ``nan`` (``cpus=1e309``)."""

    model_config = FINITE_CONFIG


class TrainedOptions(MethodOptions):
    """Options of a trainable method."""

    checkpoint: str | None = None
    """A checkpoint written by ``r2s train fit`` with this method."""


@dataclass(frozen=True)
class BuildContext:
    """Everything a method may use to build a twin for one task."""

    task_id: str
    task_hash: str
    config: TaskConfig
    statement: str
    """The generated task statement (or the task's instruction.md)."""
    visible_dir: Path
    """A copy of the task's ``visible/`` directory, owned by this build."""
    work_dir: Path
    """An empty directory the method owns."""
    twin_dir: Path
    """Where the twin package must be saved; must not exist yet."""
    backend: Backend
    seed: int = 0
    trial: int = 0
    hidden_dir: Path | None = None
    """The task's hidden directory; set only for privileged methods."""
    log: Callable[[str], None] = field(default=lambda message: None)


@dataclass(frozen=True)
class BuildOutcome:
    """What a build produced."""

    twin: LoadedTwin | None
    failure: str | None = None
    agent_record: Path | None = None
    session_output: Path | None = None
    """The method's own output, if it left one (e.g. an agent's run in its session)."""
    network: str | None = None
    infra_error: str | None = None
    """Set when the build failed for a reason outside the method (the sandbox did not start,
    the host could not copy or write the delivery): not a result, and retried."""

    @property
    def delivered(self) -> bool:
        return self.twin is not None


def _with_text_hints(error: ValidationError, method: str) -> list[str]:
    """The problems of ``error``, one per line; where text was expected and a value that reads
    as JSON (``1``, ``true``, ``null``) was given, how to give it as text: the command line
    reads ``program:path=1`` as the number 1."""
    import json

    problems = validation_problems(error)
    for index, item in enumerate(error.errors()):
        value = item.get("input")
        if item["type"] == "string_type" and (value is None or isinstance(value, int | float)):
            key = ".".join(str(part) for part in item["loc"])
            text = f'{method}:{key}="{json.dumps(value)}"'
            problems[index] += (
                " (on the command line a value that reads as JSON is read as JSON: give text "
                f"as JSON text, quoted for the shell: --method '{text}')"
            )
    return problems


class Method(ABC):
    """A way of building twins from a task's visible data."""

    name: ClassVar[str]
    kind: ClassVar[str] = "method"
    """``agent``, ``baseline`` or ``method``."""
    summary: ClassVar[str] = ""
    privileged: ClassVar[bool] = False
    method_version: ClassVar[str] = "1"
    """Bump when a method's behaviour changes, so cached builds are redone."""
    Options: ClassVar[type[MethodOptions]] = MethodOptions
    trainable: ClassVar[bool] = False
    """Whether the method learns from data (``train``); its ``Options`` then derive from
    ``TrainedOptions``."""

    def __init__(self, **options: Any) -> None:
        # Options are recorded (experiment.json, build_record.json): never a credential.
        refuse_credentials(options, f"method {self.name!r}", self._credential_fix())
        try:
            self.options = self.Options.model_validate(options)
        except ValidationError as error:
            raise ProblemsError(
                f"invalid options for method {self.name!r}", _with_text_hints(error, self.name)
            ) from error
        if getattr(self.options, "checkpoint", None) == "":  # an unset variable: not "."
            raise SetupError(
                f"method {self.name!r}: checkpoint= is empty: it would be the current directory",
                fix="give the checkpoint directory r2s train fit wrote (its --out), or leave "
                "checkpoint out",
            )
        checkpoint = self.checkpoint
        if checkpoint is not None:
            record = read_training_record(checkpoint)
            if record.method != self.name:
                raise SetupError(
                    f"{checkpoint} was trained with method {record.method!r}, not {self.name!r}",
                    fix=f"give {self.name} a checkpoint of its own (r2s train fit {self.name})",
                )
            check_content(checkpoint, record)
            # a checkpoint that cannot be read, or has no prints, fails here, with a fix
            training_index(checkpoint, str(self.checkpoint_digest()))

    @classmethod
    def _credential_fix(cls) -> str:
        """How to give this method a credential instead of an option."""
        fields = cls.Options.model_fields
        if "api_key_env" in fields:
            return (
                'put the key in an environment variable and give its name: api_key_env = "NAME" '
                "(read at run time, never recorded)"
            )
        if "agent_env" in fields:
            return (
                "pass it to the agent through an environment variable: "
                'agent_env = { NAME = "$NAME" } (Harbor reads the value at run time; only the '
                "name is passed and recorded)"
            )
        return CREDENTIAL_FIX

    @property
    def checkpoint(self) -> Path | None:
        """The checkpoint the method was given (``checkpoint=DIR``), if any."""
        value = getattr(self.options, "checkpoint", None)
        return None if value is None else Path(str(value)).expanduser().resolve()

    def checkpoint_digest(self) -> str | None:
        """The content fingerprint of the checkpoint (read once per method), if any."""
        checkpoint = self.checkpoint
        if checkpoint is None:
            return None
        cached: tuple[Path, str] | None = getattr(self, "_checkpoint_digest", None)
        if cached is None or cached[0] != checkpoint:
            cached = (checkpoint, checkpoint_digest(checkpoint))
            self._checkpoint_digest = cached
        return cached[1]

    @property
    def label(self) -> str:
        return self.name

    def config_digest(self) -> str:
        """A digest of the method and its options, part of every cache key of its builds, and
        of its checkpoint's content: training again redoes the builds."""
        options = json.dumps(self.options.model_dump(mode="json"), sort_keys=True)
        digest = self.checkpoint_digest()
        if digest is None:
            return text_digest(self.name, options, self.method_version)
        return text_digest(self.name, options, self.method_version, digest)

    def problems(self, backend: Backend) -> list[str]:
        """What prevents this method from running with ``backend``; empty when ready."""
        return []

    def applicable(self, task: Task) -> str | None:
        """Why this method cannot deliver for ``task`` at all, or ``None`` if it can.

        Only for methods that make one kind of delivery whatever the task (a baseline that
        writes ``run.py`` programs, a pipeline with one output format): a task they cannot
        deliver for is reported as not applicable instead of scoring zero. A method that could
        deliver and fails to is a result, not this. Agents adapt to every task.
        """
        return None

    def held_out(self, task: Task) -> str | None:
        """Why ``task`` is not held out from this method's training data, or ``None``.

        A method given a checkpoint must not be evaluated on a task that overlaps one of the
        tasks it learned from (the same recording: :mod:`real2simlab.task.overlap`).
        """
        checkpoint, digest = self.checkpoint, self.checkpoint_digest()
        if checkpoint is None or digest is None:
            return None
        index = training_index(checkpoint, digest)
        from real2simlab.task.overlap import default_cache, task_print

        found = index.find(task_print(task, cache=default_cache()))
        return None if found is None else f"trained on {found.task_id}: {found.reason}"

    def train(self, context: TrainContext) -> None:
        """Learn from the dataset in ``context`` and write what was learned in its
        ``checkpoint_dir``. Only for ``trainable`` methods."""
        raise SetupError(
            f"method {self.name!r} does not learn from data",
            fix="choose a method that learns (its class sets trainable = True)",
        )

    @abstractmethod
    def build(self, context: BuildContext) -> BuildOutcome:
        """Build a twin for the task in ``context``."""


def get_method(name: str, **options: Any) -> Method:
    """Create the registered method ``name`` with ``options``."""
    factory = plugins.load("method", name)
    method = factory(**options)
    if not isinstance(method, Method):
        raise TypeError(f"method plugin {name!r} does not create a Method")
    return method


def method_names() -> list[str]:
    return plugins.names("method")


def package_twin(
    context: BuildContext,
    program_dir: Path,
    *,
    method: str,
    entrypoint: Sequence[str] | None = None,
    agent: AgentInfo | None = None,
    model: str | None = None,
    privileged: bool = False,
    notes: str | None = None,
) -> LoadedTwin:
    """Freeze ``program_dir`` as the build's twin, with ``twin.json`` written by the Lab.

    ``environment``, ``output_format`` and ``origin`` come from the task and the method,
    never from the program's author. Raises ``InvalidTwinError`` if the program breaks the
    task's rules.
    """
    delivery = context.config.delivery
    convention = delivery.convention
    spec = convention_spec(convention)
    if entrypoint is None:
        entrypoint = spec.entrypoint
    manifest = TwinManifest(
        twin_id=f"{method}/{context.task_id}/trial{context.trial}",
        entrypoint=tuple(entrypoint),
        environment=EnvironmentRef(image=delivery.environment),
        output_format=delivery.output_format,
        runtime=spec.runtime,
        origin=Origin(
            method=method,
            agent=agent,
            model=model,
            task_id=context.task_id,
            task_version=context.config.version,
            created_at=datetime.now(UTC),
            privileged=privileged,
            notes=notes,
        ),
    )
    return save_twin(manifest, program_dir, context.twin_dir, rules=delivery.program)
