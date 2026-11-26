"""``fake_agent``: a scripted agent that calls no model, for testing the whole pipeline.

It goes through exactly the same flow as a real agent (workspace, sandbox, bootstrap,
redaction, solution collection, packaging) but its "CLI" is ``fake_agent_cli.py``, which
writes a physical twin, a broken or invalid twin, or nothing, for the synthetic phenomena. It
runs in the task's runtime image with no network.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path
from typing import ClassVar, Literal

from real2simlab.agents.base import AgentMethod, AgentOptions, Workspace, read_events, total_usage
from real2simlab.agents.record import Usage
from real2simlab.methods.base import BuildContext, MethodOptions
from real2simlab.sandbox.base import Backend, Mount

CLI = Path(__file__).with_name("fake_agent_cli.py")
PHYSICS = Path(__file__).resolve().parents[1] / "benchmarks" / "synthetic" / "physics"
FAKE_DIR = "/r2s/fake"


class FakeAgentOptions(AgentOptions):
    mode: Literal["physical", "broken", "invalid", "nothing"] = "physical"
    network: Literal["allowlist", "open", "none"] = "none"
    reasoning_effort: str | None = None


class FakeAgent(AgentMethod):
    name: ClassVar[str] = "fake_agent"
    agent_name: ClassVar[str] = "fake"
    summary: ClassVar[str] = "a scripted agent for tests: no model is called"
    agent_image: ClassVar[str | None] = None
    Options: ClassVar[type[MethodOptions]] = FakeAgentOptions

    def __init__(self, **options: object) -> None:
        super().__init__(**options)
        self._bundle: Path | None = None

    @property
    def label(self) -> str:
        assert isinstance(self.options, FakeAgentOptions)
        return f"fake_agent:{self.options.mode}"

    def model_name(self) -> str | None:
        assert isinstance(self.options, FakeAgentOptions)
        return f"scripted-{self.options.mode}"

    def bundle(self) -> Path:
        if self._bundle is None:
            root = Path(tempfile.mkdtemp(prefix="r2s-fake-agent-"))
            # Mounted into sandboxes for as long as this process runs; removed when it ends,
            # as every other temporary directory of the Lab is once used.
            atexit.register(shutil.rmtree, root, True)
            shutil.copyfile(CLI, root / "fake_agent_cli.py")
            shutil.copytree(
                PHYSICS, root / "synthphys", ignore=shutil.ignore_patterns("__pycache__")
            )
            # Readable by any user: as root, the Lab runs the agent as an unprivileged uid.
            for path in [root, *root.rglob("*")]:
                path.chmod(0o755 if path.is_dir() else 0o644)
            self._bundle = root
        return self._bundle

    def extra_mounts(self, context: BuildContext) -> list[Mount]:
        return [Mount(self.bundle(), FAKE_DIR, read_only=True)]

    def command(self, context: BuildContext, workspace: Workspace) -> list[str]:
        assert isinstance(self.options, FakeAgentOptions)
        return [
            "python",
            f"{FAKE_DIR}/fake_agent_cli.py",
            "--mode",
            self.options.mode,
            "--workspace",
            "/workspace",
        ]

    def version(self, backend: Backend, image: str) -> str | None:
        return "1.0"

    def usage(self, trajectory: Path) -> Usage | None:
        return total_usage(read_events(trajectory))

    def cost(self, trajectory: Path) -> float | None:
        return 0.0  # it calls no model

    def problems(self, backend: Backend) -> list[str]:
        return []
