"""A scripted stand-in for a coding agent, for tests and CI. Calls no model.

It runs in the agent sandbox exactly where a real agent CLI would, reads the workspace, and
writes ``solution/`` according to ``--mode``:

* ``physical``: a twin that simulates the synthetic phenomenon with slightly wrong parameters;
* ``broken``: a program that crashes;
* ``invalid``: a program that writes an invalid output;
* ``nothing``: writes no solution.

Needs NumPy and SciPy (the ``r2s-python`` image) and the ``synthphys`` package next to it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from synthphys import PHENOMENA  # noqa: E402  # pyright: ignore[reportMissingImports]

RUN_SIMULATE = """\
import json
from pathlib import Path

import r2s_twin
from synthphys import PHENOMENA

MODEL = json.loads(Path(__file__).with_name("model.json").read_text())
phenomenon = PHENOMENA[MODEL["phenomenon"]]
twin = r2s_twin.Twin()
tracks = phenomenon.simulate(phenomenon.SETUP, MODEL["params"], phenomenon.FPS)
twin.write_tracks(fps=phenomenon.FPS,
                  positions={n: d["position"] for n, d in tracks.items()},
                  camera=phenomenon.CAMERA)
"""

RUN_BROKEN = 'raise SystemExit("this twin is broken on purpose")\n'

RUN_INVALID = """\
import os
from pathlib import Path
Path(os.environ["R2S_OUTPUT_DIR"], "meta.json").write_text("{\\"spec\\": \\"wrong\\"}")
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="physical")
    parser.add_argument("--workspace", default="/workspace")
    args = parser.parse_args(argv)
    workspace = Path(args.workspace)
    statement = sys.stdin.read()
    print(json.dumps({"type": "fake.start", "mode": args.mode, "statement_chars": len(statement)}))
    capture = json.loads((workspace / "visible" / "capture.json").read_text())
    name = capture["capture_id"].split("/")[-1]
    if name not in PHENOMENA:
        print(json.dumps({"type": "fake.error", "message": f"not a synthetic task: {name}"}))
        return 1
    phenomenon = PHENOMENA[name]
    solution = workspace / "solution"
    solution.mkdir(exist_ok=True)
    if args.mode == "nothing":
        print(json.dumps({"type": "fake.done", "delivered": False}))
        return 0
    shutil.copyfile(workspace / "tools" / "r2s_twin.py", solution / "r2s_twin.py")
    if args.mode == "physical":
        shutil.copytree(
            HERE / "synthphys", solution / "synthphys", ignore=shutil.ignore_patterns("__pycache__")
        )
        model = {"phenomenon": name, "params": phenomenon.GUESS_PARAMS}
        (solution / "model.json").write_text(json.dumps(model))
        (solution / "run.py").write_text(RUN_SIMULATE)
    elif args.mode == "broken":
        (solution / "run.py").write_text(RUN_BROKEN)
    elif args.mode == "invalid":
        (solution / "run.py").write_text(RUN_INVALID)
    else:
        print(json.dumps({"type": "fake.error", "message": f"unknown mode {args.mode}"}))
        return 2
    if args.mode == "physical":
        # Like a real agent, run the program once in the workspace.
        import os
        import subprocess

        env = {**os.environ, "R2S_OUTPUT_DIR": str(workspace / "output"), "R2S_SEED": "0"}
        subprocess.run(
            [sys.executable, "run.py"], cwd=solution, env=env, check=False, stdin=subprocess.DEVNULL
        )
    usage = {"input_tokens": 1000, "cached_input_tokens": 0, "output_tokens": 100}
    print(json.dumps({"type": "turn.completed", "usage": usage}))
    print(json.dumps({"type": "fake.done", "delivered": True, "mode": args.mode}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
