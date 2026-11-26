"""The mistakes every user makes on the command line: a wrong path, an empty directory, a file
where a directory belongs, the same command run twice, an output directory that already holds
the user's own files. Each must end in a clear ``error:`` line (exit status 1), never in a
traceback, never in a score, and never delete anything of the user's."""

from __future__ import annotations

import contextlib
import errno
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import real2simlab
from real2simlab.cli.main import main

Command = Callable[[Path, Path], list[str]]


def _commands(task: Path) -> dict[str, Command]:
    """Each command with ``path`` in the argument under test and ``out`` for its output."""
    return {
        "eval --tasks": lambda p, o: ["eval", "--tasks", str(p), "--method", "static",
                                      "--backend", "local", "--out", str(o)],
        "train export --tasks": lambda p, o: ["train", "export", "--tasks", str(p), "--out", str(o)],
        "train export --runs": lambda p, o: ["train", "export", "--runs", str(p), "--out", str(o)],
        "train export --against": lambda p, o: ["train", "export", "--tasks", str(task),
                                                "--against", str(p), "--out", str(o)],
        "train fit --data": lambda p, o: ["train", "fit", "nearest_twin", "--data", str(p),
                                          "--out", str(o)],
        "train reward --task": lambda p, o: ["train", "reward", "--task", str(p), "--solution",
                                             str(o), "--backend", "local"],
        "report": lambda p, o: ["report", str(p)],
        "task validate": lambda p, o: ["task", "validate", str(p)],
        "task show": lambda p, o: ["task", "show", str(p)],
        "twin check": lambda p, o: ["twin", "check", str(p), "--out", str(o), "--backend", "local"],
        "twin run": lambda p, o: ["twin", "run", str(p), "--out", str(o), "--backend", "local"],
        "export harbor": lambda p, o: ["export", "harbor", str(p), "--out", str(o)],
        "run": lambda p, o: ["run", str(p)],
        "capture from-video": lambda p, o: ["capture", "from-video", "--capture-id", "c",
                                            "--fov", "60", str(p), str(o)],
        "task new --video": lambda p, o: ["task", "new", "--id", "a/b", "--video", str(p), str(o)],
        "env unpack --from-tar": lambda p, o: ["env", "unpack", "--from-tar", str(p), "x"],
    }  # fmt: skip


COMMANDS = sorted(_commands(Path()).keys())


def _fails_clearly(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    code = main(argv)  # an exception escaping main is the traceback a user would see
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert code == 1, f"r2s {' '.join(argv)} exited {code}"
    assert "error" in text and "Traceback" not in text, text
    return text


@pytest.mark.parametrize("kind", ["missing", "empty", "file"])
@pytest.mark.parametrize("name", COMMANDS)
def test_a_wrong_path_is_a_clear_error(
    name: str, kind: str, bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / kind
    if kind == "empty":
        path.mkdir()
    elif kind == "file":
        path.write_text("not what you meant\n")
    if name == "train reward --task":  # the solution is real here; the task is wrong
        (tmp_path / "out").mkdir()
    _fails_clearly(_commands(bouncing_task)[name](path, tmp_path / "out"), capsys)


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_a_wrong_solution_path_is_not_a_reward_of_0(
    kind: str, bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    solution = tmp_path / "solutoin"  # a typo
    if kind == "file":
        solution.write_text("x\n")
    argv = ["train", "reward", "--task", str(bouncing_task), "--solution", str(solution)]
    assert "reward" not in _fails_clearly([*argv, "--backend", "local"], capsys).split("error:")[0]


def test_task_list_of_a_missing_path_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fails_clearly(["task", "list", str(tmp_path / "missing")], capsys)


def test_building_twice_says_how_to_replace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fix names the command's option, not the Python API's (`overwrite=True`)."""
    for argv in (
        ["bench", "build", "synthetic", "--select", "bouncing_ball", "--tier", "calibrated"],
        ["train", "tasks", "--count", "1", "--select", "bouncing_ball"],
    ):
        out = ["--out", str(tmp_path / argv[1])]
        assert main([*argv, *out]) == 0
        capsys.readouterr()
        text = _fails_clearly([*argv, *out], capsys)
        assert "fix: pass --overwrite to replace it, or choose another --out" in text, text
        assert "overwrite=True" not in text, text
        assert main([*argv, *out, "--overwrite"]) == 0


def test_a_name_given_twice_is_built_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--select`` and ``--tier`` repeat, and a script that joins lists gives a value twice
    (A7-06): it is built once, into a new directory without ``--overwrite``, and counted once
    (the second pass found the tasks the first wrote: "already exists", or with ``--overwrite``
    twice the work and twice the count)."""
    twice = ["--select", "bouncing_ball", "--select", "bouncing_ball", "--tier", "calibrated",
             "--tier", "calibrated"]  # fmt: skip
    for argv, said in (
        (["train", "tasks", "--count", "1", *twice], "wrote 1 training task(s)"),
        (["bench", "build", "synthetic", *twice], "built 1 task(s)"),
    ):
        out = tmp_path / argv[1]
        assert main([*argv, "--out", str(out)]) == 0, capsys.readouterr().err
        assert said in capsys.readouterr().out
        assert len(list(out.rglob("task.toml"))) == 1


@pytest.mark.parametrize(
    ("name", "select"),
    [
        ("iris", ["drop_50", "drop_50"]),
        ("fourdcodebench", ["case", "case"]),
        ("physcodegym", ["case", "case"]),
        ("physimcode", ["engine/experiment/sample", "engine/experiment/sample"]),
        ("physicsiq", ["scenario", "scenario/center"]),
        ("video2world", ["family", "sample"]),
    ],
)
def test_every_benchmark_builds_a_name_given_twice_once(
    name: str, select: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same for every adapter's ``select`` (and its ``variants``): a name given twice, also
    under another name for the same task (a Physics-IQ scenario and its center view, a
    Video2World family and one of its samples), is built once."""
    import importlib

    from real2simlab.benchmarks.base import get_benchmark

    module = importlib.import_module(f"real2simlab.benchmarks.{name}")
    built: list[object] = []

    def build(*args: object, **options: object) -> object:
        if "settings" in options:  # iris builds all its settings in one call
            built.extend(options["settings"])  # type: ignore[arg-type]
            return []
        # an instance (Video2World's holds dicts) by its key
        built.append(tuple(getattr(arg, "key", arg) for arg in args if not isinstance(arg, Path)))
        return tmp_path

    for builder in ("build_case", "build_tasks", "build_instance"):
        if hasattr(module, builder):
            monkeypatch.setattr(module, builder, build)
    if name == "video2world":  # one instance, named by its family and by its sample
        one = module.Instance("family", "sample", {}, {})
        monkeypatch.setattr(module, "instances", lambda: {one.key: one})
    options: dict[str, object] = {}
    if hasattr(module, "VARIANTS"):
        variant = next(v for v in module.VARIANTS if v != "native")
        options["variants"] = [variant, variant]
    get_benchmark(name).build(tmp_path / "out", select=select, **options)
    assert built and len(built) == len(set(built)), built


@pytest.mark.parametrize(
    "argv",
    [
        ["export", "harbor", "{task}", "--out", "{mine}"],
        ["task", "new", "--id", "a/b", "{mine}"],
        ["twin", "check", "{task}/hidden/generator", "--backend", "local", "--out", "{mine}"],
        ["train", "export", "--tasks", "{task}", "--overwrite", "--out", "{mine}"],
        ["train", "reward", "--task", "{task}", "--solution", "{task}/hidden/generator",
         "--backend", "local", "--work", "{mine}"],
        ["capture", "from-video", "--capture-id", "c", "--fov", "60", "{video}", "{mine}"],
    ],
    ids=lambda argv: " ".join(argv[:2]),
)  # fmt: skip
def test_a_directory_of_the_users_is_never_cleared(
    argv: list[str], bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "notes.txt").write_text("keep me")
    video = next(bouncing_task.glob("visible/**/*.mp4"))
    values = {"task": str(bouncing_task), "mine": str(mine), "video": str(video)}
    _fails_clearly([part.format(**values) for part in argv], capsys)
    assert [p.name for p in mine.iterdir()] == ["notes.txt"]
    assert (mine / "notes.txt").read_text() == "keep me"


def test_the_labs_warnings_are_one_plain_line(bouncing_task: Path, tmp_path: Path) -> None:
    # in a process of its own: pytest's configuration hides this warning in-process
    argv = ["twin", "check", str(bouncing_task / "hidden" / "generator"), "--backend", "local"]
    done = subprocess.run(
        [sys.executable, "-m", "real2simlab.cli.main", *argv, "--out", str(tmp_path / "check")],
        capture_output=True,
        text=True,
        check=True,
    )
    err = done.stderr
    assert err.count("WITHOUT ISOLATION") == 1, err
    assert ".py:" not in err, err  # no file name and line, no source line


@pytest.mark.parametrize(
    "argv",
    [
        ["bench", "build", "synthetic", "--select", "not_seen", "--out", "{tmp}/t"],
        ["bench", "build", "synthetic", "--tier", "nowhere", "--out", "{tmp}/t"],
        ["bench", "build", "synthetic", "--option", "nope=1", "--out", "{tmp}/t"],
        ["bench", "build", "synthetic", "--split", "train", "--option", "count=null",
         "--out", "{tmp}/t"],
        ["bench", "build", "synthetic", "--option", "seed=1.5", "--out", "{tmp}/t"],
        ["bench", "build", "synthetic", "--option", "select=pendulum", "--out", "{tmp}/t"],
        ["bench", "build", "physicsiq", "--option", "variants=5", "--out", "{tmp}/t"],
        ["bench", "build", "nobench", "--out", "{tmp}/t"],
        ["train", "tasks", "--count", "1", "--select", "nope", "--out", "{tmp}/t"],
        ["eval", "--tasks", "{task}", "--method", "nomethod", "--out", "{tmp}/r"],
        ["eval", "--tasks", "{task}", "--method", "program:nope=1", "--out", "{tmp}/r"],
        ["eval", "--tasks", "{task}", "--method", "static", "--backend", "nobackend",
         "--out", "{tmp}/r"],
    ],
    ids=lambda argv: " ".join(argv[2:]).split(" --out")[0],
)  # fmt: skip
def test_an_unknown_name_or_a_bad_value_is_a_clear_error(
    argv: list[str], bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    video = next(bouncing_task.glob("visible/**/*.mp4"))
    values = {"tmp": str(tmp_path), "task": str(bouncing_task), "video": str(video)}
    _fails_clearly([part.format(**values) for part in argv], capsys)


@pytest.mark.parametrize(
    "text",
    [
        'name = "x"\nmethods = 123\n[tasks]\npaths = ["t"]\n',
        'name = "x"\n[tasks]\npaths = ["t"]\n[[methods]]\nname = "oracle"\noptions = 1\n',
        'name = "x"\n[tasks]\npaths = ["t"]\n[[methods]]\nname = "oracle"\noptions = ["a"]\n',
        'name = "x"\n[tasks]\npaths = "t"\n[[methods]]\nname = "oracle"\n',
        'name = "x"\n[tasks]\npaths = ["t"]\n[run]\ntrials = "two"\n[[methods]]\nname = "oracle"\n',
        'name = "x"\n[tasks]\npaths = ["t"]\n[[methods]]\nname = "oracle"\n[unknown]\na = 1\n',
        'name = "x"\n[tasks\n',
    ],
)
def test_a_malformed_experiment_file_is_a_clear_error(
    text: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "experiment.toml"
    config.write_text(text)
    _fails_clearly(["run", str(config), "--backend", "local"], capsys)


def test_a_bug_says_it_is_a_bug(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import real2simlab.cli.tasks as tasks_cli

    def broken(args: object) -> int:
        raise RuntimeError("an internal inconsistency")

    monkeypatch.setattr(tasks_cli, "_list", broken)
    assert main(["task", "list"]) == 1
    err = capsys.readouterr().err
    assert "Traceback" in err and "bug in Real2SimLab" in err and "/issues" in err


def _raising(monkeypatch: pytest.MonkeyPatch, raised: Callable[[], BaseException]) -> None:
    """Make ``r2s task list`` raise what ``raised`` makes, as a command deep inside would."""
    import real2simlab.cli.tasks as tasks_cli

    def failing(args: object) -> int:
        raise raised()

    monkeypatch.setattr(tasks_cli, "_list", failing)


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (lambda: FileNotFoundError(2, "No such file or directory", "/data/tasks"),
         "/data/tasks does not exist"),
        (lambda: FileExistsError(17, "File exists", "/data/run/output"), "already exists"),
        (lambda: IsADirectoryError(21, "Is a directory", "/data/run"), "is a directory"),
        (lambda: NotADirectoryError(20, "Not a directory", "/data/f.txt/run"), "not a directory"),
        (lambda: PermissionError(13, "Permission denied", "/data/ro/run"), "permission denied"),
        # errno's names, not Linux's numbers: macOS numbers these errors differently.
        (lambda: OSError(errno.ENAMETOOLONG, os.strerror(errno.ENAMETOOLONG), "/data/" + "a" * 260),
         "too long"),
        (lambda: OSError(errno.ELOOP, os.strerror(errno.ELOOP), "/data/loop/x"),
         "symbolic links"),
        (lambda: RuntimeError("Symlink loop from '/data/loop/x'"), "symbolic links"),
        (lambda: OSError(errno.ENOTEMPTY, os.strerror(errno.ENOTEMPTY), "/data/.run.partial-1",
                         None, "/data/run"),
         "(/data/.run.partial-1 -> /data/run)"),
    ],
    ids=["missing", "exists", "a directory", "not a directory", "no access", "name too long",
         "loop (3.13)", "loop (3.12)", "moved onto a directory that is not empty"],
)  # fmt: skip
def test_a_path_error_says_what_is_wrong_and_how_to_fix_it(
    raised: Callable[[], BaseException],
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A path that is missing, already there, a file for a directory, not accessible, too long
    for the file system, a loop of symbolic links, or a directory that is not empty where one
    is moved (another command wrote it first): one error line naming it, and a fix that suits
    it; not a hint to read a traceback, nor "this machine's system refused"."""
    _raising(monkeypatch, raised)
    err = _fails_clearly(["task", "list"], capsys)
    assert expected in err and "\n  fix: " in err, err
    assert "system refused" not in err and "bug in Real2SimLab" not in err, err
    assert len(err.strip().splitlines()) == 2, err  # the error line and its fix


def test_a_directory_made_in_a_loop_of_links_is_said_to_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Python 3.13's ``resolve()`` lets a loop of symbolic links through, and making a directory
    in it then fails with ``FileExistsError``: the error says that the path loops, with the fix
    for that, not that it "already exists"."""
    loop = tmp_path / "loop"
    loop.symlink_to("loop")  # a link that points at itself
    _raising(
        monkeypatch,
        lambda: FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(loop)),
    )
    err = _fails_clearly(["task", "list"], capsys)
    assert "symbolic links" in err and "already exists" not in err, err
    assert len(err.strip().splitlines()) == 2, err  # the error line and its fix


def test_a_program_that_is_not_installed_is_named_as_a_program(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tool the Lab runs is not on PATH: "the program ... was not found" and how to make it
    found, not "<name> does not exist" as if it were a path the user gave."""

    def failing() -> BaseException:
        try:
            subprocess.run(["r2s-no-such-tool", "-p"], check=False, stdin=subprocess.DEVNULL)
        except FileNotFoundError as error:
            return error
        raise AssertionError("r2s-no-such-tool ran")

    _raising(monkeypatch, failing)
    err = _fails_clearly(["task", "list"], capsys)
    assert "program r2s-no-such-tool was not found" in err and "PATH" in err, err
    assert "\n  fix: " in err and "does not exist" not in err, err


def test_a_working_directory_that_does_not_exist_is_not_called_a_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tool the Lab runs in a directory that is not there (``cwd``): "the working directory
    ... does not exist", not "the program <that directory> was not found"."""
    gone = tmp_path / "gone"

    def failing() -> BaseException:
        try:
            subprocess.run(["true"], cwd=gone, check=False, stdin=subprocess.DEVNULL)
        except FileNotFoundError as error:
            return error
        raise AssertionError(f"ran in {gone}, which does not exist")

    _raising(monkeypatch, failing)
    err = _fails_clearly(["task", "list"], capsys)
    assert f"the working directory {gone} does not exist" in err and "\n  fix: " in err, err
    assert "the program" not in err, err


def _hub_error(kind: str, status: int, url: str) -> BaseException:
    """A huggingface_hub error, as it raises one for the HTTP answer ``status`` to ``url``."""
    errors = pytest.importorskip("huggingface_hub.errors")
    base = next(
        c
        for c in errors.HfHubHTTPError.__mro__[1:]
        if c.__module__.split(".")[0] != "huggingface_hub"
    )  # the HTTPError of the HTTP library it is built on: requests, httpx or httpx2
    library = __import__(base.__module__.split(".")[0])
    if library.__name__ == "requests":
        response = library.Response()
        response.status_code, response.url = status, url
        response.request = library.Request("GET", url).prepare()
    else:
        response = library.Response(status, request=library.Request("GET", url))
    return getattr(errors, kind)(f"{status} Client Error. (Request ID: test)", response=response)


def _local_entry_missing() -> BaseException:
    """What huggingface_hub raises when a file is neither downloadable nor in its cache."""
    errors = pytest.importorskip("huggingface_hub.errors")
    return errors.LocalEntryNotFoundError("Cannot find the requested files in the local cache")


def _caused(error: BaseException, cause: BaseException) -> BaseException:
    error.__cause__ = cause
    return error


def _while_handling(
    handled: BaseException, raised: BaseException, *, hidden: bool = False
) -> BaseException:
    """``raised`` as an ``except`` block that handles ``handled`` raises it: with ``handled``
    as its context, or (``hidden``) ``raise ... from None``."""
    try:
        raise handled
    except BaseException:
        try:
            if hidden:
                raise raised from None
            raise raised  # no `from`: what was being handled is its context
        except BaseException as error:
            return error


PIQ = "https://huggingface.co/datasets/Anates-Labs-Research/Physics-IQ-Verified"
FORCED = "Force download failed due to the above error."  # huggingface_hub 2's ValueError


@pytest.mark.parametrize(
    ("raised", "expected", "fix"),
    [
        (lambda: _caused(ValueError(FORCED), _hub_error("GatedRepoError", 401,
                                                        f"{PIQ}/resolve/b65f/x.mp4")),
         "Anates-Labs-Research/Physics-IQ-Verified", f"accept its terms at {PIQ}"),
        (lambda: _hub_error("RepositoryNotFoundError", 401,
                            "https://huggingface.co/api/datasets/someone/gone/revision/abc"),
         "someone/gone", "hf auth login"),
        (lambda: _caused(ValueError(FORCED), __import__("socket").gaierror(
            -3, "Temporary failure in name resolution")),
         "a download failed: [Errno -3] Temporary failure in name resolution", "network"),
        (lambda: _caused(_local_entry_missing(), ConnectionRefusedError(111, "Connection refused")),
         "a download failed: [Errno 111] Connection refused", "network"),
        (lambda: pytest.importorskip("httpx2").ProxyError("407 Proxy Authentication Required"),
         "a download failed: 407 Proxy Authentication Required", "proxy settings"),
        (_local_entry_missing, "which HF_HUB_OFFLINE turns off", "unset HF_HUB_OFFLINE"),
    ],
    ids=["gated (401)", "no such repository", "no network (ValueError)",
         "no network (FileNotFoundError)", "a proxy that refuses", "offline mode"],
)  # fmt: skip
def test_a_refused_or_failed_download_is_explained_by_its_cause(
    raised: Callable[[], BaseException],
    expected: str,
    fix: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Hugging Face's library wraps a failed download in a ``ValueError`` ("... the above
    error", with nothing above) or a ``FileNotFoundError``; a gated or missing repository is
    not a network failure. Each says what happened, with the fix that suits it."""
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    if "HF_HUB_OFFLINE" in expected:  # downloads turned off, the file not in the cache
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    else:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    _raising(monkeypatch, raised)
    err = _fails_clearly(["task", "list"], capsys)
    lines = err.strip().splitlines()
    assert expected in lines[0] and "\n  fix: " in err and fix in err, err
    assert "above error" not in err and "system refused" not in err, err
    if "download failed" not in expected and "OFFLINE" not in expected:  # a refusal
        assert "network" not in err and "HF_TOKEN" in err, err  # logging in helps, not that


@pytest.mark.parametrize(
    "raised",
    [
        lambda: _while_handling(_hub_error("GatedRepoError", 401, f"{PIQ}/resolve/b65f/x.mp4"),
                                KeyError("tracks")),
        lambda: _while_handling(_local_entry_missing(), KeyError("tracks")),
        lambda: _while_handling(ConnectionResetError(104, "Connection reset by peer"),
                                TypeError("unsupported operand type(s) for +: 'int' and 'str'")),
    ],
    ids=["KeyError, a gated repository", "KeyError, not in the cache", "TypeError, no network"],
)  # fmt: skip
def test_a_bug_raised_while_a_failed_download_is_handled_is_still_a_bug(
    raised: Callable[[], BaseException],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A mistake in the code (a ``KeyError``, a ``TypeError``) raised in an ``except`` block
    that handles a failed download is the bug it is, reported as one with its traceback, not
    told as that download ("gated: log in", "check the network") for the user to fix."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    _raising(monkeypatch, raised)
    assert main(["task", "list"]) == 1
    err = capsys.readouterr().err
    assert "Traceback" in err and "bug in Real2SimLab" in err, err
    assert not any(line.startswith(("error:", "  fix:")) for line in err.splitlines()), err


def test_an_input_error_that_is_also_an_index_error_is_told_as_an_input_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """numpy's ``AxisError`` is an ``IndexError`` and a ``ValueError``: it is still told as the
    input error it also is (one line and a fix), not reported as a bug."""
    exceptions = pytest.importorskip("numpy.exceptions")
    _raising(monkeypatch, lambda: exceptions.AxisError(2, 2))
    err = _fails_clearly(["task", "list"], capsys)
    assert "axis 2 is out of bounds" in err and "\n  fix: " in err, err


def test_an_error_raised_from_none_is_told_by_its_own_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``raise ValueError(...) from None`` in an ``except`` block that handles a failed
    connection: the user reads the error the code raised, not the connection it hid."""
    refused = ConnectionRefusedError(111, "Connection refused")
    _raising(monkeypatch, lambda: _while_handling(refused, ValueError("bad --seed value"),
                                                  hidden=True))  # fmt: skip
    err = _fails_clearly(["task", "list"], capsys)
    assert err.strip().splitlines()[0] == "error: bad --seed value", err
    assert "\n  fix: " in err and "download" not in err and "network" not in err, err


@pytest.mark.parametrize("trouble", ["loop", "long"])
@pytest.mark.parametrize(
    "command",
    ["eval", "twin run", "twin check", "train export", "train reward", "export harbor",
     "bench build"],
)  # fmt: skip
def test_an_output_path_the_file_system_cannot_take_is_a_clear_error(
    command: str,
    trouble: str,
    bouncing_task: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--out`` inside a loop of symbolic links, or with a name longer than a file name may be:
    one error line saying so and a fix (another path), never a traceback asking for a bug
    report, nor "this machine's system refused the operation; run the same command again"."""
    (tmp_path / "loop").symlink_to("loop")  # a link that points at itself
    out = tmp_path / "loop" / "x" if trouble == "loop" else tmp_path / ("a" * 260)
    generator = str(bouncing_task / "hidden" / "generator")
    argv = {
        "eval": _eval(bouncing_task, out),
        "twin run": ["twin", "run", generator, "--backend", "local", "--out", str(out)],
        "twin check": ["twin", "check", generator, "--backend", "local", "--out", str(out)],
        "train export": ["train", "export", "--tasks", str(bouncing_task), "--allow-eval-tasks",
                         "--out", str(out)],
        "train reward": ["train", "reward", "--task", str(bouncing_task), "--solution",
                         f"{generator}/program", "--backend", "local", "--work", str(out)],
        "export harbor": ["export", "harbor", str(bouncing_task), "--out", str(out)],
        "bench build": ["bench", "build", "synthetic", "--select", "bouncing_ball", "--tier",
                        "calibrated", "--out", str(out)],
    }[command]  # fmt: skip
    err = _fails_clearly(argv, capsys)
    assert ("symbolic links" if trouble == "loop" else "too long") in err, err
    assert "\n  fix: " in err and ("loop" if trouble == "loop" else "shorter") in err, err
    assert "system refused" not in err and "bug in Real2SimLab" not in err, err


@pytest.mark.skipif(os.geteuid() == 0, reason="root may write into any directory")
def test_a_directory_that_cannot_be_written_says_how_to_fix_it(
    bouncing_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An --out in a directory without write permission, and the Lab's cache under a HOME that
    cannot be written (nothing else names a cache): each error ends with a fix, the cache's
    naming R2S_CACHE_DIR, rather than with a hint to read a traceback."""
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        err = _fails_clearly(_eval(bouncing_task, locked / "run"), capsys)
        assert "permission denied" in err and "\n  fix: " in err, err
        assert "R2S_DEBUG" not in err, err
        monkeypatch.delenv("R2S_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(locked))
        argv = ["train", "export", "--tasks", str(bouncing_task), "--allow-eval-tasks",
                "--against", str(bouncing_task), "--out", str(tmp_path / "data")]  # fmt: skip
        err = _fails_clearly(argv, capsys)
        assert "permission denied" in err and "R2S_CACHE_DIR" in err.split("fix:")[-1], err
    finally:
        locked.chmod(0o755)


def test_one_name_for_an_option_that_takes_several_is_a_list_of_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--option tiers=calibrated`` is ``["calibrated"]``, not the letters of the word."""
    argv = ["bench", "build", "synthetic", "--select", "bouncing_ball", "--option"]
    assert main([*argv, "tiers=calibrated", "--out", str(tmp_path / "tasks")]) == 0, (
        capsys.readouterr().err
    )
    assert sorted(p.name for p in (tmp_path / "tasks").rglob("task.toml"))


@pytest.mark.parametrize("name", ["fourdcodebench", "physcodegym", "physicsiq", "synthetic"])
def test_every_list_option_of_an_adapter_takes_one_name(name: str) -> None:
    import inspect

    from real2simlab.benchmarks.base import get_benchmark
    from real2simlab.options import checked_options

    build = get_benchmark(name).build
    listed = [
        key
        for key, parameter in inspect.signature(build).parameters.items()
        if "Sequence[" in str(parameter.annotation) and key != "select"
    ]
    assert listed, f"{name} has no list options to check"
    assert checked_options(build, dict.fromkeys(listed, "one"), name) == {
        key: ["one"] for key in listed
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["twin", "run", "x", "--seed", "-1"],
        ["twin", "check", "x", "--seed", "one"],
        ["train", "tasks", "--count", "0", "--out", "x"],
        ["train", "reward", "--task", "x", "--solution", "y", "--seed", "-2"],
        ["eval", "--tasks", "x", "--method", "static", "--seeds", "0", "-1"],
        ["eval", "--tasks", "x", "--method", "static", "--trials", "0"],
        ["eval", "--tasks", "x", "--method", "static", "--concurrency", "1.5"],
        ["eval", "--tasks", "x", "--method", "static", "--max-agent-runs", "-1"],
        # a seed a twin cannot be given: an evaluation stopped on it after its builds, and
        # `r2s train tasks` named its samples with it ("File name too long")
        ["eval", "--tasks", "x", "--method", "static", "--seeds", "9" * 260],
        ["twin", "run", "x", "--seed", str(2**32)],
        ["train", "tasks", "--seed", "9" * 260, "--out", "x"],
        ["train", "reward", "--task", "x", "--solution", "y", "--seed", str(2**32)],
        ["train", "fit", "x", "--data", "y", "--out", "z", "--seed", str(2**40)],
    ],
    ids=lambda argv: " ".join(a[:12] for a in argv[-2:]),
)
def test_a_bad_number_stops_at_the_command_line(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    err = capsys.readouterr().err
    assert stopped.value.code == 2 and "error: argument --" in err, err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "title", ['Drop a "red" ball', "back\\slash and\ttab", "Bällfall — \u7403 ünïcode ✓"]
)
def test_any_title_makes_a_task_file_that_loads(
    title: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from real2simlab.task import load_task

    assert main(["task", "new", "--id", "mybench/drop", "--title", title, str(tmp_path / "t")]) == 0
    assert load_task(tmp_path / "t").config.title == title


@pytest.mark.parametrize("bad", ["id", "video"])
def test_a_failed_task_new_leaves_nothing_behind(
    bad: str, bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same command, corrected, must then work: no half-made directory in its way."""
    video = next(bouncing_task.glob("visible/**/*.mp4"))
    target = tmp_path / "new"
    wrong = {"id": ["--id", "My Task", "--video", str(video)],
             "video": ["--id", "a/b", "--video", str(bouncing_task / "task.toml")]}[bad]  # fmt: skip
    _fails_clearly(["task", "new", *wrong, str(target)], capsys)
    assert not target.exists()
    assert main(["task", "new", "--id", "a/b", "--video", str(video), str(target)]) == 0


REFLECTION = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]]  # det -1: a mirror
NEARLY_RIGID = [[1.00025, 0, 0, 0], [0, 1.00025, 0, 0], [0, 0, 1.00025, 1], [0, 0, 0, 1]]


@pytest.mark.parametrize("mine", [False, True], ids=["new", "an empty directory of mine"])
@pytest.mark.parametrize(
    ("given", "named"),
    [
        (["--object", "red ball=0.067"], "object name 'red ball'"),
        (["--object", "ball.v2=0.067"], "object name 'ball.v2'"),
        (["--extrinsics", "{reflection}"], "extrinsics must be a rigid transform"),
        (["--extrinsics", "{nearly_rigid}"], "extrinsics must be a rigid transform"),
        (["--capture-id", ""], "capture id must not be empty"),
    ],
    ids=["name with a space", "name with a dot", "a mirror", "not quite a rotation", "empty id"],
)
def test_a_capture_argument_it_cannot_hold_is_refused_before_anything_is_written(
    given: list[str],
    named: str,
    mine: bool,
    bouncing_task: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Whatever a capture cannot hold (an object name, extrinsics that are no rotation, an empty
    id) is refused in one line saying what is wrong, before the video is copied: the corrected
    command then works, and an empty directory of the user's stays empty."""
    video = next(bouncing_task.glob("visible/**/*.mp4"))
    matrices = {"reflection": REFLECTION, "nearly_rigid": NEARLY_RIGID}
    for name, matrix in matrices.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(matrix))
    values = {name: str(tmp_path / f"{name}.json") for name in matrices}
    out = tmp_path / "capture"
    if mine:
        out.mkdir()
    argv = ["capture", "from-video", str(video), str(out), "--fov", "60", "--capture-id", "lab/c"]
    err = _fails_clearly([*argv, *(part.format(**values) for part in given)], capsys)
    assert named in err and "validation error" not in err and "pydantic" not in err, err
    assert len(err.strip().splitlines()) <= 2, err
    assert (list(out.iterdir()) == []) if mine else not out.exists()
    assert main(argv) == 0, capsys.readouterr().err  # corrected: nothing in its way
    assert (out / "capture.json").is_file()


def test_a_video_that_cannot_be_read_says_how_to_fix_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file that is not a video ffmpeg can read, given as the recording: one error saying
    that ffprobe cannot read it, and a fix (check the file), not the error alone."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not a video\n")
    for argv in (
        ["capture", "from-video", str(video), str(tmp_path / "capture"), "--fov", "60",
         "--capture-id", "lab/c"],
        ["task", "new", "--id", "lab/c", "--video", str(video), str(tmp_path / "task")],
    ):  # fmt: skip
        err = _fails_clearly(argv, capsys)
        assert "ffprobe cannot read" in err, err
        assert "\n  fix: check that the file is a video" in err, err
    assert [p.name for p in tmp_path.iterdir()] == ["clip.mp4"]  # nothing written


def test_tasks_left_out_by_include_or_exclude_do_not_stop_a_run(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A task directory may hold a task still being written (``r2s task new`` makes one without
    its reference). --include, or include/exclude in an experiment file, choose tasks by id:
    one they leave out is not in the run, so it does not stop it. One that is chosen (or whose
    id cannot be read) is refused with a fix."""
    tasks = tmp_path / "tasks"
    shutil.copytree(bouncing_task, tasks / "bouncing_ball")
    assert main(["task", "new", "--id", "mybench/drop_01", str(tasks / "drop_01")]) == 0
    capsys.readouterr()
    assert main(_eval(tasks, tmp_path / "inc", "--include", "synthetic/*")) == 0, (
        capsys.readouterr().err
    )
    experiment = tmp_path / "experiment.toml"
    experiment.write_text(
        f'name = "skip-skeleton"\n[tasks]\npaths = ["{tasks}"]\nexclude = ["mybench/*"]\n'
        '[[methods]]\nname = "static"\n[run]\nbackend = "local"\n'
    )
    assert main(["run", str(experiment), "--out", str(tmp_path / "exc")]) == 0, (
        capsys.readouterr().err
    )
    err = _fails_clearly(_eval(tasks, tmp_path / "all"), capsys)
    assert "drop_01 is invalid" in err and "\n  fix: " in err and "--include" in err, err
    (tasks / "broken").mkdir()
    (tasks / "broken" / "task.toml").write_text("id = [\n")  # its id cannot be read
    err = _fails_clearly(_eval(tasks, tmp_path / "broken", "--include", "synthetic/*"), capsys)
    assert "broken is invalid" in err and "\n  fix: " in err, err


def test_two_twin_runs_into_one_directory_at_once_refuse_one_with_a_fix(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """``r2s twin run --out D`` twice at once: one runs, the other is refused as the same
    command run second is ("run directory already exists", with a fix), never with an error
    about a directory inside the first one's run (``D/output already exists``)."""
    import threading

    from real2simlab.errors import OutputExistsError
    from real2simlab.runner import run_twin
    from real2simlab.sandbox.base import ResolvedImage
    from real2simlab.sandbox.local import LocalBackend
    from real2simlab.twin import load_twin

    both_checked = threading.Barrier(2, timeout=60)

    class Slow(LocalBackend):
        def resolve_image(self, reference: str) -> ResolvedImage:
            both_checked.wait()  # both have seen that the run directory does not exist yet
            return super().resolve_image(reference)

    out = tmp_path / "run"
    outcomes: list[object] = []

    def run() -> None:
        twin = load_twin(bouncing_task / "hidden" / "generator")
        try:
            outcomes.append(run_twin(twin, out, backend=Slow(quiet=True)))
        except Exception as error:
            outcomes.append(error)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=300)
    errors = [o for o in outcomes if isinstance(o, Exception)]
    assert len(outcomes) == 2 and len(errors) == 1, outcomes
    assert isinstance(errors[0], OutputExistsError) and errors[0].fix, repr(errors[0])
    assert "run directory already exists" in str(errors[0]), errors[0]
    (done,) = [o for o in outcomes if not isinstance(o, Exception)]
    assert getattr(done, "succeeded", False), done


def test_two_harbor_exports_into_one_directory_at_once_refuse_one_with_a_fix(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``r2s export harbor TASK --out D`` twice at once: one exports, the other is refused as
    the same command run second is ("destination already exists", with a fix), never with
    "Directory not empty" about its own staging directory and a fix that blames the machine;
    no staging directory is left behind."""
    import threading

    from real2simlab.bridges import harbor
    from real2simlab.errors import OutputExistsError

    both_checked = threading.Barrier(2, timeout=60)
    write_task = harbor._write_task

    def slow(*args: object) -> None:
        both_checked.wait()  # both have seen that the destination does not exist yet
        write_task(*args)

    monkeypatch.setattr(harbor, "_write_task", slow)
    out = tmp_path / "exports" / "harbor"
    outcomes: list[object] = []

    def export() -> None:
        try:
            outcomes.append(harbor.export_task(bouncing_task, out))
        except Exception as error:
            outcomes.append(error)

    threads = [threading.Thread(target=export) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=300)
    errors = [o for o in outcomes if isinstance(o, Exception)]
    assert len(outcomes) == 2 and len(errors) == 1, outcomes
    assert isinstance(errors[0], OutputExistsError) and errors[0].fix, repr(errors[0])
    assert f"destination already exists: {out}" in str(errors[0]), errors[0]
    assert (out / "task.toml").is_file() and (out / "tests" / "test.sh").is_file()
    assert [p.name for p in out.parent.iterdir()] == ["harbor"]  # no staging directory left


def _eval(task: Path, out: Path, *more: str) -> list[str]:
    return ["eval", "--tasks", str(task), "--method", "static", "--backend", "local",
            "--out", str(out), *more]  # fmt: skip


def test_an_evaluation_never_writes_into_a_directory_of_other_files(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mine = tmp_path / "my-results"
    mine.mkdir()
    (mine / "results.json").write_text("my own measurements\n")
    assert "--out" in _fails_clearly(_eval(bouncing_task, mine), capsys)
    assert [p.name for p in mine.iterdir()] == ["results.json"]
    assert (mine / "results.json").read_text() == "my own measurements\n"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(_eval(bouncing_task, empty)) == 0  # an empty directory is fine
    assert main(_eval(bouncing_task, empty)) == 0  # and so is resuming the run in it


def test_a_run_directory_in_use_is_refused(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import fcntl

    out = tmp_path / "run"
    assert main(_eval(bouncing_task, out)) == 0
    capsys.readouterr()
    with (out / "state" / "lock").open("a+") as held:  # another command holds it
        fcntl.flock(held, fcntl.LOCK_EX)
        err = _fails_clearly(_eval(bouncing_task, out), capsys)
        assert "another r2s command is using" in err, err
    assert main(_eval(bouncing_task, out)) == 0  # free again


def test_a_run_that_cannot_start_leaves_no_directory(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "never"
    argv = ["eval", "--tasks", str(bouncing_task), "--method", "nomethod", "--backend", "local",
            "--out", str(out)]  # fmt: skip
    _fails_clearly(argv, capsys)
    assert not out.exists()


def _open_under(root: Path) -> list[str]:
    """The files under ``root`` this process has open (``/proc/self/fd``)."""
    found: list[str] = []
    for descriptor in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:  # closed meanwhile (the descriptor listdir itself used)
            continue
        if target.startswith(f"{root}{os.sep}"):
            found.append(target)
    return sorted(found)


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="reads /proc/self/fd")
def test_a_run_that_cannot_start_closes_its_lock_before_removing_its_directory(
    bouncing_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """On NFS a file that is still open is not removed but renamed (``.nfsXXXX``) until it is
    closed, so a run directory removed while its lock is open stays, as an empty ``state/``
    that the same command then refuses as someone else's. Nothing under the directory is open
    when it is removed (a local disk cannot show the difference, so this checks the order)."""
    out = tmp_path / "never"
    open_then: list[list[str]] = []
    remove = shutil.rmtree

    def rmtree(path: str | os.PathLike[str], *args: object, **kwargs: object) -> None:
        if Path(path) == out or Path(path).name.startswith(".never.trash-"):  # moved aside
            open_then.append(_open_under(Path(path)))
        remove(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(shutil, "rmtree", rmtree)
    argv = ["eval", "--tasks", str(bouncing_task), "--method", "nomethod", "--backend", "local",
            "--out", str(out)]  # fmt: skip
    _fails_clearly(argv, capsys)
    assert open_then == [[]] and not out.exists() and not list(tmp_path.glob(".never.*"))


def test_a_run_that_cannot_start_never_removes_a_run_begun_there_meanwhile(
    bouncing_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A run that cannot start removes the directory it created, and another command given
    the same ``--out`` may begin there the moment the first one lets go of it. What the first
    one removes is only what it created, moved aside while it still held the directory: the
    other command's run is left whole."""
    out = tmp_path / "run"
    remove = shutil.rmtree
    others: list[int] = []

    def rmtree(path: str | os.PathLike[str], *args: object, **kwargs: object) -> None:
        if not others:  # the removal by the run that cannot start: another command runs first
            others.append(-1)
            others[0] = main(["eval", "--tasks", str(bouncing_task), "--method", "static",
                              "--backend", "local", "--out", str(out)])  # fmt: skip
        remove(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(shutil, "rmtree", rmtree)
    argv = ["eval", "--tasks", str(bouncing_task), "--method", "nomethod", "--backend", "local",
            "--out", str(out)]  # fmt: skip
    _fails_clearly(argv, capsys)
    assert others == [0], capsys.readouterr().err
    assert (out / "results.json").is_file(), sorted(p.name for p in tmp_path.iterdir())
    assert not list(tmp_path.glob(".run.trash-*"))


def test_a_typo_in_backend_options_is_a_clear_error(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    experiment = tmp_path / "experiment.toml"
    experiment.write_text(
        f'name = "typo"\n[tasks]\npaths = ["{bouncing_task}"]\n[[methods]]\nname = "static"\n'
        '[run]\nbackend = "local"\nbackend_options = { xyz = 1 }\n'
    )
    assert "xyz" in _fails_clearly(["run", str(experiment), "--out", str(tmp_path / "r")], capsys)


def _error_and_fix(text: str) -> tuple[str, str]:
    """The two lines a usage error ends in: ``error: ...`` and ``  fix: ...``."""
    lines = text.strip().splitlines()
    assert len(lines) == 2, text
    assert lines[0].startswith("error: ") and lines[1].startswith("  fix: "), text
    assert "R2SError" not in text, text  # the class of a bare error, said instead of a fix
    return lines[0], lines[1]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('harbor:agent_env={"A":"$KEY_A","B":"$KEY_B"}',
         {"name": "harbor", "agent_env": {"A": "$KEY_A", "B": "$KEY_B"}}),
        ('fake_agent:allowlist=["a.example","b.example"],cpus=2',
         {"name": "fake_agent", "allowlist": ["a.example", "b.example"], "cpus": 2}),
        ('fake_agent:extra_args=["--a","--b"]', {"name": "fake_agent", "extra_args": ["--a", "--b"]}),
        ('program:path=p,entrypoint=["python","./run.py"]',
         {"name": "program", "path": "p", "entrypoint": ["python", "./run.py"]}),
        ('mini_swe:model_kwargs={"top_p":0.9,"seed":1}',
         {"name": "mini_swe", "model_kwargs": {"top_p": 0.9, "seed": 1}}),
        ('fake_agent:gpus="0,1"', {"name": "fake_agent", "gpus": "0,1"}),
        ('program:path="prog/with,comma"', {"name": "program", "path": "prog/with,comma"}),
        ('program:path="1"', {"name": "program", "path": "1"}),
        ('fast=codex:model="a,b",reasoning_effort=low',
         {"name": "codex", "label": "fast", "model": "a,b", "reasoning_effort": "low"}),
        ('fake_agent: allowlist = ["a", "b"] , cpus=2',
         {"name": "fake_agent", "allowlist": ["a", "b"], "cpus": 2}),
        ("codex:model=gpt-5-codex,reasoning_effort=medium,",
         {"name": "codex", "model": "gpt-5-codex", "reasoning_effort": "medium"}),
        ("program:path=[x]/y", {"name": "program", "path": "[x]/y"}),  # only starts like JSON
        ("oracle", {"name": "oracle"}),
    ],
)  # fmt: skip
def test_a_method_option_keeps_the_commas_of_its_json_value(
    text: str, expected: dict[str, object]
) -> None:
    """``--method NAME:key=value,...`` splits at the commas between options, not at those of
    a JSON list, table or quoted text (A2-06): several ``agent_env`` entries, an allowlist of
    two domains, ``gpus="0,1"`` and a path with a comma can all be given."""
    from real2simlab.cli.common import parse_method

    assert parse_method(text) == expected


def test_a_method_whose_options_hold_commas_runs_from_the_command_line(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    program = tmp_path / "prog" / "with,comma"
    shutil.copytree(bouncing_task / "hidden" / "generator" / "program", program)
    out = tmp_path / "run"
    method = f'program:path="{program}",entrypoint=["python","./run.py"]'
    argv = ["eval", "--tasks", str(bouncing_task), "--method", method, "--backend", "local",
            "--out", str(out)]  # fmt: skip
    assert main(argv) == 0, capsys.readouterr().err
    (entry,) = json.loads((out / "experiment.json").read_text())["config"]["methods"]
    assert entry["options"] == {"path": str(program), "entrypoint": ["python", "./run.py"]}


@pytest.mark.parametrize(
    ("method", "said", "suggested"),
    [
        ("fake_agent:gpus=0,1", "'1'", """--method 'fake_agent:gpus="0,1"'"""),
        ("program:path=prog/with,comma", "'comma'", """--method 'program:path="prog/with,comma"'"""),
        ("fake_agent:allowlist=[a.example,b.example]", "not valid JSON", None),
        ("codex:reasoning_effort", "'reasoning_effort'", None),
        ("codex:=low", "'=low'", None),
    ],
)  # fmt: skip
def test_a_method_option_that_is_not_key_value_says_how_to_write_it(
    method: str,
    said: str,
    suggested: str | None,
    bouncing_task: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A comma outside a JSON value separates options: what follows it is not an option. The
    error names the ``--method`` and the piece, and the fix says how to write a value that
    holds a comma (as JSON), not ``R2SError: expected key=value, got '1'``."""
    out = tmp_path / "run"
    argv = ["eval", "--tasks", str(bouncing_task), "--method", method, "--backend", "local",
            "--out", str(out)]  # fmt: skip
    error, fix = _error_and_fix(_fails_clearly(argv, capsys))
    assert f"--method {method!r}" in error and said in error, error
    assert "JSON" in fix and "experiment file" in fix, fix
    if suggested is not None:
        assert suggested in fix, fix
    assert not out.exists()


def test_train_fit_reads_its_method_as_eval_does(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``r2s train fit METHOD`` takes the same ``NAME:key=value,...``: a JSON list is one
    value (here an option ``nearest_twin`` does not take, refused by its name), and a piece
    that is not an option is the same error with its fix."""
    argv = ["--data", str(tmp_path / "data"), "--out", str(tmp_path / "fit")]
    text = _fails_clearly(["train", "fit", "nearest_twin:k=1,foo=[1,2]", *argv], capsys)
    assert "foo" in text and "'2]'" not in text and "R2SError" not in text, text
    error, fix = _error_and_fix(
        _fails_clearly(["train", "fit", "nearest_twin:k=1,2", *argv], capsys)
    )
    assert "'2' is not an option" in error and "JSON" in fix, error + fix


def test_a_benchmark_option_without_a_value_says_how_to_write_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["bench", "build", "synthetic", "--option", "colour", "--out", str(tmp_path / "t")]
    error, fix = _error_and_fix(_fails_clearly(argv, capsys))
    assert "--option 'colour'" in error and "--option NAME=VALUE" in fix, error + fix
    assert not (tmp_path / "t").exists()


def test_no_error_is_raised_bare() -> None:
    """``R2SError`` itself prints as ``error: R2SError: ...`` without a fix, and
    ``SystemExit("...")`` as a bare line without ``error:``: what the Lab raises on purpose is
    a subclass that says what to do (``UsageError``, ``SetupError``, ... with a ``fix``)."""
    import ast

    root = Path(real2simlab.__file__).parent
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                continue
            called = node.exc.func
            name = getattr(called, "id", None) or getattr(called, "attr", None)
            text = node.exc.args and isinstance(node.exc.args[0], ast.Constant | ast.JoinedStr)
            if name == "R2SError" or (name == "SystemExit" and text):
                found.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not found, found


def _experiment_named(name: str, task: Path, directory: Path) -> Path:
    path = directory / "exp.toml"
    path.write_text(
        f"name = {json.dumps(name)}\n[tasks]\npaths = [{json.dumps(str(task))}]\n"
        '[[methods]]\nname = "oracle"\n[run]\nbackend = "local"\n',
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "name",
    ["../../escaped", "{tmp}/elsewhere", "team/exp1", "..", ".", " ", "a\nb", "n" * 300],
    ids=["above", "absolute", "nested", "parent", "itself", "blank", "control", "too long"],
)
def test_an_experiment_name_that_is_not_one_directory_name_needs_out(
    name: str,
    bouncing_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``r2s run`` without ``--out`` writes into ``runs/<name>``: a name that is not one
    directory name (a path out of ``runs/`` or anywhere, ``..``) is refused before anything is
    written, as a method label would be (A2-07); with ``--out`` it is only a name."""
    work = tmp_path / "proj" / "sub"
    work.mkdir(parents=True)
    monkeypatch.chdir(work)
    experiment = _experiment_named(name.format(tmp=tmp_path), bouncing_task, work)
    before = sorted(tmp_path.rglob("*"))
    error, fix = _error_and_fix(_fails_clearly(["run", str(experiment)], capsys))
    assert "experiment name" in error and "runs/<name>" in error, error
    assert "--out" in fix, fix
    assert sorted(tmp_path.rglob("*")) == before  # nothing written, here or anywhere
    if name == "../../escaped":
        out = tmp_path / "chosen"
        assert main(["run", str(experiment), "--out", str(out)]) == 0, capsys.readouterr().err
        assert (out / "results.json").is_file() and not (tmp_path / "escaped").exists()


@pytest.mark.parametrize(
    "name",
    [
        "\u5f39\u8df3\u7403 pilot v1.0",
        "\u5f39\u8df3\u7403\u3000\u5b9e\u9a8c",
        "\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645",
    ],
    ids=["spaces", "a full-width space", "a zero-width non-joiner"],
)
def test_an_experiment_name_of_any_words_is_its_run_directory(
    name: str,
    bouncing_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Spaces and letters of any script make one directory name: ``runs/<name>`` as before,
    also with a space other than the ASCII one (U+3000 between words in Chinese) or a
    character that joins or parts letters (U+200C in Persian): only control characters are no
    part of a directory name."""
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(_experiment_named(name, bouncing_task, tmp_path))]) == 0, (
        capsys.readouterr().err
    )
    assert (tmp_path / "runs" / name / "results.json").is_file()


def test_eval_without_tasks_says_how_to_give_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Neither ``--tasks`` nor ``--benchmark``: an ``error:`` line with its fix, as every
    other usage error, not a bare line (A2-08)."""
    out = tmp_path / "run"
    argv = ["eval", "--method", "oracle", "--backend", "local", "--out", str(out)]
    try:
        text = _fails_clearly(argv, capsys)
    except SystemExit as stopped:
        pytest.fail(f"r2s eval ended in SystemExit({stopped.code!r}), not an error with a fix")
    error, fix = _error_and_fix(text)
    assert "--tasks" in error and "--benchmark" in error, error
    assert "--tasks DIR" in fix and "--benchmark NAME" in fix, fix
    assert not out.exists()


OPTIONS_READ = {"physicsiq": ["release"]}
"""Options an adapter's ``build`` reads from its ``**options`` rather than naming them."""


def _benchmark_options(name: str) -> list[str]:
    import inspect

    from real2simlab.benchmarks.base import get_benchmark

    parameters = inspect.signature(get_benchmark(name).build).parameters.values()
    named = [p.name for p in parameters if p.kind is p.KEYWORD_ONLY and p.name != "select"]
    return sorted([*named, *OPTIONS_READ.get(name, [])])


def _benchmarks() -> list[str]:
    from real2simlab.benchmarks.base import benchmark_names

    return sorted(benchmark_names())


@pytest.mark.parametrize("name", _benchmarks())
def test_an_unknown_benchmark_option_names_the_options_it_takes(
    name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each benchmark checks its own options (its ``build`` takes ``**options``): an unknown
    name is refused saying what it takes, with a fix, as the shared check does for a backend
    (A8-03), before anything is downloaded or written."""
    out = tmp_path / "t"
    argv = ["bench", "build", name, "--option", "no_such_option=1", "--out", str(out)]
    error, _ = _error_and_fix(_fails_clearly(argv, capsys))
    assert error.startswith("error: unknown options for") and "no_such_option" in error, error
    taken = error.partition("it takes:")[2]
    assert all(repr(option) in taken for option in _benchmark_options(name)), error
    assert not out.exists()


@pytest.mark.parametrize(
    ("name", "typo", "meant"),
    [
        ("physicsiq", "relase", "release"),
        ("physicsiq", "varaints", "variants"),
        ("fourdcodebench", "varaints", "variants"),
        ("synthetic", "tier", "tiers"),
        ("physcodegym", "variant", "variants"),
    ],
)
def test_a_misspelled_benchmark_option_is_matched_to_the_one_meant(
    name: str, typo: str, meant: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["bench", "build", name, "--option", f"{typo}=x", "--out", str(tmp_path / "t")]
    error, fix = _error_and_fix(_fails_clearly(argv, capsys))
    assert repr(meant) in error.partition("it takes:")[2], error
    assert f"{meant} instead of {typo}" in fix, fix


@pytest.mark.parametrize(
    "argv",
    [
        ["eval", "--tasks", "{task}", "--method", "oracle", "--backend", "local",
         "--out", "{task}/run"],
        ["eval", "--tasks", "{task}", "--method", "program:path={program}", "--backend", "local",
         "--out", "{program}/run"],
        ["twin", "check", "{twin}", "--backend", "local", "--out", "{twin}/check"],
        ["twin", "check", "{twin}", "--backend", "local", "--out", "{task}"],
        ["twin", "run", "{twin}", "--backend", "local", "--out", "{program}/out"],
        ["export", "harbor", "{task}", "--out", "{task}/harbor"],
        ["train", "reward", "--task", "{task}", "--solution", "{program}", "--backend", "local",
         "--work", "{program}/work"],
    ],
    ids=lambda argv: " ".join(argv[:2]) + " -> " + argv[-1].strip("{}").replace("}/", "/"),
)  # fmt: skip
def test_an_output_inside_its_input_is_refused(
    argv: list[str], bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Writing an output inside what is read (a run in its task, an export inside what it
    copies) would change the input, or copy the output into itself: refused, nothing written."""
    task = Path(shutil.copytree(bouncing_task, tmp_path / "task"))
    twin = task / "hidden" / "generator"
    values = {"task": str(task), "twin": str(twin), "program": str(twin / "program")}
    before = sorted(str(p.relative_to(task)) for p in task.rglob("*"))
    err = _fails_clearly([part.format(**values) for part in argv], capsys)
    assert "inside" in err or "overlaps" in err, err
    assert sorted(str(p.relative_to(task)) for p in task.rglob("*")) == before


@pytest.mark.parametrize(
    "argv",
    [
        ["export", "harbor", "{task}", "--out", "{out}"],
        ["train", "export", "--tasks", "{task}", "--allow-eval-tasks", "--out", "{out}"],
    ],
    ids=lambda argv: " ".join(argv[:2]),
)
@pytest.mark.parametrize("mistake", ["link", "copy", "statement"])
def test_a_task_that_shows_hidden_data_is_not_exported(
    argv: list[str],
    mistake: str,
    bouncing_task: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every export validates the task as ``r2s eval`` does: hidden data in ``visible/`` (a
    link into ``hidden/``, a copy of a hidden file) or as the statement (``instruction.md``, a
    copy of a hidden file) ends in one error naming the file, with the fix, and nothing is
    written."""
    task = Path(shutil.copytree(bouncing_task, tmp_path / "task"))
    (task / "hidden" / "decoy.txt").write_text("R2S-TEST-DECOY cli export")
    shown = "visible/notes.txt"
    if mistake == "link":
        (task / "visible" / "notes.txt").symlink_to(Path("..") / "hidden" / "decoy.txt")
    elif mistake == "copy":
        shutil.copyfile(task / "hidden" / "decoy.txt", task / "visible" / "notes.txt")
    else:
        shown = "instruction.md"
        shutil.copyfile(task / "hidden" / "decoy.txt", task / shown)
    out = tmp_path / "out"
    err = _fails_clearly([part.format(task=task, out=out) for part in argv], capsys)
    assert shown in err and "fix: " in err and "r2s task validate" in err, err
    assert sorted(path.name for path in tmp_path.iterdir()) == ["task"]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["twin", "run", "{twin}", "--timeout", "0"], "must be above 0"),
        (["twin", "run", "{twin}", "--timeout", "nan"], "a finite number"),
        (["export", "harbor", "{task}", "--timeout", "-60"], "must be above 0"),
        (["train", "export", "--tasks", "{task}", "--min-reward", "inf"], "a finite number"),
        (["eval", "--tasks", "{task}", "--method", "codex:time_budget_s=0"], "greater than 0"),
        (["eval", "--tasks", "{task}", "--method", "codex:memory_mb=-1"], "greater than 0"),
        (["eval", "--tasks", "{task}", "--method", "mini_swe:step_limit=-1"], "greater than or"),
        (["eval", "--tasks", "{task}", "--method", "fake_agent:cpus=1e309"], "a finite number"),
        (["eval", "--tasks", "{task}", "--method", "harbor:cpus=inf"], "a finite number"),
        (["eval", "--tasks", "{task}", "--method", "mini_swe:temperature=Infinity"], "a finite"),
    ],
    ids=lambda value: " ".join(value[:2]) + " " + value[-1] if isinstance(value, list) else "",
)  # fmt: skip
def test_a_number_out_of_range_is_refused_before_anything_runs(
    argv: list[str],
    expected: str,
    bouncing_task: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A time budget of 0 or a timeout of ``nan`` would make every run fail (or never end):
    refused by name, before anything starts."""
    values = {"task": str(bouncing_task), "twin": str(bouncing_task / "hidden" / "generator")}
    out = tmp_path / "out"
    code: int | None
    try:
        code = main([*(part.format(**values) for part in argv), "--out", str(out)])
    except SystemExit as exit_:  # argparse refuses a value of the wrong type
        code = exit_.code if isinstance(exit_.code, int) else 2
    err = capsys.readouterr().err
    assert code in (1, 2) and "error:" in err and expected in err, err
    assert "Traceback" not in err and not out.exists()


def _set_number(task: Path, file: str, old: str, new: str) -> None:
    """Change one number of ``task``: a line of ``task.toml``, or the value at a dotted path
    of a JSON file (``new`` is JSON text, where ``Infinity`` and ``NaN`` are numbers)."""
    path = task / file
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        *parents, last = old.split(".")
        node = data
        for key in parents:
            node = node[key]
        node[last] = json.loads(new)
        path.write_text(json.dumps(data))
    else:
        text = path.read_text()
        assert text.count(old) == 1, f"{old!r} not once in {path}"
        path.write_text(text.replace(old, new))


@pytest.mark.parametrize(
    ("file", "old", "new"),
    [
        ("task.toml", "cpus = 2.0", "cpus = inf"),
        ("task.toml", "fps = 30.0", "fps = inf"),
        ("task.toml", "max_size_mb = 50.0", "max_size_mb = inf"),
        ("visible/capture.json", "views.cam.rgb.fps", "Infinity"),
        ("visible/capture.json", "world.gravity.value", "[0.0, 0.0, NaN]"),
    ],
    ids=["cpus=inf", "output-fps=inf", "max_size_mb=inf", "capture-fps=Infinity", "gravity-NaN"],
)  # fmt: skip
def test_a_number_of_a_task_that_is_not_finite_is_refused(
    file: str, old: str, new: str, task_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``inf`` is above 0, so ``cpus = inf`` (or an output's ``fps``) passed as a positive
    number and ``r2s task validate`` said ok; every number of a task and of its capture must
    be finite."""
    _set_number(task_copy, file, old, new)
    text = _fails_clearly(["task", "validate", str(task_copy)], capsys)
    assert "finite" in text, text


@pytest.mark.parametrize("command", ["eval", "train reward", "twin check", "export harbor"])
def test_a_task_with_cpus_inf_stops_before_anything_runs(
    command: str, task_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With ``cpus = inf`` every command that runs the twin, and ``r2s export harbor``, ended
    in a traceback (``int(inf)`` for the twin's threads), ``r2s eval`` leaving its run
    'running': refused as a broken task instead, with nothing scored."""
    _set_number(task_copy, "task.toml", "cpus = 2.0", "cpus = inf")
    twin = Path(shutil.copytree(task_copy / "hidden" / "generator", tmp_path / "twin"))
    out = tmp_path / "out"
    argv = {
        "eval": ["eval", "--tasks", str(task_copy), "--method", "oracle", "--backend", "local",
                 "--out", str(out)],
        "train reward": ["train", "reward", "--task", str(task_copy), "--solution", str(twin),
                         "--backend", "local"],
        "twin check": ["twin", "check", str(twin), "--task", str(task_copy), "--backend",
                       "local", "--out", str(out)],
        "export harbor": ["export", "harbor", str(task_copy), "--out", str(out)],
    }[command]  # fmt: skip
    text = _fails_clearly(argv, capsys)
    assert "cpus" in text and "finite" in text, text
    assert not out.exists() or not list(out.rglob("seed-*.json"))


def test_select_and_tier_without_a_benchmark_are_refused(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--select`` and ``--tier`` choose among the tasks a ``--benchmark`` builds: with
    ``--tasks`` alone they were dropped without a word, and every task ran."""
    out = tmp_path / "run"
    text = _fails_clearly(["eval", "--tasks", str(bouncing_task), "--select", "no-such-task",
                           "--tier", "no-such-tier", "--method", "static", "--backend", "local",
                           "--out", str(out)], capsys)  # fmt: skip
    assert "--select" in text and "fix:" in text and "--include" in text, text
    assert not out.exists()


def _not_a_program(kind: str, tmp_path: Path) -> str:
    """A ``--solution`` that holds no program with ``run.py``."""
    path = tmp_path / kind.replace(" ", "-")
    if kind in ("oracel", "Oracle"):
        return kind  # the keyword misspelt: a relative path that is not there
    if kind == "a file":
        path.write_text("print('hello')\n")
    elif kind != "missing":
        path.mkdir()
    if kind == "no run.py":
        (path / "model.json").write_text("{}\n")
    if kind == "package without run.py":
        (path / "twin.json").write_text("{}\n")
        (path / "program").mkdir()
        (path / "program" / "main.py").write_text("\n")
    return str(path)


SOLUTION_MISTAKES = ["missing", "oracel", "Oracle", "a file", "empty", "no run.py",
                     "package without run.py"]  # fmt: skip


@pytest.mark.parametrize("kind", SOLUTION_MISTAKES)
def test_a_harbor_solution_without_a_program_is_refused(
    kind: str,
    bouncing_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``r2s export harbor --solution`` took any path: a typo, ``oracel``, a file or an empty
    directory exported a ``solve.sh`` that recreates nothing (status 0), and Harbor's oracle
    then scored the task's own reference 0. Refused before anything is written."""
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "harbor"
    argv = ["export", "harbor", str(bouncing_task), "--out", str(out)]
    text = _fails_clearly([*argv, "--solution", _not_a_program(kind, tmp_path)], capsys)
    assert "--solution" in text and "fix: give 'oracle'" in text and "run.py" in text, text
    if kind in ("oracel", "Oracle"):
        assert "did you mean 'oracle'?" in text, text
    assert not out.exists() and not list(tmp_path.glob(".*harbor*"))


@pytest.mark.parametrize("kind", ["missing", "a file", "empty", "package without run.py"])
def test_a_harbor_method_solution_without_a_program_is_refused(
    kind: str, bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ``harbor`` method's ``solution=`` (for Harbor's oracle agent) went through the
    same ``solve.sh``: a wrong path built a method that "delivered nothing" (0). Refused
    before anything runs."""
    method = f"harbor:agent=oracle,harbor=/bin/false,solution={_not_a_program(kind, tmp_path)}"
    out = tmp_path / "run"
    text = _fails_clearly(["eval", "--tasks", str(bouncing_task), "--method", method,
                           "--backend", "local", "--out", str(out)], capsys)  # fmt: skip
    assert "harbor method's solution" in text and "fix: give solution=" in text, text
    assert not list(out.rglob("seed-*.json"))


def test_a_twin_package_is_taken_wherever_a_program_is(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``r2s twin check`` takes a twin package (twin.json and program/) or a program directory;
    so do ``r2s train reward --solution`` and ``--method program:path=``, rather than scoring
    the package as an undelivered program (a reward of 0)."""
    import json

    package = bouncing_task / "hidden" / "generator"  # a twin package
    argv = ["train", "reward", "--task", str(bouncing_task), "--backend", "local", "--json"]
    rewards = []
    for solution in (package, package / "program"):
        assert main([*argv, "--solution", str(solution)]) == 0
        rewards.append(json.loads(capsys.readouterr().out)["reward"])
    assert rewards[0] == rewards[1] > 0.99
    run = ["eval", "--tasks", str(bouncing_task), "--method", f"program:path={package}",
           "--backend", "local", "--out", str(tmp_path / "run")]  # fmt: skip
    assert main(run) == 0
    (method,) = json.loads((tmp_path / "run" / "results.json").read_text())["methods"]
    assert method["delivered_rate"] == 1.0 and method["reproduction_skill"]["mean"] > 0.99


@pytest.mark.parametrize(
    "files",
    [
        {"experiment.json": "my experiment notes\n", "notes.txt": "keep this too\n"},
        {"experiment.json": '{"tasks": {}}\n'},  # JSON, but not the Lab's
        {"state/": "", "notes.txt": "keep\n"},
        {"r2s.log": "my own log\n"},
    ],
    ids=["notes", "json", "state", "log"],
)
def test_files_named_like_a_runs_do_not_make_a_run_directory(
    files: dict[str, str], bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run directory is known by what its records say, not by their names: a directory of
    the user's that holds an ``experiment.json`` (or ``state/``, ``r2s.log``) is refused."""
    mine = tmp_path / "mine"
    mine.mkdir()
    for name, text in files.items():
        if name.endswith("/"):
            (mine / name).mkdir()
        else:
            (mine / name).write_text(text)
    before = {p.relative_to(mine): p.read_bytes() for p in mine.rglob("*") if p.is_file()}
    dirs = sorted(p.relative_to(mine) for p in mine.rglob("*"))
    assert "not a Real2SimLab run directory" in _fails_clearly(_eval(bouncing_task, mine), capsys)
    assert {p.relative_to(mine): p.read_bytes() for p in mine.rglob("*") if p.is_file()} == before
    assert sorted(p.relative_to(mine) for p in mine.rglob("*")) == dirs


def test_a_missing_program_path_is_refused_before_anything_runs(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    argv = ["eval", "--tasks", str(bouncing_task), "--method",
            f"program:path={tmp_path / 'progam'}", "--backend", "local", "--out", str(out)]  # fmt: skip
    err = _fails_clearly(argv, capsys)
    assert "progam does not exist" in err and "fix:" in err, err
    assert not out.exists()  # no run, no score of 0


def test_an_input_inside_the_run_directory_is_refused(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A program an earlier run delivered, evaluated into that same run directory: redoing
    its build would delete it. Refused, and the program stays."""
    out = tmp_path / "run"
    assert main(["eval", "--tasks", str(bouncing_task), "--method", "mine=static",
                 "--backend", "local", "--out", str(out)]) == 0  # fmt: skip
    program = next(out.glob("builds/*/mine/trial-0/twin/program"))
    files = sorted(p.relative_to(program) for p in program.rglob("*"))
    capsys.readouterr()
    argv = ["eval", "--tasks", str(bouncing_task), "--method", f"mine=program:path={program}",
            "--backend", "local", "--out", str(out)]  # fmt: skip
    assert "holds or lies inside the input" in _fails_clearly(argv, capsys)
    assert sorted(p.relative_to(program) for p in program.rglob("*")) == files


def test_a_relative_work_directory_scores_as_an_absolute_one(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The local backend runs a program in its own directory: a relative --work must not
    become a HOME or TMPDIR that points elsewhere from there (a reward of 0)."""
    import json as json_module

    program = Path(shutil.copytree(bouncing_task / "hidden" / "generator" / "program",
                                   tmp_path / "program"))  # fmt: skip
    source = (program / "run.py").read_text()
    (program / "run.py").write_text(
        "from pathlib import Path\n(Path.home() / 'cache.txt').write_text('x')\n"
        "import os, tempfile\nPath(tempfile.mkdtemp()).rmdir()\n" + source
    )
    monkeypatch.chdir(tmp_path)
    here = str(Path(real2simlab.__file__).resolve().parents[1])  # this package, not another
    rewards = []
    for work in ("work-relative", str(tmp_path / "work-absolute")):
        out = tmp_path / f"{Path(work).name}.json"
        argv = ["train", "reward", "--task", str(bouncing_task), "--solution", str(program),
                "--backend", "local", "--work", work, "--json"]  # fmt: skip
        done = subprocess.run([sys.executable, "-m", "real2simlab.cli.main", *argv],
                              capture_output=True, text=True, check=False,
                              env={**os.environ, "PYTHONPATH": here})  # fmt: skip
        assert done.returncode == 0, done.stderr
        out.write_text(done.stdout)
        rewards.append(json_module.loads(done.stdout)["reward"])
    assert rewards[0] == rewards[1] > 0.99, rewards


def test_a_check_or_work_directory_of_the_users_is_never_cleared(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``twin check --out`` and ``train reward --work`` clear their directory: only one they
    wrote, known by its content, not one that merely holds the same names."""
    twin = bouncing_task / "hidden" / "generator"
    mine = tmp_path / "mine"
    (mine / "run").mkdir(parents=True)
    (mine / "run" / "notes.txt").write_text("keep me")
    _fails_clearly(["twin", "check", str(twin), "--backend", "local", "--out", str(mine)], capsys)
    work = tmp_path / "work"
    work.mkdir()
    (work / ".r2s-reward").write_text("my own file\n")
    argv = ["train", "reward", "--task", str(bouncing_task), "--solution", str(twin / "program"),
            "--backend", "local", "--work", str(work)]  # fmt: skip
    _fails_clearly(argv, capsys)
    assert (mine / "run" / "notes.txt").read_text() == "keep me"
    assert (work / ".r2s-reward").read_text() == "my own file\n"
    check = tmp_path / "check"  # what a check writes, it may clear and write again
    for _ in range(2):
        assert main(["twin", "check", str(twin), "--backend", "local", "--out", str(check)]) == 0


@pytest.mark.parametrize("marker", ["task.toml", "export.json", "training.json"])
def test_overwrite_replaces_only_what_the_lab_wrote(
    marker: str, bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--overwrite`` replaces a task, a dataset or a checkpoint: a directory of the user's
    holding a file of the same name is none of them."""
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / marker).write_text("my own file\n")
    (mine / "notes.txt").write_text("keep me")
    data = tmp_path / "data"
    assert main(["train", "export", "--tasks", str(bouncing_task), "--allow-eval-tasks",
                 "--out", str(data)]) == 0  # fmt: skip
    argv = {
        "task.toml": ["bench", "build", "synthetic", "--select", "bouncing_ball", "--tier",
                      "calibrated", "--out", str(tmp_path), "--overwrite"],
        "export.json": ["train", "export", "--tasks", str(bouncing_task), "--allow-eval-tasks",
                        "--out", str(mine), "--overwrite"],
        "training.json": ["train", "fit", "nearest_twin", "--data", str(data), "--out", str(mine),
                          "--overwrite"],
    }[marker]  # fmt: skip
    if marker == "task.toml":  # the task directory the build would replace
        target = tmp_path / "bouncing_ball" / "calibrated"
        target.parent.mkdir(parents=True)
        mine.rename(target)
        mine = target
    capsys.readouterr()
    _fails_clearly(argv, capsys)
    assert (mine / marker).read_text() == "my own file\n"
    assert (mine / "notes.txt").read_text() == "keep me"


def _files(root: Path) -> dict[str, bytes | None]:
    """Everything under ``root``: each file's content, ``None`` for a directory."""
    return {
        path.relative_to(root).as_posix(): None if path.is_dir() else path.read_bytes()
        for path in sorted(root.rglob("*"))
    }


LOOK_ALIKES = {
    "eval: experiment.json": ("experiment.json", '{"lab_version": "not-real2simlab", "a": 1}'),
    "eval: experiment.json of another spec": (
        "experiment.json", '{"spec": "r2s-results/1.0", "lab_version": "1.0.0"}'),
    "eval: state/status.json": ("state/status.json", '{"status": "done", "lab_version": "2.1"}'),
    "train export: export.json": ("export.json", '{"lab_version": "some-other-tool"}'),
    "train export: export.json of another spec": ("export.json", '{"spec": "r2s-task/1.0"}'),
    "train fit: training.json": ("training.json", '{"lab_version": "not-real2simlab"}'),
    "train fit: training.json of another spec": ("training.json", '{"spec": "r2s-run/1.0"}'),
    "bench build: task.toml with a lab_version": ("task.toml", 'lab_version = "not-r2s"'),
    "bench build: task.toml of another spec": ("task.toml", 'spec = "r2s-mine"'),
}  # fmt: skip
"""Files with a name the Lab gives its records, holding what another tool or the user wrote:
some ``lab_version``, or a ``spec`` that starts with ``r2s-`` but is not the record's."""


@pytest.mark.parametrize("case", sorted(LOOK_ALIKES))
def test_a_look_alike_of_a_lab_record_does_not_make_a_directory_the_labs(
    case: str, bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``r2s eval`` writes into, and ``--overwrite`` replaces, a directory only when its record
    is one the Lab wrote, judged by all of its content: with a look-alike the directory is the
    user's, and nothing in it is written over or removed."""
    marker, text = LOOK_ALIKES[case]
    command = case.split(":")[0]
    mine = tmp_path / "mine"
    if command == "bench build":
        mine = tmp_path / "tasks" / "bouncing_ball" / "calibrated"
    (mine / marker).parent.mkdir(parents=True)
    (mine / marker).write_text(text + "\n")
    (mine / "results.json").write_text("IRREPLACEABLE USER RESULT\n")
    (mine / "report" / "tasks").mkdir(parents=True)
    (mine / "report" / "tasks" / "mine.html").write_text("MY PAGE\n")
    (mine / "personal.txt").write_text("DO NOT DELETE\n")
    data = tmp_path / "data"
    export = ["train", "export", "--tasks", str(bouncing_task), "--allow-eval-tasks"]
    if command == "train fit":
        assert main([*export, "--out", str(data)]) == 0
    argv = {
        "eval": _eval(bouncing_task, mine),
        "train export": [*export, "--out", str(mine), "--overwrite"],
        "train fit": ["train", "fit", "nearest_twin", "--data", str(data), "--out", str(mine),
                      "--overwrite"],
        "bench build": ["bench", "build", "synthetic", "--select", "bouncing_ball", "--tier",
                        "calibrated", "--out", str(tmp_path / "tasks"), "--overwrite"],
    }[command]  # fmt: skip
    before = _files(mine)
    capsys.readouterr()
    _fails_clearly(argv, capsys)
    assert _files(mine) == before


def test_a_look_alike_of_the_export_record_is_a_changed_dataset(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``r2s train fit`` trains on a dataset only as ``r2s train export`` wrote it: an
    ``export.json`` replaced by another tool's file of that name is not the export's record, so
    the dataset is refused as changed, not trained on with nothing to check it against."""
    data, model = tmp_path / "data", tmp_path / "model"
    assert main(["train", "export", "--tasks", str(bouncing_task), "--allow-eval-tasks",
                 "--out", str(data)]) == 0  # fmt: skip
    (data / "export.json").write_text('{"lab_version": "some-other-tool"}\n')
    capsys.readouterr()
    error = _fails_clearly(["train", "fit", "nearest_twin", "--data", str(data), "--out",
                            str(model)], capsys)  # fmt: skip
    assert "not an export record" in error, error
    assert not model.exists()


def test_the_records_the_lab_writes_mark_its_directories(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """What the Lab writes, and wrote in earlier versions, is its own record, so its commands
    resume and replace their own directories as before."""
    from real2simlab._fs import lab_record

    run, data, model, new = (tmp_path / name for name in ("run", "data", "model", "new"))
    export = ["train", "export", "--tasks", str(bouncing_task), "--allow-eval-tasks", "--out",
              str(data)]  # fmt: skip
    fit = ["train", "fit", "nearest_twin", "--data", str(data), "--out", str(model)]
    assert main(_eval(bouncing_task, run)) == 0
    assert main(export) == 0
    assert main(fit) == 0
    assert main(["task", "new", "--id", "mybench/drop", "--title", "A drop", str(new)]) == 0
    for record in (run / "experiment.json", run / "state" / "status.json", data / "export.json",
                   model / "training.json", bouncing_task / "task.toml", new / "task.toml"):  # fmt: skip
        assert lab_record(record), record
    # as earlier versions wrote them: the experiment before its tasks were resolved, the status
    # while the run was running, an export before it recorded its content
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    config = json.loads((run / "experiment.json").read_text())["config"]
    exported = json.loads((data / "export.json").read_text())
    del exported["content"]
    for name, record in {
        "experiment.json": {"config": config, "lab_version": "1.0.0", "base_dir": "."},
        "status.json": {"status": "running", "lab_version": "1.0.0",
                        "started_at": "2026-11-17T10:00:00+00:00"},
        "export.json": exported,
    }.items():  # fmt: skip
        (earlier / name).write_text(json.dumps(record, indent=2) + "\n")
        assert lab_record(earlier / name), name
    assert main(_eval(bouncing_task, run)) == 0  # resumed
    assert main([*export, "--overwrite"]) == 0  # replaced
    assert main([*fit, "--overwrite"]) == 0


def test_a_check_json_of_the_users_does_not_make_a_check_directory(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without a check's marker, a ``check.json`` makes ``--out`` an earlier check's directory
    (which a check clears) only if it is the one a check writes, its two runs ``run`` and
    ``rerun``: the user's own list of checks with ``ok`` fields is not, and nothing is
    cleared."""
    twin = bouncing_task / "hidden" / "generator"
    mine = tmp_path / "mine"
    (mine / "run").mkdir(parents=True)
    checks = [{"name": "lint", "ok": True}, {"name": "unit tests", "ok": False}]
    (mine / "check.json").write_text(json.dumps(checks) + "\n")
    (mine / "run" / "personal.txt").write_text("DO NOT DELETE\n")
    (mine / "run" / "notes.md").write_text("mine\n")
    before = _files(mine)
    _fails_clearly(["twin", "check", str(twin), "--backend", "local", "--out", str(mine)], capsys)
    assert _files(mine) == before
    # a check directory as the versions before the marker left it is still a check's
    old = tmp_path / "old"
    argv = ["twin", "check", str(twin), "--backend", "local", "--out", str(old)]
    assert main(argv) == 0
    (old / ".r2s-check").unlink()
    assert main(argv) == 0


def test_a_file_added_to_a_check_directory_is_named_and_kept(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The error names the files the check did not write, not the check's own."""
    check = tmp_path / "check"
    argv = ["twin", "check", str(bouncing_task / "hidden" / "generator"), "--backend", "local",
            "--out", str(check)]  # fmt: skip
    assert main(argv) == 0
    (check / "notes.txt").write_text("keep me")
    capsys.readouterr()
    error = _fails_clearly(argv, capsys)
    assert "did not write (notes.txt)" in error, error
    assert (check / "notes.txt").read_text() == "keep me"


def test_a_reward_never_clears_a_file_added_to_its_work_directory(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--work`` used again is cleared of what the reward wrote, never of a file the user has
    put there since (as ``twin check --out``); what a file manager leaves (``.DS_Store``) is
    not the user's."""
    work = tmp_path / "work"
    argv = ["train", "reward", "--task", str(bouncing_task), "--solution",
            str(bouncing_task / "hidden" / "generator" / "program"), "--backend", "local",
            "--work", str(work)]  # fmt: skip
    assert main(argv) == 0
    (work / "notes.txt").write_text("keep me")
    capsys.readouterr()
    error = _fails_clearly(argv, capsys)
    assert "notes.txt" in error, error
    assert (work / "notes.txt").read_text() == "keep me"
    (work / "notes.txt").unlink()
    (work / ".DS_Store").write_bytes(b"\0")
    assert main(argv) == 0


def test_a_state_lock_of_the_users_does_not_make_a_run_directory(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A directory holding only ``state/lock`` is a run's (one stopped before its first status)
    only when the lock is one a run wrote: another program's ``state/lock`` stays as it is."""
    mine = tmp_path / "mine"
    (mine / "state").mkdir(parents=True)
    (mine / "state" / "lock").write_text("held by my sync tool\n")
    before = _files(mine)
    _fails_clearly(_eval(bouncing_task, mine), capsys)
    assert _files(mine) == before


def test_tasks_an_earlier_command_left_in_out_are_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--overwrite`` replaces the tasks the command writes; tasks an earlier command wrote
    there (another ``--count`` or ``--seed``) stay, and the command says so: every command
    given that directory as ``--tasks`` uses them too."""
    out = tmp_path / "tasks"
    argv = ["train", "tasks", "--select", "bouncing_ball", "--tier", "calibrated", "--out",
            str(out)]  # fmt: skip
    assert main([*argv, "--count", "2"]) == 0
    assert "also holds" not in capsys.readouterr().err
    assert main([*argv, "--count", "1", "--overwrite"]) == 0
    error = capsys.readouterr().err
    assert "also holds 1 task(s)" in error, error


@pytest.mark.parametrize(
    ("argv", "hint"),
    [
        (["train", "tasks", "--count", "2", "--select", "bouncing_ball", "--out", "{out}"],
         "--overwrite"),
        (["bench", "build", "synthetic", "--select", "bouncing_ball", "--out", "{out}"],
         "--overwrite"),
        (["eval", "--tasks", "{task}", "--method", "static", "--backend", "local",
          "--out", "{out}"], "to resume"),
    ],
    ids=["train tasks", "bench build", "eval"],
)  # fmt: skip
def test_an_interruption_says_what_running_again_does(
    argv: list[str],
    hint: str,
    bouncing_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only a command that resumes says the same command resumes; one that writes tasks says
    to pass --overwrite, which then works."""
    from real2simlab.benchmarks.synthetic import SyntheticBenchmark
    from real2simlab.orchestrate.experiment import Experiment

    def interrupted(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(SyntheticBenchmark, "build", interrupted)
    monkeypatch.setattr(Experiment, "run", interrupted)
    values = {"out": str(tmp_path / "out"), "task": str(bouncing_task)}
    assert main([part.format(**values) for part in argv]) == 130
    err = capsys.readouterr().err
    assert err.startswith("interrupted") and hint in err, err


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (["--fov", "0"], "above 0 and below 180"),
        (["--fov", "180"], "above 0 and below 180"),
        (["--fov", "200"], "above 0 and below 180"),
        (["--fov", "nan"], "above 0 and below 180"),
        (["--intrinsics", "0,500,160,120"], "positive focal lengths"),
        (["--intrinsics", "500,500"], "fx,fy,cx,cy"),
        (["--object", "ball=-1"], "positive length"),
        (["--object", "ball=0"], "positive length"),
        (["--object", "ball"], "NAME=SIZE_M"),
        (["--object", "=0.1"], "the object's name"),
    ],
)
def test_a_capture_is_never_written_with_an_impossible_calibration(
    given: list[str],
    expected: str,
    bouncing_task: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    video = next(bouncing_task.glob("visible/**/*.mp4"))
    out = tmp_path / "capture"
    try:
        code = main(["capture", "from-video", "--capture-id", "c", str(video), str(out), *given])
    except SystemExit as exit_:  # argparse refuses a value it cannot use
        code = exit_.code if isinstance(exit_.code, int) else 2
    err = capsys.readouterr().err
    assert code in (1, 2) and expected in err and "Traceback" not in err, err
    assert not out.exists()


def test_a_long_task_id_fits_in_file_names(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A valid id of many short segments can be longer than a file name may be: the Lab's
    directories for it are cut and told apart by a digest."""
    from real2simlab.task.io import read_task_config, write_task_config

    task = Path(shutil.copytree(bouncing_task, tmp_path / "task"))
    long_id = "my-dataset/" + "/".join(f"camera-{i:02d}-experiment-2026-11-22" for i in range(8))
    config = read_task_config(task / "task.toml")
    write_task_config(config.model_copy(update={"id": long_id}), task / "task.toml")
    assert main(["task", "validate", str(task)]) == 0
    assert main(_eval(task, tmp_path / "run")) == 0
    assert main(["train", "export", "--tasks", str(task), "--allow-eval-tasks", "--runs",
                 str(tmp_path / "run"), "--out", str(tmp_path / "data")]) == 0  # fmt: skip
    assert main(["report", str(tmp_path / "run")]) == 0
    assert "Traceback" not in capsys.readouterr().err


def test_a_copy_that_fails_for_lack_of_space_is_a_clear_error(
    bouncing_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A full disk (or ``ulimit -f``) during a directory copy: one error line and the fix."""
    from real2simlab.train import dataset

    def full(source: Path, target: Path, **_: object) -> None:
        raise shutil.Error([(str(source), str(target), "[Errno 27] File too large")])

    monkeypatch.setattr(dataset.shutil, "copytree", full)
    argv = ["train", "export", "--tasks", str(bouncing_task), "--allow-eval-tasks",
            "--out", str(tmp_path / "data")]  # fmt: skip
    err = _fails_clearly(argv, capsys)
    assert "File too large" in err and "free space" in err and "ulimit -f" in err, err
    assert not (tmp_path / "data").exists()


def test_a_command_in_a_piped_script_leaves_the_rest_of_the_script(
    bouncing_task: Path, tmp_path: Path
) -> None:
    """``... | bash``: no process the Lab starts (ffmpeg decoding a video for the report, a
    sandbox, a scorer) reads the script that comes after the command."""
    import shlex

    here = str(Path(real2simlab.__file__).resolve().parents[1])
    r2s = shlex.join([sys.executable, "-m", "real2simlab.cli.main"])
    argv = shlex.join(_eval(bouncing_task, tmp_path / "run"))
    script = f"{r2s} {argv} > {tmp_path / 'eval.log'} 2>&1\nprintf 'FOLLOW-UP-RAN\\n'\n"
    done = subprocess.run(["bash"], input=script, capture_output=True, text=True, check=False,
                          env={**os.environ, "PYTHONPATH": here}, timeout=300)  # fmt: skip
    assert done.returncode == 0 and done.stdout == "FOLLOW-UP-RAN\n", (done, script)
    assert (tmp_path / "run" / "report").is_dir()  # the report (videos decoded) was built


def test_no_process_the_lab_starts_reads_its_standard_input() -> None:
    """Every ``subprocess`` call in the package says what its standard input is (``stdin`` or
    ``input``): inheriting it lets a child read a script the command was piped from."""
    import ast

    calls = {"run", "Popen", "check_output", "check_call", "call"}
    inheriting = []
    for path in sorted(Path(real2simlab.__file__).parent.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in calls
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
                and not {k.arg for k in node.keywords} & {"stdin", "input", None}
            ):
                inheriting.append(f"{path.name}:{node.lineno}")
    assert not inheriting, f"give these stdin=subprocess.DEVNULL: {inheriting}"


def test_a_rebuilt_report_has_no_page_of_results_no_longer_counted(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """After a configuration change a budget does not let run, the earlier configuration's
    scores are not counted: no task page of the report shows them, however often it is built."""
    run = tmp_path / "run"
    first = ["eval", "--tasks", str(bouncing_task), "--method", "fake_agent", "--backend", "local",
             "--max-agent-runs", "1", "--out", str(run)]  # fmt: skip
    assert main(first) == 0
    (page,) = (run / "report" / "tasks").glob("*.html")
    assert "reproduction_skill" in page.read_text()
    changed = [*first[:4], "fake_agent:mode=broken", *first[5:]]
    assert main(changed) == 0  # the budget is spent: the changed method is not run
    for _ in range(2):  # built by the evaluation, then again by r2s report
        assert not list((run / "report" / "tasks").glob("*.html"))
        assert main(["report", str(run)]) == 0
    assert "Traceback" not in capsys.readouterr().err


def test_every_table_marks_an_open_network_and_a_privileged_method(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A method built with an open network, or one that reads hidden data, is marked in every
    table a result is read or copied from (the docs promise it): the scorecard printed by the
    command, every table of report.md and of the HTML pages, and results.csv."""
    import csv

    run = tmp_path / "run"
    argv = ["eval", "--tasks", str(bouncing_task), "--method", "guess=fake_agent:network=open",
            "--method", "restricted=fake_agent:network=none", "--method", "oracle",
            "--backend", "local", "--out", str(run)]  # fmt: skip
    assert main(argv) == 0
    printed = capsys.readouterr().out
    marked = {"guess": "guess (open network)", "oracle": "oracle (privileged)"}

    def rows_of(text: str, name: str) -> list[str]:
        return [line for line in text.splitlines() if line.startswith(f"| {name}")]

    report = (run / "report" / "report.md").read_text()
    sections = report.split("\n## ")
    for name, label in marked.items():
        assert rows_of(printed, name) and all(label in r for r in rows_of(printed, name))
        for section in sections:
            if section.startswith(("Scorecard", "Results")):
                assert all(label in r for r in rows_of(section, name)), (section, name)
                assert rows_of(section, name), (section, name)
    assert all("(" not in r.split("|")[1] for r in rows_of(printed + report, "restricted"))
    with (run / "report" / "results.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    assert list(table[0])[-2:] == ["privileged", "network"]  # added after the existing columns
    for row in table:
        assert row["network"] == {"guess": "open", "restricted": "none"}.get(row["method"], "")
        assert row["privileged"] == str(row["method"] == "oracle")
    pages = [run / "report" / "index.html", *(run / "report" / "tasks").glob("*.html")]
    for page in pages:
        text = page.read_text()
        assert "guess (open network)" in text and "oracle (privileged)" in text, page


def _refused() -> Exception:
    import urllib.error

    return urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))


def _wrapped() -> Exception:
    try:
        raise _refused()
    except Exception as cause:
        try:
            raise RuntimeError("the download library gave up") from cause
        except RuntimeError as error:
            return error


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (_refused, "a download failed: [Errno 111] Connection refused"),
        (_wrapped, "a download failed"),
        (lambda: __import__("http.client").client.IncompleteRead(b"x", 10), "a download failed"),
        (lambda: __import__("socket").gaierror(-2, "Name or service not known"), "download"),
        (lambda: OSError(30, "Read-only file system", "/data/x"), "Read-only file system"),
        (lambda: OSError(24, "Too many open files"), "Too many open files"),
    ],
    ids=["refused", "wrapped", "incomplete", "dns", "read-only", "open files"],
)
def test_a_machine_or_network_failure_is_a_clear_error_not_a_bug(
    raised: Callable[[], BaseException],
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A download that fails (no network, a broken proxy) or an operating-system limit is the
    machine's problem: one error line and the fix, not a traceback asking for a bug report."""
    import real2simlab.cli.tasks as tasks_cli

    def failing(args: object) -> int:
        raise raised()

    monkeypatch.setattr(tasks_cli, "_list", failing)
    monkeypatch.setenv("HTTPS_PROXY", "http://user:secret@127.0.0.1:9")
    err = _fails_clearly(["task", "list"], capsys)
    assert expected in err and "fix:" in err and "bug" not in err, err
    assert "secret" not in err  # a proxy is named, never its credentials
    if "download" in expected:
        assert "HTTPS_PROXY" in err


def test_a_broken_proxy_while_building_tasks_is_a_clear_error(tmp_path: Path) -> None:
    """The audit's case: ``r2s bench build physicsiq`` behind a proxy that refuses."""
    here = str(Path(real2simlab.__file__).resolve().parents[1])
    proxy = "http://127.0.0.1:9"
    env = {**os.environ, "PYTHONPATH": here, "R2S_CACHE_DIR": str(tmp_path / "cache"),
           "HTTPS_PROXY": proxy, "https_proxy": proxy, "HTTP_PROXY": proxy, "http_proxy": proxy,
           "NO_PROXY": "", "no_proxy": ""}  # fmt: skip
    argv = ["bench", "build", "physicsiq", "--select", "ball-and-block-fall",
            "--out", str(tmp_path / "tasks")]  # fmt: skip
    done = subprocess.run([sys.executable, "-m", "real2simlab.cli.main", *argv],
                          capture_output=True, text=True, env=env, check=False,
                          stdin=subprocess.DEVNULL, timeout=300)  # fmt: skip
    assert done.returncode == 1, done.stderr
    assert "Traceback" not in done.stderr and "fix:" in done.stderr, done.stderr
    assert "download failed" in done.stderr or "pip install" in done.stderr, done.stderr


@pytest.mark.parametrize("sha256", [[], ["--sha256", "d" * 64]], ids=["plain", "--sha256"])
def test_a_file_given_to_from_url_is_a_clear_error(
    sha256: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``r2s env unpack NAME --from-url FILE``: the error was that FILE.sha256 holds no
    checksum, with ``--sha256`` as the fix, then, with it, "unknown url type" without a fix. The
    fix is ``--from-tar`` with that file."""
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    tarball = tmp_path / "r2s-python.tar.gz"
    tarball.write_bytes(b"a tarball")
    err = _fails_clearly(
        ["env", "unpack", "r2s-python", "--from-url", str(tarball), *sha256], capsys
    )
    assert f"--from-tar {tarball}" in err and "checksum" not in err, err


PYTHON_IMAGE = "ghcr.io/real2simhub/r2s-python:1.0"


def _image_tarball(path: Path, images: list[tuple[list[str] | None, bytes]]) -> Path:
    """A tarball as ``docker save`` writes it, of one or more images, each (its names in
    ``RepoTags``, or ``None`` when saved by image id; its one layer, whose digest its
    configuration lists)."""
    import hashlib
    import io
    import tarfile

    entries: list[dict[str, object]] = []
    with tarfile.open(path, "w") as tar:

        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        for index, (tags, data) in enumerate(images):
            digest = "sha256:" + hashlib.sha256(data).hexdigest()
            config = {"config": {"Env": [f"IMAGE={index}"]}, "rootfs": {"diff_ids": [digest]}}
            add(f"config{index}.json", json.dumps(config).encode())
            add(f"layer{index}.tar", data)
            entries.append(
                {"Config": f"config{index}.json", "RepoTags": tags, "Layers": [f"layer{index}.tar"]}
            )
        add("manifest.json", json.dumps(entries).encode())
    return path


def _registered(cache: Path) -> dict[str, str]:
    index = cache / "rootfs" / "index.json"
    return json.loads(index.read_text()) if index.exists() else {}


@pytest.mark.parametrize(
    "name",
    ["r2s-blender-4.0", "r2s-pyton", "ghcr.io/real2simhub/r2s-verifier:1.0", "legacy", "url"],
)
def test_env_unpack_refuses_a_tarball_that_holds_another_image(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A2-04: ``r2s env unpack NAME --from-tar FILE`` registered whatever FILE held as NAME: a
    tarball of r2s-python as r2s-blender-4.0 (a twin calling Blender then failed in r2s-python
    and was scored 0), as a misspelt name, as another environment's full reference, from a
    tarball that names its image only in the older ``repositories`` file, or downloaded with
    ``--from-url`` (its checksum says it arrived whole, not which image it is). The tarball
    names its image: another one is refused, with what it holds and how to unpack that, and
    nothing is registered."""
    import functools
    import hashlib
    import http.server
    import io
    import tarfile
    import threading

    from .test_bwrap import layer

    cache = tmp_path / "cache"
    monkeypatch.setenv("R2S_CACHE_DIR", str(cache))
    tarball = tmp_path / "r2s-python.tar"
    if name == "legacy":  # an older `docker save`: no RepoTags, a `repositories` file instead
        built = _image_tarball(tmp_path / "built.tar", [(None, layer([("etc", "dir")]))])
        with tarfile.open(built) as source, tarfile.open(tarball, "w") as tar:
            for member in source.getmembers():
                tar.addfile(member, source.extractfile(member))
            repositories = json.dumps({PYTHON_IMAGE.rsplit(":", 1)[0]: {"1.0": "0" * 64}})
            info = tarfile.TarInfo("repositories")
            info.size = len(repositories)
            tar.addfile(info, io.BytesIO(repositories.encode()))
    else:
        _image_tarball(tarball, [([PYTHON_IMAGE], layer([("etc", "dir")]))])
    held = tarball
    if name == "url":
        monkeypatch.setenv("NO_PROXY", "127.0.0.1")
        monkeypatch.setenv("no_proxy", "127.0.0.1")

        class Quiet(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

        handler = functools.partial(Quiet, directory=str(tmp_path))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/{tarball.name}"
            digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
            argv = ["env", "unpack", "r2s-blender-4.0", "--from-url", url, "--sha256", digest]
            err = _fails_clearly(argv, capsys)
        finally:
            server.shutdown()
            server.server_close()
        held = cache / "downloads" / tarball.name  # what was downloaded, kept in the cache
    else:
        reference = "r2s-blender-4.0" if name == "legacy" else name
        err = _fails_clearly(["env", "unpack", reference, "--from-tar", str(tarball)], capsys)
    assert f"{held} holds {PYTHON_IMAGE}" in err, err
    assert f"fix: unpack it as what it is: r2s env unpack r2s-python --from-tar {held}" in err, err
    if name == "r2s-pyton":  # a name that is no environment's: the environments are listed
        assert "r2s-pyton is not a registered environment" in err and "r2s-blender-4.0" in err
    assert _registered(cache) == {}, err


@pytest.mark.parametrize(
    ("tags", "name"),
    [
        ([PYTHON_IMAGE], "r2s-python"),
        ([PYTHON_IMAGE], PYTHON_IMAGE),
        (["registry.example.com/mirror/r2s-python:1.0"], "r2s-python"),  # another prefix
        (["r2s-python:1.0"], "r2s-python"),
        (["other.example.com/tool:2", PYTHON_IMAGE], "r2s-python"),  # one of its names
        (["python:3.12-slim"], "docker.io/library/python:3.12-slim"),  # Docker Hub's short name
        (["example.org/qa/tool:latest"], "example.org/qa/tool"),  # `latest` filled in
        (None, "r2s-python"),  # saved by image id: no name to check
        (["<none>:<none>"], "r2s-python"),  # what some tools write for no name
        ([""], "r2s-python"),
    ],
)
def test_env_unpack_takes_a_tarball_of_the_image_named(
    tags: list[str] | None,
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The check of A2-04 reads names as Docker does: an environment's image under another
    registry prefix (``R2S_IMAGE_PREFIX`` where it was exported, a mirror) is still that
    environment's; Docker Hub's short names and ``latest`` are filled in. A tarball saved by
    image id names no image: it is unpacked under the name given, with a note that it could
    not be checked; so does one whose only name is empty or ``<none>:<none>`` (it was refused
    as holding an image of that name)."""
    from real2simlab.envs import image_reference
    from real2simlab.envs.rootfs import find_rootfs

    from .test_bwrap import layer

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    tarball = _image_tarball(tmp_path / "image.tar", [(tags, layer([("etc", "dir")]))])
    assert main(["env", "unpack", name, "--from-tar", str(tarball)]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith(f"unpacked {image_reference(name)} "), captured
    assert find_rootfs(image_reference(name)) is not None
    if tags is None or tags in (["<none>:<none>"], [""]):
        assert f"note: {tarball} names no image" in captured.err, captured.err
    else:
        assert "note:" not in captured.err, captured.err


@pytest.mark.parametrize("source", ["--from-tar", "docker"])
def test_env_unpack_takes_sha256_only_with_a_download(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A10-01: ``--sha256`` given without ``--from-url`` (with ``--from-tar FILE``, or with the
    image saved from Docker) was not checked against anything and silently ignored: a hash of
    zeros, or the hash of another file, unpacked the image all the same. It is refused, before
    anything is read or asked of Docker, with what checks a tarball instead."""
    from .test_bwrap import layer

    cache = tmp_path / "cache"
    monkeypatch.setenv("R2S_CACHE_DIR", str(cache))
    tarball = _image_tarball(tmp_path / "image.tar", [([PYTHON_IMAGE], layer([("etc", "dir")]))])
    argv = ["env", "unpack", "r2s-python", "--sha256", "0" * 64]
    if source == "--from-tar":
        argv += ["--from-tar", str(tarball)]
    err = _fails_clearly(argv, capsys)
    assert "error: --sha256 is the checksum of a download (--from-url)" in err, err
    assert "fix: leave out --sha256: every layer is checked" in err, err
    if source == "--from-tar":
        assert f"sha256sum {tarball}" in err, err
    assert _registered(cache) == {}, err


def test_an_out_name_that_fitted_before_still_fits(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review of A3-09's fix: the working copy beside an output bore this machine's tag, 13
    bytes more, so an ``--out`` whose name has 225 to 237 bytes (which worked) failed with
    "File name too long" and a fix that did not name the name. Such a copy is named without
    the tag again (what a command killed while writing it left is then not removed, as
    before)."""
    from real2simlab._fs import staging_dir

    out = tmp_path / ("h" * 230)
    assert main(["export", "harbor", str(bouncing_task), "--out", str(out)]) == 0, (
        capsys.readouterr().err
    )
    assert out.is_dir() and [path.name for path in tmp_path.iterdir()] == [out.name]
    staging = staging_dir(tmp_path / ("d" * 237))  # the longest name that fitted
    assert staging.is_dir() and len(staging.name.encode()) <= 255


def test_env_unpack_takes_the_image_named_out_of_a_tarball_of_several(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``docker save A B`` writes both images into one tarball: the image named is unpacked,
    not the first one (which was registered under the name, whatever it held)."""
    from real2simlab.envs.rootfs import find_rootfs

    from .test_bwrap import layer

    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    tarball = _image_tarball(
        tmp_path / "two.tar",
        [(["example.org/qa/first:1"], layer([("first", b"1")])),
         ([PYTHON_IMAGE], layer([("second", b"2")]))],
    )  # fmt: skip
    assert main(["env", "unpack", "r2s-python", "--from-tar", str(tarball)]) == 0
    unpacked = find_rootfs(PYTHON_IMAGE)
    assert unpacked is not None
    assert (unpacked.rootfs / "second").is_file() and not (unpacked.rootfs / "first").exists()
    assert unpacked.env == {"IMAGE": "1"}
    err = _fails_clearly(["env", "unpack", "r2s-verifier", "--from-tar", str(tarball)], capsys)
    assert f"holds example.org/qa/first:1 and {PYTHON_IMAGE}" in err, err


@pytest.mark.parametrize("damage", ["", "null", "[]", "{}", '{"agent_runs": -1}',
                                    '{"agent_runs": 1, "cost_usd": "free"}',
                                    '{"agent_runs": 1}',
                                    '{"agent_runs": 1, "cost_usd": 0, "cost_unknown_runs": -1}',
                                    '{"agent_runs": 1, "cost_usd": 1' + "0" * 400 + "}",
                                    '{"agent_runs": 1, "cost_usd": Infinity}',
                                    None])  # fmt: skip
def test_a_damaged_budget_record_never_resets_what_was_spent(
    damage: str | None, bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run's spending (``state/budget.json``) that is cut, damaged or missing: the agent-run
    budget is never taken as unspent. A damaged record stops the run, also one without what
    was spent (``cost_usd``, which the Lab always writes: it is not $0); a missing one is
    counted from the agent records, said to be at least what was spent (``None`` here: the
    file removed)."""
    run = tmp_path / "run"
    first = ["eval", "--tasks", str(bouncing_task), "--method", "fake_agent:mode=nothing",
             "--backend", "local", "--max-agent-runs", "1", "--out", str(run)]  # fmt: skip
    assert main(first) == 0
    budget = run / "state" / "budget.json"
    assert json.loads(budget.read_text())["agent_runs"] == 1
    if damage is None:
        budget.unlink()
    else:
        budget.write_text(damage)
    capsys.readouterr()
    again = [*first[:4], "fake_agent:mode=physical", *first[5:]]
    if damage is None:
        assert main(again) == 0
        assert "counted at least 1 agent run(s)" in (run / "r2s.log").read_text()
        assert json.loads(budget.read_text())["agent_runs"] == 1
    else:
        err = _fails_clearly(again, capsys)
        assert "budget record" in err and "budget.json" in err, err
    records = list((run / "builds").rglob("agent_record.json"))
    assert len(records) == 1 and '"delivered": false' in records[0].read_text()  # no new run


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("{}\n", "holds a JSON dict"),
        ('{"matrix": [[1, 0, 0, 0]]}', "holds a JSON dict"),
        ("[1, 2]", "finite 4x4 matrix"),
        ('[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, "one"]]', "finite 4x4 matrix"),
        ("[[{}, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]", "finite 4x4 matrix"),
        ("[[1, 0], [0, 1, 0]]", "finite 4x4 matrix"),
        ("not json", "not a JSON 4x4 matrix"),
    ],
)
def test_a_capture_is_never_written_with_an_extrinsics_file_of_another_shape(
    text: str,
    expected: str,
    bouncing_task: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    video = next(bouncing_task.glob("visible/**/*.mp4"))
    (tmp_path / "extrinsics.json").write_text(text)
    out = tmp_path / "capture"
    argv = ["capture", "from-video", "--capture-id", "c", str(video), str(out),
            "--extrinsics", str(tmp_path / "extrinsics.json")]  # fmt: skip
    err = _fails_clearly(argv, capsys)
    assert expected in err and "bug" not in err, err
    assert not out.exists()


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ({"timeot_s": 5}, "unknown options for scorer 'physimcode': ['timeot_s']; it takes: none"),
        (5, "scoring.native.options.scorer must be a table"),
    ],
)
def test_a_native_scorer_option_mistake_is_found_by_validation_not_a_crash(
    task_copy: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    table: object,
    expected: str,
) -> None:
    """``[scoring.native.options.scorer]`` are the scorer's own arguments: a name it does not
    take (a typo) is an error of the task file, from ``r2s task validate`` and before an
    evaluation starts, never a ``TypeError`` from inside it."""
    from real2simlab.task import TaskConfig
    from real2simlab.task.io import read_task_config, write_task_config

    data = read_task_config(task_copy / "task.toml").to_json()
    data["scoring"] = {
        "protocol": "native",
        "native": {"scorer": "physimcode", "options": {"scorer": table}},
    }
    write_task_config(TaskConfig.model_validate(data), task_copy / "task.toml")
    assert main(["task", "validate", str(task_copy)]) == 1
    assert expected in capsys.readouterr().out
    assert main(_eval(task_copy, tmp_path / "run")) == 1
    err = capsys.readouterr().err
    assert expected in err and "Traceback" not in err
    assert not (tmp_path / "run" / "results").exists()


def test_a_native_scorer_takes_the_options_it_names() -> None:
    from real2simlab.errors import UsageError
    from real2simlab.scoring.native import get_scorer

    assert get_scorer("physcodegym_pixels", lpips=False).lpips is False
    with pytest.raises(UsageError, match=r"lpips=.*must be true or false"):
        get_scorer("physcodegym_pixels", lpips="maybe")


def test_a_method_label_too_long_for_a_directory_is_refused_before_anything_runs(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A label names a directory of every build and run: one too long for a file name is a
    clear error of the command line, not ``OSError: File name too long`` mid-experiment."""
    for length, code in ((260, 1), (101, 1), (100, 0)):
        out = tmp_path / f"run-{length}"
        argv = _eval(bouncing_task, out)
        argv[argv.index("--method") + 1] = "x" * length + "=static"
        assert main(argv) == code, length
        err = capsys.readouterr().err
        assert "Traceback" not in err
        if code:
            assert f"has {length} characters" in err and "at most 100" in err
            assert not (out / "builds").exists()


def test_a_task_id_longer_than_a_file_name_still_finds_its_program(tmp_path: Path) -> None:
    from real2simlab.methods import get_method

    task_id = "a" * 150 + "/" + "b" * 150  # its one-directory name would be 302 bytes
    program = tmp_path / "programs" / ("b" * 150)
    program.mkdir(parents=True)
    (program / "run.py").write_text("print('twin')\n")
    method = get_method("program", path=str(tmp_path / "programs"))
    assert method.locate(task_id) == program  # pyright: ignore[reportAttributeAccessIssue]
    assert method.locate("c" * 300) is None  # pyright: ignore[reportAttributeAccessIssue]


def test_a_program_that_lost_an_executable_bit_is_a_changed_program(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A helper's executable bit is part of a program: after ``chmod -x`` the same evaluation
    runs the program again (its old score is not reused), its reward is the current
    program's, and a frozen program that lost the bit is not exported with its old reward."""
    solution = tmp_path / "solution"
    solution.mkdir()
    for path in (bouncing_task / "hidden" / "reference").iterdir():
        shutil.copyfile(path, solution / path.name)
    (solution / "run.py").write_text(
        "import subprocess\nsubprocess.run(['./helper.sh'], check=True)\n"
    )
    (solution / "helper.sh").write_text(
        '#!/bin/sh\ncp meta.json camera.json tracks.npz "$R2S_OUTPUT_DIR/"\n'
    )
    (solution / "helper.sh").chmod(0o755)
    run = ["eval", "--tasks", str(bouncing_task), "--method", f"program:path={solution}",
           "--backend", "local", "--out", str(tmp_path / "run")]  # fmt: skip

    def skill() -> float:
        (method,) = json.loads((tmp_path / "run" / "results.json").read_text())["methods"]
        return method["reproduction_skill"]["mean"]

    assert main(run) == 0 and skill() > 0.99
    (solution / "helper.sh").chmod(0o644)
    assert main(run) == 0 and skill() == 0.0
    reward = ["train", "reward", "--task", str(bouncing_task), "--solution", str(solution),
              "--backend", "local", "--json"]  # fmt: skip
    capsys.readouterr()
    assert main(reward) == 0 and json.loads(capsys.readouterr().out)["reward"] == 0.0
    (solution / "helper.sh").chmod(0o755)
    assert main(run) == 0 and skill() > 0.99
    frozen = next((tmp_path / "run" / "builds").rglob("twin/program/helper.sh"))
    frozen.chmod(0o644)  # the frozen program changed after it was scored
    export = ["train", "export", "--runs", str(tmp_path / "run"), "--allow-eval-tasks",
              "--out", str(tmp_path / "data")]  # fmt: skip
    capsys.readouterr()
    assert main(export) == 1
    assert "changed after it was scored" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("scorer", "options", "expected"),
    [
        (
            "physimcode",
            {"rerunn": True},
            "unknown options ['rerunn']; the scorer reads ['rerun', 'scorer']",
        ),
        ("physimcode", {"rerun": "yes"}, "rerun must be true or false"),
        ("fourdcodebench", {"kind": "synthetic"}, "the scorer needs ['case']"),
    ],
)
def test_a_native_option_the_scorer_does_not_read_is_found_by_validation(
    task_copy: Path,
    capsys: pytest.CaptureFixture[str],
    scorer: str,
    options: dict[str, object],
    expected: str,
) -> None:
    """A task's other native options are read by the Lab (``rerun``) or the scorer (declared by
    it): a typo is not silently ignored, and a missing one is not a ``KeyError`` mid-run."""
    from real2simlab.task import TaskConfig
    from real2simlab.task.io import read_task_config, write_task_config

    data = read_task_config(task_copy / "task.toml").to_json()
    data["scoring"] = {"protocol": "native", "native": {"scorer": scorer, "options": options}}
    write_task_config(TaskConfig.model_validate(data), task_copy / "task.toml")
    assert main(["task", "validate", str(task_copy)]) == 1
    assert expected in capsys.readouterr().out


@pytest.mark.parametrize(
    "layout",
    ["synthetic/bouncing_ball/calibrated", "synthetic__bouncing_ball__calibrated", "calibrated"],
)
def test_a_directory_of_programs_named_after_the_task_id_is_found(
    bouncing_task: Path, tmp_path: Path, layout: str
) -> None:
    """``program:path=`` a directory of per-task programs, "one per task, named after the task
    id" (the tutorial): the id as nested directories, as one name with ``__``, or one segment
    of it. Not found, the task would be scored as delivering nothing."""
    import json

    programs = tmp_path / "programs"
    shutil.copytree(bouncing_task / "hidden" / "generator" / "program", programs / layout)
    run = ["eval", "--tasks", str(bouncing_task), "--method", f"mine=program:path={programs}",
           "--backend", "local", "--out", str(tmp_path / "run")]  # fmt: skip
    assert main(run) == 0
    (method,) = json.loads((tmp_path / "run" / "results.json").read_text())["methods"]
    assert method["delivered_rate"] == 1.0 and method["reproduction_skill"]["mean"] > 0.99


def test_a_task_never_gets_the_program_of_another_task(tmp_path: Path) -> None:
    """The task's own directory (nested, or with ``__``) wins over a directory named after one
    segment, which tasks share; a segment directory that also holds other tasks' programs
    (a nested layout) is not shared by anyone."""
    from real2simlab.methods import get_method

    root = tmp_path / "programs"

    def program(name: str, package: bool = False) -> Path:
        where = root / name / ("program" if package else "")
        where.mkdir(parents=True)
        (where / "run.py").write_text("print('twin')\n")
        if package:
            (root / name / "twin.json").write_text("{}\n")
        return where

    nested = program("synthetic/bouncing_ball/calibrated")
    shared = program("calibrated")
    flat = program("synthetic__ramp__calibrated")
    a, a_b = program("a"), program("a/b")
    package = program("drop", package=True)
    locate = get_method("program", path=str(root)).locate  # pyright: ignore[reportAttributeAccessIssue]
    assert locate("synthetic/bouncing_ball/calibrated") == nested
    assert locate("synthetic/ramp/calibrated") == flat
    assert locate("synthetic/pendulum/calibrated") == shared
    assert locate("a") == a and locate("a/b") == a_b
    assert locate("a/c") is None  # a/ is task a's program and holds a/b's: no one shares it
    assert locate("iris/drop") == package
    assert locate("synthetic/pendulum/rough") is None


def test_an_image_whose_license_must_be_accepted_is_not_built_silently(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building Isaac Sim's image without accepting NVIDIA's license is one error line that
    says how to accept it, before Docker is even asked (``all`` skips it, never accepts)."""
    monkeypatch.delenv("R2S_ACCEPT_NVIDIA_EULA", raising=False)
    text = _fails_clearly(["env", "build", "r2s-video2world-isaac"], capsys)
    assert "R2S_ACCEPT_NVIDIA_EULA=1 r2s env build r2s-video2world-isaac" in text
    assert "NVIDIA_Omniverse_License_Agreement" in text


def test_a_program_directory_is_the_program_for_every_task_whatever_it_holds(
    tmp_path: Path,
) -> None:
    """``program:path=`` a directory with ``run.py`` is the program, as ``r2s train reward``
    takes it: a subdirectory with a ``run.py`` of its own (a helper, a vendored package) does
    not make it "a directory of per-task programs" holding none for the task. One that also
    holds a program under the task's id, or a segment of it, is ambiguous: refused, never
    one of the two picked silently."""
    from real2simlab.errors import SetupError
    from real2simlab.methods import get_method

    root = tmp_path / "program"
    (root / "tools").mkdir(parents=True)
    (root / "run.py").write_text("print('twin')\n")
    (root / "tools" / "run.py").write_text("print('helper')\n")
    locate = get_method("program", path=str(root)).locate  # pyright: ignore[reportAttributeAccessIssue]
    assert locate("synthetic/bouncing_ball/calibrated") == root
    for inside in (
        "calibrated",
        "synthetic/bouncing_ball/calibrated",
        "synthetic__bouncing_ball__calibrated",
    ):
        (root / inside).mkdir(parents=True)
        (root / inside / "run.py").write_text("print('another twin')\n")
        with pytest.raises(SetupError, match="which of them is meant") as raised:
            locate("synthetic/bouncing_ball/calibrated")
        assert f"({inside}/)" in str(raised.value) and raised.value.fix, inside
        assert locate("synthetic/pendulum/rough") == root  # nothing there is that task's
        shutil.rmtree(root / inside.split("/")[0])


def test_an_empty_solution_path_is_a_mistake_not_the_current_directory(
    bouncing_task: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--solution "$UNSET"`` and ``program:path=`` with nothing after it are mistakes in the
    command, as a path that is not there is, not the current directory: there, a reward of 0
    for a program never given, or the reward of whatever program the current directory is."""
    here = tmp_path / "here"
    shutil.copytree(bouncing_task / "hidden" / "generator" / "program", here)
    monkeypatch.chdir(here)  # the current directory is a program: it must not be taken
    experiment = tmp_path / "experiment.toml"
    experiment.write_text(
        f'name = "empty"\n[tasks]\npaths = ["{bouncing_task}"]\n'
        '[[methods]]\nname = "program"\npath = ""\n[run]\nbackend = "local"\n'
    )
    commands = {
        "train reward": ["train", "reward", "--task", str(bouncing_task), "--solution", "",
                         "--backend", "local"],
        "eval": ["eval", "--tasks", str(bouncing_task), "--method", "prog=program:path=",
                 "--backend", "local", "--out", str(tmp_path / "run")],
        "run": ["run", str(experiment), "--out", str(tmp_path / "run-file")],
        "export harbor": ["export", "harbor", str(bouncing_task), "--out",
                          str(tmp_path / "harbor"), "--solution", ""],
    }  # fmt: skip
    for name, argv in commands.items():
        capsys.readouterr()
        assert main(argv) == 1, name
        captured = capsys.readouterr()
        assert "is empty" in captured.err and "fix:" in captured.err, (name, captured.err)
        assert "Traceback" not in captured.err and "reward" not in captured.out, name
    for made in ("run", "run-file", "harbor"):
        assert not (tmp_path / made).exists(), made
    assert sorted(path.name for path in here.iterdir()) == sorted(
        path.name for path in (bouncing_task / "hidden" / "generator" / "program").iterdir()
    )


def test_an_empty_solution_is_refused_by_the_harbor_export_api_too(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``export_task(..., solution="")`` is the API's empty ``--solution``: refused before
    anything is written, not the current directory's program embedded in ``solve.sh``."""
    from real2simlab.bridges.harbor import export_task
    from real2simlab.errors import SetupError

    here = tmp_path / "here"
    shutil.copytree(bouncing_task / "hidden" / "generator" / "program", here)
    monkeypatch.chdir(here)
    with pytest.raises(SetupError, match="is empty") as raised:
        export_task(bouncing_task, tmp_path / "harbor", solution="")
    assert raised.value.fix and not (tmp_path / "harbor").exists()


# --- a credential given as an option -----------------------------------------------------------
# Options are written into the run's records (experiment.json, build_record.json) and some go
# on a command line: a credential among them is refused before anything runs, with the way to
# pass it instead. The keys below are made-up canaries.

FAKE_CANARY_KEY = "cnry-" + "option-key-7e3d"


def _scripted(tmp_path: Path) -> Path:
    """A mini_swe script: no model is called, whatever becomes of the option."""
    script = tmp_path / "script.json"
    script.write_text('["ls"]')
    return script


def _recording_clis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stand-ins for ``harbor``, ``codex`` and ``claude``, first on ``PATH``, that answer
    ``--version`` and otherwise only record the arguments they are given (in the returned
    directory): what the Lab would pass a real one is seen, and no model is called. The
    agents' own keys are set to made-up canaries, so nothing stops a run before it starts."""
    calls, tools = tmp_path / "calls", tmp_path / "bin"
    calls.mkdir()
    tools.mkdir()
    for name in ("harbor", "codex", "claude"):
        tool = tools / name
        tool.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "if sys.argv[1:] == ['--version']:\n"
            "    print('0.27.0')\n"
            "    sys.exit(0)\n"
            f"name = os.path.join({str(calls)!r}, '{name}-' + str(os.getpid()) + '.json')\n"
            "with open(name, 'w') as out:\n"
            "    json.dump(sys.argv[1:], out)\n"
            "sys.exit(1)\n"
        )
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("R2S_CODEX_API_KEY", "cnry-codex-own-key-not-real")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "cnry-claude-own-key-not-real")
    return calls


def _holding(value: str, *roots: Path) -> list[str]:
    return [
        path.name
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file() and value.encode() in path.read_bytes()
    ]


def _run_cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


@pytest.mark.parametrize(
    ("method", "fix"),
    [
        (f'mini_swe:scripted=SCRIPT,model_kwargs={{"api_key":"{FAKE_CANARY_KEY}"}}', "api_key_env"),
        (f'harbor:harbor=HARBOR,agent_kwargs={{"api_key":"{FAKE_CANARY_KEY}"}}', "agent_env"),
        (f'harbor:harbor=HARBOR,agent_env={{"OPENAI_API_KEY":"{FAKE_CANARY_KEY}"}}', "agent_env"),
        (f'harbor:harbor=HARBOR,agent_env={{"AWS_BEARER_TOKEN_BEDROCK":"{FAKE_CANARY_KEY}"}}',
         "agent_env"),
        (f'harbor:harbor=HARBOR,agent_env={{"MINI_SWE_AGENT_MODEL_API_KEY_JSON":"{FAKE_CANARY_KEY}"}}',
         "agent_env"),
        (f'codex:network=open,extra_args=["--api-key={FAKE_CANARY_KEY}"]', "environment"),
        (f"claude_code:network=open,base_url=https://qa:{FAKE_CANARY_KEY}@llm.example/v1", "environment"),
    ],
    ids=["mini_swe-model_kwargs", "harbor-agent_kwargs", "harbor-agent_env",
         "harbor-agent_env-word-inside", "harbor-agent_env-word-inside-2", "extra_args", "url"],
)  # fmt: skip
def test_a_credential_in_a_method_option_is_refused(
    method: str,
    fix: str,
    bouncing_task: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key given as a method option, under its name (also with the word inside the name, as
    in ``AWS_BEARER_TOKEN_BEDROCK``), with its flag, or in a URL, is refused before anything
    runs: it reaches no command line of the agent's CLI and no record of the run."""
    calls = _recording_clis(tmp_path, monkeypatch)
    run = tmp_path / "run"
    method = method.replace("SCRIPT", str(_scripted(tmp_path)))
    method = method.replace("HARBOR", str(tmp_path / "bin" / "harbor"))
    argv = ["eval", "--tasks", str(bouncing_task), "--method", method, "--backend", "local",
            "--out", str(run)]  # fmt: skip
    code, text = _run_cli(argv, capsys)
    assert _holding(FAKE_CANARY_KEY, run, calls) == []  # no record, no command line got it
    assert code == 1 and "Traceback" not in text, text
    assert FAKE_CANARY_KEY not in text  # the error names the option, never its value
    assert text.count("error:") == 1 and "fix:" in text
    assert fix in text.split("fix:", 1)[1]
    assert not run.exists() and not list(calls.iterdir())


def test_a_credential_in_an_experiment_file_is_refused(
    bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    experiment = tmp_path / "experiment.toml"
    experiment.write_text(
        f'name = "keys"\n[tasks]\npaths = ["{bouncing_task}"]\n[run]\nbackend = "local"\n'
        f'[[methods]]\nname = "mini_swe"\nscripted = "{_scripted(tmp_path)}"\n'
        f'model_kwargs = {{ api_key = "{FAKE_CANARY_KEY}", max_tokens = 64 }}\n'
    )
    run = tmp_path / "run"
    code, text = _run_cli(["run", str(experiment), "--out", str(run)], capsys)
    assert _holding(FAKE_CANARY_KEY, run) == []
    assert code == 1 and "Traceback" not in text, text
    assert FAKE_CANARY_KEY not in text and "model_kwargs.api_key" in text and "api_key_env" in text
    assert not run.exists()


@pytest.mark.parametrize("given", ["benchmark-name", "backend-value"])
def test_a_credential_in_a_plugin_option_is_refused(
    given: str, bouncing_task: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Benchmark, backend and scorer options are checked the same way as a method's: by name
    (a benchmark's ``hf_token``) and by value (a URL with a password, here in an option the
    local backend takes), before anything is written."""
    if given == "benchmark-name":
        out, named = tmp_path / "tasks", "hf_token"
        argv = ["bench", "build", "synthetic", "--select", "bouncing_ball", "--tier",
                "calibrated", "--option", f"hf_token={FAKE_CANARY_KEY}", "--out", str(out)]  # fmt: skip
    else:
        out, named = tmp_path / "run", "python"
        experiment = tmp_path / "experiment.toml"
        experiment.write_text(
            f'name = "keys"\n[tasks]\npaths = ["{bouncing_task}"]\n[run]\nbackend = "local"\n'
            f'backend_options = {{ python = "https://qa:{FAKE_CANARY_KEY}@llm.example/bin/python" }}\n'
            '[[methods]]\nname = "static"\n'
        )
        argv = ["run", str(experiment), "--out", str(out)]
    code, text = _run_cli(argv, capsys)
    assert _holding(FAKE_CANARY_KEY, out) == []
    assert code == 1 and "Traceback" not in text, text
    assert FAKE_CANARY_KEY not in text and named in text and "credential" in text and "fix:" in text
    assert not out.exists()


def test_a_credential_among_a_native_scorers_options_is_a_problem_of_the_task(
    task_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A key in ``[scoring.native.options.scorer]`` is one of the task file's problems, listed
    by ``r2s task validate`` like any other mistake there (a scorer takes no key from its
    options), and an evaluation names that file; neither shows the value."""
    from real2simlab.task import TaskConfig
    from real2simlab.task.io import read_task_config, write_task_config

    data = read_task_config(task_copy / "task.toml").to_json()
    data["scoring"] = {
        "protocol": "native",
        "native": {"scorer": "physimcode", "options": {"scorer": {"api_key": FAKE_CANARY_KEY}}},
    }
    write_task_config(TaskConfig.model_validate(data), task_copy / "task.toml")
    code, text = _run_cli(["task", "validate", str(task_copy)], capsys)
    assert code == 1 and "1 error(s)" in text and "Traceback" not in text, text
    assert "scoring.native.options.scorer" in text and "api_key holds a credential" in text
    assert FAKE_CANARY_KEY not in text
    code, text = _run_cli(_eval(task_copy, tmp_path / "run"), capsys)
    assert code == 1 and "Traceback" not in text, text
    assert FAKE_CANARY_KEY not in text and "credential" in text and str(task_copy) in text
    assert not (tmp_path / "run" / "results").exists()


def test_an_unset_variable_for_harbor_is_an_error_with_its_fix(
    bouncing_task: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``agent_env = {KEY = "$NAME"}`` with ``NAME`` not set stops the run before it starts,
    with the variable to set, and Harbor is never run with an empty key."""
    calls = _recording_clis(tmp_path, monkeypatch)
    monkeypatch.delenv("R2S_TEST_UNSET_KEY", raising=False)
    run = tmp_path / "run"
    method = (f"harbor:harbor={tmp_path / 'bin' / 'harbor'},"
              'agent_env={"OPENAI_API_KEY":"$R2S_TEST_UNSET_KEY"}')  # fmt: skip
    argv = ["eval", "--tasks", str(bouncing_task), "--method", method, "--backend", "local",
            "--out", str(run)]  # fmt: skip
    code, text = _run_cli(argv, capsys)
    assert code == 1 and "Traceback" not in text, text
    assert text.count("error:") == 1 and "R2S_TEST_UNSET_KEY" in text.split("error:", 1)[1]
    assert "fix: export R2S_TEST_UNSET_KEY=" in text
    assert not list(calls.iterdir()) and not run.exists()


NOT_CREDENTIALS = (
    "api_key_env", "VIBE_API_KEY_ENV", "api_key_name", "CODEX_AUTH_JSON_PATH",
    "GEMINI_OAUTH_CREDS_PATH", "AWS_ACCESS_KEY_ID", "token_type", "token_url", "max_tokens",
    "budget_tokens", "MAX_THINKING_TOKENS", "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    "LANGFUSE_PUBLIC_KEY", "GOOSE_DISABLE_KEYRING", "SSH_ASKPASS", "MODE",
)  # fmt: skip
"""Names that name, locate or count a credential, or hold none (Harbor's agents read them)."""


def test_options_that_only_name_a_credential_are_accepted() -> None:
    """Names of variables, limits on tokens, and references are not credentials."""
    from real2simlab.methods import get_method

    get_method(
        "mini_swe", model="openai/test", api_key_env="R2S_TEST_KEY", max_tokens=64,
        model_kwargs={"max_tokens": 64, "api_key": "$R2S_TEST_KEY"},
    )  # fmt: skip
    get_method("harbor", agent_env={"OPENAI_API_KEY": "$R2S_TEST_KEY", "MODE": "plain"})
    get_method("harbor", agent_env=dict.fromkeys(NOT_CREDENTIALS, "plain-value"))
    # a switch is not a credential, whatever its name says
    get_method("harbor", agent_env={"SKIP_CREDENTIAL_CHECKS": "1", "USE_API_KEY": "true"})
    get_method("codex", extra_args=["--skip-credential-checks=true", "--no-secrets", "x"])
    get_method(
        "opencode", model="custom/m", api_key_env="R2S_TEST_KEY", base_url="https://llm.example/v1"
    )
    get_method("codex", extra_args=["--api-key-env", "R2S_TEST_KEY"])


def test_a_credential_is_known_by_a_word_of_its_name_wherever_it_is() -> None:
    """A credential's word anywhere in a name, between underscores, says the option holds one;
    a name that ends by naming, locating or counting it does not. Every option the Lab's own
    plugins take, and every default of a method, passes."""
    import inspect

    from real2simlab import plugins
    from real2simlab.options import credential_option, credential_options

    held = ("api_key", "OPENAI_API_KEY", "AWS_BEARER_TOKEN_BEDROCK",
            "MINI_SWE_AGENT_MODEL_API_KEY_JSON", "AIDER_API_KEY_SETTINGS", "aws_secret_key",
            "aws_session_token", "client_secret", "hf_token", "GH_PAT", "Authorization",
            "accessToken", "LANGFUSE_SECRET_KEY", "password", "x-api-key")  # fmt: skip
    assert [name for name in held if not credential_option(name)] == []
    assert [name for name in NOT_CREDENTIALS if credential_option(name)] == []
    assert credential_options({"extra_args": ["--api-key", "v", "--bearer-token-x=v"]}) == [
        "extra_args[0]",
        "extra_args[2]",
    ]
    switches = ["--skip-credential-checks", "task", "--no-secrets", "x", "--api-key-env", "NAME"]
    assert credential_options({"extra_args": switches}) == []
    flagged: list[str] = []
    for kind in ("method", "benchmark", "backend", "scorer"):
        for name in plugins.names(kind):
            try:
                factory = plugins.load(kind, name)
            except ImportError:  # an extra that is not installed
                continue
            if kind == "method":
                fields = factory.Options.model_fields
                taken = list(fields)
                defaults = {k: f.get_default(call_default_factory=True) for k, f in fields.items()}
                flagged += credential_options({k: v for k, v in defaults.items() if v is not None})
            else:
                target = factory().build if kind == "benchmark" else factory
                taken = list(inspect.signature(target).parameters)
            flagged += [f"{kind} {name}: {option}" for option in taken if credential_option(option)]
    assert flagged == []


def test_a_credential_written_as_an_assignment_or_a_header_is_refused() -> None:
    """A key in a ``NAME=VALUE`` argument (codex's ``-c ...experimental_bearer_token=...``,
    ``--env OPENAI_API_KEY=...``) or in a ``Name: value`` header (``Authorization: Bearer
    ...``) is a credential among the options too; references to variables, settings and texts
    with a colon are not."""
    from real2simlab.options import credential_options

    key = "sk-qa-" + "c" * 24
    held = {
        "extra_args": ["-c", f"model_providers.qa.experimental_bearer_token={key}",
                       "--env", f"OPENAI_API_KEY={key}", "--header", f"Authorization: Bearer {key}"],
        "agent_kwargs": {"extra_env": f"OPENAI_API_KEY={key}"},
    }  # fmt: skip
    assert credential_options(held) == [
        "extra_args[1]", "extra_args[3]", "extra_args[5]", "agent_kwargs.extra_env"
    ]  # fmt: skip
    fine = {
        "extra_args": ["-c", "model=gpt-5", "-c", "model_providers.qa.env_key=OPENAI_API_KEY",
                       "--env", "OPENAI_API_KEY=${OPENAI_API_KEY}", "--header",
                       "Authorization: $TOKEN", "Note: keep it short", "a=b"],
        "prompt": "Answer: yes",
    }  # fmt: skip
    assert credential_options(fine) == []


def _into_a_closed_pipe(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``command`` with its stdout a pipe whose reader has gone (``... | head``, once
    ``head`` has what it wanted)."""
    read, write = os.pipe()
    os.close(read)
    try:
        return subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=write, stderr=subprocess.PIPE, text=True,
            timeout=300, check=False,
        )  # fmt: skip
    finally:
        os.close(write)


PRINTS_A_LOT = (
    "import argparse, sys\n"
    "from real2simlab.cli import main as cli\n"
    "def build_parser():\n"
    "    parser = argparse.ArgumentParser(prog='r2s')\n"
    "    parser.set_defaults(handler=lambda args: [print('x' * 99) for _ in range(10_000)] and 0)\n"
    "    return parser\n"
    "cli.build_parser = build_parser\n"
    "sys.exit(cli.main([]))\n"
)
"""``r2s`` with one command, which prints more than a pipe holds."""


@pytest.mark.skipif(sys.platform != "linux", reason="a widowed pipe as Linux's poll() reports it")
def test_a_command_whose_reader_stops_reading_ends_quietly() -> None:
    """``r2s demo | head``: whoever reads the output has stopped reading it. The command ends
    quietly with the status of a program that ``SIGPIPE`` ended (141), whether its output was
    left in the buffer or met the closed pipe while it ran. Python's own flush at exit printed
    two lines about a ``BrokenPipeError`` and ended with 120, and a print during the command
    was reported as a failed download, with the advice to check the network (A9R1-01)."""
    r2s = [sys.executable, "-m", "real2simlab.cli.main"]
    for command in ([*r2s, "bench", "list"], [sys.executable, "-c", PRINTS_A_LOT]):
        done = _into_a_closed_pipe(command)
        assert done.returncode == 141, (command[-2:], done.returncode, done.stderr[-2000:])
        for noise in ("Traceback", "Exception ignored", "BrokenPipeError", "download", "error:"):
            assert noise not in done.stderr, (command[-2:], done.stderr[-2000:])
    # Without any stdout (`r2s ... >&-`), there is nothing to flush: the command works.
    closed = subprocess.run(
        [*r2s, "bench", "list"], stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        timeout=300, check=False, preexec_fn=lambda: os.close(1),
    )  # fmt: skip
    assert closed.returncode == 0 and "Traceback" not in closed.stderr, closed.stderr[-2000:]


@contextlib.contextmanager
def _stdout_a_pipe(*, reader: bool) -> Iterator[None]:
    """``sys.stdout`` the writing end of a pipe while the block runs: with its reader there
    (``reader``), or gone, as for ``r2s ... | head`` once ``head`` has what it wanted."""
    read, write = os.pipe()
    if not reader:
        os.close(read)
    out = os.fdopen(write, "w", encoding="utf-8")
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(sys, "stdout", out)
            yield
    finally:
        with contextlib.suppress(OSError):  # what is left for a reader that has gone
            out.close()
        if reader:
            os.close(read)


EPIPE = (errno.EPIPE, os.strerror(errno.EPIPE))


@pytest.mark.skipif(sys.platform != "linux", reason="a widowed pipe as Linux's poll() reports it")
@pytest.mark.parametrize("debug", [False, True], ids=["", "R2S_DEBUG"])
def test_a_reader_that_stopped_reading_is_not_told_as_a_failed_download(
    debug: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every error a command raises is explained by ``_explain``, which tells a
    ``BrokenPipeError`` (a ``ConnectionError``) as a failed download: whether stdout's reader
    has gone is asked before it. A command printing into a pipe whose reader has gone ends
    quietly with 141, with ``R2S_DEBUG`` set too, as in the subprocess test above."""
    import real2simlab.cli.main as cli
    import real2simlab.cli.tasks as tasks_cli

    assert cli._explain(BrokenPipeError(*EPIPE)) is not None  # what asking first prevents
    if debug:
        monkeypatch.setenv("R2S_DEBUG", "1")
    else:
        monkeypatch.delenv("R2S_DEBUG", raising=False)

    def prints_a_lot(args: object) -> int:
        for _ in range(10_000):
            print("x" * 99)
        return 0

    monkeypatch.setattr(tasks_cli, "_list", prints_a_lot)
    with _stdout_a_pipe(reader=False):
        code = main(["task", "list"])
    err = capsys.readouterr().err
    assert code == 141 and err == "", (code, err)


@pytest.mark.skipif(sys.platform != "linux", reason="a widowed pipe as Linux's poll() reports it")
def test_a_broken_pipe_to_anything_but_the_reader_is_an_error_with_its_fix(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ``BrokenPipeError`` that is not stdout's (a tool the Lab feeds that died, such as
    ffmpeg given frames on its stdin) is not the reader stopping: with stdout's reader still
    there, the command fails with one ``error:`` line and a fix, not quietly with 141."""
    _raising(monkeypatch, lambda: BrokenPipeError(*EPIPE))
    with _stdout_a_pipe(reader=True):
        err = _fails_clearly(["task", "list"], capsys)
    assert "\n  fix: " in err, err


@pytest.mark.skipif(sys.platform != "linux", reason="a widowed pipe as Linux's poll() reports it")
@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (lambda: FileNotFoundError(2, "No such file or directory", "/data/tasks"),
         "error: /data/tasks does not exist\n  fix: "),
        (lambda: _while_handling(BrokenPipeError(*EPIPE), KeyError("rows")),
         "This is a bug in Real2SimLab"),
    ],
    ids=["a path error", "a bug raised while the closed pipe was handled"],
)  # fmt: skip
def test_a_command_that_fails_once_its_reader_has_gone_still_says_why(
    raised: Callable[[], BaseException],
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only the ``BrokenPipeError`` of stdout's reader having gone ends quietly: a command
    that fails for another reason meanwhile, or a bug raised while that error was handled,
    still says why on stderr, with the status 1 (``r2s eval ... | head`` hides no failure
    behind 141)."""
    _raising(monkeypatch, raised)
    with _stdout_a_pipe(reader=False):
        code = main(["task", "list"])
    err = capsys.readouterr().err
    assert code == 1 and expected in err, (code, err)


def test_the_demo_checks_the_ffmpeg_its_programs_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``r2s demo`` with ffmpeg only in ``R2S_FFMPEG`` and ``R2S_FFPROBE`` (none on PATH): the
    Lab renders with those, so the demo runs in a sandbox, whose programs use the image's
    ffmpeg; on the local backend its programs run with this machine's PATH, and the demo says
    that this is where ffmpeg is missing (not "install ffmpeg")."""
    from real2simlab import orchestrate
    from real2simlab.cli import demo
    from real2simlab.errors import SetupError

    tools, empty = tmp_path / "tools", tmp_path / "empty"
    for directory in (tools, empty):
        directory.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        (tools / name).write_text("#!/bin/sh\nexit 0\n")
        (tools / name).chmod(0o755)
        monkeypatch.setenv(f"R2S_{name.upper()}", str(tools / name))
    monkeypatch.setenv("PATH", str(empty))
    text = _fails_clearly(["demo", "--backend", "local", "--out", str(tmp_path / "local")], capsys)
    assert "local backend" in text and "PATH" in text and "\n  fix: put ffmpeg on PATH" in text
    assert not (tmp_path / "local").exists()

    def reached(*args: object, **kwargs: object) -> None:
        raise SetupError("the experiment was reached")

    monkeypatch.setattr(demo, "choose_backend", lambda requested, log=print: "bwrap")
    monkeypatch.setattr(orchestrate, "run_experiment", reached)
    text = _fails_clearly(["demo", "--backend", "bwrap", "--out", str(tmp_path / "bwrap")], capsys)
    assert "the experiment was reached" in text and "backend: bwrap" in text


def test_a_link_that_appears_in_a_task_during_a_run_is_one_error_with_its_fix(
    task_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    """A symbolic link put into a task after `r2s eval` validated it never reaches a method,
    but the next content hash of the task stops the run: that was `error: symbolic link in
    <task>: <path>` with no `fix:` line (hidden-data-validation, proposal section 3)."""
    from real2simlab.orchestrate import experiment

    validate = experiment.validate_task

    def validate_then_link(root: Path) -> object:
        report = validate(root)
        (Path(root) / "hidden" / "late").symlink_to(Path(root) / "task.toml")
        return report

    monkeypatch.setattr(experiment, "validate_task", validate_then_link)
    argv = ["eval", "--tasks", str(task_copy), "--method", "oracle", "--backend", "local",
            "--out", str(tmp_path / "run")]  # fmt: skip
    text = _fails_clearly(argv, capsys)
    assert "symbolic link in" in text and "hidden/late" in text, text
    assert "\n  fix: the Lab never follows a symbolic link" in text, text


def test_a_text_option_that_reads_as_json_says_how_to_give_it_as_text(
    bouncing_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    """``program:path=1`` (a program directory named ``1``) is read as the number 1: refused
    with "path: Input should be a valid string" and nothing on how to write it (the JSON-literal
    variant of A2-06). The error says it; written as JSON text, it runs."""
    program = tmp_path / "1"
    shutil.copytree(bouncing_task / "hidden" / "generator", program)
    argv = ["eval", "--tasks", str(bouncing_task), "--backend", "local", "--method"]
    out = ["--out", str(tmp_path / "run")]
    text = _fails_clearly([*argv, "program:path=1", *out], capsys)
    assert "path: Input should be a valid string" in text, text
    assert """--method 'program:path="1"'""" in text, text
    monkeypatch.chdir(tmp_path)  # the directory named 1 is found relative to here
    assert main([*argv, 'program:path="1"', *out]) == 0, capsys.readouterr().err
