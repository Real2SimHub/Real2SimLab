"""Runtime image builds: the command line, and what a failed build tells the user."""

from __future__ import annotations

import stat
import warnings
from pathlib import Path

import pytest

from real2simlab.envs import build, build_command, diagnose_build_failure
from real2simlab.errors import SetupError

PIP_TIMEOUT = """
#7 [3/5] RUN pip install --no-cache-dir numpy==2.1.3
#7 12.3 WARNING: Retrying (Retry(total=4)) after connection broken by
#7 12.3 'ReadTimeoutError("HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.")'
ERROR: failed to solve: process "/bin/sh -c pip install" did not complete successfully
"""
APT_DNS = """
#5 [2/5] RUN apt-get update
#5 2.1 Err:1 http://deb.debian.org/debian bookworm InRelease
#5 2.1   Temporary failure resolving 'deb.debian.org'
#5 2.2 W: Failed to fetch http://deb.debian.org/debian/dists/bookworm/InRelease
"""
NPM_RESET = (
    "npm error code ECONNRESET\nnpm error network request to https://registry.npmjs.org failed"
)
BASE_IMAGE = (
    "ERROR: failed to solve: python:3.11-slim: failed to resolve source metadata for "
    "docker.io/library/python:3.11-slim: failed to do request: Head "
    '"https://registry-1.docker.io/v2/library/python/manifests/3.11-slim": i/o timeout'
)


def test_network_failures_name_the_option() -> None:
    assert any("--pip-index-url" in hint for hint in diagnose_build_failure(PIP_TIMEOUT))
    assert any("--apt-mirror" in hint for hint in diagnose_build_failure(APT_DNS))
    assert any("--npm-registry" in hint for hint in diagnose_build_failure(NPM_RESET))
    base = diagnose_build_failure(BASE_IMAGE)
    assert "registry mirror" in base[0] and not any("--pip-index-url" in h for h in base)
    assert diagnose_build_failure(PIP_TIMEOUT)[-1].endswith("--network host")


ASSET_DOWNLOAD = """
#9 [4/6] RUN pip install --no-cache-dir mani_skill==3.0.1 && python -m mani_skill.utils.download_asset bridge_v2_real2sim -y
#9 41.0 Downloading https://files.pythonhosted.org/packages/mani_skill-3.0.1-py3-none-any.whl
#9 80.2 Successfully installed mani_skill-3.0.1 sapien-3.0.3
#9 81.0 Downloading https://huggingface.co/datasets/haosulab/ManiSkill_bridge_v2_real2sim/resolve/main/bridge_v2_real2sim_dataset.zip
#9 81.1 Traceback (most recent call last):
#9 81.1   File "/usr/lib/python3.11/urllib/request.py", line 1348, in do_open
#9 81.1 urllib.error.URLError: <urlopen error [Errno 101] Network is unreachable>
ERROR: failed to solve: process "/bin/sh -c pip install" did not complete successfully
"""


def test_a_download_after_pip_is_not_called_a_pip_failure() -> None:
    hints = diagnose_build_failure(ASSET_DOWNLOAD)
    assert not any("--pip-index-url" in hint for hint in hints), hints
    assert "huggingface.co" in hints[0] and hints[-1].endswith("--network host")


def test_other_failures_get_no_network_hint() -> None:
    assert diagnose_build_failure("E: Unable to locate package libfoo\nexit code: 100") == []


def test_failed_build_explains_itself(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\ncat <<'EOF'\n" + PIP_TIMEOUT + "\nEOF\nexit 1\n")
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
    with pytest.raises(SetupError) as failed:
        build("r2s-python", docker=str(docker))
    assert "--pip-index-url" in (failed.value.fix or "")
    assert "Read timed out" in capsys.readouterr().out  # the output is still streamed
    log = tmp_path / "build.log"
    with pytest.raises(SetupError):
        build("r2s-python", docker=str(docker), log=log)
    assert "Read timed out" in log.read_text()


def test_build_command_passes_mirrors(tmp_path: Path) -> None:
    command = build_command(
        "r2s-python",
        tmp_path,
        docker="docker",
        pip_index_url="https://pypi.example/simple/",
        apt_mirror="http://deb.example",
        network="host",
    )
    text = " ".join(command)
    assert "PIP_INDEX_URL=https://pypi.example/simple/" in text
    assert "APT_MIRROR=http://deb.example" in text and "--network host" in text


def test_an_environment_can_reuse_another_dockerfile(tmp_path: Path) -> None:
    from real2simlab.envs import build_context, get_environment

    context = build_context("r2s-agent-codex-blender", tmp_path / "ctx")
    assert "@openai/codex" in (context / "Dockerfile").read_text()
    command = build_command("r2s-agent-codex-blender", context)
    assert "BASE_IMAGE=" + get_environment("r2s-blender-4.0").image in command
    assert command[command.index("--tag") + 1].endswith("r2s-agent-codex-blender:1.0")


def _source_repo(tmp_path: Path) -> tuple[Path, str]:
    """A small git repository (two files, one commit) to vendor; and its commit."""
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    source = tmp_path / "source"
    source.mkdir()
    (source / "code.py").write_text("print('hi')\n")
    (source / "data.txt").write_text("some data\n" * 100)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com", "PATH": "/usr/bin:/bin"}  # fmt: skip
    for step in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", "-C", str(source), *step], check=True, env=env)
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()  # fmt: skip
    return source, commit


def test_vendored_checkout_survives_concurrent_callers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parallel scorers call ``ensure()`` at once: every caller gets the one checkout."""
    from concurrent.futures import ThreadPoolExecutor

    from real2simlab.vendor import VendoredRepo

    source, commit = _source_repo(tmp_path)
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    repo = VendoredRepo(name="demo", url=str(source), commit=commit, license="test")
    with ThreadPoolExecutor(max_workers=4) as pool:
        roots = list(pool.map(lambda _: repo.ensure(), range(4)))
    assert {str(r) for r in roots} == {str(repo.path())}
    assert (repo.path() / "code.py").is_file()
    assert [p.name for p in repo.path().parent.iterdir()] == [repo.path().name]


def test_damaged_vendored_checkout_is_fetched_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash can leave empty files and an empty marker in the cache: the next process checks
    the checkout before reusing it, and fetches it again."""
    from real2simlab import vendor
    from real2simlab._fs import DamagedCacheWarning

    source, commit = _source_repo(tmp_path)
    monkeypatch.setenv("R2S_CACHE_DIR", str(tmp_path / "cache"))
    repo = vendor.VendoredRepo(name="demo", url=f"file://{source}", commit=commit, license="t")
    root = repo.ensure()
    for damage in ("files", "marker", "both"):
        if damage in ("files", "both"):
            for name in ("code.py", "data.txt"):
                (root / name).write_bytes(b"")  # what a power cut leaves behind
        if damage in ("marker", "both"):
            (root / vendor.MARKER).write_text("")
        vendor._verified.clear()  # a new process
        with pytest.warns(DamagedCacheWarning, match="fetching it again"):
            assert repo.ensure() == root
        assert (root / "code.py").read_text() == "print('hi')\n"
        assert (root / "data.txt").read_text() == "some data\n" * 100
        assert repo.problem(root, "git") is None
    vendor._verified.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an intact checkout is reused silently
        assert repo.ensure() == root


PROXY = {"HTTPS_PROXY": "http://user:secret@127.0.0.1:7897", "HTTP_PROXY": "http://127.0.0.1:7897"}


def test_the_network_fix_fits_how_the_build_ran() -> None:
    """The last fix of a failed download is never a flag the build already had, and names the
    proxy this shell has (by name: a value may hold a password)."""

    def last(
        environ: dict[str, str],
        network: str | None = None,
        build_args: dict[str, str | None] | None = None,
    ) -> str:
        hints = diagnose_build_failure(
            ASSET_DOWNLOAD, network=network, build_args=build_args, environ=environ
        )
        return hints[-1]

    assert last({}).endswith("also add --network host")
    hint = last(PROXY)
    assert "--build-arg HTTPS_PROXY" in hint and "--build-arg HTTP_PROXY" in hint
    assert hint.endswith("--network host")
    assert "secret" not in hint and "7897" not in hint
    hint = last(PROXY, "host")  # the audit's second build
    assert "--build-arg HTTPS_PROXY" in hint and "--network host" not in hint
    passed = {"HTTPS_PROXY": None, "HTTP_PROXY": None}
    hint = last(PROXY, "host", passed)
    assert "already used" in hint and "--build-arg" not in hint
    hint = last({}, "host")
    assert "already used" in hint and "set a proxy" in hint and "add --network host" not in hint


def test_a_failed_build_behind_a_proxy_says_to_pass_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from real2simlab.envs import PROXY_VARIABLES

    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", PROXY["HTTPS_PROXY"])
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\ncat <<'EOF'\n" + ASSET_DOWNLOAD + "\nEOF\nexit 1\n")
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
    with pytest.raises(SetupError) as failed:
        build("r2s-python", docker=str(docker), network="host", log=tmp_path / "build.log")
    fix = failed.value.fix or ""
    assert "--build-arg HTTPS_PROXY" in fix and "also add --network host" not in fix
    assert "secret" not in fix


def test_a_build_argument_can_be_given_by_name() -> None:
    """``--build-arg HTTPS_PROXY`` (no value): Docker takes it from this shell, so a proxy's
    password is never on a command line."""
    from real2simlab.cli.twins import _build_args  # pyright: ignore[reportPrivateUsage]
    from real2simlab.errors import UsageError

    parsed = _build_args(["BLENDER_URL=https://mirror/b.tar.xz", "HTTPS_PROXY", "EMPTY="])
    assert parsed == {"BLENDER_URL": "https://mirror/b.tar.xz", "HTTPS_PROXY": None, "EMPTY": ""}
    command = build_command("r2s-python", Path("ctx"), build_args=parsed)
    assert "HTTPS_PROXY" in command and "EMPTY=" in command
    assert not any(part.startswith("HTTPS_PROXY=") for part in command)
    with pytest.raises(UsageError, match="KEY=VALUE"):
        _build_args(["=value"])


def _docker_that_cannot_save(tmp_path: Path, *, has_image: bool) -> str:
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        f"  image) exit {0 if has_image else 1};;\n"
        '  save) echo "write /out/x.tar: no space left on device" >&2; exit 1;;\n'
        "esac\nexit 2\n"
    )
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
    return str(docker)


def test_an_export_names_what_failed_not_always_a_rebuild(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``r2s env export --out DIR`` (a directory) and a failed write say how to fix the path;
    only an image that is not here is told to be built or pulled."""
    from real2simlab.cli.main import main
    from real2simlab.envs.rootfs import export_image

    image = "ghcr.io/real2simhub/r2s-python:1.0"
    folder = tmp_path / "saved-images"
    folder.mkdir()
    assert main(["env", "export", "r2s-python", "--out", str(folder)]) == 1  # before Docker
    err = capsys.readouterr().err
    assert "is a directory" in err and str(folder / "r2s-python.tar") in err
    assert "r2s env build" not in err and "Traceback" not in err
    with pytest.raises(SetupError) as failed:
        export_image(image, folder / "x.tar", _docker_that_cannot_save(tmp_path, has_image=True))
    assert "no space left" in str(failed.value) and "r2s env build" not in (failed.value.fix or "")
    assert "writable and has room" in (failed.value.fix or "")
    with pytest.raises(SetupError) as failed:
        export_image(image, folder / "x.tar", _docker_that_cannot_save(tmp_path, has_image=False))
    assert "r2s env build r2s-python" in (failed.value.fix or "")
