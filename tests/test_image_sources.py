"""Where images come from: pulling from mirrors and registries (fake docker), and downloading
an image tarball with a checksum (local HTTP server)."""

from __future__ import annotations

import hashlib
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from real2simlab._project import IMAGE_REGISTRY, REPOSITORY
from real2simlab.envs import DEFAULT_PREFIX, pull_environment
from real2simlab.envs.registry import DEFAULT_REGISTRIES, candidate_sources
from real2simlab.envs.rootfs import download_verified
from real2simlab.errors import SetupError


def fake_docker(tmp_path: Path, available: set[str]) -> tuple[str, Path]:
    """A docker that can pull only ``available`` references; every call is logged."""
    log = tmp_path / "calls.log"
    script = tmp_path / "docker"
    allowed = " ".join(sorted(available))
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{log}"\n'
        'if [ "$1" = "pull" ]; then\n'
        f'  for ref in {allowed}; do [ "$2" = "$ref" ] && exit 0; done\n'
        '  echo "Error response from daemon: manifest unknown" >&2; exit 1\n'
        "fi\n"
        'if [ "$1" = "tag" ]; then exit 0; fi\n'
        "exit 1\n"
    )
    script.chmod(0o755)
    return str(script), log


def test_sources_are_tried_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2S_IMAGE_MIRRORS", "mirror.example.com/r2s, other.example.org/x/")
    mirrors = ["mirror.example.com/r2s/r2s-python:1.0", "other.example.org/x/r2s-python:1.0"]
    # By default the canonical name is the published image: mirrors first, then the project's
    # registry, once.
    canonical = DEFAULT_PREFIX + "r2s-python:1.0"
    assert candidate_sources(canonical, "r2s-python:1.0") == [*mirrors, canonical]
    # A prefix of your own comes first; the project's registry is the last resort.
    sources = candidate_sources("registry.example.com/me/r2s-python:1.0", "r2s-python:1.0")
    assert sources == [
        "registry.example.com/me/r2s-python:1.0",
        *mirrors,
        DEFAULT_REGISTRIES[0] + "r2s-python:1.0",
    ]


def test_default_sources_belong_to_the_repository_owner() -> None:
    """Images are only ever looked up where the project publishes them: the GitHub Container
    Registry of the repository's owner (no namespace someone else could register)."""
    owner = REPOSITORY.removeprefix("https://github.com/").split("/")[0]
    assert f"ghcr.io/{owner.lower()}/" == IMAGE_REGISTRY
    assert DEFAULT_PREFIX == IMAGE_REGISTRY
    assert DEFAULT_REGISTRIES == (IMAGE_REGISTRY,)


def test_pull_from_a_mirror_tags_the_canonical_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("R2S_IMAGE_PREFIX", raising=False)
    monkeypatch.setenv("R2S_IMAGE_MIRRORS", "mirror.example.com/r2s/")
    docker, log = fake_docker(tmp_path, {"mirror.example.com/r2s/r2s-python:1.0"})
    image = pull_environment("r2s-python", docker=docker, log=lambda line: None)
    assert image == "ghcr.io/real2simhub/r2s-python:1.0"
    calls = log.read_text().splitlines()
    assert calls == [
        "pull mirror.example.com/r2s/r2s-python:1.0",
        "tag mirror.example.com/r2s/r2s-python:1.0 ghcr.io/real2simhub/r2s-python:1.0",
    ]


def test_pull_failure_explains_how_to_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("R2S_IMAGE_MIRRORS", raising=False)
    docker, _ = fake_docker(tmp_path, set())
    with pytest.raises(SetupError) as failed:
        pull_environment("r2s-python", docker=docker, log=lambda line: None)
    fix = failed.value.fix or ""
    assert "R2S_IMAGE_MIRRORS" in fix and "r2s env build r2s-python" in fix
    assert "--pip-index-url" in fix and "--network host" in fix
    assert "manifest unknown" in str(failed.value)


def test_unpublished_images_are_never_pulled(tmp_path: Path) -> None:
    docker, log = fake_docker(tmp_path, {"ghcr.io/real2simhub/r2s-agent-claude-code:1.0"})
    with pytest.raises(SetupError, match="not published") as failed:
        pull_environment("r2s-agent-claude-code", docker=docker, log=lambda line: None)
    assert failed.value.fix == "r2s env build r2s-agent-claude-code"
    assert not log.exists()


def test_docker_backend_pulls_a_missing_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from real2simlab.sandbox.docker import DockerBackend

    monkeypatch.delenv("R2S_IMAGE_PREFIX", raising=False)
    monkeypatch.delenv("R2S_NO_PULL", raising=False)
    state = tmp_path / "pulled"
    script = tmp_path / "docker"
    script.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "image" ]; then [ -f "{state}" ] && '
        """echo '{"Id": "sha256:abc", "RepoDigests": ["r@sha256:abc"]}' && exit 0; exit 1; fi\n"""
        f'if [ "$1" = "pull" ]; then touch "{state}"; exit 0; fi\n'
        "exit 1\n"
    )
    script.chmod(0o755)
    resolved = DockerBackend(docker=str(script)).resolve_image("ghcr.io/real2simhub/r2s-python:1.0")
    assert resolved.digest == "r@sha256:abc"
    monkeypatch.setenv("R2S_NO_PULL", "1")
    state.unlink()
    with pytest.raises(SetupError, match="not available locally") as missing:
        DockerBackend(docker=str(script)).resolve_image("ghcr.io/real2simhub/r2s-python:1.0")
    assert "r2s env pull r2s-python" in (missing.value.fix or "")


def test_docker_backend_info_without_docker() -> None:
    from real2simlab.sandbox.docker import DockerBackend

    info = DockerBackend(docker="/nonexistent/docker").info()
    assert info == {"backend": "docker", "isolated": True, "docker_version": None}


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    root = tmp_path / "www"
    root.mkdir()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, *args):  # type: ignore[no-untyped-def]
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", root
    finally:
        server.shutdown()


def test_download_checks_the_published_sha256(served: tuple[str, Path], tmp_path: Path) -> None:
    base, root = served
    payload = b"image tarball" * 1000
    (root / "img.tar.gz").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (root / "img.tar.gz.sha256").write_text(f"{digest}  img.tar.gz\n")
    target = download_verified(f"{base}/img.tar.gz", tmp_path / "out" / "img.tar.gz")
    assert target.read_bytes() == payload
    (root / "img.tar.gz.sha256").write_text("0" * 64 + "  img.tar.gz\n")
    with pytest.raises(SetupError, match="does not match"):
        download_verified(f"{base}/img.tar.gz", tmp_path / "bad" / "img.tar.gz")
    assert not (tmp_path / "bad" / "img.tar.gz").exists()
    explicit = download_verified(
        f"{base}/img.tar.gz", tmp_path / "explicit.tar.gz", expected_sha256=digest.upper()
    )
    assert explicit.is_file()
    with pytest.raises(SetupError, match="cannot fetch the checksum"):
        download_verified(f"{base}/missing.tar.gz", tmp_path / "missing.tar.gz")


PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


def test_a_download_without_an_answer_is_the_network_not_the_checksum(
    served: tuple[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A proxy (or a network) that does not answer: the fix offered `--sha256`, which only moves
    the wait to the image's own, longer download. It names the network and the proxy setting;
    `--sha256` is the fix only where the server says it has no checksum file, and a `--sha256`
    the file does not match is to be checked, not downloaded again and reported."""
    import socket
    import urllib.request

    base, root = served
    with socket.socket() as unused:  # a port nothing listens on: refused, as by a dead proxy
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{port}")
    monkeypatch.setattr(urllib.request, "_opener", None)  # proxies are read when it is made
    with pytest.raises(SetupError, match="cannot fetch the checksum") as raised:
        download_verified(f"{base}/img.tar.gz", tmp_path / "img.tar.gz")
    fix = raised.value.fix or ""
    assert "--sha256" not in fix and "HTTP_PROXY" in fix and "--from-tar" in fix, fix
    monkeypatch.delenv("HTTP_PROXY")
    monkeypatch.setattr(urllib.request, "_opener", None)
    with pytest.raises(SetupError, match="cannot fetch the checksum") as raised:
        download_verified(f"{base}/missing.tar.gz", tmp_path / "missing.tar.gz")
    assert "--sha256" in (raised.value.fix or "")
    (root / "img.tar.gz").write_bytes(b"image tarball")
    with pytest.raises(SetupError, match="does not match") as raised:
        download_verified(f"{base}/img.tar.gz", tmp_path / "x.tar.gz", expected_sha256="0" * 64)
    assert "check --sha256" in (raised.value.fix or "")
    with pytest.raises(SetupError, match="not a SHA-256"):
        download_verified(f"{base}/img.tar.gz", tmp_path / "y.tar.gz", expected_sha256="abc")
    assert not (tmp_path / "y.tar.gz").exists()


@pytest.mark.parametrize("given", [None, "b" * 64], ids=["published checksum", "--sha256"])
def test_an_image_the_server_does_not_have_is_not_a_network_failure(
    given: str | None, served: tuple[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The image itself is not at the URL (404) while its checksum is published, or with the
    `--sha256` a missing checksum file's fix asks for: the server answered, yet the fix was to
    check the network and proxy and run the same command again, which changes nothing. It says
    the server has no file at that URL, to check `--from-url`, and the tarball route."""
    import urllib.request

    base, root = served
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setattr(urllib.request, "_opener", None)  # proxies are read when it is made
    (root / "gone.tar.gz.sha256").write_text("a" * 64 + "  gone.tar.gz\n")
    out = tmp_path / "out"
    with pytest.raises(SetupError, match="cannot download") as raised:
        download_verified(f"{base}/gone.tar.gz", out / "gone.tar.gz", expected_sha256=given)
    fix = raised.value.fix or ""
    assert "no file at that URL" in fix and "--from-url" in fix and "--from-tar" in fix, fix
    assert "network" not in fix and "same command again" not in fix, fix
    assert not list(out.iterdir())  # no partial download left


@pytest.mark.parametrize("given", [None, "c" * 64], ids=["no --sha256", "--sha256"])
@pytest.mark.parametrize("exists", [True, False], ids=["a file", "no file"])
def test_a_path_on_this_machine_is_not_a_url_to_download(
    given: str | None, exists: bool, tmp_path: Path
) -> None:
    """`--from-url` given a path on this machine: the error said the file's checksum file
    holds no checksum and to pass `--sha256`; with it, a bare "unknown url type" without a fix.
    `--from-url` takes an http(s) URL: the fix is `--from-tar` (with that file, if it is one),
    and nothing is fetched."""
    local = tmp_path / "local.tar.gz"
    if exists:
        local.write_bytes(b"a tarball")
    with pytest.raises(SetupError, match="takes an http") as raised:
        download_verified(str(local), tmp_path / "out" / "local.tar.gz", expected_sha256=given)
    fix = raised.value.fix or ""
    assert "--from-tar" in fix and "--sha256" not in fix, fix
    assert "holds no checksum" not in str(raised.value), raised.value
    assert (str(local) in fix) is exists, fix
    assert not (tmp_path / "out").exists()
