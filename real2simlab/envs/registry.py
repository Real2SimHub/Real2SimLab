"""Where published runtime images come from, and pulling them.

Image names are ``<prefix><environment>:<IMAGE_VERSION>`` locally (the canonical name; prefix
``R2S_IMAGE_PREFIX``, default the project's registry, ``real2simlab._project.IMAGE_REGISTRY``).
When an image is missing, it is pulled from the first source that has it, then tagged with its
canonical name:

1. the canonical name itself, when ``R2S_IMAGE_PREFIX`` points at a registry of your own;
2. each prefix in ``R2S_IMAGE_MIRRORS`` (comma separated, e.g. a mirror of ghcr.io closer to
   you, or your own registry);
3. the project's registry, :data:`DEFAULT_REGISTRIES` (with the default prefix, this is the
   canonical name).

Every default source is a namespace the project's owner controls, so a missing image is never
looked up under a name someone else could register. Only redistributable images are published
(``Environment.publishable``); the others, such as the Claude Code image, are always built
locally. Set ``R2S_NO_PULL=1`` to never pull.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass

from real2simlab._project import IMAGE_REGISTRY
from real2simlab.sandbox.process import run_captured

DEFAULT_REGISTRIES: tuple[str, ...] = (IMAGE_REGISTRY,)
"""Where the project publishes its images (the release workflow). Any other source, such as a
mirror of ghcr.io, is yours to name in ``R2S_IMAGE_MIRRORS``."""
MIRRORS_ENV = "R2S_IMAGE_MIRRORS"
NO_PULL_ENV = "R2S_NO_PULL"


@dataclass(frozen=True)
class PullResult:
    image: str
    """The canonical name it now has locally."""
    source: str | None
    """The reference it was pulled from; ``None`` if every source failed."""
    tried: tuple[str, ...]
    error: str | None = None


def mirror_prefixes() -> list[str]:
    raw = os.environ.get(MIRRORS_ENV, "")
    return [
        p.strip() if p.strip().endswith("/") else p.strip() + "/"
        for p in raw.split(",")
        if p.strip()
    ]


def candidate_sources(canonical: str, name_and_tag: str) -> list[str]:
    """Every reference to try for an image, in order, without duplicates. Mirrors come before
    the project's registry, also when the canonical name is the published image."""
    published = [prefix + name_and_tag for prefix in DEFAULT_REGISTRIES]
    sources = [] if canonical in published else [canonical]
    sources += [prefix + name_and_tag for prefix in mirror_prefixes()]
    sources += published
    unique: list[str] = []
    for source in sources:
        normalised = source.removeprefix("docker.io/")
        if normalised not in unique:
            unique.append(normalised)
    return unique


def pulling_disabled() -> bool:
    return os.environ.get(NO_PULL_ENV, "") not in {"", "0", "false", "no"}


def pull(
    canonical: str,
    name_and_tag: str,
    *,
    docker: str,
    log: Callable[[str], None] | None = None,
    timeout_s: float = 3600,
) -> PullResult:
    """Pull ``name_and_tag`` from the first source that has it and tag it ``canonical``."""
    say = log or (lambda line: print(line, file=sys.stderr, flush=True))
    tried: list[str] = []
    last_error: str | None = None
    for source in candidate_sources(canonical, name_and_tag):
        tried.append(source)
        say(f"pulling {source} ...")
        try:
            # Stopped with the command (Ctrl-C, an interrupted experiment).
            result = run_captured([docker, "pull", source], timeout_s=timeout_s)
        except (OSError, subprocess.TimeoutExpired) as error:
            last_error = str(error)
            continue
        if result.returncode != 0:
            last_error = (result.stderr.strip().splitlines() or ["unknown error"])[-1]
            continue
        if source != canonical:
            try:
                tagged = run_captured([docker, "tag", source, canonical], timeout_s=300)
            except (OSError, subprocess.TimeoutExpired) as error:
                last_error = str(error)
                continue
            if tagged.returncode != 0:
                last_error = tagged.stderr.strip() or "docker tag failed"
                continue
        say(f"pulled {source}" + (f" as {canonical}" if source != canonical else ""))
        return PullResult(canonical, source, tuple(tried))
    return PullResult(canonical, None, tuple(tried), last_error)
