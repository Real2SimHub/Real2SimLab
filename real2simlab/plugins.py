"""Plugin discovery through Python entry points.

Everything that can be added without changing the core (benchmarks, methods, metrics, output
formats, sandbox backends, native scorers, CLI commands) is looked up by name in
an entry point group. Real2SimLab registers its own built-ins the same way, in
``pyproject.toml``, so a third-party package is indistinguishable from a built-in.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from real2simlab.errors import R2SError, UnknownNameError

GROUPS = {
    "benchmark": "real2simlab.benchmarks",
    "method": "real2simlab.methods",
    "metric": "real2simlab.metrics",
    "format": "real2simlab.formats",
    "backend": "real2simlab.backends",
    "scorer": "real2simlab.scorers",
    "converter": "real2simlab.converters",
    "cli": "real2simlab.cli",
}
"""Plugin kind → entry point group."""


class PluginLoadError(R2SError, ImportError):
    """An entry point exists but cannot be imported."""


@dataclass(frozen=True)
class Plugin:
    """One registered plugin: its kind, name and where it is defined."""

    kind: str
    name: str
    value: str
    distribution: str | None

    def load(self) -> Any:
        """Import and return the object the entry point names."""
        return _load(self.kind, self.name)


@cache
def _entry_points(kind: str) -> dict[str, EntryPoint]:
    group = GROUPS[kind]
    found: dict[str, EntryPoint] = {}
    for point in entry_points(group=group):
        if point.name in found and found[point.name].value != point.value:
            raise PluginLoadError(
                f"two {kind} plugins are both named {point.name!r}: "
                f"{found[point.name].value} and {point.value}"
            )
        found[point.name] = point
    return found


@cache
def _load(kind: str, name: str) -> Any:
    points = _entry_points(kind)
    if name not in points:
        raise UnknownNameError(kind, name, points)
    try:
        return points[name].load()
    except Exception as error:  # an import error in a plugin must name the plugin
        raise PluginLoadError(f"cannot load {kind} plugin {name!r}: {error}") from error


def names(kind: str) -> list[str]:
    """The names registered for ``kind``, sorted."""
    return sorted(_entry_points(kind))


def plugins(kind: str) -> list[Plugin]:
    """Every plugin registered for ``kind``, sorted by name."""
    return [
        Plugin(
            kind=kind,
            name=name,
            value=point.value,
            distribution=point.dist.name if point.dist is not None else None,
        )
        for name, point in sorted(_entry_points(kind).items())
    ]


def load(kind: str, name: str) -> Any:
    """Load the plugin ``name`` of ``kind``; raise ``UnknownNameError`` if none is registered."""
    return _load(kind, name)


def load_all(kind: str) -> dict[str, Any]:
    """Load every plugin of ``kind``."""
    return {name: _load(kind, name) for name in names(kind)}


def clear_cache() -> None:
    """Forget discovered entry points, e.g. after installing a plugin in the same process."""
    _entry_points.cache_clear()
    _load.cache_clear()


Factory = Callable[..., Any]
