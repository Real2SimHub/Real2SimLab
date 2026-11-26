"""Dotted field paths that name parts of a capture, e.g. ``views.*.depth``.

A path is a dot-separated list of keys into the JSON form of a ``Capture``. The wildcard ``*``
matches every key of a mapping, which is how per-view fields are named.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import Any, cast

WILDCARD = "*"

ALWAYS_VISIBLE: tuple[str, ...] = (
    "spec",
    "capture_id",
    "views.*.rgb",
    "views.*.frame_range",
)
"""Fields that are always kept, since a capture is invalid without them."""


def parse(path: str) -> tuple[str, ...]:
    """Split a path into its keys, rejecting empty keys."""
    keys = tuple(path.split("."))
    if not path or any(not key for key in keys):
        raise ValueError(f"invalid field path {path!r}")
    return keys


def is_present(data: Mapping[str, Any], path: str) -> bool:
    """Whether ``path`` exists and is not None; with a wildcard, it must hold for every key."""
    return _is_present(data, parse(path))


def select(data: Mapping[str, Any], paths: Iterable[str]) -> dict[str, Any]:
    """A deep copy of ``data`` that keeps only ``paths`` plus ``ALWAYS_VISIBLE``."""
    selected: dict[str, Any] = {}
    for path in (*ALWAYS_VISIBLE, *paths):
        _copy_path(data, selected, parse(path))
    return selected


def removed(full: Mapping[str, Any], kept: Mapping[str, Any]) -> list[str]:
    """The paths of non-empty values in ``full`` that are absent from ``kept``."""
    return _removed(full, kept, ())


def _is_present(data: Any, keys: tuple[str, ...]) -> bool:
    if not keys:
        return data is not None
    if not isinstance(data, Mapping):
        return False
    mapping = cast(Mapping[str, Any], data)
    head, rest = keys[0], keys[1:]
    if head == WILDCARD:
        return bool(mapping) and all(_is_present(child, rest) for child in mapping.values())
    return head in mapping and _is_present(mapping[head], rest)


def _copy_path(source: Mapping[str, Any], target: dict[str, Any], keys: tuple[str, ...]) -> None:
    head, rest = keys[0], keys[1:]
    for key in list(source) if head == WILDCARD else [head]:
        value = source.get(key)
        if value is None:
            continue
        if not rest:
            target[key] = copy.deepcopy(value)
        elif isinstance(value, Mapping):
            _copy_path(cast(Mapping[str, Any], value), target.setdefault(key, {}), rest)


def _removed(
    full: Mapping[str, Any], kept: Mapping[str, Any], prefix: tuple[str, ...]
) -> list[str]:
    found: list[str] = []
    for key, value in full.items():
        if value is None or value == {} or value == []:
            continue
        path = (*prefix, key)
        kept_value = kept.get(key)
        if kept_value is None:
            found.append(".".join(path))
        elif isinstance(value, Mapping) and isinstance(kept_value, Mapping):
            found.extend(
                _removed(cast(Mapping[str, Any], value), cast(Mapping[str, Any], kept_value), path)
            )
    return found
