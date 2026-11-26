"""Inline SVG line plots for the static report (no JavaScript, no external files)."""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Series:
    label: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    color: str
    dash: str | None = None


def _ticks(low: float, high: float, count: int = 5) -> list[float]:
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return [low]
    step = 10 ** math.floor(math.log10((high - low) / count))
    for factor in (1, 2, 5, 10):
        if (high - low) / (step * factor) <= count:
            step *= factor
            break
    start = math.ceil(low / step) * step
    return [start + i * step for i in range(int((high - start) / step) + 1)]


def _shrink(low: float, high: float) -> int:
    """The power of two a plot divides values from ``low`` to ``high`` by, so that their range
    and its padding stay below the largest float: 0 below about 1e307."""
    return max(0, math.frexp(max(abs(low), abs(high)))[1] - 1020)


def line_plot(
    series: Sequence[Series],
    *,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 360,
    height: int = 220,
) -> str:
    """An SVG line plot of ``series``; NaN values break lines."""
    margin_l, margin_r, margin_t, margin_b = 48, 10, 24, 34
    xs = np.concatenate([s.x[np.isfinite(s.y)] for s in series]) if series else np.array([0.0])
    ys = np.concatenate([s.y[np.isfinite(s.y)] for s in series]) if series else np.array([0.0])
    if xs.size == 0:
        xs, ys = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    if x1 - x0 < 1e-12:
        x1 = x0 + 1
    # Values beyond about 1e307 (a diverging twin) are placed divided by a power of two, so
    # that their range and its padding stay finite; the tick labels show the values themselves.
    shrink = _shrink(y0, y1)
    y0, y1 = math.ldexp(y0, -shrink), math.ldexp(y1, -shrink)
    pad = (y1 - y0) * 0.08 or 0.05
    y0, y1 = y0 - pad, y1 + pad
    if y1 == y0:  # one huge value, which a pad of 0.05 does not change
        y0, y1 = y0 - abs(y0) * 0.08, y1 + abs(y1) * 0.08
    plot_w, plot_h = width - margin_l - margin_r, height - margin_t - margin_b

    def px(x: float) -> float:
        return margin_l + (x - x0) / (x1 - x0) * plot_w

    def py(y: float) -> float:  # y divided by 2**shrink
        return margin_t + (1 - (y - y0) / (y1 - y0)) * plot_h

    parts = [
        f'<svg class="plot" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{html.escape(title)}">',
        f'<text x="{margin_l}" y="15" class="plot-title">{html.escape(title)}</text>',
        f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" class="plot-frame"/>',
    ]
    for tick in _ticks(y0, y1):
        value = tick * 2.0**shrink if shrink else tick
        if not math.isfinite(value):  # a tick beyond the largest float, past the values
            continue
        y = py(tick)
        parts.append(
            f'<line x1="{margin_l}" x2="{margin_l + plot_w}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{margin_l - 4}" y="{y + 3:.1f}" class="tick" text-anchor="end">{value:g}</text>'
        )
    for tick in _ticks(x0, x1):
        x = px(tick)
        parts.append(
            f'<text x="{x:.1f}" y="{margin_t + plot_h + 13}" class="tick" text-anchor="middle">{tick:g}</text>'
        )
    parts.append(
        f'<text x="{margin_l + plot_w / 2}" y="{height - 4}" class="axis" text-anchor="middle">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="11" y="{margin_t + plot_h / 2}" class="axis" text-anchor="middle" '
        f'transform="rotate(-90 11 {margin_t + plot_h / 2})">{html.escape(y_label)}</text>'
    )
    for s in series:
        segments: list[list[str]] = [[]]
        for x, y in zip(s.x, s.y, strict=False):
            if not (math.isfinite(x) and math.isfinite(y)):
                segments.append([])
                continue
            segments[-1].append(f"{px(float(x)):.1f},{py(math.ldexp(float(y), -shrink)):.1f}")
        dash = f' stroke-dasharray="{s.dash}"' if s.dash else ""
        for points in segments:
            if len(points) > 1:
                parts.append(
                    f'<polyline points="{" ".join(points)}" fill="none" stroke="{s.color}" '
                    f'stroke-width="1.8"{dash}><title>{html.escape(s.label)}</title></polyline>'
                )
    legend_x = margin_l + 6
    for index, s in enumerate(series):
        y = margin_t + 12 + 13 * index
        dash = f' stroke-dasharray="{s.dash}"' if s.dash else ""
        parts.append(
            f'<line x1="{legend_x}" x2="{legend_x + 16}" y1="{y - 3}" y2="{y - 3}" stroke="{s.color}" stroke-width="2"{dash}/>'
        )
        parts.append(
            f'<text x="{legend_x + 20}" y="{y}" class="legend">{html.escape(s.label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)
