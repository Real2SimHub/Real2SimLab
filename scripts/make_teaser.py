"""Draw the README's teaser (SVG, light and dark): methods and benchmarks connected once.

    python scripts/make_icons.py docs/assets/icons     # first: the icons it embeds
    python scripts/make_teaser.py docs/assets/teaser

Writes ``teaser-light.svg`` and ``teaser-dark.svg``. Left: the methods (coding agents, Harbor
agents, Real2Sim pipelines). Middle: what Real2SimLab does with every delivery, and what it
gives training. Right: the benchmarks, each with the headline metric its own scorer reports.
Every method and benchmark shows its own icon and the institution that publishes it; an
ellipsis at the bottom of each list stands for the ones still to come, connected the same way.
No data is drawn. The icons are embedded, so the SVG stands alone.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

W = 1280
SANS = "-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"
ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"

THEMES = {
    "light": {
        "card": "#ffffff", "border": "#d0d7de", "ink": "#1f2328", "muted": "#59636e",
        "chip": "#f6f8fa", "accent": "#4f46e5", "accent_soft": "#eef2ff", "line": "#a5b4fc",
        "emphasis": "#6d28d9",
    },
    "dark": {
        "card": "#161b22", "border": "#30363d", "ink": "#e6edf3", "muted": "#9198a1",
        "chip": "#0d1117", "accent": "#a5b4fc", "accent_soft": "#1e2550", "line": "#4f46e5",
        "emphasis": "#c4b5fd",
    },
}  # fmt: skip

# (icon, name, institution icon or None, institution, detail)
Row = tuple[str, str, str | None, str, str]
METHODS: list[Row] = [
    ("codex", "Codex", "openai", "OpenAI", "coding agent"),
    ("claude-code", "Claude Code", "anthropic", "Anthropic", "coding agent"),
    ("gemini-cli", "Gemini CLI", "google", "Google", "coding agent"),
    ("opencode", "OpenCode", "anomaly", "Anomaly", "coding agent"),
    ("mini-swe-agent", "mini-swe-agent", "princeton", "Princeton University", "any LiteLLM model"),
    ("harbor", "Harbor agents", "laude", "Laude Institute", "via Harbor"),
    ("program", "your pipeline", None, "your own Real2Sim method", "its programs"),
]
BENCHMARKS: list[Row] = [
    ("synthetic", "Synthetic", None, "built into Real2SimLab", "reproduction skill"),
    ("iris", "IRIS", "hbku", "Hamad Bin Khalifa University", "reproduction skill"),
    ("physics-iq", "Physics-IQ", "deepmind", "Google DeepMind", "Physics-IQ score"),
    ("fourdcodebench", "4DCodeBench", "jhu", "Johns Hopkins University", "dynamic IoU"),
    ("physcodegym", "PhysCodeGym", "sjtu", "Shanghai Jiao Tong University", "motion IoU"),
    ("video2world", "Video2World", "aether", "Aether AI", "V2WScore"),
    ("physimcode", "PhySimCode", "cmu", "Carnegie Mellon University", "parameter recovery"),
]
STEPS = [
    ("step-build", "Build", "sandboxed; network: the model API only"),
    ("step-freeze", "Freeze", "fingerprint the delivered program"),
    ("step-rerun", "Re-run", "offline, read-only, recorded"),
    ("step-score", "Score", "the benchmark's official scorer"),
    ("step-report", "Report", "one record, one scorecard"),
]
ROW_H = 50
MORE_H = 34
"""The ellipsis row at the bottom of each list."""


def text(x: float, y: float, value: str, *, size: float, fill: str, weight: int = 400,
         anchor: str = "start") -> str:  # fmt: skip
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" font-family="{SANS}">{value}</text>'
    )


def embed(path: Path, x: float, y: float, *, height: float, centred: bool = False) -> str:
    """The SVG at ``path`` as a nested <svg> ``height`` high, its left edge (or its centre) at x."""
    source = path.read_text(encoding="utf-8")
    match = re.search(r'<svg[^>]*viewBox="0 0 ([\d.]+) ([\d.]+)"[^>]*>(.*)</svg>', source, re.S)
    if not match:
        raise SystemExit(f"{path}: expected an <svg> with a viewBox (run make_icons.py first)")
    w, h = float(match.group(1)), float(match.group(2))
    width = height * w / h
    if centred:
        x -= width / 2
    # Element ids must stay unique once several icons share one document.
    body = re.sub(r'(id="|url\(#)', rf"\g<1>{path.stem}-", match.group(3))
    return (
        f'<svg x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'viewBox="0 0 {w:g} {h:g}">{body}</svg>'
    )


def icon(name: str, x: float, y: float, size: float) -> str:
    return embed(ASSETS / "icons" / f"{name}.svg", x, y, height=size)


def card_height(rows: int) -> float:
    return 44 + ROW_H * rows + MORE_H + 8


def column(x: float, top: float, width: float, title: str, rows: list[Row],
           c: dict[str, str]) -> tuple[list[str], list[float], float]:  # fmt: skip
    """A card with a title, one row per item (icon, name, institution, detail) and an ellipsis
    row; returns the SVG, each row's centre y and the ellipsis row's."""
    height = card_height(len(rows))
    out = [
        f'<rect x="{x}" y="{top}" width="{width}" height="{height}" rx="12" fill="{c["card"]}" '
        f'stroke="{c["border"]}"/>',
        text(x + 16, top + 28, title, size=14, fill=c["muted"], weight=700),
    ]
    centres: list[float] = []
    for index, (name, label, org, institution, detail) in enumerate(rows):
        y = top + 44 + index * ROW_H
        h = ROW_H - 6
        out.append(
            f'<rect x="{x + 10}" y="{y}" width="{width - 20}" height="{h}" rx="9" '
            f'fill="{c["chip"]}"/>'
        )
        out.append(icon(name, x + 16, y + (h - 32) / 2, 32))
        out.append(text(x + 58, y + 19, label, size=14.5, fill=c["ink"], weight=650))
        if org is None:
            out.append(text(x + 58, y + 35, institution, size=11.5, fill=c["muted"]))
        else:
            out.append(icon(f"org-{org}", x + 58, y + 24, 14))
            out.append(text(x + 77, y + 35, institution, size=11.5, fill=c["muted"]))
        out.append(
            text(x + width - 22, y + h / 2 + 4, detail, size=12, fill=c["muted"], anchor="end")
        )
        centres.append(y + h / 2)
    more_y = top + 44 + len(rows) * ROW_H
    out.append(
        f'<rect x="{x + 10}" y="{more_y}" width="{width - 20}" height="{MORE_H - 6}" rx="9" '
        f'fill="none" stroke="{c["border"]}" stroke-dasharray="4 4"/>'
    )
    more = text(x + width / 2, more_y + (MORE_H - 6) / 2 + 7, "⋯", size=22, fill=c["muted"],
                weight=700, anchor="middle")  # fmt: skip
    out.append(more)
    return out, centres, more_y + (MORE_H - 6) / 2


def headline(c: dict[str, str]) -> str:
    """The one sentence above the figure, with "once" emphasised."""
    once = f'<tspan fill="{c["emphasis"]}" font-weight="800">once</tspan>'
    return (
        f'<text x="{W / 2:.1f}" y="34" font-size="19" font-weight="600" fill="{c["ink"]}" '
        f'text-anchor="middle" font-family="{SANS}">Connect a method {once} and it runs on every '
        f"benchmark; connect a benchmark {once} and every method runs on it.</text>"
    )


def draw(theme: str) -> str:
    c = THEMES[theme]
    left, card_w, gap = 20, 380, 50
    centre_x, mid_w = left + card_w + gap, 380
    right = centre_x + mid_w + gap
    top = 62
    methods, method_y, method_more = column(left, top, card_w, "ANY METHOD", METHODS, c)
    benches, bench_y, bench_more = column(
        right, top, card_w, "ANY BENCHMARK, SCORED ITS OWN WAY", BENCHMARKS, c
    )
    card_h = card_height(len(METHODS))
    height = top + card_h + 18
    mid_h = 340
    mid_top = top + (card_h - mid_h) / 2
    hub_y = mid_top + mid_h / 2
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" '
        f'height="{height}" role="img" aria-label="Any method and any benchmark connected once '
        f'through Real2SimLab: build, freeze, re-run, score, report">',
        f'<defs><marker id="arrow-{theme}" viewBox="0 0 10 10" refX="8" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" '
        f'fill="{c["line"]}"/></marker></defs>',
        headline(c),
    ]
    # Connectors: every method into the hub, the hub out to every benchmark (N + M, not N x M).
    for y in method_y:
        out.append(
            f'<path d="M{left + card_w},{y:.1f} C{left + card_w + 40},{y:.1f} '
            f'{centre_x - 40},{hub_y:.1f} {centre_x},{hub_y:.1f}" fill="none" '
            f'stroke="{c["line"]}" stroke-width="1.4" stroke-opacity="0.8"/>'
        )
    for y in bench_y:
        out.append(
            f'<path d="M{centre_x + mid_w},{hub_y:.1f} C{centre_x + mid_w + 40},{hub_y:.1f} '
            f'{right - 40},{y:.1f} {right - 2},{y:.1f}" fill="none" stroke="{c["line"]}" '
            f'stroke-width="1.4" stroke-opacity="0.8" marker-end="url(#arrow-{theme})"/>'
        )
    # The ones still to come: the same single connection, dashed.
    out.append(
        f'<path d="M{left + card_w},{method_more:.1f} C{left + card_w + 40},{method_more:.1f} '
        f'{centre_x - 40},{hub_y:.1f} {centre_x},{hub_y:.1f}" fill="none" '
        f'stroke="{c["line"]}" stroke-width="1.4" stroke-dasharray="4 4"/>'
    )
    out.append(
        f'<path d="M{centre_x + mid_w},{hub_y:.1f} C{centre_x + mid_w + 40},{hub_y:.1f} '
        f'{right - 40},{bench_more:.1f} {right - 2},{bench_more:.1f}" fill="none" '
        f'stroke="{c["line"]}" stroke-width="1.4" stroke-dasharray="4 4"/>'
    )
    out += methods + benches
    out.append(
        f'<rect x="{centre_x}" y="{mid_top:.1f}" width="{mid_w}" height="{mid_h}" rx="14" '
        f'fill="{c["accent_soft"]}" stroke="{c["accent"]}" stroke-width="1.6"/>'
    )
    mid = centre_x + mid_w / 2
    out.append(embed(ASSETS / f"logo-{theme}.svg", mid, mid_top + 15, height=33, centred=True))
    tagline = text(mid, mid_top + 70, "one task format · one run contract · one record",
                   size=12.5, fill=c["muted"], anchor="middle")  # fmt: skip
    out.append(tagline)
    for index, (name, step, detail) in enumerate(STEPS):
        y = mid_top + 104 + index * 40
        out.append(icon(name, centre_x + 20, y - 15, 30))
        out.append(text(centre_x + 62, y + 5, step, size=15, fill=c["ink"], weight=700))
        out.append(text(centre_x + 128, y + 5, detail, size=12.5, fill=c["muted"]))
    # Training: the same tasks and scorer, for learning rather than reporting.
    rule_y = mid_top + 104 + len(STEPS) * 40 - 14
    out.append(
        f'<line x1="{centre_x + 18}" y1="{rule_y:.1f}" x2="{centre_x + mid_w - 18}" '
        f'y2="{rule_y:.1f}" stroke="{c["accent"]}" stroke-opacity="0.35"/>'
    )
    y = rule_y + 26  # as far below the rule as the last step is above it
    out.append(icon("step-train", centre_x + 20, y - 15, 30))
    out.append(text(centre_x + 62, y + 5, "Train", size=15, fill=c["ink"], weight=700))
    train = text(centre_x + 128, y + 5, "training tasks, datasets, rewards", size=12.5,
                 fill=c["muted"])  # fmt: skip
    out.append(train)
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_prefix", type=Path, help="writes <prefix>-light.svg, -dark.svg")
    args = parser.parse_args()
    for theme in THEMES:
        path = args.out_prefix.with_name(f"{args.out_prefix.name}-{theme}.svg")
        path.write_text(draw(theme), encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
