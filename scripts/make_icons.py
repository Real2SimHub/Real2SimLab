"""Draw the icons used by the README, the teaser and the documentation (SVG).

    pip install -e ".[docs]"        # the generic glyphs come from the icon sets bundled with it
    python scripts/make_icons.py docs/assets/icons [--cache DIR]

Three kinds of icon:

* ``<project>.svg`` (48 x 48): each agent's and benchmark's own icon, as it publishes it: its
  logo, or its GitHub organization's avatar when it has none. An icon published only as a
  small raster whose mark the project also publishes as a vector is redrawn from the vector
  (``Mark``), so it is as sharp as the rest;
* ``org-<institution>.svg``: the institution that publishes it (a benchmark's first
  institution): its mark (a company's avatar, a university's seal or emblem) on a 48 x 48 tile;
* ``program.svg``, ``synthetic.svg`` and ``step-*.svg``: the Lab's own: generic glyphs from
  Material Design Icons (Pictogrammers, Apache-2.0) and the Real2SimLab mark.

Logos are fetched once from the URLs below into the cache (``~/.cache/real2simlab/icons``),
then scaled by ffmpeg and embedded as PNG (a ``Mark`` stays a vector path). They are their
owners' trademarks, used only to identify the software and benchmarks the Lab runs; NOTICE
lists them.
"""

from __future__ import annotations

import argparse
import base64
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SIZE = 48
SCALE = 2  # pixels per unit: icons stay sharp at twice their size
AVATAR = "https://avatars.githubusercontent.com/u/{}?v=4&s=256"
COMMONS = "https://upload.wikimedia.org/wikipedia/commons/"
ENWIKI = "https://upload.wikimedia.org/wikipedia/en/"  # files Wikipedia uses as non-free logos
AGENT = "real2simlab-icons/1.0 (https://github.com/Real2SimHub/Real2SimLab)"


@dataclass(frozen=True)
class Logo:
    url: str
    pad: bool = False
    """Place the image, scaled to fit, on a white tile (it is not a square icon itself)."""
    crop: str | None = None
    """An ffmpeg crop (``w:h:x:y``) applied first, e.g. the mark of a logo with text."""
    white: bool = False
    """Put a square icon with a transparent background on white, so it shows on dark pages."""


@dataclass(frozen=True)
class Mark:
    """One shape of a vector logo, its ``index``-th ``<path>``, redrawn as a vector on a tile of
    ``background``, filling ``share`` of it: the layout of the project's own small icon."""

    url: str
    index: int
    background: str
    share: float


PROJECTS: dict[str, Logo | Mark] = {
    "codex": Logo(AVATAR.format(14957082)),  # github.com/openai
    # Claude Code's favicon (the spark on a dark tile) exists only as a 32 px raster; the spark
    # is the first path of its documentation's logo (code.claude.com/docs).
    "claude-code": Mark(
        "https://mintcdn.com/claude-code/c5r9_6tjPMzFdDDT/logo/light.svg",
        index=0,
        background="#30302e",
        share=0.72,
    ),
    "gemini-cli": Logo("https://geminicli.com/icon.png"),
    "opencode": Logo("https://opencode.ai/favicon.svg"),
    "mini-swe-agent": Logo("https://mini-swe-agent.com/latest/assets/mini_square.svg", pad=True),
    "harbor": Logo(
        "https://raw.githubusercontent.com/harbor-framework/harbor/main/docs-mintlify/"
        "harbor-logo.png"
    ),
    "physics-iq": Logo("https://physics-iq.github.io/assets/duckie.png", pad=True),
    "video2world": Logo(AVATAR.format(327190302), white=True),  # github.com/AetherLabsAI
    "fourdcodebench": Logo("https://4dcodebench.com/logos/bunny/icon-512.png", pad=True),
    "physcodegym": Logo(
        "https://shaoshuai0605.github.io/PhysCodeGym-Project-Page/brand/physcodegym-icon.png",
        pad=True,
    ),
    "physimcode": Logo("https://sourajakundu.github.io/PhySimCode/assets/img/favicon.svg"),
    "iris": Logo(AVATAR.format(172189460)),  # github.com/KurbanIntelligenceLab
}
ORGS: dict[str, Logo] = {
    "openai": Logo(AVATAR.format(14957082)),
    "anthropic": Logo(AVATAR.format(76263028)),
    "google": Logo(AVATAR.format(1342004)),
    "anomaly": Logo(AVATAR.format(66570915)),  # github.com/anomalyco, maker of OpenCode
    "princeton": Logo(COMMONS + "a/ae/Princeton_University_Shield.svg", pad=True),
    "laude": Logo(AVATAR.format(183158770)),  # github.com/laude-institute
    "deepmind": Logo(AVATAR.format(8596759), white=True),
    "aether": Logo(AVATAR.format(327190302), white=True),
    "sjtu": Logo("https://vi.sjtu.edu.cn/img/base/Logo.png", pad=True),
    "hbku": Logo(COMMONS + "1/18/HBKU_Logo%2C_2016.jpg", pad=True, crop="270:270:522:74"),
    # The universities' seals, as their brand guidelines publish them (via English Wikipedia).
    "jhu": Logo(ENWIKI + "0/09/Johns_Hopkins_University%27s_Academic_Seal.svg", pad=True),
    "cmu": Logo(ENWIKI + "b/bb/Carnegie_Mellon_University_seal.svg", pad=True),
}
GLYPHS: dict[str, tuple[str, str]] = {
    "program": ("material/code-braces", "#475569"),
    "step-build": ("material/hammer-wrench", "#4f46e5"),
    "step-freeze": ("material/snowflake", "#4f46e5"),
    "step-rerun": ("material/replay", "#4f46e5"),
    "step-score": ("material/scale-balance", "#4f46e5"),
    "step-report": ("material/file-chart-outline", "#4f46e5"),
    "step-train": ("material/school-outline", "#0f766e"),
}


def fetch(logo: Logo | Mark, cache: Path) -> Path:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", logo.url.split("://", 1)[1])[-120:]
    path = cache / name
    if path.is_file():
        return path
    cache.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(logo.url, headers={"User-Agent": AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())
            return path
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == 3:
                raise SystemExit(f"cannot fetch {logo.url}: {error}") from error
            time.sleep(2 + 4 * attempt)
    raise AssertionError("unreachable")


def png(source: Path, width: int, height: int, crop: str | None = None) -> bytes:
    """``source`` (raster or SVG) as a PNG fitting ``width`` x ``height``, aspect kept."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.png"
        command = ["ffmpeg", "-v", "error", "-y"]
        if source.suffix.lower() == ".svg":  # rasterize large, then scale like any image
            command += ["-width", "1024", "-height", "1024", "-keep_ar", "1"]
        filters = [f"crop={crop}"] if crop else []
        filters.append(f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos")
        command += ["-i", str(source), "-vf", ",".join(filters), "-pix_fmt", "rgba"]
        subprocess.run([*command, "-frames:v", "1", str(out)], check=True)
        return out.read_bytes()


def size_of(data: bytes) -> tuple[int, int]:
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def image(data: bytes, x: float, y: float, width: float, height: float, clip: str = "") -> str:
    encoded = base64.b64encode(data).decode()
    return (
        f'<image x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}"{clip} '
        f'href="data:image/png;base64,{encoded}"/>'
    )


BORDER = (
    f'<rect x="0.5" y="0.5" width="{SIZE - 1}" height="{SIZE - 1}" rx="10.5" fill="none" '
    'stroke="#1f2328" stroke-opacity="0.12"/>'
)


def tile(source: Path, logo: Logo, key: str) -> str:
    """A 48 x 48 rounded tile: a square icon filling it, or an image on white."""
    clip = f"r2s-clip-{key}"
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" width="{SIZE}" '
        f'height="{SIZE}"><defs><clipPath id="{clip}"><rect width="{SIZE}" height="{SIZE}" '
        'rx="11"/></clipPath></defs>'
    )
    if not logo.pad:
        data = png(source, SIZE * SCALE, SIZE * SCALE, logo.crop)
        w, h = (v / SCALE for v in size_of(data))
        body = image(data, (SIZE - w) / 2, (SIZE - h) / 2, w, h, f' clip-path="url(#{clip})"')
        if logo.white:
            body = f'<rect width="{SIZE}" height="{SIZE}" rx="11" fill="#ffffff"/>' + body
        return head + body + BORDER + "</svg>\n"
    inner = 38
    data = png(source, inner * SCALE, inner * SCALE, logo.crop)
    w, h = (v / SCALE for v in size_of(data))
    return (
        head
        + f'<rect width="{SIZE}" height="{SIZE}" rx="11" fill="#ffffff"/>'
        + image(data, (SIZE - w) / 2, (SIZE - h) / 2, w, h)
        + BORDER
        + "</svg>\n"
    )


def extent(path: str, view: list[float]) -> tuple[float, float, float, float]:
    """The bounding box (x0, y0, x1, y1) of the SVG element ``path`` in the units of ``view``
    (a viewBox), measured on a rendering 4096 pixels across."""
    vx, vy, vw, vh = view
    k = 4096 / max(vw, vh)
    w, h = round(vw * k), round(vh * k)
    with tempfile.TemporaryDirectory() as tmp:
        svg = Path(tmp) / "mark.svg"
        svg.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vx:g} {vy:g} {vw:g} {vh:g}" '
            f'width="{w}" height="{h}">{path}</svg>'
        )
        command = ["ffmpeg", "-v", "error", "-width", str(w), "-height", str(h), "-i", str(svg)]
        command += ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "-"]
        alpha = subprocess.run(command, check=True, capture_output=True).stdout[3::4]
    rows = [alpha[i * w : (i + 1) * w] for i in range(h)]
    inked = [i for i, row in enumerate(rows) if row.strip(b"\0")]
    if not inked:
        raise SystemExit(f"nothing drawn by {path[:80]}")
    left = min(w - len(rows[i].lstrip(b"\0")) for i in inked)
    right = max(len(rows[i].rstrip(b"\0")) for i in inked)
    return vx + left / k, vy + inked[0] / k, vx + right / k, vy + (inked[-1] + 1) / k


def vector_tile(source: Path, mark: Mark) -> str:
    """``mark``'s path, still a vector, centred on a 48 x 48 rounded tile of its background."""
    svg = source.read_text(encoding="utf-8")
    view = re.search(r'<svg[^>]*\sviewBox="([-\d.\s]+)"', svg)
    paths = re.findall(r"<path\b[^>]*>", svg)
    if not view or len(paths) <= mark.index:
        raise SystemExit(f"{mark.url}: expected an <svg> with a viewBox and {mark.index + 1} paths")
    element = paths[mark.index]
    d = re.search(r'\sd="([-\d.\sA-Za-z,]+)"', element)
    fill = re.search(r'\sfill="(#[0-9A-Fa-f]{3,8})"', element)
    if not d:
        raise SystemExit(f"{mark.url}: path {mark.index} has no plain path data")
    path = f'<path d="{d.group(1)}" fill="{fill.group(1) if fill else "#000000"}"/>'
    x0, y0, x1, y1 = extent(path, [float(v) for v in view.group(1).split()])
    scale = SIZE * mark.share / max(x1 - x0, y1 - y0)
    tx, ty = SIZE / 2 - scale * (x0 + x1) / 2, SIZE / 2 - scale * (y0 + y1) / 2
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" width="{SIZE}" '
        f'height="{SIZE}"><rect width="{SIZE}" height="{SIZE}" rx="11" fill="{mark.background}"/>'
        f'<g transform="translate({tx:.3f} {ty:.3f}) scale({scale:.4f})">{path}</g>{BORDER}'
        "</svg>\n"
    )


def glyph_tile(name: str, background: str) -> str:
    """A white Material Design glyph on a coloured tile."""
    try:
        import material  # type: ignore[import-untyped]
    except ImportError as error:
        raise SystemExit('mkdocs-material is needed: pip install -e ".[docs]"') from error
    source = (Path(material.__file__).parent / "templates" / ".icons" / f"{name}.svg").read_text()
    paths = "".join(f'<path d="{d}"/>' for d in re.findall(r'<path[^>]*\sd="([^"]+)"', source))
    glyph = 28.0
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" width="{SIZE}" '
        f'height="{SIZE}"><rect width="{SIZE}" height="{SIZE}" rx="11" fill="{background}"/>'
        f'<g fill="#ffffff" transform="translate({(SIZE - glyph) / 2:g} {(SIZE - glyph) / 2:g}) '
        f'scale({glyph / 24:g})">{paths}</g></svg>\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/real2simlab/icons")
    args = parser.parse_args()
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for key, logo in PROJECTS.items():
        source = fetch(logo, args.cache)
        drawn = vector_tile(source, logo) if isinstance(logo, Mark) else tile(source, logo, key)
        files[f"{key}.svg"] = drawn
    for key, logo in ORGS.items():
        files[f"org-{key}.svg"] = tile(fetch(logo, args.cache), logo, f"org-{key}")
    for key, (name, background) in GLYPHS.items():
        files[f"{key}.svg"] = glyph_tile(name, background)
    mark = (Path(__file__).resolve().parent.parent / "docs/assets/logo-mark.svg").read_text()
    files["synthetic.svg"] = mark.replace('width="240" height="240"', 'width="48" height="48"', 1)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
        print(f"wrote {out / name} ({len(text) // 1024} KiB)")


if __name__ == "__main__":
    main()
