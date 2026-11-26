"""Draw the repository's social preview image (1280 x 640 PNG): the logo, the tagline and the
README's teaser.

    python scripts/make_social_preview.py docs/assets/logo-light.svg \\
        docs/assets/teaser-light.svg docs/assets/social-preview.png

The teaser is embedded unchanged. Rasterized with ffmpeg's librsvg decoder, so no Python
plotting or SVG dependency is needed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

W, H = 1280, 640
INK = "#1f2328"
FONT = "-apple-system, Segoe UI, Helvetica, Arial, sans-serif"
TAGLINE = "A unified platform for real-to-simulation"


def embed(path: Path, x: float, y: float, width: float) -> tuple[str, float]:
    """``path``'s drawing as a nested <svg> at (x, y), scaled to ``width``; and its height."""
    source = path.read_text(encoding="utf-8")
    match = re.search(r'<svg[^>]*viewBox="0 0 (\d+) (\d+)"[^>]*>(.*)</svg>', source, re.S)
    if not match:
        raise SystemExit(f"{path}: expected an <svg> with a viewBox")
    w, h, body = int(match.group(1)), int(match.group(2)), match.group(3)
    height = width * h / w
    nested = (
        f'<svg x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'viewBox="0 0 {w} {h}">{body}</svg>'
    )
    return nested, height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logo", type=Path, help="the wordmark (SVG)")
    parser.add_argument("teaser", type=Path, help="the README's teaser (SVG)")
    parser.add_argument("out", type=Path, help="the PNG to write")
    args = parser.parse_args()
    logo_w = 560
    logo, logo_h = embed(args.logo, (W - logo_w) / 2, 26, logo_w)
    teaser_w = 1110
    teaser_y = 184
    teaser, teaser_h = embed(args.teaser, (W - teaser_w) / 2, teaser_y, teaser_w)
    if teaser_y + teaser_h > H:
        raise SystemExit(f"the teaser ({teaser_h:.0f} high) does not fit below the tagline")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"
     viewBox="0 0 {W} {H}" font-family="{FONT}">
<rect width="{W}" height="{H}" fill="#ffffff"/>
{logo}
<text x="{W // 2}" y="{26 + logo_h + 58:.0f}" font-size="34" font-weight="700" fill="{INK}"
      text-anchor="middle">{TAGLINE}</text>
{teaser}
</svg>
"""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "preview.svg"
        source.write_text(svg, encoding="utf-8")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-frames:v", "1", str(args.out)],
            check=True,
        )
    print(args.out)


if __name__ == "__main__":
    main()
