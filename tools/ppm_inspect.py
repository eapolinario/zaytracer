#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.13.*"
# dependencies = ["pillow"]
# ///
"""Crop, convert and measure regions of a rendered PPM or PNG.

Speckle is hard to argue about at full-frame scale, so this pulls a region out
at pixel resolution and reports how noisy it is. The noise metric is the mean
absolute difference between a pixel and its right and lower neighbours: a
converged smooth surface scores near zero, salt-and-pepper scores high.

    uv run tools/ppm_inspect.py image.ppm --out image.png
    uv run tools/ppm_inspect.py image.ppm --crop 760,170,1010,430 --zoom 3 --out tail.png
    uv run tools/ppm_inspect.py image.ppm --stats-grid 6x4
    uv run tools/ppm_inspect.py flip.ppm --count-red
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def local_noise(img: Image.Image) -> float:
    """Mean absolute neighbour difference, on a 0-255 scale."""
    w, h = img.size
    if w < 2 or h < 2:
        return 0.0
    px = img.load()
    total = 0
    count = 0
    for y in range(h - 1):
        for x in range(w - 1):
            r, g, b = px[x, y]
            for nx, ny in ((x + 1, y), (x, y + 1)):
                r2, g2, b2 = px[nx, ny]
                total += abs(r - r2) + abs(g - g2) + abs(b - b2)
                count += 3
    return total / count if count else 0.0


def mean_rgb(img: Image.Image) -> tuple[float, float, float]:
    w, h = img.size
    px = img.load()
    acc = [0, 0, 0]
    for y in range(h):
        for x in range(w):
            p = px[x, y]
            acc[0] += p[0]
            acc[1] += p[1]
            acc[2] += p[2]
    n = w * h
    return (acc[0] / n, acc[1] / n, acc[2] / n)


def red_fraction(img: Image.Image) -> tuple[float, float]:
    """Share of pixels carrying any red, and the mean red over those pixels.

    Written for the `normal-flip` view, where red is the signal and everything
    else is near-black background.
    """
    px = img.load()
    w, h = img.size
    hit = 0
    total = 0
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            if r > g + 12 and r > b + 12:
                hit += 1
                total += r
    return (hit / (w * h), (total / hit) if hit else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("image", type=Path)
    ap.add_argument("--crop", help="x0,y0,x1,y1 region to extract")
    ap.add_argument("--zoom", type=int, default=1, help="nearest-neighbour magnification for --out")
    ap.add_argument("--out", type=Path, help="write the (cropped) image here")
    ap.add_argument("--stats-grid", help="CxR: report mean and noise for each cell of a CxR grid")
    ap.add_argument("--count-red", action="store_true", help="report the share of reddish pixels")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    print(f"{args.image}: {img.size[0]}x{img.size[1]}")

    if args.stats_grid:
        cols, rows = (int(v) for v in args.stats_grid.lower().split("x"))
        w, h = img.size
        print(f"{'cell':>8}  {'mean rgb':>21}  {'noise':>7}")
        for r in range(rows):
            for c in range(cols):
                box = (c * w // cols, r * h // rows, (c + 1) * w // cols, (r + 1) * h // rows)
                cell = img.crop(box)
                m = mean_rgb(cell)
                print(f"  {c},{r:<4}  {m[0]:6.1f} {m[1]:6.1f} {m[2]:6.1f}  {local_noise(cell):7.2f}")
        return 0

    region = img
    if args.crop:
        box = tuple(int(v) for v in args.crop.split(","))
        if len(box) != 4:
            ap.error("--crop needs x0,y0,x1,y1")
        region = img.crop(box)
        print(f"crop {box} -> {region.size[0]}x{region.size[1]}")

    m = mean_rgb(region)
    print(f"mean rgb: {m[0]:.1f} {m[1]:.1f} {m[2]:.1f}")
    print(f"neighbour noise: {local_noise(region):.2f} / 255")

    if args.count_red:
        share, mean_red = red_fraction(region)
        print(f"reddish pixels: {share * 100:.2f}%  mean red over them: {mean_red:.1f}")

    if args.out:
        if args.zoom > 1:
            region = region.resize(
                (region.size[0] * args.zoom, region.size[1] * args.zoom), Image.NEAREST
            )
        region.save(args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
