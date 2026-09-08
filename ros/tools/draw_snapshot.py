#!/usr/bin/env python3
"""Draw a navigation snapshot (from snapshot_ros.py) as a PNG: map, scan, robot, plan, costmap.

    uv run --group macos python ros/tools/draw_snapshot.py snapshot.json out.png [--zoom 4]

Grey = map occupied, light grey = unknown, blue tint = local costmap cost, green dots = laser
returns placed through map->laser (a wall drawn on a wall means AMCL and the laser TF are
right), red = footprint, orange = plan, black arrow = heading. Prints the share of scan points
that land on or next to occupied map cells — the numeric version of "does the scan fit".
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--zoom", type=int, default=4, help="pixels per map cell")
    args = ap.parse_args()
    d = json.loads(args.snapshot.read_text())
    m = d["map"]
    w, h, res, ox, oy, z = m["w"], m["h"], m["res"], m["ox"], m["oy"], args.zoom
    grid = np.array(m["data"], dtype=np.int16).reshape(h, w)  # row 0 = y_min
    img = Image.new("RGB", (w * z, h * z), (255, 255, 255))
    px = img.load()
    assert px is not None  # a just-created image always has pixel access
    for r in range(h):
        for c in range(w):
            v = grid[r, c]
            color = (215, 215, 215) if v < 0 else (90, 90, 90) if v >= 65 else None
            if color:
                for dy in range(z):
                    for dx in range(z):
                        px[c * z + dx, (h - 1 - r) * z + dy] = color
    draw = ImageDraw.Draw(img)

    def to_px(x: float, y: float) -> tuple[float, float]:
        return ((x - ox) / res * z, (h - (y - oy) / res) * z)

    if "local" in d:
        lc = d["local"]
        arr = np.array(lc["data"], dtype=np.int16).reshape(lc["h"], lc["w"])
        for r in range(lc["h"]):
            for c in range(lc["w"]):
                v = arr[r, c]
                if v > 0:
                    x0, y0 = to_px(lc["ox"] + c * lc["res"], lc["oy"] + (r + 1) * lc["res"])
                    x1, y1 = to_px(lc["ox"] + (c + 1) * lc["res"], lc["oy"] + r * lc["res"])
                    a = min(255, 60 + int(v * 1.5))
                    draw.rectangle([x0, y0, x1, y1], fill=(255 - a // 2, 255 - a // 2, 255))
    if d.get("plan"):
        draw.line([to_px(x, y) for x, y in d["plan"]], fill=(255, 140, 0), width=max(2, z // 2))
    fit = None
    if "scan" in d and "map_to_laser" in d:
        s, laser = d["scan"], d["map_to_laser"]
        cos_yaw, sin_yaw = math.cos(laser["yaw"]), math.sin(laser["yaw"])
        on_wall = 0
        total = 0
        for i, rng in enumerate(s["ranges"]):
            if rng is None or rng < 0.05:
                continue
            a = s["angle_min"] + i * s["inc"]
            lx, ly = rng * math.cos(a), rng * math.sin(a)
            wx = laser["x"] + cos_yaw * lx - sin_yaw * ly
            wy = laser["y"] + sin_yaw * lx + cos_yaw * ly
            col, row = int((wx - ox) / res), int((wy - oy) / res)
            total += 1
            if 0 <= row < h and 0 <= col < w:
                patch = grid[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2]
                if (patch >= 65).any():
                    on_wall += 1
            px_x, px_y = to_px(wx, wy)
            draw.ellipse([px_x - 1.5, px_y - 1.5, px_x + 1.5, px_y + 1.5], fill=(0, 170, 0))
        fit = on_wall / max(1, total)
    if d.get("footprint"):
        draw.polygon([to_px(x, y) for x, y in d["footprint"]], outline=(220, 0, 0), width=2)
    if "map_to_base_link" in d:
        b = d["map_to_base_link"]
        tail = to_px(b["x"], b["y"])
        head = to_px(b["x"] + 0.4 * math.cos(b["yaw"]), b["y"] + 0.4 * math.sin(b["yaw"]))
        draw.line([tail, head], fill=(0, 0, 0), width=3)
    amcl = d.get("amcl", {})
    ax, ay = amcl.get("x", float("nan")), amcl.get("y", float("nan"))
    label = f"AMCL ({ax:+.2f}, {ay:+.2f}) yaw {math.degrees(amcl.get('yaw', 0)):+.0f} deg"
    if fit is not None:
        label += f" | scan on map walls: {fit:.0%}"
    draw.text((6, 6), label, fill=(0, 0, 0))
    img.save(args.out)
    print(label, "->", args.out)


if __name__ == "__main__":
    main()
