#!/usr/bin/env python3
"""Time the whole-map search on this board with a recorded scan: how long until a carried robot
knows where it is. Runs inside the container (numpy only, no ROS).

    python3 /tools/bench_global_search.py /maps/flat3_slam.yaml /maps/rec/reloc_fail_XXXX.json
"""

import json
import math
import sys
import time

import numpy as np
import yaml

from pepin.localization import Localizer
from pepin.mapping import GridSpec, OccupancyGrid
from pepin.odometry import Pose2D


def read_pgm(path: str) -> np.ndarray:
    """A binary (P5) PGM as a uint8 array; no image library in the container."""
    with open(path, "rb") as f:
        data = f.read()
    tokens: list[bytes] = []
    pos = 0
    while len(tokens) < 4:
        while data[pos : pos + 1].isspace():
            pos += 1
        if data[pos : pos + 1] == b"#":
            pos = data.index(b"\n", pos) + 1
            continue
        end = pos
        while not data[end : end + 1].isspace():
            end += 1
        tokens.append(data[pos:end])
        pos = end
    pos += 1  # the single whitespace after maxval
    width, height = int(tokens[1]), int(tokens[2])
    return np.frombuffer(data[pos : pos + width * height], dtype=np.uint8).reshape(height, width)


def grid_from_pgm(yaml_path: str) -> OccupancyGrid:
    """map_server's trinary reading of the pgm, then the relocalizer's +-4 log-odds."""
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    folder = yaml_path.rsplit("/", 1)[0]
    img = read_pgm(f"{folder}/{meta['image']}").astype(np.float64)
    occ = (255.0 - img) / 255.0
    data = np.full(img.shape, -1, dtype=np.int16)
    data[occ > meta["occupied_thresh"]] = 100
    data[occ < meta["free_thresh"]] = 0
    data = np.flipud(data)
    ox, oy, _ = meta["origin"]
    res = meta["resolution"]
    spec = GridSpec(res, ox, oy, img.shape[1] * res, img.shape[0] * res)
    grid = OccupancyGrid(spec)
    grid.log_odds[:] = np.where(data >= 65, 4.0, np.where((data >= 0) & (data <= 35), -4.0, 0.0))
    return grid


def main() -> None:
    grid = grid_from_pgm(sys.argv[1])
    with open(sys.argv[2]) as f:
        points = np.array(json.load(f)["points"], dtype=np.float64)
    loc = Localizer(grid, Pose2D())
    for _ in range(2):
        t0 = time.perf_counter()
        best, confidence = loc.global_search(points)
        took = time.perf_counter() - t0
        print(
            f"whole map in {took:.2f} s -> ({best.pose.x:+.2f}, {best.pose.y:+.2f}, "
            f"{math.degrees(best.pose.theta):+.0f} deg) confidence {confidence:.2f}"
        )


if __name__ == "__main__":
    main()
