#!/usr/bin/env python
"""Build an occupancy map from a recorded session and save it as an image.

Each scan is placed at the last recorded pose before it. With raw odometry
poses this is the "before" map: drift shows as smeared walls.

Usage:
    uv run python scripts/build_map.py data/sessions/<session>.jsonl [--every 4] [--resolution 0.05]
"""

import argparse
import logging
from pathlib import Path

from pepin.geometry import BaseGeometry
from pepin.lidar import LidarMount
from pepin.log import setup_logging
from pepin.mapping import GridSpec, OccupancyGrid
from pepin.odometry import DiffDriveOdometry, Pose2D
from pepin.recording import pose_from_record, read_session, scan_from_record

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("data/maps")


def main() -> None:
    parser = argparse.ArgumentParser(description="Occupancy map from a session.")
    parser.add_argument("session", type=Path)
    parser.add_argument("--every", type=int, default=4, help="use every N-th scan")
    parser.add_argument("--resolution", type=float, default=0.05, help="cell size, m")
    parser.add_argument("--size", type=float, default=16.0, help="map side length, m")
    parser.add_argument(
        "--track-width",
        type=float,
        help="re-integrate odometry from the recorded wheel travel with this track width, m",
    )
    args = parser.parse_args()
    setup_logging("build_map")

    mount = LidarMount.from_json("config/lidar.json")
    half = args.size / 2
    grid = OccupancyGrid(GridSpec(args.resolution, -half, -half, args.size, args.size))
    pose = Pose2D()
    path: list[tuple[float, float]] = []
    scans_used = 0
    odom = (
        DiffDriveOdometry(BaseGeometry(track_width_m=args.track_width))
        if args.track_width
        else None
    )
    for i, record in enumerate(read_session(args.session)):
        if record["topic"] == "pose":
            if odom is not None and "d_left" in record:
                pose = odom.update(record["d_left"], record["d_right"])
            else:
                pose = pose_from_record(record)
            path.append((pose.x, pose.y))
        elif record["topic"] == "scan" and i % args.every == 0:
            grid.integrate(pose, scan_from_record(record).points_xy(mount))
            scans_used += 1
    logger.info("integrated %d scans along a %d-pose path", scans_used, len(path))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_track{args.track_width:.3f}" if args.track_width else ""
    out = OUTPUT_DIR / f"{args.session.stem}_odom{suffix}.png"
    fig, ax = plt.subplots(figsize=(10, 10))
    extent = (-half, half, -half, half)
    ax.imshow(1 - grid.probability(), cmap="gray", origin="lower", extent=extent, vmin=0, vmax=1)
    xs, ys = zip(*path, strict=True)
    ax.plot(xs, ys, color="tab:red", linewidth=0.8, label="odometry path")
    ax.plot(xs[0], ys[0], "go", label="start")
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_title(f"Odometry-only occupancy map — {args.session.stem}{suffix}")
    ax.legend(loc="upper right")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    logger.info("map saved to %s", out)
    print(out)


if __name__ == "__main__":
    main()
