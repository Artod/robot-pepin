#!/usr/bin/env python
"""Build an occupancy map from a recorded session and save it as an image.

Each scan is placed at the last recorded pose before it. With raw odometry
poses this is the "before" map: drift shows as smeared walls. With --match,
every scan's pose is corrected by scan matching against the map built so
far (odometry only supplies the guess and the motion between scans): the
"after" map.

Usage:
    uv run python scripts/build_map.py data/sessions/<session>.jsonl [--every 4] [--resolution 0.05]
"""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from pepin.geometry import BaseGeometry
from pepin.lidar import LidarMount
from pepin.log import setup_logging
from pepin.mapping import GridSpec, OccupancyGrid
from pepin.odometry import DiffDriveOdometry, Pose2D
from pepin.recording import pose_from_record, read_session, scan_from_record
from pepin.scanmatch import (
    CorrelativeMatcher,
    SearchWindow,
    apply_motion,
    relative_motion,
    should_keyframe,
)

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
    parser.add_argument(
        "--match", action="store_true", help="correct each scan's pose by scan matching"
    )
    parser.add_argument(
        "--flip-angles",
        action="store_true",
        help="negate recorded scan angles (sessions recorded before the lidar mirror fix)",
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
    matcher = CorrelativeMatcher(grid) if args.match else None
    corrected = Pose2D()  # pose of the last matched scan, in the map frame
    odom_at_last_scan = Pose2D()
    improved = 0
    for i, record in enumerate(read_session(args.session)):
        if record["topic"] == "pose":
            if odom is not None and "d_left" in record:
                pose = odom.update(record["d_left"], record["d_right"])
            else:
                pose = pose_from_record(record)
            if matcher is None:
                path.append((pose.x, pose.y))
        elif record["topic"] == "scan" and i % args.every == 0:
            scan = scan_from_record(record)
            if args.flip_angles:
                scan = replace(scan, angles=-scan.angles)
            points = scan.points_xy(mount)
            if matcher is None:
                grid.integrate(pose, points)
            else:
                motion = relative_motion(odom_at_last_scan, pose)
                if scans_used and not should_keyframe(motion):
                    continue  # sub-step drift is invisible to the search; wait for more motion
                guess = apply_motion(corrected, motion)
                result = matcher.match(guess, points, SearchWindow().widened_for(motion))
                improved += result.improved
                corrected, odom_at_last_scan = result.pose, pose
                grid.integrate(corrected, points)
                matcher.invalidate()
                path.append((corrected.x, corrected.y))
            scans_used += 1
    if matcher is not None:
        logger.info("scan matching improved on the odometry guess for %d scans", improved)
    logger.info("integrated %d scans along a %d-pose path", scans_used, len(path))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_track{args.track_width:.3f}" if args.track_width else ""
    suffix += "_matched" if args.match else "_odom"
    out = OUTPUT_DIR / f"{args.session.stem}{suffix}.png"
    fig, ax = plt.subplots(figsize=(10, 10))
    extent = (-half, half, -half, half)
    ax.imshow(1 - grid.probability(), cmap="gray", origin="lower", extent=extent, vmin=0, vmax=1)
    xs, ys = zip(*path, strict=True)
    ax.plot(xs, ys, color="tab:red", linewidth=0.8, label="odometry path")
    ax.plot(xs[0], ys[0], "go", label="start")
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_title(f"Occupancy map — {args.session.stem}{suffix}")
    ax.legend(loc="upper right")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    logger.info("map saved to %s", out)
    print(out)


if __name__ == "__main__":
    main()
