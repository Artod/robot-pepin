#!/usr/bin/env python
"""Render a recorded drive as a real-time SLAM video: the map grows as the robot moves.

Replays the session through GraphSlam at the recorded pace and writes one
frame per --fps of session time, so the video lines up with a camera
recording started at the same moment. Output: data/videos/<session>_slam.mp4.

Usage:
    uv run python scripts/render_slam.py data/sessions/<session>.jsonl [--fps 10]
"""

import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from pepin.lidar import LidarMount
from pepin.log import setup_logging
from pepin.mapping import GridSpec, transform_to_world
from pepin.odometry import Pose2D
from pepin.recording import pose_from_record, read_session, scan_from_record
from pepin.slam import GraphSlam

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("data/videos")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time SLAM replay video from a session.")
    parser.add_argument("session", type=Path)
    parser.add_argument(
        "--fps", type=float, default=10.0, help="video frames per second of session time"
    )
    parser.add_argument("--size", type=float, default=12.0, help="map side shown, m")
    parser.add_argument("--resolution", type=float, default=0.05)
    args = parser.parse_args()
    setup_logging("render_slam")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required (brew install ffmpeg)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mount = LidarMount.from_json("config/lidar.json")
    half = args.size / 2
    slam = GraphSlam(GridSpec(args.resolution, -half, -half, args.size, args.size))
    records = list(read_session(args.session))
    t_start = records[0]["t"]
    frame_period = 1.0 / args.fps
    next_frame_t = t_start
    pose = Pose2D()
    latest_points = None
    path: list[tuple[float, float]] = []

    workdir = Path(tempfile.mkdtemp(prefix="slam_frames_"))
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    frame = 0
    extent = (-half, half, -half, half)
    for record in records:
        if record["topic"] == "pose":
            pose = pose_from_record(record)
        elif record["topic"] == "scan":
            latest_points = scan_from_record(record).points_xy(mount)
            slam.process(pose, latest_points)
            path.append((slam.pose.x, slam.pose.y))
        while record["t"] >= next_frame_t:
            ax.clear()
            ax.imshow(
                1 - slam.grid.probability(),
                cmap="gray",
                origin="lower",
                extent=extent,
                vmin=0,
                vmax=1,
            )
            if path:
                xs, ys = zip(*path, strict=True)
                ax.plot(xs, ys, color="tab:red", linewidth=1.2)
            if latest_points is not None:
                world = transform_to_world(latest_points, slam.pose)
                ax.scatter(world[:, 0], world[:, 1], s=2, color="gold")
            p = slam.pose
            ax.arrow(
                p.x, p.y, 0.3 * np.cos(p.theta), 0.3 * np.sin(p.theta), color="tab:blue", width=0.04
            )
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"SLAM  t={next_frame_t - t_start:5.1f}s  "
                f"keyframes={len(slam.keyframes)}  loops={len(slam.closures)}"
            )
            fig.savefig(workdir / f"{frame:05d}.png", bbox_inches="tight", pad_inches=0.05)
            frame += 1
            next_frame_t += frame_period
    plt.close(fig)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{args.session.stem}_slam.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(args.fps),
            "-i",
            str(workdir / "%05d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
    )
    shutil.rmtree(workdir, ignore_errors=True)
    logger.info("%d frames -> %s", frame, out)
    print(out)


if __name__ == "__main__":
    main()
