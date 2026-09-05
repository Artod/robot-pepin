#!/usr/bin/env python
"""Replay a recorded navigate session through the Navigator, offline, deterministically.

The session file has everything the navigator saw (odometry, revolutions,
ToF) and everything it decided (mode, reason, the twist). Feeding the same
senses into a fresh Navigator with today's code shows what the robot *would*
do now — the way to check a fix against a real run instead of a synthetic
room. The replay is open-loop: the recorded poses do not react to the new
decisions, so it answers "would it have moved, and why not" rather than
"where would it have ended up".

Usage:
    uv run python scripts/replay_nav.py data/sessions/<name>.jsonl --map data/maps/<map>.npz
"""

import argparse
import collections
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from pepin.feeds import Sense
from pepin.kinematics import STOP
from pepin.lidar import LaserScan
from pepin.mapping import OccupancyGrid
from pepin.navigator import Navigator, NavigatorConfig
from pepin.odometry import Pose2D
from pepin.recording import read_session, scan_from_record
from pepin.robot import RobotConfig
from pepin.tof import TofRanges


def ticks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group the flat record stream into ticks: everything up to and including each ``cmd``."""
    out: list[dict[str, Any]] = []
    group: dict[str, Any] = {"scans": []}
    for r in records:
        topic = r["topic"]
        if topic == "scan":
            group["scans"].append(r)
        elif topic == "cmd":
            group["cmd"] = r
            out.append(group)
            group = {"scans": []}
        else:
            group[topic] = r
    return out


def start_pose(records: list[dict[str, Any]], fallback: Pose2D) -> Pose2D:
    """The ``--init`` pose from the run's note, when the note recorded one."""
    for r in records:
        if r["topic"] == "note":
            m = re.search(r"from \[([-\d.]+), ([-\d.]+), ([-\d.]+)\]", r.get("text", ""))
            if m:
                x, y, th = (float(v) for v in m.groups())
                return Pose2D(x, y, math.radians(th))
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a navigate session offline.")
    parser.add_argument("session", type=Path)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--goal", nargs=2, type=float, default=None, help="default: from the note")
    parser.add_argument("--no-tof", action="store_true")
    parser.add_argument("--trace", type=int, default=0, help="print the first N ticks")
    args = parser.parse_args()

    records = list(read_session(args.session))
    config = RobotConfig.load()
    mount, hull = config.lidar_mount, config.footprint
    goal = tuple(args.goal) if args.goal else None
    if goal is None:
        for r in records:
            if r["topic"] == "note":
                m = re.search(r"navigate to .*?\(([-\d.]+), ([-\d.]+)\)", r.get("text", ""))
                if m:
                    goal = (float(m.group(1)), float(m.group(2)))
    if goal is None:
        raise SystemExit("no goal in the session note; pass --goal X Y")
    start = start_pose(records, Pose2D())
    nav = Navigator(
        OccupancyGrid.load(args.map),
        start,
        goal,
        NavigatorConfig(
            tof_mounts={} if args.no_tof else config.tof_mounts, footprint=config.footprint
        ),
    )

    def points(scan: LaserScan) -> np.ndarray:
        pts = scan.points_xy(mount)
        return pts[~hull.inside(pts, margin_m=0.0)]  # what Robot.observe hands the navigator

    reasons: collections.Counter[str] = collections.Counter()
    modes: collections.Counter[str] = collections.Counter()
    recorded: collections.Counter[str] = collections.Counter()
    forward = still = 0
    last_scan_t: float | None = None
    t0 = records[0]["t"]
    for tick in ticks(records):
        cmd, odom = tick.get("cmd"), tick.get("pose")
        if cmd is None or odom is None:
            continue
        now = cmd["t"] - t0
        scans = [scan_from_record(r) for r in tick["scans"]]
        if scans:
            last_scan_t = now
        age = float("inf") if last_scan_t is None else now - last_scan_t
        tof_r = tick.get("tof")
        tof = (
            None
            if tof_r is None or args.no_tof
            else TofRanges(tof_r.get("front"), tof_r.get("left"), tof_r.get("right"), tof_r["age"])
        )
        sense = Sense(
            now, Pose2D(odom["x"], odom["y"], odom["theta"]), [points(s) for s in scans], age, tof
        )
        d = nav.step(sense)
        what = d.hold or d.veto or ("done" if d.done else "")
        if modes.total() < args.trace:
            pts = sense.scans[-1] if sense.scans else None
            ahead = behind = float("nan")
            if pts is not None and len(pts):
                fwd = pts[(pts[:, 0] > 0) & (np.abs(pts[:, 1]) < 0.35)]
                rear = pts[(pts[:, 0] < 0) & (np.abs(pts[:, 1]) < 0.35)]
                ahead = float(np.hypot(fwd[:, 0], fwd[:, 1]).min()) if len(fwd) else float("nan")
                behind = (
                    float(np.hypot(rear[:, 0], rear[:, 1]).min()) if len(rear) else float("nan")
                )
            pose = d.pose
            facing = nav._follower._facing if nav._follower is not None else None
            print(
                f"t={now:5.1f} pose=({pose.x:+.2f},{pose.y:+.2f},{math.degrees(pose.theta):+4.0f})"
                f" conf={d.confidence:.2f} v={d.twist.linear:+.2f} w={d.twist.angular:+.2f}"
                f" target={d.target} facing={facing} ahead={ahead:.2f} behind={behind:.2f}"
                f" plan={len(nav.plan) if nav.plan else None} | {what}"
            )
        reasons[what.split(":")[0] if what else "free"] += 1
        modes["hold" if d.hold else "drive"] += 1
        if d.twist.linear > 0.02:
            forward += 1
        if d.twist == STOP:
            still += 1
        nav_r = tick.get("nav")
        if nav_r is not None:
            recorded[(nav_r.get("reason") or "free").split(":")[0]] += 1
    n = sum(modes.values())
    print(f"{args.session.name}: {n} ticks, start {start}, goal {goal}")
    print(f"recorded run : {dict(recorded.most_common(6))}")
    print(f"replay now   : {dict(reasons.most_common(6))}")
    print(f"replay twist : forward {forward}/{n} ticks, standing still {still}/{n}")


if __name__ == "__main__":
    main()
