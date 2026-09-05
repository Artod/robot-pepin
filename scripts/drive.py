#!/usr/bin/env python
"""Drive the base from the keyboard while recording odometry and lidar, with a live view.

Keys: W/S change forward speed and straighten out, A/D change turn rate,
space stops, Q quits. The wheels belong to the board's base server
(:mod:`pepin.base_link`): this loop sends the wanted twist twenty times a
second and reads the odometry the board integrated; if the messages stop,
the board's deadman stops the cart. Forward motion needs a lidar scan younger
than a second and a clear box ahead; the ToF reflex stops the cart for
anything low the lidar cannot see. Every session is written to
data/sessions/<timestamp>_<name>.jsonl for offline mapping.

Usage:
    uv run python scripts/drive.py --name lap1        # with the rerun viewer
    uv run python scripts/drive.py --name lap1 --no-viz
    uv run python scripts/drive.py --name lap4 --map data/maps/<map>.npz   # + live localisation
"""

import argparse
import logging
import math
import time
from pathlib import Path

from pepin.base_link import BaseClient
from pepin.geometry import BaseConfig
from pepin.kinematics import Twist
from pepin.lidar import LaserScan, LidarClient, LidarMount
from pepin.localization import Localizer
from pepin.log import setup_logging
from pepin.mapping import OccupancyGrid, transform_to_world
from pepin.odometry import Pose2D
from pepin.recording import SessionRecorder
from pepin.safety import guard_forward
from pepin.teleop import DriveState, KeyReader, apply_key
from pepin.tof import TOF_PORT, TofClient, apply_reflex
from pepin.transport import board_address
from pepin.video import CameraRecorder

logger = logging.getLogger(__name__)

LOOP_HZ = 20
STATUS_EVERY_S = 2.0
SCAN_TIMEOUT_S = 1.0  # no lidar scan for this long => no forward motion
BASE_TIMEOUT_S = 1.0  # no word from the base server for this long => the board has stopped anyway


class Viewer:
    """Live rerun view: robot path, heading, and the latest scan in the odometry frame."""

    def __init__(self, enabled: bool, grid: OccupancyGrid | None = None) -> None:
        self.enabled = enabled
        self._path: list[tuple[float, float]] = []
        if enabled:
            import rerun as rr

            self._rr = rr
            rr.init("pepin", spawn=True)
            if grid is not None:
                rr.log(
                    "world/map",
                    rr.Points2D(
                        grid.occupied_xy(), colors=[90, 90, 90], radii=grid.spec.resolution_m / 2
                    ),
                    static=True,
                )

    def localized(self, pose: Pose2D, confidence: float) -> None:
        """Draw the map-frame pose from the localiser as a blue arrow."""
        if not self.enabled:
            return
        rr = self._rr
        rr.log(
            "world/localized",
            rr.Arrows2D(
                origins=[[pose.x, pose.y]],
                vectors=[[0.3 * math.cos(pose.theta), 0.3 * math.sin(pose.theta)]],
                colors=[[60, 120, 255]] if confidence >= 0.4 else [[180, 180, 180]],
                radii=0.02,
            ),
        )

    def update(self, t: float, pose: Pose2D, scan: LaserScan | None, mount: LidarMount) -> None:
        """Odometry path, heading arrow and the newest scan at time ``t``."""
        if not self.enabled:
            return
        rr = self._rr
        rr.set_time("t", duration=t)
        self._path.append((pose.x, pose.y))
        rr.log("world/path", rr.LineStrips2D([self._path], colors=[80, 160, 255], radii=0.01))
        rr.log(
            "world/robot",
            rr.Arrows2D(
                origins=[[pose.x, pose.y]],
                vectors=[[0.25 * math.cos(pose.theta), 0.25 * math.sin(pose.theta)]],
                colors=[255, 80, 80],
                radii=0.015,
            ),
        )
        if scan is not None:
            world = transform_to_world(scan.points_xy(mount), pose)
            rr.log("world/scan", rr.Points2D(world, colors=[255, 200, 0], radii=0.01))


def guarded_command(
    intent: Twist,
    scan: LaserScan | None,
    scan_age_s: float,
    mount: LidarMount,
    tof: TofClient | None,
) -> tuple[Twist, str]:
    """The twist allowed to reach the wheels, and why it differs from the intent ("" if same)."""
    command = intent
    if command.linear > 0:
        if scan is None or scan_age_s > SCAN_TIMEOUT_S:
            return Twist(0.0, command.angular), "no fresh lidar scan — forward blocked"
        command, blocker = guard_forward(command, scan.points_xy(mount))
        if blocker is not None:
            return command, f"lidar guard: obstacle {blocker:.2f} m ahead, forward blocked"
    if tof is not None:
        decision = apply_reflex(command, tof.ranges())
        if decision.blocked:
            return decision.twist, f"reflex: {decision.reason}"
    return command, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard driving with recording and live view.")
    parser.add_argument("--name", default="drive", help="session name for the recording file")
    parser.add_argument(
        "--video", action="store_true", help="record the overview camera on the board"
    )
    parser.add_argument("--no-viz", action="store_true", help="do not start the rerun viewer")
    parser.add_argument("--map", type=Path, help="saved occupancy grid (.npz) to localise on")
    parser.add_argument(
        "--no-tof", action="store_true", help="skip the ToF stream and its stop reflex"
    )
    parser.add_argument(
        "--init",
        nargs=3,
        type=float,
        metavar=("X", "Y", "THETA_DEG"),
        default=(0.0, 0.0, 0.0),
        help="initial pose on the map (default: the map origin)",
    )
    args = parser.parse_args()
    setup_logging("drive", console=False)  # the terminal is the key-input UI
    print("resolving the board...", end=" ", flush=True)
    host = board_address()
    print(host)
    camera = (
        CameraRecorder(host, f"{time.strftime('%Y%m%d_%H%M%S')}_{args.name}")
        if args.video
        else None
    )
    if camera is not None:
        print("starting the camera recorder (ssh)...", flush=True)
        camera.start()

    BaseConfig.from_json("config/base.json")  # fail early if the config is broken
    mount = LidarMount.from_json("config/lidar.json")
    grid = OccupancyGrid.load(args.map) if args.map else None
    localizer = (
        Localizer(grid, Pose2D(args.init[0], args.init[1], math.radians(args.init[2])))
        if grid is not None
        else None
    )
    print("connecting to the base server...", flush=True)
    link = BaseClient(host).start()
    lidar = LidarClient(host, mount).start()
    tof = None if args.no_tof else TofClient(host, TOF_PORT).start()
    viewer = Viewer(enabled=not args.no_viz, grid=grid)
    try:
        first = link.wait_for_state(timeout_s=5.0)
        if first is None:
            raise SystemExit(
                "no telemetry from the base server — on the board: systemctl status pepin-base"
            )
        print("W/S speed  A/D turn  space stop  Q quit")
        with SessionRecorder("data/sessions", args.name) as rec, KeyReader() as keys:
            rec.note(f"session {args.name} start")
            t0 = time.monotonic()
            next_status = t0
            pose = first.pose
            state = DriveState()
            last_sent: Twist | None = None
            last_reason = ""
            link_warned = 0.0
            while not state.quit:
                tick = time.monotonic()
                key = keys.read()
                if key is not None:
                    state = apply_key(state, key)

                base = link.state(tick)
                if base is None or base.age_s > BASE_TIMEOUT_S:
                    if tick - link_warned > 2.0:
                        logger.warning("no word from the base server; the board has stopped")
                        link_warned = tick
                else:
                    pose = base.pose
                    rec.pose(pose, (base.d_left_m, base.d_right_m))

                latest_scan = lidar.latest
                command, reason = guarded_command(
                    state.twist, latest_scan, lidar.age_s(tick), mount, tof
                )
                if reason and reason != last_reason:
                    logger.warning(reason)
                last_reason = reason
                # Every tick, even unchanged: the board's deadman wants a heartbeat.
                link.set_twist(command)
                if command != last_sent:
                    rec.command(command)
                    last_sent = command
                if tof is not None:
                    r = tof.ranges()
                    rec.write(
                        "tof", {"front": r.front, "left": r.left, "right": r.right, "age": r.age_s}
                    )

                new_scans = lidar.drain()
                if localizer is not None and not new_scans:
                    localizer.predict(pose)
                for scan in new_scans:
                    rec.scan(scan)
                    if localizer is not None:
                        loc = localizer.update(pose, scan.points_xy(mount))
                        rec.write(
                            "loc",
                            {
                                "x": loc.x,
                                "y": loc.y,
                                "theta": loc.theta,
                                "confidence": localizer.confidence,
                            },
                        )
                        viewer.localized(loc, localizer.confidence)
                viewer.update(tick - t0, pose, latest_scan, mount)
                if tick >= next_status:
                    next_status = tick + STATUS_EVERY_S
                    line = (
                        f"v={command.linear:+.2f} w={command.angular:+.2f} | "
                        f"x={pose.x:+.2f} y={pose.y:+.2f} th={math.degrees(pose.theta):+.0f}deg"
                        + (f" | {reason}" if reason else "")
                        + (
                            f" | map x={localizer.pose.x:+.2f} y={localizer.pose.y:+.2f} "
                            f"conf={localizer.confidence:.2f}"
                            if localizer is not None
                            else ""
                        )
                        + (
                            f" | base age={base.age_s * 1000:.0f}ms bus_p95={base.bus_p95_ms:.0f}ms"
                            if base is not None
                            else " | base: no telemetry"
                        )
                    )
                    print(f"\r{line}   ", end="", flush=True)
                    logger.info("tick: %s", line)
                time.sleep(max(0.0, 1.0 / LOOP_HZ - (time.monotonic() - tick)))
            rec.note("session end")
            print(f"\nsession saved: {rec.path} ({rec.records} records)")
            logger.info("session saved: %s (%d records)", rec.path, rec.records)
    finally:
        link.stop()
        link.close()
        lidar.close()
        if tof is not None:
            tof.close()
        if camera is not None:
            camera.stop()


if __name__ == "__main__":
    main()
