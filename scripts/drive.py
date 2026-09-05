#!/usr/bin/env python
"""Drive the base from the keyboard while recording odometry and lidar, with a live view.

Keys: W/S change forward speed and straighten out, A/D change turn rate,
space stops, Q quits, Ctrl-C stops cleanly. The hardware is one
:class:`pepin.robot.Robot`: the board owns the wheels (its deadman stops the
cart if our messages stop), this loop sends the wanted twist twenty times a
second and reads what the feeds saw. Forward motion needs a lidar scan younger
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

from pepin.kinematics import Twist
from pepin.lidar import LaserScan, LidarMount
from pepin.localization import Localizer
from pepin.log import setup_logging
from pepin.mapping import OccupancyGrid, transform_to_world
from pepin.odometry import Pose2D
from pepin.recording import SessionRecorder
from pepin.robot import Observation, Robot, RobotConfig
from pepin.safety import guard_forward
from pepin.teleop import DriveState, KeyReader, apply_key
from pepin.tof import apply_reflex
from pepin.transport import board_address

logger = logging.getLogger(__name__)

LOOP_HZ = 20
STATUS_EVERY_S = 2.0
SCAN_TIMEOUT_S = 1.0  # no lidar scan for this long => no forward motion


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


def guarded_command(intent: Twist, obs: Observation, mount: LidarMount) -> tuple[Twist, str]:
    """The twist allowed to reach the wheels, and why it differs from the intent ("" if same)."""
    command = intent
    sense = obs.sense
    if command.linear > 0:
        if not sense.scans and sense.scan_age_s > SCAN_TIMEOUT_S:
            return Twist(0.0, command.angular), "no fresh lidar scan — forward blocked"
        points = sense.scans[-1] if sense.scans else None
        if points is not None:
            command, blocker = guard_forward(command, points)
            if blocker is not None:
                return command, f"lidar guard: obstacle {blocker:.2f} m ahead, forward blocked"
    if sense.tof is not None:
        decision = apply_reflex(command, sense.tof)
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
    config = RobotConfig.load()
    if args.no_tof:
        config = config.without("tof")
    grid = OccupancyGrid.load(args.map) if args.map else None
    localizer = (
        Localizer(grid, Pose2D(args.init[0], args.init[1], math.radians(args.init[2])))
        if grid is not None
        else None
    )
    print("resolving the board...", end=" ", flush=True)
    host = board_address()
    print(host)
    video_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{args.name}" if args.video else None
    print("connecting to the robot (base link, lidar, tof, camera)...", flush=True)
    with Robot.connect(config, host=host, video_name=video_name) as robot:
        try:
            first = robot.wait_ready()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        viewer = Viewer(enabled=not args.no_viz, grid=grid)
        print("W/S speed  A/D turn  space stop  Q quit  Ctrl-C stop")
        try:
            with SessionRecorder("data/sessions", args.name) as rec, KeyReader() as keys:
                rec.note(f"session {args.name} start")
                t0 = time.monotonic()
                next_status = t0
                pose = first.pose
                state = DriveState()
                last_sent: Twist | None = None
                last_reason = ""
                link_warned = 0.0
                latest_scan: LaserScan | None = None
                while not state.quit:
                    tick = time.monotonic()
                    key = keys.read()
                    if key is not None:
                        state = apply_key(state, key)
                    obs = robot.observe(tick)
                    if obs is None:
                        if tick - link_warned > 2.0:
                            logger.warning("no word from the base server; the board has stopped")
                            link_warned = tick
                        time.sleep(1.0 / LOOP_HZ)
                        continue
                    pose = obs.state.pose
                    rec.pose(pose, (obs.state.d_left_m, obs.state.d_right_m))
                    if obs.scans:
                        latest_scan = obs.scans[-1]
                    command, reason = guarded_command(state.twist, obs, robot.mount)
                    if reason and reason != last_reason:
                        logger.warning(reason)
                    last_reason = reason
                    robot.drive(command)  # every tick, even unchanged: the deadman heartbeat
                    if command != last_sent:
                        rec.command(command)
                        last_sent = command
                    if obs.sense.tof is not None:
                        r = obs.sense.tof
                        rec.write(
                            "tof",
                            {"front": r.front, "left": r.left, "right": r.right, "age": r.age_s},
                        )
                    if localizer is not None and not obs.scans:
                        localizer.predict(pose)
                    for scan in obs.scans:
                        rec.scan(scan)
                        if localizer is not None:
                            loc = localizer.update(pose, scan.points_xy(robot.mount))
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
                    viewer.update(tick - t0, pose, latest_scan, robot.mount)
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
                            + f" | base age={obs.state.age_s * 1000:.0f}ms "
                            f"bus_p95={obs.state.bus_p95_ms:.0f}ms"
                        )
                        print(f"\r{line}   ", end="", flush=True)
                        logger.info("tick: %s", line)
                    time.sleep(max(0.0, 1.0 / LOOP_HZ - (time.monotonic() - tick)))
                rec.note("session end")
                print(f"\nsession saved: {rec.path} ({rec.records} records)")
                logger.info("session saved: %s (%d records)", rec.path, rec.records)
        except KeyboardInterrupt:
            logger.info("interrupted by the user")
            print("\ninterrupted — stopping the wheels")


if __name__ == "__main__":
    main()
