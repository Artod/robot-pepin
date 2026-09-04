#!/usr/bin/env python
"""Drive autonomously to a goal on a saved map.

The decisions live in :class:`pepin.navigator.Navigator` (localise, remember
obstacles, plan, follow, guard); this script only wires hardware to it: the
servo bus, wheel odometry, the lidar and ToF readers, the session recorder,
the keyboard and the rerun view. Space pauses, Q quits, at any time.

Motion is refused whenever the robot cannot see (no lidar scan for a second,
lost localiser) and the ToF reflex holds it when its data is stale (drive
without ToF explicitly with --no-tof). A bus outage stops the wheels after 1 s
and aborts the run after 10 s.

Usage:
    uv run python scripts/navigate.py --map data/maps/lap3_loop.npz --goal -2.0 0.5
"""

import argparse
import logging
import math
import time
from pathlib import Path

from pepin.base import BusWatchdog, DiffDriveBase
from pepin.bus import verify_motors
from pepin.feetech import FeetechTcpClient
from pepin.geometry import BaseConfig
from pepin.lidar import LaserScan, LidarClient, LidarMount
from pepin.log import setup_logging
from pepin.mapping import OccupancyGrid, transform_to_world
from pepin.navigator import Decision, Navigator, NavigatorConfig, Sense
from pepin.odometry import DiffDriveOdometry, Pose2D
from pepin.planning import PlannerConfig, path_length
from pepin.recording import SessionRecorder
from pepin.teleop import KEY_BINDINGS, KeyReader
from pepin.tof import TOF_PORT, TofClient
from pepin.transport import SERVO_BUS_PORT, board_address
from pepin.video import CameraRecorder

logger = logging.getLogger(__name__)

LOOP_HZ = 20
STATUS_EVERY_S = 0.5
REPRIME_AFTER_S = 0.5  # a bus outage longer than this may hide half a wheel turn: re-prime


class Viewer:
    """Map, planned path, localised pose and the live scan in the map frame (rerun)."""

    def __init__(self, enabled: bool, grid: OccupancyGrid) -> None:
        self.enabled = enabled
        if not enabled:
            return
        import rerun as rr

        self._rr = rr
        rr.init("pepin-navigate", spawn=True)
        rr.log(
            "world/map",
            rr.Points2D(grid.occupied_xy(), colors=[90, 90, 90], radii=grid.spec.resolution_m / 2),
            static=True,
        )

    def path(self, waypoints: list[tuple[float, float]] | None) -> None:
        """Draw the current plan as a line strip; None clears it."""
        if self.enabled:
            strips = [waypoints] if waypoints else []
            self._rr.log(
                "world/plan", self._rr.LineStrips2D(strips, colors=[0, 200, 80], radii=0.015)
            )

    def robot(
        self, t: float, decision: Decision, scan: LaserScan | None, mount: LidarMount
    ) -> None:
        """Robot arrow, chased waypoint and the newest scan, in the map frame at time ``t``."""
        if not self.enabled:
            return
        rr, pose = self._rr, decision.pose
        rr.set_time("t", duration=t)
        rr.log(
            "world/robot",
            rr.Arrows2D(
                origins=[[pose.x, pose.y]],
                vectors=[[0.3 * math.cos(pose.theta), 0.3 * math.sin(pose.theta)]],
                colors=[60, 120, 255],
                radii=0.02,
            ),
        )
        if decision.target is not None:
            rr.log("world/target", rr.Points2D([decision.target], colors=[255, 120, 0], radii=0.05))
        if scan is not None:
            world = transform_to_world(scan.points_xy(mount), pose)
            rr.log("world/scan", rr.Points2D(world, colors=[255, 200, 0], radii=0.01))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous point-to-point navigation on a saved map."
    )
    parser.add_argument("--map", type=Path, required=True, help="occupancy grid .npz")
    parser.add_argument("--goal", nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument(
        "--init",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "THETA_DEG"),
        help="starting pose on the map (default: the map origin)",
    )
    parser.add_argument("--name", default="nav", help="session name")
    parser.add_argument(
        "--robot-radius",
        type=float,
        default=PlannerConfig().robot_radius_m,
        help="inflation radius, m (default: the planner's)",
    )
    parser.add_argument(
        "--occupied", type=float, default=0.55, help="occupancy probability treated as an obstacle"
    )
    parser.add_argument("--no-viz", action="store_true", help="no rerun window")
    parser.add_argument("--no-tof", action="store_true", help="drive without the ToF reflex")
    parser.add_argument("--video", action="store_true", help="record the overview camera")
    return parser.parse_args()


def read_travel(
    base: DiffDriveBase, watchdog: BusWatchdog, now: float
) -> tuple[float, float] | None:
    """Wheel travel since the last tick, or None when the bus did not answer (policy applied)."""
    try:
        travel = base.read_wheel_travel()
    except TimeoutError as exc:
        watchdog.handle(base, now, exc)
        return None
    down = watchdog.recovered(now)
    if down is not None:
        logger.info("bus recovered after %.1f s (%d ticks lost)", down, watchdog.failures)
        if down > REPRIME_AFTER_S:
            base.reprime()  # half a wheel turn may have gone unseen: drop it, do not alias it
            return (0.0, 0.0)
    return travel


def record(rec: SessionRecorder, sense: Sense, decision: Decision) -> None:
    """Session topics for offline replay: localised pose, ToF ranges, the command."""
    rec.write(
        "loc",
        {
            "x": decision.pose.x,
            "y": decision.pose.y,
            "theta": decision.pose.theta,
            "confidence": decision.confidence,
        },
    )
    if sense.tof is not None:
        r = sense.tof
        rec.write("tof", {"front": r.front, "left": r.left, "right": r.right, "age": r.age_s})
    rec.command(decision.twist)


def status_line(decision: Decision, sense: Sense) -> str:
    """One-line state for the terminal."""
    pose = decision.pose
    what = (
        f"HOLD: {decision.hold}"
        if decision.hold
        else f"cmd v={decision.twist.linear:+.2f} w={decision.twist.angular:+.2f}"
        + (f" ({decision.veto})" if decision.veto else "")
    )
    tof = (
        f"tof f={sense.tof.front} l={sense.tof.left} r={sense.tof.right} age={sense.tof.age_s:.1f}s"
        if sense.tof is not None
        else "tof off"
    )
    return (
        f"pose x={pose.x:+.2f} y={pose.y:+.2f} th={math.degrees(pose.theta):+.0f} "
        f"conf={decision.confidence:.2f} | {what} | {tof}"
    )


def run(
    nav: Navigator,
    base: DiffDriveBase,
    odom: DiffDriveOdometry,
    lidar: LidarClient,
    tof: TofClient | None,
    mount: LidarMount,
    rec: SessionRecorder,
    keys: KeyReader,
    viewer: Viewer,
) -> None:
    """The 20 Hz loop: sense, decide, act, show — until Q or the goal."""
    watchdog = BusWatchdog()
    t0 = time.monotonic()
    next_status = t0
    last_hold = ""
    while True:
        tick = time.monotonic()
        action = KEY_BINDINGS.get(keys.read() or "")
        if action == "quit":
            break
        if action == "stop":
            nav.paused = not nav.paused
            logger.info("paused" if nav.paused else "resumed")

        travel = read_travel(base, watchdog, tick)
        if travel is None:
            time.sleep(1.0 / LOOP_HZ)
            continue
        odom_pose = odom.update(*travel)
        rec.pose(odom_pose, travel)
        new_scans = lidar.drain()
        for scan in new_scans:
            rec.scan(scan)
        sense = Sense(
            now=tick,
            odom_pose=odom_pose,
            scans=[scan.points_xy(mount) for scan in new_scans],
            scan_age_s=lidar.age_s(tick),
            tof=tof.ranges() if tof is not None else None,
        )
        decision = nav.step(sense)
        record(rec, sense, decision)
        if decision.hold != last_hold:
            last_hold = decision.hold
            if decision.hold:
                logger.warning("holding still: %s", decision.hold)
        if decision.veto:
            logger.info("guard: %s", decision.veto)
        if decision.plan_changed:
            viewer.path(nav.plan)
            if nav.plan is not None:
                logger.info("replanned: %d waypoints, %.2f m", len(nav.plan), path_length(nav.plan))

        try:
            base.set_twist(decision.twist)
        except TimeoutError as exc:
            watchdog.handle(base, tick, exc)
            time.sleep(1.0 / LOOP_HZ)
            continue
        viewer.robot(tick - t0, decision, lidar.latest, mount)
        if tick >= next_status:
            next_status = tick + STATUS_EVERY_S
            print(f"\r{status_line(decision, sense)}   ", end="", flush=True)
        if decision.done:
            base.stop()
            logger.info("goal reached: pose %s", decision.pose)
            print(f"\ngoal reached at x={decision.pose.x:+.2f} y={decision.pose.y:+.2f}")
            break
        time.sleep(max(0.0, 1.0 / LOOP_HZ - (time.monotonic() - tick)))


def main() -> None:
    args = parse_args()
    setup_logging("navigate", console=False)
    host = board_address()
    logger.info("board at %s", host)

    base_cfg = BaseConfig.from_json("config/base.json")
    mount = LidarMount.from_json("config/lidar.json")
    grid = OccupancyGrid.load(args.map)
    goal = (args.goal[0], args.goal[1])
    start = Pose2D(args.init[0], args.init[1], math.radians(args.init[2]))
    nav = Navigator(
        grid,
        start,
        goal,
        NavigatorConfig(
            planner=PlannerConfig(
                occupied_threshold=args.occupied, robot_radius_m=args.robot_radius
            )
        ),
    )
    if nav.plan is None:
        raise SystemExit(
            f"no path from {args.init[:2]} to {goal} — is the start clear of walls and the goal "
            "inside known free space?"
        )
    logger.info("plan: %d waypoints, %.2f m", len(nav.plan), path_length(nav.plan))

    camera = (
        CameraRecorder(host, f"{time.strftime('%Y%m%d_%H%M%S')}_{args.name}")
        if args.video
        else None
    )
    if camera is not None:
        camera.start()
    lidar = LidarClient(host, mount).start()
    tof = None if args.no_tof else TofClient(host, TOF_PORT).start()
    viewer = Viewer(enabled=not args.no_viz, grid=grid)
    viewer.path(nav.plan)

    print(f"navigating to {goal}; space = pause, Q = quit")
    motors = DiffDriveBase.motor_ids(base_cfg)
    try:
        with (
            FeetechTcpClient(host, SERVO_BUS_PORT, motors) as bus,
            SessionRecorder("data/sessions", args.name) as rec,
            KeyReader() as keys,
        ):
            verify_motors(bus, list(motors))
            odom = DiffDriveOdometry(base_cfg.geometry)
            rec.note(f"navigate to {goal} from {args.init}")
            with DiffDriveBase(bus, base_cfg) as base:
                base.read_wheel_travel()  # prime encoders
                run(nav, base, odom, lidar, tof, mount, rec, keys, viewer)
            print(f"\nsession saved: {rec.path}")
    finally:
        # On Q, on the goal and on any exception: readers and the camera must not outlive the run.
        lidar.close()
        if tof is not None:
            tof.close()
        if camera is not None:
            camera.stop()


if __name__ == "__main__":
    main()
