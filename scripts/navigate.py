#!/usr/bin/env python
"""Drive autonomously to a place or a point on a saved map.

The verbs live in :class:`pepin.driver.Driver` (goto, pause, resume, cancel,
one Status per tick), the decisions in :class:`pepin.navigator.Navigator`, the
hardware in :class:`pepin.robot.Robot`. This script is the terminal front end:
arguments, the keyboard (space pauses and resumes, Q quits, Ctrl-C stops
cleanly), the status line, the session recorder and the rerun view.

Motion is refused whenever the robot cannot see (no lidar scan for a second,
lost localiser, no word from the base for a second) and the ToF reflex holds
it when its data is stale (drive without ToF explicitly with --no-tof).

Usage:
    uv run python scripts/navigate.py --map data/maps/lap3_loop.npz --goal kitchen
    uv run python scripts/navigate.py --map data/maps/lap3_loop.npz --goal -2.0 0.5
"""

import argparse
import logging
import math
import time
from pathlib import Path

from pepin.driver import Driver, Mode, Status
from pepin.lidar import LaserScan, LidarMount
from pepin.log import setup_logging
from pepin.mapping import OccupancyGrid, transform_to_world
from pepin.navigator import NavigatorConfig
from pepin.odometry import Pose2D
from pepin.places import load_places, resolve_goal
from pepin.planning import PlannerConfig
from pepin.recording import SessionRecorder
from pepin.robot import Robot, RobotConfig
from pepin.teleop import KEY_BINDINGS, KeyReader
from pepin.transport import board_address

logger = logging.getLogger(__name__)

LOOP_HZ = 20
STATUS_EVERY_S = 0.5


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

    def robot(self, t: float, status: Status, scan: LaserScan | None, mount: LidarMount) -> None:
        """Robot arrow, chased waypoint and the newest scan, in the map frame at time ``t``."""
        if not self.enabled:
            return
        rr, pose = self._rr, status.pose
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
        if status.target is not None:
            rr.log("world/target", rr.Points2D([status.target], colors=[255, 120, 0], radii=0.05))
        if scan is not None:
            world = transform_to_world(scan.points_xy(mount), pose)
            rr.log("world/scan", rr.Points2D(world, colors=[255, 200, 0], radii=0.01))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous point-to-point navigation on a saved map."
    )
    parser.add_argument("--map", type=Path, required=True, help="occupancy grid .npz")
    parser.add_argument(
        "--goal",
        nargs="+",
        required=True,
        metavar="X Y | NAME",
        help="map coordinates, or a place named with scripts/places.py",
    )
    parser.add_argument(
        "--init",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "THETA_DEG"),
        help="approximate starting pose on the map; the first scan refines it",
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
    parser.add_argument("--no-tof", action="store_true", help="drive without the ToF sensors")
    parser.add_argument("--video", action="store_true", help="record the overview camera")
    return parser.parse_args()


def run(driver: Driver, robot: Robot, keys: KeyReader, viewer: Viewer) -> Status | None:
    """The 20 Hz loop: keys, one tick, the view, the status line; until Q or arrival."""
    t0 = time.monotonic()
    next_status = t0
    last_reason = ""
    status: Status | None = None
    while True:
        tick = time.monotonic()
        action = KEY_BINDINGS.get(keys.read() or "")
        if action == "quit":
            driver.cancel()
            return status
        if action == "stop":
            if driver.mode is Mode.PAUSED:
                driver.resume()
            else:
                driver.pause()
            logger.info("%s", driver.mode)
        status = driver.tick(tick)
        if status.reason != last_reason:
            last_reason = status.reason
            if status.reason:
                logger.warning("%s: %s", status.mode, status.reason)
        viewer.robot(
            tick - t0, status, robot.lidar.latest if robot.lidar is not None else None, robot.mount
        )
        if tick >= next_status:
            next_status = tick + STATUS_EVERY_S
            line = status.summary()
            print(f"\r{line}   ", end="", flush=True)
            logger.info("tick: %s", line)
        if status.mode is Mode.ARRIVED:
            return status
        time.sleep(max(0.0, 1.0 / LOOP_HZ - (time.monotonic() - tick)))


def main() -> None:
    args = parse_args()
    setup_logging("navigate", console=False)
    config = RobotConfig.load()
    if args.no_tof:
        config = config.without("tof")
    grid = OccupancyGrid.load(args.map)
    try:
        goal, place = resolve_goal(args.goal, args.map)  # a typo must fail before we connect
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    destination: str | tuple[float, float] = place.name if place is not None else goal
    start = Pose2D(args.init[0], args.init[1], math.radians(args.init[2]))
    nav_config = NavigatorConfig(
        tof_mounts=config.tof_mounts if config.enabled("tof") else {},
        footprint=config.footprint,
        planner=PlannerConfig(occupied_threshold=args.occupied, robot_radius_m=args.robot_radius),
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
        servos = robot.link.ping()
        if servos is not None and not all(servos.values()):
            silent = sorted(k for k, ok in servos.items() if not ok)
            logger.warning("servos not answering on the board: %s", silent)
        logger.info("base server up: bus_p95 %.0f ms, armed=%s", first.bus_p95_ms, first.armed)
        viewer = Viewer(enabled=not args.no_viz, grid=grid)
        try:
            with SessionRecorder("data/sessions", args.name) as rec, KeyReader() as keys:
                driver = Driver(
                    robot,
                    grid,
                    start,
                    config=nav_config,
                    places=load_places(args.map),
                    recorder=rec,
                    on_plan=viewer.path,
                )
                driver.goto(destination)
                rec.note(f"navigate to {destination} {goal} from {args.init}")
                print(f"navigating to {destination} {goal}; space = pause, Q = quit, Ctrl-C = stop")
                last = run(driver, robot, keys, viewer)
                if last is not None and last.mode is Mode.ARRIVED:
                    print(f"\ngoal reached at x={last.pose.x:+.2f} y={last.pose.y:+.2f}")
                print(f"session saved: {rec.path}")
        except KeyboardInterrupt:
            logger.info("interrupted by the user")
            print("\ninterrupted — stopping the wheels")


if __name__ == "__main__":
    main()
