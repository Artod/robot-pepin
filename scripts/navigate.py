#!/usr/bin/env python
"""Drive autonomously to a goal on a saved map: localise, plan, follow, stop when close.

Place the robot at the map origin (or pass --init), give a goal in map
coordinates, and the robot plans an A* path around the walls, follows it
while re-localising on every scan, and refuses to drive into anything the
ToF sensors see up close. Space stops, Q quits, at any time.

Usage:
    uv run python scripts/navigate.py --map data/maps/lap3_loop.npz --goal -2.0 0.5
"""

import argparse
import logging
import math
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from pepin.base import DiffDriveBase
from pepin.bus import verify_motors
from pepin.control import PathFollower
from pepin.feetech import FeetechTcpClient
from pepin.geometry import BaseConfig
from pepin.kinematics import Twist
from pepin.lidar import LaserScan, LidarMount, LidarStream, TcpSource
from pepin.localization import Localizer
from pepin.log import setup_logging
from pepin.mapping import OccupancyGrid, transform_to_world
from pepin.odometry import DiffDriveOdometry, Pose2D
from pepin.planning import GridPlanner, PlannerConfig, path_length
from pepin.recording import SessionRecorder
from pepin.safety import guard_forward
from pepin.teleop import KeyReader
from pepin.tof import TOF_PORT, TofClient, apply_reflex
from pepin.transport import LIDAR_PORT, SERVO_BUS_PORT, board_address
from pepin.video import CameraRecorder

logger = logging.getLogger(__name__)

LOOP_HZ = 20
REPLAN_EVERY_S = 3.0
SCAN_TIMEOUT_S = 1.0  # no lidar for this long => the robot must not move
OBSTACLE_MEMORY_SCANS = 10  # ~1 s of lidar: a person who just moved away frees the path


def lidar_thread(
    host: str, mount: LidarMount, scans: "queue.Queue[LaserScan]", stop: threading.Event
) -> None:
    """Feed scans into the queue; reconnect with a warning if the bridge drops (board booting)."""
    while not stop.is_set():
        try:
            stream = LidarStream(TcpSource(host, LIDAR_PORT), mount)
        except OSError as exc:
            logger.warning("lidar bridge unreachable (%s); retrying", exc)
            stop.wait(2.0)
            continue
        try:
            for scan in stream.scans():
                scans.put(scan)
                if stop.is_set():
                    break
        except ConnectionError as exc:
            logger.warning("lidar stream lost (%s); reconnecting", exc)
            stop.wait(2.0)
        finally:
            stream.close()


class Viewer:
    """Map, planned path, localised pose and the live scan in the map frame."""

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

    def path(self, waypoints: list[tuple[float, float]]) -> None:
        if self.enabled:
            self._rr.log(
                "world/plan", self._rr.LineStrips2D([waypoints], colors=[0, 200, 80], radii=0.015)
            )

    def robot(
        self,
        t: float,
        pose: Pose2D,
        scan: LaserScan | None,
        mount: LidarMount,
        target: tuple[float, float] | None,
    ) -> None:
        if not self.enabled:
            return
        rr = self._rr
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
        if target is not None:
            rr.log("world/target", rr.Points2D([target], colors=[255, 120, 0], radii=0.05))
        if scan is not None:
            import numpy as np

            local = scan.points_xy(mount)
            c, s = math.cos(pose.theta), math.sin(pose.theta)
            world = local @ np.array([[c, s], [-s, c]]) + np.array([pose.x, pose.y])
            rr.log("world/scan", rr.Points2D(world, colors=[255, 200, 0], radii=0.01))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous point-to-point navigation on a saved map."
    )
    parser.add_argument("--map", type=Path, required=True, help="occupancy grid .npz")
    parser.add_argument("--goal", nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument(
        "--init", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "THETA_DEG")
    )
    parser.add_argument("--name", default="navigate")
    parser.add_argument(
        "--video", action="store_true", help="record the overview camera on the board"
    )
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument(
        "--no-tof", action="store_true", help="skip the ToF stream and its stop reflex"
    )
    parser.add_argument(
        "--robot-radius", type=float, default=0.30, help="inflation radius for planning, m"
    )
    parser.add_argument(
        "--occupied", type=float, default=0.55, help="occupancy probability treated as an obstacle"
    )
    args = parser.parse_args()
    setup_logging("navigate", console=False)
    host = board_address()
    logger.info("board at %s", host)
    camera = (
        CameraRecorder(host, f"{time.strftime('%Y%m%d_%H%M%S')}_{args.name}")
        if args.video
        else None
    )
    if camera is not None:
        camera.start()

    base_cfg = BaseConfig.from_json("config/base.json")
    mount = LidarMount.from_json("config/lidar.json")
    grid = OccupancyGrid.load(args.map)
    planner = GridPlanner(
        grid, PlannerConfig(occupied_threshold=args.occupied, robot_radius_m=args.robot_radius)
    )
    goal = (args.goal[0], args.goal[1])
    localizer = Localizer(grid, Pose2D(args.init[0], args.init[1], math.radians(args.init[2])))

    plan = planner.plan((localizer.pose.x, localizer.pose.y), goal)
    if plan is None:
        raise SystemExit(f"no path from {args.init[:2]} to {goal} — is the goal inside free space?")
    logger.info("plan: %d waypoints, %.2f m", len(plan), path_length(plan))
    follower = PathFollower(plan)

    scans: queue.Queue[LaserScan] = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=lidar_thread, args=(host, mount, scans, stop), daemon=True).start()
    viewer = Viewer(enabled=not args.no_viz, grid=grid)
    viewer.path(plan)
    tof = None if args.no_tof else TofClient(host, TOF_PORT).start()

    print(f"navigating to {goal}; space = stop, Q = quit")
    motors = DiffDriveBase.motor_ids(base_cfg)
    with (
        FeetechTcpClient(host, SERVO_BUS_PORT, motors) as bus,
        SessionRecorder("data/sessions", args.name) as rec,
        KeyReader() as keys,
    ):
        verify_motors(bus, list(motors))
        odom = DiffDriveOdometry(base_cfg.geometry)
        rec.note(f"navigate to {goal} from {args.init}")
        with DiffDriveBase(bus, base_cfg) as base:
            base.read_wheel_travel()
            t0 = time.monotonic()
            paused = False
            latest_scan: LaserScan | None = None
            last_scan_at = 0.0
            blind_warned = 0.0
            last_replan = t0
            recent_hits: deque = deque(maxlen=OBSTACLE_MEMORY_SCANS)
            blocked_since: float | None = None
            while True:
                tick = time.monotonic()
                key = keys.read()
                if key in ("q", "й"):
                    break
                if key == " ":
                    paused = not paused
                    base.stop()
                    logger.info("paused" if paused else "resumed")
                try:
                    travel = base.read_wheel_travel()
                except TimeoutError as exc:
                    logger.warning("bus timeout: %s", exc)
                    time.sleep(0.05)
                    continue
                odom_pose = odom.update(*travel)
                rec.pose(odom_pose, travel)
                if scans.empty():
                    localizer.predict(odom_pose)
                while not scans.empty():
                    latest_scan = scans.get_nowait()
                    last_scan_at = tick
                    rec.scan(latest_scan)
                    loc = localizer.update(odom_pose, latest_scan.points_xy(mount))
                    rec.write(
                        "loc",
                        {
                            "x": loc.x,
                            "y": loc.y,
                            "theta": loc.theta,
                            "confidence": localizer.confidence,
                        },
                    )
                pose = localizer.pose

                if tick - last_replan > REPLAN_EVERY_S and not paused:
                    fresh = planner.plan((pose.x, pose.y), goal)
                    if fresh is not None:
                        follower = PathFollower(fresh)
                        viewer.path(fresh)
                    last_replan = tick

                out = follower.step(pose)
                command = Twist(0.0, 0.0) if paused else out.twist
                if tick - last_scan_at > SCAN_TIMEOUT_S:
                    command = Twist(0.0, 0.0)  # blind: no lidar means no motion, ever
                    if tick - blind_warned > 2.0:
                        logger.warning(
                            "no lidar scans for %.1f s — holding still", tick - last_scan_at
                        )
                        print(
                            "\rNO LIDAR DATA — holding still (check the board)      ",
                            end="",
                            flush=True,
                        )
                        blind_warned = tick
                if latest_scan is not None:
                    command, blocker = guard_forward(command, latest_scan.points_xy(mount))
                    if blocker is not None:
                        logger.warning(
                            "lidar guard: obstacle %.2f m ahead, forward blocked", blocker
                        )
                if tof is not None:
                    ranges = tof.ranges()
                    rec.write(
                        "tof", {"front": ranges.front, "left": ranges.left, "right": ranges.right}
                    )
                    decision = apply_reflex(command, ranges)
                    if decision.blocked:
                        logger.warning("reflex: %s", decision.reason)
                        if blocked_since is None:
                            blocked_since = tick
                    command = decision.twist
                    if ranges.front is not None and ranges.front < 1.0:
                        recent_hits.append(
                            transform_to_world(np.array([[ranges.front, 0.0]]), pose)
                        )
                base.set_twist(command)
                rec.command(command)
                viewer.robot(tick - t0, pose, latest_scan, mount, out.target)
                if int((tick - t0) * 2) != int((tick - t0 - 1.0 / LOOP_HZ) * 2):
                    r = tof.ranges() if tof is not None else None
                    print(
                        f"\rpose x={pose.x:+.2f} y={pose.y:+.2f} "
                        f"th={math.degrees(pose.theta):+.0f} "
                        f"conf={localizer.confidence:.2f} | "
                        f"cmd v={command.linear:+.2f} w={command.angular:+.2f} | "
                        + (
                            f"tof f={r.front} l={r.left} r={r.right} age={r.age_s:.1f}s"
                            if r
                            else "tof off"
                        )
                        + "   ",
                        end="",
                        flush=True,
                    )
                if out.done:
                    base.stop()
                    logger.info("goal reached: pose %s", pose)
                    print(f"\ngoal reached at x={pose.x:+.2f} y={pose.y:+.2f}")
                    break
                time.sleep(max(0.0, 1.0 / LOOP_HZ - (time.monotonic() - tick)))
    stop.set()
    if tof is not None:
        tof.close()
    if camera is not None:
        camera.stop()
    print(f"session saved: {rec.path}")


if __name__ == "__main__":
    main()
