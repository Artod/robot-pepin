#!/usr/bin/env python
"""Drive autonomously to a goal on a saved map.

The decisions live in :class:`pepin.navigator.Navigator` (localise, remember
obstacles, plan, follow, guard); the hardware lives in :class:`pepin.robot.Robot`
(the base link to the board that owns the wheels, the lidar and ToF feeds, the
camera). This script wires the two together with the session recorder, the
keyboard and the rerun view. Space pauses, Q quits, Ctrl-C stops cleanly.

Motion is refused whenever the robot cannot see (no lidar scan for a second,
lost localiser, no word from the base for a second) and the ToF reflex holds
it when its data is stale (drive without ToF explicitly with --no-tof).

Usage:
    uv run python scripts/navigate.py --map data/maps/lap3_loop.npz --goal -2.0 0.5
"""

import argparse
import logging
import math
import time
from pathlib import Path

from pepin.lidar import LaserScan, LidarMount
from pepin.log import setup_logging
from pepin.mapping import OccupancyGrid, transform_to_world
from pepin.navigator import Decision, Navigator, NavigatorConfig
from pepin.odometry import Pose2D
from pepin.planning import PlannerConfig, path_length
from pepin.recording import SessionRecorder
from pepin.robot import Observation, Robot, RobotConfig
from pepin.teleop import KEY_BINDINGS, KeyReader
from pepin.transport import board_address

logger = logging.getLogger(__name__)

LOOP_HZ = 20
STATUS_EVERY_S = 0.5
SLOW_TICK_S = 0.25  # a tick slower than this is logged with a per-stage breakdown


class StageTimer:
    """Per-tick stopwatch: ``mark`` closes a stage, ``breakdown`` names where a slow tick went."""

    def __init__(self) -> None:
        self._t = time.perf_counter()
        self.stages: dict[str, float] = {}

    def mark(self, name: str) -> None:
        """Record the time since the previous mark under ``name``."""
        now = time.perf_counter()
        self.stages[name] = now - self._t
        self._t = now

    def breakdown(self) -> str:
        """Stages and their durations in milliseconds, in order."""
        return ", ".join(f"{k} {v * 1000:.0f} ms" for k, v in self.stages.items())


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


def record(rec: SessionRecorder, obs: Observation, decision: Decision) -> None:
    """Session topics for offline replay: odometry, scans, localised pose, ToF, the command."""
    rec.pose(obs.state.pose, (obs.state.d_left_m, obs.state.d_right_m))
    for scan in obs.scans:
        rec.scan(scan)
    rec.write(
        "loc",
        {
            "x": decision.pose.x,
            "y": decision.pose.y,
            "theta": decision.pose.theta,
            "confidence": decision.confidence,
        },
    )
    if obs.sense.tof is not None:
        r = obs.sense.tof
        rec.write("tof", {"front": r.front, "left": r.left, "right": r.right, "age": r.age_s})
    rec.command(decision.twist)


def status_line(decision: Decision, obs: Observation) -> str:
    """One-line state for the terminal, with the base link's health at the end."""
    pose, tof, state = decision.pose, obs.sense.tof, obs.state
    what = (
        f"HOLD: {decision.hold}"
        if decision.hold
        else f"cmd v={decision.twist.linear:+.2f} w={decision.twist.angular:+.2f}"
        + (f" ({decision.veto})" if decision.veto else "")
    )
    tof_text = (
        f"tof f={tof.front} l={tof.left} r={tof.right} age={tof.age_s:.1f}s"
        if tof is not None
        else "tof off"
    )
    base = (
        f"base age={state.age_s * 1000:.0f}ms bus_p95={state.bus_p95_ms:.0f}ms"
        + (" DEADMAN" if state.deadman else "")
        + ("" if state.bus_ok else " BUS-DOWN")
    )
    return (
        f"pose x={pose.x:+.2f} y={pose.y:+.2f} th={math.degrees(pose.theta):+.0f} "
        f"conf={decision.confidence:.2f} | {what} | {tof_text} | {base}"
    )


def run(
    nav: Navigator, robot: Robot, rec: SessionRecorder, keys: KeyReader, viewer: Viewer
) -> None:
    """The 20 Hz loop: observe, decide, act, show, until Q or the goal; never waits on the net."""
    t0 = time.monotonic()
    next_status = t0
    last_hold = ""
    base_warned = 0.0
    while True:
        tick = time.monotonic()
        timer = StageTimer()
        action = KEY_BINDINGS.get(keys.read() or "")
        if action == "quit":
            break
        if action == "stop":
            nav.paused = not nav.paused
            logger.info("paused" if nav.paused else "resumed")

        obs = robot.observe(tick)
        if obs is None:
            # The board's deadman has already stopped the wheels; we just stop pretending to drive.
            if tick - base_warned > 2.0:
                logger.warning("no word from the base server")
                print("\rHOLD: no telemetry from the base server        ", end="", flush=True)
                base_warned = tick
            time.sleep(1.0 / LOOP_HZ)
            continue
        timer.mark("sensors")
        decision = nav.step(obs.sense)
        timer.mark("navigator")
        record(rec, obs, decision)
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

        robot.drive(decision.twist)  # fire-and-forget; the board applies it within its tick
        viewer.robot(
            tick - t0,
            decision,
            robot.lidar.latest if robot.lidar is not None else None,
            robot.mount,
        )
        timer.mark("act+viewer")
        total = time.monotonic() - tick
        if total > SLOW_TICK_S:
            logger.warning("slow tick %.2f s: %s", total, timer.breakdown())
        if tick >= next_status:
            next_status = tick + STATUS_EVERY_S
            line = status_line(decision, obs)
            print(f"\r{line}   ", end="", flush=True)
            logger.info("tick: %s", line)
        if decision.done:
            robot.stop()
            logger.info("goal reached: pose %s", decision.pose)
            print(f"\ngoal reached at x={decision.pose.x:+.2f} y={decision.pose.y:+.2f}")
            break
        time.sleep(max(0.0, 1.0 / LOOP_HZ - (time.monotonic() - tick)))


def main() -> None:
    args = parse_args()
    setup_logging("navigate", console=False)
    config = RobotConfig.load()
    if args.no_tof:
        config = config.without("tof")
    grid = OccupancyGrid.load(args.map)
    goal = (args.goal[0], args.goal[1])
    start = Pose2D(args.init[0], args.init[1], math.radians(args.init[2]))
    nav = Navigator(
        grid,
        start,
        goal,
        NavigatorConfig(
            tof_mounts=config.tof_mounts if config.enabled("tof") else {},
            planner=PlannerConfig(
                occupied_threshold=args.occupied, robot_radius_m=args.robot_radius
            ),
        ),
    )
    if nav.plan is None:
        raise SystemExit(
            f"no path from {args.init[:2]} to {goal} — is the start clear of walls and the goal "
            "inside known free space?"
        )
    logger.info("plan: %d waypoints, %.2f m", len(nav.plan), path_length(nav.plan))

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
        viewer.path(nav.plan)
        print(f"navigating to {goal}; space = pause, Q = quit, Ctrl-C = stop")
        try:
            with SessionRecorder("data/sessions", args.name) as rec, KeyReader() as keys:
                rec.note(f"navigate to {goal} from {args.init}")
                run(nav, robot, rec, keys, viewer)
                print(f"\nsession saved: {rec.path}")
        except KeyboardInterrupt:
            logger.info("interrupted by the user")
            print("\ninterrupted — stopping the wheels")


if __name__ == "__main__":
    main()
