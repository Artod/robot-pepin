#!/usr/bin/env python
"""Drive the base from the keyboard while recording odometry and lidar, with a live view.

Keys: W/S change forward speed and straighten out, A/D change turn rate,
space stops, Q quits. The control loop runs at 20 Hz; lidar scans arrive from
a background thread and are recorded and drawn in the robot's odometry frame.
Every session is written to data/sessions/<timestamp>_<name>.jsonl for offline
mapping. Bus timeouts (wifi hiccups) only cost the affected ticks; a long
outage stops the wheels and then aborts the session.

Usage:
    uv run python scripts/drive.py --name lap1        # with the rerun viewer
    uv run python scripts/drive.py --name lap1 --no-viz
"""

import argparse
import contextlib
import logging
import math
import queue
import threading
import time

import numpy as np

from pepin.base import DiffDriveBase
from pepin.bus import verify_motors
from pepin.feetech import FeetechTcpClient
from pepin.geometry import BaseConfig
from pepin.kinematics import Twist
from pepin.lidar import LaserScan, LidarMount, LidarStream, TcpSource
from pepin.log import setup_logging
from pepin.odometry import DiffDriveOdometry, Pose2D
from pepin.recording import SessionRecorder
from pepin.teleop import DriveState, KeyReader, apply_key
from pepin.transport import DEFAULT_HOST, LIDAR_PORT, SERVO_BUS_PORT

logger = logging.getLogger(__name__)

LOOP_HZ = 20
STATUS_EVERY_S = 2.0
BUS_STOP_AFTER_S = 1.0  # consecutive bus downtime after which the wheels are stopped
BUS_GIVE_UP_AFTER_S = 10.0  # consecutive bus downtime after which the session aborts


def lidar_thread(mount: LidarMount, scans: "queue.Queue[LaserScan]", stop: threading.Event) -> None:
    stream = LidarStream(TcpSource(DEFAULT_HOST, LIDAR_PORT), mount)
    try:
        for scan in stream.scans():
            scans.put(scan)
            if stop.is_set():
                break
    finally:
        stream.close()
        logger.info(
            "lidar thread stopped: %d frames, %d crc failures",
            stream.parser.frames,
            stream.parser.crc_failures,
        )


class Viewer:
    """Live rerun view: robot path, heading, and the latest scan in the odometry frame."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._path: list[tuple[float, float]] = []
        if enabled:
            import rerun as rr

            self._rr = rr
            rr.init("pepin", spawn=True)

    def update(self, t: float, pose: Pose2D, scan: LaserScan | None, mount: LidarMount) -> None:
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
            local = scan.points_xy(mount)
            c, s = math.cos(pose.theta), math.sin(pose.theta)
            world = local @ np.array([[c, s], [-s, c]]) + np.array([pose.x, pose.y])
            rr.log("world/scan", rr.Points2D(world, colors=[255, 200, 0], radii=0.01))


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard driving with recording and live view.")
    parser.add_argument("--name", default="drive", help="session name for the recording file")
    parser.add_argument("--no-viz", action="store_true", help="do not start the rerun viewer")
    args = parser.parse_args()
    setup_logging("drive", console=False)  # the terminal is the key-input UI

    base_cfg = BaseConfig.from_json("config/base.json")
    mount = LidarMount.from_json("config/lidar.json")
    motors = DiffDriveBase.motor_ids(base_cfg)

    scans: queue.Queue[LaserScan] = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=lidar_thread, args=(mount, scans, stop), daemon=True).start()
    viewer = Viewer(enabled=not args.no_viz)

    print("W/S speed  A/D turn  space stop  Q quit")
    with (
        FeetechTcpClient(DEFAULT_HOST, SERVO_BUS_PORT, motors) as bus,
        SessionRecorder("data/sessions", args.name) as rec,
        KeyReader() as keys,
    ):
        verify_motors(bus, list(motors))
        odom = DiffDriveOdometry(base_cfg.geometry)
        state = DriveState()
        rec.note(f"session {args.name} start")
        with DiffDriveBase(bus, base_cfg) as base:
            base.read_wheel_travel()  # prime encoders
            t0 = time.monotonic()
            next_status = t0
            latest_scan: LaserScan | None = None
            bus_failures = 0
            failing_since: float | None = None
            stop_commanded = False
            while not state.quit:
                tick = time.monotonic()
                key = keys.read()
                new_twist: Twist | None = None
                if key is not None:
                    new_state = apply_key(state, key)
                    if new_state.twist != state.twist:
                        new_twist = new_state.twist
                    state = new_state
                pose = odom.pose  # kept unchanged on a tick the bus does not answer
                try:
                    if new_twist is not None:
                        base.set_twist(new_twist)
                        rec.command(new_twist)
                    travel = base.read_wheel_travel()
                except TimeoutError as exc:
                    # The encoders are absolute and the unwrapper tolerates gaps,
                    # so a lost tick only costs resolution, not odometry validity.
                    bus_failures += 1
                    if failing_since is None:
                        failing_since = tick
                    down = tick - failing_since
                    logger.warning(
                        "bus timeout, tick skipped (%d total, down %.1f s): %s",
                        bus_failures,
                        down,
                        exc,
                    )
                    if down >= BUS_GIVE_UP_AFTER_S:
                        raise RuntimeError("bus unreachable") from exc
                    if not stop_commanded and down >= BUS_STOP_AFTER_S:
                        stop_commanded = True
                        logger.warning("bus down %.1f s: commanding stop", down)
                        with contextlib.suppress(TimeoutError):
                            base.stop()
                else:
                    if failing_since is not None:
                        logger.info(
                            "bus recovered after %.1f s (%d ticks lost)",
                            tick - failing_since,
                            bus_failures,
                        )
                        failing_since = None
                        stop_commanded = False
                    pose = odom.update(*travel)
                    rec.pose(pose, travel)
                while not scans.empty():
                    latest_scan = scans.get_nowait()
                    rec.scan(latest_scan)
                viewer.update(tick - t0, pose, latest_scan, mount)
                if tick >= next_status:
                    next_status = tick + STATUS_EVERY_S
                    print(
                        f"\rv={state.twist.linear:+.2f} w={state.twist.angular:+.2f} | "
                        f"x={pose.x:+.2f} y={pose.y:+.2f} th={math.degrees(pose.theta):+.0f}deg | "
                        f"{bus.latency.summary()}   ",
                        end="",
                        flush=True,
                    )
                    logger.info(
                        "pose x=%.3f y=%.3f th=%.3f twist=%s %s",
                        pose.x,
                        pose.y,
                        pose.theta,
                        state.twist,
                        bus.latency.summary(),
                    )
                time.sleep(max(0.0, 1.0 / LOOP_HZ - (time.monotonic() - tick)))
        rec.note("session end")
    stop.set()
    print(f"\nsession saved: {rec.path} ({rec.records} records)")
    logger.info("session saved: %s (%d records)", rec.path, rec.records)


if __name__ == "__main__":
    main()
