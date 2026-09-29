"""The robot and the laptop's RTAB-Map, simulated for Nav2: everything Nav2 reads, from pepin.sim.

Stands in for exactly the publishers the stack has under ``PEPIN_LOCALIZER=rtabmap``, on the
same topics and frames: RTAB-Map's grid latched on ``/map``, its ``map -> odom`` (identity here:
the sim's pose is the truth), the places node's ``/places`` and rtabmap_frame's placement word
(so the goal server's start gate passes, as it does once RTAB-Map has recognised the room); the
board's ``odom -> base_link`` and ``/odom`` at 20 Hz, ``base_link -> laser`` from
config/lidar.json, and ``/scan`` at 10 Hz in the laser frame, raycast from the grid and the boxes
through the same mount and scan filter. ``/cmd_vel`` drives a unicycle at the base's own limits
behind the base bridge's 0.5 s deadman.

With ``--rate N`` (N > 0) the world owns the time: it publishes ``/clock`` and runs the room N
times faster than the wall clock, every node started with ``use_sim_time``. Without it every
stamp is the wall clock, as on the robot.

A control socket (JSON lines, one request per connection, like the goal server's) answers
``state`` (pose and odometer), ``place`` (teleport; both costmaps are emptied after it) and
``boxes`` (replace the furniture; the costmaps are emptied too).

    python3 /sim/sim_world.py --world /sim/worlds/flat.yaml --places /sim/worlds/flat.places.json
        [--rate N] [--start NAME | --x X --y Y --yaw-deg D] [--boxes JSON] [--port P]
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped, Twist
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid, Odometry
from pepin_bringup.node_kit import spin_main
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from pepin.deployment import config_file
from pepin.footprint import Footprint
from pepin.geometry import BaseConfig
from pepin.mounts import lidar_mount, load_lidar
from pepin.odometry import Pose2D
from pepin.places import PLACES_TOPIC
from pepin.sim import Box, Grid, LidarModel, SimWorld, UnicycleBase, places_from_payload
from pepin.watch import PLACEMENT_TOPIC, Placement

TICK_S = 0.02  # the world steps 50 times per (sim) second
TF_HZ = 20.0  # map -> odom, odom -> base_link and /odom
SCAN_HZ = 10.0  # the LD19 at the board's configuration
REPORT_S = 10.0  # the report line, wall seconds
CONTROL_PORT = 3390
CLEAR_SERVICES = (
    "/local_costmap/clear_entirely_local_costmap",
    "/global_costmap/clear_entirely_global_costmap",
)
LATCHED = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
)


def quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """x, y, z, w of the rotation Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def stamp(seconds: float) -> Time:
    """builtin_interfaces/Time of ``seconds``."""
    nanoseconds = round(seconds * 1e9)
    return Time(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


class WorldClock:
    """The world's time: the wall clock (``rate`` 0), or sim seconds that advance one tick per
    step and are paced at ``rate`` times the wall clock."""

    def __init__(self, rate: float) -> None:
        self.rate = rate
        self.now = time.time()
        self._wall0 = time.monotonic()
        self._sim0 = self.now
        self._last = time.monotonic()

    @property
    def simulated(self) -> bool:
        """Whether this clock publishes /clock."""
        return self.rate > 0

    def advance(self) -> float:
        """One step: returns its dt (sim seconds) and moves :attr:`now`."""
        if self.simulated:
            self.now += TICK_S
            return TICK_S
        wall = time.monotonic()
        dt = min(wall - self._last, 0.2)
        self._last = wall
        self.now = time.time()
        return dt

    def sleep(self) -> None:
        """Until the next step is due."""
        if self.simulated:
            due = self._wall0 + (self.now - self._sim0) / self.rate
            time.sleep(max(0.0, due - time.monotonic()))
        else:
            time.sleep(max(0.0, self._last + TICK_S - time.monotonic()))

    def achieved(self) -> float:
        """Sim seconds per wall second since the start (1.0 on the wall clock)."""
        wall = time.monotonic() - self._wall0
        return (self.now - self._sim0) / wall if self.simulated and wall > 0 else 1.0


class SimWorldNode(Node):
    """Publishes the simulated robot and map; steps the world on a thread of its own."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("sim_world")
        self._time = WorldClock(args.rate)
        grid = Grid.load(Path(args.world))
        base = UnicycleBase.from_config(
            BaseConfig.from_json(config_file("base.json")),
            Pose2D(args.x, args.y, math.radians(args.yaw_deg)),
        )
        hull = Footprint.from_config(json.loads(config_file("base.json").read_text())["footprint"])
        self._lidar_mount = load_lidar()
        self.world = SimWorld(grid, base, LidarModel(self._lidar_mount, hull=hull), hull)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._changed = threading.Event()  # a teleport or new boxes: the stepper republishes
        self._scan_ms: list[float] = []

        self._tf = TransformBroadcaster(self)
        self._static = StaticTransformBroadcaster(self)
        self._clock_pub = self.create_publisher(Clock, "/clock", 10)
        self._odom = self.create_publisher(Odometry, "/odom", 10)
        self._scan = self.create_publisher(LaserScan, "/scan", 10)
        self._map = self.create_publisher(OccupancyGrid, "/map", LATCHED)
        self._places = self.create_publisher(String, PLACES_TOPIC, LATCHED)
        self._placement = self.create_publisher(String, PLACEMENT_TOPIC, LATCHED)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)
        self._clear = [self.create_client(ClearEntireCostmap, name) for name in CLEAR_SERVICES]

        self._publish_static()
        self._publish_latched(grid, Path(args.places) if args.places else None)
        self._port = args.port
        self._server = threading.Thread(target=self._serve, daemon=True)
        self._server.start()
        self._stepper = threading.Thread(target=self._run)
        self._stepper.start()
        self.create_timer(REPORT_S, self._report)  # this node keeps the wall clock
        self.get_logger().info(
            f"sim world up: grid {grid.width}x{grid.height} at {grid.resolution:.3f} m, cart at"
            f" ({args.x:+.3f}, {args.y:+.3f}, {args.yaw_deg:+.1f} deg), limits"
            f" {base.max_linear:.2f} m/s {base.max_angular:.2f} rad/s, deadman {base.deadman_s} s,"
            f" clock {'sim x' + str(args.rate) if self._time.simulated else 'wall'},"
            f" control port {self._port}"
        )

    # -- what the laptop's RTAB-Map would publish, latched once ----------------------------------

    def _publish_latched(self, grid: Grid, places: Path | None) -> None:
        """/map (RTAB-Map's grid, as rtabmap_frame relays it), /places and the placement word."""
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = stamp(self._time.now)
        msg.info.resolution = grid.resolution
        msg.info.width, msg.info.height = grid.width, grid.height
        msg.info.origin.position.x = grid.origin_x
        msg.info.origin.position.y = grid.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.cells.ravel().tolist()
        self._map.publish(msg)
        if places is not None:
            self._places.publish(String(data=places.read_text().strip()))
        # The sim's pose is the truth: RTAB-Map "recognised" the room at its first update.
        word = Placement(updates=1, recognised=1, seeds=0, loaded=True).to_json(self._time.now)
        self._placement.publish(String(data=word))

    def _publish_static(self) -> None:
        """base_link -> laser, the board launch's static edge, from config/lidar.json."""
        x, y, z, roll, pitch, yaw = lidar_mount(self._lidar_mount).transform()
        t = TransformStamped()
        t.header.stamp = stamp(self._time.now)
        t.header.frame_id, t.child_frame_id = "base_link", "laser"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, z
        qx, qy, qz, qw = quaternion(roll, pitch, yaw)
        t.transform.rotation.x, t.transform.rotation.y = qx, qy
        t.transform.rotation.z, t.transform.rotation.w = qz, qw
        self._static.sendTransform([t])

    # -- the world's loop -------------------------------------------------------------------------

    def _on_cmd(self, msg: Twist) -> None:
        """The last link of Nav2's command chain (the velocity smoother's output)."""
        with self._lock:
            self.world.base.command(float(msg.linear.x), float(msg.angular.z), self._time.now)

    def _run(self) -> None:
        """Step the base, then publish what is due: /clock every step, TF and odom at TF_HZ,
        the scan at SCAN_HZ, all in the world's time."""
        next_tf = next_scan = self._time.now
        while not self._stop.is_set():
            with self._lock:
                dt = self._time.advance()
                self.world.step(dt, self._time.now)
                now, pose, twist = self._time.now, self.world.pose, self.world.twist
            if self._time.simulated:
                self._clock_pub.publish(Clock(clock=stamp(now)))
            if self._changed.is_set():
                self._changed.clear()
                self._publish_pose(now, pose, twist)
                for client in self._clear:
                    if client.service_is_ready():
                        client.call_async(ClearEntireCostmap.Request())
                next_tf = now + 1.0 / TF_HZ
            if now >= next_tf:
                self._publish_pose(now, pose, twist)
                next_tf = max(next_tf + 1.0 / TF_HZ, now - 1.0 / TF_HZ)
            if now >= next_scan:
                self._publish_scan(now)
                next_scan = max(next_scan + 1.0 / SCAN_HZ, now - 1.0 / SCAN_HZ)
            self._time.sleep()

    def _publish_pose(self, now: float, pose: Pose2D, twist: tuple[float, float]) -> None:
        """map -> odom (identity), odom -> base_link and /odom."""
        header_stamp = stamp(now)
        map_odom = TransformStamped()
        map_odom.header.stamp = header_stamp
        map_odom.header.frame_id, map_odom.child_frame_id = "map", "odom"
        map_odom.transform.rotation.w = 1.0
        odom_base = TransformStamped()
        odom_base.header.stamp = header_stamp
        odom_base.header.frame_id, odom_base.child_frame_id = "odom", "base_link"
        odom_base.transform.translation.x, odom_base.transform.translation.y = pose.x, pose.y
        odom_base.transform.rotation.z = math.sin(pose.theta / 2.0)
        odom_base.transform.rotation.w = math.cos(pose.theta / 2.0)
        self._tf.sendTransform([map_odom, odom_base])
        odom = Odometry()
        odom.header.stamp = header_stamp
        odom.header.frame_id, odom.child_frame_id = "odom", "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y = pose.x, pose.y
        odom.pose.pose.orientation = odom_base.transform.rotation
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = twist
        self._odom.publish(odom)

    def _publish_scan(self, now: float) -> None:
        """One revolution from where the cart is now."""
        t0 = time.perf_counter()
        with self._lock:
            ranges = self.world.scan()
        self._scan_ms.append((time.perf_counter() - t0) * 1e3)
        lidar = self.world.lidar
        msg = LaserScan()
        msg.header.stamp = stamp(now)
        msg.header.frame_id = "laser"
        msg.angle_min = 0.0
        msg.angle_increment = lidar.angle_increment
        msg.angle_max = lidar.angle_increment * (lidar.beams - 1)
        msg.scan_time = 1.0 / SCAN_HZ
        msg.range_min = self._lidar_mount.min_range_m
        msg.range_max = self._lidar_mount.max_range_m
        msg.ranges = [float(r) for r in ranges]
        self._scan.publish(msg)

    def _report(self) -> None:
        """Every REPORT_S wall seconds: pose, odometer, the clock's pace, the scan's cost."""
        with self._lock:
            state = self.world.state(self._time.now)
            linear, angular = self.world.twist
        scan_ms, self._scan_ms = self._scan_ms, []
        cost = f"{statistics.median(scan_ms):.1f} ms median" if scan_ms else "no scans"
        self.get_logger().info(
            f"sim: ({state['x']:+.2f}, {state['y']:+.2f}, {state['yaw_deg']:+.0f} deg) at"
            f" {linear:+.2f} m/s {angular:+.2f} rad/s, path {state['path_m']:.2f} m, contacts"
            f" {state['contacts']} ({state['blocked_s']:.1f} s blocked), clock x"
            f"{self._time.achieved():.2f}, scan {cost} ({len(scan_ms)} scans)"
        )

    # -- the control socket -----------------------------------------------------------------------

    def _serve(self) -> None:
        """One JSON request per connection, one JSON answer."""
        server = socket.create_server(("127.0.0.1", self._port), reuse_port=True)
        while not self._stop.is_set():
            connection, _ = server.accept()
            with connection, connection.makefile("rw", encoding="utf-8") as stream:
                try:
                    request = json.loads(stream.readline() or "{}")
                except ValueError as error:
                    request = {"cmd": f"unparsable: {error}"}
                with self._lock:
                    answer = self.world.answer(request, self._time.now)
                if answer.get("event") in ("placed", "boxes"):
                    self._after_change(answer)
                stream.write(json.dumps(answer) + "\n")
                stream.flush()

    def _after_change(self, answer: dict[str, Any]) -> None:
        """A teleport or new furniture: the stepper publishes the new pose at its next step and
        empties both costmaps, whose marks belong to the room before the change."""
        ready = sum(1 for client in self._clear if client.service_is_ready())
        answer["costmaps_cleared"] = ready
        self._changed.set()
        self.get_logger().info(
            f"{answer['event']}: cart at ({answer['x']:+.3f}, {answer['y']:+.3f},"
            f" {answer['yaw_deg']:+.1f} deg), overlap {answer['overlap']}, boxes {answer['boxes']},"
            f" {ready} costmaps to clear"
        )

    def close(self) -> None:
        """Stop the stepping thread and join it (node_kit.spin_main calls this on the way out)."""
        self._stop.set()
        if self._stepper.is_alive():
            self._stepper.join(timeout=2.0)


def parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Our flags, and whatever is left for rclpy."""
    parser = argparse.ArgumentParser(description="the simulated robot for Nav2")
    parser.add_argument("--world", required=True, help="a map_server yaml (ros/sim/worlds)")
    parser.add_argument("--places", default="", help="the /places payload (<world>.places.json)")
    parser.add_argument(
        "--rate", type=float, default=0.0, help="sim seconds per wall second; 0 = wall"
    )
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--start", default="", help="a place of --places to start at (over x/y/yaw)"
    )
    parser.add_argument("--boxes", default="", help="a JSON list of boxes in the map frame")
    parser.add_argument("--port", type=int, default=CONTROL_PORT)
    return parser.parse_known_args(argv)


def main() -> None:
    """Build the node from the command line and spin it."""
    args, rest = parse(sys.argv[1:])
    if args.start:
        pose = places_from_payload(Path(args.places).read_text())[args.start]
        args.x, args.y, args.yaw_deg = pose.x, pose.y, math.degrees(pose.theta)

    def factory() -> SimWorldNode:
        node = SimWorldNode(args)
        if args.boxes:
            node.world.boxes = tuple(Box.from_dict(b) for b in json.loads(args.boxes))
        return node

    spin_main(factory, rest)


if __name__ == "__main__":
    main()
