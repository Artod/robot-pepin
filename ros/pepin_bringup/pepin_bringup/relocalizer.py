"""Global relocalisation on top of AMCL: the robot always knows where it is, no human needed.

AMCL tracks well but only searches where its particles already are: carry the
cart across the room, push it, lift it onto the carpet, and it keeps believing
the old pose. This node closes that gap with the correlative whole-map search
from :mod:`pepin.localization` (a fraction of a second on the pooled grid, with
the twin check). Every second it scores how well the current scan lies on the
map at AMCL's pose; when the fit stays poor it searches the whole map and, if a
clearly better pose exists, re-seeds AMCL through ``/initialpose``. The same
search answers the ``/relocalize`` service on demand, and ``/where_am_i``
reports pose and fit as text.

Frames: the scan is transformed into ``base_link`` with the static laser
transform looked up once; poses are in ``map``.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.msg import ParticleCloud
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from pepin.localization import Localizer
from pepin.mapping import GridSpec, OccupancyGrid
from pepin.odometry import Pose2D
from pepin.scanmatch import CorrelativeMatcher, SearchWindow

OCCUPIED_LOG_ODDS, FREE_LOG_ODDS = 4.0, -4.0
MAX_BACKOFF_S = 60.0  # a robot that cannot find itself must not saturate the board searching
DUMP_DIR = (
    "/maps/rec"  # every failed whole-map search leaves its scan here, for the offline autopsy
)


def backoff_wait(cooldown_s: float, failed_searches: int) -> float:
    """Seconds to wait before the next search: the cooldown doubled per failure, capped."""
    return min(cooldown_s * 2.0**failed_searches, MAX_BACKOFF_S)


class _RosLogHandler(logging.Handler):
    """Forwards the Python-side localizer's log lines to the node's ROS logger."""

    def __init__(self, node: Node) -> None:
        super().__init__()
        self._node = node

    def emit(self, record: logging.LogRecord) -> None:
        text = f"[{record.name}] {record.getMessage()}"
        if record.levelno >= logging.WARNING:
            self._node.get_logger().warning(text)
        else:
            self._node.get_logger().info(text)


def yaw_of(q: Any) -> float:
    """Yaw of a geometry_msgs quaternion (planar robot)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def grid_from_msg(msg: OccupancyGridMsg) -> OccupancyGrid:
    """A nav_msgs map as our log-odds grid (occupied +4, free -4, unknown 0)."""
    info = msg.info
    spec = GridSpec(
        info.resolution,
        info.origin.position.x,
        info.origin.position.y,
        info.width * info.resolution,
        info.height * info.resolution,
    )
    grid = OccupancyGrid(spec)
    data = np.array(msg.data, dtype=np.int16).reshape(info.height, info.width)
    grid.log_odds[:] = np.where(
        data >= 65, OCCUPIED_LOG_ODDS, np.where((data >= 0) & (data <= 35), FREE_LOG_ODDS, 0.0)
    )
    grid.version += 1
    return grid


class Relocalizer(Node):
    """Watches the scan-to-map fit at AMCL's pose and re-seeds AMCL from a whole-map search."""

    def __init__(self) -> None:
        super().__init__("relocalizer")
        self._scan_topic = str(self.declare_parameter("scan_topic", "/scan").value)
        self._lost_fit = float(self.declare_parameter("lost_fit", 0.35).value)
        self._lost_checks = int(self.declare_parameter("lost_checks", 5).value)
        self._min_inliers = float(self.declare_parameter("min_global_inliers", 0.5).value)
        self._cooldown_s = float(self.declare_parameter("cooldown_s", 8.0).value)
        self._check_period_s = float(self.declare_parameter("check_period_s", 1.0).value)

        self._failed_searches = 0  # each failure doubles the wait before the next search
        self._last_odom: Pose2D | None = None  # odom->base_link at the previous check
        self._navigating = False  # a NavigateToPose goal is executing
        self._grid: OccupancyGrid | None = None
        self._matcher: CorrelativeMatcher | None = None
        logging.getLogger("pepin").addHandler(
            _RosLogHandler(self)
        )  # localizer reasons in the ROS log
        self._localizer: Localizer | None = None
        self._points: np.ndarray | None = None
        self._laser_tf: tuple[float, float, float, bool] | None = None  # x, y, yaw, mirrored
        self._poor_streak = 0
        self._last_reseed = -1e9
        self._searching = False
        self.fit = float("nan")
        # Stage one: a wide window around the pose AMCL believes in (a push, a short carry); the
        # whole map only when that fails. On four A53 cores the whole-map lattice takes ~15 s,
        # the window about a second.
        self._near = SearchWindow(xy_m=2.0, xy_step_m=0.2, theta_deg=180.0, theta_step_deg=10.0)

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(OccupancyGridMsg, "/map", self._on_map, latched)
        self.create_subscription(LaserScan, self._scan_topic, self._on_scan, 5)
        self._fit_pub = self.create_publisher(Float32, "localization_fit", 5)
        self._pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 5)
        # Eyes for the operator: AMCL's particles as arrows and its pose history as a line.
        # nav2_msgs/ParticleCloud is unknown to Foxglove; PoseArray and Path are not.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._particles_pub = self.create_publisher(PoseArray, "/particle_poses", 1)
        self._trail_pub = self.create_publisher(Path, "/amcl_path", latched)
        self._trail = Path()
        self._trail.header.frame_id = "map"
        self.create_subscription(
            ParticleCloud, "/particle_cloud", self._on_particles, qos_profile_sensor_data
        )
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self._on_amcl_pose, 10)
        self.create_service(Trigger, "relocalize", self._on_relocalize)
        self.create_service(Trigger, "where_am_i", self._on_where)
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self.create_timer(self._check_period_s, self._check)
        self.get_logger().info("relocalizer up: watching the scan-to-map fit")

    # -- inputs -------------------------------------------------------------

    def _on_map(self, msg: OccupancyGridMsg) -> None:
        self._grid = grid_from_msg(msg)
        self._matcher = CorrelativeMatcher(self._grid)
        self._localizer = Localizer(self._grid, Pose2D())
        self.get_logger().info(f"map received: {msg.info.width}x{msg.info.height} cells")

    def _on_scan(self, msg: LaserScan) -> None:
        if self._laser_tf is None and not self._lookup_laser(msg.header.frame_id):
            return
        assert self._laser_tf is not None
        lx, ly, lyaw, mirrored = self._laser_tf
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        ok = np.isfinite(ranges) & (ranges > 0.05) & (ranges < msg.range_max)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        px, py = ranges[ok] * np.cos(angles[ok]), ranges[ok] * np.sin(angles[ok])
        if mirrored:
            py = -py
        c, s = math.cos(lyaw), math.sin(lyaw)
        self._points = np.column_stack((lx + c * px - s * py, ly + s * px + c * py))

    def _lookup_laser(self, frame: str) -> bool:
        """The static base_link <- laser transform, as x, y, yaw and whether roll is pi."""
        try:
            t = self._tf.lookup_transform("base_link", frame, rclpy.time.Time())
        except Exception:  # not yet available: try on the next scan
            return False
        q = t.transform.rotation
        roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        mirrored = abs(abs(roll) - math.pi) < 0.2
        self._laser_tf = (t.transform.translation.x, t.transform.translation.y, yaw_of(q), mirrored)
        self.get_logger().info(
            f"laser mount: x {self._laser_tf[0]:.3f} yaw {math.degrees(self._laser_tf[2]):.1f} deg"
            f"{' upside down' if mirrored else ''}"
        )
        self.create_subscription(
            GoalStatusArray, "/navigate_to_pose/_action/status", self._on_nav_status, 10
        )
        return True

    def _on_particles(self, msg: ParticleCloud) -> None:
        """Every AMCL particle set, thinned to 300 arrows, as a PoseArray for the 3D view."""
        step = max(1, len(msg.particles) // 300)
        out = PoseArray()
        out.header = msg.header
        out.poses = [particle.pose for particle in msg.particles[::step]]
        self._particles_pub.publish(out)

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        """Append AMCL's pose to the trail (last 600 poses) and republish it, latched."""
        stamped = PoseStamped()
        stamped.header = msg.header
        stamped.pose = msg.pose.pose
        self._trail.poses.append(stamped)
        del self._trail.poses[:-600]
        self._trail.header.stamp = msg.header.stamp
        self._trail_pub.publish(self._trail)

    def _on_nav_status(self, msg: GoalStatusArray) -> None:
        """Remember whether Nav2 is executing a goal: no automatic re-seeding while it drives."""
        self._navigating = any(
            s.status in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING)
            for s in msg.status_list
        )

    def _moving(self) -> bool:
        """Did base_link move in the odom frame since the previous check (1 cm or 1 degree)?"""
        try:
            t = self._tf.lookup_transform("odom", "base_link", rclpy.time.Time())
        except Exception:
            return False
        now = Pose2D(
            t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation)
        )
        before, self._last_odom = self._last_odom, now
        if before is None:
            return False
        turned = abs(
            math.atan2(math.sin(now.theta - before.theta), math.cos(now.theta - before.theta))
        )
        return math.hypot(now.x - before.x, now.y - before.y) > 0.01 or turned > math.radians(1.0)

    def _amcl_pose(self) -> Pose2D | None:
        try:
            t = self._tf.lookup_transform("map", "base_link", rclpy.time.Time())
        except Exception:
            return None
        return Pose2D(
            t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation)
        )

    # -- the watch ------------------------------------------------------------

    def _check(self) -> None:
        """Once a second: score the fit; after several poor scores, search (in a worker thread)."""
        if self._matcher is None or self._points is None:
            return
        pose = self._amcl_pose()
        if pose is None:
            return
        self.fit = self._matcher.inlier_fraction(pose, self._points)
        self._fit_pub.publish(Float32(data=float(self.fit)))
        wait = backoff_wait(self._cooldown_s, self._failed_searches)
        if self._searching or time.monotonic() - self._last_reseed < wait:
            return  # a search is running, AMCL is applying a seed, or we are backing off
        if self._moving() or self._navigating:
            # A driving robot scores low for honest reasons (scan and pose a few ms apart, AMCL
            # mid-correction), and a robot wedged at its start scores low too; re-seeding either
            # from a whole-map search teleports its belief (2026-09-06: two 3 m jumps during one
            # lap while the wheels stood still, the goal then "succeeded" in the wrong room).
            # While Nav2 executes a goal, AMCL and the costmaps are in charge; a lost robot fails
            # its goal and can be re-seeded afterwards or on request (/relocalize).
            self._poor_streak = 0
            return
        self._poor_streak = self._poor_streak + 1 if self.fit < self._lost_fit else 0
        if self._poor_streak >= self._lost_checks:
            self.get_logger().warning(f"fit {self.fit:.2f} for {self._poor_streak} s: searching")
            self._searching = True
            threading.Thread(target=self._search_and_seed, daemon=True).start()

    def _dump_failure(
        self, points: Any, current: Pose2D | None, best: Pose2D | None, confidence: float
    ) -> None:
        """Write the scan and the verdict of a failed search as JSON (a few tens of kB)."""
        try:
            import json
            import os

            os.makedirs(DUMP_DIR, exist_ok=True)
            path = f"{DUMP_DIR}/reloc_fail_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(path, "w") as f:
                json.dump(
                    {
                        "points": np.round(points, 3).tolist(),
                        "current": None
                        if current is None
                        else [current.x, current.y, current.theta],
                        "best": None if best is None else [best.x, best.y, best.theta],
                        "confidence": confidence,
                    },
                    f,
                )
            self.get_logger().info(f"search dumped to {path}")
        except OSError as exc:
            self.get_logger().warning(f"could not dump the failed search: {exc}")

    def _search_and_seed(self) -> None:
        """Worker thread: the search must not block the executor (scan and service callbacks)."""
        try:
            self._relocalize()
        finally:
            self._searching = False

    def _relocalize(self) -> str:
        """Nearby window first, then the whole map; re-seed AMCL when clearly better."""
        assert self._localizer is not None and self._matcher is not None
        points = self._points
        assert points is not None
        current = self._amcl_pose()
        current_fit = self._matcher.inlier_fraction(current, points) if current else 0.0
        started = time.monotonic()
        stage, found, confidence = "nearby", None, 0.0
        if current is not None:
            coarse = self._matcher.match(current, points, self._near)
            found, confidence = self._localizer.refine(coarse.pose, points)
        if found is None or confidence < self._min_inliers or confidence < current_fit + 0.1:
            stage = "whole map"
            found, confidence = self._localizer.global_search(
                points, theta_step_deg=10.0, thin_to=90, prior=current
            )
        took = time.monotonic() - started
        self._last_reseed = time.monotonic()  # grace starts now: AMCL needs a moment to apply it
        self._poor_streak = 0
        if confidence < self._min_inliers or confidence < current_fit + 0.1:
            text = (
                f"no better pose ({stage}, {took:.1f} s): "
                f"best {confidence:.2f} vs now {current_fit:.2f}"
            )
            self._failed_searches += 1
            self.get_logger().warning(text)
            self._dump_failure(points, current, found.pose if found else None, confidence)
            return text
        self._failed_searches = 0
        self._seed(found.pose)
        text = (
            f"relocalised ({stage}, {took:.1f} s) to ({found.pose.x:+.2f}, {found.pose.y:+.2f}, "
            f"{math.degrees(found.pose.theta):+.0f} deg): fit {current_fit:.2f} -> {confidence:.2f}"
        )
        self.get_logger().info(text)
        return text

    def _seed(self, pose: Pose2D) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = pose.x
        msg.pose.pose.position.y = pose.y
        msg.pose.pose.orientation.z = math.sin(pose.theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(pose.theta / 2.0)
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.05**2
        cov[35] = math.radians(5.0) ** 2
        msg.pose.covariance = cov
        self._pose_pub.publish(msg)

    # -- services ---------------------------------------------------------

    def _on_relocalize(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        if self._localizer is None or self._points is None:
            res.success, res.message = False, "no map or no scan yet"
            return res
        if self._searching:
            res.success, res.message = False, "a search is already running"
            return res
        res.message = self._relocalize()
        res.success = res.message.startswith("relocalised")
        return res

    def _on_where(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        pose = self._amcl_pose()
        if pose is None:
            res.success, res.message = False, "no map->base_link transform yet"
            return res
        res.success = True
        res.message = (
            f"x {pose.x:+.2f} m, y {pose.y:+.2f} m, yaw {math.degrees(pose.theta):+.0f} deg;"
            f" scan-to-map fit {self.fit:.2f} (good > 0.5, lost < {self._lost_fit})"
        )
        return res


def main(args: list[str] | None = None) -> None:
    """Entry point."""
    rclpy.init(args=args)
    node = Relocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
