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

import math
import threading
import time
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from pepin.localization import Localizer
from pepin.mapping import GridSpec, OccupancyGrid
from pepin.odometry import Pose2D
from pepin.scanmatch import CorrelativeMatcher, SearchWindow

OCCUPIED_LOG_ODDS, FREE_LOG_ODDS = 4.0, -4.0


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
        self._lost_checks = int(self.declare_parameter("lost_checks", 3).value)
        self._min_inliers = float(self.declare_parameter("min_global_inliers", 0.5).value)
        self._cooldown_s = float(self.declare_parameter("cooldown_s", 8.0).value)
        self._check_period_s = float(self.declare_parameter("check_period_s", 1.0).value)

        self._grid: OccupancyGrid | None = None
        self._matcher: CorrelativeMatcher | None = None
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
        return True

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
        if self._searching or time.monotonic() - self._last_reseed < self._cooldown_s:
            return  # a search is running, or AMCL has not applied the new pose yet
        self._poor_streak = self._poor_streak + 1 if self.fit < self._lost_fit else 0
        if self._poor_streak >= self._lost_checks:
            self.get_logger().warning(f"fit {self.fit:.2f} for {self._poor_streak} s: searching")
            self._searching = True
            threading.Thread(target=self._search_and_seed, daemon=True).start()

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
                points, theta_step_deg=10.0, thin_to=90
            )
        took = time.monotonic() - started
        self._last_reseed = time.monotonic()  # grace starts now: AMCL needs a moment to apply it
        self._poor_streak = 0
        if confidence < self._min_inliers or confidence < current_fit + 0.1:
            text = (
                f"no better pose ({stage}, {took:.1f} s): "
                f"best {confidence:.2f} vs now {current_fit:.2f}"
            )
            self.get_logger().warning(text)
            return text
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
