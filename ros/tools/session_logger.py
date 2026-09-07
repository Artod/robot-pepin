#!/usr/bin/env python3
"""Log the raw lidar scans and odometry to a jsonl session our offline SLAM can rebuild a map from.

Also logs AMCL's pose as ``loc`` when Nav2 runs. Runs inside the container; writes to
a host-mounted path, flushes every line
and fsyncs every two seconds, so a power cut mid-drive costs at most the last
two seconds — the file is the meteor-proof copy (a rosbag runs alongside it).

Format: exactly `pepin.recording.SessionRecorder`'s (topics ``scan`` and
``pose``), so `scripts/build_map.py --match --loop` consumes it unchanged.
The recorded angles are ROBOT-frame radians (that is what `LaserScan.angles`
holds after ingestion in the Python stack); the verified relation to the ROS
driver's counter-clockwise angle `a` is ``robot = -a - radians(87.5)`` (the
upside-down mount mirrors, the head points 87.5 degrees off; scan-vs-map fit
0.62-0.67 against the lap3 map with exactly this transform). Returns that land
inside the cart's own hull (its rear posts) are written as null, as the old
stack's masked sectors did — otherwise every pose grows phantom dots on the map.

    python3 /tools/session_logger.py /maps/rec/NAME.jsonl
"""

import json
import math
import os
import sys
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

from pepin.recording import scan_record_from_ros

FSYNC_EVERY_S = 2.0


class SessionLogger(Node):
    """Subscribes to the raw scan and odometry; appends one jsonl record per message."""

    def __init__(self, path: str) -> None:
        super().__init__("session_logger")
        # Deliberately not a context manager: the file must outlive __init__ and is closed in
        # main()'s finally with a final fsync. Line-buffered: every record hits the OS at once.
        self._file = open(path, "a", buffering=1)  # noqa: SIM115
        self._last_sync = time.monotonic()
        self.scans = 0
        self.poses = 0
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(LaserScan, "/ldlidar_node/scan", self._on_scan, qos)
        self.create_subscription(Odometry, "/odom", self._on_odom, 20)
        self.create_subscription(PoseWithCovarianceStamped, "/tracker_pose", self._on_amcl, 10)
        self.get_logger().info(f"logging to {path}")

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        now = time.monotonic()
        if now - self._last_sync > FSYNC_EVERY_S:
            self._last_sync = now
            os.fsync(self._file.fileno())

    def _on_scan(self, msg: LaserScan) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        record = scan_record_from_ros(
            stamp,
            msg.angle_min,
            msg.angle_increment,
            list(msg.ranges),
            list(msg.intensities),
            msg.range_min,
            msg.range_max,
            msg.scan_time,
            mount_yaw_rad=math.radians(87.5),
            mount_x_m=0.005,
        )
        self._write(record)
        self.scans += 1

    def _on_odom(self, msg: Odometry) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        q = msg.pose.pose.orientation
        theta = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._write(
            {
                "t": stamp,
                "topic": "pose",
                "x": round(msg.pose.pose.position.x, 4),
                "y": round(msg.pose.pose.position.y, 4),
                "theta": round(theta, 5),
            }
        )
        self.poses += 1

    def _on_amcl(self, msg: PoseWithCovarianceStamped) -> None:
        """The localizer's belief in the map frame; covariance trace as a stand-in confidence."""
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        q = msg.pose.pose.orientation
        theta = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        cov = msg.pose.covariance
        self._write(
            {
                "t": stamp,
                "topic": "loc",
                "x": round(msg.pose.pose.position.x, 4),
                "y": round(msg.pose.pose.position.y, 4),
                "theta": round(theta, 5),
                "confidence": round(1.0 / (1.0 + cov[0] + cov[7] + cov[35]), 3),
            }
        )


def main() -> None:
    rclpy.init()
    node = SessionLogger(sys.argv[1])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"logged {node.scans} scans, {node.poses} poses")
        node._file.flush()
        os.fsync(node._file.fileno())
        node._file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
