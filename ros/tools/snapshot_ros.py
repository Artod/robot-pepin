#!/usr/bin/env python3
"""Grab one snapshot of the navigation state from ROS topics and write it as JSON.

Runs INSIDE the container (rclpy): the static map, the newest laser scan, the
map->laser and map->base_link transforms, the AMCL pose, the current plan and
the published footprint. `draw_snapshot.py` on the laptop turns the JSON into a
picture — eyes on the robot without Foxglove.

    docker exec pepin-ros /pepin_entrypoint.sh python3 /tmp/snapshot_ros.py /tmp/snapshot.json
"""

import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PolygonStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


def yaw_of(q) -> float:  # type: ignore[no-untyped-def]
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Snapshot(Node):
    def __init__(self) -> None:
        super().__init__("snapshot")
        self.data: dict = {}
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        sensor = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(OccupancyGrid, "/map", self.on_map, latched)
        self.create_subscription(LaserScan, "/scan", self.on_scan, sensor)
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.on_amcl, 5)
        self.create_subscription(Path, "/plan", self.on_plan, 5)
        footprint_topic = "/local_costmap/published_footprint"
        self.create_subscription(PolygonStamped, footprint_topic, self.on_fp, 5)
        self.create_subscription(OccupancyGrid, "/local_costmap/costmap", self.on_local, 2)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)

    def on_map(self, m: OccupancyGrid) -> None:
        self.data["map"] = {
            "w": m.info.width,
            "h": m.info.height,
            "res": m.info.resolution,
            "ox": m.info.origin.position.x,
            "oy": m.info.origin.position.y,
            "data": list(m.data),
        }

    def on_local(self, m: OccupancyGrid) -> None:
        self.data["local"] = {
            "w": m.info.width,
            "h": m.info.height,
            "res": m.info.resolution,
            "ox": m.info.origin.position.x,
            "oy": m.info.origin.position.y,
            "data": list(m.data),
        }

    def on_scan(self, s: LaserScan) -> None:
        ranges = [None if (r != r or r == float("inf")) else r for r in s.ranges]
        self.data["scan"] = {
            "angle_min": s.angle_min,
            "inc": s.angle_increment,
            "ranges": ranges,
            "frame": s.header.frame_id,
            "stamp": s.header.stamp.sec + s.header.stamp.nanosec * 1e-9,
        }

    def on_amcl(self, p: PoseWithCovarianceStamped) -> None:
        q = p.pose.pose.orientation
        self.data["amcl"] = {
            "x": p.pose.pose.position.x,
            "y": p.pose.pose.position.y,
            "yaw": yaw_of(q),
            "cov_xx": p.pose.covariance[0],
            "cov_yy": p.pose.covariance[7],
        }

    def on_plan(self, p: Path) -> None:
        self.data["plan"] = [[ps.pose.position.x, ps.pose.position.y] for ps in p.poses]

    def on_fp(self, p: PolygonStamped) -> None:
        self.data["footprint"] = [[pt.x, pt.y] for pt in p.polygon.points]

    def transforms(self) -> None:
        for child in ("laser", "base_link"):
            try:
                t = self.tf.lookup_transform("map", child, rclpy.time.Time())
                q = t.transform.rotation
                self.data[f"map_to_{child}"] = {
                    "x": t.transform.translation.x,
                    "y": t.transform.translation.y,
                    "yaw": yaw_of(q),
                    "roll_pi": abs(
                        abs(
                            math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
                        )
                        - math.pi
                    )
                    < 0.1,
                }
            except Exception as exc:
                self.data[f"map_to_{child}_error"] = str(exc)[:120]


def main() -> None:
    rclpy.init()
    node = Snapshot()
    budget_s = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        if "map" in node.data and "scan" in node.data and time.monotonic() > deadline - 6.0:
            break
    node.transforms()
    node.data["wall"] = time.time()
    with open(sys.argv[1], "w") as f:
        json.dump(node.data, f)
    summary = {k: (len(v) if isinstance(v, list) else "ok") for k, v in node.data.items()}
    summary.pop("map", None)
    print("snapshot:", summary)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
