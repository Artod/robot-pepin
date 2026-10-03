#!/usr/bin/env python3
"""One pose topic of a bag as the scorer's CSV (stamp, x, y, yaw): ros/tools/vio_score.py's input.

    python3 /repo/ros/tools/bag_poses.py /rec/0601_..._arm_E.bag /odometry/filtered > E.csv

Reads nav_msgs/Odometry, geometry_msgs/PoseWithCovarianceStamped or PoseStamped by the message's
own header stamp (not the bag's receive time). Runs where rosbag2_py lives (the laptop image).
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Any


def yaw_of(q: Any) -> float:
    """Yaw of a geometry_msgs/Quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def pose_of(msg: Any) -> Any:
    """The geometry_msgs/Pose inside an Odometry, a PoseWithCovarianceStamped or a PoseStamped."""
    pose = msg.pose
    return pose.pose if hasattr(pose, "pose") else pose


def main(argv: list[str] | None = None) -> int:
    """Print the CSV."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("bag")
    parser.add_argument("topic")
    args = parser.parse_args(argv)
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=args.bag, storage_id=""),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if args.topic not in types:
        print(f"{args.topic} is not in {args.bag}: {sorted(types)}", file=sys.stderr)
        return 2
    kind = get_message(types[args.topic])
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))
    print("stamp,x,y,yaw")
    count = 0
    while reader.has_next():
        _topic, payload, _ns = reader.read_next()
        msg = deserialize_message(payload, kind)
        pose = pose_of(msg)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        print(
            f"{stamp:.9f},{pose.position.x:.6f},{pose.position.y:.6f},{yaw_of(pose.orientation):.6f}"
        )
        count += 1
    print(f"{count} poses of {args.topic}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
