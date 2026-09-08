#!/usr/bin/env python3
"""Does a rotation reach the pose? Compare gyro, wheels, EKF and tracker over the same turn.

Runs inside the container. With ``--spin`` it turns the robot in place itself (0.5 rad/s, stopped
by the base's deadman the moment this exits); without it, turn the robot by hand while it samples.

    python3 /tools/yaw_check.py [--spin DEGREES] [--seconds N]
"""

import math
import sys
import time
from itertools import pairwise

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

RATE = 0.5  # rad/s while spinning: slow enough for the tracker's window, fast enough to see


def yaw_of(q) -> float:  # type: ignore[no-untyped-def]
    """Yaw of a quaternion, radians."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def turned(values: list[float]) -> str:
    """How far a stream of yaw readings turned in total, unwrapped, with its message count."""
    if len(values) < 2:
        return "no data"
    total = 0.0
    for a, b in pairwise(values):
        total += math.atan2(math.sin(b - a), math.cos(b - a))
    return f"{math.degrees(total):+7.1f} deg ({len(values)} msgs)"


def main() -> None:
    args = sys.argv[1:]
    degrees = float(args[args.index("--spin") + 1]) if "--spin" in args else 0.0
    seconds = float(args[args.index("--seconds") + 1]) if "--seconds" in args else 10.0
    rclpy.init()
    node = rclpy.create_node("yaw_check")
    gyro = {"sum": 0.0, "t": None}
    wheels: list[float] = []
    ekf: list[float] = []
    tracker: list[float] = []

    def on_imu(msg: Imu) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if gyro["t"] is not None:
            gyro["sum"] += msg.angular_velocity.z * (stamp - gyro["t"])  # already in base axes
        gyro["t"] = stamp

    node.create_subscription(Imu, "/imu/data_raw", on_imu, 50)
    node.create_subscription(
        Odometry, "/odom", lambda m: wheels.append(yaw_of(m.pose.pose.orientation)), 20
    )
    node.create_subscription(
        Odometry, "/odometry/filtered", lambda m: ekf.append(yaw_of(m.pose.pose.orientation)), 20
    )
    node.create_subscription(
        PoseWithCovarianceStamped,
        "/tracker_pose",
        lambda m: tracker.append(yaw_of(m.pose.pose.orientation)),
        10,
    )
    publisher = node.create_publisher(Twist, "/cmd_vel", 10)
    spin_until = time.monotonic() + abs(math.radians(degrees)) / RATE if degrees else 0.0
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if time.monotonic() < spin_until:
            command = Twist()
            command.angular.z = RATE if degrees > 0 else -RATE
            publisher.publish(command)
        elif spin_until:
            publisher.publish(Twist())
        rclpy.spin_once(node, timeout_sec=0.05)
    publisher.publish(Twist())
    print(f"gyro integrated : {math.degrees(gyro['sum']):+7.1f} deg")
    print(f"wheel odometry  : {turned(wheels)}")
    print(f"EKF odom->base  : {turned(ekf)}")
    print(f"tracker in map  : {turned(tracker)}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
