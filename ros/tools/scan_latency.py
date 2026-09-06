#!/usr/bin/env python3
"""Measure scan latency inside the container: header stamp vs arrival time, for N scans.

    docker exec pepin-ros /pepin_entrypoint.sh python3 /tools/scan_latency.py [/scan] [N]
A constant offset means the driver stamps wrongly (or buffers a fixed number of revolutions);
a growing one means the pipeline cannot keep up.
"""

import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


def main() -> None:
    topic = sys.argv[1] if len(sys.argv) > 1 else "/scan"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    rclpy.init()
    node = Node("scan_latency")
    lags: list[float] = []

    def on_scan(msg: LaserScan) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        lags.append(time.time() - stamp)

    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
    node.create_subscription(LaserScan, topic, on_scan, qos)
    deadline = time.monotonic() + 20.0
    while len(lags) < count and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    if lags:
        median = sorted(lags)[len(lags) // 2]
        lo, hi = min(lags), max(lags)
        print(
            f"{topic}: {len(lags)} scans, lag min {lo:.2f} s, median {median:.2f} s, max {hi:.2f} s"
        )
    else:
        print(f"{topic}: no scans in 20 s")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
