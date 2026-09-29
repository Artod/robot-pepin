#!/usr/bin/env python3
"""Measure a topic's rate for a few seconds and print it on one line.

`ros2 topic hz` costs ~4.5 s of CLI start-up on this board and prints a paragraph;
this answers "is /odom alive and how fast" in one line, any message type.

    python3 /tools/topic_rate.py /odom [seconds=5] [latched]

``latched`` asks a latched topic the only question it can answer: is its last message here. The
subscription is TRANSIENT_LOCAL, so the publisher's held copy arrives at once; one message is a
pass. A grid published once per change has no rate (RTAB-Map's /map while it localises: one grid
in a whole start, 2026-09-24), and counting its messages in five seconds failed a healthy board.
"""

import sys
import time
from itertools import pairwise

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message


def main() -> None:
    topic = sys.argv[1]
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    latched = len(sys.argv) > 3 and sys.argv[3] == "latched"
    rclpy.init()
    node = rclpy.create_node("pepin_topic_rate")
    try:
        deadline = time.monotonic() + 5.0
        types: list[str] = []
        while time.monotonic() < deadline and not types:
            rclpy.spin_once(node, timeout_sec=0.2)
            types = dict(node.get_topic_names_and_types()).get(topic, [])
        if not types:
            print(f"{topic}: not advertised")
            sys.exit(2)
        stamps: list[float] = []
        qos: QoSProfile | int = 50
        if latched:
            qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
        node.create_subscription(
            get_message(types[0]), topic, lambda _m: stamps.append(time.monotonic()), qos
        )
        end = time.monotonic() + seconds
        while time.monotonic() < end and not (latched and stamps):
            rclpy.spin_once(node, timeout_sec=0.1)
        if latched:
            if not stamps:
                print(f"{topic}: no latched copy in {seconds:.0f} s")
                sys.exit(1)
            print(f"{topic}: latched copy received (a latched topic has no rate)")
            return
        if len(stamps) < 2:
            print(f"{topic}: {len(stamps)} messages in {seconds:.0f} s")
            sys.exit(1)
        gaps = [b - a for a, b in pairwise(stamps)]
        print(
            f"{topic}: {len(stamps) / seconds:.1f} Hz over {seconds:.0f} s"
            f" (period min {min(gaps) * 1000:.0f} ms, max {max(gaps) * 1000:.0f} ms)"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
