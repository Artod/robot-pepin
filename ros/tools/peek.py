"""Print one message from a topic, as a flat dict of its numeric fields: the ``ros2 topic echo``
that costs the board one rclpy node instead of the CLI's seconds of discovery.

    docker exec pepin-ros /pepin_entrypoint.sh python3 /tools/peek.py /odometry/filtered nav_msgs/msg/Odometry [seconds]
"""

from __future__ import annotations

import importlib
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


def flatten(msg: object, prefix: str = "") -> dict[str, float]:
    """``{"pose.pose.position.x": 1.0, ...}`` for every numeric leaf of a ROS message."""
    out: dict[str, float] = {}
    fields = getattr(msg, "get_fields_and_field_types", None)
    if fields is None:
        return out
    for name in fields():
        value = getattr(msg, name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[prefix + name] = value
        elif hasattr(value, "get_fields_and_field_types"):
            out.update(flatten(value, f"{prefix}{name}."))
    return out


def main() -> None:
    topic, type_name = sys.argv[1], sys.argv[2]
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
    package, _, name = type_name.rpartition("/")
    cls = getattr(importlib.import_module(package.replace("/", ".")), name)
    rclpy.init()
    node = Node("peek")
    got: list[object] = []
    node.create_subscription(cls, topic, got.append, qos_profile_sensor_data)
    end = time.monotonic() + seconds
    while time.monotonic() < end and not got:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not got:
        print(f"{topic}: nothing in {seconds:.0f} s")
        return
    flat = flatten(got[0])
    keep = {k: round(v, 4) for k, v in flat.items() if "covariance" not in k}
    print(topic, keep)


if __name__ == "__main__":
    main()
