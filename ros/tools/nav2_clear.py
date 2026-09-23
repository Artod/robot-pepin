"""Clear both Nav2 costmaps entirely (the local and the global), without a restart.

    docker exec pepin-ros /pepin_entrypoint.sh python3 /tools/nav2_clear.py

Used before a drive whose sensors were changed while the stack ran: a costmap keeps the marks
an earlier volume painted until a ray clears them, and a phantom nobody sees again stays.
"""

from __future__ import annotations

import sys

import rclpy
from nav2_msgs.srv import ClearEntireCostmap
from rclpy.node import Node

SERVICES = ("/local_costmap/clear_entirely_local_costmap", "/global_costmap/clear_entirely_global_costmap")


def main() -> None:
    rclpy.init()
    node = Node("nav2_clear")
    failed = 0
    for name in SERVICES:
        client = node.create_client(ClearEntireCostmap, name)
        if not client.wait_for_service(timeout_sec=15.0):
            print(f"{name}: not reachable in 15 s")
            failed += 1
            continue
        future = client.call_async(ClearEntireCostmap.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=15.0)
        print(f"{name}: {'cleared' if future.done() and future.result() is not None else 'no answer'}")
        failed += 0 if future.done() else 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
