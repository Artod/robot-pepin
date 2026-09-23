"""Ask Nav2's lifecycle manager to bring its nodes up (or down) without restarting the stack.

    docker exec pepin-ros /pepin_entrypoint.sh python3 /tools/nav2_manage.py startup|shutdown|reset [manager]

The manager's own bringup at launch can abort when map -> odom is not yet in TF (RTAB-Map on the
laptop localises a minute after the board is up); this re-runs it once the frame exists.
"""

from __future__ import annotations

import sys

import rclpy
from nav2_msgs.srv import ManageLifecycleNodes
from rclpy.node import Node

COMMANDS = {"startup": 0, "pause": 1, "resume": 2, "reset": 3, "shutdown": 4}


def main() -> None:
    command = sys.argv[1]
    manager = sys.argv[2] if len(sys.argv) > 2 else "lifecycle_manager_navigation_all"
    rclpy.init()
    node = Node("nav2_manage")
    client = node.create_client(ManageLifecycleNodes, f"/{manager}/manage_nodes")
    if not client.wait_for_service(timeout_sec=20.0):
        print(f"{manager}: manage_nodes service not reachable in 20 s")
        sys.exit(1)
    request = ManageLifecycleNodes.Request()
    request.command = COMMANDS[command]
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=120.0)
    result = future.result()
    print(f"{manager}: {command} -> {'success' if result and result.success else 'FAILED or timed out'}")
    sys.exit(0 if result and result.success else 1)


if __name__ == "__main__":
    main()
