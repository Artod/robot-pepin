#!/usr/bin/env python3
"""Call a std_srvs/Trigger service and print its message; exit 1 when it reports failure.

`ros2 service call` spends ~4.5 s starting the CLI on this board (measured 2026-09-06);
a bare rclpy client answers in about a second.

    python3 /tools/call.py /where_am_i [timeout_s]
"""

import sys

import rclpy
from std_srvs.srv import Trigger


def main() -> None:
    service = sys.argv[1]
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    rclpy.init()
    node = rclpy.create_node("pepin_call")
    client = node.create_client(Trigger, service)
    try:
        if not client.wait_for_service(timeout_sec=10.0):
            print(f"{service}: no such service (is the stack up?)")
            sys.exit(2)
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
        result = future.result()
        if result is None:
            print(f"{service}: no answer within {timeout:.0f} s")
            sys.exit(3)
        print(result.message)
        sys.exit(0 if result.success else 1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
