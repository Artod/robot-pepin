#!/usr/bin/env python3
"""Swap the map under a running Nav2 without restarting anything.

Calls map_server's LoadMap service; AMCL, both costmaps' static layers and the
relocalizer all subscribe to /map and pick the new one up. The frame stays the
same (all our maps start at the base), so the pose estimate carries over and
the relocalizer re-checks the fit on its next tick.

    python3 /tools/load_map.py /maps/flat3.yaml
"""

import sys

import rclpy
from nav2_msgs.srv import LoadMap

RESULTS = {
    LoadMap.Response.RESULT_SUCCESS: "loaded",
    LoadMap.Response.RESULT_MAP_DOES_NOT_EXIST: "no such file",
    LoadMap.Response.RESULT_INVALID_MAP_DATA: "invalid map data",
    LoadMap.Response.RESULT_INVALID_MAP_METADATA: "invalid yaml",
    LoadMap.Response.RESULT_UNDEFINED_FAILURE: "failed",
}


def main() -> None:
    url = sys.argv[1]
    rclpy.init()
    node = rclpy.create_node("pepin_load_map")
    client = node.create_client(LoadMap, "/map_server/load_map")
    try:
        if not client.wait_for_service(timeout_sec=15.0):
            print("map_server is not up (nav mode running?)")
            sys.exit(2)
        request = LoadMap.Request()
        request.map_url = url
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)
        result = future.result()
        if result is None:
            print("no answer from map_server")
            sys.exit(3)
        info = result.map.info
        verdict = RESULTS.get(result.result, str(result.result))
        print(f"{verdict}: {url} ({info.width}x{info.height} cells)")
        sys.exit(0 if result.result == LoadMap.Response.RESULT_SUCCESS else 1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
