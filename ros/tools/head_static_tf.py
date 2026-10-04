#!/usr/bin/env python3
"""The camera's static edges for an offline replay: camera_link -> camera_optical and, with a
measured head IMU, camera_optical -> head_imu, from config/camera.json (pepin.mounts), the numbers
camera_stream broadcasts live. A drive's bag carries the board's base_link -> camera_link (the
neck chain); these two are the laptop's and are not in it.

    python3 /repo/ros/tools/head_static_tf.py --ros-args -p use_sim_time:=true
"""

from __future__ import annotations

import contextlib
import sys


def main() -> int:
    """Broadcast the edges latched and spin until interrupted."""
    import rclpy
    from pepin_bringup.msgs import transform_from_mount
    from rclpy.node import Node
    from tf2_ros import StaticTransformBroadcaster

    from pepin.mounts import load_camera_mounts

    rclpy.init(args=sys.argv)
    node = Node("head_static_tf")
    camera = load_camera_mounts()
    stamp = node.get_clock().now().to_msg()
    edges = [transform_from_mount(camera.link_frame, camera.optical_frame, camera.optical, stamp)]
    if camera.imu is not None:
        edges.append(
            transform_from_mount(camera.optical_frame, camera.imu_frame, camera.imu, stamp)
        )
    broadcaster = StaticTransformBroadcaster(node)
    broadcaster.sendTransform(edges)
    node.get_logger().info(
        "static: " + ", ".join(f"{e.header.frame_id} -> {e.child_frame_id}" for e in edges)
    )
    with contextlib.suppress(KeyboardInterrupt):
        rclpy.spin(node)
    return 0


if __name__ == "__main__":
    sys.exit(main())
