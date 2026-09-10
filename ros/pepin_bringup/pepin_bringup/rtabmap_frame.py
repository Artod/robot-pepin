"""Tie RTAB-Map's frame to the tracker's map the way RTAB-Map itself would.

RTAB-Map keeps its graph in its own frame (``rtabmap``) and computes a correction from that
frame to its "odometry" frame every time a loop closes; its odometry here is the tracker's pose,
so that frame is ``map``, and RTAB-Map is not allowed to publish into the tracker's tree. With a
fixed identity in its place the voxels drifted away from the cart after every closure: the
cloud moved with the graph, the cart did not. This node reads the correction from
``/rtabmap/mapGraph`` and broadcasts its inverse as ``map -> rtabmap`` at 10 Hz, so a voxel and
the cart that saw it are drawn where RTAB-Map says they are, in one tree with the tracker's map.
"""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rtabmap_msgs.msg import MapGraph
from tf2_ros import TransformBroadcaster

from pepin.depth import Array, invert, quaternion_from_matrix, rotation_matrix

RATE_HZ = 10.0


class RtabmapFrame(Node):
    """Broadcasts map -> rtabmap from RTAB-Map's latest correction."""

    def __init__(self) -> None:
        super().__init__("rtabmap_frame")
        self._tf = TransformBroadcaster(self)
        self._rotation: Array = np.eye(3)
        self._translation: Array = np.zeros(3)
        self._graphs = 0
        self.create_subscription(MapGraph, "/rtabmap/mapGraph", self._on_graph, 5)
        self.create_timer(1.0 / RATE_HZ, self._broadcast)
        self.get_logger().info(
            "map -> rtabmap follows /rtabmap/mapGraph (identity until the first graph)"
        )

    def _on_graph(self, msg: MapGraph) -> None:
        t, q = msg.map_to_odom.translation, msg.map_to_odom.rotation
        self._rotation, self._translation = invert(
            rotation_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])
        )
        self._graphs += 1

    def _broadcast(self) -> None:
        out = TransformStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id, out.child_frame_id = "map", "rtabmap"
        x, y, z = (float(v) for v in self._translation)
        out.transform.translation.x, out.transform.translation.y, out.transform.translation.z = (
            x,
            y,
            z,
        )
        qx, qy, qz, qw = quaternion_from_matrix(self._rotation)
        out.transform.rotation.x, out.transform.rotation.y = qx, qy
        out.transform.rotation.z, out.transform.rotation.w = qz, qw
        self._tf.sendTransform(out)


def main() -> None:
    rclpy.init()
    node = RtabmapFrame()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
