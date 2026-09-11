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
from rclpy.node import Node
from rtabmap_msgs.msg import MapGraph
from tf2_ros import TransformBroadcaster

from pepin.tsdf import RigidPose
from pepin_bringup.msgs import pose_from_transform, transform_from_pose
from pepin_bringup.node_kit import spin_main

RATE_HZ = 10.0
FRAMES = ("map", "rtabmap")  # parent, child


class RtabmapFrame(Node):
    """Broadcasts map -> rtabmap from RTAB-Map's latest correction."""

    def __init__(self) -> None:
        super().__init__("rtabmap_frame")
        self._tf = TransformBroadcaster(self)
        self._pose = RigidPose(np.eye(3), np.zeros(3))
        self._graphs = 0
        self.create_subscription(MapGraph, "/rtabmap/mapGraph", self._on_graph, 5)
        self.create_timer(1.0 / RATE_HZ, self._broadcast)
        self.get_logger().info(
            "map -> rtabmap follows /rtabmap/mapGraph (identity until the first graph)"
        )

    def _on_graph(self, msg: MapGraph) -> None:
        """RTAB-Map's rtabmap -> map correction, kept the other way round: map -> rtabmap."""
        self._pose = pose_from_transform(msg.map_to_odom).inverse()
        self._graphs += 1

    def _broadcast(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self._tf.sendTransform(transform_from_pose(*FRAMES, self._pose, stamp))


def main() -> None:
    spin_main(RtabmapFrame)


if __name__ == "__main__":
    main()
