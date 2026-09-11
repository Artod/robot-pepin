"""Put RTAB-Map's graph correction where the mode needs it: a frame here, or a message home.

RTAB-Map computes one correction per graph optimisation and publishes it as ``map_to_odom`` on
``/rtabmap/mapGraph``: the jump its "odometry" frame takes when the graph moves under it. Where
that correction belongs depends on which map the robot drives.

On a KNOWN map the board's tracker owns ``map -> odom`` and RTAB-Map's "odometry" IS the
tracker's pose, so its own frame (``rtabmap``) hangs off ``map``: this node broadcasts
``map -> rtabmap`` at 10 Hz, the inverse of the correction. With a fixed identity in its place
the voxels drifted away from the cart after every closure — the cloud moved with the graph, the
cart did not.

In online SLAM (``slam`` on, ros/laptop.sh vslam --slam) RTAB-Map IS the map and the correction
is literally ``map -> odom`` — but it must become a transform ON THE BOARD, where Nav2 and the
reflexes look it up, and ``/tf`` crosses the bridge board -> laptop only (a topic allowed as a
publisher on both sides loops until nothing crosses at all). So the correction travels as a
message on ``/map_odom`` and pepin_bringup.slam_frame broadcasts it there. This node then
publishes no transform at all: the laptop reads ``map -> odom`` back over the bridge, from the
one owner.
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rtabmap_msgs.msg import MapGraph
from tf2_ros import TransformBroadcaster

from pepin.flags import Flag, FlagSet
from pepin.tsdf import RigidPose
from pepin_bringup.msgs import pose_from_transform, transform_from_pose
from pepin_bringup.node_kit import Switches, spin_main

RATE_HZ = 10.0
KNOWN_MAP_FRAMES = ("map", "rtabmap")  # parent, child
SLAM_FRAMES = ("map", "odom")
CORRECTION_TOPIC = "/map_odom"

FLAGS = FlagSet(
    Flag(
        "slam",
        False,
        live=False,
        description=(
            "RTAB-Map is the map (online SLAM): its correction is map -> odom and goes to the "
            "board as a message on /map_odom, where pepin_bringup.slam_frame broadcasts it; off, "
            "the board's tracker owns map -> odom and this node broadcasts map -> rtabmap here. "
            "Not live: the two modes are two different edges, and a transform once sent stands"
        ),
    ),
)


class RtabmapFrame(Node):
    """Broadcasts map -> rtabmap, or sends map -> odom to the board, from RTAB-Map's correction."""

    def __init__(self) -> None:
        super().__init__("rtabmap_frame")
        self._switches = Switches(self, FLAGS)
        self._slam = self._switches.on("slam")
        self._pose = RigidPose(np.eye(3), np.zeros(3))
        self._graphs = 0
        self._tf = None if self._slam else TransformBroadcaster(self)
        self._correction = (
            self.create_publisher(TransformStamped, CORRECTION_TOPIC, 5) if self._slam else None
        )
        self.create_subscription(MapGraph, "/rtabmap/mapGraph", self._on_graph, 5)
        self.create_timer(1.0 / RATE_HZ, self._publish)
        where = (
            f"map -> odom on {CORRECTION_TOPIC}, for the board"
            if self._slam
            else "map -> rtabmap on /tf, here"
        )
        self.get_logger().info(
            f"rtabmap frame up: {where}, from /rtabmap/mapGraph"
            f" (identity until the first graph); flags: {self._switches.state(live_only=False)}"
        )

    @property
    def frames(self) -> tuple[str, str]:
        """The parent and child of the edge this mode publishes."""
        return SLAM_FRAMES if self._slam else KNOWN_MAP_FRAMES

    def _on_graph(self, msg: MapGraph) -> None:
        """RTAB-Map's correction: map -> odom as it stands in SLAM, its inverse on a known map
        (there the correction reads rtabmap -> map, and ``map`` is the tree's root)."""
        pose = pose_from_transform(msg.map_to_odom)
        self._pose = pose if self._slam else pose.inverse()
        self._graphs += 1

    def _publish(self) -> None:
        """The latest correction, at :data:`RATE_HZ`: a transform here, or a message to the
        board — held between graphs, because a correction only moves when the graph does."""
        stamp = self.get_clock().now().to_msg()
        message = transform_from_pose(*self.frames, self._pose, stamp)
        if self._correction is not None:
            self._correction.publish(message)
        elif self._tf is not None:
            self._tf.sendTransform(message)


def main() -> None:
    spin_main(RtabmapFrame)


if __name__ == "__main__":
    main()
