"""The board's ``map -> odom`` while the laptop is mapping: one owner of the drive's frame.

In online SLAM the map is built on the laptop (RTAB-Map), but ``map -> odom`` is a transform the
BOARD needs: Nav2's global costmap, the behaviour tree and every goal are looked up in ``map``
there, and a lookup over WiFi is not a lookup. ``/tf`` crosses the bridge board -> laptop only —
a topic allowed as a publisher on both sides loops until nothing crosses at all — so RTAB-Map's
correction arrives as a message (``/map_odom``, published by pepin_bringup.rtabmap_frame with
its ``slam`` switch on) and this node broadcasts it here, at :data:`RATE_HZ`.

Two things it does NOT do, both on purpose. It never invents a correction: with no message yet
it broadcasts identity, which is exactly the truth at the start of a session (the map is born at
the cart's first pose). And it re-stamps the correction with the current clock instead of
forwarding the sender's stamp, because a correction is not a measurement: it stands until the
graph moves again, and a transform stamped a second ago is one Nav2's 0.3 s tolerance refuses.
This is the tracker's seat in SLAM mode — the relocalizer does not run, and nothing else may
publish this edge.
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from pepin.tsdf import RigidPose
from pepin_bringup.msgs import pose_from_transform, transform_from_pose
from pepin_bringup.node_kit import spin_main

RATE_HZ = 10.0
FRAMES = ("map", "odom")  # parent, child
CORRECTION_TOPIC = "/map_odom"


class SlamFrame(Node):
    """Broadcasts map -> odom on the board from the laptop's SLAM correction."""

    def __init__(self) -> None:
        super().__init__("slam_frame")
        self._tf = TransformBroadcaster(self)
        self._pose = RigidPose(np.eye(3), np.zeros(3))
        self._corrections = 0
        self.create_subscription(TransformStamped, CORRECTION_TOPIC, self._on_correction, 5)
        self.create_timer(1.0 / RATE_HZ, self._broadcast)
        self.get_logger().info(
            f"slam frame up: map -> odom at {RATE_HZ:.0f} Hz from {CORRECTION_TOPIC}"
            " (identity until the laptop's first graph)"
        )

    def _on_correction(self, msg: TransformStamped) -> None:
        """The laptop's latest map -> odom; held until the next one arrives."""
        self._pose = pose_from_transform(msg)
        self._corrections += 1

    def _broadcast(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self._tf.sendTransform(transform_from_pose(*FRAMES, self._pose, stamp))


def main() -> None:
    spin_main(SlamFrame)


if __name__ == "__main__":
    main()
