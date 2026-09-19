"""The board's ``map -> odom`` from the laptop's correction: the RETIRED owner of the drive's frame.

OFF BY DEFAULT SINCE WORLD R (2026-09-19), and kept whole because CLAUDE.md rule 19 says the old
behaviour must stay reachable: ``ros/nav.launch.py slam:=true`` (the board's ``PEPIN_SLAM``) starts
this node INSTEAD of the tracker, since two publishers of one edge fight. What replaced it is the
tracker owning ``map -> odom`` in every situation — a known room, a room being mapped this minute,
a kidnap, a link that is down — with RTAB-Map's graph speaking to it as a measurement and its grid
arriving as the one map. Turning this on also needs the laptop to publish ``/map_odom`` again
(pepin_bringup.rtabmap_frame's own ``slam`` switch); the bridge keeps a route for it.

The argument it was built on still stands where it is used: ``map -> odom`` is a transform the BOARD
needs — Nav2's global costmap, the behaviour tree and every goal are looked up in ``map`` there, and
a lookup over WiFi is not a lookup — while ``/tf`` crosses the bridge board -> laptop only, since a
topic allowed as a publisher on both sides loops until nothing crosses at all. So the correction
arrives as a message (``/map_odom``) and this node broadcasts it here, at :data:`RATE_HZ`.

Two things it does NOT do, both on purpose. It never invents a correction: with no message yet
it broadcasts identity, which is exactly the truth at the start of a session (the map is born at
the cart's first pose) and is what lets the board's Nav2 come up before the laptop's half does.
And it re-stamps the correction with the current clock instead of forwarding the sender's stamp,
because a correction is not a measurement: it stands until the graph moves again, and a
transform stamped a second ago is one Nav2's 0.3 s tolerance refuses. While it runs it holds the
tracker's seat — the relocalizer does not run, and nothing else may publish this edge.

The price of that re-stamping is that the edge is no evidence at all about the laptop: it stays
milliseconds old with the laptop shut down, because the last correction is re-broadcast for
ever. So this node says in the log when the stream stops and when it returns, and what a goal
is judged on is the correction's own age (pepin.watch.Correction, read by the goal server).
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from pepin.tsdf import RigidPose
from pepin.watch import CORRECTION_FRESH_S
from pepin_bringup.msgs import pose_from_transform, transform_from_pose
from pepin_bringup.node_kit import spin_main

RATE_HZ = 10.0
FRAMES = ("map", "odom")  # parent, child
CORRECTION_TOPIC = "/map_odom"
SILENCE_S = CORRECTION_FRESH_S  # the correction's pulse is 10 Hz: this much is not a hiccup


class SlamFrame(Node):
    """Broadcasts map -> odom on the board from the laptop's SLAM correction."""

    def __init__(self) -> None:
        super().__init__("slam_frame")
        self._tf = TransformBroadcaster(self)
        self._pose = RigidPose(np.eye(3), np.zeros(3))
        self._corrections = 0
        self._heard_at = self._seconds()  # the start counts as the last thing heard from
        self._silent = False
        self.create_subscription(TransformStamped, CORRECTION_TOPIC, self._on_correction, 5)
        self.create_timer(1.0 / RATE_HZ, self._broadcast)
        self.get_logger().info(
            f"slam frame up: map -> odom at {RATE_HZ:.0f} Hz from {CORRECTION_TOPIC}"
            " (identity until the laptop's first graph)"
        )

    def _seconds(self) -> float:
        """The node's clock in seconds."""
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _on_correction(self, msg: TransformStamped) -> None:
        """The laptop's latest map -> odom; held until the next one arrives."""
        self._pose = pose_from_transform(msg)
        self._corrections += 1
        self._heard_at = self._seconds()
        if self._silent:
            self._silent = False
            self.get_logger().info(
                f"the correction is back on {CORRECTION_TOPIC} ({self._corrections} so far)"
            )

    def _broadcast(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self._tf.sendTransform(transform_from_pose(*FRAMES, self._pose, stamp))
        self._report_silence(self._seconds())

    def _report_silence(self, now: float) -> None:
        """Say it once when the correction stops coming, because nothing else here would show
        it: the edge goes on being broadcast from the last one, at the same rate, with a fresh
        stamp. The broadcast is not stopped — Nav2 on this board would lose its global frame
        for a wireless hiccup, and the drive is cut where the decision belongs (the goal
        server's ``correction_watch``)."""
        silence = now - self._heard_at
        if self._silent or silence <= SILENCE_S:
            return
        self._silent = True
        heard = f"{self._corrections} heard" if self._corrections else "never heard one"
        self.get_logger().warning(
            f"nothing on {CORRECTION_TOPIC} for {silence:.1f} s ({heard}): map -> odom is still"
            " broadcast from the last correction, but the laptop's SLAM half is not feeding it"
        )


def main() -> None:
    spin_main(SlamFrame)


if __name__ == "__main__":
    main()
