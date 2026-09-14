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

import math

import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from rclpy.node import Node
from rtabmap_msgs.msg import MapGraph
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from pepin.flags import Flag, FlagSet
from pepin.measurements import graph_measurement
from pepin.odometry import Pose2D
from pepin.tsdf import RigidPose
from pepin_bringup.msgs import pose_from_transform, stamp_seconds, transform_from_pose, yaw_of
from pepin_bringup.node_kit import Switches, spin_main

RATE_HZ = 10.0
KNOWN_MAP_FRAMES = ("map", "rtabmap")  # parent, child
SLAM_FRAMES = ("map", "odom")
CORRECTION_TOPIC = "/map_odom"
# The graph's word for the board's fusion. A topic of its own, NOT the camera's
# /localization/measurement: pepin.measurements.MeasurementGate fuses everything waiting in it
# into one word named "camera", so a graph measurement dropped in there would move the pose under
# the camera's name. The board gains a gate of its own for this one, named "graph", the day its
# roster does.
MEASUREMENT_TOPIC = "/localization/graph_measurement"
TRACKER_POSE_TOPIC = "/tracker_pose"
# How far the correction may move the cart before the word is refused: an accepted closure on a
# flat of this size shifts the pose by centimetres to a few tens of them, and anything past this
# is the graph having blown up rather than found a place.
MAX_CORRECTION_M = 1.5

FLAGS = FlagSet(
    Flag(
        "slam",
        False,
        description="RTAB-Map is the map (online SLAM): its correction is map -> odom and goes to"
        " the board as a message on /map_odom, where pepin_bringup.slam_frame broadcasts it; off,"
        " the board's tracker owns map -> odom and this node broadcasts map -> rtabmap here",
        why="default by design, unmeasured: this says which edge is published — a mode, not a"
        " tunable — and the two modes are two different graphs of frames, which is also why it is"
        " not live. What the mode is worth was measured in the first session: from an empty"
        " database a room came up as a 341x341 map over 21 and then 55 graph nodes, a 1 m goal"
        " with a 90 degree turn landed within 2.8 cm and home within 6.6 cm after about 4 m of"
        " driving, one loop-closure hypothesis was rejected by the scan check (5 % against the 10"
        " % it needs) and none was accepted",
        on_when="in an unknown room, launched as one mode end to end (ros/thin.sh slam on the"
        " board, ros/laptop.sh vslam --slam): set at start, never mid-run",
        off_when="in every known-map mode, where the board's tracker owns map -> odom: the two"
        " publishers must never both run",
        live=False,
    ),
    Flag(
        "graph_measurement",
        False,
        description="beside a known map, publish RTAB-Map's correction of the tracker's own pose"
        f' as a measurement on {MEASUREMENT_TOPIC} (source "graph") every time the graph moves,'
        " for the board's fusion to weigh like any other word; off, the correction only moves"
        " this laptop's map -> rtabmap and nothing reaches the pose",
        why="OFF, because on the stack as it stands the correction never moves at all: over 3 h"
        " on 2026-09-14 every closure RTAB-Map found was thrown away by RGBD/OptimizeMaxError"
        " (5 links an iteration, rejected on a NEIGHBOUR edge 28042->28043 whose residual is"
        " 0.888 m against a 0.244 m sigma, ratio 3.64 over the 3.0 the parameter allows), so"
        " there is not yet one accepted correction to judge this word on. It is also OFF because"
        ' nothing on the board subscribes yet: the roster has no "graph" source and the'
        " measurement gate fuses by name",
        on_when="once a closure is accepted AND the board has a gate for it: then on, with"
        " /localization/graph_measurement recorded beside /tracker_pose for a drive, to see what"
        " the graph would have done to the pose before it is allowed to do it",
        off_when="whenever the graph's own frame may have started anywhere but the tracker's"
        " truth -- a session begun while the cart was lost puts every graph word off by that"
        " offset, since the two frames are only tied at the start pose",
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
        self._correction2d = Pose2D()  # the same correction in the plane, as the fusion reads it
        self._belief: Pose2D | None = None  # what the board's tracker says, and when
        self._belief_stamp = 0.0
        self._map_id = ""  # the map that belief is on; the board refuses a word about another
        self._sent = 0  # graph measurements published
        self._refused = 0  # ...and corrections too large to be a closure
        self._tf = None if self._slam else TransformBroadcaster(self)
        self._correction = (
            self.create_publisher(TransformStamped, CORRECTION_TOPIC, 5) if self._slam else None
        )
        self._measurement = (
            None if self._slam else self.create_publisher(String, MEASUREMENT_TOPIC, 5)
        )
        self.create_subscription(MapGraph, "/rtabmap/mapGraph", self._on_graph, 5)
        self.create_subscription(
            PoseWithCovarianceStamped, TRACKER_POSE_TOPIC, self._on_tracker_pose, 5
        )
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
        self.create_timer(30.0, self._report)

    def _report(self) -> None:
        """Every 30 s: how many graph corrections arrived, how many became measurements and how
        many were too large to be a closure, with the switches."""
        self.get_logger().info(
            f"rtabmap frame: {self._graphs} graphs, {self._sent} graph measurements sent,"
            f" {self._refused} refused as too large; correction"
            f" ({self._correction2d.x:+.2f}, {self._correction2d.y:+.2f},"
            f" {math.degrees(self._correction2d.theta):+.1f} deg);"
            f" flags: {self._switches.state(live_only=False)}"
        )

    def _on_tracker_pose(self, msg: PoseWithCovarianceStamped) -> None:
        """The board's belief: the pose RTAB-Map is fed as its odometry, and the one a graph
        correction is applied to."""
        position = msg.pose.pose.position
        self._belief = Pose2D(position.x, position.y, yaw_of(msg.pose.pose.orientation))
        self._belief_stamp = stamp_seconds(msg.header.stamp)
        self._map_id = msg.header.frame_id

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
        self._correction2d = Pose2D(
            float(pose.translation[0]),
            float(pose.translation[1]),
            math.atan2(float(pose.rotation[1, 0]), float(pose.rotation[0, 0])),
        )
        self._publish_measurement()

    def _publish_measurement(self) -> None:
        """The graph's answer about where the cart is, on the tracker's own map, as one
        measurement -- the correction applied to the belief RTAB-Map was fed. Sent only beside a
        known map, only with the flag on, only with a belief to correct, and only while the
        correction is small enough to be a closure rather than a graph that has blown up."""
        if self._measurement is None or not self._switches.on("graph_measurement"):
            return
        if self._belief is None or not self._map_id:
            return
        if math.hypot(self._correction2d.x, self._correction2d.y) > MAX_CORRECTION_M:
            self._refused += 1
            return
        remote = graph_measurement(
            self._belief, self._correction2d, self._belief_stamp, self._map_id
        )
        self._sent += 1
        self._measurement.publish(String(data=remote.to_json(graphs=self._graphs)))

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
