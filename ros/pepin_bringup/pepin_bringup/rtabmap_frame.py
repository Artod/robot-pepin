"""Put RTAB-Map's graph correction where the mode needs it: a frame here, or a message home.

RTAB-Map computes one correction per graph optimisation and publishes it as ``map_to_odom`` on
``/rtabmap/mapGraph``: the jump its "odometry" frame takes when the graph moves under it. Where
that correction belongs depends on which map the robot drives.

On a KNOWN map the board's tracker owns ``map -> odom`` and RTAB-Map runs on the EKF's own
``odom -> base_link`` (vslam.launch.py's ``graph_odom``, the default). Its graph therefore lives
in a frame of its own that starts at the ODOM frame's origin, and this node ties that frame to
the lidar map: at the first graph it reads where the tracker says the cart is and where the
graph has it, and the difference between the two is the ANCHOR — ``map <- rtabmap``, broadcast
at 10 Hz and constant for the session. RTAB-Map cannot be told that pose itself ("Initial pose
can only be set in localization mode (Mem/IncrementalMemory=false), ignoring it",
librtabmap_core 0.22.1), so the offset is learned here instead of imposed there.

With the anchor known, the graph's answer about the cart is a measurement on the lidar map:
``anchor . map_to_odom . (odom -> base_link at the graph's stamp)`` — the graph's own opinion,
built from its own odometry and every closure it has accepted — published on
:data:`MEASUREMENT_TOPIC` for the board's fusion to weigh like any other word
(``graph_measurement``, off until measured). Nothing here owns ``map -> odom``.

With ``graph_odom`` off the old arrangement is back: RTAB-Map's "odometry" IS the tracker's
pose, ``map -> rtabmap`` is the inverse of the correction (a fixed identity in its place let
the voxels drift away from the cart after every closure — the cloud moved with the graph, the
cart did not), and the measurement is the tracker's belief moved by that correction.

THE ANCHOR IS A PROPERTY OF THE PAIR (this lidar map, this graph database), not of a session, so
it is kept where the pair can find it: ``<map id>.graph_anchor.json`` beside the map the board
serves (:mod:`pepin.anchors`, the node's ``anchor_dir``). At start the file for the served map is
read and its anchor adopted; with no file the anchor is learned from the tracker as before and
written. It is re-learned — and the file rewritten — only on evidence that holds: the lidar
driving with a fit at or above :data:`TRUSTED_FIT` and its belief disagreeing with the graph's
word by more than half a metre or 20 degrees for five seconds (``anchor_relearn``, on). With the
lidar silent nothing is re-learned: there is nothing to re-learn FROM.

THE WAKE-UP is what the file buys. The cart sleeps on the charger, the board restarts, the odom
frame resets — and the tracker's saved pose is the only thing anyone knows. With the anchor on
file, the FIRST graph message already reads as a place on the map, before the tracker has said
anything at all, and the word goes out with no belief to check it against (a word is only checked
against a belief that exists). If the graph does NOT recognise the place — an unseen corner, the
camera blind — no word is published at all: the tracker keeps its saved pose, and the owner
carries the cart to a place it knows, or to the charger.

ON A RESTART the database is kept, and RTAB-Map opens a new session whose nodes are placed at
the odometry's poses again: with the board's EKF still running, the odom frame is the one the
previous session was built in and the anchor on file lands on the same relation. A BOARD restart
resets the odom frame, so the sessions no longer share one; the stored anchor is then stale by
exactly that reset, which is what ``anchor_relearn`` is for — until it fires, a word too far from
the tracker's own belief is refused by :data:`MAX_DISAGREEMENT_M` rather than fused.

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
from pathlib import Path

import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rtabmap_msgs.msg import MapGraph
from std_msgs.msg import Float32, String
from tf2_ros import TransformBroadcaster

from pepin.anchors import Anchor, AnchorWatch, load_anchor, save_anchor
from pepin.flags import Flag, FlagSet
from pepin.measurements import compose, graph_anchor, graph_measurement
from pepin.odometry import Pose2D
from pepin.tsdf import RigidPose
from pepin_bringup.msgs import (
    map_id,
    pose_from_transform,
    stamp_seconds,
    transform_from_pose,
    transform_from_rpy,
    yaw_of,
)
from pepin_bringup.node_kit import Switches, TfLookup, bridged_qos_profile, spin_main

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
FIT_TOPIC = "/localization_fit"
ODOM_FRAME, BASE_FRAME = "odom", "base_link"
# Where the anchors live, as the container sees them: the same /maps the map yaml and the graph
# database are served from, so the pair's anchor travels with the pair.
ANCHOR_DIR = "/maps"
# What makes the tracker's belief worth re-learning the anchor from: a lidar scan matched this
# well, this recently. /localization_fit carries only a fit a scan measured (relocalizer.py) and
# falls to 0.0 when no source has spoken, so a fresh 0.6 IS "the lidar is driving"; depth_fusion
# paints its band at 0.50 and this is the stricter half of that.
TRUSTED_FIT = 0.6
FIT_FRESH_S = 2.0
# How far from the tracker's own belief the graph may put the cart before the word is refused:
# an accepted closure on a flat of this size moves the pose by centimetres to a few tens of
# them, and anything past this is the graph having blown up, or an anchor that no longer holds
# (two database sessions merged by a closure), rather than a place found.
MAX_DISAGREEMENT_M = 1.5
# How far apart in time the tracker's belief and the graph's own place may be when the anchor
# between the two frames is learned from them: at the cart's 0.3 m/s half a second is 15 cm of
# frame error, baked in for the whole session, and the tracker publishes at 10 Hz.
ANCHOR_MAX_SKEW_S = 0.5

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
        "graph_odom",
        True,
        description="beside a known map, RTAB-Map is built on the EKF's odom -> base_link and"
        " this node ties its graph to the lidar map with an anchor learned once at the first"
        " graph (map -> rtabmap, constant); off, RTAB-Map's odometry is the tracker's own pose"
        " and map -> rtabmap carries the inverse of the correction, as before 2026-09-14."
        " Set by vslam.launch.py's argument of the same name: the two are one decision",
        why="ON, because the tracker's pose teleports when it relocalises and a graph cannot be"
        " built on odometry that jumps: a neighbour edge between two nodes one second apart"
        " carried 0.888 m against its 0.244 m sigma, a 3.64 error ratio over the 3.0 of"
        " RGBD/OptimizeMaxError, and on it RTAB-Map rejected EVERY loop closure it found over"
        " three hours on 2026-09-14 (5 at a time, each registered with 67 visual inliers against"
        " a Vis/MinInliers of 20). The EKF's odometry is continuous by construction and is what"
        " every other consumer already rides",
        on_when="always, beside a known map: this is what makes a closure possible at all",
        off_when="to reproduce the old graph, or to read a database recorded under it: the"
        " nodes of the two arrangements are in different frames, so the switch is a launch"
        " argument and never moves mid-session",
        live=False,
    ),
    Flag(
        "graph_measurement",
        True,
        description="beside a known map, publish where RTAB-Map's graph says the cart is as a"
        f' measurement on {MEASUREMENT_TOPIC} (source "graph") every time the graph moves, for'
        " the board's fusion to weigh like any other word; off, the graph's answer stays on this"
        " laptop and nothing reaches the pose",
        why="on since 2026-09-14 16:20: with the lidar driving (sources=lidar,graph, tapes"
        " 0275/0276)"
        " the tracker took 5 of 27 words and sat 0.7-0.8 cm from the lidar truth, and at rest the"
        " word"
        " stays 0-8 cm from the tracker; the word is what a lidar-less cart localises on (test C)."
        " Before: OFF, because on the stack as it stands the correction never moves at all: over 3"
        " h"
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
    Flag(
        "anchor_relearn",
        True,
        description="re-learn the stored anchor (and rewrite its file) when the lidar is driving"
        f" with a fit of at least {TRUSTED_FIT:.1f} and its belief disagrees with the graph's word"
        " by more than half a metre or 20 degrees for five seconds; off, the anchor read from the"
        " file (or learned at the first graph) is kept whatever happens and a disagreeing word is"
        " simply refused",
        why="on, because the anchor ties the graph's frame to the ODOM frame's origin and a board"
        " restart moves that origin: without this the file would be a lie from the first restart"
        " and every word after it refused by the 1.5 m disagreement gate. The evidence is"
        " deliberately narrow because a wrongly re-learned anchor is a cart confidently in the"
        " wrong room, while a stale one only costs the graph's vote: 0.5 m or 20 deg is past"
        " anything a closure accounts for (the word sat 2.2-3.2 cm from the lidar truth over a"
        " printer errand, 2026-09-14), the 5 s hold is what tells a closure landing from a frame"
        f" that has moved, and a fit of {TRUSTED_FIT:.1f} within {FIT_FRESH_S:.0f} s is what says"
        " the lidar, not dead reckoning, is behind the belief being learned from",
        on_when="always, on a known map: it is what keeps the stored anchor honest across a board"
        " restart",
        off_when="while reading a database recorded in another odom frame on purpose, or in any"
        " session where the file must not change (a measurement of what the stored anchor is"
        " worth)",
    ),
)


class RtabmapFrame(Node):
    """Broadcasts map -> rtabmap, or sends map -> odom to the board, from RTAB-Map's correction."""

    def __init__(self) -> None:
        super().__init__("rtabmap_frame")
        # Declared before the switches: rclpy runs the flags' callback on every declaration.
        self._anchor_dir = Path(str(self.declare_parameter("anchor_dir", ANCHOR_DIR).value))
        self._switches = Switches(self, FLAGS)
        self._slam = self._switches.on("slam")
        self._graph_odom = self._switches.on("graph_odom")
        self._pose = RigidPose(np.eye(3), np.zeros(3))
        self._graphs = 0
        self._correction2d = Pose2D()  # the same correction in the plane, as the fusion reads it
        self._belief: Pose2D | None = None  # what the board's tracker says, and when
        self._belief_stamp = 0.0
        self._map_id = ""  # the map that belief is on; the board refuses a word about another
        self._anchor: Pose2D | None = None  # map <- rtabmap: read from the file, or learned once
        self._origin = "none"  # ...and where this copy of it came from, for the report line
        self._relearns = 0  # ...and how many times it has been re-learned since the node came up
        self._watch = AnchorWatch()  # what says the stored anchor no longer holds
        self._fit, self._fit_at = 0.0, -math.inf  # the lidar's own fit, and when it last spoke
        self._word: Pose2D | None = None  # the last place the graph put the cart, on the map
        self._gap_m = 0.0  # ...and how far that was from the tracker's own belief
        self._sent = 0  # graph measurements published
        self._refused = 0  # ...and places too far from the tracker's to be a closure
        self._blind = 0  # ...and graphs with no odom -> base_link to compose with
        self._lookup = TfLookup(self)  # the EKF's odom -> base_link, at the graph's own stamp
        self._tf = None if self._slam else TransformBroadcaster(self)
        self._correction = (
            self.create_publisher(TransformStamped, CORRECTION_TOPIC, 5) if self._slam else None
        )
        self._measurement = (
            None
            if self._slam
            else self.create_publisher(
                String, MEASUREMENT_TOPIC, bridged_qos_profile(MEASUREMENT_TOPIC)
            )
        )
        self.create_subscription(MapGraph, "/rtabmap/mapGraph", self._on_graph, 5)
        # The map the word is about, named the way the board names it (size@origin, msgs.map_id):
        # the tracker refuses a word about another map, and a frame id is not a map id.
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGridMsg, "/map", self._on_map, latched)
        self.create_subscription(
            PoseWithCovarianceStamped, TRACKER_POSE_TOPIC, self._on_tracker_pose, 5
        )
        self.create_subscription(Float32, FIT_TOPIC, self._on_fit, 5)
        self.create_timer(1.0 / RATE_HZ, self._publish)
        where = (
            f"map -> odom on {CORRECTION_TOPIC}, for the board"
            if self._slam
            else "map -> rtabmap on /tf, here"
            + (
                " (the anchor, once the first graph and the tracker meet)"
                if self._graph_odom
                else ""
            )
        )
        self.get_logger().info(
            f"rtabmap frame up: {where}, from /rtabmap/mapGraph"
            f" (identity until the first graph); flags: {self._switches.state(live_only=False)}"
        )
        self.create_timer(30.0, self._report)

    def _report(self) -> None:
        """Every 30 s: how many graphs arrived, how many became measurements, how many were
        refused for putting the cart too far from the tracker's belief and how many found no
        odometry to compose with, with the anchor and where it came from (the file beside the map,
        learned here at the first graph, or re-learned N times since), the last word's gap to the
        tracker and the switches."""
        anchor = (
            "not yet"
            if self._anchor is None
            else f"({self._anchor.x:+.2f}, {self._anchor.y:+.2f},"
            f" {math.degrees(self._anchor.theta):+.1f} deg) from {self._origin}"
            + (f" {self._relearns}" if self._relearns else "")
        )
        word = (
            "none yet"
            if self._word is None
            else f"({self._word.x:+.2f}, {self._word.y:+.2f},"
            f" {math.degrees(self._word.theta):+.1f} deg), {self._gap_m * 100:.0f} cm from"
            " the tracker"
        )
        self.get_logger().info(
            f"rtabmap frame: {self._graphs} graphs, {self._sent} graph measurements sent,"
            f" {self._refused} refused as too far, {self._blind} without odometry;"
            f" correction ({self._correction2d.x:+.2f}, {self._correction2d.y:+.2f},"
            f" {math.degrees(self._correction2d.theta):+.1f} deg), anchor {anchor},"
            f" last word {word};"
            f" flags: {self._switches.state(live_only=False)}"
        )

    def _on_tracker_pose(self, msg: PoseWithCovarianceStamped) -> None:
        """The board's belief: the pose RTAB-Map is fed as its odometry, and the one a graph
        correction is applied to."""
        position = msg.pose.pose.position
        self._belief = Pose2D(position.x, position.y, yaw_of(msg.pose.pose.orientation))
        self._belief_stamp = stamp_seconds(msg.header.stamp)

    def _on_fit(self, msg: Float32) -> None:
        """How well the lidar's last scan matched the map, and when: what says the tracker's
        belief is worth re-learning the anchor from."""
        self._fit, self._fit_at = float(msg.data), self._now()

    def _on_map(self, msg: OccupancyGridMsg) -> None:
        """The served map's identity (size@origin), the name the board checks a word against —
        and the name of the anchor file this pair keeps, adopted the moment the map is known."""
        served = map_id(msg)
        if served == self._map_id:
            return
        self._map_id = served
        if self._anchor is None and self._graph_odom and not self._slam:
            self._adopt(served)

    def _now(self) -> float:
        """This node's clock in seconds: when a fit arrived, how long a disagreement has held."""
        return stamp_seconds(self.get_clock().now().to_msg())

    def _adopt(self, served: str) -> None:
        """Take the anchor stored for the served map, if there is one worth believing: this is
        the wake-up path — with it the first graph already reads as a place on the map, before
        the tracker has said anything at all."""
        try:
            stored = load_anchor(self._anchor_dir, served)
        except (OSError, ValueError) as error:
            self.get_logger().warning(f"stored anchor ignored: {error}")
            return
        if stored is None:
            self.get_logger().info(
                f"no anchor stored for map {served} in {self._anchor_dir}: it will be learned"
                " from the tracker at the first graph, and written there"
            )
            return
        self._anchor, self._origin = stored.pose, "file"
        self.get_logger().info(f"graph anchor read from file: {stored.described()}, map {served}")

    def _keep(self, anchor: Pose2D, origin: str) -> None:
        """Remember an anchor learned here and write it beside its map, so the next session —
        and the next wake-up — starts from it. Nothing is written without a map to name it."""
        self._anchor, self._origin = anchor, origin
        if not self._map_id:
            return
        record = Anchor(anchor, self._map_id, self._now(), origin, self._relearns)
        try:
            path = save_anchor(self._anchor_dir, record)
        except OSError as error:
            self.get_logger().warning(f"anchor not stored: {error}")
            return
        self.get_logger().info(f"graph anchor stored in {path}: {record.described()}")

    @property
    def frames(self) -> tuple[str, str]:
        """The parent and child of the edge this mode publishes."""
        return SLAM_FRAMES if self._slam else KNOWN_MAP_FRAMES

    def _on_graph(self, msg: MapGraph) -> None:
        """RTAB-Map's correction, and what follows from it: in SLAM it IS ``map -> odom``; on a
        known map it says where the graph believes the odom frame sits in the graph's own frame,
        which with the EKF's ``odom -> base_link`` gives the cart's place in that frame — the
        anchor is learned from it once and the measurement composed from it every time."""
        pose = pose_from_transform(msg.map_to_odom)
        self._graphs += 1
        self._correction2d = Pose2D(
            float(pose.translation[0]),
            float(pose.translation[1]),
            math.atan2(float(pose.rotation[1, 0]), float(pose.rotation[0, 0])),
        )
        if self._slam:
            self._pose = pose
            return
        if not self._graph_odom:  # the old arrangement: the correction IS the frame
            self._pose = pose.inverse()
            if self._belief is not None:
                self._offer(self._belief, self._correction2d, self._belief_stamp)
            return
        cart = self._cart_in_graph(msg)
        if cart is None:
            self._blind += 1
            return
        place, stamp = cart
        anchor = self._anchor
        if anchor is None:
            anchor = self._learn_anchor(place, stamp)
            if anchor is None:
                return
            self._keep(anchor, "learned")
        elif self._stale(anchor, place, stamp):
            relearned = self._learn_anchor(place, stamp)
            if relearned is not None:
                self._relearns += 1
                anchor = relearned
                self._keep(anchor, "relearned")
        self._offer(place, anchor, stamp)

    def _stale(self, anchor: Pose2D, place: Pose2D, stamp: float) -> bool:
        """Whether the anchor in hand no longer holds: the word it makes of this graph and the
        tracker's belief of the same instant, disagreeing past what a closure accounts for, for
        longer than :data:`pepin.anchors.RELEARN_HOLD_S`, while the LIDAR is what the tracker is
        believing (a fresh :data:`TRUSTED_FIT`). Every other case feeds the watch a "not
        trusted", which is also what resets its clock."""
        now = self._now()
        belief = self._belief
        trusted = (
            self._switches.on("anchor_relearn")
            and belief is not None
            and self._fit >= TRUSTED_FIT
            and now - self._fit_at <= FIT_FRESH_S
            and abs(stamp - self._belief_stamp) <= ANCHOR_MAX_SKEW_S
        )
        if belief is None:
            return self._watch.update(now, False, 0.0, 0.0)
        word = compose(anchor, place)
        turn = math.atan2(math.sin(word.theta - belief.theta), math.cos(word.theta - belief.theta))
        gap_m = math.hypot(word.x - belief.x, word.y - belief.y)
        if self._watch.update(now, trusted, gap_m, math.degrees(turn)):
            self.get_logger().warning(
                f"graph anchor stale: the word sits {gap_m * 100:.0f} cm and"
                f" {math.degrees(turn):+.1f} deg from a tracker driving on the lidar"
                f" (fit {self._fit:.2f}) — re-learning it"
            )
            return True
        return False

    def _cart_in_graph(self, msg: MapGraph) -> tuple[Pose2D, float] | None:
        """Where the graph has the cart, in the graph's own frame, and at which moment: the
        correction composed with the EKF's ``odom -> base_link`` at the graph's stamp (the
        newest transform when that one is already out of the buffer). ``None`` when the
        odometry is not there at all — a silent bridge."""
        transform = self._lookup.transform(ODOM_FRAME, BASE_FRAME, msg.header.stamp)
        transform = (
            transform if transform is not None else self._lookup.transform(ODOM_FRAME, BASE_FRAME)
        )
        if transform is None:
            return None
        odom = pose_from_transform(transform)
        planar = Pose2D(
            float(odom.translation[0]),
            float(odom.translation[1]),
            math.atan2(float(odom.rotation[1, 0]), float(odom.rotation[0, 0])),
        )
        return compose(self._correction2d, planar), stamp_seconds(transform.header.stamp)

    def _learn_anchor(self, place: Pose2D, stamp: float) -> Pose2D | None:
        """The session's anchor (``map <- rtabmap``) from the tracker's belief and the graph's
        own place for the same cart, or ``None`` while there is no belief close enough in time
        to learn it from. Learned once and never again: it is the frame the graph is read in."""
        if self._belief is None or abs(stamp - self._belief_stamp) > ANCHOR_MAX_SKEW_S:
            return None
        anchor = graph_anchor(self._belief, place)
        self.get_logger().info(
            f"graph anchored: map <- rtabmap = ({anchor.x:+.2f}, {anchor.y:+.2f},"
            f" {math.degrees(anchor.theta):+.1f} deg), from the tracker at"
            f" ({self._belief.x:+.2f}, {self._belief.y:+.2f}) and the graph at"
            f" ({place.x:+.2f}, {place.y:+.2f}) on map {self._map_id}"
        )
        return anchor

    def _offer(self, place: Pose2D, frame: Pose2D, stamp: float) -> None:
        """One graph word: ``place`` (where the graph has the cart) read on the tracker's own
        map through ``frame`` (the anchor, or the correction itself with ``graph_odom`` off),
        published as a measurement for the board's fusion.

        The word is remembered whatever the flag says — the report line is how a session is
        judged before it is allowed to move anything — and sent only with the flag on, a map to
        name, and a place no further from the tracker's belief than :data:`MAX_DISAGREEMENT_M`:
        past that the graph has blown up or the anchor no longer holds.

        With NO belief at all the word still goes out, but only on an anchor read from the file:
        that is the wake-up (a lidar-less start, nothing on ``/tracker_pose`` yet), and it is the
        one case where the graph is the only thing that knows where the cart is. An anchor
        learned in this session cannot produce that word — it was learned FROM a belief."""
        remote = graph_measurement(place, frame, stamp, self._map_id)
        belief = self._belief
        self._word = remote.pose
        self._gap_m = (
            math.hypot(remote.x - belief.x, remote.y - belief.y) if belief is not None else math.inf
        )
        if self._measurement is None or not self._switches.on("graph_measurement"):
            return
        if not self._map_id:
            return
        if belief is None:
            if self._origin != "file":
                return
        elif self._gap_m > MAX_DISAGREEMENT_M:
            self._refused += 1
            return
        self._sent += 1
        self._measurement.publish(String(data=remote.to_json(graphs=self._graphs)))

    def _publish(self) -> None:
        """The edge this mode owns, at :data:`RATE_HZ`: the anchor (identity until it is
        learned) on a known map, the correction as a message to the board in SLAM, and the
        correction's inverse with ``graph_odom`` off — each held between graphs, because none
        of them moves until the graph does."""
        stamp = self.get_clock().now().to_msg()
        parent, child = self.frames
        message = (
            transform_from_rpy(
                parent,
                child,
                (self._anchor.x, self._anchor.y, 0.0),
                (0.0, 0.0, self._anchor.theta),
                stamp,
            )
            if self._anchor is not None and self._graph_odom and not self._slam
            else transform_from_pose(parent, child, self._pose, stamp)
        )
        if self._correction is not None:
            self._correction.publish(message)
        elif self._tf is not None:
            self._tf.sendTransform(message)


def main() -> None:
    spin_main(RtabmapFrame)


if __name__ == "__main__":
    main()
