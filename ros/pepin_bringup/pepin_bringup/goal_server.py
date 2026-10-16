"""The robot waits for orders instead of being started for each one.

Every goal used to boot its own client on the board: ssh, docker exec, import rclpy, import the
Nav2 commander, build a node, discover the action server — 8 to 15 seconds before the wheels
could move, paid again for every command (measured 2026-09-08). This node is already running,
already connected to Nav2 and already holding the places book, so a command costs a socket write.

It speaks JSON lines on a TCP port (like the base and ToF servers), one connection at a time:

    {"cmd": "go", "place": "printer"}      {"cmd": "go", "x": -11.4, "y": 0.8, "yaw_deg": 140}
    {"cmd": "cancel"}                      {"cmd": "where"}
    {"cmd": "places"}                      {"cmd": "planner", "name": "navfn"}
    {"cmd": "clearance", "radius_m": 1.0}  (where's ``navigating``, and depth_fusion's newest
                                           obstacle marks near the cart, for the voice's wheels)
    {"cmd": "plan", "x": 1.2, "y": 0.4, "yaw_deg": 90}
    {"cmd": "spin", "deg": -90}            (a turn in place, + left: the behaviour server's Spin)
    {"cmd": "pose_at", "stamp": 1760470123.456, "frame": "camera_link"}
                                           (map -> frame at a stamp, out of TF's history)
    {"cmd": "pose_at", "stamps": [1760470123.456, null], "frames": ["camera_link", "base_link"]}
                                           (every frame at every stamp: ``poses``)

and answers with one JSON line per event: accepted, feedback, arrival (the moment Nav2 reports
the goal reached; the end-of-drive pivot and the tape's close come after it), done. A ``plan``
is asked, never driven: would a goal there be driven, and where does its plan end
(:meth:`GoalServer.plan`)? It also owns
the run's recording: it starts one when a goal starts and closes it when the goal ends, so a
recording can no longer outlive its run.

ONE OWNER OF MOTION, ONE RULE FOR A NEW REQUEST (2026-10-14). A ``go`` or a ``spin`` carries
its ``source``: ``tool`` (a person's ask: the brain's tools, ros/goto.sh; the default) or
``gaze`` (the speaker gaze's own body turn). One run at a time is this node's, and a request
that comes while one is under way, until its ``done`` (the pivot and the tape's close
included), is decided here (:meth:`GoalServer._admit`): the same source asking for the same
target is answered ``already`` with that run's number, no new goal (its end goes to its own
requester); the gaze's turn never cancels a person's: refused while one runs; anything else
cancels the run under way (its goal and its pivot, by their handles) and starts once that run
has ended, ``replaced`` naming it, while the cancelled run's ``done`` names what it was
cancelled for (``cancelled_for``). Nav2 has no such rule across its servers: the BT navigator
preempts a NavigateToPose with the newest one (the old goal ABORTED), the behaviour server
aborts a running Spin for a new one, and nothing arbitrates a Spin against a NavigateToPose,
whose tree sends Spins of its own as recoveries; nor does any of them know "the same goal".

A CANCEL MEANS EVERY GOAL ON THE BOARD, not only this node's own (since 2026-09-25): both
navigators' cancel services are asked with a zero goal id, exactly as ``goto_ros.py cancel``
asks them, so a goal any client sent is stopped from here too — by a socket write
(pepin.goal_link) instead of a fresh ROS process, whose new zenoh session stalls every
laptop -> board stream for about three seconds.

WHERE THE POSE COMES FROM. The laptop's RTAB-Map owns ``map -> odom`` (one localiser,
2026-09-22), so this node reads ``map -> base_link`` from TF and judges a goal by how fresh that
edge is (:class:`pepin.watch.GoalGate`) and by whether this start of RTAB-Map is placed at all
(``start_needs_placement``). ``where`` answers ``"pose": "tf"``. The board tracker that answered
``/where_am_i`` with a fit before is on the tag alt/tracker-2026-09-22.

WHERE A FRAME WAS AT A STAMP (``pose_at``, 2026-10-14). The same buffer is the one history of
the poses on this Mac: RTAB-Map's ``map -> odom``, the EKF's ``odom -> base_link`` and the board's
``base_link -> camera_link`` from the neck's encoders, each under its measurement's own stamp. A
picture is placed with the camera where it was when the picture was exposed, never where it is
when someone asks: on 2026-10-14 a face seen during the gaze's own turn was placed with the pose of
after it and the cart drove 4.9 m into the kitchen. ``pose_at`` answers ``map -> frame`` at a
stamp (:meth:`GoalServer.pose_at`); without a stamp, the newest there is, as ``where`` reads it.

AND BECAUSE IT ALREADY PARSES ``/tf``, IT IS THE BOARD'S ONE LISTENER. A TF listener is a
subscription to the whole stream — RTAB-Map's ``map -> odom`` at 20 Hz, the board's
``odom -> base_link`` at 50 Hz, the statics — deserialised in Python whatever one pose the reader
wanted out of it, and two of them ran on a 4-core A53: this one and the tape recorder's. So this
node republishes the pose it reads as ``/pose`` (PoseStamped in ``map``, 5 Hz, stamped with the
transform's own stamp), and pepin_bringup.run_recorder subscribes to that instead of running a
listener of its own.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import math
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rclpy
from action_msgs.msg import GoalStatusArray
from action_msgs.srv import CancelGoal
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration as RosDuration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time as RosTime
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from pepin.face_events import DriveFace, FaceSink
from pepin.flags import Flag, FlagSet, load_knobs, with_knobs
from pepin.goal_link import CANCEL_CONFIRM_S, NAV_ACTIONS, PLAN_WAIT_S, cancel_outcome
from pepin.places import PLACES_TOPIC, heading_residual_deg, places_from_json
from pepin.runlink import (
    RUN_COMMAND_TOPIC,
    RUN_STATUS_TOPIC,
    RunLink,
    RunStatus,
    start_command,
    stop_command,
)
from pepin.watch import (
    BY_PLACEMENT,
    PLACEMENT_TOPIC,
    GoalGate,
    Placement,
    Preflight,
    Readiness,
)
from pepin_bringup.msgs import scan_arrays, stamp_from_seconds, stamp_seconds, yaw_of
from pepin_bringup.node_kit import Switches, TfLookup, spin_main, tf_failure_kind

PORT = 3337
# The planner to select, and the controller that follows it. One controller now: the lattice
# planner no longer expands in reverse, so there is nothing a reversing controller would add.
PIVOT_TOLERANCE_DEG = 11.5  # the goal checker's 0.20 rad: below this the heading is met
PIVOT_ALLOWANCE_S = 15.0  # a Spin's time allowance for up to half a turn (0.6 rad/s: 5-6 s)
SPIN_MAX_DEG = 360.0  # a `spin` request turns at most a whole turn either way
RECORDER_PATIENCE_S = (
    8.0  # the recorder answers over the bridge; 3 s once named a drive after the previous tape
)
# WHO ASKS FOR MOTION (a request's `source`): a person's ask through a tool or an operator, the
# default; or the speaker gaze's own body turn, which never cancels a person's.
SOURCES = ("tool", "gaze")
A_CANCEL = "a cancel"  # what a run the operator's cancel stopped was cancelled for
SAME_GOAL_M = 0.10  # two goals this near are one target (the goal checker's xy_goal_tolerance)
# A stopped run waits this long at most for Nav2's answer to its cancel, then lets the goal go
# and ends (a Nav2 restarted mid-goal never answers: its run would hold the node for good).
STOPPED_WAIT_S = 10.0
UNANSWERED = f"Nav2 did not answer the cancel within {STOPPED_WAIT_S:g} s: the goal was let go"
# A run cancelled for another request is waited for this long at most before the new one starts:
# the longest a stopped run takes to end, its pivot awaited to the Spin's patience and then its
# tape closed (Nav2's cancel itself answers well within a second).
REPLACE_WAIT_S = PIVOT_ALLOWANCE_S + 5.0 + RECORDER_PATIENCE_S
SETTLE_POLL_S = 0.05  # a spin after a replaced run polls the navigators' status lists this often

MAP_FRAME = "map"
BASE_FRAME = "base_link"
# RTAB-Map's grid as both costmaps read it (relayed by rtabmap_frame, latched): the one place that
# still tells an unsurveyed cell from floor, since the global costmap counts unknown as free
# (track_unknown_space false) and every planner has allow_unknown true.
MAP_TOPIC = "/map"
OCCUPIED_AT = 65  # a /map cell at or above this is occupied (the grids' occupied_thresh 0.65)
# nav2_msgs/ComputePathToPose error codes (Jazzy) that blame the goal, not the planner: another
# goal may plan (ros/tools/planner_check.py's GOAL_CODES).
GOAL_CODES = {204: "goal outside the map", 206: "goal occupied", 208: "no valid path"}
# Nav2's own global costmap (the full grid each publish: always_send_full_costmap, 2 Hz), kept raw
# and decoded only when `plan` asks: the lidar's and the camera's marks, a person's legs and body
# among them. A goal where the cart's hull would cover a lethal cell is inside that obstacle's
# footprint grown by the cart's (the polygon collision check every controller makes): Nav2 cannot
# reach it, and the tree answers it with minutes of backing up and spinning beside the person
# (2026-10-13, "come a bit closer to me"). The inscribed band (253) is left to the planner: with
# base_link 6 cm behind the bumper it is where a bumper-to-furniture place stands. The check is
# the flag goal_hull_check, OFF since the evening of 2026-10-13 (Artem: two judges of one goal is a
# split brain; the Smac Hybrid planner with its goal tolerance is to be the one that answers an
# occupied goal with the nearest free pose, the "taxi"): the code stays for the A/B.
COSTMAP_TOPIC = "/global_costmap/costmap"
LETHAL_AT = 100  # nav2_costmap_2d's LETHAL_OBSTACLE (254) as its OccupancyGrid carries it
# THE ONE POSE TOPIC. This node already parses /tf for the pose; every other node beside it that
# wants it reads this instead of starting a second listener (pepin_bringup.run_recorder). 5 Hz is
# the rate the tape thinned its `loc` rows to anyway and the local costmap's own
# update_frequency.
POSE_TOPIC = "/pose"
POSE_HZ = 5.0
TF_WAIT_S = 0.3  # how long a pose lookup waits for the edge: the drive thread asks, not a callback
TF_FIRST_WAIT_S = 2.0  # ...and the first one waits for the listener's buffer to fill at all
# `pose_at`: the frames it answers, from `map` (where a thing is) or from `odom` (how the camera
# moved between two stamps, smooth: a localisation update of map -> odom between them is no
# motion), and the history it can answer from. The buffer
# is tf2_ros.Buffer's default (node_kit.TfLookup builds it without a cache_time): tf2's
# BufferCore keeps 10 s behind each edge's newest sample, so a stamp older than that is refused
# with the reason before any lookup. A stamp a little newer than the chain's newest edge (a
# camera frame against the last neck line, 50 Hz) is waited for up to POSE_AT_WAIT_S; one more
# than POSE_AT_AHEAD_S past this node's clock means the clocks disagree (ros/time.sh offset).
POSE_AT_FRAMES = (BASE_FRAME, "camera_link")
POSE_AT_PARENTS = (MAP_FRAME, "odom")
TF_HISTORY_S = 10.0
POSE_AT_WAIT_S = 0.5
POSE_AT_AHEAD_S = 1.0

# The live flags (CLAUDE.md rule 19); their state is printed in the node's start line.
FLAGS = FlagSet(
    Flag(
        "controller",
        "shim_mppi",  # 2026-10-02: the shim drives, MPPI parks (was mppi; drives from 0295)
        choices=("mppi", "rpp", "rpp_shim", "graceful", "dwb", "shim_mppi"),
        description="what follows the plan: mppi is Nav2's MPPI controller for every planner,"
        " held to the mark's heading by the yaw-checking goal checker; rpp is each planner's own"
        " Regulated Pure Pursuit from PLANNERS, ending on position alone as before 2026-09-23;"
        " rpp_shim is the reversing RPP inside Nav2's RotationShimController, which turns the cart"
        " to the mark's heading in place once it is inside the goal tolerance;"
        " graceful and dwb are Nav2's Graceful and DWB controllers, for an A/B against mppi;"
        " shim_mppi drives each goal on rpp_shim and hands it to mppi for the rest of that goal"
        " once the cart is within the knob park_distance_m of the goal (PARKERS)"
        " (ros/goto.sh controller NAME)."
        " Published latched on controller_selector and goal_checker_selector; the behaviour"
        " tree plans at once on a controller change and hands FollowPath the new controller"
        " together with that plan",
        why="the six legs of 2026-09-23 all stopped 12-60 deg short of the mark's heading:"
        " the reversing RPP cannot rotate in place, the tree ended the drive on position"
        " (xy_only_goal_checker), and the pivot that finishes the heading lives here in"
        " _pivot_to, which drives sent through goto_ros.py never reach. Each leg also ran 5-9"
        " recoveries, which is what an RPP answers a refused arc with; MPPI samples another"
        " trajectory instead",
        on_when="mppi by default since Nav2 moved to the Mac (2026-10-01): the board's MPPI"
        " crawled at a median 0.06 m/s, which is why rpp_shim was the default from the evening of"
        " 2026-09-23 (six legs: 21-33 s, 6-11 recoveries, 4-10 cm and 2-5 deg at the mark), and"
        " the Mac's MPPI is what every drive since 2026-09-30 was set to by hand after each start",
        off_when="rpp_shim for an A/B against the board's RPP drives, rpp for the position-only"
        " drives of before, or if MPPI's cycle does not fit the control period (the controller"
        " server's 'Control loop missed its desired rate')",
    ),
    Flag(
        "start_needs_placement",
        True,
        description="a goal or a mark waits for the laptop's word"
        f" on {PLACEMENT_TOPIC} (pepin_bringup.rtabmap_frame, latched) that this start of"
        " RTAB-Map is PLACED — a node of the loaded map recognised, or an operator's seed — and"
        " nothing heard is refused like not placed. Off, a fresh map -> base_link is enough, as"
        " before 2026-09-23",
        why="on, measured 2026-09-23: after a restart RTAB-Map publishes map -> odom from the"
        " pose it SAVED at its last shutdown, and a fresh transform was taken for a localisation"
        " — 'at home' at the bookshelf with 0 of 198 updates recognised, then 76 cm off inside"
        " the table. The laptop's flag of the same name only changes what rtabmap_frame SAYS:"
        " it cannot lift a refusal of silence, from a node that is down, respawning or running"
        " code from before the word existed. This one is the board-side switch the refusal"
        " itself answers to",
        on_when="always: a pose nobody has vouched for since"
        " RTAB-Map's start is not a pose to drive on",
        off_when="when the word cannot come and the cart is known to stand where RTAB-Map's pose"
        " says: pepin-vslam down or started before this build (ros/laptop.sh vslam restarts it"
        " on the checkout), or a dark room with no seed at hand",
    ),
    Flag(
        "goal_hull_check",
        False,
        description="`plan` refuses a goal where the cart's hull standing there would cover a"
        " lethal cell of Nav2's global costmap (hull_on_costmap: a person's legs, a chair) before"
        " any planner is asked, blaming the goal; off, the planner alone judges the goal and"
        " `plan` reports the costmap as unchecked",
        why="off since 2026-10-13 evening (Artem): the check refused every approach stand 0.3 m"
        " from a seated person (the hull over 1-26 lethal cells, his shins) before Smac Hybrid"
        " ever ran, so its goal tolerance (0.25 m, the nearest free pose, the taxi) was never"
        " tried; two judges of one goal are a split brain, the planner is the one",
        on_when="a planner without a goal tolerance (NavFn, Theta*) drives approach goals: a"
        " goal inside a person's inflated footprint then costs minutes of recoveries beside them",
        off_when="Smac Hybrid (or Lattice) is the planner for goals near people: its tolerance"
        " answers an occupied goal; measure the stops it picks before trusting it (2026-10-14)",
    ),
    Flag(
        "face_events",
        True,
        description="the drive's moments on the head's face (pepin.face_events, through the"
        " board's head server on PEPIN_HOST:3340): focused while a goal runs, struggling for a"
        " moment at each new Nav2 recovery, happy on arrival, sad on an abort or a refusal, a"
        " flat line on a cancel (config/face.json's events say what each looks like); and a"
        " brain lease every 2 s, so the face falls asleep when this node or the WiFi is gone",
        why="on since 2026-10-05: set live for the demo's voice drives, the head showed each"
        " drive's moments with no cost to the drive (a send never blocks; a head server that is"
        " away costs one logged reconnection every 5 s); the robot shows what it does",
        on_when="the head is on the robot and its server answers (uv run python -m"
        " pepin.head_link status)",
        off_when="the face gets in the way of a test, or the head server is not there (each"
        " moment is then dropped, the reconnection logged every 5 s)",
    ),
    Flag(
        "spin_requests",
        True,
        description="`spin` turns the cart in place by `deg` (+ left, at most a whole turn) with"
        " one goal on the behaviour server's Spin, answered like a drive (accepted with the run,"
        " done with Nav2's status, the pose and the turn measured on map -> base_link), recorded"
        " and stopped by a cancel like one; the brain's turn tool sends it. Off, `spin` is"
        " refused with this flag's name",
        why="on since 2026-10-14: approach(me, face) sent a turn in place as a pose goal through"
        " FollowPathShim, MPPI parked it, and it ended 40 deg short after 30 s (run 12, 13:40);"
        " the end-of-drive pivot's Spin turns 40 deg in about 1.5 s",
        on_when="the turn tool is offered to the brain (`повернись ко мне`, `налево`)",
        off_when="a test of the voice's turns without the cart moving, or the behaviour server's"
        " Spin misbehaves (Collision Ahead beside furniture: nav2_params.yaml"
        " simulate_ahead_time)",
    ),
)
FACE_LEASE_EVERY_S = 2.0  # pepin.face_events.LEASE_S lasts three of these


def face_client() -> FaceSink:
    """The head server's door for this node's moments (``source`` goal), on the board."""
    from pepin.audio_link import board_host
    from pepin.head_link import HEAD_PORT, HeadClient

    return HeadClient(board_host(), HEAD_PORT, source="goal").start()


PLANNERS = {
    "navfn": ("GridBased", "FollowPath"),
    "lattice": ("Lattice", "FollowPath"),  # experimental: see nav2_params.yaml
    "theta": ("ThetaStar", "FollowPath"),
    "smac": ("Smac2D", "FollowPath"),
    "hybrid": ("Hybrid", "FollowPathRS"),  # footprint-aware; the reversing RPP
}
# WHAT FOLLOWS THE PLAN, and the goal checker that ends the drive with it (flag `controller`).
# "rpp" is the catalogue above: each planner's own Regulated Pure Pursuit, ending on position
# alone — the heading is this node's pivot after the drive (_pivot_to), a path goto_ros.py never
# took. "mppi" is one controller for every planner, and it turns to the heading itself, so the
# tree holds it to the heading as well.
RPP_GOAL_CHECKER = "xy_only_goal_checker"
# The lidar as Nav2 receives it: `where` says how old the last scan is. Report only: the driver
# is the board's, and a restart of it from here (the re-plug recovery of 2026-09-28) went with
# Nav2 to the Mac, where neither its port nor its process is.
LIDAR_SCAN_TOPIC = "/scan"
LIDAR_SILENT_S = 5.0  # no scan for this long: `where` says so
# `clearance` (the voice's expressive wheels, pepin.expression.wheels, ten times a second while
# they move): depth_fusion's obstacle marks near the cart, decoded only when asked. The marks are
# the voxel volume's own surface sliced round the whole turn in base_link (the lidar, the camera's
# depth and the ToF fused, remembered behind the cart), never Nav2's inflated costmaps; a bearing
# the volume holds nothing on is NaN and blocks nothing. No marks, no clearance: the raw scan is
# not a stand-in, so a silent depth_fusion refuses every expressive motion with its reason.
MARKS_TOPIC = "/depth_marks"
MARKS_SILENT_S = 5.0  # no marks for this long: `clearance` says depth_fusion is silent
CLEARANCE_RADIUS_M = 1.0  # the default; the request may name its own, up to the max
CLEARANCE_MAX_RADIUS_M = 3.0


def scan_message(raw: bytes) -> Any:
    """A raw ``LaserScan`` (as a raw subscription keeps it: the marks) as a message."""
    from rclpy.serialization import deserialize_message

    return deserialize_message(raw, LaserScan)


def costmap_message(raw: bytes) -> Any:
    """A raw ``OccupancyGrid`` (Nav2's global costmap, as the raw subscription keeps it)."""
    from rclpy.serialization import deserialize_message

    return deserialize_message(raw, OccupancyGrid)


def plan_end(pose: Any, x: float, y: float, yaw_deg: float) -> dict[str, Any]:
    """Where a plan ends, from its last pose (map frame), and how far that is from the goal asked
    at ``(x, y, yaw_deg)``: ``end`` {x, y, yaw_deg}, ``short_m`` and ``short_deg`` (the end's
    heading minus the goal's, (-180, 180]). A planner with a goal tolerance (Smac Hybrid's) ends
    at the nearest pose it reaches, not at the goal. A last pose whose quaternion is all zeros
    carries no heading: ``yaw_deg`` and ``short_deg`` are None."""
    q = pose.orientation
    ex, ey = float(pose.position.x), float(pose.position.y)
    headed = math.hypot(q.x, q.y, q.z, q.w) > 1e-9
    end_deg = math.degrees(yaw_of(q)) if headed else None
    return {
        "end": {
            "x": round(ex, 3),
            "y": round(ey, 3),
            "yaw_deg": None if end_deg is None else round(end_deg, 1),
        },
        "short_m": round(math.hypot(ex - x, ey - y), 3),
        "short_deg": None if end_deg is None else round(heading_residual_deg(end_deg, yaw_deg), 1),
    }


# Goals that are running on an action server, by action_msgs/GoalStatus: ACCEPTED, EXECUTING and
# CANCELING (a canceling goal still drives until its server lets it go).
ACTIVE_STATUSES = (1, 2, 3)
FOLLOWERS = {
    "mppi": ("FollowPathMPPI", "general_goal_checker"),
    # The reversing RPP inside Nav2's RotationShimController: RPP's pace, and the shim turns the
    # cart in place to the mark's heading once it is inside the goal tolerance.
    "rpp_shim": ("FollowPathShim", "general_goal_checker"),
    # Both turn to the mark's heading themselves (Graceful's final rotation, DWB's RotateToGoal).
    "graceful": ("FollowPathGraceful", "general_goal_checker"),
    "dwb": ("FollowPathDWB", "general_goal_checker"),
    # Drive on the shim, park on MPPI: the pair a goal starts on (PARKERS hands over).
    "shim_mppi": ("FollowPathShim", "general_goal_checker"),
}
# DRIVE ON ONE CONTROLLER, PARK ON ANOTHER (flag values listed here): a goal starts on the pair
# FOLLOWERS names and hands over to this one once map -> base_link is within the knob
# park_distance_m of the goal (straight line), for the rest of that goal; the next goal starts on
# FOLLOWERS' pair again. The tree plans at once on the new controller id and FollowPath passes
# it with that plan to the controller server, which swaps the plugin in place on the same goal.
PARKERS = {"shim_mppi": ("FollowPathMPPI", "general_goal_checker")}


@dataclass(eq=False)
class Run:
    """One request for motion, this node's while it runs: who asked (``source``), what
    (``kind`` go or spin, ``target`` its pose x, y, yaw_deg or its turn in degrees, and in
    ``words``), its number once the recorder named its tape, the cancel count it started
    under; whether it was stopped (a cancel, or another request: ``cancelled_for``), for a
    request waiting for the run before it to end whether a newer one dropped it, and whether
    it cancelled a run under way itself."""

    source: str
    kind: str
    target: tuple[float, ...]
    words: str
    number: int | None = None
    cancels: int = 0
    stopped: bool = False
    cancelled_for: str | None = None
    dropped: bool = False
    replaced: bool = False  # it cancelled a run under way to start

    @property
    def label(self) -> str:
        """``go printer (tool)``: the request and who asked."""
        return f"{self.words} ({self.source})"

    def same(self, other: Run) -> bool:
        """One source asking for one target: a pose the other's arrival would count as reached
        (within the goal checker's SAME_GOAL_M and PIVOT_TOLERANCE_DEG), or a turn within the
        heading's tolerance of the other's."""
        if self.source != other.source or self.kind != other.kind:
            return False
        if self.kind == "spin":
            return abs(self.target[0] - other.target[0]) <= PIVOT_TOLERANCE_DEG
        x, y, yaw = self.target
        ox, oy, oyaw = other.target
        near = math.hypot(x - ox, y - oy) <= SAME_GOAL_M
        return near and abs(heading_residual_deg(yaw, oyaw)) <= PIVOT_TOLERANCE_DEG

    def about(self) -> dict[str, Any]:
        """The run as another request's answer names it: its number, who asked, what."""
        return {"run": self.number, "source": self.source, "target": self.words}


def _for(run: Run) -> dict[str, Any]:
    """A stopped run's end names the request it was cancelled for (none for a cancel)."""
    reason = run.cancelled_for
    return {} if reason is None or reason == A_CANCEL else {"cancelled_for": reason}


class GoalServer(Node):
    """Takes goals over a socket and drives them through Nav2, with the run recorded."""

    def __init__(self) -> None:
        super().__init__("goal_server")
        # Callbacks can run BEFORE this constructor is done: node_kit.TfLookup starts a spin
        # thread for THIS node, so from the moment the pose publisher below creates one, every
        # timer and subscription of this node may fire against a half-built object. Until the
        # last line of __init__ the timers that were added with that listener drop their tick
        # (the pattern depth_fusion paid for on 2026-09-22, when a pair that met a half-built
        # node killed the executor and silenced the marks).
        self._up = False
        self._record_dir = Path(str(self.declare_parameter("record_dir", "/maps/rec").value))
        self._port = int(self.declare_parameter("port", PORT).value)
        self._client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        # A plan without a drive (the `plan` command): the planner server's own action, asked
        # as the navigator asks it, so a goal is checked before anything moves.
        self._planner = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
        self._map: OccupancyGrid | None = None
        self.create_subscription(
            OccupancyGrid,
            MAP_TOPIC,
            self._on_map,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        # Nav2's global costmap, taken raw (decoded only when `plan` asks): latched and
        # reliable, as Nav2 publishes it and ros/tools/planner_check.py reads it.
        self._costmap: bytes | None = None
        self.create_subscription(
            OccupancyGrid,
            COSTMAP_TOPIC,
            self._on_costmap,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
            raw=True,
        )
        self._spin = ActionClient(self, Spin, "spin")
        # Every goal on either navigator, whoever sent it: the action servers' own cancel
        # services, made once here so a cancel never waits for discovery.
        self._cancel_clients = {
            action: self.create_client(CancelGoal, f"/{action}/_action/cancel_goal")
            for action in NAV_ACTIONS
        }
        # The cart's pose: map -> base_link, RTAB-Map's correction composed with the board's own
        # odometry. The listener behind it is started by the pose topic below.
        self._tf: TfLookup | None = None
        self._gate = GoalGate()
        # THE ROOM'S PLACES, FROM THE GRAPH (World R). The laptop's places node republishes every
        # named place as a pose in `map`, recomputed whenever the graph bends (latched, so the last
        # known book survives a WiFi drop). Once it has been heard, a name means what IT says and
        # nothing else: the yaml beside the old map holds coordinates of a frame that no longer
        # exists, and on 2026-09-19 `go home` took (-9.39, +2.53) from it, the planner answered
        # "Goal Coordinates ... outside bounds" and the behaviour tree spun recoveries.
        self._graph_places: dict[str, dict[str, float]] | None = None
        self.create_subscription(
            String,
            PLACES_TOPIC,
            self._on_places,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        # WHETHER RTAB-MAP'S START IS PLACED (pepin.watch.Placement, from the laptop's
        # rtabmap_frame, latched): a fresh map -> base_link is not a pose until this start of
        # RTAB-Map has recognised the loaded map or been seeded.
        # None until heard, which _ready refuses as "nobody said" (flag start_needs_placement).
        self._placement: Placement | None = None
        self.create_subscription(
            String,
            PLACEMENT_TOPIC,
            self._on_placement,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        # Latched: the behaviour tree reads its selector once, whenever it next ticks.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._planner_pick = self.create_publisher(String, "planner_selector", latched)
        self._controller_pick = self.create_publisher(String, "controller_selector", latched)
        self._checker_pick = self.create_publisher(String, "goal_checker_selector", latched)
        # Remembered across restarts: a container that comes back with a different planner than
        # the one being tested makes every comparison a lie.
        # Under maps/rec, which sync.sh excludes: kept in maps/ the file was deleted by the very
        # next deploy (rsync --delete), so every restart silently went back to the default planner
        # and a drive was credited to a planner that never ran.
        self._planner_path = self._record_dir / ".planner"
        self.planner = "navfn"  # the saved pick is published once the switches exist (below)
        self.controller = ""  # the controller id last published on controller_selector
        self.goal_checker = ""  # ...and the goal checker id on goal_checker_selector
        self._goal_handle: Any = None
        # THE RUN UNDER WAY (a go or a spin), from its admission until its session is over, and
        # a request waiting for that run's end after cancelling it (_admit: the one rule).
        self._current: Run | None = None
        self._next: Run | None = None
        # The end-of-drive pivot (behaviour server Spin) is a goal of its own: a cancel stops it
        # too. Every cancel bumps _cancels, so a drive that started before it never sends a goal
        # after it (see _go: the recorder alone may take 8 s between the two).
        self._spin_handle: Any = None
        self._cancels = 0
        # When the last cancel came (ros/stop.sh always sends one): `clearance` tells its age, and
        # the voice's wheels take it for the red button even when the base's stop line was
        # overwritten by their own next twist before any state line showed it.
        self._cancelled_at: float | None = None
        # Whether ANY goal runs on the navigators, whoever sent it: their latched status lists,
        # read here beside them. `where` answers it, and the measured motions refuse on it.
        self._nav_active: dict[str, bool] = dict.fromkeys((*NAV_ACTIONS, "spin"), False)
        status_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for action in self._nav_active:
            self.create_subscription(
                GoalStatusArray,
                f"/{action}/_action/status",
                lambda msg, a=action: self._on_nav_status(a, msg),
                status_qos,
            )
        # The lidar as Nav2 receives it, taken raw: only its arrival matters here (`where`).
        self._last_scan_at: float | None = None
        self._started_at = time.monotonic()
        self.create_subscription(
            LaserScan, LIDAR_SCAN_TOPIC, self._on_scan, qos_profile_sensor_data, raw=True
        )
        # depth_fusion's obstacle marks (pepin-vslam), taken raw: `clearance` decodes the newest
        # only when it is asked.
        self._last_marks_at: float | None = None
        self._last_marks: bytes | None = None
        self.create_subscription(
            LaserScan,
            MARKS_TOPIC,
            self._on_marks,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            raw=True,
        )
        # The recorder is a node of its own (run_recorder, beside this one): one command opens a
        # tape, the latched status names it (pepin.runlink).
        self._runs = RunLink()
        self._run_word = threading.Event()  # set on every status heard
        self._run_pub = self.create_publisher(String, RUN_COMMAND_TOPIC, 10)
        self.create_subscription(
            String,
            RUN_STATUS_TOPIC,
            self._on_run_status,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._lock = threading.Lock()
        self._turn = threading.Condition(self._lock)  # a run's end, for the request waiting on it
        # The drive's moments on the head's face (flag face_events): before the switches, whose
        # callback runs on the declarations too.
        self._drive_face: DriveFace | None = None
        # Last, after every other declare_parameter: rclpy runs the switches' callback on
        # declarations too, and a name outside the table is refused there (node_kit.Switches).
        self._switches = Switches(
            self, with_knobs(FLAGS, load_knobs("goal_server")), on_change=self._on_switch
        )
        self._face_switch(bool(self._switches["face_events"]))
        self.create_timer(FACE_LEASE_EVERY_S, self._renew_face_lease)
        # The saved planner (the default where none was saved), with the follower the
        # `controller` flag names for it: after the switches, because the pick reads that flag,
        # and always, because the tree's own defaults are the RPP pair's.
        saved = ""
        with contextlib.suppress(OSError):
            saved = self._planner_path.read_text().strip()
        self.pick_planner(saved or self.planner)
        self._start_pose_topic()  # /pose, the one pose topic
        threading.Thread(target=self._serve, daemon=True).start()
        self.get_logger().info(
            f"goal server ready on port {self._port} (pose: map -> base_link from TF);"
            f" {POSE_TOPIC} at {POSE_HZ:.0f} Hz from map -> base_link, for the other readers;"
            f" {self._switches.state()}"
        )
        self._up = True  # last: everything above exists, the timers may run

    def _on_scan(self, _raw: bytes) -> None:
        """A scan arrived (left unparsed): the lidar reaches Nav2."""
        self._last_scan_at = time.monotonic()

    def _on_marks(self, raw: bytes) -> None:
        """depth_fusion's marks arrived (kept unparsed until `clearance` asks)."""
        self._last_marks, self._last_marks_at = raw, time.monotonic()

    def clearance(self, radius_m: float, now: float) -> dict[str, Any]:
        """``clearance``'s answer: whether anything navigates, the marks' status, how long ago
        the last cancel came (None: none since the start), the newest marks' age by their own
        stamp, and their points within ``radius_m`` of the cart (base_link, metres: the slice is
        published in base_link, its ranges from that frame's origin)."""
        import numpy as np

        from pepin.expression.clearance import near_points

        cancelled = self._cancelled_at
        answer: dict[str, Any] = {
            "event": "clearance",
            "navigating": self.navigating(),
            "marks": self.marks_status(now),
            "cancel_age_s": None if cancelled is None else round(now - cancelled, 3),
            "marks_age_s": None,
            "points": [],
        }
        raw = self._last_marks
        if raw is None:
            return answer
        msg = scan_message(raw)
        angles, ranges = scan_arrays(msg)
        radius = min(max(radius_m, 0.0), CLEARANCE_MAX_RADIUS_M)
        points = near_points(angles, ranges, np.eye(3), np.zeros(3), radius)
        clock = self.get_clock().now().nanoseconds * 1e-9
        answer["marks_age_s"] = round(max(0.0, clock - stamp_seconds(msg.header.stamp)), 3)
        answer["points"] = [[round(float(x), 3), round(float(y), 3)] for x, y in points]
        return answer

    def marks_status(self, now: float) -> str:
        """``ok``, or how long depth_fusion's marks have not reached this node."""
        last = self._started_at if self._last_marks_at is None else self._last_marks_at
        silent = now - last
        if silent < MARKS_SILENT_S:
            return "ok"
        heard = "" if self._last_marks_at is not None else ", none since the start"
        return f"silent {silent:.0f} s{heard} (depth_fusion, in pepin-vslam: ros/laptop.sh vslam)"

    def lidar_status(self, now: float) -> str:
        """``ok``, or how long no scan has reached this node (report only)."""
        last = self._started_at if self._last_scan_at is None else self._last_scan_at
        silent = now - last
        if silent < LIDAR_SILENT_S:
            return "ok"
        heard = "" if self._last_scan_at is not None else ", none since the start"
        return f"silent {silent:.0f} s{heard} (the board's lidar: ros/board.sh census)"

    def _on_nav_status(self, action: str, msg: GoalStatusArray) -> None:
        """One navigator's latched status list: whether a goal of anyone's runs on it."""
        self._nav_active[action] = any(s.status in ACTIVE_STATUSES for s in msg.status_list)

    def navigating(self) -> bool:
        """A run of this node, or any goal running on a navigator or the spin behaviour."""
        with self._lock:
            driving = self._current is not None or self._next is not None
            driving = driving or self._spin_handle is not None
        return driving or any(self._nav_active.values())

    def _plan_races(self) -> bool:
        """A plan now would race a navigator's own: a ``go`` of this node's under way, or any
        navigator's goal of anyone's (the planner server serves the newest request and aborts
        the one it was planning for the tree). A turn in place plans nothing."""
        with self._lock:
            run = self._current
        busy = run is not None and run.kind == "go"
        return busy or any(self._nav_active[action] for action in NAV_ACTIONS)

    @staticmethod
    def _wait(future: Any, timeout: float) -> Any:
        """Wait for a future from a session thread; the node's own spin drives it to completion.

        Spinning here instead raises "executor is already spinning": a process may spin in one
        place only, and that place is main().
        """
        done = threading.Event()
        future.add_done_callback(lambda _future: done.set())
        return future.result() if done.wait(timeout) else None

    # -- the run's recording ---------------------------------------------------

    def _on_run_status(self, msg: String) -> None:
        status = RunStatus.from_json(msg.data)
        if status is not None:
            self._runs.observe(status)
            self._run_word.set()

    def _await_recorder(self, done: Any) -> bool:
        """Wait up to RECORDER_PATIENCE_S for the recorder's status to satisfy ``done``."""
        deadline = time.monotonic() + RECORDER_PATIENCE_S
        while not done():
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            self._run_word.clear()
            self._run_word.wait(left)
        return True

    def start_recording(self, name: str) -> Path | None:
        """Ask the recorder for this run's tape: its path once confirmed, None if nobody answered.

        A drive is not held hostage by its recorder: after RECORDER_PATIENCE_S it goes anyway,
        loudly, with no recording named in its events.
        """
        self._run_pub.publish(String(data=start_command(name)))
        if not self._await_recorder(lambda: self._runs.started(name)):
            self.get_logger().warning(
                f"no recorder confirmed run '{name}' in {RECORDER_PATIENCE_S:.0f} s: "
                "driving unrecorded"
            )
            return None
        path = Path(str(self._runs.recording))
        self.get_logger().info(f"run {self._runs.run}: recording {path}")
        return path

    def close(self) -> None:
        """Before the node goes (node_kit.spin_main's order): close a tape still open."""
        self.stop_recording()

    def stop_recording(self) -> None:
        """Close the run's tape (the recorder flushes and syncs it); harmless when none is open."""
        if self._runs.stopped():
            return
        self._run_pub.publish(String(data=stop_command()))
        if not self._await_recorder(self._runs.stopped):
            self.get_logger().warning("the recorder did not confirm the tape closed")

    def _serve(self) -> None:
        """One connection at a time: read a command, stream its events back, close."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", self._port))
        listener.listen(1)
        while rclpy.ok():
            try:
                connection, _ = listener.accept()
            except OSError:
                continue
            threading.Thread(target=self._session, args=(connection,), daemon=True).start()

    def _session(self, connection: socket.socket) -> None:
        with connection:
            connection.settimeout(600.0)
            try:
                line = connection.makefile("r").readline()
                request = json.loads(line) if line.strip() else {}
            except (OSError, ValueError) as error:
                self._send(connection, {"event": "error", "detail": str(error)[:120]})
                return
            try:
                self._handle(request, connection)
            except Exception as error:  # a bad command must not take the server down
                self._send(connection, {"event": "error", "detail": str(error)[:200]})

    @staticmethod
    def _send(connection: socket.socket, payload: dict[str, Any]) -> None:
        with contextlib.suppress(OSError):  # the caller may have hung up mid-run
            connection.sendall((json.dumps(payload) + "\n").encode())

    # -- commands --------------------------------------------------------------

    def _handle(self, request: dict[str, Any], connection: socket.socket) -> None:
        command = str(request.get("cmd", ""))
        if command == "places":
            self._send(connection, {"event": "places", "places": self.places()})
        elif command == "where":
            pose = self._pose_now()
            self._send(
                connection,
                {
                    "event": "where",
                    "planner": self.planner,
                    "pose": "tf" if pose else "none",
                    "navigating": self.navigating(),
                    "lidar": self.lidar_status(time.monotonic()),
                    **pose,
                },
            )
        elif command == "clearance":
            radius = float(request.get("radius_m", CLEARANCE_RADIUS_M))
            self._send(connection, self.clearance(radius, time.monotonic()))
        elif command == "planner":
            self._send(connection, self.pick_planner(str(request.get("name", ""))))
        elif command == "cancel":
            # This node's own goal is one of "every": its handle is only let go here, so the
            # navigators' answer counts it once instead of rejecting a goal a second request had
            # already put into canceling.
            had_goal = self.cancel(send=False)
            navigators = self.cancel_every_goal()
            self._send(
                connection, {"event": "cancelled", "had_goal": had_goal, "navigators": navigators}
            )
        elif command == "go":
            self._go(request, connection)
        elif command == "spin":
            self._spin_in_place(request, connection)
        elif command == "plan":
            self._send(connection, self.plan(request))
        elif command == "pose_at":
            self._send(connection, self.pose_at(request))
        else:
            self._send(connection, {"event": "error", "detail": f"unknown command {command!r}"})

    def _on_map(self, msg: OccupancyGrid) -> None:
        """RTAB-Map's newest grid (latched): what ``plan`` reads a goal's cell from."""
        self._map = msg

    def _on_costmap(self, raw: bytes) -> None:
        """Nav2's newest global costmap (kept unparsed until ``plan`` asks)."""
        self._costmap = raw

    def hull_on_costmap(self, x: float, y: float, yaw_deg: float) -> tuple[str, str]:
        """What Nav2's global costmap holds under the cart's whole hull standing at ``(x, y,
        yaw_deg)`` (:func:`pepin.footprint.hull_cells`): ``clear``, ``lethal`` (a marked
        obstacle under the hull: the goal is inside its footprint grown by the cart's), or ``no
        costmap`` before one is heard; with the reason in words (empty when clear)."""
        from pepin.footprint import hull_cells

        if self._costmap is None:
            return "no costmap", f"no global costmap heard on {COSTMAP_TOPIC}"
        grid = costmap_message(self._costmap)
        info = grid.info
        width, height, data = int(info.width), int(info.height), grid.data
        origin = (float(info.origin.position.x), float(info.origin.position.y))
        lethal = sum(
            1
            for col, row in hull_cells(x, y, math.radians(yaw_deg), origin, float(info.resolution))
            if 0 <= col < width and 0 <= row < height and int(data[row * width + col]) >= LETHAL_AT
        )
        if lethal:
            return "lethal", (
                f"the cart standing there would cover {lethal} lethal cells of Nav2's costmap:"
                " the goal is inside an obstacle's footprint (a person, furniture)"
            )
        return "clear", ""

    def map_cell(self, x: float, y: float) -> str:
        """What RTAB-Map's grid says under map point ``(x, y)``: ``free``, ``occupied``,
        ``unknown`` (never surveyed), ``outside`` the grid, or ``no map`` before one is heard.
        The grid's origin is taken unrotated, as RTAB-Map writes it."""
        grid = self._map
        if grid is None:
            return "no map"
        info = grid.info
        col = math.floor((x - info.origin.position.x) / info.resolution)
        row = math.floor((y - info.origin.position.y) / info.resolution)
        if not (0 <= col < info.width and 0 <= row < info.height):
            return "outside"
        value = int(grid.data[row * info.width + col])
        if value < 0:
            return "unknown"
        return "occupied" if value >= OCCUPIED_AT else "free"

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        """Would a goal at ``(x, y, yaw_deg)`` be driven: never motion. RTAB-Map's cell under it
        first (unknown, occupied or outside the grid is refused before any plan: the costmap
        cannot tell unknown from floor), then, behind the flag ``goal_hull_check``, Nav2's global
        costmap under the cart's whole hull standing there (:meth:`hull_on_costmap`: a goal inside
        a person's or a thing's inflated footprint is refused, never sent to the navigator; off,
        the planner alone judges it), then one ComputePathToPose from where the cart stands with
        the selected planner. ``blames`` says whose the refusal is: ``goal`` (another goal may
        plan) or ``planner`` (none will); an ``error`` event when it cannot be asked. A plan
        found says where it ends (:func:`plan_end`: ``end``, ``short_m``, ``short_deg``): a
        planner with a goal tolerance may end short of the goal, and that end is what a drive
        there reaches."""
        try:
            x, y = float(request["x"]), float(request["y"])
            yaw_deg = float(request.get("yaw_deg", 0.0))
        except (KeyError, TypeError, ValueError):
            return {"event": "error", "detail": "plan needs numbers x and y (and yaw_deg)"}
        answer: dict[str, Any] = {"event": "plan", "x": x, "y": y, "yaw_deg": yaw_deg}
        if self._plan_races():
            return {"event": "error", "detail": "a drive is under way: a plan now would race it"}
        cell = self.map_cell(x, y)
        answer["cell"] = cell
        if cell == "no map":
            return answer | {
                "ok": False,
                "blames": "planner",
                "reason": f"no grid heard on {MAP_TOPIC}: free cannot be told from unknown",
            }
        if cell != "free":
            reason = {
                "occupied": "the goal is on an occupied cell of the map",
                "unknown": "the goal is on a cell the map never surveyed",
                "outside": "the goal is outside the map",
            }[cell]
            return answer | {"ok": False, "blames": "goal", "reason": reason}
        if self._switches.on("goal_hull_check"):
            hull, why = self.hull_on_costmap(x, y, yaw_deg)
            answer["costmap"] = hull
            if hull == "no costmap":
                return answer | {"ok": False, "blames": "planner", "reason": why}
            if hull != "clear":
                return answer | {"ok": False, "blames": "goal", "reason": why}
        else:
            answer["costmap"] = "unchecked"  # the planner's to judge (the flag's note)
        if not self._planner.wait_for_server(timeout_sec=2.0):
            return {"event": "error", "detail": "the planner server is not up"}
        goal = ComputePathToPose.Goal()
        goal.goal = self._pose_msg(x, y, yaw_deg)
        goal.planner_id = PLANNERS[self.planner][0]
        goal.use_start = False
        handle = self._wait(self._planner.send_goal_async(goal), 5.0)
        if handle is None or not handle.accepted:
            return {"event": "error", "detail": "the planner did not take the request"}
        outcome = self._wait(handle.get_result_async(), PLAN_WAIT_S)
        if outcome is None:
            return {"event": "error", "detail": f"no plan in {PLAN_WAIT_S:.0f} s"}
        result = outcome.result
        poses = list(result.path.poses)
        if len(poses) >= 2:
            length = sum(
                math.hypot(
                    b.pose.position.x - a.pose.position.x, b.pose.position.y - a.pose.position.y
                )
                for a, b in itertools.pairwise(poses)
            )
            planned = {"ok": True, "length_m": round(length, 2), "poses": len(poses)}
            return answer | planned | plan_end(poses[-1].pose, x, y, yaw_deg)
        code = int(getattr(result, "error_code", 0))
        said = str(getattr(result, "error_msg", "") or "")
        reason = GOAL_CODES.get(code) or said or f"planner error code {code}"
        blames = "goal" if code in GOAL_CODES else "planner"
        return answer | {"ok": False, "blames": blames, "reason": reason, "code": code}

    def _on_placement(self, msg: String) -> None:
        """rtabmap_frame's word on what RTAB-Map's present start rests on; a message that does
        not parse leaves the last one standing."""
        heard = Placement.from_json(msg.data)
        if heard is not None:
            self._placement = heard

    def _placement_standing(self) -> Placement | None:
        """rtabmap_frame's last word while an rtabmap_frame is there to stand behind it; ``None``
        with no publisher left on the topic. A latched word outlives its node here: the laptop's
        vslam restarting would otherwise leave the OLD start's "placed" in force until the new
        node's first word, over a map -> odom already fresh at the new start's saved pose."""
        if self.count_publishers(PLACEMENT_TOPIC) == 0:
            return None
        return self._placement

    def _on_places(self, msg: String) -> None:
        """The graph's book, as the places node last published it."""
        self._graph_places = {
            name: {"x": place.x, "y": place.y, "yaw_deg": place.theta_deg or 0.0}
            for name, place in places_from_json(msg.data).items()
        }

    def places(self) -> dict[str, dict[str, float]]:
        """The named places of the map in use: the graph's book, and ONLY it. Until the book has
        been heard there are no places — a name is refused with that reason."""
        return {} if self._graph_places is None else dict(self._graph_places)

    def _why_no_place(self, name: str) -> str:
        """The refusal for a name that cannot be answered, saying which of the two it is: the
        book has not arrived at all, or it has and the name is not in it."""
        if self._graph_places is None:
            return (
                f"no place {name!r} yet: the graph's book of places has not arrived from the"
                " laptop (is ros/laptop.sh vslam up, and has RTAB-Map published its first graph?)"
                " — a name is never answered from the old map's file"
            )
        return f"no such place: {name!r}"

    def _pose_now(self) -> dict[str, float]:
        """Where the cart stands: ``map -> base_link`` with the age of that edge (``age_s``);
        empty when nobody publishes it."""
        return self._tf_pose()

    def _tf_pose(self) -> dict[str, float]:
        """The cart's pose from ``map -> base_link``, with the age of that edge in seconds
        (``age_s``). Empty when nobody publishes it."""
        wait = TF_WAIT_S
        if self._tf is None:
            # Started on the first ask when the pose topic has not started it yet. Its buffer
            # starts empty, so this one lookup waits longer.
            self._tf, wait = TfLookup(self), TF_FIRST_WAIT_S
        transform = self._tf.transform(MAP_FRAME, BASE_FRAME, timeout_s=wait)
        if transform is None:
            return {}
        now = self.get_clock().now().nanoseconds * 1e-9
        return {
            "x": transform.transform.translation.x,
            "y": transform.transform.translation.y,
            "yaw_deg": math.degrees(yaw_of(transform.transform.rotation)),
            "age_s": max(0.0, now - stamp_seconds(transform.header.stamp)),
        }

    def pose_at(self, request: dict[str, Any]) -> dict[str, Any]:
        """``parent -> frame`` (one of :data:`POSE_AT_FRAMES`, from ``map`` unless ``odom`` is
        asked) at ``stamp``, from TF's history: x, y, z, the quaternion (qx, qy, qz, qw),
        yaw_deg, the transform's ``stamp``, this node's ``now`` and the stamp's ``age_s`` (now
        minus stamp). No stamp asks the newest there is, stamped and aged as ``where``'s. Stamps
        are seconds on this node's ROS clock, the Docker VM's, which the board's chrony follows
        (README "One clock"): the clock of every TF stamp and of the camera channel's records
        (camera_stream's board capture stamp). ``ok`` False names ``kind`` and ``reason``:
        ``frame``, ``parent``, ``too_old`` (past :data:`TF_HISTORY_S`, not looked up),
        ``ahead`` (the clocks disagree), or tf2's own (``Lookup``, ``Extrapolation``,
        ``Connectivity``, ``Timeout``) after waiting :data:`POSE_AT_WAIT_S`.

        SEVERAL AT ONCE (2026-10-15: the speaker gaze places every sighting and every voice in
        the map, a request an observation): ``frames`` (a list) and ``stamps`` (a list, null
        for the newest) ask every frame at every stamp in one request, answered as ``poses``,
        stamp by stamp and within one the frames in their order, each exactly the single answer
        above without its ``event``; the whole batch waits :data:`POSE_AT_WAIT_S` at most."""
        frames, stamps = request.get("frames"), request.get("stamps")
        if frames is None and stamps is None:
            single = self._poses(request, [request.get("frame", BASE_FRAME)],
                                 [request.get("stamp")])  # fmt: skip
            if "detail" in single:
                return single
            return {"event": "pose_at", **single["poses"][0]}
        frames = [request.get("frame", BASE_FRAME)] if frames is None else frames
        stamps = [request.get("stamp")] if stamps is None else stamps
        if not isinstance(frames, list) or not isinstance(stamps, list):
            return {"event": "error", "detail": "pose_at's frames and stamps are lists"}
        answer = self._poses(request, frames, stamps)
        if "detail" in answer:
            return answer
        return {"event": "pose_at", "parent": answer["parent"], "now": answer["now"],
                "poses": answer["poses"]}  # fmt: skip

    def _poses(
        self, request: dict[str, Any], frames: list[Any], stamps: list[Any]
    ) -> dict[str, Any]:
        """Every frame at every stamp (:meth:`pose_at`), within one wait: ``{"parent", "now",
        "poses"}``, or an ``error`` event for a stamp that is no number."""
        parent = str(request.get("parent", MAP_FRAME))
        asked: list[float | None] = []
        for value in stamps:
            try:
                asked.append(None if value is None else float(value))
            except (TypeError, ValueError):
                return {"event": "error", "detail": f"pose_at's stamp is seconds, not {value!r}"}
        now = self.get_clock().now().nanoseconds * 1e-9
        wait = POSE_AT_WAIT_S
        if self._tf is None:
            self._tf, wait = TfLookup(self), TF_FIRST_WAIT_S
        deadline = time.monotonic() + wait
        poses = [
            self._pose_one(str(frame), parent, stamp, now, deadline)
            for stamp in asked
            for frame in frames
        ]
        return {"parent": parent, "now": now, "poses": poses}

    def _pose_one(
        self, frame: str, parent: str, stamp: float | None, now: float, deadline: float
    ) -> dict[str, Any]:
        """One ``parent -> frame`` at ``stamp`` (:meth:`pose_at`), the lookup waiting until
        ``deadline`` (this node's monotonic clock) at most."""
        answer: dict[str, Any] = {"frame": frame, "parent": parent, "stamp": stamp, "now": now}
        if stamp is not None:
            answer["age_s"] = round(now - stamp, 3)
        if frame not in POSE_AT_FRAMES:
            known = " or ".join(POSE_AT_FRAMES)
            return answer | {"ok": False, "kind": "frame", "reason": f"frame is {known}"}
        if parent not in POSE_AT_PARENTS:
            known = " or ".join(POSE_AT_PARENTS)
            return answer | {"ok": False, "kind": "parent", "reason": f"parent is {known}"}
        if stamp is not None and now - stamp > TF_HISTORY_S:
            return answer | {
                "ok": False,
                "kind": "too_old",
                "reason": f"the stamp is {now - stamp:.1f} s old and TF keeps"
                f" {TF_HISTORY_S:g} s of history",
            }
        if stamp is not None and stamp - now > POSE_AT_AHEAD_S:
            return answer | {
                "ok": False,
                "kind": "ahead",
                "reason": f"the stamp is {stamp - now:.1f} s past this node's clock: the"
                " board's and this Mac's clocks disagree (ros/time.sh offset)",
            }
        when = RosTime() if stamp is None else RosTime.from_msg(stamp_from_seconds(stamp))
        wait = max(0.0, deadline - time.monotonic())
        tf = self._tf
        assert tf is not None  # made by _poses before the first lookup
        try:
            transform = tf.buffer.lookup_transform(
                parent, frame, when, timeout=RosDuration(seconds=wait)
            )
        except Exception as error:  # tf2's Lookup / Extrapolation / Connectivity / Timeout
            reason = str(error).strip()[:200] or type(error).__name__
            return answer | {"ok": False, "kind": tf_failure_kind(error), "reason": reason}
        if stamp is None:
            held = stamp_seconds(transform.header.stamp)
            answer["stamp"], answer["age_s"] = held, round(max(0.0, now - held), 3)
        t, q = transform.transform.translation, transform.transform.rotation
        return answer | {
            "ok": True,
            "x": t.x,
            "y": t.y,
            "z": t.z,
            "qx": q.x,
            "qy": q.y,
            "qz": q.z,
            "qw": q.w,
            "yaw_deg": math.degrees(yaw_of(q)),
        }

    def _ready(self, pose: dict[str, float] | None = None) -> Readiness:
        """May a goal start now (:class:`pepin.watch.GoalGate`): the age of map -> base_link, and
        whether this start of RTAB-Map is placed (flag ``start_needs_placement``). ``pose`` is a
        reading already taken by the caller, so the edge is not looked up twice."""
        edge = self._tf_pose() if pose is None else pose
        ready = self._gate.verdict(edge.get("age_s"))
        if ready.ready:
            placed = Preflight.placement(
                self._placement_standing(), asked=self._switches.on("start_needs_placement")
            )
            if not placed.ok:
                return Readiness(False, rule=BY_PLACEMENT, reason=placed.detail)
        return ready

    def pick_planner(self, name: str) -> dict[str, Any]:
        """Choose the planner, the controller that follows it (flag ``controller``) and the goal
        checker that ends the drive with that controller; all three, or none."""
        pair = PLANNERS.get(name.lower())
        if pair is None:
            return {"event": "error", "detail": f"planner must be one of {sorted(PLANNERS)}"}
        planner, rpp = pair
        controller, checker = FOLLOWERS.get(self._switches["controller"], (rpp, RPP_GOAL_CHECKER))
        self._planner_pick.publish(String(data=planner))
        self._follow(controller, checker)
        self.planner = name.lower()
        with contextlib.suppress(OSError):
            self._planner_path.write_text(f"{self.planner}\n")
        self.get_logger().info(f"planner {planner} with controller {controller} ({checker})")
        return {
            "event": "planner",
            "planner": planner,
            "controller": controller,
            "goal_checker": checker,
        }

    def _follow(self, controller: str, checker: str) -> None:
        """Hand the tree one controller and the goal checker that ends the drive with it (both
        latched: the tree reads them at its next tick, a running goal included)."""
        self._controller_pick.publish(String(data=controller))
        self._checker_pick.publish(String(data=checker))
        self.controller, self.goal_checker = controller, checker

    def _hand_over(self, x: float, y: float) -> dict[str, Any] | None:
        """Under a flag value that parks on another controller (PARKERS), once the cart is
        within ``park_distance_m`` of the goal (x, y): the parking pair to the tree, logged, and
        the event that says so; None otherwise, and always once handed over."""
        parker = PARKERS.get(self._switches["controller"])
        if parker is None or self.controller == parker[0]:
            return None
        pose = self._pose_now()
        if not pose:
            return None
        distance = math.hypot(pose["x"] - x, pose["y"] - y)
        if distance >= self._switches["park_distance_m"]:
            return None
        was = self.controller
        self._follow(*parker)
        self.get_logger().info(
            f"run {self._runs.run}: {distance:.2f} m from the goal, {was} -> {parker[0]}"
            f" ({parker[1]}) parks it"
        )
        return {"event": "handover", "from": was, "to": parker[0], "distance_m": round(distance, 2)}

    def _hand_back(self) -> None:
        """After a goal: the pair the ``controller`` flag names again, if a handover moved it."""
        planner = PLANNERS[self.planner][1]
        pair = FOLLOWERS.get(self._switches["controller"], (planner, RPP_GOAL_CHECKER))
        if (self.controller, self.goal_checker) != pair:
            was = self.controller
            self._follow(*pair)
            self.get_logger().info(f"goal over: {was} -> {pair[0]} ({pair[1]}) for the next one")

    def _on_switch(self, name: str, old: Any, new: Any) -> None:
        """A live flag changed: a new ``controller`` re-publishes the pick with its follower;
        ``face_events`` opens or closes the head server's door."""
        if name == "controller" and new != old:
            self.pick_planner(self.planner)
        if name == "face_events" and hasattr(self, "_drive_face"):
            self._face_switch(bool(new))

    def _face_switch(self, on: bool) -> None:
        """The face's door open (``face_events`` on) or closed."""
        if on and self._drive_face is None:
            self._drive_face = DriveFace(face_client())
            self.get_logger().info("face events on: the drive's moments go to the head server")
        elif not on and self._drive_face is not None:
            face, self._drive_face = self._drive_face, None
            face.close()
            self.get_logger().info("face events off")

    def _renew_face_lease(self) -> None:
        """The brain lease (pepin.head_server): this node is here; dropped until up."""
        face = self._drive_face
        if self._up and face is not None:
            face.lease()

    def cancel(self, send: bool = True) -> bool:
        """The operator's stop (the red button, a tool's cancel): the run under way stopped and
        a request waiting to start dropped; True when a run was going.

        Lets the handles go and counts the cancel, so a drive still on its way to Nav2 is never
        sent and a handle that arrives after it is cancelled at once (:meth:`_go`); the run
        stays this node's until its session has seen Nav2's answer and closed its tape. The
        end-of-drive pivot is cancelled whatever ``send`` says: no navigator owns it.
        ``send=False`` lets the drive's handle go without cancelling it: the caller cancels
        every goal on the navigators itself (:meth:`cancel_every_goal`).
        """
        with self._turn:
            run = self._current
            going = run is not None and not run.stopped
            if run is not None:
                run.stopped = True
            waiting, self._next = self._next, None
            if waiting is not None:
                waiting.dropped, waiting.cancelled_for = True, A_CANCEL
            handle, spin = self._let_go()
            self._cancelled_at = time.monotonic()
            self._turn.notify_all()
        if handle is not None and send:
            handle.cancel_goal_async()
        if spin is not None:
            spin.cancel_goal_async()
        return going or spin is not None

    def _let_go(self) -> tuple[Any, Any]:
        """Under the lock: the drive's and the spin's handles let go (returned, to be cancelled
        outside it) and the cancel counted."""
        handle, self._goal_handle = self._goal_handle, None
        spin, self._spin_handle = self._spin_handle, None
        self._cancels += 1
        return handle, spin

    # -- the one rule ----------------------------------------------------------------------

    @staticmethod
    def _source_of(request: dict[str, Any]) -> str | None:
        """The request's ``source`` (``tool`` when it names none); None for an unknown one."""
        source = str(request.get("source", SOURCES[0]))
        return source if source in SOURCES else None

    def _admit(self, run: Run, connection: socket.socket) -> bool:
        """THE ONE RULE for a request while a run is under way (the module's notes). ``run``
        becomes the run under way at once when none is; the same source asking for the same
        target as the run ahead (the one waiting to start, else the one under way unless
        stopped) is answered ``already`` with it; the gaze's turn is refused while a person's
        run is ahead; anything else cancels the run under way and drops one waiting, then
        waits (REPLACE_WAIT_S at most) for the run under way to end and starts, a ``replaced``
        event naming each request it displaced (``run.replaced``: it cancelled a run under
        way). False when answered here, nothing to drive; True when ``run`` is under way, and
        the caller ends it with :meth:`_over`."""
        deadline = time.monotonic() + REPLACE_WAIT_S
        answer: dict[str, Any] | None = None
        displaced: list[Run] = []
        handles: tuple[Any, Any] = (None, None)
        with self._turn:
            current, waiting = self._current, self._next
            if current is None and waiting is None:
                self._start(run)
                return True
            ahead = waiting
            if ahead is None and current is not None and not current.stopped:
                ahead = current
            if ahead is not None and ahead.same(run):
                detail = f"already under way: {ahead.words}" + (
                    "" if ahead.number is None else f" (run {ahead.number})"
                )
                answer = {"event": "already", **ahead.about(), "detail": detail}
            elif run.source == "gaze" and ahead is not None and ahead.source != "gaze":
                about = "" if ahead.number is None else f"run {ahead.number}: "
                answer = {
                    "event": "error",
                    "detail": f"a drive runs ({about}{ahead.label}): the gaze's turn waits",
                }
            else:
                if waiting is not None:
                    waiting.dropped, waiting.cancelled_for = True, run.label
                    displaced.append(waiting)
                if current is not None and not current.stopped:
                    current.stopped, current.cancelled_for = True, run.label
                    displaced.append(current)
                    handles = self._let_go()
                self._next = run
                self._turn.notify_all()  # a dropped request's session wakes
        if answer is not None:
            self._send(connection, answer)
            self.get_logger().info(f"{run.label}: {answer['detail']}")
            return False
        handle, spin = handles
        if handle is not None:
            handle.cancel_goal_async()
        if spin is not None:
            spin.cancel_goal_async()
        run.replaced = any(not other.dropped for other in displaced)
        for other in displaced:
            what = "dropped before it started" if other.dropped else f"run {other.number} cancelled"
            self.get_logger().info(f"{other.label}: {what} for {run.label}")
        with self._turn:
            while not run.dropped and self._current is not None:
                left = deadline - time.monotonic()
                if left <= 0.0:
                    break
                self._turn.wait(left)
            if run.dropped:
                why = _for(run).get("cancelled_for")
                detail = "cancelled before it started" + (f" for {why}" if why else "")
                answer = {"event": "error", "detail": detail, **_for(run)}
            elif self._current is not None:
                if self._next is run:
                    self._next = None
                late = self._current
                answer = {
                    "event": "error",
                    "detail": f"not started: {late.label} (run {late.number}) did not end within"
                    f" {REPLACE_WAIT_S:g} s of its cancel",
                }
            else:
                self._next = None
                self._start(run)
        if answer is not None:
            self._send(connection, answer)
            self.get_logger().warning(f"{run.label}: {answer['detail']}")
            return False
        for other in displaced:
            self._send(connection, {"event": "replaced", **other.about(),
                                    "cancelled_run": other.number,
                                    "started": not other.dropped})  # fmt: skip
        return True

    def _start(self, run: Run) -> None:
        """Under the lock: ``run`` is the run under way, any cancel from here on its own."""
        self._current = run
        run.cancels = self._cancels

    def _over(self, run: Run) -> None:
        """``run``'s session is over: no longer under way; a request waiting on it may start."""
        with self._turn:
            if self._current is run:
                self._current = None
            self._turn.notify_all()

    def _settle(self, deadline: float) -> None:
        """After a cancelled run, until ``deadline``: wait for Nav2's latched status lists to
        show its goal ended (they come apart from the action's result)."""
        while any(self._nav_active.values()) and time.monotonic() < deadline:
            time.sleep(SETTLE_POLL_S)

    def cancel_every_goal(self) -> dict[str, dict[str, Any]]:
        """Cancel every goal on both navigators, whoever sent it; what each said, per action.

        A zero goal id and stamp in ``CancelGoal`` means all goals (goto_ros.py's ``cancel_all``,
        asked from here). Both requests go out before either answer is awaited, and one
        :data:`CANCEL_CONFIRM_S` deadline covers the whole cancel. Each entry is ``outcome`` in
        goto_ros.py's words, plus ``cancelling`` (how many goals) when the navigator answered.
        """
        deadline = time.monotonic() + CANCEL_CONFIRM_S
        said: dict[str, dict[str, Any]] = {}
        pending: dict[str, Any] = {}
        for index, (action, client) in enumerate(self._cancel_clients.items()):
            share = max(deadline - time.monotonic(), 0.0) / (len(self._cancel_clients) - index)
            if not client.service_is_ready() and not client.wait_for_service(
                timeout_sec=share / 2.0
            ):
                said[action] = {"outcome": "no server answered"}
                continue
            pending[action] = client.call_async(CancelGoal.Request())
        for action, future in pending.items():
            answer = self._wait(future, max(deadline - time.monotonic(), 0.0))
            if answer is None:
                said[action] = {
                    "outcome": f"NOT confirmed in {CANCEL_CONFIRM_S:.0f} s — use ros/stop.sh"
                }
                continue
            said[action] = {
                "outcome": cancel_outcome(int(answer.return_code)),
                "cancelling": len(answer.goals_canceling),
            }
        self.get_logger().info(
            "cancel every goal: "
            + "; ".join(f"{action} {said[action]['outcome']}" for action in self._cancel_clients)
        )
        return {action: said[action] for action in self._cancel_clients}

    def _go(self, request: dict[str, Any], connection: socket.socket) -> None:
        """Send one goal and stream its progress until it ends or the caller hangs up. A goal
        that would be refused (no such place, the gate, Nav2 down, an unknown source) is
        refused before the one rule (:meth:`_admit`) can cancel anything for it."""
        target = self._target_of(request)
        if target is None:
            detail = self._why_no_place(str(request.get("place", "")))
            self._send(connection, {"event": "error", "detail": detail})
            return
        x, y, yaw_deg, name = target
        source = self._source_of(request)
        if source is None:
            detail = f"source must be one of {', '.join(SOURCES)}, not {request.get('source')!r}"
            self._send(connection, {"event": "error", "detail": detail})
            return
        ready = self._ready()
        if not ready.ready:
            self._send(connection, {"event": "error", "detail": ready.reason})
            self.get_logger().warning(f"goal refused: {ready.reason}")
            return
        if not self._client.wait_for_server(timeout_sec=5.0):
            self._send(connection, {"event": "error", "detail": "Nav2 is not up"})
            return
        goal = NavigateToPose.Goal()
        goal.pose = self._pose_msg(x, y, yaw_deg)
        started = time.monotonic()
        feedback: dict[str, Any] = {}
        words = f"go {name}" if name else f"go ({x:.2f}, {y:.2f}, {yaw_deg:.0f} deg)"
        run = Run(source, "go", (x, y, yaw_deg), words)
        if not self._admit(run, connection):
            return
        try:
            self._drive(run, goal, name, started, feedback, connection)
        finally:
            self._over(run)

    def _drive(
        self,
        run: Run,
        goal: Any,
        name: str | None,
        started: float,
        feedback: dict[str, Any],
        connection: socket.socket,
    ) -> None:
        """The admitted ``go`` (:meth:`_go`): taped, sent, reported (``arrival`` the moment
        Nav2 reports the goal reached, the pivot after it, ``done`` last)."""
        x, y, yaw_deg = run.target
        cancels = run.cancels  # any cancel from its admission on is this drive's
        record = self.start_recording(name or f"{x:.0f}_{y:.0f}")
        run.number = None if record is None else self._runs.run
        mode = str(self._switches["controller"])
        parks = (
            f": {PARKERS[mode][0]} within {self._switches['park_distance_m']:.2f} m"
            if mode in PARKERS
            else ""
        )
        self.get_logger().info(
            f"run {self._runs.run}: planner {PLANNERS[self.planner][0]}, controller "
            f"{self.controller} ({mode}{parks}) -> {name or 'coordinates'}"
            f" ({x:.2f}, {y:.2f}, {yaw_deg:.0f} deg), asked by {run.source}"
        )
        face = self._drive_face  # this drive's, whatever the flag does meanwhile
        try:
            if self._cancelled_since(cancels):  # the recorder's wait is up to 8 s of it
                detail = "cancelled before Nav2 had the goal"
                self._send(connection, {"event": "error", "detail": detail, **_for(run)})
                return
            handover = self._hand_over(x, y)  # a goal that starts within the parking distance
            send = self._client.send_goal_async(
                goal,
                lambda f: feedback.update(
                    distance=f.feedback.distance_remaining,
                    recoveries=f.feedback.number_of_recoveries,
                ),
            )
            handle = self._wait(send, 10.0)
            if handle is None or not handle.accepted:
                if face is not None:
                    face.refused()
                self._send(connection, {"event": "error", "detail": "the goal was refused"})
                return
            with self._lock:
                late = self._cancels != cancels
                if not late:
                    self._goal_handle = handle
            if late:  # the cancel came while Nav2 was taking the goal: it is cancelled now
                handle.cancel_goal_async()
                detail = "cancelled while Nav2 took the goal: cancelled"
                self._send(connection, {"event": "error", "detail": detail, **_for(run)})
                return
            self._send(
                connection,
                {
                    "event": "accepted",
                    "run": self._runs.run,
                    "source": run.source,
                    "planner": PLANNERS[self.planner][0],
                    "controller": self.controller,
                    "mode": mode,
                    **({"parks_within_m": self._switches["park_distance_m"]} if parks else {}),
                    "pose": "tf",
                    # early: a drive that never reaches "done" is still fetched
                    "recording": None if record is None else str(record),
                    "place": name,
                    "x": x,
                    "y": y,
                    "yaw_deg": yaw_deg,
                    "sent_in_ms": round((time.monotonic() - started) * 1000),
                },
            )
            if face is not None:
                face.accepted()
            if handover is not None:
                self._send(connection, handover)
            result_future = handle.get_result_async()
            last = 0.0
            stopped_at: float | None = None
            answered = False
            while rclpy.ok() and not (answered := result_future.done()):
                time.sleep(0.05)  # the node's own spin serves the action; this thread only reports
                if self._unanswered(stopped_at):
                    break
                if stopped_at is None and self._cancelled_since(cancels):
                    stopped_at = time.monotonic()
                handover = self._hand_over(x, y)
                if handover is not None:
                    self._send(connection, handover)
                if face is not None and feedback:
                    face.progress(int(feedback.get("recoveries") or 0))
                now = time.monotonic()
                if feedback and now - last > 1.0:
                    last = now
                    self._send(
                        connection, {"event": "feedback", "t": round(now - started, 1), **feedback}
                    )
            with self._lock:
                self._goal_handle = None
            outcome = result_future.result() if answered else None
            status = getattr(outcome, "status", 0) if outcome else 0
            # Now the heading, taped: only after a drive that ended on position alone (the RPP
            # pair's checker). Every other pair, a parking MPPI included, turns to the heading
            # itself and was held to it by the yaw-checking goal checker.
            ended_on_position = self.goal_checker == RPP_GOAL_CHECKER
            pivots = status == 4 and ended_on_position and not self._cancelled_since(cancels)
            if status == 4:  # reached: said at once, before the pivot and the tape's close
                self._send(
                    connection,
                    {
                        "event": "arrival",
                        "run": self._runs.run,
                        "status": int(status),
                        "seconds": round(time.monotonic() - started, 1),
                        "arrival": self._pose_now(),
                        "pivot": pivots,
                    },
                )
            if pivots:
                self._pivot_to(yaw_deg, connection, cancels)
            if face is not None:
                face.done(int(status))
            self.stop_recording()  # closed before the answer: the caller fetches it on reading
            self._send(
                connection,
                {
                    "event": "done",
                    "run": self._runs.run,
                    "planner": self.planner,
                    "status": int(status),
                    "seconds": round(time.monotonic() - started, 1),
                    "arrival": self._pose_now(),
                    "recording": None if record is None else str(record),
                    **_for(run),
                    **({} if answered else {"detail": UNANSWERED}),
                },
            )
        finally:  # a refused goal or a broken connection must not leave a recorder running
            self.stop_recording()
            self._hand_back()
            if face is not None:
                face.abandoned()  # nothing when the drive's end was shown

    def _cancelled_since(self, cancels: int) -> bool:
        """Whether a cancel came after the one counted as ``cancels``."""
        with self._lock:
            return self._cancels != cancels

    @staticmethod
    def _unanswered(stopped_at: float | None) -> bool:
        """A run stopped at ``stopped_at`` whose goal Nav2 has not let go of in STOPPED_WAIT_S:
        its session stops waiting for the result."""
        return stopped_at is not None and time.monotonic() - stopped_at > STOPPED_WAIT_S

    def _pivot_to(self, yaw_deg: float, connection: socket.socket, cancels: int) -> None:
        """Turn in place to the mark's heading once the drive has met the position.

        The controller that can reverse (RPP, FollowPathRS) cannot rotate in place, and at a
        mark Hybrid-A* writes a 10 cm cusp plan whose carrot sits straight ahead: three printer
        approaches spent 35-61 s shuttling for the last 30 degrees. The drive therefore ends on
        position alone and the behaviour server's Spin does the heading: 40 degrees in ~1.5 s.
        """
        here = self._pose_now().get("yaw_deg")
        if here is None:  # nothing answered about the pose: a blind spin is worse than no spin
            self._send(connection, {"event": "pivot", "status": 0, "detail": "no pose to turn by"})
            return
        residual = heading_residual_deg(yaw_deg, here)
        if abs(residual) <= PIVOT_TOLERANCE_DEG:
            return
        event: dict[str, Any] = {"event": "pivot", "residual_deg": round(residual, 1)}
        handle, refusal = self._start_spin(residual, PIVOT_ALLOWANCE_S, cancels)
        if handle is None:
            self._send(connection, event | {"status": 0, "detail": refusal})
            return
        outcome = self._wait(handle.get_result_async(), PIVOT_ALLOWANCE_S + 5.0)
        self._spin_over(handle)
        status = getattr(outcome, "status", 0) if outcome else 0
        after = heading_residual_deg(yaw_deg, self._pose_now().get("yaw_deg", here))
        self._send(connection, event | {"status": int(status), "after_deg": round(after, 1)})
        self.get_logger().info(f"pivot {residual:+.0f} deg: status {status}, {after:+.0f} deg left")

    def _start_spin(
        self, turn_deg: float, allowance_s: float, cancels: int, feedback: Any = None
    ) -> tuple[Any, str]:
        """One goal on the behaviour server's Spin, ``turn_deg`` (+ left) within ``allowance_s``,
        held in ``_spin_handle`` so a cancel stops it (:meth:`cancel`: no navigator owns it);
        ``feedback`` hears Nav2's progress. The handle, or None and why not: no spin behaviour,
        refused, or a cancel after the one counted as ``cancels`` (the late handle is cancelled
        here)."""
        if not self._spin.wait_for_server(timeout_sec=2.0):
            return None, "no spin behaviour"
        goal = Spin.Goal()
        goal.target_yaw = math.radians(turn_deg)
        goal.time_allowance = Duration(sec=math.ceil(allowance_s))
        handle = self._wait(self._spin.send_goal_async(goal, feedback), 5.0)
        if handle is None or not handle.accepted:
            return None, "the spin was refused"
        with self._lock:  # a cancel from here on stops the spin (cancel); before, it is done here
            late = self._cancels != cancels
            if not late:
                self._spin_handle = handle
        if late:
            handle.cancel_goal_async()
            return None, "cancelled"
        return handle, ""

    def _spin_over(self, handle: Any) -> None:
        """A spin's handle let go, unless a cancel already took it."""
        with self._lock:
            if self._spin_handle is handle:
                self._spin_handle = None

    def _spin_in_place(self, request: dict[str, Any], connection: socket.socket) -> None:
        """``spin``: the cart turned in place by ``deg`` (+ left) with one goal on the behaviour
        server's Spin (:meth:`_start_spin`, as the end-of-drive pivot sends it), answered and
        recorded like a drive: ``accepted`` (the run, the pose it starts from), ``feedback`` each
        second (Nav2's own count of the turn), ``arrival`` the moment Nav2 reports it reached,
        ``done`` once its tape is closed (both: Nav2's status, the seconds, the pose after it
        and the turn measured on map -> base_link: :meth:`turn_measured`). A cancel stops it,
        and so does another request by the one rule (:meth:`_admit`). No map pose is needed:
        the behaviour server turns on odometry (its local_frame); without one the turn is not
        measured, and a drive's gate does not apply. Anyone's goal on a navigator refuses it
        (two goals would share the wheels), once a run it cancelled has left Nav2's lists."""
        if not self._switches.on("spin_requests"):
            detail = "spin requests are off (flag spin_requests of the goal server)"
            self._send(connection, {"event": "error", "detail": detail})
            return
        try:
            deg = float(request["deg"])
        except (KeyError, TypeError, ValueError):
            self._send(connection, {"event": "error", "detail": "spin needs deg: degrees, + left"})
            return
        if not math.isfinite(deg) or deg == 0.0 or abs(deg) > SPIN_MAX_DEG:
            detail = f"deg must be non-zero and within +-{SPIN_MAX_DEG:.0f}, not {deg:g}"
            self._send(connection, {"event": "error", "detail": detail})
            return
        source = self._source_of(request)
        if source is None:
            detail = f"source must be one of {', '.join(SOURCES)}, not {request.get('source')!r}"
            self._send(connection, {"event": "error", "detail": detail})
            return
        run = Run(source, "spin", (deg,), f"spin {deg:+g} deg")
        asked = time.monotonic()
        if not self._admit(run, connection):
            return
        try:
            if run.replaced:  # the cancelled goal leaves Nav2's status lists after its result
                self._settle(asked + REPLACE_WAIT_S)
            if any(self._nav_active.values()):  # anyone's goal: two would share the wheels
                detail = "a goal runs on Nav2: cancel first"
                self._send(connection, {"event": "error", "detail": detail})
                return
            self._turn_in_place(run, connection)
        finally:
            self._over(run)

    def _turn_in_place(self, run: Run, connection: socket.socket) -> None:
        """The admitted ``spin`` (:meth:`_spin_in_place`): taped, sent, reported."""
        deg = run.target[0]
        cancels = run.cancels  # any cancel from its admission on is this spin's
        started = time.monotonic()
        allowance = PIVOT_ALLOWANCE_S * max(1.0, abs(deg) / 180.0)
        before = self._pose_now()
        record = self.start_recording(f"spin_{'left' if deg > 0 else 'right'}_{abs(deg):.0f}")
        run.number = None if record is None else self._runs.run
        self.get_logger().info(
            f"run {self._runs.run}: spin {deg:+.0f} deg in place, asked by {run.source}"
        )
        face = self._drive_face  # this spin's, whatever the flag does meanwhile
        spun: dict[str, float] = {}
        handle: Any = None
        try:
            if self._cancelled_since(cancels):  # the recorder's wait is up to 8 s of it
                detail = "cancelled before Nav2 had the spin"
                self._send(connection, {"event": "error", "detail": detail, **_for(run)})
                return
            handle, refusal = self._start_spin(
                deg,
                allowance,
                cancels,
                lambda f: spun.update(deg=math.degrees(f.feedback.angular_distance_traveled)),
            )
            if handle is None:
                if face is not None and refusal != "cancelled":
                    face.refused()
                detail = f"not turning: {refusal}"
                self._send(connection, {"event": "error", "detail": detail, **_for(run)})
                return
            self._send(
                connection,
                {
                    "event": "accepted",
                    "run": self._runs.run,
                    "source": run.source,
                    "deg": deg,
                    "pose": "tf" if before else "none",
                    "recording": None if record is None else str(record),
                    **{k: before[k] for k in ("x", "y", "yaw_deg") if k in before},
                    "sent_in_ms": round((time.monotonic() - started) * 1000),
                },
            )
            if face is not None:
                face.accepted()
            result_future = handle.get_result_async()
            deadline = time.monotonic() + allowance + 5.0  # the pivot's patience
            last = 0.0
            stopped_at: float | None = None
            while rclpy.ok() and not result_future.done() and time.monotonic() < deadline:
                time.sleep(0.05)  # the node's own spin serves the action; this thread only reports
                if self._unanswered(stopped_at):
                    break
                if stopped_at is None and self._cancelled_since(cancels):
                    stopped_at = time.monotonic()
                now = time.monotonic()
                if spun and now - last > 1.0:
                    last = now
                    progress = {"t": round(now - started, 1), "spun_deg": round(spun["deg"], 1)}
                    self._send(connection, {"event": "feedback", **progress})
            self._spin_over(handle)
            answered = result_future.done()
            outcome = result_future.result() if answered else None
            status = int(getattr(outcome, "status", 0)) if outcome else 0
            after = self._pose_now() if before else {}
            end: dict[str, Any] = {
                "run": self._runs.run,
                "status": status,
                "seconds": round(time.monotonic() - started, 1),
                "arrival": after,
            }
            if "deg" in spun:
                end["spun_deg"] = round(spun["deg"], 1)
            if not answered and stopped_at is not None:
                end["detail"] = UNANSWERED
            elif not answered:
                end["detail"] = f"no result from the behaviour server in {allowance + 5.0:.0f} s"
            end |= self.turn_measured(deg, before, after, status)
            if answered and status == 4:  # reached: said at once, before the tape's close
                self._send(connection, {"event": "arrival", **end})
            if face is not None:
                face.done(status)
            self.stop_recording()  # closed before the answer: the caller fetches it on reading
            done: dict[str, Any] = {
                "event": "done",
                **end,
                "seconds": round(time.monotonic() - started, 1),
                "recording": None if record is None else str(record),
                **_for(run),
            }
            self._send(connection, done)
            self.get_logger().info(
                f"spin {deg:+.0f} deg: status {status} in {end['seconds']:.1f} s"
                + (f", turned {end['turned_deg']:+.0f} deg" if "turned_deg" in end else "")
            )
        finally:  # a refused spin or a broken connection must not leave a recorder running
            self.stop_recording()
            if face is not None:
                face.abandoned()  # nothing when the spin's end was shown
            if handle is not None:
                self._spin_over(handle)

    @staticmethod
    def turn_measured(
        deg: float, before: dict[str, float], after: dict[str, float], status: int
    ) -> dict[str, Any]:
        """A spin of ``deg`` measured on map -> base_link: ``turned_deg`` (the heading's change,
        unwrapped toward the ask) and ``short_deg`` (the heading still to go, + left), and a
        ``detail`` when a spin Nav2 calls succeeded is off by more than the goal checker's
        tolerance (:data:`PIVOT_TOLERANCE_DEG`). Empty without a heading at both ends."""
        if "yaw_deg" not in before or "yaw_deg" not in after:
            return {}
        short = heading_residual_deg(before["yaw_deg"] + deg, after["yaw_deg"])
        measured: dict[str, Any] = {"turned_deg": round(deg - short, 1)}
        measured["short_deg"] = round(short, 1)
        if status == 4 and abs(short) > PIVOT_TOLERANCE_DEG:
            way = "short" if short * deg > 0 else "past the ask"
            measured["detail"] = (
                f"turned {deg - short:+.0f} of {deg:+.0f} deg by the map's heading:"
                f" {abs(short):.0f} deg {way}, beyond the goal checker's"
                f" {PIVOT_TOLERANCE_DEG:g} deg"
            )
        return measured

    def _target_of(self, request: dict[str, Any]) -> tuple[float, float, float, str | None] | None:
        """The goal asked for: a named place, or plain coordinates."""
        if "place" in request:
            place = self.places().get(str(request["place"]))
            if place is None:
                return None
            return place["x"], place["y"], place["yaw_deg"], str(request["place"])
        if "x" in request and "y" in request:
            return (
                float(request["x"]),
                float(request["y"]),
                float(request.get("yaw_deg", 0.0)),
                None,
            )
        return None

    # ---- the one pose topic --------------------------------------------------------------------
    def _start_pose_topic(self) -> None:
        """Wire ``/pose``: the cart's pose in ``map``, five times a second, out of the TF
        listener this node already owns.

        The listener is built HERE, in the constructor, rather than on the first ask: this timer
        wants it from the first tick, and it is the one listener the stack keeps — a TF listener
        is a subscription to the whole /tf stream, deserialised in Python whatever the reader
        wanted out of it, and two of them (this node's and the tape recorder's) cost a board
        measured at 252 % about 56 % of a core (2026-09-22). It starts a spin thread for this
        node, which is why ``_up`` guards the tick.
        """
        self._pose_pub = self.create_publisher(PoseStamped, POSE_TOPIC, 5)
        if self._tf is None:
            self._tf = TfLookup(self)
        self.create_timer(1.0 / POSE_HZ, self._publish_pose)

    def _publish_pose(self) -> None:
        """One pose onto ``/pose``, or nothing at all: this is a relay of an edge, never a
        measurement of its own.

        Stamped with the TRANSFORM's stamp and not with now — a pose stamped "now" is how a
        stale reading becomes a fresh lie (2026-09-19). The edge is read from whatever is in the
        buffer and never waited for: this runs on the executor thread, beside the socket's own
        callbacks, and a missing edge is simply no message.
        """
        if not self._up or self._tf is None:
            return
        transform = self._tf.transform(MAP_FRAME, BASE_FRAME, None, 0.0)
        if transform is None:
            return
        message = PoseStamped()
        message.header.stamp = transform.header.stamp
        message.header.frame_id = MAP_FRAME
        message.pose.position.x = transform.transform.translation.x
        message.pose.position.y = transform.transform.translation.y
        message.pose.position.z = transform.transform.translation.z
        message.pose.orientation = transform.transform.rotation
        self._pose_pub.publish(message)

    def _pose_msg(self, x: float, y: float, yaw_deg: float) -> PoseStamped:
        """A goal pose in the map frame."""
        message = PoseStamped()
        message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.position.x = x
        message.pose.position.y = y
        message.pose.orientation.z = math.sin(math.radians(yaw_deg) / 2.0)
        message.pose.orientation.w = math.cos(math.radians(yaw_deg) / 2.0)
        return message


def main() -> None:
    # Through node_kit.spin_main like the other nodes: close() (the tape) first, then the TF
    # listener's non-daemon thread — with its own spin this node never exited on SIGINT — and
    # kill -USR2 prints its stacks (its main thread burns 43-46 % of an A53 core, 2026-09-24).
    spin_main(GoalServer)


if __name__ == "__main__":
    main()
