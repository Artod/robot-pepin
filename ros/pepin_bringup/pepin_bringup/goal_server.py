"""The robot waits for orders instead of being started for each one.

Every goal used to boot its own client on the board: ssh, docker exec, import rclpy, import the
Nav2 commander, build a node, discover the action server — 8 to 15 seconds before the wheels
could move, paid again for every command (measured 2026-09-08). This node is already running,
already connected to Nav2 and already holding the places book, so a command costs a socket write.

It speaks JSON lines on a TCP port (like the base and ToF servers), one connection at a time:

    {"cmd": "go", "place": "printer"}      {"cmd": "go", "x": -11.4, "y": 0.8, "yaw_deg": 140}
    {"cmd": "cancel"}                      {"cmd": "where"}
    {"cmd": "mark", "name": "printer"}     {"cmd": "places"}

and answers with one JSON line per event: accepted, feedback, arrival, done. It also owns the
run's recording: it starts one when a goal starts and closes it when the goal ends, so a
recording can no longer outlive its run.

WHERE THE POSE COMES FROM. On a saved map the scan-matching tracker is the answer to both "where
am I" and "may I drive": it serves ``/where_am_i`` and publishes ``/localization_fit``. In online
SLAM that node does not run at all — RTAB-Map owns the pose on the laptop and the board's
slam_frame only broadcasts its correction — so this node reads ``map -> base_link`` from TF
instead and judges the goal by how fresh that edge is (:class:`pepin.watch.GoalGate`). The
``tf_pose`` flag is that fallback: off, the node is the old one, which asked the tracker and
refused every goal in SLAM mode with "the tracker is not up" (2026-09-13 14:05).

SINCE 2026-09-22 THAT FALLBACK IS THE SHIPPED PATH. ``PEPIN_LOCALIZER=rtabmap`` (the ``localizer``
parameter, from pepin.deployment.localizer) means the laptop's RTAB-Map owns ``map -> odom`` and no
tracker runs anywhere: there is nobody to ask for a pose, no fit and no sigma, so ``where`` answers
``"pose": "tf"`` with no ``fit`` in it and a goal is judged by how fresh ``map -> base_link`` is.
``PEPIN_LOCALIZER=tracker`` is the stack described below, unchanged.

WHAT SAYS THE SLAM HALF IS STILL THERE. Not that transform: slam_frame re-broadcasts the LAST
correction at 10 Hz with a fresh stamp, so ``map -> base_link`` stays milliseconds old with the
laptop shut down — the gate would pass and Nav2, whose costmaps read the same fresh edge, would
never time out either. The correction itself is the pulse (``/map_odom``, published between
graphs too), so this node listens to it, refuses a goal when it has stopped and cuts a running
drive when it stops mid-way: the SLAM analogue of the blind-drive watch, under the
``correction_watch`` flag.

WHO WATCHES THE CORRECTION ITSELF. When ``map -> odom`` steps, the marks in Nav2's local costmap
were laid where the cart used to be, and nothing else takes them back. The lidar tracker used to
empty that grid on such a step while it owned the edge; RTAB-Map owns it now, so the watch sits
here — a 5 Hz read of the edge into :class:`pepin.watch.JumpClear`, one asynchronous clear per
jump, under the ``jump_clear`` flag, which ships OFF (nobody has yet watched RTAB-Map's own
corrections with it).

AND BECAUSE IT ALREADY PARSES ``/tf``, IT IS THE BOARD'S ONE LISTENER. A TF listener is a
subscription to the whole stream — RTAB-Map's ``map -> odom`` at 20 Hz, the board's
``odom -> base_link`` at 50 Hz, the statics — deserialised in Python whatever one pose the reader
wanted out of it, and two of them ran on a 4-core A53: this one and the tape recorder's. So this
node republishes the pose it reads as ``/pose`` (PoseStamped in ``map``, 5 Hz, stamped with the
transform's own stamp) under the ``pose_topic`` flag, and pepin_bringup.run_recorder subscribes to
that instead of running a listener of its own (its ``loc_from`` flag). The topic is not published
at all where a tracker runs — ``/tracker_pose`` is that topic with a covariance on it — nor on a
split stack, where this node is the laptop's and the recorder is the board's: a pose that crossed
the WiFi to be written to a tape is what putting that recorder on the board prevents.
"""

from __future__ import annotations

import contextlib
import json
import math
import socket
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, TransformStamped
from lifecycle_msgs.srv import ChangeState, GetState
from nav2_msgs.action import NavigateToPose, Spin
from nav2_msgs.srv import ClearEntireCostmap
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float32, Header, String
from std_srvs.srv import Trigger

from pepin.deployment import (
    BOARD_NAV_NODES,
    DEFAULT_LOCALIZER,
    HEARTBEAT_HZ,
    HEARTBEAT_TOPIC,
    next_transition,
    runs_here,
)
from pepin.flags import Flag, FlagSet
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
    CORRECTION_FRESH_S,
    DRIVE_FIT,
    DRIVE_SIGMA_M,
    LOST_SIGMA_M,
    SIGMA_TOPIC,
    TF_FRESH_S,
    BlindDriveWatch,
    Correction,
    GoalGate,
    JumpClear,
    Readiness,
    Sigma,
)
from pepin_bringup.msgs import stamp_seconds, yaw_of
from pepin_bringup.node_kit import Switches, TfLookup

PORT = 3337
GOOD_FIT = DRIVE_FIT  # below this the robot is told to find itself before it drives (pepin.watch)
# The planner to select, and the controller that follows it. One controller now: the lattice
# planner no longer expands in reverse, so there is nothing a reversing controller would add.
PIVOT_TOLERANCE_DEG = 11.5  # the goal checker's 0.20 rad: below this the heading is met
PIVOT_ALLOWANCE_S = 15.0
RECORDER_PATIENCE_S = (
    8.0  # the recorder answers over the bridge; 3 s once named a drive after the previous tape
)
BRINGUP_ROUND_S = 10.0  # a lifecycle query or transition that has not answered by then is abandoned

MAP_FRAME = "map"
BASE_FRAME = "base_link"
ODOM_FRAME = "odom"
CORRECTION_TOPIC = "/map_odom"  # the SLAM half's pulse; the board's slam_frame reads it too
# Nav2's own service for emptying the grid the controller steers on, and the watch's rules over it
# (pepin.watch.JumpClear). The relocalizer holds the same two names for the same call: it made it
# while IT owned map -> odom, and under World R that edge is RTAB-Map's, so the call moved here to
# the one node on the board that is awake between goals. Spelled out rather than imported from
# that module: a node does not import another node.
CLEAR_LOCAL_COSTMAP = "/local_costmap/clear_entirely_local_costmap"
CLEAR_MIN_GAP_S = 1.0
JUMP_WATCH_HZ = 5.0  # how often map -> odom is read for a step: twice Nav2's own control period
# THE BOARD'S ONE POSE TOPIC (flag pose_topic). This node already parses /tf for the pose; every
# other node on the board that wants it reads this instead of starting a second listener
# (pepin_bringup.run_recorder's loc_from). 5 Hz is the rate the tape thinned its `loc` rows to
# anyway (RunRecorder.LOC_TF_PERIOD_S) and the local costmap's own update_frequency.
POSE_TOPIC = "/pose"
POSE_HZ = 5.0
TF_WAIT_S = 0.3  # how long a pose lookup waits for the edge: the drive thread asks, not a callback
TF_FIRST_WAIT_S = 2.0  # ...and the first one waits for the listener's buffer to fill at all
TRACKER_WAIT_S = 1.0  # the one probe for "is there a tracker at all" — its relocalise service

# The live flags (CLAUDE.md rule 19); their state is printed in the node's start line.
FLAGS = FlagSet(
    Flag(
        "tf_pose",
        True,
        description="where no tracker answers, the cart's pose is read from TF (map ->"
        " base_link) and a goal is judged by how fresh that edge is; off, only the tracker is"
        " ever asked",
        why="off, this node refused every goal of the first online-SLAM session — 'the tracker"
        " is not up' (2026-09-13 14:05), the goals driven by publishing /goal_pose by hand,"
        " which is Nav2 without a run, a tape or a verdict. In SLAM mode there IS no tracker:"
        " RTAB-Map owns the pose and pepin_bringup.slam_frame re-broadcasts its correction as"
        f" map -> odom at 10 Hz, so {TF_FRESH_S:.1f} s without a transform is ten missed"
        " broadcasts, not jitter. The known-map modes are untouched: there the tracker answers"
        " first and its fit decides, exactly as before",
        on_when="on in online SLAM, and anywhere else the pose is owned by something that"
        " publishes map -> base_link instead of a fit",
        off_when="to have a stack without a tracker refuse goals outright again — the old"
        " behaviour, and the honest one where a fit is the only evidence trusted",
    ),
    Flag(
        "correction_watch",
        True,
        description="where no tracker answers, the SLAM correction (/map_odom) must be arriving"
        " for a goal to start, and a drive is cut when it stops; off, the age of map ->"
        " base_link is the only evidence read",
        why="map -> base_link is no evidence that the SLAM half is alive: slam_frame"
        " re-broadcasts the LAST correction at 10 Hz with a fresh stamp, so with the laptop shut"
        " down the edge is still 0.1 s old, the gate passes, and Nav2 — whose costmaps read that"
        " same edge against a 0.3 s tolerance — does not abort either. The cart would drive a"
        " map that stopped growing, on dead reckoning, with nothing to notice. The correction is"
        " a 10 Hz pulse whatever the graph does (pepin_bringup.rtabmap_frame publishes between"
        f" optimisations too), so {CORRECTION_FRESH_S:.1f} s of silence is twenty missed messages"
        " over the bridge, not a hiccup",
        on_when="in online SLAM, where the pose is owned by a machine on the other side of the"
        " bridge",
        off_when="when this node cannot hear /map_odom in a stack that is otherwise healthy —"
        " 'ros/go.sh where' prints 'correction_s' where one has ever landed, and prints none at"
        " all in that case; the drive then rests on the transform alone, as it did before",
    ),
    Flag(
        "sigma_gate",
        True,
        description="a goal starts, and a running drive is cut, on the tracker's fused"
        " uncertainty (/localization/sigma); off, on its scan-to-map fit as before",
        why="a fit is ONE SENSOR'S metric — the share of one lidar revolution's beams that landed"
        " on the map — and it says nothing about a pose the camera is holding. On a camera-only"
        " drive it is 0.00 by construction, and every rule built on it read a healthy tracker as"
        f" lost (2026-09-15). The sigma comes out of the fusion itself, so {DRIVE_SIGMA_M:.2f} m"
        f" to start and {LOST_SIGMA_M:.2f} m to cut mean the same thing whichever source spoke —"
        " and it goes on growing along the odometry when none does, which a fit never did",
        on_when="always on a stack whose tracker publishes the topic; a board that does not is"
        " judged by its fit by itself, with no flag to set",
        off_when="to put the fit rules back for a comparison, or if a sigma ever refuses drives"
        " the cart is plainly fit for",
    ),
    Flag(
        "places_from_the_file",
        False,
        description="before the graph's book of places has been heard, a name is answered from"
        " the yaml beside the map (coordinates of the frozen-grid era); off, a name is refused"
        " until the book arrives, with that reason",
        why="that file holds coordinates of a frame that no longer exists, and the board keeps its"
        " own stale copy (the sync excludes it). Twice a name was answered from it and sent the"
        " cart at a point outside the map: `home` -> (-9.39, +2.53) on 2026-09-19, `printer` ->"
        " (-11.38, +0.77) on 2026-09-21, 2.1 s after RTAB-Map's first graph of a cold start of"
        " both halves — the planner said 'outside bounds', the behaviour tree ran 33 recoveries"
        " in 28 s and backed the cart into a sofa. A refusal costs a second try a minute later",
        on_when="only on a robot driven without the laptop's graph at all, on the old frozen map",
        off_when="always under World R: a place rides a graph node, and only the graph can say"
        " where that node is now",
    ),
    Flag(
        "start_on_a_known_pose",
        True,
        description="where the tracker publishes a sigma, a goal is refused for the pose's sake"
        " only when there is NO pose — nothing has ever corrected it, or the sigma stopped"
        f" arriving; off, a drive starts under {DRIVE_SIGMA_M:.2f} m and anything over it buys a"
        " whole-map search first, as before",
        why="2026-09-19, camera-only at the bookshelf: parked close to it the camera recognises"
        " nothing (PnP 0 of 20 inliers), so no word arrives and the belief grows along the"
        " odometry — 0.26 m, a pose the cart plainly had. The old rule refused the goal and sent"
        " it to _find_myself, which is a whole-map LIDAR search judged by the fit, and with the"
        " lidar out of the tracker's sources that fit is 0.00 by construction: 'still lost (fit"
        " 0.00)', goal after goal, with nothing the cart could do to earn a drive. A sigma is"
        " evidence for stopping a drive that is already running (BlindDriveWatch,"
        f" {LOST_SIGMA_M:.2f} m), where the readings keep coming and a cut costs a stop; it is"
        " not evidence for refusing to move at all",
        on_when="always where a sigma is published, and above all camera-only: it is the"
        " difference between a cart that drives on what it knows and one that waits for a sensor"
        " it does not have",
        off_when="to put the 0.25 m start threshold back for a comparison, or where a drive must"
        " never begin on a pose looser than Nav2's own arrival tolerance",
    ),
    Flag(
        "jump_clear",
        False,
        description="map -> odom is read from TF five times a second and, when it STEPS further"
        f" than {JumpClear.clear_costmap_jump_m:.2f} m, Nav2's local costmap is emptied"
        f' ("{CLEAR_LOCAL_COSTMAP}", asynchronously, at most once per {CLEAR_MIN_GAP_S:.0f} s):'
        " the marks in that grid were laid where the cart used to be. The step in that edge is the"
        " correction alone — the cart's own motion lives in odom -> base_link — whoever published"
        " it. Off, nothing reads the edge and no listener is started for it",
        why="OFF, AND SINCE 2026-09-22 ITS PREMISE IS GONE: the local costmap is built in the"
        " ODOM frame (ros/params/nav2_params.yaml), and a step in map -> odom does not move a"
        " grid that is not drawn in map — the marks stay exactly where the cart saw them. A clear"
        " would now throw away good evidence for nothing. The flag stays because the frame is one"
        " word away from being map again, and there it is the right behaviour: the lidar tracker"
        " did exactly this while it owned map -> odom (pepin.watch.JumpClear, written for the"
        " camera-only return of 2026-09-16, where the pose lagged 1.4 m behind the cart and Nav2"
        " spent 29 recoveries fighting marks placed at the poses before each correction). Even"
        " then the other side of the trade was unmeasured: RTAB-Map corrects in centimetres at a"
        " loop closure, which the costmap absorbs, and the raytracing of the live scans re-clears"
        " a stranded mark within seconds anyway. The threshold and the gap are the tracker's"
        " measured ones, inherited unchanged",
        on_when="only together with a local costmap put back into the map frame, and then when a"
        " drive is seen fighting a second copy of the room after a correction: recoveries at"
        " obstacles that are not there, the grid holding marks offset from the live scans by the"
        " size of the last jump",
        off_when="the shipped state, and the only sane one while that costmap is in odom: a clear"
        " there costs a controller its picture of the room and buys nothing",
    ),
    Flag(
        "pose_topic",
        True,
        description=f"the pose this node reads out of TF is republished as {POSE_TOPIC}"
        f" (geometry_msgs/PoseStamped in {MAP_FRAME}, {POSE_HZ:.0f} Hz, stamped with the"
        " transform's own stamp), so the other nodes on this board can have the pose without a"
        " TF listener of their own. Inert where a tracker runs (there /tracker_pose is that topic"
        " already) and on a split stack, where the reader is on the other machine and reads the"
        " edge itself. Off, nothing is published and this node's listener goes back to being"
        " started on the first ask",
        why="a TF listener is a subscription to the whole /tf stream — RTAB-Map's map -> odom at"
        " 20 Hz plus the board's odom -> base_link at 50 Hz plus the statics — deserialised in"
        " Python whatever the reader wanted out of it. Two of them ran on a 4-core A53 to read"
        " one pose now and then: this node's, for `where`, the preflight and the jump watch"
        " (~22 % of a core), and pepin_bringup.run_recorder's 5 Hz read for the tape's `loc`"
        " rows (~34 %), on a board measured at 252 % with the real-time loops starving"
        " (2026-09-22). This node owns navigation and the jump watch, so its listener is the one"
        " that stays and the tape reads the topic instead (run_recorder's loc_from)",
        on_when="always where this node and the tape recorder share a machine: it is what lets"
        " every other node there read the pose for the price of a 5 Hz PoseStamped",
        off_when="to put the two independent listeners back for a comparison — turn this off"
        " here and run_recorder's loc_from to tf, or the tape loses its pose rows",
    ),
    Flag(
        "controller",
        "rpp_shim",
        choices=("mppi", "rpp", "rpp_shim"),
        description="what follows the plan: mppi is Nav2's MPPI controller for every planner,"
        " held to the mark's heading by the yaw-checking goal checker; rpp is each planner's own"
        " Regulated Pure Pursuit from PLANNERS, ending on position alone as before 2026-09-23;"
        " rpp_shim is the reversing RPP inside Nav2's RotationShimController, which turns the cart"
        " to the mark's heading in place once it is inside the goal tolerance."
        " Published latched on controller_selector and goal_checker_selector, so a change is"
        " read by the behaviour tree at its next tick",
        why="the six legs of 2026-09-23 all stopped 12-60 deg short of the mark's heading:"
        " the reversing RPP cannot rotate in place, the tree ended the drive on position"
        " (xy_only_goal_checker), and the pivot that finishes the heading lives here in"
        " _pivot_to, which drives sent through goto_ros.py never reach. Each leg also ran 5-9"
        " recoveries, which is what an RPP answers a refused arc with; MPPI samples another"
        " trajectory instead",
        on_when="rpp_shim by default since the evening of 2026-09-23 (six legs: 21-33 s, 6-11"
        " recoveries, 4-10 cm and 2-5 deg at the mark; MPPI on the same board crawled at a median"
        " 0.06 m/s); mppi where its sampling is wanted, rpp for the position-only drives of before",
        off_when="rpp for an A/B against the RPP drives of before, or if MPPI's cycle does not"
        " fit the board's control period (the controller server's 'Control loop missed its"
        " desired rate')",
    ),
)

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
FOLLOWERS = {
    "mppi": ("FollowPathMPPI", "general_goal_checker"),
    # The reversing RPP inside Nav2's RotationShimController: RPP's pace, and the shim turns the
    # cart in place to the mark's heading once it is inside the goal tolerance.
    "rpp_shim": ("FollowPathShim", "general_goal_checker"),
}


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
        self._places_path = Path(str(self.declare_parameter("places", "/maps/places.yaml").value))
        self._record_dir = Path(str(self.declare_parameter("record_dir", "/maps/rec").value))
        self._port = int(self.declare_parameter("port", PORT).value)
        self._client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self._spin = ActionClient(self, Spin, "spin")
        self._relocalize = self.create_client(Trigger, "relocalize")
        self._where = self.create_client(Trigger, "where_am_i")
        self.fit = 0.0
        self._fit_heard = False  # a tracker has spoken here at least once
        self._tracker_probed = False  # ...or its service was waited for, once (_tracker_here)
        self.create_subscription(Float32, "localization_fit", self._on_fit, 10)
        # ...and the number that outranks it: how sure the tracker's FUSION is, whichever
        # source spoke into it (pepin_bringup.relocalizer, pepin.watch.Sigma). x is the
        # position sigma in metres, y the heading sigma in degrees, z the seconds since
        # the last accepted word.
        self._heard_sigma: Sigma | None = None
        self._sigma_at: float | None = None  # when it landed HERE: the age a gate reads
        self.create_subscription(String, SIGMA_TOPIC, self._on_sigma, 10)
        # The pose's other source: map -> base_link, which the tracker owns on a saved map and
        # pepin_bringup.slam_frame owns in SLAM mode. Read only when no tracker answers, and the
        # listener behind it is not started until then (_tf_pose).
        self._tf: TfLookup | None = None
        # The SLAM half's pulse. RTAB-Map's correction is published at 10 Hz whether or not the
        # graph moved, so its ARRIVAL is what says the machine that owns the pose is still there;
        # the transform composed from it says nothing, because slam_frame re-broadcasts the last
        # one for ever (pepin.watch.Correction). Only the arrival time is kept: the correction
        # itself belongs to slam_frame, which is the one publisher of the edge.
        self._correction_at: float | None = None
        self.create_subscription(TransformStamped, CORRECTION_TOPIC, self._on_correction, 5)
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
        self._goal_handle: Any = None
        self._driving = (
            False  # from before send_goal until the drive is finally over: cancel() clears it
        )
        # The laptop's pulse: on a split stack the board's link watch cancels a drive when this
        # stops. Harmless on one machine, where nothing listens.
        self._beat = self.create_publisher(Header, HEARTBEAT_TOPIC, 10)
        self.create_timer(1.0 / HEARTBEAT_HZ, self._heartbeat)
        # The laptop half brings the board's Nav2 up (pepin.deployment.next_transition): the
        # board's tree cannot load before this side's costmap service exists, and the board's
        # own manager gives up after one failure. Every few seconds: read the four states, send
        # the one transition that is due, read again. Idempotent, so a restart on either side
        # simply continues.
        self._side = str(self.declare_parameter("side", "all").value)
        self._board_state: dict[str, str] = {}
        self._board_up_logged = False
        if self._side == "laptop":
            self._state_clients = {
                node: self.create_client(GetState, f"/{node}/get_state") for node in BOARD_NAV_NODES
            }
            self._change_clients = {
                node: self.create_client(ChangeState, f"/{node}/change_state")
                for node in BOARD_NAV_NODES
            }
            self._bringup_busy = False
            self._bringup_since = 0.0
            self.create_timer(3.0, self._bring_board_up)
        # The recorder is a node where the sensors are (run_recorder, on the board): one command
        # opens a tape, the latched status names it (pepin.runlink).
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
        # WHO OWNS map -> odom in this stack (PEPIN_LOCALIZER, pepin.deployment.localizer; the
        # launch passes the resolved word). "tracker": the board's relocalizer answers
        # /where_am_i, publishes the fit and the sigma, and pepin_bringup.slam_frame is the one
        # thing that could put a pulse on /map_odom — everything below is as it has always been.
        # "rtabmap": there is no tracker to ask, and no /map_odom either, because the laptop's
        # RTAB-Map broadcasts map -> odom straight into TF. So the pose is read from
        # map -> base_link (the `tf_pose` path, which was written for exactly this) and the
        # correction watch is not consulted: it would refuse every goal with "no SLAM correction
        # has ever arrived" on a stack where no message-shaped correction exists to arrive.
        self._localizer = str(self.declare_parameter("localizer", DEFAULT_LOCALIZER).value)
        # Last, after every other declare_parameter: rclpy runs the switches' callback on
        # declarations too, and a name outside the table is refused there (node_kit.Switches).
        self._switches = Switches(self, FLAGS, on_change=self._on_switch)
        # The saved planner (the default where none was saved), with the follower the
        # `controller` flag names for it: after the switches, because the pick reads that flag,
        # and always, because the tree's own defaults are the RPP pair's.
        saved = ""
        with contextlib.suppress(OSError):
            saved = self._planner_path.read_text().strip()
        self.pick_planner(saved or self.planner)
        self._start_jump_watch()  # the map -> odom jump watch (flag jump_clear, default off)
        self._start_pose_topic()  # /pose, the board's one pose topic (flag pose_topic)
        threading.Thread(target=self._serve, daemon=True).start()
        self.get_logger().info(
            f"goal server ready on port {self._port}; localizer {self._localizer}"
            f" ({self._pose_source_note()}); {self._pose_topic_note()};"
            f" {self._switches.state()}"
        )
        self._up = True  # last: everything above exists, the timers may run

    def _pose_source_note(self) -> str:
        """Where this node will look for the cart's pose, in one phrase for the report line."""
        if self._localizer == "tracker":
            return "the board's tracker answers /where_am_i, its fit and sigma gate a goal"
        return "no tracker: map -> base_link from TF, no /map_odom pulse to watch"

    def _watching_correction(self) -> bool:
        """Whether the SLAM half's ``/map_odom`` pulse is evidence here at all: only where a
        message-shaped correction exists, which is the ``tracker`` stack with
        pepin_bringup.slam_frame. Under ``rtabmap`` RTAB-Map broadcasts the transform itself and
        nothing publishes that topic, so a watch of it would refuse every goal."""
        return self._switches.on("correction_watch") and self._localizer == "tracker"

    def _heartbeat(self) -> None:
        self._beat.publish(Header(stamp=self.get_clock().now().to_msg(), frame_id="laptop"))

    def _bring_board_up(self) -> None:
        """Every 3 s on the laptop: read the board's lifecycle states and send the due step."""
        now = time.monotonic()
        if self._bringup_busy:
            # A call whose answer never comes (the board restarted under it, the bridge re-routing)
            # must not hold the bring-up for good: after BRINGUP_ROUND_S the round is abandoned.
            if now - self._bringup_since < BRINGUP_ROUND_S:
                return
            self.get_logger().warning("board bring-up: a round got no answer; asking afresh")
            self._board_state.clear()
        self._bringup_busy = True
        self._bringup_since = now
        pending = set(BOARD_NAV_NODES)
        for node, client in self._state_clients.items():
            if not client.service_is_ready():
                self._board_state.pop(node, None)
                pending.discard(node)
                continue
            future = client.call_async(GetState.Request())
            future.add_done_callback(lambda f, n=node: self._board_state_read(n, f, pending))
        if not pending:
            self._bringup_busy = False

    def _board_state_read(self, node: str, future: Any, pending: set[str]) -> None:
        try:
            self._board_state[node] = str(future.result().current_state.label)
        except Exception:  # a dropped call: the next round asks again
            self._board_state.pop(node, None)
        pending.discard(node)
        if pending:
            return
        step = next_transition(self._board_state)
        if step is None:
            self._bringup_busy = False
            if all(self._board_state.get(n) == "active" for n in BOARD_NAV_NODES):
                if not self._board_up_logged:
                    self.get_logger().info("board Nav2 is up")
                    self._board_up_logged = True
            else:
                self._board_up_logged = False
            return
        node, transition = step
        self._board_up_logged = False
        request = ChangeState.Request()
        request.transition.id = transition
        self.get_logger().info(f"board bring-up: {node} <- transition {transition}")
        future = self._change_clients[node].call_async(request)
        future.add_done_callback(lambda f: self._board_transition_done(node, transition, f))

    def _board_transition_done(self, node: str, transition: int, future: Any) -> None:
        try:
            ok = bool(future.result().success)
        except Exception:
            ok = False
        if not ok:
            self.get_logger().warning(
                f"board bring-up: {node} refused transition {transition}; asking again in 3 s"
            )
        self._bringup_busy = False

    def _on_fit(self, msg: Float32) -> None:
        self.fit = float(msg.data)
        self._fit_heard = True

    def _on_sigma(self, msg: String) -> None:
        """The tracker's fused uncertainty landed: the JSON of pepin.watch.Sigma, and when it
        arrived. A message that does not parse is dropped, not obeyed."""
        heard = Sigma.from_json(msg.data, 0.0)
        if heard is None:
            return
        self._heard_sigma, self._sigma_at = heard, self._clock_s()
        self._fit_heard = True  # a sigma is a tracker speaking, exactly as a fit is

    def _sigma(self) -> Sigma | None:
        """How sure the pose is and how old that word is — ``None`` where the board publishes no
        sigma at all (a build from before 2026-09-15) or the flag is off, and the fit rules
        answer instead."""
        if not self._switches.on("sigma_gate") or self._heard_sigma is None:
            return None
        landed = self._clock_s() - (self._sigma_at or 0.0)
        return replace(self._heard_sigma, age_s=max(0.0, landed))

    def _on_correction(self, _msg: TransformStamped) -> None:
        """The SLAM correction landed here; only when it did is kept."""
        self._correction_at = self._clock_s()

    def _clock_s(self) -> float:
        """The node's clock in seconds."""
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _correction(self) -> Correction:
        """How long ago the SLAM correction last landed here (inside: ``None`` when none ever
        has) — the evidence a goal without a tracker is judged on beside the transform."""
        if self._correction_at is None:
            return Correction(None)
        return Correction(max(0.0, self._clock_s() - self._correction_at))

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
            # Which of the two spoke, said outright: a fit of 0.00 beside a pose read from TF
            # is not a lost robot, it is a stack with no tracker in it.
            source = "tracker" if "fit" in pose else "tf" if pose else "none"
            # ...and how long ago the SLAM correction last landed, where one ever has: the one
            # reading that tells a live SLAM half from a dead one before a goal is sent.
            correction = self._correction().age_s
            # The fit is the TRACKER's number and nobody else's. Printed beside a pose read from
            # TF it is a standing 0.00 that reads as a lost robot (a stack with no tracker has no
            # fit, it does not have a bad one), so where no tracker speaks the key is not there.
            fit = {"fit": self.fit} if source == "tracker" else {}
            self._send(
                connection,
                {
                    "event": "where",
                    **fit,
                    "planner": self.planner,
                    "pose": source,
                    "localizer": self._localizer,
                    **({} if correction is None else {"correction_s": round(correction, 2)}),
                    **pose,
                },
            )
        elif command == "mark":
            self._send(connection, self.mark(str(request.get("name", ""))))
        elif command == "planner":
            self._send(connection, self.pick_planner(str(request.get("name", ""))))
        elif command == "cancel":
            self._send(connection, {"event": "cancelled", "had_goal": self.cancel()})
        elif command == "go":
            self._go(request, connection)
        else:
            self._send(connection, {"event": "error", "detail": f"unknown command {command!r}"})

    def _on_places(self, msg: String) -> None:
        """The graph's book, as the places node last published it."""
        self._graph_places = {
            name: {"x": place.x, "y": place.y, "yaw_deg": place.theta_deg or 0.0}
            for name, place in places_from_json(msg.data).items()
        }

    def places(self) -> dict[str, dict[str, float]]:
        """The named places of the map in use: the graph's book, and ONLY it. Until the book has
        been heard there are no places — a name is refused with that reason — unless
        ``places_from_the_file`` asks for the yaml beside the map, the answer this server gave
        before World R."""
        if self._graph_places is not None:
            return dict(self._graph_places)
        if not self._switches.on("places_from_the_file"):
            return {}
        try:
            data: dict[str, dict[str, float]] = json.loads(self._places_path.read_text())
            return data
        except (OSError, ValueError):
            return {}

    def _why_no_place(self, name: str) -> str:
        """The refusal for a name that cannot be answered, saying which of the two it is: the
        book has not arrived at all, or it has and the name is not in it."""
        if self._graph_places is None and not self._switches.on("places_from_the_file"):
            return (
                f"no place {name!r} yet: the graph's book of places has not arrived from the"
                " laptop (is ros/laptop.sh vslam up, and has RTAB-Map published its first graph?)"
                " — a name is never answered from the old map's file"
            )
        return f"no such place: {name!r}"

    def _pose_now(self) -> dict[str, float]:
        """Where the cart stands: the tracker's own pose where a tracker answers, else the TF
        the SLAM correction feeds. Empty when neither does; ``fit`` is in it only from a tracker
        and ``age_s`` only from TF, so a reader can tell which one spoke.

        Where no tracker runs the service is not asked at all: /where_am_i has no server there,
        and the probe for it cost a full second of standing still per read — four per goal, two
        of them between arrival and the corrective pivot.
        """
        if self._switches.on("tf_pose") and not self._tracker_here():
            return self._tf_pose()
        pose = self._tracker_pose()
        if pose or not self._switches.on("tf_pose"):
            return pose
        return self._tf_pose()

    def _tracker_pose(self) -> dict[str, float]:
        """The tracker's pose, asked over its own service (empty when it does not answer)."""
        if not self._where.wait_for_service(timeout_sec=1.0):
            return {}
        result = self._wait(self._where.call_async(Trigger.Request()), 5.0)
        if result is None:
            return {}
        text = result.message
        try:
            return {
                "x": float(text.split("x ")[1].split(" m")[0]),
                "y": float(text.split("y ")[1].split(" m")[0]),
                "yaw_deg": float(text.split("yaw ")[1].split(" deg")[0]),
                "fit": float(text.split("fit ")[1].split(" ")[0]),
            }
        except (IndexError, ValueError):
            return {}

    def _tf_pose(self) -> dict[str, float]:
        """The cart's pose from ``map -> base_link`` alone, with the age of that edge in seconds
        (``age_s``): what SLAM mode has instead of a tracker. Empty when nobody publishes it."""
        wait = TF_WAIT_S
        if self._tf is None:
            # Started on the first ask and never before: a TF listener is a subscription to /tf,
            # sixty messages a second on the board, and where the tracker answers nothing here
            # ever reads it. Its buffer starts empty, so this one lookup waits longer.
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

    def _tracker_here(self) -> bool:
        """Whether a scan-matching tracker runs beside this node at all: it publishes the fit
        and serves the whole-map search. In SLAM mode neither exists.

        The waiting probe is paid ONCE per process: afterwards the answer comes from the graph
        (``service_is_ready``) and from the fit, so a tracker that comes up late is still found,
        while a stack that has none stops paying a second for every pose read. Under
        ``PEPIN_LOCALIZER=rtabmap`` even that one second is not paid: the launch does not start a
        tracker, so the answer is known before the first ask.
        """
        if self._localizer != "tracker":
            return False
        if self._fit_heard:
            return True
        if not self._tracker_probed:
            self._tracker_probed = True
            return bool(self._relocalize.wait_for_service(timeout_sec=TRACKER_WAIT_S))
        return bool(self._relocalize.service_is_ready())

    def _ready(self, pose: dict[str, float] | None = None) -> Readiness:
        """May a goal start now (:class:`pepin.watch.GoalGate`): the tracker's sigma where it
        publishes one and its fit where it does not; where no tracker runs, the age of
        map -> base_link AND the age of the SLAM correction, which is the only one of the two a
        dead laptop stops. ``pose`` is a reading already taken by the caller (mark's), so the
        edge is not looked up twice. The gate is kept in step with its live flag here rather
        than at the switch, so one reading and one rule answer every caller."""
        self._gate = replace(
            self._gate, start_on_a_known_pose=self._switches.on("start_on_a_known_pose")
        )
        if self._switches.on("tf_pose") and not self._tracker_here():
            edge = self._tf_pose() if pose is None else pose
            watched = self._correction() if self._watching_correction() else None
            return self._gate.verdict(None, edge.get("age_s"), watched)
        return self._gate.verdict(self.fit, None, sigma=self._sigma())

    def mark(self, name: str) -> dict[str, Any]:
        """Remember where the robot stands as ``name``; refused on the same evidence a goal is
        — a weak fit where a tracker speaks, a stale transform where none does."""
        pose = self._pose_now()
        if not name or not pose:
            return {"event": "error", "detail": "no name, or nothing answered about the pose"}
        ready = self._ready(pose)  # the same reading the mark is written from, not a second one
        if not ready.ready:
            return {"event": "error", "detail": ready.reason}
        places = self.places()
        # The fit is written only when a tracker gave one: a place marked in SLAM mode carries
        # no fit at all rather than a 0.00 that would read as "marked while lost".
        places[name] = {k: round(pose[k], 3) for k in ("x", "y", "yaw_deg")} | (
            {"fit": round(pose["fit"], 2)} if "fit" in pose else {}
        )
        self._places_path.write_text(json.dumps(places, indent=2, sort_keys=True) + "\n")
        return {"event": "marked", "name": name, **places[name]}

    def pick_planner(self, name: str) -> dict[str, Any]:
        """Choose the planner, the controller that follows it (flag ``controller``) and the goal
        checker that ends the drive with that controller; all three, or none."""
        pair = PLANNERS.get(name.lower())
        if pair is None:
            return {"event": "error", "detail": f"planner must be one of {sorted(PLANNERS)}"}
        planner, rpp = pair
        controller, checker = FOLLOWERS.get(self._switches["controller"], (rpp, RPP_GOAL_CHECKER))
        self._planner_pick.publish(String(data=planner))
        self._controller_pick.publish(String(data=controller))
        self._checker_pick.publish(String(data=checker))
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

    def _on_switch(self, name: str, old: Any, new: Any) -> None:
        """A live flag changed: a new ``controller`` re-publishes the pick with its follower."""
        if name == "controller" and new != old:
            self.pick_planner(self.planner)

    def cancel(self) -> bool:
        """Stop the running drive, if any; True when there was one.

        Clears ``_driving`` as well as the handle: during the lost -> relocalise -> resume window
        there is no handle to cancel, and the flag is what stops the resume from re-sending the
        goal the operator just cancelled.
        """
        with self._lock:
            handle, self._goal_handle = self._goal_handle, None
            was_driving, self._driving = self._driving, False
        if handle is not None:
            handle.cancel_goal_async()
        return was_driving

    def _go(
        self,
        request: dict[str, Any],
        connection: socket.socket,
        resume: bool = True,
        record: Path | None = None,
    ) -> None:
        """Send one goal and stream its progress until it ends or the caller hangs up.

        ``resume``: a drive stopped for being lost is sent again after a relocalisation, once —
        as a continuation of the same drive: ``record`` is the tape already open, so the resumed
        leg keeps the run's number and file instead of becoming a second run.
        """
        target = self._target_of(request)
        if target is None:
            detail = self._why_no_place(str(request.get("place", "")))
            self._send(connection, {"event": "error", "detail": detail})
            return
        x, y, yaw_deg, name = target
        ready = self._ready()
        if not ready.ready:
            if not ready.search:  # nothing to search with: no tracker, no fresh transform
                self._send(connection, {"event": "error", "detail": ready.reason})
                self.get_logger().warning(f"goal refused: {ready.reason}")
                return
            if not self._find_myself(connection):
                return
        if not self._client.wait_for_server(timeout_sec=5.0):
            self._send(connection, {"event": "error", "detail": "Nav2 is not up"})
            return
        goal = NavigateToPose.Goal()
        goal.pose = self._pose_msg(x, y, yaw_deg)
        started = time.monotonic()
        feedback: dict[str, Any] = {}
        with self._lock:
            if self._driving and record is None:
                self._send(
                    connection, {"event": "error", "detail": "already driving: cancel first"}
                )
                return
            self._driving = True
        if record is None:
            record = self.start_recording(name or f"{x:.0f}_{y:.0f}")
        self.get_logger().info(
            f"run {self._runs.run}: planner {PLANNERS[self.planner][0]} "
            f"-> {name or 'coordinates'} ({x:.2f}, {y:.2f}, {yaw_deg:.0f} deg),"
            f" pose from {'the tracker' if ready.tracker else 'TF (no tracker here)'}"
        )
        try:
            send = self._client.send_goal_async(
                goal,
                lambda f: feedback.update(
                    distance=f.feedback.distance_remaining,
                    recoveries=f.feedback.number_of_recoveries,
                ),
            )
            handle = self._wait(send, 10.0)
            if handle is None or not handle.accepted:
                self._send(connection, {"event": "error", "detail": "the goal was refused"})
                return
            with self._lock:
                self._goal_handle = handle
            self._send(
                connection,
                {
                    "event": "accepted",
                    "run": self._runs.run,
                    "planner": PLANNERS[self.planner][0],
                    "pose": "tracker" if ready.tracker else "tf",
                    # early: a drive that never reaches "done" is still fetched
                    "recording": None if record is None else str(record),
                    "place": name,
                    "x": x,
                    "y": y,
                    "yaw_deg": yaw_deg,
                    "sent_in_ms": round((time.monotonic() - started) * 1000),
                },
            )
            result_future = handle.get_result_async()
            last = 0.0
            # The blind-drive watch reads the tracker's fit: where no tracker publishes one it is
            # not armed at all (it would read the standing 0.0 as lost four seconds in).
            blind = BlindDriveWatch() if ready.tracker else None
            # Its SLAM analogue. Nav2 is NOT the backstop here: slam_frame keeps broadcasting
            # map -> odom from the last correction, so the costmaps keep a fresh map ->
            # base_link and nothing times out — the cart would follow its plan by dead reckoning
            # across a map that stopped growing. The correction's arrival is what stops it.
            pulse = not ready.tracker and self._watching_correction()
            stopped_lost = False
            cut = ""
            while rclpy.ok() and not result_future.done():
                time.sleep(0.05)  # the node's own spin serves the action; this thread only reports
                now = time.monotonic()
                # blind: stop, don't finish. The sigma where the tracker publishes one — a
                # camera-only drive has no fit of its own to read — the fit where it does not.
                if blind is not None and blind.observe(self.fit, now, sigma=self._sigma()):
                    stopped_lost = True
                    self._send(
                        connection,
                        {
                            "event": "lost",
                            "rule": blind.rule,
                            "reading": blind.phrase(),
                            "fit": self.fit,
                            "t": round(now - started, 1),
                        },
                    )
                    self.get_logger().warning(
                        f"lost mid-drive ({blind.phrase()}, judged by the {blind.rule}):"
                        " stopping to relocalise"
                    )
                    handle.cancel_goal_async()
                    break
                if pulse and (correction := self._correction()).stale(
                    self._gate.correction_fresh_s
                ):
                    cut = correction.phrase()  # no tracker, nothing to search with: stop, period
                    self._send(
                        connection,
                        {
                            "event": "lost",
                            "correction_s": correction.age_s,
                            "t": round(now - started, 1),
                        },
                    )
                    self.get_logger().warning(f"{cut}: stopping the drive, the map is not growing")
                    handle.cancel_goal_async()
                    break
                if feedback and now - last > 1.0:
                    last = now
                    self._send(
                        connection, {"event": "feedback", "t": round(now - started, 1), **feedback}
                    )
            with self._lock:
                self._goal_handle = None
            if stopped_lost:
                # Stopped on purpose: find ourselves standing still, then the same goal, once.
                self._wait(result_future, 10.0)
                with self._lock:
                    still_wanted = self._driving
                if still_wanted and self._find_myself(connection) and resume:
                    self._send(connection, {"event": "resuming", "fit": self.fit})
                    self.get_logger().info(
                        f"resuming the goal after relocalising (fit {self.fit:.2f})"
                    )
                    self._go(request, connection, resume=False)
                return
            if cut:  # let the cancel land before the tape is closed
                self._wait(result_future, 10.0)
            outcome = result_future.result()
            status = getattr(outcome, "status", 0) if outcome else 0
            if status == 4 and not cut:  # position met: now the heading, on the tape still
                self._pivot_to(yaw_deg, connection)
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
                    **({"detail": cut} if cut else {}),
                },
            )
        finally:  # a refused goal or a broken connection must not leave a recorder running
            if (
                resume
            ):  # the outermost call owns the tape and the flag; a resumed leg is the same drive
                self.stop_recording()
                with self._lock:
                    self._driving = False

    def _pivot_to(self, yaw_deg: float, connection: socket.socket) -> None:
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
        if not self._spin.wait_for_server(timeout_sec=2.0):
            self._send(connection, event | {"status": 0, "detail": "no spin behaviour"})
            return
        goal = Spin.Goal()
        goal.target_yaw = math.radians(residual)
        goal.time_allowance = Duration(sec=int(PIVOT_ALLOWANCE_S))
        handle = self._wait(self._spin.send_goal_async(goal), 5.0)
        if handle is None or not handle.accepted:
            self._send(connection, event | {"status": 0, "detail": "the spin was refused"})
            return
        outcome = self._wait(handle.get_result_async(), PIVOT_ALLOWANCE_S + 5.0)
        status = getattr(outcome, "status", 0) if outcome else 0
        after = heading_residual_deg(yaw_deg, self._pose_now().get("yaw_deg", here))
        self._send(connection, event | {"status": int(status), "after_deg": round(after, 1)})
        self.get_logger().info(f"pivot {residual:+.0f} deg: status {status}, {after:+.0f} deg left")

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

    def _find_myself(self, connection: socket.socket) -> bool:
        """A weak fit before (or during) a drive buys one whole-map search; blind driving is never
        allowed. The tracker may already be searching on its own — then its answer is awaited
        rather than asked for twice — and its fit is published once a second, so the verdict
        waits for a fresh reading instead of reading a stale one (run 0054: relocalised at 0.61,
        judged "still lost" at the 0.31 published a moment earlier, never resumed)."""
        self._send(connection, {"event": "searching", "fit": self.fit})
        self.get_logger().info(f"searching the whole map (fit {self.fit:.2f})")
        if not self._relocalize.wait_for_service(timeout_sec=2.0):
            self._send(connection, {"event": "error", "detail": "the tracker is not up"})
            return False
        result = self._wait(self._relocalize.call_async(Trigger.Request()), 60.0)
        detail = result.message if result else "no answer"
        deadline = time.monotonic() + 15.0  # an episode is two whole-map searches, 4-7 s each here
        while self.fit < GOOD_FIT and time.monotonic() < deadline:
            time.sleep(0.2)
        self._send(connection, {"event": "searched", "detail": detail, "fit": self.fit})
        if self.fit >= GOOD_FIT:
            self.get_logger().info(f"found myself: fit {self.fit:.2f}")
            return True
        self.get_logger().warning(f"still lost after the search: fit {self.fit:.2f}")
        self._send(connection, {"event": "error", "detail": f"still lost (fit {self.fit:.2f})"})
        return False

    # ---- the board's one pose topic (flag pose_topic) -----------------------------------------
    def _start_pose_topic(self) -> None:
        """Wire ``/pose``: the cart's pose in ``map``, five times a second, out of the TF
        listener this node already owns.

        Only where no tracker runs, and only where the reader is on this machine. Under
        ``tracker`` the board's relocalizer publishes ``/tracker_pose``, which is this topic with
        a covariance on it, and a second opinion on one pose is what CLAUDE.md's one-localiser
        rule exists to forbid. On a SPLIT stack this node is the laptop's while the tape recorder
        is the board's, and a pose that crossed the WiFi to be written to the tape is exactly
        what putting that recorder on the board is meant to prevent — so there it reads the edge
        itself (its ``loc_from`` tf) and nothing is published here. Either way the timer is not
        created and the flag is inert; the start line says which it is.

        The listener is built HERE, in the constructor, rather than on the first ask: this timer
        wants it from the first tick, and it is the one listener the board keeps (the flag's
        ``why``). It starts a spin thread for this node, which is why ``_up`` guards the tick.
        """
        if self._localizer == "tracker" or not runs_here(self._side, "run_recorder"):
            return
        self._pose_pub = self.create_publisher(PoseStamped, POSE_TOPIC, 5)
        if self._tf is None:
            self._tf = TfLookup(self)
        self.create_timer(1.0 / POSE_HZ, self._publish_pose)

    def _pose_topic_note(self) -> str:
        """Where the rest of the board gets the pose from, in one phrase for the start line."""
        if self._localizer == "tracker":
            return f"{POSE_TOPIC} not published: the tracker's own /tracker_pose is that topic"
        if not runs_here(self._side, "run_recorder"):
            return (
                f"{POSE_TOPIC} not published: its reader is on the other machine (side"
                f" {self._side}) and reads the edge there"
            )
        if not self._switches.on("pose_topic"):
            return f"{POSE_TOPIC} off: every reader of the pose parses /tf for itself"
        return f"{POSE_TOPIC} at {POSE_HZ:.0f} Hz from map -> base_link, for the board's readers"

    def _publish_pose(self) -> None:
        """One pose onto ``/pose``, or nothing at all: this is a relay of an edge, never a
        measurement of its own.

        Stamped with the TRANSFORM's stamp and not with now — a pose stamped "now" is how a
        stale reading becomes a fresh lie (2026-09-19). The edge is read from whatever is in the
        buffer and never waited for: this runs on the executor thread, beside the socket's own
        callbacks, and a missing edge is simply no message.
        """
        if not self._up or not self._switches.on("pose_topic") or self._tf is None:
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

    # ---- the map -> odom jump watch (flag jump_clear) -----------------------------------------
    def _start_jump_watch(self) -> None:
        """Wire the jump watch: the rules (:class:`pepin.watch.JumpClear`), a client of Nav2's
        clearing service and a 5 Hz timer that reads ``map -> odom``.

        The watch lives HERE because this node is the one on the board that is awake between goals
        and already owns a TF path. It used to live in the lidar tracker, which published that edge
        itself and could watch its own steps; RTAB-Map owns it now, and a correction nobody watches
        leaves the local costmap holding a copy of the room offset by the jump."""
        self._jumps = JumpClear(self._clear_local_costmap, min_gap_s=CLEAR_MIN_GAP_S)
        self._clear_costmap = self.create_client(ClearEntireCostmap, CLEAR_LOCAL_COSTMAP)
        self._clears = 0
        self.create_timer(1.0 / JUMP_WATCH_HZ, self._watch_map_odom)

    def _watch_map_odom(self) -> None:
        """Five times a second: the newest ``map -> odom`` TF holds, into the watch.

        Nothing at all happens while ``jump_clear`` is off — not even the TF listener is started,
        which is the point of starting it here rather than in the constructor: a listener is a
        subscription to ``/tf``, sixty messages a second on the board. The first reading after the
        flag goes on is the watch's baseline and never a jump.

        The edge is read from whatever is in the buffer, never waited for: this runs on the
        executor thread, beside the socket's own callbacks. A missing edge is simply no reading.
        """
        if not self._switches.on("jump_clear"):
            return
        if self._tf is None:
            self._tf = TfLookup(self)
        transform = self._tf.transform(MAP_FRAME, ODOM_FRAME, None, 0.0)
        if transform is None:
            return
        self._jumps.moved(
            (
                transform.transform.translation.x,
                transform.transform.translation.y,
                yaw_of(transform.transform.rotation),
            ),
            time.monotonic(),
        )

    def _clear_local_costmap(self, jump_m: float) -> None:
        """Ask Nav2 to empty its local costmap because ``map -> odom`` has just stepped
        ``jump_m``: the marks in that grid were laid where the cart used to be
        (:class:`pepin.watch.JumpClear` decides when).

        The call is asynchronous and its answer is never waited for — this runs on the executor
        thread, which the socket handler and the drive both need — and it is the only line this
        watch logs, so the flag's state and the count are in it."""
        self._clears += 1
        self._clear_costmap.call_async(ClearEntireCostmap.Request())
        self.get_logger().info(
            f"map -> odom jumped {jump_m:.2f} m: local costmap cleared, its marks were laid at the"
            f" old pose ({self._clears} clears this run; jump_clear=on,"
            f" jump {self._jumps.clear_costmap_jump_m:.2f} m, no oftener than"
            f" {CLEAR_MIN_GAP_S:.0f} s)"
        )

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
    rclpy.init()
    node = GoalServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_recording()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
