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

A CANCEL MEANS EVERY GOAL ON THE BOARD, not only this node's own (flag ``cancel_every_goal``):
both navigators' cancel services are asked with a zero goal id, exactly as ``goto_ros.py cancel``
asks them, so a drive ``ros/goto.sh`` started through goto_ros.py is stopped from here too — by a
socket write from the laptop (pepin.goal_link) instead of a fresh ROS process on the board, whose
new zenoh session stalls every laptop -> board stream for about three seconds.

WHERE THE POSE COMES FROM. The laptop's RTAB-Map owns ``map -> odom`` (one localiser,
2026-09-22), so this node reads ``map -> base_link`` from TF and judges a goal by how fresh that
edge is (:class:`pepin.watch.GoalGate`) and by whether this start of RTAB-Map is placed at all
(``start_needs_placement``). ``where`` answers ``"pose": "tf"``. The board tracker that answered
``/where_am_i`` with a fit before is on the tag alt/tracker-2026-09-22.

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
on a split stack, where this node is the laptop's and the recorder is the board's: a pose that
crossed the WiFi to be written to a tape is what putting that recorder on the board prevents.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from action_msgs.srv import CancelGoal
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.srv import ChangeState, GetState
from nav2_msgs.action import NavigateToPose, Spin
from nav2_msgs.srv import ClearEntireCostmap
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, String

from pepin.deployment import (
    BOARD_NAV_NODES,
    HEARTBEAT_HZ,
    HEARTBEAT_TOPIC,
    next_transition,
    runs_here,
)
from pepin.flags import Flag, FlagSet
from pepin.goal_link import CANCEL_CONFIRM_S, NAV_ACTIONS, cancel_outcome
from pepin.lidar_watch import LidarWatch
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
    JumpClear,
    Placement,
    Preflight,
    Readiness,
)
from pepin_bringup.msgs import stamp_seconds, yaw_of
from pepin_bringup.node_kit import Switches, TfLookup, spin_main

PORT = 3337
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
# Nav2's own service for emptying the grid the controller steers on, and the watch's rules over it
# (pepin.watch.JumpClear). The board tracker made this call while IT owned map -> odom; that edge
# is RTAB-Map's now, so the call lives here, in the one node on the board that is awake between
# goals.
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

# The live flags (CLAUDE.md rule 19); their state is printed in the node's start line.
FLAGS = FlagSet(
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
        " TF listener of their own. Inert on a split stack, where the reader is on the other"
        " machine and reads the edge itself. Off, nothing is published and this node's"
        " listener goes back to being started on the first ask",
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
    Flag(
        "start_needs_placement",
        True,
        description="a goal or a mark waits for the laptop's word"
        f" on {PLACEMENT_TOPIC} (pepin_bringup.rtabmap_frame, latched) that this start of"
        " RTAB-Map is PLACED — a node of the loaded map recognised, or an operator's seed — and"
        " nothing heard is refused like not placed. ros/tools/goto_ros.py asks this node for"
        " this same flag before its own preflight. Off, a fresh map -> base_link is enough, as"
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
        "cancel_every_goal",
        True,
        description="a cancel on the socket also asks both navigators' own cancel services"
        f" ({' and '.join(NAV_ACTIONS)}, <action>/_action/cancel_goal) for EVERY goal — a zero"
        " goal id, whoever sent it — and answers what each said (``navigators``); off, it"
        " cancels only the goal this node sent, as before 2026-09-25",
        why="ros/goto.sh drives through ros/tools/goto_ros.py, a goal this node never sent, so"
        " this cancel reached nothing, and goto.sh's own cancel started goto_ros.py on the board"
        " mid-drive: a new ROS process is a new zenoh session, and each one stalled all laptop ->"
        " board delivery for 2.6-3.1 s about 1.5 s after it started (34 of 39 cases, journal"
        " 2026-09-25). Asked from this long-lived node the same cancel costs a socket write;"
        f" one {CANCEL_CONFIRM_S:.0f} s deadline is shared by both navigators, as in goto_ros.py",
        on_when="always: the operator's cancel means every goal on the board, whichever client"
        " sent it",
        off_when="to put the old answer back for a comparison — pepin.goal_link then finds no"
        " navigators in the answer and ros/goto.sh cancel falls back to goto_ros.py",
    ),
    Flag(
        "lidar_watch",
        True,
        description="the lidar's driver is restarted when its port is there and no scan has"
        " come for 5 s, and once per absence of the port so it respawns idle instead of spinning"
        " a core (pepin.lidar_watch); `where` says `lidar` either way",
        why="the driver opens its port once: a lidar re-plugged or plugged in after the start"
        " stayed dead until a stack restart (journal 2026-09-28)",
        on_when="always",
        off_when="while the lidar is deliberately held silent with its port present",
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
# The lidar the watch looks after: its scan, its port as udev names it, and its driver's process
# (the component container of robot.launch.py).
LIDAR_SCAN_TOPIC = "/scan"
LIDAR_PORT = "/dev/lidar"
LIDAR_PROCESS = "__node:=lidar_container"
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
        # Every goal on either navigator, whoever sent it (flag cancel_every_goal): the action
        # servers' own cancel services, made once here so a cancel never waits for discovery.
        self._cancel_clients = {
            action: self.create_client(CancelGoal, f"/{action}/_action/cancel_goal")
            for action in NAV_ACTIONS
        }
        # The cart's pose: map -> base_link, RTAB-Map's correction composed with the board's own
        # odometry. The listener behind it is started on the first ask (_tf_pose) or by the pose
        # topic below, whichever comes first.
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
        self._goal_handle: Any = None
        self._driving = (
            False  # from before send_goal until the drive is finally over: cancel() clears it
        )
        # The laptop's pulse: on a split stack the board's link watch cancels a drive when this
        # stops. Harmless on one machine, where nothing listens.
        self._beat = self.create_publisher(Header, HEARTBEAT_TOPIC, 10)
        self.create_timer(1.0 / HEARTBEAT_HZ, self._heartbeat)
        # The lidar's driver, brought back when its port is there and it says nothing. The scan
        # is taken raw: only its arrival matters here, and a parsed one costs a core's percent.
        self._lidar = LidarWatch(started_at=time.monotonic())
        self.create_subscription(
            LaserScan, LIDAR_SCAN_TOPIC, self._on_scan, qos_profile_sensor_data, raw=True
        )
        self.create_timer(1.0, self._watch_lidar)
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
            f"goal server ready on port {self._port} (pose: map -> base_link from TF);"
            f" {self._pose_topic_note()};"
            f" {self._switches.state()}"
        )
        self._up = True  # last: everything above exists, the timers may run

    def _on_scan(self, _raw: bytes) -> None:
        """A scan arrived (unparsed): the lidar is alive."""
        self._lidar.scan(time.monotonic())

    def _watch_lidar(self) -> None:
        """Once a second: end the driver's process when the watch says so; the launch respawns
        it within two seconds on the port that is there now."""
        if not self._switches.on("lidar_watch"):
            return
        now, port = time.monotonic(), os.path.exists(LIDAR_PORT)
        if self._lidar.due(now, port):
            rule = (
                "port present: respawn it on the port"
                if port
                else "port lost: respawn it without one, to idle (once per absence)"
            )
            self.get_logger().warning(
                f"lidar {self._lidar.status(now, port)}: ending its driver ({LIDAR_PROCESS}),"
                f" kick {self._lidar.kicks}, rule {rule}"
            )
            subprocess.run(["pkill", "-INT", "-f", LIDAR_PROCESS], check=False, timeout=5)

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

    def _clock_s(self) -> float:
        """The node's clock in seconds."""
        return float(self.get_clock().now().nanoseconds) * 1e-9

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
                    "lidar": self._lidar.status(time.monotonic(), os.path.exists(LIDAR_PORT)),
                    **pose,
                },
            )
        elif command == "mark":
            self._send(connection, self.mark(str(request.get("name", ""))))
        elif command == "planner":
            self._send(connection, self.pick_planner(str(request.get("name", ""))))
        elif command == "cancel":
            # Under cancel_every_goal this node's own goal is one of "every": its handle is only
            # let go here, so the navigators' answer counts it once instead of rejecting a goal
            # a second request had already put into canceling.
            every = self._switches.on("cancel_every_goal")
            answer: dict[str, Any] = {"event": "cancelled", "had_goal": self.cancel(send=not every)}
            if every:
                answer["navigators"] = self.cancel_every_goal()
            self._send(connection, answer)
        elif command == "go":
            self._go(request, connection)
        else:
            self._send(connection, {"event": "error", "detail": f"unknown command {command!r}"})

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
        """Where the cart stands: ``map -> base_link`` with the age of that edge (``age_s``);
        empty when nobody publishes it."""
        return self._tf_pose()

    def _tf_pose(self) -> dict[str, float]:
        """The cart's pose from ``map -> base_link``, with the age of that edge in seconds
        (``age_s``). Empty when nobody publishes it."""
        wait = TF_WAIT_S
        if self._tf is None:
            # Started on the first ask when the pose topic did not start it: a TF listener is a
            # subscription to /tf, sixty messages a second on the board. Its buffer starts
            # empty, so this one lookup waits longer.
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

    def _ready(self, pose: dict[str, float] | None = None) -> Readiness:
        """May a goal start now (:class:`pepin.watch.GoalGate`): the age of map -> base_link, and
        whether this start of RTAB-Map is placed (flag ``start_needs_placement``). ``pose`` is a
        reading already taken by the caller (mark's), so the edge is not looked up twice."""
        edge = self._tf_pose() if pose is None else pose
        ready = self._gate.verdict(None, edge.get("age_s"))
        if ready.ready:
            placed = Preflight.placement(
                self._placement_standing(), asked=self._switches.on("start_needs_placement")
            )
            if not placed.ok:
                return Readiness(False, tracker=False, rule=BY_PLACEMENT, reason=placed.detail)
        return ready

    def mark(self, name: str) -> dict[str, Any]:
        """Remember where the robot stands as ``name``; refused on the same evidence a goal is
        — a stale transform, or a start of RTAB-Map nobody has placed."""
        pose = self._pose_now()
        if not name or not pose:
            return {"event": "error", "detail": "no name, or nothing answered about the pose"}
        ready = self._ready(pose)  # the same reading the mark is written from, not a second one
        if not ready.ready:
            return {"event": "error", "detail": ready.reason}
        places = self.places()
        places[name] = {k: round(pose[k], 3) for k in ("x", "y", "yaw_deg")}
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

    def cancel(self, send: bool = True) -> bool:
        """Stop the running drive, if any; True when there was one.

        Clears ``_driving`` as well as the handle. ``send=False`` lets the handle go without
        cancelling it: the caller cancels every goal on the navigator itself
        (:meth:`cancel_every_goal`).
        """
        with self._lock:
            handle, self._goal_handle = self._goal_handle, None
            was_driving, self._driving = self._driving, False
        if handle is not None and send:
            handle.cancel_goal_async()
        return was_driving

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
        """Send one goal and stream its progress until it ends or the caller hangs up."""
        target = self._target_of(request)
        if target is None:
            detail = self._why_no_place(str(request.get("place", "")))
            self._send(connection, {"event": "error", "detail": detail})
            return
        x, y, yaw_deg, name = target
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
        with self._lock:
            if self._driving:
                self._send(
                    connection, {"event": "error", "detail": "already driving: cancel first"}
                )
                return
            self._driving = True
        record = self.start_recording(name or f"{x:.0f}_{y:.0f}")
        self.get_logger().info(
            f"run {self._runs.run}: planner {PLANNERS[self.planner][0]} "
            f"-> {name or 'coordinates'} ({x:.2f}, {y:.2f}, {yaw_deg:.0f} deg)"
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
            result_future = handle.get_result_async()
            last = 0.0
            while rclpy.ok() and not result_future.done():
                time.sleep(0.05)  # the node's own spin serves the action; this thread only reports
                now = time.monotonic()
                if feedback and now - last > 1.0:
                    last = now
                    self._send(
                        connection, {"event": "feedback", "t": round(now - started, 1), **feedback}
                    )
            with self._lock:
                self._goal_handle = None
            outcome = result_future.result()
            status = getattr(outcome, "status", 0) if outcome else 0
            if status == 4:  # position met: now the heading, on the tape still
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
                },
            )
        finally:  # a refused goal or a broken connection must not leave a recorder running
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

    # ---- the board's one pose topic (flag pose_topic) -----------------------------------------
    def _start_pose_topic(self) -> None:
        """Wire ``/pose``: the cart's pose in ``map``, five times a second, out of the TF
        listener this node already owns.

        Only where the reader is on this machine. On a SPLIT stack this node is the laptop's
        while the tape recorder is the board's, and a pose that crossed the WiFi to be written to
        the tape is exactly what putting that recorder on the board is meant to prevent — so there
        it reads the edge itself (its ``loc_from`` tf) and nothing is published here: the timer
        is not created and the flag is inert; the start line says so.

        The listener is built HERE, in the constructor, rather than on the first ask: this timer
        wants it from the first tick, and it is the one listener the board keeps (the flag's
        ``why``). It starts a spin thread for this node, which is why ``_up`` guards the tick.
        """
        if not runs_here(self._side, "run_recorder"):
            return
        self._pose_pub = self.create_publisher(PoseStamped, POSE_TOPIC, 5)
        if self._tf is None:
            self._tf = TfLookup(self)
        self.create_timer(1.0 / POSE_HZ, self._publish_pose)

    def _pose_topic_note(self) -> str:
        """Where the rest of the board gets the pose from, in one phrase for the start line."""
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
    # Through node_kit.spin_main like the other nodes: close() (the tape) first, then the TF
    # listener's non-daemon thread — with its own spin this node never exited on SIGINT — and
    # kill -USR2 prints its stacks (its main thread burns 43-46 % of an A53 core, 2026-09-24).
    spin_main(GoalServer)


if __name__ == "__main__":
    main()
