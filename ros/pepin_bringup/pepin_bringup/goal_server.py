"""The robot waits for orders instead of being started for each one.

Every goal used to boot its own client on the board: ssh, docker exec, import rclpy, import the
Nav2 commander, build a node, discover the action server — 8 to 15 seconds before the wheels
could move, paid again for every command (measured 2026-09-08). This node is already running,
already connected to Nav2 and already holding the places book, so a command costs a socket write.

It speaks JSON lines on a TCP port (like the base and ToF servers), one connection at a time:

    {"cmd": "go", "place": "printer"}      {"cmd": "go", "x": -11.4, "y": 0.8, "yaw_deg": 140}
    {"cmd": "cancel"}                      {"cmd": "where"}
    {"cmd": "places"}                      {"cmd": "planner", "name": "navfn"}

and answers with one JSON line per event: accepted, feedback, arrival, done. It also owns the
run's recording: it starts one when a goal starts and closes it when the goal ends, so a
recording can no longer outlive its run.

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
import json
import math
import socket
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from action_msgs.msg import GoalStatusArray
from action_msgs.srv import CancelGoal
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose, Spin
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from pepin.flags import Flag, FlagSet, load_knobs, with_knobs
from pepin.goal_link import CANCEL_CONFIRM_S, NAV_ACTIONS, cancel_outcome
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

MAP_FRAME = "map"
BASE_FRAME = "base_link"
# THE ONE POSE TOPIC. This node already parses /tf for the pose; every other node beside it that
# wants it reads this instead of starting a second listener (pepin_bringup.run_recorder). 5 Hz is
# the rate the tape thinned its `loc` rows to anyway and the local costmap's own
# update_frequency.
POSE_TOPIC = "/pose"
POSE_HZ = 5.0
TF_WAIT_S = 0.3  # how long a pose lookup waits for the edge: the drive thread asks, not a callback
TF_FIRST_WAIT_S = 2.0  # ...and the first one waits for the listener's buffer to fill at all

# The live flags (CLAUDE.md rule 19); their state is printed in the node's start line.
FLAGS = FlagSet(
    Flag(
        "controller",
        "mppi",
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
        " Published latched on controller_selector and goal_checker_selector, so a change is"
        " read by the behaviour tree at its next tick",
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
# The lidar as Nav2 receives it: `where` says how old the last scan is. Report only: the driver
# is the board's, and a restart of it from here (the re-plug recovery of 2026-09-28) went with
# Nav2 to the Mac, where neither its port nor its process is.
LIDAR_SCAN_TOPIC = "/scan"
LIDAR_SILENT_S = 5.0  # no scan for this long: `where` says so
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
# FOLLOWERS' pair again. The tree's FollowPath passes the new controller id to the controller
# server at its next tick, which swaps the plugin in place on the same goal.
PARKERS = {"shim_mppi": ("FollowPathMPPI", "general_goal_checker")}


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
        self._driving = (
            False  # from before send_goal until the drive is finally over: cancel() clears it
        )
        # The end-of-drive pivot (behaviour server Spin) is a goal of its own: a cancel stops it
        # too. Every cancel bumps _cancels, so a drive that started before it never sends a goal
        # after it (see _go: the recorder alone may take 8 s between the two).
        self._spin_handle: Any = None
        self._cancels = 0
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
        # The lidar as Nav2 receives it, taken raw: only its arrival matters here.
        self._last_scan_at: float | None = None
        self._started_at = time.monotonic()
        self.create_subscription(
            LaserScan, LIDAR_SCAN_TOPIC, self._on_scan, qos_profile_sensor_data, raw=True
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
        # Last, after every other declare_parameter: rclpy runs the switches' callback on
        # declarations too, and a name outside the table is refused there (node_kit.Switches).
        self._switches = Switches(
            self, with_knobs(FLAGS, load_knobs("goal_server")), on_change=self._on_switch
        )
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
        """A scan arrived (unparsed): the lidar reaches Nav2."""
        self._last_scan_at = time.monotonic()

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
        """A drive of this node, or any goal running on a navigator or the spin behaviour."""
        with self._lock:
            driving = self._driving or self._spin_handle is not None
        return driving or any(self._nav_active.values())

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
        """A live flag changed: a new ``controller`` re-publishes the pick with its follower."""
        if name == "controller" and new != old:
            self.pick_planner(self.planner)

    def cancel(self, send: bool = True) -> bool:
        """Stop the running drive, if any; True when there was one.

        Clears ``_driving`` as well as the handle, and counts the cancel, so a drive still on
        its way to Nav2 is never sent and a handle that arrives after it is cancelled at once
        (:meth:`_go`). The end-of-drive pivot is cancelled whatever ``send`` says: no navigator
        owns it. ``send=False`` lets the drive's handle go without cancelling it: the caller
        cancels every goal on the navigators itself (:meth:`cancel_every_goal`).
        """
        with self._lock:
            handle, self._goal_handle = self._goal_handle, None
            spin, self._spin_handle = self._spin_handle, None
            was_driving, self._driving = self._driving, False
            self._cancels += 1
        if handle is not None and send:
            handle.cancel_goal_async()
        if spin is not None:
            spin.cancel_goal_async()
        return was_driving or spin is not None

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
            cancels = self._cancels  # any cancel from here on is this drive's
        record = self.start_recording(name or f"{x:.0f}_{y:.0f}")
        mode = str(self._switches["controller"])
        parks = (
            f": {PARKERS[mode][0]} within {self._switches['park_distance_m']:.2f} m"
            if mode in PARKERS
            else ""
        )
        self.get_logger().info(
            f"run {self._runs.run}: planner {PLANNERS[self.planner][0]}, controller "
            f"{self.controller} ({mode}{parks}) -> {name or 'coordinates'}"
            f" ({x:.2f}, {y:.2f}, {yaw_deg:.0f} deg)"
        )
        try:
            if self._cancelled_since(cancels):  # the recorder's wait is up to 8 s of it
                self._send(
                    connection, {"event": "error", "detail": "cancelled before Nav2 had the goal"}
                )
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
                self._send(connection, {"event": "error", "detail": "the goal was refused"})
                return
            with self._lock:
                late = self._cancels != cancels
                if not late:
                    self._goal_handle = handle
            if late:  # the cancel came while Nav2 was taking the goal: it is cancelled now
                handle.cancel_goal_async()
                self._send(
                    connection,
                    {"event": "error", "detail": "cancelled while Nav2 took the goal: cancelled"},
                )
                return
            self._send(
                connection,
                {
                    "event": "accepted",
                    "run": self._runs.run,
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
            if handover is not None:
                self._send(connection, handover)
            result_future = handle.get_result_async()
            last = 0.0
            while rclpy.ok() and not result_future.done():
                time.sleep(0.05)  # the node's own spin serves the action; this thread only reports
                handover = self._hand_over(x, y)
                if handover is not None:
                    self._send(connection, handover)
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
            # Now the heading, taped: only after a drive that ended on position alone (the RPP
            # pair's checker). Every other pair, a parking MPPI included, turns to the heading
            # itself and was held to it by the yaw-checking goal checker.
            ended_on_position = self.goal_checker == RPP_GOAL_CHECKER
            if status == 4 and ended_on_position and not self._cancelled_since(cancels):
                self._pivot_to(yaw_deg, connection, cancels)
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
            self._hand_back()
            with self._lock:
                self._driving = False

    def _cancelled_since(self, cancels: int) -> bool:
        """Whether a cancel came after the one counted as ``cancels``."""
        with self._lock:
            return self._cancels != cancels

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
        with self._lock:  # a cancel from here on stops the spin (cancel); before, it is done here
            late = self._cancels != cancels
            if not late:
                self._spin_handle = handle
        if late:
            handle.cancel_goal_async()
            self._send(connection, event | {"status": 0, "detail": "cancelled"})
            return
        outcome = self._wait(handle.get_result_async(), PIVOT_ALLOWANCE_S + 5.0)
        with self._lock:
            if self._spin_handle is handle:
                self._spin_handle = None
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
