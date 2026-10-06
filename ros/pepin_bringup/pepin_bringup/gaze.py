"""ROS 2 node on the laptop: the gaze arbiter, the one owner of every head decision.

Nothing else moves the neck. Consumers ask (:mod:`pepin.gaze`: bands, TTLs, preemption, home),
and this node alone talks to the board's base server, over its JSON-lines port: ``neck_target``
(joint angles, written while driving, each one renewing the board's lease, whose lapse sends the
head home and lets it go; renewed every ``target_renew_s``, never slower than half
config/neck.json's ``motion.lease_s``) to a base server whose state lines carry the neck's
encoders, ``neck_goto``/``neck_home`` (one move at a time, at rest) to an older one — picked by
itself from those lines (:class:`pepin.gaze.EitherHead`).

THE DOORS. The behaviour tree's AskGaze node calls ``/gaze/stall_look`` (std_srvs/Trigger) when
the controller has failed; the LLM tools POST requests to :data:`pepin.gaze_link.GAZE_PORT` on
this Mac's loopback (``/look``, ``/renew``, ``/state``). Path gaze and reverse gaze are this
node's own requests while a drive runs.

THE ONE GATE. ``/gaze/state`` (std_msgs/String, JSON; 10 Hz and on every change) carries the
phase (home, still, saccade, returning), ``since`` (when that phase began: a move at the write
that started it, a settled head at the reading it settled on), the head's pan and tilt, the
request holding it, and the blind interval: ``blind`` from the write that starts a move until the
head has settled plus one frame period, with ``blind_from`` and ``blind_until`` (null while the
head still moves), all in the board's clock. The frame consumers' gate
(:mod:`pepin.gaze_gate`) opens an interval at a blind state's ``since`` and closes it at the
settled phase's ``since`` plus its own ``gate_settle_s``.

THE STALL LOOK (:meth:`Gaze._stall`, behind ``stall_look``). The plan's first ``stall_ahead_m``
swept by the hull over the local costmap gives the lethal cells that block; the ones the lidar
does not back (camera-only or unexplained, :mod:`pepin.stall_look`) nearest along the plan are
the candidates; depth_fusion's ``/fusion/column`` says what the volume holds over them, and its
weighted centroid is where the head looks — ``frames`` still frames, then the columns again: the
verdict is carved, partly carved or confirmed, and a lidar-backed cell the look carved is
counted as a false carve. A blocker deeper than ``stall_max_depression_deg`` below the lens
answers FAILURE: the tree backs up 0.10 m and asks again. With a base server that moves the neck
only at rest the head is home again before the answer, so the drive never resumes with a turned
head; with ``neck_target`` it goes home as the drive resumes. Every stall look is one JSON line
on ``/gaze/stall`` and one log line.

PATH GAZE and REVERSE GAZE (behind ``path_gaze`` and ``reverse_gaze``, :mod:`pepin.path_gaze`)
run only with a base server that moves the neck while driving: with one that does not, a head
turned during a pause would drive on turned. Path gaze FOLLOWS in a zone
(:class:`pepin.path_gaze.PathFollower`: ``path_deadband_deg``, ``path_hyst_s``,
``path_cooldown_s``, ``path_tail_s``); while it has no aim — reversing before a reverse look, no
plan point ahead — its look is renewed where it is for up to ``path_hold_s``, so the head does
not lapse home and back; the next look replaces it or the drive's end lets it go. With
``glance_dwell_s`` above 0 the reverse look and the stall look are atomic GLANCES
(:class:`pepin.gaze.Look`'s ``hold_s``): the reverse look waits for ``reverse_frames`` clean
frames or ``glance_dwell_s`` once settled, the stall look for ``frames`` or its TTL, and only a
better band takes the head before. After a drive the head goes home at ``return_deg_s``.

A DRIVE'S START (a new goal on either navigator) drops every request but the operator's, so the
head goes home (or to path gaze) before the wheels turn; a drive's end drops the navigation
requests (a glance under way ends first).

THE LOOKS' FRAMES. Every look that held the head is booked when it lets go
(:class:`pepin.gaze.HeldLook`): the frames depth_fusion fused (``/fusion/frame``: past the depth
stream's gaze gate) while it held the head still. The report line counts them by source and
names every look that wrote none; a drive's end adds one line for the drive: its writes and its
looks.

Flags (:data:`FLAGS`) and knobs (config/knobs.json's ``gaze`` block), all live: ``ros/flags.sh
set gaze <name> <value>``; the report line every 30 s carries the counts, the driver, the last
stall look and the flags.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from action_msgs.msg import GoalStatusArray
from map_msgs.srv import GetPointMapROI
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger

from pepin.base_link import BASE_PORT
from pepin.deployment import config_file
from pepin.face_events import FaceSink, StallFace
from pepin.flags import Flag, FlagSet, load_knobs, with_knobs
from pepin.gaze import (
    DRIVE_REFUSAL,
    DRIVING,
    IDLE,
    LATE_S,
    NAVIGATION,
    OPERATOR,
    PERSON,
    Aim,
    Arbiter,
    BaseServerHead,
    EitherHead,
    GazeSettings,
    HeadReading,
    Look,
    LookTally,
    NeckTargetHead,
    Outcome,
    Reach,
    aim_at_point,
    depression_deg,
    home_aim,
    look_from_json,
)
from pepin.gaze_link import GAZE_PORT, JsonDoor
from pepin.marks_audit import scan_points, transform_xy
from pepin.neck import JOINT_NAMES, NeckConfig
from pepin.path_gaze import (
    PathFollower,
    PathGazeLaw,
    ReverseLaw,
    ReverseWatch,
    path_aim,
    remaining_m,
    reverse_aim,
    tight_rear,
    time_to_end,
)
from pepin.stall_look import (
    Evidence,
    centroid,
    column_box,
    evidence,
    find_blockers,
    in_columns,
    verdict,
)
from pepin.tsdf import RigidPose
from pepin_bringup.link import JsonLineLink
from pepin_bringup.msgs import fields_from_cloud, scan_arrays, stamp_seconds
from pepin_bringup.node_kit import Switches, TfLookup, spin_main

BOARD_HOST = "10.0.0.187"  # the board, as the ros/*.sh scripts default it (PEPIN_HOST wins)
STATE_TOPIC = "/gaze/state"
STALL_TOPIC = "/gaze/stall"
STALL_SERVICE = "/gaze/stall_look"
NECK_TOPIC = "/neck/state"
FRAME_TOPIC = "/fusion/frame"  # pepin_bringup.depth_fusion: each fused camera frame's stamp
COLUMN_SERVICE = "/fusion/column"  # ...and what the volume holds in a box
PLAN_TOPIC = "/plan"
COSTMAP_TOPIC = "/local_costmap/costmap"
SCAN_TOPIC = "/scan"
MARKS_TOPIC = "/depth_marks"
NAV_ACTIONS = ("navigate_to_pose", "navigate_through_poses")
ACTIVE_STATUSES = (1, 2, 3)  # accepted, executing, canceling
BASE_FRAME = "base_link"
STALL_SOURCE = "nav.stall"
PATH_SOURCE = "nav.path"
REVERSE_SOURCE = "nav.reverse"
RETURN_SOURCE = "nav.return"  # the slow way home after a drive

STEP_HZ = 20.0
STATE_HZ = 10.0
REPORT_S = 30.0
TF_WAIT_S = 0.2
COLUMN_WAIT_S = 1.0  # one /fusion/column answer
SCAN_MAX_AGE_S = 2.0  # a scan or a fan older than this explains no cell
ANSWER_SLACK_S = 2.0  # a door waits for a request's TTL plus the move timeout plus this
EXECUTOR_THREADS = 4  # the stall look blocks one while the subscriptions keep coming

FLAGS = FlagSet(
    Flag(
        "stall_look",
        False,
        description="when the controller fails, the behaviour tree's AskGaze asks for a look"
        " at the camera-only and unexplained marks blocking the hull in the plan's first"
        " stall_ahead_m: the head saccades to their voxel centroid, holds for 'frames' still"
        " frames so the volume carves a phantom or confirms a thing, comes home, and the tree"
        " clears and replans; off, AskGaze answers at once and the tree runs as before",
        why="off until the floor test of the proof of concept has measured it (a pillow under"
        " the nose, a person who stood in the path and left, the printer's phantom island:"
        " stall to wheels moving, recoveries per drive, carved against confirmed, false carves)."
        " The case for it: on run 0568 80 of 87 reversals came from the tree's blind recoveries"
        " (critic.md), and the volume carves a phantom at 1 m in 0.27 s of frames"
        " (config/fusion.json)",
        on_when="for the stall-look floor test, then on every drive once it passes",
        off_when="a look that makes the stall worse (the head slow to come home, a carve of"
        " something real): off, and the tree is exactly the one before",
    ),
    Flag(
        "path_gaze",
        False,
        description="while a drive runs, the head looks along the plan path_lookahead_s ahead"
        " (pan clamped to path_pan_clamp_deg, a dead-band of path_deadband_deg); only with a"
        " base server that moves the neck while driving (neck_target), otherwise idle",
        why="default by design, unmeasured: the arithmetic (design gaze 4.3) says 4-6 saccades of"
        " ~0.5 s on a 30 s drive, under 10 % of frames blind and dropped by the gaze gate, but no"
        " drive has measured that, nor the phantoms carved along the path before arrival",
        on_when="with the board's neck_target base server and the frame gates in place",
        off_when="a drive that loses VO or paints the volume wrong while the head moves",
    ),
    Flag(
        "reverse_gaze",
        False,
        description="a reverse leg longer than reverse_min_s, or any reverse with the rear"
        " tight, turns the head reverse_pan_deg toward the side the rear swings to; only with a"
        " base server that moves the neck while driving",
        why="off until the body self-filter lands: looking back at the working tilt the cart's"
        " own rear edge is 52 deg down, 3 deg inside the frame's bottom edge (config/neck.json's"
        " lens 1.203 m up, the hull's 0.30 m rear), so a look back paints the cart's top shelf"
        " into the volume as an obstacle; reverse_tilt_deg stays at home's 23.8 for that reason",
        on_when="after the self-filter, for the rear the lidar's masked wedges leave unseen",
        off_when="the volume grows marks on the cart itself while it reverses",
    ),
    Flag(
        "face_events",
        True,
        description="the stall look's moments on the head's face (pepin.face_events.StallFace,"
        " through the board's head server on PEPIN_HOST:3340): surprised as the head turns to"
        " the blocker, a small grin when the look carved a phantom, worried when it confirmed a"
        " thing (config/face.json's events)",
        why="on (Artem, 2026-10-05: the robot shows what it does): a moment never holds the look"
        " up (a send is dropped while the head server is away) and the head server drops a"
        " repeat within its 6 s gap",
        on_when="always, with the head on the robot",
        off_when="the face gets in the way of a test; a moment sent while off is simply not sent",
    ),
)


@dataclass
class _Inputs:
    """The newest messages a stall look reads, swapped whole under one lock."""

    costmap: Any = None
    plan: Any = None
    scan: Any = None
    marks: Any = None


def face_client(host: str) -> FaceSink:
    """The head server's door for the stall look's moments (``source`` gaze), on the board."""
    from pepin.head_link import HEAD_PORT, HeadClient

    return HeadClient(host, HEAD_PORT, source="gaze").start()


class Gaze(Node):
    """The gaze arbiter: requests in through the doors, one writer of the neck out."""

    def __init__(self) -> None:
        super().__init__("gaze")
        self._up = False
        host = str(self.declare_parameter("host", os.environ.get("PEPIN_HOST", BOARD_HOST)).value)
        port = int(self.declare_parameter("port", BASE_PORT).value)
        self._http_port = int(self.declare_parameter("http_port", GAZE_PORT).value)
        neck_path = str(self.declare_parameter("neck_config", str(config_file("neck.json"))).value)
        self._switches = Switches(self, with_knobs(FLAGS, load_knobs("gaze")))
        self._face_host = host
        self._face: StallFace | None = None
        if self._switches.on("face_events"):  # opened now: its first moment is not lost
            self._face = StallFace(face_client(host))
        self._cfg = NeckConfig.from_json(neck_path)
        self._reach = Reach.of(self._cfg)
        self._link = JsonLineLink(host, port, self._on_line, name="base server")
        self._head = EitherHead(
            BaseServerHead(self._cfg, self._link.send),
            NeckTargetHead(
                self._cfg,
                self._link.send,
                slow_deg_s=lambda: float(self._switches["slow_deg_s"]),
                # never slower than half the board's lease (config/neck.json's motion block)
                renew_s=lambda: min(
                    float(self._switches["target_renew_s"]), self._cfg.motion.lease_s / 2.0
                ),
                return_deg_s=lambda: float(self._switches["return_deg_s"]),
            ),
        )
        self._arbiter = Arbiter(self._head, home_aim(self._cfg), self._settings())
        self._tf = TfLookup(self, on_failure=self._on_tf_failure)
        self._inputs_lock = threading.Lock()
        self._inputs = _Inputs()
        self._goals: dict[str, set[bytes]] = {action: set() for action in NAV_ACTIONS}
        self._driving = False
        self._reverse = ReverseWatch()
        self._path_at = 0.0
        self._follow = PathFollower()
        self._path_quiet_since: float | None = None
        self._looks = LookTally()  # the report window's
        self._drive_looks: LookTally | None = None
        self._drive_writes = 0
        self._summary_at: float | None = None
        self._state_key: tuple[Any, ...] = ()
        self._last_stall = "none yet"
        self._stalls: dict[str, int] = {}
        self._tf_misses = 0
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._state_pub = self.create_publisher(String, STATE_TOPIC, reliable)
        self._stall_pub = self.create_publisher(String, STALL_TOPIC, reliable)
        self._column = self.create_client(GetPointMapROI, COLUMN_SERVICE)
        self.create_subscription(JointState, NECK_TOPIC, self._on_neck, reliable)
        self.create_subscription(Header, FRAME_TOPIC, self._on_frame, reliable)
        self.create_subscription(Path, PLAN_TOPIC, self._on_plan, 1)
        self.create_subscription(OccupancyGrid, COSTMAP_TOPIC, self._on_costmap, 1)
        self.create_subscription(
            LaserScan,
            SCAN_TOPIC,
            self._on_scan,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.create_subscription(
            LaserScan,
            MARKS_TOPIC,
            self._on_marks,
            QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE),
        )
        status_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for action in NAV_ACTIONS:
            self.create_subscription(
                GoalStatusArray,
                f"/{action}/_action/status",
                lambda msg, a=action: self._on_nav_status(a, msg),
                status_qos,
            )
        # The stall look waits for the head and for /fusion/column inside its own callback: its
        # group is its own, and the executor has threads to spare for everything else.
        self.create_service(
            Trigger,
            STALL_SERVICE,
            self._on_stall,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.create_timer(1.0 / STEP_HZ, self._step)
        self.create_timer(1.0 / STATE_HZ, self._publish_state)
        self.create_timer(REPORT_S, self._report)
        self._door = self._open_door()
        self._link.start()
        self.get_logger().info(
            f"gaze up: base server {host}:{port}, door :{self._http_port}, {STALL_SERVICE},"
            f" {STATE_TOPIC}; home pan 0 tilt {self._cfg.reference.pitch_deg:.1f} deg;"
            f" flags: {self._switches.state()}"
        )
        self._up = True

    def close(self) -> None:
        """Close the door, the link, the face's door and the TF listener, before the node is
        destroyed."""
        if self._door is not None:
            self._door.close()
        if self._face is not None:
            self._face.close()
        self._link.stop()
        self._tf.close()

    def _stall_face(self) -> StallFace | None:
        """The stall look's face while ``face_events`` is on (opened at the first moment when the
        flag came on live: that moment may be lost to the connection)."""
        if not self._switches.on("face_events"):
            return None
        if self._face is None:
            self._face = StallFace(face_client(self._face_host))
        return self._face

    # ---- settings ------------------------------------------------------------------------------
    def _knob(self, name: str) -> float:
        return float(self._switches[name])

    def _settings(self) -> GazeSettings:
        """The arbiter's numbers from the live knobs."""
        k = self._knob
        return GazeSettings(
            frames=int(self._switches["frames"]),
            settle_tol_deg=k("settle_tol_deg"),
            move_timeout_s=k("move_timeout_s"),
            frame_period_s=k("frame_period_s"),
            ttl_s=(
                k("ttl_operator_s"),
                k("ttl_navigation_s"),
                k("ttl_person_s"),
                k("ttl_sensor_s"),
                k("ttl_driving_s"),
                k("ttl_idle_s"),
            ),
        )

    def _path_law(self) -> PathGazeLaw:
        k = self._knob
        return PathGazeLaw(
            lookahead_s=k("path_lookahead_s"),
            min_m=k("path_min_m"),
            max_m=k("path_max_m"),
            deadband_deg=k("path_deadband_deg"),
            pan_clamp_deg=k("path_pan_clamp_deg"),
            near_m=k("path_near_m"),
            near_offset_deg=k("path_near_offset_deg"),
            hyst_s=k("path_hyst_s"),
            cooldown_s=k("path_cooldown_s"),
            tail_s=k("path_tail_s"),
            hold_s=k("path_hold_s"),
        )

    def _reverse_law(self) -> ReverseLaw:
        k = self._knob
        return ReverseLaw(
            pan_deg=k("reverse_pan_deg"),
            tilt_deg=k("reverse_tilt_deg"),
            min_s=k("reverse_min_s"),
            rear_m=k("reverse_rear_m"),
        )

    # ---- inputs --------------------------------------------------------------------------------
    @staticmethod
    def _now() -> float:
        """Wall seconds: the board follows this laptop's clock (chrony), and the readings and
        frames the arbiter compares with it carry the board's."""
        return time.time()

    def _on_line(self, message: dict[str, Any]) -> None:
        """The base server's lines (the link's reader thread): state lines and neck answers."""
        self._head.on_line(message, self._now())

    def _on_neck(self, msg: JointState) -> None:
        if not self._up:
            return
        names = list(msg.name)
        try:
            pan = float(msg.position[names.index(JOINT_NAMES[0])])
            tilt = float(msg.position[names.index(JOINT_NAMES[1])])
        except (ValueError, IndexError):
            return
        self._arbiter.observe(HeadReading(pan, tilt, stamp_seconds(msg.header.stamp)))

    def _on_frame(self, msg: Header) -> None:
        if self._up:
            self._arbiter.frame(stamp_seconds(msg.stamp))

    def _on_plan(self, msg: Path) -> None:
        if self._up:
            with self._inputs_lock:
                self._inputs.plan = msg

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        if self._up:
            with self._inputs_lock:
                self._inputs.costmap = msg

    def _on_scan(self, msg: LaserScan) -> None:
        if self._up:
            with self._inputs_lock:
                self._inputs.scan = msg

    def _on_marks(self, msg: LaserScan) -> None:
        if self._up:
            with self._inputs_lock:
                self._inputs.marks = msg

    def _on_tf_failure(self, kind: str, text: str) -> None:
        self._tf_misses += 1

    def _on_nav_status(self, action: str, msg: GoalStatusArray) -> None:
        """A navigator's status list: a new goal is a drive's start, no goal left its end."""
        if not self._up:
            return
        active = {
            bytes(bytearray(s.goal_info.goal_id.uuid))
            for s in msg.status_list
            if s.status in ACTIVE_STATUSES
        }
        before = set[bytes]().union(*self._goals.values())
        self._goals[action] = active
        after = set[bytes]().union(*self._goals.values())
        now = self._now()
        if after - before:
            self._driving = True
            self._follow.reset()
            self._path_quiet_since = None
            if self._summary_at is not None:
                self._drive_line()  # the last drive's, before its late looks are all in
            self._drive_looks = LookTally()
            self._drive_writes = self._arbiter.counts["writes"]
            gone = self._arbiter.release(
                now, "the drive's start", keep=lambda r: r.band == OPERATOR
            )
            # A head someone else moved (a jog, a hand) is taken home too: no drive starts crooked.
            adopted = self._arbiter.adopt(now, math.radians(self._knob("drive_home_tol_deg")))
            self.get_logger().info(
                f"gaze: a drive starts; {gone} requests let go"
                + (", the head found off home is sent there" if adopted else "")
            )
        elif before and not after:
            self._driving = False
            self._arbiter.release(
                now, "the drive's end", keep=lambda r: not r.source.startswith("nav.")
            )
            self._return_home(now)
            self._summary_at = now + LATE_S + 0.25  # the drive's last looks counted first

    def _return_home(self, now: float) -> None:
        """After a drive, home at ``return_deg_s`` rather than a saccade (0: the arbiter's own
        saccade home), when the head is off home at all."""
        if self._knob("return_deg_s") <= 0.0:
            return
        target = self._arbiter.state(now).target
        if target is None or target.off(self._arbiter.home) <= 1e-6:
            return
        ttl = self._knob("ttl_idle_s")
        self._arbiter.submit(
            Look(RETURN_SOURCE, (), IDLE, 0, 0.0, ttl, speed="return", kind="home"), now
        )

    # ---- the loop ------------------------------------------------------------------------------
    def _step(self) -> None:
        if not self._up:
            return
        now = self._now()
        self._arbiter.settings = self._settings()
        self._drive_gaze(now)
        self._arbiter.step(now)
        state = self._arbiter.state(now)
        key = (state.phase, state.blind, state.request_id)
        if key != self._state_key:  # every change goes out at once, between the 10 Hz lines
            self._state_key = key
            self._state_pub.publish(String(data=state.to_json()))
        self._book_looks(now)

    def _book_looks(self, now: float) -> None:
        """The looks that let go, into the report's tally and the drive's; the drive's line once
        its last looks are counted."""
        for booked in self._arbiter.take_looks(now):
            self._looks.add(booked)
            if self._drive_looks is not None:
                self._drive_looks.add(booked)
        if self._summary_at is not None and now >= self._summary_at:
            self._drive_line()

    def _drive_line(self) -> None:
        """One line for the drive that ended: its writes and its looks."""
        if self._drive_looks is not None:
            writes = self._arbiter.counts["writes"] - self._drive_writes
            self.get_logger().info(
                f"gaze: the drive's head: {writes} writes, {self._drive_looks.text()}"
            )
        self._drive_looks, self._summary_at = None, None

    def _publish_state(self) -> None:
        if self._up:
            self._state_pub.publish(String(data=self._arbiter.state(self._now()).to_json()))

    def _hold(self, source: str, aim: Aim, now: float) -> None:
        """A driving request (band 4) for ``aim``, renewed while it is the same aim."""
        same = [r for r in self._arbiter.pending() if r.source == source and r.views == (aim,)]
        if same:
            self._arbiter.renew(source, now)
            return
        ttl = self._knob("ttl_driving_s")
        self._arbiter.submit(Look(source, (aim,), DRIVING, 0, ttl, ttl), now)

    def _glance(self, source: str, aim: Aim, now: float) -> None:
        """A driving look that, with ``glance_dwell_s`` above 0, is an atomic glance of
        ``reverse_frames`` frames; one under way is only renewed, never re-aimed."""
        if self._arbiter.glancing(source):
            self._arbiter.renew(source, now)
            return
        same = [r for r in self._arbiter.pending() if r.source == source and r.views == (aim,)]
        if same:
            self._arbiter.renew(source, now)
            return
        ttl = self._knob("ttl_driving_s")
        frames = int(self._switches["reverse_frames"])
        hold = self._knob("glance_dwell_s")
        self._arbiter.submit(Look(source, (aim,), DRIVING, frames, ttl, ttl, hold_s=hold), now)

    def _drive_gaze(self, now: float) -> None:
        """Path gaze or reverse gaze, renewed every ``path_period_s`` while a drive runs."""
        v, w = self._head.twist
        law = self._reverse_law()
        self._reverse.update(v, w, now, law)
        if not self._driving or not self._head.moves_while_driving:
            return
        if now - self._path_at < self._knob("path_period_s"):
            return
        self._path_at = now
        if self._switches.on("reverse_gaze") and self._reverse.reversing:
            long_leg = self._reverse.reversing_for(now) >= law.min_s
            if long_leg or self._tight_rear(law):
                rear = reverse_aim(self._reverse.hold(), law, self._reach)
                self._glance(REVERSE_SOURCE, rear, now)
                return
        if not self._switches.on("path_gaze"):
            return
        target = None if self._reverse.reversing else self._path_target(v)
        if target is None:
            self._keep_path(now)  # a mode change, or nothing ahead: the aim stays
            return
        wanted, end_in = target
        fresh = not any(r.source == PATH_SOURCE for r in self._arbiter.pending())
        aim = self._follow.update(wanted, now, self._path_law(), end_in_s=end_in, fresh=fresh)
        if aim is None:
            return
        self._path_quiet_since = None
        self._hold(PATH_SOURCE, aim, now)

    def _keep_path(self, now: float) -> None:
        """Path gaze has no aim this period: its look is renewed where it is, for up to
        ``path_hold_s`` (0: it lapses after its TTL, as before)."""
        hold = self._knob("path_hold_s")
        if hold <= 0.0:
            return
        if self._path_quiet_since is None:
            self._path_quiet_since = now
        if now - self._path_quiet_since < hold:
            self._arbiter.renew(PATH_SOURCE, now)

    def _path_target(self, speed: float) -> tuple[Aim, float] | None:
        """Path gaze's aim from the newest plan and the cart's pose in its frame, and the
        seconds to the plan's end at this speed."""
        with self._inputs_lock:
            plan = self._inputs.plan
        if plan is None or len(plan.poses) < 2:
            return None
        here = self._tf.pose(plan.header.frame_id, BASE_FRAME, None, 0.0)
        if here is None:
            return None
        path = np.array([[p.pose.position.x, p.pose.position.y] for p in plan.poses], dtype=float)
        yaw = math.atan2(float(here.rotation[1, 0]), float(here.rotation[0, 0]))
        pose = (float(here.translation[0]), float(here.translation[1]), yaw)
        aim = path_aim(
            path,
            pose,
            speed,
            self._path_law(),
            lens_z_m=self._cfg.reference.z_m,
            home=self._arbiter.home,
            reach=self._reach,
        )
        if aim is None:
            return None
        return aim, time_to_end(remaining_m(path, (pose[0], pose[1])), speed)

    def _tight_rear(self, law: ReverseLaw) -> bool:
        with self._inputs_lock:
            grid = self._inputs.costmap
        if grid is None:
            return False
        here = self._tf.pose(grid.header.frame_id, BASE_FRAME, None, 0.0)
        if here is None:
            return False
        info = grid.info
        values = np.asarray(grid.data, dtype=np.int16).reshape(info.height, info.width)
        yaw = math.atan2(float(here.rotation[1, 0]), float(here.rotation[0, 0]))
        pose = (float(here.translation[0]), float(here.translation[1]), yaw)
        origin = (float(info.origin.position.x), float(info.origin.position.y))
        return tight_rear(values, origin, float(info.resolution), pose, law.rear_m)

    # ---- asking the arbiter --------------------------------------------------------------------
    def _ask(self, look: Look) -> Outcome:
        """Submit and wait for the final answer (from a thread that may wait)."""
        done = threading.Event()
        answers: list[Outcome] = []

        def finished(outcome: Outcome) -> None:
            answers.append(outcome)
            done.set()

        first = self._arbiter.submit(look, self._now(), finished)
        if first.status == "denied":
            return first
        patience = look.ttl_s + self._knob("move_timeout_s") + ANSWER_SLACK_S
        if not done.wait(patience):
            return Outcome(look.id, look.source, "expired", reason=f"no answer in {patience:.1f} s")
        return answers[0]

    def _driving_refusal(self, look: Look) -> str | None:
        """Why a person's or a sensor's look cannot move the head during a drive: only on a base
        server that moves the neck at rest alone."""
        if self._head.moves_while_driving or look.band < PERSON:
            return None
        if self._driving or self._head.wheels_moving:
            return DRIVE_REFUSAL
        return None

    # ---- the stall look ------------------------------------------------------------------------
    def _on_stall(self, _request: Any, response: Any) -> Any:
        """``/gaze/stall_look``: the tree's AskGaze. FAILURE (``success`` false) means back off
        first; everything else answers success, the verdict in ``message``."""
        if not self._switches.on("stall_look"):
            response.success, response.message = True, "stall_look off: nothing looked at"
            return response
        started = time.monotonic()
        try:
            success, line, record = self._stall()
        except Exception as exc:  # a failed look must never fail the drive
            success, line, record = True, f"the stall look failed: {exc!r}", {"error": repr(exc)}
        record.update(
            {
                "t": round(self._now(), 3),
                "success": success,
                "line": line,
                "took_ms": round((time.monotonic() - started) * 1000.0),
            }
        )
        self._stall_pub.publish(String(data=json.dumps(record)))
        word = str(record.get("verdict", "no look"))
        self._stalls[word] = self._stalls.get(word, 0) + 1
        self._last_stall = line
        self.get_logger().info(f"stall look: {line} ({record['took_ms']} ms)")
        response.success, response.message = success, line
        return response

    def _stall(self) -> tuple[bool, str, dict[str, Any]]:
        """The look itself: (success, the verdict line, the JSON record)."""
        with self._inputs_lock:
            inputs = _Inputs(**vars(self._inputs))
        record: dict[str, Any] = {}
        grid, plan = inputs.costmap, inputs.plan
        if grid is None or plan is None or len(plan.poses) < 2:
            return True, "no local costmap or no plan yet: nothing looked at", record
        frame = grid.header.frame_id
        here = self._tf.pose(frame, BASE_FRAME, None, TF_WAIT_S)
        to_grid = self._tf.pose(frame, plan.header.frame_id, None, TF_WAIT_S)
        if here is None or to_grid is None:
            return (
                True,
                f"no {frame} <- {BASE_FRAME} or plan frame in TF: nothing looked at",
                record,
            )
        path = np.array([[p.pose.position.x, p.pose.position.y] for p in plan.poses], dtype=float)
        path = transform_xy(path, to_grid.rotation, to_grid.translation)
        info = grid.info
        res = float(info.resolution)
        ahead = self._knob("stall_ahead_m")
        blockers = find_blockers(
            np.asarray(grid.data, dtype=np.int16).reshape(info.height, info.width),
            (float(info.origin.position.x), float(info.origin.position.y)),
            res,
            path,
            (float(here.translation[0]), float(here.translation[1])),
            self._placed(inputs.scan, frame),
            self._placed(inputs.marks, frame),
            ahead_m=ahead,
            margin_m=self._knob("stall_margin_m"),
            match_cells=self._knob("stall_match_cells"),
        )
        said = blockers.describe(ahead)
        record["blockers"] = {
            "cells": blockers.count,
            "lidar": int(blockers.lidar.sum()),
            "camera_only": int(blockers.camera.sum()),
            "unexplained": int(blockers.unexplained.sum()),
        }
        candidates = blockers.candidates(self._knob("stall_cluster_m"))
        if blockers.count == 0:
            return True, said, record
        if not candidates.any():
            return True, f"{said}: all the lidar's, which a look cannot carve", record
        # The marks' own band above the cart's floor plane: the floor's crossing stands under
        # every cell and is no blocker (the costmap's camera marks start at the same height).
        z0 = float(here.translation[2])
        band = (z0 + self._knob("stall_column_bottom_m"), z0 + self._knob("stall_column_top_m"))
        box = column_box(blockers.xy, res, band)
        before = self._column_points(box, frame)
        if isinstance(before, str):
            return True, f"{said}; {before}: nothing looked at", record
        points, weights = before
        cells, lidar_cells = blockers.xy[candidates], blockers.xy[blockers.lidar]
        over = in_columns(points, cells, res)
        target = centroid(points[over], weights[over])
        if target is None:
            record["verdict"] = "stale"
            return (
                True,
                f"{said}; the volume holds nothing over the {len(cells)} candidates: a stale"
                " mark, the clear takes it",
                record,
            )
        target_base = _into(here, target)
        aim = self._reach.clamp(aim_at_point(self._cfg, target_base))
        down = depression_deg(self._cfg, aim, target_base)
        record["centroid"] = [round(v, 3) for v in target]
        record["depression_deg"] = round(down, 1)
        if down > self._knob("stall_max_depression_deg"):
            record["verdict"] = "back off"
            return (
                False,
                f"{said}; the blocker is {down:.0f} deg below the lens, in the frame's last"
                " rows: back off first",
                record,
            )
        ttl = self._knob("ttl_navigation_s")
        face = self._stall_face()
        if face is not None:
            face.looking()
        # a glance (glance_dwell_s > 0): its frames or its TTL once settled, nothing cuts it short
        glance = ttl if self._knob("glance_dwell_s") > 0.0 else 0.0
        look = self._ask(
            Look(
                STALL_SOURCE,
                (aim,),
                NAVIGATION,
                int(self._switches["frames"]),
                0.0,
                ttl,
                kind="point",
                hold_s=glance,
            )
        )
        after = self._column_points(box, frame)
        seen_before = evidence(points, weights, cells, res)
        seen_after = (
            evidence(after[0], after[1], cells, res) if not isinstance(after, str) else None
        )
        lidar_before = evidence(points, weights, lidar_cells, res)
        lidar_after = (
            evidence(after[0], after[1], lidar_cells, res) if not isinstance(after, str) else None
        )
        word = verdict(seen_before, seen_after) if seen_after is not None else "unknown"
        if face is not None:
            face.verdict(word)
        home = None
        if not self._head.moves_while_driving:
            home = self._ask(Look(STALL_SOURCE, (), NAVIGATION, 0, 0.0, ttl, kind="home"))
        record.update(
            {
                "verdict": word,
                "look": look.as_dict(),
                "aim": aim.as_dict(),
                "before": vars(seen_before),
                "after": None if seen_after is None else vars(seen_after),
                "lidar_before": vars(lidar_before),
                "lidar_after": None if lidar_after is None else vars(lidar_after),
                "home": None if home is None else home.as_dict(),
            }
        )
        return (
            True,
            self._stall_line(
                said,
                target,
                frame,
                aim,
                look,
                seen_before,
                seen_after,
                word,
                lidar_before,
                lidar_after,
                home,
            ),
            record,
        )

    @staticmethod
    def _stall_line(
        said: str,
        target: tuple[float, float, float],
        frame: str,
        aim: Aim,
        look: Outcome,
        before: Evidence,
        after: Evidence | None,
        word: str,
        lidar_before: Evidence,
        lidar_after: Evidence | None,
        home: Outcome | None,
    ) -> str:
        x, y, z = target
        carved = lidar_before.occupied - lidar_after.occupied if lidar_after is not None else "?"
        return (
            f"{said}; centroid ({x:.2f}, {y:.2f}, {z:.2f}) in {frame}; looked pan"
            f" {math.degrees(aim.pan_rad):+.0f} tilt {math.degrees(aim.tilt_rad):.0f} deg:"
            f" {look.status}{' (' + look.reason + ')' if look.reason else ''},"
            f" {look.frames_seen} frames in {look.took_ms:.0f} ms; candidates {before.text()} ->"
            f" {after.text() if after is not None else 'no answer'}: {word}; lidar-backed cells"
            f" carved {carved}"
            + ("" if home is None else f"; home {home.status} in {home.took_ms:.0f} ms")
        )

    def _placed(self, scan: Any, frame: str) -> Any:
        """A fan's returns placed in ``frame`` at its own stamp; none when old or unplaceable."""
        if scan is None:
            return np.zeros((0, 2))
        if abs(self._now() - stamp_seconds(scan.header.stamp)) > SCAN_MAX_AGE_S:
            return np.zeros((0, 2))
        pose = self._tf.pose(frame, scan.header.frame_id, scan.header.stamp, TF_WAIT_S)
        if pose is None:
            return np.zeros((0, 2))
        angles, ranges = scan_arrays(scan)
        return transform_xy(scan_points(angles, ranges), pose.rotation, pose.translation)

    def _column_points(self, box: Any, frame: str) -> tuple[Any, Any] | str:
        """``/fusion/column`` over the box: (points, weights), or why there is no answer."""
        if not self._column.service_is_ready():
            return f"no {COLUMN_SERVICE} (depth_fusion)"
        request = GetPointMapROI.Request()
        request.x, request.y, request.z = box.centre
        request.l_x, request.l_y, request.l_z = box.size
        future = self._column.call_async(request)
        answered = threading.Event()
        future.add_done_callback(lambda _f: answered.set())
        if not answered.wait(COLUMN_WAIT_S):
            return f"{COLUMN_SERVICE} did not answer in {COLUMN_WAIT_S:.1f} s"
        cloud = future.result().sub_map
        if cloud.header.frame_id != frame:
            return f"the volume is in {cloud.header.frame_id}, the costmap in {frame}"
        fields = fields_from_cloud(cloud)
        points = np.column_stack((fields["x"], fields["y"], fields["z"]))
        return points, fields["weight"]

    # ---- the door ------------------------------------------------------------------------------
    def _open_door(self) -> JsonDoor | None:
        try:
            return JsonDoor(
                "0.0.0.0",
                self._http_port,
                {
                    ("POST", "/look"): self._door_look,
                    ("POST", "/renew"): self._door_renew,
                    ("GET", "/state"): self._door_state,
                },
            ).start()
        except OSError as exc:
            self.get_logger().error(f"gaze door not opened on :{self._http_port}: {exc}")
            return None

    def _door_look(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /look``: a request, answered when it ends."""
        look = look_from_json(body, self._arbiter.settings, self._cfg, self._to_base)
        if isinstance(look, str):
            return {"status": "denied", "reason": look, "source": body.get("source", "")}
        why = self._driving_refusal(look)
        if why is not None:
            return Outcome(look.id, look.source, "denied", reason=why).as_dict()
        return self._ask(look).as_dict()

    def _door_renew(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /renew``: restart the TTL of a source's requests."""
        return {"renewed": self._arbiter.renew(str(body.get("source", "")), self._now())}

    def _door_state(self, _body: dict[str, Any]) -> dict[str, Any]:
        """``GET /state``: the state, the live requests, the driver and whether a drive runs."""
        state = self._arbiter.state(self._now()).as_dict()
        state["pending"] = [
            {"source": r.source, "band": r.band, "kind": r.kind, "id": r.id}
            for r in self._arbiter.pending()
        ]
        state["driver"] = "neck_target" if self._head.speaks_target else "neck_goto"
        state["driving"] = self._driving
        state["reach"] = {
            "pan_rad": list(self._reach.pan),
            "tilt_rad": list(self._reach.tilt),
            "home": self._arbiter.home.as_dict(),
        }
        return state

    def _to_base(self, frame: str, xyz: dict[str, float]) -> tuple[float, float, float] | str:
        """A point of ``frame`` in base_link, now."""
        point = (xyz["x"], xyz["y"], xyz["z"])
        if frame == BASE_FRAME:
            return point
        pose = self._tf.pose(BASE_FRAME, frame, None, TF_WAIT_S)
        if pose is None:
            return f"no {BASE_FRAME} <- {frame} in TF"
        placed = pose.rotation @ np.asarray(point, dtype=float) + pose.translation
        return float(placed[0]), float(placed[1]), float(placed[2])

    # ---- the report ----------------------------------------------------------------------------
    def _report(self) -> None:
        c = self._arbiter.counts
        state = self._arbiter.state(self._now())
        link = "connected" if self._link.connected else "DOWN"
        stalls = ", ".join(f"{word} {n}" for word, n in sorted(self._stalls.items())) or "none"
        self.get_logger().info(
            f"gaze: driver {'neck_target' if self._head.speaks_target else 'neck_goto'}"
            f" (base server {link}), phase {state.phase}"
            f"{' ' + state.source if state.source else ''}, driving {self._driving};"
            f" requests {c['requests']} (done {c['done']}, denied {c['denied']}, preempted"
            f" {c['preempted']}, expired {c['expired']}); writes {c['writes']}, settled"
            f" {c['settled']}, refused {c['refused']}, timeouts {c['timeouts']};"
            f" {self._looks.text()}; stall looks: {stalls}; last: {self._last_stall}"
            f"{f'; {self._tf_misses} TF misses' if self._tf_misses else ''};"
            f" flags: {self._switches.state()}"
        )
        self._looks = LookTally()


def _into(pose: RigidPose, point: tuple[float, float, float]) -> tuple[float, float, float]:
    """A point of the pose's parent frame in the pose's own frame (base_link from odom)."""
    local = pose.rotation.T @ (np.asarray(point, dtype=float) - pose.translation)
    return float(local[0]), float(local[1]), float(local[2])


def main(args: list[str] | None = None) -> None:
    """Entry point: spin on several threads (the stall look waits inside its service)."""
    spin_main(Gaze, args, executor=lambda: MultiThreadedExecutor(num_threads=EXECUTOR_THREADS))


if __name__ == "__main__":
    main()
