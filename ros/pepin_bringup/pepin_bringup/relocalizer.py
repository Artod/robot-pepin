"""Global relocalisation on top of AMCL: the robot always knows where it is, no human needed.

AMCL tracks well but only searches where its particles already are: carry the
cart across the room, push it, lift it onto the carpet, and it keeps believing
the old pose. This node closes that gap with the correlative whole-map search
from :mod:`pepin.localization` (a fraction of a second on the pooled grid, with
the twin check). Every second it scores how well the current scan lies on the
map at the tracked pose; when the fit stays poor it searches the whole map and,
if a clearly better pose exists, re-seeds the tracker. The operator seeds it the
same way by publishing on ``/initialpose`` (Foxglove's pose estimate); the node
never publishes there itself. The same search answers the ``/relocalize``
service on demand, and ``/where_am_i`` reports pose and fit as text.

Frames: the scan is transformed into ``base_link`` with the static laser
transform looked up once; poses are in ``map``.

Sources: the lidar's revolution is matched here, on the board, through the one trigger path
(:class:`pepin.sources.SourceFeed`) — the scan waits at its gate until the odometry covers its
whole revolution and then drives an update. The camera is a source too, but its scans are
matched on the laptop that produces them (:mod:`pepin_bringup.laptop_localizer`) and only the
POSE they measured arrives here, on ``/localization/measurement``: this node carries each
measurement to the moment of its next update over the same odometry history a riding scan would
have been carried along, and the tracker fuses it by information beside the lidar's own match
(:mod:`pepin.measurements`). Matching those scans HERE is what the day of 2026-09-13 measured
and refused: three matches a revolution, 147 ms instead of 45, every second revolution dropped
(4.7 Hz) and the live pose 50 cm p90 off the lidar's truth. With the lidar stale or absent the
measurements drive the updates by themselves, so a dead lidar still hands the tracker to the
camera without a restart; with the link down there are simply no measurements and the tracker
is the lidar-only one it always was. ``/localization/sources`` carries every source's word on
each update as JSON for the operator. The watch, the whole-map search and the first fix run on
full revolutions only (:meth:`pepin.sources.SourceFeed.full_picture`).

The watchdog: the same whole-map search runs CONTINUOUSLY on the laptop, which does it in a
tenth of a second instead of seconds (:mod:`pepin_bringup.laptop_localizer`), and its answers arrive
here on ``/localization/candidate``. Each one is first carried from the moment of its own scan
to this moment over the odometry between the two (``carry_candidates``) — a search plus a
wireless hop is a quarter of a second, and on a driving cart that is the difference between
where the cart was and where it is — and then :class:`pepin.watchdog.CandidateGate` judges it
against this tracker's own pose and fit; a streak of candidates from different scans that
disagree with the tracker and agree with each other re-seeds it through the pending seed — the
very path this node's own search uses, with the ``accept_candidates`` flag as the switch.
Nothing here depends on the laptop: with no candidate arriving, the board's own slow search is
the fallback it always was.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from typing import Any

import numpy as np
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.msg import ParticleCloud
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformBroadcaster

from pepin.dynamic import StaticMask, berth_for, dynamic_marks, occluded, toe_reach_m
from pepin.flags import Flag, FlagSet
from pepin.fusion import sigma_from_fit
from pepin.localization import SWITCHES as TRACKER_SWITCHES
from pepin.localization import Localizer
from pepin.mapping import OccupancyGrid
from pepin.measurements import (
    MEASUREMENT_MAX_AGE_S,
    MeasurementGate,
    RemoteMeasurement,
)
from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import CorrelativeMatcher, SearchWindow
from pepin.slip import SlipWatch
from pepin.sources import CAMERA, CONTACT, DEPTH, LIDAR, ScanObservation, SourceFeed, SourceRegistry
from pepin.timeline import (
    MatchPacer,
    MotionEdge,
    MotionFilter,
    OdomHistory,
    TimedScan,
    deskewed,
    standing_still,
    timed_scan_from_ros,
)
from pepin.watch import DRIVE_FIT, LOST_FIT, LostWatch, Verdict
from pepin.watchdog import CANDIDATE_STREAK, CandidateGate, GlobalCandidate
from pepin_bringup.msgs import (
    cloud_from_points,
    grid_from_msg,
    map_id,
    planar_mount,
    pose_with_covariance,
    stamp_from_seconds,
    transform_from_rpy,
    yaw_of,
)
from pepin_bringup.node_kit import Switches, TfLookup, spin_main

DUMP_DIR = (
    "/maps/rec"  # every failed whole-map search leaves its scan here, for the offline autopsy
)
LAST_POSE_FILE = "/maps/last_pose.json"  # where the robot stood when the stack last ran
LAST_POSE_MAX_AGE_S = 3600.0
SLIP_SAID_AFTER = 3  # consecutive slipping scans before the log says it once
# The laptop's whole-map watchdog (pepin_bringup.laptop_localizer) publishes a candidate here
# about once a second, as one self-contained JSON message (pepin.watchdog.GlobalCandidate): the
# board
# never has to join two topics to judge one answer, and a message that does not parse is counted,
# not obeyed. The board's own slow search is untouched: it is the fallback when none arrives.
CANDIDATE_TOPIC = "/localization/candidate"
# ...and on this one, several times a second, the pose it measured out of a camera scan
# (pepin.measurements.RemoteMeasurement): the same shape of message, judged the same way — a
# map id that is not ours is refused, a message that does not parse is counted, not obeyed.
MEASUREMENT_TOPIC = "/localization/measurement"


def map_to_odom(pose: Pose2D, odom: Pose2D) -> tuple[float, float, float]:
    """The map -> odom transform (x, y, yaw) that puts the robot, seen at ``odom`` in the odom
    frame, at ``pose`` in the map frame: T_map_odom = T_map_base * inv(T_odom_base)."""
    yaw = math.atan2(math.sin(pose.theta - odom.theta), math.cos(pose.theta - odom.theta))
    c, s = math.cos(yaw), math.sin(yaw)
    return pose.x - (c * odom.x - s * odom.y), pose.y - (s * odom.x + c * odom.y), yaw


# The tracker's flags: what applies without a restart (CLAUDE.md rule 19). Each is a switch of
# the Localizer (pepin.localization: an attribute, or the roster's ``sources``) and the flag
# callback writes it there (``Localizer.switch``); the next map's tracker is built with the
# current values. Every other parameter is refused live (the answer names it), because a
# "success" that changed nothing is a lie.
FLAGS = FlagSet(
    Flag(
        "rest_lock",
        True,
        description="hold the pose while the cart stands still (wheels quiet 0.6 s and the gyro"
        " under 1.5 deg/s): a match's residual is blended in with a time constant instead of"
        " taken whole",
        why="the best-measured switch in the tracker. On tape 0182 (6 s at rest, a full turn, 8 s"
        " at rest) the published pose's rest band goes from 8.34 deg with the lattice alone to"
        " 0.51 deg with the rest lock, and on the robot it reads 0.42 deg at sd 0.13; while"
        " driving (tape 0170) it took p90 |yaw rate - gyro| from 7.74 to 4.86 deg/s and the"
        " correction's sd from 0.45 to 0.29 deg. It costs +0.04-0.08 ms a scan. Flipped live in"
        " the demo: 9 deg to 0.3",
        on_when="always: a still cart whose pose wanders is the first thing a watcher sees",
        off_when="to show what the raw match does (the demo's A/B), or where the cart is carried"
        " by hand and the wheels have no say in whether it stands still",
    ),
    Flag(
        "explained_vote",
        True,
        description="returns the static map cannot explain (a person, a moved chair) do not score"
        " the match",
        why="alone it took the rest band from 10.5 deg to 2.38, and with the rest lock and the"
        " sub-cell refinement to 0.51 (tape 0173). The mask is dropped when fewer than half the"
        " returns are explained or fewer than 60 come back, because a vote taken only on what"
        " already fits is mildly self-confirming",
        on_when="in a room with people and furniture that moves — the room this robot lives in",
        off_when="in an empty room where every return should count, or to measure what a crowd"
        " costs the match",
    ),
    Flag(
        "rest_tau_s",
        6.0,
        description="the rest lock's time constant: seconds for a residual to die at rest",
        why="6 s against the 3 s first tried: at the node's roughly 1 Hz rest cadence 3 s let"
        " 2.5x more match noise through, and 6 s halves that while still converging a nudge in"
        " seconds — the rest band reads 0.37 deg at 6 s",
        on_when="lengthen it for a cart that stands for minutes and must not drift at all",
        off_when="shorten it when a nudged cart has to take its new pose quickly — a demo where"
        " the cart is pushed by hand",
        range=(0.1, 60.0),
    ),
    Flag(
        "rest_gain",
        0.05,
        description="the rest lock's share per match when no match cadence is known",
        why="default by design, unmeasured on its own: 0.05 against the driving gain of 0.5 was a"
        " guess, kept after the robot run because the rest bands it is inside of (0.42-0.51 deg)"
        " came out right. It has never been swept",
        on_when="raise it towards the driving gain when the rest lock is too slow to accept a"
        " real correction",
        off_when="0 lets no match move the pose while the cart stands: a hard hold, and a way to"
        " see how far the odometry alone wanders",
        range=(0.0, 1.0),
    ),
    Flag(
        "sources",
        (LIDAR,),
        description="what corrects the pose: the lidar's revolution (/scan), matched here, and"
        " the camera (`camera`), whose scans the laptop matches and whose ANSWER arrives on"
        " /localization/measurement. The lidar drives the updates while it is fresh and the"
        " camera's word rides along, carried to its moment; a stale lidar hands the updates to"
        " the measurements. `depth` and `contact` name the camera's raw scans, which this node"
        " no longer subscribes to — enabling them changes nothing here",
        why="the lidar alone, because the camera cannot carry the map by itself: replayed on run"
        " 0171 against flat3 the depth band alone loses the map in 0.5 s (122 cm, 124 deg) and"
        " the contact line alone in 12 s (80 cm, 28 deg) — the camera's 0.15-1.3 m band is a"
        " different cross-section of the room than the lidar's 0.2 m map, so a look-alike place"
        " scores fit 0.90 at its own match and 0.12 at the truth. Fused with the lidar and gated"
        " on disagreement, all three together stay within 0.7/1.6/5.7 cm and 0.21/0.56/1.9 deg of"
        " lidar-only and never lose the map (scratch/camera_only_localization.py). `camera` is"
        " that same fusion with the matching moved to the laptop: on this board the raw scans"
        " took the tracker to 147 ms and 4.7 Hz and the live pose 50 cm p90 off the lidar's truth"
        " (scratch/drive_bisect.py, runs 0238-0241), while a measurement costs a matrix inverse",
        on_when="add `camera` where the lidar is blocked or blind — parked bumper to furniture,"
        " or a lidar that stopped: the fusion is measured and gated, and the board pays nothing"
        " for it",
        off_when="drop a source the moment /localization/sources shows it disagreeing with the"
        " others; the lidar alone is the safe state, and it is what the board falls back to by"
        " itself when the link dies. `depth`/`contact` stay on the roster because the library"
        " still matches those scans where there is CPU for it — an offline replay"
        " (scratch/camera_only_localization.py), another robot — not because this board will",
        choices=(LIDAR, DEPTH, CONTACT, CAMERA),
    ),
    Flag(
        "measurement_max_age_s",
        MEASUREMENT_MAX_AGE_S,
        description="how old a pose measurement from the laptop may be, in seconds, at the"
        " moment of the update that would take it: past this it is dropped instead of carried",
        why="the number the day of 2026-09-13 asked for: the camera's word pulled the live pose"
        " 50 cm p90 off the truth while the board matched at 4.7 Hz with 147 ms per scan, and"
        " every one of those measurements was fused as if it spoke for the moment it was used"
        " at. On the new path a measurement is 0.1-0.3 s old when an update takes it (a camera"
        " frame at 5 Hz plus the link), so half a second is the slack around that, not a"
        " threshold anybody has hit; the failure it is against — a bridge that stalls and"
        " delivers a burst — is seconds",
        on_when="raise it only to see what a stale measurement does; the carry over odometry is"
        " honest for as long as the odometry is",
        off_when="lower it towards the measurement's own age (0.3 s) where the cart drives fast"
        " and a carry over a tenth of a second is already a decimetre",
        range=(0.05, 5.0),
    ),
    Flag(
        "fusion",
        True,
        description="fuse every enabled source's word by its information — a match made here, a"
        " measurement made on the laptop; off: the widest source corrects alone and the others"
        " only report",
        why="with all three sources the fused pose stays within 0.7-5.7 cm of lidar-only and"
        " never loses the map. One defect was found and fixed on the way: an edge-bound lidar"
        " used to be out-voted by a blind fan's plateau, so the anchor's bound is now taken alone"
        " — a 12 cm slip at rest is carried by the second match instead of held for 2.5 s, and"
        " recovery while driving is 3.2 cm against lidar-only's 2.9",
        on_when="whenever more than one source is enabled",
        off_when="to see which source is actually moving the pose: off, the others still report",
    ),
    Flag(
        "toe_reach",
        toe_reach_m(),
        description="how far past the leg the lidar sees a standing person's toe reaches, metres:"
        " the term the dynamic rings are sized on (pepin.dynamic.berth_for). The default is"
        " computed from the lidar's mount (config/lidar.json)",
        why="one measured number and three assumed ones: the mount is 0.383 m by tape, and the"
        " reach is 0.21 + (z - 0.07) tan 10 deg — a 28 cm shoe whose ankle sits 7 cm back, a shin"
        " leaning 10 degrees — typed anthropometry that has never been measured against a person"
        " in front of this cart. It matters in metres: a point planner's ring is 0.41 m at the"
        " 0.20 that stood here before and 0.48 m at this 0.27. The flag exists because the cart"
        " once ran over feet",
        on_when="raise it for boots, or for a cart that must give more room: every dynamic ring"
        " widens by the same amount",
        off_when="lower it to compare berths in the field without a restart; 0 rings only what"
        " the beams themselves see",
        range=(0.0, 0.6),
    ),
    Flag(
        "near_rings",
        True,
        description="a return is ringed as soon as it clears the cart's own outline, and only the"
        " marks that would land on that outline are dropped; off, nothing within the ring plus"
        " the outline is ringed at all — the older rule, whose blind disc grows with the ring",
        why="exact geometry, no field A/B of the two rules. The old rule blanks a disc of ring +"
        " 0.457 m (the cart's circumscribed radius plus one costmap cell), so at the ring today's"
        " reach asks for, 0.48 m, a person standing 0.90 m ahead would not be ringed at all — the"
        " very case the ring exists for. The new rule trims only the marks that land on the"
        " cart's own outline, which is what the run-0087 failure actually was: a mark on itself"
        " that refuses its every command",
        on_when="wherever a person may come within a metre of the cart — the close approach this"
        " robot is built for",
        off_when="to reproduce the older rule side by side; remember its blind disc grows with"
        " the ring (7 cm of extra ring stopped a person at 0.90 m from being ringed at all)",
    ),
    Flag(
        "accept_candidates",
        True,
        description=f"re-seed from the laptop watchdog's whole-map candidates ({CANDIDATE_TOPIC},"
        " pepin.watchdog): a place that disagrees with the tracked pose candidate_streak times in"
        " a row, about the same place each time, is adopted through the path the board's own"
        " search uses",
        why="measured on the kidnap tape (run 0171: the odometry jumps 1 m and 40 deg while the"
        " scans do not) this is the difference between coming back in 2.9 s over 28 scans and"
        " never coming back — the tracker's own window still had 0.69 m of error after 39 s, and"
        " the board's own searches found the truth four times (fits 0.76/0.72/0.79/0.75 against"
        " the tracker's 0.50-0.60) and died unconfirmed each time, because at a metre off this"
        " flat still fits 0.53, just under the 0.55 that declares the cart lost. Over the"
        " undisturbed tape it re-seeded 0 times, and against another flat's map 9 of 12"
        " candidates were called unknown_map",
        on_when="whenever the laptop's watchdog runs and the map is the right one",
        off_when="where a teleport is more dangerous than being lost — under a live goal, or in a"
        " room the map does not cover: off, the candidates are still judged, counted and reported",
    ),
    Flag(
        "candidate_streak",
        CANDIDATE_STREAK,
        description="how many candidates in a row must disagree with the tracker and agree with"
        " each other before one of them re-seeds it: the price of a teleport, in seconds",
        why="deliberate conservatism above a measurement that was neutral: on the kidnap tape a"
        " streak of 1 recovered in 0.8 s (8 scans) and this streak of 3 in 2.9 s (28 scans), and"
        " both re-seeded 0 times over the undisturbed tape, where the pose never left the"
        " reference by more than 0.000 m. Nothing measured prefers 3; the argument is that a"
        " look-alike keeps looking alike, so one agreement is not proof",
        on_when="raise it in a room of look-alike corners, where a wrong teleport costs more than"
        " three seconds of being lost",
        off_when="1 is the fastest recovery measured (0.8 s) and on that tape just as safe — the"
        " value to try when a demo has to show the cart coming back",
        range=(1, 10),
    ),
    Flag(
        "carry_candidates",
        True,
        description="a candidate's pose is moved from the moment of its own scan to now over the"
        " odometry between the two stamps (pepin.watchdog.carried) before it is judged and fused,"
        " and one the odometry history no longer covers is dropped",
        why="by argument from a measured latency, not by a measured gain: a whole-map search"
        " takes 0.12-0.25 s plus a wireless hop, so at 0.8 m/s an uncarried pose is installed"
        " about 20 cm backwards along the drive every time — a bias, not noise. On the kidnap"
        " tape the carry changes nothing measurable (2.9 s, 28 scans, 0 false re-seeds either"
        " way), because that cart was barely moving when it was lost",
        on_when="whenever the cart may re-seed while driving",
        off_when="only to reproduce the old behaviour, where the pose the laptop measured a"
        " search and a hop ago is installed as the pose now",
    ),
    Flag(
        "distinct_scans",
        True,
        description="a streak is counted in scans, not in messages: a candidate whose scan id is"
        " already in the run is a second opinion that heard the first one's scan, counted as"
        " replay and not lengthening the streak",
        why="the failure it answers is real: a frozen /scan on the laptop published the same"
        " search answer once a second and the board counted three of them as three seconds of"
        " evidence — the same replay that fooled the board's own two-search rule on 2026-09-09."
        " On the kidnap tape it costs nothing (2.9 s, 28 scans unchanged). A sender that names no"
        " scan says id 0, and a repeated 0 reads as replay too",
        on_when="wherever the candidates cross a bridge that can freeze — which is this robot's",
        off_when="only to reproduce the old counting, where one scan's answer repeated could"
        " re-seed the tracker",
    ),
)
BERTH_FLAGS = ("toe_reach", "near_rings")  # the flags that resize the berth, not the tracker


class _RosLogHandler(logging.Handler):
    """Forwards the Python-side localizer's log lines to the node's ROS logger."""

    def __init__(self, node: Node) -> None:
        super().__init__()
        self._node = node

    def emit(self, record: logging.LogRecord) -> None:
        text = f"[{record.name}] {record.getMessage()}"
        if record.levelno >= logging.WARNING:
            self._node.get_logger().warning(text)
        else:
            self._node.get_logger().info(text)


class Relocalizer(Node):
    """Tracks the pose on the map from every enabled scan source and owns map -> odom; watches
    the fit and re-seeds itself (and AMCL) from a whole-map search when it stays poor."""

    def __init__(self) -> None:
        super().__init__("relocalizer")
        self._scan_topic = str(self.declare_parameter("scan_topic", "/scan").value)
        self._min_inliers = float(self.declare_parameter("min_global_inliers", 0.45).value)
        self._check_period_s = float(self.declare_parameter("check_period_s", 1.0).value)
        self._watch_args = dict(
            lost_fit=float(self.declare_parameter("lost_fit", LOST_FIT).value),
            lost_checks=int(self.declare_parameter("lost_checks", 3).value),
            cooldown_s=float(self.declare_parameter("search_cooldown_s", 8.0).value),
        )
        self._watch = LostWatch(**self._watch_args)  # type: ignore[arg-type]
        # One lock for the episode: the claim of _searching, every _watch call and the
        # _pending_seed swap. The worker, the executor's timers and the service thread all
        # touch these, and the single-threaded executor was the only thing serialising them.
        self._episode = threading.Lock()
        # Tracking: every scan corrects the pose by scan matching around the wheels' prediction
        # and this node owns map -> odom (AMCL then only paints particles). The wheels are trusted
        # for one scan interval, 0.1 s, where even a 25% yaw error is a fraction of a degree.
        self._track = bool(self.declare_parameter("track", True).value)
        # The three levers on the tracker's heading jitter, each switchable so a run can be
        # compared with and without it (scratch/tracker_rest_band.py measures all three offline):
        #   subcell_refine  the matcher answers between its candidates (parabola over the score)
        #   rest_lock       a standing cart averages the match in slowly instead of re-deciding
        #   explained_vote  returns the static map cannot explain do not score the match
        self._subcell_refine = bool(self.declare_parameter("subcell_refine", True).value)
        # The rest lock averages in seconds, not in matches: a standing cart is matched about
        # once a second (MotionFilter below), a replay feeds every scan, and both must settle
        # at the same speed. rest_gain is only the fallback for a caller that times nothing.
        # LIVE_PARAMS apply live (ros2 param set /relocalizer rest_lock false): a demo compares
        # them without a stack restart; subcell_refine is the matcher's construction and takes
        # effect at the next start. The switches are built at the end of this method, after the
        # last declare_parameter, and hold the values the next map's tracker is built with.
        self._odom_wz = 0.0  # newest fused yaw rate (gyro-driven): the second witness of rest
        # map -> odom is published 20 times a second, dated 0.1 s ahead (AMCL's habit, shorter):
        # a consumer asking for "now" always finds a transform and never extrapolates.
        self._tf_future_s = float(self.declare_parameter("tf_future_s", 0.1).value)
        self._tracker_initialised = False
        self._tracker_initialising = False
        # 0.05 s: the lidar turns at 10 Hz, and a gap of 0.15 s used to skip every other scan;
        # the pacer's busy rule (skip as long as a slow match took) is what protects the board.
        self._pacer = MatchPacer(
            min_gap_s=float(self.declare_parameter("min_match_gap_s", 0.05).value)
        )
        self._last_match_stamp_s: float | None = None  # scan stamp of the previous match: dt_s
        self._last_scan_age_s = 0.0
        self._last_map_odom = (0.0, 0.0, 0.0)  # the belief until the first fix: the base
        self._slip = SlipWatch()  # wheels claiming a step the picture does not show
        self._map_id = ""
        self._pending_seed: tuple[str, Pose2D, float] | None = None
        self._scan_id = 0
        # Time alignment (pepin.timeline): the odometry is kept as a history and every scan waits
        # at the gate until the history covers its whole revolution, so a scan is matched against
        # the pose it was taken at, beam by beam — never against the newest pose. The TF lookup at
        # the scan's stamp used to fail 118 times in 119 (the EKF runs 35-70 ms behind the lidar)
        # and fell back to "now": 1-2 degrees of false correction per scan in every pivot, with the
        # sign of the turn, and the cart steered by the wobble (runs 0080-0083, 2026-09-09).
        self._odom_topic = str(self.declare_parameter("odom_topic", "/odometry/filtered").value)
        self._history = OdomHistory(horizon_s=5.0)
        # Every source's scans wait at the feed (a gate per source) and one of them drives the
        # update: the lidar while it is fresh, else the camera. The roster is shared with the
        # tracker, so the ``sources`` flag switches both at once.
        self._registry = SourceRegistry()
        self._feed = SourceFeed(self._registry, max_wait_s=0.5)
        # The camera's word, measured on the laptop: the newest per source waits here until an
        # update takes it, carried to that update's moment over the same history a riding scan
        # would have been carried along (pepin.measurements).
        self._measurements = MeasurementGate(self._registry)
        self._motion = MotionFilter(min_m=0.005, min_deg=0.3, max_gap_s=1.0)
        self._rested = 0  # scans left unmatched because the cart stood still (per report)
        # What the map does not explain (a person, a moved chair) is published as lethal rings for
        # both costmaps: the point planners' berth around new objects (pepin.dynamic).
        self._static_mask: StaticMask | None = None
        self._dynamic_pub = self.create_publisher(PointCloud2, "/dynamic_obstacles", 5)
        self._dynamic_count = 0  # marks published since the last report
        self._planner_id = "GridBased"  # until the goal server says who plans
        self._berth = berth_for(self._planner_id)  # resized once the switches exist
        self.create_subscription(
            String,
            "planner_selector",
            self._on_planner,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._deskew_failed = 0  # scans matched raw because the history had a hole (per report)

        self._motion_edge = MotionEdge()  # odom->base_link moved since the previous check
        self._navigating = False  # a NavigateToPose goal is executing
        self._grid: OccupancyGrid | None = None
        self._matcher: CorrelativeMatcher | None = None
        logging.getLogger("pepin").addHandler(
            _RosLogHandler(self)
        )  # localizer reasons in the ROS log
        self._localizer: Localizer | None = None
        self._laser_tf: tuple[float, float, float, bool] | None = None  # x, y, yaw, mirrored
        self._searching = False
        self.fit = float("nan")
        # The laptop's watchdog: its candidates are judged against this tracker's own pose and
        # fit, and a streak of disagreements about one place re-seeds through _pending_seed —
        # the very path the board's own search uses. All the judging is in pepin.watchdog.
        self._candidates = CandidateGate()
        # The watch judges full revolutions only; while a fan drives it is off and the report
        # line says so (the fit is still measured and published, on the fan).
        self._watch_on = True
        # Stage one: a wide window around the pose AMCL believes in (a push, a short carry); the
        # whole map only when that fails. On four A53 cores the whole-map lattice takes ~15 s,
        # the window about a second.

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(OccupancyGridMsg, "/map", self._on_map, latched)
        # Depth 1: a match takes 40 ms and scans come every 100 ms; a deeper queue let the tracker
        # fall half a second behind reality and lose the lock in every turn (2026-09-06).
        self.create_subscription(
            LaserScan,
            self._scan_topic,
            self._on_scan,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.create_subscription(Odometry, self._odom_topic, self._on_odom, 20)
        # Depth 1: a candidate is a snapshot of a moment, and the newest one is the only one
        # worth judging; a backlog of them would re-seed from a scan seconds old.
        self.create_subscription(
            String,
            CANDIDATE_TOPIC,
            self._on_candidate,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        # Depth 5, not 1: unlike a candidate — a snapshot of now, where only the newest is worth
        # judging — every measurement carries the stamp it was measured at and is carried to the
        # update that takes it, so a short queue is a few tenths of a second of the camera's
        # history rather than a backlog of stale opinions.
        self.create_subscription(
            String,
            MEASUREMENT_TOPIC,
            self._on_measurement,
            QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE),
        )
        self._fit_pub = self.create_publisher(Float32, "localization_fit", 5)
        # Every source's word on each update, as JSON (Localizer.sources_report): the demo's
        # view of the lidar and the camera agreeing, disagreeing, or one of them gone.
        self._sources_pub = self.create_publisher(String, "/localization/sources", 5)
        # The fit at the pose actually published (the blend), beside the tracker's own fit at
        # the matched pose: the two part ways while a carry is being absorbed.
        self._published_fit_pub = self.create_publisher(Float32, "localization_fit_published", 5)
        # The operator's own word (Foxglove's "set pose", ros/goto.sh seed): the map has twins —
        # 2026-09-13 the whole-map search seeded the cart 6 m from its base at fit 0.77 and the
        # watchdog's true candidate (0.72 vs 0.69) could not beat it by the margin — and nothing
        # else lets a person who can see the cart say where it is. A message this node sent
        # itself (the AMCL-era echo above) re-seeds the same pose and changes nothing.
        self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self._on_operator_seed, 5
        )
        self._tf_pub = TransformBroadcaster(self)
        self._tracker_pub = self.create_publisher(PoseWithCovarianceStamped, "/tracker_pose", 5)
        self._slip_pub = self.create_publisher(Bool, "/slip", 5)  # wheels move, the world does not
        self.create_timer(30.0, self._report_tracking)
        self.create_timer(2.0, self._remember_pose)
        self.create_timer(0.2, self._apply_pending_seed)  # the worker's fix, applied here
        self.create_timer(0.05, self._send_map_odom)  # the frame stays alive, scans or not
        self.create_subscription(
            String, "/pepin/note", lambda m: self.get_logger().info(f"note: {m.data}"), 5
        )
        # Eyes for the operator: AMCL's particles as arrows and its pose history as a line.
        # nav2_msgs/ParticleCloud is unknown to Foxglove; PoseArray and Path are not.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._particles_pub = self.create_publisher(PoseArray, "/particle_poses", 1)
        self._trail_pub = self.create_publisher(Path, "/amcl_path", latched)
        self._trail = Path()
        self._trail.header.frame_id = "map"
        self.create_subscription(
            ParticleCloud, "/particle_cloud", self._on_particles, qos_profile_sensor_data
        )
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self._on_amcl_pose, 10)
        self.create_service(Trigger, "relocalize", self._on_relocalize)
        self.create_service(Trigger, "where_am_i", self._on_where)
        # Static transforms only (the laser mount). /tf itself is not read here: this node owns
        # map -> odom and keeps odom -> base_link in its own history, and 40 tf messages a second
        # deserialised in Python cost a quarter of an A53 core for nothing.
        self._tf = TfLookup(self, buffer=Buffer())  # no listener: /tf_static is read below
        self.create_subscription(
            TFMessage,
            "/tf_static",
            self._on_tf_static,
            QoSProfile(
                depth=100,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self.create_timer(self._check_period_s, self._check)
        # Declared after every other parameter: rclpy runs the switches' callback on
        # declarations too, and it refuses everything that is not a flag.
        self._switches = Switches(self, FLAGS, on_change=self._on_switch)
        self._registry.enable(self._switches["sources"])
        self._resize_berth()
        for gate in (self._candidates, self._measurements):  # a launch override reaches them too
            for name in gate.switches:
                gate.switch(name, self._switches[name])
        self.get_logger().info("relocalizer up: watching the scan-to-map fit")

    # -- inputs -------------------------------------------------------------

    def _now_s(self) -> float:
        """The node's clock in seconds: the same clock the scans are stamped with."""
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _on_switch(self, name: str, _old: Any, new: Any) -> None:
        """A flag changed (``ros2 param set``): a berth flag resizes the rings around new
        objects at once; every other one is written through to whichever object owns it — the
        Localizer (``sources`` reaches the roster the feed shares with it, so the anchor moves
        with the flag) or the candidate gate — so the next scan, and the next candidate, are
        handled with it. Each object names its own flags (``switches``), so a set of one
        object's flag is never refused by the other."""
        if name in BERTH_FLAGS:
            self._resize_berth()  # the switches already hold the new value
            return
        for target in (self._localizer, self._candidates, self._measurements):
            if target is not None and name in target.switches:
                target.switch(name, new)

    def _resize_berth(self) -> None:
        """Rebuild the berth around new objects from the planner in charge and the berth flags,
        and log the two distances it comes down to."""
        self._berth = berth_for(
            self._planner_id,
            reach_m=float(self._switches["toe_reach"]),
            near_rings=bool(self._switches["near_rings"]),
        )
        self.get_logger().info(
            f"planner {self._planner_id}: dynamic rings {self._berth.ring_m:.2f} m, none within"
            f" {self._berth.near_m:.2f} m, marks trimmed within {self._berth.trim_m:.2f} m"
        )

    def _on_map(self, msg: OccupancyGridMsg) -> None:
        self._grid = grid_from_msg(msg)
        with self._episode:  # a candidate found on the old map is evidence about nothing here
            self._watch = LostWatch(**self._watch_args)  # type: ignore[arg-type]
            self._pending_seed = None
            self._candidates.forget()
            self._measurements.forget()  # nor is a pose measured against the old one
        self._matcher = CorrelativeMatcher(self._grid)
        self._static_mask = StaticMask(self._grid)
        # lost_after huge: update() must never run a whole-map search in the executor thread on
        # this board (10-20 s); the 1 Hz watcher below does that in a worker and re-seeds.
        # Tracking window sized for this board: 7x7 positions x 13 headings x 120 beams is about
        # 60 ms per scan on an A53 (the laptop default, 9x9x49x200, took 360 ms: 2 Hz).
        self._map_id = map_id(msg)
        # The tracker's own flags only: the gate's (accept_candidates, candidate_streak) are
        # this node's and the berth's (BERTH_FLAGS) size the rings around new objects, so
        # neither belongs to a Localizer — each object names what it owns.
        flags = {n: v for n, v in self._switches.flags.as_dict().items() if n in TRACKER_SWITCHES}
        self._registry.enable(flags.pop("sources"))  # the roster is the feed's and the tracker's
        self._localizer = Localizer(
            self._grid,
            self._last_known_pose(),  # a restart is not a trip back to the base
            window=SearchWindow(xy_m=0.09, xy_step_m=0.03, theta_deg=9.0, theta_step_deg=1.5),
            # Lost (five weak scans): a wider, coarser local search every scan re-locks after a
            # slip; the whole map stays the worker's job (global_retry False).
            recovery=SearchWindow(xy_m=0.25, xy_step_m=0.05, theta_deg=20.0, theta_step_deg=2.0),
            max_points=120,
            # Half of each match's residual per scan: one match carries about a degree of
            # noise, and at full gain that noise reaches the wheels through map -> odom ten
            # times a second (the robot weaved, 2026-09-07). Only the residual is damped, so
            # the motion itself never lags; a residual too big to be noise is taken whole.
            correction_gain=0.5,
            recovery_min_inliers=0.5,  # this flat's true pose scores 0.5-0.65 on its maps
            lost_after=3,
            global_retry=False,
            interpolate=self._subcell_refine,
            sources=self._registry,
            **flags,
        )
        self.get_logger().info(f"map received: {msg.info.width}x{msg.info.height} cells")
        self._tracker_initialised = False  # a new map: find ourselves on it again
        self._motion.reset()
        self._last_match_stamp_s = None  # the next match is the first one on this map

    def _on_scan(self, msg: LaserScan) -> None:
        """The lidar's revolution, moved into base_link by the mount looked up once."""
        if self._laser_tf is None and not self._lookup_laser(msg.header.frame_id):
            return
        assert self._laser_tf is not None
        self._offer(LIDAR, msg, self._laser_tf)

    def _offer(self, name: str, msg: LaserScan, mount: tuple[float, float, float, bool]) -> None:
        """Any source's scan into the feed, and a try at matching whatever the feed releases."""
        self._scan_id += 1  # which scan a search was computed on: a second opinion needs a new one
        scan = timed_scan_from_ros(
            Time.from_msg(msg.header.stamp).nanoseconds * 1e-9,
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            msg.range_max,
            msg.scan_time,
            mount,
            self._scan_id,
        )
        self._feed.offer(name, scan)
        self._track_pending()

    def _on_odom(self, msg: Odometry) -> None:
        """Every fused odometry sample feeds the history; a scan waiting for it gets matched."""
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self._history.add(
            Time.from_msg(msg.header.stamp).nanoseconds * 1e-9, Pose2D(p.x, p.y, yaw_of(q))
        )
        # The gyro's word on whether the cart turns, taken from the filter that already fuses it:
        # subscribing to /imu/data_raw here would cost a slice of a core for a number we have.
        self._odom_wz = float(msg.twist.twist.angular.z)
        self._track_pending()

    def _on_candidate(self, msg: String) -> None:
        """The laptop watchdog's whole-map candidate (one JSON message, pepin.watchdog): carried
        to this moment, then judged by the gate against this tracker's own pose and fit, on this
        tracker's own map.

        A streak of disagreements about one place becomes a pending seed, which the 0.2 s timer
        applies through :meth:`_seed` — the same door the board's own search uses, so a
        candidate can do nothing a search could not. No re-seed while a goal runs: a teleport
        mid-drive is worse than a poor fit, and the drive's own watches stop it soon enough.
        Nothing is decided here; the verdicts reach the operator in the report line and in the
        ``candidates`` block of /localization/sources.
        """
        try:
            candidate = GlobalCandidate.from_json(msg.data)
        except (KeyError, TypeError, ValueError) as exc:
            self._candidates.malformed(str(exc))
            return
        answer = self._candidates.observe(
            candidate,
            self._tracked_pose(),
            self.fit,
            map_id=self._map_id,
            allow=not self._navigating,
            odometry=self._history,  # the trail the candidate is carried to this moment along
        )
        with self._episode:  # the search worker writes _pending_seed from its own thread
            self._pending_seed = answer.pending(self._map_id) or self._pending_seed

    def _on_measurement(self, msg: String) -> None:
        """The laptop's pose measurement off a camera scan (one JSON message,
        pepin.measurements): it waits at the gate for the next update, which carries it from the
        moment of its own scan to that update's moment and fuses it.

        Nothing is decided here — a measurement on another map is refused by the gate, a message
        that does not parse is counted — and the try at an update is the same one every input
        makes: with the lidar alive the measurement rides its next revolution, and with the lidar
        stale or gone it drives an update of its own (:meth:`_track_on_measurements`).
        """
        try:
            remote = RemoteMeasurement.from_json(msg.data)
        except (KeyError, TypeError, ValueError) as exc:
            self._measurements.malformed(str(exc))
            return
        self._measurements.offer(remote, self._map_id)
        self._track_pending()

    def _track_on_measurements(self, now: float) -> None:
        """An update driven by the camera's measurements alone, at the stamp of the newest one.

        This is what a dead lidar leaves: no scan waits at the feed and nothing is fresh, so the
        feed has no anchor and the only word about where the cart is comes over the link. What
        may drive such an update, and when, is the gate's decision
        (:meth:`pepin.measurements.MeasurementGate.drive`); here it is carried out. The rest
        filter that spares the matcher while the cart stands does not apply — there is nothing
        to match, the match was made on the laptop — and the rest LOCK still does its work
        inside the tracker. The tracker takes its first fix this way too, from its saved pose,
        with no whole-map search: a fan cannot find the cart, and a pose measured off one cannot
        either.
        """
        loc = self._localizer
        plan = self._measurements.drive(self._feed.anchor(now), self._history)
        if loc is None or plan is None:
            return
        if not self._tracker_initialised:
            self._tracker_initialised = True
            self.get_logger().warning(
                f"tracker starts at ({loc.pose.x:+.2f}, {loc.pose.y:+.2f}, "
                f"{math.degrees(loc.pose.theta):+.0f} deg) on the camera's measurements without "
                "a first search: a fan cannot find the cart"
            )
        previous_stamp = self._last_match_stamp_s
        self._last_match_stamp_s = plan.stamp
        dt_s = (
            plan.stamp - previous_stamp
            if previous_stamp is not None and plan.stamp > previous_stamp
            else None
        )
        self._last_scan_age_s = now - plan.stamp
        pose = loc.update_from(
            plan.odom,
            [],
            measurements=[plan.measurement],
            at_rest=standing_still(self._history, plan.stamp, self._odom_wz),
            dt_s=dt_s,
        )
        self._publish_update(loc, pose, plan.odom, plan.stamp, now)

    def _publish_update(
        self, loc: Localizer, pose: Pose2D, odom: Pose2D, stamp: float, now: float
    ) -> None:
        """What every update publishes, however it was driven: map -> odom from the pose at this
        stamp, the tracked pose with the covariance its fit buys, the fit at the pose actually
        published, and every source's word as JSON — the scans', the watchdog's and the
        camera's."""
        self._last_map_odom = map_to_odom(pose, odom)
        self._send_map_odom()
        self._publish_tracker_pose(
            pose, loc.confidence, Time(nanoseconds=int(stamp * 1e9)).to_msg()
        )
        self._published_fit_pub.publish(Float32(data=float(loc.published_fit)))
        report = loc.sources_report(now)
        report["candidates"] = self._candidates.status()  # the laptop's word, beside the scans'
        report["measurements"] = self._measurements.status()
        self._sources_pub.publish(String(data=json.dumps(report)))

    def _track_pending(self) -> None:
        """Match the anchor's scan at the feed once the odometry history covers its whole
        revolution, with the other sources' scans — and the camera's measurements from the
        laptop — carried to its moment riding along.

        Called on every input: a scan usually arrives before the odometry of its last beams and
        is released by the odometry sample that completes it, 30-70 ms later. Which source's
        scan is released is the feed's call — in practice the lidar's, the only scan source this
        board still matches, and when nothing of its is fresh or waiting the camera's
        measurements drive the update instead (:meth:`_track_on_measurements`). The scan is
        deskewed
        with the history (every beam moved to where the robot was at the stamp) and the pose
        the localizer predicts from is the interpolated pose at that same stamp, so the
        residual the matcher reports is odometry error and nothing else.
        """
        loc = self._localizer
        if loc is None or not self._track:  # ``track`` false: AMCL owns the pose, we only watch
            return
        now = self._now_s()
        taken = self._feed.take(self._history, now)
        if taken is None:
            self._track_on_measurements(now)  # a dead lidar: the camera's word drives instead
            return
        anchor, scan = taken
        if not self._tracker_initialised:
            if self._registry.source(anchor).partial:
                # A fan cannot find the cart (two frames of the same view agree on the same
                # look-alike): no first search on it. The tracker follows the fan from its
                # saved pose and the watch judges the pose once a full revolution drives.
                self._tracker_initialised = True
                self.get_logger().warning(
                    f"tracker starts at ({loc.pose.x:+.2f}, {loc.pose.y:+.2f}, "
                    f"{math.degrees(loc.pose.theta):+.0f} deg) on the {anchor} fan without a "
                    "first search: a fan cannot find the cart"
                )
            else:
                if not self._tracker_initialising and not self._searching:
                    self._tracker_initialising = True
                    threading.Thread(
                        target=self._initialise_tracker, args=(scan.points,), daemon=True
                    ).start()
                return
        mono = time.monotonic()
        if self._pacer.skip(mono, self._searching):
            return
        odom = self._history.at(scan.stamp)
        if odom is None:  # cannot happen past the gate; a guard, not a fallback
            return
        if not self._motion.due(odom, scan.stamp):
            self._rested += 1  # standing still: the last match still holds, and so does the pose
            self._publish_dynamic(scan.points, loc.pose, scan.stamp)
            return
        # Seconds of robot time since the previous match, from the scan stamps (never the wall
        # clock): the rest lock's gain is a time constant, and this cadence is what it needs.
        previous_stamp = self._last_match_stamp_s
        self._last_match_stamp_s = scan.stamp
        dt_s = (
            scan.stamp - previous_stamp
            if previous_stamp is not None and scan.stamp > previous_stamp
            else None
        )
        self._last_scan_age_s = now - scan.stamp
        points, whole = deskewed(scan, self._history)
        self._deskew_failed += not whole
        # Slip: the wheels claim a step but the scan is the same picture as a tenth of a second ago.
        # Then the wheel step is a lie; the pose is corrected from where it was, and Nav2's progress
        # checker (map frame) sees the truth: no progress -> a recovery instead of a 60 s wheelspin.
        slip = self._slip.observe(scan.ranges, odom)
        if self._slip.streak == SLIP_SAID_AFTER:
            self.get_logger().warning(
                "wheels turning, world standing still: slip, wheel step ignored"
            )
        self._slip_pub.publish(Bool(data=slip))
        # What the sensors say, and nothing else: the wheels' and the gyro's word on rest, the
        # static map's mask. Whether either is used is the Localizer's switch (live).
        at_rest = standing_still(self._history, scan.stamp, self._odom_wz)
        # The anchor's scan and every other enabled source's waiting scan, each moved into the
        # base frame of the anchor's stamp through the odometry between the two (the feed).
        scans = [
            ScanObservation(anchor, points, scan.stamp),
            *self._feed.gather(anchor, scan.stamp, self._history),
        ]
        t0 = time.perf_counter()
        pose = loc.update_from(
            odom,
            scans,
            measurements=self._measurements.take(scan.stamp, self._history),
            trust_odometry=not slip,
            at_rest=at_rest,
            dt_s=dt_s,
            mask=self._static_mask,
        )
        self._pacer.matched(mono, time.perf_counter() - t0)
        self._publish_update(loc, pose, odom, scan.stamp, now)
        self._publish_dynamic(points, pose, scan.stamp)  # the pose of this scan's own moment

    def _last_known_pose(self) -> Pose2D:
        """The pose saved by the previous run if it is recent and was good, else the map origin."""
        try:
            with open(LAST_POSE_FILE) as f:
                saved = json.load(f)
            age = time.time() - float(saved["time"])
            same_map = saved.get("map") == self._map_id  # a pose means nothing on another map
            if same_map and age <= LAST_POSE_MAX_AGE_S and float(saved.get("fit", 0.0)) >= LOST_FIT:
                pose = Pose2D(float(saved["x"]), float(saved["y"]), float(saved["theta"]))
                self.get_logger().info(
                    f"starting from the last known pose ({pose.x:+.2f}, {pose.y:+.2f}, "
                    f"{math.degrees(pose.theta):+.0f} deg), saved {age:.0f} s ago"
                )
                return pose
        except (OSError, KeyError, ValueError, TypeError):
            pass
        return Pose2D()

    def _remember_pose(self) -> None:
        """Every 2 s: write the tracked pose and its fit, so the next start knows where we are."""
        loc = self._localizer
        if loc is None or not self._tracker_initialised or self.fit < LOST_FIT:
            return
        record = {
            "x": loc.pose.x,
            "y": loc.pose.y,
            "theta": loc.pose.theta,
            "fit": self.fit,
            "time": time.time(),
            "map": self._map_id,
        }
        try:
            with open(LAST_POSE_FILE + ".tmp", "w") as f:
                json.dump(record, f)
            os.replace(LAST_POSE_FILE + ".tmp", LAST_POSE_FILE)
        except OSError:
            pass

    def _initialise_tracker(self, points: Any) -> None:
        """Worker thread: the first fix is a candidate like any other, confirmed by a second search.

        It used to seed whatever one whole-map search returned, unconditionally — even a fix the
        localizer had rejected — and that single seed could discard a candidate the watch was
        holding. Now the tracker starts from its saved pose, a whole-map search proposes, and the
        1 Hz check asks again on a fresh scan; the pose moves once two searches agree.
        """
        loc = self._localizer
        assert loc is not None
        scan = self._scan_id
        try:
            found, confidence = loc.global_search(points, prior=loc.pose)
            yaw = math.degrees(found.pose.theta)
            where = f"({found.pose.x:+.2f}, {found.pose.y:+.2f}, {yaw:+.0f} deg)"
            self.get_logger().info(
                f"tracker starts at ({loc.pose.x:+.2f}, {loc.pose.y:+.2f}, "
                f"{math.degrees(loc.pose.theta):+.0f} deg); first search proposes {where} "
                f"at fit {confidence:.2f}"
            )
            with self._episode:
                self._watch.answer(found.pose, confidence, 0.0, scan, time.monotonic())
            self._tracker_initialised = True
        finally:
            self._tracker_initialising = False

    def _occluded(self, points: Any, pose: Pose2D) -> bool:
        """A person beside the cart rather than a lost cart (:func:`pepin.dynamic.occluded`),
        judged on the newest scan ``points`` at ``pose``."""
        return occluded(points, pose, self._static_mask, self._matcher, self._watch.lost_fit)

    def _on_planner(self, msg: String) -> None:
        """The berth around new objects depends on who plans: a footprint planner brings the
        hull itself, a point planner needs it in the ring (pepin.dynamic.berth_for)."""
        self._planner_id = msg.data
        self._resize_berth()

    def _publish_dynamic(self, points: Any, pose: Pose2D, stamp_s: float) -> None:
        """Lethal rings around the returns the map does not explain, in the map frame.

        Only from a pose the tracker trusts: with the pose off by more than the static mask's
        margin the whole scan is "news", and the rings would paint the costmaps lethal exactly
        when localisation is already in trouble (review, 2026-09-09).
        """
        if self._static_mask is None or self.fit < DRIVE_FIT:
            return
        marks = dynamic_marks(points, pose, self._static_mask, self._berth)
        self._dynamic_count += len(marks)
        xyz = np.column_stack([marks, np.zeros(len(marks))])  # the marks lie on the floor
        self._dynamic_pub.publish(cloud_from_points(xyz, None, stamp_from_seconds(stamp_s), "map"))

    def _send_map_odom(self) -> None:
        """Broadcast the current map -> odom, 20 times a second and after every match, dated
        ``tf_future_s`` (0.1 s) ahead like AMCL does.

        The contract for every consumer: compose the LATEST map -> odom with odom -> base_link
        at your own stamp. map -> odom is a slowly moving correction, not a trajectory — its
        stamp says "valid from here", never "measured here" — while odom -> base_link carries
        the motion and is exact at its stamp. A consumer that asks tf2 for map -> base_link at a
        stamp gets exactly that composition; the short future date keeps "now" inside the
        buffer so nobody extrapolates, and keeps a fresh correction from waiting half a second
        behind a stale future-dated one.
        """
        x, y, yaw = self._last_map_odom
        future = self.get_clock().now() + Duration(seconds=self._tf_future_s)
        self._tf_pub.sendTransform(
            transform_from_rpy("map", "odom", (x, y, 0.0), (0.0, 0.0, yaw), future.to_msg())
        )

    def _publish_tracker_pose(self, pose: Pose2D, confidence: float, stamp: Any) -> None:
        """The tracked pose for the operator's view and the trail; sigma grows as the fit drops
        (:func:`pepin.fusion.sigma_from_fit` — the same numbers a fusion weighs this pose by,
        so what the operator sees and what the watchdog's candidate is weighed against agree)."""
        sigma_xy, sigma_yaw = sigma_from_fit(confidence)
        msg = pose_with_covariance(pose.x, pose.y, pose.theta, sigma_xy, sigma_yaw, stamp, "map")
        self._tracker_pub.publish(msg)
        self._append_trail(msg)

    def _report_tracking(self) -> None:
        """Every 30 s: where every released scan went (per source, with the rides), what the
        tracker did with the matched ones (its switches, the rest lock's cadence and gain,
        carries, lost/weak, the fit at the match and at the published pose, the largest
        map -> odom step), who drives and every source's health, what the laptop's watchdog
        proposed and what came of it, and the cost."""
        loc = self._localizer
        if loc is None:
            return
        feed, pacer, track = self._feed.report(), self._pacer.report(), loc.report()
        self.get_logger().info(
            f"tracker: {feed.summary()}, rested {self._rested}, {pacer.summary()}, deskew "
            f"failed {self._deskew_failed}; {loc.settings()}; {track.summary()}; "
            f"sources: {self._feed.status(self._now_s())}; "
            f"watch {'fit' if self._watch_on else 'off: no full-turn source, fit'} "
            f"{self.fit:.2f}, dynamic marks {self._dynamic_count} "
            f"(rings {self._berth.ring_m:.2f} m from {self._berth.near_m:.2f} m out, trimmed "
            f"within {self._berth.trim_m:.2f} m), "
            f"scan age at match {self._last_scan_age_s * 1000:.0f} ms; "
            f"{self._candidates.report()}; {self._measurements.report()}; "
            f"flags: {self._switches.state()}"
        )
        if feed.expired:
            self.get_logger().warning(
                f"odometry ran late: {feed.expired} scans were never covered by {self._odom_topic} "
                "and matched nothing"
            )
        self._rested = self._deskew_failed = self._dynamic_count = 0

    def _lookup_laser(self, frame: str) -> bool:
        """The static base_link <- laser transform, as x, y, yaw and whether roll is pi."""
        t = self._tf.transform("base_link", frame)
        if t is None:  # not yet available: try on the next scan
            return False
        x, _y, yaw, mirrored = self._laser_tf = planar_mount(t)
        self.get_logger().info(
            f"laser mount: x {x:.3f} yaw {math.degrees(yaw):.1f} deg"
            f"{' upside down' if mirrored else ''}"
        )
        self.create_subscription(
            GoalStatusArray, "/navigate_to_pose/_action/status", self._on_nav_status, 10
        )
        return True

    def _on_particles(self, msg: ParticleCloud) -> None:
        """Every AMCL particle set, thinned to 300 arrows, as a PoseArray for the 3D view."""
        step = max(1, len(msg.particles) // 300)
        out = PoseArray()
        out.header = msg.header
        out.poses = [particle.pose for particle in msg.particles[::step]]
        self._particles_pub.publish(out)

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        """AMCL's pose feeds the trail only when AMCL, not this node, owns the localisation."""
        if not self._track:
            self._append_trail(msg)

    def _append_trail(self, msg: PoseWithCovarianceStamped) -> None:
        """Append a pose to the trail (last 600 poses) and republish it, latched."""
        stamped = PoseStamped()
        stamped.header = msg.header
        stamped.pose = msg.pose.pose
        self._trail.poses.append(stamped)
        del self._trail.poses[:-600]
        self._trail.header.stamp = msg.header.stamp
        self._trail_pub.publish(self._trail)

    def _on_nav_status(self, msg: GoalStatusArray) -> None:
        """Remember whether Nav2 is executing a goal: no automatic re-seeding while it drives."""
        self._navigating = any(
            s.status in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING)
            for s in msg.status_list
        )

    def _moving(self) -> bool:
        """Did base_link move in the odom frame since the previous check (1 cm or 1 degree)?"""
        return self._motion_edge.moved(self._history.newest)

    def _on_tf_static(self, msg: TFMessage) -> None:
        """The static frames (latched): the laser mount is read from them once."""
        for transform in msg.transforms:
            self._tf.buffer.set_transform_static(transform, "static")

    def _tracked_pose(self) -> Pose2D | None:
        """map -> base_link now: this node's map -> odom over the newest odometry pose."""
        odom = self._history.newest
        if odom is None:
            return None
        x, y, yaw = self._last_map_odom
        c, s = math.cos(yaw), math.sin(yaw)
        return Pose2D(
            x + c * odom.x - s * odom.y, y + s * odom.x + c * odom.y, wrap_angle(yaw + odom.theta)
        )

    # -- the watch ------------------------------------------------------------

    def _check(self) -> None:
        """Once a second: score the fit on the newest picture of the source that drives the
        tracker; the watch decides whether to search the whole map — on a full revolution
        only: while anything narrower drives, the fit is published and the watch is off.

        With no scan of our own at all — the camera's measurements driving the tracker — there
        is nothing here to score, and the fit published is the one the laptop measured and the
        tracker kept. It is the honest number: the goal server and the watchdog both read this
        topic, and a tracker running on the camera alone must still be able to say how well it
        sits on the map.
        """
        now = self._now_s()
        full = self._feed.full_picture(now)
        picture = full if full is not None else self._feed.picture(now)
        loc = self._localizer
        pose = self._tracked_pose()
        if self._matcher is None or loc is None or pose is None:
            return
        # Sampled exactly once: _moving() is an edge detector on odometry, and a second call in
        # the same tick compared a reading with itself and told the watch the robot stood still
        # at every speed — the "never re-seed a moving robot" gate was dead (review, 2026-09-09).
        moving = self._moving()
        # The tracker's OWN confidence — the inlier fraction of its last match, scan and pose
        # from the same instant — not a re-evaluation of the TF pose against the latest scan.
        # Those two are 100-200 ms apart, and at 0.5 rad/s that is 3-6 degrees: the re-evaluated
        # fit collapsed to 0.19-0.31 during every pivot while the tracker itself sat at 0.90-0.95
        # (run 0070, second by second), and a blind-drive rule built on the false number stopped
        # healthy drives mid-turn and moved the belief by 0.4 m to "recover" from nothing.
        self.fit = (
            float(loc.confidence)
            if moving or picture is None
            else self._matcher.inlier_fraction(pose, picture.points)
        )
        self._fit_pub.publish(Float32(data=float(self._watch.reported_fit(self.fit))))
        # A fan drives, or nothing here does: such a fit cannot say "lost" (LOST_FIT was tuned on
        # full revolutions) and a search on it would re-seed the tracker on a look-alike the twin
        # check cannot see — the watch is off, and the report line says so.
        self._watch_on = full is not None
        if full is None or self._searching or not self._tracker_initialised:
            return
        # The watch gets the tracker's OWN fit. reported_fit() is for the outside world: capped
        # at 0.35 while a candidate pends, and fed back here it kept a twin candidate alive at
        # fit 0.76 and ran a whole-map search every second for half an hour (18:00 today).
        occluded = self._occluded(full.points, pose)
        # Under the episode lock, like every other _watch call and every claim of _searching:
        # the worker answers a search on its own thread and the two share this state.
        with self._episode:
            search = (
                self._watch.observe(
                    self.fit,
                    moving=moving,
                    navigating=self._navigating,
                    now=time.monotonic(),
                    occluded=occluded,
                )
                and not self._searching
            )
            if search:
                self._searching = True
        if search:
            self.get_logger().warning(f"fit {self.fit:.2f}: searching the whole map")
            threading.Thread(target=self._search_and_seed, args=(full,), daemon=True).start()
        elif occluded and self.fit < self._watch.lost_fit:
            self.get_logger().info(
                f"fit {self.fit:.2f} but the scan is mostly things the map does not know: "
                "occluded, not lost"
            )

    def _dump_failure(
        self, points: Any, current: Pose2D | None, best: Pose2D | None, confidence: float
    ) -> None:
        """Write the scan and the verdict of a failed search as JSON (a few tens of kB)."""
        try:
            os.makedirs(DUMP_DIR, exist_ok=True)
            path = f"{DUMP_DIR}/reloc_fail_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(path, "w") as f:
                json.dump(
                    {
                        "points": np.round(points, 3).tolist(),
                        "current": None
                        if current is None
                        else [current.x, current.y, current.theta],
                        "best": None if best is None else [best.x, best.y, best.theta],
                        "confidence": confidence,
                    },
                    f,
                )
            self.get_logger().info(f"search dumped to {path}")
        except OSError as exc:
            self.get_logger().warning(f"could not dump the failed search: {exc}")

    def _apply_pending_seed(self) -> None:
        """Adopt what the search worker found, unless the map changed while it was searching.

        The worker computes; the executor applies. A search runs for seconds, and a map swap in
        the middle used to hand the new localizer a pose measured on the old map.
        """
        with self._episode:
            pending, self._pending_seed = self._pending_seed, None
        if pending is None:
            return
        map_id, pose, confidence = pending
        if map_id != self._map_id:
            self.get_logger().warning("a search finished on the old map: its fix is dropped")
            return
        self._seed(pose, confidence)

    def _search_and_seed(self, picture: TimedScan) -> None:
        """Worker thread: the search must not block the executor (scan and service callbacks)."""
        try:
            self._relocalize(picture)
        finally:
            with self._episode:
                self._searching = False

    def _relocalize(self, picture: TimedScan) -> str:
        """One whole-map search on ``picture`` — the full revolution the caller decided on
        (:meth:`~pepin.sources.SourceFeed.full_picture`; never a fan); the watch decides what
        its answer is worth."""
        assert self._localizer is not None and self._matcher is not None
        # The scan's id and the map it was taken on, captured together at the start: a map
        # swap during the seconds of a search must not stamp the old scan's fix as the new map's.
        map_id = self._map_id
        points, scan = picture.points, picture.scan_id
        current = self._tracked_pose()
        current_fit = self._matcher.inlier_fraction(current, points) if current else 0.0
        started = time.monotonic()
        # Always the whole map: a "nearby first" shortcut accepted a 0.66 impostor two metres from
        # a carried robot (2026-09-06 18:14) and the whole-map stage never ran. The previous belief
        # only breaks ties between look-alikes.
        credible = current is not None and current_fit >= 0.4  # a stale belief must not break ties
        found, confidence = self._localizer.global_search(
            points,
            theta_step_deg=10.0,
            thin_to=90,
            prior=current if credible else None,
            refuse_twins=credible,
        )
        took = time.monotonic() - started
        answer = found.pose if found is not None else None
        with self._episode:
            verdict = self._watch.answer(answer, confidence, current_fit, scan, time.monotonic())
            if verdict.verdict is Verdict.APPLY and verdict.pose is not None:
                self._pending_seed = (map_id, verdict.pose, verdict.confidence)
        where = (
            "nothing"
            if answer is None
            else f"({answer.x:+.2f}, {answer.y:+.2f}, {math.degrees(answer.theta):+.0f} deg)"
        )
        fits = f"fit {confidence:.2f} vs now {current_fit:.2f}"
        text = {
            Verdict.NOTHING: f"no better pose ({took:.1f} s): {fits}",
            Verdict.CANDIDATE: f"candidate ({took:.1f} s): {where} {fits}; asking once more",
            Verdict.REPLAY: "same scan as the candidate: no second opinion yet",
            Verdict.HOLD: f"second search disagrees ({took:.1f} s): candidate {where}",
            Verdict.APPLY: f"relocalised, agreed by a second search ({took:.1f} s): to {where}",
            Verdict.GIVEN_UP: (
                f"searches keep disagreeing: the scan fits two places alike, last {where}; "
                f"tracking at fit {current_fit:.2f} until the robot moves"
            ),
        }[verdict.verdict]
        if verdict.verdict is Verdict.NOTHING:
            self._dump_failure(points, current, answer, confidence)
        # Two call sites on purpose: rclpy keys a logger call's severity on its source line and
        # raises "Logger severity cannot be changed between calls" when one line logs both.
        if verdict.verdict in (Verdict.CANDIDATE, Verdict.APPLY):
            self.get_logger().info(text)
        else:
            self.get_logger().warning(text)
        return text

    def _on_operator_seed(self, msg: PoseWithCovarianceStamped) -> None:
        """A pose handed in on /initialpose is adopted as the truth at full confidence."""
        p = msg.pose.pose
        pose = Pose2D(float(p.position.x), float(p.position.y), yaw_of(p.orientation))
        self.get_logger().info(
            f"seeded by the operator at ({pose.x:.2f}, {pose.y:.2f},"
            f" {math.degrees(pose.theta):.0f} deg)"
        )
        self._seed(pose, 1.0)

    def _seed(self, pose: Pose2D, confidence: float) -> None:
        """Adopt ``pose`` with the confidence it was measured at: the localizer, the fit, map ->
        odom and the watch follow at once.

        Nothing goes back out on /initialpose. This node listens there for the operator, and
        the old "tell AMCL" publication fed the node its own seed: every adoption came back as
        an operator seed, 20 times a second, and two hand seeds alternated for an hour with
        map -> odom flipping 16 cm at 38 Hz and the rest lock reset on every turn (2026-09-13).
        """
        if self._localizer is not None:
            self._localizer.adopt(pose, confidence)
        self.fit = confidence
        # map->odom follows the seed NOW: the next matched scan may be a second away at rest, and
        # for that second /localization_fit would say "found" while the transform still placed the
        # cart where it stood before a carry (a review probe, 2026-09-11).
        newest = self._history.newest
        self._last_map_odom = (
            map_to_odom(pose, newest) if newest is not None else self._last_map_odom
        )
        self._send_map_odom()
        self._motion.reset()
        with self._episode:  # the worker may be inside _watch.answer() right now
            self._watch.seeded(time.monotonic())

    # -- services ---------------------------------------------------------

    def _on_relocalize(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        """Start a whole-map episode now (or report the one running); the answer is on
        /localization_fit within a few seconds. The search runs in the worker, never here: a
        loop on the executor thread froze the scan, so its "second opinion" was the first search
        replayed to the millimetre and every candidate was rubber-stamped (review, 2026-09-09)."""
        now = self._now_s()
        picture = self._feed.full_picture(now)
        if self._localizer is None or picture is None:
            res.success = False
            res.message = (
                "no map or no scan yet"
                if self._localizer is None or self._feed.picture(now) is None
                else "no full-turn scan to search with: a fan cannot find the cart"
            )
            return res
        with self._episode:
            if self._searching:
                res.success, res.message = True, "a search is already running"
                return res
            self._searching = True
        threading.Thread(target=self._search_and_seed, args=(picture,), daemon=True).start()
        res.success = True
        res.message = (
            "searching the whole map; a fix needs two searches that agree — watch /localization_fit"
        )
        return res

    def _on_where(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        pose = self._tracked_pose()
        if pose is None:
            res.success, res.message = False, "no map->base_link transform yet"
            return res
        res.success = True
        res.message = (
            f"x {pose.x:+.2f} m, y {pose.y:+.2f} m, yaw {math.degrees(pose.theta):+.0f} deg;"
            f" scan-to-map fit {self._watch.reported_fit(self.fit):.2f}"
            f" (good > {DRIVE_FIT}, lost < {self._watch.lost_fit})"
            + ("" if self._watch.confirmed else "; UNCONFIRMED: waiting for a second search")
        )
        return res


def main(args: list[str] | None = None) -> None:
    """Entry point."""
    spin_main(Relocalizer, args)


if __name__ == "__main__":
    main()
