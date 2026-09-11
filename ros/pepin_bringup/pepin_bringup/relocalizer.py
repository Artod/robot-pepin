"""Global relocalisation on top of AMCL: the robot always knows where it is, no human needed.

AMCL tracks well but only searches where its particles already are: carry the
cart across the room, push it, lift it onto the carpet, and it keeps believing
the old pose. This node closes that gap with the correlative whole-map search
from :mod:`pepin.localization` (a fraction of a second on the pooled grid, with
the twin check). Every second it scores how well the current scan lies on the
map at AMCL's pose; when the fit stays poor it searches the whole map and, if a
clearly better pose exists, re-seeds AMCL through ``/initialpose``. The same
search answers the ``/relocalize`` service on demand, and ``/where_am_i``
reports pose and fit as text.

Frames: the scan is transformed into ``base_link`` with the static laser
transform looked up once; poses are in ``map``.

Sources: the tracker matches whatever scan-shaped evidence the ``sources`` flag lets in —
``/scan``, the camera's ``/depth_scan`` and ``/contact_scan`` (LaserScans already in
``base_link``) — through one path (:class:`pepin.sources.SourceFeed`): the lidar's revolution
drives every update while it is fresh and the camera's fans ride along, carried to its moment;
when the lidar is stale or absent the fans drive the updates themselves, so a dead lidar hands
the tracker to the camera without a restart. ``/localization/sources`` carries every source's
word on each update as JSON for the operator. The watch, the whole-map search and the first
fix run on full revolutions only (:meth:`pepin.sources.SourceFeed.full_picture`): a fan sees a
quarter of the room and two frames of the same view agree on the same look-alike, so while a
fan drives the tracker follows it, the watch is off and ``/relocalize`` refuses.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from functools import partial
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

from pepin.dynamic import StaticMask, berth_for, dynamic_marks, occluded
from pepin.flags import Flag, FlagSet
from pepin.localization import Localizer
from pepin.mapping import GridSpec, OccupancyGrid
from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import CorrelativeMatcher, SearchWindow
from pepin.slip import SlipWatch
from pepin.sources import CONTACT, DEPTH, LIDAR, ScanObservation, SourceFeed, SourceRegistry
from pepin.timeline import (
    MatchPacer,
    MotionEdge,
    MotionFilter,
    OdomHistory,
    TimedScan,
    deskew,
    standing_still,
    timed_scan_from_ros,
)
from pepin.watch import DRIVE_FIT, LOST_FIT, LostWatch, Verdict
from pepin_bringup.msgs import (
    cloud_from_points,
    planar_mount,
    pose_with_covariance,
    stamp_from_seconds,
    transform_from_rpy,
    yaw_of,
)
from pepin_bringup.node_kit import Switches, TfLookup, spin_main

OCCUPIED_LOG_ODDS, FREE_LOG_ODDS = 4.0, -4.0
DUMP_DIR = (
    "/maps/rec"  # every failed whole-map search leaves its scan here, for the offline autopsy
)
LAST_POSE_FILE = "/maps/last_pose.json"  # where the robot stood when the stack last ran
LAST_POSE_MAX_AGE_S = 3600.0
SLIP_SAID_AFTER = 3  # consecutive slipping scans before the log says it once
# The camera's scans arrive in base_link already (pepin_bringup.depth_stream, contact_scan):
# no mount to apply, unlike the lidar's, which is looked up from /tf_static.
NO_MOUNT = (0.0, 0.0, 0.0, False)
CAMERA_SCANS = ((DEPTH, "/depth_scan"), (CONTACT, "/contact_scan"))


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
        description="hold the pose while the cart stands still (wheels and gyro agree): a"
        " match's residual is blended in with a time constant instead of taken whole",
    ),
    Flag(
        "explained_vote",
        True,
        description="returns the static map cannot explain (a person, a moved chair) do not"
        " score the match",
    ),
    Flag(
        "rest_tau_s",
        6.0,
        range=(0.1, 60.0),
        description="the rest lock's time constant: seconds for a residual to die at rest",
    ),
    Flag(
        "rest_gain",
        0.05,
        range=(0.0, 1.0),
        description="the rest lock's share per match when no match cadence is known",
    ),
    Flag(
        "sources",
        (LIDAR,),
        choices=(LIDAR, DEPTH, CONTACT),
        description="the scan sources matched against the map: the lidar's revolution (/scan),"
        " the camera's depth band (/depth_scan), the floor-contact line (/contact_scan); the"
        " lidar drives the updates while it is fresh and the others ride along, a stale lidar"
        " hands the updates to them. The fused modes are measured offline"
        " (scratch/camera_only_localization.py) and turned on live",
    ),
    Flag(
        "fusion",
        True,
        description="fuse every enabled source's match by its information; off: the widest"
        " source corrects alone and the others only report",
    ),
)


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


def grid_from_msg(msg: OccupancyGridMsg) -> OccupancyGrid:
    """A nav_msgs map as our log-odds grid (occupied +4, free -4, unknown 0)."""
    info = msg.info
    spec = GridSpec(
        info.resolution,
        info.origin.position.x,
        info.origin.position.y,
        info.width * info.resolution,
        info.height * info.resolution,
    )
    grid = OccupancyGrid(spec)
    data = np.array(msg.data, dtype=np.int16).reshape(info.height, info.width)
    grid.log_odds[:] = np.where(
        data >= 65, OCCUPIED_LOG_ODDS, np.where((data >= 0) & (data <= 35), FREE_LOG_ODDS, 0.0)
    )
    grid.version += 1
    return grid


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
        self._motion = MotionFilter(min_m=0.005, min_deg=0.3, max_gap_s=1.0)
        self._rested = 0  # scans left unmatched because the cart stood still (per report)
        # What the map does not explain (a person, a moved chair) is published as lethal rings for
        # both costmaps: the point planners' berth around new objects (pepin.dynamic).
        self._static_mask: StaticMask | None = None
        self._dynamic_pub = self.create_publisher(PointCloud2, "/dynamic_obstacles", 5)
        self._dynamic_count = 0  # marks published since the last report
        self._berth = berth_for("GridBased")  # until the goal server says who plans
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
        for name, topic in CAMERA_SCANS:
            self.create_subscription(
                LaserScan,
                topic,
                partial(self._on_camera_scan, name),
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            )
        self.create_subscription(Odometry, self._odom_topic, self._on_odom, 20)
        self._fit_pub = self.create_publisher(Float32, "localization_fit", 5)
        # Every source's word on each update, as JSON (Localizer.sources_report): the demo's
        # view of the lidar and the camera agreeing, disagreeing, or one of them gone.
        self._sources_pub = self.create_publisher(String, "/localization/sources", 5)
        # The fit at the pose actually published (the blend), beside the tracker's own fit at
        # the matched pose: the two part ways while a carry is being absorbed.
        self._published_fit_pub = self.create_publisher(Float32, "localization_fit_published", 5)
        self._pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 5)
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
        self.get_logger().info("relocalizer up: watching the scan-to-map fit")

    # -- inputs -------------------------------------------------------------

    def _now_s(self) -> float:
        """The node's clock in seconds: the same clock the scans are stamped with."""
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _on_switch(self, name: str, _old: Any, new: Any) -> None:
        """A flag changed (``ros2 param set``): it is the Localizer's own switch, written
        through at once so the next scan is matched with it (``sources`` reaches the roster
        the feed shares with it, so the anchor moves with the flag)."""
        if self._localizer is not None:
            self._localizer.switch(name, new)

    def _on_map(self, msg: OccupancyGridMsg) -> None:
        self._grid = grid_from_msg(msg)
        with self._episode:  # a candidate found on the old map is evidence about nothing here
            self._watch = LostWatch(**self._watch_args)  # type: ignore[arg-type]
            self._pending_seed = None
        self._matcher = CorrelativeMatcher(self._grid)
        self._static_mask = StaticMask(self._grid)
        # lost_after huge: update() must never run a whole-map search in the executor thread on
        # this board (10-20 s); the 1 Hz watcher below does that in a worker and re-seeds.
        # Tracking window sized for this board: 7x7 positions x 13 headings x 120 beams is about
        # 60 ms per scan on an A53 (the laptop default, 9x9x49x200, took 360 ms: 2 Hz).
        origin = msg.info.origin.position
        self._map_id = f"{msg.info.width}x{msg.info.height}@{origin.x:.2f},{origin.y:.2f}"
        flags = self._switches.flags.as_dict()
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

    def _on_camera_scan(self, name: str, msg: LaserScan) -> None:
        """A camera fan (``/depth_scan``, ``/contact_scan``): a LaserScan in base_link already,
        taken in one instant (its ``scan_time`` is zero: no beam to deskew)."""
        self._offer(name, msg, NO_MOUNT)

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
        if self._track:
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
        if self._track:
            self._track_pending()

    def _track_pending(self) -> None:
        """Match the anchor's scan at the feed once the odometry history covers its whole
        revolution, with the other sources' scans carried to its moment riding along.

        Called on every input: a scan usually arrives before the odometry of its last beams and
        is released by the odometry sample that completes it, 30-70 ms later. Which source's
        scan is released is the feed's call (the lidar while it is fresh, else the camera; a
        stale lidar's death is what moves the anchor, never a restart). The scan is deskewed
        with the history (every beam moved to where the robot was at the stamp) and the pose
        the localizer predicts from is the interpolated pose at that same stamp, so the
        residual the matcher reports is odometry error and nothing else.
        """
        loc = self._localizer
        if loc is None:
            return
        now = self._now_s()
        taken = self._feed.take(self._history, now)
        if taken is None:
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
        points = deskew(scan.points, scan.times, self._history, scan.stamp)
        if points is None:
            self._deskew_failed += 1
            points = scan.points
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
            trust_odometry=not slip,
            at_rest=at_rest,
            dt_s=dt_s,
            mask=self._static_mask,
        )
        self._pacer.matched(mono, time.perf_counter() - t0)
        self._last_map_odom = map_to_odom(pose, odom)
        self._send_map_odom()
        self._publish_tracker_pose(
            pose, loc.confidence, Time(nanoseconds=int(scan.stamp * 1e9)).to_msg()
        )
        self._published_fit_pub.publish(Float32(data=float(loc.published_fit)))
        self._sources_pub.publish(String(data=json.dumps(loc.sources_report(now))))
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
        self._berth = berth_for(msg.data)
        self.get_logger().info(
            f"planner {msg.data}: dynamic rings {self._berth.ring_m:.2f} m, "
            f"none within {self._berth.near_m:.2f} m"
        )

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
        """The tracked pose for the operator's view and the trail; sigma grows as the fit drops."""
        msg = pose_with_covariance(
            pose.x,
            pose.y,
            pose.theta,
            0.05 + 0.3 * (1.0 - confidence),
            math.radians(3.0) + math.radians(20.0) * (1.0 - confidence),
            stamp,
            "map",
        )
        self._tracker_pub.publish(msg)
        self._append_trail(msg)

    def _report_tracking(self) -> None:
        """Every 30 s: where every released scan went (per source, with the rides), what the
        tracker did with the matched ones (its switches, the rest lock's cadence and gain,
        carries, lost/weak, the fit at the match and at the published pose, the largest
        map -> odom step), who drives and every source's health, and the cost."""
        loc = self._localizer
        if loc is None:
            return
        feed, pacer, track = self._feed.report(), self._pacer.report(), loc.report()
        self.get_logger().info(
            f"tracker: {feed.summary()}, rested {self._rested}, {pacer.summary()}, deskew "
            f"failed {self._deskew_failed}; {loc.settings()}; {track.summary()}; "
            f"sources: {self._feed.status(self._now_s())}; "
            f"watch {'fit' if self._watch_on else 'off: no full-turn source, fit'} "
            f"{self.fit:.2f}, dynamic marks {self._dynamic_count}, "
            f"scan age at match {self._last_scan_age_s * 1000:.0f} ms; "
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
        only: while a fan drives, the fit is published and the watch is off."""
        now = self._now_s()
        full = self._feed.full_picture(now)
        picture = full if full is not None else self._feed.picture(now)
        if self._matcher is None or picture is None:
            return
        pose = self._tracked_pose()
        if pose is None:
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
            float(self._localizer.confidence)
            if self._localizer is not None and moving
            else self._matcher.inlier_fraction(pose, picture.points)
        )
        self._fit_pub.publish(Float32(data=float(self._watch.reported_fit(self.fit))))
        # A fan drives: its fit cannot say "lost" (LOST_FIT was tuned on full revolutions) and
        # a search on it would re-seed the tracker on a look-alike the twin check cannot see:
        # the watch is off, and the report line says so.
        self._watch_on = full is not None
        if self._searching or not self._tracker_initialised or not self._watch_on:
            return
        # The watch gets the tracker's OWN fit. reported_fit() is for the outside world: capped
        # at 0.35 while a candidate pends, and fed back here it kept a twin candidate alive at
        # fit 0.76 and ran a whole-map search every second for half an hour (18:00 today).
        occluded = self._occluded(picture.points, pose)
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
            threading.Thread(target=self._search_and_seed, args=(picture,), daemon=True).start()
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

    def _seed(self, pose: Pose2D, confidence: float) -> None:
        """Adopt ``pose`` with the confidence it was measured at; AMCL is told via /initialpose."""
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
        self._pose_pub.publish(
            pose_with_covariance(
                pose.x,
                pose.y,
                pose.theta,
                0.05,
                math.radians(5.0),
                self.get_clock().now().to_msg(),
                "map",
            )
        )

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
