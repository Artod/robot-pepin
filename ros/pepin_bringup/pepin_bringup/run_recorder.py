"""ROS wiring for `pepin.tape.RunTape`: the topics of a drive, turned into records.

The node that owns the goal owns the tape, so a run's file opens the moment the goal does and
closes with it — and, because the tape always holds the last seconds, it opens with the seconds
*before* the goal on it. The decisions live in `pepin.tape`; this file only converts messages.

Topics: the raw lidar scan, wheel odometry, the tracker's pose, Nav2's plan, the commanded
twist, the three ToF ranges and the local costmap — what `scripts/build_map.py` and the replays
read, plus the proof of what the robot itself believed: which cells it held for occupied when it
refused to move (a question a run could not answer before, and the one every stall raises).

Since 2026-09-14 also the fusion's two String topics (``fusion_records``): every pose the laptop
measured out of a camera scan (``meas``, /localization/measurement) and the tracker's own account
of each update (``srcs``, /localization/sources). They were ros/tools/session_logger.py's alone,
and that second recorder cost the board a whole rclpy process — 15 % of a core and ~140 MB —
deserialising the same lidar stream this node already deserialises. One tape now holds the lot,
so ros/goto.sh no longer starts the session logger when the numbered tape is being written.
"""

from __future__ import annotations

import math
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from action_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan, Range
from std_msgs.msg import String

from pepin.deployment import DEFAULT_LOCALIZER
from pepin.flags import Flag, FlagSet
from pepin.mapcache import run_length_encode
from pepin.mounts import Mounts
from pepin.recording import imu_record, scan_record_from_ros
from pepin.runlink import (
    IDLE,
    RECORDING,
    RUN_COMMAND_TOPIC,
    RUN_STATUS_TOPIC,
    RunStatus,
    parse_command,
)
from pepin.tape import RunTape, camera_clip_path, next_run_number
from pepin_bringup.bridge_kick import BridgeKick
from pepin_bringup.node_kit import Switches, TfLookup

# The live flags (CLAUDE.md rule 19); their state is printed in the node's ready line.
FLAGS = FlagSet(
    Flag(
        "fusion_records",
        True,
        description="the camera's measurements (/localization/measurement) and the tracker's"
        " account of each update (/localization/sources) go on the numbered tape as the"
        " 'meas' and 'srcs' records scratch/camera_error.py reads",
        why="they were recorded only by ros/tools/session_logger.py, a second recorder that"
        " ros/goto.sh started for every drive: another rclpy process on a 4-core A53, 15 % of a"
        " core and ~140 MB, deserialising the same 10 Hz lidar stream this node already"
        " deserialises. Two JSON strings a revolution cost this node almost nothing, and one"
        " tape then holds a whole drive",
        on_when="always: without them a camera measurement cannot be compared to the lidar's"
        " truth after the fact",
        off_when="when the fusion is off anyway and the tape should stay small",
    ),
    Flag(
        "bridge_kick",
        True,
        description="the laptop's request to restart THIS board's zenoh bridge (/bridge/kick)"
        " is answered by touching /run/pepin/bridge_kick, which a systemd path unit on the board"
        " turns into `systemctl restart pepin-bridge`; off, the request is logged and ignored",
        why="of two bridges the one that started LAST gets working routes: a route's DDS"
        " endpoint is built when the route is created and only while the far bridge is already"
        " announcing. On 2026-09-15 a 5 s wireless stall made the board's bridge close the"
        " transport and reconnect with the same zenoh id, and thirteen of its pub routes came"
        " back with an empty dds_reader — nothing crossed from the board until its bridge was"
        " restarted by hand. ros/laptop.sh cures that with ssh (settle_bridge); the laptop's"
        " watch has no ssh and must never have one, so it asks here and the board's own systemd"
        " does the restart. The handler costs this board one subscription to a topic that"
        " carries nothing on a healthy link",
        on_when="always on a split or vision stack: it is the only way the laptop can put the"
        " board's routes back without a human",
        off_when="while bisecting the bridge by hand, so nothing restarts under you",
    ),
    Flag(
        "planner_records",
        True,
        description="what the PLANNER saw goes on the tape too: the global costmap (run-length"
        " encoded, at most one grid per new plan), the goal status of Nav2's three actions"
        " (navigate_to_pose, compute_path_to_pose, follow_path) and the pose graph's own words"
        " (/localization/graph_measurement) beside the camera's; off, the tape holds what it held"
        " before 2026-09-18",
        why="the tape was blind exactly where the failures were. On 2026-09-17 two legs piled up"
        " 78 and 90 recoveries in ~125 s with no path (ros/maps/rec/20260917_192935_goto.log,"
        " ..._201425_goto.log) and the tapes could not say why: they carry /plan and the LOCAL"
        " costmap, and the planner reads the GLOBAL one. The cart's own footprint was clear in"
        " every one of the 1740 taped local grids (scratch/footprint_in_costmap.py), so the answer"
        " was in the grid nobody recorded. Cost, measured on those tapes"
        " (scratch/costmap_rle_cost.py): the planner's grid is 239x215 = 51385 cells, 195 kB of"
        " raw JSON, and 16 kB run-length encoded over the four classes that decide whether the"
        " cart FITS (unknown / free / inflated / the 99-100 lethal band) — 12-fold, and the"
        " gradient it drops is cost, not feasibility. One grid per plan at the tapes' own 1.2 s"
        " plan cadence is 13 kB/s beside the 55 kB/s the scans already write, and one encode of"
        " 51k cells, 2.8 ms on the laptop's core. The status topics carry a message per"
        " transition and the graph's words arrive at 1 Hz",
        on_when="always while Nav2 is the thing being debugged",
        off_when="on a long autonomy run where the tape must stay small, or to reproduce a tape"
        " recorded before 2026-09-18",
    ),
)

# How a taped grid's cells are encoded: :func:`pepin.mapcache.run_length_encode`, the one
# implementation, shared with the map the relocalizer persists — a costmap and a saved map compress
# the same way and a reader of either needs one decoder.
RLE = "rle"
# The four values a feasibility question needs, and the only ones the global grid is taped with:
# Nav2's own unknown and free, everything inflated between them, and the 99-100 band that stops a
# footprint. The gradient in between is what a cost function reads, and it is also what makes a
# costmap incompressible — the local grids of 2026-09-17 shrink 1.5-fold raw and 8.6-fold reduced
# (scratch/costmap_rle_cost.py). A record says which it is (``classes``), so no reader guesses.
UNKNOWN, FREE, INFLATED, LETHAL_BAND = -1, 0, 1, 99


def feasibility_classes(cells: list[int]) -> list[int]:
    """One costmap's cells reduced to the four values that decide whether the cart fits
    (:data:`UNKNOWN`, :data:`FREE`, :data:`INFLATED`, :data:`LETHAL_BAND`)."""
    return [
        UNKNOWN
        if value < 0
        else (FREE if value == 0 else (LETHAL_BAND if value >= 99 else INFLATED))
        for value in cells
    ]


def _yaw(orientation: object) -> float:
    """Yaw in radians from a quaternion message."""
    q = orientation
    x, y, z, w = (getattr(q, name) for name in ("x", "y", "z", "w"))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _stamp(header: object) -> float:
    """The message's own time in seconds; wall time when it carries none."""
    stamp = getattr(header, "stamp", None)
    seconds = stamp.sec + stamp.nanosec * 1e-9 if stamp else 0.0
    return seconds if seconds > 0 else time.time()


class RunRecorder:
    """Subscribes a node to the topics of a drive and feeds them to the run's tape."""

    # While no run is open the tape only keeps a sketch: turning every scan into a record costs
    # 40% of a core on this board (measured 2026-09-08, load average 9 with the controller loop
    # down to 3 Hz), and a prelude does not need 10 Hz. During a run nothing is thinned.
    IDLE_PERIOD_S: ClassVar[dict[str, float]] = {"pose": 0.2, "loc": 0.2}
    # How often map -> base_link is read into the tape where no tracker publishes a pose: the
    # same 5 Hz the tracker's own records are thinned to above, so a tape looks the same either
    # way and a replay does not have to know which stack wrote it.
    LOC_TF_PERIOD_S: ClassVar[float] = 0.2

    def __init__(
        self,
        node: Node,
        directory: Path,
        tape: RunTape | None = None,
        fusion_records: Callable[[], bool] = lambda: True,
        planner_records: Callable[[], bool] = lambda: True,
        loc_from_tf: bool = False,
    ) -> None:
        self._node = node
        self._fusion_records = fusion_records
        self._planner_records = planner_records  # the ``planner_records`` flag, read per record
        self._plan_seq = 0  # how many plans this run has seen: the global costmap's throttle...
        self._gcostmap_seq = -1  # ...and the plan the last taped grid belonged to
        self._last_kept: dict[str, float] = {}
        self._directory = directory
        self._tape = tape or RunTape()
        self.number = 0  # the run's number, said aloud instead of a timestamp
        # The laser's place on the cart, from config/lidar.json: the replays turn the driver's
        # own bearings into robot-frame ones with it (pepin.recording.scan_record_from_ros
        # wants the offset the mount's yaw is the negative of).
        lidar = Mounts.load().lidar
        self._mount_yaw_rad = -math.radians(lidar.yaw_deg)
        self._mount_x_m = lidar.x_m
        scan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._scan_qos = scan_qos
        # Always on: small messages, and they are what the prelude is made of.
        node.create_subscription(Odometry, "/odom", self._on_odom, 20)
        node.create_subscription(PoseWithCovarianceStamped, "/tracker_pose", self._on_loc, 10)
        node.create_subscription(Twist, "/cmd_vel", self._on_cmd, 20)
        # WHERE THE `loc` RECORDS COME FROM when no tracker runs (PEPIN_LOCALIZER=rtabmap):
        # /tracker_pose has no publisher there, and a tape with no pose in it is a drive nobody
        # can replay. The same edge every other consumer composes — the laptop's map -> odom over
        # the transport with the board's own odom -> base_link — read here on a timer, because TF
        # is a lookup and not a topic. Subscribed only in that role, so a stack with a tracker
        # pays neither the timer nor the /tf subscription behind the listener.
        self._tf = TfLookup(node) if loc_from_tf else None
        if self._tf is not None:
            node.create_timer(self.LOC_TF_PERIOD_S, self._loc_from_tf)
        # Only while a run is open: rclpy turns every LaserScan into Python objects BEFORE our
        # callback can decline it, and that deserialisation alone cost 38% of a core between
        # goals on this board (measured 2026-09-08, load average 8.4). A drive gets them from
        # the moment its tape opens, which is before the goal is even sent.
        self._during_run: list[Any] = []
        # rclpy subscriptions may be created and destroyed ONLY on the thread that spins the
        # executor. Doing it from the socket thread killed the node mid-run with "cannot use
        # Destroyable because destruction was requested" — the executor was building its wait set
        # out of the very objects being torn down (run 0027). So the socket thread asks, and this
        # timer, which runs where the executor runs, does it.
        node.create_timer(0.1, self._apply_pending)

    @property
    def recording(self) -> bool:
        """True while a run is being written."""
        return self._tape.recording

    def start(self, name: str) -> Path:
        """Open a numbered tape for this run; returns the path, prelude already in it.

        The stamp is UTC and says so with a trailing ``Z``. This process runs in a container
        whose clock is UTC while the board's own shell, the laptop and ros/goto.sh's files are
        all on the flat's local time, and a bare ``220039`` was read as a drive four hours later
        than it was (2026-09-13). The letter is what stops that; ros/maps/README.md says the
        rest. Set a TZ in the container and this becomes local time on its own.
        """
        self.number = next_run_number(self._directory)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + "Z"
        return self._tape.start(self._directory / f"{self.number:04d}_{stamp}_{name}.jsonl")

    def stop(self) -> None:
        """Close the run's tape and give the board back the cores the lidar stream was costing."""
        self._tape.stop()

    def _apply_pending(self) -> None:
        """Executor thread: keep the run-only subscriptions in step with whether a tape is open.

        Level-triggered on the tape's own state, not on a flag set from the socket thread: an
        edge written there and cleared here could be lost, leaving a run with no scans, and a
        tape that closed itself on its own limit left the subscriptions attached for good.
        """
        want = "listen" if self._tape.recording else "deafen"
        if want == "listen" and not self._during_run:
            strings = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
            # The global costmap is published whole ONCE and then only as updates, so a plain
            # subscription started mid-drive hears nothing at all: it has to be latched.
            latched = QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            self._during_run = [
                self._node.create_subscription(
                    LaserScan, "/ldlidar_node/scan", self._on_scan, self._scan_qos
                ),
                self._node.create_subscription(PathMsg, "/plan", self._on_plan, 5),
                # What the tracker pairs scans with, and the gyro it is built from: without
                # them a wobble of map -> odom cannot be told from wheel slip after the fact.
                self._node.create_subscription(Odometry, "/odometry/filtered", self._on_ekf, 20),
                self._node.create_subscription(
                    Imu, "/imu/data_raw", self._on_imu, qos_profile_sensor_data
                ),
                self._node.create_subscription(
                    OccupancyGrid, "/local_costmap/costmap", self._on_costmap, 1
                ),
                # The ToF too: three sensors at 14 Hz are 42 messages a second that rclpy
                # deserialises before the tape can decline them — a third of a core between
                # goals, for a prelude nobody reads. A drive gets them from its first moment.
                *(
                    self._node.create_subscription(
                        Range, f"/tof/{sensor}", lambda msg, s=sensor: self._on_tof(s, msg), 10
                    )
                    for sensor in ("front", "left", "right")
                ),
                # What the camera measured and what the tracker did with it: two String topics
                # this board already carries, so a subscriber here costs a copy and no new route.
                *(
                    self._node.create_subscription(
                        String, f"/localization/{topic}", handler, strings
                    )
                    for topic, handler in (
                        ("measurement", self._on_measurement),
                        ("graph_measurement", self._on_measurement),
                        ("sources", self._on_sources),
                    )
                ),
                # ...and what the PLANNER saw, which is not the local grid the controller reads:
                # the global costmap (latched, so this subscription gets the full grid at once and
                # the updates after it) and the goal status of the three actions a drive runs
                # through. Behind ``planner_records``; see the flag for the cost.
                *(
                    self._node.create_subscription(
                        OccupancyGrid, "/global_costmap/costmap", self._on_global_costmap, latched
                    )
                    for _ in (0,)
                    if self._planner_records()
                ),
                *(
                    self._node.create_subscription(
                        GoalStatusArray,
                        f"/{action}/_action/status",
                        lambda msg, a=action: self._on_action_status(a, msg),
                        5,
                    )
                    for action in ("navigate_to_pose", "compute_path_to_pose", "follow_path")
                    if self._planner_records()
                ),
            ]
        elif want == "deafen" and self._during_run:
            for subscription in self._during_run:
                self._node.destroy_subscription(subscription)
            self._during_run = []

    def _keep(self, topic: str) -> bool:
        """True when this record is worth building: everything during a run, a sketch when idle."""
        if self._tape.recording:
            return True
        period = self.IDLE_PERIOD_S.get(topic)
        if period is None:
            return False  # the plan and the costmap only matter while a run is being driven
        now = time.monotonic()
        if now - self._last_kept.get(topic, 0.0) < period:
            return False
        self._last_kept[topic] = now
        return True

    def _on_scan(self, msg: LaserScan) -> None:
        if not self._keep("scan"):
            return
        self._tape.add(
            scan_record_from_ros(
                _stamp(msg.header),
                msg.angle_min,
                msg.angle_increment,
                list(msg.ranges),
                list(msg.intensities),
                msg.range_min,
                msg.range_max,
                msg.scan_time,
                mount_yaw_rad=self._mount_yaw_rad,
                mount_x_m=self._mount_x_m,
            )
        )

    def _on_odom(self, msg: Odometry) -> None:
        if not self._keep("pose"):
            return
        self._tape.add(
            {
                "t": _stamp(msg.header),
                "topic": "pose",
                "x": round(msg.pose.pose.position.x, 4),
                "y": round(msg.pose.pose.position.y, 4),
                "theta": round(_yaw(msg.pose.pose.orientation), 5),
            }
        )

    def _on_ekf(self, msg: Odometry) -> None:
        """The fused odometry (odom -> base_link) the tracker pairs scans with, pose and rates."""
        if not self._keep("ekf"):
            return
        self._tape.add(
            {
                "t": _stamp(msg.header),
                "topic": "ekf",
                "x": round(msg.pose.pose.position.x, 4),
                "y": round(msg.pose.pose.position.y, 4),
                "theta": round(_yaw(msg.pose.pose.orientation), 5),
                "vx": round(msg.twist.twist.linear.x, 4),
                "wz": round(msg.twist.twist.angular.z, 4),
            }
        )

    def _on_imu(self, msg: Imu) -> None:
        """The gyro's rates and the accelerometer, raw and in base_link: the heading truth the
        wheels are checked against, and which way was up — without the accelerometer no replay
        can put a frame of a cart tipped over a slipper where it really was
        (``pepin.recording.lean_history``)."""
        if not self._keep("imu"):
            return
        w, a = msg.angular_velocity, msg.linear_acceleration
        self._tape.add(imu_record(_stamp(msg.header), (w.x, w.y, w.z), (a.x, a.y, a.z)))

    def _on_plan(self, msg: PathMsg) -> None:
        """Nav2's global plan as a polyline (at most 200 points), so a replay can draw it.

        A new plan is also what opens the window for one global costmap record: the grid the
        planner read is worth taping exactly when the planner has just read it.
        """
        if not self._keep("plan"):
            return
        self._plan_seq += 1
        step = max(1, len(msg.poses) // 200)
        self._tape.add(
            {
                "t": _stamp(msg.header),
                "topic": "plan",
                "points": [
                    [round(p.pose.position.x, 3), round(p.pose.position.y, 3)]
                    for p in msg.poses[::step]
                ],
            }
        )

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        """The local costmap as the controller sees it (1 Hz, 3x3 m at 5 cm: about 14 kB/s).

        Costs are the ROS 0-100 scale plus -1 for unknown; 99-100 is the lethal/inscribed band.
        """
        if not self._keep("costmap"):
            return
        info = msg.info
        self._tape.add(
            {
                "t": _stamp(msg.header),
                "topic": "costmap",
                "origin": [round(info.origin.position.x, 3), round(info.origin.position.y, 3)],
                "resolution": round(info.resolution, 3),
                "width": int(info.width),
                "height": int(info.height),
                "data": list(msg.data),
            }
        )

    def _on_tof(self, sensor: str, msg: Range) -> None:
        """One ToF reading and its ceiling: a beam at max range means the sensor saw nothing."""
        if not self._keep("tof"):
            return
        self._tape.add(
            {
                "t": _stamp(msg.header),
                "topic": "tof",
                "sensor": sensor,
                "range": round(float(msg.range), 3),
                "max": round(float(msg.max_range), 3),
            }
        )

    def _on_global_costmap(self, msg: OccupancyGrid) -> None:
        """The grid the PLANNER plans on, at most one record per new plan.

        Taped as the four classes that decide whether the cart fits, run-length encoded
        (:func:`feasibility_classes`, :func:`run_length_encode`): 16 kB instead of 195 kB for this
        flat's 239x215 cells. The throttle is the plan counter and not a clock — a grid nobody
        planned on answers no question, and a drive that re-plans ten times a second would
        otherwise write ten grids a second.
        """
        if not self._keep("gcostmap") or not self._planner_records():
            return
        if self._plan_seq == self._gcostmap_seq:
            return
        self._gcostmap_seq = self._plan_seq
        info = msg.info
        self._tape.add(
            {
                "t": _stamp(msg.header),
                "topic": "gcostmap",
                "origin": [round(info.origin.position.x, 3), round(info.origin.position.y, 3)],
                "resolution": round(info.resolution, 3),
                "width": int(info.width),
                "height": int(info.height),
                "encoding": RLE,
                "classes": [UNKNOWN, FREE, INFLATED, LETHAL_BAND],
                "plan": self._plan_seq,
                "data": run_length_encode(feasibility_classes(list(msg.data))),
            }
        )

    def _on_action_status(self, action: str, msg: Any) -> None:
        """What became of Nav2's own actions — the planner's, the controller's, the whole drive's.

        One record per status array, with each goal's status code (action_msgs/GoalStatus: 2
        executing, 4 succeeded, 5 canceled, 6 aborted). This is the only place a tape can say that
        a plan was ABORTED rather than never asked for, which is the question 78 recoveries with no
        path raise (2026-09-17). The topics carry a message per transition, so they are nearly free.
        """
        if not self._keep("nav") or not self._planner_records():
            return
        self._tape.add(
            {
                "t": time.time(),
                "topic": "nav",
                "action": action,
                "status": [int(s.status) for s in msg.status_list],
            }
        )

    def _on_measurement(self, msg: String) -> None:
        """One pose the laptop measured out of a camera scan OR out of the pose graph, kept
        verbatim: nothing here parses it, so a malformed message is on the tape as evidence instead
        of lost. ``t`` is when it ARRIVED; the moment it speaks for is ``stamp`` inside the JSON,
        and the two together are the age the board's gate charges it
        (scratch/word_age.py; until 2026-09-18 the graph's own words were not taped at all, so that
        age could not be read off a tape for the one source a camera-only drive runs on)."""
        if not self._fusion_records():
            return
        self._tape.add({"t": time.time(), "topic": "meas", "json": msg.data})

    def _on_sources(self, msg: String) -> None:
        """The tracker's own account of one update (Localizer.sources_report), verbatim: who
        anchored, what was fused, what was rejected, and every source's fit, delta, sigma and
        self-check ratio."""
        if not self._fusion_records():
            return
        self._tape.add({"t": time.time(), "topic": "srcs", "json": msg.data})

    def _on_cmd(self, msg: Twist) -> None:
        """What the controller asked the wheels for: the only record of the command side."""
        self._tape.add(
            {
                "t": time.time(),
                "topic": "cmd",
                "linear": round(msg.linear.x, 4),
                "angular": round(msg.angular.z, 4),
            }
        )

    def _on_loc(self, msg: PoseWithCovarianceStamped) -> None:
        """The tracker's belief in the map frame; covariance trace as a stand-in confidence."""
        if not self._keep("loc"):
            return
        cov = msg.pose.covariance
        self._tape.add(
            {
                "t": _stamp(msg.header),
                "topic": "loc",
                "source": "tracker",
                "x": round(msg.pose.pose.position.x, 4),
                "y": round(msg.pose.pose.position.y, 4),
                "theta": round(_yaw(msg.pose.pose.orientation), 5),
                "confidence": round(1.0 / (1.0 + cov[0] + cov[7] + cov[35]), 3),
            }
        )

    def _loc_from_tf(self) -> None:
        """The same record read from ``map -> base_link``, where no tracker publishes a pose.

        No ``confidence``: TF carries no covariance, and a made-up number in a tape is worse
        than a missing one — ``source`` says which of the two wrote the record.
        """
        if self._tf is None or not self._keep("loc"):
            return
        transform = self._tf.transform("map", "base_link", timeout_s=0.0)
        if transform is None:
            return
        self._tape.add(
            {
                "t": _stamp(transform.header),
                "topic": "loc",
                "source": "tf",
                "x": round(transform.transform.translation.x, 4),
                "y": round(transform.transform.translation.y, 4),
                "theta": round(_yaw(transform.transform.rotation), 5),
            }
        )


CAMERA_STREAM = "http://127.0.0.1:8080/stream"  # ustreamer on the board's host network


class RunRecorderNode(Node):
    """Records every drive where its sensors are, on the goal server's word.

    The goal server (on the board or on the laptop) publishes one command on
    ``pepin/run``; this node opens or closes the numbered tape, copies the camera stream next
    to it, and answers on the latched ``pepin/run_status`` with the run's number and path.
    """

    def __init__(self) -> None:
        super().__init__("run_recorder")
        self._record_dir = Path(str(self.declare_parameter("record_dir", "/maps/rec").value))
        # WHO OWNS map -> odom (PEPIN_LOCALIZER, pepin.deployment.localizer; the launch passes
        # the resolved word). Declared BEFORE the switches: rclpy runs their callback on
        # declarations too and refuses a name outside the flags table (node_kit.Switches).
        # Under "rtabmap" nothing publishes /tracker_pose, so the tape's `loc` records are read
        # from TF instead; under "tracker" the tape is exactly what it has always been.
        self._localizer = str(self.declare_parameter("localizer", DEFAULT_LOCALIZER).value)
        self._switches = Switches(self, FLAGS)
        self._recorder = RunRecorder(
            self,
            self._record_dir,
            fusion_records=lambda: self._switches.on("fusion_records"),
            planner_records=lambda: self._switches.on("planner_records"),
            loc_from_tf=self._localizer != "tracker",
        )
        # The laptop's one way to restart the board's zenoh bridge without an ssh key
        # (pepin_bringup.bridge_kick): this node hosts the handler because it is the only one
        # of ours that runs on the board in every mode, and the handler is three lines and a
        # file write.
        self._kick = BridgeKick(self, enabled=lambda: self._switches.on("bridge_kick"))
        self._camera: subprocess.Popen[bytes] | None = None  # curl copying the stream
        self._camera_clip: Path | None = None
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._status_pub = self.create_publisher(String, RUN_STATUS_TOPIC, latched)
        self.create_subscription(String, RUN_COMMAND_TOPIC, self._on_command, 10)
        self._say(RunStatus(IDLE))
        loc_source = "/tracker_pose" if self._localizer == "tracker" else "TF map -> base_link"
        self.get_logger().info(
            f"run recorder ready: tapes in {self._record_dir}; localizer {self._localizer}"
            f" (loc from {loc_source}); flags: {self._switches.state()}"
        )

    def _say(self, status: RunStatus) -> None:
        self._status_pub.publish(String(data=status.to_json()))

    def _on_command(self, msg: String) -> None:
        parsed = parse_command(msg.data)
        if parsed is None:
            self.get_logger().warning(f"run command not understood: {msg.data[:80]}")
            return
        cmd, name = parsed
        if cmd == "start" and name is not None:
            if self._recorder.recording:
                self.stop()
            path = self._recorder.start(name)
            self._start_camera(path)
            self.get_logger().info(f"run {self._recorder.number}: recording {path}")
            self._say(RunStatus(RECORDING, self._recorder.number, str(path), name))
        else:
            self.stop()

    def stop(self) -> None:
        """Close the tape (flushed and synced) and the clip; harmless when no run is open."""
        was = self._recorder.recording
        self._recorder.stop()
        self._stop_camera()
        if was:
            self.get_logger().info(f"run {self._recorder.number}: closed")
        self._say(RunStatus(IDLE, self._recorder.number, None, None))

    def _start_camera(self, tape: Path) -> None:
        """Copy the camera's MJPEG stream next to the tape: no laptop in the loop.

        curl writes the stream as-is (a few percent of a core); the laptop converts after.
        """
        self._stop_camera()
        clip = camera_clip_path(tape)
        self._camera_clip = clip
        try:
            self._camera = subprocess.Popen(
                ["curl", "-s", "-m", "1800", CAMERA_STREAM, "-o", str(clip)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            self._camera = None
            self.get_logger().warning(f"camera clip not started: {exc}")

    def _stop_camera(self) -> None:
        """End the clip; a stream that was never reachable leaves an empty file, removed here."""
        proc, self._camera = self._camera, None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        clip, self._camera_clip = self._camera_clip, None
        if clip is not None and clip.exists() and clip.stat().st_size == 0:
            clip.unlink()


def main() -> None:
    import rclpy

    rclpy.init()
    node = RunRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
