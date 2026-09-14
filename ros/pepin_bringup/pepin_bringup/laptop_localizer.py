"""The map matching the board cannot afford, run on the laptop: a watchdog and the camera.

The board's tracker follows the cart in a window around the odometry's prediction and asks
"where am I, really?" only after several scans have fitted the map badly — and the answer costs
it seconds of a Cortex-A53 it is also driving with. The laptop does the same search in a tenth
of a second and has nothing else to do with that tenth. This node is what it does with it, in
two halves that share a map, a belief and a report line:

THE WATCHDOG (``global_watch``). Once a second the whole map is searched on the board's newest
lidar revolution and the place found goes back as a candidate (:class:`pepin.watchdog.Global
Candidate`) on ``/localization/candidate``; the board's tracker judges it against its own
fresher pose and re-seeds itself when a streak of them agrees. Never a re-seed of its own: this
node proposes, the board decides.

THE CAMERA (``camera_sources``). The depth band and the floor-contact line are produced HERE —
the network runs on this machine's GPU — and until 2026-09-13 the scans crossed the link so the
Orange Pi could match them: three matches a revolution took its tracker from 45 ms to 147, it
kept every second revolution (4.7 Hz), and the camera's word was by then stale enough to pull
the live pose 50 cm p90 off the lidar's truth (scratch/drive_bisect.py, run 0238). So the
matching moved to the data. Each camera scan, ``camera_match_hz`` times a second per source, is
matched in a SMALL window around the board's own belief carried to that scan's stamp — the
camera refines a pose, it never searches for one — and the result travels as a measurement
(:class:`pepin.measurements.RemoteMeasurement`) on ``/localization/measurement``: a place, the
covariance read off the score surface with the source's own trust charged into it, the fit, the
scan's stamp and the map it means something on. The board carries it to its next update and
fuses it by information. The match takes the same vote the board's tracker takes on its own
scans (``explained_vote``): the returns the map cannot explain — a person's legs, a chair that
moved — are silenced, which matters more to a +-40 degree fan one person can fill than to a
full revolution. With this node off, or the link down, the board simply has no camera
measurements and tracks on the lidar as it always did.

In: ``/scan`` and ``/map`` (the board's, over the bridge; the laser mount from ``/tf_static``),
``/depth_scan`` and ``/contact_scan`` (local — no bridge hop), ``/odometry/filtered`` (the trail
a belief is carried along), and the board's own word, ``/tracker_pose`` with
``/localization_fit``. ``/map_camera`` — the camera's own band of the world volume
(pepin_bringup.depth_fusion, ``map_source=volume``) — is what the camera scans are matched
against when it exists, because a tabletop the lidar's plane never sees is in that slice and in
no other; the measurement is still labelled with ``/map``'s id, since that is the map the board
holds. Out: the candidate, the measurement, and ``/localization/candidate_pose`` for the
operator's 3D view.

One search, one revolution: a revolution already in hand is never searched twice. A message
repeating the stamp in hand is dropped, a revolution nobody has replaced within
``watch_max_scan_age_s`` stops being searched at all, and the id of the revolution travels with
the candidate — so a frozen ``/scan`` here cannot become three pieces of evidence over there
(:class:`pepin.watchdog.CandidateGate`, ``distinct_scans``).

The scan is NOT deskewed here (no odometry history of the board's own beams) and it is matched
raw: the exhaustive pass runs on a 0.1 m grid at 9 degree headings, where a revolution's smear
is under one cell, and the board re-measures every candidate in its own tracking window before
adopting it. The cost of a search and of a camera match is in every report line.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer

from pepin.dynamic import StaticMask
from pepin.flags import Flag, FlagSet
from pepin.fusion import COVARIANCE_CHOICES, PEAK
from pepin.localization import Localizer
from pepin.measurements import RemoteMeasurement
from pepin.odometry import Pose2D
from pepin.scanmatch import SearchWindow, apply_motion, relative_motion
from pepin.sources import CONTACT, DEPTH, WATCHDOG, SourceRegistry
from pepin.timeline import OdomHistory, TimedScan, timed_scan_from_ros
from pepin.watchdog import CandidateVerdict, GlobalCandidate, ambiguity, judge
from pepin_bringup.msgs import (
    grid_from_msg,
    map_id,
    planar_mount,
    pose_with_covariance,
    stamp_from_seconds,
    yaw_of,
)
from pepin_bringup.node_kit import Switches, Tally, TfLookup, Worker, spin_main

CANDIDATE_TOPIC = "/localization/candidate"  # what pepin_bringup.relocalizer subscribes to
CANDIDATE_POSE_TOPIC = "/localization/candidate_pose"  # the same, for Foxglove
MEASUREMENT_TOPIC = "/localization/measurement"  # the camera's poses, for the board's tracker
CAMERA_MAP_TOPIC = "/map_camera"  # the camera's own band of the world volume, when there is one
CAMERA_SCANS = ((DEPTH, "/depth_scan"), (CONTACT, "/contact_scan"))
# The camera's scans arrive in base_link already (pepin_bringup.depth_stream, contact_scan):
# no mount to apply, unlike the lidar's, which is looked up from /tf_static.
NO_MOUNT = (0.0, 0.0, 0.0, False)
TICK_S = 0.2  # how often the period is checked; the period itself is the flag
MIN_POINTS = 60  # a revolution this thin cannot say where the cart is on a whole map
STAGES = ("search", "measure", "camera", "mask")
# The search's arguments, the board's own (pepin_bringup.relocalizer._relocalize): the same
# question, so an answer from here and an answer from there are comparable numbers.
THETA_STEP_DEG = 10.0
THIN_TO = 90
# The laptop's tracking window, used only to re-measure the winner and read its covariance off
# the score surface: finer than the board's 9 cm / 1.5 deg because this machine can afford it.
WINDOW = SearchWindow(xy_m=0.09, xy_step_m=0.015, theta_deg=9.0, theta_step_deg=0.75)
# ...and the steps a camera match is read on, its extent being the two flags below.
CAMERA_STEP_M = 0.015
CAMERA_STEP_DEG = 0.75
# A camera scan is matched around the pose the tracker believes in, carried to the scan's own
# stamp. How old that belief may be before the carry is a guess rather than a carry: the board
# publishes a pose per matched scan — about 10 Hz driving, about 1 Hz at rest, where the cart is
# not moving and the carry is exact — so two seconds is a dead board or a dead bridge, not a
# quiet one.
BELIEF_MAX_AGE_S = 2.0
# A camera match is made in the tracker's own terms, not a whole-map search's: a fan of a few
# dozen returns is judged by how many of them land on walls the map knows, with the tracking
# floor for how much of the scan must be judgeable at all (pepin.scanmatch.inlier_fraction).
TRACK_MIN_KNOWN = 0.25
ODOM_HORIZON_S = 5.0

FLAGS = FlagSet(
    Flag(
        "global_watch",
        True,
        description="run the whole-map search once every watch_period_s and publish what it finds"
        f" on {CANDIDATE_TOPIC}; off, this half of the node is a subscriber that costs nothing and"
        " the board is back to searching for itself only once it is already lost",
        why="measured on the kidnap tape (run 0171, where the odometry jumps 1 m and 40 deg while"
        " the scans do not): the tracker's own window never recovered — 0.69 m of error still"
        " there after 39 s — and the board's own slow search found the truth four times (fits"
        " 0.76/0.72/0.79/0.75 against the tracker's 0.50-0.60) and died unconfirmed every time,"
        " because at a metre off this flat still fits 0.53, just under the 0.55 that declares the"
        " cart lost. This search costs 125 ms median (p90 165, max 258) against the board's 3700"
        " ms, and with the shipped streak of 3 it brought the cart back in 2.9 s with 0 false"
        " re-seeds over the undisturbed tape",
        on_when="whenever the map is a known one and the laptop is up",
        off_when="in SLAM mode: the map is RTAB-Map's there and still being built, so a whole-map"
        " search searches a map that changes under it",
    ),
    Flag(
        "watch_period_s",
        1.0,
        description="seconds between searches",
        why="it follows from the measured cost and the streak: one search is 125 ms median, 165"
        " ms p90, 258 ms max of one core on this machine, so 1 Hz is 12-26 % of a core, and one"
        " candidate a second is what makes the shipped streak of 3 cost 2.9 s of recovery",
        on_when="shorten it when recovery must be faster than three seconds and the laptop has"
        " the core to spare",
        off_when="lengthen it on a busy laptop, or on a map large enough that a search costs more"
        " than the measured 0.26 s",
        range=(0.2, 60.0),
    ),
    Flag(
        "watch_max_scan_age_s",
        1.0,
        description="how long a revolution may sit in hand and still be searched, counted from"
        " when it ARRIVED here",
        why="default by design, unmeasured as a number; the rule behind it is measured. /scan"
        " arrives at 9.8-10.8 Hz, so a healthy revolution is about 0.1 s old and one second is"
        " ten missed ones. The age is taken on this machine's monotonic clock and never from the"
        " stamp, because the board's clock runs 2.3-2.8 s ahead of the Mac's: a stamp-based age"
        " would either never fire or switch the watch off for good",
        on_when="raise it when the bridge is slow but honest and candidates are being dropped as"
        " stale",
        off_when="a huge value is the old behaviour, which searched whatever was held — including"
        " a frozen scan, publishing the same answer again as if it were news",
        range=(0.1, 3600.0),
    ),
    Flag(
        "camera_sources",
        (DEPTH, CONTACT),
        description="which camera scans are matched here and sent to the board as pose"
        f" measurements on {MEASUREMENT_TOPIC}: the depth band, the floor-contact line; empty,"
        " nothing is matched and the board tracks on the lidar alone",
        why="both, because fused they are what stays within 0.7/1.6/5.7 cm of lidar-only over run"
        " 0171 while neither carries the map alone (the depth band alone loses it in 0.5 s, the"
        " contact line in 12 s: scratch/camera_only_localization.py). Matching them HERE is the"
        " day's verdict: on the board the same pair cost 147 ms a scan, 4.7 Hz and 50 cm p90 of"
        " live error (scratch/drive_bisect.py, runs 0238-0241), and on this machine a match is"
        " a few milliseconds of a core that has nothing else to do",
        on_when="whenever the camera is meant to help the pose — parked bumper to furniture, a"
        " blocked or dead lidar",
        off_when="empty is the switch that takes the camera out of the tracker's pose without"
        " touching the costmap layers, and the state to leave it in while the camera's own"
        " numbers are in doubt",
        choices=(DEPTH, CONTACT),
    ),
    Flag(
        "camera_match_hz",
        5.0,
        description="how often each camera source is matched and a measurement published",
        why="the cadence the offline replay fused at and the cadence the camera delivers: the"
        " depth pipeline runs at 9-11 fps and the contact scan beside it, and the replay that"
        " cost 0.7 cm fused every frame. 5 Hz per source is half of what arrives — two matches a"
        " frame period, a few ms each here — and it is what the board's own update rate can"
        " absorb without a measurement ever waiting longer than its carry is honest",
        on_when="raise it towards the camera's own rate when the pose must follow the camera"
        " closely and this machine is idle",
        off_when="lower it on a busy laptop: the board fuses whatever arrives, and a measurement"
        " that comes at 2 Hz is still carried honestly to the update that takes it",
        range=(0.2, 30.0),
    ),
    Flag(
        "camera_window_m",
        0.09,
        description="half-width of the window a camera scan is matched in, metres, around the"
        " board's belief carried to that scan's moment",
        why="the tracker's own window (the board's 0.09 m), which is what the offline replay"
        " matched the camera scans in for its 0.7 cm: the camera REFINES a pose that the lidar"
        " and the odometry already hold to centimetres, it does not search for one. Wider is not"
        " better here — a +-40 degree fan has look-alikes a hand's width away that a full"
        " revolution does not",
        on_when="widen it where the board's belief is poor and the camera is expected to pull it"
        " back — a long blind stretch, a lidar that has been off",
        off_when="narrow it to make a camera match cheaper and safer still; below the odometry's"
        " own error over a fifth of a second it stops being able to correct anything",
        range=(0.01, 1.0),
    ),
    Flag(
        "camera_window_deg",
        9.0,
        description="half-width of the same window in heading, degrees",
        why="the tracker's own 9 degrees, the replay's settings: a fan's heading is the one thing"
        " it measures well, and the belief it starts from is never more than a degree or two out"
        " while the lidar is alive",
        on_when="widen it after a stretch on odometry alone, where the heading is what drifts",
        off_when="narrow it where the cart turns little and every degree of search is cost",
        range=(0.5, 90.0),
    ),
    Flag(
        "camera_min_fit",
        0.25,
        description="a camera match whose fit is below this is not sent: it is counted as"
        " low fit and the board never hears about it",
        why="0.25 is the fit at which the tracker itself calls a scan weak"
        " (pepin.localization's lost_below): below it the scan explains nothing and its pose is"
        " the window's tie-break, not a measurement. It refuses only that much — the camera's"
        " fans sit at 0.5-0.6 against the lidar's map on run 0171, and the worst live"
        " camera-only fits of 2026-09-13 were 0.35-0.46. The covariance already widens a poor"
        " match a hundredfold at the bound; this floor is for what is not a match at all",
        on_when="raise it to send only matches the map really explains — a room the camera sees"
        " badly, a map that has moved on",
        off_when="lower it to let the board's own disagreement gate do all the judging, which is"
        " what it is there for",
        range=(0.0, 1.0),
    ),
    Flag(
        "covariance",
        PEAK,
        description="how sure a camera measurement says it is: peak — the spread of that match's"
        " own score peak at the camera matcher's temperature (config/matcher.json); fit — the"
        " fit-scaled second moment of the whole surface, with the source's trust in it, that"
        " shipped before it. It is the number the board's information filter weighs the fan by",
        why="the fan's covariance decides everything the camera is allowed to do to the pose,"
        " and the fit-scaled one was never held against an error. On the peak path the scale is"
        " calibrated (T = 0.016, mean NEES 2.99 over 10047 lidar matches of the four goto tapes"
        " of 2026-09-13: scratch/peak_temperature.py) and reads 0.9-1.2 cm at a good fit, so the"
        " number a fan sends means something. It does not by itself weigh the camera down: both"
        " covariances shrink about sixfold together, and on the real matcher's own lattices a"
        " +-40 deg fan 5 cm off the truth keeps its share of the across-wall information —"
        " 29.8 % on fit, 34.8 % here, pulling the fused pose 17.4 mm of the 5 cm against 15.9"
        " (scratch/peak_skeptic_fuse.py). The camera's own temperature is"
        " PROVISIONAL, the lidar's"
        " number: no camera scan is on those tapes, and until an operator records"
        " /localization/measurement against /tracker_pose and runs scratch/peak_temperature.py"
        " --camera, the source's trust (0.5) keeps widening the fan on top of its peak",
        on_when="on: the board weighs the camera by a spread that means something",
        off_when="fit is what every tape before 2026-09-13 was recorded with, for an A/B; and"
        " the switch to reach for if a calibrated fan ever misbehaves in the field. Flip it"
        " TOGETHER with relocalizer's flag of the same name: a laptop on peak against a board on"
        " fit hands the same fan 75 % of the across-wall information and 38.9 mm of a 5 cm pull"
        " instead of 34.8 % and 17.4 mm (scratch/peak_skeptic_fuse.py)",
        choices=COVARIANCE_CHOICES,
    ),
    Flag(
        "explained_vote",
        True,
        description="returns the map cannot explain (a person, a moved chair) do not score a"
        " camera match: the same vote the board's tracker takes on its own scans"
        " (relocalizer's explained_vote), taken here, on the grid the camera is matched"
        " against",
        why="it is the board's own switch and it followed the match here: until 2026-09-13 these"
        " two fans were matched inside Localizer.update_from, which builds the vote from the"
        " static mask whenever explained_vote is on, and moving the matching to this machine"
        " took the vote off them silently. Measured on the furnished room with a person standing"
        " in the fan (scratch/camera_vote_probe.py): with 10 to 18 of the 41 beams on his legs,"
        " he moves the measured pose by 2.2 cm median and 2.5 cm at worst without the vote, and"
        " by 0.3 cm with it — a systematic pull that grows with how much of the fan he fills,"
        " replaced by a slide of a few mm. The worst voted case is 3.8 cm, a thinned fan sliding"
        " inside its own plateau, and the fan's sigma there is 8-11 cm, so the fusion already"
        " discounts it. The fans' floors are on the roster (vote_min_points 20): a mask that"
        " would leave a fan too thin to fix a pose is dropped and the whole scan votes",
        on_when="in a room with people and furniture that moves — the room this robot lives in",
        off_when="to measure what the vote costs or buys the camera (A/B against the board's"
        " lidar-only pose), or in an empty room where every return should count",
    ),
)


class LaptopLocalizer(Node):
    """Searches the whole map for the cart once a second, and matches the camera's scans around
    the board's belief — the two things the board has no CPU for."""

    def __init__(self) -> None:
        super().__init__("laptop_localizer")
        self._scan_topic = str(self.declare_parameter("scan_topic", "/scan").value)
        self._odom_topic = str(self.declare_parameter("odom_topic", "/odometry/filtered").value)
        self._switches = Switches(self, FLAGS, on_change=self._on_switch)
        self._tally = Tally(STAGES)
        self._localizer: Localizer | None = None
        self._map_id = ""
        self._grid: Any = None  # the board's map, for a camera matcher built without /map_camera
        self._camera: Localizer | None = None  # the matcher the camera scans are refined by
        self._camera_map = ""  # which grid that is: "/map" or "/map_camera"
        self._camera_map_id = ""  # the band's shape and origin: a new one is worth a log line
        self._mask: StaticMask | None = None  # what that grid explains; built on first use
        self._mask_of: tuple[Any, int] | None = None  # ...the grid and version it was built on
        self._roster = SourceRegistry(enabled=(DEPTH, CONTACT))  # for the sources' own trust
        self._laser: tuple[float, float, float, bool] | None = None  # x, y, yaw, mirrored
        self._scan: TimedScan | None = None  # the newest revolution, in base_link
        self._scan_at = 0.0  # monotonic, when that revolution arrived here
        self._pose: Pose2D | None = None  # what the board's tracker believes
        self._pose_stamp = 0.0  # ...and the moment that belief speaks for
        self._fit = 0.0  # ...and how well its scan fits the map there
        self._history = OdomHistory(horizon_s=ODOM_HORIZON_S)  # the trail a belief is carried on
        self._last_search = 0.0  # monotonic, of the last search STARTED
        self._last_match: dict[str, float] = {}  # scan stamp of the last match, per camera source
        self._last: GlobalCandidate | None = None  # the newest candidate, for the report line
        self._sent: RemoteMeasurement | None = None  # ...and the newest measurement
        self._scan_id = 0
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        newest = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGridMsg, "/map", self._on_map, latched)
        self.create_subscription(OccupancyGridMsg, CAMERA_MAP_TOPIC, self._on_camera_map, latched)
        self.create_subscription(LaserScan, self._scan_topic, self._on_scan, newest)
        for name, topic in CAMERA_SCANS:
            self.create_subscription(LaserScan, topic, partial(self._on_camera_scan, name), newest)
        self.create_subscription(Odometry, self._odom_topic, self._on_odom, 20)
        self.create_subscription(
            PoseWithCovarianceStamped, "/tracker_pose", self._on_tracker_pose, 5
        )
        self.create_subscription(Float32, "/localization_fit", self._on_fit, 5)
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
        self._pub = self.create_publisher(String, CANDIDATE_TOPIC, newest)
        self._pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, CANDIDATE_POSE_TOPIC, newest
        )
        # Depth 5: a measurement is a moment of its own and the board carries each one to the
        # update that takes it, so a short queue is a few tenths of a second of history, not a
        # backlog of stale opinions about now.
        self._measurement_pub = self.create_publisher(
            String, MEASUREMENT_TOPIC, QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        )
        self._worker = Worker(
            self._search, name="laptop_localizer", on_error=self._on_error
        ).start()
        self.create_timer(TICK_S, self._tick)
        self.create_timer(30.0, self._report)
        self.get_logger().info(
            f"laptop localizer up: the whole map searched every"
            f" {float(self._switches['watch_period_s']):.1f} s on {self._scan_topic}"
            f" -> {CANDIDATE_TOPIC}; the camera matched at"
            f" {float(self._switches['camera_match_hz']):.1f} Hz -> {MEASUREMENT_TOPIC};"
            f" flags: {self._switches.state()}"
        )

    def close(self) -> None:
        """Stop the search thread and wait for its current search, before the node is destroyed."""
        if not self._worker.stop():
            self.get_logger().warning("the search did not finish in time; leaving anyway")
        self._tf.close()

    def _on_error(self, text: str) -> None:
        self._tally.count("failed")
        self._tally.note("failure", text.strip().splitlines()[-1])
        self.get_logger().error(f"the whole-map search failed:\n{text}")

    def _on_switch(self, name: str, _old: Any, new: Any) -> None:
        """A flag changed: the two that size the camera's window rebuild its matcher, so the
        next scan is matched in the window just asked for; ``covariance`` is written through to
        both matchers, so the next measurement carries the covariance just asked for; the rest
        are read where they are used."""
        if name in ("camera_window_m", "camera_window_deg"):
            self._camera = None
            return
        if name == "covariance":
            for target in (self._localizer, self._camera):
                if target is not None:
                    target.switch(name, new)

    # ---- inputs ----------------------------------------------------------------------------
    def _on_map(self, msg: OccupancyGridMsg) -> None:
        """The board's saved map: a tracker of our own is built on it, for its search alone —
        and, until ``/map_camera`` says otherwise, for the camera's matches too."""
        self._grid = grid_from_msg(msg)
        self._map_id = map_id(msg)
        self._localizer = Localizer(
            self._grid,
            Pose2D(),
            window=WINDOW,
            global_retry=False,
            covariance=str(self._switches["covariance"]),
        )
        if self._camera_map != CAMERA_MAP_TOPIC:
            self._camera = None  # rebuilt on the next camera scan, on this grid
        self._tally.count("maps")
        self.get_logger().info(
            f"map received: {msg.info.width}x{msg.info.height} cells, id {self._map_id}"
        )

    def _on_camera_map(self, msg: OccupancyGridMsg) -> None:
        """The camera's own band of the world volume (pepin_bringup.depth_fusion): the slice the
        camera's scans are cut from, and so the one they are matched against. The seats and
        tabletops in it are in no other view of the map; the measurement is still labelled with
        /map's id, because that is the map the board holds."""
        grid = grid_from_msg(msg)
        self._camera = Localizer(
            grid,
            Pose2D(),
            window=self._camera_window(),
            global_retry=False,
            covariance=str(self._switches["covariance"]),
        )
        self._camera_map = CAMERA_MAP_TOPIC
        self._tally.count("camera_maps")
        # The volume republishes its band once a second whether or not it changed shape, and one
        # line per publication buried the log (3600 identical lines an hour, 2026-09-13). The
        # line a person needs is the one where the band becomes a different grid; the rest are a
        # count in the report.
        shape = map_id(msg)
        if shape != self._camera_map_id:
            self._camera_map_id = shape
            self.get_logger().info(
                f"camera map received: {msg.info.width}x{msg.info.height} cells, id {shape};"
                " the camera's scans are matched against the volume's own band from now on"
            )

    def _on_scan(self, msg: LaserScan) -> None:
        """The board's lidar revolution, moved into base_link by the mount read once.

        A message whose stamp is not newer than the one in hand is the same revolution over
        again (the bridge redelivering, a publisher looping) and is dropped: the id below is
        then the identity of the REVOLUTION, which is what the board's gate counts a streak in,
        and not a count of messages.
        """
        if self._laser is None and not self._lookup_laser(msg.header.frame_id):
            return
        assert self._laser is not None
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self._scan is not None and stamp <= self._scan.stamp:
            self._tally.count("repeat")
            return
        self._scan_id += 1
        self._scan_at = time.monotonic()
        self._scan = timed_scan_from_ros(
            stamp,
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            msg.range_max,
            msg.scan_time,
            self._laser,
            self._scan_id,
        )
        self._tally.count("scans")

    def _on_odom(self, msg: Odometry) -> None:
        """The board's fused odometry: the trail the tracker's belief is carried along to the
        moment of a camera scan, and the only thing here that knows the cart has moved."""
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self._history.add(
            Time.from_msg(msg.header.stamp).nanoseconds * 1e-9, Pose2D(p.x, p.y, yaw_of(q))
        )

    def _lookup_laser(self, frame: str) -> bool:
        """The static base_link <- laser transform as x, y, yaw and whether roll is pi."""
        t = self._tf.transform("base_link", frame)
        if t is None:  # not yet available: try on the next scan
            return False
        x, _y, yaw, mirrored = self._laser = planar_mount(t)
        self.get_logger().info(f"laser mount: x {x:.3f} yaw {yaw:.2f} rad, mirrored {mirrored}")
        return True

    def _on_tf_static(self, msg: TFMessage) -> None:
        """The static frames (latched): the laser mount is read from them once."""
        for transform in msg.transforms:
            self._tf.buffer.set_transform_static(transform, "static")

    def _on_tracker_pose(self, msg: PoseWithCovarianceStamped) -> None:
        """What the board's tracker believes, and the moment it speaks for: the verdict beside a
        candidate is judged against it, and every camera match starts from it."""
        p = msg.pose.pose
        self._pose = Pose2D(p.position.x, p.position.y, yaw_of(p.orientation))
        self._pose_stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

    def _on_fit(self, msg: Float32) -> None:
        """The tracker's own scan-to-map fit: what a candidate must beat to disagree."""
        self._fit = float(msg.data)

    # ---- the camera ------------------------------------------------------------------------
    def _camera_window(self) -> SearchWindow:
        """The window a camera scan is refined in, from the two flags that size it."""
        return SearchWindow(
            xy_m=float(self._switches["camera_window_m"]),
            xy_step_m=CAMERA_STEP_M,
            theta_deg=float(self._switches["camera_window_deg"]),
            theta_step_deg=CAMERA_STEP_DEG,
        )

    def _camera_matcher(self) -> Localizer | None:
        """The matcher the camera's scans are refined by: the volume's camera band once
        ``/map_camera`` has arrived, else the board's own map; ``None`` before any map."""
        if self._camera is None and self._grid is not None:
            self._camera = Localizer(
                self._grid,
                Pose2D(),
                window=self._camera_window(),
                global_retry=False,
                covariance=str(self._switches["covariance"]),
            )
            self._camera_map = "/map"
        return self._camera

    def _camera_mask(self, localizer: Localizer) -> StaticMask | None:
        """What the grid the camera is matched against explains, for the match's vote
        (``explained_vote``): ``None`` with the flag off. Built on first use and kept until that
        grid is replaced — /map_camera arrives once a second and a dilation of the whole grid is
        not worth doing per scan, while a mask of the PREVIOUS band would silence the returns of
        a room that has since moved on."""
        if not self._switches.on("explained_vote"):
            return None
        grid = localizer.grid
        if self._mask is None or self._mask_of != (grid, grid.version):
            with self._tally.measure("mask"):
                self._mask = StaticMask(grid)
            self._mask_of = (grid, grid.version)
        return self._mask

    def _belief_at(self, stamp: float) -> Pose2D | None:
        """The board's pose at ``stamp``: its newest belief carried there over the odometry
        between the two moments. ``None`` — counted with its reason — when there is no belief
        yet, when it is older than :data:`BELIEF_MAX_AGE_S` (a dead board, a dead bridge), or
        when the odometry trail does not cover both moments, because a carry over a gap is a
        guess and a guess is not a place to start a match from."""
        if self._pose is None:
            self._tally.count("no_belief")
            return None
        if stamp - self._pose_stamp > BELIEF_MAX_AGE_S:
            self._tally.count("stale_belief")
            return None
        then, now = self._history.at(self._pose_stamp), self._history.at(stamp)
        if then is None or now is None:
            self._tally.count("no_odometry")
            return None
        return apply_motion(self._pose, relative_motion(then, now))

    def _on_camera_scan(self, source: str, msg: LaserScan) -> None:
        """One camera scan (``/depth_scan``, ``/contact_scan``): matched around the board's
        belief at its own stamp and published as a measurement.

        Paced by ``camera_match_hz`` per source on the SCAN's stamps, not on this machine's
        clock: the pacing is then the same offline in a replay as it is live, and a burst of
        frames after a stall cannot become a burst of matches. Everything that stops a match is
        counted with its own name, so the report line says why the board heard nothing.
        """
        if source not in self._switches["camera_sources"]:
            self._tally.count("camera_off")
            return
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        period = 1.0 / max(float(self._switches["camera_match_hz"]), 1e-6)
        last = self._last_match.get(source)
        if last is not None and 0.0 <= stamp - last < period:
            self._tally.count("paced")
            return
        localizer = self._camera_matcher()
        if localizer is None:
            self._tally.count("no_map")
            return
        scan = timed_scan_from_ros(
            stamp,
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            msg.range_max,
            msg.scan_time,
            NO_MOUNT,
            0,
        )
        if len(scan.points) < self._roster.source(source).min_points:
            self._tally.count("thin_fan")
            return
        belief = self._belief_at(stamp)
        if belief is None:
            return
        self._last_match[source] = stamp
        mask = self._camera_mask(localizer)
        silenced = localizer.stats.silenced_scans
        with self._tally.measure("camera"):
            measured = localizer.measure(
                belief,
                scan.points,
                source,
                stamp,
                min_known=TRACK_MIN_KNOWN,
                trust=self._roster.source(source).trust,
                mask=mask,
            )
        if localizer.stats.silenced_scans > silenced:
            self._tally.count("voted")  # this fan had furniture in it and it did not score
        self._tally.sample(f"fit_{source}", measured.fit)
        if measured.fit < float(self._switches["camera_min_fit"]):
            self._tally.count("low_fit")
            return
        remote = RemoteMeasurement.of(measured, self._map_id)
        self._sent = remote
        self._tally.count(f"sent_{source}")
        self._measurement_pub.publish(
            String(
                data=remote.to_json(
                    belief_age_ms=round((stamp - self._pose_stamp) * 1e3, 1),
                    matched_on=self._camera_map,
                )
            )
        )

    # ---- the watch -------------------------------------------------------------------------
    def _tick(self) -> None:
        """Every :data:`TICK_S`: offer the newest revolution to the search thread, if the period
        has passed, the watch is on, a map is there, the revolution is fresh and thick enough.

        Fresh is counted on THIS machine's monotonic clock, from the moment the revolution
        arrived — never as the node's clock minus the scan's stamp. The stamp is the board's
        and the board's clock runs 2.3-2.8 s ahead of this Mac's (journal, 2026-09-09): that
        subtraction is a lie whose sign depends on which host drifted, and it would either
        never fire or switch the watch off for good. Arrival is local, and a frozen ``/scan``
        is exactly what it measures.
        """
        now = time.monotonic()
        scan = self._scan
        if not self._switches.on("global_watch"):
            self._tally.count("off")
            return
        if now - self._last_search < float(self._switches["watch_period_s"]):
            return
        if self._localizer is None or scan is None:
            self._tally.count("nothing_to_search")
            return
        if now - self._scan_at > float(self._switches["watch_max_scan_age_s"]):
            self._tally.count("stale")  # nothing new has arrived: the same answer is not news
            return
        if len(scan.points) < MIN_POINTS:
            self._tally.count("thin")
            return
        self._last_search = now
        if self._worker.offer(scan):
            self._tally.count("busy")  # the previous search is still running: it is replaced

    def _search(self, scan: TimedScan) -> None:
        """The search thread: the whole map on one revolution, then the winner re-measured in
        the tracking window for its covariance, then out as a candidate with our own verdict.

        The verdict travels for the operator only; the board judges the candidate again against
        its own pose, which by then is fresher than the one this node has heard.
        """
        localizer, map_id_now = self._localizer, self._map_id
        if localizer is None:
            return
        started = time.perf_counter()
        places = localizer.global_candidates(scan.points, THETA_STEP_DEG, THIN_TO)
        took_s = time.perf_counter() - started
        self._tally.spent("search", took_s)
        if not places:
            self._tally.count("found_nothing")
            return
        best, _fit = places[0]
        with self._tally.measure("measure"):
            measured = localizer.measure(best.pose, scan.points, WATCHDOG, scan.stamp)
        candidate = GlobalCandidate(
            x=measured.x,
            y=measured.y,
            yaw=measured.yaw,
            covariance=measured.covariance,
            score=measured.fit,
            ambiguity=ambiguity(
                [(match.pose, localizer.rank(match.pose, scan.points)) for match, _ in places]
            ),
            stamp=scan.stamp,
            map_id=map_id_now,
            scan_id=scan.scan_id,  # the board counts a streak in scans, not in messages
        )
        pose = self._pose
        verdict = CandidateVerdict.NOTHING if pose is None else judge(candidate, pose, self._fit)
        self._tally.count(str(verdict))
        self._tally.count("published")
        self._last = candidate
        self._pub.publish(
            String(data=candidate.to_json(verdict=str(verdict), search_ms=round(took_s * 1e3, 1)))
        )
        sigma_x, sigma_y, sigma_yaw = measured.sigmas
        self._pose_pub.publish(
            pose_with_covariance(
                candidate.x,
                candidate.y,
                candidate.yaw,
                max(sigma_x, sigma_y),
                sigma_yaw,
                stamp_from_seconds(scan.stamp),
                "map",
            )
        )

    # ---- the report ------------------------------------------------------------------------
    def _report(self) -> None:
        """Every 30 s: how many searches ran and what they said, how many measurements each
        camera source sent and what stopped the rest, what both cost, and the flags."""
        w = self._tally.take()
        c = w.counts
        verdicts = ", ".join(f"{v} {c[str(v)]}" for v in CandidateVerdict)
        last = "none yet" if self._last is None else self._last.text()
        self.get_logger().info(
            f"laptop localizer: {c['published']} candidates from {c['scans']} scans"
            f" ({verdicts}); last {last}; tracker fit {self._fit:.2f};"
            f" skipped: off {c['off']}, no map or scan {c['nothing_to_search']},"
            f" nothing new {c['stale']}, thin {c['thin']}, still searching {c['busy']},"
            f" found nothing {c['found_nothing']}, failed {c['failed']};"
            f" revolutions heard twice {c['repeat']}; {self._camera_line(w)};"
            f" ms median/max: {w.stages()}; flags: {self._switches.state()}"
        )
        if c["failed"]:
            self.get_logger().warning(f"the last failure: {w.notes.get('failure', '')}")

    def _camera_line(self, w: Any) -> str:
        """The camera half of the report: what each source sent and at what fit, what the
        matcher matched against, and every reason a scan did not become a measurement."""
        c = w.counts
        sent = ", ".join(
            f"{name} {c[f'sent_{name}']} at fit {self._median(w, name)}"
            for name, _topic in CAMERA_SCANS
        )
        last = "none yet" if self._sent is None else self._sent.text()
        return (
            f"measurements: {sent} (against {self._camera_map or 'no map'}"
            f" {self._camera_map_id or '-'}, received {c['camera_maps']}x,"
            f" {c['voted']} with unexplained returns silenced), last {last};"
            f" rejected: no belief {c['no_belief']}, stale belief {c['stale_belief']},"
            f" no odometry {c['no_odometry']}, low fit {c['low_fit']}, thin {c['thin_fan']},"
            f" no map {c['no_map']}; paced {c['paced']}, source off {c['camera_off']}"
        )

    @staticmethod
    def _median(w: Any, source: str) -> str:
        """The median fit of one camera source's matches in this window, or ``-``."""
        values = sorted(w.samples.get(f"fit_{source}", []))
        return f"{values[len(values) // 2]:.2f}" if values else "-"


def main(args: list[str] | None = None) -> None:
    """Entry point."""
    spin_main(LaptopLocalizer, args)


if __name__ == "__main__":
    main()
