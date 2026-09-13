"""The whole-map search, run on the laptop once a second, forever: a watchdog over the tracker.

The board's tracker follows the cart in a window around the odometry's prediction and asks
"where am I, really?" only after several scans have fitted the map badly — and the answer costs
it seconds of a Cortex-A53 it is also driving with. The laptop does the same search in a tenth
of a second and has nothing else to do with that tenth. So it asks the question continuously,
while the tracker is still healthy, and sends the answer to the board as a candidate.

In: ``/scan`` (the board's lidar, over the bridge; its mount comes from ``/tf_static``),
``/map``, and the tracker's own word — ``/tracker_pose`` and ``/localization_fit`` — so the
verdict published beside the candidate is judged against what the board believed at that
moment. Out: ``/localization/candidate``, one self-contained JSON message
(:class:`pepin.watchdog.GlobalCandidate`) that the board's tracker judges again against its own
fresher pose, and ``/localization/candidate_pose``, the same pose with its covariance for the
operator's 3D view. The board decides; this node only proposes, and proposes the same way
whether the tracker is healthy or lost.

Never a re-seed of its own, and never a map: this node holds no frame, publishes no transform
and owns nothing. Switched off (``global_watch`` false) it is a subscriber that costs nothing,
and the board's own slow search is the fallback it always was. In SLAM mode there is no saved
map to search — RTAB-Map builds the map while the cart drives, and a search against a map that
is still growing would be a search against the answer — so the launch starts this node with the
flag off.

One search, one revolution: a revolution already in hand is never searched twice. A message
repeating the stamp in hand is dropped, a revolution nobody has replaced within
``watch_max_scan_age_s`` stops being searched at all, and the id of the revolution travels with
the candidate — so a frozen ``/scan`` here cannot become three pieces of evidence over there
(:class:`pepin.watchdog.CandidateGate`, ``distinct_scans``).

The scan is NOT deskewed here (no odometry history on this side) and it is matched raw: the
exhaustive pass runs on a 0.1 m grid at 9 degree headings, where a revolution's smear is under
one cell, and the board re-measures every candidate in its own tracking window before adopting
it. The cost of one search is in every report line.
"""

from __future__ import annotations

import time

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer

from pepin.flags import Flag, FlagSet
from pepin.localization import Localizer
from pepin.odometry import Pose2D
from pepin.scanmatch import SearchWindow
from pepin.sources import WATCHDOG
from pepin.timeline import TimedScan, timed_scan_from_ros
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
TICK_S = 0.2  # how often the period is checked; the period itself is the flag
MIN_POINTS = 60  # a revolution this thin cannot say where the cart is on a whole map
STAGES = ("search", "measure")
# The search's arguments, the board's own (pepin_bringup.relocalizer._relocalize): the same
# question, so an answer from here and an answer from there are comparable numbers.
THETA_STEP_DEG = 10.0
THIN_TO = 90
# The laptop's tracking window, used only to re-measure the winner and read its covariance off
# the score surface: finer than the board's 9 cm / 1.5 deg because this machine can afford it.
WINDOW = SearchWindow(xy_m=0.09, xy_step_m=0.015, theta_deg=9.0, theta_step_deg=0.75)

FLAGS = FlagSet(
    Flag(
        "global_watch",
        True,
        description="run the whole-map search once every watch_period_s and publish what it finds"
        f" on {CANDIDATE_TOPIC}; off, this node is a subscriber that costs nothing and the board"
        " is back to searching for itself only once it is already lost",
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
)


class GlobalWatch(Node):
    """Searches the whole map for the cart once a second and publishes the place it found."""

    def __init__(self) -> None:
        super().__init__("global_watch")
        self._scan_topic = str(self.declare_parameter("scan_topic", "/scan").value)
        self._switches = Switches(self, FLAGS)
        self._tally = Tally(STAGES)
        self._localizer: Localizer | None = None
        self._map_id = ""
        self._laser: tuple[float, float, float, bool] | None = None  # x, y, yaw, mirrored
        self._scan: TimedScan | None = None  # the newest revolution, in base_link
        self._scan_at = 0.0  # monotonic, when that revolution arrived here
        self._pose: Pose2D | None = None  # what the board's tracker believes
        self._fit = 0.0  # ...and how well its scan fits the map there
        self._last_search = 0.0  # monotonic, of the last search STARTED
        self._last: GlobalCandidate | None = None  # the newest candidate, for the report line
        self._scan_id = 0
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        newest = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGridMsg, "/map", self._on_map, latched)
        self.create_subscription(LaserScan, self._scan_topic, self._on_scan, newest)
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
        self._worker = Worker(self._search, name="global_watch", on_error=self._on_error).start()
        self.create_timer(TICK_S, self._tick)
        self.create_timer(30.0, self._report)
        self.get_logger().info(
            f"global watch up: the whole map searched every"
            f" {float(self._switches['watch_period_s']):.1f} s on {self._scan_topic}"
            f" -> {CANDIDATE_TOPIC}; flags: {self._switches.state()}"
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

    # ---- inputs ----------------------------------------------------------------------------
    def _on_map(self, msg: OccupancyGridMsg) -> None:
        """The board's saved map: a tracker of our own is built on it, for its search alone."""
        grid = grid_from_msg(msg)
        self._map_id = map_id(msg)
        self._localizer = Localizer(grid, Pose2D(), window=WINDOW, global_retry=False)
        self._tally.count("maps")
        self.get_logger().info(
            f"map received: {msg.info.width}x{msg.info.height} cells, id {self._map_id}"
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
        """What the board's tracker believes right now: the verdict is judged against it."""
        p = msg.pose.pose
        self._pose = Pose2D(p.position.x, p.position.y, yaw_of(p.orientation))

    def _on_fit(self, msg: Float32) -> None:
        """The tracker's own scan-to-map fit: what a candidate must beat to disagree."""
        self._fit = float(msg.data)

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
        """Every 30 s: how many searches ran and what they said, the newest candidate, what the
        searches cost, and the flags."""
        w = self._tally.take()
        c = w.counts
        verdicts = ", ".join(f"{v} {c[str(v)]}" for v in CandidateVerdict)
        last = "none yet" if self._last is None else self._last.text()
        self.get_logger().info(
            f"global watch: {c['published']} candidates from {c['scans']} scans"
            f" ({verdicts}); last {last}; tracker fit {self._fit:.2f};"
            f" skipped: off {c['off']}, no map or scan {c['nothing_to_search']},"
            f" nothing new {c['stale']}, thin {c['thin']}, still searching {c['busy']},"
            f" found nothing {c['found_nothing']}, failed {c['failed']};"
            f" revolutions heard twice {c['repeat']}; ms median/max: {w.stages()};"
            f" flags: {self._switches.state()}"
        )
        if c["failed"]:
            self.get_logger().warning(f"the last failure: {w.notes.get('failure', '')}")


def main(args: list[str] | None = None) -> None:
    """Entry point."""
    spin_main(GlobalWatch, args)


if __name__ == "__main__":
    main()
