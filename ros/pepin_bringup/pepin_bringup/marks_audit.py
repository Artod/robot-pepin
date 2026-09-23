"""The camera's phantom obstacles as a live number: who painted each lethal cell of the local map.

Run 0431 (2026-09-22) drove 0 m towards the printer. Nine FollowPath goals aborted in ten seconds,
Hybrid-A* found no valid path, and the operator watched lethal blobs stand in front of a cart with
open floor ahead of it. It took an hour of tape replay
(``scratch/one_localiser/tape_0431_phantoms.py``) to say that 54-73 % of those cells had no lidar
return anywhere near them while the camera's fan covered them. This node is that replay run live,
once per published costmap: the operator gets the count while the room is still in front of him,
and can turn the camera layer off, or point at the table the camera is right about, within
seconds instead of after the drive.

WHAT IT DOES, once per ``/local_costmap/costmap`` message (the board publishes it at 1 Hz). Every
lethal cell within ``radius_m`` of the cart is taken and asked two questions in turn, both in the
GRID'S OWN FRAME — ``header.frame_id``, read from the message and never assumed, because that grid
moved from ``map`` to ``odom`` on 2026-09-22 (ros/params/nav2_params.yaml) and will move again:

* is there a ``/scan`` return within ``match_cells`` of it, the lidar's beams placed in that frame
  through TF at the SCAN'S own stamp? Then it is the lidar's and it is not news.
* if not: does a ``/depth_marks`` beam — the camera's fan, ``pepin_bringup.depth_fusion``'s slice
  of the accumulated volume, in ``base_link`` — land on it? Then it is **camera-only**.
* neither: **unexplained** — a ToF whisker's cone, a mark nothing has raytraced away yet, or the
  frames having come apart. A number that grows is a symptom on its own.

WHAT IT CANNOT TELL, and does not pretend to: a camera-only cell is either a phantom (SGBM lifting
herringbone parquet, a stale voxel, a wrong pose) or a real thing standing above the lidar's plane
that only the camera can see — a table top, a seat, a shelf. The arithmetic cannot separate those
and the operator can: he is looking at the room. The node reports the count and the distance to
the nearest one, and says nothing about which it is.

WHERE IT RUNS, and why here. The laptop (CLAUDE.md rule 20). It consumes the camera's own fan and
only ever reads: nothing real-time depends on it, it must not cost the board a millisecond, and
with the laptop gone the board drives on exactly as before. Its cost is the arithmetic of
:mod:`pepin.marks_audit` — brute-force distances over a 60x60 grid, a few milliseconds once a
second, on one worker thread that drops a costmap arriving while the previous one is still being
read.

WHAT THE OPERATOR READS. ``/marks_audit``, a ``std_msgs/String`` of JSON with the four counts and
the nearest camera-only distance (a Foxglove plot or a Raw Messages panel: no custom message type
is worth one number a second), ``/marks_audit/phantoms``, a ``sensor_msgs/PointCloud2`` of the
camera-only cells as RED points in the grid's frame — drop it into the 3D panel beside the costmap
and the phantoms are the cells that light up — and one report line every 10 s in the log.

The flags (:data:`FLAGS`, all live, ``ros/flags.sh set marks_audit <name> <value>``):
``marks_audit`` (the whole node: off, it subscribes and computes nothing), ``radius_m``,
``match_cells``, ``inscribed_counts`` (judge the inflation's 99 band too) and ``phantom_cloud``
(publish the red points); their state is in the start line and in every report line.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String

from pepin.flags import Flag, FlagSet
from pepin.marks_audit import (
    MATCH_CELLS,
    RADIUS_M,
    MarksVerdict,
    audit_marks,
    scan_points,
    transform_xy,
)
from pepin_bringup.msgs import cloud_from_points, scan_arrays, stamp_seconds
from pepin_bringup.node_kit import Switches, Tally, TfLookup, Worker, spin_main

COSTMAP_TOPIC = "/local_costmap/costmap"
SCAN_TOPIC = "/scan"
MARKS_TOPIC = "/depth_marks"  # the camera's fan, in base_link (pepin_bringup.depth_fusion)
AUDIT_TOPIC = "/marks_audit"
PHANTOM_TOPIC = "/marks_audit/phantoms"
BASE_FRAME = "base_link"

REPORT_S = 10.0  # shorter than the stack's usual 30 s: this line is read DURING a drive
STAGES = ("place", "classify", "publish")
TF_TIMEOUT_S = 0.1  # the costmap's stamp is the newest thing TF is asked about; it arrives late
MAX_AGE_S = 2.0  # a scan or a fan older than this explains nothing about the grid just published
PHANTOM_RED = (255, 32, 32)

# The node's flags (CLAUDE.md rule 19), declared last in __init__ so the kit's callback sees no
# other declaration. Nothing is cached from them: every costmap reads them, so a change takes
# the next one — which is the point, with the cart standing in front of the blob.
FLAGS = FlagSet(
    Flag(
        "marks_audit",
        True,
        description="the audit runs; off, the node keeps its subscriptions and computes,"
        " publishes and reports nothing",
        why="on, and the cost is bounded by the grid rather than estimated: the local costmap is"
        " 3 x 3 m at 5 cm published at 1 Hz (ros/params/nav2_params.yaml), so the very worst frame"
        " this can be handed is 3600 lethal cells against a 450-return revolution — 1.6 M"
        " distances of numpy once a second, on the laptop, while the board waits for none of it."
        " It is on because the drive that needs the number is the drive nobody knew would go"
        " wrong: run 0431 took a tape and an hour of replay to say that 54-73 % of its near-ahead"
        " lethal cells had no lidar behind them",
        on_when="always, and especially on any drive where the camera layer is marking",
        off_when="to take this laptop's last percent back for a profile of something else",
    ),
    Flag(
        "radius_m",
        RADIUS_M,
        description="how far around the cart a lethal cell is judged, metres",
        why="2.0 m is the lidar layer's own obstacle_max_range (ros/params/nav2_params.yaml): past"
        " it the lidar does not mark at all, so every cell out there would be unbacked by"
        " construction and the classification would say nothing. It is also about where the"
        " controller's collision checks bite",
        on_when="raise it to watch the camera's marks out to its own 2.5 m obstacle_max_range,"
        " knowing that the band between 2.0 and 2.5 m is camera-or-nothing by construction",
        off_when="lower it to the cart's immediate surroundings when only the cells that block a"
        " recovery matter",
        range=(0.2, 3.0),
    ),
    Flag(
        "match_cells",
        MATCH_CELLS,
        description="how near a beam must land to a cell, in costmap cells, to account for it",
        why="1.5 cells is what the tape analysis used (scratch/one_localiser/tape_0431_phantoms.py,"
        " BACK_CELLS): at 5 cm that is 7.5 cm, which covers a cell's own half-diagonal (3.5 cm)"
        " plus the pose error between the scan's stamp and the grid's. Tighter blames the camera"
        " for the lidar's own marks; looser lets the lidar explain a phantom standing beside it",
        on_when="raise it when the two sensors are known to disagree in time (a laggy link) and"
        " the unexplained count is climbing with no obstacle to show for it",
        off_when="lower it to see how tightly the lidar's returns really sit on its own marks",
        range=(0.5, 5.0),
    ),
    Flag(
        "inscribed_counts",
        False,
        description="the inflation's 99 band (costmap 253, INSCRIBED_INFLATED_OBSTACLE) is judged"
        " as a mark too; off, only the 100s a sensor actually wrote",
        why="off, which is the threshold the analysis this node reproduces used: 100 for the LOCAL"
        " grid's marks (scratch/one_localiser/tape_0431_phantoms.py, LETHAL = 100), with the 99"
        " band reserved for the reduced GLOBAL grid, whose cells are classes and not costs. An"
        " inscribed cell (99, costmap 253) is the inflation layer's arithmetic around a mark (100,"
        " costmap 254) and not an observation, so asking which sensor painted it answers a"
        " question nobody asked while multiplying every count by the inflation's own area",
        on_when="to see the band the planners' footprint check really refuses — the cells that"
        " made Hybrid-A* call a start pose blocked",
        off_when="whenever the question is which sensor SAW something",
    ),
    Flag(
        "phantom_cloud",
        True,
        description="the camera-only cells are published as red points on /marks_audit/phantoms;"
        " off, only the counts go out",
        why="on, and the bandwidth is bounded by the grid: the cloud can never hold more than the"
        " 3600 cells of a 3 x 3 m grid at 5 cm, 32 bytes each in the PCL layout"
        " (pepin_bringup.msgs.cloud_from_points), so 115 kB/s at the impossible worst and about"
        " 3 kB/s for the hundred cells a real frame has — and none of it crosses the radio, since"
        " both this node and Foxglove's bridge are on the laptop. The counts alone do not say"
        " WHERE the phantoms are, and where is what sends the operator to the right furniture",
        on_when="always, while looking at the 3D panel",
        off_when="on a link already saturated by the surface cloud",
    ),
)


class MarksAudit(Node):
    """Classifies every local-costmap frame's lethal cells by the sensor that can account for
    them, and publishes the counts, the phantom cells and a report line."""

    def __init__(self) -> None:
        super().__init__("marks_audit")
        # Callbacks can run BEFORE this constructor is done: TfLookup starts a spin thread for
        # this node and every subscription is live from the moment it exists (depth_fusion's
        # AttributeError of 2026-09-22, which killed the executor and silenced the marks). Until
        # the last line of __init__ every callback drops what it is handed.
        self._up = False
        self._switches = Switches(self, FLAGS)
        self._tally = Tally(STAGES)
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self._audit_pub = self.create_publisher(String, AUDIT_TOPIC, reliable)
        self._phantom_pub = self.create_publisher(PointCloud2, PHANTOM_TOPIC, reliable)
        self._scan: LaserScan | None = None  # the newest revolution...
        self._marks: LaserScan | None = None  # ...and the newest slice of the volume
        self._verdict: MarksVerdict | None = None  # the last frame's, for the report line
        self._tf = TfLookup(self, on_failure=self._on_tf_failure)
        self._worker = Worker(self._audit, name="marks_audit", on_error=self._on_work_error).start()
        # One deep: the grid arrives once a second and only the newest one describes the room the
        # operator is looking at. Plain RELIABLE, volatile — what run_recorder has read this topic
        # with on the board since it existed. Nav2 publishes it RELIABLE and transient-local, and
        # a volatile reader matches that; a transient-local READER would not match a publisher
        # that ever loses the durability, which is a race nobody needs for a 1 Hz topic.
        self.create_subscription(OccupancyGrid, COSTMAP_TOPIC, self._on_costmap, 1)
        # RELIABLE, like every other reader of this topic here (the tracker, the depth stream, the
        # fusion): the board publishes it reliably and a best-effort reader of a reliable writer
        # over the link gets nothing at all, in silence.
        self.create_subscription(
            LaserScan,
            SCAN_TOPIC,
            self._on_scan,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE),
        )
        # The camera's fan is published on this laptop by depth_fusion with exactly this profile.
        self.create_subscription(LaserScan, MARKS_TOPIC, self._on_marks, reliable)
        self.create_timer(REPORT_S, self._report)
        self.get_logger().info(
            f"marks audit up: every {COSTMAP_TOPIC} frame's lethal cells split into lidar-backed"
            f" ({SCAN_TOPIC}), camera-only ({MARKS_TOPIC}) and unexplained, in the grid's own"
            f" frame; out as JSON on {AUDIT_TOPIC} and red points on {PHANTOM_TOPIC}; a"
            " camera-only cell is a phantom OR a real thing above the lidar's plane and this node"
            f" cannot tell which; flags: {self._switches.state()}"
        )
        self._up = True

    def close(self) -> None:
        """Stop the worker and the TF listener, before the node is destroyed under them."""
        if not self._worker.stop():
            self.get_logger().warning("the audit worker did not finish its grid; leaving anyway")
        self._tf.close()

    # ---- inputs ------------------------------------------------------------------------------
    def _on_work_error(self, text: str) -> None:
        self.get_logger().error(f"the marks audit failed on a grid:\n{text}")

    def _on_tf_failure(self, kind: str, text: str) -> None:
        """A transform this audit needed was not there: counted by kind, and the last message
        kept for the report line. Never fatal — a grid nobody can place is simply not judged."""
        self._tally.count(f"tf_{kind.lower()}")
        self._tally.note("tf", text)

    def _on_scan(self, msg: LaserScan) -> None:
        """The lidar's newest revolution, kept whole: it is placed in the grid's frame at ITS own
        stamp, so the message travels rather than a set of points in some other frame."""
        if not self._up:
            return  # the node is still being built (see __init__)
        self._scan = msg
        self._tally.count("scans")

    def _on_marks(self, msg: LaserScan) -> None:
        """The camera's newest fan (the volume's surface sliced around the cart), kept whole."""
        if not self._up:
            return  # the node is still being built (see __init__)
        self._marks = msg
        self._tally.count("marks")

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        """One published local costmap, handed to the worker; one still being read is replaced."""
        if not self._up:
            return  # the node is still being built (see __init__)
        self._tally.count("grids")
        if not self._switches.on("marks_audit"):
            self._tally.count("off")
            return
        if self._worker.offer(msg):
            self._tally.count("dropped")

    # ---- the audit ---------------------------------------------------------------------------
    def _placed(self, scan: LaserScan | None, frame: str, at: float) -> tuple[Any, float]:
        """One sensor's returns placed in ``frame`` as (n, 2), and the age in seconds of the
        message they came from. No message, a stale one, or no transform at its stamp: an empty
        array and the age (``inf`` when there was nothing at all)."""
        if scan is None:
            return np.zeros((0, 2)), float("inf")
        age = at - stamp_seconds(scan.header.stamp)
        if not math.isfinite(age) or abs(age) > MAX_AGE_S:
            return np.zeros((0, 2)), age
        pose = self._tf.pose(frame, scan.header.frame_id, scan.header.stamp, TF_TIMEOUT_S)
        if pose is None:
            return np.zeros((0, 2)), age
        angles, ranges = scan_arrays(scan)
        return transform_xy(scan_points(angles, ranges), pose.rotation, pose.translation), age

    def _audit(self, msg: OccupancyGrid) -> None:
        """One grid classified, published and counted, or counted and dropped when the cart's own
        place in that grid's frame is not in TF."""
        frame = msg.header.frame_id  # odom since 2026-09-22; read, never assumed
        at = stamp_seconds(msg.header.stamp)
        with self._tally.measure("place"):
            here = self._tf.pose(frame, BASE_FRAME, msg.header.stamp, TF_TIMEOUT_S)
            if here is None:
                self._tally.count("unplaced")
                return
            lidar_xy, scan_age = self._placed(self._scan, frame, at)
            camera_xy, marks_age = self._placed(self._marks, frame, at)
        info = msg.info
        with self._tally.measure("classify"):
            verdict = audit_marks(
                np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width),
                (float(info.origin.position.x), float(info.origin.position.y)),
                float(info.resolution),
                (float(here.translation[0]), float(here.translation[1])),
                lidar_xy,
                camera_xy,
                radius_m=float(self._switches["radius_m"]),
                match_cells=float(self._switches["match_cells"]),
                inscribed_counts=self._switches.on("inscribed_counts"),
            )
        with self._tally.measure("publish"):
            self._publish(verdict, msg, at, scan_age, marks_age)
        self._verdict = verdict  # one frozen dataclass, swapped for the report timer to read
        self._tally.count("audited")
        self._tally.count("camera_only", verdict.camera_only)
        self._tally.count("unexplained", verdict.unexplained)
        if np.isfinite(verdict.nearest_camera_only_m):
            self._tally.sample("nearest", verdict.nearest_camera_only_m)

    def _publish(
        self,
        verdict: MarksVerdict,
        msg: OccupancyGrid,
        at: float,
        scan_age: float,
        marks_age: float,
    ) -> None:
        """The counts as JSON, and the camera-only cells as red points in the grid's frame.

        ``nearest_camera_only_m`` is ``null`` rather than infinity when there are none: JSON has
        no infinity, and a plot reads a gap better than a made-up number.
        """
        nearest = verdict.nearest_camera_only_m
        self._audit_pub.publish(
            String(
                data=json.dumps(
                    {
                        "t": round(at, 3),
                        "frame": msg.header.frame_id,
                        "lethal": verdict.lethal,
                        "lidar_backed": verdict.lidar_backed,
                        "camera_only": verdict.camera_only,
                        "unexplained": verdict.unexplained,
                        "nearest_camera_only_m": (
                            round(nearest, 3) if np.isfinite(nearest) else None
                        ),
                        "radius_m": round(float(self._switches["radius_m"]), 2),
                        "scan_age_s": None if not math.isfinite(scan_age) else round(scan_age, 3),
                        "marks_age_s": None
                        if not math.isfinite(marks_age)
                        else round(marks_age, 3),
                    }
                )
            )
        )
        if not self._switches.on("phantom_cloud"):
            return
        xy = verdict.camera_only_xy
        points = np.column_stack((xy, np.zeros(len(xy))))
        colours = np.tile(np.array(PHANTOM_RED, dtype=np.uint8), (len(xy), 1))
        self._phantom_pub.publish(
            cloud_from_points(points, colours, msg.header.stamp, msg.header.frame_id)
        )

    # ---- the report --------------------------------------------------------------------------
    def _report(self) -> None:
        """The last verdict, the window's counts, the sensors' liveness and the flags in one
        line — the line ros/restart.sh reads and the one to watch during a drive."""
        w = self._tally.take()
        c = w.counts
        verdict = self._verdict
        nearest = w.samples.get("nearest", [])
        tf_lost = sum(n for name, n in c.items() if name.startswith("tf_"))
        self.get_logger().info(
            "marks audit: "
            + (verdict.report() if verdict is not None else "no grid judged yet")
            + f"; {w.rate('audited'):.1f} grids/s of {c['grids']} received"
            + (f", {c['dropped']} dropped" if c["dropped"] else "")
            + (f", {c['off']} switched off" if c["off"] else "")
            + (f", {c['unplaced']} unplaceable" if c["unplaced"] else "")
            + f"; {c['scans']} scans, {c['marks']} camera fans"
            + (
                f"; window camera-only median nearest {float(np.median(nearest)):.2f} m"
                if nearest
                else ""
            )
            + (f"; {tf_lost} TF misses ({w.notes.get('tf', '')})" if tf_lost else "")
            + f"; flags: {self._switches.state()}; ms median/max: {w.stages()}"
        )
        if self._switches.on("marks_audit") and c["grids"] == 0:
            self.get_logger().warning(
                f"no {COSTMAP_TOPIC} in this window: is the board's Nav2 up and does the grid"
                " reach this laptop?"
            )
        if c["marks"] == 0 and c["grids"]:
            self.get_logger().warning(
                f"no {MARKS_TOPIC} in this window: every unbacked cell will read 'unexplained',"
                " because the camera's fan is not being published (depth_fusion)"
            )


def main() -> None:
    spin_main(MarksAudit)


if __name__ == "__main__":
    main()
