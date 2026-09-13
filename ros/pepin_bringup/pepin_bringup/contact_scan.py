"""The floor's own edge as a second lidar: a depth frame in, a scan at height zero out.

The camera's depth already reaches the board as ``/depth_scan`` — the pixels 8 cm to 1.3 m above
the floor folded onto the plane, which is where a table top or a seat stands. What that scan
cannot say is where a body TOUCHES the floor: the legs under a seat, a box lower than the lidar's
plane, the plinth of a sofa set back from its front. :mod:`pepin.contact` reads exactly that off
the same frame: going up each image column the pixels are floor until the depth stops growing the
way the floor's would, and the ray through that boundary pixel, intersected with the floor plane,
is a range that does not depend on the network's scale at all — only the mount, the optics and the
cart's lean enter it. This node is that function wired to ROS: ``/camera/depth`` (32FC1 metres,
published by :mod:`pepin_bringup.depth_stream` once the lidar has fitted the depth's law),
``/camera/camera_info`` for the optics and ``/imu/data_raw`` for which way is up; out goes
``/contact_scan``, a LaserScan in base_link over the same half-degree fan as ``/depth_scan``, NaN
where no column could say anything (a costmap neither marks nor clears there).

Reading the ANCHORED depth costs nothing: :func:`pepin.depth.floor_anchor` snaps pixels within
4 cm of the plane onto it, and 4 cm is inside the contact band itself
(:class:`pepin.contact.DepthNoise`, 13 cm at 2 m), so an anchored pixel was already floor before
it was snapped and no column's verdict moves. Frames that arrive while one is being read are
dropped — newest wins, like the depth node — and the floor's geometry
(:class:`pepin.contact.FloorPlane`, a tenth of a second of trigonometry on a 1280x720 image) is
rebuilt only when the lean or the optics move.

The range cap is measured, not chosen: past 2 m the network's floor leaves the band on its own and
every column ends on open parquet (:data:`pepin.contact.CONTACT_MAX_RANGE`, measured in
``scratch/contact_vs_lidar.py`` on run 0171). Where the lidar returns at 1.50-1.75 m the median
difference is +1 cm and 3 % of the marks are false; below 1.25 m the picture's bottom row already
looks 1.02 m ahead, so the scan reports the foot of whatever stands behind the near overhang and
the lidar owns that metre.

The flags (:data:`FLAGS`, all live, ``ros/flags.sh set contact_scan <name> <value>``):
``contact_scan`` (publish or not), ``shadow`` (take the band's own width back off the range),
``imu_lean`` (the floor plane follows the gyro too, not the accelerometer alone) and
``max_range``; their state and the last frame's :class:`pepin.contact.ContactVerdict` are printed
in every report line.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, LaserScan

from pepin.camera import CameraConfig, mount_transform
from pepin.contact import (
    CONTACT_MAX_RANGE,
    N_BINS,
    ContactVerdict,
    FloorPlane,
    contact_scan,
)
from pepin.depth import SCAN_HALF_FOV, SCAN_STEP, Array, CameraPose, Intrinsics
from pepin.flags import Flag, FlagSet
from pepin_bringup.msgs import array_from_image, scan_from_ranges
from pepin_bringup.node_kit import LeanFeed, Switches, Tally, Worker, spin_main

CONFIG = "/ws/config/camera.json"
RANGE_MIN_M = 0.10  # the LaserScan's floor; the contact line itself never comes nearer than 1.0 m
LEAN_EPSILON = 0.003  # how far the up vector may move before the floor's geometry is rebuilt
STAGES = ("plane", "scan", "publish")
RANGE_CEILING_M = 10.0  # the widest a drive may open max_range to while measuring a new cap

# The node's flags (CLAUDE.md rule 19), declared last in __init__ so the kit's callback sees no
# other declaration; their state is printed in every report line. Nothing is cached from them:
# every frame reads them, so a change takes the next one.
FLAGS = FlagSet(
    Flag(
        "contact_scan",
        True,
        description="the contact line is published; off, the node is a subscriber that costs"
        " nothing — the costmap's own contact_layer.enabled is the other end of the same demo"
        " switch, and either one alone takes the camera's floor line out",
        why="default by design, unmeasured as a switch: it is one end of a pair, so the line can"
        " be taken out of the map in one command from either side. The line's own accuracy is"
        " measured (see max_range), but the validation drive that was asked for — an open floor"
        " showing about 0 % marks and a taped box at 1.2, 1.6 and 2.0 m landing within 10 cm —"
        " has not been run",
        on_when="wherever the camera's floor line should be seen: the redundancy demo, or a low"
        " obstacle the lidar's plane looks over",
        off_when="to take the line out in one command, and on a run where this node's cost must"
        " be zero",
    ),
    Flag(
        "shadow",
        True,
        description="the last floor pixel on a face stands a band's width UP that face, so its"
        " ray lands past the foot: on, that width is taken back off the range"
        " (pepin.contact.band_shadow); off is the raw boundary ray",
        why="the uncorrected ray reports an obstacle about 10 % of its range too far — at 1.5 m"
        " the band is 0.120 m tall and the raw ray lands at 1.66 m, 16 cm of phantom clearance,"
        " in the direction a costmap pays for. That 10 % is geometry on this mount, not a field"
        " A/B: no run compares the line against the lidar with the correction off",
        on_when="wherever the line feeds a costmap: an obstacle reported too far is the failure a"
        " bumper pays for",
        off_when="to see the raw boundary ray, or on a mount whose band is thin enough that the"
        " correction is inside the noise",
    ),
    Flag(
        "imu_lean",
        True,
        description="the floor plane leans with the gyro as well as the accelerometer"
        " (pepin.lean: the lean of a wheel climbing a threshold is followed within a sample"
        " instead of being gated away as a push); off, the accelerometer alone, as it always has"
        " been",
        why="on: the gyro's sign was verified by hand on 2026-09-13 (the cart tipped nose-down"
        " read pitch +10.5 deg, left-side-down read roll -9.4 deg, the fast path following at"
        " once with quality 0.9 while held; scratch/lean_tip_test.txt), and the gyro's zero"
        " offset is learned. Off, the floor anchor keeps the accelerometer-only lean, which by"
        " design ignores any tip shorter than 10 s",
        on_when="after a hand tip through a known angle shows the reported lean following it the"
        " right way; the gain is the threshold case, where a real lean is followed within a"
        " sample",
        off_when="wherever the reported lean disagrees with the cart's visible attitude",
    ),
    Flag(
        "max_range",
        CONTACT_MAX_RANGE,
        description="metres past which a column is called clear instead of ended; the costmap's"
        " contact_layer.obstacle_max_range must match it",
        why="where the floor stops being the floor, not where the optics run out: on 22"
        " open-floor frames of run 0171 the network's floor sits at 1.01 of the geometric plane"
        " at 1.0-1.5 m, 0.96 at 1.5-2.0, 0.89 at 2.0-2.5 and 0.80 at 2.5-3.0 — and 0.89 of the"
        " plane is 13 cm of height, the width of the band itself, so past 2 m the floor leaves"
        " the band on its own and the column ends on nothing. The marks agree: 3 % false below"
        " 1.75 m, 44 % beyond it, and without the cap every mark past 2 m is false"
        " (scratch/contact_vs_lidar.py). Measured 2026-09-11 on the old geometry (lidar 0.20 m,"
        " camera 1.23 m at 26 deg, hfov 78), all three since corrected — the cap has not been"
        " re-measured",
        on_when="raise it only after the floor ratio is re-measured on the corrected geometry,"
        " and raise the costmap's obstacle_max_range with it",
        off_when="lower it where the floor is patterned, wet or dark, which shortens the range"
        " the network's floor stays flat over",
        range=(RANGE_MIN_M, RANGE_CEILING_M),
    ),
)


class ContactScan(Node):
    """Publishes the contact line of every depth frame as a LaserScan at floor height."""

    def __init__(self) -> None:
        super().__init__("contact_scan")
        config = Path(str(self.declare_parameter("config", CONFIG).value))
        cfg = CameraConfig.load(config)
        x, y, z, _roll, pitch, _yaw = mount_transform(cfg)
        self._camera = CameraPose(x, y, z, pitch)
        # Declared after every other parameter: rclpy runs the switches' callback on
        # declarations too, and it refuses everything that is not a flag.
        self._switches = Switches(self, FLAGS, on_change=self._on_switch)
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        newest = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(LaserScan, "/contact_scan", reliable)
        self._tally = Tally(STAGES)
        self._intr: Intrinsics | None = None
        self._lean = LeanFeed(
            self,
            config.parent,
            use_gyro=self._switches.on("imu_lean"),
            on_unmounted=self._no_imu_mount,
        )
        self._plane: FloorPlane | None = None
        self._plane_up: Array | None = None  # the lean it was built for
        self._plane_intr: Intrinsics | None = None  # and the optics
        self._verdict: ContactVerdict | None = None  # the last frame's, for the report line
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, reliable)
        self.create_subscription(Image, "/camera/depth", self._on_depth, newest)
        self._worker = Worker(self._process, name="contact", on_error=self._on_work_error).start()
        self.create_timer(30.0, self._report)
        self.get_logger().info(
            f"contact scan up: /camera/depth -> /contact_scan in base_link, camera at {z:.2f} m"
            f" and {np.degrees(pitch):.0f} deg down; the fan of /depth_scan,"
            f" {np.degrees(2 * SCAN_HALF_FOV):.0f} deg in steps of {np.degrees(SCAN_STEP):.1f}"
            f" deg; flags: {self._switches.state()}"
        )

    def close(self) -> None:
        """Stop the worker and wait for it, before the node is destroyed under it."""
        if not self._worker.stop():
            self.get_logger().warning("the contact worker did not finish its frame; leaving anyway")

    def _on_work_error(self, text: str) -> None:
        self.get_logger().error(f"the contact scan failed on a frame:\n{text}")

    # ---- inputs ------------------------------------------------------------------------------
    def _on_switch(self, name: str, _old: object, new: object) -> None:
        """A flag changed: ``imu_lean`` is the estimator's switch, the rest are only read."""
        if name == "imu_lean":
            self._lean.use_gyro = bool(new)

    def _no_imu_mount(self, frame_id: str) -> None:
        """IMU readings outside base_link with no mount to turn them: the floor stays level and
        a leaning cart reads its contacts long, which the report line says once a window."""
        self._tally.count("unleaned")
        self.get_logger().error(
            f"IMU readings in {frame_id} and no mount: the floor stays level",
            throttle_duration_sec=60,
        )

    def _on_info(self, msg: CameraInfo) -> None:
        self._intr = Intrinsics.from_camera_info(msg.k, msg.width, msg.height)

    def _on_depth(self, msg: Image) -> None:
        self._tally.count("depth_in")
        if self._worker.offer(msg):
            self._tally.count("dropped")

    # ---- the frame ---------------------------------------------------------------------------
    def _floor_plane(self, intr: Intrinsics) -> FloorPlane:
        """The floor's geometry for the current lean and optics, rebuilt only when one of them
        moves: per pixel where its ray lands on the plane, how far that is and in which bearing."""
        up = self._lean.up
        if (
            self._plane is None
            or self._plane_up is None
            or self._plane_intr != intr  # a camera_info of another size: every array is wrong
            or float(np.linalg.norm(up - self._plane_up)) > LEAN_EPSILON
        ):
            self._plane = FloorPlane.of(intr, self._camera, up)
            self._plane_up = np.asarray(up, dtype=float).copy()
            self._plane_intr = intr
            self._tally.count("planes")
        return self._plane

    def _process(self, msg: Image) -> None:
        """One depth frame through the floor's geometry and out as a scan, or counted and
        dropped: switched off, no optics yet, or a frame the intrinsics do not describe."""
        if not self._switches.on("contact_scan"):
            self._tally.count("off")
            return
        intr = self._intr
        if intr is None:
            self._tally.count("no_intrinsics")
            return
        if msg.encoding != "32FC1" or (msg.width, msg.height) != (intr.width, intr.height):
            self._tally.count("bad_frame")  # not the camera the camera_info describes
            return
        depth = array_from_image(msg)
        if depth is None:
            self._tally.count("bad_frame")
            return
        with self._tally.measure("plane"):
            plane = self._floor_plane(intr)
        max_range = float(self._switches["max_range"])
        with self._tally.measure("scan"):
            angle_min, step, ranges, verdict = contact_scan(
                depth, plane, max_range=max_range, shadow=self._switches.on("shadow")
            )
        with self._tally.measure("publish"):
            self._pub.publish(
                scan_from_ranges(
                    ranges,
                    float(angle_min),
                    float(step),
                    msg.header.stamp,
                    "base_link",
                    RANGE_MIN_M,
                    max_range,
                )
            )
        self._verdict = verdict  # one frozen dataclass, swapped for the report timer to read
        self._tally.count("frames")
        self._tally.count("marked", verdict.marked)
        if verdict.contact:
            self._tally.sample("range", verdict.median_range_m)

    # ---- the report --------------------------------------------------------------------------
    def _report(self) -> None:
        """The window's numbers, the last frame's verdict and the flags in one line."""
        w = self._tally.take()
        c = w.counts
        unseen = f"{c['no_intrinsics']} without optics, {c['bad_frame']} unreadable"
        verdict = self._verdict
        ranges = w.samples.get("range", [])
        self.get_logger().info(
            f"contact: {w.rate('frames'):.1f} scans/s published of {c['depth_in']} depth frames"
            f" ({c['dropped']} dropped, {c['off']} switched off, {unseen}),"
            f" {c['planes']} floor rebuilds"
            + f", {c['marked'] / max(c['frames'] * N_BINS, 1) * 100:.1f}% of the fan marked"
            + (f" (median contact {float(np.median(ranges)):.2f} m)" if ranges else "")
            + (f"; last frame: {verdict}" if verdict is not None else "; no frame yet")
            + f"; {self._lean.report()}; flags: {self._switches.state()};"
            + f" ms median/max: {w.stages()}"
        )
        if c["unleaned"]:
            self.get_logger().warning(
                f"{c['unleaned']} IMU readings could not be turned into base_link:"
                " the floor plane is level, and a leaning cart reads its contacts long"
            )
        if self._switches.on("contact_scan") and c["depth_in"] == 0:
            self.get_logger().warning(
                "no depth frame in this window: is /camera/depth alive (the depth law needs the"
                " lidar) and does the bridge route it?"
            )


def main() -> None:
    spin_main(ContactScan)


if __name__ == "__main__":
    main()
