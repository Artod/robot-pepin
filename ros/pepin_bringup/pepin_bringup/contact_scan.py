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

Switches, live (``ros2 param set /contact_scan <name> <value>``): ``contact_scan`` (publish or
not), ``shadow`` (take the band's own width back off the range) and ``max_range``; their state and
the last frame's :class:`pepin.contact.ContactVerdict` are printed in every report line.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan

from pepin.camera import CameraConfig, mount_transform
from pepin.contact import (
    CONTACT_MAX_RANGE,
    N_BINS,
    ContactVerdict,
    FloorPlane,
    contact_scan,
)
from pepin.depth import SCAN_HALF_FOV, SCAN_STEP, UP_LEVEL, Array, CameraPose, Intrinsics, Tilt
from pepin.mounts import Mounts
from pepin_bringup.msgs import array_from_image, imu_arrays, scan_from_ranges, stamp_seconds
from pepin_bringup.node_kit import Switches, Tally, Worker, spin_main

CONFIG = "/ws/config/camera.json"
RANGE_MIN_M = 0.10  # the LaserScan's floor; the contact line itself never comes nearer than 1.0 m
LEAN_EPSILON = 0.003  # how far the up vector may move before the floor's geometry is rebuilt
STAGES = ("plane", "scan", "publish")


class ContactScan(Node):
    """Publishes the contact line of every depth frame as a LaserScan at floor height."""

    def __init__(self) -> None:
        super().__init__("contact_scan")
        config = Path(str(self.declare_parameter("config", CONFIG).value))
        cfg = CameraConfig.load(config)
        x, y, z, _roll, pitch, _yaw = mount_transform(cfg)
        self._camera = CameraPose(x, y, z, pitch)
        # The three live switches (CLAUDE.md rule 19), declared last so the kit's callback sees no
        # other declaration, and printed in every report line:
        #   contact_scan  publish or not. Off, the node is a subscriber that costs nothing: the
        #                 costmap's own `contact_layer.enabled` is the other end of the same
        #                 demo switch, and either one alone takes the camera's floor line out.
        #   shadow        the last floor pixel on a face stands a band's width UP that face, so
        #                 its ray lands past the foot; on (measured), the width is taken back off
        #                 (pepin.contact.band_shadow). Off is the raw boundary ray.
        #   max_range     metres past which a column is called clear instead of ended. The
        #                 default is where the floor is still the floor, not where the optics run
        #                 out; the costmap's contact_layer.obstacle_max_range must match it.
        self._switches = Switches(
            self,
            {"contact_scan": True, "shadow": True, "max_range": CONTACT_MAX_RANGE},
        )
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        newest = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(LaserScan, "/contact_scan", reliable)
        self._tally = Tally(STAGES)
        self._intr: Intrinsics | None = None
        self._imu_mount = self._imu_rotation(config.parent)
        self._tilt: Tilt | None = None
        self._plane: FloorPlane | None = None
        self._plane_up: Array | None = None  # the lean it was built for
        self._plane_intr: Intrinsics | None = None  # and the optics
        self._verdict: ContactVerdict | None = None  # the last frame's, for the report line
        self.create_subscription(Imu, "/imu/data_raw", self._on_imu, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, reliable)
        self.create_subscription(Image, "/camera/depth", self._on_depth, newest)
        self._worker = Worker(self._process, name="contact", on_error=self._on_work_error).start()
        self.create_timer(30.0, self._report)
        self.get_logger().info(
            f"contact scan up: /camera/depth -> /contact_scan in base_link, camera at {z:.2f} m"
            f" and {np.degrees(pitch):.0f} deg down; the fan of /depth_scan,"
            f" {np.degrees(2 * SCAN_HALF_FOV):.0f} deg in steps of {np.degrees(SCAN_STEP):.1f}"
            f" deg; switches: {self._switches.state()}"
        )

    def close(self) -> None:
        """Stop the worker and wait for it, before the node is destroyed under it."""
        if not self._worker.stop():
            self.get_logger().warning("the contact worker did not finish its frame; leaving anyway")

    def _on_work_error(self, text: str) -> None:
        self.get_logger().error(f"the contact scan failed on a frame:\n{text}")

    # ---- inputs ------------------------------------------------------------------------------
    def _imu_rotation(self, config_dir: Path) -> Array | None:
        """The rotation from the chip's axes into base_link (``config/imu.json`` through
        :class:`pepin.mounts.Mounts`), read once; ``None`` (with one error) when the files are
        missing or broken. A reading already published in base_link — the C++ bridge's, which has
        applied this very rotation — must not be turned a second time."""
        try:
            rotation: Array = Mounts.load(config_dir).imu.rotation()
        except (OSError, KeyError, ValueError, TypeError) as exc:
            self.get_logger().error(
                f"no IMU mount in {config_dir} ({exc}): a reading outside base_link cannot"
                " lean the floor, and the contact line will be read off a level plane"
            )
            return None
        return rotation

    def _on_imu(self, msg: Imu) -> None:
        """The accelerometer says which way is up: a base_link reading (the C++ bridge's) as is,
        any other frame through the mount in config/imu.json."""
        if self._tilt is None:
            if msg.header.frame_id == "base_link":
                rotation = np.eye(3)  # the bridge rotated it already: see _imu_rotation
            elif self._imu_mount is not None:
                rotation = self._imu_mount
            else:
                self._tally.count("unleaned")
                self.get_logger().error(
                    f"IMU readings in {msg.header.frame_id} and no mount: the floor stays level",
                    throttle_duration_sec=60,
                )
                return
            self._tilt = Tilt(rotation)
        accel, _gyro = imu_arrays(msg)
        self._tilt.observe(accel, stamp_seconds(msg.header.stamp))

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
        up = self._tilt.up if self._tilt is not None else UP_LEVEL
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
        """The window's numbers, the last frame's verdict and the switches in one line."""
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
            + f"; switches: {self._switches.state()}; ms median/max: {w.stages()}"
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
