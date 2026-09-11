"""The camera as a depth sensor, on the laptop: images in, metric depth images out.

Depth Anything V2 (metric, indoor, the small model) turns each camera frame into a depth image
with the right shape and the wrong size — and the wrong size is not one number: the far end of
a room comes out too far by more than the near end. The lidar fixes that (``pepin.depth``): the
scan taken nearest the frame's exposure, carried to the frame's moment through the odometry and
projected into the image through the two static mounts, names the true depth at the pixels it
hits, spanning a metre to four, and those pixels fit an affine law in inverse depth
(1 / z = a / D + b) pooled across minutes of frames and applied to the whole image. Nothing is
published until that law exists: the raw network's depth is 1.5-2x too far and would put the
costmap's obstacles where there are none, so frames are withheld until POOL_MIN_SAMPLES beam
pairs are pooled — or until the law saved by the last run (``/maps/depth_law.json``, a day old
at most) is loaded at start. Pixels on an object's edge are dropped, and the depth cut between
8 cm and 1.3 m above the floor and folded onto the plane goes out as ``/depth_scan`` (a
LaserScan in base_link) before the floor anchor touches anything: the board's local costmap
marks and clears with it like with the lidar, so a table top stops the cart the way a wall
does. The floor anchor (pixels within centimetres of the floor plane snap to it, the plane
leaning with the accelerometer) is for the 3D model: the anchored depth goes out on
``/camera/depth`` (32FC1 metres, the image's stamp and frame), where the fusion builds the model
from it and the costmap reads obstacles the lidar's plane misses. Frames that arrive while the
network is busy are dropped: the newest one wins. Every stage is timed and reported.

The flags (:data:`FLAGS`, ``ros2 param set /depth_stream <flag> <value>``): ``floor_anchor``,
``edge_filter``, ``lidar_anchor``; their state is printed in every report line.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan

from pepin.camera import CameraConfig, intrinsics, mount_transform
from pepin.depth import (
    POOL_MIN_SAMPLES,
    SCAN_WINDOW_S,
    AffineScale,
    Array,
    CameraPose,
    Intrinsics,
    Tilt,
    apply_affine,
    beam_pairs,
    carry,
    depth_to_scan,
    drop_edges,
    edge_mask,
    floor_anchor,
    floor_depth,
    load_law,
    nearest_stamp,
    project,
    save_law,
    scan_points,
    to_base,
)
from pepin.flags import Flag, FlagSet
from pepin.mounts import Mounts
from pepin.tsdf import RigidPose
from pepin_bringup.msgs import (
    array_from_image,
    image_from_array,
    imu_arrays,
    scan_from_ranges,
    stamp_seconds,
)
from pepin_bringup.node_kit import Switches, Tally, TfLookup, Window, Worker, spin_main

CONFIG = "/ws/config/camera.json"
LAW_FILE = "/maps/depth_law.json"  # ros/maps on the laptop, mounted at /maps by ros/laptop.sh
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SCAN_RANGE_M = 6.0
STAGES = ("network", "samples", "law", "edges", "floor", "scan", "publish")
CARRY_WAIT_S = 0.2  # how long the carry waits for odometry to cover the scan-to-frame gap

# The live flags (CLAUDE.md rule 19), declared last in __init__ so the kit's callback sees no
# other declaration, and printed in every report line.
FLAGS = FlagSet(
    Flag(
        "floor_anchor",
        True,
        description="pixels within centimetres of the floor plane snap to it in the published"
        " image (the scan is built before it); the plane leans with the cart, from the IMU's up"
        " vector",
    ),
    Flag(
        "edge_filter",
        True,
        description="flying pixels at object edges are dropped from the published depth and the"
        " scan; the law's beam pairs skip them regardless",
    ),
    Flag(
        "lidar_anchor",
        True,
        description="the lidar fits the depth's law; off, the last law is held (the failure mode"
        " of a lidar that stops) — with no law yet nothing is published until it is back on",
    ),
)


class MonoDepth:
    """Depth Anything V2 behind one call: an RGB array in, a float32 depth image of the same
    size out, in the network's own (approximate) metres."""

    def __init__(self, model_name: str, threads: int) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._torch = torch
        torch.set_num_threads(threads)
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModelForDepthEstimation.from_pretrained(model_name).eval()

    def __call__(self, rgb: npt.NDArray[np.uint8]) -> Array:
        torch = self._torch
        with torch.no_grad():
            inputs = self._processor(images=rgb, return_tensors="pt")
            predicted = self._model(**inputs).predicted_depth.unsqueeze(1)
            full = torch.nn.functional.interpolate(
                predicted, size=rgb.shape[:2], mode="bilinear", align_corners=False
            )
        depth: Array = full[0, 0].numpy().astype(np.float32)
        return depth


class DepthStream(Node):
    """Publishes a lidar-scaled depth image and its planar scan for every camera frame the
    network can keep up with, once a depth law exists."""

    def __init__(self) -> None:
        super().__init__("depth_stream")
        board = str(self.declare_parameter("board", "127.0.0.1").value)
        config = Path(str(self.declare_parameter("config", CONFIG).value))
        model_name = str(
            self.declare_parameter(
                "model", os.environ.get("PEPIN_DEPTH_MODEL", DEFAULT_MODEL)
            ).value
        )
        # Eight threads: 0.3 s a frame at 18 threads took nine cores of the laptop (876 % CPU);
        # the map adds a node a second, so a slower frame costs nothing the map would notice.
        threads = int(self.declare_parameter("threads", 8).value)
        cfg = CameraConfig.load(config, board=board)
        x, y, z, _roll, pitch, _yaw = mount_transform(cfg)
        self._camera = CameraPose(x, y, z, pitch)
        self._hfov_deg = cfg.hfov_deg  # the nominal optics until a camera_info arrives
        self._intr: Intrinsics | None = None
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        newest = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(Image, "/camera/depth", reliable)
        self._scan_pub = self.create_publisher(LaserScan, "/depth_scan", reliable)
        self._scan_max_range = float(self.declare_parameter("scan_max_range", 3.0).value)
        self._law_file = Path(str(self.declare_parameter("law_file", LAW_FILE).value))
        self._switches = Switches(self, FLAGS)
        self._imu_mount = self._imu_rotation(config.parent)
        self._tilt: Tilt | None = None
        self._floor: Array | None = None  # the expected floor depth image, for the current tilt
        self._floor_up: Array | None = None
        self._floor_intr: Intrinsics | None = None  # the optics it was computed for
        self._tally = Tally(STAGES)
        self.create_subscription(Imu, "/imu/data_raw", self._on_imu, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, reliable)
        self.create_subscription(Image, "/camera/image", self._on_image, newest)
        self.create_subscription(LaserScan, "/scan", self._on_scan, reliable)
        self._tf = TfLookup(self, on_failure=self._on_tf_failure)
        self._lidar_mount: RigidPose | None = None
        self._scans: deque[LaserScan] = deque()  # the last SCAN_WINDOW_S of scans, by stamp
        self._scan_lock = threading.Lock()
        self._law = AffineScale()
        self._last_verdict_wall = time.time()  # the law's age is the beams', not the node's
        saved = load_law(self._law_file, time.time())
        if saved is not None:
            self._law.seed(saved[0], saved[1])
            self.get_logger().info(
                f"depth law from {self._law_file}: a {saved[0]:.2f} b {saved[1]:+.3f}"
                f" on {saved[2]} beams; publishing at once"
            )
        else:
            self.get_logger().info(
                f"no saved depth law at {self._law_file}: publishing waits for"
                f" {POOL_MIN_SAMPLES} pooled beams"
            )
        self.get_logger().info(f"loading {model_name} on CPU")
        self._net = MonoDepth(model_name, threads)
        self._worker = Worker(self._process, name="depth", on_error=self._on_work_error).start()
        self.create_timer(30.0, self._report)
        self.get_logger().info("depth stream up: /camera/image -> /camera/depth, /depth_scan")

    def close(self) -> None:
        """Stop the worker and the TF listener and wait for them: called before the node is
        destroyed, so no thread is left inside the network or DDS at interpreter exit."""
        if not self._worker.stop():
            self.get_logger().warning("the depth worker did not finish its frame; leaving anyway")
        self._tf.close()

    def _on_work_error(self, text: str) -> None:
        self.get_logger().error(f"depth failed on a frame:\n{text}")

    def _on_tf_failure(self, kind: str, text: str) -> None:
        self._tally.count("tf_" + kind)
        self._tally.note(kind, text)

    def _imu_rotation(self, config_dir: Path) -> Array | None:
        """The rotation from the chip's axes into base_link (``config/imu.json`` through
        :class:`pepin.mounts.Mounts`), read once; ``None`` (with one error) when the files are
        missing or broken. It is only needed for a reading published in a frame other than
        base_link: the C++ bridge (base_bridge.cpp, ``to_base_axes`` with ``imu_up_axis`` "y",
        ``imu_frame`` "base_link") already turns the accelerometer into base_link — the same
        rotation as this file's roll +90 deg — so for its readings the mount must not be applied
        a second time."""
        try:
            rotation: Array = Mounts.load(config_dir).imu.rotation()
        except (OSError, KeyError, ValueError, TypeError) as exc:
            self.get_logger().error(
                f"no IMU mount in {config_dir} ({exc}): a reading outside base_link"
                " cannot lean the floor"
            )
            return None
        return rotation

    def _on_imu(self, msg: Imu) -> None:
        """The accelerometer says which way is up: a base_link reading (the C++ bridge's) as is,
        any other frame through the mount in config/imu.json."""
        if not self._switches.on("floor_anchor"):
            return  # the up vector has no other reader; the switch back on picks the tilt up again
        if self._tilt is None:
            if msg.header.frame_id == "base_link":
                rotation = np.eye(3)  # the bridge rotated it already: see _imu_rotation
            elif self._imu_mount is not None:
                rotation = self._imu_mount
            else:
                # Without the mount the up vector would be the chip's own axes, and the floor
                # would be anchored to a plane tilted by however the chip is glued on.
                self._switches.set("floor_anchor", False)
                self.get_logger().error(
                    f"IMU readings in {msg.header.frame_id} and no mount: floor anchor off"
                )
                return
            self._tilt = Tilt(rotation)
        accel, _gyro = imu_arrays(msg)
        self._tilt.observe(accel, stamp_seconds(msg.header.stamp))

    def _floor_expected(self, intr: Intrinsics) -> Array:
        """The floor's depth image for the current lean, recomputed only when the lean moves."""
        up = self._tilt.up if self._tilt is not None else np.array([0.0, 0.0, 1.0])
        if (
            self._floor is None
            or self._floor_up is None
            or self._floor_intr != intr  # a camera_info of another size: the cache is a wrong shape
            or np.linalg.norm(up - self._floor_up) > 0.003
        ):
            self._floor = floor_depth(intr, self._camera, up)
            self._floor_up = up.copy()
            self._floor_intr = intr
        return self._floor

    def _on_info(self, msg: CameraInfo) -> None:
        self._intr = Intrinsics.from_camera_info(msg.k, msg.width, msg.height)

    def _on_scan(self, msg: LaserScan) -> None:
        """Keep the last SCAN_WINDOW_S of scans: the frame picks the one nearest its exposure.

        Under the lock: the worker copies the deque on its own thread, and a popleft in the
        middle of that copy raises "deque mutated during iteration".
        """
        newest = stamp_seconds(msg.header.stamp)
        with self._scan_lock:
            self._scans.append(msg)
            while newest - stamp_seconds(self._scans[0].header.stamp) > SCAN_WINDOW_S:
                self._scans.popleft()

    def _on_image(self, msg: Image) -> None:
        if self._worker.offer(msg):
            self._tally.count("dropped")

    def _process(self, msg: Image) -> None:
        """One frame through the network, the law, the edges, the scan, the floor and out —
        or withheld before the scan when no law exists yet."""
        rgb = array_from_image(msg)
        if rgb is None or rgb.ndim != 3:
            self.get_logger().warning(
                f"cannot read {msg.encoding} images", throttle_duration_sec=30
            )
            return
        tally = self._tally
        with tally.measure("network"):
            depth = self._net(rgb)
        with tally.measure("samples"):  # includes the TF wait for the carry
            samples = self._lidar_samples(msg)
        t_edges = time.perf_counter()
        edge = edge_mask(depth)  # on the raw depth: relative, so the same after the law
        t_edges = time.perf_counter() - t_edges
        with tally.measure("law"):
            pairs = beam_pairs(depth, samples, edge) if samples is not None else None
            a_law, b_law = self._law.observe(pairs)
        tally.count("processed")
        if pairs is None:
            tally.count("held")
        else:
            tally.count("verdicts")
            self._last_verdict_wall = time.time()
            tally.count("samples", int(pairs[0].size))
        if not self._law.ready:
            tally.count("withheld")
            self.get_logger().info(
                f"no depth law yet ({self._law.pooled} of {POOL_MIN_SAMPLES} beams pooled):"
                " nothing published",
                throttle_duration_sec=10,
            )
            return
        metric = apply_affine(depth, a_law, b_law)
        t0 = time.perf_counter()
        if self._switches.on("edge_filter"):
            metric, dropped = drop_edges(metric, edge)
            tally.count("edge_px", dropped)
        tally.spent("edges", t_edges + time.perf_counter() - t0)
        with tally.measure("scan"):  # before the floor anchor: what stops the cart is measured
            scan = self._as_scan(metric, msg)
        with tally.measure("floor"):
            if self._switches.on("floor_anchor") and self._intr is not None:
                metric, anchored = floor_anchor(
                    metric, self._floor_expected(self._intr), self._camera.z
                )
                tally.count("floor_px", anchored)
        with tally.measure("publish"):
            self._pub.publish(
                image_from_array(metric, "32FC1", msg.header.stamp, msg.header.frame_id)
            )
            self._scan_pub.publish(scan)
        tally.count("frames")

    def _as_scan(self, depth: Array, image: Image) -> LaserScan:
        """The scaled depth folded onto the floor plane, in base_link, stamped like the image."""
        angle_min, step, ranges = depth_to_scan(
            depth, self._intr_or_nominal(image), self._camera, max_range=self._scan_max_range
        )
        return scan_from_ranges(
            ranges,
            float(angle_min),
            float(step),
            image.header.stamp,
            "base_link",
            0.1,
            float(self._scan_max_range),
        )

    def _intr_or_nominal(self, image: Image) -> Intrinsics:
        """The camera_info's optics, or the nominal pinhole of config/camera.json's field of
        view until one arrives."""
        if self._intr is not None:
            return self._intr
        fx, fy, cx, cy = intrinsics(image.width, image.height, self._hfov_deg)
        return Intrinsics(fx, fy, cx, cy, image.width, image.height)

    def _lidar_samples(self, image: Image) -> Array | None:
        """The beams of the scan nearest the frame's exposure as pixels of this frame with their
        true depth (column, row, metres): needs a scan within SCAN_MAX_AGE_S and both mounts."""
        intr = self._intr
        if intr is None or not self._switches.on("lidar_anchor"):
            return None
        with self._scan_lock:
            scans = list(self._scans)
        frame_t = stamp_seconds(image.header.stamp)
        i = nearest_stamp([stamp_seconds(s.header.stamp) for s in scans], frame_t)
        if i is None:
            return None
        scan = scans[i]
        self._tally.sample("scan_age", abs(stamp_seconds(scan.header.stamp) - frame_t))
        mount = self._mount_of(scan.header.frame_id)
        if mount is None:
            return None
        xy = scan_points(
            np.asarray(scan.ranges), scan.angle_min, scan.angle_increment, SCAN_RANGE_M
        )
        in_base = to_base(xy, mount.rotation, mount.translation)
        carried = self._carried_to_frame(in_base, scan.header.stamp, image.header.stamp)
        return project(carried, self._camera, intr)

    def _carried_to_frame(self, points: Array, scan_stamp: Any, frame_stamp: Any) -> Array:
        """The scan's points as base_link would see them at the frame's moment: the cart's own
        motion between the two stamps, from odometry. Without it a 100 ms older scan is 2
        degrees stale at 20 deg/s and the columns anchor to the wrong bearings."""
        motion = self._tf.motion(
            "base_link", scan_stamp, frame_stamp, "odom", timeout_s=CARRY_WAIT_S
        )
        if motion is None:
            self._tally.count("uncarried")
            return points
        return carry(points, motion.rotation, motion.translation)

    def _mount_of(self, frame: str) -> RigidPose | None:
        """base_link <- the laser's frame, looked up once and kept: where the beams start."""
        if self._lidar_mount is None:
            pose = self._tf.pose("base_link", frame)
            if pose is None:
                self.get_logger().warning(
                    f"no base_link -> {frame} transform yet", throttle_duration_sec=30
                )
                return None
            self._lidar_mount = pose
            t = pose.translation
            self.get_logger().info(
                f"lidar mount from TF: {frame} at {t[0]:.2f} {t[1]:.2f} {t[2]:.2f} m in base_link"
            )
        return self._lidar_mount

    def _report(self) -> None:
        """The window's numbers in one line, the law saved, the counters reset."""
        w = self._tally.take()
        c = w.counts
        per_verdict = c["samples"] / max(c["verdicts"], 1)
        if self._switches.on("lidar_anchor") and c["verdicts"] == 0 and c["frames"]:
            self.get_logger().warning(
                "no lidar beam judged the depth in this window: the law is held"
                f" ({time.time() - self._last_verdict_wall:.0f} s old); is /scan alive?"
            )
        law = self._law
        source = "" if law.fitted else " (from file)" if law.ready else " (none yet)"
        self.get_logger().info(
            f"depth: {w.rate('frames'):.1f} frames/s published ({c['processed']} through the"
            f" net, {c['dropped']} dropped, {c['withheld']} withheld), law a {law.a:.2f}"
            f" b {law.b:+.3f} on {law.pooled} beams{source} from {c['verdicts']} lidar verdicts"
            f" ({per_verdict:.0f} samples each; held {c['held']} of {c['processed']} frames)"
            f"{self._extras(w)}, flags: {self._switches.state()},"
            f" ms median/max: {w.stages()}"
        )
        if law.fitted:
            try:
                save_law(self._law_file, law.a, law.b, law.pooled, self._last_verdict_wall)
            except OSError as exc:
                self.get_logger().warning(
                    f"cannot save the depth law to {self._law_file}: {exc}",
                    throttle_duration_sec=300,
                )

    def _extras(self, w: Window) -> str:
        """The parts of the report line a window may have nothing to say about: how old the
        anchoring scans were, the scans no odometry could carry, the pixels each anchor
        touched, and the last TF failure of each kind."""
        c, extra = w.counts, ""
        ages = w.samples.get("scan_age", [])
        if ages:
            extra += f", scan age median {float(np.median(ages)):.2f} s max {max(ages):.2f} s"
        if c["uncarried"]:
            extra += f", scans uncarried {c['uncarried']}"
        if self._intr is not None and c["frames"]:
            pixels = self._intr.width * self._intr.height * c["frames"]
            if self._switches.on("floor_anchor"):
                lean = self._tilt.roll_pitch_deg if self._tilt is not None else (0.0, 0.0)
                extra += (
                    f", floor {c['floor_px'] / pixels * 100:.0f}% of pixels"
                    f" (lean roll {lean[0]:+.1f} pitch {lean[1]:+.1f} deg)"
                )
            if self._switches.on("edge_filter"):
                extra += f", edges dropped {c['edge_px'] / pixels * 100:.0f}% of pixels"
        if w.notes:
            extra += ", tf: " + "; ".join(
                f"{kind} {c['tf_' + kind]}: {text}" for kind, text in w.notes.items()
            )
        return extra


def main() -> None:
    spin_main(DepthStream)


if __name__ == "__main__":
    main()
