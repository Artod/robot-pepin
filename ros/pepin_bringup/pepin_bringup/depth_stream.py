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
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan
from tf2_ros import Buffer, TransformListener

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
    decode_rgb,
    depth_to_scan,
    drop_edges,
    edge_mask,
    floor_anchor,
    floor_depth,
    imu_mount_rotation,
    load_law,
    nearest_stamp,
    project,
    rotation_matrix,
    save_law,
    scan_points,
    to_base,
)
from pepin.telemetry import LatencyTracker

CONFIG = "/ws/config/camera.json"
IMU_CONFIG = "/ws/config/imu.json"
LAW_FILE = "/maps/depth_law.json"  # ros/maps on the laptop, mounted at /maps by ros/laptop.sh
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SCAN_RANGE_M = 6.0
STAGES = ("network", "samples", "law", "edges", "floor", "scan", "publish")
LIVE_PARAMETERS = ("floor_anchor", "edge_filter", "lidar_anchor")  # the rest are read at start


def stamp_seconds(stamp: Any) -> float:
    """A ROS stamp as seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


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
        # The floor as a second anchor (pepin.depth.floor_depth): pixels within centimetres of
        # the floor plane snap to it in the published image (the scan is built before it); the
        # plane leans with the cart (the IMU's up vector).
        # Live switch: ros2 param set /depth_stream floor_anchor false
        self._floor_anchor = bool(self.declare_parameter("floor_anchor", True).value)
        # Flying pixels at object edges are dropped from the published depth and the scan
        # (pepin.depth.edge_mask); the law's beam pairs skip edge pixels regardless, the mask
        # being computed for them anyway.
        # Live switch: ros2 param set /depth_stream edge_filter false
        self._edge_filter = bool(self.declare_parameter("edge_filter", True).value)
        # The lidar fits the depth's law; off, the last law is held — the failure mode of a
        # lidar that stops, and a measure of what the lidar buys the depth. With no law yet
        # (nothing pooled, nothing saved) nothing is published until it is switched back on.
        self._lidar_anchor = bool(self.declare_parameter("lidar_anchor", True).value)
        self.add_on_set_parameters_callback(self._on_params)
        self._imu_mount = self._imu_mount_from(Path(IMU_CONFIG))
        self._tilt: Tilt | None = None
        self._floor: Array | None = None  # the expected floor depth image, for the current tilt
        self._floor_up: Array | None = None
        self._floor_intr: Intrinsics | None = None  # the optics it was computed for
        self.create_subscription(Imu, "/imu/data_raw", self._on_imu, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, reliable)
        self.create_subscription(Image, "/camera/image", self._on_image, newest)
        self.create_subscription(LaserScan, "/scan", self._on_scan, reliable)
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self, spin_thread=True)
        self._lidar_mount: tuple[Array, Array] | None = None
        self._scans: deque[LaserScan] = deque()  # the last SCAN_WINDOW_S of scans, by stamp
        self._scan_lock = threading.Lock()
        self._pending: Image | None = None
        self._wake = threading.Condition()
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
        self._stats = self._fresh_stats()
        self._timing = self._fresh_timing()
        self._scan_ages: list[float] = []  # |frame - anchoring scan| per verdict, this window
        self._since = time.monotonic()
        self.get_logger().info(f"loading {model_name} on CPU")
        self._net = MonoDepth(model_name, threads)
        threading.Thread(target=self._work, daemon=True).start()
        self.create_timer(30.0, self._report)
        self.get_logger().info("depth stream up: /camera/image -> /camera/depth, /depth_scan")

    @staticmethod
    def _fresh_stats() -> dict[str, int]:
        return dict.fromkeys(
            (
                "processed",
                "frames",
                "dropped",
                "withheld",
                "held",
                "verdicts",
                "samples",
                "floor_px",
                "uncarried",
                "edge_px",
            ),
            0,
        )

    @staticmethod
    def _fresh_timing() -> dict[str, LatencyTracker]:
        return {name: LatencyTracker(name) for name in STAGES}

    def _imu_mount_from(self, path: Path) -> Array | None:
        """The rotation from the chip's axes into base_link, from config/imu.json, read once;
        ``None`` (with one error) when the file is missing or broken. It is only needed for a
        reading published in a frame other than base_link: the C++ bridge (base_bridge.cpp,
        ``to_base_axes`` with ``imu_up_axis`` "y", ``imu_frame`` "base_link") already turns the
        accelerometer into base_link — the same rotation as this file's roll +90 deg — so for
        its readings the mount must not be applied a second time."""
        try:
            mount = json.loads(path.read_text())["mount"]
            rotation: Array = imu_mount_rotation(
                float(mount["roll_deg"]), float(mount["pitch_deg"]), float(mount["yaw_deg"])
            )
        except (OSError, KeyError, ValueError, TypeError) as exc:
            self.get_logger().error(
                f"no IMU mount in {path} ({exc}): a reading outside base_link cannot lean the floor"
            )
            return None
        return rotation

    def _on_params(self, params: list[Any]) -> SetParametersResult:
        """The live switches; every other parameter is read at start and a change is refused."""
        for p in params:
            if p.name not in LIVE_PARAMETERS:
                return SetParametersResult(
                    successful=False, reason=f"{p.name} is read at start: restart the node"
                )
        for p in params:
            setattr(self, f"_{p.name}", bool(p.value))
            self.get_logger().info(f"{p.name.replace('_', ' ')} {'on' if p.value else 'off'}")
        return SetParametersResult(successful=True)

    def _on_imu(self, msg: Imu) -> None:
        """The accelerometer says which way is up: a base_link reading (the C++ bridge's) as is,
        any other frame through the mount in config/imu.json."""
        if not self._floor_anchor:
            return  # the up vector has no other reader; the switch back on picks the tilt up again
        if self._tilt is None:
            if msg.header.frame_id == "base_link":
                rotation = np.eye(3)  # the bridge rotated it already: see _imu_mount_from
            elif self._imu_mount is not None:
                rotation = self._imu_mount
            else:
                # Without the mount the up vector would be the chip's own axes, and the floor
                # would be anchored to a plane tilted by however the chip is glued on.
                self._floor_anchor = False
                self.get_logger().error(
                    f"IMU readings in {msg.header.frame_id} and no mount: floor anchor off"
                )
                return
            self._tilt = Tilt(rotation)
        a = msg.linear_acceleration
        self._tilt.observe(np.array([a.x, a.y, a.z]), stamp_seconds(msg.header.stamp))

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
        """Keep the last SCAN_WINDOW_S of scans: the frame picks the one nearest its exposure."""
        newest = stamp_seconds(msg.header.stamp)
        with self._scan_lock:
            self._scans.append(msg)
            while newest - stamp_seconds(self._scans[0].header.stamp) > SCAN_WINDOW_S:
                self._scans.popleft()

    def _on_image(self, msg: Image) -> None:
        with self._wake:
            if self._pending is not None:
                self._stats["dropped"] += 1
            self._pending = msg
            self._wake.notify()

    def _work(self) -> None:
        while rclpy.ok():
            with self._wake:
                while self._pending is None:
                    self._wake.wait(1.0)
                    if not rclpy.ok():
                        return
                msg, self._pending = self._pending, None
            try:
                self._process(msg)
            except Exception:  # a raise must not kill the worker silently
                self.get_logger().error(f"depth failed on a frame:\n{traceback.format_exc()}")

    def _process(self, msg: Image) -> None:
        """One frame through the network, the law, the edges, the scan, the floor and out —
        or withheld before the scan when no law exists yet."""
        rgb = decode_rgb(bytes(msg.data), msg.height, msg.width, msg.encoding)
        if rgb is None:
            self.get_logger().warning(
                f"cannot read {msg.encoding} images", throttle_duration_sec=30
            )
            return
        s, timing = self._stats, self._timing
        with timing["network"].measure():
            depth = self._net(rgb)
        with timing["samples"].measure():  # includes the TF wait for the carry
            samples = self._lidar_samples(msg)
        t_edges = time.perf_counter()
        edge = edge_mask(depth)  # on the raw depth: relative, so the same after the law
        t_edges = time.perf_counter() - t_edges
        with timing["law"].measure():
            pairs = beam_pairs(depth, samples, edge) if samples is not None else None
            a_law, b_law = self._law.observe(pairs)
        s["processed"] += 1
        if pairs is None:
            s["held"] += 1
        else:
            s["verdicts"] += 1
            self._last_verdict_wall = time.time()
            s["samples"] += int(pairs[0].size)
        if not self._law.ready:
            s["withheld"] += 1
            self.get_logger().info(
                f"no depth law yet ({self._law.pooled} of {POOL_MIN_SAMPLES} beams pooled):"
                " nothing published",
                throttle_duration_sec=10,
            )
            return
        metric = apply_affine(depth, a_law, b_law)
        t0 = time.perf_counter()
        if self._edge_filter:
            metric, dropped = drop_edges(metric, edge)
            s["edge_px"] += dropped
        timing["edges"].add(t_edges + time.perf_counter() - t0)
        with timing["scan"].measure():  # before the floor anchor: what stops the cart is measured
            scan = self._as_scan(metric, msg)
        with timing["floor"].measure():
            if self._floor_anchor and self._intr is not None:
                metric, anchored = floor_anchor(
                    metric, self._floor_expected(self._intr), self._camera.z
                )
                s["floor_px"] += anchored
        with timing["publish"].measure():
            out = Image()
            out.header = msg.header
            out.height, out.width = msg.height, msg.width
            out.encoding = "32FC1"
            out.is_bigendian = 0
            out.step = msg.width * 4
            out.data = metric.astype(np.float32).tobytes()
            self._pub.publish(out)
            self._scan_pub.publish(scan)
        s["frames"] += 1

    def _as_scan(self, depth: Array, image: Image) -> LaserScan:
        """The scaled depth folded onto the floor plane, in base_link, stamped like the image."""
        angle_min, step, ranges = depth_to_scan(
            depth, self._intr_or_nominal(image), self._camera, max_range=self._scan_max_range
        )
        scan = LaserScan()
        scan.header.stamp = image.header.stamp
        scan.header.frame_id = "base_link"
        scan.angle_min, scan.angle_max = (
            float(angle_min),
            float(angle_min + step * (ranges.size - 1)),
        )
        scan.angle_increment = float(step)
        scan.range_min, scan.range_max = 0.1, float(self._scan_max_range)
        scan.ranges = [float(r) for r in ranges]
        return scan

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
        if intr is None or not self._lidar_anchor:
            return None
        with self._scan_lock:
            scans = list(self._scans)
        frame_t = stamp_seconds(image.header.stamp)
        i = nearest_stamp([stamp_seconds(s.header.stamp) for s in scans], frame_t)
        if i is None:
            return None
        scan = scans[i]
        self._scan_ages.append(abs(stamp_seconds(scan.header.stamp) - frame_t))
        mount = self._mount_of(scan.header.frame_id)
        if mount is None:
            return None
        rotation, translation = mount
        xy = scan_points(
            np.asarray(scan.ranges), scan.angle_min, scan.angle_increment, SCAN_RANGE_M
        )
        in_base = to_base(xy, rotation, translation)
        carried = self._carried_to_frame(in_base, scan.header.stamp, image.header.stamp)
        return project(carried, self._camera, intr)

    def _carried_to_frame(self, points: Array, scan_stamp: Any, frame_stamp: Any) -> Array:
        """The scan's points as base_link would see them at the frame's moment: the cart's own
        motion between the two stamps, from odometry. Without it a 100 ms older scan is 2
        degrees stale at 20 deg/s and the columns anchor to the wrong bearings."""
        try:
            t = self._tf.lookup_transform_full(
                "base_link",
                Time.from_msg(frame_stamp),
                "base_link",
                Time.from_msg(scan_stamp),
                "odom",
                timeout=Duration(seconds=0.2),
            )
        except Exception:
            self._stats["uncarried"] += 1
            return points
        q, v = t.transform.rotation, t.transform.translation
        return carry(points, rotation_matrix(q.x, q.y, q.z, q.w), np.array([v.x, v.y, v.z]))

    def _mount_of(self, frame: str) -> tuple[Array, Array] | None:
        if self._lidar_mount is None:
            try:
                tf = self._tf.lookup_transform("base_link", frame, Time())
            except Exception:
                self.get_logger().warning(
                    f"no base_link -> {frame} transform yet", throttle_duration_sec=30
                )
                return None
            q, t = tf.transform.rotation, tf.transform.translation
            self._lidar_mount = (rotation_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z]))
            self.get_logger().info(
                f"lidar mount from TF: {frame} at {t.x:.2f} {t.y:.2f} {t.z:.2f} m in base_link"
            )
        return self._lidar_mount

    def _report(self) -> None:
        """The window's numbers in one line, the law saved, the counters reset."""
        s, now = self._stats, time.monotonic()
        span = max(now - self._since, 1e-6)
        per_verdict = s["samples"] / max(s["verdicts"], 1)
        if self._lidar_anchor and s["verdicts"] == 0 and s["frames"]:
            self.get_logger().warning(
                "no lidar beam judged the depth in this window: the law is held"
                f" ({time.time() - self._last_verdict_wall:.0f} s old); is /scan alive?"
            )
        law = self._law
        source = "" if law.fitted else " (from file)" if law.ready else " (none yet)"
        extra = ""
        if self._scan_ages:
            extra += (
                f", scan age median {statistics.median(self._scan_ages):.2f} s"
                f" max {max(self._scan_ages):.2f} s"
            )
        if s["uncarried"]:
            extra += f", scans uncarried {s['uncarried']}"
        if self._intr is not None and s["frames"]:
            pixels = self._intr.width * self._intr.height * s["frames"]
            if self._floor_anchor:
                lean = self._tilt.roll_pitch_deg if self._tilt is not None else (0.0, 0.0)
                extra += (
                    f", floor {s['floor_px'] / pixels * 100:.0f}% of pixels"
                    f" (lean roll {lean[0]:+.1f} pitch {lean[1]:+.1f} deg)"
                )
            if self._edge_filter:
                extra += f", edges dropped {s['edge_px'] / pixels * 100:.0f}% of pixels"
        stages = " ".join(
            f"{name} {t.summary().median_ms:.0f}/{t.summary().max_ms:.0f}"
            for name, t in self._timing.items()
        )
        self.get_logger().info(
            f"depth: {s['frames'] / span:.1f} frames/s published ({s['processed']} through the"
            f" net, {s['dropped']} dropped, {s['withheld']} withheld), law a {law.a:.2f}"
            f" b {law.b:+.3f} on {law.pooled} beams{source} from {s['verdicts']} lidar verdicts"
            f" ({per_verdict:.0f} samples each; held {s['held']} of {s['processed']} frames)"
            f"{extra}, ms median/max: {stages}"
        )
        if law.fitted:
            try:
                save_law(self._law_file, law.a, law.b, law.pooled, self._last_verdict_wall)
            except OSError as exc:
                self.get_logger().warning(
                    f"cannot save the depth law to {self._law_file}: {exc}",
                    throttle_duration_sec=300,
                )
        self._stats = self._fresh_stats()
        self._timing = self._fresh_timing()
        self._scan_ages = []
        self._since = now


def main() -> None:
    rclpy.init()
    node = DepthStream()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
