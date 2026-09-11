"""The camera as a depth sensor, on the laptop: images in, metric depth images out.

Depth Anything V2 (metric, indoor, the small model) turns each camera frame into a depth image
with the right shape and the wrong size — and the wrong size is not one number: the far end of
a room comes out too far by more than the near end. The lidar fixes that (``pepin.depth``): its
scan, carried to the frame's moment through the odometry and projected into the image through
the two static mounts, names the true depth at the pixels it hits, spanning a metre to four,
and those pixels fit an affine law in inverse depth (1 / z = a / D + b) that is steadied across
frames and applied to the whole image. The floor is a second anchor for its own pixels; pixels
on an object's edge are dropped. The metric depth goes out on ``/camera/depth`` (32FC1 metres,
the image's stamp and frame), where the fusion builds the 3D model from it and the costmap
reads obstacles the lidar's plane misses. The same depth, cut between 8 cm and 1.3 m above the
floor and folded onto the plane, goes out as ``/depth_scan`` (a LaserScan in base_link): the
board's local costmap marks and clears with it like with the lidar, so a table top stops the
cart the way a wall does. Frames that arrive while the network is busy are dropped: the newest
one wins.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
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

from pepin.camera import CameraConfig, mount_transform
from pepin.depth import (
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
    floor_anchor,
    floor_depth,
    imu_mount_rotation,
    project,
    rotation_matrix,
    scan_points,
    to_base,
)

CONFIG = "/ws/config/camera.json"
IMU_CONFIG = "/ws/config/imu.json"
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SCAN_MAX_AGE_S = 0.5  # a scan older than this against the frame does not judge its scale
SCAN_RANGE_M = 6.0


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
    """Publishes a lidar-scaled depth image for every camera frame the network can keep up with."""

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
        self._intr: Intrinsics | None = None
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        newest = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(Image, "/camera/depth", reliable)
        self._scan_pub = self.create_publisher(LaserScan, "/depth_scan", reliable)
        self._scan_max_range = float(self.declare_parameter("scan_max_range", 3.0).value)
        # The floor as a second anchor (pepin.depth.floor_depth): pixels whose depth agrees with
        # the floor plane snap to it; the plane leans with the cart (the IMU's up vector).
        # Live switch: ros2 param set /depth_stream floor_anchor false
        self._floor_anchor = bool(self.declare_parameter("floor_anchor", True).value)
        # Flying pixels at object edges are dropped (pepin.depth.drop_edges); live switch:
        # ros2 param set /depth_stream edge_filter false
        self._edge_filter = bool(self.declare_parameter("edge_filter", True).value)
        # The lidar sets the depth's scale column by column; off, the last scale is held — the
        # failure mode of a lidar that stops, and a measure of what the lidar buys the depth.
        self._lidar_anchor = bool(self.declare_parameter("lidar_anchor", True).value)
        self.add_on_set_parameters_callback(self._on_params)
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
        self._scan: LaserScan | None = None
        self._pending: Image | None = None
        self._wake = threading.Condition()
        self._law = AffineScale()
        self._stats = {
            "frames": 0,
            "dropped": 0,
            "verdicts": 0,
            "samples": 0,
            "model_s": 0.0,
            "floor_px": 0,
            "uncarried": 0,
            "edge_px": 0,
        }
        self._since = time.monotonic()
        self.get_logger().info(f"loading {model_name} on CPU")
        self._net = MonoDepth(model_name, threads)
        threading.Thread(target=self._work, daemon=True).start()
        self.create_timer(30.0, self._report)
        self.get_logger().info("depth stream up: /camera/image -> /camera/depth")

    def _on_params(self, params: list[Any]) -> SetParametersResult:
        for p in params:
            if p.name == "floor_anchor":
                self._floor_anchor = bool(p.value)
                self.get_logger().info(f"floor anchor {'on' if self._floor_anchor else 'off'}")
            elif p.name == "edge_filter":
                self._edge_filter = bool(p.value)
                self.get_logger().info(f"edge filter {'on' if self._edge_filter else 'off'}")
            elif p.name == "lidar_anchor":
                self._lidar_anchor = bool(p.value)
                self.get_logger().info(f"lidar anchor {'on' if self._lidar_anchor else 'off'}")
        return SetParametersResult(successful=True)

    def _on_imu(self, msg: Imu) -> None:
        """The accelerometer says which way is up; the board's bridge already publishes the
        reading in base_link, any other frame goes through the mount in config/imu.json."""
        if not self._floor_anchor:
            return  # the up vector has no other reader; the switch back on picks the tilt up again
        if self._tilt is None:
            rotation = np.eye(3)
            if msg.header.frame_id != "base_link":
                try:
                    with open(IMU_CONFIG) as f:
                        mount = json.load(f)["mount"]
                except (OSError, KeyError, ValueError) as exc:
                    # Without the mount the up vector would be the chip's own axes, and the floor
                    # would be anchored to a plane tilted by however the chip is glued on.
                    self._floor_anchor = False
                    self.get_logger().error(f"no IMU mount ({exc}): floor anchor off")
                    return
                rotation = imu_mount_rotation(
                    mount["roll_deg"], mount["pitch_deg"], mount["yaw_deg"]
                )
            self._tilt = Tilt(rotation)
        a = msg.linear_acceleration
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._tilt.observe(np.array([a.x, a.y, a.z]), stamp)

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
        self._scan = msg

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
        rgb = decode_rgb(bytes(msg.data), msg.height, msg.width, msg.encoding)
        if rgb is None:
            self.get_logger().warning(
                f"cannot read {msg.encoding} images", throttle_duration_sec=30
            )
            return
        t0 = time.monotonic()
        depth = self._net(rgb)
        self._stats["model_s"] += time.monotonic() - t0
        samples = self._lidar_samples(msg)
        pairs = beam_pairs(depth, samples) if samples is not None else None
        verdict: tuple[float, int] | None = None
        a_law, b_law = self._law.observe(pairs)
        if pairs is not None:
            verdict = (1.0 / a_law, int(pairs[0].size))
        if verdict is not None:
            self._stats["verdicts"] += 1
            self._stats["samples"] += verdict[1]
        metric = apply_affine(depth, a_law, b_law)
        if self._edge_filter:
            metric, dropped = drop_edges(metric)
            self._stats["edge_px"] += dropped
        if self._floor_anchor and self._intr is not None:
            metric, anchored = floor_anchor(metric, self._floor_expected(self._intr))
            self._stats["floor_px"] += anchored
        out = Image()
        out.header = msg.header
        out.height, out.width = msg.height, msg.width
        out.encoding = "32FC1"
        out.is_bigendian = 0
        out.step = msg.width * 4
        out.data = metric.astype(np.float32).tobytes()
        self._pub.publish(out)
        self._scan_pub.publish(self._as_scan(metric, msg))
        self._stats["frames"] += 1

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
        if self._intr is not None:
            return self._intr
        fx = image.width / (2 * 0.7)  # a 70-degree lens until the camera_info arrives
        return Intrinsics(fx, fx, image.width / 2, image.height / 2, image.width, image.height)

    def _lidar_samples(self, image: Image) -> Array | None:
        """The newest scan's beams as pixels of this frame with their true depth (column, row,
        metres): needs a scan fresh against the frame and both mounts."""
        scan, intr = self._scan, self._intr
        if scan is None or intr is None or not self._lidar_anchor:
            return None
        age = abs(
            Time.from_msg(image.header.stamp).nanoseconds
            - Time.from_msg(scan.header.stamp).nanoseconds
        )
        if age > SCAN_MAX_AGE_S * 1e9:
            return None
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
        s, now = self._stats, time.monotonic()
        span = max(now - self._since, 1e-6)
        frames = max(s["frames"], 1)
        per_frame = s["model_s"] / frames
        per_verdict = s["samples"] / max(s["verdicts"], 1)
        floor = ""
        if self._floor_anchor and self._intr is not None:
            pixels = max(self._intr.width * self._intr.height * max(s["frames"], 1), 1)
            lean = self._tilt.roll_pitch_deg if self._tilt is not None else (0.0, 0.0)
            floor = (
                f", floor {s['floor_px'] / pixels * 100:.0f}% of pixels"
                f" (lean roll {lean[0]:+.1f} pitch {lean[1]:+.1f} deg)"
            )
        floor += f", scans uncarried {s['uncarried']}" if s["uncarried"] else ""
        if self._edge_filter and self._intr is not None:
            pixels = max(self._intr.width * self._intr.height * max(s["frames"], 1), 1)
            floor += f", edges dropped {s['edge_px'] / pixels * 100:.0f}% of pixels"
        self.get_logger().info(
            f"depth: {s['frames'] / span:.1f} frames/s ({s['dropped']} dropped),"
            f" model {per_frame:.2f} s/frame, law a {self._law.a:.2f} b {self._law.b:+.3f}"
            f" on {self._law.pooled} beams"
            f" from {s['verdicts']} lidar verdicts ({per_verdict:.0f} samples each;"
            f" held {self._law.held} frames){floor}"
        )
        self._stats = dict.fromkeys(s, 0)
        self._stats["model_s"] = 0.0
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
