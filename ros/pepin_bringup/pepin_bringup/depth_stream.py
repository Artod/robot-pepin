"""The camera as a depth sensor, on the laptop: images in, metric depth images out.

Depth Anything V2 (metric, indoor, the small model) turns each camera frame into a depth image
with the right shape and the wrong size: on our lens it overestimates distances about twofold.
The lidar fixes that (``pepin.depth``): its scan, projected into the image through the two
static mounts, names the true depth at the pixels it hits, and the median ratio scales the
frame. The scaled depth goes out on ``/camera/depth`` (32FC1 metres, the image's stamp and
frame), where RTAB-Map builds a 3D map from it and later the costmap reads obstacles the lidar's
plane misses. The same depth, cut between 8 cm and 1.3 m above the floor and folded onto the plane,
goes out as ``/depth_scan`` (a LaserScan in base_link): the board's local costmap marks and
clears with it like with the lidar, so a table top stops the cart the way a wall does. Frames
that arrive while the network is busy are dropped: the newest one wins.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tf2_ros import Buffer, TransformListener

from pepin.camera import CameraConfig, mount_transform
from pepin.depth import (
    Array,
    CameraPose,
    DepthScale,
    Intrinsics,
    depth_to_scan,
    project,
    rotation_matrix,
    scale_from_samples,
    scan_points,
    to_base,
)

CONFIG = "/ws/config/camera.json"
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

    def __call__(self, rgb: Array) -> Array:
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
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, reliable)
        self.create_subscription(Image, "/camera/image", self._on_image, newest)
        self.create_subscription(LaserScan, "/scan", self._on_scan, reliable)
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self, spin_thread=True)
        self._lidar_mount: tuple[Array, Array] | None = None
        self._scan: LaserScan | None = None
        self._pending: Image | None = None
        self._wake = threading.Condition()
        self._scale = DepthScale()
        self._stats = {"frames": 0, "dropped": 0, "verdicts": 0, "samples": 0, "model_s": 0.0}
        self._since = time.monotonic()
        self.get_logger().info(f"loading {model_name} on CPU")
        self._net = MonoDepth(model_name, threads)
        threading.Thread(target=self._work, daemon=True).start()
        self.create_timer(30.0, self._report)
        self.get_logger().info("depth stream up: /camera/image -> /camera/depth")

    def _on_info(self, msg: CameraInfo) -> None:
        self._intr = Intrinsics(
            float(msg.k[0]),
            float(msg.k[4]),
            float(msg.k[2]),
            float(msg.k[5]),
            msg.width,
            msg.height,
        )

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
            self._process(msg)

    def _process(self, msg: Image) -> None:
        if msg.encoding not in ("bgr8", "rgb8"):
            self.get_logger().warning(
                f"cannot read {msg.encoding} images", throttle_duration_sec=30
            )
            return
        pixels = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, 3)
        rgb: Array = np.ascontiguousarray(pixels[:, :, ::-1] if msg.encoding == "bgr8" else pixels)
        t0 = time.monotonic()
        depth = self._net(rgb)
        self._stats["model_s"] += time.monotonic() - t0
        verdict = self._lidar_verdict(depth, msg)
        scale = self._scale.observe(verdict)
        if verdict is not None:
            self._stats["verdicts"] += 1
            self._stats["samples"] += verdict[1]
        out = Image()
        out.header = msg.header
        out.height, out.width = msg.height, msg.width
        out.encoding = "32FC1"
        out.is_bigendian = 0
        out.step = msg.width * 4
        out.data = (depth * scale).astype(np.float32).tobytes()
        self._pub.publish(out)
        self._scan_pub.publish(self._as_scan(depth * scale, msg))
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

    def _lidar_verdict(self, depth: Array, image: Image) -> tuple[float, int] | None:
        """What the newest scan says the frame's scale is: needs a fresh scan and both mounts."""
        scan, intr = self._scan, self._intr
        if scan is None or intr is None:
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
        samples = project(to_base(xy, rotation, translation), self._camera, intr)
        return scale_from_samples(depth, samples)

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
        self.get_logger().info(
            f"depth: {s['frames'] / span:.1f} frames/s ({s['dropped']} dropped),"
            f" model {per_frame:.2f} s/frame, scale {self._scale.value:.2f}"
            f" from {s['verdicts']} lidar verdicts ({per_verdict:.0f} samples each;"
            f" held {self._scale.held} frames)"
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
