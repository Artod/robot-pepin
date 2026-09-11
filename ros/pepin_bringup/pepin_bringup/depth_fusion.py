"""The room as one surface: every depth frame fused into a TSDF on the laptop.

RTAB-Map assembles its cloud by concatenating one cloud per node, so two frames of one wall that
disagree by a few centimetres are two walls. This node fuses the same frames (``/camera/depth``
with its ``/camera/image``, placed by the tracker's pose from TF) into ``pepin.tsdf``: one
signed distance per voxel, updated by a distance-weighted average, so the wall is one surface,
sharpened by near observations and never blurred back by far ones. Before a frame is fused, its
points in the lidar's height band — exact by construction — are turned about the cart to fit
the model, and the corrected heading places the frame (frame-to-model; the tracker's jitter at
rest stays out of the model). ``/fusion/surface`` (PointCloud2, map frame, colours from the
camera) is the zero-crossing of the field, published once a second beside RTAB-Map's cloud.

Switches, live (``ros2 param set /depth_fusion <name> <value>``): ``enabled`` (fuse or not),
``align`` (frame-to-model on/off), ``min_weight`` (how many observations a voxel needs before
it is shown), ``surface_hz``. ``/fusion/reset`` (std_srvs/Trigger) empties the model.
"""

from __future__ import annotations

import math
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
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from pepin.depth import Intrinsics, decode_rgb, rotation_matrix
from pepin.tsdf import (
    YAW_SEARCH,
    GridSpec,
    RigidPose,
    SlowCorrection,
    Tsdf,
    align_yaw,
    backproject,
)

CONFIG = "/ws/config/fusion.json"
TF_WAIT_S = 0.3
IMAGES_KEPT = 40  # depth arrives a fraction of a second after its image; pair by stamp
BAND_Z_M = (0.10, 0.35)  # around the lidar's plane (0.20 m): the frame's exact points
BAND_STRIDE = 3
BAND_MIN_POINTS = 50  # a frame with fewer points in the band is not worth a yaw search


class DepthFusion(Node):
    """Fuses depth frames into the TSDF and publishes its surface."""

    def __init__(self) -> None:
        super().__init__("depth_fusion")
        config = Path(str(self.declare_parameter("config", CONFIG).value))
        self._spec = GridSpec.load(config)
        self._enabled = bool(self.declare_parameter("enabled", True).value)
        self._align = bool(self.declare_parameter("align", True).value)
        self._min_weight = float(self.declare_parameter("min_weight", 2.0).value)
        # The tracker's map->odom jitters by degrees while the cart turns; a slow copy of it kept
        # turns clean but lagged after a drive across the room (the frames of the drive landed
        # rotated against the lidar), so it is off by default. Live switch: smooth_map_odom.
        self._smooth = bool(self.declare_parameter("smooth_map_odom", False).value)
        self._slow = SlowCorrection()
        surface_hz = float(self.declare_parameter("surface_hz", 1.0).value)
        self.add_on_set_parameters_callback(self._on_params)
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        newest = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(PointCloud2, "/fusion/surface", reliable)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, reliable)
        self.create_subscription(Image, "/camera/image", self._on_image, reliable)
        self.create_subscription(Image, "/camera/depth", self._on_depth, newest)
        self.create_service(Trigger, "/fusion/reset", self._on_reset)
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self, spin_thread=True)
        self._intr: Intrinsics | None = None
        self._images: dict[int, Image] = {}
        self._pending: Image | None = None
        self._wake = threading.Condition()
        self._lock = threading.Lock()  # the model, between the worker and the publisher
        self._model = Tsdf(self._spec)
        self._stats = self._fresh_stats()
        self._since = time.monotonic()
        threading.Thread(target=self._work, daemon=True).start()
        self.create_timer(1.0 / max(surface_hz, 0.1), self._publish_surface)
        self.create_timer(30.0, self._report)
        nx, ny, nz = self._spec.shape
        self.get_logger().info(
            f"fusion up: {nx}x{ny}x{nz} voxels of {self._spec.voxel_m * 100:.0f} cm from"
            f" {self._spec.origin}; align {'on' if self._align else 'off'}"
        )

    @staticmethod
    def _fresh_stats() -> dict[str, Any]:
        return {
            "frames": 0,
            "dropped": 0,
            "no_tf": 0,
            "no_image": 0,
            "integrate_s": 0.0,
            "align_s": 0.0,
            "corrections": [],
            "gains": [],
            "voxels": 0,
        }

    # ---- switches ------------------------------------------------------------------------
    def _on_params(self, params: list[Any]) -> SetParametersResult:
        for p in params:
            if p.name == "enabled":
                self._enabled = bool(p.value)
            elif p.name == "align":
                self._align = bool(p.value)
            elif p.name == "min_weight":
                self._min_weight = float(p.value)
            elif p.name == "smooth_map_odom":
                self._smooth = bool(p.value)
        self.get_logger().info(
            f"fusion switches: enabled {self._enabled}, align {self._align},"
            f" min_weight {self._min_weight}"
        )
        return SetParametersResult(successful=True)

    def _on_reset(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        with self._lock:
            self._model = Tsdf(self._spec)
        response.success = True
        response.message = "the model is empty"
        self.get_logger().info("fusion: model reset")
        return response

    # ---- inputs --------------------------------------------------------------------------
    def _on_info(self, msg: CameraInfo) -> None:
        self._intr = Intrinsics.from_camera_info(msg.k, msg.width, msg.height)

    def _on_image(self, msg: Image) -> None:
        # The worker pops from this dict on its own thread: take the oldest key by pop, never by
        # ``del``, or the frame it pops between the lookup and the delete raises here.
        self._images[Time.from_msg(msg.header.stamp).nanoseconds] = msg
        while len(self._images) > IMAGES_KEPT:
            oldest = next(iter(self._images), None)
            if oldest is None:
                break
            self._images.pop(oldest, None)

    def _on_depth(self, msg: Image) -> None:
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
            if not self._enabled:
                continue
            try:
                self._fuse(msg)
            except Exception:  # a raise must not kill the worker silently
                self.get_logger().error(f"fusion failed on a frame:\n{traceback.format_exc()}")

    # ---- the frame -----------------------------------------------------------------------
    def _pose(self, target: str, source: str, stamp: Any) -> RigidPose | None:
        try:
            t = self._tf.lookup_transform(
                target, source, stamp, timeout=Duration(seconds=TF_WAIT_S)
            )
        except Exception:
            return None
        q, v = t.transform.rotation, t.transform.translation
        return RigidPose(rotation_matrix(q.x, q.y, q.z, q.w), np.array([v.x, v.y, v.z]))

    def _placement(self, frame: str, stamp: Any) -> tuple[RigidPose | None, RigidPose | None]:
        """Where the camera and the cart are in the map at ``stamp``: the tracker's map<-odom,
        slowed, composed with the odometry's odom<-camera and odom<-base_link — or the raw
        map<-... lookups when the smoothing is off."""
        if not self._smooth:
            return self._pose("map", frame, stamp), self._pose("map", "base_link", stamp)
        correction = self._pose("map", "odom", stamp)
        cam = self._pose("odom", frame, stamp)
        base = self._pose("odom", "base_link", stamp)
        if correction is None or cam is None or base is None:
            return None, None
        slow = self._slow.observe(correction, Time.from_msg(stamp).nanoseconds * 1e-9)
        return SlowCorrection.compose(slow, cam), SlowCorrection.compose(slow, base)

    def _fuse(self, msg: Image) -> None:
        intr = self._intr
        if intr is None or msg.encoding != "32FC1":
            return
        stamp = msg.header.stamp
        camera, base = self._placement(msg.header.frame_id, stamp)
        if camera is None or base is None:
            self._stats["no_tf"] += 1
            return
        depth = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(msg.height, msg.width)
        image = self._images.pop(Time.from_msg(stamp).nanoseconds, None)
        rgb: npt.NDArray[np.uint8] | None = None
        if image is not None:
            rgb = decode_rgb(bytes(image.data), image.height, image.width, image.encoding)
        if rgb is None:
            self._stats["no_image"] += 1
        if self._align:
            t0 = time.monotonic()
            camera = self._aligned(depth, intr, camera, base)
            self._stats["align_s"] += time.monotonic() - t0
        t0 = time.monotonic()
        with self._lock:
            touched = self._model.integrate(depth.astype(float), rgb, intr, camera)
        self._stats["integrate_s"] += time.monotonic() - t0
        self._stats["frames"] += 1
        self._stats["voxels"] += touched

    def _aligned(
        self, depth: Any, intr: Intrinsics, camera: RigidPose, base: RigidPose
    ) -> RigidPose:
        """The camera pose turned about the cart by the yaw that seats the frame's lidar-height
        band on the model; the pose as given when the model cannot judge."""
        points = backproject(depth.astype(float), intr, stride=BAND_STRIDE)
        in_map = points @ camera.rotation.T + camera.translation
        band = in_map[(in_map[:, 2] >= BAND_Z_M[0]) & (in_map[:, 2] <= BAND_Z_M[1])]
        if band.shape[0] < BAND_MIN_POINTS:
            return camera
        pivot = (float(base.translation[0]), float(base.translation[1]))
        with self._lock:
            found = align_yaw(self._model, band, pivot)
        if found is None:
            return camera
        yaw, gain, _judged = found
        self._stats["corrections"].append(math.degrees(yaw))
        self._stats["gains"].append(gain)
        return camera.turned_about(pivot, yaw)

    # ---- outputs -------------------------------------------------------------------------
    def _publish_surface(self) -> None:
        with self._lock:  # a copy under the lock (milliseconds), the crossing search outside it
            snapshot = self._model.snapshot()
        points, colours = snapshot.surface(self._min_weight)
        self._stats["surface"] = int(points.shape[0])
        cloud = np.zeros(
            points.shape[0],
            dtype=[
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("pad", "<f4"),
                ("rgb", "<f4"),
                ("pad2", "<f4", 3),
            ],
        )
        cloud["x"], cloud["y"], cloud["z"] = points[:, 0], points[:, 1], points[:, 2]
        packed = (
            (colours[:, 0].astype(np.uint32) << 16)
            | (colours[:, 1].astype(np.uint32) << 8)
            | colours[:, 2].astype(np.uint32)
        )
        cloud["rgb"] = packed.view(np.float32)
        msg = PointCloud2()
        msg.header = Header(frame_id="map", stamp=self.get_clock().now().to_msg())
        msg.height, msg.width = 1, points.shape[0]
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=16, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 32
        msg.row_step = 32 * points.shape[0]
        msg.is_dense = True
        msg.data = cloud.tobytes()
        self._pub.publish(msg)

    def _report(self) -> None:
        s, self._stats = self._stats, self._fresh_stats()
        elapsed = max(time.monotonic() - self._since, 1e-6)
        self._since = time.monotonic()
        frames = max(int(s["frames"]), 1)
        corr = np.abs(np.array(s["corrections"])) if s["corrections"] else np.zeros(0)
        if corr.size:
            bound = math.degrees(max(abs(y) for y in YAW_SEARCH))
            at_bound = float(np.mean(corr >= bound - 1e-6)) * 100
            signed = float(np.median(np.array(s["corrections"])))
            turns = (
                f"{corr.size} turns (|yaw| median {np.median(corr):.2f} deg, max {corr.max():.2f},"
                f" signed median {signed:+.2f}, {at_bound:.0f}% at the {bound:.0f} deg bound,"
                f" gain median {np.median(s['gains']):.3f})"
            )
        else:
            turns = "no turns"
        integrate_ms = s["integrate_s"] / frames * 1e3
        align_ms = s["align_s"] / frames * 1e3
        self.get_logger().info(
            f"fusion: {s['frames']} frames ({s['frames'] / elapsed:.1f}/s, {s['dropped']} dropped),"
            f" integrate {integrate_ms:.0f} ms, align {align_ms:.0f} ms, {turns},"
            f" no tf {s['no_tf']}, no image {s['no_image']}, surface {s.get('surface', 0)} points"
        )


def main() -> None:
    rclpy.init()
    node = DepthFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
