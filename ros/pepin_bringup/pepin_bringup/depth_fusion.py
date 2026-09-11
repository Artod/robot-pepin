"""The room as one surface: every depth frame fused into a TSDF on the laptop.

RTAB-Map assembles its cloud by concatenating one cloud per node, so two frames of one wall that
disagree by a few centimetres are two walls. This node fuses the same frames (``/camera/depth``
paired with its ``/camera/image`` by stamp — the depth carries the image's header — and placed
by the tracker's pose from TF at that stamp) into ``pepin.tsdf``: one signed distance per voxel,
updated by a distance-weighted average, so the wall is one surface, sharpened by near
observations and never blurred back by far ones. Before a frame is fused, its points in the
lidar's height band — exact by construction — are turned about the cart to fit the model, and
the corrected heading places the frame (frame-to-model; the tracker's jitter at rest stays out
of the model). A frame whose best turn is the search's edge is refused: the truth may lie
beyond, and a turn to the bound would bake the remainder in. Frames are fused only while the
tracker reports a fit the cart may drive on (``/localization_fit`` >= ``pepin.watch.DRIVE_FIT``):
a lost tracker's pose would paint the room somewhere else. ``/fusion/surface`` (PointCloud2,
map frame, colours from the camera, stamped with the last fused frame — the board's clock) is
the zero-crossing of the field, published beside RTAB-Map's cloud.

Switches, live (``ros2 param set /depth_fusion <name> <value>``): ``enabled`` (fuse or not),
``align`` (frame-to-model on/off), ``min_weight`` (how many observations a voxel needs before
it is shown), ``surface_hz``. ``/fusion/reset`` (std_srvs/Trigger) empties the model, the
pairing queues and the tallies.
"""

from __future__ import annotations

import math
import threading
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from message_filters import Subscriber, TimeSynchronizer
from rcl_interfaces.msg import SetParametersResult
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Float32, Header
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from pepin.depth import Intrinsics, decode_rgb, rotation_matrix
from pepin.tsdf import (
    YAW_SEARCH,
    AlignReason,
    GridSpec,
    RigidPose,
    Tsdf,
    align_yaw,
    backproject,
)
from pepin.watch import DRIVE_FIT

CONFIG = "/ws/config/fusion.json"
TF_WAIT_S = 0.3
PAIR_QUEUE = 40  # depth arrives a fraction of a second after its image; pair by exact stamp
BAND_Z_M = (0.10, 0.35)  # around the lidar's plane (0.20 m): the frame's exact points
BAND_STRIDE = 3
BAND_MIN_POINTS = 50  # a frame with fewer points in the band is not worth a yaw search


@dataclass
class Tally:
    """One report period's numbers: counts and seconds by name, every turn applied, the
    alignment's refusals by reason, and the last TF failure's text by kind."""

    counts: Counter[str] = field(default_factory=Counter)
    seconds: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    corrections_deg: list[float] = field(default_factory=list)
    gains: list[float] = field(default_factory=list)
    refusals: Counter[str] = field(default_factory=Counter)
    tf_errors: dict[str, str] = field(default_factory=dict)


class Stats:
    """The tallies the worker fills and the report timer empties, under one lock: the two run
    on different threads, and a swap racing an increment lost counts (or worse)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tally = Tally()

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._tally.counts[name] += n

    def spent(self, name: str, seconds: float) -> None:
        with self._lock:
            self._tally.seconds[name] += seconds

    def turned(self, yaw_deg: float, gain: float) -> None:
        with self._lock:
            self._tally.corrections_deg.append(yaw_deg)
            self._tally.gains.append(gain)

    def refused(self, reason: AlignReason) -> None:
        with self._lock:
            self._tally.refusals[reason.value] += 1

    def tf_failed(self, kind: str, text: str) -> None:
        with self._lock:
            self._tally.counts["no_tf"] += 1
            self._tally.counts["tf_" + kind] += 1
            self._tally.tf_errors[kind] = text

    def take(self) -> Tally:
        """The period's tally, and a fresh one starts."""
        with self._lock:
            taken, self._tally = self._tally, Tally()
        return taken


class DepthFusion(Node):
    """Fuses depth frames into the TSDF and publishes its surface."""

    def __init__(self) -> None:
        super().__init__("depth_fusion")
        config = Path(str(self.declare_parameter("config", CONFIG).value))
        self._spec = GridSpec.load(config)
        self._enabled = bool(self.declare_parameter("enabled", True).value)
        self._align = bool(self.declare_parameter("align", True).value)
        self._min_weight = float(self.declare_parameter("min_weight", 2.0).value)
        surface_hz = float(self.declare_parameter("surface_hz", 1.0).value)
        self.add_on_set_parameters_callback(self._on_params)
        self._stats = Stats()
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(PointCloud2, "/fusion/surface", reliable)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, reliable)
        self.create_subscription(Float32, "/localization_fit", self._on_fit, reliable)
        self.create_service(Trigger, "/fusion/reset", self._on_reset)
        # the depth copies the image's header, so the pair has one exact stamp; the synchronizer
        # keeps PAIR_QUEUE of each and calls back under its own lock, on the executor thread
        depth_sub = Subscriber(self, Image, "/camera/depth", qos_profile=reliable)
        image_sub = Subscriber(self, Image, "/camera/image", qos_profile=reliable)
        depth_sub.registerCallback(lambda _msg: self._stats.count("depth_in"))
        self._sync = TimeSynchronizer([depth_sub, image_sub], PAIR_QUEUE)
        self._sync.registerCallback(self._on_pair)
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self, spin_thread=True)
        self._intr: Intrinsics | None = None
        self._fit = 0.0  # no report yet reads as lost: every gate here compares with <
        self._pending: tuple[Image, Image] | None = None
        self._wake = threading.Condition()
        self._lock = threading.Lock()  # the model and its last stamp, worker vs publisher
        self._model = Tsdf(self._spec)
        self._last_stamp: Any = None  # the last fused frame's header stamp, the board's clock
        self._surface_points = 0
        self._since = time.monotonic()
        threading.Thread(target=self._work, daemon=True).start()
        self._surface_timer = self.create_timer(self._period(surface_hz), self._publish_surface)
        self.create_timer(30.0, self._report)
        nx, ny, nz = self._spec.shape
        self.get_logger().info(
            f"fusion up: {nx}x{ny}x{nz} voxels of {self._spec.voxel_m * 100:.0f} cm from"
            f" {self._spec.origin}; align {'on' if self._align else 'off'}, fused while"
            f" /localization_fit >= {DRIVE_FIT:.2f}"
        )

    @staticmethod
    def _period(surface_hz: float) -> float:
        return 1.0 / max(surface_hz, 0.1)

    # ---- switches ------------------------------------------------------------------------
    def _on_params(self, params: list[Any]) -> SetParametersResult:
        for p in params:
            if p.name == "enabled":
                self._enabled = bool(p.value)
            elif p.name == "align":
                self._align = bool(p.value)
            elif p.name == "min_weight":
                self._min_weight = float(p.value)
            elif p.name == "surface_hz":
                try:
                    self._surface_timer.timer_period_ns = int(self._period(float(p.value)) * 1e9)
                except (AttributeError, TypeError) as e:  # an rclpy without a live period
                    return SetParametersResult(
                        successful=False, reason=f"surface_hz cannot change live: {e}"
                    )
        self.get_logger().info(
            f"fusion switches: enabled {self._enabled}, align {self._align},"
            f" min_weight {self._min_weight},"
            f" surface every {self._surface_timer.timer_period_ns * 1e-9:.2f} s"
        )
        return SetParametersResult(successful=True)

    def _on_reset(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        with self._lock:
            self._model = Tsdf(self._spec)
            self._last_stamp = None
        with self._wake:
            self._pending = None
        with self._sync.lock:
            for queue in self._sync.queues:
                queue.clear()
        self._stats.take()
        self._since = time.monotonic()
        response.success = True
        response.message = "the model is empty"
        self.get_logger().info("fusion: model, pairing queues and tallies reset")
        return response

    # ---- inputs --------------------------------------------------------------------------
    def _on_info(self, msg: CameraInfo) -> None:
        self._intr = Intrinsics.from_camera_info(msg.k, msg.width, msg.height)

    def _on_fit(self, msg: Float32) -> None:
        self._fit = float(msg.data)

    def _on_pair(self, depth: Image, image: Image) -> None:
        """A depth frame with its picture, same stamp: the newest pair waits for the worker,
        an older one still waiting is dropped (the model wants the latest view, not a backlog)."""
        self._stats.count("pairs")
        with self._wake:
            if self._pending is not None:
                self._stats.count("dropped")
            self._pending = (depth, image)
            self._wake.notify()

    def _work(self) -> None:
        while rclpy.ok():
            with self._wake:
                while self._pending is None:
                    self._wake.wait(1.0)
                    if not rclpy.ok():
                        return
                pair, self._pending = self._pending, None
            if not self._enabled:
                continue
            try:
                self._fuse(*pair)
            except Exception:  # a raise must not kill the worker silently
                self.get_logger().error(f"fusion failed on a frame:\n{traceback.format_exc()}")

    # ---- the frame -----------------------------------------------------------------------
    def _pose(self, target: str, source: str, stamp: Any) -> RigidPose | None:
        """map <- ``source`` at ``stamp`` from TF; ``None`` (counted by kind, the text kept for
        the report) when the chain is missing, not yet published for that time, or late."""
        try:
            t = self._tf.lookup_transform(
                target, source, stamp, timeout=Duration(seconds=TF_WAIT_S)
            )
        except Exception as e:  # tf2's Lookup / Extrapolation / Connectivity / Timeout
            kind = type(e).__name__.removesuffix("Exception") or "Unknown"
            self._stats.tf_failed(kind, f"{target}<-{source}: {str(e).strip()[:160]}")
            return None
        q, v = t.transform.rotation, t.transform.translation
        return RigidPose(rotation_matrix(q.x, q.y, q.z, q.w), np.array([v.x, v.y, v.z]))

    def _fuse(self, msg: Image, image: Image) -> None:
        intr = self._intr
        if intr is None:
            self._stats.count("no_intrinsics")
            return
        if msg.encoding != "32FC1" or (msg.width, msg.height) != (intr.width, intr.height):
            self._stats.count("bad_frame")  # not the camera the intrinsics describe
            return
        if self._fit < DRIVE_FIT:
            self._stats.count("low_fit")
            return
        stamp = msg.header.stamp
        camera = self._pose("map", msg.header.frame_id, stamp)
        base = self._pose("map", "base_link", stamp) if camera is not None else None
        if camera is None or base is None:
            return
        depth = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(msg.height, msg.width)
        rgb = decode_rgb(bytes(image.data), image.height, image.width, image.encoding)
        if rgb is None or rgb.shape[:2] != depth.shape:
            self._stats.count("no_image")  # an encoding or size the decoder cannot pair
            rgb = None
        if self._align:
            t0 = time.monotonic()
            aligned = self._aligned(depth, intr, camera, base)
            self._stats.spent("align", time.monotonic() - t0)
            if aligned is None:
                return  # AT_BOUND: counted, not integrated
            camera = aligned
        t0 = time.monotonic()
        with self._lock:
            touched = self._model.integrate(depth, rgb, intr, camera)
            self._last_stamp = stamp
        self._stats.spent("integrate", time.monotonic() - t0)
        self._stats.count("frames")
        self._stats.count("voxels", touched)

    def _aligned(
        self, depth: Any, intr: Intrinsics, camera: RigidPose, base: RigidPose
    ) -> RigidPose | None:
        """The camera pose turned about the cart by the yaw that seats the frame's lidar-height
        band on the model; the pose as given when the model cannot judge or the frame already
        fits; ``None`` when the best turn is the search's bound (the frame must not go in)."""
        points = backproject(depth, intr, stride=BAND_STRIDE, range_max=self._spec.range_max_m)
        in_map = points @ camera.rotation.T + camera.translation
        band = in_map[(in_map[:, 2] >= BAND_Z_M[0]) & (in_map[:, 2] <= BAND_Z_M[1])]
        if band.shape[0] < BAND_MIN_POINTS:
            self._stats.refused(AlignReason.UNJUDGED)
            return camera
        pivot = (float(base.translation[0]), float(base.translation[1]))
        with self._lock:
            verdict = align_yaw(self._model, band, pivot)
        if verdict.reason is AlignReason.ALIGNED:
            self._stats.turned(math.degrees(verdict.yaw), verdict.gain)
            return camera.turned_about(pivot, verdict.yaw)
        self._stats.refused(verdict.reason)
        return None if verdict.reason is AlignReason.AT_BOUND else camera

    # ---- outputs -------------------------------------------------------------------------
    def _publish_surface(self) -> None:
        with self._lock:  # a copy under the lock (milliseconds), the crossing search outside it
            snapshot = self._model.snapshot()
            stamp = self._last_stamp
        points, colours = snapshot.surface(self._min_weight)
        self._surface_points = int(points.shape[0])  # a level the report reads, not a tally
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
        # the board's clock: the surface is as old as the last frame in it, not as new as now
        msg.header = Header(
            frame_id="map", stamp=stamp if stamp is not None else self.get_clock().now().to_msg()
        )
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
        t = self._stats.take()
        elapsed = max(time.monotonic() - self._since, 1e-6)
        self._since = time.monotonic()
        c = t.counts
        frames = max(int(c["frames"]), 1)
        corr = np.abs(np.array(t.corrections_deg)) if t.corrections_deg else np.zeros(0)
        if corr.size:
            bound = math.degrees(max(abs(y) for y in YAW_SEARCH))
            signed = float(np.median(np.array(t.corrections_deg)))
            turns = (
                f"{corr.size} turns (|yaw| median {np.median(corr):.2f} deg, max {corr.max():.2f}"
                f" of the {bound:.0f} deg search, signed median {signed:+.2f},"
                f" gain median {np.median(t.gains):.3f})"
            )
        else:
            turns = "no turns"
        refused = ", ".join(
            f"{r.value} {t.refusals[r.value]}" for r in AlignReason if r.value in t.refusals
        )
        skipped = (
            f"low fit {c['low_fit']}, at bound {t.refusals[AlignReason.AT_BOUND.value]},"
            f" no tf {c['no_tf']}, bad frame {c['bad_frame']}, no intrinsics {c['no_intrinsics']}"
        )
        tf_text = "; ".join(f"{k} {c['tf_' + k]}: {v}" for k, v in t.tf_errors.items())
        unpaired = max(int(c["depth_in"]) - int(c["pairs"]), 0)  # depths whose image never came
        self.get_logger().info(
            f"fusion: {c['frames']} frames ({c['frames'] / elapsed:.1f}/s, {c['dropped']} dropped,"
            f" {unpaired} unpaired), integrate {t.seconds['integrate'] / frames * 1e3:.0f} ms,"
            f" align {t.seconds['align'] / frames * 1e3:.0f} ms, {turns};"
            f" refused: {refused or 'none'}; skipped: {skipped}; no image {c['no_image']};"
            f" surface {self._surface_points} points" + (f"; tf: {tf_text}" if tf_text else "")
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
