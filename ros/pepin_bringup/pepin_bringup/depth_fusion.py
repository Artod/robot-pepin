"""The room as one surface: every depth frame fused into a TSDF on the laptop.

RTAB-Map assembles its cloud by concatenating one cloud per node, so two frames of one wall that
disagree by a few centimetres are two walls. This node fuses the same frames (``/camera/depth``
paired with its ``/camera/image`` by stamp — the depth carries the image's header — and placed
by the tracker's pose from TF at that stamp, through the :class:`pepin.frame_pose.FramePoser`
the depth nodes share, over the kit's :class:`TfHistory`) into ``pepin.tsdf``: one signed
distance per voxel, updated by a distance-weighted average, so the wall is one surface,
sharpened by near observations and never blurred back by far ones. Before a frame is fused, its
points in the lidar's height band — exact by construction — are turned about the cart to fit
the model, and the corrected heading places the frame (frame-to-model; the tracker's jitter at
rest stays out of the model). A frame whose best turn is the search's edge is refused: the
truth may lie beyond, and a turn to the bound would bake the remainder in. Frames are fused only
while the tracker reports a fit the cart may drive on (``/localization_fit`` >=
``pepin.watch.DRIVE_FIT``): a lost tracker's pose would paint the room somewhere else.
``/fusion/surface`` (PointCloud2, map frame, colours from the camera, stamped with the last
fused frame — the board's clock) is the zero-crossing of the field, published beside RTAB-Map's
cloud.

The flags (:data:`FLAGS`, ``ros/flags.sh set depth_fusion <flag> <value>``): ``enabled``,
``align``, ``min_weight``, ``surface_hz``; their state is printed in every report line.
``/fusion/reset`` (std_srvs/Trigger) empties the model, the pairing queues and the tallies.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

import numpy as np
from message_filters import Subscriber, TimeSynchronizer
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Float32
from std_srvs.srv import Trigger

from pepin.depth import Intrinsics
from pepin.flags import Flag, FlagSet
from pepin.frame_pose import FramePoser
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
from pepin_bringup.msgs import array_from_image, cloud_from_points, stamp_seconds
from pepin_bringup.node_kit import (
    Switches,
    Tally,
    TfHistory,
    TfLookup,
    Window,
    Worker,
    spin_main,
)

CONFIG = "/ws/config/fusion.json"
TF_WAIT_S = 0.3
PAIR_QUEUE = 40  # depth arrives a fraction of a second after its image; pair by exact stamp
BAND_Z_M = (0.10, 0.35)  # around the lidar's plane (0.20 m): the frame's exact points
BAND_STRIDE = 3
BAND_MIN_POINTS = 50  # a frame with fewer points in the band is not worth a yaw search
AT_BOUND_STREAK = 30  # ~3 s of frames refused at the search's bound: the model no longer fits
STAGES = ("align", "integrate")

# The live flags (CLAUDE.md rule 19), declared last in __init__ so the kit's callback sees no
# other declaration; their state is printed in every report line.
FLAGS = FlagSet(
    Flag("enabled", True, description="frames are fused into the model; off, they are dropped"),
    Flag(
        "self_heal",
        True,
        description="a streak of frames refused at the alignment bound empties the model, so it"
        " re-seeds from the next frame instead of staying frozen until a human resets it",
    ),
    Flag(
        "align",
        True,
        description="frame-to-model: a frame's lidar-height band is turned about the cart to"
        " fit the model before it is fused, and a frame whose best turn is the search's bound"
        " is refused",
    ),
    Flag(
        "min_weight",
        2.0,
        range=(0.0, 100.0),
        description="observations a voxel needs before it is shown in /fusion/surface",
    ),
    Flag(
        "surface_hz",
        1.0,
        range=(0.1, 10.0),
        description="how often /fusion/surface is published (the crossing search costs a"
        " fraction of a second)",
    ),
)


class DepthFusion(Node):
    """Fuses depth frames into the TSDF and publishes its surface."""

    def __init__(self) -> None:
        super().__init__("depth_fusion")
        config = Path(str(self.declare_parameter("config", CONFIG).value))
        self._spec = GridSpec.load(config)
        self._switches = Switches(self, FLAGS, on_change=self._on_switch)
        self._tally = Tally(STAGES)
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(PointCloud2, "/fusion/surface", reliable)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, reliable)
        self.create_subscription(Float32, "/localization_fit", self._on_fit, reliable)
        self.create_service(Trigger, "/fusion/reset", self._on_reset)
        # the depth copies the image's header, so the pair has one exact stamp; the synchronizer
        # keeps PAIR_QUEUE of each and calls back under its own lock, on the executor thread
        depth_sub = Subscriber(self, Image, "/camera/depth", qos_profile=reliable)
        image_sub = Subscriber(self, Image, "/camera/image", qos_profile=reliable)
        depth_sub.registerCallback(lambda _msg: self._tally.count("depth_in"))
        self._sync = TimeSynchronizer([depth_sub, image_sub], PAIR_QUEUE)
        self._sync.registerCallback(self._on_pair)
        self._tf = TfLookup(self, on_failure=self._on_tf_failure)
        self._poser = FramePoser(TfHistory(self._tf, timeout_s=TF_WAIT_S))
        self._intr: Intrinsics | None = None
        self._fit = 0.0  # no report yet reads as lost: every gate here compares with <
        self._lock = threading.Lock()  # the model and its last stamp, worker vs publisher
        self._model = Tsdf(self._spec)
        self._last_stamp: Any = None  # the last fused frame's header stamp, the board's clock
        self._bound_streak = 0  # consecutive frames refused at the bound (self-healing)
        self._surface_points = 0
        self._worker = Worker(self._on_work, name="fusion", on_error=self._on_work_error).start()
        self._surface_timer = self.create_timer(
            self._period(self._switches["surface_hz"]), self._publish_surface
        )
        self.create_timer(30.0, self._report)
        nx, ny, nz = self._spec.shape
        self.get_logger().info(
            f"fusion up: {nx}x{ny}x{nz} voxels of {self._spec.voxel_m * 100:.0f} cm from"
            f" {self._spec.origin}; {self._switches.state()}; fused while"
            f" /localization_fit >= {DRIVE_FIT:.2f}"
        )

    def close(self) -> None:
        """Stop the worker and the TF listener and wait for them, before the node is destroyed."""
        if not self._worker.stop():
            self.get_logger().warning("the fusion worker did not finish its frame; leaving anyway")
        self._tf.close()

    @staticmethod
    def _period(surface_hz: float) -> float:
        return 1.0 / max(surface_hz, 0.1)

    # ---- switches ------------------------------------------------------------------------
    def _on_switch(self, name: str, _old: Any, new: Any) -> None:
        """A flag changed: only ``surface_hz`` has anything to do beyond being read."""
        if name != "surface_hz":
            return
        try:
            self._surface_timer.timer_period_ns = int(self._period(float(new)) * 1e9)
        except (AttributeError, TypeError) as exc:  # an rclpy without a live period
            raise ValueError(f"surface_hz cannot change live: {exc}") from exc

    def _on_work_error(self, text: str) -> None:
        self.get_logger().error(f"fusion failed on a frame:\n{text}")

    def _on_tf_failure(self, kind: str, text: str) -> None:
        self._tally.count("no_tf")
        self._tally.count("tf_" + kind)
        self._tally.note(kind, text)

    def _on_reset(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        with self._lock:
            self._model = Tsdf(self._spec)
            self._last_stamp = None
        self._worker.clear()
        with self._sync.lock:
            for queue in self._sync.queues:
                queue.clear()
        self._tally.take()
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
        self._tally.count("pairs")
        if self._worker.offer((depth, image)):
            self._tally.count("dropped")

    def _on_work(self, pair: tuple[Image, Image]) -> None:
        """The worker's item: a pair is fused unless the node is switched off."""
        if self._switches.on("enabled"):
            self._fuse(*pair)

    # ---- the frame -----------------------------------------------------------------------
    def _fuse(self, msg: Image, image: Image) -> None:
        intr = self._intr
        if intr is None:
            self._tally.count("no_intrinsics")
            return
        if msg.encoding != "32FC1" or (msg.width, msg.height) != (intr.width, intr.height):
            self._tally.count("bad_frame")  # not the camera the intrinsics describe
            return
        if msg.header.frame_id != self._poser.camera:
            self._tally.count("bad_frame")  # not the camera the poser places
            return
        if self._fit < DRIVE_FIT:
            self._tally.count("low_fit")
            return
        stamp = msg.header.stamp
        at = stamp_seconds(stamp)
        camera = self._poser.camera_in_map(at)
        base = self._poser.base_in_map(at) if camera is not None else None
        if camera is None or base is None:
            return
        depth = array_from_image(msg)
        rgb = array_from_image(image)
        if depth is None:
            self._tally.count("bad_frame")
            return
        if rgb is None or rgb.ndim != 3 or rgb.shape[:2] != depth.shape:
            self._tally.count("no_image")  # an encoding or size the decoder cannot pair
            rgb = None
        if self._switches.on("align"):
            with self._tally.measure("align"):
                aligned = self._aligned(depth, intr, camera, base)
            if aligned is None:
                return  # AT_BOUND: counted, not integrated
            camera = aligned
        with self._tally.measure("integrate"), self._lock:
            touched = self._model.integrate(depth, rgb, intr, camera)
            self._last_stamp = stamp
        self._tally.count("frames")
        self._tally.count("voxels", touched)

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
            self._refused(AlignReason.UNJUDGED)
            return camera
        pivot = (float(base.translation[0]), float(base.translation[1]))
        with self._lock:
            verdict = align_yaw(self._model, band, pivot)
        if verdict.reason is AlignReason.ALIGNED:
            self._bound_streak = 0
            self._tally.sample("yaw_deg", math.degrees(verdict.yaw))
            self._tally.sample("gain", verdict.gain)
            return camera.turned_about(pivot, verdict.yaw)
        self._refused(verdict.reason)
        if verdict.reason is not AlignReason.AT_BOUND:
            self._bound_streak = 0
            return camera
        self._bound_streak += 1
        if self._switches.on("self_heal") and self._bound_streak >= AT_BOUND_STREAK:
            self._self_heal()
            return camera  # the first frame of the new model goes in as given
        return None

    def _self_heal(self) -> None:
        """Empty a model that no more frame fits: after ``AT_BOUND_STREAK`` refusals in a row the
        room has moved on (a head that turned, a law that drifted) and the surface would stay
        frozen forever; the next frame seeds a fresh model instead."""
        with self._lock:
            self._model = Tsdf(self._spec)
            self._last_stamp = None
        self._bound_streak = 0
        self._tally.count("self_heals")
        self.get_logger().warning(
            f"{AT_BOUND_STREAK} frames in a row refused at the alignment bound: the model is"
            " emptied and re-seeds from the next frame (flag self_heal)"
        )

    def _refused(self, reason: AlignReason) -> None:
        self._tally.count("refused_" + reason.value)

    # ---- outputs -------------------------------------------------------------------------
    def _publish_surface(self) -> None:
        with self._lock:  # a copy under the lock (milliseconds), the crossing search outside it
            snapshot = self._model.snapshot()
            stamp = self._last_stamp
        points, colours = snapshot.surface(self._switches["min_weight"])
        self._surface_points = int(points.shape[0])  # a level the report reads, not a tally
        # the board's clock: the surface is as old as the last frame in it, not as new as now
        self._pub.publish(
            cloud_from_points(
                points,
                colours,
                stamp if stamp is not None else self.get_clock().now().to_msg(),
                "map",
            )
        )

    def _report(self) -> None:
        w = self._tally.take()
        c = w.counts
        skipped = (
            f"low fit {c['low_fit']}, at bound {c['refused_at_bound']},"
            f" self-heals {c['self_heals']},"
            f" no tf {c['no_tf']}, bad frame {c['bad_frame']}, no intrinsics {c['no_intrinsics']}"
        )
        tf_text = "; ".join(f"{k} {c['tf_' + k]}: {v}" for k, v in w.notes.items())
        unpaired = max(int(c["depth_in"]) - int(c["pairs"]), 0)  # depths whose image never came
        self.get_logger().info(
            f"fusion: {c['frames']} frames ({w.rate('frames'):.1f}/s, {c['dropped']} dropped,"
            f" {unpaired} unpaired), integrate {w.ms_per('integrate', 'frames'):.0f} ms,"
            f" align {w.ms_per('align', 'frames'):.0f} ms, {self._turns(w)};"
            f" refused: {self._refusals(w) or 'none'}; skipped: {skipped};"
            f" no image {c['no_image']}; surface {self._surface_points} points;"
            f" flags: {self._switches.state()}" + (f"; tf: {tf_text}" if tf_text else "")
        )

    @staticmethod
    def _turns(w: Window) -> str:
        """The window's heading corrections: how many, how big, and how much score they bought."""
        yaws = w.samples.get("yaw_deg", [])
        if not yaws:
            return "no turns"
        size = np.abs(np.array(yaws))
        bound = math.degrees(max(abs(y) for y in YAW_SEARCH))
        return (
            f"{size.size} turns (|yaw| median {np.median(size):.2f} deg, max {size.max():.2f}"
            f" of the {bound:.0f} deg search, signed median {float(np.median(yaws)):+.2f},"
            f" gain median {np.median(w.samples['gain']):.3f})"
        )

    @staticmethod
    def _refusals(w: Window) -> str:
        """Why the alignment refused frames this window, in the reasons' own order."""
        return ", ".join(
            f"{r.value} {w.counts['refused_' + r.value]}"
            for r in AlignReason
            if w.counts["refused_" + r.value]
        )


def main() -> None:
    spin_main(DepthFusion)


if __name__ == "__main__":
    main()
