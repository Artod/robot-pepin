"""The camera as a depth sensor, on the laptop: images in, metric depth images out.

Depth Anything V2 (metric, indoor, the small model) turns each camera frame into a depth image
with the right shape and the wrong size — and the wrong size is not one number: the far end of
a room comes out too far by more than the near end. The lidar fixes that: the scan taken
nearest the frame's exposure, carried to the frame's moment through the odometry and projected
into the image through the mounts, names the true depth at the pixels it hits, spanning a
metre to four, and those pixels fit an affine law in inverse depth (1 / z = a / D + b) pooled
across minutes of frames and applied to the whole image. That correction is a
:class:`pepin.depth_pipeline.DepthPipeline` (:func:`pepin.depth_pipeline.standard_pipeline`):
an ordered list of stages — the edge filter, the lidar anchor, the floor and wall anchors, the
law, the wall correction, the floor anchor — each switched by the flag of its name, each
counted and timed per frame, the node owning none of the arithmetic. Nothing is published
until the law exists: the raw network's depth is 1.5-2x too far and would put the costmap's
obstacles where there are none, so frames are withheld until POOL_MIN_SAMPLES beam pairs are
pooled — or until the law saved by the last run (``/maps/depth_law.json``, a day old at most)
is loaded at start. The depth as it stands before the floor anchor, cut between 8 cm and
1.3 m above the floor and folded onto the plane, goes out as ``/depth_scan`` (a LaserScan in
base_link): the board's local costmap marks and clears with it like with the lidar, so a
table top stops the cart the way a wall does. The floor anchor (pixels within centimetres of
the floor plane snap to it, the plane leaning with the accelerometer) is for the 3D model: the
anchored depth goes out on ``/camera/depth`` (32FC1 metres, the image's stamp and frame),
where the fusion builds the model from it and the costmap reads obstacles the lidar's plane
misses. Frames that arrive while the network is busy are dropped: the newest one wins. Every
stage is timed and reported.

Where the camera sits is asked of TF at every frame's stamp (:class:`pepin.frame_pose.FramePoser`
over the kit's :class:`TfHistory`): the neck moves, and ``base_link -> camera_link`` is published
live from its encoders by the board's neck node; config/camera.json's mount is the fallback
while TF has no such edge yet, and the report line counts the frames that used it. The same
poser carries the scan to the frame's moment through the odometry. A pan of the head is counted
too: the projections assume the camera looks along the cart's x.

The network runs where ``depth_backend`` says: ``local`` is the CPU model in this container
(0.2-0.3 s a frame), ``remote`` the same network on the laptop's GPU behind
:mod:`pepin.depth_service` (ros/depth_host.sh, 26 ms a round trip), ``auto`` the service while
it answers and the CPU model while it does not (:class:`pepin.depth_service.Fallback`). The CPU
model is built on its first frame, not at start: a node on the service never pays its gigabyte.
A model that cannot be built (no cached weights and no hub, no memory) is not tried again: in
``local`` mode the node leaves as it did when the model failed in the constructor — exit code
1, the launch respawns it, the respawn retries — and in ``auto`` the frame is lost until the
service answers; the report line says which.

The flags (:data:`FLAGS`, ``ros/flags.sh set depth_stream <flag> <value>``): one per stage of
the pipeline — ``edge_filter``, ``lidar_anchor``, ``floor_pairs``, ``wall_anchor``,
``affine_law``, ``wall_correct``, ``floor_anchor`` — and ``depth_backend``; their state is
printed in every report line.
"""

from __future__ import annotations

import math
import os
import threading
import time
import traceback
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
    UP_LEVEL,
    Array,
    CameraPose,
    Intrinsics,
    Tilt,
    depth_to_scan,
    load_law,
    nearest_stamp,
    optical_heading,
    save_law,
    scan_points,
    to_base,
)
from pepin.depth_pipeline import AffineLaw, FrameContext, standard_pipeline
from pepin.depth_service import (
    DEFAULT_URL,
    MODES,
    DepthModelError,
    Fallback,
    LazyDepth,
    RemoteDepth,
)
from pepin.flags import Flag, FlagSet
from pepin.frame_pose import FramePoser
from pepin.mounts import Mounts
from pepin.tsdf import RigidPose
from pepin_bringup.msgs import (
    array_from_image,
    image_from_array,
    imu_arrays,
    scan_from_ranges,
    stamp_seconds,
)
from pepin_bringup.node_kit import (
    Fatal,
    Switches,
    Tally,
    TfHistory,
    TfLookup,
    Window,
    Worker,
    spin_main,
)

CONFIG = "/ws/config/camera.json"
LAW_FILE = "/maps/depth_law.json"  # ros/maps on the laptop, mounted at /maps by ros/laptop.sh
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SCAN_RANGE_M = 6.0
STAGES = ("network", "pose", "samples", "pipeline", "scan", "publish")
CARRY_WAIT_S = 0.2  # how long TF is given to cover a frame's stamp (the carry, the camera pose)
PAN_NOTICE_RAD = math.radians(1.0)  # a head turned more than this is projected as if it were not
SCAN_BEFORE = "floor_anchor"  # the scan is built from the depth as it stands before this stage

# The live flags (CLAUDE.md rule 19), declared last in __init__ so the kit's callback sees no
# other declaration, and printed in every report line. One bool per stage of the pipeline, in
# its running order and under its name (standard_pipeline's names; the defaults are the chain
# as measured on run 0171, scratch/pipeline_vs_truth.py: the lidar's affine law alone).
FLAGS = FlagSet(
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
    Flag(
        "floor_pairs",
        False,
        description="the floor's pixels pair the network's depth with the plane's geometric depth,"
        " a second hoop for the law that needs no lidar; off by default: measured, it pulls the"
        " law off the lidar's row (1.4x too far there), where the costmap lives",
    ),
    Flag(
        "wall_anchor",
        False,
        description="the lidar's returns extruded up the image while the network's depth stays"
        " continuous pair the rows above the lidar's with the wall's depth, a third hoop; off by"
        " default: measured, it puts the lidar's row 10 % too near while fixing the rows above",
    ),
    Flag(
        "affine_law",
        True,
        description="the network's depth through 1 / z = a / D + b, fitted on the pooled pairs;"
        " off, the raw network's depth goes out unwithheld (1.5-2x too far: an A/B measure of"
        " the correction, never a way to drive)",
    ),
    Flag(
        "wall_correct",
        False,
        description="after the law, the pixels the wall walk covered are set to the extruded"
        " plane's depth outright; off by default (the same walk as wall_anchor, applied instead"
        " of fitted)",
    ),
    Flag(
        "floor_anchor",
        True,
        description="pixels within centimetres of the floor plane snap to it in the published"
        " image (the scan is built before it); the plane leans with the cart, from the IMU's up"
        " vector",
    ),
    Flag(
        "depth_backend",
        "local",
        choices=MODES,
        env="PEPIN_DEPTH_BACKEND",
        description="where the network runs: local (the CPU model in this container), remote"
        " (the laptop's GPU service, ros/depth_host.sh), auto (the service while it answers, the"
        " CPU model while it does not)",
    ),
)
FLOOR_STAGES = ("floor_anchor", "floor_pairs")  # the stages that read the IMU's up vector


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
        self._camera_config = CameraPose(x, y, z, pitch)  # the fallback while TF has no edge
        self._hfov_deg = cfg.hfov_deg  # the nominal optics until a camera_info arrives
        self._intr: Intrinsics | None = None
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        newest = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(Image, "/camera/depth", reliable)
        self._scan_pub = self.create_publisher(LaserScan, "/depth_scan", reliable)
        self._scan_max_range = float(self.declare_parameter("scan_max_range", 3.0).value)
        self._law_file = Path(str(self.declare_parameter("law_file", LAW_FILE).value))
        # The depth service as this container sees it (host.docker.internal is the laptop);
        # read at start: the client reconnects by itself, the address does not move.
        depth_url = str(
            self.declare_parameter(
                "depth_url", os.environ.get("PEPIN_DEPTH_URL", DEFAULT_URL)
            ).value
        )
        self._law = AffineLaw()
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
        self._pipeline = standard_pipeline(self._law)
        self._switches = Switches(self, FLAGS, on_change=self._on_switch)
        for name in self._pipeline.names:  # a launch override reaches the stage it names
            self._pipeline.set(name, self._switches.on(name))
        self._imu_mount = self._imu_rotation(config.parent)
        self._tilt: Tilt | None = None
        self._tally = Tally(STAGES)
        self.create_subscription(Imu, "/imu/data_raw", self._on_imu, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, reliable)
        self.create_subscription(Image, "/camera/image", self._on_image, newest)
        self.create_subscription(LaserScan, "/scan", self._on_scan, reliable)
        self._tf = TfLookup(self, on_failure=self._on_tf_failure)
        self._poser = FramePoser(TfHistory(self._tf, timeout_s=CARRY_WAIT_S))
        self._lidar_mount: RigidPose | None = None
        self._scans: deque[LaserScan] = deque()  # the last SCAN_WINDOW_S of scans, by stamp
        self._scan_lock = threading.Lock()
        # The CPU model is built on its first frame (LazyDepth): in remote or auto mode with
        # the service answering it is never loaded, and the node is up in a second, not ten.
        self._local = LazyDepth(lambda: MonoDepth(model_name, threads))
        self._net = Fallback(
            RemoteDepth(depth_url), self._local, mode=self._switches["depth_backend"]
        )
        self.get_logger().info(
            f"depth backend {self._net.status}: service at {depth_url}; the CPU model"
            f" ({model_name}, {threads} threads) loads on its first local frame"
        )
        self._fatal = Fatal(self)  # the worker's way out when no backend can answer
        self._worker = Worker(self._process, name="depth", on_error=self._on_work_error).start()
        self.create_timer(30.0, self._report)
        self.get_logger().info(
            "depth stream up: /camera/image -> /camera/depth, /depth_scan; camera pose from TF"
            f" (config/camera.json's pitch {math.degrees(pitch):.1f} deg while TF has no edge)"
        )

    def close(self) -> None:
        """Stop the worker and the TF listener and wait for them: called before the node is
        destroyed, so no thread is left inside the network or DDS at interpreter exit."""
        if not self._worker.stop():
            self.get_logger().warning("the depth worker did not finish its frame; leaving anyway")
        self._tf.close()

    def _on_switch(self, name: str, _old: Any, new: Any) -> None:
        """A flag changed: ``depth_backend`` is the switch's mode, a stage's flag switches
        that stage of the pipeline."""
        if name == "depth_backend":
            self._net.mode = str(new)
        elif name in self._pipeline.switches:
            self._pipeline.set(name, bool(new))

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
        any other frame through the mount in config/imu.json. Read only while a floor stage
        is on — the up vector has no other reader, and a stage switched back on picks the
        tilt up afresh."""
        if not any(self._switches.on(name) for name in FLOOR_STAGES):
            return
        if self._tilt is None:
            if msg.header.frame_id == "base_link":
                rotation = np.eye(3)  # the bridge rotated it already: see _imu_rotation
            elif self._imu_mount is not None:
                rotation = self._imu_mount
            else:
                # Without the mount the up vector would be the chip's own axes, and the floor
                # would be anchored to a plane tilted by however the chip is glued on.
                for name in FLOOR_STAGES:
                    if self._switches.on(name):
                        self._switches.set(name, False)
                self.get_logger().error(
                    f"IMU readings in {msg.header.frame_id} and no mount: the floor stages are off"
                )
                return
            self._tilt = Tilt(rotation)
        accel, _gyro = imu_arrays(msg)
        self._tilt.observe(accel, stamp_seconds(msg.header.stamp))

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
        if self._fatal.leaving:
            return  # nothing can answer: no more frames on the way out
        if self._worker.offer(msg):
            self._tally.count("dropped")

    def _process(self, msg: Image) -> None:
        """One frame through the network, then the pipeline — the anchors, the law, the edges,
        the floor — the scan from the depth before the floor anchor, and out; or withheld by
        a law that does not exist yet."""
        rgb = array_from_image(msg)
        if rgb is None or rgb.ndim != 3:
            self.get_logger().warning(
                f"cannot read {msg.encoding} images", throttle_duration_sec=30
            )
            return
        tally = self._tally
        with tally.measure("network"):
            try:
                depth = self._net(rgb)
            except DepthModelError as exc:
                self._no_model(exc)
                return
        with tally.measure("pose"):  # includes the TF wait for the neck's edge
            cam = self._camera_at(msg.header.stamp)
        with tally.measure("samples"):  # includes the TF wait for the carry
            lidar = self._lidar_points(msg)
        ctx = FrameContext(
            self._intr_or_nominal(msg),
            cam,
            up=self._tilt.up if self._tilt is not None else UP_LEVEL,
            lidar=lidar,
            stamp=stamp_seconds(msg.header.stamp),
        )
        with tally.measure("pipeline"):
            result = self._pipeline.run(depth, ctx)
        tally.count("processed")
        judged = result.verdict("lidar_anchor").pairs
        if judged:
            tally.count("verdicts")
            tally.count("samples", judged)
            self._last_verdict_wall = time.time()
        else:
            tally.count("held")
        if result.withheld:
            tally.count("withheld")
            self.get_logger().info(
                f"no depth law yet ({self._law.pooled} of {POOL_MIN_SAMPLES} pairs pooled):"
                " nothing published",
                throttle_duration_sec=10,
            )
            return
        with tally.measure("scan"):  # before the floor anchor: what stops the cart is measured
            scan = self._as_scan(result.before(SCAN_BEFORE), msg, ctx)
        with tally.measure("publish"):
            self._pub.publish(
                image_from_array(result.depth, "32FC1", msg.header.stamp, msg.header.frame_id)
            )
            self._scan_pub.publish(scan)
        tally.count("frames")

    def _no_model(self, exc: DepthModelError) -> None:
        """The CPU model cannot be built (no cached weights and no hub, or no memory): the cause
        with its traceback the first time, the remembered sentence after. In ``local`` mode it
        is the only backend, so the node leaves as it did when the model was built in the
        constructor — exit code 1, the launch respawns it, the respawn retries the load. In
        ``auto`` the frame is lost, the service keeps being probed, the report line says why."""
        self._tally.count("no_model")
        detail = "".join(traceback.format_exception(exc)) if exc.__cause__ else f"{exc}"
        if self._net.mode == "local":
            self.get_logger().fatal(f"{detail}\nno other backend in local mode: leaving")
            self._fatal.leave(f"depth_stream: {exc}; local mode has no other backend")
            return
        self.get_logger().error(
            f"{detail}\nthe service is down too: frames are lost until it answers",
            throttle_duration_sec=30,
        )

    def _camera_at(self, stamp: Any) -> CameraPose:
        """Where the camera sat at ``stamp``: ``base_link <- camera_optical`` from TF (the
        neck's live edge, or the static one), as the pipeline's pose; config/camera.json's
        mount, counted, while TF has no such edge. A head turned past PAN_NOTICE_RAD is
        counted too: the projections assume it looks along the cart's x."""
        pose = self._poser.camera_in_base(stamp_seconds(stamp))
        if pose is None:
            self._tally.count("camera_from_config")
            return self._camera_config
        _pitch, pan = optical_heading(pose.rotation)
        if abs(pan) > PAN_NOTICE_RAD:
            self._tally.count("camera_panned")
        return CameraPose.from_optical(pose.rotation, pose.translation)

    def _as_scan(self, depth: Array, image: Image, ctx: FrameContext) -> LaserScan:
        """The depth folded onto the floor plane, in base_link, stamped like the image."""
        angle_min, step, ranges = depth_to_scan(
            depth, ctx.intr, ctx.cam, max_range=self._scan_max_range
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

    def _lidar_points(self, image: Image) -> Array | None:
        """The scan nearest the frame's exposure as (n, 3) base_link points at the frame's
        moment (the cart's own motion between the two stamps taken out through the odometry;
        without it a 100 ms older scan is 2 degrees stale at 20 deg/s, and the points pass as
        they are, counted): needs the camera's optics, a scan within SCAN_MAX_AGE_S and the
        lidar's mount; ``None`` otherwise, or with the lidar anchor off."""
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
        scan_t = stamp_seconds(scan.header.stamp)
        self._tally.sample("scan_age", abs(scan_t - frame_t))
        mount = self._mount_of(scan.header.frame_id)
        if mount is None:
            return None
        xy = scan_points(
            np.asarray(scan.ranges), scan.angle_min, scan.angle_increment, SCAN_RANGE_M
        )
        in_base = to_base(xy, mount.rotation, mount.translation)
        carried = self._poser.carry(in_base, scan_t, frame_t)
        if carried is None:
            self._tally.count("uncarried")
            return in_base
        return carried

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
        """The window's numbers in one line — the pipeline's own words per stage, the backend,
        the switches — the law saved, the counters and the stage totals reset."""
        w = self._tally.take()
        c = w.counts
        per_verdict = c["samples"] / max(c["verdicts"], 1)
        if self._switches.on("lidar_anchor") and c["verdicts"] == 0 and c["frames"]:
            self.get_logger().warning(
                "no lidar beam judged the depth in this window: the law is held"
                f" ({time.time() - self._last_verdict_wall:.0f} s old); is /scan alive?"
            )
        law = self._law
        self.get_logger().info(
            f"depth: {w.rate('frames'):.1f} frames/s published ({c['processed']} through the"
            f" net, {c['dropped']} dropped, {c['withheld']} withheld); {self._pipeline.report()};"
            f" {c['verdicts']} lidar verdicts ({per_verdict:.0f} pairs each; held {c['held']} of"
            f" {c['processed']} frames){self._extras(w)}, backend {self._net.status}"
            f"{self._model_note()}, flags: {self._switches.state()}, ms median/max:"
            f" {w.stages()}"
        )
        self._pipeline.reset_stats()
        if law.fitted:
            try:
                save_law(self._law_file, law.a, law.b, law.pooled, self._last_verdict_wall)
            except OSError as exc:
                self.get_logger().warning(
                    f"cannot save the depth law to {self._law_file}: {exc}",
                    throttle_duration_sec=300,
                )

    def _model_note(self) -> str:
        """The CPU model's state for the report line: nothing once it answers, ``not loaded``
        while no frame has asked for it, and why it failed when one did."""
        if self._local.built:
            return ""
        if self._local.failed:
            return f" (CPU model failed: {self._local.failed})"
        return " (CPU model not loaded)"

    def _extras(self, w: Window) -> str:
        """The parts of the report line a window may have nothing to say about: how old the
        anchoring scans were, the scans no odometry could carry, the frames whose camera pose
        came from the config instead of TF, the frames with the head turned, the cart's lean,
        and the last TF failure of each kind."""
        c, extra = w.counts, ""
        ages = w.samples.get("scan_age", [])
        if ages:
            extra += f", scan age median {float(np.median(ages)):.2f} s max {max(ages):.2f} s"
        if c["uncarried"]:
            extra += f", scans uncarried {c['uncarried']}"
        if c["camera_from_config"]:
            extra += f", camera pose from config {c['camera_from_config']} frames (no TF edge)"
        if c["camera_panned"]:
            extra += f", head panned {c['camera_panned']} frames (projected as if not)"
        if self._tilt is not None and any(self._switches.on(name) for name in FLOOR_STAGES):
            roll, pitch = self._tilt.roll_pitch_deg
            extra += f", lean roll {roll:+.1f} pitch {pitch:+.1f} deg"
        if w.notes:
            extra += ", tf: " + "; ".join(
                f"{kind} {c['tf_' + kind]}: {text}" for kind, text in w.notes.items()
            )
        return extra


def main() -> None:
    spin_main(DepthStream)


if __name__ == "__main__":
    main()
