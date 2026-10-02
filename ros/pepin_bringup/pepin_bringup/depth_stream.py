"""The camera as a depth sensor, on the laptop: images in, metric depth images out.

WHERE THE RAW DEPTH COMES FROM is one node parameter, ``depth_source``, and everything after it
is the same chain. ``stereo`` (the head on the robot) measures it; ``network`` (a single camera)
has Depth Anything V2 guess it, with the right shape and the wrong size. The lidar is the
witness: the scan taken nearest the frame's exposure, carried to the frame's moment through the
odometry and projected into the image through the mounts, names the true depth at the pixels it
hits, and those pixels fit an affine law in inverse depth (1 / z = a / D + b) pooled across
minutes of frames. That chain is a :class:`pepin.depth_pipeline.DepthPipeline`
(:func:`pepin.depth_pipeline.standard_pipeline`): the edge filter, the lidar anchor, the law, the
floor anchor — each switched by the flag of its name, each counted and timed per frame, the node
owning none of the arithmetic. Under ``network`` nothing is published until the law exists (the
raw network's depth is 1.5-2x too far): frames are withheld until POOL_MIN_SAMPLES beam pairs are
pooled, or until the law saved by the last run (``/maps/depth_law.json``, a day old at most) is
loaded at start. Under ``stereo`` the law WATCHES (``law_watch``): the depth goes out as measured,
and the law's numbers in the report line are the head's health. The monocular network's
scale-recovering stages (floor pairs, wall anchor, parallax anchor, range law, frame law) are on
the tag ``alt/mono-depth-2026-09-21``.

The depth as it stands before the floor anchor, cut between 8 cm and 1.3 m above the floor and
folded onto the plane, is ``/depth_scan`` (a LaserScan in base_link): the board's local costmap
marks and clears with it like with the lidar, so a table top stops the cart the way a wall does.
The floor anchor (pixels within centimetres of the floor plane snap to it, the plane leaning with
the accelerometer) is for the 3D model: the anchored depth goes out on ``/camera/depth`` (32FC1
metres, the image's stamp and frame). Frames that arrive while the source is busy are dropped:
the newest one wins. Every stage is timed and reported.

Where the camera sits is asked of TF at every frame's stamp (:class:`pepin.frame_pose.FramePoser`
over the kit's :class:`TfHistory`): the neck moves, and ``base_link -> camera_link`` is published
live from its encoders by the board's neck node; the last edge TF held stands in when the
frame's stamp cannot be had, and config/camera.json's mount (straight ahead) only while TF has
never had such an edge; the report line counts both. The same poser carries the scan to the
frame's moment through the odometry. Neither ask ever waits for an edge that has stopped: TF
goes through :class:`LiveEdgeHistory`, which refuses any blocking lookup of an edge whose newest
sample is more than ``tf_dead_s`` behind the frame — the last edge and the uncarried scan at
once, both counted, instead of CARRY_WAIT_S burnt per frame on a route that has died
(2026-09-16: the neck's edge 344 s old, 0.9-3 frames/s). The pan is part of the camera's pose
(:class:`pepin.depth.CameraPose`'s yaw): the lidar's beams are projected into the picture where
the head looks, and the published fan turns with it — the bearings of /depth_scan are the cart's
whichever way the neck looks, and the fan's angular window sits off base_link's x by the pan.

THE STEREO SOURCE: the node also subscribes to ``/camera/right/image`` and
``/camera/right/camera_info``, pairs the right eye with the left picture by EXACT stamp (both
halves of one transport frame carry the same one, so a right eye that has not arrived within
``stereo_pair_wait_s`` is never coming and the frame is dropped and counted), and
:class:`pepin.stereo_depth.StereoDepth` measures the depth — ``z = fx * baseline / disparity``,
metric by construction, NaN wherever the match is not trusted and NaN past the rig's own reach
(2.2 m, where the disparity error model crosses 10 cm; see that module). ``fx`` and the baseline
are read off the two ``camera_info`` messages, so this node never opens the calibration file. The
picture, its stamp and its frame are the left eye's throughout.

WHICH ENGINE matches the two eyes is the live ``stereo_matcher`` flag in {sgbm, raft}, default
``raft``. Both are built at start and the flag points the source at one of them, so an A/B costs
no restart and nothing is rebuilt mid-drive. ``raft`` is RAFT-Stereo on the laptop's GPU, in the
same native host process the mono network uses (:mod:`pepin.stereo_host`, ``ros/depth_host.sh
stereo``): it turns the reflection phantoms SGBM paints on the parquet from 450 separate specks
into 94 blobs and costs 89 ms a pair against 17; it is the default since 2026-09-23.
A pair the host cannot answer goes to SGBM, and the report line counts it and says how long the
host has been down.

The network runs where ``depth_backend`` says: ``local`` is the CPU model in this container
(0.2-0.3 s a frame), ``remote`` the same network on the laptop's GPU behind
:mod:`pepin.depth_service` (ros/depth_host.sh, 26 ms a round trip), ``auto`` the service while
it answers and the CPU model while it does not (:class:`pepin.depth_service.Fallback`). The CPU
model is built on its first frame, not at start: a node on the service never pays its gigabyte.
A model that cannot be built (no cached weights and no hub, no memory) is not tried again: in
``local`` mode the node leaves as it did when the model failed in the constructor — exit code
1, the launch respawns it, the respawn retries — and in ``auto`` the frame is lost until the
service answers; the report line says which.

How far the cart leans is one estimator's (``pepin.lean`` through the kit's ``LeanFeed``, off
``/imu/data_raw``): the floor plane's up vector, and — with ``imu_lean`` on — the lean the poser
composes into the scan's carry and the camera's place, so a body tipped over a slipper does not
place its frame as if it stood level. A lean gravity did not vote for (``lean_min_quality``,
the signature of a drifting gyro rather than of a tipping body) is treated as no lean at all.

The flags and knobs (:data:`FLAGS` and config/knobs.json, ``ros/flags.sh set depth_stream <flag>
<value>``): one per stage of the pipeline — ``edge_filter``, ``lidar_anchor``, ``affine_law``,
``floor_anchor`` — plus ``depth_backend``, ``stereo_matcher``, ``scale_ceiling``, the largest 1 /
scale the law may be fitted to, ``tf_dead_s``, how stale a TF edge may be before no frame waits for
it, ``imu_lean``, ``lean_min_quality`` and ``scan_hz``, the cap on how often ``/depth_scan`` is
published (5 Hz, the board's local costmap's own ``update_frequency`` — every frame is still
processed, the cap is on the publisher); their state is printed in every report line.

THE GAZE GATE (``gaze_gate`` and the ``gate_*`` knobs, :mod:`pepin.gaze_gate`): a picture whose
exposure overlaps a head saccade (``/gaze/state``) or a body yaw above ``gate_yaw_dps`` is dropped
before the worker is offered it, so it costs no network time and every consumer of the depth —
depth_fusion's volume, contact_scan, the costmap's clearing fan, rgbd_odometry — simply never
receives it. Counted per window in the report line; without ``/gaze/state`` nothing is dropped.
"""

from __future__ import annotations

import math
import os
import threading
import time
import traceback
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, LaserScan

from pepin.camera import CameraConfig, mount_transform, optics
from pepin.contact import fan_min_z
from pepin.depth import (
    MIN_SAMPLES,
    POOL_MIN_SAMPLES,
    SCAN_WINDOW_S,
    Array,
    CameraPose,
    Intrinsics,
    carry,
    carry_speed,
    depth_to_scan,
    floor_depth,
    load_law,
    nearest_stamp,
    optical_heading,
    plane_in_view_from,
    save_law,
    scan_points,
    set_scale_ceiling,
    to_base,
)
from pepin.depth_pipeline import (
    AffineLaw,
    FrameContext,
    standard_pipeline,
)
from pepin.depth_service import (
    DEFAULT_URL,
    MODES,
    DepthBackend,
    DepthModelError,
    Fallback,
    LazyDepth,
    RemoteDepth,
)
from pepin.flags import Flag, FlagSet, load_knobs, with_knobs
from pepin.frame_pose import FramePoser, settled_pose
from pepin.gaze_gate import GATE_KNOBS, GAZE_GATE
from pepin.stereo_depth import (
    MATCHERS as STEREO_MATCHERS,
)
from pepin.stereo_depth import (
    Baseline,
    DisparityMatcher,
    MatcherSettings,
    RaftMatcher,
    StereoDepth,
    StereoMatcher,
    StereoUnavailableError,
)
from pepin.tsdf import RigidPose
from pepin_bringup.msgs import (
    array_from_image,
    image_from_array,
    scan_from_ranges,
    stamp_seconds,
)
from pepin_bringup.node_kit import (
    Fatal,
    GazeFeed,
    LeanFeed,
    Switches,
    Tally,
    TfHistory,
    TfLookup,
    Window,
    Worker,
    gate_counts,
    spin_main,
)

CONFIG = "/ws/config/camera.json"
LAW_FILE = "/maps/depth_law.json"  # ros/maps on the laptop, mounted at /maps by ros/laptop.sh
STEREO_LAW_FILE = "/maps/depth_law_stereo.json"  # a metric source keeps its own law, never the
# network's: the mono law's a 1.28 applied to a depth that is already metres is a quarter too far
DEPTH_SOURCES = ("network", "stereo")  # where a frame's RAW depth comes from; a node parameter
RIGHT_IMAGE = "/camera/right/image"
RIGHT_INFO = "/camera/right/camera_info"
# How long a left picture waits for the right eye of its own stamp. Both are published from one
# transport frame, so this is the transport's jitter and nothing else: 50 ms is ten times the
# measured map->odom period and a right eye later than that is not coming.
PAIR_WAIT_S = 0.05
PAIR_BUFFER = 8  # right eyes held while their left halves are matched; the newest wins anyway
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SCAN_RANGE_M = 6.0
STAGES = ("network", "pose", "samples", "pipeline", "scan", "publish")
CARRY_WAIT_S = 0.2  # how long TF is given to cover a frame's stamp (the carry, the camera pose)
CAMERA_TF_MAX_AGE_S = 1.0  # the neck's newest edge is the head's pose while it is at most this old
TF_DEAD_S = 3.0  # an edge whose newest sample is older than this is dead: no frame waits for it
PAN_NOTICE_RAD = math.radians(1.0)  # a head turned more than this is worth a line in the report
SCAN_BEFORE = "floor_anchor"  # the scan is built from the depth as it stands before this stage
# How far this camera answers for its own depth, in metres. ONE number for two uses — the
# published image's reach (:meth:`DepthStream._vouched`) and /depth_scan's own cap
# (``scan_max_range``) — because they are the same physical claim, and two literals would drift:
# the costmap's obstacle_max_range of 2.5 m has to stay under the scan's cap, and the camera half
# of RTAB-Map's grid has to stay under the image's. The rig's own reach (pepin.stereo_depth's
# DEPTH_SIGMA_M, ~4 m on this rig since 2026-09-30) binds first: the scan's cap is the smaller of
# the two. Measured: see the ``depth_reach_m`` flag.
DEPTH_REACH_M = 4.0

# What a METRIC source changes in the chain, by flag name (the rest keep FLAGS' defaults): the
# lidar's pairs are still collected and the affine law still fitted, as a witness
# (``law_watch``). The scale-recovering stages the mono network had are on the tag
# alt/mono-depth-2026-09-21.
STEREO_DEFAULTS: dict[str, Any] = {
    "law_watch": True,
}

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
        why="the band probe found 21 % of the band's pixels more than 15 cm from any lidar return,"
        " with no bearing trend to blame the focal length or the mount yaw on — flying pixels; the"
        " 8 % step that drops them costs 1.2-1.6 ms a frame and 2 % of the pixels"
        " (scratch/pipeline_vs_truth.txt, scratch/band_frame_probe.py). The cause is measured, the"
        " benefit is not: the one A/B on the robot, one turn each way, showed no difference",
        on_when="whenever the scan feeds a costmap: the pixels it drops are the ones that become"
        " an obstacle with nothing behind them",
        off_when="to see the raw band's tail in Foxglove, or when a thin real object (a chair leg"
        " at range) is missing from the scan and the 8 % step is the suspect",
    ),
    Flag(
        "lidar_anchor",
        True,
        description="the lidar's returns pair with the network's depth and fit the law; off, the"
        " last law is held (the failure mode of a lidar that stops) — with no law yet nothing is"
        " published until it is back on",
        why="the beams are the only metric ruler on board. Without them a floor-only fit reads the"
        " lidar's own row 1.98-2.46x too far and the raw network 1.62-1.96x"
        " (scratch/pipeline_vs_truth.txt, scratch/lidar_height_check.txt), and the pairing costs"
        " 0.0-0.1 ms a frame. The one on/off turn on the robot is split: a held law was better"
        " above the band (9.6-19.3 cm against 16-46 cm) and worse at it (12.0 cm against 7-9 cm),"
        " and the band is the row the costmap drives on",
        on_when="whenever the lidar spins — it is what makes the network's depth metric. It can"
        " only judge past the range at which its own plane enters the picture"
        " (pepin.depth.plane_in_view_from: 0.71 m with the head 23.7 deg down, the lens 0.82 m"
        " above the plane, a 640x360 frame). Parked closer than that — the working case at a"
        " desk — not one beam lands in the image, the report says so (lidar plane out of the"
        " picture N frames) instead of asking whether /scan is alive, and the law is held",
        off_when="to rehearse a lidar that dies mid-run (the law freezes, nothing else changes),"
        " or to compare the slices above the band, where the frozen law measured better",
    ),
    Flag(
        "affine_law",
        True,
        description="the network's depth through 1 / z = a / D + b, fitted on the pooled pairs;"
        " off, the raw network's depth goes out unwithheld",
        why="the raw network is 1.6-2.0x too far — the lidar's own row reads 1.958 raw against"
        " 1.033 through the law, and the band against the beams 35.5/112.0 cm against 3.8/20.6 cm,"
        " 14 % -> 57 % of it within 5 cm (scratch/lidar_height_check.txt). The fit costs 0.4 ms",
        on_when="always, to drive",
        off_when="as an A/B measure of the correction, at rest — never a way to drive: every"
        " published metre is then 1.6-2.0x long",
    ),
    Flag(
        "floor_anchor",
        True,
        description="pixels within centimetres of the floor plane snap to it in the published"
        " image (the scan is built before it); the plane leans with the cart, from the IMU's up"
        " vector",
        why="measured on the robot, one turn each way — the floor's sd 5.8 -> 3.3 cm and 43 % ->"
        " 71 % of it within 3 cm, for 0.4-0.6 ms a frame. The scan is built before this stage on"
        " purpose: at 5.8 cm of patchy floor noise a probe called 63 of 161 bearings lethal, which"
        " is where the scan's 0.15 m floor cut comes from",
        on_when="whenever the floor should read as floor in the published depth and in"
        " /fusion/surface",
        off_when="to measure the raw floor's noise again (the number the 0.15 m scan cut was"
        " chosen from), or where the floor is not a plane — a ramp, a threshold — and snapping"
        " would invent one",
    ),
    Flag(
        "depth_backend",
        "local",
        description="where the network runs: local (the CPU model in this container), remote (the"
        " laptop's GPU service, ros/depth_host.sh), auto (the service while it answers, the CPU"
        " model while it does not)",
        why="default by design, unmeasured as a choice: local is the value that needs nothing else"
        " running, and ros/laptop.sh exports PEPIN_DEPTH_BACKEND=auto whenever the laptop's Metal"
        " backend answers, so the field default is auto. What the choice is worth is measured:"
        " Depth Anything V2 Small is 20.6 ms a frame on MPS against 172-204 ms on the container's"
        " CPU, 26 ms end to end from the container through the JPEG service — 6.6x"
        " (scratch/depth_backend_bench.py), and on the robot the remote backend published 9.5 fps"
        " with 0 frames falling back to local",
        on_when="remote while the GPU service is up and the frame rate matters; auto for a run"
        " that must survive the service dying mid-drive",
        off_when="local when the laptop's service is not there or is being restarted, or to"
        " measure the container's own worst case (5.8 fps)",
        choices=MODES,
        env="PEPIN_DEPTH_BACKEND",
    ),
    Flag(
        "stereo_matcher",
        "raft",
        description="which engine turns the two eyes into a disparity: sgbm (OpenCV's semi-global"
        " block matcher, in this container) or raft (RAFT-Stereo on the laptop's GPU through the"
        " same host the mono network uses, ros/depth_host.sh stereo). Both are built at start and"
        " this picks which one answers the next pair, so an A/B needs no restart; a pair the host"
        " cannot answer falls to sgbm and the report line counts it. Under depth_source: network"
        " it does nothing",
        why="raft since 2026-09-23, when the drive settled it: six legs of that day (three with"
        " the lidar, three camera-only) all ran on raft at 101-111 ms a pair, 5.8 published"
        " frames/s, 71 % of the pixels valid, and reached every mark; a restart that fell back"
        " to the old sgbm default was a silent change of the sensor under the next test. What"
        " raft buys was measured before that through THIS"
        " path on 2026-09-22, on the 8 rectified pairs of scratch/stereo_net/frames/pairs.npz"
        " (host_smoke.py, sgbm_vs_raft.py, phantom_where.py): the airborne phantom PIXELS the"
        " lamp's reflection on the parquet puts 0.5-1.5 m up and under 2.5 m ahead fall only"
        " 6651 -> 5265, and raft is the worse of the two on three of the eight pairs — but what a"
        " costmap is told falls 450 separate specks -> 94 blobs (192 -> 37 of 50 px or more),"
        " halves in the robot's own path (4486 -> 2566), and the floor comes back whole: 4.7x as"
        " many points within 5 cm of the fitted plane. What it costs is measured too: 89.2 ms a"
        " pair end to end through the host against SGBM's 16.8 here, so 11 fps against the"
        " camera's 10 and the scan_hz cap's 5 — the whole budget, with the pose and the pipeline"
        " still to pay for, and the routers not yet in the path. The drive settles it",
        on_when="on the parquet, where SGBM's reflection phantoms are the thing marking the"
        " costmap, and with ros/depth_host.sh stereo up — watch the report's ms a pair and the"
        " published frames/s before trusting it",
        off_when="whenever the frame rate matters more than the phantoms, when the host is not"
        " running (it falls back by itself, but paying a round trip per pair to be refused is"
        " not free), and as the A/B half of any claim about either engine",
        choices=STEREO_MATCHERS,
        env="PEPIN_STEREO_MATCHER",
    ),
    Flag(
        "law_watch",
        False,
        description="the affine law is fitted on the lidar's pairs and printed, and the depth is"
        " published exactly as the source measured it; no frame waits for a law",
        why="off for the network, whose depth is 1.6-2.0x long until the law corrects it. ON"
        " under depth_source stereo (STEREO_DEFAULTS): a calibrated head is metric by"
        " construction — the checkerboard calibration of 2026-09-21 reads a printed board's"
        " span to +0.8 % on frames it never saw (scratch/stereo/board_metric_check.py) — and the"
        " law fitted on a cluttered room pulled b to its -0.200 bound, because the lidar's plane"
        " is 0.8 m under the lens and its far returns project onto whatever stands in front of"
        " them. Watching, the same fit is the head's health line: a 1.00 while the rig is as"
        " calibrated, anything else once it has been knocked. It costs the fit alone",
        on_when="the source is metric (stereo) and the lidar is a witness, not a ruler",
        off_when="the depth needs the lidar's scale (the mono network), or as an A/B of what the"
        " law would do to a stereo depth",
    ),
    Flag(
        "imu_lean",
        True,
        description="the cart's lean (pepin.lean, from /imu/data_raw) is followed with the gyro as"
        " well as the accelerometer and carried into the scan's carry and the camera's place in"
        " the map; off, the floor plane leans with the accelerometer alone, as it always has, and"
        " nothing else is leaned",
        why="on: the gyro's sign was verified by hand on 2026-09-13 (the cart tipped nose-down"
        " read pitch +10.5 deg, left-side-down read roll -9.4 deg, the fast path following at"
        " once with quality 0.9 while held; scratch/lean_tip_test.txt), and the gyro's zero"
        " offset is learned. Off, the floor anchor keeps the accelerometer-only lean, which by"
        " design ignores any tip shorter than 10 s",
        on_when="after a hand tip through a known angle shows the reported lean following it the"
        " right way and returning to zero",
        off_when="the moment the lean in the report line disagrees with the cart's visible"
        " attitude",
    ),
    # A frame taken during a saccade or a fast body yaw never reaches the network: no depth, no
    # /depth_scan, nothing for depth_fusion, contact_scan or rgbd_odometry to drop downstream.
    GAZE_GATE,
)
FLOOR_STAGES = ("floor_anchor",)  # the stages that read the IMU's up vector


def flags_for(source: str) -> FlagSet:
    """The node's flags with the defaults of its depth source: :data:`FLAGS` as declared for the
    mono network, the same flags with :data:`STEREO_DEFAULTS` for a metric stereo head."""
    if source != "stereo":
        return FlagSet(*FLAGS)
    return FlagSet(
        *(
            replace(flag, default=STEREO_DEFAULTS[flag.name])
            if flag.name in STEREO_DEFAULTS
            else flag
            for flag in FLAGS
        )
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


@dataclass(frozen=True)
class Views:
    """The pictures of one moment a depth source may read: the left eye — which is the mono
    path's only picture — and, on a stereo head, the right eye of the very same stamp."""

    rgb: npt.NDArray[np.uint8]
    right: npt.NDArray[np.uint8] | None = None


class DepthSource(Protocol):
    """Whatever turns the views of one moment into a RAW depth image in metres, before any
    correction stage has seen it. The node asks one of these per frame and knows no more about
    where a metre came from."""

    name: str

    def __call__(self, views: Views) -> Array:
        """The raw depth of this moment, same size as the left picture, metres."""
        ...

    def report(self) -> str:
        """This source's clause of the node's report line."""
        ...


class NetworkSource:
    """The mono network as a depth source: the left picture alone through the backend switch,
    the right eye ignored — bit for bit what the node did before there was a second source.

    The switch itself stays the node's (``depth_backend`` moves its mode live and the node's
    own error handling reads it), so this asks for it per frame instead of holding a second
    reference that a live change would not reach."""

    name = "network"

    def __init__(
        self, backend: Callable[[], DepthBackend], note: Callable[[], str] = lambda: ""
    ) -> None:
        self._backend, self._note = backend, note

    def __call__(self, views: Views) -> Array:
        """The network's depth of the left picture."""
        depth: Array = self._backend()(views.rgb)
        return depth

    def report(self) -> str:
        """Which backend answered and what the CPU model is doing."""
        return f"backend {getattr(self._backend(), 'status', '?')}{self._note()}"


class StereoSource:
    """The stereo head as a depth source: the two eyes of one moment through
    :class:`pepin.stereo_depth.StereoDepth`, which measures metres instead of guessing them.

    A frame with no right eye never reaches here — the node pairs by stamp and counts what it
    cannot pair — so the one thing this refuses is a rig whose ``camera_info`` has not arrived."""

    name = "stereo"

    def __init__(self, depth: StereoDepth) -> None:
        self.depth = depth

    def __call__(self, views: Views) -> Array:
        """The measured depth of this pair; raises when the right eye or the rig is missing."""
        if views.right is None:
            raise StereoUnavailableError("no right eye for this frame")
        out: Array = np.asarray(self.depth(views.rgb, views.right), dtype=float)
        return out

    def report(self) -> str:
        """The matcher, its milliseconds, the share of pixels answered for and the rig's range."""
        return self.depth.describe()


class EdgeHistory(Protocol):
    """What :class:`LiveEdgeHistory` needs of a TF history: the ask that waits, the ask that
    does not, and the newest edge there is (:class:`pepin_bringup.node_kit.TfHistory`)."""

    def pose_at(self, stamp: float, frame: str, fixed: str) -> RigidPose | None:
        """``fixed <- frame`` at ``stamp`` (seconds), waiting for the buffer to cover it."""
        ...

    def pose_at_nowait(self, stamp: float, frame: str, fixed: str) -> RigidPose | None:
        """``fixed <- frame`` at ``stamp`` from what the buffer already holds; never waits."""
        ...

    def latest_pose(self, frame: str, fixed: str) -> tuple[RigidPose, float] | None:
        """The newest ``fixed <- frame`` there is and the stamp it holds for; never waits."""
        ...


class LiveEdgeHistory:
    """TF for a frame's path: the history a :class:`pepin.frame_pose.FramePoser` asks, which
    refuses to WAIT for an edge that is already dead.

    A lookup TF cannot answer costs its whole timeout, and on a frame's path that is paid once
    per frame. Live on 2026-09-16 the board's TF route died: ``base_link <- camera_optical``
    stopped 344 s back, and every frame still spent CARRY_WAIT_S waiting for a stamp no
    publisher was going to fill (pose 212/226 ms, the stream at 0.9-3 frames/s) before falling
    to the config mount it could have used at once. So: an edge whose newest sample sits more
    than ``dead_s`` seconds behind the asked-for stamp is dead, :meth:`pose_at` answers
    ``None`` immediately, ``on_dead(frame, fixed, stale_s)`` counts it, and the caller's own
    fallback — the config mount, the uncarried scan — runs on this frame instead of the next.

    An edge TF has never carried at all is not dead but unborn: the wait stands there, because
    the publisher may be one message away (that case is the old ``no TF edge``). The live case
    the wait exists for — a fresh edge that does not yet cover this frame's stamp — is
    untouched, and ``dead_s`` 0 turns the guard off altogether. Every non-blocking ask passes
    through unjudged: they cost nothing whatever the route does.
    """

    def __init__(
        self,
        history: EdgeHistory,
        *,
        dead_s: Callable[[], float],
        on_dead: Callable[[str, str, float], None],
    ) -> None:
        self.history = history
        self._dead_s = dead_s
        self._on_dead = on_dead

    def pose_at(self, stamp: float, frame: str, fixed: str) -> RigidPose | None:
        """``fixed <- frame`` at ``stamp``, waiting for TF to cover the moment — unless that
        edge is dead, when the answer is ``None`` at once and the wait is never paid."""
        stale = self.stale(stamp, frame, fixed)
        if stale is not None:
            self._on_dead(frame, fixed, stale)
            return None
        return self.history.pose_at(stamp, frame, fixed)

    def pose_at_nowait(self, stamp: float, frame: str, fixed: str) -> RigidPose | None:
        """``fixed <- frame`` at ``stamp`` from what the buffer holds; never waits, so never
        judged."""
        return self.history.pose_at_nowait(stamp, frame, fixed)

    def latest_pose(self, frame: str, fixed: str) -> tuple[RigidPose, float] | None:
        """The newest ``fixed <- frame`` TF holds and the stamp it is for; never waits."""
        return self.history.latest_pose(frame, fixed)

    def stale(self, stamp: float, frame: str, fixed: str) -> float | None:
        """How many seconds behind ``stamp`` the newest ``fixed <- frame`` edge sits when that
        is more than ``dead_s`` — the edge is dead and no wait can cure it — and ``None`` while
        it is alive, absent from TF altogether, or the guard is off (``dead_s`` 0)."""
        dead_s = self._dead_s()
        if dead_s <= 0.0:
            return None
        latest = self.latest_pose(frame, fixed)
        if latest is None:
            return None
        stale = stamp - latest[1]
        return stale if stale > dead_s else None


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
        # Where a frame's RAW depth comes from. A parameter, not a live flag: it decides which
        # topics this node subscribes to, and the launch that starts the stereo rig is the one
        # thing that knows which head is on the robot.
        source_name = str(self.declare_parameter("depth_source", DEPTH_SOURCES[0]).value)
        if source_name not in DEPTH_SOURCES:
            raise ValueError(f"depth_source must be one of {DEPTH_SOURCES}, not {source_name!r}")
        self._stereo_on = source_name == "stereo"
        cfg = CameraConfig.load(config, board=board)
        x, y, z, _roll, pitch, _yaw = mount_transform(cfg)
        self._camera_config = CameraPose(x, y, z, pitch)  # the fallback while TF has no edge
        self._last_cam = self._camera_config  # the head's last pose, for the report's geometry
        self._camera_cfg = cfg  # config/camera.json's own optics until a camera_info arrives
        self._intr: Intrinsics | None = None
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        newest = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(Image, "/camera/depth", reliable)
        self._scan_pub = self.create_publisher(LaserScan, "/depth_scan", reliable)
        self._scan_at = 0.0  # monotonic seconds of the last published fan: the scan_hz cap
        self._scan_max_range = float(self.declare_parameter("scan_max_range", DEPTH_REACH_M).value)
        self._expected_key: object = None  # the optics and head pose the floor ruler was cut for
        self._expected: Array = np.zeros((0, 0))
        # A law is a source's own: the mono network's a 1.28 applied to a depth that is already
        # metric would put every obstacle a quarter too far, and the other way round.
        default_law = STEREO_LAW_FILE if self._stereo_on else LAW_FILE
        self._law_file = Path(str(self.declare_parameter("law_file", default_law).value))
        # The depth service as this container sees it (host.docker.internal is the laptop);
        # read at start: the client reconnects by itself, the address does not move.
        depth_url = str(
            self.declare_parameter(
                "depth_url", os.environ.get("PEPIN_DEPTH_URL", DEFAULT_URL)
            ).value
        )
        self._law = AffineLaw()
        self._last_verdict_wall = time.time()  # the law's age is the beams', not the node's
        self._seed_laws(time.time())
        self._pipeline = standard_pipeline(self._law)
        # The stereo source's start-up parameters, declared HERE: once the flag kit below is
        # built, its parameter callback refuses every name that is not a flag ("not a flag of
        # this node"), so a parameter declared after it kills the node at start — it did, on the
        # robot's first stereo start (2026-09-21), and the ROS stubs of the unit tests cannot
        # show it.
        self._pair_wait_s = float(self.declare_parameter("stereo_pair_wait_s", PAIR_WAIT_S).value)
        self._stereo_reach_m = float(self.declare_parameter("stereo_reach_m", 0.0).value)
        # How long a pair waits for the GPU host before it falls to SGBM. 2 s is the mono
        # service's own timeout: the host answers a pair in 91 ms, so anything near this is the
        # host gone rather than the host slow, and three of them in a row give it up until the
        # next probe (pepin.stereo_depth.RaftMatcher).
        self._stereo_host_timeout_s = float(
            self.declare_parameter("stereo_host_timeout_s", 2.0).value
        )
        self._stereo_matcher_settings = self._matcher_settings()
        self._switches = Switches(
            self,
            with_knobs(flags_for(source_name), load_knobs("depth_stream")),
            on_change=self._on_switch,
        )
        self._law.watching = bool(self._switches["law_watch"])
        for name in self._pipeline.names:  # a launch override reaches the stage it names
            self._pipeline.set(name, self._switches.on(name))
        set_scale_ceiling(float(self._switches["scale_ceiling"]))  # and the law's bound
        self._tally = Tally(STAGES)
        self._lean = LeanFeed(
            self,
            config.parent,
            use_gyro=self._switches.on("imu_lean"),
            on_unmounted=self._no_imu_mount,
            enabled=self._leans_anything,
        )
        self._gaze = GazeFeed(
            self,
            exposure_s=float(self._switches["gate_exposure_s"]),
            settle_s=float(self._switches["gate_settle_s"]),
            yaw_dps=float(self._switches["gate_yaw_dps"]),
        )
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, reliable)
        self.create_subscription(Image, "/camera/image", self._on_image, newest)
        self.create_subscription(LaserScan, "/scan", self._on_scan, reliable)
        # The right eye, and the two numbers that turn a disparity into metres. Only under
        # ``depth_source: stereo``: a mono head publishes neither topic, and a subscription to a
        # topic nobody writes costs the transport a discovery entry for nothing.
        self._right_tx: float | None = None  # the right eye's P[0,3] = -fx * baseline
        self._right: deque[tuple[tuple[int, int], Image]] = deque(maxlen=PAIR_BUFFER)
        self._right_ready = threading.Condition()
        self._stereo: StereoDepth | None = None
        self._matchers: dict[str, DisparityMatcher] = {}
        if self._stereo_on:
            # BOTH engines are built here and the live stereo_matcher flag picks which one answers
            # the next pair: a matcher REBUILT mid-drive would change what the costmap is marked
            # from, but choosing between two that already exist does not, so an A/B needs no
            # restart. RaftMatcher holds nothing but an HTTP client until it is first called — the
            # network itself lives in the host process — so 'sgbm' costs no torch and no load.
            sgbm = StereoMatcher(self._stereo_matcher_settings)
            self._matchers = {
                "sgbm": sgbm,
                "raft": RaftMatcher(depth_url, sgbm, timeout_s=self._stereo_host_timeout_s),
            }
            self._stereo = StereoDepth(
                matcher=self._matchers[str(self._switches["stereo_matcher"])],
                reach=self._stereo_reach_m,
            )
            self.create_subscription(CameraInfo, RIGHT_INFO, self._on_right_info, reliable)
            self.create_subscription(Image, RIGHT_IMAGE, self._on_right, newest)
        self._tf = TfLookup(self, on_failure=self._on_tf_failure)
        # Every lookup of a frame's path goes through the guard: one that would wait for an
        # edge already dead (the board's TF route gone) is refused instead, and the caller's
        # fallback runs on this frame — the camera pose from the config, the scan uncarried.
        self._history = self._guarded(CARRY_WAIT_S)
        self._poser = self._new_poser(self._history)
        self._camera_frame, self._base_frame = self._poser.camera, self._poser.base
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
        # The one object the worker asks for a raw depth. The mono path is the same backend
        # switch it always was, wrapped; the stereo path measures instead.
        self._source: DepthSource = (
            StereoSource(self._stereo)
            if self._stereo is not None
            else NetworkSource(lambda: self._net, self._model_note)
        )
        self._fatal = Fatal(self)  # the worker's way out when no backend can answer
        self._worker = Worker(self._process, name="depth", on_error=self._on_work_error).start()
        self.create_timer(30.0, self._report)
        if self._stereo is not None:
            self.get_logger().info(
                f"raw depth from the stereo head, matcher"
                f" {self._switches['stereo_matcher']} of {', '.join(self._matchers)}"
                f" ({self._stereo.matcher.describe()}):"
                f" {RIGHT_IMAGE} paired with /camera/image by exact stamp,"
                f" {self._pair_wait_s * 1e3:.0f} ms of grace; fx and the baseline"
                f" come from {RIGHT_INFO}"
            )
        self.get_logger().info(
            "depth stream up: /camera/image -> /camera/depth, /depth_scan; camera pose from TF"
            f" (config/camera.json's pitch {math.degrees(pitch):.1f} deg while TF has no edge)"
        )

    def _swap_matcher(self, name: str) -> None:
        """The live ``stereo_matcher`` flag: point the stereo source at the OTHER engine, which
        already exists. Nothing is loaded or rebuilt here, so the swap costs one attribute and
        takes effect on the next pair. A mono node has no engines and this does nothing."""
        if self._stereo is None or name not in self._matchers:
            return
        self._stereo.matcher = self._matchers[name]
        self.get_logger().info(f"stereo matcher -> {name} ({self._stereo.matcher.describe()})")

    def _matcher_settings(self) -> MatcherSettings:
        """The stereo matcher's numbers as this launch set them: the module's measured defaults
        unless a parameter says otherwise. They are parameters and not live flags because a
        matcher rebuilt mid-drive would change what the costmap is being marked from."""
        default = MatcherSettings()
        return MatcherSettings(
            num_disparities=int(
                self.declare_parameter("stereo_num_disparities", default.num_disparities).value
            ),
            block_size=int(self.declare_parameter("stereo_block_size", default.block_size).value),
            mode=str(self.declare_parameter("stereo_mode", default.mode).value),
            downscale=int(self.declare_parameter("stereo_downscale", default.downscale).value),
            uniqueness_ratio=int(
                self.declare_parameter("stereo_uniqueness_ratio", default.uniqueness_ratio).value
            ),
            texture_threshold=float(
                self.declare_parameter("stereo_texture_threshold", default.texture_threshold).value
            ),
        )

    def close(self) -> None:
        """Stop the worker and the TF listener and wait for them: called before the node is
        destroyed, so no thread is left inside the network or DDS at interpreter exit."""
        if not self._worker.stop():
            self.get_logger().warning("the depth worker did not finish its frame; leaving anyway")
        self._tf.close()

    def _seed_laws(self, now: float) -> None:
        """Hand the affine law what the last run saved in the law file, and say so in the log.
        Without a file nothing is published until POOL_MIN_SAMPLES beam pairs are pooled."""
        saved = load_law(self._law_file, now)
        if saved is None and self._stereo_on:
            # A stereo depth is already metres. Withholding it until POOL_MIN_SAMPLES beams have
            # been pooled would make the lidar grant a licence it did not issue, and the first
            # minute of every run would publish nothing. The identity law is what "no correction
            # needed" is written as; the live fit replaces it as the beams arrive, and what it
            # fits to is then the READOUT of how metric this head is (a 1.00 b +0.000 is right).
            self._law.seed(1.0, 0.0)
            self.get_logger().info(
                f"no saved depth law at {self._law_file}: a stereo depth is metric already, so"
                " the identity law (a 1.00 b +0.000) stands until the beams fit one"
            )
            return
        if saved is None:
            self.get_logger().info(
                f"no saved depth law at {self._law_file}: publishing waits for"
                f" {POOL_MIN_SAMPLES} pooled beams"
            )
            return
        self._law.seed(saved[0], saved[1])
        self.get_logger().info(
            f"depth law from {self._law_file}: a {saved[0]:.2f} b {saved[1]:+.3f}"
            f" on {saved[2]} beams; publishing at once"
        )

    def _on_switch(self, name: str, _old: Any, new: Any) -> None:
        """A flag changed: ``depth_backend`` is the switch's mode, ``scale_ceiling`` the law's
        upper bound, ``imu_lean`` the poser's and the estimator's, ``lean_min_quality`` the
        poser's floor under a lean, a stage's flag switches that stage of the pipeline."""
        if name == "depth_backend":
            self._net.mode = str(new)
        elif name == "stereo_matcher":
            self._swap_matcher(str(new))
        elif name == "scale_ceiling":
            set_scale_ceiling(float(new))  # the next fit is bounded by it; the law in hand is not
        elif name == "law_watch":
            self._law.watching = bool(new)  # from the next frame on
        elif name == "imu_lean":
            self._poser.apply_lean = bool(new)
            self._lean.use_gyro = bool(new)
        elif name == "lean_min_quality":
            self._poser.min_lean_quality = float(new)
        elif name in GATE_KNOBS:
            self._gaze.set(name, float(new))
        elif name in self._pipeline.switches:
            self._pipeline.set(name, bool(new))

    def _on_work_error(self, text: str) -> None:
        self.get_logger().error(f"depth failed on a frame:\n{text}")

    def _guarded(self, timeout_s: float) -> LiveEdgeHistory:
        """TF with ``timeout_s`` to spare for a stamp it does not cover yet, behind the guard
        that refuses to spend it on an edge already dead (:class:`LiveEdgeHistory`)."""
        return LiveEdgeHistory(
            TfHistory(self._tf, timeout_s=timeout_s),
            dead_s=lambda: float(self._switches["tf_dead_s"]),
            on_dead=self._edge_dead,
        )

    def _new_poser(self, history: EdgeHistory) -> FramePoser:
        """A frame poser over ``history``, wearing the node's lean switches: the camera's pose
        and the scan's carry at a frame's stamp."""
        return FramePoser(
            history,
            lean=self._lean,
            apply_lean=self._switches.on("imu_lean"),
            min_lean_quality=float(self._switches["lean_min_quality"]),
        )

    def _on_tf_failure(self, kind: str, text: str) -> None:
        self._tally.count("tf_" + kind)
        self._tally.note(kind, text)

    def _edge_dead(self, frame: str, fixed: str, stale_s: float) -> None:
        """A wait :class:`LiveEdgeHistory` refused, counted under the name the report line
        calls that edge by — the neck's ``base_link <- camera_optical``, the odometry's
        ``odom <- base_link``, the tracker's ``map <- base_link``, anything else ``tf`` — with
        how stale it was, so the window says how long the route has been gone."""
        poser = self._poser
        name = {
            (poser.camera, poser.base): "neck",
            (poser.base, poser.odom_frame): "odom",
            (poser.base, poser.map_frame): "map",
        }.get((frame, fixed), "tf")
        self._tally.count(f"{name}_edge_dead")
        self._tally.sample(f"{name}_edge_stale_s", stale_s)

    def _leans_anything(self) -> bool:
        """Whether anything in this node wants the lean this second: a floor stage (the plane
        the anchors snap to) or ``imu_lean`` (the poser). Nothing does — the readings are not
        even decoded, and a stage switched back on picks the lean up afresh."""
        return self._switches.on("imu_lean") or any(
            self._switches.on(name) for name in FLOOR_STAGES
        )

    def _no_imu_mount(self, frame_id: str) -> None:
        """IMU readings outside base_link with no mount to turn them: the up vector would be
        the chip's own axes and the floor would be anchored to a plane tilted by however the
        chip is glued on, so the floor stages go off and the poser leans nothing."""
        for name in FLOOR_STAGES:
            if self._switches.on(name):
                self._switches.set(name, False)
        if self._switches.on("imu_lean"):
            self._switches.set("imu_lean", False)
        self.get_logger().error(
            f"IMU readings in {frame_id} and no mount: the floor stages are off"
        )

    def _on_info(self, msg: CameraInfo) -> None:
        """The camera's optics as they are published; the pipeline's projections are a pinhole."""
        self._intr = Intrinsics.from_camera_info(msg.k, msg.width, msg.height)
        self._stereo_geometry()

    def _on_right_info(self, msg: CameraInfo) -> None:
        """The right eye's rectified projection: ``P[0, 3]`` is ``-fx * baseline``, ROS's stereo
        convention, and it is the only place this node learns how far apart the eyes are."""
        self._right_tx = float(msg.p[3])
        self._stereo_geometry()

    def _stereo_geometry(self) -> None:
        """Hand the stereo source the rig's two numbers as soon as both ``camera_info`` messages
        have named them — the left eye's ``fx`` and the right eye's ``P[0, 3]`` — and say in the
        log what the rig can then measure and how far. Done once; a rig that describes itself
        wrongly is refused loudly rather than half-believed."""
        stereo = self._stereo
        if stereo is None or stereo.geometry is not None:
            return
        if self._intr is None or self._right_tx is None:
            return
        try:
            rig = Baseline.from_projection(self._intr.fx, self._right_tx)
        except StereoUnavailableError as exc:
            self.get_logger().error(
                f"the stereo rig cannot be read: {exc}", throttle_duration_sec=30
            )
            return
        stereo.geometry = rig
        # The fan must not announce a range the head never measures: the costmap clears an
        # "inf" bearing out to the scan's own range_max, and between the rig's reach and the
        # mono-era 3.0 m that would be free space nobody saw.
        self._scan_max_range = min(self._scan_max_range, float(stereo.reach))
        self.get_logger().info(
            f"stereo rig: fx {rig.fx:.1f} px, baseline {rig.baseline_m * 100:.2f} cm, so one"
            f" pixel of disparity is {rig.fx * rig.baseline_m:.1f} m — this head measures"
            f" {stereo.near:.2f} to {stereo.reach:.2f} m, and /depth_scan says so"
            f" (range_max {self._scan_max_range:.2f} m)"
        )

    def _on_right(self, msg: Image) -> None:
        """Keep the last few right eyes by their exact stamp, and wake whichever left picture is
        waiting for one of them. Bounded: the worker keeps only the newest left frame anyway, so
        a right eye whose left half was dropped is simply pushed out of the ring."""
        with self._right_ready:
            self._right.append(((int(msg.header.stamp.sec), int(msg.header.stamp.nanosec)), msg))
            self._right_ready.notify_all()

    def _right_at(self, stamp: Any) -> npt.NDArray[np.uint8] | None:
        """The right eye of EXACTLY this stamp, waiting up to ``stereo_pair_wait_s`` for it, or
        ``None``.

        Exact, not nearest: both eyes are cut from ONE transport frame and published with one
        stamp, so a near miss is a different exposure and pairing it would measure a disparity
        that is half parallax from the cart's own motion. The wait exists only because the two
        publications cross the transport separately and the left one can win the race."""
        key = (int(stamp.sec), int(stamp.nanosec))
        deadline = time.monotonic() + max(self._pair_wait_s, 0.0)
        with self._right_ready:
            while True:
                for held, msg in reversed(self._right):
                    if held == key:
                        right = array_from_image(msg)
                        return None if right is None or right.ndim not in (2, 3) else right
                left = deadline - time.monotonic()
                if left <= 0.0:
                    return None
                self._right_ready.wait(left)

    def _views(self, msg: Image) -> Views | None:
        """The pictures this frame's depth is measured from: the left one always, and under
        ``depth_source: stereo`` the right eye of the same stamp. ``None`` — the frame is
        dropped — for a picture this node cannot read or a right eye that never arrived."""
        rgb = array_from_image(msg)
        if rgb is None or rgb.ndim != 3:
            self.get_logger().warning(
                f"cannot read {msg.encoding} images", throttle_duration_sec=30
            )
            return None
        if not self._stereo_on:
            return Views(rgb)
        t0 = time.perf_counter()
        right = self._right_at(msg.header.stamp)
        self._tally.sample("pair_wait_s", time.perf_counter() - t0)
        if right is None:
            self._tally.count("unpaired")
            self.get_logger().warning(
                f"no right eye stamped like this picture within {self._pair_wait_s * 1e3:.0f} ms:"
                f" is {RIGHT_IMAGE} alive and does the camera publish both eyes with one stamp?",
                throttle_duration_sec=30,
            )
            return None
        if right.shape[:2] != rgb.shape[:2]:  # the size; the right eye is grey or colour
            self._tally.count("unpaired")
            self.get_logger().warning(
                f"the right eye is {right.shape[:2]} and the left {rgb.shape[:2]}: not one rig",
                throttle_duration_sec=30,
            )
            return None
        return Views(rgb, right)

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
        self._tally.count("images")
        if self._switches.on("gaze_gate"):
            # Judged before the worker is offered it: a frame of a saccade must not push out
            # the still frame waiting before it.
            verdict = self._gaze.verdict(stamp_seconds(msg.header.stamp))
            if verdict is not None:
                self._tally.count(f"gaze_{verdict}")
                return
        if self._worker.offer(msg):
            self._tally.count("dropped")

    def _process(self, msg: Image) -> None:
        """One frame through the network, then the pipeline — the anchors, the law, the edges,
        the floor — the scan from the depth before the floor anchor, and out; or withheld by
        a law that does not exist yet."""
        tally = self._tally
        with tally.measure("network"):  # the raw depth source: the network, or the stereo matcher
            views = self._views(msg)
            if views is None:
                return
            try:
                depth = self._source(views)
            except DepthModelError as exc:
                self._no_model(exc)
                return
            except StereoUnavailableError as exc:
                self._no_stereo(exc)
                return
        with tally.measure("pose"):  # includes the TF wait for the neck's edge
            cam, cam_optical = self._camera_at(msg.header.stamp)
        with tally.measure("samples"):  # includes the TF wait for the carry
            lidar = self._lidar_points(msg)
        # The camera edge goes in whole so the fan can turn with the neck's pan.
        ctx = FrameContext(
            self._intr_or_nominal(msg),
            cam,
            up=self._lean.up,
            lidar=lidar,
            stamp=stamp_seconds(msg.header.stamp),
            cam_optical=cam_optical,
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
            # Why this frame got no verdict, so the report does not blame a lidar that is
            # answering: the beams are cached on the context the pipeline just read, so asking
            # costs nothing. No scan at all is already counted by the absence of a scan_age.
            if lidar is not None:
                beams = ctx.beams
                blind = beams is None or beams.shape[0] == 0
                tally.count("beams_out_of_frame" if blind else "beams_too_few")
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
                image_from_array(
                    self._vouched(result.depth), "32FC1", msg.header.stamp, msg.header.frame_id
                )
            )
            if self._scan_due():
                self._scan_pub.publish(scan)
                tally.count("scans")
        tally.count("frames")

    def _scan_due(self) -> bool:
        """Whether ``/depth_scan`` may go out now, and the cap's clock moved on when it may
        (``scan_hz``; 0 is one fan per frame).

        The fan itself is measured either way — it is built before the floor anchor, out of a
        frame that has already been through the whole pipeline — so this holds back a
        publication and nothing else. Monotonic seconds: a cap on a publisher is not a
        measurement of anything in the room.
        """
        hz = float(self._switches["scan_hz"])
        if hz <= 0.0:
            return True
        now = time.monotonic()
        if now - self._scan_at < 1.0 / hz:
            self._tally.count("scans_thinned")
            return False
        self._scan_at = now
        return True

    def _vouched(self, depth: Array) -> Array:
        """The depth as this camera is willing to answer for it: NaN past ``depth_reach_m``, on a
        copy so nothing the pipeline still holds is touched (since 2026-09-19).

        A sensor answers for its own data: NaN is the one value that makes no point in any consumer
        downstream — no cell in RTAB-Map's grid (rtabmap/core/util3d.cpp:644), no mark and no
        raytrace in Nav2's obstacle layer, no voxel in the volume — so the range this network's
        scale stops being a measurement over is stated once, here, instead of in each consumer's
        own cap. Without it 44 % of the camera's costmap marks within 2.5 m were BEHIND the wall
        the lidar sees (run 0224, 2026-09-11), and it is what lets ONE Grid/RangeMax serve both
        sensors in vslam.launch.py."""
        reach = float(self._switches["depth_reach_m"])
        beyond = np.asarray(depth) > reach  # NaN compares false: what is already unknown stays so
        self._tally.count("beyond_reach", int(np.count_nonzero(beyond)))
        if not beyond.any():
            return depth
        vouched: Array = np.array(depth, copy=True)  # the pipeline's own dtype, not a cast
        vouched[beyond] = np.nan
        return vouched

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

    def _no_stereo(self, exc: StereoUnavailableError) -> None:
        """The stereo head cannot answer this frame — no ``camera_info`` from both eyes yet, or
        two eyes of different sizes. Nothing is published and nothing is fatal: the rig describes
        itself on a latched-enough topic and the next frame is likely to have it."""
        self._tally.count("no_stereo")
        self.get_logger().warning(
            f"the stereo head cannot answer: {exc}; is {RIGHT_INFO} alive?",
            throttle_duration_sec=30,
        )

    def _camera_at(self, stamp: Any) -> tuple[CameraPose, RigidPose | None]:
        """Where the camera sat at ``stamp``: ``base_link <- camera_optical`` from TF (the
        neck's live edge, or the static one) twice over — as the pipeline's pose (position,
        pitch and the pan as a yaw: every projection turns with the head) and as the edge
        itself, for the fan. A head turned past PAN_NOTICE_RAD is counted.

        Four asks, cheapest first, and only the third can wait: the frame's own stamp from what TF
        already holds; the newest edge, while it is younger than CAMERA_TF_MAX_AGE_S and the head
        has not moved over the half second before it (:func:`pepin.frame_pose.settled_pose` — a
        turning head is never stood in for by an old sample); the blocking lookup at the stamp —
        refused outright by :class:`LiveEdgeHistory` once the neck's edge is more than ``tf_dead_s``
        behind this frame, so a dead route costs no wait; and then the newest edge TF holds at any
        age, the head's last known pose, counted. config/camera.json's mount, which looks straight
        ahead, only while TF has never carried the edge at all."""
        at = stamp_seconds(stamp)
        pose = self._history.pose_at_nowait(at, self._camera_frame, self._base_frame)
        if pose is None:
            # The neck's edge crosses the bridge late (bursts of +0.75 s, 2026-09-15): a lookup
            # at the frame's own stamp waited CARRY_WAIT_S on every frame ("Extrapolation ...
            # into the future" x104 a window, 3.5 frames/s, VO starved). The newest edge of a
            # head that stands still is the head's pose; of a turning one it is not.
            pose = settled_pose(
                self._history,
                at,
                self._camera_frame,
                self._base_frame,
                max_age_s=CAMERA_TF_MAX_AGE_S,
            )
            if pose is not None:
                self._tally.count("camera_tf_latest")
        if pose is None:
            pose = self._poser.camera_in_base(at)  # the old path: wait for the stamp
        if pose is None:
            latest = self._history.latest_pose(self._camera_frame, self._base_frame)
            if latest is not None:
                pose = latest[0]  # where the head was last seen, never "straight ahead"
                self._tally.count("camera_tf_stale")
        if pose is None:
            self._tally.count("camera_from_config")
            self._last_cam = self._camera_config
            return self._camera_config, None
        _pitch, pan = optical_heading(pose.rotation)
        if abs(pan) > PAN_NOTICE_RAD:
            self._tally.count("camera_panned")
        self._last_cam = CameraPose.from_optical(pose.rotation, pose.translation)
        return self._last_cam, pose

    def _fan_pan(self, ctx: FrameContext) -> float:
        """How far left the head looks while the fan is folded, radians CCW from the cart's x:
        the yaw of the very ``base_link <- camera_optical`` edge the volume path took for this
        frame (:meth:`_camera_at`). With no such edge the frame's pose is config/camera.json's
        mount, whose yaw is zero — straight ahead — and the report line's ``camera pose from
        config`` counter is the count of those frames. A head panned 20 deg is 20 deg of costmap,
        a metre sideways at 3 m, which the fan of before 2026-09-15 ignored."""
        if ctx.cam_optical is None:
            return 0.0
        return optical_heading(ctx.cam_optical.rotation)[1]

    def _as_scan(self, depth: Array, image: Image, ctx: FrameContext) -> LaserScan:
        """The depth folded onto the floor plane, in base_link, stamped like the image, its
        bearings turned by the neck's pan, and the floor kept out of it by raising the band's
        lower edge with the floor's own noise (:func:`pepin.contact.fan_min_z`, 3 sigma of it):
        a floor pixel stands camera_height * (relative depth error) above the floor at every
        range, so with the flat 0.15 m edge the floor marked itself (scratch/fan_floor_leak.py,
        scratch/fan_gate_offline.py: k = fan / lidar 0.499 -> 0.595 at 25.8 deg)."""
        floor_of: float | Array = fan_min_z(self._floor_expected(ctx), ctx.cam.z)
        angle_min, step, ranges = depth_to_scan(
            depth,
            ctx.intr,
            ctx.cam,
            pan=self._fan_pan(ctx),
            min_z=floor_of,
            max_range=self._scan_max_range,
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

    def _floor_expected(self, ctx: FrameContext) -> Array:
        """The depth every pixel would have if its ray ended on the floor, cached per optics and
        head pose: the band gate's ruler (:func:`pepin.contact.fan_min_z`), one mgrid over the
        picture that must not be paid for on every frame while the head stands still."""
        key = (ctx.intr, ctx.cam, float(ctx.up[0]), float(ctx.up[1]), float(ctx.up[2]))
        if self._expected_key != key:
            self._expected = floor_depth(ctx.intr, ctx.cam, ctx.up)
            self._expected_key = key
        return self._expected

    def _intr_or_nominal(self, image: Image) -> Intrinsics:
        """The camera_info's optics, or config/camera.json's own until one arrives — the
        checkerboard's calibration scaled to this frame's size when the file carries one, the
        nominal pinhole of the configured field of view when it does not.

        One reader for both (:func:`pepin.camera.optics`), so a calibration reaches the depth's
        fallback the moment ros/calibrate.sh writes it, with no second place to remember. The
        distortion is dropped here on purpose: these projections are a pinhole, and the
        rectified picture camera_stream can publish is the place that answers for the lens.
        """
        if self._intr is not None:
            return self._intr
        lens = optics(self._camera_cfg, image.width, image.height)
        return Intrinsics(lens.fx, lens.fy, lens.cx, lens.cy, lens.width, lens.height)

    def _lidar_points(self, image: Image) -> Array | None:
        """The scan nearest the frame's exposure as (n, 3) base_link points at the frame's
        moment (the cart's own motion between the two stamps taken out through the odometry;
        without it a 100 ms older scan is 2 degrees stale at 20 deg/s, and the points pass as
        they are, counted): needs the camera's optics, a scan within SCAN_MAX_AGE_S and the
        lidar's mount; ``None`` otherwise, or with the lidar anchor off — and ``None`` as well
        when the carry itself is impossible (``carry_max_speed_mps``), so a runaway odometry
        frame cannot drag the beams into the picture and refit the law from them.

        The carry is asked of TF through :class:`LiveEdgeHistory`, so an odometry edge more
        than ``tf_dead_s`` behind this frame is not waited for either: the points pass as they
        are, counted as uncarried, with the dead edge named in the report line."""
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
        moved = self._poser.motion(scan_t, frame_t)
        if moved is None:
            self._tally.count("uncarried")
            return in_base
        # A carry the cart could not have driven (pepin.depth.carry_speed): the odometry is
        # lying, and these beams would land on the wrong pixels and refit the law from them.
        if carry_speed(moved.translation, frame_t - scan_t) > self._switches["carry_max_speed_mps"]:
            self._tally.count("carry_insane")
            return None
        return carry(in_base, moved.rotation, moved.translation)

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
                f" ({time.time() - self._last_verdict_wall:.0f} s old); {self._blind_because(c)}"
            )
        law = self._law
        self.get_logger().info(
            f"depth: {w.rate('frames'):.1f} frames/s published ({c['processed']} through the"
            f" net, {c['dropped']} dropped, {c['withheld']} withheld); {self._scan_line(w)};"
            f" {self._pipeline.report()};"
            f" {c['verdicts']} lidar verdicts ({per_verdict:.0f} pairs each; held {c['held']} of"
            f" {c['processed']} frames){self._extras(w)}, source {self._source.name}:"
            f" {self._source.report()}, {self._gate_line(w)}, flags: {self._switches.state()},"
            f" ms median/max: {w.stages()}"
        )
        self._pipeline.reset_stats()
        if law.fitted:
            try:
                save_law(
                    self._law_file,
                    law.a,
                    law.b,
                    law.pooled,
                    self._last_verdict_wall,
                )
            except OSError as exc:
                self.get_logger().warning(
                    f"cannot save the depth law to {self._law_file}: {exc}",
                    throttle_duration_sec=300,
                )

    def _gate_line(self, w: Window) -> str:
        """What the gaze gate kept from the network this window, and the head's state."""
        if not self._switches.on("gaze_gate"):
            return "gaze gate off"
        return f"gaze gate: {gate_counts(w.counts, w.counts['images'])}; {self._gaze.text()}"

    def _blind_because(self, counts: Counter[str]) -> str:
        """Why no beam judged the depth this window, in the words of the counter that won:
        the lidar silent, its plane under the bottom of the picture (the cart parked closer
        than :func:`pepin.depth.plane_in_view_from`), or too few beams surviving the edges.

        Worth its own sentence: the old line asked "is /scan alive?" for all three, and a cart
        parked half a metre from a wall — the working case — sent a morning into the bridge's
        QoS while the lidar was answering at 9.5 Hz (2026-09-14).
        """
        if counts["beams_out_of_frame"] >= max(counts["beams_too_few"], 1):
            near = self._plane_in_view()
            where = "" if near is None else f", and it shows only past {near:.2f} m ahead"
            return (
                "the scans arrive but the lidar's plane is out of the picture"
                f"{where}: back the cart off or tilt the head down to refit the law"
            )
        if counts["beams_too_few"]:
            return (
                f"beams land in the picture but under {MIN_SAMPLES} of them survive the edge"
                " mask on any frame"
            )
        return "no scan came within SCAN_MAX_AGE_S of any frame: is /scan alive?"

    def _pixels(self) -> int:
        """How many pixels a published depth image has, from the optics the last camera_info
        described; 0 while none has arrived (the report's share then reads 0.0 %)."""
        return 0 if self._intr is None else self._intr.width * self._intr.height

    def _plane_in_view(self) -> float | None:
        """How far ahead the lidar's plane enters the picture at the head's last pose, or
        ``None`` while the optics or the mount are still unknown."""
        mount = self._lidar_mount
        if self._intr is None or mount is None:
            return None
        return plane_in_view_from(self._intr, self._last_cam, float(mount.translation[2]))

    def _model_note(self) -> str:
        """The CPU model's state for the report line: nothing once it answers, ``not loaded``
        while no frame has asked for it, and why it failed when one did."""
        if self._local.built:
            return ""
        if self._local.failed:
            return f" (CPU model failed: {self._local.failed})"
        return " (CPU model not loaded)"

    def _scan_line(self, w: Window) -> str:
        """What went out on ``/depth_scan`` this window and what the ``scan_hz`` cap held back:
        the rate the costmaps' clearing source actually arrives at, so a fan rate below the
        frame rate is read as the cap and not as a pipeline that has stopped answering."""
        published = int(w.counts["scans"])
        if float(self._switches["scan_hz"]) <= 0.0:
            return f"/depth_scan {published} fans ({w.rate('scans'):.1f}/s, scan_hz 0: every frame)"
        return (
            f"/depth_scan {published} fans ({w.rate('scans'):.1f}/s,"
            f" {int(w.counts['scans_thinned'])} held by the scan_hz"
            f" {float(self._switches['scan_hz']):g} cap)"
        )

    def _dead_edge(self, w: Window, name: str, unit: str = "frames") -> str:
        """``neck edge dead 97 frames (344 s stale)`` when :class:`LiveEdgeHistory` refused to
        wait for that edge this window, and nothing when it did not — the words that tell a
        route which has died from an edge TF has simply never carried."""
        dead = w.counts[f"{name}_edge_dead"]
        if not dead:
            return ""
        stale = w.samples.get(f"{name}_edge_stale_s", [0.0])
        return f"{name} edge dead {dead} {unit} ({max(stale):.0f} s stale)"

    def _extras(self, w: Window) -> str:
        """The parts of the report line a window may have nothing to say about: how old the
        anchoring scans were, the scans no odometry could carry, the TF edges found dead, the
        frames whose camera pose came from the config instead of TF, the frames with the head
        turned, the frames whose beams fell outside the picture or were too few to judge it,
        the cart's lean, and the last TF failure of each kind."""
        c, extra = w.counts, ""
        ages = w.samples.get("scan_age", [])
        if ages:
            extra += f", scan age median {float(np.median(ages)):.2f} s max {max(ages):.2f} s"
        if c["uncarried"]:
            extra += f", scans uncarried {c['uncarried']}"
        for edge in ("odom", "map"):  # the edges a carry may ask about
            dead = self._dead_edge(w, edge, "asks")
            if dead:
                extra += f", {dead}"
        if self._stereo_on:
            waits = w.samples.get("pair_wait_s", [])
            extra += f", unpaired {c['unpaired']} frames"
            if waits:
                extra += (
                    f" (right eye waited {float(np.median(waits)) * 1e3:.1f} ms median,"
                    f" {max(waits) * 1e3:.1f} max)"
                )
            if c["no_stereo"]:
                extra += f", rig unknown {c['no_stereo']} frames"
        if c["carry_insane"]:
            extra += f", carry insane {c['carry_insane']} frames (the odometry ran away)"
        if c["camera_tf_stale"]:
            neck = self._dead_edge(w, "neck") or "no edge at the stamp"
            extra += f", camera pose from the last edge {c['camera_tf_stale']} frames ({neck})"
        if c["camera_from_config"]:
            neck = self._dead_edge(w, "neck") or "no TF edge"
            extra += f", camera pose from config {c['camera_from_config']} frames ({neck}"
            extra += "; fan pan from the mount)"
        if c["frames"]:
            share = c["beyond_reach"] / max(c["frames"] * self._pixels(), 1) * 100.0
            extra += (
                f", published NaN past {float(self._switches['depth_reach_m']):.1f} m over"
                f" {share:.1f}% of the pixels"
            )
        if c["camera_panned"]:
            extra += f", head panned {c['camera_panned']} frames (projected with the pan)"
        if c["beams_out_of_frame"]:
            near = self._plane_in_view()
            where = "" if near is None else f" (it shows past {near:.2f} m ahead)"
            extra += f", lidar plane out of the picture {c['beams_out_of_frame']} frames{where}"
        if c["beams_too_few"]:
            extra += f", beams under {MIN_SAMPLES} clean {c['beams_too_few']} frames"
        extra += f", {self._lean.report()}"
        if w.notes:
            extra += ", tf: " + "; ".join(
                f"{kind} {c['tf_' + kind]}: {text}" for kind, text in w.notes.items()
            )
        return extra


def main() -> None:
    spin_main(DepthStream)


if __name__ == "__main__":
    main()
