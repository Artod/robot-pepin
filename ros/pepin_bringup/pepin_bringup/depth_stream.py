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
is loaded at start. That file carries both laws: the affine numbers, seeded into every law of
the chain so none of them withholds while another publishes, and the ray law's own record
beside them, written whenever that stage has fitted one on the live pool and restored only
when it still stands on its own terms. The depth as it stands before the floor anchor, cut
between 8 cm and 1.3 m above the floor and folded onto the plane, goes out as ``/depth_scan``
(a LaserScan in base_link): the board's local costmap marks and clears with it like with the
lidar, so a
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

How far the cart leans is one estimator's (``pepin.lean`` through the kit's ``LeanFeed``, off
``/imu/data_raw``): the floor plane's up vector, and — with ``imu_lean`` on — the lean the poser
composes into the scan's carry and the camera's place, so a body tipped over a slipper does not
place its frame as if it stood level. A lean gravity did not vote for (``lean_min_quality``,
the signature of a drifting gyro rather than of a tipping body) is treated as no lean at all.

The flags (:data:`FLAGS`, ``ros/flags.sh set depth_stream <flag> <value>``): one per stage of
the pipeline — ``edge_filter``, ``lidar_anchor``, ``floor_pairs``, ``wall_anchor``,
``parallax_anchor``, ``affine_law``, ``ray_law``, ``wall_correct``, ``floor_anchor`` — plus
``depth_backend``, ``scale_ceiling``, the largest 1 / scale the law may be fitted to,
``imu_lean`` and ``lean_min_quality``; their state is printed in every report line.
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
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, LaserScan

from pepin.camera import CameraConfig, mount_transform, optics
from pepin.depth import (
    POOL_MIN_SAMPLES,
    SCALE_CEILING,
    SCAN_WINDOW_S,
    Array,
    CameraPose,
    Intrinsics,
    depth_to_scan,
    load_law,
    load_ray,
    nearest_stamp,
    optical_heading,
    save_law,
    scan_points,
    set_scale_ceiling,
    to_base,
)
from pepin.depth_pipeline import AffineLaw, FrameContext, RayLaw, standard_pipeline
from pepin.depth_service import (
    DEFAULT_URL,
    MODES,
    DepthModelError,
    Fallback,
    LazyDepth,
    RemoteDepth,
)
from pepin.elevation import RayGain
from pepin.flags import Flag, FlagSet
from pepin.frame_pose import FramePoser
from pepin.lean import LEAN_QUALITY_FLOOR
from pepin.parallax import to_gray
from pepin.tsdf import RigidPose
from pepin_bringup.msgs import (
    array_from_image,
    image_from_array,
    scan_from_ranges,
    stamp_seconds,
)
from pepin_bringup.node_kit import (
    Fatal,
    LeanFeed,
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
        on_when="whenever the lidar spins — it is what makes the network's depth metric",
        off_when="to rehearse a lidar that dies mid-run (the law freezes, nothing else changes),"
        " or to compare the slices above the band, where the frozen law measured better",
    ),
    Flag(
        "floor_pairs",
        False,
        description="the floor's pixels pair the network's depth with the plane's geometric depth,"
        " a second hoop for the law that needs no lidar",
        why="useless as measured — it takes the floor pixels the network itself drew and checks"
        " the network against them, and the median D/E over the pixels it selects is the median"
        " over every ray that meets the plane (0.86 against 0.89, 1.12 against 1.22, 1.76 against"
        " 1.69: scratch/horizon_law_eval.txt). What it does change is the law: the lidar's own row"
        " goes 1.010 -> 1.391 and the band 12.9/38.2 cm -> 24.1/48.7 on run 0171"
        " (scratch/pipeline_vs_truth.txt), 3.8/20.6 -> 4.8/26.7 cm at the corrected mount"
        " (scratch/lidar_height_check.txt). A floor anchor needs a floor cue the network did not"
        " draw itself",
        on_when="when the floor's depth comes from something independent of the network — a"
        " measured plane, a second sensor; nothing measured so far supports turning it on",
        off_when="in every run that drives: it buys nothing above 0.8 m either (65.2 cm against"
        " 48.2) and moves the row the costmap reads by 38 %",
    ),
    Flag(
        "wall_anchor",
        False,
        description="the lidar's returns extruded up the image, where the network's depth stays"
        " continuous, pair the rows above the lidar's with the wall's depth — a third hoop",
        why="redundant with the lidar and worse where the cart drives. The wall extrusion agrees"
        " with the beams to 3 % once the mount height is right (lidar/wall 0.966 at the tape's"
        " 0.383 m against 0.816 at the assumed 0.200, scratch/lidar_height_check.txt), and"
        " switching it on pulls the lidar's own row 10 % near (1.010 -> 0.896) and stretches the"
        " band's tail 2.6x (p90 20.6 -> 53.6 cm) on run 0171 (scratch/pipeline_vs_truth.txt,"
        " scratch/lidar_height_check.txt). What it buys is the rows above: the 3D error at z"
        " 0.80-1.20 m 47.4 -> 31.5 cm",
        on_when="on a robot with no lidar, where the extrusion is the only wall cue; or when what"
        " reads the depth is above 0.5 m (a manipulator's reach) and no costmap is reading the"
        " band",
        off_when="whenever the cart drives on the band: that is the row the costmap reads, and"
        " wall pairs move it 10 %",
    ),
    Flag(
        "parallax_anchor",
        False,
        description="the corners this frame shares with the previous one, triangulated against the"
        " odometry's transform between the two stamps (pepin.parallax), pair the network's depth"
        " with a depth in metres the cart measured by moving — a hoop that needs no lidar and no"
        " assumed plane and that lands at every elevation the picture has",
        why="not measured live yet: on runs 0171 and 0165 it costs 3.5-3.8 ms a frame and yields"
        " 30-190 pairs on only 31-36 % of frames (nothing at all while the cart stands still or"
        " turns on the spot), and at those runs' 2.9 cm median baseline the triangulated depth is"
        " +25-37 % too far under 1.5 m (19-30 samples a run) while it sits within 3 % of the lidar"
        " from 1.5 to 3 m (scratch/parallax_vs_lidar.txt). The odometry's own +-25 % scale band"
        " multiplies near and far alike and cannot make a range-dependent bias; the cause is not"
        " known",
        on_when="after a run at driving speed (0.3 s of gap = 10-15 cm of baseline instead of 3"
        " cm) with the calibrated focal length (fx 724.1, HFOV 82.9 deg) either explains the"
        " near-field bias or clears it",
        off_when="wherever the cart stands, turns on the spot or faces blank walls: it yields"
        " nothing there and costs its 3.5 ms anyway",
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
        "ray_law",
        False,
        description="the law's scale follows the ray's angle off the optical axis, a / D + b"
        " fitted per elevation (pepin.elevation) instead of one pair of numbers for the whole"
        " picture; it needs wall_anchor on, because on the lidar's own beams a return's elevation"
        " and its 1 / z are the same variable (|corr| 1.000) and the angular fit is refused",
        why="for good, unless new data arrives: the network's error follows the world's elevation,"
        " not the ray's angle. Across three neck pitches (tapes 0235/0236/0237 at 11.1, 25.8, 40.9"
        " deg) the confound gate refuses the fit at two of them, and the one law that could be"
        " fitted helps in sample and hurts at both other pitches (scratch/ray_law_pitch_eval.txt);"
        " carried between tapes and geometries no angular law beats plain scale out of sample"
        " (mean |ln ratio| 13.6 % for scale against 15.1-15.4 % for the ray laws, while the room's"
        " own elevation reaches 12.9 %: scratch/horizon_law_eval.txt). Fixing the camera's pose"
        " was worth 9 of those 22 points, the best angular term 0.7 more",
        on_when="only on data that separates the ray's angle from the world's elevation — a pitch"
        " sweep with the calibrated lens whose fit the confound gate does not refuse",
        off_when="it ships off; the scale itself still moves 21-25 % over 30 deg of neck pitch,"
        " which asks for a refit, not for an angular law",
    ),
    Flag(
        "wall_correct",
        False,
        description="after the law, the pixels the wall walk covered are set to the extruded"
        " plane's depth outright (the same walk as wall_anchor, applied instead of fitted)",
        why="default by design, unmeasured as a win — standalone it is a wash — run 0171 keeps the"
        " same law and the same lidar row (1.010, |·-1| q3 0.320) and the band reads 12.8/39.3 cm"
        " against today's 12.9/38.2, or 3.6/20.2 against 3.8/20.6 at the corrected mount, with the"
        " 3D error slightly better at every slice (scratch/pipeline_vs_truth.txt,"
        " scratch/lidar_height_check.txt). It is off because it is the wall walk and the wall walk"
        " is off; no number says it hurts",
        on_when="with wall_anchor on, when what reads the depth is the wall above the beams and a"
        " plane is a better answer there than a fitted one",
        off_when="wherever the published depth must stay the network's own measurement rather than"
        " a plane drawn over it",
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
        "scale_ceiling",
        SCALE_CEILING,
        description="the largest 1 / scale the law may be fitted to (the upper half of"
        " pepin.depth.A_BOUNDS); a law that lands on a bound prints AT BOUND",
        why="at 3.0 the law was a clipped constant once the lidar's plane was measured at its true"
        " 0.383 m — the fit saturated at a 3.00 with b pinned at -0.200 and stopped being a fit"
        " (scratch/lidar_height_fix_report.txt). Opened to 5.0 the same run fits a 2.80 in a, and"
        " the COLMAP control at 0.50-0.80 m reads 1.14 [0.90..1.24] against the clipped law's 1.32"
        " [1.08..1.43]; the clipped law's tighter band against the beams (3.8 against 5.2 cm"
        " median) is luck, not fit. b still sits on its own bound, -0.200: the next one to"
        " question",
        on_when="raise it above 5.0 only when a law reports AT BOUND in a and the mount height and"
        " the lens behind that law have been checked first",
        off_when="set it back to 3.0 to reproduce the clipped law in the field, side by side, with"
        " no restart",
        range=(0.5, 20.0),
    ),
    Flag(
        "imu_lean",
        False,
        description="the cart's lean (pepin.lean, from /imu/data_raw) is followed with the gyro as"
        " well as the accelerometer and carried into the scan's carry and the camera's place in"
        " the map; off, the floor plane leans with the accelerometer alone, as it always has, and"
        " nothing else is leaned",
        why="until the gyro's roll and pitch signs are checked by tipping the cart by hand. What"
        " is worth is measured — 5 degrees of lean walks a 3 m ray 26 cm off the plane — and so is"
        " the chip at rest (30 s level: accel mean (-0.051, -0.067, +9.945) m/s2 = roll -0.39 deg,"
        " pitch +0.29 deg; gyro bias (-0.001, -0.028, +0.074) deg/s at 0.03-0.05 deg/s of noise,"
        " config/imu.json's level block). What is not measured is the correction's sign: only the"
        " yaw axis was checked against a 90 degree turn on the robot, roll and pitch come from a"
        " mount mapping nobody has turned through a known angle, and a level floor hides a swap or"
        " a flip. The recordings cannot settle it either — of 93 tapes with IMU records none carry"
        " the accelerometer (scratch/lean_effect.py), so every lean proof so far is a synthetic"
        " bump replayed over real geometry. With the flag off the report line still prints the"
        " accelerometer-only lean, which is the A/B",
        on_when="after a hand tip through a known angle shows the reported lean following it the"
        " right way and returning to zero",
        off_when="the moment the lean in the report line disagrees with the cart's visible"
        " attitude",
    ),
    Flag(
        "lean_min_quality",
        LEAN_QUALITY_FLOOR,
        description="how much of the lean gravity must have voted for (pepin.lean's quality,"
        " printed beside the lean in this line) before a frame is placed by it: below it the lean"
        " is treated as unknown and the frame is placed level",
        why="chosen on a simulation, not on the robot: in scratch/lean_quality_floor_probe.py a"
        " 0.2 deg/s gyro bias reports 3.0 degrees of tip on a level floor at quality 0.02 or less,"
        " nothing past 0.13 degrees of it survives a floor of 0.5, and a real 6 degree threshold"
        " climb keeps quality 1.00 throughout — so the floor costs the feature nothing. The 0.2"
        " deg/s is hypothetical: this chip's worst measured axis is 0.074 deg/s (config/imu.json's"
        " level block). The same floor is declared in depth_fusion, so the pose and the scan gate"
        " make one decision",
        on_when="raise it towards 1.0 on a robot that only ever leans when something real pushes"
        " it",
        off_when="0 believes every lean, as before the floor existed: an A/B of the gyro's own"
        " drift",
        range=(0.0, 1.0),
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
        self._camera_cfg = cfg  # config/camera.json's own optics until a camera_info arrives
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
        self._ray = RayLaw()
        self._last_verdict_wall = time.time()  # the law's age is the beams', not the node's
        self._seed_laws(time.time())
        self._pipeline = standard_pipeline(self._law, ray=self._ray)
        self._switches = Switches(self, FLAGS, on_change=self._on_switch)
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
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, reliable)
        self.create_subscription(Image, "/camera/image", self._on_image, newest)
        self.create_subscription(LaserScan, "/scan", self._on_scan, reliable)
        self._tf = TfLookup(self, on_failure=self._on_tf_failure)
        self._poser = FramePoser(
            TfHistory(self._tf, timeout_s=CARRY_WAIT_S),
            lean=self._lean,
            apply_lean=self._switches.on("imu_lean"),
            min_lean_quality=float(self._switches["lean_min_quality"]),
        )
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

    def _seed_laws(self, now: float) -> None:
        """Hand both laws what the last run saved in the law file, and say so in the log: the
        affine numbers to the affine law *and* to the ray law (each pools and fits on its own,
        so a ray law left unseeded would withhold every frame of the warm-up while the affine
        law publishes), and the angular gain to the ray law when the file holds one that still
        stands (:meth:`pepin.elevation.RayGain.restore` judges it). Without a file nothing is
        published until POOL_MIN_SAMPLES beam pairs are pooled."""
        saved = load_law(self._law_file, now)
        if saved is None:
            self.get_logger().info(
                f"no saved depth law at {self._law_file}: publishing waits for"
                f" {POOL_MIN_SAMPLES} pooled beams"
            )
            return
        self._law.seed(saved[0], saved[1])
        self._ray.seed(saved[0], saved[1])
        record = load_ray(self._law_file, now)
        gain = RayGain.restore(record) if record is not None else None
        if gain is not None:
            self._ray.seed_gain(gain)
        ray_note = f"; ray law {gain.describe()}" if gain is not None else "; no ray law saved"
        self.get_logger().info(
            f"depth law from {self._law_file}: a {saved[0]:.2f} b {saved[1]:+.3f}"
            f" on {saved[2]} beams; publishing at once{ray_note}"
        )

    def _on_switch(self, name: str, _old: Any, new: Any) -> None:
        """A flag changed: ``depth_backend`` is the switch's mode, ``scale_ceiling`` the law's
        upper bound, ``imu_lean`` the poser's and the estimator's, ``lean_min_quality`` the
        poser's floor under a lean, a stage's flag switches that stage of the pipeline."""
        if name == "depth_backend":
            self._net.mode = str(new)
        elif name == "scale_ceiling":
            set_scale_ceiling(float(new))  # the next fit is bounded by it; the law in hand is not
        elif name == "imu_lean":
            self._poser.apply_lean = bool(new)
            self._lean.use_gyro = bool(new)
        elif name == "lean_min_quality":
            self._poser.min_lean_quality = float(new)
        elif name in self._pipeline.switches:
            self._pipeline.set(name, bool(new))

    def _on_work_error(self, text: str) -> None:
        self.get_logger().error(f"depth failed on a frame:\n{text}")

    def _on_tf_failure(self, kind: str, text: str) -> None:
        self._tally.count("tf_" + kind)
        self._tally.note(kind, text)

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
            cam, cam_optical = self._camera_at(msg.header.stamp)
        with tally.measure("samples"):  # includes the TF wait for the carry
            lidar = self._lidar_points(msg)
        # The parallax anchor is the one stage that reads the picture itself; the grey copy is
        # made only while it is on, the poser it triangulates against is the same TF the carry
        # uses, and the camera edge goes in whole so its baseline carries the neck's pan.
        ctx = FrameContext(
            self._intr_or_nominal(msg),
            cam,
            up=self._lean.up,
            lidar=lidar,
            stamp=stamp_seconds(msg.header.stamp),
            gray=to_gray(rgb) if self._pipeline.on("parallax_anchor") else None,
            motion=self._poser,
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

    def _camera_at(self, stamp: Any) -> tuple[CameraPose, RigidPose | None]:
        """Where the camera sat at ``stamp``: ``base_link <- camera_optical`` from TF (the
        neck's live edge, or the static one) twice over — as the pipeline's pitch-only pose,
        and as the edge itself for the stage that needs the whole rotation (the parallax
        anchor's baseline turns with the neck's pan). Without such an edge in TF:
        config/camera.json's mount, counted, and no edge. A head turned past PAN_NOTICE_RAD is
        counted too: the projections assume it looks along the cart's x."""
        pose = self._poser.camera_in_base(stamp_seconds(stamp))
        if pose is None:
            self._tally.count("camera_from_config")
            return self._camera_config, None
        _pitch, pan = optical_heading(pose.rotation)
        if abs(pan) > PAN_NOTICE_RAD:
            self._tally.count("camera_panned")
        return CameraPose.from_optical(pose.rotation, pose.translation), pose

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
                save_law(
                    self._law_file,
                    law.a,
                    law.b,
                    law.pooled,
                    self._last_verdict_wall,
                    ray=self._ray.saved_state(),
                )
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
