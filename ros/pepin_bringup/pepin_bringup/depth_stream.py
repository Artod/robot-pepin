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
is loaded at start. That file carries every law: the affine numbers, seeded into every law of
the chain so none of them withholds while another publishes, and the ray law's and the range
law's own records beside them, each written whenever its stage has fitted one on the live pool
and restored only when it still stands on its own terms. One affine law is not the shape of
this camera's error — its residual tilts 12 % per metre of range — so the law that ships live
is the range law, the same pooled pairs read per bin of the network's own depth
(:class:`pepin.depth.RangeLaw`, the ``range_law`` flag), and behind it the frame law
(:class:`pepin.depth_pipeline.FrameLaw`, the ``frame_law`` flag) corrects what the range law
published by the beams of the frame in hand — the per-image alignment the field performs, and
the only law a pitch of the neck cannot leave stale. The depth as it stands before the floor
anchor, cut between 8 cm and 1.3 m above the floor and folded onto the plane, is ``/depth_scan``
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
``parallax_anchor``, ``affine_law``, ``ray_law``, ``range_law``, ``frame_law``,
``wall_correct``, ``floor_anchor`` — plus ``depth_backend``, ``scale_ceiling``, the largest
1 / scale the law may be fitted to, ``law_slew``, how fast that law may move between fits,
``imu_lean`` and ``lean_min_quality``; their state is printed in every report line.
"""

from __future__ import annotations

import math
import os
import threading
import time
import traceback
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, LaserScan

from pepin.camera import CameraConfig, mount_transform, optics
from pepin.contact import (
    FAN_FLOOR_GATE,
    FAN_FLOOR_GATES,
    FloorPlane,
    contact_scan,
    fan_min_z,
    gate_by_contact,
)
from pepin.depth import (
    MIN_SAMPLES,
    POOL_MIN_SAMPLES,
    SCALE_CEILING,
    SCAN_MIN_Z_M,
    SCAN_WINDOW_S,
    Array,
    CameraPose,
    Intrinsics,
    carry,
    carry_speed,
    depth_to_scan,
    floor_depth,
    load_law,
    load_range,
    load_ray,
    nearest_stamp,
    optical_heading,
    plane_in_view_from,
    save_law,
    scan_points,
    set_scale_ceiling,
    to_base,
)
from pepin.depth_pipeline import (
    FIELD_GRIDS,
    LIDAR_SIGMA_M,
    PARALLAX_MAP_MAX_AGE_S,
    PARALLAX_MAP_WAIT,
    PARALLAX_MATCHER,
    PARALLAX_MIN_BASELINE_M,
    PARALLAX_MOTION,
    PARALLAX_MOTIONS,
    PARALLAX_WEIGHT,
    PIPELINE_DEFAULTS,
    AffineLaw,
    FloorPairs,
    FrameContext,
    FrameLaw,
    LidarAnchor,
    ParallaxAnchor,
    RangeLawStage,
    RayLaw,
    grid_of,
    standard_pipeline,
)
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
from pepin.parallax import MATCHERS, to_gray
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
CAMERA_TF_MAX_AGE_S = 1.0  # the neck's newest edge is the head's pose while it is at most this old
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
        "floor_pairs",
        PIPELINE_DEFAULTS["floor_pairs"],
        description="the floor's pixels pair the network's depth with the plane's geometric depth,"
        " a second hoop for the law that needs no lidar. Each pair weighs its own sigma — the"
        " plane's depth under a ray is h / sin(angle below the horizon), so the mount's pitch"
        " uncertainty makes it grow as the square of the range (floor_sigma_pitch_deg) — and the"
        " frame's whole floor is refused unless the plane fitted to those pixels stands up"
        " (floor_normal_tol_deg); the report line counts the frames refused",
        why="it stays off, but it is no longer the same knob. Under ONE law it moved the lidar's"
        " own row by 5-8 % and the band 12.9/38.2 cm -> 24.1/48.7 on run 0171"
        " (scratch/pipeline_vs_truth.txt, 2026-09-11), because the network's error is regime-wise"
        " (floor 1.1x, the lidar's row 1.6x, above it 2.0x) and one law fitted across the two"
        " lands between them. Under the 3x3 scale field the same pairs cost far less and"
        " sometimes pay: on the held-out beams of the four tapes of 2026-09-15"
        " (scratch/scale_field_eval.txt) the floor takes the single law from 11.4 % to 17.6 % of"
        " median |residual| on run 0171 and from 4.5 % to 16.3 % on tape 0237, where under the"
        " field it takes 8.3 % to 10.6 % and 4.3 % to 5.0 %, and on tape 0236 it IMPROVES the"
        " field, 14.1 % -> 13.6 %. What it still does not do is give a metric scale on its own:"
        " with every beam withheld the floor-only field reads 19.8 % at the 40.9 deg pitch, 78.6 %"
        " at 25.8 and 93.2 % on the drive, where the floor is barely in the picture",
        on_when="with the field on and a head pitched down far enough that the floor fills a"
        " third of the picture, when what reads the depth is above the lidar's row; and on a cart"
        " with no lidar at all, where it is the only ruler there is",
        off_when="in every run that drives on the lidar's row: it still costs 2.3 points of"
        " residual there on the drive, and nothing yet says the rows above are worth that",
    ),
    Flag(
        "wall_anchor",
        PIPELINE_DEFAULTS["wall_anchor"],
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
        PIPELINE_DEFAULTS["parallax_anchor"],
        description="the corners this frame shares with the previous one, triangulated against the"
        " odometry's transform between the two stamps (pepin.parallax), pair the network's depth"
        " with a depth in metres the cart measured by moving — a hoop that needs no lidar and no"
        " assumed plane and that lands at every elevation the picture has",
        why="OFF again since 2026-09-15 04:20: on the live stack the anchor's ask for the tracker's"
        " motion (parallax_motion=tracker) waited its 0.2 s carry timeout on EVERY frame (report:"
        " pose 211/222 ms, 'no odometry 41, gap 41'), the stream fell from 8.7 to 1.5 frames/s and"
        " rgbd_odometry starved (0 poses/s); off, 6.1 frames/s and VO back within 40 s. Until the"
        " ask is non-blocking the anchor stays off. Before that: it is the second ruler of the"
        " scale, and it is now weighed like one. Every pair"
        " carries 1 / sigma^2 from its own triangulation against a beam's 1 / sigma^2 at"
        " lidar_sigma_m (pepin.depth.pair_weight), so a corner at 7-10 cm of noise counts about"
        " 0.03 of a beam and 200 of them do not outvote 30 beams: measured on the errands of"
        " 2026-09-14, the lidar keeps 96-100 % of a frame's fit weight wherever it reaches"
        " (scratch/parallax_ruler_eval.txt: parallax 0 % of the weight at the median, 0-4 %"
        " p10-p90 over the 40 frames carrying both). What the anchor buys is where the lidar"
        " does not"
        " reach — above the plane, nearer than the 0.71 m at which the plane enters the picture,"
        " and every frame with no beams at all, where it is the only metric ruler left and the"
        " frame law fits on it instead of decaying to the pool. Its own depth reads 0.93-1.01 of"
        " the lidar at 1.5-2 m on the tracker's motion at the 1.0-1.5 s gaps measured"
        " (scratch/parallax_pose_sweep.txt), at 3.5-3.9 ms a frame",
        on_when="always, now that the weights are its noise: on a frame the lidar covers it"
        " changes the law by a few per cent, and on a frame the lidar does not cover it is the"
        " law",
        off_when="to A/B what it buys (or set parallax_weight 0, which keeps the pairs and their"
        " report line and takes their vote away), and on a cart whose tracker is dead AND whose"
        " odometry is untrusted: the baseline is then a guess and every depth is proportional to"
        " it",
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
        PIPELINE_DEFAULTS["ray_law"],
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
        "range_law",
        PIPELINE_DEFAULTS["range_law"],
        description="the law's scale follows the range: the same pooled pairs binned by the"
        " network's own depth (17 log bins, 0.3-12 m, 50 pairs a bin) with a robust ratio"
        " true / network measured in each, interpolated between the filled bins"
        " (pepin.depth.RangeLaw), instead of one pair of numbers for the whole picture; on, its"
        " image replaces the affine law's, off, the affine law's stands. Until two bins fill it"
        " falls back to the affine law rather than withholding the frame",
        why="one affine law is the wrong shape for this camera. Standing at home on 2026-09-14"
        " (scratch/depth_scale_by_range.py, 182 frames, 18 879 beams) the PUBLISHED depth — a"
        " 1.76 b 0, median ratio 0.996 over the whole pool — ran +8.7 % (+9.3 cm) at 0.8-1.2 m,"
        " +5.2 % (+7.0 cm) at 1.2-1.6 m and -3.3 % (-5.9 cm) at 1.6-2.0 m: 12 % of tilt per metre"
        " of range. Which ranges the pool holds then decides the law — a drive brings 0.5 m and"
        " 4 m pairs, the shift term opens and the same tilt is described as a 2.3 b -0.19, back"
        " at rest as a 1.75 b 0 — so the fused volume is painted under one law and scored under"
        " another, and depth_fusion refuses those frames at the yaw search's bound",
        on_when="always, until a law that follows the range is measured to be worse than one that"
        " does not",
        off_when="as an A/B against the affine law at rest, and the moment a report line shows a"
        " bin's ratio jumping between windows (a pool that has gone degenerate, not a lens)",
    ),
    Flag(
        "frame_law",
        PIPELINE_DEFAULTS["frame_law"],
        description="after the range law, THIS frame's own beams fit a scale (and, where the"
        " frame's depths span 2.5x, a shift) over what the range law published, and that"
        " correction is applied to the whole image (pepin.depth.fit_frame, Huber IRLS on 30"
        " pairs or more); a frame with too few beams holds the last one, decaying back to the"
        " range law with a 2 s time constant. This is the per-image scale-and-shift alignment"
        " the monocular-depth field performs: Depth Anything V2's metric heads are evaluated"
        " after exactly such an alignment against sparse truth, and a robot with a depth sensor"
        " aligns its monocular depth against that sensor's points frame by frame",
        why="the pool's law describes the last 64 s, not this picture. Measured on 2026-09-14"
        " (scratch/frame_law_eval.py; every frame's pairs split odd / even, the odd fitting, the"
        " even judging, so no law grades its own pairs): on run 0171's drive the median"
        " |residual| reads 7.5 % against the range law's 23.2 % and the affine law's 26.6 %, and"
        " the residual's spread across the top, middle and bottom third of the image falls from"
        " 38.0 % (range) and 24.2 % (affine) to 10.0 % — the pitch question. Carried across neck"
        " pitches it is the whole answer: tape 0235's pool laws read on 0236 and 0237 leave"
        " 36.7 % and 49.5 %, where the frame law, refitting itself, reads 16.3 % and 4.5 %. At"
        " one pitch with the pool law fresh it is a wash (0236: 5.9 % against 5.9 %; 0237: 3.7 %"
        " against 3.7 %; 0235: 15.6 % against 15.2 %), so it never pays to switch it off. It"
        " does not fix the network's saturation past ~1.8 m: no scale can",
        on_when="always, while the lidar's beams reach the picture — a law fitted on the frame"
        " in hand cannot be stale, and at one steady pitch it costs nothing",
        off_when="to A/B the pool's law against it, and where the beams are known to pair with"
        " the wrong surface (a mirror, a glass front): a bad frame then moves the whole image"
        " instead of a bin of the pool. The report line names the frames held",
    ),
    Flag(
        "wall_correct",
        PIPELINE_DEFAULTS["wall_correct"],
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
        "law_slew",
        0.0,
        range=(0.0, 1.0),
        description="how fast the affine law may move, as the largest relative change of the"
        " published inverse depth over the pool's own depth range, per second; 0 applies every"
        " fit whole, as the node always did. A law still walking to its fit says so in the"
        " report line (slewing to a X b Y)",
        why="2026-09-14: standing at home the law reads a 1.74 b 0.000 on 62 000 pairs; 30 s of"
        " driving takes it to a 2.33 b -0.200 and back. Neither the camera nor the room changed:"
        " the pool is 600 frames, which at 9.4 frames/s is 64 s, so half a minute replaces half"
        " of it, and the drive's wider depth range opens the shift term"
        " (pepin.depth.MIN_DEPTH_SPREAD 2.5 — standing, the beams span 0.8-2.0 m, a ratio of"
        " 2.05). The same resting beams fitted with a free shift give a 2.26 b -0.183"
        " (scratch/depth_scale_by_range.py), which is the drive's law: one set of pairs, two"
        " descriptions, 11 % apart in metres at 1 m. The volume is painted with whichever was in"
        " force, and the next frame no longer fits it — 220-270 frames per 30 s refused at the"
        " alignment bound, a pure 5 % scale mismatch being enough to pin that search at its edge"
        " (scratch/align_vs_scale.py)",
        on_when="0.005 (30 % a minute) to let the law follow the room but not one drive's worth"
        " of pairs; raise it only after a drive has been read with it on",
        off_when="0 to reproduce today's behaviour, where every fit is applied whole",
    ),
    Flag(
        "carry_max_speed_mps",
        1.0,
        range=(0.1, 20.0),
        description="metres per second the carry from the scan's moment to the frame's may imply"
        " before the frame's lidar beams are thrown away instead of anchoring the law; the frame"
        " still publishes its depth, it simply judges nothing",
        why="2026-09-14: with the EKF running away (43 km at 60 m/s) the carry moved the scan"
        " 1-2 m over the 0.02-0.03 s between the scan and the frame, dragged beams across the"
        " picture and refitted the law from those pairs — a went 1.65 -> 2.05 and the law file"
        " had to be thrown away (ros/maps/depth_law.json.corrupt-20260914). This cart's top"
        " speed is 0.3 m/s, so one metre per second is three times anything it can drive and"
        " still far under what a runaway frame shows. The board's own guard"
        " (relocalizer's odometry_guard) stops the pose; this one stops the law",
        on_when="raise it only on a faster base",
        off_when="raise it to 20 to reproduce the old behaviour, where any carry was applied"
        " whatever it implied",
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
    Flag(
        "parallax_min_baseline_m",
        PARALLAX_MIN_BASELINE_M,
        description="how much parallax the anchor picks its partner frame to reach: walking back"
        " through the last second of frames it pairs with the first one inside the gap window"
        " whose baseline reaches this, and with the widest baseline it has when none does",
        why="the frame before this one is 0.1 s back, and 0.1 s at the cart's 0.2-0.3 m/s is 2 cm"
        " of baseline: the live errand of 2026-09-14 14:12 paired every frame 2.1 cm apart, kept a"
        " 22.8 cm sigma and threw 13885 corners away for too little parallax. Both legs of that"
        " errand re-measured offline against the lidar's own ranges"
        " (scratch/parallax_baseline_sweep.txt, 4700 matched points): the per-pair sigma falls"
        " 16.3 cm at 2 cm of parallax to 12.2 at 5 cm, 6.9 at 9 and 4.1 at 18, and the depth from"
        " 1.5 to 3 m goes from 0.75 and 0.49 of the lidar at 2 cm — too near, the thin baseline's"
        " own skew — to 1.02-1.13 from 5 cm on. The cost is the flow: 7.8 % of corners lost at a"
        " 0.1 s gap, 25 % at 0.4 s, 32 % at 0.5 s, and the points kept per frame peak at the 0.4 s"
        " gap (median 17) before collapsing past 0.6 s. 10 cm of baseline is that 0.4-0.5 s at"
        " this speed. What no baseline touches is a +9 to +13 % offset at 1.0-1.5 m, flat across"
        " every bin: a scale-like error, not the range-dependent one reported on 2026-09-12",
        on_when="raise it towards 0.15 on a cart that drives faster than 0.3 m/s, where the"
        " longer gap still tracks — measured, not assumed: past a 0.6 s gap the points kept per"
        " frame fall to single figures",
        off_when="0 restores the old behaviour exactly: every partner reaches a baseline of 0, so"
        " the walk stops at the newest one — the frame before this one",
        range=(0.0, 1.0),
    ),
    Flag(
        "parallax_matcher",
        PARALLAX_MATCHER,
        description="who finds the corners two frames share: klt follows them with optical flow,"
        " orb describes and recognises them. The matcher sets how far back a partner may sit —"
        " 0.60 s for the flow, 1.5 s for the describer — and what a point's place is trusted to,"
        " half a pixel against a whole one",
        why="klt wins at every gap this cart reaches. Both matchers over both errands of"
        " 2026-09-14 on the same frame pairs (scratch/parallax_matcher_sweep.txt, runs"
        " 0267/0268/0273/0274): at the 0.5 s gap the anchor actually pairs across, the flow reads"
        " 0.993 of the lidar at 1.5-2 m against the describer's 1.031, with half the noise"
        " (13.5 cm against 29.6 per pair) at half the cost (3.9 ms a frame against 8.3). What the"
        " describer does buy is the long gap: 11 pairs a frame at 1.0 and 1.5 s where the flow"
        " gives 2 and 0, having lost 70-78 % of its corners. It buys them at the wrong depth —"
        " both matchers read 1.28-1.45 of the lidar at 1.5-2 m once the gap passes a second,"
        " because a second of this odometry's drift inflates the baseline every depth is"
        " proportional to. The gap is capped by the odometry, not by the matcher",
        on_when="orb on a cart whose pose over 1.5 s is better than its wheels and gyro (a"
        " loop-closing graph, a second odometry), where the describer's 12.8 cm of baseline at"
        " 1.5 s is worth its noise; or on a robot that pauses between steps, where the flow has"
        " no short gap to work with",
        off_when="klt whenever the cart drives on wheel odometry: measured better, quieter and"
        " cheaper at every gap under a second",
        choices=MATCHERS,
    ),
    Flag(
        "camera_tf_latest",
        True,
        description="take the newest base_link <- camera_optical edge TF holds (at most"
        f" {CAMERA_TF_MAX_AGE_S:.0f} s old) when the frame's own stamp is not covered yet, instead"
        " of waiting CARRY_WAIT_S for it; off: the old wait at the exact stamp",
        why="on since 2026-09-15 06:00: the neck's edge crosses the bridge late (bursts of +0.75 s)"
        " and the head stands still while the cart drives; waiting for the exact stamp cost"
        " 0.2 s on every frame ('Extrapolation ... into the future' x104 a window), the stream"
        " fell to 3.5 frames/s and rgbd_odometry starved (0 poses/s)",
        on_when="always while the head does not move during a frame (it does not: neck moves"
        " are refused while the wheels turn)",
        off_when="a head that pans while driving, where a 1 s old edge would be a wrong pose",
    ),
    Flag(
        "frame_shift_needs_beams",
        True,
        description="the per-frame law fits a shift only when the lidar is one of the rulers of"
        " that frame's pool; a pool of parallax corners alone gets a scale and no shift. Off,"
        " the shift is decided by the pool's depth spread alone, whoever measured it",
        why="the spread gate cannot see WHO spans the room. The lidar's row spans little and"
        " usually keeps the shift shut; the corners land at every elevation and range in the"
        " picture and open it every time, and a two-parameter fit on a ruler of 7-10 cm a pair"
        " runs to the law's bounds. Measured over the four errands of 2026-09-14"
        " (scratch/parallax_ruler_eval.txt, 101 frames, the corners fitting and the beams"
        " judging): with the shift a parallax-only law reads 44.5 % median |residual| and its"
        " scale jumps 1.69 between consecutive frames, with the scale alone 30.9 % and 1.15."
        " Neither is a law to drive on — this is the gate that makes the lidar-off case merely"
        " bad instead of unbounded",
        on_when="always while the parallax anchor's own sigma stays where it is measured",
        off_when="when a parallax pool is trusted to identify a shift — a calibrated focal"
        " length and a per-pair sigma under a couple of centimetres",
    ),
    Flag(
        "field_grid",
        PIPELINE_DEFAULTS["field_grid"],
        description="how many nodes the per-frame law carries over the picture, written the way"
        " an image size is (columns x rows): each node holds its own scale and shift in inverse"
        " depth, fitted on the anchors that land near it, and a pixel's law is the bilinear blend"
        " of the nodes around it. 1x1 is one law for the whole picture — the frame law exactly as"
        " it was. The report line prints the grid and the pair weight every node saw",
        why="this network's error is a property of WHERE in the picture a pixel is: against COLMAP"
        " on run 0171 it reads 1.1x on the floor, 1.6x at the lidar's row and 2.0x from 0.3 m up"
        " (scratch/pipeline_vs_truth.txt), and one law fitted across all three is wrong in all"
        " three. Measured on the held-out beams of the four tapes (scratch/scale_field_eval.txt,"
        " 2026-09-15: every frame's pairs split odd / even, the odd fitting and the even judging,"
        " over the RAW network so the numbers are the law's whole correction) the median"
        " |residual| goes 11.4 -> 8.3 % on run 0171's drive, 15.6 -> 15.4 % at the 11.1 deg pitch,"
        " 16.3 -> 14.1 % at 25.8 and 4.5 -> 4.3 % at 40.9, and on the drive the residual's spread"
        " across the top, middle and bottom third of the picture falls from 25.8/13.6/9.2 % to"
        " 11.8/9.9/6.8 %. 4x3 is a wash against 3x3 (8.5 / 15.2 / 12.8 / 4.6 %) and costs 0.2 ms"
        " more. The whole field costs 0.94 ms a frame against the single law's 0.46 on a 640x360"
        " frame, 2x2 0.81, 4x3 1.14, 4x4 1.19",
        on_when="3x3 as shipped; 4x3 where the error is suspected to run across the picture"
        " rather than up it (a lens the calibration does not describe at the edges)",
        off_when="1x1 reproduces the single per-frame law exactly, bit for bit, which is the A/B"
        " of the whole field and the thing to set the moment a node's scale looks wild in the"
        " report line",
        choices=FIELD_GRIDS,
    ),
    Flag(
        "field_prior",
        PIPELINE_DEFAULTS["field_prior"],
        range=(0.0, 1000.0),
        description="how hard each node of the field is pulled toward the frame's own GLOBAL fit,"
        " in pair-weight units (a lidar beam is 1). A node that saw no pair comes back as the"
        " global fit exactly, so the field degrades to the single law wherever the anchors are"
        " sparse; a node that saw many follows its own",
        why="1.0 because a node of this camera does not hold hundreds of beams: over the four"
        " tapes a non-empty node of a 3x3 field holds a median of 3.7-5.1 of pair weight (p10"
        " 0.2-1.3, p90 10.8-17.3). Swept on the held-out beams"
        " (scratch/_field_prior_sweep.py, 2026-09-15) the four tapes read 11.3 / 15.9 / 15.3 /"
        " 3.8 % of median |residual| at a pull of 20 — the field is the single law again, which"
        " reads 11.4 / 15.6 / 16.3 / 4.5 % — against 8.3 / 15.4 / 14.1 / 4.3 % at 1.0 and"
        " 7.9 / 15.5 / 14.5 / 5.0 % with no pull at all. The pull enters the fit as two rows at"
        " the ends of the pairs' own depth range, so it leans on the line harder than its weight"
        " in pairs suggests",
        on_when="raise it toward 10 on a cart whose anchors are thin and scattered, where a node"
        " fitted on two beams is a whole quadrant of the picture fitted on two beams",
        off_when="0 lets every node follow its own pairs alone (measured better on three tapes of"
        " four and worse at the 40.9 deg pitch); lower it while reading the node table in the"
        " report line, never blind",
    ),
    Flag(
        "field_carry",
        PIPELINE_DEFAULTS["field_carry"],
        range=(0.0, 1000.0),
        description="how hard each node is pulled toward what it was on the LAST frame, in the"
        " same pair-weight units, decaying as exp(-dt / field_carry_tau_s)",
        why="1.0, and it changes almost nothing while the lidar reaches the picture: swept over"
        " 0, 1, 5 and 20 on the four tapes it moves the residual by under a point, and 20 costs"
        " the drive 2.3 (8.3 -> 10.6 %) by carrying a node's stale scale into frames that had"
        " something better to say. What it is there for is the frames with no beams at all —"
        " with floor pairs as the only ruler it is what holds the scale of the nodes that saw"
        " nothing this time",
        on_when="raise it toward 5 on a run whose anchors flicker (a lidar in and out of the"
        " picture, a camera-only stretch), where a node's last value is better than the frame's"
        " global fit",
        off_when="0 makes every frame's field independent of the last, which is what to set when"
        " a node's scale is suspected of lagging the scene",
    ),
    Flag(
        "field_carry_tau_s",
        PIPELINE_DEFAULTS["field_carry_tau_s"],
        range=(0.0, 60.0),
        description="the seconds over which a node's pull toward its own last value decays: a"
        " node starved for one time constant keeps a third of the carry, one starved for five"
        " seconds is the frame's global fit again",
        why="default by design: the frame law's own hold constant"
        " (pepin.depth.FRAME_HOLD_TAU_S, 2 s), so a node's memory and the stage's decay back to"
        " the pool's law run at the same rate. Not measured as a choice of its own — the tapes"
        " that exist all carry beams on every frame, where the carry barely matters",
        on_when="raise it on a cart that drives slowly enough for a node's scene to survive"
        " several seconds",
        off_when="0 drops the carry the moment a node is starved, which is the A/B of the memory",
    ),
    Flag(
        "floor_sigma_pitch_deg",
        PIPELINE_DEFAULTS["floor_sigma_pitch_deg"],
        range=(0.0, 20.0),
        description="what the camera's pitch is trusted to, in degrees, which is what a floor"
        " pair's own noise is made of: the plane's depth under a ray is h / sin(angle below the"
        " horizon), so a pitch error of this size is a depth error of z^2 / h times it, and the"
        " pair's weight is that sigma against a lidar beam's in inverse depth",
        why="1.5 deg is the measurement, not a guess: config/neck.json's ticks_note reads 'head"
        " level by eye the tilt servo reads 2068 ticks and the picture is 1.0 deg down (+-1.5)'."
        " On this mount that makes a floor pixel at 1 m worth about a hundredth of a beam and one"
        " at 3 m a fortieth, where every floor pixel used to carry a flat tenth whatever its"
        " range — 1000 of them outvoting 30 beams by three to one",
        on_when="raise it after a neck re-assembly, or on any run where the head's pitch comes"
        " from an encoder nobody has checked against a level",
        off_when="lower it only after the pitch is measured better than 1.5 deg — a checkerboard"
        " against a plumb line, not a fit through the same depths it would then weigh",
    ),
    Flag(
        "floor_normal_tol_deg",
        PIPELINE_DEFAULTS["floor_normal_tol_deg"],
        range=(0.0, 90.0),
        description="how far the plane fitted to a frame's floor pixels may lean from the cart's"
        " up vector before that frame's floor pairs are thrown away whole; the camera's distance"
        " to that plane must also land within the network's own band of the camera's height. The"
        " report line counts the frames refused and prints the last plane's lean",
        why="a table top, a ramp and a law that is wrong by a fifth all draw a plane the geometry"
        " never meant, and pairs taken off it move every node they touch. 5 deg because the floor"
        " pixels are selected by a height band that is already 12 cm wide at 2 m, which a lean of"
        " 3-4 deg fits inside. It bites: on the tapes of 2026-09-15 it refused 8 of 12 frames on"
        " run 0171's drive, 9 of 9 at the 11.1 deg pitch (where the floor is a shallow sliver at"
        " the bottom of the picture and the plane through it is not identified), 7 of 14 at 25.8"
        " and 1 of 10 at 40.9 deg, where the floor fills the frame (scratch/scale_field_eval.txt)",
        on_when="lower it toward 2 on a floor known to be flat, to refuse everything but the"
        " clean frames",
        off_when="90 accepts every plane, which is the floor anchor as it behaved before the"
        " gate: the A/B of what the gate is refusing",
    ),
    Flag(
        "parallax_weight",
        PARALLAX_WEIGHT,
        description="the multiplier on every parallax pair's own 1 / sigma^2 before it joins the"
        " frame's fit. 1.0 takes the triangulation's noise at face value against a beam's;"
        " 0 keeps the anchor running and its report line honest while its pairs get no vote;"
        " above 1 the corners speak louder than their noise says they should",
        why="the A/B of the second ruler without restarting the node. The weight a pair already"
        " carries is physics: sigma_(1/z) = sigma_px / (f * b_perp) against the beam's"
        " lidar_sigma_m / z^2, capped at a beam's 1 (pepin.parallax). At the measured 7-10 cm of"
        " per-pair sigma at 1.5-2 m that is about 0.03 of a beam, so where the lidar reaches the"
        " corners move the law by a few per cent — which is the point: they are there to hold the"
        " scale where the beams stop, not to argue with them",
        on_when="raise it only with a measurement that says the triangulation is better than its"
        " own sigma claims — a calibrated focal length and a pose better than the tracker's",
        off_when="0 to measure what the corners are doing to the law without losing the frames"
        " they are measured on: the report line still prints their share and their sigma",
        range=(0.0, 10.0),
    ),
    Flag(
        "lidar_sigma_m",
        LIDAR_SIGMA_M,
        description="what one lidar beam's range is trusted to, in metres. 0 (the default) gives"
        " every beam the flat weight of 1 — the reference pair the parallax corners are weighed"
        " against, so both rulers still share one unit. Above 0, a beam's weight is 1 / sigma^2"
        " in inverse depth, sigma_m / z^2, which reads as a weight proportional to z^4",
        why="0 because weighing the beams by range was measured and it is worse. On the four"
        " errands of 2026-09-14 (scratch/parallax_ruler_recheck.txt, 112 frames, the odd beams"
        " fitting and the even ones judging) sigma_m 1.5 cm took the LIDAR-ONLY frame law from"
        " 7.4 % to 11.4 % of median |residual| overall and from 6.4 % to 18.3 % over 1.0-1.5 m,"
        " while 3-12 m improved 10.5 % -> 4.4 %: at z^4 a beam at 8 m counts 256 beams at 2 m,"
        " so the far beams fit themselves and the near field — everything the cart parks against"
        " — pays for it. The maths behind that: the fit minimises the residual of the NETWORK's"
        " 1 / D, whose own noise (0.02-0.10 of inverse depth) is far above a beam's"
        " (0.0002-0.023) at every range, so a beam's sigma is not the residual's sigma and"
        " 1 / sigma_beam^2 is not that pair's share of this fit. The ratio between two DIFFERENT"
        " rulers (a 7-10 cm corner against a beam) is a different question and is what"
        " pepin.depth.pair_weight is still used for",
        on_when="only with a measurement that beats the flat weight on the near bands — e.g."
        " after the network's own per-pair noise enters the weight (1 / (sigma_net^2 +"
        " a^2 sigma_ruler^2)), which is the fit this knob is a crude stand-in for",
        off_when="0 is the shipped default; leave it there",
        range=(0.0, 0.2),
    ),
    Flag(
        "parallax_motion",
        PARALLAX_MOTION,
        description="whose word the parallax anchor's baseline is: tracker takes the motion"
        " between the two frames from the lidar tracker's map pose (TF map -> base_link at both"
        " stamps), odom from the EKF's wheels and gyro (odom -> base_link, what the stage always"
        " used). A window the tracker cannot answer — no map pose at those stamps, a silent or"
        " stale tracker — falls back to the odometry on its own, and the report line counts how"
        " many windows each source actually gave",
        why="a baseline is a length and every triangulated depth is proportional to it, and over"
        " a second the wheels and gyro do not know one: the distance per interval scatters"
        " p10/p90 0.5-2.0 of the tracker's on carpet (scratch/tape_odometry_error.py) while the"
        " tracker's map pose is good to 1-2 cm over a second. The four errands of 2026-09-14"
        " re-measured with each motion on the same frame pairs (scratch/parallax_pose_sweep.txt,"
        " runs 0267/0268/0273/0274): the odometry reads 15.8 cm of travel at a 1.0 s gap and"
        " 25.5 cm at 1.5 s where the tracker reads 13.9 and 18.6, and the depth follows — at"
        " 1-2 m the flow reads 1.263 and 1.342 of the lidar on the odometry's motion against"
        " 1.138 and 0.944 on the tracker's, the describer 1.347 and 1.506 against 0.951 and"
        " 0.968. The 1.25-1.45 over-reading past a second of gap, the one both matchers shared,"
        " was the baseline",
        on_when="tracker wherever the tracker is alive, which is the default: it is the one pose"
        " on this cart measured against the map rather than integrated",
        off_when="odom on a robot with no tracker at all, or to A/B the baseline against the"
        " numbers above without restarting the node",
        choices=PARALLAX_MOTIONS,
    ),
    Flag(
        "parallax_map_wait",
        PARALLAX_MAP_WAIT,
        description="how the parallax anchor asks the tracker for a baseline: off (the default)"
        " uses the newest map pose TF already holds when it is within"
        f" {PARALLAX_MAP_MAX_AGE_S:.1f} s of the frame and falls straight back to the odometry"
        " when it is not; on restores the old ask, which WAITS up to the node's TF timeout for a"
        " map pose at the frame's own stamp. The report line counts the frames that fell back",
        why="the old ask cost the whole 0.2 s timeout on every frame, because map -> base_link is"
        " published behind a frame's stamp and the lookup could never be satisfied in time: live"
        " on 2026-09-15 with parallax_anchor on and parallax_motion tracker the stream fell from"
        " 8.7 to 1.5 frames/s (301 frames dropped in a window) and rgbd_odometry starved to 0"
        " poses/s. A map pose 0.1-0.3 s behind the frame shortens the baseline by that much; a"
        " frame not processed at all is worth nothing",
        on_when="never in the frame path — only to reproduce the 2026-09-15 stall on purpose",
        off_when="always, which is the default",
    ),
    Flag(
        "fan_floor_gate",
        FAN_FLOOR_GATE,
        description="what keeps the floor out of /depth_scan: band (the default) raises the"
        " band's lower edge with the floor's own noise, 3 sigma of it"
        " (pepin.contact.fan_min_z), contact drops every mark nearer than that bearing's"
        " floor-contact range (pepin.contact.gate_by_contact, a range no depth law enters), off"
        " is the flat 0.15 m edge the fan always had. The report line counts the bearings gated",
        why="a floor pixel stands camera_height * (relative depth error) above the floor at every"
        " range, so 12.5 % short is exactly the 0.15 m edge on this mount while the per-frame"
        " law's own median residual is 10.2 % — the floor marks itself as an obstacle, and it is"
        " why the fan read 0.58 of the lidar at the working pitch on 2026-09-15"
        " (scratch/fan_floor_leak.py). Measured on the three pitch tapes of 2026-09-12"
        " (scratch/fan_gate_offline.py, the per-frame law, 9 frames each): band moves k ="
        " fan / lidar 0.499 -> 0.595 at 25.8 deg without removing a single bearing (the mark"
        " simply lands on the real obstacle instead of the floor in front of it), and does"
        " nothing at 11.1 (0.851 -> 0.853) or 40.9 (0.232). contact is the aggressive one and"
        " overshoots: it removes 455 of 1399 marks at 11.1 deg and takes k past 1 to 1.215, and"
        " at 25.8 and 40.9 it removes every mark there is",
        on_when="band as shipped; contact only against a scene where the floor plane is trusted"
        " and the fan is known to be floor — and never without reading the bearings gated",
        off_when="off to reproduce a costmap from before this gate",
        choices=FAN_FLOOR_GATES,
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
        self._last_cam = self._camera_config  # the head's last pose, for the report's geometry
        self._camera_cfg = cfg  # config/camera.json's own optics until a camera_info arrives
        self._intr: Intrinsics | None = None
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        newest = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(Image, "/camera/depth", reliable)
        self._scan_pub = self.create_publisher(LaserScan, "/depth_scan", reliable)
        self._scan_max_range = float(self.declare_parameter("scan_max_range", 3.0).value)
        self._expected_key: object = None  # the optics and head pose the floor ruler was cut for
        self._expected: Array = np.zeros((0, 0))
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
        self._range = RangeLawStage(self._law)
        self._last_verdict_wall = time.time()  # the law's age is the beams', not the node's
        self._seed_laws(time.time())
        self._pipeline = standard_pipeline(self._law, ray=self._ray, range_stage=self._range)
        self._switches = Switches(self, FLAGS, on_change=self._on_switch)
        for name in self._pipeline.names:  # a launch override reaches the stage it names
            self._pipeline.set(name, self._switches.on(name))
        set_scale_ceiling(float(self._switches["scale_ceiling"]))  # and the law's bound
        self._law.slew_per_s = float(self._switches["law_slew"])  # how fast the law may move
        self._ask_parallax(float(self._switches["parallax_min_baseline_m"]))  # and the ring's ask
        self._ask_matcher(str(self._switches["parallax_matcher"]))  # and who matches its corners
        self._ask_motion(str(self._switches["parallax_motion"]))  # and whose motion it triangulates
        self._ask_map_wait(bool(self._switches["parallax_map_wait"]))  # asked without waiting
        self._ask_parallax_weight(float(self._switches["parallax_weight"]))  # and its vote
        self._ask_lidar_sigma(float(self._switches["lidar_sigma_m"]))  # and what a beam is worth
        self._ask_frame_shift(bool(self._switches["frame_shift_needs_beams"]))  # and the gate
        self._ask_field()  # and the shape of the per-frame law: its grid and its two pulls
        self._ask_floor()  # and what a floor pair is worth and when a floor is not a floor
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
        self._history = TfHistory(self._tf, timeout_s=CARRY_WAIT_S)
        self._poser = FramePoser(
            self._history,
            lean=self._lean,
            apply_lean=self._switches.on("imu_lean"),
            min_lean_quality=float(self._switches["lean_min_quality"]),
        )
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
        """Hand every law what the last run saved in the law file, and say so in the log: the
        affine numbers to the affine law *and* to the ray law (each pools and fits on its own,
        so a ray law left unseeded would withhold every frame of the warm-up while the affine
        law publishes), the angular gain to the ray law and the range law's bins to the range
        law when the file holds them and they still stand
        (:meth:`pepin.elevation.RayGain.restore`, :meth:`pepin.depth.RangeLaw.restore` judge
        them). Without a file nothing is published until POOL_MIN_SAMPLES beam pairs are
        pooled."""
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
        ranged = load_range(self._law_file, now)
        if ranged is not None:
            self._range.seed(ranged)
        range_note = (
            f"; range law {ranged.describe()}" if ranged is not None else "; no range law saved"
        )
        self.get_logger().info(
            f"depth law from {self._law_file}: a {saved[0]:.2f} b {saved[1]:+.3f}"
            f" on {saved[2]} beams; publishing at once{ray_note}{range_note}"
        )

    def _on_switch(self, name: str, _old: Any, new: Any) -> None:
        """A flag changed: ``depth_backend`` is the switch's mode, ``scale_ceiling`` the law's
        upper bound, ``imu_lean`` the poser's and the estimator's, ``lean_min_quality`` the
        poser's floor under a lean, a stage's flag switches that stage of the pipeline."""
        if name == "depth_backend":
            self._net.mode = str(new)
        elif name == "scale_ceiling":
            set_scale_ceiling(float(new))  # the next fit is bounded by it; the law in hand is not
        elif name == "law_slew":
            self._law.slew_per_s = float(new)  # from the next fit on
        elif name == "imu_lean":
            self._poser.apply_lean = bool(new)
            self._lean.use_gyro = bool(new)
        elif name == "lean_min_quality":
            self._poser.min_lean_quality = float(new)
        elif name == "parallax_min_baseline_m":
            self._ask_parallax(float(new))
        elif name == "parallax_matcher":
            self._ask_matcher(str(new))
        elif name == "parallax_motion":
            self._ask_motion(str(new))
        elif name == "parallax_map_wait":
            self._ask_map_wait(bool(new))
        elif name == "parallax_weight":
            self._ask_parallax_weight(float(new))
        elif name == "lidar_sigma_m":
            self._ask_lidar_sigma(float(new))
        elif name == "frame_shift_needs_beams":
            self._ask_frame_shift(bool(new))
        elif name.startswith("field_"):
            self._ask_field()
        elif name.startswith("floor_") and name not in self._pipeline.switches:
            self._ask_floor()
        elif name in self._pipeline.switches:
            self._pipeline.set(name, bool(new))

    def _ask_map_wait(self, wait: bool) -> None:
        """Tell the parallax anchor whether it may WAIT for a map pose at the frame's stamp (it
        may not, by default: the wait cost the stream 7 frames a second on 2026-09-15)."""
        stage = self._pipeline.stage("parallax_anchor")
        if isinstance(stage, ParallaxAnchor):
            stage.map_wait = wait

    def _ask_parallax(self, baseline_m: float) -> None:
        """Tell the parallax anchor how much baseline to pick its partner frame to reach."""
        stage = self._pipeline.stage("parallax_anchor")
        if isinstance(stage, ParallaxAnchor):
            stage.min_baseline_m = baseline_m

    def _ask_matcher(self, matcher: str) -> None:
        """Tell the parallax anchor who finds the corners two frames share; the gap window it
        looks back over follows the choice (0.60 s for the flow, 1.5 s for the describer)."""
        stage = self._pipeline.stage("parallax_anchor")
        if isinstance(stage, ParallaxAnchor):
            stage.matcher = matcher

    def _ask_motion(self, source: str) -> None:
        """Tell the parallax anchor whose motion its baseline is, the tracker's map pose or the
        EKF's odometry; a window the chosen source cannot answer falls back to the odometry."""
        stage = self._pipeline.stage("parallax_anchor")
        if isinstance(stage, ParallaxAnchor):
            stage.motion_source = source

    def _ask_parallax_weight(self, weight: float) -> None:
        """Tell the parallax anchor how much of its own 1 / sigma^2 its pairs vote with; 0 keeps
        the pairs and their report line and takes their vote out of every fit."""
        stage = self._pipeline.stage("parallax_anchor")
        if isinstance(stage, ParallaxAnchor):
            stage.weight = weight

    def _ask_lidar_sigma(self, sigma_m: float) -> None:
        """Tell the lidar anchor what one beam's range is trusted to, in metres: its pairs then
        weigh 1 / sigma^2 in inverse depth. 0 restores the flat weight of 1 a beam used to have."""
        stage = self._pipeline.stage("lidar_anchor")
        if isinstance(stage, LidarAnchor):
            stage.sigma_m = sigma_m

    def _ask_field(self) -> None:
        """Tell the per-frame law what shape it is: how many nodes it carries over the picture
        and how hard each one is pulled toward the frame's global fit and toward its own last
        value. A new grid starts every node again from the next frame's fit."""
        stage = self._pipeline.stage("frame_law")
        if not isinstance(stage, FrameLaw):
            return
        field = stage.field
        field.prior = float(self._switches["field_prior"])
        field.carry = float(self._switches["field_carry"])
        field.carry_tau_s = float(self._switches["field_carry_tau_s"])
        grid = grid_of(str(self._switches["field_grid"]))
        if field.grid != grid:
            field.grid = grid

    def _ask_floor(self) -> None:
        """Tell the floor anchor what the mount's pitch is trusted to (a floor pair's own sigma)
        and how far the plane fitted to its pixels may lean before the frame's floor is
        refused."""
        stage = self._pipeline.stage("floor_pairs")
        if isinstance(stage, FloorPairs):
            stage.sigma_pitch_deg = float(self._switches["floor_sigma_pitch_deg"])
            stage.normal_tol_deg = float(self._switches["floor_normal_tol_deg"])

    def _ask_frame_shift(self, needs_beams: bool) -> None:
        """Tell the per-frame law whether a shift needs the lidar in the pool that fits it; a
        parallax-only pool then gets a scale alone."""
        stage = self._pipeline.stage("frame_law")
        if isinstance(stage, FrameLaw):
            stage.shift_needs_beams = needs_beams

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
        at = stamp_seconds(stamp)
        pose = self._history.pose_at_nowait(at, self._camera_frame, self._base_frame)
        if pose is None and self._switches.on("camera_tf_latest"):
            # The neck's edge crosses the bridge late (bursts of +0.75 s, 2026-09-15) and the
            # head does not move while the cart drives: a lookup at the frame's own stamp
            # waited CARRY_WAIT_S on every frame ("Extrapolation ... into the future" x104 a
            # window, 3.5 frames/s, VO starved). The newest edge, if young, is the head's pose.
            latest = self._history.latest_pose(self._camera_frame, self._base_frame)
            if latest is not None and at - latest[1] <= CAMERA_TF_MAX_AGE_S:
                pose = latest[0]
                self._tally.count("camera_tf_latest")
        if pose is None:
            pose = self._poser.camera_in_base(at)  # the old path: wait for the stamp
        if pose is None:
            self._tally.count("camera_from_config")
            self._last_cam = self._camera_config
            return self._camera_config, None
        _pitch, pan = optical_heading(pose.rotation)
        if abs(pan) > PAN_NOTICE_RAD:
            self._tally.count("camera_panned")
        self._last_cam = CameraPose.from_optical(pose.rotation, pose.translation)
        return self._last_cam, pose

    def _as_scan(self, depth: Array, image: Image, ctx: FrameContext) -> LaserScan:
        """The depth folded onto the floor plane, in base_link, stamped like the image — with
        the floor gated out of it the way ``fan_floor_gate`` says (:data:`FAN_FLOOR_GATES`)."""
        gate = str(self._switches["fan_floor_gate"])
        floor_of: float | Array = SCAN_MIN_Z_M
        if gate == "band":
            floor_of = fan_min_z(self._floor_expected(ctx), ctx.cam.z)
        angle_min, step, ranges = depth_to_scan(
            depth, ctx.intr, ctx.cam, min_z=floor_of, max_range=self._scan_max_range
        )
        before = int(np.count_nonzero(np.isfinite(ranges)))
        if gate == "contact":
            plane = FloorPlane.of(ctx.intr, ctx.cam, ctx.up)
            _min, _step, contact, _verdict = contact_scan(depth, plane)
            ranges, _removed = gate_by_contact(ranges, contact)
        self._tally.count("fan_gated", before - int(np.count_nonzero(np.isfinite(ranges))))
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
        frame cannot drag the beams into the picture and refit the law from them."""
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
                    range_law=self._range.saved_state(),
                )
            except OSError as exc:
                self.get_logger().warning(
                    f"cannot save the depth law to {self._law_file}: {exc}",
                    throttle_duration_sec=300,
                )

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

    def _extras(self, w: Window) -> str:
        """The parts of the report line a window may have nothing to say about: how old the
        anchoring scans were, the scans no odometry could carry, the frames whose camera pose
        came from the config instead of TF, the frames with the head turned, the frames whose
        beams fell outside the picture or were too few to judge it, the cart's lean, and the
        last TF failure of each kind."""
        c, extra = w.counts, ""
        ages = w.samples.get("scan_age", [])
        if ages:
            extra += f", scan age median {float(np.median(ages)):.2f} s max {max(ages):.2f} s"
        if c["uncarried"]:
            extra += f", scans uncarried {c['uncarried']}"
        if c["carry_insane"]:
            extra += f", carry insane {c['carry_insane']} frames (the odometry ran away)"
        if c["camera_from_config"]:
            extra += f", camera pose from config {c['camera_from_config']} frames (no TF edge)"
        if c["fan_gated"]:
            extra += f", floor-gated {c['fan_gated']} bearings ({self._switches['fan_floor_gate']})"
        if c["camera_panned"]:
            extra += f", head panned {c['camera_panned']} frames (projected as if not)"
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
