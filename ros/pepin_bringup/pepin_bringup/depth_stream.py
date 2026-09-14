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
from pepin.depth import (
    MIN_SAMPLES,
    POOL_MIN_SAMPLES,
    SCALE_CEILING,
    SCAN_WINDOW_S,
    Array,
    CameraPose,
    Intrinsics,
    carry,
    carry_speed,
    depth_to_scan,
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
    PARALLAX_MIN_BASELINE_M,
    AffineLaw,
    FrameContext,
    ParallaxAnchor,
    RangeLawStage,
    RayLaw,
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
        "range_law",
        True,
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
        True,
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
        elif name in self._pipeline.switches:
            self._pipeline.set(name, bool(new))

    def _ask_parallax(self, baseline_m: float) -> None:
        """Tell the parallax anchor how much baseline to pick its partner frame to reach."""
        stage = self._pipeline.stage("parallax_anchor")
        if isinstance(stage, ParallaxAnchor):
            stage.min_baseline_m = baseline_m

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
        pose = self._poser.camera_in_base(stamp_seconds(stamp))
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
