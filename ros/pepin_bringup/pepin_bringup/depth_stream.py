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
the chain so none of them withholds while another publishes, and the range law's own record
beside them, written whenever its stage has fitted one on the live pool and restored only when
it still stands on its own terms. One affine law is not the shape of
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
poser carries the scan to the frame's moment through the odometry. Neither ask ever waits for
an edge that has stopped: TF goes through :class:`LiveEdgeHistory`, which refuses any blocking
lookup of an edge whose newest sample is more than ``tf_dead_s`` behind the frame — the config
mount and the uncarried scan at once, both counted, instead of CARRY_WAIT_S burnt per frame on
a route that has died (2026-09-16: the neck's edge 344 s old, 0.9-3 frames/s). What the
PIPELINE asks about the cart's motion goes to a second poser over the same TF with no wait at
all, because the parallax anchor asks once per stored view and a wait there is paid per view,
not per frame; a view TF cannot answer yet is left out of that frame's bundle. A pan of the
head is counted too, and behind ``scan_honours_pan`` the published fan turns with it — the
bearings of /depth_scan are the cart's whichever way the neck looks, and the fan's angular
window sits off base_link's x by the pan. The depth image the pipeline corrects is still
projected as if the head looked along that x: the anchors read pixels and heights, not bearings.

WHERE THE RAW DEPTH COMES FROM is one node parameter, ``depth_source``, and everything after it
is the same chain. ``network`` (the default) is the mono network described above.  ``stereo`` is
the calibrated stereo head: the node then also subscribes to ``/camera/right/image`` and
``/camera/right/camera_info``, pairs the right eye with the left picture by EXACT stamp (both
halves of one transport frame carry the same one, so a right eye that has not arrived within
``stereo_pair_wait_s`` is never coming and the frame is dropped and counted), and
:class:`pepin.stereo_depth.StereoDepth` measures the depth instead of guessing it —
``z = fx * baseline / disparity``, metric by construction, NaN wherever the match is not
trusted and NaN past the rig's own reach (2.2 m, where the disparity error model crosses 10 cm;
see that module). ``fx`` and the baseline are read off the two ``camera_info`` messages, so this
node never opens the calibration file. The picture, its stamp and its frame are the left eye's
throughout, exactly as the mono path publishes them.

WHAT THE CORRECTION STAGES MEAN under a metric source. Until the head was calibrated every one
of them stayed on (2026-09-20: nothing to measure against, so switching any off was a guess). With
the checkerboard calibration of 2026-09-21 (epipolar 0.23 px; a printed board's span read to
+0.8 % on frames the fit never saw) the defaults under ``depth_source: stereo`` are
:data:`STEREO_DEFAULTS`, every one still a live flag:
* The scale-recovering stages are OFF — ``floor_pairs``, ``wall_anchor``, ``parallax_anchor``,
  ``range_law``, ``frame_law``. They exist to give the mono network a scale; a stereo depth has
  one, and they cost ~30 ms a frame between them.
* The lidar's pairs are still collected and the affine law still fitted, but it WATCHES
  (``law_watch``): the depth goes out as measured, and the law's numbers in the report line are
  the head's health — a 1.00 b +0.000 while the rig is as calibrated. Applied, the law hurt: in a
  cluttered room the lidar's plane, 0.8 m under the lens, pairs its far returns with whatever
  stands in front of them, and b went to its -0.200 bound. The law file is per source
  (``law_file``) and with no file ``stereo`` seeds the identity law.
* ``edge_filter`` and ``floor_anchor`` stay on: they clean a measurement rather than rescale it —
  the halo around every hole the matcher left, and the matcher's ripple on a glossy floor.

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
``parallax_anchor``, ``affine_law``, ``range_law``, ``frame_law``,
``wall_correct``, ``floor_anchor`` — plus ``depth_backend``, ``scale_ceiling``, the largest
1 / scale the law may be fitted to, ``law_slew``, how fast that law may move between fits,
``tf_dead_s``, how stale a TF edge may be before no frame waits for it, ``imu_lean``,
``lean_min_quality`` and ``scan_hz``, the cap on how often ``/depth_scan`` is published (5 Hz,
the board's local costmap's own ``update_frequency`` — every frame is still processed, the cap
is on the publisher); their state is printed in every report line.
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
    nearest_stamp,
    optical_heading,
    plane_in_view_from,
    retired_laws,
    save_law,
    scan_points,
    set_scale_ceiling,
    to_base,
)
from pepin.depth_pipeline import (
    FIELD_GRIDS,
    LIDAR_SIGMA_M,
    PARALLAX_CORRECTION_REACH_M,
    PARALLAX_CORRECTION_TOL_M,
    PARALLAX_DRIFT_TOL_PX,
    PARALLAX_MAP_MAX_AGE_S,
    PARALLAX_MAP_WAIT,
    PARALLAX_MATCHER,
    PARALLAX_MAX_TRACKS,
    PARALLAX_MIN_BASELINE_M,
    PARALLAX_MIN_GAP_S,
    PARALLAX_MIN_TOTAL_BASELINE_M,
    PARALLAX_MOTION,
    PARALLAX_MOTIONS,
    PARALLAX_REDETECT_EVERY,
    PARALLAX_SIGMA_MODEL,
    PARALLAX_SPLIT_TOL_SIGMA,
    PARALLAX_TRACK_MAX_VIEWS,
    PARALLAX_TRACK_MIN_OBS,
    PARALLAX_TRACK_WINDOW_S,
    PARALLAX_TRACKING,
    PARALLAX_TRACKINGS,
    PARALLAX_UNDISTORT,
    PARALLAX_VERIFY_EVERY,
    PARALLAX_WEIGHT,
    PIPELINE_DEFAULTS,
    AffineLaw,
    FloorPairs,
    FrameContext,
    FrameLaw,
    LidarAnchor,
    ParallaxAnchor,
    RangeLawStage,
    WallAnchor,
    grid_of,
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
from pepin.flags import Flag, FlagSet
from pepin.frame_pose import FramePoser
from pepin.lean import LEAN_QUALITY_FLOOR
from pepin.parallax import MATCHERS, TRACK_SIGMA_MODELS, to_gray
from pepin.stereo_depth import (
    Baseline,
    MatcherSettings,
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
# published image's reach (the ``depth_reach`` flag) and /depth_scan's own cap
# (``scan_max_range``) — because they are the same physical claim, and two literals would drift:
# the costmap's obstacle_max_range of 2.5 m has to stay under the scan's cap, and the camera half
# of RTAB-Map's grid has to stay under the image's. Measured: see the ``depth_reach_m`` flag.
DEPTH_REACH_M = 3.0
TRACK_FLAGS = (  # the ones that decide the shape of a parallax measurement: track or pair
    "parallax_tracking",
    "parallax_track_min_obs",
    "parallax_track_window_s",
    "parallax_min_total_baseline_m",
    "parallax_track_max_views",
    "parallax_sigma_model",
    "parallax_split_tol_sigma",
    "parallax_max_tracks",
    "parallax_redetect_every",
    "parallax_verify_every",
    "parallax_drift_tol_px",
    "parallax_correction_tol_m",
    "parallax_undistort",
)

# What a METRIC source changes in the chain, by flag name (the rest keep FLAGS' defaults; every
# one stays a live flag, so the mono chain can be A/B-ed on a stereo depth at any time). Off:
# the stages that exist to recover a scale the mono network does not have — the rulers from
# assumed geometry, the parallax ring, the range and frame laws. The lidar's pairs are still
# collected and the affine law still fitted, as a witness (``law_watch``). What is left cleans a
# measurement rather than rescaling it: ``edge_filter``, ``floor_anchor``, the reach.
STEREO_DEFAULTS: dict[str, Any] = {
    "floor_pairs": False,
    "wall_anchor": False,
    "parallax_anchor": False,
    "range_law": False,
    "frame_law": False,
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
        "floor_pairs",
        PIPELINE_DEFAULTS["floor_pairs"],
        description="the floor's pixels pair the network's depth with the plane's geometric depth,"
        " a second hoop for the law that needs no lidar. Each pair weighs its own sigma — the"
        " plane's depth under a ray is h / sin(angle below the horizon), so the mount's pitch"
        " uncertainty makes it grow as the square of the range (floor_sigma_pitch_deg) — and the"
        " frame's whole floor is refused unless the plane fitted to those pixels stands up"
        " (floor_normal_tol_deg); the report line counts the frames refused",
        why="since 2026-09-16, when the height band was capped and the plane judged in metres"
        " (floor_band_max_m, floor_plane_band): the floor pairs feed 55-125 of every 79-125 door"
        " frames at 10 % of the fit weight — the beams keep the rest — and cost the lidar chain"
        " nothing, lifting the lidar row on run 0171 from 17.0 to 11.3 % of median |residual| on"
        " the block split (scratch/scale_field_eval.py), and alone, with no lidar in the chain at"
        " all, they read a door 2 m away to 4-9 % (scratch/wall_truth_eval.py): the scale field"
        " keeps them off the lidar's own nodes. Before the cap they were the same knob that moved"
        " the band 12.9/38.2 cm -> 24.1/48.7 on run 0171 (scratch/pipeline_vs_truth.txt,"
        " 2026-09-11) under ONE law, because the network's error is regime-wise (floor 1.1x, the"
        " lidar's row 1.6x, above it 2.0x) and one law fitted across the two lands between them;"
        " under the 3x3 scale field the regimes are separate nodes and an uncapped band still"
        " took run 0171 from 11.4 to 17.6 % of median |residual| (scratch/scale_field_eval.txt,"
        " 2026-09-15). The cap is what made the pairs safe, not the field alone",
        on_when="always, as shipped: on a frame the lidar covers the field keeps the floor off"
        " the beams' own nodes, and on a frame with no beams at all — nearer than the 0.71 m at"
        " which the lidar's plane enters the picture, or on a cart with no lidar — it is a metric"
        " ruler that needs nothing but the mount's height",
        off_when="to reproduce a chain from before 2026-09-16, or on a floor the plane cannot be"
        " fitted to (glass, deep pile, a slope the mount does not know) when the report line's"
        " count of refused frames is already most of them",
    ),
    Flag(
        "wall_anchor",
        PIPELINE_DEFAULTS["wall_anchor"],
        description="the lidar's returns extruded up the image, where the network's depth stays"
        " continuous, pair the rows above the lidar's with the wall's depth — a third hoop. Each"
        " pair carries its own sigma: the beam's 1.5 cm through the plane's geometry, plus"
        " wall_sigma_height per metre of height above the line (the price of the world"
        " assumption), and the pairs of one column SHARE that beam's weight instead of each"
        " carrying it. A column must climb 0.5 m undisturbed to count at all",
        why="ON since 2026-09-15, on two judges, after shipping off since 2026-09-14. What"
        " changed is the weighing and the field. Under one global law a wall pixel carried a flat"
        " fifth of a beam whatever its height, fifty of them stood on one beam, and the ruler"
        " pulled the lidar's own row 10 % near (1.010 -> 0.896, scratch/pipeline_vs_truth.txt)."
        " Now: (1) the non-circular judge, COLMAP's reconstruction of the furnished home scene"
        " 0171 split into points ON the lidar's vertical extrusion and points OFF it (furniture,"
        " clutter, the far room — the pixels this ruler can only damage), 5159 off-plane"
        " observations: median |corrected/true - 1| falls in EVERY height band, 20.5 -> 17.9 %"
        " below the line, 16.7 -> 13.7 % up to 0.3 m, 45.9 -> 30.2 % at 0.3-0.6 m, 99.8 -> 76.7 %"
        " at 0.6-1.0 m, 19.1 -> 16.1 % overall, and the 210 on-plane points 9.6 -> 9.0"
        " (scratch/wall_vs_colmap.txt). (2) The lidar's own row on a CONTIGUOUS held-out split"
        " (the first half of the scan fits and walks, the second half judges): 12.9 -> 11.5 % on"
        " run 0171, 17.5 -> 16.1 on tape 0235, 32.7 -> 25.0 on 0236, and on 0237 (neck 40.9 deg"
        " down) the climb gate refuses every column, so the ruler is a no-op"
        " (scratch/wall_field_row_eval.txt). The wall brings 4-9 % of a frame's fit weight there"
        " against 67-85 % when every pair votes for itself. What it cannot do is tell a leaning"
        " surface from its own error: a sofa back leaning 0.3 m per metre of height reads like a"
        " wall to every shape gate, and only 3-6 % of the COLMAP points the camera sees actually"
        " stand on the lidar's extrusion — the gates cut what is walked to 2-11 % of it, and"
        " wall_sigma_height prices the rest",
        on_when="it ships on; it is also the only wall cue on a robot with no lidar",
        off_when="if a costmap regression ever traces to the rows above the beams, or on a scene"
        " of low furniture where the extrusion has nothing to extrude — it is a no-op there"
        " rather than a cost, but off is the way to prove that",
    ),
    Flag(
        "parallax_anchor",
        PIPELINE_DEFAULTS["parallax_anchor"],
        description="the corners this frame shares with the previous one, triangulated against the"
        " odometry's transform between the two stamps (pepin.parallax), pair the network's depth"
        " with a depth in metres the cart measured by moving — a hoop that needs no lidar and no"
        " assumed plane and that lands at every elevation the picture has",
        why="since 2026-09-16, once the ask stopped blocking (parallax_map_wait off) and the"
        " corners were followed forward (parallax_tracking forward): the forward tracks cost 6 ms"
        " a frame and the depth stream held 7-8 frames/s live through the door drives, and alone"
        " — with no lidar in the chain — they read a door 2 m away to 5-6 % of median |residual|"
        " above the lidar's row on the straight legs (scratch/wall_truth_eval.py). It had been"
        " OFF from 2026-09-15 04:20, when the ask for the tracker's motion waited its 0.2 s carry"
        " timeout on EVERY frame and the stream fell from 8.7 to 1.5 frames/s (parallax_map_wait"
        " carries that measurement). It is the second ruler of the scale, and it is weighed like"
        " one. Every pair"
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
        " another, and depth_fusion refuses those frames at the yaw search's bound. What it costs"
        " is MEMORY, measured 2026-09-15 and left standing: the law is fitted over a pool of the"
        " last frames, so it describes the last minute's scene, and above the lidar's row the"
        " door tapes read a per-run offset that flips sign between the approach and the retreat"
        " (0.984 / 1.060 / 0.987 / 1.009 on 0318/0320/0321/0322, a spread of 0.076). Switched off"
        " the flip goes away — all four read +2.1 to +4.4 %, a spread of 0.023 — and the residual"
        " above the row halves on three of the four (12.8 -> 4.7 % in the top third of 0318;"
        " scratch/wte_03*_norange.txt). It stays ON because the judge that is not circular says"
        " the opposite: on the COLMAP scene of run 0171, off costs 16.1 -> 19.8 % off the wall"
        " plane and 9.0 -> 10.9 on it (scratch/wall_vs_colmap.txt), and the held-out lidar row is"
        " unmoved on three tapes of four. The memory is real and the cure is not this switch",
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
        why="a wash, measured twice, and a wash is not a reason to overwrite a measurement with"
        " an assumption. Beside wall_anchor on the COLMAP scene of run 0171 it moves nothing that"
        " can be read: 16.1 % of median |corrected/true - 1| off the wall plane against"
        " wall_anchor's own 16.1, and the same 9.0 % on it, band for band"
        " (scratch/wall_vs_colmap.txt, 2026-09-15). Standalone it was a wash before that too —"
        " the same law and the same lidar row on run 0171, the band 12.8/39.3 cm against"
        " 12.9/38.2 (scratch/pipeline_vs_truth.txt). So the pairs role carries the ruler and the"
        " pixels stay the network's own",
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
        "parallax_tracking",
        PARALLAX_TRACKING,
        description="which ruler follows the corners. forward detects a corner once and follows"
        " it FORWARD one hop a frame, keeping it alive for as long as it survives — two flow"
        " calls a frame whatever the window. window is the backward build: the CURRENT frame's"
        " corners re-tracked through every view of the window on every frame, two flow calls"
        " per view. pair is this frame against one partner chosen out of the ring, what the"
        " stage did until 2026-09-15. parallax_track_min_obs under 3 is the pair whatever this"
        " says",
        why="the window's cost IS its window, and the window is what the measurement wants:"
        " every error term of a parallax depth divides by the baseline, and the tracker's map"
        " pose is absolute, so reaching further back costs the pose nothing. Measured over the"
        " four errands of 2026-09-14, 69 judged frames, every clip frame fed to the forward"
        " store (scratch/parallax_forward_eval.txt, 2026-09-15). COST: forward is 5.9 ms a"
        " frame at a 1.5 s window and 6.4 at 5 s (hop 2.5, detect 0.0, drift bound 0.5, solve"
        " 3.2) against the backward window's 31.3 at 1.5 s, which grows with every view added."
        " DEPTH: the corners' own depth against the lidar's ranges reads 0.94 / 0.98 / 1.02 of"
        " it at 1.5 / 3 / 5 s, against the backward window's 0.887 and the pair's 0.788 — the"
        " long window is what pulls the ratio onto 1.00. RESIDUAL of a parallax-only law at the"
        " beams: 12.7 % at 3 s and 13.8 % at 5 s over the frames each could fit, against the"
        " window's 16.0 % and the pair's 30.2 %; but on the 264 beams of the frames EVERY way"
        " fitted, the backward window still reads 13.4 % against forward's 17.5-18.1 %."
        " CORNERS are where forward pays: 6-12 a frame against the window's 39.5, because it"
        " follows parallax_max_tracks corners where the window asks the detector for 400 fresh"
        " ones on every frame, and because 3800 of its corners per 1200 frames are closed by"
        " the drift bound. It therefore fits a law on 5-9 of 69 frames where the window fits on"
        " 19. Raising parallax_max_tracks to 400 is the measured fix (12.5 corners a frame,"
        " 9 frames, 9.4 ms — still a third of the backward build)",
        on_when="forward wherever the window is worth more than half a second: it is the only"
        " one that can afford a long one, and the depths it gives sit on the lidar's ranges",
        off_when="window to reproduce a number measured between 2026-09-15 and this change, or"
        " where 39 corners a frame matter more than 25 ms; pair for the A/B against everything"
        " measured before 2026-09-15",
        choices=PARALLAX_TRACKINGS,
    ),
    Flag(
        "parallax_max_tracks",
        PARALLAX_MAX_TRACKS,
        description="how many corners the forward store follows at once. New ones are detected"
        " into the gaps between the live ones, balanced over a grid so the top of the picture"
        " is filled as well as the floor; the cap is the width of one flow call, not the number"
        " of calls, so it is close to free",
        why="200 is the brief's number and 400 is the one the measurement likes: over the four"
        " errands of 2026-09-14 (scratch/parallax_forward_eval.txt) a 3 s window keeps 7.0"
        " corners a frame at 200 and 12.5 at 400, fits a law on 5 frames of 69 against 9, and"
        " costs 6.0 ms a frame against 9.4 — where the backward window, which asks the detector"
        " for 400 fresh corners on EVERY frame, keeps 39.5 at 31.3 ms. The residual barely"
        " moves (12.7 % against 13.1 % over each one's own frames, 18.1 % against 17.8 % on the"
        " frames every way fitted): the corner count buys FRAMES that can be fitted at all, not"
        " a better fit",
        on_when="400 on this laptop, which is the corner budget the backward build always had",
        off_when="lower wherever the flow's milliseconds matter: the cost is linear in the"
        " corners and the report line prints it as the hop",
        range=(20, 2000),
    ),
    Flag(
        "parallax_redetect_every",
        PARALLAX_REDETECT_EVERY,
        description="frames between two hunts for new corners in the forward store. A hunt also"
        " starts early whenever the live corners fall under 60 % of parallax_max_tracks — a"
        " turn or a doorway can take three corners in four in one frame, and waiting for the"
        " cadence would waste the window",
        why="the detector is 0.0-0.1 ms a frame at this cadence over the four errands of"
        " 2026-09-14 (scratch/parallax_forward_eval.txt) because the early floor does most of"
        " the work: the store loses about 6 corners a frame to the flow, the"
        " forward-backward check and the drift bound together, so the floor fires long before"
        " the fifth frame. 5 is therefore a cheap upper bound rather than the real cadence",
        on_when="lower on a camera whose view changes fast (a turn in place), where a corner"
        " born late still has the whole window ahead of it",
        off_when="higher wherever the detector shows in the report line's detect ms",
        range=(1, 120),
    ),
    Flag(
        "parallax_track_min_obs",
        PARALLAX_TRACK_MIN_OBS,
        description="how many frames a corner must be seen in before its depth is a"
        " measurement. 3 or more makes a corner a TRACK: it is followed back through the window"
        " of ring frames and all of its rays are met in one least-squares solve, with the"
        " single worst observation dropped and the rest solved again. 2 is the PAIR the anchor"
        " measured until 2026-09-15 — this frame against one partner chosen for its baseline —"
        " and is arithmetically the same code at two views",
        why="a pair rests on one baseline and a track on all of them. The four errands of"
        " 2026-09-14 measured both ways on the same pictures, the same poses and the same judge"
        " (scratch/parallax_tracks_audit.txt, 66 frames, klt, the tracker's map pose, the"
        " corners fitting and the lidar's beams judging): a pair rests on 5.2 cm of parallax"
        " and a 1.5 s track on 14.0 cm, and the law that follows is better where it can be"
        " compared at all. On the 12 frames a pair and a 16-view track BOTH fitted a law, the"
        " parallax-only residual at the beams is 24.1 % against 19.0 %; on the 12 a pair and the"
        " shipped 8-view track both fitted, 28.0 % against 14.2 %. The sigma barely moves —"
        " 9.5 cm a pair, 9.1 at 16 views, 8.4 at 8 — because the first eval's 10.6 -> 6.0 cm was"
        " the closed-form formula and not the measurement (parallax_sigma_model, 2026-09-15)."
        " The cost is the flow's extra hops: 5.7 ms a frame becomes 27.0 at the shipped 8 views"
        " and 52.5 at 16. Every 'all frames' comparison is over DIFFERENT frames for each way,"
        " since a pair and a track do not fail on the same ones — only the common-frame rows"
        " above are an A/B",
        on_when="raise it towards 4-5 on a robot whose camera runs faster than this one's"
        " 6-9 frames/s, where the extra views cost little tracking and buy baseline",
        off_when="2 restores the pair exactly, which is the A/B; and on a board too slow for the"
        " flow's extra hops, whose cost is in the report line's ms",
        range=(2.0, 12.0),
    ),
    Flag(
        "parallax_track_window_s",
        PARALLAX_TRACK_WINDOW_S,
        description="how far back in time a track may reach, in seconds. Following corners"
        " forward (parallax_tracking) it is the age at which an observation is dropped and"
        " nothing more — the corner lives on, the cost does not move, and a window changed"
        " live takes effect on the very next frame with nothing reset. On the backward window"
        f" it is also the cost: every ring frame between {PARALLAX_MIN_GAP_S:.2f} s and this is"
        " a view to re-track through, and it sets how long the ring holds a frame",
        why="3.0 since 2026-09-15, when the forward store stopped charging for the window."
        " Every error term of a parallax depth divides by the baseline (pixel noise as"
        " z^2 sigma_px / (f B), the pose's own centimetre as 1 / B) and the tracker's map pose"
        " is ABSOLUTE, so reaching further back costs the pose nothing. Measured forward over"
        " the four errands of 2026-09-14 (scratch/parallax_forward_eval.txt, 69 judged frames):"
        " the corners' own depth reads 0.938 of the lidar at 1.5 s, 0.976 at 3 and 1.020 at 5,"
        " and a parallax-only law leaves 21.3 %, 12.7 % and 13.8 % of residual at the beams"
        " over the frames each could fit. What does NOT arrive is the baseline the premise"
        " promised: 13.4, 14.7 and 15.2 cm of effective parallax, because these errands turn"
        " and pause rather than drive straight, and only 43 of 69 frames reached 3 s of window"
        " and 29 of 69 reached 5. The cost is flat — 5.9 ms a frame at 1.5 s, 6.4 at 5 —"
        " against the backward window's 31.3 at 1.5 s alone. Swept BACKWARD over 48 frames"
        " before that (scratch/parallax_tracks_eval.txt, 2026-09-15): at"
        " 0.5 s a track has 5 observations, 11.2 cm of effective baseline, a 7.8 cm sigma, a law"
        " on 15 of 48 frames and 21.6 % of residual; at 1.0 s, 6 observations, 12.9 cm, 6.3 cm,"
        " 26 frames, 17.8 %; at 1.5 s, 6 observations, 14.0 cm, 6.0 cm, 31 frames, 15.3 %."
        " Nothing has turned over yet at 1.5 s, and the reason is the pose: the 1.25-1.45"
        " over-reading a PAIR shows past a second of gap was the odometry's baseline"
        " (scratch/parallax_pose_sweep.txt) and the tracker's map pose does not have it — the"
        " corners' own depth reads 0.861 of the lidar at 1.5 s against 0.852 at 0.5 s and the"
        " pair's 0.782. The cost is roughly linear in the window: 19.3 ms a frame at 0.5 s, 39.4"
        " at 1.0, 47.7 at 1.5",
        on_when="longer on a robot whose pose over that window is better than this cart's"
        " tracker — the baseline is a length and every depth is proportional to it — and on one"
        " that drives straight for that long, which this cart's errands do not",
        off_when="shorter wherever the flow loses the corners before the window ends, or where"
        " the milliseconds in the report line matter more than the sigma; and back to 1.5 with"
        " parallax_tracking window, whose cost really is its window",
        range=(0.2, 10.0),
    ),
    Flag(
        "parallax_min_total_baseline_m",
        PARALLAX_MIN_TOTAL_BASELINE_M,
        description="the effective parallax a track's views must add up to before its depth is"
        " kept: the quadrature sum of each view's perpendicular baseline, sqrt(sum b^2). It is a"
        " GATE and no longer the number the sigma divides by (parallax_sigma_model decides"
        " that); it replaces parallax_min_baseline_m for a track, where no single view carries"
        " the whole baseline",
        why="the same 10 cm parallax_min_baseline_m asks of one partner, asked of the whole"
        " bundle instead, because it is the length the geometry rests on however the views are"
        " spread. A track over the default window reaches 14.0 cm of it on this cart at"
        " 0.2-0.3 m/s where a single 0.5 s pair reaches 5.2"
        " (scratch/parallax_tracks_eval.txt, the four errands of 2026-09-14), so the gate costs"
        " the default window little while still throwing out the corners on the epipole and the"
        " stretches where the cart barely moved. It is NOT the sigma: since 2026-09-15 the"
        " sigma comes from the solve's own covariance, because sqrt(sum b^2) is exact only for"
        " a camera moving across the ray and up to twice optimistic for one driving along it"
        " (scratch/parallax_sigma_mc.txt). It is also not parallax_min_baseline_m, which"
        " chooses a PARTNER and is unused while a corner is a track",
        on_when="raise it to keep only the corners the window really moved across, at the cost"
        " of the corners near the epipole and of a slow stretch of the errand",
        off_when="0 keeps every track the other gates let through, whatever its parallax: the"
        " sigma already says how little such a corner is worth",
        range=(0.0, 1.0),
    ),
    Flag(
        "parallax_sigma_model",
        PARALLAX_SIGMA_MODEL,
        description="what a track's sigma is. covariance propagates the midpoint solve's own"
        " normal matrix through each view's range, C = N^-1 (sum r^2 P) N^-1, and widens it only"
        " by the part of the reprojection RMS that pixel noise does not already explain;"
        " baseline is the closed form z^2 * sigma_px / (f * sqrt(sum b^2)) the stage shipped"
        " with. The sigma is a pair's whole vote in the frame's fit, which weighs 1 / sigma^2",
        why="the closed form is exact for a camera moving ACROSS the ray — a sidestep, the only"
        " geometry the tests and the first eval ever used — and optimistic wherever the views"
        " also spread ALONG it, because a view that sees the point from further away reads its"
        " pixel into a bigger depth error while the formula credits it with the same z. That is"
        " a cart driving forward, which is the errand. Monte Carlo against the solve's own"
        " scatter (scratch/parallax_sigma_mc.txt, 2026-09-15): honest for a sidestep, 1.28x"
        " optimistic at 10 views driving forward, 1.49x at 16 and 2.02x over the grid — a vote"
        " up to 4 times too loud. The covariance reads the same scatter to within 0-16 %, on the"
        " safe side. On the four errands of 2026-09-14 a track's reported sigma goes 5.7 cm to"
        " 9.1 against the pair's 9.5, so most of the sigma the tracks change first claimed was"
        " the formula and not the measurement",
        on_when="covariance always: it is right for every shape of bundle, and a 3x3 inverse per"
        " track is 3.5 ms a frame of the 27",
        off_when="baseline only to reproduce a number measured before 2026-09-15",
        choices=TRACK_SIGMA_MODELS,
    ),
    Flag(
        "parallax_track_max_views",
        PARALLAX_TRACK_MAX_VIEWS,
        description="how many frames one track may rest on. The window"
        " (parallax_track_window_s) says how far back to reach and this says how finely to"
        " sample it: more frames inside the same window are more observations and more of the"
        " flow's hops, which is where the time goes",
        why="the cap is the cost: 85 % of a track's milliseconds are the flow's hops and there"
        " is one hop per view. Over the four errands of 2026-09-14"
        " (scratch/parallax_tracks_audit.txt) 16 views cost 52.5 ms a frame and 8 cost 27.0"
        " (against a pair's 5.7), and on the 12 frames both could fit a law the law was no worse"
        " at 8 — 14.2 % of median |residual| at the lidar's beams against 19.0 % at 16, with the"
        " pair at 24.1-28.0 % on the same frames. The per-corner sigma is 8.4 cm at 8 views"
        " against 9.1 at 16 and the pair's 9.5",
        on_when="raise it on a robot whose camera is faster than this one's 6-9 frames/s, where"
        " a view is a smaller step and the hops are cheaper to hold",
        off_when="lower it wherever the report line's ms matter more than the corner count: 5"
        " views cost 18.1 ms and still read 15.5 % against the pair's 28.8 on the frames both"
        " fitted",
        range=(2, 16),
    ),
    Flag(
        "parallax_split_tol_sigma",
        PARALLAX_SPLIT_TOL_SIGMA,
        description="how far a track's older half and its newer half may disagree about its"
        " depth, in combined sigmas, before the track is dropped. Each half is triangulated on"
        " its own with the current frame as its anchor; a static point has one depth and every"
        " subset of its views must read it. 0 (the default) does not compute the halves at all;"
        " a huge value computes them and gates on nothing, which is how to measure. The report"
        " line counts the tracks it removes as split",
        why="the reprojection gate is read against the RMS over the observations, so one bad"
        " observation in n is divided by sqrt(n) before the gate sees it — at 8 views 47.7 % of"
        " the tracks kept already sit over 1.0 px of the 1.5 px budget"
        " (scratch/parallax_tracks_audit.txt). The split is the only test left that reads a"
        " depth changing with the window. 3 is what the real errands say: over 1490 tracks of"
        " 2026-09-14 the halves disagree by a median 0.54 sigma, p90 1.54, p99 3.01, and 1.1 %"
        " sit over 3 (scratch/parallax_tracks_eval.txt) — while a clean synthetic corner at the"
        " flow's 0.4 px never reached 2.0 sigma over 1200 draws at 3-8 views, driving and"
        " sidestepping (scratch/parallax_split_probe.py). So 3 takes the tail that pixel noise"
        " cannot explain and leaves the other 98.9 %. Taking it changes nothing, which is why"
        " the default is 0: with the gate armed the parallax-only law reads the same 15.0 % over"
        " all frames and the same 14.2 % and 19.0 % on the frames a pair also fitted at 8 and 16"
        " views, while the two extra half-solves cost 3.9 ms a frame of the stage's 27.0"
        " (scratch/parallax_tracks_audit.txt) — a measurable cost for no measurable benefit."
        " Be clear about how little it buys even armed: it is"
        " NOT the answer to the two holes scratch/parallax_gates_probe.txt found, and nothing"
        " on ONE track is. A point whose own motion is parallel to the camera's puts every ray"
        " through one place at the wrong depth — an object receding at 0.1 m/s from a cart"
        " driving at 0.25 reads 3.33 m for a 2.00 m truth with the halves 0.00 sigma apart, and"
        " 1.19 even with the cart turning at 0.5 rad/s — and a corner sliding along its epipolar"
        " line in proportion to the baseline is a pure scale error every subset shares (2.49 m"
        " for 2.00, the halves 0.18 sigma apart). Even a whole half of a window sliding 2 px off"
        " the corner, 21 % of depth, reads only 2.0 sigma. Those need the network's own depth or"
        " a second sensor",
        on_when="3 on a robot whose flow mistracks mid-window often enough to be worth 3.9 ms a"
        " frame; lower than 3 only with a measurement, since a clean corner already reaches 2.0"
        " sigma on pixel noise and 4.2 % of real tracks sit above 2",
        off_when="0 is the shipped default and costs nothing at all: the halves are not solved",
        range=(0.0, 20.0),
    ),
    Flag(
        "parallax_undistort",
        PARALLAX_UNDISTORT,
        description="the tracked corners are straightened with the lens camera_info publishes"
        " before the epipolar test, the triangulation and the reprojection measure with them."
        " The flow, the drift bound and the depth image itself keep the picture's own pixels —"
        " only the geometry is a pinhole. A no-op on an uncalibrated camera, on a picture"
        " camera_stream already rectified (its camera_info then carries no distortion), and on"
        " parallax_tracking window or pair, which measure in the picture's pixels as they"
        " always did",
        why="the geometry is a pinhole and the picture is not: this is an 83 degree lens with"
        " k1 -0.150, k2 -0.129, k3 +0.092 (config/camera.json, 45 views, rms 0.23 px), which is"
        " 8 px of displacement at the top edge of a 640x360 frame and over 20 in the corners,"
        " against an epipolar gate 1.5 px wide. Measured on the door tapes 0321/0322 of"
        " 2026-09-15 (scratch/parallax_rows_probe.txt, 77 judged frames) it is worth much less"
        " than that sounds, because both views of one corner are bent in nearly the same way"
        " and the error largely cancels: the epipolar residual's median moves 1.18 -> 1.04 px"
        " in the middle third of the picture and not at all in the top (1.10 -> 1.11). What it"
        " does buy is corners through every gate — 333 -> 411 kept in the top third, 2572 ->"
        " 2730 in the middle, 2812 -> 3019 at the bottom, 8 % overall and 23 % at the top —"
        " for one cv2.undistortPoints over a few hundred points a frame. Where that lands is"
        " the top of the picture, which is the part the lidar never sees: the weight the"
        " parallax corners carry into the frame's fit goes from 0.18 to 0.33 of a lidar beam"
        " per frame in the top third, while the middle and the bottom give back 3.61 -> 3.41"
        " and 4.01 -> 3.53 — the same corners with an honest sigma instead of a flattered one",
        on_when="always while the published picture carries a distortion: the pixels the gates"
        " measure ought to be the pixels the geometry assumes",
        off_when="to reproduce a number measured before 2026-09-16, or to A/B what the lens is"
        " worth on a tape",
    ),
    Flag(
        "parallax_correction_tol_m",
        PARALLAX_CORRECTION_TOL_M,
        description="how far the tracker's map -> odom correction may jump between two frames"
        " before the forward store's whole window is dropped — the corners live on, their"
        " observations do not. Read as the metres the jump puts on a point"
        f" {PARALLAX_CORRECTION_REACH_M:.0f} m ahead, so a turn of the map counts as well as a"
        " shift. Only read with parallax_motion tf; 0 never drops a window",
        why="the price of a map pose that answers on every frame. tf stores each view's pose as"
        " the tracker's estimate AS OF THAT FRAME, so a relocalisation landing inside the"
        " window moves every view before it relative to every view after it: the bundle then"
        " reads a displacement the camera never made, in a geometry where every depth is"
        " proportional to the baseline. 5 cm is a tenth of the shortest baseline the stage will"
        " keep (parallax_min_total_baseline_m 10 cm) and several times the 1-2 cm the tracker's"
        " pose is good to over a second (scratch/parallax_pose_sweep.txt) — over it the jump is"
        " a correction and not noise. What it costs when it fires is one window, which the"
        " store rebuilds in about a window's worth of frames; the report line counts the"
        " bundles it drops as correction",
        on_when="lower on a robot that relocalises smoothly and often, where a small correction"
        " is common and a large one is really a jump",
        off_when="0 to see what the corrections are worth, or on a tracker that never"
        " relocalises at all — the report line's correction count is how often it fires",
        range=(0.0, 1.0),
    ),
    Flag(
        "parallax_verify_every",
        PARALLAX_VERIFY_EVERY,
        description="frames between two rounds of the forward store's long-range drift bound:"
        " every corner re-tracked DIRECTLY from the picture its oldest kept view was taken in,"
        " started at where the hops say it is, and closed when the two disagree by more than"
        " parallax_drift_tol_px. A round is spread one kept picture per frame, so no frame pays"
        " for more than one extra flow call. 0 turns the bound off",
        why="a per-hop forward-backward check cannot see the drift that matters. Lucas-Kanade"
        " slides along an edge and along the epipolar line by a fraction of a pixel a hop, each"
        " hop passing its own check, and at this camera's 14 frames a second a 3 s window is"
        " forty hops — which is a corner somewhere else at a depth that is wrong and"
        " consistent. Nothing about a forward track catches that, because there is no fresh"
        " re-track in it: the backward window got one for free every frame. Synthetically a"
        " flow nudged 2 px a hop is caught 20-odd times over 30 frames while an honest one"
        " loses at most 3 corners of 200 (tests/unit/test_parallax.py). On the four errands of"
        " 2026-09-14 the bound costs 0.4-0.6 ms a frame and closes about 3800 corners per 1200"
        " frames, a third of all the corners the store loses"
        " (scratch/parallax_forward_eval.txt) — so it is also the biggest single reason the"
        " forward ruler keeps fewer corners a frame than the backward window. What it buys on"
        " real pictures is not separated from what it costs: measured on one errand it left"
        " more corners standing after the epipolar gate than turning it off did"
        " (scratch/_forward_probe.py), which is the opposite sign to the corner count",
        on_when="every 10 frames is the default, which at this camera is about a fifth of a 3 s"
        " window; lower it on a longer window, where a corner has more hops to slide over",
        off_when="0 wherever the corner count matters more than the corner's truthfulness, or"
        " to A/B what the bound is really worth — the report line counts what it closes as"
        " drift",
        range=(0, 240),
    ),
    Flag(
        "parallax_drift_tol_px",
        PARALLAX_DRIFT_TOL_PX,
        description="how far a corner's hopped position may sit from where its own birth patch"
        " lands when it is re-tracked directly into this frame, before the corner is closed."
        " Only read when parallax_verify_every is above 0",
        why="a pixel is twice what the flow is trusted to place a corner to"
        " (DISPARITY_SIGMA_PX 0.5), so a corner over it has moved by more than its own noise"
        " and the two pictures no longer agree about what it is. It is not a free gate: on the"
        " four errands of 2026-09-14 the bound at 1 px closes about 3800 corners per 1200"
        " frames and roughly halves the corners a frame against turning it off, while 2 px sits"
        " between the two (scratch/_forward_probe.py, scratch/parallax_forward_eval.txt). A"
        " direct re-track over a whole window can also disagree for reasons that are not drift"
        " — the patch has turned and been lit differently — which is why the number is a"
        " tolerance and not a half-pixel",
        on_when="tighter on a robot whose window is long and whose flow is the suspect: a"
        " sliding corner is a depth that is wrong and consistent, and no other gate sees it",
        off_when="looser (2 px) wherever the corner count is the binding constraint, which on"
        " these errands it is",
        range=(0.1, 20.0),
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
        "tf_dead_s",
        TF_DEAD_S,
        range=(0.0, 600.0),
        description="how far behind a frame's stamp TF's newest edge may be before that edge is"
        " taken for dead and no lookup on the frame's path waits for it: the camera pose falls"
        " to config/camera.json's mount and the lidar's scan passes uncarried, both at once and"
        " both counted. 0 turns the guard off — every lookup waits CARRY_WAIT_S again",
        why="2026-09-16: the board's TF route died, base_link <- camera_optical stopped 344 s"
        " back, and every frame still spent the whole 0.2 s wait on a lookup no publisher was"
        " going to answer — pose 212/226 ms in the report line, the stream down to 0.9-3"
        " frames/s. Three seconds is three missed republishes of the neck at 10 Hz and well over"
        " any WiFi hiccup, so a route that is merely stuttering still gets its wait",
        on_when="always: a wait that cannot succeed costs the frame and buys nothing",
        off_when="0 to reproduce the old behaviour, or raise it on a link whose TF genuinely"
        " arrives in bursts longer than three seconds",
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
        "field_pairs_cap",
        PIPELINE_DEFAULTS["field_pairs_cap"],
        range=(0, 200_000),
        description="pairs per RULER the per-frame law's field is fitted on: a block longer than"
        " this is thinned to that many, evenly spaced, its total weight preserved so the thinning"
        " cannot change which ruler writes the law. 0 fits every pair, as the stage did",
        why="the fit costs the pool's total, and the rulers are not the same size: the lidar"
        " brings tens of beams, the floor and the wall together up to 80 000 pairs of one frame,"
        " and the field's fit ran 35 ms on them. At 2000 a block the same frames fit in 4.7 ms"
        " and the nodes move 0.00 % of their value at the median, 0.22 % at the worst"
        " (2026-09-16, scratch/chain_profile.txt) — a fit reads weight, and 2000 evenly spaced"
        " pairs of a block carry the same weight in the same places as its 40 000",
        on_when="lower it on a slower board, where the frame law is the stage in the way; the"
        " report line's pair count is what was fitted",
        off_when="0 to fit every pair — the A/B of the cap itself, and the thing to set if a"
        " node's law is ever suspected of following the thinning rather than the scene",
    ),
    Flag(
        "field_prior",
        PIPELINE_DEFAULTS["field_prior"],
        range=(0.0, 1000.0),
        description="how hard each node of the field is pulled toward the frame's own GLOBAL fit,"
        " in PAIRS (a lidar beam is 1): a prior carrying as much information about that node's"
        " law as that many pairs of weight 1 would at that node. A node that saw no pair comes"
        " back as the global fit, so the field degrades to the single law wherever the anchors"
        " are sparse; a node that saw many follows its own",
        why="0.3 because the number now means pairs and the lidar's own row wants it light."
        " Swept 0.03 / 0.1 / 0.3 / 1 / 3 against a carry of 0.1 / 1 / 3 (2026-09-15) on the row,"
        " held out on the CONTIGUOUS split — half a frame's beams fit, the other half judges,"
        " then the reverse (scratch/field_prior_row_sweep.py): the mean median |residual| over"
        " the drive and the three neck pitches reads 11.2-11.9 % at 0.03, 11.8-12.3 % at 0.3,"
        " 13.1-13.4 % at 1 and 13.7-14.1 % at 3, against the single law's 15.8 %. Above the row,"
        " where no beam judges (scratch/wall_truth_eval.py, tapes 0313 and 0268), the same"
        " hundred-fold of prior moves the lidar chain by under a point (17.1 / 15.6 % at 0.3"
        " against 16.7 / 15.0 at 3), because up there the field has almost nothing of its own to"
        " fit — the parallax anchor lands 0.000-0.004 of pair weight a frame in the top row of"
        " nodes. So the row decides",
        on_when="raise it toward 3 on a cart whose anchors are thin and scattered, where a node"
        " fitted on two beams is a whole quadrant of the picture fitted on two beams — and when"
        " the rows ABOVE the lidar's matter more than the row itself",
        off_when="0 lets every node follow its own pairs alone; lower it while reading the node"
        " table in the report line, never blind — an unpulled node of two weak pairs is what"
        " puts a law on its bound",
    ),
    Flag(
        "field_carry",
        PIPELINE_DEFAULTS["field_carry"],
        range=(0.0, 1000.0),
        description="how hard each node is pulled toward what it was on the LAST frame, in the"
        " same pairs, decaying as exp(-dt / field_carry_tau_s)",
        why="3.0, the one knob of the field that measured better everywhere it was looked at"
        " (2026-09-15, the same sweep as field_prior). At the lidar's own row, held out on the"
        " contiguous split, it takes the drive's TOP third of the picture from 28.7 to 23.5 % at"
        " prior 0.3 and leaves 1 node fit of 846 pinned on a bound against 8 at a carry of 1;"
        " above the row it is worth 0.2-0.5 points on both wall tapes and both wall-pixel"
        " selections. What it is there for is the frames with no beams at all — with floor pairs"
        " as the only ruler it is what holds the scale of the nodes that saw nothing this time",
        on_when="raise it further on a run whose anchors flicker (a lidar in and out of the"
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
        "wall_sigma_height",
        PIPELINE_DEFAULTS["wall_sigma_height"],
        range=(0.0, 2.0),
        description="metres of doubt a wall pair carries per metre of HEIGHT above the lidar's"
        " line — the price of the world assumption. A pair's sigma is sqrt(sigma_lidar^2 +"
        " (wall_sigma_height * h)^2), so the ruler fades as it leaves the beams that vouch for"
        " it instead of switching off at a threshold: at 0.05 a pixel a metre up is trusted to"
        " 5 cm, about a fortieth of a beam's weight at 2 m",
        why="'the surface goes on upwards' is true of a door and a wall and false of a sofa, a"
        " shelf, a table and a chair, and NOTHING in the picture settles it: a surface leaning"
        " back 0.3 m per metre of height departs from the plane by 0.08 % a row while this"
        " network's own scale climbs about 0.4 % a row (1.6x at the lidar's row, 2.0x by 0.3 m"
        " above it, scratch/pipeline_vs_truth.txt), so a gate tight enough to refuse the lean"
        " refuses every real wall — which is why there is a growing error bar here and not a"
        " sharper gate (a unit test holds that boundary: the step of a shelf yields zero pairs,"
        " the 0.3 m/m lean yields the same pairs as a flat wall). 0.05 is a CHOSEN error bar,"
        " not a measured one: on the COLMAP scene only 3-6 % of the points the camera sees stand"
        " on the lidar's extrusion at all, and the gated walk covers too few of them (29) to"
        " measure its own error against (scratch/wall_vs_colmap.txt). What is measured is the"
        " end-to-end effect at this value",
        on_when="raise it toward 0.3 in a room of low furniture, sofas and shelves, where the"
        " extrusion is most often a lie: the pairs then fade within half a metre of the beams",
        off_when="lower it toward 0.01 in a corridor of flat walls and doors, where the"
        " assumption holds to the top of the picture",
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
        " gate: the A/B of what the gate is refusing. It is read at all only with"
        " floor_plane_band off",
    ),
    Flag(
        "floor_band_max_m",
        PIPELINE_DEFAULTS["floor_band_max_m"],
        range=(0.0, 2.0),
        description="the widest, in metres, the floor's height band may ever grow — the band that"
        " decides whether a pixel is on the floor at all. 0 leaves it uncapped, which is the"
        " behaviour before this knob",
        why="the band is 2 * h * (0.03 + 0.01 E), the network's relative error turned into"
        " height, so it grows with the floor's own depth and never stops: 12 cm at 2 m, 20 cm at"
        " 5 m, 2.2 m at 90 m. The rows within half a degree of the horizon all sit at such"
        " depths, and there the band admits everything between the floor and the ceiling — which"
        " in a room is the WALL standing at that bearing. On tape 0318 (a closed door 2 m ahead)"
        " 7 % of the candidates were the door at head height: 79 pixels whose floor depth reads"
        " 90 m, standing 1.18 m over the floor. They turned the fitted plane from 12 degrees of"
        " lean into 50, and once a frame got through the gate they took the floor-only law with"
        " them — a hundredth of the truth on the next frame (scratch/floor_gate_probe.txt)."
        " 20 cm is where the band stops separating the floor from what stands on it (the cart's"
        " own scan calls something an obstacle from 15 cm up)",
        on_when="raise it toward 30 cm on a floor the network reads badly, and watch what the"
        " gate then lets in",
        off_when="0 is the A/B: the band grows without limit, as it did before",
    ),
    Flag(
        "floor_plane_band",
        PIPELINE_DEFAULTS["floor_plane_band"],
        description="judge the plane fitted to a frame's floor pixels in METRES — it must stay"
        " inside the very height band each pixel was selected by — instead of in fixed degrees"
        " off the cart's up vector (floor_normal_tol_deg). The plane is also fitted differently:"
        " the height regressed on the ground position, not total least squares",
        why="a fixed angle asks for something the geometry does not always carry. A door 2 m"
        " ahead leaves a floor strip 0.85 m deep in view and the pixels are chosen inside a band"
        " 12 cm wide, so the selection itself admits any lean up to 16 deg; a 5-degree gate is"
        " then a test of the law's row bias, not of the floor, and it refused EVERY frame of all"
        " four door tapes, 90 of 95 of tape 0313 and 8 of 12 of run 0171. The band test tightens"
        " by itself wherever more floor is in view (scratch/floor_gate_eval.txt)",
        on_when="on is the measured default",
        off_when="off restores the degree gate, which is the A/B of what moved",
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
        description="whose word the parallax anchor's baseline is. tf builds the cart's map"
        " pose out of TF's two halves on every frame — the newest map -> odom (the tracker's"
        " correction, published slowly and changing slowly) composed with odom -> base_link at"
        " this frame's own stamp (the EKF at 20 Hz) — and takes the motion between any two"
        " frames from the two poses, which is arithmetic. tracker asks for the tracker's own"
        " map pose at BOTH stamps (TF map -> base_link), odom for the EKF's wheels and gyro"
        " alone. A window the source cannot answer falls back to the odometry, and the report"
        " line counts how many windows each source actually gave",
        why="tf because a bundle may not span two motion sources and the tracker's own map pose"
        " is not there on every frame. Live on 2026-09-15, a 30 s drive with the forward store"
        " at a 5 s window: 'map pose stale -> odom' fired 821 times, every fallback cut the"
        " window (16718 cut bundles), and the window never accumulated past a span of 2.10 s"
        " with 4.0 observations a track and 17.3 cm of baseline. Nothing about the tracker was"
        " wrong — its pose is published behind the frames and covers their stamps only"
        " sometimes. Splitting the question in two asks the slow half for its newest value and"
        " the fast half for this exact moment, so the answer exists on every frame and there is"
        " nothing to cut. The price is that a view's pose is the tracker's estimate AS OF THAT"
        " FRAME: a correction landing inside the window moves the older views by the correction"
        " and invents a displacement the camera never made, which is what"
        " parallax_correction_tol_m watches for. tracker and odom keep the cut, and for the"
        " backward window and the pair — which ask for a motion between two stamps rather than"
        " for a pose — tf asks exactly what tracker asks. On the baseline itself the tracker's"
        " pose remains the measured one over the wheels, and"
        " a baseline is a length and every triangulated depth is proportional to it: over"
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
        on_when="tf wherever a window longer than a frame or two matters: it is the only one"
        " that answers on every frame, and the tracker's correction is still inside it",
        off_when="tracker to reproduce a number measured before 2026-09-15, or where the"
        " tracker relocalises so often that every window is cut anyway; odom on a robot with no"
        " tracker at all, or to A/B the baseline against the numbers above without a restart",
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
    Flag(
        "scan_honours_pan",
        True,
        description="fold /depth_scan onto the floor through the neck's pan: the fan's bearings"
        " turn with the head and its angular window turns with them, so angle_min comes out at"
        " pan - 40 deg instead of -40. The pan is the yaw of the same base_link <-"
        " camera_optical edge the volume path reads (camera_tf_latest); with no such edge the"
        " config mount's straight-ahead yaw stands in, and the report line's config counter says"
        " for how many frames. Off: the fan is projected as if the head looked along the cart's"
        " x, whatever the encoders say",
        why="the fan carried no pan at all until 2026-09-15, and the report line said so ('head"
        " panned N frames (projected as if not)'). At rest that is not nothing: the pan"
        " reference measured that day (config/neck.json pan_note,"
        " pepin.extrinsics.pan_from_bearings, six windows in four scenes) puts the resting head"
        " +0.79 deg left of the cart's x, which is 4 cm of bearing error at 3 m — under"
        " PAN_NOTICE_RAD, so the old fan did not even count it. A head panned on purpose puts"
        " the whole fan in the wrong place: 20 deg of neck is 20 deg of costmap, one metre"
        " sideways at 3 m",
        on_when="always once the neck's edge is in TF — a scan whose bearings are the cart's is"
        " what Nav2's obstacle layer assumes it is being handed",
        off_when="to reproduce a costmap from before 2026-09-15, or to read a fan against a"
        " measurement taken while the projection ignored the pan (the yaw-offset probes of"
        " config/neck.json's pan_note were)",
    ),
    Flag(
        "depth_reach",
        True,
        description="the PUBLISHED depth image is NaN past depth_reach_m: the camera answers for"
        " its own data and says nothing where it does not vouch for the range. /depth_scan is"
        " unaffected (it is capped at the same range already) and so is every law — the gate is"
        " applied to the image on its way out, after the pipeline",
        why="a NaN depth pixel makes no point in any consumer: rtabmap drops it from the cloud"
        " before the grid (pcl::isFinite, rtabmap/core/util3d.cpp:644), Nav2's obstacle layer"
        " neither marks nor raytraces it (verified against obstacle_layer.cpp 1.3.12 on"
        " 2026-09-11), and pepin.tsdf integrates only finite depths. Without the gate the camera's"
        " far half is a fiction that outvotes the lidar: 44 % of the camera's costmap marks within"
        " 2.5 m were BEHIND the wall the lidar sees (run 0224, camera layer alone, 2026-09-11),"
        " and the wall-truth eval put the network 1.24-1.27 of the truth in the middle and top"
        " thirds of the picture against 0.998 at the beams (errand 0313,"
        " scratch/wall_truth_eval.py). It is what lets ONE Grid/RangeMax serve both sensors in"
        " vslam.launch.py: the lidar's 8 m, with the camera's reach carried in the camera's data",
        on_when="always while any grid, costmap or volume is built from BOTH this depth and the"
        " lidar — which is every mode since 2026-09-19",
        off_when="to measure the network past its reach (a range-law session that wants the far"
        " bins), and to reproduce a volume or a costmap from before this gate",
    ),
    Flag(
        "depth_reach_m",
        DEPTH_REACH_M,
        description="metres past which the published depth is NaN; the same number /depth_scan is"
        " capped at",
        why=f"{DEPTH_REACH_M:.1f} is the reach this stack already stands on in two places: the"
        " scan's own cap (scan_max_range, which the costmap's obstacle_max_range of 2.5 m must"
        " stay under — 2026-09-11 05:25, or an inf ray marks a lethal ring) and the camera-only"
        " grid's Grid/RangeMax. What the range law measured across it: after the law 0.8-1.2 m"
        " reads +0.1 %, 1.2-1.6 +0.4 %, 1.6-2.0 -0.5 %, and 2.0-2.5 m stays -20 % under any law"
        " z = f(d) at one neck pitch because the network SATURATES there (true 1.75 and 2.2 m"
        " arrive at the same network depth ~3.1, 2026-09-15 15:30) — softened the next day to a"
        " place fact, with the frame law reading +2.7 % over 2.5-6 m on a drive. So the honest"
        " statement is that the last metre before 3 m is worth a fifth of itself at worst and"
        " nothing is claimed past it",
        on_when="raise it only with a wall-truth measurement at the new range on the current"
        " geometry, and raise Grid/RangeMax's camera half nowhere — it is the lidar's",
        off_when="lower it where the network is known to be worse: a dark room, a patterned floor,"
        " a head pitched far down (the saturation moves with the pitch)",
        range=(0.3, 12.0),
    ),
    Flag(
        "scan_hz",
        5.0,
        description="the cap on how often /depth_scan is PUBLISHED, in hertz; 0 publishes one fan"
        " per frame, which is what this topic did until 2026-09-22. The cap is on the publisher"
        " alone: every frame still goes through the network and the whole pipeline, every law is"
        " still fitted from it, and the depth image on /camera/depth is not thinned at all",
        why="the consumers of this topic are the board's two costmaps, which read it at their own"
        " update_frequency — 5.0 local, 2.0 global (ros/params/nav2_params.yaml) — and"
        " pepin_bringup.depth_fusion, which uses it to CLEAR. This node publishes at the"
        " camera's rate, ~9 Hz, so roughly four of every nine fans crossed the zenoh routers to"
        " the board to be overwritten in the layer before it was next read. The stop reflex is"
        " bounded by the costmap tick and not by this publisher, so nothing about how fast the"
        " cart stops changes",
        on_when="raise it with the local costmap's own update_frequency, never above the"
        " camera's frame rate (a cap above the source publishes every frame and nothing more)",
        off_when="0 is the pre-2026-09-22 behaviour, one fan per frame: what a bench test on one"
        " machine (no routers in the path) may as well use, and the A/B for whether a mark the"
        " costmap failed to clear is the cap's fault",
        range=(0.0, 30.0),
    ),
)
FLOOR_STAGES = ("floor_anchor", "floor_pairs")  # the stages that read the IMU's up vector


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
        self._dist: tuple[float, ...] | None = None  # camera_info's own, once it has arrived
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
        self._range = RangeLawStage(self._law)
        self._last_verdict_wall = time.time()  # the law's age is the beams', not the node's
        self._seed_laws(time.time())
        self._pipeline = standard_pipeline(self._law, range_stage=self._range)
        # The stereo source's start-up parameters, declared HERE: once the flag kit below is
        # built, its parameter callback refuses every name that is not a flag ("not a flag of
        # this node"), so a parameter declared after it kills the node at start — it did, on the
        # robot's first stereo start (2026-09-21), and the ROS stubs of the unit tests cannot
        # show it.
        self._pair_wait_s = float(self.declare_parameter("stereo_pair_wait_s", PAIR_WAIT_S).value)
        self._stereo_reach_m = float(self.declare_parameter("stereo_reach_m", 0.0).value)
        self._stereo_matcher_settings = self._matcher_settings()
        self._switches = Switches(self, flags_for(source_name), on_change=self._on_switch)
        self._law.watching = bool(self._switches["law_watch"])
        for name in self._pipeline.names:  # a launch override reaches the stage it names
            self._pipeline.set(name, self._switches.on(name))
        set_scale_ceiling(float(self._switches["scale_ceiling"]))  # and the law's bound
        self._law.slew_per_s = float(self._switches["law_slew"])  # how fast the law may move
        self._ask_parallax(float(self._switches["parallax_min_baseline_m"]))  # and the ring's ask
        self._ask_matcher(str(self._switches["parallax_matcher"]))  # and who matches its corners
        self._ask_motion(str(self._switches["parallax_motion"]))  # and whose motion it triangulates
        self._ask_map_wait(bool(self._switches["parallax_map_wait"]))  # asked without waiting
        self._ask_parallax_weight(float(self._switches["parallax_weight"]))  # and its vote
        self._ask_track()  # and whether a corner is a track through the window or a pair
        self._ask_lidar_sigma(float(self._switches["lidar_sigma_m"]))  # and what a beam is worth
        self._ask_frame_shift(bool(self._switches["frame_shift_needs_beams"]))  # and the gate
        self._ask_field()  # and the shape of the per-frame law: its grid and its two pulls
        self._ask_floor()  # and what a floor pair is worth and when a floor is not a floor
        self._ask_wall()  # and what the world assumption above the beams is worth
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
        # The right eye, and the two numbers that turn a disparity into metres. Only under
        # ``depth_source: stereo``: a mono head publishes neither topic, and a subscription to a
        # topic nobody writes costs the transport a discovery entry for nothing.
        self._right_tx: float | None = None  # the right eye's P[0,3] = -fx * baseline
        self._right: deque[tuple[tuple[int, int], Image]] = deque(maxlen=PAIR_BUFFER)
        self._right_ready = threading.Condition()
        self._stereo: StereoDepth | None = None
        if self._stereo_on:
            self._stereo = StereoDepth(
                matcher=StereoMatcher(self._stereo_matcher_settings),
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
        # What the PIPELINE asks about the cart's motion (the parallax anchor, once per stored
        # view): the same poser over a TF that waits for nothing at all. A wait is paid per
        # view here, not per frame — two views cost 834 ms of one frame offline
        # (scratch/chain_profile.txt) — and a view TF cannot answer yet is simply left out of
        # this frame's bundle, which the stage counts as its own.
        self._frame_history = self._guarded(0.0)
        self._frame_poser = self._new_poser(self._frame_history)
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
                f"raw depth from the stereo head ({self._stereo.matcher.settings.describe()}):"
                f" {RIGHT_IMAGE} paired with /camera/image by exact stamp,"
                f" {self._pair_wait_s * 1e3:.0f} ms of grace; fx and the baseline"
                f" come from {RIGHT_INFO}"
            )
        self.get_logger().info(
            "depth stream up: /camera/image -> /camera/depth, /depth_scan; camera pose from TF"
            f" (config/camera.json's pitch {math.degrees(pitch):.1f} deg while TF has no edge)"
        )

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
        """Hand every law what the last run saved in the law file, and say so in the log: the
        affine numbers to the affine law, and the range law's bins to the range law when the
        file holds them and they still stand (:meth:`pepin.depth.RangeLaw.restore` judges
        them). A record of a law that no longer exists (:func:`pepin.depth.retired_laws`) is
        named in the log line and left alone — the next save drops it. Without a file nothing
        is published until POOL_MIN_SAMPLES beam pairs are pooled."""
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
        retired = retired_laws(self._law_file)
        retired_note = (
            f"; ignoring the retired {', '.join(retired)} law record the file still carries"
            if retired
            else ""
        )
        ranged = load_range(self._law_file, now)
        if ranged is not None:
            self._range.seed(ranged)
        range_note = (
            f"; range law {ranged.describe()}" if ranged is not None else "; no range law saved"
        )
        self.get_logger().info(
            f"depth law from {self._law_file}: a {saved[0]:.2f} b {saved[1]:+.3f}"
            f" on {saved[2]} beams; publishing at once{range_note}{retired_note}"
        )

    def _on_switch(self, name: str, _old: Any, new: Any) -> None:
        """A flag changed: ``depth_backend`` is the switch's mode, ``scale_ceiling`` the law's
        upper bound, ``imu_lean`` the poser's and the estimator's, ``lean_min_quality`` the
        poser's floor under a lean, a stage's flag switches that stage of the pipeline."""
        if name == "depth_backend":
            self._net.mode = str(new)
        elif name == "scale_ceiling":
            set_scale_ceiling(float(new))  # the next fit is bounded by it; the law in hand is not
        elif name == "law_watch":
            self._law.watching = bool(new)  # from the next frame on
        elif name == "law_slew":
            self._law.slew_per_s = float(new)  # from the next fit on
        elif name == "imu_lean":  # both posers over the same TF lean the same way
            self._poser.apply_lean = self._frame_poser.apply_lean = bool(new)
            self._lean.use_gyro = bool(new)
        elif name == "lean_min_quality":
            self._poser.min_lean_quality = self._frame_poser.min_lean_quality = float(new)
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
        elif name in TRACK_FLAGS:  # the table already holds the new value (node_kit.Switches)
            self._ask_track()
        elif name == "lidar_sigma_m":
            self._ask_lidar_sigma(float(new))
        elif name == "frame_shift_needs_beams":
            self._ask_frame_shift(bool(new))
        elif name.startswith("field_"):
            self._ask_field()
        elif name.startswith("floor_") and name not in self._pipeline.switches:
            self._ask_floor()
        elif name == "wall_sigma_height":
            self._ask_wall()
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

    def _ask_track(self) -> None:
        """Hand the parallax anchor the whole shape of a measurement, read from the switch
        table: which ruler follows the corners (forward, the old backward window, or one
        partner), how far back it reaches and over how many views, how much parallax those
        views must add up to, what the sigma is taken from, how far the two halves of a track
        may disagree — and the forward store's own corners, detection cadence and drift bound.
        They move together, so they are set together, and every one of them is pushed into the
        store on the next frame with nothing reset: a window changed live simply uses more or
        fewer of the observations already held. At ``parallax_track_min_obs`` 2 the rest are
        unused and the anchor pairs exactly as it did before 2026-09-15."""
        stage = self._pipeline.stage("parallax_anchor")
        if isinstance(stage, ParallaxAnchor):
            stage.tracking_mode = str(self._switches["parallax_tracking"])
            stage.track_min_obs = int(self._switches["parallax_track_min_obs"])
            stage.track_window_s = float(self._switches["parallax_track_window_s"])
            stage.min_total_baseline_m = float(self._switches["parallax_min_total_baseline_m"])
            stage.track_max_views = int(self._switches["parallax_track_max_views"])
            stage.sigma_model = str(self._switches["parallax_sigma_model"])
            stage.split_tol_sigma = float(self._switches["parallax_split_tol_sigma"])
            stage.max_tracks = int(self._switches["parallax_max_tracks"])
            stage.redetect_every = int(self._switches["parallax_redetect_every"])
            stage.verify_every = int(self._switches["parallax_verify_every"])
            stage.drift_tol_px = float(self._switches["parallax_drift_tol_px"])
            stage.correction_tol_m = float(self._switches["parallax_correction_tol_m"])
            stage.undistort = bool(self._switches["parallax_undistort"])

    def _ask_lidar_sigma(self, sigma_m: float) -> None:
        """Tell the lidar anchor what one beam's range is trusted to, in metres: its pairs then
        weigh 1 / sigma^2 in inverse depth. 0 restores the flat weight of 1 a beam used to have."""
        stage = self._pipeline.stage("lidar_anchor")
        if isinstance(stage, LidarAnchor):
            stage.sigma_m = sigma_m

    def _ask_field(self) -> None:
        """Tell the per-frame law what shape it is: how many nodes it carries over the picture,
        how hard each one is pulled toward the frame's global fit and toward its own last value,
        and how many pairs of one ruler it may fit on. A new grid starts every node again from
        the next frame's fit."""
        stage = self._pipeline.stage("frame_law")
        if not isinstance(stage, FrameLaw):
            return
        field = stage.field
        field.prior = float(self._switches["field_prior"])
        field.carry = float(self._switches["field_carry"])
        field.carry_tau_s = float(self._switches["field_carry_tau_s"])
        stage.pairs_cap = int(self._switches["field_pairs_cap"])
        grid = grid_of(str(self._switches["field_grid"]))
        if field.grid != grid:
            field.grid = grid

    def _ask_floor(self) -> None:
        """Tell the floor anchor what the mount's pitch is trusted to (a floor pair's own sigma),
        how wide its height band may grow, which plane gate judges its pixels and — for the
        degree gate — how far that plane may lean before the frame's floor is refused."""
        stage = self._pipeline.stage("floor_pairs")
        if isinstance(stage, FloorPairs):
            stage.sigma_pitch_deg = float(self._switches["floor_sigma_pitch_deg"])
            stage.normal_tol_deg = float(self._switches["floor_normal_tol_deg"])
            stage.band_max_m = float(self._switches["floor_band_max_m"])
            stage.plane_band = bool(self._switches["floor_plane_band"])

    def _ask_wall(self) -> None:
        """Tell both wall stages what a metre of height above the lidar's line costs a pair in
        certainty — the price of "the surface goes on upwards" (:func:`pepin.depth_pipeline.
        wall_sigma`)."""
        for name in ("wall_anchor", "wall_correct"):
            stage = self._pipeline.stage(name)
            if isinstance(stage, WallAnchor):
                stage.sigma_height = float(self._switches["wall_sigma_height"])

    def _ask_frame_shift(self, needs_beams: bool) -> None:
        """Tell the per-frame law whether a shift needs the lidar in the pool that fits it; a
        parallax-only pool then gets a scale alone."""
        stage = self._pipeline.stage("frame_law")
        if isinstance(stage, FrameLaw):
            stage.shift_needs_beams = needs_beams

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
        """A frame poser over ``history``, wearing the node's lean switches: the node keeps two
        of them over the same TF — one that may wait for a frame's stamp (the camera's pose,
        the scan's carry) and one that may not (what the pipeline asks per view)."""
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
        """The camera's optics as they are published — and the distortion the published picture
        still carries, which is empty once camera_stream's ``undistort`` rectifies it. The
        pipeline's own projections stay a pinhole; the parallax anchor is the one stage that
        straightens the pixels it measures with (``parallax_undistort``)."""
        self._intr = Intrinsics.from_camera_info(msg.k, msg.width, msg.height)
        self._dist = tuple(float(v) for v in msg.d)
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
                        return None if right is None or right.ndim != 2 else right
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
        if right.shape != rgb.shape[:2]:
            self._tally.count("unpaired")
            self.get_logger().warning(
                f"the right eye is {right.shape} and the left {rgb.shape[:2]}: not one rig",
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
            rgb = views.rgb
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
        # The parallax anchor is the one stage that reads the picture itself; the grey copy is
        # made only while it is on, the poser it triangulates against is the same TF the carry
        # asks but with nothing to wait with (a lookup here is paid once per stored view), and
        # the camera edge goes in whole so its baseline carries the neck's pan.
        ctx = FrameContext(
            self._intr_or_nominal(msg),
            cam,
            up=self._lean.up,
            lidar=lidar,
            stamp=stamp_seconds(msg.header.stamp),
            gray=to_gray(rgb) if self._pipeline.on("parallax_anchor") else None,
            motion=self._frame_poser,
            cam_optical=cam_optical,
            dist=self._lens_dist(msg),
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
        copy so nothing the pipeline still holds is touched. Off (``depth_reach``), the array
        itself, which is what every consumer read before 2026-09-19.

        A sensor answers for its own data: NaN is the one value that makes no point in any consumer
        downstream — no cell in RTAB-Map's grid, no mark and no raytrace in Nav2's obstacle layer,
        no voxel in the volume — so the range this network's scale stops being a measurement over
        is stated once, here, instead of in each consumer's own cap."""
        if not self._switches.on("depth_reach"):
            return depth
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
        neck's live edge, or the static one) twice over — as the pipeline's pitch-only pose,
        and as the edge itself for the stage that needs the whole rotation (the parallax
        anchor's baseline turns with the neck's pan). Without such an edge in TF:
        config/camera.json's mount, counted, and no edge. A head turned past PAN_NOTICE_RAD is
        counted too: the volume path and the fan (``scan_honours_pan``) turn with it, while the
        pitch-only pose the rest of the pipeline reads still assumes the cart's x.

        Three asks, cheapest first, and only the last of them can wait: the frame's own stamp
        from what TF already holds, the newest edge while it is younger than
        CAMERA_TF_MAX_AGE_S (``camera_tf_latest``), and then the blocking lookup — which
        :class:`LiveEdgeHistory` refuses outright once the neck's edge is more than
        ``tf_dead_s`` behind this frame, so a dead route costs the config mount and not 0.2 s
        of every frame."""
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

    def _fan_pan(self, ctx: FrameContext) -> float:
        """How far left the head looks while the fan is folded, radians CCW from the cart's x:
        the yaw of the very ``base_link <- camera_optical`` edge the volume path took for this
        frame (:meth:`_camera_at`, flag ``camera_tf_latest``). With no such edge the frame's
        pose is config/camera.json's mount, whose yaw is zero — straight ahead — and the report
        line's ``camera pose from config`` counter is the count of those frames. Returns 0.0
        while ``scan_honours_pan`` is off: the fan of before 2026-09-15, folded as if the head
        looked along the cart's x."""
        if not self._switches.on("scan_honours_pan") or ctx.cam_optical is None:
            return 0.0
        return optical_heading(ctx.cam_optical.rotation)[1]

    def _as_scan(self, depth: Array, image: Image, ctx: FrameContext) -> LaserScan:
        """The depth folded onto the floor plane, in base_link, stamped like the image — with
        the floor gated out of it the way ``fan_floor_gate`` says (:data:`FAN_FLOOR_GATES`) and
        its bearings turned by the neck's pan the way ``scan_honours_pan`` says."""
        gate = str(self._switches["fan_floor_gate"])
        floor_of: float | Array = SCAN_MIN_Z_M
        if gate == "band":
            floor_of = fan_min_z(self._floor_expected(ctx), ctx.cam.z)
        angle_min, step, ranges = depth_to_scan(
            depth,
            ctx.intr,
            ctx.cam,
            pan=self._fan_pan(ctx),
            min_z=floor_of,
            max_range=self._scan_max_range,
        )
        before = int(np.count_nonzero(np.isfinite(ranges)))
        if gate == "contact":
            # Both fans are indexed by the bearing across the PICTURE, so the contact scan's
            # bins line up with the panned fan's bin for bin without a pan of its own.
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

    def _lens_dist(self, image: Image) -> tuple[float, ...]:
        """The distortion the published picture carries: camera_info's own once it has arrived,
        config/camera.json's calibration scaled to this frame's size until then. Empty when the
        camera is uncalibrated or the picture is already rectified."""
        if self._dist is not None:
            return self._dist
        return tuple(optics(self._camera_cfg, image.width, image.height).dist)

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
            f" {self._source.report()}, flags: {self._switches.state()}, ms median/max:"
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
        for edge in ("odom", "map"):  # the edges the carry and the parallax ask about
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
        if c["camera_from_config"]:
            neck = self._dead_edge(w, "neck") or "no TF edge"
            extra += f", camera pose from config {c['camera_from_config']} frames ({neck}"
            extra += "; fan pan from the mount)" if self._switches.on("scan_honours_pan") else ")"
        if c["fan_gated"]:
            extra += f", floor-gated {c['fan_gated']} bearings ({self._switches['fan_floor_gate']})"
        if self._switches.on("depth_reach") and c["frames"]:
            share = c["beyond_reach"] / max(c["frames"] * self._pixels(), 1) * 100.0
            extra += (
                f", published NaN past {float(self._switches['depth_reach_m']):.1f} m over"
                f" {share:.1f}% of the pixels"
            )
        if c["camera_panned"]:
            how = "with the pan" if self._switches.on("scan_honours_pan") else "as if not"
            extra += f", head panned {c['camera_panned']} frames (projected {how})"
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
