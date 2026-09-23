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
``/fusion/surface`` (PointCloud2 in the frame the volume is painted in — ``volume_frame``, below —
colours from the camera, stamped with the last fused frame, the board's clock) is the zero-crossing
of the field, published beside RTAB-Map's cloud.

THE VOLUME (:mod:`pepin.worldmap`). ``/scan`` is integrated into the same volume along the beams'
own rays — carving free space, a surface at each return, on the lidar's own weight channel, which
the camera's scale-uncertain depth may not repaint. The rays follow the body: with
``imu_lean`` on the scan is placed by the leaning pose, so a beam that climbs 44 cm over 5 m
while the cart tips writes a tabletop where it hit one instead of a wall at the plane, and a
revolution taken past ``lean_gate_deg`` is dropped rather than believed. A lean gravity did
not vote for (``lean_min_quality``: a drifting gyro's own signature) is no lean at all, and the
measurement is placed level instead of by a number nobody measured.

THE VOLUME IS OPEN-LOOP, AND THAT IS THE ARCHITECTURE. It is painted at the pose the tracker
gives, and NOTHING localises against it: no slice of it goes out as a map, no matcher reads it, no
pose is estimated on it. What a slice of it DOES do since 2026-09-21 is mark the costmap
(``/depth_marks`` below) — an obstacle the planner routes around, never a measurement anything
seats itself on, and the loop that rule forbids is the one through the POSE. A tracker that
matches the slice it is painting has a null space it cannot
see out of — turn the map and the heading together and a bearing-only scan maps onto itself — and a
cart parked with its wheels blocked walked 7 degrees and 5-7 cm in 35 minutes through it at fit
0.97-0.99 (2026-09-18), every step under a tenth of a degree. The room's own geometry is RTAB-Map's
loop-closed graph and its occupancy grid; this node paints the 3D surface beside it.

``/fusion/surface`` and ``/depth_marks`` — the same surface, as a cloud for a person and as a fan
for the costmap — are therefore everything this node says about the room, and the volume
is snapshotted to ``world_path`` every ``snapshot_s`` and at shutdown. The snapshot is named after
the graph DATABASE it shares a frame with (:func:`pepin.worldmap.world_path_for`): every voxel was
painted at a pose in that database's optimised frame, so a fresh database means a fresh volume.

A VIEW IS EVIDENCE ONCE (``view_gate``): a parked cart sends the same revolution ten times a
second, and the volume's weights used to count every one as an independent observation.

THE GRAPH MOVES THE VOLUME, IN EVERY MODE, AND THE SIGNAL IS THE GRAPH'S OWN NODES. RTAB-Map
optimises, the room's expression in ``map`` moves with it, and from that moment every voxel painted
before the optimisation is stale by the difference — the room did not move, the frame it is
described in did, and without this a loop drive could never close in the map itself (2026-09-13).

What that difference is had to be settled again under World R, because the volume is painted at the
BOARD TRACKER's pose and no longer at RTAB-Map's. It is NOT the change in ``map -> odom``, whoever
owns that edge: a correction between ``map`` and ``odom`` moves for two opposite reasons — the room
bent (the volume must follow) and the CART was found after some drift (the volume must not, or it is
dragged off the room by the whole size of the recovery, metres after a carry). Only the graph's node
poses tell the two apart, and they do it exactly: a re-localisation leaves every node where it was.
So the signal is the change in the OPTIMISED NODE POSES between two ``/rtabmap/mapGraph`` messages,
read at the newest node the two share — where the cart has just been painting — and accumulated
(:class:`pepin.graphbend.GraphBend`, which holds the argument and the file:line). There is no loop
in this: nothing localises against the volume, so the only thing the volume follows is the graph,
and the graph has never heard of it.

The mechanism downstream is unchanged. Before a frame or a revolution is integrated, the accumulated
bend is compared with the one the volume is painted under, and past ``follow_correction_min_m`` /
``follow_correction_min_deg`` the whole content is carried rigidly by the difference
(:meth:`pepin.worldmap.WorldMap.shift`, 108 ms on the laptop for the live 2.4 M-voxel grid as it
stood on 2026-09-14, 9 ms under ``follow_correction_law`` nearest, on the worker thread, no oftener
than ``follow_correction_min_s``). Smaller corrections are measured against the same anchor and move
the volume together when they add up. While a move is owed but the rate has not let it through,
nothing is painted at all: an observation placed under the new correction and fused into a volume
still standing in the old one is carried past the truth by the whole of that move when it lands. The
grid never moves and nothing extra is published: the next surface out of this node is simply the
moved one.

Beside a LOADED database the volume will normally never move, and that is a prediction rather than a
mode: with ``Mem/IncrementalMemory`` false nothing is written, so no constraint is added, no
optimisation happens and every node comes back where it was. The moves begin the moment the memory
rule lets RTAB-Map learn again (:class:`pepin.graphmode.ModeRule`).

NOTHING IS PAINTED AT A POSE NOBODY TRUSTS. The volume is written in the MAP frame and a TSDF
cannot be un-integrated, so an observation placed by a wrong pose does not add noise — it
deletes the room. Both paint paths therefore ask the same question before they write
(:class:`pepin.watch.PaintTrust`, flags ``fit_gate`` for the camera and ``lidar_fit_gate`` for
the lidar): the fit is at ``DRIVE_FIT`` or better, it was HEARD within the source patience, the
tracker's own ``sigma_xy`` is within ``paint_sigma_m`` where it publishes one
(``/localization/sigma``), and the ``map -> odom`` edge the pose is built on is within a second
of the observation. Anything else is withheld and counted, with the reason in the report line.
Until 2026-09-16 only the camera was asked and only about the number: a fit that stops arriving
keeps its last good value here for ever, and when the laptop's routes from the board died at
19:17 on 2026-09-15 the fusion went on integrating revolutions at a frozen pose — the snapshot
that session ended with keeps 52.9 % of the saved map's walls and has carved 2070 of them free
(scratch/volume_vs_file_seating.py).

THE COSTMAP'S CAMERA MARKS COME FROM THE VOLUME (``/depth_marks``, 2026-09-21). A single stereo
frame is not evidence that something is THERE: SGBM on a herringbone parquet answers small blobs
of disparity 2-5 px too large, which lift floor pixels to 0.15-0.24 m — inside the band the fan
marks in — at about one false bearing a frame, and the first stereo drive left the costmap with
100-300 lethal cells the lidar never saw, 115 "collision ahead" a minute and 44 recoveries (tape
ros/maps/rec/0415_*). The same frames fused into this volume look clean, because a weighted
average and the free space later rays carve are exactly what one wrong opinion does not survive.
So this node slices the volume's own surface — the same surface ``/fusion/surface`` draws, at the
same ``min_weight`` — around the cart into a LaserScan over the whole turn in ``base_link``
(:mod:`pepin.volume_scan`), published at the rate the volume is integrated and stamped with the
observation that was just integrated. The costmap's camera layer MARKS from it and CLEARS from
``/depth_scan``: the frame is the eyewitness of what is open now, the model is what remembers
what is there. ``marks_source`` frame relays ``/depth_scan`` onto the same topic unchanged, which
is the pre-2026-09-21 costmap without a restart. No floor-specific rule anywhere in this.
The topic goes out at ``marks_hz`` (5 Hz), which is the board's local costmap's own
``update_frequency``: the fan crosses the routers to a grid that is read five times a second,
and a frame held back by the cap is still fused into the volume. The stop reflex is bounded by
that costmap tick, never by this publisher.

A PIXEL WITH NO DEPTH IS ALSO A MEASUREMENT (``no_depth_free``, 2026-09-22). Until then only a
ray that MEASURED a surface moved any voxel, and the stereo depth is NaN past the rig's own reach
(2.46 m), so anything standing in front of something farther than that was never carved by
anything: an airborne cluster of 133 voxels at camera height sat unmoved for 60 s with 94 % of its
pixels NaN, and the operator's face stayed in the volume after he walked away, while the same
person at 2 m — a wall at 2.2 m behind him — cleared in about 3 s
(scratch/one_localiser/black_voxels.py). So a depthless pixel now carves free space along its own
ray out to the source's reach less a truncation, at ``no_depth_weight`` of what a measurement
there weighs, because a NaN is also what a textureless wall looks like. The reach is measured off
the frames (:class:`pepin.tsdf.ObservedReach`) and printed in the report line; the lidar's layer
is protected from the carve exactly as it is from the camera's own marks. What the published
depth CANNOT say is why a pixel is NaN — beyond the reach, the edge filter's flying pixels, a
rectification margin and a refused match are one silence in a 32FC1 image — so the whole defence
is the weight, and the measured cost is the camera band's own cells: 81 % of them kept over a
minute, 3 of the 99 lost being cells the lidar's returns mark occupied
(scratch/one_localiser/volume_ab.py). A count of AIRBORNE points does not fall on a volume built
from empty in one minute — carving makes more voxels KNOWN, and a zero crossing needs two known
neighbours, so the cloud grows by about a tenth — which is why the phantom is painted and timed
instead of counted.

AND SINCE 2026-09-22 THE VOLUME LIVES IN ``odom`` (``volume_frame``). Its job is LOCAL OBSTACLE
MEMORY — the nvblox local mapper beside a pose graph, the pattern STVL follows — and local memory
must not depend on global localisation at all. Painted in ``map`` and kept across a day it did the
opposite: on 2026-09-21 the volume held the walls of some twenty re-seatings of the tracker at
once, the yaw aligner sat at its +-4 deg bound refusing frames, and the slice put 300-650 lethal
cells around the cart that the lidar had never seen — proved layer by layer against Nav2's own
grids (scratch/nav2_hang/layer_blame.py), and gone the moment the file was set aside. So under
``odom`` every frame and every revolution is placed by ``odom -> base_link`` and the camera's own
edge, never through ``map -> odom``: no snapshot is read or written, the yaw alignment, the paint
gates and the graph's correction are all inert (there is no global pose in the path to be wrong
about), and the volume is a rolling window — past ``window_recentre_m`` (config/fusion.json) from
the window's centre the box slides onto the cart and what leaves it is forgotten
(:class:`pepin.tsdf.WindowShift`). ``/fusion/surface`` then carries ``odom`` as its frame;
``/depth_marks`` is in ``base_link`` either way, because a fan of ranges about the cart never
depended on the frame the volume was painted in. ``volume_frame`` map is the whole of the old
behaviour, byte for byte.

The flags (:data:`FLAGS`, ``ros/flags.sh set depth_fusion <flag> <value>``): ``enabled``,
``volume_frame``,
``fit_gate``, ``lidar_fit_gate``, ``paint_sigma_m``, ``imu_lean``, ``lean_gate_deg``,
``lean_min_quality``, ``self_heal``, ``align``, ``min_weight``, ``marks_source``,
``marks_min_z``, ``marks_hz``, ``marks_clear``, ``surface_hz``,
``band_half_z``, ``lidar_layer``, ``no_return_free``, ``no_depth_free``, ``no_depth_weight``,
``no_depth_reach_m``, ``colour_fallback``, ``view_gate``, ``snapshot_s``,
``resume_volume``, ``follow_correction``, ``follow_correction_min_m``,
``follow_correction_min_deg``, ``follow_correction_min_s``, ``follow_correction_law``; their
state is printed in every report line, beside the band itself and the source of the plane it is
centred on.
``/fusion/reset`` (std_srvs/Trigger) empties the model, the pairing queues and the tallies.
"""

from __future__ import annotations

import json
import math
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from message_filters import Subscriber, TimeSynchronizer
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rtabmap_msgs.msg import MapGraph
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger

from pepin.depth import Intrinsics, rotation_matrix
from pepin.flags import Flag, FlagSet
from pepin.frame_pose import BASE_FRAME, MAP_FRAME, ODOM_FRAME, FramePoser
from pepin.graphbend import GraphBend
from pepin.lean import LEAN_QUALITY_FLOOR, SCAN_LEAN_GATE_DEG, LeanGate
from pepin.mounts import LASER_FRAME, load_lidar_mount
from pepin.tsdf import (
    YAW_SEARCH,
    AlignReason,
    DepthLaw,
    GridSpec,
    ObservedReach,
    RigidPose,
    align_yaw,
    backproject,
    band_half_z_m,
    window_recentre_m,
)
from pepin.volume_scan import (
    MARKS_ANGLE_MIN,
    MARKS_MIN_RANGE_M,
    MARKS_MIN_Z_M,
    MARKS_RANGE_M,
    MARKS_STEP,
    MarksLaw,
    empty_marks,
    fan_counts,
    free_ranges,
    marks_ranges,
    marks_window,
)
from pepin.watch import DRIVE_FIT, PAINT_SIGMA_M, SOURCE_PATIENCE_S, PaintTrust
from pepin.worldmap import (
    CorrectionFollower,
    LidarLaw,
    PlanarMount,
    SnapshotClock,
    SnapshotTrust,
    ViewGate,
    WorldMap,
    bearings_in_base,
    world_path_for,
)
from pepin_bringup.msgs import (
    array_from_image,
    cloud_from_points,
    rpy_from_transform,
    scan_arrays,
    scan_from_ranges,
    stamp_seconds,
)
from pepin_bringup.node_kit import (
    LeanFeed,
    Switches,
    Tally,
    TfHistory,
    TfLookup,
    Window,
    Worker,
    spin_main,
)

CONFIG = "/ws/config/fusion.json"
LIDAR_CONFIG = "/ws/config/lidar.json"
# The graph database the volume's frame belongs to, as the containers see it, and the volume's own
# file beside it (`world_path_for`): /maps/rtabmap.db -> /maps/rtabmap.world.npz. Every voxel was
# painted at a pose in that database's optimised frame, so the two travel together and a fresh
# database means a fresh volume.
DATABASE = "/maps/rtabmap.db"
SCAN_TOPIC = "/scan"
# THE CAMERA'S TWO WORDS TO THE COSTMAP, and which of them is evidence of what. ``/depth_scan``
# is one frame folded onto the plane (pepin_bringup.depth_stream): an eyewitness of what is OPEN
# right now, and from 2026-09-21 that is all it is asked for — it clears. ``/depth_marks`` is
# this node's answer to what is THERE: the accumulated volume's own surface, sliced around the
# cart (pepin.volume_scan), and it only marks. The two names are one layer in
# ros/params/nav2_params.yaml.
DEPTH_SCAN_TOPIC = "/depth_scan"
MARKS_TOPIC = "/depth_marks"
# ...and, behind ``marks_clear``, the same fan's third word: how far each bearing is KNOWN OPEN
# (pepin.volume_scan.free_ranges). A separate topic and a CLEARING-ONLY source in the same layer,
# because a LaserScan cannot carry "clear to here" and "nothing here" on one range: Nav2's
# ObstacleLayer marks at the end of every finite range it is given, so one source that cleared a
# ray at 1.2 m would plant a lethal cell at 1.2 m — at the frontier of knowledge.
FREE_TOPIC = "/depth_free"
VOLUME = "volume"  # what marks_source chooses between: the model's surface...
FRAME = "frame"  # ...or the single frame's own fan, relayed
# RTAB-Map's optimised graph: the node ids and their poses in ``map``, which is the only signal
# that says the ROOM has moved (pepin.graphbend). Published once per processed snapshot, so about
# once a second at Rtabmap/DetectionRate 1.0 — and only while somebody subscribes, which this does.
GRAPH_TOPIC = "/rtabmap/mapGraph"
SIGMA_TOPIC = "/localization/sigma"  # the tracker's post-fusion sigma, JSON; may never come
TF_WAIT_S = 0.3
BAND_TF_WAIT_S = 5.0  # the static base_link -> laser edge at start: the board publishes it once
PAIR_QUEUE = 40  # depth arrives a fraction of a second after its image; pair by exact stamp
BAND_STRIDE = 3
BAND_MIN_POINTS = 50  # a frame with fewer points in the band is not worth a yaw search
AT_BOUND_STREAK = 30  # ~3 s of frames refused at the search's bound: the model no longer fits
STAGES = ("align", "integrate", "scan", "marks")

# The live flags (CLAUDE.md rule 19), declared last in __init__ so the kit's callback sees no
# other declaration; their state is printed in every report line.
FLAGS = FlagSet(
    Flag(
        "enabled",
        True,
        description="frames are fused into the model; off, they are dropped",
        why="default by design, unmeasured: the kill switch, so the model can be stopped growing"
        " without stopping the node, the camera or the report line",
        on_when="whenever the fused surface or the volume's map is wanted",
        off_when="to freeze the model where it stands — a snapshot to save, a picture to read, a"
        " run where the camera is carried by hand",
    ),
    Flag(
        "volume_frame",
        ODOM_FRAME,
        choices=(ODOM_FRAME, MAP_FRAME),
        description="which frame the volume is painted in. odom: every frame and revolution is"
        " placed by odom -> base_link (plus the camera's own edge) and nothing in the paint path"
        " reads map -> odom at all — no snapshot is read or written, align, the paint gates"
        " (fit_gate, lidar_fit_gate, paint_sigma_m) and follow_correction are inert, and the"
        " volume is a rolling window that slides onto the cart past window_recentre_m of"
        " config/fusion.json and forgets what leaves it. map: the room-sized model of before"
        " 2026-09-22, resumed from and saved to world_path, seated by the yaw search, gated by the"
        " tracker's fit and sigma and carried by the graph's bend. /fusion/surface carries this"
        " frame's own name; /depth_marks is in base_link either way. CHANGING THIS EMPTIES THE"
        " VOLUME: voxels painted in the other frame are a room drawn in coordinates nothing here"
        " shares",
        why="MEASURED 2026-09-21, and it is why this flag exists. The volume's job is LOCAL"
        " OBSTACLE MEMORY (nvblox's local mapper beside a pose graph, STVL's pattern), and local"
        " memory that depends on global localisation inherits every one of its mistakes. Painted"
        " in map and kept, it accumulated the walls of some twenty re-seatings of the tracker in"
        " one evening: the aligner sat at its +-4 deg bound (240 refusals), revolutions were"
        " withheld at sigma 2.98 m while the slice kept publishing, and /depth_marks put 300-650"
        " lethal cells around the cart that the lidar never saw — 199 of 219 outside the camera's"
        " own 94 deg cone and 116 of them BEHIND the cart, measured layer by layer against Nav2's"
        " own grids (scratch/nav2_hang/layer_blame.py). With the file set aside the same drive"
        " marked 0 alien cells. In odom there is no global pose in the path to be wrong about: the"
        " odometry pose is always the pose, which is why the gates that ask whether the tracker is"
        " trustworthy have nothing to judge here and say so in the report line instead of"
        " silently passing. What the window costs when it slides is a copy of the overlap: 0.3-0.9"
        " ms measured on the node's 120x120x34 test grid and 4.7 ms on the live 280x250x34 one"
        " (tests/unit/test_tsdf.py), timed into that same line",
        on_when="odom, everywhere the cart drives: the costmap's camera marks then remember only"
        " what this odometry run has seen around the cart, and a re-seating of the global pose"
        " cannot move a single voxel of them",
        off_when="map for a mapping run whose product is the painted room itself — a surface to"
        " look at in Foxglove, a volume to resume tomorrow — and for the A/B of 2026-09-21, on a"
        " cart nobody is driving",
    ),
    Flag(
        "fit_gate",
        True,
        description="camera frames are fused only while the tracker's pose is trusted"
        " (pepin.watch.PaintTrust: /localization_fit >= 0.50, HEARD within the source patience,"
        " sigma_xy <= paint_sigma_m where a sigma is published, and a map -> odom edge within a"
        " second of the frame); off, every frame is fused",
        why="a fit that STOPS arriving leaves its last good value in this node for ever, so the"
        " gate asks when it was heard as well as what it said: the session whose routes died at"
        " 19:17 on 2026-09-15 went on fusing at a frozen pose for hours. Otherwise: where a"
        " tracker speaks, and the off state is measured: in the first online-SLAM"
        " session, where nobody publishes /localization_fit and the gate had to come off, the"
        " fused floor came out rough — offset +4.7 cm, sd 5.5 cm, 34 % within 3 cm at a tilt of"
        " 0.61 deg — against sd 3.3 cm and 71 % within 3 cm in the known-map mode with a"
        " centimetre tracker pose. The 0.50 itself is the drive rung of the tracker's own ladder"
        " (pepin.watch: blind 0.30, drive 0.50, lost 0.55), inherited, not swept for fusion",
        on_when="in the known-map modes (split, vision), where the board's tracker publishes the"
        " fit: it keeps a frame taken while the pose was wrong out of the model. The launch"
        " brings it up on there and off in SLAM mode; this is how to put it back on by hand",
        off_when="in SLAM mode, where RTAB-Map owns the pose and no tracker speaks — with the"
        " gate on nothing is ever fused there. vslam.launch.py passes fit_gate:=false in that"
        " mode, so nobody has to remember it at the start of a session",
    ),
    Flag(
        "lidar_fit_gate",
        True,
        description="lidar revolutions are integrated only while the tracker's pose is trusted —"
        " the very test fit_gate applies to a camera frame (pepin.watch.PaintTrust); off, every"
        " revolution is integrated at whatever pose TF gives, which is what this node did until"
        " 2026-09-16. A withheld revolution is counted and never painted",
        why="measured by its absence, on the volume itself. A revolution places a wall and"
        " carves free space along its beams, in the MAP frame, and a TSDF cannot be"
        " un-integrated — so a revolution written at a wrong pose does not add noise, it deletes"
        " the room. The camera path has been gated since the beginning and the lidar path never"
        " was; on 2026-09-15 the laptop's routes from the board died at 19:17 and the fusion"
        " went on integrating at the last pose TF held, and the snapshot at the end of that"
        " session keeps 52.9 % of the saved map's walls, has carved 2070 of them free, and the"
        " node's own tracker replayed on its slice seats a median 1.90 m from where the file"
        " puts it, re-seating 3.7-3.8 m off on two of four tapes"
        " (scratch/volume_vs_file_seating.py). That measurement is also part of why nothing"
        " localises against this volume any more. What it costs a HEALTHY"
        " drive was measured too, on the same four tapes with the predicate applied to every"
        " recorded revolution (scratch/paint_gate_on_a_tape.py): 452 of 10048 withheld, 4.5 %,"
        " and that is an upper bound — 439 of them are the recorded pose's own gap (the tape"
        " carries the tracker's pose at about 2 Hz while the node reads a map -> odom edge"
        " broadcast at 20 Hz), leaving 13 revolutions, 0.13 %, where the tracker really had gone"
        " quiet past the source patience. Not one revolution of the four drives was refused for"
        " a LOW fit: a drive the stack was willing to make paints as it always did",
        on_when="the default, wherever a tracker publishes a fit: the volume then holds only"
        " what was seen from a pose the stack was willing to drive on",
        off_when="in SLAM mode, where no tracker speaks and the gate would integrate nothing at"
        " all (vslam.launch.py passes lidar_fit_gate:=false there, as it does fit_gate) — and to"
        " reproduce the old behaviour for a comparison, on a volume nobody will navigate on",
    ),
    Flag(
        "paint_sigma_m",
        PAINT_SIGMA_M,
        description="how sure of itself the tracker must be, in metres of sigma_xy, before this"
        " node paints with its pose — read from /localization/sigma, and ignored entirely while"
        " nothing publishes that topic (the fit gate stands on its own until it exists)",
        why="default by design, unmeasured, and chosen against the voxel: the grid is 5 cm, so"
        " 10 cm of standard deviation puts a wall within two voxels of where it stands and the"
        " surface averages that out, while half a metre writes it into the room. The fit says"
        " how well the last scan matched, the sigma says how well the tracker knows where it is"
        " after fusing everything it has — a lidar-starved tracker riding the odometry can hold"
        " a good fit for a while and a sigma that grows the whole time",
        on_when="raise it in a room where the tracker is honestly less sure and the volume is"
        " being built anyway (an unmapped corner, a first pass)",
        off_when="lower it for a mapping run whose product must be exact: fewer frames, all of"
        " them from a pose the tracker was certain of",
        range=(0.01, 2.0),
    ),
    Flag(
        "imu_lean",
        True,
        description="the cart's lean (pepin.lean, from /imu/data_raw) is followed with the gyro"
        " as well as the accelerometer, and a frame is placed with the lean at its stamp composed"
        " on base_link before the planar odometry instead of as if the cart stood level; the"
        " lidar's scan follows the same switch — its beams are walked as the 3D rays the leaning"
        " body sends them along, and lean_gate_deg drops the scans taken too far from level",
        why="on: the gyro's sign was verified by hand on 2026-09-13 (the cart tipped nose-down"
        " read pitch +10.5 deg, left-side-down read roll -9.4 deg, the fast path following at"
        " once with quality 0.9 while held; scratch/lean_tip_test.txt), and the gyro's zero"
        " offset is learned. Off, the floor anchor keeps the accelerometer-only lean, which by"
        " design ignores any tip shorter than 10 s",
        on_when="after a hand tip through a known angle shows the reported lean following it the"
        " right way and returning to zero",
        off_when="wherever the lean in the report line disagrees with the cart's visible"
        " attitude; off, the lean is still estimated and reported, only not applied",
    ),
    Flag(
        "lean_gate_deg",
        SCAN_LEAN_GATE_DEG,
        description="a scan taken while the cart leans more than this many degrees is not"
        " integrated into the map; only with imu_lean on, which is where the lean is known at all",
        why="default by design, unmeasured: chosen, not fitted. The stake is arithmetic: a beam"
        " at 5 m lands r sin(lean) off the sensor's plane — 26 cm at this 3 degrees, 44 cm at 5 —"
        " so a tipped revolution is looking at another slice of the room. The one replay that"
        " exists (a synthetic 5 degree bump over run 0171, scratch/lidar_lean_effect.txt) has the"
        " gate refusing 21 of 81 revolutions and keeping fewer walls than simply walking the"
        " beams as 3D rays (66.7 % against 74.9 % at the top of the bump, 89.2 % against 92.0 %"
        " six seconds later): there, it cost more than it bought",
        on_when="lower it where the map must stay clean and revolutions are plentiful",
        off_when="90 admits every revolution again, as before the gate existed, and the report"
        " line's leaned_out count says what would have been dropped",
        range=(0.0, 90.0),
    ),
    Flag(
        "lean_min_quality",
        LEAN_QUALITY_FLOOR,
        description="how much of the lean gravity must have voted for (pepin.lean's quality,"
        " printed beside the lean in this line) before a frame or a scan is placed by it: below"
        " it the lean is treated as unknown — the measurement is placed level and the scan gate"
        " admits it",
        why="chosen on a simulation, not on the robot: in scratch/lean_quality_floor_probe.py a"
        " 0.2 deg/s gyro bias reports 3.0 degrees of tip on a level floor at quality 0.02 or"
        " less, nothing past 0.13 degrees of it survives a floor of 0.5, and a real 6 degree"
        " threshold climb keeps quality 1.00 throughout — so the floor costs the feature nothing."
        " The 0.2 deg/s is hypothetical: this chip's worst measured axis is 0.074 deg/s"
        " (config/imu.json's level block). A drifting gyro that reported 3 degrees would"
        " otherwise sit exactly on lean_gate_deg and refuse every revolution",
        on_when="raise it towards 1.0 on a robot that only ever leans when something real pushes"
        " it",
        off_when="0 believes every lean, as before the floor existed: an A/B of the gyro's own"
        " drift",
        range=(0.0, 1.0),
    ),
    Flag(
        "self_heal",
        False,
        description="a streak of 30 frames refused at the alignment bound empties the model, so"
        " it re-seeds from the next frame instead of staying frozen until a human resets it",
        why="off: measured harmful in its first evening (2026-09-11). It was written after one"
        " freeze (the head two hours at 31.5 deg against the config's 26, the law swinging"
        " 0.94-1.60: 284 refusals in one 30 s window, a surface 40 s stale, one hand-sent"
        " /fusion/reset brought back 273 frames per 30 s) — and then fired three times in the"
        " next two hours (19:54, 19:55, 20:05) on ordinary turns and a lidar-off test, wiping"
        " a good model each time: 30 frames at the bound is 3 s, which any pivot reaches. The"
        " cure for the freeze it was written for was the mount (0.383 m) and the TF camera pose,"
        " not the wipe",
        on_when="only with a much longer streak (a minute) and only at rest — as written it is a"
        " model-wiper; until then a stale surface is reset by hand (/fusion/reset)",
        off_when="always, as shipped: the report line keeps 'at bound N' visible, and a model"
        " that stops accepting frames is a mount or pose problem to fix, not to hide",
    ),
    Flag(
        "align",
        True,
        description="frame-to-model: a frame's lidar-height band is turned about the cart to fit"
        " the model before it is fused, and a frame whose best turn is the search's bound (+-4"
        " deg) is refused",
        why="every A/B favours it by a centimetre or two of local surface thickness — 14.0 cm off"
        " against 12.1 on after the scan-carry fix, 12.6 against 11.6 in the demo, with"
        " RTAB-Map's cloud on the same turns at 17.5-20.0 cm — and the score curve on live frames"
        " peaks where it should (0.498 at 0 deg against 0.150 at either +-4 bound). The win is"
        " small, and it was once entirely fake: before the carry fix 89 % of frames answered AT"
        " the bound with a 4.00 deg median turn",
        on_when="on for a model that must stay thin enough to read a wall's face",
        off_when="where the pose is already better than the search can be (a graph's corrections"
        " in SLAM mode), or to prove that a thick surface is the pose's fault: off, no frame is"
        " turned and none is refused",
    ),
    Flag(
        "min_weight",
        4.0,
        description="observations a voxel needs before it is shown in /fusion/surface and read"
        " out as /depth_marks, the two things this node publishes about the room",
        why="4.0 since 2026-09-22 (Artem's call, half a second of frames): at 2 a herringbone"
        " parquet's SGBM floor lift painted lethal cells 0.12-0.30 m past the bumper that lived"
        " one or two frames and stopped the cart four times in 23 s (scratch/one_localiser/"
        "tape_0430_lethal_source.py); the map slice measured 905 walls at 2 and 817 at 6"
        " (scratch/worldmap_from_tape.txt). For the cloud itself nothing was measured; it is the"
        " same number so the picture and the volume's own report agree",
        on_when="raise it to show only what several frames agree on",
        off_when="0 shows every voxel ever touched, noise included — a look at what one pass"
        " sees; SINCE 2026-09-21 IT DOES CHANGE WHAT THE CART DRIVES ON: the same number decides"
        " what /depth_marks marks the camera layer with",
        range=(0.0, 100.0),
    ),
    Flag(
        "marks_source",
        VOLUME,
        choices=(VOLUME, FRAME),
        description="where the camera's MARKS in the costmap come from (/depth_marks): volume,"
        " the accumulated model's own surface sliced around the cart at min_weight"
        " (pepin.volume_scan — the very surface /fusion/surface draws); frame, the latest"
        " /depth_scan relayed unchanged, which is what marked the costmap until 2026-09-21."
        " Either way /depth_scan itself keeps CLEARING the layer: a single frame is the"
        " eyewitness of what is open now",
        why="the first stereo drive measured what one frame is worth as a mark (tape"
        " ros/maps/rec/0415_*): SGBM on the herringbone parquet answers small blobs of"
        " disparity 2-5 px too large, which lift FLOOR pixels to 0.15-0.24 m — inside the band"
        " the fan marks in — at about one false bearing a frame, a different bearing each time."
        " In the costmap that is 100-300 lethal cells the lidar never saw, 115 'collision ahead'"
        " a minute and 44 recoveries in one drive. The same frames fused into the volume look"
        " clean, because fusing is what a single opinion cannot survive: a weighted average and"
        " the free space every later ray carves through the blob. So the marks come from the"
        " model and the clearing stays with the frames — the nvblox arrangement (a probabilistic"
        " volume, a 2D slice of it, the costmap), and no floor-specific rule anywhere in it",
        on_when="volume: wherever the camera layer marks at all. A mark then needs the same"
        " agreement a point of /fusion/surface needs, and the bearings behind the head are"
        " answered too — the volume remembers the table the cart has driven past",
        off_when="frame reproduces the pre-2026-09-21 costmap exactly (the fan itself, marks and"
        " all) without a restart: the A/B for whether a missing mark is the volume's fault, and"
        " the way back if the volume is ever seen to hold a ghost",
    ),
    Flag(
        "marks_min_z",
        MARKS_MIN_Z_M,
        description="the floor of the height band /depth_marks reads the volume in, metres above"
        " the cart's own floor plane; the band's top is the volume's own camera band"
        " (config/fusion.json's camera_band_m)",
        why="default by design, unmeasured as a marks floor: it is pepin.depth's SCAN_MIN_Z_M,"
        " the height /depth_scan has always marked from and the floor of config/fusion.json's"
        " camera_band_m, so the two scans of one layer speak about one band. RAISING IT IS NOT"
        " THE CURE FOR A FLOOR THAT MARKS ITSELF — that is a floor-specific heuristic, and the"
        " thing this topic exists to avoid; what keeps the parquet out of the marks is that a"
        " blob one frame invented is not a surface in the volume",
        on_when="raise it only to measure what a band costs — how much of a real low obstacle"
        " (a plinth, a box) leaves the marks with it",
        off_when="lower it toward the floor to see what the volume itself holds down there,"
        " never to chase a false mark",
        range=(0.0, 1.0),
    ),
    Flag(
        "marks_hz",
        5.0,
        description="the cap on how often /depth_marks is PUBLISHED, in hertz; 0 publishes every"
        " frame, which is what this topic did until 2026-09-22. Only the publication is thinned:"
        " every frame and every revolution is still fused into the volume, and a slice that is"
        " not published is not computed either (the gate is read before the crossing search)",
        why="the one consumer of this topic is the board's LOCAL costmap, whose"
        " update_frequency is 5.0 (ros/params/nav2_params.yaml, 'the stop reflex's slowest link:"
        " a mark waits for this tick'). The topic was published at the rate the volume is"
        " integrated — the camera's 9-9.5 fps plus ~10 Hz of revolutions — so between two and"
        " four of every five fans crossed the zenoh routers to the board only to be overwritten"
        " in the layer before it was next read. The stop reflex is bounded by the costmap tick"
        " and not by this publisher, so nothing about how fast the cart stops changes",
        on_when="raise it only with the costmap's own update_frequency, and only after measuring"
        " what the board does with the extra fans",
        off_when="0 is the pre-2026-09-22 behaviour, one fan per fused frame: the A/B for"
        " whether a missing mark is the cap's fault, and what a bench test on one machine (no"
        " routers in the path) may as well use",
        range=(0.0, 30.0),
    ),
    Flag(
        "marks_clear",
        False,
        description="the fan also says where the volume is KNOWN OPEN: a second, clearing-only"
        f" scan on {FREE_TOPIC} carrying, per bearing, the range of the last column the volume"
        " has observed FREE before the first column it has not (pepin.volume_scan.free_ranges)."
        " A bearing the volume cannot vouch for stays NaN, which clears nothing. Off, the topic"
        " is silent and the camera layer clears from the single frame alone, as it has since"
        " 2026-09-21",
        why="OFF, because the measurement that would justify it says it would do almost nothing."
        " The case FOR it is real: the fan marks over the whole turn while /depth_scan clears"
        " only the head's forward 83 deg, so on run 0434's turn unbacked lethal cells were born"
        " at 109/s against 27/s standing, 52 % of them BEHIND the cart where nothing can ever"
        " raytrace them away, and the count climbed 24 -> 904 in 38 s"
        " (scratch/one_localiser/tape_0434_turn.py). But a clearing ray stops at the first column"
        " that is not open, and on the parked cart of 2026-09-23 the volume held an occupied"
        " column on 717 of 720 bearings: where the camera's own frame shares a bearing with a"
        " mark it AGREES with it within 0.20 m 96 % of the time and sees past it 3 %"
        " (scratch/one_localiser/live_fan_vs_lidar.py), and on the saved room volume the walk"
        " could vouch for 87 bearings of 720. The marks are not stale memory the volume has"
        " already carved — they are what the volume currently holds, and clearing cannot remove"
        " what the model still believes",
        on_when="after the near marks themselves are answered: with the volume no longer holding"
        " a shell at 0.5-0.75 m on every bearing, the walk reaches past it and this is what stops"
        " the ratchet behind the cart. Turn it on together with the yaml's depth_free source and"
        " watch 'clear' in the report line rise off its floor",
        off_when="as shipped, and whenever a cell must not be erased by the camera's own memory:"
        " silent topic, and the layer clears from /depth_scan as it did before",
    ),
    Flag(
        "surface_hz",
        1.0,
        description="how often /fusion/surface is published (the crossing search costs a fraction"
        " of a second)",
        why="default by design, unmeasured; what is measured is the cost it protects — the"
        " surface build took 45 ms a second and stalled the node's executor until it was moved"
        " onto a snapshot taken outside the model lock",
        on_when="raise it for a demo where the surface must follow the head, watching the stage"
        " timings in the report line",
        off_when="lower it towards 0.1 on a busy machine, or where the model matters and the"
        " picture does not",
        range=(0.1, 10.0),
    ),
    Flag(
        "band_half_z",
        band_half_z_m(),
        description="half the height band around the lidar's plane a frame is seated on, metres"
        " (config/fusion.json's band_half_z_m is the default); the band's centre is the plane the"
        " published base_link -> laser edge names, and both are printed in the report line",
        why="default by design, unmeasured as a width: the centre the band sits on is measured,"
        " this half-width is not. The lidar's plane is 0.383 m by tape (2026-09-12), where beams"
        " and vertical walls read the network's scale 3 % apart against 18 % at the 0.200 m that"
        " had been assumed, and moving the band there took the fused band's distance to the lidar"
        " from 12.9 cm to 3.8-5.2 cm. The 0.125 m is the width the band has always had (0.10-0.35"
        " m around the assumed plane) and has never been swept. For scale: the band is the best"
        " layer the camera has — median 9.2 cm against the beams, against 15.7/39.4/46.7 cm for"
        " the slices above it — and a 5 degree lean moves a beam's world height by up to 55.7 cm,"
        " wider than the band itself",
        on_when="widen it when frames are refused for want of band points (the count is in the"
        " report line): a narrow band on a leaning cart has nothing to seat on",
        off_when="narrow it to keep only the rows the beams truly anchor, at the price of fewer"
        " points to align on",
        range=(0.02, 0.5),
    ),
    Flag(
        "lidar_layer",
        True,
        description="/scan is integrated into the volume at the lidar's plane (rays carve free"
        " space, returns mark a surface); off, the volume is the camera's alone, as it was",
        why="replayed from tape 0171 into an empty volume the lidar layer reproduces the saved"
        " map's walls to a median 0.0 cm, p90 13.0 cm, 79.3 % within one cell, and where the"
        " volume says free the saved map agrees 91.3 % of the time — for 1 ms a scan (575 scans"
        " in 0.8 s). It is also protected from the camera: 0 of 13499 lidar cells were changed by"
        " depth, while the camera filled 922 cells the lidar never reached",
        on_when="on wherever the surface must show what the lidar knows, which is every mode: the"
        " beams are the only metric truth in the volume",
        off_when="to measure the camera alone — what the depth adds, and where it lies",
    ),
    Flag(
        "no_return_free",
        False,
        description="a beam that came back with nothing carves free space out to the sensor's"
        " reach (an open door reads as open); off, it writes nothing at all",
        why="default by design, unmeasured: no false-carve rate was ever taken, and with the real"
        " /scan the branch is unreachable anyway — pepin.msgs.scan_arrays turns everything past"
        " range_max into NaN and config/lidar.json's max_range_m is that same 12.0 m, so a"
        " doorway carved nothing and stayed unknown. It stays off because a mirror, a black chair"
        " leg and anything nearer than the 0.05 m minimum all say the identical nothing, and"
        " carving them out to 12 m would rub out the wall behind them",
        on_when="when a beam carries something that separates an open bearing from a mirror or a"
        " black surface — return quality, or the same emptiness confirmed from several"
        " viewpoints; nothing on this robot does today",
        off_when="leave it off: an open door stays unknown, which a planner may be told to cross"
        " (allow_unknown) rather than being told a lie",
    ),
    Flag(
        "no_depth_free",
        True,
        description="a camera pixel with NO depth carves free space along its own ray, from"
        " 0.20 m out to the source's own reach less one truncation, at no_depth_weight of what a"
        " measurement at that range weighs; off, a NaN pixel touches nothing at all, which is"
        " what this node did until 2026-09-22",
        why="THE PHANTOMS THAT NEVER DECAYED. Only a ray that MEASURED something moved any voxel"
        " (pepin.tsdf.Tsdf.integrate), and the stereo depth is cut at the rig's own reach"
        " (2.46 m, pepin.stereo_depth), so anything standing in front of something farther than"
        " that had NaN at every one of its pixels and could never be carved. Measured on the live"
        " volume (scratch/one_localiser/black_voxels.py, 2026-09-22): an airborne cluster of 133"
        " voxels at camera height, 94 % of its pixels NaN, 0 % ever seen free, not one voxel"
        " rewritten in 60 s; the operator's face 175 voxels, all 175 at the same millimetre a"
        " minute later; the same person at 2 m with a wall behind him cleared in about 3 s. AND"
        " THE A/B, 60 s of live frames replayed into two volumes off the robot"
        " (scratch/one_localiser/volume_ab.py: 374 frames, 297 revolutions, the cart parked): a"
        " saturated obstacle painted 0.4 m ahead is 90 % gone in 18 frames, 2.8 s, with the carve"
        " and NEVER without it (94 of its 330 voxels still a surface after the whole minute), and"
        " max_weight 20 alone changes nothing about that — which is the mechanism, nothing ever"
        " touches those voxels. The cost on the same minute: the camera band keeps 81 % of its"
        " occupied cells (99 lost, 41 new), and 3 of the 99 lost are cells the lidar's own"
        " returns mark occupied. It costs this node 2.6 ms a frame on the live grid and the live"
        " 800x600 image, 10.1 -> 12.7 ms, writing 37628 voxels instead of 8705"
        " (scratch/one_localiser/carve_cost.py) — the laptop's, where every camera feature lives",
        on_when="on: it is the only thing that carves a phantom whose background lies beyond the"
        " rig's reach, and the lidar's own layer is protected from it by lidar_layer",
        off_when="off to reproduce the pre-2026-09-22 volume exactly, or if a textureless near"
        " wall (a matcher refusal, not an empty ray) is ever seen to be eaten out of the surface"
        " — the marks audit counts what the camera holds that the lidar does not. The published"
        " depth cannot say WHY a pixel is NaN (beyond the reach, the edge filter's flying pixels,"
        " a rectification margin, a refused match all read the same), so the weight is the whole"
        " of the defence and no_depth_weight is where to turn it down",
    ),
    Flag(
        "no_depth_weight",
        0.5,
        description="what a depthless ray's carve weighs, as a share of what a measurement AT"
        " the source's reach weighs (0.67 at the stereo rig's own 2.44 m, so 0.34 by default);"
        " 0 carves nothing, 1 makes a NaN as convincing as a measurement",
        why="a NaN is not evidence of emptiness: the matcher refuses a textureless wall, a"
        " rectification margin and an over-exposed window with the same silence, and at full"
        " weight those rays would eat a real surface. Half of the weakest honest reading of the"
        " ray is 0.34, which at max_weight 20 clears a saturated phantom in 17 frames by the"
        " integration law and 18 measured (2.8 s at the tape's 6.4 fps,"
        " scratch/one_localiser/volume_ab.py) while a measured surface re-marks itself at"
        " 1.0-4.0 a frame",
        on_when="raise it toward 1 where phantoms outlive their 3 s and the walls are all"
        " lidar-backed anyway",
        off_when="lower it where a near wall the matcher cannot texture is seen to thin; 0 is"
        " no_depth_free off",
        range=(0.0, 2.0),
    ),
    Flag(
        "no_depth_reach_m",
        0.0,
        description="the reach a depthless ray carves to, metres, when it must be stated; 0 (the"
        " default) MEASURES it from the frames themselves — the largest finite depth seen in the"
        " last 60 — and the report line prints what it found",
        why="the reach is the SOURCE's, and nothing publishes it: depth_stream's"
        " depth_reach_m is a looser gate (3.0 m) than the stereo rig itself (2.46 m by its own"
        " error model, DEPTH_SIGMA_M), and carving to 3.0 m would carve through half a metre the"
        " camera never looked at. The published depth is NaN above the reach by construction, so"
        " the largest finite metre in a frame cannot exceed it and equals it whenever anything far"
        " is in view: 2.54 m over the 374 taped frames of 2026-09-22"
        " (scratch/one_localiser/volume_ab.py), which is the number the carve used",
        on_when="state it to pin the carve where a measurement says it belongs — another rig,"
        " or a source whose far pixels are all NaN for another reason",
        off_when="0 leaves it measured, which is what follows a rig change by itself",
        range=(0.0, 12.0),
    ),
    Flag(
        "colour_fallback",
        True,
        description="a surface point whose nearer voxel was never painted by a camera takes the"
        " OTHER neighbour's colour on /fusion/surface; off, it keeps the black that means"
        " 'no camera ever wrote here', which is what the cloud showed until 2026-09-22",
        why="the lidar writes field and weight but no colour (a beam has no colour to give), and"
        " the readout takes the colour of the neighbour nearer the surface — for a beam's own"
        " return always the uncoloured one: 1532 pure-black points on the live volume, 100 % of"
        " them in the lidar's own height band and every one a real wall"
        " (scratch/one_localiser/depth_nan_why.py). AND THE FALLBACK IS A SMALL FIX, measured:"
        " on the taped minute of 2026-09-22 painted with every revolution (21777 lidar voxels,"
        " 5682 surface points) it recovers 16 of 987 black points, 1.6 %"
        " (scratch/one_localiser/volume_ab.py). The other 971 are crossings NEITHER of whose"
        " voxels a camera ever painted — the camera's own surface sits in other voxels than the"
        " beams' — so their black is the truth about them, and the only way to colour them would"
        " be to invent a colour no camera saw. That is why the scan path still writes none",
        on_when="on: it costs nothing at the readout, and every colour it hands out is one the"
        " camera really wrote into that voxel",
        off_when="off to see exactly which points only the lidar holds — the black IS that"
        " measurement, and it is how the 1532 were found",
    ),
    Flag(
        "snapshot_s",
        60.0,
        description="how often the volume is written to world_path (0: only at shutdown)",
        why="default by design, unmeasured: the write holds the model lock for about half a"
        " second on a grid of noise and less on a real one, which at the node's 9.0-9.5 fps is"
        " four or five frames dropped once a minute",
        on_when="shorten it for a long mapping run nobody will be there to shut down cleanly",
        off_when="0 writes only at shutdown — the setting for a demo where no frame may be dropped",
        range=(0.0, 3600.0),
    ),
    Flag(
        "resume_volume",
        True,
        description="a volume snapshot at world_path is loaded at start, so a room the cart has"
        " painted before comes back as it was left; off, the volume starts empty and grows from the"
        " sensors. world_path belongs to the graph DATABASE whose frame the voxels were painted in"
        " (rtabmap.db -> rtabmap.world.npz), so a fresh database means a fresh volume",
        why="resuming its own snapshot is what makes yesterday's painting yesterday's surface"
        " instead of a picture thrown away every"
        " morning. Measured, on the four tapes of 2026-09-13 painted through"
        " scratch/volume_drive_regression.py: a volume seeded from flat3_straight and driven"
        " through a whole tape keeps 79.8 % of the walls it LOOKED at (the two thirds of the flat"
        " a single errand never enters are not counted), carves 447-791 of them free per tape,"
        " and chaining all four drives through one volume instead of re-seeding each time costs"
        " 2.8 points of that share (77.0 %) — the erosion does not run away, and ten passes of"
        " one tape cost 3.9 more points and nothing after that. The one hard rule around it is a"
        " guard: a snapshot is resumed only onto the grid config/fusion.json describes, so a"
        " changed grid starts empty instead of resuming into the wrong place",
        on_when="on in the room the snapshot was taken in, beside the database it was painted"
        " in — the default",
        off_when="off for a new room, beside a fresh database, or to measure how fast the"
        " volume fills from nothing (ros/laptop.sh --fresh passes it off)",
        live=False,
    ),
    Flag(
        "view_gate",
        True,
        description="a revolution taken from a place the volume has already integrated is not"
        " integrated again (pepin.worldmap.ViewGate: the pose must have moved a whole voxel at the"
        " scan's own farthest return before it counts as a new view); off, every revolution is"
        " painted, which is what this node did until 2026-09-18",
        why="A VIEW IS EVIDENCE ONCE, and it was being counted ten times a second: a parked cart"
        " sends the same revolution ten times a second and every one of them used to weigh as an"
        " independent observation, so a standing cart's own paint outgrew everything else in the"
        " volume within a second. Measured on the robot while the board's tracker still matched the"
        " volume it was painting: a cart parked with its wheels blocked walked 7 degrees and 5-7 cm"
        " in 35 minutes at fit 0.97-0.99, every step under a tenth of a degree. That closed loop is"
        " gone — nothing localises against this volume now — and the gate stays because what it"
        " measures is the volume's own honesty: a weight that counts one view a thousand times"
        " calls a single glance a wall the room agrees on. Offline"
        " (scratch/volume_closed_loop.py, 2000 revolutions of a standing cart from tape"
        " 20260913_190422) the old law drifts 0.268 deg/min at fit 1.00 and this gate alone holds"
        " 1993 of the 2000 revolutions and drifts 0.018 deg/min, with 100 % of the room's walls"
        " kept against 94.2 %. The threshold is not one: a return at"
        " range r moves in the map by the translation plus r times the turn, so 'a new view' is"
        " 'no return of this scan stays in the cell it was in', which is the grid's voxel and the"
        " scan's own reach and nothing chosen",
        on_when="always: it is what keeps a weight a count of observations of the room rather than"
        " a count of seconds parked",
        off_when="to reproduce the drift for a comparison, or where the volume must integrate a"
        " long stare on purpose (a mapping run of one corner with the cart on a tripod)",
    ),
    Flag(
        "follow_correction",
        True,
        description="the graph's optimisation moves the voxels, not only the pose: when the"
        " accumulated move of RTAB-Map's own node poses (/rtabmap/mapGraph, read at the newest"
        " shared node) differs from the one the volume is painted under by more than"
        " follow_correction_min_m / _min_deg, the whole content is carried rigidly by that"
        " difference before the next observation goes in. Every mode — the graph is the one source"
        " of truth about the room under World R, and the node poses are the only signal that says"
        " the ROOM moved rather than the cart having been found",
        why="the correction never reached the voxels (2026-09-13): RTAB-Map closed a loop, the"
        " cloud moved with the graph, and the painted room stayed where the pose used to be, so"
        " no loop drive could close in the map itself. The move is not free: on the live"
        " 280x250x34 grid as it stood on 2026-09-14 (2.4 M voxels, 297 k of them painted,"
        " scratch/volume_shift_cost.py) one move of 10 cm / 3 deg costs 108 ms of the worker"
        " thread under the default law and leaves 91 % of the occupied cells, each of the 99th"
        " percentile 9.7 cm from where the correction points — a resampled field is a weighted"
        " average and a surface averaged with the free space in front of it thins. The sensors"
        " repaint what thins within a second of driving; a map left behind the graph never comes"
        " back. Which law pays best is follow_correction_law's question, not this one's."
        " What is NOT measured is a bend of the node poses on this database: beside a loaded one"
        " nothing is written, so nothing optimises and nothing should ever move — which is itself"
        " the live check",
        on_when="always: any drive where the memory rule lets RTAB-Map learn is a drive where a"
        " closure can land, and a map left behind the graph never comes back",
        off_when="to see the old behaviour under the same graph — the graph and the pose move,"
        " the voxels stay — or if a closure is ever seen to smear the map instead of moving it",
    ),
    Flag(
        "follow_correction_min_m",
        0.05,
        description="how far the graph must have bent before the volume is resampled; smaller"
        " bends are kept against the same anchor and move it together when they add up",
        why="one voxel of the grid (5 cm): below it a move cannot change which cell a wall is"
        " in, and the move is not free — 108 ms on this laptop for the live 280x250x34 grid"
        " under the default law, and 9 % of its occupied cells thinned away per move"
        " (scratch/volume_shift_cost.py, 2026-09-14). A smaller threshold spends both to move"
        " the map within the cell it is already in",
        on_when="raise it if graph noise moves the volume more often than the drive needs",
        off_when="lower it toward zero only to watch the mechanism work on tiny corrections; the"
        " map thins at every move",
        range=(0.0, 5.0),
    ),
    Flag(
        "follow_correction_min_deg",
        1.0,
        description="how far the graph's bend must have TURNED before the volume is resampled: the"
        " other half of the threshold, because a turn moves the far end of the flat metres while"
        " the origin stands still",
        why="1 degree is 1.7 cm at a metre (a third of a voxel, where the cart is) and 9 cm at"
        " the 5 m end of the flat — the whole +-9 cm window the laptop's matcher searches. Below"
        " it a turn cannot move a near wall out of its cell; above it a far wall leaves the"
        " matcher's window, and the move costs the measured 108 ms"
        " (scratch/volume_shift_cost.py, 2026-09-14)",
        on_when="raise it with a graph that jitters in heading without closing anything",
        off_when="lower it when a closure's turn must reach the map before its translation does",
        range=(0.0, 180.0),
    ),
    Flag(
        "follow_correction_min_s",
        2.0,
        description="the shortest time between two moves of the volume: a burst of graph"
        " optimisations costs one resample, not one each. The correction is not lost (it is owed"
        " against the same anchor and applied at the next move) — but the frames and"
        " revolutions of that window are not painted, because a volume that owes a move is not"
        " the map they were placed in",
        why="a move costs 108 ms of the worker thread on the live grid"
        " (scratch/volume_shift_cost.py, 2026-09-14), so one every 2 s holds the resample under"
        " 6 % of that thread however hard RTAB-Map optimises. Its price is the observations of"
        " that window: painting them into a volume still standing in the old correction and then"
        " moving the lot puts them past the truth by the whole move — a 30 cm closure left a"
        " freshly painted wall 20 cm beyond where the graph says it is"
        " (scratch/follow_refute.py, 2026-09-14) — so they are refused instead, and two seconds"
        " of a drive is the cheap half of that trade",
        on_when="raise it if a mapping run is ever seen to spend its frames on resampling",
        off_when="0 applies every correction that clears the thresholds, at once",
        range=(0.0, 60.0),
    ),
    Flag(
        "follow_correction_law",
        "nearest",
        description="how the move resamples the volume: blend is the fusion's own weighted"
        " average of the four source columns, nearest takes the one column the cell came from",
        why="MEASURED 2026-09-18, and it reversed the default. blend was chosen because a"
        " weighted average THINS a wall and the sensors repaint a thin wall, while a quantisation"
        " bias is never repainted. LidarLaw.beam_footprint took that premise away: a far crossing"
        " now weighs only the share of its own disc that the voxel covers, so the free space in"
        " front of a wall is weak while the return is full weight, and the average is pulled INTO"
        " the wall. On the synthetic box, one move of 10 cm / 3 deg: blend 430 occupied cells ->"
        " 516 (+20 %, a wall two cells thick) with its worst cell 5.85 cm from where the"
        " correction points — past the voxel — against nearest 430 -> 431 with its worst at"
        " exactly 5.00 cm, half a voxel, which is its whole documented cost"
        " (tests/unit/test_follow_correction.py). A widened wall is a bias in the layer the cart"
        " drives by, which is the failure the old default existed to avoid. Beside that, the cost:"
        " on the live snapshot of 2026-09-14 (280x250x34, 297 k painted voxels) one"
        " move of 10 cm / 3 deg costs blend 108 ms and leaves the 99th occupied cell 9.7 cm from"
        " where the correction points (91 % of the cells survive); nearest costs 9 ms, keeps"
        " every cell and puts the 99th within half a voxel, 2.5 cm — sharp, cheap, and"
        " systematically quantised, which is the bias class the grid snapping of 2026-09-13 was"
        " written to kill. blend is the default because a bias is not repainted by the sensors"
        " and a thinned wall is; the 9.7 cm says that argument is not settled, and the morning's"
        " loop drive is what settles it (scratch/volume_shift_cost.py)",
        on_when="nearest: every cell kept, each within half a voxel, and 9 ms instead of 108",
        off_when="blend to reproduce the old default, or on a volume painted with"
        " beam_footprint off, where its thinning argument still holds — the A/B is a live flag,"
        " no restart",
        choices=("blend", "nearest"),
    ),
)


def pose_from_pose_msg(msg: Any) -> RigidPose:
    """A ``geometry_msgs/Pose`` as a rotation matrix and a translation (``map <- that node``):
    what ``/rtabmap/mapGraph`` carries for every node of the optimised graph.

    Its own function because a Pose spells its two halves ``position`` and ``orientation`` while
    :func:`pepin_bringup.msgs.pose_from_transform` reads a Transform's ``translation`` and
    ``rotation``, and a getattr that guessed between them would read one field of each on a message
    that happened to have both.
    """
    p, q = msg.position, msg.orientation
    return RigidPose(rotation_matrix(q.x, q.y, q.z, q.w), np.array([p.x, p.y, p.z]))


def band_z_m(plane_z_m: float, half_m: float) -> tuple[float, float]:
    """The height band whose points are exact by construction, metres above the floor:
    ``plane_z_m`` (the lidar's own plane) plus and minus ``half_m``.

    The depth image is anchored on the beams, so this is the layer a frame may be turned by. It
    holds only while the band's centre is the plane the beams actually come from, which is why
    the caller reads that plane from TF (:meth:`DepthFusion._plane_z_m`) rather than from the
    config this process happens to have."""
    return (plane_z_m - half_m, plane_z_m + half_m)


class DepthFusion(Node):
    """Fuses depth frames into the TSDF and publishes its surface."""

    def __init__(self) -> None:
        super().__init__("depth_fusion")
        # Callbacks can run BEFORE this constructor is done: TfLookup (node_kit) starts a spin
        # thread for this node, and the pair/scan subscriptions below are live from the moment
        # they exist. A frame that arrived in that window met a half-built node
        # (AttributeError on _worker, 2026-09-22: the exception left rclpy's spin, the process
        # lived on with no executor and the marks went silent). Until the last line of __init__
        # every callback drops what it is handed.
        self._up = False
        config = Path(str(self.declare_parameter("config", CONFIG).value))
        self._spec = GridSpec.load(config)
        self.declare_parameter("mode", "vision")
        # The band's centre is the lidar's plane, and the only copy of it both sides of the
        # bridge agree on is the published base_link -> laser edge: this process reads
        # config/lidar.json from the laptop's checkout while the beams are published from the
        # board's own synced copy, and between a mount change and ros/sync.sh the two differ by
        # the whole change. TF is asked at start (and again while it has not answered); the
        # mount here is only the fallback, and the report line says which one the band sits on.
        self._plane_z_m = load_lidar_mount().z_m
        self._plane_source = "config"
        self._band_z_m = band_z_m(self._plane_z_m, band_half_z_m())
        # Which mode the stack was brought up in: the one thing it decides here is whether the
        # GRAPH owns map -> odom, which is what says whether the volume must follow a correction.
        self._mode = str(self.get_parameter("mode").value)
        # NOTHING LOCALISES AGAINST THIS VOLUME, so it publishes no map and claims to be no room.
        # Its frame is the graph DATABASE's — every voxel was painted at a pose in RTAB-Map's
        # optimised frame — so the snapshot is named after the database it belongs to
        # (pepin.worldmap.world_path_for: /maps/rtabmap.db -> /maps/rtabmap.world.npz) and a fresh
        # database means a fresh volume. There is no pgm in the loop at all, neither as a seed nor
        # as an exported cache: a saved picture put walls into the volume the lidar had not seen,
        # and an export made a second file claiming to be the map (2026-09-18, when it wrote itself
        # over the seed).
        #
        # config/fusion.json's box describes a volume being BORN, and a newborn is centred on the
        # cart: a volume that already exists owns its own grid, which comes back with its snapshot
        # (:meth:`_start_state`). A box laid out around the map's origin would not even contain a
        # cart that woke up at (-9.4, +2.5) — 2026-09-13, when 239 revolutions integrated the far
        # wall and nothing else.
        self._spec = self._spec.centred_on_start()
        self._database = Path(str(self.declare_parameter("database", DATABASE).value))
        self._world_path = Path(
            str(self.declare_parameter("world_path", str(world_path_for(self._database))).value)
        )
        # The lidar's plane is calibrated, never typed: it comes from config/lidar.json, the one
        # file the board's launch publishes the laser transform from.
        self._mount = PlanarMount.from_config(
            str(self.declare_parameter("lidar_config", LIDAR_CONFIG).value)
        )
        self._switches = Switches(self, FLAGS, on_change=self._on_switch)
        self._tally = Tally(STAGES)
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(PointCloud2, "/fusion/surface", reliable)
        # The costmap's camera MARKS: the volume's surface around the cart, one range per
        # bearing, published at the rate the volume is integrated (:meth:`_publish_marks`).
        self._marks_pub = self.create_publisher(LaserScan, MARKS_TOPIC, reliable)
        # ...and, under marks_clear, the same fan's clearing half: how far each bearing is known
        # open. A topic of its own because one LaserScan cannot say "clear to here" without also
        # marking there (see FREE_TOPIC). Silent while the flag is off.
        self._free_pub = self.create_publisher(LaserScan, FREE_TOPIC, reliable)
        self._marks_at = 0.0  # monotonic seconds of the last published fan: the marks_hz cap
        # ...and the frame the marks used to come from, so the old behaviour is one live flag
        # away (marks_source frame relays this message unchanged). Local to the laptop: the
        # depth stream publishes it here, and only the OUTPUT of this node crosses to the board.
        self.create_subscription(LaserScan, DEPTH_SCAN_TOPIC, self._on_depth_scan, reliable)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, reliable)
        self.create_subscription(Float32, "/localization_fit", self._on_fit, reliable)
        # How sure the tracker is of itself, beside how well its last scan fitted: JSON on
        # the tracker's own topic (sigma_xy metres, sigma_yaw degrees). Nobody may publish
        # it yet, and the gate below works without it — an absent sigma is not a refusal.
        self.create_subscription(String, SIGMA_TOPIC, self._on_sigma, reliable)
        # RTAB-Map's optimised graph, once per processed snapshot: the node poses whose movement is
        # what old paint is stale by (pepin.graphbend, and :meth:`_follow`'s docstring for why it is
        # these and not map -> odom). Two deep — only the newest graph says where the room is now.
        self.create_subscription(
            MapGraph,
            GRAPH_TOPIC,
            self._on_graph,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.create_service(Trigger, "/fusion/reset", self._on_reset)
        # the depth copies the image's header, so the pair has one exact stamp; the synchronizer
        # keeps PAIR_QUEUE of each and calls back under its own lock, on the executor thread
        depth_sub = Subscriber(self, Image, "/camera/depth", qos_profile=reliable)
        image_sub = Subscriber(self, Image, "/camera/image", qos_profile=reliable)
        depth_sub.registerCallback(lambda _msg: self._tally.count("depth_in"))
        self._sync = TimeSynchronizer([depth_sub, image_sub], PAIR_QUEUE)
        self._sync.registerCallback(self._on_pair)
        self._tf = TfLookup(self, on_failure=self._on_tf_failure)
        self._lean = LeanFeed(
            self,
            config.parent,
            use_gyro=self._switches.on("imu_lean"),
            on_unmounted=self._on_unmounted,
        )
        history = TfHistory(self._tf, timeout_s=TF_WAIT_S)
        self._poser = FramePoser(
            history,
            lean=self._lean,
            apply_lean=self._switches.on("imu_lean"),
            min_lean_quality=float(self._switches["lean_min_quality"]),
        )
        # ...and the same questions asked of the ODOMETRY instead of the map: one poser per frame
        # the volume can be painted in (``volume_frame``), sharing one TF history and one lean, so
        # the paint path picks a frame rather than carrying a frame name through every lookup.
        # Under odom nothing in that path touches map -> odom at all, which is the whole point.
        self._odom_poser = FramePoser(
            history,
            map_frame=ODOM_FRAME,
            lean=self._lean,
            apply_lean=self._switches.on("imu_lean"),
            min_lean_quality=float(self._switches["lean_min_quality"]),
        )
        # How far the cart may leave the window's centre before the box slides onto it, from the
        # node's own config (pepin.tsdf.window_recentre_m); read only under volume_frame odom.
        self._window_recentre_m = window_recentre_m(config)
        self._recentre_ms = 0.0  # the last slide's cost, a level the report line reads
        self._read_plane(BAND_TF_WAIT_S)
        # The scan's own answer to the lean: a frame can be placed leaning, a revolution taken
        # too far from level can only be dropped (pepin.lean.LeanGate).
        self._gate = LeanGate(float(self._switches["lean_gate_deg"]))
        self._intr: Intrinsics | None = None
        self._fit = 0.0  # no report yet reads as lost: every gate here compares with <
        self._fit_at = -math.inf  # ...and WHEN it was heard: a fit that stopped is not a fit
        self._sigma_xy_m: float | None = None  # the tracker's own, while it publishes one
        self._sigma_at = -math.inf
        self._lock = threading.Lock()  # the model and its last stamp, worker vs publisher
        self._world = WorldMap(self._spec, self._mount)
        self._last_stamp: Any = None  # the last fused frame's header stamp, the board's clock
        self._bound_streak = 0  # consecutive frames refused at the bound (self-healing)
        # How far the depth SOURCE answers, measured off its own frames: what a depthless ray may
        # carve to (no_depth_free), and not the publisher's looser gate (:class:`ObservedReach`).
        self._reach = ObservedReach()
        self._surface_points = 0
        # The last fan in three numbers — bearings that MARK, bearings that CLEAR, bearings that
        # say nothing — report levels, not tallies: what one slice held, not how many went out.
        self._marks_bearings = 0
        self._marks_clearing = 0
        self._marks_silent = 0
        self._worker = Worker(self._on_work, name="fusion", on_error=self._on_work_error).start()
        # The scan has its own worker: integrating a revolution takes milliseconds, but it waits
        # for the lock a camera frame holds, and the executor thread must not wait with it.
        self._scans = Worker(self._on_scan_work, name="scan", on_error=self._on_work_error).start()
        # Reliable, like every other subscriber of this topic (the tracker, the depth stream):
        # the board publishes it reliably and a best-effort reader of a reliable writer over
        # the bridge gets nothing at all.
        self.create_subscription(
            LaserScan,
            SCAN_TOPIC,
            self._on_scan,
            QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE),
        )
        self._laser: tuple[PlanarMount, float, bool] | None = None  # mount, yaw, upside down
        self._snapshots = SnapshotClock(float(self._switches["snapshot_s"]))
        # ...and whether the volume in memory is fit to replace the one on disk at all
        # (pepin.worldmap.SnapshotTrust): a run that loses the tracker must leave the last good
        # volume where it is. ros/maps/world_live.npz.mess-20260917 is the file that rule is for.
        self._trust = SnapshotTrust(SOURCE_PATIENCE_S)
        # ...and whether a revolution is a NEW view at all (pepin.worldmap.ViewGate): a view is
        # evidence once, and the threshold is the grid's own voxel.
        self._views = ViewGate(self._spec.voxel_m)
        # The volume follows THE GRAPH's own bend, in every mode: the accumulated move of RTAB-Map's
        # optimised node poses (pepin.graphbend.GraphBend), which is the one signal that says the
        # ROOM moved rather than the cart having been found somewhere else. It used to be the change
        # in map -> odom, and that could only ever be right where the graph owned that edge.
        self._bend = GraphBend()
        self._graphs = 0  # how many graphs have arrived: zero means there is nothing to follow
        self._follower = CorrectionFollower()
        self._follow_ms = 0.0  # the last resample's cost, a level the report line reads
        self._followed_at = 0.0  # monotonic seconds of the last move: the throttle
        self._start_state()
        self._surface_timer = self.create_timer(
            self._period(self._switches["surface_hz"]), self._publish_surface
        )
        self.create_timer(30.0, self._report)
        nx, ny, nz = self._spec.shape
        self.get_logger().info(
            f"fusion up: {nx}x{ny}x{nz} voxels of {self._spec.voxel_m * 100:.0f} cm from"
            f" {self._spec.origin}; {self._switches.state()}; {self._band_text()}; fused while"
            f" /localization_fit >= {DRIVE_FIT:.2f}; mode {self._mode}; nothing localises against"
            f" this volume — it is painted open-loop and published as /fusion/surface (frame"
            f" {self._volume_frame}) and as the costmap's camera marks on {MARKS_TOPIC}"
            f" ({self._switches['marks_source']}); {self._frame_line()}"
        )
        self._up = True

    @property
    def _volume_frame(self) -> str:
        """The frame the volume is painted in and ``/fusion/surface`` is published in: the
        ``volume_frame`` flag, whose values are the frames' own names."""
        return str(self._switches["volume_frame"])

    @property
    def _odom_volume(self) -> bool:
        """Whether the volume is the rolling LOCAL window in ``odom`` (the default) rather than
        the room-sized model in ``map``: the one question that decides which pose paints it,
        whether it is snapshotted, and whether the map-frame gates mean anything at all."""
        return self._volume_frame == ODOM_FRAME

    @property
    def _poser_now(self) -> FramePoser:
        """The poser that places the next observation: the odometry's under ``volume_frame``
        odom — where nothing in the paint path reads ``map -> odom`` — else the map's."""
        return self._odom_poser if self._odom_volume else self._poser

    def _start_state(self) -> None:
        """What the volume starts as: the snapshot beside this graph database if there is one, else
        a new empty volume born under the cart.

        TWO STATES, NOT THREE, AND NEITHER OF THEM IS A PICTURE. There used to be a middle one —
        a saved pgm written into the lidar's layer — and it was the reason the volume had to be
        carved back into shape: it put walls in it the lidar had not seen, on an evening
        that had passed. So the volume is either one this robot has already painted (resume it, and
        the GRID the snapshot carries with it: a volume that exists owns its own lattice) or one it
        has not (born empty, here, now). ``resume_volume`` is the override for a measurement.

        ``self._resumed_age_s`` is how old what was resumed was, so a volume two weeks stale is
        visible rather than assumed fresh.

        UNDER ``volume_frame`` odom THERE IS ONLY ONE STATE: born empty, here, now. A snapshot is
        a room saved in a frame somebody else's optimisation defines, and the odometry frame of
        this run is not that frame — it is born wherever the wheels were switched on. Nothing is
        read and nothing is written; the window fills from the sensors and slides with the cart.
        """
        self._resumed_age_s = math.inf
        # A NEWBORN IS CENTRED ON THE CART, not on the map's origin. The map frame is born under
        # the cart only where this session creates it; a cart that wakes up at (-9.4, +2.5) in a
        # frame somebody else made is outside a box laid out around the origin, and a volume that
        # does not contain the robot integrates the far wall and nothing else (2026-09-13: 239
        # revolutions, 2026-09-18: a kicked node on a box from -7.0).
        here = self._tf.transform(self._volume_frame, BASE_FRAME, timeout_s=TF_WAIT_S)
        if here is not None:
            at = (float(here.transform.translation.x), float(here.transform.translation.y))
            self._spec = self._spec.centred_on(at)
        if (
            not self._odom_volume
            and self._switches.on("resume_volume")
            and self._world_path.exists()
        ):
            try:
                resumed = WorldMap.load(self._world_path, self._mount)
            except (ValueError, OSError, zipfile.BadZipFile, KeyError) as exc:
                # a snapshot cut mid-write by a hard reset is a 403-byte zip (2026-09-14), and one
                # of an older version is another volume's format: born empty rather than dying at
                # every respawn
                self.get_logger().warning(f"{self._world_path}: not resumed ({exc})")
            else:
                if abs(resumed.spec.voxel_m - self._spec.voxel_m) > 1e-9:
                    # The one thing that cannot be the same volume: another lattice pitch. The box
                    # and its origin may differ freely — they are the room's, not the config's.
                    self.get_logger().warning(
                        f"{self._world_path}: saved at {resumed.spec.voxel_m * 100:.1f} cm a voxel,"
                        f" not {self._spec.voxel_m * 100:.1f}; born empty instead"
                    )
                else:
                    self._world = resumed
                    self._spec = resumed.spec
                    self._resumed_age_s = max(0.0, time.time() - self._world_path.stat().st_mtime)
                    stats = resumed.maturity()
                    self.get_logger().info(
                        f"resumed {self._world_path}: {stats['voxels']:.0f} voxels,"
                        f" {stats['frames']:.0f} frames, stamp {resumed.stamp:.0f}, written"
                        f" {self._resumed_age_s / 3600:.1f} h ago"
                    )
                    return
        self._world = WorldMap(self._spec, self._mount)
        born = (
            f"no snapshot is read or written in {ODOM_FRAME}"
            if self._odom_volume
            else f"no snapshot at {self._world_path}"
        )
        self.get_logger().info(
            f"{born}: the volume is born empty under the cart in {self._volume_frame} on"
            f" {self._spec.shape} voxels from {self._spec.origin}"
        )

    def close(self) -> None:
        """Stop the workers and the TF listener, snapshot the volume, and wait for them all,
        before the node is destroyed: a run's volume outlives the run."""
        if not self._worker.stop():
            self.get_logger().warning("the fusion worker did not finish its frame; leaving anyway")
        self._scans.stop()
        self._snapshot()
        self._tf.close()

    @staticmethod
    def _period(surface_hz: float) -> float:
        return 1.0 / max(surface_hz, 0.1)

    # ---- the lidar's plane ---------------------------------------------------------------
    def _read_plane(self, wait_s: float) -> bool:
        """Take the band's centre from the published ``base_link -> laser`` edge, the height the
        beams the band is anchored on actually come from; ``True`` when TF answered.

        A failure leaves the fallback in place (config/lidar.json as this process reads it) and
        is said out loud: the two can disagree by a whole mount change until ros/sync.sh has
        run, and a band centred on the wrong plane holds no exact points at all.
        """
        if self._plane_source == "tf":
            return True
        edge = self._tf.transform(BASE_FRAME, LASER_FRAME, timeout_s=wait_s)
        if edge is None:
            self.get_logger().warning(
                f"no {BASE_FRAME} -> {LASER_FRAME} yet: the band sits on config/lidar.json's"
                f" {self._plane_z_m:.3f} m, which is this side's copy of the mount"
            )
            self._set_band()
            return False
        self._plane_z_m = float(edge.transform.translation.z)
        self._plane_source = "tf"
        self._set_band()
        return True

    def _set_band(self) -> None:
        """Rebuild the height band from the current plane and the ``band_half_z`` flag."""
        self._band_z_m = band_z_m(self._plane_z_m, float(self._switches["band_half_z"]))

    def _band_text(self) -> str:
        """The band for a report line: its two heights, its centre and where that centre came
        from — ``band 0.26-0.51 m (plane 0.383 m from tf)``."""
        return (
            f"band {self._band_z_m[0]:.2f}-{self._band_z_m[1]:.2f} m"
            f" (plane {self._plane_z_m:.3f} m from {self._plane_source})"
        )

    # ---- switches ------------------------------------------------------------------------
    def _on_switch(self, name: str, _old: Any, new: Any) -> None:
        """A flag changed: ``imu_lean`` is the estimator's switch and the poser's — one name,
        one meaning, in every node that has it — ``lean_gate_deg`` the scan gate's,
        ``lean_min_quality`` the poser's floor under a lean, ``band_half_z`` rebuilds the height
        band, the two rates retime their timer, ``snapshot_s`` its clock, ``volume_frame`` empties
        the volume, and the rest are only read where they are used."""
        if name == "imu_lean":
            self._poser.apply_lean = self._odom_poser.apply_lean = bool(new)
            self._lean.use_gyro = bool(new)
            return
        if name == "band_half_z":
            self._set_band()
            return
        if name == "lean_gate_deg":
            self._gate.gate_deg = float(new)
            return
        if name == "lean_min_quality":
            self._poser.min_lean_quality = self._odom_poser.min_lean_quality = float(new)
            return
        if name == "volume_frame":
            self._reframe(str(new))
            return
        if name == "snapshot_s":
            self._snapshots = SnapshotClock(float(new), self._snapshots.last_s)
            return
        if name != "surface_hz":
            return
        timer = self._surface_timer
        try:
            timer.timer_period_ns = int(self._period(float(new)) * 1e9)
        except (AttributeError, TypeError) as exc:  # an rclpy without a live period
            raise ValueError(f"{name} cannot change live: {exc}") from exc

    def _reframe(self, frame: str) -> None:
        """``volume_frame`` has just changed: empty the volume and lay a new box under the cart in
        the frame that is now in force.

        A VOLUME CANNOT BE CARRIED BETWEEN FRAMES. Its voxels are metres of ``map`` or metres of
        ``odom``, and the two are related by a correction that is exactly what this flag exists to
        keep out of the painting — so the old content is not moved, it is dropped, and the window
        fills again from the sensors within seconds of driving. Everything that remembers where
        the last paint happened goes with it: the view gate's last pose, the correction the
        content was anchored in, the stamp of the last fused frame.
        """
        here = self._tf.transform(frame, BASE_FRAME, timeout_s=TF_WAIT_S)
        at = (
            (float(here.transform.translation.x), float(here.transform.translation.y))
            if here is not None
            else self._spec.centre_xy
        )
        with self._lock:
            self._spec = self._spec.centred_on(at)
            self._world = self._fresh_world()
            self._last_stamp = None
            self._follower = CorrectionFollower()
            self._views = ViewGate(self._spec.voxel_m)
        self._recentre_ms = 0.0
        self.get_logger().warning(
            f"volume_frame is {frame} now: the volume is emptied and born again under the cart"
            f" ({at[0]:+.2f}, {at[1]:+.2f} in {frame}) — voxels painted in the other frame are a"
            f" room drawn in coordinates nothing here shares; {self._frame_line()}"
        )

    def _on_work_error(self, text: str) -> None:
        self.get_logger().error(f"fusion failed on a frame:\n{text}")

    def _on_unmounted(self, frame_id: str) -> None:
        """IMU readings the node cannot turn into base_link: the lean stays unknown and every
        frame is placed level, whatever ``imu_lean`` says."""
        self._tally.count("unleaned")
        self.get_logger().error(
            f"IMU readings in {frame_id} and no mount: frames are placed as if the cart were level",
            throttle_duration_sec=60,
        )

    def _on_tf_failure(self, kind: str, text: str) -> None:
        self._tally.count("no_tf")
        self._tally.count("tf_" + kind)
        self._tally.note(kind, text)

    def _fresh_world(self) -> WorldMap:
        """An empty volume on the same grid: every cell forgotten.

        Nothing outside this node reads the volume, so emptying it costs no tracker anything — it
        costs the surface cloud until the sensors have painted one again. The snapshot on disk is
        untouched by a reset, and the guard in :meth:`_snapshot` keeps it that way until the
        painting is trusted again.
        """
        return WorldMap(self._spec, self._mount)

    def _on_reset(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        with self._lock:
            self._world = self._fresh_world()
            self._last_stamp = None
            self._follower = CorrectionFollower()  # a new volume is painted under what is in
            # force now, not under the correction the emptied one was standing in
        self._worker.clear()
        self._scans.clear()
        with self._sync.lock:
            for queue in self._sync.queues:
                queue.clear()
        self._tally.take()
        response.success = True
        response.message = "the volume is empty"
        self.get_logger().info(
            "fusion: model, pairing queues and tallies reset; the volume is empty and"
            f" {self._world_path} is untouched"
        )
        return response

    # ---- inputs --------------------------------------------------------------------------
    def _on_info(self, msg: CameraInfo) -> None:
        if not self._up:
            return  # the node is still being built (see __init__)
        self._intr = Intrinsics.from_camera_info(msg.k, msg.width, msg.height)

    def _on_fit(self, msg: Float32) -> None:
        """The tracker's scan-to-map fit, and the moment it was heard: a topic that stops
        arriving leaves the last good number behind, and painting on that is what emptied the
        volume on 2026-09-15."""
        if not self._up:
            return  # the node is still being built (see __init__)
        self._fit = float(msg.data)
        self._fit_at = time.monotonic()

    def _on_sigma(self, msg: String) -> None:
        """The tracker's post-fusion sigma as JSON (``sigma_xy`` metres, ``sigma_yaw``
        degrees). A message this node cannot read is counted and ignored — the fit gate stands
        on its own, and a malformed sigma may not stop the room being painted."""
        if not self._up:
            return  # the node is still being built (see __init__)
        try:
            self._sigma_xy_m = float(json.loads(msg.data)["sigma_xy"])
        except (ValueError, TypeError, KeyError):
            self._tally.count("bad_sigma")
            return
        self._sigma_at = time.monotonic()

    def _on_pair(self, depth: Image, image: Image) -> None:
        """A depth frame with its picture, same stamp: the newest pair waits for the worker,
        an older one still waiting is dropped (the model wants the latest view, not a backlog)."""
        if not self._up:
            return  # the node is still being built (see __init__)
        self._tally.count("pairs")
        if self._worker.offer((depth, image)):
            self._tally.count("dropped")

    def _on_scan(self, msg: LaserScan) -> None:
        """A lidar revolution: straight to its worker, newest first (an older one still waiting
        is dropped — the volume wants the room as it is, not a backlog)."""
        if not self._up:
            return
        self._tally.count("scans_in")
        if self._scans.offer(msg):
            self._tally.count("scans_dropped")

    def _on_scan_work(self, msg: LaserScan) -> None:
        """One revolution into the volume at the pose TF gives for its stamp: rays carve free
        space and returns mark a surface, along the rays the body really sent them
        (pepin.worldmap).

        The pose is the poser's, so with ``imu_lean`` on it carries the lean of that moment and
        the beams climb with the body; with it off the pose is the planar one and the beams
        sweep the plane, as they always did. Past ``lean_gate_deg`` there is nothing worth
        writing — the beams are in another slice of the room — and the scan is dropped.

        The graph's correction is followed before the layer's own gate: the volume follows the
        map even where this node writes no lidar layer at all, and a revolution is the steady
        pulse that carries it between camera frames. A revolution that arrives while the volume
        owes the graph a move is dropped, not written into a map that is about to move under
        it."""
        if not self._follow(msg.header.stamp):
            return  # the volume owes the graph a move: nothing goes in until it has been made
        if not self._switches.on("lidar_layer"):
            return
        if self._laser is None and not self._lookup_laser(msg.header.frame_id):
            return
        assert self._laser is not None
        mount, yaw, mirrored = self._laser
        at = stamp_seconds(msg.header.stamp)
        poser = self._poser_now
        base = poser.base_in_map(at)
        if base is None:
            return  # counted by the TF failure handler
        if not self._gate.admits(poser.lean_at(at)):
            self._tally.count("leaned_out")
            return
        if self._switches.on("lidar_fit_gate") and not self._odom_volume:
            # ...and in odom there is nothing for it to judge: the odometry pose IS the pose this
            # volume is drawn in, and a tracker's fit says nothing about it (the report line says
            # so rather than leaving the flag looking as though it were doing something).
            refusal = self._paint_refusal(at)
            if refusal is not None:
                self._withhold("untrusted", refusal)
                return  # a revolution painted at a pose nobody trusts carves the room away
        angles, ranges = scan_arrays(msg)
        if self._switches.on("view_gate"):
            # A VIEW IS EVIDENCE ONCE. The reach is this scan's own farthest return, so what
            # counts as "moved" is what moves one of ITS returns into another voxel.
            reach = float(np.nanmax(ranges)) if np.isfinite(ranges).any() else 0.0
            # ...and this is the CART's heading, which is not the mount's ``yaw`` above: writing it
            # into that name turned every beam of the revolution by the cart's own heading a second
            # time, through bearings_in_base, on every scan the gate admitted (the default).
            heading = math.atan2(float(base.rotation[1, 0]), float(base.rotation[0, 0]))
            if not self._views.admits(
                float(base.translation[0]), float(base.translation[1]), heading, reach
            ):
                self._tally.count("same_view")
                return
        self._roll_window(base)
        with self._tally.measure("scan"), self._lock:
            self._world.law = self._law()
            touched = self._world.integrate_scan(
                bearings_in_base(angles, yaw, mirrored), ranges, base, mount, stamp=at
            )
        self._tally.count("revolutions")
        self._tally.count("scan_voxels", touched)
        # ...and the costmap hears what the volume holds now, at the pose this revolution was
        # painted by and on its own stamp.
        self._publish_marks(base, msg.header.stamp)
        now = time.monotonic()
        self._trust.painted(now)  # ...and the map on disk may be replaced by this one
        if self._snapshots.due(now) and self._switches["snapshot_s"] > 0.0:
            self._snapshot()

    def _roll_window(self, base: RigidPose) -> None:
        """Keep the rolling window on the cart, before the observation that found it there goes
        in: past ``window_recentre_m`` from the window's centre the box slides onto the cart by
        whole voxels, the overlap is kept where it stands and what left the window is forgotten
        (:meth:`pepin.worldmap.WorldMap.recentre`).

        ONLY IN ``odom``. A volume in ``map`` is the room and the room does not move with the
        cart; a volume in ``odom`` is local obstacle memory, whose whole reason to exist is that it
        holds what is around the cart NOW and inherits nothing from a global pose.

        Called from both paint paths, under the model lock, and timed: the slide is one copy per
        channel — measured 0.3-0.9 ms on the 120x120x34 test grid and 4.7 ms on the live
        280x250x34 one (tests/unit/test_tsdf.py) against the 9-108 ms a graph correction's
        resample costs — and the report line carries the last one.
        """
        if not self._odom_volume:
            return
        at = (float(base.translation[0]), float(base.translation[1]))
        if self._spec.off_centre_m(at) <= self._window_recentre_m:
            return
        started = time.perf_counter()
        with self._lock:
            move = self._world.recentre(at)
            self._spec = self._world.spec
        self._recentre_ms = (time.perf_counter() - started) * 1e3
        self._tally.count("recentres")
        self.get_logger().info(
            f"the cart left the window's centre: it slides {move.text()} onto"
            f" ({at[0]:+.2f}, {at[1]:+.2f}) in {ODOM_FRAME}, what left it is forgotten"
            f" ({self._recentre_ms:.0f} ms)"
        )

    def _paint_refusal(self, at: float) -> str | None:
        """Why the tracker's pose may not be painted into the model at ``at``, or ``None`` when
        it may: :class:`pepin.watch.PaintTrust` over the fit this node last HEARD, the tracker's
        own sigma where it publishes one, and the age of the ``map -> odom`` edge the pose is
        built on. Both paint paths ask it — a revolution places a wall exactly as a camera frame
        does, and until 2026-09-15 only the camera was asked."""
        now = time.monotonic()
        # A sigma nobody has refreshed is no sigma: the topic can die while its last small
        # number sits here, and a frozen number must not hold the gate open. The fit's own
        # freshness would catch the same silence — both come from the tracker — but a feed is
        # only evidence while it speaks, so it is dropped here rather than relied on there.
        fresh_sigma = self._sigma_xy_m if now - self._sigma_at <= SOURCE_PATIENCE_S else None
        return PaintTrust(max_sigma_xy_m=float(self._switches["paint_sigma_m"])).refusal(
            fit=self._fit,
            fit_age_s=now - self._fit_at,
            sigma_xy_m=fresh_sigma,
            edge_age_s=self._poser.map_correction_age_s(at),
        )

    def _withhold(self, kind: str, refusal: str) -> None:
        """Count one observation the pose was not good enough to paint with, keep the last
        reason for the report line, and tell the snapshot guard why: a run whose painting has
        stopped must not overwrite the last good map, and must be able to say what stopped it."""
        self._tally.count(kind)
        self._tally.note(kind, refusal)
        self._trust.withheld(refusal)

    def _law(self) -> LidarLaw:
        """How a beam writes into the volume right now: the defaults with the live flags in
        them, rebuilt per scan so a flag set mid-run takes effect on the next revolution."""
        return LidarLaw(no_return_free=self._switches.on("no_return_free"))

    def _depth_law(self) -> DepthLaw:
        """How a depth frame writes into the volume right now — what a pixel with NO depth may
        carve (:class:`pepin.tsdf.DepthLaw`) — rebuilt per frame so a flag set mid-run takes
        effect on the next one.

        The reach is the source's own: ``no_depth_reach_m`` when somebody has stated it, and
        otherwise the one measured off the frames (:class:`pepin.tsdf.ObservedReach`), never the
        publisher's looser ``depth_reach_m`` gate.
        """
        stated = float(self._switches["no_depth_reach_m"])
        return DepthLaw(
            no_depth_free=self._switches.on("no_depth_free"),
            no_depth_weight=float(self._switches["no_depth_weight"]),
            reach_m=stated if stated > 0.0 else self._reach.m,
        )

    def _carve_line(self) -> str:
        """The depthless-ray half of the report: how far a NaN pixel carves, what it weighs and
        where the reach came from — or that nothing carves, and why."""
        law = self._depth_law()
        stated = float(self._switches["no_depth_reach_m"])
        source = (
            f"stated {stated:.2f} m"
            if stated > 0.0
            else f"measured {self._reach.m:.2f} m over {self._reach.frames} frames"
        )
        if not law.no_depth_free:
            return f"carve: off, a pixel with no depth touches nothing (reach {source})"
        to = law.carve_to_m(self._spec.truncation_m)
        if to <= 0.0:
            return f"carve: on but idle — no reach yet ({source}), so nothing is carved"
        weight = law.no_depth_weight * float(self._spec.observation_weight(np.array(to)))
        return (
            f"carve: a pixel with no depth carves to {to:.2f} m (reach {source}) at weight"
            f" {weight:.2f}, {law.no_depth_weight:g} of a measurement there"
        )

    def _lookup_laser(self, frame: str) -> bool:
        """The static base_link <- laser transform: the beams' angle domain (the sensor hangs
        upside down, so its angles run clockwise) and the mount the rays start from. The height
        stays the calibrated one from config/lidar.json; the ranges are the scan's own."""
        transform = self._tf.transform("base_link", frame)
        if transform is None:
            return False  # not yet there: try on the next scan
        x, y, z, roll, _pitch, yaw = rpy_from_transform(transform)
        mirrored = abs(abs(roll) - math.pi) < 0.2
        self._laser = (self._mount, yaw, mirrored)
        self.get_logger().info(
            f"laser mount: x {x:.3f} y {y:.3f} z {z:.3f} yaw {math.degrees(yaw):.1f} deg,"
            f" {'upside down' if mirrored else 'upright'}; the layer sits at"
            f" {self._mount.z_m:.3f} m (config/lidar.json)"
        )
        return True

    # ---- the graph's correction ----------------------------------------------------------
    def _on_graph(self, msg: Any) -> None:
        """One ``/rtabmap/mapGraph``: how far the graph has bent the ROOM since the last one
        (:class:`pepin.graphbend.GraphBend`).

        Read here and not in the paint path because it arrives on its own cadence (once per
        processed snapshot, ``Rtabmap/DetectionRate``) and because the bend is a fact about the
        graph rather than about any one frame. ``poses_id`` and ``poses`` are parallel arrays of
        the OPTIMISED poses (rtabmap_msgs/msg/MapGraph.msg:11-12); ``map_to_odom`` in the same
        message is deliberately not read — it moves when the CART is found and not only the room.
        """
        if not self._up:
            return  # the node is still being built (see __init__)
        self._graphs += 1
        poses = {
            int(node): pose_from_pose_msg(pose)
            for node, pose in zip(msg.poses_id, msg.poses, strict=False)
        }
        with self._lock:
            bend = self._bend.observe(poses)
        if bend is not None:
            self._tally.count("bends")

    def _follow(self, stamp: Any) -> bool:
        """Carry the volume to where the graph now says the room is, before anything is painted
        into it at ``stamp``; returns whether that observation may go in at all.

        WHAT THE VOLUME IS BEHIND is the accumulated bend of the graph's own node poses
        (:meth:`_on_graph`), never the change in ``map -> odom``: the volume is painted at the
        BOARD TRACKER's pose, and that edge moves both when the room bends (the volume must follow)
        and when the cart is FOUND after drifting (the volume must not — following a recovery drags
        the painted room off the real one by the whole size of the recovery). The node poses are the
        only signal in the graph that separates the two, because a re-localisation leaves every node
        exactly where it was. The stamp is therefore not used to look anything up here; it is the
        moment the observation belongs to, and the bend in force is the bend in force.

        Below the thresholds nothing moves: the difference is owed against the same anchor and
        applied when it grows, and the observation goes in, at most a voxel out. Above them the
        whole content is carried rigidly (:meth:`pepin.worldmap.WorldMap.shift`) on the caller's
        worker thread — and if the rate (``follow_correction_min_s``) will not let that move
        through yet, the observation is REFUSED rather than painted. An observation placed under
        the new correction and painted into a volume still standing in the old one is carried
        past the truth by the whole of the move when it finally lands: measured on the synthetic
        box (scratch/follow_refute.py, 2026-09-14), a 30 cm closure left a freshly painted wall
        20 cm beyond where the graph puts it. Losing two seconds of frames after a burst of
        optimisations is the cheap half of that trade; the correction itself is never lost.

        The whole of it — what is owed, the rate and the move — happens under the model lock,
        because both workers come through here and two of them that read the same debt would
        pay it twice.

        IN ``odom`` THERE IS NOTHING TO FOLLOW. The graph optimises the room's expression in
        ``map``; a volume painted through ``odom -> base_link`` was never expressed in it, so an
        optimisation cannot make one of its voxels stale. The window follows the CART instead
        (:meth:`_roll_window`).
        """
        if self._odom_volume:
            return True
        if not self._switches.on("follow_correction"):
            return True
        if not self._graphs:
            self._tally.count("no_correction")  # no graph has ever arrived: nothing to follow
            return True
        correction = self._bend.drift
        with self._lock:
            if self._follower.painted_in is None:
                self._follower.anchor(correction)  # an empty volume is born in what is in force
                return True
            shift = self._follower.pending(
                correction,
                float(self._switches["follow_correction_min_m"]),
                float(self._switches["follow_correction_min_deg"]),
            )
            if shift is None:
                return True
            now = time.monotonic()
            if now - self._followed_at < float(self._switches["follow_correction_min_s"]):
                self._tally.count("follow_held")  # owed against the same anchor, paid at the next
                return False  # ...and nothing is painted into a volume that owes a move
            started = time.perf_counter()
            self._world.shift(shift, str(self._switches["follow_correction_law"]))
            self._follow_ms = (time.perf_counter() - started) * 1e3
            self._followed_at = now
            self._follower.moved(correction, shift)
        self._tally.count("follows")
        self.get_logger().info(
            f"the graph bent the room ({self._bend.text()}): the volume follows it by"
            f" {shift.text()} ({self._follow_ms:.0f} ms, {self._follower.applied} moves this run)"
        )
        return True

    def _snapshot(self) -> None:
        """Write the volume to ``world_path`` — the surface the next run wakes up on, beside the
        graph database whose frame it was painted in.

        NO PICTURE OF IT IS WRITTEN. This used to export a map_server pair beside the snapshot so
        the board could boot on it; the board drives on the graph's own grid instead, and a pgm in
        the loop is a second file claiming to be the map (on 2026-09-18 the first live export wrote
        itself over the seed's own pgm). ``WorldMap.export_pgm_yaml``
        remains, for an operator and for the offline instruments, and nothing in the running loop
        calls it.

        ONLY WHILE THE PAINTING IS TRUSTED. A snapshot replaces the last one, so a run that has
        stopped painting at a pose the gate vouches for must leave the last good volume exactly
        where it is: ``ros/maps/world_live.npz.mess-20260917`` is what the other rule produces —
        one false camera word, painted, saved, permanent. The question is asked of
        :class:`pepin.worldmap.SnapshotTrust`, which is satisfied by an observation actually
        going in (the gates are what make that mean something: with ``fit_gate`` and
        ``lidar_fit_gate`` on, a painted observation IS a trusted pose, and in SLAM mode, where
        the launch turns them off because no tracker speaks, it is whatever the graph says).

        The lock is held for the write (half a second for a grid of noise, less for a real one),
        so a snapshot costs the camera a frame or two once every ``snapshot_s``.

        NOTHING IS WRITTEN IN ``odom``. A snapshot exists to be resumed, and a rolling window
        painted in the odometry frame of one run cannot be resumed by another: that frame is born
        where the wheels were switched on, and the file would describe a room in coordinates the
        next run never had. The counter says it once a window rather than silently doing nothing.
        """
        if self._odom_volume:
            self._tally.count("snapshot_skipped")
            return
        refusal = self._trust.refusal(time.monotonic())
        if refusal is not None:
            self._tally.count("snapshot_refused")
            self._tally.note("snapshot_refused", refusal)
            self.get_logger().warning(
                f"{self._world_path}: NOT written — the painting is not trusted ({refusal});"
                " the last good snapshot stands",
                throttle_duration_sec=60,
            )
            return
        try:
            with self._lock:
                self._world.save(self._world_path)
        except OSError as exc:
            self.get_logger().warning(f"{self._world_path}: not written ({exc})")
            return
        self._snapshots.done(time.monotonic())
        self._tally.count("snapshots")

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
        if self._switches.on("fit_gate") and not self._odom_volume:
            # ...and in odom the gate has nothing to judge: the frame is placed by the odometry,
            # which is the very frame the volume is drawn in (the report line says so).
            refusal = self._paint_refusal(stamp_seconds(msg.header.stamp))
            if refusal is not None:
                self._withhold("low_fit", refusal)
                return
        stamp = msg.header.stamp
        at = stamp_seconds(stamp)
        poser = self._poser_now
        camera = poser.camera_in_map(at)
        base = poser.base_in_map(at) if camera is not None else None
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
        if not self._follow(stamp):
            return  # the model this frame would be seated on still owes the graph a move
        if self._switches.on("align") and not self._odom_volume:
            # ...and in odom the search is inert: it seats a frame against a model that has been
            # accumulating the room, and a rolling local window is not that model — on 2026-09-21
            # the aligner sat at its own +-4 deg bound refusing frames because the volume it was
            # matching held four seatings of the same wall.
            with self._tally.measure("align"):
                aligned = self._aligned(depth, intr, camera, base)
            if aligned is None:
                return  # AT_BOUND: counted, not integrated
            camera = aligned
        self._roll_window(base)
        self._reach.saw(depth)  # the source's own reach, measured off the frames themselves
        with self._tally.measure("integrate"), self._lock:
            touched = self._world.integrate_depth(
                depth, rgb, intr, camera, stamp=at, law=self._depth_law()
            )
            self._last_stamp = stamp
        self._tally.count("frames")
        self._tally.count("voxels", touched)
        self._publish_marks(base, stamp)  # the camera's own turn to move the marks
        now = time.monotonic()
        self._trust.painted(now)  # ...and the map on disk may be replaced by this one
        # The camera's own clock on the snapshot as well as the lidar's: a camera-only run (the
        # lidar muted, 2026-09-17) painted for an hour and never saved, because only the scan
        # path ever looked at the alarm.
        if self._snapshots.due(now) and self._switches["snapshot_s"] > 0.0:
            self._snapshot()

    def _aligned(
        self, depth: Any, intr: Intrinsics, camera: RigidPose, base: RigidPose
    ) -> RigidPose | None:
        """The camera pose turned about the cart by the yaw that seats the frame's lidar-height
        band on the model; the pose as given when the model cannot judge or the frame already
        fits; ``None`` when the best turn is the search's bound (the frame must not go in)."""
        points = backproject(depth, intr, stride=BAND_STRIDE, range_max=self._spec.range_max_m)
        in_map = points @ camera.rotation.T + camera.translation
        lo, hi = self._band_z_m
        band = in_map[(in_map[:, 2] >= lo) & (in_map[:, 2] <= hi)]
        if band.shape[0] < BAND_MIN_POINTS:
            self._refused(AlignReason.UNJUDGED)
            return camera
        pivot = (float(base.translation[0]), float(base.translation[1]))
        with self._lock:
            verdict = align_yaw(self._world.volume, band, pivot)
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
            self._world = self._fresh_world()
            self._last_stamp = None
            self._follower = CorrectionFollower()  # as after a reset: the new model is born in
            # the correction in force, and owes the graph nothing the old one owed
        self._bound_streak = 0
        self._tally.count("self_heals")
        self.get_logger().warning(
            f"{AT_BOUND_STREAK} frames in a row refused at the alignment bound: the model is"
            " emptied and re-seeds from the next frame (flag self_heal)"
        )

    def _refused(self, reason: AlignReason) -> None:
        self._tally.count("refused_" + reason.value)

    # ---- outputs -------------------------------------------------------------------------
    def _on_depth_scan(self, msg: LaserScan) -> None:
        """One fan from a single depth frame (``/depth_scan``). It is the costmap's CLEARING
        source and this node does not read it at all — except under ``marks_source`` frame,
        where it is relayed onto ``/depth_marks`` unchanged, which is exactly how the camera
        layer marked before the volume took the job over (CLAUDE.md rule 19)."""
        if not self._up:
            return  # the node is still being built (see __init__)
        self._tally.count("depth_scans")
        if str(self._switches["marks_source"]) != FRAME:
            return
        if not self._marks_due():
            return
        self._marks_pub.publish(msg)
        self._tally.count("marks")

    def _marks_due(self) -> bool:
        """Whether ``/depth_marks`` may go out now, and the cap's clock moved on when it may
        (``marks_hz``; 0 is every frame).

        One gate for both sources of the topic — the volume's slice and the relayed frame — so
        the rate on the wire is the flag's whatever ``marks_source`` says. Monotonic seconds:
        this is a cap on a publisher, not a measurement of anything in the room.
        """
        hz = float(self._switches["marks_hz"])
        if hz <= 0.0:
            return True
        now = time.monotonic()
        if now - self._marks_at < 1.0 / hz:
            self._tally.count("marks_thinned")
            return False
        self._marks_at = now
        return True

    def _marks_law(self) -> MarksLaw:
        """How the volume is read out as marks right now: the node's own ``min_weight`` — the
        one criterion for what this model calls a surface, shared with ``/fusion/surface`` — the
        band between ``marks_min_z`` and the volume's own camera band, and the fan's reach."""
        return MarksLaw(
            min_weight=float(self._switches["min_weight"]),
            band_m=(float(self._switches["marks_min_z"]), self._spec.camera_band_m[1]),
            range_m=MARKS_RANGE_M,
        )

    def _publish_marks(self, base: RigidPose, stamp: Any) -> None:
        """Publish what the volume holds around the cart as ``/depth_marks``: one range per half
        degree of the whole turn, in base_link, stamped with the observation that was just
        integrated — the board's clock, the pose that observation was placed by.

        Called from both paint paths, so the marks go out at the rate the volume is integrated —
        about 10 Hz of revolutions plus the camera's own 9-9.5 fps while the cart drives, the
        camera alone while it stands still (a revolution from a place already seen is not
        integrated at all, ``view_gate``), and nothing while the paint gates withhold. The
        neighbourhood is copied under the model lock and read outside it; both halves are timed
        into the report line's ``ms a slice``.

        A bearing with no surface in the band is NaN: ``/depth_marks`` never clears and never says
        a thing about free space. Under ``marks_clear`` the same walk also goes out on
        ``/depth_free`` — the range each bearing is KNOWN OPEN to — as a clearing-only source of
        the same layer; off (as shipped) that topic is silent and the clearing is ``/depth_scan``'s
        alone. The free walk is inside the same measured stage, so its cost shows in "ms a slice".

        ``marks_hz`` caps the RATE of this topic (5 Hz, the local costmap's own
        ``update_frequency``): the frame that is not published is still fused, and the slice it
        would have been read out as is not computed at all — the gate is asked before the
        crossing search, which is the whole cost of this method.
        """
        if str(self._switches["marks_source"]) != VOLUME:
            return  # the frame's own fan is being relayed instead, on arrival
        if not self._marks_due():
            return  # the costmap has not read the last fan yet (marks_hz)
        law = self._marks_law()
        with self._tally.measure("marks"):
            # The lock is held for the COPY of the neighbourhood and not for the reading of it:
            # a window is a fraction of a millisecond, the crossing search over it is
            # milliseconds, and this runs at the rate the volume is integrated — the other
            # worker must not queue behind it.
            with self._lock:
                window = marks_window(self._world.volume, base, law)
            clears = bool(self._switches["marks_clear"])
            if window is None:
                ranges = free = empty_marks(law)
            else:
                ranges = marks_ranges(window, base, law)
                free = free_ranges(window, base, law) if clears else empty_marks(law)
        self._marks_bearings, self._marks_clearing, self._marks_silent = fan_counts(ranges, free)
        reach = law.range_m + self._spec.voxel_m  # a consumer drops a range AT range_max
        self._marks_pub.publish(
            scan_from_ranges(
                ranges, MARKS_ANGLE_MIN, MARKS_STEP, stamp, BASE_FRAME, MARKS_MIN_RANGE_M, reach
            )
        )
        if clears:
            self._free_pub.publish(
                scan_from_ranges(
                    free, MARKS_ANGLE_MIN, MARKS_STEP, stamp, BASE_FRAME, MARKS_MIN_RANGE_M, reach
                )
            )
        self._tally.count("marks")

    def _publish_surface(self) -> None:
        """The model's surface as a cloud, in the frame the volume is painted in — ``odom`` for
        the rolling window, ``map`` for the room (``volume_frame``). The frame is the volume's own
        and never a fixed name: a window painted through the odometry and drawn in ``map`` would
        be shown wherever the last correction happened to put it."""
        with self._lock:  # a copy under the lock (milliseconds), the crossing search outside it
            snapshot = self._world.volume.snapshot()
            stamp = self._last_stamp
        points, colours = snapshot.surface(
            self._switches["min_weight"], colour_fallback=self._switches.on("colour_fallback")
        )
        self._surface_points = int(points.shape[0])  # a level the report reads, not a tally
        # the board's clock: the surface is as old as the last frame in it, not as new as now
        self._pub.publish(
            cloud_from_points(
                points,
                colours,
                stamp if stamp is not None else self.get_clock().now().to_msg(),
                self._volume_frame,
            )
        )

    def _report(self) -> None:
        self._read_plane(0.0)  # a board that came up after this node still moves the band
        w = self._tally.take()
        c = w.counts
        skipped = (
            f"pose not trusted {c['low_fit']}, at bound {c['refused_at_bound']},"
            f" self-heals {c['self_heals']}, unleaned {c['unleaned']},"
            f" leaned out {c['leaned_out']},"
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
            f" {self._marks_line(w)}; {self._frame_line(w)};"
            f" {self._band_text()}; {self._carve_line()}; {self._world_line(w)};"
            f" {self._follow_line(w)};"
            f" {self._lean.report()};"
            f" flags: {self._switches.state()}" + (f"; tf: {tf_text}" if tf_text else "")
        )

    def _frame_line(self, w: Window | None = None) -> str:
        """The frame half of the report: which frame the volume is painted in and what that
        decides — under ``odom`` where the rolling window stands, how far the cart may leave its
        centre, what the last slide cost and which paths are inert because of it; under ``map``
        the file it is saved to and the machinery that seats and carries it.

        The inert paths are NAMED rather than left silent: ``align=on`` in the flag state with the
        volume in ``odom`` would otherwise read as a search that is running.
        """
        if not self._odom_volume:
            return (
                f"frame: the volume is the room, in {MAP_FRAME} — snapshot {self._world_path} (the"
                f" frame of {self._database}), seated by align, gated by the tracker's fit and"
                f" sigma, carried by the graph's bend on {GRAPH_TOPIC}"
            )
        cx, cy = self._spec.centre_xy
        slides = f", {int(w.counts['recentres'])} slides this window" if w is not None else ""
        return (
            f"frame: the volume is local memory, in {ODOM_FRAME} — a rolling window centred on"
            f" ({cx:+.2f}, {cy:+.2f}), re-centred on the cart past {self._window_recentre_m:.1f} m"
            f" (last slide {self._recentre_ms:.0f} ms{slides}); the odometry pose IS the pose, so"
            " align, the paint gates (fit_gate, lidar_fit_gate, paint_sigma_m) and"
            " follow_correction are inert here and no snapshot is read or written"
        )

    def _marks_line(self, w: Window) -> str:
        """The costmap half of the report: where the camera's marks came from this window, how
        many went out and how fast, what one slice of the volume cost, how many bearings it
        filled and in which band — the numbers a drive is judged on without a debugger."""
        c = w.counts
        if str(self._switches["marks_source"]) == FRAME:
            return (
                f"marks: {MARKS_TOPIC} relayed from {DEPTH_SCAN_TOPIC}, {int(c['marks'])} of"
                f" {int(c['depth_scans'])} frames ({w.rate('marks'):.1f}/s){self._cap_text(w)} —"
                " the volume is not read (marks_source frame)"
            )
        law = self._marks_law()
        return (
            f"marks: {int(c['marks'])} from the volume ({w.rate('marks'):.1f}/s,"
            f" {w.ms_per('marks', 'marks'):.1f} ms a slice){self._cap_text(w)},"
            f" {self._marks_bearings} bearings of"
            f" {round(2 * math.pi / MARKS_STEP)} filled, band {law.band_m[0]:.2f}-"
            f"{law.band_m[1]:.2f} m within {law.range_m:.1f} m at min_weight {law.min_weight:g};"
            f" {self._clear_text()}"
        )

    def _clear_text(self) -> str:
        """The clearing half of the fan in the report line: how many bearings of the last slice
        marked, how many cleared and how many said nothing — the three numbers ``marks_clear`` is
        judged by, and the reason it ships off (a ray stops at the first column that is not open,
        and the volume's own marks are what stand in the way)."""
        if not bool(self._switches["marks_clear"]):
            return f"marks_clear off: {FREE_TOPIC} silent, the layer clears from the frame alone"
        return (
            f"clearing on {FREE_TOPIC}: {self._marks_bearings} bearings mark,"
            f" {self._marks_clearing} clear, {self._marks_silent} say nothing"
        )

    def _cap_text(self, w: Window) -> str:
        """What the ``marks_hz`` cap did this window, or nothing at all when it is off: how many
        fans it held back, so a rate below the camera's own is read as the cap and not as a node
        that has stopped slicing."""
        hz = float(self._switches["marks_hz"])
        if hz <= 0.0:
            return " (marks_hz 0: every frame)"
        return f" ({int(w.counts['marks_thinned'])} held by the marks_hz {hz:g} cap)"

    def _world_line(self, w: Window) -> str:
        """The volume half of the report: what the lidar wrote, what the two layers hold, how many
        revolutions were the same view again, and how old the snapshot is."""
        c = w.counts
        with self._lock:
            text = self._world.report()
        return (
            f"world: {c['revolutions']} revolutions ({c['scans_dropped']} dropped,"
            f" {w.ms_per('scan', 'revolutions'):.0f} ms), {self._withheld_line(w)}, {text};"
            f" {self._views.report()}; {self._snapshot_line(w)}"
        )

    def _snapshot_line(self, w: Window) -> str:
        """How safe the volume's file is: whether this one was resumed and how old it was, the
        snapshot's age — and, when the guard is refusing, that the file on disk is NOT this volume
        and why.

        A volume two weeks stale and one that stopped being saved an hour ago look identical from
        the outside.
        """
        if self._odom_volume:
            return (
                f"volume born this run; no snapshot at all in {ODOM_FRAME} — a window painted"
                " through the odometry cannot be resumed by another run"
            )
        age = self._snapshots.age_s(time.monotonic())
        resumed = (
            "born this run"
            if self._resumed_age_s == math.inf
            else f"resumed {self._resumed_age_s / 3600:.1f} h old"
        )
        refused = int(w.counts["snapshot_refused"])
        held = (
            f", NOT SAVED {refused}x ({w.notes.get('snapshot_refused', '')}): the last good file"
            " stands"
            if refused
            else ""
        )
        return (
            f"volume {resumed}; snapshot"
            f" {'never' if age == math.inf else f'{age:.0f} s old'} at {self._world_path}{held}"
        )

    def _withheld_line(self, w: Window) -> str:
        """What the pose gate kept out of the volume this period: how many revolutions were
        withheld, the last reason, and the sigma the tracker is publishing — or that nobody is."""
        if self._odom_volume:
            return (
                f"lidar revolutions withheld: no gate in {ODOM_FRAME} (the odometry pose is the"
                " pose; there is no tracker word in this frame to be wrong)"
            )
        if not self._switches.on("lidar_fit_gate"):
            return "lidar revolutions withheld: gate off (every revolution is painted)"
        reason = w.notes.get("untrusted", "")
        told = self._sigma_xy_m
        sigma = f"sigma {told:.2f} m" if told is not None else f"no {SIGMA_TOPIC}"
        return (
            f"lidar revolutions withheld: {int(w.counts['untrusted'])}"
            + (f" (pose not trusted: {reason})" if reason else " (pose not trusted)")
            + f", {sigma}"
        )

    def _follow_line(self, w: Window) -> str:
        """The graph half of the report: how far the graph has bent the room, how many times the
        volume has moved with it, the last move and what the resample cost."""
        if self._odom_volume:
            return (
                f"follow: inert in {ODOM_FRAME} (the graph bends the room's expression in"
                f" {MAP_FRAME}, and no voxel of this window is expressed in it); graph"
                f" {self._bend.text()}"
            )
        if not self._switches.on("follow_correction"):
            return f"follow: off (the graph bends, the voxels stay); graph {self._bend.text()}"
        if not self._graphs:
            return f"follow: nothing on {GRAPH_TOPIC} yet — no graph, nothing to follow"
        held = int(w.counts["follow_held"])
        anchored = "anchored" if self._follower.painted_in is not None else "no bend yet"
        return (
            f"follow: {self._follower.applied} moves ({anchored}), last"
            f" {self._follower.last.text()} in {self._follow_ms:.0f} ms,"
            f" {int(w.counts['follows'])} this window, {held} observations refused while a"
            f" move was owed; graph {self._bend.text()}, {self._graphs} graphs,"
            f" {int(w.counts['bends'])} bends this window"
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
