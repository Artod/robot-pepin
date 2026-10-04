"""The cart's local obstacle memory: depth frames, lidar revolutions and ToF fans fused into a
TSDF on the laptop.

RTAB-Map assembles its cloud by concatenating one cloud per node, so two frames of one wall that
disagree by a few centimetres are two walls. This node fuses the same frames (``/camera/depth``
paired with its ``/camera/image`` by stamp — the depth carries the image's header — and placed
by the pose TF gives at that stamp, through the :class:`pepin.frame_pose.FramePoser` the depth
nodes share, over the kit's :class:`TfHistory`) into ``pepin.tsdf``: one signed distance per
voxel, updated by a distance-weighted average, so the wall is one surface, sharpened by near
observations and never blurred back by far ones. ``/fusion/surface`` (PointCloud2 in ``odom``,
colours from the camera, stamped with the last fused frame, the board's clock) is the
zero-crossing of the field.

THE VOLUME LIVES IN ``odom`` (2026-09-22). Its job is LOCAL OBSTACLE MEMORY — the nvblox local
mapper beside a pose graph, the pattern STVL follows — and local memory must not depend on global
localisation at all. Painted in ``map`` and kept across a day it did the opposite: on 2026-09-21
the volume held the walls of some twenty re-seatings of the tracker at once and the slice put
300-650 lethal cells around the cart that the lidar had never seen (scratch/nav2_hang/
layer_blame.py). So every frame and every revolution is placed by ``odom -> base_link`` and the
camera's own edge, never through ``map -> odom``; no snapshot is read or written; and the volume
is a rolling window — past ``window_recentre_m`` (config/fusion.json) from the window's centre the
box slides onto the cart and what leaves it is forgotten (:class:`pepin.tsdf.WindowShift`). The
room-sized map-frame model (its snapshot, the yaw seating, the tracker's paint gates, the graph's
bend) is on the tag alt/volume-map-2026-10-02.

THE LIDAR'S LAYER (:mod:`pepin.worldmap`). ``/scan`` is integrated into the same volume along the
beams' own rays — carving free space, a surface at each return, on the lidar's own weight channel,
which the camera's scale-uncertain depth may not repaint. The rays follow the body: with
``imu_lean`` on the scan is placed by the leaning pose, and a revolution taken past
``lean_gate_deg`` is dropped rather than believed. A lean gravity did not vote for
(``lean_min_quality``: a drifting gyro's own signature) is no lean at all. A beam that came back
with nothing writes nothing: a mirror, a black chair leg and an open door say the same nothing.

A VIEW IS EVIDENCE ONCE (:class:`pepin.worldmap.ViewGate`): a parked cart sends the same
revolution ten times a second, and the volume's weights used to count every one as an independent
observation — a weight that counts one view a thousand times calls a single glance a wall the room
agrees on. A revolution counts as a new view once no return of it stays in the voxel it was in.

THE VOLUME IS OPEN-LOOP: nothing localises against it. No slice of it goes out as a map, no
matcher reads it, no pose is estimated on it (a tracker matching the slice it painted walked a
parked cart 7 degrees in 35 minutes, 2026-09-18). What a slice of it DOES do is mark the costmap.

THE COSTMAP'S CAMERA MARKS COME FROM THE VOLUME (``/depth_marks``, 2026-09-21). A single stereo
frame is not evidence that something is THERE: SGBM on a herringbone parquet answers small blobs
of disparity 2-5 px too large, which lift floor pixels into the band the fan marks in, and the
first stereo drive left the costmap with 100-300 lethal cells the lidar never saw (tape
ros/maps/rec/0415_*). The same frames fused into this volume look clean, because a weighted
average and the free space later rays carve are exactly what one wrong opinion does not survive.
So this node slices the volume's own surface — the same surface ``/fusion/surface`` draws, at the
same ``min_weight`` — around the cart into a LaserScan over the whole turn in ``base_link``
(:mod:`pepin.volume_scan`), stamped with the observation that was just integrated. The costmap's
camera layer MARKS from it and CLEARS from ``/depth_scan`` (pepin_bringup.depth_stream): the frame
is the eyewitness of what is open now, the model is what remembers what is there. The topic goes
out at ``marks_hz`` (5 Hz), the board's local costmap's own ``update_frequency``; a frame held back
by the cap is still fused into the volume.

A PIXEL WITH NO DEPTH IS ALSO A MEASUREMENT (2026-09-22). Until then only a ray that MEASURED a
surface moved any voxel, and the stereo depth is NaN past the rig's own reach, so anything
standing in front of something farther than that was never carved: the operator's face stayed in
the volume after he walked away (scratch/one_localiser/black_voxels.py). So a depthless pixel
carves free space along its own ray out to the source's reach less a truncation, at
``no_depth_weight`` of what a measurement there weighs, because a NaN is also what a textureless
wall looks like. The reach is measured off the frames (:class:`pepin.tsdf.ObservedReach`) and
printed in the report line; the lidar's layer is protected from the carve as it is from the
camera's own marks. Measured: a saturated phantom 90 % gone in 2.8 s with the carve and never
without it, at 81 % of the camera band's cells kept (scratch/one_localiser/volume_ab.py).

The surface's colour: the lidar writes no colour (a beam has none to give), so a crossing whose
nearer voxel only the lidar painted takes the OTHER neighbour's camera colour rather than black
(:meth:`pepin.tsdf.Tsdf.surface`, ``colour_fallback``).

The flags and knobs (:data:`FLAGS` and config/knobs.json, ``ros/flags.sh set depth_fusion <flag>
<value>``): ``enabled``, ``tof_rays``, ``imu_lean``, ``lean_gate_deg``, ``lean_min_quality``,
``min_weight``, ``marks_min_z``, ``marks_hz``, ``marks_clear``, ``grid_out``, ``grid_hz``,
``grid_size_m``, ``grid_resolution_m``, ``surface_hz``, ``band_half_z``, ``lidar_layer``,
``no_depth_weight``, ``no_depth_reach_m``, ``self_filter``, ``arm_filter``; their state is printed
in every report line.

THE CART'S OWN BODY (``self_filter``, :mod:`pepin.body`, config/body.json): a head that looks back
or down to a side sees the cart's top shelf, wheels and mast, and would paint them into the volume
— a wall where the robot stands. Under the flag every camera frame is clipped per ray at its entry
into the body's boxes: what lies on or past it is not written, measured or carved. The frames of a
head in motion never arrive here at all: depth_stream's gaze gate keeps them from the network.

THE ROBOT'S OWN ARM (``arm_filter``, :mod:`pepin.arm`, config/arm.json): the SO-101 at the cart's
front is in the picture whenever the head looks down ahead, and painted it is an obstacle at the
bumper the stall look then stares at. Its links are boxes posed by its joints (the file's parked
pose, or ``/arm/joint_states`` at each observation's stamp) through the vendored URDF; under the
flag the camera's frames and the whiskers' fans are clipped at them as at the body, and after
every integration — camera, revolution or fan — every voxel inside a grown link is forgotten, so
no return off the arm and no surface painted before the arm moved in stays in the volume, the
marks or a ``/fusion/column`` answer. ``/fusion/arm`` (visualization_msgs/MarkerArray, base_link,
2 Hz) draws the grown links whether or not the flag is on.
``/fusion/reset`` (std_srvs/Trigger) empties the model, the pairing queues and the tallies.
For the gaze arbiter: ``/fusion/frame`` (std_msgs/Header) is every fused camera frame at its own
stamp, and ``/fusion/column`` (map_msgs/GetPointMapROI) the surface points of a box of the
volume with their weights.

THE CAMERA GRIDS (``grid_out``, off as shipped, 2026-09-24). ``/depth_marks`` is accumulated by
each costmap's camera_layer into a grid of its own — two more copies of this memory, the global
one smeared by every ``map -> odom`` jump and both wiped by the tree's clears
(scratch/costmap_split). Under ``grid_out`` the same columns go out as grids the costmaps only
DRAW (a StaticLayer each, ``camera_grid_layer``): ``/camera_grid`` about the cart in ``odom``
and ``/camera_grid_map`` (+ ``_updates``) on the lattice of ``grid_map_topic``
(:mod:`pepin.camera_grid`); ros/camera_grid.sh flips the node and the layers together.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from map_msgs.msg import OccupancyGridUpdate
from map_msgs.srv import GetPointMapROI
from message_filters import Subscriber, TimeSynchronizer
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan, PointCloud2
from std_msgs.msg import ColorRGBA, Header
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from pepin.arm import ARM_FILE, ARM_TOPIC, ArmMask, ArmModel, ArmPose, JointHistory, arm_pose
from pepin.body import BODY_FILE, BodyMask, BodyModel, OrientedBox, RayDepth, oriented_ray_depth
from pepin.camera_grid import (
    OCCUPIED,
    GridWindow,
    MapCanvas,
    MapGeometry,
    grid_volume,
    to_map_xy,
)
from pepin.depth import Intrinsics, quaternion_from_matrix
from pepin.flags import Flag, FlagSet, load_knobs, with_knobs
from pepin.frame_pose import BASE_FRAME, MAP_FRAME, ODOM_FRAME, FramePoser
from pepin.lean import LeanGate
from pepin.live_settings import LiveFile
from pepin.mounts import LASER_FRAME, load_lidar_mount
from pepin.tof_rays import TOF_RATE_HZ, TOF_WEIGHT, fan_image, fans_to_clear, optical_pose, tof_law
from pepin.tsdf import (
    DepthLaw,
    GridSpec,
    ObservedReach,
    RayClip,
    RigidPose,
    band_half_z_m,
    window_recentre_m,
)
from pepin.volume_scan import (
    MARKS_ANGLE_MIN,
    MARKS_MIN_RANGE_M,
    MARKS_RANGE_M,
    MARKS_STEP,
    Column,
    MarksLaw,
    band_surface,
    column_points,
    column_window,
    empty_marks,
    fan_counts,
    free_ranges,
    marks_ranges,
    marks_window,
)
from pepin.worldmap import (
    TOF,
    PlanarMount,
    ViewGate,
    WorldMap,
    bearings_in_base,
)
from pepin_bringup.msgs import (
    array_from_image,
    cloud_from_fields,
    cloud_from_points,
    grid_update,
    map_geometry,
    occupancy_grid,
    rpy_from_transform,
    scan_arrays,
    scan_from_ranges,
    stamp_from_seconds,
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
SCAN_TOPIC = "/scan"
# THE CAMERA'S TWO WORDS TO THE COSTMAP, and which of them is evidence of what. ``/depth_scan``
# is one frame folded onto the plane (pepin_bringup.depth_stream): an eyewitness of what is OPEN
# right now — it clears. ``/depth_marks`` is this node's answer to what is THERE: the accumulated
# volume's own surface, sliced around the cart (pepin.volume_scan), and it only marks. The two
# names are one layer in ros/params/nav2_params.yaml.
MARKS_TOPIC = "/depth_marks"
# ...and, behind ``marks_clear``, the same fan's third word: how far each bearing is KNOWN OPEN
# (pepin.volume_scan.free_ranges). A separate topic and a CLEARING-ONLY source in the same layer,
# because a LaserScan cannot carry "clear to here" and "nothing here" on one range: Nav2's
# ObstacleLayer marks at the end of every finite range it is given, so one source that cleared a
# ray at 1.2 m would plant a lethal cell at 1.2 m — at the frontier of knowledge.
FREE_TOPIC = "/depth_free"
# THE GAZE ARBITER'S TWO QUESTIONS (pepin_bringup.gaze). Which camera frames went into the volume,
# each as a header at its own stamp, published as it is fused: a look waits for still frames
# stamped after the head settled, and only the fused ones carve or confirm anything. And what the
# volume holds in a box (map_msgs/GetPointMapROI, the box in the volume's frame): the surface
# points with their voxel weights, asked before and after a stall look over the blocking cells.
FRAME_TOPIC = "/fusion/frame"
COLUMN_SERVICE = "/fusion/column"
# THE CAMERA GRIDS (``grid_out``, 2026-09-24): the same surface as grids that the costmaps'
# camera_grid_layer (a StaticLayer, ros/params/nav2_params.yaml) only DRAWS, so Nav2 keeps no copy
# of the memory of its own (pepin.camera_grid). /camera_grid is a square about the cart in the
# volume's frame, for the rolling local costmap; /camera_grid_map is the same cells on the lattice
# of the map the global costmap's static layer reads (``grid_map_topic``), with Nav2's own
# ``<map_topic>_updates`` beside it.
GRID_TOPIC = "/camera_grid"
GRID_MAP_TOPIC = "/camera_grid_map"
GRID_UPDATES_TOPIC = GRID_MAP_TOPIC + "_updates"
MAP_TOPIC = "/map"  # the global static layer's map (RTAB-Map's grid)
GRID_TF_WAIT_S = 0.05  # map <- odom at the grid's stamp: a paint worker waits no longer
GEOMETRY_SETTLE_S = 1.0  # a new full grid reaches the layer before any update in its geometry
TF_WAIT_S = 0.3
BAND_TF_WAIT_S = 5.0  # the static base_link -> laser edge at start: the board publishes it once
PAIR_QUEUE = 40  # depth arrives a fraction of a second after its image; pair by exact stamp
STAGES = ("integrate", "body", "arm", "forget", "scan", "tof", "marks", "grid", "grid_map")
# THE ARM'S BOXES FOR THE EYE (pepin.arm): the grown links in base_link, teal, whether or not
# arm_filter is on — the arm's own points standing inside them in Foxglove is the check of the
# mount and the joints before the flag is.
ARM_MARKERS_TOPIC = "/fusion/arm"
ARM_MARKERS_HZ = 2.0
ARM_RGBA = (0.0, 0.6, 0.6, 0.35)
# The three ToF fans, in their own frames (pepin_bringup.tof_bridge publishes both; the frame
# names are pepin.mounts.TOF_FRAME, the topic its _SCAN_TOPIC under the root namespace).
TOF_NAMES = ("front", "left", "right")
TOF_SCAN_TOPIC = "/tof/{name}/scan"

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
        "tof_rays",
        True,
        description="the three ToF fans (/tof/<name>/scan) are written into the volume as rays"
        " from their own frames, through the camera's integrator: a return marks a surface at its"
        " range and carves the ray free up to it, +inf carves free out to the fan's trusted range"
        " (its range_max, pepin.tof_horizon.trusted_max_range), every fan at weight"
        " pepin.tof_rays.TOF_WEIGHT. Both costmaps then read what the whiskers saw out of"
        " the volume (/depth_marks, /camera_grid) like the camera's; off, the fans touch the"
        " volume at all and feed only the local costmap's own layers, as before 2026-10-01",
        why="a pillow under the cart's nose is below the lidar's plane and too near for the"
        " stereo rig, so only the ToF saw it, only the local costmap's reflex layers knew, and the"
        " global planner kept routing through it (2026-10-01). The volume is the one memory of"
        " what is where and both costmaps already read it, so the whiskers write it too. At"
        " weight 1.0 a hit speaks in the marks after 4 fans (the node's min_weight 4.0, 0.27 s at"
        " 15 Hz) and a saturated one (max_weight 20) is carved by 15 misses, 1.0 s"
        " (pepin.tof_rays). Checked live 2026-10-01 with the camera's frames off: the front"
        " whisker at 0.75 m put its marks at 0.75-0.78 m inside its cone, none under the cart; a"
        " hand at 0.15 m marks (the ToF's own near limit, 0.08 m)",
        on_when="on every drive: the planner must know what the whiskers saw",
        off_when="a sensor reading its own surroundings (a cable hanging in its cone) paints"
        " phantoms into the volume that the planner cannot pass; the local costmap's own layers"
        " still get the fans either way",
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
        "grid_out",
        False,
        description="publish the volume's current occupied columns (the /depth_marks rule) as"
        " grids the costmaps' camera_grid_layer only draws: /camera_grid, a square about the cart"
        " in the volume's frame, and /camera_grid_map with its _updates on the lattice of"
        " grid_map_topic; off, all three are silent",
        why="off until a drive has measured it: the nvblox pattern against camera_layer's own"
        " copies of the memory, smeared by map -> odom jumps and wiped by the tree's clears"
        " (journal 2026-09-24, scratch/costmap_split)",
        on_when="with camera_grid_layer on and camera_layer off in both costmaps:"
        " ros/camera_grid.sh on",
        off_when="ros/camera_grid.sh off, back to /depth_marks alone; turning it off publishes"
        " one empty grid on each topic so a layer left on holds nothing stale",
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
        "self_filter",
        False,
        description="the cart's own body (config/body.json: boxes in base_link grown by margin_m)"
        " is cut out of every camera frame: a pixel whose depth lies on or past its ray's entry"
        " into the body measures no room, and no voxel on or past that entry is written, measured"
        " or carved (pepin.body, pepin.tsdf.Tsdf.integrate's clip); off, every ray is written"
        " whole, as before",
        why="OFF until the body is taped: the boxes are config/base.json's measured footprint"
        " and the IKEA RASKOG catalogue (top 0.78 m), the mast's section and top are guesses."
        " What is measured is the mechanism (tests/unit/test_body.py: a frame looking down at the"
        " own shelf paints 0 voxels inside the body and carves nothing behind it, a frame that"
        " does not see the body is written bit for bit as without the filter) and its cost"
        " (scratch/gaze_vision/cost.py, the live 280x250x34 grid, the 800x600 eye): nothing at"
        " the working tilt for any pan within +-95 deg (no ray meets the body; the cached mask"
        " says so in 3-5 us), +1.6-3.0 ms of integration (on 19-24 ms, two runs) on a"
        " reverse-gaze frame whose rays meet the body for 24 %, and 4.5-6 ms to rebuild the ray"
        " grid once the head has moved",
        on_when="before the head looks back or down to a side (reverse-gaze, side looks at tilt"
        " 45-63 deg), after the boxes are taped; the check: the head at the tilt limit at pan 0,"
        " +-90 and 180, 20 frames each, no voxel born inside the boxes",
        off_when="if the report line shows rays meeting the body at the working pose (a box too"
        " large), or an obstacle beside the cart disappears from the volume",
    ),
    Flag(
        "arm_filter",
        False,
        description="the robot's own arm (config/arm.json: the SO-101's links as boxes posed by"
        " its joints through the vendored URDF, grown by margin_m) is cut out of every camera"
        " frame and whisker fan as the body is, and every voxel inside a grown link is forgotten"
        " after each integration, whoever painted it (pepin.arm, pepin.worldmap.WorldMap.forget);"
        " off, the arm is painted like the room, as before",
        why="OFF until the mount is taped: config/arm.json's mount is a placeholder, the joints"
        " are the parked pose read from the encoders (source config: nothing drives the arm"
        " yet). What is measured is the mechanism (tests/unit/test_arm.py: frames looking at the"
        " arm in four poses paint 0 voxels inside it while the floor beside it is painted, a"
        " surface painted before the arm moved in is forgotten, the column a stall look asks"
        " holds no arm) and its cost (scratch/arm_mask/cost.py, the live 280x250x34 grid, the"
        " 800x600 eye, two runs): with the arm in 20-48 % of the rays the ray grid is 2.1-5.4 ms"
        " to rebuild (paid on every frame whose head or arm moved: 0.2 deg / 3 mm, as the"
        " body's), 4-5 us cached, and the clip adds 0.8-2.4 ms to a 15-19 ms integration; the"
        " parked arm out of the working look's picture costs 0.23 ms and nothing to integrate;"
        " the forget is 0.13-0.20 ms an integration (camera, revolution or fan), FK 0.1 ms when"
        " the joints move",
        on_when="after the mount is taped and the boxes sit on the arm's own points in Foxglove"
        " (/fusion/arm over /fusion/surface with the flag off): from then on whenever the arm is"
        " on the cart",
        off_when="if an obstacle in front of the cart disappears from the volume where the arm"
        " is not (a wrong mount or joint sign), or the report line counts frames on a stale pose",
    ),
)


def band_z_m(plane_z_m: float, half_m: float) -> tuple[float, float]:
    """The height band whose points are exact by construction, metres above the floor:
    ``plane_z_m`` (the lidar's own plane) plus and minus ``half_m``.

    The depth image is anchored on the beams, so this is the layer a frame may be turned by. It
    holds only while the band's centre is the plane the beams actually come from, which is why
    the caller reads that plane from TF (:meth:`DepthFusion._plane_z_m`) rather than from the
    config this process happens to have."""
    return (plane_z_m - half_m, plane_z_m + half_m)


def _on_cart(frame: RigidPose, base: RigidPose) -> RigidPose:
    """``base_link <- frame`` from the two poses in the volume's frame at one stamp."""
    rb = base.rotation.T
    return RigidPose(rb @ frame.rotation, rb @ (frame.translation - base.translation))


def arm_markers(boxes: tuple[OrientedBox, ...], stamp: Any, frame_id: str) -> Any:
    """The boxes as one MarkerArray of CUBEs in ``frame_id`` (namespace ``arm``), the old ones
    deleted first so a box that is gone does not linger."""
    out = MarkerArray()
    clear = Marker()
    clear.header = Header(stamp=stamp, frame_id=frame_id)
    clear.ns, clear.action = "arm", Marker.DELETEALL
    out.markers.append(clear)
    for i, box in enumerate(boxes):
        marker = Marker()
        marker.header = Header(stamp=stamp, frame_id=frame_id)
        marker.ns, marker.id, marker.type, marker.action = "arm", i, Marker.CUBE, Marker.ADD
        p = marker.pose.position
        p.x, p.y, p.z = (float(v) for v in box.centre)
        q = marker.pose.orientation
        q.x, q.y, q.z, q.w = (float(v) for v in quaternion_from_matrix(box.rotation))
        marker.scale.x, marker.scale.y, marker.scale.z = (float(2.0 * v) for v in box.half)
        r, g, b, a = ARM_RGBA
        marker.color = ColorRGBA(r=r, g=g, b=b, a=a)
        out.markers.append(marker)
    return out


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
        # The band's centre is the lidar's plane, and the only copy of it both sides of the
        # bridge agree on is the published base_link -> laser edge: this process reads
        # config/lidar.json from the laptop's checkout while the beams are published from the
        # board's own synced copy, and between a mount change and ros/sync.sh the two differ by
        # the whole change. TF is asked at start (and again while it has not answered); the
        # mount here is only the fallback, and the report line says which one the band sits on.
        self._plane_z_m = load_lidar_mount().z_m
        self._plane_source = "config"
        self._band_z_m = band_z_m(self._plane_z_m, band_half_z_m())
        # NOTHING LOCALISES AGAINST THIS VOLUME, so it publishes no map and claims to be no room.
        # config/fusion.json's box describes a volume being BORN, and a newborn is centred on the
        # cart (:meth:`_start_state`).
        self._spec = self._spec.centred_on_start()
        # The lidar's plane is calibrated, never typed: it comes from config/lidar.json, the one
        # file the board's launch publishes the laser transform from.
        self._mount = PlanarMount.from_config(
            str(self.declare_parameter("lidar_config", LIDAR_CONFIG).value)
        )
        # The map whose lattice /camera_grid_map copies: the one the global costmap's static layer
        # reads (/map; the launch says).
        self._grid_map_topic = str(self.declare_parameter("grid_map_topic", MAP_TOPIC).value)
        self._switches = Switches(
            self, with_knobs(FLAGS, load_knobs("depth_fusion")), on_change=self._on_switch
        )
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
        # ...and, under grid_out, the same surface as grids the costmaps only draw (GRID_TOPIC):
        # latched like any map, so a layer that starts later gets the last one. The updates are
        # volatile: Nav2 subscribes to them with the system default QoS.
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._grid_pub = self.create_publisher(OccupancyGrid, GRID_TOPIC, latched)
        self._grid_map_pub = self.create_publisher(OccupancyGrid, GRID_MAP_TOPIC, latched)
        self._grid_update_pub = self.create_publisher(
            OccupancyGridUpdate, GRID_UPDATES_TOPIC, reliable
        )
        self._grid_lock = threading.Lock()  # the grid clock, the window, the canvas, the geometry
        self._grid_at = 0.0  # monotonic seconds of the last published grid: the grid_hz cap
        self._grid_window: GridWindow | None = None  # the square /camera_grid last carried
        self._grid_cells = 0  # occupied cells in that grid, a level the report line reads
        self._geometry: MapGeometry | None = None  # grid_map_topic's lattice, once heard
        self._canvas: MapCanvas | None = None  # the map grid being drawn while grid_out is on
        self._canvas_at = -math.inf  # when its full grid went out (GEOMETRY_SETTLE_S)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, reliable)
        self.create_service(Trigger, "/fusion/reset", self._on_reset)
        self._frame_pub = self.create_publisher(Header, FRAME_TOPIC, reliable)
        self.create_service(GetPointMapROI, COLUMN_SERVICE, self._on_column)
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
        # The poses are the ODOMETRY's: the frame the volume is painted in is ``odom``, and nothing
        # in the paint path touches map -> odom at all, which is the whole point.
        self._poser = FramePoser(
            TfHistory(self._tf, timeout_s=TF_WAIT_S),
            map_frame=ODOM_FRAME,
            lean=self._lean,
            apply_lean=self._switches.on("imu_lean"),
            min_lean_quality=float(self._switches["lean_min_quality"]),
        )
        # How far the cart may leave the window's centre before the box slides onto it, from the
        # node's own config (pepin.tsdf.window_recentre_m).
        self._window_recentre_m = window_recentre_m(config)
        # The cart's own body (self_filter): config/body.json beside this node's config, read
        # again whenever the file changes, so a taped correction reaches the next frame.
        self._body_file = LiveFile(config.parent / BODY_FILE)
        self._body_data: dict[str, Any] = {}
        self._body_note = ""  # what the body is, for the log and the report line
        self._body = BodyMask(BodyModel(()))
        # The robot's own arm (arm_filter): config/arm.json beside this node's config, read again
        # whenever it changes; its joints from the file or from the topic it names (the topic is
        # subscribed at start: a new name in the file takes a restart).
        self._arm_file = LiveFile(config.parent / ARM_FILE)
        self._arm_lock = threading.Lock()  # the file, the model and the mask's caches
        self._arm_data: dict[str, Any] = {}
        self._arm_note = ""  # what the arm is, for the log and the report line
        self._arm = ArmMask(None)
        self._joints = JointHistory()
        self._read_arm()
        model = self._arm.model
        self._arm_topic = model.topic if model is not None else ARM_TOPIC
        self.create_subscription(JointState, self._arm_topic, self._on_joints, reliable)
        self._arm_pub = self.create_publisher(MarkerArray, ARM_MARKERS_TOPIC, reliable)
        self.create_timer(1.0 / ARM_MARKERS_HZ, self._publish_arm)
        self._recentre_ms = 0.0  # the last slide's cost, a level the report line reads
        self._read_plane(BAND_TF_WAIT_S)
        # The scan's own answer to the lean: a frame can be placed leaning, a revolution taken
        # too far from level can only be dropped (pepin.lean.LeanGate).
        self._gate = LeanGate(float(self._switches["lean_gate_deg"]))
        self._intr: Intrinsics | None = None
        self._lock = threading.Lock()  # the model and its last stamp, worker vs publisher
        self._world = WorldMap(self._spec, self._mount)
        self._last_stamp: Any = None  # the last fused frame's header stamp, the board's clock
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
        # The ToF fans have their own worker too: a fan is sub-millisecond to integrate but
        # waits for the lock a camera frame holds, and the executor thread must not. Newest
        # first, as for the scans: a fan still waiting is dropped and counted.
        self._tofs = Worker(self._on_tof_work, name="tof", on_error=self._on_work_error).start()
        for name in TOF_NAMES:
            self.create_subscription(
                LaserScan,
                TOF_SCAN_TOPIC.format(name=name),
                self._on_tof,
                QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE),
            )
        # Whether a revolution is a NEW view at all (pepin.worldmap.ViewGate): a view is evidence
        # once, and the threshold is the grid's own voxel.
        self._views = ViewGate(self._spec.voxel_m)
        self._start_state()
        self._surface_timer = self.create_timer(
            self._period(self._switches["surface_hz"]), self._publish_surface
        )
        self.create_timer(30.0, self._report)
        # Last, because a latched map is delivered the moment this exists and its callback needs
        # the grid state above (a message dropped before _up would never be sent again).
        self.create_subscription(OccupancyGrid, self._grid_map_topic, self._on_map, latched)
        nx, ny, nz = self._spec.shape
        self.get_logger().info(
            f"fusion up: {nx}x{ny}x{nz} voxels of {self._spec.voxel_m * 100:.0f} cm from"
            f" {self._spec.origin}; {self._switches.state()}; {self._band_text()}; nothing"
            f" localises against this volume — it is painted open-loop and published as"
            f" /fusion/surface (frame {ODOM_FRAME}) and as the costmap's camera marks on"
            f" {MARKS_TOPIC}; {self._frame_line()}"
        )
        self._up = True

    def _start_state(self) -> None:
        """What the volume starts as: empty, born under the cart, now. A NEWBORN IS CENTRED ON THE
        CART: a box laid out around the frame's origin would not contain a cart standing elsewhere
        in it, and a volume that does not contain the robot integrates the far wall and nothing
        else (2026-09-13: 239 revolutions). Nothing is read from disk: the odometry frame of this
        run is born wherever the wheels were switched on, and no file describes a room in it."""
        here = self._tf.transform(ODOM_FRAME, BASE_FRAME, timeout_s=TF_WAIT_S)
        if here is not None:
            at = (float(here.transform.translation.x), float(here.transform.translation.y))
            self._spec = self._spec.centred_on(at)
        self._world = WorldMap(self._spec, self._mount)
        self.get_logger().info(
            f"the volume is born empty under the cart in {ODOM_FRAME} on {self._spec.shape}"
            f" voxels from {self._spec.origin}"
        )

    def close(self) -> None:
        """Stop the workers and the TF listener, and wait for them all, before the node is
        destroyed."""
        if not self._worker.stop():
            self.get_logger().warning("the fusion worker did not finish its frame; leaving anyway")
        self._scans.stop()
        self._tofs.stop()
        # Best effort, and never at the snapshot's expense: after a SIGINT rclpy has already shut
        # the context and nothing can be published, so a stopped fusion leaves its last grid in
        # a layer that is still on (ros/README.md, "Camera grid A/B"); on a loud exit with the
        # context up the layers get empty grids.
        try:
            self._clear_grids()
        except Exception as exc:  # rclpy's invalid-context error, whatever its class
            self.get_logger().info(f"the camera grids were not cleared at exit: {exc}")
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
        band, the surface rate retimes its timer, ``grid_out`` starts the map grid or clears both
        grids, and the rest are only read where they are used."""
        if name == "imu_lean":
            self._poser.apply_lean = bool(new)
            self._lean.use_gyro = bool(new)
            return
        if name == "band_half_z":
            self._set_band()
            return
        if name == "lean_gate_deg":
            self._gate.gate_deg = float(new)
            return
        if name == "lean_min_quality":
            self._poser.min_lean_quality = float(new)
            return
        if name == "grid_out":
            if new:
                with self._grid_lock:
                    if self._geometry is not None:
                        self._start_canvas()
            else:
                self._clear_grids()
            return
        if name != "surface_hz":
            return
        timer = self._surface_timer
        try:
            timer.timer_period_ns = int(self._period(float(new)) * 1e9)
        except (AttributeError, TypeError) as exc:  # an rclpy without a live period
            raise ValueError(f"{name} cannot change live: {exc}") from exc

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
        costs the surface cloud and the marks until the sensors have painted them again.
        """
        return WorldMap(self._spec, self._mount)

    def _on_reset(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        with self._lock:
            self._world = self._fresh_world()
            self._last_stamp = None
        self._worker.clear()
        self._scans.clear()
        self._tofs.clear()
        with self._sync.lock:
            for queue in self._sync.queues:
                queue.clear()
        self._tally.take()
        response.success = True
        response.message = "the volume is empty"
        self.get_logger().info(
            "fusion: model, pairing queues and tallies reset; the volume is empty"
        )
        return response

    def _on_column(self, request: Any, response: Any) -> Any:
        """``/fusion/column``: the surface points in an axis-aligned box of the volume (centre
        ``x, y, z`` and sides ``l_x, l_y, l_z`` in the volume's frame; ``r`` > 0 a cube of half
        side ``r``), read by the rule of the marks (``min_weight``), each with its voxel's weight
        and the lidar's weight there, stamped with the last fused frame. The box is copied under
        the model's lock and read outside it, as the marks are."""
        centre = np.array([request.x, request.y, request.z], dtype=float)
        sides = [request.l_x, request.l_y, request.l_z] if request.r <= 0.0 else [2 * request.r] * 3
        half = np.array(sides, dtype=float) / 2.0
        lo, hi = centre - half, centre + half
        with self._lock:
            window = column_window(self._world.volume, self._world.lidar_weight, lo, hi)
            stamp = self._last_stamp
        column = Column.empty()
        if window is not None:
            column = column_points(*window, lo, hi, float(self._switches["min_weight"]))
        points = column.points
        response.sub_map = cloud_from_fields(
            {
                "x": points[:, 0],
                "y": points[:, 1],
                "z": points[:, 2],
                "weight": column.weight,
                "lidar": column.lidar,
            },
            stamp if stamp is not None else stamp_from_seconds(0.0),
            ODOM_FRAME,
        )
        self._tally.count("columns")
        return response

    # ---- inputs --------------------------------------------------------------------------
    def _on_info(self, msg: CameraInfo) -> None:
        if not self._up:
            return  # the node is still being built (see __init__)
        self._intr = Intrinsics.from_camera_info(msg.k, msg.width, msg.height)

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
        writing — the beams are in another slice of the room — and the scan is dropped."""
        if not self._switches.on("lidar_layer"):
            return
        if self._laser is None and not self._lookup_laser(msg.header.frame_id):
            return
        assert self._laser is not None
        mount, yaw, mirrored = self._laser
        at = stamp_seconds(msg.header.stamp)
        base = self._poser.base_in_map(at)
        if base is None:
            return  # counted by the TF failure handler
        if not self._gate.admits(self._poser.lean_at(at)):
            self._tally.count("leaned_out")
            return
        angles, ranges = scan_arrays(msg)
        # A VIEW IS EVIDENCE ONCE. The reach is this scan's own farthest return, so what counts as
        # "moved" is what moves one of ITS returns into another voxel.
        reach = float(np.nanmax(ranges)) if np.isfinite(ranges).any() else 0.0
        # ...and this is the CART's heading, which is not the mount's ``yaw`` above: writing it
        # into that name turned every beam of the revolution by the cart's own heading a second
        # time, through bearings_in_base, on every scan the gate admitted.
        heading = math.atan2(float(base.rotation[1, 0]), float(base.rotation[0, 0]))
        if not self._views.admits(
            float(base.translation[0]), float(base.translation[1]), heading, reach
        ):
            self._tally.count("same_view")
            return
        self._roll_window(base)
        with self._tally.measure("scan"), self._lock:
            touched = self._world.integrate_scan(
                bearings_in_base(angles, yaw, mirrored), ranges, base, mount, stamp=at
            )
            self._forget_arm(base, at)  # a beam's return off the arm is not the room
        self._tally.count("revolutions")
        self._tally.count("scan_voxels", touched)
        # ...and the costmap hears what the volume holds now, at the pose this revolution was
        # painted by and on its own stamp.
        self._publish_marks(base, msg.header.stamp)
        self._publish_grid(base, msg.header.stamp)

    def _on_tof(self, msg: LaserScan) -> None:
        """A ToF fan: straight to its worker, newest first."""
        if not self._up:
            return
        self._tally.count("tof_in")
        if self._tofs.offer(msg):
            self._tally.count("tof_dropped")

    def _on_tof_work(self, msg: LaserScan) -> None:
        """One ToF fan into the volume at the pose TF gives its frame for its stamp, as a tiny
        depth image through the camera's integrator (pepin.tof_rays): a return marks and carves
        up to itself, +inf carves free out to the fan's trusted range, NaN says nothing.
        Dropped and counted when TF has no pose for the frame at that moment. The marks and grids
        go out afterwards as after any integration, so a parked cart whose lidar sees the same
        view still carries a new whisker mark to the costmaps."""
        if not self._switches.on("tof_rays"):
            self._tally.count("tof_off")
            return
        at = stamp_seconds(msg.header.stamp)
        sensor = self._poser.frame_in_map(msg.header.frame_id, at)
        base = self._poser.base_in_map(at) if sensor is not None else None
        if sensor is None or base is None:
            self._tally.count("tof_no_tf")  # the lookup's own handler counted the edge
            return
        ranges = np.asarray(msg.ranges, dtype=float)
        if np.isfinite(ranges).any():
            self._tally.count("tof_hits")
        elif np.isinf(ranges).any():
            self._tally.count("tof_misses")
        else:
            self._tally.count("tof_silent")
            return  # nothing to write: every beam says "I do not know"
        depth, intr = fan_image(ranges, msg.angle_min, msg.angle_increment)
        self._roll_window(base)
        optical = optical_pose(sensor)
        clip = self._fan_clip(intr, optical, base, at)
        with self._tally.measure("tof"), self._lock:
            touched = self._world.integrate_depth(
                depth,
                None,
                intr,
                optical,
                stamp=at,
                law=tof_law(float(msg.range_max)),
                sensor=TOF,
                clip=clip,
            )
            self._forget_arm(base, at)
        self._tally.count("tof_fans")
        self._tally.count("tof_voxels", touched)
        self._publish_marks(base, msg.header.stamp)
        self._publish_grid(base, msg.header.stamp)

    def _roll_window(self, base: RigidPose) -> None:
        """Keep the rolling window on the cart, before the observation that found it there goes
        in: past ``window_recentre_m`` from the window's centre the box slides onto the cart by
        whole voxels, the overlap is kept where it stands and what left the window is forgotten
        (:meth:`pepin.worldmap.WorldMap.recentre`).

        Called from every paint path, under the model lock, and timed: the slide is one copy per
        channel — measured 0.3-0.9 ms on the 120x120x34 test grid and 4.7 ms on the live
        280x250x34 one (tests/unit/test_tsdf.py) — and the report line carries the last one.
        """
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
            no_depth_free=True,
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
        self._roll_window(base)
        self._reach.saw(depth)  # the source's own reach, measured off the frames themselves
        clip = self._clip(intr, camera, base, at)
        with self._tally.measure("integrate"), self._lock:
            touched = self._world.integrate_depth(
                depth, rgb, intr, camera, stamp=at, law=self._depth_law(), clip=clip
            )
            self._forget_arm(base, at)
            self._last_stamp = stamp
        self._frame_pub.publish(Header(stamp=stamp, frame_id=ODOM_FRAME))
        self._tally.count("frames")
        self._tally.count("voxels", touched)
        self._publish_marks(base, stamp)  # the camera's own turn to move the marks
        self._publish_grid(base, stamp)

    def _clip(
        self, intr: Intrinsics, camera: RigidPose, base: RigidPose, at: float
    ) -> RayClip | None:
        """The frame's self-filter: the cart's body (``self_filter``) and the arm
        (``arm_filter``) folded into one ray grid, either alone, or ``None``."""
        body = self._body_clip(intr, camera, base) if self._switches.on("self_filter") else None
        if not self._switches.on("arm_filter"):
            return body
        arm = self._arm_clip(intr, camera, base, at, body.stride if body is not None else None)
        if body is None or arm is None:
            return body if arm is None else arm
        return body.nearer(arm)

    def _arm_clip(
        self,
        intr: Intrinsics,
        camera: RigidPose,
        base: RigidPose,
        at: float,
        stride_px: int | None = None,
    ) -> RayDepth | None:
        """Where each ray of this frame enters the arm's grown links, or ``None`` when no ray
        does: config/arm.json as it stands, the joints at the frame's stamp, the camera's pose
        on the cart then; the ray grid rebuilt only when one of them moved (:class:`ArmMask`)."""
        pose = self._arm_pose(at)
        if pose is None:
            return None
        with self._tally.measure("arm"), self._arm_lock:
            clip = self._arm.for_frame(intr, _on_cart(camera, base), pose, stride_px)
        if clip is not None:
            self._tally.count("arm_frames")
            self._tally.sample("arm_share", clip.share)
        return clip

    def _arm_pose(self, at: float, count: bool = False) -> ArmPose | None:
        """The arm's joints for an observation stamped ``at`` (:func:`pepin.arm.arm_pose`);
        ``None`` while config/arm.json gives no arm. ``count`` tallies where they came from —
        once an observation, by the forget every integration ends with."""
        self._read_arm()
        with self._arm_lock:
            model = self._arm.model
        if model is None:
            return None
        pose = arm_pose(model, self._joints, at)
        if pose is not None and count:
            self._tally.count("arm_" + pose.source)
        return pose

    def _arm_boxes(self, at: float, count: bool = False) -> tuple[OrientedBox, ...]:
        """The arm's grown links in base_link at ``at`` (none while there is no arm)."""
        pose = self._arm_pose(at, count)
        if pose is None:
            return ()
        with self._arm_lock:
            return self._arm.boxes(pose)

    def _forget_arm(self, base: RigidPose, at: float) -> None:
        """Under the model's lock, after an integration: every voxel inside the arm's grown
        links, as they stood at ``at``, back to unobserved (``arm_filter``;
        :meth:`pepin.worldmap.WorldMap.forget`)."""
        if not self._switches.on("arm_filter"):
            return
        with self._tally.measure("forget"):
            boxes = [box.placed(base) for box in self._arm_boxes(at, count=True)]
            forgotten = self._world.forget(boxes)
        self._tally.count("arm_forgotten", forgotten)

    def _fan_clip(
        self, intr: Intrinsics, sensor: RigidPose, base: RigidPose, at: float
    ) -> RayDepth | None:
        """A whisker fan's arm clip (``arm_filter``): its few rays against the arm's links, each
        ray its own (the fan is a handful of pixels: nothing worth a cache)."""
        if not self._switches.on("arm_filter"):
            return None
        boxes = self._arm_boxes(at)
        if not boxes:
            return None
        clip = oriented_ray_depth(boxes, intr, _on_cart(sensor, base), 1)
        return clip if bool(np.isfinite(clip.z).any()) else None

    def _on_joints(self, msg: JointState) -> None:
        """``/arm/joint_states``: one sample of the arm's joints at its own stamp."""
        if not self._up:
            return  # the node is still being built (see __init__)
        self._joints.add(stamp_seconds(msg.header.stamp), list(msg.name), list(msg.position))
        self._tally.count("arm_joints_in")

    def _read_arm(self) -> None:
        """config/arm.json into the mask whenever the file changed (one ``stat``); a file that
        is missing or broken gives no arm, and says so in the log and the report."""
        with self._arm_lock:
            data = self._arm_file.read()
            if data == self._arm_data and self._arm_note:
                return
            self._arm_data = data
            try:
                model = ArmModel.from_dict(data)
            except (ValueError, OSError) as exc:
                self._arm.model = None
                self._arm_note = f"{self._arm_file.path} unreadable ({exc}): no arm"
                self.get_logger().error(f"arm filter: {self._arm_note}")
                return
            self._arm.model = model
            mount = "" if model.mount_measured else ", the mount NOT measured"
            self._arm_note = (
                f"{len(model.links)} boxes grown {model.margin_m * 100:.0f} cm, joints from"
                f" {model.source}{mount}"
            )
            self.get_logger().info(f"arm filter: the arm is {self._arm_note}")

    def _publish_arm(self) -> None:
        """``/fusion/arm``: the arm's grown links in base_link as they stand now, at the newest
        joint sample's stamp (the file's pose: stamp zero, the newest transform)."""
        if not self._up:
            return
        newest = self._joints.newest()
        at = newest if newest is not None else 0.0
        pose = self._arm_pose(at)
        if pose is None:
            return
        with self._arm_lock:
            boxes = self._arm.boxes(pose)
        stamp = stamp_from_seconds(at if pose.source == "topic" else 0.0)
        self._arm_pub.publish(arm_markers(boxes, stamp, BASE_FRAME))

    def _body_clip(self, intr: Intrinsics, camera: RigidPose, base: RigidPose) -> RayDepth | None:
        """Where each ray of this frame enters the cart's own body (``self_filter``), or ``None``
        when no ray does: config/body.json as it stands now, the camera's pose on the cart at the
        frame's stamp (``base_link <- camera_optical``, the two halves of the frame's own chain),
        the ray grid rebuilt only when the head, the optics or the file moved."""
        self._read_body()
        with self._tally.measure("body"):
            clip = self._body.for_frame(intr, _on_cart(camera, base))
        if clip is not None:
            self._tally.count("body_frames")
            self._tally.sample("body_share", clip.share)
        return clip

    def _read_body(self) -> None:
        """config/body.json into the mask whenever the file changed (one ``stat`` a frame); a
        file that is missing or broken cuts nothing and says so in the log and the report."""
        data = self._body_file.read()
        if data == self._body_data and self._body_note:
            return
        self._body_data = data
        try:
            model = BodyModel.from_dict(data)
        except ValueError as exc:
            self._body.model = BodyModel(())
            self._body_note = f"{self._body_file.path} unreadable ({exc}): nothing is cut"
            self.get_logger().error(f"self filter: {self._body_note}")
            return
        self._body.model = model
        names = ", ".join(box.name for box in model.boxes)
        self._body_note = f"{names} grown {model.margin_m * 100:.0f} cm"
        self.get_logger().info(f"self filter: the body is {self._body_note}")

    # ---- outputs -------------------------------------------------------------------------
    def _marks_due(self) -> bool:
        """Whether ``/depth_marks`` may go out now, and the cap's clock moved on when it may
        (``marks_hz``; 0 is every frame).

        Monotonic seconds: this is a cap on a publisher, not a measurement of anything in the
        room.
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

    # ---- the camera grids (grid_out) -------------------------------------------------------
    def _grid_due(self) -> bool:
        """Whether the camera grids may go out now (``grid_hz``); the clock moves on when so."""
        now = time.monotonic()
        with self._grid_lock:
            if now - self._grid_at < 1.0 / float(self._switches["grid_hz"]):
                return False
            self._grid_at = now
            return True

    def _publish_grid(self, base: RigidPose, stamp: Any) -> None:
        """Under ``grid_out``: the volume's occupied columns about the cart as ``/camera_grid``
        (volume frame, the observation's stamp) and, through ``map <- volume``, as an update of
        ``/camera_grid_map``. Called from every paint path, thinned by ``grid_hz``; the column
        rule is the fan's (:func:`pepin.volume_scan.band_surface`), the neighbourhood copied under
        the model lock and read outside it, as the marks are."""
        if not self._switches.on("grid_out") or not self._grid_due():
            return
        law = self._marks_law()
        with self._tally.measure("grid"):
            window = GridWindow.around(
                (float(base.translation[0]), float(base.translation[1])),
                size_m=float(self._switches["grid_size_m"]),
                resolution_m=float(self._switches["grid_resolution_m"]),
                lattice=(self._spec.origin[0], self._spec.origin[1]),
            )
            with self._lock:
                twin = grid_volume(self._world.volume, window, base, law)
            points = band_surface(twin, base, law) if twin is not None else np.zeros((0, 3))
            values = window.draw(points)
        self._grid_cells = int(np.count_nonzero(values == OCCUPIED))
        with self._grid_lock:
            self._grid_window = window
            self._grid_pub.publish(occupancy_grid(window.fields(values), stamp, ODOM_FRAME))
        self._tally.count("grids")
        self._publish_grid_map(window, points, stamp)

    def _map_from_volume(self, stamp: Any) -> RigidPose | None:
        """``map <- odom`` at ``stamp``, this laptop's own TF (``None`` when it has no answer
        within ``GRID_TF_WAIT_S``)."""
        return self._tf.pose(MAP_FRAME, ODOM_FRAME, stamp, timeout_s=GRID_TF_WAIT_S)

    def _publish_grid_map(self, window: GridWindow, points: Any, stamp: Any) -> None:
        """The same cells on the map's lattice as ONE update of ``/camera_grid_map`` that erases
        the last window and draws this one (:class:`pepin.camera_grid.MapCanvas`). Nothing while
        the map is unknown, while a new full grid settles or without the transform, each counted.

        Published under the grid lock: an update drawn on a canvas a new geometry has replaced
        must never reach the layer after that geometry's full grid.
        """
        with self._grid_lock:
            canvas = self._canvas
            settling = time.monotonic() - self._canvas_at < GEOMETRY_SETTLE_S
        if canvas is None:
            self._tally.count("grid_map_no_map")
            return
        if settling:
            self._tally.count("grid_map_settling")
            return
        to_map = self._map_from_volume(stamp)
        if to_map is None:
            self._tally.count("grid_map_no_tf")
            return
        with self._tally.measure("grid_map"), self._grid_lock:
            if self._canvas is not canvas:
                return  # a new geometry arrived meanwhile and its full grid is already out
            rect = canvas.draw(to_map_xy(points, to_map), to_map_xy(window.corners(), to_map))
            if rect is None:
                self._tally.count("grid_map_off_map")
                return
            self._grid_update_pub.publish(grid_update(canvas.update(rect), stamp, MAP_FRAME))
        self._tally.count("grid_map_updates")

    def _on_map(self, msg: OccupancyGrid) -> None:
        """``grid_map_topic``: its lattice is the one ``/camera_grid_map`` must have EXACTLY — a
        grid of any other geometry makes the global costmap resize itself and drop every layer's
        marks. A new geometry starts a new canvas under ``grid_out``; the same one is nothing."""
        geometry = map_geometry(msg)
        with self._grid_lock:
            if self._geometry is not None and self._geometry.same_as(geometry):
                return
            self._geometry = geometry
            if self._switches.on("grid_out"):
                self._start_canvas()

    def _start_canvas(self) -> None:
        """(Under the grid lock.) A fresh canvas on the map's geometry and its full grid, EMPTY
        and latched: a layer that joins later gets no stale cell from it, and the next update
        (after ``GEOMETRY_SETTLE_S``) draws the window whole."""
        assert self._geometry is not None
        self._canvas = MapCanvas(self._geometry)
        self._canvas_at = time.monotonic()
        now = self.get_clock().now().to_msg()
        self._grid_map_pub.publish(occupancy_grid(self._canvas.full(), now, MAP_FRAME))
        self._tally.count("grid_maps")
        self.get_logger().info(
            f"{GRID_MAP_TOPIC}: a full grid on {self._grid_map_topic}'s"
            f" {self._geometry.text()}; updates on {GRID_UPDATES_TOPIC} from now on"
        )

    def _clear_grids(self) -> None:
        """One empty grid wherever a layer may still be drawing ours — the last window on
        ``/camera_grid``, the last map window as an update — and the canvas dropped: what
        ``grid_out`` off leaves behind is nothing."""
        now = self.get_clock().now().to_msg()
        with self._grid_lock:
            window, self._grid_window = self._grid_window, None
            canvas, self._canvas = self._canvas, None
            if window is not None:
                empty = np.zeros((window.cells, window.cells), dtype=np.int8)
                self._grid_pub.publish(occupancy_grid(window.fields(empty), now, ODOM_FRAME))
            rect = canvas.forget() if canvas is not None else None
            if canvas is not None and rect is not None:
                self._grid_update_pub.publish(grid_update(canvas.update(rect), now, MAP_FRAME))
        self._grid_cells = 0

    def _publish_surface(self) -> None:
        """The model's surface as a cloud, in ``odom``, the frame the volume is painted in: a
        window painted through the odometry and drawn in ``map`` would be shown wherever the last
        correction happened to put it."""
        with self._lock:  # a copy under the lock (milliseconds), the crossing search outside it
            snapshot = self._world.volume.snapshot()
            stamp = self._last_stamp
        points, colours = snapshot.surface(self._switches["min_weight"], colour_fallback=True)
        self._surface_points = int(points.shape[0])  # a level the report reads, not a tally
        # the board's clock: the surface is as old as the last frame in it, not as new as now
        self._pub.publish(
            cloud_from_points(
                points,
                colours,
                stamp if stamp is not None else self.get_clock().now().to_msg(),
                ODOM_FRAME,
            )
        )

    def _report(self) -> None:
        self._read_plane(0.0)  # a board that came up after this node still moves the band
        w = self._tally.take()
        c = w.counts
        skipped = (
            f"unleaned {c['unleaned']}, leaned out {c['leaned_out']},"
            f" no tf {c['no_tf']}, bad frame {c['bad_frame']}, no intrinsics {c['no_intrinsics']}"
        )
        tf_text = "; ".join(f"{k} {c['tf_' + k]}: {v}" for k, v in w.notes.items())
        unpaired = max(int(c["depth_in"]) - int(c["pairs"]), 0)  # depths whose image never came
        self.get_logger().info(
            f"fusion: {c['frames']} frames ({w.rate('frames'):.1f}/s, {c['dropped']} dropped,"
            f" {unpaired} unpaired), integrate {w.ms_per('integrate', 'frames'):.0f} ms;"
            f" skipped: {skipped};"
            f" no image {c['no_image']}; surface {self._surface_points} points;"
            f" {self._marks_line(w)}; {self._grid_line(w)}; {self._frame_line(w)};"
            f" {self._band_text()}; {self._carve_line()}; {self._body_line(w)};"
            f" {self._arm_line(w)};"
            f" {self._tof_line(w)};"
            f" {self._world_line(w)};"
            f" {self._lean.report()};"
            f" flags: {self._switches.state()}" + (f"; tf: {tf_text}" if tf_text else "")
        )

    def _body_line(self, w: Window) -> str:
        """The self filter's half of the report: on how many frames the body was in view and on
        what share of their rays, what finding that out cost, and what the body is."""
        if not self._switches.on("self_filter"):
            return "self filter off"
        c = w.counts
        shares = w.samples.get("body_share", [])
        seen = f" ({float(np.median(shares)) * 100:.1f} % of their rays)" if shares else ""
        last = self._body.last
        inside = (
            f"; the camera stands in {', '.join(last.skipped)}: skipped"
            if last is not None and last.skipped
            else ""
        )
        return (
            f"self filter: the body met {int(c['body_frames'])} of {int(c['frames'])}"
            f" frames{seen}, {w.ms_per('body', 'frames'):.2f} ms a frame,"
            f" {self._body.rebuilds} ray grids ({self._body.last_ms:.1f} ms the last);"
            f" {self._body_note or 'config/body.json not read yet'}{inside}"
        )

    def _arm_line(self, w: Window) -> str:
        """The arm's half of the report: on how many frames it was in view and on what share of
        their rays, what that cost, where the joints came from, what was forgotten."""
        if not self._switches.on("arm_filter"):
            return f"arm filter off ({self._arm_note or 'config/arm.json not read yet'})"
        c = w.counts
        shares = w.samples.get("arm_share", [])
        seen = f" ({float(np.median(shares)) * 100:.1f} % of their rays)" if shares else ""
        last = self._arm.last
        inside = (
            f"; the camera stands in {', '.join(last.skipped)}: skipped"
            if last is not None and last.skipped
            else ""
        )
        return (
            f"arm filter: the arm met {int(c['arm_frames'])} of {int(c['frames'])} frames{seen},"
            f" {w.ms_per('arm', 'frames'):.2f} ms a frame, {self._arm.rebuilds} ray grids"
            f" ({self._arm.last_ms:.1f} ms the last); forgot {int(c['arm_forgotten'])} voxels"
            f" ({w.ms_per('forget', 'frames'):.2f} ms a frame); joints config"
            f" {int(c['arm_config'])}, topic {int(c['arm_topic'])}, stale {int(c['arm_stale'])}"
            f" ({self._arm_topic} heard {self._joints.heard});"
            f" {self._arm_note or 'config/arm.json not read yet'}{inside}"
        )

    def _tof_line(self, w: Window) -> str:
        """The whiskers' half of the report: how many fans the volume took and what one cost,
        how many were hits, misses or silent, how many were dropped — for TF, in the queue, or
        by the flag — and the clearing arithmetic a drive is judged by."""
        c = w.counts
        clear = fans_to_clear(self._spec.max_weight)
        return (
            f"tof: {int(c['tof_fans'])} fans ({w.rate('tof_fans'):.1f}/s,"
            f" {w.ms_per('tof', 'tof_fans'):.1f} ms a fan) — {int(c['tof_hits'])} hits,"
            f" {int(c['tof_misses'])} misses, {int(c['tof_silent'])} silent; dropped: no tf"
            f" {int(c['tof_no_tf'])}, queue {int(c['tof_dropped'])}, flag off {int(c['tof_off'])};"
            f" weight {TOF_WEIGHT:g} a fan, a saturated hit carved by {clear} misses"
            f" ({clear / TOF_RATE_HZ:.1f} s at {TOF_RATE_HZ:.0f} Hz)"
        )

    def _frame_line(self, w: Window | None = None) -> str:
        """The frame half of the report: where the rolling window stands, how far the cart may
        leave its centre and what the last slide cost."""
        cx, cy = self._spec.centre_xy
        slides = f", {int(w.counts['recentres'])} slides this window" if w is not None else ""
        return (
            f"frame: the volume is local memory, in {ODOM_FRAME} — a rolling window centred on"
            f" ({cx:+.2f}, {cy:+.2f}), re-centred on the cart past {self._window_recentre_m:.1f} m"
            f" (last slide {self._recentre_ms:.0f} ms{slides})"
        )

    def _marks_line(self, w: Window) -> str:
        """The costmap half of the report: where the camera's marks came from this window, how
        many went out and how fast, what one slice of the volume cost, how many bearings it
        filled and in which band — the numbers a drive is judged on without a debugger."""
        c = w.counts
        law = self._marks_law()
        return (
            f"marks: {int(c['marks'])} from the volume ({w.rate('marks'):.1f}/s,"
            f" {w.ms_per('marks', 'marks'):.1f} ms a slice){self._cap_text(w)},"
            f" {self._marks_bearings} bearings of"
            f" {round(2 * math.pi / MARKS_STEP)} filled, band {law.band_m[0]:.2f}-"
            f"{law.band_m[1]:.2f} m within {law.range_m:.1f} m at min_weight {law.min_weight:g};"
            f" {self._clear_text()}"
        )

    def _grid_line(self, w: Window) -> str:
        """The camera grids' half of the report: how many went out and what one cost, how many
        cells the last one held and where its square stands, and the map grid's geometry, updates
        and the ticks it held back (a settling geometry, no transform, off the map)."""
        if not self._switches.on("grid_out"):
            return f"grid: off ({GRID_TOPIC} and {GRID_MAP_TOPIC} silent)"
        c = w.counts
        with self._grid_lock:
            window, geometry = self._grid_window, self._geometry
        where = (
            f"{window.size_m:.1f} m at {window.resolution_m * 100:.0f} cm from"
            f" ({window.origin[0]:+.2f}, {window.origin[1]:+.2f}) in {ODOM_FRAME}"
            if window is not None
            else "no window yet"
        )
        if geometry is None:
            on_map = f"no {self._grid_map_topic} yet, {GRID_MAP_TOPIC} silent"
        else:
            on_map = (
                f"{GRID_MAP_TOPIC} on {geometry.text()}: {int(c['grid_map_updates'])} updates"
                f" ({w.ms_per('grid_map', 'grid_map_updates'):.1f} ms), {int(c['grid_maps'])}"
                f" full; held: settling {int(c['grid_map_settling'])}, no tf"
                f" {int(c['grid_map_no_tf'])}, off the map {int(c['grid_map_off_map'])}"
            )
        return (
            f"grid: {int(c['grids'])} on {GRID_TOPIC} ({w.rate('grids'):.1f}/s,"
            f" {w.ms_per('grid', 'grids'):.1f} ms), {self._grid_cells} occupied cells, {where};"
            f" {on_map}"
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
        """The volume half of the report: what the lidar wrote, what the two layers hold and how
        many revolutions were the same view again."""
        c = w.counts
        with self._lock:
            text = self._world.report()
        return (
            f"world: {c['revolutions']} revolutions ({c['scans_dropped']} dropped,"
            f" {w.ms_per('scan', 'revolutions'):.0f} ms), {text}; {self._views.report()}"
        )


def main() -> None:
    spin_main(DepthFusion)


if __name__ == "__main__":
    main()
