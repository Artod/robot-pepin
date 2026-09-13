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

THE WORLD MAP. The same volume is also the map itself (:mod:`pepin.worldmap`): ``/scan`` is
integrated into it along the beams' own rays — carving free space, a surface at each return, on
the lidar's own weight channel, which the camera's scale-uncertain depth may not repaint — and
the layer at the lidar's plane reads out as an occupancy grid. The rays follow the body: with
``imu_lean`` on the scan is placed by the leaning pose, so a beam that climbs 44 cm over 5 m
while the cart tips writes a tabletop where it hit one instead of a wall at the plane, and a
revolution taken past ``lean_gate_deg`` is dropped rather than believed. A lean gravity did
not vote for (``lean_min_quality``: a drifting gyro's own signature) is no lean at all, and the
measurement is placed level instead of by a number nobody measured.

With ``map_source=volume`` that grid goes out as ``/map`` at ``map_hz`` (transient local), so
the tracker and Nav2 localise and plan on the volume instead of on a frozen file, and a "known
room" is just a volume that was seeded (``seed_map``) or resumed from a snapshot
(``resume_volume``) instead of an empty one. Exactly one publisher of /map, and both halves of
that are launch decisions this node is told: the bridge mode says which side owns the topic
(:func:`pepin.deployment.map_owner`) and the ``world_map`` parameter says whether the launch
kept RTAB-Map's grid off it. Without both, the volume is refused on /map however ``map_source``
is set afterwards. The volume is snapshotted to ``world_path`` every ``snapshot_s`` and at
shutdown.

The flags (:data:`FLAGS`, ``ros/flags.sh set depth_fusion <flag> <value>``): ``enabled``,
``fit_gate``, ``imu_lean``, ``lean_gate_deg``, ``lean_min_quality``, ``self_heal``, ``align``,
``min_weight``, ``map_min_weight``, ``surface_hz``, ``band_half_z``, ``lidar_layer``,
``no_return_free``, ``map_source``, ``map_hz``, ``snapshot_s``, ``resume_volume``; their state
is printed in every report line, beside the band itself and the source of the plane it is
centred on. ``/fusion/reset``
(std_srvs/Trigger) empties the model, the pairing queues and the tallies.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from message_filters import Subscriber, TimeSynchronizer
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2
from std_msgs.msg import Float32
from std_srvs.srv import Trigger

from pepin.deployment import map_owner
from pepin.depth import Intrinsics
from pepin.flags import Flag, FlagSet
from pepin.frame_pose import BASE_FRAME, FramePoser
from pepin.lean import LEAN_QUALITY_FLOOR, SCAN_LEAN_GATE_DEG, LeanGate
from pepin.mounts import LASER_FRAME, load_lidar_mount
from pepin.tsdf import (
    YAW_SEARCH,
    AlignReason,
    GridSpec,
    RigidPose,
    align_yaw,
    backproject,
    band_half_z_m,
)
from pepin.watch import DRIVE_FIT
from pepin.worldmap import (
    LidarLaw,
    PlanarMount,
    SliceLaw,
    SnapshotClock,
    WorldMap,
    bearings_in_base,
    trinary_from_log_odds,
)
from pepin_bringup.msgs import (
    array_from_image,
    cloud_from_points,
    occupancy_grid,
    rpy_from_transform,
    scan_arrays,
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
WORLD_PATH = "/maps/world_live.npz"  # the volume's snapshot; ros/maps is mounted there
SCAN_TOPIC = "/scan"
TF_WAIT_S = 0.3
BAND_TF_WAIT_S = 5.0  # the static base_link -> laser edge at start: the board publishes it once
PAIR_QUEUE = 40  # depth arrives a fraction of a second after its image; pair by exact stamp
BAND_STRIDE = 3
BAND_MIN_POINTS = 50  # a frame with fewer points in the band is not worth a yaw search
AT_BOUND_STREAK = 30  # ~3 s of frames refused at the search's bound: the model no longer fits
STAGES = ("align", "integrate", "scan", "map")

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
        "fit_gate",
        True,
        description="frames are fused only while the tracker reports /localization_fit >= 0.50;"
        " off, every frame is fused",
        why="where a tracker speaks, and the off state is measured: in the first online-SLAM"
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
        2.0,
        description="observations a voxel needs before it is shown in /fusion/surface (the debug"
        " cloud only: /map has map_min_weight)",
        why="inherited from the map slice, where it was measured: at min_weight 2 the lidar slice"
        " holds 905 walls and at 6 it holds 817, the cells a single pass wrote falling out"
        " (scratch/worldmap_from_tape.txt). For the debug cloud itself nothing was measured; it"
        " is the same number so the picture and the map agree",
        on_when="raise it to show only what several frames agree on",
        off_when="0 shows every voxel ever touched, noise included — a look at what one pass"
        " sees; it changes nothing the cart drives on",
        range=(0.0, 100.0),
    ),
    Flag(
        "map_min_weight",
        2.0,
        description="observations a voxel needs before it speaks in /map. Its own flag, and"
        " capped at the lidar's own weight cap",
        why="the cap is measured: lidar cells saturate at LidarLaw.max_weight 20 while the"
        " volume's own cap is 60, so anything above 20 leaves the whole map unknown — a synthetic"
        " box at min_weight 21 published free 0, occupied 0, unknown 14400, with Nav2 and the"
        " tracker driving on that. The 2.0 is the slice's own measured maturity (905 walls at 2,"
        " 817 at 6)",
        on_when="raise it towards 20 for a map that must be certain — a world the cart has driven"
        " more than once, saved to file",
        off_when="lower it towards 0 in a fresh room, where the cart must plan through what a"
        " single pass saw",
        range=(0.0, LidarLaw.max_weight),
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
        on_when="on wherever the volume is the map (map_source volume) or the surface must show"
        " what the lidar knows",
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
        "map_source",
        "file",
        description="where /map comes from: the saved file another node serves, or the volume's"
        " own lidar layer published from here at map_hz. Only where the stack was launched with"
        " world_map:=true; anywhere else volume is refused, because another node is on /map",
        why="the other state has been seen to break a run: on 2026-09-10 RTAB-Map's own grid"
        " landed on /map beside the board's static map and fed the laptop's global costmap a"
        " second, growing map. Two publishers of one /map is the failure, so the deployment's"
        " map_owner and the launch's world_map:=true must both agree before volume is allowed."
        " The volume itself is good enough — its walls sit within one cell of the saved map 79.3"
        " % of the time",
        on_when="volume where the laptop owns /map (ros/laptop.sh vslam --world-map) and the room"
        " is being mapped as it is driven",
        off_when="file wherever a map server or RTAB-Map already publishes /map, which is every"
        " other mode",
        choices=("file", "volume"),
    ),
    Flag(
        "map_hz",
        1.0,
        description="how often the volume's layer goes out as /map when map_source is volume",
        why="default by design, unmeasured: 1 Hz is the cadence RTAB-Map's own map updates at,"
        " and /map is transient-local, so a subscriber that arrives late is served the last one"
        " regardless",
        on_when="raise it when the map is built while driving and the costmap lags visibly behind"
        " the room",
        off_when="lower it on a busy laptop: every publication is a whole grid over the bridge",
        range=(0.1, 5.0),
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
        description="a volume snapshot at world_path is loaded at start, so a known room is a"
        " resumed volume; off, the volume starts empty and grows from the sensors",
        why="default by design, unmeasured. The one hard rule around it is a guard: a snapshot is"
        " resumed only onto the grid config/fusion.json describes (280x250x34 voxels of 5 cm from"
        " -19.5, -5.5, -0.15), so a changed grid starts empty instead of resuming into the wrong"
        " place",
        on_when="on in the room the snapshot was taken in",
        off_when="off for a new room, after the map's origin moves, or to measure how fast the"
        " volume fills from nothing",
        live=False,
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


class DepthFusion(Node):
    """Fuses depth frames into the TSDF and publishes its surface."""

    def __init__(self) -> None:
        super().__init__("depth_fusion")
        config = Path(str(self.declare_parameter("config", CONFIG).value))
        self._spec = GridSpec.load(config)
        # An unknown place has no coordinates yet: the map frame is born under the cart, and a
        # box laid out for the served map (flat3's -19.5..-5.5 m) would not even contain it
        # (2026-09-13: 239 revolutions integrated only the far wall). In SLAM mode the box is
        # centred on the start; the served-map box stays what the config says.
        if str(self.declare_parameter("mode", "vision").value) == "slam":
            self._spec = self._spec.centred_on_start()
        # The band's centre is the lidar's plane, and the only copy of it both sides of the
        # bridge agree on is the published base_link -> laser edge: this process reads
        # config/lidar.json from the laptop's checkout while the beams are published from the
        # board's own synced copy, and between a mount change and ros/sync.sh the two differ by
        # the whole change. TF is asked at start (and again while it has not answered); the
        # mount here is only the fallback, and the report line says which one the band sits on.
        self._plane_z_m = load_lidar_mount().z_m
        self._plane_source = "config"
        self._band_z_m = band_z_m(self._plane_z_m, band_half_z_m())
        # Which side owns /map in the mode the stack was brought up in: with the board serving a
        # saved map, a second publisher here would give the costmaps two maps and the tracker a
        # map to rebuild on every second (2026-09-10 01:00, RTAB-Map's grid beside the board's).
        self._mode = str(self.get_parameter("mode").value)
        # The other half of the same question, and it is a launch decision, not a live one: in
        # SLAM the launch remaps RTAB-Map's grid onto /map unless it was brought up with
        # world_map:=true, and no flag set afterwards can move that remap. Told here so that
        # flipping map_source live cannot put a second publisher on /map.
        self._world_map = bool(self.declare_parameter("world_map", False).value)
        self._map_mine = map_owner(self._mode) == "laptop" and self._world_map
        self._map_refusal = self._why_not_mine()
        self._world_path = Path(str(self.declare_parameter("world_path", WORLD_PATH).value))
        self._seed_map = str(self.declare_parameter("seed_map", "").value)
        # The lidar's plane is calibrated, never typed: it comes from config/lidar.json, the one
        # file the board's launch publishes the laser transform from.
        self._mount = PlanarMount.from_config(
            str(self.declare_parameter("lidar_config", LIDAR_CONFIG).value)
        )
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
        self._lean = LeanFeed(
            self,
            config.parent,
            use_gyro=self._switches.on("imu_lean"),
            on_unmounted=self._on_unmounted,
        )
        self._poser = FramePoser(
            TfHistory(self._tf, timeout_s=TF_WAIT_S),
            lean=self._lean,
            apply_lean=self._switches.on("imu_lean"),
            min_lean_quality=float(self._switches["lean_min_quality"]),
        )
        self._read_plane(BAND_TF_WAIT_S)
        # The scan's own answer to the lean: a frame can be placed leaning, a revolution taken
        # too far from level can only be dropped (pepin.lean.LeanGate).
        self._gate = LeanGate(float(self._switches["lean_gate_deg"]))
        self._intr: Intrinsics | None = None
        self._fit = 0.0  # no report yet reads as lost: every gate here compares with <
        self._lock = threading.Lock()  # the model and its last stamp, worker vs publisher
        self._world = WorldMap(self._spec, self._mount)
        self._last_stamp: Any = None  # the last fused frame's header stamp, the board's clock
        self._bound_streak = 0  # consecutive frames refused at the bound (self-healing)
        self._surface_points = 0
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
        self._map_pub: Any = None  # made on the first publish: only where this side owns /map
        self._snapshots = SnapshotClock(float(self._switches["snapshot_s"]))
        self._start_state()
        self._surface_timer = self.create_timer(
            self._period(self._switches["surface_hz"]), self._publish_surface
        )
        self._map_timer = self.create_timer(
            self._period(self._switches["map_hz"]), self._publish_map
        )
        self.create_timer(30.0, self._report)
        nx, ny, nz = self._spec.shape
        self.get_logger().info(
            f"fusion up: {nx}x{ny}x{nz} voxels of {self._spec.voxel_m * 100:.0f} cm from"
            f" {self._spec.origin}; {self._switches.state()}; {self._band_text()}; fused while"
            f" /localization_fit >= {DRIVE_FIT:.2f}; mode {self._mode}, /map"
            f" {'is this volume' if self._map_mine else f'not ours ({self._map_refusal})'};"
            f" snapshot {self._world_path}"
        )

    def _why_not_mine(self) -> str | None:
        """Why this node may not publish /map in the stack it was launched into, or ``None``
        when it may: the phrase the report line and the refusal log say."""
        if map_owner(self._mode) != "laptop":
            return f"/map is the {map_owner(self._mode)}'s in {self._mode} mode"
        if not self._world_map:
            return "launched without world_map: RTAB-Map's grid is on /map"
        return None

    def _start_state(self) -> None:
        """What the volume starts as: a resumed snapshot, else a saved map written into the
        lidar's layer, else empty. All three are the same machine afterwards — the difference
        between a known room and an unknown one is only what is already in the cells."""
        if self._switches.on("resume_volume") and self._world_path.exists():
            try:
                resumed = WorldMap.load(self._world_path, self._mount)
            except (ValueError, OSError) as exc:
                self.get_logger().warning(f"{self._world_path}: not resumed ({exc})")
            else:
                if resumed.spec != self._spec:  # the config is the truth, not the old file
                    self.get_logger().warning(
                        f"{self._world_path}: saved on another grid ({resumed.spec.shape} from"
                        f" {resumed.spec.origin}); starting empty on the config's"
                    )
                else:
                    self._world = resumed
                    stats = resumed.maturity()
                    self.get_logger().info(
                        f"resumed {self._world_path}: {stats['voxels']:.0f} voxels,"
                        f" {stats['frames']:.0f} frames, stamp {resumed.stamp:.0f}"
                    )
                    return
        if self._seed_map:
            from pepin.mapping import grid_from_pgm

            seeded = self._world.seed_from_grid(
                *trinary_from_log_odds(grid_from_pgm(self._seed_map))
            )
            self.get_logger().info(f"seeded the lidar layer from {self._seed_map}: {seeded} cells")

    def close(self) -> None:
        """Stop the workers and the TF listener, snapshot the volume, and wait for them all,
        before the node is destroyed: a run's map outlives the run."""
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
        band, the two rates retime their timer, ``snapshot_s`` its clock, and the rest are only
        read where they are used."""
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
        if name == "snapshot_s":
            self._snapshots = SnapshotClock(float(new), self._snapshots.last_s)
            return
        timers = {"surface_hz": "_surface_timer", "map_hz": "_map_timer"}
        if name not in timers:
            return
        timer = getattr(self, timers[name])
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

    def _on_reset(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        with self._lock:
            self._world = WorldMap(self._spec, self._mount)
            self._last_stamp = None
        self._worker.clear()
        self._scans.clear()
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

    def _on_scan(self, msg: LaserScan) -> None:
        """A lidar revolution: straight to its worker, newest first (an older one still waiting
        is dropped — the volume wants the room as it is, not a backlog)."""
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
        with self._tally.measure("scan"), self._lock:
            self._world.law = self._law()
            touched = self._world.integrate_scan(
                bearings_in_base(angles, yaw, mirrored), ranges, base, mount, stamp=at
            )
        self._tally.count("revolutions")
        self._tally.count("scan_voxels", touched)
        if self._snapshots.due(time.monotonic()) and self._switches["snapshot_s"] > 0.0:
            self._snapshot()

    def _law(self) -> LidarLaw:
        """How a beam writes into the volume right now: the defaults with the live flags in
        them, rebuilt per scan so a flag set mid-run takes effect on the next revolution."""
        return LidarLaw(no_return_free=self._switches.on("no_return_free"))

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

    def _snapshot(self) -> None:
        """Write the volume to ``world_path`` — the warm cache a next run resumes from. The
        lock is held for the write (half a second for a grid of noise, less for a real one), so
        a snapshot costs the camera a frame or two once every ``snapshot_s``."""
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
        if self._switches.on("fit_gate") and self._fit < DRIVE_FIT:
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
            touched = self._world.integrate_depth(depth, rgb, intr, camera, stamp=at)
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
            self._world = WorldMap(self._spec, self._mount)
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
            snapshot = self._world.volume.snapshot()
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

    def _publish_map(self) -> None:
        """The volume's lidar layer as /map, at ``map_hz``, when the flag says the map comes
        from the volume and nothing else in this stack is on /map."""
        if self._switches["map_source"] != "volume":
            return
        if not self._map_mine:
            self._tally.count("map_refused")
            return
        with self._tally.measure("map"), self._lock:
            view = self._world.lidar_slice(SliceLaw(min_weight=self._switches["map_min_weight"]))
            stamp = self._last_stamp
        if self._map_pub is None:  # transient local: a late subscriber still gets the map
            self._map_pub = self.create_publisher(
                OccupancyGridMsg,
                "/map",
                QoSProfile(
                    depth=1,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    reliability=ReliabilityPolicy.RELIABLE,
                ),
            )
            self.get_logger().info(
                f"/map is the volume's now: {view.shape[1]}x{view.shape[0]} cells of"
                f" {view.resolution_m * 100:.0f} cm from {view.origin}, the layer"
                f" {view.band_m[0]:.2f}-{view.band_m[1]:.2f} m"
            )
        self._map_pub.publish(
            occupancy_grid(
                view.message_fields(),
                stamp if stamp is not None else self.get_clock().now().to_msg(),
                "map",
            )
        )
        self._tally.count("maps")

    def _report(self) -> None:
        self._read_plane(0.0)  # a board that came up after this node still moves the band
        w = self._tally.take()
        c = w.counts
        skipped = (
            f"low fit {c['low_fit']}, at bound {c['refused_at_bound']},"
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
            f" {self._band_text()}; {self._world_line(w)}; {self._lean.report()};"
            f" flags: {self._switches.state()}" + (f"; tf: {tf_text}" if tf_text else "")
        )

    def _world_line(self, w: Window) -> str:
        """The map half of the report: what the lidar wrote, what the slices hold, where /map
        comes from and how old the snapshot is."""
        c = w.counts
        with self._lock:
            text = self._world.report()
        source = str(self._switches["map_source"])
        if source == "volume" and self._map_refusal is not None:
            source = f"volume (refused {c['map_refused']}x: {self._map_refusal})"
        elif source == "volume":
            source = f"volume ({c['maps']} published, {w.ms_per('map', 'maps'):.0f} ms)"
        age = self._snapshots.age_s(time.monotonic())
        return (
            f"world: {c['revolutions']} revolutions ({c['scans_dropped']} dropped,"
            f" {w.ms_per('scan', 'revolutions'):.0f} ms), {text}; /map from {source}; snapshot"
            f" {'never' if age == math.inf else f'{age:.0f} s old'}"
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
