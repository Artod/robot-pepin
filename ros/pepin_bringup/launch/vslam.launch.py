"""Camera + lidar mapping on the laptop: RTAB-Map, either beside the known map or AS the map.

Two modes, one launch, because everything around RTAB-Map is the same in both: the neck camera's
stream becomes images (pepin_bringup.camera_stream), a depth network scaled by the lidar turns
them into depth images (pepin_bringup.depth_stream), the same frames are read once more at floor
height (pepin_bringup.contact_scan) and fused into one surface (pepin_bringup.depth_fusion), and
the operator's Foxglove connects here for the 3D view. What the mode changes is which map the
robot drives on — and, with it, whether the whole-map watchdog half of the laptop's localizer
(pepin_bringup.laptop_localizer) has a saved map to search for the cart on. Its other half — the
camera's scans matched here and sent to the board as pose measurements — runs in both modes and
simply finds no belief to start from where no tracker publishes one.

KNOWN MAP (the default). The board's tracker owns ``map -> odom`` on a saved map, and RTAB-Map
is built on the SAME continuous odometry every other consumer rides — the board EKF's
``odom -> base_link`` over the bridge (``odom_frame_id: odom``) — with its graph in a frame of
its own (``map_frame_id: rtabmap``, tied to ``map`` by pepin_bringup.rtabmap_frame's anchor).
It never publishes into anyone's tree (``publish_tf`` false): two localisations coexist, its
grid is the map the tracker will be handed one day, and what it learns about the cart's place
travels to the board as ONE MORE MEASUREMENT for the tracker's fusion (rtabmap_frame's
``graph_measurement``), never as a second owner of ``map -> odom``.

Until 2026-09-14 that odometry was the tracker's own pose (``graph_odom:=false`` still is), and
the price was the whole point of the graph: the tracker's pose teleports when it relocalises, a
neighbour edge between two nodes one second apart then carried 0.888 m against its 0.244 m
sigma, and on that 3.64 error ratio RTAB-Map rejected EVERY loop closure it found for three
hours. The graph's answer is only worth fusing because it is now built on odometry that never
jumps.

ONLINE SLAM (``slam:=true`` — ros/laptop.sh vslam --slam). There is no saved map: the robot is
put somewhere unknown and RTAB-Map builds ONE map while it drives. Its "odometry" is then the
EKF's wheels-plus-gyro odometry over the bridge (``odom_frame_id: odom``) and it owns
``map_frame_id: map`` — its grid is published as ``/map``, which the board's global costmap reads
as its static layer, and its correction becomes the board's ``map -> odom`` (rtabmap_frame's
``slam`` switch sends it as a message, pepin_bringup.slam_frame broadcasts it there). The
database starts empty every session unless ``resume:=true``, and it is a file of its own: a SLAM
session must never wipe the known map's graph. Two of the fusion's switches are this mode's to
set, not the operator's: ``fit_gate`` comes up OFF (no tracker runs here, so
``/localization_fit`` never comes and the gate would fuse nothing at all) and ``map_source``
comes up already saying where ``/map`` will come from. Both stay live — this is the default a
session starts from, not a lock. With ``camera_only:=true`` the lidar is not subscribed at all
and the grid comes from the camera's depth — the honest test of "the camera as the primary
sense", and the one case where the map is only as true as the network's scale.

THE CAMERA AS A THIRD ODOMETRY (``vo``, on by default). Beside all of that, rtabmap_odom's
``rgbd_odometry`` reads the same picture and depth and answers with a pose per frame, and
pepin_bringup.visual_odometry gates it, gives it a documented covariance and publishes ``/vo``
for the board's EKF (ros/params/ekf.yaml's ``odom1``: x and y differentially, no yaw — the gyro
owns heading). Nothing reaches the filter until that node's ``vo_publish`` flag is on; with
``vo:=false`` neither process starts at all.

Arguments: ``board`` (the robot's address for the camera stream), ``slam``, ``camera_only``,
``resume``, ``graph_odom`` (known-map mode: whose odometry the graph is built on),
``neighbor_refining`` (known-map mode: whether ICP refines the neighbour links and stiffens them
with its own covariance — false, or no closure the graph finds survives RGBD/OptimizeMaxError),
``vo``, ``database`` (empty: chosen by the mode), ``bridge_admin`` (the laptop
bridge's REST admin, asked whether it still lists this launch's previous incarnation),
``static_camera_tf``
(default true: the camera node broadcasts base_link -> camera_link from config/camera.json; false
when the board's neck node publishes that edge live — ros/feature.sh neck on, ``ros/laptop.sh
vslam --neck`` — since two publishers of one edge fight).
"""

from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetLaunchConfiguration,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pepin.deployment import CONTAINER_STOP_TIMEOUT_S, laptop_launch_nodes, map_owner

# How long a node of this launch is given to end on SIGINT before the launch escalates to
# SIGTERM, and then how long before SIGKILL. launch's own defaults are 5 s and 5 s, which is
# under RTAB-Map's close of a 20-28 GB database and under the board's dozen nodes leaving DDS:
# every shutdown ended in SIGKILLs mid-write, and eight of them left ros/maps/rtabmap.db
# malformed (2026-09-13). The window is the container's own stop window
# (pepin.deployment.CONTAINER_STOP_TIMEOUT_S, ros/lib.sh, board/pepin-ros.service), so on a
# `docker stop` nothing inside escalates before docker's SIGKILL at its end, and on a shutdown
# from inside (the bridge watch's exit) the nodes get the same seconds.
SHUTDOWN = [
    SetLaunchConfiguration("sigterm_timeout", str(CONTAINER_STOP_TIMEOUT_S)),
    SetLaunchConfiguration("sigkill_timeout", "5"),
]

# What RTAB-Map is told in both modes: how a transform between two nodes is found, how the graph
# is built and closed, and what a place looks like. Only the frames and the grid depend on the
# mode (:data:`KNOWN_MAP`, :data:`SLAM_LIDAR`, :data:`SLAM_CAMERA_ONLY`).
RTABMAP = {
    # registration: ICP on the 2D scans in the plane; the camera only says "I have been here"
    "Reg/Strategy": "1",
    "Reg/Force3DoF": "true",
    "Icp/VoxelSize": "0.05",
    "Icp/MaxCorrespondenceDistance": "0.15",
    "Icp/PointToPlane": "false",
    # graph: close loops with nearby nodes by space, and optimise from the OLDEST node — the
    # map frame then stays where the session started and every jump lands in map -> odom,
    # which is the one correction the board is told about.
    "RGBD/ProximityBySpace": "true",
    "RGBD/LinearUpdate": "0.05",
    "RGBD/AngularUpdate": "0.05",
    "RGBD/OptimizeFromGraphEnd": "false",
    # A mono camera has no 3D features, so the visual registration RTAB-Map runs to seed a
    # loop-closure transform always fails ("old=0" features, 299 hypotheses rejected in one
    # session): the camera only names the node, and ICP on the two scans, started from
    # identity, gives the transform — the hypothesis is a revisit of the same spot.
    "RGBD/LoopClosureIdentityGuess": "true",
    "Rtabmap/DetectionRate": "1.0",
    # appearance: GFTT/ORB words, a few hundred per image
    "Kp/DetectorStrategy": "8",
    "Kp/MaxFeatures": "400",
    "Mem/RehearsalSimilarity": "0.30",
    # What a node costs on disk. Measured on the known map's database 2026-09-14 (20.93 GB,
    # 33020 nodes, 85 m travelled, one evening): 634 KB a node, of which the depth PNG is 432 KB,
    # the JPEG 78 KB and the keypoint descriptors 28 KB. Both values below are free for loop
    # detection and neither touches the graph.
    #   ImagePostDecimation is applied when the frame is COMPRESSED FOR THE DATABASE, after the
    # words, their 3D positions and the local grid have been computed from the full-resolution
    # frame: quartering the pixels of what is stored costs the detector nothing (it never reads
    # them again -- RGBD/LoopClosureReextractFeatures is false) and takes the two images from
    # 510 KB a node to about 155.
    #   NotLinkedNodesKept false deletes the nodes RTAB-Map itself has already dropped out of the
    # graph. 13891 of the 33020 above -- 42 % -- appear in no link at all, neighbour or closure:
    # the cart stood still, the node failed RGBD/LinearUpdate and was unlinked, and only this
    # parameter's default kept its 634 KB. An unlinked node is in no path, so no proximity search
    # reaches it and no closure can be computed against it; what it still costs is its words in
    # the dictionary. Together: about 20 GB a day becomes about 5.
    "Mem/ImagePostDecimation": "2",
    "Mem/NotLinkedNodesKept": "false",
    # What happens to the database when the process is killed. RTAB-Map 0.22.1's defaults
    # (rtabmap/core/Parameters.h:270-274 in the image) are JournalMode 3 = MEMORY and
    # Synchronous 0 = OFF: sqlite keeps the rollback journal in RAM, so a SIGKILL takes the only
    # record of the half-written transaction with it and what is left on disk is a torn file.
    # That is exactly how ros/maps/rtabmap.db became "database disk image is malformed" after
    # the eight kills of 2026-09-13, and how a reader saw a torn page on 2026-09-15.
    #   1 = TRUNCATE: the journal is a file beside the database, so an interrupted transaction
    # is rolled back at the next open and a KILLED process cannot corrupt anything. TRUNCATE and
    # not 0 = DELETE because it zeroes the journal's header instead of unlinking the file, which
    # is one directory operation less per commit — the cheapest on-disk mode.
    #   1 = NORMAL: sqlite fsyncs at the end of each commit instead of never. This is the whole
    # cost of the change — one flush per node written, at Rtabmap/DetectionRate 1.0 that is one
    # a second — and it is what carries the guarantee past a process kill to most power losses.
    # It is not FULL (2): FULL syncs the journal header as well, and a robot that loses power
    # mid-commit has a bad day either way; the kill is the failure that actually happens here.
    # A WAL mode is not on offer — this parameter is an int over DELETE/TRUNCATE/PERSIST/MEMORY/
    # OFF and RTAB-Map never passes anything else to `PRAGMA journal_mode`.
    "DbSqlite3/JournalMode": "1",
    "DbSqlite3/Synchronous": "1",
    "Optimizer/GravitySigma": "0",
    # a mono network's depth frays at object edges and far away: lone voxels go
    "Grid/NoiseFilteringRadius": "0.10",
    "Grid/NoiseFilteringMinNeighbors": "5",
    "Grid/CellSize": "0.05",
    "Grid/DepthDecimation": "4",  # 160x90 depth samples a frame: plenty for 5 cm voxels
    "Grid/MaxGroundHeight": "0.08",
    "Grid/MaxObstacleHeight": "1.3",  # the band the costmap scan uses too
    "Grid/NormalsSegmentation": "false",  # height decides ground vs obstacle; the floor is flat
}

# Beside a known map: RTAB-Map's own frame, and a grid from the lidar AND the camera's depth —
# the lidar keeps the plane exact, the depth adds what stands above it (table tops, seats,
# shelves). Grid/3D keeps the voxels so the operator sees the room in three dimensions
# (cloud_map). Nothing here reaches Nav2: the tracker's saved map is what the robot drives on.
KNOWN_MAP = {
    # RTAB-Map's "odometry" is the EKF's odom -> base_link, the same continuous odometry every
    # other consumer rides, and its graph keeps a frame of its own tied to the map by
    # pepin_bringup.rtabmap_frame (``graph_odom``, the default).
    "odom_frame_id": "odom",
    "map_frame_id": "rtabmap",
    "subscribe_scan": True,
    "Grid/Sensor": "2",  # 0=scan, 1=depth, 2=both (0.22's name for the old Grid/FromDepth)
    "Grid/3D": "true",
    "Grid/RangeMax": "5.0",
    "Grid/RayTracing": "false",  # 3D ray tracing costs more than it clears at 1 Hz
    # WHY THE NEIGHBOUR LINKS ARE NOT REFINED HERE (2026-09-14, the second rejection disease).
    # With the graph on the EKF's odometry RTAB-Map finds its closures — 5 or 6 an iteration,
    # 8459 against 1, 1137, 1462, 1813, 2398, registered with 211 visual inliers against a
    # Vis/MinInliers of 20 — and throws every one of them away on RGBD/OptimizeMaxError, which
    # compares each link's residual after optimisation with that link's own standard deviation:
    #   "Rejecting all added loop closures (5, first is 8459 <-> 1) ... maximum graph error ratio
    #    11.737604 (edge 7083->7084, type=0, abs error=0.217136 m, stddev=0.018499)"
    #   "... ratio 3.530088 (edge 962->1023, type=0, abs error=0.435329 deg, stddev=0.002152)"
    # Both culprits are NEIGHBOUR links (type=0), and the stddev is why. Refined, a neighbour
    # link carries ICP's own fit residual as its covariance, and ICP on two scans of a flat is
    # sure of itself: over the 607 neighbour links in ros/maps/rtabmap.db the median sigma is
    # 0.75 cm and 0.135 deg (scratch/graph_link_sigmas.py). At OptimizeMaxError's 3.0 that means
    # the optimised graph may not disagree with ANY neighbour link by more than 2.2 cm or 0.40
    # deg — while a closure across a session boundary asks for tens of centimetres by
    # construction, and the whole correction lands on whichever link joins the two parts (node
    # 7083 and 7084 are one second and 0.4 cm apart; nothing is wrong with that edge, it is
    # simply the hinge). The check could therefore never accept a closure on this graph, however
    # right the closure was.
    # Unrefined, the link carries the ODOMETRY's uncertainty instead (odom_tf_*_variance below),
    # which is what the check was written to compare against — how far the chain may have
    # drifted, not how well two clouds overlay. Nothing that decides whether a closure is CORRECT
    # is touched: Vis/MinInliers, the ICP checks and OptimizeMaxError itself all stand, and a
    # wrong closure asking metres of one link is still refused. What is given up is the scan's
    # correction of the wheels between two nodes 5 cm apart — a centimetre of local metric
    # accuracy, against a graph that can close a loop at all. Back with neighbor_refining:=true.
    "RGBD/NeighborLinkRefining": "false",
}

# RTAB-Map reads its odometry from TF here (odom_frame_id above), and TF carries no covariance,
# so rtabmap_slam gives every odometry link a constant one from these two — which is what the
# neighbour links above are worth once ICP no longer overwrites them. 0.001 m2 is 3.2 cm per
# link, rtabmap_launch's own default; 0.0001 rad2 is 0.57 deg. Per link, over an evening's 600
# links, that is a chain 1-sigma of 78 cm and 14 deg: honest for wheels and a gyro, and loose
# enough that a real loop closure's correction has somewhere to go, while 3 sigma on one link
# (9.5 cm, 1.7 deg) still catches the closure that is simply wrong.
TF_ODOMETRY_VARIANCE = {
    "odom_tf_linear_variance": 0.001,
    "odom_tf_angular_variance": 0.0001,
}

# The old known-map table, one launch argument away (``graph_odom:=false``): RTAB-Map's
# "odometry" is the TRACKER's pose (map -> odom -> base_link), so a node lands where the lidar
# says the cart is and two clouds a few degrees apart coincide (2026-09-10: with raw odometry,
# the graph rejected most closures as inconsistent and the voxels smeared). What it cost is the
# reason for the default above: that pose teleports on a relocalisation, and a neighbour edge
# between two nodes a second apart then carries a jump no graph can accept — 0.888 m against a
# 0.244 m sigma, a ratio of 3.64 over the 3.0 of RGBD/OptimizeMaxError, on which RTAB-Map
# rejected EVERY loop closure it found for three hours ("Rejecting all added loop closures (5,
# first is 31448 <-> 30978) ... maximum graph error ratio 3.64 (edge 28042->28043)", 2026-09-14),
# though each was registered with 67 visual inliers against a Vis/MinInliers of 20.
TRACKER_ODOM = {"odom_frame_id": "map"}

# Online SLAM: the EKF's odometry under it, the map frame its own, the grid the one Nav2 plans
# on. Grid/Sensor 0 — the 2D grid is the LIDAR's alone even though the depth is subscribed for
# the graph's appearance and for the fusion: the camera's metric scale is scene-dependent (a 0.94
# to 1.98 across one afternoon, 44 % of its costmap marks BEHIND the wall the lidar sees,
# 2026-09-11), and a map the cart plans on may not be built out of that. A 2D scan carries its
# own free space (rtabmap ray-traces from the viewpoint to each return), so nothing else is
# needed to tell floor from unknown.
SLAM = {
    "odom_frame_id": "odom",
    "map_frame_id": "map",
}
SLAM_LIDAR = {
    "subscribe_scan": True,
    "Grid/Sensor": "0",
    "Grid/3D": "false",
    "Grid/RangeMax": "8.0",  # the flat's walls; the LD19's last metres are noise, not geometry
    "Grid/RayTracing": "false",  # a 2D scan already carves the space it flew through
    "RGBD/NeighborLinkRefining": "true",  # the scan refines the wheels' link between two nodes
}
# The lidar unplugged: the grid is the camera's depth, ray-traced so the floor it flew over
# becomes free space instead of unknown, and capped at 3 m because that is where this network's
# scale stops being a measurement. Neighbour links are NOT refined here — ICP on a cloud whose
# scale drifts would push the odometry it is meant to correct.
#   THE REGISTRATION HAS TO BE VISUAL HERE. The common table asks for ICP (Reg/Strategy 1)
# because everywhere else a 2D scan is what makes two nodes metric. Without the lidar there is no
# scan in a node at all, and rtabmap_ros does not notice: its only scan-aware rule about ICP sets
# RGBD/ProximityPathMaxNeighbors, and it fires when a scan IS subscribed (rtabmap_slam's
# CoreWrapper). So ICP would be asked to register two point clouds that do not exist — every loop
# closure and every proximity link fails before it is scored, the graph can never be corrected,
# and the session is dead reckoning with a database. Reg/Strategy 0 is RTAB-Map's own RGB-D
# default: the words are matched and their 3D positions come from the depth image, which is the
# one thing this mode does have. It is also why KNOWN_MAP's note about visual registration
# failing ("old=0 features") does not apply — that is ICP-only memory keeping no 3D words.
SLAM_CAMERA_ONLY = {
    "subscribe_scan": False,
    "Reg/Strategy": "0",  # Vis: there is no scan to run ICP on
    "Grid/Sensor": "1",
    "Grid/3D": "false",
    "Grid/RangeMax": "3.0",
    "Grid/RayTracing": "true",
    "RGBD/NeighborLinkRefining": "false",
}

# The database is the map. Beside a known map it is kept across restarts (the bridge watch
# restarts this container whenever the board's bridge is new, and a launch that wiped it lost
# the map every time); a SLAM session starts empty by default and writes a file of its own, so
# an evening of mapping can never delete the graph the known-map mode accumulated.
# The camera as a third odometry (rtabmap_odom's rgbd_odometry, ros/params/ekf.yaml's odom1).
# Measured on this laptop 2026-09-14 at rest, on /camera/image + /camera/depth + the camera's
# own CameraInfo: 9.1 poses/s out of a ~8-10 Hz depth stream, 20 ms of CPU a frame (p90 25 ms,
# a quarter of one core), 162 ms median from the image's stamp to the pose, 630 inlier features
# a frame, and 0.48 cm / 0.075 deg of drift over 85 s with the wheels reporting a hard zero
# (scratch/vo_probe.py, one lost frame: the first). Reg/Force3DoF (the whole stack is planar;
# every other table here says the same) and Odom/ResetCountdown — a lost tracking re-initialises
# instead of staying lost, and the jump that costs is dropped by pepin.visual_odometry.VoGate,
# which then measures from the new origin — were measured with this table as it stands: 9.4-9.7
# poses/s through the gate, none dropped, 0.2 cm and 0.0 deg of drift over 60 s at rest.
VO_RAW_TOPIC = "/vo/raw"  # rgbd_odometry's own output; /vo is what the gate publishes for the EKF
VISUAL_ODOMETRY = {
    # The EKF owns odom -> base_link. This node names its frame "odom" because that is the frame
    # its poses are differences in, and publishes no transform at all.
    "odom_frame_id": "odom",
    "publish_tf": False,
    # No guess from TF: a visual odometry seeded with the filter's own answer is not a third
    # opinion, and the loop would hide exactly the slip it exists to catch.
    "guess_frame_id": "",
    # All three streams share a stamp by construction: depth_stream publishes /camera/depth
    # carrying the exact header.stamp and frame_id of the /camera/image it was computed from
    # (depth_stream.py, the only publish there), and camera_stream stamps /camera/camera_info
    # with the picture's stamp in the same call. Measured on the wire 2026-09-14 with the whole
    # stack up (scratch/vo_stamp_pairs.py, 120 s): the picture and its CameraInfo at 11.6 Hz,
    # the depth at 8.5 Hz, and 1024 of 1024 depth frames plus 1397 of 1397 CameraInfos had a
    # bit-equal image stamp among the images received — nearest-image offset 0.0 ms at the
    # median, at p90 and at the worst. The depth is not stamped differently, it merely ARRIVES
    # 77 ms later (p90 119 ms, worst 497 ms), and sync_queue_size below holds 2.6 s of pictures
    # at that rate, five times the worst lag. So there is exactly one correct pair per depth
    # frame and nothing to approximate. ApproximateTime, which this table used to ask for on the
    # false claim that "they never share a stamp", took the NEIGHBOURING picture 213 times in
    # 6 h (rgbd_odometry's own "the time difference between rgb and depth frames is high":
    # offset median 0.101 s, p90 0.267 s, worst 0.997 s — one to nine frames off), and a pose
    # built from two different moments is the "jump of 11 cm in 0.100 s" the gate drops. Exact
    # sync cannot make that pair at all. Measured side by side for 5 min, a second rgbd_odometry
    # on /vo/exact against the launch's own on /vo/raw (scratch/vo_exact_node.sh,
    # scratch/vo_exact_ab.py): 8.92 vs 8.91 poses/s, 0 lost frames either way, the same gap
    # profile (median 99 ms, p90 200 ms, worst 1.10 s, 14 gaps over 0.5 s), exact sync covering
    # 8 camera stamps the approximate one missed against 3 the other way. No mis-pairing fired
    # in that window, so the rate is parity; what changes is that the mis-pairing is now
    # impossible. RTAB-Map's own node below keeps approx_sync: it syncs /scan and the board's
    # odometry too, which carry the board's stamps and share none of ours, and it has never
    # logged this warning.
    # What this does NOT fix is the "no pose on /vo/raw" starvation that the same warning was
    # blamed for. Caught live 2026-09-14 15:21 with both nodes running: BOTH published exactly
    # zero poses for three minutes, and the reason is rgbd_odometry's other warning — "Could not
    # find a connection between 'base_link' and 'camera_optical' ... Tf has two or more
    # unconnected trees". The board's bridge had gone silent (bridge_watch: /odom, /scan and the
    # board's /tf all at 0.0 Hz), so base_link was not in the tree and no pair of any kind could
    # become a pose; the depth had also fallen to 3.3 frames/s against the picture's 11.5, which
    # is when the approximation's guesses were worst (0.4-0.5 s off). Starvation is a bridge
    # outage and belongs to bridge_watch; the mis-pairing is this table's business.
    "approx_sync": False,
    "sync_queue_size": 30,
    "topic_queue_size": 10,
    "wait_for_transform": 0.5,
    # A lost frame is published (9999 on the covariance diagonal) rather than swallowed: the
    # gate drops it and the report line counts it, which is how a blind minute becomes visible.
    "publish_null_when_lost": True,
    "Reg/Force3DoF": "true",
    "Odom/ResetCountdown": "1",
}

KNOWN_MAP_DATABASE = "/maps/rtabmap.db"
SLAM_DATABASE = "/maps/rtabmap_slam.db"


# A node that exits comes back by itself after this pause, so a code change costs one kicked
# process (ros/laptop.sh kick <module>: SIGINT, the signal the launch itself sends at shutdown)
# instead of a container restart. A kicked node leaves DDS properly (its main destroys the node
# and the context, the participant is disposed, the bridge forgets the name at once) and the
# pause starts at its exit, so the successor never overlaps a ghost. A CRASHED node does (nothing
# disposed, the name outlives it by the DDS lease, the bridge drops its routes when the ghost
# expires), so every respawned command starts through a ghost wait of its own name
# (``_after_ghost``). A container is stopped with SIGINT (ros/laptop.sh, --stop-signal) so the
# launch shuts its nodes down properly; the ghost wait at the top covers what still lingers.
# The watches below are not respawned: their exit IS the signal (bridge_watch -> shutdown,
# ghost_wait -> start the nodes).
RESPAWN = {"respawn": True, "respawn_delay": 2.0}


def _after_ghost(*names: str) -> list:  # type: ignore[type-arg]
    """A command prefix that waits until the laptop's bridge lists none of ``names`` and then
    becomes the command (pepin_bringup.ghost_wait; an unreachable admin is not waited for)."""
    return [
        "python3 -m pepin_bringup.ghost_wait ",
        LaunchConfiguration("bridge_admin"),
        f" {' '.join(names)} --",
    ]


def rtabmap_parameters(
    slam: bool, camera_only: bool, graph_odom: bool = True, neighbor_refining: bool = False
) -> dict[str, object]:
    """Everything RTAB-Map is told for one mode: the common table under the mode's frames and
    grid. ``camera_only`` is read in SLAM mode alone — beside a known map the lidar is what
    makes the graph metric; ``graph_odom`` is read there alone too (false puts the graph back on
    the tracker's pose, :data:`TRACKER_ODOM`), and so is ``neighbor_refining`` (true puts ICP's
    own covariance back on the neighbour links, on which no loop closure survives the error-ratio
    check — see :data:`KNOWN_MAP`)."""
    mode: dict[str, object] = dict(KNOWN_MAP) if graph_odom else {**KNOWN_MAP, **TRACKER_ODOM}
    if not slam and neighbor_refining:
        mode["RGBD/NeighborLinkRefining"] = "true"
    if slam:
        mode = {**SLAM, **(SLAM_CAMERA_ONLY if camera_only else SLAM_LIDAR)}
    return {**RTABMAP, **TF_ODOMETRY_VARIANCE, **mode}


def _flag(context: LaunchContext, name: str) -> bool:
    """A launch argument as a bool, the way every other launch here reads one."""
    return LaunchConfiguration(name).perform(context).lower() == "true"


def _describe(context: LaunchContext) -> list:  # type: ignore[type-arg]
    """The nodes of this launch, once the arguments have values."""
    board = LaunchConfiguration("board")
    slam = _flag(context, "slam")
    camera_only = _flag(context, "camera_only")
    resume = _flag(context, "resume")
    world_map = _flag(context, "world_map")
    graph_odom = _flag(context, "graph_odom")
    neighbor_refining = _flag(context, "neighbor_refining")
    mode = "slam" if slam else "vision"
    # Whether the fused volume may be /map at all: the mode's owner (pepin.deployment) and the
    # launch's own world_map, the two halves the node checks before it publishes anything.
    volume_owns_map = world_map and map_owner(mode) == "laptop"
    seed_map = LaunchConfiguration("seed_map").perform(context)
    database = LaunchConfiguration("database").perform(context) or (
        SLAM_DATABASE if slam else KNOWN_MAP_DATABASE
    )
    # The bridge keeps routes by node name: the nodes start only once it has forgotten the
    # previous incarnation of this launch (pepin_bringup.ghost_wait), or RTAB-Map's /scan and
    # map routes die with the ghost ten seconds after they were made (2026-09-10).
    ghost_wait = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "pepin_bringup.ghost_wait",
            LaunchConfiguration("bridge_admin"),
            *laptop_launch_nodes("slam"),
        ],
        output="screen",
    )
    depth = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "pepin_bringup.depth_stream",
            "--ros-args",
            "-p",
            ["board:=", board],
        ],
        output="screen",
        prefix=_after_ghost("/depth_stream"),
        **RESPAWN,
    )
    # The same depth read at the floor instead of above it (pepin_bringup.contact_scan): where
    # the floor ENDS in each image column is where a body touches it — chair legs, a low box, a
    # plinth — and that range is geometry, not the network's scale. It goes to the board as
    # /contact_scan for the local costmap's contact_layer (off until validated on the robot).
    contact = ExecuteProcess(
        cmd=["python3", "-m", "pepin_bringup.contact_scan"],
        output="screen",
        prefix=_after_ghost("/contact_scan"),
        **RESPAWN,
    )
    # The map matching the board cannot afford, run here (pepin_bringup.laptop_localizer): the
    # whole-map search once a second on the board's /scan against the board's /map, whose answer
    # goes back over the bridge as a candidate the tracker judges (pepin.watchdog), and the
    # camera's own scans matched around the board's belief, whose answer goes back as a pose
    # measurement the tracker fuses (pepin.measurements). A tenth of a second here, a few seconds
    # on the board — which is why the board only ever asked once it was already lost — and a few
    # milliseconds against the 147 ms a camera scan cost the board's tracker.
    # The search is off in SLAM mode: there the map is RTAB-Map's, it is still being built, and
    # there is no saved map to search. Live either way: ros/flags.sh set laptop_localizer
    # global_watch on.
    watch = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "pepin_bringup.laptop_localizer",
            "--ros-args",
            "-p",
            f"global_watch:={'false' if slam else 'true'}",
        ],
        output="screen",
        prefix=_after_ghost("/laptop_localizer"),
        **RESPAWN,
    )
    # RTAB-Map's correction, put where the mode needs it (pepin_bringup.rtabmap_frame): beside a
    # known map it is map -> rtabmap on this laptop's own /tf, so the voxels and the cart stay
    # together after a closure; in SLAM it is map -> odom and goes to the board as a message.
    frame = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "pepin_bringup.rtabmap_frame",
            "--ros-args",
            "-p",
            f"slam:={'true' if slam else 'false'}",
            "-p",
            f"graph_odom:={'true' if graph_odom else 'false'}",
        ],
        output="screen",
        prefix=_after_ghost("/rtabmap_frame"),
        **RESPAWN,
    )
    # The operator's Foxglove connects here for the 3D view: the cloud stays on the laptop and
    # the board's topics arrive over the bridge, so nothing crosses the WiFi twice.
    foxglove = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        output="screen",
        parameters=[{"port": 8765, "address": "0.0.0.0", "send_buffer_limit": 100_000_000}],
        prefix=_after_ghost("/foxglove_bridge"),
        **RESPAWN,
    )
    camera = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "pepin_bringup.camera_stream",
            "--ros-args",
            "-p",
            ["board:=", board],
            "-p",
            ["static_camera_tf:=", LaunchConfiguration("static_camera_tf")],
        ],
        output="screen",
        prefix=_after_ghost("/camera_stream"),
        **RESPAWN,
    )
    # The same frames fused into one surface (pepin_bringup.depth_fusion): RTAB-Map keeps the
    # graph, the closures and the place recognition; the voxels the operator looks at come from
    # here, beside RTAB-Map's own cloud for comparison.
    # The volume is also the map (pepin.worldmap): the node is told which bridge mode the stack
    # is in AND whether it was brought up for the world map, because exactly one side may
    # publish /map and with world_map:=true it is the laptop's volume rather than RTAB-Map's
    # grid (see the remapping below). Both are launch decisions and the node refuses the volume
    # on /map without them, so no live flag can put a second publisher there.
    fusion = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "pepin_bringup.depth_fusion",
            "--ros-args",
            "-p",
            f"mode:={mode}",
            "-p",
            f"world_map:={'true' if world_map else 'false'}",
            # The flag comes up already saying where /map will come from, computed from the same
            # table the node checks (pepin.deployment.map_owner): volume only where this side
            # owns /map AND the launch kept RTAB-Map's grid off it. The operator had to set this
            # live in the first world-map session; a launch argument is not a thing to remember.
            "-p",
            f"map_source:={'volume' if volume_owns_map else 'file'}",
            # No tracker runs in SLAM mode, so /localization_fit never comes and a gate waiting
            # for it fused 0 frames of the first session (2026-09-13 14:05). It stays live: the
            # operator can put it back on with ros/flags.sh set depth_fusion fit_gate true.
            "-p",
            f"fit_gate:={'false' if slam else 'true'}",
        ]
        # The served map the volume's lidar layer starts as, and whose cell lattice the volume's
        # grid is snapped to. A known room is a seeded volume and nothing else — and a slice
        # seeded from the file IS that file, cell for cell (scratch/volume_vs_pgm.py).
        # Only when there IS one: rcl refuses to parse an override with an empty value
        # ("Couldn't parse parameter override rule: '-p seed_map:='"), and the node would die
        # at rclpy.init on every unseeded launch — which is every launch there has ever been.
        + (["-p", f"seed_map:={seed_map}"] if seed_map else []),
        output="screen",
        prefix=_after_ghost("/depth_fusion"),
        **RESPAWN,
    )
    # The camera's own opinion of how the cart moved: rtabmap's rgbd_odometry on the same three
    # topics RTAB-Map itself reads, and pepin_bringup.visual_odometry between it and the board's
    # EKF (the gate, the covariance, the drift at rest). Off (vo:=false) neither runs and the
    # board's odometry is the wheels and the gyro exactly as before. It never publishes a
    # transform: the EKF owns odom -> base_link, and it takes no guess from TF either
    # (guess_frame_id empty), so this measurement stays independent of the filter it feeds.
    rgbd_odometry = Node(
        package="rtabmap_odom",
        executable="rgbd_odometry",
        name="rgbd_odometry",
        output="screen",
        parameters=[{"frame_id": "base_link", **VISUAL_ODOMETRY}],
        remappings=[
            ("rgb/image", "/camera/image"),
            ("rgb/camera_info", "/camera/camera_info"),
            ("depth/image", "/camera/depth"),
            ("odom", VO_RAW_TOPIC),
        ],
        condition=IfCondition(LaunchConfiguration("vo")),
        prefix=_after_ghost("/rgbd_odometry"),
        **RESPAWN,
    )
    vo = ExecuteProcess(
        cmd=["python3", "-m", "pepin_bringup.visual_odometry"],
        output="screen",
        prefix=_after_ghost("/visual_odometry"),
        condition=IfCondition(LaunchConfiguration("vo")),
        **RESPAWN,
    )
    remappings = [
        ("rgb/image", "/camera/image"),
        ("rgb/camera_info", "/camera/camera_info"),
        ("depth/image", "/camera/depth"),
        ("scan", "/scan"),
    ]
    if slam and not world_map:
        # In SLAM mode this grid IS the map: /map, transient local, read by the board's global
        # costmap over the bridge. Beside a known map it stays /rtabmap/map, out of Nav2's way.
        # With world_map the volume publishes /map instead, and RTAB-Map keeps its own name:
        # two publishers of /map is the failure this whole table exists to prevent.
        remappings.append(("map", "/map"))
    rtabmap = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        # Its outputs are relative names (map, mapGraph, mapPath, info): without a namespace
        # its "map" landed on /map next to the board's static map and fed the laptop's
        # global costmap a second, growing map (2026-09-10 01:00). In SLAM mode there IS no
        # second map and the grid is remapped onto /map on purpose (see remappings).
        namespace="rtabmap",
        output="screen",
        parameters=[
            {
                "frame_id": "base_link",
                # Never into anyone's tree: beside a known map the tracker owns map -> odom, and
                # in SLAM the board's slam_frame does, from the correction sent to it. One owner.
                "publish_tf": False,
                "database_path": database,
                # An empty start is explicit: a SLAM session is a new map unless resume:=true,
                # and the known-map database is only ever wiped by ros/laptop.sh vslam --fresh.
                "delete_db_on_start": slam and not resume,
                "subscribe_depth": True,  # rgb + depth + camera_info from the camera nodes
                "subscribe_rgb": False,
                "approx_sync": True,
                "sync_queue_size": 30,
                "topic_queue_size": 10,
                "wait_for_transform": 0.5,
                "odom_sensor_sync": False,
                # the grid is republished every second: the operator watches it grow
                "map_always_update": True,
                **rtabmap_parameters(slam, camera_only, graph_odom, neighbor_refining),
            }
        ],
        remappings=remappings,
    )
    scan = "camera only (no /scan)" if camera_only else "lidar + camera"
    start = "resumed" if resume else "empty"
    grid = "the fused volume" if volume_owns_map else "RTAB-Map's grid"
    vo_note = (
        f" rgbd_odometry -> {VO_RAW_TOPIC} -> /vo for the board's EKF (withheld until"
        " visual_odometry's vo_publish is on)"
        if _flag(context, "vo")
        else " no visual odometry (vo:=false): the board's odometry is the wheels and the gyro"
    )
    report = (
        f"vslam up: online SLAM, {scan}, {start} database {database}, /map from {grid};"
        " the fusion's fit_gate is off (no tracker publishes a fit here) and"
        " the global watch is off (the map is being built);" + vo_note
        if slam
        else f"vslam up: beside the known map, lidar + camera, database {database};"
        " the laptop localizer proposes a place once a second and measures the camera's pose"
        f" at 5 Hz for the board's tracker; neighbor_refining="
        f"{'on' if neighbor_refining else 'off'} (off: the neighbour links carry the odometry's"
        " own covariance, so a loop closure has somewhere to go);" + vo_note
    )
    return [
        LogInfo(msg=report),
        ghost_wait,
        RegisterEventHandler(
            OnProcessExit(
                target_action=ghost_wait,
                on_exit=[
                    camera,
                    depth,
                    contact,
                    fusion,
                    watch,
                    rtabmap,
                    frame,
                    foxglove,
                    rgbd_odometry,
                    vo,
                ],
            )
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            *SHUTDOWN,
            DeclareLaunchArgument("board", default_value="10.0.0.187"),
            DeclareLaunchArgument("slam", default_value="false"),
            DeclareLaunchArgument("camera_only", default_value="false"),  # SLAM mode only
            DeclareLaunchArgument("resume", default_value="false"),  # SLAM mode only
            # Known-map mode only: whose odometry the graph is built on (see KNOWN_MAP and
            # TRACKER_ODOM). A mode, not a tunable — it decides which frame every node of the
            # database was placed in, so it is set at a launch and never mid-session.
            DeclareLaunchArgument("graph_odom", default_value="true"),
            # Known-map mode only: ICP refines the neighbour links and its own covariance
            # comes with them (RGBD/NeighborLinkRefining, see KNOWN_MAP). Default false
            # since 2026-09-14: refined links are stiffer than the odometry they replace and
            # RGBD/OptimizeMaxError then rejects every closure the graph finds.
            DeclareLaunchArgument("neighbor_refining", default_value="false"),
            # The volume is /map instead of RTAB-Map's grid (pepin_bringup.depth_fusion)
            DeclareLaunchArgument("world_map", default_value="false"),
            # The map the fused volume's lidar layer is seeded from (a map_server yaml as the
            # container sees it, e.g. /maps/flat3_straight.yaml); empty seeds nothing.
            DeclareLaunchArgument("seed_map", default_value=""),
            # The camera as a third odometry: rtabmap_odom's rgbd_odometry and the node that
            # gates it (pepin_bringup.visual_odometry). On by default because it is measured at
            # rest and costs only this laptop (0.25 core); what it costs the ROBOT is still
            # nothing until visual_odometry's vo_publish flag is turned on.
            DeclareLaunchArgument("vo", default_value="true"),
            DeclareLaunchArgument("database", default_value=""),  # empty: by mode
            DeclareLaunchArgument("bridge_admin", default_value="http://pepin-zenoh:8000"),
            DeclareLaunchArgument("static_camera_tf", default_value="true"),
            # A new board bridge means new subscriptions are needed: the watch exits, the launch
            # shuts down, the container's restart policy brings this half back.
            ExecuteProcess(
                cmd=[
                    "python3",
                    "-m",
                    "pepin_bringup.bridge_watch",
                    LaunchConfiguration("board"),
                    LaunchConfiguration("bridge_admin"),  # this side's bridge: the flow watch
                ],
                output="screen",
                on_exit=[Shutdown(reason="the board's bridge restarted")],
            ),
            OpaqueFunction(function=_describe),
        ]
    )
