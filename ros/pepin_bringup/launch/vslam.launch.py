"""Camera + lidar mapping on the laptop: RTAB-Map on ONE snapshot of whatever is looking.

WORLD R, the decision this launch now carries. RTAB-Map's loop-closed graph is the one source of
truth about the room. There is ONE frame: RTAB-Map's map frame IS ``map`` (``map_frame_id: map``,
in every situation). The board's tracker owns ``map -> odom``, so RTAB-Map must never publish that
transform (``publish_tf`` false, in every situation) — its correction reaches the board as a
MEASUREMENT and as its occupancy grid. And RTAB-Map's odometry stays the EKF's
``odom -> base_link`` over the bridge (``odom_frame_id: odom``), never the tracker's pose: that
pose teleports when it relocalises, a neighbour edge between two nodes one second apart then
carried 0.888 m against its 0.244 m sigma, and on that 3.64 error ratio RTAB-Map rejected EVERY
loop closure it found for three hours (2026-09-10..14; the argument that used to put it back,
``graph_odom``, is gone with the table it selected).

ONE INPUT, ONE TABLE. RTAB-Map used to subscribe to a synchronised triple — picture, depth, scan —
and to carry THREE parameter tables by mode: the grid from the scan and the depth beside a known
map, from the scan alone in lidar SLAM, from the depth alone with the lidar unplugged. Both were
the same mistake. The triple meant that when the camera stopped, the synchroniser stopped firing
and RTAB-Map starved while the lidar went on delivering ten revolutions a second; and a "mode" is
nothing but which sensors are alive, which belongs in the data. So
pepin_bringup.sensor_pack assembles one ``rtabmap_msgs/SensorData`` per moment out of whatever is
fresh — picture + depth + optics, the scan, either alone — and RTAB-Map reads that single topic
(``subscribe_sensor_data``, mutually exclusive with every other subscription:
rtabmap_sync/CommonDataSubscriber.cpp:453-509 of 0.22.1-jazzy). One table below serves a node with
a scan, a node with a picture and a node with both.

WHAT THE ONE TABLE HAD TO SETTLE, and where each answer was read, is written beside the parameter
it settles (:data:`RTABMAP`). The three that are new: the grid is built from BOTH sensors per node
(``Grid/Sensor`` 2) so a node with no scan still contributes from the depth and a node with no
picture still contributes from the scan; ``RGBD/ProximityPathMaxNeighbors`` has to be SAID now,
because the wrapper only inserted it while a scan was subscribed; and a node with no picture is a
"bad signature" that ``Mem/BadSignaturesIgnored`` decides the fate of.

THE CAMERA'S REACH IS THE CAMERA'S BUSINESS, not the grid's. One ``Grid/RangeMax`` now serves both
sensors, and it is the LIDAR's (8 m): the camera's depth arrives already NaN past the range this
network's scale is still a measurement over (pepin_bringup.depth_stream's ``depth_reach``), and a
NaN pixel makes no point in RTAB-Map's cloud at all (rtabmap/core/util3d.cpp:644). That is what
keeps "the grid from both sensors" from putting camera marks behind a wall the lidar sees — 44 % of
the camera's costmap marks were behind it with un-gated depth (2026-09-11).

THERE IS NO SLAM MODE ANY MORE (World R, 2026-09-19). Mapping a new room and driving a known one
were never two arrangements of this launch: they are one database that is either empty or full. So
there is ONE database (:data:`DATABASE`), it is never wiped by this launch (only by
``ros/laptop.sh vslam --fresh``), its grid always goes to ``/map`` — the one map, which the board's
tracker adopts and republishes for the costmaps — and the session decides one word: a database that
does not exist yet starts in mapping mode, because ``Mem/IncrementalMemory false`` on an empty
database is mapping with the mapping switched off (:func:`rtabmap_memory`). ``camera_only:=true`` is
what it always meant: sensor_pack's ``sources`` is the camera alone, so the nodes carry no scan and
the grid comes from the depth — the honest test of the camera as the primary sense, and the one case
where the map is only as true as the network's scale.

THE CAMERA AS A THIRD ODOMETRY (``vo``, on by default). Beside all of that, rtabmap_odom's
``rgbd_odometry`` reads the same picture and depth — on its OWN subscriptions, which is right:
it legitimately dies with the camera — and answers with a pose per frame, and
pepin_bringup.visual_odometry gates it, gives it a documented covariance and publishes ``/vo``
for the board's EKF (ros/params/ekf.yaml's ``odom1``: x and y differentially, no yaw — the gyro
owns heading). Nothing reaches the filter until that node's ``vo_publish`` flag is on; with
``vo:=false`` neither process starts at all.

Arguments: ``board`` (the robot's address for the camera stream), ``camera_only``
(sensor_pack's ``sources``), ``sensor_pack`` (false: the way back to the synchronised
triple — this node does not start and RTAB-Map subscribes to the picture, the depth and the scan
itself, :data:`TRIPLE_SUBSCRIPTIONS`), ``neighbor_refining`` (whether ICP refines the neighbour
links and stiffens them with its own covariance — false, or no closure the graph finds survives
RGBD/OptimizeMaxError), ``memory`` (beside a LOADED database: ``trust`` — start LOCALISING and let
pepin_bringup.rtabmap_frame move the mode live on trust in the pose — ``map``, or ``localise``),
``vo``, ``database`` (empty: :data:`DATABASE`), ``bridge_admin`` (the laptop bridge's REST
admin, asked whether it still lists this launch's previous incarnation), ``static_camera_tf``
(default true: the camera node broadcasts base_link -> camera_link from config/camera.json; false
when the board's neck node publishes that edge live — ros/feature.sh neck on, ``ros/laptop.sh
vslam --neck`` — since two publishers of one edge fight), ``camera`` (which rig of
config/camera.json the head is; empty, the default, means that file's own ``"active"`` or
``PEPIN_CAMERA`` — :func:`camera_rig` resolves it once for the whole launch and the report line
names it).
"""

import json
from pathlib import Path

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

from pepin.camera import active_camera
from pepin.deployment import (
    CONTAINER_STOP_TIMEOUT_S,
    config_file,
    laptop_launch_nodes,
    rmw_is_zenoh,
)

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

# EVERYTHING RTAB-MAP IS TOLD, for every situation: the input, the frames, how a transform between
# two nodes is found, how the graph is built and closed, what a place looks like, and what the grid
# is made of. There is no second table — a node with a scan, a node with a picture and a node with
# both are all served from here, because which sensors a node has is now data
# (pepin_bringup.sensor_pack) and not a config.
RTABMAP = {
    # ---- the input: ONE message, not three subscriptions -------------------------------------
    # subscribe_sensor_data takes a whole rtabmap_msgs/SensorData — picture, depth, optics, the
    # camera's local transform, the scan and its own local transform — and is mutually exclusive
    # with subscribe_depth / subscribe_rgb / subscribe_stereo / subscribe_rgbd / subscribe_scan /
    # subscribe_scan_cloud, each of which rtabmap turns OFF with a warning when this is on
    # (rtabmap_sync/CommonDataSubscriber.cpp:453-509). They are said false here anyway: a reader of
    # this table must not have to know the precedence, and TRIPLE_SUBSCRIPTIONS puts them back.
    "subscribe_sensor_data": True,
    "subscribe_depth": False,
    "subscribe_rgb": False,
    "subscribe_scan": False,
    # No odometry topic: the odometry is read from TF (odom_frame_id below), which is what makes
    # the snapshot the ONLY message that has to arrive on time. With this true there would be a
    # synchroniser between /odom and the snapshots and the camera's starvation would be back in
    # another shape (CommonDataSubscriberSensorData.cpp:70-112: with both false there is no
    # synchroniser at all, just a plain subscription).
    "subscribe_odom": False,
    # ---- the frames: World R's one frame, owned by nobody here --------------------------------
    # RTAB-Map's own map frame IS map, and it publishes no transform into it: the board's tracker
    # owns map -> odom and this graph's correction travels as a measurement and as a grid.
    "map_frame_id": "map",
    # The EKF's odom -> base_link over the bridge — the same continuous odometry every other
    # consumer rides, and never the tracker's pose (see the module docstring, 2026-09-14).
    "odom_frame_id": "odom",
    # registration: ICP on the 2D scans in the plane; the camera only says "I have been here".
    # WHY 1 (Icp) AND NOT 2 (VisIcp), read in the sources rather than guessed. Strategy 2 is
    # RegistrationVis with RegistrationIcp as its CHILD, and a child's answer REPLACES the
    # parent's: if Vis succeeds and ICP then finds no scan it returns null and the whole
    # registration fails (rtabmap/core/Registration.cpp:209-220 with RegistrationIcp.cpp:459,
    # where a missing scan leaves the transform null). Worse, with an image-requiring pipeline
    # Memory never even calls the pipeline unless BOTH nodes have words
    # (rtabmap/core/Memory.cpp:2927-2929) — so under strategy 2 a node with no picture, which is
    # exactly what a lidar-only snapshot makes, can never be linked at all. Under strategy 1 the
    # third clause of that same condition (``!guess.isNull() && !isImageRequired()``) lets EVERY
    # pair through and ICP registers whatever scans are there, which is why the identity guess
    # below is what makes a closure possible. What strategy 1 cannot do either is link a node with
    # no SCAN: ICP has nothing to match and Vis is not in the pipeline. A camera-only node
    # therefore still carries its words and its depth, is still recognised by appearance, and
    # still gets its neighbour link from the odometry — but contributes no metric closure. That is
    # the honest limit of one table, and the live check for it is named in the report.
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
    #   Under Reg/Strategy 1 this is also what lets the pipeline run AT ALL on a pair where one
    # node has no words: Memory only reaches the registration through
    # ``!guess.isNull() && !isImageRequired()`` (Memory.cpp:2929), and with the guess null it
    # would instead ask its private RegistrationVis for a seed (Memory.cpp:2964-2970) — the very
    # visual step that has never succeeded on this camera.
    "RGBD/LoopClosureIdentityGuess": "true",
    # Proximity detection by space also by MERGING the close scans of a path, not only one to one.
    # It has to be said here now, and that is a real consequence of the new input: rtabmap_slam's
    # wrapper inserts this 10 itself, but only ``if(this->isSubscribedToScan2d() || ...)`` and only
    # when Reg/Strategy is ICP (CoreWrapper.cpp:489-504). With subscribe_sensor_data the wrapper
    # sees no scan subscription, the insert never fires, and the parameter falls back to its
    # default 0 — which DISABLES one-to-many proximity detection, the path most of this graph's
    # proximity links come from. Same value the wrapper used to insert, so the graph is unchanged.
    "RGBD/ProximityPathMaxNeighbors": "10",
    # A node with NO PICTURE — which is what a lidar-only snapshot makes — is a "bad signature":
    # isBadSignature() is exactly "no visual words" (rtabmap/core/Signature.cpp:341-344). This
    # false is what KEEPS it: Memory::cleanup() moves the last signature to the trash only when
    # the flag is true (Memory.cpp:2123). It is also RTAB-Map's own default
    # (Parameters.h:222), said out loud because World R depends on it — a lidar-only node must
    # stay in the graph, where proximity detection by space still links it by its scan (that path
    # is not gated on the signature being good: Rtabmap.cpp:2631-2634, unlike the appearance-based
    # global closure at Rtabmap.cpp:1971, which a node with no picture cannot take part in).
    "Mem/BadSignaturesIgnored": "false",
    # WHY THE NEIGHBOUR LINKS ARE NOT REFINED (2026-09-14, the second rejection disease; back with
    # ``neighbor_refining:=true``). With the graph on the EKF's odometry RTAB-Map finds its
    # closures — 5 or 6 an iteration, 8459 against 1, 1137, 1462, 1813, 2398, registered with 211
    # visual inliers against a Vis/MinInliers of 20 — and threw every one of them away on
    # RGBD/OptimizeMaxError, which compares each link's residual after optimisation with that
    # link's own standard deviation:
    #   "Rejecting all added loop closures (5, first is 8459 <-> 1) ... maximum graph error ratio
    #    11.737604 (edge 7083->7084, type=0, abs error=0.217136 m, stddev=0.018499)"
    #   "... ratio 3.530088 (edge 962->1023, type=0, abs error=0.435329 deg, stddev=0.002152)"
    # Both culprits are NEIGHBOUR links (type=0), and the stddev is why. Refined, a neighbour link
    # carries ICP's own fit residual as its covariance, and ICP on two scans of a flat is sure of
    # itself: over the 607 neighbour links in ros/maps/rtabmap.db the median sigma is 0.75 cm and
    # 0.135 deg (scratch/graph_link_sigmas.py). At OptimizeMaxError's 3.0 that means the optimised
    # graph may not disagree with ANY neighbour link by more than 2.2 cm or 0.40 deg — while a
    # closure across a session boundary asks for tens of centimetres by construction, and the whole
    # correction lands on whichever link joins the two parts (node 7083 and 7084 are one second and
    # 0.4 cm apart; nothing is wrong with that edge, it is simply the hinge). The check could
    # therefore never accept a closure on this graph, however right the closure was.
    # Unrefined, the link carries the ODOMETRY's uncertainty instead (:data:`TF_ODOMETRY_VARIANCE`),
    # which is what the check was written to compare against — how far the chain may have drifted,
    # not how well two clouds overlay. Nothing that decides whether a closure is CORRECT is
    # touched: Vis/MinInliers, the ICP checks and OptimizeMaxError itself all stand. What is given
    # up is the scan's correction of the wheels between two nodes 5 cm apart — a centimetre of
    # local metric accuracy, against a graph that can close a loop at all. Measured beside the
    # known map's database; a fresh SLAM database used to ship with it true, and that is the one
    # value of this table a SLAM session gives up on purpose, because the failure is total (no
    # closure ever) and the cost is a centimetre.
    # The START value, for a cart whose lidar is alive; at run time it FOLLOWS what the snapshots
    # carry beside Reg/Strategy and Grid/Sensor (pepin.graphmode.REGISTRATION_PARAMETERS). True
    # while there is a scan: unrefined, a parked cart's map turned +27 deg in 40 min with the
    # gyro's bias (2026-09-19); refined, it held to a degree. The cost recorded above still stands.
    # False since the gyro's bias is tracked at rest (pepin.graphmode has the measurement): refined
    # links made RGBD/OptimizeMaxError reject 87 closures on the first real drive.
    "RGBD/NeighborLinkRefining": "false",
    "Rtabmap/DetectionRate": "1.0",
    # appearance: GFTT/ORB words, a few hundred per image
    "Kp/DetectorStrategy": "8",
    "Kp/MaxFeatures": "400",
    # ONE FRAME ACROSS SESSIONS. Without this, every time the memory goes (back) to mapping beside
    # a loaded database RTAB-Map opens a NEW map rooted at the odometry's pose, unlinked to the old
    # one until some later closure: /rtabmap/mapGraph then carries that one new node, the grid is
    # re-rendered from it alone in the ODOMETRY's frame, and the tracker, the places and the
    # costmaps all find themselves in another frame. Live 2026-09-19 at the bookshelf: "places: 3
    # in the book, 0 the graph can place (68 graphs, 1 nodes)", the tracker at (-1.60, -1.13) —
    # the odometry's coordinates — with the cart standing on the place marked (-2.72, -1.22).
    # With it, a new map is started only ON a closure with the previous one, so its first node is
    # already tied to the old graph and the frame never changes.
    "Rtabmap/StartNewMapOnLoopClosure": "true",
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
    # Post 1, on purpose. RTAB-Map takes the depth's decimation from Mem/ImagePreDecimation, not
    # from Post (Memory.cpp:5711 warns and does nothing), so Post 2 stored nodes whose RGB was
    # 320x180 beside a 640x360 depth — and a node like that trips a fatal assert the day anything
    # reprojects it (util3d.cpp:324). Seen live 2026-09-19 as "Depth image is bigger than RGB image
    # after post decimation". The disk it cost is bounded by the memory rule now: nothing is
    # written while the database only localises.
    "Mem/ImagePostDecimation": "1",
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
    # ---- the grid: written by the sensor that can vouch at the grid's grade ---------------------
    # 0=scan, 1=depth, 2=both. The START value is the scan, and the value FOLLOWS what the
    # snapshots carry at run time beside Reg/Strategy (pepin.graphmode.REGISTRATION_PARAMETERS,
    # set by rtabmap_frame through update_parameters): a node with a scan gives the grid its scan,
    # a node without one adds nothing (the value does NOT follow the snapshots: a change
    # re-renders the WHOLE grid, 2026-09-19). "Both" (2) was tried first and measured on the parked
    # cart 2026-09-19: the mono network's smeared depth beside the lidar's walls made a newborn map
    # of ~2000 occupied cells on 5 x 4 m, the lidar tracker's match on it went flat (sigma 0.47 m /
    # 34 deg at fit 0.97, the pose wandering 10 cm and 4 deg in 90 s on a still map); from the scan
    # alone the same room is ~500 cells and the tracker reads 0.01 m / 0.1 deg. RTAB-Map's grid has
    # no per-sensor weight, so a sensor that cannot vouch at the lidar's grade must not write while
    # the lidar does. The camera's obstacles still reach Nav2 through the costmap's camera layer.
    "Grid/Sensor": "0",
    # A 2D grid, not voxels. With Grid/3D true the operator saw RTAB-Map's own cloud_map beside the
    # fusion's volume; under World R the grid is the thing the BOARD is handed, and the room in
    # three dimensions is pepin.worldmap's volume, which is built from the same depth.
    "Grid/3D": "false",
    # THE LIDAR's reach, for both sensors, because the camera's reach is now in the camera's data:
    # depth_stream publishes NaN past the range this network's scale is a measurement over, and a
    # NaN depth pixel is dropped from the cloud before the grid ever sees it
    # (pcl::isFinite in rtabmap/core/util3d.cpp:644, which fills the valid indices
    # cloudRGBFromSensorData hands to the grid). 8.0 is the flat's walls; the LD19's last metres
    # are noise, not geometry.
    "Grid/RangeMax": "8.0",
    # Free space has to be TRACED now, and that is the price of Grid/Sensor 2: the cheap 2D path
    # that carves a scan's own free space is taken only when Grid/Sensor is 0
    # (LocalGridMaker.cpp:198), so with 2 both halves go through the generic path, where empty
    # cells exist only under ``!grid3D_ && rayTracing_`` (LocalGridMaker.cpp:567-582). Without this
    # the grid would be walls and unknown, with no known floor for a costmap's static layer. It is
    # the 2D tracing, not the 3D one the known-map table refused as too expensive at 1 Hz — the
    # camera-only SLAM mode shipped with exactly this pair (3D false, RayTracing true) from
    # 2026-09-13, so it is the arrangement that has run, not a new one.
    "Grid/RayTracing": "true",
    # a mono network's depth frays at object edges and far away: lone voxels go
    "Grid/NoiseFilteringRadius": "0.10",
    "Grid/NoiseFilteringMinNeighbors": "5",
    "Grid/CellSize": "0.05",
    "Grid/DepthDecimation": "4",  # 160x90 depth samples a frame: plenty for 5 cm voxels
    "Grid/MaxGroundHeight": "0.08",
    "Grid/MaxObstacleHeight": "1.3",  # the band the costmap scan uses too
    "Grid/NormalsSegmentation": "false",  # height decides ground vs obstacle; the floor is flat
}

# A KNOWN ROOM IS LOCALISED IN, NOT RE-MAPPED (``localize``, the default beside a LOADED database
# since 2026-09-18). Everything here follows from what the parked cart measured that evening.
#
# Mem/IncrementalMemory false is the whole of it: nothing is written to the database. That removes
# the disease pepin.graphnodes exists to survive — the database stops growing, so no new session
# becomes a new PIECE, no optimisation re-roots the graph, and the frame that moved from (-9.17,
# -0.29, -147.5 deg) to (-11.10, +5.71, -27.8 deg) across one prune stops moving. Every entry of the
# node table stays true, and the 43 sessions no lidar-held drive has visited become a fixed,
# shrinking debt instead of a growing one.
#
# Mem/InitWMWithAllNodes true puts the whole database in the working memory at start, which is what
# makes a wake-up anywhere in the flat possible: without it RTAB-Map can only recognise what its
# memory management has paged in, and on the charger that is whatever the last session ended with.
#
# RGBD/LinearUpdate and RGBD/AngularUpdate 0 make a PARKED cart localise. With the defaults
# (0.1 m / 0.1 rad) Memory/Small_movement read 1 on every update at rest and RTAB-Map skipped the
# whole pipeline — measured live: not one update named a node until these were set to 0, and then
# every update did (proximity_detection_id 88429). A wake-up is a cart that has not moved, so this
# is not a tuning choice; it is the mode working at all.
#
# RGBD/OptimizeMaxError 0 (the check OFF) needs the argument spelled out, because it is the check
# that rejects a CORRECT recognition. Parked in mapping mode RTAB-Map found this very place —
# Loop/Highest_hypothesis_id 87418 at 0.978, Loop/Visual_inliers 328, ratio 0.41 — and threw it away
# itself: Loop/Optimization_max_ang_error_ratio 5.03 against the parameter's 3.0. That ratio is a
# statement about THE GRAPH and not about the recognition: accepting the closure would contradict
# the piecewise-inconsistent database (sessions 1.6 m and 129 deg apart,
# scratch/graph_tie_fit.py), so the check faithfully refuses every good recognition this database
# can produce. Here it costs
# nothing to switch off: with IncrementalMemory false the accepted link is never written, so a wrong
# recognition cannot pollute anything — it can only produce one wrong WORD, and judging a word is
# our job, done three times over (rtabmap_frame's 3-DOF chi-square against the odometry-carried
# belief, the board's own information-filter gate, and the whole-map candidate rules that need three
# agreeing pieces of evidence). The alternative — leaving it at 3.0 — is a camera that never speaks
# beside this database, which is what the last three days measured.
#
# WHAT IT COSTS rtabmap_frame: the ids of this run become TEMPORARY (nothing is written), so a new
# node may not be tabled — the node detects that by the id never appearing in /rtabmap/mapGraph and
# the table then grows only through RECOGNISED nodes. Recognition and the "new start" detection are
# unchanged (a matched id is still below this start's first ref, and a falling distance counter is
# still a restart); Memory/Distance_travelled still creeps from VO jitter at rest, which is why the
# word's sigma stopped reading it at all (graph_word_from_localization).
LOCALIZE = {
    "Mem/IncrementalMemory": "false",
    "Mem/InitWMWithAllNodes": "true",
    "RGBD/LinearUpdate": "0",
    "RGBD/AngularUpdate": "0",
    "RGBD/OptimizeMaxError": "0",
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

# THE WAY BACK (``sensor_pack:=false``, CLAUDE.md rule 19): RTAB-Map on the three subscriptions it
# read until 2026-09-19, so a regression in the snapshots is turned off in the field instead of
# reverted. Nothing else in the table moves — the mode tables are gone, and this arrangement ran
# for nine days with exactly the Grid/Sensor 2 grid above. What comes back with it is the bug the
# snapshots exist to fix: the synchroniser needs the picture, the depth AND the scan for one
# moment, so a camera that stops takes RTAB-Map down with it while the lidar keeps delivering.
TRIPLE_SUBSCRIPTIONS = {
    "subscribe_sensor_data": False,
    "subscribe_depth": True,  # rgb + depth + camera_info from the camera nodes
    "subscribe_scan": True,
    "approx_sync": True,  # the board's stamps and ours share nothing
    "sync_queue_size": 30,
    "odom_sensor_sync": False,
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

# THE DATABASE IS THE MAP, and there is one of it (World R): the room's graph, its closures and
# every node's local grid, kept across restarts — the bridge watch restarts this container whenever
# the board's bridge is new, and a launch that wiped the file lost the room every time. An empty
# room is this file absent, which is what ros/laptop.sh vslam --fresh makes (and restart.sh moves
# the volume of the old frame aside with it, since the volume is painted in the graph's frame).
DATABASE = "/maps/rtabmap.db"
# The one topic RTAB-Map reads. The same literal is pepin_bringup.sensor_pack's own
# SENSOR_DATA_TOPIC: one name written on both sides of the contract, so a test can pin it.
SENSOR_DATA_TOPIC = "/rtabmap/sensor_data"


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


def rtabmap_memory(memory: str, loaded: bool) -> str:
    """Which memory mode a session starts in: the ``memory`` argument beside a database that
    EXISTS, and always ``map`` beside one that does not.

    A database born in this same second has nothing in it to be recognised, and
    ``Mem/IncrementalMemory false`` there would mean a session that never writes a node — mapping
    with the mapping switched off. This is the one place a session still decides a parameter, it
    decides one word rather than a table, and it decides it from a fact on disk rather than from a
    mode somebody had to remember to pass."""
    return memory if loaded else "map"


def rtabmap_parameters(
    neighbor_refining: bool = False,
    memory: str = "trust",
    sensor_pack: bool = True,
) -> dict[str, object]:
    """Everything RTAB-Map is told, for every situation: :data:`RTABMAP` under the odometry links'
    covariance, and at most two overlays that are not modes.

    ``neighbor_refining`` true puts ICP's own covariance back on the neighbour links, on which no
    loop closure survives the error-ratio check (see :data:`RTABMAP`); ``sensor_pack`` false is the
    way back to the three subscriptions (:data:`TRIPLE_SUBSCRIPTIONS`).

    ``memory`` is the INITIAL mode and nothing more (already resolved by :func:`rtabmap_memory`
    when this is called): anything but ``map`` starts RTAB-Map localising (:data:`LOCALIZE`),
    because that is what a wake-up in a known room needs and because a database that is not written
    to cannot grow a new piece. From there pepin_bringup.rtabmap_frame owns the switch and moves it
    live on trust in the pose (``graph_memory``), calling RTAB-Map's own set_mode services — so this
    decides where a session begins, not where it stays."""
    table: dict[str, object] = dict(RTABMAP)
    table.update(TF_ODOMETRY_VARIANCE)
    if neighbor_refining:
        table["RGBD/NeighborLinkRefining"] = "true"
    if memory != "map":
        table.update(LOCALIZE)
    if not sensor_pack:
        table.update(TRIPLE_SUBSCRIPTIONS)
    return table


def _flag(context: LaunchContext, name: str) -> bool:
    """A launch argument as a bool, the way every other launch here reads one."""
    return LaunchConfiguration(name).perform(context).lower() == "true"


def camera_rig(name: str) -> str:
    """Which camera of ``config/camera.json`` this launch's nodes read: the ``camera`` argument
    if it says anything, else ``PEPIN_CAMERA`` in the container's environment (ros/laptop.sh
    passes it in), else the file's own ``"active"`` (pepin.camera.active_camera decides, here as
    everywhere).

    Resolved ONCE, here, and handed to the camera node, so the launch's report line names the
    rig that is actually being published and a typo stops the launch at start instead of leaving
    one node on another camera. The other readers of that file — the depth node, the contact
    scan — answer the same question from the same file and the same variable.
    """
    return active_camera(json.loads(config_file("camera.json").read_text()), name)


def _describe(context: LaunchContext) -> list:  # type: ignore[type-arg]
    """The nodes of this launch, once the arguments have values."""
    board = LaunchConfiguration("board")
    rig = camera_rig(LaunchConfiguration("camera").perform(context).strip())
    camera_only = _flag(context, "camera_only")
    resume_volume = _flag(context, "resume_volume")
    neighbor_refining = _flag(context, "neighbor_refining")
    packing = _flag(context, "sensor_pack")
    database = LaunchConfiguration("database").perform(context) or DATABASE
    # An empty room is a database that is not there yet, and that is the only thing the session
    # still decides (:func:`rtabmap_memory`): a file nobody has written cannot be localised in.
    loaded = Path(database).is_file()
    memory = rtabmap_memory(LaunchConfiguration("memory").perform(context).strip().lower(), loaded)
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
    # The search runs in every session now: there is one map, the board's tracker is always on it,
    # and a room still being mapped is exactly where a second opinion about the place is worth
    # having. It stays live: ros/flags.sh set laptop_localizer global_watch off.
    watch = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "pepin_bringup.laptop_localizer",
        ],
        output="screen",
        prefix=_after_ghost("/laptop_localizer"),
        **RESPAWN,
    )
    # What RTAB-Map's graph says about the cart's place (pepin_bringup.rtabmap_frame): one more
    # measurement for the tracker's fusion, in every session — the board's tracker owns map -> odom
    # and this is a word it weighs, never a correction it obeys. One switch is passed: which memory
    # mode the session begins in (the same word RTAB-Map itself is given).
    # The room's vocabulary, kept current against a graph that bends (pepin_bringup.places): a
    # place is the cart's pose relative to a labelled RTAB-Map node, so it rides the node when a
    # loop closes. The book lives beside the database whose ids it uses.
    places = ExecuteProcess(
        # No database override: the argument may be empty, and the node's own default is the
        # same /maps/rtabmap.db the launch falls back to (as for depth_fusion).
        cmd=["python3", "-m", "pepin_bringup.places"],
        output="screen",
        prefix=_after_ghost("/places"),
        **RESPAWN,
    )
    frame = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "pepin_bringup.rtabmap_frame",
            "--ros-args",
            "-p",
            f"graph_memory:={memory}",
        ],
        output="screen",
        prefix=_after_ghost("/rtabmap_frame"),
        **RESPAWN,
    )
    # The operator's Foxglove connects here for the 3D view: the cloud stays on the laptop and
    # the board's topics arrive over the bridge, so nothing crosses the WiFi twice.
    #
    # What the parameters are for (foxglove_bridge 3.4.1, its own launch file lists every one):
    # ``topic_whitelist`` is written out as the default ``['.*']`` on purpose — the bridge
    # advertises a topic when a PUBLISHER for it appears and withdraws the channel when the last
    # one goes, so a topic that only shows up after the board's routes are repaired IS advertised
    # then, and no whitelist entry can hold a channel open for a topic nobody publishes. What
    # a client sees as "the panel went empty" is that withdrawal, or this process restarting:
    # channel ids are per bridge process and start again at 1, which is why every restart of this
    # half ends with ros/foxglove.sh reopen. ``max_qos_depth`` 25 is what the bridge silently
    # clamps to anyway (it warns about /tf, /tf_static and /rosout at every start); saying it
    # keeps the warnings out of the log. ``send_buffer_limit`` 100 MB is ten times the default
    # because the surface cloud is megabytes a frame and the default drops the client instead.
    foxglove = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        output="screen",
        parameters=[
            {
                "port": 8765,
                "address": "0.0.0.0",
                "send_buffer_limit": 100_000_000,
                "topic_whitelist": [".*"],
                "max_qos_depth": 25,
            }
        ],
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
            # The rig, resolved once for the whole launch (camera_rig): the node then publishes
            # that camera's optics and mount, and says which it got in its first line.
            "-p",
            f"camera:={rig}",
        ],
        output="screen",
        prefix=_after_ghost("/camera_stream"),
        **RESPAWN,
    )
    # The same frames fused into one surface (pepin_bringup.depth_fusion): RTAB-Map keeps the
    # graph, the closures and the place recognition; the voxels the operator looks at come from
    # here, beside RTAB-Map's own cloud for comparison.
    #
    # Only what that node still declares is passed. Its ``room``, ``world_map`` and ``map_source``
    # parameters went with the room entity on 2026-09-19 — the volume's file follows the database's
    # own path, because it is painted in that graph's frame — and its ``mode`` is left at its
    # default, since the one thing that parameter decided (whether the GRAPH owns map -> odom) has
    # one answer now: it never does.
    fusion = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "pepin_bringup.depth_fusion",
            "--ros-args",
            # A tracker always publishes a fit now, so both paint gates are on in every session
            # (they used to come up off in SLAM, where nothing published one and a gate waiting for
            # it fused 0 frames of the first session, 2026-09-13 14:05). They stay live:
            # ros/flags.sh set depth_fusion fit_gate false, and lidar_fit_gate beside it.
            "-p",
            "fit_gate:=true",
            "-p",
            "lidar_fit_gate:=true",
            # The volume resumes its OWN snapshot and reads the saved pair only when there is no
            # snapshot yet. --fresh passes this false, which is the whole of "an unknown room".
            "-p",
            f"resume_volume:={'true' if resume_volume else 'false'}",
        ],
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
    # ONE input (pepin_bringup.sensor_pack): a snapshot of whatever is fresh, on one topic. Its
    # ``sources`` is where ``camera_only`` now lives — the old SLAM_CAMERA_ONLY table said the same
    # thing by unsubscribing the scan, and this says it by not packing one.
    pack = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "pepin_bringup.sensor_pack",
            "--ros-args",
            "-p",
            f"sources:={'camera' if camera_only else 'camera,lidar'}",
        ],
        output="screen",
        prefix=_after_ghost("/sensor_pack"),
        condition=IfCondition(LaunchConfiguration("sensor_pack")),
        **RESPAWN,
    )
    # RTAB-Map's own subscription is the snapshot topic, spelled out rather than left to the
    # namespace: the name is the contract between this launch and sensor_pack, and one literal on
    # both sides is what a test can pin.
    remappings = [("sensor_data", SENSOR_DATA_TOPIC)]
    if not packing:
        # The way back (TRIPLE_SUBSCRIPTIONS): RTAB-Map reads the three topics itself again.
        remappings += [
            ("rgb/image", "/camera/image"),
            ("rgb/camera_info", "/camera/camera_info"),
            ("depth/image", "/camera/depth"),
            ("scan", "/scan"),
        ]
    # THE GRID IS THE MAP (World R): rtabmap's own "map" publisher, remapped onto /map in every
    # session. It is transient-local, depth 1, reliable (rtabmap_util/MapsManager.cpp: latch, true
    # by default), so whoever subscribes late is handed the current grid at once; it crosses the
    # bridge to the board's tracker, which adopts it and republishes it for the costmaps. Nothing
    # else publishes /map — the board's map_server is off (ros/nav.launch.py) — so the two
    # publishers of one /map that broke 2026-09-10 cannot happen.
    #   RTAB-Map builds the grid ONLY while something subscribes to it (MapsManager's
    # get_subscription_count gate, and map_cleanup drops the per-node grid cache when the last
    # subscriber goes): the board's tracker and the operator's Foxglove are what keep it alive, and
    # a bridge with no route for /map means a laptop that assembles no map at all.
    #   Since 2026-09-19 the grid reaches /map THROUGH pepin_bringup.rtabmap_frame (GRID_TOPIC ->
    # MAP_TOPIC, flag grid_needs_tie): before this start has recognised a node of the database it
    # loaded, RTAB-Map's graph is the current node alone and its grid is one scan drawn where the
    # odometry puts the cart — the tracker adopted that, matched the scan on a picture of itself
    # at fit 1.00 and stood 1.26 m off. /map still has ONE publisher, and that node's subscription
    # is what keeps the grid being built.
    remappings.append(("map", "/rtabmap/grid"))
    rtabmap = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        # Its outputs are relative names (map, mapGraph, mapPath, info): without a namespace its
        # "map" landed on /map next to the board's static map and fed the laptop's global costmap a
        # second, growing map (2026-09-10 01:00). There is no second map any more, and the grid is
        # remapped onto /map on purpose (see remappings); the namespace keeps every OTHER output of
        # this node (mapGraph, mapPath, info, the clouds) where its readers expect it.
        namespace="rtabmap",
        output="screen",
        parameters=[
            {
                "frame_id": "base_link",
                # Never into anyone's tree: the board's tracker owns map -> odom in every
                # situation, and this graph's word reaches it as a measurement. One owner.
                "publish_tf": False,
                "database_path": database,
                # Never here: the one database is wiped by ros/laptop.sh vslam --fresh, before this
                # process starts, so that "an empty room" is a fact on disk this launch can read
                # (`loaded` above) rather than a flag it has to be told twice.
                "delete_db_on_start": False,
                # The subscription's own depth. There is no synchroniser on the snapshot path
                # (CommonDataSubscriberSensorData.cpp:104), so this is the DDS history of one
                # plain subscription; approx_sync and sync_queue_size come back with
                # TRIPLE_SUBSCRIPTIONS and are read only there.
                "topic_queue_size": 10,
                "wait_for_transform": 0.5,
                # The grid carries the CURRENT frame's cells, not only the keyframes': with this
                # false MapsManager drops pose 0 from what it assembles (MapsManager.cpp:473-476)
                # and the grid moves only when a node lands. True is one re-render per processed
                # frame — at Rtabmap/DetectionRate 1.0, one a second — which is what the board's
                # tracker throttles with its own map_refresh_s.
                "map_always_update": True,
                **rtabmap_parameters(neighbor_refining, memory, packing),
            }
        ],
        remappings=remappings,
    )
    vo_note = (
        f" rgbd_odometry -> {VO_RAW_TOPIC} -> /vo for the board's EKF (withheld until"
        " visual_odometry's vo_publish is on)"
        if _flag(context, "vo")
        else " no visual odometry (vo:=false): the board's odometry is the wheels and the gyro"
    )
    # What RTAB-Map is fed and who decides which sensors are in it. There is no scan/camera mode
    # any more: the snapshot carries whatever was fresh, and camera_only is one flag of one node.
    feed = (
        f"one {SENSOR_DATA_TOPIC} per moment from pepin_bringup.sensor_pack (sources"
        f" {'camera' if camera_only else 'camera,lidar'}; ros/flags.sh set sensor_pack sources"
        " lidar for the lidar alone)"
        if packing
        else "the old synchronised triple straight off /camera/image, /camera/depth and /scan"
        " (sensor_pack:=false): RTAB-Map starves when the camera stops"
    )
    session = (
        f"{'the loaded' if loaded else 'a NEW, empty'} database {database}, memory {memory}"
        f"{'' if loaded else ' (nothing to recognise in a database born now)'}; its grid is the one"
        " map, published latched on /map for the board's tracker; the laptop localizer proposes a"
        " place once a second and measures the camera's pose at 5 Hz for the board's tracker"
    )
    report = (
        f"vslam up: {session}; camera rig: {rig} (config/camera.json's active, or PEPIN_CAMERA,"
        f" or camera:=); RTAB-Map reads {feed}; one parameter table for every situation,"
        " map_frame_id map and publish_tf off (the board's tracker owns map -> odom);"
        f" neighbor_refining={'on' if neighbor_refining else 'off'} (off: the neighbour links"
        " carry the odometry's own covariance, so a loop closure has somewhere to go);" + vo_note
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
                    pack,
                    rtabmap,
                    frame,
                    places,
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
            # Which sensors sensor_pack may put in a snapshot: true is the camera alone, which is
            # what the SLAM_CAMERA_ONLY table used to say by unsubscribing the scan. Live either
            # way: ros/flags.sh set sensor_pack sources camera / lidar / camera,lidar.
            DeclareLaunchArgument("camera_only", default_value="false"),
            # The input. True: one snapshot topic from pepin_bringup.sensor_pack. False: the
            # arrangement of before 2026-09-19 — no packer, and RTAB-Map back on the synchronised
            # triple (TRIPLE_SUBSCRIPTIONS), where a camera that stops starves the mapper.
            DeclareLaunchArgument("sensor_pack", default_value="true"),
            # ICP refines the neighbour links and its own covariance comes with them
            # (RGBD/NeighborLinkRefining, see RTABMAP). Default false since 2026-09-14: refined
            # links are stiffer than the odometry they replace and RGBD/OptimizeMaxError then
            # rejects every closure the graph finds.
            DeclareLaunchArgument("neighbor_refining", default_value="false"),
            # Whether the database may LEARN, read beside a LOADED one (a SLAM session is always
            # "map": see rtabmap_memory). "trust" (the default) starts RTAB-Map LOCALISING —
            # nothing written, the whole database in working memory, a parked cart recognised
            # (LOCALIZE) — and hands the switch to pepin_bringup.rtabmap_frame, which moves it live
            # on trust in the pose: the database may learn only while a sharp pose that is NOT held
            # by the graph exists. "map" starts and stays in mapping mode (the arrangement of
            # before 2026-09-18); "localise" freezes the database for the session. The same word is
            # passed to rtabmap_frame's graph_memory flag, so one argument sets the initial mode
            # and who decides after it.
            DeclareLaunchArgument("memory", default_value="trust"),
            # The volume resumes the database's own snapshot (pepin.worldmap.world_path_for names
            # it after the database, since it is painted in that graph's frame). False is
            # ros/laptop.sh vslam --fresh: a room built from nothing whatever is on disk.
            DeclareLaunchArgument("resume_volume", default_value="true"),
            # The camera as a third odometry: rtabmap_odom's rgbd_odometry and the node that
            # gates it (pepin_bringup.visual_odometry). On by default because it is measured at
            # rest and costs only this laptop (0.25 core); what it costs the ROBOT is still
            # nothing until visual_odometry's vo_publish flag is turned on.
            DeclareLaunchArgument("vo", default_value="true"),
            DeclareLaunchArgument("database", default_value=""),  # empty: DATABASE
            DeclareLaunchArgument("bridge_admin", default_value="http://pepin-zenoh:8000"),
            DeclareLaunchArgument("static_camera_tf", default_value="true"),
            # WHICH CAMERA the head is, by the name of a block in config/camera.json ("overview",
            # the mono webcam; "stereo", the side-by-side module). Empty — the default — leaves it
            # to PEPIN_CAMERA in the container (ros/laptop.sh forwards it) and then to that file's
            # own "active", so switching the rig is one word in one file and no launch argument at
            # all. A name no block answers to stops this launch at start.
            DeclareLaunchArgument("camera", default_value=""),
            # A new board bridge means new subscriptions are needed: the watch exits, the launch
            # shuts down, the container's restart policy brings this half back. Under
            # PEPIN_RMW=zenoh there is no bridge to watch, and a watch that found no admin would
            # exit at once and shut this launch down on every start.
            *(
                []
                if rmw_is_zenoh()
                else [
                    ExecuteProcess(
                        cmd=[
                            "python3",
                            "-m",
                            "pepin_bringup.bridge_watch",
                            LaunchConfiguration("board"),
                            LaunchConfiguration(
                                "bridge_admin"
                            ),  # this side's bridge: the flow watch
                        ],
                        output="screen",
                        on_exit=[Shutdown(reason="the board's bridge restarted")],
                    )
                ]
            ),
            OpaqueFunction(function=_describe),
        ]
    )
