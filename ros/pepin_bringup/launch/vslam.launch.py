"""Camera + lidar mapping on the laptop: RTAB-Map, either beside the known map or AS the map.

Two modes, one launch, because everything around RTAB-Map is the same in both: the neck camera's
stream becomes images (pepin_bringup.camera_stream), a depth network scaled by the lidar turns
them into depth images (pepin_bringup.depth_stream), the same frames are read once more at floor
height (pepin_bringup.contact_scan) and fused into one surface (pepin_bringup.depth_fusion), and
the operator's Foxglove connects here for the 3D view. What the mode changes is which map the
robot drives on.

KNOWN MAP (the default). The board's tracker owns ``map -> odom`` on a saved map and RTAB-Map's
"odometry" is that tracker's pose (``odom_frame_id: map``), so RTAB-Map keeps its graph in a
frame of its own (``map_frame_id: rtabmap``, tied to ``map`` by pepin_bringup.rtabmap_frame) and
never publishes into the tracker's tree: two localisations coexist and its grid is the map the
tracker will be handed one day.

ONLINE SLAM (``slam:=true`` — ros/laptop.sh vslam --slam). There is no saved map: the robot is
put somewhere unknown and RTAB-Map builds ONE map while it drives. Its "odometry" is then the
EKF's wheels-plus-gyro odometry over the bridge (``odom_frame_id: odom``) and it owns
``map_frame_id: map`` — its grid is published as ``/map``, which the board's global costmap reads
as its static layer, and its correction becomes the board's ``map -> odom`` (rtabmap_frame's
``slam`` switch sends it as a message, pepin_bringup.slam_frame broadcasts it there). The
database starts empty every session unless ``resume:=true``, and it is a file of its own: a SLAM
session must never wipe the known map's graph. With ``camera_only:=true`` the lidar is not
subscribed at all and the grid comes from the camera's depth — the honest test of "the camera as
the primary sense", and the one case where the map is only as true as the network's scale.

Arguments: ``board`` (the robot's address for the camera stream), ``slam``, ``camera_only``,
``resume``, ``database`` (empty: chosen by the mode), ``bridge_admin`` (the laptop bridge's REST
admin, asked whether it still lists this launch's previous incarnation), ``static_camera_tf``
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
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pepin.deployment import laptop_launch_nodes

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
    # RTAB-Map's "odometry" is the tracker's pose (map -> odom -> base_link), not the wheels': a
    # node lands where the lidar says the cart is, to a centimetre, and two clouds a few degrees
    # apart coincide (2026-09-10: with raw odometry, the graph rejected most closures as
    # inconsistent and the voxels smeared).
    "odom_frame_id": "map",
    "map_frame_id": "rtabmap",
    "subscribe_scan": True,
    "Grid/Sensor": "2",  # 0=scan, 1=depth, 2=both (0.22's name for the old Grid/FromDepth)
    "Grid/3D": "true",
    "Grid/RangeMax": "5.0",
    "Grid/RayTracing": "false",  # 3D ray tracing costs more than it clears at 1 Hz
    "RGBD/NeighborLinkRefining": "true",
}

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
SLAM_CAMERA_ONLY = {
    "subscribe_scan": False,
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


def rtabmap_parameters(slam: bool, camera_only: bool) -> dict[str, object]:
    """Everything RTAB-Map is told for one mode: the common table under the mode's frames and
    grid. ``camera_only`` is read in SLAM mode alone — beside a known map the lidar is what
    makes the graph metric."""
    mode: dict[str, object] = dict(KNOWN_MAP)
    if slam:
        mode = {**SLAM, **(SLAM_CAMERA_ONLY if camera_only else SLAM_LIDAR)}
    return {**RTABMAP, **mode}


def _flag(context: LaunchContext, name: str) -> bool:
    """A launch argument as a bool, the way every other launch here reads one."""
    return LaunchConfiguration(name).perform(context).lower() == "true"


def _describe(context: LaunchContext) -> list:  # type: ignore[type-arg]
    """The nodes of this launch, once the arguments have values."""
    board = LaunchConfiguration("board")
    slam = _flag(context, "slam")
    camera_only = _flag(context, "camera_only")
    resume = _flag(context, "resume")
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
    fusion = ExecuteProcess(
        cmd=["python3", "-m", "pepin_bringup.depth_fusion"],
        output="screen",
        prefix=_after_ghost("/depth_fusion"),
        **RESPAWN,
    )
    remappings = [
        ("rgb/image", "/camera/image"),
        ("rgb/camera_info", "/camera/camera_info"),
        ("depth/image", "/camera/depth"),
        ("scan", "/scan"),
    ]
    if slam:
        # In SLAM mode this grid IS the map: /map, transient local, read by the board's global
        # costmap over the bridge. Beside a known map it stays /rtabmap/map, out of Nav2's way.
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
                **rtabmap_parameters(slam, camera_only),
            }
        ],
        remappings=remappings,
    )
    scan = "camera only (no /scan)" if camera_only else "lidar + camera"
    start = "resumed" if resume else "empty"
    report = (
        f"vslam up: online SLAM, {scan}, {start} database {database}, grid -> /map"
        if slam
        else f"vslam up: beside the known map, lidar + camera, database {database}"
    )
    return [
        LogInfo(msg=report),
        ghost_wait,
        RegisterEventHandler(
            OnProcessExit(
                target_action=ghost_wait,
                on_exit=[camera, depth, contact, fusion, rtabmap, frame, foxglove],
            )
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("board", default_value="10.0.0.187"),
            DeclareLaunchArgument("slam", default_value="false"),
            DeclareLaunchArgument("camera_only", default_value="false"),  # SLAM mode only
            DeclareLaunchArgument("resume", default_value="false"),  # SLAM mode only
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
                ],
                output="screen",
                on_exit=[Shutdown(reason="the board's bridge restarted")],
            ),
            OpaqueFunction(function=_describe),
        ]
    )
