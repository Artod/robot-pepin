"""Camera + lidar SLAM on the laptop: RTAB-Map builds and keeps a map the board never has to.

The board keeps its tracker on the static map (map -> odom, the reflexes' frame). This launch
runs beside it, on the laptop, over the bridge: the neck camera's stream becomes images, a depth
network scaled by the lidar turns them into depth images (pepin_bringup.depth_stream), and
RTAB-Map fuses both with the lidar scans and the tracker's pose (map -> base_link from TF, its
"odometry") into a pose graph — loop closures found by what the camera sees, transforms refined
by ICP on the scans. It publishes its own frame (``rtabmap``, tied to ``map`` by its own
correction, see pepin_bringup.rtabmap_frame) and never ``map -> odom``, so the two
localisations coexist; its occupancy grid is the map the tracker will be handed one day.

Arguments: ``board`` (the robot's address for the camera stream), ``database`` (RTAB-Map's
database; deleted on start while the map is being learnt from scratch), ``bridge_admin`` (the
laptop bridge's REST admin, asked whether it still lists this launch's previous incarnation).
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pepin.deployment import laptop_launch_nodes

RTABMAP = {
    # registration: ICP on the 2D scans in the plane; the camera only says "I have been here"
    "Reg/Strategy": "1",
    "Reg/Force3DoF": "true",
    "Icp/VoxelSize": "0.05",
    "Icp/MaxCorrespondenceDistance": "0.15",
    "Icp/PointToPlane": "false",
    # the grid comes from the lidar AND the camera's depth (pepin_bringup.depth_stream): the lidar
    # keeps the plane exact, the depth adds what stands above it — table tops, seats, shelves.
    # Grid/3D keeps the voxels so the operator sees the room in three dimensions (cloud_map).
    "Grid/Sensor": "2",
    "Grid/3D": "true",
    "Grid/RangeMax": "5.0",
    "Grid/CellSize": "0.05",
    "Grid/RayTracing": "false",  # 3D ray tracing costs more than it clears at 1 Hz
    "Grid/DepthDecimation": "4",  # 160x90 depth samples a frame: plenty for 5 cm voxels
    "Grid/MaxGroundHeight": "0.08",
    "Grid/MaxObstacleHeight": "1.3",  # the band the costmap scan uses too
    "Grid/NormalsSegmentation": "false",  # height decides ground vs obstacle; the floor is flat
    # a mono network's depth frays at object edges and far away: lone voxels go, the image's
    # borders (nominal optics, worst there) are not used
    "Grid/NoiseFilteringRadius": "0.10",
    "Grid/NoiseFilteringMinNeighbors": "5",
    # graph: refine neighbour links with ICP, close loops with nearby nodes by space
    "RGBD/NeighborLinkRefining": "true",
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
}


def generate_launch_description() -> LaunchDescription:
    board = LaunchConfiguration("board")
    database = LaunchConfiguration("database")
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
    )
    # RTAB-Map's loop-closure correction as odom -> rtabmap (pepin_bringup.rtabmap_frame): the
    # voxels and the cart stay together after a closure.
    frame = ExecuteProcess(
        cmd=["python3", "-m", "pepin_bringup.rtabmap_frame"],
        output="screen",
    )
    # The operator's Foxglove connects here for the 3D view: the cloud stays on the laptop and
    # the board's topics arrive over the bridge, so nothing crosses the WiFi twice.
    foxglove = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        output="screen",
        parameters=[{"port": 8765, "address": "0.0.0.0", "send_buffer_limit": 100_000_000}],
    )
    camera = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "pepin_bringup.camera_stream",
            "--ros-args",
            "-p",
            ["board:=", board],
        ],
        output="screen",
    )
    # The same frames fused into one surface (pepin_bringup.depth_fusion): RTAB-Map keeps the
    # graph, the closures and the place recognition; the voxels the operator looks at come from
    # here, beside RTAB-Map's own cloud for comparison.
    fusion = ExecuteProcess(
        cmd=["python3", "-m", "pepin_bringup.depth_fusion"],
        output="screen",
    )
    rtabmap = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        # Its outputs are relative names (map, mapGraph, mapPath, info): without a namespace
        # its "map" landed on /map next to the board's static map and fed the laptop's
        # global costmap a second, growing map (2026-09-10 01:00).
        namespace="rtabmap",
        output="screen",
        arguments=["-d"],  # start from an empty database while the map is being learnt
        parameters=[
            {
                "frame_id": "base_link",
                # RTAB-Map's "odometry" is the tracker's pose (map -> odom -> base_link), not the
                # wheels': a node lands where the lidar says the cart is, to a centimetre, and two
                # clouds a few degrees apart coincide (2026-09-10: with raw odometry, the graph
                # rejected most closures as inconsistent and the voxels smeared).
                "odom_frame_id": "map",
                "map_frame_id": "rtabmap",
                "publish_tf": False,
                "database_path": database,
                "subscribe_depth": True,  # rgb + depth + camera_info from the camera nodes
                "subscribe_rgb": False,
                "subscribe_scan": True,
                "approx_sync": True,
                "sync_queue_size": 30,
                "topic_queue_size": 10,
                "wait_for_transform": 0.5,
                "odom_sensor_sync": False,
                # the grid is republished every second: the operator watches it grow
                "map_always_update": True,
                **RTABMAP,
            }
        ],
        remappings=[
            ("rgb/image", "/camera/image"),
            ("rgb/camera_info", "/camera/camera_info"),
            ("depth/image", "/camera/depth"),
            ("scan", "/scan"),
        ],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("board", default_value="10.0.0.187"),
            DeclareLaunchArgument("database", default_value="/maps/rtabmap.db"),
            DeclareLaunchArgument("bridge_admin", default_value="http://pepin-zenoh:8000"),
            # A new board bridge means new subscriptions are needed: the watch exits, the launch
            # shuts down, the container's restart policy brings this half back.
            ExecuteProcess(
                cmd=["python3", "-m", "pepin_bringup.bridge_watch", board],
                output="screen",
                on_exit=[Shutdown(reason="the board's bridge restarted")],
            ),
            ghost_wait,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=ghost_wait,
                    on_exit=[camera, depth, fusion, rtabmap, frame, foxglove],
                )
            ),
        ]
    )
