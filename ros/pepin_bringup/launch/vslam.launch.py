"""Camera + lidar SLAM on the laptop: RTAB-Map builds and keeps a map the board never has to.

The board keeps its tracker on the static map (map -> odom, the reflexes' frame). This launch
runs beside it, on the laptop, over the bridge: the neck camera's stream becomes images, and
RTAB-Map fuses them with the lidar scans and the board's odometry (odom -> base_link from TF)
into a pose graph — loop closures found by what the camera sees, transforms refined by ICP on
the scans. It publishes its own frame (``rtabmap``) and never ``map -> odom``, so the two
localisations coexist; its occupancy grid is the map the tracker will be handed one day.

Arguments: ``board`` (the robot's address for the camera stream), ``database`` (RTAB-Map's
database; deleted on start while the map is being learnt from scratch).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

RTABMAP = {
    # registration: ICP on the 2D scans in the plane; the camera only says "I have been here"
    "Reg/Strategy": "1",
    "Reg/Force3DoF": "true",
    "Icp/VoxelSize": "0.05",
    "Icp/MaxCorrespondenceDistance": "0.15",
    "Icp/PointToPlane": "false",
    # the grid comes from the lidar, not from depth the camera does not have
    "Grid/FromDepth": "false",
    "Grid/RangeMax": "6.0",
    "Grid/CellSize": "0.05",
    "Grid/RayTracing": "true",
    # graph: refine neighbour links with ICP, close loops with nearby nodes by space
    "RGBD/NeighborLinkRefining": "true",
    "RGBD/ProximityBySpace": "true",
    "RGBD/LinearUpdate": "0.05",
    "RGBD/AngularUpdate": "0.05",
    "RGBD/OptimizeFromGraphEnd": "false",
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
    return LaunchDescription(
        [
            DeclareLaunchArgument("board", default_value="10.0.0.187"),
            DeclareLaunchArgument("database", default_value="/maps/rtabmap.db"),
            # A new board bridge means new subscriptions are needed: the watch exits, the launch
            # shuts down, the container's restart policy brings this half back.
            ExecuteProcess(
                cmd=["python3", "-m", "pepin_bringup.bridge_watch", board],
                output="screen",
                on_exit=[Shutdown(reason="the board's bridge restarted")],
            ),
            ExecuteProcess(
                cmd=[
                    "python3",
                    "-m",
                    "pepin_bringup.camera_stream",
                    "--ros-args",
                    "-p",
                    ["board:=", board],
                ],
                output="screen",
            ),
            Node(
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
                        "odom_frame_id": "odom",
                        "map_frame_id": "rtabmap",
                        "publish_tf": False,
                        "database_path": database,
                        "subscribe_depth": False,
                        "subscribe_rgb": True,
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
                    ("scan", "/scan"),
                ],
            ),
        ]
    )
