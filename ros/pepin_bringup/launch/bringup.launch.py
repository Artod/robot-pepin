"""Everything in one launch process: sensors and bridges, and Nav2 when ``nav:=true``.

Two separate ``ros2 launch`` processes cost ~110 MB of Python each on a 1.5 GB
board that also runs Nav2; this file includes robot.launch.py and, optionally,
nav.launch.py so there is exactly one. Arguments are those of the two files
(``map``, ``params_file``, ``laser_roll``, ``tof``, ...) plus:

- ``nav``: Nav2 on the saved ``map``.
- ``slam``: online SLAM (ros/thin.sh slam). Nav2 too — the cart must navigate the map it is
  building — but without map_server and without the tracker: the map arrives from the laptop's
  RTAB-Map as ``/map`` and its correction as ``map -> odom``. It implies ``nav``, so the board
  needs one switch, not two that can disagree.
- ``recorder``: ``jsonl`` (default) or ``bag`` — who writes a drive down (ros/README.md,
  "Two recorders").
- ``slam_toolbox``: the old lidar-only mapper (ros/mode.sh slam_toolbox), which builds a map to
  SAVE and cannot navigate on it. Never together with ``nav`` or ``slam``: two map -> odom
  publishers, and the reason this argument is not called ``slam`` any more.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetLaunchConfiguration,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression

from pepin.deployment import CONTAINER_STOP_TIMEOUT_S

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


def generate_launch_description() -> LaunchDescription:
    launch_dir = os.path.join(get_package_share_directory("pepin_bringup"), "launch")
    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, "robot.launch.py")),
        launch_arguments={
            "base_bridge_cpp": LaunchConfiguration("base_bridge_cpp"),
            "imu": LaunchConfiguration("imu"),
            "ekf": LaunchConfiguration("ekf"),
            "tof": LaunchConfiguration("tof"),
            "neck": LaunchConfiguration("neck"),
        }.items(),
    )
    # SLAM mode navigates: one switch on the board, so PEPIN_NAV and PEPIN_SLAM can never
    # disagree about whether the cart may drive the map it is building.
    driving = PythonExpression(
        [
            "'",
            LaunchConfiguration("nav"),
            "' == 'true' or '",
            LaunchConfiguration("slam"),
            "' == 'true'",
        ]
    )
    nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, "nav.launch.py")),
        condition=IfCondition(driving),
        launch_arguments={
            "side": LaunchConfiguration("side"),
            "slam": LaunchConfiguration("slam"),
            "recorder": LaunchConfiguration("recorder"),
        }.items(),
    )
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, "slam.launch.py")),
        condition=IfCondition(LaunchConfiguration("slam_toolbox")),
    )
    return LaunchDescription(
        [
            *SHUTDOWN,  # before the includes: a scoped include inherits what is set above it
            DeclareLaunchArgument("nav", default_value="false"),
            DeclareLaunchArgument("side", default_value="all"),  # all | board | laptop
            DeclareLaunchArgument("slam", default_value="false"),  # online SLAM: implies nav
            DeclareLaunchArgument("slam_toolbox", default_value="false"),  # never with nav/slam
            DeclareLaunchArgument("base_bridge_cpp", default_value="false"),
            DeclareLaunchArgument("imu", default_value="false"),
            DeclareLaunchArgument("ekf", default_value="true"),
            DeclareLaunchArgument("tof", default_value="false"),
            DeclareLaunchArgument("neck", default_value="false"),
            # Which recorder writes a drive: jsonl (the Python node, the default) or bag
            # (`ros2 bag record` + ros/tools/bag_to_tape.py). The board carries it as
            # PEPIN_RECORDER in /etc/default/pepin-ros.
            DeclareLaunchArgument("recorder", default_value="jsonl"),
            robot,
            nav,
            slam_toolbox,
        ]
    )
