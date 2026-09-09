"""Everything in one launch process: sensors and bridges, and Nav2 when ``nav:=true``.

Two separate ``ros2 launch`` processes cost ~110 MB of Python each on a 1.5 GB
board that also runs Nav2; this file includes robot.launch.py and, optionally,
nav.launch.py so there is exactly one. Arguments are those of the two files
(``map``, ``params_file``, ``laser_roll``, ``tof``, ...) plus ``nav`` and ``slam`` (both
default false; mapping and AMCL never run together: two map->odom publishers).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    launch_dir = os.path.join(get_package_share_directory("pepin_bringup"), "launch")
    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, "robot.launch.py")),
        launch_arguments={
            "base_bridge_cpp": LaunchConfiguration("base_bridge_cpp"),
            "imu": LaunchConfiguration("imu"),
            "tof": LaunchConfiguration("tof"),
        }.items(),
    )
    nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, "nav.launch.py")),
        condition=IfCondition(LaunchConfiguration("nav")),
        launch_arguments={"side": LaunchConfiguration("side")}.items(),
    )
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, "slam.launch.py")),
        condition=IfCondition(LaunchConfiguration("slam")),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("nav", default_value="false"),
            DeclareLaunchArgument("side", default_value="all"),  # all | board | laptop
            DeclareLaunchArgument("slam", default_value="false"),  # never together with nav
            DeclareLaunchArgument("base_bridge_cpp", default_value="false"),
            DeclareLaunchArgument("imu", default_value="false"),
            DeclareLaunchArgument("tof", default_value="false"),
            robot,
            nav,
            slam,
        ]
    )
