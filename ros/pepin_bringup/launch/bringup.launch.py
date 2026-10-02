"""The board's one launch process: robot.launch.py's sensors and bridges, with the stop window.

The board is a sensor box; navigation runs on the laptop. This file includes robot.launch.py
under the container's stop window and passes on the switches of the board's unit
(board/pepin-ros.service): ``imu``, ``ekf``, ``laser_odom``, ``tof``, ``board_bag``.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetLaunchConfiguration,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from pepin.deployment import CONTAINER_STOP_TIMEOUT_S

# How long a node of this launch is given to end on SIGINT before the launch escalates to
# SIGTERM, and then how long before SIGKILL. launch's own defaults are 5 s and 5 s, under the
# board's nodes leaving the middleware: every shutdown ended in SIGKILLs mid-write. The window is
# the container's own stop window (pepin.deployment.CONTAINER_STOP_TIMEOUT_S, ros/lib.sh,
# board/pepin-ros.service), so on a `docker stop` nothing inside escalates before docker's
# SIGKILL at its end.
SHUTDOWN = [
    SetLaunchConfiguration("sigterm_timeout", str(CONTAINER_STOP_TIMEOUT_S)),
    SetLaunchConfiguration("sigkill_timeout", "5"),
]


def generate_launch_description() -> LaunchDescription:
    launch_dir = os.path.join(get_package_share_directory("pepin_bringup"), "launch")
    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_dir, "robot.launch.py")),
        launch_arguments={
            "imu": LaunchConfiguration("imu"),
            "ekf": LaunchConfiguration("ekf"),
            "laser_odom": LaunchConfiguration("laser_odom"),
            "tof": LaunchConfiguration("tof"),
            "board_bag": LaunchConfiguration("board_bag"),
        }.items(),
    )
    return LaunchDescription(
        [
            *SHUTDOWN,  # before the include: a scoped include inherits what is set above it
            DeclareLaunchArgument("imu", default_value="true"),
            DeclareLaunchArgument("ekf", default_value="true"),
            DeclareLaunchArgument("laser_odom", default_value="true"),
            DeclareLaunchArgument("tof", default_value="true"),
            DeclareLaunchArgument("board_bag", default_value="false"),
            robot,
        ]
    )
