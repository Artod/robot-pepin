"""Nav2 on a saved map (AMCL) or with slam_toolbox, over robot.launch.py's sensors.

Thin wrapper around nav2_bringup's ``bringup_launch.py``: our parameter file,
our map, composition on (one process: the board has 1.5 GB). Arguments:

- ``map`` (default /maps/20260903_182653_lap3_loop.yaml — the flat's lap3 map)
- ``params_file`` (default /params/nav2_params.yaml)
- ``slam`` (default False): True runs slam_toolbox instead of map_server + AMCL
  and builds the map while driving.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    bringup = os.path.join(
        get_package_share_directory("nav2_bringup"), "launch", "bringup_launch.py"
    )
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(bringup),
        launch_arguments={
            "map": LaunchConfiguration("map"),
            "params_file": LaunchConfiguration("params_file"),
            "slam": LaunchConfiguration("slam"),
            "use_sim_time": "False",
            "autostart": "True",
            "use_composition": "True",
            "use_respawn": "False",
        }.items(),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("map", default_value="/maps/20260903_182653_lap3_loop.yaml"),
            DeclareLaunchArgument("params_file", default_value="/params/nav2_params.yaml"),
            DeclareLaunchArgument("slam", default_value="False"),
            nav2,
        ]
    )
