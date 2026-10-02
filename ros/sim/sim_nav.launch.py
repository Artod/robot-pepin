"""Nav2 for the simulator: the stack's own navigation launch, included whole, not copied.

pepin_bringup's nav.launch.py is what the Mac runs for the robot (ros/laptop.sh nav):
the five Nav2 servers composed in one container on ros/params/nav2_params.yaml (both static
layers on ``/map``, RTAB-Map's grid), the lifecycle manager, the goal server on its socket and
the run recorder. The sim changes one thing, and only when the world runs on its own clock
(``use_sim_time:=true``, ros/sim.sh up --rate N): every node of the included launch is given
``use_sim_time`` (launch_ros's global parameters reach the Node actions and every composable
node's load request; Nav2 passes it on to the costmaps).

    ros2 launch /sim/sim_nav.launch.py [use_sim_time:=true] [map:=/sim/worlds/flat.yaml]
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_context import LaunchContext
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import SetParameter
from launch_ros.substitutions import FindPackageShare


def sim_time(context: LaunchContext) -> list:  # type: ignore[type-arg]
    """``use_sim_time`` for every node that follows, only when asked: on the wall clock the
    included launch runs exactly as it does on the robot."""
    if LaunchConfiguration("use_sim_time").perform(context).lower() != "true":
        return []
    return [SetParameter(name="use_sim_time", value=True)]


def generate_launch_description() -> LaunchDescription:
    """The one included launch, after the optional clock parameter."""
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("map", default_value="/sim/worlds/flat.yaml"),
            OpaqueFunction(function=sim_time),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("pepin_bringup"), "launch", "nav.launch.py"]
                    )
                ),
                launch_arguments={
                    "map": LaunchConfiguration("map"),
                    "params_file": "/params/nav2_params.yaml",
                    "recorder": "jsonl",
                }.items(),
            ),
        ]
    )
