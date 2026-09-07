"""Nav2 on a saved map, trimmed for the board: one composed process, only what a drive needs.

nav2_bringup's launch adds docking, route, waypoint and smoother servers and a
collision monitor; on four Cortex-A53 cores that load made the controller loop
run at 3.5 Hz instead of 8 and the scans arrive seconds late. This launch
composes map_server, AMCL, planner, controller, behaviors, bt_navigator and the
velocity smoother into one container at a lower CPU priority than the sensor
nodes (see robot.launch.py), so a busy planner never starves the lidar.

Arguments: ``map`` (default the flat's lap3 map), ``params_file``.
Command chain: controller/behaviors -> cmd_vel_nav -> velocity_smoother -> /cmd_vel -> base_bridge.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description() -> LaunchDescription:
    params = LaunchConfiguration("params_file")
    map_file = LaunchConfiguration("map")
    to_smoother = [("cmd_vel", "cmd_vel_nav")]
    nodes = [
        ComposableNode(
            package="nav2_map_server",
            plugin="nav2_map_server::MapServer",
            name="map_server",
            parameters=[params, {"yaml_filename": map_file}],
        ),
        ComposableNode(
            package="nav2_controller",
            plugin="nav2_controller::ControllerServer",
            name="controller_server",
            parameters=[params],
            remappings=to_smoother,
        ),
        ComposableNode(
            package="nav2_planner",
            plugin="nav2_planner::PlannerServer",
            name="planner_server",
            parameters=[params],
        ),
        ComposableNode(
            package="nav2_behaviors",
            plugin="behavior_server::BehaviorServer",
            name="behavior_server",
            parameters=[params],
            remappings=to_smoother,
        ),
        ComposableNode(
            package="nav2_bt_navigator",
            plugin="nav2_bt_navigator::BtNavigator",
            name="bt_navigator",
            parameters=[params],
        ),
        ComposableNode(
            package="nav2_velocity_smoother",
            plugin="nav2_velocity_smoother::VelocitySmoother",
            name="velocity_smoother",
            parameters=[params],
            remappings=[("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")],
        ),
        ComposableNode(
            package="nav2_lifecycle_manager",
            plugin="nav2_lifecycle_manager::LifecycleManager",
            name="lifecycle_manager_localization",
            parameters=[{"autostart": True, "node_names": ["map_server"]}],
        ),
        ComposableNode(
            package="nav2_lifecycle_manager",
            plugin="nav2_lifecycle_manager::LifecycleManager",
            name="lifecycle_manager_navigation",
            parameters=[
                {
                    "autostart": True,
                    "node_names": [
                        "controller_server",
                        "planner_server",
                        "behavior_server",
                        "bt_navigator",
                        "velocity_smoother",
                    ],
                }
            ],
        ),
    ]
    container = ComposableNodeContainer(
        name="nav2_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_isolated",
        output="screen",
        prefix="nice -n 5",  # planning yields to the sensor nodes (nice -10) under load
        # The whole params file goes to the container process as well: the costmaps are
        # sub-nodes (/local_costmap/local_costmap) created inside controller/planner and
        # only see parameters given to the process, not the ones given to their parents.
        parameters=[params],
        composable_node_descriptions=nodes,
    )
    # Kidnapped-robot recovery: whole-map search re-seeds AMCL when the scan stops fitting.
    relocalizer = Node(package="pepin_bringup", executable="relocalizer", output="screen")
    return LaunchDescription(
        [
            DeclareLaunchArgument("map", default_value="/maps/20260903_182653_lap3_loop.yaml"),
            DeclareLaunchArgument("params_file", default_value="/params/nav2_params.yaml"),
            container,
            relocalizer,
        ]
    )
