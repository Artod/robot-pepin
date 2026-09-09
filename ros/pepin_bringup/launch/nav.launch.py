"""Nav2 on a saved map, trimmed for the board: one composed process, only what a drive needs.

nav2_bringup's launch adds docking, route, waypoint and smoother servers and a
collision monitor; on four Cortex-A53 cores that load made the controller loop
run at 3.5 Hz instead of 8 and the scans arrive seconds late. This launch
composes map_server, AMCL, planner, controller, behaviors, bt_navigator and the
velocity smoother into one container at a lower CPU priority than the sensor
nodes (see robot.launch.py), so a busy planner never starves the lidar.

Arguments: ``map`` (default the flat's lap3 map), ``params_file``, and ``side``:
``all`` (default) is the whole stack on one machine, as before; ``board`` and ``laptop`` are the
two halves of the thin-client split — the board keeps the reflexes (controller, behaviours, tree,
map, tracker) and gets a link watch, the laptop takes the planner and the goal server. The split
itself is data in ``pepin.deployment`` so a test can hold it; this file only reads it.
Command chain: controller/behaviors -> cmd_vel_nav -> velocity_smoother -> /cmd_vel -> base_bridge.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

from pepin.deployment import SIDES, autostart_for, nav_nodes, runs_here


def _describe(context: LaunchContext) -> list:  # type: ignore[type-arg]
    side = LaunchConfiguration("side").perform(context)
    params = LaunchConfiguration("params_file")
    map_file = LaunchConfiguration("map")
    to_smoother = [("cmd_vel", "cmd_vel_nav")]
    catalogue = {
        "map_server": ComposableNode(
            package="nav2_map_server",
            plugin="nav2_map_server::MapServer",
            name="map_server",
            parameters=[params, {"yaml_filename": map_file}],
        ),
        "controller_server": ComposableNode(
            package="nav2_controller",
            plugin="nav2_controller::ControllerServer",
            name="controller_server",
            parameters=[params],
            remappings=to_smoother,
        ),
        "planner_server": ComposableNode(
            package="nav2_planner",
            plugin="nav2_planner::PlannerServer",
            name="planner_server",
            parameters=[params],
        ),
        "behavior_server": ComposableNode(
            package="nav2_behaviors",
            plugin="behavior_server::BehaviorServer",
            name="behavior_server",
            parameters=[params],
            remappings=to_smoother,
        ),
        "bt_navigator": ComposableNode(
            package="nav2_bt_navigator",
            plugin="nav2_bt_navigator::BtNavigator",
            name="bt_navigator",
            parameters=[params],
        ),
        "velocity_smoother": ComposableNode(
            package="nav2_velocity_smoother",
            plugin="nav2_velocity_smoother::VelocitySmoother",
            name="velocity_smoother",
            parameters=[params],
            remappings=[("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")],
        ),
    }
    nodes = [catalogue[name] for name in nav_nodes(side)]
    if runs_here(side, "map_server"):
        nodes.append(catalogue["map_server"])
        nodes.append(
            ComposableNode(
                package="nav2_lifecycle_manager",
                plugin="nav2_lifecycle_manager::LifecycleManager",
                name="lifecycle_manager_localization",
                parameters=[{"autostart": True, "node_names": ["map_server"]}],
            )
        )
    nodes.append(
        ComposableNode(
            package="nav2_lifecycle_manager",
            plugin="nav2_lifecycle_manager::LifecycleManager",
            # One manager per side, named by side: a manager only bonds with the nodes it
            # started, so the board's never waits for a planner that lives on the laptop.
            name=f"lifecycle_manager_navigation_{side}",
            parameters=[{"autostart": autostart_for(side), "node_names": list(nav_nodes(side))}],
        )
    )
    container = ComposableNodeContainer(
        name=f"nav2_container_{side}" if side != "all" else "nav2_container",
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
    actions: list = [container]  # type: ignore[type-arg]
    if runs_here(side, "relocalizer"):
        # Kidnapped-robot recovery: the whole map is searched when the scan stops fitting.
        actions.append(Node(package="pepin_bringup", executable="relocalizer", output="screen"))
    if runs_here(side, "goal_server"):
        # Waits for orders on a socket so a goal costs a socket write, not a client boot.
        # Its places book follows the map in use: /maps/flat3.yaml -> /maps/flat3.places.yaml.
        places = PythonExpression(
            ["'", LaunchConfiguration("map"), "'.rsplit('.', 1)[0] + '.places.yaml'"]
        )
        actions.append(
            Node(
                package="pepin_bringup",
                executable="goal_server",
                output="screen",
                parameters=[{"places": places, "side": side}],
            )
        )
    if runs_here(side, "link_watch"):
        # As a module, not a console script: a new entry point needs an image rebuild, a module
        # on the mounted package path does not.
        actions.append(
            ExecuteProcess(cmd=["python3", "-m", "pepin_bringup.link_watch"], output="screen")
        )
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("map", default_value="/maps/20260903_182653_lap3_loop.yaml"),
            DeclareLaunchArgument("params_file", default_value="/params/nav2_params.yaml"),
            DeclareLaunchArgument("side", default_value="all", choices=list(SIDES)),
            OpaqueFunction(function=_describe),
        ]
    )
