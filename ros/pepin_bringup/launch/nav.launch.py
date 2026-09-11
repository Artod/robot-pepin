"""Nav2 on a saved map, trimmed for the board: one composed process, only what a drive needs.

nav2_bringup's launch adds docking, route, waypoint and smoother servers and a
collision monitor; on four Cortex-A53 cores that load made the controller loop
run at 3.5 Hz instead of 8 and the scans arrive seconds late. This launch
composes map_server, AMCL, planner, controller, behaviors, bt_navigator and the
velocity smoother into one container at a lower CPU priority than the sensor
nodes (see robot.launch.py), so a busy planner never starves the lidar.

Arguments: ``map`` (default the flat's lap3 map), ``params_file``, ``bridge_admin`` (the REST
admin of the bridge on this host, for the ghost waits; empty = by side, see
pepin.deployment.bridge_admin_for) and ``side``:
``all`` (default) is the whole stack on one machine, as before; ``board`` and ``laptop`` are the
two halves of the thin-client split — the board keeps the reflexes (controller, behaviours, tree,
map, tracker) and gets a link watch, the laptop takes the planner and the goal server. The split
itself is data in ``pepin.deployment`` so a test can hold it; this file only reads it.
Command chain: controller/behaviors -> cmd_vel_nav -> velocity_smoother -> /cmd_vel -> base_bridge.

``slam:=true`` (ros/thin.sh slam) is the same stack on a map that does not exist yet: RTAB-Map on
the laptop builds it while the cart drives and publishes it as ``/map``, which the global
costmap's static layer reads (transient local, and every planner here allows unknown space), so
this launch starts no map_server and no scan-matching tracker — pepin_bringup.slam_frame owns
``map -> odom`` instead, from the correction the laptop sends it. One map, one owner of the
frame, in either mode.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

from pepin.deployment import (
    SIDES,
    autostart_for,
    bridge_admin_for,
    laptop_launch_nodes,
    nav_container_nodes,
    nav_nodes,
    runs_here,
)

# Our own nodes come back by themselves after this pause: a code change costs one kicked process
# (ros/thin.sh kick <node> on the board, ros/laptop.sh kick <node> here; SIGINT, what the launch
# sends at shutdown) instead of a stack restart with the bridge and the laptop containers behind
# it. A kicked node leaves DDS properly and the pause starts at its exit, so its successor never
# meets its ghost in the bridge; a CRASHED node does (nothing disposed, the name outlives it by
# the DDS lease, the bridge drops its routes when the ghost expires — the Nav2 container at
# 2026-09-11 03:07), so every respawned command starts through a ghost wait of its own names
# (``_after_ghost``). The watches are not respawned: their exit is the signal (bridge_watch ->
# shutdown, ghost_wait -> start the nodes, link_watch cancels and keeps running).
RESPAWN = {"respawn": True, "respawn_delay": 2.0}


def _after_ghost(admin: str, *names: str) -> str:
    """A command prefix that waits until the bridge at ``admin`` lists none of ``names`` and
    then becomes the command (pepin_bringup.ghost_wait; an unreachable admin is not waited for)."""
    return f"python3 -m pepin_bringup.ghost_wait {admin} {' '.join(names)} --"


def _describe(context: LaunchContext) -> list:  # type: ignore[type-arg]
    side = LaunchConfiguration("side").perform(context)
    # Online SLAM: the map is being built on the laptop while the cart drives, so nothing here
    # serves a saved map and nothing here matches a scan against one. The global costmap's static
    # layer takes /map over the bridge and pepin_bringup.slam_frame owns map -> odom instead.
    slam = LaunchConfiguration("slam").perform(context).lower() == "true"
    admin = LaunchConfiguration("bridge_admin").perform(context) or bridge_admin_for(side)
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
    if runs_here(side, "map_server", slam):
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
        # Planning yields to the sensor nodes (nice -10) under load; the ghost wait keeps the
        # niceness through its exec.
        prefix=f"nice -n 5 {_after_ghost(admin, *nav_container_nodes(side))}",
        # A crash of one Nav2 node takes the whole container with it (SIGABRT at a goal,
        # 2026-09-10 16:06: the board drove nothing until a stack restart). Respawned, the
        # container is back in seconds and the laptop's bring-up activates its nodes again.
        respawn=True,
        respawn_delay=2.0,
        # The whole params file goes to the container process as well: the costmaps are
        # sub-nodes (/local_costmap/local_costmap) created inside controller/planner and
        # only see parameters given to the process, not the ones given to their parents.
        parameters=[params],
        composable_node_descriptions=nodes,
    )
    # The ROS nodes; the watches below are plain processes and start at once.
    actions: list = [container]  # type: ignore[type-arg]
    if runs_here(side, "slam_frame", slam):
        # The laptop's RTAB-Map correction, broadcast here as map -> odom
        # (pepin_bringup.slam_frame): /tf crosses the bridge board -> laptop only, so the
        # correction arrives as a message on /map_odom and becomes a transform where Nav2 and
        # the behaviours look it up. The tracker's seat while there is no map to track against.
        actions.append(
            ExecuteProcess(
                cmd=["python3", "-m", "pepin_bringup.slam_frame"],
                output="screen",
                prefix=_after_ghost(admin, "/slam_frame"),
                **RESPAWN,
            )
        )
    if runs_here(side, "relocalizer", slam):
        # Kidnapped-robot recovery: the whole map is searched when the scan stops fitting.
        # Respawned, it re-seeds from /maps/last_pose.json (written every 2 s while the fit is
        # good): a kick at rest costs the seconds it takes to start, nothing else.
        actions.append(
            Node(
                package="pepin_bringup",
                executable="relocalizer",
                output="screen",
                prefix=_after_ghost(admin, "/relocalizer"),
                **RESPAWN,
            )
        )
    if runs_here(side, "run_recorder"):
        # As a module, like link_watch: the image's console scripts are generated at build time
        # and the sources are mounted over them, so a new executable would need a rebuild.
        actions.append(
            ExecuteProcess(
                cmd=["python3", "-m", "pepin_bringup.run_recorder"],
                output="screen",
                prefix=_after_ghost(admin, "/run_recorder"),
                **RESPAWN,
            )
        )
    if runs_here(side, "goal_server"):
        # Waits for orders on a socket so a goal costs a socket write, not a client boot.
        # Its places book follows the map in use: /maps/flat3.yaml -> /maps/flat3.places.yaml.
        # A SLAM session gets a book of its own, empty until the drive marks something: the saved
        # map's places are coordinates in a frame this new map does not share, and "go printer"
        # would drive at a spot that means nothing here.
        places: object = PythonExpression(
            ["'", LaunchConfiguration("map"), "'.rsplit('.', 1)[0] + '.places.yaml'"]
        )
        if slam:
            places = "/maps/slam.places.yaml"
        actions.append(
            Node(
                package="pepin_bringup",
                executable="goal_server",
                output="screen",
                parameters=[{"places": places, "side": side}],
                prefix=_after_ghost(admin, "/goal_server"),
                **RESPAWN,
            )
        )
    if side == "laptop":
        # The bridge keeps routes by node name: the nodes start only once it has forgotten the
        # previous incarnation of this launch (pepin_bringup.ghost_wait), or their routes die
        # with the ghost ten seconds after they were made (2026-09-10, RTAB-Map without /scan).
        ghost_wait = ExecuteProcess(
            cmd=["python3", "-m", "pepin_bringup.ghost_wait", admin, *laptop_launch_nodes("nav")],
            output="screen",
        )
        actions = [
            ghost_wait,
            RegisterEventHandler(OnProcessExit(target_action=ghost_wait, on_exit=actions)),
        ]
        # A new board bridge means new subscriptions are needed: the watch exits, the launch
        # shuts down, the container's restart policy brings this half back
        # (pepin_bringup.bridge_watch).
        actions.append(
            ExecuteProcess(
                cmd=["python3", "-m", "pepin_bringup.bridge_watch", LaunchConfiguration("board")],
                output="screen",
                on_exit=[Shutdown(reason="the board's bridge restarted")],
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
            # Online SLAM (ros/thin.sh slam): no map_server, no tracker, /map from the laptop.
            DeclareLaunchArgument("slam", default_value="false"),
            DeclareLaunchArgument("board", default_value="10.0.0.187"),
            DeclareLaunchArgument("bridge_admin", default_value=""),  # empty: by side
            OpaqueFunction(function=_describe),
        ]
    )
