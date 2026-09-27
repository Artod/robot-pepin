"""Nav2 on a saved map, trimmed for the board: one composed process, only what a drive needs.

nav2_bringup's launch adds docking, route, waypoint and smoother servers and a
collision monitor; on four Cortex-A53 cores that load made the controller loop
run at 3.5 Hz instead of 8 and the scans arrive seconds late. This launch
composes map_server, AMCL, planner, controller, behaviors, bt_navigator and the
velocity smoother into one container at a lower CPU priority than the sensor
nodes (see robot.launch.py), so a busy planner never starves the lidar.

Arguments: ``map`` (default the flat's lap3 map), ``params_file`` and ``side``:
``all`` (default) is the whole stack on one machine, as before; ``board`` and ``laptop`` are the
two halves of the thin-client split — the board keeps the reflexes (controller, behaviours, tree,
map) and gets a link watch, the laptop takes the planner and the goal server. The split
itself is data in ``pepin.deployment`` so a test can hold it; this file only reads it.
Command chain: controller/behaviors -> cmd_vel_nav -> velocity_smoother -> /cmd_vel -> base_bridge.

ONE OWNER OF ``map -> odom``: the laptop's RTAB-Map publishes it, and both costmaps' static layer
reads ``/map`` (RTAB-Map's grid, relayed by pepin_bringup.rtabmap_frame). The board tracker that
owned the edge before, and the message-path owner of it (``slam:=true``), are on the tag
alt/tracker-2026-09-22.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetLaunchConfiguration,
)
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode
from pepin_bringup.launch_kit import respawned_container

from pepin.deployment import (
    CONTAINER_STOP_TIMEOUT_S,
    SIDES,
    autostart_for,
    nav_nodes,
    runs_here,
)

# Our own nodes come back by themselves after this pause: a code change costs one kicked process
# (ros/thin.sh kick <node> on the board, ros/laptop.sh kick <node> here; SIGINT, what the launch
# sends at shutdown) instead of a stack restart with the laptop containers behind it. The link
# watch is not respawned: it cancels and keeps running. (Under CycloneDDS every respawned command
# started through a ghost wait of the bridge's admin; tag alt/cyclone-bridges-2026-09-20.)
RESPAWN = {"respawn": True, "respawn_delay": 2.0}


def nav_parts(context: LaunchContext) -> list:  # type: ignore[type-arg]
    """The Nav2 nodes this side composes, described anew on every call (a respawned container
    loads fresh ones, see pepin_bringup.launch_kit): the servers of pepin.deployment.nav_nodes,
    map_server when ``map_server:=true`` asks for it, and this side's lifecycle manager."""
    side = LaunchConfiguration("side").perform(context)
    # A pgm served by map_server is no longer part of the running loop: both costmaps' static
    # layers read RTAB-Map's grid on /map (ros/params/nav2_params.yaml). So map_server is off by
    # default in a known room and `map_server:=true` is what brings a file back — the very first
    # boot of a room nobody has ever mapped, or a session that must start from a frozen pgm.
    map_server = LaunchConfiguration("map_server").perform(context).lower() == "true"
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
    if map_server and runs_here(side, "map_server"):
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
    return nodes


def _describe(context: LaunchContext) -> list:  # type: ignore[type-arg]
    side = LaunchConfiguration("side").perform(context)
    params = LaunchConfiguration("params_file")
    # A crash of one Nav2 node takes the whole container with it (SIGABRT at a goal, 2026-09-10
    # 16:06: the board drove nothing until a stack restart). Respawned, the container is back in
    # seconds WITH its nodes, described anew for that start (pepin_bringup.launch_kit: a reload
    # of the first start's descriptions would lose the controller's cmd_vel -> cmd_vel_nav remap
    # and drive the wheels past the velocity smoother), and the laptop's bring-up activates the
    # board's nodes again (goal_server, pepin.deployment.next_transition).
    container, load = respawned_container(
        f"nav2_container_{side}" if side != "all" else "nav2_container",
        nav_parts,
        # Planning yields to the sensor nodes (nice -10) under load.
        "nice -n 5",
        # The whole params file goes to the container process as well: the costmaps are
        # sub-nodes (/local_costmap/local_costmap) created inside controller/planner and
        # only see parameters given to the process, not the ones given to their parents.
        parameters=[params],
    )
    # The ROS nodes; the watches below are plain processes and start at once.
    actions: list = [container, load]  # type: ignore[type-arg]
    if runs_here(side, "run_recorder"):
        # WHICH RECORDER WRITES THIS DRIVE (the ``recorder`` argument, the board's
        # PEPIN_RECORDER). ``jsonl`` is pepin_bringup.run_recorder, which subscribes to every
        # topic and writes the numbered tape itself: 34-43 % of a core for the deserialisation,
        # the TF buffer and json.dumps of 450 floats ten times a second. ``bag`` is
        # pepin_bringup.bag_recorder, a node with no subscription to any of those topics that
        # starts and stops `ros2 bag record` (MCAP, no compression) on the same word — the board
        # then copies serialised bytes, and ros/tools/bag_to_tape.py makes the tape on the
        # laptop. Default jsonl until the two are measured on the robot (CLAUDE.md rule 19: a
        # default flips on a measurement, not on a design).
        #
        # As a module, like link_watch: the image's console scripts are generated at build time
        # and the sources are mounted over them, so a new executable would need a rebuild.
        if LaunchConfiguration("recorder").perform(context) == "bag":
            actions.append(
                ExecuteProcess(
                    cmd=["python3", "-m", "pepin_bringup.bag_recorder"],
                    output="screen",
                    **RESPAWN,
                )
            )
        else:
            actions.append(
                ExecuteProcess(
                    cmd=[
                        "python3",
                        "-m",
                        "pepin_bringup.run_recorder",
                        "--ros-args",
                        # Which of the two readers of map -> base_link writes the tape's `loc`
                        # rows (the `loc_from` flag).
                        # The goal server parses /tf here anyway and republishes what it reads on
                        # /pose, so this node needs no listener of its own — but only where that
                        # node is on THIS machine. On a split stack it is the laptop's, and a tape
                        # that lost its pose rows with the WiFi is the one thing this recorder is
                        # on the board to prevent, so there it reads the edge itself.
                        "-p",
                        f"loc_from:={'pose_topic' if runs_here(side, 'goal_server') else 'tf'}",
                    ],
                    output="screen",
                    **RESPAWN,
                )
            )
    if runs_here(side, "goal_server"):
        # Waits for orders on a socket so a goal costs a socket write, not a client boot.
        # Its places book follows the map in use: /maps/flat3.yaml -> /maps/flat3.places.yaml.
        places: object = PythonExpression(
            ["'", LaunchConfiguration("map"), "'.rsplit('.', 1)[0] + '.places.yaml'"]
        )
        actions.append(
            Node(
                package="pepin_bringup",
                executable="goal_server",
                output="screen",
                parameters=[{"places": places, "side": side}],
                **RESPAWN,
            )
        )
    if runs_here(side, "link_watch"):
        # As a module, not a console script: a new entry point needs an image rebuild, a module
        # on the mounted package path does not.
        actions.append(
            ExecuteProcess(cmd=["python3", "-m", "pepin_bringup.link_watch"], output="screen")
        )
    return actions


# How long a node of this launch is given to end on SIGINT before the launch escalates to
# SIGTERM, and then how long before SIGKILL. launch's own defaults are 5 s and 5 s, which is
# under RTAB-Map's close of a 20-28 GB database and under the board's dozen nodes leaving DDS:
# every shutdown ended in SIGKILLs mid-write, and eight of them left ros/maps/rtabmap.db
# malformed (2026-09-13). The window is the container's own stop window
# (pepin.deployment.CONTAINER_STOP_TIMEOUT_S, ros/lib.sh, board/pepin-ros.service), so on a
# `docker stop` nothing inside escalates before docker's SIGKILL at its end.
SHUTDOWN = [
    SetLaunchConfiguration("sigterm_timeout", str(CONTAINER_STOP_TIMEOUT_S)),
    SetLaunchConfiguration("sigkill_timeout", "5"),
]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            *SHUTDOWN,
            DeclareLaunchArgument("map", default_value="/maps/20260903_182653_lap3_loop.yaml"),
            DeclareLaunchArgument("params_file", default_value="/params/nav2_params.yaml"),
            DeclareLaunchArgument("side", default_value="all", choices=list(SIDES)),
            # The pgm is out of the loop: on a known room RTAB-Map's grid is the only map. true
            # serves `map` through map_server again, which is what the FIRST boot of an unmapped
            # room needs.
            DeclareLaunchArgument("map_server", default_value="false", choices=["true", "false"]),
            # jsonl: the Python recorder writes the numbered tape on the board. bag: `ros2 bag
            # record` writes an MCAP bag instead and ros/tools/bag_to_tape.py makes the tape from
            # it on the laptop (ros/README.md, "Two recorders").
            DeclareLaunchArgument("recorder", default_value="jsonl", choices=["jsonl", "bag"]),
            OpaqueFunction(function=_describe),
        ]
    )
