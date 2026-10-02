"""Nav2 on a saved map: one composed process, only what a drive needs, on this Mac.

nav2_bringup's launch adds docking, route, waypoint and smoother servers and a collision
monitor; this launch composes the planner, controller, behaviours, bt_navigator and the velocity
smoother into one container (map_server only on request), with the goal server and the run
recorder beside it. ``ros/laptop.sh nav`` runs it in the ``pepin-macnav`` container: the board is
a sensor box whose scans, ToF and odometry arrive over the routers, and the velocity goes back on
/cmd_vel. Arguments: ``map`` (whose places book the goal server reads), ``params_file``,
``map_server`` and ``recorder``.
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

from pepin.deployment import CONTAINER_STOP_TIMEOUT_S, NAV_NODES

# Our own nodes come back by themselves after this pause: a code change costs one kicked process
# (ros/laptop.sh kick <node>; SIGINT, what the launch sends at shutdown) instead of a stack
# restart. (Under CycloneDDS every respawned command started through a ghost wait of the bridge's
# admin; tag alt/cyclone-bridges-2026-09-20.)
RESPAWN = {"respawn": True, "respawn_delay": 2.0}


def nav_parts(context: LaunchContext) -> list:  # type: ignore[type-arg]
    """The Nav2 nodes, described anew on every call (a respawned container loads fresh ones, see
    pepin_bringup.launch_kit): the servers of pepin.deployment.NAV_NODES, map_server when
    ``map_server:=true`` asks for it, and the lifecycle manager."""
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
    nodes = [catalogue[name] for name in NAV_NODES]
    if map_server:
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
            name="lifecycle_manager_navigation",
            parameters=[{"autostart": True, "node_names": list(NAV_NODES)}],
        )
    )
    return nodes


def _describe(context: LaunchContext) -> list:  # type: ignore[type-arg]
    params = LaunchConfiguration("params_file")
    # A crash of one Nav2 node takes the whole container with it (SIGABRT at a goal, 2026-09-10
    # 16:06: the cart drove nothing until a stack restart). Respawned, the container is back in
    # seconds WITH its nodes, described anew for that start (pepin_bringup.launch_kit: a reload
    # of the first start's descriptions would lose the controller's cmd_vel -> cmd_vel_nav remap
    # and drive the wheels past the velocity smoother).
    container, load = respawned_container(
        "nav2_container",
        nav_parts,
        # Planning yields to the sensor nodes (nice -10) under load.
        "nice -n 5",
        # The whole params file goes to the container process as well: the costmaps are
        # sub-nodes (/local_costmap/local_costmap) created inside controller/planner and
        # only see parameters given to the process, not the ones given to their parents.
        parameters=[params],
    )
    # The ROS nodes; the recorder and the goal server below are processes of their own.
    actions: list = [container, load]  # type: ignore[type-arg]
    # WHICH RECORDER WRITES THIS DRIVE (the ``recorder`` argument). ``jsonl`` is
    # pepin_bringup.run_recorder, which subscribes to every topic and writes the numbered tape
    # itself: 34-43 % of a core for the deserialisation, the TF buffer and json.dumps of 450
    # floats ten times a second. ``bag`` is pepin_bringup.bag_recorder, a node with no
    # subscription to any of those topics that starts and stops `ros2 bag record` (MCAP, no
    # compression) on the same word, and ros/tools/bag_to_tape.py makes the tape from it. Default
    # jsonl until the two are measured (CLAUDE.md rule 19: a default flips on a measurement, not
    # on a design).
    #
    # As modules: the image's console scripts are generated at build time and the sources are
    # mounted over them, so a new executable would need a rebuild.
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
                    # Which of the two readers of map -> base_link writes the tape's `loc` rows
                    # (the `loc_from` flag): the goal server beside it parses /tf anyway and
                    # republishes what it reads on /pose, so this node needs no listener of its
                    # own.
                    "-p",
                    "loc_from:=pose_topic",
                ],
                output="screen",
                **RESPAWN,
            )
        )
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
            parameters=[{"places": places}],
            **RESPAWN,
        )
    )
    return actions


# How long a node of this launch is given to end on SIGINT before the launch escalates to
# SIGTERM, and then how long before SIGKILL. launch's own defaults are 5 s and 5 s, which is
# under RTAB-Map's close of a 20-28 GB database and under a dozen nodes leaving DDS: every
# shutdown ended in SIGKILLs mid-write, and eight of them left ros/maps/rtabmap.db malformed
# (2026-09-13). The window is the container's own stop window
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
            # The pgm is out of the loop: on a known room RTAB-Map's grid is the only map. true
            # serves `map` through map_server again, which is what the FIRST boot of an unmapped
            # room needs.
            DeclareLaunchArgument("map_server", default_value="false", choices=["true", "false"]),
            # jsonl: the Python recorder writes the numbered tape. bag: `ros2 bag record` writes
            # an MCAP bag instead and ros/tools/bag_to_tape.py makes the tape from it
            # (ros/README.md, "Two recorders").
            DeclareLaunchArgument("recorder", default_value="jsonl", choices=["jsonl", "bag"]),
            OpaqueFunction(function=_describe),
        ]
    )
