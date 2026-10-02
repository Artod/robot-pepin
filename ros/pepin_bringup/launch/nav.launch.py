"""Nav2 on a saved map: one composed process, only what a drive needs, on this Mac.

nav2_bringup's launch adds docking, route, waypoint and smoother servers and a collision
monitor; this launch composes the planner, controller, behaviours, bt_navigator and the velocity
smoother into one container, with the goal server and the run recorder beside it.
``ros/laptop.sh nav`` runs it in the ``pepin-macnav`` container: the board is a sensor box
whose scans, ToF and odometry arrive over the routers, and the velocity goes back on /cmd_vel.
Arguments: ``map`` (the map this container was started with, which ros/goto.sh reads
off it), ``params_file`` and ``recorder``.
Command chain: controller/behaviors -> cmd_vel_nav -> velocity_smoother -> /cmd_vel -> base_bridge.

ONE OWNER OF ``map -> odom``: the laptop's RTAB-Map publishes it, and both costmaps' static layer
reads ``/map`` (RTAB-Map's grid, relayed by pepin_bringup.rtabmap_frame). The board tracker that
owned the edge before, and the message-path owner of it (``slam:=true``), are on the tag
alt/tracker-2026-09-22.
"""

from pathlib import Path
from typing import Any

import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetLaunchConfiguration,
)
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode
from pepin_bringup.launch_kit import bt_navigator_overrides, gaze_bt_installed, respawned_container

from pepin.deployment import CONTAINER_STOP_TIMEOUT_S, NAV_NODES, config_file
from pepin.geometry import BaseConfig
from pepin.speed import NAV2_SPEED, nav2_overrides

# Our own nodes come back by themselves after this pause: a code change costs one kicked process
# (ros/laptop.sh kick <node>; SIGINT, what the launch sends at shutdown) instead of a stack
# restart. (Under CycloneDDS every respawned command started through a ghost wait of the bridge's
# admin; tag alt/cyclone-bridges-2026-09-20.)
RESPAWN = {"respawn": True, "respawn_delay": 2.0}


def speed_overrides(context: LaunchContext) -> dict[str, dict[str, Any]]:
    """THE CART'S ONE SPEED: config/base.json's max_wheel_speed_m_s, the base server's wheel
    ceiling, for every controller's and the velocity smoother's linear limit (pepin.speed), given
    after the params file so it wins over the number there. Read at every start of the
    container, so a respawn picks up the file as it is; ros/speed.sh moves them all live."""
    speed = BaseConfig.from_json(config_file("base.json")).max_wheel_speed_m_s
    path = LaunchConfiguration("params_file").perform(context)
    with open(path) as f:
        overrides = nav2_overrides(speed, yaml.safe_load(f))
    print(
        f"nav speed {speed:.2f} m/s (config/base.json max_wheel_speed_m_s) on"
        f" {len(NAV2_SPEED)} parameters: " + ", ".join(p.label for p in NAV2_SPEED),
        flush=True,
    )
    return overrides


def gaze_overrides(context: LaunchContext) -> dict[str, Any]:
    """The stall look's BT node for bt_navigator (pepin_bringup.launch_kit): the AskGaze plugin
    when this image carries it (ros/laptop-build.sh gaze), else the tree without the look."""
    path = LaunchConfiguration("params_file").perform(context)
    with open(path) as f:
        tree = yaml.safe_load(f)["bt_navigator"]["ros__parameters"]["default_nav_to_pose_bt_xml"]
    installed = gaze_bt_installed()
    overrides = bt_navigator_overrides(Path(tree), installed)
    print(
        "nav: the stall look's AskGaze is loaded (pepin_gaze_bt)"
        if installed
        else "nav: no pepin_gaze_bt in this image: the tree runs without the stall look"
        f" ({overrides['default_nav_to_pose_bt_xml']}; ros/laptop-build.sh gaze)",
        flush=True,
    )
    return overrides


def nav_parts(context: LaunchContext) -> list:  # type: ignore[type-arg]
    """The Nav2 nodes, described anew on every call (a respawned container loads fresh ones, see
    pepin_bringup.launch_kit): the servers of pepin.deployment.NAV_NODES and the lifecycle
    manager. No map_server: both costmaps' static layers read RTAB-Map's grid on /map
    (ros/params/nav2_params.yaml), and an unmapped room starts from ros/laptop.sh vslam --fresh."""
    params = LaunchConfiguration("params_file")
    speed = speed_overrides(context)
    to_smoother = [("cmd_vel", "cmd_vel_nav")]
    catalogue = {
        "controller_server": ComposableNode(
            package="nav2_controller",
            plugin="nav2_controller::ControllerServer",
            name="controller_server",
            parameters=[params, speed["controller_server"]],
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
            parameters=[params, gaze_overrides(context)],
        ),
        "velocity_smoother": ComposableNode(
            package="nav2_velocity_smoother",
            plugin="nav2_velocity_smoother::VelocitySmoother",
            name="velocity_smoother",
            parameters=[params, speed["velocity_smoother"]],
            remappings=[("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")],
        ),
    }
    nodes = [catalogue[name] for name in NAV_NODES]
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
                # The tape's `loc` rows come from the goal server's /pose: it parses /tf anyway,
                # so this node needs no listener of its own.
                cmd=["python3", "-m", "pepin_bringup.run_recorder"],
                output="screen",
                **RESPAWN,
            )
        )
    # Waits for orders on a socket so a goal costs a socket write, not a client boot. Its places
    # are the graph's book (pepin_bringup.places on /places), never a file beside the map.
    actions.append(
        Node(
            package="pepin_bringup",
            executable="goal_server",
            output="screen",
            **RESPAWN,
        )
    )
    # The gaze arbiter: the one owner of the head (pepin_bringup.gaze). Beside Nav2 because the
    # tree's AskGaze, the plan, the local costmap and the goal status are its inputs; its door
    # for the tools is :3339, published on this Mac's loopback by ros/laptop.sh nav.
    actions.append(
        ExecuteProcess(
            cmd=["python3", "-m", "pepin_bringup.gaze"],
            output="screen",
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
            DeclareLaunchArgument("map", default_value="/maps/flat3_straight.yaml"),
            DeclareLaunchArgument("params_file", default_value="/params/nav2_params.yaml"),
            # jsonl: the Python recorder writes the numbered tape. bag: `ros2 bag record` writes
            # an MCAP bag instead and ros/tools/bag_to_tape.py makes the tape from it
            # (ros/README.md, "Two recorders").
            DeclareLaunchArgument("recorder", default_value="jsonl", choices=["jsonl", "bag"]),
            OpaqueFunction(function=_describe),
        ]
    )
