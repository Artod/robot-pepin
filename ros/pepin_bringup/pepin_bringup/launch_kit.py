"""Pieces shared by the launch files (ros/pepin_bringup/launch/*.launch.py).

The launch files import this module from the package, which is on the launch's path on both
machines. Nothing here runs inside a node.
"""

from collections.abc import Callable
from typing import Any

from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.launch_context import LaunchContext
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes

# How long a dead container stays dead before the launch starts it again.
RESPAWN_DELAY_S = 2.0

# The nodes one container carries, described for ONE start of it (see respawned_container).
PartsFactory = Callable[[LaunchContext], list[Any]]


def respawned_container(name: str, parts: PartsFactory, prefix: str, **process: Any) -> list[Any]:
    """A component container that comes back after a crash WITH its nodes, wired as at the
    first start: the process is respawned ``RESPAWN_DELAY_S`` after it dies, and every start of it
    loads ``parts(context)``, descriptions built again for that start. Returns the container and
    its load handler; ``process`` goes to the container (e.g. its ``parameters``).

    Both halves were measured on the board on 2026-09-24. launch_ros loads a container's own
    ``composable_node_descriptions`` once, so a respawned lidar_container came back EMPTY. And a
    ComposableNode keeps its remappings as a one-shot generator
    (launch_ros.utilities.normalize_remap_rules), so a description loaded a second time loads
    with NO remaps: the respawned hull filter listened to /scan instead of the driver's scan and
    published ``scan_filtered``; /scan stayed silent (scratch/link_autopsy/respawn_remap_probe.py).
    """
    container = ComposableNodeContainer(
        name=name,
        namespace="",
        package="rclcpp_components",
        executable="component_container_isolated",
        output="screen",
        prefix=prefix,
        composable_node_descriptions=[],
        respawn=True,
        respawn_delay=RESPAWN_DELAY_S,
        **process,
    )

    def load(_started: Any, context: LaunchContext) -> list[Any]:
        """The load of this start's nodes into the process that has just started."""
        nodes = parts(context)
        return [LoadComposableNodes(target_container=container, composable_node_descriptions=nodes)]

    return [container, RegisterEventHandler(OnProcessStart(target_action=container, on_start=load))]
