"""Pieces shared by the launch files (ros/pepin_bringup/launch/*.launch.py).

The launch files import this module from the package, which is on the launch's path on both
machines. Nothing here runs inside a node.
"""

import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
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


# THE STALL LOOK'S BT NODE. AskGaze is a plugin of its own package (ros/pepin_gaze_bt), built into
# the laptop image by ros/laptop-build.sh gaze; bt_navigator loads it by this name. A tree that
# names AskGaze cannot be loaded without it, so an image that lacks the package gets the same tree
# with the stall look cut out — exactly the tree before the look existed.
GAZE_BT_PACKAGE = "pepin_gaze_bt"
GAZE_BT_LIBRARY = "pepin_ask_gaze_bt_node"
GAZE_BT_NODE = "AskGaze"
STALL_LOOK = "StallLook"  # the name of the tree's element that holds the whole look
NO_GAZE_TREE = Path("/tmp/pepin_nav_to_pose.no_gaze.xml")


def gaze_bt_installed() -> bool:
    """Whether this image carries the AskGaze plugin (the ament index knows its package)."""
    try:
        from ament_index_python.packages import (  # lazy: only a launch has it
            PackageNotFoundError,
            get_package_prefix,
        )
    except ImportError:
        return False
    try:
        get_package_prefix(GAZE_BT_PACKAGE)
    except PackageNotFoundError:
        return False
    return True


def without_ask_gaze(xml: str) -> str:
    """The tree with every element named ``StallLook`` cut out of its parent, comments kept;
    ``ValueError`` when an AskGaze stands outside one (cutting it alone could leave a control
    node empty, which no tree loads)."""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(xml, parser=parser)
    for parent in list(root.iter()):
        for child in list(parent):
            if child.get("name") == STALL_LOOK:
                parent.remove(child)
    if root.find(f".//{GAZE_BT_NODE}") is not None:
        raise ValueError(f"an {GAZE_BT_NODE} outside an element named {STALL_LOOK}")
    return ET.tostring(root, encoding="unicode")


def bt_navigator_overrides(
    tree: Path, installed: bool, stripped: Path = NO_GAZE_TREE
) -> dict[str, Any]:
    """bt_navigator's parameters for the stall look: the plugin when the image has it, else
    the default tree with the look cut out (written to ``stripped``)."""
    if installed:
        return {"plugin_lib_names": [GAZE_BT_LIBRARY]}
    stripped.write_text(without_ask_gaze(tree.read_text()))
    return {"default_nav_to_pose_bt_xml": str(stripped)}
