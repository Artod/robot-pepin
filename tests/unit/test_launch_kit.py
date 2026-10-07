"""pepin_bringup.launch_kit: a respawned container loads nodes described for THAT start.

launch_ros is not installed here, so the four launch classes the kit composes are faked; each
fake keeps what it was handed. The trap the kit exists for is launch_ros's own (a
ComposableNode's remappings are a one-shot generator, so a description loaded twice loads the
second time with no remaps: scratch/link_autopsy/respawn_remap_probe.py, measured on the board
on 2026-09-24); what is held here is that no description is ever loaded twice.
"""

from __future__ import annotations

import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest
import ros_stubs


class Recorded:
    """A launch action or handler that keeps its arguments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def _kit(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """pepin_bringup.launch_kit imported over fake launch modules."""
    ros_stubs.install()
    fakes = {
        "launch": {},
        "launch.actions": {"RegisterEventHandler": type("RegisterEventHandler", (Recorded,), {})},
        "launch.event_handlers": {"OnProcessStart": type("OnProcessStart", (Recorded,), {})},
        "launch.launch_context": {"LaunchContext": type("LaunchContext", (), {})},
        "launch_ros": {},
        "launch_ros.actions": {
            "ComposableNodeContainer": type("ComposableNodeContainer", (Recorded,), {}),
            "LoadComposableNodes": type("LoadComposableNodes", (Recorded,), {}),
        },
    }
    for name, attributes in fakes.items():
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "pepin_bringup.launch_kit", raising=False)
    import pepin_bringup.launch_kit as kit

    return kit


def test_every_start_loads_nodes_its_factory_built_for_that_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = _kit(monkeypatch)
    built: list[str] = []

    def parts(context: str) -> list[str]:
        built.append(context)
        return [f"described for {context}"]

    container, handler = kit.respawned_container(
        "lidar_container", parts, "nice -n -10", parameters=["/params/p.yaml"]
    )
    assert built == [], "nothing is described before the process starts"
    started = handler.args[0]
    assert started.kwargs["target_action"] is container
    first = started.kwargs["on_start"](object(), "first start")
    respawn = started.kwargs["on_start"](object(), "respawn")
    assert built == ["first start", "respawn"], "a fresh description for every start"
    assert first[0].kwargs["composable_node_descriptions"] == ["described for first start"]
    assert respawn[0].kwargs["composable_node_descriptions"] == ["described for respawn"]
    assert all(load[0].kwargs["target_container"] is container for load in (first, respawn))


def test_the_container_itself_respawns_and_carries_no_nodes_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nodes handed to the container action would be loaded once, at the first start only."""
    kit = _kit(monkeypatch)
    container, _ = kit.respawned_container("nav2_container_board", lambda _c: [], "nice -n 5")
    assert container.kwargs["composable_node_descriptions"] == []
    assert container.kwargs["respawn"] is True
    assert 0.0 < container.kwargs["respawn_delay"] <= 5.0
    assert container.kwargs["name"] == "nav2_container_board"
    assert container.kwargs["prefix"] == "nice -n 5"
    assert container.kwargs["executable"] == "component_container_isolated"


TREE = Path(__file__).resolve().parents[2] / "ros/params/pepin_nav_to_pose.xml"


def test_an_image_with_askgaze_loads_the_plugin_and_the_tree_as_it_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kit = _kit(monkeypatch)
    overrides = kit.bt_navigator_overrides(TREE, True, tmp_path / "cut.xml")
    assert overrides == {"plugin_lib_names": ["pepin_ask_gaze_bt_node"]}
    assert not (tmp_path / "cut.xml").exists()
    assert kit.gaze_bt_installed() is False  # no ament index here


def test_an_image_without_askgaze_gets_the_tree_before_the_stall_look(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cut out, the stall look leaves the FollowPath recovery exactly as it was: the controller
    check, the two clears, the wait and the replan (and the controller the plan was made under)."""
    kit = _kit(monkeypatch)
    overrides = kit.bt_navigator_overrides(TREE, False, tmp_path / "cut.xml")
    assert overrides == {"default_nav_to_pose_bt_xml": str(tmp_path / "cut.xml")}
    cut = ET.parse(tmp_path / "cut.xml")
    assert not list(cut.iter("AskGaze")) and not cut.findall(".//Fallback[@name='StallLook']")
    recovery = cut.find(
        ".//RecoveryNode[@name='FollowPath']/Fallback/Sequence[@name='LookAndReplan']"
    )
    assert recovery is not None
    assert [child.tag for child in recovery] == [
        "WouldAControllerRecoveryHelp",
        "ClearEntireCostmap",
        "ClearEntireCostmap",
        "Wait",
        "ComputePathToPose",
        "Script",
    ]
    whole = ET.parse(TREE)
    assert len(list(whole.iter("AskGaze"))) == 2
    assert len(list(cut.iter())) == len(list(whole.iter())) - 6  # the look's six nodes
    stray = '<root><Sequence><Wait/><Fallback name="x"><AskGaze/></Fallback></Sequence></root>'
    with pytest.raises(ValueError, match="outside an element named StallLook"):
        kit.without_ask_gaze(stray)
    assert "<!--" in (tmp_path / "cut.xml").read_text(), "the tree's comments stay"
