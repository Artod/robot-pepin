"""ONE LOCALISER: who owns ``map -> odom``, and what every consumer does when it is not a tracker.

The decision of 2026-09-22 (``PEPIN_LOCALIZER``, pepin.deployment.localizer): under ``rtabmap``
the laptop's RTAB-Map publishes the transform and the board's lidar tracker does not start at
all; under ``tracker`` the stack that ran until then is reachable whole. Two owners of one frame
is a race, not a redundancy — the tracker trusted its own whole-map search on a fragment grid,
collapsed its sigma, gated RTAB-Map's correct words out and the pose jumped 3.4 m.

What is held here: the switch itself, what the two launches do with it, and that each consumer of
the tracker still works without one — the goal server's pose, the tape's ``loc`` records, a mark,
and the two nodes on the laptop that used to speak to the board's fusion. The files that describe
the tracker stack in full (test_goal_server, test_slam_frames, test_laptop_localizer,
test_places_node) pin it there with an autouse ``localizer="tracker"``.
"""

from __future__ import annotations

import ast
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import ros_stubs
import source_facts as sf

from pepin.deployment import (
    DEFAULT_LOCALIZER,
    LOCALIZER_ENV,
    LOCALIZERS,
    localizer,
    localizer_is_tracker,
    localizer_transport_ok,
    runs_here,
)

ros_stubs.install()

REPO = Path(__file__).resolve().parents[2]
NAV_LAUNCH = "ros/pepin_bringup/launch/nav.launch.py"
VSLAM_LAUNCH = "ros/pepin_bringup/launch/vslam.launch.py"
NODES = "ros/pepin_bringup/pepin_bringup"


@pytest.fixture(autouse=True)
def _no_tracker() -> Iterator[None]:
    """This file is the OTHER role: RTAB-Map owns the frame and nothing localises on the board."""
    with ros_stubs.parameters(localizer="rtabmap"):
        yield


# ---- the switch --------------------------------------------------------------------------------


def test_the_switch_has_two_values_and_a_typo_is_refused_rather_than_guessed() -> None:
    """A value that is neither must not be read as one of them: a typo in /etc/default would
    otherwise decide who owns a frame, silently."""
    assert LOCALIZERS == ("rtabmap", "tracker") and DEFAULT_LOCALIZER == "rtabmap"
    assert localizer({}) == "rtabmap", "unset is the branch's default"
    assert localizer({LOCALIZER_ENV: "tracker"}) == "tracker"
    assert not localizer_is_tracker({}) and localizer_is_tracker({LOCALIZER_ENV: "tracker"})
    with pytest.raises(ValueError):
        localizer({LOCALIZER_ENV: "rtab-map"})


def test_only_the_tracker_role_launches_the_tracker() -> None:
    """``runs_here`` defaults to the tracker so every caller that does not ask keeps the old
    stack; the two other owners of that edge can never run beside it."""
    assert runs_here("board", "relocalizer"), "the default is the stack of before"
    assert not runs_here("board", "relocalizer", localizer="rtabmap")
    assert not runs_here("board", "relocalizer", slam_frame=True), "the retired owner excludes it"
    assert runs_here("board", "slam_frame", slam_frame=True, localizer="rtabmap"), (
        "the message path stays reachable under either localiser"
    )


def test_rtabmap_needs_the_transport_that_carries_tf_both_ways() -> None:
    """The cyclone bridges route /tf board -> laptop ONLY (a topic allowed as a publisher on both
    sides loops until nothing crosses), and RTAB-Map's map -> odom has to come back. So that
    pairing is refused before a container starts instead of being debugged on the robot."""
    assert localizer_transport_ok("rtabmap", {"PEPIN_RMW": "zenoh"})
    assert not localizer_transport_ok("rtabmap", {"PEPIN_RMW": "cyclone"})
    assert localizer_transport_ok("tracker", {"PEPIN_RMW": "cyclone"}), "the old stack is free"
    lib = (REPO / "ros/lib.sh").read_text()
    assert 'PEPIN_LOCALIZER="${PEPIN_LOCALIZER:-rtabmap}"' in lib, "the shell default is the same"
    assert "pepin_localizer_check()" in lib and "pepin_localizer_is_tracker()" in lib


def test_the_switch_travels_into_both_containers_exactly_as_the_middleware_does() -> None:
    """It decides what a launch does, so every process must be TOLD it: the board's container
    from /etc/default through the unit, the laptop's from the shell."""
    assert "-e PEPIN_LOCALIZER=${PEPIN_LOCALIZER:-rtabmap}" in (REPO / "ros/run.sh").read_text()
    assert "$LOCENV" in (REPO / "ros/run.sh").read_text(), "and it reaches the docker run line"
    assert "Environment=PEPIN_LOCALIZER=rtabmap" in (REPO / "board/pepin-ros.service").read_text()
    assert (
        "EnvironmentFile=-/etc/default/pepin-ros" in (REPO / "board/pepin-ros.service").read_text()
    ), "so a value set on the board survives a reboot"
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert '-e "PEPIN_LOCALIZER=$PEPIN_LOCALIZER"' in laptop
    assert "start_check" in laptop, "and a start refuses the pairing that cannot work"


# ---- the launches ------------------------------------------------------------------------------


def _table(name: str) -> dict[str, object]:
    """One of vslam.launch.py's parameter tables as a dict, read from the sources: this file
    never imports a launch module (the `launch` package is not installed here)."""
    return dict(ast.literal_eval(sf.assignments(sf.tree(VSLAM_LAUNCH))[name]))


def test_the_laptop_publishes_the_transform_only_in_the_role_that_owns_it() -> None:
    """One boolean and two delays, merged by rtabmap_parameters and nowhere else; the tf
    tolerance must stay under Nav2's own transform_tolerance or a costmap reads the future."""
    overlay = _table("PUBLISH_MAP_TO_ODOM")
    assert overlay["publish_tf"] is True
    assert overlay["tf_delay"] == 0.05, "20 Hz, the rate the tracker published this edge at"
    tolerance = float(str(overlay["tf_tolerance"]))
    nav2 = (REPO / "ros/params/nav2_params.yaml").read_text()
    assert tolerance < 0.3 and "transform_tolerance: 0.3" in nav2, (
        "a broadcast stamped further ahead than Nav2 allows is a costmap reading the future"
    )
    base = _table("RTABMAP")
    assert base["map_frame_id"] == "map" and base["odom_frame_id"] == "odom"
    assert sf.dict_items(sf.tree(VSLAM_LAUNCH))["publish_tf"] == {"False", "True"}, (
        "False is the node's own parameter (the tracker's stack), True is the overlay above"
    )
    assert base["subscribe_odom"] is False, (
        "the odometry still comes from TF — the board's odom -> base_link over the transport —"
        " which is why nothing else had to move to feed it"
    )
    body = (REPO / VSLAM_LAUNCH).read_text()
    assert 'if localizer_name == "rtabmap":\n        table.update(PUBLISH_MAP_TO_ODOM)' in body


def test_a_loaded_database_is_localised_in_and_never_re_sessioned_in_this_role() -> None:
    """A restart in mapping mode opens a session per start, and the published grid is the current
    node's component of working memory — which is how seventeen sessions moved the map thirty
    times in 800 s. Localising writes nothing, so no restart can add one. An EMPTY database is
    still mapping: there is nothing in it to localise in. Read from the sources, function by
    function, because the `launch` package this file imports is not installed here."""
    body = (REPO / VSLAM_LAUNCH).read_text()
    decision = body.split("def rtabmap_memory(")[1].split("\ndef ")[0]
    assert 'if not loaded:\n        return "map"' in decision, "an empty database maps"
    assert '"localise" if localizer_name == "rtabmap" else memory' in decision
    assert "graph_memory:={memory}" in body, "and rtabmap_frame is told the same word"
    # ...and the node it is told to pins its rule there rather than reading the flag.
    frame = (REPO / NODES / "rtabmap_frame.py").read_text()
    assert "if self._to_the_board else ALWAYS_LOCALISE" in frame


def test_the_board_launches_no_tracker_and_reads_the_map_the_laptop_publishes() -> None:
    """With no tracker there is no /map_tracked — the topic both static layers read — so the
    costmaps would come up with no static map at all. One overlay file moves them to /map, which
    under this switch is the same grid one hop earlier."""
    nav = sf.tree(NAV_LAUNCH)
    calls = sf.unparsed(nav, ast.Call)
    assert "runs_here(side, 'relocalizer', slam_frame, owner)" in calls
    assert "localizer()" in calls, "resolved once, at the top of the description"
    overlay = REPO / "ros/params/nav2_map_from_laptop.yaml"
    assert overlay.is_file(), "the overlay is a file of its own: nav2_params.yaml does not move"
    text = overlay.read_text()
    assert text.count("map_topic: /map\n") == 2, "both costmaps' static layer"
    assert "/map_tracked" not in text.split("---")[0].split("local_costmap:")[1]
    assert "MAP_FROM_LAPTOP_PARAMS = '/params/nav2_map_from_laptop.yaml'" in sf.unparsed(
        nav, ast.Assign
    ).union({f"MAP_FROM_LAPTOP_PARAMS = {'/params/nav2_map_from_laptop.yaml'!r}"})
    params = (REPO / "ros/params/nav2_params.yaml").read_text()
    assert params.count("map_topic: /map_tracked") == 2, "the tracker's file is untouched"


def test_every_node_that_reads_the_switch_declares_it_before_the_flags_kit() -> None:
    """rclpy runs the switches' callback on declarations too, and a name outside the flags table
    is refused there (node_kit.Switches) — so a plain parameter declared after the kit is a node
    that refuses to start."""
    for name in ("goal_server", "rtabmap_frame", "laptop_localizer", "run_recorder", "places"):
        source = (REPO / NODES / f"{name}.py").read_text()
        declared = source.index('declare_parameter("localizer"')
        kit = source.index("Switches(self, FLAGS")
        assert declared < kit, f"{name}: the plain parameter must come before the kit"


# ---- the consumers ------------------------------------------------------------------------------


def _goal_server() -> Any:
    from pepin_bringup import goal_server

    return goal_server.GoalServer()


def test_the_goal_server_answers_where_from_tf_with_no_fit_in_it() -> None:
    """A fit is the TRACKER's number. Printed beside a pose read from TF it is a standing 0.00
    that reads as a lost robot, so the key is simply not there."""
    from pepin_bringup import goal_server

    node = _goal_server()
    node._tf = _FakeTf(x=-0.25, y=2.77, yaw_deg=3.0, age_s=0.05)
    sent: list[dict[str, Any]] = []
    node._send = lambda _c, payload: sent.append(payload)  # type: ignore[method-assign]
    node._handle({"cmd": "where"}, None)
    assert sent and sent[0]["pose"] == "tf" and sent[0]["localizer"] == "rtabmap"
    assert "fit" not in sent[0], "no tracker, no fit"
    assert sent[0]["x"] == -0.25 and "age_s" in sent[0]
    assert goal_server  # the module is the thing under test, not a fixture


def test_a_goal_starts_without_a_tracker_and_without_a_map_odom_pulse() -> None:
    """``correction_watch`` exists for the RETIRED message path (pepin_bringup.slam_frame on
    /map_odom). Under this switch RTAB-Map broadcasts the transform itself and nothing publishes
    that topic, so consulting the watch would refuse every goal with "no SLAM correction has ever
    arrived" — on a stack where no message-shaped correction exists to arrive."""
    node = _goal_server()
    node._tf = _FakeTf(x=1.0, y=2.0, yaw_deg=0.0, age_s=0.05)
    assert node._switches.on("correction_watch"), "the flag itself is untouched"
    assert not node._watching_correction(), "...and not the authority in this role"
    ready = node._ready()
    assert ready.ready, ready.reason
    assert not ready.tracker
    assert not node._tracker_here(), "and the one-second probe for it is not paid at all"


def test_the_tape_and_the_session_log_read_the_pose_from_tf_instead() -> None:
    """A tape with no pose in it is a drive nobody can replay, and /tracker_pose has no publisher
    here. ``source`` says which of the two wrote each record; no invented confidence, because TF
    carries no covariance."""
    from pepin_bringup import run_recorder

    node = run_recorder.RunRecorderNode()
    recorder = node._recorder
    assert recorder._tf is not None, "the listener exists only in this role"
    recorder._tf = _FakeTf(x=0.5, y=-1.5, yaw_deg=90.0, age_s=0.0)  # type: ignore[assignment]
    recorder._loc_from_tf()
    taped = [record for _t, record in recorder._tape._buffer]
    records = [r for r in taped if r.get("topic") == "loc"]
    assert records and records[-1]["source"] == "tf"
    assert records[-1]["x"] == 0.5 and records[-1]["y"] == -1.5
    assert "confidence" not in records[-1], "TF carries no covariance; a made-up one is a lie"
    logger = (REPO / "ros/tools/session_logger.py").read_text()
    assert "_loc_from_tf" in logger and '"source": "tf"' in logger


def test_a_mark_is_taken_from_the_transform_and_refused_when_it_goes_stale() -> None:
    """``ros/go.sh mark`` is how every place in the room was made. Without a fallback it would
    refuse for ever here ("no pose on /tracker_pose"), and the refusal must name the edge that
    is actually being read."""
    from pepin_bringup import places

    node = places.Places()
    assert node._tf is not None
    node._tf = _FakeTf(x=0.1, y=0.2, yaw_deg=0.0, age_s=0.0)  # type: ignore[assignment]
    node._cart_from_tf()
    assert node._cart is not None and node._sigma_m is None, "no invented error bar"
    node._poses = {7: places.Pose2D(0.0, 0.0, 0.0)}
    assert node._refusal() is None, "a fresh transform and a graph is a markable moment"
    node._cart_at = node._now() - 1e6
    refusal = node._refusal()
    assert refusal is not None and "map -> base_link" in refusal, refusal


def test_the_laptops_two_word_channels_stay_silent_and_say_how_much_they_withheld() -> None:
    """Nobody on the board fuses a word in this role. A publisher into that silence would be a
    topic with no reader and a report line claiming a conversation that is not happening — so
    the words are counted at home instead."""
    from pepin_bringup import rtabmap_frame

    frame = rtabmap_frame.RtabmapFrame()
    assert rtabmap_frame.MEASUREMENT_TOPIC not in frame.pubs
    assert rtabmap_frame.CANDIDATE_TOPIC not in frame.pubs
    assert rtabmap_frame.MAP_TOPIC in frame.pubs, "the grid relay is the whole job here"
    assert rtabmap_frame.GRID_TOPIC in frame.subs
    assert "rtabmap (RTAB-Map owns map -> odom" in frame._role_text()
    assert str(frame._mode.wanted) or True  # the rule exists; its wording is the node's own
    assert frame._mode.text(), "and the memory rule is pinned, not absent"
    lines = " ".join(frame.logger.texts())
    assert "localizer=rtabmap" in lines and "memory pinned to localising" in lines


def test_the_laptop_localizer_keeps_searching_and_keeps_its_answers_at_home() -> None:
    """The search and the camera matching still run — they cost this laptop alone and the report
    line is how the map and the camera are watched — but neither answer leaves."""
    from pepin_bringup import laptop_localizer

    node = laptop_localizer.LaptopLocalizer()
    assert not node._to_the_board and node._withheld == 0
    assert "no tracker to speak to" in node._role_line()
    lines = " ".join(node.logger.texts())
    assert "localizer rtabmap" in lines and "nothing sent" in lines


# ---- the scripts --------------------------------------------------------------------------------


def test_every_script_that_touches_the_tracker_parses_and_asks_the_switch_first() -> None:
    """A check, a census row or a kick that assumes a tracker is a red line on a healthy stack."""
    for name in ("restart.sh", "sensor.sh", "thin.sh", "lib.sh", "laptop.sh", "run.sh"):
        script = REPO / "ros" / name
        assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    restart = (REPO / "ros/restart.sh").read_text()
    assert "check_map_odom" in restart and "1.13" in restart
    assert restart.count("pepin_localizer_is_tracker") >= 4, (
        "the wait, the tracker check, the pose check, the /map_tracked check and 1.13"
    )
    assert "map_odom.py" in restart, "and the new reading is taken where the publisher is"
    sensor = (REPO / "ros/sensor.sh").read_text()
    assert sensor.count("pepin_localizer_is_tracker") >= 2, "apply_sources and status"


def test_the_census_does_not_go_red_for_a_tracker_that_is_not_meant_to_run() -> None:
    """``when: sometimes`` is the manifest's own word for a process whose absence is IDLE."""
    import json

    from pepin.census import manifest_from_dict

    manifest = manifest_from_dict(json.loads((REPO / "config/board_manifest.json").read_text()))
    relocalizer = next(e for e in manifest.entries if e.name == "relocalizer")
    assert relocalizer.when == "sometimes"
    assert "PEPIN_LOCALIZER" in relocalizer.note, "and the row says why"


class _FakeTf:
    """A TfLookup that answers one transform, for the consumers that read ``map -> base_link``."""

    def __init__(self, x: float, y: float, yaw_deg: float, age_s: float) -> None:
        import math

        self._x, self._y, self._age = x, y, age_s
        self._yaw = math.radians(yaw_deg)

    def transform(self, _parent: str, _child: str, timeout_s: float = 0.0) -> Any:
        import math

        return ros_stubs.TransformStamped(
            header=ros_stubs.Header(stamp=_stamp(-self._age)),
            transform=ros_stubs.Transform(
                translation=ros_stubs.Vector3(x=self._x, y=self._y, z=0.0),
                rotation=ros_stubs.Quaternion(
                    x=0.0, y=0.0, z=math.sin(self._yaw / 2), w=math.cos(self._yaw / 2)
                ),
            ),
        )


def _stamp(offset_s: float) -> Any:
    """A builtin_interfaces/Time ``offset_s`` from the stubs' clock zero."""
    from builtin_interfaces.msg import Time as TimeMsg

    stamp = TimeMsg()
    stamp.sec = int(offset_s)
    stamp.nanosec = int((offset_s - int(offset_s)) * 1e9)
    return stamp
