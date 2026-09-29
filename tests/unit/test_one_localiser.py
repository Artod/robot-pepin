"""ONE LOCALISER: RTAB-Map on the laptop owns ``map -> odom``, and every consumer on the board
reads the pose from TF.

The decision of 2026-09-22. Two owners of one frame is a race, not a redundancy — the board's
tracker trusted its own whole-map search on a fragment grid, collapsed its sigma, gated RTAB-Map's
correct words out and the pose jumped 3.4 m. The tracker is on the tag alt/tracker-2026-09-22.

What is held here: what the two launches do, and that each consumer works without a tracker — the
goal server's pose and placement gate, the tape's ``loc`` records, a mark, and rtabmap_frame's
placement word.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import pytest
import ros_stubs
import source_facts as sf
import yaml

ros_stubs.install()

REPO = Path(__file__).resolve().parents[2]
NAV_LAUNCH = "ros/pepin_bringup/launch/nav.launch.py"
VSLAM_LAUNCH = "ros/pepin_bringup/launch/vslam.launch.py"
NODES = "ros/pepin_bringup/pepin_bringup"


# ---- the launches ------------------------------------------------------------------------------


def _table(name: str) -> dict[str, object]:
    """One of vslam.launch.py's parameter tables as a dict, read from the sources: this file
    never imports a launch module (the `launch` package is not installed here)."""
    return dict(ast.literal_eval(sf.assignments(sf.tree(VSLAM_LAUNCH))[name]))


def test_the_laptop_publishes_the_transform() -> None:
    """One boolean and two delays, merged by rtabmap_parameters and nowhere else. The tf tolerance
    is how far AHEAD each broadcast is stamped, i.e. how long the last correction may stand as
    current on the far side of the radio: 0.5 s since 2026-09-22, because the board's WiFi stalls
    0.4-1.2 s and a 0.1 s stamp went stale inside one spike. Its consumers' own waits are the
    other half of the budget and are held here beside it."""
    overlay = _table("PUBLISH_MAP_TO_ODOM")
    assert overlay["publish_tf"] is True
    tolerance = float(str(overlay["tf_tolerance"]))
    assert tolerance == 0.5, "one measured WiFi stall, with margin"
    nav2 = yaml.safe_load((REPO / "ros/params/nav2_params.yaml").read_text())
    assert nav2["global_costmap"]["global_costmap"]["ros__parameters"]["transform_tolerance"] >= (
        tolerance
    ), "the grid that still spans this edge waits at least as long as the stamp promises"
    assert nav2["bt_navigator"]["ros__parameters"]["transform_tolerance"] >= tolerance, (
        "the tree's map -> base_link is the last lookup on the board that crosses the WiFi"
    )
    assert nav2["local_costmap"]["local_costmap"]["ros__parameters"]["global_frame"] == "odom", (
        "and the controller's own grid stopped crossing it altogether"
    )
    base = _table("RTABMAP")
    assert base["map_frame_id"] == "map" and base["odom_frame_id"] == "odom"
    assert base["subscribe_odom"] is False, (
        "the odometry still comes from TF — the board's odom -> base_link over the transport —"
        " which is why nothing else had to move to feed it"
    )
    body = (REPO / VSLAM_LAUNCH).read_text()
    assert "    table.update(PUBLISH_MAP_TO_ODOM)\n" in body, "in every session"


def test_a_loaded_database_is_localised_in_and_never_re_sessioned() -> None:
    """A restart in mapping mode opens a session per start, and the published grid is the current
    node's component of working memory — which is how seventeen sessions moved the map thirty
    times in 800 s. Localising writes nothing, so no restart can add one. An EMPTY database is
    still mapping: there is nothing in it to localise in. Read from the sources, function by
    function, because the `launch` package this file imports is not installed here."""
    body = (REPO / VSLAM_LAUNCH).read_text()
    decision = body.split("def rtabmap_memory(")[1].split("\ndef ")[0]
    assert 'return "localise" if loaded else "map"' in decision, "an empty database maps"
    # ...and rtabmap_frame pins its memory rule there.
    frame = (REPO / NODES / "rtabmap_frame.py").read_text()
    assert "ModeRule(MODE_HOLD_S, ALWAYS_LOCALISE, GRAPH)" in frame


def test_both_costmaps_read_the_map_the_laptop_publishes() -> None:
    """Both static layers read /map — RTAB-Map's grid, relayed by rtabmap_frame — straight from
    ros/params/nav2_params.yaml, with no overlay file and no process parameter in between."""
    params = yaml.safe_load((REPO / "ros/params/nav2_params.yaml").read_text())
    for costmap in ("local_costmap", "global_costmap"):
        static = params[costmap][costmap]["ros__parameters"]["static_layer"]
        assert static["map_topic"] == "/map", costmap
        assert static["map_subscribe_transient_local"] is True, costmap
    assert "MAP_FROM_LAPTOP_PARAMS" not in (REPO / NAV_LAUNCH).read_text()
    assert not (REPO / "ros/params/nav2_map_from_laptop.yaml").exists()


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
    assert sent and sent[0]["pose"] == "tf"
    assert "fit" not in sent[0], "TF carries no fit"
    assert sent[0]["x"] == -0.25 and "age_s" in sent[0]
    assert goal_server  # the module is the thing under test, not a fixture


def _placed(node: Any, **fields: Any) -> None:
    """rtabmap_frame's latched word reaching the goal server: this start of RTAB-Map is placed
    (one recognition of the loaded map) unless ``fields`` say otherwise."""
    from pepin.watch import PLACEMENT_TOPIC, Placement

    word = {"updates": 40, "recognised": 1, "seeds": 0, "loaded": True, **fields}
    node.subs[PLACEMENT_TOPIC][1](
        ros_stubs.String(data=Placement(**word).to_json(0.0))  # type: ignore[arg-type]
    )


def test_a_goal_starts_on_a_fresh_placed_transform_and_nothing_else() -> None:
    """RTAB-Map broadcasts the transform itself: a fresh map -> base_link and a placed start are
    the whole of the evidence, and no tracker service is asked."""
    node = _goal_server()
    node._tf = _FakeTf(x=1.0, y=2.0, yaw_deg=0.0, age_s=0.05)
    _placed(node)
    ready = node._ready()
    assert ready.ready, ready.reason
    assert not {"where_am_i", "relocalize"} & set(node.service_clients), "no tracker to ask"


def test_a_goal_is_refused_on_rtabmap_s_saved_start_pose_until_it_is_placed() -> None:
    """2026-09-23: after a restart map -> base_link was milliseconds fresh and the cart "at home"
    at the bookshelf — RTAB-Map's SAVED start pose, 0 of 198 updates recognised. The goal server
    refuses that with what to do, and starts the goal once a recognition or a seed has placed
    this start; nothing heard at all is refused too, never read as placed."""
    from pepin.watch import BY_PLACEMENT

    node = _goal_server()
    node._tf = _FakeTf(x=0.0, y=0.0, yaw_deg=0.0, age_s=0.05)
    silent = node._ready()
    assert not silent.ready and silent.rule == BY_PLACEMENT and "nothing on" in silent.reason
    _placed(node, updates=198, recognised=0)
    saved = node._ready()
    assert not saved.ready and saved.rule == BY_PLACEMENT
    assert "0 of 198 updates" in saved.reason and "ros/goto.sh seed" in saved.reason
    _placed(node, updates=199, recognised=0, seeds=1)
    assert node._ready().ready, "the operator's seed places it"
    _placed(node, updates=198, recognised=0, required=False)
    assert node._ready().ready, "the laptop's flag off: the pose is taken as it is"
    node._tf = _FakeTf(x=0.0, y=0.0, yaw_deg=0.0, age_s=5.0)
    stale = node._ready()
    assert not stale.ready and stale.rule != BY_PLACEMENT, "a stale frame is refused first"


def test_the_goal_server_s_own_switch_lifts_a_refusal_of_silence() -> None:
    """Rule 19 on the board's side of the word: with the laptop's rtabmap_frame down, respawning
    or on code from before the word, nothing arrives and the laptop's flag cannot be set. The goal
    server's flag of the same name is the switch the refusal answers to, and the refusal says
    so; a mark is judged by the same rule."""
    from pepin.watch import BY_PLACEMENT

    node = _goal_server()
    node._tf = _FakeTf(x=0.0, y=0.0, yaw_deg=0.0, age_s=0.05)
    silent = node._ready()
    assert not silent.ready and silent.rule == BY_PLACEMENT
    assert "ros/flags.sh set goal_server start_needs_placement false" in silent.reason
    assert "ros/laptop.sh vslam" in silent.reason, "an old pepin-vslam is named, not only dead"
    assert node.mark("desk")["event"] == "error", "a mark waits for the same word"
    assert node._switches.set("start_needs_placement", False) is True
    assert node._ready().ready, "off: the fresh frame is enough, as before 2026-09-23"
    _placed(node, updates=198, recognised=0)
    assert node._ready().ready, "and a word of not placed is not asked either"
    assert "start_needs_placement=off" in node._switches.state()


def test_a_placed_word_does_not_outlive_the_node_that_said_it() -> None:
    """The laptop's vslam restarting: the old rtabmap_frame's latched "placed" must not stand for
    the new start of RTAB-Map, whose map -> odom is fresh at its saved pose before the new node
    has said anything. No publisher left is nobody saying; the new node's word is heard afresh."""
    from pepin.watch import BY_PLACEMENT, PLACEMENT_TOPIC

    node = _goal_server()
    node._tf = _FakeTf(x=0.0, y=0.0, yaw_deg=0.0, age_s=0.05)
    _placed(node)
    assert node._ready().ready
    node.publisher_counts[PLACEMENT_TOPIC] = 0
    gone = node._ready()
    assert not gone.ready and gone.rule == BY_PLACEMENT and "nothing on" in gone.reason
    node.publisher_counts[PLACEMENT_TOPIC] = 1
    _placed(node, updates=1, recognised=0)
    assert not node._ready().ready, "the new start's own word: not placed yet"


def _rtabmap_update(frame: Any, ref_id: int, matched: int = 0) -> None:
    """One /rtabmap/info: the node this update CREATED and the older node it recognised."""
    from pepin_bringup import rtabmap_frame

    frame.subs[rtabmap_frame.INFO_TOPIC][1](
        ros_stubs.Info(ref_id=ref_id, loop_closure_id=matched, header=ros_stubs.Header())
    )


def _placement_said(frame: Any) -> Any:
    """The last placement rtabmap_frame published (latched), as the board reads it."""
    from pepin.watch import PLACEMENT_TOPIC, Placement

    frame._publish()
    return Placement.from_json(frame.pubs[PLACEMENT_TOPIC].sent[-1].data)


def test_rtabmap_frame_says_a_start_is_placed_by_a_recognition_of_the_loaded_map() -> None:
    """The count behind "0 recognised a node" in the report line, latched for the board: an
    update of this start that names an OLDER node than its own first one recognised the loaded
    map; one naming a node of its own does not."""
    from pepin_bringup import rtabmap_frame

    from pepin.watch import PLACEMENT_TOPIC

    frame = rtabmap_frame.RtabmapFrame()
    latched = frame.pubs[rtabmap_frame.MAP_TOPIC].qos
    assert frame.pubs[PLACEMENT_TOPIC].qos is latched, "latched, as the map is"
    assert _placement_said(frame).loaded is None, "no update yet: nothing is placed"
    _rtabmap_update(frame, 5600)
    _rtabmap_update(frame, 5601, matched=5600)  # its own node: not the loaded map
    said = _placement_said(frame)
    assert not said.placed and said.updates == 2 and said.recognised == 0 and said.loaded
    _rtabmap_update(frame, 5602, matched=5046)
    assert _placement_said(frame).placed
    sent = len(frame.pubs[PLACEMENT_TOPIC].sent)
    frame._publish()
    assert len(frame.pubs[PLACEMENT_TOPIC].sent) == sent, "published on a change only"
    frame._report()
    assert "start placed: 1 of 3 updates" in frame.logger.texts("info")[-1]


def test_a_seed_places_the_start_but_not_one_sent_before_rtabmap_runs() -> None:
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from pepin_bringup import rtabmap_frame

    frame = rtabmap_frame.RtabmapFrame()
    seed = frame.subs[rtabmap_frame.RTABMAP_INITIAL_POSE][1]
    seed(PoseWithCovarianceStamped())
    _rtabmap_update(frame, 5600)
    assert not _placement_said(frame).placed, "sent into a start that had not begun"
    seed(PoseWithCovarianceStamped())
    said = _placement_said(frame)
    assert said.placed and said.seeds == 1


def test_an_rtabmap_restart_under_a_running_frame_node_unplaces_the_start() -> None:
    """The numbering going back down is RTAB-Map restarted with this node still up: the old
    start's recognitions and seeds say nothing about the new one, which is back at its saved
    pose."""
    from pepin_bringup import rtabmap_frame

    frame = rtabmap_frame.RtabmapFrame()
    _rtabmap_update(frame, 5600)
    _rtabmap_update(frame, 5601, matched=5046)
    assert _placement_said(frame).placed
    _rtabmap_update(frame, 5600)  # the same database loaded again: numbering starts over
    said = _placement_said(frame)
    assert not said.placed and said.updates == 1
    assert frame._restarts == 1


def test_an_empty_database_is_placed_and_the_flag_off_places_everything() -> None:
    from pepin_bringup import rtabmap_frame

    frame = rtabmap_frame.RtabmapFrame()
    _rtabmap_update(frame, 1)
    assert _placement_said(frame).placed, "its start pose is its own map's origin"
    other = rtabmap_frame.RtabmapFrame()
    _rtabmap_update(other, 5600)
    assert not _placement_said(other).placed
    assert other._switches.set("start_needs_placement", False) is True
    said = _placement_said(other)
    assert said.placed and not said.required, "live: republished on the flip"


def test_goto_s_seed_topic_is_the_one_rtabmap_frame_hears() -> None:
    from pepin_bringup import rtabmap_frame

    goto = (REPO / "ros/tools/goto_ros.py").read_text()
    assert f'RTABMAP_INITIAL_POSE = "{rtabmap_frame.RTABMAP_INITIAL_POSE}"' in goto
    assert "Preflight.placement(placement_now(nav), asked=asked is not False)" in goto, (
        "the preflight asks it, under the goal server's switch"
    )


class _AskingNav(ros_stubs.Node):
    """goto's navigator as far as a parameter question goes: every client it makes is answered
    with ``response`` (``None``: nobody serves it)."""

    def __init__(self, response: Any) -> None:
        super().__init__("goto")
        self.response = response

    def create_client(self, srv_type: Any, name: str) -> Any:
        client = super().create_client(srv_type, name)
        client.ready = self.response is not None
        client.response = self.response
        return client


def test_goto_obeys_the_goal_server_s_placement_switch() -> None:
    """goto_ros reads the goal server's live flag through its parameter service: off lifts the
    refusal on this path too; a goal server that does not answer, or has no such flag (an older
    build: NOT_SET), leaves the default on. The client is destroyed whatever the answer."""
    from pepin_bringup import goal_server
    from rcl_interfaces.msg import ParameterType, ParameterValue  # the stubs'
    from test_goto_interrupt import load_goto

    goto = load_goto()
    assert goto.PLACEMENT_FLAG in goal_server.FLAGS.names, "the flag goto asks for exists there"
    service = f"{goto.GOAL_SERVER}/get_parameters"

    def held(value: bool) -> Any:
        return ros_stubs.GetParameters.Response(
            values=[ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=value)]
        )

    older = ros_stubs.GetParameters.Response(values=[ParameterValue(type=0)])
    for response, heard in ((None, None), (held(False), False), (held(True), True), (older, None)):
        nav = _AskingNav(response)
        assert goto.goal_server_flag(nav, goto.PLACEMENT_FLAG) is heard
        assert nav.destroyed_clients == [service], "no client is left behind"
        if response is not None:
            assert nav.service_clients[service].calls[0].names == [goto.PLACEMENT_FLAG]


def _loc_rows(recorder: Any) -> list[dict[str, Any]]:
    """The tape's `loc` records so far."""
    return [r for _t, r in recorder._tape._buffer if r.get("topic") == "loc"]


def test_the_tape_reads_the_pose_from_the_goal_server_s_topic() -> None:
    """A tape with no pose in it is a drive nobody can replay. Since 2026-09-22 the edge arrives
    as a topic — the goal server parses /tf for
    navigation anyway — so this node starts NO listener of its own: two rclpy TF listeners on a
    4-core A53 cost ~56 % of a core to read one pose. The row is the row it always was, ``source``
    ``tf`` because it is that same edge, and no invented confidence."""
    from pepin_bringup import run_recorder

    node = run_recorder.RunRecorderNode()
    recorder = node._recorder
    assert recorder._tf is None, "no second listener on the board"
    assert run_recorder.POSE_TOPIC in node.subs, "the pose arrives as a topic"
    node.subs[run_recorder.POSE_TOPIC][1](_pose_msg(x=0.5, y=-1.5, yaw_deg=90.0))
    records = _loc_rows(recorder)
    assert records and records[-1]["source"] == "tf"
    assert records[-1]["x"] == 0.5 and records[-1]["y"] == -1.5
    assert records[-1]["theta"] == pytest.approx(math.pi / 2, abs=1e-4)
    assert "confidence" not in records[-1], "TF carries no covariance; a made-up one is a lie"
    assert run_recorder.POSE_TOPIC in " ".join(node.logger.texts()), "the ready line says so"


def test_the_old_listener_is_one_flag_away_and_costs_nothing_while_it_is_off() -> None:
    """CLAUDE.md rule 19: ``loc_from`` tf is the path of before 2026-09-22, reachable live. The
    listener is built on the first tick that asks for it — never in the constructor — so the
    switch that is off costs this board no /tf subscription at all."""
    from pepin_bringup import run_recorder

    node = run_recorder.RunRecorderNode()
    recorder = node._recorder
    node.subs[run_recorder.POSE_TOPIC][1](_pose_msg(x=0.5, y=-1.5, yaw_deg=90.0))
    assert len(_loc_rows(recorder)) == 1
    assert node._switches.set("loc_from", "tf") == "pose_topic"
    node.subs[run_recorder.POSE_TOPIC][1](_pose_msg(x=9.0, y=9.0, yaw_deg=0.0))
    assert len(_loc_rows(recorder)) == 1, "the topic is ignored while the listener owns the rows"
    recorder._tf = _FakeTf(x=1.5, y=-0.5, yaw_deg=0.0, age_s=0.0)  # type: ignore[assignment]
    recorder._last_kept.clear()  # the 5 Hz thinning, not the flag, is what would drop this one
    recorder._loc_from_tf()
    records = _loc_rows(recorder)
    assert len(records) == 2 and records[-1]["source"] == "tf", "the same row, the other reader"
    assert records[-1]["x"] == 1.5 and records[-1]["y"] == -0.5
    logger = (REPO / "ros/tools/session_logger.py").read_text()
    assert "_loc_from_tf" in logger and '"source": "tf"' in logger


def test_the_goal_server_republishes_the_pose_it_already_reads() -> None:
    """The other half of the switch: this node owns navigation and the jump watch, so it keeps
    the board's one TF listener and puts what it reads on /pose — stamped with the TRANSFORM's
    own stamp, never with now (2026-09-19, the chair)."""
    from pepin_bringup import goal_server

    node = _goal_server()
    assert goal_server.POSE_TOPIC in node.pubs, "published on a whole (side all) stack"
    node._tf = _FakeTf(x=-0.25, y=2.77, yaw_deg=90.0, age_s=0.05)
    node._publish_pose()
    sent = node.pubs[goal_server.POSE_TOPIC].sent
    assert len(sent) == 1 and sent[0].header.frame_id == "map"
    assert sent[0].pose.position.x == -0.25 and sent[0].pose.position.y == 2.77
    assert sent[0].pose.orientation.z == pytest.approx(math.sin(math.pi / 4))
    assert sent[0].header.stamp == _stamp(-0.05), "the edge's own stamp, not this moment"
    assert node._switches.set("pose_topic", False) is True
    node._publish_pose()
    assert len(sent) == 1, "off, nothing is published and the readers fall back to loc_from tf"


def test_a_mark_is_taken_from_the_transform_and_refused_when_it_goes_stale() -> None:
    """``ros/go.sh mark`` is how every place in the room was made: the pose is map -> base_link,
    and the refusal names that edge."""
    from pepin_bringup import places

    node = places.Places()
    assert node._tf is not None
    node._tf = _FakeTf(x=0.1, y=0.2, yaw_deg=0.0, age_s=0.0)  # type: ignore[assignment]
    node._cart_from_tf()
    assert node._cart is not None
    node._poses = {7: places.Pose2D(0.0, 0.0, 0.0)}
    assert node._refusal() is None, "a fresh transform and a graph is a markable moment"
    node._cart_at = node._now() - 1e6
    refusal = node._refusal()
    assert refusal is not None and "map -> base_link" in refusal, refusal


def test_rtabmap_frame_relays_the_grid_and_speaks_no_word_to_the_board() -> None:
    """Nobody on the board fuses a word: rtabmap_frame's job is the grid relay, the memory mode
    pinned to localising, the registration and the placement word."""
    from pepin_bringup import rtabmap_frame

    frame = rtabmap_frame.RtabmapFrame()
    assert rtabmap_frame.MAP_TOPIC in frame.pubs, "the grid relay"
    assert rtabmap_frame.GRID_TOPIC in frame.subs
    assert not any(
        "localization/" in topic for topic in frame.pubs if topic != "/localization/placement"
    )
    assert frame._mode.text(), "the memory rule is pinned, not absent"
    assert "memory pinned to localising" in " ".join(frame.logger.texts())


def test_the_census_has_no_row_for_a_tracker() -> None:
    """The board's manifest names what may run there; a tracker is not one of them any more."""
    import json

    from pepin.census import manifest_from_dict

    manifest = manifest_from_dict(json.loads((REPO / "config/board_manifest.json").read_text()))
    names = {e.name for e in manifest.entries}
    assert "relocalizer" not in names and "slam_frame" not in names


class _FakeTf:
    """A TfLookup that answers one transform, for the consumers that read ``map -> base_link``."""

    def __init__(self, x: float, y: float, yaw_deg: float, age_s: float) -> None:
        import math

        self._x, self._y, self._age = x, y, age_s
        self._yaw = math.radians(yaw_deg)

    def transform(self, _parent: str, _child: str, _at: Any = None, timeout_s: float = 0.0) -> Any:
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


def test_both_zenoh_routers_start_on_the_same_patient_transport_config() -> None:
    """The one link that crosses the WiFi is router to router, and the shipped router config ends
    the session after 5 s of a blocked RELIABLE push: 8-12 closures a minute on 2026-09-22's radio,
    each costing both machines a re-discovery of everything. ``ros/zenoh/router.json5`` is a FULL
    copy of the image's default — ``ZENOH_ROUTER_CONFIG_URI`` REPLACES the configuration instead of
    merging into it, so a fragment would silently drop every ROS setting — and one file serves both
    routers because the only difference between them (who dials whom) rides in
    ``ZENOH_CONFIG_OVERRIDE``, which rmw_zenoh applies on top."""
    config = (REPO / "ros/zenoh/router.json5").read_text()
    assert 'mode: "router"' in config and '"tcp/[::]:7447"' in config, "a copy, not a fragment"
    assert "lease: 60000," in config, "...and the ROS tuning in it is untouched"
    assert "wait_before_close: 20000000," in config, "20 s: four times the worst measured stall"
    assert "wait_before_drop: 50000," in config, "50 ms: the period of the fastest thing crossing"
    assert "keep_alive: 4," in config, "four per lease, as a lossy link wants"
    assert "connect:" in config and '"<proto>/<address>"' in config, (
        "no endpoint of either machine is written into the shared file"
    )
    unit = (REPO / "board/pepin-zrouter.service").read_text()
    assert "-v /root/pepin-ros/zenoh/router.json5:/zenoh/router.json5:ro" in unit, (
        "mounted from the path ros/sync.sh rsyncs ros/ to"
    )
    assert "ZENOH_ROUTER_CONFIG_URI=/zenoh/router.json5" in unit
    assert "if [ -f /root/pepin-ros/zenoh/router.json5 ]" in unit, (
        "a board that has not been synced yet starts on the shipped default, it does not fail:"
        " docker would have made a DIRECTORY of a missing mount source"
    )
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "-e ZENOH_ROUTER_CONFIG_URI=/zenoh/router.json5" in laptop
    assert "${PEPIN_ZROUTER_CONFIG:-$HERE/zenoh/router.json5}:/zenoh/router.json5:ro" in laptop
    assert "PEPIN_ZROUTER_CONFIG" in laptop, "empty = the shipped default, without a rebuild"
    assert "zenoh" not in (REPO / "ros/sync.sh").read_text(), "nothing excludes it from a deploy"


def _pose_msg(x: float, y: float, yaw_deg: float, at_s: float = 0.0) -> Any:
    """What the goal server puts on ``/pose``: the cart in ``map``, on the edge's own stamp."""
    import math

    yaw = math.radians(yaw_deg)
    return ros_stubs.PoseStamped(
        header=ros_stubs.Header(stamp=_stamp(at_s), frame_id="map"),
        pose=ros_stubs.Pose(
            position=ros_stubs.Point(x=x, y=y, z=0.0),
            orientation=ros_stubs.Quaternion(
                x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)
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
