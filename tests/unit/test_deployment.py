"""The split between the board and the laptop, and the link watch that makes it safe."""

import json
from pathlib import Path

import pytest

from pepin.deployment import (
    BOARD_NAV_NODES,
    LAPTOP_NAV_NODES,
    SIDES,
    LinkWatch,
    nav_nodes,
    runs_here,
)

REPO = Path(__file__).resolve().parents[2]


def test_all_is_exactly_the_union_of_the_two_sides_with_nothing_shared() -> None:
    assert set(nav_nodes("all")) == set(BOARD_NAV_NODES) | set(LAPTOP_NAV_NODES)
    assert not set(BOARD_NAV_NODES) & set(LAPTOP_NAV_NODES), "a node on both sides collides"


def test_the_reflexes_stay_on_the_board_and_the_planner_leaves() -> None:
    for reflex in ("controller_server", "behavior_server", "bt_navigator", "velocity_smoother"):
        assert runs_here("board", reflex) and not runs_here("laptop", reflex)
    assert runs_here("laptop", "planner_server") and not runs_here("board", "planner_server")
    assert runs_here("board", "relocalizer") and runs_here("board", "map_server")
    assert runs_here("laptop", "goal_server") and not runs_here("board", "goal_server")
    assert runs_here("board", "link_watch") and not runs_here("all", "link_watch")
    assert not runs_here("board", "goal_server"), "the heartbeat rides with the goal server"


def test_an_unknown_side_is_refused_loudly() -> None:
    with pytest.raises(ValueError):
        nav_nodes("cloud")
    assert SIDES == ("all", "board", "laptop")


def test_a_lost_link_cuts_a_running_drive_once_and_only_after_the_patience() -> None:
    watch = LinkWatch(patience_s=2.5)
    watch.beat(now=0.0)
    assert not watch.should_cut(navigating=True, now=2.0)
    assert watch.should_cut(navigating=True, now=3.0), "silent past the patience: cut"
    assert watch.should_cut(navigating=True, now=3.5), "still asking until the cancel was sent"
    watch.cut_sent()
    assert not watch.should_cut(navigating=True, now=4.0), "once per outage, once it was sent"
    watch.beat(now=5.0)
    assert not watch.should_cut(navigating=True, now=6.0)
    assert watch.should_cut(navigating=True, now=9.0), "a second outage cuts again"


def test_no_drive_no_cut_and_no_laptop_no_cut() -> None:
    watch = LinkWatch()
    assert not watch.should_cut(navigating=True, now=100.0), (
        "never heard a laptop: not a split stack"
    )
    watch.beat(now=0.0)
    assert not watch.should_cut(navigating=False, now=100.0), "standing still needs no plan"


def test_the_board_half_is_brought_up_by_the_laptop_one_step_at_a_time() -> None:
    from pepin.deployment import (
        BOARD_NAV_NODES,
        TRANSITION_ACTIVATE,
        TRANSITION_CONFIGURE,
        autostart_for,
        next_transition,
    )

    assert autostart_for("all") and autostart_for("laptop") and not autostart_for("board")
    assert BOARD_NAV_NODES[-1] == "bt_navigator", "the tree loads last: it needs the planner side"
    fresh = dict.fromkeys(BOARD_NAV_NODES, "unconfigured")
    assert next_transition(fresh) == ("controller_server", TRANSITION_CONFIGURE)
    half = {**fresh, "controller_server": "active", "behavior_server": "inactive"}
    assert next_transition(half) == (
        "behavior_server",
        TRANSITION_ACTIVATE,
    )  # a failed try continues
    almost = dict.fromkeys(BOARD_NAV_NODES, "active")
    almost["bt_navigator"] = "inactive"
    assert next_transition(almost) == ("bt_navigator", TRANSITION_ACTIVATE)
    assert next_transition(dict.fromkeys(BOARD_NAV_NODES, "active")) is None
    assert next_transition({**almost, "bt_navigator": "activating"}) is None  # in transit: wait
    assert next_transition({}) is None  # nobody answered: wait, do not guess


def test_the_tape_is_written_where_the_sensors_are() -> None:
    from pepin.deployment import runs_here

    assert runs_here("all", "run_recorder") and runs_here("board", "run_recorder")
    assert not runs_here("laptop", "run_recorder")
    assert runs_here("laptop", "goal_server") and not runs_here("board", "goal_server")


def test_the_laptop_notices_a_new_board_bridge() -> None:
    from pepin.deployment import BridgeIdentity, bridge_zid

    reply = '[{"key":"@/41206010aba91e0f57df10a0e746e960/router","value":{"sessions":[]}}]'
    assert bridge_zid(reply) == "41206010aba91e0f57df10a0e746e960"
    assert (
        bridge_zid("") is None and bridge_zid("[]") is None and bridge_zid('[{"key":"x"}]') is None
    )
    seen = BridgeIdentity(silence_s=60.0)
    assert not seen.observe("a", now=0.0)  # the first bridge seen
    assert not seen.observe("a", now=5.0) and not seen.observe(None, now=10.0)  # a hiccup
    assert seen.observe("b", now=15.0)  # a new bridge: restart
    assert not seen.observe("b", now=20.0)


def test_a_silent_bridge_restarts_the_half_once_and_a_blind_start_restarts_on_first_contact() -> (
    None
):
    """A wedged bridge stays "Up" and answers nothing: after a minute of silence the half
    restarts. A watch that never had a bridge to talk to restarts its half when one appears —
    its nodes subscribed against nothing, and a bridge that comes AFTER the containers breaks
    exactly those subscriptions (run 0148). A watch that heard the bridge first does not."""
    from pepin.deployment import BridgeIdentity

    seen = BridgeIdentity(silence_s=60.0)
    assert not seen.observe("a", now=0.0)
    assert not seen.observe(None, now=30.0) and not seen.observe(None, now=59.0)
    assert seen.observe(None, now=61.0), "a minute of silence: restart"
    assert not seen.observe(None, now=120.0), "once per silence"
    blind = BridgeIdentity(silence_s=60.0)
    assert not blind.observe(None, now=0.0) and not blind.observe(None, now=5.0)
    assert not blind.observe(None, now=30.0), "no contact yet, and no silence to time"
    assert blind.observe("a", now=35.0), "the first bridge for a half that started without one"
    assert not blind.observe("a", now=40.0)
    heard = BridgeIdentity(silence_s=60.0)
    assert not heard.observe("a", now=0.0) and not heard.observe("a", now=5.0)


def test_the_routes_settle_against_the_bridge_s_own_count_not_a_fixed_number() -> None:
    """Vision mode routes fewer topics than the fixed 20 and waited the whole patience: the
    threshold is half of what the previous bridge had at first contact."""
    from pepin.deployment import routes_settled

    assert routes_settled(count=12, expected=24, stable_s=15.0, settle_s=15.0)
    assert not routes_settled(count=11, expected=24, stable_s=15.0, settle_s=15.0)
    assert not routes_settled(count=30, expected=24, stable_s=14.9, settle_s=15.0), "still moving"
    assert routes_settled(count=1, expected=None, stable_s=15.0, settle_s=15.0), "no reference"
    assert not routes_settled(count=0, expected=None, stable_s=15.0, settle_s=15.0)
    assert not routes_settled(count=0, expected=1, stable_s=15.0, settle_s=15.0)


def test_the_bridge_s_vision_mode_routes_the_board_s_plan_out_and_nothing_back_but_the_map() -> (
    None
):
    """With the whole drive on the board, the plan and the costmaps the laptop half would have
    published come from the board; the laptop publishes only its map and depth, so nothing is a
    publisher on both sides (a loop); no service or action crosses (they aborted the container)."""
    import re

    from pepin.deployment import (
        BRIDGE_MODES,
        VISION_BOARD_PUBLISHES,
        VISION_LAPTOP_PUBLISHES,
        bridge_admin_for,
        bridge_allow,
        bridge_config,
        bridge_config_name,
    )

    board, laptop = bridge_allow("board", "vision"), bridge_allow("laptop", "vision")
    pub_b, sub_l = re.compile(board["publishers"][0]), re.compile(laptop["subscribers"][0])
    pub_l, sub_b = re.compile(laptop["publishers"][0]), re.compile(board["subscribers"][0])
    for name in ("/plan", "/global_costmap/costmap", "/local_plan", "/goal_pose", "/scan", "/tf"):
        assert pub_b.search(name) and sub_l.search(name), name
        assert not pub_l.search(name), f"{name} would loop"
    for name in ("/map", "/rtabmap/mapGraph", "/depth_scan"):
        assert pub_l.search(name) and sub_b.search(name), name
        assert not pub_b.search(name), f"{name} would loop"
    assert not set(VISION_BOARD_PUBLISHES) & set(VISION_LAPTOP_PUBLISHES)
    for kind in ("service_servers", "service_clients", "action_servers", "action_clients"):
        for side in (board, laptop):
            assert not re.compile(side[kind][0]).search("/navigate_to_pose"), kind
            assert not re.compile(side[kind][0]).search("/relocalize"), "nothing, not all"
    assert bridge_config("board", "vision")["plugins"]["ros2dds"]["allow"] == board  # type: ignore[index]
    assert bridge_allow("board") == bridge_allow("board", "split")
    assert BRIDGE_MODES == ("split", "vision")
    assert bridge_config_name("board") == "zenoh-bridge-board.json"
    assert bridge_config_name("laptop", "vision") == "zenoh-bridge-laptop-vision.json"
    with pytest.raises(ValueError):
        bridge_allow("board", "cloud")
    with pytest.raises(ValueError):
        bridge_config_name("board", "cloud")
    assert bridge_admin_for("laptop") == "http://pepin-zenoh:8000"
    assert bridge_admin_for("board") == bridge_admin_for("all") == "http://127.0.0.1:8000"


def test_the_laptop_publishes_the_one_map_in_every_mode_and_the_board_publishes_none() -> None:
    """World R: RTAB-Map's grid IS the map, so /map crosses laptop -> board in both modes and the
    board publishes no map at all — what it sends back is /map_tracked, the grid its tracker
    ACCEPTED, which both costmaps' static layers and the laptop's own words are keyed to."""
    import re

    from pepin.deployment import BRIDGE_MODES, bridge_allow

    for mode in BRIDGE_MODES:
        laptop = re.compile(bridge_allow("laptop", mode)["publishers"][0])
        board = re.compile(bridge_allow("board", mode)["publishers"][0])
        assert laptop.search("/map"), f"{mode}: the laptop publishes the one map"
        assert not board.search("/map"), f"{mode}: and the board publishes none"
        assert board.search("/map_tracked"), f"{mode}: the grid the tracker accepted comes back"
        assert not laptop.search("/map_tracked"), f"{mode}: one publisher of it, the tracker"


def test_the_container_s_names_are_the_ones_its_respawn_must_see_gone() -> None:
    from pepin.deployment import laptop_launch_nodes, nav_container_nodes

    board = nav_container_nodes("board")
    assert board[0] == "/nav2_container_board" and "/bt_navigator" in board
    assert "/local_costmap/local_costmap" in board and "/global_costmap/global_costmap" not in board
    assert "/map_server" in board and "/lifecycle_manager_navigation_board" in board
    laptop = nav_container_nodes("laptop")
    assert "/planner_server" in laptop and "/global_costmap/global_costmap" in laptop
    assert "/map_server" not in laptop and "/controller_server" not in laptop
    assert set(laptop_launch_nodes("nav")) == set(laptop) | {"/goal_server"}
    whole = nav_container_nodes("all")
    assert whole[0] == "/nav2_container" and set(board[1:]) | set(laptop[1:]) <= set(whole) | {
        "/lifecycle_manager_navigation_board",
        "/lifecycle_manager_navigation_laptop",
    }


def test_the_mounts_are_read_from_config_wherever_the_library_runs(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """config/imu.json and config/lidar.json are found from a checkout, from a directory named
    by PEPIN_CONFIG_DIR, and a missing file names every place looked instead of guessing. What
    the files then say is pepin.mounts' business (test_mounts.py); this is the search."""
    from pathlib import Path

    from pepin.deployment import config_file
    from pepin.mounts import Mounts

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.delenv("PEPIN_CONFIG_DIR", raising=False)
    assert config_file("imu.json") == repo / "config/imu.json"
    assert config_file("lidar.json") == repo / "config/lidar.json"
    with pytest.raises(FileNotFoundError, match=r"config/no_such\.json is in none of"):
        config_file("no_such.json")
    assert Mounts.load() == Mounts.load(repo / "config"), "config_file finds the checkout's own"
    elsewhere = repo / "tests"
    monkeypatch.setenv("PEPIN_CONFIG_DIR", str(elsewhere))
    assert config_file("coverage_floor.txt") == elsewhere / "coverage_floor.txt"
    assert config_file("imu.json") == repo / "config/imu.json", "the override is searched first"


def test_a_launch_waits_until_the_bridge_has_forgotten_its_ghost() -> None:
    """The bridge lists nodes as @/<zid>/ros2/node/<participant>/<name>; a launch asks which of
    its own names are still there. A ghost of the SLAM half keeps the tracker's twin out."""
    from pepin.deployment import LAPTOP_NAV_NODES, laptop_launch_nodes, lingering_nodes

    reply = (
        '[{"key":"@/6655/ros2/node/0110d87f/rtabmap/rtabmap","value":{}},'
        '{"key":"@/6655/ros2/node/0110d87f/rtabmap/transform_listener_impl_aaaa","value":{}},'
        '{"key":"@/6655/ros2/node/01105959/camera_stream","value":{}},'
        '{"key":"@/6655/ros2/route/topic/sub/scan","value":{}}]'
    )
    assert lingering_nodes(reply, laptop_launch_nodes("slam")) == {
        "/rtabmap/rtabmap",
        "/camera_stream",
    }
    assert lingering_nodes(reply, laptop_launch_nodes("nav")) == set()
    assert lingering_nodes("", laptop_launch_nodes("slam")) == set()
    assert lingering_nodes('[{"value":1}, 3]', ("/x",)) == set()
    assert set(laptop_launch_nodes("nav")) >= {f"/{n}" for n in LAPTOP_NAV_NODES}
    assert "/goal_server" in laptop_launch_nodes("nav")
    with pytest.raises(ValueError):
        laptop_launch_nodes("board")


def test_a_node_s_flags_are_reached_where_its_process_lives() -> None:
    """ros/flags.sh execs into the container a node runs in: the laptop's SLAM container for
    the camera nodes, its navigation container for the planner and the goal server when the
    stack is split, the board's for them on a whole board and for the tracker, the sensors and
    anything it does not know — pepin-laptop exists only in the split (ros/laptop.sh)."""
    from pepin.deployment import node_host

    for split in (False, True):
        assert node_host("depth_stream", split=split) == ("laptop", "pepin-vslam")
        assert node_host("/depth_fusion", split=split) == ("laptop", "pepin-vslam")
        assert node_host("relocalizer", split=split) == ("board", "pepin-ros")
        assert node_host("neck_state", split=split) == ("board", "pepin-ros")
    assert node_host("goal_server", split=True) == ("laptop", "pepin-laptop")
    assert node_host("planner_server", split=True) == ("laptop", "pepin-laptop")
    assert node_host("goal_server") == ("board", "pepin-ros"), "a whole board, the default"
    assert node_host("planner_server") == ("board", "pepin-ros")


def test_the_split_keeps_the_plan_on_the_laptop_and_vision_moves_it_to_the_board() -> None:
    """In the split the laptop plans, so /plan crosses from it and the global costmap crosses
    from nobody; in vision mode the board plans and both cross from it alone."""
    import re

    from pepin.deployment import bridge_allow

    def publishes(side: str, mode: str, name: str) -> bool:
        return re.compile(bridge_allow(side, mode)["publishers"][0]).search(name) is not None

    assert publishes("laptop", "split", "/plan") and not publishes("board", "split", "/plan")
    for side in ("board", "laptop"):
        assert not publishes(side, "split", "/global_costmap/costmap")
    for name in ("/plan", "/global_costmap/costmap"):
        assert publishes("board", "vision", name) and not publishes("laptop", "vision", name)


def test_the_necks_joint_states_reach_the_laptop_in_both_modes() -> None:
    """The neck's servos are read on the board and the laptop's SLAM wants to know where the
    camera looks, in either split: /neck/state crosses board -> laptop in both modes and is
    never a laptop publisher (a topic allowed on both sides loops until nothing crosses). The
    transform itself is not a topic of its own — it rides /tf, which already crosses."""
    import re

    from pepin.deployment import BOARD_PUBLISHES, LAPTOP_PUBLISHES, bridge_allow

    assert "neck/state" in BOARD_PUBLISHES and "neck/state" not in LAPTOP_PUBLISHES
    for mode in ("split", "vision"):
        board, laptop = bridge_allow("board", mode), bridge_allow("laptop", mode)
        assert re.compile(board["publishers"][0]).search("/neck/state"), mode
        assert re.compile(laptop["subscribers"][0]).search("/neck/state"), mode
        assert not re.compile(laptop["publishers"][0]).search("/neck/state"), mode
        assert re.compile(board["publishers"][0]).search("/tf"), mode


def test_the_retired_frame_owner_keeps_its_route_and_nothing_else_does() -> None:
    """CLAUDE.md rule 19: nav.launch.py's slam:=true puts pepin_bringup.slam_frame back in the
    tracker's seat, and it needs the laptop's correction to reach the board. So /map_odom keeps a
    route, one way, and it is judged by nothing when nobody publishes it."""
    import re

    from pepin.deployment import ON_DEMAND_TOPICS, bridge_allow

    board, laptop = bridge_allow("board", "vision"), bridge_allow("laptop", "vision")
    assert re.compile(laptop["publishers"][0]).search("/map_odom")
    assert re.compile(board["subscribers"][0]).search("/map_odom")
    assert not re.compile(board["publishers"][0]).search("/map_odom"), "it would loop"
    assert "/map_odom" in ON_DEMAND_TOPICS, "silence on it is the arrangement that ships"


def test_the_written_bridge_configs_are_the_ones_the_table_generates() -> None:
    """Four files, two modes by two sides, generated from pepin.deployment and never edited by
    hand. The two SLAM ones went with the mode on 2026-09-19."""
    from pepin.deployment import BRIDGE_MODES, bridge_config, bridge_config_name

    for mode in BRIDGE_MODES:
        for side in ("board", "laptop"):
            name = bridge_config_name(side, mode)
            written = json.loads((REPO / "ros" / name).read_text())
            assert written == bridge_config(side, mode), f"regenerate ros/{name}"
    assert not list((REPO / "ros").glob("zenoh-bridge-*-slam.json")), "the SLAM mode is gone"
    assert len(list((REPO / "ros").glob("zenoh-bridge-*.json"))) == 4


def test_the_tracker_always_runs_and_the_retired_frame_owner_takes_its_seat_or_nothing() -> None:
    """runs_here is the whole split. The tracker owns map -> odom in every situation, so it runs
    wherever the reflexes do; the retired owner (slam_frame) is the one thing that stands it
    down, because two publishers of one edge fight."""
    from pepin.deployment import runs_here

    for node in ("map_server", "relocalizer"):
        assert runs_here("board", node) and runs_here("all", node), node
        assert not runs_here("laptop", node), node
    assert runs_here("board", "relocalizer") and not runs_here("board", "relocalizer", True)
    assert runs_here("board", "map_server", True), "a served pgm is orthogonal to who owns the edge"
    assert runs_here("board", "slam_frame", True) and runs_here("all", "slam_frame", True)
    assert not runs_here("board", "slam_frame"), "off by default: the tracker owns that edge"
    assert not runs_here("laptop", "slam_frame", True), "the frame lives where the drive is"
    # Everything else is untouched by it: the reflexes and the recorder stay put.
    for node in ("controller_server", "bt_navigator", "run_recorder", "goal_server"):
        assert runs_here("board", node, True) == runs_here("board", node), node


# ---- the bridge's routes, and whether they carry anything --------------------------------------

# Two routes as the REST admin really prints them (scratch/bridge_state_182039_laptop_routes.json,
# 2026-09-13): the board's zid publishing /imu/data_raw out of its DDS, the laptop's bringing it
# in for three nodes. One admin answers for both bridges — the zenoh admin space is network-wide,
# which is the whole reason a "sub" route shows up on a side whose allow-list is pub-only.
BOARD_ZID = "abee59d7f052e5eeffe2098f0b8ef347"
LAPTOP_ZID = "ed8b4614af3e4d0f8250fb60d94969c5"
ADMIN_ROUTES = (
    '[{"key":"@/' + BOARD_ZID + '/ros2/route/topic/pub/imu/data_raw","value":'
    '{"dds_reader":"01109a4181cdf60267ddde3100000e04","local_nodes":["/base_bridge"],'
    '"ros2_name":"/imu/data_raw","ros2_type":"sensor_msgs/msg/Imu","remote_routes":["x:imu"]}},'
    '{"key":"@/' + LAPTOP_ZID + '/ros2/route/topic/sub/imu/data_raw","value":'
    '{"dds_writer":"01100073db4cacbc065f74cb00002803","is_active":true,'
    '"local_nodes":["/contact_scan","/depth_fusion","/bridge_watch"],'
    '"ros2_name":"/imu/data_raw","ros2_type":"sensor_msgs/msg/Imu"}},'
    '{"key":"@/' + BOARD_ZID + '/ros2/route/topic/pub/neck/state","value":'
    '{"dds_reader":"","local_nodes":["/neck_state"],"ros2_name":"/neck/state",'
    '"ros2_type":"sensor_msgs/msg/JointState","remote_routes":[]}},'
    '{"key":"@/' + LAPTOP_ZID + '/ros2/route/topic/sub/neck/state","value":'
    '{"dds_writer":"","is_active":false,"local_nodes":[],"ros2_name":"/neck/state",'
    '"ros2_type":"sensor_msgs/msg/JointState"}},'
    '{"key":"@/' + LAPTOP_ZID + '/ros2/route/service/srv/relocalize","value":{}}]'
)


def test_an_allow_list_regex_reads_back_as_the_names_it_was_written_from() -> None:
    """The watch asks the running bridge for its own allow-list instead of guessing the mode,
    so the regex the bridge echoes (doubled anchors and all) must parse back to the names."""
    from pepin.deployment import (
        BRIDGE_MODES,
        _names_regex,
        allowed_names,
        bridge_allow,
        incoming_topics,
    )

    assert allowed_names(_names_regex(("tf", "imu/data_raw"))) == ("/imu/data_raw", "/tf")
    assert allowed_names("^^/(scan|tf)$$") == ("/scan", "/tf"), "as the bridge's admin prints it"
    assert allowed_names("^$") == () and allowed_names("") == () and allowed_names(".*") == ()
    for mode in BRIDGE_MODES:
        for side in ("board", "laptop"):
            arriving = allowed_names(bridge_allow(side, mode)["subscribers"][0])
            assert arriving == incoming_topics(side, mode), (side, mode)
    assert "/imu/data_raw" in incoming_topics("laptop") and "/scan" in incoming_topics("laptop")
    assert "/depth_scan" in incoming_topics("board") and "/imu/data_raw" not in incoming_topics(
        "board"
    )


def test_both_bridges_are_read_from_one_admin_and_told_apart_by_their_zid() -> None:
    from pepin.deployment import bridge_routes

    routes = bridge_routes(ADMIN_ROUTES)
    assert len(routes) == 4, "topic routes only; the service route is not one"
    imu_out = next(r for r in routes if r.topic == "/imu/data_raw" and r.direction == "pub")
    assert imu_out.zid == BOARD_ZID and imu_out.local_nodes == ("/base_bridge",)
    assert imu_out.type_name == "sensor_msgs/msg/Imu" and imu_out.active, "a reader = a live route"
    neck = next(r for r in routes if r.topic == "/neck/state" and r.direction == "pub")
    assert not neck.active, "no remote wants it, so the bridge built no DDS reader for it"
    assert bridge_routes("") == () and bridge_routes("{}") == ()


def test_a_topic_flows_only_when_one_side_publishes_it_and_the_other_waits_for_it() -> None:
    """The watch's own subscription never counts as a local subscriber: it must not be the
    reason a topic looks wanted, or it would keep a route alive for nobody."""
    from pepin.deployment import bridge_routes, topic_flows

    routes = bridge_routes(ADMIN_ROUTES)
    allowed = ("/imu/data_raw", "/neck/state", "/tf")
    flows = {f.topic: f for f in topic_flows(routes, LAPTOP_ZID, allowed, watcher="bridge_watch")}
    assert set(flows) == {"/imu/data_raw", "/neck/state"}, "/tf has no route at all"
    imu = flows["/imu/data_raw"]
    assert imu.should_flow and imu.type_name == "sensor_msgs/msg/Imu"
    assert imu.subscribers_here == ("/contact_scan", "/depth_fusion"), "the watch is not one"
    assert flows["/neck/state"].published_there and not flows["/neck/state"].should_flow
    # Read from the board's side the same routes mean the opposite: it publishes, it waits for
    # nothing, so nothing is due to arrive there.
    assert not any(f.should_flow for f in topic_flows(routes, BOARD_ZID, allowed))


def test_a_latched_or_on_demand_topic_is_never_judged_for_flow() -> None:
    """/map goes out once (transient durability), /tf_static likewise, /plan only while a drive
    runs: quiet by nature, not dead. The watch that judged them restarted a healthy bridge
    twenty seconds after every start (2026-09-13) and left the real topics at a trickle."""
    from pepin.deployment import ON_DEMAND_TOPICS, TopicFlow

    waiting = ("/depth_fusion",)
    # /depth_marks is one message per observation INTEGRATED into the volume, so it stops
    # whenever the paint gates withhold (a tracker that lost its fit) while /depth_scan goes on
    # flowing beside it: a legitimate state of a healthy link, judged by nothing.
    for topic in ("/map", "/tf_static", "/plan", "/depth_marks"):
        assert topic in ON_DEMAND_TOPICS
        flow = TopicFlow(topic, "some/msg/Type", True, waiting)
        assert flow.should_flow and not flow.judged, topic
    imu = TopicFlow("/imu/data_raw", "sensor_msgs/msg/Imu", True, waiting)
    assert imu.judged, "a periodic topic that should flow is judged"
    assert not TopicFlow("/imu/data_raw", "sensor_msgs/msg/Imu", False, waiting).judged


def test_a_route_that_carries_nothing_is_starved_and_a_quiet_topic_is_not() -> None:
    """The failure the route count cannot see: the route is there, the far side publishes, and
    the counter does not move. A topic nobody publishes or nobody here reads never is."""
    from pepin.deployment import FlowWatch

    watch = FlowWatch(silence_s=20.0, cooldown_s=90.0)
    for tick in range(4):  # /scan flowing, /imu dead, /neck wanted by no one
        now = float(tick * 5)
        watch.observe("/scan", delivered=10 * tick, expected=True, now=now)
        watch.observe("/imu/data_raw", delivered=0, expected=True, now=now)
        watch.observe("/neck/state", delivered=0, expected=False, now=now)
        assert watch.starved(now) == (), f"not yet at {now} s"
    watch.observe("/scan", delivered=50, expected=True, now=20.0)
    watch.observe("/imu/data_raw", delivered=0, expected=True, now=20.0)
    watch.observe("/neck/state", delivered=0, expected=False, now=20.0)
    assert watch.starved(20.0) == ("/imu/data_raw",)
    watch.repaired(20.0)
    assert watch.starved(60.0) == (), "the cooldown: fresh routes need seconds to carry anything"
    assert not watch.settled(60.0) and watch.settled(210.0)
    watch.observe("/scan", delivered=99, expected=True, now=115.0)
    watch.observe("/imu/data_raw", delivered=0, expected=True, now=115.0)
    assert watch.starved(115.0) == ("/imu/data_raw",), "still dead after the cooldown"
    assert watch.starved(150.0) == (), "a round that reached no admin observes, and accuses, nobody"


def test_a_bridged_topic_carries_one_qos_on_both_sides() -> None:
    """The route's DDS QoS is whichever declaration created it and is never revised, so the two
    sides must not disagree: /imu/data_raw is written RELIABLE ten deep by the board."""
    from pepin.deployment import BRIDGE_MODES, BRIDGED_QOS, bridged_qos, incoming_topics

    assert bridged_qos("/imu/data_raw") == ("reliable", 10) == bridged_qos("imu/data_raw")
    assert bridged_qos("/scan") is None
    assert bridged_qos("/localization/graph_measurement") == ("reliable", 5), "vision mode's"
    # A rule is about a topic that crosses in SOME mode: the graph's word only exists beside a
    # known map, where the tracker it is measured for runs.
    crossing = {
        t
        for mode in BRIDGE_MODES
        for side in ("laptop", "board")
        for t in incoming_topics(side, mode)
    }
    for topic in BRIDGED_QOS:
        assert topic in crossing, topic
        assert BRIDGED_QOS[topic][0] in ("reliable", "best_effort"), topic


# Four routes of a live dump, taken from the laptop through pepin-vslam on 2026-09-14 (62 routes
# in the reply, 0 of them dead): the laptop's bridge publishing /vo and /depth_scan out, the
# board's bringing both in. Trimmed to the keys the reading uses, otherwise verbatim.
LIVE_LAPTOP_ZID = "8f5e481e794a4ff3eb664f7dbead5870"
LIVE_BOARD_ZID = "1dcf0420b2c8f50a264578889510f6f3"
LIVE_ROUTES = (
    '[{"key":"@/' + LIVE_LAPTOP_ZID + '/ros2/route/topic/pub/depth_scan","value":'
    '{"dds_reader":"01108106ded89678d760c22100002b04","local_nodes":["/depth_stream"],'
    '"remote_routes":["' + LIVE_BOARD_ZID + ':depth_scan"],"ros2_name":"/depth_scan",'
    '"ros2_type":"sensor_msgs/msg/LaserScan"}},'
    '{"key":"@/' + LIVE_LAPTOP_ZID + '/ros2/route/topic/pub/vo","value":'
    '{"dds_reader":"01108106ded89678d760c22100001d04","local_nodes":["/visual_odometry"],'
    '"remote_routes":["' + LIVE_BOARD_ZID + ':vo"],"ros2_name":"/vo",'
    '"ros2_type":"nav_msgs/msg/Odometry"}},'
    '{"key":"@/' + LIVE_BOARD_ZID + '/ros2/route/topic/sub/depth_scan","value":'
    '{"dds_writer":"0110b40e222108dd30e3933e00001e03","is_active":true,'
    '"local_nodes":["/local_costmap/local_costmap","/global_costmap/global_costmap"],'
    '"remote_routes":["' + LIVE_LAPTOP_ZID + ':depth_scan"],"ros2_name":"/depth_scan",'
    '"ros2_type":"sensor_msgs/msg/LaserScan"}},'
    '{"key":"@/' + LIVE_BOARD_ZID + '/ros2/route/topic/sub/vo","value":'
    '{"dds_writer":"0110b40e222108dd30e3933e00001803","is_active":true,'
    '"local_nodes":["/ekf_filter_node"],"remote_routes":["' + LIVE_LAPTOP_ZID + ':vo"],'
    '"ros2_name":"/vo","ros2_type":"nav_msgs/msg/Odometry"}}]'
)


def test_a_healthy_live_dump_has_no_dead_route() -> None:
    """The reading is checked against the real thing first: a bridge that works has an endpoint
    on every route both sides hold."""
    from pepin.deployment import bridge_routes, dead_routes

    routes = bridge_routes(LIVE_ROUTES)
    assert len(routes) == 4
    assert dead_routes(routes, LIVE_LAPTOP_ZID) == ()
    assert dead_routes(routes, LIVE_BOARD_ZID) == ()


def test_a_pub_route_with_a_publisher_a_remote_and_no_reader_is_dead() -> None:
    """The failure of 2026-09-14: /vo had the publisher and the board's matching route and no
    dds_reader, so the EKF got nothing while every count in the admin looked right."""
    from pepin.deployment import bridge_routes, dead_routes

    routes = bridge_routes(LIVE_ROUTES.replace("01108106ded89678d760c22100001d04", ""))
    assert dead_routes(routes, LIVE_LAPTOP_ZID) == ("/vo",), "and /depth_scan beside it is fine"
    assert dead_routes(routes, LIVE_BOARD_ZID) == (), "the board's own routes are unharmed"


def test_a_sub_route_without_its_writer_is_dead_and_a_route_nobody_wants_is_not() -> None:
    """The mirror on the incoming side, and the two innocents: a route with no local node to
    serve and a route no bridge on the far side holds are both endpoint-less on purpose."""
    from pepin.deployment import bridge_routes, dead_routes

    routes = bridge_routes(LIVE_ROUTES.replace("0110b40e222108dd30e3933e00001803", ""))
    assert dead_routes(routes, LIVE_BOARD_ZID) == ("/vo",)
    quiet = bridge_routes(ADMIN_ROUTES)  # /neck/state: no reader, but no remote route either
    assert dead_routes(quiet, BOARD_ZID) == () and dead_routes(quiet, LAPTOP_ZID) == ()


# The board's side of the same reply on 2026-09-15: its pub routes for /scan and /tf, ours as
# the remote route on each, and an empty dds_reader on the first — thirteen of these carried
# nothing all evening while the watch judged the laptop's routes alone and said "dead routes 0".
BOARD_PUB_ROUTES = (
    '[{"key":"@/' + LIVE_BOARD_ZID + '/ros2/route/topic/pub/scan","value":'
    '{"dds_reader":"","local_nodes":["/ldlidar_node"],'
    '"remote_routes":["' + LIVE_LAPTOP_ZID + ':scan"],"ros2_name":"/scan",'
    '"ros2_type":"sensor_msgs/msg/LaserScan"}},'
    '{"key":"@/' + LIVE_BOARD_ZID + '/ros2/route/topic/pub/tf","value":'
    '{"dds_reader":"0110b40e222108dd30e3933e00002203","local_nodes":["/base_bridge"],'
    '"remote_routes":["' + LIVE_LAPTOP_ZID + ':tf"],"ros2_name":"/tf",'
    '"ros2_type":"tf2_msgs/msg/TFMessage"}},'
    '{"key":"@/' + LIVE_BOARD_ZID + '/ros2/route/topic/pub/neck/state","value":'
    '{"dds_reader":"","local_nodes":["/neck_state"],"remote_routes":[],'
    '"ros2_name":"/neck/state","ros2_type":"sensor_msgs/msg/JointState"}}]'
)


def test_the_board_s_own_pub_route_without_a_reader_is_seen_from_the_laptop() -> None:
    """The fault of 2026-09-15, and the only side it can be read from: the admin space is
    network-wide, so the board's routes arrive in the reply the laptop's watch already fetches.
    Judged only when the board's route names OUR bridge — a route nobody on this side asked for
    has no reader on purpose (/neck/state here)."""
    from pepin.deployment import bridge_routes, dead_routes, far_dead_routes

    routes = bridge_routes(BOARD_PUB_ROUTES)
    assert far_dead_routes(routes, LIVE_LAPTOP_ZID) == ("/scan",)
    assert dead_routes(routes, LIVE_LAPTOP_ZID) == (), "none of these routes is ours"
    assert far_dead_routes(routes, "0000000000000000000000000000ffff") == (), (
        "another bridge's business: only the routes that name this one are judged here"
    )
    assert far_dead_routes(bridge_routes(LIVE_ROUTES), LIVE_LAPTOP_ZID) == (), (
        "a healthy dump: the board's sub routes are not judged here, and nothing else is dead"
    )


def test_the_bridge_config_drops_reliable_blocking_and_caps_what_crosses_the_wifi() -> None:
    """The two settings that killed the link on 2026-09-15 and the load that led to it: a
    RELIABLE route blocked on a full queue made the board's bridge close the transport itself,
    and 51.6 Hz of /tf plus 46.6 of /imu is more than the radio carries under load."""
    import re

    from pepin.deployment import BRIDGE_MODES, bridge_config, pub_max_frequencies

    for mode in BRIDGE_MODES:
        for side in ("board", "laptop"):
            plugin = bridge_config(side, mode)["plugins"]["ros2dds"]  # type: ignore[index]
            assert plugin["reliable_routes_blocking"] is False, (side, mode)
        board = bridge_config("board", mode)["plugins"]["ros2dds"]["pub_max_frequencies"]  # type: ignore[index]
        laptop = bridge_config("laptop", mode)["plugins"]["ros2dds"]["pub_max_frequencies"]  # type: ignore[index]
        assert sorted(board) == [
            "^/imu/data_raw$=20",
            "^/odom$=20",
            "^/odometry/filtered$=20",
            "^/tf$=20",
        ], mode
        assert laptop == [], f"{mode}: the plugin downsamples where the publisher is, and it is"
        " the board that publishes all four"
    # The plugin does not anchor these regexes and matches with is_match (1.7.0 config.rs), so
    # the anchors are ours: /tf_static is latched and must never be downsampled.
    caps = [entry.split("=")[0] for entry in pub_max_frequencies(("tf", "odom"))]
    assert [c for c in caps if re.compile(c).search("/tf_static")] == []
    assert [c for c in caps if re.compile(c).search("/tf")] == ["^/tf$"]
    assert pub_max_frequencies(("/tf",)) == pub_max_frequencies(("tf",)) == ["^/tf$=20"]
    assert pub_max_frequencies(("depth_scan",)) == [], "a topic with no cap gets no entry"


def _unit_command(unit: str, prefix: str, marker: str) -> str:
    """The shell text of the one ``prefix`` line of a board unit that contains ``marker``, with
    systemd's own escapes undone (``$$`` and ``%%``) — what /bin/sh is actually handed."""
    text = (REPO / "board" / unit).read_text()
    line = next(ln for ln in text.splitlines() if ln.startswith(prefix) and marker in ln)
    return line.split("-c '", 1)[1].rsplit("'", 1)[0].replace("$$", "$").replace("%%", "%")


def test_both_routers_log_the_transport_lifecycle_and_nothing_per_message() -> None:
    """2026-09-23: after a Mac wake the router-to-router session stayed dead for 28 minutes and
    neither router wrote a line, because zenoh logs re-dials, refused handshakes and expired links
    only at debug. Both routers get the same RUST_LOG, and it names no per-message module."""
    import re

    lib = (REPO / "ros/lib.sh").read_text()
    shell = re.search(r'PEPIN_ZROUTER_LOG="\$\{PEPIN_ZROUTER_LOG:-([^}]+)\}"', lib)
    assert shell, "ros/lib.sh defines the laptop router's filter"
    unit = (REPO / "board/pepin-zrouter.service").read_text()
    board = re.search(r"^Environment=PEPIN_ZROUTER_LOG=(\S+)$", unit, re.M)
    assert board and board.group(1) == shell.group(1), "one filter for both routers"
    directives = shell.group(1).split(",")
    assert directives[0] == "info", "everything else at the level rmw_zenohd always had"
    assert "zenoh::net::runtime::orchestrator=debug" in directives, "the laptop's re-dials"
    assert "zenoh_transport::unicast::establishment=debug" in directives, "refused handshakes"
    for noisy in ("universal::rx", "universal::tx", "routing", "pipeline", "=trace"):
        assert not any(noisy in d for d in directives), f"{noisy}: a line per message"
    runs = [ln for ln in unit.splitlines() if ln.startswith("ExecStart=")]
    assert runs[0].count("-e RUST_LOG=$PEPIN_ZROUTER_LOG") == 2, "both docker run bodies"
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert '-e "RUST_LOG=$PEPIN_ZROUTER_LOG"' in laptop.split("zrouter_up() {")[1].split("\n}")[0]


# Transports each router served on 2026-09-24 04:00Z (established TCP sessions on 7447): the
# board's 10 nodes on the loopback and the laptop's router; the laptop router's 14 sessions of
# pepin-vslam alone and its link to the board — pepin-laptop's nodes, the goal and flag tools and
# `docker exec` probes come on top of that one, about ten more.
BOARD_ROUTER_TRANSPORTS = 11
LAPTOP_ROUTER_TRANSPORTS = 15
LAPTOP_ROUTER_UNCOUNTED = 10


def test_both_routers_have_more_rx_workers_than_the_sessions_they_serve() -> None:
    """2026-09-23: with zenoh's two RX workers blocked in 20-s pushes to the sleeping laptop, the
    board router's close of that link never ran and its own nodes' sessions timed out on it. Each
    session pushing toward a frozen peer holds one worker and the close needs one more (4
    publishers wedged 4 workers and not 5, scratch/link_autopsy/wedge_threshold.py), so each
    router gets more workers than its census; 2 is the old behaviour, one variable away."""
    import re

    lib = (REPO / "ros/lib.sh").read_text()
    shell = re.search(r'PEPIN_ZROUTER_RX_WORKERS="\$\{PEPIN_ZROUTER_RX_WORKERS:-(\d+)\}"', lib)
    unit = (REPO / "board/pepin-zrouter.service").read_text()
    board = re.search(r"^Environment=PEPIN_ZROUTER_RX_WORKERS=(\d+)$", unit, re.M)
    assert shell and board
    assert int(board.group(1)) > BOARD_ROUTER_TRANSPORTS + 1, "the board's census, with room"
    laptop_peak = LAPTOP_ROUTER_TRANSPORTS + LAPTOP_ROUTER_UNCOUNTED
    assert int(shell.group(1)) > laptop_peak + 1, "the laptop's census and what it missed"
    runtime = '-e "ZENOH_RUNTIME=(rx: (worker_threads: $PEPIN_ZROUTER_RX_WORKERS))"'
    runs = [ln for ln in unit.splitlines() if ln.startswith("ExecStart=")]
    assert runs[0].count(runtime) == 2, "both docker run bodies"
    assert runtime in (REPO / "ros/laptop.sh").read_text().split("zrouter_up() {")[1]


def test_the_board_router_keeps_its_log_across_a_restart_like_the_stack_does() -> None:
    """Under `docker run --rm` the board router's log of the 2026-09-23 wake (the 20-s closures
    to the sleeping laptop) went with its first restart. The container now outlives its process,
    and the next start copies the log before it removes the name."""
    unit = (REPO / "board/pepin-zrouter.service").read_text()
    runs = [ln for ln in unit.splitlines() if ln.startswith("ExecStart=")]
    assert runs and all("docker run --rm" not in ln for ln in runs), "the log must outlive it"
    pre = [ln for ln in unit.splitlines() if ln.startswith("ExecStartPre=")]
    archive = next(i for i, ln in enumerate(pre) if "docker logs -t pepin-zrouter" in ln)
    remove = next(i for i, ln in enumerate(pre) if "docker rm -f pepin-zrouter" in ln)
    assert archive < remove, "copied before the name is removed"
    assert pre[archive].startswith("ExecStartPre=-"), "a failed copy never blocks the router"
    stack = (REPO / "board/pepin-ros.service").read_text()
    assert "docker logs pepin-ros > /root/pepin-ros/logs/" in stack, "the same directory"


@pytest.mark.slow
def test_the_board_router_archive_line_does_what_it_says(tmp_path: Path) -> None:
    """The unit's archive command run by /bin/sh against a fake docker: a log file for a
    container that exists, nothing for one that does not, nothing when switched off."""
    import os
    import subprocess

    fake = tmp_path / "docker"
    fake.write_text(
        "#!/bin/sh\n"
        'case "$1 $2" in\n'
        '  "container inspect") [ -f "$FAKE_EXISTS" ] ;;\n'
        '  "logs -t") echo "2026-09-23T21:26:52Z Unable to push"; echo "err line" >&2 ;;\n'
        "  *) exit 3 ;;\n"
        "esac\n"
    )
    fake.chmod(0o755)
    logs = tmp_path / "logs"
    cmd = (
        _unit_command("pepin-zrouter.service", "ExecStartPre=", "docker logs")
        .replace("/usr/bin/docker", str(fake))
        .replace("/root/pepin-ros/logs", str(logs))
    )
    exists = tmp_path / "exists"

    def run(archive: str | None) -> None:
        env = {"PATH": os.environ["PATH"], "FAKE_EXISTS": str(exists)}
        if archive is not None:
            env["PEPIN_LOG_ARCHIVE"] = archive
        subprocess.run(["/bin/sh", "-c", cmd], env=env, check=True, timeout=10)

    run(None)
    assert not logs.exists(), "no container, no file (a first boot)"
    exists.touch()
    run("off")
    assert not logs.exists(), "switched off: nothing kept, as before"
    run(None)
    (saved,) = logs.glob("zrouter_*.log")
    assert saved.read_text() == "2026-09-23T21:26:52Z Unable to push\nerr line\n"


@pytest.mark.slow
def test_every_laptop_container_removal_keeps_the_log_first(tmp_path: Path) -> None:
    """ros/lib.sh's pepin_remove_container is the one way a container leaves this laptop
    (test_scripts_parse holds that); it must stop, copy the log, and only then remove."""
    import os
    import subprocess

    calls = tmp_path / "calls"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "docker"
    fake.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{calls}"\n'
        'case "$1 $2" in\n'
        '  "container inspect") [ "$3" = alive ] ;;\n'
        '  "logs -t") echo "vslam line for $3" ;;\n'
        "esac\n"
    )
    fake.chmod(0o755)
    out = tmp_path / "archive"
    script = "source ros/lib.sh && pepin_remove_container alive gone"
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "PEPIN_LOG_DIR": str(out),
        "PEPIN_STOP_TIMEOUT_S": "1",
    }
    result = subprocess.run(
        ["bash", "-c", script], cwd=REPO, env=env, capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    (saved,) = out.glob("*_alive.log")
    assert saved.read_text() == "vslam line for alive\n"
    assert not list(out.glob("*_gone.log")), "a container that is not there leaves no file"
    order = calls.read_text().splitlines()
    stop = next(i for i, c in enumerate(order) if c.startswith("stop"))
    copy = next(i for i, c in enumerate(order) if c.startswith("logs -t alive"))
    remove = next(i for i, c in enumerate(order) if c.startswith("rm -f"))
    assert stop < copy < remove, order
    env["PEPIN_LOG_ARCHIVE"] = "off"
    env["PEPIN_LOG_DIR"] = str(tmp_path / "archive_off")
    subprocess.run(["bash", "-c", script], cwd=REPO, env=env, check=True, timeout=20)
    assert not (tmp_path / "archive_off").exists(), "PEPIN_LOG_ARCHIVE=off keeps nothing"
