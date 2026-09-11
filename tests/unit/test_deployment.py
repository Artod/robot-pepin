"""The split between the board and the laptop, and the link watch that makes it safe."""

import pytest

from pepin.deployment import (
    BOARD_NAV_NODES,
    LAPTOP_NAV_NODES,
    SIDES,
    LinkWatch,
    nav_nodes,
    runs_here,
)


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
    for name in ("/rtabmap/map", "/depth_scan"):
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
    the camera nodes, its navigation container for the planner and the goal server, the board's
    for the tracker, the sensors and anything it does not know."""
    from pepin.deployment import node_host

    assert node_host("depth_stream") == ("laptop", "pepin-vslam")
    assert node_host("/depth_fusion") == ("laptop", "pepin-vslam")
    assert node_host("goal_server") == ("laptop", "pepin-laptop")
    assert node_host("planner_server") == ("laptop", "pepin-laptop")
    assert node_host("relocalizer") == ("board", "pepin-ros")
    assert node_host("neck_state") == ("board", "pepin-ros")


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
