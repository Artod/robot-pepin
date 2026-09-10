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
    seen = BridgeIdentity()
    assert not seen.observe(None)  # unreachable: not a change
    assert not seen.observe("a")  # the first bridge seen
    assert not seen.observe("a") and not seen.observe(None)
    assert seen.observe("b")  # a new bridge: restart
    assert not seen.observe("b")


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
