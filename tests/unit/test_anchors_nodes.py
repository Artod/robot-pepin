"""The per-node table (pepin.graphnodes): frame-free by construction, and honest about what it
cannot place.

The failure this module exists to stop, in one line: one global tie was fitted to one piece of a
database whose pieces sit 1.6 m and 129 degrees apart, so a TRUE recognition of a node in another
piece came out 90 degrees wrong at home.
"""

import json
import math
from pathlib import Path

import pytest

from pepin.anchors import room_from_identity, room_of
from pepin.graphnodes import (
    NODES_SUFFIX,
    NodePose,
    NodeTable,
    append_node,
    load_nodes,
    node_covariance,
    nodes_path,
    read_sessions,
    write_nodes,
)
from pepin.measurements import compose, inverse
from pepin.odometry import Pose2D

ROOM = "flat3_straight"


def entry(
    node_id: int, x: float, y: float, yaw: float = 0.0, session: int = 0, **kw: object
) -> NodePose:
    """One table entry: where OUR map says the cart was when that database node was created."""
    return NodePose(node_id=node_id, session=session, stamp=7.0, cart=Pose2D(x, y, yaw), **kw)  # type: ignore[arg-type]


def test_the_table_is_named_by_the_room_and_survives_the_round_trip(tmp_path: Path) -> None:
    """The files are named by the ROOM and not by the grid's size@origin, because the id of a
    growing volume changes and the room does not."""
    assert nodes_path("/maps", ROOM).name == f"{ROOM}{NODES_SUFFIX}"
    append_node(tmp_path, ROOM, entry(41, 1.0, 2.0, 0.5))
    back = load_nodes(tmp_path, ROOM)
    assert list(back) == [41]
    assert (back[41].cart.x, back[41].cart.y) == (1.0, 2.0)
    assert back[41].session == 0 and back[41].stamp == 7.0
    assert json.loads(nodes_path(tmp_path, ROOM).read_text())["node"] == 41


def test_an_entry_is_replaced_only_by_a_sharper_measurement(tmp_path: Path) -> None:
    """A node is measured again on a later drive, and the better seating wins — never the newer one,
    and never a disagreement: whatever is in the table is what every later word carries."""
    sharp = entry(41, 1.0, 2.0, sigma=(0.005, 0.005, math.radians(0.2)))
    soft = entry(41, 9.0, 9.0, sigma=(0.03, 0.03, math.radians(1.0)))
    append_node(tmp_path, ROOM, soft)
    append_node(tmp_path, ROOM, sharp)
    append_node(tmp_path, ROOM, soft)
    assert load_nodes(tmp_path, ROOM)[41].cart.x == pytest.approx(1.0), "the sharp one stands"

    table = NodeTable({41: sharp})
    assert table.offer(soft) is False and table.entries[41].cart.x == pytest.approx(1.0)
    assert table.offer(sharp) is True, "the same sharpness may always be re-taken"


def test_a_half_written_line_costs_one_entry_and_not_the_table(tmp_path: Path) -> None:
    """A live node appends to this file and a power cut leaves half a line."""
    append_node(tmp_path, ROOM, entry(41, 1.0, 2.0))
    with nodes_path(tmp_path, ROOM).open("a") as log:
        log.write('{"node": 44, "session"\n')
    append_node(tmp_path, ROOM, entry(45, 2.0, 2.0))
    assert sorted(load_nodes(tmp_path, ROOM)) == [41, 45]
    assert load_nodes(tmp_path, "no-such-room") == {}


def test_the_word_is_the_relative_pose_and_the_global_frame_cancels() -> None:
    """The point of the whole design. The optimiser may place its pieces anywhere and move them at
    every re-optimisation; a word built from the DIFFERENCE of two of its node poses does not care.
    Here the same physical situation is presented in two wildly different graph frames."""
    table = NodeTable({41: entry(41, 1.0, 2.0), 44: entry(44, 2.0, 2.0)})
    near = table.hang(41, Pose2D(10.6, 10.0), {41: Pose2D(10.0, 10.0), 44: Pose2D(11.0, 10.0)})
    far = table.hang(
        41,
        Pose2D(-60.0, -60.6, -math.pi / 2),
        {41: Pose2D(-60.0, -60.0, -math.pi / 2), 44: Pose2D(-60.0, -61.0, -math.pi / 2)},
    )
    assert not isinstance(near, str) and not isinstance(far, str)
    assert near.pose.x == pytest.approx(1.6) and near.pose.y == pytest.approx(2.0)
    assert far.pose.x == pytest.approx(1.6) and far.pose.y == pytest.approx(2.0)
    assert near.lever_m == pytest.approx(0.6) and not near.substituted
    assert "node 41 (session 0) 0.60 m away" in near.text()


def test_an_untabled_node_is_carried_by_its_own_session_and_charged_for_it() -> None:
    """A database has far more nodes than our map has measured, and a session is the piece the
    measurement found rigid. So a recognition of an untabled node is carried by the nearest tabled
    node of that session — at the cost of the piece's OWN measured rigidity, read off the fit's
    residuals and never a constant."""
    # three tabled nodes of session 0, whose graph poses are NOT exactly rigid against our map
    graph = {41: Pose2D(10.0, 10.0), 44: Pose2D(11.0, 10.0), 47: Pose2D(10.0, 11.0)}
    table = NodeTable(
        {
            41: entry(41, 1.0, 2.0),
            44: entry(44, 2.0, 2.0),
            47: entry(47, 1.0, 3.2),  # 20 cm out of a rigid fit: that is the piece's rigidity
        },
        sessions={41: 0, 44: 0, 47: 0, 77: 0},
    )
    graph[77] = Pose2D(11.05, 10.0)
    hung = table.hang(77, Pose2D(11.1, 10.0), graph)
    assert not isinstance(hung, str)
    assert hung.substituted and hung.entry.node_id == 44 and hung.matched == 77
    assert hung.rigidity_m is not None and hung.rigidity_m > 0.05, "measured, not assumed"
    assert f"{hung.rigidity_m * 100:.0f} cm" in hung.text()
    plain = table.hang(44, Pose2D(11.05, 10.0), graph)
    assert not isinstance(plain, str) and plain.rigidity_m is None, "no substitution, no charge"


def test_a_session_our_map_never_drove_gets_no_word() -> None:
    """43 of the 50 sessions of the real database have no entry at all. A localisation against one
    of them is a word this map cannot place, and the honest answer names the session and publishes
    nothing — hanging it on a piece 129 degrees away is how the pose flew."""
    table = NodeTable({41: entry(41, 1.0, 2.0)}, sessions={41: 0, 900: 31})
    refused = table.hang(900, Pose2D(3.0, 4.0), {41: Pose2D(10.0, 10.0), 900: Pose2D(3.0, 4.0)})
    assert isinstance(refused, str) and "session 31" in refused and "node 900" in refused

    blind = table.hang(41, Pose2D(10.6, 10.0), {})
    assert isinstance(blind, str), "no live pose for any tabled node is no word either"


def test_with_the_database_unreadable_one_run_is_one_piece() -> None:
    """RTAB-Map holds a hot journal on its own file, so the session mapping may simply not be
    readable. The fallback is the honest one — every node of one run is placed by one odometry, so
    they are one piece — and a word may then hang on any tabled node, charged with the rigidity that
    whole set actually shows."""
    table = NodeTable({41: entry(41, 1.0, 2.0), 44: entry(44, 2.0, 2.0)})
    assert table.sessions_known is False
    assert table.session_of(41) == 0, "a tabled node knows its own session from the day it was made"
    assert table.session_of(999) == -1, "and an unknown one is this run's"
    hung = table.hang(999, Pose2D(11.1, 10.0), {41: Pose2D(10.0, 10.0), 44: Pose2D(11.0, 10.0)})
    assert not isinstance(hung, str) and hung.entry.node_id == 44 and hung.substituted


def test_a_missing_database_file_is_no_sessions_and_no_exception(tmp_path: Path) -> None:
    """A read that raises must never take the node down over a mapping it can live without."""
    assert read_sessions(tmp_path / "nothing.db") == {}


def test_the_word_s_covariance_grows_with_the_arm_and_carries_the_registration() -> None:
    """Three terms and each one measured: the entry's own seating carried through the composition
    (a heading error costs the word the arm ``|rel|`` buys), RTAB-Map's own covariance of the
    registration rotated into the map, and the piece's rigidity when a substitute was used."""
    held = entry(41, 1.0, 2.0, sigma=(0.01, 0.01, math.radians(1.0)))
    near = node_covariance(held, Pose2D(0.05, 0.0))
    far = node_covariance(held, Pose2D(4.0, 0.0))
    assert far[1][1] > near[1][1], "four metres of arm on a one-degree heading"
    assert far[0][0] == pytest.approx(near[0][0]), "and nothing along the arm itself"

    import numpy as np

    with_registration = node_covariance(held, Pose2D(0.05, 0.0), np.diag([0.36**2, 0.36**2, 0.01]))
    assert with_registration[0][0] > near[0][0]
    with_piece = node_covariance(held, Pose2D(0.05, 0.0), None, 0.20)
    assert with_piece[0][0] == pytest.approx(near[0][0] + 0.04)


def test_the_room_is_read_from_the_provenance_and_from_the_volume_s_own_file() -> None:
    """The identity depth_fusion publishes names the room; the size@origin id names a grid. Measured
    on the parked cart on 2026-09-18: the served id moved from 239x215@-18.53,-4.38 to
    280x250@-19.48,-5.48 for the same room at the same pose, and every file named by the id went
    invisible."""
    assert room_of("seed:flat3_straight") == "flat3_straight"
    assert room_of("resume:flat3_straight") == "flat3_straight"
    assert room_of("fresh") == "", "a room born unknown has no name to be filed under"

    assert room_from_identity('{"from": "seed:flat3_straight"}') == "flat3_straight"
    assert (
        room_from_identity('{"from": "fresh", "world_path": "/maps/flat3_straight.world.npz"}')
        == "flat3_straight"
    ), "a volume born fresh still keeps its files under the room's name"
    assert room_from_identity('{"from": "fresh"}') == ""
    assert room_from_identity("not json") == "", (
        "a name guessed from a broken message is a wrong flat"
    )


def test_a_whole_table_can_be_written_at_once_by_an_offline_build(tmp_path: Path) -> None:
    """scratch/graph_node_table.py builds the table from the tapes and the database; the live node
    then reads exactly that file."""
    write_nodes(tmp_path, ROOM, [entry(44, 2.0, 2.0), entry(41, 1.0, 2.0)])
    lines = nodes_path(tmp_path, ROOM).read_text().splitlines()
    assert [json.loads(line)["node"] for line in lines] == [41, 44], "oldest node first"
    assert sorted(load_nodes(tmp_path, ROOM)) == [41, 44]


def test_the_table_places_a_word_exactly_where_the_composition_says() -> None:
    """One arithmetic check against the definition, so the geometry cannot drift: the word is
    ``Table[N] . inverse(P_N) . P_current``, spelled out."""
    held = entry(41, -3.0, 1.5, 0.7)
    current, node_pose = Pose2D(4.0, -2.0, 2.1), Pose2D(1.0, 1.0, 0.3)
    hung = NodeTable({41: held}).hang(41, current, {41: node_pose})
    assert not isinstance(hung, str)
    expected = compose(held.cart, compose(inverse(node_pose), current))
    assert hung.pose.x == pytest.approx(expected.x)
    assert hung.pose.y == pytest.approx(expected.y)
    assert hung.pose.theta == pytest.approx(expected.theta)


def test_the_database_learns_only_from_a_pose_worth_learning_from() -> None:
    """The rule, and it names no sensor. A lidar-held seating at 1-2 cm teaches; a mono camera-only
    pose held by graph words at ~20 cm fails the seating test BY ITSELF; a stereo matcher good to a
    few cm will teach the day it exists, with nothing here changed."""
    from pepin.graphnodes import LOCALISING, MAPPING, ModeRule

    rule = ModeRule(hold_s=2.0, graph="graph")
    assert rule.verdict(None, "lidar").mapping is True
    assert rule.verdict(None, "depth").mapping is True, "the rule reads sharpness, not a name"
    soft = rule.verdict("the lidar's seating is soft (20.0/20.0 cm, over 3.0 cm)", "depth")
    assert soft.mapping is False and "not worth learning from" in soft.why
    pupil = rule.verdict(None, "graph")
    assert pupil.mapping is False and "pupil is not the teacher" in pupil.why
    assert rule.verdict(None, None).mapping is False, "nobody holding the pose teaches nothing"
    assert rule.verdict(None, "lidar", "1.2/0.4 cm, 0.30 deg").text().startswith(f"{MAPPING}: ")
    assert rule.verdict(None, "graph").text().startswith(f"{LOCALISING}: ")


def test_the_mode_is_asked_for_once_and_then_only_when_it_has_held() -> None:
    """A verdict is acted on only after it has survived as long as the evidence it rests on takes to
    refresh — the seating's own freshness window — so one missed /tracker_pose cannot flap the mode.
    The very first verdict is the initial mode and is asked for at once."""
    from pepin.graphnodes import ModeRule

    rule = ModeRule(hold_s=2.0)
    first = rule.update(0.0, None, "lidar")
    assert first is not None and first.mapping is True and rule.switches == 1
    assert rule.update(0.5, None, "lidar") is None, "no change, nothing to say"

    assert rule.update(1.0, None, "graph") is None, "the clock starts here"
    assert rule.update(2.5, None, "graph") is None, "1.5 s is not 2 s"
    assert rule.update(1.0, None, "lidar") is None, "...and a blip puts it back with no call"
    assert rule.update(3.0, None, "graph") is None
    switched = rule.update(5.5, None, "graph")
    assert switched is not None and switched.mapping is False and rule.switches == 2
    assert rule.mode == "localising" and "2 switches" in rule.text()


def test_the_mode_can_be_pinned_either_way() -> None:
    """CLAUDE.md rule 19: the old behaviour stays reachable. "map" is the arrangement of before
    2026-09-18; "localise" freezes a database for a session."""
    from pepin.graphnodes import ALWAYS_LOCALISE, ALWAYS_MAP, ModeRule

    pinned = ModeRule(hold_s=2.0, override=ALWAYS_MAP)
    verdict = pinned.update(0.0, "the lidar is not driving the tracker (fit 0.00)", "graph")
    assert verdict is not None and verdict.mapping is True
    assert pinned.update(100.0, "still soft", "graph") is None, "pinned means pinned"

    frozen = ModeRule(hold_s=2.0, override=ALWAYS_LOCALISE)
    told = frozen.update(0.0, None, "lidar")
    assert told is not None and told.mapping is False
