"""Named places: the coordinates beside a frozen map, and the places that live in a graph that
bends — the cart's pose relative to a labelled node, which rides the node when a loop closes."""

import json
import math
from pathlib import Path

import pytest

from pepin.odometry import Pose2D
from pepin.places import (
    MARK_TOPIC,
    MARKED_TOPIC,
    PLACES_TOPIC,
    GraphPlace,
    Place,
    graph_place_pose,
    graph_places_path,
    load_graph_places,
    load_places,
    places_from_json,
    places_json,
    resolve_goal,
    save_graph_places,
    save_places,
)


def test_places_round_trip_next_to_the_map(tmp_path: Path) -> None:
    map_path = tmp_path / "flat.npz"
    map_path.write_bytes(b"")
    saved = save_places(
        map_path, {"kitchen": Place("kitchen", -1.5, 2.0, 90.0), "dock": Place("dock", 0.0, 0.0)}
    )
    assert saved == tmp_path / "flat.places.json"
    loaded = load_places(map_path)
    assert loaded["kitchen"].xy == (-1.5, 2.0)
    assert loaded["kitchen"].theta == pytest.approx(math.pi / 2)
    assert loaded["dock"].theta is None


def test_goal_resolves_by_name_or_numbers_and_names_the_known_places(tmp_path: Path) -> None:
    map_path = tmp_path / "flat.npz"
    map_path.write_bytes(b"")
    save_places(map_path, {"kitchen": Place("kitchen", -1.5, 2.0)})
    assert resolve_goal(["-2.0", "0.5"], map_path) == ((-2.0, 0.5), None)
    xy, place = resolve_goal(["kitchen"], map_path)
    assert xy == (-1.5, 2.0) and place is not None and place.name == "kitchen"
    with pytest.raises(ValueError, match="kitchen"):
        resolve_goal(["bedroom"], map_path)
    assert load_places(tmp_path / "other.npz") == {}


def test_the_residual_heading_is_the_short_way_round() -> None:
    from pepin.places import heading_residual_deg

    assert heading_residual_deg(140.0, 100.0) == 40.0  # counter-clockwise
    assert heading_residual_deg(-170.0, 170.0) == 20.0  # across the seam
    assert heading_residual_deg(10.0, 50.0) == -40.0
    assert heading_residual_deg(0.0, 180.0) == 180.0
    assert heading_residual_deg(35.0, 35.0) == 0.0


# ---- a place in a graph that bends ------------------------------------------------------------
def _pose(x: float, y: float, yaw_deg: float = 0.0) -> Pose2D:
    return Pose2D(x, y, math.radians(yaw_deg))


def test_a_place_is_the_carts_pose_relative_to_its_node_and_rides_it() -> None:
    """The whole point. The cart stands 60 cm in front of a node; the graph then moves that node
    by 30 cm and turns it, and the place moves with the furniture instead of staying on the old
    coordinate."""
    node, cart = _pose(2.0, 0.0), _pose(2.6, 0.0, 90.0)
    place = GraphPlace.measured("bookshelf", 7, node, cart)
    assert place.node == 7
    assert place.reach_m == pytest.approx(0.60)
    assert (place.at(node).x, place.at(node).y) == pytest.approx((2.6, 0.0)), "unmoved: unchanged"

    bent = _pose(2.0, 0.30, 10.0)  # the closure landed: the node moved and turned
    now = place.at(bent)
    assert now.x == pytest.approx(2.0 + 0.60 * math.cos(math.radians(10.0)))
    assert now.y == pytest.approx(0.30 + 0.60 * math.sin(math.radians(10.0)))
    assert math.degrees(now.theta) == pytest.approx(100.0), "the heading rides the node too"


def test_a_place_whose_node_the_graph_has_dropped_is_refused_not_guessed() -> None:
    """A pruned node is a place nobody can drive to, and a stale coordinate is worse than a
    refusal: it would send the cart somewhere that looks right."""
    place = GraphPlace("printer", 42, 0.5, 0.0, 0.0)
    assert graph_place_pose(place, {42: _pose(1.0, 1.0)}) is not None
    assert graph_place_pose(place, {1: _pose(1.0, 1.0)}) is None


def test_the_book_lives_beside_the_database_and_survives_a_restart(tmp_path: Path) -> None:
    """The label on the node is NOT the storage: beside a loaded database nothing is written and a
    label on a node in the working memory only flips a dirty bit, so the id and the offset are
    ours to keep."""
    database = tmp_path / "rtabmap.db"
    assert graph_places_path(database) == tmp_path / "rtabmap.places.json"
    places = {
        "home": GraphPlace("home", 3, 0.10, -0.20, 45.0, marked_at=1758000000.5),
        "printer": GraphPlace("printer", 91, 0.60, 0.0, -70.0),
    }
    save_graph_places(graph_places_path(database), places)
    assert load_graph_places(graph_places_path(database)) == places


def test_an_unreadable_book_costs_its_own_entries_and_never_the_room(tmp_path: Path) -> None:
    book = tmp_path / "rtabmap.places.json"
    assert load_graph_places(book) == {}, "an absent book is an empty one"
    book.write_text("not json at all")
    assert load_graph_places(book) == {}
    book.write_text(
        '{"places": {"good": {"node": 5, "dx": 1.0, "dy": 0.0, "dtheta_deg": 0.0},'
        ' "bad": {"node": "nine"}}}'
    )
    loaded = load_graph_places(book)
    assert set(loaded) == {"good"}, "one unreadable entry costs its own place, not the book"


def test_the_topic_carries_coordinates_with_the_node_they_ride() -> None:
    """What ``/places`` says: the coordinate a consumer drives to, plus the node and the reach
    that say how much to believe it."""
    places = {"printer": GraphPlace("printer", 91, 0.60, 0.0, -70.0, marked_at=12.0)}
    payload = places_json(places, {91: _pose(1.0, 2.0, 0.0)})
    heard = json.loads(payload)
    assert heard["printer"] == {
        "x": 1.6,
        "y": 2.0,
        "yaw_deg": -70.0,
        "node": 91,
        "reach_m": 0.6,
        "marked_at": 12.0,
    }
    back = places_from_json(payload)
    assert back["printer"].xy == (1.6, 2.0)
    assert back["printer"].theta_deg == pytest.approx(-70.0)


def test_a_place_the_graph_cannot_answer_for_is_left_out_of_the_topic() -> None:
    places = {
        "here": GraphPlace("here", 1, 0.0, 0.0, 0.0),
        "gone": GraphPlace("gone", 999, 0.0, 0.0, 0.0),
    }
    assert set(json.loads(places_json(places, {1: _pose(0.0, 0.0)}))) == {"here"}


def test_a_payload_the_board_cannot_read_is_an_empty_book_and_not_an_exception() -> None:
    """The reader is a goal client on the board: a payload it cannot parse must fall back to the
    file beside the map, not refuse the drive."""
    assert places_from_json("") == {}
    assert places_from_json("[1, 2, 3]") == {}
    assert places_from_json('{"half": {"x": 1.0}}') == {}, "an entry missing a field is skipped"


def test_the_three_topic_names_are_literals_both_ends_agree_on() -> None:
    assert (PLACES_TOPIC, MARK_TOPIC, MARKED_TOPIC) == (
        "/places",
        "/places/mark",
        "/places/marked",
    )
