"""Where to stand for a relation to a target (pepin.stand_pose): the frames of a person, a thing
and a place, the distances, the facing, and the ring's alternatives."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pepin.pixel_to_map import Pose2D
from pepin.stand_pose import RELATIONS, Distances, Stand, Target, facing_from, stands

REPO = Path(__file__).resolve().parents[2]
AT_ORIGIN = Pose2D(0.0, 0.0, 0.0)
FACING_ROBOT = math.pi  # a person at +x looking back toward the origin


def xy(stand: Stand) -> tuple[float, float]:
    return stand.x, stand.y


def test_to_a_person_is_on_the_line_toward_the_robot_facing_them() -> None:
    person = Target("person", 3.0, 0.0, FACING_ROBOT)
    first, left, right = stands(person, "to", AT_ORIGIN)
    assert xy(first) == pytest.approx((2.2, 0.0))
    assert first.yaw_deg == pytest.approx(0.0)
    for alternative, turn in ((left, 30.0), (right, -30.0)):  # the ring about the person
        a = math.radians(180.0 + turn)
        assert xy(alternative) == pytest.approx((3.0 + 0.8 * math.cos(a), 0.8 * math.sin(a)))
        assert math.hypot(alternative.x - 3.0, alternative.y) == pytest.approx(0.8)
        facing = math.atan2(-alternative.y, 3.0 - alternative.x)  # still facing the person
        assert math.radians(alternative.yaw_deg) == pytest.approx(facing)
    assert "turned 30 deg left" in left.note


def test_a_person_is_never_stood_nearer_than_the_minimum_plus_two_sigma() -> None:
    blurred = Target("person", 3.0, 0.0, FACING_ROBOT, sigma_m=0.2)
    assert xy(stands(blurred, "to", AT_ORIGIN)[0]) == pytest.approx((2.0, 0.0))  # 0.6 + 0.4
    sharp = Target("person", 3.0, 0.0, FACING_ROBOT)
    spoken = stands(sharp, "to", AT_ORIGIN, distance_m=0.3)[0]  # "come up close, 30 cm"
    assert xy(spoken) == pytest.approx((2.4, 0.0))
    assert xy(stands(sharp, "to", AT_ORIGIN, distance_m=1.5)[0]) == pytest.approx((1.5, 0.0))


def test_right_of_a_person_is_their_own_right_side_by_side() -> None:
    """Facing the robot (west), the person's right is north: the robot's left as it sees them."""
    person = Target("person", 3.0, 0.0, FACING_ROBOT)
    first = stands(person, "right_of", AT_ORIGIN)[0]
    assert xy(first) == pytest.approx((3.0, 0.7))
    assert first.yaw_deg == pytest.approx(180.0)  # looking where the person looks
    assert xy(stands(person, "left_of", AT_ORIGIN)[0]) == pytest.approx((3.0, -0.7))
    turned = Target("person", 3.0, 0.0, 0.0)  # now looking away, east
    assert xy(stands(turned, "right_of", AT_ORIGIN)[0]) == pytest.approx((3.0, -0.7))


def test_in_front_of_and_behind_a_person_follow_their_facing() -> None:
    person = Target("person", 3.0, 0.0, math.radians(90.0))  # looking north
    front = stands(person, "in_front_of", AT_ORIGIN)[0]
    assert xy(front) == pytest.approx((3.0, 0.8))
    assert front.yaw_deg == pytest.approx(-90.0)  # facing them
    back = stands(person, "behind", AT_ORIGIN)[0]
    assert xy(back) == pytest.approx((3.0, -0.8))
    assert back.yaw_deg == pytest.approx(90.0)  # over their shoulder


def test_a_thing_s_sides_are_the_viewer_s_and_the_robot_ends_facing_it() -> None:
    shelf = Target("thing", 2.0, 0.0)
    right = stands(shelf, "right_of", AT_ORIGIN)[0]
    assert xy(right) == pytest.approx((2.0, -0.7))
    assert right.yaw_deg == pytest.approx(90.0)
    behind = stands(shelf, "behind", AT_ORIGIN)[0]
    assert xy(behind) == pytest.approx((2.8, 0.0))
    assert behind.yaw_deg == pytest.approx(180.0)
    front = stands(shelf, "in_front_of", AT_ORIGIN)[0]
    assert xy(front) == pytest.approx((1.2, 0.0))
    assert xy(stands(shelf, "to", AT_ORIGIN, distance_m=0.3)[0]) == pytest.approx((1.7, 0.0))


def test_beside_takes_the_nearer_side_first_and_the_other_next() -> None:
    person = Target("person", 3.0, 0.0, FACING_ROBOT)  # their right is north
    near, far, *ring = stands(person, "beside", Pose2D(0.0, 1.0, 0.0))
    assert (xy(near), xy(far)) == (pytest.approx((3.0, 0.7)), pytest.approx((3.0, -0.7)))
    assert len(ring) == 2
    assert xy(stands(person, "beside", Pose2D(0.0, -1.0, 0.0))[0]) == pytest.approx((3.0, -0.7))
    shelf = Target("thing", 2.0, 0.0)  # a thing's sides are the viewer's: both equally near
    near, far, *_ = stands(shelf, "beside", AT_ORIGIN)
    assert (xy(near), xy(far)) == (pytest.approx((2.0, -0.7)), pytest.approx((2.0, 0.7)))


def test_a_place_is_a_standing_pose_and_its_sides_are_its_own() -> None:
    printer = Target("place", 1.0, 1.0, math.radians(90.0))
    (itself,) = stands(printer, "to", AT_ORIGIN)
    assert (xy(itself), itself.yaw_deg) == (pytest.approx((1.0, 1.0)), pytest.approx(90.0))
    assert len(stands(printer, "beside", AT_ORIGIN)) == 1
    right = stands(printer, "right_of", AT_ORIGIN)
    assert xy(right[0]) == pytest.approx((1.7, 1.0))
    assert right[0].yaw_deg == pytest.approx(90.0)
    assert len(right) == 3
    assert xy(stands(printer, "behind", AT_ORIGIN)[0]) == pytest.approx((1.0, 0.2))


def test_every_relation_answers_for_every_kind() -> None:
    for kind in ("person", "thing", "place"):
        target = Target(kind, 2.0, 1.0, 0.5)  # type: ignore[arg-type]
        for relation in RELATIONS:
            assert stands(target, relation, AT_ORIGIN), (kind, relation)  # type: ignore[arg-type]


def test_a_person_s_facing_is_toward_the_robot_turned_by_their_head() -> None:
    assert facing_from(3.0, 0.0, AT_ORIGIN, 0.0) == pytest.approx(math.pi)
    assert facing_from(3.0, 0.0, AT_ORIGIN, 20.0) == pytest.approx(math.pi + math.radians(20.0))


def test_the_config_file_holds_the_code_s_defaults() -> None:
    assert Distances.load(REPO / "config" / "approach.json") == Distances()
