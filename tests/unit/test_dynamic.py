"""New objects get a berth; the mapped world keeps its close approach."""

import math

import numpy as np

from pepin.dynamic import (
    COSTMAP_CELL_M,
    StaticMask,
    berth_for,
    dedupe,
    dynamic_marks,
    ring_points,
    rings,
    to_map,
)
from pepin.mapping import GridSpec, OccupancyGrid
from pepin.odometry import Pose2D

POINT = berth_for("GridBased")
RING_M, NEAR_M = POINT.ring_m, POINT.near_m
PER_CENTRE = 1 + ring_points(RING_M)  # the centre itself, then the ring closing around it


def room() -> OccupancyGrid:
    """A 4 x 4 m room at 5 cm with an occupied wall along x = 3.0."""
    grid = OccupancyGrid(GridSpec(0.05, 0.0, 0.0, 4.0, 4.0))
    grid.log_odds[:, 60] = 4.0  # column 60 = x in [3.0, 3.05)
    return grid


def test_the_mask_explains_the_wall_and_its_surroundings_but_not_the_middle_of_the_room() -> None:
    mask = StaticMask(room())
    on_wall = np.array([[3.02, 1.0]])
    near_wall = np.array([[2.90, 1.0]])  # 12 cm off the mapped wall: map error, not a person
    middle = np.array([[1.5, 2.0]])
    off_grid = np.array([[9.0, 9.0]])
    assert mask.explains(on_wall)[0] and mask.explains(near_wall)[0]
    assert not mask.explains(middle)[0]
    assert mask.explains(off_grid)[0], "unknown ground is not evidence of a person"


def test_a_person_in_the_room_becomes_a_lethal_ring_and_the_wall_does_not() -> None:
    mask = StaticMask(room())
    pose = Pose2D(0.5, 2.0, 0.0)
    legs = np.array([[1.50, 0.02], [1.52, -0.03], [1.55, 0.05]])  # 1.5 m ahead, in the base frame
    wall = np.array([[2.49, 0.10], [2.49, -0.10]])  # the mapped wall at x = 3.0
    marks = dynamic_marks(np.vstack([legs, wall]), pose, mask, POINT)
    legs_map = to_map(legs, pose)
    near_legs = np.hypot(marks[:, 0] - legs_map[:, 0].mean(), marks[:, 1] - legs_map[:, 1].mean())
    centres = marks[near_legs < 0.10]
    assert 1 <= len(centres) <= 3, "three returns 5 cm apart are one object, or nearly"
    assert len(marks) == len(centres) * PER_CENTRE, (
        "each centre carries its ring; the wall carries none"
    )
    ring = marks[near_legs >= 0.10]
    to_centre = np.hypot(
        ring[:, None, 0] - centres[None, :, 0], ring[:, None, 1] - centres[None, :, 1]
    ).min(axis=1)
    # a ring point may sit nearer a neighbouring centre (5 cm apart) than its own
    assert np.all(to_centre <= RING_M + 0.02) and np.all(to_centre >= RING_M - 0.06)


def test_what_stands_beside_a_parked_cart_gets_no_ring() -> None:
    mask = StaticMask(room())
    pose = Pose2D(0.5, 2.0, 0.0)
    beside = np.array([[0.30, 0.30]])  # 42 cm away: the contact band and the footprint handle it
    assert len(dynamic_marks(beside, pose, mask, POINT)) == 0
    farther = np.array([[NEAR_M + 0.05, 0.0]])
    assert len(dynamic_marks(farther, pose, mask, POINT)) == PER_CENTRE


def test_a_ring_leaves_no_gap_a_point_planner_could_walk_through() -> None:
    """A point planner is stopped by lethal cells, never by the gaps between them: twelve marks
    on a 0.41 m ring stand 0.22 m apart and the plan goes straight between two of them."""
    ring = rings(np.zeros((1, 2)), RING_M)[1:]
    step = np.hypot(*(ring - np.roll(ring, 1, axis=0)).T)
    assert step.max() <= COSTMAP_CELL_M + 1e-9
    assert ring_points(RING_M) == len(ring) > 12
    assert ring_points(0.001) == 12, "a tiny ring still gets a full dozen"


def test_helpers() -> None:
    assert len(dedupe(np.array([[0.01, 0.01], [0.02, 0.03], [0.5, 0.5]]))) == 2
    assert rings(np.zeros((0, 2)), RING_M).shape == (0, 2)
    moved = to_map(np.array([[1.0, 0.0]]), Pose2D(1.0, 1.0, math.pi / 2))
    assert np.allclose(moved, [[1.0, 2.0]])
    assert len(dynamic_marks(np.zeros((0, 2)), Pose2D(), StaticMask(room()), POINT)) == 0


def test_occlusion_is_near_and_unexplained_while_the_far_scan_is_the_judge() -> None:
    from pepin.dynamic import occlusion_split

    pts = np.array([[0.5, 0.0], [0.6, 0.1], [0.5, -0.1], [3.0, 0.0], [0.0, 3.0], [-3.0, 0.0]])
    explained = np.array([False, False, True, True, True, True])
    share, far = occlusion_split(pts, explained, near_m=1.5)
    assert abs(share - 2 / 3) < 1e-9
    assert far.tolist() == [False, False, False, True, True, True]
    share, far = occlusion_split(pts[3:], explained[3:], near_m=1.5)
    assert share == 0.0 and far.all()
    share, far = occlusion_split(np.zeros((0, 2)), np.zeros(0, dtype=bool))
    assert share == 0.0 and len(far) == 0


def test_the_berth_is_the_toes_for_a_footprint_planner_and_the_hull_for_a_point_one() -> None:
    from pepin.dynamic import (
        COSTMAP_CELL_M,
        HAND_M,
        TOE_REACH_M,
        berth_for,
        footprint_planner_ring_m,
        near_exclusion_m,
        point_planner_ring_m,
    )
    from pepin.footprint import HULL

    assert point_planner_ring_m() == HULL.half_width_m - HULL.inscribed_radius_m + TOE_REACH_M
    assert footprint_planner_ring_m() == TOE_REACH_M + HAND_M
    assert near_exclusion_m(0.25) == 0.25 + HULL.circumscribed_radius_m + COSTMAP_CELL_M
    hybrid = berth_for("Hybrid")
    assert hybrid.ring_m == footprint_planner_ring_m() == 0.25
    assert berth_for("Lattice") == hybrid
    for point in ("GridBased", "Smac2D", "ThetaStar", ""):
        b = berth_for(point)
        assert b.ring_m == point_planner_ring_m() and b.near_m == near_exclusion_m(b.ring_m)
    assert hybrid.near_m < berth_for("GridBased").near_m  # a person at 0.7 m is seen by Hybrid


# -- who may vote in the scan match -------------------------------------------


def wall_scan(n: int = 200) -> np.ndarray:
    """``n`` returns spread along the mapped wall at x = 3.0, seen from the origin."""
    ys = np.linspace(0.2, 3.8, n)
    return np.column_stack((np.full(n, 3.02), ys))


def test_the_returns_the_map_explains_are_the_ones_that_vote() -> None:
    from pepin.dynamic import voting_mask

    mask = StaticMask(room())
    points = wall_scan()
    points[:20] = [1.5, 2.0]  # a blanket in the middle of the room: the map has no wall there
    vote = voting_mask(points, Pose2D(), mask)
    assert vote is not None
    assert not vote[:20].any() and vote[20:].all()


def test_a_scan_the_map_barely_explains_lets_everything_vote() -> None:
    """A pose far off puts the whole scan on open floor; a mask built there would silence
    exactly the returns that could fix it."""
    from pepin.dynamic import voting_mask

    assert voting_mask(wall_scan(), Pose2D(-1.5, 0.0, 0.0), StaticMask(room())) is None


def test_too_few_explained_returns_let_everything_vote() -> None:
    from pepin.dynamic import VOTE_MIN_POINTS, voting_mask

    mask = StaticMask(room())
    assert voting_mask(wall_scan(VOTE_MIN_POINTS - 1), Pose2D(), mask) is None
    assert voting_mask(wall_scan(VOTE_MIN_POINTS), Pose2D(), mask) is not None
