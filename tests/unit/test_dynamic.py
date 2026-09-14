"""What the static map does not explain, and what that costs the pose."""

import math

import numpy as np

from pepin.dynamic import STATIC_M, StaticMask, to_map
from pepin.mapping import GridSpec, OccupancyGrid
from pepin.odometry import Pose2D


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


def test_the_reach_of_the_explanation_is_a_number_the_mask_is_built_with() -> None:
    """``map_grow`` (the tracker's flag) is this argument: at 0.15 m a wall's returns and the
    12 cm of map error beside them are the map; at 0.02 m only the mapped cell itself is."""
    grid = room()
    wide, narrow = StaticMask(grid, STATIC_M), StaticMask(grid, 0.02)
    beside = np.array([[2.90, 1.0]])  # 12 cm off the mapped wall at x = 3.0
    assert wide.grow_m == STATIC_M == 0.15 and narrow.grow_m == 0.02
    assert wide.explains(beside)[0] and not narrow.explains(beside)[0]
    assert narrow.explains(np.array([[3.02, 1.0]]))[0], "the mapped cell is always the map"


def test_helpers() -> None:
    moved = to_map(np.array([[1.0, 0.0]]), Pose2D(1.0, 1.0, math.pi / 2))
    assert np.allclose(moved, [[1.0, 2.0]])
    assert to_map(np.zeros((0, 2)), Pose2D()).shape == (0, 2)
    assert len(StaticMask(room()).explains(np.zeros((0, 2)))) == 0


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


def test_exactly_half_explained_votes_and_one_under_half_does_not() -> None:
    from pepin.dynamic import VOTE_MIN_SHARE, voting_mask

    mask = StaticMask(room())
    blanket = np.tile([[1.5, 2.0]], (100, 1))  # a hundred returns the map has no wall for
    vote = voting_mask(np.vstack([wall_scan(100), blanket]), Pose2D(), mask)
    assert VOTE_MIN_SHARE == 0.5 and vote is not None and int(vote.sum()) == 100
    assert voting_mask(np.vstack([wall_scan(99), blanket]), Pose2D(), mask) is None
    assert voting_mask(np.zeros((0, 2)), Pose2D(), mask) is None


class _Judge:
    """A matcher that always scores the same: what ``occluded`` asks of the real one."""

    def __init__(self, fit: float) -> None:
        self.fit = fit
        self.judged = 0  # how many points it was handed the last time

    def inlier_fraction(self, pose: Pose2D, points: np.ndarray) -> float:
        self.judged = len(points)
        return self.fit


def _crowded_scan() -> np.ndarray:
    """25 returns on a person half a metre ahead and 25 on the wall 2.5 m ahead, base frame."""
    person = np.column_stack([np.full(25, 0.5), np.linspace(-0.1, 0.1, 25)])
    wall = np.column_stack([np.full(25, 2.5), np.linspace(-0.5, 0.5, 25)])
    return np.vstack([person, wall])


def test_a_person_beside_the_cart_is_occlusion_while_the_walls_beyond_still_fit() -> None:
    """The tracker's own question, extracted: the near scan is mostly news, the far scan still
    lies on the map at this pose — a person, not a lost cart."""
    from pepin.dynamic import occluded

    pose = Pose2D(0.5, 2.0, 0.0)  # the wall at x = 3.0 is 2.5 m ahead
    points, mask, judge = _crowded_scan(), StaticMask(room()), _Judge(0.8)
    assert occluded(points, pose, mask, judge, lost_fit=0.35)
    assert judge.judged == 25, "only the far returns judge the pose"


def test_a_wrong_pose_is_lost_not_occluded_and_a_thin_scan_says_nothing() -> None:
    """The far test is what tells a person from a twin: a pose the walls refuse is not
    occluded. Before the map, the matcher or twenty returns, the answer is False."""
    from pepin.dynamic import occluded

    pose, points, mask = Pose2D(0.5, 2.0, 0.0), _crowded_scan(), StaticMask(room())
    assert not occluded(points, pose, mask, _Judge(0.1), lost_fit=0.35), "the walls refuse it"
    assert not occluded(points[:10], pose, mask, _Judge(0.8), lost_fit=0.35), "too few returns"
    assert not occluded(None, pose, mask, _Judge(0.8), lost_fit=0.35)
    assert not occluded(points, pose, None, _Judge(0.8), lost_fit=0.35), "no map yet"
    assert not occluded(points, pose, mask, None, lost_fit=0.35), "no matcher yet"


def test_a_scan_the_map_explains_from_end_to_end_is_not_occluded() -> None:
    """Nothing new near the cart: the near share is nothing and the question does not arise."""
    from pepin.dynamic import occluded

    pose, mask = Pose2D(0.5, 2.0, 0.0), StaticMask(room())
    on_wall = np.column_stack([np.full(50, 2.5), np.linspace(-0.5, 0.5, 50)])
    assert not occluded(on_wall, pose, mask, _Judge(0.9), lost_fit=0.35)
