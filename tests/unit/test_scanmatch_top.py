"""The multi-peak search: k rivals kept apart by non-maximum suppression."""

import math
from itertools import combinations

import numpy as np
import pytest
from synthetic import raycast_room

from pepin.mapping import GridSpec, OccupancyGrid
from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import CorrelativeMatcher, SearchWindow

SPEC = GridSpec(0.05, -4, -3, 8, 6)
MAPPING_POSES = (Pose2D(0, 0, 0), Pose2D(1, 0.5, 0.7), Pose2D(-1, -0.5, -2.0))
WINDOW = SearchWindow(0.2, 0.05, 12.0, 2.0)
TRUTH = Pose2D(0.4, -0.3, 0.5)
GUESS = Pose2D(0.45, -0.35, 0.5 + math.radians(5.0))


def room_map() -> OccupancyGrid:
    """The empty rectangle, mapped from three poses inside it."""
    grid = OccupancyGrid(SPEC)
    for pose in MAPPING_POSES:
        grid.integrate(pose, raycast_room(pose))
    return grid


def scan() -> np.ndarray:
    """One revolution seen from :data:`TRUTH`."""
    return raycast_room(TRUTH)


def test_peaks_come_back_best_first_starting_at_the_true_pose() -> None:
    peaks = CorrelativeMatcher(room_map()).match_top(GUESS, scan(), 4, WINDOW)
    assert len(peaks) == 4
    assert [p.score for p in peaks] == sorted((p.score for p in peaks), reverse=True)
    assert peaks[0].pose.x == pytest.approx(TRUTH.x, abs=0.051)
    assert peaks[0].pose.y == pytest.approx(TRUTH.y, abs=0.051)
    assert peaks[0].pose.theta == pytest.approx(TRUTH.theta, abs=math.radians(2.1))


def test_the_best_peak_is_what_a_plain_match_returns() -> None:
    matcher = CorrelativeMatcher(room_map())
    points = scan()
    top = matcher.match_top(GUESS, points, 3, WINDOW)[0]
    single = matcher.match(GUESS, points, WINDOW)
    assert top.pose == single.pose
    assert top.score == single.score and top.guess_score == single.guess_score


def test_every_pair_of_peaks_stands_apart_in_position_or_in_heading() -> None:
    apart_steps = 3
    peaks = CorrelativeMatcher(room_map()).match_top(GUESS, scan(), 5, WINDOW, apart_steps)
    assert len(peaks) == 5
    for a, b in combinations(peaks, 2):
        far_xy = max(abs(a.pose.x - b.pose.x), abs(a.pose.y - b.pose.y))
        far_theta = abs(wrap_angle(a.pose.theta - b.pose.theta))
        assert (
            far_xy >= apart_steps * WINDOW.xy_step_m - 1e-9
            or far_theta >= math.radians(apart_steps * WINDOW.theta_step_deg) - 1e-9
        )


def test_a_lattice_too_small_for_k_returns_the_candidates_it_has() -> None:
    matcher = CorrelativeMatcher(room_map())
    tiny = SearchWindow(0.02, 0.02, 0.5, 0.5)  # 3 x 3 positions, 3 headings
    assert len(matcher.match_top(TRUTH, scan(), 50, tiny, apart_steps=1)) == 9 * 3
    assert len(matcher.match_top(TRUTH, scan(), 50, tiny)) == 1  # 3 steps suppress the lot
