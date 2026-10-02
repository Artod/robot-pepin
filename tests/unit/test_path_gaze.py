"""Path gaze and reverse gaze (pepin.path_gaze): the point ahead, the clamp, the dead-band, the
rear's side and the tight rear."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pepin.gaze import Aim, Reach, home_aim
from pepin.neck import NeckConfig
from pepin.path_gaze import (
    PathGazeLaw,
    ReverseLaw,
    ReverseWatch,
    lookahead_m,
    path_aim,
    reverse_aim,
    settle,
    tight_rear,
)

REPO = Path(__file__).resolve().parents[2]
CFG = NeckConfig.from_json(REPO / "config/neck.json")
HOME = home_aim(CFG)
REACH = Reach.of(CFG)
LAW = PathGazeLaw()
STRAIGHT = np.column_stack((np.arange(0.0, 4.0, 0.05), np.zeros(80)))


def aim(path: np.ndarray, pose: tuple[float, float, float], v: float) -> Aim:
    found = path_aim(path, pose, v, LAW, lens_z_m=1.2, home=HOME, reach=REACH)
    assert found is not None
    return found


def test_the_lookahead_is_two_seconds_of_speed_within_its_bounds() -> None:
    assert lookahead_m(0.0, LAW) == 0.6
    assert lookahead_m(0.3, LAW) == pytest.approx(0.6)
    assert lookahead_m(0.6, LAW) == pytest.approx(1.2)
    assert lookahead_m(-2.0, LAW) == 1.5


def test_a_straight_path_is_straight_ahead_at_home_tilt_when_far() -> None:
    found = aim(STRAIGHT, (0.0, 0.0, 0.0), 0.6)
    assert found.pan_rad == pytest.approx(0.0, abs=1e-9)
    assert found.tilt_rad == pytest.approx(HOME.tilt_rad)


def test_a_near_point_tilts_the_head_down() -> None:
    found = aim(STRAIGHT, (0.0, 0.0, 0.0), 0.1)  # 0.6 m ahead
    expected = math.atan2(1.2, 0.6) - math.radians(LAW.near_offset_deg)
    assert found.tilt_rad == pytest.approx(expected, abs=1e-3)


def test_a_turn_ahead_pans_toward_it_and_a_goal_behind_is_clamped() -> None:
    left = np.array([[0.0, 0.0], [0.3, 0.0], [0.3, 1.5]])
    assert aim(left, (0.0, 0.0, 0.0), 0.6).pan_rad > math.radians(30)
    behind = np.array([[0.0, 0.0], [-0.1, 0.1], [-1.5, 0.1]])
    assert aim(behind, (0.0, 0.0, 0.0), 0.6).pan_rad == pytest.approx(math.radians(60))
    assert aim(behind, (0.0, 0.0, math.pi / 2), 0.6).pan_rad == pytest.approx(math.radians(60))


def test_no_path_ahead_is_no_aim() -> None:
    assert (
        path_aim(np.zeros((1, 2)), (0, 0, 0), 0.3, LAW, lens_z_m=1.2, home=HOME, reach=REACH)
        is None
    )
    end = STRAIGHT[-1]
    assert (
        path_aim(STRAIGHT, (end[0], end[1], 0.0), 0.3, LAW, lens_z_m=1.2, home=HOME, reach=REACH)
        is None
    )


def test_the_dead_band_keeps_the_head_where_it_is() -> None:
    current = Aim(0.1, HOME.tilt_rad)
    assert settle(current, Aim(0.1 + math.radians(7), HOME.tilt_rad), 8.0) is current
    wanted = Aim(0.1 + math.radians(9), HOME.tilt_rad)
    assert settle(current, wanted, 8.0) is wanted
    assert settle(None, wanted, 8.0) is wanted


def test_a_reverse_leg_is_timed_and_the_rear_swings_away_from_the_turn() -> None:
    watch, law = ReverseWatch(), ReverseLaw()
    watch.update(-0.1, 0.3, 0.0, law)
    watch.update(-0.1, 0.0, 0.8, law)
    assert watch.reversing and watch.reversing_for(1.2) == pytest.approx(1.2)
    assert watch.side == -1  # turning left while backing: the rear swings right
    assert reverse_aim(watch.side, law, REACH).pan_rad == pytest.approx(-math.radians(150))
    watch.update(0.1, 0.0, 1.3, law)
    assert not watch.reversing and watch.reversing_for(1.4) == 0.0
    watch.update(-0.1, -0.2, 2.0, law)
    assert watch.side == 1 and reverse_aim(1, law, REACH).tilt_rad == pytest.approx(HOME.tilt_rad)


def test_a_tight_rear_is_a_lethal_cell_just_behind_the_hull() -> None:
    res, size = 0.05, 40
    origin = (-1.0, -1.0)
    grid = np.zeros((size, size), dtype=np.int16)

    def mark(x: float, y: float) -> None:
        grid[int((y - origin[1]) / res), int((x - origin[0]) / res)] = 100

    assert not tight_rear(grid, origin, res, (0.0, 0.0, 0.0), 0.3)
    mark(0.5, 0.0)  # ahead: not the rear
    mark(-0.5, 0.6)  # behind but beside the hull
    assert not tight_rear(grid, origin, res, (0.0, 0.0, 0.0), 0.3)
    mark(-0.45, 0.1)
    assert tight_rear(grid, origin, res, (0.0, 0.0, 0.0), 0.3)
    assert not tight_rear(grid, origin, res, (0.0, 0.0, math.pi / 2), 0.3)  # facing +y
    assert tight_rear(grid, origin, res, (0.0, 0.0, math.pi), 0.3)  # (0.5, 0) is behind now
