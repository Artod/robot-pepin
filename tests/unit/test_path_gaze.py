"""Path gaze and reverse gaze (pepin.path_gaze): the point ahead, the clamp, the dead-band, the
following in a zone (hysteresis, cooldown, tail; the baseline preset is the dead-band), the
rear's side and the tight rear."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pepin.gaze import Aim, Reach, home_aim
from pepin.neck import NeckConfig
from pepin.path_gaze import (
    PathFollower,
    PathGazeLaw,
    ReverseLaw,
    ReverseWatch,
    lookahead_m,
    path_aim,
    remaining_m,
    reverse_aim,
    settle,
    tight_rear,
    time_to_end,
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


# Drive 306's reverse leg out of home (2026-10-05 19:36:31Z, the tape's commands at 10 Hz): backing
# at 0.06-0.14 m/s while the controller swung the turn through zero four times.
# fmt: off
DRIVE_306_LEG_W = (
    -0.4, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5,
    0.56, 0.58, 0.6, 0.6, 0.61, 0.61, 0.41, 0.21, 0.01, -0.19, -0.39, -0.59, -0.78, -0.83, -0.87,
    -0.91, -0.91, -0.71, -0.51, -0.31, -0.11, 0.09, 0.29, 0.49, 0.69, 0.89, 0.69, 0.49, 0.29, 0.09,
    0.29, 0.49, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3,
    0.1, -0.1, 0.1, 0.3,
)
# fmt: on


def test_a_reverse_look_keeps_its_side_while_the_controller_wiggles() -> None:
    law = ReverseLaw()
    held, free = ReverseWatch(), ReverseWatch()
    sides, free_sides = [], []
    for k, w in enumerate(DRIVE_306_LEG_W):
        now = 0.1 * k
        held.update(-0.08, w, now, law)
        free.update(-0.08, w, now, law)
        if held.reversing_for(now) >= law.min_s and k % 2 == 0:  # the node's 0.2 s period
            sides.append(held.hold())
            free_sides.append(free.side)
    assert len(set(free_sides)) == 2  # following w, the look would have crossed the back
    assert len(sides) > 20 and set(sides) == {sides[0]}
    held.update(0.1, 0.0, 7.0, law)  # the leg ends: the next one chooses again
    held.update(-0.1, -sides[0] * 0.5, 8.0, law)
    assert held.side == sides[0]
    held.update(-0.1, sides[0] * 0.5, 8.1, law)
    assert held.side == -sides[0]


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


# ---- following in a zone -------------------------------------------------------------------------
PRESETS = json.loads((REPO / "config/gaze_presets.json").read_text())
FOLLOW = PathGazeLaw()


def law_of(preset: str) -> PathGazeLaw:
    """config/gaze_presets.json's set as path gaze's law (the knobs path gaze reads)."""
    knobs = PRESETS[preset]
    return replace(
        PathGazeLaw(),
        deadband_deg=knobs["path_deadband_deg"],
        hyst_s=knobs["path_hyst_s"],
        cooldown_s=knobs["path_cooldown_s"],
        tail_s=knobs["path_tail_s"],
        tail_m=knobs["path_tail_m"],
        hold_s=knobs["path_hold_s"],
    )


def pan(deg: float) -> Aim:
    return Aim(math.radians(deg), HOME.tilt_rad)


def test_the_baseline_follower_is_the_dead_band_it_replaced() -> None:
    """At the baseline preset the follower answers settle() on every input, whatever the tail
    and the hand-overs say."""
    rng = np.random.default_rng(306)
    follower, law, current = PathFollower(), law_of("baseline"), None
    for k in range(400):
        wanted = pan(float(rng.uniform(-60.0, 60.0)) if k % 7 else float(rng.normal(0.0, 4.0)))
        held = follower.update(
            wanted,
            0.2 * k,
            law,
            end_in_s=float(rng.uniform(0.0, 3.0)),
            left_m=float(rng.uniform(0.0, 1.0)),
            fresh=bool(rng.integers(2)),
        )
        current = settle(current, wanted, 8.0)
        assert held == current


def test_the_follow_defaults_are_the_knobs_and_the_preset() -> None:
    assert law_of("follow") == FOLLOW
    assert (
        FOLLOW.deadband_deg,
        FOLLOW.hyst_s,
        FOLLOW.cooldown_s,
        FOLLOW.tail_s,
        FOLLOW.tail_m,
    ) == (22.0, 0.3, 2.0, 0.5, 0.35)


def test_the_head_follows_in_its_zone() -> None:
    follower = PathFollower()
    assert follower.update(pan(10), 0.0, FOLLOW) == pan(10)  # the drive's first aim: at once
    for t, deg in ((0.2, 25), (0.4, -5), (0.6, 31)):  # within 22 deg of 10
        assert follower.update(pan(deg), t, FOLLOW) == pan(10)


def test_a_new_aim_must_stay_out_of_the_zone_for_the_hysteresis() -> None:
    follower = PathFollower()
    follower.update(pan(0), 0.0, FOLLOW)
    assert follower.update(pan(40), 3.0, FOLLOW) == pan(0)  # out since 3.0
    assert follower.update(pan(5), 3.2, FOLLOW) == pan(0)  # back in: the clock restarts
    assert follower.update(pan(40), 3.4, FOLLOW) == pan(0)
    assert follower.update(pan(45), 3.6, FOLLOW) == pan(0)  # 0.2 s out
    assert follower.update(pan(45), 3.7, FOLLOW) == pan(45)  # 0.3 s out: the newest aim


def test_moves_are_a_cooldown_apart() -> None:
    follower = PathFollower()
    follower.update(pan(0), 0.0, FOLLOW)
    follower.update(pan(40), 0.0, FOLLOW)
    assert follower.update(pan(40), 0.3, FOLLOW) == pan(0)  # out 0.3 s, but 0.3 s since a move
    assert follower.update(pan(40), 1.9, FOLLOW) == pan(0)
    assert follower.update(pan(40), 2.0, FOLLOW) == pan(40)
    assert follower.update(pan(-10), 2.5, FOLLOW) == pan(40)
    assert follower.update(pan(-10), 3.9, FOLLOW) == pan(40)
    assert follower.update(pan(-10), 4.0, FOLLOW) == pan(-10)


def test_a_path_look_that_lost_the_head_is_aimed_at_once_but_never_in_the_tail() -> None:
    follower = PathFollower()
    follower.update(pan(0), 0.0, FOLLOW)
    assert follower.update(pan(50), 0.1, FOLLOW, fresh=True) == pan(50)  # no hysteresis
    assert follower.update(pan(-50), 0.2, FOLLOW, fresh=True) == pan(-50)  # no cooldown
    assert follower.update(pan(0), 5.0, FOLLOW, end_in_s=0.4) == pan(-50)  # the tail: held
    assert follower.update(pan(-50), 5.1, FOLLOW, end_in_s=0.4, fresh=True) is None
    follower.reset()
    assert follower.update(pan(30), 6.0, FOLLOW, end_in_s=0.4) is None  # nothing new in it
    assert follower.update(pan(30), 6.0, FOLLOW, end_in_s=0.6) == pan(30)


def test_the_plans_last_metres_are_the_tail_whatever_the_speed() -> None:
    """Drive 0330's end: 0.16 m of plan left at 0.06 m/s reads 2.5 s, far outside the 0.5 s
    tail; the metres alone hold the aim."""
    follower = PathFollower()
    follower.update(pan(2), 0.0, FOLLOW, end_in_s=5.6, left_m=1.68)
    end_in = time_to_end(0.16, 0.06)
    assert end_in > FOLLOW.tail_s
    for t in (3.0, 3.3, 3.6, 6.0):  # out of the zone, past the hysteresis and the cooldown
        assert follower.update(pan(-38), t, FOLLOW, end_in_s=end_in, left_m=0.16) == pan(2)
    assert follower.update(pan(-38), 6.2, FOLLOW, end_in_s=math.inf, left_m=0.13) == pan(2)
    assert follower.update(pan(-38), 6.4, FOLLOW, end_in_s=end_in, left_m=0.16, fresh=True) is None
    # out of the tail (a replan longer by a few cm): the hysteresis starts there
    assert follower.update(pan(-38), 6.6, FOLLOW, end_in_s=end_in, left_m=0.36) == pan(2)
    assert follower.update(pan(-38), 6.9, FOLLOW, end_in_s=end_in, left_m=0.36) == pan(-38)
    off = replace(FOLLOW, tail_m=0.0)  # the knob at 0: the seconds alone, as before
    follower.reset()
    follower.update(pan(2), 0.0, off)
    assert follower.update(pan(-38), 3.0, off, end_in_s=end_in, left_m=0.16) == pan(2)
    assert follower.update(pan(-38), 3.4, off, end_in_s=end_in, left_m=0.16) == pan(-38)


def test_the_time_to_the_plans_end() -> None:
    assert remaining_m(STRAIGHT, (0.0, 0.0)) == pytest.approx(3.95)
    assert remaining_m(STRAIGHT, (3.0, 0.1)) == pytest.approx(0.95)
    assert remaining_m(np.zeros((1, 2)), (0.0, 0.0)) == 0.0
    assert time_to_end(0.95, 0.19) == pytest.approx(5.0)
    assert time_to_end(0.95, -0.19) == pytest.approx(5.0)
    assert time_to_end(0.95, 0.01) == math.inf  # standing: no end in sight
