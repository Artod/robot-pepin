"""Scan matching on a synthetic rectangular room."""

import math

import numpy as np
import pytest

from pepin.mapping import GridSpec, OccupancyGrid
from pepin.odometry import Pose2D
from pepin.scanmatch import (
    CorrelativeMatcher,
    SearchWindow,
    apply_motion,
    relative_motion,
    should_keyframe,
)

ROOM_W, ROOM_H = 6.0, 4.0


def raycast_room(pose: Pose2D, beams: int = 360) -> np.ndarray:
    """Robot-frame hit points of rays from ``pose`` against the walls of the room."""
    pts = []
    for k in range(beams):
        a = pose.theta + 2 * math.pi * k / beams
        dx, dy = math.cos(a), math.sin(a)
        ts = []
        for wall_x in (-ROOM_W / 2, ROOM_W / 2):
            t = (wall_x - pose.x) / dx if abs(dx) > 1e-9 else -1
            if t > 0 and abs(pose.y + t * dy) <= ROOM_H / 2:
                ts.append(t)
        for wall_y in (-ROOM_H / 2, ROOM_H / 2):
            t = (wall_y - pose.y) / dy if abs(dy) > 1e-9 else -1
            if t > 0 and abs(pose.x + t * dx) <= ROOM_W / 2:
                ts.append(t)
        r = min(ts)
        pts.append((r * math.cos(a - pose.theta), r * math.sin(a - pose.theta)))
    return np.array(pts)


def room_map() -> OccupancyGrid:
    grid = OccupancyGrid(GridSpec(0.05, -4, -3, 8, 6))
    for pose in (Pose2D(0, 0, 0), Pose2D(1, 0.5, 0.7), Pose2D(-1, -0.5, -2.0)):
        grid.integrate(pose, raycast_room(pose))
    return grid


def test_motion_helpers_round_trip() -> None:
    a, b = Pose2D(1.0, 2.0, 0.3), Pose2D(1.5, 2.2, -0.4)
    back = apply_motion(a, relative_motion(a, b))
    assert back.x == pytest.approx(b.x) and back.y == pytest.approx(b.y)
    assert back.theta == pytest.approx(b.theta)


def test_recovers_an_injected_pose_error() -> None:
    truth = Pose2D(0.4, -0.3, 0.9)
    matcher = CorrelativeMatcher(room_map(), SearchWindow(0.08, 0.02, 6.0, 0.5))
    guess = Pose2D(truth.x + 0.06, truth.y - 0.04, truth.theta + math.radians(4.0))
    result = matcher.match(guess, raycast_room(truth))
    assert result.improved
    assert result.pose.x == pytest.approx(truth.x, abs=0.021)
    assert result.pose.y == pytest.approx(truth.y, abs=0.021)
    assert result.pose.theta == pytest.approx(truth.theta, abs=math.radians(0.51))


def test_empty_map_returns_the_guess() -> None:
    matcher = CorrelativeMatcher(OccupancyGrid(GridSpec(0.05, -4, -3, 8, 6)))
    guess = Pose2D(0.2, 0.1, 0.3)
    result = matcher.match(guess, raycast_room(guess))
    assert result.pose == guess and not result.improved


def test_score_prefers_the_true_pose_over_a_rotated_one() -> None:
    matcher = CorrelativeMatcher(room_map())
    pts = raycast_room(Pose2D(0, 0, 0))
    assert matcher.score(Pose2D(0, 0, 0), pts) > matcher.score(Pose2D(0, 0, 0.3), pts)


def test_keyframe_needs_enough_motion() -> None:
    assert not should_keyframe(Pose2D(0.01, 0.0, math.radians(1.0)))
    assert should_keyframe(Pose2D(0.04, 0.0, 0.0))
    assert should_keyframe(Pose2D(0.0, 0.0, math.radians(3.0)))


def test_window_widens_for_large_odometry_steps_at_constant_cost() -> None:
    base = SearchWindow()
    w = base.widened_for(Pose2D(0.2, 0.0, math.radians(20.0)))
    assert w.xy_m == pytest.approx(0.3) and w.theta_deg == pytest.approx(30.0)
    assert w.xy_m / w.xy_step_m == pytest.approx(base.xy_m / base.xy_step_m)
    assert w.theta_deg / w.theta_step_deg == pytest.approx(base.theta_deg / base.theta_step_deg)
    assert base.widened_for(Pose2D(0.0, 0.0, 0.0)) == base


def test_match_around_recovers_after_a_large_turn() -> None:
    truth = Pose2D(0.3, -0.2, 1.2)
    matcher = CorrelativeMatcher(room_map())
    guess = Pose2D(truth.x + 0.05, truth.y, truth.theta + math.radians(3.0))
    big_turn = Pose2D(0.0, 0.0, math.radians(80.0))  # the odometry step that led here
    result = matcher.match_around(guess, raycast_room(truth), big_turn, SearchWindow())
    assert result.pose.theta == pytest.approx(truth.theta, abs=math.radians(0.51))
