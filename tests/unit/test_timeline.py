"""Scans and odometry meet at the same instant, beam by beam."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pepin.mapping import grid_from_pgm
from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import CorrelativeMatcher, SearchWindow
from pepin.timeline import (
    MotionFilter,
    OdomHistory,
    ScanGate,
    TimedScan,
    beam_times,
    deskew,
    standing_still,
    timed_scan_from_ros,
)

REPO = Path(__file__).resolve().parents[2]


def history_of(samples: list[tuple[float, float, float, float]]) -> OdomHistory:
    h = OdomHistory()
    for t, x, y, yaw in samples:
        h.add(t, Pose2D(x, y, yaw))
    return h


# -- the history --------------------------------------------------------------


def test_the_history_interpolates_between_samples_and_never_outside() -> None:
    h = history_of([(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 1.0)])
    mid = h.at(0.25)
    assert mid is not None
    assert (mid.x, mid.y, mid.theta) == pytest.approx((0.25, 0.0, 0.25))
    assert h.at(-0.01) is None and h.at(1.01) is None
    assert h.at(1.0) is not None and h.at(0.0) is not None


def test_the_heading_is_interpolated_the_short_way_across_pi() -> None:
    h = history_of([(0.0, 0.0, 0.0, math.pi - 0.1), (1.0, 0.0, 0.0, -math.pi + 0.1)])
    mid = h.at(0.5)
    assert mid is not None
    assert abs(abs(mid.theta) - math.pi) < 1e-9  # halfway through the crossing, not through 0


def test_old_samples_fall_off_and_late_ones_are_ignored() -> None:
    h = OdomHistory(horizon_s=1.0)
    for t in (0.0, 0.5, 1.0, 2.0):
        h.add(t, Pose2D(t, 0.0, 0.0))
    assert h.oldest_t == 1.0 and h.newest_t == 2.0
    h.add(1.5, Pose2D(9.0, 9.0, 0.0))  # arrived late: cannot be inserted behind the newest
    assert len(h) == 2 and h.at(1.5) is not None and h.at(1.5).x == pytest.approx(1.5)


# -- deskew -------------------------------------------------------------------


def skewed_wall(
    omega: float, v: float = 0.0, n: int = 90, period: float = 0.1
) -> tuple[np.ndarray, np.ndarray, OdomHistory]:
    """A wall at x = 2 m seen by a robot turning at ``omega`` (driving at ``v``) for one revolution.

    Beam i is taken at t_i = stamp - i/(n-1) * period; the robot's pose then is (v t, 0, omega t).
    Each beam is what the wall looks like from that pose, in the base frame of that moment.
    """
    stamp = 1.0
    times = beam_times(stamp, n, period)
    bearings = np.linspace(-0.6, 0.6, n)  # beam directions in the base frame
    pts = []
    for t, b in zip(times, bearings, strict=True):
        yaw = omega * t
        world_dir = b + yaw
        x0 = v * t
        r = (2.0 - x0) / math.cos(world_dir)  # distance along the beam to x = 2
        pts.append((r * math.cos(b), r * math.sin(b)))
    h = OdomHistory()
    for t in np.linspace(stamp - 0.3, stamp + 0.1, 9):  # 20 Hz samples around the revolution
        h.add(float(t), Pose2D(v * t, 0.0, omega * t))
    return np.array(pts), times, h


def wall_flatness(points: np.ndarray, pose: Pose2D) -> float:
    """Spread of the points' world x around 2 m once placed at ``pose``: 0 for a straight wall."""
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    wx = pose.x + c * points[:, 0] - s * points[:, 1]
    return float(np.abs(wx - 2.0).max())


def test_deskew_straightens_a_wall_scanned_during_a_pivot() -> None:
    omega = 1.0  # rad/s: 5.7 degrees across one revolution
    points, times, h = skewed_wall(omega)
    ref = h.at(1.0)
    assert ref is not None
    assert wall_flatness(points, ref) > 0.05  # the raw revolution is visibly bent
    fixed = deskew(points, times, h, 1.0)
    assert fixed is not None
    assert wall_flatness(fixed, ref) < 0.002


def test_deskew_also_undoes_the_travel_during_the_revolution() -> None:
    points, times, h = skewed_wall(omega=0.8, v=0.3)
    fixed = deskew(points, times, h, 1.0)
    ref = h.at(1.0)
    assert fixed is not None and ref is not None
    assert wall_flatness(fixed, ref) < 0.002


def test_deskew_is_the_identity_standing_still_and_refuses_uncovered_times() -> None:
    points, times, h = skewed_wall(omega=0.0)
    fixed = deskew(points, times, h, 1.0)
    assert fixed is not None and np.allclose(fixed, points)
    short = history_of([(0.95, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)])  # starts after the first beam
    assert deskew(points, times, short, 1.0) is None


def test_the_ros_scan_conversion_keeps_a_time_per_valid_beam() -> None:
    ranges = [1.0, float("nan"), 0.01, 2.0, 30.0]
    scan = timed_scan_from_ros(5.0, ranges, 0.0, 0.1, 12.0, 0.1, (0.0, 0.0, 0.0, False), 7)
    assert len(scan.points) == 2 and len(scan.times) == 2
    assert scan.times[0] == pytest.approx(5.0) and scan.times[1] == pytest.approx(5.0 - 3 / 4 * 0.1)
    assert np.isnan(scan.ranges[[1, 2, 4]]).all() and scan.ranges[3] == 2.0
    assert scan.first_t == pytest.approx(5.0 - 0.075) and scan.scan_id == 7


# -- the gate -----------------------------------------------------------------


def scan_at(stamp: float, scan_id: int = 1) -> TimedScan:
    times = beam_times(stamp, 3, 0.1)
    return TimedScan(stamp, np.zeros((3, 2)), times, np.ones(3), scan_id)


def test_a_scan_waits_for_odometry_that_covers_its_whole_revolution() -> None:
    gate = ScanGate()
    h = history_of([(0.80, 0.0, 0.0, 0.0), (0.95, 0.0, 0.0, 0.0)])  # the filter runs 50 ms behind
    gate.offer(scan_at(1.0))
    assert gate.take(h, now=1.01) is None  # no pose yet for the last 50 ms of the revolution
    h.add(1.00, Pose2D())
    released = gate.take(h, now=1.06)
    assert released is not None and released.stamp == 1.0
    assert gate.take(h, now=1.07) is None  # released once
    stats = gate.report()
    assert (stats.offered, stats.released, stats.replaced, stats.expired) == (1, 1, 0, 0)
    assert stats.waited_s == [pytest.approx(0.06)]


def test_a_newer_scan_replaces_a_waiting_one_and_a_stale_one_expires() -> None:
    gate = ScanGate(max_wait_s=0.5)
    h = history_of([(0.0, 0.0, 0.0, 0.0), (0.5, 0.0, 0.0, 0.0)])
    gate.offer(scan_at(1.0, scan_id=1))
    gate.offer(scan_at(1.1, scan_id=2))
    assert gate.pending is not None and gate.pending.scan_id == 2
    assert gate.take(h, now=1.2) is None
    assert gate.take(h, now=1.7) is None  # 0.6 s with no odometry: dropped
    assert gate.pending is None
    stats = gate.report()
    assert (stats.offered, stats.released, stats.replaced, stats.expired) == (2, 0, 1, 1)


def test_the_gate_never_releases_a_scan_against_another_moment() -> None:
    """The property the tracker was missing: with odometry 60 ms late on every message, each
    scan is released only once its own moment is covered, and the pose read for it is the pose
    of that moment, not the newest one."""
    gate, h = ScanGate(), OdomHistory()
    released = []
    stamps = np.arange(1.02, 3.0, 0.1)  # never on an odometry sample: real clocks do not coincide
    odom_t = np.arange(0.0, 3.5, 0.05)
    i = 0
    for now in np.arange(1.0, 3.6, 0.01):
        while i < len(odom_t) and odom_t[i] + 0.06 <= now:  # every sample arrives 60 ms late
            h.add(float(odom_t[i]), Pose2D(float(odom_t[i]), 0.0, 0.5 * float(odom_t[i])))
            i += 1
        for s in stamps:
            if abs(s - now) < 0.005:
                gate.offer(scan_at(float(s)))
        scan = gate.take(h, float(now))
        if scan is not None:
            pose = h.at(scan.stamp)
            assert pose is not None and pose.x == pytest.approx(scan.stamp)
            assert wrap_angle(pose.theta - 0.5 * scan.stamp) == pytest.approx(0.0)
            released.append(scan.stamp)
    assert len(released) == len(stamps) and gate.report().expired == 0


# -- the motion filter --------------------------------------------------------


def test_the_matcher_rests_while_the_cart_stands_and_wakes_on_any_step() -> None:
    f = MotionFilter(min_m=0.005, min_deg=0.3, max_gap_s=1.0)
    assert f.due(Pose2D(), 0.0)
    assert not f.due(Pose2D(0.001, 0.0, 0.0), 0.1)
    assert f.due(Pose2D(0.006, 0.0, 0.0), 0.2)  # 6 mm
    assert not f.due(Pose2D(0.006, 0.0, math.radians(0.2)), 0.3)
    assert f.due(Pose2D(0.006, 0.0, math.radians(0.5)), 0.4)  # half a degree
    assert not f.due(Pose2D(0.006, 0.0, math.radians(0.5)), 1.35)
    assert f.due(Pose2D(0.006, 0.0, math.radians(0.5)), 1.45)  # a second passed
    f.reset()
    assert f.due(Pose2D(0.006, 0.0, math.radians(0.5)), 1.46)


# -- rest ---------------------------------------------------------------------


def _resting_history(jitter_deg: float = 0.1) -> OdomHistory:
    """A second of a standing cart: the filter's own noise, nothing else."""
    return history_of(
        [(0.1 * i, 0.0005 * (i % 2), 0.0, math.radians(jitter_deg) * (i % 2)) for i in range(21)]
    )


def test_a_standing_cart_is_recognised_by_the_wheels_and_the_gyro_together() -> None:
    assert standing_still(_resting_history(), 2.0, yaw_rate=0.0)
    assert not standing_still(_resting_history(), 2.0, yaw_rate=math.radians(20.0)), (
        "the gyro alone must be able to say the cart turns"
    )


def test_a_turning_cart_is_never_called_at_rest() -> None:
    turning = history_of([(0.1 * i, 0.0, 0.0, math.radians(3.0 * i)) for i in range(21)])
    assert not standing_still(turning, 2.0, yaw_rate=math.radians(30.0))
    # Even with a gyro that says nothing (a dead sensor), the wheels refuse it.
    assert not standing_still(turning, 2.0, yaw_rate=0.0)


def test_rest_is_unknown_before_the_history_reaches_back_far_enough() -> None:
    assert not standing_still(history_of([(0.0, 0.0, 0.0, 0.0)]), 0.0, yaw_rate=0.0)


# -- the real beam order ------------------------------------------------------


@pytest.fixture(scope="module")
def flat():  # type: ignore[no-untyped-def]
    return grid_from_pgm(REPO / "ros" / "maps" / "flat3_straight.yaml")


def test_the_newest_beam_is_at_index_zero_on_real_pivots(flat) -> None:  # type: ignore[no-untyped-def]
    """Six scans of run 0083 taken while turning fastest: deskewed in the driver's order they fit
    the map better than deskewed in the opposite order. Flip ``beam_times`` and this goes red."""
    fixture = json.loads((REPO / "tests/fixtures/pivot_scans_run0083.json").read_text())
    matcher = CorrelativeMatcher(flat, max_points=120)
    window = SearchWindow(xy_m=0.09, xy_step_m=0.03, theta_deg=9.0, theta_step_deg=1.5)
    forward, backward = [], []
    for case in fixture["scans"]:
        h = history_of([(o["t"], o["x"], o["y"], o["theta"]) for o in case["odom"]])
        scan = case["scan"]
        ranges = np.array([np.nan if r is None else r for r in scan["ranges"]], dtype=float)
        angles = np.array(scan["angles"], dtype=float)
        ok = np.isfinite(ranges)
        points = np.column_stack((ranges[ok] * np.cos(angles[ok]), ranges[ok] * np.sin(angles[ok])))
        times = beam_times(scan["t"], len(ranges), 1.0 / scan["speed_rps"])
        guess = Pose2D(case["pose"]["x"], case["pose"]["y"], case["pose"]["theta"])
        for order, sink in ((times[ok], forward), (times[ok][::-1], backward)):
            fixed = deskew(points, order, h, scan["t"])
            assert fixed is not None
            best = matcher.match(guess, fixed, window).pose
            sink.append(matcher.inlier_fraction(best, fixed))
    assert np.mean(forward) > np.mean(backward) + 0.03, (forward, backward)
