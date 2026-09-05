"""Localisation on a frozen synthetic map with drifting odometry."""

import math
from itertools import pairwise

import numpy as np
import pytest
from synthetic import raycast_room

from pepin.localization import Localizer
from pepin.mapping import GridSpec, OccupancyGrid
from pepin.odometry import Pose2D
from pepin.scanmatch import apply_motion, relative_motion

SPEC = GridSpec(0.05, -4, -3, 8, 6)
MAPPING_POSES = (Pose2D(0, 0, 0), Pose2D(1, 0.5, 0.7), Pose2D(-1, -0.5, -2.0), Pose2D(0.5, -1, 2.5))
PILLAR = (-2.0, 1.0, -1.6, 1.4)  # a box in one corner: the furnished room has no 180-degree twin


def room_map() -> OccupancyGrid:
    """The empty rectangle: identical to itself turned by 180 degrees."""
    grid = OccupancyGrid(SPEC)
    for pose in MAPPING_POSES:
        grid.integrate(pose, raycast_room(pose))
    return grid


def furnished_room_map() -> OccupancyGrid:
    """The rectangle with a box in one corner, so a scan fits exactly one place."""
    grid = OccupancyGrid(SPEC)
    for pose in MAPPING_POSES:
        grid.integrate(pose, raycast_room(pose, pillar=PILLAR))
    return grid


def test_tracks_the_truth_while_odometry_over_counts_turns() -> None:
    truth = [Pose2D(0.0, 0.0, 0.0)]
    for i in range(1, 25):
        truth.append(Pose2D(0.08 * i, 0.03 * i, 0.06 * i))
    odom = [truth[0]]
    for a, b in pairwise(truth):
        m = relative_motion(a, b)
        odom.append(apply_motion(odom[-1], Pose2D(m.x, m.y, m.theta * 1.15)))  # 15% turn over-count
    loc = Localizer(room_map(), truth[0])
    for o, t in zip(odom, truth, strict=True):
        est = loc.update(o, raycast_room(t))
    assert est.x == pytest.approx(truth[-1].x, abs=0.05)
    assert est.y == pytest.approx(truth[-1].y, abs=0.05)
    assert abs(est.theta - truth[-1].theta) < math.radians(2.0)
    assert (
        math.hypot(odom[-1].x - truth[-1].x, odom[-1].y - truth[-1].y) > 0.1
    )  # odometry alone drifted
    assert loc.confidence > 0.6


def test_degenerate_scan_leaves_the_odometry_prediction_in_place() -> None:
    loc = Localizer(room_map(), Pose2D())
    loc.update(Pose2D(), raycast_room(Pose2D()))
    garbage = np.array([[0.3, 0.3], [0.4, -0.2], [-0.3, 0.1]])  # three points cannot fix a pose
    est = loc.update(Pose2D(0.1, 0.0, 0.0), garbage)
    assert est.x == pytest.approx(0.1, abs=1e-9) and loc.confidence == 0.0


def test_grid_round_trips_through_npz(tmp_path) -> None:
    grid = room_map()
    grid.save(tmp_path / "m.npz")
    back = OccupancyGrid.load(tmp_path / "m.npz")
    assert back.spec == grid.spec
    assert np.array_equal(back.log_odds, grid.log_odds)
    assert len(back.occupied_xy()) > 100


def test_recovers_after_a_large_unmodelled_jump() -> None:
    loc = Localizer(room_map(), Pose2D(), lost_after=2)
    truth = Pose2D(0.0, 0.0, 0.0)
    loc.update(truth, raycast_room(truth))
    # The robot is kicked 0.3 m / 12 deg without the wheels noticing (odometry unchanged).
    kicked = Pose2D(0.3, -0.2, math.radians(12.0))
    garbage = np.array([[0.3, 0.3], [0.4, -0.2], [-0.3, 0.1]])  # no walls: confidence collapses
    for _ in range(3):
        loc.update(truth, garbage)
    assert loc.lost
    est = None
    for _ in range(3):
        est = loc.update(truth, raycast_room(kicked))
    assert est is not None and not loc.lost
    assert est.x == pytest.approx(kicked.x, abs=0.05) and est.y == pytest.approx(kicked.y, abs=0.05)
    assert abs(est.theta - kicked.theta) < math.radians(1.5)


def test_recovery_window_grows_with_drift_and_resets_on_acceptance() -> None:
    loc = Localizer(room_map(), Pose2D(), lost_after=1)
    small = loc._recovery_window()
    garbage = np.array([[0.3, 0.3], [0.4, -0.2]])
    loc.update(Pose2D(), garbage)  # weak scan: lost from now on
    loc.update(Pose2D(0.5, 0.0, math.radians(40.0)), garbage)  # big move while lost, no fix
    grown = loc._recovery_window()
    assert grown.theta_deg > small.theta_deg and grown.xy_m > small.xy_m
    assert grown.theta_deg / grown.theta_step_deg == pytest.approx(
        small.theta_deg / small.theta_step_deg
    )
    loc.update(Pose2D(0.5, 0.0, math.radians(40.0)), raycast_room(loc.pose))
    assert loc._recovery_window() == small  # accepted match resets the uncertainty


def test_predict_moves_the_pose_by_odometry_between_scans() -> None:
    loc = Localizer(room_map(), Pose2D())
    loc.update(Pose2D(), raycast_room(Pose2D()))
    est = loc.predict(Pose2D(0.2, 0.0, 0.0))
    assert est.x == pytest.approx(0.2, abs=0.05)
    est = loc.update(Pose2D(0.4, 0.0, 0.0), raycast_room(Pose2D(0.4, 0.0, 0.0)))
    assert est.x == pytest.approx(0.4, abs=0.05)


def test_initial_fix_recovers_a_hand_placed_start_offset() -> None:
    """Placed 30 cm / 25 deg off the mark: the wide first search finds the truth before moving."""
    from pepin.scanmatch import SearchWindow

    truth = Pose2D(0.30, -0.20, math.radians(25.0))
    loc = Localizer(room_map(), Pose2D(0.0, 0.0, 0.0))
    confidence = loc.initialize(raycast_room(truth), SearchWindow(0.6, 0.06, 40.0, 4.0))
    assert confidence >= 0.6
    assert math.hypot(loc.pose.x - truth.x, loc.pose.y - truth.y) < 0.06
    assert abs(loc.pose.theta - truth.theta) < math.radians(3.0)


def test_a_robot_put_down_anywhere_is_found_by_the_global_search() -> None:
    """1.2 m and 60 deg off the assumed start: beyond the wide window, the whole map is searched."""
    from pepin.scanmatch import SearchWindow

    truth = Pose2D(1.2, -0.8, math.radians(60.0))
    loc = Localizer(furnished_room_map(), Pose2D(0.0, 0.0, 0.0))
    scan = raycast_room(truth, pillar=PILLAR)
    confidence = loc.initialize(scan, SearchWindow(0.6, 0.06, 40.0, 4.0))
    assert confidence >= 0.6 and not loc.lost
    assert math.hypot(loc.pose.x - truth.x, loc.pose.y - truth.y) < 0.08
    assert abs(loc.pose.theta - truth.theta) < math.radians(4.0)


def test_a_symmetric_room_refuses_the_global_fix_instead_of_guessing() -> None:
    """The empty rectangle fits the scan equally well turned by 180 degrees: say so, hold."""
    from pepin.scanmatch import SearchWindow

    truth = Pose2D(1.2, -0.8, math.radians(60.0))
    start = Pose2D(0.0, 0.0, 0.0)
    loc = Localizer(room_map(), start)
    confidence = loc.initialize(raycast_room(truth), SearchWindow(0.6, 0.06, 40.0, 4.0))
    assert confidence == 0.0 and loc.pose == start and loc.lost


def test_the_matcher_follows_a_map_that_keeps_growing() -> None:
    """Online mapping: scans integrated after the matcher was built change its scores."""
    from pepin.scanmatch import CorrelativeMatcher

    grid = room_map()
    matcher = CorrelativeMatcher(grid)
    pose = Pose2D(-1.0, 0.0, 0.0)
    with_wall = raycast_room(pose, pillar=(0.0, -2.0, 0.2, 2.0))  # a wall across the room
    before = matcher.inlier_fraction(pose, with_wall)
    for _ in range(6):
        grid.integrate(pose, with_wall)
    assert grid.version == 4 + 6
    assert matcher.inlier_fraction(pose, with_wall) > before + 0.2
