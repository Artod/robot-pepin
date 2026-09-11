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
    before = loc.pose  # a good scan may have nudged the pose a fraction of a cell off the truth
    est = loc.update(Pose2D(0.1, 0.0, 0.0), garbage)
    assert est.x == pytest.approx(before.x + 0.1, abs=1e-9) and loc.confidence == 0.0


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


def test_a_symmetric_room_refuses_the_global_fix_without_a_prior() -> None:
    """The empty rectangle fits the scan equally well turned by 180 degrees: say so, hold."""
    truth = Pose2D(1.2, -0.8, math.radians(60.0))
    loc = Localizer(room_map(), Pose2D())
    _, confidence = loc.global_search(raycast_room(truth))
    assert confidence == 0.0


def test_a_symmetric_room_picks_the_twin_nearer_the_start_pose() -> None:
    """With a start pose to go by, the twin nearer to it wins instead of a refusal."""
    from pepin.scanmatch import SearchWindow

    truth = Pose2D(1.2, -0.8, math.radians(60.0))
    start = Pose2D(1.0, -0.5, math.radians(40.0))  # the operator's rough guess, near the truth
    loc = Localizer(room_map(), start)
    confidence = loc.initialize(raycast_room(truth), SearchWindow(0.6, 0.06, 40.0, 4.0))
    assert confidence > 0.5 and not loc.lost
    assert math.hypot(loc.pose.x - truth.x, loc.pose.y - truth.y) < 0.15


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


def test_a_robot_pushed_by_hand_while_lost_relocalises_from_the_whole_map() -> None:
    """Odometry saw nothing; the local recovery window cannot reach; the global search can."""

    truth = Pose2D(-1.0, 0.5, 0.3)
    loc = Localizer(furnished_room_map(), truth)
    odom = Pose2D(0.0, 0.0, 0.0)
    for _ in range(3):
        loc.update(odom, raycast_room(truth, pillar=PILLAR))
    assert loc.confidence > 0.6
    pushed = Pose2D(1.3, -0.9, math.radians(140.0))  # carried across the room, odometry frozen
    for _ in range(40):
        loc.update(odom, raycast_room(pushed, pillar=PILLAR))
    assert not loc.lost
    assert math.hypot(loc.pose.x - pushed.x, loc.pose.y - pushed.y) < 0.10
    assert abs(loc.pose.theta - pushed.theta) < math.radians(5.0)


def test_adopting_a_pose_clears_what_the_old_belief_implied() -> None:
    """A re-seed that kept the drift kept searching as if the robot were still lost."""
    loc = Localizer(room_map(), Pose2D(), lost_after=2)
    garbage = np.array([[0.3, 0.3], [0.4, -0.2], [-0.3, 0.1]])
    loc.update(Pose2D(), garbage)
    loc.update(Pose2D(0.8, 0.0, math.radians(90.0)), garbage)  # motion while blind: drift grows
    loc.update(Pose2D(1.6, 0.0, math.radians(180.0)), garbage)
    assert loc.lost and loc._recovery_window().xy_m > loc._recovery.xy_m
    loc.adopt(Pose2D(1.0, 2.0, 0.5), 0.8)
    assert loc.pose == Pose2D(1.0, 2.0, 0.5) and loc.confidence == 0.8
    assert not loc.lost and loc._recovery_window().xy_m == loc._recovery.xy_m


def test_a_stamp_that_does_not_advance_holds_the_pose_and_no_stamp_uses_the_per_match_law() -> None:
    """At rest the lock's gain is a time constant: a scan of the same moment as the previous
    one (dt 0) moves the pose by nothing at all, a match two seconds later takes
    1 - exp(-2 / 6) of the residual, and a caller that times nothing (dt None) gets the
    per-match rest_gain, whatever it was given before."""
    loc = Localizer(room_map(), Pose2D(), rest_tau_s=6.0, rest_gain=0.05)
    loc.update(Pose2D(), raycast_room(Pose2D()))
    nudged = raycast_room(Pose2D(0.03, 0.0, 0.0))  # the scan says 3 cm ahead; the wheels, nothing
    before = loc.pose
    held = loc.update(Pose2D(), nudged, at_rest=True, dt_s=0.0)
    assert (held.x, held.y, held.theta) == pytest.approx((before.x, before.y, before.theta))
    same_moment = loc.report()
    assert same_moment.rest_locked == 1 and same_moment.rest_gain.hi == 0.0
    loc.update(Pose2D(), nudged, at_rest=True, dt_s=2.0)
    timed = loc.report()
    assert timed.rest_gain.hi == pytest.approx(1.0 - math.exp(-2.0 / 6.0))
    assert timed.rest_dt_s.hi == 2.0
    loc.update(Pose2D(), nudged, at_rest=True, dt_s=None)
    untimed = loc.report()
    assert untimed.rest_gain.hi == pytest.approx(0.05) and untimed.rest_dt_s.n == 0
    assert loc.pose.x > before.x  # the two timed matches did move it toward the scan


def test_two_agreeing_rest_residuals_beyond_the_carry_thresholds_are_taken_whole() -> None:
    """A standing cart lifted 10 cm: the first rest match beyond carry_m is blended like noise
    and remembered as a hint; the next one agreeing with it, the whole scan fitting the map
    better there (carry_gain past CARRY_MIN_GAIN), is a carry, taken whole. A residual inside
    the thresholds is never a carry however often it repeats; two beyond them that disagree (a
    wandering match) are not one; and an agreeing pair that gains the map nothing (a person
    by the lidar, a chair pushed against the cart) is not one either."""
    from pepin.localization import CARRY_MIN_GAIN

    grid = room_map()
    loc = Localizer(grid, Pose2D(), carry_m=0.06, carry_deg=4.0, rest_tau_s=6.0)
    here, lifted = Pose2D(), Pose2D(0.10, 0.0, 0.0)
    first, gain = loc._blend(here, lifted, at_rest=True, dt_s=1.0, carry_gain=0.2)
    assert gain == pytest.approx(1.0 - math.exp(-1.0 / 6.0)) and first.x < 0.02
    again = Pose2D(0.11, 0.01, 0.0)
    carried, gain = loc._blend(here, again, at_rest=True, dt_s=1.0, carry_gain=0.2)
    assert gain == 1.0 and (carried.x, carried.y) == pytest.approx((0.11, 0.01))
    assert loc.stats.carries == 1
    loc = Localizer(grid, Pose2D(), carry_m=0.06, carry_deg=4.0)
    for _ in range(3):
        _, gain = loc._blend(here, Pose2D(0.03, 0.0, 0.0), at_rest=True, dt_s=1.0, carry_gain=0.2)
        assert gain < 1.0
    loc._blend(here, lifted, at_rest=True, dt_s=1.0, carry_gain=0.2)
    _, gain = loc._blend(here, Pose2D(-0.10, 0.0, 0.0), at_rest=True, dt_s=1.0, carry_gain=0.2)
    assert gain < 1.0 and loc.stats.carries == 0
    loc = Localizer(grid, Pose2D(), carry_m=0.06, carry_deg=4.0)
    loc._blend(here, lifted, at_rest=True, dt_s=1.0, carry_gain=CARRY_MIN_GAIN / 2)
    _, gain = loc._blend(here, again, at_rest=True, dt_s=1.0, carry_gain=CARRY_MIN_GAIN / 2)
    assert gain < 1.0 and loc.stats.carries == 0, "agreeing twice is not enough: the map must fit"
