"""The tracker's heading must not wander while nothing moves.

Three switches, one subject: the matcher answering between its candidates, the rest lock that
averages instead of re-deciding while the cart stands, and the silencing of returns the static
map cannot explain. Each is checked on and off, because each must be comparable on the robot.
"""

import math

import numpy as np
from synthetic import raycast_room
from test_localization import PILLAR, furnished_room_map

from pepin.dynamic import StaticMask, voting_mask
from pepin.localization import Localizer
from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import SearchWindow

WINDOW = SearchWindow(xy_m=0.09, xy_step_m=0.03, theta_deg=9.0, theta_step_deg=1.5)
TRUTH = Pose2D(0.15, -0.35, math.radians(12.0))
BEAMS = 90  # a thinner scan than the robot's: the same physics, milliseconds instead of seconds


def tracker(**overrides: object) -> Localizer:
    """A localizer wired like the node's: the board's window, its gain, no global retry."""
    settings: dict[str, object] = {
        "window": WINDOW,
        "max_points": BEAMS,
        "correction_gain": 0.5,
        "lost_after": 1_000_000,
        "global_retry": False,
    }
    settings.update(overrides)
    return Localizer(furnished_room_map(), TRUTH, **settings)  # type: ignore[arg-type]


def nudged(offset_deg: float, **overrides: object) -> Localizer:
    """The node's tracker believing it stands ``offset_deg`` off the heading it really has."""
    loc = tracker(**overrides)
    loc.pose = Pose2D(TRUTH.x, TRUTH.y, TRUTH.theta + math.radians(offset_deg))
    return loc


def standing_error_deg(loc: Localizer, matches: int, dt_s: float | None) -> float:
    """Heading error left after ``matches`` matches of the same standing robot, in degrees."""
    points = raycast_room(TRUTH, beams=BEAMS, pillar=PILLAR)
    for _ in range(matches):
        loc.update(TRUTH, points, at_rest=True, dt_s=dt_s)
    return abs(math.degrees(wrap_angle(loc.pose.theta - TRUTH.theta)))


def noisy_scans(n: int, sigma_m: float = 0.02, seed: int = 7) -> list[np.ndarray]:
    """``n`` scans of the same standing robot, each with the lidar's own range noise on it."""
    rng = np.random.default_rng(seed)
    clean = raycast_room(TRUTH, beams=BEAMS, pillar=PILLAR)
    out = []
    for _ in range(n):
        r = np.hypot(clean[:, 0], clean[:, 1])
        scaled = (r + rng.normal(0.0, sigma_m, len(r))) / r
        out.append(clean * scaled[:, None])
    return out


def yaw_band_deg(loc: Localizer, scans: list[np.ndarray], at_rest: bool) -> float:
    """Max minus min of the tracked heading over ``scans``, in degrees, at a standing pose."""
    yaws = []
    for points in scans:
        loc.update(TRUTH, points, at_rest=at_rest)
        yaws.append(math.degrees(loc.pose.theta))
    return max(yaws) - min(yaws)


# -- the rest lock ------------------------------------------------------------


def test_the_rest_lock_narrows_the_heading_band_of_a_standing_cart() -> None:
    scans = noisy_scans(24)
    free = yaw_band_deg(tracker(), scans, at_rest=False)
    locked = yaw_band_deg(tracker(), scans, at_rest=True)
    assert locked < free / 2.0, f"band {locked:.2f} deg locked vs {free:.2f} deg free"


def test_the_switch_off_restores_the_per_scan_decision() -> None:
    scans = noisy_scans(24)
    free = yaw_band_deg(tracker(), scans, at_rest=False)
    off = yaw_band_deg(tracker(rest_lock=False), scans, at_rest=True)
    assert off == free


def test_the_rest_lock_settles_in_seconds_not_in_matches() -> None:
    """The same residual, the same three seconds of standing, two match cadences: the node
    matches a standing cart about once a second, a replay feeds every scan, and a per-match gain
    would make the two disagree tenfold (a 20 s time constant on the robot, 2 s offline)."""
    start_deg = 4.0  # a nudge inside the search window, well under the jump threshold
    once_a_second = standing_error_deg(nudged(start_deg, rest_tau_s=1.0), 3, dt_s=1.0)
    ten_a_second = standing_error_deg(nudged(start_deg, rest_tau_s=1.0), 30, dt_s=0.1)
    assert abs(once_a_second - ten_a_second) < 0.05, (
        f"the cadence changed the answer: {once_a_second:.3f} vs {ten_a_second:.3f} deg"
    )
    # Three time constants: a twentieth of the nudge is left, whatever the cadence.
    assert max(once_a_second, ten_a_second) < 0.5


def test_without_dt_the_rest_lock_is_the_old_per_match_gain() -> None:
    """Callers that time nothing keep the numbers they had: 0.05 of the residual per match, and
    ``rest_tau_s`` is not consulted at all."""
    start_deg = 4.0
    untimed = standing_error_deg(nudged(start_deg), 12, dt_s=None)
    other_tau = standing_error_deg(nudged(start_deg, rest_tau_s=0.01), 12, dt_s=None)
    assert untimed == other_tau, "dt_s=None must not look at the time constant"
    assert abs(untimed - start_deg * 0.95**12) < 0.1, f"not the old law: {untimed:.3f} deg left"


def test_a_lost_tracker_drops_the_lock_and_takes_what_the_recovery_found() -> None:
    """The cart was slid aside while it stood: the fit collapses, the wide search finds it, and
    the lock must not make the fix crawl in — neither at a twentieth per match nor at a gain
    taken from the clock."""
    moved = Pose2D(TRUTH.x + 0.22, TRUTH.y, TRUTH.theta)
    scan = raycast_room(moved, beams=BEAMS, pillar=PILLAR)
    for dt_s, locked_gain in ((None, 0.05), (1.0, 1.0 - math.exp(-1.0 / 3.0))):
        loc = tracker(
            lost_after=3, recovery=SearchWindow(0.3, 0.05, 6.0, 1.5), relocalise_min_inliers=0.4
        )
        loc.weak_scans = 5  # three weak scans in a row already happened: the tracker is lost
        loc.update(TRUTH, scan, at_rest=True, dt_s=dt_s)
        crawled = TRUTH.x + locked_gain * (moved.x - TRUTH.x)  # what the lock would have given
        assert loc.pose.x > crawled + 0.02, f"the lock swallowed the fix: x {loc.pose.x:.3f}"


# -- who votes ----------------------------------------------------------------


def moved_furniture(pull_m: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """A scan of the room with one wedge of beams stopping short, and the mask of that wedge.

    A sofa pulled out from the wall, a blanket over a chair: a run of returns that stand where
    the map has nothing. They still look like an obstacle to the matcher, which is the problem.
    """
    points = raycast_room(TRUTH, beams=BEAMS, pillar=PILLAR)
    r = np.hypot(points[:, 0], points[:, 1])
    wedge = np.zeros(len(points), dtype=bool)
    wedge[8:26] = True
    scaled = np.where(wedge, np.maximum(r - pull_m, 0.2) / r, 1.0)
    return points * scaled[:, None], wedge


def test_unexplained_returns_do_not_pull_the_pose() -> None:
    points, wedge = moved_furniture()
    mask = StaticMask(furnished_room_map())
    vote = voting_mask(points, TRUTH, mask, min_points=20)
    assert vote is not None and not vote[wedge].any(), "the wedge is what the map cannot explain"

    def error(vote: np.ndarray | None) -> float:
        loc = tracker(correction_gain=1.0)
        for _ in range(3):
            loc.update(TRUTH, points, vote=vote)
        return math.hypot(loc.pose.x - TRUTH.x, loc.pose.y - TRUTH.y) + abs(
            wrap_angle(loc.pose.theta - TRUTH.theta)
        )

    assert error(vote) < error(None)


def test_confidence_is_always_measured_on_the_whole_scan() -> None:
    """The fit the watch and the occlusion verdict are built on must not change with the mask."""
    points, _ = moved_furniture()
    mask = StaticMask(furnished_room_map())
    vote = voting_mask(points, TRUTH, mask, min_points=20)
    assert vote is not None
    loc = tracker(correction_gain=1.0)  # gain 1: the tracked pose IS the matched pose
    loc.update(TRUTH, points, vote=vote)
    matcher = loc._matcher
    assert loc.confidence == matcher.inlier_fraction(loc.pose, points)
    assert loc.confidence < matcher.inlier_fraction(loc.pose, points[vote]), (
        "the returns the map cannot explain must still count against the reported fit"
    )


def test_a_mask_that_would_starve_the_match_is_ignored() -> None:
    """Fewer returns left than a pose can be fixed from: the whole scan votes instead."""
    points = raycast_room(TRUTH, beams=BEAMS, pillar=PILLAR)
    nothing = np.zeros(len(points), dtype=bool)
    loc, plain = tracker(), tracker()
    loc.update(TRUTH, points, vote=nothing)
    plain.update(TRUTH, points)
    assert (loc.pose.x, loc.pose.y, loc.pose.theta) == (
        plain.pose.x,
        plain.pose.y,
        plain.pose.theta,
    )


# -- sub-cell refinement ------------------------------------------------------


def test_without_refinement_the_heading_correction_snaps_to_the_search_step() -> None:
    """The switch off is the lattice the tracker used to answer on: 1.5 degree steps."""
    points = raycast_room(TRUTH, beams=BEAMS, pillar=PILLAR)
    lattice = Localizer(
        furnished_room_map(), TRUTH, window=WINDOW, max_points=BEAMS, interpolate=False
    )
    off_lattice = 0
    for error_deg in (0.4, -0.4, 0.9, -0.9):
        loc = Localizer(
            furnished_room_map(),
            Pose2D(TRUTH.x, TRUTH.y, TRUTH.theta + math.radians(error_deg)),
            window=WINDOW,
            max_points=BEAMS,
        )
        lattice.pose = loc.pose
        for tracked, sink in ((loc, 1), (lattice, 0)):
            before = tracked.pose.theta
            tracked.update(TRUTH, points)
            step = math.degrees(wrap_angle(tracked.pose.theta - before))
            on_step = abs(step - 1.5 * round(step / 1.5)) < 0.05
            off_lattice += sink * (not on_step)
            assert sink or on_step, f"the lattice answered {step:.3f} deg, off its own step"
    assert off_lattice, "refinement never left the lattice"
