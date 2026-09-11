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


def yaw_band_deg(
    loc: Localizer, scans: list[np.ndarray], at_rest: bool, dt_s: float | None = 0.1
) -> float:
    """Max minus min of the tracked heading over ``scans``, in degrees, at a standing pose.

    ``dt_s`` is passed the way the node passes it (a scan every 0.1 s): the deployed law is the
    timed one, and a test of the untimed fallback would test a gain the robot never uses.
    """
    yaws = []
    for points in scans:
        loc.update(TRUTH, points, at_rest=at_rest, dt_s=dt_s)
        yaws.append(math.degrees(loc.pose.theta))
    return max(yaws) - min(yaws)


def heading_error_deg(loc: Localizer) -> float:
    """How far the tracked heading is from the truth, in degrees."""
    return abs(math.degrees(wrap_angle(loc.pose.theta - TRUTH.theta)))


def offset_m(loc: Localizer, truth: Pose2D = TRUTH) -> float:
    """How far the tracked position is from ``truth``, in metres."""
    return math.hypot(loc.pose.x - truth.x, loc.pose.y - truth.y)


# -- the rest lock ------------------------------------------------------------


def test_the_rest_lock_narrows_the_heading_band_of_a_standing_cart() -> None:
    scans = noisy_scans(24)
    free = yaw_band_deg(tracker(), scans, at_rest=False)
    locked = yaw_band_deg(tracker(), scans, at_rest=True)
    assert locked < free / 2.0, f"band {locked:.2f} deg locked vs {free:.2f} deg free"


def test_the_switch_is_the_trackers_own_and_flips_live() -> None:
    """The node's parameter callback writes ``rest_lock`` on the Localizer between two scans;
    a copy taken at construction would leave ``ros2 param set`` a no-op."""
    scans = noisy_scans(24)
    free = yaw_band_deg(tracker(), scans, at_rest=False)
    loc = tracker()
    loc.rest_lock = False
    assert yaw_band_deg(loc, scans, at_rest=True) == free
    loc.rest_lock = True
    assert yaw_band_deg(loc, scans, at_rest=True) < free / 2.0


def test_a_long_gap_does_not_hand_the_lock_one_noisy_match_whole() -> None:
    """After a whole-map search or an expired scan the first rest match comes 15 s after the
    previous one: ``1 - exp(-15 / 6)`` says take it nine tenths whole, which is one noisy match
    taken whole. The gain is capped at the driving gain, so that match is worth exactly what it
    would be worth while driving."""
    points = raycast_room(TRUTH, beams=BEAMS, pillar=PILLAR)
    locked, driving = nudged(4.0), nudged(4.0)
    locked.update(TRUTH, points, at_rest=True, dt_s=15.0)
    driving.update(TRUTH, points, at_rest=False)
    assert locked.pose == driving.pose
    assert heading_error_deg(locked) > 1.0, (
        "more than half of the nudge must remain after one match"
    )
    assert locked.stats.rest_gain.hi <= 0.5


def test_the_rest_lock_settles_in_seconds_not_in_matches() -> None:
    """The same residual, the same three seconds of standing, two match cadences: the node
    matches a standing cart about once a second, a replay feeds every scan, and a per-match gain
    would make the two disagree tenfold (a 20 s time constant on the robot, 2 s offline)."""
    # A nudge inside the search window, under the carry threshold (two matches agreeing on a
    # 4 degree residual would be taken whole, which is another test); the driving gain at 1 so
    # the cap on the rest gain (its own test too) does not bind at tau 1 s / dt 1 s.
    start_deg = 2.5
    once = nudged(start_deg, rest_tau_s=1.0, correction_gain=1.0)
    ten = nudged(start_deg, rest_tau_s=1.0, correction_gain=1.0)
    once_a_second = standing_error_deg(once, 3, dt_s=1.0)
    ten_a_second = standing_error_deg(ten, 30, dt_s=0.1)
    assert abs(once_a_second - ten_a_second) < 0.05, (
        f"the cadence changed the answer: {once_a_second:.3f} vs {ten_a_second:.3f} deg"
    )
    # Three time constants: a twentieth of the nudge is left, whatever the cadence.
    assert max(once_a_second, ten_a_second) < 0.3


def test_without_dt_the_rest_lock_is_the_old_per_match_gain() -> None:
    """Callers that time nothing keep the numbers they had: 0.05 of the residual per match, and
    ``rest_tau_s`` is not consulted at all."""
    start_deg = 2.5  # under the carry threshold: this is the lock's law, not the carry's
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


def test_the_scan_that_loses_the_tracker_is_not_blended_under_the_lock() -> None:
    """The lost verdict is counted before the blend reads it: the very scan whose poor fit makes
    the tracker lost is already taken at the driving gain, like the ones after it. (It used to
    read the verdict first and count second, so that scan crawled in at the rest gain.)"""
    moved = Pose2D(TRUTH.x + 0.5, TRUTH.y - 0.4, TRUTH.theta + math.radians(30.0))
    scan = raycast_room(moved, beams=BEAMS, pillar=PILLAR)
    resting, driving = tracker(lost_after=1), tracker(lost_after=1)
    resting.update(TRUTH, scan, at_rest=True, dt_s=1.0)
    driving.update(TRUTH, scan, at_rest=False)
    assert resting.lost and driving.lost, "a scan from 0.6 m away must fit the map poorly here"
    assert resting.pose == driving.pose
    assert resting.stats.rest_locked == 0 and resting.stats.weak == 1


# -- a carry at rest ----------------------------------------------------------


def carried_to(dx: float = 0.0, dy: float = 0.0, dtheta_deg: float = 0.0) -> Pose2D:
    """Where the cart really stands after being lifted or skidded, no wheel tick, no gyro."""
    return Pose2D(TRUTH.x + dx, TRUTH.y + dy, TRUTH.theta + math.radians(dtheta_deg))


def test_two_rest_matches_agreeing_on_a_new_pose_are_a_carry_taken_whole() -> None:
    """A sideways skid of 7 cm at rest: the wheels and the gyro see nothing, the lock would
    absorb it with its time constant while the stop reflex ran on the old pose. The first match
    beyond the carry thresholds is noise until the next one agrees; then the pose jumps."""
    for truth in (carried_to(dy=0.07), carried_to(dtheta_deg=5.0)):
        scan = raycast_room(truth, beams=BEAMS, pillar=PILLAR)
        loc = tracker()
        loc.update(TRUTH, scan, at_rest=True, dt_s=1.0)
        after_one = offset_m(loc, truth) + abs(wrap_angle(loc.pose.theta - truth.theta))
        assert loc.stats.carries == 0, "one match beyond the threshold is not yet a carry"
        loc.update(TRUTH, scan, at_rest=True, dt_s=1.0)
        assert loc.stats.carries == 1
        # What is left is one match's noise on 90 beams (3 cm / 1.5 deg lattice), not the carry.
        assert offset_m(loc, truth) < 0.04 and after_one > 0.04, (
            f"the carry was not taken whole: {offset_m(loc, truth):.3f} m off after two matches"
        )
        assert abs(math.degrees(wrap_angle(loc.pose.theta - truth.theta))) < 1.0


def test_a_single_wild_match_at_rest_is_noise_and_the_lock_holds() -> None:
    """One match 7 cm off, the next one back on the wall: the pose stays where the cart is."""
    wild = raycast_room(carried_to(dy=0.07), beams=BEAMS, pillar=PILLAR)
    honest = raycast_room(TRUTH, beams=BEAMS, pillar=PILLAR)
    loc = tracker()
    for scan in (wild, honest, wild, honest):
        loc.update(TRUTH, scan, at_rest=True, dt_s=1.0)
    assert loc.stats.carries == 0
    assert offset_m(loc) < 0.04, f"a wild match moved the standing cart {offset_m(loc):.3f} m"


def test_a_carry_is_only_ever_read_at_rest() -> None:
    """While driving the wheels explain the residual: the driving gain and the jump rule apply,
    and a match beyond the carry thresholds leaves no hint behind for the next rest match."""
    scan = raycast_room(carried_to(dy=0.07), beams=BEAMS, pillar=PILLAR)
    loc = tracker()
    loc.update(TRUTH, scan, at_rest=False)
    loc.update(TRUTH, scan, at_rest=True, dt_s=1.0)
    assert loc.stats.carries == 0


def test_the_fit_is_reported_at_the_match_and_at_the_published_pose() -> None:
    """The watch keeps the fit at the matched pose (its verdicts do not move); the published
    pose gets its own, and the two part ways exactly while a carry is being absorbed."""
    scan = raycast_room(carried_to(dy=0.07), beams=BEAMS, pillar=PILLAR)
    loc = tracker()
    loc.update(TRUTH, scan, at_rest=True, dt_s=1.0)
    matcher = loc._matcher
    assert loc.published_fit == matcher.inlier_fraction(loc.pose, scan)
    assert loc.confidence > loc.published_fit + 0.1, "the published pose is 6 cm off the wall"
    gain_one = tracker(correction_gain=1.0)
    gain_one.update(TRUTH, scan, at_rest=False)
    assert gain_one.published_fit == gain_one.confidence


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
    assert loc.stats.silenced_scans == 1 and loc.stats.silenced_points == 0


def test_the_tracker_reads_the_static_mask_at_its_own_prediction_and_the_switch_is_live() -> None:
    """The node hands over the StaticMask and nothing else; the tracker reads it where it is
    about to match from, and ``explained_vote`` decides live whether it reads it at all."""
    points, wedge = moved_furniture()
    mask = StaticMask(furnished_room_map())
    with_mask, explicit, plain = tracker(), tracker(), tracker()
    with_mask.update(TRUTH, points, mask=mask)
    explicit.update(TRUTH, points, vote=voting_mask(points, TRUTH, mask))
    plain.update(TRUTH, points)
    assert with_mask.pose == explicit.pose != plain.pose
    assert with_mask.stats.silenced_points == int(wedge.sum())
    off = tracker()
    off.explained_vote = False
    off.update(TRUTH, points, mask=mask)
    assert off.pose == plain.pose and off.stats.silenced_scans == 0


def grazing_wall_scans(
    n: int, sigma_m: float = 0.02, seed: int = 3
) -> tuple[Pose2D, list[np.ndarray]]:
    """The cart 30 cm from the long wall, looking along it: ``n`` noisy scans and the truth.

    Along a wall seen at a grazing angle a few degrees of heading error move the far returns off
    the mapped wall by tens of centimetres (4 deg at 3 m is 21 cm, past the mask's 15 cm), so a
    mask read at the wrong prediction silences exactly the returns that measure the heading.
    """
    truth = Pose2D(-1.0, -1.7, 0.0)
    rng = np.random.default_rng(seed)
    clean = raycast_room(truth, beams=BEAMS, pillar=PILLAR)
    r = np.hypot(clean[:, 0], clean[:, 1])
    return truth, [clean * ((r + rng.normal(0.0, sigma_m, len(r))) / r)[:, None] for _ in range(n)]


def test_the_vote_mask_does_not_lock_in_a_heading_error_along_a_grazing_wall() -> None:
    """A 4 degree error beside a long wall with a noisy scan: the mask, read at the wrong
    prediction, must not silence the evidence that fixes it — after a few matches the heading
    is as good as without the mask."""
    truth, scans = grazing_wall_scans(6)
    mask = StaticMask(furnished_room_map())
    for error_deg in (3.0, -4.0, 5.0):
        masked = Localizer(furnished_room_map(), truth, window=WINDOW, max_points=BEAMS)
        masked.pose = Pose2D(truth.x, truth.y, truth.theta + math.radians(error_deg))
        bare = Localizer(furnished_room_map(), truth, window=WINDOW, max_points=BEAMS)
        bare.pose = masked.pose
        for scan in scans:
            masked.update(truth, scan, mask=mask)
            bare.update(truth, scan)
        left = abs(math.degrees(wrap_angle(masked.pose.theta - truth.theta)))
        without = abs(math.degrees(wrap_angle(bare.pose.theta - truth.theta)))
        assert left < 1.0, (
            f"{error_deg:+.0f} deg: {left:.2f} deg left with the mask, {without:.2f} without"
        )
        assert left <= without + 0.5


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


def test_an_intruder_that_agrees_twice_is_not_a_carry() -> None:
    """Two rest matches agreeing on a pose 8 cm from the held one: a carry when the whole scan
    fits the map better there (gain 0.12), match noise when it does not (a person standing by
    the lidar shifts the match twice but gains the map nothing, 0.04)."""
    from pepin.localization import CARRY_MIN_GAIN
    from pepin.odometry import Pose2D

    loc = tracker()
    held = Pose2D(0.0, 0.0, 0.0)
    shifted = Pose2D(0.08, 0.0, 0.0)
    assert not loc._carried(shifted, held, 0.12)  # the first one is only the hint
    assert loc._carried(shifted, held, 0.12)  # the second agrees and the map agrees
    loc._rest_hint = None
    assert not loc._carried(shifted, held, 0.04)
    assert not loc._carried(shifted, held, 0.04)  # agrees twice, gains nothing: an intruder
    assert CARRY_MIN_GAIN == 0.08
