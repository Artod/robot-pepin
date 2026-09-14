"""The whole-map search: finding the robot when nobody knows where it stands.

The map is the furnished room plus a patch of lone occupied cells two meters
east of it — clutter seen once, a chair leg, a curtain. Pooled four times
coarser that junk reads as a solid wall and outscores the real room, which is
why the coarse pass hands several peaks to the fine grid instead of one winner.
"""

import math

import numpy as np
from synthetic import raycast_room

from pepin.localization import GLOBAL_POOL_FACTOR, Localizer, pooled
from pepin.mapping import GridSpec, OccupancyGrid
from pepin.odometry import Pose2D
from pepin.scanmatch import CorrelativeMatcher

SPEC = GridSpec(0.05, -4, -3, 12, 6)  # the 6 x 4 m room plus an unvisited wing to the east
MAPPING_POSES = (Pose2D(0, 0, 0), Pose2D(1, 0.5, 0.7), Pose2D(-1, -0.5, -2.0), Pose2D(0.5, -1, 2.5))
PILLAR = (-2.0, 1.0, -1.6, 1.4)  # a box in one corner: the room has no 180-degree twin
SPECKLE = (5.0, -1.5, 7.0, 1.5)  # x0, y0, x1, y1 of the junk field, 2 m clear of the walls
SPECKLE_STEP_CELLS = GLOBAL_POOL_FACTOR  # one lone cell per 0.2 m: exactly one per coarse cell
OCCUPIED = 5.0  # log-odds of a cell the map is sure about
TRUTH = Pose2D(1.2, -0.8, math.radians(60.0))
IN_THE_JUNK = Pose2D(5.6, -0.6, math.radians(145.0))  # where the coarse pass wants to put the robot


def furnished_room_map() -> OccupancyGrid:
    """The rectangle with a box in one corner, mapped from four poses inside it."""
    grid = OccupancyGrid(SPEC)
    for pose in MAPPING_POSES:
        grid.integrate(pose, raycast_room(pose, pillar=PILLAR))
    return grid


def speckled_map() -> OccupancyGrid:
    """The same room with a field of isolated occupied cells in otherwise unknown space."""
    grid = furnished_room_map()
    x0, y0, x1, y1 = SPECKLE
    step = SPECKLE_STEP_CELLS
    rows = slice(*(round((y - SPEC.y_min_m) / SPEC.resolution_m) for y in (y0, y1)), step)
    cols = slice(*(round((x - SPEC.x_min_m) / SPEC.resolution_m) for x in (x0, x1)), step)
    grid.log_odds[rows, cols] = OCCUPIED
    return grid


def scan() -> np.ndarray:
    """One revolution seen from :data:`TRUTH`, inside the room."""
    return raycast_room(TRUTH, pillar=PILLAR)


def test_the_pooled_grid_scores_the_speckle_field_above_the_true_room() -> None:
    """The premise of the multi-peak search: pooling turns the junk into a wall."""
    coarse = CorrelativeMatcher(pooled(speckled_map(), GLOBAL_POOL_FACTOR))
    points = scan()
    assert coarse.score(IN_THE_JUNK, points) > coarse.score(TRUTH, points)


def test_finds_a_pose_anywhere_on_a_clean_map() -> None:
    found, confidence = Localizer(furnished_room_map(), Pose2D()).global_search(scan())
    assert confidence > 0.5
    assert math.hypot(found.pose.x - TRUTH.x, found.pose.y - TRUTH.y) < 0.08
    assert abs(found.pose.theta - TRUTH.theta) < math.radians(4.0)


def test_the_speckle_field_does_not_win_the_whole_map_search() -> None:
    found, confidence = Localizer(speckled_map(), Pose2D()).global_search(scan())
    assert confidence > 0.5
    assert math.hypot(found.pose.x - TRUTH.x, found.pose.y - TRUTH.y) < 0.08
    assert abs(found.pose.theta - TRUTH.theta) < math.radians(4.0)


def test_the_junk_explains_no_scan_once_the_fine_grid_looks_at_it() -> None:
    """Why the ranking works: on the fine grid the lone cells cover too little to judge."""
    loc = Localizer(speckled_map(), Pose2D())
    points = scan()
    assert loc.refine(IN_THE_JUNK, points)[1] == 0.0
    assert loc.refine(TRUTH, points)[1] > 0.5


def test_the_ranked_places_are_what_the_search_chose_from() -> None:
    """The watchdog needs the whole ranking, not the winner: how alike the runner-up explains
    the scan is what tells a confident fix from a twin (pepin.watchdog.ambiguity)."""
    from pepin.watchdog import ambiguity

    loc = Localizer(speckled_map(), Pose2D())
    points = scan()
    places = loc.global_candidates(points, theta_step_deg=10.0, thin_to=90)
    found, confidence = loc.global_search(points, theta_step_deg=10.0, thin_to=90)
    assert places[0][0].pose == found.pose and places[0][1] == confidence
    assert len(places) > 1, "several distinct places were weighed"
    ranked = [(match.pose, loc.rank(match.pose, points)) for match, _ in places]
    assert ambiguity(ranked) < 0.9, "the furnished room is not a twin of anywhere"
    fits = [(match.pose, fit) for match, fit in places]
    assert ambiguity(fits) > ambiguity(ranked), "why the rank, not the fit: the fit saturates"


def test_a_place_measured_on_the_map_carries_how_sure_it_is() -> None:
    """What travels to the board with a candidate: the pose sharpened in the tracking window,
    the fit of the whole scan there, and a covariance read off the score surface."""
    from pepin.sources import WATCHDOG

    loc = Localizer(furnished_room_map(), Pose2D())
    points = scan()
    measured = loc.measure(TRUTH, points, WATCHDOG, stamp=12.5)
    assert measured.source == WATCHDOG and measured.stamp == 12.5
    assert measured.fit > 0.5
    assert math.hypot(measured.x - TRUTH.x, measured.y - TRUTH.y) < 0.08
    sx, sy, syaw = measured.sigmas
    assert 0.0 < sx < 0.1 and 0.0 < sy < 0.1 and 0.0 < syaw < math.radians(10.0)
    far = loc.measure(IN_THE_JUNK, points, WATCHDOG)
    assert far.fit == 0.0 and far.sigmas[0] > sx, "a place that fits nothing answers wide"


def test_the_tracker_s_own_belief_is_a_measurement_like_any_other() -> None:
    """So a candidate and the pose it argues with are weighed on one scale."""
    from pepin.sources import TRACKER

    loc = Localizer(furnished_room_map(), Pose2D())
    loc.adopt(TRUTH, 0.8)
    belief = loc.belief(stamp=3.0)
    assert belief.source == TRACKER and belief.fit == 0.8 and belief.stamp == 3.0
    assert belief.pose == TRUTH
    loc.adopt(TRUTH, 0.1)
    assert loc.belief().sigmas[0] > belief.sigmas[0], "a poor fit is a wide answer"


def test_the_tracker_names_the_flags_it_owns() -> None:
    """A node routes every flag to whichever object names it; the tracker refuses the rest."""
    loc = Localizer(furnished_room_map(), Pose2D())
    assert "rest_lock" in loc.switches and "accept_candidates" not in loc.switches
    settings: dict[str, object] = {"sources": loc.sources.enabled, "covariance": "fit"}
    for name in loc.switches:
        loc.switch(name, settings.get(name, True))
