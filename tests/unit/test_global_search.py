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
