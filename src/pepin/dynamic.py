"""What the map did not know about: new objects get a berth the point planners must respect.

NavFn, Smac 2D and Theta* plan a point and trust the inflation layer's inscribed band to stand in
for the robot's width. On this cart that band is 6 cm — ``base_link`` sits at the front edge of
a 55 cm wide hull — so the plan passed a standing person's shins at 6 cm and a wheel took their
toes (2026-09-09). Widening the band strands every start parked against furniture, which is the
working case here. So the width is added where it is needed and nowhere else: lidar returns
that the static map does not explain (a person, a moved chair, a bag) are marked into the
costmaps as lethal rings of :attr:`Berth.ring_m` radius, a berth the point planners cannot enter.
Mapped furniture keeps its 6 cm and the cart still parks against it.

Points within :attr:`Berth.near_m` of the robot get no ring: whatever stands beside a parked cart is
handled by the contact band and the controller's own footprint check, and a ring landing on the
cart's outline would refuse its departure (run 0087).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from pepin.footprint import HULL, Footprint
from pepin.mapping import OccupancyGrid
from pepin.odometry import Pose2D

COSTMAP_CELL_M = 0.05  # the Nav2 costmaps' resolution (test_nav_contract pins the YAML to it)
RING_MIN_POINTS = 12  # the fewest marks any ring is drawn with; a big ring needs many more
STATIC_M = 0.15  # a return within this distance of a mapped occupied cell is the map, not news
MAX_NEWS_POINTS = (
    60  # a person is a handful of returns; hundreds mean the pose is wrong, not the room
)
DEDUPE_M = 0.10  # returns closer together than this are one object


class StaticMask:
    """Which world points the static map explains: occupied cells grown by :data:`STATIC_M`."""

    def __init__(self, grid: OccupancyGrid, grow_m: float = STATIC_M) -> None:
        self._grid = grid
        occupied = grid.log_odds > 0.0
        cells = max(1, math.ceil(grow_m / grid.spec.resolution_m))
        mask = np.zeros_like(occupied)
        rows, cols = occupied.shape
        for dr in range(-cells, cells + 1):
            for dc in range(-cells, cells + 1):
                if dr * dr + dc * dc > cells * cells:
                    continue
                src = occupied[max(0, -dr) : rows - max(0, dr), max(0, -dc) : cols - max(0, dc)]
                mask[max(0, dr) : rows - max(0, -dr), max(0, dc) : cols - max(0, -dc)] |= src
        self._mask = mask

    def explains(self, points_map: NDArray[np.float64]) -> NDArray[np.bool_]:
        """True for each (N, 2) map-frame point that lies on or next to a mapped obstacle.

        Points off the grid count as explained: unknown ground is not evidence of a person.
        """
        if len(points_map) == 0:
            return np.zeros(0, dtype=bool)
        cells = self._grid.world_to_cell(points_map)
        rows, cols = self._mask.shape
        inside = (
            (cells[:, 0] >= 0) & (cells[:, 0] < rows) & (cells[:, 1] >= 0) & (cells[:, 1] < cols)
        )
        out = np.ones(len(points_map), dtype=bool)
        out[inside] = self._mask[cells[inside, 0], cells[inside, 1]]
        return out


def to_map(points_base: NDArray[np.float64], pose: Pose2D) -> NDArray[np.float64]:
    """(N, 2) points in the base frame at ``pose`` to the map frame."""
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    x, y = points_base[:, 0], points_base[:, 1]
    return np.column_stack((pose.x + c * x - s * y, pose.y + s * x + c * y))


def dedupe(points: NDArray[np.float64], cell_m: float = DEDUPE_M) -> NDArray[np.float64]:
    """One point per ``cell_m`` cell: a person is a handful of returns, not a hundred."""
    if len(points) == 0:
        return points
    keys = np.floor(points / cell_m).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def ring_points(radius_m: float, cell_m: float = COSTMAP_CELL_M) -> int:
    """How many marks close a ring of ``radius_m``: no two of them more than ``cell_m`` apart.

    A point planner is stopped by lethal cells, not by the gaps between them: twelve marks on a
    0.41 m ring stand 0.22 m apart, four costmap cells of clear floor the plan walks straight
    through, which is the opposite of a berth.
    """
    return max(RING_MIN_POINTS, math.ceil(2.0 * math.pi * radius_m / cell_m))


def rings(
    centres: NDArray[np.float64], radius_m: float, n: int | None = None
) -> NDArray[np.float64]:
    """Each centre, plus ``n`` points closing a circle of ``radius_m`` around it, as (M, 2)."""
    if n is None:
        n = ring_points(radius_m, COSTMAP_CELL_M)
    if len(centres) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ring = np.column_stack((radius_m * np.cos(angles), radius_m * np.sin(angles)))
    return np.concatenate([centres, (centres[:, None, :] + ring[None, :, :]).reshape(-1, 2)])


def dynamic_marks(
    points_base: NDArray[np.float64], pose: Pose2D, mask: StaticMask, berth: Berth
) -> NDArray[np.float64]:
    """The lethal marks a scan adds to the costmaps: rings of ``berth.ring_m`` around every
    return the map does not explain, farther than ``berth.near_m`` from the robot. (M, 2) in the
    map frame; empty when the scan agrees with the map."""
    if len(points_base) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    ranges = np.hypot(points_base[:, 0], points_base[:, 1])
    far = ranges >= berth.near_m
    world = to_map(points_base[far], pose)
    unexplained = ~mask.explains(world)
    news = world[unexplained]
    if len(news) > MAX_NEWS_POINTS:  # the nearest first: what is close is what the wheels meet
        nearest = np.argsort(ranges[far][unexplained])[:MAX_NEWS_POINTS]
        news = news[nearest]
    return rings(dedupe(news), berth.ring_m)


VOTE_MIN_SHARE = 0.5  # below this the pose, not the room, is what the scan disagrees with
VOTE_MIN_POINTS = 60  # too few explained returns fix no pose; let the whole scan speak


def voting_mask(
    points_base: NDArray[np.float64],
    pose: Pose2D,
    mask: StaticMask,
    min_share: float = VOTE_MIN_SHARE,
    min_points: int = VOTE_MIN_POINTS,
) -> NDArray[np.bool_] | None:
    """Which returns may vote in the scan match at ``pose``: the ones the static map explains.

    A blanket over a chair, a bag on the floor, a person: returns the map has no wall for pull the
    correlative score toward whatever cell happens to lie under them, and the heading follows the
    furniture instead of the walls. Silencing them costs nothing — they carry no information about
    a pose measured against a static map.

    Returns ``None`` (everything votes, the behaviour without this filter) when the mask would be
    self-confirming: fewer than ``min_share`` of the scan explained, or fewer than ``min_points``
    left. Then it is the pose that is wrong, and the returns it does not explain are exactly the
    evidence that can fix it.
    """
    if len(points_base) == 0:
        return None
    explained = mask.explains(to_map(points_base, pose))
    if explained.mean() < min_share or int(explained.sum()) < min_points:
        return None
    return explained


NEAR_OCCLUSION_M = 1.5  # a person beside the cart is within this; the walls are beyond


def occlusion_split(
    points_base: NDArray[np.float64], explained: NDArray[np.bool_], near_m: float = NEAR_OCCLUSION_M
) -> tuple[float, NDArray[np.bool_]]:
    """How much of the NEAR scan the map cannot explain, and which points are FAR.

    A person beside the cart is close and unexplained while the walls beyond still fit; a wrong
    pose leaves the far points unexplained too. Returns the unexplained share among points
    within ``near_m`` (0.0 when there are none) and the mask of the points beyond it: the caller
    judges the pose on those alone.
    """
    if len(points_base) == 0:
        return 0.0, np.zeros(0, dtype=bool)
    far = np.hypot(points_base[:, 0], points_base[:, 1]) >= near_m
    near = ~far
    share = 0.0 if not near.any() else float(1.0 - explained[near].mean())
    return share, far


TOE_REACH_M = 0.20  # a foot reaches this far past the shin the lidar sees at 0.20 m
HAND_M = 0.05  # the margin a planner that knows the hull passes a person with
FOOTPRINT_PLANNERS = ("Hybrid", "Lattice")  # Nav2 plugin ids that check the polygon


@dataclass(frozen=True)
class Berth:
    """How new objects are marked for the planner in charge.

    ``ring_m`` is the lethal ring drawn around every unexplained return; ``near_m`` the distance
    inside which nothing is ringed (a ring there would land on the cart's own outline and refuse
    its every command — run 0087).
    """

    ring_m: float
    near_m: float


def point_planner_ring_m(hull: Footprint = HULL, toe_reach_m: float = TOE_REACH_M) -> float:
    """The ring a point planner needs so a wheel clears a person's toes.

    NavFn, Smac 2D and Theta* keep their path the hull's inscribed radius from a lethal cell and
    no farther; the wheel line is the half-width out from the path. The ring makes up the
    difference and adds the toes the lidar cannot see: 0.275 - 0.0625 + 0.20 on this cart.
    """
    return hull.half_width_m - hull.inscribed_radius_m + toe_reach_m


def footprint_planner_ring_m(toe_reach_m: float = TOE_REACH_M, hand_m: float = HAND_M) -> float:
    """The ring for a planner that checks the polygon itself: the toes and a hand's width."""
    return toe_reach_m + hand_m


def near_exclusion_m(
    ring_m: float, hull: Footprint = HULL, cell_m: float = COSTMAP_CELL_M
) -> float:
    """Closer than this no ring is drawn: the ring plus the hull's circumscribed radius plus one
    cell, so a ring can never touch the cart's own outline."""
    return ring_m + hull.circumscribed_radius_m + cell_m


def berth_for(planner_id: str, hull: Footprint = HULL) -> Berth:
    """The berth for the Nav2 planner plugin in charge (``planner_selector``)."""
    ring = (
        footprint_planner_ring_m()
        if planner_id in FOOTPRINT_PLANNERS
        else point_planner_ring_m(hull)
    )
    return Berth(ring, near_exclusion_m(ring, hull))


OCCLUDED_SHARE = 0.25  # a quarter of the near scan on things the map does not know
MIN_JUDGEABLE_POINTS = 20  # fewer returns than this say nothing about occlusion or fit


class FitJudge(Protocol):
    """Scores how well a scan lies on the map at a pose — what
    :class:`pepin.scanmatch.CorrelativeMatcher` does, as little of it as :func:`occluded` needs."""

    def inlier_fraction(self, pose: Pose2D, points: NDArray[np.float64]) -> float:
        """The share of ``points`` (base frame, at ``pose``) that land on mapped obstacles."""
        ...


def occluded(
    points: NDArray[np.float64] | None,
    pose: Pose2D,
    mask: StaticMask | None,
    judge: FitJudge | None,
    lost_fit: float,
) -> bool:
    """A person beside the cart, not a lost cart: the NEAR scan is mostly things the map does
    not know (:func:`occlusion_split`) while the walls beyond still fit ``pose`` at least as
    well as ``lost_fit``. A wrong pose fails the far test and is not occluded (the first version
    judged the whole scan and hid a twin behind "occluded"). False before the map, the matcher
    or a scan of :data:`MIN_JUDGEABLE_POINTS` returns exists."""
    if mask is None or judge is None or points is None or len(points) < MIN_JUDGEABLE_POINTS:
        return False
    near_share, far = occlusion_split(points, mask.explains(to_map(points, pose)))
    if near_share <= OCCLUDED_SHARE or far.sum() < MIN_JUDGEABLE_POINTS:
        return False
    return judge.inlier_fraction(pose, points[far]) >= lost_fit
