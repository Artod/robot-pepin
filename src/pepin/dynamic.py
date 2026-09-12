"""What the map did not know about: new objects get a berth the point planners must respect.

NavFn, Smac 2D and Theta* plan a point and trust the inflation layer's inscribed band to stand in
for the robot's width. On this cart that band is 6 cm — ``base_link`` sits at the front edge of
a 55 cm wide hull — so the plan passed a standing person's shins at 6 cm and a wheel took their
toes (2026-09-09). Widening the band strands every start parked against furniture, which is the
working case here. So the width is added where it is needed and nowhere else: lidar returns
that the static map does not explain (a person, a moved chair, a bag) are marked into the
costmaps as lethal rings of :attr:`Berth.ring_m` radius, a berth the point planners cannot enter.
Mapped furniture keeps its 6 cm and the cart still parks against it.

A mark within :attr:`Berth.trim_m` of ``base_link`` is dropped: that is the cart's own outline,
and a lethal cell on it refuses its departure (run 0087). Whatever stands that close is handled
by the contact band and the controller's own footprint check anyway. :attr:`Berth.near_m` — the
range inside which a return is not ringed at all — is the outline too, so the blind disc stays
the cart's size whatever the ring grows to; the older rule that made it the ring plus the
outline is still reachable (``berth_for(..., near_rings=False)``, the tracker's ``near_rings``
flag).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
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
    half-metre ring stand 0.26 m apart, five costmap cells of clear floor the plan walks straight
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
    return the map does not explain, farther than ``berth.near_m`` from the robot, with every
    mark nearer than ``berth.trim_m`` dropped so no ring lands on the cart's own outline.
    (M, 2) in the map frame; empty when the scan agrees with the map."""
    if len(points_base) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    ranges = np.hypot(points_base[:, 0], points_base[:, 1])
    far = ranges >= berth.near_m
    kept = points_base[far]
    unexplained = ~mask.explains(to_map(kept, pose))
    news = kept[unexplained]
    if len(news) > MAX_NEWS_POINTS:  # the nearest first: what is close is what the wheels meet
        nearest = np.argsort(ranges[far][unexplained])[:MAX_NEWS_POINTS]
        news = news[nearest]
    # The ring is built in the base frame so the trim can measure each mark from the cart
    # itself; the marks go to the map frame whole, as one transform.
    marks = rings(dedupe(news), berth.ring_m)
    if berth.trim_m > 0.0 and len(marks):
        marks = marks[np.hypot(marks[:, 0], marks[:, 1]) >= berth.trim_m]
    return to_map(marks, pose)


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


SHOE_AHEAD_OF_ANKLE_M = 0.21  # a 28 cm shoe with the ankle 7 cm back from the heel
ANKLE_HEIGHT_M = 0.07  # where the leg's pivot sits above the floor
SHIN_LEAN_DEG = 10.0  # how far back a relaxed standing shin may lean off vertical


@cache
def toe_reach_m(lidar_z_m: float | None = None) -> float:
    """How far a standing person's toe reaches past the point the lidar's beam meets on the leg,
    metres, rounded up to the centimetre.

    The lidar sees one horizontal slice of a person: whatever the beam lands on, the shoe sticks
    out further and a wheel takes it (2026-09-09). The flat 0.20 m that stood here was the shoe
    alone, written while the mount was assumed to be at ankle height. The tape measure of
    2026-09-12 put the plane most of the way up the shin instead, and a shin leaning back off
    vertical carries the beam's contact point *behind* the ankle, so the toe reaches further
    past it: the shoe ahead of the ankle plus ``(z - ankle) * tan(lean)``, rounded up to the
    centimetre. The height itself is never typed here — it is ``config/lidar.json`` through
    :func:`pepin.mounts.load_lidar_mount`, read once (``lidar_z_m`` overrides it, for tests and
    for asking what another mount would cost); a re-measured lidar takes a restart, like every
    other mount on the cart.
    """
    if lidar_z_m is None:
        from pepin.mounts import load_lidar_mount

        lidar_z_m = load_lidar_mount().z_m
    above_ankle = max(lidar_z_m - ANKLE_HEIGHT_M, 0.0)
    reach = SHOE_AHEAD_OF_ANKLE_M + above_ankle * math.tan(math.radians(SHIN_LEAN_DEG))
    return math.ceil(reach * 100.0) / 100.0


HAND_M = 0.05  # the margin a planner that knows the hull passes a person with
FOOTPRINT_PLANNERS = ("Hybrid", "Lattice")  # Nav2 plugin ids that check the polygon


@dataclass(frozen=True)
class Berth:
    """How new objects are marked for the planner in charge.

    ``ring_m`` is the lethal ring drawn around every unexplained return; ``near_m`` the distance
    inside which a return is not ringed at all; ``trim_m`` the distance inside which a single
    mark is dropped, whatever it belongs to — a mark there lands on the cart's own outline and
    refuses its every command (run 0087). The failure was a *mark* on the outline, so the trim
    is what answers it and ``near_m`` need not grow with the ring
    (:func:`hull_clearance_m`, :func:`near_exclusion_m`, :func:`berth_for`).
    """

    ring_m: float
    near_m: float
    trim_m: float = 0.0


def point_planner_ring_m(hull: Footprint = HULL, reach_m: float | None = None) -> float:
    """The ring a point planner needs so a wheel clears a person's toes.

    NavFn, Smac 2D and Theta* keep their path the hull's inscribed radius from a lethal cell and
    no farther; the wheel line is the half-width out from the path. The ring makes up the
    difference and adds the toes the lidar cannot see (:func:`toe_reach_m`): on this cart
    0.275 - 0.0625 + the reach.
    """
    return (
        hull.half_width_m
        - hull.inscribed_radius_m
        + (toe_reach_m() if reach_m is None else reach_m)
    )


def footprint_planner_ring_m(reach_m: float | None = None, hand_m: float = HAND_M) -> float:
    """The ring for a planner that checks the polygon itself: the toes and a hand's width."""
    return (toe_reach_m() if reach_m is None else reach_m) + hand_m


def hull_clearance_m(hull: Footprint = HULL, cell_m: float = COSTMAP_CELL_M) -> float:
    """Closer than this to ``base_link`` no mark may be drawn: the hull's circumscribed radius
    plus one cell, the circle the cart occupies whatever way it is turned."""
    return hull.circumscribed_radius_m + cell_m


def near_exclusion_m(
    ring_m: float, hull: Footprint = HULL, cell_m: float = COSTMAP_CELL_M
) -> float:
    """Closer than this no ring is drawn at all: the ring plus the hull's circumscribed radius
    plus one cell, so no point of a ring can touch the cart's own outline.

    This is the ``near_rings=False`` berth, kept reachable (CLAUDE.md rule 19). It pays for the
    guarantee with a blind disc that grows with the ring: raise the reach by 7 cm and a person
    standing 0.90 m ahead stops being ringed, which is the case the ring exists for. The
    ``near_rings=True`` berth trims the offending marks instead (:func:`hull_clearance_m`).
    """
    return ring_m + hull_clearance_m(hull, cell_m)


def berth_for(
    planner_id: str,
    hull: Footprint = HULL,
    reach_m: float | None = None,
    near_rings: bool = True,
) -> Berth:
    """The berth for the Nav2 planner plugin in charge (``planner_selector``).

    ``reach_m`` overrides the toe reach the mount implies (:func:`toe_reach_m`), the live knob
    over the ring's size. With ``near_rings`` on, a return is ringed as soon as it clears the
    cart's own outline and only the marks that would land on that outline are dropped; off, the
    old rule applies and nothing within the ring plus the outline is ringed at all.
    """
    ring = (
        footprint_planner_ring_m(reach_m)
        if planner_id in FOOTPRINT_PLANNERS
        else point_planner_ring_m(hull, reach_m)
    )
    clearance = hull_clearance_m(hull)
    near = clearance if near_rings else near_exclusion_m(ring, hull)
    return Berth(ring, near, clearance)


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
