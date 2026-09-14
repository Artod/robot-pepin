"""What the static map does not explain — a person, a moved chair, a bag — and what that costs
the pose.

One question, asked of every lidar return: does the saved map have an obstacle here? Grow the
map's occupied cells by :data:`STATIC_M` and the answer is a mask (:class:`StaticMask`). Two
consumers, both of them localisation:

* :func:`voting_mask` — a return the map cannot explain carries no information about a pose
  measured against that map, so it does not score the match (the tracker's ``explained_vote``).
* :func:`occluded` — a NEAR scan the map cannot explain while the walls beyond still fit is a
  person beside the cart, not a lost cart.

Nothing here reaches a costmap any more. Until 2026-09-14 this module also painted lethal rings
of a person's toe reach around every unexplained return and published them as
``/dynamic_obstacles``: a 70 cm tape gap read 58 cm lethal-to-lethal, and the drive through it
cost four minutes of recoveries where the same gap took 36 s with the rings off. The rings were
an overfit to one obstacle (a standing person's feet) and they were redundant besides — a mark
went into the lidar's own costmap layer, at a cell that layer's ``scan`` source had already
marked from the same return. A new object is the cell the beam found; the lidar layer marks it
and the lidar's own rays clear it when it leaves.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from pepin.mapping import OccupancyGrid
from pepin.odometry import Pose2D

STATIC_M = 0.15  # a return within this distance of a mapped occupied cell is the map, not news


class StaticMask:
    """Which world points the static map explains: occupied cells grown by :data:`STATIC_M`."""

    def __init__(self, grid: OccupancyGrid, grow_m: float = STATIC_M) -> None:
        self._grid = grid
        self.grow_m = grow_m
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
