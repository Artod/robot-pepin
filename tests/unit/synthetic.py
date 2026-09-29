"""A rectangular room and an ideal lidar for unit tests."""

import math

import numpy as np

from pepin.mapping import GridSpec, OccupancyGrid
from pepin.odometry import Pose2D

ROOM_W, ROOM_H = 6.0, 4.0


Box = tuple[float, float, float, float]  # x0, y0, x1, y1 of an axis-aligned obstacle


def raycast_room(pose: Pose2D, beams: int = 180, pillar: Box | None = None) -> np.ndarray:
    """Robot-frame hit points of rays from ``pose`` against the walls (and a box, if given)."""
    vertical = [(-ROOM_W / 2, -ROOM_H / 2, ROOM_H / 2), (ROOM_W / 2, -ROOM_H / 2, ROOM_H / 2)]
    horizontal = [(-ROOM_H / 2, -ROOM_W / 2, ROOM_W / 2), (ROOM_H / 2, -ROOM_W / 2, ROOM_W / 2)]
    if pillar is not None:
        x0, y0, x1, y1 = pillar
        vertical += [(x0, y0, y1), (x1, y0, y1)]
        horizontal += [(y0, x0, x1), (y1, x0, x1)]
    pts = []
    for k in range(beams):
        a = pose.theta + 2 * math.pi * k / beams
        dx, dy = math.cos(a), math.sin(a)
        ts = []
        for wall_x, lo, hi in vertical:
            t = (wall_x - pose.x) / dx if abs(dx) > 1e-9 else -1
            if t > 0 and lo <= pose.y + t * dy <= hi:
                ts.append(t)
        for wall_y, lo, hi in horizontal:
            t = (wall_y - pose.y) / dy if abs(dy) > 1e-9 else -1
            if t > 0 and lo <= pose.x + t * dx <= hi:
                ts.append(t)
        r = min(ts)
        pts.append((r * math.cos(a - pose.theta), r * math.sin(a - pose.theta)))
    return np.array(pts)


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
