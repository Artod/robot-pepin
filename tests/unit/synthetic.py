"""A rectangular room and an ideal lidar for unit tests."""

import math

import numpy as np

from pepin.odometry import Pose2D

ROOM_W, ROOM_H = 6.0, 4.0


def raycast_room(pose: Pose2D, beams: int = 180) -> np.ndarray:
    """Robot-frame hit points of rays from ``pose`` against the walls of the room."""
    pts = []
    for k in range(beams):
        a = pose.theta + 2 * math.pi * k / beams
        dx, dy = math.cos(a), math.sin(a)
        ts = []
        for wall_x in (-ROOM_W / 2, ROOM_W / 2):
            t = (wall_x - pose.x) / dx if abs(dx) > 1e-9 else -1
            if t > 0 and abs(pose.y + t * dy) <= ROOM_H / 2:
                ts.append(t)
        for wall_y in (-ROOM_H / 2, ROOM_H / 2):
            t = (wall_y - pose.y) / dy if abs(dy) > 1e-9 else -1
            if t > 0 and abs(pose.x + t * dx) <= ROOM_W / 2:
                ts.append(t)
        r = min(ts)
        pts.append((r * math.cos(a - pose.theta), r * math.sin(a - pose.theta)))
    return np.array(pts)
