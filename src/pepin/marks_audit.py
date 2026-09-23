"""Who painted the lethal cells the controller refuses to drive through: the lidar, or the camera.

On 2026-09-22 run 0431 made 0 m of progress towards the printer. The controller aborted nine
FollowPath goals in ten seconds, Hybrid-A* answered "no valid path found" over and over, and in
Foxglove the operator saw lethal blobs strung along the path where nothing stands. The tape said
afterwards (``scratch/one_localiser/tape_0431_phantoms.py``) that 54-73 % of the near-ahead lethal
cells had no lidar return anywhere near them and that the camera's fan covered them. That answer
took an hour of replay. This module is the same arithmetic, cheap enough to run live, so the
operator reads the number while he is standing in front of the room the cells claim to describe.

THE ENCODING, verified against the topic and not assumed. ``/local_costmap/costmap`` is a
``nav_msgs/OccupancyGrid`` on the ROS 0..100 scale, which is what
``nav2_costmap_2d::Costmap2DPublisher`` writes its internal 0..255 costs out as: 254
(``LETHAL_OBSTACLE``) becomes :data:`LETHAL` 100, 253 (``INSCRIBED_INFLATED_OBSTACLE``) becomes
:data:`INSCRIBED` 99, unknown becomes -1, and the whole inflation ramp is squeezed into 1..98. So
the threshold here is 100, not 253; 253 would be the ``costmap_raw`` topic
(``nav2_msgs/Costmap``), which nothing on this robot subscribes to.

WHAT THE CLASSES MEAN, and the one thing this cannot tell. A lethal cell with a ``/scan`` return
within ``match_cells`` of it is the lidar's and is not news. One without, but with a
``/depth_marks`` beam landing on it, is the CAMERA'S ALONE — and that is either a phantom (SGBM
lifting herringbone parquet, a stale voxel, a wrong pose) or a real thing only the camera can see,
a table top or a seat above the lidar's plane. Nothing in this arithmetic separates those two, and
nothing pretends to: the operator has the room in front of him and decides. A cell neither sensor
explains is ``unexplained`` — the ToF whiskers, a stale mark nothing has raytraced away yet, or a
frame error, and a number that grows is the signal that the frames have come apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

# The ROS OccupancyGrid values nav2_costmap_2d publishes its two hard costs as.
LETHAL = 100  # costmap 254, LETHAL_OBSTACLE: a sensor marked this cell
INSCRIBED = 99  # costmap 253, INSCRIBED_INFLATED_OBSTACLE: the robot's centre may not stand here

MATCH_CELLS = 1.5  # how many costmap cells a beam may land from a cell and still explain it
RADIUS_M = 2.0  # how far around the cart cells are judged: the lidar's own obstacle_max_range


@dataclass(frozen=True)
class MarksVerdict:
    """One costmap frame's hard cells, split by the sensor that can account for them.

    ``camera_only_xy`` is (n, 2) in the grid's own frame — the cells to draw red — and
    ``nearest_camera_only_m`` is the distance from the cart to the nearest of them, ``inf`` when
    there are none.
    """

    lethal: int
    lidar_backed: int
    camera_only: int
    unexplained: int
    nearest_camera_only_m: float
    camera_only_xy: Array

    def report(self) -> str:
        """The counts as the node's report line says them."""
        nearest = (
            "none"
            if not np.isfinite(self.nearest_camera_only_m)
            else f"{self.nearest_camera_only_m:.2f} m"
        )
        return (
            f"lethal {self.lethal} (lidar {self.lidar_backed}, camera-only {self.camera_only},"
            f" unexplained {self.unexplained}), nearest camera-only {nearest}"
        )


def cell_centres(
    mask: npt.NDArray[np.bool_], origin: tuple[float, float], resolution: float
) -> Array:
    """The grid-frame centres (n, 2) of the cells ``mask`` selects, row-major from the origin
    corner — where an OccupancyGrid's (0, 0) cell sits."""
    rows, cols = np.nonzero(mask)
    return np.column_stack(
        (origin[0] + (cols + 0.5) * resolution, origin[1] + (rows + 0.5) * resolution)
    )


def transform_xy(points: Array, rotation: Array, translation: Array) -> Array:
    """(n, 2) points of a sensor's frame placed in the frame ``rotation``/``translation`` describe.

    The full 3x3 rotation on (x, y, 0) and not a yaw, so an upside-down lidar or a tilted mount
    lands where it really points; the height is dropped afterwards, which is what a planar costmap
    does with it anyway.
    """
    if len(points) == 0:
        return np.zeros((0, 2))
    xyz = np.column_stack((points, np.zeros(len(points))))
    placed: Array = xyz @ np.asarray(rotation, dtype=float).T + np.asarray(
        translation, dtype=float
    ).reshape(3)
    return placed[:, :2]


def scan_points(angles: Array, ranges: Array) -> Array:
    """A LaserScan's bearings and ranges as (n, 2) points in the sensor's own frame; bearings
    that are not a return (NaN, as :func:`pepin_bringup.msgs.scan_arrays` leaves them) are gone."""
    good = np.isfinite(ranges)
    a, r = angles[good], ranges[good]
    return np.column_stack((r * np.cos(a), r * np.sin(a)))


def nearest_distance(targets: Array, points: Array) -> Array:
    """For every target (n, 2) the distance to the nearest of ``points`` (m, 2); ``inf`` for every
    target when there are no points.

    Brute force, on purpose: the local costmap is 60x60 cells and a revolution is about 450
    returns, so the worst case this can ever be handed is 3600x450 = 1.6 M distances, a few
    milliseconds of numpy once a second. scipy is not in the container image
    (``docker exec pepin-vslam python3 -c "import scipy"`` -> ModuleNotFoundError, 2026-09-22), so
    a cKDTree would be a new dependency bought for nothing.
    """
    if len(targets) == 0:
        return np.zeros(0)
    if len(points) == 0:
        return np.full(len(targets), np.inf)
    dx = targets[:, 0, None] - points[None, :, 0]
    dy = targets[:, 1, None] - points[None, :, 1]
    out: Array = np.sqrt((dx * dx + dy * dy).min(axis=1))
    return out


def audit_marks(
    grid: npt.NDArray[np.integer[Any]],
    origin: tuple[float, float],
    resolution: float,
    robot_xy: tuple[float, float],
    lidar_xy: Array,
    camera_xy: Array,
    *,
    radius_m: float = RADIUS_M,
    match_cells: float = MATCH_CELLS,
    inscribed_counts: bool = False,
) -> MarksVerdict:
    """Split one costmap's hard cells within ``radius_m`` of the cart into lidar-backed,
    camera-only and unexplained.

    ``grid`` is (height, width) of ROS OccupancyGrid values as published; ``origin`` and
    ``robot_xy`` are metres in the grid's own frame (``header.frame_id``, ``odom`` on this robot
    since 2026-09-22), and ``lidar_xy`` / ``camera_xy`` are the two sensors' returns already
    placed in that same frame. ``inscribed_counts`` adds the inflation's 99 band to the marks
    themselves; off — the default — only a cell a sensor actually marked is judged, because an
    inscribed cell is the inflation's arithmetic and blaming a sensor for it says nothing.
    """
    floor = INSCRIBED if inscribed_counts else LETHAL
    centres = cell_centres(np.asarray(grid) >= floor, origin, resolution)
    if len(centres):
        near = np.hypot(centres[:, 0] - robot_xy[0], centres[:, 1] - robot_xy[1]) <= radius_m
        centres = centres[near]
    tolerance = match_cells * resolution
    backed = nearest_distance(centres, lidar_xy) <= tolerance
    unbacked = centres[~backed]
    seen = nearest_distance(unbacked, camera_xy) <= tolerance
    camera_only = unbacked[seen]
    nearest = float("inf")
    if len(camera_only):
        nearest = float(
            np.hypot(camera_only[:, 0] - robot_xy[0], camera_only[:, 1] - robot_xy[1]).min()
        )
    return MarksVerdict(
        lethal=len(centres),
        lidar_backed=int(backed.sum()),
        camera_only=len(camera_only),
        unexplained=len(unbacked) - len(camera_only),
        nearest_camera_only_m=nearest,
        camera_only_xy=camera_only,
    )
