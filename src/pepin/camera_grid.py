"""The volume's CURRENT occupied columns as occupancy grids the costmaps only draw (2026-09-24).

``/depth_marks`` hands both costmaps a 720-bearing fan, and each costmap's ``camera_layer`` (an
ObstacleLayer) accumulates it into a grid of its own: a second and a third copy of the camera's
obstacle memory. The global one is in ``map`` and is smeared by every ``map -> odom`` jump, the
behaviour tree's clears wipe both, and the planner and the controller end up disagreeing
(scratch/costmap_split). This module is the nvblox arrangement instead: the volume says what it
holds NOW as a grid, and a ``nav2_costmap_2d::StaticLayer`` in each costmap draws the latest one
and remembers nothing.

One set of cells, two grids:

* :class:`GridWindow` — a square about the cart in the volume's own frame (``odom``), its corner
  snapped to :data:`GRID_SNAP_M` so the rolling local costmap's layer sees a new origin only every
  half metre of travel: ``/camera_grid``.
* :class:`MapCanvas` — the same cells on the lattice of the map the global costmap's static layer
  reads (its origin, size and resolution exactly, so the costmap never resizes for this grid),
  drawn through ``map -> odom``: a full grid once per geometry, then a
  :class:`GridUpdateFields` rectangle that covers the window before and after each redraw.

A cell is OCCUPIED (100) when a surface point of the volume stands in it within the fan's height
band — :func:`pepin.volume_scan.band_surface`, the rule ``/depth_marks`` reads — and FREE (0)
otherwise. Under the costmap's ``use_maximum`` a 0 never lowers what another layer wrote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pepin.tsdf import Array, GridSpec, RigidPose, Tsdf
from pepin.volume_scan import MarksLaw, marks_box
from pepin.worldmap import OccupancyGridFields

Int8 = npt.NDArray[np.int8]

OCCUPIED = 100
FREE = 0
GRID_SIZE_M = 6.0  # twice the fan's reach (MARKS_RANGE_M): the local costmap's 3 m window fits
GRID_RESOLUTION_M = 0.05  # the volume's own voxel
# The window's corner moves in these steps, so the cart stays within a quarter of a step of the
# centre and the local layer re-sizes itself (a log line in Nav2) only every half metre.
GRID_SNAP_M = 0.5
# Nav2's StaticLayer calls two geometries equal within this (static_layer.cpp's EPSILON).
GEOMETRY_EPS = 1e-5


@dataclass(frozen=True)
class GridWindow:
    """A square grid in the volume's frame: its corner, cell size and cells per side."""

    origin: tuple[float, float]
    resolution_m: float
    cells: int

    @classmethod
    def around(
        cls,
        xy: tuple[float, float],
        size_m: float = GRID_SIZE_M,
        resolution_m: float = GRID_RESOLUTION_M,
        snap_m: float = GRID_SNAP_M,
        lattice: tuple[float, float] = (0.0, 0.0),
    ) -> GridWindow:
        """The window of ``size_m`` about ``xy`` whose corner sits on a ``snap_m`` lattice
        anchored at ``lattice`` (the volume's origin, so the cells are its voxel columns)."""
        cells = max(1, round(size_m / resolution_m))
        half = cells * resolution_m / 2
        step = max(snap_m, resolution_m)
        corner = tuple(
            anchor + step * round((c - half - anchor) / step)
            for c, anchor in zip(xy, lattice, strict=True)
        )
        return cls((corner[0], corner[1]), resolution_m, cells)

    @property
    def size_m(self) -> float:
        """The side of the square, metres."""
        return self.cells * self.resolution_m

    @property
    def centre(self) -> tuple[float, float]:
        """The square's centre in the volume's frame."""
        half = self.size_m / 2
        return (self.origin[0] + half, self.origin[1] + half)

    def corners(self) -> Array:
        """The square's four corners, (4, 2) metres in the volume's frame."""
        x0, y0 = self.origin
        s = self.size_m
        return np.array([[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s]])

    def draw(self, points: Array) -> Int8:
        """The window's cells, rows y and columns x: OCCUPIED where any of ``points`` ((n, 2+)
        metres, volume frame) falls, FREE elsewhere; points outside the square are ignored."""
        values = np.full((self.cells, self.cells), FREE, dtype=np.int8)
        if points.shape[0] == 0:
            return values
        ii = np.floor((points[:, 0] - self.origin[0]) / self.resolution_m).astype(int)
        jj = np.floor((points[:, 1] - self.origin[1]) / self.resolution_m).astype(int)
        inside = (ii >= 0) & (ii < self.cells) & (jj >= 0) & (jj < self.cells)
        values[jj[inside], ii[inside]] = OCCUPIED
        return values

    def fields(self, values: Int8) -> OccupancyGridFields:
        """``values`` as the fields of a nav_msgs/OccupancyGrid on this window."""
        return OccupancyGridFields(
            resolution=self.resolution_m,
            width=self.cells,
            height=self.cells,
            origin_x=self.origin[0],
            origin_y=self.origin[1],
            data=np.ascontiguousarray(values).reshape(-1),
        )


def window_box(
    spec: GridSpec, window: GridWindow, base_in_map: RigidPose, law: MarksLaw
) -> tuple[slice, slice, slice] | None:
    """The voxel box the window's cells are read from: its square, the fan's height band above
    the cart's floor, one voxel of margin (:func:`pepin.volume_scan.marks_box`); ``None`` when
    the window misses the volume."""
    z0 = float(base_in_map.translation[2])
    band = (z0 + law.band_m[0], z0 + law.band_m[1])
    return marks_box(spec, window.centre, band, window.size_m / 2)


def grid_volume(
    volume: Tsdf, window: GridWindow, base_in_map: RigidPose, law: MarksLaw
) -> Tsdf | None:
    """The neighbourhood the window is read from, copied out (:meth:`pepin.tsdf.Tsdf.window`):
    the one call that belongs under the model's lock; ``None`` outside the volume."""
    box = window_box(volume.spec, window, base_in_map, law)
    return None if box is None else volume.window(box)


@dataclass(frozen=True)
class CellRect:
    """A rectangle of map cells: first column, first row, columns, rows."""

    x: int
    y: int
    width: int
    height: int

    def union(self, other: CellRect | None) -> CellRect:
        """The smallest rectangle holding both (this one when ``other`` is ``None``)."""
        if other is None:
            return self
        x0, y0 = min(self.x, other.x), min(self.y, other.y)
        x1 = max(self.x + self.width, other.x + other.width)
        y1 = max(self.y + self.height, other.y + other.height)
        return CellRect(x0, y0, x1 - x0, y1 - y0)

    @property
    def rows(self) -> slice:
        """The rectangle's rows, for indexing a (height, width) array."""
        return slice(self.y, self.y + self.height)

    @property
    def cols(self) -> slice:
        """The rectangle's columns."""
        return slice(self.x, self.x + self.width)


@dataclass(frozen=True)
class MapGeometry:
    """The lattice of an occupancy grid: cell size, cells across and up, corner (map metres)."""

    resolution_m: float
    width: int
    height: int
    origin_x: float
    origin_y: float

    def same_as(self, other: MapGeometry) -> bool:
        """Whether Nav2's StaticLayer would call the two the same map (no resize)."""
        return (
            self.width == other.width
            and self.height == other.height
            and abs(self.resolution_m - other.resolution_m) < GEOMETRY_EPS
            and abs(self.origin_x - other.origin_x) < GEOMETRY_EPS
            and abs(self.origin_y - other.origin_y) < GEOMETRY_EPS
        )

    def text(self) -> str:
        """``400x300 at 5 cm from (-10.00, -7.50)`` for a report line."""
        return (
            f"{self.width}x{self.height} at {self.resolution_m * 100:.0f} cm from"
            f" ({self.origin_x:+.2f}, {self.origin_y:+.2f})"
        )

    def rect_of(self, xy: Array) -> CellRect | None:
        """The cells covering the bounding box of ``xy`` ((n, 2) map metres), clipped to the
        map; ``None`` when that box misses it."""
        if xy.shape[0] == 0:
            return None
        r = self.resolution_m
        x0 = max(0, math.floor((float(xy[:, 0].min()) - self.origin_x) / r))
        y0 = max(0, math.floor((float(xy[:, 1].min()) - self.origin_y) / r))
        x1 = min(self.width, math.ceil((float(xy[:, 0].max()) - self.origin_x) / r))
        y1 = min(self.height, math.ceil((float(xy[:, 1].max()) - self.origin_y) / r))
        if x1 <= x0 or y1 <= y0:
            return None
        return CellRect(x0, y0, x1 - x0, y1 - y0)


@dataclass(frozen=True)
class GridUpdateFields:
    """A map_msgs/OccupancyGridUpdate without ROS: the rectangle and its cells, row-major."""

    x: int
    y: int
    width: int
    height: int
    data: Int8

    def as_list(self) -> list[int]:
        """The cells as plain ints, the form rclpy accepts for an int8[] field."""
        return [int(v) for v in self.data]


class MapCanvas:
    """The camera's occupied cells on one map geometry, redrawn whole at every tick.

    Only the last window is ever on it: each :meth:`draw` erases the cells the previous window
    covered and draws the new ones, and the rectangle it returns covers both — so an update that
    carries it removes every stale cell and adds every current one, and nothing accumulates.
    """

    def __init__(self, geometry: MapGeometry) -> None:
        self.geometry = geometry
        self.values: Int8 = np.full((geometry.height, geometry.width), FREE, dtype=np.int8)
        self._window: CellRect | None = None

    def draw(self, points_xy: Array, footprint_xy: Array) -> CellRect | None:
        """Erase the last window, draw ``points_xy`` ((n, 2) map metres), and return the
        rectangle an update must carry: the last window's cells joined with the bounding box of
        ``footprint_xy`` (the new window's corners in map metres); ``None`` when neither touches
        the map."""
        old = self._window
        if old is not None:
            self.values[old.rows, old.cols] = FREE
        g = self.geometry
        if points_xy.shape[0]:
            ii = np.floor((points_xy[:, 0] - g.origin_x) / g.resolution_m).astype(int)
            jj = np.floor((points_xy[:, 1] - g.origin_y) / g.resolution_m).astype(int)
            inside = (ii >= 0) & (ii < g.width) & (jj >= 0) & (jj < g.height)
            self.values[jj[inside], ii[inside]] = OCCUPIED
        new = g.rect_of(footprint_xy)
        self._window = new
        if new is None:
            return old
        return new.union(old)

    def forget(self) -> CellRect | None:
        """Erase the last window and return its rectangle (an update that clears it), or
        ``None`` when nothing was drawn."""
        old = self._window
        if old is not None:
            self.values[old.rows, old.cols] = FREE
        self._window = None
        return old

    def full(self) -> OccupancyGridFields:
        """The whole canvas as the fields of a nav_msgs/OccupancyGrid."""
        g = self.geometry
        return OccupancyGridFields(
            resolution=g.resolution_m,
            width=g.width,
            height=g.height,
            origin_x=g.origin_x,
            origin_y=g.origin_y,
            data=np.ascontiguousarray(self.values).reshape(-1),
        )

    def update(self, rect: CellRect) -> GridUpdateFields:
        """The cells of ``rect`` as an update's fields."""
        cells = self.values[rect.rows, rect.cols]
        return GridUpdateFields(
            rect.x, rect.y, rect.width, rect.height, np.ascontiguousarray(cells).reshape(-1)
        )


def to_map_xy(points: Array, map_from_volume: RigidPose) -> Array:
    """``points`` ((n, 2) or (n, 3) volume-frame metres; z 0 when absent) carried into the map
    frame by ``map_from_volume`` (map <- volume), as (n, 2)."""
    if points.shape[0] == 0:
        return np.zeros((0, 2))
    xyz = points if points.shape[1] == 3 else np.c_[points[:, :2], np.zeros(points.shape[0])]
    moved = xyz @ map_from_volume.rotation.T + map_from_volume.translation
    return np.asarray(moved[:, :2], dtype=np.float64)
