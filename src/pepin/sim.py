"""A kinematic world for Nav2: the cart as a unicycle, the room as the grid it drives on, the lidar
as rays cast into that grid and into boxes of furniture the grid does not hold.

Nothing here is ROS (``ros/sim/sim_world.py`` is the node around it) and nothing here is physics.
What it models is what Nav2's behaviour depends on: where the hull is, what the lidar sees from
there through the cart's own mount and scan filter, and when the base stops by itself (the base
bridge's 0.5 s deadman). What it does not model is listed in ros/README.md, "Simulation": wheel
slip, lidar noise, the camera and the ToF sensors, WiFi, light.

The grid is RTAB-Map's, the one the stack drives on (``/map`` under ``PEPIN_LOCALIZER=rtabmap``):
:func:`grid_from_rtabmap_db` reads the grid a database saved at its last shutdown
(``Admin.opt_map``, what RTAB-Map publishes when it loads the database to localise), and
:func:`graph_poses_from_rtabmap_db` the node poses the places ride on (pepin.places).
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pepin.footprint import CONTACT_BAND_M, Footprint, hull_box
from pepin.geometry import BaseConfig
from pepin.lidar import LidarMount
from pepin.measurements import compose
from pepin.mounts import lidar_mount
from pepin.odometry import Pose2D, wrap_angle

# The base bridge's deadman (ros/pepin_base_cpp/src/base_bridge.cpp, cmd_timeout_s): no /cmd_vel
# for this long and the wheels are stopped.
DEADMAN_S = 0.5
# The LD19 as the board's launch configures it (robot.launch.py: lidar.bins 455, one revolution
# from angle 0 to 2 pi, counter-clockwise in the laser frame).
LIDAR_BEAMS = 455
# How finely the hull is sampled for contact: half a costmap cell.
HULL_SAMPLE_M = 0.025
# /map's cell values (nav_msgs/OccupancyGrid) and the trinary pgm's pixels (map_server's
# convention; pepin.mapping.grid_from_pgm reads the same three).
OCCUPIED, FREE, UNKNOWN = 100, 0, -1
PGM_OCCUPIED, PGM_FREE, PGM_UNKNOWN = 0, 254, 205
# RTAB-Map's cv::Mat depth codes (the low three bits of a cv type) as numpy types.
_CV_DEPTHS: dict[int, type[np.generic]] = {
    0: np.uint8,
    1: np.int8,
    2: np.uint16,
    3: np.int16,
    4: np.int32,
    5: np.float32,
    6: np.float64,
}


@dataclass(frozen=True)
class Grid:
    """An occupancy grid laid out as ``/map`` is: ``cells[row, col]``, row 0 at ``origin_y``,
    column 0 at ``origin_x``, values :data:`OCCUPIED`, :data:`FREE`, :data:`UNKNOWN`."""

    cells: NDArray[np.int8]
    resolution: float
    origin_x: float
    origin_y: float

    @property
    def width(self) -> int:
        """Columns (cells along x)."""
        return int(self.cells.shape[1])

    @property
    def height(self) -> int:
        """Rows (cells along y)."""
        return int(self.cells.shape[0])

    def occupied_at(self, xs: NDArray[np.float64], ys: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Whether each world point falls in an occupied cell; outside the grid is not."""
        cols = np.floor((xs - self.origin_x) / self.resolution).astype(np.int64)
        rows = np.floor((ys - self.origin_y) / self.resolution).astype(np.int64)
        inside = (cols >= 0) & (cols < self.width) & (rows >= 0) & (rows < self.height)
        hit = np.zeros(xs.shape, dtype=bool)
        hit[inside] = self.cells[rows[inside], cols[inside]] == OCCUPIED
        return hit

    def to_pgm(self) -> bytes:
        """The trinary binary pgm map_server reads: top row = the largest y."""
        pixels = np.full(self.cells.shape, PGM_UNKNOWN, dtype=np.uint8)
        pixels[self.cells == OCCUPIED] = PGM_OCCUPIED
        pixels[self.cells == FREE] = PGM_FREE
        header = f"P5\n{self.width} {self.height}\n255\n".encode()
        return header + pixels[::-1].tobytes()

    @classmethod
    def from_pgm(cls, data: bytes, resolution: float, origin_x: float, origin_y: float) -> Grid:
        """A trinary pgm back as cells: dark occupied, white free, anything between unknown
        (the thresholds of pepin.mapping.grid_from_pgm)."""
        parts: list[bytes] = []
        at = 0
        while len(parts) < 4:  # magic, width, height, maxval; comments skipped
            while data[at : at + 1].isspace():
                at += 1
            if data[at : at + 1] == b"#":
                at = data.index(b"\n", at) + 1
                continue
            end = at
            while not data[end : end + 1].isspace():
                end += 1
            parts.append(data[at:end])
            at = end
        if parts[0] != b"P5":
            raise ValueError("not a binary pgm")
        width, height = int(parts[1]), int(parts[2])
        at += 1  # the single whitespace after maxval
        raw = np.frombuffer(data[at : at + width * height], dtype=np.uint8)
        pixels = raw.reshape(height, width)[::-1]  # the pgm's top row is the largest y
        cells = np.full(pixels.shape, UNKNOWN, dtype=np.int8)
        cells[pixels < 64] = OCCUPIED
        cells[pixels > 250] = FREE
        return cls(cells, resolution, origin_x, origin_y)

    def save(self, yaml_path: Path, note: str = "") -> None:
        """Write ``<stem>.pgm`` and the map_server yaml beside it; ``note`` becomes comments."""
        pgm = yaml_path.with_suffix(".pgm")
        pgm.write_bytes(self.to_pgm())
        comments = "".join(f"# {line}\n" for line in note.splitlines())
        yaml_path.write_text(
            f"{comments}image: {pgm.name}\nmode: trinary\nresolution: {self.resolution:.6f}\n"
            f"origin: [{self.origin_x:.6f}, {self.origin_y:.6f}, 0.0]\nnegate: 0\n"
            "occupied_thresh: 0.65\nfree_thresh: 0.25\n"
        )

    @classmethod
    def load(cls, yaml_path: Path) -> Grid:
        """A map_server yaml and its pgm (the image path relative to the yaml)."""
        import yaml

        meta = yaml.safe_load(yaml_path.read_text())
        origin = meta["origin"]
        return cls.from_pgm(
            (yaml_path.parent / meta["image"]).read_bytes(),
            float(meta["resolution"]),
            float(origin[0]),
            float(origin[1]),
        )


def rtabmap_blob(blob: bytes) -> NDArray[Any]:
    """RTAB-Map's compressData: a zlib stream followed by int32 rows, cols and the cv type."""
    rows, cols, cvtype = struct.unpack("<iii", blob[-12:])
    depth, channels = cvtype & 7, (cvtype >> 3) + 1
    array = np.frombuffer(zlib.decompress(blob[:-12]), _CV_DEPTHS[depth])
    shape = (rows, cols, channels) if channels > 1 else (rows, cols)
    return array.reshape(shape)


def _admin(db_path: Path, columns: str) -> tuple[Any, ...]:
    """One row of a database's Admin table, read without a lock (sqlite immutable): the caller
    hands a copy, never the file a running RTAB-Map writes."""
    with sqlite3.connect(f"file:{db_path}?immutable=1", uri=True) as db:
        row = db.execute(f"SELECT {columns} FROM Admin").fetchone()
    if row is None or row[0] is None:
        raise ValueError(f"{db_path}: no saved {columns.split(',')[0]} in Admin")
    return tuple(row)


def grid_from_rtabmap_db(db_path: Path) -> Grid:
    """The grid an RTAB-Map database saved at its last shutdown (Admin.opt_map): the map it
    publishes on ``/map`` the moment it loads the database to localise."""
    blob, x_min, y_min, resolution = _admin(
        db_path, "opt_map, opt_map_x_min, opt_map_y_min, opt_map_resolution"
    )
    cells = rtabmap_blob(bytes(blob)).astype(np.int8)
    return Grid(cells, float(resolution), float(x_min), float(y_min))


def graph_poses_from_rtabmap_db(db_path: Path) -> dict[int, Pose2D]:
    """The saved graph's node poses (Admin.opt_ids / opt_poses, 3x4 each) as planar poses."""
    ids_blob, poses_blob = _admin(db_path, "opt_ids, opt_poses")
    ids = rtabmap_blob(bytes(ids_blob)).ravel()
    poses = rtabmap_blob(bytes(poses_blob)).reshape(-1, 3, 4)
    return {
        int(i): Pose2D(float(p[0, 3]), float(p[1, 3]), math.atan2(float(p[1, 0]), float(p[0, 0])))
        for i, p in zip(ids, poses, strict=True)
    }


@dataclass(frozen=True)
class Box:
    """A piece of furniture the grid does not hold: a rectangle ``length`` along its own heading
    and ``width`` across it, centred at ``x, y`` in the map. Only the lidar sees it; ``/map``
    never does, as on the robot where the grid is RTAB-Map's and the chair moved."""

    name: str
    x: float
    y: float
    length: float
    width: float
    yaw_deg: float = 0.0

    def corners(self) -> NDArray[np.float64]:
        """The four corners in the map, counter-clockwise."""
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        hx, hy = self.length / 2.0, self.width / 2.0
        local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
        corners: NDArray[np.float64] = np.c_[
            self.x + c * local[:, 0] - s * local[:, 1], self.y + s * local[:, 0] + c * local[:, 1]
        ]
        return corners

    def contains(self, xs: NDArray[np.float64], ys: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Whether each world point lies inside (edges included)."""
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        dx, dy = xs - self.x, ys - self.y
        u, v = c * dx + s * dy, -s * dx + c * dy
        eps = 1e-9
        inside: NDArray[np.bool_] = (np.abs(u) <= self.length / 2.0 + eps) & (
            np.abs(v) <= self.width / 2.0 + eps
        )
        return inside

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], frame: Pose2D | None = None) -> Box:
        """One box of a scenario file; with ``frame`` (the start pose) its x, y and yaw are in
        that pose's frame (x forward, y left) and are resolved into the map here."""
        box = cls(
            name=str(data.get("name", "box")),
            x=float(data["x"]),
            y=float(data["y"]),
            length=float(data["length"]),
            width=float(data["width"]),
            yaw_deg=float(data.get("yaw_deg", 0.0)),
        )
        if frame is None:
            return box
        at = compose(frame, Pose2D(box.x, box.y, math.radians(box.yaw_deg)))
        return replace(box, x=at.x, y=at.y, yaw_deg=math.degrees(at.theta))

    def to_dict(self) -> dict[str, Any]:
        """The box as a scenario file writes it (map frame)."""
        return {
            "name": self.name,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "length": self.length,
            "width": self.width,
            "yaw_deg": round(self.yaw_deg, 3),
        }


def cast_grid(
    grid: Grid, x: float, y: float, angles: NDArray[np.float64], max_range: float
) -> NDArray[np.float64]:
    """Range to the first occupied cell along each world bearing from ``(x, y)``, exact to the
    cell boundary the ray enters by (a DDA over the cells, all rays at once); ``inf`` where none
    lies within ``max_range``. The cell the ray starts in is never a hit: a lidar does not see
    the cell it stands in, and on this grid that cell is the cart's own noise."""
    res = grid.resolution
    gx, gy = (x - grid.origin_x) / res, (y - grid.origin_y) / res
    dx, dy = np.cos(angles), np.sin(angles)
    n = len(angles)
    ix: NDArray[np.int64] = np.full(n, math.floor(gx), dtype=np.int64)
    iy: NDArray[np.int64] = np.full(n, math.floor(gy), dtype=np.int64)
    sx = np.where(dx > 0, 1, -1)
    sy = np.where(dy > 0, 1, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tdx = np.where(dx != 0, 1.0 / np.abs(dx), np.inf)
        tdy = np.where(dy != 0, 1.0 / np.abs(dy), np.inf)
        tmx = np.where(dx > 0, (ix + 1 - gx) * tdx, np.where(dx < 0, (gx - ix) * tdx, np.inf))
        tmy = np.where(dy > 0, (iy + 1 - gy) * tdy, np.where(dy < 0, (gy - iy) * tdy, np.inf))
    ranges = np.full(n, np.inf)
    active = np.ones(n, dtype=bool)
    limit = max_range / res
    occupied = grid.cells == OCCUPIED
    while active.any():
        step_x = tmx < tmy
        t_entry = np.where(step_x, tmx, tmy)
        ix = np.where(active & step_x, ix + sx, ix)
        iy = np.where(active & ~step_x, iy + sy, iy)
        tmx = np.where(active & step_x, tmx + tdx, tmx)
        tmy = np.where(active & ~step_x, tmy + tdy, tmy)
        inside = (ix >= 0) & (ix < grid.width) & (iy >= 0) & (iy < grid.height)
        occ = np.zeros(n, dtype=bool)
        occ[inside] = occupied[iy[inside], ix[inside]]
        within = t_entry <= limit
        hit = active & occ & within
        ranges[hit] = t_entry[hit] * res
        left = ~inside & (
            ((ix < 0) & (sx < 0))
            | ((ix >= grid.width) & (sx > 0))
            | ((iy < 0) & (sy < 0))
            | ((iy >= grid.height) & (sy > 0))
        )
        active &= ~hit & within & ~left
    return ranges


def cast_boxes(
    boxes: Sequence[Box], x: float, y: float, angles: NDArray[np.float64], max_range: float
) -> NDArray[np.float64]:
    """Range to the nearest box face along each world bearing from ``(x, y)`` (slab test in each
    box's own frame); ``inf`` for none within ``max_range``. A ray that starts inside a box
    sees nothing of it."""
    ranges: NDArray[np.float64] = np.full(len(angles), np.inf)
    dx, dy = np.cos(angles), np.sin(angles)
    for box in boxes:
        c, s = math.cos(math.radians(box.yaw_deg)), math.sin(math.radians(box.yaw_deg))
        px, py = c * (x - box.x) + s * (y - box.y), -s * (x - box.x) + c * (y - box.y)
        ux, uy = c * dx + s * dy, -s * dx + c * dy
        hx, hy = box.length / 2.0, box.width / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            t_in = np.full(len(angles), -np.inf)
            t_out = np.full(len(angles), np.inf)
            for p, u, h in ((px, ux, hx), (py, uy, hy)):
                t1, t2 = (-h - p) / u, (h - p) / u
                parallel = u == 0
                lo = np.where(parallel, np.where(abs(p) <= h, -np.inf, np.inf), np.minimum(t1, t2))
                hi = np.where(parallel, np.where(abs(p) <= h, np.inf, -np.inf), np.maximum(t1, t2))
                t_in, t_out = np.maximum(t_in, lo), np.minimum(t_out, hi)
        hit = (t_in <= t_out) & (t_in >= 0.0) & (t_in <= max_range)
        ranges = np.where(hit, np.minimum(ranges, t_in), ranges)
    return ranges


@dataclass(frozen=True)
class LidarModel:
    """The LD19 as mounted on this cart: the beams in the laser frame the board publishes, each
    beam's bearing in base_link through the mount's own rotation (upside down, turned by its
    calibrated offset), the posts' masked sectors, and the scan filter's hull box — returns
    inside it are cut to NaN on the board (robot.launch.py's LaserScanBoxFilter), so here too."""

    mount: LidarMount
    beams: int = LIDAR_BEAMS
    hull: Footprint = field(default_factory=Footprint)
    band_m: float = CONTACT_BAND_M

    @property
    def angle_increment(self) -> float:
        """The laser-frame step between beams, radians: the driver's bins span 0 to 2 pi with
        both ends included, so the first and the last beam look the same way (every tape's
        scan: 455 angles, first = last, step 2 pi / 454)."""
        return 2.0 * math.pi / (self.beams - 1)

    def laser_angles(self) -> NDArray[np.float64]:
        """Each beam's angle in the laser frame, from 0 counter-clockwise (the driver's)."""
        angles: NDArray[np.float64] = np.arange(self.beams, dtype=np.float64)
        return angles * self.angle_increment

    def bearings(self) -> NDArray[np.float64]:
        """Each beam's bearing in base_link, radians CCW from forward, through the mount's
        rotation (pepin.mounts.lidar_mount: the transform the board's launch publishes)."""
        r = lidar_mount(self.mount).rotation()
        a = self.laser_angles()
        # the rotation applied to (cos a, sin a, 0), written out: no BLAS for 455 two-vectors
        x = r[0, 0] * np.cos(a) + r[0, 1] * np.sin(a)
        y = r[1, 0] * np.cos(a) + r[1, 1] * np.sin(a)
        bearings: NDArray[np.float64] = np.arctan2(y, x)
        return bearings

    def masked(self) -> NDArray[np.bool_]:
        """The beams that run into the cart's posts (config/lidar.json masked_sectors_deg)."""
        return np.array(
            [self.mount.is_masked(self.mount.to_sensor_angle_deg(b)) for b in self.bearings()],
            dtype=bool,
        )

    def scan(self, grid: Grid, boxes: Sequence[Box], pose: Pose2D) -> NDArray[np.float64]:
        """One revolution's ranges in laser order from the cart at ``pose``: ``inf`` for no
        return within the range, NaN where the posts block the beam, the return falls inside the
        hull box or under the minimum range."""
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        ox = pose.x + c * self.mount.x_m - s * self.mount.y_m
        oy = pose.y + s * self.mount.x_m + c * self.mount.y_m
        bearings = self.bearings()
        world = bearings + pose.theta
        ranges = np.minimum(
            cast_grid(grid, ox, oy, world, self.mount.max_range_m),
            cast_boxes(boxes, ox, oy, world, self.mount.max_range_m),
        )
        box = hull_box(self.hull, self.band_m)
        with np.errstate(invalid="ignore"):
            px = self.mount.x_m + ranges * np.cos(bearings)
            py = self.mount.y_m + ranges * np.sin(bearings)
        on_hull = (
            (px >= box["min_x"])
            & (px <= box["max_x"])
            & (py >= box["min_y"])
            & (py <= box["max_y"])
        )
        cut = self.masked() | on_hull | (ranges < self.mount.min_range_m)
        return np.where(cut, np.nan, ranges)


def arc(pose: Pose2D, linear: float, angular: float, dt: float) -> Pose2D:
    """Where a unicycle at ``pose`` ends after ``dt`` seconds of a constant twist: the exact arc
    (a straight line when it does not turn)."""
    if abs(angular) < 1e-9:
        return Pose2D(
            pose.x + linear * math.cos(pose.theta) * dt,
            pose.y + linear * math.sin(pose.theta) * dt,
            pose.theta,
        )
    theta = pose.theta + angular * dt
    radius = linear / angular
    return Pose2D(
        pose.x + radius * (math.sin(theta) - math.sin(pose.theta)),
        pose.y - radius * (math.cos(theta) - math.cos(pose.theta)),
        wrap_angle(theta),
    )


@dataclass(frozen=True)
class Motion:
    """What one step of the base did: the twist it drove at (zero when blocked or stopped by the
    deadman), whether the hull was refused the step, and how far it went."""

    linear: float
    angular: float
    blocked: bool
    travel_m: float
    turn_rad: float


@dataclass
class UnicycleBase:
    """The base server as the planner sees it: the last /cmd_vel held until the next, each axis
    clamped to the base's own limit (pepin.base, config/base.json), and a stop once no command
    has arrived for ``deadman_s``. Nothing else: no acceleration limit (the velocity smoother
    owns that), no slip."""

    max_linear: float
    max_angular: float
    deadman_s: float = DEADMAN_S
    pose: Pose2D = field(default_factory=Pose2D)
    _linear: float = field(default=0.0, init=False, repr=False)
    _angular: float = field(default=0.0, init=False, repr=False)
    _commanded_at: float | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(cls, config: BaseConfig, pose: Pose2D | None = None) -> UnicycleBase:
        """The limits of ``config/base.json``."""
        return cls(config.max_speed_m_s, config.max_yaw_rate_rad_s, pose=pose or Pose2D())

    def command(self, linear: float, angular: float, now: float) -> None:
        """A /cmd_vel arrived at ``now`` (the world's clock)."""
        self._linear = max(-self.max_linear, min(self.max_linear, linear))
        self._angular = max(-self.max_angular, min(self.max_angular, angular))
        self._commanded_at = now

    def halt(self) -> None:
        """Forget the last command, as a base that was switched off and on again."""
        self._linear = self._angular = 0.0
        self._commanded_at = None

    def twist(self, now: float) -> tuple[float, float]:
        """The twist in force at ``now``: the last command, or zero past the deadman."""
        if self._commanded_at is None or now - self._commanded_at > self.deadman_s:
            return 0.0, 0.0
        return self._linear, self._angular

    def step(
        self, dt: float, now: float, blocked: Callable[[Pose2D, Pose2D], bool] | None = None
    ) -> Motion:
        """Drive ``dt`` seconds at the twist in force; ``blocked(old, new)`` may refuse the new
        pose (the hull ran into something), and then the cart stays where it is."""
        linear, angular = self.twist(now)
        if linear == 0.0 and angular == 0.0:
            return Motion(0.0, 0.0, False, 0.0, 0.0)
        new = arc(self.pose, linear, angular, dt)
        if blocked is not None and blocked(self.pose, new):
            return Motion(0.0, 0.0, True, 0.0, 0.0)
        self.pose = new
        return Motion(linear, angular, False, abs(linear) * dt, abs(angular) * dt)


def hull_samples(hull: Footprint, spacing: float = HULL_SAMPLE_M) -> NDArray[np.float64]:
    """Points covering the hull rectangle (edges included) in base_link, ``spacing`` apart."""
    xs = np.linspace(
        -hull.rear_m, hull.front_m, max(2, math.ceil((hull.front_m + hull.rear_m) / spacing) + 1)
    )
    ys = np.linspace(
        -hull.half_width_m,
        hull.half_width_m,
        max(2, math.ceil(2 * hull.half_width_m / spacing) + 1),
    )
    gx, gy = np.meshgrid(xs, ys)
    samples: NDArray[np.float64] = np.c_[gx.ravel(), gy.ravel()]
    return samples


def overlap(grid: Grid, boxes: Sequence[Box], pose: Pose2D, samples: NDArray[np.float64]) -> int:
    """How many hull samples at ``pose`` sit in an occupied cell or inside a box: 0 is clear."""
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    xs = pose.x + c * samples[:, 0] - s * samples[:, 1]
    ys = pose.y + s * samples[:, 0] + c * samples[:, 1]
    touching = grid.occupied_at(xs, ys)
    for box in boxes:
        touching |= box.contains(xs, ys)
    return int(np.count_nonzero(touching))


@dataclass
class Odometer:
    """The truth of a drive as only a simulator knows it: how far the hull really went and
    turned, how often it was refused a step against something, and for how long."""

    path_m: float = 0.0
    turn_rad: float = 0.0
    contacts: int = 0
    blocked_s: float = 0.0
    touched: dict[str, int] = field(default_factory=dict)  # contacts by what the hull met
    _was_blocked: bool = False

    def add(
        self, motion: Motion, dt: float, touching: Callable[[], list[str]] | None = None
    ) -> None:
        """Count one step; a contact is the first refused step of a run of them, and
        ``touching`` names what the hull met there (box names, "grid")."""
        self.path_m += motion.travel_m
        self.turn_rad += motion.turn_rad
        if motion.blocked:
            self.blocked_s += dt
            if not self._was_blocked:
                self.contacts += 1
                for name in touching() if touching is not None else ["?"]:
                    self.touched[name] = self.touched.get(name, 0) + 1
        self._was_blocked = motion.blocked


class SimWorld:
    """The cart in the room: base, grid, boxes, lidar and odometer, stepped by the caller's clock.

    The node around it (ros/sim/sim_world.py) publishes what this computes and answers
    :meth:`answer` on its control socket; everything a test needs is here.
    """

    def __init__(
        self,
        grid: Grid,
        base: UnicycleBase,
        lidar: LidarModel,
        hull: Footprint,
        boxes: Sequence[Box] = (),
    ) -> None:
        self.grid = grid
        self.base = base
        self.lidar = lidar
        self.hull = hull
        self.boxes: tuple[Box, ...] = tuple(boxes)
        self.odometer = Odometer()
        self._samples = hull_samples(hull)
        self._refused: Pose2D | None = None
        self._twist: tuple[float, float] = (0.0, 0.0)

    @property
    def pose(self) -> Pose2D:
        """The cart's true pose in the map (map -> odom is identity in the sim)."""
        return self.base.pose

    @property
    def twist(self) -> tuple[float, float]:
        """The twist the base drove at in the last step (what /odom reports)."""
        return self._twist

    def overlap(self, pose: Pose2D | None = None) -> int:
        """Hull samples in something at ``pose`` (default: where the cart is)."""
        return overlap(self.grid, self.boxes, pose or self.pose, self._samples)

    def _blocked(self, old: Pose2D, new: Pose2D) -> bool:
        """A step is refused when it puts more of the hull into something than there was: a cart
        that starts touching the grid's noise may still drive out of it, never deeper."""
        refused = self.overlap(new) > self.overlap(old)
        if refused:
            self._refused = new
        return refused

    def touching(self, pose: Pose2D | None = None) -> list[str]:
        """What the hull at ``pose`` (default: the last refused step) is in: box names, and
        "grid" for an occupied cell of the grid."""
        at = pose or self._refused or self.pose
        c, s = math.cos(at.theta), math.sin(at.theta)
        xs = at.x + c * self._samples[:, 0] - s * self._samples[:, 1]
        ys = at.y + s * self._samples[:, 0] + c * self._samples[:, 1]
        names = [box.name for box in self.boxes if box.contains(xs, ys).any()]
        return names + (["grid"] if self.grid.occupied_at(xs, ys).any() else [])

    def step(self, dt: float, now: float) -> Motion:
        """Advance the base ``dt`` seconds at ``now`` and count it."""
        motion = self.base.step(dt, now, self._blocked)
        self._twist = (motion.linear, motion.angular)
        self.odometer.add(motion, dt, self.touching)
        return motion

    def scan(self) -> NDArray[np.float64]:
        """The lidar's revolution from where the cart is (laser order, see LidarModel.scan)."""
        return self.lidar.scan(self.grid, self.boxes, self.pose)

    def place(self, pose: Pose2D) -> None:
        """Put the cart at ``pose``, standing still (the base forgets the last command)."""
        self.base.pose = pose
        self.base.halt()
        self._twist = (0.0, 0.0)

    def state(self, now: float) -> dict[str, Any]:
        """Where the cart is and what the odometer says, as the control socket answers it."""
        return {
            "event": "state",
            "t": round(now, 3),
            "x": round(self.pose.x, 4),
            "y": round(self.pose.y, 4),
            "yaw_deg": round(math.degrees(self.pose.theta), 2),
            "path_m": round(self.odometer.path_m, 4),
            "turn_deg": round(math.degrees(self.odometer.turn_rad), 1),
            "contacts": self.odometer.contacts,
            "touched": dict(self.odometer.touched),
            "blocked_s": round(self.odometer.blocked_s, 2),
            "overlap": self.overlap(),
            "boxes": [box.name for box in self.boxes],
        }

    def answer(self, request: Mapping[str, Any], now: float) -> dict[str, Any]:
        """One control-socket request: ``state``, ``place`` (x, y, yaw_deg) or ``boxes`` (a list
        of boxes in the map frame, replacing the current ones)."""
        cmd = request.get("cmd")
        try:
            if cmd == "state":
                return self.state(now)
            if cmd == "place":
                pose = Pose2D(
                    float(request["x"]),
                    float(request["y"]),
                    math.radians(float(request["yaw_deg"])),
                )
                self.place(pose)
                return {**self.state(now), "event": "placed"}
            if cmd == "boxes":
                self.boxes = tuple(Box.from_dict(b) for b in request.get("boxes", ()))
                return {**self.state(now), "event": "boxes"}
        except (KeyError, TypeError, ValueError) as error:
            return {"event": "error", "detail": f"{cmd}: {error}"}
        return {"event": "error", "detail": f"unknown command {cmd!r}: state, place or boxes"}


@dataclass(frozen=True)
class Target:
    """A pose a scenario names: a place of the world's book, or coordinates (in the map, or in
    the start pose's frame when ``frame`` is ``start``)."""

    place: str | None = None
    x: float = 0.0
    y: float = 0.0
    yaw_deg: float = 0.0
    frame: str = "map"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | str) -> Target:
        """``{place: NAME}``, ``NAME`` or ``{x, y, yaw_deg[, frame: start]}``."""
        if isinstance(data, str):
            return cls(place=data)
        if "place" in data:
            return cls(place=str(data["place"]))
        frame = str(data.get("frame", "map"))
        if frame not in ("map", "start"):
            raise ValueError(f"frame must be map or start, not {frame!r}")
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            yaw_deg=float(data.get("yaw_deg", 0.0)),
            frame=frame,
        )

    def resolve(self, places: Mapping[str, Pose2D], start: Pose2D | None = None) -> Pose2D:
        """The pose in the map; a place missing from the book or a start-frame target without a
        start is a ValueError."""
        if self.place is not None:
            if self.place not in places:
                raise ValueError(f"no place {self.place!r} in the world's book {sorted(places)}")
            return places[self.place]
        pose = Pose2D(self.x, self.y, math.radians(self.yaw_deg))
        if self.frame == "start":
            if start is None:
                raise ValueError("a target in the start frame needs a start")
            return compose(start, pose)
        return pose


@dataclass(frozen=True)
class Scenario:
    """A start, furniture and the goals driven from it, with what counts as slow.

    ``boxes`` in a file may be in the start pose's frame (``frame: start``), so a pocket is
    written as the drive meets it — "0.40 m ahead of the nose" — and resolved here.
    """

    name: str
    start: Target
    legs: tuple[Target, ...]
    boxes: tuple[Mapping[str, Any], ...] = ()
    note: str = ""
    world: str = "flat"
    timeout_s: float = 300.0
    slow_s: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Scenario:
        """A parsed scenario file."""
        legs = tuple(Target.from_dict(leg) for leg in data["legs"])
        if not legs:
            raise ValueError("a scenario drives at least one leg")
        slow = data.get("slow_s")
        return cls(
            name=str(data["name"]),
            start=Target.from_dict(data["start"]),
            legs=legs,
            boxes=tuple(data.get("boxes", ())),
            note=str(data.get("note", "")),
            world=str(data.get("world", "flat")),
            timeout_s=float(data.get("timeout_s", 300.0)),
            slow_s=None if slow is None else float(slow),
        )

    @classmethod
    def load(cls, path: Path) -> Scenario:
        """A scenario yaml."""
        import yaml

        return cls.from_dict(yaml.safe_load(path.read_text()))

    def start_pose(self, places: Mapping[str, Pose2D]) -> Pose2D:
        """Where the cart is placed before the first leg."""
        if self.start.frame == "start":
            raise ValueError("the start itself cannot be in the start frame")
        return self.start.resolve(places)

    def resolved_boxes(self, places: Mapping[str, Pose2D]) -> tuple[Box, ...]:
        """The furniture in the map frame."""
        start = self.start_pose(places)
        return tuple(
            Box.from_dict(b, start if b.get("frame", "map") == "start" else None)
            for b in self.boxes
        )


def places_from_payload(text: str) -> dict[str, Pose2D]:
    """The /places payload (pepin.places.places_json) as poses in the map."""
    heard = json.loads(text)
    return {
        name: Pose2D(float(e["x"]), float(e["y"]), math.radians(float(e["yaw_deg"])))
        for name, e in heard.items()
    }


@dataclass(frozen=True)
class LegScore:
    """One goal as the sim judges it: Nav2's verdict and time, the recoveries the behaviour tree
    ran, and the drive as it really happened — the hull's path against the straight line, the
    contacts, and where it stopped against the goal."""

    goal: str
    status: str
    seconds: float
    recoveries: int
    path_m: float
    straight_m: float
    contacts: int
    blocked_s: float
    error_m: float
    heading_error_deg: float
    touched: Mapping[str, int] = field(default_factory=dict)

    @property
    def detour(self) -> float:
        """Path over straight line (1.0 is a straight drive; NaN when the goal was where the
        cart stood)."""
        return self.path_m / self.straight_m if self.straight_m > 1e-6 else math.nan

    def line(self) -> str:
        """One line for the terminal."""
        return (
            f"SCORE {self.goal}: {self.status} in {self.seconds:.1f} s,"
            f" recoveries {self.recoveries},"
            f" path {self.path_m:.2f} m vs straight {self.straight_m:.2f} m (x{self.detour:.2f}),"
            f" contacts {self.contacts} ({self.blocked_s:.1f} s blocked{self._what()}), arrival"
            f" {self.error_m:.2f} m / {self.heading_error_deg:+.0f} deg off"
        )

    def _what(self) -> str:
        """``: cabinet 3, grid 1`` — what the contacts met, or nothing."""
        if not self.touched:
            return ""
        return ": " + ", ".join(f"{name} {n}" for name, n in sorted(self.touched.items()))

    def to_dict(self) -> dict[str, Any]:
        """The score as a JSON record."""
        return {
            "goal": self.goal,
            "status": self.status,
            "seconds": round(self.seconds, 2),
            "recoveries": self.recoveries,
            "path_m": round(self.path_m, 3),
            "straight_m": round(self.straight_m, 3),
            "detour": None if math.isnan(self.detour) else round(self.detour, 3),
            "contacts": self.contacts,
            "touched": dict(self.touched),
            "blocked_s": round(self.blocked_s, 2),
            "error_m": round(self.error_m, 3),
            "heading_error_deg": round(self.heading_error_deg, 1),
        }


def score_leg(
    goal: str,
    target: Pose2D,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    status: str,
    recoveries: int,
) -> LegScore:
    """A leg's score from the world's ``state`` before and after it (the odometer's deltas, the
    sim clock's seconds) and what the goal server said (status, recoveries)."""
    straight = math.hypot(target.x - float(before["x"]), target.y - float(before["y"]))
    error = math.hypot(target.x - float(after["x"]), target.y - float(after["y"]))
    heading = math.degrees(wrap_angle(math.radians(float(after["yaw_deg"])) - target.theta))
    return LegScore(
        goal=goal,
        status=status,
        seconds=float(after["t"]) - float(before["t"]),
        recoveries=recoveries,
        path_m=float(after["path_m"]) - float(before["path_m"]),
        straight_m=straight,
        contacts=int(after["contacts"]) - int(before["contacts"]),
        touched={
            name: n - int(before.get("touched", {}).get(name, 0))
            for name, n in after.get("touched", {}).items()
            if n > int(before.get("touched", {}).get(name, 0))
        },
        blocked_s=float(after["blocked_s"]) - float(before["blocked_s"]),
        error_m=error,
        heading_error_deg=heading,
    )
