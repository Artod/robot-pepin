"""One map for both sensors: the volume IS the map, and each sensor reads its own slice.

Until now a "known room" and an "unknown room" were two different machines. The tracker matched
scans against a frozen picture (``flat3.pgm`` through map_server) while the TSDF volume
(:mod:`pepin.tsdf`), built from the camera alone, was a pretty surface nobody localised against.
This module makes the volume the map itself:

* the lidar writes its own layer into it — every beam carves free space along its run and marks
  a surface at its return, at the height the ray itself is at (the beam plus or minus one
  voxel: the level sweep of a level body, and the climbing ray of a body leaning over a
  slipper, which at 5 degrees is 44 cm off the plane at 5 m), on its own weight channel;
* the camera keeps writing its band through :class:`pepin.tsdf.Tsdf`, and inside the rows of the
  lidar's own plane it is not allowed to overwrite what the lidar has spoken for: the network's
  depth is scale-uncertain, the lidar's returns are metric truth, and one bad law would otherwise
  push a wall half a metre in the one layer the cart drives by. Only that plane is handed back —
  where a leaning beam climbs into the camera's band it is an observation like any other, which
  the camera may correct;
* a horizontal band of the volume reads out as an occupancy grid (:class:`OccupancySlice`), so
  the lidar localises against the slice at its plane, the camera against its band
  (``camera_band_m``), and Nav2 gets the 2D projection as ``/map`` — one entity, three views;
* the whole thing snapshots to an ``.npz`` and loads back, so a known room is a loaded snapshot
  and an unknown one an empty volume. There is no mode switch, and "maturity" is not a flag: it
  is the weight in a cell, which grows with every observation and is what a slice thresholds on.

No ROS here (the message is returned as plain fields), no file formats beyond the snapshot and
the map_server pair the existing tooling already reads.

A loop closure moves the whole map, not the pose alone. Every frame in here was placed through
``map -> odom`` at its own stamp, so when a pose graph optimises and that edge jumps, the room
this volume drew is stale by exactly that difference: :meth:`WorldMap.shift` carries the content
rigidly through it (:class:`pepin.tsdf.PlanarShift`), and the node that paints decides when
(``follow_correction``, pepin_bringup.depth_fusion). Out of scope, still: a graph that corrects
its nodes differently from one another — that deformation cannot be a rigid move and wants the
frames replayed at their corrected poses, which is why the snapshot records every integration's
stamp, sensor and pose (:attr:`WorldMap.frames`). The measurements themselves stay in the run
tape, which is where they already live; the snapshot is a warm cache, never the only copy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from pepin.depth import Intrinsics
from pepin.tsdf import (
    BLEND,
    NEAREST,
    Array,
    Float32,
    GridSpec,
    PlanarShift,
    RigidPose,
    ShiftedColumns,
    Tsdf,
    Uint8,
)

if TYPE_CHECKING:
    from pepin.mapping import OccupancyGrid

Int8 = npt.NDArray[np.int8]
Ints = npt.NDArray[np.intp]

# The three values a nav_msgs/OccupancyGrid carries, and the trinary pgm map_server writes.
FREE, OCCUPIED, UNKNOWN = 0, 100, -1
PGM_FREE, PGM_OCCUPIED, PGM_UNKNOWN = 254, 0, 205

LIDAR, CAMERA = "lidar", "camera"  # the sensors a snapshot's frame list names
SNAPSHOT_VERSION = 1


@dataclass(frozen=True)
class PlanarMount:
    """Where a ray sensor sits on the cart (base_link metres) and how far it may be believed."""

    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0  # the sensor's plane above the floor: the height of its layer in the volume
    min_range_m: float = 0.05
    max_range_m: float = 12.0

    @classmethod
    def from_config(cls, path: str | Path) -> PlanarMount:
        """The lidar's mount from ``config/lidar.json`` — the height too, which is calibrated
        and must never be typed into code."""
        from pepin.lidar import LidarMount

        m = LidarMount.from_json(path)
        return cls(m.x_m, m.y_m, m.z_m, m.min_range_m, m.max_range_m)


@dataclass(frozen=True)
class LidarLaw:
    """How one beam writes into the volume: what a return is worth, what a crossing is worth,
    how thick the sensor's layer is, and how confident the layer may ever get.

    A crossing weighs less than a return on purpose. One return says "a surface is exactly
    here"; one crossing says only "nothing on this line", and a beam leaves two hundred
    crossings for every return — at equal weights the free run of the neighbouring beams would
    rub out a wall seen edge-on. ``max_weight`` is the lidar's own cap, well below the volume's:
    a cell the lidar owns saturates in a couple of seconds and a chair that moves is cleared in
    a couple more, which is what keeps the map alive instead of frozen.

    ``no_return_free`` is the open door: a beam that came back with nothing carves free space
    out to the mount's reach and marks nothing there. It is off by default because a beam comes
    back with nothing from a mirror, from a black chair leg and from anything closer than the
    sensor's minimum too — carrying any of those out to twelve metres would rub out the wall
    behind them.
    """

    hit_weight: float = 1.0
    free_weight: float = 0.3
    max_weight: float = 20.0
    layer_half_m: float = 0.05  # the plane plus or minus this: one voxel either side
    step_voxels: float = 0.5  # how finely a beam is sampled, in voxels
    no_return_free: bool = False  # a beam with no return carves to the reach, or writes nothing


@dataclass(frozen=True)
class SliceLaw:
    """When a column of the volume counts as occupied, free or unknown.

    The column is read over the band, and only voxels carrying ``min_weight`` observations speak
    (distances are in truncation units, as the field stores them — 1.0 is a whole truncation
    away from any surface). The column is OCCUPIED when some voxel comes within
    ``occupied_below`` of the zero crossing, half a voxel by default: that is the surface
    itself. It is FREE when even the smallest distance in it stays above ``free_above`` — the
    nearest thing in the column decides, so one shelf leg makes the whole column not-free. In
    between lies the truncation halo of a surface, and a column there is left unknown rather
    than called open floor right in front of a wall. A column deep inside a wall, where the
    field is negative and never crosses, is unknown too: nothing ever observed in there.
    """

    min_weight: float = 2.0  # observations a voxel needs before it may speak at all
    free_above: float = 0.5  # half a truncation, one voxel, from anything
    occupied_below: float | None = None  # default: half a voxel, in truncation units

    def occupied_t(self, spec: GridSpec) -> float:
        """The occupied threshold in the field's own units for ``spec``: half a voxel, which is
        "the surface falls inside this cell". A crossing that falls exactly on a cell boundary
        puts both neighbours exactly half a voxel away, and the field is float32: without the
        last bit of slack such a wall would be in no cell at all. The cell holding a return is
        anchored on the surface by the return's own sample (:meth:`WorldMap._beam_cells`), so
        the threshold does not have to absorb where the ray sampling happened to land."""
        if self.occupied_below is not None:
            return self.occupied_below
        return 0.5 * spec.voxel_m / spec.truncation_m + 1e-5


@dataclass(frozen=True)
class OccupancyGridFields:
    """A nav_msgs/OccupancyGrid without ROS: what a node copies into the message.

    ``data`` is row-major from the origin corner, one int8 per cell (0 free, 100 occupied,
    -1 unknown), exactly as ``info.width``/``info.height`` describe it.
    """

    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    data: Int8

    def as_list(self) -> list[int]:
        """The cells as plain ints, the form rclpy accepts for an int8[] field."""
        return [int(v) for v in self.data]


@dataclass(frozen=True)
class OccupancySlice:
    """One horizontal band of the volume flattened to an occupancy grid, rows y, columns x.

    ``values`` is the trinary picture (0 free, 100 occupied, -1 unknown); ``sdf`` keeps the
    column's smallest signed distance and ``weight`` its largest weight, so a caller can see
    how mature a cell is instead of only what it was thresholded into.
    """

    values: Int8
    sdf: Float32
    weight: Float32
    resolution_m: float
    origin: tuple[float, float]
    band_m: tuple[float, float]

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, columns) = (cells along y, cells along x)."""
        rows, cols = self.values.shape
        return int(rows), int(cols)

    def counts(self) -> dict[str, int]:
        """How many cells are free, occupied, unknown and known (free or occupied)."""
        free = int(np.count_nonzero(self.values == FREE))
        occupied = int(np.count_nonzero(self.values == OCCUPIED))
        unknown = int(np.count_nonzero(self.values == UNKNOWN))
        return {"free": free, "occupied": occupied, "unknown": unknown, "known": free + occupied}

    def message_fields(self) -> OccupancyGridFields:
        """The slice as the fields of a nav_msgs/OccupancyGrid."""
        rows, cols = self.shape
        return OccupancyGridFields(
            resolution=self.resolution_m,
            width=cols,
            height=rows,
            origin_x=self.origin[0],
            origin_y=self.origin[1],
            data=np.ascontiguousarray(self.values).reshape(-1),
        )

    def to_log_odds(self) -> OccupancyGrid:
        """The slice as the :class:`pepin.mapping.OccupancyGrid` the tracker matches against —
        the same three values ``/map`` carries, turned into log-odds the same way the node's
        ``grid_from_msg`` turns them (occupied +4, free -4, unknown 0)."""
        from pepin.mapping import FREE_LOG_ODDS, OCCUPIED_LOG_ODDS, OccupancyGrid
        from pepin.mapping import GridSpec as MapSpec

        rows, cols = self.shape
        grid = OccupancyGrid(
            MapSpec(
                self.resolution_m,
                self.origin[0],
                self.origin[1],
                cols * self.resolution_m,
                rows * self.resolution_m,
            )
        )
        grid.log_odds[:] = np.where(
            self.values == OCCUPIED,
            OCCUPIED_LOG_ODDS,
            np.where(self.values == FREE, FREE_LOG_ODDS, 0.0),
        )
        grid.version += 1
        return grid

    def to_pgm(self) -> bytes:
        """The slice as a binary trinary pgm, map_server's own byte order (top row first)."""
        rows, cols = self.shape
        pixels = np.where(
            self.values == OCCUPIED,
            PGM_OCCUPIED,
            np.where(self.values == FREE, PGM_FREE, PGM_UNKNOWN),
        ).astype(np.uint8)
        return f"P5\n{cols} {rows}\n255\n".encode() + pixels[::-1].tobytes()

    def to_yaml(self, image: str) -> str:
        """The map_server metadata for :meth:`to_pgm`, field for field as ros/maps/*.yaml."""
        return (
            f"image: {image}\n"
            "mode: trinary\n"
            f"resolution: {self.resolution_m:.3f}\n"
            f"origin: [{self.origin[0]:.3f}, {self.origin[1]:.3f}, 0]\n"
            "negate: 0\n"
            "occupied_thresh: 0.65\n"
            "free_thresh: 0.196\n"
        )


class WorldMap:
    """The one map: a TSDF volume both sensors write into, sliced per sensor and per consumer.

    Holds a :class:`pepin.tsdf.Tsdf` (composition, never inheritance: the volume's fusion law is
    its own business) plus one extra channel — how much of each voxel's weight the lidar put
    there, which is what makes the lidar's layer defensible against the camera.
    """

    def __init__(
        self,
        spec: GridSpec,
        mount: PlanarMount | None = None,
        law: LidarLaw | None = None,
        protect_lidar_layer: bool = True,
    ) -> None:
        self.spec = spec
        self.mount = mount if mount is not None else PlanarMount()
        self.law = law if law is not None else LidarLaw()
        self.protect_lidar_layer = protect_lidar_layer
        self.volume = Tsdf(spec)
        self.lidar_weight: Float32 = np.zeros(spec.shape, dtype=np.float32)
        self.lidar_plane_m: float = self.mount.z_m
        self.stamp: float = 0.0  # the newest observation in the volume (the sensors' clock)
        # Every integration: (stamp, sensor, 3x4 pose). The measurements stay in the run tape;
        # this is the index a loop closure would replay them by (see the module docstring).
        self.frames: list[tuple[float, str, Array]] = []
        self._rows: tuple[int, int] | None = None  # z rows the lidar's plane has ever swept

    @property
    def protected_rows(self) -> tuple[int, int] | None:
        """The z voxel rows the camera hands back to the lidar (:meth:`integrate_depth`): the
        layer the lidar's plane has swept, or ``None`` while no scan has been written."""
        return self._rows

    # ---- integration ---------------------------------------------------------------------
    def integrate_scan(
        self,
        angles: Array,
        ranges: Array,
        pose_base_in_map: RigidPose,
        mount: PlanarMount | None = None,
        stamp: float | None = None,
    ) -> int:
        """Write one lidar revolution into the volume along the beams' own rays; returns how
        many voxels were touched.

        ``angles`` are robot-frame bearings (radians, CCW from forward — what ``pepin.lidar``
        and the run tape carry) and ``ranges`` the metres along them, NaN where there was no
        return. ``pose_base_in_map`` is the body's whole rigid pose, lean and all: the beams
        leave the mount on the leaning cart and climb or dive with it, so their samples land in
        the voxels the ray really passes through rather than in the rows of a level plane. On a
        body tipped 5 degrees a return at 5 m is 44 cm off that plane — a tabletop, or the air
        under a seat, written as a wall in the one layer the cart drives by.

        A beam carves free space along its whole run and marks a surface at its end; a beam
        whose range is beyond the mount's reach carves free space out to that reach and marks
        nothing, because the reach is how far this sensor may be believed; a beam that leaves
        the volume's height is carved as far as it stays inside. A beam with no return at all
        (NaN) writes nothing, unless ``LidarLaw.no_return_free`` is on — then it too carves to
        the reach and marks nothing, which is how an open door stays open. Every sample is
        written on the lidar's weight channel, at its own height and one voxel either side, and
        one scan speaks at most once about a voxel however many of its beams cross it.
        """
        mount = mount if mount is not None else self.mount
        if self._layer_rows(pose_base_in_map, mount) is None:
            return 0  # the cart's own plane is not in this volume
        cells = self._beam_cells(angles, ranges, pose_base_in_map, mount)
        if cells is None:
            return 0
        at, w_obs, t_obs = cells
        w_old = self.volume.weight[at]
        w_new = w_old + w_obs
        self.volume.sdf[at] = (self.volume.sdf[at] * w_old + t_obs * w_obs) / w_new
        # the lidar's own cap, below the volume's: a cell it owns stays movable
        self.volume.weight[at] = np.minimum(self.law.max_weight, w_new)
        self.lidar_weight[at] = np.minimum(self.law.max_weight, self.lidar_weight[at] + w_obs)
        self._note(stamp, LIDAR, pose_base_in_map)
        return int(at[0].size)

    def integrate_depth(
        self,
        depth: Array | Float32,
        rgb: Uint8 | None,
        intr: Intrinsics,
        pose: RigidPose,
        stamp: float | None = None,
    ) -> int:
        """Fuse one depth frame the way :meth:`pepin.tsdf.Tsdf.integrate` does, then hand the
        lidar's layer back to the lidar: inside that layer the voxels the lidar has spoken for
        keep the field and the weight they had. Returns the voxels the frame touched.

        The layer is the rows the lidar's *plane* sweeps (:meth:`_widen_rows`), never the rows a
        leaning beam happened to climb into: a beam that tips into the camera's band writes there
        like any other observation, and the camera may write over it on the next frame. Only the
        layer the cart drives by is defended, and only where the lidar actually spoke."""
        rows = self._rows if self.protect_lidar_layer else None
        if rows is None:
            touched = self.volume.integrate(depth, rgb, intr, pose)
            self._note(stamp, CAMERA, pose)
            return touched
        lo, hi = rows
        keep_sdf = self.volume.sdf[:, :, lo:hi].copy()
        keep_weight = self.volume.weight[:, :, lo:hi].copy()
        touched = self.volume.integrate(depth, rgb, intr, pose)
        owned = self.lidar_weight[:, :, lo:hi] > 0.0
        np.copyto(self.volume.sdf[:, :, lo:hi], keep_sdf, where=owned)
        np.copyto(self.volume.weight[:, :, lo:hi], keep_weight, where=owned)
        self._note(stamp, CAMERA, pose)
        return touched

    def shift(self, shift: PlanarShift, law: str = BLEND) -> None:
        """Move the whole map by a rigid planar ``shift``: the volume, the lidar's own weight
        channel and the index of the frames that painted it. ``law`` is how the content is
        resampled (:meth:`pepin.tsdf.Tsdf.shift`): the fusion's weighted average, or the
        nearest source column.

        This is how the map follows a graph correction. Every frame in here was placed through
        ``map -> odom`` at its own stamp; when the graph optimises and that edge jumps, the same
        observations belong ``shift`` away — the room is not re-measured, it is carried, as one
        body. The lidar's weight travels through the very same column map as the field, so the
        layer the camera must hand back stays the layer the lidar wrote, cell for cell.

        What does not move: the grid (the box and its lattice are the frame everything is cut
        on), the plane and the rows the lidar's layer occupies (a planar move has no z in it),
        and the stamp. The move also re-seeds nothing and re-publishes nothing by itself — the
        trackers follow ``map -> odom`` already, and the next slice out of this volume is simply
        the moved one.
        """
        if shift.nothing:
            return
        columns = self.volume.shift(shift, law)
        self.lidar_weight = self._moved_claim(columns, law)
        self.frames = [
            (stamp, sensor, self._shifted_frame(shift, pose)) for stamp, sensor, pose in self.frames
        ]

    def _moved_claim(self, columns: ShiftedColumns, law: str = BLEND) -> Float32:
        """The lidar's weight channel after the move: its magnitude by the same weighted average
        as the field, its EXTENT by the majority of what the cell was made of.

        Where the lidar has spoken is a fact, not a quantity, and a blend of a non-negative
        channel spreads a fact: every cell with one owned column among its four sources comes
        back owned, so the claim grows by a ring at every move and :meth:`integrate_depth` then
        hands the camera's work to a lidar that never wrote there. Measured on the synthetic box
        (scratch/follow_refute.py, 2026-09-14): 7058 owned cells of the layer became 8305 over
        ten corrections of 10 cm / 3 deg — +18 % of room nobody swept, and monotone, because
        ``lidar_weight`` never decays. So the claim is carried by the same bilinear vote as the
        field and kept where that vote is more than half the lidar's: the count then stands
        (7058 -> 7060 after one move, and what it loses over ten is the room walking off the
        grid), and every occupied cell of a moved seeded wall is still the lidar's.
        """
        if law == NEAREST:
            return columns.nearest(self.lidar_weight)  # a nearest move cannot spread anything
        moved = columns.blend(self.lidar_weight)
        share = columns.blend((self.lidar_weight > 0.0).astype(np.float32))
        moved[share < 0.5] = 0.0
        return moved

    @staticmethod
    def _shifted_frame(shift: PlanarShift, pose: Array) -> Array:
        """One row of the frame index (a 3x4 map pose) after the move."""
        moved = shift.applied_to(RigidPose(pose[:, :3], pose[:, 3]))
        return np.hstack([moved.rotation, moved.translation.reshape(3, 1)])

    def _note(self, stamp: float | None, sensor: str, pose: RigidPose) -> None:
        """Record that a frame went in: its stamp, its sensor and where it was placed."""
        if stamp is None:
            return
        self.stamp = max(self.stamp, stamp)
        self.frames.append(
            (stamp, sensor, np.hstack([pose.rotation, pose.translation.reshape(3, 1)]))
        )

    def _layer_rows(self, pose: RigidPose, mount: PlanarMount) -> tuple[int, int] | None:
        """The z voxel rows of the sensor's layer at ``pose`` as a level body would sweep it,
        remembered as the lidar's own; ``None`` when that plane misses the volume.

        This is the plane the layer is *read out* at (:meth:`lidar_slice`, ``/map``) and the
        layer the camera hands back, and it must not wobble with the body — a leaning cart
        still drives on the same floor. Where the beams of a leaning body actually wrote is
        another question, and not one ownership answers.
        """
        plane = float(pose.translation[2]) + mount.z_m
        rows = self._plane_rows(plane)
        if rows is None:
            return None
        self.lidar_plane_m = plane
        self._widen_rows(*rows)
        return rows

    def _plane_rows(self, plane_m: float) -> tuple[int, int] | None:
        """The z voxel rows a level sweep of ``plane_m`` occupies — the plane plus or minus the
        layer's half-thickness, clipped to the volume; ``None`` when that plane misses it."""
        origin_z, voxel, nz = self.spec.origin[2], self.spec.voxel_m, self.spec.shape[2]
        lo = math.floor((plane_m - self.law.layer_half_m - origin_z) / voxel)
        hi = math.floor((plane_m + self.law.layer_half_m - origin_z) / voxel) + 1
        lo, hi = max(lo, 0), min(hi, nz)
        return (lo, hi) if hi > lo else None

    def _widen_rows(self, lo: int, hi: int) -> None:
        """Remember that the lidar's plane has swept these z rows: the band the camera's fusion
        hands back to it (:meth:`integrate_depth`).

        The band is the plane's own layer and grows only with the plane (the cart's z on a ramp),
        never with a leaning beam's climb: a tip of 3 degrees puts an 8 m ray 40 cm off the plane,
        and a band that grew with it would hand the lidar half the camera's band for good — the
        claim outlives the tip, because ``lidar_weight`` never decays. Inside the band the
        protection is still per cell (``lidar_weight > 0``), so it defends nothing the lidar
        never wrote there.
        """
        self._rows = (
            (lo, hi) if self._rows is None else (min(self._rows[0], lo), max(self._rows[1], hi))
        )

    def _beam_cells(
        self, angles: Array, ranges: Array, pose: RigidPose, mount: PlanarMount
    ) -> tuple[tuple[Ints, Ints, Ints], Array, Array] | None:
        """The scan as (voxel index triple, weight, signed distance in truncation units), one
        entry per voxel the scan touches; ``None`` when no beam reaches the volume.

        The beam is sampled on a ladder of rungs, plus one sample AT the return itself. That
        last one is what puts the wall in the map: the rungs land where the ladder's spacing
        puts them, so a wall lying on a cell boundary can have no rung within half a voxel of
        it in either neighbour, and the cell's averaged distance then reads as "no surface
        here" on both sides — the wall falls out of the map, and more viewpoints average it
        away further (measured on the synthetic box: 18 % of such a wall survived nine
        viewpoints; with the return sampled, 95-100 % at every offset through a cell).

        A sample's weight slides from the free weight far along the beam to the hit weight at
        the return: the samples near the surface are the ones carrying its position, and the
        return's own sample carries the full hit weight at distance zero. Each sample is spread
        over the layer's half-thickness about its own height, so a climbing ray keeps the
        thickness a level sweep always had.
        """
        s = self.spec
        r = np.asarray(ranges, dtype=float)
        a = np.asarray(angles, dtype=float)
        finite = np.isfinite(r)
        # a beam with no return is a beam to the reach with nothing at its end, or no beam at all
        empty = ~finite if self.law.no_return_free else np.zeros(r.shape, dtype=bool)
        valid = (finite & (r >= mount.min_range_m)) | empty
        if not np.any(valid):
            return None
        seen = r[valid]
        hit = np.isfinite(seen) & (seen <= mount.max_range_m)
        reach = np.where(hit, seen, mount.max_range_m)
        step = s.voxel_m * self.law.step_voxels
        limit = reach + np.where(hit, s.truncation_m, 0.0)
        ladder = np.arange(1, math.ceil(float(limit.max()) / step) + 1) * step
        inside = ladder[None, :] <= limit[:, None]
        if not np.any(inside):
            return None
        along = np.broadcast_to(ladder[None, :], inside.shape)[inside]
        beam = np.broadcast_to(np.arange(reach.size)[:, None], inside.shape)[inside]
        # the return itself, sampled exactly where it came back: distance zero, full hit weight
        along = np.concatenate([along, reach[hit]])
        beam = np.concatenate([beam, np.flatnonzero(hit)])
        x, y, z = self._sample_points(a[valid][beam], along, pose, mount)
        # a beam that ran out of reach carries no surface: its whole run is free space, and a
        # distance that shrinks toward its end would otherwise read as a wall at the reach
        t = np.where(hit[beam], np.minimum(1.0, (reach[beam] - along) / s.truncation_m), 1.0)
        w = self.law.free_weight + (self.law.hit_weight - self.law.free_weight) * (
            1.0 - np.minimum(1.0, np.abs(t))
        )
        nx, ny, nz = s.shape
        ix = np.floor((x - s.origin[0]) / s.voxel_m).astype(int)
        iy = np.floor((y - s.origin[1]) / s.voxel_m).astype(int)
        half = self.law.layer_half_m
        lo = np.clip(np.floor((z - half - s.origin[2]) / s.voxel_m).astype(int), 0, nz)
        hi = np.clip(np.floor((z + half - s.origin[2]) / s.voxel_m).astype(int) + 1, 0, nz)
        on = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (hi > lo)
        if not np.any(on):
            return None
        ix, iy, lo, hi, w, t = ix[on], iy[on], lo[on], hi[on], w[on], t[on]
        rows = hi - lo  # how many voxels of the layer this sample is spread over
        sample = np.repeat(np.arange(rows.size), rows)
        iz = lo[sample] + (np.arange(int(rows.sum())) - np.repeat(np.cumsum(rows) - rows, rows))
        flat = (ix[sample] * ny + iy[sample]) * nz + iz
        cell, of_cell = np.unique(flat, return_inverse=True)
        sum_w = np.bincount(of_cell, weights=w[sample], minlength=cell.size)
        sum_wt = np.bincount(of_cell, weights=(w * t)[sample], minlength=cell.size)
        spoke = sum_w > 0.0
        cell, sum_w, sum_wt = cell[spoke], sum_w[spoke], sum_wt[spoke]
        t_obs = sum_wt / sum_w
        # one scan is one observation of a voxel, however many beams crossed it
        w_obs = np.minimum(sum_w, self.law.hit_weight)
        column, iz = np.divmod(cell, nz)
        ix, iy = np.divmod(column, ny)
        return (ix, iy, iz), w_obs, t_obs

    def _sample_points(
        self, bearings: Array, along: Array, pose: RigidPose, mount: PlanarMount
    ) -> tuple[Array, Array, Array]:
        """Where the samples of the beams sit in the map: one (x, y, z) per (sensor-frame
        bearing, metres along the beam) pair.

        A body standing level sweeps its sensor's plane, and that is the arithmetic this has
        always been — kept exactly, so nothing but the lean itself can move a level run's map.
        A leaning body turns the whole ray through the pose's own rotation: the mount rides up
        or down with the body and every beam leaves it along ``R @ (cos b, sin b, 0)``, which
        is the ray whose height grows as ``r sin(lean)``.
        """
        m = np.asarray(pose.rotation, dtype=float)
        if bool(m[2, 0] == 0.0 and m[2, 1] == 0.0 and m[2, 2] == 1.0):
            yaw = math.atan2(float(m[1, 0]), float(m[0, 0]))
            ox = float(pose.translation[0]) + math.cos(yaw) * mount.x_m - math.sin(yaw) * mount.y_m
            oy = float(pose.translation[1]) + math.sin(yaw) * mount.x_m + math.cos(yaw) * mount.y_m
            heading = bearings + yaw
            return (
                ox + np.cos(heading) * along,
                oy + np.sin(heading) * along,
                np.full(along.shape, float(pose.translation[2]) + mount.z_m),
            )
        origin = np.asarray(pose.translation, dtype=float) + m @ np.array(
            [mount.x_m, mount.y_m, mount.z_m], dtype=float
        )
        cos_b, sin_b = np.cos(bearings), np.sin(bearings)
        return (
            origin[0] + (m[0, 0] * cos_b + m[0, 1] * sin_b) * along,
            origin[1] + (m[1, 0] * cos_b + m[1, 1] * sin_b) * along,
            origin[2] + (m[2, 0] * cos_b + m[2, 1] * sin_b) * along,
        )

    # ---- readout -------------------------------------------------------------------------
    def slice(self, z_lo: float, z_hi: float, law: SliceLaw | None = None) -> OccupancySlice:
        """The band ``z_lo``..``z_hi`` of the volume as an occupancy grid over its whole x-y
        footprint: the nearest surface in each column decides, by :class:`SliceLaw`."""
        law = law if law is not None else SliceLaw()
        s = self.spec
        nx, ny, nz = s.shape
        lo = max(0, math.floor((z_lo - s.origin[2]) / s.voxel_m))
        hi = min(nz, math.ceil((z_hi - s.origin[2]) / s.voxel_m))
        hi = max(hi, lo + 1) if lo < nz else nz
        sdf = self.volume.sdf[:, :, lo:hi]
        weight = self.volume.weight[:, :, lo:hi]
        known = weight >= law.min_weight
        nearest = np.where(known, sdf, np.inf).min(axis=2)  # how open the column is
        crossing = np.where(known, np.abs(sdf), np.inf).min(axis=2)  # how near a surface it is
        maturity = np.where(known, weight, 0.0).max(axis=2)
        values = np.full((nx, ny), UNKNOWN, dtype=np.int8)
        seen = np.isfinite(nearest)
        values[seen & (nearest >= law.free_above)] = FREE
        values[seen & (crossing <= law.occupied_t(s))] = OCCUPIED
        return OccupancySlice(
            values=np.ascontiguousarray(values.T),
            sdf=np.ascontiguousarray(nearest.T.astype(np.float32)),
            weight=np.ascontiguousarray(maturity.T.astype(np.float32)),
            resolution_m=s.voxel_m,
            origin=(s.origin[0], s.origin[1]),
            band_m=(s.origin[2] + lo * s.voxel_m, s.origin[2] + hi * s.voxel_m),
        )

    def lidar_slice(self, law: SliceLaw | None = None) -> OccupancySlice:
        """The layer at the lidar's plane: what the lidar localises against, and what goes out
        as ``/map``. The plane is the mount's height from ``config/lidar.json`` until a scan
        says otherwise (the cart's own z, should the floor ever not be zero)."""
        half = self.law.layer_half_m
        return self.slice(self.lidar_plane_m - half, self.lidar_plane_m + half, law)

    def camera_band_slice(self, law: SliceLaw | None = None) -> OccupancySlice:
        """The band the camera speaks for (``camera_band_m`` of config/fusion.json, the band
        /depth_scan marks in): what the camera sources localise against — seats and tabletops
        the lidar's plane never sees are in here and in no other view of the map.

        The band a MATCHER is handed is cut with its own, far higher ``min_weight`` than the
        one a picture is cut with (:meth:`hardness`): a cell the camera painted two frames ago
        at the pose it is now asking about is not evidence about that pose.
        """
        lo, hi = self.spec.camera_band_m
        return self.slice(lo, hi, law)

    def hardness(self, law: SliceLaw, floor: SliceLaw | None = None) -> dict[str, float]:
        """How much of the camera's band is hard enough to localise on: the occupied cells the
        matcher's cut (``law``) keeps, the occupied cells the map's own, lenient cut (``floor``,
        the default law where the caller has none) finds, and the share of the second the first
        keeps.

        A share near 1 means the band is as hard as it is big — every wall in it has been
        integrated for seconds from more than one frame. A share near 0 means the matcher would
        be handed a band the camera painted a moment ago at the very pose it is asking about,
        which is how camera-only localisation walked away in 20-33 cm steps (2026-09-13) — or,
        on a volume just seeded from a saved map, a band with nothing in it at all:
        :meth:`seed_from_grid` writes one weight per cell and a matcher cut above it sees none
        of them.
        """
        hard = self.camera_band_slice(law).counts()
        soft = self.camera_band_slice(floor if floor is not None else SliceLaw()).counts()
        share = hard["occupied"] / soft["occupied"] if soft["occupied"] else 0.0
        return {
            "occupied": float(hard["occupied"]),
            "occupied_floor": float(soft["occupied"]),
            "share": float(share),
            "min_weight": float(law.min_weight),
        }

    def to_occupancy_grid_message_fields(
        self, slice_: OccupancySlice | None = None
    ) -> OccupancyGridFields:
        """The fields of the ``/map`` message a node publishes: the lidar's slice by default."""
        return (slice_ if slice_ is not None else self.lidar_slice()).message_fields()

    def export_pgm_yaml(self, path: str | Path, slice_: OccupancySlice | None = None) -> Path:
        """Write ``path.pgm`` + ``path.yaml`` in map_server's format, the pair ros/maps/*.yaml
        already is, so map_server, ``pepin.mapping.grid_from_pgm`` and the operator scripts read
        the volume with no new reader. Returns the yaml's path."""
        base = Path(path).with_suffix("")
        view = slice_ if slice_ is not None else self.lidar_slice()
        pgm = base.with_suffix(".pgm")
        pgm.write_bytes(view.to_pgm())
        yaml_path = base.with_suffix(".yaml")
        yaml_path.write_text(view.to_yaml(pgm.name))
        return yaml_path

    def maturity(self) -> dict[str, float]:
        """How grown-up the volume is, for a report line: voxels either sensor has spoken for,
        voxels the lidar owns, and the weight in them."""
        known = self.volume.weight > 0.0
        lidar = self.lidar_weight > 0.0
        weights = self.volume.weight[known]
        return {
            "voxels": float(np.count_nonzero(known)),
            "lidar_voxels": float(np.count_nonzero(lidar)),
            "mean_weight": float(weights.mean()) if weights.size else 0.0,
            "max_weight": float(weights.max()) if weights.size else 0.0,
            "frames": float(len(self.frames)),
            "plane_m": self.lidar_plane_m,
        }

    def report(self, lidar_law: SliceLaw | None = None, camera_law: SliceLaw | None = None) -> str:
        """One phrase for the node's report line: both slices as the node actually cuts them,
        how hard the camera's band is, and how mature the volume is.

        The laws are the live ones (the node's flags), not the defaults: a report that shows a
        slice nobody publishes says nothing about what the matchers were handed.
        """
        stats = self.maturity()
        lidar = self.lidar_slice(lidar_law).counts()
        law = camera_law if camera_law is not None else SliceLaw()
        camera = self.camera_band_slice(law).counts()
        hard = self.hardness(law, lidar_law)
        return (
            f"lidar slice {lidar['occupied']} occupied / {lidar['free']} free /"
            f" {lidar['unknown']} unknown, camera band {camera['occupied']} occupied /"
            f" {camera['free']} free (hard {hard['share'] * 100:.0f} % of"
            f" {hard['occupied_floor']:.0f} above weight {hard['min_weight']:.0f}),"
            f" {stats['voxels']:.0f} voxels"
            f" ({stats['lidar_voxels']:.0f} the lidar's, mean weight {stats['mean_weight']:.1f})"
        )

    # ---- the snapshot --------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write the whole volume to one ``.npz``: the field, both weight channels, the colour,
        the grid it lives on, the newest observation's stamp and the frame index a later
        re-fusion would replay. Returns the path written."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frames = (
            np.array([[f[0], f[1] == LIDAR, *f[2].reshape(-1)] for f in self.frames], dtype=float)
            if self.frames
            else np.zeros((0, 14))
        )
        np.savez_compressed(
            out,
            version=np.array(SNAPSHOT_VERSION),
            spec=np.frombuffer(json.dumps(self._spec_json()).encode(), dtype=np.uint8),
            sdf=self.volume.sdf,
            weight=self.volume.weight,
            lidar_weight=self.lidar_weight,
            rgb=self.volume.rgb,
            colour_weight=self.volume.colour_weight,
            stamp=np.array(self.stamp),
            plane_m=np.array(self.lidar_plane_m),
            frames=frames,
        )
        return out if out.suffix else out.with_suffix(".npz")

    def _spec_json(self) -> dict[str, Any]:
        return {
            "origin_m": list(self.spec.origin),
            "shape": list(self.spec.shape),
            "voxel_m": self.spec.voxel_m,
            "truncation_m": self.spec.truncation_m,
            "max_weight": self.spec.max_weight,
            "range_max_m": self.spec.range_max_m,
            "weight_ref_m": self.spec.weight_ref_m,
            "weight_cap": self.spec.weight_cap,
            "camera_band_m": list(self.spec.camera_band_m),
        }

    @classmethod
    def load(
        cls, path: str | Path, mount: PlanarMount | None = None, law: LidarLaw | None = None
    ) -> WorldMap:
        """A volume saved by :meth:`save`, grid and all — a known room is this and nothing
        else. Raises ``ValueError`` for a snapshot of another version."""
        data = np.load(Path(path))
        version = int(data["version"])
        if version != SNAPSHOT_VERSION:
            raise ValueError(f"world snapshot version {version}, not {SNAPSHOT_VERSION}")
        raw = json.loads(bytes(data["spec"].tobytes()).decode())
        spec = GridSpec(
            origin=(raw["origin_m"][0], raw["origin_m"][1], raw["origin_m"][2]),
            shape=(raw["shape"][0], raw["shape"][1], raw["shape"][2]),
            voxel_m=raw["voxel_m"],
            truncation_m=raw["truncation_m"],
            max_weight=raw["max_weight"],
            range_max_m=raw["range_max_m"],
            weight_ref_m=raw["weight_ref_m"],
            weight_cap=raw["weight_cap"],
            camera_band_m=(raw["camera_band_m"][0], raw["camera_band_m"][1]),
        )
        world = cls(spec, mount=mount, law=law)
        world.volume.sdf[:] = data["sdf"]
        world.volume.weight[:] = data["weight"]
        world.volume.rgb[:] = data["rgb"]
        world.volume.colour_weight[:] = data["colour_weight"]
        world.lidar_weight[:] = data["lidar_weight"]
        world.stamp = float(data["stamp"])
        world.lidar_plane_m = float(data["plane_m"])
        world.frames = [
            (float(row[0]), LIDAR if row[1] else CAMERA, row[2:].reshape(3, 4))
            for row in data["frames"]
        ]
        # The band the camera hands back is the plane's own layer, so it is rebuilt from the
        # saved plane — not from every row a leaning beam reached, which is what ``lidar_weight``
        # carries and would hand the lidar the camera's band across a restart.
        world._rows = world._plane_rows(world.lidar_plane_m) if world.lidar_weight.any() else None
        return world

    def seed_from_grid(
        self, values: Int8, resolution_m: float, origin: tuple[float, float], weight: float = 4.0
    ) -> int:
        """Write a saved 2D map (the trinary values of ``/map`` or of a pgm) into the lidar's
        layer as its starting state: an occupied cell becomes a surface, a free one open air,
        an unknown one stays unknown. This is all a "known room" ever was — after it the volume
        keeps growing from the sensors, and nothing in the stack can tell the two apart.

        Returns how many cells were seeded. Cells are matched by their centres, so a saved map
        of a different resolution or origin still lands where it belongs.

        The seeded rows become the lidar's layer (:meth:`_widen_rows`), so the camera hands them
        back from the very first frame: a seeded wall is the lidar's word until the lidar itself
        says otherwise. Without that claim the depth repainted the pgm's walls in the seconds
        before the first revolution arrived, and the slice a tracker would match on started out
        worse than the file it was seeded from.
        """
        rows = self._rows
        if rows is None:
            rows = self._layer_rows(RigidPose(np.eye(3), np.zeros(3)), self.mount)
            if rows is None:
                return 0
        self._widen_rows(*rows)
        s = self.spec
        nx, ny, _nz = s.shape
        cx = s.origin[0] + (np.arange(nx) + 0.5) * s.voxel_m
        cy = s.origin[1] + (np.arange(ny) + 0.5) * s.voxel_m
        col = np.floor((cx - origin[0]) / resolution_m).astype(int)
        row = np.floor((cy - origin[1]) / resolution_m).astype(int)
        height, width = values.shape
        ok_x, ok_y = (col >= 0) & (col < width), (row >= 0) & (row < height)
        ix = np.flatnonzero(ok_x)
        iy = np.flatnonzero(ok_y)
        if ix.size == 0 or iy.size == 0:
            return 0
        taken = values[np.ix_(row[iy], col[ix])]
        t = np.where(taken == OCCUPIED, 0.0, 1.0).astype(np.float32)
        known = taken != UNKNOWN
        gx, gy = np.meshgrid(ix, iy, indexing="ij")
        gx, gy = gx[known.T], gy[known.T]
        t = t.T[known.T]
        for iz in range(*rows):
            at = (gx, gy, np.full(gx.shape, iz))
            self.volume.sdf[at] = t
            self.volume.weight[at] = np.maximum(self.volume.weight[at], weight)
            self.lidar_weight[at] = np.maximum(self.lidar_weight[at], weight)
        return int(gx.size)


def trinary_from_log_odds(grid: OccupancyGrid) -> tuple[Int8, float, tuple[float, float]]:
    """A log-odds grid — a saved map read by :func:`pepin.mapping.grid_from_pgm`, or the
    tracker's own — as the (values, resolution, origin) :meth:`WorldMap.seed_from_grid` takes."""
    values = np.where(
        grid.log_odds > 0.0, OCCUPIED, np.where(grid.log_odds < 0.0, FREE, UNKNOWN)
    ).astype(np.int8)
    return values, grid.spec.resolution_m, (grid.spec.x_min_m, grid.spec.y_min_m)


def bearings_in_base(angles: Array, mount_yaw: float, mirrored: bool) -> Array:
    """Sensor-frame beam angles as robot-frame bearings, the conversion
    :func:`pepin.timeline.timed_scan_from_ros` makes on its points: mirrored for an upside-down
    sensor, then turned by the mount's yaw."""
    a = np.asarray(angles, dtype=float)
    return (mount_yaw - a) if mirrored else (mount_yaw + a)


@dataclass
class CorrectionFollower:
    """Which ``map -> odom`` the volume's content is painted under, and the move it owes the
    graph when that edge has moved on.

    Every frame in the volume was placed through the correction in force at its own stamp, so
    the volume as a whole stands in one of them — the one this remembers. A graph optimisation
    replaces the correction, and from that moment the painted room is stale by the difference:
    :meth:`pending` is that difference as a :class:`pepin.tsdf.PlanarShift` once it is worth the
    resample, and ``None`` while it is not. A refused difference is not forgotten — it is
    measured against the same anchor next time, so a run of corrections too small to move the
    volume one by one accumulates and moves it together.
    """

    painted_in: RigidPose | None = None
    applied: int = 0
    last: PlanarShift = field(default_factory=lambda: PlanarShift(0.0, 0.0, 0.0))

    def anchor(self, correction: RigidPose) -> None:
        """The correction the content is painted under from now on: the first one ever seen
        (an empty volume is born in it) and the one in force after every move."""
        self.painted_in = correction

    def pending(self, correction: RigidPose, min_m: float, min_deg: float) -> PlanarShift | None:
        """The move the volume owes the graph, or ``None`` when it stands close enough already
        (below ``min_m`` and ``min_deg``) or when nothing has anchored it yet."""
        if self.painted_in is None:
            return None
        shift = PlanarShift.between(self.painted_in, correction)
        if shift.translation_m < min_m and abs(shift.yaw_deg) < min_deg:
            return None
        return shift

    def moved(self, correction: RigidPose, shift: PlanarShift) -> None:
        """The volume has just been moved by ``shift``: it is painted under ``correction`` now."""
        self.anchor(correction)
        self.applied += 1
        self.last = shift


@dataclass
class SnapshotClock:
    """When the volume was last written to disk: the age a report line prints and the alarm a
    node saves on."""

    every_s: float
    last_s: float = field(default=0.0)

    def due(self, now: float) -> bool:
        """True when another ``every_s`` has passed (and the first time it is asked)."""
        return self.last_s <= 0.0 or now - self.last_s >= self.every_s

    def done(self, now: float) -> None:
        """A snapshot was just written."""
        self.last_s = now

    def age_s(self, now: float) -> float:
        """Seconds since the last snapshot; ``inf`` when there has been none."""
        return math.inf if self.last_s <= 0.0 else now - self.last_s
