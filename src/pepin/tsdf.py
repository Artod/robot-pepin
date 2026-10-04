"""A truncated signed distance field: the room as one surface, not as a pile of clouds.

Every depth frame is integrated into a voxel grid where each voxel keeps one number — how far
it is from the nearest surface, positive in front of it, negative behind, clipped at the
truncation — and a weight, the confidence gathered so far. A new observation moves the number
by a weighted average, so three frames of one wall taken a few degrees apart make one wall
(blurred by their disagreement), never three. Observations weigh by their distance: a wall
measured from 1 m outweighs the same wall measured from 4 m, so the model sharpens when the
cart comes close and does not blur back when it leaves. The surface is where the number
crosses zero, read out at sub-voxel positions between neighbouring voxels. A frame is placed
by TF at its own stamp, nothing else.

The model is a WINDOW, not a room. A volume painted in the odometry frame is local obstacle
memory: it follows the cart, so :meth:`Tsdf.recentre` slides the box by whole voxels
(:class:`WindowShift`) and drops what leaves it — a copy, not a resample: nothing moves in the
world, the box simply covers somewhere else. (The frame-to-model yaw search and the planar
resample that carried a map-frame model with the graph are on the tag alt/volume-map-2026-10-02.)
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from pepin.depth import NEAR_M, Intrinsics

Array = npt.NDArray[np.float64]
Float32 = npt.NDArray[np.float32]
Floats = npt.NDArray[np.floating[Any]]
Uint8 = npt.NDArray[np.uint8]


class RayClip(Protocol):
    """How far along each pixel's ray a frame may write: the optical depth past which the ray is
    inside something that is not the room (the cart's own body, :class:`pepin.body.RayDepth`)."""

    def at(self, v: npt.NDArray[np.intp], u: npt.NDArray[np.intp]) -> Float32:
        """The limit of the pixels at rows ``v``, columns ``u``; ``inf`` for no limit."""
        ...


@dataclass(frozen=True)
class GridSpec:
    """The voxel grid: where it starts in the map frame, how many voxels, and the fusion law."""

    origin: tuple[float, float, float]
    shape: tuple[int, int, int]
    voxel_m: float = 0.05
    truncation_m: float = 0.10  # two voxels: a surface is felt this far on either side
    max_weight: float = 60.0  # the model stays movable when the furniture moves
    range_max_m: float = 4.0  # farther depth is noise and geometry error, not information
    weight_ref_m: float = 2.0  # an observation from this distance weighs 1; (ref/d)^2 otherwise
    weight_cap: float = 4.0  # a very near observation weighs at most this
    # The height band the camera speaks for, the one /depth_scan marks in (``pepin.depth``):
    # what :meth:`pepin.worldmap.WorldMap.camera_band_slice` reads out of the volume.
    camera_band_m: tuple[float, float] = (0.15, 1.30)

    @classmethod
    def load(cls, path: str | Path) -> GridSpec:
        """The grid from ``config/fusion.json``."""
        with open(path) as f:
            data = json.load(f)
        extra: dict[str, Any] = {}
        if "camera_band_m" in data:  # absent in an older file: the default band stands
            lo, hi = data["camera_band_m"]
            extra["camera_band_m"] = (float(lo), float(hi))
        return cls(
            origin=tuple(data["origin_m"]),
            shape=tuple(data["shape"]),
            voxel_m=float(data["voxel_m"]),
            truncation_m=float(data["truncation_m"]),
            max_weight=float(data["max_weight"]),
            range_max_m=float(data["range_max_m"]),
            weight_ref_m=float(data["weight_ref_m"]),
            weight_cap=float(data["weight_cap"]),
            **extra,
        )

    def centred_on_start(self) -> GridSpec:
        """The same box with its x-y footprint centred on the map's ORIGIN: what a session whose
        map frame is born under the cart needs. Height is untouched.

        Only the origin, so it is right only where the cart is AT the origin. A cart that wakes up
        at (-9.4, +2.5) — a resumed room, or a map frame it did not create — is outside a box laid
        out like this (2026-09-18: a node kicked without its room parameter came up on
        (-7.0, -6.25) and could not have contained the cart at all). :meth:`centred_on` is the one
        to call when the cart's own pose is known.
        """
        return self.centred_on((0.0, 0.0))

    def centred_on(self, xy: tuple[float, float]) -> GridSpec:
        """The same box with its x-y footprint centred on ``xy`` — the cart's own pose when a
        volume is born, so the room grows outward from where the robot actually stands. Height is
        untouched, and the box is only moved, never resized."""
        nx, ny, _nz = self.shape
        return dataclasses.replace(
            self,
            origin=(xy[0] - nx * self.voxel_m / 2, xy[1] - ny * self.voxel_m / 2, self.origin[2]),
        )

    def aligned_to(self, origin_xy: tuple[float, float], resolution_m: float) -> GridSpec:
        """The same box moved by less than one voxel so that its x-y cell lattice is the saved
        map's: every voxel column of the volume is then a cell of that file, not a third of a
        cell away from one.

        A slice cut from a grid a third of a cell off the map it was seeded from carries every
        wall up to half a voxel aside, and a tracker matching on it answers there: replayed on
        the four tapes of 2026-09-13 the live pose sat a median 2.3-2.9 cm from where the very
        same map as a file put it, a bias and not noise (scratch/volume_vs_pgm.py,
        scratch/drive_bisect.py --map). Height is untouched, and the box only ever moves within
        one voxel, so what it covers is what it covered. Grids of another resolution are left
        alone: nothing can make their cells the same cells.
        """
        if abs(resolution_m - self.voxel_m) > 1e-9:
            return self
        x, y, z = self.origin
        snapped = tuple(
            o - (o - m) + round((o - m) / self.voxel_m) * self.voxel_m
            for o, m in ((x, origin_xy[0]), (y, origin_xy[1]))
        )
        return dataclasses.replace(self, origin=(snapped[0], snapped[1], z))

    def observation_weight(self, depth: Floats) -> Floats:
        """How much a measurement at ``depth`` metres counts: (ref / d)^2, capped."""
        w: Floats = np.minimum(self.weight_cap, (self.weight_ref_m / np.maximum(depth, 1e-3)) ** 2)
        return w

    @property
    def centre_xy(self) -> tuple[float, float]:
        """The middle of the box's x-y footprint, in the frame it is laid out in: what a rolling
        window keeps the cart near (:class:`WindowShift`)."""
        nx, ny, _nz = self.shape
        return (self.origin[0] + nx * self.voxel_m / 2, self.origin[1] + ny * self.voxel_m / 2)

    def off_centre_m(self, xy: tuple[float, float]) -> float:
        """How far ``xy`` stands from :attr:`centre_xy`, metres — the question a rolling window
        asks of the cart before every observation."""
        cx, cy = self.centre_xy
        return math.hypot(xy[0] - cx, xy[1] - cy)

    def voxels_to_centre(self, xy: tuple[float, float]) -> tuple[int, int]:
        """How many WHOLE voxels along x and y the box must slide for its footprint's centre to
        land on ``xy``, as near as the lattice allows.

        Whole voxels, so the lattice never moves: an integer slide is a copy of the overlap and
        nothing else — no resampling, no quantisation, and every surviving voxel keeps the metres
        it was painted at. A fractional slide would need a resample, which a window that only
        changes what it covers has no reason to pay.
        """
        cx, cy = self.centre_xy
        return (round((xy[0] - cx) / self.voxel_m), round((xy[1] - cy) / self.voxel_m))

    def moved_by_voxels(self, di: int, dj: int) -> GridSpec:
        """The same box slid by whole voxels along x and y: the origin moves, the lattice, the
        shape and the height do not."""
        x, y, z = self.origin
        return dataclasses.replace(self, origin=(x + di * self.voxel_m, y + dj * self.voxel_m, z))


BAND_HALF_Z_M = 0.125  # the fallback when config/fusion.json is unreadable or silent
# How far the cart may stand from a rolling window's centre before the window is slid onto it —
# the fallback when config/fusion.json is unreadable or silent (see :func:`window_recentre_m`).
WINDOW_RECENTRE_M = 2.0


def band_half_z_m(path: str | Path | None = None) -> float:
    """Half the height band around the lidar's plane whose points are exact by construction,
    metres, from ``band_half_z_m`` in ``config/fusion.json``.

    The band is the layer a frame may be turned by: the depth image is anchored on the beams,
    so only their own height is trustworthy enough to seat a frame on the model. The plane
    itself is never written here — it is the lidar's mount, and the node reads it from the
    published ``base_link -> laser`` edge. :data:`BAND_HALF_Z_M` when the file is missing or
    does not name it (the value the band has always had, 0.10-0.35 around an assumed 0.20 m).
    """
    if path is None:
        from pepin.deployment import config_file

        try:
            path = config_file("fusion.json")
        except FileNotFoundError:
            return BAND_HALF_Z_M
    try:
        with open(path) as f:
            return float(json.load(f)["band_half_z_m"])
    except (OSError, ValueError, KeyError, TypeError):
        return BAND_HALF_Z_M


def window_recentre_m(path: str | Path | None = None) -> float:
    """How far the cart may travel from a rolling window's centre before the window is slid onto
    it, metres, from ``window_recentre_m`` in ``config/fusion.json``.

    It is a distance and not a fraction because what it must cover is a distance: the marks' fan
    reaches 3.0 m (:data:`pepin.volume_scan.MARKS_RANGE_M`) and the camera integrates to
    ``range_max_m``, so the window must still hold that much room around the cart at the moment
    it is furthest off centre. On the shipped grid (250 voxels of 5 cm on the shorter side, a
    half-extent of 6.25 m) the 2.0 m of config/fusion.json leaves at least 4.25 m of painted
    memory in every direction, and at a drive's 0.2 m/s one slide is paid about every ten
    seconds. :data:`WINDOW_RECENTRE_M` when the file is missing or does not name it.
    """
    if path is None:
        from pepin.deployment import config_file

        try:
            path = config_file("fusion.json")
        except FileNotFoundError:
            return WINDOW_RECENTRE_M
    try:
        with open(path) as f:
            return float(json.load(f)["window_recentre_m"])
    except (OSError, ValueError, KeyError, TypeError):
        return WINDOW_RECENTRE_M


@dataclass(frozen=True)
class RigidPose:
    """A frame's placement in the map: 3x3 rotation and translation (map <- frame)."""

    rotation: Array
    translation: Array

    def inverse(self) -> RigidPose:
        """The same transform the other way (frame <- map)."""
        r = self.rotation.T
        return RigidPose(r, -(r @ self.translation))


@dataclass(frozen=True)
class WindowShift:
    """A rolling window's move: the whole voxels the box slides along x and y, and the copy that
    carries it out.

    A volume painted in ``odom`` is LOCAL OBSTACLE MEMORY, not a room, so its box follows the
    cart and whatever leaves it is forgotten (the nvblox local mapper's arrangement; the room's
    own geometry is the pose graph's). Painted in ``map`` and kept, the same volume accumulated
    the walls of some twenty re-seatings of the tracker and put 300-650 lethal cells around the
    cart that the lidar never saw (2026-09-21, scratch/nav2_hang/layer_blame.py).

    NOTHING MOVES IN THE WORLD HERE. The box slides; every voxel that survives describes exactly
    the metres it was painted at, and the slide costs one copy per channel instead of the 9-108 ms
    :meth:`Tsdf.shift` pays to resample content that really has to move.
    """

    di: int
    dj: int

    @property
    def nothing(self) -> bool:
        """True when the box does not slide at all: the cart is still in the middle of it."""
        return self.di == 0 and self.dj == 0

    def text(self) -> str:
        """The slide for a report line: voxels along x and y."""
        return f"{self.di:+d}, {self.dj:+d} voxels"

    def boxes(self, nx: int, ny: int) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
        """The overlap of a grid ``nx`` by ``ny`` columns with itself after the slide: which
        columns survive and where they land; ``None`` when the box has jumped clear of itself
        and nothing at all survives (a carried cart, a pose that teleported)."""
        if abs(self.di) >= nx or abs(self.dj) >= ny:
            return None
        src = (
            slice(max(self.di, 0), nx + min(self.di, 0)),
            slice(max(self.dj, 0), ny + min(self.dj, 0)),
        )
        dst = (
            slice(max(-self.di, 0), nx - max(self.di, 0)),
            slice(max(-self.dj, 0), ny - max(self.dj, 0)),
        )
        return src, dst

    def rolled[T: np.generic](self, field: npt.NDArray[T], empty: float = 0.0) -> npt.NDArray[T]:
        """One channel of the grid after the slide: a fresh array of ``empty`` with the overlap
        copied into it, so what left the window is gone and what stayed is untouched. Works on a
        colour channel too — only the first two axes are indexed."""
        out: npt.NDArray[T] = np.full_like(field, empty)
        boxes = self.boxes(int(field.shape[0]), int(field.shape[1]))
        if boxes is not None:
            src, dst = boxes
            out[dst] = field[src]
        return out


@dataclass(frozen=True)
class DepthLaw:
    """How a depth frame writes into the volume — and, since 2026-09-22, what a pixel with NO
    depth is allowed to say (``no_depth_free``).

    Until then a NaN pixel touched nothing at all: :meth:`Tsdf.integrate` only moved the voxels
    whose pixel carried a finite depth, so a voxel could only ever be carved by a ray that
    MEASURED something behind it. On a stereo rig cut at its own reach that leaves a hole in the
    law. The published depth is NaN past the rig's reach (2.46 m on the MMlove rig,
    ``pepin.stereo_depth``), so a person standing at 0.4 m with an open corridor behind him has
    NaN at every one of his pixels the moment he leaves — nothing carves him, and he stays for
    ever. Measured on the live volume of 2026-09-22 (scratch/one_localiser/black_voxels.py): an
    airborne cluster of 133 voxels, 94 % of its pixels NaN, 0 % ever seen free, not one voxel
    rewritten in 60 s; the operator's face 175 voxels, all 175 at the same millimetre a minute
    later. The same person at 2 m, with a wall at 2.2 m behind him, cleared in about 3 s.

    So ``no_depth_free`` lets a NaN pixel carve free space along its own ray, from
    :data:`pepin.depth.NEAR_M` out to ``reach_m - truncation``: as far as the rig answers for,
    stopping a truncation short so a surface standing AT the reach is not rubbed out by the very
    rays that could not measure it. ``reach_m`` is the source's own reach and nothing else — the
    node measures it from the frames (the largest finite depth they carry), because
    ``depth_reach_m`` on the publisher is a looser gate (4.0 m) than the rig itself (~3.9 m by
    its error model since 2026-09-30, 2.46 m before at a 10 cm budget) and carving to the gate
    would carve through metres the camera never looked at.

    ``no_depth_weight`` is why this is not as loud as a measurement. A NaN is not evidence of
    emptiness: the matcher also refuses a near textureless wall, a rectification margin and an
    over-exposed window, and carving those at full weight would eat a real surface. A carving
    ray therefore weighs ``no_depth_weight`` of what a measurement AT THE REACH weighs
    (``GridSpec.observation_weight(reach)``: 0.67 at the stereo rig's 2.44 m, so 0.34 at the
    default 0.5) — the weakest honest reading of the ray, not the (ref/z)^2 a near voxel would
    claim. At ``max_weight`` 20 that clears a saturated phantom in 17 frames by the integration
    law and 18 on the taped frames of 2026-09-22 (2.8 s at 6.4 fps,
    scratch/one_localiser/volume_ab.py, against NEVER at the old law). It writes 37628 voxels a
    frame instead of 8705 and costs 10.1 -> 12.7 ms on the live grid
    (scratch/one_localiser/carve_cost.py), and it leaves a wall the lidar also holds untouched
    (that layer is protected in :meth:`pepin.worldmap.WorldMap.integrate_depth`).
    """

    no_depth_free: bool = False  # a pixel with no depth carves its ray, or touches nothing
    no_depth_weight: float = 0.5  # of what a measurement at the reach weighs
    reach_m: float = 0.0  # the source's own reach, metres; 0: unknown, so nothing is carved
    # A fixed weight for every measured pixel instead of the grid's (ref / d)^2: what a ranger
    # gets whose error does not shrink as the surface comes nearer (a ToF fan,
    # :mod:`pepin.tof_rays`). A depthless pixel then weighs ``no_depth_weight`` of THIS, since
    # this is what a measurement at the reach weighs. None: the grid's own law, as for a camera.
    hit_weight: float | None = None
    # A depthless pixel carves to the reach ITSELF, and a measured pixel writes nothing beyond
    # it: for a ranger whose "nothing" is certain out to its reach (a ToF fan's +inf means no
    # return within its trusted range, the sensor itself seeing much farther), so that no halo
    # a hit leaves behind the surface can outlive the sensor's own power to carve it. Off, the
    # camera's law: carve one truncation short of the reach, write a hit's whole halo.
    carve_to_reach: bool = False
    near_m: float = NEAR_M  # nearer, a pixel writes nothing: the camera 0.20, a ToF 0.08

    def carve_to_m(self, truncation_m: float) -> float:
        """How far down a depthless ray free space may be written, metres: the source's reach
        less one truncation (a surface standing at the reach keeps its halo) — the reach itself
        under ``carve_to_reach`` — and 0 — carve nothing — while no reach is known, the flag is
        off, or the weight is 0 (a ray that weighs nothing says nothing, and it must not reach
        the average as a zero divisor)."""
        if not self.no_depth_free or self.reach_m <= 0.0 or self.no_depth_weight <= 0.0:
            return 0.0
        if self.carve_to_reach:
            return self.reach_m
        return max(0.0, self.reach_m - truncation_m)

    @property
    def write_limit_m(self) -> float:
        """How far along its ray a MEASURED pixel may write at all: unbounded, or the reach
        under ``carve_to_reach`` (a hit near the reach then marks without its back halo)."""
        if self.carve_to_reach and self.reach_m > 0.0:
            return self.reach_m
        return math.inf

    def measurement_weight(self, spec: GridSpec, depth: Floats) -> Floats:
        """What a measured pixel at ``depth`` weighs under this law: :attr:`hit_weight` when
        the source has a fixed one, the grid's (ref / d)^2 otherwise."""
        if self.hit_weight is not None:
            return np.full(np.shape(depth), self.hit_weight, dtype=float)
        return spec.observation_weight(depth)


class ObservedReach:
    """How far the depth source actually answers, metres, measured from the frames themselves.

    Nothing on the wire carries it. The publisher's own gate is looser than the rig (3.0 m
    against the stereo rig's 2.46 m, ``pepin_bringup.depth_stream``'s ``depth_reach_m`` against
    :meth:`pepin.stereo_depth.StereoDepth.reach`), and a law that carved to the gate would carve
    through half a metre nobody measured. But the depth is NaN above the reach BY CONSTRUCTION,
    so the largest finite metre in a frame can never exceed the reach and equals it whenever
    anything far is in view: the running maximum over the last ``frames`` frames is the reach,
    and it can only ever under-state it, which carves less rather than more.

    One ``nanmax`` per frame, 0.3 ms on the live 800x600 image.
    """

    def __init__(self, frames: int = 60) -> None:
        self._seen: list[float] = []
        self._keep = frames

    def saw(self, depth: Array | Float32) -> float:
        """Take one frame's largest finite depth into the window; returns the reach so far."""
        d = np.asarray(depth)
        finite = np.isfinite(d)
        self._seen.append(float(d[finite].max()) if bool(finite.any()) else 0.0)
        if len(self._seen) > self._keep:
            del self._seen[: len(self._seen) - self._keep]
        return self.m

    @property
    def m(self) -> float:
        """The reach the window has seen, metres; 0 while no frame has carried a depth."""
        return max(self._seen) if self._seen else 0.0

    @property
    def frames(self) -> int:
        """How many frames the answer rests on."""
        return len(self._seen)


class Tsdf:
    """The fused model: signed distances, weights and colours on a grid, and the operations
    on it — integrate a frame, slide the window, read the surface."""

    def __init__(self, spec: GridSpec) -> None:
        self.spec = spec
        nx, ny, nz = spec.shape
        self.sdf: Float32 = np.ones((nx, ny, nz), dtype=np.float32)  # in truncation units
        self.weight: Float32 = np.zeros((nx, ny, nz), dtype=np.float32)
        self.rgb: Uint8 = np.zeros((nx, ny, nz, 3), dtype=np.uint8)
        # colour has its own weight: a voxel seen as free space for a while and then as a wall
        # would otherwise start its colour from black
        self.colour_weight: Float32 = np.zeros((nx, ny, nz), dtype=np.float32)

    def snapshot(self) -> Tsdf:
        """A copy of what ``surface`` reads (field, weight, colour): read the surface from it
        while the original keeps integrating."""
        twin = Tsdf.__new__(Tsdf)
        twin.spec = self.spec
        twin.sdf, twin.weight, twin.rgb = self.sdf.copy(), self.weight.copy(), self.rgb.copy()
        # the colour's own weight travels with the colour: it is what says a voxel was never
        # painted by a camera at all (:meth:`surface`, ``colour_fallback``)
        twin.colour_weight = self.colour_weight.copy()
        return twin

    def window(self, box: tuple[slice, slice, slice]) -> Tsdf:
        """One box of the grid as a volume of its own — the same voxels, on a :class:`GridSpec`
        whose origin is the box's own corner, so everything read out of it lands in the same map
        metres it did here.

        What :meth:`snapshot` is for the whole model, this is for a neighbourhood: a copy of a
        few megabytes instead of the grid, taken in well under a millisecond, so a caller that
        needs the surface around the cart many times a second (``/depth_marks``,
        :mod:`pepin.volume_scan`) holds the model's lock for the copy and does the reading
        outside it. A box is only voxels, and a planar move never touches the lattice, so the
        twin stays true until the very voxels it copied change.
        """
        corner = [sl.indices(n)[0] for sl, n in zip(box, self.sdf.shape, strict=True)]
        ox, oy, oz = (
            o + i * self.spec.voxel_m for o, i in zip(self.spec.origin, corner, strict=True)
        )
        twin = Tsdf.__new__(Tsdf)
        sdf = self.sdf[box].copy()
        nx, ny, nz = sdf.shape
        twin.spec = dataclasses.replace(self.spec, origin=(ox, oy, oz), shape=(nx, ny, nz))
        twin.sdf, twin.weight, twin.rgb = sdf, self.weight[box].copy(), self.rgb[box].copy()
        twin.colour_weight = self.colour_weight[box].copy()
        return twin

    def recentre(self, xy: tuple[float, float]) -> WindowShift:
        """Slide the box by whole voxels so its footprint is centred on ``xy`` — the cart —
        keeping every voxel that stays inside and dropping the ones that leave; returns the move
        (:attr:`WindowShift.nothing` when the cart is already in the middle of a voxel of it).

        THE ROLLING WINDOW of a volume that is local obstacle memory (``volume_frame`` odom on
        pepin_bringup.depth_fusion). Nothing is resampled and nothing is carried: the grid's
        lattice is untouched, a surviving voxel keeps the metres it was painted at, and the ones
        that fall off the trailing edge are forgotten because a local map is not a room. A voxel
        the slide has brought in from outside starts as unobserved — weight zero, a whole
        truncation from any surface — exactly as a newborn volume's does.

        The cost is one copy per channel and no arithmetic: measured 0.3-0.9 ms on the fusion
        node's 120x120x34 test grid and 4.7 ms on the live 280x250x34 one
        (tests/unit/test_tsdf.py), against the 9-108 ms :meth:`shift` pays to resample.
        """
        move = WindowShift(*self.spec.voxels_to_centre(xy))
        if move.nothing:
            return move
        weight = move.rolled(self.weight)
        sdf = move.rolled(self.sdf)
        sdf[weight == 0.0] = 1.0  # unobserved: a whole truncation from any surface
        self.sdf, self.weight = sdf, weight
        self.rgb = move.rolled(self.rgb)
        self.colour_weight = move.rolled(self.colour_weight)
        self.spec = self.spec.moved_by_voxels(move.di, move.dj)
        return move

    # ---- geometry helpers ----------------------------------------------------------------
    def index_box(self, lo_m: Array, hi_m: Array) -> tuple[slice, slice, slice] | None:
        """Voxel index ranges of the box ``lo_m``..``hi_m`` (map metres), clipped to the grid;
        ``None`` when the box misses the grid."""
        s = self.spec
        origin = np.array(s.origin)
        lo = np.floor((lo_m - origin) / s.voxel_m).astype(int) - 1
        hi = np.ceil((hi_m - origin) / s.voxel_m).astype(int) + 1
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.array(s.shape))
        if np.any(hi <= lo):
            return None
        return slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2])

    def _frustum_box(
        self, intr: Intrinsics, pose: RigidPose, far_m: float | None = None
    ) -> tuple[slice, slice, slice] | None:
        """The voxels a frame can touch: the bounding box of the camera and its four corner
        rays at ``far_m`` (``range_max`` by default) plus the truncation (a surface at the far
        limit is felt that far behind it), so a frame integrates its own view, not a 8 m cube
        around it. :meth:`integrate` passes the frame's own far limit — the deepest pixel it
        carries, or how far its depthless pixels carve — so a short-ranged source (a ToF fan
        reaching 1 m) does not pay for four metres of voxels its rays cannot reach."""
        s = self.spec
        far = (s.range_max_m if far_m is None else min(s.range_max_m, far_m)) + s.truncation_m
        us, vs = (-0.5, intr.width - 0.5), (-0.5, intr.height - 0.5)  # the pixels' outer edges
        corners = np.array(
            [
                [(u - intr.cx) / intr.fx * far, (v - intr.cy) / intr.fy * far, far]
                for u in us
                for v in vs
            ]
        )
        pts = np.vstack([pose.translation, corners @ pose.rotation.T + pose.translation])
        return self.index_box(pts.min(axis=0), pts.max(axis=0))

    def centres(self, box: tuple[slice, slice, slice]) -> Float32:
        """The map-frame centres (n, 3) of the voxels of ``box``, in its C order."""
        return self._centres(box)

    def _centres(self, box: tuple[slice, slice, slice]) -> Float32:
        s = self.spec
        ix, iy, iz = np.mgrid[box[0], box[1], box[2]]
        centres: Float32 = (
            (np.stack([ix, iy, iz], axis=-1).reshape(-1, 3) + 0.5) * s.voxel_m + np.array(s.origin)
        ).astype(np.float32)
        return centres

    def voxel_of(self, points_map: Array) -> tuple[Array, Array]:
        """Integer voxel indices of map points and a mask of the ones inside the grid."""
        s = self.spec
        idx = np.floor((np.asarray(points_map) - np.array(s.origin)) / s.voxel_m).astype(int)
        inside = np.all((idx >= 0) & (idx < np.array(s.shape)), axis=1)
        return idx, inside

    # ---- integration ---------------------------------------------------------------------
    def integrate(
        self,
        depth: Array | Float32,
        rgb: Uint8 | None,
        intr: Intrinsics,
        pose: RigidPose,
        law: DepthLaw | None = None,
        clip: RayClip | None = None,
    ) -> int:
        """Fuse one depth frame (metres, optical frame) taken from ``pose`` (map <- optical);
        returns how many voxels were updated. Colour goes only into voxels within the
        truncation of the surface, never into the free space a ray crosses on its way.

        ``law`` is what a pixel with NO depth may say (:class:`DepthLaw`). Off — the default,
        and everything this method did until 2026-09-22 — a NaN pixel touches nothing, and only
        a ray that measured a surface moves any voxel. On, a NaN pixel carves free space along
        its own ray out to ``law.carve_to_m``, at ``law.no_depth_weight`` of what a measurement
        at that range weighs: the fix for a phantom standing in front of something the rig
        cannot reach, which no ray could ever carve.

        ``clip`` is the self-filter (:mod:`pepin.body`): per pixel, the depth at which its ray
        enters the cart's own body. A pixel whose depth lies there or beyond measured the body
        (or something behind it) and measures no room — it is read as a pixel with no depth —
        and no voxel on or past that depth along the ray is written, measured or carved. ``None``
        writes every ray whole, as before.
        """
        s = self.spec
        depth_law = law if law is not None else DepthLaw()
        carve_to = depth_law.carve_to_m(s.truncation_m)
        # the frame's own far limit: nothing beyond its deepest pixel or its carve can be written
        d_all = np.asarray(depth, dtype=np.float32)
        d_finite = d_all[np.isfinite(d_all)]
        far_m = max(float(d_finite.max()) if d_finite.size else 0.0, carve_to)
        box = self._frustum_box(intr, pose, far_m)
        if box is None:
            return 0
        centres = self._centres(box)
        inv = pose.inverse()
        # voxel centres in the optical frame, float32 throughout: a centimetre is 1e-2 of a
        # metre, far above float32's 1e-7, and the frame is millions of voxels
        cam = centres @ inv.rotation.T.astype(np.float32) + inv.translation.astype(np.float32)
        z = cam[:, 2]
        front = z > depth_law.near_m
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(front, intr.fx * cam[:, 0] / z + intr.cx, -1.0)
            v = np.where(front, intr.fy * cam[:, 1] / z + intr.cy, -1.0)
        # pixel i's ray passes through coordinate i (``backproject``): the nearest pixel, not the
        # one to the left, or every voxel reads the depth half a pixel aside
        ui, vi = np.rint(u).astype(int), np.rint(v).astype(int)
        seen = front & (ui >= 0) & (ui < intr.width) & (vi >= 0) & (vi < intr.height)
        if not np.any(seen):
            return 0
        d = np.full(centres.shape[0], np.nan, dtype=np.float32)
        d[seen] = d_all[vi[seen], ui[seen]]
        inside: npt.NDArray[np.bool_] | None = None  # voxels on or past their ray's body entry
        if clip is not None:
            limit = np.full(centres.shape[0], np.inf, dtype=np.float32)
            limit[seen] = clip.at(vi[seen], ui[seen])
            d[d >= limit] = np.nan  # the body's own pixel, or one behind it: no room measured
            inside = z >= limit
        finite = np.isfinite(d)
        measured = finite & (d > depth_law.near_m) & (d <= s.range_max_m)
        sdf = d - z  # positive: the voxel is between the camera and the surface
        touch = measured & (sdf > -s.truncation_m) & (z <= depth_law.write_limit_m)
        # A pixel with NO depth: its ray is free space as far as the source answers for it, at a
        # weight that says so (:class:`DepthLaw`). The two masks are disjoint by construction —
        # a voxel reads one pixel, and that pixel either measured something or did not.
        carving = carve_to > depth_law.near_m
        carve = (
            seen & ~finite & (z > depth_law.near_m) & (z <= carve_to)
            if carving
            else np.zeros(centres.shape[0], dtype=bool)
        )
        if inside is not None:  # a ray that enters the body writes nothing from there on
            touch &= ~inside
            carve &= ~inside
        written = touch | carve
        if not np.any(written):
            return 0
        flat = np.flatnonzero(written)
        told = touch[flat]  # this voxel's pixel measured a surface, rather than saying nothing
        t = np.ones(flat.size, dtype=np.float32)  # a carved voxel: a whole truncation from any
        t[told] = np.minimum(1.0, sdf[flat][told] / s.truncation_m)  # surface, i.e. free space
        # the weakest honest reading of a depthless ray: what a measurement AT the reach weighs,
        # reduced (:class:`DepthLaw`), because a NaN is also what a textureless wall looks like
        w_free = np.float32(
            depth_law.no_depth_weight * float(depth_law.measurement_weight(s, np.array(carve_to)))
            if carving
            else 0.0
        )
        w_obs = np.full(flat.size, w_free, dtype=np.float32)
        w_obs[told] = depth_law.measurement_weight(s, d[flat][told]).astype(np.float32)
        shape = (box[0].stop - box[0].start, box[1].stop - box[1].start, box[2].stop - box[2].start)
        ix, iy, iz = np.unravel_index(flat, shape)
        ix, iy, iz = ix + box[0].start, iy + box[1].start, iz + box[2].start
        w_old = self.weight[ix, iy, iz]
        w_new = w_old + w_obs
        self.sdf[ix, iy, iz] = (self.sdf[ix, iy, iz] * w_old + t * w_obs) / w_new
        self.weight[ix, iy, iz] = np.minimum(s.max_weight, w_new)
        if rgb is not None:
            near = t < 1.0  # within the truncation: the surface itself, not the ray's free run
            if np.any(near):
                self._blend_colour(
                    (ix[near], iy[near], iz[near]),
                    np.asarray(rgb)[vi[flat][near], ui[flat][near]],
                    w_obs[near],
                )
        return int(flat.size)

    def _blend_colour(self, at: tuple[Array, Array, Array], colour: Uint8, w_obs: Float32) -> None:
        """The colour of the voxels ``at`` moved toward ``colour`` by the same weighted average
        as the field, on the colour's own weight; blended in float, stored as bytes."""
        cw_old = self.colour_weight[at]
        cw_new = cw_old + w_obs
        old = self.rgb[at].astype(np.float32)
        mixed = (old * cw_old[:, None] + colour.astype(np.float32) * w_obs[:, None]) / cw_new[
            :, None
        ]
        self.rgb[at] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
        self.colour_weight[at] = np.minimum(self.spec.max_weight, cw_new)

    # ---- readout -------------------------------------------------------------------------
    def surface(
        self,
        min_weight: float = 2.0,
        box: tuple[slice, slice, slice] | None = None,
        colour_fallback: bool = False,
    ) -> tuple[Array, Uint8]:
        """Points where the field crosses zero between two weighted neighbours, interpolated
        to the crossing along each axis: the model's surface as (n, 3) map points and colours
        (each point's colour from the neighbour nearer the surface, the smaller |sdf|).

        ``colour_fallback`` is the LIDAR'S OWN LAYER, drawn in the camera's colours rather than
        in black. The lidar writes the field and the weight of the voxels its beams cross
        (:meth:`pepin.worldmap.WorldMap.integrate_scan`) and no colour at all — colour is the
        camera's word — so a wall the beams put in the volume is a crossing between two voxels
        of which the nearer one may never have been painted, and the rule above then reads its
        black. On the live volume of 2026-09-22 that was 1532 pure-black points, every one of
        them a real wall in the lidar's own height band (scratch/one_localiser/depth_nan_why.py).
        With the fallback on, a point whose nearer neighbour carries no colour weight takes the
        OTHER neighbour's colour, and only if that one has none either does it stay black. The
        alternative — writing a neutral grey on the scan path — was refused: it would put a
        colour in the volume that no camera ever saw, and a voxel the camera has genuinely never
        looked at must read as exactly that. Off is the old behaviour, black and all.

        IT IS A SMALL FIX, and the measurement says why: on the taped minute of 2026-09-22
        (scratch/one_localiser/volume_ab.py) it recovers 16 of 987 black points. The other 971
        are crossings NEITHER of whose voxels a camera has painted — the camera's own surface
        lands in different voxels from the beams' — so nothing but an invented colour could
        light them up.

        ``box`` is a sub-box of the grid (voxel index slices) to search instead of the whole of
        it — the same test on fewer voxels, for a caller that needs the surface AROUND THE CART
        many times a second rather than the room once (:mod:`pepin.volume_scan`). A crossing is
        still a crossing between two neighbours, so a surface on the box's outer face, whose
        other neighbour lies outside, is not reported: the box is widened by the caller, not
        here. Nothing else changes — ``None`` searches everything, as it always did.
        """
        s = self.spec
        pts: list[Array] = []
        cols: list[Uint8] = []
        sel = box if box is not None else (slice(None), slice(None), slice(None))
        corner = np.array([sl.indices(n)[0] for sl, n in zip(sel, self.sdf.shape, strict=True)])
        field, rgb = self.sdf[sel], self.rgb[sel]
        known = self.weight[sel] >= min_weight
        # The crossing test without np.sign, which cost two float passes per axis: "the signs
        # differ and the first is not zero" is "the first is positive and the second is not, or
        # the first is negative and the second is not" — the same mask (verified cell for cell
        # on the live volume, scratch/marks_slice_cost.py) at a third of the time, which is what
        # lets /depth_marks take this readout twenty times a second.
        positive, negative = field > 0.0, field < 0.0
        painted = self.colour_weight[sel] > 0.0 if colour_fallback else None
        for axis in range(3):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[axis] = slice(0, -1)
            b[axis] = slice(1, None)
            sa, sb = field[tuple(a)], field[tuple(b)]
            ka, kb = known[tuple(a)], known[tuple(b)]
            pa, pb = positive[tuple(a)], positive[tuple(b)]
            na, nb = negative[tuple(a)], negative[tuple(b)]
            cross = ka & kb & ((pa & ~pb) | (na & ~nb))
            if not np.any(cross):
                continue
            ia = np.argwhere(cross)
            va, vb = sa[cross], sb[cross]
            frac = va / (va - vb)
            idx = ia.astype(float)
            idx[:, axis] += frac
            pts.append((idx + corner + 0.5) * s.voxel_m + np.array(s.origin))
            ib = ia.copy()
            ib[:, axis] += 1
            takes_b = np.abs(vb) < np.abs(va)
            if painted is not None:
                # the nearer voxel was never painted by a camera (colour weight 0) but the other
                # one was: take the colour that exists rather than the black that means nothing
                ca, cb = painted[tuple(a)][cross], painted[tuple(b)][cross]
                takes_b = np.where(ca == cb, takes_b, cb)
            nearer = np.where(takes_b[:, None], ib, ia)
            cols.append(rgb[nearer[:, 0], nearer[:, 1], nearer[:, 2]])
        if not pts:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
        return np.concatenate(pts), np.concatenate(cols)
