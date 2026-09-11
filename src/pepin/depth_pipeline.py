"""The depth correction as a pipeline: anchors say what is true, laws turn depth into metres.

:mod:`pepin.depth` holds the pieces — the lidar's beams projected into the image, the affine law
fitted on their (network, true) pairs, the edge filter, the floor snapped to its plane — and the
node ran them in one fixed order. Here the same pieces run as an ordered list of stages, each
switchable by name, so a node's flags toggle stages instead of branches, every stage's cost and
effect is counted per frame, and a new source of truth is one more stage in the list.

Two roles. An :class:`Anchor` is an external truth about some pixels: it contributes
:class:`Pairs` — (network depth, true depth, weight) — to the pool the laws fit on, and / or
corrects the pixels it knows directly. A :class:`Law` is the map from the network's depth to
metres, fitted on the pooled pairs and applied to the whole image. Today's chain is
:class:`EdgeFilter` -> :class:`LidarAnchor` -> :class:`AffineLaw` -> :class:`FloorAnchor`, bit
for bit what ``pepin.depth`` computed for the node (the tests hold the two against each other).
Two new anchors stand beside the lidar: :class:`FloorPairs` — the floor's pixels pair the
network's depth with the plane's geometric depth, a second hoop the law can be fitted through
with no lidar at all — and :class:`WallAnchor` — the lidar's returns extruded up the image
columns while the network's depth stays continuous (a wall goes on, a chair back ends at its
top), a third hoop above the lidar's row. Two more laws let the data say whether the error above
the lidar's row depends on the elevation: :class:`ElevationLaw` adds a term in the ray's lift,
:class:`RowLaw` fits a law per band of rows.

Measured on run 0171 (29 frames against the run's lidar cloud and its COLMAP reference,
scratch/pipeline_vs_truth.py, 2026-09-11), which set the defaults of :func:`standard_pipeline`:
the network's error is not one law — 1.1x too far on the floor, 1.6x at the lidar's row, 2.0x
from 0.3 m up — so every hoop but the lidar's pulls the law off the row the costmap lives on.
Floor pairs alone put the lidar's row 2.0x too far (a camera without lidar sees walls where the
raw network does); wall pairs at 0.2 of a beam put it 10 % too near while fixing the 0.5-0.8 m
slice (0.86 -> 1.00 of the truth); the elevation term is real (c +0.1) but linear in lift is the
wrong shape (the error steps within 60 rows of the lidar's row and is flat above), and the row
law overfits. The lidar-only affine law stays the default; the new anchors ship switched off.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from itertools import pairwise
from typing import Protocol

import numpy as np
import numpy.typing as npt

from pepin.contact import DEPTH_NOISE, DepthNoise
from pepin.depth import (
    A_BOUNDS,
    B_BOUNDS,
    EDGE_REL_STEP,
    FLOOR_HEIGHT_TOLERANCE,
    MIN_DEPTH_SPREAD,
    MIN_SAMPLES,
    NEAR_M,
    POOL_FRAMES,
    POOL_MIN_SAMPLES,
    UP_LEVEL,
    Array,
    CameraPose,
    Intrinsics,
    Mask,
    apply_affine,
    beam_hits,
    drop_edges,
    edge_mask,
    fit_affine,
    floor_anchor,
    floor_depth,
    in_image,
    project,
    project_all,
)

LEAN_STEP = 0.003  # the floor's expected depth is recomputed when the up vector moves this much
FLOOR_PAIR_STRIDE = 8  # every 8th row and column of the floor: 3600 candidates of a 640x360 frame
FLOOR_PAIR_WEIGHT = 0.1  # a floor pixel's share against a lidar beam's 1: the lidar keeps its row
WALL_ROW_STRIDE = 4  # rows between two wall pairs of one column
WALL_PAIR_WEIGHT = 0.2
WALL_MAX_HEIGHT = 2.0  # metres above the floor a wall point may stand: higher is a ceiling
WALL_NEIGHBOUR_GAP = 0.30  # metres between a return and its scan neighbours for a wall direction
WALL_SLOPE_TOL = 0.004  # per row: how much faster than the plane the network's depth may climb
WALL_SLOPE_WINDOW = 6  # rows either side over which that climb is measured (the noise averaged)
MIN_LIFT_SPREAD = 0.15  # the pool's elevation span (5th-95th of lift) before an elevation term
ROW_BANDS = 6  # bands of elevation of the row law


# ---- what flows through the pipeline ----------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Pairs:
    """What an anchor knows about some pixels: the network's depth ``d`` there, the true depth
    ``z`` (metres along the optical axis), each pair's ``weight`` in a fit (a lidar beam is 1),
    and the ray's ``lift`` — its elevation above the optical axis per unit depth,
    ``-(row - cy) / fy`` — which the elevation-aware laws read."""

    d: Array
    z: Array
    weight: Array
    lift: Array

    @classmethod
    def of(cls, d: Array, z: Array, lift: Array, weight: float | Array = 1.0) -> Pairs:
        """Pairs from arrays, ``weight`` one number for all or one per pair."""
        dd = np.asarray(d, dtype=float)
        w = (
            np.full(dd.shape, float(weight))
            if np.ndim(weight) == 0
            else np.asarray(weight, dtype=float)
        )
        return cls(dd, np.asarray(z, dtype=float), w, np.asarray(lift, dtype=float))

    @property
    def size(self) -> int:
        """How many pairs."""
        return int(self.d.size)

    @staticmethod
    def join(parts: Sequence[Pairs]) -> Pairs | None:
        """All the parts as one, or ``None`` for no parts."""
        if not parts:
            return None
        return Pairs(
            np.concatenate([p.d for p in parts]),
            np.concatenate([p.z for p in parts]),
            np.concatenate([p.weight for p in parts]),
            np.concatenate([p.lift for p in parts]),
        )


def lift_of(rows: npt.ArrayLike, intr: Intrinsics) -> Array:
    """The elevation of the ray through ``rows`` above the optical axis, per unit depth."""
    out: Array = -(np.asarray(rows, dtype=float) - intr.cy) / intr.fy
    return out


@dataclass(frozen=True, eq=False)
class FrameContext:
    """What a frame brings besides the network's depth: the optics, the camera's place on the
    cart, which way is up (base_link), the lidar's returns as base_link points carried to the
    frame's moment (``None`` without a scan) and the frame's stamp in seconds."""

    intr: Intrinsics
    cam: CameraPose
    up: Array = field(default_factory=lambda: UP_LEVEL.copy())
    lidar: Array | None = None
    stamp: float = 0.0

    @cached_property
    def beams(self) -> Array | None:
        """The lidar's returns as pixels of this frame with their true depth (column, row,
        metres), inside the image only; ``None`` without lidar."""
        if self.lidar is None:
            return None
        return project(self.lidar, self.cam, self.intr)


@dataclass(eq=False)
class Frame:
    """A frame on its way through the pipeline: the network's raw depth, the frame's context,
    the pairs the anchors have contributed so far, and the raw depth's edge mask, computed
    once (relative, so it is the raw image's and every law's alike)."""

    raw: Array
    ctx: FrameContext
    pairs: list[Pairs] = field(default_factory=list)

    @cached_property
    def edge(self) -> Mask:
        """Pixels on a depth discontinuity of the raw depth (:func:`pepin.depth.edge_mask`)."""
        return edge_mask(self.raw)

    @property
    def pool(self) -> Pairs | None:
        """Every pair contributed so far, as one."""
        return Pairs.join(self.pairs)


@dataclass(frozen=True)
class Verdict:
    """What a stage did to one frame, for the report: whether it ran, the pairs it contributed,
    the pixels it changed, a note in its own words, and whether it withholds the frame (a law
    with nothing to apply yet: the raw network's depth must not go out)."""

    stage: str
    on: bool
    pairs: int = 0
    pixels: int = 0
    note: str = ""
    withhold: bool = False


# ---- the roles ----------------------------------------------------------------------------------
class Stage(Protocol):
    """One step of the pipeline: a depth image in, a depth image and a verdict out."""

    name: str

    def run(self, depth: Array, frame: Frame) -> tuple[Array, Verdict]:
        """Do the stage's work on ``depth`` (never in place) and say what was done."""
        ...

    def describe(self) -> str:
        """The stage's state in a few words, for the report line."""
        ...


class Law(Protocol):
    """The map from the network's depth to metres."""

    name: str

    @property
    def ready(self) -> bool:
        """Whether there is a law worth applying."""
        ...

    def fit(self, pairs: Pairs | None) -> None:
        """Feed a frame's pooled pairs (``None`` when no anchor had any) and refit."""
        ...

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The depth image in metres (NaN where the law cannot place a pixel)."""
        ...

    def describe(self) -> str:
        """The law's parameters in a few words."""
        ...


class Anchor(Protocol):
    """An external truth about some pixels of a frame."""

    name: str

    def pairs(self, frame: Frame) -> Pairs | None:
        """(network, true) pairs for the laws' pool, from the raw depth; ``None`` for none."""
        ...

    def correct(self, depth: Array, frame: Frame) -> tuple[Array, int]:
        """``depth`` with the pixels this anchor knows set right, and how many."""
        ...

    def describe(self) -> str:
        """The anchor's settings in a few words."""
        ...


class AnchorStage:
    """An anchor as a stage: its pairs join the frame's pool, its correction touches the
    depth. Subclasses fill in :meth:`pairs` and / or :meth:`correct`."""

    name: str = "anchor"

    def pairs(self, frame: Frame) -> Pairs | None:
        """No pairs unless a subclass says otherwise."""
        return None

    def correct(self, depth: Array, frame: Frame) -> tuple[Array, int]:
        """No correction unless a subclass says otherwise."""
        return depth, 0

    def describe(self) -> str:
        """Nothing to say unless a subclass has settings."""
        return ""

    def run(self, depth: Array, frame: Frame) -> tuple[Array, Verdict]:
        """Contribute the pairs, apply the correction."""
        found = self.pairs(frame)
        if found is not None and found.size:
            frame.pairs.append(found)
        out, touched = self.correct(depth, frame)
        return out, Verdict(
            self.name, True, pairs=0 if found is None else found.size, pixels=touched
        )


class LawStage:
    """A law as a stage: fitted on the frame's pool, applied to the depth; the frame is
    withheld while no law exists. Subclasses fill in :meth:`fit`, :meth:`apply`, ``ready``."""

    name: str = "law"

    @property
    def ready(self) -> bool:
        """Whether there is a law worth applying."""
        return False

    def fit(self, pairs: Pairs | None) -> None:
        """Feed a frame's pooled pairs and refit."""

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The depth in metres."""
        return depth

    def describe(self) -> str:
        """The law in a few words."""
        return ""

    def run(self, depth: Array, frame: Frame) -> tuple[Array, Verdict]:
        """Fit on the pool, apply — or withhold the frame while there is no law."""
        pool = frame.pool
        self.fit(pool)
        n = 0 if pool is None else pool.size
        if not self.ready:
            return depth, Verdict(self.name, True, pairs=n, note="no law yet", withhold=True)
        out = self.apply(depth, frame.ctx)
        return out, Verdict(self.name, True, pairs=n, pixels=int(out.size), note=self.describe())


# ---- the pipeline -----------------------------------------------------------------------------
@dataclass
class StageStats:
    """A stage's running totals: frames it ran on, pairs contributed, pixels changed, and the
    time it took (seconds in all, and the longest frame)."""

    frames: int = 0
    pairs: int = 0
    pixels: int = 0
    seconds: float = 0.0
    max_seconds: float = 0.0

    def add(self, verdict: Verdict, seconds: float) -> None:
        """Count one frame's verdict and its time."""
        self.frames += 1
        self.pairs += verdict.pairs
        self.pixels += verdict.pixels
        self.seconds += seconds
        self.max_seconds = max(self.max_seconds, seconds)

    @property
    def ms(self) -> tuple[float, float]:
        """(mean, max) milliseconds a frame took."""
        return (1e3 * self.seconds / self.frames if self.frames else 0.0, 1e3 * self.max_seconds)


@dataclass(eq=False)
class Result:
    """One frame's way through the pipeline: the final depth, each stage's verdict, the depth
    after every stage that ran (by name; references, the stages never write in place), the
    frame with its pool, and whether a law withheld it (nothing should be published)."""

    depth: Array
    verdicts: list[Verdict]
    after: dict[str, Array]
    frame: Frame
    withheld: bool

    def verdict(self, name: str) -> Verdict:
        """The verdict of the stage called ``name``."""
        for v in self.verdicts:
            if v.stage == name:
                return v
        raise KeyError(name)

    def before(self, name: str) -> Array:
        """The depth as it stood when the stage called ``name`` began: the raw depth when no
        stage ran before it, else the output of the last one that did (the node's scan is
        built from the depth before the floor anchor, so what stops the cart is what was
        measured). ``KeyError`` when the run never reached that stage."""
        out = self.frame.raw
        for v in self.verdicts:
            if v.stage == name:
                return out
            if v.on:
                out = self.after[v.stage]
        raise KeyError(name)


class DepthPipeline:
    """An ordered list of stages, each switchable by name (a node's flags), run on every
    frame with per-stage statistics; a stage switched off is skipped and says so in its
    verdict."""

    def __init__(self, stages: Sequence[Stage], off: Sequence[str] = ()) -> None:
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise ValueError(f"stage names must be unique: {names}")
        self._stages = list(stages)
        self._on = dict.fromkeys(names, True)
        self._stats = {n: StageStats() for n in names}
        for name in off:
            self.set(name, False)

    @property
    def names(self) -> list[str]:
        """The stages' names, in running order."""
        return [s.name for s in self._stages]

    def stage(self, name: str) -> Stage:
        """The stage called ``name``."""
        for s in self._stages:
            if s.name == name:
                return s
        raise KeyError(name)

    def set(self, name: str, on: bool) -> None:
        """Switch a stage on or off by name."""
        if name not in self._on:
            raise KeyError(name)
        self._on[name] = bool(on)

    def on(self, name: str) -> bool:
        """Whether the stage called ``name`` runs."""
        return self._on[name]

    @property
    def switches(self) -> dict[str, bool]:
        """Every stage's switch, in running order."""
        return dict(self._on)

    @property
    def stats(self) -> dict[str, StageStats]:
        """Every stage's running totals, by name."""
        return self._stats

    def reset_stats(self) -> None:
        """Start the totals afresh (a report window)."""
        self._stats = {n: StageStats() for n in self._on}

    def run(self, depth: Array, ctx: FrameContext) -> Result:
        """One frame through every stage that is on, in order; stops at a law that withholds."""
        frame = Frame(np.asarray(depth, dtype=float), ctx)
        out = frame.raw
        verdicts: list[Verdict] = []
        after: dict[str, Array] = {}
        for stage in self._stages:
            if not self._on[stage.name]:
                verdicts.append(Verdict(stage.name, False))
                continue
            t0 = time.perf_counter()
            out, verdict = stage.run(out, frame)
            self._stats[stage.name].add(verdict, time.perf_counter() - t0)
            verdicts.append(verdict)
            after[stage.name] = out
            if verdict.withhold:
                return Result(out, verdicts, after, frame, withheld=True)
        return Result(out, verdicts, after, frame, withheld=False)

    def report(self) -> str:
        """One line: every stage, on or off, its own words and its totals."""
        parts = []
        for stage in self._stages:
            st = self._stats[stage.name]
            state = "on" if self._on[stage.name] else "off"
            words = stage.describe()
            mean_ms, max_ms = st.ms
            parts.append(
                f"{stage.name} {state}"
                + (f" [{words}]" if words else "")
                + f" {st.frames} frames, {st.pairs} pairs, {st.pixels} px,"
                f" {mean_ms:.1f}/{max_ms:.1f} ms"
            )
        return "; ".join(parts)


# ---- today's stages ---------------------------------------------------------------------------
class EdgeFilter(AnchorStage):
    """Flying pixels dropped: the pixels on a depth discontinuity of the raw depth
    (:func:`pepin.depth.edge_mask`) become NaN. Off, the mask still keeps the beams off the
    edges (the anchors read ``Frame.edge`` regardless), as the node always did."""

    name = "edge_filter"

    def correct(self, depth: Array, frame: Frame) -> tuple[Array, int]:
        """The depth with its edge pixels NaN, and how many."""
        return drop_edges(depth, frame.edge)

    def describe(self) -> str:
        return f"step {EDGE_REL_STEP:.0%}"


class LidarAnchor(AnchorStage):
    """The lidar's beams: where a beam lands in the image, the network's raw depth pairs with
    the beam's true depth (:func:`pepin.depth.beam_hits`; edge pixels left out). Off, the laws
    get no pairs from the lidar and hold — the failure mode of a lidar that stops, and the
    measure of what the lidar buys."""

    name = "lidar_anchor"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def pairs(self, frame: Frame) -> Pairs | None:
        """The beams' pairs, or ``None`` without a scan or under MIN_SAMPLES clean hits."""
        beams = frame.ctx.beams
        if beams is None:
            return None
        hits = beam_hits(frame.raw, beams, frame.edge)
        if hits is None:
            return None
        cols = beams[hits, 0].astype(int)
        rows = beams[hits, 1].astype(int)
        return Pairs.of(
            frame.raw[rows, cols], beams[hits, 2], lift_of(rows, frame.ctx.intr), self.weight
        )

    def describe(self) -> str:
        return f"weight {self.weight:g}"


class AffineLaw(LawStage):
    """The affine law in inverse depth, 1 / z = a / D + b, fitted on the pairs of the last
    ``pool_frames`` frames (:func:`pepin.depth.fit_affine`, each pair by its weight): a turn's
    worth of pairs spans the room's depths, so the fit is conditioned where one frame's is not;
    a frame without pairs keeps the law. Until POOL_MIN_SAMPLES pairs are pooled there is no
    law worth applying (``ready`` is false: the raw network's depth is 1.5-2x too far and must
    not reach the costmap) — unless a saved law was ``seed``-ed, which holds until the live
    pool can replace it. Bit for bit :class:`pepin.depth.AffineScale` on unit weights."""

    name = "affine_law"

    def __init__(self, pool_frames: int = POOL_FRAMES) -> None:
        self.a = 1.0
        self.b = 0.0
        self.frames = 0
        self.held = 0
        self._pool: list[Pairs] = []
        self._pool_frames = pool_frames
        self._seeded = False

    def seed(self, a: float, b: float) -> None:
        """Start from a saved law (the map's): applied until POOL_MIN_SAMPLES live pairs exist."""
        self.a, self.b, self._seeded = a, b, True

    @property
    def ready(self) -> bool:
        """Whether a law worth applying exists: enough live pairs, or a seed."""
        return self._seeded or self.fitted

    @property
    def fitted(self) -> bool:
        """Whether the law rests on POOL_MIN_SAMPLES live pairs (worth saving)."""
        return self.pooled >= POOL_MIN_SAMPLES

    @property
    def pooled(self) -> int:
        """How many live pairs the pool holds."""
        return int(sum(p.size for p in self._pool))

    @property
    def pool(self) -> Pairs | None:
        """Everything in the pool, as one."""
        return Pairs.join(self._pool)

    def fit(self, pairs: Pairs | None) -> None:
        """Feed a frame's pairs (or ``None``); the law is refitted on the pool."""
        self.frames += 1
        if pairs is None or pairs.size == 0:
            self.held += 1
            return
        self._pool.append(pairs)
        del self._pool[: -self._pool_frames]
        if self._seeded and not self.fitted:
            return  # the map's law outranks a fit on a handful of pairs
        pool = self.pool
        assert pool is not None
        self.a, self.b = fit_affine(pool.d, pool.z, pool.weight)

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The depth through 1 / z = a / D + b."""
        return apply_affine(depth, self.a, self.b)

    def describe(self) -> str:
        source = "" if self.fitted else " (seed)" if self._seeded else " (none yet)"
        return f"a {self.a:.2f} b {self.b:+.3f} on {self.pooled} pairs{source}"


class FloorGeometry:
    """The floor's expected depth image for the current optics, mount and lean
    (:func:`pepin.depth.floor_depth`), recomputed only when the lean moves by more than
    LEAN_STEP or the optics change — the trigonometry over a whole image is milliseconds the
    frame rate would notice."""

    def __init__(self, lean_step: float = LEAN_STEP) -> None:
        self._lean_step = lean_step
        self._expected: Array | None = None
        self._up: Array | None = None
        self._key: tuple[Intrinsics, CameraPose] | None = None

    def expected(self, ctx: FrameContext) -> Array:
        """Per pixel, the depth its ray would have on the floor (NaN at and above the horizon)."""
        up = np.asarray(ctx.up, dtype=float)
        if (
            self._expected is None
            or self._up is None
            or self._key != (ctx.intr, ctx.cam)
            or float(np.linalg.norm(up - self._up)) > self._lean_step
        ):
            self._expected = floor_depth(ctx.intr, ctx.cam, up)
            self._up = up.copy()
            self._key = (ctx.intr, ctx.cam)
        return self._expected


class FloorAnchor(AnchorStage):
    """Pixels within ``tolerance`` of the floor plane snap to the plane's exact depth
    (:func:`pepin.depth.floor_anchor`), the plane leaning with the cart's up vector."""

    name = "floor_anchor"

    def __init__(
        self, geometry: FloorGeometry | None = None, tolerance: float = FLOOR_HEIGHT_TOLERANCE
    ) -> None:
        self.geometry = geometry if geometry is not None else FloorGeometry()
        self.tolerance = tolerance

    def correct(self, depth: Array, frame: Frame) -> tuple[Array, int]:
        """The depth with its floor pixels on the plane, and how many."""
        ctx = frame.ctx
        return floor_anchor(depth, self.geometry.expected(ctx), ctx.cam.z, self.tolerance)

    def describe(self) -> str:
        return f"tolerance {self.tolerance * 100:.0f} cm"


# ---- the floor as a second hoop ---------------------------------------------------------------
class FloorPairs(AnchorStage):
    """The floor pairs the network's depth with the plane's geometric depth: every
    ``stride``-th pixel whose ray meets the floor and that stands within the network's height
    band of the plane (:class:`pepin.contact.DepthNoise`, wider than the anchor's snap: these
    pairs feed a robust fit, not a correction). Which pixels are floor is judged with the law
    as it stands; before any law exists, with a scale bootstrapped from the pixels themselves —
    the median ratio of network to plane depth, re-taken over the pixels that ratio calls
    floor, three times. So the law can be fitted from the floor alone, with no lidar."""

    name = "floor_pairs"

    def __init__(
        self,
        law: Law,
        geometry: FloorGeometry | None = None,
        *,
        stride: int = FLOOR_PAIR_STRIDE,
        band: DepthNoise = DEPTH_NOISE,
        weight: float = FLOOR_PAIR_WEIGHT,
    ) -> None:
        self.law = law
        self.geometry = geometry if geometry is not None else FloorGeometry()
        self.stride = stride
        self.band = band
        self.weight = weight

    def pairs(self, frame: Frame) -> Pairs | None:
        """(network, floor) pairs of the frame's floor pixels, or ``None`` under MIN_SAMPLES."""
        ctx = frame.ctx
        s = self.stride
        expected = self.geometry.expected(ctx)
        raw = frame.raw
        metric = self.law.apply(raw, ctx)[::s, ::s] if self.law.ready else None
        expected, raw, edge = expected[::s, ::s], raw[::s, ::s], frame.edge[::s, ::s]
        rows = np.arange(0, frame.raw.shape[0], s)[:, None] * np.ones_like(raw, dtype=int)
        ok = np.isfinite(expected) & np.isfinite(raw) & (raw > NEAR_M) & ~edge
        if int(ok.sum()) < MIN_SAMPLES:
            return None
        band = self.band.height_band(expected, ctx.cam.z)
        if metric is None:
            scale = float(np.median(raw[ok] / expected[ok]))
            floor = ok
            for _ in range(3):
                height = ctx.cam.z * (1.0 - raw / (scale * expected))
                floor = ok & (np.abs(height) < band)
                if int(floor.sum()) < MIN_SAMPLES:
                    return None
                scale = float(np.median(raw[floor] / expected[floor]))
        else:
            with np.errstate(invalid="ignore"):
                height = ctx.cam.z * (1.0 - metric / expected)
            floor = ok & np.isfinite(height) & (np.abs(height) < band)
        if int(floor.sum()) < MIN_SAMPLES:
            return None
        return Pairs.of(raw[floor], expected[floor], lift_of(rows[floor], ctx.intr), self.weight)

    def describe(self) -> str:
        return f"stride {self.stride}, weight {self.weight:g}"


# ---- the walls as a third hoop ----------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class WallWalk:
    """Where the lidar's returns were extruded up the picture: the column of every usable
    return, the (rows, returns) mask of the rows walked above it, and the geometric depth of
    the vertical surface at every such row."""

    cols: npt.NDArray[np.intp]
    walked: Mask
    depth: Array

    @property
    def count(self) -> int:
        """How many pixels the walk covered."""
        return int(self.walked.sum())


class WallAnchor(AnchorStage):
    """The lidar's returns extruded up the image: a return and its scan neighbours give the
    local direction of the surface they lie on; the vertical plane through that line is the
    wall, and for the rows above the return in its column — as long as the network's depth
    stays continuous from row to row (a step over ``rel_step`` of itself is another object:
    a chair back ends at its top, a wall goes on) and climbs the column no faster than the
    plane's own depth does (a table top or a seat is continuous with its front but recedes
    a percent a row where a vertical surface moves a tenth of that: the log-depth slopes over
    ``slope_window`` rows may differ by ``slope_tol`` a row) — the pixel's ray meets that
    plane at a depth the geometry knows. With ``pairs`` those (network, wall) pairs are a
    third hoop above the lidar's row; with ``correct`` the walked pixels are set to the
    wall's depth outright."""

    name = "wall_anchor"

    def __init__(
        self,
        *,
        rel_step: float = EDGE_REL_STEP,
        row_stride: int = WALL_ROW_STRIDE,
        weight: float = WALL_PAIR_WEIGHT,
        max_height: float = WALL_MAX_HEIGHT,
        neighbour_gap: float = WALL_NEIGHBOUR_GAP,
        slope_tol: float = WALL_SLOPE_TOL,
        slope_window: int = WALL_SLOPE_WINDOW,
        pairs: bool = True,
        correct: bool = False,
    ) -> None:
        self.rel_step = rel_step
        self.row_stride = row_stride
        self.weight = weight
        self.max_height = max_height
        self.neighbour_gap = neighbour_gap
        self.slope_tol = slope_tol
        self.slope_window = slope_window
        self.contribute = pairs
        self.correct_pixels = correct

    def walk(self, frame: Frame) -> WallWalk | None:
        """Extrude every usable return up its column; ``None`` without lidar or returns."""
        ctx = frame.ctx
        points = ctx.lidar
        if points is None or points.shape[0] < 3:
            return None
        intr, cam = ctx.intr, ctx.cam
        u, v, forward = project_all(points, cam, intr)
        inside = in_image(u, v, forward, intr)
        # the surface's direction at each return: the chord between its scan neighbours
        xy = points[:, :2]
        prev = np.roll(xy, 1, axis=0)
        nxt = np.roll(xy, -1, axis=0)
        near_prev = np.hypot(*(xy - prev).T) < self.neighbour_gap
        near_next = np.hypot(*(nxt - xy).T) < self.neighbour_gap
        near_prev[0] = near_next[-1] = False  # the roll wraps the scan's ends onto each other
        chord = np.where(
            (near_prev & near_next)[:, None],
            nxt - prev,
            np.where(near_next[:, None], nxt - xy, xy - prev),
        )
        usable = inside & (near_prev | near_next)
        if int(usable.sum()) == 0:
            return None
        idx = np.flatnonzero(usable)
        tangent = chord[idx] / np.linalg.norm(chord[idx], axis=1)[:, None]
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)  # horizontal, unit
        cols = u[idx].astype(int)
        rows0 = v[idx].astype(int)
        h = frame.raw.shape[0]
        raw = frame.raw[:, cols]  # (h, n): each return's column
        lift = lift_of(np.arange(h), intr)[:, None]
        left = -(cols[None, :] - intr.cx) / intr.fx
        c, s = np.cos(cam.pitch), np.sin(cam.pitch)
        shape = (h, cols.size)
        dx = np.broadcast_to(c + s * lift, shape)
        dz = np.broadcast_to(-s + c * lift, shape)
        dy = np.broadcast_to(left, shape)
        n_dot_d = normal[None, :, 0] * dx + normal[None, :, 1] * dy
        offset = normal[:, 0] * (xy[idx, 0] - cam.x) + normal[:, 1] * (xy[idx, 1] - cam.y)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = offset[None, :] / n_dot_d
            z_up = cam.z + t * dz
            step = np.abs(raw[:-1] - raw[1:]) / raw[1:]
        bad = ~np.isfinite(raw) | ~np.isfinite(t) | (t <= NEAR_M) | (z_up > self.max_height)
        bad[:-1] |= ~np.isfinite(step) | (step > self.rel_step)
        w = self.slope_window
        if w > 0 and h > 2 * w:
            with np.errstate(divide="ignore", invalid="ignore"):
                ln_d, ln_t = np.log(raw), np.log(t)
                climb_d = (ln_d[: -2 * w] - ln_d[2 * w :]) / (2 * w)  # up the column, per row
                climb_t = (ln_t[: -2 * w] - ln_t[2 * w :]) / (2 * w)
                apart = np.abs(climb_d - climb_t)
            bad[w:-w] |= ~np.isfinite(apart) | (apart > self.slope_tol)
        # rows above the return with no bad row between: the count of bad rows at or above a
        # row, less the count at or above the return, must be zero
        above = np.vstack([np.cumsum(bad[::-1], axis=0)[::-1], np.zeros((1, bad.shape[1]))])
        at_return = above[rows0, np.arange(rows0.size)]
        row = np.arange(h)[:, None]
        walked: Mask = (row < rows0[None, :]) & (above[:h] - at_return[None, :] == 0)
        return WallWalk(cols, walked, t)

    def pairs(self, frame: Frame) -> Pairs | None:
        """(network, wall) pairs every ``row_stride`` rows of the walk, or ``None`` under
        MIN_SAMPLES or with ``pairs`` off."""
        if not self.contribute:
            return None
        walk = self.walk(frame)
        if walk is None:
            return None
        rows_all = np.arange(frame.raw.shape[0])[:, None]
        rows0 = walk.walked.shape[0] - 1 - np.argmax(walk.walked[::-1], axis=0)
        strided = walk.walked & ((rows0[None, :] - rows_all) % self.row_stride == 0)
        r, k = np.nonzero(strided)
        if r.size < MIN_SAMPLES:
            return None
        return Pairs.of(
            frame.raw[r, walk.cols[k]], walk.depth[r, k], lift_of(r, frame.ctx.intr), self.weight
        )

    def correct(self, depth: Array, frame: Frame) -> tuple[Array, int]:
        """With ``correct`` on, the walked pixels at the wall's depth; else the depth as is."""
        if not self.correct_pixels:
            return depth, 0
        walk = self.walk(frame)
        if walk is None:
            return depth, 0
        out = np.asarray(depth, dtype=float).copy()
        r, k = np.nonzero(walk.walked)
        out[r, walk.cols[k]] = walk.depth[r, k]
        return out, int(r.size)

    def describe(self) -> str:
        roles = ("pairs " if self.contribute else "") + (
            "correcting" if self.correct_pixels else ""
        )
        return (
            f"step {self.rel_step:.0%}, slope {self.slope_tol:.1%}/row, weight {self.weight:g},"
            f" {roles.strip() or 'idle'}"
        )


class WallCorrection(WallAnchor):
    """The wall anchor's correcting role as its own stage, after the law: the walked pixels
    are set to the extruded plane's metric depth, which a law run afterwards would scale
    again as if it were the network's (the pairs role stays before the law, where the raw
    depth is)."""

    name = "wall_correct"

    def __init__(self, **settings: float | int | bool) -> None:
        super().__init__(**{**settings, "pairs": False, "correct": True})  # type: ignore[arg-type]


# ---- laws that read the elevation -------------------------------------------------------------
class ElevationLaw(AffineLaw):
    """The affine law with a term in the ray's elevation, 1 / z = a / D + b + c * lift, for an
    error that grows up the picture. Fitted the way the affine law is (the noisy 1 / D
    regressed on the exact 1 / z and lift, then inverted). Each term has its own gate: ``c``
    needs the pool to span elevations (MIN_LIFT_SPREAD between its 5th and 95th lift
    percentiles), ``b`` needs it to span depths (MIN_DEPTH_SPREAD, as the affine law) and is
    0 otherwise; the fit falls back to the affine law when its (a, b) leave the bounds."""

    name = "elevation_law"

    def __init__(self, pool_frames: int = POOL_FRAMES) -> None:
        super().__init__(pool_frames)
        self.c = 0.0

    def fit(self, pairs: Pairs | None) -> None:
        """The affine fit, then the elevation term when the pool can carry it."""
        super().fit(pairs)
        pool = self.pool
        self.c = 0.0
        if pool is None or not self.fitted:
            return
        lo, hi = np.percentile(pool.lift, (5, 95))
        z_lo, z_hi = np.percentile(pool.z, (5, 95))
        if hi - lo < MIN_LIFT_SPREAD:
            return
        with_shift = z_hi / z_lo >= MIN_DEPTH_SPREAD
        x, y = 1.0 / pool.d, 1.0 / pool.z
        w = np.sqrt(pool.weight)
        columns = [y, pool.lift] + ([np.ones_like(y)] if with_shift else [])
        design = np.stack(columns, axis=1) * w[:, None]
        coef, *_ = np.linalg.lstsq(design, x * w, rcond=None)
        res = np.abs(x - (coef[0] * y + coef[1] * pool.lift + (coef[2] if with_shift else 0.0)))
        keep = res <= np.percentile(res, 75)
        if int(keep.sum()) >= 4:
            coef, *_ = np.linalg.lstsq(design[keep], (x * w)[keep], rcond=None)
        alpha, gamma = float(coef[0]), float(coef[1])
        beta = float(coef[2]) if with_shift else 0.0
        if alpha == 0.0:
            return
        a, b, c = 1.0 / alpha, -beta / alpha, -gamma / alpha
        if A_BOUNDS[0] <= a <= A_BOUNDS[1] and B_BOUNDS[0] <= b <= B_BOUNDS[1]:
            self.a, self.b, self.c = a, b, c

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The depth through the law, the elevation of each row from the optics."""
        d = np.asarray(depth, dtype=float)
        lift = lift_of(np.arange(d.shape[0]), ctx.intr)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = self.a / d + self.b + self.c * lift
            z = np.where(np.isfinite(inv) & (inv > 1e-6), 1.0 / inv, np.nan)
        out: Array = z
        return out

    def describe(self) -> str:
        return super().describe() + f" c {self.c:+.3f}"


class RowLaw(AffineLaw):
    """A law per band of elevation: the pool's lift range cut into ``bands`` equal bands, an
    affine law fitted in each band that holds POOL_MIN_SAMPLES pairs (the whole pool's law
    elsewhere), the bands' (a, b) interpolated along the rows at apply time — the shape a
    row-dependent error would take, held against the elevation term's straight line."""

    name = "row_law"

    def __init__(self, pool_frames: int = POOL_FRAMES, bands: int = ROW_BANDS) -> None:
        super().__init__(pool_frames)
        self.bands = bands
        self.centres: Array = np.zeros(0)
        self.a_of: Array = np.zeros(0)
        self.b_of: Array = np.zeros(0)

    def fit(self, pairs: Pairs | None) -> None:
        """The affine fit, then one per band of elevation."""
        super().fit(pairs)
        pool = self.pool
        if pool is None or not self.fitted:
            self.centres = np.zeros(0)
            return
        edges = np.linspace(pool.lift.min(), pool.lift.max() + 1e-9, self.bands + 1)
        centres, a_of, b_of = [], [], []
        for lo, hi in pairwise(edges):
            sel = (pool.lift >= lo) & (pool.lift < hi)
            if int(sel.sum()) >= POOL_MIN_SAMPLES:
                a, b = fit_affine(pool.d[sel], pool.z[sel], pool.weight[sel])
            else:
                a, b = self.a, self.b
            centres.append(0.5 * (lo + hi))
            a_of.append(a)
            b_of.append(b)
        self.centres, self.a_of, self.b_of = np.array(centres), np.array(a_of), np.array(b_of)

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The depth through the band laws interpolated along the rows."""
        d = np.asarray(depth, dtype=float)
        if self.centres.size == 0:
            return apply_affine(d, self.a, self.b)
        lift = lift_of(np.arange(d.shape[0]), ctx.intr)
        a = np.interp(lift, self.centres, self.a_of)[:, None]
        b = np.interp(lift, self.centres, self.b_of)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = a / d + b
            z = np.where(np.isfinite(inv) & (inv > 1e-6), 1.0 / inv, np.nan)
        out: Array = z
        return out

    def describe(self) -> str:
        if self.centres.size == 0:
            return super().describe()
        bands = " ".join(f"{a:.2f}/{b:+.3f}" for a, b in zip(self.a_of, self.b_of, strict=True))
        return f"{self.pooled} pairs, bands a/b {bands}"


# ---- the chain ----------------------------------------------------------------------------------
def standard_pipeline(
    law: LawStage | None = None,
    *,
    floor_pairs: bool = False,
    wall_anchor: bool = False,
    wall_correct: bool = False,
) -> DepthPipeline:
    """The node's chain: edges -> lidar -> (floor pairs) -> (wall pairs) -> law -> (wall
    correction) -> floor anchor; the three new stages are in the list and switched by the
    flags of the same name (``wall_anchor`` is the pairs role, ``wall_correct`` the pixels)."""
    the_law = law if law is not None else AffineLaw()
    geometry = FloorGeometry()
    stages: list[Stage] = [
        EdgeFilter(),
        LidarAnchor(),
        FloorPairs(the_law, geometry),
        WallAnchor(),
        the_law,
        WallCorrection(),
        FloorAnchor(geometry),
    ]
    flags = (
        ("floor_pairs", floor_pairs),
        ("wall_anchor", wall_anchor),
        ("wall_correct", wall_correct),
    )
    return DepthPipeline(stages, off=[name for name, on in flags if not on])


__all__ = [
    "AffineLaw",
    "Anchor",
    "AnchorStage",
    "DepthPipeline",
    "EdgeFilter",
    "ElevationLaw",
    "FloorAnchor",
    "FloorGeometry",
    "FloorPairs",
    "Frame",
    "FrameContext",
    "Law",
    "LawStage",
    "LidarAnchor",
    "Pairs",
    "Result",
    "RowLaw",
    "Stage",
    "StageStats",
    "Verdict",
    "WallAnchor",
    "WallCorrection",
    "WallWalk",
    "lift_of",
    "standard_pipeline",
]
