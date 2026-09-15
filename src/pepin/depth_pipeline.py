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
top), a third hoop above the lidar's row; and :class:`ParallaxAnchor` — the corners this frame
shares with the previous one, triangulated against the odometry's transform between the two
stamps (:mod:`pepin.parallax`), a fourth hoop that needs no second sensor and no assumed plane
and that measures at every elevation the picture has. Two more laws let the data say whether the
error above the lidar's row depends on the elevation: :class:`ElevationLaw` adds a term in the
ray's lift, :class:`RowLaw` fits a law per band of rows.

Measured on run 0171 (29 frames against the run's lidar cloud and its COLMAP reference,
scratch/pipeline_vs_truth.py, 2026-09-11), which set the defaults of :func:`standard_pipeline`:
the network's error is not one law — 1.1x too far on the floor, 1.6x at the lidar's row, 2.0x
from 0.3 m up — so every hoop but the lidar's pulls the law off the row the costmap lives on.
Floor pairs alone put the lidar's row 2.0x too far (a camera without lidar sees walls where the
raw network does); wall pairs at 0.2 of a beam put it 10 % too near while fixing the 0.5-0.8 m
slice (0.86 -> 1.00 of the truth); the elevation term is real (c +0.1) but linear in lift is the
wrong shape (the error steps within 60 rows of the lidar's row and is flat above), and the row
law overfits. The lidar-only anchor stays the default; the new anchors ship switched off.

The law that ships live is :class:`RangeLawStage`, not the affine one: fitted on the same pool,
it bins the pairs by the network's own depth and measures a ratio in each bin, because the
affine law's residual tilts 12 % per metre of range (:class:`pepin.depth.RangeLaw`, measured
2026-09-14 in scratch/depth_scale_by_range.py). The affine law keeps running before it — it is
what the law file carries, what seeds every law at start, and the fallback the range law
publishes through until two of its bins fill.

:class:`RayLaw` (:mod:`pepin.elevation`) is the third such law and the one parameterised the
way the error is: by the ray's angle off the optical axis, so the neck may tilt without
refitting. Measured held out on the same 29 frames (scratch/ray_law_eval.txt, 2026-09-12) it
tightens the scatter of the beams' residual on three drive halves of four (q3 0.46 -> 0.13 on
the widest) and moves the median 5-10 % near, and its scale reproduces between two tapes of the
same room at the bottom and the middle of the frame (12 % and 8 % apart) but not at the top
(30 %). It ships switched off, and on the lidar's beams alone it is refused outright: a
return's elevation is a curve of its range, so there the angle and the depth are one regressor
(:func:`pepin.elevation.separable`) and the affine law stands instead.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from pepin.contact import DEPTH_NOISE, DepthNoise
from pepin.depth import (
    B_BOUNDS,
    EDGE_REL_STEP,
    FLOOR_HEIGHT_TOLERANCE,
    FRAME_HOLD_TAU_S,
    FRAME_MIN_PAIRS,
    FRAME_MIN_SPREAD,
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
    RangeLaw,
    a_bounds,
    apply_affine,
    at_bound,
    beam_hits,
    drop_edges,
    edge_mask,
    fit_affine,
    fit_frame,
    floor_anchor,
    floor_depth,
    in_image,
    inverse_sigma,
    pair_weight,
    project,
    project_all,
)
from pepin.elevation import RAY_AZIMUTH_DEGREE, RAY_DEGREE, RayGain, fit_ray, ray_angles

if TYPE_CHECKING:  # the tracker's own module stays a lazy import inside the parallax stage
    from pepin.parallax import Features, Motion, ParallaxTruth

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
PARALLAX_MIN_GAP_S = 0.08  # a partner frame nearer in time than this has no baseline to speak of
PARALLAX_MAX_GAP_S = 0.60  # farther back than this the view has changed more than the flow follows
PARALLAX_ORB_MAX_GAP_S = 1.5  # the describer's window: a keypoint is recognised, not followed
PARALLAX_MIN_BASELINE_M = 0.10  # the parallax a partner is chosen to reach: 0.4 s at 0.25 m/s
PARALLAX_MATCHER = "klt"  # who finds the correspondences: the flow or the describer
PARALLAX_MOTION = "tracker"  # whose word on the baseline: the lidar tracker's map pose, or odometry
PARALLAX_MAP_WAIT = False  # ask the map pose without waiting: a wait costs the whole frame rate
PARALLAX_MAP_MAX_AGE_S = 0.3  # a map pose older than this is not this frame's: odometry answers
PARALLAX_MOTIONS = ("tracker", "odom")
PARALLAX_RING_FRAMES = 24  # frames kept to choose a partner from: 1.5 s at any rate the node runs
PARALLAX_WEIGHT = 1.0  # the multiplier on a parallax pair's own 1 / sigma^2 (the A/B's knob)
PARALLAX_TRACK_MIN_OBS = 3  # frames a corner must be seen in to be a track; 2 is the old pair
PARALLAX_TRACK_WINDOW_S = 1.5  # how far back a track reaches, seconds: the ring's own span
PARALLAX_MIN_TOTAL_BASELINE_M = 0.10  # the effective parallax a track's views must add up to
# The three above are pepin.parallax's TRACK_* defaults, restated here so the pipeline's
# constants read in one place (the tracker's own module stays a lazy import in the stage).
LIDAR_SIGMA_M = 0.0  # metres: one beam's range noise, 0 meaning every beam weighs a flat 1 —
# the reference pair itself (REF_SIGMA_INV, "a beam at 2 m"), which is what the parallax anchor
# has always weighed its corners against, so the two rulers still share one unit.
# Measured 2026-09-15 (scratch/parallax_ruler_recheck.txt, the four errands of 2026-09-14,
# 112 frames, odd beams fitting and even beams judging): weighing each beam 1 / sigma^2 at
# sigma_m = 1.5 cm makes the LIDAR-ONLY law worse, 7.4 % -> 11.4 % of median |residual| overall
# and 6.4 % -> 18.3 % over 1.0-1.5 m, because sigma_m / z^2 puts the weight as z^4 (a beam at
# 8 m counts 256 beams at 2 m) and the far beams then fit themselves: 3-12 m improves
# 10.5 % -> 4.4 % and everything the cart parks against loses. The reason is that the fit
# minimises the residual of the NETWORK's 1 / D, whose own noise (0.02-0.10 of inverse depth at
# a few per cent of range) dwarfs a beam's (0.0002-0.023) everywhere — so a beam's sigma is not
# the residual's sigma, and 1 / sigma_beam^2 is not that pair's weight in this fit.
# Live knob: the node's lidar_sigma_m, > 0 to weigh the beams by range again.


# ---- what flows through the pipeline ----------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Pairs:
    """What an anchor knows about some pixels: the network's depth ``d`` there, the true depth
    ``z`` (metres along the optical axis), each pair's ``weight`` in a fit (a lidar beam is 1),
    and where the ray points — its ``lift``, the elevation above the optical axis per unit
    depth, ``-(row - cy) / fy``, and its ``left``, the same for the azimuth,
    ``-(column - cx) / fx`` (zero where an anchor does not say). The angle-aware laws read
    those two: the tangents of the ray's angles off the optical axis, the camera's own
    parameter, which no tilt of the neck moves."""

    d: Array
    z: Array
    weight: Array
    lift: Array
    left: Array

    @classmethod
    def of(
        cls,
        d: Array,
        z: Array,
        lift: Array,
        weight: float | Array = 1.0,
        left: Array | None = None,
    ) -> Pairs:
        """Pairs from arrays, ``weight`` one number for all or one per pair, ``left`` zero
        (the optical axis' own column) when the anchor does not know the azimuth."""
        dd = np.asarray(d, dtype=float)
        w = (
            np.full(dd.shape, float(weight))
            if np.ndim(weight) == 0
            else np.asarray(weight, dtype=float)
        )
        lifted = np.asarray(lift, dtype=float)
        sideways = np.zeros_like(dd) if left is None else np.asarray(left, dtype=float)
        return cls(dd, np.asarray(z, dtype=float), w, lifted, sideways)

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
            np.concatenate([p.left for p in parts]),
        )


def lift_of(rows: npt.ArrayLike, intr: Intrinsics) -> Array:
    """The elevation of the ray through ``rows`` above the optical axis, per unit depth."""
    out: Array = -(np.asarray(rows, dtype=float) - intr.cy) / intr.fy
    return out


def left_of(columns: npt.ArrayLike, intr: Intrinsics) -> Array:
    """The azimuth of the ray through ``columns`` left of the optical axis, per unit depth."""
    out: Array = -(np.asarray(columns, dtype=float) - intr.cx) / intr.fx
    return out


class Rigid(Protocol):
    """A rigid transform: a 3x3 rotation and a translation (:class:`pepin.tsdf.RigidPose`)."""

    @property
    def rotation(self) -> Array:
        """The 3x3 rotation."""
        ...

    @property
    def translation(self) -> Array:
        """The translation."""
        ...


class MotionSource(Protocol):
    """Who can say how the cart moved between two moments —
    :meth:`pepin.frame_pose.FramePoser.motion` in a node, a tape offline, a fake in a test."""

    def motion(self, from_stamp: float, to_stamp: float) -> Rigid | None:
        """base_link at ``from_stamp`` into base_link at ``to_stamp``, or ``None`` when the
        odometry does not cover both moments."""
        ...


@runtime_checkable
class MapMotionSource(Protocol):
    """A motion source that can also answer through the map, where the lidar tracker's
    corrections live (:meth:`pepin.frame_pose.FramePoser.map_motion`) — what the parallax
    anchor asks for a baseline over a whole second. A source without the method is simply an
    odometry-only one; the anchor falls back to it."""

    def map_motion(self, from_stamp: float, to_stamp: float) -> Rigid | None:
        """base_link at ``from_stamp`` into base_link at ``to_stamp`` through the map, or
        ``None`` when the map pose does not cover both moments (no tracker, or a stale one)."""
        ...


@runtime_checkable
class RecentMapMotionSource(Protocol):
    """A map motion source that can answer WITHOUT waiting — the only kind the frame path may
    ask, because a TF lookup that cannot be answered costs its whole timeout inside the frame.

    Live on 2026-09-15: the blocking ask waited the node's 0.2 s on every frame (map -> base_link
    is not in TF at a frame's stamp yet), the stream fell from 8.7 to 1.5 frames/s and
    rgbd_odometry starved to 0 poses/s. The recent ask uses the newest map pose already held,
    when it is fresh enough to be this frame's, and says no at once when it is not."""

    def map_motion_recent(
        self, from_stamp: float, to_stamp: float, max_age_s: float
    ) -> Rigid | None:
        """base_link at ``from_stamp`` into base_link at ``to_stamp`` through the map, using only
        poses already held and only when the newest is within ``max_age_s`` of ``to_stamp``;
        ``None`` at once otherwise. Never waits."""
        ...


@dataclass(frozen=True, eq=False)
class FrameContext:
    """What a frame brings besides the network's depth: the optics, the camera's place on the
    cart, which way is up (base_link), the lidar's returns as base_link points carried to the
    frame's moment (``None`` without a scan), the frame's stamp in seconds — and, for the
    anchors that read the picture rather than a sensor, the frame's grey image, the source of
    the cart's motion between stamps, and TF's whole ``base_link <- camera_optical`` edge, the
    one place a panned neck is carried (all three ``None`` unless a node supplies them)."""

    intr: Intrinsics
    cam: CameraPose
    up: Array = field(default_factory=lambda: UP_LEVEL.copy())
    lidar: Array | None = None
    stamp: float = 0.0
    gray: npt.NDArray[np.uint8] | None = None
    motion: MotionSource | None = None
    cam_optical: Rigid | None = None

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
    sources: list[str] = field(default_factory=list)  # which anchor each block of pairs is from

    def add(self, source: str, found: Pairs) -> None:
        """Take one anchor's pairs into the pool, remembering whose they are (the report line
        says how much of the frame's weight each ruler brought)."""
        self.pairs.append(found)
        self.sources.append(source)

    @property
    def rulers(self) -> dict[str, float]:
        """How much fit weight each anchor contributed to this frame, by the anchor's name —
        the sum of its pairs' weights, which is the only honest measure of who is fitting the
        law when one ruler brings 30 pairs at weight 1 and another 200 at weight 0.03."""
        out: dict[str, float] = {}
        for name, part in zip(self.sources, self.pairs, strict=False):
            out[name] = out.get(name, 0.0) + float(np.sum(part.weight))
        return out

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
            frame.add(self.name, found)
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
    measure of what the lidar buys.

    A beam weighs a flat 1 by default — the reference pair every other ruler is weighed against
    (:data:`pepin.depth.REF_SIGMA_INV`), so the parallax anchor's corners can share the pool
    with the beams in one unit. ``sigma_m`` above 0 weighs each beam ``1 / sigma^2`` in inverse
    depth instead (``sigma_m / z^2``, :func:`pepin.depth.pair_weight`), which reads as a weight
    proportional to ``z^4`` and measured WORSE on the beams alone (:data:`LIDAR_SIGMA_M`); it is
    kept as the live ``lidar_sigma_m`` knob, not as the default."""

    name = "lidar_anchor"

    def __init__(self, weight: float = 1.0, sigma_m: float = LIDAR_SIGMA_M) -> None:
        self.weight = weight
        self.sigma_m = sigma_m

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
        intr = frame.ctx.intr
        z = beams[hits, 2]
        weight: float | Array = self.weight
        if self.sigma_m > 0.0:
            weight = self.weight * pair_weight(inverse_sigma(self.sigma_m, z))
        return Pairs.of(
            frame.raw[rows, cols],
            z,
            lift_of(rows, intr),
            weight,
            left_of(cols, intr),
        )

    def describe(self) -> str:
        """The weight a beam carries: its own 1 / sigma^2 at ``sigma_m`` of range noise, or the
        flat number every beam shared before."""
        if self.sigma_m <= 0.0:
            return f"weight {self.weight:g} flat"
        return f"weight {self.weight:g} / sigma^2, sigma {self.sigma_m * 100:.1f} cm"


class AffineLaw(LawStage):
    """The affine law in inverse depth, 1 / z = a / D + b, fitted on the pairs of the last
    ``pool_frames`` frames (:func:`pepin.depth.fit_affine`, each pair by its weight): a turn's
    worth of pairs spans the room's depths, so the fit is conditioned where one frame's is not;
    a frame without pairs keeps the law. Until POOL_MIN_SAMPLES pairs are pooled there is no
    law worth applying (``ready`` is false: the raw network's depth is 1.5-2x too far and must
    not reach the costmap) — unless a saved law was ``seed``-ed, which holds until the live
    pool can replace it.

    ``slew_per_s`` caps how fast the law may move: the largest relative change of the published
    inverse depth over the pool's own depth range, per second (0 applies every fit whole, which
    is bit for bit :class:`pepin.depth.AffineScale` on unit weights). The pool is a queue of
    frames, not of seconds, so at 9.4 frames/s a 600-frame pool is 64 s deep and half a minute
    of driving replaces half of it; and the shift term switches on and off with the pool's
    depth spread (:data:`pepin.depth.MIN_DEPTH_SPREAD`), so the same pairs are described first
    as a scale alone and then as a scale and a shift. Both were seen on 2026-09-14: a 1.74 b 0
    standing, a 2.33 b -0.200 within 30 s of driving, and back — while the volume kept the
    paint of whichever law was in force."""

    name = "affine_law"

    def __init__(
        self,
        pool_frames: int = POOL_FRAMES,
        slew_per_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.a = 1.0
        self.b = 0.0
        self.frames = 0
        self.held = 0
        self.slew_per_s = slew_per_s
        self._pool: list[Pairs] = []
        self._pool_frames = pool_frames
        self._seeded = False
        self._clock = clock
        self._last_fit: float | None = None  # when the law last moved, for the slew's seconds
        self._asked: tuple[float, float] | None = None  # the fit the slew is still walking to

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
        self.a, self.b = self._toward(*fit_affine(pool.d, pool.z, pool.weight), pool)

    def _toward(self, a_fit: float, b_fit: float, pool: Pairs) -> tuple[float, float]:
        """The law to hold now, walking toward the fresh fit no faster than ``slew_per_s``.

        The step is measured where it hurts: the largest relative move of the published inverse
        depth over the pool's own depth range (its 5th and 95th percentiles of network depth),
        not in ``a``, because the fit's two parameters trade against each other and agree near
        the pool's middle while they differ at its ends. Over the allowance the law takes the
        blend of old and new that exactly spends it — a blend of two affine laws is affine.
        ``slew_per_s`` at or below zero, or the first live fit (nothing to walk from), applies
        the fit whole."""
        now = self._clock()
        last, self._last_fit = self._last_fit, now
        if self.slew_per_s <= 0.0 or last is None:
            self._asked = None
            return a_fit, b_fit
        step = self._reach(a_fit, b_fit, pool)
        allowed = self.slew_per_s * max(0.0, now - last)
        if step <= allowed:
            self._asked = None
            return a_fit, b_fit
        self._asked = (a_fit, b_fit)
        t = allowed / step
        return self.a + t * (a_fit - self.a), self.b + t * (b_fit - self.b)

    def _reach(self, a_fit: float, b_fit: float, pool: Pairs) -> float:
        """How far the fresh law is from the one in hand: the largest relative change of the
        published inverse depth at the pool's 5th and 95th percentiles of network depth."""
        d = pool.d[np.isfinite(pool.d) & (pool.d > 0.0)]
        if d.size == 0:
            return 0.0
        ends = np.percentile(d, (5, 95))
        old = self.a / ends + self.b
        new = a_fit / ends + b_fit
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(new - old) / np.abs(old)
        return float(np.max(rel[np.isfinite(rel)], initial=0.0))

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The depth through 1 / z = a / D + b."""
        return apply_affine(depth, self.a, self.b)

    def describe(self) -> str:
        """The law for the report line: its two numbers, the pairs behind them, where they came
        from, and — when a parameter sits on its bound — that the pool asked for more than the
        bounds allow, so the number is a ceiling and not a fit."""
        source = "" if self.fitted else " (seed)" if self._seeded else " (none yet)"
        clipped = at_bound(self.a, self.b)
        edge = f" [{clipped} AT BOUND]" if clipped else ""
        asked = ""
        if self._asked is not None:
            asked = f", slewing to a {self._asked[0]:.2f} b {self._asked[1]:+.3f}"
        return f"a {self.a:.2f} b {self.b:+.3f} on {self.pooled} pairs{source}{edge}{asked}"


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
        cols = np.arange(0, frame.raw.shape[1], s)[None, :] * np.ones_like(raw, dtype=int)
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
        return Pairs.of(
            raw[floor],
            expected[floor],
            lift_of(rows[floor], ctx.intr),
            self.weight,
            left_of(cols[floor], ctx.intr),
        )

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
        intr = frame.ctx.intr
        return Pairs.of(
            frame.raw[r, walk.cols[k]],
            walk.depth[r, k],
            lift_of(r, intr),
            self.weight,
            left_of(walk.cols[k], intr),
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


# ---- motion as a hoop, with no lidar at all ---------------------------------------------------
@dataclass(eq=False)
class PreviousFrame:
    """A frame the parallax anchor keeps in its ring: its grey image, its stamp and where the
    lens sat on the cart when it was taken — the whole ``base_link <- camera_optical``, pan
    included, since the neck may have turned since — plus the describer's reading of the
    picture once it has been asked for, so a frame a dozen tracks reach back through is
    described once and not once per later frame (the flow keeps nothing: it re-reads the
    pixels)."""

    gray: npt.NDArray[np.uint8]
    stamp: float
    place: Rigid
    features: Features | None = None  # filled in by the describer the first time it is asked


class ParallaxAnchor(AnchorStage):
    """The cart's own motion as a source of truth: a corner seen in this frame and in the
    previous one, with the odometry's transform between the two stamps, is two rays with a
    known baseline, and where they meet is a depth in metres (:mod:`pepin.parallax`). No lidar
    is involved and no floor plane is assumed — this is the hoop a camera keeps when every
    other sensor is off, and the only one that measures at every elevation the picture has, so
    it is the pool a law over the ray's angle can be fitted on.

    A corner is a TRACK, not a pair (``track_min_obs``, the node's ``parallax_track_min_obs``,
    3 by default; 2 restores the pair the stage measured until 2026-09-15). The corners of this
    frame are followed back through every ring frame inside ``track_window_s``
    (:func:`pepin.parallax.build_tracks` — hop by hop for the flow, recognised frame by frame
    for the describer) and all of a track's rays are met in ONE least-squares solve with the
    known camera placements, with a robust pass that drops a track's single worst observation
    (:func:`pepin.parallax.triangulate_tracks`). The point of it is the sigma: the two-view
    formula ``z^2 * sigma_px / (f * b)`` keeps its shape with ``b`` the quadrature sum of the
    views' perpendicular baselines, so four views 5 cm apart are worth one pair at 10 cm, and
    ``min_total_baseline_m`` is asked of that sum rather than of one partner's step. Two
    observations reproduce the pair's own numbers arithmetically, which is what makes 2 the off
    position of the knob rather than a different code path.

    The anchor keeps the last second and a half of frames in a ring (grey image, stamp, the
    lens's whole placement on the cart, and the describer's reading of it once asked for) and
    asks :class:`FrameContext`'s motion source for the transform between two stamps. While
    pairing, the partner is not the frame before this one: walking back from the
    newest, it is the first frame inside the gap window whose baseline reaches
    ``min_baseline_m``, and the widest baseline in the window when none does
    (:meth:`_partner`). Pairing with the frame before meant pairing 0.1 s apart, which at a
    cart's 0.2-0.3 m/s is 2 cm of baseline and a 23 cm sigma with most corners rejected for
    too little parallax (live run 2026-09-14 14:12); 10 cm asked for is 0.3-0.5 s back. The
    placement is TF's ``base_link <- camera_optical`` when the node supplies it, because that
    is the only pose carrying the neck's pan and a baseline turned by a pan-free pose points
    the wrong way; without it the mount's pitch stands in. A partner closer in time than
    ``min_gap_s`` has no baseline worth triangulating and one farther back than ``max_gap_s``
    has changed more than the flow follows; a frame while the cart stands still, or turns on
    the spot, yields nothing at all and says which. Each pair carries its own weight: the
    ratio of its inverse-depth variance to a lidar beam's, capped at 1, so a short baseline or
    a badly tracked corner counts for little without being thrown away.

    Measured offline on both legs of the errand of 2026-09-14 at 0.2-0.3 m/s
    (scratch/parallax_baseline_sweep.py, 4700 points matched to the lidar's own ranges, the
    calibrated fx 724.1): the parallax a pair rests on decides its noise, sigma 16.3 cm at
    2 cm of it, 12.2 at 5, 6.9 at 9, 4.1 at 18, and it decides the far field, 0.75 and 0.49 of
    the lidar at 1.5-2 and 2-3 m on a 2 cm baseline against 1.02-1.13 from 5 cm on. What it
    does not decide is a +9 to +13 % offset at 1.0-1.5 m, the same at every baseline — a
    scale-like error still unexplained, and not the range-dependent one reported on 2026-09-12
    (that shape was the thin baseline's skew, not the near field). Under a metre nothing can be
    checked this way: the lidar's plane leaves the bottom of the picture below 1.0 m. The stage
    costs 3-5 ms a frame and ships switched off.

    Who matches the corners is a switch (``matcher``, the node's ``parallax_matcher``) and it
    moves the gap window with it: the flow may look 0.60 s back, the describer 1.5 s, because a
    keypoint is recognised rather than followed. Measured on all four errands of 2026-09-14,
    both matchers on the same frame pairs (scratch/parallax_matcher_sweep.txt): ORB keeps 11
    pairs a frame at 1.0 and 1.5 s where the flow keeps 2 and 0, but at 0.5 s — the gap a 10 cm
    ask lands on at this speed — the flow reads 0.993 of the lidar at 1.5-2 m against ORB's
    1.031, with half the per-pair sigma (13.5 cm against 29.6) at half the cost (3.9 ms a frame
    against 8.3). Past a second of gap both read 1.25-1.45: the odometry's own drift over that
    second, not the matcher's doing. So the flow is the default and the longer window is there
    for a pose that deserves it.

    Which pose that is, is the second switch (``motion_source``, the node's ``parallax_motion``).
    A baseline is a length, and over a second the wheels and gyro do not know one: the same four
    errands re-measured with the motion taken from the lidar tracker's map pose instead
    (scratch/parallax_pose_sweep.txt, 2026-09-14) show the odometry reading 15.8 cm of travel at
    a 1.0 s gap and 25.5 at 1.5 s where the tracker reads 13.9 and 18.6 — and every depth is
    proportional to that length. At 1-2 m the flow reads 1.263 and 1.342 of the lidar on the
    odometry's motion at those two gaps and 1.138 and 0.944 on the tracker's; the describer
    1.347 and 1.506 against 0.951 and 0.968. The over-reading past a second was the baseline all
    along, and ``tracker`` is the default. It is not free of its own: at the 0.5 s gap the anchor
    really pairs across, the tracker's motion reads 0.932 at 1-2 m where the odometry's reads
    1.067 — the two bracket the truth — and the per-pair sigma stays 7-10 cm either way, which
    is why these pairs are still measured against the lidar rather than trusted under it. A
    source with no map to ask (a tape's bare odometry, a cart whose tracker is silent, a stale
    map -> odom TF cannot interpolate at these stamps) falls back to the odometry per window,
    and the report line counts both."""

    name = "parallax_anchor"

    def __init__(
        self,
        *,
        weight: float = PARALLAX_WEIGHT,
        min_gap_s: float = PARALLAX_MIN_GAP_S,
        max_gap_s: float | None = None,
        min_baseline_m: float = PARALLAX_MIN_BASELINE_M,
        matcher: str = PARALLAX_MATCHER,
        motion_source: str = PARALLAX_MOTION,
        map_wait: bool = PARALLAX_MAP_WAIT,
        map_max_age_s: float = PARALLAX_MAP_MAX_AGE_S,
        track_min_obs: int = PARALLAX_TRACK_MIN_OBS,
        track_window_s: float = PARALLAX_TRACK_WINDOW_S,
        min_total_baseline_m: float = PARALLAX_MIN_TOTAL_BASELINE_M,
    ) -> None:
        self.weight = weight
        self.min_gap_s = min_gap_s
        self._max_gap_s = max_gap_s  # None: whatever window the matcher in use can carry
        self.min_baseline_m = min_baseline_m
        self.matcher = matcher
        self.motion_source = motion_source  # live: the node's parallax_motion flag writes it
        self.map_wait = map_wait  # live: parallax_map_wait — the old blocking ask, for an A/B
        self.map_max_age_s = map_max_age_s
        self.track_min_obs = track_min_obs  # live: parallax_track_min_obs; 2 is the old pair
        self.track_window_s = track_window_s  # live: parallax_track_window_s
        self.min_total_baseline_m = min_total_baseline_m  # live: parallax_min_total_baseline_m
        self.stale = 0  # frames whose map pose was too old (or absent) and fell back to odometry
        self.used: dict[str, int] = dict.fromkeys(PARALLAX_MOTIONS, 0)  # who gave each baseline
        self.frames = 0  # frames that reached the triangulation
        self.contributed = 0  # of those, the ones that gave at least one pair
        self.rejected: dict[str, int] = {}
        self._ring: deque[PreviousFrame] = deque(maxlen=PARALLAX_RING_FRAMES)
        self._baseline: list[float] = []
        self._sigma: list[float] = []
        self._sigma_two: list[float] = []  # the same corners read as the widest pair alone
        self._obs: list[float] = []  # observations a track rests on, median per frame
        self._gap: list[float] = []
        self._kept: list[int] = []

    @property
    def tracking(self) -> bool:
        """Whether a corner is a TRACK through the window (``track_min_obs`` of 3 or more) or
        the PAIR of this frame and one chosen partner (2, what the stage did until
        2026-09-15)."""
        return self.track_min_obs > 2

    @property
    def window_s(self) -> float:
        """How far back this frame reaches for a partner or for a track's oldest view: the
        track window while tracking, the matcher's own gap window while pairing."""
        return self.track_window_s if self.tracking else self.max_gap_s

    @property
    def max_gap_s(self) -> float:
        """How far back a partner frame may sit, in seconds: the window the constructor was
        given, or the one the matcher in use can carry — 0.60 s for the flow, which loses three
        corners in four by 1.5 s, and 1.5 s for the describer, which recognises a keypoint
        instead of following it."""
        if self._max_gap_s is not None:
            return self._max_gap_s
        return PARALLAX_ORB_MAX_GAP_S if self.matcher == "orb" else PARALLAX_MAX_GAP_S

    @max_gap_s.setter
    def max_gap_s(self, seconds: float | None) -> None:
        """Pin the window to a number of seconds, or to ``None`` to let the matcher set it."""
        self._max_gap_s = seconds

    @property
    def sigma_m(self) -> float | None:
        """The per-pair triangulation noise of the last frames, metres (their median), or
        ``None`` before the first pair — the number every pair's weight is 1 / sigma^2 of, in
        inverse depth (:func:`pepin.depth.pair_weight`), and the one to read against the
        lidar's ``lidar_sigma_m`` when the two rulers disagree."""
        return float(np.median(self._sigma)) if self._sigma else None

    @property
    def gap_s(self) -> float | None:
        """How far back in time the last triangulated frame reached, in seconds — its partner
        while pairing, its oldest track view while tracking — or ``None`` before the first one.
        The number the report line's span is the median of."""
        return self._gap[-1] if self._gap else None

    def _count(self, reason: str, n: int = 1) -> None:
        """Tally one cause of a lost pair or a lost frame, for the report line."""
        if n:
            self.rejected[reason] = self.rejected.get(reason, 0) + n

    def _moved(
        self, ctx: FrameContext, then: float, now: float, ask_tracker: bool
    ) -> tuple[Rigid | None, str]:
        """How the cart moved between two frame stamps and whose word it is: the tracker's map
        pose while ``ask_tracker`` and the source can answer through the map
        (:class:`MapMotionSource`), the odometry otherwise — the fallback a node needs when the
        tracker is silent or its map pose too stale for TF to interpolate at these stamps.

        ``ask_tracker`` is the caller's memory of that fallback within one walk of the ring: a
        TF lookup that cannot be answered costs its whole timeout, and a dead tracker must cost
        it once a frame rather than once a candidate."""
        source = ctx.motion
        if source is None:
            return None, ""
        if ask_tracker:
            through_map = self._through_map(source, then, now)
            if through_map is not None:
                return through_map, "tracker"
            self.stale += 1
        return source.motion(then, now), "odom"

    def _through_map(self, source: MotionSource, then: float, now: float) -> Rigid | None:
        """The tracker's word on the motion, asked the way ``map_wait`` says: the non-blocking
        ask of the newest map pose within ``map_max_age_s`` of the frame (the default), or the
        old ask that waits for TF to cover the frame's stamp. ``None`` when the source cannot
        answer at all — a tape's bare odometry, a silent tracker, a map pose too old."""
        if not self.map_wait and isinstance(source, RecentMapMotionSource):
            return source.map_motion_recent(then, now, self.map_max_age_s)
        if self.map_wait and isinstance(source, MapMotionSource):
            return source.map_motion(then, now)
        return None

    def _partner(self, ctx: FrameContext, place: Rigid) -> tuple[PreviousFrame, Motion] | None:
        """The frame of the ring this one is paired with and the camera motion between them:
        walking back from the newest, the first partner inside the gap window whose baseline
        reaches ``min_baseline_m``, and the widest baseline in the window when none does.

        Newest-first is the point of it: the shortest gap that carries the asked-for parallax
        is the one whose view has changed least, so the flow still tracks. A cart at 0.25 m/s
        reaches 10 cm about 0.4 s back and pairs across four frames; a cart standing still
        never reaches it and pairs with the widest it has, as the stage always did."""
        from pepin.parallax import camera_motion

        best: tuple[PreviousFrame, Motion] | None = None
        spoke = ""
        candidates = 0
        ask_tracker = self.motion_source == "tracker"
        for previous in reversed(self._ring):
            gap = ctx.stamp - previous.stamp
            if gap < self.min_gap_s:
                continue
            if gap > self.max_gap_s:
                break
            candidates += 1
            moved, source = self._moved(ctx, previous.stamp, ctx.stamp, ask_tracker)
            ask_tracker &= source != "odom"  # a silent tracker is discovered once, not per frame
            if moved is None:
                continue
            motion = camera_motion(moved.rotation, moved.translation, previous.place, place)
            if best is None or motion.baseline > best[1].baseline:
                best, spoke = (previous, motion), source
            if motion.baseline >= self.min_baseline_m:
                break
        if best is None:
            if self._ring:
                self._count("gap" if candidates == 0 else "no odometry")
            return None
        self.used[spoke] = self.used.get(spoke, 0) + 1
        return best

    def _window(self, ctx: FrameContext, place: Rigid) -> list[tuple[PreviousFrame, Motion]]:
        """Every frame of the ring a track may reach back through, OLDEST first, each with the
        camera motion from its optical frame into this frame's — what
        :func:`pepin.parallax.build_tracks` wants. The walk is newest-first (so a silent
        tracker is discovered once a frame, as :meth:`_partner`'s is) and stops at
        ``track_window_s``; a window a motion source cannot answer simply loses that view, and
        the frame is triangulated on the ones it can."""
        from pepin.parallax import camera_motion

        out: list[tuple[PreviousFrame, Motion]] = []
        spoke = ""
        candidates = 0
        ask_tracker = self.motion_source == "tracker"
        for previous in reversed(self._ring):
            gap = ctx.stamp - previous.stamp
            if gap < self.min_gap_s:
                continue
            if gap > self.track_window_s:
                break
            candidates += 1
            moved, source = self._moved(ctx, previous.stamp, ctx.stamp, ask_tracker)
            ask_tracker &= source != "odom"  # a silent tracker is discovered once, not per frame
            if moved is None:
                continue
            out.append(
                (previous, camera_motion(moved.rotation, moved.translation, previous.place, place))
            )
            spoke = spoke or source
        if not out:
            if self._ring:
                self._count("gap" if candidates == 0 else "no odometry")
            return out
        self.used[spoke] = self.used.get(spoke, 0) + 1
        out.reverse()
        return out

    def _remember(self, ctx: FrameContext, gray: npt.NDArray[np.uint8], place: Rigid) -> None:
        """Put this frame in the ring and drop the frames no later frame can reach: the track
        window while tracking, the matcher's gap window while pairing."""
        while self._ring and ctx.stamp - self._ring[0].stamp > self.window_s:
            self._ring.popleft()
        self._ring.append(PreviousFrame(gray, ctx.stamp, place))

    def _track_truth(
        self,
        ctx: FrameContext,
        gray: npt.NDArray[np.uint8],
        views: list[tuple[PreviousFrame, Motion]],
    ) -> tuple[ParallaxTruth, float]:
        """This frame's corners as tracks through ``views``, and the span in seconds the oldest
        of them reaches back over. The describer's reading of each ring frame is kept on the
        frame, so a window of a dozen views costs one description each and not one per frame."""
        from pepin.parallax import track_truth

        grays = [previous.gray for previous, _ in views] + [gray]
        motions = [motion for _, motion in views]
        features: list[Features | None] = [previous.features for previous, _ in views] + [None]
        truth = track_truth(
            grays,
            motions,
            ctx.intr,
            matcher=self.matcher,
            features=features if self.matcher == "orb" else None,
            min_obs=self.track_min_obs,
            min_total_baseline_m=self.min_total_baseline_m,
        )
        if self.matcher == "orb":
            for (previous, _), found in zip(views, features, strict=False):
                previous.features = found
        return truth, ctx.stamp - views[0][0].stamp

    def pairs(self, frame: Frame) -> Pairs | None:
        """(network, triangulated) pairs at the corners of this frame: each corner followed
        back through the window of ring frames and triangulated from every view it was seen in
        (``track_min_obs`` 3 or more), or shared with the one partner frame chosen out of the
        ring (2, what the stage did until 2026-09-15). ``None`` when there is no usable motion
        behind this frame, or when nothing survived the gates."""
        from pepin.parallax import CameraPlacement, parallax_truth

        ctx = frame.ctx
        gray = ctx.gray
        if gray is None:
            self._count("no image")
            return None
        # TF's edge when the node read one: it is the only pose that carries the neck's pan,
        # and a baseline built from a pan-free pose points the wrong way (pepin.parallax).
        place: Rigid = (
            ctx.cam_optical if ctx.cam_optical is not None else CameraPlacement.of(ctx.cam)
        )
        if self.tracking:
            views = self._window(ctx, place)
            self._remember(ctx, gray, place)
            if not views:
                return None
            truth, span = self._track_truth(ctx, gray, views)
        else:
            chosen = self._partner(ctx, place)
            self._remember(ctx, gray, place)
            if chosen is None:
                return None
            previous, motion = chosen
            truth = parallax_truth(previous.gray, gray, ctx.intr, motion, matcher=self.matcher)
            span = ctx.stamp - previous.stamp
        self.frames += 1
        for reason, n in truth.rejected.items():
            self._count(reason, n)
        if truth.kept == 0:
            self._count(truth.verdict or "none")
            return None
        cols = np.rint(truth.points[:, 0]).astype(int)
        rows = np.rint(truth.points[:, 1]).astype(int)
        d = frame.raw[rows, cols]
        ok = np.isfinite(d) & (d > NEAR_M) & ~frame.edge[rows, cols]
        self._count("edge", int((~ok).sum()))
        if not bool(ok.any()):
            return None
        self.contributed += 1
        self._baseline.append(float(np.median(truth.baseline[ok])))
        self._sigma.append(float(np.median(truth.sigma[ok])))
        self._gap.append(span)
        self._kept.append(int(ok.sum()))
        if truth.observations is not None:
            self._obs.append(float(np.median(truth.observations[ok])))
        if truth.sigma_two is not None:
            self._sigma_two.append(float(np.median(truth.sigma_two[ok])))
        del self._baseline[:-POOL_FRAMES], self._sigma[:-POOL_FRAMES], self._gap[:-POOL_FRAMES]
        del self._kept[:-POOL_FRAMES], self._obs[:-POOL_FRAMES], self._sigma_two[:-POOL_FRAMES]
        return Pairs.of(
            d[ok],
            truth.z[ok],
            lift_of(rows[ok], ctx.intr),
            self.weight * truth.weight[ok],
            left_of(cols[ok], ctx.intr),
        )

    def describe(self) -> str:
        """The verdict for the report line: who matched the corners and how far back it may
        look, whether a corner is a track or a pair and how many views one must be seen in,
        whose motion the baseline came from and how many windows each source actually answered,
        the parallax asked for, the frames that triangulated, the span and the effective
        baseline they rest on, the corners a frame yields and their sigma — and, while
        tracking, the observations a track carries and what the same corners' sigma would have
        been read as the widest single pair, which is the whole point of the change."""
        spoke = ", ".join(f"{k} {v}" for k, v in self.used.items() if v)
        shape = (
            f" >= {self.track_min_obs} obs, asks {self.min_total_baseline_m * 100:.0f} cm total"
            if self.tracking
            else f", asks {self.min_baseline_m * 100:.0f} cm"
        )
        asked = (
            f"{self.matcher} <= {self.window_s:.2f} s{shape}"
            f" on the {self.motion_source}'s motion ({spoke or 'none yet'}),"
            f" weight {self.weight:g} / sigma^2"
            + (f", map pose stale -> odom {self.stale}" if self.stale else "")
            + (" [map_wait: the frame path WAITS for TF]" if self.map_wait else "")
        )
        dropped = ", ".join(f"{k} {v}" for k, v in self.rejected.items())
        if not self._baseline:
            return f"{asked}, no pairs yet" + (f" ({dropped})" if dropped else "")
        what = "tracks" if self.tracking else "pairs"
        gained = (
            f" ({np.median(self._obs):.1f} obs a track,"
            f" 2-view sigma {np.median(self._sigma_two) * 100:.1f} cm)"
            if self._obs and self._sigma_two
            else ""
        )
        return (
            f"{asked}, {self.contributed}/{self.frames} frames,"
            f" span {np.median(self._gap):.2f} s,"
            f" baseline {np.median(self._baseline) * 100:.1f} cm,"
            f" {np.median(self._kept):.0f} {what} a frame,"
            f" sigma {np.median(self._sigma) * 100:.1f} cm{gained}"
            + (f", rejected: {dropped}" if dropped else "")
        )


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
        a_lo, a_hi = a_bounds()
        if a_lo <= a <= a_hi and B_BOUNDS[0] <= b <= B_BOUNDS[1]:
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


# ---- the law of the ray -----------------------------------------------------------------------
class RayLaw(AffineLaw):
    """The affine law with its scale a smooth function of the ray's angle off the optical axis
    (:mod:`pepin.elevation`): ``1 / z = a(eps, az) / D + b(eps, az)``, fitted on the same pool,
    with the plain affine law as its fallback whenever the angular fit does not stand.

    The error this corrects belongs to the camera and the network, not to the room: it is
    parameterised by the ray's own elevation (and azimuth) in radians, so the neck may tilt and
    pan without moving the law — the head's pose enters where it always did, through TF into
    :class:`FrameContext`'s :class:`pepin.depth.CameraPose`, which decides which world point a
    ray meets, not how far the network thinks it is. As a stage the law reads the frame's raw
    depth and keeps the holes of the depth handed to it, so it can stand either as the chain's
    only law or behind the affine law as the flag ``ray_law`` switching between the two live
    (rule 19): with the flag off, the affine law's image goes on unchanged; with it on, the
    same pixels come from the ray's law. Off it also falls back, without a word, whenever
    :func:`pepin.elevation.fit_ray` returns nothing — too few pairs, too narrow a cone, a slope
    that turns non-positive — and the frame is withheld only where the affine law would withhold
    it (no law at all).

    Cost: the stage refits on every frame as the affine law does, and the angular fit is the
    wider design — 8.5 ms a frame (10.5 max) against the affine law's 2.9 on a 47 000-pair pool
    of run 0171's frames, measured on the Mac. The stage is off by default; a pool that large
    only exists with the wall anchor on."""

    name = "ray_law"

    def __init__(
        self,
        pool_frames: int = POOL_FRAMES,
        *,
        degree: int = RAY_DEGREE,
        azimuth_degree: int = RAY_AZIMUTH_DEGREE,
    ) -> None:
        super().__init__(pool_frames)
        self.degree = degree
        self.azimuth_degree = azimuth_degree
        self.gain: RayGain | None = None
        self._seeded_gain: RayGain | None = None
        self._live_gain = False

    def seed_gain(self, gain: RayGain) -> None:
        """Start from a saved gain (the map's), applied until the live pool can fit its own."""
        self._seeded_gain = gain
        self.gain = gain
        self._live_gain = False

    @property
    def ray_ready(self) -> bool:
        """Whether an angular law exists; false means this stage is the affine law."""
        return self.gain is not None

    @property
    def ray_fitted(self) -> bool:
        """Whether the angular law rests on the live pool rather than on a seed."""
        return self._live_gain

    def saved_state(self) -> dict[str, Any] | None:
        """The angular law that stands as plain JSON values for :func:`pepin.depth.save_law`,
        ``None`` while there is none. A seed the live pool has not replaced is written back
        rather than dropped: the file holds one record for both laws, so anything left out of a
        save is erased, and this law belongs to the camera and the network, not to the room the
        last window happened to show (:mod:`pepin.elevation`) — the whole file is only ever
        written while the affine law rests on live pairs."""
        return self.gain.state() if self.gain is not None else None

    @property
    def clipped(self) -> bool:
        """Whether the angular law meets a bound inside its own span (a law at its limit)."""
        return self.gain is not None and self.gain.clipped

    def fit(self, pairs: Pairs | None) -> None:
        """The affine fit, then the angular one on the whole pool; a pool that cannot carry an
        angular law leaves the seeded gain, or none, and the affine law stands."""
        super().fit(pairs)
        pool = self.pool
        if pool is None or not self.fitted:
            return
        elevation, azimuth = ray_angles(pool.lift, pool.left)
        gain = fit_ray(
            pool.d,
            pool.z,
            elevation,
            pool.weight,
            azimuth=azimuth,
            degree=self.degree,
            azimuth_degree=self.azimuth_degree,
        )
        if gain is not None:
            self.gain, self._live_gain = gain, True
        elif self._seeded_gain is None:
            self.gain, self._live_gain = None, False

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The depth through the angular law — one (a, b) per pixel, from that pixel's ray —
        or through the plain affine law while no angular law stands."""
        d = np.asarray(depth, dtype=float)
        if self.gain is None:
            return apply_affine(d, self.a, self.b)
        rows = np.arange(d.shape[0])[:, None]
        columns = np.arange(d.shape[1])[None, :]
        elevation, azimuth = ray_angles(lift_of(rows, ctx.intr), left_of(columns, ctx.intr))
        return self.gain.apply(d, elevation, azimuth)

    def run(self, depth: Array, frame: Frame) -> tuple[Array, Verdict]:
        """Fit on the pool and correct the frame's raw depth, keeping the holes of the depth
        handed in (the edge filter's, and the affine law's where it ran before this stage);
        the frame is withheld only while no law of any kind exists."""
        pool = frame.pool
        self.fit(pool)
        n = 0 if pool is None else pool.size
        if not self.ready:
            return depth, Verdict(self.name, True, pairs=n, note="no law yet", withhold=True)
        out = np.where(np.isfinite(depth), self.apply(frame.raw, frame.ctx), np.nan)
        return out, Verdict(self.name, True, pairs=n, pixels=int(out.size), note=self.describe())

    def describe(self) -> str:
        """The affine law's words, then the angular law's — marked ``(seed)`` while it is the
        one the last run saved — or why there is none."""
        if self.gain is None:
            return super().describe() + "; affine fallback"
        source = "" if self._live_gain else " (seed)"
        return super().describe() + "; " + self.gain.describe() + source


class RangeLawStage(LawStage):
    """The law whose scale follows the range: the pooled pairs binned by the network's own
    depth and a robust ratio measured in each bin (:class:`pepin.depth.RangeLaw`), applied to
    the same raw depth the affine law reads. On, its image replaces the affine law's.

    The affine law it is built on stays its fallback and its readiness: while the pool has not
    filled :data:`pepin.depth.RANGE_MIN_BINS` bins — the warm-up, or a cart facing one wall at
    one range — the frame goes out through the affine law rather than being withheld, which is
    also what a seed of the saved affine numbers means here. The pool is this stage's own queue
    of the last ``pool_frames`` frames' pairs (the same objects the affine law pools: the
    references cost nothing), so switching the affine law off for an A/B does not blind it."""

    name = "range_law"

    def __init__(self, affine: AffineLaw, pool_frames: int = POOL_FRAMES) -> None:
        self.affine = affine
        self.law: RangeLaw | None = None
        self._live = False  # whether the law rests on the live pool rather than on a seed
        self._pool: list[Pairs] = []
        self._pool_frames = pool_frames

    def seed(self, law: RangeLaw) -> None:
        """Start from a saved range law (the map's): applied until the live pool can fit one."""
        self.law, self._live = law, False

    @property
    def ready(self) -> bool:
        """Whether a law worth applying exists — this one's, or the affine law behind it."""
        return self.law is not None or self.affine.ready

    @property
    def fitted(self) -> bool:
        """Whether the range law rests on the live pool (worth saving)."""
        return self._live

    @property
    def pooled(self) -> int:
        """How many live pairs the pool holds."""
        return int(sum(p.size for p in self._pool))

    @property
    def pool(self) -> Pairs | None:
        """Everything in the pool, as one."""
        return Pairs.join(self._pool)

    def saved_state(self) -> dict[str, Any] | None:
        """The range law that stands as plain JSON values for :func:`pepin.depth.save_law`,
        ``None`` while there is none. A seed the live pool has not replaced is written back
        rather than dropped: the file holds one record for every law, so anything left out of
        a save is erased."""
        return self.law.state() if self.law is not None else None

    def fit(self, pairs: Pairs | None) -> None:
        """Feed a frame's pairs (or ``None``); the law is refitted on the pool. A pool that
        cannot fill two bins leaves the seeded law, or none, and the affine law stands."""
        if pairs is None or pairs.size == 0:
            return
        self._pool.append(pairs)
        del self._pool[: -self._pool_frames]
        if self.pooled < POOL_MIN_SAMPLES:
            return
        pool = self.pool
        assert pool is not None
        law = RangeLaw.fit(pool.d, pool.z, pool.weight)
        if law is not None:
            self.law, self._live = law, True

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The depth through the range law — each pixel scaled by the ratio measured at its own
        range — or through the plain affine law while no range law stands."""
        if self.law is None:
            return apply_affine(depth, self.affine.a, self.affine.b)
        return self.law.apply(depth)

    def run(self, depth: Array, frame: Frame) -> tuple[Array, Verdict]:
        """Fit on the pool and correct the frame's raw depth, keeping the holes of the depth
        handed in (the edge filter's, and the affine law's where it ran before this stage);
        the frame is withheld only while no law of any kind exists."""
        pool = frame.pool
        self.fit(pool)
        n = 0 if pool is None else pool.size
        if not self.ready:
            return depth, Verdict(self.name, True, pairs=n, note="no law yet", withhold=True)
        out = np.where(np.isfinite(depth), self.apply(frame.raw, frame.ctx), np.nan)
        return out, Verdict(self.name, True, pairs=n, pixels=int(out.size), note=self.describe())

    def describe(self) -> str:
        """The law for the report line: its bins and their ratios — marked ``(seed)`` while it
        is the one the last run saved — or that the affine law is standing in for it."""
        if self.law is None:
            return f"affine fallback on {self.pooled} pairs"
        source = "" if self._live else " (seed)"
        return f"{self.law.describe()} on {self.pooled} pairs{source}"


class FrameLaw(LawStage):
    """The law of THIS frame: scale and shift fitted on the beams of the frame in hand
    (:func:`pepin.depth.fit_frame`), not on the pool — the alignment the field performs.

    Depth Anything V2's metric heads are evaluated after a per-image scale-and-shift alignment
    against sparse ground truth, and a robot carrying a depth sensor aligns its monocular depth
    against that sensor's points frame by frame. The pool's laws (:class:`AffineLaw`,
    :class:`RangeLawStage`) describe the camera over the last ``POOL_FRAMES`` frames, which is
    the right thing when the network's error is a property of the lens and the wrong thing when
    it is a property of the scene: a new room, a new light, a neck that has tilted. This stage
    asks each frame's own beams what the network is doing right now.

    It corrects what ``prior`` published, not the raw network: the law behind it in the chain
    (the range law, or the affine one) keeps the shape it measured over the pool, and this
    stage removes only what that law left on this frame. Measured on 2026-09-14
    (scratch/frame_law_eval.py, every frame's pairs split odd / even, odd fitting, even
    judging): over the raw network the per-frame law reads a median |residual| of 10.2 % on run
    0171's drive against the range law's 23.2 %, and over the range law 7.5 %; the far band
    (2.0-2.5 m), where the network saturates and no scale can help, is the one place the pool's
    shape still earns its keep. ``prior`` must therefore be a law that maps depth to depth pixel
    by pixel (both pool laws are); an angular law is not one.

    A frame carrying at least ``min_pairs`` pairs gets its own law. A frame carrying fewer
    holds the last one, decaying back to the prior with a time constant of ``tau_s``: the blend
    is of the two published inverse depths, weight ``exp(-held / tau_s)`` on the frame's own, so
    a cart that turns away from every surface the lidar and the camera share returns to the
    pool's law in a few seconds instead of carrying one frame's numbers forever. With no law of
    its own yet the prior's image goes out untouched, so switching this stage on never withholds
    a frame the chain would otherwise have published."""

    name = "frame_law"

    def __init__(
        self,
        prior: Law,
        min_pairs: int = FRAME_MIN_PAIRS,
        tau_s: float = FRAME_HOLD_TAU_S,
        clock: Callable[[], float] = time.monotonic,
        shift_needs_beams: bool = True,
    ) -> None:
        self.prior = prior
        self.a = 1.0
        self.b = 0.0
        self.min_pairs = min_pairs
        self.tau_s = tau_s
        self.shift_needs_beams = shift_needs_beams
        self.frames = 0
        self.fits = 0
        self.held = 0
        self.pairs = 0  # pairs behind the law in hand
        self._fitted = False
        self._clock = clock
        self._last_fit: float | None = None  # when a frame last spoke for itself
        self._rulers: dict[str, float] = {}  # weight per anchor behind the law in hand

    @property
    def ready(self) -> bool:
        """Whether a law worth applying exists — this frame's, or the prior behind it."""
        return self._fitted or self.prior.ready

    @property
    def weight(self) -> float:
        """How much of the frame's own law stands right now: 1.0 the moment it was fitted,
        decaying as ``exp(-seconds held / tau_s)`` toward the prior's law, 0.0 with no law of
        its own. A ``tau_s`` at or below zero drops to the prior as soon as a frame is held."""
        if not self._fitted or self._last_fit is None:
            return 0.0
        if self.tau_s <= 0.0:
            return 1.0 if self.held == 0 else 0.0
        return float(math.exp(-max(0.0, self._clock() - self._last_fit) / self.tau_s))

    def fit(self, pairs: Pairs | None, ctx: FrameContext | None = None, beams: bool = True) -> None:
        """Feed this frame's pairs and its context (the prior is asked what it makes of those
        pairs' depths, and the fit is that law's residual): they fit this frame's law outright,
        or the last law is held and starts decaying toward the prior's. Without a context there
        is no prior to correct and the frame is held.

        ``beams`` says whether the lidar is one of the rulers in this pool. On a pool with no
        beams at all — parallax corners alone, the lidar-off case — the shift is shut off and
        the frame gets a scale only while ``shift_needs_beams`` stands: the corners span the
        whole picture's depths, so the spread gate opens, and a two-parameter fit on a ruler of
        7-10 cm per pair runs to the law's bounds. Measured over the four errands of 2026-09-14
        (scratch/parallax_ruler_eval.txt, 101 frames, the corners fitting and the beams judging):
        with the shift the parallax-only law reads 44.5 % median |residual| and its scale jumps
        1.69 between consecutive frames, with the scale alone 30.9 % and 1.15."""
        self.frames += 1
        law = None
        if pairs is not None and pairs.size and ctx is not None:
            spread = math.inf if self.shift_needs_beams and not beams else FRAME_MIN_SPREAD
            law = fit_frame(
                self.prior.apply(pairs.d, ctx),
                pairs.z,
                pairs.weight,
                self.min_pairs,
                min_spread=spread,
            )
        if law is None or pairs is None:
            self.held += 1
            return
        self.a, self.b = law
        self.pairs, self._fitted = pairs.size, True
        self.fits += 1
        self._last_fit = self._clock()

    def apply(self, depth: Array, ctx: FrameContext) -> Array:
        """The prior's depth through this frame's own law, the prior's alone where this one has
        decayed (the blend is of the published inverse depths) or while no frame has spoken."""
        prior = self.prior.apply(depth, ctx)
        w = self.weight
        if w <= 0.0:
            return prior
        own = apply_affine(prior, self.a, self.b)
        if w >= 1.0:
            return own
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = w / own + (1.0 - w) / prior
            out: Array = np.where(np.isfinite(inv) & (inv > 1e-6), 1.0 / inv, np.nan)
        return out

    def run(self, depth: Array, frame: Frame) -> tuple[Array, Verdict]:
        """Fit on this frame's pairs and correct the frame's raw depth, keeping the holes of
        the depth handed in (the edge filter's, and the laws' that ran before); the frame is
        withheld only while no law of any kind exists."""
        pool = frame.pool
        rulers = frame.rulers
        self.fit(pool, frame.ctx, beams=rulers.get("lidar_anchor", 0.0) > 0.0)
        if rulers and self.pairs:
            self._rulers = rulers
        n = 0 if pool is None else pool.size
        if not self.ready:
            return depth, Verdict(self.name, True, pairs=n, note="no law yet", withhold=True)
        out = np.where(np.isfinite(depth), self.apply(frame.raw, frame.ctx), np.nan)
        return out, Verdict(self.name, True, pairs=n, pixels=int(out.size), note=self.describe())

    @property
    def rulers(self) -> str:
        """Which ruler fitted the law in hand, by share of the fit's total weight — e.g.
        ``rulers: lidar 98%, parallax 2%, 312 pts``. Pairs are not the measure: a frame can
        carry 200 parallax corners at 0.03 of a beam's weight each and 30 beams, and the beams
        still write the law. Empty before the first fit."""
        total = sum(self._rulers.values())
        if not self._rulers or total <= 0.0:
            return ""
        shares = ", ".join(
            f"{name.removesuffix('_anchor').removesuffix('_pairs')} {w / total:.0%}"
            for name, w in sorted(self._rulers.items(), key=lambda kv: -kv[1])
        )
        return f"rulers: {shares}, {self.pairs} pts"

    def describe(self) -> str:
        """The law for the report line: this frame's two numbers over the prior's depth, the
        pairs behind them, which rulers' weight fitted them, how many frames have been held
        against how many seen, and how much of the frame's own law still stands against the
        prior's."""
        if not self._fitted:
            return f"prior stands, {self.held}/{self.frames} frames held"
        clipped = at_bound(self.a, self.b)
        edge = f" [{clipped} AT BOUND]" if clipped else ""
        rulers = self.rulers
        return (
            f"a {self.a:.2f} b {self.b:+.3f} on {self.pairs} pairs{edge}, "
            + (f"{rulers}, " if rulers else "")
            + f"{self.held}/{self.frames} frames held (own {self.weight:.2f})"
        )


# ---- the chain ----------------------------------------------------------------------------------
def standard_pipeline(
    law: AffineLaw | None = None,
    *,
    ray: RayLaw | None = None,
    range_stage: RangeLawStage | None = None,
    frame_stage: FrameLaw | None = None,
    floor_pairs: bool = False,
    wall_anchor: bool = False,
    parallax_anchor: bool = False,
    ray_law: bool = False,
    range_law: bool = True,
    frame_law: bool = True,
    wall_correct: bool = False,
) -> DepthPipeline:
    """The node's chain: edges -> lidar -> (floor pairs) -> (wall pairs) -> (parallax) -> law
    -> (ray law) -> range law -> frame law -> (wall correction) -> floor anchor; the seven
    switchable stages are in the list and switched by the flags of the same name
    (``wall_anchor`` is the pairs role, ``wall_correct`` the pixels). The ray law, the range law
    and the frame law all sit behind the affine one and correct the same raw depth by their own
    rule instead — by the ray's angle, by the range, by this frame's own beams — and on, each
    replaces the image of the law before it; off, that law's stands. ``range_law`` and
    ``range_law`` and ``frame_law`` are the two of the seven on by
    default: one affine law leaves a residual that tilts 12 % per metre
    (:class:`pepin.depth.RangeLaw`), a law fitted on a minute of pool describes the last
    minute's scene rather than this frame's (:class:`FrameLaw`), and the lidar's one row is not
    the whole picture — the parallax anchor is the second ruler of the scale, weighed against
    the beams by its own noise and measuring where they cannot reach (:class:`ParallaxAnchor`).

    Every law may be handed in so the caller keeps them: the affine and the ray law pool and
    fit on their own, so a saved law must be seeded into **both** (:meth:`AffineLaw.seed`), or
    the ray law withholds every frame of the warm-up while the affine law publishes from the
    seed; the range law falls back to the affine law it is built on and needs no seed of its
    own to publish."""
    the_law = law if law is not None else AffineLaw()
    the_ray = ray if ray is not None else RayLaw()
    the_range = range_stage if range_stage is not None else RangeLawStage(the_law)
    the_frame = frame_stage if frame_stage is not None else FrameLaw(the_range)
    geometry = FloorGeometry()
    stages: list[Stage] = [
        EdgeFilter(),
        LidarAnchor(),
        FloorPairs(the_law, geometry),
        WallAnchor(),
        ParallaxAnchor(),
        the_law,
        the_ray,
        the_range,
        the_frame,
        WallCorrection(),
        FloorAnchor(geometry),
    ]
    flags = (
        ("floor_pairs", floor_pairs),
        ("wall_anchor", wall_anchor),
        ("parallax_anchor", parallax_anchor),
        ("ray_law", ray_law),
        ("range_law", range_law),
        ("frame_law", frame_law),
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
    "FrameLaw",
    "Law",
    "LawStage",
    "LidarAnchor",
    "MapMotionSource",
    "MotionSource",
    "Pairs",
    "ParallaxAnchor",
    "PreviousFrame",
    "RangeLawStage",
    "RayLaw",
    "RecentMapMotionSource",
    "Result",
    "Rigid",
    "RowLaw",
    "Stage",
    "StageStats",
    "Verdict",
    "WallAnchor",
    "WallCorrection",
    "WallWalk",
    "left_of",
    "lift_of",
    "standard_pipeline",
]
