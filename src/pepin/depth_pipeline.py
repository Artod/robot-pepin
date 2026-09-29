"""The depth correction as a pipeline: anchors say what is true, laws turn depth into metres.

:mod:`pepin.depth` holds the pieces — the lidar's beams projected into the image, the affine law
fitted on their (measured, true) pairs, the edge filter, the floor snapped to its plane — and here
they run as an ordered list of stages, each switchable by name, so a node's flags toggle stages
instead of branches, every stage's cost and effect is counted per frame, and a new source of truth
is one more stage in the list.

Two roles. An :class:`Anchor` is an external truth about some pixels: it contributes
:class:`Pairs` — (measured depth, true depth, weight) — to the pool the law fits on, and / or
corrects the pixels it knows directly. A :class:`Law` is the map from the measured depth to
metres, fitted on the pooled pairs and applied to the whole image. The chain is
:class:`EdgeFilter` -> :class:`LidarAnchor` -> :class:`AffineLaw` -> :class:`FloorAnchor`. Under
the metric stereo head the affine law watches rather than corrects (``watching``): its numbers
are the head's health.

The scale-recovering stages the monocular network needed (floor pairs, wall anchor, parallax
anchor, range law, frame law with its scale field, wall correction) are on the tag
``alt/mono-depth-2026-09-21``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol

import numpy as np
import numpy.typing as npt

from pepin.depth import (
    EDGE_REL_STEP,
    FLOOR_HEIGHT_TOLERANCE,
    POOL_FRAMES,
    POOL_MIN_SAMPLES,
    UP_LEVEL,
    Array,
    CameraPose,
    Intrinsics,
    Mask,
    apply_affine,
    at_bound,
    beam_hits,
    drop_edges,
    edge_mask,
    fit_affine,
    floor_anchor,
    floor_depth,
    inverse_sigma,
    pair_weight,
    project,
)

# Pairs of one anchor a frame the law keeps when the frame has no beams: about the lidar's own
# count, because its pool is POOL_FRAMES (600) frames deep and a fit costs the total.
POOL_CAP_PER_SOURCE = 100
LEAN_STEP = 0.003  # the floor's expected depth is recomputed when the up vector moves this much
LIDAR_SIGMA_M = 0.0  # metres: one beam's range noise, 0 meaning every beam weighs a flat 1 —
# the reference pair itself (REF_SIGMA_INV, "a beam at 2 m").
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


@dataclass(frozen=True, eq=False)
class FrameContext:
    """What a frame brings besides its measured depth: the optics, the camera's place on the
    cart, which way is up (base_link), the lidar's returns as base_link points carried to the
    frame's moment (``None`` without a scan), the frame's stamp in seconds, and TF's whole
    ``base_link <- camera_optical`` edge, the one place a panned neck is carried (``None``
    unless a node supplies it)."""

    intr: Intrinsics
    cam: CameraPose
    up: Array = field(default_factory=lambda: UP_LEVEL.copy())
    lidar: Array | None = None
    stamp: float = 0.0
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

    def pool_capped(self, cap: int = POOL_CAP_PER_SOURCE) -> Pairs | None:
        """The pool the law reads: the lidar's beams alone when the frame has any (a fit costs
        the pool's total over POOL_FRAMES frames); with no beams in the frame, every other
        anchor's block thinned to at most ``cap`` pairs."""
        beams = [p for n, p in zip(self.sources, self.pairs, strict=False) if n == "lidar_anchor"]
        if beams:
            return Pairs.join(beams)
        return self.pool_thinned(cap)

    def pool_thinned(self, cap: int = POOL_CAP_PER_SOURCE) -> Pairs | None:
        """Every pair contributed so far, each anchor's block thinned to at most ``cap`` of them
        — evenly spaced through the block, their weights scaled so the block carries the total
        weight it did — or :attr:`pool` whole when ``cap`` is 0. Evenly spaced so that a law is
        reproducible from its pairs, per block so that thinning cannot change which anchor
        writes it."""
        if cap <= 0:
            return self.pool
        parts: list[Pairs] = []
        for part in self.pairs:
            if part.size <= cap:
                parts.append(part)
                continue
            keep = np.linspace(0, part.size - 1, cap).round().astype(int)
            scale = float(np.sum(part.weight)) / max(float(np.sum(part.weight[keep])), 1e-12)
            parts.append(
                Pairs(
                    part.d[keep],
                    part.z[keep],
                    part.weight[keep] * scale,
                    part.lift[keep],
                    part.left[keep],
                )
            )
        return Pairs.join(parts)


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
        pool = frame.pool_capped()
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

    A beam weighs a flat 1 by default — the reference pair (:data:`pepin.depth.REF_SIGMA_INV`).
    ``sigma_m`` above 0 weighs each beam ``1 / sigma^2`` in inverse
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
        # Watching: the law is fitted and reported, and the depth goes out as it came in. It is
        # what the law is for under a METRIC source (a calibrated stereo head): nothing to
        # correct, and a fit that leaves a 1.00 says the head has been knocked.
        self.watching = False

    def run(self, depth: Array, frame: Frame) -> tuple[Array, Verdict]:
        """Fit and apply as every law does — or, while :attr:`watching`, fit only: the depth is
        returned untouched and no frame is ever withheld for want of a law."""
        if not self.watching:
            return super().run(depth, frame)
        pool = frame.pool_capped()
        self.fit(pool)
        n = 0 if pool is None else pool.size
        return depth, Verdict(self.name, True, pairs=n, note=self.describe())

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
        role = "watching, depth untouched: " if self.watching else ""
        return f"{role}a {self.a:.2f} b {self.b:+.3f} on {self.pooled} pairs{source}{edge}{asked}"


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


def standard_pipeline(law: AffineLaw | None = None) -> DepthPipeline:
    """The node's chain: edges -> lidar -> affine law -> floor anchor, every stage on. The
    affine law may be handed in so the caller keeps it: it pools and fits on its own and takes
    the saved law as its seed (:meth:`AffineLaw.seed`)."""
    geometry = FloorGeometry()
    stages: list[Stage] = [
        EdgeFilter(),
        LidarAnchor(),
        law if law is not None else AffineLaw(),
        FloorAnchor(geometry),
    ]
    return DepthPipeline(stages)


__all__ = [
    "AffineLaw",
    "Anchor",
    "AnchorStage",
    "DepthPipeline",
    "EdgeFilter",
    "FloorAnchor",
    "FloorGeometry",
    "Frame",
    "FrameContext",
    "Law",
    "LawStage",
    "LidarAnchor",
    "Pairs",
    "Result",
    "Rigid",
    "Stage",
    "StageStats",
    "Verdict",
    "left_of",
    "lift_of",
    "standard_pipeline",
]
