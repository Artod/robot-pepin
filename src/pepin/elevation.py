"""The network's metric error as a property of the ray, not of the room.

The affine law (:func:`pepin.depth.fit_affine`) gives the whole image one pair of numbers,
1 / z = a / D + b, and measured on run 0171 that is not what the network does: corrected to
the lidar's beams, the picture is right at the beams' own rows and 1.6-2.0x too far above them
(scratch/pipeline_vs_truth.txt). The error is a function of *where in the cone of view* a pixel
sits — a property of the lens and the network — so the parameter here is the ray's own angle
off the optical axis, elevation and azimuth in radians, and nothing else: not the image row
(the same row is another angle on another lens), not the height of the point above the floor
(that is the room's). The head may therefore tilt and pan freely: the neck moves the camera,
and its pose enters only where it always did, through TF and the projection, while this law
stays the camera's.

The shape is the affine law with an angle-dependent slope::

    1 / z = a(eps, az) / D + b(eps, az),  a = 1 / alpha(eps, az),  b = -beta / alpha(eps, az)

fitted the way :func:`pepin.depth.fit_affine` fits its two numbers — the noisy 1 / D regressed
on the exact 1 / z, never the other way round (regression dilution), the worst quarter of the
residuals dropped once — with ``alpha`` a low-order polynomial in the normalised angles
(:data:`RAY_SCALE`) instead of one number. That keeps the fit linear in its unknowns and keeps
one law over the whole image: a multiplicative gain in inverse depth, which is what a ratio
that changes with elevation and not with depth means (an additive term in the lift,
:class:`pepin.depth_pipeline.ElevationLaw`, is the wrong shape for it, and a law fitted per
band of rows, :class:`pepin.depth_pipeline.RowLaw`, spends a whole affine fit per band and
overfits where a band holds one wall).

Two guards make it safe on a robot. The pool sees only the angles its anchors reach — the
lidar's beams on run 0171 span -24 to +18 degrees of elevation, the near returns low and the
far ones by the horizon — so outside the fitted span the gain is **held at the span's edge**,
never extrapolated by a polynomial that is free to run away above the picture; and every
angle's (a, b) is clipped into the affine law's own bounds (:data:`pepin.depth.A_BOUNDS`,
:data:`pepin.depth.B_BOUNDS`), the fit reporting whether that clip binds anywhere inside the
span (:attr:`RayGain.clipped`) so a report line can say the law is at its bound. A fit that
does not clear its gates is no fit at all (``None``), and the caller keeps the affine law.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pepin.depth import (
    A_BOUNDS,
    B_BOUNDS,
    MIN_DEPTH_SPREAD,
    MIN_SAMPLES,
    POOL_MIN_SAMPLES,
    Array,
)

RAY_SCALE = 0.5  # rad: the angle the polynomial's variable counts in (u = eps / RAY_SCALE), so
# u runs to about +-0.85 across our 78 degree lens and the terms of a cubic stay comparable
RAY_DEGREE = 2  # the elevation polynomial's degree: the measured shape is a rise that flattens
RAY_AZIMUTH_DEGREE = 0  # the azimuth polynomial's degree: 0 until the data says it is systematic
MIN_RAY_SPAN = 0.15  # rad (8.6 deg) of elevation a pool must span before any angular term is
# fitted; each further degree needs another MIN_RAY_SPAN and its own bin of pairs
RAY_BIN_MIN = MIN_SAMPLES  # pairs each bin of the span must hold for the degree it carries
RAY_GRID = 65  # angles across the fitted span the bounds are checked on
RESIDUAL_KEEP = 75  # percentile of the residuals kept for the second pass, as fit_affine


def ray_angles(lift: Array, left: Array | None = None) -> tuple[Array, Array]:
    """The rays' angles off the optical axis in radians — (elevation, azimuth) — from their
    tangents (``lift`` = ``-(row - cy) / fy``, ``left`` = ``-(column - cx) / fx``); the azimuth
    is zero when no ``left`` is given."""
    elevation: Array = np.arctan(np.asarray(lift, dtype=float))
    azimuth: Array = (
        np.zeros_like(elevation) if left is None else np.arctan(np.asarray(left, dtype=float))
    )
    return elevation, azimuth


@dataclass(frozen=True)
class RayGain:
    """The inverse-depth slope as a polynomial in the ray's angles: ``alpha`` ascending in the
    normalised elevation, ``azimuth`` ascending from the first power of the normalised azimuth
    (empty when none was fitted), the shift ``beta``, the elevation span it was fitted over
    (radians, outside which the gain is held at its edge), whether a bound binds inside that
    span and how many pairs it rests on."""

    alpha: Array
    beta: float
    lo: float
    hi: float
    clipped: bool
    pairs: int
    azimuth: Array

    @property
    def degree(self) -> int:
        """The elevation polynomial's degree."""
        return int(self.alpha.size) - 1

    def slope(self, elevation: Array, azimuth: Array | None = None) -> Array:
        """``alpha`` at these ray angles (radians), the elevation held at the fitted span's
        edge outside it: the factor the exact inverse depth enters the network's with."""
        u = np.clip(np.asarray(elevation, dtype=float), self.lo, self.hi) / RAY_SCALE
        out: Array = np.asarray(np.polyval(self.alpha[::-1], u), dtype=float)
        if self.azimuth.size and azimuth is not None:
            v = np.asarray(azimuth, dtype=float) / RAY_SCALE
            out = out + np.asarray(np.polyval(np.append(self.azimuth[::-1], 0.0), v), dtype=float)
        return out

    def law(self, elevation: Array, azimuth: Array | None = None) -> tuple[Array, Array]:
        """The affine law this gain amounts to at these ray angles: (a, b) of
        ``1 / z = a / D + b``, each inside the affine law's own bounds."""
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = self.slope(elevation, azimuth)
            a = np.clip(np.where(alpha > 0.0, 1.0 / alpha, A_BOUNDS[1]), *A_BOUNDS)
            b = np.clip(np.where(alpha > 0.0, -self.beta / alpha, 0.0), *B_BOUNDS)
        return a, b

    def apply(self, depth: Array, elevation: Array, azimuth: Array | None = None) -> Array:
        """The network's depth in metres through this gain; the angles broadcast against the
        image (a column vector of elevations is one law per row). Pixels the law cannot place
        (a non-positive inverse depth) become NaN, as :func:`pepin.depth.apply_affine` does."""
        d = np.asarray(depth, dtype=float)
        a, b = self.law(elevation, azimuth)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = a / d + b
            out: Array = np.where(np.isfinite(inv) & (inv > 1e-6), 1.0 / inv, np.nan)
        return out

    def scale_at(self, elevation_deg: float) -> float:
        """The law's ``a`` at one elevation in degrees, for a report line."""
        a, _b = self.law(np.array([math.radians(elevation_deg)]))
        return float(a[0])

    def describe(self) -> str:
        """The gain in a few words: its degree, its ``a`` at the bottom, middle and top of the
        fitted span, that span in degrees, and whether a bound binds inside it."""
        lo, hi = math.degrees(self.lo), math.degrees(self.hi)
        mid = 0.5 * (lo + hi)
        scales = " ".join(f"{self.scale_at(e):.2f}@{e:+.0f}" for e in (lo, mid, hi))
        return (
            f"ray deg {self.degree}{'+az' if self.azimuth.size else ''} a {scales}"
            f" b {-self.beta * self.scale_at(mid):+.3f} span {lo:+.0f}..{hi:+.0f} deg"
            f" on {self.pairs} pairs{' CLIPPED' if self.clipped else ''}"
        )

    def state(self) -> dict[str, Any]:
        """The gain as plain JSON values, for :func:`pepin.depth.save_law`."""
        return {
            "alpha": [float(c) for c in self.alpha],
            "beta": float(self.beta),
            "lo": float(self.lo),
            "hi": float(self.hi),
            "clipped": bool(self.clipped),
            "pairs": int(self.pairs),
            "azimuth": [float(c) for c in self.azimuth],
        }

    @classmethod
    def restore(cls, state: Any) -> RayGain | None:
        """The gain a :meth:`state` was written from, or ``None`` when the record is missing,
        malformed, empty or rests on fewer than POOL_MIN_SAMPLES pairs (a saved law is trusted
        no further than a live one)."""
        try:
            alpha = np.asarray([float(c) for c in state["alpha"]], dtype=float)
            azimuth = np.asarray([float(c) for c in state.get("azimuth", ())], dtype=float)
            gain = cls(
                alpha,
                float(state["beta"]),
                float(state["lo"]),
                float(state["hi"]),
                bool(state.get("clipped", False)),
                int(state["pairs"]),
                azimuth,
            )
        except (TypeError, ValueError, KeyError, IndexError):
            return None
        finite = bool(np.isfinite(alpha).all() and np.isfinite(azimuth).all())
        if alpha.size == 0 or not finite or not math.isfinite(gain.beta):
            return None
        if gain.hi <= gain.lo or gain.pairs < POOL_MIN_SAMPLES:
            return None
        return gain


def usable_degree(elevation: Array, lo: float, hi: float, degree: int) -> int:
    """The highest degree up to ``degree`` this pool of ray elevations can carry: every degree
    needs another MIN_RAY_SPAN of span, and every bin of the span cut into ``degree + 1`` equal
    parts must hold RAY_BIN_MIN pairs — a quadratic fitted on two clusters is a line through
    the gap. 0 means the pool says nothing about the angle."""
    span = hi - lo
    for deg in range(degree, 0, -1):
        if span < deg * MIN_RAY_SPAN:
            continue
        edges = np.linspace(lo, hi, deg + 2)
        counts = np.histogram(elevation, bins=edges)[0]
        if int(counts.min()) >= RAY_BIN_MIN:
            return deg
    return 0


def _solve(design: Array, target: Array, weight: Array) -> Array:
    """Weighted least squares with the worst quarter of the residuals dropped once and the fit
    repeated (:func:`pepin.depth._fit_noisy_on_exact`'s robustness, several columns wide)."""
    w = np.sqrt(weight)[:, None]
    coef, *_ = np.linalg.lstsq(design * w, target * w[:, 0], rcond=None)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):  # Accelerate's BLAS
        residual = np.abs(target - design @ coef)  # raises flags on finite matrices
    keep = residual <= np.percentile(residual, RESIDUAL_KEEP)
    if int(keep.sum()) > design.shape[1]:
        coef, *_ = np.linalg.lstsq(design[keep] * w[keep], (target * w[:, 0])[keep], rcond=None)
    out: Array = coef
    return out


def fit_ray(
    d: Array,
    z: Array,
    elevation: Array,
    weight: Array | None = None,
    *,
    azimuth: Array | None = None,
    degree: int = RAY_DEGREE,
    azimuth_degree: int = RAY_AZIMUTH_DEGREE,
) -> RayGain | None:
    """The law ``1 / z = a(ray) / D + b(ray)`` over (network, true, ray angle) pairs, or
    ``None`` when the pool cannot carry one.

    ``d`` is the network's depth, ``z`` the true depth along the optical axis, ``elevation``
    and ``azimuth`` the rays' angles off the optical axis in radians, ``weight`` each pair's
    share in the squares (a lidar beam is 1). The regression is
    ``1 / D = alpha(ray) * (1 / z) + beta`` — the noisy variable on the left, as
    :func:`pepin.depth.fit_affine` — with ``alpha`` a polynomial in the normalised elevation
    (and azimuth) and one shift ``beta``, fitted only when the pool spans MIN_DEPTH_SPREAD in
    depth, exactly as the affine law's shift is. ``None`` comes back under POOL_MIN_SAMPLES
    pairs, on a pool too narrow in angle for even a straight line (:func:`usable_degree`), or
    on a fit whose slope is not positive across its own span; the caller then keeps its affine
    law."""
    d = np.asarray(d, dtype=float)
    z = np.asarray(z, dtype=float)
    eps = np.asarray(elevation, dtype=float)
    if d.size < POOL_MIN_SAMPLES:
        return None
    w = np.ones_like(d) if weight is None else np.asarray(weight, dtype=float)
    span = np.percentile(eps, (2, 98))
    lo, hi = float(span[0]), float(span[1])
    deg = usable_degree(eps, lo, hi, degree)
    if deg < 1:
        return None
    x, y = 1.0 / d, 1.0 / z
    u = np.clip(eps, lo, hi) / RAY_SCALE
    columns = [y * u**k for k in range(deg + 1)]
    az_deg = 0
    if azimuth is not None and azimuth_degree > 0:
        v = np.asarray(azimuth, dtype=float) / RAY_SCALE
        az_deg = azimuth_degree
        columns += [y * v**j for j in range(1, az_deg + 1)]
    z_lo, z_hi = np.percentile(z, (5, 95))
    with_shift = bool(z_hi / z_lo >= MIN_DEPTH_SPREAD)
    if with_shift:
        columns.append(np.ones_like(y))
    coef = _solve(np.stack(columns, axis=1), x, w)
    alpha = coef[: deg + 1]
    az_coef = coef[deg + 1 : deg + 1 + az_deg]
    beta = float(coef[-1]) if with_shift else 0.0
    if not (np.isfinite(coef).all() and math.isfinite(beta)):
        return None
    gain = RayGain(alpha, beta, lo, hi, False, int(d.size), az_coef)
    grid: Array = np.linspace(lo, hi, RAY_GRID, dtype=float)
    slope = gain.slope(grid, np.zeros_like(grid) if az_deg else None)
    if not bool(np.all(slope > 0.0)):
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        scale, shift = 1.0 / slope, -beta / slope
    clipped = bool(
        np.any(scale < A_BOUNDS[0])
        or np.any(scale > A_BOUNDS[1])
        or np.any(shift < B_BOUNDS[0])
        or np.any(shift > B_BOUNDS[1])
    )
    return RayGain(alpha, beta, lo, hi, clipped, int(d.size), az_coef)


__all__ = [
    "MIN_RAY_SPAN",
    "RAY_AZIMUTH_DEGREE",
    "RAY_DEGREE",
    "RAY_SCALE",
    "RayGain",
    "fit_ray",
    "ray_angles",
    "usable_degree",
]
