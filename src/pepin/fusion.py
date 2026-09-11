"""One pose from several sensors: every source's match is a measurement with a covariance,
and the measurements are fused by their information.

A lidar sees the whole room and pins the pose in every direction; a camera's virtual scan sees
one wall through a +-40 degree fan and pins the distance to that wall and the heading, but not
where along the wall the cart stands. Averaging such answers by hand would let the camera drag
the pose along the wall; weighting each by its inverse covariance (a Kalman update with no
process step, the classic information filter) takes from every source only what it knows. The
covariance itself is read off the correlative matcher's score surface: how far one can move the
candidate pose before the scan stops fitting, per direction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import ScoreSurface

Matrix = NDArray[np.float64]

# The temperature of the likelihood the covariance is the spread of, as a share of the peak's
# own per-beam score: a candidate that has lost a tenth of the peak's score is 1/e as likely.
# Relative to the peak, not to the surface's range: the free-floor penalty at the window's edge
# would otherwise widen every answer. PEAK_FLOOR is the peak a scan fitting nothing is judged
# by, so its flat surface reads wide instead of sharp noise.
TEMPERATURE = 0.1
PEAK_FLOOR = 0.2
FIT_FLOOR = 0.1  # a fit below this inflates the covariance no further (a hundredfold)
COV_RIDGE = 1e-12  # keeps a covariance invertible when a lattice step is zero
# A match the window bounded is a bound, not a measurement, and is widened by this so it can
# only ever be a faint vote. The window bounds a match two ways: the winner sits on the
# lattice's edge (the best pose lay outside what was searched, :func:`at_edge`), or the
# likelihood is flat along some direction (:func:`bound_directions`) — there the winner is the
# lattice's own tie-break toward the guess and the spread about it is the window's, so an
# un-widened plateau would vote for the prediction as if it had measured it (a review probe,
# 2026-09-11: a fan blind along a wall out-voted the lidar's edge-bound match by 113 to 80 in
# information and held a 12 cm slip for seconds). On a fan that sees one wall the plateau is
# the direction along the wall; on a scan that fits nothing, every direction.
BOUND_INFLATION = 100.0
# A direction is a plateau when the likelihood's spread along it is at least this share of a
# flat surface's (the lattice's uniform second moment about the winner). A likelihood that
# falls to 1/e only at the window's edge — a drop of one temperature across the half-width —
# reads 0.70 on the node's 7 position samples, 0.73 on its 13 heading samples and 0.75 in the
# limit; anything flatter is the window's answer, not the scan's.
PLATEAU_RATIO = 0.7
# A measurement this far (Mahalanobis, squared, 3 degrees of freedom) from the surest one
# does not describe the same pose: chi-square at 99 %. Left out of the fusion, named in
# ``rejected``. A camera scan of a table top the lidar's map has no wall for matches the map
# well somewhere within the window, and only its disagreement with the lidar gives it away.
GATE = 11.34


@dataclass(frozen=True)
class PoseMeasurement:
    """One source's word on the pose: where, how sure in each direction (a 3x3 covariance over
    x, y, yaw in metres and radians), from which source, at what stamp, with what fit."""

    x: float
    y: float
    yaw: float
    covariance: Matrix
    source: str
    stamp: float
    fit: float
    rejected: tuple[str, ...] = ()  # sources a fusion left out for disagreeing (see ``fuse``)
    edge: bool = False  # the match sat on the window's edge: a bound, the truth lies beyond

    @property
    def pose(self) -> Pose2D:
        """The measured pose."""
        return Pose2D(self.x, self.y, self.yaw)

    @property
    def information(self) -> Matrix:
        """The inverse covariance: the weight this measurement carries in a fusion."""
        return np.asarray(np.linalg.inv(self.covariance + COV_RIDGE * np.eye(3)), dtype=np.float64)

    @property
    def sigmas(self) -> tuple[float, float, float]:
        """Standard deviations in x, y (metres) and yaw (radians): the diagonal's roots."""
        d = np.sqrt(np.maximum(np.diag(self.covariance), 0.0))
        return float(d[0]), float(d[1]), float(d[2])

    def text(self) -> str:
        """``source (x, y, yaw deg) +- sx/sy cm, syaw deg, fit`` for a report line."""
        sx, sy, st = self.sigmas
        return (
            f"{self.source} ({self.x:+.2f}, {self.y:+.2f}, {math.degrees(self.yaw):+.1f} deg) "
            f"+- {sx * 100:.1f}/{sy * 100:.1f} cm, {math.degrees(st):.1f} deg, fit {self.fit:.2f}"
            + (", edge" if self.edge else "")
        )


def disagreement(a: PoseMeasurement, b: PoseMeasurement) -> float:
    """The squared Mahalanobis distance between two measurements of the same pose, under the
    sum of their covariances; the heading difference wrapped."""
    d = np.array([b.x - a.x, b.y - a.y, wrap_angle(b.yaw - a.yaw)])
    joint = a.covariance + b.covariance + COV_RIDGE * np.eye(3)
    return float(d @ np.linalg.solve(joint, d))


def fuse(measurements: Sequence[PoseMeasurement], gate: float = GATE) -> PoseMeasurement | None:
    """The information-weighted pose of several measurements: ``None`` for none, the one
    itself for one, and for more the information-filter update ``sum(L_i) mu = sum(L_i mu_i)``
    with the headings taken relative to the surest source's (the most heading information),
    so nothing is averaged across +-pi. A measurement whose :func:`disagreement` with the
    surest one exceeds ``gate`` is left out and named in ``rejected`` (``gate`` ``inf`` fuses
    all). The result's covariance is ``inv(sum(L_i))``, its fit the information-weighted mean
    of the fits, its stamp the newest, its source the names fused joined with ``+``."""
    if not measurements:
        return None
    if len(measurements) == 1:
        return measurements[0]
    infos = [m.information for m in measurements]
    surest = measurements[int(np.argmax([info[2, 2] for info in infos]))]
    reference = surest.yaw
    kept = [
        (m, info)
        for m, info in zip(measurements, infos, strict=True)
        if m is surest or disagreement(surest, m) <= gate
    ]
    rejected = tuple(m.source for m in measurements if all(m is not k for k, _ in kept))
    total = np.zeros((3, 3))
    weighted = np.zeros(3)
    weight = 0.0
    fit = 0.0
    for m, info in kept:
        z = np.array([m.x, m.y, wrap_angle(m.yaw - reference)])
        total += info
        weighted += info @ z
        share = float(np.trace(info))
        weight += share
        fit += share * m.fit
    covariance = np.asarray(np.linalg.inv(total), dtype=np.float64)
    mean = covariance @ weighted
    return PoseMeasurement(
        x=float(mean[0]),
        y=float(mean[1]),
        yaw=wrap_angle(reference + float(mean[2])),
        covariance=(covariance + covariance.T) / 2.0,
        source="+".join(m.source for m, _ in kept),
        stamp=max(m.stamp for m, _ in kept),
        fit=fit / weight if weight > 0.0 else 0.0,
        rejected=rejected,
    )


def covariance_from_score_surface(surface: ScoreSurface, fit: float, trust: float = 1.0) -> Matrix:
    """How sure a match is, per direction, from the lattice it was chosen on.

    Every candidate is weighted by ``exp((score - best) / (TEMPERATURE * best))`` with the scores
    per beam in units of the wall's peak (so a map's log-odds scale does not enter; Olson's
    likelihood with a temperature: the lattice's own spread, not a beam-noise model); the
    covariance is the weighted second moment of the candidates about the winner (not about
    their mean: the winner is the pose reported, and on a plateau the truth is anywhere on it),
    plus a lattice step's quantisation (``step^2 / 12``) per axis. A scan that fits equally
    well anywhere along a wall keeps its weight along it and loses it across, which is the
    anisotropy a fusion needs. But a direction the likelihood never falls along inside the
    window was not measured at all: the window bounded it, and the winner there is the
    lattice's tie-break toward the guess. Such directions (:func:`bound_directions`), like a
    winner on the window's edge (:func:`at_edge`), are widened by ``BOUND_INFLATION`` — a
    scan that fits nowhere, flat everywhere, carries no vote. The whole matrix is divided by
    ``max(fit, FIT_FLOOR)^2`` — a poor fit is a wide answer — and by ``trust``, the source's
    own weight (a camera's depth is not a lidar's range).
    """
    weights, dxy, dtheta = _likelihood(surface)
    cov = _spread(weights, dxy, dtheta)
    flat = _bound_directions(cov, dxy, dtheta)
    step_xy, step_theta = surface.xy_step_m, surface.theta_step
    cov += np.diag([step_xy**2 / 12.0, step_xy**2 / 12.0, step_theta**2 / 12.0])
    cov /= max(fit, FIT_FLOOR) ** 2
    cov /= max(trust, 1e-6)
    if at_edge(surface):
        return cov * BOUND_INFLATION
    for v in flat:
        # A congruence: the variance along ``v`` grows BOUND_INFLATION-fold, its covariance
        # with the other directions by the root of that, and the matrix stays positive definite.
        widen = np.eye(3) + (math.sqrt(BOUND_INFLATION) - 1.0) * np.outer(v, v)
        cov = widen @ cov @ widen
    return cov


def _likelihood(surface: ScoreSurface) -> tuple[Matrix, Matrix, NDArray[np.float64]]:
    """Every candidate's likelihood weight (T, P) relative to the winner, with the position
    offsets (P, 2) and the wrapped heading offsets (T,) about it."""
    denominator = max(surface.n_points, 1) * (surface.top if surface.top > 0.0 else 1.0)
    per_beam = surface.scores / denominator
    best = per_beam[surface.k, surface.i]
    weights = np.exp((per_beam - best) / (TEMPERATURE * max(float(best), PEAK_FLOOR)))  # (T, P)
    dxy = surface.positions - surface.positions[surface.i]  # (P, 2)
    dtheta = np.arctan2(  # (T,), wrapped: a lattice never spans a half turn, but a test may
        np.sin(surface.headings - surface.headings[surface.k]),
        np.cos(surface.headings - surface.headings[surface.k]),
    )
    return weights, dxy, dtheta


def _spread(weights: Matrix, dxy: Matrix, dtheta: NDArray[np.float64]) -> Matrix:
    """The weighted second moment (3x3 over x, y, yaw) of the candidates about the winner."""
    total = float(weights.sum())
    w_xy = weights.sum(axis=0)  # (P,)
    w_theta = weights.sum(axis=1)  # (T,)
    # einsum, not matmul: Accelerate's BLAS raises spurious divide-by-zero warnings on these
    cov = np.zeros((3, 3))
    cov[:2, :2] = np.einsum("p,pi,pj->ij", w_xy, dxy, dxy) / total
    cov[2, 2] = float((w_theta * dtheta * dtheta).sum() / total)
    cov[2, :2] = cov[:2, 2] = np.einsum("kp,k,pj->j", weights, dtheta, dxy) / total
    return cov


def _bound_directions(
    spread: Matrix, dxy: Matrix, dtheta: NDArray[np.float64]
) -> list[NDArray[np.float64]]:
    """:func:`bound_directions` from the spread and the offsets already in hand."""
    flat: list[NDArray[np.float64]] = []
    if len(dxy) > 1:
        uniform = dxy.T @ dxy / len(dxy)  # a flat surface's second moment about the winner
        values, vectors = np.linalg.eigh(spread[:2, :2])
        for value, v in zip(values, vectors.T, strict=True):
            if value >= PLATEAU_RATIO * float(v @ uniform @ v):
                flat.append(np.array([v[0], v[1], 0.0]))
    if len(dtheta) > 1 and spread[2, 2] >= PLATEAU_RATIO * float((dtheta * dtheta).mean()):
        flat.append(np.array([0.0, 0.0, 1.0]))
    return flat


def bound_directions(surface: ScoreSurface) -> list[NDArray[np.float64]]:
    """Unit directions in (x, y, yaw) the scan does not resolve inside the window: along each
    the likelihood's spread is at least ``PLATEAU_RATIO`` of a flat surface's, so the window
    bounded the answer and the winner is the tie-break toward the guess. The heading is judged
    on its own; the position along the eigen-directions of its spread, so a wall at any angle
    to the map's axes is found along itself. Empty for a single-candidate lattice."""
    weights, dxy, dtheta = _likelihood(surface)
    return _bound_directions(_spread(weights, dxy, dtheta), dxy, dtheta)


def at_edge(surface: ScoreSurface) -> bool:
    """True when the winner sits on the lattice's border in position or heading: the scan
    wanted to go farther than the window let it, and the answer is the window, not the scan."""
    headings, positions = len(surface.headings), len(surface.positions)
    if headings > 1 and surface.k in (0, headings - 1):
        return True
    side = round(math.sqrt(positions))
    if side * side != positions or side < 2:
        return False
    ix, iy = divmod(surface.i, side)
    return ix in (0, side - 1) or iy in (0, side - 1)
