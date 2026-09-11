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
        )


def fuse(measurements: Sequence[PoseMeasurement]) -> PoseMeasurement | None:
    """The information-weighted pose of several measurements: ``None`` for none, the one
    itself for one, and for more the information-filter update ``sum(L_i) mu = sum(L_i mu_i)``
    with the headings taken relative to the surest source's, so nothing is averaged across
    +-pi. The result's covariance is ``inv(sum(L_i))``, its fit the information-weighted mean
    of the fits, its stamp the newest, its source the names joined with ``+``."""
    if not measurements:
        return None
    if len(measurements) == 1:
        return measurements[0]
    infos = [m.information for m in measurements]
    reference = measurements[int(np.argmax([info[2, 2] for info in infos]))].yaw
    total = np.zeros((3, 3))
    weighted = np.zeros(3)
    weight = 0.0
    fit = 0.0
    for m, info in zip(measurements, infos, strict=True):
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
        source="+".join(m.source for m in measurements),
        stamp=max(m.stamp for m in measurements),
        fit=fit / weight if weight > 0.0 else 0.0,
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
    anisotropy a fusion needs; a scan that fits nowhere has a flat surface and a wide covariance
    in every direction, bounded by the window — the window is the prior. The whole matrix is
    divided by ``max(fit, FIT_FLOOR)^2`` — a poor fit is a wide answer — and by ``trust``, the
    source's own weight (a camera's depth is not a lidar's range).
    """
    denominator = max(surface.n_points, 1) * (surface.top if surface.top > 0.0 else 1.0)
    per_beam = surface.scores / denominator
    best = per_beam[surface.k, surface.i]
    weights = np.exp((per_beam - best) / (TEMPERATURE * max(float(best), PEAK_FLOOR)))  # (T, P)
    dxy = surface.positions - surface.positions[surface.i]  # (P, 2)
    dtheta = np.arctan2(  # (T,), wrapped: a lattice never spans a half turn, but a test may
        np.sin(surface.headings - surface.headings[surface.k]),
        np.cos(surface.headings - surface.headings[surface.k]),
    )
    total = float(weights.sum())
    w_xy = weights.sum(axis=0)  # (P,)
    w_theta = weights.sum(axis=1)  # (T,)
    # einsum, not matmul: Accelerate's BLAS raises spurious divide-by-zero warnings on these
    cov = np.zeros((3, 3))
    cov[:2, :2] = np.einsum("p,pi,pj->ij", w_xy, dxy, dxy) / total
    cov[2, 2] = float((w_theta * dtheta * dtheta).sum() / total)
    cov[2, :2] = cov[:2, 2] = np.einsum("kp,k,pj->j", weights, dtheta, dxy) / total
    step_xy, step_theta = surface.xy_step_m, surface.theta_step
    cov += np.diag([step_xy**2 / 12.0, step_xy**2 / 12.0, step_theta**2 / 12.0])
    cov /= max(fit, FIT_FLOOR) ** 2
    cov /= max(trust, 1e-6)
    return cov
