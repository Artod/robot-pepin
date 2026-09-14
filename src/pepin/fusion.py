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

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from pepin.deployment import config_file
from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import ScoreSurface, apply_motion

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
# A pose with no score surface behind it — the tracker's own belief, an operator's word — is
# still a measurement, and :func:`from_fit` gives it the only spread it has: its fit. A pose that
# fits the map perfectly is worth SIGMA_XY_M / SIGMA_YAW_DEG, one that fits nothing those plus
# the LOST pair. The numbers are the ones the tracker has always published its pose with
# (pepin_bringup.relocalizer's covariance on /tracker_pose), moved here so the tracker's word and
# a scan's match are weighed on one scale.
SIGMA_XY_M = 0.05
SIGMA_XY_LOST_M = 0.30
SIGMA_YAW_DEG = 3.0
SIGMA_YAW_LOST_DEG = 20.0


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


def carry_pose(pose: Pose2D, covariance: Matrix, motion: Pose2D) -> tuple[Pose2D, Matrix]:
    """A measured pose and its covariance moved forward over ``motion`` — the odometry's step,
    in the base frame of the moment the pose was measured at: where that place has become.

    The covariance travels through the composition's Jacobian, ``J = [[1, 0, -dy], [0, 1, dx],
    [0, 0, 1]]`` over the carry's map-frame displacement: a heading known to a degree is 2 mm of
    position error after 10 cm of carry, and that coupling is the only thing the move adds. This
    is the GEOMETRY of the move alone; what the odometry's own error over the trail costs is
    :func:`odometry_covariance`, added by :func:`carried` — the caller that reads the result as
    a measurement of the new moment. A caller that only wants a pose put somewhere else (a
    prediction the matcher will search around) wants this one.
    """
    moved = apply_motion(pose, motion)
    dx, dy = moved.x - pose.x, moved.y - pose.y
    jacobian = np.array([[1.0, 0.0, -dy], [0.0, 1.0, dx], [0.0, 0.0, 1.0]])
    return moved, np.asarray(
        jacobian @ np.asarray(covariance, dtype=float) @ jacobian.T, dtype=np.float64
    )


# -- what the odometry's own trail costs a measurement carried over it ----------------------
# A measurement carried to a later moment is only as sure as the odometry that carried it, and
# this cart's odometry is not free. Until the peak covariance landed the carry added nothing and
# said so: at a match covariance of 12.8 cm (:func:`sigma_from_fit` at fit 0.74) the odometry's
# error over a fraction of a second really was two orders under it. At 1 cm / 0.5 deg
# (:func:`peak_covariance`) the claim is false, and the self-check read the odometry's own turn
# error as the lidar scattering: on tape 20260913_190024 it widened the lidar on 172 of the 307
# updates the cart was moving for (scratch/lidar_selfcheck_replay.py).
#
# The model is one sigma per direction, growing with the carry's own geometry:
#     sigma_yaw = ODOM_YAW_FLOOR_RAD + ODOM_YAW_PER_TURN * |turn|
#     sigma_xy  = ODOM_XY_FLOOR_M + ODOM_XY_PER_M * distance + sigma_yaw * distance
#
# ODOM_YAW_PER_TURN is the rotation this cart's WHEELS report and do not perform, and it is
# large. Over that tape the odometry turned 604 deg against the lidar's 359 (41 % of the
# reported turn never happened), and per carry the error is 0.52 of the reported turn at the
# median with an RMS of 0.77 — 0.7 once the lidar's own per-match noise (0.84 deg joint against
# a 2.8 deg median carry, a share of 0.30) is taken back out in quadrature
# (scratch/tape_odometry_error.py). It is the same 40-60 % in-place slip the gyro measured on
# 2026-09-11 (0.344 rad/s at the encoders, 0.20 and 0.14 rad/s by the gyro): a differential
# drive's heading is the difference of two wheels, and on carpet that difference is half fiction.
# With the IMU up /odometry/filtered is far better than this, measured the same way on tape
# 0240_20260913_204114 (which does carry ekf and imu): 678 deg of odometry against 716 of lidar
# and a per-carry RMS of 0.28, the lidar's own noise still inside it. 0.7 is the wheels-only end
# on purpose — it is what the tracker is left holding when the IMU drops, and tape 20260913_190024
# recorded no ekf and no imu at all — so with the gyro alive the term is some 2.5x conservative.
# What it costs: while the cart turns, this term IS the denominator's heading and the check
# cannot see a source over-claiming its heading. That price is paid only while turning, and not
# on what the check exists for — a camera claiming 25 cm while its own answers fall 0.8-1.5 cm
# apart is a POSITION over-claim, and the position term is under 3 mm at these carries.
# ODOM_XY_PER_M is a class, not a measurement: the same tape's per-carry distance share is 0.96
# and its whole-drive path lengths are 4.37 m of odometry against 4.81 m of lidar, but a 1-3 cm
# carry is buried in the lidar's own 1 cm, so nothing sharper can be read off it. 2 % is the
# class of a wheel-odometry scale error, and it is 2 mm over the longest carry made here (the
# camera's 0.3 s, ~10 cm at 0.3 m/s).
# The floors are what a carry costs when the odometry reports no motion at all: a wheel quantum,
# and the gyro's bias over one carry (-0.02 to -0.17 deg/s measured at rest, 0.02 deg over
# 0.13 s). Both are a fifth or less of what the peak gives a good match, so they never set the
# answer — they only keep the sum from being zero.
ODOM_XY_FLOOR_M = 0.002
ODOM_XY_PER_M = 0.02
ODOM_YAW_FLOOR_RAD = math.radians(0.05)
ODOM_YAW_PER_TURN = 0.7


def odometry_covariance(motion: Pose2D) -> Matrix:
    """The odometry's OWN error over one carry as a 3x3 covariance over x, y, yaw (metres^2,
    radians^2): the floors plus what the carry's distance and turn are worth.

    ``motion`` is the carry's step in the base frame of the moment carried FROM
    (:func:`pepin.scanmatch.relative_motion` over the odometry). Diagonal: a differential
    drive's two scale errors are not correlated in any way this cart has measured, and the one
    coupling that matters — a heading wrong by ``sigma_yaw`` puts the end of a carry of
    ``distance`` that far to the side — is folded into the position sigma rather than modelled
    as an off-diagonal term nobody could calibrate.
    """
    distance = math.hypot(motion.x, motion.y)
    sigma_yaw = ODOM_YAW_FLOOR_RAD + ODOM_YAW_PER_TURN * abs(motion.theta)
    sigma_xy = ODOM_XY_FLOOR_M + ODOM_XY_PER_M * distance + sigma_yaw * distance
    return np.asarray(np.diag([sigma_xy**2, sigma_xy**2, sigma_yaw**2]), dtype=np.float64)


def carried(measurement: PoseMeasurement, motion: Pose2D, stamp: float) -> PoseMeasurement:
    """The same measurement read at a later moment: its pose and covariance carried over
    ``motion`` (:func:`carry_pose`), widened by what the odometry's own error over that carry
    costs (:func:`odometry_covariance`), and re-stamped to ``stamp``.

    The answer is the same answer, read from where the cart has got to — and it is only as sure
    as the trail that moved it, which is why the odometry term belongs HERE and not in the
    geometry. The fit is the scan's own and does not change.
    """
    pose, covariance = carry_pose(measurement.pose, measurement.covariance, motion)
    return replace(
        measurement,
        x=pose.x,
        y=pose.y,
        yaw=pose.theta,
        covariance=covariance + odometry_covariance(motion),
        stamp=stamp,
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


def sigma_from_fit(fit: float) -> tuple[float, float]:
    """How sure a pose known only by its fit is: ``(position metres, heading radians)``, from
    :data:`SIGMA_XY_M` / :data:`SIGMA_YAW_DEG` at a perfect fit to those plus the LOST pair at
    a fit of zero. Linear on purpose: it is a report of confidence, not a measurement. NaN — no
    match yet — is as lost as zero, and never a NaN covariance downstream."""
    lost = 1.0 if fit != fit else 1.0 - min(max(fit, 0.0), 1.0)
    return (
        SIGMA_XY_M + SIGMA_XY_LOST_M * lost,
        math.radians(SIGMA_YAW_DEG + SIGMA_YAW_LOST_DEG * lost),
    )


def from_fit(pose: Pose2D, fit: float, source: str, stamp: float = 0.0) -> PoseMeasurement:
    """A pose that has no score surface behind it as a measurement any fusion can take: the
    pose itself, an isotropic position sigma and a heading sigma from :func:`sigma_from_fit`.
    What the tracker's own belief is worth beside a fresh match of a scan."""
    sigma_xy, sigma_yaw = sigma_from_fit(fit)
    return PoseMeasurement(
        pose.x,
        pose.y,
        pose.theta,
        np.diag([sigma_xy**2, sigma_xy**2, sigma_yaw**2]),
        source,
        stamp,
        fit,
    )


# -- the covariance of the score peak ------------------------------------------------------
# Which covariance a match carries: the spread of its own score peak (:func:`peak_covariance`,
# a temperature calibrated against the replay's truth) or the fit-scaled surface moment that
# shipped before it (:func:`covariance_from_score_surface`, whose scale nobody measured). The
# words of the ``covariance`` flag on the tracker and on the laptop's matcher.
PEAK = "peak"
FIT = "fit"
COVARIANCE_CHOICES = (PEAK, FIT)
# A peak sharper than this is the lattice's own resolution speaking, not a measurement: the
# floor keeps a one-cell peak's covariance invertible and its information finite. A millimetre
# is a fifth of the lidar's measured error at a good fit (0.5-0.6 cm, scratch/drive_bisect.py
# on the four goto tapes of 2026-09-13), so it never sets the answer, only bounds it.
MIN_SIGMA_XY_M = 0.001
MIN_SIGMA_YAW_RAD = math.radians(0.05)


@lru_cache(maxsize=1)
def _peak_temperatures() -> dict[str, float]:
    """``config/matcher.json``'s ``peak_temperature`` block: one temperature per matcher."""
    block = json.loads(config_file("matcher.json").read_text())["peak_temperature"]
    return {str(name): float(value) for name, value in block.items()}


def peak_temperature(matcher: str) -> float:
    """The temperature :func:`peak_covariance` reads ``matcher``'s score peak with, from
    ``config/matcher.json`` (``KeyError`` for a matcher the file does not name).

    One number per matcher — the lidar's revolution, the camera's fans — in units of per-beam
    score: a candidate a temperature below the peak weighs 1/e. It is the only free scale in
    the covariance, and it is calibrated, not chosen (scratch/peak_temperature.py).
    """
    return _peak_temperatures()[matcher]


def peak_covariance(surface: ScoreSurface, temperature: float) -> Matrix:
    """How sure a match is, per direction, as the spread of its own score peak: the 3x3
    covariance over x, y, yaw in metres^2 and radians^2.

    Olson's correlative scan matching, done as he does it. Every candidate of the lattice is
    weighted ``w_i = exp((s_i - s_max) / T)`` with ``s`` the per-beam score in units of the
    field's peak (so no map's log-odds scale enters and a 40-beam fan is on the lidar's scale),
    and the covariance is the weighted second moment of the candidate poses about their
    weighted MEAN — ``K = sum(w x x^T) / sum(w) - u u^T / sum(w)^2`` — not about the winner.
    A sharp peak keeps weight only on its own cell and reads as millimetres; a scan that fits
    equally well anywhere along a corridor keeps its weight along the corridor and loses it
    across, so the answer is a ridge, which is the anisotropy a fusion needs.

    Nothing invented is charged on top: no division by the fit, no per-source discount, and no
    lattice quantisation term (the ``step^2 / 12`` :func:`covariance_from_score_surface` adds).
    Dropping that term is not free, and the honest statement of what it costs: this moment is a
    SUM OVER THE LATTICE'S OWN NODES, so a peak sharper than the step is read through whichever
    phase of the grid it happens to fall on. Swept across one 3 cm cell on the furnished room's
    whole revolution (scratch/peak_skeptic.py) the reported sigma_x walks 0.82 to 1.65 cm, a
    factor of two, with nothing but that phase — which is exactly the spread ``step^2 / 12``
    (0.87 cm per axis at a 3 cm step) used to cover. The calibration absorbs it on AVERAGE and
    only on average: at T = 0.016 the position comes out 2-3x conservative in variance, so the
    phase noise sits inside that margin. The diagonal is floored at :data:`MIN_SIGMA_XY_M` /
    :data:`MIN_SIGMA_YAW_RAD` so a one-cell peak stays invertible. What a window's edge and a
    plateau do to it is :func:`from_peak`'s business, because those are bounds, not spreads.
    """
    weights = peak_weights(surface, temperature)
    positions, headings = surface.positions, surface.headings
    total = float(weights.sum())
    if total <= 0.0:  # every candidate underflowed: the winner alone, at the floor
        return np.diag([MIN_SIGMA_XY_M**2, MIN_SIGMA_XY_M**2, MIN_SIGMA_YAW_RAD**2])
    w_xy = weights.sum(axis=0)  # (P,)
    w_theta = weights.sum(axis=1)  # (T,)
    dtheta = _wrapped(headings - headings[surface.k])  # about the winner: never across +-pi
    dxy = positions - positions[surface.i]
    mean_xy = w_xy @ dxy / total
    mean_theta = float(w_theta @ dtheta / total)
    dxy = dxy - mean_xy
    dtheta = dtheta - mean_theta
    # einsum, not matmul: Accelerate's BLAS raises spurious divide-by-zero warnings on these
    cov = np.zeros((3, 3))
    cov[:2, :2] = np.einsum("p,pi,pj->ij", w_xy, dxy, dxy) / total
    cov[2, 2] = float((w_theta * dtheta * dtheta).sum() / total)
    cov[2, :2] = cov[:2, 2] = np.einsum("kp,k,pj->j", weights, dtheta, dxy) / total
    cov[0, 0] = max(cov[0, 0], MIN_SIGMA_XY_M**2)
    cov[1, 1] = max(cov[1, 1], MIN_SIGMA_XY_M**2)
    cov[2, 2] = max(cov[2, 2], MIN_SIGMA_YAW_RAD**2)
    return np.asarray((cov + cov.T) / 2.0, dtype=np.float64)


def peak_weights(surface: ScoreSurface, temperature: float) -> Matrix:
    """Every candidate's likelihood weight (T, P), ``exp((s - s_max) / temperature)`` on the
    per-beam score — the lattice as a probability, before any moment is taken of it."""
    denominator = max(surface.n_points, 1) * (surface.top if surface.top > 0.0 else 1.0)
    per_beam = surface.scores / denominator
    best = float(per_beam[surface.k, surface.i])
    return np.asarray(np.exp((per_beam - best) / max(temperature, 1e-9)), dtype=np.float64)


def _wrapped(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Angles wrapped into +-pi (a lattice never spans a half turn, but a test may)."""
    return np.asarray(np.arctan2(np.sin(angles), np.cos(angles)), dtype=np.float64)


def from_peak(
    surface: ScoreSurface,
    pose: Pose2D,
    fit: float,
    source: str,
    stamp: float = 0.0,
    temperature: float | None = None,
    trust: float = 1.0,
    matcher: str | None = None,
) -> PoseMeasurement:
    """A match as a measurement whose covariance is its score peak's (:func:`peak_covariance`).

    ``pose`` is the pose the matcher decided (already refined between the cells when it
    interpolates; :meth:`pepin.scanmatch.ScoreSurface.peak` is the same apex read off the
    lattice alone). ``temperature`` defaults to ``matcher``'s in ``config/matcher.json``, or
    the lidar's when neither is given. ``trust`` widens the answer by the source's own weight
    the way the fit-scaled path did — the camera's depth is not a range measurement — and is
    the one discount the temperature does not model.

    A bound is still not a spread: a winner on the window's edge (:func:`at_edge`), and every
    direction the likelihood never falls along inside the window (:func:`bound_directions`),
    are widened by :data:`BOUND_INFLATION` exactly as before, so a scan that fits nowhere
    cannot vote for the prediction as if it had measured it. WHETHER a direction is a bound is
    still judged by the old adaptive likelihood — that test and its :data:`PLATEAU_RATIO` were
    tuned on this robot's plateaus and are not what this change is about; only HOW WIDE the
    directions that are measurements come out is new.
    """
    if temperature is None:
        temperature = peak_temperature(matcher if matcher is not None else "lidar")
    cov = peak_covariance(surface, temperature)
    cov = cov / max(trust, 1e-6)
    if at_edge(surface):
        cov = cov * BOUND_INFLATION
    else:
        for v in bound_directions(surface):
            widen = np.eye(3) + (math.sqrt(BOUND_INFLATION) - 1.0) * np.outer(v, v)
            cov = widen @ cov @ widen
    return PoseMeasurement(
        pose.x,
        pose.y,
        pose.theta,
        np.asarray((cov + cov.T) / 2.0, dtype=np.float64),
        source,
        stamp,
        fit,
        edge=at_edge(surface),
    )


def published_covariance(fused: PoseMeasurement | None, fit: float, choice: str = PEAK) -> Matrix:
    """The 3x3 a tracker publishes beside its pose, under the ``covariance`` switch.

    ``peak`` with a fused match in hand: that match's own covariance — the spread of the score
    peak the pose was corrected by, anisotropic, so a corridor reads as a ridge along the
    corridor. Otherwise (``fit``, or no match yet: a carried belief, a re-seed, the moment
    before the first scan) the isotropic pair :func:`sigma_from_fit` draws from the inlier
    fraction, which is what every tape before 2026-09-13 carries.
    """
    if choice == PEAK and fused is not None:
        return np.asarray(fused.covariance, dtype=np.float64)
    sigma_xy, sigma_yaw = sigma_from_fit(fit)
    return np.diag([sigma_xy**2, sigma_xy**2, sigma_yaw**2])


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
