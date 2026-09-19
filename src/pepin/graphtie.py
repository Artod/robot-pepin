"""The tie between a lidar map and a visual database: a CALIBRATION of the pair, not a seating.

``map <- rtabmap`` is one rigid transform between two static things — a map the board serves and
a database RTAB-Map loads. Both are files; a transform between two files cannot move. Measured
2026-09-17 with the cart parked, RTAB-Map's own localisation in that database repeated to 0.5 cm
over 33 samples, and across a restart the implied transform came back the same within the
recognition's own heading noise ((-10.611, +5.893, -35.69 deg) against (-11.012, +5.749, -29.24
deg)). So the tie belongs with camera extrinsics: measured once over many places, written down,
and thereafter read.

WHY NOT ONE SEATING. Until 2026-09-17 the tie was fitted to a single (tracker pose, graph place)
sample and re-fitted at runtime whenever the two disagreed for five seconds. Both halves are the
same mistake — fitting a constant to a variable — and both were measured:

* one seating has a LEVER ARM. RTAB-Map's heading at one place is good to 4-6 deg; four metres
  away that is 0.4 m of position. The stored anchor took three values metres apart in one evening
  ((-9.22, -0.40, -127 deg) -> (-6.66, +2.49, -117) -> (-6.18, +4.34, -113));
* a re-learn bakes in a LIE. RTAB-Map once reported a 4.4 deg sigma while its recognition was
  ~93 deg wrong; a tie re-fitted on that sample is wrong for every word afterwards;
* and a runtime re-learn needs the LIDAR, so camera-only inherited a lidar dependency exactly
  where there is no lidar.

WHAT IS HERE INSTEAD. A :class:`TiePair` is one (cart on the map, cart in the database) sample,
taken only while the lidar has the tracker pinned in both axes and in heading
(:func:`pepin.anchors.seating_refusal`). Pairs are appended to a log beside the map
(:func:`pairs_path`) and :func:`fit_tie` reads the whole log and answers with one
:class:`Tie` — the transform, its covariance, how many pairs it stands on, how many of them it
had to throw away, and over how many metres they are spread. Nothing about this runs on a
disagreement, and nothing about it needs the lidar once the log exists: camera-only, no pair is
taken and the tie on file is used as it is.

THE FIT is a weighted 2D rigid registration (Procrustes/Kabsch on the positions, the pairs' own
headings as a second, independent measurement of the same rotation), made robust by a
deterministic search over minimal samples: ONE pair with a heading already determines a whole
rigid transform, so every pair is tried as a hypothesis, the one with the most inliers wins, and
the winner is refined over its inliers alone — a pair past :data:`pepin.fusion.GATE` is dropped
and not softened, which is the same rule the fusion applies to a measurement it does not believe.
RANSAC over random samples would answer differently on every run; this answers the same thing
every time, and at these sizes it costs a millisecond.
A false recognition is therefore not a threat to the tie but a MEASUREMENT: the share of pairs
outside the gate is the database's own false-recognition rate, and the report line says it.

THE COVARIANCE IS THE POINT. A tie fitted from pairs at one spot must say so rather than pretend:
its formal heading variance is ``1 / sum(w_i |g_i - centroid|^2)``, which blows up exactly when
the pairs have no spread, and the pairs' own headings then carry the rotation alone. The
covariance is inflated by the misfit when the data disagree with a rigid model by more than their
own error bars allow (the Birge ratio, ``sqrt(chi2 / dof)``), so a database whose sessions sit in
different frames yields a WIDE tie instead of a confident wrong one. And it is carried into every
word: :meth:`Tie.word_covariance` propagates the tie's uncertainty through the composition, so a
word far from the pairs' centroid is wide by the lever arm the heading sigma buys there.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pepin.anchors import ANCHOR_MAX_SIGMA_DEG, ANCHOR_MAX_SIGMA_M, map_slug
from pepin.fusion import GATE, Matrix
from pepin.measurements import compose, inverse
from pepin.odometry import Pose2D, wrap_angle

__all__ = [
    "FILE_TIE_SIGMA_DEG",
    "FILE_TIE_SIGMA_M",
    "HYPOTHESIS_BUDGET",
    "PAIRS_SUFFIX",
    "Tie",
    "TiePair",
    "append_pair",
    "file_tie",
    "fit_tie",
    "identity_tie",
    "load_pairs",
    "pairs_path",
]

PAIRS_SUFFIX = ".graph_pairs.jsonl"
# How many pairs are tried as the minimal-sample hypothesis of the robust search. This is a
# COMPUTE BUDGET and not a threshold: the search only ever improves with more hypotheses, and the
# stride that enforces it is spread evenly over the log, so the places the cart visited are still
# all represented. 256 hypotheses against a few thousand pairs is a millisecond of numpy, which is
# what a refit may cost inside a node that also broadcasts a transform at 10 Hz.
HYPOTHESIS_BUDGET = 256
# How many reweighting passes the refinement makes. Each pass re-decides which pairs are inside the
# gate and re-fits over those; from a minimal-sample start that already holds the inlier majority
# the set stops changing after one or two, and five is past that and still bounded.
REFINE_PASSES = 5
# What a tie READ FROM THE OLD ONE-SEATING FILE is worth, which is the whole reason the file is
# only a starting point. Measured 2026-09-17 across an RTAB-Map restart on the same map and the
# same database: the transform implied before and after was (-10.611, +5.893, -35.69 deg) against
# (-11.012, +5.749, -29.24 deg) — the same frame, re-measured from another single seating, moving
# 0.40 m in x, 0.14 m in y and 6.45 deg. That spread IS the error bar of a one-seating tie, so it
# is what such a tie declares until pairs replace it.
FILE_TIE_SIGMA_M = math.hypot(0.401, 0.144)
FILE_TIE_SIGMA_DEG = 6.45
# The least a pair may claim about itself, per axis, when the caller offers no covariance: the
# same seating gate the pair had to pass to be taken at all (pepin.anchors). A pair cannot be
# sharper than the test that admitted it, and a zero sigma would make one pair infinitely heavy.
PAIR_FLOOR_M = ANCHOR_MAX_SIGMA_M
PAIR_FLOOR_RAD = math.radians(ANCHOR_MAX_SIGMA_DEG)


@dataclass(frozen=True)
class TiePair:
    """One instant seen from both sides: where the lidar-held tracker says the cart is on the map,
    where RTAB-Map says the same cart is in the database it loaded, and how sharply each was said.

    ``sigma`` is the JOINT error bar of the two (x, y in metres, heading in radians) — the
    tracker's seating and the database's own localisation added in quadrature by whoever built the
    pair. It is what the fit weighs the pair by and what the gate judges its residual against.
    """

    stamp: float
    cart: Pose2D
    place: Pose2D
    sigma: tuple[float, float, float] = (PAIR_FLOOR_M, PAIR_FLOOR_M, PAIR_FLOOR_RAD)
    note: str = ""

    @property
    def sigma_xy(self) -> float:
        """The pair's position error bar in metres: the worse of the two axes, never under the
        floor of the seating test that admitted it."""
        return max(self.sigma[0], self.sigma[1], PAIR_FLOOR_M)

    @property
    def sigma_yaw(self) -> float:
        """The pair's heading error bar in radians, never under the seating test's own floor."""
        return max(self.sigma[2], PAIR_FLOOR_RAD)

    def implied(self) -> Pose2D:
        """The whole rigid tie this ONE pair implies: ``map <- graph`` carrying ``place`` exactly
        onto ``cart``. One pair with a heading is a complete minimal sample, which is what the
        robust search enumerates."""
        return compose(self.cart, inverse(self.place))

    def to_json(self) -> str:
        """One line of the pairs log: everything a later fit needs and nothing it does not."""
        return json.dumps(
            {
                "stamp": round(self.stamp, 4),
                "map": [round(self.cart.x, 4), round(self.cart.y, 4), round(self.cart.theta, 5)],
                "graph": [
                    round(self.place.x, 4),
                    round(self.place.y, 4),
                    round(self.place.theta, 5),
                ],
                "sigma": [round(float(s), 6) for s in self.sigma],
                **({"note": self.note} if self.note else {}),
            }
        )

    @classmethod
    def from_json(cls, text: str) -> TiePair:
        """One line back; ``ValueError``, ``KeyError`` or ``TypeError`` for a line that is not a
        pair, which the loader counts as a damaged line rather than dying on."""
        raw: dict[str, Any] = json.loads(text)
        cart, place = raw["map"], raw["graph"]
        sigma = raw.get("sigma", [PAIR_FLOOR_M, PAIR_FLOOR_M, PAIR_FLOOR_RAD])
        return cls(
            stamp=float(raw["stamp"]),
            cart=Pose2D(float(cart[0]), float(cart[1]), float(cart[2])),
            place=Pose2D(float(place[0]), float(place[1]), float(place[2])),
            sigma=(float(sigma[0]), float(sigma[1]), float(sigma[2])),
            note=str(raw.get("note", "")),
        )


def pairs_path(directory: Path | str, map_id: str) -> Path:
    """Where the pairs of ``map_id`` live: ``<directory>/<slug>.graph_pairs.jsonl``, beside the
    one-seating anchor file of the same pair (:func:`pepin.anchors.anchor_path`)."""
    return Path(directory) / f"{map_slug(map_id)}{PAIRS_SUFFIX}"


def append_pair(directory: Path | str, map_id: str, pair: TiePair) -> Path:
    """Append one pair to the log and answer with the path. Append-only on purpose: a calibration
    is worth what it was measured over, and a log that is rewritten cannot be argued with."""
    path = pairs_path(directory, map_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as log:
        log.write(pair.to_json() + "\n")
    return path


def load_pairs(directory: Path | str, map_id: str) -> list[TiePair]:
    """Every pair logged for ``map_id``, oldest first; an empty list when nothing is logged yet.

    A line that does not parse is skipped rather than raised on: the log is appended to by a live
    node and a power cut leaves half a line, which is not a reason to refuse the other thousand.
    """
    path = pairs_path(directory, map_id)
    if not path.exists():
        return []
    pairs: list[TiePair] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            pairs.append(TiePair.from_json(line))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError):
            continue
    return pairs


@dataclass(frozen=True)
class Tie:
    """``map <- graph`` for one (map, database) pair, with an error bar and its provenance.

    ``covariance`` is a 3x3 over the tie's own (x, y, yaw) — metres^2 and radians^2 — and it is
    referred to ``centroid``, the point in the GRAPH frame the pairs are spread about: a rotation
    error costs nothing there and grows with the lever arm away from it (:meth:`sigma_at`).
    ``centroid`` is ``None`` for a tie that has no such reference point (one read from the old
    one-seating file), and its covariance is then flat.
    """

    pose: Pose2D
    covariance: Matrix
    centroid: tuple[float, float] | None = None
    origin: str = "pairs"
    pairs: int = 0
    inliers: int = 0
    extent_m: float = 0.0
    residual_rms_m: float = 0.0
    heading_gap_deg: float = 0.0

    @property
    def outlier_share(self) -> float:
        """The share of pairs the fit had to throw away, which IS the measured rate at which this
        database recognises the wrong place."""
        if self.pairs <= 0:
            return 0.0
        return max(0.0, (self.pairs - self.inliers) / self.pairs)

    @property
    def volume(self) -> float:
        """The generalised variance of the tie (the determinant of its covariance): the one number
        two ties are compared on, because a refit replaces the tie in hand only when it is
        statistically BETTER and never because something disagreed with it."""
        return float(abs(np.linalg.det(np.asarray(self.covariance, dtype=float))))

    def lever_m(self, place: Pose2D) -> float:
        """How far ``place`` (graph frame) is from the point the tie was measured about: the arm a
        heading error turns into a position error. 0 for a tie with no reference point."""
        if self.centroid is None:
            return 0.0
        return math.hypot(place.x - self.centroid[0], place.y - self.centroid[1])

    def sigma_at(self, place: Pose2D) -> tuple[float, float]:
        """What the tie is worth where the cart actually is: (position sigma in metres, heading
        sigma in radians) at ``place`` in the graph frame, the position widened by the lever arm
        the heading sigma buys there."""
        covariance = np.asarray(self.covariance, dtype=float)
        sigma_yaw = math.sqrt(max(float(covariance[2, 2]), 0.0))
        sigma_xy = math.sqrt(max(float(covariance[0, 0]), float(covariance[1, 1]), 0.0))
        return math.hypot(sigma_xy, self.lever_m(place) * sigma_yaw), sigma_yaw

    def word_covariance(self, place: Pose2D) -> Matrix:
        """The tie's uncertainty as it lands on a WORD about ``place``: the tie's own covariance
        carried through the composition ``tie . place``, so a word far from the centroid is wide
        by the arm and not by the tie's own translation alone.

        The Jacobian is the composition's: a rotation error of the tie moves the word
        perpendicular to the vector from the centroid to the place, which is the same
        ``[[1, 0, -dy], [0, 1, dx], [0, 0, 1]]`` a carry uses
        (:func:`pepin.fusion.carry_pose`).
        """
        covariance = np.asarray(self.covariance, dtype=float)
        if self.centroid is None:
            return np.asarray(covariance, dtype=np.float64)
        cos, sin = math.cos(self.pose.theta), math.sin(self.pose.theta)
        ax, ay = place.x - self.centroid[0], place.y - self.centroid[1]
        dx, dy = cos * ax - sin * ay, sin * ax + cos * ay
        jacobian = np.array([[1.0, 0.0, -dy], [0.0, 1.0, dx], [0.0, 0.0, 1.0]])
        return np.asarray(jacobian @ covariance @ jacobian.T, dtype=np.float64)

    def described(self) -> str:
        """The tie in one phrase for a report line: where it points, where it came from and what
        it stands on."""
        where = (
            f"({self.pose.x:+.2f}, {self.pose.y:+.2f}, {math.degrees(self.pose.theta):+.1f} deg)"
        )
        if self.origin == "identity":
            return f"{where} identity (a frame born with this map)"
        if self.origin != "pairs":
            sigma_xy, sigma_yaw = self.sigma_at(Pose2D())
            return (
                f"{where} from {self.origin}, +- {sigma_xy * 100:.0f} cm,"
                f" {math.degrees(sigma_yaw):.1f} deg"
            )
        return (
            f"{where} from pairs {self.inliers}/{self.pairs} inliers over"
            f" {self.extent_m:.1f} m, rms {self.residual_rms_m * 100:.0f} cm,"
            f" {self.outlier_share * 100:.0f} % outliers, headings"
            f" {self.heading_gap_deg:+.1f} deg off the geometry"
        )


def identity_tie() -> Tie:
    """The tie of a frame BORN with this map: a map and a database started at the same pose in the
    same second are the same frame by construction, so the transform is identity and there is
    nothing to measure. No pairs, no file, no lidar."""
    return Tie(
        pose=Pose2D(),
        covariance=np.zeros((3, 3), dtype=np.float64),
        centroid=(0.0, 0.0),
        origin="identity",
    )


def file_tie(pose: Pose2D) -> Tie:
    """A tie read from the old one-seating anchor file, carrying the error bar such a tie was
    MEASURED to have (:data:`FILE_TIE_SIGMA_M`, :data:`FILE_TIE_SIGMA_DEG`).

    It has no reference point — nobody wrote down where the seating was — so its covariance is
    flat: the measured spread already includes whatever lever arm that evening had.
    """
    return Tie(
        pose=pose,
        covariance=np.asarray(
            np.diag(
                [
                    FILE_TIE_SIGMA_M**2,
                    FILE_TIE_SIGMA_M**2,
                    math.radians(FILE_TIE_SIGMA_DEG) ** 2,
                ]
            ),
            dtype=np.float64,
        ),
        centroid=None,
        origin="file",
    )


def _arrays(
    pairs: list[TiePair],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """The pairs as columns: map positions, map headings, graph positions, graph headings, and
    the two error bars — everything the fit touches, so no loop reads a dataclass again."""
    cart = np.array([[p.cart.x, p.cart.y] for p in pairs], dtype=float)
    cart_yaw = np.array([p.cart.theta for p in pairs], dtype=float)
    place = np.array([[p.place.x, p.place.y] for p in pairs], dtype=float)
    place_yaw = np.array([p.place.theta for p in pairs], dtype=float)
    sigma_xy = np.array([p.sigma_xy for p in pairs], dtype=float)
    sigma_yaw = np.array([p.sigma_yaw for p in pairs], dtype=float)
    return cart, cart_yaw, place, place_yaw, sigma_xy, sigma_yaw


def _wrap(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Every angle of an array brought into (-pi, pi]."""
    return np.asarray(np.arctan2(np.sin(angles), np.cos(angles)), dtype=np.float64)


def _residuals(
    tie: Pose2D,
    cart: NDArray[np.float64],
    cart_yaw: NDArray[np.float64],
    place: NDArray[np.float64],
    place_yaw: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Where a candidate tie puts every pair's graph place against where the map says the cart
    was: the position residual per pair (metres, 2 columns) and the heading residual (radians)."""
    cos, sin = math.cos(tie.theta), math.sin(tie.theta)
    rotation = np.array([[cos, -sin], [sin, cos]])
    moved = place @ rotation.T + np.array([tie.x, tie.y])
    return moved - cart, _wrap(place_yaw + tie.theta - cart_yaw)


def _mahalanobis(
    position: NDArray[np.float64],
    heading: NDArray[np.float64],
    sigma_xy: NDArray[np.float64],
    sigma_yaw: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The squared 3-DOF Mahalanobis distance of every pair's residual under its OWN joint
    covariance — the same quantity :func:`pepin.fusion.disagreement` computes for two
    measurements, and therefore the same one :data:`pepin.fusion.GATE` judges."""
    return np.asarray(
        (position[:, 0] ** 2 + position[:, 1] ** 2) / sigma_xy**2 + heading**2 / sigma_yaw**2,
        dtype=np.float64,
    )


def _weighted_fit(
    cart: NDArray[np.float64],
    cart_yaw: NDArray[np.float64],
    place: NDArray[np.float64],
    place_yaw: NDArray[np.float64],
    position_weight: NDArray[np.float64],
    heading_weight: NDArray[np.float64],
) -> tuple[Pose2D, tuple[float, float], float, float, float]:
    """One closed-form weighted rigid fit: the tie, the graph-frame centroid it is referred to,
    the formal variance of its translation and of its heading, and the heading the pairs' own
    headings alone say.

    The rotation is measured twice and independently: by the GEOMETRY of the positions (the 2D
    Procrustes closed form, whose variance is ``1 / sum(w |g - c|^2)`` and therefore blows up when
    the pairs have no spread) and by the pairs' own HEADINGS (a weighted circular mean, whose
    variance is ``1 / sum(w)`` however tight the pairs stand). The two are combined by inverse
    variance, which is what makes a fit from one spot fall back on the headings by itself instead
    of by a rule.
    """
    total = float(position_weight.sum())
    cart_centre = (position_weight[:, None] * cart).sum(0) / total
    place_centre = (position_weight[:, None] * place).sum(0) / total
    cart_local, place_local = cart - cart_centre, place - place_centre
    turns = place_local[:, 0] * cart_local[:, 1] - place_local[:, 1] * cart_local[:, 0]
    aligns = place_local[:, 0] * cart_local[:, 0] + place_local[:, 1] * cart_local[:, 1]
    spread = float((position_weight * (place_local**2).sum(1)).sum())
    geometry_yaw = math.atan2(
        float((position_weight * turns).sum()), float((position_weight * aligns).sum())
    )
    geometry_var = 1.0 / spread if spread > 0.0 else math.inf

    offsets = _wrap(cart_yaw - place_yaw)
    heading_yaw = math.atan2(
        float((heading_weight * np.sin(offsets)).sum()),
        float((heading_weight * np.cos(offsets)).sum()),
    )
    heading_total = float(heading_weight.sum())
    heading_var = 1.0 / heading_total if heading_total > 0.0 else math.inf

    if math.isinf(geometry_var) and math.isinf(heading_var):
        yaw, yaw_var = heading_yaw, math.inf
    else:
        information = (0.0 if math.isinf(geometry_var) else 1.0 / geometry_var) + (
            0.0 if math.isinf(heading_var) else 1.0 / heading_var
        )
        turn = wrap_angle(geometry_yaw - heading_yaw)
        weight = 0.0 if math.isinf(geometry_var) else (1.0 / geometry_var) / information
        yaw = wrap_angle(heading_yaw + weight * turn)
        yaw_var = 1.0 / information

    cos, sin = math.cos(yaw), math.sin(yaw)
    tx = cart_centre[0] - (cos * place_centre[0] - sin * place_centre[1])
    ty = cart_centre[1] - (sin * place_centre[0] + cos * place_centre[1])
    return (
        Pose2D(float(tx), float(ty), yaw),
        (float(place_centre[0]), float(place_centre[1])),
        1.0 / total if total > 0.0 else math.inf,
        yaw_var,
        math.degrees(wrap_angle(geometry_yaw - heading_yaw)),
    )


def fit_tie(pairs: list[TiePair], gate: float = GATE) -> Tie | None:
    """The tie of a whole log: one rigid ``map <- graph`` fitted over every pair, robust to the
    pairs a false recognition put there, with the covariance the data actually support.

    ``None`` for an empty log. Otherwise:

    1. every pair is tried as a MINIMAL SAMPLE — one pair with a heading already determines a
       rigid transform — and the hypothesis with the most pairs inside ``gate`` wins. This is
       RANSAC with the randomness taken out: the same log answers the same thing every time, and
       a structured minority (a session of the database sitting in another frame) cannot drag the
       answer the way a plain least-squares start would;
    2. the winner is refined by iteratively reweighted least squares, the weight of a pair being
       its own inverse variance INSIDE the gate and zero outside it — the same rule
       :func:`pepin.fusion.fuse` applies to a measurement it does not believe, for the same
       reason: past the chi-square, a pair is not a noisy measurement of this transform, it is a
       measurement of another one, and softening its pull instead of dropping it would let a
       session of the database that sits 1.6 m away still bend the answer;
    3. the covariance is the formal one of that weighted fit, inflated by ``sqrt(chi2 / dof)``
       when the inliers disagree with a rigid model by more than their own error bars allow —
       the Birge ratio, which is what turns a database whose sessions sit in different frames
       into a WIDE tie instead of a confident wrong one.

    Everything a report line needs rides on the answer: how many pairs, how many survived, how far
    apart they stand, the residual rms, and how far the geometry's rotation is from the one the
    pairs' own headings say (the consistency check — a big gap is a graph that bends).
    """
    if not pairs:
        return None
    cart, cart_yaw, place, place_yaw, sigma_xy, sigma_yaw = _arrays(pairs)
    count = len(pairs)

    stride = max(1, count // HYPOTHESIS_BUDGET)
    best: tuple[int, float, Pose2D] = (-1, math.inf, pairs[0].implied())
    for index in range(0, count, stride):
        candidate = pairs[index].implied()
        position, heading = _residuals(candidate, cart, cart_yaw, place, place_yaw)
        distance = _mahalanobis(position, heading, sigma_xy, sigma_yaw)
        inside = distance <= gate
        score = (int(inside.sum()), float(distance[inside].sum()) if inside.any() else math.inf)
        if score[0] > best[0] or (score[0] == best[0] and score[1] < best[1]):
            best = (score[0], score[1], candidate)
    tie = best[2]

    base_position = 1.0 / sigma_xy**2
    base_heading = 1.0 / sigma_yaw**2

    def explained(candidate: Pose2D) -> NDArray[np.bool_]:
        """Which pairs this candidate tie explains inside the gate."""
        offset, turn = _residuals(candidate, cart, cart_yaw, place, place_yaw)
        return np.asarray(_mahalanobis(offset, turn, sigma_xy, sigma_yaw) <= gate)

    def refit(mask: NDArray[np.bool_]) -> tuple[Pose2D, tuple[float, float], float, float, float]:
        """The weighted least-squares tie over the pairs ``mask`` keeps, and nothing else."""
        weight = np.where(mask, 1.0, 0.0)
        return _weighted_fit(
            cart, cart_yaw, place, place_yaw, base_position * weight, base_heading * weight
        )

    # The hypothesis explains at least the pair it was made from, so the first mask is never empty
    # and the first fit is always defined. From there the set is re-decided and re-fitted until it
    # stops changing; a set that collapses to nothing is a log too contradictory to refine, and the
    # tie in hand is kept rather than replaced by the mean of two poses it does not explain.
    kept = explained(tie)
    fitted = refit(kept)
    for _ in range(REFINE_PASSES - 1):
        again = explained(fitted[0])
        if not again.any() or bool((again == kept).all()):
            break
        kept, fitted = again, refit(again)
    tie, centroid, translation_var, heading_var, gap_deg = fitted
    position, heading = _residuals(tie, cart, cart_yaw, place, place_yaw)
    squared = _mahalanobis(position, heading, sigma_xy, sigma_yaw)
    inside = squared <= gate
    inliers = int(inside.sum())

    # The misfit of the pairs the tie actually stands on, against their own error bars: one when
    # the rigid model explains them, more when the database does not hold still.
    degrees_of_freedom = 3 * inliers - 3
    chi_square = float(squared[inside].sum()) if inliers else 0.0
    scale = (
        math.sqrt(chi_square / degrees_of_freedom)
        if degrees_of_freedom > 0 and chi_square > degrees_of_freedom
        else 1.0
    )
    covariance = np.diag(
        [
            translation_var * scale**2,
            translation_var * scale**2,
            heading_var * scale**2,
        ]
    )
    local = place - np.asarray(centroid)
    extent = float(np.sqrt((local**2).sum(1)).max()) if count else 0.0
    residual = position[inside] if inliers else position
    rms = float(np.sqrt((residual**2).sum(1).mean())) if len(residual) else 0.0
    return Tie(
        pose=tie,
        covariance=np.asarray(covariance, dtype=np.float64),
        centroid=centroid,
        origin="pairs",
        pairs=count,
        inliers=inliers,
        extent_m=extent,
        residual_rms_m=rms,
        heading_gap_deg=gap_deg,
    )
