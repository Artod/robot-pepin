"""A second opinion on where the cart is, computed where there is a CPU to compute it.

The tracker on the board localises in a window around the odometry's prediction (+-9 cm,
+-9 degrees, about 50 ms a scan) and, when the fit stays low for several scans, falls back to an
exhaustive whole-map search: FFT correlation over every shift at 40 headings, seconds on a
Cortex-A53. So the board can only ask "where am I, really?" after it has already admitted it is
lost, and then it pays for the question with seconds of a core it also drives with.

The laptop runs the same search in a tenth of a second. This module is the pure half of running
it there CONTINUOUSLY, as a watchdog: once a second the laptop searches the whole map on the
newest scan and publishes what it found — a place, a covariance read off the correlation peak,
the fit at that place and how ambiguous the map's answer was
(:class:`GlobalCandidate`). :func:`judge` says what one such candidate is worth beside the pose
the tracker holds:

* ``agree`` — the same place: the tracker is right, nothing to do (this is the normal verdict,
  once a second, forever, and its count is how one knows the watchdog is alive).
* ``disagree`` — a different place, explaining the scan clearly better, and the map answers with
  one place only. Three of those in a row, each from a DIFFERENT scan and all agreeing WITH EACH
  OTHER, re-seed the tracker through the path its own search uses (:class:`CandidateGate`).
* ``unknown_map`` — the best place on the map fits nothing, or the map answers with two places
  alike. Then the scan is not a scan of this map: another room, another flat, a map from before
  the furniture moved. Counted and said out loud; no automatic mode switch — that is the
  owner's decision, and a robot that changes maps on its own is worse than a lost one.
* ``nothing`` — a different place that does NOT explain the scan better. The candidate claims
  nothing, so neither does this module.

The board's own slow search stays exactly as it was: the fallback for when no candidate arrives
(the laptop is off, the link is down, the watch is switched off).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray

from pepin.fusion import Matrix, PoseMeasurement, from_fit, fuse
from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import apply_motion, relative_motion
from pepin.sources import TRACKER, WATCHDOG
from pepin.watch import ADMIT_FIT, ADMIT_MARGIN, AGREE_DEG, AGREE_M

__all__ = [
    "CandidateGate",
    "CandidateVerdict",
    "GateAnswer",
    "GlobalCandidate",
    "OdomTrail",
    "ambiguity",
    "carried",
    "judge",
    "same_place",
]

# A candidate this close to the tracked pose and this well aligned with it says the same place.
# The tracker's window walks a residual in at 9 cm a scan, so half a metre is something it
# corrects by itself within a second of driving — there is nothing here to re-seed. The same
# pair the board's own two-search rule calls agreement (pepin.watch.AGREE_M / AGREE_DEG), so
# "the two searches agree" and "the candidate agrees with the tracker" mean one thing.
SAME_PLACE_M = AGREE_M
SAME_PLACE_DEG = AGREE_DEG
# How much better than the tracker's own fit a candidate must explain the scan before its
# disagreement is evidence of anything. The margin a whole-map answer must beat the tracker by
# on the board (pepin.watch.ADMIT_MARGIN): a candidate is never cheaper evidence than a search.
BEAT_MARGIN = ADMIT_MARGIN
# Below this fit the best place on the whole map explains nothing: not "the cart is elsewhere"
# but "this is not a scan of this map". The board's floor for admitting a search's answer at
# all (pepin.watch.ADMIT_FIT); this flat's true pose scores 0.5-0.65 on its own maps.
UNKNOWN_MAP_FIT = ADMIT_FIT
# The second-best place's fit over the best's, counting only places that are a DIFFERENT place
# (farther apart than the pair below). Above this the map answers "here — or just as well
# there": a symmetric room, a corridor, a map the scan does not belong to. Such a candidate is
# never acted on. 0.90 is the margin within which the whole-map search itself calls two places
# twins (pepin.localization.TWIN_MARGIN, 1 - 0.10), read on the very measure that search ranks
# by — so "ambiguous" here and "twins" there are one judgement made in two places.
AMBIGUITY_MAX = 0.90
AMBIGUITY_APART_M = 0.5
AMBIGUITY_APART_DEG = 30.0
# Candidates from DIFFERENT scans that disagree with the tracker and agree with each other, in a
# row, before the tracker is re-seeded. One search a second on a standing cart is one second of
# evidence per candidate — one scan's answer repeated is not two of them (``distinct_scans``); a
# look-alike keeps looking alike, so the streak is not proof — it is the price of a teleport,
# and it is what keeps a single unlucky search (a person filling half the fan, a door that
# opened) from moving a healthy tracker.
CANDIDATE_STREAK = 3
# ...and unknown_map verdicts in a row before the tracker says the map does not fit. The same
# length for the same reason: one is a blocked lidar, three is a different room.
UNKNOWN_STREAK = 3


class OdomTrail(Protocol):
    """Where the cart was, in the odom frame, at any moment the tracker still remembers — the
    little of :class:`pepin.timeline.OdomHistory` a candidate's carry needs, so the gate can be
    driven by a fake in a test and by any robot's own trail."""

    def at(self, t: float) -> Pose2D | None:
        """The odometry pose at ``t``, or ``None`` when ``t`` is outside what is remembered."""

    @property
    def newest(self) -> Pose2D | None:
        """The newest odometry pose, or ``None`` when nothing has been recorded yet."""

    @property
    def newest_t(self) -> float | None:
        """When that newest pose was recorded, or ``None`` when there is none."""


class CandidateVerdict(StrEnum):
    """What one whole-map candidate is worth beside the pose the tracker holds."""

    AGREE = "agree"  # the same place: the tracker is right
    DISAGREE = "disagree"  # another place, clearly better, and the map is sure of it
    UNKNOWN_MAP = "unknown_map"  # nothing on this map fits, or two places fit alike
    NOTHING = "nothing"  # another place, but no better than what the tracker already has


def same_place(a: Pose2D, b: Pose2D) -> bool:
    """Whether two poses are the same place, to :data:`SAME_PLACE_M` and
    :data:`SAME_PLACE_DEG`."""
    return math.hypot(a.x - b.x, a.y - b.y) <= SAME_PLACE_M and abs(
        wrap_angle(a.theta - b.theta)
    ) <= math.radians(SAME_PLACE_DEG)


def ambiguity(places: Sequence[tuple[Pose2D, float]]) -> float:
    """How alike the map's runner-up explains the scan: the best score among places that are a
    DIFFERENT place from the first one, over the first one's score.

    ``places`` is the whole-map search's answer, best first, as (pose, score), where the score
    is the measure the search itself ranks places by (:meth:`pepin.localization.Localizer.rank`:
    how exactly the scan sits on the walls, minus a charge for what the map denies). Not the
    inlier fraction: that saturates one cell off a wall, and on a plain rectangular room it
    calls two places a metre apart 0.89 and 0.85 — a twin, which they are not.

    0.0 when the map holds one place that fits this scan and nothing else (the happy case);
    1.0 when the best place scores nothing at all, because then every rival is its equal.
    """
    if not places:
        return 1.0
    best_pose, best_score = places[0]
    if best_score <= 0.0:
        return 1.0
    rivals = [
        score
        for pose, score in places[1:]
        if math.hypot(pose.x - best_pose.x, pose.y - best_pose.y) > AMBIGUITY_APART_M
        or abs(wrap_angle(pose.theta - best_pose.theta)) > math.radians(AMBIGUITY_APART_DEG)
    ]
    return max([*rivals, 0.0]) / best_score


@dataclass(frozen=True)
class GlobalCandidate:
    """One whole-map search's answer, ready to travel: where the cart is (map frame), how sure
    that is per direction (a 3x3 covariance over x, y, yaw read off the correlation peak's own
    shape), the fit at the peak, how ambiguous the map's answer was (:func:`ambiguity`), the
    stamp and identity of the scan it was computed on and the map it was computed against.

    ``scan_id`` identifies the REVOLUTION, not the message: two candidates carrying one scan id
    are one search's answer said twice, and a streak built of them is one scan's evidence
    repeated (:meth:`CandidateGate.observe`). 0 means the sender did not say, and is read as
    "not a new scan" — the conservative half of the same rule.
    """

    x: float
    y: float
    yaw: float
    covariance: Matrix
    score: float  # the inlier fraction at the peak: how well the scan fits the map there
    ambiguity: float
    stamp: float
    map_id: str
    scan_id: int = 0

    @property
    def pose(self) -> Pose2D:
        """The place the search found."""
        return Pose2D(self.x, self.y, self.yaw)

    def measurement(self) -> PoseMeasurement:
        """This candidate as a measurement for :func:`pepin.fusion.fuse`: the same shape any
        scan source's match has, named :data:`pepin.sources.WATCHDOG`."""
        return PoseMeasurement(
            self.x, self.y, self.yaw, self.covariance, WATCHDOG, self.stamp, self.score
        )

    def to_json(self, **extra: Any) -> str:
        """The candidate as one JSON message — everything a tracker needs to judge it, so a
        reader never has to join two topics. ``extra`` adds the sender's own notes (the verdict
        it reached, what the search cost) for the operator; a reader ignores what it does not
        know."""
        return json.dumps(
            {
                "x": round(self.x, 4),
                "y": round(self.y, 4),
                "yaw": round(self.yaw, 5),
                "covariance": [[round(float(v), 8) for v in row] for row in self.covariance],
                "score": round(self.score, 4),
                "ambiguity": round(self.ambiguity, 4),
                "stamp": self.stamp,
                "scan": self.scan_id,
                "map": self.map_id,
                **extra,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> GlobalCandidate:
        """A candidate back from :meth:`to_json`; ``ValueError``, ``KeyError`` or ``TypeError``
        for anything else, which is what a subscriber counts as a malformed message."""
        raw = json.loads(text)
        covariance: NDArray[np.float64] = np.asarray(raw["covariance"], dtype=float)
        if covariance.shape != (3, 3):
            raise ValueError(f"a candidate's covariance is 3x3, not {covariance.shape}")
        return cls(
            x=float(raw["x"]),
            y=float(raw["y"]),
            yaw=float(raw["yaw"]),
            covariance=covariance,
            score=float(raw["score"]),
            ambiguity=float(raw["ambiguity"]),
            stamp=float(raw["stamp"]),
            map_id=str(raw["map"]),
            # A sender that does not name its scan says 0, and every such candidate then looks
            # like a replay of the previous one: a version skew loses the re-seeds and says so
            # in the report line, instead of building a streak out of one scan.
            scan_id=int(raw.get("scan", 0)),
        )

    def text(self) -> str:
        """``(x, y, yaw deg) fit 0.63, ambiguity 0.21`` for a report line."""
        return (
            f"({self.x:+.2f}, {self.y:+.2f}, {math.degrees(self.yaw):+.0f} deg) "
            f"fit {self.score:.2f}, ambiguity {self.ambiguity:.2f}"
        )


def carried(candidate: GlobalCandidate, motion: Pose2D, stamp: float) -> GlobalCandidate:
    """The same answer moved from the moment of its own scan to ``stamp``: where the cart is NOW
    if the search was right about where it was THEN.

    ``motion`` is the odometry's step between the two moments, in the base frame of the
    candidate's moment (:func:`pepin.scanmatch.relative_motion` over the tracker's odometry
    history) — the very carry a riding scan gets on its way to the anchor's instant
    (:meth:`pepin.sources.SourceFeed.gather`). A search costs 0.12-0.25 s and the link another
    hop, so an uncarried candidate is a pose of a quarter-second ago installed as the pose now:
    at 0.3 m/s that is 7 cm, and it is a bias, never noise — it always points backwards along
    the drive.

    The covariance travels through the composition's Jacobian, ``J = [[1, 0, -dy], [0, 1, dx],
    [0, 0, 1]]`` over the carry's map-frame displacement: a heading the search knew to a degree
    is 2 mm of position error after 10 cm of carry, and that coupling is the only thing the
    move adds. The odometry's OWN error over a fraction of a second (millimetres) is not added:
    it is two orders under the peak's own spread. The score and the ambiguity are the scan's
    own and do not change — the answer is the same answer, read at a later moment.
    """
    pose = apply_motion(candidate.pose, motion)
    dx, dy = pose.x - candidate.x, pose.y - candidate.y
    jacobian = np.array([[1.0, 0.0, -dy], [0.0, 1.0, dx], [0.0, 0.0, 1.0]])
    covariance = jacobian @ np.asarray(candidate.covariance, dtype=float) @ jacobian.T
    return replace(
        candidate, x=pose.x, y=pose.y, yaw=pose.theta, covariance=covariance, stamp=stamp
    )


def judge(candidate: GlobalCandidate, current_pose: Pose2D, current_fit: float) -> CandidateVerdict:
    """What ``candidate`` is worth beside the pose the tracker holds and the fit it holds it at.

    ``current_fit`` NaN — the tracker has not matched a scan yet — reads as 0.0, the convention
    of :meth:`pepin.watch.LostWatch.reported_fit`: every comparison here is a ``<`` or a ``>=``,
    and NaN passes them all silently.

    In order, because the order is the argument:

    1. The map's own answer first. A best place below :data:`UNKNOWN_MAP_FIT`, or a runner-up
       explaining the scan within :data:`AMBIGUITY_MAX` of it from somewhere else, means the
       search could not say where the cart is — ``unknown_map``. Judging such an answer against
       the tracker would be reading tea leaves.
    2. Then the cheap case: within :data:`SAME_PLACE_M` / :data:`SAME_PLACE_DEG` of the tracked
       pose the candidate confirms it — ``agree``.
    3. Elsewhere, it must also explain the scan better than the tracker's own fit by
       :data:`BEAT_MARGIN` to be ``disagree``; otherwise it is a worse explanation of the same
       scan from farther away, and claims ``nothing``.
    """
    if current_fit != current_fit:  # NaN: the tracker has not matched a scan yet
        current_fit = 0.0
    if candidate.score < UNKNOWN_MAP_FIT or candidate.ambiguity > AMBIGUITY_MAX:
        return CandidateVerdict.UNKNOWN_MAP
    if same_place(candidate.pose, current_pose):
        return CandidateVerdict.AGREE
    if candidate.score >= current_fit + BEAT_MARGIN:
        return CandidateVerdict.DISAGREE
    return CandidateVerdict.NOTHING


@dataclass(frozen=True)
class GateAnswer:
    """What the tracker does with one candidate: the verdict, and — only when a streak closed —
    the pose to re-seed from, as the measurement it was fused into."""

    verdict: CandidateVerdict
    seed: PoseMeasurement | None = None

    def pending(self, map_id: str) -> tuple[str, Pose2D, float] | None:
        """The re-seed as the tracker node's pending-seed triple (map, pose, fit), or ``None``
        when this candidate changes nothing."""
        return None if self.seed is None else (map_id, self.seed.pose, self.seed.fit)


@dataclass
class CandidateGate:
    """The tracker's side of the watchdog: judges every candidate that arrives, counts the
    verdicts, and says once — after :attr:`candidate_streak` disagreements about ONE place — to
    re-seed from it.

    Pure: the node hands it the candidate, the pose it holds, the fit it holds it at and whether
    a re-seed is allowed at all; it answers, and a report line reads its counters. The re-seed
    pose is not the candidate itself but the two fused by their information
    (:func:`pepin.fusion.fuse`, no gate — the disagreement is the point): a candidate sure to a
    centimetre against a tracker lost at fit 0.2 carries the fusion by two hundred to one and
    the seed IS the candidate, while a candidate the window bounded (a covariance widened a
    hundredfold) barely moves a healthy tracker. One rule instead of two.
    """

    accept_candidates: bool = True  # off: candidates are judged and counted, never acted on
    candidate_streak: int = CANDIDATE_STREAK
    distinct_scans: bool = True  # off: a streak may be built out of one scan's answer repeated
    carry_candidates: bool = True  # off: a candidate's pose is read as the pose now, uncarried
    unknown_streak: int = UNKNOWN_STREAK
    map_fits: bool = True  # False once unknown_streak candidates in a row said it does not
    last: CandidateVerdict | None = None  # the newest verdict, for the status
    _run: list[GlobalCandidate] = field(default_factory=list, init=False)  # the disagreeing ones
    _unknown_run: int = field(default=0, init=False)
    _counts: dict[str, int] = field(default_factory=dict, init=False)
    _reseeds: int = field(default=0, init=False)
    _malformed: int = field(default=0, init=False)
    _reason: str = field(default="", init=False)  # the last malformed message's complaint

    switches: ClassVar[tuple[str, ...]] = (
        "accept_candidates",
        "candidate_streak",
        "distinct_scans",
        "carry_candidates",
    )

    def switch(self, name: str, value: Any) -> None:
        """A live flag by its name (:attr:`switches`); ``ValueError`` for any other name."""
        if name not in self.switches:
            raise ValueError(f"{name}: not a switch of the candidate gate")
        setattr(self, name, int(value) if name == "candidate_streak" else bool(value))

    def malformed(self, reason: str) -> None:
        """A message that was not a candidate arrived; counted, with the complaint kept."""
        self._malformed += 1
        self._reason = reason

    def stale(self) -> None:
        """A candidate whose moment the odometry trail no longer covers, so it cannot be
        carried to now: counted, and the streak ends — a gap in the odometry is a gap in the
        evidence, not agreement."""
        self._count("stale")
        self._run = []

    def carry(self, candidate: GlobalCandidate, odometry: OdomTrail) -> GlobalCandidate | None:
        """``candidate`` moved from the moment of its own scan to the newest odometry sample —
        the moment a tracked pose read now speaks for — or ``None`` (counted as ``stale``) when
        ``odometry`` no longer covers the candidate's moment. ``carry_candidates`` off hands
        the candidate back untouched, which is to read a pose of a quarter-second ago as the
        pose now."""
        if not self.carry_candidates:
            return candidate
        then, newest, newest_t = odometry.at(candidate.stamp), odometry.newest, odometry.newest_t
        if then is None or newest is None or newest_t is None:
            self.stale()
            return None
        return carried(candidate, relative_motion(then, newest), newest_t)

    def observe(
        self,
        candidate: GlobalCandidate,
        current_pose: Pose2D | None,
        current_fit: float,
        map_id: str,
        allow: bool = True,
        odometry: OdomTrail | None = None,
    ) -> GateAnswer:
        """One candidate against the tracker's ``current_pose`` (``None`` before the first fix)
        and ``current_fit``, on the map ``map_id``.

        ``odometry`` is the tracker's own odometry trail: given, the candidate is first carried
        from the moment of its scan to the moment ``current_pose`` speaks for
        (:meth:`carry`) — a search plus a wireless hop is a quarter of a second, and the two
        poses must describe ONE instant or the comparison is between two different nows — and a
        candidate the trail cannot reach is counted and refused. Omitted, the caller has already
        put the two in one moment.

        A candidate computed on another map is evidence about nothing here and breaks every
        streak (the laptop may be a map behind after a swap). ``allow`` is the node's word on
        whether a re-seed is possible at all right now — a goal is running, say; the candidate
        is still judged and counted, so the report tells the truth either way.

        A streak is three SCANS, not three messages: a candidate carrying a scan id already in
        the run is the same second opinion said twice (a frozen ``/scan`` on the laptop, a
        message delivered twice) and is counted as ``replay`` without lengthening the run — the
        rule :meth:`pepin.watch.LostWatch.answer` has for the board's own two searches, after a
        frozen scan rubber-stamped every candidate there on 2026-09-09. ``distinct_scans`` off
        is the old behaviour.
        """
        if candidate.map_id != map_id:
            self._count("elsewhere")
            self.forget()
            return GateAnswer(CandidateVerdict.NOTHING)
        if current_pose is None:
            self._count("no_pose")
            return GateAnswer(CandidateVerdict.NOTHING)
        if odometry is not None:
            moved = self.carry(candidate, odometry)
            if moved is None:
                return GateAnswer(CandidateVerdict.NOTHING)
            candidate = moved
        if current_fit != current_fit:  # NaN: the tracker has not matched a scan yet
            current_fit = 0.0
        verdict = judge(candidate, current_pose, current_fit)
        self._count(str(verdict))
        self.last = verdict
        if verdict is CandidateVerdict.UNKNOWN_MAP:
            self._run = []
            self._unknown_run += 1
            self.map_fits = self.map_fits and self._unknown_run < self.unknown_streak
            return GateAnswer(verdict)
        self._unknown_run = 0
        if verdict is not CandidateVerdict.DISAGREE:
            self.forget()  # a candidate that confirms, or claims nothing, ends any run
            return GateAnswer(verdict)
        if self._run and not same_place(self._run[-1].pose, candidate.pose):
            self._run = []  # disagreeing about a DIFFERENT place each time proves nothing
        if self.distinct_scans and any(held.scan_id == candidate.scan_id for held in self._run):
            self._count("replay")  # one scan's answer heard twice: no second opinion in it
            return GateAnswer(verdict)
        self._run.append(candidate)
        if len(self._run) < self.candidate_streak or not (self.accept_candidates and allow):
            return GateAnswer(verdict)
        self._run = []
        self._reseeds += 1
        held = from_fit(current_pose, current_fit, TRACKER, candidate.stamp)
        seed = fuse([held, candidate.measurement()], gate=math.inf)
        return GateAnswer(verdict, seed)

    def status(self) -> dict[str, Any]:
        """The gate as the operator sees it on ``/localization/sources``: the newest verdict,
        whether the map still fits, how many candidates each verdict has taken since the node
        started and how many re-seeds came of them."""
        return {
            "verdict": None if self.last is None else str(self.last),
            "map_fits": self.map_fits,
            "reseeds": self._reseeds,
            "accept": self.accept_candidates,
        }

    def report(self) -> str:
        """One phrase for the tracker's report line; the counters are reset, the latched
        ``map does not fit`` is not (only a candidate that agrees clears it)."""
        counts, reseeds, malformed = self._counts, self._reseeds, self._malformed
        seen = sum(counts.values())
        body = ", ".join(f"{name} {n}" for name, n in sorted(counts.items())) or "none"
        self._counts, self._reseeds, self._malformed = {}, 0, 0
        return (
            f"candidates {seen} ({body}), re-seeds {reseeds}, "
            f"malformed {malformed}{f' ({self._reason})' if self._reason else ''}"
            + ("" if self.map_fits else "; THE MAP DOES NOT FIT what the lidar sees")
        )

    def _count(self, name: str) -> None:
        self._counts[name] = self._counts.get(name, 0) + 1

    def forget(self) -> None:
        """Every streak ends and the map is given the benefit of the doubt again: what a
        candidate that confirms the tracker does, and what a new map means for the ones held."""
        self._run, self._unknown_run, self.map_fits = [], 0, True
