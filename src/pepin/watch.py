"""When to stop trusting the tracked pose and search the whole map for the robot.

The decision is pure: a fit or a sigma, whether the robot is moving, whether a goal is running,
and the clock. Keeping it out of the ROS node is what makes it testable — the node only
supplies the readings and acts on the answer.

:class:`GoalGate` is the same question one step earlier — may a drive START — and it is the one
rule that also has to answer where no tracker exists at all (online SLAM): there the evidence is
the age of ``map -> base_link`` and of the SLAM correction (:class:`Correction`), not a fit.

:class:`SourceSilence` is the question under both of those: is there still a sensor speaking at
all? A fit measured minutes ago is not a fit, and the tracker published one for 141 s while it
drove on dead reckoning alone (2026-09-14).

:class:`Sigma` is the answer the whole file now prefers to a fit: one uncertainty, out of the
fusion, whatever spoke into it (:data:`DRIVE_SIGMA_M`).

:class:`JumpClear` watches the other side of a correction: when a word moves the pose far
enough, the obstacle grid built at the old pose has to be thrown away.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

import numpy as np

from pepin.fusion import Matrix, carry_pose, odometry_covariance, sigma_from_fit
from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import relative_motion

# The one fit scale everything reads. All are inlier fractions of a scan against the map, and
# their order is the design: a drive is stopped below BLIND, an unconfirmed fix may not report
# more than the CAP, a drive may start from DRIVE, and standing still below LOST for a while asks
# the whole map. Pinned by a test, because six of these once lived in three files, unordered.
BLIND_FIT = 0.30
PROVISIONAL_FIT_CAP = 0.35
DRIVE_FIT = 0.50
LOST_FIT = 0.55
ADMIT_FIT = 0.45  # a whole-map answer below this is no answer
ADMIT_MARGIN = 0.10  # ...and it must beat what the tracker already has by this much

# -- the one uncertainty: the fusion's own sigma --------------------------------------------
# A fit is ONE SENSOR'S metric — the share of one scan's beams that landed on the map — and it
# says nothing about the pose when that sensor is not the one holding it. On a camera-only
# drive there is no lidar scan to score, so /localization_fit is 0.00 by construction
# (the relocalizer's local_fit rule) or 0.11 from the graph's distance-decayed trust, and every
# rule built on that number read a healthy tracker as lost: on 2026-09-15 goto cancelled its own
# camera drives — "localization lost for 19 s while the wheels travelled 1.0 m" — while the
# tracker was following the camera's measurements perfectly well.
#
# The tracker fuses every source's word into an information filter, and the covariance that
# comes out of it IS the pose's certainty, whoever spoke: position sigma in metres, heading
# sigma in degrees, published on /localization/sigma at the tracker's rate and grown along the
# odometry between corrections (pepin_bringup.relocalizer.PoseSpread). These two numbers are
# the sigma ladder, and they are read against the cart itself:
#
#   the footprint is 0.55 m wide (ros/params/nav2_params.yaml: +-0.275 m about base_link)
#   Nav2's xy_goal_tolerance is 0.10 m
#
# DRIVE_SIGMA_M — a goal may START when the pose is known to better than this. Above Nav2's own
# arrival tolerance would make "arrived" a guess; at the half-width a planned gap is not a gap.
# LOST_SIGMA_M — a drive already running is CUT above this. Under the half-width plus a margin:
# past it the cart's own outline is uncertain by about half its width and a corridor plan is
# fiction.
# RAISED 2026-09-16, 0.15 -> 0.25 AND 0.25 -> 0.40, AND THE REASON IS NOT "GOALS WERE REFUSED".
# The ladder was written when the lidar's fused match was the only thing that ever held the pose,
# and it measures itself at 1-8 cm. The pose graph does not: on the tapes of 2026-09-16 its word
# sat 22-23 cm and 12 deg from the lidar's in motion, which is now what it CLAIMS as well
# (pepin.measurements.GRAPH_FLOOR_XY_M). An honest source whose honest sigma is 0.20 m can never
# pass a 0.15 m gate, so the old numbers did not refuse an uncertain pose — they refused the
# graph, and the only way to drive on it would have been to let it keep lying about itself. The
# thresholds move instead, and they move with the floor they now have to admit: 0.25 m to start
# (the graph's own claim plus a little) and 0.40 m to cut a drive already running.
# WHAT IS SPENT: 0.25 m is no longer under Nav2's 0.10 m xy_goal_tolerance, so "arrived" on a
# graph-held drive is arrived to a quarter of a metre, and the cut at 0.40 m sits ABOVE the
# 0.275 m half-width — a plan through a gap narrower than the cart plus 0.4 m is not to be
# trusted while the sigma is up there. Both stay under the 0.55 m footprint's full width, which
# is the line where the cart's outline stops meaning anything at all. A lidar drive is unaffected:
# it runs at 1-8 cm and never comes near either number.
# Their translation to the old scale, for anyone reading a tape: where nothing has measured the
# pose at all the spread falls back to the fit (:func:`pepin.fusion.sigma_from_fit`, 0.05 +
# 0.30 * (1 - fit) metres), and on THAT scale 0.25 m is fit 0.33 — a hundredth from the
# :data:`BLIND_FIT` 0.30 the cut used to be, so a drive on an unmeasured pose is now cut where
# the fit rung already cut it — while 0.40 m is off the end of the fit scale entirely, whose
# worst reading is 0.35 m at a fit of zero. That is why :data:`UNKNOWN_SIGMA` below no longer
# comes straight off the fit: the two scales stopped overlapping and the sentinel has to be
# held over the cut on purpose.
# The topic the tracker publishes them on, and the shape of what it publishes: a
# std_msgs/String carrying JSON — ``sigma_xy`` in metres, ``sigma_yaw`` in degrees, ``stamp``
# the tracker's clock when it was published and ``word_age_s`` the seconds since the last
# accepted word of any source. One text message rather than a typed one because every consumer
# of it already speaks this dialect (/localization/sources, /localization/measurement) and
# because pepin_bringup.depth_fusion reads exactly these names. The name and the shape live
# here, with the numbers read off them: this file is the contract.
SIGMA_TOPIC = "/localization/sigma"
# How long a window of that topic one certainty judgement reads (:class:`SigmaWindow`): a single
# sample is a single scan's luck, and in a nook it flickers 0.01 <-> 0.31 m between revolutions.
# Two seconds is about 20 publications at the tracker's check period.
SIGMA_MEDIAN_S = 2.0
DRIVE_SIGMA_M = 0.25
LOST_SIGMA_M = 0.40
# What the sigma reads before anything has corrected the pose at all. It is a SENTINEL and not a
# measurement: nothing has been measured, the cart may be anywhere the map is, and the only
# property that has to hold is that it fails every rung of the ladder above. The fit-of-zero
# spread (:func:`pepin.fusion.sigma_from_fit` — 0.35 m and 23 deg) said that well enough while
# the cut sat at 0.25 m; with the cut at 0.40 m since 2026-09-16 it no longer does, and a drive
# that had never heard a word would have run uncut. So the position is held over the cut
# explicitly rather than left to a coincidence between two scales that were never tied together.
# The heading keeps the fit's 23 deg, which is over every heading rung with room to spare.
UNKNOWN_SIGMA = (
    max(sigma_from_fit(0.0)[0], LOST_SIGMA_M + 0.05),
    math.degrees(sigma_from_fit(0.0)[1]),
)

# How far the pose graph's own word may sit from the tracker before a camera-only drive is
# refused: half of one 0.20 m planning cell — the graph recognising the room and agreeing to
# within a fifth of the cart's width is what replaces the lidar's scan-to-map fit as evidence
# that the room under the cart is the room on the map.
GRAPH_AGREE_M = 0.10

# Which rule judged, for the line the operator reads. Never guessed downstream: a stack whose
# board still runs a build without /localization/sigma is judged by the fit, and the print says
# so rather than leaving a reader to infer it from the number's size.
BY_SIGMA = "sigma"
BY_FIT = "fit"
BY_TF = "tf"

AGREE_M = 0.5  # two fixes this close...
AGREE_DEG = 30.0  # ...and this aligned are the same place
CONFIRM_TRIES = 4  # fresh searches allowed to agree before the fix is given up as a twin

# How old the map -> base_link edge may be and still be a pose to start a drive on, where no
# tracker publishes a fit. The edge is re-broadcast at 10 Hz (pepin_bringup.slam_frame) over
# odometry published at 50 Hz, so a whole second without one is ten missed broadcasts: the
# BROADCASTER or the odometry has stopped, not jittered. It says nothing about the machine that
# computes the correction — that is what :class:`Correction` is for. Nav2's tolerance is 0.3 s.
TF_FRESH_S = 1.0

# How long the SLAM correction may be silent before the half of the stack that owns the pose
# counts as gone. pepin_bringup.rtabmap_frame publishes it at 10 Hz whether or not the graph
# moved, so it is a pulse and not an event stream — but it crosses the bridge over WiFi, where
# the laptop's own heartbeat is given 2.5 s (pepin.deployment.LinkWatch). Twenty missed messages
# is not a hiccup.
CORRECTION_FRESH_S = 2.0

# How long every scan source and every remote measurement may be silent at once before the fit
# this machine publishes stops meaning anything. The lidar delivers at 10 Hz and the camera's
# measurements at 5 Hz per source, so three seconds is thirty missed revolutions, not a hiccup —
# and a cart standing still is NOT silent: its lidar keeps turning while the tracker's motion
# filter spares the matcher.
SOURCE_PATIENCE_S = 3.0


class Verdict(StrEnum):
    """What a whole-map answer is worth. APPLY is the only one that moves the tracker."""

    NOTHING = "nothing"  # no usable answer: not admitted, or no better than the tracker
    CANDIDATE = "candidate"  # a first answer, held; ask again on a fresh scan
    REPLAY = "replay"  # the same scan as the candidate: nothing learned
    HOLD = "hold"  # a second answer that disagrees: it is the candidate now
    APPLY = "apply"  # two searches agree: move, once
    GIVEN_UP = "given_up"  # they keep disagreeing: a twin the scan cannot settle


@dataclass(frozen=True)
class Answer:
    """A verdict and, when it is APPLY, the pose and confidence to adopt."""

    verdict: Verdict
    pose: Pose2D | None = None
    confidence: float = 0.0


@dataclass
class LostWatch:
    """Decides when a whole-map search is worth its seconds, and how long to wait after a failure.

    ``lost_fit`` and ``lost_checks`` set the ordinary trigger: the scan has fitted the map poorly
    for that many checks in a row while the robot stood still. A *collapse* — the fit was healthy
    and fell off a cliff, which is what being carried or turned by hand looks like — asks at once.
    Every failed search doubles the wait before the next one, capped, so a robot that cannot find
    itself does not saturate a board that is also driving.
    """

    lost_fit: float = LOST_FIT
    lost_checks: int = 3
    admit_fit: float = ADMIT_FIT
    admit_margin: float = ADMIT_MARGIN
    cooldown_s: float = 8.0
    max_backoff_s: float = 20.0
    collapse_from: float = 0.5  # a fit at least this good...
    collapse_to: float = 0.30  # ...falling below this is a carry, not noise
    # A whole-map search proposes; it never moves the tracker by itself. Its answer is held as a
    # CANDIDATE and applied only when a later search lands on the same place — one teleport per
    # episode, never a flip-flop. A flat is full of look-alikes: on 2026-09-09 a scan at the base
    # fitted the base at 0.71 and a corner four metres away at 0.69, the search run mid-carry
    # chose the corner and the tracker settled there at 0.72; then, with the first version of
    # this rule, every disagreeing answer was seeded and the map spun between the two.
    _candidate: Pose2D | None = field(default=None, init=False)
    _candidate_scan: int = field(default=-1, init=False)  # the scan the candidate was found on
    _confirm_tries: int = field(default=0, init=False)
    _streak: int = field(default=0, init=False)
    _last_fit: float = field(default=0.0, init=False)
    _failures: int = field(default=0, init=False)
    _quiet_until: float = field(default=0.0, init=False)

    @property
    def confirmed(self) -> bool:
        """False while a search's answer is waiting for a second search to agree."""
        return self._candidate is None

    def reported_fit(self, fit: float) -> float:
        """The fit the outside world may see: capped while a candidate is pending, so nothing
        downstream mistakes a lucky look-alike for a localised robot. NaN — no match yet — is
        reported as 0.0: every gate compares with ``<``, and NaN passes them all silently."""
        if fit != fit:  # NaN
            return 0.0
        return fit if self.confirmed else min(fit, PROVISIONAL_FIT_CAP)

    def wait_s(self) -> float:
        """How long to stay quiet after a search: the cooldown doubled per failure, capped."""
        return min(self.cooldown_s * 2.0 ** min(self._failures, 16), self.max_backoff_s)

    def observe(
        self, fit: float, moving: bool, navigating: bool, now: float, occluded: bool = False
    ) -> bool:
        """One check, once a second: True when the whole map should be searched right now.

        ``fit`` is the tracker's OWN fit, never :meth:`reported_fit`: the capped provisional
        number fed back here kept a candidate alive at fit 0.76 and searched the map every
        second for half an hour (2026-09-09 18:00). A moving robot and a robot under a goal are
        never re-seeded: their fit dips for honest reasons (a scan and a pose milliseconds
        apart, a correction in progress). ``occluded`` — a large share of the scan is things the
        map does not know (a person beside the cart) — explains a low fit without the pose being
        wrong: no search, no streak, and a pending candidate is not asked about either.
        """
        if fit != fit:  # NaN: no match yet
            fit = 0.0
        collapsed = self._last_fit >= self.collapse_from and fit < self.collapse_to
        self._last_fit = fit
        if moving or navigating or occluded:
            self._streak = 0
            return False
        if not self.confirmed:
            if fit >= self.lost_fit:
                # The tracker found its own feet while a candidate waited: a healthy lock is
                # better evidence than any one-scan search, so the candidate is dropped and the
                # tracker is never overridden.
                self._candidate, self._confirm_tries = None, 0
                return False
            return True  # a pending candidate is checked on the very next scan, cooldown or not
        if now < self._quiet_until:
            return False
        if collapsed:
            self._streak, self._failures = self.lost_checks, 0  # a new carry is a new question
        else:
            self._streak = self._streak + 1 if fit < self.lost_fit else 0
        if self._streak < self.lost_checks:
            return False
        self._streak = 0
        return True

    def searched(self, found: bool, now: float) -> None:
        """Record how a search ended; a failure lengthens the quiet period, a fix clears it."""
        self._failures = 0 if found else self._failures + 1
        self._quiet_until = now + self.wait_s()

    def seeded(self, now: float) -> None:
        """A pose was adopted (a hand, a restart, or an agreed candidate): give the tracker a
        moment before judging it again."""
        self._streak = 0
        self._candidate, self._confirm_tries = None, 0
        self._quiet_until = max(self._quiet_until, now + self.cooldown_s)

    def answer(
        self, pose: Pose2D | None, confidence: float, current_fit: float, scan: int, now: float
    ) -> Answer:
        """The one door for a whole-map search's result; every admission rule lives here.

        ``scan`` is the identity of the scan the search ran on. An answer is admitted only if
        it fits at least ``admit_fit`` and beats the tracker's own ``current_fit`` by
        ``admit_margin``; the first admitted answer is a CANDIDATE and moves nothing; a later
        search on a fresh scan that agrees is APPLY (the pose to adopt travels with it), one
        that disagrees becomes the candidate (HOLD), one on the candidate's own scan is a REPLAY;
        after ``CONFIRM_TRIES`` disagreements the question is GIVEN_UP. Two searches a second
        apart on a standing robot prove stability, not truth: a twin that fits alike keeps
        fitting alike, which is why the tracker's own recovery (``observe`` drops a candidate
        when the fit is healthy) and motion remain the stronger evidence.
        """
        admitted = (
            pose is not None
            and confidence >= self.admit_fit
            and confidence >= current_fit + self.admit_margin
        )
        if self._candidate is None:
            if not admitted:
                self._failures += 1
                self._quiet_until = now + self.wait_s()
                return Answer(Verdict.NOTHING)
            assert pose is not None
            self._failures = 0
            self._candidate, self._candidate_scan, self._confirm_tries = pose, scan, 0
            self._quiet_until = now  # ask again at once
            return Answer(Verdict.CANDIDATE, pose, confidence)
        if scan == self._candidate_scan:
            return Answer(Verdict.REPLAY)
        self._confirm_tries += 1
        if admitted and pose is not None and self.agrees(self._candidate, pose):
            self._candidate, self._confirm_tries = None, 0
            self._quiet_until = now + self.cooldown_s
            return Answer(Verdict.APPLY, pose, confidence)
        if self._confirm_tries >= CONFIRM_TRIES:
            self._candidate, self._confirm_tries = None, 0
            self._failures += 1
            self._quiet_until = now + self.wait_s()
            return Answer(Verdict.GIVEN_UP)
        if admitted and pose is not None:
            self._candidate, self._candidate_scan = pose, scan
        return Answer(Verdict.HOLD, pose, confidence)

    @staticmethod
    def agrees(a: Pose2D, b: Pose2D) -> bool:
        """Two poses that are the same place, to within a cell of carry and a bin of heading."""
        return math.hypot(a.x - b.x, a.y - b.y) <= AGREE_M and abs(
            wrap_angle(a.theta - b.theta)
        ) <= math.radians(AGREE_DEG)


@dataclass(frozen=True)
class Sigma:
    """The tracker's word on how sure the pose is, as it arrives on ``/localization/sigma``:
    the position sigma in metres, the heading sigma in degrees, and how many seconds ago it was
    published.

    ``None`` in place of one of these — never a made-up number — is what says the board
    publishes no such word at all (a build from before 2026-09-15), and the fit rules are the
    fallback there. A word that HAS arrived and then stopped is a different thing entirely: the
    tracker is dead or stalled, and :meth:`fresh` is what the gates ask before believing it.
    """

    xy_m: float
    yaw_deg: float
    age_s: float

    @classmethod
    def from_json(cls, text: str, age_s: float) -> Sigma | None:
        """One ``/localization/sigma`` message as the tracker writes it, read ``age_s`` seconds
        after it landed; ``None`` for anything that does not parse or does not carry the two
        numbers — a message nobody can read is not a reason to drive, and not a reason to
        raise in a subscription either."""
        try:
            heard = json.loads(text)
            return cls(float(heard["sigma_xy"]), float(heard["sigma_yaw"]), age_s)
        except (TypeError, ValueError, KeyError):
            return None

    def to_json(self, stamp: float, word_age_s: float) -> str:
        """The message the tracker publishes: the two numbers, the clock it was published at,
        and how long it is since a source's word last corrected the pose."""
        return json.dumps(
            {
                "sigma_xy": round(self.xy_m, 4),
                "sigma_yaw": round(self.yaw_deg, 3),
                "stamp": round(stamp, 3),
                "word_age_s": round(word_age_s, 3),
            }
        )

    def fresh(self, patience_s: float = SOURCE_PATIENCE_S) -> bool:
        """True while this word is young enough to stand for the pose right now. The tracker
        publishes at its own rate and at least once a check period (1 s) with nothing driving
        it, so the source patience is many missed publications, not a hiccup."""
        return self.age_s <= patience_s

    def phrase(self) -> str:
        """``0.06 m / 1.2 deg`` — what the operator reads beside a verdict."""
        return f"{self.xy_m:.2f} m / {self.yaw_deg:.1f} deg"


@dataclass
class SigmaWindow:
    """The last :data:`SIGMA_MEDIAN_S` of ``/localization/sigma`` read as ONE number: the median
    of the samples in the window, not the newest of them.

    One sample is not the certainty of the pose. In a nook the lidar's match flickers between two
    hypotheses from scan to scan and the published sigma with it — 0.01 m one revolution, 0.31 m
    the next (2026-09-16) — so a gate that reads whichever sample happened to be last either
    starts a drive on a pose that is about to be 0.31 m wrong or refuses one that is 0.01 m
    right, and which of the two it does is luck. The median over two seconds is the reading that
    describes the nook instead of the revolution: it takes about 20 tracker publications, and it
    moves only when MOST of them move, which is what a real loss of the pose looks like.

    Two seconds and not more: it is the delay a genuine loss now costs every gate built on it,
    and the drive watch's own patience is 4-15 s on top, so the total is unchanged in character.

    ``age_s`` on the answer is the NEWEST sample's, never the median's: freshness is the question
    "is the tracker still publishing", and the median must not make a dead tracker look alive.
    """

    window_s: float = SIGMA_MEDIAN_S
    _samples: list[tuple[float, Sigma]] = field(default_factory=list, init=False)

    def add(self, sigma: Sigma, now: float) -> None:
        """One message landed at ``now`` (this machine's monotonic clock)."""
        self._samples.append((now, sigma))
        self._trim(now)

    def median(self, now: float) -> Sigma | None:
        """The window's reading at ``now``, or ``None`` while no sample has ever arrived — which
        is what says this board publishes no sigma at all and the fit rules answer instead. With
        every sample older than the window the newest one is still answered with, aged: a tracker
        that has stopped is a reading the gates must SEE and refuse, not an absence they mistake
        for an old build."""
        self._trim(now)
        if not self._samples:
            return None
        newest_at = self._samples[-1][0]
        return Sigma(
            statistics.median(s.xy_m for _, s in self._samples),
            statistics.median(s.yaw_deg for _, s in self._samples),
            max(0.0, now - newest_at),
        )

    def span(self) -> float:
        """Seconds between the oldest and the newest sample now kept: how much of the window has
        actually filled. 0.0 with one sample or none — a median of one sample is that sample, and
        a caller that wants a window rather than a reading waits on this."""
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1][0] - self._samples[0][0]

    def clear(self) -> None:
        """Forget every sample. What a whole-map search leaves behind: the pose it seeded is not
        the pose the samples before it described, so blending the two describes neither."""
        self._samples.clear()

    def _trim(self, now: float) -> None:
        """Drop everything older than the window, keeping the newest sample whatever its age."""
        cutoff = now - self.window_s
        kept = [entry for entry in self._samples if entry[0] >= cutoff]
        self._samples = kept or self._samples[-1:]


@dataclass
class PoseSpread:
    """How sure the tracked pose is, carried between corrections: the fusion's own covariance at
    the last accepted word, grown along the odometry since.

    This is the tracker's answer to "do I know where I am", and the reason it exists beside the
    fit is that a fit belongs to ONE sensor. The lidar's inlier fraction is 0.00 on a camera-only
    drive because there is no lidar scan to score, and every rule built on it read a healthy
    tracker as lost (2026-09-15). The information filter already weighs every source's word by
    its covariance (:mod:`pepin.fusion`), so the covariance that comes out of an update IS the
    certainty of the pose, whoever spoke into it.

    Two steps, the filter's own:

    * :meth:`corrected` — an update landed. The covariance becomes the fused measurement's
      (:func:`pepin.fusion.published_covariance`, the same 3x3 published on /tracker_pose, so
      the sigma and the pose never tell two stories), anchored at the odometry of that moment.
      Without a fused measurement — a re-seed, a carried belief — it is what the fit buys.
    * :meth:`carried` — the prediction step, run between corrections: the covariance travels
      through the composition's Jacobian and is widened by what the odometry's own error over
      that step costs (:func:`pepin.fusion.odometry_covariance`, this cart's measured 2 % per
      metre and 0.7 of every reported turn). The tracker's filter has no process noise of its
      own — its correction is a blend, not a Kalman step — so this is the honest one: it is the
      same model the measurement carry already pays, applied to the belief itself. It is
      accumulated STEP BY STEP, so a cart that drives a circle back to where it started is not
      reported as certain: it is the path that costs, not the displacement.

    Before the first correction there is no covariance at all and the sigma is what a fit of
    zero buys (0.35 m, 23 deg): not localised, which is what a gate should read then.
    """

    covariance: Matrix | None = None
    pose: Pose2D | None = None  # the map pose the covariance belongs to (the Jacobian's arm)
    odom: Pose2D | None = None  # ...and the odometry reading of that same moment
    corrected_at_s: float | None = field(default=None)

    def corrected(self, covariance: Matrix, pose: Pose2D, odom: Pose2D | None, now: float) -> None:
        """An accepted word of any source: adopt its fused covariance and anchor it here."""
        self.covariance = np.asarray(covariance, dtype=np.float64)
        self.pose, self.odom, self.corrected_at_s = pose, odom, now

    def carried(self, odom: Pose2D | None) -> None:
        """The prediction step up to ``odom``: nothing corrected the pose, so it grew."""
        if self.covariance is None or self.pose is None or self.odom is None or odom is None:
            return
        motion = relative_motion(self.odom, odom)
        pose, covariance = carry_pose(self.pose, self.covariance, motion)
        self.pose, self.odom = pose, odom
        self.covariance = covariance + odometry_covariance(motion)

    def sigma(self) -> tuple[float, float]:
        """``(position metres, heading degrees)``: the position sigma is the root of the LARGEST
        eigenvalue of the covariance's 2x2 position block — the widest direction, never the
        average — so a pose pinned across a corridor and loose along it reads as loose."""
        if self.covariance is None:
            return UNKNOWN_SIGMA
        block = np.asarray(self.covariance, dtype=np.float64)
        widest = float(np.max(np.linalg.eigvalsh(block[:2, :2])))
        return math.sqrt(max(widest, 0.0)), math.degrees(math.sqrt(max(float(block[2, 2]), 0.0)))

    def age_s(self, now: float) -> float:
        """Seconds since the last accepted word; ``inf`` before the first one."""
        return math.inf if self.corrected_at_s is None else max(0.0, now - self.corrected_at_s)

    def text(self, now: float) -> str:
        """``sigma 0.06 m / 1.2 deg (word 0.1 s ago)`` for the tracker's report line."""
        xy, yaw = self.sigma()
        age = self.age_s(now)
        return f"sigma {xy:.2f} m / {yaw:.1f} deg " + (
            "(no word yet)" if age == math.inf else f"(word {age:.1f} s ago)"
        )


@dataclass(frozen=True)
class Readiness:
    """Whether a goal may start now, whether a tracker is there at all, and why not when it may
    not — the phrase the operator reads on a refusal, and which rule (:data:`BY_SIGMA`,
    :data:`BY_FIT`, :data:`BY_TF`) reached it."""

    ready: bool
    tracker: bool  # a tracker publishes a fit here: what arms the blind-drive watch
    search: bool = False  # the tracker is up but lost: one whole-map search is owed first
    reason: str = ""
    rule: str = ""


@dataclass(frozen=True)
class Correction:
    """The last SLAM correction to reach this machine: ``age_s`` seconds ago, or ``None`` when
    none ever has.

    It is the pulse of the half of the stack that owns the pose in online SLAM, and the only
    honest one. ``map -> odom`` is not: pepin_bringup.slam_frame re-broadcasts the LAST
    correction at 10 Hz with a fresh stamp for ever, so the edge — and every transform composed
    from it — stays milliseconds old with the laptop shut down.
    """

    age_s: float | None

    def stale(self, patience_s: float = CORRECTION_FRESH_S) -> bool:
        """True when the correction stopped arriving, or never did: the SLAM half is not here."""
        return self.age_s is None or self.age_s > patience_s

    def phrase(self) -> str:
        """What to say about it on a refusal: never heard, or silent for this long."""
        if self.age_s is None:
            return "no SLAM correction has ever arrived"
        return f"the SLAM correction stopped {self.age_s:.1f} s ago"


@dataclass(frozen=True)
class GoalGate:
    """May the cart be sent to a goal: on the fusion's sigma where the tracker publishes one, on
    its fit where it does not, on the age of ``map -> base_link`` and of the SLAM correction
    where no tracker runs at all.

    Two stacks, one question. On a saved map the scan-matching tracker publishes
    ``/localization_fit`` and a drive starts from ``drive_fit`` — under it the goal buys one
    whole-map search first. In online SLAM there is no tracker at all: RTAB-Map owns the pose on
    the laptop and the board only re-broadcasts its correction, so nobody ever publishes a fit
    and the gate that waited for one refused every goal ("the tracker is not up", 2026-09-13
    14:05). SLAM mode has two readings instead. The transform says the board's own half is
    running: ``map -> base_link`` younger than ``fresh_s``, else missing or stale and how stale.
    The :class:`Correction` says the laptop's half is — and it is the one that can die without
    the transform noticing, because the edge goes on being broadcast from the last correction.
    A caller that passes no correction is not watching one (a known-map stack, or the watch
    switched off). There is nothing to search with here, so a refusal is final until the half
    that stopped comes back.

    On a saved map the fit is now only the FALLBACK: where the tracker publishes its fused
    sigma (:class:`Sigma`), that is what the goal is judged on, because a fit belongs to the
    lidar and a drive may be held by the camera alone (:data:`DRIVE_SIGMA_M`).
    """

    drive_fit: float = DRIVE_FIT
    drive_sigma_m: float = DRIVE_SIGMA_M
    fresh_s: float = TF_FRESH_S
    correction_fresh_s: float = CORRECTION_FRESH_S
    sigma_patience_s: float = SOURCE_PATIENCE_S

    def verdict(
        self,
        fit: float | None,
        tf_age_s: float | None,
        correction: Correction | None = None,
        sigma: Sigma | None = None,
    ) -> Readiness:
        """One goal's answer. ``sigma`` is the tracker's fused uncertainty where it publishes
        one and outranks every other reading; ``fit`` is the tracker's, or ``None`` where no
        tracker speaks; ``tf_age_s`` is how many seconds ago ``map -> base_link`` was stamped
        (``None``: nothing publishes it); ``correction`` is the SLAM half's pulse where it is
        watched. Returns the :class:`Readiness` the caller acts on, with the rule that reached
        it named."""
        if sigma is not None:
            if not sigma.fresh(self.sigma_patience_s):
                return Readiness(
                    False,
                    tracker=True,
                    rule=BY_SIGMA,
                    reason=f"the tracker's sigma stopped {sigma.age_s:.1f} s ago: it publishes"
                    " one every check period whatever the sensors do, so this is the tracker"
                    " itself, not a quiet sensor",
                )
            if sigma.xy_m <= self.drive_sigma_m:
                return Readiness(True, tracker=True, rule=BY_SIGMA)
            return Readiness(
                False,
                tracker=True,
                search=True,
                rule=BY_SIGMA,
                reason=f"the pose is known to {sigma.phrase()}, over the {self.drive_sigma_m:.2f}"
                " m a drive starts from: stand still or relocalize",
            )
        if fit is not None:
            if fit >= self.drive_fit:
                return Readiness(True, tracker=True, rule=BY_FIT)
            return Readiness(
                False,
                tracker=True,
                search=True,
                rule=BY_FIT,
                reason=f"fit {fit:.2f} under {self.drive_fit:.2f}: stand still or relocalize",
            )
        if tf_age_s is None:
            return Readiness(
                False,
                tracker=False,
                rule=BY_TF,
                reason="no tracker, and nothing publishes map -> base_link: the SLAM half of the"
                " stack is not up",
            )
        if tf_age_s > self.fresh_s:
            return Readiness(
                False,
                tracker=False,
                rule=BY_TF,
                reason=f"no tracker, and map -> base_link is {tf_age_s:.1f} s old: a drive needs"
                f" it fresher than {self.fresh_s:.1f} s",
            )
        if correction is not None and correction.stale(self.correction_fresh_s):
            return Readiness(
                False,
                tracker=False,
                rule=BY_TF,
                reason=f"no tracker, and {correction.phrase()}: the half of the stack that owns"
                " the pose is not here. map -> base_link stays fresh either way — it is"
                " re-broadcast from the last correction",
            )
        return Readiness(True, tracker=False, rule=BY_TF)


@dataclass
class BlindDriveWatch:
    """Stops a drive whose tracker has lost its lock: the pose uncertain past ``lost_sigma_m``
    — or, where no sigma is published, the fit below ``lost_fit`` — for ``patience_s``.

    The tracker never re-searches while a goal runs (a teleport mid-drive is worse than a poor
    fit), so a drive that loses its lock keeps driving on a wrong map — run 0052 spent a minute
    at fit 0.05-0.29 and arrived drunk. Better to stop, search standing still, and go again.

    The sigma is what makes this watch honest on a camera-only drive: the fit it used to read is
    the lidar's alone and is 0.00 there by construction, so the watch cut healthy drives
    (2026-09-15). A sigma that has STOPPED arriving is itself a lost lock — the tracker publishes
    one every check period whatever the sensors do — and counts as lost, with the phrase saying
    which of the two it was.
    """

    lost_fit: float = 0.30
    lost_sigma_m: float = LOST_SIGMA_M
    patience_s: float = 4.0
    sigma_patience_s: float = SOURCE_PATIENCE_S
    rule: str = field(default=BY_FIT, init=False)  # which reading the last observe judged on
    _reading: str = field(default="", init=False)
    _lost_since: float | None = field(default=None, init=False)

    def observe(self, fit: float, now: float, sigma: Sigma | None = None) -> bool:
        """True the moment the pose has been uncertain for longer than the patience. ``sigma``
        is the tracker's fused uncertainty where it publishes one; without it the ``fit`` is
        judged, exactly as before."""
        if sigma is not None:
            self.rule = BY_SIGMA
            lost = not sigma.fresh(self.sigma_patience_s) or sigma.xy_m > self.lost_sigma_m
            self._reading = (
                f"the tracker's sigma stopped {sigma.age_s:.1f} s ago"
                if not sigma.fresh(self.sigma_patience_s)
                else f"sigma {sigma.phrase()} over {self.lost_sigma_m:.2f} m"
                if lost
                else f"sigma {sigma.phrase()}"
            )
        else:
            self.rule = BY_FIT
            lost = fit < self.lost_fit
            self._reading = f"fit {fit:.2f}" + (f" under {self.lost_fit:.2f}" if lost else "")
        if not lost:
            self._lost_since = None
            return False
        if self._lost_since is None:
            self._lost_since = now
        return now - self._lost_since > self.patience_s

    def phrase(self) -> str:
        """What the last :meth:`observe` read, for the line that says why a drive was cut:
        ``sigma 0.31 m / 4.2 deg over 0.25 m`` or ``fit 0.12 under 0.30``."""
        return self._reading

    @property
    def lost_since(self) -> float | None:
        """When the reading first went bad, ``None`` while it is healthy. A caller with a rule of
        its own reads it: goto also asks how far the wheels carried the cart since that moment,
        because a cart spinning on a stuck wheel with a poor reading is not driving blind."""
        return self._lost_since


@dataclass
class JumpClear:
    """When a correction has moved the pose so far that the obstacle grid behind it is a lie,
    and that grid must be emptied.

    The evidence is the step ``map -> odom`` takes. The cart's own motion lives in
    ``odom -> base_link``, so this transform moves only when an accepted word corrects the pose
    and the step IS that correction, whatever drove the update. Three rules on it: the first
    transform is a baseline and never a jump, a step no longer than ``clear_costmap_jump_m`` is
    a correction the grid absorbs (0 clears never), and two steps closer together than
    ``min_gap_s`` buy one clear — emptying a grid costs its owner a rebuild from the live scans,
    and a clear per update would leave a controller steering on an empty map.

    What emptying means is the caller's business: :meth:`moved` calls ``clear`` with the jump in
    metres and this class decides nothing else. On the robot that callback asks Nav2 to clear
    the local costmap, because the camera's marks there were laid at the pose before the jump
    and nothing else takes them back (2026-09-16: the camera layer clears only inside its own
    80 degree fan, and camera-only there is no lidar layer to scrub the rest).
    """

    clear: Callable[[float], None]
    min_gap_s: float = 1.0
    clear_costmap_on_jump: bool = True
    clear_costmap_jump_m: float = 0.10
    _previous: tuple[float, float, float] | None = field(default=None, init=False)
    _cleared_s: float = field(default=-math.inf, init=False)

    switches: ClassVar[tuple[str, ...]] = ("clear_costmap_on_jump", "clear_costmap_jump_m")

    def switch(self, name: str, value: Any) -> None:
        """A live flag by its name (:attr:`switches`); ``ValueError`` for any other name."""
        if name not in self.switches:
            raise ValueError(f"{name}: not a switch of the jump watch")
        setattr(self, name, bool(value) if name == self.switches[0] else float(value))

    def moved(self, map_odom: tuple[float, float, float], now: float) -> float:
        """Take the ``map -> odom`` a word has just published, clear the grid when it jumped, and
        return how far the pose moved (0.0 for the first transform, the baseline)."""
        previous, self._previous = self._previous, map_odom
        if previous is None:
            return 0.0
        jump_m = math.hypot(map_odom[0] - previous[0], map_odom[1] - previous[1])
        if self.due(jump_m, now):
            self._cleared_s = now
            self.clear(jump_m)
        return jump_m

    def due(self, jump_m: float, now: float) -> bool:
        """Whether a jump of ``jump_m`` earns a clear now: the switch on, the step past
        ``clear_costmap_jump_m``, and the previous clear at least ``min_gap_s`` behind."""
        return (
            self.clear_costmap_on_jump
            and self.clear_costmap_jump_m > 0.0
            and jump_m > self.clear_costmap_jump_m
            and now - self._cleared_s >= self.min_gap_s
        )


@dataclass(frozen=True)
class SourceSilence:
    """What a tracker may claim about its pose once no source has spoken for ``patience_s``.

    On 2026-09-14 the board published fit 0.70 for 141 s with ``sources=camera`` and not one
    measurement arriving: the tracker had nothing but dead reckoning, and the goal server —
    which reads only that number — accepted `printer` and `home` and drove them. The fit is the
    one word the rest of the stack has for "do I know where I am", so a fit nobody measured
    recently must not be published as if somebody had.

    It goes to 0.0 rather than decaying: 0.0 is under every rung of the ladder above at once —
    under :data:`DRIVE_FIT` so :class:`GoalGate` refuses the next goal, and under
    :data:`BLIND_FIT` so :class:`BlindDriveWatch` stops the one already running after its own
    patience — and a decay would only choose the second at which each of those happens while
    saying the same thing. Nothing here reaches :meth:`LostWatch.observe`, which takes the
    tracker's OWN fit: a silent sensor is not evidence that the pose is wrong, and a whole-map
    search on no scan at all would re-seed on nothing.
    """

    patience_s: float = SOURCE_PATIENCE_S
    zeroes_fit: bool = True  # the node's switch: off, the silence is reported and nothing else

    def silent(self, age_s: float) -> bool:
        """True when the freshest source has been quiet for longer than the patience
        (``age_s`` is ``inf`` where no source has ever spoken)."""
        return age_s > self.patience_s

    def held_at_zero(self, age_s: float) -> bool:
        """Whether the fit is being published as 0.0 right now: silent past the patience, with
        the switch on. What a report line says out loud, so a reader is never left guessing
        whether the number in front of them is measured or withheld."""
        return self.zeroes_fit and self.silent(age_s)

    def reported(self, fit: float, age_s: float) -> float:
        """The fit to publish: the measured one while a source still speaks, 0.0 once none
        has for longer than the patience; with the switch off, always the measured one."""
        return 0.0 if self.held_at_zero(age_s) else fit

    def phrase(self, age_s: float) -> str:
        """How long since a source last spoke, for a report line: ``no source ever``,
        ``last source 0.4 s ago``, or ``no source for 141.0 s``."""
        if age_s == math.inf:
            return "no source ever"
        if self.silent(age_s):
            return f"no source for {age_s:.1f} s"
        return f"last source {age_s:.1f} s ago"


# How far the tracker's own ``map -> odom`` edge may stand from the moment being painted. The
# tracker re-broadcasts it at 20 Hz whatever it is doing (pepin_bringup.relocalizer's
# _send_map_odom, every 0.05 s), so a whole second without one is twenty missed broadcasts:
# the tracker, or the link that carries it, has stopped. Painting under a frozen edge is
# painting the room where the cart WAS.
PAINT_EDGE_FRESH_S = 1.0
# ...and how sure the tracker must be of itself, in metres, where it publishes a sigma at all
# (/localization/sigma, its post-fusion covariance). A voxel is 5 cm: at 10 cm of standard
# deviation a wall is written within two voxels of where it stands, which the surface averages
# out; at half a metre it is written into the room. 0.25 and not 0.10 because of what the modes
# can offer: with a lidar the tracker knows itself to 1-2 cm and this number never bites, while
# CAMERA-ONLY the honest sigma is the graph word's own floor, 0.20 m (2026-09-17: the volume
# stopped growing the moment the lidar was muted, and a robot that cannot paint without a lidar
# can never wake up in a room it has not mapped). The price is written down: a wall painted at
# 0.20 m of pose error is thickened by that much until the lidar or a closure sharpens it.
PAINT_SIGMA_M = 0.25


@dataclass(frozen=True)
class PaintTrust:
    """Whether the tracker's pose may be painted into the world model right now.

    The volume is painted in the MAP frame: every voxel written while the pose is wrong is
    written in the wrong place, and a TSDF cannot be un-integrated. Measured on 2026-09-15,
    when the laptop's routes from the board died mid-session and the fusion went on integrating
    revolutions at the last pose TF held: the live volume kept 52.9 % of the saved map's walls,
    carved 2070 of them free, and the tracker replayed on its slice seated a median 1.90 m from
    where the file put it (scratch/volume_vs_file_seating.py).

    Four questions, in the order they can be answered without the previous one: is the fit good
    enough to drive on, was it MEASURED recently (a fit that stopped arriving keeps its last
    good value for ever — the failure above), is the tracker's own sigma small enough where it
    publishes one, and is the correction TF stands on fresh enough to place this moment.
    """

    drive_fit: float = DRIVE_FIT
    patience_s: float = SOURCE_PATIENCE_S
    max_sigma_xy_m: float = PAINT_SIGMA_M
    edge_fresh_s: float = PAINT_EDGE_FRESH_S

    def refusal(
        self,
        fit: float,
        fit_age_s: float,
        sigma_xy_m: float | None = None,
        edge_age_s: float | None = None,
    ) -> str | None:
        """``None`` when the pose may be painted with, else the phrase a report line says.

        ``sigma_xy_m`` and ``edge_age_s`` are ``None`` where nobody said — no sigma on the wire,
        no ``map -> odom`` edge in TF at all — and an absent sigma is not a refusal (the fit
        gate is the whole test until that topic exists), while an absent edge is: a pose cannot
        be placed on a correction that is not there.
        """
        if fit_age_s > self.patience_s:
            return f"the fit stopped {fit_age_s:.1f} s ago"
        # A fit of zero is not a bad fit: with no lidar of its own the tracker publishes 0.00 by
        # construction (the local_fit flag), and the pose is then held by the camera's words. The
        # sigma is the number that answers for it — one covariance out of every source that
        # spoke. Camera-only this branch used to refuse every frame, and the volume stopped
        # growing the moment the lidar was muted (2026-09-17: the cloud froze at the base).
        vouched = sigma_xy_m is not None and sigma_xy_m <= self.max_sigma_xy_m
        if fit < self.drive_fit and not vouched:
            return f"fit {fit:.2f} under {self.drive_fit:.2f} and no sigma to vouch for it"
        if sigma_xy_m is not None and sigma_xy_m > self.max_sigma_xy_m:
            return f"sigma {sigma_xy_m:.2f} m over {self.max_sigma_xy_m:.2f} m"
        if edge_age_s is None:
            return "no map -> odom edge"
        if abs(edge_age_s) > self.edge_fresh_s:
            return f"the map -> odom edge is {edge_age_s:.1f} s from the scan"
        return None


# -- the preflight: what is asked before a goal is sent ------------------------------------
# The tracker's own account of every update goes out on /localization/sources as JSON
# (pepin.localization.Localizer.sources_report): the anchor, the sources fused, and per source
# its health, the fit of its match, the correction it proposed from the prediction (``delta``,
# in cm) and the roots of its covariance. That message is the only place the stack says WHO is
# holding the pose, and the preflight reads it — a sigma says how sure the pose is, never on
# whose word, and a camera-only drive has to answer one more question before it starts.
LIDAR_SOURCE = "lidar"
GRAPH_SOURCE = "graph"


@dataclass(frozen=True)
class SourceWord:
    """One source's line of ``/localization/sources``: whether the flag has it on at all, how
    healthy the node found it (``fresh 9.9 Hz`` / ``stale 2.1 s`` / ``absent`` / ``off``), the
    fit of its last match and how far that match sat from the tracker's prediction."""

    name: str
    health: str
    fit: float | None = None
    delta_m: float | None = None

    @property
    def enabled(self) -> bool:
        """False for a source the ``sources`` flag has off: it is not evidence of anything."""
        return self.health != "off"

    def spoke(self, patience_s: float = SOURCE_PATIENCE_S) -> bool:
        """Whether this source has said anything within ``patience_s`` — ``fresh`` by the
        roster's own reckoning, or ``stale`` by less than the patience."""
        if self.health.startswith("fresh"):
            return True
        if self.health.startswith("stale"):
            try:
                return float(self.health.split()[1]) <= patience_s
            except (IndexError, ValueError):
                return False
        return False

    @property
    def recognised(self) -> bool:
        """For the pose graph: whether its last word was a recognition at all. Its fit IS the
        trust rtabmap_frame put on the word it sent, so a fit above zero is a graph that knows
        where the cart is and nothing else is."""
        return self.fit is not None and self.fit > 0.0


def source_words(report: Mapping[str, Any]) -> list[SourceWord]:
    """The ``sources`` block of one ``/localization/sources`` message as :class:`SourceWord`s.

    A malformed or half-written entry costs its own fields and never the message: what is not a
    number is simply not read, because this is evidence for a refusal, not a place to raise.
    """
    words: list[SourceWord] = []
    sources = report.get("sources")
    if not isinstance(sources, Mapping):
        return words
    for name, entry in sources.items():
        if not isinstance(entry, Mapping):
            continue
        delta = entry.get("delta")
        distance: float | None = None
        if isinstance(delta, Sequence) and not isinstance(delta, str) and len(delta) >= 2:
            try:  # the report carries the correction in centimetres
                distance = math.hypot(float(delta[0]), float(delta[1])) / 100.0
            except (TypeError, ValueError):
                distance = None
        try:
            fit = None if entry.get("fit") is None else float(entry["fit"])
        except (TypeError, ValueError):
            fit = None
        words.append(SourceWord(str(name), str(entry.get("health", "")), fit, distance))
    return words


@dataclass(frozen=True)
class Check:
    """One preflight question and its answer: the name the operator reads, whether it passed,
    and the reading that decided it."""

    name: str
    ok: bool
    detail: str

    def line(self) -> str:
        """The one line printed per check: ``preflight certainty  ok       sigma 0.06 m ...``."""
        return f"preflight {self.name:9s} {'ok     ' if self.ok else 'REFUSED'}  {self.detail}"


@dataclass(frozen=True)
class Preflight:
    """The three questions asked before a goal is sent, each answered out loud.

    ``sources``   has the tracker heard anything at all? A word of ANY enabled source within the
                  patience. A tracker that publishes numbers while every sensor is silent is the
                  failure of 2026-09-14, and no sigma small enough can make up for it.
    ``certainty`` is the pose sure enough to drive on? The fused sigma against
                  :data:`DRIVE_SIGMA_M`, or the fit against :data:`DRIVE_FIT` on a board that
                  publishes no sigma yet.
    ``agreement`` on a camera-only drive there is no scan-to-map fit at all, so the evidence
                  that the room under the cart is the room on the map is the pose graph: it has
                  recognised the place, and its word sits within ``graph_agree_m`` of the
                  tracker. With the lidar among the sources this check is the lidar itself.
    """

    drive_sigma_m: float = DRIVE_SIGMA_M
    drive_fit: float = DRIVE_FIT
    graph_agree_m: float = GRAPH_AGREE_M
    patience_s: float = SOURCE_PATIENCE_S

    def checks(
        self, words: Sequence[SourceWord], sigma: Sigma | None, fit: float | None = None
    ) -> list[Check]:
        """All three, in the order they are printed; ``words`` empty means no
        ``/localization/sources`` message was heard at all."""
        return [self.sources(words), self.certainty(sigma, fit), self.agreement(words)]

    @staticmethod
    def passed(checks: Sequence[Check]) -> bool:
        """True when every check passed: the one thing a caller needs to decide to drive."""
        return all(check.ok for check in checks)

    def sources(self, words: Sequence[SourceWord]) -> Check:
        """Has any enabled source spoken within the patience."""
        if not words:
            return Check(
                "sources",
                False,
                "the tracker published no word at all: nothing on /localization/sources"
                " (is the relocalizer up?)",
            )
        live = [w for w in words if w.enabled and w.spoke(self.patience_s)]
        roster = ", ".join(f"{w.name} {w.health}" for w in words) or "no source on the roster"
        if live:
            return Check("sources", True, roster)
        return Check("sources", False, f"no source has spoken in {self.patience_s:.0f} s: {roster}")

    def certainty(self, sigma: Sigma | None, fit: float | None) -> Check:
        """Is the pose sure enough to start a drive: the sigma where there is one, the fit
        where the board publishes none (an older build), and the line says which judged."""
        if sigma is None:
            if fit is None:
                return Check(
                    "certainty",
                    False,
                    "no /localization/sigma and no /localization_fit from this board: nothing"
                    " says how sure the pose is",
                )
            return Check(
                "certainty",
                fit >= self.drive_fit,
                f"no /localization/sigma on this board: judged by the lidar's fit {fit:.2f}"
                f" (needs {self.drive_fit:.2f})",
            )
        if not sigma.fresh(self.patience_s):
            return Check(
                "certainty",
                False,
                f"the tracker's sigma stopped {sigma.age_s:.1f} s ago: the tracker itself is"
                " not publishing, whatever the sensors do",
            )
        return Check(
            "certainty",
            sigma.xy_m <= self.drive_sigma_m,
            f"judged by sigma: the pose is known to {sigma.phrase()}"
            f" (needs {self.drive_sigma_m:.2f} m)",
        )

    def agreement(self, words: Sequence[SourceWord]) -> Check:
        """Where no lidar holds the pose, whether the graph recognises the room and agrees with
        the tracker about where in it the cart stands."""
        by_name = {w.name: w for w in words}
        lidar = by_name.get(LIDAR_SOURCE)
        if lidar is not None and lidar.enabled and lidar.spoke(self.patience_s):
            return Check(
                "agreement", True, f"the lidar is holding the pose ({lidar.health}): matched here"
            )
        graph = by_name.get(GRAPH_SOURCE)
        if graph is None or not graph.enabled:
            return Check(
                "agreement",
                False,
                "camera-only and the graph is not among the sources: nothing recognises the room",
            )
        if not graph.spoke(self.patience_s) or not graph.recognised:
            return Check(
                "agreement",
                False,
                f"camera-only and the graph has recognised nothing ({graph.health},"
                f" fit {'none' if graph.fit is None else f'{graph.fit:.2f}'}): the cart may be"
                " anywhere on this map",
            )
        if graph.delta_m is None:
            return Check(
                "agreement",
                False,
                "camera-only and the graph's last word carries no delta: it measured nothing on"
                " the tracker's last update",
            )
        return Check(
            "agreement",
            graph.delta_m <= self.graph_agree_m,
            f"camera-only: the graph recognises the room (fit {graph.fit:.2f}) and its word is"
            f" {graph.delta_m:.2f} m from the tracker (allowed {self.graph_agree_m:.2f} m)",
        )
