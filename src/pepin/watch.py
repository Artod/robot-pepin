"""When to stop trusting the tracked pose and search the whole map for the robot.

The decision is pure: a fit, whether the robot is moving, whether a goal is running, and the
clock. Keeping it out of the ROS node is what makes it testable — the node only supplies the
readings and acts on the answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from pepin.odometry import Pose2D, wrap_angle

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

AGREE_M = 0.5  # two fixes this close...
AGREE_DEG = 30.0  # ...and this aligned are the same place
CONFIRM_TRIES = 4  # fresh searches allowed to agree before the fix is given up as a twin


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

    def observe(self, fit: float, moving: bool, navigating: bool, now: float) -> bool:
        """One check, once a second: True when the whole map should be searched right now.

        A moving robot and a robot under a goal are never re-seeded: their fit dips for honest
        reasons (a scan and a pose milliseconds apart, a correction in progress), and a search
        that teleports the belief mid-drive is worse than a poor fit.
        """
        collapsed = self._last_fit >= self.collapse_from and fit < self.collapse_to
        self._last_fit = fit
        if moving or navigating:
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


@dataclass
class BlindDriveWatch:
    """Stops a drive whose tracker has lost its lock: below ``lost_fit`` for ``patience_s``.

    The tracker never re-searches while a goal runs (a teleport mid-drive is worse than a poor
    fit), so a drive that loses its lock keeps driving on a wrong map — run 0052 spent a minute
    at fit 0.05-0.29 and arrived drunk. Better to stop, search standing still, and go again.
    """

    lost_fit: float = 0.30
    patience_s: float = 4.0
    _lost_since: float | None = field(default=None, init=False)

    def observe(self, fit: float, now: float) -> bool:
        """True the moment the fit has been poor for longer than the patience."""
        if fit >= self.lost_fit:
            self._lost_since = None
            return False
        if self._lost_since is None:
            self._lost_since = now
        return now - self._lost_since > self.patience_s
