"""When to stop trusting the tracked pose and search the whole map for the robot.

The decision is pure: a fit, whether the robot is moving, whether a goal is running, and the
clock. Keeping it out of the ROS node is what makes it testable — the node only supplies the
readings and acts on the answer.

:class:`GoalGate` is the same question one step earlier — may a drive START — and it is the one
rule that also has to answer where no tracker exists at all (online SLAM): there the evidence is
the age of ``map -> base_link`` and of the SLAM correction (:class:`Correction`), not a fit.

:class:`SourceSilence` is the question under both of those: is there still a sensor speaking at
all? A fit measured minutes ago is not a fit, and the tracker published one for 141 s while it
drove on dead reckoning alone (2026-09-14).
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
class Readiness:
    """Whether a goal may start now, whether a tracker is there at all, and why not when it may
    not — the phrase the operator reads on a refusal."""

    ready: bool
    tracker: bool  # a tracker publishes a fit here: what arms the blind-drive watch
    search: bool = False  # the tracker is up but lost: one whole-map search is owed first
    reason: str = ""


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
    """May the cart be sent to a goal: on the tracker's fit where a tracker runs, on the age of
    ``map -> base_link`` and of the SLAM correction where none does.

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
    """

    drive_fit: float = DRIVE_FIT
    fresh_s: float = TF_FRESH_S
    correction_fresh_s: float = CORRECTION_FRESH_S

    def verdict(
        self, fit: float | None, tf_age_s: float | None, correction: Correction | None = None
    ) -> Readiness:
        """One goal's answer. ``fit`` is the tracker's, or ``None`` where no tracker speaks;
        ``tf_age_s`` is how many seconds ago ``map -> base_link`` was stamped (``None``: nothing
        publishes it); ``correction`` is the SLAM half's pulse where it is watched. Returns the
        :class:`Readiness` the caller acts on."""
        if fit is not None:
            if fit >= self.drive_fit:
                return Readiness(True, tracker=True)
            return Readiness(
                False,
                tracker=True,
                search=True,
                reason=f"fit {fit:.2f} under {self.drive_fit:.2f}: stand still or relocalize",
            )
        if tf_age_s is None:
            return Readiness(
                False,
                tracker=False,
                reason="no tracker, and nothing publishes map -> base_link: the SLAM half of the"
                " stack is not up",
            )
        if tf_age_s > self.fresh_s:
            return Readiness(
                False,
                tracker=False,
                reason=f"no tracker, and map -> base_link is {tf_age_s:.1f} s old: a drive needs"
                f" it fresher than {self.fresh_s:.1f} s",
            )
        if correction is not None and correction.stale(self.correction_fresh_s):
            return Readiness(
                False,
                tracker=False,
                reason=f"no tracker, and {correction.phrase()}: the half of the stack that owns"
                " the pose is not here. map -> base_link stays fresh either way — it is"
                " re-broadcast from the last correction",
            )
        return Readiness(True, tracker=False)


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
