"""When to stop trusting the tracked pose and search the whole map for the robot.

The decision is pure: a fit, whether the robot is moving, whether a goal is running, and the
clock. Keeping it out of the ROS node is what makes it testable — the node only supplies the
readings and acts on the answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pepin.odometry import Pose2D, wrap_angle

AGREE_M = 0.5  # two fixes this close...
AGREE_DEG = 30.0  # ...and this aligned are the same place
CONFIRM_TRIES = 4  # fresh searches allowed to agree before the fix is given up as a twin
PROVISIONAL_FIT_CAP = 0.35  # what an unconfirmed fix may report: below every "good enough" gate


@dataclass
class LostWatch:
    """Decides when a whole-map search is worth its seconds, and how long to wait after a failure.

    ``lost_fit`` and ``lost_checks`` set the ordinary trigger: the scan has fitted the map poorly
    for that many checks in a row while the robot stood still. A *collapse* — the fit was healthy
    and fell off a cliff, which is what being carried or turned by hand looks like — asks at once.
    Every failed search doubles the wait before the next one, capped, so a robot that cannot find
    itself does not saturate a board that is also driving.
    """

    lost_fit: float = 0.55
    lost_checks: int = 3
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
        downstream mistakes a lucky look-alike for a localised robot."""
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

    def proposed(self, pose: Pose2D, now: float) -> None:
        """A search's first answer: held, checked again at once, applied only if agreed with."""
        self._candidate, self._confirm_tries = pose, 0
        self._quiet_until = now

    def second_opinion(self, found: Pose2D | None, now: float) -> str:
        """A fresh search's answer while a candidate is pending.

        ``"apply"``: it agrees with the candidate — adopt ``found`` now, the fix is confirmed.
        ``"hold"``: it does not — ``found`` becomes the candidate, nothing moves. ``"given_up"``:
        the searches keep disagreeing, a twin the scan cannot settle; quiet until the robot
        moves or the tracker itself gives up.
        """
        assert self._candidate is not None
        self._confirm_tries += 1
        if found is not None and self.agrees(self._candidate, found):
            self._candidate, self._confirm_tries = None, 0
            self._quiet_until = now + self.cooldown_s
            return "apply"
        if self._confirm_tries >= CONFIRM_TRIES:
            self._candidate, self._confirm_tries = None, 0
            self._failures += 1
            self._quiet_until = now + self.wait_s()
            return "given_up"
        if found is not None:
            self._candidate = found
        return "hold"

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
