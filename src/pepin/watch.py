"""When to stop trusting the tracked pose and search the whole map for the robot.

The decision is pure: a fit, whether the robot is moving, whether a goal is running, and the
clock. Keeping it out of the ROS node is what makes it testable — the node only supplies the
readings and acts on the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    _streak: int = field(default=0, init=False)
    _last_fit: float = field(default=0.0, init=False)
    _failures: int = field(default=0, init=False)
    _quiet_until: float = field(default=0.0, init=False)

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
        """A pose was adopted from anywhere: give the tracker a moment before judging it again."""
        self._streak = 0
        self._quiet_until = max(self._quiet_until, now + self.cooldown_s)
