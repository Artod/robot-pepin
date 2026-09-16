"""Wheel slip seen from the lidar: the wheels report motion, the world around the robot does not.

Two scans a tenth of a second apart from a robot that really moved differ everywhere; from a
robot spinning its wheels on a carpet edge they are the same picture. That difference is the
only honest slip signal this cart has (no wheel-drop switch, no optical floor sensor), and it
needs no map: it compares consecutive scans beam by beam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pepin.odometry import Pose2D


def scan_changed(
    previous: NDArray[np.float64],
    current: NDArray[np.float64],
    threshold_m: float = 0.02,
    min_share: float = 0.15,
    min_valid: int = 60,
) -> tuple[bool, float]:
    """Did the surroundings move between two scans of equal binning?

    Returns (changed, share of comparable beams whose range moved by more than
    ``threshold_m``). A straight 4 cm step moves only the beams looking along the
    motion, so the share, not the median, is what separates driving (a quarter of
    the beams move) from slipping (none do). Fewer than ``min_valid`` comparable
    beams counts as "changed": a blind scan must not read as slip.
    """
    if previous.shape != current.shape:
        return True, math.inf
    both = np.isfinite(previous) & np.isfinite(current)
    if int(both.sum()) < min_valid:
        return True, math.inf
    share = float((np.abs(previous[both] - current[both]) > threshold_m).mean())
    return share > min_share, share


def slipping(
    motion: Pose2D,
    changed: bool,
    min_travel_m: float = 0.03,
    min_turn_deg: float = 3.0,
) -> bool:
    """Wheels claim more than ``min_travel_m`` or ``min_turn_deg`` while the scan stood still."""
    claimed = math.hypot(motion.x, motion.y) >= min_travel_m or abs(motion.theta) >= math.radians(
        min_turn_deg
    )
    return claimed and not changed


class SlipWatch:
    """Slip scan by scan, for a tracker: the wheels' step since the previous scan against
    whether the picture changed (:func:`scan_changed`, :func:`slipping`). ``streak`` counts
    the consecutive slipping scans, so a caller can say it once when the third one lands."""

    def __init__(self) -> None:
        self._last_ranges: NDArray[np.float64] | None = None
        self._last_odom: Pose2D | None = None
        self.streak = 0

    def observe(self, ranges: NDArray[np.float64], odom: Pose2D) -> bool:
        """True when the wheels claim a step since the previous scan and the ranges (equal
        binning) show the same picture. The first scan is never slip: nothing to compare."""
        from pepin.scanmatch import relative_motion

        changed = True
        if self._last_ranges is not None:
            changed, _ = scan_changed(self._last_ranges, ranges)
        step = Pose2D() if self._last_odom is None else relative_motion(self._last_odom, odom)
        slip = slipping(step, changed)
        self._last_ranges, self._last_odom = ranges, odom
        self.streak = self.streak + 1 if slip else 0
        return slip


# ---- the same question asked of the pictures ---------------------------------------------------
# The scan comparison above needs the lidar. Camera-only the cart has one witness left: the visual
# odometry, which answers "did the PICTURE move". On 2026-09-16, with the cart held by hand, the
# wheels invented 36 cm in 2.4 s and the EKF followed them to 34 cm while the lidar measured 3 cm --
# the filter's own Mahalanobis gate cannot see a slip, because it judges each wheel sample against a
# prediction built from the wheel samples before it. Standing still the visual odometry walks 0.2 cm
# in 42 s (worst 0.6 cm in 2 s, measured the same day), a noise of some 0.3 cm/s against the 13 cm/s
# the wheels were claiming: a ratio of forty, so the test needs no subtlety. This is NOT a judge of
# metres -- the visual odometry's scale rides the depth network and is 5-10 % off, which is why the
# ratio sits far outside that band and a disagreement must hold before it counts.
# The wheels are called liars when the picture shows less than this share of the speed they
# claim. Honest driving disagrees by the depth scale's 5-10 %; a slip disagrees by everything.
PICTURE_SLIP_RATIO = 0.5
# How long the disagreement must hold before the wheels are dropped. The visual odometry
# publishes at 2.7-3.3 Hz, so this is one or two of its frames: long enough that a single late
# frame is not a slip, short enough that 0.4 s of a wheel's lie is all that reaches the filter.
PICTURE_SLIP_HOLD_S = 0.4
# Below this the wheels are not claiming to drive anywhere and there is nothing to disbelieve.
PICTURE_SLIP_MIN_SPEED_M_S = 0.05
# A visual odometry older than this cannot testify: it drops frames and a 1.7 s gap was seen
# live. With no witness the wheels are believed -- the cart must keep moving when the camera
# half is gone (the board drives alone on a WiFi loss, by design).
PICTURE_VO_FRESH_S = 0.5


@dataclass(frozen=True)
class PictureSlipVerdict:
    """What the watch makes of this moment: whether the wheels are slipping, the two speeds it
    compared (m/s), how long the disagreement has held, and the reason in words for the log."""

    slipping: bool
    wheel_speed: float
    vo_speed: float
    held_s: float
    reason: str

    @property
    def said(self) -> str:
        """The verdict as a sentence for the log."""
        return (
            f"wheels {self.wheel_speed:.2f} m/s, picture {self.vo_speed:.2f} m/s"
            f" ({self.reason})"
        )


class PictureSlip:
    """Compares the speed the wheels report with the speed the pictures show, and says when the
    wheels are to be disbelieved. Pure arithmetic: feed it both speeds and the clock."""

    def __init__(
        self,
        ratio: float = PICTURE_SLIP_RATIO,
        hold_s: float = PICTURE_SLIP_HOLD_S,
        min_speed: float = PICTURE_SLIP_MIN_SPEED_M_S,
        fresh_s: float = PICTURE_VO_FRESH_S,
    ) -> None:
        self.ratio = ratio
        self.hold_s = hold_s
        self.min_speed = min_speed
        self.fresh_s = fresh_s
        self._since: float | None = None  # when the present disagreement began
        self._last: float | None = None  # the clock of the last verdict
        self.slips = 0  # disagreements that grew into a verdict, since the start

    @property
    def slipping(self) -> bool:
        """Whether the disagreement in hand has lasted long enough to call the wheels liars."""
        return self._held is not None and self._held >= self.hold_s

    @property
    def _held(self) -> float | None:
        """How long the present disagreement has lasted, or ``None`` when there is none."""
        if self._since is None or self._last is None:
            return None
        return self._last - self._since

    def change(
        self,
        now: float,
        wheel_speed: float,
        vo_speed: float,
        vo_at: float | None,
        muted: bool,
        watching: bool = True,
    ) -> tuple[bool, PictureSlipVerdict] | None:
        """Whether the wheels' voice must be taken away or given back at this moment, with the
        verdict that decided it — ``None`` while nothing changes. With ``watching`` false the
        watch stands down and only ever gives the voice back."""
        verdict = self.feed(now, wheel_speed, vo_speed, vo_at)
        wanted = verdict.slipping and watching
        return None if wanted == muted else (wanted, verdict)

    def feed(
        self, now: float, wheel_speed: float, vo_speed: float, vo_at: float | None
    ) -> PictureSlipVerdict:
        """One moment's verdict from the wheels' speed, the picture's speed and the moment that
        picture is from (``None`` for no visual odometry at all). A verdict that turns true is
        counted once, when it turns."""
        was = self.slipping  # before this moment's clock lands, or a verdict counts itself
        self._last = now
        wheel, seen = abs(float(wheel_speed)), abs(float(vo_speed))
        if vo_at is None or now - vo_at > self.fresh_s:
            self._since = None
            return PictureSlipVerdict(False, wheel, seen, 0.0, "no picture to judge by")
        if wheel < self.min_speed:
            self._since = None
            return PictureSlipVerdict(False, wheel, seen, 0.0, "the wheels claim nothing")
        if seen >= self.ratio * wheel:
            self._since = None
            return PictureSlipVerdict(False, wheel, seen, 0.0, "the picture agrees")
        if self._since is None:
            self._since = now
        held = now - self._since
        if held < self.hold_s:
            return PictureSlipVerdict(False, wheel, seen, held, "disagreeing, not long enough yet")
        if not was:
            self.slips += 1
        return PictureSlipVerdict(True, wheel, seen, held, "the wheels turn, the picture stands")


class PictureSpeed:
    """The camera's own speed from consecutive visual-odometry poses, and the moment it was last
    measured. A gap longer than ``max_gap_s`` starts again rather than dividing by it."""

    def __init__(self, max_gap_s: float = 2.0) -> None:
        self.max_gap_s = max_gap_s
        self.speed = 0.0
        self.at: float | None = None
        self._seen: tuple[float, float, float] | None = None

    def feed(self, x: float, y: float, now: float) -> float:
        """Take one pose and its moment; returns the speed in hand (m/s)."""
        seen, self._seen = self._seen, (float(x), float(y), float(now))
        gap = 0.0 if seen is None else now - seen[2]
        if seen is not None and 0.0 < gap < self.max_gap_s:
            self.speed = math.hypot(x - seen[0], y - seen[1]) / gap
            self.at = now
        return self.speed


