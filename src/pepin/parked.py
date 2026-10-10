"""Whether the robot is parked — the base not driving and the head not turning, for long enough
that a depth pair taken now shows what the last one showed — and how many pairs a parked robot
is worth.

WHY. RAFT-Stereo takes the laptop's GPU 96 ms a pair; at 5.9 pairs/s that is over half of its
time, and everything else on that GPU pays for it: measured on this laptop (2026-10-09, four runs
a condition) Whisper went 190 -> 276 ms an utterance and the TTS's first-chunk p90 96 -> 469 ms
beside it, and with one pair a second Whisper was back to 198 ms. A parked robot sees the same
room in every pair, so the depth node lets one through to the matcher every ``1 / pairs_hz``
seconds while parked and drops the rest at the door (:meth:`ParkedGate.admit`); the first sign of
motion lets the next pair through at full rate.

PARKED is two witnesses, both still for ``after_s``:

- the base: every odometry twist under :data:`SPEED_MPS` in the plane and :data:`TURN_DPS` in yaw
  (:meth:`ParkedGate.observe_base`);
- the head: the camera's pan and tilt in base_link within :data:`HEAD_DEG` of where they stood
  when the stillness began (:meth:`ParkedGate.observe_head`), so a slow drift counts as well as a
  saccade.

Leaving takes one sample over a threshold — no hysteresis on the way out — and entering takes
``after_s`` of both. A witness silent for :data:`STALE_S`, or one the caller cannot read
(:meth:`ParkedGate.lose_head`), is not still: whatever cannot be seen is taken to be moving, and
its stillness starts over at its next sample. So a node whose odometry or neck is gone runs at
full rate, as it did before the gate existed.

Nothing here is ROS and nothing reads a clock: the caller hands in its own monotonic ``now``, and
calls from one thread (the node's executor).
"""

from __future__ import annotations

import math

# The thresholds are the gate's definition of "still", not knobs: they sit far from both sides of
# what they separate. On the ~15 s at rest before the first command of 44 drive bags (0356-0414,
# 33120 samples) the EKF's twist peaked at 0.012 m/s and 0.83 deg/s and never crossed either, and
# crossed one 242 ms (p50; p90 331) after the first non-zero /cmd_vel. The neck's 2 s range at rest
# was 0.016 deg p50 and 0.025 p90 (the encoders' TF, 50 Hz): 0.5 deg is twenty times that.
SPEED_MPS = 0.02  # the base's speed in the plane under which it stands still
TURN_DPS = 1.0  # ...and its yaw rate, degrees a second
HEAD_DEG = 0.5  # how far the camera's pan or tilt may wander from where the stillness began
STALE_S = 0.5  # a witness silent this long is not still (both arrive at 50 Hz: 25 samples)
PAIRS_HZ = 1.0  # the knob parked_pairs_hz's default: pairs a second while parked; 0 is off
AFTER_S = 2.0  # the knob parked_after_s's default: seconds of stillness before parked


class Stillness:
    """One witness's stillness: since when it has been still and when it was last heard, both on
    the caller's monotonic clock."""

    def __init__(self) -> None:
        self.since: float | None = None  # the start of the current still run; None: none yet
        self.heard: float | None = None

    def still(self, now: float) -> None:
        """A sample under the thresholds: the run goes on, or starts if the witness was silent."""
        if self.since is None or self.heard is None or now - self.heard > STALE_S:
            self.since = now
        self.heard = now

    def moved(self, now: float) -> None:
        """A sample over a threshold: the run starts again from this moment."""
        self.since, self.heard = now, now

    def lose(self) -> None:
        """The witness cannot be read: nothing is still until it is heard again."""
        self.since, self.heard = None, None

    def still_for(self, now: float) -> float | None:
        """Seconds of the current still run, or ``None`` while unheard or silent past
        :data:`STALE_S`."""
        if self.since is None or self.heard is None or now - self.heard > STALE_S:
            return None
        return now - self.since


class ParkedGate:
    """The parked state and the pair rate it allows (the module docstring has the rule).

    ``pairs_hz`` and ``after_s`` are live: the owning node writes them when its knobs change."""

    def __init__(self, pairs_hz: float = PAIRS_HZ, after_s: float = AFTER_S) -> None:
        self.pairs_hz = pairs_hz  # live: parked_pairs_hz; 0 is off (every pair admitted)
        self.after_s = after_s  # live: parked_after_s
        self.base = Stillness()
        self.head = Stillness()
        self._anchor: tuple[float, float] | None = None  # pan, tilt (rad) the head's run began at
        self._admitted: float | None = None  # when the last pair went through

    def observe_base(self, now: float, speed_mps: float, turn_dps: float) -> None:
        """One odometry twist: speed in the plane (m/s) and |yaw rate| (deg/s). A NaN moves."""
        if speed_mps < SPEED_MPS and turn_dps < TURN_DPS:
            self.base.still(now)
        else:
            self.base.moved(now)

    def observe_head(self, now: float, pan_rad: float, tilt_rad: float) -> None:
        """The camera's pan and tilt in base_link (rad): still while both stay within
        :data:`HEAD_DEG` of where the current run began; past it, the run starts again here."""
        anchor = self._anchor
        if anchor is None or self.head.still_for(now) is None:
            self._anchor = (pan_rad, tilt_rad)
            self.head.still(now)
            return
        drift = max(abs(pan_rad - anchor[0]), abs(tilt_rad - anchor[1]))
        if not drift <= math.radians(HEAD_DEG):  # a NaN moves
            self._anchor = (pan_rad, tilt_rad)
            self.head.moved(now)
        else:
            self.head.still(now)

    def lose_head(self) -> None:
        """No pose of the head to judge by (no edge, or a dead one): not still."""
        self._anchor = None
        self.head.lose()

    def parked(self, now: float) -> bool:
        """Whether both witnesses have been still for ``after_s``."""
        base, head = self.base.still_for(now), self.head.still_for(now)
        return base is not None and head is not None and min(base, head) >= self.after_s

    def admit(self, now: float) -> bool:
        """Whether a pair arriving at ``now`` goes on to the matcher: every one while moving or
        with ``pairs_hz`` 0, and while parked one each ``1 / pairs_hz`` seconds from the last
        that went through."""
        if self.pairs_hz <= 0.0 or not self.parked(now):
            self._admitted = now
            return True
        if self._admitted is None or now - self._admitted >= 1.0 / self.pairs_hz:
            self._admitted = now
            return True
        return False

    def text(self, now: float) -> str:
        """``parked (still 24 s)``, or ``moving`` and what each witness says, for a report line."""
        if self.parked(now):
            base, head = self.base.still_for(now), self.head.still_for(now)
            return f"parked (still {min(base or 0.0, head or 0.0):.0f} s)"
        return f"moving ({_witness('base', self.base, now)}, {_witness('head', self.head, now)})"


def _witness(name: str, still: Stillness, now: float) -> str:
    """``base still 0.4 s``, ``base unheard 3.1 s`` or ``base never heard``."""
    held = still.still_for(now)
    if held is not None:
        return f"{name} still {held:.1f} s"
    if still.heard is None:
        return f"{name} never heard" if name == "base" else f"{name} pose unknown"
    return f"{name} unheard {now - still.heard:.1f} s"


__all__ = [
    "AFTER_S",
    "HEAD_DEG",
    "PAIRS_HZ",
    "SPEED_MPS",
    "STALE_S",
    "TURN_DPS",
    "ParkedGate",
    "Stillness",
]
