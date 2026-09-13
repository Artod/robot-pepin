"""How far the cart leans, from the IMU: one estimator every consumer takes its lean from.

The cart drives over slippers, thresholds and carpet edges, and its body leans a few degrees
for a second or two. The planar EKF that owns ``odom -> base_link`` cannot see that (two_d_mode:
by design — Nav2 and the tracker want a planar odometry), so every consumer that places a
measurement in three dimensions places it as if the cart stood level. A 5 degree lean puts a
wall 3 m ahead 26 cm out of place in the fused volume.

The accelerometer and the gyro are free information that nobody was using for this. This module
turns them into one number every consumer asks for:

* :class:`LeanEstimator` — gravity's direction from the accelerometer for the slow truth, the
  gyro's rates for the fast part (a complementary filter), with the gates the floor anchor has
  always had: a non-finite sample, a reading away from 1 g (braking, a bump) and a lean beyond
  the dead band (a push leans the apparent gravity without leaning the cart) never reach the
  filter, and a lean outlasting the time constant is a floor, not a push.
* :class:`Lean` — one reading: roll and pitch about base_link's own x and y, its stamp and how
  much of it is measured rather than integrated.
* :class:`LeanHistory` — the last seconds of leans, readable at any moment in between, so a
  consumer asks for the lean at its frame's exposure and not for the lean now.
* :class:`LeanSource` — the port a :class:`pepin.frame_pose.FramePoser` takes a lean through.

Angles are radians, positive roll is right side down and positive pitch is nose down — the ROS
convention, the same signs ``pepin.mounts.rotation_from_rpy`` composes.
"""

from __future__ import annotations

import math
import threading
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pepin.depth import UP_LEVEL, Array

GRAVITY = 9.81
LEAN_TAU_S = 10.0  # the accelerometer's up vector, low-passed: a push or a bump is not a slope
LEAN_GATE_DEG = 1.0  # a sample leaning more than this from the running up is a push, not gravity
LEAN_NORM_TOLERANCE = 1.0  # m/s^2 away from gravity: the reading is not gravity alone
QUALITY_TAU_S = 1.0  # how fast the share of accepted samples forgets (the reading's quality)
HISTORY_S = 5.0  # how far back a consumer may ask for a lean
HISTORY_MAX = 2000  # and how many samples that may ever cost (50 Hz * 5 s = 250)
HISTORY_SLACK_S = 0.1  # a stamp this far past the newest sample reads as the newest lean


def _rotation_xy(roll: float, pitch: float) -> Array:
    """Ry(pitch) Rx(roll): the leaning body's axes into the level ones, no yaw."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    rotation: Array = np.array(
        [[cp, sp * sr, sp * cr], [0.0, cr, -sr], [-sp, cp * sr, cp * cr]], dtype=float
    )
    return rotation


@dataclass(frozen=True)
class Lean:
    """How the cart's body sat at one moment: roll and pitch (radians) about base_link's own x
    and y — positive right side down, nose down — with the stamp they belong to (seconds) and
    ``quality``, the share of the reading that gravity itself voted for (1.0 while the
    accelerometer is believed, falling towards 0 while the lean is carried by the gyro alone)."""

    roll: float
    pitch: float
    stamp: float
    quality: float = 1.0

    @property
    def roll_deg(self) -> float:
        """The roll in degrees, for a report line."""
        return math.degrees(self.roll)

    @property
    def pitch_deg(self) -> float:
        """The pitch in degrees, for a report line."""
        return math.degrees(self.pitch)

    @property
    def size_deg(self) -> float:
        """How far the body is from level, in degrees: the angle between up and base_link's z."""
        return math.degrees(math.acos(float(np.clip(self.up_vector()[2], -1.0, 1.0))))

    def up_vector(self) -> Array:
        """Which way is up in base_link: the unit vector gravity's opposite points along."""
        up: Array = _rotation_xy(self.roll, self.pitch).T @ np.array([0.0, 0.0, 1.0])
        return up

    def rotation(self) -> Array:
        """The 3x3 rotation ``level <- base_link``: what to put before a planar pose so a point
        measured on the leaning body lands where it really is."""
        return _rotation_xy(self.roll, self.pitch)

    @classmethod
    def from_up(cls, up: Array, stamp: float, quality: float = 1.0) -> Lean:
        """The lean whose up vector is ``up`` (base_link, any length): the exact inverse of
        :meth:`up_vector`."""
        ux, uy, uz = (float(v) for v in np.asarray(up, dtype=float))
        return cls(math.atan2(uy, uz), math.atan2(-ux, math.hypot(uy, uz)), stamp, quality)


LEVEL = Lean(0.0, 0.0, 0.0, 0.0)


class LeanSource(Protocol):
    """Where a consumer asks how the cart was leaning at a stamp: an estimator in a node, a
    tape offline, a fake in a test."""

    def lean_at(self, stamp: float) -> Lean | None:
        """The lean at ``stamp`` (seconds), or ``None`` when nothing is known about it."""
        ...


class LeanHistory:
    """The leans of the last ``horizon_s`` seconds, readable at any moment in between.

    Roll, pitch and quality are interpolated linearly between two samples (at 50 Hz, a
    hundredth of a degree). Nothing is extrapolated backwards: a stamp before the oldest sample
    reads as ``None``. Forwards the newest lean is held for ``slack_s`` — a camera frame is
    stamped a few tens of milliseconds ahead of the last IMU sample that reached the node, and
    refusing those would leave every frame unleaned.

    Every method is under one lock: the IMU callback fills the history on the node's executor
    thread while a worker thread asks it for the lean at its frame's stamp, and the two racing
    cost frames — a trim landing between the bisect and the index it found raises ``IndexError``
    out of the worker (the node logs a failed frame and drops it), and one landing between the
    two ``popleft`` calls pairs a stamp with another sample's lean, silently.
    """

    def __init__(
        self,
        horizon_s: float = HISTORY_S,
        slack_s: float = HISTORY_SLACK_S,
        max_len: int = HISTORY_MAX,
    ) -> None:
        self._horizon_s = horizon_s
        self._slack_s = slack_s
        self._max_len = max_len
        self._lock = threading.Lock()
        self._t: deque[float] = deque()
        self._leans: deque[Lean] = deque()

    def add(self, lean: Lean) -> None:
        """Append one lean; one older than the newest is ignored (a late message, a clock step)."""
        with self._lock:
            if self._t and lean.stamp <= self._t[-1]:
                return
            self._t.append(lean.stamp)
            self._leans.append(lean)
            horizon = lean.stamp - self._horizon_s
            while self._t and (self._t[0] < horizon or len(self._t) > self._max_len):
                self._t.popleft()
                self._leans.popleft()

    def __len__(self) -> int:
        with self._lock:
            return len(self._t)

    @property
    def newest(self) -> Lean | None:
        """The last lean taken in, or ``None`` when empty."""
        with self._lock:
            return self._leans[-1] if self._leans else None

    def at(self, stamp: float) -> Lean | None:
        """The lean at ``stamp`` (seconds), interpolated between the two samples around it;
        the newest lean within ``slack_s`` past the end, ``None`` before the start or beyond."""
        with self._lock:
            if not self._t:
                return None
            if stamp >= self._t[-1]:
                return self._leans[-1] if stamp - self._t[-1] <= self._slack_s else None
            if stamp < self._t[0]:
                return None
            i = bisect_left(self._t, stamp)
            if self._t[i] == stamp:
                return self._leans[i]
            before, after = self._leans[i - 1], self._leans[i]
        span = after.stamp - before.stamp
        k = 0.0 if span <= 0.0 else (stamp - before.stamp) / span
        return Lean(
            before.roll + k * (after.roll - before.roll),
            before.pitch + k * (after.pitch - before.pitch),
            stamp,
            before.quality + k * (after.quality - before.quality),
        )

    def lean_at(self, stamp: float) -> Lean | None:
        """:class:`LeanSource`: the history answers for the lean itself."""
        return self.at(stamp)


class LeanEstimator:
    """Which way is up and how far the body is from it, from the IMU: the accelerometer turned
    into base_link through its mount and low-passed for the slow truth, the gyro integrated on
    top of it for the fast part.

    At rest the chip reads +g along up; while the cart accelerates the reading leans, so three
    gates stand before the accelerometer's vote: a non-finite sample is ignored, a norm away
    from 1 g by more than ``LEAN_NORM_TOLERANCE`` (a bump, braking) is ignored, and a sample
    leaning more than ``LEAN_GATE_DEG`` from the running up is a push (0.3 m/s^2 leans the
    apparent gravity 1.8 deg, which the norm cannot see) — unless the lean outlasts the time
    constant, which no push does: then the cart stands on a slope and the up vector is re-seeded
    from the sample. The gyro has no such doubt: a world-fixed up vector turns in the body by
    ``-omega x up``, so the fast lean of a wheel climbing a threshold is followed within a
    sample while the accelerometer is still being disbelieved. ``use_gyro=False`` leaves exactly
    the accelerometer-only filter the floor anchor has always run.

    Every accepted sample is appended to :attr:`history`, so a consumer can ask for the lean at
    its frame's stamp (:meth:`lean_at`) instead of the lean now.
    """

    def __init__(
        self,
        imu_to_base: Array,
        tau_s: float = LEAN_TAU_S,
        *,
        use_gyro: bool = True,
        history: LeanHistory | None = None,
    ) -> None:
        self._rotation = np.asarray(imu_to_base, dtype=float)
        self._tau = tau_s
        self.use_gyro = use_gyro
        self.history = LeanHistory() if history is None else history
        self._up: Array = UP_LEVEL.copy()
        self._seeded = False
        self._last_t: float | None = None
        self._leaning_since: float | None = None
        self._quality = 0.0

    def observe(self, accel_imu: Array, t: float, gyro_imu: Array | None = None) -> None:
        """Feed one IMU sample (m/s^2 and rad/s, the IMU's own axes) at time ``t`` (seconds).

        The gyro turns the up vector first (nothing gates it: a rate is a rate), then the
        accelerometer pulls it back towards gravity through the gates.
        """
        a = self._rotation @ np.asarray(accel_imu, dtype=float)
        if not bool(np.all(np.isfinite(a))):
            return  # a NaN would pass the norm gate and poison the up vector for good
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        if self.use_gyro and gyro_imu is not None and self._seeded:
            self._turn(gyro_imu, dt)
        norm = float(np.linalg.norm(a))
        if abs(norm - GRAVITY) > LEAN_NORM_TOLERANCE:  # braking, a bump: not gravity alone
            self._decay(dt)
            self._last_t = t
            self._record(t)
            return
        fresh = a / norm
        if not self._seeded:
            self._up, self._seeded, self._last_t, self._quality = fresh, True, t, 1.0
            self._record(t)
            return
        lean = math.degrees(math.acos(float(np.clip(np.dot(fresh, self._up), -1.0, 1.0))))
        if lean > LEAN_GATE_DEG:
            if self._leaning_since is None:
                self._leaning_since = t
            elif t - self._leaning_since > self._tau:
                self._up, self._leaning_since = fresh, None  # a lean that lasts is the floor
            self._decay(dt)
            self._last_t = t
            self._record(t)
            return
        self._leaning_since = None
        k = 1.0 - math.exp(-dt / self._tau)
        blended = (1.0 - k) * self._up + k * fresh
        self._up = blended / np.linalg.norm(blended)
        self._quality += (1.0 - math.exp(-dt / QUALITY_TAU_S)) * (1.0 - self._quality)
        self._last_t = t
        self._record(t)

    def _turn(self, gyro_imu: Array, dt: float) -> None:
        """Turn the up vector by the body's own rotation over ``dt``: a world-fixed vector's
        body coordinates change by ``-omega x up``. The yaw rate — the big one in a pivot —
        drops out of the cross product while the cart is near level, which is the point."""
        if dt <= 0.0:
            return
        w = self._rotation @ np.asarray(gyro_imu, dtype=float)
        if not bool(np.all(np.isfinite(w))):
            return
        turned: Array = self._up - np.cross(w, self._up) * dt
        norm = float(np.linalg.norm(turned))
        if norm > 1e-9:
            self._up = turned / norm

    def _decay(self, dt: float) -> None:
        """A sample gravity did not vote for: the reading is that much less measured."""
        self._quality *= math.exp(-dt / QUALITY_TAU_S)

    def _record(self, t: float) -> None:
        """Put the current lean on the history — only once gravity has been seen at all: a
        level lean nobody measured is not a reading a consumer should be able to ask for."""
        if self._seeded:
            self.history.add(self.lean_now(t))

    @property
    def up(self) -> Array:
        """The unit vector pointing up, in base_link."""
        return self._up

    @property
    def quality(self) -> float:
        """How much of the current lean gravity itself voted for, 0 to 1."""
        return self._quality

    @property
    def roll_pitch_deg(self) -> tuple[float, float]:
        """The cart's roll and pitch in degrees (positive: right side down, nose down)."""
        lean = self.lean_now(0.0)
        return lean.roll_deg, lean.pitch_deg

    def lean_now(self, stamp: float) -> Lean:
        """The lean the filter holds this instant, stamped ``stamp``."""
        return Lean.from_up(self._up, stamp, self._quality)

    def lean_at(self, stamp: float) -> Lean | None:
        """:class:`LeanSource`: the lean at ``stamp`` from the history of accepted samples."""
        return self.history.at(stamp)


def imu_mount_rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Array:
    """The 3x3 rotation taking a vector from the IMU's axes to base_link, from the mount's
    roll-pitch-yaw in degrees (the ROS convention, like the static transform the board
    publishes; :func:`pepin.mounts.rotation_from_rpy` is the one implementation)."""
    from pepin.mounts import rotation_from_rpy

    return rotation_from_rpy(*(math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg)))


__all__ = [
    "GRAVITY",
    "HISTORY_SLACK_S",
    "LEAN_GATE_DEG",
    "LEAN_NORM_TOLERANCE",
    "LEAN_TAU_S",
    "LEVEL",
    "Lean",
    "LeanEstimator",
    "LeanHistory",
    "LeanSource",
    "imu_mount_rotation",
]
