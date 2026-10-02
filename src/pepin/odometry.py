"""Wheel-encoder odometry for a differential-drive base.

The pose is integrated with the exact arc model: between two encoder reads
the robot is assumed to move along a circular arc of constant curvature.
For straight segments this degenerates gracefully to a line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from pepin.geometry import BaseGeometry
from pepin.kinematics import Twist


def wrap_angle(angle: float) -> float:
    """Map any angle to the interval (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class Pose2D:
    """Planar pose: position in meters, heading in radians (CCW from +x)."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


class EncoderUnwrapper:
    """Turns wrapping absolute encoder readings into signed tick deltas.

    The STS3215 reports position in ``[0, ticks_per_rev)`` and wraps at the
    boundary while spinning continuously. The first reading primes the
    unwrapper and yields a delta of zero.
    """

    def __init__(self, ticks_per_rev: int) -> None:
        """``ticks_per_rev`` is the encoder's wrap modulus (4096 on the STS3215)."""
        self._full = ticks_per_rev
        self._half = ticks_per_rev // 2
        self._last: int | None = None

    def delta(self, reading: int) -> int:
        """Signed ticks since the previous reading, taking the shorter way round the wrap.

        Ambiguous beyond half a revolution: a wheel that outruns the poll rate
        folds back and reads as a small motion the other way.
        """
        if self._last is None:
            self._last = reading
            return 0
        d = (reading - self._last + self._half) % self._full - self._half
        self._last = reading
        return d

    def reset(self) -> None:
        """Forget the last reading; the next :meth:`delta` primes again and returns zero."""
        self._last = None


class DiffDriveOdometry:
    """Integrates left/right wheel travel into a planar pose."""

    def __init__(self, geometry: BaseGeometry, pose: Pose2D | None = None) -> None:
        """Only the track width matters here; ``pose`` seeds the integration (origin by default)."""
        self._track = geometry.track_width_m
        self._pose = pose or Pose2D()

    @property
    def pose(self) -> Pose2D:
        """Pose integrated so far, in the frame the odometry started in."""
        return self._pose

    def reset(self, pose: Pose2D | None = None) -> None:
        """Teleport the estimate to ``pose`` (origin by default), e.g. after a scan match."""
        self._pose = pose or Pose2D()

    def update(self, d_left_m: float, d_right_m: float) -> Pose2D:
        """Advance the pose by the distance in meters each wheel rolled since the last update.

        Mean wheel travel is the arc length, the left/right difference over the
        track width is the turn. Returns the new pose.
        """
        ds = (d_left_m + d_right_m) / 2.0
        dtheta = (d_right_m - d_left_m) / self._track
        p = self._pose
        if abs(dtheta) < 1e-9:
            dx, dy = ds * math.cos(p.theta), ds * math.sin(p.theta)
        else:
            radius = ds / dtheta
            dx = radius * (math.sin(p.theta + dtheta) - math.sin(p.theta))
            dy = -radius * (math.cos(p.theta + dtheta) - math.cos(p.theta))
        self._pose = replace(p, x=p.x + dx, y=p.y + dy, theta=wrap_angle(p.theta + dtheta))
        return self._pose


class TwistFromPose:
    """Turns consecutive wheel poses into the body twist the wheels actually measured.

    The base server reports ``v`` and ``w`` as the twist it was COMMANDED to apply, not one it
    measured: ``base_server.snapshot`` copies ``self.twist``, which is whatever /cmd_vel last
    asked for (src/pepin/base_server.py:466). Its x/y/theta, on the other hand, are integrated
    from the wheel travel and are a measurement. So the honest wheel velocity is the difference
    of two consecutive poses over their gap, which is what this returns -- and what a filter may
    fuse without closing a loop from its own command back into its own state estimate.

    Feed every pose as it arrives; the first one primes and returns a zero twist.
    """

    def __init__(self, max_gap_s: float = 1.0) -> None:
        """``max_gap_s``: a longer silence re-primes instead of dividing by a stale gap."""
        self._max_gap_s = max_gap_s
        self._pose: Pose2D | None = None
        self._stamp = 0.0

    def reset(self) -> None:
        """Forget the last pose; the next sample primes again and returns a zero twist."""
        self._pose = None

    def update(self, pose: Pose2D, stamp: float) -> Twist:
        """The body twist between the previous pose and this one, in m/s and rad/s.

        Forward speed is the straight-line step signed by the direction the robot was facing
        (a differential cart cannot move sideways, so the step is forward or backward); the yaw
        rate is the wrapped heading step over the gap. Returns a zero twist on the first sample
        and after a gap longer than ``max_gap_s``.
        """
        previous, last_stamp = self._pose, self._stamp
        self._pose, self._stamp = pose, stamp
        dt = stamp - last_stamp
        if previous is None or dt <= 0.0 or dt > self._max_gap_s:
            return Twist(0.0, 0.0)
        dx, dy = pose.x - previous.x, pose.y - previous.y
        forward = dx * math.cos(previous.theta) + dy * math.sin(previous.theta)
        return Twist(forward / dt, wrap_angle(pose.theta - previous.theta) / dt)
