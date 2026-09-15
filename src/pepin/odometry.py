"""Wheel-encoder odometry for a differential-drive base.

The pose is integrated with the exact arc model: between two encoder reads
the robot is assumed to move along a circular arc of constant curvature.
For straight segments this degenerates gracefully to a line.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Protocol

from pepin.geometry import BaseGeometry
from pepin.kinematics import Twist

_log = logging.getLogger(__name__)


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


class PoseSink(Protocol):
    """Anything that keeps a trail of poses in time — the tracker's odometry history."""

    def add(self, stamp: float, pose: Pose2D) -> None:
        """Remember where the robot was at ``stamp``."""


class RunawayWatch:
    """The odometry frame teleporting: a step no cart could drive, with no twist to justify it.

    An EKF fed one bad velocity does not fail loudly — it flies. On 2026-09-14 the board's
    ``robot_localization`` reached 43 km from the flat at 60 m/s after a bad /vo input, and
    every consumer followed it: two costmaps chased the pose at 200 % CPU, and the depth
    pipeline carried its lidar scans one to two metres across 25 ms of frame gap and refitted
    the depth law from the wreckage. Nothing in the message says "wrong"; what says it is the
    arithmetic between two samples of the same topic.

    Two arms, both needed. The step is impossible — faster than ``max_speed_mps`` (five times
    this cart's 0.3 m/s top speed) or longer than ``max_step_m`` in one sample. And the twist
    that came with it cannot account for it: either the wheels say the cart stands still
    (``still_mps``, ``still_radps``), or the twist itself claims more than the cart can do
    (``cart_top_mps``), which is not a measurement but the same runaway seen from the other
    field. On the tape of that evening (ros/maps/rec/0260_20260914_155145Z_home.jsonl, 146 ekf
    records over 7.3 s) the frame sat at x 3493.7 m with |vx| never above 0.031 m/s and a worst
    single step of 0.825 m/s — a pose contradicting its own twist, sample after sample.

    The reference does not move while the frame is away: the caller is holding a pose carried
    from the last sample that made sense, and advancing to a pose 3.5 km out would hand it that
    jump the moment one sample passed. So an episode ends only when the frame comes back to
    somewhere reachable from where it left.

    The frame can also spin in place, and that is the third arm. On 2026-09-14 between 14:48 and
    14:50 the board's EKF turned its odom -> base_link yaw by about 90 degrees while the cart
    stood on its charger — x and y never left (0, 0) — and the tracker, refusing its corrections
    under occlusion ("fit 0.45 but the scan is mostly things the map does not know"), followed
    the frame round. A step in HEADING is judged the way a step in position is: against what the
    only thing that can turn this cart could have done in the time between the samples. At rest
    the gyro reads 0.3 deg/s on average and 1 deg/s at worst, so a heading step beyond
    ``max_yaw_rate_radps`` (the twist's own rate where it has one, else half a turn a second)
    times the interval, plus ``yaw_margin_rad``, is not something the cart did. Only while the
    wheels are at rest: a turning cart is never refused, and the cause of the jump is not this
    class's business — its business is that the carried pose does not follow it.
    """

    def __init__(
        self,
        max_speed_mps: float = 1.5,
        max_step_m: float = 0.5,
        still_mps: float = 0.05,
        still_radps: float = 0.25,
        cart_top_mps: float = 0.3,
        max_yaw_rate_radps: float = math.pi,
        yaw_margin_rad: float = math.radians(5.0),
    ) -> None:
        self._max_speed_mps = max_speed_mps
        self._max_step_m = max_step_m
        self._still_mps = still_mps
        self._still_radps = still_radps
        self._cart_top_mps = cart_top_mps
        self._max_yaw_rate_radps = max_yaw_rate_radps
        self._yaw_margin_rad = yaw_margin_rad
        self.reason = ""  # what the last refusal was about: "position" or "yaw"
        self._pose: Pose2D | None = None
        self._stamp: float | None = None
        self.streak = 0  # consecutive samples refused: one episode, so it is said once

    def feed(
        self, history: PoseSink, pose: Pose2D, stamp: float, vx: float, wz: float, guard: bool
    ) -> bool:
        """Hand ``history`` this odometry sample unless the frame ran away; True when it was
        carried. ``guard`` false carries every sample, as before the guard existed, and keeps
        the reference following the frame so the switch can be thrown back live.

        The refusal is logged once per episode through the ``pepin`` logger, which the node
        forwards to the ROS log.
        """
        if not self.judge(pose, stamp, vx, wz) or not guard:
            self.adopt(pose, stamp)
            history.add(stamp, pose)
            return True
        if self.streak == 1 and self.reason == "yaw":
            _log.error(
                f"the odometry frame turned to {math.degrees(pose.theta):+.0f} deg while its"
                f" twist read {math.degrees(wz):+.1f} deg/s and the wheels stood still: the step"
                " is not carried and the heading stays where it was"
            )
        elif self.streak == 1:
            _log.error(
                f"the odometry frame ran away to ({pose.x:+.1f}, {pose.y:+.1f}) m while its twist"
                f" read {vx:+.3f} m/s, {wz:+.3f} rad/s: the step is not carried and the pose"
                " stays where it was"
            )
        return False

    def judge(self, pose: Pose2D, stamp: float, vx: float, wz: float) -> bool:
        """True when this odometry sample must not reach the pose that is being carried: an
        impossible step from the last trusted sample with a twist that cannot account for it.

        The first sample, and any sample not newer than the reference, is never refused —
        there is nothing to compare it with.
        """
        if self._pose is None or self._stamp is None or stamp <= self._stamp:
            return False
        dt = stamp - self._stamp
        step = math.hypot(pose.x - self._pose.x, pose.y - self._pose.y)
        impossible = step > self._max_step_m or step / dt > self._max_speed_mps
        still = abs(vx) <= self._still_mps and abs(wz) <= self._still_radps
        unbelievable = abs(vx) > self._cart_top_mps  # a twist no wheel of this cart can turn
        # The heading: what the gyro itself claims it is turning at bounds what the frame may
        # turn by, and the cap stands in where there is no rate to read.
        rate = min(abs(wz), self._max_yaw_rate_radps) if math.isfinite(wz) else 0.0
        turn = abs(wrap_angle(pose.theta - self._pose.theta))
        spun = still and turn > rate * dt + self._yaw_margin_rad
        refused = (impossible and (still or unbelievable)) or spun
        if refused:
            self.streak += 1
            self.reason = "yaw" if spun and not impossible else "position"
        return refused

    def adopt(self, pose: Pose2D, stamp: float) -> None:
        """Take this sample as the one the next is measured against; the episode, if any, ends."""
        self._pose, self._stamp = pose, stamp
        self.streak = 0
        self.reason = ""


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
