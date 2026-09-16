"""The camera's own opinion of how the cart moved, on its way to the EKF.

RTAB-Map's ``rgbd_odometry`` watches the neck camera's picture and the network's metric depth
and answers with a pose per frame: a third input beside the wheels and the gyro, and the only
one that cannot be fooled by a wheel turning on carpet without the cart going anywhere. What it
cannot do is own a frame — the EKF does that — and what it cannot be trusted about is SCALE: its
translation is the depth image's translation, and that depth is a network's, corrected by a law
fitted against the lidar (0.94 to 1.98 across one afternoon, 2026-09-11). A metre it reports is
a metre of that law.

So nothing here passes rtabmap's word on unchanged. :class:`VoGate` drops what the filter must
never see — a frame the tracker lost, and a step no cart of this speed could have made — and
:func:`planar_covariance` replaces rtabmap's own covariance with a documented constant, because
a number that came out of a registration on a scaled depth is not a measurement of that scale's
error. :class:`VoTrack` publishes the sum of the steps that passed instead of rtabmap's own pose,
because the EKF differences the stream it RECEIVES and a gate that only re-anchors itself hands
it every refused jump as a velocity; :class:`PublishCap` says how often that may happen.
:class:`RestWatch` is the measurement this module exists to make possible: with the
wheels standing still, every centimetre the visual odometry walks is its own drift, and that
number decides whether it may be fused at all.

Nothing here is ROS: poses and wheel speeds in, verdicts and a report line out
(:mod:`pepin_bringup.visual_odometry` is the node around it).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "LOST_VARIANCE",
    "REST_LINEAR_M_S",
    "REST_YAW_RAD_S",
    "SCALE_ERROR",
    "SIGMA_FLOOR_M",
    "PublishCap",
    "RestDrift",
    "RestWatch",
    "VoGate",
    "VoPose",
    "VoTrack",
    "is_lost",
    "planar_covariance",
    "scaled_covariance",
]

# What rtabmap writes on the diagonal of a pose it did not measure: its odometry nodes publish a
# message per frame either way, and the lost ones carry 9999 rather than a covariance.
LOST_VARIANCE = 9999.0
# Below these the cart stands still: the base's own caps are 0.30 m/s and 1.0 rad/s
# (pepin.deployment), and the wheels report a hard zero when no wheel turns, so this is a guard
# against a tick of quantisation rather than a threshold anything is tuned to.
# What the visual odometry's own error is made of, per frame. The registration's variance --
# rtabmap's own number, 4.4e-5 m^2 (sigma 6.6 mm) on a healthy frame measured 2026-09-16 -- says
# how well the two pictures' points fell on each other, and nothing else: the points were built
# from the NETWORK's depth, whose scale still carries a 5-10 % residual after the frame law
# (measured against the lidar's beams, 2026-09-15/16). A registration can therefore be perfect
# while the metres it reports are 10 % short, and that part of the error grows with the step the
# cart took, not with the picture's quality. SCALE_ERROR is that fraction; SIGMA_FLOOR_M keeps a
# suspiciously tiny registration variance from claiming millimetre certainty.
SCALE_ERROR = 0.10
SIGMA_FLOOR_M = 0.005
REST_LINEAR_M_S = 0.01
REST_YAW_RAD_S = 0.02
# How close to rtabmap's own origin a pose has to land to be its re-initialisation rather than a
# drive: the source's poses at rest are millimetres apart (4.2 mm the largest step in 85 s,
# 2026-09-14), and a cart that drove away and came back would have to park within 5 cm of where
# rgbd_odometry started to be mistaken for one.
RESET_RADIUS_M = 0.05
# How long the wheels must have been still before the drift counts as drift. A drive ends with
# the cart rocking on its own suspension for a moment; that motion is real and not the camera's.
REST_SETTLE_S = 1.0


@dataclass(frozen=True)
class VoPose:
    """One visual-odometry pose as it left rtabmap: when, and where in ITS OWN frame.

    The frame is rtabmap's odometry origin — wherever the node happened to start — and is never
    the EKF's. Only differences between two of these ever reach the filter.
    """

    stamp: float
    x: float
    y: float
    yaw: float


def is_lost(covariance: Sequence[float]) -> bool:
    """Whether rtabmap published this pose to say it had none: :data:`LOST_VARIANCE` on the
    diagonal of the 6x6 pose covariance, which is how its odometry nodes report a lost frame."""
    if len(covariance) < 36:
        return True
    return any(covariance[i * 6 + i] >= LOST_VARIANCE for i in range(6))


def scaled_covariance(
    registration_variance: float,
    step_m: float,
    yaw_sigma_deg: float,
    scale_error: float = SCALE_ERROR,
    floor_sigma_m: float = SIGMA_FLOOR_M,
) -> list[float]:
    """The pose covariance of one visual-odometry frame: the registration's own sigma (from
    ``registration_variance``, floored at ``floor_sigma_m``) and the depth scale's share of the
    step just taken, added in quadrature -- ``sqrt(sigma_reg^2 + (scale_error * step_m)^2)`` on x
    and y, ``yaw_sigma_deg`` on yaw. A frame the cart barely moved in is worth its registration;
    a long step is worth what the network's scale is worth, which is the part rtabmap cannot see.
    """
    reg = math.sqrt(max(float(registration_variance), 0.0))
    sigma = math.hypot(max(reg, float(floor_sigma_m)), float(scale_error) * max(float(step_m), 0.0))
    return planar_covariance(sigma, yaw_sigma_deg)


def planar_covariance(sigma_m: float, yaw_sigma_deg: float, unfused: float = 1e6) -> list[float]:
    """A 6x6 row-major pose covariance for a planar measurement: ``sigma_m`` on x and y,
    ``yaw_sigma_deg`` on yaw, and ``unfused`` on z, roll and pitch.

    The huge numbers are not decoration: robot_localization reads the matrix of every axis it is
    configured for, and a small variance on an axis nobody measured is a claim that the cart
    never leaves the floor — which is true, and which the EKF already knows from ``two_d_mode``.
    A metre squared of variance there says "ask someone else" in the one language the filter
    reads.
    """
    if sigma_m <= 0.0 or yaw_sigma_deg <= 0.0:
        raise ValueError(f"a sigma is positive, not {sigma_m} m / {yaw_sigma_deg} deg")
    diagonal = [
        sigma_m**2,
        sigma_m**2,
        unfused,
        unfused,
        unfused,
        math.radians(yaw_sigma_deg) ** 2,
    ]
    matrix = [0.0] * 36
    for i, value in enumerate(diagonal):
        matrix[i * 6 + i] = value
    return matrix


class VoGate:
    """Lets through the visual-odometry poses the EKF may see, and says why it dropped the rest.

    Three reasons, all cheap and all about the same failure: rtabmap losing its tracking. A lost
    frame it announces itself (:func:`is_lost`); a RE-START of the tracking it does not — the
    pose simply jumps, and a jump differenced into a velocity is the one thing that can yank an
    odometry frame nothing else in this stack can move. So a step faster than a cart that cannot
    exceed 0.30 m/s and 1.0 rad/s (pepin.deployment) is not a measurement of anything.

    The third is the hole in the first two: both ceilings are RATIOS, and a long enough gap
    makes any jump legal. The source runs at 9-10 poses/s, so a gap is a stalled or restarted
    process — and a restarted ``rgbd_odometry`` comes back with its pose at the origin, metres
    from where it left off, after a gap of seconds. At one metre per second, a jump of X metres
    passes whenever the gap exceeds X seconds; the gap ceiling is what stops the very case the
    speed ceiling was written for.

    ``admit`` answers ``None`` for a pose the node may publish, or the reason it may not.
    """

    def __init__(
        self,
        max_speed_m_s: float = 1.0,
        max_turn_deg_s: float = 180.0,
        max_gap_s: float = 1.0,
        reset_radius_m: float = RESET_RADIUS_M,
    ) -> None:
        self.max_speed_m_s = max_speed_m_s  # live: the node's vo_max_speed flag writes it
        self.max_turn_deg_s = max_turn_deg_s  # live: the node's vo_max_turn flag writes it
        self.max_gap_s = max_gap_s  # live: the node's vo_max_gap_s flag writes it
        self.reset_radius_m = reset_radius_m  # live: the node's vo_reset_radius_m flag writes it
        self._last: VoPose | None = None

    @property
    def anchor(self) -> VoPose | None:
        """The pose the next step will be measured from: the last one this gate saw and kept.

        It is what :class:`VoTrack` has to continue from after a refusal — the two must anchor
        on the same pose or the published stream would carry the jump the gate just refused.
        """
        return self._last

    def admit(self, pose: VoPose, lost: bool) -> str | None:
        """``None`` when this pose may go to the filter, else the reason it may not.

        A refused jump still becomes the anchor the next step is measured from. That is the
        whole point: rtabmap re-initialising its tracking (``Odom/ResetCountdown``) puts the
        pose back at its origin, and a gate that kept the old anchor would call every pose after
        that a jump and go deaf for the rest of the session. One restart costs one sample. A
        pose that arrives out of order is refused WITHOUT moving the anchor — there is nothing
        to re-anchor to in the past. A pose that arrives after a gap is refused and re-anchors
        too: across a gap the speed and turn ceilings measure nothing (a jump divided by
        seconds is a walking pace), and the gap itself is the evidence.
        """
        if lost:
            return "rtabmap lost the frame"
        previous, self._last = self._last, pose
        if previous is None:
            return None
        dt = pose.stamp - previous.stamp
        if dt <= 0.0:
            self._last = previous
            return f"a pose {abs(dt):.3f} s out of order"
        radius = self.reset_radius_m
        if radius > 0.0 and _at_origin(pose, radius) and not _at_origin(previous, radius):
            was = math.hypot(previous.x, previous.y)
            return f"a reset to rtabmap's origin from {was:.2f} m out"
        if dt > self.max_gap_s:
            return f"a gap of {dt:.1f} s (the speed of a jump across it means nothing)"
        step = math.hypot(pose.x - previous.x, pose.y - previous.y)
        turn = abs(_wrapped(pose.yaw - previous.yaw))
        if step / dt > self.max_speed_m_s:
            return f"a jump of {step * 100:.0f} cm in {dt:.3f} s"
        if math.degrees(turn) / dt > self.max_turn_deg_s:
            return f"a turn of {math.degrees(turn):.0f} deg in {dt:.3f} s"
        return None


class VoTrack:
    """The pose the EKF is allowed to difference: the steps the gate admitted, summed.

    The board's filter fuses ``/vo`` differentially — it subtracts the previous message IT
    received from this one and divides by the gap — so what has to be continuous is the
    PUBLISHED stream, and the gate alone cannot make it so. A refused jump re-anchors the gate
    and nothing else: the filter still holds the pose from before the jump, and the next pose
    that passes hands it the whole discontinuity as one velocity. That is how a tracking restart
    at rtabmap's origin, three metres from where the cart was, became tens of metres per second
    of vy in the board's EKF — a velocity no wheel and no gyro measures, so nothing ever pulls it
    back (2026-09-14: odom -> base_link 43 km out, still travelling at 60 m/s a quarter of an
    hour after the topic went silent).

    So this carries the sum of the steps that passed, and across a refusal it simply stands
    still: a restart costs one sample of motion instead of a teleport. The published pose is
    absolute, which is also what makes a rate cap (:class:`PublishCap`) lossless — the filter
    differences whatever two messages reached it, and a skipped one only lengthens the gap.
    """

    def __init__(self) -> None:
        self._from: VoPose | None = None
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0

    def advance(self, pose: VoPose) -> VoPose:
        """Add an admitted pose's step to the running total; returns the pose to publish — the
        same stamp, the summed position and heading."""
        if self._from is not None:
            self._x += pose.x - self._from.x
            self._y += pose.y - self._from.y
            self._yaw = _wrapped(self._yaw + _wrapped(pose.yaw - self._from.yaw))
        self._from = pose
        return VoPose(stamp=pose.stamp, x=self._x, y=self._y, yaw=self._yaw)

    def anchor(self, pose: VoPose | None) -> None:
        """A pose the gate refused: the total stands still across it and the next step is
        measured from ``pose`` — the gate's own new anchor (:attr:`VoGate.anchor`)."""
        self._from = pose

    @property
    def pose(self) -> tuple[float, float, float]:
        """Where the published track stands now: metres, metres, radians since the node came up."""
        return self._x, self._y, self._yaw


class PublishCap:
    """How often a gated pose may leave for the board's EKF, and that its stamps only go forward.

    The rate is a load decision, not a quality one: the board's filter runs at 20 Hz on four
    A53s, and with 9 visual poses a second reaching it, it logged "Failed to meet update rate"
    continuously (it took 56-94 ms of every 50 ms period) while Nav2's costmaps sat at 200 % CPU
    (2026-09-14). A source measured at 9.4 poses/s carries the same distance in three.

    The stamp rule is not a rate at all but the invariant the differential fusion rests on: two
    messages the filter cannot order are a division by a gap of zero or a negative one. Both
    refusals are harmless to a :class:`VoTrack`, whose published pose is absolute.
    """

    def __init__(self, hz: float = 3.0) -> None:
        self.hz = hz  # live: the node's vo_publish_hz flag writes it; 0 publishes every pose
        self._last: float | None = None

    def refuse(self, stamp: float) -> str | None:
        """``None`` when a pose with this stamp may be published — and it is then remembered as
        the last published one — else the reason it may not."""
        if self._last is not None:
            if stamp <= self._last:
                return f"a stamp {self._last - stamp:.3f} s behind the last published"
            if self.hz > 0.0 and stamp - self._last < 1.0 / self.hz:
                return f"the {self.hz:.1f} Hz cap ({(stamp - self._last) * 1000:.0f} ms since)"
        self._last = stamp
        return None


@dataclass(frozen=True)
class RestDrift:
    """How far the visual odometry walked while the wheels stood still: metres, degrees and the
    seconds of stillness they took."""

    metres: float
    degrees: float
    seconds: float

    def __str__(self) -> str:
        return f"{self.metres * 100:.1f} cm and {self.degrees:.1f} deg in {self.seconds:.0f} s"


class RestWatch:
    """The visual odometry's drift at rest: what it says it moved while no wheel turned.

    A camera cannot be checked against a truth here — there is none on the robot — but it can be
    checked against physics: the cart is on its charger, the wheels report zero, and whatever the
    visual odometry accumulates in that minute is its own. Under a centimetre a minute it may be
    fused; a decimetre is a source that would walk the odometry away by itself.

    Fed the wheels' twist (``wheels``) and every pose that passed the gate (``pose``); ``drift``
    is the current stretch of stillness, ``worst`` the largest one this node has seen.

    Both are fed the RECEIVING node's own clock, not the messages' stamps: the wheels are
    stamped by the board and the visual poses by the laptop, and those two clocks have been
    seconds apart (the board ran 2.3-2.8 s ahead of the Mac on 2026-09-04; measured 0.13 s apart
    on 2026-09-14). A stillness window compared across them would either swallow a real drive or
    count the seconds a cart spends rocking after one. The drift's own duration is still the
    difference of two visual stamps, which are one clock's.

    And nothing is measured before a single wheel message has arrived: with no ``/odom`` — a
    dead bridge route, a QoS that never matched — the cart's stillness is not known, and a drift
    reported then would be a number about nothing (:meth:`report` says so instead).
    """

    def __init__(self, settle_s: float = REST_SETTLE_S) -> None:
        self._settle_s = settle_s
        self._heard_wheels = False
        self._moving_until: float | None = None
        self._anchor: VoPose | None = None
        self._drift: RestDrift | None = None
        self._worst: RestDrift | None = None

    def wheels(self, now: float, linear_m_s: float, yaw_rad_s: float) -> None:
        """One wheel-odometry twist, ``now`` by the receiving node's clock: while it is above
        the rest thresholds the drift is not being measured, and the next stretch of stillness
        starts from a fresh anchor."""
        self._heard_wheels = True
        if abs(linear_m_s) > REST_LINEAR_M_S or abs(yaw_rad_s) > REST_YAW_RAD_S:
            self._moving_until = now + self._settle_s
            self._anchor = None
            self._drift = None

    def restart(self) -> None:
        """The visual odometry's frame moved under the watch (a refused jump, a tracking
        restart): this stretch of stillness starts again from the next pose, because a drift
        measured across a re-initialised origin is not a drift."""
        self._anchor = None
        self._drift = None

    def pose(self, pose: VoPose, now: float) -> None:
        """One visual-odometry pose the gate admitted, ``now`` by the receiving node's clock;
        counted only once the wheels have been heard from and have been still for the settling
        time."""
        if not self._heard_wheels:
            return
        if self._moving_until is not None and now < self._moving_until:
            return
        if self._anchor is None:
            self._anchor = pose
            return
        drift = RestDrift(
            metres=math.hypot(pose.x - self._anchor.x, pose.y - self._anchor.y),
            degrees=abs(math.degrees(_wrapped(pose.yaw - self._anchor.yaw))),
            seconds=pose.stamp - self._anchor.stamp,
        )
        self._drift = drift
        if self._worst is None or drift.metres > self._worst.metres:
            self._worst = drift

    @property
    def drift(self) -> RestDrift | None:
        """The current stretch of stillness, or ``None`` while the cart moves (or has not stood
        still long enough for a second pose)."""
        return self._drift

    @property
    def worst(self) -> RestDrift | None:
        """The longest walk at rest this watch has seen, over the node's whole life."""
        return self._worst

    def report(self) -> str:
        """The drift in a few words for a report line."""
        if not self._heard_wheels:
            return "no /odom yet: nothing says whether the cart stands still, so no drift"
        if self._drift is None:
            return "moving (no drift measured)"
        worst = f", worst so far {self._worst}" if self._worst is not None else ""
        return f"at rest {self._drift}{worst}"


def _at_origin(pose: VoPose, radius_m: float) -> bool:
    """Whether a pose sits within ``radius_m`` of rtabmap's own origin — where a re-initialised
    ``rgbd_odometry`` puts it (``Odom/ResetCountdown``)."""
    return math.hypot(pose.x, pose.y) <= radius_m


def _wrapped(angle: float) -> float:
    """``angle`` folded into (-pi, pi]."""
    return math.remainder(angle, math.tau)
