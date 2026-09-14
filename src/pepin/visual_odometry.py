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
error. :class:`RestWatch` is the measurement this module exists to make possible: with the
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
    "RestDrift",
    "RestWatch",
    "VoGate",
    "VoPose",
    "is_lost",
    "planar_covariance",
]

# What rtabmap writes on the diagonal of a pose it did not measure: its odometry nodes publish a
# message per frame either way, and the lost ones carry 9999 rather than a covariance.
LOST_VARIANCE = 9999.0
# Below these the cart stands still: the base's own caps are 0.30 m/s and 1.0 rad/s
# (pepin.deployment), and the wheels report a hard zero when no wheel turns, so this is a guard
# against a tick of quantisation rather than a threshold anything is tuned to.
REST_LINEAR_M_S = 0.01
REST_YAW_RAD_S = 0.02
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

    Two reasons, both cheap and both about the same failure: rtabmap losing its tracking. A lost
    frame it announces itself (:func:`is_lost`); a RE-START of the tracking it does not — the
    pose simply jumps, and a jump differenced into a velocity is the one thing that can yank an
    odometry frame nothing else in this stack can move. So a step faster than a cart that cannot
    exceed 0.30 m/s and 1.0 rad/s (pepin.deployment) is not a measurement of anything.

    ``admit`` answers ``None`` for a pose the node may publish, or the reason it may not.
    """

    def __init__(self, max_speed_m_s: float = 1.0, max_turn_deg_s: float = 180.0) -> None:
        self.max_speed_m_s = max_speed_m_s  # live: the node's vo_max_speed flag writes it
        self.max_turn_deg_s = max_turn_deg_s  # live: the node's vo_max_turn flag writes it
        self._last: VoPose | None = None

    def admit(self, pose: VoPose, lost: bool) -> str | None:
        """``None`` when this pose may go to the filter, else the reason it may not.

        A refused jump still becomes the anchor the next step is measured from. That is the
        whole point: rtabmap re-initialising its tracking (``Odom/ResetCountdown``) puts the
        pose back at its origin, and a gate that kept the old anchor would call every pose after
        that a jump and go deaf for the rest of the session. One restart costs one sample. A
        pose that arrives out of order is refused WITHOUT moving the anchor — there is nothing
        to re-anchor to in the past.
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
        step = math.hypot(pose.x - previous.x, pose.y - previous.y)
        turn = abs(_wrapped(pose.yaw - previous.yaw))
        if step / dt > self.max_speed_m_s:
            return f"a jump of {step * 100:.0f} cm in {dt:.3f} s"
        if math.degrees(turn) / dt > self.max_turn_deg_s:
            return f"a turn of {math.degrees(turn):.0f} deg in {dt:.3f} s"
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
    """

    def __init__(self, settle_s: float = REST_SETTLE_S) -> None:
        self._settle_s = settle_s
        self._moving_until: float | None = None
        self._anchor: VoPose | None = None
        self._drift: RestDrift | None = None
        self._worst: RestDrift | None = None

    def wheels(self, stamp: float, linear_m_s: float, yaw_rad_s: float) -> None:
        """One wheel-odometry twist: while it is above the rest thresholds the drift is not
        being measured, and the next stretch of stillness starts from a fresh anchor."""
        if abs(linear_m_s) > REST_LINEAR_M_S or abs(yaw_rad_s) > REST_YAW_RAD_S:
            self._moving_until = stamp + self._settle_s
            self._anchor = None
            self._drift = None

    def restart(self) -> None:
        """The visual odometry's frame moved under the watch (a refused jump, a tracking
        restart): this stretch of stillness starts again from the next pose, because a drift
        measured across a re-initialised origin is not a drift."""
        self._anchor = None
        self._drift = None

    def pose(self, pose: VoPose) -> None:
        """One visual-odometry pose the gate admitted; counted only while the wheels are still
        and have been for the settling time."""
        if self._moving_until is not None and pose.stamp < self._moving_until:
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
        if self._drift is None:
            return "moving (no drift measured)"
        worst = f", worst so far {self._worst}" if self._worst is not None else ""
        return f"at rest {self._drift}{worst}"


def _wrapped(angle: float) -> float:
    """``angle`` folded into (-pi, pi]."""
    return math.remainder(angle, math.tau)
