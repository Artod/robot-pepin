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

A VISUAL-INERTIAL source (OpenVINS on the head IMU and the two eyes, vio.md) enters through the
same gate and track: :func:`compose_base_pose` turns its IMU pose in its own gravity frame into
base_link's through the neck's TF chain, :data:`VIO_STEP_FRACTION` is its per-step covariance
model (its own marginal covariance only grows and is read as a health signal, :class:`VioHealth`),
and :class:`VioLost` says when it has diverged (it never resets itself); :class:`VioGuard` refuses
every single sample no cart of this speed could have made and says when to restart it. As a BODY
VELOCITY instead of a pose (the relay's ``vo_output twist``), the composed step becomes
:func:`se2_twist` and its weight is OpenVINS's own velocity covariance, matched by stamp
(:class:`TwistCovariances`) and turned into base_link's axes (:func:`base_twist_covariance`). At
the IMU's rate instead (the relay's ``vio_twist_source imu``), odomimu's own velocity state and
gyro are carried into base_link through the neck (:func:`imu_base_twist`), sampled evenly
(:class:`EvenRate`, :class:`ImuQueue`) with the neck differenced over the step
(:class:`NeckBaseline`). While the VIO is lost only its yaw rate is sent, its variance grown by
the gyro bias's walk since the last visual update (:class:`YawOnly`,
:func:`yaw_only_covariance`); :func:`scaled_twist_covariance` is the relay's sigma scales.

Nothing here is ROS: poses and wheel speeds in, verdicts and a report line out
(:mod:`pepin_bringup.visual_odometry` is the node around it).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "LOST_VARIANCE",
    "REST_LINEAR_M_S",
    "REST_YAW_RAD_S",
    "SCALE_ERROR",
    "SIGMA_FLOOR_M",
    "VIO_IMU_HALF_SAMPLE_S",
    "VIO_IMU_LAG_S",
    "VIO_LOST_RULE",
    "VIO_LOST_RULES",
    "VIO_LOST_S",
    "VIO_LOST_SPEED_M_S",
    "VIO_MAX_SPEED_M_S",
    "VIO_MIN_FEATURES",
    "VIO_NECK_DT_FLOOR_S",
    "VIO_NECK_GAP_S",
    "VIO_REINIT_RATIO",
    "VIO_RESTART_GAP_S",
    "VIO_RESTART_REJECTS",
    "VIO_RESTART_REST_S",
    "VIO_REST_SPEED_M_S",
    "VIO_STEP_FRACTION",
    "VIO_TWIST_HORIZON_S",
    "VIO_TWIST_MATCH_S",
    "VIO_WHEEL_DIFF_M_S",
    "WEIGHTLESS_VARIANCE",
    "EvenRate",
    "ImuQueue",
    "ImuState",
    "NeckBaseline",
    "PublishCap",
    "RestDrift",
    "RestWatch",
    "TwistCovariances",
    "VioGuard",
    "VioHealth",
    "VioLost",
    "VoGate",
    "VoPose",
    "VoTrack",
    "YawOnly",
    "base_twist_covariance",
    "body_velocity",
    "compose_base_pose",
    "homogeneous",
    "imu_base_twist",
    "is_lost",
    "planar_covariance",
    "rotation_vector",
    "scaled_covariance",
    "scaled_twist_covariance",
    "se2_twist",
    "weightless_covariance",
    "yaw_only_covariance",
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

# The VIO's drift per metre of step, the `vio` covariance mode's scale error (vio.md M3): 3 % is
# an ESTIMATE until Phase 1 measures it; the floor beside it is vo_sigma_m (one wheel sample at
# 10 Hz, scratch/vo_weight.py), so a VIO pose never weighs more than a wheel sample.
VIO_STEP_FRACTION = 0.03
# A step change of OpenVINS's marginal covariance trace by this factor is a re-initialisation.
VIO_REINIT_RATIO = 10.0
# The three LOST rules (vio.md section 4, S9): the composed base speed disagreeing with the
# wheels by this much for this long while the board does not say rest; moving faster than
# VIO_REST_SPEED_M_S while /zupt says rest; fewer features than this while the base moves.
VIO_LOST_SPEED_M_S = 0.1
VIO_LOST_S = 1.0
VIO_MIN_FEATURES = 20
VIO_REST_SPEED_M_S = 0.03
# Which features the third rule counts: `used`, the features OpenVINS's last update used (MSCKF +
# SLAM), or `tracked`, the tracker's persistent tracks (the left eye's features seen in 3+ frames,
# /ov_msckf/health, the keeper's dark count). The used ones fall to 0-8 whenever the head moves
# or the scene is plain while the tracks stay 20-45 (drives 0343-0363, moving: 46 % of the frames
# under 20 used, 7 % under 20 persistent outside a head swing); but a filter with few features
# in its update coasts on the IMU, and the twists `tracked` adds are the coasting ones: replayed
# on those drives 3.12 instead of 2.52 full twists per moving second, the added ones' vx sigma
# multiplier 1.76 [1.42, 2.59] against 0.84 [0.70, 1.02], 19.5 % of them off the lidar truth by
# more than 0.2 m/s against 1.1 % (scratch/vio_lost_rule/replay.py). So `used` stays the default.
VIO_LOST_RULES = ("used", "tracked")
VIO_LOST_RULE = "used"
# /zupt is published while the board's witnesses agree on rest; heard within this, it holds.
ZUPT_FRESH_S = 0.5
# A wheel speed or a feature count older than this says nothing any more.
WITNESS_FRESH_S = 1.0
# The plausibility guard on the VIO's OUTPUT (:class:`VioGuard`): a composed base velocity this
# cart cannot make (the wheels top out at 0.32 m/s, the base cap is 0.30), or one this far from
# the wheels', is a diverged filter, not a measurement. OpenVINS diverged on the first fast head
# pan of 2026-10-04's live run (7.7 m inside the move, then km) and never re-initialised.
VIO_MAX_SPEED_M_S = 1.0
VIO_WHEEL_DIFF_M_S = 0.2  # 0.5 let 8-11 samples a drive through on 2026-10-04 (max 0.29)
# That many implausible samples in a row (2 s at the camera's 10 Hz) with the wheels at rest for
# VIO_RESTART_REST_S restarts it: its static initialisation needs stillness (vio_config.py).
VIO_RESTART_REJECTS = 20
VIO_RESTART_REST_S = 2.0
# Not sooner than this after the previous restart: the respawn takes 2 s and the initialisation
# waits for the next motion, and a loop of restarts would starve it of both.
VIO_RESTART_GAP_S = 10.0
# The variance that tells robot_localization "no weight" on an axis it is configured to read
# (planar_covariance's ``unfused``): the twist axes the EKF does not fuse, and a whole pose whose
# weight has moved to the twist.
WEIGHTLESS_VARIANCE = 1e6
# OpenVINS's odomimu comes once per head-IMU sample (200 Hz, 5 ms apart) stamped in the IMU's
# clock, the same clock as poseimu's camera time + t_d: a velocity covariance is matched to a pose
# when it is the nearest within four samples. A matching rule, never a weight.
VIO_TWIST_MATCH_S = 0.02
# How much odomimu history is kept for that match: poseimu arrives 0.15-0.3 s after its stamp
# (the update waits for the image), odomimu as the IMU sample lands.
VIO_TWIST_HORIZON_S = 2.0
# THE TWIST AT THE IMU'S RATE (the relay's ``vio_twist_source imu``): odomimu carries OpenVINS's
# velocity STATE and the bias-corrected gyro once per head-IMU sample (200 Hz nominal, 5 ms; 163 Hz
# on 2026-10-05's drives, gaps to 30 ms). Its POSES cannot be differenced at that rate: each is
# the last update's state propagated on the IMU alone, and every update moves the base of that
# propagation, so the pose stream jumps once per camera frame -- differenced over ~20 ms the pair
# that straddles an update reads |dp/dt - v| 7-11 cm/s p50, 33-38 cm/s p90 and 2.2-2.5 m/s p99
# (drives 0329/0330, scratch/vio_relay_night/odomimu_steps.py), against 0.1-0.2 cm/s p50 for the
# pairs inside one propagation. So the 50 Hz twist is odomimu's own velocity and rate, carried
# into base_link through the neck (:func:`imu_base_twist`). A sample is taken when it is at least
# one period minus half an IMU sample after the last one taken (:class:`EvenRate`), so the neck's
# rate is never differenced over less than ~17 ms (the floor below) nor across a gap longer than
# VIO_NECK_GAP_S (a mean over a gap is not a rate).
VIO_IMU_HALF_SAMPLE_S = 0.0025
VIO_NECK_DT_FLOOR_S = 0.010
VIO_NECK_GAP_S = 0.1
# How far behind the newest odomimu a sample is taken: its neck chain must already be in the TF
# buffer (the bridge's camera edge at 50 Hz crosses the same WiFi as the head IMU), so the lookup
# never waits and a 50 Hz loop never blocks. The twist then reaches the EKF ~0.1 s after its
# stamp, against 0.15-0.3 s for poseimu's step (the update waits for the image).
VIO_IMU_LAG_S = 0.1


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

    def reanchor(self, pose: VoPose) -> None:
        """A pose the node may not use but the next step must start from — a frame taken while
        the head turned (:mod:`pepin.gaze_gate`): it becomes the anchor, as a refused jump does,
        so the motion across it reaches nobody. One older than the anchor moves nothing."""
        if self._last is None or pose.stamp > self._last.stamp:
            self._last = pose

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

    The steps are composed in SE(2), ``track = track * (from^-1 * pose)``: each step is taken in
    the BODY frame of the pose it starts from and replayed from the track's own heading. The
    filter differences ``prev.inverseTimes(cur)`` (robot_localization, ros_filter.cpp), i.e. a
    body-frame step, so the track's heading must be the one that step was measured in. Summing
    x and y in the source's axes (until 2026-10-03) told the EKF vy +0.200 m/s for a cart driving
    straight at 0.2 m/s whenever the source's frame started at yaw 90 deg (a VIO's gravity frame
    has an arbitrary yaw), and vx +0.173 / vy +0.100 m/s for the rest of the session after one
    30 deg turn inside a gated interval (scratch/head_imu/votrack_frame.py).
    """

    def __init__(self) -> None:
        self._from: VoPose | None = None
        self._step: tuple[VoPose, VoPose] | None = None
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0

    def advance(self, pose: VoPose) -> VoPose:
        """Add an admitted pose's step to the running total; returns the pose to publish — the
        same stamp, the summed position and heading."""
        self._step = None if self._from is None else (self._from, pose)
        if self._from is not None:
            dx, dy = pose.x - self._from.x, pose.y - self._from.y
            c, s = math.cos(self._from.yaw), math.sin(self._from.yaw)
            forward, left = c * dx + s * dy, -s * dx + c * dy  # the step in the body frame
            c, s = math.cos(self._yaw), math.sin(self._yaw)
            self._x += c * forward - s * left
            self._y += s * forward + c * left
            self._yaw = _wrapped(self._yaw + _wrapped(pose.yaw - self._from.yaw))
        self._from = pose
        return VoPose(stamp=pose.stamp, x=self._x, y=self._y, yaw=self._yaw)

    def anchor(self, pose: VoPose | None) -> None:
        """A pose the gate refused: the total stands still across it and the next step is
        measured from ``pose`` — the gate's own new anchor (:attr:`VoGate.anchor`)."""
        self._from = pose
        self._step = None

    @property
    def last_step(self) -> tuple[VoPose, VoPose] | None:
        """The step the last :meth:`advance` added, ``(from, to)`` in the source's frame;
        ``None`` when it only set the anchor or an anchor came after it."""
        return self._step

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

    def __init__(self, hz: float = 3.0, burst: float = 2.0) -> None:
        self.hz = hz  # live: the node's vo_publish_hz flag writes it; 0 publishes every pose
        self.burst = burst  # poses that may go out back to back after a pause
        self._last: float | None = None
        self._tokens = burst
        self._refilled: float | None = None

    def refuse(self, stamp: float) -> str | None:
        """``None`` when a pose with this stamp may be published — and it is then remembered as
        the last published one — else the reason it may not.

        The rate is a BUDGET, not a minimum gap: a token bucket that refills at ``hz`` and holds
        ``burst``. The board's EKF needs the AVERAGE held down (it missed its 20 Hz at 9.4 poses a
        second, 2026-09-14); it does not need poses spaced evenly. The visual odometry releases
        its pairs in bunches — exact sync pairs a camera frame with the depth frame made from it
        and hands them over as they come — and a minimum gap of 1/hz threw a third of them away
        at an average of 5.8 a second (3.9 published, 2026-09-17), for a board that could take all
        of them. Every pose the slip watch does not get is a slower verdict."""
        if self._last is not None and stamp <= self._last:
            return f"a stamp {self._last - stamp:.3f} s behind the last published"
        if self.hz > 0.0:
            gap = 0.0 if self._refilled is None else max(stamp - self._refilled, 0.0)
            self._tokens = min(self.burst, self._tokens + gap * self.hz)
            self._refilled = stamp
            if self._tokens < 1.0:
                return f"the {self.hz:.1f} Hz budget (a burst of {self.burst:.0f} spent)"
            self._tokens -= 1.0
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


def homogeneous(rotation: Any, translation: Any) -> Any:
    """A 4x4 transform from a 3x3 rotation and a translation."""
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(rotation, dtype=float)
    matrix[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return matrix


def compose_base_pose(t_g_i: Any, t_i_b: Any, stamp: float) -> VoPose:
    """base_link's planar pose in the VIO's gravity frame G: ``T_G_B = T_G_I * T_I_B``.

    ``t_g_i`` is the IMU's pose in G (OpenVINS's poseimu), ``t_i_b`` base_link's pose in the
    IMU's frame at the same stamp (TF ``head_imu <- base_link``: the static head_imu edge, the
    neck's encoders, the mast's sway). The yaw is ``atan2(R[1,0], R[0,0])`` of the COMPOSED
    rotation (vio.md N3: never "IMU yaw minus pan" — the 23.8 deg mount pitch and the tilt make
    that subtraction wrong). With the cart parked and the head panning, T_G_I and T_I_B change
    in opposite ways and the product stands still to the TF's accuracy."""
    t_g_b = np.asarray(t_g_i, dtype=float) @ np.asarray(t_i_b, dtype=float)
    yaw = math.atan2(float(t_g_b[1, 0]), float(t_g_b[0, 0]))
    return VoPose(stamp=stamp, x=float(t_g_b[0, 3]), y=float(t_g_b[1, 3]), yaw=yaw)


class VioHealth:
    """OpenVINS's own covariance read as a health signal, never as a weight: its marginal pose
    covariance only grows (global x, y and yaw are unobservable), so a STEP of its trace by
    ``reinit_ratio`` either way is a re-initialisation, counted for the report line. So is a new
    frame name: OpenVINS's in-process reset (ros/patches/openvins-reset.patch) starts a new
    gravity frame and names it (``global``, then ``global_<n>``), and a warm-seeded filter's
    trace need not step at all."""

    def __init__(self, reinit_ratio: float = VIO_REINIT_RATIO) -> None:
        self.reinit_ratio = reinit_ratio
        self.reinits = 0
        self.last_trace: float | None = None
        self.last_frame: str | None = None

    def observe(self, covariance: Sequence[float], frame: str = "") -> bool:
        """One poseimu covariance (6x6 row-major) and its frame: True when it marks a
        re-initialisation."""
        renamed = bool(frame) and self.last_frame is not None and frame != self.last_frame
        if frame:
            self.last_frame = frame
        if len(covariance) < 36:
            if renamed:
                self.reinits += 1
            return renamed
        trace = sum(float(covariance[i * 6 + i]) for i in range(6))
        previous, self.last_trace = self.last_trace, trace
        stepped = False
        if previous is not None and previous > 0.0 and trace > 0.0:
            stepped = max(trace / previous, previous / trace) > self.reinit_ratio
        if renamed or stepped:
            self.reinits += 1
            return True
        return False


class VioLost:
    """When the VIO has diverged: OpenVINS does not reset itself when lost, it propagates on the
    IMU and drifts — at 0.5 m/s that passes the gate's 1 m/s ceiling on a cart capped at 0.30.

    Three rules, each counted (vio.md section 4): (a) the composed base speed disagrees with the
    wheels by more than ``lost_speed_m_s`` for ``lost_s`` while the board does not say rest; (b)
    it moves faster than ``rest_speed_m_s`` while ``/zupt`` (wheels, gyro and command witnessed)
    says rest; (c) fewer than ``min_features`` features for ``lost_s`` while the base moves
    (``/zupt`` silent), never at rest. ``rule`` says which features (c) counts: ``tracked``, the
    tracker's persistent tracks (:meth:`tracked`), what the picture still holds; or ``used``, the
    features OpenVINS's last update used (:meth:`features`), which a moving head or a plain
    scene takes to 0-8 in good light and which are 0 at rest (no track ends there: 0 of 300 poses
    on 2026-10-04 with the head still). A still cart is not lost for seeing nothing new. Times
    are the receiving node's clock."""

    def __init__(
        self,
        lost_speed_m_s: float = VIO_LOST_SPEED_M_S,
        lost_s: float = VIO_LOST_S,
        min_features: int = VIO_MIN_FEATURES,
        rest_speed_m_s: float = VIO_REST_SPEED_M_S,
        rule: str = VIO_LOST_RULE,
    ) -> None:
        self.lost_speed_m_s = lost_speed_m_s  # live: vio_lost_speed_m_s
        self.lost_s = lost_s  # live: vio_lost_s
        self.min_features = min_features  # live: vio_min_features
        self.rest_speed_m_s = rest_speed_m_s
        self.rule = rule  # live: vio_lost_rule
        self._wheels: tuple[float, float] | None = None  # (now, |speed|)
        self._zupt: float | None = None
        self._features: tuple[float, int] | None = None  # the used count
        self._tracked: tuple[float, int] | None = None  # the persistent tracks
        self._disagree_since: float | None = None
        self._few_since: float | None = None  # since when few features AND the base moving
        self.counts = {"wheels": 0, "rest": 0, "features": 0}
        self.last: str | None = None
        self._was_lost = False

    def wheels(self, now: float, speed_m_s: float) -> None:
        """The wheels' forward speed (m/s) from /odom."""
        self._wheels = (now, abs(speed_m_s))

    def zupt(self, now: float) -> None:
        """A /zupt message: the board says the cart is at rest."""
        self._zupt = now

    def features(self, now: float, count: int) -> None:
        """How many features the VIO used in its last update (the node sums OpenVINS's MSCKF and
        SLAM features): rule (c)'s count under ``used``."""
        self._features = (now, count)

    def tracked(self, now: float, count: int) -> None:
        """How many persistent tracks the VIO's tracker holds on its last frame (health's
        ``persistent``): rule (c)'s count under ``tracked``."""
        self._tracked = (now, count)

    @property
    def feature_count(self) -> int | None:
        """The last count heard under the rule in force, for the report line."""
        counted = self._counted()
        return None if counted is None else counted[1]

    def _counted(self) -> tuple[float, int] | None:
        """The ``(time, count)`` rule (c) reads: the tracks under ``tracked``, else the used."""
        return self._tracked if self.rule == "tracked" else self._features

    def updated(self, now: float) -> bool:
        """Whether a sample that passed the rules had a visual update behind it, the moment the
        yaw-only mode's bias walk starts from (:meth:`YawOnly.visual`). Under ``used`` every
        passing sample (the rule itself allows at most ``lost_s`` of few used features); under
        ``tracked`` only one whose last update, heard within ``WITNESS_FRESH_S``, used at least
        ``min_features`` (at least one): a filter that still tracks but does not update coasts on
        the IMU, and its gyro bias walks as in the dark."""
        if self.rule != "tracked":
            return True
        used = self._features
        return (
            used is not None
            and now - used[0] <= WITNESS_FRESH_S
            and used[1] >= max(self.min_features, 1)
        )

    def check(self, now: float, vio_speed_m_s: float) -> str | None:
        """Why the VIO is lost now, or ``None``; a new episode is counted under its rule."""
        reason, rule = self._reason(now, abs(vio_speed_m_s))
        if reason is not None and not self._was_lost and rule is not None:
            self.counts[rule] += 1
            self.last = reason
        self._was_lost = reason is not None
        return reason

    def _reason(self, now: float, speed: float) -> tuple[str | None, str | None]:
        features = self._counted()
        what = "tracked features" if self.rule == "tracked" else "features"
        at_rest = self._zupt is not None and now - self._zupt <= ZUPT_FRESH_S
        few = (
            self.min_features > 0
            and features is not None
            and now - features[0] <= WITNESS_FRESH_S
            and features[1] < self.min_features
        )
        if few and not at_rest:
            if self._few_since is None:
                self._few_since = now
            if now - self._few_since >= self.lost_s:
                assert features is not None
                return (
                    f"{features[1]} {what} (under {self.min_features}) for"
                    f" {now - self._few_since:.1f} s while moving",
                    "features",
                )
        else:
            self._few_since = None
        if at_rest:
            self._disagree_since = None
            if speed > self.rest_speed_m_s:
                return f"moving at {speed:.2f} m/s while /zupt says rest", "rest"
            return None, None
        wheels = self._wheels
        if wheels is None or now - wheels[0] > WITNESS_FRESH_S:
            self._disagree_since = None
            return None, None
        if abs(speed - wheels[1]) <= self.lost_speed_m_s:
            self._disagree_since = None
            return None, None
        if self._disagree_since is None:
            self._disagree_since = now
        if now - self._disagree_since >= self.lost_s:
            return (
                f"{speed:.2f} m/s against the wheels' {wheels[1]:.2f} for"
                f" {now - self._disagree_since:.1f} s",
                "wheels",
            )
        return None, None

    def report(self) -> str:
        """The lost episodes by rule and the last reason, for the report line."""
        total = sum(self.counts.values())
        last = f" (last: {self.last})" if self.last else ""
        return (
            f"lost {total} (wheels {self.counts['wheels']}, rest {self.counts['rest']},"
            f" features {self.counts['features']}){last}"
        )


def body_velocity(previous: VoPose, pose: VoPose) -> tuple[float, float] | None:
    """The step from ``previous`` to ``pose`` as a body-frame velocity (forward, left), m/s, in
    the heading of ``previous``; ``None`` when the stamps do not advance."""
    dt = pose.stamp - previous.stamp
    if dt <= 0.0:
        return None
    dx, dy = pose.x - previous.x, pose.y - previous.y
    c, s = math.cos(previous.yaw), math.sin(previous.yaw)
    return (c * dx + s * dy) / dt, (-s * dx + c * dy) / dt


def se2_twist(previous: VoPose, pose: VoPose) -> tuple[float, float, float] | None:
    """The constant body twist (forward m/s, left m/s, yaw rad/s) that carries ``previous`` to
    ``pose`` over their stamps' difference: the SE(2) logarithm of the step in ``previous``'s body
    frame, so a cart on an arc reads its own forward speed with no sideways leak (the chord in
    the starting heading leaks ``v * w * dt / 2`` sideways); ``None`` when the stamps do not
    advance."""
    dt = pose.stamp - previous.stamp
    if dt <= 0.0:
        return None
    dx, dy = pose.x - previous.x, pose.y - previous.y
    c, s = math.cos(previous.yaw), math.sin(previous.yaw)
    forward, left = c * dx + s * dy, -s * dx + c * dy
    turn = _wrapped(pose.yaw - previous.yaw)
    if abs(turn) < 1e-9:
        return forward / dt, left / dt, 0.0
    # The step is V * (vx, vy) * dt with V = [[a, -b], [b, a]]; V^-1 = [[a, b], [-b, a]] / (a^2+b^2)
    a, b = math.sin(turn) / turn, (1.0 - math.cos(turn)) / turn
    norm = a * a + b * b
    return (a * forward + b * left) / norm / dt, (a * left - b * forward) / norm / dt, turn / dt


def weightless_covariance() -> list[float]:
    """A 6x6 row-major covariance that robot_localization fuses with no weight on any axis."""
    matrix = [0.0] * 36
    for i in range(6):
        matrix[i * 6 + i] = WEIGHTLESS_VARIANCE
    return matrix


# The twist axes robot_localization reads for a planar cart: vx, vy and vyaw (indices of the
# 6x6 linear-then-angular block).
_PLANAR_TWIST = (0, 1, 5)


def base_twist_covariance(
    imu_covariance: Sequence[float], r_b_i: Any, lever: Any = None
) -> list[float] | None:
    """OpenVINS's twist covariance turned into base_link's axes, for the three the EKF fuses.

    ``imu_covariance`` is odomimu's 6x6 row-major twist covariance: the IMU's linear velocity
    and its angular velocity, both in the IMU's own axes (Propagator::fast_state_propagate);
    ``r_b_i`` is base_link <- head_imu at the same stamp. The result is ``J C J^T`` with ``J``
    the rows vx, vy, vyaw of ``R6 = diag(R_B_I, R_B_I)``, kept with their cross terms and
    :data:`WEIGHTLESS_VARIANCE` on every other axis. ``lever`` (the IMU's position in base_link,
    metres) adds what :func:`imu_base_twist` does with the yaw rate: ``vx += wz * py``,
    ``vy -= wz * px``, so the gyro's noise reaches vx and vy through the arm; ``None`` is the
    composed step's case (no arm term). ``None`` when the source is not a covariance: too short, a
    non-finite entry, or a kept 3x3 block that is not positive definite (OpenVINS before its
    first update, a zero matrix)."""
    if len(imu_covariance) < 36:
        return None
    source = np.asarray(list(imu_covariance)[:36], dtype=float).reshape(6, 6)
    if not np.all(np.isfinite(source)):
        return None
    r = np.asarray(r_b_i, dtype=float)
    jacobian = np.zeros((3, 6))
    jacobian[0, :3], jacobian[1, :3], jacobian[2, 3:] = r[0], r[1], r[2]
    if lever is not None:
        px, py = float(lever[0]), float(lever[1])
        jacobian[0, 3:] += py * r[2]
        jacobian[1, 3:] -= px * r[2]
    block = jacobian @ source @ jacobian.T
    block = 0.5 * (block + block.T)
    if not np.all(np.linalg.eigvalsh(block) > 0.0):
        return None
    matrix = weightless_covariance()
    for a, i in enumerate(_PLANAR_TWIST):
        for b, j in enumerate(_PLANAR_TWIST):
            matrix[i * 6 + j] = float(block[a, b])
    return matrix


class TwistCovariances:
    """OpenVINS's velocity covariances by stamp (odomimu, one per IMU sample), kept for
    ``horizon_s`` so the pose of the same instant can find its own: :meth:`nearest` is the one
    closest to a stamp, within ``match_s``. A stamp that goes backwards (a restarted OpenVINS)
    starts the history again."""

    def __init__(
        self, match_s: float = VIO_TWIST_MATCH_S, horizon_s: float = VIO_TWIST_HORIZON_S
    ) -> None:
        self.match_s = match_s
        self.horizon_s = horizon_s
        self._items: deque[tuple[float, tuple[float, ...]]] = deque()
        self.received = 0

    def add(self, stamp: float, covariance: Sequence[float]) -> None:
        """One odomimu twist covariance (6x6 row-major) at its stamp, seconds."""
        items = self._items
        if items and stamp < items[-1][0]:
            items.clear()
        items.append((stamp, tuple(float(v) for v in covariance)))
        while items and items[0][0] < stamp - self.horizon_s:
            items.popleft()
        self.received += 1

    def nearest(self, stamp: float) -> tuple[float, ...] | None:
        """The covariance whose stamp is closest to ``stamp`` within ``match_s``, or ``None``."""
        best: tuple[float, tuple[float, ...]] | None = None
        for item in reversed(self._items):
            gap = abs(item[0] - stamp)
            if gap <= self.match_s and (best is None or gap < abs(best[0] - stamp)):
                best = item
            if item[0] < stamp - self.match_s:
                break
        return None if best is None else best[1]


def rotation_vector(rotation: Any) -> Any:
    """The axis times the angle (rad) of a 3x3 rotation: its logarithm as a vector."""
    r = np.asarray(rotation, dtype=float)
    skew = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    cosine = min(max((float(np.trace(r)) - 1.0) / 2.0, -1.0), 1.0)
    angle = math.acos(cosine)
    if angle < 1e-6:
        return 0.5 * skew
    return skew * (angle / (2.0 * math.sin(angle)))


def imu_base_twist(
    velocity: Any, rate: Any, t_b_i: Any, t_b_i_before: Any, dt: float
) -> tuple[float, float, float] | None:
    """base_link's planar twist (forward m/s, left m/s, yaw rad/s) from OpenVINS's IMU state.

    ``velocity`` is odomimu's twist.linear (the IMU's velocity in G, in the IMU's own axes),
    ``rate`` its twist.angular (the bias-corrected gyro, IMU axes); ``t_b_i`` is base_link <-
    head_imu (4x4) at the sample's stamp and ``t_b_i_before`` the same ``dt`` seconds earlier: the
    neck's (and the mast's) own motion over the step, ``R_rel = R(t) R(t - dt)^T`` and
    ``p_dot = (p(t) - p(t - dt)) / dt`` in base axes. Rigid-body kinematics with a planar base
    (``w_B = (0, 0, wz)``: its roll and pitch are the TF's, as the mast filter models them):

        wz  = (R_B_I w_I)_z - log(R_rel)_z / dt
        v_B = R_B_I v_I - w_B x p_B_I - p_dot

    ``None`` when ``dt`` is under :data:`VIO_NECK_DT_FLOOR_S` (a neck rate differenced over less
    is the encoders' quantisation, 0.088 deg a tick)."""
    if dt < VIO_NECK_DT_FLOOR_S:
        return None
    now, before = np.asarray(t_b_i, dtype=float), np.asarray(t_b_i_before, dtype=float)
    r, p = now[:3, :3], now[:3, 3]
    neck_rate = rotation_vector(r @ before[:3, :3].T) / dt
    p_dot = (p - before[:3, 3]) / dt
    wz = float((r @ np.asarray(rate, dtype=float))[2] - neck_rate[2])
    v = r @ np.asarray(velocity, dtype=float)
    forward = float(v[0] + wz * p[1] - p_dot[0])
    left = float(v[1] - wz * p[0] - p_dot[1])
    return forward, left, wz


@dataclass(frozen=True)
class ImuState:
    """One odomimu sample: its stamp (s), OpenVINS's IMU velocity and bias-corrected rate (both
    in the IMU's own axes) and their 6x6 row-major covariance."""

    stamp: float
    velocity: tuple[float, float, float]
    rate: tuple[float, float, float]
    covariance: tuple[float, ...]


class EvenRate:
    """Which stamps of a stream to take for ``hz`` evenly spaced samples: one at least
    ``1 / hz - VIO_IMU_HALF_SAMPLE_S`` after the last one taken and strictly later (a token
    bucket would hand out two IMU samples 5 ms apart after any pause); ``hz`` 0 takes every
    stamp that moves forward."""

    def __init__(self, hz: float) -> None:
        self.hz = hz  # live: the relay's vio_publish_hz writes it
        self._last: float | None = None

    def take(self, stamp: float) -> bool:
        """Whether this stamp is taken; it then becomes the last one taken."""
        last = self._last
        if last is not None:
            if stamp <= last:
                return False
            if self.hz > 0.0 and stamp - last < 1.0 / self.hz - VIO_IMU_HALF_SAMPLE_S:
                return False
        self._last = stamp
        return True

    def reset(self) -> None:
        """Forget the last stamp (a restarted source, whose clock may start again)."""
        self._last = None


class ImuQueue:
    """odomimu samples waiting for their neck chain: :meth:`due` hands out, oldest first, the
    ones at least ``lag_s`` behind the newest that :class:`EvenRate` takes, and drops the rest as
    they pass. A stamp that goes backwards (a restarted OpenVINS) empties it."""

    def __init__(self, hz: float, lag_s: float = VIO_IMU_LAG_S) -> None:
        self.rate = EvenRate(hz)
        self.lag_s = lag_s
        self._items: deque[ImuState] = deque()

    def add(self, sample: ImuState) -> None:
        """One odomimu sample, in arrival order."""
        if self._items and sample.stamp <= self._items[-1].stamp:
            self._items.clear()
            self.rate.reset()
        self._items.append(sample)

    def due(self) -> list[ImuState]:
        """The samples to process now: lag_s behind the newest and taken by the rate."""
        items = self._items
        if not items:
            return []
        horizon = items[-1].stamp - self.lag_s
        out = []
        while items and items[0].stamp <= horizon:
            sample = items.popleft()
            if self.rate.take(sample.stamp):
                out.append(sample)
        return out

    def clear(self) -> None:
        """Drop every waiting sample (another input took over)."""
        self._items.clear()
        self.rate.reset()


class NeckBaseline:
    """The neck chain at the last sample taken, for the neck's own rate over the step to the
    next: :meth:`step` answers the earlier transform and the seconds between, or ``None`` when
    there is nothing to difference against (the first sample, a gap over ``gap_s``). A step
    under ``floor_s`` answers ``None`` and keeps the older baseline, so the next one differences
    over a longer step; otherwise the new transform becomes the baseline."""

    def __init__(self, floor_s: float = VIO_NECK_DT_FLOOR_S, gap_s: float = VIO_NECK_GAP_S) -> None:
        self.floor_s = floor_s
        self.gap_s = gap_s
        self._last: tuple[float, Any] | None = None

    def step(self, stamp: float, t_b_i: Any) -> tuple[Any, float] | None:
        """``(t_b_i_before, dt)`` for this sample, or ``None``."""
        last = self._last
        if last is not None and 0.0 < stamp - last[0] < self.floor_s:
            return None
        self._last = (stamp, t_b_i)
        if last is None:
            return None
        dt = stamp - last[0]
        if dt <= 0.0 or dt > self.gap_s:
            return None
        return last[1], dt

    def reset(self) -> None:
        """Nothing to difference against until the next sample (a re-init, an input switch)."""
        self._last = None


def scaled_twist_covariance(
    covariance: Sequence[float], linear_scale: float, yaw_scale: float
) -> list[float]:
    """``S C S`` with ``S`` = ``linear_scale`` on vx and vy and ``yaw_scale`` on vyaw: the sigmas
    of the three fused axes multiplied by the relay's ``vio_sigma_scale`` and
    ``vio_yaw_sigma_scale``, a cross term by the product of its two scales (the matrix stays a
    covariance), every other entry untouched."""
    out = [float(v) for v in covariance]
    scale = {0: float(linear_scale), 1: float(linear_scale), 5: float(yaw_scale)}
    for i, si in scale.items():
        for j, sj in scale.items():
            out[i * 6 + j] *= si * sj
    return out


def yaw_only_covariance(covariance: Sequence[float], bias_variance: float) -> list[float]:
    """A twist covariance that keeps only the yaw rate: vx and vy at
    :data:`WEIGHTLESS_VARIANCE` with their cross terms 0 (robot_localization fuses them with a
    gain of ~1e-8: nothing), vyaw's own variance plus ``bias_variance`` (:meth:`YawOnly.variance`).
    """
    out = [float(v) for v in covariance]
    for i in (0, 1):
        for j in (0, 1, 5):
            out[i * 6 + j] = out[j * 6 + i] = 0.0
        out[i * 6 + i] = WEIGHTLESS_VARIANCE
    out[35] += float(bias_variance)
    return out


class YawOnly:
    """Since when the VIO has only its yaw rate to give, and what that rate is worth.

    OpenVINS lost (no features in the dark, the lost rules of :class:`VioLost`) still
    propagates on the IMU: its velocity then drifts on the accelerometer and is withheld, but its
    rate is the gyro less the bias the last visual updates estimated, and the gyro needs no
    light. What the dark costs the rate is the bias's walk since the last visual update: after
    ``t`` seconds its sigma is ``brw * sqrt(t)`` (``brw`` the head IMU's gyro random walk,
    rad/s/sqrt(s)), added to the rate's own variance:

        sigma_yaw(t)^2 = sigma_reported^2 + brw^2 * t

    ``brw`` ``None`` (config/head_imu.json unreadable) refuses: no yaw is sent while lost. Stamps
    are the samples' own (the IMU's clock)."""

    def __init__(self, brw: float | None) -> None:
        self.brw = brw
        self._visual: float | None = None  # the last sample with visual support
        self._since: float | None = None  # the start of the lost episode
        self.reason: str | None = None
        self.episodes = 0

    def visual(self, stamp: float, updated: bool = True) -> None:
        """A sample that passed the lost rules: the episode ends, and when ``updated`` (a visual
        update behind it, :meth:`VioLost.updated`) the visual updates vouch for the bias again;
        otherwise the bias walk keeps counting from the last sample that had one."""
        if updated and (self._visual is None or stamp > self._visual):
            self._visual = stamp
        self._since = None

    def lost(self, stamp: float, reason: str) -> None:
        """A sample the lost rules refused: the episode starts (once) and its reason is kept."""
        if self._since is None:
            self._since = stamp
            self.episodes += 1
        self.reason = reason

    @property
    def active(self) -> bool:
        """Whether the VIO is lost now (yaw only)."""
        return self._since is not None

    def age(self, stamp: float) -> float:
        """Seconds since the last visual sample at ``stamp`` (since the episode's start when the
        VIO never had one); 0 while not lost."""
        if self._since is None:
            return 0.0
        base = self._visual if self._visual is not None else self._since
        return max(stamp - base, 0.0)

    def variance(self, stamp: float) -> float | None:
        """The bias's share ``brw^2 * t`` (rad/s)^2 at ``stamp``; ``None`` without a ``brw``."""
        if self.brw is None:
            return None
        return self.brw**2 * self.age(stamp)

    def reset(self) -> None:
        """A new filter (a re-init, a restart, an input switch): nothing is lost, nothing seen."""
        self._visual = None
        self._since = None


class VioGuard:
    """The plausibility guard on the VIO's output: a composed base velocity faster than
    ``max_speed_m_s``, or farther than ``wheel_diff_m_s`` from the wheels' (forward speed, no
    sideways motion), is a diverged filter and the sample is not sent. Unlike :class:`VioLost`
    it judges every sample on its own, without a duration, so a divergence of metres per second
    never reaches the gate. ``wheel_rule`` False (the relay's ``vio_guard`` off) judges the speed
    alone: the VIO may then disagree with slipping wheels, and only a divergence is refused.

    It also says when to restart the VIO: ``restart_rejects`` implausible samples in a row while
    the wheels have said rest for ``rest_s`` (OpenVINS never resets itself, and its static
    initialisation needs stillness), not sooner than ``restart_gap_s`` after the last restart;
    ``restart_rejects`` 0 never restarts. Times are the receiving node's clock."""

    def __init__(
        self,
        max_speed_m_s: float = VIO_MAX_SPEED_M_S,
        wheel_diff_m_s: float = VIO_WHEEL_DIFF_M_S,
        restart_rejects: int = VIO_RESTART_REJECTS,
        rest_s: float = VIO_RESTART_REST_S,
        restart_gap_s: float = VIO_RESTART_GAP_S,
        wheel_rule: bool = True,
    ) -> None:
        self.max_speed_m_s = max_speed_m_s  # live: vio_max_speed_m_s
        self.wheel_diff_m_s = wheel_diff_m_s  # live: vio_wheel_diff_m_s
        self.wheel_rule = wheel_rule  # live: the relay's vio_guard flag
        self.restart_rejects = restart_rejects  # live: vio_restart_rejects
        self.rest_s = rest_s
        self.restart_gap_s = restart_gap_s
        self._wheels: tuple[float, float] | None = None  # (now, forward m/s)
        self._rest_since: float | None = None
        self._restart_at: float | None = None
        self.consecutive = 0
        self.counts = {"speed": 0, "wheels": 0}
        self.restarts = 0
        self.last: str | None = None
        self.last_restart: str | None = None

    def wheels(self, now: float, linear_m_s: float, yaw_rad_s: float) -> None:
        """One /odom twist: the wheels' forward speed and whether the cart stands still."""
        self._wheels = (now, linear_m_s)
        if abs(linear_m_s) >= REST_LINEAR_M_S or abs(yaw_rad_s) >= REST_YAW_RAD_S:
            self._rest_since = None
        elif self._rest_since is None:
            self._rest_since = now

    def at_rest_s(self, now: float) -> float:
        """How long the wheels have said rest, seconds; 0 while they move or are silent."""
        wheels = self._wheels
        if self._rest_since is None or wheels is None or now - wheels[0] > WITNESS_FRESH_S:
            return 0.0
        return now - self._rest_since

    def check(self, now: float, velocity: tuple[float, float] | None) -> str | None:
        """Why this sample is implausible, or ``None``; a rejection is counted under its rule
        and lengthens the run of rejections in a row, a pass ends it. ``None`` velocity (the
        first sample, a stamp that did not advance) is not judged and changes nothing."""
        if velocity is None:
            return None
        forward, left = velocity
        speed = math.hypot(forward, left)
        reason: str | None = None
        rule = ""
        wheels = self._wheels
        if speed > self.max_speed_m_s:
            reason, rule = f"{speed:.2f} m/s (over {self.max_speed_m_s:.2f})", "speed"
        elif self.wheel_rule and wheels is not None and now - wheels[0] <= WITNESS_FRESH_S:
            diff = math.hypot(forward - wheels[1], left)
            if diff > self.wheel_diff_m_s:
                reason = (
                    f"{forward:+.2f}/{left:+.2f} m/s against the wheels' {wheels[1]:+.2f}"
                    f" (over {self.wheel_diff_m_s:.2f})"
                )
                rule = "wheels"
        if reason is None:
            self.consecutive = 0
            return None
        self.consecutive += 1
        self.counts[rule] += 1
        self.last = reason
        return reason

    def restart_due(self, now: float) -> str | None:
        """The reason to restart the VIO now, or ``None``; a restart it answers is counted, and
        the run of rejections starts again from zero."""
        if self.restart_rejects <= 0 or self.consecutive < self.restart_rejects:
            return None
        rest = self.at_rest_s(now)
        if rest < self.rest_s:
            return None
        if self._restart_at is not None and now - self._restart_at < self.restart_gap_s:
            return None
        reason = (
            f"{self.consecutive} implausible samples in a row (last: {self.last}), the wheels at"
            f" rest {rest:.1f} s"
        )
        self._restart_at = now
        self.restarts += 1
        self.consecutive = 0
        self.last_restart = reason
        return reason

    def report(self) -> str:
        """The rejections by rule, the run in a row and the restarts, for the report line."""
        last = f" (last: {self.last})" if self.last else ""
        restart = f" (last: {self.last_restart})" if self.last_restart else ""
        rule = "" if self.wheel_rule else ", the wheel rule off"
        return (
            f"guard rejected {sum(self.counts.values())} (speed {self.counts['speed']}, wheels"
            f" {self.counts['wheels']}{rule}){last}, {self.consecutive} in a row, restarts"
            f" {self.restarts}{restart}"
        )


def _at_origin(pose: VoPose, radius_m: float) -> bool:
    """Whether a pose sits within ``radius_m`` of rtabmap's own origin — where a re-initialised
    ``rgbd_odometry`` puts it (``Odom/ResetCountdown``)."""
    return math.hypot(pose.x, pose.y) <= radius_m


def _wrapped(angle: float) -> float:
    """``angle`` folded into (-pi, pi]."""
    return math.remainder(angle, math.tau)
