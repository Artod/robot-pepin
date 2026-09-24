"""The zero-velocity update of a parked cart: when the base bridge may tell the EKF that the cart
is not moving, and how much that claim weighs.

WHY, MEASURED. Parked on its charger on 2026-09-24 the EKF's heading crept ~0.08 deg/min, about
5 deg an hour, while the yaw-rate sources it fuses said, over 120 s: the gyro after the bias
tracker -0.001 deg/min, the wheels 0, the camera's VO -0.57, rf2o +1.5
(scratch/link_autopsy/rest_yaw_sources.py). The filter follows its inputs, and at rest one of
them is wrong. ros/params/ekf.yaml already fuses a zero-velocity update (odom2, /zupt: vx, vy,
vyaw), but its only publisher was the lidar tracker's slip watch, which does not start under
PEPIN_LOCALIZER=rtabmap -- so nothing ever told the filter what the wheels and the gyro both
knew.

THE CURE. The base bridge already knows when the cart stands still: the rest the WHEELS witness,
the same witness the gyro's bias tracker trusts (:mod:`pepin.gyro`). While that rest is certain
it publishes ``/zupt`` -- a twist of exactly zero -- and nothing at all otherwise. Certain means
three witnesses agree (:class:`ZuptGate`): the wheels have stood still past the settle window,
no non-zero /cmd_vel is fresh, and the bias-corrected gyro is quiet -- so a cart turned by hand
while its wheels stand still is not frozen by its own update.

THIS MODULE IS THE REFERENCE for ros/pepin_base_cpp/include/pepin_base_cpp/zupt.hpp, the copy
that runs on the board. tests/unit/test_zupt.py pins the contract both implement
(``test_zupt_contract_the_cpp_bridge_mirrors`` is written for exactly that purpose, and
ros/pepin_base_cpp/test/zupt_contract.cpp replays it); the maths changes here first.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# What the update claims on vx, vy and vyaw: a variance of 1e-6, a sigma of 1 mm/s and 1 mrad/s.
# Tight because the claim is true -- the gate below only lets it out while three witnesses agree
# that the cart is standing still -- and because a loose one is out-voted by the very source it
# exists to answer. Beside it the filter hears the gyro at 4e-4, rf2o at 2.5e-3 (vx 9e-4), the
# wheels at 1e-3 (vyaw 0.01) and the camera's differenced yaw at ~1.6e-3 (ros/params/ekf.yaml):
# 1e-6 is 400x under the tightest of them. The EKF's own structure sets the rest
# (scratch/zupt_variance_sim.py, the yaw half of ekf.yaml fed the sources' measured means at
# rest): the process noise regrows vyaw's uncertainty between updates, so an update tighter than
# ~1e-5 is already a reset of vyaw, and 1e-5, 1e-6 and 1e-8 cut the modelled rf2o pull alike
# (7.9x, 13.2x, 14.3x, median over phases, at 10 Hz) where 4e-4 -- the slip watch's own yaw
# sigma -- cuts it 1.3x. robot_localization raises any variance under 1e-9 to 1e-9
# (ekf.cpp:145), so 1e-6 is also well clear of the floor.
REST_ZUPT_VARIANCE = 1e-6
# ...and what it says about everything it does not claim (vz, vroll, vpitch; and the pose).
UNCLAIMED_VARIANCE = 1e6
# |bias-corrected yaw rate| at or above this is a turn: 0.005 rad/s = 0.29 deg/s. The parked
# chip's per-sample noise is 0.036 deg/s and the largest |yaw - bias| of the 30 s at-rest trace
# is 0.139 deg/s (scratch/zupt_gyro_quiet_threshold.py over scratch/imu_level_gyro.csv,
# 2026-09-12) -- 7.9 sigma and 2.1x the parked maximum -- while a slow hand quarter turn in 10 s
# (9 deg/s) is 31x over it.
GYRO_QUIET_RAD_S = 0.005
# How often the update is published while the cart is at rest: the slip watch's 10 Hz. Past 1e-5
# the RATE matters more than the variance (scratch/zupt_variance_sim.py): at 10 Hz the update is
# phase-locked to rf2o's ~10 Hz scans, and the modelled pull it leaves ranges with the phase
# between them from -0.012 to +0.018 deg/min against +0.020 without it (median 13x less); at 50 Hz
# it is 12x less at every phase, and the zero-mean walk of the sources' noise shrinks with it.
ZUPT_HZ = 10.0


class ZuptVerdict(Enum):
    """Why a zero-velocity update is, or is not, published at one tick; the value is the words
    the bridge's report line prints."""

    AT_REST = "at rest"
    COMMANDED = "a command is live"
    NO_REST = "the wheels do not witness rest"
    SETTLING = "settling after the last motion"
    NO_GYRO = "no gyro reading to judge by"
    GYRO_TURNING = "the gyro reports a turn"


@dataclass(frozen=True)
class RestEvidence:
    """Each witness's last word, as a time on the one monotonic clock the bridge's threads share;
    0 means never (or, for ``still_since``, not at rest).

    ``still_since`` is :func:`pepin.gyro.rest_witnessed`'s answer -- the start of the wheels' rest
    spell, already 0 when the witness is stale. ``command_at`` is the last non-zero /cmd_vel.
    ``gyro_at`` is the last bias-corrected gyro sample, ``gyro_turn_at`` the last one that
    :func:`gyro_turning` called a turn.
    """

    still_since: float = 0.0
    command_at: float = 0.0
    gyro_at: float = 0.0
    gyro_turn_at: float = 0.0


def gyro_turning(yaw_rate: float, quiet_rad_s: float) -> bool:
    """True when a bias-corrected yaw rate (rad/s) is a turn: at or above ``quiet_rad_s`` in
    magnitude. A NaN is a turn: a reading nobody can judge is not a quiet one."""
    return not abs(yaw_rate) < quiet_rad_s


class ZuptGate:
    """Whether the cart is CERTAINLY standing still at one moment -- the rule behind /zupt.

    Five vetoes, checked in this order, the first that holds being the verdict. A non-zero
    command younger than ``command_hold_s`` (the bridge's own ``cmd_timeout_s``: the age up to
    which it keeps re-sending a command to the board) -- the operator's intent is known before
    any wheel turns. No rest witnessed by the wheels (:class:`pepin.gyro.RestWitness`: they
    moved, a command is being applied, the stream broke, or nobody is watching). Rest witnessed
    for less than ``settle_s`` (``imu_bias_s``: the window the bias tracker waits for the chassis
    to stop rocking before it trusts a sample). No gyro sample younger than ``max_gap_s`` (no
    IMU, no bias yet, or reads failing): without it a cart turned by hand on still wheels cannot
    be told from a parked one, so there is no update. And a gyro turn -- the newest sample, or
    any within ``settle_s``: a turn is motion like any other and the settle window is served
    after it just as after a wheel's move.

    Pure and clockless, like the bias tracker: every time comes from the caller.
    """

    def __init__(self, settle_s: float, command_hold_s: float, max_gap_s: float) -> None:
        """``settle_s``: rest needed before the update, and the hold after a gyro turn.
        ``command_hold_s``: how long a non-zero command vetoes. ``max_gap_s``: how old the
        newest gyro sample may be (the wheels' own staleness is judged before, in
        :func:`pepin.gyro.rest_witnessed`)."""
        self._settle_s = settle_s
        self._command_hold_s = command_hold_s
        self._max_gap_s = max_gap_s

    def judge(self, now: float, evidence: RestEvidence) -> ZuptVerdict:
        """The verdict at monotonic time ``now``; only :attr:`ZuptVerdict.AT_REST` publishes."""
        if evidence.command_at > 0.0 and now - evidence.command_at < self._command_hold_s:
            return ZuptVerdict.COMMANDED
        if evidence.still_since <= 0.0:
            return ZuptVerdict.NO_REST
        if now - evidence.still_since < self._settle_s:
            return ZuptVerdict.SETTLING
        if evidence.gyro_at <= 0.0 or now - evidence.gyro_at > self._max_gap_s:
            return ZuptVerdict.NO_GYRO
        turn = evidence.gyro_turn_at
        if turn > 0.0 and (turn >= evidence.gyro_at or now - turn < self._settle_s):
            return ZuptVerdict.GYRO_TURNING
        return ZuptVerdict.AT_REST


def rest_zupt_twist_covariance() -> list[float]:
    """The row-major 6x6 twist covariance of the update: :data:`REST_ZUPT_VARIANCE` on vx, vy
    and vyaw -- the indices ekf.yaml's odom2 fuses -- and :data:`UNCLAIMED_VARIANCE` on the
    three it does not."""
    diagonal = (
        REST_ZUPT_VARIANCE,
        REST_ZUPT_VARIANCE,
        UNCLAIMED_VARIANCE,
        UNCLAIMED_VARIANCE,
        UNCLAIMED_VARIANCE,
        REST_ZUPT_VARIANCE,
    )
    matrix = [0.0] * 36
    for i, value in enumerate(diagonal):
        matrix[i * 6 + i] = value
    return matrix
