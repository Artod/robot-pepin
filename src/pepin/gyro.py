"""The MPU6050's zero, re-measured from every block of rest the wheels witness.

Lives beside the wheel code rather than inside :mod:`pepin.odometry` on purpose: odometry.py is
wheel geometry (tick unwrapping, pose integration, the twist differenced off two poses), this is
the IMU's own error, and the C++ side splits the same way (``mpu6050.hpp`` is not
``twist_from_pose.hpp``). What the two share is the wheels' word on standing still, which arrives
here as one timestamp.

THIS MODULE IS THE REFERENCE for ros/pepin_base_cpp/include/pepin_base_cpp/gyro_bias.hpp, the
copy that runs on the board. tests/unit/test_gyro.py pins the contract both implement
(``test_gyro_bias_contract_the_cpp_bridge_mirrors`` is written for exactly that purpose); the
maths changes here first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GyroBias:
    """A gyro zero in rad/s, on the chip's own axes (the bias is subtracted before mounting)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class RestWitness:
    """The wheels' word on whether the cart is standing still, boiled down to one timestamp.

    Fed one wheel state line at a time, it answers with the time since which rest has been
    witnessed without a break -- which is the time of the last motion when there is none, and 0
    when the question cannot be answered at all. A consumer on another thread needs nothing but
    that number and :func:`rest_witnessed`.

    A line witnesses rest only when all three hold. The MEASURED wheel twist is exactly zero: one
    encoder tick is pi * 0.125 m / 4096 = 9.6e-5 m of travel (config/base.json), so a cart that
    moves at all -- driven, pushed, or turned by hand with the torque off -- reports a non-zero
    twist. The base server is applying no twist: ``moving`` is its own word on a live non-zero
    command, whoever sent it, which is the case of a command whose blocked wheels never tick. And
    the line continues an unbroken stream: across a gap longer than ``max_gap_s`` the twist
    estimator re-primes and returns a zero twist that measured nothing, while the cart may have
    moved the whole time.
    """

    def __init__(self, max_gap_s: float) -> None:
        """``max_gap_s``: the same gap after which the twist estimator re-primes (TwistFromPose)."""
        self._max_gap_s = max_gap_s
        self._stamp_s = 0.0
        self._at = 0.0
        self._since = 0.0

    @property
    def since(self) -> float:
        """The monotonic time the current rest spell began; 0 when the cart is not known still."""
        return self._since

    @property
    def at(self) -> float:
        """The monotonic time of the last line judged -- how fresh the answer above is."""
        return self._at

    def forget(self) -> None:
        """Start the stream and the spell over: what comes next measured nothing about the past.

        Called wherever the twist estimator is re-primed -- /odom muted, say. Whatever makes the
        next twist not a measurement makes the rest it would witness unknown, and a zero twist
        that measured nothing must not be read as a cart standing still.
        """
        self._stamp_s = 0.0
        self._since = 0.0

    def judge(self, stamp_s: float, at: float, moving: bool, twist_is_zero: bool) -> float:
        """One wheel state line; returns the spell start to publish (see :attr:`since`).

        Two clocks on purpose: ``stamp_s`` is the board's own clock, on which the gap between
        consecutive state lines is measured, and ``at`` is the local monotonic clock the gyro's
        samples are stamped with, on which the rest is counted.
        """
        unbroken = self._stamp_s > 0.0 and 0.0 < stamp_s - self._stamp_s <= self._max_gap_s
        self._stamp_s = stamp_s
        self._at = at
        still = unbroken and twist_is_zero and not moving
        if not still or self._since <= 0.0:
            self._since = at
        return self._since


def rest_witnessed(since: float, at: float, now: float, max_gap_s: float) -> float:
    """The rest spell a reader on another thread may still believe in, or 0 if nobody is watching.

    ``since`` and ``at`` are :class:`RestWitness`'s two numbers as they were last published (two
    atomics in the C++ bridge, so the 50 Hz gyro loop never waits on the wheels' reader thread).
    A witness older than ``max_gap_s`` is no witness: the link may be down or /odom muted, and a
    cart nobody is watching is not at rest, however still the last thing anyone saw was.
    """
    if since <= 0.0 or now - at > max_gap_s:
        return 0.0
    return since


class GyroBiasTracker:
    """The gyro's zero, replaced by the mean of every finished block of rest.

    WHY, MEASURED. The bridge used to estimate the bias once, over the first ``imu_bias_s`` after
    start, and subtract that number forever. The chip's bias moves with temperature: parked hours
    after boot with the wheels blocked, /odom (the wheels) read 0.00 deg/min of yaw while
    /odometry/filtered -- the EKF, whose only yaw-rate source is this gyro -- read +0.19, -0.54
    and +0.67 deg/min in three measurements of one night (scratch/odom_drift_at_rest.py,
    2026-09-19), and RTAB-Map, which builds its graph on that odometry, turned its map +27 deg in
    40 min and +88 deg in four hours under a cart that never moved. On the level floor the chip's
    raw yaw axis reads -0.028 deg/s, which is -1.7 deg/min if integrated (scratch/imu_level_*.csv,
    2026-09-12).

    THE CURE. While the cart stands still the gyro's reading IS its bias, and the wheels know
    when it stands still -- their twist is differenced off consecutive encoder poses, and one tick
    is 9.6e-5 m of travel, so a cart that moves at all reports a non-zero twist. A REST BLOCK is
    ``block_s * rate_hz`` samples taken inside one unbroken spell of witnessed rest that began at
    least ``block_s`` before the first of them (the chassis settling). A finished block REPLACES
    the bias with its mean: one block, one mean, no gain and no time constant to tune.

    HOW QUIET THE MEAN IS, measured on the 30 s at-rest tape
    (scratch/gyro_block_mean_noise.py): per-sample noise 0.036 deg/s, and the mean of a 2.0 s
    block scatters by 0.30 deg/min (1 sigma) -- so the cure does not reach zero, it turns a
    one-way creep into a zero-mean random walk: 40 min of 2 s blocks accumulate ~0.35 deg against
    the +27 deg measured. A longer ``imu_bias_s`` shrinks the error in force at any one moment,
    which is what a DRIVE inherits (0.21 deg/min at 5 s, 0.02 at 10); the parked walk stays
    ~0.35 deg either way, the sqrt trade.

    Pure and clockless: every time comes from the caller, no thread and no lock lives here.
    """

    def __init__(self, block_s: float, rate_hz: float) -> None:
        """``block_s`` seconds of rest per block, at ``rate_hz`` samples a second.

        Both are the node's existing parameters (``imu_bias_s``, ``imu_rate_hz``): the block is
        the length the boot calibration already asked the operator to hold still for, and the
        settle window before it is the same length again. ``block_s`` <= 0 is "no calibration" --
        the zero bias is ready at once, as before.
        """
        self._settle_s = block_s
        self._samples = max(1, round(block_s * rate_hz)) if block_s > 0.0 else 0
        self._bias = GyroBias()
        self._ready = self._samples == 0
        self._blocks = 0
        self._at_s = 0.0
        self._spell = 0.0
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_z = 0.0
        self._count = 0

    @property
    def bias(self) -> GyroBias:
        """The zero to subtract from the chip's readings, rad/s."""
        return self._bias

    @property
    def ready(self) -> bool:
        """True once a bias exists; before that the caller publishes nothing."""
        return self._ready

    @property
    def blocks(self) -> int:
        """How many rest blocks have been averaged into a bias since the node started."""
        return self._blocks

    @property
    def block_samples(self) -> int:
        """How many samples one block holds; 0 when no calibration was asked for."""
        return self._samples

    def age_s(self, t: float) -> float:
        """Seconds since the last block finished; ``inf`` while none has."""
        return math.inf if self._blocks == 0 else t - self._at_s

    def update(
        self,
        t: float,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
        still_since: float,
    ) -> bool:
        """One gyro sample at monotonic time ``t``; True when a block just replaced the bias.

        ``still_since`` is the wheels' word: the monotonic time since which rest has been
        witnessed without a break, or <= 0 when the cart is not known to be standing still (it is
        moving, a command is live, or nobody is watching -- the link is down or /odom is muted).
        It is compared for exact equality, not nearness: it is the identity of a rest spell, not a
        measurement, and the caller repeats the same value for every sample of one spell.

        A sample is ignored while the chassis settles (less than ``block_s`` since the spell
        began) and a spell that ends throws away whatever it had accumulated -- so a block is
        never a mean across a move.
        """
        if self._samples == 0:
            return False
        if still_since <= 0.0 or t - still_since < self._settle_s:
            self._forget()
            self._spell = still_since
            return False
        if still_since != self._spell:
            self._forget()
            self._spell = still_since
        self._sum_x += gyro_x
        self._sum_y += gyro_y
        self._sum_z += gyro_z
        self._count += 1
        if self._count < self._samples:
            return False
        count = float(self._count)
        self._bias = GyroBias(self._sum_x / count, self._sum_y / count, self._sum_z / count)
        self._blocks += 1
        self._at_s = t
        self._ready = True
        self._forget()
        return True

    def _forget(self) -> None:
        """Throw away the block being accumulated; the bias already taken is untouched."""
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_z = 0.0
        self._count = 0
