"""The mast's sway from the head gyro: the Python twin of ros/pepin_base_cpp's mast.hpp.

The neck column bends between the shelf top and the pan servo; after a hard tilt the camera rings
at 5.3 Hz, 0.3-0.4 deg p-p for 0.5-1.0 s, and wheel jerks shake it too (2026-10-02). The neck's
encoders cannot see it; the head gyro sees the cart's rates, the neck joints' rates and the sway,
and the sway is what is left (vio.md section 5). Per head sample: the head rate in base_link's
axes minus the base gyro's YAW rate only, HELD while the neck moves (the joint rates are then
unknown; the output is NaN), ARMED at the first still sample, integrated with a leak
(``theta += omega dt - theta dt / tau``, ``tau = 1 / (2 pi crossover)``: the encoders are the
low-frequency truth), and for ``arm_window_s`` after arming published minus its running mean over
one ring period (the mast is already deflected when the encoders settle).

The bridge composes theta INTO its one ``base_link -> camera_link`` edge as a rotation about the
hinge point (:func:`compose_sway`) behind the ``mast_sway`` flag, and publishes ``/mast/state``
for the gaze gate. This module is the reference the C++ is held to (tests/unit's contract test)
and the offline tool for the sway measurement on recorded drives. Standard library only.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

Vector = tuple[float, float, float]
Matrix = tuple[float, ...]  # 3x3 row-major


@dataclass(frozen=True)
class MastSettings:
    """The filter's tunables (config/head_imu.json's ``mast`` block)."""

    crossover_hz: float = 0.5
    arm_window_s: float = 0.5
    ring_hz: float = 5.3
    max_dt_s: float = 0.05


@dataclass(frozen=True)
class MastOutput:
    """One output: sway angles (rad) and rates (rad/s) about base_link's axes, or ``held``
    (both NaN) while the neck moves or nothing vouches for them."""

    theta: Vector
    omega: Vector
    held: bool


NAN3: Vector = (math.nan, math.nan, math.nan)


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> Matrix:
    """The rotation yaw * pitch * roll, row-major (mast.hpp rotation_from_rpy)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp, cp * sr, cp * cr,
    )  # fmt: skip


def multiply(a: Matrix, b: Matrix) -> Matrix:
    """a * b of two row-major 3x3."""
    return tuple(
        sum(a[i * 3 + k] * b[k * 3 + j] for k in range(3)) for i in range(3) for j in range(3)
    )


def apply(m: Matrix, v: Sequence[float]) -> Vector:
    """m * v."""
    return (
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    )


def head_rate_in_base(
    head_rate: Sequence[float], camera_from_imu: Matrix, pan_rad: float, pitch_rad: float
) -> Vector:
    """The head's rate in base_link's axes: through R(camera_link <- head_imu) and the neck's
    Rz(pan) Ry(pitch) (mast.hpp head_rate_in_base)."""
    return apply(multiply(rotation_from_rpy(0.0, pitch_rad, pan_rad), camera_from_imu), head_rate)


def compose_sway(
    theta: Sequence[float],
    hinge: Sequence[float],
    neck_xyz: Sequence[float],
    neck_rpy: Sequence[float],
) -> tuple[Vector, Matrix]:
    """T(H) R(theta) T(-H) T_neck: the edge's translation and rotation (row-major)."""
    sway = rotation_from_rpy(theta[0], theta[1], theta[2])
    lever = [neck_xyz[i] - hinge[i] for i in range(3)]
    turned = apply(sway, lever)
    xyz = (hinge[0] + turned[0], hinge[1] + turned[1], hinge[2] + turned[2])
    return xyz, multiply(sway, rotation_from_rpy(*neck_rpy))


@dataclass
class MastFilter:
    """The leaky integrator with its hold, its arming and its arming-window mean."""

    settings: MastSettings = field(default_factory=MastSettings)
    held: bool = True
    _has_last: bool = False
    _last_t: float = 0.0
    _armed_at: float = 0.0
    _theta: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    _window: deque[tuple[float, Vector]] = field(default_factory=deque)
    _sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def hold(self) -> None:
        """The neck moved or nothing vouches: theta back to 0, held."""
        self.held = True
        self._theta = [0.0, 0.0, 0.0]
        self._window = deque()
        self._sum = [0.0, 0.0, 0.0]
        self._has_last = False

    @property
    def raw_theta(self) -> Vector:
        """The integrator's own theta, without the arming correction."""
        return (self._theta[0], self._theta[1], self._theta[2])

    def update(self, t: float, head_rate_base: Sequence[float], base_yaw_rate: float) -> MastOutput:
        """One sample (mast.hpp MastFilter::update)."""
        omega = (head_rate_base[0], head_rate_base[1], head_rate_base[2] - base_yaw_rate)
        if not all(math.isfinite(v) for v in (*omega, t)):
            self.hold()
            return MastOutput(NAN3, NAN3, True)
        gap = self._has_last and (t <= self._last_t or t - self._last_t > self.settings.max_dt_s)
        if self.held or not self._has_last or gap:
            self.hold()
            self.held = False
            self._armed_at = t
            self._last_t = t
            self._has_last = True
            self._remember(t)
            return self._output(t, omega)
        dt = t - self._last_t
        self._last_t = t
        tau = 1.0 / (2.0 * math.pi * self.settings.crossover_hz)
        for axis in range(3):
            self._theta[axis] += omega[axis] * dt - self._theta[axis] * dt / tau
        self._remember(t)
        return self._output(t, omega)

    def _remember(self, t: float) -> None:
        theta = self.raw_theta
        self._window.append((t, theta))
        for axis in range(3):
            self._sum[axis] += theta[axis]
        period = 1.0 / self.settings.ring_hz if self.settings.ring_hz > 0.0 else 0.0
        while self._window and t - self._window[0][0] >= period:
            _, old = self._window.popleft()
            for axis in range(3):
                self._sum[axis] -= old[axis]

    def _output(self, t: float, omega: Vector) -> MastOutput:
        arming = t - self._armed_at < self.settings.arm_window_s and bool(self._window)
        n = len(self._window)
        theta = tuple(
            self._theta[axis] - (self._sum[axis] / n if arming else 0.0) for axis in range(3)
        )
        return MastOutput((theta[0], theta[1], theta[2]), omega, False)


def ring_rate(t: float, amplitude_rad: float, hz: float, phase: float) -> float:
    """The rate of a ring ``amplitude sin(2 pi hz t + phase)``: for tests and the offline tool."""
    omega = 2.0 * math.pi * hz
    return amplitude_rad * omega * math.cos(omega * t + phase)
