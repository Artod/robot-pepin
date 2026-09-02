"""Driver for the differential-drive base: velocity commands and encoder travel.

The two wheel servos run in continuous-rotation (velocity) mode. Commands are
sent as encoder ticks per second in the servo's own sign convention; the
``direction`` of each :class:`~pepin.geometry.WheelMotor` maps that onto
"robot forward" for both commands and encoder readings.
"""

from __future__ import annotations

from types import TracebackType

from pepin.bus import MotorBus
from pepin.geometry import BaseConfig
from pepin.kinematics import DiffDriveKinematics, Twist
from pepin.odometry import EncoderUnwrapper

LEFT = "left"
RIGHT = "right"


def _clamp(value: float, limit: float) -> float:
    """``value`` restricted to the symmetric interval [-limit, +limit]."""
    return max(-limit, min(limit, value))


class DiffDriveBase:
    """Commands the two drive wheels and reports how far each has travelled."""

    def __init__(self, bus: MotorBus, config: BaseConfig) -> None:
        """``config`` fixes the geometry, the per-wheel sign convention and the speed limits."""
        self._bus = bus
        self._cfg = config
        self._kin = DiffDriveKinematics(config.geometry)
        self._direction = {LEFT: config.left.direction, RIGHT: config.right.direction}
        ticks = config.geometry.ticks_per_rev
        self._unwrap = {name: EncoderUnwrapper(ticks) for name in (LEFT, RIGHT)}

    @staticmethod
    def motor_ids(config: BaseConfig) -> dict[str, int]:
        """Name-to-id table for constructing the bus this driver will run on."""
        return {LEFT: config.left.motor_id, RIGHT: config.right.motor_id}

    def enable(self) -> None:
        """Power the wheels; they hold still until the first command."""
        self.stop()
        self._bus.enable_torque([LEFT, RIGHT])

    def disable(self) -> None:
        """Stop and release the wheels so the cart can be pushed by hand."""
        self.stop()
        self._bus.disable_torque([LEFT, RIGHT])

    def set_twist(self, twist: Twist) -> None:
        """Drive at the given body velocity (m/s, rad/s), clamped to the configured limits.

        One unacknowledged broadcast write: the wheels hold this velocity until the
        next command, so a stalled control loop leaves the base rolling.
        """
        safe = Twist(
            linear=_clamp(twist.linear, self._cfg.max_speed_m_s),
            angular=_clamp(twist.angular, self._cfg.max_yaw_rate_rad_s),
        )
        rates = self._kin.twist_to_wheels(safe)
        ticks = {
            LEFT: self._direction[LEFT] * self._kin.rad_s_to_ticks_s(rates.left),
            RIGHT: self._direction[RIGHT] * self._kin.rad_s_to_ticks_s(rates.right),
        }
        self._bus.sync_write("Goal_Velocity", ticks, normalize=False)

    def stop(self) -> None:
        """Command zero velocity to both wheels; torque stays on, so the base holds position."""
        self._bus.sync_write("Goal_Velocity", {LEFT: 0, RIGHT: 0}, normalize=False)

    def read_wheel_travel(self) -> tuple[float, float]:
        """Distance in meters each wheel rolled since the previous call (first call: zeros).

        Call this faster than the wheels cover half a revolution so that the
        encoder wrap can be resolved unambiguously.
        """
        raw = self._bus.sync_read("Present_Position", [LEFT, RIGHT], normalize=False)
        m_per_tick = self._cfg.geometry.m_per_tick
        return tuple(  # type: ignore[return-value]
            self._unwrap[name].delta(raw[name]) * self._direction[name] * m_per_tick
            for name in (LEFT, RIGHT)
        )

    def __enter__(self) -> DiffDriveBase:
        """Enter with the wheels powered and stopped."""
        self.enable()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop and release the wheels, including when the body raised."""
        self.disable()
