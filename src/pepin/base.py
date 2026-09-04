"""Driver for the differential-drive base: velocity commands and encoder travel.

The two wheel servos run in continuous-rotation (velocity) mode. Commands are
sent as encoder ticks per second in the servo's own sign convention; the
``direction`` of each :class:`~pepin.geometry.WheelMotor` maps that onto
"robot forward" for both commands and encoder readings.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Literal

from pepin.bus import MotorBus
from pepin.geometry import BaseConfig
from pepin.kinematics import DiffDriveKinematics, Twist
from pepin.odometry import EncoderUnwrapper

logger = logging.getLogger(__name__)

LEFT = "left"
RIGHT = "right"


def _clamp(value: float, limit: float) -> float:
    """``value`` restricted to the symmetric interval [-limit, +limit]."""
    return max(-limit, min(limit, value))


@dataclass
class BusWatchdog:
    """Escalation policy for consecutive bus failures inside a control loop.

    The wheels hold their last velocity while the bus is silent, so a loop that
    keeps failing must first command a stop (after ``stop_after_s``) and then
    abort rather than hope (after ``give_up_after_s``). Shared by every drive
    script so the policy cannot drift between them.
    """

    stop_after_s: float = 1.0
    give_up_after_s: float = 10.0
    failing_since: float | None = field(default=None, init=False)
    failures: int = field(default=0, init=False)  # lifetime count, for the recovery log line
    stop_sent: bool = field(default=False, init=False)

    def failed(self, now: float) -> Literal["skip", "stop", "abort"]:
        """Record one failed tick at ``now`` and say what the loop must do about it."""
        self.failures += 1
        if self.failing_since is None:
            self.failing_since = now
        down = now - self.failing_since
        if down >= self.give_up_after_s:
            return "abort"
        if not self.stop_sent and down >= self.stop_after_s:
            self.stop_sent = True
            return "stop"
        return "skip"

    def recovered(self, now: float) -> float | None:
        """Record a good tick; returns the length of the outage that just ended, if any."""
        if self.failing_since is None:
            return None
        down = now - self.failing_since
        self.failing_since = None
        self.stop_sent = False
        return down

    def handle(self, base: DiffDriveBase, now: float, exc: Exception) -> None:
        """Apply the policy to one failed tick: warn, command a stop, or raise ``RuntimeError``."""
        verdict = self.failed(now)
        down = now - (self.failing_since if self.failing_since is not None else now)
        logger.warning(
            "bus failure: %s (down %.1f s, %d ticks lost so far): %s",
            verdict,
            down,
            self.failures,
            exc,
        )
        if verdict == "stop":
            with_suppressed_timeout(base.stop)
        elif verdict == "abort":
            raise RuntimeError(f"bus unreachable for {down:.0f} s") from exc


def with_suppressed_timeout(action: Callable[[], None]) -> None:
    """Run ``action``; a ``TimeoutError`` is logged, not raised (best-effort stop on a dead bus)."""
    try:
        action()
    except TimeoutError as exc:
        logger.warning("%s failed: %s", getattr(action, "__name__", "action"), exc)


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

    def reprime(self) -> None:
        """Forget the last encoder readings after a long bus outage.

        The unwrapper cannot tell more than half a wheel revolution of unseen
        motion from a small backward step, so after an outage the next read
        reports zero travel instead of an aliased jump into the odometry.
        """
        for unwrap in self._unwrap.values():
            unwrap.reset()

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
        """Stop and release the wheels, including when the body raised.

        Never raises: if the bus is dead there is nothing more to do here, and a
        second exception would only hide the one that ended the drive.
        """
        try:
            self.disable()
        except OSError as exc:
            logger.warning("could not stop the wheels on exit (bus down): %s", exc)
