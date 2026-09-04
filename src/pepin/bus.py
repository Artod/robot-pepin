"""Servo bus abstraction.

Drivers depend on the small :class:`MotorBus` protocol rather than on a
concrete library, so they can be unit-tested against a fake and moved
between transports (direct USB via lerobot, TCP bridge via
:class:`pepin.feetech.FeetechTcpClient`) without touching their logic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class MotorBus(Protocol):
    """The subset of a servo bus that the drivers use; raw register units."""

    def sync_write(self, data_name: str, values: dict[str, int], *, normalize: bool = True) -> None:
        """Write one register on several motors at once, keyed by motor name."""
        ...

    def sync_read(
        self, data_name: str, motors: list[str], *, normalize: bool = True
    ) -> dict[str, int]:
        """Read one register from several motors in a single round trip, keyed by motor name."""
        ...

    def enable_torque(self, motors: list[str] | None = None) -> None:
        """Energise the listed motors, or every known motor when none are listed."""
        ...

    def disable_torque(self, motors: list[str] | None = None) -> None:
        """Release the listed motors, or every known motor when none are listed."""
        ...

    def ping(self, motor: str, num_retry: int = 0) -> int | None:
        """The motor's error byte if it answers, None if it stays silent."""
        ...


def verify_motors(
    bus: MotorBus,
    motors: list[str],
    *,
    attempts: int = 4,
    before_attempt: Callable[[], None] | None = None,
) -> None:
    """Ping each motor by address until it answers; raise naming the silent ones.

    A garbled reply that makes the transport raise counts as a miss;
    ``before_attempt`` (typically a buffer flush) runs before every try.
    """

    def answers(motor: str) -> bool:
        """One ping attempt; any transport error counts as no answer."""
        if before_attempt is not None:
            before_attempt()
        try:
            return bus.ping(motor) is not None
        except ConnectionError:
            raise  # a dead bridge is not a wiring fault; say so instead of "not answering"
        except Exception:
            return False

    silent = [m for m in motors if not any(answers(m) for _ in range(attempts))]
    if silent:
        raise ConnectionError(f"motors not answering after {attempts} pings each: {silent}")
