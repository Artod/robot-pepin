"""The retrying handshake against a fake bus with a flaky and a dead motor."""

import pytest

from pepin.bus import verify_motors


class FlakyBus:
    """Answers a ping only every ``period``-th call per motor; ``dead`` never answers."""

    def __init__(self, period: int, dead: set[str] = frozenset()) -> None:
        self._period = period
        self._dead = dead
        self._calls: dict[str, int] = {}

    def ping(self, motor: str, num_retry: int = 0) -> int | None:
        if motor in self._dead:
            return None
        n = self._calls[motor] = self._calls.get(motor, 0) + 1
        return 777 if n % self._period == 0 else None

    def sync_write(self, data_name: str, values: dict[str, int], *, normalize: bool = True) -> None:
        raise NotImplementedError

    def sync_read(
        self, data_name: str, motors: list[str], *, normalize: bool = True
    ) -> dict[str, int]:
        raise NotImplementedError

    def enable_torque(self, motors: list[str] | None = None) -> None:
        raise NotImplementedError

    def disable_torque(self, motors: list[str] | None = None) -> None:
        raise NotImplementedError


def test_flaky_link_passes_when_a_retry_gets_through() -> None:
    verify_motors(FlakyBus(period=3), ["left", "right"], attempts=4)


def test_dead_motor_is_named_in_the_error() -> None:
    with pytest.raises(ConnectionError, match=r"\['right'\]"):
        verify_motors(FlakyBus(period=1, dead={"right"}), ["left", "right"], attempts=2)


def test_too_few_attempts_for_a_flaky_motor_fails() -> None:
    with pytest.raises(ConnectionError):
        verify_motors(FlakyBus(period=5), ["left"], attempts=2)


class GarblingBus(FlakyBus):
    """First reply per motor is garbage (SDK raises), then it answers."""

    def ping(self, motor: str, num_retry: int = 0) -> int | None:
        n = self._calls[motor] = self._calls.get(motor, 0) + 1
        if n == 1:
            raise IndexError("truncated packet")
        return 777


def test_garbled_reply_counts_as_a_miss_and_is_retried() -> None:
    flushes: list[int] = []
    verify_motors(
        GarblingBus(period=1), ["left"], attempts=3, before_attempt=lambda: flushes.append(1)
    )
    assert len(flushes) == 2  # one flush before the garbled try, one before the good one
