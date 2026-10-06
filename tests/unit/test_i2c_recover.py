"""The I2C bus recovery: nine clocks and a STOP through the TWI line control register, against a
simulated bus whose slave holds SDA low for a number of clocks; the check's verdict over a window
of register reads, against a scripted register (pepin.i2c_recover)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pepin.i2c_recover import (
    RELEASED,
    SCL_CTL,
    SCL_EN,
    SCL_STATE,
    SDA_CTL,
    SDA_EN,
    SDA_STATE,
    Lines,
    Sampled,
    awake,
    clock_out,
    holders,
    sample,
)


class FakeBus:
    """TWI_LCR over a bus with one stuck slave: it drives SDA low until it has seen ``hold``
    rising SCL edges (None: for ever); ``scl_stuck`` is a slave holding the clock low."""

    def __init__(self, hold: int | None, scl_stuck: bool = False) -> None:
        self.reg = RELEASED
        self.hold = hold
        self.scl_stuck = scl_stuck
        self.rising = 0
        self.stops = 0
        self.writes: list[int] = []

    def _scl(self, reg: int) -> bool:
        return not self.scl_stuck and (not reg & SCL_EN or bool(reg & SCL_CTL))

    def _slave_holds(self) -> bool:
        return self.hold is None or self.rising < self.hold

    def _sda(self, reg: int) -> bool:
        master_low = bool(reg & SDA_EN) and not reg & SDA_CTL
        return not master_low and not self._slave_holds()

    def read(self) -> int:
        reg = self.reg & ~(SDA_STATE | SCL_STATE)
        return (
            reg
            | (SDA_STATE if self._sda(self.reg) else 0)
            | (SCL_STATE if self._scl(self.reg) else 0)
        )

    def write(self, value: int) -> None:
        scl_before, sda_before = self._scl(self.reg), self._sda(self.reg)
        self.writes.append(value)
        scl_after = self._scl(value)
        if not scl_before and scl_after:
            self.rising += 1
        self.reg = value & ~(SDA_STATE | SCL_STATE)
        if scl_before and scl_after and not sda_before and self._sda(self.reg):
            self.stops += 1


def _no_sleep(_: float) -> None:
    return None


def test_an_idle_bus_is_left_alone() -> None:
    bus = FakeBus(hold=0)
    result = clock_out(bus, _no_sleep)
    assert result.freed and result.clocks == 0 and not result.stop_sent
    assert bus.writes == []


@pytest.mark.parametrize("hold", [1, 3, 9])
def test_a_slave_stuck_mid_byte_is_clocked_free_and_the_bus_ends_on_a_stop(hold: int) -> None:
    bus = FakeBus(hold=hold)
    result = clock_out(bus, _no_sleep)
    assert result.before == Lines(sda=False, scl=True)
    assert result.clocks == hold
    assert result.stop_sent and bus.stops == 1
    assert result.freed
    assert bus.reg == RELEASED, "both lines handed back to the controller"
    assert "bus FREE" in result.line()


def test_a_slave_that_never_lets_go_gets_nine_clocks_and_no_stop() -> None:
    bus = FakeBus(hold=None)
    result = clock_out(bus, _no_sleep)
    assert result.clocks == 9 and not result.stop_sent and not result.freed
    assert bus.reg == RELEASED
    assert "STILL HELD" in result.line()


def test_a_clock_held_low_is_not_clocked() -> None:
    bus = FakeBus(hold=None, scl_stuck=True)
    result = clock_out(bus, _no_sleep)
    assert result.clocks == 0 and not result.freed
    assert bus.rising == 0
    assert bus.reg == RELEASED
    assert "SCL held low" in result.after.verdict()


def test_the_lines_read_from_the_vendor_idle_value() -> None:
    assert Lines.of(0x3A) == Lines(sda=True, scl=True)  # TWI_LCR_IDLE_STATUS
    assert Lines.of(0x2A) == Lines(sda=False, scl=True)
    assert "stuck mid-byte" in Lines.of(0x2A).verdict()


def test_holders_finds_the_processes_with_the_bus_open(tmp_path: Path) -> None:
    device = tmp_path / "i2c-2"
    device.touch()
    for pid, name, target in ((101, "python", device), (202, "base_bridge", tmp_path / "x")):
        (tmp_path / str(pid) / "fd").mkdir(parents=True)
        (tmp_path / str(pid) / "comm").write_text(name + "\n")
        os.symlink(target, tmp_path / str(pid) / "fd" / "3")
    (tmp_path / "self").mkdir()
    assert holders(str(device), tmp_path) == ["101 python"]


def test_awake_holds_the_controller_on_and_puts_the_setting_back(tmp_path: Path) -> None:
    power = tmp_path / "power"
    power.mkdir()
    (power / "control").write_text("auto\n")
    (power / "runtime_status").write_text("active\n")
    with awake(tmp_path) as status:
        assert (power / "control").read_text() == "on"
        assert status == "active"
    assert (power / "control").read_text() == "auto"


IDLE = 0x3A  # TWI_LCR_IDLE_STATUS: both lines high, both handed to the controller
SCL_LOW = 0x1A  # what the single-read check saw on the live board, 2026-10-05 21:27
SDA_LOW = 0x2A
BOTH_LOW = 0x0A


class ScriptedLcr:
    """A line register that answers reads from a script, cycling it; a check never writes."""

    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.reads = 0

    def read(self) -> int:
        value = self.values[self.reads % len(self.values)]
        self.reads += 1
        return value

    def write(self, value: int) -> None:
        raise AssertionError(f"the check wrote 0x{value:02x} to the register")


class FakeClock:
    """A monotonic clock that only moves when slept on."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now


def _check(values: list[int]) -> tuple[Sampled, ScriptedLcr]:
    lcr = ScriptedLcr(values)
    fake = FakeClock()
    return sample(lcr, sleep=fake.sleep, clock=fake.clock), lcr


def test_check_reads_the_register_fifty_times_a_millisecond_apart() -> None:
    lcr = ScriptedLcr([IDLE])
    fake = FakeClock()
    result = sample(lcr, sleep=fake.sleep, clock=fake.clock)
    assert lcr.reads == 50 and len(result.values) == 50
    assert fake.sleeps == [0.001] * 49
    assert result.window_s == pytest.approx(0.049)


def test_check_on_an_idle_bus_says_idle() -> None:
    result, _ = _check([IDLE])
    assert not result.held
    assert result.line("active") == (
        "TWI_LCR 0x3a (runtime active, 50 reads over 49 ms): idle: SDA and SCL high"
    )


def test_check_calls_scl_held_low_when_every_read_agrees() -> None:
    result, _ = _check([SCL_LOW])
    assert result.held and result.scl_low == 50
    assert "SCL held low" in result.line("active")


def test_check_calls_sda_held_low_when_every_read_has_it_low_under_a_high_clock() -> None:
    result, _ = _check([SDA_LOW])
    assert result.held
    assert "stuck mid-byte" in result.verdict()


@pytest.mark.parametrize(
    "script",
    [
        [SCL_LOW] + [IDLE] * 49,  # the board's false alarm: the one read landed mid-transfer
        [IDLE, SCL_LOW, BOTH_LOW, SDA_LOW, IDLE, IDLE, IDLE, IDLE],  # transfers in flight
        [SCL_LOW] * 49 + [IDLE],  # one high read is enough: the clock moves
        [SDA_LOW] * 49 + [BOTH_LOW],  # SDA always low, but the clock is being driven
    ],
)
def test_check_on_a_busy_bus_is_alive_not_held(script: list[int]) -> None:
    result, _ = _check(script)
    assert not result.held
    verdict = result.verdict()
    assert verdict.startswith("busy: ")
    assert f"SCL low in {result.scl_low} of 50 reads" in verdict
    assert result.line("active").startswith(f"TWI_LCR 0x{script[0]:02x} (runtime active, 50 reads")


def test_check_needs_at_least_one_read() -> None:
    with pytest.raises(ValueError):
        sample(ScriptedLcr([IDLE]), reads=0)
