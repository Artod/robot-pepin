"""Free a locked I2C bus on the board without a power cycle: nine clocks and a STOP.

The base IMU and the three VL53L1X share ``i2c-2`` (TWI3 at 0x05002c00, header pins 3 SDA / 5
SCL). When a slave loses its place mid-byte it keeps driving SDA low, waiting for clocks the
master never sends; the controller then cannot make a START, every transfer times out after 2 s
(``mv64xxx: I2C bus locked`` in dmesg), and the bus stays dead. The mainline driver asks the I2C
core to recover the bus at that moment, but on this SoC the core has no way to drive the pins: the
generic recovery needs ``scl-gpios`` and a ``gpio`` pinctrl state, and sunxi's strict pinmux
refuses a GPIO request on pins muxed to the controller (board/README.md, "The I2C bus").

The controller itself can drive its lines: Allwinner's TWI Line Control Register (TWI_LCR, offset
0x20; bit 0 SDA_EN, 1 SDA_CTL, 2 SCL_EN, 3 SCL_CTL, 4 SDA state, 5 SCL state; the vendor kernel's
``twi_send_clk_9pulse`` uses it before every transfer). This tool reads it through ``/dev/mem``
(the board's kernel has no STRICT_DEVMEM), clocks SCL until the slave lets SDA go (nine clocks at
most, at 1 kHz), and ends with a STOP so every slave starts from idle.

On the board, as root (/opt/pepin/pepin is not installed in the /opt/pepin venv: the same
working directory and PYTHONPATH as the pepin-* units)::

    cd /opt/pepin && PYTHONPATH=/opt/pepin /opt/pepin/bin/python -m pepin.i2c_recover check
    systemctl stop pepin-tof
    cd /opt/pepin && PYTHONPATH=/opt/pepin /opt/pepin/bin/python -m pepin.i2c_recover recover
    systemctl start pepin-tof    # tof_init re-addresses the sensors

``check`` reads the lines 50 times over about 50 ms and calls a line held only when it is low in
every read: on a live bus (the IMU at 100 Hz, three ToF) one read can land mid-transfer with SCL
low. ``recover`` refuses while a process holds ``/dev/i2c-2`` open (``--force`` overrides): the
driver's own timeout handling resets the controller and would cut the clocks short.
"""

from __future__ import annotations

import argparse
import contextlib
import mmap
import os
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEVICE = "5002c00.i2c"  # TWI3 on PH4 (SCL) / PH5 (SDA): /dev/i2c-2 on this board
ADAPTER = "i2c-2"
LCR_OFFSET = 0x20
SDA_EN = 1 << 0
SDA_CTL = 1 << 1
SCL_EN = 1 << 2
SCL_CTL = 1 << 3
SDA_STATE = 1 << 4
SCL_STATE = 1 << 5
RELEASED = SDA_CTL | SCL_CTL  # both lines back to the controller, their control bits high
HALF_PERIOD_S = 0.0005  # 1 kHz: slow enough for any slave and any wiring
MAX_CLOCKS = 9  # a byte and its acknowledge: a slave mid-byte is free after at most nine
# check: 50 reads 1 ms apart span ~50 ms, five IMU periods (100 Hz); a transfer at 400 kHz lasts
# well under a millisecond, so a live bus cannot keep a line low through all of them.
CHECK_READS = 50
CHECK_INTERVAL_S = 0.001


class LineControl(Protocol):
    """The TWI_LCR register: anything that reads and writes it."""

    def read(self) -> int:
        """The register's value."""
        ...

    def write(self, value: int) -> None:
        """Set the register."""
        ...


@dataclass(frozen=True)
class Lines:
    """The two bus lines as the controller sees them: True is high (released)."""

    sda: bool
    scl: bool

    @classmethod
    def of(cls, lcr: int) -> Lines:
        """The lines from a TWI_LCR value."""
        return cls(sda=bool(lcr & SDA_STATE), scl=bool(lcr & SCL_STATE))

    def verdict(self) -> str:
        """What the two levels mean for the bus."""
        if self.sda and self.scl:
            return "idle: SDA and SCL high"
        if not self.scl:
            return (
                "SCL held low: a slave stretching the clock for ever or a short; clocks cannot "
                "help, a power cycle (or XSHUT for the ToF at 0x31/0x32) can"
            )
        return "SDA held low with SCL high: a slave stuck mid-byte; nine clocks free it"


@dataclass(frozen=True)
class Sampled:
    """The lines over a window of register reads; a line is held only if low in every read."""

    values: tuple[int, ...]
    window_s: float

    @property
    def sda_low(self) -> int:
        """Reads that saw SDA low."""
        return sum(1 for v in self.values if not v & SDA_STATE)

    @property
    def scl_low(self) -> int:
        """Reads that saw SCL low."""
        return sum(1 for v in self.values if not v & SCL_STATE)

    @property
    def held(self) -> bool:
        """SCL low in every read, or SDA low in every read with SCL high in every one."""
        n = len(self.values)
        return n > 0 and (self.scl_low == n or (self.sda_low == n and self.scl_low == 0))

    def verdict(self) -> str:
        """What the reads mean for the bus: idle, held, or busy with transfers."""
        n = len(self.values)
        if self.held or (self.sda_low == 0 and self.scl_low == 0):
            return Lines.of(self.values[0]).verdict()
        return (
            f"busy: SDA low in {self.sda_low}, SCL low in {self.scl_low} of {n} reads; "
            "transfers in flight, nothing held"
        )

    def line(self, status: str) -> str:
        """One line for the operator: the first read, runtime status, the window, a verdict."""
        first = self.values[0] if self.values else 0
        return (
            f"TWI_LCR 0x{first:02x} (runtime {status}, {len(self.values)} reads over "
            f"{self.window_s * 1000:.0f} ms): {self.verdict()}"
        )


def sample(
    lcr: LineControl,
    reads: int = CHECK_READS,
    interval_s: float = CHECK_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Sampled:
    """``reads`` reads of the register ``interval_s`` apart, with the window they really spanned."""
    if reads < 1:
        raise ValueError(f"reads must be at least 1, not {reads}")
    start = clock()
    values = [lcr.read()]
    for _ in range(reads - 1):
        sleep(interval_s)
        values.append(lcr.read())
    return Sampled(tuple(values), clock() - start)


@dataclass(frozen=True)
class Recovery:
    """What one recovery did: the lines before and after, the clocks given, the STOP."""

    before: Lines
    after: Lines
    clocks: int
    stop_sent: bool

    @property
    def freed(self) -> bool:
        """The bus is idle now."""
        return self.after.sda and self.after.scl

    def line(self) -> str:
        """One line for the operator."""
        outcome = "bus FREE" if self.freed else "bus STILL HELD: " + self.after.verdict()
        return (
            f"before: {self.before.verdict()}; {self.clocks} clock(s), "
            f"STOP {'sent' if self.stop_sent else 'not sent'}; {outcome}"
        )


def _sda_released(lcr: LineControl) -> bool:
    """SDA high on three reads in a row, as the vendor driver checks it (no glitch counts)."""
    return all(Lines.of(lcr.read()).sda for _ in range(3))


def clock_out(
    lcr: LineControl,
    sleep: Callable[[float], None] = time.sleep,
    half_period_s: float = HALF_PERIOD_S,
    max_clocks: int = MAX_CLOCKS,
) -> Recovery:
    """Clock SCL until the slave releases SDA, then send a STOP; both lines handed back after."""
    before = Lines.of(lcr.read())
    if before.sda and before.scl:
        return Recovery(before, before, 0, False)
    base = lcr.read() & ~(SDA_EN | SDA_CTL | SCL_EN | SCL_CTL)
    clocks = 0
    stop_sent = False
    try:
        lcr.write(base | SCL_EN | SCL_CTL | SDA_CTL)  # take SCL, high; SDA still the bus's
        sleep(half_period_s)
        if not Lines.of(lcr.read()).scl:
            return Recovery(before, Lines.of(lcr.read()), 0, False)
        while clocks < max_clocks and not _sda_released(lcr):
            lcr.write(base | SCL_EN | SDA_CTL)  # SCL low
            sleep(half_period_s)
            lcr.write(base | SCL_EN | SCL_CTL | SDA_CTL)  # SCL high
            sleep(half_period_s)
            clocks += 1
        if _sda_released(lcr):
            # STOP: SDA low while SCL is low, SCL up, then SDA up while SCL is high.
            lcr.write(base | SCL_EN | SDA_CTL)
            sleep(half_period_s)
            lcr.write(base | SCL_EN | SDA_EN)
            sleep(half_period_s)
            lcr.write(base | SCL_EN | SCL_CTL | SDA_EN)
            sleep(half_period_s)
            lcr.write(base | SCL_EN | SCL_CTL | SDA_EN | SDA_CTL)
            sleep(half_period_s)
            stop_sent = True
    finally:
        lcr.write(base | RELEASED)
        sleep(half_period_s)
    return Recovery(before, Lines.of(lcr.read()), clocks, stop_sent)


class DevMemLcr:
    """TWI_LCR of the controller at ``base`` through ``/dev/mem``; a context manager."""

    def __init__(self, base: int, path: str = "/dev/mem") -> None:
        page = mmap.PAGESIZE
        self._page_base = base & ~(page - 1)
        self._offset = base - self._page_base + LCR_OFFSET
        self._fd = os.open(path, os.O_RDWR | os.O_SYNC)
        self._map = mmap.mmap(
            self._fd, page, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
            offset=self._page_base,
        )  # fmt: skip
        self._view = memoryview(self._map).cast("I")  # 32-bit accesses, as the bus needs

    def read(self) -> int:
        """The register's value."""
        return int(self._view[self._offset // 4])

    def write(self, value: int) -> None:
        """Set the register."""
        self._view[self._offset // 4] = value & 0xFFFFFFFF

    def close(self) -> None:
        """Unmap and close."""
        self._view.release()
        self._map.close()
        os.close(self._fd)

    def __enter__(self) -> DevMemLcr:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def holders(device: str = "/dev/" + ADAPTER, proc: Path = Path("/proc")) -> list[str]:
    """``pid name`` of every process with ``device`` open (container processes included)."""
    found: list[str] = []
    for entry in sorted(proc.iterdir(), key=lambda p: p.name):
        if not entry.name.isdigit():
            continue
        with contextlib.suppress(OSError):
            for fd in (entry / "fd").iterdir():
                with contextlib.suppress(OSError):
                    if os.readlink(fd) == device:
                        name = (entry / "comm").read_text().strip()
                        found.append(f"{entry.name} {name}")
                        break
    return found


@contextlib.contextmanager
def awake(device_dir: Path) -> Iterator[str]:
    """Hold the controller out of runtime suspend (its clock gated, its registers in reset) while
    its register is used; the previous setting is put back after. Yields the runtime status."""
    control = device_dir / "power" / "control"
    previous = control.read_text().strip()
    control.write_text("on")
    try:
        time.sleep(0.01)
        yield (device_dir / "power" / "runtime_status").read_text().strip()
    finally:
        control.write_text(previous)


def _device_dir(sysfs: Path) -> Path:
    """The controller's sysfs directory, checked to be the adapter this tool is about."""
    device_dir = sysfs / DEVICE
    if not (device_dir / ADAPTER).exists():
        raise SystemExit(f"{device_dir} carries no {ADAPTER}: not the bus this tool knows")
    return device_dir


def main(argv: list[str] | None = None) -> int:
    """``check``: the lines and the bus's users; ``recover``: nine clocks and a STOP."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("action", choices=("check", "recover"))
    parser.add_argument("--force", action="store_true", help="recover with the bus held open")
    parser.add_argument("--sysfs", type=Path, default=Path("/sys/bus/platform/devices"))
    args = parser.parse_args(argv)
    device_dir = _device_dir(args.sysfs)
    users = holders()
    print(f"{ADAPTER} open in: {', '.join(users) if users else 'no process'}")
    base = int(DEVICE.split(".")[0], 16)
    with awake(device_dir) as status, DevMemLcr(base) as lcr:
        lines = sample(lcr)
        print(lines.line(status))
        if args.action == "check":
            return 1 if lines.held else 0
        if users and not args.force:
            print("refused: stop them first (systemctl stop pepin-tof; the bridge closes the bus "
                  "itself once its IMU is lost) or pass --force")  # fmt: skip
            return 2
        result = clock_out(lcr)
        print(result.line())
        return 0 if result.freed else 1


if __name__ == "__main__":
    sys.exit(main())
