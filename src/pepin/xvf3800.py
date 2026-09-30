"""The reSpeaker XVF3800 array's USB control interface: firmware version and voice direction.

The array answers vendor control transfers on endpoint 0 while ALSA streams its audio, so the
direction of arrival is polled beside a running capture without touching the audio interfaces.
One read is one IN transfer: ``bmRequestType`` 0xC0 (device to host, vendor, device),
``bRequest`` 0, ``wValue`` = 0x80 | command id, ``wIndex`` = resource id, and a length of the
payload plus one status byte. The answer's first byte is that status: 0 done, 64 "busy, ask
again", anything else an error; the payload follows, little-endian.

The resource and command ids and the payload shapes are Seeed's, as their Python host tool
defines them (``python_control/xvf_host.py`` of respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY,
commit 4b49bfd of 2026-09-29). That repository carries no licence, so this module implements the
protocol it documents rather than copying the tool. ``DOA_VALUE`` is Seeed's own command
(firmware 2.0.6 and later): their C ``xvf_host`` for rpi_64bit, pinned by
``board/xvf_host_install.sh``, predates it and does not know the name.

On the board::

    python -m pepin.xvf3800 version
    python -m pepin.xvf3800 doa --watch 5

``pyusb`` is imported lazily: the module and its tests load on a laptop without it.
"""

from __future__ import annotations

import argparse
import math
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

VID = 0x2886  # Seeed Technology
PID = 0x001A  # reSpeaker XVF3800 USB 4-Mic Array, USB (not DFU) firmware
REQUEST_TYPE_IN = 0xC0  # device-to-host | vendor | recipient device
READ_FLAG = 0x80  # set in wValue for a read
STATUS_DONE = 0
STATUS_BUSY = 64  # SERVICER_COMMAND_RETRY: the device is not ready, ask again

_FORMATS = {"uint8": "B", "uint16": "H", "int32": "i", "uint32": "I", "float": "f"}


@dataclass(frozen=True)
class Command:
    """One readable parameter: where it lives (resource, command id) and what it carries."""

    name: str
    resid: int
    cmdid: int
    count: int
    kind: str  # a key of _FORMATS

    @property
    def length(self) -> int:
        """Bytes to ask for: one status byte and the payload."""
        return 1 + struct.calcsize("<" + _FORMATS[self.kind] * self.count)


COMMANDS = {
    c.name: c
    for c in (
        Command("VERSION", 48, 0, 3, "uint8"),  # major, minor, patch
        Command("DOA_VALUE", 20, 18, 2, "uint16"),  # degrees 0..359, speech detected 0/1
        Command("AEC_AZIMUTH_VALUES", 33, 75, 4, "float"),  # radians: focused 1, 2, free, auto
        Command("AEC_SPENERGY_VALUES", 33, 80, 4, "float"),  # speech energy of the same beams
        Command("AEC_AECCONVERGED", 33, 3, 1, "int32"),  # 1 once the echo canceller converged
    )
}


class XvfError(OSError):
    """The array refused or garbled an answer (an OSError: the caller treats it like I/O)."""


class XvfBusyError(XvfError):
    """Status 64: the device asks to be asked again."""


def parse_response(command: Command, raw: bytes) -> tuple[float, ...]:
    """The values of one control answer (status byte first); raises :class:`XvfBusyError` for
    status 64 and :class:`XvfError` for any other status or a length that does not fit."""
    if not raw:
        raise XvfError(f"{command.name}: empty answer")
    if raw[0] == STATUS_BUSY:
        raise XvfBusyError(f"{command.name}: busy")
    if raw[0] != STATUS_DONE:
        raise XvfError(f"{command.name}: status {raw[0]}")
    if len(raw) != command.length:
        raise XvfError(f"{command.name}: {len(raw)} bytes, expected {command.length}")
    fmt = "<" + _FORMATS[command.kind] * command.count
    values: tuple[float, ...] = struct.unpack(fmt, raw[1:])
    return values


def parse_doa(values: tuple[float, ...]) -> tuple[int, bool]:
    """``DOA_VALUE``'s two numbers as (degrees 0..359, speech detected); raises
    :class:`XvfError` for anything else — an angle past 359 is a garbled read, not a voice."""
    if len(values) != 2:
        raise XvfError(f"DOA_VALUE: {len(values)} values, expected 2")
    deg, speech = int(values[0]), int(values[1])
    if not 0 <= deg < 360 or speech not in (0, 1):
        raise XvfError(f"DOA_VALUE: implausible {values}")
    return deg, bool(speech)


def firmware_warning(version: tuple[int, int, int]) -> str | None:
    """What this firmware cannot do for the audio server, or None when it is new enough.

    From Seeed's USB firmware changelog: ``DOA_VALUE`` exists from 2.0.6, and before 2.0.10
    the angle froze whenever the LED ring was not in its "doa" effect."""
    if version < (2, 0, 6):
        return "no DOA_VALUE before firmware 2.0.6: update the firmware (dfu-util)"
    if version < (2, 0, 10):
        return "before firmware 2.0.10 the DOA freezes unless LED_EFFECT is 4 (doa)"
    return None


class ControlTransport(Protocol):
    """One vendor IN control transfer to the array; the real one is :class:`PyUsbTransport`."""

    def read(self, value: int, index: int, length: int) -> bytes:
        """``length`` bytes answered for ``wValue``/``wIndex``; raises OSError on USB failure."""
        ...

    def close(self) -> None:
        """Release the device handle."""
        ...


class PyUsbTransport:
    """Control transfers through pyusb (libusb): no interface is claimed, so the kernel's audio
    driver keeps streaming. Needs write access to the device node (root, or the audio group via
    ``board/99-pepin-usb.rules``)."""

    def __init__(self, vid: int = VID, pid: int = PID, timeout_ms: int = 200) -> None:
        """Find the array on the bus; raises OSError when it is not there."""
        import usb.core  # pyusb: pip install pyusb (board/README.md, Microphone array)

        device = usb.core.find(idVendor=vid, idProduct=pid)
        if device is None:
            raise OSError(f"no USB device {vid:04x}:{pid:04x} (reSpeaker XVF3800)")
        self._device = device
        self._timeout_ms = timeout_ms

    def read(self, value: int, index: int, length: int) -> bytes:
        """One IN transfer; pyusb's USBError is an OSError already."""
        answer = self._device.ctrl_transfer(
            REQUEST_TYPE_IN, 0, value, index, length, self._timeout_ms
        )
        return bytes(answer)

    def close(self) -> None:
        """Free libusb's handle on the device."""
        import usb.util

        usb.util.dispose_resources(self._device)


class Xvf3800:
    """Reads named parameters off the array, asking again while it answers "busy"."""

    def __init__(
        self,
        transport: ControlTransport,
        *,
        tries: int = 6,
        retry_wait_s: float = 0.002,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """``tries`` bounds the reads of one parameter; ``retry_wait_s`` is slept between them."""
        self._transport = transport
        self._tries = tries
        self._retry_wait_s = retry_wait_s
        self._sleep = sleep

    def read(self, name: str) -> tuple[float, ...]:
        """The values of parameter ``name`` (a key of :data:`COMMANDS`); raises OSError."""
        command = COMMANDS[name]
        for _ in range(self._tries):
            raw = self._transport.read(READ_FLAG | command.cmdid, command.resid, command.length)
            try:
                return parse_response(command, raw)
            except XvfBusyError:
                self._sleep(self._retry_wait_s)
        raise XvfError(f"{name}: still busy after {self._tries} tries")

    def version(self) -> tuple[int, int, int]:
        """The firmware as (major, minor, patch)."""
        major, minor, patch = (int(v) for v in self.read("VERSION"))
        return major, minor, patch

    def doa(self) -> tuple[int, bool]:
        """The voice direction in the array's frame, degrees 0..359, and whether it hears speech."""
        return parse_doa(self.read("DOA_VALUE"))

    def close(self) -> None:
        """Release the device."""
        self._transport.close()


def open_array() -> Xvf3800:
    """The array on this machine's USB, through pyusb; raises OSError when it is absent."""
    return Xvf3800(PyUsbTransport())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read the reSpeaker XVF3800's firmware version and voice direction over USB."
    )
    parser.add_argument("what", choices=["version", "doa", "azimuth", "spenergy", "converged"])
    parser.add_argument("--watch", type=float, default=0.0, help="repeat at this rate (Hz)")
    args = parser.parse_args()
    array = open_array()
    try:
        while True:
            if args.what == "version":
                print("VERSION " + ".".join(str(v) for v in array.version()))
            elif args.what == "doa":
                deg, speech = array.doa()
                print(f"DOA_VALUE {deg} deg speech={int(speech)}")
            elif args.what == "azimuth":
                beams = array.read("AEC_AZIMUTH_VALUES")
                print("AEC_AZIMUTH_VALUES deg " + " ".join(f"{math.degrees(r):.1f}" for r in beams))
            elif args.what == "spenergy":
                print("AEC_SPENERGY_VALUES " + " ".join(f"{e:.0f}" for e in array.read(
                    "AEC_SPENERGY_VALUES")))  # fmt: skip
            else:
                print(f"AEC_AECCONVERGED {int(array.read('AEC_AECCONVERGED')[0])}")
            if args.watch <= 0:
                return
            time.sleep(1.0 / args.watch)
    except KeyboardInterrupt:
        pass
    finally:
        array.close()


if __name__ == "__main__":
    main()
