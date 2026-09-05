"""Bridges between the laptop and the robot's serial devices.

The board runs ser2net, which exposes each serial device as a raw TCP port.
lerobot's motor bus wants a local tty path, so :class:`SerialBridge` uses
``socat`` to materialise a pseudo-terminal that forwards to that port.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from types import TracebackType

logger = logging.getLogger(__name__)

DEFAULT_HOST = "pepin.local"
SERVO_BUS_PORT = 3333
LIDAR_PORT = 3334
BOARD_MAC = "6c:35:cd:01:44:cb"  # the Zero 3's wifi interface, for arp-based fallback
_CACHE = Path.home() / ".cache" / "pepin" / "board_ip"


def _normalize_mac(mac: str) -> str:
    """Lower-case MAC with two-digit octets (macOS arp prints '6c:35:cd:1:44:cb')."""
    return ":".join(f"{int(part, 16):02x}" for part in mac.split(":"))


def ipv4_from_arp(mac: str = BOARD_MAC, arp_output: str | None = None) -> str | None:
    """The IPv4 the LAN currently associates with ``mac``, from the ARP table, or None."""
    if arp_output is None:
        try:
            # -n: no reverse DNS per entry; without it macOS takes ~5 s to print the table.
            arp_output = subprocess.run(
                ["arp", "-an"], capture_output=True, text=True, timeout=5
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    wanted = _normalize_mac(mac)
    for line in arp_output.splitlines():
        if "(" not in line or " at " not in line:
            continue
        ip = line.split("(")[1].split(")")[0]
        found = line.split(" at ")[1].split()[0]
        try:
            if ":" in found and _normalize_mac(found) == wanted:
                return ip
        except ValueError:
            continue  # "(incomplete)" and other non-MAC tokens
    return None


def board_address(host: str = DEFAULT_HOST, attempts: int = 2, pause_s: float = 1.0) -> str:
    """Resolve the board to an IPv4 address once, fast paths first, and remember it.

    Order: ``PEPIN_HOST`` in the environment (a typed IP wins), then the ARP
    table by the board's MAC (the LAN's own word, instant), then the cached
    last-known address, and only then mDNS — which on macOS is slow, answers
    "unknown host" for a while after a reboot, or returns an IPv6 link-local
    address that is useless to ssh and TCP. One session spent 44 s here.
    """
    started = time.monotonic()

    def remember(address: str, how: str) -> str:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE.write_text(address)
        logger.info("board at %s via %s (%.1f s)", address, how, time.monotonic() - started)
        return address

    forced = os.environ.get("PEPIN_HOST")
    if forced:
        return remember(forced, "PEPIN_HOST")
    from_arp = ipv4_from_arp()
    if from_arp:
        return remember(from_arp, "arp")
    if _CACHE.exists():
        cached = _CACHE.read_text().strip()
        if cached and ":" not in cached:
            return remember(cached, "cache")
    for _attempt in range(attempts):
        try:
            infos = socket.getaddrinfo(host, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        except socket.gaierror:
            infos = []
        if infos:
            return remember(str(infos[0][4][0]), "mdns")
        time.sleep(pause_s)
    raise ConnectionError(f"cannot resolve {host}: not in ARP, no cache, mDNS silent")


class SerialBridge:
    """A local pty that forwards to a ser2net TCP port on the robot.

    Use as a context manager; the pty path is yielded and the forwarding
    process is terminated on exit.
    """

    def __init__(self, tcp_port: int, link: str | Path, host: str = DEFAULT_HOST) -> None:
        """``tcp_port`` is the ser2net port on ``host``; ``link`` is where the local pty
        symlink will be created. Nothing is started until :meth:`open`."""
        if shutil.which("socat") is None:
            raise RuntimeError("socat is required for SerialBridge (brew install socat)")
        self._target = f"tcp:{host}:{tcp_port}"
        self._link = Path(link)
        self._proc: subprocess.Popen[bytes] | None = None

    @property
    def path(self) -> Path:
        """Where the pty appears; only usable between :meth:`open` and :meth:`close`."""
        return self._link

    def open(self, timeout_s: float = 3.0, settle_s: float = 0.5) -> Path:
        """Start the forwarder and return the pty path once the link is usable.

        socat creates the pty before its TCP side is connected, so the link
        appearing is not enough: ``settle_s`` covers the TCP handshake over
        wifi. Bytes written earlier would be silently lost.
        """
        self._link.unlink(missing_ok=True)
        self._proc = subprocess.Popen(
            ["socat", f"pty,link={self._link},raw,echo=0", self._target],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + timeout_s
        while not self._link.exists():
            if self._proc.poll() is not None or time.monotonic() > deadline:
                self.close()
                raise ConnectionError(f"socat could not bridge {self._target}")
            time.sleep(0.05)
        time.sleep(settle_s)
        return self._link

    def close(self) -> None:
        """Terminate the socat forwarder; the pty disappears with it."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=2)
        self._proc = None

    def __enter__(self) -> Path:
        """Start the forwarder and hand back the pty path to open as a serial port."""
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Tear the bridge down; the pty path becomes invalid."""
        self.close()
