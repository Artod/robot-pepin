"""Bridges between the laptop and the robot's serial devices.

The board runs ser2net, which exposes each serial device as a raw TCP port.
lerobot's motor bus wants a local tty path, so :class:`SerialBridge` uses
``socat`` to materialise a pseudo-terminal that forwards to that port.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path
from types import TracebackType

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
            arp_output = subprocess.run(
                ["arp", "-a"], capture_output=True, text=True, timeout=5
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    wanted = _normalize_mac(mac)
    for line in arp_output.splitlines():
        if "(" not in line or " at " not in line:
            continue
        ip = line.split("(")[1].split(")")[0]
        found = line.split(" at ")[1].split()[0]
        if ":" in found and _normalize_mac(found) == wanted:
            return ip
    return None


def board_address(host: str = DEFAULT_HOST, attempts: int = 4, pause_s: float = 1.0) -> str:
    """Resolve the board to an IPv4 address once, with retries, and remember it.

    mDNS on macOS is flaky: slow, "unknown host" for a while, or IPv6
    link-local only — and a bare fe80:: address is useless to ssh and TCP
    without an interface scope. So: IPv4 from the name, else IPv4 from the
    ARP table by the board's MAC, else the cached last-known address.
    """
    for _attempt in range(attempts):
        try:
            infos = socket.getaddrinfo(host, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        except socket.gaierror:
            infos = []
        addresses = [str(info[4][0]) for info in infos]
        if not addresses:
            from_arp = ipv4_from_arp()
            if from_arp:
                addresses = [from_arp]
        if addresses:
            _CACHE.parent.mkdir(parents=True, exist_ok=True)
            _CACHE.write_text(addresses[0])
            return addresses[0]
        time.sleep(pause_s)
    if _CACHE.exists():
        cached = _CACHE.read_text().strip()
        if cached and ":" not in cached:
            return cached
    raise ConnectionError(f"cannot resolve {host} to IPv4 and no cached address is known")


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
