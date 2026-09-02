"""Bridges between the laptop and the robot's serial devices.

The board runs ser2net, which exposes each serial device as a raw TCP port.
lerobot's motor bus wants a local tty path, so :class:`SerialBridge` uses
``socat`` to materialise a pseudo-terminal that forwards to that port.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from types import TracebackType

DEFAULT_HOST = "pepin.local"
SERVO_BUS_PORT = 3333
LIDAR_PORT = 3334


class SerialBridge:
    """A local pty that forwards to a ser2net TCP port on the robot.

    Use as a context manager; the pty path is yielded and the forwarding
    process is terminated on exit.
    """

    def __init__(self, tcp_port: int, link: str | Path, host: str = DEFAULT_HOST) -> None:
        if shutil.which("socat") is None:
            raise RuntimeError("socat is required for SerialBridge (brew install socat)")
        self._target = f"tcp:{host}:{tcp_port}"
        self._link = Path(link)
        self._proc: subprocess.Popen[bytes] | None = None

    @property
    def path(self) -> Path:
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
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=2)
        self._proc = None

    def __enter__(self) -> Path:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
