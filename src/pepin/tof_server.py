"""ToF server: runs on the board and streams the three VL53L1X ranges as JSON lines.

Each line is ``{"t": <board monotonic s>, "front": mm, "left": mm, "right": mm}``
at the ranging rate; a sensor that is missing, failed to start, or reports a
failure status sends ``null`` — never a bogus small range (the VL53L1X returns
15-25 mm together with its failure codes, which would read as "something two
centimetres ahead"). The socket is served by :class:`pepin.streams.JsonLinesServer`
so a client can always connect and see which sensors are alive.

Run on the board (see ``board/pepin-tof.service``)::

    python -m pepin.tof_server --port 3335

The sensor library is imported lazily: the module (and its tests, with a fake
ranger) also load on a laptop without the I2C driver.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import threading
import time
from typing import Any, Protocol

from pepin.streams import JsonLinesServer

logger = logging.getLogger(__name__)

# Which address sits on which side, checked by hand on the robot 2026-09-08: the sensor at
# 0x31 is on the LEFT of the cart looking the way it drives. The names were the other way
# round since the wiring, so every side reading went into the costmap mirrored.
SENSORS = {"front": 0x30, "left": 0x31, "right": 0x32}
I2C_BUS = 2
# Medium, measured on the robot 2026-09-08 after a clean re-address (share of frames the sensor
# called a measurement): short front 11% / left 100% / right 98%; medium 31 / 100 / 100;
# long 12 / 100 / 94. The front one is the reason: mounted at 0.27 m it looks over the room at
# a wall 1.93 m away, which short mode can only report as wraparound (status 7).
RANGING_MODE = 2  # 1 short (1.3 m, fastest), 2 medium (3 m), 3 long (4 m)
MIN_RANGE_MM = 40  # below the sensor's own minimum: noise, not an object
TIMING_BUDGET_MS = 50
INTER_MEASUREMENT_MS = 66  # ~15 Hz per sensor


class Ranger(Protocol):
    """Anything that produces one record of ranges per call; the real one talks I2C."""

    def read(self) -> dict[str, Any]:
        """Millimetres per sensor name (``None`` for no valid return) plus ``t``."""
        ...

    def close(self) -> None:
        """Stop ranging and release the bus."""
        ...


class RangeReader:
    """Keeps every reachable VL53L1X ranging continuously and serves the latest valid reading."""

    def __init__(self) -> None:
        import VL53L1X  # pimoroni driver around ST's ULD API, talks to /dev/i2c-<bus>

        self._sensors: dict[str, Any] = {}
        for name, address in SENSORS.items():
            try:
                sensor = VL53L1X.VL53L1X(i2c_bus=I2C_BUS, i2c_address=address)
                sensor.open()
                # A restart used to start ranging on a sensor that was already ranging, which
                # leaves the VL53L1X answering 'hardware fail' (status 5) until its next power
                # cycle — three service restarts in a row killed two sensors on 2026-09-08.
                with contextlib.suppress(Exception):
                    sensor.stop_ranging()
                sensor.set_timing(TIMING_BUDGET_MS * 1000, INTER_MEASUREMENT_MS)
                sensor.start_ranging(RANGING_MODE)
                self._sensors[name] = sensor
                logger.info("%s @0x%02x: ranging", name, address)
            except Exception as exc:  # a missing sensor must not take the others down
                logger.error("%s @0x%02x: FAILED to start (%s)", name, address, exc)

    def read(self) -> dict[str, Any]:
        """One record: valid ranges in millimetres, ``None`` where the sensor has no target.

        The VL53L1X's range status travels with the reading (``status``): 0 is a measurement and
        every other value says WHY there is none — 1 sigma too high, 2 signal too weak, 4 out of
        bounds, 7 wraparound. Without it a dead sensor and an empty room look identical on the
        wire, which is exactly the argument that cost a day (2026-09-08).
        """
        record: dict[str, Any] = {"t": time.monotonic()}
        status_of: dict[str, int | None] = {}
        for name in SENSORS:
            sensor = self._sensors.get(name)
            if sensor is None:
                record[name] = None
                status_of[name] = None
                continue
            try:
                mm = sensor.get_distance()
                status = int(sensor.get_range_status())
                # Only status 0 is a measurement; failure statuses carry a tiny bogus range.
                record[name] = mm if status == 0 and mm >= MIN_RANGE_MM else None
                status_of[name] = status
            except Exception:
                record[name] = None
                status_of[name] = None
        record["status"] = status_of
        return record

    def close(self) -> None:
        """Stop ranging on every sensor."""
        for sensor in self._sensors.values():
            try:
                sensor.stop_ranging()
                sensor.close()
            except Exception:
                pass


def serve(
    ranger: Ranger, server: JsonLinesServer, hz: float, stop: threading.Event | None = None
) -> None:
    """Read at ``hz`` and broadcast every record to whoever is connected, until ``stop`` is set."""
    period = 1.0 / hz
    try:
        while stop is None or not stop.is_set():
            started = time.monotonic()
            server.commands()  # this server takes no commands; drain so the inbox cannot grow
            server.broadcast(ranger.read())
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    finally:
        ranger.close()
        server.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream VL53L1X ranges over TCP.")
    parser.add_argument("--port", type=int, default=3335)
    parser.add_argument("--hz", type=float, default=15.0)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s"
    )
    server = JsonLinesServer(args.port).start()
    serve(RangeReader(), server, args.hz)


if __name__ == "__main__":
    main()
