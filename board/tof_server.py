#!/opt/pepin/bin/python
"""Stream the three VL53L1X time-of-flight ranges over TCP as JSON lines.

Runs on the Orange Pi next to ser2net. Each line: {"t": <monotonic s>,
"front": mm, "left": mm, "right": mm} at the ranging rate; a sensor that
is missing or fails to answer reports null. The socket is bound before the
sensors are touched, so a client can always connect and see what is alive.
Addresses are the ones tof-init assigns at boot.

Usage (as a systemd service, see pepin-tof.service):
    /opt/pepin/bin/python /opt/pepin/tof_server.py --port 3335
"""

import argparse
import json
import socket
import sys
import threading
import time

import VL53L1X  # pimoroni driver around ST's ULD API, talks to /dev/i2c-<bus>

SENSORS = {"front": 0x30, "right": 0x31, "left": 0x32}
I2C_BUS = 2
RANGING_MODE = 1  # 1 short (1.3 m, fastest), 2 medium (3 m), 3 long (4 m)
TIMING_BUDGET_MS = 50
INTER_MEASUREMENT_MS = 66  # ~15 Hz per sensor


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", file=sys.stderr, flush=True)


class RangeReader:
    """Keeps every reachable sensor ranging continuously and serves the latest reading."""

    def __init__(self) -> None:
        self._sensors = {}
        for name, address in SENSORS.items():
            try:
                sensor = VL53L1X.VL53L1X(i2c_bus=I2C_BUS, i2c_address=address)
                sensor.open()
                sensor.set_timing(TIMING_BUDGET_MS * 1000, INTER_MEASUREMENT_MS)
                sensor.start_ranging(RANGING_MODE)
                self._sensors[name] = sensor
                log(f"{name} @0x{address:02x}: ranging")
            except Exception as exc:  # a missing sensor must not take the others down
                log(f"{name} @0x{address:02x}: FAILED to start ({exc})")

    def read(self) -> dict:
        record = {"t": time.monotonic()}
        for name in SENSORS:
            sensor = self._sensors.get(name)
            if sensor is None:
                record[name] = None
                continue
            try:
                mm = sensor.get_distance()
                record[name] = mm if mm > 0 else None
            except Exception:
                record[name] = None
        return record

    def close(self) -> None:
        for sensor in self._sensors.values():
            try:
                sensor.stop_ranging()
                sensor.close()
            except Exception:
                pass


def serve(port: int, hz: float) -> None:
    # Dual-stack: the laptop may resolve pepin.local to a link-local IPv6 address.
    server = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    server.bind(("::", port))
    server.listen(4)
    log(f"listening on {port}")
    clients: list[socket.socket] = []
    lock = threading.Lock()

    def accept_loop() -> None:
        while True:
            conn, peer = server.accept()
            with lock:
                clients.append(conn)
            log(f"client {peer[0]} connected")

    threading.Thread(target=accept_loop, daemon=True).start()
    reader = RangeReader()
    period = 1.0 / hz
    try:
        while True:
            started = time.monotonic()
            line = (json.dumps(reader.read()) + "\n").encode()
            with lock:
                for conn in list(clients):
                    try:
                        conn.sendall(line)
                    except OSError:
                        clients.remove(conn)
                        conn.close()
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream VL53L1X ranges over TCP.")
    parser.add_argument("--port", type=int, default=3335)
    parser.add_argument("--hz", type=float, default=15.0)
    args = parser.parse_args()
    serve(args.port, args.hz)


if __name__ == "__main__":
    main()
