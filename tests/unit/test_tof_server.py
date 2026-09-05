"""The ToF server over real localhost sockets, with a fake ranger instead of the I2C sensors."""

import threading
import time
from typing import Any

from pepin.streams import JsonLinesServer
from pepin.tof import TofClient
from pepin.tof_server import serve


class FakeRanger:
    def __init__(self) -> None:
        self.closed = False

    def read(self) -> dict[str, Any]:
        return {"t": time.monotonic(), "front": 250, "left": None, "right": 442}

    def close(self) -> None:
        self.closed = True


def test_ranges_stream_to_a_tof_client_and_the_server_shuts_down_cleanly() -> None:
    ranger = FakeRanger()
    server = JsonLinesServer(0).start()
    stop = threading.Event()
    worker = threading.Thread(target=serve, args=(ranger, server, 50.0, stop), daemon=True)
    worker.start()
    client = TofClient("127.0.0.1", server.port).start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and client.ranges().front is None:
        time.sleep(0.01)
    r = client.ranges()
    assert r.front == 0.25 and r.left is None and r.right == 0.442 and r.age_s < 1.0
    client.close()
    stop.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive() and ranger.closed
