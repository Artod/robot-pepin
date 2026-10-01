"""The base link's wire format and the client's non-blocking state."""

import contextlib
import json
import time

from pepin.base_link import BaseClient, decode_state
from pepin.kinematics import Twist
from pepin.streams import encode


def test_state_message_round_trips_and_ages_on_the_laptop_clock() -> None:
    message = {
        "type": "state", "t": 12.5, "x": 1.0, "y": -0.5, "theta": 0.3, "dl": 0.01, "dr": 0.012,
        "v": 0.15, "w": 0.0, "moving": True, "armed": True, "deadman": False, "bus_ok": True,
        "bus_p95_ms": 9.1,
    }  # fmt: skip
    client = BaseClient("unused")
    client._ingest(json.loads(encode(message)))
    state = client.state(now=client._received_at + 0.25)
    assert state is not None
    assert state.pose.x == 1.0 and state.moving and state.bus_p95_ms == 9.1
    assert abs(state.age_s - 0.25) < 1e-6
    assert decode_state(message, received_at=0.0).stamp_s == 12.5


def test_commands_are_dropped_not_raised_while_the_link_is_down() -> None:
    client = BaseClient("unused")
    client.set_twist(Twist(0.1, 0.0))  # no socket yet: logged, not an exception
    client.stop()
    assert client.state() is None


def test_neck_jog_and_the_neck_request_are_the_boards_lines() -> None:
    sent: list[dict[str, object]] = []
    client = BaseClient("unused")
    client.send = sent.append  # type: ignore[method-assign]
    client.neck_jog(1, -1, slow=True)
    client.neck_jog(0, 0)
    client.ask_neck()
    assert sent == [
        {"cmd": "neck_jog", "pan": 1, "tilt": -1, "slow": True},
        {"cmd": "neck_jog", "pan": 0, "tilt": 0, "slow": False},
        {"cmd": "neck"},
    ]


def test_neck_answers_and_refused_jogs_are_kept_for_the_window() -> None:
    client = BaseClient("unused")
    assert client.neck() is None and client.neck_error() is None
    client._ingest(
        {"type": "neck", "pan_ticks": 2029, "tilt_ticks": 2311, "age_s": 0.0, "read_ms": 1.2}
    )
    reading = client.neck()
    assert reading is not None and reading.ticks == (2029, 2311)
    client._ingest({"type": "neck_jog", "error": "the wheels are moving"})
    assert client.neck_error() == "the wheels are moving"
    assert client.neck_error(now=time.monotonic() + 5.0) is None, "old news is dropped"
    client._ingest({"type": "neck_jog"})  # not an error: nothing to keep
    assert client.neck_error() == "the wheels are moving"


def test_pong_wakes_a_waiting_ping() -> None:
    import threading

    client = BaseClient("unused")
    pong = {"type": "pong", "servos": {"7": True, "8": False}}
    threading.Timer(0.02, lambda: client._ingest(pong)).start()
    assert client.ping(timeout_s=1.0) == {"7": True, "8": False}
    assert client.ping(timeout_s=0.01) is None  # nobody answers this time


class FakeBaseServer:
    """The base server's port: streams state lines to whoever connects, and answers one request
    with the lines given for its ``cmd``. What it was asked is kept in ``asked``."""

    def __init__(self, answers: dict[str, list[dict[str, object]]]) -> None:
        import socket
        import threading

        self.answers = answers
        self.asked: list[dict[str, object]] = []
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(2)
        self._listener.settimeout(10.0)
        self.port = int(self._listener.getsockname()[1])
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            try:
                connection, _ = self._listener.accept()
            except OSError:
                return
            with connection:
                connection.sendall(encode({"type": "state", "t": 1.0, "x": 0.0}))
                connection.sendall(b"\n")  # the blank line a reader must skip
                request = json.loads(connection.makefile("r").readline())
                self.asked.append(request)
                connection.sendall(encode({"type": "state", "t": 1.05, "x": 0.0}))
                for answer in self.answers.get(str(request.get("cmd")), []):
                    connection.sendall(encode(answer))
                with contextlib.suppress(OSError):
                    connection.recv(1)  # hold the line until the client hangs up

    def close(self) -> None:
        self._listener.close()


def test_a_one_shot_request_skips_the_state_stream_and_returns_the_asked_reply() -> None:
    from pepin.base_link import ask

    reply = {
        "type": "neck_goto", "pan_ticks": 2029, "tilt_ticks": 2360, "reached": True, "ms": 1200.0,
    }  # fmt: skip
    server = FakeBaseServer({"neck_home": [reply]})
    try:
        got = ask("127.0.0.1", {"cmd": "neck_home"}, "neck_goto", wait_s=2.0, port=server.port)
    finally:
        server.close()
    assert got == reply
    assert server.asked == [{"cmd": "neck_home"}]


def test_a_one_shot_request_gives_up_on_a_silent_server_and_fails_on_a_closed_port() -> None:
    import socket

    import pytest

    from pepin.base_link import ask

    server = FakeBaseServer({})  # states only: no answer of the asked type ever comes
    try:
        assert ask("127.0.0.1", {"cmd": "neck"}, "neck", wait_s=0.2, port=server.port) is None
    finally:
        server.close()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed = int(probe.getsockname()[1])
    with pytest.raises(OSError):
        ask("127.0.0.1", {"cmd": "neck"}, "neck", wait_s=0.2, port=closed)
