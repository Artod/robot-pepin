"""The base link's wire format and the client's non-blocking state."""

import json

from pepin.base_link import BaseClient, decode_state, encode
from pepin.kinematics import Twist


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


def test_pong_wakes_a_waiting_ping() -> None:
    import threading

    client = BaseClient("unused")
    pong = {"type": "pong", "servos": {"7": True, "8": False}}
    threading.Timer(0.02, lambda: client._ingest(pong)).start()
    assert client.ping(timeout_s=1.0) == {"7": True, "8": False}
    assert client.ping(timeout_s=0.01) is None  # nobody answers this time
