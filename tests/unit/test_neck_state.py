"""The neck node under the ROS stubs: when its transform may claim the head stood still.

The board's neck node polls the encoders slowly (2 Hz) and republishes the last edge in between
with a fresh stamp, which is a claim that the head did not move. Measured 2026-09-30 during a
30 deg turn, that claim put the camera's frames up to 16 deg off. Here: the held edge only while
two readings agree, the bus asked fast while they do not, and measured edges only during a move.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import ros_stubs

ros_stubs.install()

import pepin_bringup.neck_state as neck_state  # noqa: E402

from pepin.depth import rotation_matrix  # noqa: E402
from pepin.neck import RAD_PER_TICK  # noqa: E402


class FakeLink:
    """The base server's socket as a list of the requests sent; replies are fed by hand."""

    def __init__(self, host: str, port: int, on_line: Any, name: str = "") -> None:
        self.on_line = on_line
        self.sent: list[bytes] = []

    def start(self) -> None:
        """Nothing to connect to."""

    def stop(self) -> None:
        """Nothing to close."""

    def send(self, data: bytes) -> None:
        self.sent.append(data)

    def take_status_change(self) -> None:
        return None


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(neck_state, "JsonLineLink", FakeLink)
    clock = {"t": 1000.0}
    monkeypatch.setattr(neck_state.time, "monotonic", lambda: clock["t"])
    built = neck_state.NeckState()
    built.fake_time = clock
    return built


def _reply(node: Any, pan: int, tilt: int = 2311) -> None:
    node._link.on_line({"type": "neck", "pan_ticks": pan, "tilt_ticks": tilt, "age_s": 0.0})


def _tick(node: Any, seconds: float, step: float = 0.05) -> None:
    """Advance the clock, running the poll and the hold timers as rclpy would."""
    for _ in range(round(seconds / step)):
        node.fake_time["t"] += step
        node._poll()
        node._hold_tf()


def test_a_still_head_is_held_and_polled_slowly(node: Any) -> None:
    """Two equal readings and a second of quiet: the edge is republished between polls (dense
    TF for the frames) and the bus is asked at poll_hz, twice a second."""
    _reply(node, 2029)
    _tick(node, 1.2)
    _reply(node, 2029)
    held = len(node._tf.sent)
    sent = len(node._link.sent)
    _tick(node, 2.0)
    assert not node.moving
    assert len(node._link.sent) - sent == pytest.approx(4, abs=1)
    assert len(node._tf.sent) - held >= 15  # the 10 Hz hold kept TF dense


def test_a_turning_head_is_never_held_and_is_polled_fast(node: Any) -> None:
    """The first reading that differs starts a move: no held edge until the readings have
    stood still for MOVE_LINGER_S, the bus asked at move_poll_hz, and every edge sent is a
    measured one at its own reading."""
    _reply(node, 2029)
    _tick(node, 1.2)
    _reply(node, 1900)  # 11 deg to the left: a move
    assert node.moving
    held = node._counts["held"]
    sent = len(node._link.sent)
    _tick(node, 0.5)
    assert node._counts["held"] == held, "no edge is held while the head turns"
    assert len(node._link.sent) - sent == pytest.approx(10, abs=1)  # 20 Hz for half a second
    pans = []
    for ticks in (1850, 1800, 1750):
        _reply(node, ticks)
        q = node._tf.sent[-1].transform.rotation
        r = rotation_matrix(q.x, q.y, q.z, q.w)  # Rz(pan) Ry(pitch): the yaw is the pan
        pans.append(math.atan2(r[1, 0], r[0, 0]))
    assert pans[0] < pans[1] < pans[2]  # each edge is its own reading (pan_sign -1: fewer = left)
    _reply(node, 1750)
    _tick(node, neck_state.MOVE_LINGER_S + 0.1)
    assert not node.moving
    _tick(node, 0.2)
    assert node._counts["held"] > held, "held again once the head stood still"


def test_encoder_jitter_is_not_a_move(node: Any) -> None:
    """A reading STILL_TICKS away from the last is the same pose: the hold goes on."""
    _reply(node, 2029)
    _tick(node, 1.2)
    _reply(node, 2029 + neck_state.STILL_TICKS)
    assert not node.moving
    assert math.radians(0.2) > neck_state.STILL_TICKS * RAD_PER_TICK
