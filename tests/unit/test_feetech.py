"""Feetech protocol: packets, framing, encodings, and the TCP client on a fake socket."""

import pytest

from pepin.feetech import (
    REGISTERS,
    FeetechTcpClient,
    PacketParser,
    StatusPacket,
    build_packet,
    decode_value,
    encode_value,
)

PING_8 = bytes.fromhex("ff ff 08 02 01 f4")  # captured from the real bus
REPLY_8 = bytes.fromhex("ff ff 08 02 00 f5")


def test_ping_packet_matches_real_capture() -> None:
    assert build_packet(8, 0x01) == PING_8


def test_velocity_uses_sign_magnitude_little_endian() -> None:
    reg = REGISTERS["Goal_Velocity"]
    assert encode_value(100, reg) == bytes([0x64, 0x00])
    assert encode_value(-100, reg) == bytes([0x64, 0x80])
    assert decode_value(bytes([0x64, 0x80]), reg) == -100
    assert decode_value(bytes([0xFF, 0x0F]), REGISTERS["Present_Position"]) == 4095


def test_parser_handles_junk_prefix_and_split_frames() -> None:
    parser = PacketParser()
    assert parser.feed(b"\x00\x13" + REPLY_8[:3]) == []
    assert parser.feed(REPLY_8[3:]) == [StatusPacket(motor_id=8, error=0, params=b"")]


def test_parser_drops_bad_checksum_and_resyncs() -> None:
    parser = PacketParser()
    bad = REPLY_8[:-1] + b"\x00"
    assert parser.feed(bad + REPLY_8) == [StatusPacket(8, 0, b"")]


def test_parser_keeps_a_trailing_header_byte() -> None:
    parser = PacketParser()
    assert parser.feed(b"\x00\xff") == []
    assert parser.feed(REPLY_8[1:]) == [StatusPacket(8, 0, b"")]


class FakeSocket:
    """Scripted socket: records sends and serves queued replies only after a send.

    Replies queued before the request are invisible to ``recv`` until
    ``sendall`` runs, mirroring a real bus where a reply follows its request;
    this keeps the client's pre-send flush from eating the scripted reply.
    """

    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.rx: list[bytes] = []
        self._armed = False

    def settimeout(self, value: float) -> None:
        pass

    def setsockopt(self, *args: object) -> None:
        pass

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)
        self._armed = True

    def recv(self, size: int) -> bytes:
        if not self._armed or not self.rx:
            self._armed = False  # a timeout ends the transaction; the next send re-arms
            raise TimeoutError()
        return self.rx.pop(0)

    def close(self) -> None:
        pass


@pytest.fixture
def client() -> tuple[FeetechTcpClient, FakeSocket]:
    c = FeetechTcpClient("host", 1, {"left": 7, "right": 8}, timeout_s=0.02, retries=1)
    fake = FakeSocket()
    c._sock = fake  # type: ignore[assignment]
    return c, fake


def status(motor_id: int, params: bytes) -> bytes:
    body = bytes([motor_id, len(params) + 2, 0]) + params
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])


def test_sync_read_builds_broadcast_and_decodes_both_replies(client) -> None:
    c, fake = client
    fake.rx = [status(7, bytes([0x10, 0x00])) + status(8, bytes([0xE8, 0x83]))]
    assert c.sync_read("Present_Position", ["left", "right"], normalize=False) == {
        "left": 16,
        "right": -1000,
    }
    assert fake.sent[-1] == bytes.fromhex("ff ff fe 06 82 38 02 07 08 30")


def test_sync_write_packet_and_no_reply_expected(client) -> None:
    c, fake = client
    c.sync_write("Goal_Velocity", {"left": -100, "right": 100}, normalize=False)
    assert fake.sent[-1] == bytes.fromhex("ff ff fe 0a 83 2e 02 07 64 80 08 64 00 ed")


def test_a_position_goal_is_two_plain_bytes_at_address_42(client) -> None:
    """Goal_Position carries no sign bit, unlike the velocities around it: a neck goal of 2021
    ticks must go out as e5 07, not as a sign-magnitude number."""
    c, fake = client
    c.sync_write("Goal_Position", {"left": 2021}, normalize=False)
    assert fake.sent[-1] == bytes.fromhex("ff ff fe 07 83 2a 02 07 e5 07 58")


def test_write_waits_for_the_status_reply(client) -> None:
    c, fake = client
    fake.rx = [status(7, b"")]
    c.write("Torque_Enable", "left", 1)
    assert fake.sent[-1] == bytes.fromhex("ff ff 07 04 03 28 01 c8")


def test_lost_reply_is_retried_then_raises(client) -> None:
    c, fake = client
    with pytest.raises(TimeoutError, match=r"\[7\]"):
        c.sync_read("Present_Position", ["left"], normalize=False)
    assert len(fake.sent) == 2  # one retry


def test_ping_returns_none_on_silence_and_error_byte_on_reply(client) -> None:
    c, fake = client
    assert c.ping("right") is None
    fake.rx = [REPLY_8]
    assert c.ping("right") == 0


def test_normalized_values_are_refused(client) -> None:
    c, _ = client
    with pytest.raises(ValueError):
        c.sync_read("Present_Position", ["left"])


# -- link loss ----------------------------------------------------------------


class ResetSocket(FakeSocket):
    """Dies the way ser2net kicks a client: the first recv raises ConnectionResetError."""

    def recv(self, size: int) -> bytes:
        raise ConnectionResetError(54, "Connection reset by peer")


class ClosedSocket(FakeSocket):
    """Peer closed politely: recv returns b"" forever."""

    def recv(self, size: int) -> bytes:
        return b""


def test_link_loss_reconnects_and_the_retry_succeeds(client) -> None:
    c, _ = client
    c._sock = ResetSocket()  # type: ignore[assignment]
    fresh = FakeSocket()
    fresh.rx = [status(7, bytes([0x10, 0x00]))]
    c.connect = lambda: setattr(c, "_sock", fresh)  # type: ignore[method-assign]
    assert c.sync_read("Present_Position", ["left"], normalize=False) == {"left": 16}
    assert fresh.sent, "the retry must go out on the new socket"


def test_link_loss_without_a_board_is_reported_as_a_timeout(client) -> None:
    c, _ = client
    c._sock = ResetSocket()  # type: ignore[assignment]

    def refuse() -> None:
        raise ConnectionRefusedError("board down")

    c.connect = refuse  # type: ignore[method-assign]
    with pytest.raises(TimeoutError, match="link lost"):
        c.sync_read("Present_Position", ["left"], normalize=False)


def test_orderly_close_is_not_a_successful_flush(client) -> None:
    c, _ = client
    c._sock = ClosedSocket()  # type: ignore[assignment]
    with pytest.raises(ConnectionError):
        c.flush()


def test_parser_survives_a_header_with_an_impossible_length() -> None:
    parser = PacketParser()
    assert parser.feed(bytes.fromhex("ff ff ff 00")) == []
    assert parser.feed(bytes.fromhex("ff ff ff 00") + REPLY_8) == [StatusPacket(8, 0, b"")]


def test_probe_client_does_not_take_the_port_back(client) -> None:
    c, _ = client
    c._reconnect_enabled = False
    c._sock = ResetSocket()  # type: ignore[assignment]
    reconnected = []
    c.connect = lambda: reconnected.append(True)  # type: ignore[method-assign]
    with pytest.raises(TimeoutError, match="link lost"):
        c.sync_read("Present_Position", ["left"], normalize=False)
    assert not reconnected


def test_reply_of_the_wrong_size_is_not_decoded_as_a_position(client) -> None:
    c, fake = client
    fake.rx = [status(7, b"")]  # a ping-style reply arriving where a 2-byte position was expected
    with pytest.raises(TimeoutError, match="malformed"):
        c.sync_read("Present_Position", ["left"], normalize=False)


# -- the neck rides the wheels' read; one packet writes a whole profile --------------------------


@pytest.fixture
def four() -> tuple[FeetechTcpClient, FakeSocket]:
    """A client with the wheels and the neck on its bus."""
    motors = {"left": 7, "right": 8, "neck": 9, "head": 10}
    c = FeetechTcpClient("host", 1, motors, timeout_s=0.02, retries=1)
    fake = FakeSocket()
    c._sock = fake  # type: ignore[assignment]
    return c, fake


def test_optional_ids_ride_after_the_mandatory_ones_and_a_silent_one_costs_no_retry(
    four: tuple[FeetechTcpClient, FakeSocket],
) -> None:
    c, fake = four
    position = bytes([0x00, 0x08])  # 2048
    fake.rx = [status(7, position) + status(8, position) + status(9, position)]  # 10 is silent
    read = c.sync_read(
        "Present_Position", ["left", "right"], normalize=False, optional=["neck", "head"]
    )
    assert read == {"left": 2048, "right": 2048, "neck": 2048}
    assert fake.sent == [bytes.fromhex("ff ff fe 08 82 38 02 07 08 09 0a 1b")], "one packet"


def test_a_silent_mandatory_id_still_raises_whatever_rides_along(
    four: tuple[FeetechTcpClient, FakeSocket],
) -> None:
    c, fake = four
    fake.rx = [status(9, bytes([0x00, 0x08])) + status(10, bytes([0x00, 0x08]))]
    with pytest.raises(TimeoutError, match=r"\[7, 8\]"):
        c.sync_read("Present_Position", ["left", "right"], normalize=False, optional=["neck"])


def test_a_read_of_optional_ids_alone_answers_what_answered(
    four: tuple[FeetechTcpClient, FakeSocket],
) -> None:
    """The neck's operating mode: no mandatory id, no retry, whoever answered in the window."""
    c, fake = four
    fake.rx = [status(9, b"\x00")]
    read = c.sync_read(
        "Operating_Mode", [], normalize=False, optional=["neck", "head"], optional_window_s=0.03
    )
    assert read == {"neck": 0}
    assert fake.sent == [bytes.fromhex("ff ff fe 06 82 21 01 09 0a 44")]


def test_a_malformed_optional_reply_is_dropped_not_decoded(
    four: tuple[FeetechTcpClient, FakeSocket],
) -> None:
    c, fake = four
    position = bytes([0x10, 0x00])
    fake.rx = [status(7, position) + status(8, position) + status(9, b"")]
    read = c.sync_read("Present_Position", ["left", "right"], normalize=False, optional=["neck"])
    assert read == {"left": 16, "right": 16}


def test_a_block_write_is_one_packet_over_adjacent_registers(
    four: tuple[FeetechTcpClient, FakeSocket],
) -> None:
    """Torque on, ramp 114, goal 2029, time 0, speed 1365, at address 40, eight bytes a servo."""
    c, fake = four
    names = ["Torque_Enable", "Acceleration", "Goal_Position", "Goal_Time", "Goal_Velocity"]
    c.sync_write_block(names, {"neck": [1, 114, 2029, 0, 1365]})
    assert fake.sent == [bytes.fromhex("ff ff fe 0d 83 28 08 09 01 72 ed 07 00 00 55 05 77")]
    with pytest.raises(ValueError, match="contiguous"):
        c.sync_write_block(["Torque_Enable", "Goal_Position"], {"neck": [1, 2029]})
    with pytest.raises(ValueError, match="2 values for 5 registers"):
        c.sync_write_block(names, {"neck": [1, 2]})


class TimedSocket(FakeSocket):
    """Records every timeout the client sets."""

    def __init__(self) -> None:
        super().__init__()
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


def test_a_write_drops_what_arrived_without_waiting_for_a_quiet_line(client) -> None:
    """Nothing comes back for a write; waiting out a quiet window before each one cost the
    wheels' twist 5 ms of a 20 ms tick. A read still waits for it."""
    c, _ = client
    fake = TimedSocket()
    c._sock = fake  # type: ignore[assignment]
    c.sync_write("Goal_Velocity", {"left": 0, "right": 0}, normalize=False)
    assert fake.timeouts[0] == 0.0 and 0.005 not in fake.timeouts
    fake._armed = False  # nothing answers a write
    fake.rx = [status(7, bytes([0x10, 0x00]))]
    c.sync_read("Present_Position", ["left"], normalize=False)
    assert 0.005 in fake.timeouts, "the read's flush waits for a quiet line"
