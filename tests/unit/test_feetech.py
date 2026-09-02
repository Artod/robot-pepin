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
