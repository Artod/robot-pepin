"""The servo bus tap: the proxy copies both directions and logs them, the framer reads the log
back as packets, and the decoder names them."""

from __future__ import annotations

import socket
import threading
from pathlib import Path

from pepin.bus_tap import (
    FROM_BUS,
    TO_BUS,
    Stream,
    Tap,
    TapLog,
    describe,
    frame,
    parse_line,
)
from pepin.feetech import (
    INST_SYNC_READ,
    INST_SYNC_WRITE,
    INST_WRITE,
    build_packet,
    encode_value,
)
from pepin.feetech import REGISTERS as R

NECK_BLOCK = bytes([40, 8, 9, 1, 74]) + (3929).to_bytes(2, "little") + b"\x00\x00"
NECK_BLOCK += encode_value(2275, R["Goal_Velocity"])
NECK_BLOCK += bytes([10, 1, 68]) + (2310).to_bytes(2, "little") + b"\x00\x00"
NECK_BLOCK += encode_value(2275, R["Goal_Velocity"])
BLOCK = build_packet(0xFE, INST_SYNC_WRITE, NECK_BLOCK)
READ = build_packet(0xFE, INST_SYNC_READ, bytes([56, 2, 7, 8, 9, 10]))
REPLY = b"\xff\xff\x09\x04\x00" + (2031).to_bytes(2, "little")
REPLY += bytes([(~sum(REPLY[2:])) & 0xFF])


def test_the_framer_reads_packets_across_any_chunking_and_names_them() -> None:
    whole = BLOCK + READ
    for cut in (1, 3, 7, 11, len(BLOCK) - 1, len(BLOCK), len(BLOCK) + 1):
        first, carry = frame(whole[:cut])
        rest, tail = frame(whole[cut:], carry)
        packets = first + rest
        assert tail == b"" and [p.raw for p in packets] == [BLOCK, READ], cut
    block, read = frame(whole)[0]
    assert block.ok and read.ok and block.noise == 0
    text = describe(block, TO_BUS)
    assert text.startswith("ALL SYNC_WRITE @40+8 id 9: Torque_Enable=1 Acceleration=74")
    assert "Goal_Position=3929" in text and "Goal_Velocity=2275" in text
    assert "id 10: Torque_Enable=1 Acceleration=68 Goal_Position=2310" in text
    assert describe(read, TO_BUS) == "ALL SYNC_READ Present_Position x2 ids [7, 8, 9, 10]"
    (reply,), _ = frame(REPLY)
    assert describe(reply, FROM_BUS) == f"id 9 status err=0x00 {(2031).to_bytes(2, 'little').hex()}"


def test_a_bad_checksum_and_noise_between_packets_are_flagged() -> None:
    bad = bytearray(BLOCK)
    bad[12] ^= 0x10  # one bit of the goal's high byte, the checksum left as it was
    packets, _ = frame(b"\x12\x34" + bytes(bad) + READ)
    assert [p.ok for p in packets] == [False, True]
    assert packets[0].noise == 2 and "CHK FAIL" in describe(packets[0], TO_BUS)
    assert "noise 2" in describe(packets[0], TO_BUS)
    (write,), _ = frame(build_packet(9, INST_WRITE, bytes([42]) + (323).to_bytes(2, "little")))
    assert describe(write, TO_BUS) == "id 9 WRITE Goal_Position=323"


def test_a_reply_whose_checksum_is_0xff_does_not_swallow_the_next_header() -> None:
    """On the robot's log a reply ending in a 0xFF checksum ran into the next packet's header
    (ff ff ff fe ...): the framer took id 0xFF, length 0xFE and swallowed a quarter of a second
    of the stream as one bad packet. 0xFF is never an id: the header starts a byte later."""
    body = bytes([7, 4, 0]) + bytes([0x00, 0x00])
    # the checksum of a status reply is ~sum(body) & 0xFF: a body summing to 0 (mod 256) gives 0xFF
    body = bytes([7, 4, 0, 0xF5, 0x00])
    assert (~sum(body)) & 0xFF == 0xFF
    reply = b"\xff\xff" + body + b"\xff"
    packets, tail = frame(reply + READ + REPLY)
    assert tail == b""
    assert [(p.motor_id, p.ok, p.noise) for p in packets] == [
        (7, True, 0),
        (0xFE, True, 0),
        (9, True, 0),
    ]
    # split right after the reply: its checksum is consumed, nothing is carried
    first, carry = frame(reply)
    assert [p.motor_id for p in first] == [7] and carry == b""
    rest, tail = frame(READ + REPLY, carry)
    assert [p.motor_id for p in rest] == [0xFE, 9] and tail == b""
    # a chunk cut inside the header after a 0xFF of noise: the carried byte slides, noise 1
    first, carry = frame(b"\x12\xff")
    assert first == [] and carry == b"\xff"
    rest, tail = frame(READ, carry)
    assert [(p.motor_id, p.noise) for p in rest] == [(0xFE, 1)] and tail == b""


def test_the_log_line_round_trips_through_the_stream(tmp_path: Path) -> None:
    log = TapLog(tmp_path, None)
    line = log.write(TO_BUS, BLOCK, now=1791930000.5)
    log.close()
    parsed = parse_line(line)
    assert parsed is not None and parsed[0] == 1791930000.5 and parsed[2] == TO_BUS
    assert parsed[3] == BLOCK
    assert (tmp_path / "bus-tap.log").read_text() == line
    stream = Stream(TO_BUS)
    (stamped,) = stream.feed(parsed[0], parsed[3])
    assert stamped[1].raw == BLOCK and parse_line("garbage\n") is None


def test_the_tap_copies_both_ways_logs_both_and_a_new_client_takes_the_bus(
    tmp_path: Path,
) -> None:
    """A fake ser2net echoes what it gets, prefixed; the tap pumps a client's bytes to it and
    its answer back, logging each chunk; a second client ends the first pair."""
    bus = socket.create_server(("127.0.0.1", 0))
    bus.settimeout(5.0)
    served: list[socket.socket] = []

    def echo() -> None:
        while True:
            try:
                conn, _ = bus.accept()
            except OSError:
                return
            served.append(conn)
            conn.settimeout(5.0)
            try:
                data = conn.recv(4096)
                conn.sendall(b"<" + data)
            except OSError:
                return

    threading.Thread(target=echo, daemon=True).start()
    log = TapLog(tmp_path, None)
    tap = Tap(("127.0.0.1", 0), ("127.0.0.1", bus.getsockname()[1]), log)
    client = socket.create_connection(("127.0.0.1", tap.port), timeout=5.0)
    for _ in range(50):
        tap.serve_once(0.05)
        if tap._bus is not None:
            break
    client.sendall(BLOCK)
    answer = b""
    for _ in range(100):
        tap.serve_once(0.05)
        try:
            client.settimeout(0.01)
            answer += client.recv(4096)
        except OSError:
            pass
        if answer:
            break
    assert answer == b"<" + BLOCK
    lines = (tmp_path / "bus-tap.log").read_text().splitlines()
    assert [parse_line(ln + "\n")[2] for ln in lines if parse_line(ln + "\n")] == [TO_BUS, FROM_BUS]  # type: ignore[index]
    second = socket.create_connection(("127.0.0.1", tap.port), timeout=5.0)
    for _ in range(50):
        tap.serve_once(0.05)
        if tap.clients == 2 and tap._bus is not None:
            break
    client.settimeout(1.0)
    assert client.recv(10) == b"", "the first client was ended by the newcomer"
    assert len(served) == 2, "a fresh bus connection for the newcomer"
    second.close()
    tap.stop()
    bus.close()
