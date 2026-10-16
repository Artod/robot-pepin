"""The audio link's framing and the laptop's paced playback, without sockets."""

import json
import socket

import pytest

from pepin.audio_link import (
    MAX_LINE,
    MAX_PAYLOAD,
    AudioClient,
    AudioFrame,
    DoaReading,
    FrameDecoder,
    ProtocolError,
    RawFrame,
    encode,
    offset_of,
    parse_doa,
    parse_frame,
    parse_raw,
    play_paced,
)

UPTIME = 33_450.25  # the board's monotonic clock on the evening of 2026-10-15: the stamps' clock
OFFSET = 1_792_046_998.5  # its wall clock (TF's, chrony's) less that one


def test_a_payload_rides_behind_its_header_and_survives_any_cut() -> None:
    frame = AudioFrame(7, 1234.5, bytes(range(256)) * 2 + b"\x01\x02")
    wire = frame.encode() + encode({"type": "doa", "t": 1.0, "deg": 90, "speech": True})
    wire += encode({"type": "status", "capture_alive": True})
    for cut in range(1, len(wire)):  # every possible split into two chunks
        decoder = FrameDecoder()
        messages = decoder.feed(wire[:cut]) + decoder.feed(wire[cut:])
        assert [m[0]["type"] for m in messages] == ["pcm", "doa", "status"]
        header, payload = messages[0]
        assert parse_frame(header, payload) == frame
        assert parse_doa(messages[1][0]) == DoaReading(1.0, 90, True)


def test_byte_at_a_time_and_many_messages_per_chunk_decode_the_same() -> None:
    wire = b"".join(AudioFrame(i, float(i), b"\x00\x01" * 320).encode() for i in range(5))
    decoder = FrameDecoder()
    trickled = [m for b in wire for m in decoder.feed(bytes([b]))]
    assert [h["seq"] for h, _ in trickled] == [0, 1, 2, 3, 4]
    assert [h["seq"] for h, _ in FrameDecoder().feed(wire)] == [0, 1, 2, 3, 4]


def test_the_header_is_plain_json_so_a_status_probe_reads_with_nc() -> None:
    line = encode({"cmd": "status"})
    assert line == b'{"cmd":"status"}\n'
    first = AudioFrame(0, 2.0, b"\x00\x00" * 4).encode().split(b"\n", 1)[0]
    assert json.loads(first) == {"type": "pcm", "seq": 0, "t": 2.0, "bytes": 8}


@pytest.mark.parametrize(
    "garbage",
    [
        b"not json\n",
        b"[1, 2]\n",
        b'{"bytes": -1}\n',
        b'{"bytes": "640"}\n',
        b'{"bytes": true}\n',
        b'{"bytes": %d}\n' % (MAX_PAYLOAD + 1),
        b"x" * (MAX_LINE + 1),
    ],
)
def test_garbage_ends_the_framing_instead_of_guessing(garbage: bytes) -> None:
    with pytest.raises(ProtocolError):
        FrameDecoder().feed(garbage)


def test_malformed_frames_and_directions_are_refused() -> None:
    with pytest.raises(ValueError):
        parse_frame({"type": "pcm", "seq": 0, "t": 0.0}, b"\x00\x00\x00")
    with pytest.raises(ValueError):
        parse_doa({"type": "doa", "t": 0.0, "deg": 400, "speech": False})
    with pytest.raises(KeyError):
        parse_doa({"type": "doa", "t": 0.0})


def test_frame_duration_follows_its_samples() -> None:
    frame = AudioFrame(0, 0.0, b"\x00\x00" * 320)
    assert frame.samples == 320 and frame.duration_s == pytest.approx(0.02)


def test_a_raw_microphone_frame_rides_the_wire_beside_the_processed_one() -> None:
    raw = RawFrame(7, 12.5, b"\x01\x00\x02\x00\x03\x00\x04\x00" * 320, 4)
    assert raw.samples == 320 and raw.duration_s == pytest.approx(0.02)
    wire = AudioFrame(7, 12.5, b"\x09\x00" * 320).encode() + raw.encode()
    (pcm_header, pcm), (raw_header, payload) = FrameDecoder().feed(wire)
    assert pcm_header["type"] == "pcm" and raw_header == {
        "type": "raw", "seq": 7, "t": 12.5, "ch": 4, "bytes": 2560
    }  # fmt: skip
    assert parse_raw(raw_header, payload) == raw
    assert parse_frame(pcm_header, pcm).stamp_s == raw.stamp_s, "the same moment, both kinds"
    with pytest.raises(ValueError):
        parse_raw({"type": "raw", "seq": 0, "t": 0.0, "ch": 4}, b"\x00" * 6)  # not whole samples
    with pytest.raises(ValueError):
        parse_raw({"type": "raw", "seq": 0, "t": 0.0, "ch": 0}, b"")


def test_a_reading_is_placed_on_the_board_s_wall_clock_by_the_link_s_offset_alone() -> None:
    """``t`` stays the board's monotonic clock on the wire (zero bytes more a frame); the
    offset the link holds turns a reading, a frame and a raw frame into the wall clock's
    moment (``wall_s``); without one, none (an old server: never the monotonic stamp)."""
    doa = parse_doa({"type": "doa", "t": UPTIME, "deg": 90, "speech": True}, OFFSET)
    assert doa.stamp_s == UPTIME and doa.wall_s == pytest.approx(UPTIME + OFFSET)
    assert parse_doa({"type": "doa", "t": UPTIME, "deg": 90, "speech": True}).wall_s is None
    pcm = parse_frame({"type": "pcm", "seq": 3, "t": UPTIME}, b"\x00\x00" * 320, 16000, OFFSET)
    raw = parse_raw({"type": "raw", "seq": 3, "t": UPTIME, "ch": 4}, b"\x00" * 2560, 16000, OFFSET)
    assert pcm.wall_s == raw.wall_s == pytest.approx(UPTIME + OFFSET)
    assert DoaReading(UPTIME, 90, True, OFFSET).encode() == DoaReading(UPTIME, 90, True).encode()
    assert offset_of({"clock_offset_s": OFFSET}) == OFFSET
    for absent in ({}, {"clock_offset_s": None}, {"clock_offset_s": True},
                   {"clock_offset_s": float("nan")}, {"clock_offset_s": "1"}):  # fmt: skip
        assert offset_of(absent) is None, absent


@pytest.mark.slow
def test_the_client_takes_the_offset_from_the_hello_and_each_status_after_a_step() -> None:
    """The hello's offset stamps every frame and direction that follows; a status after the
    board's wall clock was stepped (chrony's first sync after a boot) replaces it; a server
    that tells none (a build before 2026-10-16) leaves ``wall_s`` None."""
    board, laptop = socket.socketpair()
    client = AudioClient("board", connector=lambda address: laptop)
    try:
        board.sendall(encode({"type": "hello", "rate": 16000, "clock_offset_s": OFFSET}))
        client.start(listen=False)
        assert client.clock_offset_s == OFFSET
        board.sendall(DoaReading(UPTIME, 90, True).encode())
        board.sendall(AudioFrame(1, UPTIME, b"\x00\x00" * 320).encode())
        board.sendall(RawFrame(1, UPTIME, b"\x00" * 2560).encode())
        frame, raw = client.next_frame(2.0), client.next_raw(2.0)
        doa = client.latest_doa()  # read before the frame on the same reader thread
        assert frame is not None and frame.wall_s == pytest.approx(UPTIME + OFFSET)
        assert raw is not None and raw.wall_s == pytest.approx(UPTIME + OFFSET)
        assert doa is not None and doa.wall_s == pytest.approx(UPTIME + OFFSET)
        board.sendall(encode({"type": "status", "clock_offset_s": OFFSET + 2.5}))
        board.sendall(AudioFrame(2, UPTIME + 0.02, b"\x00\x00" * 320).encode())
        stepped = client.next_frame(2.0)
        assert stepped is not None and stepped.wall_s == pytest.approx(UPTIME + 0.02 + OFFSET + 2.5)
        assert client.clock_offset_s == OFFSET + 2.5
        board.sendall(encode({"type": "status", "capture_alive": True}))  # an older status
        board.sendall(AudioFrame(3, UPTIME + 0.04, b"\x00\x00" * 320).encode())
        kept = client.next_frame(2.0)
        assert kept is not None and kept.clock_offset_s == OFFSET + 2.5, "kept, not dropped"
    finally:
        client.close()
        board.close()
    board, laptop = socket.socketpair()
    old = AudioClient("board", connector=lambda address: laptop)
    try:
        board.sendall(encode({"type": "hello", "rate": 16000}))
        old.start(listen=False)
        board.sendall(AudioFrame(1, UPTIME, b"\x00\x00" * 320).encode())
        frame = old.next_frame(2.0)
        assert old.clock_offset_s is None and frame is not None and frame.wall_s is None
    finally:
        old.close()
        board.close()


class FakeSpeaker:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.ended = self.flushed = False

    def play(self, pcm: bytes) -> None:
        self.chunks.append(pcm)

    def play_end(self) -> None:
        self.ended = True

    def flush(self) -> None:
        self.flushed = True


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_paced_playback_stays_one_lead_ahead_of_real_time_and_ends_the_utterance() -> None:
    speaker, time = FakeSpeaker(), FakeTime()
    pcm = b"\x01\x00" * 16_000 * 3  # three seconds at 16 kHz
    sent_s = play_paced(speaker, pcm, lead_s=1.0, clock=time.clock, sleep=time.sleep)
    assert sent_s == pytest.approx(3.0) and speaker.ended and not speaker.flushed
    assert b"".join(speaker.chunks) == pcm
    assert all(len(c) == 640 for c in speaker.chunks)
    # Three seconds of audio a second ahead: sending took about two seconds of the clock.
    assert time.now == pytest.approx(2.0, abs=0.05)


def test_a_barge_in_flushes_instead_of_ending() -> None:
    import threading

    speaker, time, stop = FakeSpeaker(), FakeTime(), threading.Event()

    def sleep(seconds: float) -> None:
        time.sleep(seconds)
        if time.now > 0.5:
            stop.set()

    sent_s = play_paced(
        speaker, b"\x00\x00" * 16_000 * 5, lead_s=0.2, stop=stop, clock=time.clock, sleep=sleep
    )
    assert speaker.flushed and not speaker.ended
    assert sent_s < 1.0
