"""The audio link's framing and the laptop's paced playback, without sockets."""

import json

import pytest

from pepin.audio_link import (
    MAX_LINE,
    MAX_PAYLOAD,
    AudioFrame,
    DoaReading,
    FrameDecoder,
    ProtocolError,
    encode,
    parse_doa,
    parse_frame,
    play_paced,
)


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
