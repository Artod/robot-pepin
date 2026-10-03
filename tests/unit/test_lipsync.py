"""The audio server's lip sync: the loudness of what the speaker plays, dated by when it sounds."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

from pepin.audio_server import LipSync, Player
from pepin.face import load_face_table, mouth_level

REPO = Path(__file__).resolve().parents[2]
RATE = 16000
WINDOW = 640  # 40 ms


class FakeClock:
    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now


def tone(seconds: float, amplitude: float) -> bytes:
    """A 440 Hz sine of that peak amplitude (0..1 of full scale)."""
    n = round(seconds * RATE)
    return struct.pack(
        f"<{n}h",
        *(round(amplitude * 32767 * math.sin(2 * math.pi * 440 * i / RATE)) for i in range(n)),
    )


def make() -> tuple[LipSync, FakeClock, list[tuple[float, float]]]:
    clock = FakeClock()
    sent: list[tuple[float, float]] = []
    table = load_face_table(REPO / "config" / "face.json")
    lips = LipSync(lambda level: sent.append((clock.now, level)), table, rate=RATE, clock=clock)
    return lips, clock, sent


def drain(lips: LipSync, clock: FakeClock, seconds: float, dt: float = 0.01) -> None:
    end = clock.now + seconds
    while clock.now < end:
        clock.now += dt
        lips.step(0.0)


def test_each_window_is_sent_when_it_is_heard_at_its_loudness() -> None:
    table = load_face_table(REPO / "config" / "face.json")
    lips, clock, sent = make()
    lips.chunk(tone(0.2, 0.05) + tone(0.2, 0.0))  # 5 windows at -29 dBFS, then 5 silent ones
    start = clock.now + table.lipsync["start_latency_s"] - table.lipsync["lead_s"]
    drain(lips, clock, 1.0)
    expected = mouth_level(20 * math.log10(0.05 / math.sqrt(2)), table)
    assert 0.4 < expected < 0.7
    assert [level for _, level in sent] == [pytest.approx(expected, abs=0.005)] * 5 + [0.0]
    for i, (when, _) in enumerate(sent[:5]):  # each within one step of its moment
        assert start + i * 0.04 <= when < start + i * 0.04 + 0.0101
    assert lips.sent == 6  # the silent windows after the first zero are not repeated


def test_windows_run_across_chunk_boundaries() -> None:
    lips, clock, sent = make()
    pcm = tone(0.4, 0.5)
    for i in range(0, len(pcm), 2 * 333):  # 333-sample chunks: none a whole window
        lips.chunk(pcm[i : i + 2 * 333])
    drain(lips, clock, 1.0)
    assert len(sent) == 10 and all(level > 0.5 for _, level in sent)


def test_a_late_chunk_restarts_the_count_where_the_sound_resumes() -> None:
    lips, clock, sent = make()
    lips.chunk(tone(0.08, 0.5))
    drain(lips, clock, 0.5)  # played out; the next chunk comes after its own moment
    lips.chunk(tone(0.08, 0.5))
    assert lips.late == 1
    resumed = clock.now
    drain(lips, clock, 0.5)
    assert sent[-1][0] >= resumed + 0.12 - 0.0101  # dated from now, not from the stale count


def test_a_stop_closes_the_mouth_at_once_and_drops_what_was_pending() -> None:
    lips, clock, sent = make()
    lips.chunk(tone(1.0, 0.5))
    drain(lips, clock, 0.3)
    n = len(sent)
    lips.stop()
    drain(lips, clock, 1.0)
    assert sent[n:] == [(pytest.approx(clock.now - 0.99, abs=0.02), 0.0)]


def test_behind_only_the_newest_level_goes_out() -> None:
    lips, clock, sent = make()
    lips.chunk(tone(0.4, 0.5))
    clock.now += 1.0
    lips.step(0.0)
    assert len(sent) == 1


class FakeSink:
    def __init__(self) -> None:
        self.written: list[bytes] = []

    def write(self, pcm: bytes) -> None:
        self.written.append(pcm)

    def finish(self) -> None:
        pass

    def abort(self) -> None:
        pass


def test_the_player_reports_what_the_device_took_and_every_end() -> None:
    lips, clock, sent = make()
    sink = FakeSink()
    player = Player(lambda: sink, rate=RATE, clock=clock, lipsync=lips)
    player.feed(tone(0.08, 0.5))
    player.step(0.0)
    assert sink.written and lips.status() == {"lipsync_sent": 0, "lipsync_late": 0}
    drain(lips, clock, 0.4)
    assert len(sent) == 2
    player.flush()
    drain(lips, clock, 0.05)
    assert sent[-1][1] == 0.0
