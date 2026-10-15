"""The Mac's own voice direction (pepin.audio_doa) on synthetic plane waves: the azimuth, two
voices at once, quiet chunks, the readings' cadence and anchor; the choice between it and the
chip (pepin.expression.doa.DoaChoice), the tagger on its readings, the register of voices
(pepin.expression.tracks.VoiceSources) and its event after each utterance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pepin.audio_doa import (
    CHUNK,
    ArrayGeometry,
    MicDoa,
    MicReading,
    Peak,
    SrpPhat,
    merge_peaks,
    pcm_of,
    plane_wave,
)
from pepin.audio_link import FRAME_SAMPLES, RATE, DoaReading, RawFrame
from pepin.expression.conversation import Heard
from pepin.expression.doa import CHIP, DoaChoice, Tagger, VoiceRule, utterance_voice, wrap_deg
from pepin.expression.executors import NoWheels
from pepin.expression.fakes import FakeDoa, FakeGazeDoor
from pepin.expression.speaker import DoaMap, HeadView
from pepin.expression.tracker import SpeakerGaze, SpeakerSettings
from pepin.expression.tracks import VoiceSources
from pepin.gaze import HOME, Aim, Reach
from pepin.neck import NeckConfig

REPO = Path(__file__).resolve().parents[2]
GEOMETRY = ArrayGeometry(((0.033, -0.033), (0.033, 0.033), (-0.033, 0.033), (-0.033, -0.033)))
MAP = DoaMap(180.0, 1)  # the array reads 180 straight ahead, growing leftward


def voice(n: int, seed: int) -> Any:
    """A voice-like signal: a few harmonics of 180 Hz under noise."""
    r = np.random.default_rng(seed)
    t = np.arange(n) / RATE
    tones = sum(
        np.sin(2 * np.pi * f * t + r.uniform(0, 6)) / k
        for k, f in enumerate((180, 360, 540, 720, 1100, 1500, 2200), 1)
    )
    return 0.5 * tones + 0.2 * r.standard_normal(n)


@pytest.fixture(scope="module")
def srp() -> SrpPhat:
    return SrpPhat(GEOMETRY)


def test_the_geometry_and_the_knobs_come_from_config_audio_json() -> None:
    assert ArrayGeometry.load(REPO / "config/audio.json") == GEOMETRY
    built = SrpPhat.load(REPO / "config/audio.json")
    assert built.geometry.count == 4 and len(built.pairs) == 6 and built.angles_deg.size == 360
    with pytest.raises(ValueError):
        ArrayGeometry.from_config({"mics_m": [[0.0, 0.0]]})


def test_a_plane_wave_lands_within_5_deg_from_every_quarter_and_between(srp: SrpPhat) -> None:
    for azimuth in (0, 45, 90, 135, 180, 225, 270, 315, 17, 203, 350):
        peaks = srp.peaks(srp.response(plane_wave(GEOMETRY, voice(CHUNK, 3), azimuth)), ratio=0.6)
        assert len(peaks) == 1, f"{azimuth}: {peaks}"
        assert abs(wrap_deg(peaks[0].deg - azimuth)) <= 5.0 and peaks[0].strength > 0.9
    errors = []
    for seed in range(12):  # 0 dB of independent noise per mic, one 32 ms chunk each
        azimuth = (seed * 37) % 360
        noise = 0.3 * np.random.default_rng(100 + seed).standard_normal((4, CHUNK))
        noisy = plane_wave(GEOMETRY, voice(CHUNK, seed), azimuth) + noise
        peaks = srp.peaks(srp.response(noisy), ratio=0.6)
        errors.append(abs(wrap_deg(peaks[0].deg - azimuth)))
    assert float(np.median(errors)) <= 10.0 and max(errors) <= 25.0, errors


def test_two_voices_at_once_give_two_peaks_and_noise_alone_a_weak_one(srp: SrpPhat) -> None:
    for a, b in ((0, 90), (30, 200), (45, 300)):
        both = plane_wave(GEOMETRY, voice(CHUNK, 7), a) + plane_wave(GEOMETRY, voice(CHUNK, 11), b)
        peaks = srp.peaks(srp.response(both), ratio=0.6)
        assert len(peaks) == 2, f"{a},{b}: {peaks}"
        found = sorted(p.deg for p in peaks)
        assert abs(wrap_deg(found[0] - a)) <= 8.0 and abs(wrap_deg(found[1] - b)) <= 8.0
    noise = np.random.default_rng(4).standard_normal((4, CHUNK))
    peaks = srp.peaks(srp.response(noise), ratio=0.6)
    assert peaks and peaks[0].strength < 0.15, "no plane wave: a weak response"
    assert srp.peaks(np.zeros(360), ratio=0.6) == [], "a flat response: no source"
    merged = merge_peaks([Peak(100.0, 0.5), Peak(104.0, 0.7), Peak(200.0, 0.3), Peak(359.0, 0.4)])
    assert merged == (Peak(104.0, 0.7), Peak(359.0, 0.4), Peak(200.0, 0.3))


@dataclass
class Series:
    """Silero's probabilities on the laptop clock: ``p`` over the ``spans`` (from, to, p),
    ``quiet`` elsewhere; None before anything was heard."""

    spans: list[tuple[float, float, float]] = field(default_factory=list)
    quiet: float | None = 0.1

    def peak(self, t0: float, t1: float) -> float | None:
        inside = [p for a, b, p in self.spans if a <= t1 and t0 <= b]
        return max(inside) if inside else self.quiet


def raw_frames(
    signal: Any, *, stamp0: float = 1000.0, seq0: int = 0, channels: int = 4
) -> list[RawFrame]:
    """``signal`` (mics x samples) cut into the board's 20 ms raw frames."""
    frames = []
    for k in range(signal.shape[1] // FRAME_SAMPLES):
        chunk = signal[:, k * FRAME_SAMPLES : (k + 1) * FRAME_SAMPLES]
        frames.append(
            RawFrame(seq0 + k, stamp0 + k * FRAME_SAMPLES / RATE, pcm_of(chunk), channels)
        )
    return frames


def test_readings_come_per_0_1_s_with_the_strongest_peak_and_quiet_chunks_yield_nothing(
    srp: SrpPhat,
) -> None:
    """One second of a voice at 120 deg, Silero 0.9 over its first half (laptop 10.0..10.5) and
    0.1 after: the voiced spans' readings carry the peak, the quiet spans' none; ten readings
    a second like the chip's, each the span's three or four chunks."""
    signal = plane_wave(GEOMETRY, voice(RATE, 5), 120)
    laptop = 10.0
    mic = MicDoa(srp, Series([(laptop, laptop + 0.5, 0.9)]), voice_p_min=0.5, reading_s=0.1,
                 clock=lambda: laptop)  # fmt: skip
    closed: list[MicReading] = []
    for k, frame in enumerate(raw_frames(signal)):
        closed += mic.feed(frame, laptop + 0.02 * (k + 1))
    assert mic.frames == 50 and mic.chunks == 31 and 14 <= mic.voiced <= 17
    assert 8 <= len(closed) <= 10 and all(3 <= r.chunks <= 4 for r in closed)
    voiced = [r for r in closed if r.speech]
    quiet = [r for r in closed if not r.speech]
    assert 4 <= len(voiced) <= 5 and len(quiet) >= 3
    for r in voiced:
        assert abs(wrap_deg(r.deg - 120)) <= 5 and r.voice_p == 0.9 and r.peaks
        assert abs(wrap_deg(r.peaks[0].deg - 120)) <= 5 and r.voiced >= 1
    for r in quiet:
        assert r.deg == -1 and r.peaks == () and r.voice_p == 0.1 and r.voiced == 0
    for k, r in enumerate(closed):  # spans on a 0.1 s grid; a reading's stamp is its first chunk's
        assert 1000.0 + 0.1 * k - 1e-6 <= r.stamp_s < 1000.0 + 0.1 * k + CHUNK / RATE, (k, r)
    latest = mic.latest_doa()
    assert latest is not None and latest is closed[-1] and latest.speech is False
    assert mic.doa_between(1000.0, 1000.25) == closed[:3]
    assert mic.raw_age_s(laptop + 1.0) == pytest.approx(0.0)
    assert mic.status()["chunk_ms_p95"] < 5.0 and mic.status()["series"]
    deaf = MicDoa(srp)  # no ears yet: nothing is voice
    deaf.feed(raw_frames(signal)[0], 1.0)
    assert deaf.voiced == 0 and math.isinf(MicDoa(srp).raw_age_s(5.0))


def test_a_seq_gap_restarts_the_buffer_on_the_new_frame_s_stamp(srp: SrpPhat) -> None:
    signal = plane_wave(GEOMETRY, voice(RATE // 2, 5), 60)
    mic = MicDoa(srp, Series([(0.0, 1e9, 0.9)]), reading_s=0.032)
    frames = raw_frames(signal)
    for k, frame in enumerate(frames[:3]):
        mic.feed(frame, 1.0 + 0.02 * k)
    later = RawFrame(frames[9].seq, frames[9].stamp_s + 5.0, frames[9].pcm, 4)
    closed = mic.feed(later, 2.0)
    assert mic.gaps == 1 and [r.stamp_s for r in closed] == [1000.0], "the span before the gap"
    assert closed[0].deg == 60 and closed[0].chunks == 1, "...closed on its one whole chunk"
    closed = mic.feed(RawFrame(later.seq + 1, later.stamp_s + 0.02, frames[10].pcm, 4), 2.02)
    assert closed == [], "the partial chunk before the gap was dropped: one new chunk, pending"
    for k in (2, 3):  # the next whole chunk closes the span opened on the first after the gap
        closed += mic.feed(RawFrame(later.seq + k, later.stamp_s + 0.02 * k, frames[9 + k].pcm, 4),
                           2.0 + 0.02 * k)  # fmt: skip
    assert len(closed) == 1 and closed[0].stamp_s == pytest.approx(later.stamp_s)
    with pytest.raises(ValueError):
        mic.feed(RawFrame(0, 0.0, b"\x00" * 1280, 2), 3.0)  # two channels, not the geometry's


def test_the_choice_reads_the_chip_while_no_raw_frames_come_and_says_so_once(
    srp: SrpPhat,
) -> None:
    chip = FakeDoa([DoaReading(1.0, 90, True)])
    mac = MicDoa(srp, clock=lambda: 100.0)
    said: list[str] = []
    now = [100.0]
    choice = DoaChoice("mac", chip, mac, report=said.append, clock=lambda: now[0])
    assert choice.active == "chip" and choice.latest_doa() == chip.readings[0]
    assert choice.doa_between(0.0, 2.0) == chip.readings and choice.active == "chip"
    assert len(said) == 1 and "no raw frames" in said[0] and "none yet" in said[0]
    mac.feed(raw_frames(np.zeros((4, FRAME_SAMPLES)))[0], 100.0)
    assert choice.active == "mac" and choice.latest_doa() is None, "the Mac's: nothing closed yet"
    assert len(said) == 2 and "the Mac's direction" in said[1]
    now[0] = 103.0
    assert choice.active == "chip" and len(said) == 3 and "none for 3 s" in said[2]
    assert choice.active == "chip" and len(said) == 3, "said once"
    quiet: list[str] = []
    plain = DoaChoice("chip", chip, mac, report=quiet.append)
    assert plain.active == "chip" and plain.latest_doa() == chip.readings[0] and quiet == []
    none = DoaChoice("mac", chip, None, report=quiet.append)
    assert none.active == "chip" and quiet == ["speaker.doa.direction mac but no Mac source: the"
                                              " chip's direction"]  # fmt: skip
    none.configure("chip", voice_p_min=0.4, peak_ratio=0.5, reading_s=0.2)
    assert none.direction == "chip"
    choice.configure("mac", voice_p_min=0.4, peak_ratio=0.5, reading_s=0.2)
    assert (mac.voice_p_min, mac.peak_ratio, mac.reading_s) == (0.4, 0.5, 0.2)


def mac_readings(start: float, deg: int, p: float = 0.9, n: int = 10) -> list[MicReading]:
    """``n`` Mac readings every 0.1 s from ``start`` (board clock): the strongest peak at
    ``deg``, a second at 200, both of a voice Silero gives ``p``."""
    peaks = (Peak(float(deg), 0.8), Peak(200.0, 0.5))
    return [MicReading(start + 0.1 * k, deg, p >= 0.5, voice_p=p, peaks=peaks, chunks=3,
                       voiced=3 if p >= 0.5 else 0) for k in range(n)]  # fmt: skip


def test_a_mac_reading_tags_itself_and_the_silero_rule_takes_it_unchanged() -> None:
    readings = [*mac_readings(2.0, 100), MicReading(3.0, -1, False, voice_p=0.1)]
    tagged = Tagger(VoiceRule("silero", 0.5)).tag(readings, 0.0)  # no series: the reading's own
    assert [t.tags.voice_p for t in tagged] == [0.9] * 10 + [0.1]
    assert [t.voice for t in tagged] == [True] * 10 + [False]
    assert [t.tags.chip for t in tagged] == [True] * 10 + [False]
    assert [t.voice for t in Tagger(CHIP).tag(readings, 0.0)] == [True] * 10 + [False]
    assert all(t.reading.peaks == readings[0].peaks for t in tagged[:10])  # type: ignore[attr-defined]
    strict = Tagger(VoiceRule("silero", 0.95)).tag(readings, 0.0)
    assert not any(t.voice for t in strict), "voice_p_min above the chunks' probability"
    found = utterance_voice(FakeDoa(readings), Heard(2.0, 3.0, "я тут"), 3.0, window_s=1.0,
                            doa_map=MAP, min_speech=2, tagger=Tagger(VoiceRule()))  # fmt: skip
    assert (
        found.voice.bearing_rad == pytest.approx(math.radians(-80.0)) and found.voice.speech == 10
    )


def test_the_register_keeps_two_voices_apart_and_forgets_after_track_forget_s() -> None:
    register = VoiceSources(match_deg=25.0, forget_s=300.0)
    a = register.heard(math.radians(20.0), 0.0, 0.8)
    b = register.heard(math.radians(-60.0), 0.0, 0.5)
    again = register.heard(math.radians(30.0), 1.0, 0.7)
    assert again is a and a.id != b.id and len(register.sources) == 2
    assert a.bearing == math.radians(30.0) and a.peaks == 2 and a.strength == 0.7 and a.who is None
    register.turned(math.radians(10.0))
    assert a.bearing == pytest.approx(math.radians(20.0))
    assert (
        register.nearest(math.radians(25.0)) is a and register.nearest(math.radians(120.0)) is None
    )
    assert register.forget(301.0) == [b] and register.sources == [a], "unheard for over 300 s"
    small = VoiceSources(max_sources=2)
    for k in range(3):
        small.heard(math.radians(60.0 * k), float(k), 0.5)
    assert [s.heard_s for s in small.sources] == [1.0, 2.0], "the stalest goes past the cap"
    small.clear()
    assert small.sources == []


def test_each_utterance_feeds_the_register_and_logs_gaze_sources_without_engaging_anyone() -> None:
    """A wake with Mac readings whose peaks sit at 150 (= -30 deg in the cart's frame) and 200
    (+20): the register holds both voices, the event lists them, the tracks hold the one person
    the wake engaged; the next utterance marks the engaged person's source."""
    doa = FakeDoa(mac_readings(2.0, 150))  # the newest reading stands for now on the board
    events: list[tuple[str, float, dict[str, Any]]] = []
    said: list[str] = []
    door = FakeGazeDoor(view=HeadView(Aim(0.0, 0.4), "", HOME, False, None, False))
    machine = SpeakerGaze(
        SpeakerSettings(doa_map=MAP),
        door=door,
        doa=doa,
        finder=None,
        wheels=NoWheels(),
        face=None,
        reach=Reach.of(NeckConfig.from_json(REPO / "config/neck.json")),
        report=said.append,
        events=lambda kind, now, **f: events.append((kind, now, f)),
    )
    machine.opened(Heard(2.0, 3.0, "эй"), 3.0)
    logged = [e for e in events if e[0] == "gaze_sources"]
    assert len(logged) == 1 and logged[0][2]["fed"] == 20, "ten readings, two peaks each"
    rows = logged[0][2]["sources"]
    assert sorted(r["bearing_deg"] for r in rows) == [-30.0, 20.0]
    assert all(r["who"] is None and r["peaks"] == 10 and not r["engaged"] for r in rows)
    assert {r["strength"] for r in rows} == {0.8, 0.5}
    assert len(machine.tracks.tracks) == 1 and machine.track is not None
    assert machine.track.pan == pytest.approx(math.radians(-30.0)), "the wake engaged the voice"
    doa.readings.extend(mac_readings(3.5, 150, n=12))
    machine.heard(Heard(3.5, 4.6, "и ещё"), 4.6)
    rows = [e for e in events if e[0] == "gaze_sources"][-1][2]["sources"]
    assert [r["engaged"] for r in sorted(rows, key=lambda r: r["bearing_deg"])] == [True, False]
    assert len(machine.sources.sources) == 2 and len(machine.tracks.tracks) == 1
    assert not any("gaze_sources" in line for line in said), "an event, no line"
