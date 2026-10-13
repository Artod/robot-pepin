"""The alive layer: the prosody listener on the voice's own audio (nuclei, their prominence, the
melody, the alignment the word clock uses) and the listeners that post the engine's own requests
(the melody, accent nods, the question tilt, the long reply's aversion, the thinking glance, the
listener's and the acknowledgement's nods, greeting and farewell due, the laugh's rock, the face
found, the breath, the quiet rule), each through a real engine on fakes and a fake clock."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

from __future__ import annotations

import itertools
import json
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pepin.arm_motion import load_arm_motion
from pepin.expression.animation import Book, Clip, Track
from pepin.expression.animator import EngineConfig, GestureEngine
from pepin.expression.book import load_book, with_rest
from pepin.expression.conversation import (
    FAREWELL_DUE,
    GREETING_DUE,
    LISTENER_PAUSE,
    Heard,
    VoiceChunk,
    moment_rank,
)
from pepin.expression.fakes import FakeAttention, FakeClock, FakeExecutor, FakePlayout
from pepin.expression.head import GESTURE, IDLE_LOOK
from pepin.expression.listeners import (
    DUE,
    NOD_PERIOD_S,
    Alive,
    AliveKnobs,
    MomentKnobs,
    PulseKnobs,
    QuietKnobs,
    alive_knobs,
    head_clip,
    head_prep_s,
    held_clip,
    melody_pieces,
    nod_clip,
    spaced,
)
from pepin.expression.markers import Marker, Part
from pepin.expression.parts import clip_look
from pepin.expression.prosody import (
    PartNews,
    ProsodyListener,
    Span,
    Turn,
    VoiceStats,
    band_weights,
    lowpass,
    pin_second,
    snap,
)
from pepin.expression.schedule import AtWord, Nucleus, Priority, Request
from pepin.gaze import GazeSettings, overlay_from_json

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config"
MOTION = load_arm_motion(CONFIG / "arm.json")
BOOK = with_rest(load_book(CONFIG / "animation.json"), MOTION.rest_deg)
RATE = 24000
BOUNDS = GazeSettings()


# -- the prosody listener ------------------------------------------------------------------------


def vowels(centres: list[float], total_s: float, *, f0: float = 220.0) -> Any:
    """Synthetic speech: a voiced vowel (seven harmonics of ``f0``, 0.15 s) at each centre."""
    t = np.arange(round(total_s * RATE)) / RATE
    x = 1e-4 * np.random.default_rng(0).standard_normal(len(t))
    tone = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 8))
    for c in centres:
        envelope = np.sin(np.pi / 2 * np.clip(1 - np.abs(t - c) / 0.075, 0, 1)) ** 2
        x = x + 0.3 * envelope * tone
    return (np.clip(x, -1, 1) * 32767).astype("<i2")


def feed(listener: ProsodyListener, turn: int, index: int, start_s: float, pcm: Any) -> None:
    """``pcm`` as the voice's 0.5 s chunks of part ``index`` starting at stream ``start_s``."""
    n = RATE // 2
    for a in range(0, len(pcm), n):
        piece = pcm[a : a + n]
        end = start_s + (a + len(piece)) / RATE
        listener.audio(VoiceChunk(turn, index, start_s, end, piece.tobytes(), RATE, 0.0))


def test_three_loud_voiced_vowels_are_three_nuclei_at_their_stream_seconds() -> None:
    listener = ProsodyListener()
    feed(listener, 4, 1, 12.0, vowels([0.4, 0.8, 1.2], 1.6))
    news = listener.analyse()
    assert not listener.complete(4, 1) and news[0].start_s == 12.0
    listener.finish(4, 1)
    listener.analyse()
    found = listener.nuclei(4, 1)
    assert [n.second for n in found] == pytest.approx([12.4, 12.8, 13.2], abs=0.02)
    assert all(n.pitch_st is not None and abs(n.pitch_st - 12 * math.log2(220 / 251)) < 0.5
               for n in found)  # fmt: skip
    assert listener.complete(4, 1)
    span = listener.span(4, 1)
    assert span is not None and len(span.pauses) == 2  # the quiet between the vowels
    assert listener.audio_s == pytest.approx(1.6) and listener.cpu_s > 0


def test_the_tap_only_queues_and_drops_everything_while_the_gestures_are_off() -> None:
    rings: list[int] = []
    on = [False]
    listener = ProsodyListener(wake=lambda: rings.append(1), listening=lambda: on[0])
    feed(listener, 1, 0, 0.0, vowels([0.3], 0.6))
    assert not listener.pending() and not rings
    on[0] = True
    feed(listener, 1, 0, 0.0, vowels([0.3], 0.6))
    assert listener.pending() and len(rings) == 2 and listener.audio_s == 0.0  # nothing computed


def test_a_cut_reply_s_nuclei_go_and_the_next_reply_reuses_its_stream_seconds() -> None:
    listener = ProsodyListener()
    feed(listener, 3, 0, 5.0, vowels([0.4, 0.8], 1.2))
    listener.analyse()
    assert listener.nuclei(3, 0)
    listener.drop(3)  # a barge-in: the player flushes and the stream's seconds come again
    feed(listener, 3, 0, 5.0, vowels([0.3], 0.5))  # a late chunk of the cut reply: not taken
    feed(listener, 4, 0, 5.0, vowels([0.6], 1.0))
    listener.finish(4, 0)
    listener.analyse()
    assert listener.nuclei(3, 0) == () and not listener.complete(3, 0)
    assert [n.second for n in listener.nuclei(4, 0)] == pytest.approx([5.6], abs=0.02)


def test_a_part_with_no_audio_is_complete_once_spoken() -> None:
    listener = ProsodyListener()
    listener.finish(2, 1)
    assert listener.analyse() == [] and listener.complete(2, 1)


def test_the_band_passes_the_vowels_formants_and_not_the_rumble() -> None:
    weights = band_weights(RATE, 960)
    t = np.arange(960) / RATE

    def level(hz: float) -> float:
        frame = np.sin(2 * np.pi * hz * t) * np.hanning(960)
        power = np.abs(np.fft.rfft(frame)) ** 2
        return 10 * math.log10(float(np.sum(power * weights)) + 1e-12)

    inside = 10 * math.log10(float(np.mean((np.sin(2 * np.pi * 1000 * t) * np.hanning(960)) ** 2)))
    assert level(1000.0) == pytest.approx(inside, abs=0.3)
    assert level(60.0) < inside - 40 and level(8000.0) < inside - 30


def test_the_contour_s_low_pass_keeps_the_level_and_takes_out_the_flutter() -> None:
    fs = 100.0
    t = np.arange(300) / fs
    slow, fast = np.sin(2 * np.pi * 0.3 * t), 0.5 * np.sin(2 * np.pi * 8.0 * t)
    y = lowpass(3.0 + slow + fast, 1.0, fs)
    assert np.abs(y[30:-30] - (3.0 + slow[30:-30])).max() < 0.1


def test_pinning_puts_a_character_in_its_phrase_and_the_snap_on_the_nearest_nucleus() -> None:
    text = "Вижу, ты сидишь в кресле."
    span = Span((10.0, 12.0), ((10.6, 10.9),))  # one pause, one comma: pinned
    assert pin_second(text, 0, span) == pytest.approx(10.0)
    assert pin_second(text, 5, span) == pytest.approx(10.6)  # the comma ends the first phrase
    second = 10.9 + 1 / (len(text) - 5) * (12.0 - 10.9)
    assert pin_second(text, 6, span) == pytest.approx(second)  # the second starts after the pause
    assert pin_second(text, len(text), span) == pytest.approx(12.0)
    unpinned = Span((10.0, 12.0))  # no pause heard: the share of the voiced span
    assert pin_second(text, 6, unpinned) == pytest.approx(10.0 + 6 / len(text) * 2.0)
    nuclei = [Nucleus(1, 0, 11.30, 1.0), Nucleus(1, 0, 11.62, 0.5)]
    assert snap(11.4, nuclei) == 11.30 and snap(11.5, nuclei) == 11.62
    assert snap(10.9, nuclei) == 10.9  # none within 0.25 s


# -- the clips: every one the arbiter takes, the nod's preparation computed ---------------------


def refusal(clip: Clip) -> str | None:
    look = clip_look(clip, IDLE_LOOK if clip.layer == "idle" else GESTURE)
    assert look is not None
    overlay = overlay_from_json(dict(look["target"]), BOUNDS)
    return overlay if isinstance(overlay, str) else None


def test_every_head_clip_of_the_listeners_passes_the_arbiter_and_the_still_ones_stay_still(
) -> None:  # fmt: skip
    clips = [nod_clip(a, p) for a in (2.0, 4.0) for p in (*NOD_PERIOD_S, 0.5)]
    clips += [held_clip(3.0 * s, -2.0, 0.25, 1.4) for s in (-1, 1)]  # the question tilt
    looks = ((0, -15), (0, 15), (15, 0), (-8, 0))  # the glance up, down, aside; the aversion
    clips += [held_clip(pan, tilt, 0.35, secs) for pan, tilt in looks for secs in (1.0, 3.5)]
    for clip in clips:
        assert refusal(clip) is None, clip
    rng = random.Random(3)
    keys = [(10.0 + 0.27 * k + rng.uniform(-0.04, 0.04), rng.uniform(-0.9, 0.9)) for k in range(30)]
    for piece in melody_pieces(keys, [(12.0, 12.45)], 9.0, 0.25):
        clip = head_clip([(t - piece[0][0], 0.0, v) for t, v in piece], 0.0, layer="idle")
        assert refusal(clip) is None
        look = clip_look(clip, IDLE_LOOK)
        assert look is not None and look["target"]["idle"] is True
        reach = max(abs(k["tilt_rad"]) for k in look["target"]["keys"])
        assert math.degrees(reach) <= BOUNDS.clip_still_deg + 1e-9  # never blinds the head


def test_the_nod_s_floor_is_its_computed_preparation_never_under_beat_lead_min_s() -> None:
    """The arbiter lets a tilt turn back no sooner than 0.2 s after its last turn, the start of a
    move counting (clip_max_hz 2.5): a nod's bottom comes 0.2 s after its first move at the
    earliest, whatever its size; with the head's latency every floor variant needs 0.30 s."""
    for amp in (2.0, 3.0, 4.0):
        for period in NOD_PERIOD_S:
            prep = head_prep_s(nod_clip(amp, period), 0.1)
            assert prep >= PulseKnobs().beat_lead_min_s - 1e-9
    fast = nod_clip(3.0, 0.3)  # a nod with its bottom 0.15 s in: refused, and its floor says so
    assert refusal(fast) is not None and head_prep_s(fast, 0.1) == pytest.approx(0.3)


def test_melody_pieces_skip_the_head_s_own_moves_and_what_is_already_past() -> None:
    keys = [(1.0, 0.0), (1.3, 0.5), (1.6, -0.4), (2.0, 0.3), (2.4, -0.2), (2.8, 0.0)]
    pieces = melody_pieces(keys, [(1.7, 2.1)], 1.2)
    assert pieces[0][0] == (1.2, pytest.approx(0.333))  # what is past cut, on the line to 1.3 s
    assert pieces[0][-1] == (1.6, -0.4)
    assert pieces[1][0] == (pytest.approx(2.15), 0.0) and pieces[1][1] == (2.4, -0.2)
    assert spaced([(0.0, 0), (0.1, 1), (0.3, 0), (0.4, 1)], 0.25) == [(0.0, 0), (0.4, 1)]


# -- the listeners through the engine -------------------------------------------------------------


class Scripted(ProsodyListener):
    """A prosody listener whose analyses are scripted :class:`PartNews`."""

    def __init__(self) -> None:
        super().__init__()
        self.news: deque[PartNews] = deque()

    def pending(self) -> bool:
        return bool(self.news)

    def analyse(self) -> list[PartNews]:
        out = list(self.news)
        self.news.clear()
        return out


def knobs(**changes: Any) -> AliveKnobs:
    """The defaults with some moments' and pulse's knobs changed (by name)."""
    pulse = {k: v for k, v in changes.items() if hasattr(PulseKnobs, k)}
    quiet = {k: v for k, v in changes.items() if hasattr(QuietKnobs, k)}
    moments = {k: v for k, v in changes.items() if k not in pulse and k not in quiet}
    return AliveKnobs(PulseKnobs(**pulse), MomentKnobs(**moments), QuietKnobs(**quiet))


class Rig:
    """The engine on fake parts, the listeners on scripted prosody, one fake clock."""

    def __init__(self, *, book: Book = BOOK, hour: int = 14, seed: int = 0, **changes: Any):
        self.clock = FakeClock(1000.0)
        self.playout = FakePlayout()
        self.parts = {p: FakeExecutor(p) for p in ("arm", "head", "face", "body")}  # type: ignore[arg-type]
        self.parts["arm"].latency_s, self.parts["head"].latency_s = 0.06, 0.1
        self.parts["face"].latency_s = 0.05
        self.records: list[tuple[str, dict[str, Any]]] = []
        self.view = FakeAttention()
        config = EngineConfig(arm_speed_scale=1.0, body_speed_scale=1.0, audio_lag_s=0.15,
                              idle_after_speech_s=0.0)  # fmt: skip
        self.engine = GestureEngine(
            self.parts, self.playout, lambda: (book, config), on=True, clock=self.clock,  # type: ignore[arg-type]
            record=lambda kind, fields: self.records.append((kind, fields)),
            seeds=random.Random(seed), attention=self.view,
        )  # fmt: skip
        self.prosody = Scripted()
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.knobs = knobs(**changes)
        self.alive = Alive(
            self.engine, self.prosody, knobs=lambda: self.knobs, clock=self.clock,
            hour=lambda: hour, seed=seed,
            journal=lambda kind, now, **f: self.events.append((kind, f)),
        )  # fmt: skip
        self.engine.scene_hooks.append(self.alive.scene)
        self.playing = True

    def tick(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds / 0.01))):
            self.clock.now += 0.01
            if self.playing:
                self.playout.now += 0.01
            self.alive.step()
            self.engine.step()

    def open(self) -> None:
        self.alive.opened(Heard(self.clock.now - 1.0, self.clock.now, "Пепин"))
        self.tick(0.01)

    def say(
        self, turn: int, index: int, text: str, *markers: Marker, style: str | None = None
    ) -> None:
        part = Part(text, style, tuple(markers))
        self.engine.part_queued(turn, index, part)
        self.alive.part_queued(turn, index, part)

    def news(self, turn: int, index: int, nuclei: list[tuple[float, float]], *, start: float = 0.0,
             end: float | None = None, complete: bool = True, turns: list[tuple[float, float]] = (),  # type: ignore[assignment]
             onset: float | None = None) -> None:  # fmt: skip
        found = tuple(Nucleus(turn, index, s, z, 0.0) for s, z in nuclei)
        last = max([s for s, _ in nuclei] + [start]) + 0.3
        self.prosody.news.append(PartNews(
            turn, index, start, found, tuple(Turn(s, st) for s, st in turns),
            start + 0.1 if onset is None else onset, last if end is None else end, complete,
            Span((start + 0.1, last if end is None else end)) if complete else None,
        ))  # fmt: skip

    def heard(self, text: str, seconds: float = 2.0, peak_db: float | None = -12.0) -> None:
        self.alive.speech_ended(Heard(self.clock.now - seconds, self.clock.now, text, False,
                                      peak_db))  # fmt: skip
        self.engine.speech_ended(Heard(self.clock.now - seconds, self.clock.now, text))

    def sent(self, intent: str, part: str = "head") -> list[dict[str, Any]]:
        rows = [f for k, f in self.records if k == "cue" and f["part"] == part and f["ok"]]
        return [f for f in rows if f["intent"] == intent]

    def clips(self, part: str = "head") -> list[Clip]:
        return [a[0] for name, a in self.parts[part].calls if name == "play"]

    def clips_of(self, intent: str, part: str = "head") -> list[Clip]:
        """The clips of ``intent`` the part played (its cue rows ok, in order, are its plays)."""
        rows = [f for k, f in self.records if k == "cue" and f["part"] == part and f["ok"]]
        return [c for c, f in zip(self.clips(part), rows, strict=True) if f["intent"] == intent]

    def moments(self, name: str) -> list[dict[str, Any]]:
        return [f for k, f in self.events if k == "moment" and f["name"] == name]


def test_an_accent_nod_on_the_top_nucleus_over_beat_z_once_a_part_with_its_stroke_on_it() -> None:
    rig = Rig()
    rig.open()
    rig.say(1, 0, "Это было огромное здание.")
    rig.news(1, 0, [(2.0, 0.5), (2.4, 2.5), (2.9, 3.4), (3.3, 2.2)], start=1.8)
    rig.tick(0.01)
    rig.tick(3.0)
    (accent,) = rig.sent("accent")
    nod = next(c for c in rig.clips() if c.layer == "gesture")
    # the bottom heard 0.05 s before the top nucleus: sent at stroke - its stroke_s + lag - latency
    assert accent["planned"] == pytest.approx(2.9 - 0.05 - nod.stroke_s + 0.15 - 0.1, abs=0.011)
    amp = max(nod.tracks[0].values)
    assert 3.0 < amp <= 4.0  # z 3.4 over beat_z 2: 2 deg + 0.7 of the range
    pulse = next(f for k, f in rig.events if k == "pulse")
    assert pulse["accent"]["s"] == 2.9 and pulse["z"] == [0.5, 2.5, 3.4, 2.2]


def test_a_part_being_made_takes_its_first_nucleus_over_beat_z_when_its_lead_runs_out() -> None:
    rig = Rig()
    rig.open()
    rig.say(1, 0, "Ну да.")
    rig.news(1, 0, [(1.5, 0.1), (2.0, 2.3)], complete=False)
    rig.tick(0.05)
    assert not rig.sent("accent")
    rig.news(1, 0, [(2.4, 3.5)], complete=False)  # higher, but the first was due first
    rig.tick(2.5)
    (accent,) = rig.sent("accent")
    assert accent["planned"] == pytest.approx(2.0 - 0.05 - 0.2 + 0.15 - 0.1, abs=0.04)


def test_no_nod_with_less_lead_than_its_preparation_and_none_in_a_marked_part() -> None:
    rig = Rig()
    rig.open()
    rig.playout.now = 2.0
    rig.say(1, 0, "Привет.")
    rig.news(1, 0, [(2.19, 3.0)])  # its bottom heard 0.28 s from now: under the nod's own
    rig.say(1, 1, "{laugh} Смешно.", Marker("gesture", "laugh", 0))
    rig.news(1, 1, [(6.0, 3.0)], start=5.5)
    rig.tick(4.0)
    assert not rig.sent("accent")
    pulses = [f for k, f in rig.events if k == "pulse"]
    assert pulses[0]["skipped"][0].startswith("lead 0.28 s under 0.3")
    assert pulses[1]["skipped"] == ["a marker's part"]


def test_accent_nods_two_seconds_apart_and_twelve_a_minute_at_most() -> None:
    rig = Rig()
    rig.open()
    for k in range(16):  # part 1 is 1.5 s after part 0, the rest 3 s apart
        rig.say(1, k, f"Фраза номер {k}.")
        second = 4.5 if k == 1 else 3.0 + 3.0 * k
        rig.news(1, k, [(second, 2.5)], start=second - 0.5)
    rig.tick(0.05)
    pulses = [f for k, f in rig.events if k == "pulse"]
    played = [f for f in pulses if f["accent"]]
    assert len(played) == 12
    assert "a head stroke 1.5 s away (gap 2)" in pulses[1]["skipped"][0]
    assert sum("the per-minute cap 12" in s for f in pulses for s in f["skipped"]) == 3


def test_a_nod_near_a_marker_s_head_stroke_yields() -> None:
    rig = Rig(seed=4)
    rig.open()
    rig.engine.request(Request("agreement", None, Priority.MARK, "marker"))
    rig.tick(0.05)
    mark = rig.engine.scheduler.reservations.held["head"]
    assert mark.performance.intent == "agreement"
    lead = mark.stroke + 0.6 - rig.clock.now  # a nucleus heard 0.6 s after the marker's stroke
    rig.say(1, 0, "Да, конечно.")
    rig.news(1, 0, [(rig.playout.now + lead - 0.15 + 0.05, 3.0)], start=rig.playout.now)
    rig.tick(0.05)
    pulse = next(f for k, f in rig.events if k == "pulse")
    assert not pulse["accent"] and "agreement" in " ".join(pulse["skipped"])


def test_the_melody_is_a_still_idle_tilt_within_its_knob_and_comes_back_after_a_nod() -> None:
    rig = Rig()
    rig.open()
    rig.say(1, 0, "Мы поедем завтра на дачу, если будет солнце.")
    turns = [(1.5 + 0.3 * k, 6.0 * (-1) ** k) for k in range(12)]  # ±6 st every 0.3 s
    rig.news(1, 0, [(2.6, 3.0)], start=1.0, onset=1.2, end=5.3, turns=turns)
    rig.tick(6.0)
    melody = rig.clips_of("melody")
    assert melody and all(refusal(c) is None for c in melody)
    for clip in melody:
        (track,) = clip.tracks
        assert max(abs(v) for v in track.values) <= 0.9
        assert min(np.diff(track.times)) >= 0.25 - 1e-9
    starts = [f["planned"] for f in rig.sent("melody")]
    (accent,) = rig.sent("accent")
    nod_end = 2.55 - 0.2 + 0.42  # the nod's end on the stream (its period at most 0.45)
    assert any(s > accent["planned"] for s in starts), "the melody again after the nod"
    after = min(s for s in starts if s > accent["planned"])
    assert after >= nod_end - 0.2 + 0.15 - 0.1


def test_the_question_tilt_at_a_question_s_last_nucleus_held_past_its_audio() -> None:
    rig = Rig()
    rig.open()
    rig.say(1, 0, "А тебе это нравится?")
    rig.news(1, 0, [(1.4, 0.2), (2.1, 0.1)], start=1.0, end=2.4)
    rig.tick(4.0)
    (tilt,) = rig.sent("question")
    clip = next(c for c in rig.clips() if c.layer == "gesture")
    assert tilt["planned"] == pytest.approx(2.1 - 0.05 - 0.25 + 0.15 - 0.1, abs=0.011)
    pan = next(t for t in clip.tracks if t.dof == "pan_deg")
    held_until = 2.05 - clip.stroke_s + pan.times[2]  # its hold ends, on the stream
    assert held_until == pytest.approx(2.4 + 0.5, abs=0.01)


def test_a_long_reply_looks_away_once_at_an_inner_pause_and_back_before_its_end() -> None:
    rig = Rig()
    rig.open()
    rig.say(1, 0, "Сначала одно.")
    rig.news(1, 0, [(1.3 + 0.3 * k, 0.0) for k in range(6)], start=1.0, end=3.0)
    rig.say(1, 1, "Потом другое, и ещё третье.")
    rig.news(1, 1, [(3.6 + 0.3 * k, 0.0) for k in range(14)], start=3.2, end=8.0)
    rig.tick(9.0)
    (aversion,) = rig.sent("aversion")
    (moment,) = rig.moments("long_reply")
    assert moment["played"] and aversion["planned"] == pytest.approx(2.9 + 0.15 - 0.1, abs=0.011)
    clip = next(c for c in rig.clips() if c.layer == "gesture")
    assert 2.9 + clip.duration_s + 2.4 <= 8.0 + 1e-9


def test_the_thinking_glance_at_0_8_s_back_into_the_reply_never_the_same_way_twice() -> None:
    rig = Rig(ack_nod_p=0.0, think_p=1.0)
    rig.open()
    directions: list[str] = []
    for turn in range(1, 7):
        rig.heard("Расскажи что-нибудь.")
        rig.tick(0.75)
        assert not rig.sent("think_glance")[len(directions) :]
        rig.tick(0.1)
        glance = rig.moments("think_wait")[-1]
        assert glance["played"]
        directions.append(glance["dir"])
        # back 0.5-1.3 s into the reply as the waits predict it: the first, 1.95 s (MEASURED
        # p50), then this conversation's (0.85 + 1.2 s)
        wait = 1.95 if turn == 1 else 2.05
        assert wait + 0.5 - 0.82 <= glance["seconds"] <= wait + 1.3 - 0.8 + 1e-6
        rig.tick(1.2)
        rig.alive.voice_started(turn)
        rig.tick(3.0)
    assert all(a != b for a, b in itertools.pairwise(directions))
    clip = next(c for c in rig.clips() if c.layer == "gesture")
    assert refusal(clip) is None


def test_no_glance_when_the_reply_s_audio_came_first_or_the_person_goes_on() -> None:
    rig = Rig(ack_nod_p=0.0, think_p=1.0)
    rig.open()
    rig.heard("Сколько времени?")
    rig.tick(0.4)
    rig.alive.voice_started(1)
    rig.tick(1.0)
    rig.heard("Ну и вот.")
    rig.tick(0.3)
    rig.alive.speech_started(rig.clock.now)
    rig.tick(1.0)
    assert not rig.sent("think_glance")
    whys = [m["why"] for m in rig.moments("think_wait")]
    assert whys == ["the reply's audio came first", "the person spoke again"]


def test_a_listener_nod_after_two_seconds_of_their_speech_one_per_five_and_never_after_a_question(
) -> None:  # fmt: skip
    rig = Rig(listener_nod_p=1.0, ack_nod_p=0.0, think_p=0.0)
    rig.open()
    rig.alive.speech_started(rig.clock.now)
    rig.tick(1.0)
    rig.alive.moment(LISTENER_PAUSE)
    rig.tick(1.5)
    rig.alive.moment(LISTENER_PAUSE)
    rig.tick(2.0)
    rig.alive.moment(LISTENER_PAUSE)
    rig.tick(0.5)
    assert len(rig.sent("listen_nod")) == 1
    assert [m["why"].split(" ")[-1] for m in rig.moments(LISTENER_PAUSE)] == ["speech", "", "ago"]
    rig.heard("Где у нас принтер?")
    rig.alive.speech_started(rig.clock.now)
    rig.tick(6.0)
    rig.alive.moment(LISTENER_PAUSE)
    rig.tick(0.1)
    assert rig.moments(LISTENER_PAUSE)[-1]["why"] == "after their question"


def test_the_listener_nod_reads_the_person_s_words_so_far_when_the_pause_brings_them() -> None:
    """The ears' early read of the pause rides with it: a question so far gets no nod, and a
    statement so far gets one even right after their question closed."""
    rig = Rig(listener_nod_p=1.0, ack_nod_p=0.0, think_p=0.0)
    rig.open()
    rig.alive.speech_started(rig.clock.now)
    rig.tick(2.5)
    rig.alive.moment(LISTENER_PAUSE, "А где у нас принтер?")
    rig.tick(0.1)
    assert rig.moments(LISTENER_PAUSE)[-1]["why"] == "their words so far are a question"
    rig.heard("Где у нас принтер?")
    rig.alive.speech_started(rig.clock.now)
    rig.tick(6.0)
    rig.alive.moment(LISTENER_PAUSE, "Ну я тебе так скажу")
    rig.tick(0.1)
    assert rig.moments(LISTENER_PAUSE)[-1]["played"] and len(rig.sent("listen_nod")) == 1


def test_a_reply_sooner_than_the_waits_predict_ends_the_thinking_glance_through_the_engine(
) -> None:  # fmt: skip
    """The glance holds until think_back_s into the reply as the waits predict it (1.95 s at
    first); a reply that starts 1.05 s after the close ends it think_back_s into its first
    sound: the listeners release their own request and the head eases back."""
    rig = Rig(ack_nod_p=0.0, think_p=1.0)
    rig.open()
    rig.heard("Расскажи что-нибудь.")
    rig.tick(0.85)
    (glance,) = rig.moments("think_wait")
    assert glance["played"] and len(rig.sent("think_glance")) == 1
    rig.tick(0.2)
    rig.say(1, 0, "Ну вот.")
    rig.alive.part_started(1, 0, rig.playout.now)
    rig.tick(1.6)
    faded = [f for k, f in rig.records if k == "faded" and f["intent"] == "think_glance"]
    assert "head" in {f["part"] for f in faded}  # and its thinking face, when it has one
    assert all(f["why"] == "the reply's voice" for f in faded)
    assert ("fade", ("gesture", 0.35)) in rig.parts["head"].calls


def test_a_nod_keeps_its_gap_from_a_mark_still_waiting_for_its_word_in_the_next_sentence(
) -> None:  # fmt: skip
    """A mark of the next sentence waits for its word in the engine; a nod of this sentence
    whose bottom would come 0.5 s before the mark's head stroke yields (the engine's pending)."""
    rig = Rig()
    rig.open()
    rig.say(1, 0, "Первая фраза у нас тут.")
    rig.say(1, 1, "Вторая.")
    rig.engine.request(Request("agreement", AtWord(1, 1, 0), Priority.MARK, "marker",
                               clips=(nod_clip(3.0, 0.4),)))  # fmt: skip
    rig.tick(0.02)
    (mark,) = rig.engine.pending("head", turn=1)
    second = rig.playout.now + (mark.stroke - 0.5 - rig.clock.now) - 0.15 + 0.05
    rig.news(1, 0, [(second, 3.0)])
    rig.tick(0.05)
    pulse = next(f for k, f in rig.events if k == "pulse")
    assert not pulse["accent"] and "agreement's head stroke 0.5 s away (waiting)" in " ".join(
        pulse["skipped"]
    )


def test_an_acknowledgement_after_a_statement_never_after_a_question() -> None:
    rig = Rig(ack_nod_p=1.0, think_p=0.0)
    rig.open()
    rig.heard("Я сегодня весь день чинил колесо.", seconds=2.5)
    rig.tick(0.5)
    assert len(rig.sent("ack_nod")) == 1
    rig.heard("А почему оно сломалось?", seconds=2.5)
    rig.tick(0.5)
    rig.heard("Ага.", seconds=0.6)
    rig.tick(0.5)
    assert len(rig.sent("ack_nod")) == 1
    assert [m["why"] for m in rig.moments("acknowledged")][1:] == ["a question", "0.6 s"]


def test_the_glance_keeps_the_head_gap_after_an_acknowledgement() -> None:
    rig = Rig(ack_nod_p=1.0, think_p=1.0, head_gap_s=1.5)
    rig.open()
    rig.heard("Я сегодня весь день чинил колесо.", seconds=2.5)
    sends: list[float] = []  # when each gesture went to the head
    for _ in range(200):
        before = len([c for c in rig.clips() if c.layer == "gesture"])
        rig.tick(0.01)
        if len([c for c in rig.clips() if c.layer == "gesture"]) > before:
            sends.append(rig.clock.now)
    (ack,) = rig.clips_of("ack_nod")
    (glance,) = rig.clips_of("think_glance")
    ack_stroke = sends[0] + 0.1 + ack.stroke_s
    glance_stroke = sends[1] + 0.1 + glance.stroke_s
    assert glance_stroke - ack_stroke >= 1.5 - 0.011  # delayed to keep head_gap_s, not dropped


def test_hello_from_the_person_s_words_in_the_wait_absorbs_the_brain_s_later_greeting() -> None:
    """«Пепин, привет!»: the hello plays 0.2 s after the close, before the reply's first word;
    the brain's own {greeting} in the reply is absorbed by the composition's cooldown."""
    rig = Rig(seed=2)
    rig.view.be(phase="track", seen_s=rig.clock.now + 100, eyes=False)
    rig.alive.opened(Heard(rig.clock.now - 1.5, rig.clock.now, "Пепин, привет!"))
    rig.heard("Пепин, привет!", seconds=1.5)
    rig.tick(0.15)
    assert not [f for k, f in rig.records if k == "performance"]
    rig.tick(0.1)
    performance = next(f for k, f in rig.records if k == "performance")
    assert performance["intent"] == "greeting" and performance["source"] == "moment"
    rig.say(1, 0, "Привет, Артём!", Marker("gesture", "greeting", 0))
    rig.tick(0.05)
    assert next(f for k, f in rig.records if k == "capped")["why"] == "cooldown 30 s"
    assert rig.moments(GREETING_DUE)[0]["played"]


def test_a_goodbye_waves_bye_in_the_wait_and_a_greeting_waits_for_the_body_s_turn() -> None:
    rig = Rig(seed=1)
    rig.open()
    rig.heard("Ладно, пока, Пепин.", seconds=1.5)
    rig.tick(0.3)
    assert [f["intent"] for k, f in rig.records if k == "performance"] == ["farewell"]
    rig = Rig(seed=1)
    rig.open()
    rig.view.be(phase="turning")
    rig.heard("Привет!", seconds=1.0)
    rig.tick(1.0)
    assert not [f for k, f in rig.records if k == "performance"]
    rig.view.be(phase="track")
    rig.tick(0.2)
    assert [f["intent"] for k, f in rig.records if k == "performance"] == ["greeting"]
    assert rig.moments(GREETING_DUE)[0]["deferred_s"] >= 0.8


def test_the_laugh_never_uses_the_arm_and_rocks_the_cart_once_from_its_part_s_start() -> None:
    asked = []
    for seed in range(12):  # a person's ask of the laugh plays whole, its arm included
        rig = Rig(seed=seed)
        assert rig.engine.perform("laugh", source="tool").ok
        last = rig.engine.last()
        asked += [c.part for c in last.clips] if last is not None else []
    assert "arm" in asked
    for seed in range(12):
        rig = Rig(seed=seed)
        rig.open()
        rig.say(1, 0, "Ха, смешно вышло.", Marker("gesture", "laugh", 0))
        rig.tick(0.02)
        laugh = rig.engine.last()
        assert laugh is not None and laugh.intent == "laugh"
        assert all(c.part != "arm" for c in laugh.clips)
        rig.alive.part_started(1, 0, rig.playout.now)
        rig.tick(1.5)
        rocks = rig.sent("laugh_rock", "body")
        if any(c.part == "body" for c in laugh.clips):
            assert not rocks and rig.moments("laugh_rock")[0]["why"].startswith("the laugh moves")
        else:
            assert len(rocks) == 1
            clip = next(c for c in rig.clips("body") if c.tracks[0].dof == "x_m")
            assert 0.01 <= max(clip.tracks[0].values) <= 0.02 and clip.stroke_s <= 2.5


def test_the_accent_s_rock_takes_the_reply_s_strongest_reachable_accent_once() -> None:
    """Nuclei over beat_z at 0.5 s (the strongest, but in the reply's first half second of
    sound with its rock's way), 1.6, 2.4 (stronger than 1.6) and 3.0: the cart rocks 1 cm once,
    its stroke on 2.4; a later part of the same reply rocks nothing; accent_rock_p 0 never."""
    rig = Rig(accent_rock_p=1.0)
    rig.open()
    rig.say(1, 0, "Это было огромное здание, правда.")
    rig.news(1, 0, [(0.5, 4.0), (1.6, 2.4), (2.4, 3.6), (3.0, 2.2)], start=0.2)
    rig.tick(3.5)
    (rock,) = rig.sent("accent_rock", "body")
    (clip,) = rig.clips_of("accent_rock", "body")
    assert max(clip.tracks[0].values) == pytest.approx(0.01) and clip.stroke_s == pytest.approx(
        0.45
    )
    assert rock["planned"] == pytest.approx(2.4 - 0.05 - 0.45 + 0.15, abs=0.011)
    (moment,) = rig.moments("accent_rock")
    assert moment["played"] and moment["s"] == 2.4 and moment["z"] == 3.6
    rig.say(1, 1, "И ещё одно.")
    rig.news(1, 1, [(rig.playout.now + 2.0, 4.0)], start=rig.playout.now + 0.5)
    rig.tick(3.0)
    assert len(rig.sent("accent_rock", "body")) == 1
    never = Rig(accent_rock_p=0.0)
    never.open()
    never.say(1, 0, "Это было огромное здание.")
    never.news(1, 0, [(2.4, 3.6)], start=0.2)
    never.tick(3.0)
    never.alive.reply_done(1)
    never.tick(0.1)
    assert not never.sent("accent_rock", "body") and not never.moments("accent_rock")


def test_the_accent_s_rock_waits_out_a_moving_arm_and_never_comes_in_quiet() -> None:
    """While the arm plays a mark the accent at 1.6 s passes (the board refuses the wheels while
    the arm moves); the next one, at 4.5 s, rocks the cart. In quiet hours: none, logged."""
    rig = Rig(accent_rock_p=1.0)
    rig.open()
    lift = Track("shoulder_lift", "additive", (0.0, 1.5, 3.0), (0.0, 20.0, 0.0))
    rig.engine.request(Request("mark", None, Priority.MARK, "marker",
                               clips=(Clip("arm", 0.0, 1.5, 3.0, (lift,)),)))  # fmt: skip
    rig.say(1, 0, "Это было огромное здание, правда, очень.")
    rig.news(1, 0, [(1.6, 3.0), (4.5, 2.5)], start=0.2)
    rig.tick(5.0)
    (moment,) = rig.moments("accent_rock")
    assert moment["played"] and moment["s"] == 4.5
    assert len(rig.sent("accent_rock", "body")) == 1
    quiet = Rig(hour=23, accent_rock_p=1.0)
    quiet.open()
    quiet.say(1, 0, "Это было огромное здание.")
    quiet.news(1, 0, [(2.4, 3.6)], start=0.2)
    quiet.tick(3.0)
    assert not quiet.sent("accent_rock", "body")
    assert quiet.moments("accent_rock")[0]["why"] == "quiet: the hours 23-08"


def test_the_first_clear_face_brings_one_smile() -> None:
    rig = Rig()
    rig.open()
    rig.tick(0.5)
    assert not rig.sent("smile", "face")
    rig.view.be(first_face_s=rig.clock.now)
    rig.tick(1.0)
    rig.view.be(first_face_s=rig.clock.now)
    rig.tick(1.0)
    assert len(rig.sent("smile", "face")) == 1
    face = next(c for c in rig.clips("face") if c.events and c.events[0].name == "smile")
    assert face.events[0].hold_s == 2.0


def test_the_breath_fills_the_silent_head_survives_the_person_s_start_and_stops_for_the_voice(
) -> None:  # fmt: skip
    """The book's breath (the engine's looping idle entry, 1.0 deg at 0.25 Hz, as the alive
    layer's own breath was): within the arbiter's still overlays, through the person's start,
    faded while the robot speaks, back right after its reply."""
    rig = Rig()
    rig.open()
    rig.engine.opened(Heard(rig.clock.now - 1.0, rig.clock.now, "Пепин"))
    rig.tick(0.3)
    (first,) = [c for c in rig.clips() if c.layer == "idle"]
    assert first.loop and refusal(first) is None
    assert max(first.tracks[0].values) == pytest.approx(
        2.0
    )  # the book's breath, 2 deg since 2026-10-13
    rig.alive.speech_started(rig.clock.now)
    rig.engine.speech_started(rig.clock.now)
    rig.tick(0.5)
    assert len(rig.sent("idle:breath")) == 1, "it plays on through the person's start"
    assert not [c for c in rig.parts["head"].calls if c[0] == "fade"]
    rig.say(1, 0, "Да.")
    rig.tick(9.0)
    assert ("fade", ("idle", 0.35)) in rig.parts["head"].calls, "the robot speaks: it fades"
    assert len(rig.sent("idle:breath")) == 1, "none while the robot's voice is out"
    rig.alive.reply_done(1)
    rig.engine.reply_done(1)
    rig.tick(0.3)
    assert len(rig.sent("idle:breath")) == 2


def test_quiet_leaves_the_melody_and_the_face() -> None:
    rig = Rig(hour=23, listener_nod_p=1.0, ack_nod_p=1.0, think_p=1.0)
    rig.open()
    rig.heard("Я тут посижу немного, ладно.", seconds=2.5)
    rig.tick(1.0)
    rig.alive.voice_started(1)
    rig.say(1, 0, "Конечно.")
    rig.news(1, 0, [(2.0, 3.5)], start=1.0, end=2.5, turns=[(1.5, 4.0), (1.9, -4.0)])
    rig.tick(3.0)
    rig.engine.request(Request("greeting", None, Priority.MARK, "marker"))
    rig.tick(0.05)
    gestures = [c for c in rig.clips() if c.layer == "gesture"]
    assert not gestures and not rig.clips("arm")
    assert rig.sent("melody") and rig.clips("face")
    assert next(f for k, f in rig.events if k == "quiet")["why"] == "the hours 23-08"
    style = Rig()
    style.open()
    style.say(1, 0, "Тихо.", style="whispering, slow")
    style.tick(0.01)
    assert next(f for k, f in style.events if k == "quiet")["why"] == "the style slow, whispering"
    voice = Rig()
    voice.open()
    voice.heard("Спокойной ночи.", peak_db=-34.0)
    voice.tick(0.01)
    assert next(f for k, f in voice.events if k == "quiet")["why"] == "their voice -34 dBFS"


def test_a_wrist_beat_inside_an_open_unit_and_none_without_one() -> None:
    rig = Rig(wrist_beat_p=1.0)
    rig.open()
    rig.parts["arm"].held = {"wrist_flex": 70.0}  # type: ignore[attr-defined]
    raise_ = Track("shoulder_lift", "additive", (0.0, 0.3, 0.5), (0.0, 20.0, 20.0))
    unit = Clip("arm", 0.0, 0.3, 0.5, (raise_,), then="hold", hold_s=2.5)
    rig.engine.request(Request("unit", None, Priority.MARK, "marker", clips=(unit,)))
    rig.tick(0.6)
    rig.say(1, 0, "Вот так.")
    rig.news(1, 0, [(rig.playout.now + 1.0, 3.0)], start=rig.playout.now)
    rig.tick(2.0)
    assert len(rig.sent("wrist_beat", "arm")) == 1
    clip = rig.clips("arm")[-1]
    assert clip.then == "hold" and 0 < clip.hold_s <= 2.5 and not rig.sent("accent")
    rig = Rig(wrist_beat_p=1.0)
    rig.open()
    rig.say(1, 0, "Вот так.")
    rig.news(1, 0, [(2.0, 3.0)], start=1.0)
    rig.tick(2.0)
    assert rig.sent("accent") and not rig.sent("wrist_beat", "arm")


def test_off_nothing_moves() -> None:
    rig = Rig(listener_nod_p=1.0, ack_nod_p=1.0, think_p=1.0)
    rig.engine.switch(False)
    rig.open()
    rig.heard("Я весь день чинил колесо.", seconds=2.5)
    rig.say(1, 0, "Понятно?")
    rig.news(1, 0, [(2.0, 3.5)], start=1.0, turns=[(1.5, 4.0)])
    rig.tick(3.0)
    assert not any(c for p in ("arm", "head", "face", "body") for c in rig.clips(p))
    assert not rig.prosody.listening()


def test_the_close_logs_each_part_and_the_counts_a_minute() -> None:
    rig = Rig()
    rig.open()
    rig.say(1, 0, "Раз, два.")
    rig.news(1, 0, [(2.0, 3.0)], start=1.0, end=2.5)
    rig.tick(3.0)
    rig.alive.closed("idle")
    rig.tick(0.01)
    summary = next(f for k, f in rig.events if k == "alive")
    assert summary["counts"]["accent"] == 1 and summary["reason"] == "idle"
    assert summary["accents_per_speech_min"] > 0 and "head_moves_per_min" in summary


def test_the_listeners_thread_takes_the_calls_and_stops() -> None:
    rig = Rig()
    rig.alive.clock = time.monotonic
    rig.alive.start()
    try:
        rig.alive.opened(Heard(0.0, 1.0, "Пепин"))
        deadline = time.monotonic() + 1.0
        while not rig.alive._conversation and time.monotonic() < deadline:
            time.sleep(0.005)
        assert rig.alive._conversation
    finally:
        rig.alive.close()
    assert rig.alive._thread is not None and not rig.alive._thread.is_alive()


def test_the_knobs_load_from_config_and_the_moments_due_name_the_book_s_compositions() -> None:
    data = json.loads((CONFIG / "expression.json").read_text())
    knobs_ = alive_knobs(data)
    assert (
        knobs_.pulse.beat_lead_min_s == 0.30 and knobs_.pulse.melody_max_deg == 0.0
    )  # off since 2026-10-13: the +-0.9 deg melody read as jerky
    assert knobs_.moments.head_gap_s == 1.0 and "breath_deg" not in data["moments"]
    assert knobs_.pulse.wrist_beat_deg == (15.0, 25.0)  # 2026-10-13: 6-8 deg did not show
    assert knobs_.pulse.accent_rock_p == 0.5 and knobs_.pulse.arm_up_min_sentences == 2
    (breath,) = [i for i in BOOK.idle if "breath" in i.atoms]  # the breath is the book's
    assert breath.unless == ("speaking",) and breath.after_speech_s == 0.0
    assert (
        BOOK.atoms["breath"].loop and BOOK.atoms["breath"].params["amp_deg"].hi == 2.0
    )  # 2026-10-13: 1 deg read as jerky ticks
    assert knobs_.pulse.voice_stats == VoiceStats() and knobs_.quiet.hours == (23, 8)
    assert alive_knobs({}) == AliveKnobs()
    assert DUE == {GREETING_DUE: "greeting", FAREWELL_DUE: "farewell"}
    assert set(DUE.values()) <= set(BOOK.compositions)
    assert moment_rank(GREETING_DUE) < moment_rank(LISTENER_PAUSE)


# -- offline: a recorded reply ---------------------------------------------------------------------

RECORDED = sorted((REPO / "data" / "voice").glob("2026101[0-2]/cascade_*/out.wav"))


@pytest.mark.skipif(not RECORDED, reason="no recorded voice under data/voice")
def test_a_recorded_reply_gives_nuclei_a_melody_the_arbiter_takes_and_costs_little() -> None:
    import soundfile as sf

    pcm, rate = sf.read(RECORDED[-1], dtype="int16")
    pcm = pcm[: rate * 20]
    listener = ProsodyListener()
    n = rate // 2
    news: list[PartNews] = []
    for a in range(0, len(pcm), n):
        piece = pcm[a : a + n]
        listener.audio(VoiceChunk(1, 0, 0.0, (a + len(piece)) / rate, piece.tobytes(), rate, 0.0))
        news += listener.analyse()
    listener.finish(1, 0)
    news += listener.analyse()
    seconds = len(pcm) / rate
    found = listener.nuclei(1, 0)
    assert 1.0 <= len(found) / seconds <= 8.0  # syllables a second of the voice
    assert listener.cpu_s * 1000.0 / listener.audio_s < 10.0  # MEASURED 1.2-1.3 ms a second
    turns = [t for item in news for t in item.turns]
    keys = [(turns[0].second - 0.3, 0.0)] + [(t.second, max(-0.9, min(0.9, -0.2 * t.st)))
                                             for t in turns]  # fmt: skip
    for piece in melody_pieces(spaced(keys, 0.25), [], -1.0):
        clip = head_clip([(t - piece[0][0], 0.0, v) for t, v in piece], 0.0, layer="idle")
        assert refusal(clip) is None
