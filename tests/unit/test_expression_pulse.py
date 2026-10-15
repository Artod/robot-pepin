"""The alive layer: the prosody listener on the voice's own audio (nuclei, their prominence, the
melody, the alignment the word clock uses) and the listeners that post the engine's own requests
(the melody, accent nods, the question tilt, the long reply's aversion, the thinking glance, the
listener's and the acknowledgement's nods, none of the engine's own for the person's hello or
bye, the laugh's rock, the face found, the breath, the quiet rule), each through a real engine
on fakes and a fake clock."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

from __future__ import annotations

import itertools
import json
import math
import random
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pepin.arm_motion import load_arm_motion
from pepin.expression.animation import PARTS, Book, Clip, Scene, Track
from pepin.expression.animator import EngineConfig, GestureEngine
from pepin.expression.book import load_book, loops, span, with_rest
from pepin.expression.conversation import (
    ACKNOWLEDGED,
    LISTENER_PAUSE,
    THINK_WAIT,
    Heard,
    VoiceChunk,
)
from pepin.expression.fakes import FakeAttention, FakeClock, FakeExecutor, FakePlayout
from pepin.expression.head import GESTURE, IDLE_LOOK
from pepin.expression.listeners import (
    ACK_NOD,
    LAUGHING,
    LISTEN_NOD,
    PEAK,
    STRONG,
    WEAK,
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
from pepin.expression.render import BookRenderer
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
    """The nods of the pulse's and the moments' ensembles (the book's), the held looks, and
    the melody's pieces at keys 0.5 s apart within the still overlay's bound."""
    renderer = BookRenderer(BOOK)
    head = Scene(frozenset({"head"}))
    clips = [c for intent in (WEAK, STRONG, PEAK, LAUGHING, LISTEN_NOD, ACK_NOD)
             for seed in range(20) for c in renderer.render(intent, head, seed).clips]  # fmt: skip
    assert len(clips) == 120 and {c.part for c in clips} == {"head"}
    clips += [held_clip(3.0 * s, -2.0, 0.25, 1.4) for s in (-1, 1)]  # the question tilt
    looks = ((0, -15), (0, 15), (15, 0), (-8, 0))  # the glance up, down, aside; the aversion
    clips += [held_clip(pan, tilt, 0.35, secs) for pan, tilt in looks for secs in (1.0, 3.5)]
    for clip in clips:
        assert refusal(clip) is None, clip
    rng = random.Random(3)
    keys = [(10.0 + 0.27 * k + rng.uniform(-0.04, 0.04), rng.uniform(-0.9, 0.9)) for k in range(30)]
    for piece in melody_pieces(keys, [(12.0, 12.45)], 9.0, 0.5):
        clip = head_clip([(t - piece[0][0], 0.0, v) for t, v in piece], 0.0, layer="idle")
        assert refusal(clip) is None
        look = clip_look(clip, IDLE_LOOK)
        assert look is not None and look["target"]["idle"] is True
        reach = max(abs(k["tilt_rad"]) for k in look["target"]["keys"])
        assert math.degrees(reach) <= BOUNDS.clip_still_deg + 1e-9  # never blinds the head


def test_the_nod_s_floor_is_its_computed_preparation_never_under_beat_lead_min_s() -> None:
    """The arbiter lets a tilt turn back no sooner than 0.2 s after its last turn, the start of a
    move counting (clip_max_hz 2.5): a nod's bottom comes 0.2 s after its first move at the
    earliest, whatever its size; with the head's latency every ensemble's nod needs 0.30 s at
    least, its own way down (the book's widest) when that is longer."""
    rig = Rig()
    for intent in (WEAK, STRONG, PEAK, LAUGHING):
        pre = span(BOOK, intent, "head")[0]
        floor = rig.alive._head_floor(intent)
        assert floor == pytest.approx(max(0.30, 0.1 + max(pre, 0.2)))
        assert floor >= PulseKnobs().beat_lead_min_s - 1e-9
    fast = head_clip(((0.0, 0.0, 0.0), (0.15, 0.0, 3.0), (0.3, 0.0, 0.0)), 0.15)
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


def test_an_accent_on_the_top_nucleus_over_beat_z_once_a_part_its_ensemble_on_it() -> None:
    """The part's audio all known: its top nucleus (z 3.4: the reply's strongest, over strong_z)
    is the peak, the nod's bottom and the cart's forward stroke on it."""
    rig = Rig()
    rig.open()
    rig.say(1, 0, "Это было огромное здание.")
    rig.news(1, 0, [(2.0, 0.5), (2.4, 2.5), (2.9, 3.4), (3.3, 2.2)], start=1.8)
    rig.tick(0.01)
    rig.tick(3.0)
    (accent,) = rig.sent(PEAK)
    (nod,) = rig.clips_of(PEAK)
    # the bottom heard 0.05 s before the top nucleus: sent at stroke - its stroke_s + lag - latency
    assert accent["planned"] == pytest.approx(2.9 - 0.05 - nod.stroke_s + 0.15 - 0.1, abs=0.011)
    assert 3.5 <= max(nod.tracks[0].values) <= 4.5  # the book's peak nod
    (rock,) = rig.clips_of(PEAK, "body")
    (forward,) = rock.tracks
    assert 0.012 <= max(forward.values) <= 0.015
    assert rock.offset_s + rock.stroke_s == pytest.approx(0.0)  # its stroke on the nod's
    pulse = next(f for k, f in rig.events if k == "pulse")
    assert pulse["accent"]["s"] == 2.9 and pulse["accent"]["kind"] == "peak"
    assert pulse["accent"]["parts"] == ["body", "head"] and pulse["z"] == [0.5, 2.5, 3.4, 2.2]


def test_a_part_being_made_takes_its_first_nucleus_over_beat_z_when_its_lead_runs_out() -> None:
    rig = Rig()
    rig.open()
    rig.say(1, 0, "Ну да.")
    rig.news(1, 0, [(1.5, 0.1), (2.0, 2.3)], complete=False)
    rig.tick(0.05)
    assert not rig.sent("accent")
    rig.news(1, 0, [(2.4, 3.5)], complete=False)  # higher, but the first was due first
    rig.tick(2.5)
    (accent,) = rig.sent(WEAK)  # z 2.3: under strong_z
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
    assert not [f for k, f in rig.records if k == "cue" and f["intent"] in (WEAK, STRONG, PEAK)]
    pulses = [f for k, f in rig.events if k == "pulse"]
    assert pulses[0]["skipped"][0].startswith("head: lead 0.28 s under 0.3")
    assert pulses[1]["skipped"] == ["a marker's part"]


def test_accent_nods_two_seconds_apart_and_twelve_a_minute_at_most() -> None:
    """The nods keep their gap and their cap; an accent whose nod they refuse still moves the
    rest of its ensemble (the cart's sway on its share)."""
    rig = Rig()
    rig.open()
    for k in range(16):  # part 1 is 1.5 s after part 0, the rest 3 s apart
        rig.say(1, k, f"Фраза номер {k}.")
        second = 4.5 if k == 1 else 3.0 + 3.0 * k
        rig.news(1, k, [(second, 2.5)], start=second - 0.5)
    rig.tick(0.05)
    pulses = [f for k, f in rig.events if k == "pulse"]
    nods = [f for f in pulses if f["accent"] and "head" in f["accent"]["parts"]]
    assert len(nods) == 12
    assert "head: a head stroke 1.5 s away (gap 2)" in pulses[1]["skipped"][0]
    assert pulses[1]["accent"]["parts"] == ["body"], "its sway still goes"
    assert sum("the per-minute cap 12" in s for f in pulses for s in f["skipped"]) == 3


def test_a_nod_near_a_marker_s_head_stroke_yields() -> None:
    agreement = BOOK.compositions["agreement"]
    floors = tuple(v for v in agreement.variants if v.floor)  # its head at once
    floored = replace(agreement, variants=floors)
    book = replace(BOOK, compositions={**BOOK.compositions, "agreement": floored})
    rig = Rig(book=book, seed=4)
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
    assert "head" not in (pulse["accent"] or {}).get("parts", [])
    assert "agreement" in " ".join(pulse["skipped"])


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
        assert min(np.diff(track.times)) >= 0.5 - 1e-9
    starts = [f["planned"] for f in rig.sent("melody")]
    (accent,) = rig.sent(PEAK)
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


def test_the_listener_s_nod_and_the_acknowledgement_rock_the_cart_with_nod_rock_p() -> None:
    """Either nod's ensemble rocks the cart 0.5 cm with nod_rock_p, drawn apart from the nod's
    own odds; at 0 the nods come alone."""
    rig = Rig(listener_nod_p=1.0, ack_nod_p=1.0, nod_rock_p=1.0, think_p=0.0)
    rig.open()
    rig.alive.speech_started(rig.clock.now)
    rig.tick(2.5)
    rig.alive.moment(LISTENER_PAUSE, "Ну я тебе так скажу")
    rig.tick(1.5)
    assert len(rig.sent(LISTEN_NOD)) == 1 and len(rig.sent(LISTEN_NOD, "body")) == 1
    (rock,) = rig.clips_of(LISTEN_NOD, "body")
    assert max(rock.tracks[0].values) == pytest.approx(0.005)
    rig.heard("Я сегодня весь день чинил колесо.", seconds=2.5)
    rig.tick(1.5)
    assert len(rig.sent(ACK_NOD)) == 1 and len(rig.sent(ACK_NOD, "body")) == 1
    alone = Rig(listener_nod_p=1.0, ack_nod_p=1.0, nod_rock_p=0.0, think_p=0.0)
    alone.open()
    alone.alive.speech_started(alone.clock.now)
    alone.tick(2.5)
    alone.alive.moment(LISTENER_PAUSE, "Ну я тебе так скажу")
    alone.tick(1.5)
    alone.heard("Я сегодня весь день чинил колесо.", seconds=2.5)
    alone.tick(1.0)
    assert alone.sent(LISTEN_NOD) and alone.sent(ACK_NOD) and not alone.clips("body")


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
    nod = head_clip(((0.0, 0.0, 0.0), (0.2, 0.0, 3.0), (0.4, 0.0, 0.0)), 0.2)
    rig.engine.request(Request("agreement", AtWord(1, 1, 0), Priority.MARK, "marker",
                               clips=(nod,)))  # fmt: skip
    rig.tick(0.02)
    (mark,) = rig.engine.pending("head", turn=1)
    second = rig.playout.now + (mark.stroke - 0.5 - rig.clock.now) - 0.15 + 0.05
    rig.news(1, 0, [(second, 3.0)])
    rig.tick(0.05)
    pulse = next(f for k, f in rig.events if k == "pulse")
    assert "head" not in (pulse["accent"] or {}).get("parts", [])
    assert "agreement's head stroke 0.5 s away (waiting)" in " ".join(pulse["skipped"])


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


@pytest.mark.parametrize("words", ["Пепин, привет!", "Ладно, пока, Пепин."], ids=("hi", "bye"))
def test_the_person_s_hello_or_bye_alone_moves_nothing(words: str) -> None:
    """Artem 2026-10-14: no scripted start. The person's «привет» or «пока», seen, asks the
    engine for nothing in the brain's wait: no greeting or farewell of its own, no
    acknowledgement, no thinking glance (their knobs at 1 here); the wave is the brain's own
    mark on its word. A statement as long brings both."""
    rig = Rig(seed=1, think_p=1.0, ack_nod_p=1.0)
    rig.open()
    rig.view.be(phase="track", eyes=True, seen_s=rig.clock.now + 100.0)  # seen all along
    rig.heard(words, seconds=2.0)
    rig.tick(4.0)
    asked = {f["intent"] for k, f in rig.records if k in ("performance", "cue")}
    assert asked.isdisjoint({"greeting", "farewell", ACK_NOD, "think_glance"}), asked
    assert not rig.moments(ACKNOWLEDGED)
    assert [m["why"] for m in rig.moments(THINK_WAIT)] == ["a hello or a bye"]
    said = Rig(seed=1, think_p=1.0, ack_nod_p=1.0)
    said.open()
    said.view.be(phase="track", eyes=True, seen_s=said.clock.now + 100.0)
    said.heard("Я сегодня весь день чинил колесо.", seconds=2.0)
    said.tick(4.0)
    assert said.moments(ACKNOWLEDGED)[0]["played"] and said.moments(THINK_WAIT)[0]["played"]


def test_the_laugh_is_one_body_its_face_head_gripper_and_cart_together() -> None:
    """{laugh} is its composition whole: every variant but the floor moves the face, the head,
    the arm (the gripper's click) and the wheels together, the dice between the variants; the
    listeners add no rock of their own to it."""
    everything = Scene(frozenset(PARTS))
    laugh = BOOK.compositions["laugh"]
    for k, variant in enumerate(laugh.variants):  # each for its own argument (happy, evil)
        arg = variant.needs.get("arg")
        clips = BookRenderer(BOOK).render("laugh", everything, 3, variant=k, arg=arg).clips
        parts = {c.part for c in clips}
        assert parts == ({"face", "head"} if variant.floor else {"face", "head", "arm", "body"})
        faces = {e.name for c in clips if c.part == "face" for e in c.events}
        assert faces == {"sinister"} if arg == "evil" else "sinister" not in faces
    for seed in range(4):
        rig = Rig(seed=seed)
        assert rig.engine.perform("laugh", source="tool").ok
        last = rig.engine.last()
        assert last is not None and {c.part for c in last.clips} == {"face", "head", "arm", "body"}
        rig.open()
        rig.say(1, 0, "Ха, смешно вышло.", Marker("gesture", "laugh", 0))
        rig.alive.part_started(1, 0, rig.playout.now)
        rig.tick(1.5)
        assert not rig.moments("laugh_rock")


def test_the_accents_move_the_cart_by_their_strength_and_the_weak_ones_sway_it_steadily() -> None:
    """Five sentences, one accent each, z 2.4, 2.6, 3.6, 3.2, 2.5 (strong_z 3): each nods; the
    weak ones move the cart on every second one (wheel_accent_share 0.5: steady, not dice): a
    0.5 cm sway or a 2-2.5 deg wag (the dice between the book's variants), the reply's
    strongest is the peak (its rock 1.2-1.5 cm or its wag 2.5-3 deg), the strong one after it
    rocks 1 cm or wags 2.5-3 deg; never only once a reply."""
    rig = Rig()
    rig.open()
    for k, z in enumerate((2.4, 2.6, 3.6, 3.2, 2.5)):
        rig.say(1, k, f"Фраза номер {k}.")
        rig.news(1, k, [(3.0 + 3.0 * k, z)], start=2.5 + 3.0 * k)
    rig.tick(0.05)
    rig.tick(17.0)
    kinds = [f["accent"]["kind"] for k, f in rig.events if k == "pulse"]
    assert kinds == ["weak", "weak", "peak", "strong", "weak"]
    assert len([c for c in rig.clips() if c.layer == "gesture"]) == 5  # every one nods
    moves = [(c.tracks[0].dof, max(abs(v) for v in c.tracks[0].values)) for c in rig.clips("body")]
    sizes = {"weak": {"x_m": (0.005, 0.005), "yaw_deg": (2.0, 2.5)},
             "peak": {"x_m": (0.012, 0.015), "yaw_deg": (2.5, 3.0)},
             "strong": {"x_m": (0.01, 0.01), "yaw_deg": (2.5, 3.0)}}  # fmt: skip
    assert len(moves) == 3
    for kind, (dof, size) in zip(("weak", "peak", "strong"), moves, strict=True):
        lo, hi = sizes[kind][dof]
        assert lo - 1e-6 <= size <= hi + 1e-6, (kind, dof, size)
        if dof == "yaw_deg":  # a wag nets to zero: the cart's heading as it began
            assert rig.clips("body")[moves.index((dof, size))].tracks[0].values[-1] == 0.0
    never = Rig(wheel_accent_share=0.0)
    never.open()
    for k in range(4):
        never.say(1, k, f"Фраза номер {k}.")
        never.news(1, k, [(3.0 + 3.0 * k, 2.5)], start=2.5 + 3.0 * k)
    never.tick(13.0)
    assert not never.clips("body"), "no sway at share 0, the weak ones only nod"


def test_the_cart_rocks_while_a_mark_moves_the_arm_and_nothing_moves_in_quiet() -> None:
    """One body: the accent at 1.6 s rocks the cart while the arm plays a mark (neither waits
    for the other). In quiet hours no accent moves anything, logged."""
    rig = Rig()
    rig.open()
    lift = Track("shoulder_lift", "additive", (0.0, 1.5, 3.0), (0.0, 20.0, 0.0))
    rig.engine.request(Request("mark", None, Priority.MARK, "marker",
                               clips=(Clip("arm", 0.0, 1.5, 3.0, (lift,)),)))  # fmt: skip
    rig.say(1, 0, "Это было огромное здание, правда, очень.")
    rig.news(1, 0, [(1.6, 3.6)], start=0.2)
    rig.tick(2.5)
    assert len(rig.sent(PEAK, "body")) == 1
    arm = rig.engine.scheduler.reservations.last("arm")
    assert arm is not None and arm.performance.intent == "mark"
    assert rig.clock.now < arm.ends, "the arm still moves under the rock"
    quiet = Rig(hour=23)
    quiet.open()
    quiet.say(1, 0, "Это было огромное здание.")
    quiet.news(1, 0, [(2.4, 3.6)], start=0.2)
    quiet.tick(3.0)
    assert not quiet.clips("body") and not quiet.sent(PEAK)
    assert next(f for k, f in quiet.events if k == "pulse")["skipped"] == ["quiet: the hours 23-08"]


def test_the_voice_laughing_rocks_the_cart_per_accent_as_the_wheels_can_follow() -> None:
    """A part whose accents come 0.35 s apart, seven of them (a laugh, read in its sound, no
    text): each asks for pulse_laugh and the laugh is the part's accent; the first rocks the
    cart with the happy face, those during its rock and its settling pass, a later one rocks it
    again. Two accents in a run, or accents 0.6 s apart, are no laugh."""
    rig = Rig()
    rig.open()
    rig.say(1, 0, "Ну и ну.")
    laugh = [1.0 + 0.35 * k for k in range(7)]
    rig.news(1, 0, [(s, 3.0 + 0.1 * k) for k, s in enumerate(laugh)], start=0.5)
    rig.tick(4.5)
    sounds = rig.moments("laugh_sound")
    assert [m["s"] for m in sounds] == pytest.approx(laugh)
    rocked = [m["s"] for m in sounds if "body" in m["parts"]]
    assert len(rocked) >= 2 and rocked[0] == pytest.approx(1.0)
    assert rocked[1] - rocked[0] >= 1.0, "the cart takes the next once its rock has settled"
    assert all("the cart still moves" in m["why"] for m in sounds if not m["played"])
    assert len(rig.sent(LAUGHING, "body")) == len(rocked) == len(rig.sent(LAUGHING, "face"))
    pulse = next(f for k, f in rig.events if k == "pulse")
    assert pulse["accent"] is None and pulse["skipped"] == ["its laugh is its accent"]
    for spacing, count in ((0.35, 2), (0.6, 4)):
        other = Rig()
        other.open()
        other.say(1, 0, "Ну и ну.")
        other.news(1, 0, [(1.0 + spacing * k, 3.0) for k in range(count)], start=0.5)
        other.tick(4.0)
        assert not other.moments("laugh_sound"), (spacing, count)


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


def test_an_accent_asks_for_the_arm_s_beat_inside_an_open_unit_and_none_without_one() -> None:
    """A weak accent with wrist_beat_p 1: while a unit holds the arm up its ensemble asks for the
    arm's beat beside the nod; with no unit, the nod alone (the arm never leaves rest for it)."""
    rig = Rig(wrist_beat_p=1.0)
    rig.open()
    rig.parts["arm"].held = {"wrist_flex": 70.0}  # type: ignore[attr-defined]
    raise_ = Track("shoulder_lift", "additive", (0.0, 0.3, 0.5), (0.0, 20.0, 20.0))
    unit = Clip("arm", 0.0, 0.3, 0.5, (raise_,), then="hold", hold_s=2.5)
    rig.engine.request(Request("unit", None, Priority.MARK, "marker", clips=(unit,)))
    rig.tick(0.6)
    rig.say(1, 0, "Вот так.")
    rig.news(1, 0, [(rig.playout.now + 1.0, 2.5)], start=rig.playout.now)
    rig.tick(2.0)
    pulse = next(f for k, f in rig.events if k == "pulse")
    assert pulse["accent"]["kind"] == "weak" and {"arm", "head"} <= set(pulse["accent"]["parts"])
    rig = Rig(wrist_beat_p=1.0)
    rig.open()
    rig.say(1, 0, "Вот так.")
    rig.news(1, 0, [(2.0, 2.5)], start=1.0)
    rig.tick(2.0)
    pulse = next(f for k, f in rig.events if k == "pulse")
    assert pulse["accent"]["parts"] == ["head"] and not rig.clips("arm")


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
    assert summary["counts"][PEAK] == 1 and summary["reason"] == "idle"
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


def test_the_knobs_load_from_config() -> None:
    data = json.loads((CONFIG / "expression.json").read_text())
    knobs_ = alive_knobs(data)
    assert knobs_.pulse.beat_lead_min_s == 0.30 and knobs_.pulse.melody_max_deg == 0.9
    # the glide: a key a second at least, the contour's phrase-level turns only (2026-10-13
    # evening: a key every 0.5 s read as a floating drift)
    assert knobs_.pulse.melody_key_gap_s == 1.0 and knobs_.pulse.melody_cutoff_hz == 0.5
    assert knobs_.pulse.arm_up_caps_share == 0.3  # the posture's raise, quiet
    assert knobs_.moments.head_gap_s == 1.0 and "breath_deg" not in data["moments"]
    assert knobs_.pulse.strong_z == 3.0 and knobs_.pulse.wheel_accent_share == 0.5
    assert knobs_.pulse.laugh_run == 3 and knobs_.pulse.laugh_gap_s == (0.2, 0.4)
    assert knobs_.moments.nod_rock_p == 0.3 and knobs_.pulse.arm_up_min_sentences == 2
    for gone in ("beat_amp_deg", "wrist_beat_deg", "accent_rock_p"):  # the book's sizes now
        assert gone not in data["pulse"]
    (breath,) = [i for i in BOOK.idle if "breath" in i.compositions]  # the breath is the book's
    assert breath.unless == ("speaking",) and breath.after_speech_s == 0.0
    assert (
        loops(BOOK, "breath") and BOOK.atoms["breath"].params["amp_deg"].hi == 2.0
    )  # 2026-10-13: 1 deg read as jerky ticks; the loop is the breath composition's step's
    assert knobs_.pulse.voice_stats == VoiceStats() and knobs_.quiet.hours == (23, 8)
    assert alive_knobs({}) == AliveKnobs()


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
    for piece in melody_pieces(spaced(keys, 0.5), [], -1.0, 0.5):
        clip = head_clip([(t - piece[0][0], 0.0, v) for t, v in piece], 0.0, layer="idle")
        assert refusal(clip) is None
