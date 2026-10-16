"""The sound registry (config/sounds.json: the synthesized default beat, the merge by name, the
checks), the beat grid on a synthetic drum machine and on the default beat, the default beat
made on demand, the player's contract on its fake, the music's pulse into the gesture engine
(each beat one step of the dance with the sound's style), the dance family within the caps, and
the vocabulary's lines for ``dance:STYLE`` and the reserved ``sound:NAME``."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pepin.arm_motion import load_arm_motion
from pepin.expression.animation import PARTS, Book, Clip, Scene, Variant
from pepin.expression.animator import EngineConfig, GestureEngine, engine_config
from pepin.expression.book import (
    ARG,
    face_names,
    load_book,
    mark_lines,
    span,
    vocabulary,
    with_rest,
)
from pepin.expression.fakes import (
    FakeArmLink,
    FakeAttention,
    FakeClock,
    FakeExecutor,
    FakePlayout,
)
from pepin.expression.listeners import (
    BEAT_PARTS,
    MUSIC,
    SOUND_ENDED,
    TOLD_LATE,
    WEAK_PARTS,
    MusicPulse,
)
from pepin.expression.markers import Mark, Marker, Part, read_mark
from pepin.expression.parts import ArmPart, Planned
from pepin.expression.render import BookRenderer
from pepin.expression.wheels import WheelsSettings, legs_of
from pepin.sounds import (
    DANCE,
    DEFAULT_BEAT,
    DEFAULT_BEAT_FILE,
    DEFAULT_BEAT_RATE,
    DEFAULT_STYLE,
    NO_PLAYER,
    SOUND,
    SOUND_LINE,
    BeatGrid,
    FakeSoundPlayer,
    MusicKnobs,
    Playing,
    Registry,
    Sound,
    beat_grid,
    click_track,
    dance_style,
    default_beat,
    default_beat_grid,
    frame_s,
    grid_from,
    make_synthesized,
    merge_sounds,
    read_audio,
    registry_from,
    sound_from,
    sound_path,
    write_beats,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config"
MOTION = load_arm_motion(CONFIG / "arm.json")
BOOK = with_rest(load_book(CONFIG / "animation.json"), MOTION.rest_deg)
WORDS = vocabulary(BOOK, face_names(CONFIG / "face.json"))
RATE = 24000


def public() -> dict[str, Any]:
    """The public registry's object, as written (no private tree merged)."""
    return json.loads((CONFIG / "sounds.json").read_text())  # type: ignore[no-any-return]


def drum(style: str = "rock", seconds: float = 4.0, lead_s: float = 1.0) -> Registry:
    """A registry of one sound with a file and a 120 BPM grid: the bar's first beat strong."""
    beats = int(seconds * 2)
    grid = BeatGrid(120.0, tuple(0.5 * k for k in range(beats)),
                    tuple((1.0, 0.4, 0.7, 0.4)[k % 4] for k in range(beats)))  # fmt: skip
    sound = Sound("drum", "drum.wav", 0.0, seconds, "a drum machine", style, grid)
    return Registry({"drum": sound}, MusicKnobs(strong=0.6, lead_s=lead_s, lead_min_s=0.25))


# -- the registry ---------------------------------------------------------------------------------


def test_the_public_registry_holds_the_synthesized_default_beat_alone() -> None:
    """One sound, made here (no audio in the tree): rock on its 16 beats, every 0.5 s from 0,
    each at pulse.strong or over, the grid the recipe's own."""
    registry = registry_from(public())
    assert list(registry.sounds) == [DEFAULT_BEAT]
    beat = registry.sounds[DEFAULT_BEAT]
    assert beat.file == DEFAULT_BEAT_FILE and beat.synthesized and not beat.external
    assert (beat.start_s, beat.seconds, beat.style) == (0.0, 8.0, DEFAULT_STYLE)
    assert beat.beats == default_beat_grid() and beat.beats.bpm == 120.0
    cut = beat.cut_beats()
    assert [t for t, _ in cut] == [0.5 * k for k in range(16)]
    assert min(s for _, s in cut) >= registry.pulse.strong and cut[0][1] == 1.0
    assert 1 <= len(beat.description.split()) <= 6 and beat.note
    assert set(DANCE_STYLES) == {beat.style}
    assert registry.pulse == MusicKnobs(0.6, 1.3, 0.25)
    assert registry.lines() == ("default_beat — a synthesized drum loop, 120 BPM (rock, 8 s)",)


DANCE_STYLES = BOOK.compositions[DANCE].args


def test_a_private_copy_merges_by_name_replacing_whole_and_appends_new_names() -> None:
    private = {"sounds": [
        {"name": "default_beat", "file": "loop.wav", "start_s": 1.0, "seconds": 6.0,
         "description": "a recorded loop instead", "style": "rock",
         "beats": {"bpm": 126.0, "times_s": [0.8, 1.3, 1.8], "strengths": [1, 0.5, 0.7]}},
        {"name": "fanfare", "file": "fanfare.wav", "start_s": 0.0, "seconds": 12.0,
         "description": "a brass fanfare", "style": "rock"},
        {"name": "chime", "start_s": 0.0, "seconds": 2.0, "description": "a door chime",
         "style": "voice"},
    ]}  # fmt: skip
    merged = registry_from(merge_sounds(public(), private))
    assert list(merged.sounds) == [DEFAULT_BEAT, "fanfare", "chime"]
    assert merged.sounds["fanfare"].file == "fanfare.wav"
    loop = merged.sounds[DEFAULT_BEAT]
    assert loop.file == "loop.wav" and not loop.synthesized and loop.beats is not None
    assert loop.beats.bpm == 126.0 and loop.cut_beats() == ((0.3, 0.5), (0.8, 0.7))
    assert merged.pulse == MusicKnobs(0.6, 1.3, 0.25)  # the public knobs stay
    assert merged.lines() == (
        loop.line(),
        merged.sounds["fanfare"].line(),
        merged.sounds["chime"].line(),
    )  # the ones with a file first


def test_the_registry_refuses_what_the_brain_s_list_and_the_player_cannot_take() -> None:
    with pytest.raises(ValueError, match="name"):
        sound_from({"name": "Drum Loop"})
    with pytest.raises(ValueError, match="6 words"):
        sound_from({"name": "x", "description": "one two three four five six seven"})
    with pytest.raises(ValueError, match="rising"):
        grid_from({"bpm": 100, "times_s": [1.0, 0.5], "strengths": [1, 1]})
    with pytest.raises(ValueError, match="strengths"):
        grid_from({"bpm": 100, "times_s": [1.0], "strengths": []})
    with pytest.raises(ValueError, match="twice"):
        registry_from({"sounds": [{"name": "a"}, {"name": "a"}]})
    with pytest.raises(ValueError, match="seconds"):
        sound_from({"name": "a", "seconds": 0})


def test_the_beat_tool_writes_a_grid_into_an_entry_and_appends_an_unknown_name(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sounds.json"
    entry = {"name": "a", "file": "a.wav", "start_s": 1.0, "seconds": 2.0, "description": "a",
             "style": "rock"}  # fmt: skip
    path.write_text(json.dumps({"sounds": [entry]}))
    grid = BeatGrid(120.0, (0.5, 1.0, 1.5, 2.0, 2.5), (1.0, 0.5, 0.5, 0.5, 1.0))
    write_beats(path, "a", grid)
    write_beats(path, "b", grid, "b.wav")
    text = path.read_text()
    assert '"times_s": [0.5, 1.0, 1.5, 2.0, 2.5]' in text  # one line a list
    registry = registry_from(json.loads(text))
    assert registry.sounds["a"].beats == grid and registry.sounds["a"].file == "a.wav"
    assert registry.sounds["a"].cut_beats() == ((0.0, 0.5), (0.5, 0.5), (1.0, 0.5), (1.5, 1.0))
    assert registry.sounds["b"].file == "b.wav" and registry.sounds["b"].beats == grid


def test_a_grid_new_to_the_private_copy_keeps_the_public_entry_s_cut_and_style(
    tmp_path: Path,
) -> None:
    """The private entry replaces the public one whole by name: written from the merged entry,
    it keeps the cut, the description and the style; from the defaults it would not."""
    private = tmp_path / "sounds.json"
    base = next(s for s in public()["sounds"] if s["name"] == DEFAULT_BEAT)
    grid = BeatGrid(126.0, (0.2, 0.7, 1.2), (1.0, 0.5, 0.7))
    write_beats(private, DEFAULT_BEAT, grid, "loop.wav", base)
    beat = registry_from(merge_sounds(public(), json.loads(private.read_text()))).sounds[
        DEFAULT_BEAT
    ]
    assert (beat.start_s, beat.seconds, beat.style) == (0.0, 8.0, "rock")
    assert beat.description == base["description"] and beat.file == "loop.wav"
    assert beat.beats == grid and beat.cut_beats() == ((0.2, 1.0), (0.7, 0.5), (1.2, 0.7))
    assert [s["name"] for s in json.loads(private.read_text())["sounds"]] == [DEFAULT_BEAT]


# -- the beats ------------------------------------------------------------------------------------


@pytest.mark.parametrize("bpm", [120.0, 92.0])
def test_the_beat_grid_finds_a_drum_machine_s_clicks_its_tempo_and_its_heavy_ones(
    bpm: float,
) -> None:
    seconds = 8.0
    grid = beat_grid(click_track(RATE, seconds, bpm), RATE)
    assert abs(grid.bpm - bpm) < 2.0, grid.bpm
    period = 60.0 / bpm
    clicks = [0.5 + k * period for k in range(int((seconds - 0.5 - 0.01) / period) + 1)]
    found = [t for t, s in zip(grid.times_s, grid.strengths, strict=True) if s > 0.2]
    assert len(found) == len(clicks), (found, clicks)
    for click, beat in zip(clicks, found, strict=True):
        assert abs(beat - click) <= 0.025, (click, beat)
    strong = [s for t, s in zip(grid.times_s, grid.strengths, strict=True)
              if s > 0.2 and any(abs(t - c) <= 0.025 for c in clicks[::4])]  # fmt: skip
    weak = [s for t, s in zip(grid.times_s, grid.strengths, strict=True)
            if s > 0.2 and all(abs(t - c) > 0.025 for c in clicks[::4])]  # fmt: skip
    assert max(strong) == 1.0 and sum(strong) / len(strong) > 1.3 * sum(weak) / len(weak)
    cut = grid.cut(2.0, 1.0)
    assert all(0.0 <= t < 1.0 for t, _ in cut) and len(cut) in (1, 2, 3)


def test_an_empty_or_silent_signal_gets_no_beats_worth_the_name() -> None:
    import numpy as np

    silent = beat_grid(np.zeros(RATE * 2), RATE)
    assert silent.times_s == () and silent.bpm == 120.0  # the prior, not the shortest lag's
    assert beat_grid(np.zeros(0), RATE).times_s == ()
    broken = np.zeros(RATE)
    broken[100] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        beat_grid(broken, RATE)


@pytest.mark.parametrize("intro", ["silent", "quiet"])
def test_the_grid_starts_on_the_first_real_beat_after_an_intro(intro: str) -> None:
    """Before 2026-10-14's review the chain began one period early in the silence, and ran
    through a quiet intro at strength 0; now the first beat is the first click."""
    import numpy as np

    x = click_track(RATE, 8.0, 120.0, offset_s=3.0)
    if intro == "quiet":
        rng = np.random.default_rng(1)
        x[: 3 * RATE] += 0.003 * rng.standard_normal(3 * RATE)
    grid = beat_grid(x, RATE)
    assert abs(grid.times_s[0] - 3.0) <= 0.025, grid.times_s[:3]
    assert min(grid.strengths) > 0.2


def test_the_grid_counts_the_real_frame_and_the_span_s_place_in_the_file() -> None:
    """At 22.05 kHz the 10 ms frame is 220 samples (9.977 ms): counted as 10 ms the beats of a
    minute drifted 0.12 s late by its end. A span cut from the file keeps the file's seconds."""
    rate = 22050
    assert frame_s(rate) == 220 / rate and frame_s(16000) == 0.01
    x = click_track(rate, 60.0, 100.0, offset_s=0.5)
    grid = beat_grid(x, rate)
    clicks = [0.5 + 0.6 * k for k in range(99)]
    late = [min(abs(t - c) for t in grid.times_s) for c in clicks if c > 50.0]
    assert max(late) <= 0.025, late
    span = beat_grid(x[10 * rate : 20 * rate], rate, offset_s=10.0)
    assert all(10.0 <= t < 20.0 for t in span.times_s)
    assert all(min(abs(t - c) for c in clicks) <= 0.025 for t in span.times_s)


# -- the default beat -----------------------------------------------------------------------------


def test_the_default_beat_s_grid_is_what_the_beat_analysis_finds_on_it(tmp_path: Path) -> None:
    """The loop as written (16-bit wav, 16 kHz mono, 8 s, peak 0.9) analysed like a recording:
    120 BPM, every beat of the registry's grid found within 25 ms (the first, on sample 0, has
    no rise before it: the analysis starts on the second)."""
    path = make_synthesized(DEFAULT_BEAT_FILE, tmp_path)
    assert path == tmp_path / "sounds" / DEFAULT_BEAT_FILE
    samples, rate = read_audio(path)
    assert rate == DEFAULT_BEAT_RATE and len(samples) == 8 * DEFAULT_BEAT_RATE
    assert 0.89 <= float(abs(samples).max()) <= 0.91
    assert samples == pytest.approx(default_beat(), abs=2.0 / 32767)  # the same seed, the same
    grid = beat_grid(samples, rate)
    assert abs(grid.bpm - 120.0) < 2.0, grid.bpm
    for t in default_beat_grid().times_s[1:]:
        assert min(abs(t - found) for found in grid.times_s) <= 0.025, t
    with pytest.raises(ValueError, match="no recipe"):
        make_synthesized("other.wav", tmp_path)


def test_a_synthesized_sound_s_file_is_made_on_first_use_in_the_public_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pepin.overlay

    monkeypatch.setattr(pepin.overlay, "public_tree", lambda: tmp_path)
    beat = registry_from(public()).sounds[DEFAULT_BEAT]
    assert sound_path(beat, make=False) is None
    made = sound_path(beat)
    assert made == tmp_path / "sounds" / DEFAULT_BEAT_FILE and made.is_file()
    stamp = made.stat().st_mtime_ns
    assert sound_path(beat) == made and made.stat().st_mtime_ns == stamp  # found, not remade
    assert sound_path(Sound("x", "x.wav")) is None  # a recording found nowhere stays absent


# -- the contract ---------------------------------------------------------------------------------


class Ears:
    """A beat listener that keeps every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def sound_started(self, playing: Playing) -> None:
        self.calls.append(("started", playing))

    def beat(self, t_beat: float, strength: float) -> None:
        self.calls.append(("beat", (round(t_beat, 3), strength)))

    def sound_ended(self, playing: Playing) -> None:
        self.calls.append(("ended", playing))


def test_the_fake_player_plays_the_contract_a_lead_ahead_and_ends_or_stops() -> None:
    clock = FakeClock(100.0)
    player = FakeSoundPlayer(drum(), clock, lag_s=0.1)
    ears = Ears()
    player.subscribe(ears)
    playing = player.play("drum")
    assert playing.t0 == 100.1 and playing.end == 104.1 and len(playing.beats) == 8
    assert ears.calls == [("started", playing)]
    player.step()  # the beats within lead_s of now: at t0 and t0 + 0.5
    assert [c for c in ears.calls if c[0] == "beat"] == [
        ("beat", (100.1, 1.0)), ("beat", (100.6, 0.4)),
    ]  # fmt: skip
    clock.now = 101.0
    player.step()
    assert [c[1][0] for c in ears.calls if c[0] == "beat"] == [100.1, 100.6, 101.1, 101.6]
    clock.now = 104.1
    player.step()
    assert len([c for c in ears.calls if c[0] == "beat"]) == 8 and ears.calls[-1][0] == "ended"
    second = player.play("drum")
    player.stop(second)
    assert ears.calls[-1] == ("ended", second) and player.playing is None
    with pytest.raises(RuntimeError, match="no file"):
        FakeSoundPlayer(registry_from({"sounds": [{"name": "chime"}]}), clock).play("chime")


# -- the music's pulse ----------------------------------------------------------------------------


class Rig:
    """The engine on fake parts, the music's pulse on it, a fake player, one fake clock."""

    def __init__(self, registry: Registry, *, on: bool = True, book: Book = BOOK) -> None:
        self.clock = FakeClock(1000.0)
        self.playout = FakePlayout()
        self.parts = {p: FakeExecutor(p) for p in PARTS}
        self.parts["arm"].latency_s, self.parts["head"].latency_s = 0.06, 0.1
        self.records: list[tuple[str, dict[str, Any]]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        config = EngineConfig(arm_speed_scale=1.0, body_speed_scale=1.0, audio_lag_s=0.15,
                              idle_after_speech_s=0.0)  # fmt: skip
        self.engine = GestureEngine(
            self.parts, self.playout, lambda: (book, config), on=on, clock=self.clock,
            record=lambda kind, fields: self.records.append((kind, fields)),
            seeds=random.Random(0), attention=FakeAttention(),
        )  # fmt: skip

        def journal(kind: str, now: float, /, **fields: Any) -> None:
            self.events.append((kind, fields))

        self.pulse = MusicPulse(self.engine, knobs=lambda: registry.pulse, clock=self.clock,
                                journal=journal).attach()  # fmt: skip
        self.player = FakeSoundPlayer(registry, self.clock, lag_s=0.1)
        self.player.subscribe(self.pulse)

    def tick(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds / 0.01))):
            self.clock.now += 0.01
            self.player.step()
            self.engine.step()

    def performances(self, intent: str) -> list[dict[str, Any]]:
        return [f for k, f in self.records if k == "performance" and f["intent"] == intent]

    def beats(self) -> list[dict[str, Any]]:
        return [f for k, f in self.events if k == "beat"]


def test_each_beat_is_one_step_of_the_dance_in_the_sound_s_style_on_the_parts_free() -> None:
    r = Rig(drum("rock", lead_s=1.3))
    r.player.play("drum")
    r.tick(5.0)
    beats = r.beats()
    assert len(beats) == 8 and all(b["style"] == "rock" for b in beats)
    assert beats[0]["parts"] == [] and set(beats[0]["skipped"].values()) == {TOLD_LATE}  # 0.1 s
    asked = [b for b in beats if b["parts"]]
    assert len(asked) >= 6
    for b in beats:
        if b["strength"] < 0.6:
            assert set(b["parts"]) <= WEAK_PARTS, b
    assert any("body" in b["parts"] for b in asked) and any("arm" in b["parts"] for b in asked)
    assert any(b["skipped"].get("arm") == "its last step" for b in beats)  # the raise's hold
    danced = r.performances(DANCE)
    assert len(danced) == len(asked)
    assert all(p["arg"] == "rock" and p["source"] == MUSIC for p in danced)
    assert [c for n, (c, *_) in r.parts["head"].calls if n == "play"]  # the head nodded
    assert r.pulse.counts["beats"] == 8 and r.pulse.counts["asked"] == len(asked)
    r.tick(0.5)
    assert r.pulse.playing is None
    ended = [f for k, f in r.events if k == "music" and f.get("ended")]
    assert ended and ended[0]["counts"]["asked"] == len(asked)


def test_a_stop_lets_go_of_the_steps_asked_ahead_and_nothing_dances_after_it() -> None:
    r = Rig(drum("rock", seconds=8.0, lead_s=1.3))
    playing = r.player.play("drum")
    r.tick(3.0)

    def plays() -> dict[str, int]:
        return {p: sum(1 for c in part.calls if c[0] == "play") for p, part in r.parts.items()}

    before = plays()
    r.player.stop(playing)
    r.tick(2.0)
    assert plays() == before  # the lead's worth of steps asked before the stop never went
    ended = [f for k, f in r.events if k == "music" and f.get("ended")]
    assert ended and ended[0]["let_go"] >= 1
    assert any(k == "dropped" and f["why"] == SOUND_ENDED for k, f in r.records)


def test_a_beat_told_nearer_than_lead_min_s_is_sat_out_by_every_part() -> None:
    r = Rig(drum("rock", lead_s=1.3))
    r.pulse.sound_started(Playing(r.player.registry.sounds["drum"], 1, r.clock.now))
    r.pulse.beat(r.clock.now + 0.2, 1.0)  # under lead_min_s 0.25
    r.pulse.beat(r.clock.now + 1.0, 1.0)
    told = r.beats()
    assert told[0]["parts"] == [] and set(told[0]["skipped"].values()) == {TOLD_LATE}
    assert told[1]["parts"] and r.pulse.counts[TOLD_LATE] == len(BEAT_PARTS)


def test_the_arm_at_rest_rises_for_rock_only_with_the_lead_that_covers_its_raise() -> None:
    """Rock's raise to raised starts 1.1 s before its beat: at a lead of 1.0 the arm at rest was
    late for every beat and never danced."""
    for lead_s, rises in ((1.0, False), (1.3, True)):
        r = Rig(drum("rock", seconds=6.0, lead_s=lead_s))
        r.player.play("drum")
        r.tick(6.5)
        assert any("arm" in b["parts"] for b in r.beats()) == rises, lead_s


def test_the_public_lead_covers_every_dance_step_s_preparation_and_its_part_s_latency() -> None:
    """The player tells each beat pulse.lead_s ahead: a step whose preparation and latency
    exceed it is late for every beat (the listener's rule), so every style's every part fits
    it at config/expression.json's latencies and speed scales."""
    data = json.loads((CONFIG / "expression.json").read_text())
    config = engine_config(data)
    latency = {p: float(v["latency_s"]) for p, v in data["parts"].items()
               if isinstance(v, dict) and "latency_s" in v}  # fmt: skip
    latency["body"] = float(data["wheels"]["latency_s"])
    scale = {"arm": config.arm_speed_scale, "body": config.body_speed_scale}
    lead = registry_from(public()).pulse.lead_s
    for style in DANCE_STYLES:
        for state in ("up", "rest"):

            def fits(variant: Variant, style: str = style, state: str = state) -> bool:
                return variant.needs.get(ARG) == style and variant.needs.get("arm", state) == state

            for part in PARTS:
                pre, _ = span(BOOK, DANCE, part, 1.0 / scale.get(part, 1.0), fits)
                assert pre + latency.get(part, 0.0) <= lead, (style, state, part, pre)


def test_a_style_the_family_lacks_dances_nothing_and_the_engine_off_dances_nothing() -> None:
    r = Rig(drum("laugh"))
    r.player.play("drum")
    r.tick(5.0)
    assert r.performances(DANCE) == [] and r.pulse.counts == {"beats": 8, "no dance": 8}
    assert r.beats() == []
    off = Rig(drum("rock"), on=False)
    off.player.play("drum")
    off.tick(5.0)
    assert off.performances(DANCE) == [] and off.pulse.counts == {"beats": 8, "off": 8}


def two_styles() -> Book:
    """The public book with a second dance style, ``sway``: rock's variants under its name (a
    private tree's styles reach the family the same way: its copy of the composition)."""
    dance = BOOK.compositions[DANCE]
    sway = tuple(replace(v, needs={**v.needs, ARG: "sway"}) for v in dance.variants)
    styled = replace(dance, args=(*dance.args, "sway"), variants=(*dance.variants, *sway))
    return replace(BOOK, compositions={**BOOK.compositions, DANCE: styled})


def test_a_bare_dance_mark_takes_the_style_of_the_sound_playing_else_the_book_s_first() -> None:
    book = two_styles()
    styles = book.compositions[DANCE].args
    r = Rig(drum("sway"), book=book)
    assert r.engine.arg_hooks[DANCE] == r.pulse.style and r.pulse.style() is None
    assert styles[0] == DEFAULT_STYLE == "rock" and DANCE_STYLES == ("rock",)
    assert dance_style(None, styles) is None
    playing = r.player.play("drum")
    assert r.pulse.style() == "sway" == dance_style(playing, styles)
    assert dance_style(playing, DANCE_STYLES) is None  # the public book has no sway
    assert dance_style(Playing(Sound("x", style="laugh"), 1, 0.0), styles) is None
    r.engine.part_queued(1, 0, Part("Танцуем вместе", None, (Marker("gesture", DANCE, 0),)))
    r.tick(0.05)
    marked = [p for p in r.performances(DANCE) if p["source"] == "marker"]
    assert marked and marked[0]["arg"] == "sway"
    r.player.stop(playing)
    r.engine.part_queued(2, 0, Part("Танцуем снова", None, (Marker("gesture", DANCE, 0),)))
    r.tick(0.05)
    marked = [p for p in r.performances(DANCE) if p["source"] == "marker"]
    assert len(marked) == 2 and marked[1]["arg"] is None  # the renderer's first: rock
    words = vocabulary(book, face_names(CONFIG / "face.json"))
    assert read_mark("dance:sway", words) == Mark("gesture", DANCE, "sway")
    assert read_mark("dance:rock", WORDS) == Mark("gesture", DANCE, "rock")
    assert read_mark("dance", WORDS) == Mark("gesture", DANCE)  # the book's bare: the parser too


def test_the_sound_mark_parses_and_the_engine_logs_no_player_yet() -> None:
    assert read_mark("sound:default_beat", WORDS) == Mark("gesture", SOUND, "default_beat")
    assert read_mark("sound:Drum Loop", WORDS) == Mark("gesture", SOUND, "drum loop")
    assert read_mark("sound", WORDS) is None  # a name it needs
    assert SOUND in WORDS.intents and SOUND not in BOOK.compositions
    r = Rig(drum())
    marker = Marker("gesture", SOUND, 0, "{sound:default_beat}", "default_beat")
    r.engine.part_queued(1, 0, Part("Вот так", None, (marker,)))
    r.tick(0.05)
    assert [f for k, f in r.records if k == "sound"] == [
        {"name": "default_beat", "turn": 1, "part": 0, "why": NO_PLAYER}
    ]
    assert not [k for k, f in r.records if k == "performance"]


# -- the dance family -----------------------------------------------------------------------------


EVERYTHING = Scene(frozenset(PARTS))
ARM_UP = Scene(frozenset(PARTS), arm_at="ready")
RENDERER = BookRenderer(BOOK)
ARM = ArmPart(FakeArmLink(), MOTION)
WHEELS = WheelsSettings.load(CONFIG / "expression.json")


def test_every_dance_variant_is_one_beat_s_step_on_its_style_within_the_arm_s_caps() -> None:
    """Each style has an up and a rest variant and a floor of the head alone; every arm clip
    plans within the caps from where the variant needs the arm; the stroke sits on the beat."""
    dance = BOOK.compositions[DANCE]
    assert dance.cooldown_s == 0.0 and dance.bind == "word" and dance.callable
    styles = {v.needs["arg"] for v in dance.variants}
    assert styles == set(dance.args) == set(DANCE_STYLES)
    for style in dance.args:
        written = [v for v in dance.variants if v.needs["arg"] == style]
        assert {v.needs.get("arm") for v in written if not v.floor} >= {"rest"}
        floors = [v for v in written if v.floor]
        assert len(floors) == 1 and {BOOK.atoms[s.atom].part for s in floors[0].steps} <= {
            "head",
            "face",
        }
    for index, variant in enumerate(dance.variants):
        up = variant.needs.get("arm") == "up"
        for seed in range(12):
            scene = ARM_UP if up else EVERYTHING
            performance = RENDERER.render(DANCE, scene, seed, variant=index,
                                          arg=variant.needs["arg"])  # fmt: skip
            assert performance.clips, (index, seed)
            for clip in performance.clips:
                assert abs(clip.stroke_s + clip.offset_s) <= 0.25, (index, clip)  # on the beat
                if clip.part == "arm":
                    # up: the voice's posture holds ready, the dance's own raise ends on raised
                    for base in [BOOK.poses["ready"], BOOK.poses["raised"]] if up else [None]:
                        planned = ARM.plan(clip, base=base)
                        assert isinstance(planned, Planned), (index, seed, planned)
                        assert planned.stretch <= 1.0 + 1e-6, (index, seed, planned.stretch)
                    assert clip.duration_s <= 4.6, (index, clip.duration_s)
                if clip.part == "body":
                    legs = legs_of(clip, WHEELS)
                    assert not isinstance(legs, str), (index, seed, legs)  # the wheels' caps


def test_the_vocabulary_lists_the_dance_s_styles_and_the_reserved_sound_last() -> None:
    lines = mark_lines(BOOK)
    assert "dance[:rock] — a dance step on the beat" in lines
    assert lines[-1] == SOUND_LINE == "sound:NAME — a registered sound by name"
    assert WORDS.args[SOUND] == ("*",) and WORDS.args[DANCE] == DANCE_STYLES


def _face_clips(performance_clips: tuple[Clip, ...]) -> list[str]:
    return [e.name for c in performance_clips if c.part == "face" for e in c.events]


def test_the_rock_step_wears_the_grin_and_the_faces_named_exist() -> None:
    faces = face_names(CONFIG / "face.json")
    named = {n for v in BOOK.compositions[DANCE].variants for s in v.steps
             for n in s.choices.get("name", ())}  # fmt: skip
    assert named == {"grin"} and named <= faces
    rest = next(i for i, v in enumerate(BOOK.compositions[DANCE].variants)
                if v.needs == {"arg": "rock", "arm": "rest"})  # fmt: skip
    performance = RENDERER.render(DANCE, EVERYTHING, 3, variant=rest, arg="rock")
    assert _face_clips(performance.clips) == ["grin"]
