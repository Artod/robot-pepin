"""The cascade's small parts: the config and its overrides, the sentence splitter, the timeline
and its report, the play-out clock, the ears, the tag line and the speech worker."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import json
import math
import threading
from pathlib import Path

import numpy as np
import pytest

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.voice_cascade.config import CascadeConfig, ProsodyConfig, apply_overrides
from pepin.voice_cascade.ears import Ears
from pepin.voice_cascade.fake import ToneVoice
from pepin.voice_cascade.player import Playback, Player, PlayoutClock
from pepin.voice_cascade.speech import SpeechJob, SpeechWorker
from pepin.voice_cascade.stages import Prosody
from pepin.voice_cascade.tagger import ProsodyTagger, comparatives
from pepin.voice_cascade.text import SentenceSplitter, speakable, syllables
from pepin.voice_cascade.timeline import Timeline, histogram, percentile, report

CONFIG = Path(__file__).resolve().parents[2] / "config" / "voice_cascade.json"
FRAME_S = FRAME_SAMPLES / RATE


# -- config ---------------------------------------------------------------------------------


def test_the_file_reads_with_the_haiku_preset() -> None:
    config = CascadeConfig.load(CONFIG)
    assert config.brain.model == "claude-haiku-5-5"
    assert (config.brain.thinking, config.brain.effort) == ("disabled", "low")
    assert config.wake == "name" and config.silence_s == 0.5
    assert config.voice_engine == "qwen3" and config.tts_voice == "serena"
    assert not config.barge_in and not config.backchannel and config.prosody_tags
    assert config.price.input == pytest.approx(0.10)


def test_overrides_switch_a_preset_and_a_flag_for_one_run() -> None:
    config = CascadeConfig.load(
        CONFIG, ["brain.preset=sonnet", "flags.barge_in=true", 'backchannel.lines=["мм"]']
    )
    assert config.brain.model == "claude-sonnet-5-5"
    assert config.brain.thinking == "between_tools"
    assert config.barge_in and config.backchannel_lines == ("мм",)
    opus = CascadeConfig.load(CONFIG, ["brain.preset=opus", "brain.effort=medium"])
    assert (opus.brain.thinking, opus.brain.effort) == ("adaptive", "medium")


def test_an_override_without_a_value_and_values_out_of_range_are_refused() -> None:
    with pytest.raises(ValueError, match=r"section\.key=value"):
        apply_overrides({}, ["flags.barge_in"])
    assert apply_overrides({}, ["ears.wake=always"]) == {"ears": {"wake": "always"}}
    for bad in (["ears.wake=sometimes"], ["brain.thinking=enabled"], ["brain.preset=gpt"]):
        with pytest.raises(ValueError):
            CascadeConfig.load(CONFIG, bad)
    with pytest.raises(ValueError, match="no price"):
        CascadeConfig.load(CONFIG, ["brain.model=claude-unknown"])


def test_the_file_parses_as_json_with_a_note_per_section() -> None:
    data = json.loads(CONFIG.read_text())
    for section, body in data.items():
        if isinstance(body, dict):
            assert "note" in body, section


# -- text -----------------------------------------------------------------------------------


def test_sentences_end_on_a_stop_and_a_space_not_inside_a_number() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("Сейчас гля") == []
    assert splitter.feed("ну. До принтера 3.") == ["Сейчас гляну."]
    assert splitter.feed("5 метра! Еду") == ["До принтера 3.5 метра!"]
    assert splitter.flush() == "Еду"
    assert splitter.feed("Раз\nДва") == ["Раз"]


def test_a_long_first_clause_is_cut_at_its_comma() -> None:
    splitter = SentenceSplitter(soft_chars=40)
    out = splitter.feed("Я сейчас стою у книжной полки в комнате, рядом с окном и столом")
    assert out == ["Я сейчас стою у книжной полки в комнате,"]
    assert splitter.feed(" и ещё немного текста без точки, очень длинного и долгого") == []


def test_speech_text_loses_its_markup() -> None:
    assert speakable("**Да**, `ок` # заголовок") == "Да , ок заголовок"
    assert speakable("Еду. [The drive has started.] (смеётся)") == "Еду."
    assert speakable("[note only]") == ""
    assert syllables("Привет, Pepin") == 4


# -- timeline -------------------------------------------------------------------------------


def test_a_timeline_counts_from_the_moment_the_person_stopped() -> None:
    line = Timeline(10.0)
    line.mark("closed", 10.5)
    line.mark("closed", 11.0)  # the first stamp wins
    line.mark("transcript", 10.76)
    ms = line.ms()
    assert ms["closed"] == 500 and ms["transcript"] == 760 and ms["speaker_end"] is None
    assert line.steps()["whisper"] == 260 and line.steps()["tts"] is None
    assert line.line().startswith("cut 500 | text 760 | token -")


def test_percentiles_and_the_report() -> None:
    assert percentile([1, 2, 3, 4], 0.5) == 2.5 and math.isnan(percentile([], 0.5))
    rows = [{"a": 100}, {"a": 300}, {"a": None}]
    assert histogram(rows, ["a"])[1].split() == ["a", "2", "200", "280", "300"]
    turn = {
        "ms": {"closed": 500, "transcript": 760},
        "steps": {"whisper": 260},
        "usage": {"input": 4, "cache_read": 4700, "cache_write": 100, "output": 20,
                  "calls": 1, "usd": 0.0001},
    }  # fmt: skip
    lines = report([turn, turn])
    assert any(line.startswith("transcript") and "760" in line for line in lines)
    assert lines[-1] == "total 0.020 US cents over 2 turns"


# -- the player -----------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Sink:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.flushes = 0
        self.ends = 0

    def play(self, pcm: bytes) -> None:
        self.played.append(pcm)

    def play_end(self) -> None:
        self.ends += 1

    def flush(self) -> None:
        self.flushes += 1


def pcm_s(seconds: float) -> bytes:
    return b"\x01\x00" * int(seconds * RATE)


def test_the_playout_clock_knows_what_has_played() -> None:
    clock, sink = Clock(), Sink()
    playout = PlayoutClock(sink, RATE, clock)
    playout.play(pcm_s(1.0))
    playout.play(pcm_s(1.0))  # queued behind the first
    assert playout.when(0.0) == 100.0 and playout.when(1.0) == 101.0
    assert playout.when(1.0, end=True) == 101.0 and playout.when(2.0, end=True) == 102.0
    clock.now = 100.5
    assert playout.played() == pytest.approx(0.5)
    playout.flush()
    assert playout.sent_s == pytest.approx(0.5) and sink.flushes == 1
    clock.now = 103.0
    playout.play(pcm_s(0.5))
    assert playout.when(0.5) == 103.0


def test_the_player_says_which_sentences_were_heard() -> None:
    clock = Clock()
    player = Player(Sink(), rate=RATE, lead_s=10.0, clock=clock)
    player.feed(pcm_s(1.0), "reply", 1, 0)
    player.feed(pcm_s(2.0), "reply", 1, 1)
    player.pacer.step()  # everything is sent at once (a long lead)
    assert player.span(1) == pytest.approx((0.0, 3.0))
    clock.now += 2.0  # the first sentence and half the second
    played = player.flush()
    assert played == pytest.approx(2.0)
    heard = player.heard(1, played, ["Раз.", "Два три четыре пять."])
    assert heard.sentences == 1 and heard.partial == "Два три"
    assert player.playout.sent_s == pytest.approx(2.0)
    player.forget(1)
    assert player.span(1) is None


def test_the_player_tells_a_backchannel_from_a_reply() -> None:
    player = Player(Sink(), rate=RATE, clock=Clock())
    assert player.playback() == "silent"
    player.feed(pcm_s(0.3), "backchannel", 0, -1)
    assert player.playback() == "backchannel" and player.span(0) is None
    player.feed(pcm_s(0.3), "reply", 1, 0)
    assert player.playback() == "reply"


# -- the ears -------------------------------------------------------------------------------


def tone(seconds: float, amplitude: float = 0.3) -> list[bytes]:
    t = np.arange(int(seconds * RATE)) / RATE
    pcm = (amplitude * 32767 * np.sin(2 * np.pi * 300 * t)).astype("<i2").tobytes()
    step = FRAME_SAMPLES * 2
    return [pcm[i : i + step] for i in range(0, len(pcm) - step + 1, step)]


def quiet(seconds: float) -> list[bytes]:
    return [b"\x00\x00" * FRAME_SAMPLES] * round(seconds / FRAME_S)


def hear(ears: Ears, frames: list[bytes], t0: float = 0.0) -> list[tuple[str, float]]:
    out = []
    for i, frame in enumerate(frames):
        out += [(e.kind, e.t) for e in ears.feed(frame, t0 + i * FRAME_S)]
    return out


def ears_with(mode: list[Playback], **changes: object) -> Ears:
    config = CascadeConfig().with_changes(**changes)
    return Ears(config, lambda: mode[0])


def test_an_utterance_starts_pauses_and_ends_after_the_silence() -> None:
    ears = ears_with(["silent"])
    events = hear(ears, quiet(0.5) + tone(1.0) + quiet(0.35) + tone(0.4) + quiet(0.8))
    kinds = [k for k, _ in events]
    assert kinds == ["start", "pause", "pause", "segment"]  # mid-utterance, then the end
    segment_t = events[-1][1]
    assert segment_t == pytest.approx(0.5 + 1.0 + 0.35 + 0.4 + 0.5, abs=0.05)


def test_the_robot_s_reply_is_not_heard_without_barge_in() -> None:
    mode: list[Playback] = ["reply"]
    ears = ears_with(mode)
    assert hear(ears, quiet(0.5) + tone(1.0) + quiet(0.8)) == []
    assert ears.echo_peak_db > -20


def test_a_voice_over_the_threshold_cuts_in_and_is_heard_from_its_start() -> None:
    mode: list[Playback] = ["reply"]
    ears = ears_with(mode, barge_in=True)
    echo = tone(0.6, amplitude=0.0003)  # 40+ dB under the voice
    events = hear(ears, quiet(1.0) + echo)
    assert events == []
    voice = tone(0.3)
    cut = hear(ears, voice, t0=2.0)
    assert [k for k, _ in cut] == ["barge_in", "start"]
    assert cut[0][1] == pytest.approx(2.0 + 0.24, abs=0.03)
    mode[0] = "silent"  # the player was flushed
    rest = hear(ears, tone(0.5) + quiet(0.7), t0=2.3)
    segments = [k for k, _ in rest if k == "segment"]
    assert segments == ["segment"]


def test_a_backchannel_s_echo_counts_as_silence() -> None:
    mode: list[Playback] = ["silent"]
    ears = ears_with(mode)
    events = hear(ears, quiet(0.5) + tone(1.0))
    mode[0] = "backchannel"
    events += hear(ears, tone(0.6, amplitude=0.001) + quiet(0.1), t0=1.5)
    assert [k for k, _ in events] == ["start", "pause", "segment"]


# -- the tag line ---------------------------------------------------------------------------


def fixed(loudness: float, active_s: float = 2.0, pitch: float = 7.0, span: float = 6.0):
    def measure(pcm: bytes) -> Prosody:
        return Prosody(active_s, loudness, pitch, span, 0.0)

    return measure


def test_the_tag_line_waits_for_a_baseline_then_describes_without_naming_a_feeling() -> None:
    config = ProsodyConfig(min_baseline=3)
    levels = iter([60.0, 62.0, 61.0, 61.0, 50.0])
    tagger = ProsodyTagger(config, measurer=lambda pcm: Prosody(2.0, next(levels), 7.0, 6.0))
    text = "как у тебя дела сегодня"  # 8 syllables over 2 s
    assert [tagger.line(tagger.measure(b""), text) for _ in range(3)] == ["", "", ""]
    assert tagger.line(tagger.measure(b""), text) == "[Артём: голос как обычно]"
    assert tagger.line(tagger.measure(b""), text) == "[Артём: тише обычного]"


def test_slower_and_flatter_and_a_model_label() -> None:
    tagger = ProsodyTagger(ProsodyConfig(min_baseline=2))
    for _ in range(3):
        tagger.line(Prosody(2.0, 60.0, 7.0, 8.0), "раз два три четыре пять шесть")
    line = tagger.line(
        Prosody(4.0, 60.0, 7.0, 2.0, label="happy 0.9"), "раз два три четыре пять шесть"
    )
    assert line == "[Артём: медленнее обычного, интонация ровнее, модель эмоций: happy 0.9]"
    assert comparatives(["тише", "медленнее", "ниже"]) == "тише, медленнее и ниже"


def test_praat_measures_a_voiced_tone() -> None:
    from pepin.voice_cascade.tagger import measure

    t = np.arange(RATE) / RATE
    x = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in (1, 2, 3))
    pcm = (0.2 * 32767 * x / 1.8).astype("<i2").tobytes()
    prosody = measure(b"\x00\x00" * 4000 + pcm + b"\x00\x00" * 4000)
    assert prosody.f0_median_st == pytest.approx(12 * math.log2(1.5), abs=0.3)
    assert prosody.active_s == pytest.approx(1.0, abs=0.1)
    assert measure(b"\x00\x00" * 100).loudness_db == -120.0


# -- the speech worker ----------------------------------------------------------------------


def test_sentences_are_spoken_in_order_and_a_cut_turn_is_dropped() -> None:
    clock = Clock()
    sink = Sink()
    player = Player(sink, rate=RATE, lead_s=100.0, clock=clock)
    voice = ToneVoice(rate=24_000, seconds_per_char=0.01)
    heard: list[tuple[int, int]] = []
    finished: dict[tuple[int, str], threading.Event] = {
        (1, "end"): threading.Event(),
        (3, "reply"): threading.Event(),
    }

    def on_done(job: SpeechJob, t: float) -> None:
        if (job.turn, job.kind) in finished:
            finished[(job.turn, job.kind)].set()

    worker = SpeechWorker(voice, player, clips=["мм"], on_done=on_done)
    worker.on_audio = lambda job, pcm, t: heard.append((job.turn, job.index))
    worker.start()
    assert worker.ready.wait(2.0) and worker.error is None
    assert worker.clips["мм"] and voice.spoken == ["мм"]
    worker.say(SpeechJob(1, 0, "Раз два."))
    worker.say(SpeechJob(1, 1, kind="end"))
    assert finished[(1, "end")].wait(2.0)
    worker.cut(2)  # turn 2 is cut before its jobs are made
    worker.say(SpeechJob(2, 0, "не будет сказано"))
    worker.say(SpeechJob(2, -1, "мм", "backchannel"))
    worker.say(SpeechJob(3, 0, "последнее"))
    assert finished[(3, "reply")].wait(2.0)
    worker.close()
    assert heard[0] == (1, 0) and (2, 0) not in heard and (2, -1) in heard and (3, 0) in heard
    assert voice.spoken == ["мм", "Раз два.", "последнее"]  # the backchannel came from memory
    assert player.span(1) is not None and player.span(2) is None
