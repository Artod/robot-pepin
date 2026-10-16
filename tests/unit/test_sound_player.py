"""The sound player (pepin.sound_player): a registry sound through the voice's own stream (the
real voice player and its pacer on a fake clock, a fake audio client collecting the PCM and when
each block left), its beats a lead ahead, speech winning over it (it waits for the reply, speech
fades it out), a barge-in's flush and a stop ending it, its row in the journal; the cut decoded
to 16 kHz mono."""

from __future__ import annotations

import json
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pepin.overlay
from pepin.expression.fakes import FakeClock
from pepin.sound_player import (
    BARGE_IN,
    END,
    NEWER,
    STOP,
    WAITED,
    PlayerKnobs,
    StreamSoundPlayer,
    load_cut,
    playable,
    player_knobs,
    resample,
)
from pepin.sounds import BeatGrid, MusicKnobs, Playing, Registry, Sound, registry_from
from pepin.voice_cascade.player import Player

RATE = 16000
LEAD_S = 0.3  # the pacer's lead in these tests
BLOCK_S = 0.04
LAG_S = 0.15  # the engine's audio_lag_s
FADE_S = 0.1
SPEECH = 1000  # every sample of the fake speech: told apart from the drum in the stream


class FakeAudioClient:
    """The board's speaker as the voice's pacer writes it: every block with the moment it was
    sent, the utterance ends and the flushes."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.blocks: list[tuple[float, bytes]] = []
        self.calls: list[tuple[float, str]] = []

    def play(self, pcm: bytes) -> None:
        self.blocks.append((self.clock.now, pcm))

    def play_end(self) -> None:
        self.calls.append((self.clock.now, "play_end"))

    def flush(self) -> None:
        self.calls.append((self.clock.now, "flush"))

    def seconds(self) -> float:
        return sum(len(pcm) for _, pcm in self.blocks) / 2 / RATE

    def samples(self) -> Any:
        return np.frombuffer(b"".join(pcm for _, pcm in self.blocks), dtype="<i2")


class Listener:
    """A beat listener that keeps every call, with the clock's time."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[tuple[str, float, Any]] = []

    def sound_started(self, playing: Playing) -> None:
        self.calls.append(("started", self.clock.now, playing))

    def beat(self, t_beat: float, strength: float) -> None:
        self.calls.append(("beat", self.clock.now, (t_beat, strength)))

    def sound_ended(self, playing: Playing) -> None:
        self.calls.append(("ended", self.clock.now, playing))

    def beats(self) -> list[tuple[float, float]]:
        """(when told, the beat's time)."""
        return [(at, args[0]) for kind, at, args in self.calls if kind == "beat"]


def drum_wav(path: Path, seconds: float = 3.0, rate: int = 44100, channels: int = 2) -> Path:
    """A click every 0.5 s from 0 (a 10 ms burst at 0.6 of full scale), as a 16-bit wav."""
    n = round(seconds * rate)
    x = np.zeros(n)
    burst = round(0.01 * rate)
    for k in range(round(seconds * 2)):
        start = round(k * 0.5 * rate)
        x[start : start + burst] = 0.6
    pcm = np.round(np.repeat(x[:, None], channels, axis=1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return path


def registry(seconds: float = 2.0, start_s: float = 0.0) -> Registry:
    grid = BeatGrid(120.0, tuple(0.5 * k for k in range(6)), (1.0, 0.4, 0.7, 0.4, 1.0, 0.4))
    drum = Sound("drum", "drum.wav", start_s, seconds, "a drum machine", "rock", grid)
    gone = Sound("gone", "gone.wav", 0.0, 2.0, "a recording not here", "folk")
    return Registry({"drum": drum, "gone": gone}, MusicKnobs(lead_s=1.0))


class Rig:
    """The voice's player on a fake audio client, the sound player through it, one clock."""

    def __init__(self, tree: Path, *, seconds: float = 2.0, wait_s: float = 5.0) -> None:
        drum_wav(tree / "sounds" / "drum.wav")
        self.clock = FakeClock(100.0)
        self.client = FakeAudioClient(self.clock)
        self.levels: list[float] = []
        self.player = Player(self.client, rate=RATE, lead_s=LEAD_S, block_s=BLOCK_S,
                             on_level=self.levels.append, clock=self.clock)  # fmt: skip
        self.busy = False
        self.rows: list[tuple[str, float, dict[str, Any]]] = []
        self.said: list[str] = []
        reg = registry(seconds)
        self.sounds = StreamSoundPlayer(
            self.player,
            voice_busy=lambda: self.busy,
            registry=lambda: reg,
            knobs=lambda: PlayerKnobs(fade_s=FADE_S, wait_s=wait_s),
            lag_s=LAG_S,
            clock=self.clock,
            journal=lambda kind, now, **fields: self.rows.append((kind, now, fields)),
            say=self.said.append,
        )
        self.listener = Listener(self.clock)
        self.sounds.subscribe(self.listener)

    def tick(self, seconds: float) -> None:
        for _ in range(round(seconds / 0.01)):
            self.clock.now = round(self.clock.now + 0.01, 6)
            self.player.pacer.step()
            self.sounds.step()

    def started(self) -> Playing:
        (playing,) = [args for kind, _, args in self.listener.calls if kind == "started"]
        assert isinstance(playing, Playing)
        return playing

    def row(self) -> tuple[float, dict[str, Any]]:
        (row,) = [(now, fields) for kind, now, fields in self.rows if kind == "sound"]
        return row


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A public tree of its own: its sounds/ is where the registry's files are found."""
    monkeypatch.setattr(pepin.overlay, "public_tree", lambda: tmp_path)
    monkeypatch.setenv(pepin.overlay.ENV, ".")
    return tmp_path


# -- the cut ------------------------------------------------------------------------------------


def test_the_cut_is_16_khz_mono_from_its_start_with_its_ends_ramped(tmp_path: Path) -> None:
    """A 44.1 kHz stereo file's cut from 0.5 s for 1 s: 16 000 samples, the clicks where they
    were (at 0 and 0.5 s of the cut), the first sample ramped from silence."""
    path = drum_wav(tmp_path / "drum.wav")
    pcm = load_cut(path, 0.5, 1.0, RATE)
    x = np.frombuffer(pcm, dtype="<i2").astype(float) / 32767
    assert len(x) == RATE and abs(x[0]) < 0.01
    loud = np.flatnonzero(np.abs(x) > 0.3)
    assert loud.min() < 0.01 * RATE and abs(loud[loud > 0.25 * RATE].min() - 0.5 * RATE) < 40
    assert len(load_cut(path, 2.5, 5.0, RATE)) == 2 * round(0.5 * RATE), "the file ends it sooner"
    assert load_cut(path, 9.0, 1.0, RATE) == b""
    tone = np.sin(2 * np.pi * 1000 * np.arange(44100) / 44100)
    down = resample(tone, 44100, RATE)
    assert len(down) == RATE and np.argmax(np.abs(np.fft.rfft(down))) == 1000  # 1 Hz bins
    with pytest.raises(ValueError, match="16-bit"):
        bad = tmp_path / "bad.wav"
        with wave.open(str(bad), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(1)
            handle.setframerate(8000)
            handle.writeframes(b"\x80" * 800)
        load_cut(bad, 0.0, 1.0, RATE)


def test_the_knobs_read_the_player_block_and_the_public_registry_ships_them() -> None:
    assert player_knobs({}) == PlayerKnobs() == PlayerKnobs(0.2, 30.0)
    assert player_knobs({"fade_s": 0.3, "note": "x"}) == PlayerKnobs(0.3, 30.0)
    data = json.loads((Path(__file__).resolve().parents[2] / "config/sounds.json").read_text())
    assert player_knobs(data["player"]) == PlayerKnobs()


# -- the stream ---------------------------------------------------------------------------------


def test_a_sound_streams_in_the_voice_s_blocks_in_real_time_and_tells_its_beats_ahead(
    tree: Path,
) -> None:
    """The cut goes into the voice's pacer whole and leaves in 40 ms blocks never more than the
    lead ahead of the clock; t0 is its first block's play-out plus the lag; each beat is told
    pulse.lead_s (1.0) before it sounds, the first ones at once; the end at t0 + 2 s, the
    board told the utterance is over, no mouth moved, one row and two lines."""
    r = Rig(tree)
    playing = r.sounds.play("drum")
    assert not playing.waiting and playing.sound.seconds == 2.0 and len(playing.beats) == 4
    assert r.sounds.current() == playing
    r.tick(0.02)
    started = r.started()
    first_sent = r.client.blocks[0][0]
    assert started.t0 == pytest.approx(first_sent + LAG_S) and started.id == playing.id
    assert r.player.playback() == "sound"
    r.tick(2.5)
    assert r.client.seconds() == pytest.approx(2.0, abs=1e-3)
    assert {len(pcm) for _, pcm in r.client.blocks} == {2 * round(BLOCK_S * RATE)}
    for k, (at, _) in enumerate(r.client.blocks):
        assert first_sent + k * BLOCK_S - at <= LEAD_S + BLOCK_S + 1e-6, "never past the lead"
    told = r.listener.beats()
    assert [t for _, t in told] == pytest.approx([started.t0 + 0.5 * k for k in range(4)])
    for at, t in told:
        assert t - at == pytest.approx(1.0, abs=0.011) or at <= started.t0 - LAG_S + 0.011
    assert [kind for kind, _, _ in r.listener.calls][-1] == "ended"
    assert r.listener.calls[-1][1] == pytest.approx(started.end, abs=0.011)
    assert ("play_end" in [c for _, c in r.client.calls]) and r.levels == []
    now, fields = r.row()
    assert now == started.t0 and fields["stopped_by"] == END and fields["name"] == "drum"
    assert fields["played_s"] == pytest.approx(2.0) and fields["waited_s"] <= 0.011  # a tick
    assert len(r.said) == 2 and "drum: end after 2.0 s" in r.said[-1]
    assert r.sounds.current() is None and r.player.playback() == "silent"


def test_a_shorter_play_and_a_mark_s_style_cut_the_entry_and_dance_its_style(tree: Path) -> None:
    r = Rig(tree)
    playing = r.sounds.play("drum", seconds=1.2, style="folk")
    assert playing.sound.seconds == 1.2 and playing.sound.style == "folk"
    assert len(playing.beats) == 3  # 0, 0.5, 1.0
    r.tick(2.0)
    assert r.client.seconds() == pytest.approx(1.2, abs=1e-3)
    assert r.row()[1]["style"] == "folk" and r.sounds.play("drum", seconds=9.0).sound.seconds == 2.0


def test_a_name_not_in_the_registry_and_a_file_that_is_nowhere_are_refused(tree: Path) -> None:
    r = Rig(tree)
    with pytest.raises(KeyError):
        r.sounds.play("nothing")
    with pytest.raises(RuntimeError, match="no file"):
        r.sounds.play("gone")
    assert r.sounds.current() is None and r.client.blocks == []
    assert playable(r.sounds.load_registry().sounds["drum"])
    assert not playable(r.sounds.load_registry().sounds["gone"])
    assert playable(registry_from({"sounds": [{"name": "x", "file": "x.wav",
                                               "synthesized": True}]}).sounds["x"])  # fmt: skip


# -- speech wins --------------------------------------------------------------------------------


def test_a_sound_asked_during_a_reply_waits_for_it_and_starts_after(tree: Path) -> None:
    """ "Включаю! {sound:drum}": the reply is in flight (the brain, its speech), so the sound
    waits, nothing of it in the stream; the reply over, it plays, t0 after the wait."""
    r = Rig(tree)
    r.busy = True
    playing = r.sounds.play("drum")
    assert playing.waiting and playing.t0 == 100.0
    r.tick(1.0)
    assert r.client.blocks == [] and r.listener.calls == []
    r.busy = False
    r.tick(0.03)
    started = r.started()
    assert not started.waiting and started.t0 > 101.0 and started.id == playing.id
    assert "after the voice (1.0 s)" in r.said[0]
    r.tick(2.5)
    assert r.row()[1]["waited_s"] == pytest.approx(1.0, abs=0.011)


def test_a_barge_in_drops_a_sound_still_waiting_for_the_reply_it_cut(tree: Path) -> None:
    """ "Включаю гимн..." and the person cuts in: the flush drops the sound the cut reply asked
    for, never fed, so it does not play after the next reply."""
    r = Rig(tree)
    r.player.on_flush = r.sounds.flushed
    r.busy = True
    r.sounds.play("drum")
    r.tick(0.3)
    r.player.flush()
    r.busy = False
    r.tick(0.5)
    assert r.client.blocks == [] and r.listener.calls == [] and r.sounds.current() is None
    assert r.row()[1]["stopped_by"] == BARGE_IN


def test_a_sound_waits_at_most_wait_s_for_the_voice(tree: Path) -> None:
    r = Rig(tree, wait_s=1.0)
    r.busy = True
    r.sounds.play("drum")
    r.tick(1.5)
    assert r.client.blocks == [] and r.listener.calls == [] and r.sounds.current() is None
    now, fields = r.row()
    assert now == 100.0 and fields["stopped_by"] == WAITED and fields["played_s"] == 0.0


def test_speech_takes_the_speaker_the_sound_fades_out_and_the_reply_follows_it(
    tree: Path,
) -> None:
    """A reply's first chunk while the sound plays: what is sent plays out, the next FADE_S of
    the sound ramps to silence, the rest is dropped (stopped by speech), and the reply's audio
    comes right after it in the stream, its mark where the board plays it."""
    r = Rig(tree)
    r.sounds.play("drum")
    r.tick(0.6)
    sent = r.client.seconds()
    queued = r.player.queued_audio()
    assert queued.fade_s == FADE_S and queued.sent_s == pytest.approx(sent - 0.6 + 0.01, abs=0.05)
    speech = np.full(round(0.4 * RATE), SPEECH, dtype="<i2").tobytes()
    r.player.feed(speech, "reply", 1, 0)
    r.player.end()
    assert r.player.playback() == "reply"
    r.tick(0.02)
    assert [k for k, _, _ in r.listener.calls].count("ended") == 1
    assert r.row()[1]["stopped_by"] == "speech" and r.sounds.current() is None
    r.tick(1.5)
    x = r.client.samples()
    first_speech = int(np.flatnonzero(x == SPEECH)[0])
    assert first_speech / RATE == pytest.approx(sent + FADE_S, abs=0.002)
    assert np.all(x[first_speech:] == SPEECH) and len(x) - first_speech == len(speech) // 2
    mark = r.player.mark(1, 0)
    assert mark is not None and mark[0] == pytest.approx(sent + FADE_S, abs=0.002)
    assert r.levels, "the reply's blocks move the mouth"


def test_a_barge_in_s_flush_ends_the_sound_and_a_stop_fades_it(tree: Path) -> None:
    r = Rig(tree)
    r.sounds.play("drum")
    r.tick(0.5)
    r.player.flush()  # the barge-in, on the mic's thread
    r.tick(0.02)
    assert r.row()[1]["stopped_by"] == BARGE_IN and r.listener.calls[-1][0] == "ended"
    r.rows.clear()
    playing = r.sounds.play("drum")
    r.tick(0.5)
    before = r.client.seconds()
    r.sounds.stop(playing)
    r.tick(1.0)
    assert r.row()[1]["stopped_by"] == STOP
    assert r.client.seconds() == pytest.approx(before + FADE_S, abs=0.002), "the fade, no more"
    r.sounds.stop(playing)  # over: nothing
    assert len(r.rows) == 1


def test_a_newer_sound_replaces_the_one_playing(tree: Path) -> None:
    r = Rig(tree)
    first = r.sounds.play("drum")
    r.tick(0.5)
    r.busy = True  # the first one's tail: the voice's player still speaks
    second = r.sounds.play("drum", seconds=1.0)
    r.tick(0.02)
    rows = [f for k, _, f in r.rows if k == "sound"]
    assert rows[0]["stopped_by"] == NEWER and r.sounds.current() == second != first
    r.busy = False
    r.tick(2.0)  # the first one's sent lead and fade (0.4 s) play out before it
    assert [f["stopped_by"] for k, _, f in r.rows] == [NEWER, END]
    assert [k for k, _, _ in r.listener.calls].count("started") == 2


def test_the_thread_steps_on_its_own_and_close_stops_the_sound(tree: Path) -> None:
    r = Rig(tree)
    r.sounds.start()
    r.sounds.play("drum")
    for _ in range(200):
        if r.listener.calls:
            break
        r.player.pacer.step()
        time.sleep(0.005)
    assert r.listener.calls and r.listener.calls[0][0] == "started"
    r.sounds.close()
    assert r.row()[1]["stopped_by"] == STOP and r.sounds.current() is None
