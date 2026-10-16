"""The sound player: a registry sound (:mod:`pepin.sounds`) through the voice's own stream to the
board's speaker, its beats told to the body.

THE PATH. The board plays one stream (PCM s16le mono at the hello's ``play_rate``, 16 kHz,
:mod:`pepin.audio_link`), and the voice's player (:class:`pepin.voice_cascade.player.Player`) is
its one writer: the cut goes into that player as a ``sound`` (:meth:`SoundChannel.play_sound`),
whose pacer sends it in ``block_s`` blocks (40 ms) ``lead_s`` ahead in real time as it sends
speech, so the board's echo canceller, the barge-in detector's echo model and the ears' gate all
take it for the robot's own sound. The cut is the entry's ``start_s`` and ``seconds`` of its
file (16-bit PCM wav, the standard library's ``wave``), its channels averaged, resampled to the
player's rate when the file's differs (an FFT resample: the files of sounds/ are 16 kHz mono
already), 5 ms ramps at its two ends so a cut mid-waveform does not click.

SPEECH WINS. A sound never starts over the voice: asked while a reply is in flight (the brain
still thinking, its tools running, its speech queued or playing: ``voice_busy``), it waits for
that reply's end, at most ``player.wait_s``, so "Включаю! {sound:x}" says the words and then
plays (a dance mark mid-sentence: the beat after the sentence). Speech that comes while it plays
takes the speaker as a reply takes over a filler: its part not sent yet ramps to silence over
``player.fade_s`` and the rest is dropped (stopped by ``speech``). The board's one stream cannot
mix: a tune ducked under the voice would need a mix on this Mac before the pacer, which would
move the speech's own stream clock (the marks the gestures and a barge-in's heard words are
read on), so it is not done. A barge-in's flush drops it (``barge-in``), a newer sound replaces
it (``a newer sound``), ``stop`` ends it with the fade.

THE CLOCK. ``t0``, when the cut's first sample is heard, on the engine's clock (laptop
monotonic): the player's own play-out clock for its first block (:meth:`SoundChannel.started_at`,
read once that block is sent, a few milliseconds after the feed) plus ``lag_s``, the engine's
``timing.audio_lag_s`` (the board's aplay start and the WiFi hop: the lag the word clock puts on
speech). The beats are the entry's grid within the cut (:meth:`pepin.sounds.Sound.cut_beats`)
from ``t0``, each told ``pulse.lead_s`` before it sounds; the end at ``t0`` plus the cut's
seconds. Every listener call (:class:`pepin.sounds.BeatListener`) comes from :meth:`step`, on
this player's own thread (:meth:`start`) or a test's clock.

THE LOG. One ``sound`` row a play in the conversation's events.jsonl, written at its end and
stamped at its start (``t0``; the moment it was asked when it never started): the name, the
style, the seconds asked and played, how long it waited for the voice, its beats, and what
stopped it (``end``, ``stop``, ``speech``, ``barge-in``, ``a newer sound``, ``the voice kept
the speaker``, ``unreadable: ...``). One line in the cascade's output at its start and end.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
import wave
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from pepin.expression.conversation import EventSink
from pepin.sounds import (
    REGISTRY_FILE,
    BeatListener,
    Playing,
    Registry,
    Sound,
    load_registry,
    sound_path,
)

logger = logging.getLogger(__name__)

SAMPLE_BYTES = 2  # s16le
EDGE_S = 0.005  # the cut's ramps in and out: no click where it cuts a waveform
TICK_S = 0.01  # the player's thread steps this often
FIRST_BLOCK_S = 0.5  # the first block not sent by then: t0 is the feed's moment (a stalled pacer)
# What stopped a play (the voice's player tells "speech" and "barge-in" itself).
END = "end"
STOP = "stop"  # also the tool's word for stopping
NEWER = "a newer sound"
WAITED = "the voice kept the speaker"


@dataclass(frozen=True)
class PlayerKnobs:
    """config/sounds.json's ``player``: the fade speech or a stop cuts a sound with, and how long
    a sound waits for the voice to let go of the speaker."""

    fade_s: float = 0.2
    wait_s: float = 30.0


def player_knobs(data: Mapping[str, Any]) -> PlayerKnobs:
    """The ``player`` block as knobs (defaults for what it lacks; a note is skipped)."""
    d = PlayerKnobs()
    return PlayerKnobs(float(data.get("fade_s", d.fade_s)), float(data.get("wait_s", d.wait_s)))


def load_player_knobs() -> PlayerKnobs:
    """The player's knobs of the registry in force (merged over the private trees)."""
    from pepin.deployment import config_file
    from pepin.overlay import read_json

    block = read_json(config_file(REGISTRY_FILE)).get("player", {})
    return player_knobs(block if isinstance(block, Mapping) else {})


class SoundChannel(Protocol):
    """The voice's player as a sound goes through it
    (:class:`pepin.voice_cascade.player.Player`)."""

    rate: int

    def play_sound(self, pcm: bytes, fade_s: float, ended: Callable[[str], None]) -> float:
        """Queue the sound's PCM at :attr:`rate`; the stream second its first sample is at;
        ``ended(why)`` hears a cut (speech, a barge-in, a stop)."""
        ...

    def stop_sound(self, why: str = STOP) -> bool:
        """The sound's part not sent yet faded out and dropped; ``ended(why)`` told."""
        ...

    def started_at(self, offset: float) -> float | None:
        """When stream second ``offset`` starts playing; None until its block is sent."""
        ...


def resample(samples: Any, src: int, dst: int) -> Any:
    """``samples`` (float, mono) from ``src`` Hz to ``dst`` Hz by FFT (band-limited: the
    spectrum cut or padded at the lower Nyquist; a whole cut at once, seconds of audio)."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float64)
    n = len(x)
    m = round(n * dst / src)
    if src == dst or n == 0 or m == 0:
        return x if src == dst else np.zeros(m)
    spectrum = np.fft.rfft(x)
    kept = np.zeros(m // 2 + 1, dtype=complex)
    k = min(len(kept), len(spectrum))
    kept[:k] = spectrum[:k]
    out: Any = np.fft.irfft(kept, m) * (m / n)
    return out


def load_cut(path: str | Path, start_s: float, seconds: float, rate: int) -> bytes:
    """The cut ``[start_s, start_s + seconds)`` of a 16-bit PCM wav as s16le mono at ``rate``
    (the file's end ends it sooner), its two ends ramped over :data:`EDGE_S`; ``ValueError``
    for a file that is no 16-bit PCM wav."""
    import numpy as np

    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getsampwidth() != SAMPLE_BYTES:
                raise ValueError(f"{path}: a 16-bit PCM wav, not {8 * handle.getsampwidth()}-bit")
            file_rate, channels, frames = (handle.getframerate(), handle.getnchannels(),
                                           handle.getnframes())  # fmt: skip
            first = min(frames, max(0, round(start_s * file_rate)))
            count = min(frames - first, max(0, round(seconds * file_rate)))
            handle.setpos(first)
            raw = handle.readframes(count)
    except (wave.Error, EOFError) as error:
        raise ValueError(f"{path}: {error}") from error
    x = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(np.float64).mean(axis=1)
    x = resample(x, file_rate, rate)
    edge = min(len(x) // 2, round(EDGE_S * rate))
    if edge > 0:
        ramp = np.linspace(0.0, 1.0, edge, endpoint=False)
        x[:edge] *= ramp
        x[len(x) - edge :] *= ramp[::-1]
    pcm: bytes = np.clip(np.round(x), -32768, 32767).astype("<i2").tobytes()
    return pcm


@dataclass
class _Play:
    """One play on its way: asked, fed, playing, then ended."""

    playing: Playing  # what play() answered (t0 the asked moment while it waits)
    asked: float
    knobs: PlayerKnobs
    lead_s: float  # the registry's pulse.lead_s: each beat told this long before it sounds
    path: Path | None = None
    offset: float | None = None  # the stream second it was fed at
    fed_at: float | None = None
    started: Playing | None = None  # with the real t0, once heard
    told: int = 0  # beats told
    cut: str | None = None  # why it was cut (the channel's word or ours), until it ends


class StreamSoundPlayer:
    """:class:`pepin.sounds.SoundPlayer` through the voice's stream (the module's notes)."""

    def __init__(
        self,
        channel: SoundChannel,
        *,
        voice_busy: Callable[[], bool],
        registry: Callable[[], Registry] = load_registry,
        knobs: Callable[[], PlayerKnobs] = load_player_knobs,
        lag_s: float = 0.15,
        clock: Callable[[], float] = time.monotonic,
        journal: EventSink | None = None,
        say: Callable[[str], None] | None = None,
    ) -> None:
        """``voice_busy()``: a reply is in flight or the voice's audio is queued (the loop's
        word); ``registry`` and ``knobs`` are read again at every play; ``lag_s``: the engine's
        ``timing.audio_lag_s``; ``journal``: the conversation's events.jsonl; ``say``: one line
        of the cascade's output."""
        self.channel = channel
        self.voice_busy = voice_busy
        self.load_registry = registry
        self.load_knobs = knobs
        self.lag_s = lag_s
        self.clock = clock
        self.journal = journal
        self.say = say
        self.listeners: list[BeatListener] = []
        self._lock = threading.Lock()
        self._play: _Play | None = None
        self._replaced: list[_Play] = []  # cut by a newer play, ended at the next step
        self._ids = itertools.count(1)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- the contract (any thread) --------------------------------------------------------------

    def play(self, name: str, seconds: float | None = None, style: str | None = None) -> Playing:
        """Entry ``name``'s cut, ``seconds`` of it at most (None: all), danced in ``style`` (None:
        the entry's): it plays from now, or once the voice lets go of the speaker (``waiting``;
        ``t0`` is then the moment asked: the listeners' ``sound_started`` carries the real one).
        A sound asked or playing is replaced. ``KeyError`` for a name not in the registry,
        ``RuntimeError`` for an entry whose file is nowhere."""
        registry = self.load_registry()
        sound = registry.sounds[name]
        path = sound_path(sound, make=False)
        if path is None and not sound.synthesized:
            raise RuntimeError(f"sound {name!r} has no file here (sounds/{sound.file})")
        cut = sound.seconds if seconds is None else max(0.0, min(seconds, sound.seconds))
        sound = replace(sound, seconds=cut, style=style or sound.style)
        now = self.clock()
        waiting = self.voice_busy()
        playing = Playing(sound, next(self._ids), now, sound.cut_beats(), waiting=waiting)
        play = _Play(playing, now, self.load_knobs(), registry.pulse.lead_s, path)
        with self._lock:
            older, self._play = self._play, play
            if older is not None:
                self._replaced.append(older)
        if older is not None:
            self._cut(older, NEWER)
        return playing

    def stop(self, playing: Playing) -> None:
        """The play of ``playing`` (by its id) ends now: faded out when it plays, dropped while
        it waits; nothing when it is over."""
        with self._lock:
            current = self._play
        if current is not None and current.playing.id == playing.id:
            self._cut(current, STOP)

    def current(self) -> Playing | None:
        """The sound asked or playing (its started Playing once heard), or None."""
        with self._lock:
            play = self._play
        if play is None:
            return None
        return play.started or play.playing

    def subscribe(self, listener: BeatListener) -> None:
        """``listener`` hears every play from now."""
        self.listeners.append(listener)

    # -- the thread ---------------------------------------------------------------------------

    def start(self) -> StreamSoundPlayer:
        """Step on a daemon thread of its own every :data:`TICK_S` until :meth:`close`."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="sounds", daemon=True)
            self._thread.start()
        return self

    def close(self) -> None:
        """The thread ended, the sound playing stopped (its row written)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        current = self.current()
        if current is not None:
            self.stop(current)
            self.step()

    def _run(self) -> None:
        while not self._stop.wait(TICK_S):
            try:
                self.step()
            except Exception:
                logger.exception("sounds: a step broke")

    def step(self) -> None:
        """The play moved on: fed once the voice lets go, started once its first block is sent,
        its beats told ``pulse.lead_s`` ahead, ended at its end or its cut."""
        now = self.clock()
        with self._lock:
            play, replaced, self._replaced = self._play, self._replaced, []
        for older in replaced:
            self._end(older, older.cut or NEWER, now)
        if play is None:
            return
        if play.cut is not None:
            self._end(play, play.cut, now)
            return
        if play.offset is None:
            self._feed(play, now)
            return
        if play.started is None:
            self._start(play, now)
            if play.started is None:
                return
        started = play.started
        while play.told < len(started.beats):
            at, strength = started.beats[play.told]
            t_beat = started.t0 + at
            if t_beat - play.lead_s > now:
                break
            play.told += 1
            for listener in self.listeners:
                listener.beat(t_beat, strength)
        if now >= started.end:
            self._end(play, END, now)

    # -- the steps -------------------------------------------------------------------------------

    def _feed(self, play: _Play, now: float) -> None:
        """The cut into the voice's stream once the voice lets go (or dropped past ``wait_s``)."""
        if self.voice_busy():
            if now - play.asked > play.knobs.wait_s:
                self._end(play, WAITED, now)
            return
        sound = play.playing.sound
        try:
            path = play.path if play.path is not None else sound_path(sound)
            if path is None:
                raise ValueError(f"sounds/{sound.file}: nowhere")
            pcm = load_cut(path, sound.start_s, sound.seconds, self.channel.rate)
        except (OSError, ValueError) as error:
            self._end(play, f"unreadable: {error}", now)
            return
        if not pcm:
            self._end(play, "unreadable: the cut is empty (start_s past the file's end?)", now)
            return
        seconds = len(pcm) / SAMPLE_BYTES / self.channel.rate
        if seconds < sound.seconds - 1e-3:  # the file ended sooner
            sound = replace(sound, seconds=seconds)
            play.playing = replace(play.playing, sound=sound, beats=sound.cut_beats())
        play.fed_at = now

        def ended(why: str, play: _Play = play) -> None:
            with self._lock:
                if play.cut is None:
                    play.cut = why

        play.offset = self.channel.play_sound(pcm, play.knobs.fade_s, ended)

    def _start(self, play: _Play, now: float) -> None:
        """``t0`` from the first block's play-out (the feed's moment past :data:`FIRST_BLOCK_S`
        without one); the listeners told."""
        assert play.offset is not None and play.fed_at is not None
        at = self.channel.started_at(play.offset)
        if at is None and now - play.fed_at < FIRST_BLOCK_S:
            return
        t0 = (at if at is not None else play.fed_at) + self.lag_s
        play.started = replace(play.playing, t0=t0, waiting=False)
        for listener in self.listeners:
            listener.sound_started(play.started)
        sound = play.started.sound
        waited = play.fed_at - play.asked
        after = f", after the voice ({waited:.1f} s)" if waited > 0.05 else ""
        self._say(f"{sound.name}: {sound.seconds:g} s, {sound.style},"
                  f" {len(play.started.beats)} beats{after}")  # fmt: skip

    def _cut(self, play: _Play, why: str) -> None:
        """Cut ``play`` for ``why``: a waiting one is dropped, a fed one faded out of the stream;
        it ends at the next step."""
        with self._lock:
            if play.cut is not None:
                return
            play.cut = why
            fed = play.offset is not None
        if fed:
            self.channel.stop_sound(why)

    def _end(self, play: _Play, why: str, now: float) -> None:
        """``play`` is over: the listeners told (once it started), its row written."""
        with self._lock:
            if self._play is play:
                self._play = None
        started = play.started
        if started is not None:
            for listener in self.listeners:
                listener.sound_ended(started)
        sound = (started or play.playing).sound
        played = max(0.0, min(now, started.end) - started.t0) if started is not None else 0.0
        waited = (play.fed_at if play.fed_at is not None else now) - play.asked
        if self.journal is not None:
            self.journal("sound", started.t0 if started is not None else play.asked,
                         name=sound.name, style=sound.style, seconds=round(sound.seconds, 2),
                         played_s=round(played, 2), waited_s=round(waited, 2),
                         beats=len(play.playing.beats), told=play.told, stopped_by=why)  # fmt: skip
        self._say(f"{sound.name}: {why} after {played:.1f} s")

    def _say(self, line: str) -> None:
        if self.say is not None:
            self.say(f"  (sound: {line})")


def playable(sound: Sound) -> bool:
    """Whether the player can play ``sound`` here: its file is in a tree, or this library makes
    it (the default beat)."""
    return sound.synthesized or (
        sound.file is not None and sound_path(sound, make=False) is not None
    )
