"""The speaker side: the voice's PCM paced to the speaker (:class:`pepin.voice_live.audio.Pacer`),
with a clock of what has actually played.

Every part of a reply (a sentence, or a piece of one a style marker cut; one voice call each)
is marked by where its audio sits in the stream fed to the pacer, and the speaker is wrapped
(:class:`PlayoutClock`) so the time each block plays is known. A barge-in asks
:meth:`Player.flush` how far the stream had played: that, against the marks, is which parts
the person heard (:meth:`Player.heard`, by the gesture engine's word clock when it is lent:
:attr:`Player.words`). The same marks place the body's cues on the reply's audio
(:meth:`Player.mark`, :meth:`Player.played`), and what is queued ahead of the speaker tells the
word clock how long a reply's first word waits (:meth:`Player.queued_audio`). The same clock
times the speaker's start and end of each turn, and keeps each block's level
(:meth:`PlayoutClock.level_between`): the reference a barge-in detector predicts the robot's
own echo from.

A FILLER THE REPLY TAKES OVER. Filler and reply share the one stream, so a reply ready while a
thinking filler still plays used to queue behind it (2026-10-12: 9 of 19 fillers, 145-1095 ms,
median 456). A backchannel fed with a fade (:meth:`Player.feed`, ``fade_s``) goes to the
speaker only that far ahead instead of the pacer's lead, since the board plays whatever it was
sent; the reply's first chunk then turns the part of it not sent yet into a linear ramp to
silence over ``fade_s``, on the stream's own samples, and follows it at once. The reply waits
for what was already sent and the ramp, never for the rest of the filler.

A SOUND (:mod:`pepin.sound_player`: a registry sound, a dance's beat) goes through the same
stream (:meth:`Player.play_sound`): the board plays one stream and two writers would interleave,
and the barge-in detector's echo model and the ears' gate see it as the robot's own sound
(:meth:`Player.playback` says ``sound``). SPEECH WINS: speech fed while a sound has audio not yet
sent takes the speaker as a reply takes a filler over (the sound's unsent part ramps to silence
over its ``fade_s`` and the rest is dropped; its ``ended("speech")`` is told), a flush drops it
(``ended("barge-in")``), :meth:`Player.stop_sound` ends it the same way. Its blocks' levels move
no mouth (``on_level`` is the voice's).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pepin.audio_link import SAMPLE_BYTES, Speaker
from pepin.expression.forecast import CHAR_S, QueuedAudio
from pepin.voice_cascade.stages import Played
from pepin.voice_live.wake import rms_db

Kind = Literal["reply", "backchannel", "sound"]
Playback = Literal["silent", "reply", "backchannel", "sound"]
HISTORY_BLOCKS = 3000  # ~2 min of 40 ms blocks: enough to time any turn's end
NOTHING_DB = -120.0  # the level of no block


@dataclass(frozen=True)
class FillerFade:
    """A reply's first chunk met a filler still playing: the filler's turn, how long the ramp
    to silence is (0: too little of it was left to fade), how much of the filler it dropped,
    and how much was already sent and plays before the ramp (seconds)."""

    turn: int
    faded_s: float
    cut_s: float
    ahead_s: float


FadeSink = Callable[[FillerFade], None]
SoundEnded = Callable[[str], None]  # why a sound was cut: speech, barge-in, or a stop's word


@dataclass
class _Sound:
    """A sound in the stream: its first and last stream second, the ramp speech fades it out
    with, and who hears its cut."""

    start: float
    end: float
    fade_s: float
    ended: SoundEnded


class WordLaw(Protocol):
    """When a reply's characters are heard (the gesture engine's word clock,
    :class:`pepin.expression.forecast.WordClock`): read from the barge-in's thread."""

    def heard_share(self, turn: int, index: int, played: float) -> float | None:
        """The share of the part's characters heard by stream second ``played``; None when it
        does not hold the part."""
        ...


class PlayoutClock:
    """A :class:`pepin.audio_link.Speaker` wrapper that keeps the play-out timeline: each block
    plays when the one before it ends, or now when the speaker is idle (the pacer's own model of
    the speaker). Thread-safe (the pacer plays, a barge-in flushes)."""

    def __init__(self, speaker: Speaker, rate: int, clock: Callable[[], float]) -> None:
        """``rate``: the speaker's play rate."""
        self.speaker, self.rate, self.clock = speaker, rate, clock
        self._lock = threading.Lock()
        self.sent_s = 0.0  # stream seconds sent and not flushed away
        self.cursor = 0.0  # when the last block sent ends
        self._blocks: deque[tuple[float, float, float]] = deque(maxlen=HISTORY_BLOCKS)
        self._levels: deque[tuple[float, float, float]] = deque(maxlen=HISTORY_BLOCKS)

    def play(self, pcm: bytes) -> None:
        """Send a block; note where in the stream it is, when it plays and how loud it is."""
        duration = len(pcm) / SAMPLE_BYTES / self.rate
        db = rms_db(pcm)
        with self._lock:
            start = max(self.cursor, self.clock())
            self._blocks.append((self.sent_s, start, duration))
            self._levels.append((start, start + duration, db))
            self.sent_s += duration
            self.cursor = start + duration
        self.speaker.play(pcm)

    def play_end(self) -> None:
        """The utterance is complete."""
        self.speaker.play_end()

    def flush(self) -> None:
        """The speaker drops what it has not played yet."""
        with self._lock:
            now = self.clock()
            self.sent_s -= max(0.0, self.cursor - now)
            self.cursor = min(self.cursor, now)
            while self._levels and self._levels[-1][0] >= now:  # never to play
                self._levels.pop()
            if self._levels and self._levels[-1][1] > now:
                start, _, db = self._levels.pop()
                self._levels.append((start, now, db))
        self.speaker.flush()

    def level_between(self, t0: float, t1: float) -> float:
        """The loudest block playing at any moment of ``t0..t1`` (dBFS; -120 for none)."""
        loudest = NOTHING_DB
        with self._lock:
            for start, end, db in reversed(self._levels):  # in play order, none overlapping
                if end <= t0:
                    break
                if start < t1:
                    loudest = max(loudest, db)
        return loudest

    def played(self) -> float:
        """Stream seconds played by now."""
        with self._lock:
            return self.sent_s - max(0.0, self.cursor - self.clock())

    def when(self, offset: float, *, end: bool = False) -> float | None:
        """When the stream's second ``offset`` starts playing (``end``: finishes playing, for
        the last sample of a stretch); None when it was never sent or is too old to remember.
        A block boundary belongs to the block after it, or with ``end`` to the one before."""
        eps = 1e-6
        with self._lock:
            for start_offset, start, duration in reversed(self._blocks):
                low, high = start_offset, start_offset + duration
                inside = (
                    low + eps < offset <= high + eps if end else low - eps <= offset < high - eps
                )
                if inside:
                    return start + (offset - start_offset)
        return None


class Player:
    """The voice's PCM to the speaker at real time, with the marks and the clock above."""

    def __init__(
        self,
        speaker: Speaker,
        *,
        rate: int,
        lead_s: float = 0.3,
        block_s: float = 0.04,
        on_level: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``rate``: the speaker's play rate (the PCM fed must already be at it)."""
        from pepin.voice_live.audio import Pacer

        self.rate = rate
        self.clock = clock
        self.playout = PlayoutClock(speaker, rate, clock)
        self._on_level = on_level
        self.pacer = Pacer(
            self.playout,
            rate=rate,
            lead_s=lead_s,
            block_s=block_s,
            on_level=self._level if on_level is not None else None,
            clock=clock,
        )
        self._lock = threading.Lock()
        self._fed = 0.0  # stream seconds fed (aligned to the played clock at every flush)
        self._kind: Kind = "reply"
        self._marks: dict[tuple[int, int], list[float]] = {}  # (turn, part) -> [start, end]
        self._fadeable: tuple[int, float] | None = None  # the queued filler's (turn, fade_s)
        self._sound: _Sound | None = None  # the sound in the stream, until speech or a stop
        self.flushed_at: float | None = None
        self.on_fade: FadeSink | None = None  # told on the feeding thread, after the reply's chunk
        self.words: WordLaw | None = None  # the word clock :meth:`heard` reads, when lent

    def start(self) -> threading.Event:
        """Run the pacer on its thread; set the returned event to stop it."""
        return self.pacer.start()

    def feed(self, pcm: bytes, kind: Kind, turn: int, index: int, fade_s: float = 0.0) -> None:
        """Queue a chunk of part ``index`` of reply ``turn`` (a backchannel: index -1). A
        backchannel with ``fade_s`` is a filler the reply takes over (the module's notes): sent
        at most ``fade_s`` ahead, ramped to silence over ``fade_s`` by the reply's first chunk
        while it still plays, which :attr:`on_fade` hears."""
        if not pcm:
            return
        duration = len(pcm) / SAMPLE_BYTES / self.rate
        fade: FillerFade | None = None
        with self._lock:
            taken = self._take_sound()  # speech wins
            if kind == "reply" and self._fadeable is not None:
                fade = self._fade_filler(*self._fadeable)
            elif kind == "backchannel":
                self._fadeable = (turn, fade_s) if fade_s > 0 else None
                self.pacer.hold(fade_s if fade_s > 0 else None)
            start = self._fed
            self._fed += duration
            self._kind = kind
            if kind == "reply":
                mark = self._marks.setdefault((turn, index), [start, start])
                mark[1] = self._fed
            self.pacer.feed(pcm)
        if taken is not None:
            taken.ended("speech")
        if fade is not None and self.on_fade is not None:
            self.on_fade(fade)

    def play_sound(self, pcm: bytes, fade_s: float, ended: SoundEnded) -> float:
        """Queue a sound's PCM (at the player's rate) behind whatever is queued, as one
        utterance: the stream second its first sample is at. Speech fed after it, a flush or
        :meth:`stop_sound` cut it and tell ``ended`` why (the module's notes); a sound already
        in the stream is cut first (``ended("a newer sound")``)."""
        duration = len(pcm) // SAMPLE_BYTES / self.rate
        with self._lock:
            older = self._take_sound()
            start = self._fed
            self._fed += duration
            self._kind = "sound"
            self._sound = _Sound(start, self._fed, fade_s, ended)
            self._fadeable = None  # the voice let go: no filler's hold sends it short of the lead
            self.pacer.hold(None)
            self.pacer.feed(pcm[: len(pcm) - len(pcm) % SAMPLE_BYTES])
            self.pacer.end()  # the board closes its device after it
        if older is not None:
            older.ended("a newer sound")
        return start

    def stop_sound(self, why: str = "stop") -> bool:
        """The sound in the stream ends: its part not yet sent ramps to silence over its
        ``fade_s`` and the rest is dropped (what was sent plays out); its ``ended(why)`` is
        told. False when no sound had audio left to send."""
        with self._lock:
            taken = self._take_sound()
        if taken is not None:
            taken.ended(why)
        return taken is not None

    def started_at(self, offset: float) -> float | None:
        """When stream second ``offset`` starts playing (:meth:`PlayoutClock.when`): None until
        its block is sent."""
        return self.playout.when(offset)

    def _take_sound(self) -> _Sound | None:
        """The sound in the stream, its part not yet sent faded and dropped (under the lock);
        None without one, or once all of it was sent (it ends by itself, within the lead)."""
        sound, self._sound = self._sound, None
        if sound is None or self.playout.sent_s >= sound.end - 1e-6:
            return None
        _, cut = self.pacer.fade(sound.fade_s)
        self._fed -= cut
        return sound

    def _level(self, value: float) -> None:
        """A block's level as it plays: the voice's moves the mouth, a sound's does not."""
        if self._kind != "sound" and self._on_level is not None:
            self._on_level(value)

    def _fade_filler(self, turn: int, fade_s: float) -> FillerFade | None:
        """The reply's first chunk is here: the queued filler ramps out (under the lock)."""
        self._fadeable = None
        ahead = max(0.0, self.playout.sent_s - self.playout.played())
        faded, cut = self.pacer.fade(fade_s)
        self.pacer.hold(None)
        self._fed -= cut
        if not (faded or ahead):
            return None  # the filler had ended
        return FillerFade(turn, faded, cut, ahead)

    def end(self) -> None:
        """The turn's audio is complete: play out what is queued."""
        with self._lock:
            self._fadeable = None
            self.pacer.hold(None)
        self.pacer.end()

    def flush(self) -> float:
        """Barge-in: silence the speaker now (a sound playing too: ``ended("barge-in")``); the
        stream seconds that had played."""
        with self._lock:
            played = self.playout.played()
            sound, self._sound = self._sound, None
            self.pacer.flush()
            self._fed = self.playout.sent_s
            self._fadeable = None
            self.flushed_at = self.clock()
        if sound is not None and played < sound.end - 1e-6:
            sound.ended("barge-in")
        return played

    def queued_audio(self) -> QueuedAudio:
        """What the player holds ahead of the speaker now (any thread): seconds fed and not
        played, of them already sent to the speaker, and the fade a queued filler is taken over
        with (0: none, or one the reply waits for whole)."""
        with self._lock:
            played = self.playout.played()
            fed = max(0.0, self._fed - played)
            sent = max(0.0, self.playout.sent_s - played)
            fade = self._fadeable[1] if self._fadeable is not None else 0.0
            sound = self._sound
            if sound is not None and self.playout.sent_s < sound.end - 1e-6:
                fade = sound.fade_s  # the reply takes the speaker from it as from a filler
        return QueuedAudio(fed, min(sent, fed), fade)

    def playback(self) -> Playback:
        """What the speaker is doing: nothing, a reply, a backchannel, or a sound."""
        return self._kind if self.pacer.speaking() else "silent"

    def speaking(self) -> bool:
        """Audio is queued or playing (with the room's tail)."""
        return self.pacer.speaking()

    def played(self) -> float:
        """Stream seconds played by now (the clock a part's :meth:`mark` is on)."""
        return self.playout.played()

    def mark(self, turn: int, index: int) -> tuple[float, float] | None:
        """Where part ``index`` of reply ``turn`` sits in the stream so far: its first and last
        second fed (the last grows while its audio is still being made); None before its
        first chunk, or once the turn is forgotten. Read from any thread (the body's cues)."""
        with self._lock:
            mark = self._marks.get((turn, index))
            return None if mark is None else (mark[0], mark[1])

    def span(self, turn: int) -> tuple[float, float] | None:
        """Where reply ``turn``'s audio sits in the stream: its first and last second."""
        with self._lock:
            spans = [v for (t, _), v in self._marks.items() if t == turn]
        if not spans:
            return None
        return min(s for s, _ in spans), max(e for _, e in spans)

    def heard(self, turn: int, played: float, texts: list[str]) -> Played:
        """Which of reply ``turn``'s sentences (``texts``, in order) had played by stream second
        ``played``: the whole ones, then the words of the one cut, in proportion to its
        characters heard (the word clock's law when it holds the part: the leading silence,
        then the rest; else the share of its fed audio, at least :data:`CHAR_S` a character)."""
        with self._lock:
            marks = {i: tuple(v) for (t, i), v in self._marks.items() if t == turn}
        whole = 0
        partial = ""
        for index, text in enumerate(texts):
            mark = marks.get(index)
            if mark is None:  # not spoken (an aside), or not reached yet
                if any(i > index and start < played for i, (start, _) in marks.items()):
                    whole = index + 1
                    continue
                break
            start, end = mark
            if end <= played + 0.05:
                whole = index + 1
                continue
            if start < played:
                words = text.split()
                law = self.words.heard_share(turn, index, played) if self.words else None
                share = (played - start) / max(end - start, CHAR_S * len(text))
                share = law if law is not None else share
                partial = " ".join(words[: round(len(words) * min(1.0, share))])
            break
        return Played(whole, partial)

    def forget(self, turn: int) -> None:
        """Drop turn ``turn``'s marks (its line is written)."""
        with self._lock:
            for key in [k for k in self._marks if k[0] <= turn]:
                del self._marks[key]
