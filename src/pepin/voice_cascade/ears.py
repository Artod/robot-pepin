"""The ears, on the mic's own thread: frames in, the moments the loop acts on out.

A loudness gate (:class:`pepin.voice_live.wake.Segmenter`) cuts utterances: ``start`` when the
person begins, ``pause`` when they have been quiet ``backchannel_delay_s`` mid-utterance (the
cue for a backchannel "мм"), ``segment`` when the silence has lasted ``silence_s`` (the end of
the turn). While the robot's reply plays, the gate is held shut (the echo is not the person); with
``barge_in`` on, a voice ``barge_in_over_floor_db`` above the room's floor (and at least
``barge_in_min_dbfs``) for ``barge_in_hold_s`` is the person cutting in: ``barge_in``, and the
gate reopens on the speech that triggered it, so their words are heard from the start. The
XVF3800's echo canceller keeps the robot's own voice ~40 dB under the person's, which is what
makes a level threshold safe. While only a backchannel plays, the gate stays open and the
frames under that threshold count as silence. The loudest echo of each reply is kept
(:attr:`Ears.echo_peak_db`) to set the threshold from data.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.player import Playback
from pepin.voice_live.wake import Segment, Segmenter, rms_db

EarKind = Literal["start", "pause", "segment", "barge_in"]
FRAME_S = FRAME_SAMPLES / RATE
PREROLL_S = 0.6  # the speech before a barge-in is confirmed, given to the gate when it is
SILENT_FRAME = b"\x00\x00" * FRAME_SAMPLES


@dataclass(frozen=True)
class EarEvent:
    """One moment: what, when (laptop monotonic seconds), the segment (``segment`` only) and
    the frame's level (``barge_in``: the voice that cut in)."""

    kind: EarKind
    t: float
    segment: Segment | None = None
    level_db: float = 0.0


class Ears:
    """The gate and the barge-in detector; :meth:`feed` one 20 ms frame at a time."""

    def __init__(self, config: CascadeConfig, playback: Callable[[], Playback]) -> None:
        """``playback`` says what the speaker is doing now (the player's)."""
        self.config = config
        self.playback = playback
        self.gate = Segmenter(
            start_db=config.start_db, silence_s=config.silence_s, max_s=config.max_utterance_s
        )
        self._preroll: deque[tuple[bytes, float]] = deque(maxlen=int(PREROLL_S / FRAME_S))
        self._loud_s = 0.0
        self._barged = False
        self._paused = False
        self._started_s = 0.0
        self.echo_peak_db = -120.0  # loudest frame heard while a reply played (since reset)
        self.voice_peak_db = -120.0  # loudest frame of the person's last utterance

    def threshold_db(self) -> float:
        """The level a voice must reach to cut in."""
        cfg = self.config
        return max(self.gate.floor + cfg.barge_in_over_floor_db, cfg.barge_in_min_dbfs)

    def feed(self, pcm: bytes, t: float) -> list[EarEvent]:
        """One frame at laptop time ``t``; the moments it makes."""
        mode = self.playback()
        db = rms_db(pcm)
        if mode == "silent":
            self._barged, self._loud_s = False, 0.0
            self._preroll.clear()  # a later barge-in starts from that reply's own frames
        if mode == "reply" and not self._barged:
            return self._during_reply(pcm, t, db)
        if mode == "backchannel" and db < self.threshold_db():
            pcm = SILENT_FRAME
        return self._listen(pcm, t, db)

    def _during_reply(self, pcm: bytes, t: float, db: float) -> list[EarEvent]:
        self.echo_peak_db = max(self.echo_peak_db, db)
        if self.gate.talking:
            self.gate.reset()
        if not self.config.barge_in:
            return []
        self._preroll.append((pcm, t))
        self._loud_s = self._loud_s + FRAME_S if db > self.threshold_db() else 0.0
        if self._loud_s < self.config.barge_in_hold_s - 1e-9:
            return []
        self._barged, self._loud_s = True, 0.0
        events = [EarEvent("barge_in", t, level_db=db)]
        frames = list(self._preroll)
        self._preroll.clear()
        for frame, at in frames:
            events += self._listen(frame, at, rms_db(frame))
        return events

    def _listen(self, pcm: bytes, t: float, db: float) -> list[EarEvent]:
        events: list[EarEvent] = []
        was_talking = self.gate.talking
        segment = self.gate.feed(pcm, t)
        if not was_talking and self.gate.talking:
            self._started_s, self._paused = t, False
            self.voice_peak_db = db
            events.append(EarEvent("start", t))
        if self.gate.talking:
            self.voice_peak_db = max(self.voice_peak_db, db)
            quiet_s = t - self.gate.last_voice_s
            if quiet_s < FRAME_S:
                self._paused = False
            elif (
                not self._paused
                and quiet_s >= self.config.backchannel_delay_s
                and self.gate.last_voice_s - self._started_s >= self.config.backchannel_min_speech_s
            ):
                self._paused = True
                events.append(EarEvent("pause", t))
        if segment is not None:
            events.append(EarEvent("segment", t, segment))
        return events
