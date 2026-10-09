"""The ears, on the mic's own thread: frames in, the moments the loop acts on out.

A loudness gate (:class:`pepin.voice_live.wake.Segmenter`) cuts utterances: ``start`` when the
person begins, ``pause`` when they have been quiet ``backchannel_delay_s`` mid-utterance (the
cue for a backchannel "мм"), ``segment`` when the silence has lasted ``silence_s`` (the end of
the turn). While the robot's reply plays, the gate is held shut (the echo is not the person); with
``barge_in`` on, a :class:`pepin.voice_cascade.barge_in.BargeInDetector` (``barge_in.detector``:
loudness, a voice-activity model, or Whisper's words) decides that the person is cutting in:
``barge_in``, and the gate reopens on the speech that triggered it (the detector's preroll), so
their words are heard from the start. The XVF3800's echo canceller keeps the robot's own voice
~40 dB under the person's. While only a backchannel plays, the gate stays open and the frames
under the level threshold (``barge_in_over_floor_db`` over the floor, at least
``barge_in_min_dbfs``) count as silence. The loudest echo of each reply is kept
(:attr:`Ears.echo_peak_db`) to set the threshold from data.

With an end-of-turn model (``ears.turn_model``, :mod:`pepin.voice_cascade.turn`), a pause of
``turn.min_silence_s`` inside an utterance sends its last ``turn.window_s`` seconds to the
model's worker, and the utterance closes on the first frame after the answer when
p >= ``turn.threshold`` (unless the person spoke again meanwhile); otherwise the fixed silence
closes it as before. The segment event carries p, the model's milliseconds and the time saved.
"""

from __future__ import annotations

import concurrent.futures
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.voice_cascade.barge_in import BargeInDetector, Frame, LevelDetector
from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.player import Playback
from pepin.voice_cascade.turn import TurnAnswer, TurnWorker
from pepin.voice_live.wake import Segment, Segmenter, rms_db

EarKind = Literal["start", "pause", "segment", "barge_in"]
EarlyReader = Callable[[bytes], "concurrent.futures.Future[str]"]  # Whisper, started early
FRAME_S = FRAME_SAMPLES / RATE
SILENT_FRAME = b"\x00\x00" * FRAME_SAMPLES


@dataclass(frozen=True)
class EarEvent:
    """One moment: what, when (laptop monotonic seconds), the segment (``segment`` only), the
    frame's level and the detector's reason (``barge_in``: the voice that cut in), and for a
    segment with an end-of-turn model: what closed it (``model`` or ``silence``), the model's p
    for its last pause, the model's milliseconds and the time saved against the fixed
    silence."""

    kind: EarKind
    t: float
    segment: Segment | None = None
    level_db: float = 0.0
    why: str | None = None
    closed_by: str | None = None
    p: float | None = None
    model_ms: float | None = None
    saved_ms: float | None = None
    early: concurrent.futures.Future[str] | None = None  # Whisper on it, started in its pause


class TurnSegmenter(Segmenter):
    """The loudness gate, with the end of an utterance callable from outside and a floor that
    never follows the room below ``floor_min_db``."""

    def __init__(self, *, floor_min_db: float = -85.0, **gate: float) -> None:
        """``gate``: the :class:`pepin.voice_live.wake.Segmenter` thresholds."""
        super().__init__(**gate)
        self.floor_min_db = floor_min_db
        self.floor = max(self.floor, floor_min_db)

    def feed(self, pcm: bytes, now: float) -> Segment | None:
        """One frame, as the gate; then the floor held at its minimum."""
        segment = super().feed(pcm, now)
        self.floor = max(self.floor, self.floor_min_db)
        return segment

    def close(self, ended_s: float) -> Segment | None:
        """End the utterance in progress now (its last voiced frame at ``ended_s``); None when
        it is shorter than the gate's minimum."""
        if not self.talking:
            return None
        segment = Segment(b"".join(self._voiced), ended_s)
        self.reset()
        return segment if segment.duration_s >= self.min_s else None

    def audio(self) -> bytes:
        """The utterance in progress so far (its lead-in included)."""
        return b"".join(self._voiced)


class Ears:
    """The gate and the barge-in detector; :meth:`feed` one 20 ms frame at a time."""

    def __init__(
        self,
        config: CascadeConfig,
        playback: Callable[[], Playback],
        turn: TurnWorker | None = None,
        early: EarlyReader | None = None,
        detector: BargeInDetector | None = None,
    ) -> None:
        """``playback`` says what the speaker is doing now (the player's); ``turn`` is the
        end-of-turn model's worker (None: the fixed silence alone); ``early`` starts Whisper on
        the utterance so far (None: Whisper waits for the close); ``detector`` decides a
        barge-in (None: the level threshold of the config)."""
        self.config = config
        self.playback = playback
        self.turn = turn
        self.early = early
        self._early: concurrent.futures.Future[str] | None = None
        self.early_asked = 0  # Whisper calls started in a pause
        self.early_dropped = 0  # of them, for a pause the person talked through
        self.gate = TurnSegmenter(
            floor_min_db=config.floor_min_dbfs,
            start_db=config.start_db,
            silence_s=config.silence_s,
            max_s=config.max_utterance_s,
        )
        self._window: deque[bytes] = deque(maxlen=round(config.turn_window_s / FRAME_S))
        self._requests = 0
        self._asked: int | None = None  # the request of the pause in progress
        self.detector: BargeInDetector = detector or LevelDetector(
            over_floor_db=config.barge_in_over_floor_db,
            min_dbfs=config.barge_in_min_dbfs,
            hold_s=config.barge_in_hold_s,
        )
        self._preroll: deque[tuple[bytes, float]] = deque(
            maxlen=int(self.detector.preroll_s / FRAME_S)
        )
        self._barged = False
        self._paused = False
        self._frames = 0  # frames of the utterance in progress
        self._quiet = 0  # of them, quiet since the last loud one
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
            self._barged = False
            self._preroll.clear()  # a later barge-in starts from that reply's own frames
        if mode == "reply" and not self._barged:
            return self._during_reply(pcm, t, db)
        self.detector.feed(Frame(pcm, t, db, self.gate.floor), speaking=False)
        if mode == "backchannel" and db < self.threshold_db():
            pcm = SILENT_FRAME
        return self._listen(pcm, t, db)

    def _during_reply(self, pcm: bytes, t: float, db: float) -> list[EarEvent]:
        self.echo_peak_db = max(self.echo_peak_db, db)
        if self.gate.talking:
            self.gate.reset()
            self._asked = None
            self._drop_early()
        frame = Frame(pcm, t, db, self.gate.floor)
        if not self.config.barge_in:
            self.detector.feed(frame, speaking=False)
            return []
        self._preroll.append((pcm, t))
        why = self.detector.feed(frame, speaking=True)
        if why is None:
            return []
        self._barged = True
        events = [EarEvent("barge_in", t, level_db=db, why=why)]
        frames = list(self._preroll)
        self._preroll.clear()
        for heard, at in frames:
            events += self._listen(heard, at, rms_db(heard))
        return events

    def _listen(self, pcm: bytes, t: float, db: float) -> list[EarEvent]:
        """The gate on one frame. The pause is counted in frames, as the gate counts its
        silence: frames that arrive in a burst over WiFi carry their arrival times, not 20 ms
        steps."""
        events: list[EarEvent] = []
        loud = rms_db(pcm) > self.gate.floor + self.gate.start_db  # the gate's own test
        answer = self._answer()
        if (
            answer is not None
            and self.gate.talking
            and not loud
            and answer.p >= self.config.turn_threshold
        ):
            quiet_s = self._quiet * FRAME_S
            segment = self.gate.close(t - quiet_s)
            self._asked = None
            if segment is not None:
                saved_ms = (self.config.silence_s - quiet_s) * 1000
                events.append(
                    self._closed(EarEvent("segment", t, segment, closed_by="model", p=answer.p,
                                          model_ms=answer.ms, saved_ms=saved_ms))
                )  # fmt: skip
            return events
        was_talking = self.gate.talking
        segment = self.gate.feed(pcm, t)
        if not was_talking and self.gate.talking:
            self._frames = self._quiet = 0
            self._paused = False
            self._asked = None
            self._drop_early()
            self._window.clear()
            self.voice_peak_db = db
            events.append(EarEvent("start", t))
        if self.gate.talking:
            self.voice_peak_db = max(self.voice_peak_db, db)
            self._window.append(pcm)
            self._frames += 1
            self._quiet = 0 if loud else self._quiet + 1
            self._ask()
            self._read_early()
            quiet_s = self._quiet * FRAME_S
            spoken_s = (self._frames - self._quiet) * FRAME_S
            if not self._quiet:
                self._paused = False
            elif (
                not self._paused
                and quiet_s >= self.config.backchannel_delay_s - 1e-9
                and spoken_s >= self.config.backchannel_min_speech_s
            ):
                self._paused = True
                events.append(EarEvent("pause", t))
        if segment is not None:
            last = self._answer()
            self._asked = None
            if self.turn is None:
                events.append(self._closed(EarEvent("segment", t, segment)))
            else:
                events.append(
                    self._closed(EarEvent("segment", t, segment, closed_by="silence",
                                          p=last.p if last else None,
                                          model_ms=last.ms if last else None, saved_ms=0.0))
                )  # fmt: skip
        elif not self.gate.talking:
            self._drop_early()  # an utterance too short for a segment, or none at all
        return events

    def _read_early(self) -> None:
        """At ``transcriber.early_s`` into a pause, start Whisper on the utterance so far (once
        a pause); a voiced frame drops it: the words are not finished."""
        if self.early is None:
            return
        if not self._quiet:
            self._drop_early()
        elif self._early is None and self._quiet * FRAME_S >= self.config.early_s - 1e-9:
            self.early_asked += 1
            self._early = self.early(self.gate.audio())

    def _drop_early(self) -> None:
        if self._early is not None:
            self._early.cancel()  # not started yet: never runs; running: its answer is unused
            self.early_dropped += 1
            self._early = None

    def _closed(self, event: EarEvent) -> EarEvent:
        """A segment ``event`` with the early Whisper call of its last pause."""
        early, self._early = self._early, None
        return replace(event, early=early)

    def _ask(self) -> None:
        """At ``turn.min_silence_s`` into a pause, hand its audio to the model (once a pause);
        a voiced frame ends the pause, and its question with it."""
        if self.turn is None:
            return
        if not self._quiet:
            self._asked = None
        elif self._asked is None and self._quiet * FRAME_S >= self.config.turn_min_silence_s - 1e-9:
            self._requests += 1
            self._asked = self._requests
            self.turn.ask(self._requests, b"".join(self._window))

    def _answer(self) -> TurnAnswer | None:
        """The model's answer for the pause in progress, once it has come."""
        if self.turn is None or self._asked is None:
            return None
        latest = self.turn.latest
        return latest if latest is not None and latest.id == self._asked else None
