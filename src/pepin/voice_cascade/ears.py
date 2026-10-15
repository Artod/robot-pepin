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

With ``ears.turn_model`` two-gates (:mod:`pepin.voice_cascade.turn_text`), the transcriber's
early read of a pause (``transcriber.early_s``) is judged as it lands, on the transcriber's
thread: words complete and the sound agreeing (``turn.acoustic``: the pause itself, or Smart
Turn's p >= ``turn.threshold``) close the utterance at ``turn.early_close_s`` of quiet, words
unfinished hold it open to ``turn.incomplete_s``, anything else closes it at ``silence_s``. The
segment event carries what closed it (``both``, ``silence`` or ``wait``), the verdict and why.

With a listener for the early words (``words``), each early read that lands while its pause is
still the one in progress is told as a ``words`` event, on the transcriber's thread (after
two-gates' verdict): the utterance so far, its words and the verdict. A pause the person talked
through tells nothing; the read that closes the utterance is the segment's ``early``.
"""

from __future__ import annotations

import concurrent.futures
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.voice_cascade.barge_in import BargeInDetector, Frame, LevelDetector
from pepin.voice_cascade.config import TWO_GATES, CascadeConfig
from pepin.voice_cascade.player import Playback
from pepin.voice_cascade.turn import TurnAnswer, TurnWorker
from pepin.voice_cascade.turn_text import Acoustic, Deadline, Judgement, TextJudge, two_gates
from pepin.voice_live.wake import Segment, Segmenter, rms_db

EarKind = Literal["start", "pause", "segment", "barge_in", "words"]
EarlyReader = Callable[[bytes], "concurrent.futures.Future[str]"]  # Whisper, started early
FRAME_S = FRAME_SAMPLES / RATE
SILENT_FRAME = b"\x00\x00" * FRAME_SAMPLES


@dataclass(frozen=True)
class EarEvent:
    """One moment: what, when (laptop monotonic seconds), the segment (``segment`` only), the
    frame's level and the detector's reason (``barge_in``: the voice that cut in), the lead-in
    kept in front of the opening (``start``: seconds), and for a segment with an end-of-turn
    model: what closed it (``model`` or ``silence``; two-gates: ``both``, ``silence`` or
    ``wait``), the model's p for its last pause, the model's milliseconds and the time saved
    against the fixed silence (negative: waited longer), and two-gates' verdict on the words of
    its last pause with the reason and the judge's milliseconds. ``words``: an early read
    landed in the pause in progress; ``t`` is the frame it was started on, ``segment`` the
    utterance so far (``ended_s``: its last loud frame), ``early`` the read, ``text`` its words
    (alone, never joined to an utterance it continues) and the verdict when two-gates judged
    them."""

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
    lead_s: float | None = None  # start: the audio kept from before the opening
    verdict: str | None = None  # two-gates: complete, incomplete or unsure (None: not read)
    verdict_why: str | None = None
    text_ms: float | None = None
    text: str | None = None  # words: the early read's words


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
        self.reset(keep_quiet_tail=True)
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
        judge: TextJudge | None = None,
        words: Callable[[EarEvent], None] | None = None,
    ) -> None:
        """``playback`` says what the speaker is doing now (the player's); ``turn`` is the
        end-of-turn model's worker (None: the fixed silence alone); ``early`` starts Whisper on
        the utterance so far (None: Whisper waits for the close); ``detector`` decides a
        barge-in (None: the level threshold of the config); ``judge`` reads the early words
        for two-gates (None: no verdict, so the fixed silence); ``words`` hears each early read
        of the pause in progress as it lands (on the transcriber's thread)."""
        self.config = config
        self.playback = playback
        self.turn = turn
        self.early = early
        self.judge = judge
        self.words = words
        self.two_gates = config.turn_model == TWO_GATES
        self._early: concurrent.futures.Future[str] | None = None
        self._reads = 0  # early reads started (each names its pause)
        self._reading: int | None = None  # the read of the pause in progress
        self._judged: tuple[int, Judgement] | None = None  # the latest verdict and its read
        # the early read of the last utterance closed, and when: one that starts within
        # continuation_s joins it in the loop, so its words are judged after those
        self._last_read: concurrent.futures.Future[str] | None = None
        self._last_closed_s = -1e9
        self._carry: concurrent.futures.Future[str] | None = None
        self.early_asked = 0  # Whisper calls started in a pause
        self.early_dropped = 0  # of them, for a pause the person talked through
        self.gate = TurnSegmenter(
            floor_min_db=config.floor_min_dbfs,
            start_db=config.start_db,
            # two-gates closes the utterance itself; the gate's own close is the longest wait
            silence_s=config.turn_incomplete_s if self.two_gates else config.silence_s,
            max_s=config.max_utterance_s,
            pre_s=config.lead_s,
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
        self.gate.speech = self._speech_now
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

    def _speech_now(self) -> bool:
        """Whether the barge-in model hears speech in the latest frames (``ears.vad_open``):
        the gate's second opinion before a loud run opens an utterance. True without a
        probability (a level or words detector) or with the knob at 0: loudness alone."""
        p = getattr(self.detector, "p", None)
        if self.config.vad_open <= 0 or not isinstance(p, float):
            return True
        return p >= self.config.vad_open

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
            and not self.two_gates  # there the sound alone never closes the turn
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
            joins = t - self._last_closed_s <= self.config.continuation_s
            self._carry = self._last_read if joins else None
            self.voice_peak_db = db
            # measured here, on the mic's thread: the loop reads the event frames later
            lead_s = len(self.gate.audio()) / 2 / RATE
            events.append(EarEvent("start", t, lead_s=lead_s))
        if self.gate.talking:
            self.voice_peak_db = max(self.voice_peak_db, db)
            self._window.append(pcm)
            self._frames += 1
            self._quiet = 0 if loud else self._quiet + 1
            self._ask()
            self._read_early(t)
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
        deadline: Deadline | None = None
        if self.two_gates and (self.gate.talking or segment is not None):
            deadline = self._deadline()
            quiet_s = self._quiet * FRAME_S
            if segment is None and self._quiet and quiet_s >= deadline.quiet_s - 1e-9:
                segment = self.gate.close(t - quiet_s)
        if segment is not None:
            last = self._answer()
            self._asked = None
            if deadline is not None:
                events.append(self._closed(self._gated(t, segment, deadline, last)))
            elif self.turn is None:
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

    def _read_early(self, t: float) -> None:
        """At ``transcriber.early_s`` into a pause, start Whisper on the utterance so far (once
        a pause); a voiced frame drops it: the words are not finished."""
        if self.early is None:
            return
        if not self._quiet:
            self._drop_early()
        elif self._early is None and self._quiet * FRAME_S >= self.config.early_s - 1e-9:
            self.early_asked += 1
            self._reads += 1
            self._reading = self._reads
            pcm = self.gate.audio()
            so_far = Segment(pcm, t - self._quiet * FRAME_S)
            self._early = self.early(pcm)
            judge = self.judge if self.two_gates else None
            if judge is not None or self.words is not None:
                callback = self._judging(self._reads, judge, self._carry, t, so_far)
                self._early.add_done_callback(callback)

    def _judging(
        self,
        read: int,
        judge: TextJudge | None,
        carry: concurrent.futures.Future[str] | None,
        t: float,
        so_far: Segment,
    ) -> Callable[[concurrent.futures.Future[str]], None]:
        """The callback that judges early read ``read``'s words where they land (the
        transcriber's thread; at once if they are already there), after the words of the
        utterance it continues (``carry``, when they are read), then tells them as a
        ``words`` event while their pause is still the one in progress."""

        def judged(future: concurrent.futures.Future[str]) -> None:
            if future.cancelled() or future.exception() is not None:
                return
            text = alone = future.result()
            read_before = carry is not None and carry.done() and not carry.cancelled()
            if carry is not None and read_before and carry.exception() is None:
                text = f"{carry.result()} {text}".strip()
            verdict = judge.judge(text) if judge is not None else None
            if verdict is not None:
                self._judged = (read, verdict)
            if self.words is not None and self._reading == read:
                self.words(
                    EarEvent("words", t, so_far, early=future, text=alone,
                             verdict=verdict.verdict if verdict else None,
                             verdict_why=verdict.why if verdict else None,
                             text_ms=verdict.ms if verdict else None)
                )  # fmt: skip

        return judged

    def _drop_early(self) -> None:
        self._reading = None
        if self._early is not None:
            self._early.cancel()  # not started yet: never runs; running: its answer is unused
            self.early_dropped += 1
            self._early = None

    def _closed(self, event: EarEvent) -> EarEvent:
        """A segment ``event`` with the early Whisper call of its last pause."""
        early, self._early = self._early, None
        self._reading = None
        self._last_read, self._last_closed_s = early, event.t
        return replace(event, early=early)

    def _verdict(self) -> Judgement | None:
        """The judge's verdict on the pause in progress, once its words are read."""
        judged = self._judged
        if judged is None or self._reading is None or judged[0] != self._reading:
            return None
        return judged[1]

    def _deadline(self) -> Deadline:
        """Two-gates: how much quiet closes the pause in progress, from what is known now."""
        cfg = self.config
        acoustic: Acoustic = "agrees"  # turn.acoustic pause: the pause is the sound's gate
        if cfg.turn_acoustic != "pause":
            answer = self._answer()
            acoustic = (
                "unknown"
                if answer is None
                else "agrees"
                if answer.p >= cfg.turn_threshold
                else "disagrees"
            )
        verdict = self._verdict()
        return two_gates(
            verdict.verdict if verdict is not None else None,
            acoustic,
            early_close_s=cfg.turn_early_close_s,
            silence_s=cfg.silence_s,
            incomplete_s=cfg.turn_incomplete_s,
        )

    def _gated(
        self, t: float, segment: Segment, deadline: Deadline, answer: TurnAnswer | None
    ) -> EarEvent:
        """A two-gates segment event: what closed it, the verdict, the sound's p."""
        verdict = self._verdict()
        quiet_s = t - segment.ended_s
        return EarEvent(
            "segment", t, segment, closed_by=deadline.closed_by,
            p=answer.p if answer else None, model_ms=answer.ms if answer else None,
            saved_ms=(self.config.silence_s - quiet_s) * 1000,
            verdict=verdict.verdict if verdict else None,
            verdict_why=verdict.why if verdict else None,
            text_ms=verdict.ms if verdict else None,
        )  # fmt: skip

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
