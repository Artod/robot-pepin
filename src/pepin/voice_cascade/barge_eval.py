"""Barge-in detectors replayed on recorded conversations (``scripts/barge_in_eval.py``).

A conversation's record (:mod:`pepin.voice_cascade.log`) keeps the mic as the loop heard it,
the frames under the robot's own voice included: in.wav starts with the wake utterance (the
first ``heard`` event's ``seconds``), then one 20 ms frame per 20 ms from the open, so in.wav
second ``wake_s + t`` is event second ``t``. Each turn that played is a :class:`Window`: it
ends at the turn row's own time (an interrupted turn is written at the barge-in, a committed one
the pacer's tail after its last sample) and starts the played span before.

The truth comes from the record itself (:func:`sound_events`, :func:`is_speech`): the loud
moments inside a window, each read by Whisper; a moment is the person's speech when Whisper
finds words in it that are not its noise lines, not backchannels alone and not the robot's own
sentence. A replay (:func:`replay`) runs the whole mic through :class:`pepin.voice_cascade.ears.
Ears` with a detector, the speaker "playing" through each window (an interrupted window goes on
``extend_s`` past its barge-in: the person went on talking, so a slower detector is judged on
the rest of their words). :func:`score` then calls each window's first trigger caught (inside
the person's speech: the reaction is the time since its onset), false (on anything else) or
missing (the person spoke and nothing fired).
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.voice_cascade.barge_in import BargeInDetector, interrupting_words, own_echo
from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.ears import Ears
from pepin.voice_cascade.player import Playback
from pepin.voice_live.wake import is_hallucination, words

FRAME_S = FRAME_SAMPLES / RATE
FRAME_BYTES = FRAME_SAMPLES * 2
TAIL_S = 0.3  # the pacer counts this much after the last sample as still speaking
EXTEND_S = 2.0  # an interrupted window replays this far past its barge-in
CONTEXT_S = 0.3  # Whisper hears this much either side of a loud moment
EARLY_S = 0.1  # a trigger this soon before a voice's onset is still that voice's
LATE_S = 0.8  # ... and this long after it ends (Whisper's answer comes after the words)
Outcome = Literal["caught", "false", "missed", "quiet"]


@dataclass(frozen=True)
class Window:
    """One reply on the mic's timeline (event seconds): its first sample, its last (or the
    barge-in), what it said and how it ended."""

    turn: int
    outcome: str
    start: float
    end: float
    reply: str

    def replay_end(self, extend_s: float = EXTEND_S) -> float:
        """Until when the replay keeps the speaker "playing" (before :func:`spans` stops it at
        the next window)."""
        return self.end + (extend_s if self.outcome == "interrupted" else TAIL_S)


def spans(windows: Sequence[Window], extend_s: float = EXTEND_S) -> list[tuple[Window, float]]:
    """Each window in time order with the end of its replay: :meth:`Window.replay_end`, but
    two frames of silence before the next window at the latest (the speaker always went quiet
    between two replies)."""
    ordered = sorted(windows, key=lambda w: w.start)
    out = []
    for w, after in zip(ordered, [*ordered[1:], None], strict=True):
        stop = w.replay_end(extend_s)
        if after is not None:
            stop = max(w.start, min(stop, after.start - 2 * FRAME_S))
        out.append((w, stop))
    return out


@dataclass
class Recording:
    """One conversation: the mic from the open on (s16le, 16 kHz), its windows, whether it ran
    with barge-in on, the level of the robot's voice (dBFS, the loudest 5 % of its 40 ms
    blocks) and how it was woken (``always``: the latency bench, recorded speech replayed)."""

    name: str
    mic: bytes
    windows: list[Window]
    barge_in: bool
    voice_db: float
    wake: str = "name"

    @classmethod
    def load(cls, directory: Path) -> Recording:
        """From a ``cascade_*`` directory (a WAV header left unfinished by a crash is read
        past)."""
        rows = [json.loads(line) for line in (directory / "events.jsonl").open()]
        opened = next(r for r in rows if r["event"] == "open")
        wake_s = float(next(r for r in rows if r["event"] == "heard")["seconds"])
        mic, rate = read_wav(directory / "in.wav")
        if rate != RATE:
            raise ValueError(f"{directory}/in.wav: {rate} Hz, expected {RATE}")
        skip = round(wake_s * RATE) * 2
        mic = mic[skip : skip + (len(mic) - skip) // FRAME_BYTES * FRAME_BYTES]
        voice, voice_rate = read_wav(directory / "out.wav")
        return cls(
            directory.name,
            mic,
            windows_of(rows),
            bool(opened.get("flags", {}).get("barge_in")),
            loud_level(voice, voice_rate),
            str(opened.get("wake", "name")),
        )

    @property
    def seconds(self) -> float:
        """The mic's length from the open."""
        return len(self.mic) / 2 / RATE

    def pcm(self, t0: float, t1: float) -> bytes:
        """The mic between event seconds ``t0`` and ``t1``."""
        a = max(0, round(t0 * RATE)) * 2
        b = max(a, round(t1 * RATE) * 2)
        return self.mic[a:b]

    def floor_db(self) -> float:
        """The room's floor: the 30th percentile of the mic's frame levels."""
        levels = frame_levels(self.mic)
        return float(np.percentile(levels, 30)) if len(levels) else -120.0

    def played(self, t0: float, t1: float) -> float:
        """The robot's voice level over ``t0..t1`` as the player would report it, taken as its
        loud level wherever a window plays (an upper bound: the record has no block clock)."""
        playing = any(w.start < t1 and w.end > t0 for w in self.windows)
        return self.voice_db if playing else -120.0


def read_wav(path: Path) -> tuple[bytes, int]:
    """A 16-bit mono WAV's samples and rate, read from its data chunk to the file's end."""
    raw = path.read_bytes()
    rate = int.from_bytes(raw[24:28], "little")
    data = raw.find(b"data", 12)
    body = raw[data + 8 :]
    return body[: len(body) // 2 * 2], rate


def windows_of(rows: Sequence[dict[str, Any]]) -> list[Window]:
    """The windows of a conversation's events: every turn that reached the speaker."""
    out = []
    for r in rows:
        if r["event"] != "turn":
            continue
        ms = r.get("ms", {})
        if ms.get("speaker_start") is None or ms.get("speaker_end") is None:
            continue
        end = float(r["t"]) - (0.0 if r["outcome"] == "interrupted" else TAIL_S)
        start = end - (ms["speaker_end"] - ms["speaker_start"]) / 1000
        out.append(Window(int(r["turn"]), str(r["outcome"]), start, end, str(r.get("reply", ""))))
    return out


def frame_levels(pcm: bytes, frame_bytes: int = FRAME_BYTES) -> npt.NDArray[np.float64]:
    """Each frame's level in dBFS."""
    x = np.frombuffer(pcm[: len(pcm) // frame_bytes * frame_bytes], dtype="<i2")
    if not len(x):
        return np.zeros(0)
    power = (x.astype(np.float64).reshape(-1, frame_bytes // 2) / 32768) ** 2
    levels: npt.NDArray[np.float64] = 10 * np.log10(power.mean(axis=1) + 1e-12)
    return levels


def loud_level(pcm: bytes, rate: int) -> float:
    """The 95th percentile of the 40 ms blocks above -60 dBFS (-120 for none)."""
    levels = frame_levels(pcm, int(rate * 0.04) * 2)
    voiced = levels[levels > -60]
    return float(np.percentile(voiced, 95)) if len(voiced) else -120.0


@dataclass
class Sound:
    """A loud moment inside a window (event seconds), what Whisper made of it, and the verdict:
    the person's speech or not."""

    start: float
    end: float
    peak_db: float
    text: str = ""
    no_speech: float = 1.0
    logprob: float = -9.0
    speech: bool = False

    def row(self) -> dict[str, Any]:
        """As a JSON object."""
        return dict(vars(self))


def sound_events(
    levels: npt.NDArray[np.float64],
    t0: float,
    floor_db: float,
    *,
    over_db: float = 15.0,
    min_dbfs: float = -55.0,
) -> list[Sound]:
    """Runs of frames (the first at event second ``t0``) ``over_db`` above the room's floor and
    at least ``min_dbfs``, joined across gaps up to 0.25 s; two frames at least."""
    threshold = max(floor_db + over_db, min_dbfs)
    runs: list[Sound] = []
    for i, db in enumerate(levels):
        if db <= threshold:
            continue
        t = t0 + i * FRAME_S
        if runs and t - runs[-1].end <= 0.25:
            runs[-1].end = t + FRAME_S
            runs[-1].peak_db = max(runs[-1].peak_db, float(db))
        else:
            runs.append(Sound(t, t + FRAME_S, float(db)))
    return [r for r in runs if r.end - r.start >= 2 * FRAME_S - 1e-9]


def is_speech(sound: Sound, reply: str) -> bool:
    """The truth's rule: Whisper hears words (no-speech probability under 0.6, mean
    log-probability over -2.5: the canceller mangles a voice over the robot's, and Whisper
    reads "Пепин, покажи" as "Паркетин, по кази" at -2.1) over at least 0.15 s, and they take
    the turn (:func:`pepin.voice_cascade.barge_in.interrupting_words`); the robot's own words
    count as its echo only when quiet (under -35 dBFS: the person is louder)."""
    if sound.end - sound.start < 0.15 or sound.no_speech >= 0.6 or sound.logprob <= -2.5:
        return False
    said = words(sound.text)
    if not said or is_hallucination(sound.text):
        return False
    if sound.peak_db < -35.0 and own_echo(said, reply):
        return False
    return bool(interrupting_words(sound.text, ""))


Detailed = Callable[[bytes], tuple[str, float, float]]  # pcm -> (text, no_speech, logprob)


def ground_truth(
    recording: Recording, read: Detailed, extend_s: float = EXTEND_S
) -> dict[int, list[Sound]]:
    """Each window's loud moments, read and judged (:func:`is_speech`)."""
    truth: dict[int, list[Sound]] = {}
    floor = recording.floor_db()
    for w, end in spans(recording.windows, extend_s):
        sounds = sound_events(frame_levels(recording.pcm(w.start, end)), w.start, floor)
        for s in sounds:
            pcm = recording.pcm(max(w.start - 0.5, s.start - CONTEXT_S), s.end + CONTEXT_S)
            s.text, s.no_speech, s.logprob = read(pcm.ljust(RATE // 5 * 2, b"\x00"))
            s.speech = is_speech(s, w.reply)
        truth[w.turn] = sounds
    return truth


@dataclass(frozen=True)
class Trigger:
    """A replayed barge-in: when (event seconds) and the detector's reason."""

    t: float
    why: str


DetectorFactory = Callable[[Callable[[], str]], BargeInDetector]  # own_text -> a detector


def replay(
    recording: Recording,
    config: CascadeConfig,
    make: DetectorFactory,
    extend_s: float = EXTEND_S,
) -> dict[int, Trigger]:
    """The mic through the ears with a fresh detector (``make``, given the reply in play), the
    speaker playing through each window; each window's first barge-in."""
    plays = spans(recording.windows, extend_s)
    current: list[Window | None] = [None]
    mode: list[Playback] = ["silent"]
    detector = make(lambda: current[0].reply if current[0] is not None else "")
    ears = Ears(config.with_changes(barge_in=True), lambda: mode[0], detector=detector)
    triggers: dict[int, Trigger] = {}
    k = 0
    for i in range(len(recording.mic) // FRAME_BYTES):
        t = (i + 1) * FRAME_S  # the frame's end: when it arrived
        while k < len(plays) and plays[k][1] <= t:
            k += 1
        window = plays[k][0] if k < len(plays) and plays[k][0].start <= t else None
        current[0], mode[0] = window, "reply" if window is not None else "silent"
        for event in ears.feed(recording.mic[i * FRAME_BYTES : (i + 1) * FRAME_BYTES], t):
            if event.kind == "barge_in" and window is not None and window.turn not in triggers:
                triggers[window.turn] = Trigger(event.t, event.why or "")
    return triggers


@dataclass
class Verdict:
    """One window's result for one detector."""

    turn: int
    outcome: Outcome
    trigger: Trigger | None = None
    reaction_s: float | None = None  # caught: since the person's onset
    sound: Sound | None = None  # what the trigger fell on


def judge(window: Window, sounds: Sequence[Sound], trigger: Trigger | None) -> Verdict:
    """Caught, false, missed or quiet (see the module's notes)."""
    speech = [s for s in sounds if s.speech]
    if trigger is None:
        return Verdict(window.turn, "missed" if speech else "quiet")
    hit = sorted(
        (s for s in sounds if s.start - EARLY_S <= trigger.t <= s.end + LATE_S),
        key=lambda s: s.start,
    )
    voices = [s for s in hit if s.speech]
    if voices:
        return Verdict(window.turn, "caught", trigger, trigger.t - voices[-1].start, voices[-1])
    return Verdict(window.turn, "false", trigger, None, hit[-1] if hit else None)


@dataclass
class Tally:
    """Counts over windows: those the person spoke in (positives), caught, false, and the
    reactions (seconds)."""

    windows: int = 0
    positives: int = 0
    caught: int = 0
    false: int = 0
    reactions: list[float] = field(default_factory=list)

    def add(self, verdict: Verdict, positive: bool) -> None:
        """One window's verdict; ``positive``: the person spoke in it."""
        self.windows += 1
        self.positives += positive
        if verdict.outcome == "caught":
            self.caught += 1
            if verdict.reaction_s is not None:
                self.reactions.append(verdict.reaction_s)
        elif verdict.outcome == "false":
            self.false += 1

    def merge(self, other: Tally) -> None:
        """Add another tally's counts."""
        self.windows += other.windows
        self.positives += other.positives
        self.caught += other.caught
        self.false += other.false
        self.reactions += other.reactions

    def median_ms(self) -> float | None:
        """The median reaction in milliseconds (None without a catch)."""
        return statistics.median(self.reactions) * 1000 if self.reactions else None

    def cell(self) -> str:
        """``caught/positives  false  median ms``."""
        median = self.median_ms()
        ms = "-" if median is None else f"{median:.0f}"
        return f"{self.caught}/{self.positives} {self.false}F {ms}ms"


class Ready:
    """A :class:`pepin.voice_cascade.barge_in.Pending` known in advance, ready at a time."""

    def __init__(self, text: str, ready_s: float) -> None:
        """``ready_s``: the laptop time it may be read from."""
        self.text, self.ready_s = text, ready_s

    def result(self, t: float) -> str | None:
        """The words once ``t`` has reached ``ready_s``."""
        return self.text if t >= self.ready_s else None

    def cancel(self) -> None:
        """Nothing runs."""


class OfflineReader:
    """Whisper for a replay: the words at once, ready when the call would have answered (its
    measured time, or ``fixed_ms``)."""

    def __init__(
        self,
        transcribe: Callable[[bytes], str],
        fixed_ms: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """``transcribe``: the words in 16 kHz s16le PCM."""
        self.transcribe, self.fixed_ms, self.clock = transcribe, fixed_ms, clock
        self.ms: list[float] = []  # each call's own time

    def __call__(self, pcm: bytes, t: float) -> Ready:
        """See :data:`pepin.voice_cascade.barge_in.Reader`."""
        start = self.clock()
        text = self.transcribe(pcm)
        ms = (self.clock() - start) * 1000
        self.ms.append(ms)
        return Ready(text, t + (self.fixed_ms if self.fixed_ms is not None else ms) / 1000)
