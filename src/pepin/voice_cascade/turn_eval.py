"""End-of-turn methods replayed on Artem's recorded utterances (``scripts/turn_eval.py``).

Clips. A conversation's record (:mod:`pepin.voice_cascade.log`, the mic from the open as
:class:`pepin.voice_cascade.barge_eval.Recording` reads it) gives each utterance the live gate
closed as a ``heard`` event: ``t`` when it was read, ``whisper_ms`` the read after the close,
``seconds`` its audio (the lead-in and the closing silence included), so its last loud frame is
at ``t - whisper_ms - silence`` (:func:`fragments_of`). Utterances the run joined (the next
opened within ``continuation_s`` of the close, as the loop joins them) are one turn
(:func:`turns_of`): the truth is the turn's last word. A clip (:func:`session_clips`) is a turn's
audio from its opening (or the end of the reply it cut into) to ``TAIL_S`` after its last word
(or the next reply), replayed with the speaker silent and the gate's floor set from the room.
Live cuts (WAVs padded 0.2 s, :func:`wav_clips`) join as clips whose truth is their last loud
frame, with the pad's own quiet looped after them.

Pauses. :func:`with_pause` stretches a clip at a boundary between two words (the transcriber's
token times, at the quietest frame just before the next word) by ``pause_s`` of the clip's own
quiet: the person stopping mid-sentence. The 2026-10-09 measurement inserted 0.25 s.

A replay (:func:`replay`) feeds a clip through :class:`pepin.voice_cascade.ears.Ears` with a
method's config: the transcriber's early read is real (the engine on the audio the ears hand it)
and lands ``read_ms`` after the ask on the replay's clock (:class:`TimedReader`); Smart Turn
answers inline (its ~9 ms fit in a frame). :class:`Tally` counts the closes before the truth
(early cuts; merged when the voice came back within ``continuation_s``, split otherwise) and
the close after the last word.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.voice_cascade.barge_eval import Recording, frame_levels, read_wav
from pepin.voice_cascade.barge_in import SpeechProbability, VadDetector
from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.ears import Ears
from pepin.voice_cascade.player import Playback
from pepin.voice_cascade.stt.runtime import samples
from pepin.voice_cascade.turn import TurnWorker
from pepin.voice_cascade.turn_text import Judgement, TextJudge

FRAME_S = FRAME_SAMPLES / RATE
FRAME_BYTES = FRAME_SAMPLES * 2
TAIL_S = 1.5  # a clip goes on this long after the last word (unless a reply starts sooner)
TOLERANCE_S = 0.2  # a close this close before the truth still ends the turn (a word's decay)
PAD_S = 0.2  # the live cuts' padding either side


class StreamVad(SpeechProbability, Protocol):
    """A voice-activity model whose stream can start again (Silero's state)."""

    def reset(self) -> None:
        """A new stream."""
        ...


@dataclass(frozen=True)
class Clip:
    """One turn to replay: its audio (16 kHz s16le), the room's floor the gate starts from,
    where the person's last word ends (clip seconds: the truth), what was said, and an
    inserted pause (where it starts, how long; 0: as spoken)."""

    name: str
    pcm: bytes
    floor_db: float
    end_s: float
    text: str
    source: str = ""
    pause_at_s: float | None = None
    pause_s: float = 0.0

    @property
    def seconds(self) -> float:
        """The clip's length."""
        return len(self.pcm) / 2 / RATE


@dataclass(frozen=True)
class Fragment:
    """An utterance the live gate closed (event seconds): its audio's start (the lead-in
    included), the gate's opening, its last loud frame, the close, and the words read."""

    start: float
    opened: float
    end: float
    closed: float
    text: str


def fragments_of(rows: Sequence[dict[str, Any]], silence_s: float) -> list[Fragment]:
    """The utterances of a conversation's events after the wake (noise lines dropped)."""
    heard = [r for r in rows if r["event"] == "heard"][1:]  # the first is the wake
    opens = [float(r["t"]) for r in rows if r["event"] == "start"]
    out = []
    for h in heard:
        if h.get("noise") or not str(h.get("text", "")).strip():
            continue
        closed = float(h["t"]) - float(h.get("whisper_ms", 0)) / 1000
        start = closed - float(h["seconds"])
        opened = next((t for t in opens if start <= t <= closed), start)
        out.append(Fragment(start, opened, closed - silence_s, closed, str(h["text"])))
    return out


def turns_of(fragments: Sequence[Fragment], continuation_s: float) -> list[list[Fragment]]:
    """Fragments joined as the loop joins them: one opening within ``continuation_s`` of the
    previous close continues its turn."""
    turns: list[list[Fragment]] = []
    for f in fragments:
        if turns and f.opened - turns[-1][-1].closed <= continuation_s:
            turns[-1].append(f)
        else:
            turns.append([f])
    return turns


def run_silence_s(rows: Sequence[dict[str, Any]]) -> float | None:
    """The run's own ``ears.silence_s`` from its turns' closes (None when an end-of-turn model
    closed any: then the close is no fixed silence after the last word)."""
    turns = [r for r in rows if r["event"] == "turn" and r.get("kind") == "person"]
    if any((r.get("eot") or {}).get("closed_by") not in (None, "silence") for r in turns):
        return None
    closed = [r["ms"]["closed"] for r in turns if (r.get("ms") or {}).get("closed") is not None]
    return statistics.median(closed) / 1000 if closed else None


def vad_detector(config: CascadeConfig, vad: SpeechProbability) -> VadDetector:
    """The run's voice-activity test (``barge_in.vad_*``) over ``vad``: the gate opens only on
    speech (``ears.vad_open``), as it did live. No echo guard: a replay's speaker is silent."""
    return VadDetector(
        vad,
        threshold=config.barge_in_vad_threshold,
        hold_s=config.barge_in_vad_hold_s,
        min_dbfs=config.barge_in_vad_min_dbfs,
    )


def gate_floors(rec: Recording, config: CascadeConfig, vad: StreamVad | None = None) -> list[float]:
    """The gate's floor at each frame of a conversation, as the run's gate followed the room:
    the whole mic through the ears, held shut while a reply played (the floor frozen there)."""
    windows = sorted((w.start, w.end) for w in rec.windows)
    mode: list[Playback] = ["silent"]
    detector = None
    if vad is not None:
        vad.reset()
        detector = vad_detector(config, vad)
    ears = Ears(config.with_changes(barge_in=False), lambda: mode[0], detector=detector)
    floors = []
    k = 0
    for i in range(len(rec.mic) // FRAME_BYTES):
        t = (i + 1) * FRAME_S
        while k < len(windows) and windows[k][1] <= t:
            k += 1
        mode[0] = "reply" if k < len(windows) and windows[k][0] <= t else "silent"
        ears.feed(rec.mic[i * FRAME_BYTES : (i + 1) * FRAME_BYTES], t)
        floors.append(ears.gate.floor)
    return floors


def speech_chunks(pcm: bytes, vad: StreamVad) -> list[float]:
    """The voice-activity model's p for each of its chunks of ``pcm`` (a fresh stream)."""
    vad.reset()
    x = samples(pcm)
    n = vad.chunk
    return [vad(x[i : i + n]) for i in range(0, len(x) - n + 1, n)]


def voice_end(
    pcm: bytes,
    floor_db: float,
    start_db: float,
    vad: StreamVad | None,
    limit_s: float,
    threshold: float = 0.5,
) -> float | None:
    """Where the voice ends (seconds): the end of the last frame up to ``limit_s`` that the gate
    calls loud and the voice-activity model calls speech (its chunk or the next: the model
    answers a little late), so room noise after the words moves no truth."""
    loud = frame_levels(pcm) > floor_db + start_db
    if vad is not None:
        p = speech_chunks(pcm, vad)
        span = vad.chunk / RATE
        for i in range(len(loud)):
            k = int(i * FRAME_S / span)
            loud[i] = loud[i] and max(p[k : k + 2], default=0.0) >= threshold
    last = [i for i in np.flatnonzero(loud) if (i + 1) * FRAME_S <= limit_s + 1e-9]
    return (int(last[-1]) + 1) * FRAME_S if last else None


def session_clips(
    directory: Path,
    config: CascadeConfig,
    vad: StreamVad | None = None,
    tail_s: float = TAIL_S,
    slack_s: float = 0.3,
) -> list[Clip]:
    """The person's turns of one ``cascade_*`` conversation as clips (none for a run whose
    turns an end-of-turn model closed). The truth is :func:`voice_end` up to ``slack_s`` past
    the record's last word (frames reach the laptop in bursts, so the record's arithmetic is
    a frame or ten off)."""
    rows = [json.loads(line) for line in (directory / "events.jsonl").open()]
    silence_s = run_silence_s(rows)
    if silence_s is None:
        return []
    rec = Recording.load(directory)
    floors = gate_floors(rec, config, vad)
    turns = turns_of(fragments_of(rows, silence_s), config.continuation_s)
    clips = []
    for i, turn in enumerate(turns):
        first, last = turn[0], turn[-1]
        before = [w.end for w in rec.windows if w.start < first.opened]
        start = max([first.start, *(end + FRAME_S for end in before)])
        later = [w.start for w in rec.windows if w.start > last.end]
        if i + 1 < len(turns):
            later.append(turns[i + 1][0].start)
        stop = min([last.end + tail_s, *(s - FRAME_S for s in later), rec.seconds])
        a = round(start / FRAME_S) * FRAME_BYTES
        b = round(stop / FRAME_S) * FRAME_BYTES
        if b - a < FRAME_BYTES * 10:
            continue
        pcm = rec.mic[a:b]
        floor = floors[max(0, min(len(floors), a // FRAME_BYTES) - 1)]  # the gate's, at the start
        record_end = last.end - a / 2 / RATE
        end = voice_end(pcm, floor, config.start_db, vad, record_end + slack_s)
        if end is None:
            continue
        text = " ".join(f.text for f in turn)
        name = f"{directory.parent.name}/{directory.name}#{i + 1}"
        clips.append(Clip(name, pcm, floor, end, text, "cascade"))
    return clips


def wav_clips(
    directory: Path,
    config: CascadeConfig,
    vad: StreamVad | None = None,
    tail_s: float = TAIL_S,
    pad_s: float = PAD_S,
) -> list[Clip]:
    """Live cuts (``wavs/*.wav`` padded ``pad_s`` each side, texts in ``out/manifest.json``):
    the floor from the pads, ``tail_s`` more of the pads' quiet after each, the truth
    :func:`voice_end`."""
    manifest = {m["file"]: m for m in json.loads((directory / "out" / "manifest.json").read_text())}
    clips = []
    for path in sorted((directory / "wavs").glob("*.wav")):
        pcm, rate = read_wav(path)
        if rate != RATE:
            continue
        pcm = pcm[: len(pcm) // FRAME_BYTES * FRAME_BYTES]
        pad = int(pad_s / FRAME_S) * FRAME_BYTES
        pads = pcm[:pad] + pcm[-pad:]
        levels = frame_levels(pads)
        floor = max(float(np.median(levels)) if len(levels) else -120.0, config.floor_min_dbfs)
        pcm += loop_to(quiet_frames(pads, floor + config.start_db), tail_s)
        end = voice_end(pcm, floor, config.start_db, vad, len(pcm) / 2 / RATE)
        if end is None:
            continue
        text = str(manifest.get(path.name, {}).get("text", ""))
        clips.append(Clip(f"live/{path.stem}", pcm, floor, end, text, "live"))
    return clips


def quiet_frames(pcm: bytes, threshold_db: float) -> bytes:
    """The frames of ``pcm`` under ``threshold_db`` (the room between words)."""
    levels = frame_levels(pcm)
    return b"".join(
        pcm[i * FRAME_BYTES : (i + 1) * FRAME_BYTES]
        for i, db in enumerate(levels)
        if db < threshold_db
    )


def loop_to(frames: bytes, seconds: float) -> bytes:
    """``frames`` repeated to ``seconds`` (digital silence when there are none)."""
    n = round(seconds / FRAME_S) * FRAME_BYTES
    if not frames:
        return bytes(n)
    return (frames * (n // len(frames) + 1))[:n]


Word = tuple[str, float]  # a word and its start (clip seconds)


class Aligner(Protocol):
    """Words with their start times, and the plain transcript (one engine for both)."""

    def words(self, pcm: bytes) -> list[Word]:
        """The words of ``pcm`` and where each starts."""
        ...

    def transcribe(self, pcm: bytes) -> str:
        """The words of ``pcm`` as the cascade's transcriber writes them."""
        ...


def boundaries(
    clip: Clip, words: Sequence[Word], per_clip: int = 2, margin_s: float = 0.25
) -> list[float]:
    """Up to ``per_clip`` places for a pause, spread over the clip: before a word but the
    first, ``margin_s`` clear of the first word's start and the last word's end, at the
    quietest frame of the 0.12 s before the word."""
    if len(words) < 2:
        return []
    first = words[0][1]
    starts = [t for _, t in words[1:] if first + margin_s <= t <= clip.end_s - margin_s]
    if not starts:
        return []
    if len(starts) > per_clip:
        n = len(starts)
        starts = [
            starts[min(n - 1, round((k + 1) * n / (per_clip + 1)) - 1)] for k in range(per_clip)
        ]
        starts = sorted(set(starts))
    levels = frame_levels(clip.pcm)
    out = []
    for t in starts:
        lo = max(0, round((t - 0.12) / FRAME_S))
        hi = min(len(levels), round((t + 0.02) / FRAME_S) + 1)
        if hi <= lo:
            continue
        out.append((lo + int(np.argmin(levels[lo:hi]))) * FRAME_S)
    return out


def with_pause(clip: Clip, at_s: float, pause_s: float, start_db: float) -> Clip:
    """``clip`` with ``pause_s`` of its own quiet (the frames after its last word that the gate
    counts as quiet) inserted at ``at_s``; the truth moves with it."""
    at = round(at_s / FRAME_S) * FRAME_BYTES
    after_end = clip.pcm[round((clip.end_s + 0.1) / FRAME_S) * FRAME_BYTES :]
    quiet = loop_to(quiet_frames(after_end, clip.floor_db + start_db), pause_s)
    pcm = clip.pcm[:at] + quiet + clip.pcm[at:]
    name = f"{clip.name}@{at_s:.2f}+{pause_s:.2f}"
    return replace(
        clip, name=name, pcm=pcm, end_s=clip.end_s + pause_s, pause_at_s=at_s, pause_s=pause_s
    )


class TimedReader:
    """The transcriber's early read for a replay: the words read at once (cached by their
    audio) and handed over ``read_ms`` after the ask on the replay's clock (:meth:`advance`).
    ``log`` keeps each read: (clip seconds asked, its text)."""

    def __init__(self, transcribe: Callable[[bytes], str], read_ms: float) -> None:
        """``transcribe``: the engine."""
        self.transcribe, self.read_ms = transcribe, read_ms
        self.now = 0.0
        self.cache: dict[str, str] = {}
        self.log: list[tuple[float, str]] = []
        self._due: list[tuple[float, concurrent.futures.Future[str], str]] = []

    def reset(self) -> None:
        """A new clip: the clock at 0, nothing in flight, the log empty."""
        self.now = 0.0
        self._due = []
        self.log = []

    def __call__(self, pcm: bytes) -> concurrent.futures.Future[str]:
        """See :data:`pepin.voice_cascade.ears.EarlyReader`."""
        key = hashlib.sha1(pcm).hexdigest()
        text = self.cache.get(key)
        if text is None:
            text = self.cache[key] = self.transcribe(pcm)
        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        self._due.append((self.now + self.read_ms / 1000, future, text))
        self.log.append((self.now, text))
        return future

    def advance(self, t: float) -> None:
        """The replay's clock at ``t``: the reads due by then land (their judges run)."""
        self.now = t
        due = [d for d in self._due if d[0] <= t + 1e-9]
        self._due = [d for d in self._due if d[0] > t + 1e-9]
        for _, future, text in due:
            if not future.cancelled():
                future.set_result(text)


@dataclass(frozen=True)
class Close:
    """A segment of a replay (clip seconds): when it closed, its last loud frame, what closed
    it, the verdict on its words and the sound's p."""

    t: float
    ended_s: float
    closed_by: str | None = None
    verdict: str | None = None
    why: str | None = None
    p: float | None = None


@dataclass
class Result:
    """One clip's replay: its closes and the openings of the gate."""

    clip: Clip
    closes: list[Close]
    opens: list[float]
    reads: list[tuple[float, str]] = field(default_factory=list)

    def cuts(self) -> list[Close]:
        """The closes before the person's last word."""
        return [c for c in self.closes if c.ended_s < self.clip.end_s - TOLERANCE_S]

    def merged(self, cut: Close, continuation_s: float) -> bool:
        """Whether the voice came back soon enough after ``cut`` for the loop to join it."""
        back = [t for t in self.opens if t > cut.t]
        return bool(back) and back[0] - cut.t <= continuation_s + 1e-9

    def final(self) -> Close | None:
        """The close of the turn: the first at or after its last word (None: never closed)."""
        return next((c for c in self.closes if c.ended_s >= self.clip.end_s - TOLERANCE_S), None)

    def close_ms(self) -> float | None:
        """The close after the last word, in milliseconds."""
        final = self.final()
        return None if final is None else (final.t - self.clip.end_s) * 1000


def replay(
    clip: Clip,
    config: CascadeConfig,
    *,
    reader: TimedReader | None = None,
    turn: TurnWorker | None = None,
    judge: TextJudge | None = None,
    vad: StreamVad | None = None,
) -> Result:
    """``clip`` through fresh ears (the speaker silent, the gate's floor at the clip's room, the
    opening asked of ``vad`` as live); ``turn`` an inline worker of its own (its answers are
    matched by request number)."""
    if reader is not None:
        reader.reset()
    early = reader if reader is not None and config.early_s > 0 else None
    detector = None
    if vad is not None:
        vad.reset()
        detector = vad_detector(config, vad)
    ears = Ears(config, lambda: "silent", turn, early, detector, judge)
    ears.gate.floor = max(clip.floor_db, config.floor_min_dbfs)
    closes: list[Close] = []
    opens: list[float] = []
    for i in range(len(clip.pcm) // FRAME_BYTES):
        t = (i + 1) * FRAME_S  # the frame's end: when it arrived
        if reader is not None:
            reader.advance(t)
        for event in ears.feed(clip.pcm[i * FRAME_BYTES : (i + 1) * FRAME_BYTES], t):
            if event.kind == "start":
                opens.append(event.t)
            elif event.kind == "segment" and event.segment is not None:
                closes.append(
                    Close(event.t, event.segment.ended_s, event.closed_by, event.verdict,
                          event.verdict_why, event.p)
                )  # fmt: skip
    reads = list(reader.log) if reader is not None else []
    return Result(clip, closes, opens, reads)


@dataclass
class Tally:
    """One method over a set of clips: clips, those cut, cuts (merged by the loop or split),
    the closes after the last word (ms), what closed the turns, clips never closed."""

    clips: int = 0
    cut_clips: int = 0
    cuts: int = 0
    merged: int = 0
    close_ms: list[float] = field(default_factory=list)
    closed_by: dict[str, int] = field(default_factory=dict)
    unclosed: int = 0

    def add(self, result: Result, continuation_s: float) -> None:
        """One clip's replay."""
        self.clips += 1
        cuts = result.cuts()
        self.cut_clips += bool(cuts)
        self.cuts += len(cuts)
        self.merged += sum(result.merged(c, continuation_s) for c in cuts)
        ms = result.close_ms()
        if ms is None:
            self.unclosed += 1
            return
        self.close_ms.append(ms)
        final = result.final()
        key = (final.closed_by if final is not None else None) or "silence"
        self.closed_by[key] = self.closed_by.get(key, 0) + 1

    def row(self) -> str:
        """``cut clips/clips (cuts, merged) | close p50 p90 mean | closed by``."""
        ms = sorted(self.close_ms)
        p50 = statistics.median(ms) if ms else float("nan")
        p90 = ms[min(len(ms) - 1, int(0.9 * len(ms)))] if ms else float("nan")
        mean = statistics.fmean(ms) if ms else float("nan")
        by = " ".join(f"{k} {v}" for k, v in sorted(self.closed_by.items()))
        split = self.cuts - self.merged
        extra = f" unclosed {self.unclosed}" if self.unclosed else ""
        return (
            f"{self.cut_clips:>3}/{self.clips:<3} cut ({self.cuts} cuts: {self.merged} merged,"
            f" {split} split) | close {p50:4.0f} {p90:4.0f} {mean:4.0f} ms | {by}{extra}"
        )


@dataclass(frozen=True)
class Method:
    """A row of the table: a label and the config changes that make it."""

    label: str
    changes: dict[str, Any]
    text: str | None = None  # two-gates' judge: "rules" or a TEXT_MODELS name


METHODS = (
    Method("silence 0.40", {"silence_s": 0.4}),
    Method("silence 0.25", {"silence_s": 0.25}),
    Method("smart-turn 0.50", {"turn_model": "smart-turn-v3.2", "turn_threshold": 0.5}),
    Method("smart-turn 0.85", {"turn_model": "smart-turn-v3.2", "turn_threshold": 0.85}),
    Method("two-gates rules", {"turn_model": "two-gates"}, "rules"),
    Method("two-gates rules 0.30", {"turn_model": "two-gates", "turn_early_close_s": 0.3}, "rules"),
    Method(
        "two-gates rules+st 0.5",
        {"turn_model": "two-gates", "turn_acoustic": "smart-turn-v3.2", "turn_threshold": 0.5},
        "rules",
    ),
    Method(
        "two-gates rules+st 0.85",
        {"turn_model": "two-gates", "turn_acoustic": "smart-turn-v3.2", "turn_threshold": 0.85},
        "rules",
    ),
    Method("two-gates rules+namo", {"turn_model": "two-gates"}, "namo-ru"),
)


def uses_smart_turn(config: CascadeConfig) -> bool:
    """Whether a method's config asks Smart Turn (alone or as two-gates' sound)."""
    if config.turn_model == "two-gates":
        return config.turn_acoustic != "pause"
    return config.turn_model is not None


@dataclass(frozen=True)
class TextRow:
    """A judged early read: where it fell (``end``: after the last word; ``pause``: in an
    inserted pause), the text and each judge's verdict."""

    clip: str
    where: str
    text: str
    verdicts: dict[str, Judgement]


def text_rows(results: Sequence[Result], judges: dict[str, TextJudge]) -> list[TextRow]:
    """The early reads of replays, placed against their clips' truth and judged by each
    judge: the read of each inserted pause, and the first read after the last word."""
    out = []
    for r in results:
        clip = r.clip
        at = clip.pause_at_s
        for t, text in r.reads:
            if at is not None and at <= t <= at + clip.pause_s:
                where = "pause"
            elif t >= clip.end_s:
                where = "end"
            else:
                continue
            out.append(
                TextRow(clip.name, where, text, {n: j.judge(text) for n, j in judges.items()})
            )
    seen: set[tuple[str, str]] = set()
    unique = []
    for row in out:  # one end read a clip (the first), every pause read
        key = (row.clip, row.where)
        if row.where == "end" and key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def text_table(rows: Sequence[TextRow], judges: Sequence[str]) -> list[str]:
    """Per judge: at the true end complete / unsure / incomplete, in a mid-sentence pause the
    same (complete there is the risk of a cut, incomplete at the end the cost of a wait)."""
    lines = [f"{'judge':<10}{'at the end: c/u/i':>22}{'in a pause: c/u/i':>22}"]
    for name in judges:
        cells = []
        for where in ("end", "pause"):
            v = [r.verdicts[name].verdict for r in rows if r.where == where]
            counts = "/".join(str(v.count(k)) for k in ("complete", "unsure", "incomplete"))
            cells.append(f"{counts} of {len(v)}")
        lines.append(f"{name:<10}{cells[0]:>22}{cells[1]:>22}")
    return lines
