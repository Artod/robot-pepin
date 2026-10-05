"""The gate in front of the paid session: speech cut by its loudness, transcribed on the Mac, and
matched for the robot's name (open) or a goodbye (close).

Nothing reaches the Live API until :func:`is_wake` accepts a local transcript: a segment of room
audio (:class:`Segmenter`) goes to Whisper on the Mac's GPU (:class:`MlxWhisper`, mlx-whisper),
and only a transcript holding "Пепин" (or what Whisper makes of it: Пепен, Пипин, Pepin) opens
the session. Inside a session the goodbye is read from Live's own input transcription
(:func:`is_goodbye`); Whisper runs while the robot is idle, and while it drives, when the mic is
withheld from Live and only speech naming the robot or saying stop (:func:`is_stop`) reaches it.
Whisper's hallucinations on room noise — the closing lines of the subtitled videos it learned
from (:func:`is_hallucination`) — are dropped before any decision.
"""

from __future__ import annotations

import math
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pepin.audio_link import FRAME_SAMPLES, RATE

_WAKE_CYRILLIC = re.compile("п[еэиы]п{1,2}[еэиы]н[\u0430-\u044f]{0,3}")
_WAKE_LATIN = re.compile("p[eiy]pp?[eiy]n[a-z]{0,2}")
_WORD = re.compile("[a-z\u0430-\u044f]+")
_NAME_CORES = ("пепин", "пипин", "пепен")
_NAME_ENDINGS = ("пин", "пен")
# real words, and what Whisper makes of them (Репин -> Крепин)
_NOT_THE_NAME = frozenset(("папин", "репин", "крепин", "лапин", "сапин", "чапин", "щипин"))
_GOODBYE_PHRASES = (
    ("все", "спасибо"),
    ("спасибо", "все"),
    ("до", "свидания"),
    ("до", "встречи"),
    ("отбой",),
    ("bye",),
    ("goodbye",),
)
_STOP_WORDS = frozenset(
    ("стоп", "стой", "стоять", "остановись", "остановитесь", "хватит", "отмена", "stop")
)
# Whisper's lines for silence and hum: subtitle credits and video outros from its training data.
# 2026-10-05, 327 segments of room noise, motors and speech (scratch/voice_1005): "Продолжение
# следует..." 121, "Спасибо." 60 (no one thanked the robot), "Субтитры сделал DimaTorzok" 52,
# "Аплодисменты." 12, "Смотрите продолжение в следующей серии." 11, a subtitle editor's and
# proofreader's credit 1, "Добро пожаловать в наш канал!" 1. A whole transcript that is one of
# these (or names the subtitles) carries no command: a bare "спасибо" names nobody.
_HALLUCINATIONS = frozenset(
    (
        "продолжение следует",
        "спасибо",
        "спасибо за внимание",
        "спасибо за просмотр",
        "аплодисменты",
        "смотрите продолжение в следующей серии",
        "добро пожаловать в наш канал",
        "подписывайтесь на канал",
        "до новых встреч",
        "музыка",
    )
)
_HALLUCINATION_MARKS = ("субтитр", "dimatorzok", "корректор")
_FAREWELL_FILLERS = frozenset(
    ("ну", "все", "спасибо", "давай", "ладно", "хорошо", "тогда", "хей", "эй", "ok", "окей")
)


def words(text: str) -> list[str]:
    """Lower-case words of ``text``, the letter yo read as ie, punctuation dropped."""
    return _WORD.findall(text.lower().replace("\u0451", "\u0435"))


def is_name(word: str) -> bool:
    """Whether one word is the robot's name as Whisper writes it. In noise the second syllable
    survives and the first does not (Степин, Кипин, Тупин, Чаепепин: scratch/voice_live), so a
    two-syllable word ending in пин/пен counts too, unless it is a real word (папин, Репин)."""
    if _WAKE_CYRILLIC.fullmatch(word) or _WAKE_LATIN.fullmatch(word):
        return True
    if any(core in word for core in _NAME_CORES):
        return True
    return 5 <= len(word) <= 7 and word.endswith(_NAME_ENDINGS) and word not in _NOT_THE_NAME


def is_wake(text: str) -> bool:
    """Whether a transcript calls the robot by name: Пепин and the ways Whisper hears it."""
    return any(is_name(word) for word in words(text))


def is_goodbye(text: str) -> bool:
    """Whether a user's turn ends the conversation: "пока" said as a farewell (beside nothing but
    the name and words like ну/спасибо/давай, not "пока ты едешь, ..."), "всё, спасибо",
    "до свидания"."""
    said = words(text)
    if not said:
        return False
    rest = [w for w in said if w != "пока" and w not in _FAREWELL_FILLERS and not is_name(w)]
    if "пока" in said and not rest:
        return True
    for phrase in _GOODBYE_PHRASES:
        n = len(phrase)
        if any(tuple(said[i : i + n]) == phrase for i in range(len(said) - n + 1)):
            return True
    return False


def is_stop(text: str) -> bool:
    """Whether a transcript tells the robot to stop: стоп, стой, хватит, остановись, отмена."""
    return any(word in _STOP_WORDS for word in words(text))


def is_hallucination(text: str) -> bool:
    """Whether a transcript is one of Whisper's lines for noise (subtitle credits, video
    outros, a bare "Спасибо."), not anything said in the room."""
    said = " ".join(words(text))
    return said in _HALLUCINATIONS or any(mark in said for mark in _HALLUCINATION_MARKS)


def rms_db(pcm: bytes) -> float:
    """The loudness of s16le PCM in dBFS."""
    n = len(pcm) // 2
    if not n:
        return -120.0
    samples = struct.unpack(f"<{n}h", pcm[: n * 2])
    mean = sum(s * s for s in samples) / n
    return 10 * math.log10(mean / 32768**2 + 1e-12)


@dataclass(frozen=True)
class Segment:
    """One stretch of speech: its PCM (16 kHz s16le, the lead-in included) and when its last
    voiced frame arrived (laptop monotonic seconds)."""

    pcm: bytes
    ended_s: float

    @property
    def duration_s(self) -> float:
        """How long the segment lasts."""
        return len(self.pcm) / 2 / RATE


class Segmenter:
    """Speech out of the frame stream by loudness: a segment starts when the level stands
    ``start_db`` above the room's floor for three frames and ends after ``silence_s`` below it
    (or at ``max_s``); the floor follows the quiet frames. The same gate as scripts/voice.py."""

    def __init__(
        self,
        *,
        start_db: float = 10.0,
        silence_s: float = 0.5,
        max_s: float = 8.0,
        pre_s: float = 0.3,
        min_s: float = 0.3,
        floor_db: float = -55.0,
    ) -> None:
        """Thresholds in dB over the floor and seconds."""
        self.start_db, self.silence_s, self.max_s, self.min_s = start_db, silence_s, max_s, min_s
        self._pre_frames = max(1, int(pre_s * RATE / FRAME_SAMPLES))
        self.floor = floor_db
        self._pre: list[bytes] = []
        self._voiced: list[bytes] = []
        self._loud_run = 0
        self._quiet = 0
        self._start = 0.0
        self.talking = False
        self.last_voice_s = 0.0  # when the last loud frame arrived

    def feed(self, pcm: bytes, now: float) -> Segment | None:
        """One frame at laptop time ``now``; the segment it completes, if any."""
        db = rms_db(pcm)
        loud = db > self.floor + self.start_db
        if loud:
            self.last_voice_s = now
        if not self.talking:
            if not loud:
                self.floor = 0.98 * self.floor + 0.02 * db
            self._pre.append(pcm)
            del self._pre[: -self._pre_frames]
            self._loud_run = self._loud_run + 1 if loud else 0
            if self._loud_run >= 3:
                self.talking, self._start, self._quiet = True, now, 0
                self._voiced = list(self._pre)
            return None
        self._voiced.append(pcm)
        self._quiet = 0 if loud else self._quiet + 1
        quiet_s = self._quiet * FRAME_SAMPLES / RATE
        if quiet_s < self.silence_s and now - self._start < self.max_s:
            return None
        segment = Segment(b"".join(self._voiced), now - quiet_s)
        self.reset()
        return segment if segment.duration_s >= self.min_s else None

    def reset(self) -> None:
        """Forget the speech in progress (the floor is kept)."""
        self.talking = False
        self._voiced, self._pre = [], []
        self._loud_run = self._quiet = 0


class Transcriber(Protocol):
    """Speech to text on the Mac."""

    def transcribe(self, pcm: bytes) -> str:
        """The words in 16 kHz s16le mono ``pcm``."""
        ...


class MlxWhisper:
    """Whisper on the Mac's GPU (mlx-whisper), Russian forced: Artem's English comes out
    transliterated, which is all the name needs. No initial prompt by default: a prompt holding
    the name lifts recall in noise but turns near misses into the name and can be echoed back
    out of a hiss. Loaded and warmed at construction."""

    def __init__(
        self,
        repo: str = "mlx-community/whisper-large-v3-turbo",
        language: str = "ru",
        initial_prompt: str | None = None,
    ) -> None:
        """``repo``: the Hugging Face repo of an MLX Whisper model (cached after the first use)."""
        import mlx_whisper  # lazy: macOS-only, and the unit tests never load a model
        import numpy as np

        self._np = np
        self._transcribe: Callable[..., dict[str, Any]] = mlx_whisper.transcribe
        self._kwargs = {"path_or_hf_repo": repo, "language": language}
        if initial_prompt:
            self._kwargs["initial_prompt"] = initial_prompt
        self.transcribe(b"\x00\x00" * (RATE // 2))

    def transcribe(self, pcm: bytes) -> str:
        """The words in ``pcm``."""
        audio = self._np.frombuffer(pcm, dtype=self._np.int16).astype(self._np.float32) / 32768
        return str(self._transcribe(audio, **self._kwargs).get("text", "")).strip()
