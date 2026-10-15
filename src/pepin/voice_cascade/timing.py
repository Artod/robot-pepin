"""Words timed from a speech-to-text model's emissions, their vowels measured from the sound,
and the vowels a person drew out ("приииинтер" against "принтееер").

A CTC or transducer model emits each token at one encoder frame (GigaAM v3: 40 ms, 10 ms
features subsampled 4x), and greedy decoding keeps the frame where each token first shows. A
unit (a letter of a character model, a token of a BPE model) starts at its emission and ends at
the next unit's emission, or earlier where the sound falls quiet (a pause, a stop consonant's
closure); spaces and punctuation are emitted too, but they are not sounds and end nothing. Those
spans place the words well, not the letters: the models fire the letters of a drawn-out syllable
early and then wait, so the stretch lands on a later letter (scratch/gigaam_timing_1012).

So a word's vowels are measured from the sound: inside the word's span, the runs of steps that
Praat hears as voiced and that are within ``NUCLEUS_DB`` of the word's loudest step (a vowel,
with the sonorants around it), each labelled with the vowel letter (a BPE token with one vowel)
whose emission is nearest its start.

The stretch: a vowel's seconds against this speaker's usual vowel in the same place (a running
median; the last vowel of a phrase apart, as speech lengthens those). A vowel at least ``ratio``
times its usual and ``min_s`` long is stretched, and the tag spells the word the way people
write one, the letter repeated: ``приииинтер``.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from pepin.audio_link import RATE
from pepin.voice_cascade.stages import TimedWord, Unit, Vowel

Emission = tuple[str, float, float | None]  # a token's text, its frame's second, its log-prob
Flags = npt.NDArray[np.bool_]

VOWELS = frozenset("аеёиоуыэюяaeiouy")  # noqa: RUF001 -- Cyrillic and Latin vowels
FRAME_S = 0.04  # GigaAM v3's encoder frame: the shortest a unit can be
HOP_S = 0.01  # the loudness and voicing tracks' step
WINDOW_S = 0.02  # the loudness window
QUIET_DB = 20.0  # a step this far under the utterance's loud steps (95th percentile) is quiet
NUCLEUS_DB = 10.0  # a vowel's steps are within this of the word's loudest step
GAP_S = 0.03  # a vowel ends at an unvoiced or quieter run this long
MIN_VOWEL_S = 0.03
PAUSE_S = 0.15  # a word followed by this much silence ends a phrase: its last vowel is final
MAX_REPEAT = 5  # a stretched letter is written at most this many times
PITCH_FLOOR_HZ, PITCH_CEILING_HZ = 70.0, 400.0


def loudness(pcm: bytes) -> npt.NDArray[np.float64]:
    """The level of 16 kHz s16le mono ``pcm`` in dB, one value per ``HOP_S`` over ``WINDOW_S``
    windows (step ``i`` centred at ``i * HOP_S + WINDOW_S / 2``)."""
    x = _samples(pcm)
    hop, window = round(RATE * HOP_S), round(RATE * WINDOW_S)
    if len(x) < window:
        return np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(x, window)[::hop]
    return np.asarray(10 * np.log10(np.mean(frames**2, axis=1) + 1e-12), dtype=np.float64)


def voicing(pcm: bytes) -> Flags:
    """Which steps of ``pcm`` (:func:`loudness`'s) Praat hears as voiced; ~2 ms for 7 s."""
    import parselmouth

    x = _samples(pcm)
    steps = max(0, (len(x) - round(RATE * WINDOW_S)) // round(RATE * HOP_S) + 1)
    out = np.zeros(steps, dtype=bool)
    if len(x) < RATE // 10 or steps == 0:
        return out
    pitch = parselmouth.Sound(x, sampling_frequency=RATE).to_pitch_ac(
        time_step=HOP_S, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ
    )
    f0 = pitch.selected_array["frequency"]
    times = pitch.xs()
    if len(f0) == 0:
        return out
    grid = np.arange(steps) * HOP_S + WINDOW_S / 2
    index = np.clip(np.round((grid - times[0]) / HOP_S).astype(int), 0, len(f0) - 1)
    inside = (grid >= times[0] - HOP_S) & (grid <= times[-1] + HOP_S)
    return np.asarray((f0[index] > 0) & inside, dtype=bool)


def _samples(pcm: bytes) -> npt.NDArray[np.float64]:
    return np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float64) / 32768


def _step(t: float) -> int:
    return max(0, math.ceil((t - WINDOW_S / 2) / HOP_S - 1e-9))


def _time(i: int) -> float:
    return i * HOP_S + WINDOW_S / 2


def _quiet_after(quiet: Flags, start: float, until: float) -> float:
    """The first quiet second at or after ``start`` and before ``until``; ``until`` if none."""
    for i in range(_step(start), len(quiet)):
        if _time(i) >= until:
            break
        if quiet[i]:
            return _time(i)
    return until


def _letters(text: str) -> str:
    return "".join(c for c in text if c.isalpha())


def is_vowel(text: str) -> bool:
    """Whether a unit is one vowel letter."""
    return len(text) == 1 and text.lower() in VOWELS


def words_of(
    emissions: Sequence[Emission],
    pcm: bytes,
    *,
    letters: bool,
    voiced: Flags | None = None,
) -> tuple[TimedWord, ...]:
    """The words of a model's ``emissions`` over ``pcm`` (``letters``: a character model, one
    letter per emission; else BPE tokens, a leading space opening a word), each unit timed as
    the module says; with ``voiced`` (:func:`voicing` of ``pcm``) each word's vowels too. A word
    keeps its letters only; its log-prob is the sum of its emissions' (None if one is missing)."""
    level = loudness(pcm)
    quiet = (
        level < float(np.percentile(level, 95)) - QUIET_DB
        if len(level)
        else np.zeros(0, dtype=bool)
    )
    end_s = len(pcm) / 2 / RATE
    until: dict[int, float] = {}  # each sounding emission lasts at most until the next one
    following = end_s
    for i in range(len(emissions) - 1, -1, -1):
        if _letters(emissions[i][0]):
            until[i] = max(following, emissions[i][1])
            following = emissions[i][1]
    groups: list[list[int]] = []
    for i, (text, _, _) in enumerate(emissions):
        if not text.strip():  # a separator: the next token opens a word
            groups.append([])
            continue
        if not groups or (text[0].isspace() and groups[-1]):
            groups.append([])
        groups[-1].append(i)
    out: list[TimedWord] = []
    for group in groups:
        units: list[Unit] = []
        for i in group:
            text, start, _ = emissions[i]
            if i not in until:
                continue  # punctuation: not a sound
            end = _quiet_after(quiet, start + FRAME_S, until[i])
            units.append(Unit(text.strip(), start, min(end, until[i])))
        if not units:
            continue
        logprobs = [emissions[i][2] for i in group]
        known = [lp for lp in logprobs if lp is not None]
        logprob = float(sum(known)) if len(known) == len(logprobs) else None
        joined = "".join(emissions[i][0] for i in group)
        text = _letters(joined) if letters else _word(joined)
        out.append(TimedWord(text, units[0].start, units[-1].end, tuple(units), letters, logprob))
    if voiced is None:
        return tuple(out)
    return tuple(
        replace(word, vowels=vowels_of(word, level, voiced, out[n - 1].end if n else 0.0))
        for n, word in enumerate(out)
    )


def _word(joined: str) -> str:
    """A BPE word's text: its tokens joined, without the punctuation around it."""
    alpha = [i for i, c in enumerate(joined) if c.isalpha()]
    return joined[alpha[0] : alpha[-1] + 1] if alpha else ""


def vowels_of(
    word: TimedWord, level: npt.NDArray[np.float64], voiced: Flags, before: float
) -> tuple[Vowel, ...]:
    """The vowels of ``word`` as the sound has them: runs of voiced steps within ``NUCLEUS_DB``
    of its loudest, from one frame before its first emission (not before ``before``, the word
    before it's end) to its end; each labelled with the vowel letter nearest its start."""
    lo = _step(max(before, word.start - FRAME_S))
    hi = min(len(level), len(voiced), _step(word.end))
    if hi <= lo:
        return ()
    top = float(np.max(level[lo:hi]))
    ok = voiced[lo:hi] & (level[lo:hi] >= top - NUCLEUS_DB)
    gap, shortest = round(GAP_S / HOP_S), round(MIN_VOWEL_S / HOP_S)
    runs: list[tuple[int, int]] = []
    first = last = -1
    for i, good in enumerate(ok.tolist()):
        if not good:
            continue
        if first >= 0 and i - last > gap:
            runs.append((first, last))
            first = -1
        if first < 0:
            first = i
        last = i
    if first >= 0:
        runs.append((first, last))
    letters = [i for i, u in enumerate(word.units) if _vowel_count(u.text) == 1]
    out = []
    for a, b in runs:
        if b - a + 1 < shortest:
            continue
        start, end = _time(lo + a) - HOP_S / 2, _time(lo + b) + HOP_S / 2
        unit = min(letters, key=lambda i: abs(word.units[i].start - start), default=None)
        out.append(Vowel(round(start, 3), round(end, 3), unit))
    return tuple(out)


def _vowel_count(text: str) -> int:
    return sum(c.lower() in VOWELS for c in text)


def kinds(words: Sequence[TimedWord]) -> list[list[str]]:
    """Each word's vowels' kind: ``final`` (the last vowel of a word that ends the utterance or
    is followed by ``PAUSE_S`` of silence) or ``vowel``."""
    out: list[list[str]] = []
    for n, word in enumerate(words):
        row = ["vowel"] * len(word.vowels)
        later = words[n + 1] if n + 1 < len(words) else None
        if row and (later is None or later.start - word.end >= PAUSE_S - 1e-9):
            row[-1] = "final"
        out.append(row)
    return out


def samples(words: Sequence[TimedWord]) -> list[tuple[str, float]]:
    """The utterance's vowels as the baseline keeps them: (kind, seconds)."""
    return [
        (k, vowel.seconds)
        for word, row in zip(words, kinds(words), strict=True)
        for vowel, k in zip(word.vowels, row, strict=True)
    ]


def medians(baseline: Mapping[str, Sequence[float]], min_count: int) -> dict[str, float]:
    """Each kind's median seconds, for the kinds with at least ``min_count`` samples."""
    out = {}
    for k, seen in baseline.items():
        if len(seen) >= min_count and (median := statistics.median(seen)) > 0:
            out[k] = median
    return out


@dataclass(frozen=True)
class Stretch:
    """A word with a drawn-out vowel, its seconds and its ratio to the speaker's usual."""

    word: TimedWord
    vowel: Vowel
    ratio: float

    def spelled(self) -> str:
        """The word as people write a stretch: ``приииинтер``, the vowel's letter repeated by
        the ratio, 2 to ``MAX_REPEAT`` times (the word as it is when the letter is not known)."""
        index = self.vowel.unit
        if index is None:
            return self.word.text
        n = max(2, min(MAX_REPEAT, round(self.ratio)))
        parts = []
        for i, unit in enumerate(self.word.units):
            if i != index:
                parts.append(unit.text)
                continue
            j = next(k for k, c in enumerate(unit.text) if c.lower() in VOWELS)
            parts.append(unit.text[:j] + unit.text[j] * n + unit.text[j + 1 :])
        return "".join(parts)


def stretches(
    words: Sequence[TimedWord], usual: Mapping[str, float], *, ratio: float, min_s: float
) -> list[Stretch]:
    """The words of an utterance whose longest vowel took at least ``ratio`` times this
    speaker's usual vowel in its place (``usual``: :func:`medians` of ``vowel`` and ``final``)
    and at least ``min_s`` seconds."""
    out = []
    for word, row in zip(words, kinds(words), strict=True):
        best: Stretch | None = None
        for vowel, k in zip(word.vowels, row, strict=True):
            median = usual.get(k)
            if median is None or vowel.seconds < min_s:
                continue
            r = vowel.seconds / median
            if r >= ratio and (best is None or r > best.ratio):
                best = Stretch(word, vowel, r)
        if best is not None:
            out.append(best)
    return out


def log_rows(words: Sequence[TimedWord]) -> list[dict[str, object]]:
    """The words as the conversation log keeps them: the word, its span, its log-prob, and its
    vowels' seconds."""
    return [
        {
            "w": word.text,
            "t": [round(word.start, 2), round(word.end, 2)],
            "lp": None if word.logprob is None else round(word.logprob, 2),
            "v": [round(vowel.seconds, 2) for vowel in word.vowels],
        }
        for word in words
    ]
