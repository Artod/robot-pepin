"""Sounds: the registry of the sounds the robot may play, each cut's beat grid, the synthesized
default beat, and the contract between the board's player, the brain's tool and the gesture
engine's pulse.

THE REGISTRY (config/sounds.json, public; :mod:`pepin.overlay` merges a private tree's copy
over it by name: an entry of the same name replaces the public one whole, a new name is
appended): ``sounds`` is a list of entries, each a ``name`` (a mark's argument,
``{sound:default_beat}``: a short lower-case word), ``file`` (relative to ``sounds/`` in any
tree; gitignored here: no audio file is tracked in this repository, a private tree keeps its
own; null until the file is dropped in, and ``python -m pepin.overlay --check`` lists those),
``start_s`` (where the cut begins in the file), ``seconds`` (how much plays; the end is a hard
cut), ``description`` (at most six words: the brain's list), ``style`` (a word: the ``dance``
family of config/animation.json chooses its variant by it, the public book dances ``rock``; a
style the family lacks is no dance), ``beats`` (null, or the cut's grid: a recording's from
:func:`beat_grid`, a synthesized sound's by construction), ``external`` (fetched separately,
never on disk), ``synthesized`` (this module makes the file: :func:`sound_path` on first use,
``python -m pepin.sounds --make-default``) and a ``note``. The grid: ``bpm``, ``times_s`` (the
beats in the FILE's seconds, so a re-cut entry keeps it), ``strengths`` (each beat's onset over
the loudest of the span analysed, 0..1: analyse the cut and a few seconds round it, so a cut in
a quiet section is not weak against a loud chorus elsewhere) and ``method``. ``pulse`` holds
:class:`MusicKnobs`.

THE DEFAULT BEAT (:func:`default_beat`, the public registry's one sound, ``default_beat``): a
4-bar drum loop at 120 BPM, 8 s of 16 kHz mono made from numpy alone (no third-party audio in
the tree): a kick (a sine falling 110 -> 45 Hz under a 0.12 s decay, a 2 ms noise click on its
attack) on beats 1 and 3, a snare (high-passed noise under a 0.06 s decay over a 190 Hz sine's
0.04 s thump) on 2 and 4, each hit at its beat's accent, a closed hat (differenced noise,
0.012 s) every eighth, a fixed seed.
Its beats are every 0.5 s from 0 by construction (:func:`default_beat_grid`: the bar's first at
1.0, the third at 0.85, the others at 0.7: every one at ``pulse.strong`` or over), so the
registry carries the grid without analysis; :func:`beat_grid` finds the same beats on it.

THE BEATS (:func:`beat_grid`, numpy alone): the onset envelope is the spectral flux (the rise of
the compressed power in 32 log-spaced bands, frames every 10 ms (a whole number of samples: 9.98
ms at 22.05 kHz, and every time is counted in that real step), half-wave rectified, a half
second's local mean taken out; Bello 2005); the tempo is the envelope's autocorrelation over
40-220 BPM under a log-normal prior round 120 BPM, an octave wide; the beats are the envelope's
peaks on that period by dynamic programming (Ellis 2007: each frame scores its onset plus the
best earlier beat half to two periods back, less the squared log of the gap over the period),
the chain starting at the first real onset, tracked back from the best frame of the last two
periods; a beat's strength is the envelope's peak within 30 ms of it over the envelope's
highest, and the grid's leading and trailing beats under a tenth of the beats' median strength
are dropped (the tracker carries its period through a quiet intro or a fade: those are no
beats). One tempo a span (a song that changes tempo gets the dominant one: analyse the cut, not
the song). Good for a drum machine, a march, a hymn with a pulse; a rubato recording gets a grid
anyway (the method knows no better): listen before trusting it. MEASURED on synthetic kits
(kick, snare, hi-hats, 60 s; scratch/sounds/beats.py --kit): 60-140 BPM, swung or straight, F
1.00 within 70 ms, each beat 5-10 ms early; 170 and 200 BPM come out at half (85, 100: the
prior; a dance on every other beat); the strength does NOT mark the bar's first beat on a kit
(the snare's broadband rise outweighs the kick's: the snare's beats 0.9-1.0, the kick's
0.6-0.8, so at ``pulse.strong`` 0.6 every beat is strong); a minute of audio in 0.04-0.14 s on
this Mac.

THE CONTRACT (the board's player and the brain's tool land after the 2026-10-14 merge; their
shape is fixed here so the listener side is written against it, and :class:`FakeSoundPlayer`
plays it on a fake clock for tests):

* :class:`SoundPlayer` (the laptop's handle of the board's audio server): ``play(name)`` has the
  board mix the cut into its one output over the TTS, from ``start_s`` for ``seconds`` (the
  array's AEC cancels it from the mic as it cancels the voice: one output), and answers a
  :class:`Playing` once the board has scheduled it (``t0`` the engine-clock second, laptop
  monotonic, the cut's first sample reaches the speaker, the link's lag included; ``beats`` the
  cut's grid as seconds from ``t0`` with strengths: :meth:`Sound.cut_beats`, the one conversion
  from the file's seconds); ``stop(playing)`` cuts it; ``subscribe(listener)`` adds a
  :class:`BeatListener`.
* :class:`BeatListener`: ``sound_started(playing)`` as ``play`` answers; ``beat(t_beat,
  strength)`` for each beat of the cut, emitted ``pulse.lead_s`` before it sounds (``t_beat`` on
  the engine's clock; the lead covers the dance's slowest preparation, the raise to ``raised``,
  1.1 s before its beat, with the arm's latency); ``sound_ended(playing)`` at the cut's end or
  its stop (the dance's steps still to come or under way are let go with it).
* The engine's side: :class:`pepin.expression.listeners.MusicPulse` is a BeatListener that
  turns each beat into the book's ``dance`` with the sound's style as its argument, on the parts
  free for it (a beat at ``pulse.strong`` or over every part, a weaker one the head and the
  face), the way the voice's accents become ``pulse_weak`` and its kin; a bare ``{dance}`` of
  the brain's takes the style of the sound playing
  (:attr:`pepin.expression.animator.GestureEngine.arg_hooks`), else the family's first (rock).
* The brain's tool: ``sound(name)`` plays an entry (:meth:`Registry.lines` is its list); the
  mark ``{sound:NAME}`` is reserved for it: the vocabulary parses it, the engine logs it
  :data:`NO_PLAYER` until the tool lands.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

DANCE = "dance"  # the book's composition family the beats ask for (config/animation.json)
DEFAULT_STYLE = "rock"  # a bare {dance} with no sound playing: the family's first argument
DEFAULT_BEAT = "default_beat"  # the public registry's one sound, synthesized here
DEFAULT_BEAT_FILE = "default_beat.wav"  # under sounds/ (gitignored), made on demand
DEFAULT_BEAT_RATE = 16000
DEFAULT_BEAT_BPM = 120.0
DEFAULT_BEAT_BARS = 4  # of four beats: 8 s at 120 BPM
DEFAULT_BEAT_ACCENTS = (1.0, 0.7, 0.85, 0.7)  # each beat's strength in its bar
DEFAULT_BEAT_METHOD = "synthesized: a beat every 0.5 s from 0 (pepin.sounds.default_beat)"
SOUND = "sound"  # the reserved mark: {sound:NAME}, the brain's tool to come
SOUND_LINE = f"{SOUND}:NAME — a registered sound by name"  # its line in the brain's vocabulary
NO_PLAYER = "no player yet"  # what the engine logs for a {sound:NAME} until the tool lands
REGISTRY_FILE = "sounds.json"  # in config/
SOUNDS_DIR = "sounds"  # a file's root in any tree (pepin.overlay.SOUNDS)
HOP_S = 0.01  # the onset envelope's frame step
BANDS = 32  # of the spectral flux
LOCAL_S = 0.5  # the envelope's local mean taken out over this
TEMPO_RANGE_BPM = (40.0, 220.0)
PRIOR_BPM, PRIOR_OCTAVES = 120.0, 1.0  # the tempo prior's centre and width
TIGHTNESS = 100.0  # the beat tracker's cost of a gap off the period (Ellis 2007)
NEAR_S = 0.03  # a beat's strength: the envelope's peak within this of it
TRIM = 0.1  # a leading or trailing beat under this share of the beats' median strength is none
METHOD = "spectral flux 32 bands, 10 ms; autocorrelation tempo, 120 BPM prior; Ellis 2007 DP"
DESCRIPTION_WORDS = 6  # a description's most, for the brain's list


@dataclass(frozen=True)
class BeatGrid:
    """A file's beats: the tempo, each beat's second in the FILE and its strength (0..1), and
    how they were found."""

    bpm: float
    times_s: tuple[float, ...]
    strengths: tuple[float, ...]
    method: str = METHOD

    def cut(self, start_s: float, seconds: float) -> tuple[tuple[float, float], ...]:
        """The beats of the cut ``[start_s, start_s + seconds)`` as (seconds from the cut's
        start, strength)."""
        end = start_s + seconds
        return tuple((round(t - start_s, 4), s) for t, s in zip(self.times_s, self.strengths,
                                                                  strict=True)
                     if start_s <= t < end)  # fmt: skip

    def to_json(self) -> dict[str, Any]:
        """The registry's object."""
        return {
            "bpm": round(self.bpm, 2),
            "times_s": [round(t, 3) for t in self.times_s],
            "strengths": [round(s, 3) for s in self.strengths],
            "method": self.method,
        }


@dataclass(frozen=True)
class Sound:
    """One entry of the registry (the module's notes)."""

    name: str
    file: str | None = None
    start_s: float = 0.0
    seconds: float = 10.0
    description: str = ""
    style: str = DEFAULT_STYLE
    beats: BeatGrid | None = None
    note: str = ""
    external: bool = False  # the overlay's flag: the file is nowhere in the trees, on purpose
    synthesized: bool = False  # this module makes the file (:func:`make_synthesized`)

    def cut_beats(self) -> tuple[tuple[float, float], ...]:
        """The cut's beats as (seconds from its start, strength); none without a grid."""
        return self.beats.cut(self.start_s, self.seconds) if self.beats is not None else ()

    def line(self) -> str:
        """One line for the brain's list: the name, the description, the style and length."""
        return f"{self.name} — {self.description} ({self.style}, {self.seconds:g} s)"


@dataclass(frozen=True)
class MusicKnobs:
    """config/sounds.json's ``pulse``: the body on a sound's beats."""

    strong: float = 0.6  # a beat's strength from which every free part dances (under: the head)
    lead_s: float = 1.3  # the player emits a beat this long before it sounds
    lead_min_s: float = 0.25  # a beat nearer than this is skipped: no part prepares in time


@dataclass(frozen=True)
class Registry:
    """The sounds by name, in the file's order, and the pulse's knobs."""

    sounds: Mapping[str, Sound] = field(default_factory=dict)
    pulse: MusicKnobs = MusicKnobs()

    def get(self, name: str) -> Sound | None:
        """The entry, or None."""
        return self.sounds.get(name)

    def lines(self) -> tuple[str, ...]:
        """The brain's list: one line a sound (:meth:`Sound.line`), the ones with a file first."""
        ready = [s.line() for s in self.sounds.values() if s.file is not None or s.external]
        absent = [s.line() for s in self.sounds.values() if s.file is None and not s.external]
        return tuple(ready + absent)


def grid_from(data: Any, where: str = "beats") -> BeatGrid:
    """A registry entry's ``beats`` object as a grid; ``ValueError`` names the field."""
    if not isinstance(data, Mapping):
        raise ValueError(f"{where}: an object of bpm, times_s, strengths")
    times = data.get("times_s", [])
    strengths = data.get("strengths", [1.0] * len(times))
    if not isinstance(times, list) or not isinstance(strengths, list):
        raise ValueError(f"{where}.times_s and strengths: lists")
    if len(times) != len(strengths):
        raise ValueError(f"{where}: {len(times)} times_s and {len(strengths)} strengths")
    if any(t2 <= t1 for t1, t2 in itertools.pairwise(times)):
        raise ValueError(f"{where}.times_s: rising")
    return BeatGrid(float(data.get("bpm", 0.0)), tuple(float(t) for t in times),
                    tuple(min(1.0, max(0.0, float(s))) for s in strengths),
                    str(data.get("method", METHOD)))  # fmt: skip


def sound_from(entry: Mapping[str, Any], where: str = "sounds") -> Sound:
    """One entry of the registry's list as a :class:`Sound`; ``ValueError`` names the field."""
    name = entry.get("name")
    if not isinstance(name, str) or not name or name.lower() != name or " " in name:
        raise ValueError(f"{where}: name is a short lower-case word, got {name!r}")
    at = f"{where}[{name}]"
    file = entry.get("file")
    if file is not None and (not isinstance(file, str) or not file):
        raise ValueError(f"{at}.file: a path under {SOUNDS_DIR}/ or null")
    seconds = float(entry.get("seconds", Sound.seconds))
    start_s = float(entry.get("start_s", 0.0))
    if seconds <= 0.0 or start_s < 0.0:
        raise ValueError(f"{at}: start_s from 0, seconds above 0")
    description = str(entry.get("description", "")).strip()
    if len(description.split()) > DESCRIPTION_WORDS:
        raise ValueError(f"{at}.description: at most {DESCRIPTION_WORDS} words")
    style = str(entry.get("style", DEFAULT_STYLE)).lower()
    beats = entry.get("beats")
    return Sound(name, file, start_s, seconds, description, style,
                 None if beats is None else grid_from(beats, f"{at}.beats"),
                 str(entry.get("note", "")), bool(entry.get("external", False)),
                 bool(entry.get("synthesized", False)))  # fmt: skip


def knobs_from(data: Mapping[str, Any]) -> MusicKnobs:
    """The ``pulse`` block as knobs (defaults for what it lacks; a note is skipped)."""
    d = MusicKnobs()
    return MusicKnobs(float(data.get("strong", d.strong)), float(data.get("lead_s", d.lead_s)),
                      float(data.get("lead_min_s", d.lead_min_s)))  # fmt: skip


def registry_from(data: Mapping[str, Any]) -> Registry:
    """config/sounds.json's object (merged over the trees or not) as a registry; two entries of
    one name are a ``ValueError`` (the overlay merges by name before this sees them)."""
    raw = data.get("sounds", [])
    if not isinstance(raw, list):
        raise ValueError("sounds: a list of entries")
    sounds: dict[str, Sound] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ValueError("sounds: each entry an object")
        sound = sound_from(entry)
        if sound.name in sounds:
            raise ValueError(f"sounds: {sound.name!r} twice")
        sounds[sound.name] = sound
    pulse = data.get("pulse", {})
    return Registry(sounds, knobs_from(pulse if isinstance(pulse, Mapping) else {}))


def merge_sounds(base: Mapping[str, Any], top: Mapping[str, Any]) -> dict[str, Any]:
    """``top`` (a private tree's sounds.json) over ``base`` (the public one) the way the overlay
    merges them: an entry of the same name replaces the base's whole, a new one is appended, the
    other keys key by key (:func:`pepin.overlay.merge`)."""
    from pepin.overlay import merge

    merged: dict[str, Any] = merge(dict(base), dict(top))
    return merged


def load_registry(path: str | Path | None = None) -> Registry:
    """config/sounds.json wherever this library runs (merged over the private trees), or
    ``path`` (merged too when it lies in a tree, read as it is outside them)."""
    from pepin.overlay import read_json

    if path is None:
        from pepin.deployment import config_file

        path = config_file(REGISTRY_FILE)
    return registry_from(read_json(path))


def load_music_knobs() -> MusicKnobs:
    """The pulse's knobs of the registry in force."""
    return load_registry().pulse


def sound_path(sound: Sound, *, make: bool = True) -> Path | None:
    """Where ``sound``'s file is (the highest tree that has it under ``sounds/``), or None
    without a file or with one nowhere; a synthesized sound's file found nowhere is made first
    in the public tree's ``sounds/`` (:func:`make_synthesized`; ``make`` false: looked for
    alone)."""
    if sound.file is None:
        return None
    from pepin.overlay import resolve

    try:
        return resolve(f"{SOUNDS_DIR}/{sound.file}")
    except (FileNotFoundError, ValueError):
        pass
    return make_synthesized(sound.file) if make and sound.synthesized else None


def make_synthesized(file: str, root: str | Path | None = None) -> Path:
    """The synthesized sound ``file`` written under ``root``'s ``sounds/`` (the public tree's
    by default) and its path; ``ValueError`` for a file this module has no recipe for."""
    if file != DEFAULT_BEAT_FILE:
        raise ValueError(f"{file}: no recipe here (the synthesized sounds: {DEFAULT_BEAT_FILE})")
    if root is None:
        from pepin.overlay import public_tree

        root = public_tree()
    path = Path(root) / SOUNDS_DIR / file
    write_wav(path, default_beat(), DEFAULT_BEAT_RATE)
    return path


def write_beats(
    path: str | Path,
    name: str,
    grid: BeatGrid,
    file: str | None = None,
    base: Mapping[str, Any] | None = None,
) -> None:
    """``grid`` written into entry ``name`` of the registry file at ``path`` (that one file, not
    the merge: the private tree's copy when the file is private); an absent entry is appended
    with ``file``, a copy of ``base`` (the merged registry's entry of that name: the private
    copy's entry replaces the public one WHOLE by name, so it must carry the cut, the
    description and the style) or the defaults without one. The layout: two-space indents, the
    grid's lists on one line each."""
    target = Path(path)
    data = json.loads(target.read_text()) if target.is_file() else {}
    sounds = data.setdefault("sounds", [])
    entry = next((s for s in sounds if s.get("name") == name), None)
    if entry is None:
        entry = dict(base) if base is not None else {
            "name": name, "file": file, "start_s": 0.0, "seconds": Sound.seconds,
            "description": "", "style": DEFAULT_STYLE,
        }  # fmt: skip
        entry["name"] = name
        if file is not None:
            entry["file"] = file
        sounds.append(entry)
    elif file is not None:
        entry["file"] = file
    entry["beats"] = grid.to_json()
    text = json.dumps(data, indent=2, ensure_ascii=False)
    # the grid's lists back on one line each (json.dumps breaks every item onto its own)
    import re

    def inline(match: re.Match[str]) -> str:
        return "[" + ", ".join(x.strip() for x in match.group(1).split(",")) + "]"

    text = re.sub(r"\[\s*((?:-?[\d.]+,?\s*)+)\]", inline, text)
    target.write_text(text + "\n")


# -- the beats -------------------------------------------------------------------------------------


def read_audio(path: str | Path) -> tuple[Any, int]:
    """``path``'s samples (float, -1..1, mono: the channels averaged) and rate: through
    soundfile when it is installed (wav, flac, ogg, mp3), else the standard library's wave
    (16-bit PCM wav alone)."""
    import numpy as np

    try:
        import soundfile  # type: ignore[import-not-found,import-untyped,unused-ignore]
    except ImportError:
        soundfile = None
    if soundfile is not None:
        samples, rate = soundfile.read(str(path), dtype="float64", always_2d=True)
        return samples.mean(axis=1), int(rate)
    import wave

    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: 16-bit PCM wav alone without soundfile")
        rate = handle.getframerate()
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        samples = raw.reshape(-1, handle.getnchannels()).astype(np.float64) / 32768.0
    return samples.mean(axis=1), rate


def onset_envelope(samples: Any, rate: int, *, hop_s: float = HOP_S) -> Any:
    """The spectral flux every ``hop_s`` (the module's notes): a float array, 0 where nothing
    rises, one value a frame from the file's start."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if not np.all(np.isfinite(x)):
        raise ValueError("samples: NaN or infinite values (a broken decode?)")
    hop = max(1, round(rate * hop_s))
    win = 1 << (4 * hop - 1).bit_length()  # about 40 ms, a power of two
    padded = np.pad(x, (win // 2, win // 2 + hop))
    frames = np.lib.stride_tricks.sliding_window_view(padded, win)[::hop]
    power = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1)) ** 2
    freqs = np.fft.rfftfreq(win, 1.0 / rate)
    edges = np.geomspace(max(40.0, float(freqs[1])), rate / 2.0, BANDS + 1)
    bands = np.stack([power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
                      for lo, hi in itertools.pairwise(edges)], axis=1)  # fmt: skip
    top = float(bands.max()) or 1.0
    compressed = np.log1p(1e3 * bands / top)  # loud over quiet by ratio, a floor under silence
    flux = np.diff(compressed, axis=0).clip(min=0.0).sum(axis=1)
    envelope = np.concatenate([[0.0], flux])
    k = max(1, round(LOCAL_S / hop_s))
    if len(envelope) < k:  # shorter than the window: its one mean
        local = np.full_like(envelope, envelope.mean())
    else:
        local = np.convolve(envelope, np.ones(k) / k, mode="same")
    out: Any = np.clip(envelope - local, 0.0, None)
    return out


def tempo_bpm(envelope: Any, *, hop_s: float = HOP_S) -> float:
    """The envelope's tempo (the module's notes); the prior's centre for an envelope too short
    to hold a period."""
    import numpy as np

    o = np.asarray(envelope, dtype=np.float64)
    o = o - o.mean()
    n = len(o)
    lo_bpm, hi_bpm = TEMPO_RANGE_BPM
    lag_lo = max(1, round(60.0 / hi_bpm / hop_s))
    lag_hi = min(n - 2, round(60.0 / lo_bpm / hop_s))
    if lag_hi <= lag_lo or not np.any(o):  # too short, or flat (silence): no period in it
        return PRIOR_BPM
    lags = np.arange(lag_lo, lag_hi + 1)
    auto = np.array([float(np.dot(o[: n - lag], o[lag:])) / (n - lag) for lag in lags])
    bpm = 60.0 / (lags * hop_s)
    prior = np.exp(-0.5 * (np.log2(bpm / PRIOR_BPM) / PRIOR_OCTAVES) ** 2)
    best = int(np.argmax(auto * prior))
    return float(bpm[best])


def track_beats(envelope: Any, period_frames: float, *, tightness: float = TIGHTNESS) -> list[int]:
    """The beat frames on ``period_frames`` through ``envelope`` (Ellis 2007, the module's
    notes); an empty list for an empty envelope."""
    import numpy as np

    o = np.asarray(envelope, dtype=np.float64)
    n = len(o)
    if n == 0 or period_frames < 1.0:
        return []
    if float(o.max()) <= 0.0:
        return []  # silence: no beat worth the name
    norm = o / (float(o.std()) or 1.0)  # an onset counts in sigmas against the gap's cost
    quiet = 0.01 * float(norm.max())
    lo, hi = max(1, round(period_frames / 2)), max(2, round(period_frames * 2))
    offsets = np.arange(-hi, -lo + 1)
    penalty = -tightness * np.log(-offsets / period_frames) ** 2
    score = np.zeros(n)
    back = np.full(n, -1, dtype=int)
    first = True  # the chain starts at the first real onset, not in the silence before it
    for i in range(n):
        earlier = i + offsets
        valid = earlier >= 0
        # a predecessor before the start costs its gap alone (a score of 0 there)
        candidates = np.where(valid, score[np.maximum(earlier, 0)], 0.0) + penalty
        k = int(np.argmax(candidates))
        score[i] = norm[i] + candidates[k]
        if first:  # the silence before the first onset, and that onset itself: no link back
            first = norm[i] < quiet
            continue
        if valid[k]:
            back[i] = int(earlier[k])
    tail = max(0, n - hi)
    last = tail + int(np.argmax(score[tail:]))
    beats: list[int] = []
    while last >= 0:
        beats.append(last)
        last = back[last]
    beats.reverse()
    return beats


def frame_s(rate: int, hop_s: float = HOP_S) -> float:
    """The onset envelope's real frame step at ``rate``: ``hop_s`` rounded to whole samples (at
    22.05 kHz 220 samples, 9.977 ms: counted as 10 ms, a minute's beats would drift 0.14 s)."""
    return max(1, round(rate * hop_s)) / rate


def beat_grid(samples: Any, rate: int, *, hop_s: float = HOP_S, offset_s: float = 0.0) -> BeatGrid:
    """``samples``' beat grid (the module's notes): the tempo refined to the beats' median
    gap, the beats' seconds in the file (``offset_s``: where ``samples`` begin in it), their
    strengths over the envelope's peak, a quiet head's and tail's beats dropped."""
    import numpy as np

    step = frame_s(rate, hop_s)
    envelope = onset_envelope(samples, rate, hop_s=hop_s)
    bpm = tempo_bpm(envelope, hop_s=step)
    frames = track_beats(envelope, 60.0 / bpm / step)
    near = max(1, round(NEAR_S / step))
    top = float(envelope.max()) if len(envelope) else 0.0
    top = top or 1.0
    strengths = [float(envelope[max(0, f - near) : f + near + 1].max()) / top for f in frames]
    if strengths:
        floor = TRIM * float(np.median(strengths))
        first = next(i for i, s in enumerate(strengths) if s >= floor)
        last = max(i for i, s in enumerate(strengths) if s >= floor)
        frames, strengths = frames[first : last + 1], strengths[first : last + 1]
    times = tuple(round(offset_s + f * step, 4) for f in frames)
    if len(frames) > 1:
        bpm = 60.0 / (float(np.median(np.diff(frames))) * step)
    return BeatGrid(bpm, times, tuple(strengths))


def click_track(
    rate: int, seconds: float, bpm: float, *, bar: int = 4, offset_s: float = 0.5,
    strong: float = 1.0, weak: float = 0.4, click_s: float = 0.01, seed: int = 0,
) -> Any:  # fmt: skip
    """A synthetic drum machine: a decaying noise burst every beat from ``offset_s``, the bar's
    first at ``strong``, the others at ``weak`` (the beat tool's own test signal)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    x = np.zeros(round(rate * seconds))
    n = round(rate * click_s)
    burst = rng.standard_normal(n) * np.linspace(1.0, 0.0, n)
    period = 60.0 / bpm
    for k in itertools.count():
        start = round((offset_s + k * period) * rate)
        if start + len(burst) > len(x):
            break
        x[start : start + len(burst)] += burst * (strong if k % bar == 0 else weak)
    return np.clip(x, -1.0, 1.0)


# -- the default beat ------------------------------------------------------------------------------


def default_beat(
    rate: int = DEFAULT_BEAT_RATE,
    *,
    bpm: float = DEFAULT_BEAT_BPM,
    bars: int = DEFAULT_BEAT_BARS,
    seed: int = 0,
) -> Any:
    """The public default beat (the module's notes): ``bars`` of four beats at ``bpm``, a float
    array, -1..1, peak 0.9, the first kick on sample 0."""
    import numpy as np

    rng = np.random.default_rng(seed)
    period = 60.0 / bpm
    x = np.zeros(round(rate * bars * 4 * period))

    def place(start_s: float, sound: Any) -> None:
        start = round(start_s * rate)
        end = min(len(x), start + len(sound))
        x[start:end] += sound[: end - start]

    def decay(seconds: float, tau_s: float) -> Any:
        return np.exp(-np.arange(round(rate * seconds)) / (tau_s * rate))

    t = np.arange(round(rate * 0.35)) / rate
    sweep = 45.0 + (110.0 - 45.0) * np.exp(-t / 0.03)  # the kick's pitch falls 110 -> 45 Hz
    kick = np.sin(2 * np.pi * np.cumsum(sweep) / rate) * decay(0.35, 0.12)
    click = round(rate * 0.002)
    kick[:click] += 0.5 * rng.standard_normal(click) * np.linspace(1.0, 0.0, click)
    noise = np.diff(rng.standard_normal(round(rate * 0.25) + 1))  # high-passed
    body = np.sin(2 * np.pi * 190.0 * np.arange(len(noise)) / rate) * decay(0.25, 0.04)
    snare = 0.45 * noise * decay(0.25, 0.06) + 0.6 * body
    hat = 0.12 * np.diff(rng.standard_normal(round(rate * 0.06) + 1)) * decay(0.06, 0.012)
    for beat in range(bars * 4):
        at = beat * period
        accent = DEFAULT_BEAT_ACCENTS[beat % len(DEFAULT_BEAT_ACCENTS)]
        place(at, accent * (kick if beat % 2 == 0 else snare))
        place(at, hat)
        place(at + period / 2, hat)
    peak = float(np.abs(x).max()) or 1.0
    out: Any = 0.9 * x / peak
    return out


def default_beat_grid(bpm: float = DEFAULT_BEAT_BPM, bars: int = DEFAULT_BEAT_BARS) -> BeatGrid:
    """:func:`default_beat`'s beats by construction: every period from 0, the bar's accents."""
    period = 60.0 / bpm
    beats = range(bars * len(DEFAULT_BEAT_ACCENTS))
    return BeatGrid(bpm, tuple(round(k * period, 4) for k in beats),
                    tuple(DEFAULT_BEAT_ACCENTS[k % len(DEFAULT_BEAT_ACCENTS)] for k in beats),
                    DEFAULT_BEAT_METHOD)  # fmt: skip


def write_wav(path: str | Path, samples: Any, rate: int) -> None:
    """``samples`` (float, -1..1, mono) as a 16-bit PCM wav at ``path`` (its folder made)."""
    import wave

    import numpy as np

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0) * 32767.0)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.astype("<i2").tobytes())


# -- the contract ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Playing:
    """One play of a sound: the entry, the play's id, when its first sample reaches the speaker
    (engine clock) and the cut's beats as (seconds from then, strength)."""

    sound: Sound
    id: int
    t0: float
    beats: tuple[tuple[float, float], ...] = ()

    @property
    def end(self) -> float:
        """When the cut ends (engine clock)."""
        return self.t0 + self.sound.seconds


class BeatListener(Protocol):
    """What a player tells about the sound playing (the module's notes)."""

    def sound_started(self, playing: Playing) -> None:
        """The board scheduled it: its first sample at the speaker at ``playing.t0``."""
        ...

    def beat(self, t_beat: float, strength: float) -> None:
        """A beat sounds at ``t_beat`` (engine clock), told ``pulse.lead_s`` ahead."""
        ...

    def sound_ended(self, playing: Playing) -> None:
        """The cut ended, or was stopped."""
        ...


class SoundPlayer(Protocol):
    """The laptop's handle of the board's player (the module's notes)."""

    def play(self, name: str) -> Playing:
        """The cut mixed into the board's output from now; ``KeyError`` for a name not in the
        registry, ``RuntimeError`` for one without a file."""
        ...

    def stop(self, playing: Playing) -> None:
        """The cut cut short (nothing when it is over)."""
        ...

    def subscribe(self, listener: BeatListener) -> None:
        """``listener`` hears every play from now."""
        ...


class FakeSoundPlayer:
    """The contract on a clock: ``play`` schedules the cut ``lag_s`` ahead and tells the
    listeners at once; :meth:`step` (each tick of a test's clock) emits each beat ``lead_s``
    before it sounds and the end when the cut is over; every call is kept."""

    def __init__(
        self,
        registry: Registry,
        clock: Callable[[], float],
        *,
        lag_s: float = 0.1,
        lead_s: float | None = None,
    ) -> None:
        """``lead_s``: the registry's ``pulse.lead_s`` by default."""
        self.registry = registry
        self.clock = clock
        self.lag_s = lag_s
        self.lead_s = registry.pulse.lead_s if lead_s is None else lead_s
        self.listeners: list[BeatListener] = []
        self.calls: list[tuple[str, str]] = []
        self.playing: Playing | None = None
        self._ids = itertools.count(1)
        self._emitted = 0

    def play(self, name: str) -> Playing:
        """As the contract says; a sound playing already is stopped first."""
        sound = self.registry.sounds[name]
        if sound.file is None and not sound.external:
            raise RuntimeError(f"sound {name!r} has no file yet")
        if self.playing is not None:
            self.stop(self.playing)
        playing = Playing(sound, next(self._ids), self.clock() + self.lag_s, sound.cut_beats())
        self.playing, self._emitted = playing, 0
        self.calls.append(("play", name))
        for listener in self.listeners:
            listener.sound_started(playing)
        return playing

    def stop(self, playing: Playing) -> None:
        """As the contract says."""
        if self.playing is not playing:
            return
        self.calls.append(("stop", playing.sound.name))
        self._end(playing)

    def subscribe(self, listener: BeatListener) -> None:
        """As the contract says."""
        self.listeners.append(listener)

    def step(self) -> None:
        """The clock moved: the beats due within ``lead_s`` told, the end when it came."""
        playing = self.playing
        if playing is None:
            return
        now = self.clock()
        while self._emitted < len(playing.beats):
            at, strength = playing.beats[self._emitted]
            t_beat = playing.t0 + at
            if t_beat - self.lead_s > now:
                break
            self._emitted += 1
            for listener in self.listeners:
                listener.beat(t_beat, strength)
        if now >= playing.end:
            self._end(playing)

    def _end(self, playing: Playing) -> None:
        self.playing = None
        for listener in self.listeners:
            listener.sound_ended(playing)


def dance_style(playing: Playing | None, styles: Iterable[str]) -> str | None:
    """The style a bare ``{dance}`` takes while ``playing`` sounds: the sound's when the dance
    family (``styles``: its arguments) has it, else None (the family's first then, rock)."""
    if playing is None:
        return None
    style = playing.sound.style
    return style if style in set(styles) else None


def beats_of(times_s: Sequence[float], seconds: float) -> int:
    """How many of ``times_s`` fall within the first ``seconds`` (a tool's summary)."""
    return sum(1 for t in times_s if 0.0 <= t < seconds)


def period_s(bpm: float) -> float:
    """A beat's seconds at ``bpm``."""
    return 60.0 / bpm if bpm > 0 and math.isfinite(bpm) else math.inf


def main(argv: Sequence[str] | None = None) -> int:
    """Print the registry in force (the brain's list), or make the default beat."""
    import argparse

    parser = argparse.ArgumentParser(description="The sound registry and the default beat.")
    made = f"{SOUNDS_DIR}/{DEFAULT_BEAT_FILE}"
    parser.add_argument("--make-default", action="store_true", help=f"write {made}")
    parser.add_argument("--root", type=Path, help="the tree to write it in (the public one)")
    args = parser.parse_args(argv)
    if args.make_default:
        path = make_synthesized(DEFAULT_BEAT_FILE, args.root)
        grid = default_beat_grid()
        print(f"{path}: {grid.bpm:g} BPM, {len(grid.times_s)} beats, "
              f"{len(grid.times_s) * 60.0 / grid.bpm:g} s at {DEFAULT_BEAT_RATE} Hz")  # fmt: skip
        return 0
    for line in load_registry().lines():
        print(line)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
