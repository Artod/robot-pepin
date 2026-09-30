"""Sound measurements for the laptop's end of the audio link: level, mean direction, onsets, WAV.

Laptop only (numpy); the board's server never imports this module.
"""

from __future__ import annotations

import math
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np

SILENCE_DBFS = -120.0  # the level reported for digital silence
FULL_SCALE = 32768.0


def level_dbfs(pcm: bytes) -> float:
    """RMS level of s16le PCM in dB below full scale (0 = a full-scale square wave); the floor
    :data:`SILENCE_DBFS` for silence or an empty buffer."""
    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float64)
    if samples.size == 0:
        return SILENCE_DBFS
    rms = float(np.sqrt(np.mean(samples * samples)))
    return max(SILENCE_DBFS, 20.0 * math.log10(rms / FULL_SCALE)) if rms > 0 else SILENCE_DBFS


def circular_mean_deg(angles: Sequence[float]) -> float | None:
    """The mean direction of angles in degrees, in [0, 360): 350 and 10 average to 0, not 180.
    None for no angles, or for angles that cancel out (opposite directions in equal measure)."""
    if not angles:
        return None
    radians = np.radians(np.asarray(angles, dtype=np.float64))
    s, c = float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians)))
    if math.hypot(s, c) < 1e-9:
        return None
    return math.degrees(math.atan2(s, c)) % 360.0


class SustainedVoice:
    """Says when a sound has stayed above a level long enough to be someone talking.

    Fires once per episode, when the sound has lasted ``hold_s``; dips shorter than ``gap_s``
    (the pauses between syllables) do not break an episode, a longer quiet ends it and rearms.
    """

    def __init__(
        self, threshold_db: float = -40.0, hold_s: float = 0.5, gap_s: float = 0.15
    ) -> None:
        """``threshold_db`` in dBFS; ``hold_s`` and ``gap_s`` in seconds."""
        self.threshold_db = threshold_db
        self.hold_s = hold_s
        self.gap_s = gap_s
        self.onset_s: float | None = None  # when the current loud episode began
        self._last_loud = 0.0
        self._fired = False

    def update(self, level_db: float, t: float, duration_s: float) -> bool:
        """One frame's level, starting at ``t`` and lasting ``duration_s``; True at the frame
        that completes ``hold_s`` of sound."""
        end = t + duration_s
        if level_db >= self.threshold_db:
            if self.onset_s is None:
                self.onset_s = t
            self._last_loud = end
            if not self._fired and end - self.onset_s >= self.hold_s - 1e-9:
                self._fired = True
                return True
        elif self.onset_s is not None and end - self._last_loud > self.gap_s:
            self.onset_s = None
            self._fired = False
        return False


def wav_to_pcm(path: str | Path, rate: int) -> bytes:
    """A WAV file (8, 16, 24 or 32-bit integer PCM, any channels, any rate) as s16le mono at
    ``rate``: channels averaged, resampled linearly — enough for speech on a small speaker."""
    with wave.open(str(path), "rb") as wav:
        width, channels, source_rate = wav.getsampwidth(), wav.getnchannels(), wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 3:
        triplets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
        data = np.where(ints >= 1 << 23, ints - (1 << 24), ints).astype(np.float64) / (1 << 23)
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"{path}: {8 * width}-bit samples are not supported")
    mono = data.reshape(-1, channels).mean(axis=1)
    if source_rate != rate and mono.size:
        count = round(mono.size * rate / source_rate)
        positions = np.arange(count) * (source_rate / rate)
        mono = np.interp(positions, np.arange(mono.size), mono)
    return bytes((np.clip(mono, -1.0, 1.0 - 1.0 / 32768.0) * 32768.0).astype("<i2").tobytes())
