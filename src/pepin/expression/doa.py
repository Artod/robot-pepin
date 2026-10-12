"""The voice's bearing in the cart's frame, from the microphone array's direction of arrival.

The XVF3800 reads its direction about ten times a second on the board's clock and flags the
readings it takes for speech (:class:`pepin.audio_link.DoaReading`). An utterance's bearing is
the circular mean of the speech-flagged readings of its last ``window_s``, turned into the cart's
frame by :class:`pepin.expression.speaker.DoaMap`: ``sign * (deg - front_deg)``, wrapped to
-180..180, positive to the left of the cart's nose, as the neck's pan counts.

The ears stamp an utterance with the laptop's clock, the array with the board's: the newest
reading stands for "now" on the board (it is at most a tenth of a second old), so an utterance
that ended ``now - end_s`` ago ended that long before it. (Taking the window back from the newest
reading itself would land after the utterance once Whisper has taken its 0.3-1 s to read it.)

:func:`fit_doa_map` is the calibration: the readings of a voice in front of, left of, behind and
right of the cart give ``front_deg`` and ``sign``, checked by the four being a quarter turn apart.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pepin.audio_link import DoaReading
from pepin.audio_signal import circular_mean_deg
from pepin.expression.conversation import Heard
from pepin.expression.speaker import DoaMap, DoaSource

QUARTER_TOL_DEG = 15.0  # the four calibration readings must be 90 deg apart within this


def wrap_deg(deg: float) -> float:
    """``deg`` wrapped to -180 (excluded) .. 180."""
    wrapped = math.fmod(deg + 180.0, 360.0)
    if wrapped <= 0.0:
        wrapped += 360.0
    return wrapped - 180.0


def bearing_deg(doa_deg: float, doa_map: DoaMap) -> float:
    """The array's angle as a bearing in the cart's frame (degrees, + left)."""
    return wrap_deg(doa_map.sign * (doa_deg - doa_map.front_deg))


@dataclass(frozen=True)
class VoiceBearing:
    """An utterance's direction: the array's mean angle of its speech readings, the bearing in
    the cart's frame (radians, + left; None while the map is unmeasured), and the readings it
    came from."""

    doa_deg: float | None
    bearing_rad: float | None
    readings: int
    speech: int

    def text(self) -> str:
        """``voice at 135 deg = +42 deg (8 readings, 6 speech)`` for a log line."""
        if self.doa_deg is None:
            return f"voice direction unknown ({self.readings} readings, {self.speech} speech)"
        where = (
            "the map is unmeasured"
            if self.bearing_rad is None
            else f"{math.degrees(self.bearing_rad):+.0f} deg"
        )
        return (
            f"voice at {self.doa_deg:.0f} deg = {where} ({self.readings} readings,"
            f" {self.speech} speech)"
        )


def voice_of(
    readings: Sequence[DoaReading], doa_map: DoaMap | None, min_speech: int = 1
) -> VoiceBearing:
    """The bearing of the speech-flagged ``readings`` (fewer than ``min_speech``: unknown)."""
    speech = [float(r.deg) for r in readings if r.speech]
    mean = circular_mean_deg(speech) if len(speech) >= max(min_speech, 1) else None
    bearing = None
    if mean is not None and doa_map is not None:
        bearing = math.radians(bearing_deg(mean, doa_map))
    return VoiceBearing(mean, bearing, len(readings), len(speech))


def utterance_readings(
    doa: DoaSource, heard: Heard, now: float, window_s: float
) -> list[DoaReading]:
    """The readings of ``heard``'s last ``window_s`` (never before it began): ``now`` is the
    laptop's clock the utterance is stamped with, mapped onto the board's by the newest
    reading."""
    latest = doa.latest_doa()
    if latest is None:
        return []
    end = latest.stamp_s - max(0.0, now - heard.end_s)
    start = end - min(window_s, max(heard.duration_s, 0.0))
    return doa.doa_between(start, end)


@dataclass(frozen=True)
class Calibration:
    """The map the four readings give, how far each sits from its quarter turn, and why it is
    refused (None when it holds)."""

    doa_map: DoaMap
    misses_deg: tuple[float, float, float, float]
    refusal: str | None

    def text(self) -> str:
        """One line for the calibration's output."""
        m = self.doa_map
        misses = ", ".join(f"{v:+.0f}" for v in self.misses_deg)
        head = f"front_deg {m.front_deg:.0f}, sign {m.sign:+d} (quarter-turn misses {misses} deg)"
        return head if self.refusal is None else f"{head}: REFUSED, {self.refusal}"


def fit_doa_map(
    front: float, left: float, back: float, right: float, tol_deg: float = QUARTER_TOL_DEG
) -> Calibration:
    """``front_deg`` and ``sign`` from the array's readings of a voice in front of, left of,
    behind and right of the cart: the sign is the one that puts the left a quarter turn left of
    the front; refused unless every reading is within ``tol_deg`` of its quarter turn."""
    toward_left = wrap_deg(left - front)
    sign = 1 if toward_left > 0 else -1
    doa_map = DoaMap(front % 360.0, sign)
    expected = (0.0, 90.0, 180.0, -90.0)
    misses: list[float] = []
    for reading, want in zip((front, left, back, right), expected, strict=True):
        misses.append(wrap_deg(bearing_deg(reading, doa_map) - want))
    worst = max(abs(m) for m in misses)
    refusal = None
    if worst > tol_deg:
        refusal = (
            f"a reading sits {worst:.0f} deg off its quarter turn (more than {tol_deg:.0f}):"
            " measure again, the voice 1.5 m away and the room quiet"
        )
    return Calibration(doa_map, (misses[0], misses[1], misses[2], misses[3]), refusal)
