"""The voice's bearing in the cart's frame, from the microphone array's direction of arrival.

The XVF3800 reads its direction about ten times a second on the board's clock and flags the
readings it takes for speech (:class:`pepin.audio_link.DoaReading`). An utterance's bearing is
the circular mean of the speech-flagged readings of its last ``window_s``, turned into the cart's
frame by :class:`pepin.expression.speaker.DoaMap`, positive to the left of the cart's nose, as
the neck's pan counts: piecewise-linear between the array's four calibration readings (front,
left, behind, right are 0, +90, 180, -90 exactly; an angle between two of them is interpolated),
or ``sign * (deg - front_deg)`` wrapped to -180..180 when only the front is known.

The ears stamp an utterance with the laptop's clock, the array with the board's: the newest
reading stands for "now" on the board (it is at most a tenth of a second old), so an utterance
that ended ``now - end_s`` ago ended that long before it. (Taking the window back from the newest
reading itself would land after the utterance once Whisper has taken its 0.3-1 s to read it.)

The robot's own voice is in the array while it speaks (the array names the loudest voice): a
reading stamped within one of the robot's speaking spans, or ``ECHO_S`` after one, is the
robot's and never the person's (:func:`robot_readings`). An utterance that overlaps a reply keeps
its clean part (:func:`utterance_voice`: the last ``window_s`` of its clean readings), and the
live stream is the clean speech of the last moments (:func:`recent_voice`): what the head
follows while a person speaks and no face is seen.

The array turns with the cart. While the wheels turn the body, a voice that stays where it is
reads at a bearing that sweeps the other way, so each reading is brought into the cart's frame of
now by how far the cart has turned since its stamp (``turned``: the wheels' odometry on the
board's clock, the array's own) before the readings are averaged: an utterance spoken across a
turn, or before it, gives the bearing the person has now.

:func:`fit_doa_map` is the calibration: the readings of a voice in front of, left of, behind and
right of the cart give ``front_deg``, ``sign`` and the three other quarters. The array sits off the
cart's centre (at its front-left), so the four are not a quarter turn apart (2026-10-12: front 170,
left 106, behind 343, right 278, misses of +0, -26, +7, -18 deg against an offset and a sign); the
piecewise map absorbs that, and a reading is refused only when it is nearer another quarter than
its own.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from pepin.audio_link import DoaReading
from pepin.audio_signal import circular_mean_deg
from pepin.expression.conversation import Heard
from pepin.expression.speaker import DoaMap, DoaSource

ECHO_S = 0.3  # the robot's voice rings this long in the room after its span ends
Turned = Callable[[float], float]  # how far the cart has turned since a board stamp (rad, + left)
QUARTER_TOL_DEG = 45.0  # a calibration reading further than this off its quarter is nearer the next
QUARTERS_DEG = (0.0, 90.0, 180.0, 270.0, 360.0)  # front, left, behind, right, front again


def wrap_deg(deg: float) -> float:
    """``deg`` wrapped to -180 (excluded) .. 180."""
    wrapped = math.fmod(deg + 180.0, 360.0)
    if wrapped <= 0.0:
        wrapped += 360.0
    return wrapped - 180.0


def bearing_deg(doa_deg: float, doa_map: DoaMap) -> float:
    """The array's angle as a bearing in the cart's frame (degrees, + left): interpolated
    between the quarters when they are measured, else the front's offset and the sign."""
    if doa_map.quarters is None:
        return wrap_deg(doa_map.sign * (doa_deg - doa_map.front_deg))
    swept = [0.0, *(_swept(q, doa_map) for q in doa_map.quarters), 360.0]
    angle = _swept(doa_deg, doa_map)
    for k in range(4):
        low, high = swept[k], swept[k + 1]
        if angle <= high:
            part = (angle - low) / (high - low)
            return wrap_deg(QUARTERS_DEG[k] + part * (QUARTERS_DEG[k + 1] - QUARTERS_DEG[k]))
    return 0.0  # not reached: the last quarter ends at 360


def _swept(doa_deg: float, doa_map: DoaMap) -> float:
    """How far the array's angle has swept from the front toward the cart's left, 0..360."""
    return (doa_map.sign * (doa_deg - doa_map.front_deg)) % 360.0


@dataclass(frozen=True)
class VoiceBearing:
    """An utterance's direction: the array's mean angle of its speech readings, the bearing in
    the cart's frame (radians, + left; None while the map is unmeasured), and the readings it
    came from."""

    doa_deg: float | None
    bearing_rad: float | None
    readings: int
    speech: int
    robot: int = 0  # speech readings dropped as the robot's own voice

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
    readings: Sequence[DoaReading],
    doa_map: DoaMap | None,
    min_speech: int = 1,
    turned: Turned | None = None,
) -> VoiceBearing:
    """The bearing of the speech-flagged ``readings`` (fewer than ``min_speech``: unknown); with
    ``turned``, each reading's bearing less how far the cart has turned since it was read."""
    speech = [r for r in readings if r.speech]
    mean = None
    if len(speech) >= max(min_speech, 1):
        mean = circular_mean_deg([float(r.deg) for r in speech])
    bearing = None
    if mean is not None and doa_map is not None:
        if turned is None:
            bearing = math.radians(bearing_deg(mean, doa_map))
        else:
            now = [bearing_deg(r.deg, doa_map) - math.degrees(turned(r.stamp_s)) for r in speech]
            centre = circular_mean_deg(now)
            bearing = None if centre is None else math.radians(wrap_deg(centre))
    return VoiceBearing(mean, bearing, len(readings), len(speech))


Span = tuple[float, float | None]  # the robot speaking from .. to (laptop clock; None: still)


def board_offset(doa: DoaSource, now: float) -> float | None:
    """The board's clock less the laptop's, by the newest reading (it stands for ``now``)."""
    latest = doa.latest_doa()
    return None if latest is None else latest.stamp_s - now


def robot_readings(
    readings: Sequence[DoaReading], spans: Sequence[Span], offset: float, echo_s: float = ECHO_S
) -> list[bool]:
    """Which of ``readings`` fall in the robot's speaking ``spans`` (laptop clock, mapped onto
    the board's by ``offset``) or ``echo_s`` after one: the robot's voice, not the person's."""
    board = [
        (start + offset, math.inf if end is None else end + offset + echo_s) for start, end in spans
    ]
    return [any(a <= r.stamp_s <= b for a, b in board) for r in readings]


@dataclass(frozen=True)
class UtteranceDoa:
    """An utterance's readings as the gaze logs them: every reading from its start to its end
    (board clock), which were the robot's, and its bearing from the clean ones."""

    readings: tuple[DoaReading, ...]
    robot: tuple[bool, ...]
    voice: VoiceBearing


def utterance_voice(
    doa: DoaSource,
    heard: Heard,
    now: float,
    *,
    window_s: float,
    doa_map: DoaMap | None,
    min_speech: int,
    spans: Sequence[Span] = (),
    turned: Turned | None = None,
) -> UtteranceDoa:
    """``heard``'s bearing from its clean readings: the robot's own voice cut out, the last
    ``window_s`` of what is left (the last ``window_s`` of the utterance when nothing was cut),
    each turned with the cart (``turned``), with every reading of the utterance for the log."""
    offset = board_offset(doa, now)
    if offset is None:
        return UtteranceDoa((), (), VoiceBearing(None, None, 0, 0))
    start, end = heard.start_s + offset, heard.end_s + offset
    every = doa.doa_between(start, end)
    mask = robot_readings(every, spans, offset)
    clean = [r for r, mine in zip(every, mask, strict=True) if not mine]
    if len(clean) == len(every):
        window = [r for r in every if r.stamp_s >= end - min(window_s, max(heard.duration_s, 0.0))]
    else:
        last = clean[-1].stamp_s if clean else end
        window = [r for r in clean if r.stamp_s >= last - window_s]
    voice = voice_of(window, doa_map, min_speech, turned)
    dropped = sum(1 for r, mine in zip(every, mask, strict=True) if mine and r.speech)
    return UtteranceDoa(tuple(every), tuple(mask), replace(voice, robot=dropped))


def recent_voice(
    doa: DoaSource,
    now: float,
    *,
    window_s: float,
    doa_map: DoaMap | None,
    min_speech: int,
    spans: Sequence[Span] = (),
    turned: Turned | None = None,
) -> VoiceBearing:
    """The live voice: the clean speech readings of the last ``window_s`` up to the newest,
    each turned with the cart (``turned``)."""
    latest = doa.latest_doa()
    if latest is None:
        return VoiceBearing(None, None, 0, 0)
    every = doa.doa_between(latest.stamp_s - window_s, latest.stamp_s)
    mask = robot_readings(every, spans, latest.stamp_s - now)
    clean = [r for r, mine in zip(every, mask, strict=True) if not mine]
    return voice_of(clean, doa_map, min_speech, turned)


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
    """The map the four readings give (the quarters interpolated), how far each reading sits
    from its quarter turn under the front's offset and sign alone (what the interpolation
    absorbs), and why it is refused (None when it holds)."""

    doa_map: DoaMap
    misses_deg: tuple[float, float, float, float]
    refusal: str | None

    def text(self) -> str:
        """One line for the calibration's output."""
        m = self.doa_map
        misses = ", ".join(f"{v:+.0f}" for v in self.misses_deg)
        head = f"front_deg {m.front_deg:.0f}, sign {m.sign:+d}"
        if m.quarters is not None:
            left, back, right = m.quarters
            head += f", left {left:.0f}, behind {back:.0f}, right {right:.0f}"
        head += f" (quarter-turn misses {misses} deg, interpolated away)"
        return head if self.refusal is None else f"{head}: REFUSED, {self.refusal}"


def fit_doa_map(
    front: float, left: float, back: float, right: float, tol_deg: float = QUARTER_TOL_DEG
) -> Calibration:
    """The map from the array's readings of a voice in front of, left of, behind and right of
    the cart: the sign puts the left a quarter turn left of the front, and the four readings
    are the quarters the map interpolates between; refused when a reading is further than
    ``tol_deg`` (at most 45) from its quarter turn, or the four are not in turn round the
    circle."""
    toward_left = wrap_deg(left - front)
    sign = 1 if toward_left > 0 else -1
    offset = DoaMap(front % 360.0, sign)
    expected = (0.0, 90.0, 180.0, -90.0)
    misses: list[float] = []
    for reading, want in zip((front, left, back, right), expected, strict=True):
        misses.append(wrap_deg(bearing_deg(reading, offset) - want))
    doa_map = DoaMap(offset.front_deg, sign, (left % 360.0, back % 360.0, right % 360.0))
    swept = [_swept(q, doa_map) for q in (left, back, right)]
    worst = max(abs(m) for m in misses)
    refusal = None
    if worst > min(tol_deg, QUARTER_TOL_DEG):
        refusal = (
            f"a reading sits {worst:.0f} deg off its quarter turn (more than"
            f" {min(tol_deg, QUARTER_TOL_DEG):.0f}): measure again, the voice 1.5 m away and the"
            " room quiet"
        )
    elif not 0.0 < swept[0] < swept[1] < swept[2] < 360.0:
        refusal = "the four readings are not in turn round the circle: measure again"
    return Calibration(doa_map, (misses[0], misses[1], misses[2], misses[3]), refusal)
