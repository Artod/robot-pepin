"""The voice's bearing in the cart's frame, from the microphone array's direction of arrival.

The XVF3800 reads its direction about ten times a second on the board's clock and flags the
readings it takes for speech (:class:`pepin.audio_link.DoaReading`). An utterance's bearing is
the circular mean of its readings of the person's voice in its last ``window_s``, turned into the
cart's frame by :class:`pepin.expression.speaker.DoaMap`, positive to the left of the cart's nose,
as the neck's pan counts: piecewise-linear between the array's four calibration readings (front,
left, behind, right are 0, +90, 180, -90 exactly; an angle between two of them is interpolated),
or ``sign * (deg - front_deg)`` wrapped to -180..180 when only the front is known.

THE TAGS (:class:`Tagger`, between the array and every consumer). The array's own speech flag
is set by its firmware's energy and spectrum thresholds, and it calls the neck's whine and the
wheels speech (2026-10-14: the six "no words" lines came with 86 of 87 readings flagged). Each
reading carries its tags (:class:`Tags`): ``chip`` (the array's flag), ``voice_p`` (the ears'
Silero speech probability around the reading's time, None when unknown), ``own`` (the robot's
own speaking spans and their echo), ``moving`` (the first ``WHEELS_ONSET_S`` of the robot's own
wheels' turn), and the reserved ``utterance`` and ``who`` (nothing fills them yet). One rule
(:class:`VoiceRule`, ``speaker.doa.doa_tags``) says which readings are the person's voice, and
every consumer reads its verdict, never the chip's flag: ``silero``, the chip AND ``voice_p`` of
at least ``voice_p_min`` AND not own AND not moving; ``chip``, the chip AND not own (as before
2026-10-14). Measured on the day's recordings (scratch/gaze/doa_silero.py): Silero gave the
neck's whine, nobody speaking, 0.5 or more on 2.5 % of the readings it would make, but the
wheels' start 0.2-0.4 s after their command on 89 % (the cart's first jerk sounds like a
syllable), hence the motion tag on the turn's start alone (later in a turn 0-6 % of the steps,
the room's own 3-5 %: the person's voice across a turn keeps its readings).

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

WHOSE READINGS (``speaker.doa.direction``, :class:`DoaChoice`, since 2026-10-14): ``chip``, the
array's own; ``mac``, the Mac's own from the four raw microphones (:mod:`pepin.audio_doa`: SRP-PHAT
on the chunks Silero calls voice, a list of source peaks per chunk, one reading per 0.1 s like
the chip's), read while raw frames come from the board and the chip's otherwise. A Mac reading
tags itself: its flag is Silero's at the chunk, its ``voice_p`` the chunks' highest, so the
``silero`` rule applies to it unchanged; its peaks beyond the strongest ride beside it
(:attr:`pepin.audio_doa.MicReading.peaks`) for the register of voices
(:class:`pepin.expression.tracks.VoiceSources`). The Mac's angles are in the array's own
geometry frame, the chip's in its firmware's: each has its own four calibration readings
(``speaker.doa.readings`` and ``mac_readings``).

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

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from pepin.audio_doa import MicReading
from pepin.audio_link import DoaReading
from pepin.audio_signal import circular_mean_deg
from pepin.expression.conversation import Heard
from pepin.expression.speaker import DoaMap, DoaSource

logger = logging.getLogger(__name__)

ECHO_S = 0.3  # the robot's voice rings this long in the room after its span ends
# A reading's voice_p: the highest Silero chunk within this of its time, one reading period each
# way (the clocks are mapped through the newest reading; 2026-10-14, scratch/gaze/doa_silero.py)
VOICE_P_HALF_S = 0.1
# The wheels' start sounds like a syllable to Silero for this long after their command (0.2-0.4 s
# measured, 2026-10-14); later in a turn 0-6 % of the 0.1 s steps read 0.5 (the room's 3-5 %)
WHEELS_ONSET_S = 0.5
DoaTags = Literal["silero", "chip"]  # speaker.doa.doa_tags: the rule's two arms
Direction = Literal["chip", "mac"]  # speaker.doa.direction: whose readings the gaze reads
RAW_STALE_S = 2.0  # no raw frame for this long: the Mac's source has nothing, the chip's is read
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


# the robot speaking, or its wheels starting, from .. to (laptop clock; None: still)
Span = tuple[float, float | None]


class VoiceSeries(Protocol):
    """The ears' speech probability on the laptop's clock
    (:class:`pepin.voice_cascade.barge_in.SpeechSeries`: Silero, one value per 32 ms chunk)."""

    def peak(self, t0: float, t1: float) -> float | None:
        """The highest probability of the chunks within ``t0..t1``; None when none is known."""
        ...


@dataclass(frozen=True)
class Tags:
    """What is known of one reading: the array's own speech flag, the ears' speech probability
    around it (None: unknown), whether it falls in the robot's own speech or its echo, and in
    the start of the robot's own wheels' turn. ``utterance`` and ``who`` are reserved for the
    next tiers (the utterance it belongs to, whose voice it is): nothing fills them yet."""

    chip: bool
    voice_p: float | None
    own: bool
    moving: bool = False
    utterance: int | None = None
    who: str | None = None


@dataclass(frozen=True)
class VoiceRule:
    """THE rule "this reading is the person's voice", the one place it lives
    (``speaker.doa.doa_tags`` and ``voice_p_min``): ``silero``, the chip's flag AND a speech
    probability of at least ``voice_p_min`` AND not the robot's own speech AND not its wheels'
    start; ``chip``, the chip's flag AND not the robot's own speech (as before 2026-10-14)."""

    tags: DoaTags = "silero"
    voice_p_min: float = 0.5

    def voiced(self, tags: Tags) -> bool:
        """Whether a reading with ``tags`` is the person's voice."""
        if not tags.chip or tags.own:
            return False
        if self.tags == "chip":
            return True
        return not tags.moving and tags.voice_p is not None and tags.voice_p >= self.voice_p_min


CHIP = VoiceRule("chip")  # the array's own flag, the robot's speech cut out (before 2026-10-14)


@dataclass(frozen=True)
class Tagged:
    """A reading, its tags and the rule's verdict on it: the only "speech" consumers read."""

    reading: DoaReading
    tags: Tags
    voice: bool


@dataclass(frozen=True)
class Tagger:
    """The stage between the array and its consumers: each reading tagged against the robot's
    speaking ``spans`` (laptop clock; None: still speaking), the starts of its wheels' turns
    (``motions``) and the ears' ``series`` (None: no speech model hears the stream), and judged by
    ``rule``."""

    rule: VoiceRule
    series: VoiceSeries | None = None
    spans: Sequence[Span] = ()
    motions: Sequence[Span] = ()

    def tag(self, readings: Sequence[DoaReading], offset: float) -> list[Tagged]:
        """``readings`` (board clock) tagged; ``offset`` is the board's clock less the laptop's.
        A reading the Mac computed (:class:`pepin.audio_doa.MicReading`) carries its own
        Silero probability (the chunks' highest) and its flag is that probability's: those are
        its tags, no lookup in the series."""
        own = robot_readings(readings, self.spans, offset)
        moving = robot_readings(readings, self.motions, offset, 0.0)
        tagged: list[Tagged] = []
        for reading, mine, moved in zip(readings, own, moving, strict=True):
            p = None
            if isinstance(reading, MicReading):
                p = reading.voice_p
            elif self.series is not None:
                t = reading.stamp_s - offset
                p = self.series.peak(t - VOICE_P_HALF_S, t + VOICE_P_HALF_S)
            tags = Tags(reading.speech, p, mine, moved)
            tagged.append(Tagged(reading, tags, self.rule.voiced(tags)))
        return tagged


class MacSource(DoaSource, Protocol):
    """The Mac's own direction source (:class:`pepin.audio_doa.MicDoa`): readings like the
    chip's, and whether raw frames come at all."""

    def raw_age_s(self, now: float | None = None) -> float:
        """Laptop seconds since the last raw frame; infinite before the first."""
        ...

    def configure(self, *, voice_p_min: float, peak_ratio: float, reading_s: float) -> None:
        """New knobs (``speaker.doa``'s ``mac_*``)."""
        ...


class DoaChoice:
    """The direction source the gaze reads (``speaker.doa.direction``): ``chip``, the array's
    own readings over the link; ``mac``, the Mac's from the raw microphones, while raw frames
    come (none for :data:`RAW_STALE_S`: the 2-channel firmware, or ``raw_mics`` off in
    config/audio.json: the chip's are read instead, said once each way)."""

    def __init__(
        self,
        direction: Direction,
        chip: DoaSource,
        mac: MacSource | None,
        *,
        report: Callable[[str], None] | None = None,
        stale_s: float = RAW_STALE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``mac`` None: no Mac source built (no geometry, no raw frames): the chip's."""
        self.direction = direction
        self.chip, self.mac = chip, mac
        self._report = report
        self._stale_s = stale_s
        self._clock = clock
        self._said: str | None = None  # the fallback last said ("chip" or "mac")

    def configure(
        self, direction: Direction, *, voice_p_min: float, peak_ratio: float, reading_s: float
    ) -> None:
        """A new direction and the Mac source's knobs (the gaze's switch turned on)."""
        self.direction = direction
        if self.mac is not None:
            self.mac.configure(voice_p_min=voice_p_min, peak_ratio=peak_ratio, reading_s=reading_s)

    @property
    def active(self) -> Direction:
        """Whose readings answer now."""
        if self.direction != "mac":
            return "chip"
        if self.mac is None:
            self._say("chip", "speaker.doa.direction mac but no Mac source: the chip's direction")
            return "chip"
        age = self.mac.raw_age_s(self._clock())
        if age > self._stale_s:
            why = "none yet" if math.isinf(age) else f"none for {age:.0f} s"
            line = (f"no raw frames from the board ({why}: the 2-channel firmware, or raw_mics"
                    " off in config/audio.json): the chip's direction")  # fmt: skip
            self._say("chip", line)
            return "chip"
        self._say("mac", "raw frames flow: the Mac's direction (speaker.doa.direction mac)")
        return "mac"

    def _say(self, which: str, line: str) -> None:
        if self._said == which:
            return
        self._said = which
        logger.info("%s", line)
        if self._report is not None:
            self._report(line)

    def _source(self) -> DoaSource:
        if self.active == "mac" and self.mac is not None:
            return self.mac
        return self.chip

    def latest_doa(self) -> DoaReading | None:
        """The active source's newest reading."""
        return self._source().latest_doa()

    def doa_between(self, t0: float, t1: float) -> list[DoaReading]:
        """The active source's readings within ``t0..t1`` (board clock)."""
        return self._source().doa_between(t0, t1)


@dataclass(frozen=True)
class VoiceBearing:
    """An utterance's direction: the array's mean angle of its readings of the person's voice
    (the rule's), the bearing in the cart's frame (radians, + left; None while the map is
    unmeasured), the readings judged, how many the rule took (``speech``), how many the chip
    flagged, and how many of the chip's were cut as the robot's own voice."""

    doa_deg: float | None
    bearing_rad: float | None
    readings: int
    speech: int
    robot: int = 0  # the chip's speech readings dropped as the robot's own voice
    chip: int = 0  # the chip's speech readings among those judged

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
    tagged: Sequence[Tagged],
    doa_map: DoaMap | None,
    min_speech: int = 1,
    turned: Turned | None = None,
) -> VoiceBearing:
    """The bearing of the ``tagged`` readings the rule takes for the person's voice (fewer than
    ``min_speech``: unknown); with ``turned``, each reading's bearing less how far the cart has
    turned since it was read."""
    speech = [t.reading for t in tagged if t.voice]
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
    chip = sum(1 for t in tagged if t.tags.chip)
    return VoiceBearing(mean, bearing, len(tagged), len(speech), chip=chip)


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
    (board clock) with its tags and the rule's verdict, and its bearing from the clean ones."""

    tagged: tuple[Tagged, ...]
    voice: VoiceBearing

    @property
    def readings(self) -> tuple[DoaReading, ...]:
        """Every reading of the utterance."""
        return tuple(t.reading for t in self.tagged)

    @property
    def robot(self) -> tuple[bool, ...]:
        """Which of them were the robot's own voice."""
        return tuple(t.tags.own for t in self.tagged)


def utterance_voice(
    doa: DoaSource,
    heard: Heard,
    now: float,
    *,
    window_s: float,
    doa_map: DoaMap | None,
    min_speech: int,
    tagger: Tagger,
    turned: Turned | None = None,
) -> UtteranceDoa:
    """``heard``'s bearing from its clean readings: the robot's own voice cut out, the last
    ``window_s`` of what is left (the last ``window_s`` of the utterance when nothing was cut),
    the person's voice in it by ``tagger``'s rule, each turned with the cart (``turned``), with
    every reading of the utterance for the log."""
    offset = board_offset(doa, now)
    if offset is None:
        return UtteranceDoa((), VoiceBearing(None, None, 0, 0))
    start, end = heard.start_s + offset, heard.end_s + offset
    every = tagger.tag(doa.doa_between(start, end), offset)
    clean = [t for t in every if not t.tags.own]
    if len(clean) == len(every):
        since = end - min(window_s, max(heard.duration_s, 0.0))
        window = [t for t in every if t.reading.stamp_s >= since]
    else:
        last = clean[-1].reading.stamp_s if clean else end
        window = [t for t in clean if t.reading.stamp_s >= last - window_s]
    voice = voice_of(window, doa_map, min_speech, turned)
    dropped = sum(1 for t in every if t.tags.own and t.tags.chip)
    return UtteranceDoa(tuple(every), replace(voice, robot=dropped))


def recent_voice(
    doa: DoaSource,
    now: float,
    *,
    window_s: float,
    doa_map: DoaMap | None,
    min_speech: int,
    tagger: Tagger,
    turned: Turned | None = None,
) -> VoiceBearing:
    """The live voice: the clean readings of the last ``window_s`` up to the newest, the
    person's voice in them by ``tagger``'s rule, each turned with the cart (``turned``)."""
    latest = doa.latest_doa()
    if latest is None:
        return VoiceBearing(None, None, 0, 0)
    every = doa.doa_between(latest.stamp_s - window_s, latest.stamp_s)
    clean = [t for t in tagger.tag(every, latest.stamp_s - now) if not t.tags.own]
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
