"""The speaker gaze's pursuit of a person who moves (:mod:`pepin.expression.tracker`): the head's
pose at a frame's exposure, and the person's bearing ahead of their last sighting.

THE POSE OF A FRAME. A face's bearing is the head's pan when its frame was exposed plus the
face's angle through the lens, so a frame taken while the head moves is placed with the pose
the head had at the frame's own stamp, never the pose it has when the frame is read: the
perception service's record reaches the gaze 70-250 ms after its exposure (the hop to the
service, up to one detection period, the detector), 2-7 deg of a head gliding at 30 deg/s. The
record carries the exposure stamp (``Record.stamp``: camera_stream's X-Stamp, ustreamer's
capture time on the board's realtime clock, :func:`pepin.mjpeg.capture_time`) and no pose of the
neck (its fields are seq, stamp, arrived, frame, run, optics, brightness, faces, persons and ms;
camera_stream sends X-Stamp, X-Frame, X-Run, X-Optics and X-Brightness), so the pose is the
arbiter's: :class:`HeadPoses` keeps the encoder reading each state read returns (``/state``'s
``pan_rad`` and ``tilt_rad``, the newest reading the arbiter holds, and ``head_stamp``, that
reading's own stamp: the base bridge dates /neck/state by the moment of the encoder read, on the
board's realtime clock, the frames' clock) and gives the pose at a stamp between two readings by
interpolation. The frame and the pose are dated on one clock, so the placement never involves
this Mac's (a board clock 0.5 s off it through chrony lost the person, 2026-10-14 review), and
however old a reading is when read it is dated when the encoders saw it. An arbiter that sends no
``head_stamp`` (a gaze node started before it) has its readings dated by this Mac's wall clock at
the read: up to one neck reading (20 ms at 50 Hz) and the hop to the arbiter late, which while the
head moves places the person v x age behind (a re-centre on a person standing still read as a
walk of 6-16 deg/s, and the head chased it). The poses are one per state read (one per record,
10 a second), so a frame exposed across a move's start or stop is placed by a straight line
between two readings: up to a degree and a half off in the simulation.

THE BEARING AHEAD. :class:`Pursuit` keeps one person's sightings of the last ``window_s`` at
their frames' stamps and fits their speed by least squares (two sightings 0.1 s apart give a
speed as noisy as their difference, so it waits for ``MIN_SIGHTINGS``); the bearing at a later
moment is the fitted line's there, at most ``cap`` from the last sighting.

THE CAMERA'S FRAME. A sighting is the person's bearing relative to the cart at its frame's
stamp, as the neck's encoders and the lens place it, and the wheels' odometry never moves it:
the wheels slip and slide (2026-10-14, Artem: "одометрия врёт"), so while the body turns the
fit is the person's bearing as the camera sees it move, the cart's own turn included, which is
what the head has to follow. The turn's start and end change that motion at once, so a fit
across either is neither: each starts a new pursuit.

THE MATCH. A frame's people are matched against where the fit puts the person at that frame's
own stamp (:meth:`Pursuit.expected`), not against the bearing the odometry has shifted since
their last sighting: on the dock the odometry read twice the turn (2026-10-14), so after half a
second without a frame the shifted bearing was 10-20 deg off them and nearer a bystander. While
the body turns the fitted line holds through a gap (the cart's own turn, steady under Nav2);
otherwise it is trusted ``window_s`` past the newest sighting, no further.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from pepin.expression.tracks import LEAD_WINDOW_S, Source, wrap
from pepin.gaze import Aim

POSES_KEPT_S = 2.0  # the head's poses kept: a record is read at most a few tenths after exposure
MIN_SIGHTINGS = 3  # sightings a speed is fitted on (at 10 Hz, 0.2 s after the first)


@dataclass
class HeadPoses:
    """The head's pose at each encoder reading of the last ``kept_s``, keyed by the reading's
    own stamp (the board's clock; an older arbiter's: this Mac's at the read), read at any stamp
    between the first and the newest (see the module). The machine's thread alone adds and
    reads."""

    kept_s: float = POSES_KEPT_S
    _stamps: list[float] = field(default_factory=list)
    _aims: list[Aim] = field(default_factory=list)

    def add(self, stamp_s: float, aim: Aim) -> None:
        """The head at ``aim`` at ``stamp_s``; a stamp not newer than the newest (the same
        reading read twice) is ignored."""
        if self._stamps and stamp_s <= self._stamps[-1]:
            return
        self._stamps.append(stamp_s)
        self._aims.append(aim)
        while len(self._stamps) > 2 and self._stamps[1] < stamp_s - self.kept_s:
            del self._stamps[0], self._aims[0]

    def at(self, stamp_s: float) -> Aim | None:
        """The head's pose at ``stamp_s``, interpolated between the two readings around it;
        None outside the readings kept (before the first, or after the newest: exposed after
        the newest reading the gaze has read)."""
        stamps = self._stamps
        if not stamps or not stamps[0] <= stamp_s <= stamps[-1]:
            return None
        i = bisect.bisect_left(stamps, stamp_s)
        if stamps[i] == stamp_s:
            return self._aims[i]
        t0, t1 = stamps[i - 1], stamps[i]
        a0, a1 = self._aims[i - 1], self._aims[i]
        f = (stamp_s - t0) / (t1 - t0)
        return Aim(
            a0.pan_rad + f * (a1.pan_rad - a0.pan_rad),
            a0.tilt_rad + f * (a1.tilt_rad - a0.tilt_rad),
        )

    def newest_s(self) -> float | None:
        """The newest reading's stamp; None before any."""
        return self._stamps[-1] if self._stamps else None


@dataclass
class Pursuit:
    """One person's bearing over their sightings of the last ``window_s`` (see the module):
    ``track`` whose, ``source`` what placed them (a face's centre and a box's head estimate are
    a few degrees apart: a change of source starts a new pursuit), ``turning`` whether the body
    turned under them (its start or end starts a new pursuit), each sighting (frame stamp, pan
    relative to the cart at that stamp, unwrapped)."""

    track: int
    source: Source
    turning: bool = False
    window_s: float = LEAD_WINDOW_S
    sightings: list[tuple[float, float]] = field(default_factory=list)

    def add(self, stamp_s: float, pan: float) -> None:
        """A sighting at ``pan`` (radians) in the frame exposed at ``stamp_s``; the same frame
        twice (a stamp not newer than the last) is ignored."""
        if self.sightings:
            last_s, last = self.sightings[-1]
            if stamp_s <= last_s:
                return
            pan = last + wrap(pan - last)
        self.sightings.append((stamp_s, pan))
        self.sightings = [s for s in self.sightings if s[0] >= stamp_s - self.window_s]

    @property
    def last_s(self) -> float | None:
        """The newest sighting's frame stamp; None before any."""
        return self.sightings[-1][0] if self.sightings else None

    def speed(self) -> float:
        """The bearing's rate (rad/s, + left) fitted over the sightings kept; 0 with fewer than
        ``MIN_SIGHTINGS``."""
        n = len(self.sightings)
        if n < MIN_SIGHTINGS:
            return 0.0
        mean_t = sum(t for t, _ in self.sightings) / n
        mean_p = sum(p for _, p in self.sightings) / n
        var = sum((t - mean_t) ** 2 for t, _ in self.sightings)
        if var <= 0.0:
            return 0.0
        return sum((t - mean_t) * (p - mean_p) for t, p in self.sightings) / var

    def ahead(self, at_s: float, cap_rad: float) -> float:
        """How far from the last sighting the bearing will be at ``at_s`` (radians, the fitted
        line's), at most ``cap_rad`` either way; 0 without a speed."""
        n = len(self.sightings)
        if n < MIN_SIGHTINGS:
            return 0.0
        speed = self.speed()
        mean_t = sum(t for t, _ in self.sightings) / n
        mean_p = sum(p for _, p in self.sightings) / n
        there = mean_p + speed * (at_s - mean_t)
        return max(-cap_rad, min(cap_rad, there - self.sightings[-1][1]))

    def expected(self, at_s: float, horizon_s: float) -> float | None:
        """Where the fitted line puts the bearing at ``at_s`` (radians, unwrapped like the
        sightings), extrapolated at most ``horizon_s`` past the newest sighting (THE MATCH);
        None with fewer than ``MIN_SIGHTINGS``."""
        n = len(self.sightings)
        if n < MIN_SIGHTINGS:
            return None
        last_s = self.sightings[-1][0]
        at_s = min(at_s, last_s + max(0.0, horizon_s))
        mean_t = sum(t for t, _ in self.sightings) / n
        mean_p = sum(p for _, p in self.sightings) / n
        return mean_p + self.speed() * (at_s - mean_t)

    def turned(self, rad: float) -> None:
        """The cart turned ``rad`` (+ left): every sighting's bearing moved the other way (for
        sightings kept in the room's frame; the speaker gaze keeps its in the camera's, never
        moved: see the module)."""
        self.sightings = [(t, p - rad) for t, p in self.sightings]


def tilt_turn_s(clip_max_hz: float) -> float:
    """The shortest time between two turns of the tilt's direction the arbiter allows an
    overlay (``gaze.clip_max_hz``: half its period; 0 Hz: no limit)."""
    return 0.5 / clip_max_hz if clip_max_hz > 0.0 else 0.0
