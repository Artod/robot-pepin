"""A person's motion in the map, for the speaker gaze's pursuit (:mod:`pepin.expression.tracker`):
where they are heading, where they will be, and where the camera should find them next.

THE SIGHTINGS. Every sighting of a person is a point in the map placed with the camera's pose at
its frame's own exposure stamp (TF's, :mod:`pepin.expression.poses`): the head moving or not, the
cart turning or not, the point is where the person is in the room, so a fit over them is the
person's own motion and nothing of the robot's (since 2026-10-15; before, a sighting was a
bearing relative to the cart and a turn of the body started a new fit).

THE FIT. :class:`Pursuit` keeps one person's sightings of the last ``window_s`` at their frames'
stamps and fits their velocity by least squares on each axis, only once they span ``fit_s``
(0.5 s, ``speaker.lead_fit_s``; at least ``MIN_SIGHTINGS`` of them): the 10 Hz frames' jitter
over 0.2 s (three sightings, before 2026-10-16) read as walking. A change of what placed them (a
face's centre and a body box's head estimate are a few centimetres apart) starts a new fit.

THE RANGE IS NO MOTION. A face's range is its size, ``FACE_PX_M`` (80) over its width in pixels
(:mod:`pepin.expression.people_map`): a box one pixel off is range / width metres off, 5 cm at
2 m (40 px), 11 cm at 3 m (27 px), so the 0.45 m of 2026-10-15 is 4 px at 3 m; that error lies
along the ray from the eye, while the ray itself (the face's centre) is good to a fraction of a
degree. The velocity is the motion ACROSS the newest
sighting's ray (``ray``; the along-ray part dropped), so a range that jumps never becomes a
velocity or a lead: on 2026-10-15 a seated person's point jumped up to 0.45 m and the pursuit
read 31-38 deg/s and led the head 10 deg (the cap) 18 times. A walker's motion round the robot
is across the ray, and one toward or away from it moves the head little.

THE LEAD AND THE MATCH. Where the line puts them later (:meth:`Pursuit.ahead`: the head aims
there, the speaker gaze caps it by an angle) and where it puts them at a frame's stamp
(:meth:`Pursuit.expected`: a frame's people are matched there, not where the person was last
seen, so a walker hidden for half a second is himself when he comes back 20-25 deg further on;
trusted ``window_s`` past the newest sighting, no further).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pepin.expression.speaker import Point

LEAD_FIT_S = 0.5  # the sightings a velocity is fitted on span at least this (speaker.lead_fit_s)
FIT_SLACK_S = 0.15  # kept this much longer than the span asked: a frame dropped leaves it whole
LEAD_WINDOW_S = LEAD_FIT_S + FIT_SLACK_S  # sightings further apart than this give no velocity
MIN_SIGHTINGS = 3  # and at least this many of them
STILL = (0.0, 0.0, 0.0)
Vec = tuple[float, float, float]


@dataclass
class Pursuit:
    """One person's sightings of the last ``window_s`` (stamp, point) and what placed them,
    a velocity fitted once they span ``fit_s``, across the newest sighting's ``ray`` (see the
    module); the register's record keeps one."""

    window_s: float = LEAD_WINDOW_S
    source: str | None = None
    sightings: list[tuple[float, Point]] = field(default_factory=list)
    fit_s: float = LEAD_FIT_S
    ray: Vec | None = None  # the newest sighting's unit direction from the eye (map)

    def add(self, stamp_s: float, point: Point, source: str, ray: Vec | None = None) -> None:
        """A sighting at ``point`` in the frame exposed at ``stamp_s`` (board clock), on ``ray``
        from the eye (None: the whole motion counts); another source starts a new fit, the same
        frame twice (a stamp not newer) is ignored."""
        if source != self.source:
            self.source, self.sightings = source, []
        if self.sightings and stamp_s <= self.sightings[-1][0]:
            return
        self.sightings.append((stamp_s, point))
        self.sightings = [s for s in self.sightings if s[0] >= stamp_s - self.window_s]
        self.ray = ray

    def reset(self) -> None:
        """Nothing fitted (a voice placed the person: the camera's line no longer holds)."""
        self.source, self.sightings, self.ray = None, [], None

    @property
    def last_s(self) -> float | None:
        """The newest sighting's frame stamp; None before any."""
        return self.sightings[-1][0] if self.sightings else None

    def _fit(self) -> tuple[float, Point, tuple[float, float, float]] | None:
        """The mean stamp, the mean point and the velocity across the newest ray (m/s); None
        under ``MIN_SIGHTINGS`` or a span short of ``fit_s``."""
        n = len(self.sightings)
        if n < MIN_SIGHTINGS or self.sightings[-1][0] - self.sightings[0][0] < self.fit_s - 1e-6:
            return None
        mean_t = sum(t for t, _ in self.sightings) / n
        mx = sum(p.x for _, p in self.sightings) / n
        my = sum(p.y for _, p in self.sightings) / n
        mz = sum(p.z for _, p in self.sightings) / n
        var = sum((t - mean_t) ** 2 for t, _ in self.sightings)
        if var <= 0.0:
            return mean_t, Point(mx, my, mz), STILL
        vx = sum((t - mean_t) * (p.x - mx) for t, p in self.sightings) / var
        vy = sum((t - mean_t) * (p.y - my) for t, p in self.sightings) / var
        vz = sum((t - mean_t) * (p.z - mz) for t, p in self.sightings) / var
        ray = self.ray
        if ray is not None:  # THE RANGE IS NO MOTION: the part along the ray dropped
            along = vx * ray[0] + vy * ray[1] + vz * ray[2]
            vx, vy, vz = vx - along * ray[0], vy - along * ray[1], vz - along * ray[2]
        return mean_t, Point(mx, my, mz), (vx, vy, vz)

    def velocity(self) -> tuple[float, float, float]:
        """The fitted velocity in the map (m/s); still before a fit (see :meth:`_fit`)."""
        fit = self._fit()
        return STILL if fit is None else fit[2]

    def at(self, at_s: float) -> Point | None:
        """The fitted line's point at ``at_s``; None before a fit."""
        fit = self._fit()
        if fit is None:
            return None
        mean_t, mean, v = fit
        dt = at_s - mean_t
        return mean.plus(v[0] * dt, v[1] * dt, v[2] * dt)

    def ahead(self, at_s: float) -> tuple[float, float, float]:
        """How far from the newest sighting the line puts the person at ``at_s`` (metres, the
        map's axes), across the newest ray (the line's mean range against the newest sighting's
        is the range's noise too); nothing before a fit. The caller caps it."""
        there = self.at(at_s)
        if there is None or not self.sightings:
            return STILL
        last = self.sightings[-1][1]
        d = (there.x - last.x, there.y - last.y, there.z - last.z)
        ray = self.ray
        if ray is None:
            return d
        along = d[0] * ray[0] + d[1] * ray[1] + d[2] * ray[2]
        return d[0] - along * ray[0], d[1] - along * ray[1], d[2] - along * ray[2]

    def expected(self, at_s: float, horizon_s: float) -> Point | None:
        """Where the line puts the person at ``at_s`` (a frame's exposure), extrapolated at most
        ``horizon_s`` past the newest sighting (THE MATCH); None before a fit."""
        last_s = self.last_s
        if last_s is None:
            return None
        return self.at(min(at_s, last_s + max(0.0, horizon_s)))


def tilt_turn_s(clip_max_hz: float) -> float:
    """The shortest time between two turns of the tilt's direction the arbiter allows an
    overlay (``gaze.clip_max_hz``: half its period; 0 Hz: no limit)."""
    return 0.5 / clip_max_hz if clip_max_hz > 0.0 else 0.0
