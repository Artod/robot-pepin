"""A person's motion in the map, for the speaker gaze's pursuit (:mod:`pepin.expression.tracker`):
where they are heading, where they will be, and where the camera should find them next.

THE SIGHTINGS. Every sighting of a person is a point in the map placed with the camera's pose at
its frame's own exposure stamp (TF's, :mod:`pepin.expression.poses`): the head moving or not, the
cart turning or not, the point is where the person is in the room, so a fit over them is the
person's own motion and nothing of the robot's (since 2026-10-15; before, a sighting was a
bearing relative to the cart and a turn of the body started a new fit).

THE FIT. :class:`Pursuit` keeps one person's sightings of the last ``window_s`` at their frames'
stamps and fits their velocity by least squares on each axis (two sightings 0.1 s apart give a
velocity as noisy as their difference, so it waits for ``MIN_SIGHTINGS``); a change of what
placed them (a face's centre and a body box's head estimate are a few centimetres apart) starts
a new fit.

THE LEAD AND THE MATCH. Where the line puts them later (:meth:`Pursuit.ahead`: the head aims
there, the speaker gaze caps it by an angle) and where it puts them at a frame's stamp
(:meth:`Pursuit.expected`: a frame's people are matched there, not where the person was last
seen, so a walker hidden for half a second is himself when he comes back 20-25 deg further on;
trusted ``window_s`` past the newest sighting, no further).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pepin.expression.speaker import Point

LEAD_WINDOW_S = 0.5  # sightings further apart than this give no velocity to lead by
MIN_SIGHTINGS = 3  # sightings a velocity is fitted on (at 10 Hz, 0.2 s after the first)
STILL = (0.0, 0.0, 0.0)


@dataclass
class Pursuit:
    """One person's sightings of the last ``window_s`` (stamp, point) and what placed them
    (see the module); the register's record keeps one."""

    window_s: float = LEAD_WINDOW_S
    source: str | None = None
    sightings: list[tuple[float, Point]] = field(default_factory=list)

    def add(self, stamp_s: float, point: Point, source: str) -> None:
        """A sighting at ``point`` in the frame exposed at ``stamp_s`` (board clock); another
        source starts a new fit, the same frame twice (a stamp not newer) is ignored."""
        if source != self.source:
            self.source, self.sightings = source, []
        if self.sightings and stamp_s <= self.sightings[-1][0]:
            return
        self.sightings.append((stamp_s, point))
        self.sightings = [s for s in self.sightings if s[0] >= stamp_s - self.window_s]

    def reset(self) -> None:
        """Nothing fitted (a voice placed the person: the camera's line no longer holds)."""
        self.source, self.sightings = None, []

    @property
    def last_s(self) -> float | None:
        """The newest sighting's frame stamp; None before any."""
        return self.sightings[-1][0] if self.sightings else None

    def _fit(self) -> tuple[float, Point, tuple[float, float, float]] | None:
        """The mean stamp, the mean point and the velocity (m/s); None under MIN_SIGHTINGS."""
        n = len(self.sightings)
        if n < MIN_SIGHTINGS:
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
        return mean_t, Point(mx, my, mz), (vx, vy, vz)

    def velocity(self) -> tuple[float, float, float]:
        """The fitted velocity in the map (m/s); still under ``MIN_SIGHTINGS``."""
        fit = self._fit()
        return STILL if fit is None else fit[2]

    def at(self, at_s: float) -> Point | None:
        """The fitted line's point at ``at_s``; None under ``MIN_SIGHTINGS``."""
        fit = self._fit()
        if fit is None:
            return None
        mean_t, mean, v = fit
        dt = at_s - mean_t
        return mean.plus(v[0] * dt, v[1] * dt, v[2] * dt)

    def ahead(self, at_s: float) -> tuple[float, float, float]:
        """How far from the newest sighting the line puts the person at ``at_s`` (metres, the
        map's axes); nothing under ``MIN_SIGHTINGS``. The caller caps it."""
        there = self.at(at_s)
        if there is None or not self.sightings:
            return STILL
        last = self.sightings[-1][1]
        return there.x - last.x, there.y - last.y, there.z - last.z

    def expected(self, at_s: float, horizon_s: float) -> Point | None:
        """Where the line puts the person at ``at_s`` (a frame's exposure), extrapolated at most
        ``horizon_s`` past the newest sighting (THE MATCH); None under ``MIN_SIGHTINGS``."""
        last_s = self.last_s
        if last_s is None:
            return None
        return self.at(min(at_s, last_s + max(0.0, horizon_s)))


def tilt_turn_s(clip_max_hz: float) -> float:
    """The shortest time between two turns of the tilt's direction the arbiter allows an
    overlay (``gaze.clip_max_hz``: half its period; 0 Hz: no limit)."""
    return 0.5 / clip_max_hz if clip_max_hz > 0.0 else 0.0
