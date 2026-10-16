"""The people the speaker gaze knows, as points in the map: the register.

ONE RECORD A PERSON (:class:`Person`): who they are (a name face recognition gave, else their
anonymous id), where their head is (``point``: x, y, z in the ``map`` frame, metres), when and by
what that point was placed (``stamp`` on the board's clock; ``source`` face, body, voice or told),
their velocity in the map (the camera's sightings fitted, :class:`pepin.expression.pursuit.
Pursuit`: the pursuit's lead), and a confidence that halves every ``half_life_s`` of age. The
register is the live layer a social map is to be built on (persistence, decay, "где Саша?");
nothing here persists or answers queries.

WHERE A POINT COMES FROM (the speaker gaze places every observation with the poses of its own
stamp: :mod:`pepin.expression.poses`):

* a FACE: the ray its angles name through the eye, from the camera's pose at the frame's
  exposure, at its range: the stereo depth when the finder measured one, else its size
  (``FACE_PX_M`` over its width);
* a BODY box: the ray to its head estimate, at the depth when measured, else the range the
  person's record already has (a face gave it), else ``default_range_m``; a body seen while the
  face was seen within ``seen_fresh_s`` keeps the face's point (the box is the rougher);
* a VOICE: the direction the array heard it from, taken into the map by the cart's heading at
  each reading's stamp, placed by the gaze along the head's axis at that pan
  (:class:`VoiceRay`): at a person the register knows there (within ``voice_match_deg``), at
  their range and height (their tilt, remembered), else at ``default_range_m`` and the seek's
  tilt;
* TOLD: the source a tool names for a point it gives (nothing gives one yet).

THE ASSOCIATION is by angle as seen from where the observation was made (the camera's place for
a frame, the cart's for a voice), the knobs' degrees as before: a sighting joins the nearest
record within ``match_deg``, a voice the nearest within ``voice_match_deg``. FRESHNESS DECIDES
between the camera and the array: a record seen within ``seen_fresh_s`` keeps its point against
a voice (the array is the noisier sense); an older one takes the voice's. The HEIGHT changes its
source (a face's to a box's, or back) only past ``tilt_hold_deg`` of elevation from the camera,
so the head does not nod as a face comes and goes.

THE ARRAY'S SOURCE PEAKS (the register of voices of 2026-10-14, folded in here): every peak of an
utterance's voiced readings joins the nearest record within ``voice_match_deg`` as its ``peak``
(never moving its point: nobody's engagement changes by a peak), or starts a record of its own
(``origin`` peaks), at most ``VOICES_KEPT`` of those beside the ``max_people`` others.

A record unseen and unheard for ``forget_s`` is forgotten; past the caps the stalest goes, never
one the gaze holds on to.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Literal

from pepin.expression.poses import HeadModel, PosesAt, Vec
from pepin.expression.pursuit import Pursuit
from pepin.expression.speaker import FaceFrame, FaceSighting, Point

Source = Literal["face", "body", "voice", "told"]
Origin = Literal["camera", "utterance", "peaks", "told"]
RangeFrom = Literal["depth", "face", "kept", "default"]
FACE_PX_M = 80.0  # a 0.15 m face is ~40 px at 2 m, ~26 px at 3 m: metres ~ this / width in px
VOICES_KEPT = 8  # records the array's peaks alone placed, kept beside the people
TRUST: dict[str, float] = {"face": 1.0, "body": 0.8, "voice": 0.5, "told": 0.5}


def wrap(rad: float) -> float:
    """An angle wrapped to -pi..pi."""
    return math.atan2(math.sin(rad), math.cos(rad))


def azimuth(view: tuple[float, float], point: Point) -> float:
    """The map direction of ``point`` from ``view`` (radians, + left of the map's x)."""
    return math.atan2(point.y - view[1], point.x - view[0])


@dataclass(frozen=True)
class Sighting:
    """One person in one frame, placed in the map: where their head is (a face's centre, or a
    body box's estimate) and how far, the ray it lies on from the eye (``origin``, ``direction``:
    the map's), its azimuth and elevation from there, which of the two placed it, the face (the
    smile's and the size gates', person recognition's), the frame's board stamp, and whether
    that is its exposure's (``exposed``; False: a frame without its stamp, placed at TF's newest,
    which no velocity is fitted on)."""

    point: Point
    source: Source
    stamp: float
    origin: Vec
    direction: Vec
    range_m: float
    range_from: RangeFrom
    face: FaceSighting | None = None
    exposed: bool = True

    @property
    def azimuth(self) -> float:
        """The map direction of the person from the eye (radians)."""
        return math.atan2(self.direction[1], self.direction[0])

    @property
    def elevation(self) -> float:
        """How far below level the person is from the eye (radians)."""
        return math.atan2(-self.direction[2], math.hypot(self.direction[0], self.direction[1]))

    def at(self, range_m: float) -> Point:
        """The point on the same ray at ``range_m``."""
        o, d = self.origin, self.direction
        return Point(o[0] + d[0] * range_m, o[1] + d[1] * range_m, o[2] + d[2] * range_m)

    @property
    def view(self) -> tuple[float, float]:
        """Where the eye stood (map xy)."""
        return self.origin[0], self.origin[1]


def sightings(
    frame: FaceFrame | None,
    at: PosesAt,
    head: HeadModel,
    *,
    min_face_px: float,
    min_body_score: float,
    default_range_m: float,
) -> list[Sighting]:
    """The people of ``frame`` placed in the map by the camera's pose at its stamp (``at``):
    every face ``min_face_px`` wide or more, and every body box scored ``min_body_score`` or more
    with none of those faces inside its width (a body without a depth at ``default_range_m``:
    the register re-ranges it on a record that has one)."""
    camera = at.camera
    if frame is None or camera is None:
        return []
    exposed = frame.stamp > 0.0
    faces = [f for f in frame.faces if f.width_px >= min_face_px]
    found: list[Sighting] = []

    def placed(
        pan_deg: float, tilt_deg: float, range_m: float, how: RangeFrom
    ) -> tuple[Point, Vec, Vec]:
        origin_link, direction_link = head.ray(pan_deg, tilt_deg)
        eye = camera.to_map(origin_link)
        d = camera.direction(direction_link)
        origin = (eye.x, eye.y, eye.z)
        return (
            Point(eye.x + d[0] * range_m, eye.y + d[1] * range_m, eye.z + d[2] * range_m),
            origin,
            d,
        )

    for f in faces:
        how: RangeFrom = "depth" if f.depth_m is not None else "face"
        range_m = f.depth_m if f.depth_m is not None else FACE_PX_M / f.width_px
        point, origin, d = placed(f.pan_deg, f.tilt_deg, range_m, how)
        found.append(Sighting(point, "face", at.stamp, origin, d, range_m, how, f, exposed))
    for b in frame.bodies:
        if b.score < min_body_score:
            continue
        if any(abs(f.pan_deg - b.pan_deg) <= b.half_width_deg for f in faces):
            continue  # its face speaks for it
        how = "depth" if b.depth_m is not None else "default"
        range_m = b.depth_m if b.depth_m is not None else default_range_m
        point, origin, d = placed(b.head_pan_deg, b.head_tilt_deg, range_m, how)
        found.append(Sighting(point, "body", at.stamp, origin, d, range_m, how, None, exposed))
    return found


@dataclass
class Person:
    """One person in the register (see the module): every stamp is the board's clock."""

    id: int
    point: Point
    stamp: float  # the observation that placed the point
    source: Source  # what placed it
    origin: Origin  # what made the record
    created_s: float
    who: str | None = None  # a name face recognition gave; None: anonymous (the id)
    seen_s: float | None = None  # a face or a body
    face_s: float | None = None
    heard_s: float | None = None  # an utterance
    voice: Point | None = None  # the last utterance's point
    peak: Point | None = None  # the array's last source peak's point
    peak_s: float | None = None
    strength: float | None = None  # that peak's SRP-PHAT response, 0..1
    peaks: int = 0  # the peaks that placed it
    range_m: float | None = None  # from the eye when last placed
    range_from: RangeFrom | None = None
    height_from: Source | None = None  # what placed the point's height (the tilt's hysteresis)
    face: FaceSighting | None = field(default=None, repr=False)
    motion: Pursuit = field(default_factory=Pursuit, repr=False)

    def fresh(self, now: float, within_s: float) -> bool:
        """Whether a face or a body placed it within ``within_s`` of ``now``."""
        return self.seen_s is not None and now - self.seen_s <= within_s

    def last_s(self) -> float:
        """When anything last placed it."""
        stamps = (self.created_s, self.seen_s, self.heard_s, self.peak_s)
        return max(t for t in stamps if t is not None)

    @property
    def velocity(self) -> Vec:
        """Its velocity in the map (m/s), the camera's sightings fitted."""
        return self.motion.velocity()

    def confidence(self, now: float, half_life_s: float) -> float:
        """How much its point is to be trusted now: its source's trust, halved every
        ``half_life_s`` since it was placed (0 for a half-life of 0)."""
        if half_life_s <= 0.0:
            return 0.0
        halves = max(0.0, now - self.stamp) / half_life_s
        return TRUST[self.source] * math.exp(-math.log(2.0) * halves)


@dataclass(frozen=True)
class VoiceRay:
    """A voice placed from the cart: the map direction it came from, where the cart stood, its
    board stamp, and how its point is placed (the gaze's, along the head's axis at the voice's
    pan): ``place(None)`` at the default range and the seek's tilt, ``place(person)`` at that
    person's range and height."""

    azimuth: float
    view: tuple[float, float]
    stamp: float
    place: Callable[[Person | None], Point]


class Register:
    """The people known (see the module): at most ``max_people`` placed by the camera or an
    utterance, and ``VOICES_KEPT`` by the array's peaks alone."""

    def __init__(
        self,
        *,
        match_deg: float = 20.0,
        voice_match_deg: float = 25.0,
        seen_fresh_s: float = 1.0,
        forget_s: float = 300.0,
        max_people: int = 4,
        min_new_px: float = 40.0,
        tilt_hold_deg: float = 0.0,
        half_life_s: float = 60.0,
    ) -> None:
        """The association's angles in degrees, the times in seconds."""
        self.match_rad = math.radians(match_deg)
        self.voice_match_rad = math.radians(voice_match_deg)
        self.tilt_hold_rad = math.radians(tilt_hold_deg)
        self.seen_fresh_s, self.forget_s = seen_fresh_s, forget_s
        self.max_people, self.min_new_px = max_people, min_new_px
        self.half_life_s = half_life_s
        self.people: list[Person] = []
        self._ids = itertools.count(1)

    def get(self, person_id: int | None) -> Person | None:
        """The record of ``person_id``; None when forgotten (or None asked)."""
        return next((p for p in self.people if p.id == person_id), None)

    def adopt(self, other: Register) -> None:
        """Take ``other``'s people (new knobs, the same people)."""
        self.people, self._ids = other.people, other._ids

    def clear(self) -> None:
        """Forget everyone."""
        self.people = []

    def forget(self, now: float, keep: Collection[int] = ()) -> list[Person]:
        """Drop the records unseen and unheard for ``forget_s`` (never one of ``keep``); those
        dropped."""
        old = [p for p in self.people if p.id not in keep and now - p.last_s() > self.forget_s]
        self.people = [p for p in self.people if p not in old]
        return old

    def voices(self) -> list[Person]:
        """The records an array's peak placed (the register of voices' rows)."""
        return [p for p in self.people if p.peaks > 0]

    # ---- the voice -----------------------------------------------------------------------------
    def heard(self, voice: VoiceRay, keep: Collection[int] = ()) -> tuple[Person, bool]:
        """An utterance's voice: the nearest record within ``voice_match_deg`` (its point the
        voice's unless it was seen within ``seen_fresh_s`` of it), else a new one; and whether it
        is new."""
        near = self._nearest(voice.view, voice.azimuth, self.voice_match_rad)
        if near is not None:
            near.heard_s, near.voice = voice.stamp, voice.place(near)
            if near.origin == "peaks":
                near.origin = "utterance"
            if not near.fresh(voice.stamp, self.seen_fresh_s):
                self._voice_places(near, near.voice, voice.stamp)
            return near, False
        point = voice.place(None)
        person = Person(next(self._ids), point, voice.stamp, "voice", "utterance", voice.stamp,
                        heard_s=voice.stamp, voice=point, height_from="voice")  # fmt: skip
        self._add(person, keep)
        return person, True

    def followed(self, person: Person, voice: VoiceRay) -> None:
        """The live voice placed ``person`` (nothing has seen them lately)."""
        person.heard_s, person.voice = voice.stamp, voice.place(person)
        self._voice_places(person, person.voice, voice.stamp)

    def to_voice(self, person: Person) -> None:
        """``person`` lost to the camera: their last utterance's point is theirs again."""
        if person.voice is not None and person.heard_s is not None:
            self._voice_places(person, person.voice, person.heard_s)

    def peak(self, voice: VoiceRay, strength: float) -> Person:
        """One of the array's source peaks: the record it joins (the nearest within
        ``voice_match_deg``: its peak, never its point, unless only peaks placed it), else a
        new record of its own."""
        near = self._nearest(voice.view, voice.azimuth, self.voice_match_rad)
        if near is not None:
            near.peak = voice.place(near) if near.origin != "peaks" else voice.place(None)
            near.peak_s, near.strength = voice.stamp, strength
            near.peaks += 1
            if near.origin == "peaks":
                near.point, near.stamp = near.peak, voice.stamp
            return near
        point = voice.place(None)
        person = Person(next(self._ids), point, voice.stamp, "voice", "peaks", voice.stamp,
                        peak=point, peak_s=voice.stamp, strength=strength, peaks=1,
                        height_from="voice")  # fmt: skip
        self._add(person, ())
        return person

    def named(self, person: Person, who: str) -> None:
        """``person`` is ``who`` (face recognition's verdict): the name moves off anyone else."""
        for other in self.people:
            if other is not person and other.who == who:
                other.who = None
        person.who = who

    # ---- the camera ----------------------------------------------------------------------------
    def seen(
        self,
        people: list[Sighting],
        now: float,
        *,
        seeking: Person | None = None,
        seek_rad: float = 0.0,
        keep: Collection[int] = (),
    ) -> dict[int, Sighting]:
        """One frame's people placed on the records (``now``: the board's): the sought record
        takes the person nearest it within ``seek_rad`` (never one who is another kept record's:
        the person a check returns to), and an earlier sighting of them within ``match_deg`` is
        merged in; the rest join the nearest record within ``match_deg`` or start one. Which
        record saw whom."""
        claimed: dict[int, Sighting] = {}
        left = list(people)
        kept = [p for p in self.people if p.id in keep and p is not seeking]
        free = [s for s in left if all(self._off(s, p) > self.match_rad for p in kept)]
        if seeking is not None and free:
            best = min(free, key=lambda s: self._off(s, seeking))
            if self._off(best, seeking) <= seek_rad:
                left.remove(best)
                twins = [
                    p
                    for p in self.people
                    if p is not seeking
                    and p.id not in keep
                    and p.seen_s is not None
                    and self._off(best, p) <= self.match_rad
                ]
                for twin in twins:  # an earlier sighting of the person sought: one person
                    self.people.remove(twin)
                    seeking.who = seeking.who or twin.who
                self._place(seeking, best, now)
                claimed[seeking.id] = best
        pairs = sorted(
            (self._off(s, p), i, p.id)
            for i, s in enumerate(left)
            for p in self.people
            if p.id not in claimed and self._off(s, p) <= self.match_rad
        )
        used: set[int] = set()
        for _, i, person_id in pairs:
            person = self.get(person_id)
            if i in used or person is None or person_id in claimed:
                continue
            self._place(person, left[i], now)
            claimed[person_id] = left[i]
            used.add(i)
        for i, s in enumerate(left):
            if i in used or (s.face is not None and s.face.width_px < self.min_new_px):
                continue
            person = Person(next(self._ids), s.point, s.stamp, s.source, "camera", s.stamp)
            self._place(person, s, now)
            self._add(person, keep)
        return claimed

    # ---- the parts -----------------------------------------------------------------------------
    @staticmethod
    def _off(s: Sighting, person: Person) -> float:
        """The angle between a sighting and a record's point, as seen from the eye."""
        return abs(wrap(s.azimuth - azimuth(s.view, person.point)))

    def _nearest(self, view: tuple[float, float], az: float, within: float) -> Person | None:
        near = [p for p in self.people if abs(wrap(azimuth(view, p.point) - az)) <= within]
        return min(near, key=lambda p: abs(wrap(azimuth(view, p.point) - az)), default=None)

    @staticmethod
    def _voice_places(person: Person, point: Point, stamp: float) -> None:
        person.point, person.source, person.stamp = point, "voice", stamp
        person.motion.reset()

    def _place(self, person: Person, s: Sighting, now: float) -> None:
        face_fresh = person.face_s is not None and now - person.face_s <= self.seen_fresh_s
        if s.source == "body" and face_fresh:
            person.seen_s = s.stamp  # still there; the face's point is the finer one
            return
        point, range_m, how = s.point, s.range_m, s.range_from
        if how == "default" and person.range_m is not None:  # a box: the range the record has
            range_m, how = person.range_m, "kept"
            point = s.at(range_m)
        held = person.height_from
        if held in ("face", "body") and held != s.source:
            then = self._elevation(s, person.point)
            if abs(s.elevation - then) <= self.tilt_hold_rad:  # within the hysteresis: kept
                point = Point(point.x, point.y, self._height_at(s, point, then))
                s_source: Source = held
            else:
                s_source = s.source
        else:
            s_source = s.source
        person.point, person.source, person.stamp = point, s.source, s.stamp
        person.seen_s = s.stamp
        person.range_m, person.range_from, person.height_from = range_m, how, s_source
        if person.origin == "peaks":  # the camera found a voice the array alone placed
            person.origin = "camera"
        if s.exposed:  # a velocity is fitted on exposures' stamps alone
            person.motion.add(s.stamp, point, s.source)
        if s.face is not None:
            person.face_s, person.face = s.stamp, s.face
            person.who = s.face.who or person.who

    @staticmethod
    def _elevation(s: Sighting, point: Point) -> float:
        """How far below level ``point`` is from the sighting's eye (radians)."""
        o = s.origin
        return math.atan2(o[2] - point.z, math.hypot(point.x - o[0], point.y - o[1]))

    @staticmethod
    def _height_at(s: Sighting, point: Point, elevation: float) -> float:
        """The height at ``point``'s distance from the eye that lies ``elevation`` below level."""
        o = s.origin
        return o[2] - math.tan(elevation) * math.hypot(point.x - o[0], point.y - o[1])

    def _add(self, person: Person, keep: Collection[int]) -> None:
        self.people.append(person)
        peaks = person.origin == "peaks"
        cap = VOICES_KEPT if peaks else self.max_people
        while True:
            group = [p for p in self.people if (p.origin == "peaks") == peaks]
            if len(group) <= cap:
                return
            spare = [p for p in group if p.id not in keep and p is not person]
            if not spare:
                return
            self.people.remove(min(spare, key=Person.last_s))
