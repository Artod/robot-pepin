"""The people the speaker gaze knows: a few tracks, each one person's place, fed by faces, bodies
and voices.

A track is one person: their bearing in the cart's frame (``pan``, radians, + left), where the
head looks to see their face (``head``: pan and tilt, tilt + down), a rough distance, how fast
their bearing moves, when a face, a body box or their voice last placed them, and ``who`` (the
hook for person recognition: a face the finder knows by name).

* A FACE gives the head's aim itself (its centre through the lens).
* A BODY box gives an estimate of it (:class:`pepin.expression.speaker.BodySighting`: near the
  box's top), so a person standing, sitting, slouching or lying gives a tilt without a sweep; a
  face inside the box speaks for it instead. A body seen while the face was seen within
  ``seen_fresh_s`` keeps the face's aim (the box is the rougher of the two).
* The TILT changes its source (a face's to a box's, or back) only past ``tilt_hold_deg``: a box
  whose head estimate is that close to the face's height keeps the face's tilt, and a face that
  close to the box's keeps the box's (the pan follows the newest sighting either way). Without it
  the head nodded between the two whenever the face came and went in the frames.
* A VOICE gives a bearing only, and FRESHNESS DECIDES: a track seen (face or body) within
  ``seen_fresh_s`` keeps its place (the array is the noisier sense: on 2026-10-12 5-14 deg off the
  face at a normal level, 28-76 deg on quiet speech); an older track takes the voice's pan and
  keeps its tilt (the face's height, remembered), so a stale place never wins over a fresh
  bearing.

A sighting joins the nearest unclaimed track within ``match_deg``, a voice the nearest within
``voice_match_deg`` (else a new voice-only track at ``default_tilt_rad``). A sighting nobody claims
is a new track, unless it is a face under ``min_new_px`` (a small false face far off is nobody).
While the gaze SEEKS a track (it has a bearing and nobody has been seen there yet), the person
nearest it within the seek's reach IS that track: the array's bearing may be 30 deg off, and the
track a separate earlier sighting of that person made is merged into it. A track unseen and
unheard for ``forget_s`` is forgotten, and at most ``max_tracks`` are kept (the stalest goes,
never one the gaze holds on to).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Literal

from pepin.expression.speaker import FaceFrame, FaceSighting
from pepin.gaze import Aim

Source = Literal["face", "body", "voice"]
FACE_PX_M = 80.0  # a 0.15 m face is ~40 px at 2 m, ~26 px at 3 m: metres ~ this / width in px
LEAD_WINDOW_S = 0.5  # two sightings further apart than this give no speed to lead by


def wrap(rad: float) -> float:
    """An angle wrapped to -pi..pi."""
    return math.atan2(math.sin(rad), math.cos(rad))


def face_aim(head: Aim, face: FaceSighting) -> Aim:
    """The aim that puts ``face``, seen with the head at ``head``, on the optical axis."""
    pan, tilt = math.radians(face.pan_deg), math.radians(face.tilt_deg)
    return Aim(head.pan_rad + pan, head.tilt_rad + tilt)


@dataclass(frozen=True)
class Sighting:
    """One person in one frame, in the cart's frame: their bearing, where their head is (a
    face's centre, or a body box's estimate), which of the two, the face (the smile's and the
    size gates', person recognition's) and a rough distance from its width."""

    pan: float
    head: Aim
    source: Source
    face: FaceSighting | None = None
    distance_m: float | None = None


def sightings(
    frame: FaceFrame | None, head: Aim, *, min_face_px: float, min_body_score: float
) -> list[Sighting]:
    """The people of ``frame``, taken with the head at ``head``: every face ``min_face_px`` wide
    or more, and every body box scored ``min_body_score`` or more with none of those faces
    inside its width."""
    if frame is None:
        return []
    faces = [f for f in frame.faces if f.width_px >= min_face_px]
    found: list[Sighting] = []
    for f in faces:
        aim = face_aim(head, f)
        pan = wrap(aim.pan_rad)
        found.append(Sighting(pan, Aim(pan, aim.tilt_rad), "face", f, FACE_PX_M / f.width_px))
    for b in frame.bodies:
        if b.score < min_body_score:
            continue
        if any(abs(f.pan_deg - b.pan_deg) <= b.half_width_deg for f in faces):
            continue  # its face speaks for it
        aim = Aim(
            wrap(head.pan_rad + math.radians(b.head_pan_deg)),
            head.tilt_rad + math.radians(b.head_tilt_deg),
        )
        found.append(Sighting(wrap(head.pan_rad + math.radians(b.pan_deg)), aim, "body"))
    return found


@dataclass
class Track:
    """One person: where they are and what said so last (the laptop's clock)."""

    id: int
    pan: float
    head: Aim
    source: Source
    created_s: float
    seen_s: float | None = None  # a face or a body
    face_s: float | None = None
    heard_s: float | None = None
    voice_pan: float | None = None  # the last voice's bearing
    velocity: float = 0.0  # radians of pan a second, over the last two sightings
    distance_m: float | None = None
    face: FaceSighting | None = field(default=None, repr=False)
    who: str | None = None
    tilt_source: Source | None = None  # what placed the head's tilt (a voice keeps a default)

    def fresh(self, now: float, within_s: float) -> bool:
        """Whether a face or a body placed it within ``within_s``."""
        return self.seen_s is not None and now - self.seen_s <= within_s

    def last_s(self) -> float:
        """When anything last placed it."""
        return max(t for t in (self.created_s, self.seen_s, self.heard_s) if t is not None)

    def lead(self, lead_s: float, cap_rad: float) -> float:
        """How far its bearing will have gone ``lead_s`` from now, at most ``cap_rad``."""
        return max(-cap_rad, min(cap_rad, self.velocity * lead_s))


class Tracks:
    """The people known (at most ``max_tracks``), associated by angle; see the module."""

    def __init__(
        self,
        *,
        default_tilt_rad: float,
        match_deg: float = 20.0,
        voice_match_deg: float = 25.0,
        seen_fresh_s: float = 1.0,
        forget_s: float = 300.0,
        max_tracks: int = 4,
        min_new_px: float = 40.0,
        tilt_hold_deg: float = 0.0,
    ) -> None:
        """The association's angles in degrees, the times in seconds."""
        self.default_tilt_rad = default_tilt_rad
        self.tilt_hold_rad = math.radians(tilt_hold_deg)
        self.match_rad = math.radians(match_deg)
        self.voice_match_rad = math.radians(voice_match_deg)
        self.seen_fresh_s, self.forget_s = seen_fresh_s, forget_s
        self.max_tracks, self.min_new_px = max_tracks, min_new_px
        self.tracks: list[Track] = []
        self._ids = itertools.count(1)

    def get(self, track_id: int | None) -> Track | None:
        """The track of ``track_id``, None when forgotten (or None asked)."""
        return next((t for t in self.tracks if t.id == track_id), None)

    def adopt(self, other: Tracks) -> None:
        """Take ``other``'s people (new knobs, the same people)."""
        self.tracks, self._ids = other.tracks, other._ids

    def clear(self) -> None:
        """Forget everyone (the cart drove: every bearing is stale)."""
        self.tracks = []

    def forget(self, now: float, keep: Collection[int] = ()) -> list[Track]:
        """Drop the tracks unseen and unheard for ``forget_s`` (never one of ``keep``); those
        dropped."""
        old = [t for t in self.tracks if t.id not in keep and now - t.last_s() > self.forget_s]
        self.tracks = [t for t in self.tracks if t not in old]
        return old

    def heard(self, bearing: float, now: float, keep: Collection[int] = ()) -> tuple[Track, bool]:
        """A voice at ``bearing``: the nearest track within ``voice_match_deg`` (placed by it
        unless it was seen within ``seen_fresh_s``), else a new voice-only track; and whether it
        is new."""
        near = self._nearest(bearing, self.voice_match_rad, self.tracks)
        if near is not None:
            near.heard_s, near.voice_pan = now, bearing
            if not near.fresh(now, self.seen_fresh_s):
                self._voice_places(near, bearing)
            return near, False
        track = Track(
            next(self._ids),
            bearing,
            Aim(bearing, self.default_tilt_rad),
            "voice",
            now,
            heard_s=now,
            voice_pan=bearing,
            tilt_source="voice",
        )
        self._add(track, keep)
        return track, True

    def followed(self, track: Track, bearing: float, now: float) -> None:
        """The live voice placed ``track`` at ``bearing`` (nothing has seen it lately)."""
        track.heard_s, track.voice_pan = now, bearing
        self._voice_places(track, bearing)

    def seen(
        self,
        people: list[Sighting],
        now: float,
        *,
        seeking: Track | None = None,
        seek_rad: float = 0.0,
        keep: Collection[int] = (),
    ) -> dict[int, Sighting]:
        """One frame's people placed on the tracks: the sought track takes the person nearest
        it within ``seek_rad`` (never one who is another kept track's: the person a check
        returns to); the rest join the nearest track within ``match_deg`` or start one. Which
        track saw whom."""
        claimed: dict[int, Sighting] = {}
        left = list(people)
        kept = [t for t in self.tracks if t.id in keep and t is not seeking]
        free = [p for p in left if all(abs(wrap(p.pan - t.pan)) > self.match_rad for t in kept)]
        if seeking is not None and free:
            best = min(free, key=lambda p: abs(wrap(p.pan - seeking.pan)))
            if abs(wrap(best.pan - seeking.pan)) <= seek_rad:
                left.remove(best)
                twins = [
                    t
                    for t in self.tracks
                    if t is not seeking
                    and t.id not in keep
                    and t.seen_s is not None
                    and abs(wrap(t.pan - best.pan)) <= self.match_rad
                ]
                for twin in twins:  # an earlier sighting of the person sought: one person
                    self.tracks.remove(twin)
                    seeking.who = seeking.who or twin.who
                self._place(seeking, best, now)
                claimed[seeking.id] = best
        pairs = sorted(
            (abs(wrap(p.pan - t.pan)), i, t.id)
            for i, p in enumerate(left)
            for t in self.tracks
            if t.id not in claimed and abs(wrap(p.pan - t.pan)) <= self.match_rad
        )
        used: set[int] = set()
        for _, i, track_id in pairs:
            track = self.get(track_id)
            if i in used or track is None or track_id in claimed:
                continue
            self._place(track, left[i], now)
            claimed[track_id] = left[i]
            used.add(i)
        for i, p in enumerate(left):
            if i in used or (p.face is not None and p.face.width_px < self.min_new_px):
                continue
            track = Track(next(self._ids), p.pan, p.head, p.source, now)
            self._place(track, p, now)
            self._add(track, keep)
        return claimed

    def turned(self, rad: float) -> None:
        """The cart turned ``rad`` (+ left): every bearing moved the other way."""
        for t in self.tracks:
            t.pan = wrap(t.pan - rad)
            t.head = Aim(wrap(t.head.pan_rad - rad), t.head.tilt_rad)
            if t.voice_pan is not None:
                t.voice_pan = wrap(t.voice_pan - rad)

    # ---- the parts -----------------------------------------------------------------------------
    def _voice_places(self, track: Track, bearing: float) -> None:
        track.pan, track.source = bearing, "voice"
        track.head = Aim(bearing, track.head.tilt_rad)
        track.velocity = 0.0

    def _place(self, track: Track, p: Sighting, now: float) -> None:
        face_fresh = track.face_s is not None and now - track.face_s <= self.seen_fresh_s
        if p.source == "body" and face_fresh:
            track.seen_s = now  # still there; the face's aim is the finer one
            return
        then = track.seen_s
        if then is not None and 0.0 < now - then <= LEAD_WINDOW_S and track.source == p.source:
            track.velocity = wrap(p.pan - track.pan) / (now - then)
        else:
            track.velocity = 0.0
        tilt, tilt_source = p.head.tilt_rad, p.source
        held = track.tilt_source
        if (
            held in ("face", "body")
            and held != p.source
            and abs(tilt - track.head.tilt_rad) <= self.tilt_hold_rad
        ):
            tilt, tilt_source = track.head.tilt_rad, held  # within the hysteresis: kept
        track.pan, track.source, track.seen_s = p.pan, p.source, now
        track.head, track.tilt_source = Aim(p.head.pan_rad, tilt), tilt_source
        if p.face is not None:
            track.face_s, track.face, track.distance_m = now, p.face, p.distance_m
            track.who = p.face.who or track.who

    @staticmethod
    def _nearest(bearing: float, within: float, tracks: list[Track]) -> Track | None:
        near = [t for t in tracks if abs(wrap(t.pan - bearing)) <= within]
        return min(near, key=lambda t: abs(wrap(t.pan - bearing)), default=None)

    def _add(self, track: Track, keep: Collection[int]) -> None:
        self.tracks.append(track)
        while len(self.tracks) > self.max_tracks:
            spare = [t for t in self.tracks if t.id not in keep and t is not track]
            if not spare:
                break
            self.tracks.remove(min(spare, key=Track.last_s))
