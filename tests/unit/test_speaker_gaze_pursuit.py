"""The speaker gaze's PURSUIT (pepin.expression.pursuit; the tracker since 2026-10-14, Artem:
"вижу непрерывно"): a person walking is followed on every record of the camera channel, the head
moving or not, each box placed with the head's pose at its frame's exposure stamp, the head
gliding toward where they will be; and while the BODY TURNS the camera keeps them the same way,
whatever the wheels' odometry says (Artem: "колёса буксуют и проскальзывают, одометрия врёт,
голове на колёса плевать"). The engine runs on a simulated arbiter and neck (each aim driven
along its own ease, the state read at the pose of 20 ms before, as the arbiter's newest encoder
reading, stamped with that reading's own moment on the board's clock), a simulated perception
service (a frame exposed every 0.1 s, its record readable 0.15
s later, a face at the person's bearing from the head's pose AT THE EXPOSURE, with the detector's
jitter) and, for the turns, a simulated cart whose wheels slip (the odometry reads three times
the turn the cart makes)."""

from __future__ import annotations

import itertools
import logging
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
from test_speaker_gaze import BOARD, MAP, REACH, FakeSeries, array, speech
from test_speaker_gaze_spin import FakeSpins

from pepin.expression.conversation import Heard
from pepin.expression.executors import NoWheels, Wheels
from pepin.expression.fakes import FakeDoa, FakeWheels
from pepin.expression.head import SPEAKER
from pepin.expression.pursuit import HeadPoses, Pursuit
from pepin.expression.speaker import Ease, FaceFrame, FaceSighting, GazeAnswer, HeadView
from pepin.expression.tracker import (
    BY_FRAMES,
    BY_ODOMETRY,
    TILT_TURN_S,
    TURN_FRAMES_S,
    SpeakerGaze,
    SpeakerSettings,
)
from pepin.gaze import HOME, Aim, Speed

WALL = 1_760_000_000.0  # this Mac's wall clock at the simulation's 0 (the board's too, unskewed)
STEP_S = 0.005  # the neck's simulation step
TICK_S = 0.01  # the simulation's clock step
EYE_PAN_DEG, EYE_TILT_DEG = 41.0, 33.0  # the rectified eye's half view (800x600 at fx 454.5)
FACE_2M_PX = 40.0  # a face 2 m away in the rectified eye (tracks.FACE_PX_M 80 / 2 m)
HOME_POSE = (0.0, 23.8)
FRAME_PERIOD_S = 0.105  # the arbiter's blind tail after the head settles


def rad(d: float) -> float:
    return math.radians(d)


def deg(r: float) -> float:
    return math.degrees(r)


def axis(
    x: float, v: float, goal: float, speed: float, acc: float, dt: float
) -> tuple[float, float]:
    """One step of a servo's trapezoid toward ``goal``: at most ``speed``, ramped at ``acc``,
    braking in time; at the goal, still."""
    d = goal - x
    if abs(d) < 0.01 and abs(v) <= acc * dt:
        return goal, 0.0
    want = math.copysign(min(speed, math.sqrt(2.0 * acc * abs(d))), d)
    v += max(-acc * dt, min(acc * dt, want - v))
    return x + v * dt, v


@dataclass
class Neck:
    """The arbiter's door and the neck under it: an aim moves the head toward it on the aim's
    own ease (both axes, a newer aim taken in flight) and is answered done at once (the answer is
    not what is measured here); the state is the pose ``reading_age_s`` ago (the arbiter's
    newest encoder reading; up to ``reading_spread_s`` more, a read's own), stamped with that
    reading's moment unless the arbiter is an older one (``stamped`` False: no ``head_stamp``),
    blind while the head moves and one frame period after it settles, ``blind_until`` on the
    wall clock."""

    pan: float = HOME_POSE[0]
    tilt: float = HOME_POSE[1]
    t: float = 0.0
    goal: tuple[float, float] | None = None
    limits: tuple[float, float] = (200.0, 650.0)
    v: list[float] = field(default_factory=lambda: [0.0, 0.0])
    poses: list[tuple[float, float, float]] = field(default_factory=lambda: [(0.0, *HOME_POSE)])
    aims: list[tuple[float, Aim, Ease | None]] = field(default_factory=list)
    holder: str = ""
    settled_s: float | None = 0.0
    reading_age_s: float = 0.02
    reading_spread_s: float = 0.0
    stamped: bool = True
    board0: float = WALL  # the board's clock at the simulation's 0 (chrony's skew: board0 - WALL)
    rng: random.Random = field(default_factory=lambda: random.Random(1015))

    def advance(self, t: float) -> None:
        """The neck driven up to ``t``."""
        while self.t + 1e-9 < t:
            dt = min(STEP_S, t - self.t)
            self.t += dt
            if self.goal is not None:
                speed, acc = self.limits
                self.pan, self.v[0] = axis(self.pan, self.v[0], self.goal[0], speed, acc, dt)
                self.tilt, self.v[1] = axis(self.tilt, self.v[1], self.goal[1], speed, acc, dt)
                if (self.pan, self.tilt) == self.goal and self.v == [0.0, 0.0]:
                    self.goal, self.settled_s = None, self.t
            self.poses.append((self.t, self.pan, self.tilt))

    def pose_at(self, t: float) -> tuple[float, float]:
        """The head's (pan, tilt) in degrees at ``t``, the nearest step before."""
        _, pan, tilt = self.reading_at(t)
        return pan, tilt

    def reading_at(self, t: float) -> tuple[float, float, float]:
        """The nearest step at or before ``t``: its moment and the head's (pan, tilt) then."""
        lo, hi = 0, len(self.poses) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.poses[mid][0] <= t + 1e-9:
                lo = mid
            else:
                hi = mid - 1
        return self.poses[lo]

    def speed_at(self, t: float) -> float:
        """The head's pan speed (deg/s) about ``t``."""
        return abs(self.pose_at(t + 0.01)[0] - self.pose_at(t - 0.01)[0]) / 0.02

    def aim(self, aim: Aim, *, speed: Speed, ttl_s: float, ease: Ease | None = None) -> GazeAnswer:
        self.aims.append((self.t, aim, ease))
        self.goal = (deg(aim.pan_rad), deg(aim.tilt_rad))
        self.limits = (ease.speed_deg_s, ease.acc_deg_s2) if ease is not None else (200.0, 650.0)
        self.settled_s, self.holder = None, SPEAKER
        return GazeAnswer("done", head=aim)

    def renew(self) -> int:
        return 1

    def release(self) -> None:
        self.holder = ""

    def state(self) -> HeadView:
        age = self.reading_age_s + self.rng.uniform(0.0, self.reading_spread_s)
        read_s, pan, tilt = self.reading_at(self.t - age)
        until = None if self.settled_s is None else self.board0 + self.settled_s + FRAME_PERIOD_S
        blind = self.goal is not None or (until is not None and self.board0 + self.t < until)
        band = 5 if self.holder else HOME
        stamp = self.board0 + read_s if self.stamped else None
        return HeadView(Aim(rad(pan), rad(tilt)), self.holder, band, blind, until, False,
                        head_stamp=stamp)  # fmt: skip


Where = Callable[[float], tuple[float, float]]  # the person's (pan, tilt) degrees at a moment


@dataclass
class Channel:
    """The perception service: a frame exposed every ``period_s``, its record readable
    ``latency_s`` after the exposure; a look is the newest record past the last one given and
    exposed at ``not_before`` or later, or the next one waited for (the long poll: the clock
    moves on meanwhile, :attr:`wait`; up to ``wait_s``, then None), None while the service
    gives none (``down``); the person's face is where the head's pose AT THE EXPOSURE puts it
    in the eye's view, with the detector's jitter, unless it is turned ``away`` (records without
    it, by their exposure); ``others``' faces the same way (never turned away). Its stamps are
    on the board's clock (the neck's ``board0``)."""

    neck: Neck
    where: Where
    now: float = 0.0
    period_s: float = 0.1
    latency_s: float = 0.15
    wait_s: float = 0.4
    jitter_deg: float = 0.3
    face_px: float = FACE_2M_PX
    down: tuple[float, float] | None = None
    away: tuple[float, float] | None = None
    others: tuple[Where, ...] = ()
    seq: int = 0
    exposed: dict[int, float] = field(default_factory=dict)  # each record's exposure, by seq
    rng: random.Random = field(default_factory=lambda: random.Random(1014))
    wait: Callable[[float], None] = lambda t: None  # the clock moved on to t by a long poll

    def look(self, not_before: float | None = None) -> FaceFrame | None:
        if self.down is not None and self.down[0] <= self.now < self.down[1]:
            self.wait(self.now + self.wait_s)
            return None
        readable = math.floor((self.now - self.latency_s) / self.period_s + 1e-9)
        first = self.seq + 1
        if not_before is not None:
            first = max(first, math.ceil((not_before - self.neck.board0) / self.period_s - 1e-9))
        k = max(readable, first)
        exposed = k * self.period_s
        if exposed + self.latency_s > self.now + self.wait_s:
            self.wait(self.now + self.wait_s)
            return None
        if exposed + self.latency_s > self.now:
            self.wait(exposed + self.latency_s)
        self.seq = k
        self.exposed[k] = exposed
        faces: list[FaceSighting] = []
        for n, where in enumerate((self.where, *self.others)):
            face = self._face(where, exposed)
            if n == 0 and self.away is not None and self.away[0] <= exposed < self.away[1]:
                face = None
            if face is not None:
                faces.append(face)
        return FaceFrame(self.neck.board0 + exposed, tuple(faces), seq=k)

    def _face(self, where: Where, exposed: float) -> FaceSighting | None:
        """A person's face in the frame exposed at ``exposed``, None when out of the view."""
        pan, tilt = self.neck.pose_at(exposed)
        person_pan, person_tilt = where(exposed)
        off_pan = person_pan - pan + self.rng.gauss(0.0, self.jitter_deg)
        off_tilt = person_tilt - tilt + self.rng.gauss(0.0, self.jitter_deg)
        seen = abs(off_pan) <= EYE_PAN_DEG and abs(off_tilt) <= EYE_TILT_DEG and self.face_px > 0
        return FaceSighting(off_pan, off_tilt, self.face_px, 0.8) if seen else None


@dataclass
class Placed:
    """One record the person engaged was seen in: its exposure, the bearing the gaze placed
    them at, their true bearing then, and the head's pan at the exposure and at the read."""

    exposed: float
    placed: float
    true: float
    head_then: float
    head_read: float


@dataclass
class Cart:
    """The cart under the head: its true heading turns ``true_deg`` (+ left) evenly from ``t0``
    to ``t1``, and the wheels' odometry reads ``slip`` times as much (they spin and slide on the
    floor: 30 deg of odometry for a turn of 10), a state line every ``line_s`` on the board's
    clock."""

    wheels: FakeWheels
    t0: float
    t1: float
    true_deg: float
    slip: float = 3.0
    line_s: float = 0.02

    def true_at(self, t: float) -> float:
        """The cart's true heading (deg) at ``t``."""
        return self.true_deg * min(max(t - self.t0, 0.0), self.t1 - self.t0) / (self.t1 - self.t0)

    def odometry_at(self, t: float) -> float:
        """The heading the wheels' odometry reads at ``t``."""
        return self.slip * self.true_at(t)

    def tell(self, t: float) -> None:
        """The base server's state line at ``t``, when one is due."""
        latest = self.wheels.log.latest()
        if latest is None or BOARD + t >= latest[0] + self.line_s - 1e-9:
            self.wheels.turn_to(self.odometry_at(t), BOARD + t)


@dataclass
class Sim:
    """The engine, the neck, the channel, the array and the cart on one simulated clock."""

    machine: SpeakerGaze
    neck: Neck
    channel: Channel
    doa: FakeDoa
    events: list[tuple[str, float, dict[str, Any]]]
    cart: Cart | None = None
    t: float = 0.0
    due: float | None = None
    trace: list[tuple[float, float, float, str]] = field(default_factory=list)  # head, person
    placed: list[Placed] = field(default_factory=list)

    def run(self, until: float) -> None:
        """The clock to ``until``: the neck driven, the machine ticked whenever due, the head's
        and the person's pan traced, each record the person was placed by kept."""
        while self.t < until - 1e-9:
            self.t = round(self.t + TICK_S, 6)
            self.neck.advance(self.t)
            if self.cart is not None:
                self.cart.tell(self.t)
            self.channel.now = self.t
            if self.due is None or self.t >= self.due - 1e-9:
                seq, at = self.channel.seq, self.t
                self.due = self.machine.tick(at)
                self._placed(seq, at)
            person = self.channel.where(self.t)[0]
            self.trace.append((self.t, self.neck.pan, person, self.machine.phase))

    def moved_to(self, t: float) -> None:
        """The clock moved on to ``t`` while the machine waited on a long poll (the neck too)."""
        if t > self.t:
            self.neck.advance(t)
            if self.cart is not None:
                self.cart.tell(t)
            self.t = self.channel.now = t

    def _placed(self, before: int, at: float) -> None:
        track = self.machine.track
        seq = self.channel.seq
        if seq == before or track is None or track.seen_s != at or track.source != "face":
            return
        exposed = self.channel.exposed[seq]
        self.placed.append(
            Placed(exposed, deg(track.head.pan_rad), self.channel.where(exposed)[0],
                   self.neck.pose_at(exposed)[0], self.neck.pose_at(self.t)[0])
        )  # fmt: skip

    def wake(self, bearing: float) -> None:
        """The robot's name from ``bearing`` over 0..1.2 s, read at 1.3: the conversation
        opens on it."""
        self.run(1.3)
        self.doa.readings += speech(array(bearing), 0.0, 1.2) + speech(30, 1.3, 1.3)
        self.doa.readings[-1] = replace(self.doa.readings[-1], speech=False)
        self.machine.opened(Heard(0.0, 1.2, "Пепин", named=True), self.t)
        self.due = self.t

    def rows(self, kind: str) -> list[dict[str, Any]]:
        return [f for k, _, f in self.events if k == kind]

    def aims_between(self, t0: float, t1: float, why: str) -> list[dict[str, Any]]:
        return [f for k, t, f in self.events
                if k == "gaze_reaim" and t0 <= t <= t1 and f["why"] == why]  # fmt: skip


def sim(where: Where, *, wheels: Wheels | None = None, skew_s: float = 0.0, **knobs: Any) -> Sim:
    """The shipped knobs over the simple map, the head alone (arc 360: no body turn), ``knobs``
    pinned over them; ``wheels`` the cart's (none by default); ``skew_s`` how far the board's
    clock (the frames' and the encoders' stamps) runs ahead of this Mac's (chrony's error)."""
    neck = Neck(board0=WALL + skew_s)
    channel = Channel(neck, where)
    doa = FakeDoa()
    events: list[tuple[str, float, dict[str, Any]]] = []
    settings = replace(SpeakerSettings(doa_map=MAP), **{"arc_deg": 360.0, **knobs})
    wheels = wheels if wheels is not None else NoWheels()
    machine = SpeakerGaze(
        settings, door=neck, doa=doa, finder=channel, wheels=wheels, face=None, reach=REACH,
        report=lambda line: None, events=lambda kind, now, **f: events.append((kind, now, f)),
        series=FakeSeries(),
    )  # fmt: skip
    machine.spawn = lambda call: call()
    s = Sim(machine, neck, channel, doa, events)
    machine.wall = lambda: WALL + s.t
    channel.wait = s.moved_to
    return s


def walk(start: float, speed: float, t0: float, t1: float, tilt: float = 5.0) -> Where:
    """A person at ``start`` deg who walks at ``speed`` deg/s from ``t0`` to ``t1``."""

    def where(t: float) -> tuple[float, float]:
        return start + speed * min(max(t - t0, 0.0), t1 - t0), tilt

    return where


# ---- the pursuit ---------------------------------------------------------------------------------
@pytest.mark.parametrize("latency_s", [0.15, 0.25])
def test_a_person_walking_at_30_deg_s_at_2_m_is_followed_within_the_lead_cap_never_lost(
    latency_s: float,
) -> None:
    """Artem 2 m away (a 40 px face) walks 120 deg at 30 deg/s while the robot speaks, each
    record reaching the gaze ``latency_s`` after its exposure. Every record is taken (ten a
    second while it speaks too), most of them exposed while the head moves, each placed with
    the head's pose at its exposure; from a second into the walk the head stays within
    lead_max_deg (10) of him, he is never lost, the aims go out without waiting at most one a
    record, the pan always forward; he stops and the head settles on him."""
    s = sim(walk(-40.0, 30.0, 2.5, 6.5))
    s.channel.latency_s = latency_s
    s.wake(-40.0)
    s.run(2.5)
    assert s.machine.phase == "track" and abs(s.neck.pan + 40.0) < 3.0
    s.machine.speaking(True, s.t)
    s.machine.voiced(s.t + 0.2)
    frames = s.machine.counts["frames"]
    s.run(5.5)
    assert s.machine.counts["frames"] - frames >= 28, "ten a second while the robot speaks"
    s.machine.speaking(False, s.t)
    s.run(8.0)
    assert {phase for t, _, _, phase in s.trace if t >= 2.5} == {"track"}, "never lost"
    assert not [f for f in s.rows("gaze_lost") if f.get("what") == "face"]
    off = [abs(head - person) for t, head, person, _ in s.trace if 3.5 <= t <= 6.5]
    assert max(off) <= 10.0, max(off)
    assert abs(s.neck.pan - 80.0) <= 6.0, "he stopped: the head settles on him"
    pursuit = s.aims_between(2.5, 6.6, "the person moves")
    records = len([p for p in s.placed if 2.5 <= p.exposed <= 6.6])
    assert 25 <= len(pursuit) <= records, (len(pursuit), records)
    assert {f["status"] for f in pursuit} == {"sent"} and {f["speed"] for f in pursuit} == {"glide"}
    pans = [f["pan_deg"] for f in pursuit]
    assert all(b >= a for a, b in itertools.pairwise(pans)), "the pan never steps back"
    assert max(f["lead_deg"] for f in pursuit) <= 10.0
    assert 27.0 <= max(f["speed_deg_s"] for f in pursuit) <= 33.0, "his speed, fitted"
    moving = [p for p in s.placed if 3.0 <= p.exposed <= 6.5 and s.neck.speed_at(p.exposed) > 5.0]
    assert len(moving) >= 25, "most frames of the walk exposed while the head moved, all kept"
    assert s.machine.counts["frames_unposed"] == 0


def test_each_box_is_placed_with_the_head_s_pose_at_its_exposure_not_at_its_read() -> None:
    """The record reaches the gaze 0.15-0.25 s after its exposure: placed with the pose the
    head had then (the poses of the state reads, interpolated), the person's bearing is right
    within the detector's jitter and the reading's 20 ms; the head's pose at the read would put
    him 3-7 deg off while the head glides."""
    s = sim(walk(-40.0, 30.0, 2.5, 6.5))
    s.wake(-40.0)
    s.run(7.0)
    walking = [p for p in s.placed if 3.0 <= p.exposed <= 6.5]
    assert len(walking) >= 30
    errors = [abs(p.placed - p.true) for p in walking]
    assert max(errors) <= 2.0 and sum(errors) / len(errors) <= 0.8, errors
    at_read = [abs(p.head_read - p.head_then) for p in walking]
    assert sum(at_read) / len(at_read) >= 3.0, "the pose at the read: degrees off"


def test_a_person_walking_back_and_forth_and_bobbing_keeps_the_tilt_within_the_clip_rule() -> None:
    """He paces between -30 and +30 deg (30 deg/s at the middle, slowing to turn at the
    ends), his head bobbing +-4 deg at 1.5 Hz, the detector's jitter 0.5 deg: never lost, the
    head within 15 deg of him; the tilt of the aims never turns back sooner than 0.2 s after its
    last turn (the arbiter's clip_max_hz 2.5: the mast rings at 5.3 Hz)."""
    period = 2 * math.pi * 30.0 / 30.0  # +-30 deg at 30 deg/s at the middle: 6.3 s a round

    def pacing(t: float) -> tuple[float, float]:
        walking = max(t - 2.5, 0.0)
        pan = -30.0 * math.cos(2 * math.pi * walking / period)
        return pan, 5.0 + (4.0 * math.sin(2 * math.pi * 1.5 * walking))

    s = sim(pacing)
    s.channel.jitter_deg = 0.5
    s.wake(-30.0)
    s.run(12.5)
    assert {phase for t, _, _, phase in s.trace if t >= 2.5} == {"track"}, "never lost"
    aims = [(t, aim) for t, aim, _ in s.neck.aims if t >= 2.5]
    turns: list[float] = []
    way = 0
    for (_, a), (t, b) in itertools.pairwise(aims):
        step = b.tilt_rad - a.tilt_rad
        if abs(step) < 1e-6:
            continue
        now_way = 1 if step > 0 else -1
        if now_way != way:
            turns.append(t)
            way = now_way
    gaps = [b - a for a, b in itertools.pairwise(turns)]
    assert turns and min(gaps) >= TILT_TURN_S - 1e-9, gaps
    assert len(aims) <= 10 * 10 + 5, "at most an aim a record"
    off = [abs(head - person) for t, head, person, _ in s.trace if t >= 3.5]
    assert max(off) <= 15.0, "the turns at the ends cost a few degrees, never the person"


def test_looks_the_service_gave_no_record_for_are_no_misses_and_records_without_him_are() -> None:
    """The service stalls for 3 s (more than face_lost_s) while he stands: no record, no miss,
    no loss. Then he turns away: 2 s of records without him lose him (the head stays)."""
    s = sim(walk(20.0, 0.0, 0.0, 0.0))
    s.wake(20.0)
    s.run(2.5)
    assert s.machine.phase == "track"
    s.channel.down = (2.5, 5.5)
    s.run(5.4)
    assert s.machine.phase == "track" and s.machine.misses == 0
    assert s.machine.counts["frames_none"] >= 6, "each look waited 0.4 s for nothing"
    s.run(6.0)
    assert s.machine.phase == "track" and s.machine.misses == 0, "the records are back"
    s.channel.face_px = 0.0  # he turns away
    s.run(7.9)
    assert s.machine.phase == "track" and s.machine.misses >= 15
    s.run(8.3)
    assert s.machine.phase == "voice"
    assert [f for f in s.rows("gaze_lost") if f.get("what") == "face"]


def test_a_person_standing_still_costs_no_aim_and_after_a_step_aside_none_again() -> None:
    """Standing (the detector's jitter alone) no aim goes out; a step of 10 deg aside is
    followed and the head comes to rest within the dead band of him; standing again, no aim."""
    s = sim(walk(10.0, 20.0, 6.0, 6.5))
    s.wake(10.0)
    s.run(3.0)
    aims = len(s.neck.aims)
    s.run(6.0)
    assert len(s.neck.aims) == aims, "still: no aim"
    s.run(7.5)
    assert len(s.neck.aims) > aims and abs(s.neck.pan - 20.0) <= 6.0
    aims = len(s.neck.aims)
    s.run(9.0)
    assert len(s.neck.aims) == aims, "still again: no aim"


@pytest.mark.parametrize(("speed", "gap"), [(45.0, 0.5), (60.0, 0.3)])
def test_a_walker_hidden_for_a_moment_is_matched_on_his_line_not_where_he_was_last_seen(
    speed: float, gap: float
) -> None:
    """Artem walks 120 deg at 45 or 60 deg/s and his face is turned away for half a second or
    a third: he comes back 20-25 deg past his last sighting, beyond the match (20 deg) of where
    he was, so he used to become a stranger and the head lost him and went back to his voice.
    He is matched where the line over his sightings puts him at that frame's stamp (THE MATCH:
    at most the fit's own window past the newest sighting): never lost, the head on him."""
    s = sim(walk(-60.0, speed, 2.5, 2.5 + 120.0 / speed))
    s.channel.away = (3.5, 3.5 + gap)
    s.wake(-60.0)
    s.run(8.0)
    assert not [f for f in s.rows("gaze_lost") if f.get("what") == "face"]
    assert {phase for t, _, _, phase in s.trace if t >= 2.5} == {"track"}, "never lost"
    rest = [abs(head - person) for t, head, person, _ in s.trace if t >= 7.0]
    assert max(rest) <= 6.0, max(rest)


# ---- the body turns: the camera keeps him, whatever the wheels' odometry says --------------------
TURN = (2.5, 4.0)  # the cart turns over this span...
TRUE_DEG, ODOMETRY_DEG = 10.0, 30.0  # ...by 10 deg, while its slipping wheels' odometry reads 30


def slipping_turn(
    who: str, away: tuple[float, float] | None = None, turn: tuple[float, float] = TURN
) -> Sim:
    """Artem stands at +30 deg (2 m away, a 40 px face) and the cart turns 10 deg toward him
    over ``turn`` while its wheels' odometry reads 30 (they slip): ``who`` turns it, the gaze's
    own spin (the arc's rule, +-25 deg: one spin of his +30 asked of the goal server at the
    wake, accepted at once, its end told 0.1 s after the turn) or the turn tool's (the arc never
    turns the body: the loop's drive notice and keep_person 0.1 s before the turn, the drive's
    end 0.1 s after it). His face is turned ``away`` over that span of exposures. Woken by his
    name, run to 2 s after the turn."""
    return crowd_turn(who, away=away, turn=turn)


def crowd_turn(
    who: str,
    *,
    start: float = 30.0,
    true_deg: float = TRUE_DEG,
    slip: float = ODOMETRY_DEG / TRUE_DEG,
    turn: tuple[float, float] = TURN,
    away: tuple[float, float] | None = None,
    others: tuple[float, ...] = (),
    told_s: float = 0.1,
) -> Sim:
    """:func:`slipping_turn` with its numbers free: Artem at ``start`` deg, the cart turning
    ``true_deg`` toward him over ``turn`` while the odometry reads ``slip`` times that, the
    turn's end told ``told_s`` after the cart stopped, and ``others`` standing at their bearings
    (2 m away, faces never turned away), turned with the cart like him."""
    wheels = FakeWheels()
    t0, t1 = turn
    cart = Cart(wheels, t0, t1, true_deg, slip=slip)
    cart.tell(0.0)  # the wheels' link open: the odometry's heading
    own = who == "own"
    s = sim(lambda t: (start - cart.true_at(t), 5.0), wheels=wheels, body_turn="spin",
            arc_deg=50.0 if own else 360.0)  # fmt: skip
    s.cart, s.channel.away = cart, away
    s.channel.others = tuple(
        (lambda at: lambda t: (at - cart.true_at(t), 5.0))(other) for other in others
    )
    spins = FakeSpins()
    s.machine.spins = spins
    s.wake(start)
    if own:
        assert spins.asked == [pytest.approx(start)], "past the arc: one spin of his bearing"
        spins.tell("accepted", run=1, deg=start)
        s.run(t1 + told_s)
        spins.tell("done", run=1, status=4, seconds=t1 - t0, turned_deg=slip * true_deg)
    else:
        s.run(t0 - 0.1)
        s.machine.drive(True, s.t)
        assert s.machine.keep_person(s.t, f"turn: {true_deg:g}") is None
        s.run(t1 + told_s)
        s.machine.drive(False, s.t)
    s.run(t1 + 2.0)
    assert len(spins.asked) == (1 if own else 0), "no turn but the one"
    return s


@pytest.mark.parametrize("who", ["own", "tool"])
def test_while_the_body_turns_on_slipping_wheels_the_frames_keep_the_head_on_him(who: str) -> None:
    """The cart turns 10 deg toward Artem while its odometry reads 30 (the wheels slip), by
    the gaze's own spin or the turn tool's: every record is taken while it turns, each placed by
    the head's pose at its exposure and his face's angle through the lens (no odometry), and
    the head stays within 3 deg of him throughout, aimed by the frames alone (the rows say so,
    the camera's speed of his bearing beside the odometry's rate: -6.7 against 20 deg/s; none
    by the odometry, whose glide would have put him 20 deg off). He is never lost, and the head
    rests on him at +20, the turned nose 10 deg nearer."""
    s = slipping_turn(who)
    t0, t1 = TURN
    off = [abs(head - person) for t, head, person, _ in s.trace if t >= t0]
    assert max(off) <= 3.0, max(off)
    rows = s.aims_between(t0, t1 + 0.1, BY_FRAMES)
    assert len(rows) >= 5 and {f["status"] for f in rows} == {"sent"}, rows
    assert {f["speed"] for f in rows} == {"glide"}
    assert not [f for f in s.rows("gaze_reaim") if f["why"] == BY_ODOMETRY]
    middle = s.aims_between(t0 + 0.6, t1, BY_FRAMES)
    assert middle and all(abs(f["rate_deg_s"] - 20.0) <= 1.0 for f in middle), "the odometry"
    assert all(abs(f["speed_deg_s"] + 6.7) <= 2.0 for f in middle), "the camera's"
    assert s.machine.counts["frames_turning"] >= 10 * (t1 - t0), "every record while it turns"
    assert s.machine.counts["frames_unposed"] <= 2, "the wake's, exposed before a state read"
    assert {phase for t, _, _, phase in s.trace if t >= 2.0} == {"track"}, "never lost"
    assert abs(s.neck.pan - 20.0) <= 3.0


@pytest.mark.parametrize("who", ["own", "tool"])
def test_while_the_frames_show_nobody_the_odometry_glides_the_head_until_a_frame_shows_him(
    who: str,
) -> None:
    """The same 10 deg turn read as 30 by the odometry, over 2.5 s, his face turned away for
    a second of it: TURN_FRAMES_S after his last sighting the odometry moves the head (rows: by
    the odometry, no frame), a prior that slips with the wheels, so the head goes off him; his
    face back, the nearest person to that prior is him, the frames take the head again (rows:
    by the frames) and it is within 3 deg of him while the cart still turns. The odometry moves
    the head in the gap only; at the stop the pursuit's lead overshoots within its cap and the
    first frame after the turn's end re-centres the residue past refine_deg. He is never lost
    (a second without him, under face_lost_s)."""
    turn = (2.5, 5.0)
    t1, away = turn[1], (2.8, 3.8)
    s = slipping_turn(who, away=away, turn=turn)
    glides = s.aims_between(away[0], away[1] + 0.3, BY_ODOMETRY)
    assert len(glides) >= 3 and {f["speed"] for f in glides} == {"glide"}, glides
    assert all(abs(f["rate_deg_s"] - 12.0) <= 1.0 for f in glides), "the odometry's 30 in 2.5 s"
    rows = s.rows("gaze_reaim")
    assert len([f for f in rows if f["why"] == BY_ODOMETRY]) == len(glides), "in the gap only"
    first = min(t for k, t, f in s.events if k == "gaze_reaim" and f["why"] == BY_ODOMETRY)
    assert first >= away[0] + TURN_FRAMES_S, "a frame without him or two loses nothing"
    gap = [abs(head - person) for t, head, person, _ in s.trace if away[0] <= t <= away[1] + 0.2]
    assert max(gap) >= 6.0, "the odometry's prior slips with the wheels"
    assert s.aims_between(away[1], t1, BY_FRAMES), "his face back: the frames"
    back = [abs(head - person) for t, head, person, _ in s.trace if away[1] + 0.5 <= t <= t1]
    assert max(back) <= 3.0, max(back)
    stop = [abs(head - person) for t, head, person, _ in s.trace if t >= t1]
    assert max(stop) <= 10.0, "the lead's cap"
    rest = [abs(head - person) for t, head, person, _ in s.trace if t >= t1 + 0.6]
    assert max(rest) <= 3.0, max(rest)
    assert {phase for t, _, _, phase in s.trace if t >= 2.0} == {"track"}, "never lost"


@pytest.mark.parametrize("who", ["own", "tool"])
def test_at_the_stop_the_lead_ends_with_the_wheels_not_with_the_turn_s_end_notice(who: str) -> None:
    """The cart turns 30 deg toward Artem in 1.5 s (his bearing 20 deg/s in the camera) on
    wheels whose odometry reads 60, and the turn's end is told 0.3 s after it stopped (the goal
    server's word, or the tool's drive ending). The camera's fit lags the stop by half its
    window, so the lead used to carry the head 9-10 deg past him and keep it over 2 deg past for
    1.6 s; the odometry's rate falls to nothing within its 0.1 s when the wheels stop, and the
    lead with it: at most 7 deg past him, under half a second over 2 deg (the rows' lead 0
    once the wheels stand, while the turn is not yet told over)."""
    t1 = 4.0
    s = crowd_turn(who, start=40.0, true_deg=30.0, slip=2.0, turn=(2.5, t1), told_s=0.3)
    past = [(t, person - head) for t, head, person, _ in s.trace if t >= t1]
    assert max(d for _, d in past) <= 7.0, max(d for _, d in past)
    assert sum(TICK_S for _, d in past if d > 2.0) <= 0.5
    stood = s.aims_between(t1 + 0.12, t1 + 0.3, BY_FRAMES)
    assert all(f["lead_deg"] == 0.0 for f in stood), stood
    rest = [abs(head - person) for t, head, person, _ in s.trace if t >= t1 + 1.0]
    assert max(rest) <= 3.0, max(rest)


def test_a_bystander_the_frames_show_as_himself_is_never_sought_for_him_in_a_turn() -> None:
    """A friend stands 35 deg right of Artem, both seen from the wake on. Artem looks away
    just before the turn tool turns the cart 45 deg toward him on wheels whose odometry reads
    90, so no frame of the turn has placed him (no camera's fit): the gaze seeks him where the
    odometry puts him, with a seek's reach (50 deg) for its slip. The friend is within it but
    the frames show him as himself (his own track, there when Artem was last seen): never
    taken for Artem, who is found again when he looks back and kept to the end."""
    s = crowd_turn("tool", start=45.0, true_deg=45.0, slip=2.0, turn=(2.5, 5.5),
                   away=(2.2, 3.0), others=(10.0,))  # fmt: skip
    track = s.machine.track
    assert track is not None and abs(deg(track.pan)) <= 3.0, "Artem, where the turn put him"
    rest = [abs(head - person) for t, head, person, _ in s.trace if t >= 6.5]
    assert max(rest) <= 3.0, max(rest)
    friend = [t for t in s.machine.tracks.tracks if t is not track and abs(deg(t.pan) + 35.0) < 3]
    assert friend, "the friend keeps his own track"


def test_after_a_gap_the_camera_s_line_places_him_not_the_slipping_odometry() -> None:
    """The turn tool turns the cart 60 deg toward Artem at 30 deg/s on wheels whose odometry
    reads 120; his face is turned away for a second of it. The odometry puts him 30 deg past
    where he is, right where a man 45 deg beyond him has just come into view; the camera's line
    over his sightings before the gap puts him where he is: he is matched there when he looks
    back (the seek's 50 deg reach is the odometry's slip, not the camera's), the newcomer
    gets a track of his own, and the head ends on Artem: back from the odometry's 38 deg
    excursion at the glide's pace by 6 s, within 2 deg of him (with undated state reads a
    phantom re-aim stopped it 2.8 deg short at 5.5 s)."""
    s = crowd_turn("tool", start=60.0, true_deg=60.0, slip=2.0, turn=(2.5, 4.5),
                   away=(3.3, 4.3), others=(15.0,))  # fmt: skip
    track = s.machine.track
    assert track is not None and abs(deg(track.pan)) <= 3.0, "Artem, where the turn put him"
    rest = [abs(head - person) for t, head, person, _ in s.trace if t >= 6.0]
    assert max(rest) <= 2.0, max(rest)
    assert {phase for t, _, _, phase in s.trace if t >= 2.5} == {"track"}, "never lost"


def test_at_the_wake_on_its_own_spin_no_frame_before_the_head_looks_his_way_places_him() -> None:
    """Artem calls from 45 deg left, past the arc, a friend 16 deg right of him: the gaze's
    own spin starts at the wake and every record is taken from then, the first ones exposed
    while the head still points home and shows only the friend, 16 deg off Artem's voice
    (within the match): those place nobody on Artem (frames_away); the first frame that looks
    his way finds him, and he is kept through the turn."""
    s = crowd_turn("own", start=45.0, true_deg=45.0, slip=2.0, turn=(2.5, 5.5), others=(29.0,))
    first = s.rows("gaze_face")[0]
    assert abs(first["pan_deg"] - 45.0) <= 3.0, first
    assert s.machine.counts["frames_away"] >= 1
    track = s.machine.track
    assert track is not None and abs(deg(track.pan)) <= 3.0, "Artem, where the turn put him"


# ---- the head's poses on the board's clock -------------------------------------------------------
READS = [(0.02, 0.0), (0.05, 0.0), (0.02, 0.03)]  # how old a reading is when read: 20, 50, 20-50 ms


@pytest.mark.parametrize("skew_s", [-0.5, 0.0, 0.5])
@pytest.mark.parametrize(("age_s", "spread_s"), READS)
def test_a_box_is_placed_with_the_encoder_reading_at_its_stamp_whatever_its_age_or_chrony(
    skew_s: float, age_s: float, spread_s: float
) -> None:
    """The arbiter's state carries its encoder reading's own stamp (``head_stamp``: the board's
    clock, the frames' too), so each box of a walk at 30 deg/s is placed within the detector's
    jitter of him (0.2 deg on average) however old the reading was when read (20 or 50 ms, or
    anything between: the reads' own jitter) and whether the board's clock runs half a second
    behind or ahead of this Mac's (chrony's error): no frame unposed, never lost. Dated at
    their read by this Mac's clock, the same boxes were 0.6-4.8 deg off on average; a board
    0.5 s behind lost him, 0.5 s ahead left 54 frames unposed."""
    s = sim(walk(-40.0, 30.0, 2.5, 6.5), skew_s=skew_s)
    s.neck.reading_age_s, s.neck.reading_spread_s = age_s, spread_s
    s.wake(-40.0)
    s.run(7.5)
    walking = [p for p in s.placed if 3.0 <= p.exposed <= 6.5]
    assert len(walking) >= 30
    errors = [abs(p.placed - p.true) for p in walking]
    assert max(errors) <= 1.0 and sum(errors) / len(errors) <= 0.3, errors
    assert s.machine.counts["frames_unposed"] == 0
    assert {phase for t, _, _, phase in s.trace if t >= 2.5} == {"track"}, "never lost"


def test_a_re_centre_on_a_person_standing_still_reads_no_walk() -> None:
    """Artem stands 10-25 deg from where his voice put him, so the head turns onto his face
    while he stands (a re-centre), its readings read 20-50 ms after the encoders saw them.
    Dated at their read, the boxes exposed during the move were placed up to 6 deg off: a walk
    of 5-16 deg/s the head chased with up to 24 re-aims (the 2026-10-14 review: a -6 deg/s
    phantom, the head resting 3 deg off). Dated by their own stamps: at most 3 re-aims a case,
    none in most, every box within 1.5 deg of him (what is left: the 10 Hz readings' straight
    line across the move's start), the head resting within half a degree of him in most cases
    (in the others, within the dead band where an early aim's lead, fitted on three jittered
    sightings, left it), and a tenth of the re-aims the reads' dates cost."""
    people = ((20.0, 10.0), (30.0, 5.0), (-20.0, 0.0), (15.0, -5.0), (-35.0, -10.0))
    reads = [(0.02, 0.0), (0.035, 0.0), (0.05, 0.0), (0.02, 0.03)]
    reaims = {True: 0, False: 0}
    rests: list[float] = []
    for stamped in (True, False):
        for (person, voice), (age_s, spread_s) in itertools.product(people, reads):
            s = sim(walk(person, 0.0, 0.0, 0.0))
            s.neck.stamped, s.neck.reading_age_s, s.neck.reading_spread_s = stamped, age_s, spread_s
            s.wake(voice)
            s.run(6.0)
            rows = [f for f in s.rows("gaze_reaim") if f["why"].startswith("the person mov")]
            reaims[stamped] += len(rows)
            if stamped:
                case = (person, voice, age_s, spread_s)
                assert len(rows) <= 3, (case, [f["speed_deg_s"] for f in rows])
                assert max(abs(p.placed - p.true) for p in s.placed) <= 1.5, case
                rests.append(abs(s.neck.pan - person))
    assert sum(rest <= 0.5 for rest in rests) >= len(rests) // 2, rests
    assert max(rests) <= 6.0, "within the dead band"
    assert reaims[False] >= 10 * reaims[True], reaims


def test_an_arbiter_without_the_reading_s_stamp_dates_the_poses_at_the_read_said_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A gaze node started before ``head_stamp`` sends none: the head's poses are dated at
    their read by this Mac's clock, as before, said once (a warning naming the kick); every
    frame is still posed and he is followed, the boxes 20 ms of a 30 deg/s glide behind (0.6
    deg on average against 0.2 stamped)."""
    s = sim(walk(-40.0, 30.0, 2.5, 6.5))
    s.neck.stamped = False
    with caplog.at_level(logging.WARNING, logger="pepin.expression.tracker"):
        s.wake(-40.0)
        s.run(7.5)
    said = [r.getMessage() for r in caplog.records if "head_stamp" in r.getMessage()]
    assert len(said) == 1 and "ros/laptop.sh kick gaze" in said[0], said
    walking = [p for p in s.placed if 3.0 <= p.exposed <= 6.5]
    errors = [abs(p.placed - p.true) for p in walking]
    assert len(walking) >= 30 and s.machine.counts["frames_unposed"] == 0
    assert 0.4 <= sum(errors) / len(errors) <= 1.0 and max(errors) <= 1.5, errors
    assert {phase for t, _, _, phase in s.trace if t >= 2.5} == {"track"}, "never lost"


def test_the_state_reads_key_the_head_s_poses_on_the_reading_s_own_stamp() -> None:
    """A state read keeps the head's pose at its reading's ``head_stamp`` (here 40 ms before
    the read, on a board clock 0.3 s ahead of this Mac's), not at this Mac's clock of the read;
    the same reading read twice is kept once."""
    s = sim(walk(0.0, 0.0, 0.0, 0.0), skew_s=0.3)
    s.neck.reading_age_s = 0.04
    s.run(1.0)
    s.machine._state()
    s.machine._state()
    assert s.machine.poses.newest_s() == pytest.approx(WALL + 0.3 + 0.96)
    assert s.machine.poses.at(WALL + 0.3 + 0.96) == Aim(rad(HOME_POSE[0]), rad(HOME_POSE[1]))
    old = sim(walk(0.0, 0.0, 0.0, 0.0), skew_s=0.3)
    old.neck.stamped = False
    old.run(1.0)
    old.machine._state()
    assert old.machine.poses.newest_s() == pytest.approx(WALL + 1.0), "this Mac's, at the read"


# ---- the parts -----------------------------------------------------------------------------------
def test_the_head_s_poses_are_read_at_any_stamp_between_two_reads() -> None:
    poses = HeadPoses(kept_s=1.0)
    assert poses.at(10.0) is None and poses.newest_s() is None
    poses.add(10.0, Aim(0.0, 0.1))
    poses.add(10.1, Aim(0.2, 0.3))
    poses.add(10.1, Aim(9.0, 9.0))  # the same stamp again: ignored
    mid = poses.at(10.05)
    assert (
        mid is not None and mid.pan_rad == pytest.approx(0.1) and mid.tilt_rad == pytest.approx(0.2)
    )
    assert poses.at(10.1) == Aim(0.2, 0.3) and poses.at(10.0) == Aim(0.0, 0.1)
    assert poses.at(9.99) is None and poses.at(10.11) is None, "outside the reads: unknown"
    for k in range(2, 30):
        poses.add(10.0 + 0.1 * k, Aim(0.0, 0.0))
    assert poses.at(11.0) is None and poses.at(11.85) is not None, "a second kept"


def test_a_pursuit_fits_the_speed_on_the_frames_stamps_and_caps_the_lead() -> None:
    p = Pursuit(1, "face")
    p.add(100.0, rad(10.0))
    p.add(100.1, rad(13.0))
    assert p.speed() == 0.0 and p.ahead(100.5, rad(10.0)) == 0.0, "two sightings: no speed yet"
    p.add(100.1, rad(50.0))  # the same frame again
    p.add(100.2, rad(16.0))
    assert deg(p.speed()) == pytest.approx(30.0)
    assert deg(p.ahead(100.4, rad(10.0))) == pytest.approx(6.0)
    assert deg(p.ahead(101.0, rad(10.0))) == pytest.approx(10.0), "at most the cap"
    p.turned(rad(5.0))
    assert deg(p.sightings[-1][1]) == pytest.approx(11.0) and deg(p.speed()) == pytest.approx(30.0)
    for k in range(3, 12):
        p.add(100.0 + 0.1 * k, rad(16.0 - 5.0))
    assert deg(p.speed()) == pytest.approx(0.0, abs=1e-9) and len(p.sightings) == 6, "0.5 s kept"
    across = Pursuit(2, "face")
    for k, d in enumerate((170.0, 175.0, -180.0)):
        across.add(100.0 + 0.1 * k, rad(d))
    assert deg(across.speed()) == pytest.approx(50.0), "across the back: unwrapped"


def test_a_pursuit_expects_the_bearing_on_its_line_as_far_as_the_horizon_given() -> None:
    """THE MATCH's place: the fitted line at a frame's stamp, extrapolated at most the horizon
    past the newest sighting (the fit's own window outside a body turn, the whole gap in one),
    nothing before a speed is fitted."""
    p = Pursuit(1, "face")
    p.add(100.0, rad(10.0))
    p.add(100.1, rad(13.0))
    assert p.expected(100.3, 0.5) is None, "two sightings: no line yet"
    p.add(100.2, rad(16.0))
    there = p.expected(100.3, 0.5)
    assert there is not None and deg(there) == pytest.approx(19.0)
    held = p.expected(101.5, 0.5)
    assert held is not None and deg(held) == pytest.approx(31.0), "0.5 s past the newest"
    gap = p.expected(101.0, math.inf)
    assert gap is not None and deg(gap) == pytest.approx(40.0), "through a turn's gap"
