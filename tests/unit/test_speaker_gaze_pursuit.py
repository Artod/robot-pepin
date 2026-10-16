"""The speaker gaze's PURSUIT in the map (pepin.expression.pursuit, people_map; since 2026-10-15
every sighting and every voice is a point in the map placed by TF's poses at its own stamp, and
the head is asked for a point the arbiter keeps it on as the cart turns): a person walking is
followed on every record of the camera channel, the head moving or not, each box placed with the
camera's pose at its frame's exposure, the head gliding toward where they will be; and while the
BODY TURNS the arbiter keeps the head on the person's point by TF alone, whatever the wheels'
odometry would say (2026-10-15: the front wheels on the stand spun at -35 deg/s, the cart did
not turn, and the head was re-aimed off the person 143 times "by the odometry").

The engine runs on a simulated arbiter and neck (each point resolved to pan and tilt from the
cart's true pose, followed every 0.05 s as the node's step does and written again past its
settle tolerance, driven along its aim's own ease; the state read at the pose of 20 ms before,
as the arbiter's newest encoder reading, stamped with that reading's own moment on the board's
clock), a simulated TF (the cart's true heading and the neck's pose at any stamp, through the
real neck's lever arms and the real eye), a simulated perception service (a frame exposed every
0.1 s, its record readable 0.15 s later, a face where the camera's pose AT THE EXPOSURE sees the
person's point, with the detector's jitter) and, for the turns, a cart whose true heading is
TF's: Nav2 may claim any turn it likes, only TF's counts."""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from test_speaker_gaze import MAP, FakeSeries, array
from test_speaker_gaze_spin import FakeSpins

from pepin.audio_link import DoaReading
from pepin.expression.conversation import Heard
from pepin.expression.executors import NoWheels, Wheels
from pepin.expression.fakes import FakeDoa, FakeTf, FakeWheels
from pepin.expression.head import SPEAKER
from pepin.expression.people_map import FACE_PX_M
from pepin.expression.poses import BoardClock, HeadModel, Pose
from pepin.expression.pursuit import Pursuit
from pepin.expression.speaker import Ease, FaceFrame, FaceSighting, GazeAnswer, HeadView, Point
from pepin.expression.tracker import LEAD, MOVED, TILT_TURN_S, SpeakerGaze, SpeakerSettings
from pepin.gaze import HOME, Aim, Reach, Speed
from pepin.mounts import load_camera_mounts
from pepin.neck import NeckConfig

REPO = Path(__file__).resolve().parents[2]
NECK = NeckConfig.from_json(REPO / "config/neck.json")
MODEL = HeadModel(NECK, load_camera_mounts(REPO / "config").optical)  # the real arms and eye
REACH = Reach.of(NECK)
WALL = 1_760_000_000.0  # this Mac's wall clock at the simulation's 0 (the board's too, unskewed)
STEP_S = 0.005  # the neck's simulation step
TICK_S = 0.01  # the simulation's clock step
FOLLOW_S = 0.05  # the arbiter's step: a held point resolved again this often (STEP_HZ 20)
SETTLE_DEG = 1.0  # ...and written again past this (gaze settle_tol_deg)
EYE_PAN_DEG, EYE_TILT_DEG = 41.0, 33.0  # the rectified eye's half view (800x600 at fx 454.5)
RANGE_M = 2.0  # people stand 2 m from the lens (a 40 px face: tracks' FACE_PX_M 80 / 2 m)
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


Where = Callable[[float], tuple[float, float]]  # a person's (pan, tilt) degrees at a moment


def person_at(where: Where, t: float) -> Point:
    """A person's head in the map at ``t``: where the head would look to see them with the cart
    at the origin facing +x, ``RANGE_M`` from the lens."""
    pan, tilt = where(t)
    x, y, z = MODEL.axis_point(Aim(rad(pan), rad(tilt)), RANGE_M)
    return Point(x, y, z)


@dataclass
class Cart:
    """The cart's TRUE heading (TF's, + left): ``true_deg`` turned evenly from ``t0`` to ``t1``
    (whatever Nav2 or the wheels claim)."""

    t0: float = 0.0
    t1: float = 1.0
    true_deg: float = 0.0

    def heading(self, t: float) -> float:
        """The cart's heading (deg) at ``t`` (simulation seconds)."""
        span = self.t1 - self.t0
        return self.true_deg * min(max(t - self.t0, 0.0), span) / span if span > 0 else 0.0

    def pose(self, t: float) -> Pose:
        """``map <- base_link`` at ``t``."""
        return Pose.planar(0.0, 0.0, rad(self.heading(t)))


@dataclass
class Neck:
    """The arbiter's door and the neck under it: an aim asks for a point, resolved to pan and
    tilt from the cart's true pose and kept on it (resolved again every ``FOLLOW_S``, written
    past ``SETTLE_DEG``: the arbiter's follow); the head moves toward the written target on the
    aim's own ease (both axes, a newer target taken in flight) and is answered done at once (the
    answer is not what is measured here); the state is the pose ``reading_age_s`` ago (up to
    ``reading_spread_s`` more, a read's own), stamped with that reading's moment unless the
    arbiter is an older one (``stamped`` False: no ``head_stamp``), blind while the head moves
    and one frame period after it settles, ``blind_until`` on the board's clock."""

    cart: Cart
    pan: float = HOME_POSE[0]
    tilt: float = HOME_POSE[1]
    t: float = 0.0
    goal: tuple[float, float] | None = None
    target: tuple[float, float] = HOME_POSE
    limits: tuple[float, float] = (200.0, 650.0)
    v: list[float] = field(default_factory=lambda: [0.0, 0.0])
    poses: list[tuple[float, float, float]] = field(default_factory=lambda: [(0.0, *HOME_POSE)])
    aims: list[tuple[float, Point, Aim, Ease | None]] = field(default_factory=list)
    writes: list[tuple[float, float, float, str]] = field(default_factory=list)  # t, pan, tilt
    held: Point | None = None
    next_follow: float = 0.0
    holder: str = ""
    settled_s: float | None = 0.0
    reading_age_s: float = 0.02
    reading_spread_s: float = 0.0
    stamped: bool = True
    board0: float = WALL  # the board's clock at the simulation's 0 (chrony's skew: board0 - WALL)
    rng: random.Random = field(default_factory=lambda: random.Random(1015))

    def resolve(self, point: Point, t: float) -> tuple[float, float]:
        """The pan and tilt (degrees) that look at ``point`` with the cart where it truly is at
        ``t``: the arbiter's geometry and its clamp toward the head's own pan."""
        aim = MODEL.aim_at(self.cart.pose(t).to_frame(point))
        held = REACH.clamp(Aim(math.remainder(aim.pan_rad, math.tau), aim.tilt_rad),
                           from_pan=rad(self.pan))  # fmt: skip
        return deg(held.pan_rad), deg(held.tilt_rad)

    def advance(self, t: float) -> None:
        """The neck driven up to ``t``; the held point followed on the arbiter's step."""
        while self.t + 1e-9 < t:
            dt = min(STEP_S, t - self.t)
            self.t += dt
            if self.held is not None and self.t >= self.next_follow - 1e-9:
                self.next_follow = self.t + FOLLOW_S
                there = self.resolve(self.held, self.t)
                if max(abs(there[0] - self.target[0]), abs(there[1] - self.target[1])) > (
                    SETTLE_DEG
                ):
                    self.write(there, "follow")
            if self.goal is not None:
                speed, acc = self.limits
                self.pan, self.v[0] = axis(self.pan, self.v[0], self.goal[0], speed, acc, dt)
                self.tilt, self.v[1] = axis(self.tilt, self.v[1], self.goal[1], speed, acc, dt)
                if (self.pan, self.tilt) == self.goal and self.v == [0.0, 0.0]:
                    self.goal, self.settled_s = None, self.t
            self.poses.append((self.t, self.pan, self.tilt))

    def write(self, there: tuple[float, float], why: str) -> None:
        """One write of the target (the head moves toward it from where it is)."""
        self.goal = self.target = there
        self.settled_s = None
        self.writes.append((self.t, there[0], there[1], why))

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

    def aim(
        self, point: Point, *, speed: Speed, ttl_s: float, ease: Ease | None = None
    ) -> GazeAnswer:
        there = self.resolve(point, self.t)
        self.aims.append((self.t, point, Aim(rad(there[0]), rad(there[1])), ease))
        self.limits = (ease.speed_deg_s, ease.acc_deg_s2) if ease is not None else (200.0, 650.0)
        self.held, self.next_follow = point, self.t + FOLLOW_S
        self.write(there, "aim")
        self.holder = SPEAKER
        return GazeAnswer("done", head=Aim(rad(there[0]), rad(there[1])))

    def renew(self) -> int:
        return 1

    def release(self) -> None:
        self.holder, self.held = "", None

    def state(self) -> HeadView:
        age = self.reading_age_s + self.rng.uniform(0.0, self.reading_spread_s)
        read_s, pan, tilt = self.reading_at(self.t - age)
        until = None if self.settled_s is None else self.board0 + self.settled_s + FRAME_PERIOD_S
        blind = self.goal is not None or (until is not None and self.board0 + self.t < until)
        band = 5 if self.holder else HOME
        stamp = self.board0 + read_s if self.stamped else None
        return HeadView(Aim(rad(pan), rad(tilt)), self.holder, band, blind, until, False,
                        head_stamp=stamp)  # fmt: skip


@dataclass
class Channel:
    """The perception service: a frame exposed every ``period_s``, its record readable
    ``latency_s`` after the exposure; a look is the newest record past the last one given and
    exposed at ``not_before`` or later, or the next one waited for (the long poll: the clock
    moves on meanwhile, :attr:`wait`; up to ``wait_s``, then None), None while the service
    gives none (``down``); the person's face is where the camera's pose AT THE EXPOSURE (the
    cart's true heading, the head's pose) sees their point, with the detector's jitter, its
    width by its distance, unless it is turned ``away`` (records without it, by their
    exposure); ``others``' faces the same way (never turned away). Its stamps are on the board's
    clock (the neck's ``board0``)."""

    neck: Neck
    where: Where
    now: float = 0.0
    period_s: float = 0.1
    latency_s: float = 0.15
    wait_s: float = 0.4
    jitter_deg: float = 0.3
    shown: bool = True
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
            if n == 0 and (
                not self.shown or (self.away is not None and self.away[0] <= exposed < self.away[1])
            ):
                face = None
            if face is not None:
                faces.append(face)
        return FaceFrame(self.neck.board0 + exposed, tuple(faces), seq=k)

    def camera(self, exposed: float) -> Pose:
        """``map <- camera_link`` at the exposure: the cart's true pose, the head's then."""
        pan, tilt = self.neck.pose_at(exposed)
        return self.neck.cart.pose(exposed).then(MODEL.camera(Aim(rad(pan), rad(tilt))))

    def _face(self, where: Where, exposed: float) -> FaceSighting | None:
        """A person's face in the frame exposed at ``exposed``, None when out of the view."""
        camera = self.camera(exposed)
        point = person_at(where, exposed)
        pan, tilt = MODEL.angles(camera, point)
        pan += self.rng.gauss(0.0, self.jitter_deg)
        tilt += self.rng.gauss(0.0, self.jitter_deg)
        seen = abs(pan) <= EYE_PAN_DEG and abs(tilt) <= EYE_TILT_DEG
        eye = camera.to_map(MODEL.ray(0.0, 0.0)[0])
        width = FACE_PX_M / eye.distance(point)
        return FaceSighting(pan, tilt, width, 0.8) if seen else None


@dataclass
class Placed:
    """One record the person engaged was seen in: its exposure, how far (pan, degrees, as the
    cart saw it then) the point the gaze placed is from their true point, and the head's pan at
    the exposure and at the read."""

    exposed: float
    error_deg: float
    head_then: float
    head_read: float


def bearing_of(cart: Cart, point: Point, t: float) -> float:
    """The pan that looks at ``point`` with the cart where it truly is at ``t`` (degrees)."""
    aim = MODEL.aim_at(cart.pose(t).to_frame(point))
    return deg(math.remainder(aim.pan_rad, math.tau))


@dataclass
class Sim:
    """The engine, the neck, the channel, the array, TF and the cart on one simulated clock."""

    machine: SpeakerGaze
    neck: Neck
    channel: Channel
    doa: FakeDoa
    tf: FakeTf
    events: list[tuple[str, float, dict[str, Any]]]
    t: float = 0.0
    due: float | None = None
    trace: list[tuple[float, float, float, str]] = field(default_factory=list)  # head, person
    placed: list[Placed] = field(default_factory=list)

    @property
    def cart(self) -> Cart:
        return self.neck.cart

    def run(self, until: float) -> None:
        """The clock to ``until``: the neck driven, the machine ticked whenever due, the head's
        pan and the pan that looks at the person traced, each record they were placed by kept."""
        while self.t < until - 1e-9:
            self.t = round(self.t + TICK_S, 6)
            self.neck.advance(self.t)
            self.channel.now = self.t
            if self.due is None or self.t >= self.due - 1e-9:
                seq, at = self.channel.seq, self.t
                self.due = self.machine.tick(at)
                self._placed(seq, at)
            person = bearing_of(self.cart, person_at(self.channel.where, self.t), self.t)
            self.trace.append((self.t, self.neck.pan, person, self.machine.phase))

    def moved_to(self, t: float) -> None:
        """The clock moved on to ``t`` while the machine waited on a long poll (the neck too)."""
        if t > self.t:
            self.neck.advance(t)
            self.t = self.channel.now = t

    def _placed(self, before: int, at: float) -> None:
        person = self.machine.person
        seq = self.channel.seq
        if seq == before or person is None or person.source != "face":
            return
        if self.machine.seen_at.get(person.id) != at:
            return
        exposed = self.channel.exposed[seq]
        true = person_at(self.channel.where, exposed)
        error = bearing_of(self.cart, person.point, exposed) - bearing_of(self.cart, true, exposed)
        self.placed.append(
            Placed(exposed, error, self.neck.pose_at(exposed)[0], self.neck.pose_at(self.t)[0])
        )

    def speech(self, bearing: float, start: float, end: float) -> list[DoaReading]:
        """The array's readings of a voice at ``bearing`` (the cart's frame) on the board's
        clock, ten a second."""
        steps = round((end - start) * 10)
        b0 = self.neck.board0
        return [DoaReading(b0 + start + k / 10, array(bearing), True) for k in range(steps + 1)]

    def wake(self, bearing: float) -> None:
        """The robot's name from ``bearing`` over 0..1.2 s, read at 1.3: the conversation
        opens on it."""
        self.run(1.3)
        room = replace(self.speech(30, 1.3, 1.3)[0], speech=False)
        self.doa.readings += [*self.speech(bearing, 0.0, 1.2), room]
        self.machine.opened(Heard(0.0, 1.2, "Пепин", named=True), self.t)
        self.due = self.t

    def rows(self, kind: str) -> list[dict[str, Any]]:
        return [f for k, _, f in self.events if k == kind]

    def aims_between(self, t0: float, t1: float, why: str | None = None) -> list[dict[str, Any]]:
        return [f for k, t, f in self.events
                if k == "gaze_reaim" and t0 <= t <= t1
                and (why is None or f["why"] == why)]  # fmt: skip


def sim(
    where: Where,
    *,
    wheels: Wheels | None = None,
    skew_s: float = 0.0,
    cart: Cart | None = None,
    **knobs: Any,
) -> Sim:
    """The shipped knobs over the simple map, the head alone (arc 360: no body turn), ``knobs``
    pinned over them; ``wheels`` the cart's (none by default); ``skew_s`` how far the board's
    clock (the frames', the encoders', the array's and TF's stamps) runs ahead of this Mac's
    (chrony's error); ``cart`` its true turn (none by default)."""
    neck = Neck(cart if cart is not None else Cart(), board0=WALL + skew_s)
    channel = Channel(neck, where)
    doa = FakeDoa()
    events: list[tuple[str, float, dict[str, Any]]] = []
    settings = replace(SpeakerSettings(doa_map=MAP), **{"arc_deg": 360.0, **knobs})
    holder: list[Sim] = []

    def head_at(stamp: float) -> Aim:
        pan, tilt = neck.pose_at(stamp - neck.board0)
        return Aim(rad(pan), rad(tilt))

    tf = FakeTf(
        MODEL,
        now=lambda: neck.board0 + holder[0].t,
        cart_at=lambda stamp: (0.0, 0.0, neck.cart.heading(stamp - neck.board0)),
        head_at=head_at,
    )
    machine = SpeakerGaze(
        settings, door=neck, doa=doa, finder=channel, wheels=wheels or NoWheels(), face=None,
        reach=REACH, head=MODEL, poses=tf, report=lambda line: None,
        events=lambda kind, now, **f: events.append((kind, now, f)), series=FakeSeries(),
    )  # fmt: skip
    machine.spawn = lambda call: call()
    s = Sim(machine, neck, channel, doa, tf, events)
    holder.append(s)
    channel.wait = s.moved_to
    return s


def walk(start: float, speed: float, t0: float, t1: float, tilt: float = 5.0) -> Where:
    """A person at ``start`` deg who walks round the cart at ``speed`` deg/s from ``t0`` to
    ``t1`` (the map's degrees)."""

    def where(t: float) -> tuple[float, float]:
        return start + speed * min(max(t - t0, 0.0), t1 - t0), tilt

    return where


# ---- the pursuit ---------------------------------------------------------------------------------
@pytest.mark.parametrize("latency_s", [0.15, 0.25])
def test_a_person_walking_at_30_deg_s_at_2_m_is_followed_within_the_lead_cap_never_lost(
    latency_s: float,
) -> None:
    """Artem 2 m away walks 120 deg round the robot at 30 deg/s while it speaks, each record
    reaching the gaze ``latency_s`` after its exposure. Every record is taken (ten a second
    while it speaks too), most of them exposed while the head moves, each placed in the map
    with the camera's pose at its exposure (TF's); from a second into the walk the head stays
    within lead_max_deg (10) of him, he is never lost, the aims (his point ahead along his
    velocity) go out without waiting at most one a record, the pan always forward; he stops and
    the head settles on him. One pose request a record."""
    s = sim(walk(-40.0, 30.0, 2.5, 6.5))
    s.channel.latency_s = latency_s
    s.wake(-40.0)
    s.run(2.5)
    assert s.machine.phase == "track" and abs(s.neck.pan + 40.0) < 3.0
    s.machine.speaking(True, s.t)
    s.machine.voiced(s.t + 0.2)
    frames, asked = s.machine.counts["frames"], len(s.tf.asked)
    s.run(5.5)
    assert s.machine.counts["frames"] - frames >= 28, "ten a second while the robot speaks"
    assert len(s.tf.asked) - asked == s.machine.counts["frames"] - frames, "a request a record"
    s.machine.speaking(False, s.t)
    s.run(8.0)
    assert {phase for t, _, _, phase in s.trace if t >= 2.5} == {"track"}, "never lost"
    assert not [f for f in s.rows("gaze_lost") if f.get("what") == "face"]
    off = [abs(head - person) for t, head, person, _ in s.trace if 3.5 <= t <= 6.5]
    assert max(off) <= 10.0, max(off)
    assert abs(s.neck.pan - 80.0) <= 6.0, "he stopped: the head settles on him"
    pursuit = s.aims_between(2.5, 6.6, LEAD)
    records = len([p for p in s.placed if 2.5 <= p.exposed <= 6.6])
    assert 25 <= len(pursuit) <= records, (len(pursuit), records)
    assert {f["status"] for f in pursuit} == {"sent"} and {f["speed"] for f in pursuit} == {"glide"}
    pans = [f["pan_deg"] for f in pursuit]
    assert all(b >= a for a, b in itertools.pairwise(pans)), "the pan never steps back"
    assert max(f["lead_deg"] for f in pursuit) <= 10.0
    assert 27.0 <= max(f["speed_deg_s"] for f in pursuit) <= 33.0, "his speed, fitted"
    assert all(len(f["point"]) == 3 and f["source"] == "face" for f in pursuit)
    moving = [p for p in s.placed if 3.0 <= p.exposed <= 6.5 and s.neck.speed_at(p.exposed) > 5.0]
    assert len(moving) >= 25, "most frames of the walk exposed while the head moved, all kept"
    assert s.machine.counts["frames_unposed"] == 0 and s.machine.counts["frames_unplaced"] == 0


def test_each_box_is_placed_with_the_camera_s_pose_at_its_exposure_not_at_its_read() -> None:
    """The record reaches the gaze 0.15-0.25 s after its exposure: placed with the camera's
    pose then (TF's, the neck's encoders and the cart), the person's point is right within the
    detector's jitter (a walker at 30 deg/s placed within a degree, 0.3 on average); the head's
    pose at the read would put him 3-7 deg off while the head glides."""
    s = sim(walk(-40.0, 30.0, 2.5, 6.5))
    s.wake(-40.0)
    s.run(7.0)
    walking = [p for p in s.placed if 3.0 <= p.exposed <= 6.5]
    assert len(walking) >= 30
    errors = [abs(p.error_deg) for p in walking]
    assert max(errors) <= 1.0 and sum(errors) / len(errors) <= 0.4, errors
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
    aims = [(t, aim) for t, _, aim, _ in s.neck.aims if t >= 2.5]
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
    s.channel.shown = False  # he turns away
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
    he was, so he used to become a stranger and the head lost him. He is matched where the line
    over his sightings in the map puts him at that frame's stamp (THE MATCH: at most the fit's
    own window past the newest sighting): never lost, the head on him."""
    s = sim(walk(-60.0, speed, 2.5, 2.5 + 120.0 / speed))
    s.channel.away = (3.5, 3.5 + gap)
    s.wake(-60.0)
    s.run(8.0)
    assert not [f for f in s.rows("gaze_lost") if f.get("what") == "face"]
    assert {phase for t, _, _, phase in s.trace if t >= 2.5} == {"track"}, "never lost"
    rest = [abs(head - person) for t, head, person, _ in s.trace if t >= 7.0]
    assert max(rest) <= 6.0, max(rest)


# ---- the body turns: TF keeps the head on him, whatever the wheels claim ------------------------
TURN = (2.5, 4.0)  # the cart turns over this span...
TRUE_DEG = 10.0  # ...by 10 deg in TF, while Nav2's count (its wheels slipping) claims 30


def crowd_turn(
    who: str,
    *,
    start: float = 30.0,
    true_deg: float = TRUE_DEG,
    claimed_deg: float = 30.0,
    turn: tuple[float, float] = TURN,
    away: tuple[float, float] | None = None,
    others: tuple[float, ...] = (),
    told_s: float = 0.1,
    status: int = 4,
) -> Sim:
    """Artem stands at ``start`` deg (2 m away, a 40 px face) and the cart turns ``true_deg``
    toward him in TF over ``turn`` while Nav2 counts ``claimed_deg`` (the wheels slip): ``who``
    turns it, the gaze's own spin (the arc's rule, +-25 deg: one spin of his bearing asked of the
    goal server at the wake, accepted at once, its end told ``told_s`` after the turn) or the
    turn tool's (the arc never turns the body: the loop's drive notice and keep_person 0.1 s
    before the turn, the drive's end ``told_s`` after it). His face is turned ``away`` over that
    span of exposures; ``others`` stand at their bearings (2 m away, faces never turned away).
    Nav2's end carries ``status`` (4 succeeded, 6 aborted) and TF's turn. Woken by his name, run
    to 2 s after the turn."""
    t0, t1 = turn
    own = who == "own"
    s = sim(lambda t: (start, 5.0), wheels=FakeWheels(), body_turn="spin",
            arc_deg=50.0 if own else 360.0, cart=Cart(t0, t1, true_deg))  # fmt: skip
    s.channel.away = away
    s.channel.others = tuple((lambda at: lambda t: (at, 5.0))(other) for other in others)
    spins = FakeSpins()
    s.machine.spins = spins
    s.wake(start)
    if own:
        assert spins.asked == [pytest.approx(start)], "past the arc: one spin of his bearing"
        spins.tell("accepted", run=1, deg=start)
        s.run(t0 + 0.5 * (t1 - t0))
        spins.tell("feedback", spun_deg=0.5 * claimed_deg)
        s.run(t1 + told_s)
        spins.tell("done", run=1, status=status, seconds=t1 - t0, turned_deg=true_deg,
                   spun_deg=claimed_deg)  # fmt: skip
    else:
        s.run(t0 - 0.1)
        s.machine.drive(True, s.t)
        assert s.machine.keep_person(s.t, f"turn: {claimed_deg:g}") is None
        s.run(t1 + told_s)
        s.machine.drive(False, s.t)
    s.run(t1 + 2.0)
    assert len(spins.asked) == (1 if own else 0), "no turn but the one"
    return s


@pytest.mark.parametrize("who", ["own", "tool"])
def test_while_the_body_turns_on_slipping_wheels_tf_keeps_the_head_on_him(who: str) -> None:
    """The cart turns 10 deg toward Artem in TF while Nav2 counts 30 (the wheels slip), by the
    gaze's own spin or the turn tool's: every record is taken while it turns, each placed in the
    map, and the arbiter keeps the head on his point from TF's pose: within 2 deg of him
    throughout, the head written by the follow (no aim of the gaze's: he stands; nothing read of
    any odometry, whose 30 would have put him 20 deg off). He is never lost, and the head rests
    on him at +20, the turned nose 10 deg nearer."""
    s = crowd_turn(who)
    t0, t1 = TURN
    off = [abs(head - person) for t, head, person, _ in s.trace if t >= t0]
    assert max(off) <= 2.0, max(off)
    assert s.aims_between(t0, t1) == [], "he stands: the gaze asks nothing while the body turns"
    follows = [w for w in s.neck.writes if w[3] == "follow" and t0 <= w[0] <= t1 + 0.1]
    assert len(follows) >= 5, "the arbiter writes the head as TF turns"
    assert s.machine.counts["frames_turning"] >= 10 * (t1 - t0), "every record while it turns"
    assert s.machine.counts["frames_unposed"] == 0
    assert {phase for t, _, _, phase in s.trace if t >= 2.0} == {"track"}, "never lost"
    assert abs(s.neck.pan - 20.0) <= 1.5


@pytest.mark.parametrize("who", ["own", "tool"])
def test_wheels_on_the_stand_spinning_turn_no_tf_and_move_no_head(who: str) -> None:
    """2026-10-15, the front wheels on the stand: Nav2's Spin was taken and its count ran to 30
    deg while the cart did not turn at all (TF: 0), and the gaze re-aimed the head off him 143
    times by the odometry. Now nothing moves the head: no aim of the gaze's, no write of the
    arbiter's, the head within the detector's jitter of where it stood on him; the gaze's own
    spin, which Nav2 aborts (it reads TF), ends at TF's 0 deg (nothing moved: asked again at
    his next words, never again before them)."""
    s = crowd_turn(who, true_deg=0.0, claimed_deg=30.0, status=6)
    t0, t1 = TURN
    pans = [head for t, head, _, _ in s.trace if t >= t0]
    assert max(pans) - min(pans) <= 0.5, (min(pans), max(pans))
    assert s.aims_between(t0, t1 + 2.0) == [], "no aim"
    assert [w for w in s.neck.writes if w[0] >= t0] == [], "no write"
    if who == "own":
        end = [f for f in s.rows("gaze_spin") if f.get("end")][-1]
        assert end["turned_deg"] == 0.0 and s.machine.body_waits, "TF's 0: nothing moved"


@pytest.mark.parametrize("who", ["own", "tool"])
def test_while_the_frames_show_nobody_tf_keeps_the_head_on_his_point(who: str) -> None:
    """The same 10 deg turn over 2.5 s, his face turned away for a second of it: no frame
    places him, and none has to: the arbiter keeps the head on his point by TF throughout
    (within 2 deg), no glide, no seek; his face back, the frames find him where the head
    already looks. He is never lost (a second without him, under face_lost_s)."""
    turn = (2.5, 5.0)
    away = (2.8, 3.8)
    s = crowd_turn(who, away=away, turn=turn)
    gap = [abs(head - person) for t, head, person, _ in s.trace if t >= turn[0]]
    assert max(gap) <= 2.0, max(gap)
    assert s.aims_between(turn[0], turn[1] + 0.5) == []
    assert {phase for t, _, _, phase in s.trace if t >= 2.0} == {"track"}, "never lost"


@pytest.mark.parametrize("who", ["own", "tool"])
def test_at_the_stop_the_head_lands_on_him_with_no_lead_of_the_turn_s(who: str) -> None:
    """The cart turns 30 deg toward Artem in 1.5 s (his bearing 20 deg/s from the cart) and the
    turn's end is told 0.3 s after it stopped. The lead is his own motion's (none: he stands),
    never the cart's turn, so nothing carries the head past him at the stop: at most a degree
    and a half past (the arbiter's follow, a step behind the turn), at rest on him within a
    second."""
    t1 = 4.0
    s = crowd_turn(who, start=40.0, true_deg=30.0, claimed_deg=60.0, turn=(2.5, t1), told_s=0.3)
    past = [person - head for t, head, person, _ in s.trace if t >= t1]
    assert max(past) <= 1.5, max(past)
    rest = [abs(head - person) for t, head, person, _ in s.trace if t >= t1 + 1.0]
    assert max(rest) <= 1.0, max(rest)


def test_a_bystander_the_frames_show_as_himself_is_never_taken_for_him_in_a_turn() -> None:
    """A friend stands 35 deg right of Artem, both seen from the wake on. Artem looks away
    just before the turn tool turns the cart 45 deg toward him (Nav2 counting 90), so no frame
    of the first part of the turn places him: his point stays where the map has it, the friend
    keeps his own record, and Artem is found again where he is when he looks back, and kept
    to the end."""
    s = crowd_turn("tool", start=45.0, true_deg=45.0, claimed_deg=90.0, turn=(2.5, 5.5),
                   away=(2.2, 3.0), others=(10.0,))  # fmt: skip
    person = s.machine.person
    assert person is not None and s.machine.bearing is not None
    assert abs(deg(s.machine.bearing)) <= 2.0, "Artem, where the turn put him"
    rest = [abs(head - person_) for t, head, person_, _ in s.trace if t >= 6.5]
    assert max(rest) <= 2.0, max(rest)
    friend = [
        p for p in s.machine.people.people
        if p is not person and abs(bearing_of(s.cart, p.point, s.t) + 35.0) < 3.0
    ]  # fmt: skip
    assert friend, "the friend keeps his own record"


def test_after_a_gap_in_a_turn_his_point_is_where_he_is_and_the_head_never_strays() -> None:
    """The turn tool turns the cart 60 deg toward Artem at 30 deg/s (Nav2 counting 120); his
    face is turned away for a second of it, a man standing 45 deg beyond him. The odometry put
    him 30 deg past where he was, the head followed it, and the man came into view right there
    (2026-10-14); his point is the map's now (TF's), so the head never leaves him, the man never
    comes into view (45 deg off the axis: past the eye's 41), and Artem is matched where he is
    when he looks back; the head ends on him within a degree; never lost."""
    s = crowd_turn("tool", start=60.0, true_deg=60.0, claimed_deg=120.0, turn=(2.5, 4.5),
                   away=(3.3, 4.3), others=(15.0,))  # fmt: skip
    assert s.machine.bearing is not None and abs(deg(s.machine.bearing)) <= 2.0
    rest = [abs(head - person) for t, head, person, _ in s.trace if t >= 6.0]
    assert max(rest) <= 1.0, max(rest)
    assert {phase for t, _, _, phase in s.trace if t >= 2.5} == {"track"}, "never lost"
    assert len(s.machine.people.people) == 1, "the man beyond him: never in view"


def test_at_the_wake_on_its_own_spin_no_frame_before_the_head_looks_his_way_places_him() -> None:
    """Artem calls from 45 deg left, past the arc, a friend 16 deg right of him: the gaze's
    own spin starts at the wake and every record is taken from then, the first ones exposed
    while the head still points home and shows only the friend, 16 deg off Artem's voice
    (within the match): those place nobody on Artem (frames_away); the first frame that looks
    his way finds him, and he is kept through the turn."""
    s = crowd_turn("own", start=45.0, true_deg=45.0, claimed_deg=90.0, turn=(2.5, 5.5),
                   others=(29.0,))  # fmt: skip
    first = s.rows("gaze_face")[0]
    assert abs(first["pan_deg"] - 45.0) <= 3.0, first
    assert s.machine.counts["frames_away"] >= 1
    assert s.machine.bearing is not None and abs(deg(s.machine.bearing)) <= 2.0


# ---- one clock, the board's ----------------------------------------------------------------------
READS = [(0.02, 0.0), (0.05, 0.0), (0.02, 0.03)]  # how old a reading is when read: 20, 50, 20-50 ms


@pytest.mark.parametrize("skew_s", [-0.5, 0.0, 0.5])
@pytest.mark.parametrize(("age_s", "spread_s"), READS)
def test_a_box_is_placed_by_tf_at_its_stamp_whatever_the_reads_age_or_chrony(
    skew_s: float, age_s: float, spread_s: float
) -> None:
    """Every stamp the gaze places by is the board's (the frame's exposure, the array's
    readings, TF's), so each box of a walk at 30 deg/s is placed within the detector's jitter
    of him (0.3 deg on average) however old the arbiter's reading is when read and whether the
    board's clock runs half a second behind or ahead of this Mac's (chrony's error): no frame
    unposed, never lost."""
    s = sim(walk(-40.0, 30.0, 2.5, 6.5), skew_s=skew_s)
    s.neck.reading_age_s, s.neck.reading_spread_s = age_s, spread_s
    s.wake(-40.0)
    s.run(7.5)
    walking = [p for p in s.placed if 3.0 <= p.exposed <= 6.5]
    assert len(walking) >= 30
    errors = [abs(p.error_deg) for p in walking]
    assert max(errors) <= 1.0 and sum(errors) / len(errors) <= 0.4, errors
    assert s.machine.counts["frames_unposed"] == 0
    assert {phase for t, _, _, phase in s.trace if t >= 2.5} == {"track"}, "never lost"


def test_a_re_centre_on_a_person_standing_still_reads_no_walk() -> None:
    """Artem stands 10-25 deg from where his voice put him, so the head turns onto his face
    while he stands (a re-centre). Placed by TF's poses at each exposure, every box of the move
    is within a degree of him, so no walk is read: at most two re-aims a case (none in most),
    the head resting within the dead band of him (the 2026-10-14 review: dated by the reads,
    a -6 deg/s phantom the head chased with up to 24 re-aims)."""
    people = ((20.0, 10.0), (30.0, 5.0), (-20.0, 0.0), (15.0, -5.0), (-35.0, -10.0))
    rests: list[float] = []
    for (person, voice), (age_s, spread_s) in itertools.product(people, READS):
        s = sim(walk(person, 0.0, 0.0, 0.0))
        s.neck.reading_age_s, s.neck.reading_spread_s = age_s, spread_s
        s.wake(voice)
        s.run(6.0)
        rows = [f for f in s.rows("gaze_reaim") if f["why"] in (LEAD, MOVED)]
        case = (person, voice, age_s, spread_s)
        assert len(rows) <= 2, (case, [f["speed_deg_s"] for f in rows])
        assert max(abs(p.error_deg) for p in s.placed) <= 1.0, case
        rests.append(abs(s.neck.pan - person))
    assert max(rests) <= 6.0, "within the dead band"
    assert sum(rest <= 1.5 for rest in rests) >= len(rests) // 2, rests


def test_an_arbiter_without_the_reading_s_stamp_places_every_frame_all_the_same() -> None:
    """A gaze node started before ``head_stamp`` sends none: nothing of the placement reads the
    arbiter's state any more (TF's poses at the frames' stamps place them), so every frame is
    posed and he is followed as closely as with it."""
    s = sim(walk(-40.0, 30.0, 2.5, 6.5))
    s.neck.stamped = False
    s.wake(-40.0)
    s.run(7.5)
    walking = [p for p in s.placed if 3.0 <= p.exposed <= 6.5]
    errors = [abs(p.error_deg) for p in walking]
    assert len(walking) >= 30 and s.machine.counts["frames_unposed"] == 0
    assert max(errors) <= 1.0 and sum(errors) / len(errors) <= 0.4, errors
    assert {phase for t, _, _, phase in s.trace if t >= 2.5} == {"track"}, "never lost"


def test_the_board_s_now_is_the_newest_stamp_seen_never_this_mac_s() -> None:
    """The gaze ages its people on the board's clock: its "now" is the newest of the stamps it
    has seen (the arbiter's reading, a frame's exposure, the array's reading, TF's answer),
    here half a second ahead of this Mac's, never this Mac's clock."""
    clock = BoardClock()
    assert clock.now() is None
    for stamp in (100.0, None, 0.0, 99.0, 100.5):
        clock.saw(stamp)
    assert clock.now() == 100.5, "the newest; none, 0 and older ones change nothing"
    s = sim(walk(0.0, 0.0, 0.0, 0.0), skew_s=0.5)
    s.wake(0.0)
    s.run(3.0)
    board = s.machine.clock.now()
    assert board is not None and s.t - 0.3 <= board - (WALL + 0.5) <= s.t + 1e-9


def test_a_second_of_conversation_asks_the_poses_once_a_record_and_once_an_utterance() -> None:
    """The cost on the goal server: one ``pose_at`` request a record that shows someone (the
    camera and the cart at its exposure), one an utterance (the cart at each of its voiced
    readings), none for a record of nobody: ten a second while someone is tracked."""
    s = sim(walk(20.0, 0.0, 0.0, 0.0))
    s.wake(20.0)
    assert len(s.tf.asked) == 1 and len(s.tf.asked[0][0]) == 13, "the wake's voiced readings"
    s.run(2.0)
    asked = len(s.tf.asked)
    s.run(12.0)
    per_s = (len(s.tf.asked) - asked) / 10.0
    assert 9.0 <= per_s <= 10.5, per_s
    s.channel.shown = False  # nobody in the records: nothing to place
    s.run(13.0)
    asked = len(s.tf.asked)
    s.run(13.6)
    assert len(s.tf.asked) == asked, "a record of nobody asks nothing"


# ---- the parts -----------------------------------------------------------------------------------
def test_a_pursuit_fits_the_velocity_in_the_map_on_the_frames_stamps() -> None:
    p = Pursuit()
    p.add(100.0, Point(1.0, 0.0, 1.0), "face")
    p.add(100.1, Point(1.0, 0.05, 1.0), "face")
    assert p.velocity() == (0.0, 0.0, 0.0) and p.ahead(100.5) == (0.0, 0.0, 0.0), "two: none"
    p.add(100.1, Point(9.0, 9.0, 9.0), "face")  # the same frame again
    p.add(100.2, Point(1.0, 0.1, 1.0), "face")
    vx, vy, vz = p.velocity()
    assert (vx, vy, vz) == pytest.approx((0.0, 0.5, 0.0))
    assert p.ahead(100.4) == pytest.approx((0.0, 0.1, 0.0))
    for k in range(3, 12):
        p.add(100.0 + 0.1 * k, Point(1.0, 0.1, 1.0), "face")
    assert p.velocity() == pytest.approx((0.0, 0.0, 0.0), abs=1e-9) and len(p.sightings) == 6
    p.add(101.2, Point(1.0, 0.2, 1.2), "body")
    assert p.source == "body" and len(p.sightings) == 1, "another source: a new fit"
    p.reset()
    assert p.sightings == [] and p.last_s is None


def test_a_pursuit_expects_the_point_on_its_line_as_far_as_the_horizon_given() -> None:
    """THE MATCH's place: the fitted line at a frame's stamp, extrapolated at most the horizon
    past the newest sighting, nothing before a velocity is fitted."""
    p = Pursuit()
    p.add(100.0, Point(2.0, 0.0, 1.0), "face")
    p.add(100.1, Point(2.0, 0.1, 1.0), "face")
    assert p.expected(100.3, 0.5) is None, "two sightings: no line yet"
    p.add(100.2, Point(2.0, 0.2, 1.0), "face")
    there = p.expected(100.3, 0.5)
    assert there is not None and there.y == pytest.approx(0.3)
    held = p.expected(101.5, 0.5)
    assert held is not None and held.y == pytest.approx(0.7), "0.5 s past the newest"


def test_the_head_model_places_a_face_and_the_arbiter_s_geometry_finds_it_again() -> None:
    """The real neck's lever arms and the real eye: a face the camera sees at some angles,
    placed along the ray they name at its range, is seen at the same angles again (the ray's
    inverse), and the aim the arbiter resolves for the point puts it on the link's axis."""
    camera = Pose.planar(1.0, -0.5, rad(30.0)).then(MODEL.camera(Aim(rad(20.0), rad(10.0))))
    origin, direction = MODEL.ray(12.0, -7.0)
    eye = camera.to_map(origin)
    d = camera.direction(direction)
    point = eye.plus(d[0] * 2.0, d[1] * 2.0, d[2] * 2.0)
    assert MODEL.angles(camera, point) == pytest.approx((12.0, -7.0))
    base = Pose.planar(1.0, -0.5, rad(30.0))
    aim = MODEL.aim_at(base.to_frame(point))
    looked = base.then(MODEL.camera(aim))
    pan, tilt = MODEL.angles(looked, point)
    assert abs(pan) < 1.0 and abs(tilt) < 1.5, "the eye sits 3 cm off the link's axis"
