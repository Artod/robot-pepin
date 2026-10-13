"""The body parts as the gesture engine plays them (:class:`pepin.expression.executors.Executor`).

* :class:`ArmPart`: a clip sampled into joint knots over a base pose (rest, or the pose the
  last trajectory held: a unit), stretched in time where it would break the arm's caps (scaled
  by ``speed_scale``: speeds by it, accelerations by its square, as a clip stretched by its
  inverse moves), checked against the limits, and sent to the board in one
  ``arm_trajectory``; the board plays it (joining it from where the arm is by its own
  approach), then rests and lets go of the torque, or holds the last pose (``Clip.then``). Its
  plan (:class:`Planned`) says when the stroke lands: the approach plus the stretched stroke. A
  clip outside the limits is refused here with the reason, never clamped (the board would
  refuse it too). Fading the gesture layer is ``arm_rest``. B owns the arm's units here.
* :class:`FacePart`: a clip's events as timed expressions of the head server's face.
* :class:`HeadPart`: a clip's keyframes as a relative ``clip`` overlay of the gaze arbiter
  (:mod:`pepin.expression.head`, :func:`clip_look`) on top of whatever holds the head (the
  speaker gaze's look on the person, or home), posted through the arbiter's door
  (:class:`pepin.expression.door.OverlayDoor`) on a thread of its own; fading a layer is that
  layer's source released over the fade.
* :class:`Absent`: a part that cannot play, with the reason.
"""

from __future__ import annotations

import itertools
import logging
import math
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from pepin.arm_driver import ARRIVED_DEG
from pepin.arm_motion import ArmMotion
from pepin.expression.animation import Clip, Layer, PartName
from pepin.expression.executors import FaceChannel, Outcome
from pepin.expression.head import BAND, GESTURE, IDLE_LOOK, ClipKey, ClipLook, ClipTarget
from pepin.expression.mix import (
    PREP_CAPS,
    STRETCH_MARGIN,
    check_knots,
    fit_preparation,
    min_jerk_time,
    prep_min_s,
    rebased,
    sample,
    stretch_to_caps,
    track_value,
)
from pepin.expression.trajectory import JOINTS, ArmLink, Knot, Trajectory
from pepin.neck import NeckMotion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Planned:
    """A clip as the board will play it: the trajectory, the board's approach from where the arm
    is to its first knot (seconds; 0 when it is there), the factor the caps stretched the clip
    by (1: none), and when its stroke lands after the board takes it (the approach plus the
    stretched stroke): the one stroke time the engine schedules and logs."""

    trajectory: Trajectory
    approach_s: float
    stretch: float
    stroke_s: float

    @property
    def seconds(self) -> float:
        """The whole motion: the approach and the trajectory."""
        return self.approach_s + self.trajectory.knots[-1].t_s


class ArmPart:
    """The arm through the board's base server (an :class:`pepin.expression.trajectory.ArmLink`).

    A UNIT (DESIGN.md §7.10): a clip sent with ``then: hold`` leaves the arm at its last knot for
    its ``hold_s``; the next clip within it plans from that knot (:attr:`held`, the board's goal,
    never the encoders, which sag under the torque limit), started there instead of at rest
    (:func:`pepin.expression.mix.rebased`), with no approach. Past the hold the board has taken
    the arm to rest by itself, and the next clip plans from rest again."""

    part: PartName = "arm"

    def __init__(
        self,
        link: ArmLink,
        motion: ArmMotion,
        *,
        latency_s: float = 0.06,
        speed_scale: float = 1.0,
        knot_hz: float = 25.0,
        source: str = "voice.gesture",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``motion``: config/arm.json's (rest pose, limits, caps); ``speed_scale`` scales the
        caps a clip is checked against and travels with each trajectory (the board's profile
        and its way to rest); ``knot_hz``: knots a second (the board interpolates between);
        ``clock``: when a hold lapses."""
        self.link, self.motion = link, motion
        self._latency_s = latency_s
        self.speed_scale = speed_scale
        self.knot_hz = knot_hz
        self.source = source
        self.clock = clock
        self._ids = itertools.count(1)
        self.base = dict(motion.rest_deg)
        self.held: dict[str, float] | None = None  # the pose the last trajectory holds (a unit)
        self.held_until = float("-inf")  # when the board lets that hold go (engine clock)

    @property
    def latency_s(self) -> float:
        """The WiFi hop and the board's next tick."""
        return self._latency_s

    def caps(self, share: float = 1.0) -> tuple[float, float]:
        """The speed and acceleration caps at the speed scale, times ``share``."""
        scale = self.speed_scale
        motion = self.motion
        return motion.max_speed_deg_s * scale * share, motion.max_acc_deg_s2 * scale**2 * share

    def ready(self) -> str | None:
        """Why the arm cannot play: no state line has read it yet, or its servos are silent."""
        status = self.link.status()
        if status is None:
            return "no word from the arm yet (is the base server reading it?)"
        if status.phase == "silent":
            return f"the arm's servos are silent: {status.error or 'no reply'}"
        return None

    def holding(self) -> dict[str, float] | None:
        """The pose a unit holds now (None: none, or its hold has lapsed)."""
        if self.held is not None and self.clock() > self.held_until:
            self.held = None
        return self.held

    def plan(
        self,
        clip: Clip,
        name: str = "",
        *,
        base: Mapping[str, float] | None = None,
        start: Sequence[float] | None = None,
    ) -> Planned | str:
        """``clip`` as the trajectory to send over ``base`` (the rest pose by default; a unit's
        next stroke: the held pose, the clip started there) and when its stroke lands, or why it
        cannot be played. The board's approach is from ``start`` (the arm's angles when the board
        takes it; None: the newest state line's)."""
        if clip.part != "arm":
            return f"an {clip.part} clip is not the arm's"
        motion = self.motion
        max_speed, max_acc = self.caps()
        pose = dict(base) if base is not None else self.base
        if base is not None:
            clip = rebased(clip, pose, self.base)
        # a clip stretched by 1 / speed_scale needs no more knots than at full speed
        sampled = sample(clip, pose, JOINTS, self.knot_hz * min(1.0, self.speed_scale))
        knots = stretch_to_caps(sampled, max_speed * STRETCH_MARGIN, max_acc * STRETCH_MARGIN)
        # a stroke too fast for the caps takes longer rather than being refused
        if knots[-1][0] > motion.max_duration_s:
            return f"{knots[-1][0]:.1f} s is longer than the board's {motion.max_duration_s:.0f} s"
        refusal = check_knots(knots, motion.limits(), max_speed, max_acc, JOINTS)
        if refusal is not None:
            return refusal
        span = sampled[-1][0] - sampled[0][0]
        stretch = (knots[-1][0] - knots[0][0]) / span if span > 0 else 1.0
        trajectory = Trajectory(
            f"{name or 'gesture'}-{next(self._ids)}",
            self.source,
            tuple(Knot(round(t, 3), tuple(round(d, 2) for d in deg)) for t, deg in knots),
            clip.then,
            clip.hold_s if clip.then == "hold" else 0.0,
            name,
            self.speed_scale,
        )
        approach = self.approach_s(knots[0][1], start)
        return Planned(trajectory, approach, stretch, approach + stretch * clip.stroke_s)

    def approach_s(
        self, first: tuple[float, ...] | list[float], start: Sequence[float] | None = None
    ) -> float:
        """The board's approach to ``first`` from ``start`` (None: where the arm is by its
        newest state line): none within ``ARRIVED_DEG`` of it, or when the arm's place is
        unknown."""
        if start is None:
            status = self.link.status()
            if status is None or status.deg is None:
                return 0.0
            start = status.deg
        travel = max(abs(a - b) for a, b in zip(start, first, strict=True))
        if travel <= ARRIVED_DEG:
            return 0.0
        max_speed, max_acc = self.caps()
        return min_jerk_time(travel, max_speed, max_acc)

    def rest_s(self) -> float:
        """How long the board takes the arm to rest from where it is (its newest state line)."""
        return self.approach_s(self.motion.rest())

    def fit(
        self,
        clip: Clip,
        prep_s: float,
        *,
        base: Mapping[str, float] | None = None,
        start: Sequence[float] | None = None,
    ) -> Clip | None:
        """``clip`` with its preparation compressed so its planned stroke comes within
        ``prep_s`` of the board taking it, the compressed part at :data:`PREP_CAPS` of the caps
        at most (DESIGN.md §7.6 step 2); None when even that is too slow."""
        pose = dict(base) if base is not None else self.base
        planned = self.plan(clip, base=base, start=start)
        if isinstance(planned, str):
            return None
        if planned.stroke_s <= prep_s:
            return clip
        source = rebased(clip, pose, self.base) if base is not None else clip
        least = prep_min_s(source, pose, JOINTS, *self.caps(PREP_CAPS))
        target = prep_s - planned.approach_s
        if target < least - 1e-9 or target <= 0.0:
            return None
        fitted = fit_preparation(clip, target)
        check = self.plan(fitted, base=base, start=start)
        if isinstance(check, str) or check.stroke_s > prep_s + 1e-6:
            return None
        return fitted

    def final(self, planned: Planned) -> dict[str, float]:
        """The pose a planned trajectory ends on (a unit holds it)."""
        return dict(zip(JOINTS, planned.trajectory.knots[-1].deg, strict=True))

    def play(self, clip: Clip) -> Outcome:
        """Send ``clip`` to the board, from the pose a unit holds (no approach) or from rest
        (the board's approach from where the arm is); refused here when it breaks the limits.
        Its stroke is the plan's (approach and stretch included)."""
        held = self.holding()
        planned = self.plan(clip, base=held, start=None if held is None else self._knot(held))
        if isinstance(planned, str):
            return Outcome(False, planned)
        trajectory = planned.trajectory
        self.link.send(trajectory)
        if clip.then == "hold":
            self.held = self.final(planned)
            self.held_until = self.clock() + self.latency_s + planned.seconds + trajectory.hold_s
        else:
            self.held = None
        return Outcome(True, f"sent as {trajectory.id}", planned.seconds, planned.stroke_s)

    def fade(self, layer: Layer, seconds: float) -> None:
        """The gesture (or idle) layer let go: the board takes the arm to rest at its caps."""
        self.held = None
        self.link.rest()

    def rest(self) -> None:
        """To rest, then torque off (the board's)."""
        self.held = None
        self.link.rest()

    @staticmethod
    def _knot(pose: Mapping[str, float]) -> tuple[float, ...]:
        return tuple(pose[j] for j in JOINTS)


class FacePart:
    """The face through the head server (source ``expression``): each event of a clip is an
    expression held for its ``hold_s``, then the face lapses to what stood before it."""

    part: PartName = "face"

    def __init__(
        self,
        face: FaceChannel,
        *,
        latency_s: float = 0.05,
        intensity: float = 1.0,
        hold_s: float = 2.0,
        after: Callable[[float, Callable[[], None]], Callable[[], None]] | None = None,
    ) -> None:
        """``hold_s``: an event's hold when its clip gives none; ``after(seconds, fn)`` runs
        ``fn`` later and returns its cancel (a timer thread by default)."""
        self.face = face
        self._latency_s = latency_s
        self.intensity, self.hold_s = intensity, hold_s
        self.after = after or _timer
        self._pending: list[Callable[[], None]] = []

    @property
    def latency_s(self) -> float:
        """The head server's hop."""
        return self._latency_s

    def ready(self) -> str | None:
        """The face is always there (a line that does not arrive costs nothing)."""
        return None

    def play(self, clip: Clip) -> Outcome:
        """Each event at its time from now."""
        self._cancel()
        for event in clip.events:
            hold = event.hold_s if event.hold_s is not None else self.hold_s

            def show(name: str = event.name, hold: float = hold) -> None:
                self.face.express(name, intensity=self.intensity, hold_s=hold)

            if event.t_s <= 0.0:
                show()
            else:
                self._pending.append(self.after(event.t_s, show))
        return Outcome(True, seconds=clip.duration_s)

    def fade(self, layer: Layer, seconds: float) -> None:
        """No expression to come; the one showing is withdrawn (it lapses by itself anyway)."""
        self._cancel()
        clear = getattr(self.face, "clear", None)
        if callable(clear):
            clear()

    def rest(self) -> None:
        """As a fade."""
        self.fade("gesture", 0.0)

    def _cancel(self) -> None:
        for cancel in self._pending:
            cancel()
        self._pending = []


def _timer(seconds: float, fn: Callable[[], None]) -> Callable[[], None]:
    timer = threading.Timer(seconds, fn)
    timer.daemon = True
    timer.start()
    return timer.cancel


class HeadDoor(Protocol):
    """The gaze arbiter's door for head overlays (``POST /look`` of a ``clip``, ``POST
    /renew``, ``POST /release``): :class:`pepin.expression.door.OverlayDoor` over HTTP, a fake
    in tests."""

    def look(self, request: ClipLook) -> str | None:
        """Start the overlay: None when it plays (or played), else why it was refused; a door
        that does not answer is a refusal in words, never an exception."""
        ...

    def renew(self, source: str) -> None:
        """Restart the TTL of ``source``'s overlay (a loop plays on); nothing when it is gone."""
        ...

    def withdraw(self, source: str, fade_s: float) -> None:
        """End ``source``'s overlay now: the head back on its holder's aim over ``fade_s``."""
        ...


@dataclass(frozen=True)
class NeckLimits:
    """The neck's profile (config/neck.json's ``motion``): the top speed, the pan's ramp and
    the tilt's own, gentler one (a sharp tilt rocks the mast). A head key's speed ceiling is
    the one that carries each axis to its keyframe in the time it has, at that axis's ramp."""

    max_speed_deg_s: float = 299.0
    pan_acc_deg_s2: float = 2232.0
    tilt_acc_deg_s2: float = 600.0

    @classmethod
    def of(cls, motion: NeckMotion) -> NeckLimits:
        """The limits a :class:`pepin.neck.NeckMotion` puts on every move (the base server's
        own rule: the tilt's ramp never above the pan's)."""
        acc = motion.max_acc_deg_s2
        return cls(motion.max_speed_deg_s, acc, min(motion.tilt_max_acc_deg_s2, acc))


class Serial:
    """Calls run one after another on a daemon thread of their own (started by the first), so
    a door's calls keep their order and never hold up the caller."""

    def __init__(self, name: str) -> None:
        """``name``: the thread's."""
        self.name = name
        self._calls: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def __call__(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` after every call posted before it."""
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
                self._thread.start()
        self._calls.put(fn)

    def _run(self) -> None:
        while True:
            fn = self._calls.get()
            try:
                fn()
            except Exception:
                logger.exception("%s: a call broke", self.name)


Answered = Callable[[ClipLook, str | None], None]


class HeadPart:
    """The head as relative overlays through the gaze arbiter (:mod:`pepin.expression.head`): a
    gesture's clip as source ``voice.gesture``, the idle layer's as ``voice.idle`` (marked
    ``idle``: it gives way to any other overlay; ``loop`` when its atom loops); fading a layer
    releases its source over the fade, so the head eases back onto the person. A refusal is
    logged once (the same one again is only counted) and the clip dropped: never retried."""

    part: PartName = "head"

    def __init__(
        self,
        door: HeadDoor,
        *,
        latency_s: float = 0.1,
        limits: NeckLimits | None = None,
        fade_s: float = 0.35,
        run: Callable[[Callable[[], None]], None] | None = None,
        on_answer: Answered | None = None,
    ) -> None:
        """``fade_s``: how long :meth:`rest` eases the head back (config/expression.json's
        ``layers.fade_s.head``); ``run(fn)`` runs a door call off the caller's thread (one
        :class:`Serial` thread by default, so a release never overtakes the look before it; a
        test runs it inline); ``on_answer(request, refusal)`` hears every answer."""
        self.door = door
        self._latency_s = latency_s
        self.limits = limits if limits is not None else NeckLimits()
        self.fade_s = fade_s
        self.run = run or Serial("head-door")
        self.on_answer = on_answer
        self.refused = 0
        self._logged: str | None = None  # the refusal logged last

    @property
    def latency_s(self) -> float:
        """The door's HTTP, the arbiter's step and the neck_target hop."""
        return self._latency_s

    def ready(self) -> str | None:
        """The arbiter decides when the overlay arrives (a better band refuses it)."""
        return None

    def play(self, clip: Clip) -> Outcome:
        """Post the clip's keys as an overlay (off this thread)."""
        source = IDLE_LOOK if clip.layer == "idle" else GESTURE
        request = clip_look(clip, source, self.limits)
        if request is None:
            return Outcome(False, "the clip moves no head")
        self.run(lambda: self._post(request))
        return Outcome(True, seconds=clip.duration_s)

    def renew(self, layer: Layer) -> None:
        """The layer's looping overlay kept alive at the arbiter (its TTL is one period and a
        second, :func:`clip_look`)."""
        source = IDLE_LOOK if layer == "idle" else GESTURE
        self.run(lambda: self.door.renew(source))

    def fade(self, layer: Layer, seconds: float) -> None:
        """The layer's overlay released: the head back on its holder's aim over ``seconds``."""
        source = IDLE_LOOK if layer == "idle" else GESTURE
        self.run(lambda: self.door.withdraw(source, seconds))

    def rest(self) -> None:
        """Both overlays released over :attr:`fade_s` (a drive, the conversation's end)."""

        def withdraw() -> None:
            self.door.withdraw(GESTURE, self.fade_s)
            self.door.withdraw(IDLE_LOOK, self.fade_s)

        self.run(withdraw)

    def _post(self, request: ClipLook) -> None:
        refusal = self.door.look(request)
        if self.on_answer is not None:
            try:
                self.on_answer(request, refusal)
            except Exception:
                logger.exception("head part: the answer's listener broke")
        if refusal is None:
            self._logged = None
            return
        self.refused += 1
        if refusal != self._logged:
            logger.warning("head overlay refused, dropped: %s", refusal)
            self._logged = refusal


HEAD_MIN_SPEED_DEG_S = 1.0  # a key's least ceiling (a speed of 0 would be the servo's top)
HEAD_DOFS = ("pan_deg", "tilt_deg")
SAME_DEG = 1e-6  # a keyframe this close to the one before it is a hold, not a move


def keyframes(clip: Clip) -> list[tuple[float, float, float]]:
    """A head clip as ``(seconds, pan_deg, tilt_deg)`` at every key time of its tracks, each
    axis the sum of its tracks there (a merged clip's steps are offsets: they add up)."""
    tracks = [t for t in clip.tracks if t.dof in HEAD_DOFS]
    times = sorted({t for track in tracks for t in track.times})

    def axis(dof: str, t: float) -> float:
        return sum(track_value(track, t) for track in tracks if track.dof == dof)

    return [(t, axis("pan_deg", t), axis("tilt_deg", t)) for t in times]


def move_speed(travel_deg: float, seconds: float, acc_deg_s2: float) -> float:
    """The profile speed that takes one joint ``travel_deg`` from rest to rest in ``seconds`` at
    a ramp of ``acc_deg_s2`` (a trapezoid that ends on time); a move that cannot end on time at
    that ramp (:func:`needed_acc` above it) gets the triangle's peak, the fastest it can go."""
    d = abs(travel_deg)
    if d == 0.0:
        return 0.0
    reach = acc_deg_s2 * seconds
    disc = reach * reach - 4.0 * acc_deg_s2 * d
    if seconds <= 0.0 or disc < 0.0:
        return math.sqrt(acc_deg_s2 * d)
    return (reach - math.sqrt(disc)) / 2.0


def needed_acc(travel_deg: float, seconds: float) -> float:
    """The least ramp (deg/s^2) that takes a joint ``travel_deg`` from rest to rest in
    ``seconds``: speeding up half the way and braking the other half, 4 d / T^2."""
    return math.inf if seconds <= 0.0 else 4.0 * abs(travel_deg) / (seconds * seconds)


def clip_look(clip: Clip, source: str, limits: NeckLimits | None = None) -> ClipLook | None:
    """A head clip as the arbiter's overlay (offsets in radians from the holder's aim, pan +
    left, tilt + down), None when it moves nothing.

    The clip's keyframes say where the head IS at their times; the arbiter writes each key at
    its time and the head then moves there. So each keyframe is a key written at the keyframe
    before it, with the speed ceiling that ends the move on the keyframe's time (each axis at
    its own ramp, :func:`move_speed`; the larger ceiling of the two, within the neck's top
    speed): a nod's stroke lands on its word, not a move later. A keyframe where the head
    already is (a hold) writes nothing. The clip starts on the holder's aim (the book's head
    atoms start and end on it). A looping clip closes its period with its first key again
    (the arbiter's rule for a loop); the idle layer's clip is marked ``idle``."""
    limits = limits if limits is not None else NeckLimits()
    frames = keyframes(clip)
    if not frames:
        return None
    start, pan0, tilt0 = frames[0]
    t0 = start
    keys: list[ClipKey] = []
    for t, pan, tilt in frames[1:]:
        if abs(pan - pan0) < SAME_DEG and abs(tilt - tilt0) < SAME_DEG:
            t0 = t
            continue
        speed = max(
            move_speed(pan - pan0, t - t0, limits.pan_acc_deg_s2),
            move_speed(tilt - tilt0, t - t0, limits.tilt_acc_deg_s2),
        )
        keys.append({
            "at_s": round(t0 - start, 4),
            "pan_rad": math.radians(pan),
            "tilt_rad": math.radians(tilt),
            "speed_deg_s": round(min(limits.max_speed_deg_s, max(HEAD_MIN_SPEED_DEG_S, speed)), 2),
        })  # fmt: skip
        t0, pan0, tilt0 = t, pan, tilt
    if not keys:
        return None
    target: ClipTarget = {"keys": keys}
    if clip.loop:
        keys.append({**keys[0], "at_s": round(frames[-1][0] - start, 4)})
        target["loop"] = True
    if clip.layer == "idle":
        target["idle"] = True
    return {
        "source": source,
        "kind": "clip",
        "band": BAND,
        "ttl_s": clip.duration_s + 1.0,
        "target": target,
    }


class Absent:
    """A part that cannot play now, and why: never ready, every clip refused, so a
    gesture plays without it."""

    def __init__(self, part: PartName, why: str) -> None:
        """``why`` is what every refusal says."""
        self.part: PartName = part
        self.latency_s = 0.0
        self.why = why

    def ready(self) -> str | None:
        """Always the reason."""
        return self.why

    def play(self, clip: Clip) -> Outcome:
        """Refused."""
        return Outcome(False, self.why)

    def fade(self, layer: Layer, seconds: float) -> None:
        """Nothing plays."""

    def rest(self) -> None:
        """Nothing plays."""
