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
* :class:`FacePart`: a clip's events as timed expressions of the head server's face, and its
  drawing style (timed, or standing; a fade leaves it).
* :class:`EyesPart`: a clip's look as timed eyes overrides of the head server (the eye fields'
  tracks and the iris events, one override at each change of them, each lapsing at the clip's
  end), its blinks, winks and releases at their times, its effects for their hold
  (:func:`eye_calls`); fading a layer lets the eyes go (``eyes_release``: the look and the
  effect). The firmware blinks by itself: the book's idle layer names no eye atom by default.
* :class:`HeadPart`: a clip's keyframes as a relative ``clip`` overlay of the gaze arbiter
  (:mod:`pepin.expression.head`, :func:`clip_look`) on top of whatever holds the head (the
  speaker gaze's look on the person, or home), posted through the arbiter's door
  (:class:`pepin.expression.door.OverlayDoor`) on a thread of its own; fading a layer is that
  layer's source released over the fade. A PACED clip (a paced composition's head: the nod and
  the shake) lands each move :data:`SETTLE_S` before its keyframe at the most its axis gives a
  clip (the tilt's ramp up to the arbiter's ``clip_acc_max_deg_s2``), and its steps play at
  the least period that allows (:func:`least_period`, applied by the renderer).
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
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from pepin.arm_driver import ARRIVED_DEG, WIRE_SLACK
from pepin.arm_motion import ArmMotion
from pepin.expression.animation import (
    FACE_STYLE,
    STYLE_DEFAULT,
    Atom,
    Clip,
    Layer,
    PartName,
    Track,
    Value,
)
from pepin.expression.breath import ArmBreath
from pepin.expression.executors import EyesChannel, FaceChannel, Outcome
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
from pepin.head_link import EYE_RANGES
from pepin.neck import ACC_UNIT_DEG_S2, NeckMotion

logger = logging.getLogger(__name__)

# The share of the arm's caps a posture's raise and its gentle way down run at (config/expression
# .json pulse.arm_up_caps_share sets it live): 0.3 keeps the servos under ~70 deg/s, where the
# quiet gesticulation lives (Artem 2026-10-13: the raise at 0.5, 110 deg/s, whined)
GENTLE_SHARE = 0.3


@dataclass(frozen=True)
class Planned:
    """A clip as the board will play it: the trajectory, the board's approach from where the arm
    is to its first knot (seconds; 0 when it is there), the factor the caps stretched the clip
    by (1: none), when its stroke lands after the board takes it (the approach plus the
    stretched stroke): the one stroke time the engine schedules and logs; and the pose its last
    knot stands on WITHOUT the breath (``final_pose``: what a unit holds by name) with what the
    breath added there (``final_offset``: what the hold freezes)."""

    trajectory: Trajectory
    approach_s: float
    stretch: float
    stroke_s: float
    final_pose: tuple[float, ...] = ()
    final_offset: tuple[float, ...] = ()

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
    the arm to rest by itself, and the next clip plans from rest again.

    THE BREATH (:attr:`breath`, :mod:`pepin.expression.breath`): every trajectory planned over a
    held pose carries the breath's offsets in its knots, blended from the offset the last hold
    froze (:attr:`held_offset`) to the live breath (out to none for a clip that ends at rest);
    none from rest. :attr:`held` stays the clean pose (a unit's name, the posture's compare);
    :meth:`held_knot` is where the arm really stands. A knot the breath would push past a limit
    or a cap plays without it: the gesture never loses to its breath."""

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
        breath: ArmBreath | None = None,
    ) -> None:
        """``motion``: config/arm.json's (rest pose, limits, caps); ``speed_scale`` scales the
        caps a clip is checked against and travels with each trajectory (the board's profile
        and its way to rest); ``knot_hz``: knots a second (the board interpolates between);
        ``clock``: when a hold lapses; ``breath``: the layer under every held clip (None: no
        breath)."""
        self.link, self.motion = link, motion
        self._latency_s = latency_s
        self.speed_scale = speed_scale
        self.knot_hz = knot_hz
        self.source = source
        self.clock = clock
        self.breath = breath
        self.gentle_share = GENTLE_SHARE
        self._ids = itertools.count(1)
        self.base = dict(motion.rest_deg)
        self.held: dict[str, float] | None = None  # the pose the last trajectory holds (a unit)
        self.held_offset: tuple[float, ...] | None = None  # what the breath added to its knot
        self.held_until = float("-inf")  # when the board lets that hold go (engine clock)
        self._breath_refused: str | None = None  # the last refusal of a breathed clip, logged

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
            self.held, self.held_offset = None, None
        return self.held

    def held_knot(self) -> tuple[float, ...] | None:
        """Where the arm really stands while a unit holds it: the held pose and the breath's
        offset frozen on it; None without a unit."""
        held = self.holding()
        if held is None:
            return None
        offset = self.held_offset or (0.0,) * len(JOINTS)
        return tuple(held[j] + o for j, o in zip(JOINTS, offset, strict=True))

    def plan(
        self,
        clip: Clip,
        name: str = "",
        *,
        base: Mapping[str, float] | None = None,
        start: Sequence[float] | None = None,
    ) -> Planned | str:
        """``clip`` as the trajectory to send over ``base`` (the rest pose by default; a unit's
        next stroke: the held pose, the clip started there, the breath under it) and when its
        stroke lands, or why it cannot be played. The board's approach is from ``start`` (the
        arm's angles when the board takes it; None: the newest state line's). The curve is
        checked against the exact caps and its knots as the board will see them (:func:`refused`:
        rounded to the wire's 1 ms and 0.01 deg, a jaw shut on its limit samples a hair past it
        between two keys)."""
        if clip.part != "arm":
            return f"an {clip.part} clip is not the arm's"
        motion = self.motion
        max_speed, max_acc = self.caps()
        pose = dict(base) if base is not None else self.base
        if base is not None:
            clip = rebased(clip, pose, self.base)
        # a clip stretched by 1 / speed_scale needs no more knots than at full speed
        sampled = sample(clip, pose, JOINTS, self.knot_hz * min(1.0, self.speed_scale))
        stretched = stretch_to_caps(sampled, max_speed * STRETCH_MARGIN, max_acc * STRETCH_MARGIN)
        span = sampled[-1][0] - sampled[0][0]
        stretch = (stretched[-1][0] - stretched[0][0]) / span if span > 0 else 1.0
        knots = wire(stretched)
        # a stroke too fast for the caps takes longer rather than being refused
        if knots[-1][0] > motion.max_duration_s:
            return f"{knots[-1][0]:.1f} s is longer than the board's {motion.max_duration_s:.0f} s"
        refusal = refused(stretched, motion.limits(), max_speed, max_acc)
        if refusal is not None:
            return refusal
        final = tuple(knots[-1][1])
        offset = (0.0,) * len(JOINTS)
        if self.breath is not None and base is not None:  # never from rest
            t0 = self.clock() + self.latency_s
            curve = self.breath.breathe(stretched, t0, self.held_offset, live=clip.then == "hold")
            why = refused(curve, motion.limits(), max_speed, max_acc)
            if why is None:
                breathed = wire(curve)
                offset = self.breath.final_offset(knots, breathed)
                knots = breathed
            elif why != self._breath_refused:  # the gesture plays without its breath
                logger.info("arm breath left out of a clip: %s", why)
                self._breath_refused = why
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
        return Planned(trajectory, approach, stretch, approach + stretch * clip.stroke_s, final,
                       offset)  # fmt: skip

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
        """The pose a planned trajectory ends on, the breath aside (a unit holds it by name)."""
        final = planned.final_pose or planned.trajectory.knots[-1].deg
        return dict(zip(JOINTS, final, strict=True))

    def play(self, clip: Clip) -> Outcome:
        """Send ``clip`` to the board, from the pose a unit holds (no approach) or from rest
        (the board's approach from where the arm is); refused here when it breaks the limits.
        Its stroke is the plan's (approach and stretch included)."""
        held = self.holding()
        planned = self.plan(clip, base=held, start=None if held is None else self.held_knot())
        if isinstance(planned, str):
            return Outcome(False, planned)
        trajectory = planned.trajectory
        self.link.send(trajectory)
        if clip.then == "hold":
            self.held = self.final(planned)
            self.held_offset = planned.final_offset if any(planned.final_offset) else None
            self.held_until = self.clock() + self.latency_s + planned.seconds + trajectory.hold_s
        else:
            self.held, self.held_offset = None, None
        return Outcome(True, f"sent as {trajectory.id}", planned.seconds, planned.stroke_s)

    def lower(self, share: float | None = None) -> float | None:
        """A unit's held arm taken to rest gently: a trajectory from where it stands to rest at
        ``share`` of the caps (:attr:`gentle_share` by default; the book's cubic timing), the
        breath blended out on the way, then the board's rest and torque off; the seconds it
        takes, None when nothing is held (the board's own way is the quick one: :meth:`fade`)."""
        held = self.holding()
        knot = self.held_knot()
        if held is None or knot is None:
            return None
        share = self.gentle_share if share is None else share
        speed, acc = self.caps(share)
        rest = self.motion.rest_deg
        travel = max(abs(rest[j] - k) for j, k in zip(JOINTS, knot, strict=True))
        seconds = max(1.5 * travel / speed, math.sqrt(6.0 * travel / acc), 0.05)
        if self.breath is not None:  # the breath blends out on the way: at least that long
            seconds = max(seconds, self.breath.breath.blend_s)
        tracks = tuple(Track(j, "absolute", (0.0, seconds), (held[j], rest[j])) for j in JOINTS)
        clip = Clip("arm", 0.0, seconds, seconds, tracks)
        planned = self.plan(clip, "down", base=held, start=knot)
        if isinstance(planned, str):
            self.rest()  # the quick way, never nothing
            return None
        self.link.send(planned.trajectory)
        self.held, self.held_offset = None, None
        return self.latency_s + planned.seconds

    def fade(self, layer: Layer, seconds: float) -> None:
        """The gesture (or idle) layer let go: the board takes the arm to rest at its caps."""
        self.held, self.held_offset = None, None
        self.link.rest()

    def rest(self) -> None:
        """To rest, then torque off (the board's)."""
        self.held, self.held_offset = None, None
        self.link.rest()

    @staticmethod
    def _knot(pose: Mapping[str, float]) -> tuple[float, ...]:
        return tuple(pose[j] for j in JOINTS)


def wire(knots: Sequence[tuple[float, Sequence[float]]]) -> list[tuple[float, list[float]]]:
    """``knots`` as the wire carries them (:mod:`pepin.expression.trajectory`): times to the
    millisecond, angles to 0.01 deg; the board checks these, so the laptop checks the same."""
    return [(round(t, 3), [round(d, 2) for d in deg]) for t, deg in knots]


def refused(
    curve: Sequence[tuple[float, Sequence[float]]],
    limits: Sequence[tuple[float, float]],
    max_speed: float,
    max_acc: float,
) -> str | None:
    """Why ``curve`` may not be sent, None when it may: the curve over the exact caps (the
    laptop's rule), or its knots as the wire carries them (:func:`wire`) refused by the board's
    own check: outside a limit (no slack) or over the caps by more than the rounding's noise the
    board forgives (:data:`pepin.arm_driver.WIRE_SLACK`: the times' millisecond alone moves a
    40 ms step's speed by up to 2.5 %)."""
    free = [(-math.inf, math.inf)] * len(limits)
    return check_knots(curve, free, max_speed, max_acc, JOINTS) or check_knots(
        wire(curve), limits, max_speed, max_acc, JOINTS, slack=WIRE_SLACK
    )


class FacePart:
    """The face through the head server (source ``expression``): each event of a clip is an
    expression held for its ``hold_s``, then the face lapses to what stood before it, or a
    drawing style (``style``: for its ``hold_s``, standing without one; :data:`STYLE_DEFAULT`
    lets the style go). A fade withdraws the expression and leaves the style: a standing one
    stands, a timed one lapses on the head server."""

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
            show: Callable[[], None]
            if event.dof == FACE_STYLE:
                name = None if event.name == STYLE_DEFAULT else event.name
                ms = None if event.hold_s is None else round(event.hold_s * 1000.0)
                show = partial(self.face.style, name, ms)
            else:
                hold = event.hold_s if event.hold_s is not None else self.hold_s
                show = partial(self.face.express, event.name, intensity=self.intensity,
                               hold_s=hold)  # fmt: skip
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


# A look's override outlives its segment by this when another follows it, so the next one
# replaces it before it lapses (a timer's jitter and the WiFi hop never show the bare look between)
EYE_OVERLAP_S = 0.25
EPS_S = 1e-6


class EyesPart:
    """The eyes through the head server (source ``expression``, :class:`EyesChannel`): a clip's
    calls (:func:`eye_calls`) each at its time from now; fading a layer lets the eyes go."""

    part: PartName = "eyes"

    def __init__(
        self,
        channel: EyesChannel,
        *,
        latency_s: float = 0.05,
        after: Callable[[float, Callable[[], None]], Callable[[], None]] | None = None,
    ) -> None:
        """``after(seconds, fn)`` runs ``fn`` later and returns its cancel (a timer thread by
        default)."""
        self.channel = channel
        self._latency_s = latency_s
        self.after = after or _timer
        self._pending: list[Callable[[], None]] = []

    @property
    def latency_s(self) -> float:
        """The head server's hop (the firmware tweens an override over 160 ms after it)."""
        return self._latency_s

    def ready(self) -> str | None:
        """The eyes are always there (a line that does not arrive costs nothing)."""
        return None

    def play(self, clip: Clip) -> Outcome:
        """Each call at its time from now."""
        self._cancel()
        for t, call in eye_calls(clip, self.channel):
            if t <= 0.0:
                call()
            else:
                self._pending.append(self.after(t, call))
        return Outcome(True, seconds=clip.duration_s)

    def fade(self, layer: Layer, seconds: float) -> None:
        """Nothing more to come, and the eyes let go: the look and the effect."""
        self._cancel()
        self.channel.eyes_release()

    def rest(self) -> None:
        """As a fade."""
        self.fade("gesture", 0.0)

    def _cancel(self) -> None:
        for cancel in self._pending:
            cancel()
        self._pending = []


def eye_calls(clip: Clip, channel: EyesChannel) -> list[tuple[float, Callable[[], None]]]:
    """An eyes clip as the head server's calls, each at its time (seconds from the clip's
    start), in order. THE LOOK: at each change of it (a track of an eye field starts or ends, an
    iris event begins or its hold ends) one override of every field that holds then (a track
    from its first key to its last, a one-key track to the clip's end, each at its value there;
    the iris until its hold ends, else to the clip's end), timed to the next change (and
    :data:`EYE_OVERLAP_S` more when another override follows), the last to the clip's end; a
    change to nothing sends nothing (the one before lapses there). THE MOMENTS: a ``blink`` key
    is a blink of both eyes, a ``wink`` key a wink of the robot's left eye for a value above 0
    and of its right otherwise, a ``release`` key the eyes let go. THE EFFECTS: an ``effect``
    event for its hold, else to the clip's end."""
    end = clip.duration_s
    looks = [tr for tr in clip.tracks if tr.dof in EYE_RANGES]
    irises = [e for e in clip.events if e.dof == "iris"]

    def until(t_s: float, hold_s: float | None) -> float:
        return end if hold_s is None else min(end, t_s + hold_s)

    def span(track: Track) -> tuple[float, float]:
        return track.times[0], track.times[-1] if len(track.times) > 1 else end

    def look_at(t: float) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for track in looks:  # a later step's track of the same field wins (merged in order)
            lo, hi = span(track)
            if lo - EPS_S <= t < hi - EPS_S:
                fields[track.dof] = round(track_value(track, t), 3)
        shown = [e for e in irises if e.t_s - EPS_S <= t < until(e.t_s, e.hold_s) - EPS_S]
        if shown:
            fields["iris"] = shown[-1].name
        return fields

    edges = {t for track in looks for t in track.times} | {
        x for e in irises for x in (e.t_s, until(e.t_s, e.hold_s))
    }
    changes = [(t, look_at(t)) for t in sorted(edges) if t < end - EPS_S]
    calls: list[tuple[float, Callable[[], None]]] = []
    for i, (t, fields) in enumerate(changes):
        if not fields:
            continue
        nxt = changes[i + 1] if i + 1 < len(changes) else None
        stop = end if nxt is None else nxt[0] + (EYE_OVERLAP_S if nxt[1] else 0.0)
        ms = round((stop - t) * 1000.0)
        calls.append((t, partial(channel.eyes, ms=ms, **fields)))
    for track in clip.tracks:
        for t, value in zip(track.times, track.values, strict=True):
            if track.dof == "blink":
                calls.append((t, partial(channel.blink, "both")))
            elif track.dof == "wink":
                calls.append((t, partial(channel.blink, "left" if value > 0.0 else "right")))
            elif track.dof == "release":
                calls.append((t, channel.eyes_release))
    for e in clip.events:
        if e.dof == "effect":
            ms = round((until(e.t_s, e.hold_s) - e.t_s) * 1000.0)
            calls.append((e.t_s, partial(channel.eyes_effect, e.name, ms)))
    return sorted(calls, key=lambda c: c[0])


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
    the tilt's own, gentler one (a sharp tilt rocks the mast); and the gaze arbiter's bounds on
    a head clip (config/knobs.json's ``gaze``): the most a paced clip's key may ask the tilt's
    ramp for (``clip_acc_max_deg_s2``) and the rate the tilt may turn back at
    (``clip_max_hz``). A head key's speed ceiling is the one that carries each axis to its
    keyframe in the time it has, at that axis's ramp."""

    max_speed_deg_s: float = 299.0
    pan_acc_deg_s2: float = 2232.0
    tilt_acc_deg_s2: float = 600.0
    clip_tilt_acc_deg_s2: float = 600.0
    clip_max_hz: float = 2.5

    @classmethod
    def of(
        cls, motion: NeckMotion, clip_acc_deg_s2: float | None = None, clip_max_hz: float = 2.5
    ) -> NeckLimits:
        """The limits a :class:`pepin.neck.NeckMotion` puts on every move (the base server's
        own rule: the tilt's ramp never above the pan's) and a paced clip's tilt ramp (the
        arbiter's ``clip_acc_max_deg_s2``; None: the tilt's own), never under the tilt's own
        nor above the neck's top (the base server's cap on a target's own tilt ramp)."""
        acc = motion.max_acc_deg_s2
        tilt = min(motion.tilt_max_acc_deg_s2, acc)
        clip = tilt if clip_acc_deg_s2 is None else min(max(clip_acc_deg_s2, tilt), acc)
        return cls(motion.max_speed_deg_s, acc, tilt, clip, clip_max_hz)

    @property
    def turn_s(self) -> float:
        """The least time between two turns of a clip's tilt: half a period of ``clip_max_hz``
        (0: no limit)."""
        return 0.5 / self.clip_max_hz if self.clip_max_hz > 0.0 else 0.0


def load_neck_limits(neck: Path | None = None, knobs: Path | None = None) -> NeckLimits:
    """config/neck.json's ``motion`` and the gaze arbiter's clip bounds as config/knobs.json
    ships them (``gaze.clip_acc_max_deg_s2`` and ``gaze.clip_max_hz``, their defaults, read
    once: a live change on the arbiter alone leaves the gestures sized as they were)."""
    from pepin.deployment import config_file  # lazy: the parts stay import-light
    from pepin.flags import read_knobs
    from pepin.neck import NeckConfig

    motion = NeckConfig.from_json(neck if neck is not None else config_file("neck.json")).motion
    gaze = read_knobs(knobs).get("gaze", {})

    def default(name: str, fallback: float) -> float:
        spec = gaze.get(name)
        return float(spec["default"]) if isinstance(spec, dict) and "default" in spec else fallback

    defaults = NeckLimits()
    return NeckLimits.of(
        motion,
        default("clip_acc_max_deg_s2", defaults.clip_tilt_acc_deg_s2),
        default("clip_max_hz", defaults.clip_max_hz),
    )


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
    releases its source over the fade, so the head eases back onto the person. A clip posted
    while the voice's own wheels move for a gesture (``own_wheels``: the wheels executor's word)
    is marked ``own_wheels``: the arbiter lets it play over that rock, never over a navigation
    drive. A refusal is logged once (the same one again is only counted) and the clip dropped:
    never retried."""

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
        own_wheels: Callable[[], bool] | None = None,
    ) -> None:
        """``fade_s``: how long :meth:`rest` eases the head back (config/expression.json's
        ``layers.fade_s.head``); ``run(fn)`` runs a door call off the caller's thread (one
        :class:`Serial` thread by default, so a release never overtakes the look before it; a
        test runs it inline); ``on_answer(request, refusal)`` hears every answer;
        ``own_wheels()``: whether the voice's wheels move for a gesture now
        (:meth:`pepin.expression.wheels.WheelsExecutor.gesture_moving`; None: never)."""
        self.door = door
        self.own_wheels = own_wheels
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
        if self.own_wheels is not None and self.own_wheels():  # asked as the look leaves
            request["target"]["own_wheels"] = True
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
# How long before its keyframe a paced clip's move lands, so the arbiter has seen the head
# settle when the next key falls due: pepin.gaze's Arbiter writes no overlay key while the last
# one's move is in flight, and the key due when the head settles is the one written, so a key
# landing late delays the next and a third may be skipped (2026-10-15, the bus tap: two shakes
# played right, left, right, centre, two of their six keys never written). The servo's lag
# behind its profile, two still encoder readings and their way back: 0.10-0.11 s a key that
# night (an agreement nod, a shake: scratch/head_nods/neck_trace.py in the main checkout)
SETTLE_S = 0.1
# An idle key's ramp: this times the least that lands it on time (4 d / T^2: speeding up half the
# way, braking the other half), so a quarter of its time each way is a soft start and stop and the
# servo glides between keys instead of stepping (a key at the neck's full ramp starts and stops at
# once); never under the Acceleration register's one unit
GLIDE = 1.25
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


def move_time(travel_deg: float, acc_deg_s2: float, top_deg_s: float) -> float:
    """The least time (s) a joint takes over ``travel_deg`` from rest to rest at a ramp of
    ``acc_deg_s2`` and a top speed of ``top_deg_s``: a triangle, or a trapezoid at the top."""
    d = abs(travel_deg)
    if d == 0.0:
        return 0.0
    if d * acc_deg_s2 <= top_deg_s * top_deg_s:
        return 2.0 * math.sqrt(d / acc_deg_s2)
    return d / top_deg_s + top_deg_s / acc_deg_s2


def least_period(
    atom: Atom, params: Mapping[str, float], limits: NeckLimits, lead_s: float = SETTLE_S
) -> float:
    """The shortest period a head ``atom`` plays at with these parameters as a paced clip
    (:func:`clip_look`): each move between its keyframes at its axis's ramp (the tilt's a
    clip's, ``clip_tilt_acc_deg_s2``) and the top speed, landed ``lead_s`` early, and the tilt
    turning back no sooner than ``limits.turn_s`` after its last turn (the arbiter's
    ``clip_max_hz``; with ``repeats`` over 1 the turn at the cycles' seam too). 0 for an
    atom that moves neither axis."""
    ramps = {"pan_deg": limits.pan_acc_deg_s2, "tilt_deg": limits.clip_tilt_acc_deg_s2}
    cycles = 2 if params.get("repeats", 1.0) > 1.5 else 1
    least = 0.0
    for dof, ramp in ramps.items():
        points = [(k.at, _key_value(k.values[dof], params)) for k in atom.keys if dof in k.values]
        moves = [(a[0], b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(points)
                 if abs(b[1] - a[1]) >= SAME_DEG]  # fmt: skip
        for _, share, travel in moves:
            if share > 0.0:
                least = max(least, (move_time(travel, ramp, limits.max_speed_deg_s) + lead_s)
                            / share)  # fmt: skip
        if dof != "tilt_deg" or limits.turn_s <= 0.0:
            continue
        turns: list[float] = []
        direction = 0.0
        for cycle in range(cycles):
            for start, _, travel in moves:
                if math.copysign(1.0, travel) != direction:
                    turns.append(cycle + start)
                    direction = math.copysign(1.0, travel)
        for a, b in itertools.pairwise(turns):
            if b > a:
                least = max(least, limits.turn_s / (b - a))
    return least


def _key_value(raw: Value, params: Mapping[str, float]) -> float:
    """A head key's value: a number, or a drawn parameter (``"$p"``, ``"-$p"``)."""
    if isinstance(raw, str):
        sign = -1.0 if raw.startswith("-") else 1.0
        return sign * params[raw.lstrip("-").lstrip("$")]
    return float(raw)


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
    (the arbiter's rule for a loop); the idle layer's clip is marked ``idle`` and GLIDES: each
    key also carries its own ramp, ``GLIDE`` times the least that lands its move on time (never
    under the Acceleration register's unit, never over the axis's ramp), its speed the one that
    ends the move on time at that ramp: the servo eases between keys instead of stepping. A
    PACED clip's key (``Clip.paced``: the nod, the shake) lands its move :data:`SETTLE_S` before
    its keyframe, as the axis's ramp allows (else as fast as it allows), so the arbiter sees the
    head settled when the next key falls due; a tilt move that needs more than the tilt's own
    ramp for that asks the least that does, up to ``limits.clip_tilt_acc_deg_s2``, as the key's
    ``tilt_acc_deg_s2``."""
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
        ramps = (limits.pan_acc_deg_s2, limits.tilt_acc_deg_s2)
        glide = clip.layer == "idle"
        paced = clip.paced and not glide
        seconds = t - t0 - SETTLE_S if paced else t - t0  # a paced move lands early
        tilt_ask: float | None = None
        if glide:  # one ramp for both servos: the gentlest that lands the larger move on time
            least = GLIDE * max(needed_acc(pan - pan0, t - t0), needed_acc(tilt - tilt0, t - t0))
            ramp = max(ACC_UNIT_DEG_S2, least)
            ramps = (min(ramp, ramps[0]), min(ramp, ramps[1]))
        elif paced and limits.clip_tilt_acc_deg_s2 > ramps[1]:
            need = needed_acc(tilt - tilt0, seconds)
            if need > ramps[1]:
                tilt_ask = min(limits.clip_tilt_acc_deg_s2, math.ceil(need * 100.0) / 100.0)
                ramps = (ramps[0], tilt_ask)
        speed = max(
            move_speed(pan - pan0, seconds, ramps[0]),
            move_speed(tilt - tilt0, seconds, ramps[1]),
        )
        key: ClipKey = {
            "at_s": round(t0 - start, 4),
            "pan_rad": math.radians(pan),
            "tilt_rad": math.radians(tilt),
            "speed_deg_s": round(min(limits.max_speed_deg_s, max(HEAD_MIN_SPEED_DEG_S, speed)), 2),
        }
        if glide:
            key["acc_deg_s2"] = round(max(ramps), 2)
        if tilt_ask is not None:
            key["tilt_acc_deg_s2"] = tilt_ask
        keys.append(key)
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
