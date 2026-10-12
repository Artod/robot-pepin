"""The body parts as the gesture engine plays them (:class:`pepin.expression.executors.Executor`).

* :class:`ArmPart`: a clip sampled into joint knots over the rest pose, checked against the
  arm's limits and caps (scaled by ``speed_scale``: speeds by it, accelerations by its square,
  as a clip stretched by its inverse moves), and sent to the board in one ``arm_trajectory``;
  the board plays it, then rests and lets go of the torque by itself. A clip outside the
  limits or the caps is refused here with the reason, never clamped (the board would refuse it
  too). Fading the gesture layer is ``arm_rest``.
* :class:`FacePart`: a clip's events as timed expressions of the head server's face.
* :class:`HeadPart`: a clip's offsets as a relative ``clip`` look of the gaze arbiter
  (:mod:`pepin.expression.head`), posted on a thread of its own (the door answers when the
  overlay ends). The arbiter's overlay is not built yet: until it is, the cascade gives the
  engine an :class:`Absent` head and the head steps of a gesture are left out.
* :class:`Absent`: a part that cannot play, with the reason (the wheels, the head tonight).
"""

from __future__ import annotations

import itertools
import logging
import math
import threading
from collections.abc import Callable
from typing import Protocol

from pepin.arm_motion import ArmMotion
from pepin.expression.animation import Clip, Layer, PartName
from pepin.expression.executors import FaceChannel, Outcome
from pepin.expression.head import BAND, GESTURE, IDLE_LOOK, ClipKey, ClipLook
from pepin.expression.mix import check_knots, sample, track_value
from pepin.expression.trajectory import JOINTS, ArmLink, Knot, Trajectory

logger = logging.getLogger(__name__)

HEAD_SPEED_HEADROOM = 1.5  # a head key's speed ceiling over the speed its clip asks
HEAD_MIN_SPEED_DEG_S = 10.0
HEAD_MAX_SPEED_DEG_S = 299.0  # config/neck.json's motion.max_speed_deg_s


class ArmPart:
    """The arm through the board's base server (an :class:`pepin.expression.trajectory.ArmLink`)."""

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
    ) -> None:
        """``motion``: config/arm.json's (rest pose, limits, caps); ``speed_scale`` scales the
        caps a clip is checked against and travels with each trajectory (the board's profile
        and its way to rest); ``knot_hz``: knots a second (the board interpolates between)."""
        self.link, self.motion = link, motion
        self._latency_s = latency_s
        self.speed_scale = speed_scale
        self.knot_hz = knot_hz
        self.source = source
        self._ids = itertools.count(1)
        self.base = dict(motion.rest_deg)

    @property
    def latency_s(self) -> float:
        """The WiFi hop and the board's next tick."""
        return self._latency_s

    def ready(self) -> str | None:
        """Why the arm cannot play: no state line has read it yet, or its servos are silent."""
        status = self.link.status()
        if status is None:
            return "no word from the arm yet (is the base server reading it?)"
        if status.phase == "silent":
            return f"the arm's servos are silent: {status.error or 'no reply'}"
        return None

    def plan(self, clip: Clip, name: str = "") -> Trajectory | str:
        """``clip`` as the trajectory to send, or why it cannot be played."""
        if clip.part != "arm":
            return f"an {clip.part} clip is not the arm's"
        motion = self.motion
        if clip.duration_s > motion.max_duration_s:
            return (
                f"{clip.duration_s:.1f} s is longer than the board's {motion.max_duration_s:.0f} s"
            )
        # a clip stretched by 1 / speed_scale needs no more knots than at full speed
        knots = sample(clip, self.base, JOINTS, self.knot_hz * min(1.0, self.speed_scale))
        refusal = check_knots(
            knots,
            motion.limits(),
            motion.max_speed_deg_s * self.speed_scale,
            motion.max_acc_deg_s2 * self.speed_scale**2,
            JOINTS,
        )
        if refusal is not None:
            return refusal
        return Trajectory(
            f"{name or 'gesture'}-{next(self._ids)}",
            self.source,
            tuple(Knot(round(t, 3), tuple(round(d, 2) for d in deg)) for t, deg in knots),
            "rest",
            0.0,
            name,
            self.speed_scale,
        )

    def play(self, clip: Clip) -> Outcome:
        """Send ``clip`` to the board; refused here when it breaks the limits or the caps."""
        planned = self.plan(clip)
        if isinstance(planned, str):
            return Outcome(False, planned)
        self.link.send(planned)
        return Outcome(True, f"sent as {planned.id}", clip.duration_s)

    def fade(self, layer: Layer, seconds: float) -> None:
        """The gesture (or idle) layer let go: the board takes the arm to rest at its caps."""
        self.link.rest()

    def rest(self) -> None:
        """To rest, then torque off (the board's)."""
        self.link.rest()


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
    /release``): built with the arbiter's overlay, faked meanwhile."""

    def look(self, request: ClipLook) -> str | None:
        """Play the overlay; blocks until it ends. None when it played, else why not."""
        ...

    def withdraw(self, source: str) -> None:
        """End the overlay of ``source`` now; the head goes back to its holder."""
        ...


class HeadPart:
    """The head as relative overlays through the gaze arbiter (:mod:`pepin.expression.head`)."""

    part: PartName = "head"

    def __init__(
        self,
        door: HeadDoor,
        *,
        latency_s: float = 0.1,
        max_speed_deg_s: float = HEAD_MAX_SPEED_DEG_S,
        run: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        """``run(fn)`` runs a blocking door call off the caller's thread (a daemon thread by
        default; a test runs it inline)."""
        self.door = door
        self._latency_s = latency_s
        self.max_speed_deg_s = max_speed_deg_s
        self.run = run or _thread
        self.answers: list[str | None] = []

    @property
    def latency_s(self) -> float:
        """The door's HTTP, the arbiter's step and the neck_target hop."""
        return self._latency_s

    def ready(self) -> str | None:
        """The arbiter decides when the overlay arrives (a better band refuses it)."""
        return None

    def play(self, clip: Clip) -> Outcome:
        """Post the clip's keys as an overlay, on a thread of its own."""
        request = clip_look(
            clip, GESTURE if clip.layer != "idle" else IDLE_LOOK, self.max_speed_deg_s
        )
        if request is None:
            return Outcome(False, "the clip has no head track")

        def post() -> None:
            answer = self.door.look(request)
            self.answers.append(answer)
            if answer is not None:
                logger.info("head overlay refused: %s", answer)

        self.run(post)
        return Outcome(True, seconds=clip.duration_s)

    def fade(self, layer: Layer, seconds: float) -> None:
        """The overlay withdrawn: the head back on its holder's aim."""
        source = IDLE_LOOK if layer == "idle" else GESTURE
        self.run(lambda: self.door.withdraw(source))

    def rest(self) -> None:
        """Both overlays withdrawn."""

        def withdraw() -> None:
            self.door.withdraw(GESTURE)
            self.door.withdraw(IDLE_LOOK)

        self.run(withdraw)


def _thread(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, name="head-door", daemon=True).start()


def clip_look(clip: Clip, source: str, max_speed_deg_s: float) -> ClipLook | None:
    """A head clip's pan and tilt offsets (degrees) as the arbiter's overlay: one key at every
    key time of either track, in radians, each with the speed ceiling of the move into it
    (the move's own speed with headroom, within the neck's), the last back at zero."""

    tracks = {t.dof: t for t in clip.tracks if t.dof in ("pan_deg", "tilt_deg")}
    if not tracks:
        return None
    times = sorted({t for track in tracks.values() for t in track.times})
    keys: list[ClipKey] = []
    before = (0.0, 0.0, 0.0)
    for t in times:
        pan = track_value(tracks["pan_deg"], t) if "pan_deg" in tracks else 0.0
        tilt = track_value(tracks["tilt_deg"], t) if "tilt_deg" in tracks else 0.0
        dt = t - before[0]
        travel = max(abs(pan - before[1]), abs(tilt - before[2]))
        speed = HEAD_SPEED_HEADROOM * travel / dt if dt > 0 else max_speed_deg_s
        keys.append({
            "at_s": t,
            "pan_rad": math.radians(pan),
            "tilt_rad": math.radians(tilt),
            "speed_deg_s": min(max_speed_deg_s, max(HEAD_MIN_SPEED_DEG_S, speed)),
        })  # fmt: skip
        before = (t, pan, tilt)
    return {
        "source": source,
        "kind": "clip",
        "band": BAND,
        "ttl_s": clip.duration_s + 1.0,
        "target": {"keys": keys},
    }


class Absent:
    """A part that cannot play now, and why (the wheels; the head until the arbiter's overlay
    lands): never ready, every clip refused, so a gesture plays without it."""

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
