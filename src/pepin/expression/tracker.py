"""The speaker gaze: the head on the person the robot talks with, the head's base layer.

A conversation opens on the robot's name: the head turns to the voice (the array's direction of
arrival, :mod:`pepin.expression.doa`), takes a frame, finds the face, centres it and smiles once,
then holds it while the conversation lasts::

    idle --opened--> turn to the voice --> seek (up to seek_frames, seek_every_s apart)
      seek --a face--> refine onto it, smile once --> track
      seek --no face--> voice (stays on the voice, no smile; a frame every track_every_s)
      voice --a face--> refine, smile --> track
      track: a frame every track_every_s while the head stands still and is ours; the face off the
             centre by more than deadband_deg is re-aimed (slow up to saccade_over_deg, else a
             saccade); an utterance whose voice is doa_jump_deg off the person turns to it again
      drive(True) --> paused (the arbiter let the head go at the drive's start)
      drive(False) --> the next utterance orients again
      closed, or the switch off --> release --> idle (no frames, no calls)

The head is held through the arbiter's door as an ``angles`` look of the idle band, source
``voice.speaker`` (:mod:`pepin.expression.head`): TTL ``ttl_s``, renewed every ``renew_s``
(an aim renews it too); a renewal that finds nothing live (a ``look`` the person asked for, a
stall look took the head) asks again and queues behind it, so the head comes back when that look
ends. While the robot speaks its own voice is in the array: an utterance overlapping the robot's
speech moves nothing (frames go on). A frame counts only while the head is ours, still and
carries no gesture (an overlay other than the idle layer's breath): a thinking glance pulls the
head away and brings it back without the target being lost. Past ``comfort_pan_deg`` the wheels
turn the rest (:mod:`pepin.expression.plan`; refused by the stub tonight: the head looks as far as
it reaches). The smile goes through the head server's face owner (source ``expression``).

:class:`SpeakerGaze` is the machine, single-threaded and clocked by its caller (tests drive it on
a fake clock); :class:`SpeakerGazeWorker` is the cascade's observer and the live switch
``speaker_gaze``: one daemon thread that sleeps on its queue between conversations.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pepin.expression.conversation import SPEAKER_GAZE, Heard
from pepin.expression.doa import VoiceBearing, utterance_readings, voice_of
from pepin.expression.executors import FaceChannel, WheelMove, Wheels
from pepin.expression.head import IDLE_LOOK, SPEAKER
from pepin.expression.markers import Part
from pepin.expression.plan import BodyLimits, head_after, split
from pepin.expression.policy import SpeakerPolicy, face_aim
from pepin.expression.speaker import DoaMap, DoaSource, FaceFinder, GazeDoor, HeadView
from pepin.gaze import Aim, Reach, Speed

if TYPE_CHECKING:
    from pepin.expression.conversation import ConversationObserver, LiveFlag
    from pepin.expression.door import SpeakerDoor
    from pepin.expression.finder import SnapshotFinder
    from pepin.expression.plan import BoundedTurn
    from pepin.expression.speaker import TargetPolicy
    from pepin.voice_live.sources import BoardLink

logger = logging.getLogger(__name__)

Phase = Literal["idle", "seek", "voice", "track", "paused"]
CONFIG_FILE = "expression.json"
FACE_SOURCE = "expression"  # the face's expressions of the body layer, at the head server
BLIND_RETRY_S = 0.1  # a frame refused for a blind head is tried again this soon
SPOKE_KEPT = 8  # the robot's latest speaking spans kept to judge an utterance by
Report = Callable[[str], None]


def say(line: str) -> None:
    """One line of the speaker gaze in the cascade's output."""
    print(f"  (gaze: {line})", flush=True)


@dataclass(frozen=True)
class SpeakerSettings:
    """config/expression.json's ``speaker`` block (and ``parts.body``): the file's notes say
    what each is; ``doa_map`` is None while the array's front and sign are unmeasured."""

    doa_map: DoaMap | None = None
    window_s: float = 1.0
    min_speech: int = 2
    comfort_pan_deg: float = 60.0
    in_frame_deg: float = 25.0
    seek_tilt_deg: float = -5.0
    seek_frames: int = 3
    seek_every_s: float = 0.4
    track_every_s: float = 1.0
    match_deg: float = 20.0
    refine_deg: float = 2.0
    deadband_deg: float = 6.0
    saccade_over_deg: float = 15.0
    doa_jump_deg: float = 30.0
    ttl_s: float = 6.0
    renew_s: float = 2.0
    aim_wait_s: float = 2.0
    smile: str = "smile"
    smile_hold_s: float = 2.0
    smile_unseen: bool = False
    body: BodyLimits = field(default_factory=BodyLimits)

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> SpeakerSettings:
        """From config/expression.json's object; ``ValueError`` naming a value no run can use."""
        speaker = data.get("speaker", {})
        doa = speaker.get("doa", {})
        front, sign = doa.get("front_deg"), doa.get("sign")
        doa_map = None
        if front is not None and sign is not None:
            if int(sign) not in (-1, 1):
                raise ValueError(f"speaker.doa.sign is +1 or -1, not {sign!r}")
            doa_map = DoaMap(float(front) % 360.0, int(sign))
        d = cls()
        values: dict[str, Any] = {}
        for name, default in vars(d).items():
            if name in ("doa_map", "body", "window_s", "min_speech"):
                continue
            if name in speaker:
                values[name] = type(default)(speaker[name])
        body_block = data.get("parts", {}).get("body", {})
        body = BodyLimits(
            **{
                name: type(default)(body_block[name])
                for name, default in vars(BodyLimits()).items()
                if name in body_block
            }
        )
        settings = replace(
            d,
            doa_map=doa_map,
            window_s=float(doa.get("window_s", d.window_s)),
            min_speech=int(doa.get("min_speech", d.min_speech)),
            body=body,
            **values,
        )
        if settings.renew_s >= settings.ttl_s:
            raise ValueError("speaker: renew_s must be shorter than ttl_s")
        if settings.seek_frames < 0 or settings.track_every_s <= 0 or settings.seek_every_s <= 0:
            raise ValueError("speaker: seek_frames >= 0, seek_every_s and track_every_s > 0")
        return settings

    @classmethod
    def load(cls, path: str | Path | None = None) -> SpeakerSettings:
        """Read config/expression.json (or ``path``)."""
        return cls.from_config(_read_config(path))


def flag_default(path: str | Path | None = None) -> bool:
    """The ``speaker_gaze`` switch as config/expression.json ships it."""
    return bool(_read_config(path).get("flags", {}).get(SPEAKER_GAZE, False))


def _read_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        from pepin.deployment import config_file

        path = config_file(CONFIG_FILE)
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} holds no JSON object")
    return data


class SpeakerGaze:
    """The speaker gaze's state machine: every moment and every tick from one thread, each call
    to the door, the finder, the wheels and the face made from it, ``now`` the laptop's clock."""

    def __init__(
        self,
        settings: SpeakerSettings,
        *,
        door: GazeDoor,
        doa: DoaSource | None,
        finder: FaceFinder | None,
        wheels: Wheels,
        face: FaceChannel | None,
        reach: Reach,
        report: Report = say,
    ) -> None:
        """The seams it drives; ``doa`` None or ``finder`` None: no voice or no faces."""
        self.door, self.doa, self.finder = door, doa, finder
        self.wheels, self.face, self.reach = wheels, face, reach
        self.report = report
        self.settings = settings
        self.policy = self._policy(settings)
        self.phase: Phase = "idle"
        self.conversation = False
        self.voice: float | None = None  # the person's bearing in the cart's frame (radians)
        self.aim: Aim | None = None  # the aim last asked of the arbiter
        self.seek_left = 0
        self.next_frame: float | None = None
        self.next_renew: float | None = None
        self.smiled = False
        self.reorient = False  # the next utterance turns the head (after a drive, a switch)
        self.spoke: list[tuple[float, float | None]] = []  # the robot's speaking spans
        self.counts: Counter[str] = Counter()

    def configure(self, settings: SpeakerSettings) -> None:
        """New knobs (the switch turned on re-reads them)."""
        self.settings = settings
        self.policy = self._policy(settings)

    @staticmethod
    def _policy(settings: SpeakerSettings) -> SpeakerPolicy:
        return SpeakerPolicy(match_deg=settings.match_deg, doa_jump_deg=settings.doa_jump_deg)

    # ---- the moments ---------------------------------------------------------------------------
    def opened(self, wake: Heard, now: float) -> None:
        """A conversation opened on ``wake``: turn to its voice."""
        self.conversation, self.smiled, self.reorient = True, False, False
        voice = self.bearing(wake, now)
        self.report(f"wake: {voice.text()}")
        self._orient(voice.bearing_rad, now)

    def heard(self, heard: Heard, now: float) -> None:
        """An utterance of the conversation: a new speaker (or the first after a drive) turns
        the head; one over the robot's own voice, or noise, moves nothing."""
        if not self.conversation or heard.noise:
            return
        if self._robot_spoke(heard):
            self.counts["over_robot"] += 1
            return
        if self.phase == "paused" and not self.reorient:
            return  # a drive runs
        voice = self.bearing(heard, now)
        if self.reorient:
            self.reorient = False
            self.report(f"after a pause: {voice.text()}")
            self._orient(voice.bearing_rad, now)
        elif self.policy.new_speaker(self.voice, voice.bearing_rad):
            self.report(f"a voice elsewhere: {voice.text()}")
            self._orient(voice.bearing_rad, now)

    def speaking(self, on: bool, now: float) -> None:
        """The robot's voice starts (a part queued) or stops (a reply done or cut)."""
        open_span = bool(self.spoke) and self.spoke[-1][1] is None
        if on and not open_span:
            self.spoke = [*self.spoke[-(SPOKE_KEPT - 1) :], (now, None)]
        elif not on and open_span:
            self.spoke[-1] = (self.spoke[-1][0], now)

    def drive(self, active: bool, now: float) -> None:
        """A drive started (the arbiter let the head go) or ended (the next utterance orients)."""
        if active:
            self.phase, self.aim, self.next_frame, self.next_renew = "paused", None, None, None
            self.reorient = False
        elif self.conversation:
            self.reorient = True

    def closed(self, reason: str, now: float) -> None:
        """The conversation is over: the head is let go and nothing runs until the next one."""
        self.conversation = False
        self.stop(now, f"the conversation closed ({reason})")

    def stop(self, now: float, why: str) -> None:
        """Let the head go now (the switch off, the conversation's end)."""
        held = self.phase != "idle" or self.aim is not None
        self.phase, self.aim, self.next_frame, self.next_renew = "idle", None, None, None
        self.voice = None
        if held:
            self.door.release()
            self.counts["releases"] += 1
            self.report(f"released: {why}")

    def resume(self, now: float) -> None:
        """The switch turned on mid-conversation: the next utterance orients the head."""
        if self.conversation and self.phase == "idle":
            self.phase, self.reorient = "paused", True

    def warm(self) -> None:
        """Load what the first wake would wait for (the face finder's model); a finder that
        cannot load (no weights) is dropped: the head then stays on the voice."""
        warm = getattr(self.finder, "warm", None)
        if not callable(warm):
            return
        try:
            warm()
        except (OSError, ImportError, RuntimeError, ValueError) as exc:
            self.finder = None
            self.report(f"no face finder ({exc}): the head turns to the voice only")

    # ---- the clock ---------------------------------------------------------------------------
    def tick(self, now: float) -> float | None:
        """Do what is due (a renewal, a frame, a lost head asked for again); when next, None for
        nothing until a moment."""
        if self.phase in ("idle", "paused"):
            return None
        if self.aim is None:  # the door did not answer, or refused: ask again in a while
            if self.next_renew is None or now >= self.next_renew:
                self._orient(self.voice, now)
            return self.next_renew
        if self.next_renew is not None and now >= self.next_renew:
            self._renew(now)
        if self.next_frame is not None and now >= self.next_frame:
            self._frame(now)
        due = [t for t in (self.next_renew, self.next_frame) if t is not None]
        return min(due) if due else None

    # ---- the voice -----------------------------------------------------------------------------
    def bearing(self, heard: Heard, now: float) -> VoiceBearing:
        """``heard``'s voice: its last ``window_s`` of speech readings, mapped to the cart."""
        if self.doa is None:
            return VoiceBearing(None, None, 0, 0)
        s = self.settings
        readings = utterance_readings(self.doa, heard, now, s.window_s)
        return voice_of(readings, s.doa_map, s.min_speech)

    def _robot_spoke(self, heard: Heard) -> bool:
        """Whether the robot spoke during ``heard`` (its own voice is then in the array)."""
        return any(
            start <= heard.end_s and (end is None or end >= heard.start_s)
            for start, end in self.spoke
        )

    # ---- the head ----------------------------------------------------------------------------
    def _orient(self, voice: float | None, now: float) -> None:
        """Turn to ``voice`` (the wheels first past the comfortable pan), then seek a face."""
        s = self.settings
        self.voice = voice
        view = self._state()
        if view is None:
            self._lost(now, "the arbiter's door does not answer")
            return
        if view.driving:
            self.phase, self.reorient, self.aim = "paused", True, None
            return
        head = view.head if view.head is not None else Aim(0.0, math.radians(s.seek_tilt_deg))
        tilt = math.radians(s.seek_tilt_deg)
        pan = head.pan_rad
        if voice is not None and abs(voice - head.pan_rad) > math.radians(s.in_frame_deg):
            pan = self._turn(voice, tilt).pan_rad
        aim = self.reach.clamp(Aim(pan, tilt))
        if not self._hold(aim, "saccade", now):
            return
        self.counts["orients"] += 1
        if self.finder is None:
            self.phase, self.next_frame = "voice", None
            return
        self.phase, self.seek_left, self.next_frame = "seek", s.seek_frames, now
        if s.seek_frames == 0:
            self._unseen()

    def _turn(self, voice: float, tilt: float) -> Aim:
        """The head's aim at ``voice``: past the comfortable pan the wheels turn the rest first
        (as far as they go), and the person's bearing moves with the cart."""
        s = self.settings
        plan = split(
            voice,
            tilt,
            comfort_rad=math.radians(s.comfort_pan_deg),
            max_turn_rad=math.radians(s.body.max_turn_deg),
        )
        turned = 0.0
        if plan.body_rad != 0.0:
            report = self.wheels.move(WheelMove(turn_deg=math.degrees(plan.body_rad)))
            turned = math.radians(report.turned_deg)
            self.counts["wheel_turns"] += 1
            if not report.done:
                self.report(
                    f"the wheels turned {report.turned_deg:+.0f} of"
                    f" {math.degrees(plan.body_rad):+.0f} deg ({report.why}): the head looks as"
                    " far as it reaches"
                )
        self.voice = voice - turned
        return head_after(voice, turned, tilt, self.reach)

    def _hold(self, aim: Aim, speed: Speed, now: float) -> bool:
        """Ask the arbiter to hold the head at ``aim`` (replacing this source's last aim);
        whether the request lives (done, or queued behind a better look)."""
        answer = self.door.aim(aim, speed=speed, ttl_s=self.settings.ttl_s)
        self.counts["aims"] += 1
        self.next_renew = now + self.settings.renew_s
        if answer.status in ("done", "queued"):
            self.aim = aim
            if answer.status != "done":
                self.report(f"the head waits: {answer.status} {answer.reason}".rstrip())
            return True
        self._lost(now, f"{answer.status}: {answer.reason}")
        return False

    def _lost(self, now: float, why: str) -> None:
        """The head is not ours: stay on the voice, ask again after ``renew_s``."""
        self.report(f"no head ({why}); asking again in {self.settings.renew_s:.0f} s")
        self.phase, self.aim, self.next_frame = "voice", None, None
        self.next_renew = now + self.settings.renew_s

    def _renew(self, now: float) -> None:
        """Keep the look alive; one another look took is asked for again (queued behind it)."""
        self.next_renew = now + self.settings.renew_s
        live = self.door.renew()
        self.counts["renewals"] += 1
        if live == 0 and self.aim is not None:
            self.counts["reposts"] += 1
            self._hold(self.aim, "saccade", now)

    # ---- the faces ---------------------------------------------------------------------------
    def _frame(self, now: float) -> None:
        """One frame while the head is ours, still and carries no gesture; the speaker's face
        centred (refined after a turn, re-aimed past the dead band while tracking)."""
        try:
            self._look(now)
        finally:  # at the rate of the phase the frame left the machine in
            if self.next_frame is not None and self.next_frame <= now:
                seeking = self.phase == "seek"
                s = self.settings
                self.next_frame = now + (s.seek_every_s if seeking else s.track_every_s)

    def _look(self, now: float) -> None:
        s = self.settings
        if self.finder is None:
            return
        view = self._state()
        if view is None or view.head is None:
            return
        gesture = bool(view.overlay) and view.overlay != IDLE_LOOK
        if view.holder != SPEAKER or view.driving or gesture:
            self.counts["frames_skipped"] += 1
            self._missed()
            return
        if view.blind:
            self.next_frame = now + BLIND_RETRY_S
            return
        frame = self.finder.look()
        self.counts["frames"] += 1
        face = self.policy.pick(self.voice, frame, view.head)
        if face is None:
            self._missed()
            return
        self.counts["faces"] += 1
        target = self.reach.clamp(face_aim(view.head, face))
        off = target.off(self.aim if self.aim is not None else view.head)
        band = s.refine_deg if self.phase != "track" else s.deadband_deg
        if off > math.radians(band):
            speed: Speed = "slow" if off <= math.radians(s.saccade_over_deg) else "saccade"
            if not self._hold(target, speed, now):
                return
            self.counts["reaims"] += 1
        self.voice = target.pan_rad
        if self.phase != "track":
            pan, tilt = math.degrees(target.pan_rad), math.degrees(target.tilt_rad)
            self.report(f"a face at {pan:+.0f}/{tilt:+.0f} deg, {face.width_px:.0f} px: tracking")
            self.phase = "track"
        self._smile()

    def _missed(self) -> None:
        """A frame without the speaker's face: a seek runs out into staying on the voice."""
        if self.phase != "seek":
            return
        self.seek_left -= 1
        if self.seek_left <= 0:
            self._unseen()

    def _unseen(self) -> None:
        self.phase = "voice"
        self.report("no face: the head stays on the voice")
        if self.settings.smile_unseen:
            self._smile()

    def _smile(self) -> None:
        """Once a conversation, through the face's owner."""
        if self.smiled or self.face is None or not self.settings.smile:
            return
        self.smiled = True
        self.face.express(self.settings.smile, hold_s=self.settings.smile_hold_s)
        self.counts["smiles"] += 1

    def _state(self) -> HeadView | None:
        self.counts["states"] += 1
        return self.door.state()


class SpeakerGazeWorker:
    """The speaker gaze in the cascade: a :class:`pepin.expression.conversation.
    ConversationObserver` and the :class:`pepin.expression.conversation.LiveFlag`
    ``speaker_gaze``. Every moment is one non-blocking put on a queue; one daemon thread (started
    the first time the switch is on) runs the machine and sleeps on the queue between
    conversations, so the switch off and the robot alone cost nothing."""

    def __init__(
        self,
        build: Callable[[SpeakerSettings], SpeakerGaze],
        *,
        on: bool,
        load: Callable[[], SpeakerSettings] = SpeakerSettings.load,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``build`` makes the machine (once, on the thread) from the knobs ``load`` reads."""
        self._build, self._load, self._clock = build, load, clock
        self._on = False
        self._queue: queue.SimpleQueue[tuple[str, tuple[Any, ...], float]] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self.machine: SpeakerGaze | None = None
        self.idle = threading.Event()  # set while the thread waits with nothing due (tests)
        if on:
            why = self.switch(True)
            if why is not None:
                logger.warning("speaker gaze stays off: %s", why)

    # ---- the switch ----------------------------------------------------------------------------
    @property
    def flag(self) -> str:
        """``speaker_gaze``."""
        return SPEAKER_GAZE

    @property
    def on(self) -> bool:
        """Whether the speaker gaze is on."""
        return self._on

    def switch(self, on: bool) -> str | None:
        """On re-reads the knobs (refused, with why, when the file is unreadable); off lets the
        head go."""
        if on:
            try:
                settings = self._load()
            except (OSError, ValueError, TypeError) as exc:
                return f"config/{CONFIG_FILE}: {exc}"
            self._on = True
            self._post("on", settings)
        elif self._on:
            self._post("off")
            self._on = False
        return None

    def close(self, timeout_s: float = 2.0) -> None:
        """Let the head go and stop the thread."""
        if self._thread is None:
            return
        self._queue.put(("stop", (), self._clock()))
        self._thread.join(timeout_s)

    # ---- the observer ------------------------------------------------------------------------
    def opened(self, wake: Heard) -> None:
        """A conversation opened."""
        self._post("opened", wake)

    def speech_started(self, t: float) -> None:
        """Nothing: an utterance counts once it is read."""

    def speech_ended(self, heard: Heard) -> None:
        """An utterance was read."""
        self._post("heard", heard)

    def part_queued(self, turn: int, index: int, part: Part) -> None:
        """The robot speaks."""
        self._post("speaking", True)

    def part_spoken(self, turn: int, index: int) -> None:
        """Nothing."""

    def reply_cut(self, turn: int) -> None:
        """The robot stopped speaking."""
        self._post("speaking", False)

    def reply_done(self, turn: int) -> None:
        """The robot stopped speaking."""
        self._post("speaking", False)

    def moment(self, name: str) -> None:
        """Nothing: a moment is the gestures'."""

    def drive(self, active: bool) -> None:
        """A drive began or ended."""
        self._post("drive", active)

    def closed(self, reason: str) -> None:
        """The conversation closed."""
        self._post("closed", reason)

    # ---- the thread --------------------------------------------------------------------------
    def _post(self, kind: str, *args: Any) -> None:
        if not self._on and kind != "off":
            return
        self._queue.put((kind, args, self._clock()))
        self.idle.clear()  # after the put: the thread sets it again only once it has seen it
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="speaker-gaze", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        due: float | None = None
        while True:
            timeout = None if due is None else max(0.0, due - self._clock())
            if due is None:
                self.idle.set()
                if not self._queue.empty():
                    self.idle.clear()
            try:
                kind, args, at = self._queue.get(timeout=timeout)
            except queue.Empty:
                kind, args, at = "tick", (), self._clock()
            self.idle.clear()
            if kind == "stop":
                if self.machine is not None:
                    self._guarded(lambda m: m.stop(self._clock(), "the cascade stops"))
                return
            if kind == "on":
                self._switched_on(args[0])
            elif self.machine is not None:
                self._apply(kind, args, at)
            due = self._guarded(lambda m: m.tick(self._clock()))

    def _switched_on(self, settings: SpeakerSettings) -> None:
        try:
            if self.machine is None:
                self.machine = self._build(settings)
            self.machine.configure(settings)
            self.machine.warm()
            self.machine.resume(self._clock())
        except Exception:  # a broken seam never stops the voice: the gaze stays idle
            logger.exception("speaker gaze could not start")

    def _apply(self, kind: str, args: tuple[Any, ...], at: float) -> None:
        now = self._clock()
        actions: dict[str, Callable[[SpeakerGaze], None]] = {
            "off": lambda m: m.stop(now, "the switch is off"),
            "opened": lambda m: m.opened(args[0], now),
            "heard": lambda m: m.heard(args[0], now),
            "speaking": lambda m: m.speaking(args[0], at),
            "drive": lambda m: m.drive(args[0], now),
            "closed": lambda m: m.closed(args[0], now),
        }
        action = actions.get(kind)
        if action is not None:
            self._guarded(action)

    def _guarded[T](self, action: Callable[[SpeakerGaze], T]) -> T | None:
        machine = self.machine
        if machine is None:
            return None
        try:
            return action(machine)
        except Exception:  # a door that throws, a finder that fails: logged, the voice goes on
            logger.exception("speaker gaze")
            return None


def robot_gaze(host: str, doa: DoaSource, *, on: bool | None = None) -> SpeakerGazeWorker:
    """The speaker gaze on the robot: the arbiter's door on this Mac, the board's camera and
    head server at ``host``, the array's directions from ``doa`` (the cascade's audio link), the
    wheels' stub; built, and the finder's model loaded, the first time its switch is on (config/
    expression.json's ``flags.speaker_gaze`` at the start when ``on`` is None)."""

    def build(settings: SpeakerSettings) -> SpeakerGaze:
        from pepin.deployment import config_file
        from pepin.expression.door import SpeakerDoor
        from pepin.expression.executors import NoWheels
        from pepin.expression.finder import board_finder
        from pepin.head_link import HeadClient
        from pepin.neck import NeckConfig

        finder: SnapshotFinder | None = None
        try:
            finder = board_finder(host)  # config reads only: the model loads on warm()
        except (OSError, ValueError, KeyError) as exc:
            say(f"no face finder ({exc}): the head turns to the voice only")
        return SpeakerGaze(
            settings,
            door=SpeakerDoor(aim_wait_s=settings.aim_wait_s),
            doa=doa,
            finder=finder,
            wheels=NoWheels(),
            face=HeadClient(host, source=FACE_SOURCE).start(),
            reach=Reach.of(NeckConfig.from_json(config_file("neck.json"))),
        )

    return SpeakerGazeWorker(build, on=flag_default() if on is None else on)


if TYPE_CHECKING:  # the speaker gaze's real parts satisfy their seams

    def _conforms(
        worker: SpeakerGazeWorker,
        door: SpeakerDoor,
        finder: SnapshotFinder,
        turn: BoundedTurn,
        policy: SpeakerPolicy,
        link: BoardLink,
    ) -> None:
        observer: ConversationObserver = worker
        flag: LiveFlag = worker
        gaze_door: GazeDoor = door
        faces: FaceFinder = finder
        wheels: Wheels = turn
        target: TargetPolicy = policy
        doa: DoaSource = link
        del observer, flag, gaze_door, faces, wheels, target, doa
