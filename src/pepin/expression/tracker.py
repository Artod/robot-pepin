"""The speaker gaze: the head on the person the robot talks with, the head's base layer.

A conversation opens on the robot's name: the head turns to the voice (the array's direction of
arrival, :mod:`pepin.expression.doa`), takes a frame, finds the face, centres it and smiles once,
then follows whoever speaks while the conversation lasts::

    idle --opened--> turn to the voice --> seek (seek_frames, seek_every_s apart, at seek_tilt_deg
                     and again seek_sweep_deg higher)
      seek --a face--> refine onto it, smile once (a face of track_min_face_px or more) --> track
      seek --no face--> voice (back at seek_tilt_deg, stays on the voice, no smile)
      voice --a face--> refine, smile --> track
      track: track_hz frames while the head stands still, is ours and carries no gesture; a face
             off the centre by more than deadband_deg is re-aimed where it will be at the next
             frame (constant velocity over the last two sightings, at most lead_max_deg ahead;
             slow up to saccade_over_deg, else a saccade)
      track --no face in frames for face_lost_s--> voice (the head stays where it is)
      drive(True) --> paused (the arbiter let the head go at the drive's start)
      drive(False) --> the next utterance with words orients again
      closed, or the switch off --> release --> idle (no frames, no calls)

Who speaks now is looked at: look, verify, return. Each utterance is judged by its bearing:

* a sound is no speaker: without words (noise, nothing read), with fewer than ``doa_min_speech``
  of the array's speech-flagged readings, or with a peak under ``doa_min_dbfs`` (a quiet voice
  gives a wrong bearing: on 2026-10-12 a question at -31 dBFS read -76 deg off a face tracked
  at +25) it never moves the head alone; ``reaim_votes`` word-less sounds in a row from
  about one place (within ``reaim_deg`` of each other) make a look there, like an utterance;
* an utterance with words ``reaim_deg`` or more off the person held is someone speaking: the head
  looks there at once (at the remembered place of a person heard there before, else at the
  bearing). While a face is tracked the look is a check: one frame at the tilt and one
  ``seek_sweep_deg`` higher within ``verify_s`` of the first; a face there becomes the target,
  none and the head goes back to the face it left. With no face held it is a full seek;
* the robot's name in an utterance always turns the head to it (a full seek, the head staying on
  the voice when no face is found), the gates aside.

Each person looked at is remembered (``remember_targets`` of them: the face's place, its last
box, when): the seam for two interlocutors, and the way back after a check that found nobody.

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

Every decision prints one line (``say``) and, through :class:`pepin.expression.conversation.
EventSink`, lands in the conversation's events.jsonl on the session clock: ``gaze_wake``,
``gaze_voice`` (each utterance's bearing and verdict), ``gaze_reaim`` (each aim asked of the
arbiter and why), ``gaze_face`` (a face found), ``gaze_lost`` (a seek, a check or a face that came
to nothing; the door lost) and ``gaze_release``.

:class:`SpeakerGaze` is the machine, single-threaded and clocked by its caller (tests drive it on
a fake clock); :class:`SpeakerGazeWorker` is the cascade's observer and the live switch
``speaker_gaze``: one daemon thread that sleeps on its queue between conversations. Every
observer hook is one queue put: the gaze never delays the ears, the brain or the voice.
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

from pepin.audio_signal import circular_mean_deg
from pepin.expression.conversation import SPEAKER_GAZE, EventSink, Heard
from pepin.expression.doa import VoiceBearing, fit_doa_map, utterance_readings, voice_of
from pepin.expression.executors import FaceChannel, WheelMove, Wheels
from pepin.expression.head import IDLE_LOOK, SPEAKER
from pepin.expression.markers import Part
from pepin.expression.plan import BodyLimits, head_after, split
from pepin.expression.policy import SpeakerPolicy, face_aim
from pepin.expression.speaker import (
    DoaMap,
    DoaSource,
    FaceFinder,
    FaceFrame,
    FaceSighting,
    GazeDoor,
    HeadView,
    Target,
)
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
TEXT_KEPT = 40  # an utterance's words kept in its event, to pair it with the loop's "heard"
Report = Callable[[str], None]
FaceListener = Callable[[FaceFrame, FaceSighting], None]  # a frame and the speaker's face in it


def say(line: str) -> None:
    """One line of the speaker gaze in the cascade's output."""
    print(f"  (gaze: {line})", flush=True)


@dataclass(frozen=True)
class SpeakerSettings:
    """config/expression.json's ``speaker`` block (and ``parts.body``): the file's notes say
    what each is; ``doa_map`` is None while the array is unmeasured."""

    doa_map: DoaMap | None = None
    window_s: float = 1.0
    min_speech: int = 2
    comfort_pan_deg: float = 60.0
    in_frame_deg: float = 25.0
    seek_tilt_deg: float = -5.0
    seek_sweep_deg: float = 10.0
    seek_frames: int = 3
    seek_every_s: float = 0.4
    seek_min_face_px: float = 28.0
    track_min_face_px: float = 40.0
    track_hz: float = 3.0
    lead_max_deg: float = 10.0
    match_deg: float = 20.0
    refine_deg: float = 2.0
    deadband_deg: float = 6.0
    saccade_over_deg: float = 15.0
    reaim_deg: float = 35.0
    verify_s: float = 0.7
    reaim_votes: int = 2
    face_lost_s: float = 2.0
    doa_min_speech: int = 4
    doa_min_dbfs: float = -28.0
    remember_targets: int = 3
    ttl_s: float = 6.0
    renew_s: float = 2.0
    aim_wait_s: float = 2.0
    smile: str = "smile"
    smile_hold_s: float = 2.0
    smile_unseen: bool = False
    body: BodyLimits = field(default_factory=BodyLimits)

    @property
    def track_every_s(self) -> float:
        """The time between tracking frames."""
        return 1.0 / self.track_hz

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> SpeakerSettings:
        """From config/expression.json's object; ``ValueError`` naming a value no run can use."""
        speaker = data.get("speaker", {})
        doa = speaker.get("doa", {})
        doa_map = doa_map_of(doa)
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
        settings.check()
        return settings

    def check(self) -> None:
        """``ValueError`` naming a value no run can use."""
        if self.renew_s >= self.ttl_s:
            raise ValueError("speaker: renew_s must be shorter than ttl_s")
        if self.seek_frames < 0 or self.track_hz <= 0 or self.seek_every_s <= 0:
            raise ValueError("speaker: seek_frames >= 0, seek_every_s and track_hz > 0")
        if self.reaim_votes < 1 or self.remember_targets < 0 or self.verify_s < 0:
            raise ValueError("speaker: reaim_votes >= 1, remember_targets and verify_s >= 0")
        if self.lead_max_deg < 0 or self.face_lost_s < 0 or self.seek_sweep_deg < 0:
            raise ValueError("speaker: lead_max_deg, face_lost_s and seek_sweep_deg >= 0")

    @classmethod
    def load(cls, path: str | Path | None = None) -> SpeakerSettings:
        """Read config/expression.json (or ``path``)."""
        return cls.from_config(_read_config(path))


def doa_map_of(doa: Mapping[str, Any]) -> DoaMap | None:
    """``speaker.doa``'s map: interpolated between its four ``readings`` when all are measured,
    else the offset of ``front_deg`` and ``sign``, else None (unmeasured)."""
    readings = doa.get("readings") or {}
    places = [readings.get(p) for p in ("front", "left", "behind", "right")]
    known = [float(v) for v in places if v is not None]
    if len(known) == len(places):
        fit = fit_doa_map(*known)
        if fit.refusal is not None:
            raise ValueError(f"speaker.doa.readings: {fit.refusal}")
        return fit.doa_map
    front, sign = doa.get("front_deg"), doa.get("sign")
    if front is None or sign is None:
        return None
    if int(sign) not in (-1, 1):
        raise ValueError(f"speaker.doa.sign is +1 or -1, not {sign!r}")
    return DoaMap(float(front) % 360.0, int(sign))


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
        on_face: FaceListener | None = None,
        events: EventSink | None = None,
    ) -> None:
        """The seams it drives; ``doa`` None or ``finder`` None: no voice or no faces;
        ``on_face`` hears every frame's face picked as the speaker's (person recognition);
        ``events`` takes every decision as an event of the conversation's log."""
        self.door, self.doa, self.finder = door, doa, finder
        self.wheels, self.face, self.reach = wheels, face, reach
        self.report = report
        self.on_face = on_face
        self.events = events
        self.settings = settings
        self.policy = self._policy(settings)
        self.phase: Phase = "idle"
        self.conversation = False
        self.voice: float | None = None  # the person's bearing in the cart's frame (radians)
        self.aim: Aim | None = None  # the aim last asked of the arbiter
        self.seek_left = 0  # frames left at this tilt of a seek
        self.tilts_left: list[float] = []  # the seek's higher tilts still to try (radians)
        self.verifying = False  # the seek is a check that returns to ``fallback``
        self.verify_until: float | None = None  # the check's end, set at its first frame
        self.fallback: Target | None = None  # the face a check returns to when it finds nobody
        self.target: Target | None = None  # the face tracked now
        self.targets: list[Target] = []  # the people looked at before, the latest first
        self.seen: tuple[float, float] | None = None  # the last sighting: when, its bearing
        self.votes: list[float] = []  # bearings of word-less sounds in a row from elsewhere
        self.next_frame: float | None = None
        self.next_renew: float | None = None
        self.smiled = False
        self.reorient = False  # the next utterance turns the head (after a drive, a switch)
        self.spoke: list[tuple[float, float | None]] = []  # the robot's speaking spans
        self.wake: Heard | None = None  # the cascade tells it once more as heard: not twice
        self.counts: Counter[str] = Counter()

    def configure(self, settings: SpeakerSettings) -> None:
        """New knobs (the switch turned on re-reads them)."""
        self.settings = settings
        self.policy = self._policy(settings)

    @staticmethod
    def _policy(settings: SpeakerSettings) -> SpeakerPolicy:
        return SpeakerPolicy(
            match_deg=settings.match_deg,
            doa_jump_deg=settings.reaim_deg,
            min_px=settings.track_min_face_px,
            near_min_px=settings.seek_min_face_px,
        )

    # ---- the moments ---------------------------------------------------------------------------
    def opened(self, wake: Heard, now: float) -> None:
        """A conversation opened on ``wake``: turn to its voice."""
        self.conversation, self.smiled, self.reorient = True, False, False
        self.wake = wake
        self.targets, self.target, self.votes = [], None, []
        voice = self.bearing(wake, now)
        self._note("gaze_wake", now, f"wake: {voice.text()}", **_voice_fields(voice, wake))
        self._orient(voice.bearing_rad, now, "wake")

    def heard(self, heard: Heard, now: float) -> None:
        """An utterance of the conversation: whoever speaks is looked at (a check while a face
        is tracked); a sound without words, a quiet one, one over the robot's own voice or the
        wake told again moves nothing alone."""
        if not self.conversation or heard == self.wake:
            return
        if self._robot_spoke(heard):
            self.counts["over_robot"] += 1
            self._judged(now, heard, None, "over the robot's own voice")
            return
        if self.phase == "paused" and not self.reorient:
            return  # a drive runs
        voice = self.bearing(heard, now)
        bearing = voice.bearing_rad
        if self.reorient:
            if not heard.words:
                self._judged(now, heard, voice, "no words")
                return
            self.reorient = False
            self._judged(now, heard, voice, "after a pause", f"after a pause: {voice.text()}")
            self._look_toward(bearing, now, "after a pause", check=False)
            return
        if bearing is None:
            self._judged(now, heard, voice, "no bearing")
            return
        elsewhere = self.policy.new_speaker(self.voice, bearing)
        if heard.named and heard.words:
            if not elsewhere and self.phase == "track":
                self._judged(now, heard, voice, "the robot's name, from the face held")
                return
            self._judged(now, heard, voice, "the robot's name", f"the name: {voice.text()}")
            self._look_toward(bearing, now, "the robot's name", check=False)
            return
        doubt = self._doubt(voice, heard)
        if doubt is not None:
            self.counts["sounds_ignored"] += 1
            self._judged(now, heard, voice, doubt, f"not moved by {voice.text()}: {doubt}")
            return
        if not elsewhere:
            self.votes = []
            self._judged(now, heard, voice, "agrees")
            return
        if not heard.words:
            self._vote(heard, voice, bearing, now)
            return
        self._judged(now, heard, voice, "elsewhere", f"a voice elsewhere: {voice.text()}")
        self._look_toward(bearing, now, "a voice elsewhere", check=self.phase == "track")

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
            self._forget_track()
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
        self.voice, self.target, self.targets = None, None, []
        self._forget_track()
        if held:
            self.door.release()
            self.counts["releases"] += 1
            self._note("gaze_release", now, f"released: {why}", why=why)

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
                self._orient(self.voice, now, "asked again")
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

    def _doubt(self, voice: VoiceBearing, heard: Heard) -> str | None:
        """Why the utterance's bearing is no speaker's (too few speech readings, too quiet);
        None when it is one."""
        s = self.settings
        if voice.speech < s.doa_min_speech:
            return f"{voice.speech} speech readings, under {s.doa_min_speech}"
        if heard.peak_db is not None and heard.peak_db < s.doa_min_dbfs:
            return f"quiet, peak {heard.peak_db:.1f} dBFS under {s.doa_min_dbfs:.0f}"
        return None

    def _vote(self, heard: Heard, voice: VoiceBearing, bearing: float, now: float) -> None:
        """A word-less sound from elsewhere: ``reaim_votes`` in a row from about one place make
        a look there."""
        s = self.settings
        reach = math.radians(s.reaim_deg)
        if self.votes and abs(_wrap(bearing - _mean(self.votes))) > reach:
            self.votes = []  # somewhere else again: a new run
        self.votes.append(bearing)
        n = len(self.votes)
        if n < s.reaim_votes:
            self._judged(now, heard, voice, f"no words: vote {n} of {s.reaim_votes}")
            return
        where = _mean(self.votes)
        self.votes = []
        self._judged(
            now, heard, voice, f"no words: {n} in a row", f"{n} sounds from {_deg(where):+.0f} deg"
        )
        self._look_toward(where, now, "sounds in a row", check=self.phase == "track")

    def _robot_spoke(self, heard: Heard) -> bool:
        """Whether the robot spoke during ``heard`` (its own voice is then in the array)."""
        return any(
            start <= heard.end_s and (end is None or end >= heard.start_s)
            for start, end in self.spoke
        )

    # ---- the head ----------------------------------------------------------------------------
    def _look_toward(self, bearing: float | None, now: float, why: str, *, check: bool) -> None:
        """Look where a voice came from: at a remembered person's place near it, else at the
        bearing; ``check``: the face tracked now is kept to return to when nobody is there."""
        held = self.target if self.phase == "track" else None
        if held is not None:
            self._remember(held)
        place = None
        if bearing is not None:
            known = self.policy.recall(
                [t for t in self.targets if t is not held],
                bearing,
                within_rad=math.radians(self.settings.reaim_deg),
            )
            place = known.aim if known is not None else None
        self.fallback = held if check else None
        self.target, self.votes = None, []
        if place is not None:
            why = f"{why}, a person heard there before"
        self._orient(bearing, now, why, place=place, verify=check and held is not None)

    def _orient(
        self,
        voice: float | None,
        now: float,
        why: str,
        *,
        place: Aim | None = None,
        verify: bool = False,
    ) -> None:
        """Turn to ``voice`` (the wheels first past the comfortable pan) or to a remembered
        ``place``, then seek a face: ``seek_frames`` at each tilt, or (``verify``) one frame at
        each within ``verify_s``."""
        s = self.settings
        self.voice = voice if place is None else place.pan_rad
        self.seen = None
        view = self._state()
        if view is None:
            self._lost(now, "the arbiter's door does not answer")
            return
        if view.driving:
            self.phase, self.reorient, self.aim = "paused", True, None
            return
        head = view.head if view.head is not None else Aim(0.0, math.radians(s.seek_tilt_deg))
        tilt = math.radians(s.seek_tilt_deg) if place is None else place.tilt_rad
        pan = head.pan_rad
        toward = self.voice
        if toward is not None and (
            place is not None or abs(toward - head.pan_rad) > math.radians(s.in_frame_deg)
        ):
            pan = self._turn(toward, tilt).pan_rad
        aim = self.reach.clamp(Aim(pan, tilt))
        if not self._hold(aim, "saccade", now, why):
            return
        self.counts["orients"] += 1
        sweep = math.radians(s.seek_sweep_deg)
        self.tilts_left = [tilt - sweep] if sweep > 0.0 else []
        self.verifying, self.verify_until = verify and self.finder is not None, None
        if self.finder is None:
            self.phase, self.next_frame, self.fallback = "voice", None, None
            return
        self.phase, self.next_frame = "seek", now
        self.seek_left = 1 if self.verifying else s.seek_frames
        if self.seek_left == 0:
            self._unseen(now)

    def _turn(self, voice: float, tilt: float) -> Aim:
        """The head's aim at ``voice``: past the comfortable pan the wheels turn the rest first
        (as far as they go), and the person's bearing (and every remembered one) moves with the
        cart."""
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
        if turned != 0.0:
            self.targets = [_turned(t, turned) for t in self.targets]
            if self.fallback is not None:
                self.fallback = _turned(self.fallback, turned)
        self.voice = voice - turned
        return head_after(voice, turned, tilt, self.reach)

    def _hold(self, aim: Aim, speed: Speed, now: float, why: str, **fields: Any) -> bool:
        """Ask the arbiter to hold the head at ``aim`` (replacing this source's last aim);
        whether the request lives (done, or queued behind a better look)."""
        answer = self.door.aim(aim, speed=speed, ttl_s=self.settings.ttl_s)
        self.counts["aims"] += 1
        self.next_renew = now + self.settings.renew_s
        self._note(
            "gaze_reaim",
            now,
            None,
            why=why,
            pan_deg=_deg(aim.pan_rad),
            tilt_deg=_deg(aim.tilt_rad),
            speed=speed,
            status=answer.status,
            **fields,
        )
        if answer.status in ("done", "queued"):
            self.aim = aim
            if answer.status != "done":
                self.report(f"the head waits: {answer.status} {answer.reason}".rstrip())
            return True
        self._lost(now, f"{answer.status}: {answer.reason}")
        return False

    def _lost(self, now: float, why: str) -> None:
        """The head is not ours: stay on the voice, ask again after ``renew_s``."""
        line = f"no head ({why}); asking again in {self.settings.renew_s:.0f} s"
        self._note("gaze_lost", now, line, what="head", why=why)
        self._forget_track()
        self.phase, self.aim, self.next_frame = "voice", None, None
        self.next_renew = now + self.settings.renew_s

    def _renew(self, now: float) -> None:
        """Keep the look alive; one another look took is asked for again (queued behind it)."""
        self.next_renew = now + self.settings.renew_s
        live = self.door.renew()
        self.counts["renewals"] += 1
        if live == 0 and self.aim is not None:
            self.counts["reposts"] += 1
            self._hold(self.aim, "saccade", now, "asked again")

    # ---- the faces ---------------------------------------------------------------------------
    def _frame(self, now: float) -> None:
        """One frame while the head is ours, still and carries no gesture; the speaker's face
        centred (refined after a turn, re-aimed past the dead band while tracking)."""
        try:
            self._look(now)
        finally:  # at the rate of the phase the frame left the machine in
            if self.next_frame is not None and self.next_frame <= now:
                s = self.settings
                if self.phase != "seek":
                    self.next_frame = now + s.track_every_s
                else:  # a check's next frame as soon as the head is still again
                    self.next_frame = now + (BLIND_RETRY_S if self.verifying else s.seek_every_s)

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
            self._missed(now, looked=False)
            return
        if view.blind:
            self.next_frame = now + BLIND_RETRY_S
            return
        if self.verifying:
            if self.verify_until is None:
                self.verify_until = now + s.verify_s
            elif now >= self.verify_until:  # the check ran out of time: back to the face
                self._unseen(now)
                return
        frame = self.finder.look()
        self.counts["frames"] += 1
        tracking = self.phase == "track"
        face = self.policy.pick(self.voice, frame, view.head, seeking=not tracking)
        if face is None:
            self._missed(now, looked=True)
            return
        self.counts["faces"] += 1
        if self.on_face is not None and frame is not None:
            try:
                self.on_face(frame, face)
            except Exception:  # a listener never stops the gaze
                logger.exception("the speaker's face listener broke")
        seen = face_aim(view.head, face)
        seen = Aim(_wrap(seen.pan_rad), seen.tilt_rad)
        lead = self._lead(seen.pan_rad, now) if tracking else 0.0
        target = self.reach.clamp(Aim(seen.pan_rad + lead, seen.tilt_rad))
        off = target.off(self.aim if self.aim is not None else view.head)
        band = s.deadband_deg if tracking else s.refine_deg
        if off > math.radians(band):
            speed: Speed = "slow" if off <= math.radians(s.saccade_over_deg) else "saccade"
            why = "the face moved" if tracking else "onto the face"
            if not self._hold(
                target, speed, now, why, px=round(face.width_px), lead_deg=_deg(lead)
            ):
                return
            self.counts["reaims"] += 1
        self.voice, self.seen = seen.pan_rad, (now, seen.pan_rad)
        self.target = Target(seen, face, now)
        small = face.width_px < s.track_min_face_px
        if not tracking:
            pan, tilt = _deg(seen.pan_rad), _deg(seen.tilt_rad)
            line = f"a face at {pan:+.0f}/{tilt:+.0f} deg, {face.width_px:.0f} px: tracking"
            if small and not self.smiled:
                line += " (small: no smile yet)"
            self._note(
                "gaze_face",
                now,
                line,
                pan_deg=pan,
                tilt_deg=tilt,
                px=round(face.width_px),
                score=round(face.score, 2),
                small=small,
                check=self.verifying,
            )
            self.phase, self.votes, self.fallback, self.verifying = "track", [], None, False
            match = math.radians(s.match_deg)  # the person found is held, no longer remembered
            self.targets = [
                t for t in self.targets if abs(_wrap(t.aim.pan_rad - seen.pan_rad)) > match
            ]
        if not small:
            self._smile()

    def _lead(self, pan: float, now: float) -> float:
        """How far a moving face will have gone by the next frame (radians of pan): constant
        velocity over the last two sightings, at most ``lead_max_deg``; 0 without a sighting in
        the last two and a half frame periods."""
        s = self.settings
        if self.seen is None:
            return 0.0
        then, before = self.seen
        dt = now - then
        if dt <= 0.0 or dt > 2.5 * s.track_every_s:
            return 0.0
        cap = math.radians(s.lead_max_deg)
        return max(-cap, min(cap, _wrap(pan - before) / dt * s.track_every_s))

    def _missed(self, now: float, *, looked: bool) -> None:
        """A frame without the speaker's face (``looked``: one was taken): a seek tilts higher,
        then runs out (a check back to the face it left); a tracked face unseen for
        ``face_lost_s`` is lost."""
        s = self.settings
        if self.phase == "seek":
            self.seek_left -= 1
            late = self.verify_until is not None and now >= self.verify_until
            if self.seek_left > 0 and not late:
                return
            if self.tilts_left and not late and self.aim is not None:
                higher = self.reach.clamp(Aim(self.aim.pan_rad, self.tilts_left.pop(0)))
                if self._hold(higher, "saccade", now, "seek higher"):
                    self.seek_left = 1 if self.verifying else s.seek_frames
                return
            self._unseen(now)
        elif self.phase == "track" and looked and self.target is not None:
            unseen = now - self.target.seen_s
            if unseen >= s.face_lost_s:
                self._remember(self.target)
                line = f"the face is lost ({unseen:.1f} s unseen): the head stays on its place"
                self._note("gaze_lost", now, line, what="face", unseen_s=round(unseen, 2))
                self.phase = "voice"
                self._forget_track()

    def _unseen(self, now: float) -> None:
        """A seek found nobody: a check goes back to the face it left; any other stays on the
        voice at the seek tilt."""
        s = self.settings
        fallback, verifying = self.fallback, self.verifying
        self._forget_track()
        if fallback is not None and verifying:
            pan = _deg(fallback.aim.pan_rad)
            line = f"nobody toward the voice: back to the face at {pan:+.0f} deg"
            self._note("gaze_lost", now, line, what="check")
            if self._hold(self.reach.clamp(fallback.aim), "saccade", now, "back to the face"):
                self.phase, self.voice = "track", fallback.aim.pan_rad
                self.target = replace(fallback, seen_s=now)  # it gets face_lost_s to show again
            return
        self.phase = "voice"
        self._note("gaze_lost", now, "no face: the head stays on the voice", what="seek")
        tilt = math.radians(s.seek_tilt_deg)
        if self.aim is not None and abs(self.aim.tilt_rad - tilt) > 1e-9:
            self._hold(self.reach.clamp(Aim(self.aim.pan_rad, tilt)), "saccade", now, "seek tilt")
        if s.smile_unseen:
            self._smile()

    def _forget_track(self) -> None:
        """A seek's, a check's and a track's working state cleared (the people remembered stay)."""
        self.fallback, self.verifying, self.verify_until = None, False, None
        self.tilts_left, self.votes, self.seen = [], [], None
        self.target = None

    def _remember(self, target: Target) -> None:
        """``target`` the latest person looked at; one remembered at its place is replaced, and
        only ``remember_targets`` are kept."""
        match = math.radians(self.settings.match_deg)
        others = [t for t in self.targets if abs(_wrap(t.aim.pan_rad - target.aim.pan_rad)) > match]
        self.targets = [target, *others][: self.settings.remember_targets]

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

    # ---- the record --------------------------------------------------------------------------
    def _judged(
        self,
        now: float,
        heard: Heard,
        voice: VoiceBearing | None,
        verdict: str,
        line: str | None = None,
    ) -> None:
        """An utterance's verdict: its line (when it says something) and its event."""
        fields = _voice_fields(voice, heard)
        if self.phase == "track" and self.voice is not None:
            fields["face_deg"] = _deg(self.voice)
        self._note("gaze_voice", now, line, verdict=verdict, phase=self.phase, **fields)

    def _note(self, kind: str, now: float, line: str | None, /, **fields: Any) -> None:
        """One decision: its line in the cascade's output, its event in the conversation's log."""
        if line is not None:
            self.report(line)
        sink = self.events
        if sink is None:
            return
        try:
            sink(kind, now, **fields)
        except Exception:  # a log that breaks never stops the gaze
            logger.exception("the speaker gaze's event sink broke")


def _voice_fields(voice: VoiceBearing | None, heard: Heard) -> dict[str, Any]:
    """An utterance's bearing and loudness as event fields."""
    fields: dict[str, Any] = {
        "text": heard.text[:TEXT_KEPT],
        "peak_db": heard.peak_db,
        "named": heard.named,
        "words": heard.words,
    }
    if voice is not None:
        fields.update(
            doa_deg=None if voice.doa_deg is None else round(voice.doa_deg, 1),
            bearing_deg=None if voice.bearing_rad is None else _deg(voice.bearing_rad),
            readings=voice.readings,
            speech=voice.speech,
        )
    return fields


def _turned(target: Target, turned: float) -> Target:
    """``target`` as seen from the cart once it turned ``turned`` radians (+ left)."""
    return replace(target, aim=Aim(_wrap(target.aim.pan_rad - turned), target.aim.tilt_rad))


def _mean(bearings: list[float]) -> float:
    """The circular mean of bearings in radians (the first when they cancel out)."""
    mean = circular_mean_deg([math.degrees(b) for b in bearings])
    return bearings[0] if mean is None else _wrap(math.radians(mean))


def _wrap(rad: float) -> float:
    return math.atan2(math.sin(rad), math.cos(rad))


def _deg(rad: float) -> float:
    """Degrees, one decimal, for an event."""
    return round(math.degrees(rad), 1)


class SpeakerGazeWorker:
    """The speaker gaze in the cascade: a :class:`pepin.expression.conversation.
    ConversationObserver` and the :class:`pepin.expression.conversation.LiveFlag`
    ``speaker_gaze``. Every moment is one non-blocking put on a queue; one daemon thread (started
    the first time the switch is on) runs the machine and sleeps on the queue between
    conversations, so the switch off and the robot alone cost nothing. ``journal`` (the loop's
    :meth:`pepin.voice_cascade.loop.CascadeLoop.record`) takes the machine's events."""

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
        self.journal: EventSink | None = None  # the conversation's log, once the loop is built
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

    def record(self, kind: str, now: float, /, **fields: Any) -> None:
        """The machine's event to the journal (nothing before one is set)."""
        journal = self.journal
        if journal is not None:
            journal(kind, now, **fields)

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
                if self.machine.events is None:
                    self.machine.events = self.record
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


def robot_gaze(
    host: str,
    doa: DoaSource,
    *,
    on: bool | None = None,
    finder: SnapshotFinder | None = None,
    on_face: FaceListener | None = None,
) -> SpeakerGazeWorker:
    """The speaker gaze on the robot: the arbiter's door on this Mac, the board's camera and
    head server at ``host``, the array's directions from ``doa`` (the cascade's audio link), the
    wheels' stub; built, and the finder's model loaded, the first time its switch is on (config/
    expression.json's ``flags.speaker_gaze`` at the start when ``on`` is None). ``finder``: one
    shared with person recognition (else its own); ``on_face`` hears the speaker's faces."""

    def build(settings: SpeakerSettings) -> SpeakerGaze:
        from pepin.deployment import config_file
        from pepin.expression.door import SpeakerDoor
        from pepin.expression.executors import NoWheels
        from pepin.expression.finder import board_finder
        from pepin.head_link import HeadClient
        from pepin.neck import NeckConfig

        looks = finder
        if looks is None:
            try:
                looks = board_finder(host)  # config reads only: the model loads on warm()
            except (OSError, ValueError, KeyError) as exc:
                say(f"no face finder ({exc}): the head turns to the voice only")
        return SpeakerGaze(
            settings,
            door=SpeakerDoor(aim_wait_s=settings.aim_wait_s),
            doa=doa,
            finder=looks,
            wheels=NoWheels(),
            face=HeadClient(host, source=FACE_SOURCE).start(),
            reach=Reach.of(NeckConfig.from_json(config_file("neck.json"))),
            on_face=on_face,
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
