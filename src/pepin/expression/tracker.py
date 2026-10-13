"""The speaker gaze: the head on the person the robot talks with, the head's base layer.

One engine, coarse to fine, after Pepper's basic awareness (a stimulus turns the head, a look
checks for a person there, a person found is engaged and tracked, none: back to the one engaged
before):

1. STIMULI. The array's direction stream (:mod:`pepin.expression.doa`: about 10 readings a
   second, speech-flagged, on the board's clock) with the robot's own speaking spans cut out, so
   an utterance that overlaps a reply keeps its clean part; the utterances the ears close (words,
   peak level, the robot's name); one head-camera frame at a time (:mod:`pepin.expression.
   finder`: faces, and people's boxes when a body detector runs); the robot speaking.
2. PEOPLE (:mod:`pepin.expression.tracks`): a few tracks, each a bearing and a head aim placed by
   a face, a body box or a voice; freshness decides between them.
3. WHOM (:mod:`pepin.expression.policy`): the person speaking. The name always wins; a person
   seen before is switched to at once; an unknown voice while someone is engaged is a check (one
   frame there, back when nobody); a sound, a quiet or a thin utterance moves nothing.
4. HOW: one ``angles`` aim, pan and tilt together, eased on the servo's own trapezoid
   (:class:`pepin.expression.speaker.Ease`): a TURN (``turn_acc_deg_s2``: 20 deg in 0.35 s, 60
   in 0.61 s) to a new person, an ADJUST (``adjust_acc_deg_s2``) for corrections, follows and the
   seek's stops. Tracking keeps a dead band and leads a moving face. A SEEK runs only when the
   person engaged has a bearing and nobody is seen there: the last ``doa_follow_s`` of the voice
   first, then ``seek_pan_deg`` outward and inward of the bearing at the remembered tilt, one
   frame a stop, then back to the voice. While the person speaks and nothing has seen them for
   ``seen_fresh_s``, the head FOLLOWS the voice's stream.
5. THE BODY: one rule. The head serves any bearing within the forward arc (``arc_deg``, +-80
   deg); the person engaged past it turns the body toward them (:mod:`pepin.expression.plan`, at
   most ``wheels.max_turn_deg`` a request, again while they are still past it), on the wheels'
   own thread, alongside the head's own turn, neither waiting for the other. Wheels that cannot
   (a refused check, a blocked arc, the caps) leave the head alone on the person, as far as it
   reaches.
6. THE FRAME: the people's bearings are kept in the cart's frame at the newest heading of the
   wheels' odometry (the base server's state lines, on the board's clock, whoever turned the
   cart). A turn since the bearings' stamp is taken off every person, off each of the array's
   readings by its own stamp (an utterance spoken across a turn gives the bearing the person has
   now), and the head re-centres on the person as the body comes round under it.

The phases: ``seek`` (looking for the person engaged: a check, or the stops), ``track`` (they
are seen), ``voice`` (nobody seen: the head on their voice, frames go on), ``paused`` (a drive)
and ``idle`` (no conversation, or the switch off: no frames, no calls). The person engaged is
lost after ``face_lost_frames`` frames AND ``face_lost_s`` without them (one missed frame at the
1 Hz of the robot's speech is no loss); the head then goes to their voice when it was heard
after they were last seen.

The voice comes first: from a reply's first part going to the voice until its first audio chunk
is with the player (``hold_for_voice``) no frame is taken, and while the robot speaks frames come
at ``track_speaking_hz``; the detector runs on the CPU and the Neural Engine, never the GPU the
voice uses (config/expression.json ``finder``). The head is held through the arbiter's door as
an ``angles`` look of the idle band, source ``voice.speaker`` (:mod:`pepin.expression.head`):
TTL ``ttl_s``, renewed every ``renew_s`` (an aim renews it too); a renewal that finds the look
gone (a ``look`` the person asked for took the head) asks again, queued behind it. A frame counts
only while the head is ours, still and carries no gesture (the idle layer's breath aside). The
smile goes through the head server's face owner (source ``expression``), once a conversation, at
the first clear face.

Every decision prints one line (``say``) and, through :class:`pepin.expression.conversation.
EventSink`, lands in the conversation's events.jsonl on the session clock: ``gaze_wake``,
``gaze_voice`` (each utterance's bearing, verdict and track), ``gaze_doa`` (each utterance's
readings: ``[seconds before its end on the board's clock, the array's angle, 1 speech / 0 not /
-1 the robot's own voice]``, and a fourth number when the cart has turned since: how far, in
degrees, taken off that reading's bearing),
``gaze_reaim`` (each aim asked of the arbiter, why, its profile and its time), ``gaze_face`` (a
person found: by face or body), ``gaze_lost`` (a seek, a check or a person that came to nothing;
the door lost), ``gaze_wheels`` and ``gaze_release``.

:class:`SpeakerGaze` is the machine, single-threaded and clocked by its caller (tests drive it on
a fake clock); :class:`SpeakerGazeWorker` is the cascade's observer and the live switch
``speaker_gaze``: one daemon thread that sleeps on its queue between conversations. Every
observer hook is one queue put: the gaze never delays the ears, the brain or the voice.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import queue
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pepin.expression.conversation import SPEAKER_GAZE, EventSink, Heard
from pepin.expression.doa import (
    Span,
    Turned,
    UtteranceDoa,
    VoiceBearing,
    board_offset,
    fit_doa_map,
    recent_voice,
    utterance_voice,
)
from pepin.expression.executors import FaceChannel, Headings, WheelMove, WheelReport, Wheels
from pepin.expression.head import IDLE_LOOK, SPEAKER
from pepin.expression.markers import Part
from pepin.expression.plan import BodyLimits, split, sweep
from pepin.expression.policy import decide, doubt
from pepin.expression.speaker import (
    DoaMap,
    DoaSource,
    Ease,
    FaceFinder,
    FaceFrame,
    FaceSighting,
    GazeDoor,
    HeadView,
)
from pepin.expression.tracks import Sighting, Track, Tracks, sightings, wrap
from pepin.gaze import Aim, Reach, Speed

if TYPE_CHECKING:
    from pepin.expression.conversation import ConversationObserver, LiveFlag
    from pepin.expression.door import SpeakerDoor
    from pepin.expression.finder import ServiceFinder, SnapshotFinder, SwitchedFinder
    from pepin.expression.wheels import WheelsExecutor
    from pepin.voice_live.sources import BoardLink

logger = logging.getLogger(__name__)

Phase = Literal["idle", "seek", "voice", "track", "paused"]
Move = Literal["turn", "adjust"]
CONFIG_FILE = "expression.json"
FACE_SOURCE = "expression"  # the face's expressions of the body layer, at the head server
BLIND_RETRY_S = 0.1  # a frame refused for a blind head is tried again this soon
TICK_EPS_S = 1e-6
FOLLOW_EVERY_S = 0.1  # the voice's stream is read this often while the head follows it
LISTEN_MAX_S = 30.0  # an utterance never closed is not followed longer than this
SPOKE_KEPT = 8  # the robot's latest speaking spans kept to judge an utterance by
TEXT_KEPT = 40  # an utterance's words kept in its event, to pair it with the loop's "heard"
TURN_EPS_RAD = 1e-4  # a heading change under this is no turn
TURNED_LOGGED_DEG = 0.05  # a reading turned less than this since its stamp logs no turn
Report = Callable[[str], None]
FaceListener = Callable[[FaceFrame, FaceSighting], None]  # a frame and the speaker's face in it
WheelEvent = tuple[str, int, Any]  # ("moved", turn id, degrees) or ("done", turn id, WheelReport)


@dataclass
class Turning:
    """A turn of the wheels toward the speaker, running: its id, the person it turns toward, how
    far it was asked and how far the wheels have turned so far (degrees, + left)."""

    id: int
    track: int
    asked_deg: float
    turned_deg: float = 0.0


def say(line: str) -> None:
    """One line of the speaker gaze in the cascade's output."""
    print(f"  (gaze: {line})", flush=True)


@dataclass(frozen=True)
class SpeakerSettings:
    """config/expression.json's ``speaker`` block (and the ``wheels`` block's limits): the
    file's notes say what each is; ``doa_map`` is None while the array is unmeasured."""

    doa_map: DoaMap | None = None
    window_s: float = 1.0
    min_speech: int = 2
    arc_deg: float = 160.0
    seek_tilt_deg: float = 3.0
    seek_pan_deg: float = 20.0
    seek_frames: int = 1
    seek_every_s: float = 0.4
    seek_min_face_px: float = 24.0  # the rectified eye's pixels (fx 454.5): 28 in the raw eye's
    track_min_face_px: float = 34.0  # the rectified eye's pixels: 40 in the raw eye's
    min_body_score: float = 0.5
    track_hz: float = 10.0
    track_speaking_hz: float = 1.0
    hold_for_voice: bool = True
    lead_s: float = 0.3
    lead_max_deg: float = 10.0
    match_deg: float = 20.0
    voice_match_deg: float = 25.0
    tilt_hysteresis_deg: float = 12.0
    refine_deg: float = 2.0
    deadband_deg: float = 6.0
    saccade_over_deg: float = 15.0
    aim_min_deg: float = 1.0
    verify_s: float = 0.7
    face_lost_s: float = 2.0
    face_lost_frames: int = 3
    doa_min_speech: int = 6
    doa_min_dbfs: float = -18.0
    doa_follow_s: float = 0.5
    doa_moved_deg: float = 8.0
    follow_min_speech: int = 3
    follow_max_deg: float = 35.0
    seen_fresh_s: float = 1.0
    max_tracks: int = 4
    track_forget_s: float = 300.0
    turn_speed_deg_s: float = 200.0
    turn_acc_deg_s2: float = 650.0
    adjust_acc_deg_s2: float = 400.0
    ttl_s: float = 6.0
    renew_s: float = 2.0
    aim_wait_s: float = 2.0
    smile: str = "smile"
    smile_hold_s: float = 2.0
    smile_unseen: bool = False
    body: BodyLimits = field(default_factory=BodyLimits)

    @property
    def track_every_s(self) -> float:
        """The time between tracking frames while the robot listens."""
        return 1.0 / self.track_hz

    @property
    def speaking_every_s(self) -> float:
        """The time between tracking frames while the robot speaks."""
        return 1.0 / self.track_speaking_hz

    def ease(self, move: Move) -> Ease | None:
        """The profile of a turn or an adjustment; None when its ramp is 0 (the arbiter's own
        saccade or slow speed)."""
        acc = self.turn_acc_deg_s2 if move == "turn" else self.adjust_acc_deg_s2
        return Ease(self.turn_speed_deg_s, acc, move) if acc > 0.0 else None

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
        body_block = data.get("wheels", {})
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
        if self.max_tracks < 1 or self.face_lost_frames < 1 or self.verify_s < 0:
            raise ValueError("speaker: max_tracks and face_lost_frames >= 1, verify_s >= 0")
        if self.lead_max_deg < 0 or self.face_lost_s < 0 or self.seek_pan_deg < 0:
            raise ValueError("speaker: lead_max_deg, face_lost_s and seek_pan_deg >= 0")
        if self.track_speaking_hz <= 0 or self.lead_s < 0 or self.aim_min_deg < 0:
            raise ValueError("speaker: track_speaking_hz > 0, lead_s and aim_min_deg >= 0")
        if self.turn_speed_deg_s <= 0 or self.turn_acc_deg_s2 < 0 or self.adjust_acc_deg_s2 < 0:
            raise ValueError("speaker: turn_speed_deg_s > 0, turn_ and adjust_acc_deg_s2 >= 0")
        if self.doa_follow_s <= 0 or self.follow_min_speech < 1 or self.doa_moved_deg < 0:
            raise ValueError("speaker: doa_follow_s > 0, follow_min_speech >= 1, doa_moved >= 0")
        if not 0.0 < self.arc_deg <= 360.0 or self.tilt_hysteresis_deg < 0:
            raise ValueError("speaker: arc_deg above 0 and at most 360, tilt_hysteresis_deg >= 0")

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
    """The speaker gaze's engine: every moment and every tick from one thread, each call to the
    door, the finder, the wheels and the face made from it, ``now`` the laptop's clock."""

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
        """The seams it drives; ``doa`` None or ``finder`` None: no voice or no eyes;
        ``on_face`` hears every frame's face of the person engaged (person recognition);
        ``events`` takes every decision as an event of the conversation's log."""
        self.door, self.doa, self.finder = door, doa, finder
        self.wheels, self.face, self.reach = wheels, face, reach
        self.report = report
        self.on_face = on_face
        self.events = events
        self.settings = settings
        self.tracks = self._tracks_of(settings)
        self.phase: Phase = "idle"
        self.conversation = False
        self.engaged: int | None = None  # the track the head serves
        self.engaged_s = 0.0  # when it was engaged (or returned to)
        self.fallback: int | None = None  # the track a check returns to
        self.centre: float | None = None  # the bearing the look (and a seek's stops) centre on
        self.aim: Aim | None = None  # the aim last asked of the arbiter
        self.stops: list[Aim] | None = None  # a seek's stops still to look at (None: not begun)
        self.seek_left = 0  # frames left at this stop of a seek
        self.verifying = False  # the seek is a check that returns to ``fallback``
        self.verify_until: float | None = None  # the check's end, set at its first frame
        self.misses = 0  # frames in a row without the person engaged
        self.listening: float | None = None  # the person's utterance began (laptop clock)
        self.next_frame: float | None = None
        self.next_renew: float | None = None
        self.next_follow: float | None = None
        self.smiled = False
        self.reorient = False  # the next utterance turns the head (after a drive, a switch)
        self.voice_hold = False  # a reply's first chunk is being synthesised: no frames
        self.spoke: list[Span] = []  # the robot's speaking spans
        self.frame_s: float | None = None  # the board stamp the people's bearings are kept at
        self.drift = 0.0  # radians the cart turned under the head since its last aim
        self.ours = False  # the head holds this source's aim (the arbiter said so last)
        self.body_waits = False  # a turn fell short: none until the person's next utterance
        self.gave_up: set[int] = set()  # people a turn that moved fell short for: no more
        self.wake: Heard | None = None  # the cascade tells it once more as heard: not twice
        self.counts: Counter[str] = Counter()
        self.turning: Turning | None = None  # the wheels' turn toward the person, running
        self._turn_ids = itertools.count(1)
        # The wheels answer on their own thread: their word is posted here and taken on this
        # machine's (the worker replaces ``post`` with a put on its queue; alone, ``tick``
        # takes the inbox).
        self.inbox: deque[WheelEvent] = deque()
        self.post: Callable[[WheelEvent], None] = self.inbox.append

    def configure(self, settings: SpeakerSettings) -> None:
        """New knobs (the switch turned on re-reads them); the people known stay."""
        self.settings = settings
        known = self.tracks
        self.tracks = self._tracks_of(settings)
        self.tracks.adopt(known)

    @staticmethod
    def _tracks_of(s: SpeakerSettings) -> Tracks:
        return Tracks(
            default_tilt_rad=math.radians(s.seek_tilt_deg),
            match_deg=s.match_deg,
            voice_match_deg=s.voice_match_deg,
            seen_fresh_s=s.seen_fresh_s,
            forget_s=s.track_forget_s,
            max_tracks=s.max_tracks,
            min_new_px=s.track_min_face_px,
            tilt_hold_deg=s.tilt_hysteresis_deg,
        )

    @property
    def track(self) -> Track | None:
        """The person engaged now."""
        return self.tracks.get(self.engaged)

    @property
    def voice(self) -> float | None:
        """The bearing of the person engaged (radians, the cart's frame), None for nobody."""
        track = self.track
        return None if track is None else track.pan

    # ---- the moments ---------------------------------------------------------------------------
    def opened(self, wake: Heard, now: float) -> None:
        """A conversation opened on ``wake``: engage its voice (a remembered person there keeps
        their tilt) and seek them."""
        self.conversation, self.smiled, self.reorient = True, False, False
        self.wake, self.listening = wake, None
        self.body_waits, self.gave_up = False, set()
        self._sync_body(now)
        self.tracks.forget(now)
        voice = self._utterance(wake, now)
        self._note("gaze_wake", now, f"wake: {voice.text()}", **_voice_fields(voice, wake))
        bearing = voice.bearing_rad
        if bearing is None:  # the map unmeasured, no reading: where the head points
            view = self._state()
            bearing = view.head.pan_rad if view is not None and view.head is not None else 0.0
        track, _ = self.tracks.heard(bearing, now, self._kept())
        self.engaged = None
        self._engage(track, now, "wake")

    def heard(self, heard: Heard, now: float) -> None:
        """An utterance of the conversation, judged as a stimulus, then by its track."""
        self.listening = None
        if not self.conversation or heard == self.wake:
            return
        if self.phase == "paused" and not self.reorient:
            return  # a drive runs
        self.body_waits = False  # each utterance may turn the body again
        self._sync_body(now)
        s = self.settings
        voice = self._utterance(heard, now)
        if voice.robot and voice.speech < s.doa_min_speech:
            self.counts["over_robot"] += 1
            self._judged(now, heard, voice, "over the robot's own voice or its turn")
            return
        bearing = voice.bearing_rad
        if bearing is None:
            self._judged(now, heard, voice, "no bearing")
            return
        if self.reorient:
            if not heard.words:
                self._judged(now, heard, voice, "no words")
                return
            self.reorient = False
            track, _ = self.tracks.heard(bearing, now, self._kept())
            self._judged(now, heard, voice, "after a pause", f"after a pause: {voice.text()}",
                         track=track.id)  # fmt: skip
            self._engage(track, now, "after a pause")
            return
        named = heard.named and heard.words
        if not named:
            why = doubt(heard, voice, min_speech=s.doa_min_speech, min_dbfs=s.doa_min_dbfs)
            if why is not None:
                self.counts["sounds_ignored"] += 1
                line = None if not heard.words else f"not moved by {voice.text()}: {why}"
                self._judged(now, heard, voice, why, line)
                return
        engaged = self.tracks.get(self.fallback) if self.verifying else self.track
        centre = self.centre
        speaker, new = self.tracks.heard(bearing, now, self._kept())
        decision = decide(engaged, speaker, named=named)
        if decision.act == "stay":
            self._judged(now, heard, voice, decision.why, track=speaker.id)
            moved = centre is not None and abs(wrap(speaker.pan - centre)) > math.radians(
                s.doa_moved_deg
            )
            if self.phase in ("seek", "voice") and not self.verifying and moved:
                self._engage(speaker, now, "the voice moved", move="adjust")
            return
        line = f"{decision.why}: {voice.text()}"
        self._judged(now, heard, voice, decision.why, line, track=speaker.id, new=new)
        self._engage(speaker, now, decision.why, check=decision.act == "check")

    def speech_started(self, t: float, now: float) -> None:
        """The person began an utterance at ``t``: while nobody sees them, the head follows
        their voice."""
        if self.conversation:
            self.listening = t
            self.next_follow = now

    def speaking(self, on: bool, now: float) -> None:
        """The robot's voice starts (a reply's part queued: frames wait for its first chunk,
        then come at the speaking rate) or stops (a reply done or cut: the listening rate at
        once)."""
        open_span = bool(self.spoke) and self.spoke[-1][1] is None
        if on and not open_span:
            self.spoke = [*self.spoke[-(SPOKE_KEPT - 1) :], (now, None)]
            if self.settings.hold_for_voice and self.phase != "idle":
                self.voice_hold = True
                self.counts["voice_holds"] += 1
        elif not on and open_span:
            self.spoke[-1] = (self.spoke[-1][0], now)
            self.voice_hold = False
            soon = now + self.settings.track_every_s
            if self.next_frame is not None and self.next_frame > soon:
                self.next_frame = soon

    def voiced(self, now: float) -> None:
        """The reply's first chunk is with the player: frames resume."""
        self.voice_hold = False

    def drive(self, active: bool, now: float) -> None:
        """A drive started (the arbiter let the head go, the wheels are the drive's; every
        bearing goes stale) or ended (the next utterance orients)."""
        if active:
            self._forget_look()
            self._let_wheels_go("a drive starts")
            self.tracks.clear()
            self.gave_up = set()
            self.engaged, self.ours = None, False
            self.phase, self.aim, self.next_frame, self.next_renew = "paused", None, None, None
            self.reorient = False
        elif self.conversation:
            self.reorient = True

    def closed(self, reason: str, now: float) -> None:
        """The conversation is over: the head is let go and nothing runs until the next one;
        the people stay known (their places for the tilt, until forgotten)."""
        self.conversation = False
        self.stop(now, f"the conversation closed ({reason})", forget=False)

    def stop(self, now: float, why: str, *, forget: bool = True) -> None:
        """Let the head go now (the switch off, the conversation's end), and the wheels."""
        self._let_wheels_go(why)
        held = self.phase != "idle" or self.aim is not None
        self.voice_hold, self.ours = False, False
        self.phase, self.aim, self.next_frame, self.next_renew = "idle", None, None, None
        self.engaged, self.listening, self.next_follow = None, None, None
        self._forget_look()
        if forget:
            self.tracks.clear()
        if held:
            self.door.release()
            self.counts["releases"] += 1
            self._note("gaze_release", now, f"released: {why}", why=why)

    def resume(self, now: float) -> None:
        """The switch turned on mid-conversation: the next utterance orients the head."""
        if self.conversation and self.phase == "idle":
            self.phase, self.reorient = "paused", True

    def warm(self) -> None:
        """Load what the first wake would wait for (the finder's models); a finder that cannot
        load (no weights) is dropped: the head then stays on the voice."""
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
        """Do what is due (the wheels' word, the body's heading, a renewal, the voice followed,
        the body's turn, a frame, a lost head asked for again); when next, None for nothing
        until a moment. A frame waits while the body turns; the heading is read every
        ``FOLLOW_EVERY_S`` meanwhile."""
        while self.inbox:
            self.wheels_event(self.inbox.popleft(), now)
        if self.phase in ("idle", "paused"):
            return None
        self._sync_body(now)
        now_ = now + TICK_EPS_S  # a moment due a hair later is due now (clocks add in floats)
        if self.aim is None:  # the door did not answer, or refused: ask again in a while
            if self.next_renew is None or now_ >= self.next_renew:
                track = self.track
                if track is not None:
                    self._engage(track, now, "asked again", check=self.verifying)
                else:
                    self.next_renew = now + self.settings.renew_s
            return self.next_renew
        if self.next_renew is not None and now_ >= self.next_renew:
            self._renew(now)
        following = self._following(now)
        if following and self.next_follow is not None and now_ >= self.next_follow:
            self.next_follow = now + FOLLOW_EVERY_S
            self._follow(now)
        self._arc(free=self.ours)
        waiting = self.voice_hold or self.turning is not None  # the voice's first chunk, the body
        if self.next_frame is not None and now_ >= self.next_frame and not waiting:
            self._frame(now)
        frame = None if waiting else self.next_frame  # the voice's moment, the wheels' end
        follow = self.next_follow if self._following(now) else None
        body = now + FOLLOW_EVERY_S if self.turning is not None else None  # its heading
        due = [t for t in (self.next_renew, frame, follow, body) if t is not None]
        return min(due) if due else None

    # ---- the voice -----------------------------------------------------------------------------
    def _utterance(self, heard: Heard, now: float) -> VoiceBearing:
        """``heard``'s bearing from its clean readings; its readings logged (``gaze_doa``)."""
        if self.doa is None:
            return VoiceBearing(None, None, 0, 0)
        s = self.settings
        turned = self._turned_since()
        found = utterance_voice(self.doa, heard, now, window_s=s.window_s, doa_map=s.doa_map,
                                min_speech=s.min_speech, spans=self.spoke,
                                turned=turned)  # fmt: skip
        self._log_readings(heard, found, now, turned)
        return found.voice

    def _log_readings(
        self, heard: Heard, found: UtteranceDoa, now: float, turned: Turned | None
    ) -> None:
        offset = board_offset(self.doa, now) if self.doa is not None else None
        if not found.readings or offset is None:
            return
        end = heard.end_s + offset
        rows: list[list[float | int]] = [
            [round(r.stamp_s - end, 2), r.deg, -1 if mine else int(r.speech)]
            for r, mine in zip(found.readings, found.robot, strict=True)
        ]
        if turned is not None:
            degs = [math.degrees(turned(r.stamp_s)) for r in found.readings]
            if any(abs(d) >= TURNED_LOGGED_DEG for d in degs):
                rows = [[*row, round(d, 1)] for row, d in zip(rows, degs, strict=True)]
        self._note("gaze_doa", now, None, text=heard.text[:TEXT_KEPT], readings=rows)

    def _following(self, now: float) -> bool:
        """Whether the head follows the voice's stream now: the person speaks, nobody has seen
        them lately, no check (the body may be turning: its readings are turned with it)."""
        track = self.track
        if self.listening is None or track is None or self.aim is None:
            return False
        if now - self.listening > LISTEN_MAX_S:
            self.listening = None
            return False
        return (
            self.phase in ("seek", "voice")
            and not self.verifying
            and not track.fresh(now, self.settings.seen_fresh_s)
        )

    def _follow(self, now: float) -> bool:
        """The voice's last ``doa_follow_s``: moved more than ``doa_moved_deg`` (and at most
        ``follow_max_deg``: a jump is a new voice, judged when its utterance ends) from the
        person engaged, the head goes there (a seek begins again there); whether it moved."""
        track = self.track
        if self.doa is None or track is None or self.aim is None:
            return False
        s = self.settings
        voice = recent_voice(self.doa, now, window_s=s.doa_follow_s, doa_map=s.doa_map,
                             min_speech=s.follow_min_speech, spans=self.spoke,
                             turned=self._turned_since())  # fmt: skip
        bearing = voice.bearing_rad
        if bearing is None:
            return False
        moved = abs(wrap(bearing - track.pan))
        if moved <= math.radians(s.doa_moved_deg) or moved > math.radians(s.follow_max_deg):
            return False
        self.tracks.followed(track, bearing, now)
        self.counts["follows"] += 1
        self.centre = track.pan
        target = self.reach.clamp(Aim(track.pan, track.head.tilt_rad))
        if not self._hold(target, now, "the voice moved", "adjust", track=track.id):
            return True
        if self.phase == "seek":
            self.stops, self.seek_left = None, max(1, s.seek_frames)
            self.next_frame = now + BLIND_RETRY_S
        return True

    def _robot_speaking(self) -> bool:
        """Whether a span of the robot's speech is open."""
        return bool(self.spoke) and self.spoke[-1][1] is None

    # ---- engaging ------------------------------------------------------------------------------
    def _kept(self) -> set[int]:
        return {t for t in (self.engaged, self.fallback) if t is not None}

    def _engage(
        self, track: Track, now: float, why: str, *, check: bool = False, move: Move = "turn"
    ) -> None:
        """Serve ``track``: one aim at its head (the body turned toward them past the forward
        arc, started first and never waited for), then frames: a check (``check``: one frame
        within ``verify_s``, back to the person engaged before when nobody is there) or a
        seek."""
        s = self.settings
        before = self.tracks.get(self.fallback) if self.verifying else self.track
        if check and before is not None and before.id != track.id:
            self.fallback = before.id
        elif not check:
            self.fallback = None
        self.engaged, self.engaged_s, self.misses = track.id, now, 0
        self.centre = track.pan
        self.stops, self.verify_until = None, None
        self.verifying = check and self.fallback is not None
        view = self._state()
        if view is None:
            self._lost(now, "the arbiter's door does not answer")
            return
        if view.driving:
            self.phase, self.reorient, self.aim = "paused", True, None
            return
        if self.turning is not None and self.turning.track != track.id:
            self._let_wheels_go("another person is engaged")
        # The wheels first: their own checks take longer than the head's turn takes.
        self._arc(free=view.holder in ("", SPEAKER))
        aim = self.reach.clamp(track.head)
        if not self._hold(aim, now, why, move, start=view.head, track=track.id,
                          source=track.source):  # fmt: skip
            return
        self.counts["orients"] += 1
        if self.finder is None:
            self.phase, self.next_frame, self.fallback, self.verifying = "voice", None, None, False
            return
        self.phase, self.next_frame = "seek", now
        self.seek_left = 1 if self.verifying else s.seek_frames
        if self.seek_left == 0:
            self._unseen(now)

    # ---- the body ----------------------------------------------------------------------------
    def _arc(self, *, free: bool) -> None:
        """THE BODY'S ONE RULE: the head serves any bearing within the forward arc; the person
        engaged past it turns the body toward them (``free``: the head is this source's, or
        nobody's). Not during a check, nor while a turn runs, nor after one fell short: until
        the person's next utterance, or for good (this conversation) when it moved the cart and
        still fell short (stalled, blocked, the back-offs spent)."""
        track = self.track
        if (
            track is None
            or not free
            or self.verifying
            or self.turning is not None
            or self.body_waits
            or track.id in self.gave_up
        ):
            return
        deg = self._body_share(track.pan)
        if deg != 0.0:
            self._start_wheels(deg, track.id)

    def _body_share(self, bearing: float) -> float:
        """How far the wheels turn toward ``bearing`` (degrees, + left): none within the
        forward arc; past it toward the person, at most a request's worth."""
        s = self.settings
        plan = split(
            bearing,
            0.0,
            half_arc_rad=math.radians(s.arc_deg / 2.0),
            max_turn_rad=math.radians(s.body.max_turn_deg),
        )
        return math.degrees(plan.body_rad)

    def _start_wheels(self, deg: float, track: int) -> None:
        """The wheels' turn of ``deg`` toward ``track``, started on their thread: this returns
        at once, their word comes back through :attr:`post`."""
        turn = Turning(next(self._turn_ids), track, deg)
        self.turning = turn
        self.counts["wheel_turns"] += 1

        def done(report: WheelReport) -> None:
            self.post(("done", turn.id, report))

        def moved(turned_deg: float) -> None:
            self.post(("moved", turn.id, turned_deg))

        self.wheels.start(WheelMove(turn_deg=deg), done, moved)

    def _let_wheels_go(self, why: str) -> None:
        """A turn of ours stops (a drive, the close, another person, the head not ours)."""
        if self.turning is not None:
            self.turning = None
            self.wheels.halt(why)

    def wheels_event(self, event: WheelEvent, now: float) -> None:
        """The wheels' word on a turn of ours: its progress and its end bring the people round
        with the cart (the heading itself is the odometry's, read in :meth:`_sync_body`); an end
        short of the turn says why, and the body waits (the module's notes)."""
        kind, turn_id, payload = event
        turn = self.turning
        if turn is None or turn.id != turn_id:
            self._sync_body(now)  # a turn replaced or let go: the frame follows all the same
            return
        if kind == "moved":
            turn.turned_deg = float(payload)
            self._sync_body(now)
            return
        report: WheelReport = payload
        self.turning = None
        turn.turned_deg = report.turned_deg
        self._sync_body(now)
        line = None
        if not report.done:
            moved = abs(report.turned_deg) >= 1.0 or report.backed_m > 0.0
            if report.cut or not moved:
                self.body_waits = True  # asked again at the person's next utterance
            else:
                self.gave_up.add(turn.track)
            line = (
                f"the wheels turned {report.turned_deg:+.0f} of {turn.asked_deg:+.0f} deg"
                f" ({report.why}): the head looks as far as it reaches"
            )
        self._note("gaze_wheels", now, line, asked_deg=round(turn.asked_deg, 1),
                   turned_deg=round(report.turned_deg, 1), done=report.done, why=report.why,
                   backed_m=round(report.backed_m, 3), attempts=report.attempts)  # fmt: skip

    # ---- the frame ---------------------------------------------------------------------------
    def _board_now(self, headings: Headings) -> float | None:
        """The board's clock now: the newest of the array's reading and the odometry's line."""
        stamps = []
        latest = self.doa.latest_doa() if self.doa is not None else None
        if latest is not None:
            stamps.append(latest.stamp_s)
        line = headings.latest()
        if line is not None:
            stamps.append(line[0])
        return max(stamps) if stamps else None

    def _sync_body(self, now: float) -> None:
        """THE FRAME: the cart's turn since the people's bearings were stamped (the odometry's
        heading, whoever turned it) is taken off every person, the bearing the look centres on
        and a seek's stops still to come; the head then re-centres under it."""
        headings = self.wheels.headings()
        board = self._board_now(headings)
        if board is None:
            return
        frame = self.frame_s
        self.frame_s = board if frame is None else max(board, frame)
        if frame is None:
            return
        then, latest = headings.at(frame), headings.at(self.frame_s)
        if then is None or latest is None:
            return
        d = math.remainder(latest - then, math.tau)
        if abs(d) < TURN_EPS_RAD:
            return
        self.tracks.turned(d)
        self.drift += d
        if self.centre is not None:
            self.centre = wrap(self.centre - d)
        if self.stops:
            self.stops = [Aim(stop.pan_rad - d, stop.tilt_rad) for stop in self.stops]
        self._recentre(now)

    def _turned_since(self) -> Turned | None:
        """How far the cart has turned since a board stamp, up to the people's frame (None
        while the odometry has said nothing: the cart has not turned)."""
        headings = self.wheels.headings()
        frame = headings.at(self.frame_s) if self.frame_s is not None else None
        if frame is None:
            return None

        def turned(stamp_s: float) -> float:
            then = headings.at(stamp_s)
            return 0.0 if then is None else math.remainder(frame - then, math.tau)

        return turned

    def _recentre(self, now: float) -> None:
        """The cart turned under the head: the head keeps its place in the room, on the person
        engaged (a seek's stop: where it was), an adjustment when it is small, as far as it
        reaches."""
        track = self.track
        if track is None or self.aim is None or self.phase in ("idle", "paused"):
            return
        if self.phase == "seek" and self.stops is not None:  # a stop of a sweep, held in the room
            target = self.reach.clamp(Aim(self.aim.pan_rad - self.drift, self.aim.tilt_rad))
        else:
            target = self.reach.clamp(Aim(track.head.pan_rad, self.aim.tilt_rad))
        off = target.off(self.aim)
        if off <= math.radians(self.settings.refine_deg):
            return
        move: Move = "adjust" if off <= math.radians(self.settings.saccade_over_deg) else "turn"
        turned = None if self.turning is None else round(self.turning.turned_deg, 1)
        if self._hold(target, now, "the body turned", move, turned_deg=turned):
            self.counts["recentres"] += 1

    # ---- the head ----------------------------------------------------------------------------
    def _hold(
        self, aim: Aim, now: float, why: str, move: Move, *, start: Aim | None = None, **fields: Any
    ) -> bool:
        """Ask the arbiter to hold the head at ``aim`` (replacing this source's last aim), a
        turn or an adjustment (from ``start`` when no aim of ours holds the head yet); whether
        the request lives (done, or queued behind a better look)."""
        s = self.settings
        ease = s.ease(move)
        speed: Speed = "saccade" if move == "turn" else "slow"
        before = self.aim if self.aim is not None else start
        answer = self.door.aim(aim, speed=speed, ttl_s=s.ttl_s, ease=ease)
        self.counts["aims"] += 1
        self.next_renew = now + s.renew_s
        if ease is not None and before is not None:
            fields["move_s"] = round(ease.seconds(math.degrees(aim.off(before))), 2)
        self._note(
            "gaze_reaim",
            now,
            None,
            why=why,
            pan_deg=_deg(aim.pan_rad),
            tilt_deg=_deg(aim.tilt_rad),
            speed=speed if ease is None else ease.name,
            status=answer.status,
            **fields,
        )
        if answer.status in ("done", "queued"):
            self.aim, self.drift = aim, 0.0
            self.ours = answer.status == "done"
            if not self.ours:
                self.report(f"the head waits: {answer.status} {answer.reason}".rstrip())
                self._let_wheels_go("another look holds the head")
            return True
        self._lost(now, f"{answer.status}: {answer.reason}")
        return False

    def _lost(self, now: float, why: str) -> None:
        """The head is not ours: stay on the voice, ask again after ``renew_s``."""
        line = f"no head ({why}); asking again in {self.settings.renew_s:.0f} s"
        self._note("gaze_lost", now, line, what="head", why=why)
        self.ours = False
        self._let_wheels_go("the head is not this source's")
        self._forget_look()
        self.phase, self.aim, self.next_frame = "voice", None, None
        self.next_renew = now + self.settings.renew_s

    def _renew(self, now: float) -> None:
        """Keep the look alive; one another look took is asked for again (queued behind it)."""
        self.next_renew = now + self.settings.renew_s
        live = self.door.renew()
        self.counts["renewals"] += 1
        if live == 0 and self.aim is not None:
            self.counts["reposts"] += 1
            self._hold(self.aim, now, "asked again", "turn")

    # ---- the frames --------------------------------------------------------------------------
    def _frame(self, now: float) -> None:
        """One frame while the head is ours, still and carries no gesture; the person engaged
        centred (refined after a turn, re-aimed past the dead band while tracking)."""
        try:
            self._look(now)
        finally:  # at the rate of the phase the frame left the machine in
            if self.next_frame is not None and self.next_frame <= now + TICK_EPS_S:
                s = self.settings
                if self.phase != "seek":
                    self.next_frame = now + (
                        s.speaking_every_s if self._robot_speaking() else s.track_every_s
                    )
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
        self.ours = view.holder == SPEAKER
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
        # A finder that reads frames already taken (the perception service's) waits for one
        # exposed after the head settled: the arbiter's blind_until, both on the board's clock.
        frame = self.finder.look(not_before=view.blind_until)
        self.counts["frames"] += 1
        people = sightings(frame, view.head, min_face_px=s.seek_min_face_px,
                           min_body_score=s.min_body_score)  # fmt: skip
        track = self.track
        seeking = track if self.phase in ("seek", "voice") else None
        seek_rad = math.radians(s.seek_pan_deg + s.match_deg)
        seen = self.tracks.seen(people, now, seeking=seeking, seek_rad=seek_rad, keep=self._kept())
        mine = seen.get(track.id) if track is not None else None
        if track is None or mine is None:
            self._missed(now, looked=True)
            return
        self._seen(track, mine, frame, now)

    def _seen(self, track: Track, mine: Sighting, frame: FaceFrame | None, now: float) -> None:
        """The person engaged is in the frame: centred, tracked, smiled at once."""
        s = self.settings
        self.misses = 0
        self.counts["faces" if mine.face is not None else "bodies"] += 1
        if self.on_face is not None and frame is not None and mine.face is not None:
            try:
                self.on_face(frame, mine.face)
            except Exception:  # a listener never stops the gaze
                logger.exception("the speaker's face listener broke")
        tracking = self.phase == "track"
        lead = track.lead(s.lead_s, math.radians(s.lead_max_deg)) if tracking else 0.0
        target = self.reach.clamp(Aim(track.head.pan_rad + lead, track.head.tilt_rad))
        off = target.off(self.aim) if self.aim is not None else math.inf
        band = max(s.deadband_deg if tracking else s.refine_deg, s.aim_min_deg)
        px = round(mine.face.width_px) if mine.face is not None else None
        if off > math.radians(band):
            move: Move = "adjust" if off <= math.radians(s.saccade_over_deg) else "turn"
            if tracking:
                why = "the person moved"
            else:
                why = "onto the face" if mine.face is not None else "onto the person"
            if not self._hold(target, now, why, move, px=px, lead_deg=_deg(lead),
                              source=mine.source, track=track.id):  # fmt: skip
                return
            self.counts["reaims"] += 1
        self.centre = track.pan
        small = mine.face is None or mine.face.width_px < s.track_min_face_px
        if not tracking:
            pan, tilt = _deg(track.head.pan_rad), _deg(track.head.tilt_rad)
            what = f"a face, {px} px" if mine.face is not None else "a person (body)"
            line = f"{what} at {pan:+.0f}/{tilt:+.0f} deg: tracking"
            if mine.face is not None and small and not self.smiled:
                line += " (small: no smile yet)"
            score = round(mine.face.score, 2) if mine.face is not None else None
            self._note("gaze_face", now, line, pan_deg=pan, tilt_deg=tilt, px=px, score=score,
                       small=small, check=self.verifying, source=mine.source,
                       track=track.id)  # fmt: skip
            self.phase, self.fallback, self.verifying, self.stops = "track", None, False, None
            self.engaged_s = now
        if not small:
            self._smile()

    def _missed(self, now: float, *, looked: bool) -> None:
        """A frame without the person engaged (``looked``: one was taken): a check goes back,
        a seek reads the voice again, then takes its next stop, then gives up; a person tracked
        is lost after ``face_lost_frames`` such frames and ``face_lost_s``."""
        s = self.settings
        track = self.track
        if self.phase == "seek":
            self.seek_left -= 1
            late = self.verify_until is not None and now >= self.verify_until
            if self.seek_left > 0 and not late:
                return
            if self.verifying or track is None:
                self._unseen(now)
                return
            if self.stops is None:  # arrived, nobody: the voice once more, then the stops
                if self._follow(now):
                    return
                if self.listening is not None:  # they speak on: their voice leads, no sweep yet
                    self.seek_left = 1
                    return
                centre = self.centre if self.centre is not None else track.pan
                self.stops = sweep(centre, track.head.tilt_rad,
                                   span_rad=math.radians(s.seek_pan_deg))  # fmt: skip
            while self.stops:
                stop = self.reach.clamp(self.stops.pop(0))
                if self.aim is not None and stop.off(self.aim) <= math.radians(s.aim_min_deg):
                    continue  # at the neck's reach: the same place
                if self._hold(stop, now, "seek", "adjust", track=track.id):
                    self.seek_left = max(1, s.seek_frames)
                    self.next_frame = now + BLIND_RETRY_S
                return
            self._unseen(now)
        elif looked and track is not None and self.phase == "track":
            self.misses += 1
            since = max(track.seen_s or 0.0, self.engaged_s)
            unseen = now - since
            if self.misses >= s.face_lost_frames and unseen >= s.face_lost_s:
                self._lose(track, now, unseen)

    def _lose(self, track: Track, now: float, unseen: float) -> None:
        """The person engaged is gone from the frames: the head goes to their voice when it was
        heard while nothing saw them (``seen_fresh_s`` after the last sighting), else stays;
        frames go on."""
        s = self.settings
        self.phase, self.centre = "voice", track.pan
        line = f"the person is lost ({unseen:.1f} s, {self.misses} frames unseen)"
        voice = track.voice_pan
        last_seen = track.seen_s if track.seen_s is not None else -math.inf
        newer = track.heard_s is not None and track.heard_s - last_seen > s.seen_fresh_s
        aim = self.aim
        far = (
            voice is not None
            and aim is not None
            and abs(wrap(voice - aim.pan_rad)) > math.radians(s.doa_moved_deg)
        )
        if newer and far and voice is not None and aim is not None:
            self.tracks.followed(track, voice, track.heard_s or now)
            self.centre = track.pan
            line += f": to the voice at {_deg(voice):+.0f} deg"
            self._note("gaze_lost", now, line, what="face", unseen_s=round(unseen, 2),
                       frames=self.misses, track=track.id)  # fmt: skip
            self._hold(self.reach.clamp(Aim(voice, aim.tilt_rad)), now, "to the voice",
                       "adjust", track=track.id)  # fmt: skip
            return
        line += ": the head stays on its place"
        self._note("gaze_lost", now, line, what="face", unseen_s=round(unseen, 2),
                   frames=self.misses, track=track.id)  # fmt: skip

    def _unseen(self, now: float) -> None:
        """A seek found nobody: a check goes back to the person engaged before; any other
        stays on the voice."""
        s = self.settings
        fallback = self.tracks.get(self.fallback) if self.verifying else None
        self._forget_look()
        if fallback is not None:
            pan = _deg(fallback.head.pan_rad)
            line = f"nobody toward the voice: back to the face at {pan:+.0f} deg"
            self._note("gaze_lost", now, line, what="check", track=self.engaged)
            self.engaged, self.engaged_s, self.misses = fallback.id, now, 0
            self.centre = fallback.pan
            if self._hold(self.reach.clamp(fallback.head), now, "back to the face", "turn",
                          track=fallback.id):  # fmt: skip
                self.phase = "track"
            return
        self.phase = "voice"
        self._note("gaze_lost", now, "nobody seen: the head stays on the voice", what="seek",
                   track=self.engaged)  # fmt: skip
        track = self.track
        if track is not None and self.aim is not None:
            centre = self.centre if self.centre is not None else track.pan
            back = self.reach.clamp(Aim(centre, track.head.tilt_rad))
            if back.off(self.aim) > math.radians(s.aim_min_deg):
                self._hold(back, now, "back to the voice", "adjust", track=track.id)
        if s.smile_unseen:
            self._smile()

    def _forget_look(self) -> None:
        """A seek's, a check's and a track's working state cleared (the people known stay)."""
        self.fallback, self.verifying, self.verify_until = None, False, None
        self.stops, self.misses = None, 0

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
        **extra: Any,
    ) -> None:
        """An utterance's verdict: its line (when it says something) and its event."""
        fields = _voice_fields(voice, heard)
        track = self.track
        if self.phase == "track" and track is not None:
            fields["face_deg"] = _deg(track.pan)
        self._note("gaze_voice", now, line, verdict=verdict, phase=self.phase, **fields, **extra)

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
        if voice.robot:
            fields["robot"] = voice.robot
    return fields


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
        """The person began an utterance: the head may follow their voice."""
        self._post("listening", t)

    def speech_ended(self, heard: Heard) -> None:
        """An utterance was read."""
        self._post("heard", heard)

    def part_queued(self, turn: int, index: int, part: Part) -> None:
        """The robot speaks."""
        self._post("speaking", True)

    def voice_started(self, turn: int) -> None:
        """The voice's first chunk is out: frames may resume."""
        self._post("voiced", turn)

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
    def _wheels_said(self, event: WheelEvent) -> None:
        """The wheels' word (their thread): onto this queue, whatever the switch says (a turn
        let go is recognised by its id and ignored)."""
        self._queue.put(("wheels", (event,), self._clock()))
        self.idle.clear()

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
                self.machine.post = self._wheels_said
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
            "listening": lambda m: m.speech_started(args[0], now),
            "speaking": lambda m: m.speaking(args[0], at),
            "voiced": lambda m: m.voiced(now),
            "drive": lambda m: m.drive(args[0], now),
            "closed": lambda m: m.closed(args[0], now),
            "wheels": lambda m: m.wheels_event(args[0], now),
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
    finder: FaceFinder | None = None,
    on_face: FaceListener | None = None,
    wheels: Wheels | None = None,
) -> SpeakerGazeWorker:
    """The speaker gaze on the robot: the arbiter's door on this Mac, the board's camera and
    head server at ``host``, the array's directions from ``doa`` (the cascade's audio link),
    ``wheels`` (the cascade's one :class:`pepin.expression.wheels.WheelsExecutor`; none: the head
    alone); built, and the finder's models loaded, the first time its switch is on (config/
    expression.json's ``flags.speaker_gaze`` at the start when ``on`` is None). ``finder``: one
    shared with person recognition (else its own); ``on_face`` hears the speaker's faces."""

    def build(settings: SpeakerSettings) -> SpeakerGaze:
        from pepin.deployment import config_file
        from pepin.expression.door import SpeakerDoor
        from pepin.expression.executors import NoWheels
        from pepin.expression.finder import board_finder
        from pepin.head_link import HeadClient
        from pepin.neck import NeckConfig

        looks: FaceFinder | None = finder
        if looks is None:
            try:
                looks = board_finder(host)  # config reads only: the models load on warm()
            except (OSError, ValueError, KeyError) as exc:
                say(f"no face finder ({exc}): the head turns to the voice only")
        return SpeakerGaze(
            settings,
            door=SpeakerDoor(aim_wait_s=settings.aim_wait_s),
            doa=doa,
            finder=looks,
            wheels=wheels if wheels is not None else NoWheels(),
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
        service: ServiceFinder,
        switched: SwitchedFinder,
        turn: WheelsExecutor,
        link: BoardLink,
    ) -> None:
        observer: ConversationObserver = worker
        flag: LiveFlag = worker
        gaze_door: GazeDoor = door
        faces: FaceFinder = finder
        records: FaceFinder = service
        either: FaceFinder = switched
        wheels: Wheels = turn
        doa: DoaSource = link
        del observer, flag, gaze_door, faces, records, either, wheels, doa
