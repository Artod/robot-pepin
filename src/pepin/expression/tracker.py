"""The speaker gaze: the head on the person the robot talks with, the head's base layer.

One engine, coarse to fine, after Pepper's basic awareness (a stimulus turns the head, a look
checks for a person there, a person found is engaged and tracked, none: back to the one engaged
before):

1. STIMULI. The array's direction stream (:mod:`pepin.expression.doa`: about 10 readings a
   second on the board's clock, each TAGGED: the array's speech flag, the ears' Silero
   probability, the robot's own speech, the start of the robot's own wheels' turn; one rule,
   ``speaker.doa.doa_tags``, says which are the person's voice) with the robot's own speaking
   spans cut out, so an utterance that overlaps a reply keeps its clean part; the utterances
   the ears close (words, peak level, the robot's name); one head-camera frame at a time
   (:mod:`pepin.expression.finder`: faces, and people's boxes when a body detector runs); the
   robot speaking.
2. PEOPLE (:mod:`pepin.expression.tracks`): a few tracks, each a bearing and a head aim placed by
   a face, a body box or a voice; freshness decides between them.
3. WHOM (:mod:`pepin.expression.policy`): the person speaking. The name always wins; a person
   seen before is switched to at once; an unknown voice while someone is engaged is a check (one
   frame there, back when nobody); a sound, a quiet or a thin utterance moves nothing.
4. HOW: one ``angles`` aim, pan and tilt together, eased on the servo's own trapezoid
   (:class:`pepin.expression.speaker.Ease`): a TURN (``turn_acc_deg_s2``: 20 deg in 0.35 s, 60
   in 0.61 s) to a new person, an ADJUST (``adjust_acc_deg_s2``) for corrections, follows and the
   seek's stops, and while the body turns a GLIDE (below). Tracking is a PURSUIT
   (:mod:`pepin.expression.pursuit`, since 2026-10-14, Artem: "вижу непрерывно"; before, every
   re-aim waited for the head to settle and the frames for a still head, so a person walking at
   30 deg/s was sampled between moves and lost): every record of the camera channel is taken
   whether the head stands or moves, each box placed with the head's pose at its frame's own
   exposure stamp; the person's speed is fitted over their sightings at those stamps, and the
   head is aimed where they will be ``lead_s`` from now (at most ``lead_max_deg`` past the last
   sighting), every aim sent without waiting like the glide's, so the head moves on while the
   frames go on; a person standing still moves the head past the dead band only. A SEEK runs
   only when the
   person engaged has a bearing and nobody is seen there: the last ``doa_follow_s`` of the voice
   first, then ``seek_pan_deg`` outward and inward of the bearing at the remembered tilt, one
   frame a stop, then back to the voice. While the person speaks and nothing has seen them for
   ``seen_fresh_s``, the head FOLLOWS the voice's stream.
5. THE BODY: one rule. The head serves any bearing within the forward arc (``arc_deg``, +-80
   deg); the person engaged past it turns the body toward them (:mod:`pepin.expression.plan`, at
   most ``wheels.max_turn_deg`` a request, again while they are still past it), off this
   machine's thread, alongside the head's own turn, neither waiting for the other. Who turns it
   is ``body_turn``: ``spin`` (since 2026-10-14) one request of the whole turn to the goal
   server's ``spin``, Nav2's Spin with its own collision check (:class:`GoalSpins`; the wheels'
   base link opened first for the odometry's heading,
   :meth:`pepin.expression.wheels.WheelsExecutor.listen`); ``wheels`` the expressive wheels
   executor's twists (:mod:`pepin.expression.wheels`). A turn that cannot (refused, aborted, a
   blocked arc, the caps) leaves the head alone on the person, as far as it reaches. A person's
   ask outranks the gaze's own turn, and the goal server alone decides it (its one rule: the
   spin goes with the source ``gaze``): a tool's drive that comes while the gaze's spin runs
   cancels it there, and the spin's end says what it gave way to (``cancelled_for``); the
   gaze's spin asked while a tool's drive runs is refused, "a drive runs", and waits for the
   person's next words (2026-10-14: a spin that ran on after approach had the goal server
   refuse go_to(home), "already driving"). A TOOL'S DRIVE OR TURN HOLDS THE BODY (Artem,
   2026-10-14: "если повернулся по команде, доворот ждёт первую фразу"; turn(180) and
   turn(printer) were undone by the arc's rule seconds later): from the drive's notice (or the
   turn tool's :meth:`SpeakerGaze.keep_person`) the gaze's own body turn waits for the person's
   next utterance with words begun after the drive ended; the head tracks them meanwhile, as
   far as it reaches (``gaze_body_hold`` rows: held, and given back). A drive that never ran
   (the loop's word at its end: every call refused, or the turn already facing its target)
   holds nothing: the hold is lifted at once and the pause's people come back.
6. THE FRAME: the people's bearings are kept in the cart's frame at the newest heading of the
   wheels' odometry (the base server's state lines, on the board's clock, whoever turned the
   cart). A turn since the bearings' stamp is taken off every person, off each of the array's
   readings by its own stamp (an utterance spoken across a turn gives the bearing the person has
   now). WHILE THE BODY TURNS THE CAMERA KEEPS THE PERSON (since 2026-10-14, Artem: "голова
   лочится на человека по камере сама; колёса просто доворачивают"; the wheels slip, the
   odometry turns while the cart does not, and a head aimed by it turned off the person): every
   record is taken as in the pursuit, each person placed relative to the cart by the head's
   pose at the frame's exposure and their angle through the lens (the neck's encoders and the
   camera, no odometry), and the head is aimed at that place plus the pursuit's lead, fitted
   over those places (the cart's own turn is in them, as the camera saw it). Only while no
   frame has shown the person for ``TURN_FRAMES_S`` (out of view or reach, no records) does the
   odometry move the head: THE GLIDE, a prior until a frame shows them again. A frame's people
   are matched against where the camera's fit puts the person at its stamp (THE MATCH,
   :mod:`pepin.expression.pursuit`); only without that fit is the odometry's place sought with
   a seek's reach (its slip), from a frame looking there, never taking someone else the frames
   show as themselves; and a person placed by a voice alone takes nobody from a frame exposed
   before the head looked their way (2026-10-14 review: a bystander on the way). The lead
   while the body turns is at most what the odometry's rate turns over its span: the wheels'
   stop ends it before the turn's end is told (the camera's fit lags a stop by half its
   window: up to lead_max_deg past the person in the simulation). The arbiter answers an aim
   once the head has settled on it, so a re-aim per odometry tick as a stop-and-go move made
   the head lag the person by 20-35 deg in the room and saw-tooth +-5 deg (2026-10-13, the 158
   deg turn: five moves of 20-25 deg, 0.14 s of standing between). While the body turns every
   aim goes out WITHOUT WAITING (a short thread makes the door's call and posts its answer into
   the inbox, like the wheels' word; a newer aim preempts the older at the arbiter, which the
   mover takes in flight), and every ``FOLLOW_EVERY_S`` the glide's aim is the person's pan
   ``glide_ahead_s`` from now at the body's measured rate, at a speed ceiling of ``glide_gain``
   times that rate (``adjust_acc_deg_s2``'s ramp): the head follows a target that is always a
   little ahead and never stops to settle. The ``gaze_reaim`` rows say which moved the head
   (``the body turns: by the frames``, ``the body turns: by the odometry (no frame)``); the
   voice's stream, turned by the same odometry, stays the last resort (the follow); the turn's
   end re-centres the residue from whatever is known.
7. A TURN IN PLACE the gaze did not start (the turn tool's Nav2 Spin, which the cascade tells
   as a drive) keeps the person: the tool asks (:meth:`SpeakerGaze.keep_person`), the drive's
   pause is given back, and the head keeps them through the turn as through the gaze's own
   (the frames; the odometry's glide while they show nobody, none without the odometry's
   heading: the wheels' link opens at their own first motion). The drive's end re-centres.

The phases: ``seek`` (looking for the person engaged: a check, or the stops), ``track`` (they
are seen), ``voice`` (nobody seen: the head on their voice, frames go on), ``paused`` (a drive)
and ``idle`` (no conversation, or the switch off: no frames, no calls). The person engaged is
lost after ``face_lost_frames`` frames AND ``face_lost_s`` without them, the frames counted
being the records the finder gave (a look that got none is no frame); the head then goes to
their voice when it was heard after they were last seen.

The voice comes first: from a reply's first part going to the voice until its first audio chunk
is with the player (``hold_for_voice``) no frame is taken, and while the robot speaks frames come
at ``track_speaking_hz`` (10, the listening rate, since 2026-10-14: the perception service
detects at its own rate on the CPU and the Neural Engine, never the GPU the voice uses, whatever
the gaze reads; config/expression.json ``finder``). The head is held through the arbiter's door
as an ``angles`` look of the idle band, source ``voice.speaker`` (:mod:`pepin.expression.head`):
TTL ``ttl_s``, renewed every ``renew_s`` (an aim renews it too); a renewal that finds the look
gone (a ``look`` the person asked for took the head) asks again, queued behind it. A frame counts
only while the head is ours and carries no gesture (the idle layer's breath aside), and, until
the person is tracked, while it stands still; a frame skipped for any of these is no look (no
seek's stop spent, no check ended) and is counted by its cause, a gesture's tried again
``BLIND_RETRY_S`` later. The first clear face of a
conversation is told in the snapshot (``first_face_s``): the gesture engine smiles at it, the
face's one writer; ``speaker.smile`` (null as shipped) naming a face brings back this machine's
own smile through the head server's face owner (source ``expression``), once a conversation.

The gesture engine reads this machine's :class:`pepin.expression.speaker.Attention` (the phase:
seek, check, turning, track, voice; when the person engaged was last seen; whether they stand
past the arc) and keeps its head overlays and its arm clear of it; a body turn asks the engine
for the arm first (:class:`pepin.expression.speaker.ArmRelease`) and starts once it is at rest.
Owner after slice 0: the attention engine's own (nobody of A, B or C edits it).

Every decision prints one line (``say``) and, through :class:`pepin.expression.conversation.
EventSink`, lands in the conversation's events.jsonl on the session clock: ``gaze_wake``,
``gaze_voice`` (each utterance's bearing, verdict and track), ``gaze_doa`` (each utterance's
readings: ``[seconds before its end on the board's clock, the array's angle, 1 speech / 0 not /
-1 the robot's own voice]`` by the array's own flag, and a fourth number when the cart has turned
since: how far, in degrees, taken off that reading's bearing; beside them ``tags`` names the
rule, ``voice_p`` each reading's Silero probability (null: unknown), ``voiced`` the rule's 1 or 0
and, when any reading fell in the start of the wheels' own turn, ``moving``), ``gaze_sources``
(after each utterance under ``speaker.doa.direction`` mac: the register of voices,
:class:`pepin.expression.tracks.VoiceSources`, fed with every source peak of the utterance's
voiced readings: each source's id, bearing in the cart's frame, age, strength, peaks, ``who``
(reserved) and whether it is the person engaged's; the register changes nobody's engagement),
``gaze_reaim`` (each aim asked of the arbiter, why, its profile and its time), ``gaze_face`` (a
person found: by face or body), ``gaze_lost`` (a seek, a check or a person that came to nothing;
the door lost), ``gaze_wheels``, ``gaze_spin`` (a turn in place: the gaze's own Spin,
``own``, at its start and its ``end``, ``gave_way`` and ``cancelled_for`` when the goal server
cancelled it for a person's request; the turn tool's, kept through), ``gaze_body_hold`` (a
tool's drive held the gaze's own body turn, and the person's words gave it back, or the drive
never ran: ``ran`` false),
``gaze_release`` and, at the close, ``gaze_frames`` (the conversation's frames taken and
skipped, by cause: holder, driving, gesture; ``frames_posed`` placed at their exposure's pose,
``frames_unposed`` at the pose of their read, ``frames_none`` looks that got no record,
``frames_turning`` records taken while the body turned, ``frames_away`` those of them exposed
before the head looked toward a person placed by a voice alone).

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
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pepin.audio_doa import MicReading
from pepin.expression.conversation import SPEAKER_GAZE, EventSink, Heard
from pepin.expression.doa import (
    WHEELS_ONSET_S,
    Direction,
    DoaChoice,
    MacSource,
    Span,
    Tagger,
    Turned,
    UtteranceDoa,
    VoiceBearing,
    VoiceRule,
    VoiceSeries,
    bearing_deg,
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
from pepin.expression.pursuit import HeadPoses, Pursuit, tilt_turn_s
from pepin.expression.speaker import (
    NO_ATTENTION,
    ArmRelease,
    Attention,
    AttentionPhase,
    DoaMap,
    DoaSource,
    Ease,
    FaceFinder,
    FaceFrame,
    FaceSighting,
    GazeAnswer,
    GazeDoor,
    HeadView,
)
from pepin.expression.tracks import Sighting, Track, Tracks, VoiceSources, sightings, wrap
from pepin.gaze import Aim, GazeSettings, Reach, Speed

if TYPE_CHECKING:
    from pepin.audio_doa import MicDoa
    from pepin.expression.conversation import ConversationObserver, LiveFlag
    from pepin.expression.door import SpeakerDoor
    from pepin.expression.finder import ServiceFinder, SnapshotFinder, SwitchedFinder
    from pepin.expression.wheels import WheelsExecutor
    from pepin.voice_live.sources import BoardLink

logger = logging.getLogger(__name__)

Phase = Literal["idle", "seek", "voice", "track", "paused"]
Move = Literal["turn", "adjust", "glide"]
CONFIG_FILE = "expression.json"
FACE_SOURCE = "expression"  # the face's expressions of the body layer, at the head server
BLIND_RETRY_S = 0.1  # a frame refused for a blind head is tried again this soon
TICK_EPS_S = 1e-6
FOLLOW_EVERY_S = 0.1  # the voice's stream is read this often while the head follows it
RATE_WINDOW_S = 0.1  # the body's rate is read over this much of the odometry's headings
GLIDE_FLOOR_DEG_S = 20.0  # a glide's speed ceiling as the body comes to rest: the catch-up
# While the body turns, a frame that showed the person engaged this recently keeps the head on
# the frames (three records at 10 Hz: a blurred frame or two loses nothing); past it the glide
TURN_FRAMES_S = 0.3
BY_FRAMES = "the body turns: by the frames"  # gaze_reaim's why while the body turns
BY_ODOMETRY = "the body turns: by the odometry (no frame)"
SETTLED = "the body turned: by the frames"  # the turn's residue, by the first frame after it
AIMS_IN_FLIGHT = 8  # aims sent without waiting still unanswered: past it none goes (a stuck door)
LISTEN_MAX_S = 30.0  # an utterance never closed is not followed longer than this
SPOKE_KEPT = 8  # the robot's latest speaking spans kept to judge an utterance by
TEXT_KEPT = 40  # an utterance's words kept in its event, to pair it with the loop's "heard"
TURN_EPS_RAD = 1e-4  # a heading change under this is no turn
TURNED_LOGGED_DEG = 0.05  # a reading turned less than this since its stamp logs no turn
# The tilt of a pursuit never turns back sooner than this after its last turn: the arbiter's rule
# for overlays (gaze.clip_max_hz 2.5 as shipped, config/knobs.json; the mast rings at 5.3 Hz)
TILT_TURN_S = tilt_turn_s(GazeSettings.clip_max_hz)
Report = Callable[[str], None]
FaceListener = Callable[[FaceFrame, FaceSighting], None]  # a frame and the speaker's face in it
# The threads' word into the machine: ("moved", turn id, degrees), ("done", turn id,
# WheelReport), ("aimed", aim id, GazeAnswer: an aim sent without waiting, answered), ("spin",
# turn id, the goal server's event on a spin of ours)
WheelEvent = tuple[str, int, Any]
BodyTurn = Literal["spin", "wheels"]  # who turns the body toward the person (speaker.body_turn)
SpinWord = Callable[[dict[str, Any]], None]  # hears one of the goal server's events on a spin
SPIN_SUCCEEDED = 4  # action_msgs/GoalStatus, as the goal server's done carries it
SPIN_CANCELED = 5
SPIN_STATUS = {SPIN_SUCCEEDED: "succeeded", SPIN_CANCELED: "cancelled", 6: "aborted"}
# Whose the gaze's spin is at the goal server: its one rule lets a person's request (a tool's,
# source tool) cancel it and refuses it while a person's runs.
SPIN_SOURCE = "gaze"
Spawn = Callable[[Callable[[], None]], None]  # runs a door's call off the machine's thread


FACE_ANSWER_S = 1.0  # a tool's face(): how long it waits for the gaze thread's answer


@dataclass
class Turning:
    """A turn of the body toward the speaker, running: its id, the person it turns toward, how
    far it was asked and how far it has turned so far (degrees, + left), and who turns it (the
    wheels' executor, or Nav2's Spin: when it was asked, whether Nav2 took it, its run)."""

    id: int
    track: int
    asked_deg: float
    turned_deg: float = 0.0
    starts_s: float | None = None  # waiting for the arm to rest until then (None: started)
    via: BodyTurn = "wheels"
    asked_s: float | None = None  # a spin: when it was asked of the goal server
    accepted: bool = False  # a spin: Nav2 took it (the cart may have turned since)
    run: int | None = None  # a spin: the goal server's run


@dataclass(frozen=True)
class TurnInPlace:
    """A turn in place the gaze keeps the person through (the turn tool's Nav2 Spin): when it
    was asked, and whether the odometry's heading is there to glide on."""

    since_s: float
    glides: bool


@dataclass(frozen=True)
class PutAside:
    """What a drive's pause put aside, for a drive that turns out to be a turn in place: the
    people and the board stamp their bearings are kept at, the one engaged, the phase, the aim
    and whether the head was this source's."""

    people: list[Track]
    frame_s: float | None
    engaged: int | None
    phase: Phase
    aim: Aim | None
    ours: bool


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
    doa_tags: str = "silero"  # speaker.doa: the rule for the person's voice (silero, chip)
    voice_p_min: float = 0.5  # speaker.doa: Silero's probability the silero rule asks for
    direction: Direction = "mac"  # speaker.doa: whose readings, the Mac's own or the chip's
    mac_voice_p_min: float = 0.5  # speaker.doa: Silero's p a chunk's peaks are computed from
    mac_peak_ratio: float = 0.6  # speaker.doa: a second source's response over the strongest's
    mac_reading_s: float = 0.1  # speaker.doa: one Mac reading per this (the chip's cadence)
    mac_doa_map: DoaMap | None = None  # speaker.doa.mac_readings: the Mac's angles into the cart
    arc_deg: float = 60.0
    seek_tilt_deg: float = 3.0
    seek_pan_deg: float = 30.0
    seek_frames: int = 1
    seek_every_s: float = 0.4
    seek_min_face_px: float = 24.0  # the rectified eye's pixels (fx 454.5): 28 in the raw eye's
    track_min_face_px: float = 34.0  # the rectified eye's pixels: 40 in the raw eye's
    min_body_score: float = 0.5
    track_hz: float = 10.0
    track_speaking_hz: float = 10.0
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
    glide_ahead_s: float = 0.2
    glide_gain: float = 2.0
    smile: str = ""  # "": the gesture engine smiles at the first face (the face's one writer)
    smile_hold_s: float = 2.0
    smile_unseen: bool = False
    body_turn: BodyTurn = "spin"  # Nav2's Spin through the goal server; "wheels": the executor's
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
        own = ("doa_map", "body", "window_s", "min_speech", "doa_tags", "voice_p_min", "direction",
               "mac_voice_p_min", "mac_peak_ratio", "mac_reading_s", "mac_doa_map")  # fmt: skip
        for name, default in vars(d).items():
            if name in own:
                continue
            if name in speaker:
                value = speaker[name]
                values[name] = "" if name == "smile" and value is None else type(default)(value)
        body_block = data.get("wheels", {})
        body = BodyLimits(
            **{
                name: type(default)(body_block[name])
                for name, default in vars(BodyLimits()).items()
                if name in body_block
            }
        )
        direction = str(doa.get("direction", d.direction))
        if direction not in ("chip", "mac"):
            raise ValueError(f"speaker.doa.direction is chip or mac, not {direction!r}")
        settings = replace(
            d,
            doa_map=doa_map,
            window_s=float(doa.get("window_s", d.window_s)),
            min_speech=int(doa.get("min_speech", d.min_speech)),
            doa_tags=str(doa.get("doa_tags", d.doa_tags)),
            voice_p_min=float(doa.get("voice_p_min", d.voice_p_min)),
            direction="mac" if direction == "mac" else "chip",
            mac_voice_p_min=float(doa.get("mac_voice_p_min", d.mac_voice_p_min)),
            mac_peak_ratio=float(doa.get("mac_peak_ratio", d.mac_peak_ratio)),
            mac_reading_s=float(doa.get("mac_reading_s", d.mac_reading_s)),
            mac_doa_map=mac_doa_map_of(doa),
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
        if self.doa_tags not in ("silero", "chip") or not 0.0 <= self.voice_p_min <= 1.0:
            raise ValueError("speaker.doa: doa_tags silero or chip, voice_p_min within 0..1")
        if not 0.0 <= self.mac_voice_p_min <= 1.0 or not 0.0 <= self.mac_peak_ratio <= 1.0:
            raise ValueError("speaker.doa: mac_voice_p_min and mac_peak_ratio within 0..1")
        if self.mac_reading_s <= 0.0:
            raise ValueError("speaker.doa: mac_reading_s > 0")
        if self.glide_ahead_s < 0 or self.glide_gain < 1.0:
            raise ValueError("speaker: glide_ahead_s >= 0, glide_gain at least 1")
        if self.body_turn not in ("spin", "wheels"):
            raise ValueError("speaker: body_turn spin or wheels")

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


def mac_doa_map_of(doa: Mapping[str, Any]) -> DoaMap | None:
    """``speaker.doa``'s map for the Mac's own angles: interpolated between its four
    ``mac_readings`` when all are measured, else None (the chip's map stands in, said once)."""
    readings = doa.get("mac_readings") or {}
    places = [readings.get(p) for p in ("front", "left", "behind", "right")]
    known = [float(v) for v in places if v is not None]
    if len(known) != len(places):
        return None
    fit = fit_doa_map(*known)
    if fit.refusal is not None:
        raise ValueError(f"speaker.doa.mac_readings: {fit.refusal}")
    return fit.doa_map


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


class Spins(Protocol):
    """Nav2's Spin through the goal server (``{"cmd": "spin", "deg": N, "source": "gaze"}``, +
    left): the body's turn when ``speaker.body_turn`` is spin. :meth:`spin` returns at once;
    ``said`` hears each of the server's events on a thread that is not the caller's: accepted,
    feedback, arrival, and last always done or error (a refusal, a lost link). Nothing here
    cancels: the goal server's one rule cancels the gaze's spin for a person's request."""

    def spin(self, deg: float, said: SpinWord) -> None:
        """Ask for a turn in place of ``deg`` degrees."""
        ...


class GoalSpins:
    """:class:`Spins` over the goal server's socket on this Mac (:mod:`pepin.goal_link`): one
    connection a spin, read to its end on a daemon thread of its own."""

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        """The goal server at ``host:port`` (goal_link's by default); nothing connects yet."""
        from pepin import goal_link

        self.host = host or goal_link.HOST
        self.port = port or goal_link.PORT

    def spin(self, deg: float, said: SpinWord) -> None:
        """The request sent, and its events read, on a thread of their own."""
        threading.Thread(
            target=self._run, args=(deg, said), name="speaker-spin", daemon=True
        ).start()

    def _run(self, deg: float, said: SpinWord) -> None:
        from pepin import goal_link

        request = {"cmd": "spin", "deg": round(deg, 1), "source": SPIN_SOURCE}
        try:
            for event in goal_link.events(
                request, self.host, self.port, goal_link.ASK_TIMEOUT_S, goal_link.GOAL_SILENCE_S
            ):
                said(event)
                if event.get("event") in ("done", "error"):
                    return
        except Exception as error:  # refused, timed out, dropped: the spin's end all the same
            said({"event": "error", "detail": f"goal server {self.host}:{self.port}: {error}"})
            return
        said({"event": "error", "detail": "the goal server closed the spin without its end"})


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
        arm: ArmRelease | None = None,
        series: VoiceSeries | None = None,
        spins: Spins | None = None,
    ) -> None:
        """The seams it drives; ``doa`` None or ``finder`` None: no voice or no eyes;
        ``on_face`` hears every frame's face of the person engaged (person recognition);
        ``events`` takes every decision as an event of the conversation's log; ``arm``: the
        gesture engine, asked for the arm before the body turns (None: nobody to ask);
        ``series``: the ears' speech probabilities the readings are tagged with (None: no
        speech model, so the ``silero`` rule hears nobody); ``spins``: the goal server's spin,
        the body's turn under ``body_turn`` spin (None: no goal server, every spin refused)."""
        self.door, self.doa, self.finder = door, doa, finder
        self.wheels, self.face, self.reach = wheels, face, reach
        self.report = report
        self.on_face = on_face
        self.events = events
        self.settings = settings
        self.tracks = self._tracks_of(settings)
        self.sources = self._sources_of(settings)  # the register of voices (direction mac)
        self._mac_map_said = False  # the chip's calibration standing in for the Mac's, told
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
        self.moved: list[Span] = []  # the starts of our wheels' turns: their jerk is no voice
        self.series = series  # the ears' Silero probabilities (laptop clock)
        self.frame_s: float | None = None  # the board stamp the people's bearings are kept at
        self.drift = 0.0  # radians the cart turned under the head since its last aim
        self.ours = False  # the head holds this source's aim (the arbiter said so last)
        self.body_waits = False  # a turn fell short: none until the person's next utterance
        # A tool's drive or turn ran: the gaze's own body turn waits for the person's next words
        # (why), those begun after the drive ended (``hold_after_s``; None while it runs)
        self.body_hold: str | None = None
        self.hold_after_s: float | None = None
        self.held_s = 0.0  # when the hold began
        # The hold as the running drive found it (why, after, since): a drive that never ran
        # gives it back as it was (an earlier drive's that ran stands)
        self.hold_before: tuple[str | None, float | None, float] | None = None
        self.gave_up: set[int] = set()  # people a turn that moved fell short for: no more
        self.wake: Heard | None = None  # the cascade tells it once more as heard: not twice
        self.counts: Counter[str] = Counter()
        self.turning: Turning | None = None  # the wheels' turn toward the person, running
        self.spin: TurnInPlace | None = None  # a turn in place kept through (keep_person)
        self.put_aside: PutAside | None = None  # what the running drive's pause put aside
        self._turn_ids = itertools.count(1)
        # The wheels answer on their own thread: their word is posted here and taken on this
        # machine's (the worker replaces ``post`` with a put on its queue; alone, ``tick``
        # takes the inbox).
        self.inbox: deque[WheelEvent] = deque()
        self.post: Callable[[WheelEvent], None] = self.inbox.append
        self.spawn: Spawn = _on_a_thread  # an aim sent without waiting makes its call here
        self._aim_ids = itertools.count(1)
        self.aim_id = 0  # the newest aim sent without waiting (its answer is the one that counts)
        self._in_flight = 0  # aims sent without waiting, not answered yet
        self._flight_lock = threading.Lock()
        self.arm = arm
        self.spins = spins
        # THE PURSUIT (pepin.expression.pursuit): the head's pose at each encoder reading the
        # state reads return, at the reading's own stamp (the board's clock, the frames' exposure
        # stamps' too; an arbiter that sends none: this Mac's wall clock of the read, logged once),
        # and the person engaged's sightings at their frames' stamps
        self.wall: Callable[[], float] = time.time
        self.poses = HeadPoses()
        self.unstamped_said = False  # an arbiter without head_stamp was logged
        self.pursuit: Pursuit | None = None
        # A body turn's end (wall clock) while the first frame exposed after it is awaited: that
        # frame re-centres the residue; the records exposed before it are the turn's
        self.turn_end_s: float | None = None
        self._tilt_way: tuple[int, float, float] = (0, -math.inf, -math.inf)  # way, turned, moved
        self.first_face_s: float | None = None  # the conversation's first clear face
        self._shown: tuple[AttentionPhase, float] = ("idle", 0.0)  # the phase last told, since
        self._counted: Counter[str] = Counter()  # the counts when the conversation opened

    def configure(self, settings: SpeakerSettings) -> None:
        """New knobs (the switch turned on re-reads them); the people and the voices known
        stay; the direction source takes the new direction and the Mac's knobs."""
        self.settings = settings
        known = self.tracks
        self.tracks = self._tracks_of(settings)
        self.tracks.adopt(known)
        voices = self.sources
        self.sources = self._sources_of(settings)
        self.sources.adopt(voices)
        if isinstance(self.doa, DoaChoice):
            self.doa.configure(settings.direction, voice_p_min=settings.mac_voice_p_min,
                               peak_ratio=settings.mac_peak_ratio,
                               reading_s=settings.mac_reading_s)  # fmt: skip

    @staticmethod
    def _sources_of(s: SpeakerSettings) -> VoiceSources:
        return VoiceSources(match_deg=s.voice_match_deg, forget_s=s.track_forget_s)

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
        self.wake, self.listening, self.first_face_s = wake, None, None
        if self.settings.doa_tags == "silero" and self.series is None and self.doa is not None:
            self.report("speaker.doa.doa_tags silero and no speech model in the ears: no reading"
                        " is the person's voice (doa_tags chip: the array's flag)")  # fmt: skip
        self._counted = Counter(self.counts)
        self.body_waits, self.gave_up = False, set()
        self.body_hold, self.hold_after_s = None, None  # the wake is the person's words
        self.hold_before = None
        self._sync_body(now)
        self.tracks.forget(now)
        self.sources.forget(now)
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
        self._release_body(heard, now)
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

    def drive(self, active: bool, now: float, *, at: float | None = None, ran: bool = True) -> None:
        """A drive started (the arbiter let the head go, the wheels are the drive's; every
        bearing goes stale: put aside for a turn in place to take back, :meth:`keep_person`; a
        spin of the gaze's own runs on until the goal server cancels it for the drive's goal) or
        ended
        (the next utterance orients; after a turn in place kept through, the person stays and
        the residue is re-centred). Either way the gaze's own body turn waits for the person's
        next words begun after the drive's end (Artem, 2026-10-14: a turn on command keeps the
        cart where the command put it until the person speaks; :meth:`_release_body`): ``at``
        when the loop told the end (the worker's post; ``now`` when None), not when this
        machine got to it (a look may hold its thread up to the finder's wait). An end with
        ``ran`` False (the loop's word: no goal of the drive's calls was taken, a refusal or a
        turn already facing its target) moved nothing and holds nothing (:meth:`_never_ran`)."""
        if active:
            self.hold_before = (self.body_hold, self.hold_after_s, self.held_s)
            if self.conversation:
                self._hold_body(now, "a tool's drive")
            self.pursuit, self.turn_end_s = None, None
            held = self.conversation and self.phase not in ("idle", "paused")
            self.put_aside = (
                PutAside(self.tracks.tracks, self.frame_s, self.engaged, self.phase, self.aim,
                         self.ours)
                if held
                else None
            )  # fmt: skip
            self.spin = None
            self._forget_look()
            self._let_wheels_go("a drive starts")
            self.tracks.clear()
            self.sources.clear()
            self.gave_up = set()
            self.engaged, self.ours = None, False
            self.phase, self.aim, self.next_frame, self.next_renew = "paused", None, None, None
            self.reorient = False
            return
        if not ran:
            self._never_ran(now)
            return
        self.put_aside, self.hold_before = None, None
        if self.body_hold is not None:  # words begun from the drive's end on free the body
            self.hold_after_s = now if at is None else min(at, now)
        if self.spin is not None:
            self.spin = None
            self._note("gaze_spin", now, "the turn in place ended", end=True)
            self._turn_ended(now)
        elif self.conversation:
            self.reorient = True

    def _never_ran(self, now: float) -> None:
        """A drive that never ran has ended (every call refused, or a turn already facing its
        target): nothing moved, so the hold it began is lifted at once (a row; the hold of an
        earlier drive that ran, still waiting for the person's words, stands as it was), and
        what its pause put aside comes back as :meth:`keep_person` brings it (a turn in place
        kept through ends without a re-centre: it never turned)."""
        before, self.hold_before = self.hold_before, None
        if before is not None and before[0] is not None:
            self.body_hold, self.hold_after_s, self.held_s = before
        elif self.body_hold is not None:
            why, held = self.body_hold, round(now - self.held_s, 1)
            self.body_hold, self.hold_after_s = None, None
            self._note("gaze_body_hold", now, f"released: the drive never ran ({why})",
                       held=False, why=why, held_s=held, ran=False)  # fmt: skip
        if self.spin is not None:
            self.spin, self.put_aside = None, None
            self._note("gaze_spin", now, "the turn in place never ran", end=True, ran=False)
            return
        if self._take_back(now):
            self._sync_body(now)
        elif self.conversation:
            self.reorient = True

    def _take_back(self, now: float) -> bool:
        """What the running drive's pause put aside comes back: the people at their own stamp
        (a sync turns them by whatever the cart turned since: a turn of the gaze's own, halted
        by the drive, coasting), the one engaged, the phase, the aim and the head's holder;
        whether anyone engaged was (else nothing comes back)."""
        aside, self.put_aside = self.put_aside, None
        if aside is None or aside.engaged is None:
            return False
        self.tracks.tracks, self.frame_s = aside.people, aside.frame_s
        self.engaged, self.phase, self.aim, self.ours = (
            aside.engaged, aside.phase, aside.aim, aside.ours,
        )  # fmt: skip
        self.next_renew = now  # the pause renewed nothing: the look's TTL first
        self.next_frame = now if self.finder is not None else None
        return True

    def keep_person(self, now: float, why: str) -> str | None:
        """A turn in place of the body's, asked by a tool (turn: Nav2's Spin, which the cascade
        tells as a drive): the person engaged is kept through it, the drive's pause given back
        (what it put aside), and the head keeps them by the frames as through the gaze's own
        turn (THE FRAME: the odometry's glide only while the frames show nobody, none without
        its heading). None when kept (or already), else why not; the drive's end ends it. The
        gaze's own body turn then waits for the person's next words (:meth:`drive`), unless the
        drive never ran (the turn already faced its target: :meth:`_never_ran`)."""
        if self.spin is not None:
            return None
        if self.phase == "idle":
            return "no conversation"
        self._hold_body(now, f"a tool's turn ({why})")
        if self.phase == "paused" and not self._take_back(now):
            return "nobody was engaged when the drive began"
        track = self.track
        if track is None:
            return "nobody engaged"
        glides = self.wheels.headings().latest() is not None
        self.spin = TurnInPlace(now, glides)
        self.counts["spins"] += 1
        line = f"the body turns in place ({why}): the head keeps the person by the frames" + (
            "" if glides else " (no odometry heading yet)"
        )
        self._note("gaze_spin", now, line, why=why, glides=glides, track=track.id)
        self._sync_body(now)
        return None

    def closed(self, reason: str, now: float) -> None:
        """The conversation is over: the head is let go and nothing runs until the next one;
        the people stay known (their places for the tilt, until forgotten); its frames, taken
        and skipped by cause, logged."""
        self.conversation = False
        frames = {k: v for k, v in (self.counts - self._counted).items() if k.startswith("frames")}
        self._note("gaze_frames", now, None, **frames)
        self.stop(now, f"the conversation closed ({reason})", forget=False)

    def stop(self, now: float, why: str, *, forget: bool = True) -> None:
        """Let the head go now (the switch off, the conversation's end), and the wheels."""
        self._let_wheels_go(why)
        self.spin, self.put_aside, self.pursuit, self.turn_end_s = None, None, None, None
        self.body_hold, self.hold_after_s, self.hold_before = None, None, None
        held = self.phase != "idle" or self.aim is not None
        self.voice_hold, self.ours = False, False
        self.phase, self.aim, self.next_frame, self.next_renew = "idle", None, None, None
        self.engaged, self.listening, self.next_follow = None, None, None
        self._forget_look()
        if forget:
            self.tracks.clear()
            self.sources.clear()
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

    def attention(self, now: float) -> Attention:
        """What attention does now, for the gesture engine (:class:`pepin.expression.speaker.
        Attention`): a body turn (waiting for the arm or running) over everything but idle and
        paused (a turn in place kept through is one), a seek that is a check named so."""
        phase: AttentionPhase = self.phase
        if self.phase not in ("idle", "paused"):
            if self.turning is not None or self.spin is not None:
                phase = "turning"
            elif self.phase == "seek" and self.verifying:
                phase = "check"
        if phase != self._shown[0]:
            self._shown = (phase, now)
        track = self.track
        if track is None:
            return Attention(phase, self._shown[1], self.finder is not None,
                             first_face_s=self.first_face_s)  # fmt: skip
        past = self._body_share(track.pan) != 0.0
        return Attention(
            phase, self._shown[1], self.finder is not None, track.seen_s, _deg(track.pan),
            _deg(track.head.tilt_rad), track.distance_m, past, self.first_face_s,
        )  # fmt: skip

    # ---- the clock ---------------------------------------------------------------------------
    def tick(self, now: float) -> float | None:
        """Do what is due (the wheels' word, the body's heading, a renewal, the voice followed,
        the body's turn, a frame, a lost head asked for again); when next, None for nothing
        until a moment. A frame waits for a reply's first chunk only (a body turn takes every
        record: THE FRAME); the heading is read every ``FOLLOW_EVERY_S`` while the body turns."""
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
        self._arc(now, free=self.ours)
        turn = self.turning
        if turn is not None and turn.starts_s is not None and now_ >= turn.starts_s:
            self._go(turn, now)
        waiting = self.voice_hold  # a reply's first chunk: no frame until it plays
        if self.next_frame is not None and now_ >= self.next_frame and not waiting:
            self._frame(now)
        frame = None if waiting else self.next_frame  # the voice's moment
        follow = self.next_follow if self._following(now) else None
        body = now + FOLLOW_EVERY_S if self._body_turns() else None  # its heading
        if turn is not None and turn.starts_s is not None:
            body = turn.starts_s  # the arm at rest: the turn starts
        due = [t for t in (self.next_renew, frame, follow, body) if t is not None]
        return min(due) if due else None

    # ---- the voice -----------------------------------------------------------------------------
    def _tagger(self) -> Tagger:
        """The array's readings tagged against the robot's own speech and wheels and the ears'
        speech probabilities, judged by ``speaker.doa.doa_tags``."""
        s = self.settings
        rule = VoiceRule("chip" if s.doa_tags == "chip" else "silero", s.voice_p_min)
        return Tagger(rule, self.series, tuple(self.spoke), tuple(self.moved))

    def _doa_map(self) -> DoaMap | None:
        """The map of the readings read now: the Mac's own four readings under direction mac
        (the chip's standing in, said once, while they are unmeasured), else the chip's."""
        s = self.settings
        if isinstance(self.doa, DoaChoice) and self.doa.active == "mac":
            if s.mac_doa_map is not None:
                return s.mac_doa_map
            if not self._mac_map_said:
                self._mac_map_said = True
                self.report("the Mac's direction through the chip's calibration (speaker.doa."
                            "mac_readings unmeasured: ros/tools/doa_calibrate.py --source mac"
                            " --write)")  # fmt: skip
        return s.doa_map

    def _utterance(self, heard: Heard, now: float) -> VoiceBearing:
        """``heard``'s bearing from its clean readings; its readings logged (``gaze_doa``) and
        their source peaks into the register of voices (``gaze_sources``)."""
        if self.doa is None:
            return VoiceBearing(None, None, 0, 0)
        s = self.settings
        turned = self._turned_since()
        doa_map = self._doa_map()
        found = utterance_voice(self.doa, heard, now, window_s=s.window_s, doa_map=doa_map,
                                min_speech=s.min_speech, tagger=self._tagger(),
                                turned=turned)  # fmt: skip
        self._log_readings(heard, found, now, turned)
        self._register(heard, found, now, turned, doa_map)
        return found.voice

    def _register(
        self, heard: Heard, found: UtteranceDoa, now: float, turned: Turned | None,
        doa_map: DoaMap | None,
    ) -> None:  # fmt: skip
        """Every source peak of the utterance's voiced Mac readings into the register of
        voices, in the cart's frame of now (each turned with the cart like the bearing), then
        the register logged; nobody's engagement changes here."""
        offset = board_offset(self.doa, now) if self.doa is not None else None
        if doa_map is None or offset is None:
            return
        fed = 0
        for t in found.tagged:
            reading = t.reading
            if not t.voice or not isinstance(reading, MicReading):
                continue
            for peak in reading.peaks:
                bearing = bearing_deg(peak.deg, doa_map)
                if turned is not None:
                    bearing -= math.degrees(turned(reading.stamp_s))
                self.sources.heard(wrap(math.radians(bearing)), reading.stamp_s - offset,
                                   peak.strength)  # fmt: skip
                fed += 1
        if not fed:
            return
        self.sources.forget(now)
        engaged = self.track
        mine = self.sources.nearest(engaged.pan) if engaged is not None else None
        rows = [
            {
                "id": s.id,
                "bearing_deg": _deg(s.bearing),
                "age_s": round(max(0.0, now - s.heard_s), 1),
                "strength": round(s.strength, 2),
                "peaks": s.peaks,
                "who": s.who,
                "engaged": mine is not None and s.id == mine.id,
            }
            for s in self.sources.sources
        ]
        self._note("gaze_sources", now, None, text=heard.text[:TEXT_KEPT], fed=fed, sources=rows)

    def _log_readings(
        self, heard: Heard, found: UtteranceDoa, now: float, turned: Turned | None
    ) -> None:
        offset = board_offset(self.doa, now) if self.doa is not None else None
        if not found.readings or offset is None:
            return
        end = heard.end_s + offset
        rows: list[list[float | int]] = []
        for t in found.tagged:  # the array's own flag, -1 for the robot's voice
            flag = -1 if t.tags.own else int(t.tags.chip)
            rows.append([round(t.reading.stamp_s - end, 2), t.reading.deg, flag])
        if turned is not None:
            degs = [math.degrees(turned(r.stamp_s)) for r in found.readings]
            if any(abs(d) >= TURNED_LOGGED_DEG for d in degs):
                rows = [[*row, round(d, 1)] for row, d in zip(rows, degs, strict=True)]
        tags: dict[str, Any] = {
            "tags": self.settings.doa_tags,
            "voice_p": [None if t.tags.voice_p is None else round(t.tags.voice_p, 2)
                        for t in found.tagged],
            "voiced": [int(t.voice) for t in found.tagged],
        }  # fmt: skip
        if any(t.tags.moving for t in found.tagged):
            tags["moving"] = [int(t.tags.moving) for t in found.tagged]
        self._note("gaze_doa", now, None, text=heard.text[:TEXT_KEPT], readings=rows, **tags)

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
        voice = recent_voice(self.doa, now, window_s=s.doa_follow_s, doa_map=self._doa_map(),
                             min_speech=s.follow_min_speech, tagger=self._tagger(),
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
        target = self._within(Aim(track.pan, track.head.tilt_rad))
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
        if self.pursuit is not None and self.pursuit.track != track.id:
            self.pursuit = None
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
        if view.driving and not self._spinning():  # its own spin is no drive to pause for
            self.phase, self.reorient, self.aim = "paused", True, None
            return
        if self.turning is not None and self.turning.track != track.id:
            self._let_wheels_go("another person is engaged")
        # The wheels first: their own checks take longer than the head's turn takes.
        self._arc(now, free=view.holder in ("", SPEAKER))
        aim = self._within(track.head)
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
    def _arc(self, now: float, *, free: bool) -> None:
        """THE BODY'S ONE RULE: the head serves any bearing within the forward arc; the person
        engaged past it turns the body toward them (``free``: the head is this source's, or
        nobody's). Not during a check, nor while a turn runs (a turn in place kept through
        included), nor after one fell short: until the person's next utterance, or for good
        (this conversation) when it moved the cart and still fell short (stalled, blocked, the
        back-offs spent); nor after a tool's drive or turn until the person's next words
        (:meth:`_hold_body`): the head alone stays on them meanwhile, as far as it reaches."""
        track = self.track
        if (
            track is None
            or not free
            or self.verifying
            or self.turning is not None
            or self.spin is not None
            or self.body_waits
            or self.body_hold is not None
            or track.id in self.gave_up
        ):
            return
        deg = self._body_share(track.pan)
        if deg != 0.0:
            self._start_wheels(deg, track.id, now)

    def _hold_body(self, now: float, why: str) -> None:
        """A tool's drive or turn begins: the gaze's own body turn (the arc's rule) is off until
        the person's next words begun after the drive's end; said once (the head keeps tracking;
        a tool's own ask, :meth:`face_person`, is not held)."""
        self.hold_after_s = None  # the drive runs
        if self.body_hold is not None:
            return
        self.body_hold, self.held_s = why, now
        self.counts["body_holds"] += 1
        self._note("gaze_body_hold", now, f"the body's own turn waits for the person's next words"
                   f" ({why})", held=True, why=why)  # fmt: skip

    def _release_body(self, heard: Heard, now: float) -> None:
        """The person's words begun after a tool's drive ended give the gaze back its body
        turn (an utterance without words, or begun while the drive ran, does not)."""
        after = self.hold_after_s
        if self.body_hold is None or after is None or not heard.words or heard.start_s < after:
            return
        why, held = self.body_hold, round(now - self.held_s, 1)
        self.body_hold, self.hold_after_s = None, None
        self._note("gaze_body_hold", now, f"the person spoke: the body may turn again ({why},"
                   f" {held:.1f} s)", held=False, why=why, held_s=held,
                   text=heard.text[:TEXT_KEPT])  # fmt: skip

    def face_person(self, now: float) -> str | None:
        """A body turn to the person engaged, asked by a tool (approach: nobody in view, a voice
        to go by): the whole bearing in one request (the arc does not apply, nor a turn that fell
        short before: the person asked), the head on them as the body comes round. None when it
        runs (or one already does), else why not."""
        if self.phase == "idle":
            return "no conversation"
        if self.phase == "paused":  # a drive call pauses the gaze from the call, approach's too
            return (
                "the robot's own drive call (this one) has paused the gaze and the wheels"
                if not self.reorient
                else "the gaze waits for the person's next words (a drive dropped their bearing)"
            )
        track = self.track
        if track is None:
            return "nobody engaged: no voice to turn to"
        if self.turning is not None:
            return None
        s = self.settings
        most = math.radians(s.body.max_turn_deg)
        deg = math.degrees(min(max(wrap(track.pan), -most), most))
        if abs(deg) < 1.0:
            return "the person is straight ahead already"
        self.body_waits = False
        self.gave_up.discard(track.id)
        self._start_wheels(deg, track.id, now)
        return None

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

    def _start_wheels(self, deg: float, track: int, now: float) -> None:
        """The body's turn of ``deg`` toward ``track`` (``body_turn``: Nav2's Spin or the
        wheels' executor): the gesture engine's arm sent to rest first (no arm while the body
        turns: :class:`pepin.expression.speaker.ArmRelease`), started off this thread once it
        is there: this returns at once, the word comes back through :attr:`post`."""
        via = self.settings.body_turn
        turn = Turning(next(self._turn_ids), track, deg, via=via)
        self.turning = turn
        self.counts["wheel_turns" if via == "wheels" else "spin_turns"] += 1
        wait = 0.0
        if self.arm is not None:
            try:
                wait = max(0.0, self.arm.release_arm("the body turns toward the person"))
            except Exception:  # the turn never waits on a broken engine
                logger.exception("the gesture engine would not release the arm")
        if wait > 0.0:
            turn.starts_s = now + wait
            self.counts["arm_waits"] += 1
            return
        self._go(turn, now)

    def _go(self, turn: Turning, now: float) -> None:
        """The turn started off this thread (its word comes back through :attr:`post`): Nav2's
        Spin (:meth:`_spin`), or the wheels' on their thread, the readings of whose first
        ``WHEELS_ONSET_S`` are the wheels' jerk, not a voice."""
        turn.starts_s = None
        if turn.via == "spin":
            self._spin(turn, now)
            return
        self.moved = [*self.moved[-(SPOKE_KEPT - 1) :], (now, now + WHEELS_ONSET_S)]

        def done(report: WheelReport) -> None:
            self.post(("done", turn.id, report))

        def moved(turned_deg: float) -> None:
            self.post(("moved", turn.id, turned_deg))

        self.wheels.start(WheelMove(turn_deg=turn.asked_deg), done, moved)

    def _spin(self, turn: Turning, now: float) -> None:
        """The turn as Nav2's Spin: the whole of it in one request to the goal server, made off
        this thread (:attr:`spins`), its word posted as ``("spin", turn id, event)``. Marked
        asked first: the arbiter's drive flag is then this turn's (:meth:`_engage`). Refused
        without a goal server, or without an odometry heading to follow the turn (the people's
        frame and the glide are the odometry's: :meth:`_heading_follows`)."""
        turn.asked_s = now

        def said(event: dict[str, Any]) -> None:
            self.post(("spin", turn.id, event))

        spins, refusal = self.spins, None
        try:
            if spins is None:
                refusal = "no goal server to ask for a spin here"
            elif not self._heading_follows():
                refusal = "no odometry heading here to keep the people's bearings through a turn"
            else:
                spins.spin(turn.asked_deg, said)
        except Exception as error:  # a broken link refuses the turn, never stops the gaze
            logger.exception("the speaker gaze's spin request broke")
            refusal = f"{type(error).__name__}: {error}"
        if refusal is not None:
            said({"event": "error", "detail": refusal})

    def _heading_follows(self) -> bool:
        """Whether the odometry's heading will follow a turn Nav2 drives: the wheels' base link
        opened now (:meth:`pepin.expression.wheels.WheelsExecutor.listen`; otherwise it opens at
        their own first motion, which a spin never makes), or a heading there already."""
        listen = getattr(self.wheels, "listen", None)
        if callable(listen):
            listen()
            return True
        return self.wheels.headings().latest() is not None

    def _spinning(self) -> bool:
        """A spin of the gaze's own asked of the goal server and not ended."""
        turn = self.turning
        return turn is not None and turn.via == "spin" and turn.asked_s is not None

    def _spin_said(self, turn_id: int, event: Mapping[str, Any], now: float) -> None:
        """The goal server's word on a spin of ours: accepted (its start and run), feedback
        (Nav2's count of the turn so far), done or error (its end, as measured on the map, by the
        wheels' rule for a turn short of the ask: refused or nothing moved, asked again at the
        person's next utterance; moved and short, not again for them this conversation; cancelled
        by the goal server for a person's request, ``cancelled_for``, neither: the drive's end
        orients again). The people follow the odometry's heading throughout
        (:meth:`_sync_body`)."""
        turn = self.turning
        if turn is None or turn.id != turn_id:
            self._sync_body(now)
            return
        kind = event.get("event")
        asked = round(turn.asked_deg, 1)
        if kind == "accepted":
            run = event.get("run")
            turn.accepted, turn.run = True, run if isinstance(run, int) else None
            self.moved = [*self.moved[-(SPOKE_KEPT - 1) :], (now, now + WHEELS_ONSET_S)]
            glides = self.wheels.headings().latest() is not None
            waited = None if turn.asked_s is None else round(now - turn.asked_s, 2)
            line = f"the body turns {asked:+.0f} deg: Nav2's Spin, run {turn.run}"
            self._note("gaze_spin", now, line, own=True, run=turn.run, asked_deg=asked,
                       track=turn.track, glides=glides, waited_s=waited,
                       sent_in_ms=event.get("sent_in_ms"))  # fmt: skip
            return
        if kind == "feedback":
            spun = event.get("spun_deg")
            if isinstance(spun, (int, float)):
                turn.turned_deg = float(spun)
            self._sync_body(now)
            return
        if kind not in ("done", "error"):
            return  # a line the gaze does not read (unparsed, a newer server's)
        self.turning = None
        self._turn_ended(now)
        turned = _spin_turned(event, turn.accepted)
        status = event.get("status")
        done = kind == "done" and status == SPIN_SUCCEEDED and not event.get("detail")
        why = "" if done else _spin_why(event)
        way = f"the spin turned {'an unknown part' if turned is None else f'{turned:+.0f}'}"
        reach = "the head looks as far as it reaches"
        replaced = event.get("cancelled_for")
        given: dict[str, Any] = (
            {} if replaced is None else {"gave_way": True, "cancelled_for": replaced}
        )
        if done:
            line = f"{way} of {asked:+.0f} deg"
        elif replaced is not None:  # neither waits nor gives up: the drive's end orients again
            why = f"cancelled for {replaced}" + ("" if status == SPIN_CANCELED else f" ({why})")
            line = f"{way} of {asked:+.0f} deg: {why}"
        elif not turn.accepted:
            self.body_waits = True  # asked again at the person's next utterance
            line = f"no spin ({why}): {reach}"
        else:
            if turned is None or abs(turned) >= 1.0:
                self.gave_up.add(turn.track)  # moved and short: not again for them
            else:
                self.body_waits = True
            line = f"{way} of {asked:+.0f} deg ({why}): {reach}"
        self._note("gaze_spin", now, line, own=True, end=True, run=turn.run, asked_deg=asked,
                   turned_deg=None if turned is None else round(turned, 1),
                   short_deg=event.get("short_deg"), spun_deg=event.get("spun_deg"), done=done,
                   why=why, status=status, seconds=event.get("seconds"), **given)  # fmt: skip

    def _let_wheels_go(self, why: str) -> None:
        """A turn of ours stops (a drive, the close, another person, the head not ours): the
        wheels' are halted; a spin asked of the goal server runs on, its word still taken (the
        goal server cancels it for a tool's drive: its one rule; a cancel stops every goal on
        the board), and one still waiting for the arm is never asked."""
        turn = self.turning
        if turn is None:
            return
        if turn.via == "spin":
            if turn.asked_s is None:
                self.turning = None
            return
        self.turning = None
        self.wheels.halt(why)

    def wheels_event(self, event: WheelEvent, now: float) -> None:
        """The wheels' word on a turn of ours: its progress and its end bring the people round
        with the cart (the heading itself is the odometry's, read in :meth:`_sync_body`); an end
        short of the turn says why, and the body waits (the module's notes)."""
        kind, turn_id, payload = event
        if kind == "aimed":
            self._aimed(turn_id, payload, now)
            return
        if kind == "spin":
            self._spin_said(turn_id, payload, now)
            return
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
        self._turn_ended(now)
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
        and a seek's stops still to come (never off the pursuit's sightings: they stay where the
        camera placed them, relative to the cart at their frames' stamps); the head then
        re-centres under it."""
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
        self.sources.turned(d)
        self.drift += d
        if self.centre is not None:
            self.centre = wrap(self.centre - d)
        if self.stops:
            self.stops = [Aim(stop.pan_rad - d, stop.tilt_rad) for stop in self.stops]
        self._recentre(now)

    def _turn_ended(self, now: float) -> None:
        """A body turn ended: the people's frame follows the odometry to its end, and the head
        re-centres the residue from whatever is known: the first frame exposed after the end
        when the frames showed the person (:meth:`_pursued`; the records exposed before it are
        the turn's), else the odometry's prior now, whether the heading moved since the last
        tick or not."""
        self._sync_body(now)
        if self._by_frames(now):
            self.turn_end_s = self.wall()
        else:
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
        engaged (a seek's stop: where it was), as far as it reaches. While a frame showed the
        person within ``TURN_FRAMES_S``, nothing: the camera keeps them (:meth:`_pursued`, a
        turn's end re-centred by the first frame exposed after it). Else, while the body turns,
        THE GLIDE (the module's notes), the odometry's prior: the aim is where the person will
        be relative to the cart ``glide_ahead_s`` from now at the body's rate, sent without
        waiting, at a speed ceiling of ``glide_gain`` times that rate; after the turn an
        adjustment when the residue is small."""
        track = self.track
        if track is None or self.aim is None or self.phase in ("idle", "paused"):
            return
        if self._by_frames(now):
            return  # the camera keeps them: the frames' own aims (THE FRAME)
        s = self.settings
        if self.phase == "seek" and self.stops is not None:  # a stop of a sweep, held in the room
            pan = self.aim.pan_rad - self.drift
        else:
            pan = track.head.pan_rad
        turned = None if self.turning is None else round(self.turning.turned_deg, 1)
        if self._body_turns():
            rate = self._body_rate()
            target = self._within(Aim(pan - rate * s.glide_ahead_s, self.aim.tilt_rad))
            if target.off(self.aim) <= math.radians(s.aim_min_deg):
                return
            ceiling = max(abs(math.degrees(rate)) * s.glide_gain, GLIDE_FLOOR_DEG_S)
            ease = Ease(min(ceiling, s.turn_speed_deg_s), s.adjust_acc_deg_s2, "glide")
            if self._hold(target, now, BY_ODOMETRY, "glide", ease=ease, turned_deg=turned,
                          rate_deg_s=round(math.degrees(rate), 1)):  # fmt: skip
                self.counts["glides"] += 1
            return
        target = self._within(Aim(pan, self.aim.tilt_rad))
        off = target.off(self.aim)
        if off <= math.radians(s.refine_deg):
            return
        move: Move = "adjust" if off <= math.radians(s.saccade_over_deg) else "turn"
        if self._hold(target, now, "the body turned", move, turned_deg=turned):
            self.counts["recentres"] += 1

    def _body_rate(self) -> float:
        """The cart's yaw rate (rad/s, + left) over the last ``RATE_WINDOW_S`` of the
        odometry's headings; 0 before any."""
        headings = self.wheels.headings()
        latest = headings.latest()
        if latest is None:
            return 0.0
        stamp, heading = latest
        before = headings.at(stamp - RATE_WINDOW_S)
        return 0.0 if before is None else (heading - before) / RATE_WINDOW_S

    def _body_turns(self) -> bool:
        """The cart turns under the head (or is about to: its arm sent to rest): the gaze's own
        turn, or a turn in place kept through (:meth:`keep_person`). The frames keep the person
        meanwhile, the odometry's glide while they show nobody (THE FRAME)."""
        return self.turning is not None or self.spin is not None

    def _by_frames(self, now: float) -> bool:
        """Whether a frame showed the person engaged within ``TURN_FRAMES_S``: while the body
        turns the frames then aim the head, and the odometry neither aims it nor seeks them."""
        track = self.track
        seen = None if track is None else track.seen_s
        return seen is not None and now - seen <= TURN_FRAMES_S + TICK_EPS_S

    # ---- the head ----------------------------------------------------------------------------
    def _within(self, aim: Aim) -> Aim:
        """``aim`` inside the reach: a pan past it goes to the end nearer the head's own pan
        (our last aim), never across the front (:meth:`pepin.gaze.Reach.clamp`)."""
        here = self.aim.pan_rad if self.aim is not None else None
        return self.reach.clamp(aim, from_pan=here)

    def _hold(
        self,
        aim: Aim,
        now: float,
        why: str,
        move: Move,
        *,
        start: Aim | None = None,
        ease: Ease | None = None,
        flow: bool = False,
        **fields: Any,
    ) -> bool:
        """Ask the arbiter to hold the head at ``aim`` (replacing this source's last aim), a
        turn or an adjustment (from ``start`` when no aim of ours holds the head yet; ``ease``
        given: on that profile); whether the request lives (done, or queued behind a better
        look). WHILE THE BODY TURNS, and for a person tracked (``flow``: the pursuit), the aim
        is sent without waiting (:meth:`_send`): the head is taken as ours, the answer comes
        through the inbox."""
        s = self.settings
        ease = s.ease(move) if ease is None else ease
        speed: Speed = "saccade" if move == "turn" else "slow"
        before = self.aim if self.aim is not None else start
        if ease is not None and before is not None:
            fields["move_s"] = round(ease.seconds(math.degrees(aim.off(before))), 2)
        self.counts["aims"] += 1
        self.next_renew = now + s.renew_s
        if flow or self._body_turns():
            status = self._send(aim, speed, ease)
            self._note("gaze_reaim", now, None, why=why, pan_deg=_deg(aim.pan_rad),
                       tilt_deg=_deg(aim.tilt_rad), speed=speed if ease is None else ease.name,
                       status=status, **fields)  # fmt: skip
            if status == "sent":
                self._tilt_moved(before, aim, now)
                self.aim, self.drift, self.ours = aim, 0.0, True
            return status == "sent"
        answer = self.door.aim(aim, speed=speed, ttl_s=s.ttl_s, ease=ease)
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
            self._tilt_moved(before, aim, now)
            self.aim, self.drift = aim, 0.0
            self.ours = answer.status == "done"
            if not self.ours:
                self.report(f"the head waits: {answer.status} {answer.reason}".rstrip())
                self._let_wheels_go("another look holds the head")
            return True
        self._lost(now, f"{answer.status}: {answer.reason}")
        return False

    def _tilt_moved(self, before: Aim | None, aim: Aim, now: float) -> None:
        """The tilt's way remembered at every aim that moves it: a step the other way than the
        last, or one after the tilt stood still for ``TILT_TURN_S``, is a turn of its direction
        (the arbiter's clip rule counts a motion's start as one)."""
        if before is None or abs(aim.tilt_rad - before.tilt_rad) < TURN_EPS_RAD:
            return
        way = way_of(aim.tilt_rad - before.tilt_rad)
        last, turned, moved = self._tilt_way
        if way != last or now - moved > TILT_TURN_S:
            turned = now
        self._tilt_way = (way, turned, now)

    def _tilt_kept(self, tilt: float, now: float) -> float:
        """``tilt``, unless it would turn the tilt's direction back sooner than ``TILT_TURN_S``
        after its last turn (the arbiter's clip_max_hz: the mast rings at 5.3 Hz): then where
        the last aim put it."""
        aim = self.aim
        if aim is None or abs(tilt - aim.tilt_rad) < TURN_EPS_RAD:
            return tilt
        last, turned, _ = self._tilt_way
        if way_of(tilt - aim.tilt_rad) != last and last != 0 and now - turned < TILT_TURN_S:
            self.counts["tilt_held"] += 1
            return aim.tilt_rad
        return tilt

    def _send(self, aim: Aim, speed: Speed, ease: Ease | None) -> str:
        """The door's call for ``aim`` made off this thread (:attr:`spawn`), its answer posted
        as ``("aimed", id, answer)``; "sent", or "unsent" with ``AIMS_IN_FLIGHT`` still unanswered
        (the door is stuck: their answers decide)."""
        with self._flight_lock:
            if self._in_flight >= AIMS_IN_FLIGHT:
                self.counts["aims_unsent"] += 1
                return "unsent"
            self._in_flight += 1
        aim_id = next(self._aim_ids)
        self.aim_id = aim_id
        ttl_s = self.settings.ttl_s

        def call() -> None:
            try:
                answer = self.door.aim(aim, speed=speed, ttl_s=ttl_s, ease=ease)
            except Exception as error:  # a door that throws, off the machine's thread
                logger.exception("the speaker gaze's aim broke")
                answer = GazeAnswer("denied", f"{type(error).__name__}: {error}")
            finally:
                with self._flight_lock:
                    self._in_flight -= 1
            self.post(("aimed", aim_id, answer))

        self.spawn(call)
        return "sent"

    def _aimed(self, aim_id: int, answer: GazeAnswer, now: float) -> None:
        """The arbiter's answer to an aim sent without waiting: done and preempted (by our own
        newer aim) say the head is ours; queued, denied or expired on the newest aim say it is
        not (an older aim's late word changes nothing)."""
        if answer.status in ("done", "preempted") or aim_id != self.aim_id:
            return
        if self.phase in ("idle", "paused") or self.aim is None:
            return
        if answer.status == "queued":
            self.ours = False
            self.report(f"the head waits: {answer.status} {answer.reason}".rstrip())
            self._let_wheels_go("another look holds the head")
            return
        self._lost(now, f"{answer.status}: {answer.reason}")

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
        """One frame while the head is ours and carries no gesture (still, unless the person is
        tracked: :meth:`_pursue`); the person engaged centred (refined after a turn, pursued
        while tracking)."""
        try:
            self._look(now)
        finally:  # at the rate of the phase the frame left the machine in
            if self.next_frame is not None and self.next_frame <= now + TICK_EPS_S:
                s = self.settings
                if self.phase != "seek" or self._body_turns():  # a body turn: every record
                    self.next_frame = now + (
                        s.speaking_every_s if self._robot_speaking() else s.track_every_s
                    )
                else:  # a check's next frame as soon as the head is still again
                    self.next_frame = now + (BLIND_RETRY_S if self.verifying else s.seek_every_s)

    def _look(self, now: float) -> None:
        s = self.settings
        if self.finder is None:
            return
        if self.phase == "track" or self._body_turns():  # posed at the exposure, never waited
            self._pursue(now)
            return
        view = self._state()
        if view is None or view.head is None or self._skipped(view, now):
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

    def _skipped(self, view: HeadView, now: float) -> bool:
        """Whether a frame is no look now: the head is not ours, a drive runs, or a gesture
        plays (the idle layer's breath aside: a gesture's frame is tried again soon); counted
        by its cause."""
        gesture = bool(view.overlay) and view.overlay != IDLE_LOOK
        self.ours = view.holder == SPEAKER
        if self.ours and not view.driving and not gesture:
            return False
        skipped = "holder" if not self.ours else "driving" if view.driving else "gesture"
        self.counts["frames_skipped"] += 1
        self.counts[f"frames_skipped_{skipped}"] += 1
        if gesture and self.ours and not view.driving:
            self.next_frame = now + BLIND_RETRY_S  # a gesture ends soon: look right after
        return True

    def _pursue(self, now: float) -> None:
        """A frame while the person engaged is tracked, or while the body turns (any phase):
        the finder's next record whether the head stands or moves (no wait for a still head),
        the arbiter read after it (so the poses kept cover the frame's exposure), its people
        placed with the head's pose AT THE FRAME'S STAMP (:class:`pepin.expression.pursuit.
        HeadPoses`; a frame without a stamp, or outside the poses kept, with the pose now,
        counted ``frames_unposed``). The person engaged is matched where the camera's fit puts
        them at the frame's stamp (THE MATCH, :meth:`_expected`), else where their track is.
        While the body turns: a person placed only by a voice is matched only in a frame that
        looks toward them (within ``seek_pan_deg``; one exposed while the head still swings
        there is counted ``frames_away``); without the camera's fit and no frame showing them
        for ``TURN_FRAMES_S`` they are sought as a seek seeks (the nearest person within its
        reach of where the odometry put them: it may have slipped, a frame looking there,
        never someone else the frames show as themselves), and a frame without them spends
        nothing outside tracking (the glide goes on). A look that got no record is no frame: no
        miss counts toward ``face_lost_frames``."""
        s = self.settings
        if self.finder is None:
            return
        frame = self.finder.look(not_before=None)
        view = self._state()
        if view is None or view.head is None or self._skipped(view, now):
            return
        if frame is None:
            self.counts["frames_none"] += 1
            return
        self.counts["frames"] += 1
        turning = self._body_turns()
        if turning:
            self.counts["frames_turning"] += 1
        head = self.poses.at(frame.stamp) if frame.stamp > 0.0 else None
        self.counts["frames_posed" if head is not None else "frames_unposed"] += 1
        looked = head if head is not None else view.head
        people = sightings(frame, looked, min_face_px=s.seek_min_face_px,
                           min_body_score=s.min_body_score)  # fmt: skip
        track = self.track
        expected = self._expected(track, frame.stamp)
        placed: tuple[float, Aim] | None = None
        if track is not None and expected is not None:  # THE MATCH: where the camera puts them
            placed, shift = (track.pan, track.head), wrap(expected - track.head.pan_rad)
            track.pan = wrap(track.pan + shift)
            track.head = Aim(wrap(track.head.pan_rad + shift), track.head.tilt_rad)
        # A frame looks where the person is (within a seek's stop of their place) or not: one
        # exposed while the head still swings there shows whoever is on the way
        toward = track is not None and abs(wrap(looked.pan_rad - track.head.pan_rad)) <= (
            math.radians(s.seek_pan_deg)
        )
        if turning and track is not None and track.source == "voice" and not toward:
            self.counts["frames_away"] += 1  # a voice's place takes nobody by the match alone
            return
        sought = turning and expected is None and toward and not self._by_frames(now)
        seeking = track if sought else None  # the seek's reach is the odometry's slip
        seek_rad = math.radians(s.seek_pan_deg + s.match_deg)
        keep = self._kept()
        if seeking is not None and seeking.seen_s is not None:
            # Someone else the frames show as themselves (a track that was there when the
            # person engaged was last seen, seen since) is never taken for them: the seek's
            # reach is the odometry's slip, not a licence to swap people
            last = seeking.seen_s
            keep |= {t.id for t in self.tracks.tracks
                     if t is not seeking and t.created_s <= last
                     and t.fresh(now, s.seen_fresh_s)}  # fmt: skip
        seen = self.tracks.seen(people, now, seeking=seeking, seek_rad=seek_rad, keep=keep)
        mine = seen.get(track.id) if track is not None else None
        if track is not None and mine is None and placed is not None:
            track.pan, track.head = placed  # not in this frame: the odometry's bearing stands
        if track is None or mine is None:
            self._missed(now, looked=self.phase == "track")
            return
        self._seen(track, mine, frame, now)

    def _expected(self, track: Track | None, stamp: float) -> float | None:
        """THE MATCH (:mod:`pepin.expression.pursuit`): the head pan the camera's fit puts the
        person engaged at ``stamp`` (a frame's exposure), when the pursuit is theirs, of the
        source that placed them last and of this turn state (a fit across a turn's start or end
        is neither): through a gap while the body turns, ``window_s`` past the newest sighting
        otherwise. None without such a fit: the track's bearing (the odometry's) stands."""
        pursuit = self.pursuit
        if (
            track is None
            or pursuit is None
            or stamp <= 0.0
            or pursuit.track != track.id
            or pursuit.source != track.source
            or pursuit.turning != self._body_turns()
        ):
            return None
        return pursuit.expected(stamp, math.inf if pursuit.turning else pursuit.window_s)

    def _seen(self, track: Track, mine: Sighting, frame: FaceFrame | None, now: float) -> None:
        """The person engaged is in the frame: centred, tracked (pursued: :meth:`_follow`),
        smiled at once."""
        s = self.settings
        self.misses = 0
        self.counts["faces" if mine.face is not None else "bodies"] += 1
        if self.on_face is not None and frame is not None and mine.face is not None:
            try:
                self.on_face(frame, mine.face)
            except Exception:  # a listener never stops the gaze
                logger.exception("the speaker's face listener broke")
        self._sighted(track, mine, frame)
        tracking = self.phase == "track"
        px = round(mine.face.width_px) if mine.face is not None else None
        if tracking or self._body_turns():  # a body turn's aims are the frames' (THE FRAME)
            stamp = frame.stamp if frame is not None else 0.0
            if not self._pursued(track, mine, now, px, stamp):
                return
        else:
            target = self._within(Aim(track.head.pan_rad, track.head.tilt_rad))
            off = target.off(self.aim) if self.aim is not None else math.inf
            if off > math.radians(max(s.refine_deg, s.aim_min_deg)):
                move: Move = "adjust" if off <= math.radians(s.saccade_over_deg) else "turn"
                why = "onto the face" if mine.face is not None else "onto the person"
                if not self._hold(target, now, why, move, px=px, lead_deg=0.0,
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
            if self.first_face_s is None:
                self.first_face_s = now
            self._smile()

    def _sighted(self, track: Track, mine: Sighting, frame: FaceFrame | None) -> None:
        """The pursuit's sighting: the person engaged's aim at their frame's exposure stamp,
        when the sighting placed the track (a box while their face is fresh does not) and the
        frame carries a stamp; another person, another source, or the start or the end of a
        body turn starts a new pursuit (:mod:`pepin.expression.pursuit`)."""
        if frame is None or frame.stamp <= 0.0 or track.source != mine.source:
            return
        pursuit, turning = self.pursuit, self._body_turns()
        if not turning and self.turn_end_s is not None and frame.stamp < self.turn_end_s:
            return  # exposed while the body turned, read after: the turn's motion is in it
        if (
            pursuit is None
            or pursuit.track != track.id
            or pursuit.source != mine.source
            or pursuit.turning != turning
        ):
            pursuit = self.pursuit = Pursuit(track.id, mine.source, turning)
        pursuit.add(frame.stamp, mine.head.pan_rad)

    def _pursued(
        self, track: Track, mine: Sighting, now: float, px: int | None, stamp: float
    ) -> bool:
        """THE PURSUIT of the person tracked: the head aimed where they will be ``lead_s`` from
        now (their speed fitted over their sightings at the frames' stamps, at most
        ``lead_max_deg`` past the last one), sent without waiting so the frames go on while it
        moves. A person who moves an aim's worth (``aim_min_deg``) within ``lead_s`` is
        followed at every frame, a glide at ``glide_gain`` times their speed (at least
        ``GLIDE_FLOOR_DEG_S``); a person standing, past the dead band only (an adjustment). The
        tilt follows past ``refine_deg`` and never turns back sooner than the arbiter's clip
        rule (:meth:`_tilt_kept`). WHILE THE BODY TURNS (THE FRAME) the same aim by the frames
        alone: the pan this sighting's (the head's pose at the exposure and the angle through
        the lens, no odometry), followed past ``aim_min_deg`` at every frame (the cart's turn
        moves the person's bearing), at a ceiling of ``glide_gain`` times the faster of the
        fitted speed and the odometry's rate (a ceiling, never a place), the lead at most what
        that rate turns over the lead's span (the wheels stopped: none, before the turn's end is
        told). The first frame exposed (``stamp``, 0 unknown) after a turn's end re-centres its
        residue past ``refine_deg``. Whether the aim lives, or none was due."""
        s = self.settings
        turning = self._body_turns()
        aim, pursuit = self.aim, self.pursuit
        if pursuit is not None and pursuit.turning != turning:
            pursuit = None  # a fit across a turn's start or end is neither motion
        end = self.turn_end_s
        settle = not turning and end is not None and (stamp <= 0.0 or stamp >= end)
        if settle:
            self.turn_end_s = None  # the turn's residue, once
        speed = pursuit.speed() if pursuit is not None else 0.0
        wall = self.wall()
        cap = math.radians(s.lead_max_deg)
        lead = pursuit.ahead(wall + s.lead_s, cap) if pursuit is not None else 0.0
        rate = self._body_rate() if turning else 0.0
        last_s = pursuit.last_s if pursuit is not None else None
        if turning and last_s is not None and self.wheels.headings().latest() is not None:
            # The lead never runs past what the wheels turn over its span: the camera's fit lags
            # the cart's stop by half its window, the odometry's rate does not (a ceiling only:
            # on a slipping floor it reads more than the cart turns)
            reach = abs(rate) * max(0.0, wall + s.lead_s - last_s)
            lead = max(-reach, min(reach, lead))
        moving = turning or abs(speed) * s.lead_s >= math.radians(s.aim_min_deg)
        tilt = track.head.tilt_rad
        if aim is not None and abs(tilt - aim.tilt_rad) <= math.radians(s.refine_deg):
            tilt = aim.tilt_rad  # the face's own jitter moves no tilt
        pan = mine.head.pan_rad if turning else track.head.pan_rad
        target = self._within(Aim(pan + lead, tilt))
        target = Aim(target.pan_rad, self._tilt_kept(target.tilt_rad, now))
        off = target.off(aim) if aim is not None else math.inf
        band = (
            s.aim_min_deg
            if moving
            else s.refine_deg
            if settle
            else max(s.deadband_deg, s.aim_min_deg)
        )
        if off <= math.radians(band):
            return True
        ease: Ease | None = None
        move: Move = "adjust" if off <= math.radians(s.saccade_over_deg) else "turn"
        if moving:
            move = "glide"
            fastest = max(abs(speed), abs(rate))
            ceiling = max(math.degrees(fastest) * s.glide_gain, GLIDE_FLOOR_DEG_S)
            ease = Ease(min(ceiling, s.turn_speed_deg_s), s.adjust_acc_deg_s2, "glide")
        age = None if last_s is None else round(wall - last_s, 2)
        why = (
            BY_FRAMES
            if turning
            else "the person moves"
            if moving
            else SETTLED
            if settle
            else "the person moved"
        )
        body: dict[str, Any] = {}
        if turning:  # beside the glide's rows: Nav2's count and the odometry's rate
            turned = None if self.turning is None else round(self.turning.turned_deg, 1)
            body = {"turned_deg": turned, "rate_deg_s": round(math.degrees(rate), 1)}
        if not self._hold(target, now, why, move, ease=ease, flow=True, px=px,
                          lead_deg=_deg(lead), speed_deg_s=_deg(speed), age_s=age,
                          source=mine.source, track=track.id, **body):  # fmt: skip
            return False
        self.counts["reaims"] += 1
        if turning:
            self.counts["turn_reaims"] += 1
        elif moving:
            self.counts["pursuits"] += 1
        return True

    def _missed(self, now: float, *, looked: bool) -> None:
        """A frame without the person engaged (``looked``: one was taken; a frame not taken
        spends nothing): a check goes back, a seek reads the voice again, then takes its next
        stop, then gives up; a person tracked is lost after ``face_lost_frames`` such frames
        (records the finder gave: a look that got none is no frame) and ``face_lost_s``."""
        if not looked:
            return
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
                stop = self._within(self.stops.pop(0))
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
        self.phase, self.centre, self.pursuit = "voice", track.pan, None
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
            self._hold(self._within(Aim(voice, aim.tilt_rad)), now, "to the voice",
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
            if self._hold(self._within(fallback.head), now, "back to the face", "turn",
                          track=fallback.id):  # fmt: skip
                self.phase = "track"
            return
        self.phase = "voice"
        self._note("gaze_lost", now, "nobody seen: the head stays on the voice", what="seek",
                   track=self.engaged)  # fmt: skip
        track = self.track
        if track is not None and self.aim is not None:
            centre = self.centre if self.centre is not None else track.pan
            back = self._within(Aim(centre, track.head.tilt_rad))
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
        """The arbiter's state; the head's pose in it kept at its encoder reading's own stamp
        (the board's clock, the frames' too: the poses a frame is placed by), or, from an
        arbiter that sends no ``head_stamp``, at this Mac's wall clock of the read (logged once)."""
        self.counts["states"] += 1
        view = self.door.state()
        if view is not None and view.head is not None:
            stamp = view.head_stamp
            if stamp is None:
                stamp = self.wall()
                if not self.unstamped_said:
                    self.unstamped_said = True
                    logger.warning(
                        "the arbiter's state carries no head_stamp (a gaze node started before"
                        " it): the head's poses are dated at their read by this Mac's clock"
                        " (ros/laptop.sh kick gaze)"
                    )
            self.poses.add(stamp, view.head)
        return view

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


def _on_a_thread(call: Callable[[], None]) -> None:
    """``call`` on a daemon thread of its own (a door's exchange while the machine goes on)."""
    threading.Thread(target=call, name="speaker-aim", daemon=True).start()


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
            chip=voice.chip,
        )
        if voice.robot:
            fields["robot"] = voice.robot
    return fields


def _deg(rad: float) -> float:
    """Degrees, one decimal, for an event."""
    return round(math.degrees(rad), 1)


def way_of(step: float) -> int:
    """A step's direction: 1 up the scale, -1 down."""
    return 1 if step > 0.0 else -1


def _spin_turned(event: Mapping[str, Any], accepted: bool) -> float | None:
    """How far a spin turned (degrees, + left): the goal server's measure on the map, else
    Nav2's own count; nothing when it was never taken; None when unknown (the link lost)."""
    for key in ("turned_deg", "spun_deg"):
        value = event.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None if accepted else 0.0


def _spin_why(event: Mapping[str, Any]) -> str:
    """Why a spin ended short of its ask, in the goal server's words: its refusal, or Nav2's
    status and the detail."""
    if event.get("event") == "error":
        return str(event.get("detail") or "refused")
    status = event.get("status")
    word = SPIN_STATUS.get(status, f"status {status}") if isinstance(status, int) else "no status"
    detail = event.get("detail")
    return f"Nav2 {word}" + (f": {detail}" if detail else "")


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
        self._arm: ArmRelease | None = None
        self._series: VoiceSeries | None = None
        self._view: Attention = NO_ATTENTION  # the machine's latest snapshot (any thread reads)
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

    @property
    def arm(self) -> ArmRelease | None:
        """The gesture engine a body turn asks for the arm (None: nobody)."""
        return self._arm

    @arm.setter
    def arm(self, arm: ArmRelease | None) -> None:
        self._arm = arm
        if self.machine is not None:
            self.machine.arm = arm

    @property
    def series(self) -> VoiceSeries | None:
        """The ears' speech probabilities the machine tags the array's readings with."""
        return self._series

    @series.setter
    def series(self, series: VoiceSeries | None) -> None:
        self._series = series
        machine = self.machine
        if machine is not None:
            machine.series = series

    def attention(self) -> Attention:
        """The attention engine's latest snapshot (:class:`pepin.expression.speaker.
        AttentionView`); :data:`pepin.expression.speaker.NO_ATTENTION` while it is off."""
        return self._view if self._on else NO_ATTENTION

    def face_person(self, why: str) -> str | None:
        """Turn the body to the person engaged now (:meth:`SpeakerGaze.face_person`), from any
        thread:
        None when the turn runs, else why not (the gaze off, or the machine's own reason)."""
        if not self._on:
            return "the speaker gaze is off"
        reply: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._post("face", reply, why)
        try:
            return reply.get(timeout=FACE_ANSWER_S)
        except queue.Empty:
            return "no answer from the speaker gaze"

    def keep_person(self, why: str) -> Attention | str:
        """A turn in place of the body's begins (the turn tool's): the person engaged is kept
        through it (:meth:`SpeakerGaze.keep_person`), from any thread; the snapshot right after
        (their bearing as the turn starts), else why not (the gaze off, or the machine's own
        reason)."""
        if not self._on:
            return "the speaker gaze is off"
        reply: queue.Queue[Attention | str] = queue.Queue(maxsize=1)
        self._post("keep", reply, why)
        try:
            return reply.get(timeout=FACE_ANSWER_S)
        except queue.Empty:
            return "no answer from the speaker gaze"

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

    def part_started(self, turn: int, index: int, stream_s: float) -> None:
        """Nothing."""

    def text_ahead(self, turn: int, sentence: int, text: str) -> None:
        """Nothing."""

    def reply_cut(self, turn: int) -> None:
        """The robot stopped speaking."""
        self._post("speaking", False)

    def reply_done(self, turn: int) -> None:
        """The robot stopped speaking."""
        self._post("speaking", False)

    def moment(self, name: str, text: str = "") -> None:
        """Nothing: a moment is the gestures'."""

    def drive(self, active: bool, ran: bool = True) -> None:
        """A drive began or ended (``ran``: whether it set the body moving)."""
        self._post("drive", active, ran)

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
            self._view = self._guarded(lambda m: m.attention(self._clock())) or NO_ATTENTION

    def _switched_on(self, settings: SpeakerSettings) -> None:
        try:
            if self.machine is None:
                self.machine = self._build(settings)
                if self.machine.events is None:
                    self.machine.events = self.record
                self.machine.post = self._wheels_said
            self.machine.arm = self._arm
            if self._series is not None:  # the ears' (else what the machine was built with)
                self.machine.series = self._series
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
            "drive": lambda m: m.drive(args[0], now, at=at, ran=args[1]),
            "closed": lambda m: m.closed(args[0], now),
            "wheels": lambda m: m.wheels_event(args[0], now),
            "face": lambda m: args[0].put(m.face_person(now)),
            "keep": lambda m: args[0].put(m.keep_person(now, args[1]) or m.attention(now)),
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
    mac: MacSource | None = None,
) -> SpeakerGazeWorker:
    """The speaker gaze on the robot: the arbiter's door on this Mac, the board's camera and
    head server at ``host``, the array's directions from ``doa`` (the cascade's audio link) or
    from ``mac`` (the Mac's own, :class:`pepin.audio_doa.MicDoa`, fed by the link's raw frames)
    as ``speaker.doa.direction`` says (:class:`DoaChoice`: the chip's while no raw frames
    come), ``wheels`` (the cascade's one :class:`pepin.expression.wheels.WheelsExecutor`; none:
    the head alone), the goal server's spin on this Mac (:class:`GoalSpins`: the body's turn
    under ``speaker.body_turn`` spin); built, and the finder's models loaded, the first time its
    switch is on (config/
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
        choice = DoaChoice(settings.direction, doa, mac, report=say)
        choice.configure(settings.direction, voice_p_min=settings.mac_voice_p_min,
                         peak_ratio=settings.mac_peak_ratio,
                         reading_s=settings.mac_reading_s)  # fmt: skip
        return SpeakerGaze(
            settings,
            door=SpeakerDoor(aim_wait_s=settings.aim_wait_s),
            doa=choice,
            finder=looks,
            wheels=wheels if wheels is not None else NoWheels(),
            face=HeadClient(host, source=FACE_SOURCE).start() if settings.smile else None,
            reach=Reach.of(NeckConfig.from_json(config_file("neck.json"))),
            on_face=on_face,
            spins=GoalSpins(),
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
        goal: GoalSpins,
        mic: MicDoa,
        choice: DoaChoice,
    ) -> None:
        spins: Spins = goal
        observer: ConversationObserver = worker
        flag: LiveFlag = worker
        gaze_door: GazeDoor = door
        faces: FaceFinder = finder
        records: FaceFinder = service
        either: FaceFinder = switched
        wheels: Wheels = turn
        doa: DoaSource = link
        mac: MacSource = mic
        chosen: DoaSource = choice
        del observer, flag, gaze_door, faces, records, either, wheels, doa, spins, mac, chosen
