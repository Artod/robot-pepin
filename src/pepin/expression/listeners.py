"""The gesture engine's own trigger source: the rules that make the body alive between and under
the brain's marks (gesticulation DESIGN.md §6.2, §6.3, §9), each a
:class:`pepin.expression.schedule.Request` posted into the engine's one scheduler, under its
priority table (moments at ``MOMENT`` in :data:`pepin.expression.conversation.MOMENTS`' order,
the voice's accents at ``PULSE``, the melody and the held arm's sway at ``IDLE``).

ONE BODY: every event below is an ensemble of every part that can move, the book's own
compositions (config/animation.json, ``callable`` false) asked for by name with the parts found
free for it (:attr:`pepin.expression.schedule.Request.parts`): the head where its ledger has
room, the arm where a unit holds it up, the wheels where the cart stands (no drive, no body turn
of the gaze's, no motion of its own still settling) and each part where its lead covers its
preparation. The book draws the rest; the dice of the rules are this module's knobs.

:class:`Alive` is a conversation observer with a thread of its own (the cascade's calls only
post to it) that drives the :class:`pepin.expression.prosody.ProsodyListener` (the voice's
nuclei and melody, found in its audio before the speaker sounds it) and keeps the timers.

While the robot speaks (``pulse``):

* the MELODY: the part's pitch contour as a still tilt of the head (up with the pitch,
  ``melody_deg_per_st`` a semitone round the part's median, within +-``melody_max_deg``), keys
  at the contour's turns at least ``melody_key_gap_s`` apart, on the idle source (it never
  blinds the head while within the arbiter's ``clip_still_deg``), in overlays of at most
  ``SEGMENT_S``, sent again after a nod of its own; the idle layer's keys glide (each key's
  speed and ramp the gentlest that land it on time: :func:`pepin.expression.parts.clip_look`);
* an ACCENT on the first nucleus of a part with z >= ``beat_z``, or on its most prominent one
  when the part's whole audio is known before its ensemble must go; at most one a part, by its
  strength: under ``strong_z`` ``pulse_weak`` (the nod; with the arm up its wrist's beat,
  ``wrist_beat_p`` of them; the cart's 0.5 cm sway on ``wheel_accent_share`` of them, steady),
  from ``strong_z`` ``pulse_strong`` (the nod, the wrist's beat with the arm up, the cart's 1 cm
  rock), the reply's strongest known (once a reply, from ``strong_z``) ``pulse_peak`` (all three,
  bigger); the head's nod ``beat_gap_s`` after any head stroke, ``beats_per_minute``, never with
  less lead than its own preparation (``beat_lead_min_s``), never within ``mark_gap_s`` of a
  marker's head stroke (sent, or still waiting for its word); no accent in a part that carries
  a marker (the brain's gesture is that sentence's accent); the decision waits for the lead of
  the slowest part it wants (the wheels' rock: ~1 s at speed 0.4);
* the LAUGH'S SOUND: a run of ``laugh_run`` accents of a part ``laugh_gap_s`` apart is the voice
  laughing: each of its accents asks for ``pulse_laugh`` (the cart's rock, the happy face, a bob
  of the head), each part where it is free (the cart takes the next accent once its rock is
  over); no text is read;
* the ARM UP: a reply with ``arm_up_min_sentences`` parts with words at the voice (a sentence,
  or a clause a style cut; their TEXT, before their sound) raises the arm once to the book's
  ``ready`` and holds it there as an open unit while the reply goes on (``POSTURE``): sent at
  the stream second being heard (the reply's first sound when it has not begun), never in quiet
  or with less than ``POSTURE_MIN_S`` of the reply's speech known after it is up; its hold
  reaches ``POSTURE_TAIL_S`` past the reply's audio known (at most config/arm.json's
  ``hold_max_s``, the board's own limit: past it the board rests the arm by itself) and is sent
  again before it lapses while more audio comes; a mark's stroke plans from it and keeps its
  hold, the arm going back to ``ready`` after it; the accents' beats ride it; the engine rests
  it when the person speaks, at the reply's end or cut and for the body's turn; a part in a
  quiet style rests it too;
* the HELD ARM'S BREATH: while a unit holds the arm up and nothing moves it (``SWAY_GAP_S``
  after its last motion, with its hold outlasting the sway), ``arm_sway``, a slow sway of the
  wrist over the held pose (the head's share of the breath is the idle layer's loop);
* the QUESTION TILT: a part ending in "?" lifts and cocks the head at its last nucleus, held
  until half a second after its audio, once a reply;
* the LONG-REPLY AVERSION: in a reply with ``aversion_reply_s`` of audio known, a look aside or
  down at an inner pause, back ``end_gaze_s`` before the audio known ends, once a reply.

Between turns and while the person speaks (``moments``):

* the THINKING GLANCE: the turn closed and no reply audio after ``think_glance_s``: with
  ``think_p``, a look up, down or aside (``think_dirs``; never the same way twice running),
  held until ``think_back_s`` into the reply as this conversation's waits predict it, or
  ended by the engine (:meth:`pepin.expression.animator.GestureEngine.release`) that long into
  the reply's first sound when the reply comes sooner (the face ``thinking`` half the time);
  never when the audio came first;
* the LISTENER NOD on the ears' ``listener_pause``: after ``listener_min_speech_s`` of their
  speech, ``listener_gap_s`` apart, never while their last words were a question (their words
  so far when the pause brings them, else their last utterance): ``listen_nod``, its nod with
  ``listener_nod_p``, the cart's 0.5 cm rock with ``nod_rock_p``;
* the ACKNOWLEDGEMENT: a statement of ``ack_min_s`` or more (not a question: ``question``) gets
  ``ack_nod`` ``predict_delay_s`` after its close, once a turn: its nod with ``ack_nod_p``, the
  cart's 0.5 cm rock with ``nod_rock_p``;
* GREETING and FAREWELL DUE: the person's words (``greeting``; the voice loop's goodbye rule)
  play the book's ``greeting`` or ``farewell`` (or what its ``moments`` map ``greeting_due``,
  ``farewell_due`` to) ``predict_delay_s`` after the close, in the brain's wait, under the
  engine's caps; while the body turns toward the person it waits for the turn's end; a later
  marker of it is absorbed by its cooldown;
* FACE FOUND: the attention engine's first clear face of the conversation brings a smile;
* the LISTENING SHIFT (``listen_shift_deg`` 0: off): a still shift of the head every
  ``listen_shift_s`` while the person speaks;
* every discrete head stroke at least ``head_gap_s`` from the one before (a held move's return
  counts), so head moves never pile up at a turn's edge.

The arm and the wheels move together (one body): neither waits for the other; the gaze's body
turn is the one thing that outranks the pulse's wheels (and takes the arm to rest first).

The QUIET rule: a reply in a quiet style, the hours ``quiet.hours`` or the person's voice under
``quiet.peak_db`` leave the melody and the face only (a person's ask still plays whole; the
engine's own renderings, the markers', lose every part but the face through :meth:`Alive.scene`).

The breath while the robot is silent is the book's idle layer (config/animation.json ``idle``),
the engine's. Everything is logged in the conversation's events.jsonl: ``pulse`` per part,
``moment`` per decision, ``quiet`` on a change, ``alive`` at the close with the counts a minute.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import math
import random
import re
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pepin.expression.animation import Clip, Event, Layer, PartName, Scene, Track
from pepin.expression.book import span
from pepin.expression.conversation import (
    ACKNOWLEDGED,
    FACE_FOUND,
    FAREWELL_DUE,
    GREETING_DUE,
    LISTENER_PAUSE,
    LONG_REPLY,
    QUESTION_END,
    SLOW_TOOL,
    THINK_WAIT,
    EventSink,
    Heard,
    moment_rank,
)
from pepin.expression.markers import Part
from pepin.expression.parts import ArmPart
from pepin.expression.prosody import PartNews, ProsodyListener, VoiceStats
from pepin.expression.schedule import (
    HOME,
    POSTURE,
    SAME_POSE_DEG,
    AtStream,
    AtTime,
    Nucleus,
    Priority,
    Request,
    Reservation,
    pose_clip,
)
from pepin.expression.speaker import Attention
from pepin.gaze import GazeSettings
from pepin.voice_live.wake import is_goodbye

if TYPE_CHECKING:
    from pepin.expression.animator import GestureEngine

logger = logging.getLogger(__name__)

STROKE_BIAS_S = -0.05  # a nod's bottom this long before its nucleus (one constant, DESIGN §7.7)
NOD_PERIOD_S = (0.4, 0.45)  # the book's head_nod: down and up, the arbiter's 2.5 Hz at most
SEGMENT_S = 2.0  # a melody overlay's longest stretch: a gesture that cuts one costs this at most
MELODY_BATCH_S = 1.0  # a part still being made sends its melody this much at a time
MELODY_AHEAD_S = 0.4  # or sooner, when its first key is due within this
RESUME_S = 0.05  # the melody comes back this long after a nod of its own ends
QUESTION_PAN_DEG, QUESTION_TILT_DEG = 3.0, 2.0  # the question tilt: a cock and a lift
QUESTION_OUT_S, QUESTION_HOLD_S = 0.25, 0.5  # its way there, its hold past the part's audio
OUT_S, BACK_S = 0.35, 0.4  # a held look's way out and back (the glance, the aversion)
WAIT_PRIOR_S = 1.95  # the person's close to the reply's first sound, p50 (MEASURED 10-11/12)
WAITS_KEPT = 10  # the conversation's last waits the glance's hold is predicted from
THINK_MAX_S = 3.5  # a thinking glance never holds longer
THINK_FACE_P = 0.5  # the thinking face with the glance
PAUSE_GAP_S = 0.35  # two nuclei this far apart have a pause between them
DEFER_MAX_S = 3.0  # a greeting waits for the body's turn at most this long
DEFER_STEP_S = 0.1
POLL_S = 0.2  # the head's holder and attention read this often in a conversation
IDLE_WAIT_S = 1.0  # the thread's longest sleep with nothing due
DECIDE_MARGIN_S = 0.05  # a causal beat is decided this long before its lead runs out
SHIFT_MOVE_S, SHIFT_HOLD_S = 0.6, (1.5, 3.0)  # the listening shift's way and hold
BODY_SETTLE_S = 0.8  # the cart is the gestures' again this long after a clip's end (two stops)
SWAY_GAP_S = 0.5  # the held arm sways this long after its last motion at the earliest
POSTURE_MIN_S = 1.0  # the arm rises only with this much of the reply's speech known after it
POSTURE_TAIL_S = 0.5  # the posture's hold reaches this far past the reply's audio known
RENEW_AHEAD_S = 0.6  # the hold is sent again when it has less than this left...
RENEW_GAIN_S = 0.3  # ...and the reply's audio known goes on at least this much past it
RETURN_AFTER_S = 0.3  # the arm goes back to the posture this long after a mark's motion
POSTURE_CAPS = 0.5  # the posture's way at this share of the arm's caps (rest to ready ~0.8 s)
SWAY_MARGIN_S = RENEW_AHEAD_S + POLL_S  # the held arm's hold outlasts its sway by this at least
# The book's own ensembles (config/animation.json, callable false) the listeners ask for
WEAK, STRONG, PEAK = "pulse_weak", "pulse_strong", "pulse_peak"  # an accent, by its strength
LAUGHING = "pulse_laugh"  # an accent of the voice's laughing
LISTEN_NOD, ACK_NOD = "listen_nod", "ack_nod"  # the nods of the person's pause and statement
ARM_SWAY = "arm_sway"  # the held arm's breath
LAUGH_SOUND = "laugh_sound"  # the moment of the voice's laughing
# The composition a moment due from the person's words plays, unless config/animation.json's
# moments map it to another
DUE = {GREETING_DUE: "greeting", FAREWELL_DUE: "farewell"}
MINE = frozenset(
    (WEAK, STRONG, PEAK, LAUGHING, LISTEN_NOD, ACK_NOD, ARM_SWAY, "melody", "question",
     "think_glance", "aversion", "smile", "listen_shift", POSTURE)
)  # fmt: skip
QUESTION = (  # a question: its mark, or a question word first
    r"\?|^\W*(?:(?:а|и|ну)\W+)?"  # noqa: RUF001
    r"(?:кто|где|когда|почему|зачем|куда|откуда|сколько|разве|неужели)\b"
)
GREETING = (  # a greeting among the person's words
    r"\b(?:привет\w*|здравствуй\w*|добр(?:ое|ый) (?:утро|день|вечер)|салют|hello|hi|hey)\b"  # noqa: RUF001
)


def local_hour() -> int:
    """The hour of the day here (0..23)."""
    return time.localtime().tm_hour


@dataclass(frozen=True)
class PulseKnobs:
    """config/expression.json's ``pulse``: the body while the robot speaks."""

    melody_deg_per_st: float = 0.2
    melody_max_deg: float = 0.9
    melody_cutoff_hz: float = 1.0
    melody_key_gap_s: float = 0.5
    beat_z: float = 2.0
    strong_z: float = 3.0
    beat_gap_s: float = 2.0
    beats_per_minute: int = 12
    beat_lead_min_s: float = 0.30
    mark_gap_s: float = 0.6
    wrist_beat_p: float = 0.5
    wheel_accent_share: float = 0.5
    laugh_run: int = 3
    laugh_gap_s: tuple[float, float] = (0.2, 0.4)
    arm_up_min_sentences: int = 2
    voice_stats: VoiceStats = field(default_factory=VoiceStats)


@dataclass(frozen=True)
class MomentKnobs:
    """config/expression.json's ``moments``: between turns and while the person speaks (the
    breath is the book's idle layer, the engine's)."""

    head_gap_s: float = 1.0
    think_glance_s: float = 0.8
    think_p: float = 0.75
    think_deg: tuple[float, float] = (8.0, 15.0)
    think_back_s: tuple[float, float] = (0.5, 1.3)
    think_dirs: Mapping[str, float] = field(
        default_factory=lambda: {"up": 0.39, "down": 0.29, "side": 0.32}
    )
    aversion_reply_s: float = 6.0
    aversion_deg: tuple[float, float] = (5.0, 8.0)
    aversion_s: tuple[float, float] = (1.5, 2.0)
    end_gaze_s: float = 2.4
    listener_nod_p: float = 0.6
    listener_gap_s: float = 5.0
    listener_min_speech_s: float = 2.0
    ack_nod_p: float = 0.3
    ack_min_s: float = 1.5
    nod_rock_p: float = 0.3
    question: str = QUESTION
    greeting: str = GREETING
    predict_delay_s: float = 0.2
    smile_s: float = 2.0
    listen_shift_deg: float = 0.0
    listen_shift_s: tuple[float, float] = (6.0, 12.0)


@dataclass(frozen=True)
class QuietKnobs:
    """config/expression.json's ``quiet``: when only the melody and the face move."""

    styles: tuple[str, ...] = ("whispering", "quiet", "sleepy", "sad", "tired", "calm", "slow")
    hours: tuple[int, int] = (23, 8)
    peak_db: float = -30.0


@dataclass(frozen=True)
class AliveKnobs:
    """The three blocks together."""

    pulse: PulseKnobs = field(default_factory=PulseKnobs)
    moments: MomentKnobs = field(default_factory=MomentKnobs)
    quiet: QuietKnobs = field(default_factory=QuietKnobs)


def _pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value), float(value)
    return float(value[0]), float(value[1])


def alive_knobs(data: Mapping[str, Any]) -> AliveKnobs:
    """config/expression.json's object as the listeners' knobs (defaults for what it lacks)."""
    p, m, q = data.get("pulse", {}), data.get("moments", {}), data.get("quiet", {})
    dp, dm, dq = PulseKnobs(), MomentKnobs(), QuietKnobs()

    def num(block: Mapping[str, Any], key: str, default: float) -> float:
        return float(block.get(key, default))

    pulse = PulseKnobs(
        melody_deg_per_st=num(p, "melody_deg_per_st", dp.melody_deg_per_st),
        melody_max_deg=num(p, "melody_max_deg", dp.melody_max_deg),
        melody_cutoff_hz=num(p, "melody_cutoff_hz", dp.melody_cutoff_hz),
        melody_key_gap_s=num(p, "melody_key_gap_s", dp.melody_key_gap_s),
        beat_z=num(p, "beat_z", dp.beat_z),
        strong_z=num(p, "strong_z", dp.strong_z),
        beat_gap_s=num(p, "beat_gap_s", dp.beat_gap_s),
        beats_per_minute=int(p.get("beats_per_minute", dp.beats_per_minute)),
        beat_lead_min_s=num(p, "beat_lead_min_s", dp.beat_lead_min_s),
        mark_gap_s=num(p, "mark_gap_s", dp.mark_gap_s),
        wrist_beat_p=num(p, "wrist_beat_p", dp.wrist_beat_p),
        wheel_accent_share=num(p, "wheel_accent_share", dp.wheel_accent_share),
        laugh_run=int(p.get("laugh_run", dp.laugh_run)),
        laugh_gap_s=_pair(p.get("laugh_gap_s"), dp.laugh_gap_s),
        arm_up_min_sentences=int(p.get("arm_up_min_sentences", dp.arm_up_min_sentences)),
        voice_stats=VoiceStats.of(p.get("voice_stats", {})),
    )
    dirs = m.get("think_dirs", dm.think_dirs)
    moments = MomentKnobs(
        head_gap_s=num(m, "head_gap_s", dm.head_gap_s),
        think_glance_s=num(m, "think_glance_s", dm.think_glance_s),
        think_p=num(m, "think_p", dm.think_p),
        think_deg=_pair(m.get("think_deg"), dm.think_deg),
        think_back_s=_pair(m.get("think_back_s"), dm.think_back_s),
        think_dirs={str(k): float(v) for k, v in dirs.items() if k != "note"},
        aversion_reply_s=num(m, "aversion_reply_s", dm.aversion_reply_s),
        aversion_deg=_pair(m.get("aversion_deg"), dm.aversion_deg),
        aversion_s=_pair(m.get("aversion_s"), dm.aversion_s),
        end_gaze_s=num(m, "end_gaze_s", dm.end_gaze_s),
        listener_nod_p=num(m, "listener_nod_p", dm.listener_nod_p),
        listener_gap_s=num(m, "listener_gap_s", dm.listener_gap_s),
        listener_min_speech_s=num(m, "listener_min_speech_s", dm.listener_min_speech_s),
        ack_nod_p=num(m, "ack_nod_p", dm.ack_nod_p),
        ack_min_s=num(m, "ack_min_s", dm.ack_min_s),
        nod_rock_p=num(m, "nod_rock_p", dm.nod_rock_p),
        question=str(m.get("question", dm.question)),
        greeting=str(m.get("greeting", dm.greeting)),
        predict_delay_s=num(m, "predict_delay_s", dm.predict_delay_s),
        smile_s=num(m, "smile_s", dm.smile_s),
        listen_shift_deg=num(m, "listen_shift_deg", dm.listen_shift_deg),
        listen_shift_s=_pair(m.get("listen_shift_s"), dm.listen_shift_s),
    )
    hours = q.get("hours", dq.hours)
    quiet = QuietKnobs(
        tuple(str(s) for s in q.get("styles", dq.styles)),
        (int(hours[0]), int(hours[1])),
        num(q, "peak_db", dq.peak_db),
    )
    return AliveKnobs(pulse, moments, quiet)


def load_alive_knobs(path: str | Path | None = None) -> AliveKnobs:
    """config/expression.json (wherever this library runs, or ``path``)."""
    import json

    if path is None:
        from pepin.deployment import config_file

        path = config_file("expression.json")
    return alive_knobs(json.loads(Path(path).read_text()))


def head_clip(
    frames: Sequence[tuple[float, float, float]], stroke_s: float, layer: Layer = "gesture"
) -> Clip:
    """A head clip through ``frames`` (seconds from its start, pan + left, tilt + down, degrees
    off the holder's aim), its stroke ``stroke_s`` from its start, placed so its anchor is the
    stroke."""
    times = tuple(round(f[0], 4) for f in frames)
    tracks = []
    for dof, k in (("pan_deg", 1), ("tilt_deg", 2)):
        values = tuple(round(f[k], 4) for f in frames)
        if any(abs(v) > 0.0 for v in values):
            tracks.append(Track(dof, "additive", times, values))
    return Clip("head", -stroke_s, stroke_s, times[-1], tuple(tracks), layer=layer)


def _holds(part: Any) -> bool:
    """Whether a part holds a unit's pose (an arm that keeps one: its ``holding()``, or the
    ``held`` pose a part without it shows)."""
    holding = getattr(part, "holding", None)
    if callable(holding):
        return holding() is not None
    return getattr(part, "held", None) is not None


def held_clip(pan_deg: float, tilt_deg: float, out_s: float, seconds: float) -> Clip:
    """A look off the aim and back: there in ``out_s`` (its stroke), held, back by ``seconds``."""
    back = max(out_s + 0.05, seconds - BACK_S)
    end = max(seconds, back + 0.05)
    frames = ((0.0, 0.0, 0.0), (out_s, pan_deg, tilt_deg), (back, pan_deg, tilt_deg), (end, 0, 0))
    return head_clip(frames, out_s)


def head_prep_s(clip: Clip, latency_s: float, settings: GazeSettings | None = None) -> float:
    """The seconds from sending a head clip to its stroke, computed from the clip and the
    arbiter's rules: the head's latency plus the time to the stroke, never under the arbiter's
    shortest half period (a nod's bottom is a turn of the tilt: 0.2 s after its first move at
    2.5 Hz)."""
    s = settings or GazeSettings()
    half = 0.5 / s.clip_max_hz if s.clip_max_hz > 0 else 0.0
    return latency_s + max(clip.stroke_s, half)


@dataclass
class _PartState:
    """One part of a reply as the listeners follow it."""

    text: str = ""
    marked: bool = False  # a gesture marker of the brain's in it
    quiet: bool = False  # it went to the voice in a quiet moment
    question: bool = False
    start_s: float | None = None
    onset_s: float | None = None
    end_s: float = 0.0
    nuclei: list[Nucleus] = field(default_factory=list)
    keys: list[tuple[float, float]] = field(default_factory=list)  # the melody: (second, tilt)
    posted: int = 0  # the melody's keys up to this one went in overlays
    segments: list[tuple[float, float]] = field(default_factory=list)  # overlays sent (start, end)
    beat: bool = False  # this part's accent is decided (played or not)
    waiting: bool = False  # a timer decides it when its lead is about to run out
    melody_waiting: bool = False  # a timer sends the melody's next batch
    tried: set[float] = field(default_factory=set)  # nuclei a beat could not take
    laughed: set[float] = field(default_factory=set)  # its laugh's accents decided
    complete: bool = False
    skipped: list[str] = field(default_factory=list)
    accent: dict[str, Any] | None = None


@dataclass
class _Reply:
    """One reply: its style, its parts, its voice's first second and the audio known so far."""

    turn: int
    style: str = ""
    parts: dict[int, _PartState] = field(default_factory=dict)
    start_s: float | None = None
    end_s: float = 0.0
    last: Nucleus | None = None
    question: bool = False
    aversion: bool = False
    busy: list[tuple[float, float]] = field(default_factory=list)  # stream spans of my head moves
    sentences: int = 0  # parts with words at the voice (a sentence, or a clause a style cut)
    posture: str = ""  # the arm up: "" not due, "due", "up", "down" (was up), "done" (never)
    posture_why: str = ""  # why a due posture still waits
    peaked: bool = False  # its strongest accent played (pulse_peak)


@dataclass
class _Posture:
    """The arm held up while a long reply goes on: its reply, the book's pose, when the last
    clip sent of it is up (engine clock: nothing is sent again before) and how many times it
    was sent again (a renewal, a return after a mark)."""

    turn: int
    pose: Mapping[str, float]
    until: float
    renewals: int = 0


@dataclass(frozen=True)
class _Glance:
    """A thinking glance holding the head: its request (the engine ends it by that), its draw of
    ``think_back_s``, its span on the engine's clock and the close it follows."""

    request: Request
    back_s: float
    span: tuple[float, float]
    close_id: int


class Alive:
    """The engine's listeners: see the module's notes. Observer calls and the voice's audio
    only post to its thread; :meth:`step` does one round (tests call it directly)."""

    def __init__(
        self,
        engine: GestureEngine,
        prosody: ProsodyListener,
        *,
        knobs: Callable[[], AliveKnobs] = load_alive_knobs,
        clock: Callable[[], float] = time.monotonic,
        hour: Callable[[], int] = local_hour,
        seed: int | None = None,
        journal: EventSink | None = None,
        say: Callable[[str], None] | None = None,
    ) -> None:
        """``knobs`` read again at each conversation's open; ``clock`` the engine's (laptop
        monotonic); ``hour`` the local hour (the quiet hours); ``journal(kind, now, **fields)``
        the conversation's events.jsonl; ``say`` one line at each close."""
        self.engine = engine
        self.prosody = prosody
        self.load_knobs = knobs
        self.clock = clock
        self.hour = hour
        self.rng = random.Random(seed)
        self.journal = journal
        self.say = say
        self.knobs = self._read_knobs()
        self.settings = GazeSettings()
        prosody.wake = self._bell_set
        prosody.listening = lambda: self.engine.on
        self._inbox: deque[tuple[str, tuple[Any, ...]]] = deque()
        self._bell = threading.Event()
        self._thread: threading.Thread | None = None
        self._stop = False
        self._timers: list[tuple[float, int, Callable[[], None]]] = []
        self._seq = itertools.count()
        self._finishing = False  # a part's audio is all in: analyse it even with nothing queued
        self._poll_at = 0.0  # attention is read again then
        self._head_seen: Reservation | None = None  # the latest head gesture not the listeners'
        self._arm_seen: Reservation | None = None  # the arm's latest reservation
        self._glance: _Glance | None = None  # the thinking glance holding the head
        self._posture: _Posture | None = None  # the arm held up for a long reply
        self._reset()

    # -- the observer (the cascade's loop thread: post only) ---------------------------------

    def opened(self, wake: Heard) -> None:
        """A conversation opened."""
        self._post("opened", wake)

    def speech_started(self, t: float) -> None:
        """The person speaks."""
        self._post("speech_started", t)

    def speech_ended(self, heard: Heard) -> None:
        """The person's utterance closed and read."""
        self._post("speech_ended", heard)

    def part_queued(self, turn: int, index: int, part: Part) -> None:
        """A part of a reply went to the voice."""
        self._post("part_queued", turn, index, part)

    def voice_started(self, turn: int) -> None:
        """A reply's first audio is with the player."""
        self._post("voice_started", turn)

    def part_started(self, turn: int, index: int, stream_s: float) -> None:
        """A part's first audio is with the player."""
        self._post("part_started", turn, index, stream_s)

    def text_ahead(self, turn: int, sentence: int, text: str) -> None:
        """Nothing here (the word clock's)."""

    def part_spoken(self, turn: int, index: int) -> None:
        """All of a part's audio is with the player."""
        self._post("part_spoken", turn, index)

    def reply_cut(self, turn: int) -> None:
        """A reply stopped short."""
        self._post("reply_cut", turn)

    def reply_done(self, turn: int) -> None:
        """A reply played out."""
        self._post("reply_done", turn)

    def moment(self, name: str, text: str = "") -> None:
        """A moment of the loop's (the ears' pause, with the person's words so far when the
        ears have them; a slow tool)."""
        self._post("moment", name, text)

    def drive(self, active: bool) -> None:
        """A drive began or ended."""
        self._post("drive", active)

    def closed(self, reason: str) -> None:
        """The conversation closed."""
        self._post("closed", reason)

    # -- the engine's hooks (the engine's thread) -----------------------------------------------

    def stays_up(self, turn: int) -> bool:
        """Whether the arm stays up after reply ``turn``'s last stroke: its posture is due or up
        (the scheduler's :attr:`pepin.expression.schedule.Scheduler.stay_up`)."""
        reply = self._replies.get(turn)
        return reply is not None and reply.posture in ("due", "up")

    def scene(self, scene: Scene, request: Request) -> Scene:
        """What a rendering may use, for all but a person's ask (it plays whole): in a quiet
        moment the face only."""
        if request.priority <= Priority.ASK or self._quiet is None:
            return scene
        return replace(scene, ready=scene.ready & {"face"}, quiet=True)

    # -- the thread ----------------------------------------------------------------------------

    def start(self) -> Alive:
        """Run the thread; returns self so it chains."""
        self._thread = threading.Thread(target=self._run, name="alive", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        """Stop the thread."""
        self._stop = True
        self._bell.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop:
            try:
                wait = self.step()
            except Exception:
                logger.exception("alive: a step broke")
                wait = IDLE_WAIT_S
            self._bell.wait(IDLE_WAIT_S if wait is None else max(0.0, min(wait, IDLE_WAIT_S)))
            self._bell.clear()

    def _bell_set(self) -> None:
        self._bell.set()

    def _post(self, name: str, *args: Any) -> None:
        self._inbox.append((name, args))
        self._bell.set()

    def step(self) -> float | None:
        """One round: the calls posted, the audio analysed, the timers due, attention read; the
        seconds until something is due again (None: nothing until a call)."""
        self._remember()
        while self._inbox:
            name, args = self._inbox.popleft()
            try:
                getattr(self, "_" + name)(*args)
            except Exception:
                logger.exception("alive: %s broke", name)
        if self.prosody.pending() or self._finishing:
            self._finishing = False
            for news in self.prosody.analyse():
                try:
                    self._news(news)
                except Exception:
                    logger.exception("alive: the part's news broke")
        now = self.clock()
        while self._timers and self._timers[0][0] <= now:
            _, _, fn = heapq.heappop(self._timers)
            try:
                fn()
            except Exception:
                logger.exception("alive: a timer broke")
        if self._conversation and now >= self._poll_at:
            self._poll_at = now + POLL_S
            self._attend(now)
            self._keep(now)
        waits = [self._timers[0][0] - now] if self._timers else []
        if self._conversation:
            waits.append(self._poll_at - now)
        return max(0.0, min(waits)) if waits else None

    def _remember(self) -> None:
        """The engine's latest head gesture of another's (a marker's, an ask's) and the arm's
        latest reservation (the reservations' own memory: an entry stays after its end and its
        fade, until another takes the part)."""
        reservations = self.engine.scheduler.reservations
        head = reservations.last("head")
        if (
            head is not None
            and head.clip.layer == "gesture"
            and head.performance.intent not in MINE
        ):
            self._head_seen = head
        arm = reservations.last("arm")
        if arm is not None:
            self._arm_seen = arm

    def _at(self, due: float, fn: Callable[[], None]) -> None:
        heapq.heappush(self._timers, (due, next(self._seq), fn))
        self._bell.set()

    # -- the conversation (the thread) --------------------------------------------------------

    def _reset(self) -> None:
        self._conversation = False
        self._driving = False
        self._listening = False
        self._speech_s = 0.0
        self._opened_s = self.clock()
        self._close_id = 0
        self._audio_close = -1  # the close whose reply's audio came
        self._head_seen = None
        self._arm_seen = None
        self._glance = None
        self._posture = None
        self._close_s: float | None = None
        self._awaiting = False
        self._asked = False
        self._greeted = -1
        self._slow_tool = -1
        self._acked = -1
        self._last_dir: str | None = None
        self._waits: deque[float] = deque(maxlen=WAITS_KEPT)
        self._replies: dict[int, _Reply] = {}
        self._strokes: deque[float] = deque(maxlen=16)  # discrete head strokes (engine clock)
        self._returns: deque[float] = deque(maxlen=16)  # held moves' returns to the aim
        self._spans: deque[tuple[float, float]] = deque(maxlen=16)  # head moves' (start, end)
        self._glance_until = -math.inf  # a thinking glance holds the head until then
        self._beats: deque[float] = deque()
        self._last_listener = -math.inf
        self._smiled = False
        self._poll_at = self.clock()
        self._peak: float | None = None
        self._quiet: str | None = None
        self._finishing = False
        self._shift_at: float | None = None
        kinds = MINE | {"greeting", "farewell", "arm_hold"}  # arm_hold: the posture sent again
        self._counts: dict[str, int] = dict.fromkeys(sorted(kinds), 0)
        self._head_moves = 0  # the listeners' discrete head moves (the minute's rate)
        self._sway = 0.0  # the weak accents' share of the cart's sway owed (wheel_accent_share)
        # the cart's motions the listeners asked for, each (start, end + its settling) on the
        # engine clock: another goes only outside them all
        self._body_spans: deque[tuple[float, float]] = deque(maxlen=16)
        self._sway_at = -math.inf  # the held arm sways again from then (engine clock)
        self._speech_total = 0.0  # seconds of the robot's audio
        self._speaking: set[int] = set()  # replies with parts at the voice, not yet done
        self._cpu0 = (self.prosody.cpu_s, self.prosody.audio_s)

    def _read_knobs(self) -> AliveKnobs:
        try:
            knobs = self.load_knobs()
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            logger.warning("alive: the knobs did not load (%s): defaults", error)
            knobs = AliveKnobs()
        self.prosody.configure(knobs.pulse.voice_stats, knobs.pulse.melody_cutoff_hz)
        return knobs

    def _opened(self, wake: Heard) -> None:
        self._reset()
        self.knobs = self._read_knobs()
        self._conversation = True
        self._opened_s = self.clock()
        self._update_quiet(None)

    def _speech_started(self, t: float) -> None:
        now = self.clock()
        self._listening = True
        self._speech_s = t if 0.0 < now - t < 30.0 else now
        self._awaiting = False
        shift = self.knobs.moments
        if self._conversation and shift.listen_shift_deg > 0.0:
            self._shift_at = now + self.rng.uniform(*shift.listen_shift_s)
            self._at(self._shift_at, self._shift)

    def _speech_ended(self, heard: Heard) -> None:
        now = self.clock()
        self._listening = False
        self._shift_at = None
        if not self._conversation or not heard.words:
            return
        self._peak = heard.peak_db
        self._update_quiet(None)
        self._close_id += 1
        close_id = self._close_id
        self._close_s = min(heard.end_s, now) if heard.end_s > 0 else now
        self._awaiting = True
        m = self.knobs.moments
        text = heard.text.strip()
        self._asked = bool(re.search(m.question, text, re.IGNORECASE))
        due = self._close_s + m.predict_delay_s
        if re.search(m.greeting, text.lower()):
            self._greeted = close_id
            self._at(due, lambda: self._due(GREETING_DUE, close_id, due))
        elif is_goodbye(text):
            self._greeted = close_id
            self._at(due, lambda: self._due(FAREWELL_DUE, close_id, due))
        else:
            self._ack(heard, close_id, now)
        self._at(self._close_s + m.think_glance_s, lambda: self._think(close_id))

    def _part_queued(self, turn: int, index: int, part: Part) -> None:
        self._speaking.add(turn)
        reply = self._reply(turn)
        if part.style is not None:
            reply.style = part.style
        state = reply.parts.setdefault(index, _PartState())
        state.text = part.text
        gestures = [m for m in part.markers if m.channel == "gesture"]
        state.marked = bool(gestures)
        state.question = part.text.rstrip().rstrip("»\"')").endswith("?")
        reply.sentences = sum(1 for s in reply.parts.values() if s.text.strip())
        self._update_quiet(reply)
        state.quiet = self._quiet is not None
        self._posture_due(reply)

    def _voice_started(self, turn: int) -> None:
        now = self.clock()
        if self._awaiting and self._close_s is not None:
            self._waits.append(now - self._close_s)
            self._audio_close = self._close_id
        self._awaiting = False
        self._asked = False

    def _part_started(self, turn: int, index: int, stream_s: float) -> None:
        self._voice_started(turn)
        reply = self._reply(turn)
        state = reply.parts.setdefault(index, _PartState())
        state.start_s = stream_s
        if state.quiet:
            self._posture_end(reply, f"quiet: {self._quiet or 'the style'}")
        if index == 0:
            self._glance_back(stream_s)
        self._raise(reply)

    def _part_spoken(self, turn: int, index: int) -> None:
        self.prosody.finish(turn, index)
        self._finishing = True

    def _reply_cut(self, turn: int) -> None:
        self.prosody.drop(turn)
        self._speaking = {t for t in self._speaking if t > turn}
        for key in [t for t in self._replies if t <= turn]:
            self._log_reply(self._replies.pop(key))

    def _reply_done(self, turn: int) -> None:
        self.prosody.drop(turn, only=True)
        self._speaking.discard(turn)
        reply = self._replies.pop(turn, None)
        if reply is not None:
            self._log_reply(reply)

    def _moment(self, name: str, text: str = "") -> None:
        if name == LISTENER_PAUSE:
            self._listener_nod(text)
        elif name == SLOW_TOOL:
            self._slow_tool = self._close_id  # its thinking composition takes the wait

    def _drive(self, active: bool) -> None:
        self._driving = active
        posture = self._posture
        if active and posture is not None:  # the engine rests every part
            self._posture = None
            reply = self._replies.get(posture.turn)
            if reply is not None:
                reply.posture = "down"

    def _closed(self, reason: str) -> None:
        if self._conversation:
            for reply in list(self._replies.values()):
                self._log_reply(reply)
            self._summary(reason)
        self.prosody.clear()
        self._timers.clear()
        self._reset()

    def _reply(self, turn: int) -> _Reply:
        reply = self._replies.get(turn)
        if reply is None:
            reply = self._replies[turn] = _Reply(turn)
            for old in [t for t in self._replies if t < turn - 2]:
                del self._replies[old]
        return reply

    # -- the voice's audio --------------------------------------------------------------------

    def _news(self, news: PartNews) -> None:
        """What the prosody listener found in a part: the melody, the accent, the question tilt,
        the pause for an aversion."""
        reply = self._reply(news.turn)
        state = reply.parts.setdefault(news.index, _PartState())
        p = self.knobs.pulse
        self._speech_total += max(0.0, news.end_s - max(news.start_s, state.end_s))
        state.end_s = max(state.end_s, news.end_s)
        reply.end_s = max(reply.end_s, news.end_s)
        if news.onset_s is not None and state.onset_s is None:
            state.onset_s = news.onset_s
            state.keys = [(news.onset_s, 0.0)]
            reply.start_s = news.onset_s if reply.start_s is None else reply.start_s
        for turn in news.turns:
            if state.keys and turn.second - state.keys[-1][0] >= p.melody_key_gap_s:
                tilt = -p.melody_deg_per_st * turn.st
                tilt = max(-p.melody_max_deg, min(p.melody_max_deg, tilt))
                state.keys.append((turn.second, round(tilt, 3)))
        now = self.clock()
        for nucleus in news.nuclei:
            state.nuclei.append(nucleus)
            last = reply.last
            if last is not None and nucleus.second - last.second >= PAUSE_GAP_S:
                self._aversion(reply, last.second + 0.1, now)
            reply.last = nucleus
        if news.complete:
            state.complete = True
            if state.keys:
                end = news.span.voiced[1] if news.span is not None else state.end_s
                end = max(end, state.keys[-1][0] + p.melody_key_gap_s)
                state.keys.append((round(end, 4), 0.0))
            if state.question and not reply.question:
                self._question(reply, state, now)
        self._laugh(reply, state, now)
        self._accent(reply, state, now)
        self._melody(reply, state, now)
        if news.complete:
            self._log_part(reply, news.index, state)

    def _accent(self, reply: _Reply, state: _PartState, now: float) -> None:
        """The part's accent: on its most prominent nucleus when its audio is all in, else on
        its first over ``beat_z`` once the lead of the slowest part its ensemble wants is about
        to run out."""
        if state.beat or not self.engine.on:
            return
        p = self.knobs.pulse
        over = [n for n in state.nuclei if n.z >= p.beat_z and n.second not in state.tried]
        if not over:
            if state.complete:
                state.beat = True
            return
        if state.marked:
            state.beat = True
            state.skipped.append("a marker's part")
            return
        if self._quiet is not None:
            state.beat = True
            state.skipped.append(f"quiet: {self._quiet}")
            return
        if state.laughed:
            state.beat = True
            state.skipped.append("its laugh is its accent")
            return
        if not state.complete:
            first = over[0]
            lead = self._lead(first.second + STROKE_BIAS_S)
            floor = self._floor(reply, first)
            if lead - floor > DECIDE_MARGIN_S:
                if not state.waiting:
                    state.waiting = True
                    self._at(now + lead - floor - DECIDE_MARGIN_S,
                             lambda: self._decide(reply, state))  # fmt: skip
                return
            candidates = [first]
        else:
            candidates = sorted(over, key=lambda n: -n.z)
        for nucleus in candidates:
            why = self._beat(reply, state, nucleus, now)
            if why is None:
                state.beat = True
                return
            state.tried.add(nucleus.second)
            state.skipped.append(why)
        if state.complete:
            state.beat = True

    def _decide(self, reply: _Reply, state: _PartState) -> None:
        state.waiting = False
        if self._replies.get(reply.turn) is reply:
            self._accent(reply, state, self.clock())

    def _strength(self, reply: _Reply, nucleus: Nucleus) -> str:
        """The ensemble an accent on ``nucleus`` asks for: ``pulse_weak`` under ``strong_z``, the
        reply's strongest known from ``strong_z`` (once a reply) ``pulse_peak``, else
        ``pulse_strong``."""
        if nucleus.z < self.knobs.pulse.strong_z:
            return WEAK
        known = max(n.z for state in reply.parts.values() for n in state.nuclei)
        return PEAK if not reply.peaked and nucleus.z >= known else STRONG

    def _sway_due(self, intent: str) -> bool:
        """Whether an accent's ensemble wants the cart: a strong one always, a weak one when the
        sway's share owed reaches a whole one (``wheel_accent_share``: steady, not dice)."""
        share = self.knobs.pulse.wheel_accent_share
        return intent != WEAK or (share > 0.0 and self._sway + share >= 1.0 - 1e-9)

    def _floor(self, reply: _Reply, nucleus: Nucleus) -> float:
        """The lead an accent on ``nucleus`` needs for the slowest part it would want: the
        head's nod always, the arm's beat while a unit holds it, the wheels when its strength
        (or the sway's share) asks for them."""
        intent = self._strength(reply, nucleus)
        floor = self._head_floor(intent)
        if _holds(self.engine.parts.get("arm")):
            floor = max(floor, self._prep(intent, "arm"))
        if self._sway_due(intent):
            floor = max(floor, self._prep(intent, "body"))
        return floor

    def _beat(self, reply: _Reply, state: _PartState, nucleus: Nucleus, now: float) -> str | None:
        """An accent on ``nucleus``: its strength's ensemble on the parts free for it (the head's
        nod, the arm's beat over a held unit, the wheels' rock); None when sent, else why not,
        part by part."""
        p = self.knobs.pulse
        stroke_s = nucleus.second + STROKE_BIAS_S
        lead = self._lead(stroke_s)
        stroke = now + lead
        intent = self._strength(reply, nucleus)
        parts: set[PartName] = set()
        whys: list[str] = []
        why = self._head_room(intent, stroke, lead, now, p.beat_gap_s, mark_gap=p.mark_gap_s,
                              per_minute=True)  # fmt: skip
        if why is None:
            parts.add("head")
        else:
            whys.append(f"head: {why}")
        if intent != WEAK or self.rng.random() < p.wrist_beat_p:
            why = self._arm_room(intent, reply, stroke, lead, now)
            if why is None:
                parts.add("arm")
            elif _holds(self.engine.parts.get("arm")):
                whys.append(f"arm: {why}")
        if self._sway_due(intent):
            why = "the person speaks" if self._listening else None
            why = why or self._body_room(intent, stroke, lead, now)
            if why is None:
                parts.add("body")
            else:
                whys.append(f"wheels: {why}")
        if not parts:
            return "; ".join(whys)
        self._ask(intent, AtStream(round(stroke_s, 4), reply.turn), Priority.PULSE, "pulse",
                  parts, stroke, now, reply=reply)  # fmt: skip
        if "head" in parts:
            self._beats.append(now)
        if intent == WEAK:
            self._sway = min(1.0, self._sway + p.wheel_accent_share) - ("body" in parts)
        reply.peaked = reply.peaked or intent == PEAK
        state.accent = {"s": round(nucleus.second, 3), "z": round(nucleus.z, 2),
                        "lead_s": round(lead, 3), "kind": intent.removeprefix("pulse_"),
                        "parts": sorted(parts)}  # fmt: skip
        state.skipped += whys
        return None

    def _ask(
        self,
        intent: str,
        anchor: AtStream | AtTime | None,
        priority: Priority,
        source: str,
        parts: set[PartName],
        stroke: float,
        now: float,
        *,
        reply: _Reply | None = None,
        rank: int = 0,
    ) -> Request:
        """The book's own ensemble ``intent`` asked for on ``parts``, its stroke at ``stroke``
        (engine clock): the head's ledger and the cart's taken for it."""
        request = Request(intent, anchor, priority, source, bind="word", rank=rank,
                          parts=frozenset(parts))  # fmt: skip
        self.engine.request(request)
        self._counts[intent] += 1
        if "head" in parts:
            pre, post = self._reach(intent, "head")
            self._occupy(stroke - pre, stroke + post, stroke, None)
            self._head_moves += 1
            if reply is not None and isinstance(anchor, AtStream):
                self._busy(reply, anchor.second - pre, anchor.second + post)
        if "body" in parts:
            pre, post = self._reach(intent, "body")
            start = max(stroke - pre, now + self._latency("body"))
            self._body_spans.append((start, start + pre + post + BODY_SETTLE_S))
        return request

    def _reach(self, intent: str, part: PartName) -> tuple[float, float]:
        """How long before its stroke ``intent`` moves ``part`` and how long after (the book's
        widest, the arm's and the wheels' stretched as the engine plays them)."""
        config = self.engine.config
        scale = {"arm": config.arm_speed_scale, "body": config.body_speed_scale}.get(part, 1.0)
        return span(self.engine.book, intent, part, 1.0 / scale if scale > 0.0 else 1.0)

    def _prep(self, intent: str, part: PartName) -> float:
        """The lead ``intent``'s ``part`` needs: its latency and its way to the stroke."""
        return self._latency(part) + self._reach(intent, part)[0]

    def _head_floor(self, intent: str) -> float:
        """The least lead of ``intent``'s head: its preparation, never under the arbiter's
        shortest half period (a nod's bottom is a turn of the tilt) nor ``beat_lead_min_s``."""
        s = self.settings
        half = 0.5 / s.clip_max_hz if s.clip_max_hz > 0 else 0.0
        pre = max(self._reach(intent, "head")[0], half)
        return max(self.knobs.pulse.beat_lead_min_s, self._latency("head") + pre)

    def _head_room(
        self,
        intent: str,
        stroke: float,
        lead: float,
        now: float,
        gap: float,
        *,
        mark_gap: float | None = None,
        per_minute: bool = False,
    ) -> str | None:
        """Why ``intent``'s head may not stroke at ``stroke`` (engine clock), None when it may:
        its lead, the accents' per-minute cap (``per_minute``), the head's ledger."""
        pre, post = self._reach(intent, "head")
        if pre == post == 0.0:
            return "none in it"
        floor = self._head_floor(intent)
        if lead < floor:
            return f"lead {lead:.2f} s under {floor:.2f}"
        if per_minute:
            while self._beats and now - self._beats[0] > 60.0:
                self._beats.popleft()
            if len(self._beats) >= self.knobs.pulse.beats_per_minute:
                return f"the per-minute cap {self.knobs.pulse.beats_per_minute}"
        return self._head_free(stroke - pre, stroke + post, stroke, gap, mark_gap=mark_gap)

    def _arm_room(
        self, intent: str, reply: _Reply, stroke: float, lead: float, now: float
    ) -> str | None:
        """Why ``intent``'s arm beat may not go (no unit holds the arm up, a clip moves it or
        waits for it, too little lead, the hold ends first), None when it may: it rides the
        held pose and keeps the rest of its hold (the scheduler's)."""
        arm = self.engine.parts.get("arm")
        if arm is None or not _holds(arm):
            return "no unit holds it up"
        reservations = self.engine.scheduler.reservations
        moving = reservations.holder("arm", now)
        if moving is not None and moving.priority < Priority.IDLE:
            return f"it plays {moving.performance.intent}"
        pre, post = self._reach(intent, "arm")
        if pre == post == 0.0:
            return "none in it"
        if lead < self._latency("arm") + pre:
            return f"lead {lead:.2f} s under {self._latency('arm') + pre:.2f}"
        start, end = stroke - pre, stroke + post
        for waiting in self.engine.pending("arm"):
            if waiting.start < end + RETURN_AFTER_S and waiting.end > start - RETURN_AFTER_S:
                return f"{waiting.intent} waits for it"
        if isinstance(arm, ArmPart):
            lasts = arm.held_until
        else:
            last = reservations.last("arm")
            lasts = last.ends + last.clip.hold_s if last is not None else now
        posture = self._posture
        if posture is not None and posture.turn == reply.turn:
            lasts = max(lasts, now + self._lead(reply.end_s) + POSTURE_TAIL_S)  # it is kept so
        if lasts < end:
            return "its hold ends first"
        return None

    def _body_room(self, intent: str, stroke: float, lead: float | None, now: float) -> str | None:
        """Why ``intent``'s wheels may not stroke at ``stroke`` (engine clock), None when they
        may: no wheels, the gaze's body turn (the one thing that outranks the pulse's wheels),
        too little ``lead`` (None: it may go late), the cart still moving or settling from a
        clip of anyone's."""
        body = self.engine.parts.get("body")
        not_ready = body.ready() if body is not None else "no wheels"
        if not_ready is not None:
            return not_ready
        turning = self._turning()
        if turning is not None:
            return turning
        pre, post = self._reach(intent, "body")
        if pre == post == 0.0:
            return "none in it"
        if lead is not None and lead < self._prep(intent, "body"):
            return f"lead {lead:.2f} s under {self._prep(intent, 'body'):.2f}"
        start = max(stroke - pre, now + self._latency("body"))
        end = start + pre + post
        if any(start < b and end + BODY_SETTLE_S > a for a, b in self._body_spans):
            return "the cart still moves"
        held = self.engine.scheduler.reservations.holder("body", now)
        if held is not None and held.ends + BODY_SETTLE_S > start:
            return f"the cart plays {held.performance.intent}"
        for waiting in self.engine.pending("body"):
            if waiting.start < end + BODY_SETTLE_S and waiting.end + BODY_SETTLE_S > start:
                return f"{waiting.intent} waits for the cart"
        return None

    def _laugh(self, reply: _Reply, state: _PartState, now: float) -> None:
        """The voice laughing (no text read): each accent of a run of ``laugh_run`` accents of
        the part ``laugh_gap_s`` apart asks for ``pulse_laugh`` on the parts free for it, once."""
        p = self.knobs.pulse
        if p.laugh_run <= 0 or not self.engine.on or self._driving:
            return
        lo, hi = p.laugh_gap_s
        run: list[Nucleus] = []
        for nucleus in [*(n for n in state.nuclei if n.z >= p.beat_z), None]:
            if nucleus is not None and run and lo <= nucleus.second - run[-1].second <= hi:
                run.append(nucleus)
                continue
            if len(run) >= p.laugh_run:
                for accent in run:
                    if accent.second not in state.laughed:
                        state.laughed.add(accent.second)
                        self._laugh_beat(reply, accent, now)
            run = [nucleus] if nucleus is not None else []

    def _laugh_beat(self, reply: _Reply, nucleus: Nucleus, now: float) -> None:
        """One accent of the voice's laugh: the cart's rock with the happy face where the cart
        is free (the next accent once its rock is over), a bob of the head where the head is."""
        stroke_s = nucleus.second + STROKE_BIAS_S
        lead = self._lead(stroke_s)
        stroke = now + lead
        parts: set[PartName] = set()
        whys: list[str] = []
        if self._quiet is not None:
            whys.append(f"quiet: {self._quiet}")
        else:
            why = self._body_room(LAUGHING, stroke, lead, now)
            if why is None:
                parts |= {"body", "face"}
            else:
                whys.append(f"wheels: {why}")
            why = self._head_room(LAUGHING, stroke, lead, now, self.knobs.moments.head_gap_s)
            if why is None:
                parts.add("head")
            else:
                whys.append(f"head: {why}")
        if parts:
            self._ask(LAUGHING, AtStream(round(stroke_s, 4), reply.turn), Priority.PULSE, "pulse",
                      parts, stroke, now, reply=reply)  # fmt: skip
        self._moment_log(LAUGH_SOUND, None if parts else "; ".join(whys), turn=reply.turn,
                         s=round(nucleus.second, 3), z=round(nucleus.z, 2),
                         parts=sorted(parts))  # fmt: skip

    def _question(self, reply: _Reply, state: _PartState, now: float) -> None:
        """The question tilt at the part's last nucleus, held past its audio."""
        if not state.nuclei:
            return
        last = state.nuclei[-1]
        stroke_s = last.second + STROKE_BIAS_S
        hold_to = state.end_s + QUESTION_HOLD_S
        seconds = QUESTION_OUT_S + max(0.1, hold_to - stroke_s) + BACK_S
        pan = QUESTION_PAN_DEG * self.rng.choice((-1.0, 1.0))
        clip = held_clip(pan, -QUESTION_TILT_DEG, QUESTION_OUT_S, seconds)
        why = self._held(reply, "question", QUESTION_END, clip, stroke_s, now)
        reply.question = why is None
        self._moment_log(QUESTION_END, why, turn=reply.turn)

    def _aversion(self, reply: _Reply, pause_s: float, now: float) -> None:
        """The long reply's look away at an inner pause, back well before the audio ends (once a
        reply; a pause where the head is busy leaves it to the next)."""
        m = self.knobs.moments
        if reply.aversion or reply.start_s is None or not self.engine.on or self._quiet:
            return
        if reply.end_s - reply.start_s < m.aversion_reply_s:
            return
        seconds = self.rng.uniform(*m.aversion_s)
        if pause_s + seconds + m.end_gaze_s > reply.end_s:
            return
        amp = self.rng.uniform(*m.aversion_deg)
        pan, tilt = (amp * self.rng.choice((-1.0, 1.0)), 0.0)
        if self.rng.random() < 0.5:
            pan, tilt = 0.0, amp
        clip = held_clip(pan, tilt, OUT_S, seconds)
        why = self._held(reply, "aversion", LONG_REPLY, clip, pause_s + OUT_S, now)
        reply.aversion = why is None
        self._moment_log(LONG_REPLY, why, turn=reply.turn, seconds=round(seconds, 2))

    def _held(
        self, reply: _Reply, intent: str, moment: str, clip: Clip, stroke_s: float, now: float
    ) -> str | None:
        """A held head move of the reply (its stroke at stream second ``stroke_s``): sent, or
        why not (quiet, lead, the head's gap)."""
        if not self.engine.on:
            return "gestures are off"
        if self._quiet is not None:
            return f"quiet: {self._quiet}"
        lead = self._lead(stroke_s)
        prep = head_prep_s(clip, self._latency("head"), self.settings)
        if lead < prep:
            return f"lead {lead:.2f} s under {prep:.2f}"
        stroke = now + lead
        start = stroke - clip.stroke_s
        end = start + clip.duration_s
        why = self._head_free(start, end, stroke, self.knobs.moments.head_gap_s)
        if why is not None:
            return why
        request = Request(intent, AtStream(round(stroke_s, 4), reply.turn), Priority.MOMENT,
                          "moment", rank=moment_rank(moment), clips=(clip,))  # fmt: skip
        self.engine.request(request)
        self._counts[intent] += 1
        self._occupy(start, end, stroke, end)
        start_s = stroke_s - clip.stroke_s
        self._busy(reply, start_s, start_s + clip.duration_s)
        return None

    def _melody(self, reply: _Reply, state: _PartState, now: float) -> None:
        """The part's melody keys not yet sent, as idle overlays round the head's own moves; a
        part still being made sends them in batches (``MELODY_BATCH_S``) or when the first is
        about to be due."""
        if not self.engine.on or len(state.keys) - 1 <= state.posted:
            return
        keys = state.keys[state.posted :]
        if not state.complete and keys[-1][0] - keys[0][0] < MELODY_BATCH_S:
            lead = self._lead(keys[1][0]) - self._latency("head")
            if lead > MELODY_AHEAD_S:
                if not state.melody_waiting:
                    state.melody_waiting = True
                    self._at(now + lead - MELODY_AHEAD_S, lambda: self._melody_due(reply, state))
                return
        state.posted = len(state.keys) - 1
        self._send_melody(reply, state, keys)

    def _melody_due(self, reply: _Reply, state: _PartState) -> None:
        """A batch's first key is about to be due: what the part has, sent."""
        state.melody_waiting = False
        if self._replies.get(reply.turn) is not reply or len(state.keys) - 1 <= state.posted:
            return
        keys = state.keys[state.posted :]
        state.posted = len(state.keys) - 1
        self._send_melody(reply, state, keys)

    def _send_melody(
        self, reply: _Reply, state: _PartState, keys: list[tuple[float, float]]
    ) -> None:
        """``keys`` (the first one where the head is) as overlays (:func:`melody_pieces`), none
        before a thinking glance still holding the head is back."""
        now = self.clock()
        played = self.engine.playout.played()
        earliest = played - self._lag() + self._latency("head") + 0.03
        if self._glance_until > now:
            earliest = max(earliest, earliest + self._glance_until - now + RESUME_S)
        gap = self.knobs.pulse.melody_key_gap_s
        for piece in melody_pieces(keys, reply.busy, earliest, gap):
            self._melody_request(reply.turn, piece)
            state.segments.append((piece[0][0], piece[-1][0]))

    def _melody_request(self, turn: int, keys: Sequence[tuple[float, float]]) -> None:
        t0 = keys[0][0]
        times = tuple(round(t - t0, 4) for t, _ in keys)
        values = tuple(v for _, v in keys)
        clip = Clip("head", 0.0, 0.0, times[-1], (Track("tilt_deg", "additive", times, values),),
                    layer="idle")  # fmt: skip
        self.engine.request(Request("melody", AtStream(round(t0, 4), turn), Priority.IDLE, "pulse",
                                    clips=(clip,)))  # fmt: skip
        self._counts["melody"] += 1

    def _busy(self, reply: _Reply, start_s: float, end_s: float) -> None:
        """A head move of the listeners' own over stream seconds ``start_s..end_s``: the melody
        sent over it already comes back after it."""
        reply.busy.append((start_s, end_s))
        for state in reply.parts.values():
            for a, b in list(state.segments):
                if a < end_s < b:
                    rest = [(t, v) for t, v in state.keys if end_s + RESUME_S < t <= b]
                    if rest:
                        self._send_melody(reply, state, [(end_s, 0.0), *rest])

    # -- the moments ----------------------------------------------------------------------------

    def _think(self, close_id: int) -> None:
        """The thinking glance, ``think_glance_s`` after the close with no reply audio."""
        m = self.knobs.moments
        now = self.clock()
        why: str | None = None
        if close_id != self._close_id or not self._awaiting:
            heard = close_id == self._close_id and self._audio_close == close_id
            why = "the reply's audio came first" if heard else "the person spoke again"
        elif not self._conversation or not self.engine.on or self._driving:
            why = "no conversation, off or driving"
        elif self._greeted == close_id:
            why = "a greeting or a farewell takes the wait"
        elif self._slow_tool == close_id:
            why = "a slow tool's thinking takes the wait"
        elif self.rng.random() >= m.think_p:
            why = f"p {m.think_p:g}"
        if why is not None:
            self._moment_log(THINK_WAIT, why)
            return
        dirs = {d: w for d, w in m.think_dirs.items() if d != self._last_dir and w > 0}
        direction = self.rng.choices(list(dirs), list(dirs.values()))[0] if dirs else "up"
        amp = self.rng.uniform(*m.think_deg)
        pan, tilt = {"up": (0.0, -amp), "down": (0.0, amp)}.get(direction, (amp, 0.0))
        if direction == "side":
            pan *= self.rng.choice((-1.0, 1.0))
        assert self._close_s is not None
        expected = statistics.median(self._waits) if self._waits else WAIT_PRIOR_S
        reply_at = max(self._close_s + expected, now + OUT_S + 0.3)
        back_s = self.rng.uniform(*m.think_back_s)
        seconds = min(THINK_MAX_S, reply_at + back_s - now)
        clip = held_clip(pan, tilt, OUT_S, seconds)
        latency = self._latency("head")
        stroke = now + latency + clip.stroke_s
        clips: list[Clip] = []
        if self._quiet is None:
            start, end = now + latency, now + latency + seconds
            free = self._head_free(start, end, stroke, m.head_gap_s)
            if free is not None:
                retry = self._retry_at(start, end, stroke, m.head_gap_s) - latency - clip.stroke_s
                if retry - self._close_s <= m.think_glance_s + 1.0:
                    self._at(retry, lambda: self._think(close_id))
                    return
                self._moment_log(THINK_WAIT, free)
                return
            clips.append(clip)
        if self.rng.random() < THINK_FACE_P:
            clips.append(Clip("face", clip.offset_s, 0.0, seconds,
                              events=(Event(0.0, "thinking", seconds),)))  # fmt: skip
        if not clips:
            self._moment_log(THINK_WAIT, f"quiet: {self._quiet}")
            return
        request = Request("think_glance", None, Priority.MOMENT, "moment",
                          rank=moment_rank(THINK_WAIT), clips=tuple(clips))  # fmt: skip
        self.engine.request(request)
        self._counts["think_glance"] += 1
        if any(c.part == "head" for c in clips):
            self._last_dir = direction
            span = (now + latency, now + latency + seconds)
            self._occupy(*span, stroke, span[1])
            self._glance_until = span[1]
            self._glance = _Glance(request, back_s, span, close_id)
        self._moment_log(THINK_WAIT, None, dir=direction, deg=round(amp, 1),
                         seconds=round(seconds, 2), parts=[c.part for c in clips])  # fmt: skip

    def _glance_back(self, stream_s: float) -> None:
        """The reply's first part has its place in the stream: a thinking glance still due to
        hold past ``think_back_s`` into its first sound is ended then by the engine (its
        requester's release), the head's ledger shortened with it."""
        glance = self._glance
        if glance is None or glance.close_id != self._close_id:
            return
        now = self.clock()
        back = now + self._lead(stream_s) + glance.back_s
        fade = self.engine.config.fade_s.get("head", 0.0)
        if back + fade >= glance.span[1]:
            return  # its own return comes first

        def end() -> None:
            if self._glance is not glance:
                return
            self._glance = None
            self.engine.release(glance.request, "the reply's voice")
            ended = self.clock() + fade
            start, until = glance.span
            self._spans = deque([(a, ended) if (a, b) == (start, until) else (a, b)
                                 for a, b in self._spans], maxlen=self._spans.maxlen)  # fmt: skip
            self._returns = deque([ended if t == until else t for t in self._returns],
                                  maxlen=self._returns.maxlen)  # fmt: skip
            self._glance_until = min(self._glance_until, ended)

        self._at(back, end)

    def _listener_nod(self, text: str = "") -> None:
        """The listener's nod at the person's pause inside their utterance (``text``: their
        words so far, when the ears read them in time): ``listen_nod``, its nod with
        ``listener_nod_p``, the cart's rock with ``nod_rock_p``."""
        m = self.knobs.moments
        now = self.clock()
        asked = bool(re.search(m.question, text.strip(), re.IGNORECASE)) if text else self._asked
        why: str | None = None
        if not self._conversation or not self.engine.on or not self._listening:
            why = "nobody speaking, off"
        elif self._quiet is not None:
            why = f"quiet: {self._quiet}"
        elif now - self._speech_s < m.listener_min_speech_s:
            why = f"{now - self._speech_s:.1f} s of their speech"
        elif now - self._last_listener < m.listener_gap_s:
            why = f"the last {now - self._last_listener:.1f} s ago"
        elif asked:
            why = "their words so far are a question" if text else "after their question"
        if why is None:
            stroke = now + self._latency("head") + self._reach(LISTEN_NOD, "head")[0]
            parts, why = self._nod_parts(LISTEN_NOD, m.listener_nod_p, stroke, now)
            if parts:
                self._ask(LISTEN_NOD, AtTime(stroke), Priority.MOMENT, "moment", parts, stroke,
                          now, rank=moment_rank(LISTENER_PAUSE))  # fmt: skip
                self._last_listener = now
        self._moment_log(LISTENER_PAUSE, why)

    def _nod_parts(
        self, intent: str, nod_p: float, stroke: float, now: float
    ) -> tuple[set[PartName], str | None]:
        """A nod's ensemble's parts: the head with ``nod_p`` where its ledger has room, the
        cart's rock with ``nod_rock_p`` where the cart is free (it may go late); why none."""
        m = self.knobs.moments
        parts: set[PartName] = set()
        whys: list[str] = []
        if self.rng.random() < nod_p:
            pre, post = self._reach(intent, "head")
            why = self._head_free(stroke - pre, stroke + post, stroke, m.head_gap_s)
            if why is None:
                parts.add("head")
            else:
                whys.append(why)
        else:
            whys.append(f"p {nod_p:g}")
        if self.rng.random() < m.nod_rock_p:
            why = self._body_room(intent, stroke, None, now)
            if why is None:
                parts.add("body")
            else:
                whys.append(f"wheels: {why}")
        return parts, None if parts else "; ".join(whys)

    def _ack(self, heard: Heard, close_id: int, now: float) -> None:
        """The acknowledgement at the close of a statement: ``ack_nod``, its nod with
        ``ack_nod_p``, the cart's rock with ``nod_rock_p``."""
        m = self.knobs.moments
        why: str | None = None
        if not self.engine.on:
            why = "off"
        elif self._quiet is not None:
            why = f"quiet: {self._quiet}"
        elif heard.duration_s < m.ack_min_s:
            why = f"{heard.duration_s:.1f} s"
        elif self._asked:
            why = "a question"
        elif self._acked == close_id:
            why = "once a turn"
        if why is None:
            assert self._close_s is not None
            pre = self._reach(ACK_NOD, "head")[0]
            stroke = max(self._close_s + m.predict_delay_s, now + self._latency("head") + pre)
            parts, why = self._nod_parts(ACK_NOD, m.ack_nod_p, stroke, now)
            if parts:
                self._ask(ACK_NOD, AtTime(stroke), Priority.MOMENT, "moment", parts, stroke, now,
                          rank=moment_rank(ACKNOWLEDGED))  # fmt: skip
                self._acked = close_id
        self._moment_log(ACKNOWLEDGED, why)

    def _due(self, name: str, close_id: int, first: float) -> None:
        """A greeting or a farewell due from the person's words: the book's composition now, or
        once the body's turn toward the person is over."""
        now = self.clock()
        if not self._conversation or not self.engine.on or self._driving:
            self._moment_log(name, "no conversation, off or driving")
            return
        view = self._view()
        turning = view is not None and (view.phase == "turning" or view.past_arc)
        if turning and now - first < DEFER_MAX_S:
            self._at(now + DEFER_STEP_S, lambda: self._due(name, close_id, first))
            return
        intent = self.engine.book.moments.get(name, DUE[name])
        request = Request(intent, None, Priority.MOMENT, "moment", rank=moment_rank(name),
                          family=intent)  # fmt: skip
        self.engine.request(request)
        self._counts["greeting" if name == GREETING_DUE else "farewell"] += 1
        self._moment_log(name, None, deferred_s=round(now - first, 2) if now > first + 0.05 else 0)

    def _turning(self) -> str | None:
        """Why the cart is not the gestures' now (the speaker gaze turns it, or will), None
        when it is."""
        view = self._view()
        if view is not None and (view.phase == "turning" or view.past_arc):
            return "a body turn is pending"
        return None

    # -- the arm up while a long reply goes on --------------------------------------------------

    def _posture_due(self, reply: _Reply) -> None:
        """A reply with ``arm_up_min_sentences`` parts with words at the voice: its posture is
        due (once a reply), raised as soon as its speech allows (:meth:`_raise`)."""
        n = self.knobs.pulse.arm_up_min_sentences
        if reply.posture or n <= 0 or reply.sentences < n:
            return
        reply.posture = "due"
        self._raise(reply)

    def _raise(self, reply: _Reply) -> None:
        """A due posture: the arm from where it is to the book's ``ready``, held, sent at the
        stream second being heard (the reply's first sound when it is still to come); refused
        for the reply (off, quiet, a drive, no planner or pose, the arm not ready), or waiting
        for the next try (no sound yet, the person speaking, a body turn, the cart moving, a
        mark's arm, too little speech known)."""
        if reply.posture != "due" or reply.turn not in self._speaking:
            return
        now = self.clock()
        arm = self.engine.parts.get("arm")
        pose = self.engine.book.poses.get(HOME)
        why = self._posture_refusal(arm, pose)
        if why is not None:
            reply.posture = "done"
            self._moment_log(POSTURE, why, turn=reply.turn)
            return
        assert isinstance(arm, ArmPart) and pose is not None
        held = arm.holding()
        clip = pose_clip(arm, held if held is not None else arm.base, pose, 0.0, POSTURE_CAPS)
        second = self._posture_second(reply)
        if second is None:
            reply.posture_why = "the reply's voice has not begun"
            return
        wait = self._posture_wait(reply, now, second, clip)
        if wait is not None:
            reply.posture_why = wait
            return
        up_at = now + max(0.0, self._lead(second)) + clip.duration_s
        clip = replace(clip, hold_s=self._posture_hold(reply, arm, up_at, now))
        self._send_posture(reply, clip, second)
        reply.posture = "up"
        self._posture = _Posture(reply.turn, pose, up_at)
        self._counts[POSTURE] += 1
        self._moment_log(POSTURE, None, turn=reply.turn, sentences=reply.sentences,
                         base="held" if held is not None else "rest",
                         seconds=round(clip.duration_s, 2), hold_s=clip.hold_s)  # fmt: skip

    def _posture_refusal(self, arm: Any, pose: Mapping[str, float] | None) -> str | None:
        """Why the reply gets no posture at all, None when it may."""
        if not self.engine.on:
            return "gestures are off"
        if self._driving:
            return "a drive is under way"
        if self._quiet is not None:
            return f"quiet: {self._quiet}"
        if not isinstance(arm, ArmPart):
            return "no arm planner"
        if pose is None:
            return f"no {HOME} pose in the book"
        return arm.ready()

    def _posture_second(self, reply: _Reply) -> float | None:
        """The stream second the posture goes at: the one being heard, or the reply's first
        sound while it is still to come; None before any part of it is with the player."""
        starts = [s.start_s for s in reply.parts.values() if s.start_s is not None]
        if not starts:
            return None
        onset = reply.start_s
        if onset is None:
            onset = min(starts) + self.engine.config.timing.lead_silence_s
        return max(onset, self.engine.playout.played() - self._lag())

    def _posture_wait(self, reply: _Reply, now: float, second: float, clip: Clip) -> str | None:
        """Why the posture waits for a later try, None when it may go now."""
        if self._listening:
            return "the person speaks"
        turning = self._turning()
        if turning is not None:
            return turning
        start = now + max(0.0, self._lead(second))
        moving = self.engine.scheduler.reservations.holder("arm", now)
        if moving is not None and moving.priority < Priority.IDLE:
            return f"the arm plays {moving.performance.intent}"
        end = start + clip.duration_s
        for waiting in self.engine.pending("arm"):
            if waiting.start < end + RETURN_AFTER_S:
                return f"{waiting.intent} waits for the arm"
        ahead = self._lead(reply.end_s) - (start - now) - clip.duration_s
        if ahead < POSTURE_MIN_S:
            return f"{max(0.0, ahead):.1f} s of the reply's speech known after it"
        return None

    def _posture_hold(self, reply: _Reply, arm: ArmPart, up_at: float, now: float) -> float:
        """The posture's hold from ``up_at`` (engine clock): ``POSTURE_TAIL_S`` past the reply's
        audio known, within the board's ``hold_max_s``."""
        want = now + self._lead(reply.end_s) + POSTURE_TAIL_S
        return round(min(arm.motion.hold_max_s, max(0.1, want - up_at)), 3)

    def _send_posture(self, reply: _Reply, clip: Clip, second: float) -> None:
        self.engine.request(Request(POSTURE, AtStream(round(second, 4), reply.turn),
                                    Priority.MOMENT, "moment", rank=moment_rank(POSTURE),
                                    clips=(clip,)))  # fmt: skip

    def _keep(self, now: float) -> None:
        """A due posture tried again; the one up kept (:meth:`_keep_posture`); the held arm's
        breath (:meth:`_arm_sway`)."""
        for reply in list(self._replies.values()):
            if reply.posture == "due":
                self._raise(reply)
        self._keep_posture(now)
        self._arm_sway(now)

    def _arm_sway(self, now: float) -> None:
        """While a unit holds the arm up and nothing moves it (``SWAY_GAP_S`` after its last
        motion, none waiting), ``arm_sway`` over the held pose, its hold kept: only when the
        hold outlasts the sway by ``SWAY_MARGIN_S`` (a posture renewed in time after it)."""
        if not (self._conversation and self.engine.on) or self._driving or self._quiet:
            return
        arm = self.engine.parts.get("arm")
        if now < self._sway_at or self._listening or arm is None or not _holds(arm):
            return
        reservations = self.engine.scheduler.reservations
        last = reservations.last("arm")
        if reservations.holder("arm", now) is not None or (
            last is not None and now < last.ends + SWAY_GAP_S
        ):
            return
        pre, post = self._reach(ARM_SWAY, "arm")
        if pre == post == 0.0:
            return
        start = now + self._latency("arm")
        end = start + pre + post
        if any(w.start < end + RETURN_AFTER_S for w in self.engine.pending("arm")):
            return
        if isinstance(arm, ArmPart):
            lasts = arm.held_until
        else:
            lasts = last.ends + last.clip.hold_s if last is not None else now
        if lasts < end + SWAY_MARGIN_S:
            return
        self._ask(ARM_SWAY, None, Priority.IDLE, "idle", {"arm"}, start + pre, now)
        self._sway_at = end + SWAY_GAP_S

    def _keep_posture(self, now: float) -> None:
        """The posture held while its reply goes on: sent again before its hold lapses while
        more of the reply's audio is known, and back to ``ready`` after a mark's motion left the
        arm elsewhere; over once the arm is down (the person, the reply's end or cut, a body
        turn, its hold lapsed with the audio)."""
        posture = self._posture
        if posture is None:
            return
        reply = self._replies.get(posture.turn)
        arm = self.engine.parts.get("arm")
        if reply is None or posture.turn not in self._speaking or not isinstance(arm, ArmPart):
            self._posture = None
            return
        reservations = self.engine.scheduler.reservations
        if now < posture.until or reservations.holder("arm", now) is not None:
            return  # its last clip is on its way, or a clip moves the arm (it keeps the hold)
        held = arm.holding()
        if held is None:
            self._posture = None
            reply.posture = "down"
            return
        last = reservations.last("arm")
        if last is not None and now < last.ends + RETURN_AFTER_S:
            return
        away = any(abs(held.get(j, v) - v) >= SAME_POSE_DEG for j, v in posture.pose.items())
        want = now + self._lead(reply.end_s) + POSTURE_TAIL_S
        if away:
            if want - now < POSTURE_MIN_S:
                return  # the reply's audio is nearly out: the hold lapses with it
        elif arm.held_until - now > RENEW_AHEAD_S or want < arm.held_until + RENEW_GAIN_S:
            return
        clip = pose_clip(arm, held, posture.pose, 0.0, POSTURE_CAPS)
        up_at = now + arm.latency_s + clip.duration_s
        if any(w.start < up_at + RETURN_AFTER_S for w in self.engine.pending("arm")):
            return  # a clip waiting for the arm comes first: it keeps the hold, plans from it
        clip = replace(clip, hold_s=self._posture_hold(reply, arm, up_at, now))
        self._send_posture(reply, clip, self.engine.playout.played() - self._lag())
        posture.until = up_at
        posture.renewals += 1
        self._counts["arm_hold"] += 1

    def _posture_end(self, reply: _Reply, why: str, *, rest: bool = True, due: bool = True) -> None:
        """The reply's posture over: one still due never rises (with ``due``); one up goes to
        rest (with ``rest``; else the caller rests the arm)."""
        if due and reply.posture == "due":
            reply.posture = "done"
            self._moment_log(POSTURE, why, turn=reply.turn)
        posture = self._posture
        if posture is None or posture.turn != reply.turn:
            return
        self._posture = None
        reply.posture = "down"
        if rest:
            try:
                self.engine.release_arm(why)
            except Exception:
                logger.exception("alive: the arm would not rest")

    def _shift(self) -> None:
        """The still shift of the head while the person speaks (``listen_shift_deg``)."""
        m = self.knobs.moments
        now = self.clock()
        if self._shift_at is None or now + 1e-6 < self._shift_at:
            return
        self._shift_at = None
        if not (self._conversation and self._listening and self.engine.on):
            return
        if m.listen_shift_deg > 0.0 and self._quiet is None:
            amp = m.listen_shift_deg * self.rng.uniform(0.6, 1.0)
            pan = amp * self.rng.choice((-1.0, 1.0))
            tilt = amp * 0.3 * self.rng.choice((-1.0, 1.0))
            hold = self.rng.uniform(*SHIFT_HOLD_S)
            frames = ((0.0, 0.0, 0.0), (SHIFT_MOVE_S, pan, tilt), (SHIFT_MOVE_S + hold, pan, tilt),
                      (2 * SHIFT_MOVE_S + hold, 0.0, 0.0))  # fmt: skip
            clip = head_clip(frames, SHIFT_MOVE_S, layer="idle")
            self.engine.request(Request("listen_shift", None, Priority.IDLE, "idle",
                                        clips=(clip,)))  # fmt: skip
            self._counts["listen_shift"] += 1
        self._shift_at = now + self.rng.uniform(*m.listen_shift_s)
        self._at(self._shift_at, self._shift)

    def _attend(self, now: float) -> None:
        """Attention read: the first clear face of the conversation brings a smile."""
        if self._smiled or not self.engine.on:
            return
        view = self._view()
        if view is None or view.first_face_s is None or view.first_face_s < self._opened_s - 1.0:
            return
        self._smiled = True
        hold = self.knobs.moments.smile_s
        clip = Clip("face", 0.0, 0.0, hold, events=(Event(0.0, "smile", hold),))
        self.engine.request(Request("smile", None, Priority.MOMENT, "moment",
                                    rank=moment_rank(FACE_FOUND), clips=(clip,)))  # fmt: skip
        self._counts["smile"] += 1
        self._moment_log(FACE_FOUND, None)

    # -- the head's ledger and the engine's numbers ----------------------------------------------

    def _occupy(self, start: float, end: float, stroke: float, back: float | None) -> None:
        """A head move of the listeners' own from ``start`` to ``end`` (engine clock): its
        stroke (and the return of a held one) for the gap, its span for the head's time."""
        self._spans.append((start, end))
        self._strokes.append(stroke)
        if back is not None:
            self._returns.append(back)

    def _head_free(
        self, start: float, end: float, stroke: float, gap: float, *, mark_gap: float | None = None
    ) -> str | None:
        """Why a discrete head move from ``start`` to ``end`` with its stroke at ``stroke``
        (engine clock) may not go: a move of the listeners' own over that time, a stroke within
        ``gap`` (a held move's return within ``head_gap_s``), another's head move (played, or
        waiting for its word in the engine) over it or its stroke within ``gap`` (a marker's
        within ``mark_gap``); None when it may."""
        for a, b in self._spans:
            if start < b and end > a:
                return f"the head is busy {b - start:.2f} s more"
        for t in self._strokes:
            if abs(stroke - t) < gap:
                return f"a head stroke {abs(stroke - t):.1f} s away (gap {gap:g})"
        back_gap = min(gap, self.knobs.moments.head_gap_s)
        for t in self._returns:
            if abs(stroke - t) < back_gap:
                return f"a head move's return {abs(stroke - t):.1f} s away (gap {back_gap:g})"
        self._remember()
        last = self._head_seen
        if last is not None:
            if start < last.ends and end > last.started:
                return f"the head plays {last.performance.intent}"
            limit = gap if mark_gap is None or last.priority > Priority.MARK else max(gap, mark_gap)
            away = abs(stroke - last.stroke)
            if away < limit:
                return f"{last.performance.intent}'s head stroke {away:.1f} s away"
        for waiting in self.engine.pending("head"):
            if waiting.intent in MINE or waiting.priority > Priority.MOMENT:
                continue
            if start < waiting.end and end > waiting.start:
                return f"the head waits for {waiting.intent}"
            marked = mark_gap is not None and waiting.priority <= Priority.MARK
            limit = max(gap, mark_gap) if marked and mark_gap is not None else gap
            away = abs(stroke - waiting.stroke)
            if away < limit:
                return f"{waiting.intent}'s head stroke {away:.1f} s away (waiting)"
        return None

    def _retry_at(self, start: float, end: float, stroke: float, gap: float) -> float:
        """The earliest stroke from ``stroke`` on for a move of that shape that keeps clear of
        the listeners' own moves and ``gap`` from their strokes and returns."""
        lead, length = stroke - start, end - start
        later = stroke
        for _ in range(len(self._spans) + len(self._strokes) + len(self._returns) + 1):
            moved = False
            for a, b in self._spans:
                if later - lead < b and later - lead + length > a:
                    later, moved = b + lead, True
            for t in [*self._strokes, *self._returns]:
                if abs(later - t) < gap:
                    later, moved = t + gap, True
            if not moved:
                break
        return later

    def _lead(self, stream_s: float) -> float:
        """The seconds until stream second ``stream_s`` is heard."""
        return stream_s + self._lag() - self.engine.playout.played()

    def _lag(self) -> float:
        return self.engine.config.audio_lag_s

    def _latency(self, part: PartName) -> float:
        executor = self.engine.parts.get(part)
        return float(executor.latency_s) if executor is not None else 0.0

    def _view(self) -> Attention | None:
        attention = self.engine.attention
        if attention is None:
            return None
        try:
            return attention.attention()
        except Exception:
            logger.exception("alive: the attention view broke")
            return None

    def _update_quiet(self, reply: _Reply | None) -> None:
        """The quiet rule now: the hours, the person's voice, the reply's style."""
        q = self.knobs.quiet
        why: str | None = None
        lo, hi = q.hours
        hour = self.hour()
        if lo != hi and ((lo < hi and lo <= hour < hi) or (lo > hi and (hour >= lo or hour < hi))):
            why = f"the hours {lo:02d}-{hi:02d}"
        elif self._peak is not None and self._peak < q.peak_db:
            why = f"their voice {self._peak:.0f} dBFS"
        else:
            latest = reply or (self._replies[max(self._replies)] if self._replies else None)
            style = latest.style.lower() if latest is not None else ""
            words = set(re.split(r"[\s,]+", style)) & set(q.styles)
            if words:
                why = f"the style {', '.join(sorted(words))}"
        if why != self._quiet:
            self._quiet = why
            self._event("quiet", quiet=why is not None, why=why or "")

    # -- the log ---------------------------------------------------------------------------------

    def _event(self, kind: str, **fields: Any) -> None:
        journal = self.journal
        if journal is not None:
            try:
                journal(kind, self.clock(), **fields)
            except Exception:
                logger.exception("alive: the journal broke")

    def _moment_log(self, name: str, why: str | None, **fields: Any) -> None:
        self._event("moment", name=name, played=why is None, why=why or "", **fields)

    def _log_part(self, reply: _Reply, index: int, state: _PartState) -> None:
        self._event("pulse", turn=reply.turn, index=index, nuclei=len(state.nuclei),
                    z=[round(n.z, 1) for n in state.nuclei], accent=state.accent,
                    skipped=state.skipped[:4], melody_keys=len(state.keys),
                    overlays=len(state.segments))  # fmt: skip

    def _log_reply(self, reply: _Reply) -> None:
        """The reply's last rows: its parts not yet logged and the posture that never rose,
        with its last reason."""
        for index, state in reply.parts.items():
            if not state.complete and state.nuclei:
                self._log_part(reply, index, state)
        if reply.posture == "due":
            reply.posture = "done"
            self._moment_log(POSTURE, reply.posture_why or "the reply ended first", turn=reply.turn)
        if self._posture is not None and self._posture.turn == reply.turn:
            self._posture = None

    def _summary(self, reason: str) -> None:
        now = self.clock()
        minutes = max(1e-9, (now - self._opened_s) / 60.0)
        speech_min = self._speech_total / 60.0
        cpu = self.prosody.cpu_s - self._cpu0[0]
        audio = self.prosody.audio_s - self._cpu0[1]
        heads = self._head_moves + sum(self._counts[k] for k in ("question", "think_glance",
                                                                  "aversion"))  # fmt: skip
        accents = sum(self._counts[k] for k in (WEAK, STRONG, PEAK))
        per_min = {
            "accents_per_speech_min": round(accents / speech_min, 1) if speech_min > 0 else 0.0,
            "head_moves_per_min": round(heads / minutes, 1),
        }  # fmt: skip
        ms = round(cpu * 1000.0 / audio, 2) if audio > 0 else 0.0
        self._event("alive", reason=reason, minutes=round(minutes, 2),
                    speech_s=round(self._speech_total, 1), counts=dict(self._counts),
                    **per_min, prosody_ms_per_s=ms)  # fmt: skip
        if self.say is not None:
            played = ", ".join(f"{k} {v}" for k, v in self._counts.items() if v)
            self.say(f"  (alive: {played or 'nothing'}; {per_min['head_moves_per_min']} head"
                     f" moves a minute; prosody {ms} ms a second of voice)")  # fmt: skip


Key = tuple[float, float]  # a melody key: (stream second, tilt degrees)


def melody_pieces(
    keys: Sequence[Key], busy: Sequence[tuple[float, float]], earliest: float, gap: float = 0.25
) -> list[list[Key]]:
    """``keys`` (the first one where the head is) as the overlays to send: none over a head move
    of the listeners' own (``busy`` spans; after one the head starts again from the aim,
    ``RESUME_S`` past its end), none starting before ``earliest`` (the first sent on time; a
    piece that would starts there, on the line between its keys), each at most ``SEGMENT_S``
    long and sharing its first key with the end of the one before; inside each, keys at least
    ``gap`` apart (:func:`spaced`: the tilt never turns back sooner)."""
    pieces: list[list[Key]] = []
    current: list[Key] = []
    before: Key | None = None  # the key the head is on the way from
    after: float | None = None  # the end of the head move the melody comes back from

    def flush() -> None:
        if len(current) >= 2 and current[-1][0] - current[0][0] > 0.05:
            pieces.append(spaced(current, gap))

    for t, tilt in keys:
        if any(a - RESUME_S < t < b + RESUME_S for a, b in busy):
            flush()
            current = []
            before = None
            after = max(b for a, b in busy if a - RESUME_S < t < b + RESUME_S)
            continue
        if t < earliest:
            before = (t, tilt)
            continue
        if not current:
            if before is not None and before[0] < earliest:
                current = [(earliest, _between(before, (t, tilt), earliest))]
            elif before is not None:
                current = [before]
            elif after is not None:
                start = max(after + RESUME_S, earliest)
                if t - start > 0.05:
                    current = [(start, 0.0)]
        current.append((t, tilt))
        before = (t, tilt)
        if current[-1][0] - current[0][0] >= SEGMENT_S:
            flush()
            current = [(t, tilt)]
    flush()
    return pieces


def spaced(keys: Sequence[Key], gap: float) -> list[Key]:
    """``keys`` with each at least ``gap`` after the one kept before it (a key too close is
    left out; the last is kept, the ones too close before it left out instead), so a tilt
    through them never turns back sooner than ``gap`` (one move alone has no turn back)."""
    out = [keys[0]]
    for key in keys[1:-1]:
        if key[0] - out[-1][0] >= gap:
            out.append(key)
    while len(out) > 1 and keys[-1][0] - out[-1][0] < gap:
        out.pop()
    out.append(keys[-1])
    return out


def _between(a: Key, b: Key, t: float) -> float:
    """The value at ``t`` on the straight line from key ``a`` to key ``b``."""
    if b[0] <= a[0]:
        return b[1]
    share = (t - a[0]) / (b[0] - a[0])
    return round(a[1] + (b[1] - a[1]) * min(1.0, max(0.0, share)), 3)
