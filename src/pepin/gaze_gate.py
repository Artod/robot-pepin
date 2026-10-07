"""Which camera frames are for nothing: taken while the head turned, the body spun, or too dark.

A saccade sweeps the picture at hundreds of degrees a second, and the neck's transform is sampled
between ticks, so a frame exposed during one is blurred AND placed wrong: in the volume it paints
smears, in RTAB-Map it is a keyframe of a moving head, in the visual odometry a pan read as the
cart's own yaw. The cure is not a better estimate of the frame but a choice of frames — shoot
between the propeller blades: every frame consumer asks, by the frame's own stamp, whether its
exposure window overlaps a moment the head was moving, and drops or holds it if so.

THE ONE SIGNAL is ``/gaze/state`` (the gaze arbiter's, design gaze.md 3.4): a ``std_msgs/String``
carrying a JSON object with ``phase``, ``pan_rad``, ``tilt_rad``, ``since`` (the board-clock stamp
of the phase change, seconds), ``request_id``, ``source`` and ``blind`` (true from the write of a
saccade until it settled plus one frame period). The board's clock is also the frames' clock, so
a blind interval is compared with a frame's stamp directly. :class:`BlindLog` turns the stream
of states into intervals: one opens at the ``since`` of the first blind state and closes at the
latest ``since`` seen before blind cleared — the phase the head settled into — plus ``settle_s``,
the contract's "one frame period", so a lost message can shorten nothing. An interval still open
covers every frame after its start while the states keep coming; a stream silent for
:data:`STATE_STALE_S` blinds nobody (an arbiter that died mid-saccade must not blind the robot).
Without any ``/gaze/state`` there is no interval and every frame passes: today's behaviour.

THE BODY'S SPIN is the other blur: a frame whose window holds an IMU sample with ``|wz|`` above
``yaw_dps`` (:class:`YawLog`, base_link's z from ``/imu/data_raw``). 0 turns it off.

A frame's exposure window is its stamp plus and minus ``exposure_s`` by default: ustreamer's
X-Timestamp is the SEND time, 1-68 ms after the capture (2026-10-02), so neither side is known.
With grab stamps (camera_stream's ``camera_stamp grab``) the stamp sits a fixed interval AFTER the
exposure ends, and ``stamp_end`` (knob ``gate_stamp_end``) makes the window
``[stamp - 2 * exposure_s, stamp + GATE_MARGIN_S]``: the knob's half window doubled covers a whole
auto exposure up to 70 ms behind the stamp (:meth:`FrameGate.window`).

THE MAST'S SWAY is the third (vio.md section 5): ``/mast/state`` from the base bridge, the sway
rate and angle the head gyro sees beyond the base's yaw and the neck's joints (NaN while the neck
moves or the head link is silent). Small sway is KEPT — the bridge's TF corrects it — and a frame
whose window holds a sway rate above ``sway_dps`` or an angle above ``sway_deg`` is dropped
(:data:`SWAYING`, :class:`SwayLog`). Both 0 is off, as shipped.

A FRAME TOO DARK is the fourth (:data:`DARK`, :class:`DarkLog`): under-exposed stereo pairs match
noise, and a phantom is what matched noise looks like in the volume (2026-10-06, a still head on a
dark doorway: 8.3 ms manual frames made 156 camera-only lethal births in 20 s against 10-30 under
the auto exposure, with RTAB-Map's stereo inliers 0-1 against 260). The camera reports no exposure
per frame, so the witness is the picture: camera_stream measures every frame's mean luma
(:func:`frame_brightness`, a 1/8 subsample of the whole side-by-side frame, before the eyes are
cut) and publishes it on :data:`BRIGHTNESS_TOPIC` under the frame's own stamp. A frame under
``floor`` grey is dark, and so is every frame after it until one reaches ``floor + hyst``: the
eyes wait for the auto exposure to catch up after a turn from a window into the room. The rule is
per consumer (``dark_on``, the ``gate_dark`` flag of depth_stream and sensor_pack,
:data:`GATE_DARK`): the volume, the costmap's clearing fan and RTAB-Map are fed through those two,
while the visual odometry's relay never turns it on (OpenVINS still tracked 41 of 54-68 features
on the dark frames, and the VIO keeper watches it). A frame whose brightness was never heard
passes and is counted.

Nothing here is ROS (:class:`pepin_bringup.gaze_feed.GazeFeed` is the subscriber around it) and
nothing here reads a clock: the caller hands in its own monotonic ``now``.
"""

from __future__ import annotations

import json
import math
import threading
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass
from typing import Any

from pepin.flags import Flag

GAZE_STATE_TOPIC = "/gaze/state"
# The contract publishes at 10 Hz and on every change: ten periods without a word is an arbiter
# that has stopped, and an interval it left open no longer blinds anything.
STATE_STALE_S = 1.0
INTERVALS_KEPT = 64  # a minute of saccades; frames are judged within a second of their stamp
YAW_KEPT_S = 3.0  # how far back the IMU's yaw rate is kept for a frame to be judged by
YAW_GAP_TAU = 0.05  # the running mean of the IMU's sample gap: how fast it follows a new rate

BLIND = "blind"  # the head was moving during the frame's exposure
SPINNING = "spinning"  # the body was turning faster than the gate allows
SWAYING = "swaying"  # the mast swung faster or further than the TF's correction is trusted for
DARK = "dark"  # under the darkness floor, or not yet back above it plus the hysteresis

MAST_STATE_TOPIC = "/mast/state"  # the base bridge's sway estimate (sensor_msgs/JointState)
MAST_JOINTS = ("mast_roll", "mast_pitch", "mast_yaw")

# The knobs every gated node carries under these names in config/knobs.json, whose defaults are
# these (tests/unit/test_knobs.py holds every block equal to them).
GATE_KNOBS = (
    "gate_exposure_s",
    "gate_settle_s",
    "gate_yaw_dps",
    "gate_stamp_end",
    "gate_sway_dps",
    "gate_sway_deg",
)
EXPOSURE_S = 0.035  # half a 15 fps frame: an auto exposure indoors is unknown (gaze.md 4.4)
SETTLE_S = 0.1  # the contract's "settled + one frame period": 0.105 s at 9.5 fps
# The two period-keyed knobs at 0 (their default) follow the camera's rate, config/camera.json's
# rate.fps (pepin.camera.follow_period): the window in frame periods and the settle tail in frame
# periods, each the value above at 10 fps, the rate both were set at.
FOLLOW = 0.0
EXPOSURE_PER_PERIOD = 0.35
SETTLE_PER_PERIOD = 1.0
YAW_DPS = 0.0  # off: nothing has measured where the body's own turn starts to blur a frame
STAMP_END = 1  # the stamp is the capture's end since camera_stamp grab (default 2026-10-05)
SWAY_DPS = 6.0  # blur model: 1.5 px / (8.6 px/deg x 30 ms auto exposure); taps 12-21, rest 0.4
SWAY_DEG = 1.0  # beyond the mast filter's trust; a hand-pushed cart over a carpet edge 1.2-1.3
GATE_MARGIN_S = 0.01  # after a grab stamp: the capture stamp sits after the exposure ends
GATE_DEFAULTS = (FOLLOW, FOLLOW, YAW_DPS, STAMP_END, SWAY_DPS, SWAY_DEG)

# The darkness rule's knobs, carried only by the nodes that turn the rule on (depth_stream and
# sensor_pack; config/knobs.json, held equal to these by tests/unit/test_knobs.py).
BRIGHTNESS_TOPIC = "/camera/brightness"  # camera_stream's per-frame mean luma, by frame stamp
# depth_stream's word to the gaze arbiter: one std_msgs/Header per frame its gate dropped as dark,
# at the frame's own stamp (a look waiting for frames learns they are not coming, and why).
DARK_FRAME_TOPIC = "/depth/dark"
DARK_KNOBS = ("gate_dark_floor", "gate_dark_hyst")
DARK_FLOOR = 10.0  # grey, the whole frame: above the bench's collapse, under every drive frame
DARK_HYST = 2.0  # grey: a picture hovering at the floor (frame to frame p50 0.3-0.5) stays put
DARK_DEFAULTS = (DARK_FLOOR, DARK_HYST)
BRIGHTNESS_KEPT = 64  # samples: six seconds at 10 fps; a frame is judged within a second
# A frame's brightness is the sample with its own stamp: camera_stream stamps both from one value,
# so they agree to the nanosecond; the tolerance only absorbs a float's rounding.
BRIGHTNESS_MATCH_S = 0.001
LUMA_BGR = (0.114, 0.587, 0.299)  # ITU-R BT.601, OpenCV's own BGR -> grey weights
BRIGHTNESS_STEP = 8  # every 8th row and column: 200x75 samples of a 1600x600 frame

GAZE_GATE = Flag(
    "gaze_gate",
    True,
    description="frames whose exposure window (stamp +- gate_exposure_s) overlaps a head"
    " saccade (/gaze/state's blind intervals, + gate_settle_s after settling) or holds a body"
    " yaw faster than gate_yaw_dps (/imu/data_raw; 0 is off) are dropped here and counted; off,"
    " every frame passes as before",
    why="on by design, unmeasured: without /gaze/state (no arbiter) there is no interval and"
    " every frame passes, so on is today's behaviour until something publishes the state. The"
    " arithmetic it rests on (design gaze.md 4.4): blur = rate x exposure x 8.6 px/deg, 21 px at"
    " 250 deg/s and 10 ms, while SGBM tolerates 1-2 px",
    on_when="whenever the gaze arbiter moves the head: saccade frames are for nothing",
    off_when="to measure what a saccade does to the volume, RTAB-Map or the visual odometry"
    " without the gate (phase 2 of gaze.md), or if the arbiter's blind flag is wrong and frames"
    " are lost; ros/gaze_gate.sh off turns it off in every gated node at once",
)

GATE_DARK = Flag(
    "gate_dark",
    True,
    description="a frame whose mean luma (/camera/brightness, camera_stream's per frame 1/8"
    " subsample of the whole stereo frame) is under gate_dark_floor grey is dropped here and"
    " counted as dark, and so is every frame after it until one reaches gate_dark_floor +"
    " gate_dark_hyst: it writes nothing into the volume, the clearing fan or RTAB-Map, and a"
    " look's still frames do not advance on it; only under gaze_gate. The visual odometry never"
    " carries it. Off, the brightness is not consulted and every frame the other rules pass goes"
    " on as before",
    why="on by Artem's decision 2026-10-06 after the camera exposure bench: a still head on a"
    " dark doorway with 8.3 ms manual frames made 156 camera-only lethal births in 20 s (33 ms:"
    " 96, 67 ms: 34) against 10-30 under the auto exposure, RTAB-Map's stereo inliers 0-1 (33 ms:"
    " 45) against 260 at mean grey 57, and back on auto both recovered within 1 s. The floor"
    " sits where the drives' own stereo starves, not where the room is dim (config/knobs.json's"
    " gate_dark_floor note, scratch/depth_dark_floor/)",
    on_when="whenever the head camera runs on its auto exposure in rooms that can go dark: the"
    " frames under the floor are the ones a stereo matcher fills with noise",
    off_when="to measure what the dark frames do to the volume without the rule, or when the"
    " depth starves in a dim room and the frames under the floor turn out to be worth keeping",
)


def frame_brightness(frame: Any, step: int = BRIGHTNESS_STEP) -> float:
    """The mean luma of a decoded frame (a BGR or grey ``numpy`` array), 0-255, from every
    ``step``-th row and column: the auto exposure's witness, the same number live and replayed."""
    sub = frame[::step, ::step]
    if sub.ndim == 2:
        return float(sub.mean())
    means = sub.reshape(-1, sub.shape[2]).mean(axis=0)
    return float(sum(w * float(m) for w, m in zip(LUMA_BGR, means[:3], strict=False)))


def board_seconds(value: Any) -> float | None:
    """A stamp as seconds: a number, or ``{"sec": .., "nanosec": ..}``; ``None`` otherwise."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict) and "sec" in value:
        try:
            return float(value["sec"]) + float(value.get("nanosec", 0)) * 1e-9
        except (TypeError, ValueError):
            return None
    return None


@dataclass(frozen=True)
class GazeState:
    """One ``/gaze/state`` message: what the head is doing, since when, and whether frames taken
    now are blind."""

    phase: str
    blind: bool
    since: float
    pan_rad: float | None = None
    tilt_rad: float | None = None
    request_id: str = ""
    source: str = ""

    @classmethod
    def from_json(cls, text: str) -> GazeState | None:
        """The state a message carries, or ``None`` when it is not one (no JSON object, no
        boolean ``blind``, no ``since`` stamp). Unknown fields are ignored."""
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            return None
        if not isinstance(data, dict) or not isinstance(data.get("blind"), bool):
            return None
        since = board_seconds(data.get("since"))
        if since is None:
            return None
        return cls(
            phase=str(data.get("phase", "")),
            blind=data["blind"],
            since=since,
            pan_rad=_number(data.get("pan_rad")),
            tilt_rad=_number(data.get("tilt_rad")),
            request_id=str(data.get("request_id", "")),
            source=str(data.get("source", "")),
        )


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class BlindLog:
    """The head's blind intervals on the board's clock, rebuilt from the states as they come.

    A closed interval is ``(start, settled)``: ``settle_s`` is added when a frame is judged, so a
    live change of the knob reaches every interval kept."""

    def __init__(self, settle_s: float = SETTLE_S, keep: int = INTERVALS_KEPT) -> None:
        self.settle_s = settle_s  # live: the owning node's gate_settle_s writes it
        self._closed: deque[tuple[float, float]] = deque(maxlen=keep)
        self._open: float | None = None  # the start of the interval being lived through
        self._latest = -math.inf  # the latest ``since`` seen inside it
        self._heard: float | None = None  # the caller's monotonic clock at the last state
        self.last: GazeState | None = None
        self.intervals = 0  # how many have closed since the start, for the report line

    def observe(self, state: GazeState, now: float) -> None:
        """One state, received at ``now`` (the caller's monotonic seconds)."""
        self._heard = now
        self.last = state
        if state.blind:
            if self._open is None:
                self._open, self._latest = state.since, state.since
            else:
                self._latest = max(self._latest, state.since)
            return
        if self._open is not None:
            settled = max(self._latest, state.since, self._open)
            self._closed.append((self._open, settled))
            self._open, self._latest = None, -math.inf
            self.intervals += 1

    def stale(self, now: float) -> bool:
        """Whether the stream has been silent for :data:`STATE_STALE_S` (or never spoke)."""
        return self._heard is None or now - self._heard > STATE_STALE_S

    def covers(self, t0: float, t1: float, now: float) -> bool:
        """Whether ``[t0, t1]`` (board seconds) overlaps a blind interval; an open one counts
        only while the stream is alive."""
        if self._open is not None and t1 >= self._open and not self.stale(now):
            return True
        tail = self.settle_s
        return any(t1 >= start and t0 <= settled + tail for start, settled in self._closed)

    def text(self, now: float, window: int | None = None) -> str:
        """The state for a report line: the phase, since when, blind or not, and the intervals
        closed — ``window`` of them in the caller's report window (the window its blind count
        is of), :attr:`intervals` since the start."""
        last = self.last
        if last is None:
            return "no /gaze/state yet: every frame passes"
        open_ = "open" if self._open is not None else "none open"
        stale = ", STALE (the arbiter is silent: nothing is blind)" if self.stale(now) else ""
        closed = (
            f"{self.intervals} saccades"
            if window is None
            else f"{window} saccades in this window ({self.intervals} since the start)"
        )
        return (
            f"head {last.phase or '?'} since {last.since:.2f}{' blind' if last.blind else ''}"
            f" ({last.source or 'no source'}), {closed}, {open_}{stale}"
        )


class YawLog:
    """The last seconds of the body's yaw rate (rad/s about base_link's z), by IMU stamp."""

    def __init__(self, keep_s: float = YAW_KEPT_S) -> None:
        self._keep_s = keep_s
        self._stamps: list[float] = []
        self._rates: list[float] = []
        self._gap: float | None = None  # the running mean of the sample gap

    def observe(self, stamp: float, yaw_rate: float) -> None:
        """One sample; an out-of-order or non-finite one is ignored."""
        if not (math.isfinite(stamp) and math.isfinite(yaw_rate)):
            return
        if self._stamps and stamp <= self._stamps[-1]:
            return
        if self._stamps:
            gap = stamp - self._stamps[-1]
            self._gap = gap if self._gap is None else self._gap + YAW_GAP_TAU * (gap - self._gap)
        self._stamps.append(stamp)
        self._rates.append(abs(yaw_rate))
        if len(self._stamps) > 64 and stamp - self._stamps[0] > 2 * self._keep_s:
            cut = bisect_left(self._stamps, stamp - self._keep_s)
            del self._stamps[:cut], self._rates[:cut]

    def peak(self, t0: float, t1: float) -> float | None:
        """The largest ``|wz|`` among the samples within ``[t0, t1]`` widened by one sample gap
        on each side (a short exposure falls between two samples); ``None`` with none there."""
        if not self._stamps:
            return None
        gap = self._gap or 0.0
        lo = bisect_left(self._stamps, t0 - gap)
        hi = bisect_right(self._stamps, t1 + gap)
        return max(self._rates[lo:hi]) if hi > lo else None


@dataclass(frozen=True)
class SwayPeak:
    """What the mast did inside one frame's window: the largest sway rate and angle (vector
    norms, deg/s and deg), or ``held`` when a sample in it said the estimate was off (NaN)."""

    rate_dps: float
    angle_deg: float
    held: bool = False


class SwayLog:
    """The last seconds of ``/mast/state`` by stamp: the sway rate's and angle's norms, NaN kept
    as "held" (the neck moved or the head link was silent: the estimate does not vouch)."""

    def __init__(self, keep_s: float = YAW_KEPT_S) -> None:
        self._keep_s = keep_s
        self._stamps: list[float] = []
        self._rates: list[float] = []  # deg/s, NaN while held
        self._angles: list[float] = []  # deg, NaN while held
        self._gap: float | None = None

    def observe(
        self, stamp: float, angle_rad: tuple[float, ...], rate_rad_s: tuple[float, ...]
    ) -> None:
        """One state: the three sway angles and rates (rad, rad/s); out of order is ignored."""
        if not math.isfinite(stamp) or (self._stamps and stamp <= self._stamps[-1]):
            return
        if self._stamps:
            gap = stamp - self._stamps[-1]
            self._gap = gap if self._gap is None else self._gap + YAW_GAP_TAU * (gap - self._gap)
        self._stamps.append(stamp)
        self._rates.append(math.degrees(_norm(rate_rad_s)))
        self._angles.append(math.degrees(_norm(angle_rad)))
        if len(self._stamps) > 64 and stamp - self._stamps[0] > 2 * self._keep_s:
            cut = bisect_left(self._stamps, stamp - self._keep_s)
            del self._stamps[:cut], self._rates[:cut], self._angles[:cut]

    def peak(self, t0: float, t1: float) -> SwayPeak | None:
        """The sway inside ``[t0, t1]`` widened by one sample gap; ``None`` with no sample."""
        if not self._stamps:
            return None
        gap = self._gap or 0.0
        lo = bisect_left(self._stamps, t0 - gap)
        hi = bisect_right(self._stamps, t1 + gap)
        if hi <= lo:
            return None
        rates, angles = self._rates[lo:hi], self._angles[lo:hi]
        if any(math.isnan(v) for v in (*rates, *angles)):
            return SwayPeak(math.nan, math.nan, held=True)
        return SwayPeak(max(rates), max(angles))


def _norm(values: tuple[float, ...]) -> float:
    """The Euclidean norm, NaN when any component is NaN (or there are none)."""
    if not values or any(math.isnan(v) for v in values):
        return math.nan
    return math.sqrt(sum(v * v for v in values))


class DarkLog:
    """The frames' brightness in stamp order and which of them the darkness rule blinds.

    Each sample is judged as it arrives against the state the one before left (a knob changed
    live applies from the next sample on): under ``floor`` grey a frame is dark and the state
    goes dark; a dark state lasts until a frame reaches ``floor + hyst``, so a picture hovering
    at the floor does not flicker in and out."""

    def __init__(
        self, floor: float = DARK_FLOOR, hyst: float = DARK_HYST, keep: int = BRIGHTNESS_KEPT
    ) -> None:
        self.floor = floor  # live: gate_dark_floor
        self.hyst = hyst  # live: gate_dark_hyst
        self._stamps: deque[float] = deque(maxlen=keep)
        self._dark: deque[bool] = deque(maxlen=keep)
        self._in_dark = False
        self.episodes = 0  # dark spells begun since the start, for the report line
        self.last: float | None = None

    def observe(self, stamp: float, level: float) -> bool | None:
        """One frame's brightness; whether that frame is dark, ``None`` for a sample refused
        (not a number, or not after the last one)."""
        if not (math.isfinite(stamp) and math.isfinite(level)):
            return None
        if self._stamps and stamp <= self._stamps[-1]:
            return None
        if self._in_dark:
            self._in_dark = level < self.floor + self.hyst
        elif level < self.floor:
            self._in_dark = True
            self.episodes += 1
        self._stamps.append(stamp)
        self._dark.append(self._in_dark)
        self.last = level
        return self._in_dark

    def dark(self, stamp: float) -> bool | None:
        """The verdict on the frame stamped ``stamp``, ``None`` when its sample was not heard."""
        stamps = self._stamps
        i = bisect_left(stamps, stamp - BRIGHTNESS_MATCH_S)
        if i < len(stamps) and abs(stamps[i] - stamp) <= BRIGHTNESS_MATCH_S:
            return self._dark[i]
        return None

    @property
    def in_dark(self) -> bool:
        """Whether the last sample left the state dark."""
        return self._in_dark


class FrameGate:
    """The verdict on one frame by its stamp: :data:`BLIND`, :data:`SPINNING`, :data:`SWAYING`,
    :data:`DARK` or ``None`` (use it). Thread-safe: states and IMU samples arrive on one thread,
    frames are judged on another.
    """

    def __init__(
        self,
        exposure_s: float = EXPOSURE_S,
        settle_s: float = SETTLE_S,
        yaw_dps: float = YAW_DPS,
        stamp_end: bool = bool(STAMP_END),
        sway_dps: float = SWAY_DPS,
        sway_deg: float = SWAY_DEG,
        dark_on: bool = False,
        dark: DarkLog | None = None,
    ) -> None:
        self.exposure_s = exposure_s  # live: gate_exposure_s
        self.yaw_dps = yaw_dps  # live: gate_yaw_dps; 0 is off
        self.stamp_end = stamp_end  # live: gate_stamp_end; the stamp is the exposure's end
        self.sway_dps = sway_dps  # live: gate_sway_dps; 0 is off
        self.sway_deg = sway_deg  # live: gate_sway_deg; 0 is off
        self.dark_on = dark_on  # live: gate_dark, only where a node carries it
        self.dark = dark if dark is not None else DarkLog()  # its knobs: gate_dark_*
        self._blind = BlindLog(settle_s)
        self._yaw = YawLog()
        self._sway = SwayLog()
        self._lock = threading.Lock()
        self._reported = 0  # saccades closed as of the last report line (text)
        self._dark_reported = 0  # dark spells begun as of the last report line
        self._unheard = 0  # frames judged with the rule on and no brightness for their stamp

    @property
    def sway_on(self) -> bool:
        """Whether either sway threshold is set (the feed subscribes /mast/state only then)."""
        return self.sway_dps > 0.0 or self.sway_deg > 0.0

    def window(self, stamp: float) -> tuple[float, float]:
        """The exposure window a frame stamped ``stamp`` is judged by (board seconds):
        symmetric ``stamp +- exposure_s``, or under ``stamp_end``
        ``[stamp - 2 * exposure_s, stamp + GATE_MARGIN_S]``."""
        if self.stamp_end:
            return stamp - 2.0 * self.exposure_s, stamp + GATE_MARGIN_S
        return stamp - self.exposure_s, stamp + self.exposure_s

    @property
    def settle_s(self) -> float:
        """Seconds after the settle stamp a blind interval still covers (gate_settle_s)."""
        return self._blind.settle_s

    @settle_s.setter
    def settle_s(self, value: float) -> None:
        self._blind.settle_s = value

    def observe_state(self, state: GazeState, now: float) -> None:
        """A ``/gaze/state`` message, received at ``now`` (monotonic seconds)."""
        with self._lock:
            self._blind.observe(state, now)

    def observe_yaw(self, stamp: float, yaw_rate: float) -> None:
        """An IMU sample: the yaw rate about base_link's z (rad/s) at ``stamp`` (board s)."""
        with self._lock:
            self._yaw.observe(stamp, yaw_rate)

    def observe_sway(
        self, stamp: float, angle_rad: tuple[float, ...], rate_rad_s: tuple[float, ...]
    ) -> None:
        """A ``/mast/state`` sample: the sway angles (rad) and rates (rad/s) at ``stamp``."""
        with self._lock:
            self._sway.observe(stamp, angle_rad, rate_rad_s)

    def observe_brightness(self, stamp: float, level: float) -> bool | None:
        """A frame's mean luma (``/camera/brightness``) at its stamp (board s); whether the
        darkness rule blinds that frame (:meth:`DarkLog.observe`)."""
        with self._lock:
            return self.dark.observe(stamp, level)

    def verdict(self, stamp: float, now: float) -> str | None:
        """Why the frame stamped ``stamp`` (board seconds) is for nothing, or ``None``.

        The sway rule keeps small sway (the TF corrects it) and drops a frame whose window holds
        a rate above ``sway_dps`` or an angle above ``sway_deg``; a held (NaN) sample in the
        window gives no sway verdict (the head was moving: the blind interval covers it), and no
        sample at all gives none either. The darkness rule is asked last, so its count is what
        it adds to the others, and by the frame's own stamp, not a window: the brightness is the
        frame's own measurement."""
        t0, t1 = self.window(stamp)
        with self._lock:
            if self._blind.covers(t0, t1, now):
                return BLIND
            if self.yaw_dps > 0.0:
                peak = self._yaw.peak(t0, t1)
                if peak is not None and peak > math.radians(self.yaw_dps):
                    return SPINNING
            if self.sway_on:
                sway = self._sway.peak(t0, t1)
                if sway is not None and not sway.held:
                    if self.sway_dps > 0.0 and sway.rate_dps > self.sway_dps:
                        return SWAYING
                    if self.sway_deg > 0.0 and sway.angle_deg > self.sway_deg:
                        return SWAYING
            if self.dark_on:
                dark = self.dark.dark(stamp)
                if dark is None:
                    self._unheard += 1
                elif dark:
                    return DARK
        return None

    def text(self, now: float) -> str:
        """The gate's state for a report line, ONE CALL PER REPORT WINDOW: the saccades it
        counts are the ones closed since the previous call, the window the owning node's blind
        and spinning counts are of — a total since the start beside a count per window read as
        saccades the gate saw and let through (2026-10-02, the first live look)."""
        with self._lock:
            closed = self._blind.intervals
            head = self._blind.text(now, window=closed - self._reported)
            self._reported = closed
            dark = self._dark_text()
        yaw = f"yaw gate {self.yaw_dps:g} deg/s" if self.yaw_dps > 0.0 else "yaw gate off"
        t0, t1 = self.window(0.0)
        exposure = (
            f"exposure {t0 * 1e3:.0f}..+{t1 * 1e3:.0f} ms (stamp at its end)"
            if self.stamp_end
            else f"exposure +-{self.exposure_s * 1e3:.0f} ms"
        )
        sway = (
            f", sway gate {self.sway_dps:g} deg/s / {self.sway_deg:g} deg" if self.sway_on else ""
        )
        return f"{head}; {exposure}, settle +{self.settle_s * 1e3:.0f} ms, {yaw}{sway}{dark}"

    def _dark_text(self) -> str:
        """The darkness rule's part of the report line (under the lock): its floor, the dark
        spells begun and the frames it could not judge in this window, the last brightness."""
        if not self.dark_on:
            return ""
        log = self.dark
        begun, self._dark_reported = log.episodes - self._dark_reported, log.episodes
        unheard, self._unheard = self._unheard, 0
        if log.last is None:
            level = f"no {BRIGHTNESS_TOPIC} yet: nothing is dark"
        else:
            level = f"brightness {log.last:.0f}{' (dark)' if log.in_dark else ''}"
        missing = f", {unheard} frames without a brightness (passed)" if unheard else ""
        return (
            f", dark gate under {log.floor:g} grey until {log.floor + log.hyst:g}:"
            f" {begun} dark spells in this window, {level}{missing}"
        )


__all__ = [
    "BLIND",
    "BRIGHTNESS_TOPIC",
    "DARK",
    "DARK_DEFAULTS",
    "DARK_FLOOR",
    "DARK_FRAME_TOPIC",
    "DARK_HYST",
    "DARK_KNOBS",
    "EXPOSURE_S",
    "GATE_DARK",
    "GATE_DEFAULTS",
    "GATE_KNOBS",
    "GATE_MARGIN_S",
    "GAZE_GATE",
    "GAZE_STATE_TOPIC",
    "MAST_JOINTS",
    "MAST_STATE_TOPIC",
    "SETTLE_S",
    "SPINNING",
    "STAMP_END",
    "STATE_STALE_S",
    "SWAYING",
    "SWAY_DEG",
    "SWAY_DPS",
    "YAW_DPS",
    "BlindLog",
    "DarkLog",
    "FrameGate",
    "GazeState",
    "SwayLog",
    "SwayPeak",
    "YawLog",
    "board_seconds",
    "frame_brightness",
]
