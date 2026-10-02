"""Which camera frames are for nothing: taken while the head turned, or while the body spun.

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

A frame's exposure window is its stamp plus and minus ``exposure_s``: whether ustreamer's stamp
marks the start of the exposure or its end is not known, so both sides are covered.

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

# The knobs every gated node carries under these names in config/knobs.json, whose defaults are
# these (tests/unit/test_knobs.py holds every block equal to them).
GATE_KNOBS = ("gate_exposure_s", "gate_settle_s", "gate_yaw_dps")
EXPOSURE_S = 0.035  # half a 15 fps frame: an auto exposure indoors is unknown (gaze.md 4.4)
SETTLE_S = 0.1  # the contract's "settled + one frame period": 0.105 s at 9.5 fps
YAW_DPS = 0.0  # off: nothing has measured where the body's own turn starts to blur a frame

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

    def text(self, now: float) -> str:
        """The state for a report line: the phase, since when, blind or not, intervals closed."""
        last = self.last
        if last is None:
            return "no /gaze/state yet: every frame passes"
        open_ = "open" if self._open is not None else "none open"
        stale = ", STALE (the arbiter is silent: nothing is blind)" if self.stale(now) else ""
        return (
            f"head {last.phase or '?'} since {last.since:.2f}{' blind' if last.blind else ''}"
            f" ({last.source or 'no source'}), {self.intervals} saccades, {open_}{stale}"
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


class FrameGate:
    """The verdict on one frame by its stamp: :data:`BLIND`, :data:`SPINNING` or ``None`` (use
    it). Thread-safe: states and IMU samples arrive on one thread, frames are judged on another.
    """

    def __init__(
        self, exposure_s: float = EXPOSURE_S, settle_s: float = SETTLE_S, yaw_dps: float = YAW_DPS
    ) -> None:
        self.exposure_s = exposure_s  # live: gate_exposure_s
        self.yaw_dps = yaw_dps  # live: gate_yaw_dps; 0 is off
        self._blind = BlindLog(settle_s)
        self._yaw = YawLog()
        self._lock = threading.Lock()

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

    def verdict(self, stamp: float, now: float) -> str | None:
        """Why the frame stamped ``stamp`` (board seconds) is for nothing, or ``None``."""
        t0, t1 = stamp - self.exposure_s, stamp + self.exposure_s
        with self._lock:
            if self._blind.covers(t0, t1, now):
                return BLIND
            if self.yaw_dps > 0.0:
                peak = self._yaw.peak(t0, t1)
                if peak is not None and peak > math.radians(self.yaw_dps):
                    return SPINNING
        return None

    def text(self, now: float) -> str:
        """The gate's state for a report line."""
        with self._lock:
            head = self._blind.text(now)
        yaw = f"yaw gate {self.yaw_dps:g} deg/s" if self.yaw_dps > 0.0 else "yaw gate off"
        return (
            f"{head}; exposure +-{self.exposure_s * 1e3:.0f} ms, settle"
            f" +{self.settle_s * 1e3:.0f} ms, {yaw}"
        )


__all__ = [
    "BLIND",
    "EXPOSURE_S",
    "GATE_KNOBS",
    "GAZE_GATE",
    "GAZE_STATE_TOPIC",
    "SETTLE_S",
    "SPINNING",
    "STATE_STALE_S",
    "YAW_DPS",
    "BlindLog",
    "FrameGate",
    "GazeState",
    "YawLog",
    "board_seconds",
]
