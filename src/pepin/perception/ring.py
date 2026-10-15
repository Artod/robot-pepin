"""The service's memory: the last seconds of frames by number, and the newest record.

:class:`Ring` keeps every frame camera_stream sent for ``keep_s`` (by arrival on this Mac), as
it came: the JPEG bytes, the board's stamp, the optics. Putting a frame is a dict insert under a
lock, so the receiving thread never waits on anything heavy. The detector takes the newest frame
it has not seen (:meth:`Ring.wait_after`); ``see`` takes the newest frame exposed after the
head settled (:meth:`Ring.wait_newest`), waiting for one when the newest is older.

:class:`Records` holds the newest detected :class:`pepin.perception.record.Record` and answers
the long poll of ``/latest``: the newest record past a number and an exposure stamp, waited for.
Frame numbers are the service's own, so a restarted camera_stream (its own numbers from zero)
never collides with the frames already kept; the sender's numbers travel in each record. They
start at the service's start in milliseconds, so they also grow across a restart of the service.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from pepin.perception.record import Optics, Record

KEEP_S = 10.0
MAX_FRAMES = 600  # a burst after a WiFi stall cannot grow the ring past this (30 fps x 20 s)


@dataclass(frozen=True)
class Frame:
    """One frame as received: the service's number, the board's exposure stamp, this Mac's
    arrival (wall clock), the sender's number and run, the optics, the mean brightness and the
    JPEG bytes."""

    seq: int
    stamp: float
    arrived: float
    frame: int
    run: str
    optics: Optics
    brightness: float
    jpeg: bytes


class Ring:
    """The frames of the last ``keep_s`` seconds by number; thread-safe, with waits."""

    def __init__(
        self,
        keep_s: float = KEEP_S,
        *,
        clock: Callable[[], float] = time.time,
        max_frames: int = MAX_FRAMES,
        first: int | None = None,
    ) -> None:
        """``clock`` stamps arrivals (wall clock: reports compare it with the board's).
        ``first``: the number before the first frame's; None starts from this moment in
        milliseconds, so a restarted service numbers its frames past every number the one
        before it gave (frames come far slower than one a millisecond) and a client's
        ``since`` from before the restart never hides the new frames."""
        self.keep_s, self.max_frames = float(keep_s), int(max_frames)
        self._clock = clock
        self._frames: OrderedDict[int, Frame] = OrderedDict()
        self._seq = int(time.time() * 1000) if first is None else int(first)
        self._changed = threading.Condition()
        self.received = 0

    def put(
        self,
        jpeg: bytes,
        *,
        stamp: float,
        optics: Optics,
        frame: int = 0,
        run: str = "",
        brightness: float = 0.0,
    ) -> Frame:
        """Keep one frame under the next number; frames older than ``keep_s`` leave."""
        with self._changed:
            self._seq += 1
            kept = Frame(
                self._seq, float(stamp), self._clock(), int(frame), run, optics, brightness, jpeg
            )
            self._frames[kept.seq] = kept
            self.received += 1
            oldest = kept.arrived - self.keep_s
            while self._frames and (
                len(self._frames) > self.max_frames
                or next(iter(self._frames.values())).arrived < oldest
            ):
                self._frames.popitem(last=False)
            self._changed.notify_all()
            return kept

    def get(self, seq: int) -> Frame | None:
        """Frame ``seq``, or None once it has left the ring (or never came)."""
        with self._changed:
            return self._frames.get(seq)

    def newest(self) -> Frame | None:
        """The newest frame, None before the first."""
        with self._changed:
            return self._last()

    def wait_after(self, seq: int, timeout_s: float) -> Frame | None:
        """The newest frame numbered after ``seq``, waiting up to ``timeout_s`` for one."""
        with self._changed:
            self._changed.wait_for(lambda: self._seq > seq, timeout=max(0.0, timeout_s))
            last = self._last()
            return last if last is not None and last.seq > seq else None

    def wait_newest(self, min_stamp: float | None, timeout_s: float) -> Frame | None:
        """The newest frame exposed at ``min_stamp`` or later (any frame for None), waiting up
        to ``timeout_s`` for one when the newest is older; None when none came in time."""

        def found() -> bool:
            last = self._last()
            return last is not None and (min_stamp is None or last.stamp >= min_stamp)

        with self._changed:
            if self._changed.wait_for(found, timeout=max(0.0, timeout_s)):
                return self._last()
            return None

    def span_s(self) -> float:
        """How many seconds of arrivals the ring holds now."""
        with self._changed:
            if len(self._frames) < 2:
                return 0.0
            first = next(iter(self._frames.values()))
            return self._frames[self._seq].arrived - first.arrived

    def __len__(self) -> int:
        with self._changed:
            return len(self._frames)

    def _last(self) -> Frame | None:
        return self._frames.get(self._seq)


class Records:
    """The newest detected record, and the long poll on it; thread-safe."""

    def __init__(self) -> None:
        self._latest: Record | None = None
        self._changed = threading.Condition()

    def publish(self, record: Record) -> None:
        """``record`` is the newest; every waiter is woken."""
        with self._changed:
            self._latest = record
            self._changed.notify_all()

    def latest(self) -> Record | None:
        """The newest record, None before the first."""
        with self._changed:
            return self._latest

    def wait(self, since: int, min_stamp: float | None, timeout_s: float) -> Record | None:
        """The newest record numbered after ``since`` and exposed at ``min_stamp`` or later,
        waiting up to ``timeout_s`` for one; None when none came in time."""

        def found() -> bool:
            r = self._latest
            return r is not None and r.seq > since and (min_stamp is None or r.stamp >= min_stamp)

        with self._changed:
            if self._changed.wait_for(found, timeout=max(0.0, timeout_s)):
                return self._latest
            return None
