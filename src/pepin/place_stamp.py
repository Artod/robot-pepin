"""One place descriptor on every snapshot RTAB-Map receives, computed off the packer's thread and
published in the order the snapshots were packed.

WHY EVERY SNAPSHOT. RTAB-Map compares two nodes' descriptors only when both carry the same number
of them, and ABORTS otherwise (Signature.cpp:252; pepin.global_descriptor has the whole story).
So once one node carries a descriptor, every node must: a camera snapshot gets the localisation
service's ``/place`` vector, and a snapshot with no picture — or whose picture the service did
not describe in time — gets the null descriptor, which RTAB-Map scores as neither like nor unlike
anything. Nothing is ever published without one.

WHY A THREAD. The packer runs inside the node's subscription callbacks, where a blocking call is
a moment in which no picture, depth or scan is read (pepin_bringup.sensor_pack's TF_WAIT_S says
why that matters). A snapshot is handed to one worker thread with its picture; the worker asks
the service with what is left of ``timeout_s`` after the snapshot's wait in the queue, attaches
the answer or the null descriptor, and publishes. One worker, one queue: the snapshots leave in
the order they were packed. A snapshot that has already waited ``timeout_s`` in the queue (the
service slow, the worker behind) is not asked for at all and goes out with the null descriptor,
so the queue drains at the packer's rate whatever the service does.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from pepin.global_descriptor import PlaceDescriptor
from pepin.telemetry import LatencyTracker

Picture = npt.NDArray[np.uint8]
Describe = Callable[[Picture, float], tuple[npt.NDArray[np.float32], str] | None]

# Why a snapshot went out with the null descriptor: the reasons a report line counts.
NO_PICTURE = "no picture"
OFF = "switched off"
LATE = "late"
FAILED = "no answer"
WRONG_LENGTH = "wrong length"


@dataclass
class _Job:
    msg: Any
    picture: Callable[[], Picture | None] | None
    ask: bool
    timeout_s: float
    queued: float


class PlaceStamper:
    """Attaches one descriptor of length ``dim`` to every submitted message and publishes it.

    ``describe(rgb, timeout_s)`` answers ``(unit vector, tag)`` or ``None``; ``attach(msg,
    descriptor)`` puts the descriptor into the message; ``publish(msg)`` sends it. With
    ``threaded`` false every submit is handled at once on the caller's thread (tests)."""

    def __init__(
        self,
        dim: int,
        describe: Describe,
        attach: Callable[[Any, PlaceDescriptor], None],
        publish: Callable[[Any], None],
        threaded: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.dim = int(dim)
        self._describe, self._attach, self._publish = describe, attach, publish
        self._clock = clock
        self._lock = threading.Lock()
        self.described = 0
        self.nulls: Counter[str] = Counter()
        self.tag = ""  # the weights of the last vector the service answered
        self.wrong_length = ""  # what the service answered when its length was not ``dim``
        self.ask_ms = LatencyTracker("place")
        self._queue: queue.Queue[_Job | None] | None = None
        self._thread: threading.Thread | None = None
        if threaded:
            self._queue = queue.Queue()
            self._thread = threading.Thread(target=self._run, name="place_stamper", daemon=True)
            self._thread.start()

    def submit(
        self,
        msg: Any,
        picture: Callable[[], Picture | None] | None,
        ask: bool,
        timeout_s: float,
    ) -> None:
        """Hand one snapshot over: ``picture`` builds its RGB picture (``None``: it has none),
        ``ask`` is whether the service is asked at all, ``timeout_s`` the whole budget from now.
        Never blocks."""
        job = _Job(msg, picture, ask, float(timeout_s), self._clock())
        if self._queue is None:
            self._handle(job)
        else:
            self._queue.put(job)

    @property
    def waiting(self) -> int:
        """Snapshots handed over and not yet published."""
        return 0 if self._queue is None else self._queue.qsize()

    def close(self, timeout_s: float = 2.0) -> None:
        """Publish what is queued (null descriptors from here on) and stop the worker."""
        if self._queue is None or self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout_s)

    def line(self) -> str:
        """For a report line: ``58 described (55/80 ms), null: 2 no picture, 1 late``."""
        with self._lock:
            nulls = ", ".join(f"{n} {why}" for why, n in sorted(self.nulls.items()))
            described = self.described
        s = self.ask_ms.summary()
        text = f"{described} described ({s.median_ms:.0f}/{s.p95_ms:.0f} ms)"
        text += f", null: {nulls}" if nulls else ", no null"
        if self.wrong_length:
            text += f" (the service answered {self.wrong_length}, not {self.dim})"
        return text

    def _run(self) -> None:
        assert self._queue is not None
        closing = False
        while True:
            job = self._queue.get()
            if job is None:
                closing = True
                if self._queue.empty():
                    return
                continue
            if closing:
                job.ask = False
            self._handle(job)
            if closing and self._queue.empty():
                return

    def _handle(self, job: _Job) -> None:
        descriptor, why = self._descriptor(job)
        with self._lock:
            if why is None:
                self.described += 1
            else:
                self.nulls[why] += 1
        self._attach(job.msg, descriptor)
        self._publish(job.msg)

    def _descriptor(self, job: _Job) -> tuple[PlaceDescriptor, str | None]:
        """The descriptor this job gets, and why it is the null one (``None``: it is not)."""
        null = PlaceDescriptor.null(self.dim)
        if job.picture is None:
            return null, NO_PICTURE
        if not job.ask:
            return null, OFF
        left = job.timeout_s - (self._clock() - job.queued)
        if left <= 0.0:
            return null, LATE
        rgb = job.picture()
        if rgb is None:
            return null, NO_PICTURE
        t0 = time.perf_counter()
        answer = self._describe(rgb, left)
        self.ask_ms.add(time.perf_counter() - t0)
        if answer is None:
            return null, FAILED
        vector, tag = answer
        if vector.size != self.dim:
            self.wrong_length = f"{vector.size} from {tag}"
            return null, WRONG_LENGTH
        self.tag = tag
        return PlaceDescriptor(tag, np.ascontiguousarray(vector, dtype=np.float32)), None


def rgb_of(image: Any) -> Picture | None:
    """A sensor_msgs/Image as an RGB H x W x 3 picture (bgr8, rgb8, bgra8, rgba8, mono8), rows
    read at the message's own ``step``; ``None`` for any other encoding."""
    channels = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1, "8UC1": 1}
    n = channels.get(str(image.encoding))
    if n is None or image.height <= 0 or image.width <= 0:
        return None
    raw = np.frombuffer(bytes(image.data), dtype=np.uint8)
    rows = raw[: image.height * image.step].reshape(image.height, image.step)
    pixels = rows[:, : image.width * n].reshape(image.height, image.width, n)
    if n == 1:
        return np.ascontiguousarray(np.repeat(pixels, 3, axis=2))
    if image.encoding.startswith("bgr"):
        return np.ascontiguousarray(pixels[:, :, 2::-1])
    return np.ascontiguousarray(pixels[:, :, :3])
