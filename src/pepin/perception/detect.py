"""Faces and people on the newest frame, at most ``detect_hz`` times a second.

:class:`DetectLoop` is the service's one detector thread: it waits for a frame newer than the
last it saw, keeps one period between starts (:class:`Pacer`; a frame that came meanwhile is
skipped, the newest wins), decodes the JPEG once, runs the face detector on the pixels and the
person detector on the JPEG bytes themselves (Apple Vision decodes on its own; nothing is
re-encoded), and publishes a :class:`pepin.perception.record.Record`. Detection is independent
of ingest: the ring keeps every frame at the sender's rate, the detector takes what it can.

The detectors are protocols (:class:`FaceDetector`, :class:`PersonDetector`), so the loop runs
in the tests on scripted ones and a fake clock. The real ones: SCRFD from the InsightFace pack
(:class:`pepin.face_sensor.backends.Scrfd`, CoreML on the CPU and the Neural Engine, never the
GPU the voice uses) and :class:`VisionPersons`.

Angles: every face and person box is turned into the head angles that centre it through the
frame's own optics (exact atan through the rectified pinhole), with the gaze's own rules for a
person's head in their box (:func:`pepin.expression.finder.body_sighting`).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from pepin.expression.finder import PersonBox, body_sighting, head_yaw_deg
from pepin.face_sensor.geometry import Lens, crop_box
from pepin.face_sensor.models import Detection
from pepin.face_sensor.quality import measure
from pepin.perception.record import Face, Optics, Person, Record
from pepin.perception.ring import Frame, Records, Ring
from pepin.telemetry import LatencyTracker

logger = logging.getLogger(__name__)

Image = NDArray[np.uint8]
STAGES = ("decode", "faces", "persons", "total")
RATE_WINDOW_S = 10.0
IDLE_WAIT_S = 0.1  # the loop's longest sleep: a stop is seen at least this often


class FaceDetector(Protocol):
    """Faces in a BGR picture."""

    def detect(self, image: Image) -> list[Detection]:
        """Every face above the detector's threshold, with its five points."""
        ...


class PersonDetector(Protocol):
    """People's boxes in a JPEG."""

    def detect_jpeg(self, jpeg: bytes, width: int, height: int) -> list[PersonBox]:
        """Every person in the picture ``jpeg`` holds (``width`` x ``height`` pixels)."""
        ...


class VisionPersons:
    """:class:`PersonDetector` over Apple Vision's human rectangles (full bodies) read straight
    from the JPEG bytes; Vision picks the Neural Engine. ``ImportError`` from :meth:`warm`
    without pyobjc-framework-Vision (group macos)."""

    def __init__(self) -> None:
        self._vision: Any = None
        self._nsdata: Any = None

    def warm(self) -> None:
        """Import Vision now, so the first frame costs nothing extra."""
        if self._vision is None:
            import Vision  # type: ignore[import-not-found,unused-ignore]
            from Foundation import NSData  # type: ignore[import-not-found,unused-ignore]

            self._vision, self._nsdata = Vision, NSData

    def detect_jpeg(self, jpeg: bytes, width: int, height: int) -> list[PersonBox]:
        """Vision's boxes (normalised, origin at the bottom left) as pixels."""
        self.warm()
        vision = self._vision
        data = self._nsdata.dataWithBytes_length_(jpeg, len(jpeg))
        handler = vision.VNImageRequestHandler.alloc().initWithData_options_(data, {})
        request = vision.VNDetectHumanRectanglesRequest.alloc().init()
        request.setUpperBodyOnly_(False)
        done, _error = handler.performRequests_error_([request], None)
        if not done:
            return []
        boxes = []
        for found in request.results() or []:
            box = found.boundingBox()
            x, y = float(box.origin.x), float(box.origin.y)
            bw, bh = float(box.size.width), float(box.size.height)
            top, bottom = (1.0 - y - bh) * height, (1.0 - y) * height
            boxes.append(
                PersonBox(x * width, top, (x + bw) * width, bottom, float(found.confidence()))
            )
        return boxes


def lens_of(optics: Optics) -> Lens:
    """The frame's optics as the face sensor's lens (no distortion)."""
    o = optics
    return Lens(o.fx, o.fy, o.cx, o.cy, o.width, o.height, (), o.name)


def faces_of(
    detections: Sequence[Detection], image: Image, optics: Optics, min_score: float = 0.0
) -> tuple[Face, ...]:
    """The detections at ``min_score`` or more as record faces: the angles that centre each,
    its five-point yaw, and its crop's sharpness and brightness."""
    faces = []
    for det in detections:
        if det.score < min_score:
            continue
        pan, tilt = optics.bearing_deg(*det.centre)
        quality = measure(crop_box(image, det.box))
        yaw = head_yaw_deg(det)
        faces.append(
            Face(det.box, det.score, det.kps, pan, tilt, yaw, quality.sharpness, quality.brightness)
        )
    return tuple(faces)


def persons_of(boxes: Sequence[PersonBox], optics: Optics, head_frac: float) -> tuple[Person, ...]:
    """Person boxes as record persons, the head placed by the gaze's own rule."""
    lens = lens_of(optics)
    out = []
    for b in boxes:
        s = body_sighting(b, lens, head_frac=head_frac)
        out.append(
            Person(
                box=(b.x0, b.y0, b.x1, b.y1),
                score=b.score,
                pan_deg=s.pan_deg,
                half_width_deg=s.half_width_deg,
                head_pan_deg=s.head_pan_deg,
                head_tilt_deg=s.head_tilt_deg,
                height_deg=s.height_deg,
                cut=s.cut,
            )
        )
    return tuple(out)


def decode_jpeg(jpeg: bytes) -> Image | None:
    """A JPEG as BGR pixels, None when it is not one."""
    import cv2

    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    return None if image is None else np.asarray(image, dtype=np.uint8)


class Pacer:
    """At most ``hz`` starts a second: the next start is due one period after the last."""

    def __init__(self, hz: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.hz = float(hz)
        self._clock = clock
        self._last: float | None = None

    def wait_s(self) -> float:
        """Seconds until the next start is due (0: now)."""
        if self._last is None or self.hz <= 0:
            return 0.0
        return max(0.0, self._last + 1.0 / self.hz - self._clock())

    def started(self) -> None:
        """A start happened now."""
        self._last = self._clock()


class Rate:
    """Events per second over the last ``window_s`` of ``clock``."""

    def __init__(
        self, window_s: float = RATE_WINDOW_S, clock: Callable[[], float] = time.monotonic
    ):
        self.window_s = window_s
        self._clock = clock
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def add(self) -> None:
        """One event now."""
        with self._lock:
            self._times.append(self._clock())
            self._trim()

    def per_s(self) -> float:
        """The rate over the window (over the time since the first event while shorter)."""
        with self._lock:
            self._trim()
            if len(self._times) < 2:
                return 0.0
            span = max(self._clock() - self._times[0], 1e-6)
            return (len(self._times) - 1) / min(span, self.window_s)

    def _trim(self) -> None:
        oldest = self._clock() - self.window_s
        while self._times and self._times[0] < oldest:
            self._times.popleft()


class DetectLoop:
    """The detector thread's work: the newest frame, paced to ``hz``, through both detectors
    into a record (see the module)."""

    def __init__(
        self,
        ring: Ring,
        records: Records,
        faces: FaceDetector | None,
        persons: PersonDetector | None,
        *,
        hz: float = 10.0,
        head_frac: float = 0.12,
        min_score: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        decode: Callable[[bytes], Image | None] = decode_jpeg,
    ) -> None:
        """``faces`` / ``persons`` None: that detector does not run."""
        self.ring, self.records = ring, records
        self.faces, self.persons = faces, persons
        self.head_frac, self.min_score = head_frac, min_score
        self.pacer = Pacer(hz, clock)
        self._clock = clock
        self.decode = decode  # also the service's, for an embedding's frame
        self._seen = 0  # the newest frame number taken
        self.timing = {stage: LatencyTracker(stage) for stage in STAGES}
        self.rate = Rate(clock=clock)
        self.detected = 0
        self.skipped = 0  # frames the pace left behind (a newer one was taken)
        self.failed = 0
        self.last_error = ""

    @property
    def hz(self) -> float:
        """The pace: detections a second at most."""
        return self.pacer.hz

    def reset_timing(self) -> None:
        """Forget the stage timings (after a warm-up)."""
        self.timing = {stage: LatencyTracker(stage) for stage in STAGES}

    def tick(self) -> Record | None:
        """Detect the newest frame now when it is new and the pace allows; the record, or
        None when nothing was due."""
        frame = self.ring.newest()
        if frame is None or frame.seq <= self._seen or self.pacer.wait_s() > 0.0:
            return None
        return self._take(frame)

    def run(self, stop: threading.Event) -> None:
        """The thread: wait for a new frame, wait out the pace, take the newest, detect; until
        ``stop`` is set."""
        while not stop.is_set():
            if self.ring.wait_after(self._seen, IDLE_WAIT_S) is None:
                continue
            wait = self.pacer.wait_s()
            if wait > 0.0 and stop.wait(min(wait, IDLE_WAIT_S)):
                return
            frame = self.ring.newest()
            if frame is None or frame.seq <= self._seen or self.pacer.wait_s() > 0.0:
                continue
            try:
                self._take(frame)
            except Exception as exc:  # a detector that raises costs one frame, not the thread
                self.failed += 1
                self.last_error = f"{type(exc).__name__}: {exc}"[:200]
                logger.exception("detection failed on frame %d", frame.seq)

    def detect(self, frame: Frame) -> Record:
        """``frame`` through both detectors into its record (no pacing, nothing published)."""
        t0 = time.perf_counter()
        image = self.decode(frame.jpeg)
        t1 = time.perf_counter()
        if image is None:
            raise ValueError(f"frame {frame.seq} is not a JPEG")
        faces: tuple[Face, ...] = ()
        if self.faces is not None:
            faces = faces_of(self.faces.detect(image), image, frame.optics, self.min_score)
        t2 = time.perf_counter()
        persons: tuple[Person, ...] = ()
        if self.persons is not None:
            height, width = image.shape[:2]
            boxes = self.persons.detect_jpeg(frame.jpeg, int(width), int(height))
            persons = persons_of(boxes, frame.optics, self.head_frac)
        t3 = time.perf_counter()
        ms = {"decode": t1 - t0, "faces": t2 - t1, "persons": t3 - t2, "total": t3 - t0}
        for stage, seconds in ms.items():
            self.timing[stage].add(seconds)
        return Record(
            seq=frame.seq,
            stamp=frame.stamp,
            arrived=frame.arrived,
            frame=frame.frame,
            run=frame.run,
            optics=frame.optics.name,
            brightness=frame.brightness,
            faces=faces,
            persons=persons,
            ms={k: v * 1e3 for k, v in ms.items()},
        )

    def _take(self, frame: Frame) -> Record:
        self.pacer.started()
        self.skipped += max(0, frame.seq - self._seen - 1) if self._seen else 0
        self._seen = frame.seq
        record = self.detect(frame)
        self.records.publish(record)
        self.detected += 1
        self.rate.add()
        return record
