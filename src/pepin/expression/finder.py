"""Faces and people on demand for the speaker gaze: one picture of the head camera, searched by
the face detector and, when one runs, a body detector.

THE ONE CAMERA CHANNEL (:class:`ServiceFinder`, the robot's default since 2026-10-12): the
perception service (:mod:`pepin.perception_service`) already found the faces and the people on
camera_stream's rectified left eye; a look is its newest record exposed after the head settled,
and this process holds no picture and runs no network. The live switch ``vision.source``
(:class:`SwitchedFinder`, :mod:`pepin.perception.source`) goes back to the snapshot path below
for a day the service is broken. Face sizes then come from two pixel spaces: the rectified eye's
(fx 454.5) in the service's records, the raw eye's (fx 528 at the centre) in a snapshot's; the
gaze's and person recognition's pixel thresholds are the rectified eye's since that day.

THE SNAPSHOT PATH:

A look is one ``GET /snapshot`` from ustreamer on the board (with ``extra_headers=1``: the V4L2
capture stamp), the left eye cut out upright (:func:`pepin.face_sensor.frames.cut_eye`), and
SCRFD alone from the face sensor's InsightFace pack (:class:`pepin.face_sensor.backends.Scrfd`:
no ArcFace, no expression net, no landmarker), so the cascade carries one small network. Each
face's bearing comes through the eye's lens (:meth:`pepin.face_sensor.geometry.Lens.bearing_deg`,
the stereo calibration's raw left eye), and its head yaw, roughly, from the detector's five
points. Nothing runs between looks: the speaker gaze asks at the wake and about once a second
while a conversation lasts.

MEASURED on this Mac (2026-10-11, scratch/expression/finder_probe.py, CoreML): the detector adds
~170 MB to the process (180 MB peak with a cold CoreML cache; FaceAnalysis with identity off
added 550-800 MB, because it opens every network of the pack), loads in 0.1-0.25 s, detects in
6-7 ms (2.7 ms of CPU); a snapshot takes 45 ms (52 ms p90, 42 KB).

PEOPLE (:class:`BodyDetector`, config/expression.json's ``finder.bodies``): a person whose face is
not seen (turned away, lying, far, the face cut by the frame's edge) still gives the gaze an aim
from their box (:func:`body_sighting`: the head ``head_frac`` of the box's height below its top
for an upright box, the box's middle for a lying one, the frame's edge for a box the edge cuts).
:class:`VisionBodies` is Apple Vision's ``VNDetectHumanRectanglesRequest`` (full body), part of
macOS (no model file), through pyobjc-framework-Vision. MEASURED on this Mac (2026-10-12,
scratch/gaze/vision_probe.swift, 57 frames at 800x600): 3.0 ms median, 3.7 ms p90 on the Neural
Engine (17 ms on the CPU), +110-125 MB resident; a box on 14 of the 19 frames with a person in
view, the misses heads or legs cut by the frame's edge.

The weights are config/face_sensor.json's ``detector_pack`` (``models/face/buffalo_l``, its
``det_10g.onnx`` alone is read) under the checkout the cascade runs from; CoreML compiles the
network at the first load into ``coreml_cache``. Where it runs is config/expression.json's
``finder.compute_units``: ``CPUAndNeuralEngine`` keeps it off the GPU, where the voice's Qwen3-TTS
runs (2026-10-12: with the detector free to use the GPU the voice's first chunk took 230-400 ms
instead of 100-150); a network CoreML refuses runs on the CPU.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from pepin.expression.speaker import BodySighting, FaceFinder, FaceFrame, FaceSighting
from pepin.face_sensor.geometry import Lens
from pepin.face_sensor.models import Detection, Detector

if TYPE_CHECKING:
    from pepin.perception.client import Perception
    from pepin.perception.record import Record
    from pepin.perception.source import VisionSource

logger = logging.getLogger(__name__)

Image = NDArray[np.uint8]
SNAPSHOT_TIMEOUT_S = 2.0
# config/expression.json's finder.compute_units (CoreML's names) as a detector device
UNITS_DEVICE = {"CPUAndNeuralEngine": "ane", "ALL": "coreml", "CPUAndGPU": "mps", "CPU": "cpu"}


@dataclass(frozen=True)
class Picture:
    """One eye's picture (BGR) and its capture stamp (the board's realtime clock)."""

    image: Image
    stamp: float


class Snapshots(Protocol):
    """The head camera, one picture on demand."""

    def grab(self) -> Picture | None:
        """The newest picture, None when the camera does not answer."""
        ...


class UstreamerSnapshots:
    """:class:`Snapshots` from ustreamer's ``/snapshot`` on the board, cut to one eye."""

    def __init__(
        self,
        url: str,
        *,
        eye: str = "left",
        upside_down: bool = False,
        timeout_s: float = SNAPSHOT_TIMEOUT_S,
    ) -> None:
        """``url`` is the snapshot's (``http://board:8080/snapshot?extra_headers=1``)."""
        self.url, self.eye, self.upside_down, self.timeout_s = url, eye, upside_down, timeout_s

    def grab(self) -> Picture | None:
        """One snapshot decoded and cut; its capture stamp (else the send stamp, else now)."""
        from pepin.face_sensor.frames import decode_jpeg
        from pepin.mjpeg import capture_time

        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout_s) as response:
                body = response.read()
                headers = {k.lower(): v for k, v in response.headers.items()}
        except OSError as exc:
            logger.warning("no snapshot from %s: %s", self.url, exc)
            return None
        try:
            image = decode_jpeg(body, self.eye, self.upside_down)
        except ValueError as exc:
            logger.warning("a snapshot that is no picture: %s", exc)
            return None
        stamp = capture_time(headers, "grab")
        return Picture(image, stamp if stamp is not None else time.time())


def snapshot_url(stream_url: str) -> str:
    """ustreamer's snapshot URL beside its stream's, with the capture stamp asked for."""
    base = stream_url.split("?", 1)[0]
    if base.endswith("/stream"):
        base = base[: -len("/stream")]
    return base.rstrip("/") + "/snapshot?extra_headers=1"


@dataclass(frozen=True)
class PersonBox:
    """One person's box in a picture: pixels (x right, y down) and the detector's score."""

    x0: float
    y0: float
    x1: float
    y1: float
    score: float


class BodyDetector(Protocol):
    """People's boxes in one picture."""

    def detect(self, image: Image) -> list[PersonBox]:
        """Every person found in ``image`` (BGR)."""
        ...


LYING = 1.3  # a box this many times wider than tall is a person lying (or sitting sideways)
CUT_PX = 2.0  # a box's top this near the frame's top edge is cut by it


def body_sighting(box: PersonBox, lens: Lens, *, head_frac: float) -> BodySighting:
    """``box`` through ``lens``: its centre's pan and half width, and the head estimated in it
    (``head_frac`` of the height below an upright box's top; a lying box's middle, the head
    being at one end; the top edge itself when the frame cuts the box there)."""
    width, height = box.x1 - box.x0, box.y1 - box.y0
    u, middle = (box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0
    cut = box.y0 <= CUT_PX
    if cut:
        v = box.y0
    elif width > LYING * height:
        v = middle
    else:
        v = box.y0 + head_frac * height
    pan, _ = lens.bearing_deg(u, middle)
    left, _ = lens.bearing_deg(box.x0, middle)
    head_pan, head_tilt = lens.bearing_deg(u, v)
    _, top = lens.bearing_deg(u, box.y0)
    _, bottom = lens.bearing_deg(u, box.y1)
    return BodySighting(pan, abs(left - pan), head_pan, head_tilt, bottom - top, box.score, cut)


class VisionBodies:
    """:class:`BodyDetector` over Apple Vision's human rectangles (full bodies), on the device
    (Vision picks the Neural Engine); ``ImportError`` from :meth:`warm` without
    pyobjc-framework-Vision (``uv add --group macos pyobjc-framework-Vision``)."""

    def __init__(self) -> None:
        self._vision: Any = None

    def warm(self) -> None:
        """Import Vision now (the first detection then costs nothing extra)."""
        if self._vision is None:
            import Foundation  # type: ignore[import-not-found,unused-ignore]  # noqa: F401
            import Vision  # type: ignore[import-not-found,unused-ignore]

            self._vision = Vision

    def detect(self, image: Image) -> list[PersonBox]:
        """Vision's boxes (normalised, origin at the bottom left) as pixels of ``image``."""
        import cv2
        from Foundation import NSData  # type: ignore[import-not-found,unused-ignore]

        self.warm()
        vision = self._vision
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return []
        data = NSData.dataWithBytes_length_(jpeg.tobytes(), int(jpeg.size))
        handler = vision.VNImageRequestHandler.alloc().initWithData_options_(data, {})
        request = vision.VNDetectHumanRectanglesRequest.alloc().init()
        request.setUpperBodyOnly_(False)
        done, _error = handler.performRequests_error_([request], None)
        if not done:
            return []
        h, w = image.shape[:2]
        boxes = []
        for found in request.results() or []:
            box = found.boundingBox()
            x, y = float(box.origin.x), float(box.origin.y)
            bw, bh = float(box.size.width), float(box.size.height)
            top, bottom = (1.0 - y - bh) * h, (1.0 - y) * h
            boxes.append(PersonBox(x * w, top, (x + bw) * w, bottom, float(found.confidence())))
        return boxes


def head_yaw_deg(face: Detection) -> float | None:
    """How far a face is turned from the camera, from the five points (eyes, nose, mouth
    corners): the nose's offset from the eyes' midpoint against half their spacing; 0 faces the
    camera, + the face's left. A rough reading (about +-15 deg), None without the points."""
    if len(face.kps) < 3:
        return None
    (lx, _), (rx, _), (nx, _) = face.kps[0], face.kps[1], face.kps[2]
    half = (rx - lx) / 2.0
    if abs(half) < 1e-6:
        return None
    ratio = (nx - (lx + rx) / 2.0) / half
    return math.degrees(math.asin(max(-1.0, min(1.0, ratio))))


@dataclass
class Timing:
    """The last look's milliseconds: the picture, the detector."""

    grab_ms: float = 0.0
    detect_ms: float = 0.0
    body_ms: float = 0.0
    looks: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)


class SnapshotFinder:
    """:class:`pepin.expression.speaker.FaceFinder` over one snapshot and the detector, loaded
    on :meth:`warm` (the switch turned on) or at the first look; ``bodies`` adds people's boxes
    (dropped, said once, when it cannot load)."""

    def __init__(
        self,
        snapshots: Snapshots,
        load: Callable[[], Detector],
        lens: Lens,
        *,
        min_score: float = 0.0,
        bodies: BodyDetector | None = None,
        head_frac: float = 0.12,
    ) -> None:
        """``load`` builds the detector (seconds, once); ``min_score`` drops weaker faces;
        ``head_frac``: where the head sits in an upright person's box, a share of its height
        below the top."""
        self.snapshots, self._load, self.lens, self.min_score = snapshots, load, lens, min_score
        self.bodies, self.head_frac = bodies, head_frac
        self._detector: Detector | None = None
        self._lock = threading.Lock()
        self._looking = threading.Lock()
        self.timing = Timing()

    def warm(self) -> None:
        """Load the detectors now."""
        self._model()
        warm = getattr(self.bodies, "warm", None)
        if callable(warm):
            try:
                warm()
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                logger.warning("no body detector (%s): faces only", exc)
                self.bodies = None

    def look(self, not_before: float | None = None) -> FaceFrame | None:
        """The faces in a fresh picture (each with its detection, the frame with the picture);
        None when no picture came. One look at a time: the speaker gaze and person recognition
        share the finder. ``not_before`` is not needed: the snapshot is taken now, after the
        arbiter said the head is still."""
        with self._looking:
            return self._look()

    def _look(self) -> FaceFrame | None:
        t0 = time.perf_counter()
        picture = self.snapshots.grab()
        t1 = time.perf_counter()
        if picture is None:
            return None
        detections = self._model().detect(picture.image)
        t2 = time.perf_counter()
        boxes = self.bodies.detect(picture.image) if self.bodies is not None else []
        t3 = time.perf_counter()
        t = self.timing
        t.grab_ms, t.detect_ms, t.body_ms = (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3
        t.looks += 1
        t.history = [*t.history[-59:], (t.grab_ms, t.detect_ms)]
        height, width = picture.image.shape[:2]
        lens = self.lens.scaled(int(width), int(height))
        bodies = tuple(body_sighting(b, lens, head_frac=self.head_frac) for b in boxes)
        faces = []
        for det in detections:
            if det.score < self.min_score:
                continue
            pan, tilt = lens.bearing_deg(*det.centre)
            faces.append(
                FaceSighting(
                    pan, tilt, det.width, det.score, yaw_deg=head_yaw_deg(det), detection=det
                )
            )
        return FaceFrame(picture.stamp, tuple(faces), image=picture.image, bodies=bodies)

    def _model(self) -> Detector:
        with self._lock:
            if self._detector is None:
                t0 = time.perf_counter()
                self._detector = self._load()
                device = getattr(self._detector, "device", "?")
                logger.info("face detector loaded in %.1f s (%s)", time.perf_counter() - t0, device)
            return self._detector


def frame_of(record: Record, min_score: float = 0.0) -> FaceFrame:
    """A perception service record as the gaze's frame: no picture, the frame's number, each
    face with its detection (box, five points) and its crop's measures, each person's box in
    degrees as the service placed it."""
    faces = tuple(
        FaceSighting(
            f.pan_deg,
            f.tilt_deg,
            f.width_px,
            f.score,
            yaw_deg=f.yaw_deg,
            detection=Detection(f.box, f.score, f.kps),
            sharpness=f.sharpness,
            brightness=f.brightness,
        )
        for f in record.faces
        if f.score >= min_score
    )
    bodies = tuple(
        BodySighting(
            pan_deg=p.pan_deg,
            half_width_deg=p.half_width_deg,
            head_pan_deg=p.head_pan_deg,
            head_tilt_deg=p.head_tilt_deg,
            height_deg=p.height_deg,
            score=p.score,
            cut=p.cut,
        )
        for p in record.persons
    )
    return FaceFrame(record.stamp, faces, image=None, bodies=bodies, seq=record.seq)


class ServiceFinder:
    """:class:`pepin.expression.speaker.FaceFinder` over the perception service's records: the
    one camera channel's faces and people, found by the service on camera_stream's rectified
    left eye; nothing is decoded or detected in this process.

    A look is the newest record past the last one this finder returned and exposed at
    ``not_before`` or later (the arbiter's ``blind_until``: board time against board time, so a
    frame exposed while the head still moved is never taken), waited for up to ``wait_s``; None
    when none came (the service down, camera_stream's feed off, the head never still in time)."""

    def __init__(self, service: Perception, *, wait_s: float = 0.4, min_score: float = 0.0) -> None:
        """``wait_s``: how long a look waits for a record (a frame exposed at the settle reaches
        the service 70-100 ms later and is detected within one detection period)."""
        self.service, self.wait_s, self.min_score = service, wait_s, min_score
        self._seq = 0
        self._looking = threading.Lock()
        self.timing = Timing()
        self.missed = 0

    def warm(self) -> None:
        """Nothing loads here: one ``/health`` read says in the log whether the service is up
        (a service that is down is waited for, not given up on: it may start later)."""
        if self.service.health() is None:
            logger.warning(
                "the perception service at %s does not answer: no faces until it does"
                " (ros/models.sh start perception)",
                self.service.url,
            )

    def look(self, not_before: float | None = None) -> FaceFrame | None:
        """The faces of the newest record past the last look and ``not_before``; None when none
        came within ``wait_s``. One look at a time (the gaze and person recognition share it)."""
        with self._looking:
            t0 = time.perf_counter()
            record = self.service.latest(self._seq, not_before, self.wait_s)
            t = self.timing
            t.grab_ms = (time.perf_counter() - t0) * 1e3
            if record is None:
                self.missed += 1
                return None
            self._seq = record.seq
            t.detect_ms = float(record.ms.get("faces", 0.0))
            t.body_ms = float(record.ms.get("persons", 0.0))
            t.looks += 1
            t.history = [*t.history[-59:], (t.grab_ms, t.detect_ms)]
            return frame_of(record, self.min_score)


class SwitchedFinder:
    """:class:`pepin.expression.speaker.FaceFinder` that asks the live ``vision.source`` at
    every look (:mod:`pepin.perception.source`): the service's records, or the snapshot finder
    (the path before the service, built the first time it is wanted; its detectors load at its
    first look)."""

    def __init__(
        self,
        source: VisionSource,
        service: FaceFinder,
        snapshot: Callable[[], FaceFinder],
    ) -> None:
        """``snapshot`` builds the snapshot finder (config reads only)."""
        self.source, self.service = source, service
        self._build_snapshot = snapshot
        self._snapshot: FaceFinder | None = None

    @property
    def current(self) -> FaceFinder:
        """The finder the switch names now."""
        if self.source.service:
            return self.service
        if self._snapshot is None:
            self._snapshot = self._build_snapshot()
        return self._snapshot

    def warm(self) -> None:
        """Warm the finder in use (the snapshot finder loads its detectors here)."""
        warm = getattr(self.current, "warm", None)
        if callable(warm):
            warm()

    def look(self, not_before: float | None = None) -> FaceFrame | None:
        """The current finder's look."""
        return self.current.look(not_before=not_before)


def robot_finder(
    host: str | None, source: VisionSource, service: Perception, *, wait_s: float = 0.4
) -> SwitchedFinder:
    """The finder on the robot: the perception service's records, switched live to the
    snapshot finder of :func:`board_finder` by ``vision.source``."""
    return SwitchedFinder(source, ServiceFinder(service, wait_s=wait_s), lambda: board_finder(host))


def finder_device(path: str | Path | None = None) -> str:
    """config/expression.json's ``finder.compute_units`` as a detector device (``ane`` when
    unset); ``ValueError`` for a name CoreML does not have."""
    if path is None:
        from pepin.deployment import config_file

        path = config_file("expression.json")
    units = _finder_block(path).get("compute_units")
    if units is None:
        return "ane"
    if units not in UNITS_DEVICE:
        raise ValueError(f"finder.compute_units is one of {', '.join(UNITS_DEVICE)}, not {units!r}")
    return UNITS_DEVICE[units]


def board_finder(host: str | None = None) -> SnapshotFinder:
    """The finder on the robot: config/camera.json's snapshot, config/face_sensor.json's pack
    (detector only) and lens, config/expression.json's compute units and body detector
    (``finder.bodies``: ``vision`` or ``none``); the detectors load on
    :meth:`SnapshotFinder.warm`."""
    from pepin.audio_link import board_host
    from pepin.camera import CameraConfig
    from pepin.deployment import config_file
    from pepin.face_sensor.config import FaceSensorConfig

    cfg = FaceSensorConfig.load()
    device = finder_device()
    cam = CameraConfig.load(config_file("camera.json"), board=host or board_host())
    eye = cfg.source.eye
    upside_down = bool(cam.rig and cam.rig.upside_down and eye != "full")
    snapshots = UstreamerSnapshots(snapshot_url(cam.stream), eye=eye, upside_down=upside_down)
    calibration = cfg.path(cfg.stereo_calibration)
    if eye != "full" and calibration.exists():
        lens = Lens.stereo_eye(calibration, eye)
    else:
        lens = Lens.nominal(800, 600, cfg.hfov_deg)

    block = _finder_block()
    bodies: BodyDetector | None = VisionBodies() if block.get("bodies") == "vision" else None

    def load() -> Detector:
        from pepin.face_sensor.backends import Scrfd

        m = cfg.models
        return Scrfd(
            cfg.path(m.detector_pack),
            device=device,
            cache=cfg.path(m.coreml_cache),
            det_size=m.det_size,
            det_threshold=m.det_threshold,
        )

    head_frac = float(block.get("body_head_frac", 0.12))
    return SnapshotFinder(snapshots, load, lens, bodies=bodies, head_frac=head_frac)


def _finder_block(path: str | Path | None = None) -> dict[str, Any]:
    """config/expression.json's ``finder`` block."""
    if path is None:
        from pepin.deployment import config_file

        path = config_file("expression.json")
    block = json.loads(Path(path).read_text()).get("finder", {})
    return block if isinstance(block, dict) else {}
