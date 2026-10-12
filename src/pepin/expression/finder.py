"""Faces and people on demand for the speaker gaze: one picture of the head camera, searched by
the face detector and, when one runs, a body detector.

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
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from pepin.expression.speaker import BodySighting, FaceFrame, FaceSighting
from pepin.face_sensor.geometry import Lens
from pepin.face_sensor.models import Detection, Detector

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
    pyobjc-framework-Vision (``uv add --group face pyobjc-framework-Vision``)."""

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

    def look(self) -> FaceFrame | None:
        """The faces in a fresh picture (each with its detection, the frame with the picture);
        None when no picture came. One look at a time: the speaker gaze and person recognition
        share the finder."""
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
