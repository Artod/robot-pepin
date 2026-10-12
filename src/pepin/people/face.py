"""Who a face is: the speaker's face picked out of a picture, judged worth believing, embedded.

Faces come from the head camera's pictures the expression finder takes
(:class:`pepin.expression.finder.SnapshotFinder`; each sighting carries its detection and the
frame its picture). While the speaker gaze tracks the person, the face it centred is the
speaker's; otherwise the widest face is, but only when it is ``dominance`` times wider than the
next (a crowd of alike faces names nobody). A face counts only when :func:`gate` passes it:
``min_px`` wide, the detector's score, the five-point yaw (a profile is no evidence), a crop
neither blurred nor too dark or bright (:mod:`pepin.face_sensor.quality`).

The identity net (ArcFace alone, :class:`pepin.face_sensor.backends.ArcFace`) is loaded on
demand by :class:`IdentityNet` and let go by :meth:`IdentityNet.unload`; MEASURED (2026-10-12,
scratch/people_1011/REPORT.md): 0.1 s to load with a warm CoreML cache, +340 MB resident,
1.7 ms a face on CoreML.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pepin.expression.finder import head_yaw_deg
from pepin.expression.speaker import FaceFrame, FaceSighting
from pepin.face_sensor.geometry import crop_box
from pepin.face_sensor.identity import Vector
from pepin.face_sensor.models import Detection, Embedder
from pepin.face_sensor.quality import measure
from pepin.people.config import FaceKnobs

logger = logging.getLogger(__name__)

Image = NDArray[np.uint8]


@dataclass(frozen=True)
class Shot:
    """One face to identify: the picture it is in, its detection, and whether it is the
    speaker's (the gaze's tracked face, or the dominant one)."""

    image: Image
    detection: Detection
    speaker: bool

    @property
    def width(self) -> float:
        """The box width in pixels."""
        return self.detection.width


def gate(shot: Shot, knobs: FaceKnobs) -> str:
    """``ok``, or why the face is no evidence (``small``, ``doubtful``, ``turned``, ``blurry``,
    ``dark``, ``bright``, ``no landmarks``)."""
    det = shot.detection
    if len(det.kps) < 5:
        return "no landmarks"
    if det.width < knobs.min_px:
        return "small"
    if det.score < knobs.min_det:
        return "doubtful"
    yaw = head_yaw_deg(det)
    if yaw is not None and abs(yaw) > knobs.max_yaw_deg:
        return "turned"
    q = measure(crop_box(shot.image, det.box))
    if q.brightness < knobs.min_brightness:
        return "dark"
    if q.brightness > knobs.max_brightness:
        return "bright"
    if q.sharpness < knobs.min_sharpness:
        return "blurry"
    return "ok"


def strong(shot: Shot, knobs: FaceKnobs) -> bool:
    """A face close enough (``strong_px``) to overrule a disagreeing voice."""
    return shot.width >= knobs.strong_px


def shots(frame: FaceFrame, speaker: FaceSighting | None, dominance: float) -> list[Shot]:
    """Every face of ``frame`` that carries its detection, the speaker's marked: ``speaker``
    when the gaze picked one, else the widest when it is ``dominance`` times the next."""
    image = frame.image
    if image is None:
        return []
    faces = [f for f in frame.faces if f.detection is not None]
    if not faces:
        return []
    chosen: FaceSighting | None = None
    if speaker is not None:
        chosen = next((f for f in faces if f is speaker or f == speaker), None)
    else:
        ranked = sorted(faces, key=lambda f: f.width_px, reverse=True)
        if len(ranked) == 1 or ranked[0].width_px >= dominance * ranked[1].width_px:
            chosen = ranked[0]
    out = []
    for f in faces:
        assert f.detection is not None
        out.append(Shot(image, f.detection, f is chosen))
    return out


class IdentityNet:
    """The identity net, loaded on the first use (or :meth:`warm`) and let go by
    :meth:`unload`; thread-safe."""

    def __init__(self, load: Callable[[], Embedder], clock: Callable[[], float] = time.monotonic):
        """``load`` builds the net (0.1 s warm; seconds the first time)."""
        self._load, self._clock = load, clock
        self._net: Embedder | None = None
        self._lock = threading.Lock()
        self.loads = 0
        self.last_used = -1e9

    @property
    def loaded(self) -> bool:
        """Whether the net is in memory."""
        return self._net is not None

    @property
    def name(self) -> str:
        """The net's model name (loads it)."""
        return self._get().name

    def warm(self) -> None:
        """Load it now."""
        self._get()

    def embed(self, shot: Shot) -> Vector:
        """The unit-length embedding of ``shot``'s face."""
        net = self._get()
        self.last_used = self._clock()
        return net.embed(shot.image, shot.detection)

    def unload(self) -> bool:
        """Let the net go; whether it was loaded."""
        with self._lock:
            was = self._net is not None
            self._net = None
            return was

    def _get(self) -> Embedder:
        with self._lock:
            if self._net is None:
                t0 = time.perf_counter()
                self._net = self._load()
                self.loads += 1
                logger.info("identity net loaded in %.2f s", time.perf_counter() - t0)
            return self._net
