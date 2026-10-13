"""Who a face is: the speaker's face picked out of a picture, judged worth believing, embedded.

Faces come from the expression finder. On the one camera channel
(:class:`pepin.expression.finder.ServiceFinder`) each sighting carries its detection and its
crop's measures and the frame its number in the perception service's ring: the gate reads the
measures and the service embeds the face (:class:`ServiceIdentity`), so no picture enters this
process. On the snapshot path (:class:`pepin.expression.finder.SnapshotFinder`) the frame
carries its picture and the net runs here (:class:`IdentityNet`); :class:`RoutedIdentity` takes
each shot the way it came. While the speaker gaze tracks the person, the face it centred is the
speaker's; otherwise the widest face is, but only when it is ``dominance`` times wider than the
next (a crowd of alike faces names nobody). A face counts only when :func:`gate` passes it:
``min_px`` wide, the detector's score, the five-point yaw (a profile is no evidence), a crop
neither blurred nor too dark or bright (:mod:`pepin.face_sensor.quality`).

The identity net (ArcFace alone, :class:`pepin.face_sensor.backends.ArcFace`) is loaded on
demand by :class:`IdentityNet` and let go by :meth:`IdentityNet.unload`; MEASURED (2026-10-12,
scratch/people_1011/face_identity.py): 0.1 s to load with a warm CoreML cache (2.4 s the first
time), +340 MB resident, 1.7 ms a face on CoreML.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from pepin.expression.finder import head_yaw_deg
from pepin.expression.speaker import FaceFrame, FaceSighting
from pepin.face_sensor.geometry import crop_box
from pepin.face_sensor.identity import Vector
from pepin.face_sensor.models import Detection, Embedder
from pepin.face_sensor.quality import Quality, measure
from pepin.people.config import FaceKnobs

if TYPE_CHECKING:
    from pepin.perception.client import Perception
    from pepin.perception.source import VisionSource

logger = logging.getLogger(__name__)

Image = NDArray[np.uint8]


@dataclass(frozen=True)
class Shot:
    """One face to identify: the picture it is in (None when the frame stayed in the perception
    service: then its number there and the crop's measures the service took), its detection,
    and whether it is the speaker's (the gaze's tracked face, or the dominant one)."""

    image: Image | None
    detection: Detection
    speaker: bool
    seq: int | None = None
    quality: Quality | None = None

    @property
    def width(self) -> float:
        """The box width in pixels."""
        return self.detection.width


def gate(shot: Shot, knobs: FaceKnobs) -> str:
    """``ok``, or why the face is no evidence (``small``, ``doubtful``, ``turned``, ``blurry``,
    ``dark``, ``bright``, ``no landmarks``, ``unmeasured``: neither a picture nor measures)."""
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
    if shot.quality is not None:
        q = shot.quality
    elif shot.image is not None:
        q = measure(crop_box(shot.image, det.box))
    else:
        return "unmeasured"
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
    when the gaze picked one, else the widest when it is ``dominance`` times the next. A frame
    needs its picture or its number in the perception service's ring."""
    image = frame.image
    if image is None and frame.seq is None:
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
        measured = (
            Quality(f.sharpness, f.brightness)
            if f.sharpness is not None and f.brightness is not None
            else None
        )
        out.append(Shot(image, f.detection, f is chosen, frame.seq, measured))
    return out


class FaceNet(Protocol):
    """What person recognition embeds faces with: :class:`IdentityNet` in this process,
    :class:`ServiceIdentity` through the perception service, :class:`RoutedIdentity` both."""

    @property
    def name(self) -> str:
        """The model's name: the key its samples are filed under in the store."""
        ...

    @property
    def loaded(self) -> bool:
        """Whether the net is in memory (here or in the service)."""
        ...

    def warm(self) -> None:
        """Load it now."""
        ...

    def embed(self, shot: Shot) -> Vector:
        """The unit-length embedding of ``shot``'s face; raises when there is none."""
        ...

    def unload(self) -> bool:
        """Let this process's net go; whether it was loaded."""
        ...


class ServiceIdentity:
    """:class:`FaceNet` through the perception service's ``/embed``: the shot's frame number
    and detection go there, never pixels; the service loads ArcFace on demand and lets it go
    when idle."""

    def __init__(self, service: Perception, name: str) -> None:
        """``name``: the model the service runs (config/people.json's ``face.model`` stem, the
        same file the service loads)."""
        self.service, self._name = service, name
        self.requests = 0

    @property
    def name(self) -> str:
        """The model's name."""
        return self._name

    @property
    def loaded(self) -> bool:
        """The service holds it (it loads at the first embedding)."""
        return True

    def warm(self) -> None:
        """Nothing here: the service loads its net at the first embedding (0.1 s)."""

    def embed(self, shot: Shot) -> Vector:
        """The service's embedding; ``RuntimeError`` when it gives none."""
        if shot.seq is None:
            raise ValueError("a face with no frame number cannot be embedded by the service")
        vector = self.service.embed(shot.seq, shot.detection)
        if vector is None:
            why = getattr(self.service, "last_error", "") or "no answer"
            raise RuntimeError(f"the perception service gave no embedding ({why})")
        self.requests += 1
        return vector

    def unload(self) -> bool:
        """Nothing is held here."""
        return False


class RoutedIdentity:
    """:class:`FaceNet` for both camera paths: a shot with its picture (the snapshot path) is
    embedded by the net in this process, a shot by frame number by the service; ``warm`` loads
    the local net only while ``vision.source`` says snapshot, so with the service nothing loads
    here."""

    def __init__(
        self,
        local: IdentityNet | None,
        remote: ServiceIdentity | None,
        source: VisionSource,
        name: str,
    ) -> None:
        self.local, self.remote, self.source, self._name = local, remote, source, name

    @property
    def name(self) -> str:
        """The model's name (both paths run the same file)."""
        return self._name

    @property
    def loaded(self) -> bool:
        """Whether the path in use holds its net."""
        if self.source.service:
            return self.remote is not None
        return self.local is not None and self.local.loaded

    def warm(self) -> None:
        """The local net, while the source is the snapshot path."""
        if not self.source.service and self.local is not None:
            self.local.warm()

    def embed(self, shot: Shot) -> Vector:
        """By the shot's kind: pixels here, a frame number at the service."""
        if shot.image is None:
            if self.remote is None:
                raise RuntimeError("no perception service to embed a face by frame number")
            return self.remote.embed(shot)
        if self.local is None:
            raise RuntimeError("no identity net in this process (config/people.json face.model)")
        return self.local.embed(shot)

    def unload(self) -> bool:
        """Let the local net go."""
        return self.local.unload() if self.local is not None else False


class IdentityNet:
    """The identity net, loaded on the first use (or :meth:`warm`) and let go by
    :meth:`unload`; thread-safe."""

    def __init__(self, load: Callable[[], Embedder], clock: Callable[[], float] = time.monotonic):
        """``load`` builds the net (0.1 s warm; seconds the first time)."""
        self._load, self._clock = load, clock
        self._net: Embedder | None = None
        self._name: str | None = None
        self._lock = threading.Lock()
        self.loads = 0
        self.last_used = -1e9

    @property
    def loaded(self) -> bool:
        """Whether the net is in memory."""
        return self._net is not None

    @property
    def name(self) -> str:
        """The net's model name (loads it the first time only)."""
        return self._name if self._name is not None else self._get().name

    def warm(self) -> None:
        """Load it now."""
        self._get()

    def embed(self, shot: Shot) -> Vector:
        """The unit-length embedding of ``shot``'s face (a shot with its picture)."""
        if shot.image is None:
            raise ValueError("this net embeds a face in a picture; this shot has none")
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
                self._name = self._net.name
                self.loads += 1
                logger.info("identity net loaded in %.2f s", time.perf_counter() - t0)
            return self._net
