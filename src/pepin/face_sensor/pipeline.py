"""One frame through the models, and the seconds of frames behind each published line.

:class:`FacePipeline` takes a frame through every stage that is on and that the face's gate
lets through: the detector, then on the best face (the widest box) the landmarker (head pose,
jaw), the expression net (only for a face at least ``min_expression_px`` wide that looks at the
robot) and the identity embedding, timing each stage. :class:`FaceState` keeps the windows that
the published line is made of: the identity vote, the expression and attention windows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pepin.face_sensor.affect import AttentionWindow, ExpressionWindow, facing
from pepin.face_sensor.config import FaceSensorConfig
from pepin.face_sensor.frames import Frame
from pepin.face_sensor.geometry import Lens, crop_box
from pepin.face_sensor.identity import IdentityVote, Vector
from pepin.face_sensor.models import (
    Detection,
    Detector,
    Embedder,
    Expression,
    ExpressionNet,
    Landmarker,
    Mesh,
)
from pepin.face_sensor.quality import Gate, Quality, gate, measure

STAGES = ("grab", "detect", "embed", "expr", "mesh", "total")
MESH_MARGIN = 0.6  # the landmarker's crop: the box grown 60 % and squared (REPORT §2)


@dataclass(frozen=True)
class FaceObs:
    """What one frame says about its best face."""

    detection: Detection
    quality: Quality
    gate: Gate
    bearing: tuple[float, float] | None = None
    embedding: Vector | None = None
    expression: Expression | None = None
    mesh: Mesh | None = None

    @property
    def face_px(self) -> float:
        """The box width in pixels."""
        return self.detection.width


@dataclass(frozen=True)
class FrameResult:
    """One processed frame: its stamp, how many faces, the best one, and the milliseconds of each
    stage that ran."""

    stamp: float
    index: int
    faces: int
    best: FaceObs | None
    ms: dict[str, float] = field(default_factory=dict)


def best_face(faces: list[Detection]) -> Detection | None:
    """The widest box (the nearest person); the detector's score breaks a tie."""
    if not faces:
        return None
    return max(faces, key=lambda d: (round(d.width), d.score))


class FacePipeline:
    """The per-frame stages; a model left ``None`` is a stage switched off."""

    def __init__(
        self,
        cfg: FaceSensorConfig,
        detector: Detector,
        *,
        embedder: Embedder | None = None,
        expression: ExpressionNet | None = None,
        landmarker: Landmarker | None = None,
        lens: Lens | None = None,
    ) -> None:
        """Wire the models; ``lens`` gives the bearing (none without it)."""
        self.cfg = cfg
        self.detector = detector
        self.embedder = embedder
        self.expression = expression
        self.landmarker = landmarker
        self.lens = lens

    def process(self, frame: Frame) -> FrameResult:
        """Decode ``frame`` and run every stage its best face qualifies for."""
        ms: dict[str, float] = {}
        start = time.perf_counter()
        image = frame.image()
        ms["grab"] = _since(start)
        t = time.perf_counter()
        faces = self.detector.detect(image)
        ms["detect"] = _since(t)
        det = best_face(faces)
        best = None if det is None else self._face(image, det, ms)
        ms["total"] = _since(start)
        return FrameResult(frame.stamp, frame.index, len(faces), best, ms)

    def _face(self, image: Any, det: Detection, ms: dict[str, float]) -> FaceObs:
        tight = crop_box(image, det.box)
        quality = measure(tight)
        g = gate(det.width, quality, self.cfg.gate)
        bearing = None
        if self.lens is not None:
            lens = self.lens.scaled(image.shape[1], image.shape[0])
            bearing = lens.bearing_deg(*det.centre)
        mesh = None
        if self.landmarker is not None and g.attention:
            t = time.perf_counter()
            mesh = self.landmarker.read(crop_box(image, det.box, margin=MESH_MARGIN, square=True))
            ms["mesh"] = _since(t)
        expression = None
        looks_away = mesh is not None and not facing(mesh, self.cfg.attention)
        if self.expression is not None and g.expression and not looks_away:
            t = time.perf_counter()
            expression = self.expression.predict(tight)
            ms["expr"] = _since(t)
        embedding = None
        if self.embedder is not None and g.identity:
            t = time.perf_counter()
            embedding = self.embedder.embed(image, det)
            ms["embed"] = _since(t)
        return FaceObs(det, quality, g, bearing, embedding, expression, mesh)


def _since(start: float) -> float:
    return (time.perf_counter() - start) * 1e3


class FaceState:
    """The windows behind the published line: identity vote, expression, attention."""

    def __init__(self, cfg: FaceSensorConfig, vote: IdentityVote | None) -> None:
        """Empty windows; ``vote`` is ``None`` when identity is off."""
        self.cfg = cfg
        self.vote = vote
        self.expression = ExpressionWindow(cfg.expression)
        self.attention = AttentionWindow(cfg.attention)
        self.last: FrameResult | None = None
        self.last_face: FrameResult | None = None

    def update(self, result: FrameResult) -> None:
        """Feed one processed frame into the windows."""
        self.last = result
        obs = result.best
        if obs is None:
            return
        self.last_face = result
        if obs.mesh is not None:
            self.attention.add(result.stamp, obs.mesh)
        if obs.expression is not None:
            self.expression.add(result.stamp, obs.expression)
        if obs.embedding is not None and self.vote is not None:
            self.vote.add(result.stamp, obs.embedding, obs.face_px)

    def record(self, fps: float) -> dict[str, Any] | None:
        """The published line as of the newest frame; ``None`` before the first frame. Face
        fields are ``null`` when no face was seen in the last ``hold_s``."""
        if self.last is None:
            return None
        now = self.last.stamp
        out: dict[str, Any] = {
            "stamp": round(now, 3),
            "faces": self.last.faces,
            "who": None,
            "face_px": None,
            "bearing_deg": None,
            "expr": None,
            "attention": None,
            "quality": None,
            "fps": round(fps, 1),
        }
        held = self.last_face
        if held is None or held.best is None or now - held.stamp > self.cfg.hold_s:
            return out
        obs = held.best
        attention = self.attention.state(now)
        who = self.vote.current(now) if self.vote is not None else None
        out["who"] = who.to_json() if who is not None else None
        out["face_px"] = round(obs.face_px)
        if obs.bearing is not None:
            out["bearing_deg"] = {"pan": round(obs.bearing[0], 1), "tilt": round(obs.bearing[1], 1)}
        looking = attention.looking if attention is not None else None
        if obs.gate.expression and looking is not False:
            smooth = self.expression.mean(now)
            out["expr"] = smooth.to_json() if smooth is not None else None
        out["attention"] = attention.to_json() if attention is not None else None
        out["quality"] = {
            "sharpness": round(obs.quality.sharpness),
            "brightness": round(obs.quality.brightness),
            "gate": obs.gate.reason,
            "det": round(obs.detection.score, 2),
        }
        return out


def percentiles(values: list[float]) -> dict[str, float]:
    """p50, p90, mean and max of ``values`` in milliseconds (zeros when empty)."""
    if not values:
        return {"p50": 0.0, "p90": 0.0, "mean": 0.0, "max": 0.0, "n": 0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(a, 50)), 2),
        "p90": round(float(np.percentile(a, 90)), 2),
        "mean": round(float(np.mean(a)), 2),
        "max": round(float(np.max(a)), 2),
        "n": len(values),
    }
