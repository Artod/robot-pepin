"""What one frame becomes in the perception service: its optics, faces, people and stamps.

The pixel space is named, never assumed: every frame arrives with its :class:`Optics` (camera_
stream's rectified left eye, a pinhole without distortion: fx 454.5 px on the 2026-10-04
calibration, against the raw eye's 528 at the centre, so a face is 14 % fewer pixels than in a
snapshot), and every angle in a record is taken through it. Angles are the neck's convention:
pan positive to the left, tilt positive down, from the eye's optical axis, so a gaze aim adds
them to where the head pointed at the frame's stamp.

A :class:`Record` is JSON on the wire (``/latest``) and plain dataclasses on both ends; nothing
here imports a detector, so the container and the voice process read it for free.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

Box = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class Optics:
    """A pinhole without distortion at one image size, and the name of the pixel space it is
    (``rectified-left``: the stereo calibration's shared pinhole; ``nominal-left``: the raw eye
    with the vendor's field of view, while the head is uncalibrated)."""

    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def header(self) -> str:
        """The ``X-Optics`` header: ``name fx fy cx cy width height``."""
        return (
            f"{self.name} {self.fx:.3f} {self.fy:.3f} {self.cx:.3f} {self.cy:.3f}"
            f" {self.width} {self.height}"
        )

    @classmethod
    def parse(cls, text: str) -> Optics:
        """An ``X-Optics`` header back; ``ValueError`` for anything else."""
        parts = text.split()
        if len(parts) != 7:
            raise ValueError(f"X-Optics is 'name fx fy cx cy width height', not {text!r}")
        name, fx, fy, cx, cy, width, height = parts
        optics = cls(name, float(fx), float(fy), float(cx), float(cy), int(width), int(height))
        if optics.fx <= 0 or optics.fy <= 0 or optics.width <= 0 or optics.height <= 0:
            raise ValueError(f"X-Optics with no focal length or no size: {text!r}")
        return optics

    def bearing_deg(self, u: float, v: float) -> tuple[float, float]:
        """``(pan, tilt)`` in degrees that turn the optical axis onto pixel ``(u, v)``: pan
        about the vertical first (positive left), then tilt (positive down); exact through the
        pinhole, as :meth:`pepin.face_sensor.geometry.Lens.bearing_deg` takes them."""
        x, y = (u - self.cx) / self.fx, (v - self.cy) / self.fy
        pan = -math.degrees(math.atan(x))
        tilt = math.degrees(math.atan2(y, math.hypot(1.0, x)))
        return pan, tilt


@dataclass(frozen=True)
class Face:
    """One face: its box and five points (eyes, nose, mouth corners) in the frame's pixels, the
    detector's score, the head angles that centre it, how far it is turned from the camera
    (degrees, from the five points; None without them), and its crop's sharpness and
    brightness (person recognition's gate: :mod:`pepin.face_sensor.quality`)."""

    box: Box
    score: float
    kps: tuple[Point, ...]
    pan_deg: float
    tilt_deg: float
    yaw_deg: float | None
    sharpness: float
    brightness: float

    @property
    def width_px(self) -> float:
        """The box width in pixels of the frame's optics."""
        return self.box[2] - self.box[0]


@dataclass(frozen=True)
class Person:
    """One person's box (Apple Vision's human rectangle) in the frame's pixels and in degrees:
    the box centre's pan and half width, the head estimated in it (near an upright box's top, a
    lying box's middle, the top edge of a box the frame cuts), its height, the score, and
    whether the frame's top edge cuts it."""

    box: Box
    score: float
    pan_deg: float
    half_width_deg: float
    head_pan_deg: float
    head_tilt_deg: float
    height_deg: float
    cut: bool = False


@dataclass(frozen=True)
class Record:
    """One detected frame: ``seq`` the service's frame number (the ring's key, monotonic over
    the service's life), ``stamp`` the board's exposure stamp (the board's clock, as camera_
    stream publishes it), ``arrived`` this Mac's wall clock when the frame came (for ages in
    reports only: the two clocks are not one), the sender's own ``frame`` number and ``run``,
    the optics' name, the frame's mean brightness (0-255), the faces and people found, and the
    milliseconds each stage took."""

    seq: int
    stamp: float
    arrived: float
    frame: int = 0
    run: str = ""
    optics: str = ""
    brightness: float = 0.0
    faces: tuple[Face, ...] = ()
    persons: tuple[Person, ...] = ()
    ms: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The JSON object ``/latest`` answers."""
        return {
            "seq": self.seq,
            "stamp": round(self.stamp, 6),
            "arrived": round(self.arrived, 6),
            "frame": self.frame,
            "run": self.run,
            "optics": self.optics,
            "brightness": round(self.brightness, 1),
            "faces": [
                {
                    "box": [round(v, 1) for v in f.box],
                    "score": round(f.score, 4),
                    "kps": [[round(x, 1), round(y, 1)] for x, y in f.kps],
                    "pan_deg": round(f.pan_deg, 3),
                    "tilt_deg": round(f.tilt_deg, 3),
                    "yaw_deg": None if f.yaw_deg is None else round(f.yaw_deg, 2),
                    "sharpness": round(f.sharpness, 2),
                    "brightness": round(f.brightness, 1),
                }
                for f in self.faces
            ],
            "persons": [
                {
                    "box": [round(v, 1) for v in p.box],
                    "score": round(p.score, 4),
                    "pan_deg": round(p.pan_deg, 3),
                    "half_width_deg": round(p.half_width_deg, 3),
                    "head_pan_deg": round(p.head_pan_deg, 3),
                    "head_tilt_deg": round(p.head_tilt_deg, 3),
                    "height_deg": round(p.height_deg, 3),
                    "cut": p.cut,
                }
                for p in self.persons
            ],
            "ms": {k: round(v, 2) for k, v in self.ms.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Record:
        """A record from ``/latest``'s JSON; ``ValueError`` (or ``KeyError``, ``TypeError``)
        for one that is not."""
        faces = tuple(
            Face(
                box=_box(f["box"]),
                score=float(f["score"]),
                kps=tuple((float(x), float(y)) for x, y in f.get("kps", ())),
                pan_deg=float(f["pan_deg"]),
                tilt_deg=float(f["tilt_deg"]),
                yaw_deg=None if f.get("yaw_deg") is None else float(f["yaw_deg"]),
                sharpness=float(f.get("sharpness", 0.0)),
                brightness=float(f.get("brightness", 0.0)),
            )
            for f in data.get("faces", ())
        )
        persons = tuple(
            Person(
                box=_box(p["box"]),
                score=float(p["score"]),
                pan_deg=float(p["pan_deg"]),
                half_width_deg=float(p["half_width_deg"]),
                head_pan_deg=float(p["head_pan_deg"]),
                head_tilt_deg=float(p["head_tilt_deg"]),
                height_deg=float(p["height_deg"]),
                cut=bool(p.get("cut", False)),
            )
            for p in data.get("persons", ())
        )
        return cls(
            seq=int(data["seq"]),
            stamp=float(data["stamp"]),
            arrived=float(data["arrived"]),
            frame=int(data.get("frame", 0)),
            run=str(data.get("run", "")),
            optics=str(data.get("optics", "")),
            brightness=float(data.get("brightness", 0.0)),
            faces=faces,
            persons=persons,
            ms={str(k): float(v) for k, v in dict(data.get("ms", {})).items()},
        )


def _box(values: Sequence[Any]) -> Box:
    x0, y0, x1, y1 = (float(v) for v in values)
    return x0, y0, x1, y1
