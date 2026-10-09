"""Face crops and the bearing of a face: where to point the neck to centre it.

The bearing is measured from the eye's optical axis through the lens model of that eye: the raw
left eye of the stereo head has strong barrel distortion (k1 -0.37), so the face centre is
undistorted before the angles are taken. Pan is positive to the left and tilt positive downward,
the neck's own convention (:mod:`pepin.neck`), so a gaze request adds them to where the head
points now.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


def crop_box(
    image: Image,
    box: tuple[float, float, float, float],
    *,
    margin: float = 0.0,
    square: bool = False,
) -> Image:
    """The part of ``image`` under ``box`` grown by ``margin`` (a fraction of its size) and
    optionally squared, clipped to the image; never empty (at least one pixel)."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w, h = (x1 - x0) * (1.0 + margin), (y1 - y0) * (1.0 + margin)
    if square:
        w = h = max(w, h)
    height, width = image.shape[:2]
    a = int(min(max(0.0, cx - w / 2.0), width - 1))
    b = int(min(max(0.0, cy - h / 2.0), height - 1))
    c = int(max(min(float(width), cx + w / 2.0), a + 1))
    d = int(max(min(float(height), cy + h / 2.0), b + 1))
    return image[b:d, a:c]


@dataclass(frozen=True)
class Lens:
    """A pinhole with distortion at one image size: ``k`` row-major 3x3, OpenCV's ``dist``."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: tuple[float, ...] = ()
    source: str = ""

    @classmethod
    def nominal(cls, width: int, height: int, hfov_deg: float) -> Lens:
        """The undistorted pinhole that puts ``hfov_deg`` across ``width`` pixels."""
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(fx, fx, width / 2.0, height / 2.0, width, height, (), f"nominal {hfov_deg} deg")

    @classmethod
    def stereo_eye(cls, path: str | Path, eye: str = "left") -> Lens:
        """One raw eye of ``config/stereo_calibration.json`` (``k_left``/``d_left`` or the right
        pair): the lens of the unrectified picture the sensor looks at."""
        data = json.loads(Path(path).read_text())
        side = "right" if eye == "right" else "left"
        k = data[f"k_{side}"]
        return cls(
            fx=float(k[0][0]),
            fy=float(k[1][1]),
            cx=float(k[0][2]),
            cy=float(k[1][2]),
            width=int(data["width"]),
            height=int(data["height"]),
            dist=tuple(float(v) for v in data[f"d_{side}"]),
            source=f"{Path(path).name} {side} eye ({data.get('date', 'undated')})",
        )

    def scaled(self, width: int, height: int) -> Lens:
        """The same lens at another image size (the distortion acts on normalised coordinates
        and does not scale)."""
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return Lens(
            self.fx * sx,
            self.fy * sy,
            self.cx * sx,
            self.cy * sy,
            width,
            height,
            self.dist,
            self.source + f", scaled to {width}x{height}",
        )

    def ray(self, u: float, v: float) -> tuple[float, float]:
        """The normalised image coordinates ``(x, y)`` (x right, y down) of pixel ``(u, v)``,
        undistorted when the lens has distortion."""
        if not any(self.dist):
            return (u - self.cx) / self.fx, (v - self.cy) / self.fy
        import cv2

        k = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
        point = np.array([[[u, v]]], dtype=np.float64)
        x, y = cv2.undistortPoints(point, k, np.array(self.dist, dtype=np.float64))[0, 0]
        return float(x), float(y)

    def bearing_deg(self, u: float, v: float) -> tuple[float, float]:
        """``(pan, tilt)`` in degrees that turn the optical axis onto pixel ``(u, v)``: pan about
        the vertical first (positive left), then tilt (positive down)."""
        x, y = self.ray(u, v)
        pan = -math.degrees(math.atan(x))
        tilt = math.degrees(math.atan2(y, math.hypot(1.0, x)))
        return pan, tilt
