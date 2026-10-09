"""Is a face crop worth believing: its sharpness and brightness, and which stages it passes.

Sharpness is the variance of the Laplacian of the grey crop at a common scale (the crop is
shrunk to :data:`SHARP_PX` wide when larger, so a big face does not read sharper only for its
size); brightness is the crop's mean grey level, 0-255. The face's box width decides the rest:
identity holds to about 18 px, expression needs 48 (the expression nets answer at any size, so
the size gate is the only refusal they get).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pepin.face_sensor.config import GateConfig

SHARP_PX = 64


@dataclass(frozen=True)
class Quality:
    """A crop's sharpness (Laplacian variance at :data:`SHARP_PX`) and mean brightness."""

    sharpness: float
    brightness: float


@dataclass(frozen=True)
class Gate:
    """Which stages a face may feed, and the first reason it may not feed some (``ok`` when it
    passes them all)."""

    identity: bool
    expression: bool
    attention: bool
    reason: str


def measure(crop: NDArray[np.uint8]) -> Quality:
    """The sharpness and brightness of a BGR (or grey) crop."""
    import cv2

    grey = np.asarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop)
    width = grey.shape[1]
    if width > SHARP_PX:
        height = max(1, round(grey.shape[0] * SHARP_PX / width))
        grey = np.asarray(cv2.resize(grey, (SHARP_PX, height), interpolation=cv2.INTER_AREA))
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    return Quality(sharpness=sharpness, brightness=float(grey.mean()))


def gate(face_px: float, quality: Quality, cfg: GateConfig) -> Gate:
    """The stages a face of ``face_px`` wide and ``quality`` passes."""
    if quality.brightness < cfg.min_brightness:
        return Gate(False, False, False, "dark")
    if quality.brightness > cfg.max_brightness:
        return Gate(False, False, False, "bright")
    if quality.sharpness < cfg.min_sharpness:
        return Gate(False, False, False, "blurry")
    identity = face_px >= cfg.min_identity_px
    expression = face_px >= cfg.min_expression_px
    attention = face_px >= cfg.min_attention_px
    reason = "ok" if expression else "small"
    return Gate(identity, expression, attention, reason)
