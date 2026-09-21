"""A calibrated stereo head: how one transport frame becomes two upright, rectified eyes, and how
a disparity becomes metres.

The head camera can be a stereo module that delivers ONE side-by-side MJPEG frame (left half |
right half, captured by two global shutters at the same instant). Everything downstream of the
camera node — the depth stream's correction stages, the snapshots RTAB-Map receives, the obstacle
fan, the volume — already works on "a picture and a metric depth image with the same stamp", so a
stereo head is a second SOURCE of that pair and nothing else:

* :class:`SideBySide` cuts the transport frame into the two eyes AS THE ROBOT SEES THEM. A module
  mounted upside down (rotated 180 degrees about its optical axis) has each half rotated back and
  the halves swapped: after the turn the module's left lens is on the robot's right.
* :class:`StereoCalibration` is what a calibration measured — each eye's pinhole and distortion,
  and the right eye's pose in the left eye's frame (OpenCV's convention: ``x_right = R x_left +
  T``, metres) — with the evidence beside it (RMS, views, method, day). It is the file
  ``config/stereo_calibration.json``; a half-written file raises instead of being half-believed.
* :class:`Rectifier` turns a calibration into the two remap tables and the ONE rectified pinhole
  both eyes then share (rows aligned, zero disparity at infinity), and converts a disparity in
  pixels into metres: ``z = fx * baseline / disparity``.

OpenCV is imported inside the functions that need it: this module is read by launch files and
unit tests that must stay millisecond-fast.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]

CALIBRATION_FILE = "stereo_calibration.json"
# A disparity under this many pixels is "infinitely far" for a 6 cm baseline: the depth it would
# give is tens of metres with an error as large as itself, so it is reported as unknown (NaN).
MIN_DISPARITY_PX = 0.5


@dataclass(frozen=True)
class SideBySide:
    """One transport frame holding both eyes side by side."""

    upside_down: bool = False

    def eyes(self, frame: Array) -> tuple[Array, Array]:
        """``(left, right)`` as the robot sees them, upright, from one side-by-side frame.

        The frame is cut down the middle (an odd width loses its last column). Upside down, each
        half is rotated by 180 degrees and the halves trade places.
        """
        half = frame.shape[1] // 2
        first, second = frame[:, :half], frame[:, half : 2 * half]
        if not self.upside_down:
            return first, second
        return np.ascontiguousarray(second[::-1, ::-1]), np.ascontiguousarray(first[::-1, ::-1])


@dataclass(frozen=True)
class StereoCalibration:
    """What a calibration measured about the stereo head, at one image size per eye."""

    width: int
    height: int
    k_left: tuple[tuple[float, ...], ...]
    d_left: tuple[float, ...]
    k_right: tuple[tuple[float, ...], ...]
    d_right: tuple[float, ...]
    rotation: tuple[tuple[float, ...], ...]
    translation_m: tuple[float, float, float]
    rms_px: float
    date: str
    method: str
    views: int = 0
    board: str = ""

    @property
    def baseline_m(self) -> float:
        """The distance between the two optical centres, metres."""
        return float(np.linalg.norm(np.asarray(self.translation_m, dtype=np.float64)))

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> StereoCalibration:
        """From the file's dictionary; a missing number raises ``KeyError``."""

        def matrix(name: str) -> tuple[tuple[float, ...], ...]:
            rows = tuple(tuple(float(v) for v in row) for row in data[name])
            if len(rows) != 3 or any(len(row) != 3 for row in rows):
                raise ValueError(f"{name} is not 3x3")
            return rows

        translation = tuple(float(v) for v in data["translation_m"])
        if len(translation) != 3:
            raise ValueError("translation_m is not three numbers")
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            k_left=matrix("k_left"),
            d_left=tuple(float(v) for v in data["d_left"]),
            k_right=matrix("k_right"),
            d_right=tuple(float(v) for v in data["d_right"]),
            rotation=matrix("rotation"),
            translation_m=(translation[0], translation[1], translation[2]),
            rms_px=float(data["rms_px"]),
            date=str(data["date"]),
            method=str(data["method"]),
            views=int(data.get("views", 0)),
            board=str(data.get("board", "")),
        )

    def to_json(self) -> dict[str, Any]:
        """The file's dictionary, the inverse of :meth:`from_json`."""
        return {
            "width": self.width,
            "height": self.height,
            "k_left": [list(row) for row in self.k_left],
            "d_left": list(self.d_left),
            "k_right": [list(row) for row in self.k_right],
            "d_right": list(self.d_right),
            "rotation": [list(row) for row in self.rotation],
            "translation_m": list(self.translation_m),
            "rms_px": self.rms_px,
            "date": self.date,
            "method": self.method,
            "views": self.views,
            "board": self.board,
        }

    @classmethod
    def load(cls, path: str | Path) -> StereoCalibration:
        """Read ``config/stereo_calibration.json`` (``FileNotFoundError`` when there is none)."""
        return cls.from_json(json.loads(Path(path).read_text()))

    def write(self, path: str | Path) -> None:
        """Write the file, whole, so a reader never meets half of it."""
        target = Path(path)
        scratch = target.with_suffix(target.suffix + ".tmp")
        scratch.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        scratch.replace(target)


@dataclass(frozen=True)
class Rectifier:
    """The remap tables of both eyes and the one pinhole they share once rectified."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float
    left_maps: tuple[Array, Array]
    right_maps: tuple[Array, Array]

    @classmethod
    def from_calibration(cls, calibration: StereoCalibration, alpha: float = 0.0) -> Rectifier:
        """Rectify for horizontal epipolar lines and zero disparity at infinity. ``alpha`` 0 keeps
        only valid pixels (no black borders), 1 keeps every source pixel."""
        import cv2

        size = (calibration.width, calibration.height)
        k_left = np.asarray(calibration.k_left, dtype=np.float64)
        k_right = np.asarray(calibration.k_right, dtype=np.float64)
        d_left = np.asarray(calibration.d_left, dtype=np.float64)
        d_right = np.asarray(calibration.d_right, dtype=np.float64)
        rotation = np.asarray(calibration.rotation, dtype=np.float64)
        translation = np.asarray(calibration.translation_m, dtype=np.float64).reshape(3, 1)
        r1, r2, p1, p2, _q, _roi1, _roi2 = cv2.stereoRectify(
            k_left, d_left, k_right, d_right, size, rotation, translation,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha,
        )  # fmt: skip
        left = cv2.initUndistortRectifyMap(k_left, d_left, r1, p1, size, cv2.CV_32FC1)
        right = cv2.initUndistortRectifyMap(k_right, d_right, r2, p2, size, cv2.CV_32FC1)
        return cls(
            width=calibration.width,
            height=calibration.height,
            fx=float(p1[0, 0]),
            fy=float(p1[1, 1]),
            cx=float(p1[0, 2]),
            cy=float(p1[1, 2]),
            baseline_m=float(abs(p2[0, 3]) / p2[0, 0]),
            left_maps=(left[0], left[1]),
            right_maps=(right[0], right[1]),
        )

    def rectify(self, left: Array, right: Array) -> tuple[Array, Array]:
        """Both eyes undistorted and row-aligned, at the calibration's size."""
        import cv2

        return (
            cv2.remap(left, self.left_maps[0], self.left_maps[1], cv2.INTER_LINEAR),
            cv2.remap(right, self.right_maps[0], self.right_maps[1], cv2.INTER_LINEAR),
        )

    def depth_m(self, disparity_px: Array) -> Array:
        """Metres along the optical axis for a disparity image in pixels (float32); NaN where the
        disparity is missing, negative or under :data:`MIN_DISPARITY_PX`."""
        disparity = np.asarray(disparity_px, dtype=np.float32)
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        seen = np.isfinite(disparity) & (disparity >= MIN_DISPARITY_PX)
        depth[seen] = np.float32(self.fx * self.baseline_m) / disparity[seen]
        return depth

    def right_projection_tx(self) -> float:
        """``P[0, 3]`` of the right eye's CameraInfo (ROS's stereo convention): ``-fx * B``."""
        return -self.fx * self.baseline_m
