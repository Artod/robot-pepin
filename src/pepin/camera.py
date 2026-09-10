"""The camera as numbers: where it sits on the cart, what it sees, and how ROS wants that said.

The overview camera is a 1280x720 webcam on the neck, 1.23 m above the floor over the wheel
axle. Until a checkerboard calibration replaces it, the intrinsics are the nominal ones of its
field of view — enough for appearance-based loop closure, not for measuring with. Everything
here is pure so the node only carries messages: :class:`CameraConfig` reads ``config/camera.json``,
:func:`intrinsics` and :func:`camera_info_arrays` build ``sensor_msgs/CameraInfo``'s matrices,
:func:`mount_transform` and :func:`optical_rotation` the two static transforms
(``base_link -> camera_link`` x-forward, ``camera_link -> camera_optical`` z-forward as OpenCV
and RTAB-Map expect).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ROS's optical frame: z forward, x right, y down. From an x-forward link that is a roll of
# -90 degrees followed by a yaw of -90 degrees (REP 103).
OPTICAL_RPY = (-math.pi / 2, 0.0, -math.pi / 2)


@dataclass(frozen=True)
class CameraConfig:
    """One camera of ``config/camera.json``: its stream, image size, nominal optics and mount."""

    stream: str
    width: int
    height: int
    hfov_deg: float
    calibrated: bool
    x_m: float
    y_m: float
    z_m: float
    pitch_deg: float
    link_frame: str = "camera_link"
    optical_frame: str = "camera_optical"

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CameraConfig:
        """From one camera's block of ``config/camera.json``."""
        mount = data["mount"]
        frames = data.get("frames", {})
        return cls(
            stream=str(data["stream"]),
            width=int(data["width"]),
            height=int(data["height"]),
            hfov_deg=float(data["hfov_deg"]),
            calibrated=bool(data.get("calibrated", False)),
            x_m=float(mount["x_m"]),
            y_m=float(mount.get("y_m", 0.0)),
            z_m=float(mount["z_m"]),
            pitch_deg=float(mount.get("pitch_deg", 0.0)),
            link_frame=str(frames.get("link", "camera_link")),
            optical_frame=str(frames.get("optical", "camera_optical")),
        )

    @classmethod
    def load(
        cls, path: str | Path, name: str = "overview", board: str = "127.0.0.1"
    ) -> CameraConfig:
        """Read ``config/camera.json`` and fill the stream's ``{board}`` placeholder."""
        data = json.loads(Path(path).read_text())[name]
        cfg = cls.from_json(data)
        return cls(**{**cfg.__dict__, "stream": cfg.stream.format(board=board)})


def intrinsics(width: int, height: int, hfov_deg: float) -> tuple[float, float, float, float]:
    """``(fx, fy, cx, cy)`` of a pinhole with square pixels and the principal point centred:
    the focal length that puts ``hfov_deg`` across ``width`` pixels."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return fx, fx, width / 2.0, height / 2.0


def camera_info_arrays(
    width: int, height: int, hfov_deg: float
) -> tuple[list[float], list[float], list[float], list[float]]:
    """``sensor_msgs/CameraInfo``'s ``k`` (3x3), ``d`` (plumb_bob, no distortion), ``r`` (identity)
    and ``p`` (3x4) as flat row-major lists, for an uncalibrated pinhole."""
    fx, fy, cx, cy = intrinsics(width, height, hfov_deg)
    k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    d = [0.0, 0.0, 0.0, 0.0, 0.0]
    r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return k, d, r, p


def mount_transform(cfg: CameraConfig) -> tuple[float, float, float, float, float, float]:
    """``base_link -> camera_link`` as (x, y, z, roll, pitch, yaw), metres and radians; the link
    frame looks along +x, so a downward tilt of the neck is a positive pitch."""
    return cfg.x_m, cfg.y_m, cfg.z_m, 0.0, math.radians(cfg.pitch_deg), 0.0


def optical_rotation() -> tuple[float, float, float]:
    """``camera_link -> camera_optical`` as (roll, pitch, yaw): the image's z looks along the
    link's x, its x points right and its y down."""
    return OPTICAL_RPY


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """``(x, y, z, w)`` of the rotation yaw * pitch * roll (ROS's fixed-axis convention)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )
