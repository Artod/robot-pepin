"""Where every sensor sits on the cart, read from the config files in one place.

The board's launch, the laptop's camera node, the depth node, the recorder and the ToF bridge
each read — or copied — a sensor's mount for themselves: ``config/lidar.json`` through
:class:`pepin.lidar.LidarMount`, ``config/imu.json`` through a second mount class,
``config/camera.json`` inside the camera node, ``config/tof.json`` as numbers in the ToF bridge's
code, and the lidar once more as a code constant kept equal to the file by a test. A
:class:`Mount` is one sensor's ``base_link -> sensor`` placement as the files say it (metres and
degrees), and :class:`Mounts` reads all of them from the config directory: ``lidar`` (the
sensor's other fields — masks, ranges, mirror — stay on :class:`pepin.lidar.LidarMount`),
``imu``, ``camera`` (its link and its optical frame), ``tof`` by sensor name. Whatever publishes
a static transform or projects a beam reads it here, so both sides of the bridge publish one
laser and the depth's projection agrees with it. A node that needs one sensor takes that
sensor's reader (:func:`load_lidar_mount`, :func:`load_camera_mounts`) rather than
:meth:`Mounts.load`: the same parser, but a broken file of a sensor it never publishes does
not take it down at start.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pepin.camera import OPTICAL_RPY, CameraConfig
from pepin.deployment import config_file
from pepin.lidar import LidarMount

Array = npt.NDArray[np.float64]
Transform = tuple[float, float, float, float, float, float]

# The frames the cart's static transforms are published under.
LASER_FRAME = "laser"
IMU_FRAME = "imu_link"
TOF_FRAME = "tof_{name}"


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> Array:
    """The 3x3 rotation Rz(yaw) Ry(pitch) Rx(roll), radians: a vector in the sensor's axes
    into the parent frame — the same rotation :func:`pepin.camera.quaternion_from_rpy` and the
    launch's static transforms carry."""
    rx = np.array(
        [[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]]
    )
    ry = np.array(
        [[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]]
    )
    rz = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    )
    rotation: Array = rz @ ry @ rx
    return rotation


@dataclass(frozen=True)
class Mount:
    """One sensor's placement on the cart: ``base_link -> sensor`` in metres and degrees, as a
    ``mount`` block of the config files writes it."""

    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Mount:
        """A ``mount`` block: the six fields it names (``height_m`` counts as ``z_m``; a
        ``note`` is for people), the rest zero."""
        known = {f.name for f in fields(cls)}
        values = {key: float(value) for key, value in data.items() if key in known}
        if "z_m" not in values and "height_m" in data:
            values["z_m"] = float(data["height_m"])
        return cls(**values)

    def transform(self) -> Transform:
        """``(x, y, z, roll, pitch, yaw)`` in metres and radians: the static transform as the
        launch and the broadcasters want it."""
        return (
            self.x_m,
            self.y_m,
            self.z_m,
            math.radians(self.roll_deg),
            math.radians(self.pitch_deg),
            math.radians(self.yaw_deg),
        )

    def rotation(self) -> Array:
        """The 3x3 rotation of the sensor's axes into ``base_link``."""
        _x, _y, _z, roll, pitch, yaw = self.transform()
        return rotation_from_rpy(roll, pitch, yaw)

    def translation(self) -> Array:
        """The sensor's origin in ``base_link``, metres."""
        return np.array([self.x_m, self.y_m, self.z_m], dtype=np.float64)


# The optical frame of a camera relative to its link (REP 103): z looks along the link's x,
# x points right, y down — a roll of -90 degrees followed by a yaw of -90.
OPTICAL_MOUNT = Mount(
    roll_deg=math.degrees(OPTICAL_RPY[0]),
    pitch_deg=math.degrees(OPTICAL_RPY[1]),
    yaw_deg=math.degrees(OPTICAL_RPY[2]),
)


def lidar_mount(sensor: LidarMount) -> Mount:
    """The lidar's placement as a :class:`Mount`: the yaw is the negative of the calibrated
    forward offset, the way :meth:`pepin.lidar.LidarMount.transform` has always given it."""
    return Mount(
        x_m=sensor.x_m,
        y_m=sensor.y_m,
        z_m=sensor.z_m,
        roll_deg=sensor.roll_deg,
        pitch_deg=0.0,
        yaw_deg=-sensor.yaw_offset_deg,
    )


@dataclass(frozen=True)
class CameraMounts:
    """The camera's two static frames: ``base_link -> link`` (where it sits, looking along
    +x, tilted down by the neck's pitch) and ``link -> optical`` (the picture's axes)."""

    link: Mount
    optical: Mount
    link_frame: str = "camera_link"
    optical_frame: str = "camera_optical"

    @classmethod
    def from_config(cls, cfg: CameraConfig) -> CameraMounts:
        """From one camera of ``config/camera.json``."""
        link = Mount(x_m=cfg.x_m, y_m=cfg.y_m, z_m=cfg.z_m, pitch_deg=cfg.pitch_deg)
        return cls(link, OPTICAL_MOUNT, cfg.link_frame, cfg.optical_frame)


def config_path(config_dir: str | Path | None, name: str) -> Path:
    """One config file: inside ``config_dir`` when a directory is given, else wherever
    :func:`pepin.deployment.config_file` finds it (a checkout, the board's synced copy, the
    laptop containers' /ws/config)."""
    return Path(config_dir) / name if config_dir is not None else config_file(name)


def load_lidar(config_dir: str | Path | None = None) -> LidarMount:
    """The lidar as ``config/lidar.json`` writes it — its place on the cart plus its masks,
    ranges and mirror — from that file alone."""
    return LidarMount.from_json(config_path(config_dir, "lidar.json"))


def load_lidar_mount(config_dir: str | Path | None = None) -> Mount:
    """The laser's ``base_link -> laser`` placement from ``config/lidar.json`` alone: the
    numbers (and the sign of the yaw) the board's launch publishes.

    One sensor's file, not the whole directory: a node that publishes only the laser must not
    die at start because another sensor's file is missing or broken (:meth:`Mounts.load` reads
    all four)."""
    return lidar_mount(load_lidar(config_dir))


def load_camera_mounts(
    config_dir: str | Path | None = None, camera: str = "overview"
) -> CameraMounts:
    """The camera's two static frames (``base_link -> camera_link -> camera_optical``) from
    ``config/camera.json`` alone — again one sensor's file, see :func:`load_lidar_mount`."""
    data = json.loads(config_path(config_dir, "camera.json").read_text())
    return CameraMounts.from_config(CameraConfig.from_json(data[camera]))


@dataclass(frozen=True)
class Mounts:
    """Every sensor's mount, from the config directory (:meth:`load`)."""

    lidar: Mount
    lidar_sensor: LidarMount  # the same lidar with its masks, ranges and mirror
    imu: Mount
    camera: CameraMounts
    tof: Mapping[str, Mount]  # by sensor name; a sensor whose mount is null is left out

    @classmethod
    def load(cls, config_dir: str | Path | None = None, camera: str = "overview") -> Mounts:
        """Read lidar.json, imu.json, camera.json and tof.json from ``config_dir``, or from
        wherever :func:`pepin.deployment.config_file` finds them (a checkout, the board's
        synced copy, the laptop containers' /ws/config) when no directory is given.

        All four files, so a caller that needs one sensor takes the narrow reader beside this
        one (:func:`load_lidar_mount`, :func:`load_camera_mounts`) instead of dying on a file
        it has no use for."""

        def read(name: str) -> Any:
            return json.loads(config_path(config_dir, name).read_text())

        sensor = load_lidar(config_dir)
        imu = Mount.from_json(read("imu.json")["mount"])
        cam = load_camera_mounts(config_dir, camera)
        tof = {
            name: Mount.from_json(entry["mount"])
            for name, entry in read("tof.json")["sensors"].items()
            if entry.get("mount") is not None
        }
        return cls(lidar_mount(sensor), sensor, imu, cam, tof)

    def static_frames(self) -> list[tuple[str, str, Mount]]:
        """Every static transform the cart has, as ``(parent, child, mount)``: the laser, the
        IMU, the camera's link and optical frames, the ToF sensors."""
        frames = [
            ("base_link", LASER_FRAME, self.lidar),
            ("base_link", IMU_FRAME, self.imu),
            ("base_link", self.camera.link_frame, self.camera.link),
            (self.camera.link_frame, self.camera.optical_frame, self.camera.optical),
        ]
        frames += [
            ("base_link", TOF_FRAME.format(name=name), mount) for name, mount in self.tof.items()
        ]
        return frames
