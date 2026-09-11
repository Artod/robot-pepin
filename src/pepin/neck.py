"""The neck as numbers: two Feetech servos (pan, tilt) as angles and as the camera's pose.

The overview camera rides a two-servo neck: ``neck`` (id 9) pans about the vertical, ``head``
(id 10) tilts about the camera's own y axis. Both are 12-bit encoders, 4096 ticks per turn,
homed so that "straight ahead, level" reads near 2048 (scripts/calibrate_neck.py). This module
turns ticks into radians and radians into ``base_link -> camera_link`` — pure, so the board's
node only carries messages and a test can hold every number.

The angle model is anchored at a *reference pose*, the pose the camera had when its mount
(config/camera.json: 1.23 m up, 26 degrees down) was measured against the lidar, and the ticks
the servos read at that moment::

    pan_rad   = pan_sign  * (pan_ticks  - reference.pan_ticks)  * 2 pi / 4096
    pitch_rad = reference.pitch + tilt_sign * (tilt_ticks - reference.tilt_ticks) * 2 pi / 4096

so at the reference ticks the dynamic transform equals the static one exactly, whatever the
signs; the signs only decide which way the picture moves when the servos do, and they are
verified by jogging a servo in daylight (config/neck.json says whether that happened).

The reference ticks are a hardware reading and may be missing: ``null`` in config/neck.json
until someone asks the base server what the servos read at the measured pose. Missing, the
model has nothing to measure against and falls back to the static mount — the encoders are
ignored and every pose is the one the laptop's camera node publishes from config/camera.json.
The neck then moves without the transform following, which is the old behaviour, not a new
error.

The geometry is a chain: base_link -> pan pivot -> Rz(pan) -> tilt pivot -> Ry(pitch) ->
camera. The two lever arms (pan axis to tilt axis, tilt axis to lens) are the ``pivot`` block;
the pan pivot itself is derived so that the reference pose lands the camera exactly on the
measured mount. With zero lever arms the chain is a rotation about the camera's own position.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TICKS_PER_TURN = 4096
RAD_PER_TICK = 2.0 * math.pi / TICKS_PER_TURN

PAN = "neck"  # the servo names on the bus, as config/neck.json and the base server call them
TILT = "head"
JOINT_NAMES = ("neck_pan", "head_tilt")  # the sensor_msgs/JointState names, in this order


@dataclass(frozen=True)
class NeckJoint:
    """One neck servo: its bus id and the encoder range it may be commanded within."""

    name: str
    motor_id: int
    center: int
    min_ticks: int
    max_ticks: int

    def within_limits(self, ticks: int) -> bool:
        """Whether an encoder reading lies inside the calibrated safe range."""
        return self.min_ticks <= ticks <= self.max_ticks


@dataclass(frozen=True)
class NeckReference:
    """The camera's measured pose and the encoder ticks the servos read in that pose.

    ``x_m, y_m, z_m, pitch_deg`` are config/camera.json's mount on the day (a test keeps them
    equal). ``pan_ticks``/``tilt_ticks`` are None while nobody has read the servos at that pose
    (see :attr:`known`). ``pan_sign``/``tilt_sign`` say which way an increasing tick count turns
    the camera (+1: pan left / look further down); ``signs_verified`` is False until someone
    moved the neck and watched the picture.
    """

    pan_ticks: int | None
    tilt_ticks: int | None
    x_m: float
    y_m: float
    z_m: float
    pitch_deg: float
    pan_sign: int = 1
    tilt_sign: int = 1
    signs_verified: bool = False

    @property
    def known(self) -> bool:
        """Whether both reference ticks have been read: without them no angle can be measured."""
        return self.pan_ticks is not None and self.tilt_ticks is not None


@dataclass(frozen=True)
class NeckPivot:
    """The neck's lever arms, metres: pan axis -> tilt axis in the pan frame (x forward, z up
    once the pan is undone) and tilt axis -> lens in the tilt frame. Zero until measured."""

    tilt_from_pan_x_m: float = 0.0
    tilt_from_pan_z_m: float = 0.0
    camera_from_tilt_x_m: float = 0.0
    camera_from_tilt_z_m: float = 0.0


@dataclass(frozen=True)
class NeckConfig:
    """Everything config/neck.json says: the two servos, the reference pose, the lever arms."""

    pan: NeckJoint
    tilt: NeckJoint
    reference: NeckReference
    pivot: NeckPivot

    @classmethod
    def from_json(cls, path: str | Path) -> NeckConfig:
        """Load config/neck.json; raises ``KeyError`` naming what is missing."""
        data = json.loads(Path(path).read_text())
        ref, pivot = data["reference"], data.get("pivot", {})
        return cls(
            pan=_joint(PAN, data[PAN]),
            tilt=_joint(TILT, data[TILT]),
            reference=NeckReference(
                pan_ticks=_optional_ticks(ref["pan_ticks"]),
                tilt_ticks=_optional_ticks(ref["tilt_ticks"]),
                x_m=float(ref["x_m"]),
                y_m=float(ref["y_m"]),
                z_m=float(ref["z_m"]),
                pitch_deg=float(ref["pitch_deg"]),
                pan_sign=_sign(ref.get("pan_sign", 1)),
                tilt_sign=_sign(ref.get("tilt_sign", 1)),
                signs_verified=bool(ref.get("signs_verified", False)),
            ),
            pivot=NeckPivot(
                tilt_from_pan_x_m=float(pivot.get("tilt_from_pan_x_m", 0.0)),
                tilt_from_pan_z_m=float(pivot.get("tilt_from_pan_z_m", 0.0)),
                camera_from_tilt_x_m=float(pivot.get("camera_from_tilt_x_m", 0.0)),
                camera_from_tilt_z_m=float(pivot.get("camera_from_tilt_z_m", 0.0)),
            ),
        )

    def motor_ids(self) -> dict[str, int]:
        """Name-to-id table of the two servos, for a bus roster."""
        return {self.pan.name: self.pan.motor_id, self.tilt.name: self.tilt.motor_id}


def neck_servo_ids(path: str | Path) -> dict[str, int]:
    """Only the servo names and ids of config/neck.json (``{"neck": 9, "head": 10}``): what the
    base server needs to read the encoders, without the reference the geometry needs."""
    data = json.loads(Path(path).read_text())
    return {name: int(data[name]["id"]) for name in (PAN, TILT)}


def _joint(name: str, block: dict[str, Any]) -> NeckJoint:
    return NeckJoint(
        name=name,
        motor_id=int(block["id"]),
        center=int(block["center"]),
        min_ticks=int(block["min"]),
        max_ticks=int(block["max"]),
    )


def _optional_ticks(value: Any) -> int | None:
    """An encoder reading from the file, or None for a reading nobody has taken yet (``null``)."""
    return None if value is None else int(value)


def _sign(value: Any) -> int:
    sign = int(value)
    if sign not in (1, -1):
        raise ValueError(f"a servo sign is +1 or -1, not {value!r}")
    return sign


def ticks_from(reference: int, ticks: int) -> int:
    """Signed encoder travel from ``reference`` to ``ticks``, the short way round the turn:
    a 12-bit reading never means more than half a revolution away."""
    return (ticks - reference + TICKS_PER_TURN // 2) % TICKS_PER_TURN - TICKS_PER_TURN // 2


@dataclass(frozen=True)
class NeckAngles:
    """The neck's joint angles in radians: pan positive to the left, pitch positive downward
    (REP 103, the camera link looking along +x)."""

    pan_rad: float
    pitch_rad: float


def joint_angles(cfg: NeckConfig, pan_ticks: int, tilt_ticks: int) -> NeckAngles:
    """Encoder ticks of both servos into the joint angles of the model above.

    With the reference ticks unread (null in the file) the encoders measure nothing: the answer
    is the static mount — no pan, the pitch that was measured — whatever the servos read.
    """
    ref = cfg.reference
    pitch_at_rest = math.radians(ref.pitch_deg)
    if ref.pan_ticks is None or ref.tilt_ticks is None:
        return NeckAngles(0.0, pitch_at_rest)
    pan = ref.pan_sign * ticks_from(ref.pan_ticks, pan_ticks) * RAD_PER_TICK
    pitch = pitch_at_rest + ref.tilt_sign * ticks_from(ref.tilt_ticks, tilt_ticks) * RAD_PER_TICK
    return NeckAngles(pan, pitch)


def pan_pivot(cfg: NeckConfig) -> tuple[float, float, float]:
    """Where the pan axis meets the neck in base_link: the measured camera position with the
    lever arms, posed as at the reference, taken away — so the reference reproduces the mount."""
    ref, arm = cfg.reference, cfg.pivot
    pitch = math.radians(ref.pitch_deg)
    cx, cz = _pitched(arm.camera_from_tilt_x_m, arm.camera_from_tilt_z_m, pitch)
    return (
        ref.x_m - arm.tilt_from_pan_x_m - cx,
        ref.y_m,
        ref.z_m - arm.tilt_from_pan_z_m - cz,
    )


def camera_pose(
    cfg: NeckConfig, angles: NeckAngles
) -> tuple[float, float, float, float, float, float]:
    """``base_link -> camera_link`` at these joint angles as (x, y, z, roll, pitch, yaw), metres
    and radians — the shape pepin.camera.mount_transform has, so one broadcaster serves both."""
    px, py, pz = pan_pivot(cfg)
    arm = cfg.pivot
    cx, cz = _pitched(arm.camera_from_tilt_x_m, arm.camera_from_tilt_z_m, angles.pitch_rad)
    forward, up = arm.tilt_from_pan_x_m + cx, arm.tilt_from_pan_z_m + cz
    cos_pan, sin_pan = math.cos(angles.pan_rad), math.sin(angles.pan_rad)
    return (
        px + cos_pan * forward,
        py + sin_pan * forward,
        pz + up,
        0.0,
        angles.pitch_rad,
        angles.pan_rad,
    )


def _pitched(x: float, z: float, pitch: float) -> tuple[float, float]:
    """A lever (x forward, z up) after a pitch about y; a downward pitch dips its tip."""
    return x * math.cos(pitch) + z * math.sin(pitch), -x * math.sin(pitch) + z * math.cos(pitch)


@dataclass(frozen=True)
class NeckReading:
    """One ``neck`` reply from the base server: the encoders, how old the reading is (the
    board's clock), what the bus round trip cost, and the error text of a silent servo."""

    pan_ticks: int | None
    tilt_ticks: int | None
    age_s: float
    read_ms: float
    error: str | None = None

    @property
    def ticks(self) -> tuple[int, int] | None:
        """``(pan, tilt)`` when the reply carried a reading, None when it only carried an error."""
        if self.pan_ticks is None or self.tilt_ticks is None:
            return None
        return self.pan_ticks, self.tilt_ticks


def parse_neck(message: dict[str, Any]) -> NeckReading | None:
    """A ``{"type": "neck", ...}`` line as a :class:`NeckReading`; None for any other line."""
    if message.get("type") != "neck":
        return None
    pan, tilt = message.get("pan_ticks"), message.get("tilt_ticks")
    error = message.get("error")
    return NeckReading(
        pan_ticks=None if pan is None else int(pan),
        tilt_ticks=None if tilt is None else int(tilt),
        age_s=float(message.get("age_s", float("inf"))),
        read_ms=float(message.get("read_ms", 0.0)),
        error=None if error is None else str(error),
    )
