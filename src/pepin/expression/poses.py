"""Where the cart and its camera were, as the speaker gaze asks it, and the head's geometry.

ONE HISTORY OF THE POSES. The rest of the robot reads its pose from TF (``map -> odom`` by
RTAB-Map, ``odom -> base_link`` by the EKF fusing wheels, VIO and the laser, ``base_link ->
camera_link`` by the neck's encoders), and so does the speaker gaze since 2026-10-15: every
observation (a frame's people, an utterance's readings) asks the goal server's ``pose_at``
(:class:`GoalPoses`, one request an observation: every frame at every stamp of it) where the
camera and the cart were at its own stamp. Before, the gaze kept the people's bearings by the
wheels' odometry, the one private pose on the robot, and on 2026-10-15 the front wheels on the
stand spun at -35 deg/s, the cart did not turn, Nav2 (TF) aborted its spins at 0 deg and the gaze
re-aimed the head off the person 143 times "by the odometry".

THE BOARD'S CLOCK (:class:`BoardClock`). The frames' exposure stamps, the array's readings, the
head's encoder readings and TF's stamps are all on the board's realtime clock (chrony follows
this Mac); the gaze's "now" for anything compared with them is the newest of those it has seen,
never this Mac's clock.

THE HEAD (:class:`HeadModel`): config/neck.json's model, where the camera link stands at a pan
and tilt and the aim that puts a point on its optical axis (:func:`pepin.gaze.aim_at_point`, the
arbiter's own), and config/camera.json's eye in the link (the ray a face's angles name through
the lens, :meth:`pepin.face_sensor.geometry.Lens.bearing_deg`'s convention: pan about the
vertical first, then tilt).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Protocol

from pepin.expression.speaker import Point
from pepin.gaze import Aim, aim_at_point, lens_at
from pepin.mounts import OPTICAL_MOUNT, Mount
from pepin.neck import NeckAngles, NeckConfig, camera_pose

logger = logging.getLogger(__name__)

Vec = tuple[float, float, float]
Rows = tuple[Vec, Vec, Vec]
CAMERA = "camera_link"
BASE = "base_link"
# How long a pose request may take: the goal server's own wait (POSE_AT_WAIT_S, 0.5 s) and the hop
POSES_TIMEOUT_S = 1.5
OLD_SERVER = (
    "the goal server answers pose_at one frame a request (a build before the batch of"
    " 2026-10-15): restart it (ros/laptop.sh kick goal_server)"
)


def _rpy(roll: float, pitch: float, yaw: float) -> Rows:
    """The rotation of roll, then pitch, then yaw (fixed axes: R = Rz Ry Rx), as rows."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _times(rows: Rows, v: Vec) -> Vec:
    return (
        rows[0][0] * v[0] + rows[0][1] * v[1] + rows[0][2] * v[2],
        rows[1][0] * v[0] + rows[1][1] * v[1] + rows[1][2] * v[2],
        rows[2][0] * v[0] + rows[2][1] * v[1] + rows[2][2] * v[2],
    )


def _transposed_times(rows: Rows, v: Vec) -> Vec:
    return (
        rows[0][0] * v[0] + rows[1][0] * v[1] + rows[2][0] * v[2],
        rows[0][1] * v[0] + rows[1][1] * v[1] + rows[2][1] * v[2],
        rows[0][2] * v[0] + rows[1][2] * v[1] + rows[2][2] * v[2],
    )


@dataclass(frozen=True)
class Pose:
    """A frame's pose in its parent (the map): its origin, metres, and its rotation (rows: the
    frame's vectors into the parent's)."""

    origin: Vec
    rotation: Rows

    @classmethod
    def of(cls, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float) -> Pose:
        """From a translation and a unit quaternion (a ``pose_at`` answer's)."""
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        rows = (
            (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
            (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
            (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
        )
        return cls((x, y, z), rows)

    @classmethod
    def planar(cls, x: float, y: float, yaw: float, z: float = 0.0) -> Pose:
        """At ``(x, y, z)``, turned ``yaw`` radians (+ left) about the vertical."""
        return cls((x, y, z), _rpy(0.0, 0.0, yaw))

    @classmethod
    def rpy(cls, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> Pose:
        """From a translation and roll, pitch, yaw (the shape :func:`pepin.neck.camera_pose`
        gives)."""
        return cls((x, y, z), _rpy(roll, pitch, yaw))

    @property
    def yaw(self) -> float:
        """The heading of the frame's x axis in the parent (radians, + left)."""
        return math.atan2(self.rotation[1][0], self.rotation[0][0])

    @property
    def xy(self) -> tuple[float, float]:
        """The origin's ``(x, y)``."""
        return self.origin[0], self.origin[1]

    def to_map(self, v: Vec) -> Point:
        """A point given in this frame, in the parent."""
        r = _times(self.rotation, v)
        return Point(r[0] + self.origin[0], r[1] + self.origin[1], r[2] + self.origin[2])

    def to_frame(self, p: Point) -> Vec:
        """A parent's point in this frame."""
        d = (p.x - self.origin[0], p.y - self.origin[1], p.z - self.origin[2])
        return _transposed_times(self.rotation, d)

    def direction(self, v: Vec) -> Vec:
        """A direction of this frame in the parent (no translation)."""
        return _times(self.rotation, v)

    def then(self, child: Pose) -> Pose:
        """``child``, given in this frame, as a pose in the parent."""
        rows = tuple(
            tuple(
                sum(self.rotation[i][k] * child.rotation[k][j] for k in range(3)) for j in range(3)
            )
            for i in range(3)
        )
        o = self.to_map(child.origin)
        return Pose((o.x, o.y, o.z), rows)  # type: ignore[arg-type]


@dataclass(frozen=True)
class PosesAt:
    """Where the camera and the cart were at one stamp: the stamp answered (TF's own for the
    newest), ``map <- camera_link`` and ``map <- base_link`` (None: not asked, or TF had none,
    then ``why``)."""

    stamp: float
    camera: Pose | None
    base: Pose | None
    why: str = ""


class PoseSource(Protocol):
    """The one history of the poses as the speaker gaze asks it (TF, through the goal server's
    ``pose_at``): every stamp of one observation in ONE request."""

    def poses(self, stamps: Sequence[float | None], *, camera: bool) -> list[PosesAt] | str:
        """The cart (and the camera, with ``camera``) at each stamp (board clock; None: the
        newest there is), in the stamps' order; the reason in words when none could be asked
        (the goal server away, an old build)."""
        ...


def poses_of(answer: Mapping[str, Any], count: int, frames: Sequence[str]) -> list[PosesAt] | str:
    """The goal server's batched ``pose_at`` answer as :class:`PosesAt`, one a stamp; the reason
    in words for an error event or an answer without ``poses`` (an old build)."""
    if answer.get("event") == "error":
        detail = str(answer.get("detail") or "an error")
        return OLD_SERVER if "unknown command" in detail else f"the goal server: {detail}"
    rows = answer.get("poses")
    if not isinstance(rows, list):
        return OLD_SERVER
    if len(rows) != count * len(frames):
        return f"the goal server answered {len(rows)} poses for {count} x {len(frames)}"
    found: list[PosesAt] = []
    for k in range(count):
        group = rows[k * len(frames) : (k + 1) * len(frames)]
        got: dict[str, Pose] = {}
        why = ""
        stamp = 0.0
        for row in group:
            if not isinstance(row, Mapping):
                why = "a pose that is no object"
                continue
            said = row.get("stamp")
            if isinstance(said, (int, float)):
                stamp = float(said)
            if row.get("ok"):
                got[str(row.get("frame"))] = Pose.of(
                    *(float(row[key]) for key in ("x", "y", "z", "qx", "qy", "qz", "qw"))
                )
            elif not why:
                why = f"{row.get('frame')}: {row.get('kind')}: {row.get('reason')}"
        found.append(PosesAt(stamp, got.get(CAMERA), got.get(BASE), why))
    return found


class GoalPoses:
    """:class:`PoseSource` over the goal server on this Mac (:mod:`pepin.goal_link`): one
    connection a request, the camera and the cart at every stamp of it."""

    def __init__(
        self, host: str | None = None, port: int | None = None, timeout_s: float = POSES_TIMEOUT_S
    ) -> None:
        """The goal server at ``host:port`` (goal_link's by default); nothing connects yet."""
        from pepin import goal_link

        self.host = host or goal_link.HOST
        self.port = port or goal_link.PORT
        self.timeout_s = timeout_s
        self.asked = 0  # requests made, for the report

    def poses(self, stamps: Sequence[float | None], *, camera: bool) -> list[PosesAt] | str:
        """One ``pose_at`` request of every stamp (and frame); see :class:`PoseSource`."""
        from pepin import goal_link

        frames = [CAMERA, BASE] if camera else [BASE]
        request = {"cmd": "pose_at", "frames": frames, "stamps": list(stamps)}
        self.asked += 1
        try:
            answer = goal_link.ask(request, self.host, self.port, self.timeout_s)
        except OSError as error:
            return f"the goal server {self.host}:{self.port}: {error}"
        return poses_of(answer, len(stamps), frames)


class BoardClock:
    """The board's clock as the stamps that reach the gaze tell it: the newest seen is its
    "now" (see the module). Never this Mac's clock."""

    def __init__(self) -> None:
        """Nothing seen yet."""
        self._now: float | None = None

    def saw(self, stamp: float | None) -> None:
        """A stamp on the board's clock (None or 0: none)."""
        if stamp is not None and stamp > 0.0 and (self._now is None or stamp > self._now):
            self._now = stamp

    def now(self) -> float | None:
        """The newest stamp seen; None before any."""
        return self._now


def axis(aim: Aim) -> Vec:
    """The optical axis of the head at ``aim`` in base_link (unit: pan about the vertical,
    tilt below level)."""
    cp, ct = math.cos(aim.pan_rad), math.cos(aim.tilt_rad)
    return (ct * cp, ct * math.sin(aim.pan_rad), -math.sin(aim.tilt_rad))


@dataclass(frozen=True)
class HeadModel:
    """The head as the gaze places and aims (see the module): the neck's model and the eye's
    mount in the camera link (``link -> optical``: config/camera.json's eye block composed with
    the optical turn; the optical turn alone puts the eye on the link's axis)."""

    neck: NeckConfig
    optical: Mount = OPTICAL_MOUNT

    @cached_property
    def _eye(self) -> tuple[Rows, Vec]:
        rotation = self.optical.rotation()
        rows = tuple(tuple(float(v) for v in row) for row in rotation)
        origin = tuple(float(v) for v in self.optical.translation())
        return rows, origin  # type: ignore[return-value]

    def aim_at(self, point: Vec) -> Aim:
        """The pan and tilt that put a base_link point on the camera's axis (the arbiter's
        :func:`pepin.gaze.aim_at_point`, lever arms included; no reach)."""
        return aim_at_point(self.neck, point)

    def axis_point(self, aim: Aim, range_m: float) -> Vec:
        """The base_link point ``range_m`` from the lens along its axis with the head at
        ``aim`` (the arbiter resolves it to ``aim`` again)."""
        x, y, z = lens_at(self.neck, aim)
        d = axis(aim)
        return x + d[0] * range_m, y + d[1] * range_m, z + d[2] * range_m

    def lens(self, aim: Aim) -> Vec:
        """The camera link's origin in base_link with the head at ``aim``."""
        return lens_at(self.neck, aim)

    def camera(self, aim: Aim) -> Pose:
        """``base_link -> camera_link`` with the head at ``aim`` (what the board's TF says)."""
        return Pose.rpy(*camera_pose(self.neck, NeckAngles(aim.pan_rad, aim.tilt_rad)))

    def ray(self, pan_deg: float, tilt_deg: float) -> tuple[Vec, Vec]:
        """The ray through the eye of a thing seen ``pan_deg`` left and ``tilt_deg`` below its
        optical axis, in camera_link: its origin (the eye) and its unit direction."""
        x = -math.tan(math.radians(pan_deg))
        y = math.tan(math.radians(tilt_deg)) * math.hypot(1.0, x)
        n = math.sqrt(x * x + y * y + 1.0)
        rows, origin = self._eye
        return origin, _times(rows, (x / n, y / n, 1.0 / n))

    def angles(self, camera: Pose, point: Point) -> tuple[float, float]:
        """Where ``point`` (map) appears to the eye of a camera link at ``camera`` (map): the
        pan and tilt from its optical axis, degrees (:meth:`ray`'s inverse)."""
        rows, origin = self._eye
        v = camera.to_frame(point)
        d = _transposed_times(rows, (v[0] - origin[0], v[1] - origin[1], v[2] - origin[2]))
        if d[2] <= 0.0:
            return math.inf, math.inf  # behind the eye
        x, y = d[0] / d[2], d[1] / d[2]
        return -math.degrees(math.atan(x)), math.degrees(math.atan2(y, math.hypot(1.0, x)))
