"""A point in the head camera's picture as a point on the map.

The picture is the stereo head's left eye in one of two pixel spaces, named with the box: the
RECTIFIED left eye (:data:`pepin.perception.record.RECTIFIED_LEFT`: the camera channel's frames,
which ``see`` shows a model and the perception service finds faces and people on, rectified by
camera_stream already) or the RAW left eye (:data:`pepin.perception.record.RAW_LEFT`:
ustreamer's snapshot, which ``see`` shows while ``vision.source`` is snapshot). A box drawn on
it becomes a map point in five steps:

1. a raw box's pixels are undistorted and rectified with the left lens and the rectification of
   ``config/stereo_calibration.json`` (:class:`Eye`: the same pinhole
   :class:`pepin.stereo.Rectifier` hands the depth host and camera_stream, so a disparity and a
   ray speak of the same pixel); a rectified box is taken as it is, never rectified twice;
2. a depth source answers a depth image of the rectified pair (:class:`DepthSource`: RAFT-Stereo's
   disparity through the laptop's depth host, z = fx * B / d, or Depth-Anything's metric mono), and
   the box's depth is the median of the valid depths in its central part;
3. the ray through the box's centre, scaled to that depth, is a point in ``camera_optical``
   (the rectified left eye, ``config/camera.json``'s ``eye`` block composed with the optical turn);
4. ``base_link -> camera_link`` at the neck's pan and tilt (:func:`pepin.neck.camera_pose`) carries
   it into ``base_link`` (:class:`CameraChain`);
5. the robot's planar pose on the map (the goal server's ``where``) carries it onto the map.

A box from a frame that carries its stamp (the camera channel's records: the exposure's, on the
board's clock) takes steps 4 and 5 as ONE: ``map -> camera_link`` at that stamp from the one
history of the poses (TF, asked through the goal server's ``pose_at``: :class:`Stamped`), the
neck's pose of that moment inside the chain, then the link-to-optical mount (:func:`locate_at`).
The neck and the cart of now place only a box drawn without a stamp, or one whose stamp TF has no
history for: a face seen during the gaze's own turn and placed with the poses of after it sent the
cart 4.9 m into the kitchen (2026-10-14).

Every input is a protocol — the picture (:class:`PairSource`), the depth (:class:`DepthSource`),
the head (:class:`HeadSource`), the robot's pose (:class:`PoseSource`) — so :class:`Locator` runs on
fakes in the tests and on the robot's services live. The depth pair is grabbed when the box is
located, after its frame: the camera must not have moved in between (the caller's check: the
approach tool compares TF's camera link in ``odom`` at the frame's stamp with the one after
the grab).

The error is the depth's: ``sigma_z = z^2 / (fx * B) * sigma_d`` for stereo (fx * B is 27.5 px*m
on this rig, so half a pixel of disparity is 7 cm at 2 m and 16 cm at 3 m), a fraction of z for
the mono network. Each :class:`Located` carries the sigma of the source that answered it.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from pepin.depth import rotation_matrix
from pepin.mounts import Mount, config_path, load_camera_mounts
from pepin.neck import NeckAngles, NeckConfig, camera_pose
from pepin.perception.record import RAW_LEFT, RECTIFIED_LEFT
from pepin.tsdf import RigidPose

Array = npt.NDArray[np.float64]
Image = npt.NDArray[np.uint8]
Depths = npt.NDArray[np.float32]

INNER = 0.5  # the box's central part whose depths count: half its width and half its height
MIN_VALID = 0.2  # at least this share of that part must carry a depth
MONO_FRACTION = 0.15  # the mono network's 1-sigma as a fraction of z: ASSUMED, unmeasured here
HTTP_TIMEOUT_S = 3.0
BOX = re.compile(r"-?\d+(?:\.\d+)?")


class LocateError(RuntimeError):
    """No point could be located: why, in words a person (or a model) can act on."""


# -- the pieces of the answer ------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """A rectangle in pixels of one picture: ``(u0, v0)`` its top-left, ``(u1, v1)`` its
    bottom-right corner (u to the right, v down)."""

    u0: float
    v0: float
    u1: float
    v1: float

    def __post_init__(self) -> None:
        """Refuse a box with no area."""
        if not (self.u1 > self.u0 and self.v1 > self.v0):
            raise ValueError(f"a box needs x0 < x1 and y0 < y1, not {self}")

    @classmethod
    def parse(cls, text: str) -> Box:
        """``"x0,y0,x1,y1"`` (spaces, brackets and a leading word allowed); ValueError else."""
        numbers = [float(n) for n in BOX.findall(text)]
        if len(numbers) != 4:
            raise ValueError(f"a box is four numbers x0,y0,x1,y1, not {text!r}")
        return cls(*numbers)

    @property
    def centre(self) -> tuple[float, float]:
        """The box's centre ``(u, v)``."""
        return (self.u0 + self.u1) / 2.0, (self.v0 + self.v1) / 2.0

    def inner(self, fraction: float) -> Box:
        """The central part, ``fraction`` of the width and of the height, same centre."""
        u, v = self.centre
        half_w, half_h = (self.u1 - self.u0) * fraction / 2.0, (self.v1 - self.v0) * fraction / 2.0
        return Box(u - half_w, v - half_h, u + half_w, v + half_h)

    def inside(self, width: int, height: int) -> bool:
        """Whether the box overlaps a ``width`` x ``height`` picture at all."""
        return self.u1 > 0 and self.v1 > 0 and self.u0 < width and self.v0 < height


@dataclass(frozen=True)
class Head:
    """The neck's angles, radians: pan positive to the left, pitch positive downward."""

    pan_rad: float
    pitch_rad: float

    def as_dict(self) -> dict[str, float]:
        """In degrees, rounded, as a tool reports the head."""
        return {
            "pan_deg": round(math.degrees(self.pan_rad), 1),
            "tilt_deg": round(math.degrees(self.pitch_rad), 1),
        }


@dataclass(frozen=True)
class Pose2D:
    """The robot on the map: ``base_link``'s position (metres) and heading (radians, CCW)."""

    x: float
    y: float
    yaw_rad: float

    def to_map(self, forward: float, left: float) -> tuple[float, float]:
        """A point given in ``base_link`` (forward, left) as map ``(x, y)``."""
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return self.x + c * forward - s * left, self.y + s * forward + c * left

    def to_base(self, x: float, y: float) -> tuple[float, float]:
        """A map point as ``(forward, left)`` in ``base_link``."""
        dx, dy = x - self.x, y - self.y
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return c * dx + s * dy, -s * dx + c * dy


@dataclass(frozen=True)
class Pair:
    """One stereo picture: both RAW eyes (BGR, upright) and when it was received (wall clock)."""

    left: Image
    right: Image
    stamp: float


@dataclass(frozen=True)
class StereoError:
    """Stereo depth's error law: ``sigma_z = z^2 / (fx * B) * sigma_d``."""

    fb_px_m: float
    sigma_d_px: float

    def sigma(self, z: float) -> float:
        """The 1-sigma depth error at ``z`` metres."""
        return z * z / self.fb_px_m * self.sigma_d_px


@dataclass(frozen=True)
class RelativeError:
    """A depth whose 1-sigma is a fixed fraction of itself (the mono network)."""

    fraction: float

    def sigma(self, z: float) -> float:
        """The 1-sigma depth error at ``z`` metres."""
        return self.fraction * z


@dataclass(frozen=True)
class DepthImage:
    """Depth along the optical axis, metres, per pixel of the RECTIFIED left eye (NaN where there
    is none), the source's name and its error law."""

    z: Depths
    source: str
    error: StereoError | RelativeError


@dataclass(frozen=True)
class Located:
    """Where a box's thing is: on the map (x, y; z above the floor), in ``base_link``, the depth
    it was measured at with that depth's 1-sigma and the box's own scatter, which source
    answered and on how many pixels, the neck's encoders when the depth pair was grabbed, the
    robot's pose the point was placed with, and the frame's ``stamp`` when it was placed with
    TF's poses at that stamp (None: with the neck and the pose of now)."""

    x: float
    y: float
    z: float
    forward: float
    left: float
    depth_m: float
    sigma_m: float
    spread_m: float
    source: str
    pixels: int
    head: Head
    pose: Pose2D
    stamp: float | None = None

    @property
    def range_m(self) -> float:
        """Horizontal distance from ``base_link``'s origin."""
        return math.hypot(self.forward, self.left)

    @property
    def bearing_deg(self) -> float:
        """Degrees left (+) or right (-) of the robot's nose."""
        return math.degrees(math.atan2(self.left, self.forward))

    def as_dict(self) -> dict[str, Any]:
        """The answer as a tool's payload, rounded to centimetres and degrees."""
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "z": round(self.z, 2),
            "range_m": round(self.range_m, 2),
            "bearing_deg": round(self.bearing_deg),
            "depth_m": round(self.depth_m, 2),
            "sigma_m": round(self.sigma_m, 2),
            "spread_m": round(self.spread_m, 2),
            "depth_source": self.source,
        }


@dataclass(frozen=True)
class Stamped:
    """Where the camera and the cart were when one frame was exposed, from the one history of
    the poses (TF, through the goal server's ``pose_at``): ``camera`` is ``map <- camera_link``
    (the neck's pose of that moment inside it), ``base`` the cart's planar pose, ``stamp`` the
    frame's (seconds, the board's clock: TF's) and ``age_s`` how old it was when asked."""

    camera: RigidPose
    base: Pose2D
    stamp: float
    age_s: float

    @classmethod
    def from_answers(cls, camera: Mapping[str, Any], base: Mapping[str, Any]) -> Stamped:
        """From the goal server's two ``pose_at`` answers at one stamp, ``camera_link``'s and
        ``base_link``'s (x, y, z, qx, qy, qz, qw, yaw_deg, stamp, age_s)."""
        yaw = math.radians(float(base["yaw_deg"]))
        return cls(
            rigid_pose(camera),
            Pose2D(float(base["x"]), float(base["y"]), yaw),
            float(camera["stamp"]),
            float(camera["age_s"]),
        )


def turned_deg(before: RigidPose, after: RigidPose) -> float:
    """How far a frame turned between two poses of it in one parent (the camera link in
    ``odom`` at two stamps: the head's turn and the cart's together), degrees: the angle of the
    rotation between them."""
    turn = before.rotation.T @ after.rotation
    cos = max(-1.0, min(1.0, (float(np.trace(turn)) - 1.0) / 2.0))
    return math.degrees(math.acos(cos))


def rigid_pose(answer: Mapping[str, Any]) -> RigidPose:
    """A ``pose_at`` answer's transform (x, y, z and the quaternion qx, qy, qz, qw)."""
    q = (float(answer["qx"]), float(answer["qy"]), float(answer["qz"]), float(answer["qw"]))
    translation = np.array([float(answer["x"]), float(answer["y"]), float(answer["z"])])
    return RigidPose(rotation_matrix(*q), translation)


# -- the optics: the raw left eye and its rectification ---------------------------------------


@dataclass(frozen=True)
class Eye:
    """The left eye as the primitive needs it: the raw lens (``k``, ``dist``) the picture was
    taken through, the rotation into the rectified eye (``rect``) and the rectified pinhole
    (``fx``, ``fy``, ``cx``, ``cy``) that the depth images and ``camera_optical`` share."""

    k: tuple[tuple[float, float, float], ...]
    dist: tuple[float, ...]
    rect: tuple[tuple[float, float, float], ...]
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float
    width: int
    height: int

    @classmethod
    def from_calibration(cls, calibration: Any) -> Eye:
        """From a :class:`pepin.stereo.StereoCalibration`, rectified exactly as
        :meth:`pepin.stereo.Rectifier.from_calibration` rectifies (alpha 0, zero disparity)."""
        import cv2

        size = (int(calibration.width), int(calibration.height))
        k_left = np.asarray(calibration.k_left, dtype=np.float64)
        d_left = np.asarray(calibration.d_left, dtype=np.float64)
        r1, _r2, p1, p2, _q, _a, _b = cv2.stereoRectify(
            k_left, d_left,
            np.asarray(calibration.k_right, dtype=np.float64),
            np.asarray(calibration.d_right, dtype=np.float64),
            size,
            np.asarray(calibration.rotation, dtype=np.float64),
            np.asarray(calibration.translation_m, dtype=np.float64).reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0,
        )  # fmt: skip
        return cls(
            k=_rows(k_left),
            dist=tuple(float(v) for v in d_left.ravel()),
            rect=_rows(r1),
            fx=float(p1[0, 0]),
            fy=float(p1[1, 1]),
            cx=float(p1[0, 2]),
            cy=float(p1[1, 2]),
            baseline_m=float(abs(p2[0, 3]) / p2[0, 0]),
            width=size[0],
            height=size[1],
        )

    @classmethod
    def ideal(cls, width: int, height: int, f: float, baseline_m: float) -> Eye:
        """A distortion-free, unrotated pinhole with its centre in the middle: the raw and the
        rectified picture are one (synthetic scenes, fakes)."""
        cx, cy = width / 2.0, height / 2.0
        k = ((f, 0.0, cx), (0.0, f, cy), (0.0, 0.0, 1.0))
        identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        return cls(k, (), identity, f, f, cx, cy, baseline_m, width, height)

    @property
    def fb_px_m(self) -> float:
        """``fx * B``: a disparity of 1 px is this many metres of depth."""
        return self.fx * self.baseline_m

    def rectified(self, u: float, v: float) -> tuple[float, float]:
        """A raw pixel as the rectified left eye's pixel (undistorted, rotated, reprojected)."""
        k = np.asarray(self.k, dtype=np.float64)
        rect = np.asarray(self.rect, dtype=np.float64)
        p = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
        if any(self.dist):
            import cv2

            point = np.array([[[u, v]]], dtype=np.float64)
            dist = np.asarray(self.dist, dtype=np.float64)
            out = cv2.undistortPoints(point, k, dist, R=rect, P=p)[0, 0]
            return float(out[0]), float(out[1])
        ray = rect @ (np.linalg.inv(k) @ np.array([u, v, 1.0]))
        pixel = p @ (ray / ray[2])
        return float(pixel[0]), float(pixel[1])

    def rectified_box(self, box: Box) -> Box:
        """The rectified pixels' bounding box of ``box``'s corners and edge midpoints."""
        u, v = box.centre
        points = [
            self.rectified(pu, pv)
            for pu in (box.u0, u, box.u1)
            for pv in (box.v0, v, box.v1)
            if (pu, pv) != (u, v)
        ]
        us, vs = [p[0] for p in points], [p[1] for p in points]
        return Box(min(us), min(vs), max(us), max(vs))

    def ray(self, u: float, v: float) -> Array:
        """The ray through rectified pixel ``(u, v)`` in ``camera_optical``, scaled to z = 1."""
        return np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])


def _rows(matrix: Any) -> tuple[tuple[float, float, float], ...]:
    m = np.asarray(matrix, dtype=np.float64)
    return tuple((float(r[0]), float(r[1]), float(r[2])) for r in m)


# -- the chain: camera_optical -> camera_link -> base_link ------------------------------------


@dataclass(frozen=True)
class CameraChain:
    """``base_link -> camera_optical`` at any neck angle: the neck's model (config/neck.json)
    and the link-to-optical mount of the active camera (config/camera.json's ``eye`` block
    composed with the optical turn)."""

    neck: NeckConfig
    optical: Mount

    @classmethod
    def load(cls, config_dir: str | Path | None = None) -> CameraChain:
        """From config/neck.json and config/camera.json (the active camera)."""
        neck = NeckConfig.from_json(config_path(config_dir, "neck.json"))
        return cls(neck, load_camera_mounts(config_dir).optical)

    def optical_in_base(self, head: Head) -> tuple[Array, Array]:
        """``(R, t)``: a point ``p`` in ``camera_optical`` is ``R @ p + t`` in ``base_link``."""
        x, y, z, roll, pitch, yaw = camera_pose(self.neck, NeckAngles(head.pan_rad, head.pitch_rad))
        link = Mount(x, y, z, math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
        rotation: Array = link.rotation() @ self.optical.rotation()
        origin: Array = link.translation() + link.rotation() @ self.optical.translation()
        return rotation, origin

    def camera_height_m(self, head: Head) -> float:
        """The optical centre's height above the floor at ``head``."""
        return float(self.optical_in_base(head)[1][2])


# -- the computation, pure ---------------------------------------------------------------------


def box_depth(
    depth: DepthImage, box: Box, inner: float = INNER, min_valid: float = MIN_VALID
) -> tuple[float, float, int] | None:
    """``(median, spread, pixels)`` of the valid depths in the central ``inner`` part of
    ``box`` (rectified pixels; spread is 1.4826 * MAD); None when under ``min_valid`` of that
    part carries a depth."""
    height, width = depth.z.shape[:2]
    part = box.inner(inner)
    u0, u1 = max(0, math.floor(part.u0)), min(width, math.ceil(part.u1))
    v0, v1 = max(0, math.floor(part.v0)), min(height, math.ceil(part.v1))
    if u1 <= u0 or v1 <= v0:
        return None
    patch = depth.z[v0:v1, u0:u1].astype(np.float64)
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size < max(1, min_valid * patch.size):
        return None
    median = float(np.median(valid))
    spread = 1.4826 * float(np.median(np.abs(valid - median)))
    return median, spread, int(valid.size)


def rectified(eye: Eye, box: Box, space: str) -> Box:
    """``box`` in the rectified left eye's pixels: as it is when ``space`` is already that eye,
    rectified once when it is the raw eye; :class:`LocateError` for any other space."""
    if space == RECTIFIED_LEFT:
        return box
    if space == RAW_LEFT:
        return eye.rectified_box(box)
    raise LocateError(
        f"a box in {space!r} pixels cannot be placed: only the rectified left eye (the camera"
        " channel's frames) or the raw left eye (a snapshot) carries the stereo calibration"
    )


def locate(
    eye: Eye,
    chain: CameraChain,
    box: Box,
    depth: DepthImage,
    head: Head,
    pose: Pose2D,
    *,
    space: str,
    inner: float = INNER,
    min_valid: float = MIN_VALID,
) -> Located | None:
    """The map point under ``box`` (pixels of ``space``: :data:`RECTIFIED_LEFT` or
    :data:`RAW_LEFT`) with the neck at ``head`` and the cart at ``pose``, or None when
    ``depth`` has too little there."""
    measured = _in_optical(eye, box, depth, space, inner, min_valid)
    if measured is None:
        return None
    point, z, spread, pixels = measured
    rotation, origin = chain.optical_in_base(head)
    forward, left, up = (rotation @ point + origin).tolist()
    x, y = pose.to_map(forward, left)
    return Located(
        x=x,
        y=y,
        z=up,
        forward=forward,
        left=left,
        depth_m=z,
        sigma_m=depth.error.sigma(z),
        spread_m=spread,
        source=depth.source,
        pixels=pixels,
        head=head,
        pose=pose,
    )


def locate_at(
    eye: Eye,
    chain: CameraChain,
    box: Box,
    depth: DepthImage,
    at: Stamped,
    head: Head,
    *,
    space: str,
    inner: float = INNER,
    min_valid: float = MIN_VALID,
) -> Located | None:
    """The map point under ``box`` with the camera where it was at its frame's stamp: the ray
    in ``camera_optical``, the link-to-optical mount (``chain.optical``) into ``camera_link``,
    TF's ``map <- camera_link`` at that stamp (``at.camera``) onto the map; ``forward`` and
    ``left`` are in the cart's frame at that stamp (``at.base``). ``head`` is recorded only (the
    neck's encoders when the depth pair was grabbed). None when ``depth`` has too little there."""
    measured = _in_optical(eye, box, depth, space, inner, min_valid)
    if measured is None:
        return None
    point, z, spread, pixels = measured
    optical = chain.optical
    in_link = optical.rotation() @ point + optical.translation()
    x, y, up = (at.camera.rotation @ in_link + at.camera.translation).tolist()
    forward, left = at.base.to_base(x, y)
    return Located(
        x=x,
        y=y,
        z=up,
        forward=forward,
        left=left,
        depth_m=z,
        sigma_m=depth.error.sigma(z),
        spread_m=spread,
        source=depth.source,
        pixels=pixels,
        head=head,
        pose=at.base,
        stamp=at.stamp,
    )


def _in_optical(
    eye: Eye, box: Box, depth: DepthImage, space: str, inner: float, min_valid: float
) -> tuple[Array, float, float, int] | None:
    """The box centre's ray scaled to the box's depth, in ``camera_optical``, with that depth,
    its spread and its pixel count; None when ``depth`` has too little there."""
    rect = rectified(eye, box, space)
    measured = box_depth(depth, rect, inner, min_valid)
    if measured is None:
        return None
    z, spread, pixels = measured
    u, v = rect.centre
    point: Array = eye.ray(u, v) * z
    return point, z, spread, pixels


def disparity_to_depth(disparity: npt.NDArray[Any], fb_px_m: float) -> Depths:
    """Depth (m) of a disparity image (px): ``fx * B / d``, NaN where ``d`` is not positive."""
    d = np.asarray(disparity, dtype=np.float32)
    z = np.full(d.shape, np.nan, dtype=np.float32)
    good = np.isfinite(d) & (d > 0.0)
    z[good] = np.float32(fb_px_m) / d[good]
    return z


# -- the inputs ----------------------------------------------------------------------------------


class PairSource(Protocol):
    """The head camera's newest stereo picture."""

    def grab(self) -> Pair:
        """Both raw eyes, upright; :class:`LocateError` when the camera does not answer."""
        ...


class DepthSource(Protocol):
    """Depth of a rectified pair, per pixel of its left eye."""

    name: str

    def depth(self, left: Image, right: Image) -> DepthImage:
        """The depth image; :class:`LocateError` when the source does not answer."""
        ...


class HeadSource(Protocol):
    """Where the head points now."""

    def head(self) -> Head:
        """The neck's angles from its encoders."""
        ...


class PoseSource(Protocol):
    """Where the robot stands on the map now."""

    def pose(self) -> Pose2D:
        """``base_link`` in ``map``; :class:`LocateError` when nobody knows."""
        ...


Rectify = Callable[[Image, Image], tuple[Image, Image]]


@dataclass(frozen=True)
class CameraGeometry:
    """What turns pixels into rays: the eye, the chain, and the remap of a raw pair."""

    eye: Eye
    chain: CameraChain
    rectify: Rectify

    @classmethod
    def load(cls, config_dir: str | Path | None = None) -> CameraGeometry:
        """From the config files: the stereo calibration of the active camera, the neck, the
        mounts; the remap tables are :class:`pepin.stereo.Rectifier`'s."""
        from pepin.camera import CameraConfig, active_camera
        from pepin.stereo import Rectifier, StereoCalibration

        data = json.loads(config_path(config_dir, "camera.json").read_text())
        name = active_camera(data)
        cfg = CameraConfig.from_json(data[name], name)
        if cfg.rig is None:
            raise LocateError(f"the active camera {name!r} is not a stereo head")
        folder = config_path(config_dir, "camera.json").parent
        calibration = StereoCalibration.load(cfg.rig.calibration_path(folder))
        rectifier = Rectifier.from_calibration(calibration)
        return cls(
            Eye.from_calibration(calibration), CameraChain.load(config_dir), rectifier.rectify
        )


class Locator:
    """The primitive with its inputs: a box of the left eye (rectified or raw, named with it)
    in, a :class:`Located` out.

    ``depths`` are asked in order; the first whose depth covers the box answers (stereo first,
    the mono network where stereo has no texture), and a source that is down or blank there is
    named in the refusal when none answers. ``geometry`` is read from the config files on first
    use when not given."""

    def __init__(
        self,
        pairs: PairSource,
        depths: Sequence[DepthSource],
        head: HeadSource,
        pose: PoseSource,
        geometry: CameraGeometry | None = None,
        inner: float = INNER,
        min_valid: float = MIN_VALID,
    ) -> None:
        """Nothing is read or asked until :meth:`locate`."""
        self.pairs, self.depths, self.head_source, self.pose_source = pairs, depths, head, pose
        self._geometry = geometry
        self.inner, self.min_valid = inner, min_valid

    @property
    def geometry(self) -> CameraGeometry:
        """The camera's geometry, loaded once."""
        if self._geometry is None:
            self._geometry = CameraGeometry.load()
        return self._geometry

    def locate(self, box: Box, space: str, at: Stamped | None = None) -> Located:
        """Where the thing under ``box`` (pixels of ``space``: :data:`RECTIFIED_LEFT`, the
        camera channel's, or :data:`RAW_LEFT`, a snapshot's) is; :class:`LocateError` saying
        why not. ``at``: TF's poses at the box's frame's stamp, which place it
        (:func:`locate_at`, and the pose source is not asked); None places it with the neck
        and the cart of now (:func:`locate`)."""
        geometry = self.geometry
        eye = geometry.eye
        if not box.inside(eye.width, eye.height):
            raise LocateError(
                f"the box {box.u0:.0f},{box.v0:.0f},{box.u1:.0f},{box.v1:.0f} lies outside the"
                f" {eye.width}x{eye.height} picture"
            )
        rectified(eye, box, space)  # a space without the calibration is refused before any grab
        head = self.head_source.head()
        pose = self.pose_source.pose() if at is None else at.base
        pair = self.pairs.grab()
        left, right = geometry.rectify(pair.left, pair.right)
        tried: list[str] = []
        for source in self.depths:
            try:
                depth = source.depth(left, right)
            except LocateError as error:
                tried.append(f"{source.name}: {error}")
                continue
            if at is None:
                found = locate(
                    eye, geometry.chain, box, depth, head, pose,
                    space=space, inner=self.inner, min_valid=self.min_valid,
                )  # fmt: skip
            else:
                found = locate_at(
                    eye, geometry.chain, box, depth, at, head,
                    space=space, inner=self.inner, min_valid=self.min_valid,
                )  # fmt: skip
            if found is not None:
                return found
            tried.append(f"{source.name}: no depth in that box (a blank or too distant surface)")
        raise LocateError("no depth for that box: " + "; ".join(tried or ["no depth source"]))


# -- the real inputs -------------------------------------------------------------------------------

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # LAN and loopback only


class UstreamerPairs:
    """:class:`PairSource` from ustreamer on the board: ``/snapshot`` is one side-by-side frame,
    cut into the two raw eyes as every reader of the head does
    (:class:`pepin.stereo.SideBySide`)."""

    def __init__(
        self, host: str, port: int = 8080, upside_down: bool | None = None, timeout_s: float = 3.0
    ) -> None:
        """ustreamer at ``host:port``; ``upside_down`` None reads config/camera.json's rig."""
        self.url = f"http://{host}:{port}/snapshot"
        self._upside_down = upside_down
        self.timeout_s = timeout_s

    def grab(self) -> Pair:
        """One frame, both eyes."""
        import cv2

        from pepin.stereo import SideBySide

        try:
            with _OPENER.open(self.url, timeout=self.timeout_s) as response:
                body: bytes = response.read()
        except (urllib.error.URLError, OSError) as error:
            raise LocateError(f"the camera ({self.url}) is not answering: {error}") from error
        frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise LocateError(f"the camera sent {len(body)} bytes that are not a picture")
        left, right = SideBySide(upside_down=self.upside_down()).eyes(frame)
        return Pair(left, right, time.time())

    def upside_down(self) -> bool:
        """Whether the module is mounted upside down (config/camera.json's rig)."""
        if self._upside_down is None:
            from pepin.camera import active_camera
            from pepin.deployment import config_file

            data = json.loads(config_file("camera.json").read_text())
            rig = data[active_camera(data)].get("rig") or {}
            self._upside_down = bool(rig.get("upside_down", False))
        return self._upside_down


class HostStereo:
    """:class:`DepthSource` from RAFT-Stereo on the laptop's depth host (``/disparity``):
    the rectified eyes greyed, the disparity turned into depth with the rectified ``fx * B``."""

    name = "stereo"

    def __init__(self, url: str, fb_px_m: float | None = None, timeout_s: float = 3.0) -> None:
        """The host at ``url``; ``fb_px_m`` None takes the calibration's on first use."""
        from pepin.stereo_host import RemoteDisparity

        self.url = url
        self._remote = RemoteDisparity(url, timeout_s=timeout_s)
        self._fb = fb_px_m

    def depth(self, left: Image, right: Image) -> DepthImage:
        """One pair through the network."""
        import cv2

        from pepin.stereo_depth import DISPARITY_SIGMA_PX
        from pepin.stereo_host import StereoHostError

        if self._fb is None:
            self._fb = CameraGeometry.load().eye.fb_px_m
        grey = [
            cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY) if eye.ndim == 3 else eye for eye in (left, right)
        ]
        try:
            disparity = self._remote(grey[0], grey[1])
        except StereoHostError as error:
            raise LocateError(
                f"the depth host ({self.url}) did not answer a pair: {error}"
                " (ros/depth_host.sh stereo)"
            ) from error
        z = disparity_to_depth(disparity, self._fb)
        return DepthImage(z, self.name, StereoError(self._fb, DISPARITY_SIGMA_PX))


class HostMono:
    """:class:`DepthSource` from Depth-Anything's metric mono network on the same host
    (``/depth``), on the rectified left eye; its error is :data:`MONO_FRACTION` of z, ASSUMED."""

    name = "mono"

    def __init__(self, url: str, timeout_s: float = 5.0) -> None:
        """The host at ``url`` (the network loads on its first frame there)."""
        from pepin.depth_service import RemoteDepth

        self.url = url
        self._remote = RemoteDepth(url, timeout_s=timeout_s)

    def depth(self, left: Image, right: Image) -> DepthImage:
        """The left eye through the network."""
        from pepin.depth_service import DepthServiceError

        try:
            z = self._remote(np.ascontiguousarray(left[:, :, ::-1]))
        except DepthServiceError as error:
            raise LocateError(f"the mono depth ({self.url}) did not answer: {error}") from error
        return DepthImage(np.asarray(z, dtype=np.float32), self.name, RelativeError(MONO_FRACTION))
