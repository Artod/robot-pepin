"""The camera as numbers: where it sits on the cart, what it sees, and how ROS wants that said.

``config/camera.json`` holds the camera RIGS BY NAME and says which one the robot's head is
(``"active"``). Today there are two. ``overview`` is the 1280x720 mono webcam on the neck,
1.23 m above the floor over the wheel axle. ``stereo`` is the global-shutter stereo module taped
in its place: the board streams ONE side-by-side frame and the laptop cuts it into two eyes
(:mod:`pepin.stereo`), so its ``width``/``height`` are ONE EYE and the transport frame's size
lives in its ``rig`` block (:class:`StereoRig`) — a camera block with a ``rig`` block IS a
stereo head. Which one a process reads is :func:`active_camera`: an explicit name beats
``PEPIN_CAMERA`` in the environment, which beats the file's ``"active"``, which beats
``overview``. Nothing in the stack names a rig: a node loads the active camera and prints which
it got.

The optics are either measured or guessed. For the mono rig a checkerboard calibration
(``scripts/calibrate_camera.py``, ``ros/calibrate.sh``) writes an ``intrinsics`` block and flips
``calibrated``; for a stereo rig ``calibrated`` means ``config/stereo_calibration.json`` exists
and loads, and the pinhole then comes from :class:`pepin.stereo.Rectifier` rather than from this
file. Until either exists the numbers are the nominal pinhole of the configured field of view —
enough for appearance-based loop closure, not for measuring with. :func:`optics` is the one
place that decides between measured and nominal and scales the answer to the size a node
actually publishes; every consumer of the camera's focal length comes through it, so turning a
calibration on changes one file and no code.

Everything here is pure so the node only carries messages: :class:`CameraConfig` reads
``config/camera.json``, :class:`Calibration` is its ``intrinsics`` block (and
:func:`write_calibration` puts one there), :class:`StereoRig` its ``rig`` block, :class:`Optics`
is what a node publishes as ``sensor_msgs/CameraInfo``, :func:`mount_transform` and
:func:`optical_rotation` the two static transforms (``base_link -> camera_link`` x-forward,
``camera_link -> camera_optical`` z-forward as OpenCV and RTAB-Map expect).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# ROS's optical frame: z forward, x right, y down. From an x-forward link that is a roll of
# -90 degrees followed by a yaw of -90 degrees (REP 103).
OPTICAL_RPY = (-math.pi / 2, 0.0, -math.pi / 2)

# The file's key that names the robot's head, the environment variable that overrides it for one
# process, and the camera every one of them falls back to (the rig that was here first).
ACTIVE_KEY = "active"
CAMERA_ENV = "PEPIN_CAMERA"
DEFAULT_CAMERA = "overview"
# The stereo calibration file, named in a rig block and read from beside config/camera.json.
STEREO_CALIBRATION = "stereo_calibration.json"


@dataclass(frozen=True)
class Calibration:
    """What a checkerboard measured about this lens: the pinhole in pixels at the resolution it
    was shot at, the distortion coefficients that go with it, and the evidence — the RMS
    reprojection error, the day, and the board used. The ``intrinsics`` block of
    ``config/camera.json``."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: tuple[float, ...]
    rms: float
    date: str
    board: str
    views: int = 0
    model: str = "plumb_bob"

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Calibration:
        """From an ``intrinsics`` block; raises ``KeyError`` on a block missing a number, so a
        half-written calibration is never half-believed."""
        return cls(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            width=int(data["width"]),
            height=int(data["height"]),
            dist=tuple(float(v) for v in data["dist"]),
            rms=float(data["rms"]),
            date=str(data["date"]),
            board=str(data["board"]),
            views=int(data.get("views", 0)),
            model=str(data.get("model", "plumb_bob")),
        )

    def to_json(self) -> dict[str, Any]:
        """The block as ``config/camera.json`` carries it, in reading order. ``hfov_deg`` here is
        derived from ``fx`` and is for people to read: the block's own field of view, kept apart
        from the camera's nominal one outside it."""
        return {
            "width": self.width,
            "height": self.height,
            "hfov_deg": round(self.hfov_deg(), 2),
            "fx": round(self.fx, 3),
            "fy": round(self.fy, 3),
            "cx": round(self.cx, 3),
            "cy": round(self.cy, 3),
            "dist": [round(v, 6) for v in self.dist],
            "model": self.model,
            "rms": round(self.rms, 4),
            "views": self.views,
            "board": self.board,
            "date": self.date,
        }

    def scaled(self, width: int, height: int) -> Calibration:
        """The same lens described at another image size: the pinhole scales with the pixels,
        the distortion coefficients do not (they act on normalised coordinates). The half-pixel
        shift of a resampled grid is ignored — it is 0.25 px at scale 0.5, well under the
        reprojection error a calibration is accepted at."""
        sx, sy = width / self.width, height / self.height
        return Calibration(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=width,
            height=height,
            dist=self.dist,
            rms=self.rms,
            date=self.date,
            board=self.board,
            views=self.views,
            model=self.model,
        )

    def hfov_deg(self) -> float:
        """The horizontal field of view the focal length implies, degrees: the derived,
        human-readable number kept beside the matrix in the config."""
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.fx)))


@dataclass(frozen=True)
class StereoRig:
    """The ``rig`` block of a stereo camera: how the two eyes travel in one transport frame.

    ``frame_width`` x ``frame_height`` is what ustreamer puts on the wire (1600x600 today, both
    eyes side by side); the camera's own ``width``/``height`` beside this block are ONE EYE.
    ``upside_down`` says the module is mounted turned over, which
    :class:`pepin.stereo.SideBySide` undoes by rotating each half and swapping them.
    ``baseline_m_nominal`` is the vendor's distance between the lenses — a sanity check for a
    calibration's own baseline, never a measurement to compute depth with.
    """

    layout: str
    frame_width: int
    frame_height: int
    upside_down: bool = False
    baseline_m_nominal: float = 0.0
    calibration: str = STEREO_CALIBRATION

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> StereoRig:
        """From a camera block's ``rig``; an unknown layout raises, since a node that guessed
        how the eyes are packed would publish two halves of one picture."""
        layout = str(data["layout"])
        if layout != "side_by_side":
            raise ValueError(f"{layout!r}: the only stereo layout this stack cuts is side_by_side")
        return cls(
            layout=layout,
            frame_width=int(data["frame_width"]),
            frame_height=int(data["frame_height"]),
            upside_down=bool(data.get("upside_down", False)),
            baseline_m_nominal=float(data.get("baseline_m_nominal", 0.0)),
            calibration=str(data.get("calibration", STEREO_CALIBRATION)),
        )

    def calibration_path(self, config_dir: str | Path) -> Path:
        """Where the stereo calibration lives: beside ``config/camera.json``."""
        return Path(config_dir) / self.calibration


@dataclass(frozen=True)
class CameraConfig:
    """One camera of ``config/camera.json``: its stream, image size (one EYE for a stereo rig),
    optics (nominal field of view, and the checkerboard's :class:`Calibration` once there is
    one), mount, and — for a stereo head — the :class:`StereoRig` that says how its two eyes
    arrive. ``name`` is which camera of the file this is, so a node that asked for the active
    one can say which it got."""

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
    calibration: Calibration | None = None
    rig: StereoRig | None = None
    name: str = DEFAULT_CAMERA
    # Where the lens really sits and looks RELATIVE TO ``camera_link``, in the link's own axes
    # (x forward, y left, z up; metres and degrees, the six fields of a mount). The link is the
    # neck's — the board publishes it from the encoders — so a head that is not the one the neck
    # was measured with (a module taped onto the old webcam) says its own offset here, and it
    # goes into ``camera_link -> camera_optical``. Empty: the lens is the link.
    eye: tuple[tuple[str, float], ...] = ()
    # The ``net`` block: the learned stereo matcher this head may be run with, behind the depth
    # node's live ``stereo_matcher`` flag. The numbers belong to the camera and not to the module
    # that loads the network, so nothing is hand-patched into code (CLAUDE.md, the sensor's own
    # properties). An empty ``net_weights`` means the flag cannot be turned on for this head.
    net_weights: str = ""
    net_iters: int = 7
    net_device: str = "mps"
    net_gate: bool = False

    @property
    def stereo(self) -> bool:
        """Whether this camera is a stereo head (its block carries a ``rig``)."""
        return self.rig is not None

    @classmethod
    def from_json(cls, data: dict[str, Any], name: str = DEFAULT_CAMERA) -> CameraConfig:
        """From one camera's block of ``config/camera.json``. The ``intrinsics`` block is read
        only while ``calibrated`` is true: a measurement left in the file but switched off is
        history, not optics. A stereo block carries no intrinsics of its own — its ``calibrated``
        is decided by :meth:`load`, which knows where the calibration file would be."""
        mount = data["mount"]
        frames = data.get("frames", {})
        calibrated = bool(data.get("calibrated", False))
        block = data.get("intrinsics")
        rig = data.get("rig")
        net = data.get("net", {})
        return cls(
            stream=str(data["stream"]),
            width=int(data["width"]),
            height=int(data["height"]),
            hfov_deg=float(data["hfov_deg"]),
            calibrated=calibrated,
            x_m=float(mount["x_m"]),
            y_m=float(mount.get("y_m", 0.0)),
            z_m=float(mount["z_m"]),
            pitch_deg=float(mount.get("pitch_deg", 0.0)),
            link_frame=str(frames.get("link", "camera_link")),
            optical_frame=str(frames.get("optical", "camera_optical")),
            calibration=Calibration.from_json(block) if calibrated and block else None,
            rig=StereoRig.from_json(rig) if rig else None,
            name=name,
            eye=tuple(
                (key, float(value))
                for key, value in data.get("eye", {}).items()
                if isinstance(value, (int, float))
            ),
            net_weights=str(net.get("weights", "")),
            net_iters=int(net.get("iters", 7)),
            net_device=str(net.get("device", "mps")),
            net_gate=bool(net.get("gate", False)),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        name: str | None = None,
        board: str = "127.0.0.1",
        environ: Mapping[str, str] | None = None,
    ) -> CameraConfig:
        """Read ``config/camera.json``, fill the stream's ``{board}`` placeholder, and answer
        with the camera ``name`` — or, with no name (the usual call), with whichever camera is
        ACTIVE: :func:`active_camera` decides, so a node never has a rig's name in its code.

        For a stereo rig ``calibrated`` is answered here and means one thing: the calibration
        file the ``rig`` block names, beside this config, exists and parses. Nothing else in the
        stack has to know where that file lives to say whether the head can measure.
        """
        file = Path(path)
        data = json.loads(file.read_text())
        chosen = active_camera(data, name, environ)
        cfg = cls.from_json(data[chosen], chosen)
        cfg = replace(cfg, stream=cfg.stream.format(board=board))
        if cfg.rig is None:
            return cfg
        return replace(cfg, calibrated=_stereo_calibrated(cfg.rig, file.parent))


def _stereo_calibrated(rig: StereoRig, config_dir: Path) -> bool:
    """Whether the stereo calibration ``rig`` names exists beside the config and loads: a
    half-written or truncated file is not a calibration (:meth:`pepin.stereo.StereoCalibration
    .load` raises on one), and a head that cannot rectify must say so rather than measure."""
    from pepin.stereo import StereoCalibration  # a numpy import a launch file should not pay for

    try:
        StereoCalibration.load(rig.calibration_path(config_dir))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


def camera_names(data: Mapping[str, Any]) -> list[str]:
    """The cameras ``config/camera.json`` holds, in the file's order: a top-level block with a
    ``stream`` is a camera, everything else (``active``, ``notes``) is not."""
    return [
        key
        for key, block in data.items()
        if isinstance(block, dict) and "stream" in block and "mount" in block
    ]


def active_camera(
    data: Mapping[str, Any], name: str | None = None, environ: Mapping[str, str] | None = None
) -> str:
    """Which camera of ``config/camera.json`` this process reads, in precedence order: an
    explicit ``name``, then ``PEPIN_CAMERA`` in the environment (``ros/laptop.sh`` forwards it
    into the container), then the file's ``"active"``, then ``overview``. An empty string
    anywhere is "nobody said", so a launch can pass its argument through unconditionally.

    A name no block answers to raises ``ValueError`` listing the names there are: a typo must
    stop the node at start, never leave it publishing another rig's optics.
    """
    env = os.environ if environ is None else environ
    chosen = (
        name or env.get(CAMERA_ENV) or str(data.get(ACTIVE_KEY, "")) or DEFAULT_CAMERA
    ).strip()
    names = camera_names(data)
    if chosen not in names:
        raise ValueError(
            f"no camera named {chosen!r} in config/camera.json;"
            f" the cameras are {', '.join(names) or 'none'}"
        )
    return chosen


def intrinsics(width: int, height: int, hfov_deg: float) -> tuple[float, float, float, float]:
    """``(fx, fy, cx, cy)`` of a pinhole with square pixels and the principal point centred:
    the focal length that puts ``hfov_deg`` across ``width`` pixels."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return fx, fx, width / 2.0, height / 2.0


def camera_info_arrays(
    width: int, height: int, hfov_deg: float
) -> tuple[list[float], list[float], list[float], list[float]]:
    """``sensor_msgs/CameraInfo``'s ``k`` (3x3), ``d`` (plumb_bob, no distortion), ``r`` (identity)
    and ``p`` (3x4) as flat row-major lists, for an uncalibrated pinhole. A node publishing a
    real camera goes through :func:`optics` instead, which answers the same arrays for a
    calibrated lens too."""
    fx, fy, cx, cy = intrinsics(width, height, hfov_deg)
    k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    d = [0.0, 0.0, 0.0, 0.0, 0.0]
    r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return k, d, r, p


@dataclass(frozen=True)
class Optics:
    """The camera's optics at one image size, whatever their provenance: the pinhole in pixels,
    the lens distortion, and one sentence saying where the numbers came from (for the node's
    report line). Built by :func:`optics`, never by hand."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: tuple[float, ...]
    calibrated: bool
    source: str

    @property
    def hfov_deg(self) -> float:
        """The horizontal field of view these optics see, degrees."""
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.fx)))

    def k(self) -> list[float]:
        """``CameraInfo``'s row-major 3x3 K."""
        return [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]

    def camera_info_arrays(self) -> tuple[list[float], list[float], list[float], list[float]]:
        """``sensor_msgs/CameraInfo``'s ``k`` (3x3), ``d`` (plumb_bob), ``r`` (identity) and
        ``p`` (3x4) as flat row-major lists. ``p`` carries the same pinhole: the published image
        is the raw one unless the node rectifies it, and then it hands us the rectified optics."""
        d = list(self.dist) if self.dist else [0.0, 0.0, 0.0, 0.0, 0.0]
        r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return self.k(), d, r, p


def optics(cfg: CameraConfig, width: int, height: int) -> Optics:
    """The camera's optics at the image size a node publishes: the checkerboard's calibration
    scaled to that size when ``config/camera.json`` carries one, the nominal pinhole of
    ``hfov_deg`` (square pixels, principal point centred, no distortion) otherwise.

    The one reader of the camera's optics. Nothing else in the stack decides between measured
    and nominal — a node asks here and prints :attr:`Optics.source`.

    A stereo rig has no ``intrinsics`` block of its own, so the answer here is always its
    nominal ONE-EYE pinhole; the measured one is the rectified pinhole
    (:class:`pepin.stereo.Rectifier`), which the camera node puts on the wire as ``CameraInfo``.
    The source says which of those two a reader is holding, because a stack quietly measuring
    with a vendor's field of view looks exactly like one measuring with a calibration.
    """
    calibration = cfg.calibration
    if calibration is None:
        fx, fy, cx, cy = intrinsics(width, height, cfg.hfov_deg)
        if cfg.rig is not None:
            source = f"nominal {cfg.hfov_deg:.0f} deg field of view, one eye" + (
                " (the rectified pinhole of config/stereo_calibration.json is on"
                " /camera/camera_info)"
                if cfg.calibrated
                else " (uncalibrated stereo head)"
            )
        else:
            source = f"nominal {cfg.hfov_deg:.0f} deg field of view (uncalibrated)"
        return Optics(fx, fy, cx, cy, width, height, (), False, source)
    scaled = calibration.scaled(width, height)
    return Optics(
        scaled.fx,
        scaled.fy,
        scaled.cx,
        scaled.cy,
        width,
        height,
        scaled.dist,
        True,
        f"calibrated {calibration.date} on {calibration.board},"
        f" rms {calibration.rms:.2f} px, {scaled.hfov_deg():.1f} deg wide",
    )


def write_calibration(path: str | Path, result: Calibration, camera: str = "overview") -> None:
    """Put ``result`` into ``config/camera.json`` as ``<camera>.intrinsics`` and flip
    ``calibrated`` to true; every other key of the file — ``hfov_deg``, the mount, the frames,
    the notes — is left exactly as it was.

    ``hfov_deg`` outside the block stays the nominal field of view on purpose: it is what
    :func:`optics` answers while ``calibrated`` is false, so that boolean is a two-way switch
    and turning a bad calibration off restores the optics of before it. The field of view the
    measured ``fx`` implies is written inside the block (:meth:`Calibration.to_json`).

    The file is rewritten through a temporary file next to it, so an interrupted write never
    leaves the stack with half a camera.
    """
    file = Path(path)
    data = json.loads(file.read_text())
    block = data[camera]
    block["calibrated"] = True
    block["intrinsics"] = result.to_json()
    tmp = file.with_suffix(file.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(file)


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
