#!/usr/bin/env python3
"""The head's calibration dance (docs/head_imu_calibration.md, procedure D): pans and tilts that
excite both of the IMU's rotation axes in front of the AprilGrid, asked of the gaze arbiter (the
neck's one owner) on the operator band.

    uv run python ros/tools/neck_dance.py --check      # one snapshot: does Kalibr see the grid?
    uv run python ros/tools/neck_dance.py              # the plan: poses, grid in view, duration
    uv run python ros/tools/neck_dance.py --centre 0.1 -3.9 --distance 0.28    # from --check
    uv run python ros/tools/neck_dance.py --move       # the arbiter moves the head; wheels never
    uv run python ros/tools/neck_dance.py --scan [--move]   # the first version: one fixed scan

WHERE THE GRID STANDS: upright on a wall. ``--centre PAN TILT`` is the head pose (degrees, pan
left +, tilt down +) that puts the grid's centre at the picture's centre, ``--distance`` the
lens-to-grid distance; ``--check`` measures both from one snapshot and prints them. Without
``--centre`` the grid stands on the robot's centre line ``--distance`` from the lens with the
head level, its centre ``--height`` above the floor: 0.28 m and 1.2 m, the A4 print of
docs/aprilgrid at camera height, its 25 mm tags then 44 px wide (Kalibr's detector found every
tag of a clean render at 19 px and none at 12, the same print 1 m away).

THE DANCE: around the pose that centres the grid in the picture, the largest pan and tilt (inside
the neck's reach, at most ``--max-pan`` 20 and ``--max-tilt`` 15 deg either way) at which
``--min-visible`` of the grid's 36 tags are whole in a picture shrunk by ``--margin`` degrees on
every side (the model's slack: the eye block, the board's placement); the corners (pan and tilt
at once) pulled in along their diagonal until the grid is in view there too. The 188 mm grid
0.28 m away spans 37 deg of the 78 x 62 deg picture, so the dance is about +-17 deg of pan and
+-10 deg of tilt. The poses: the centre, pan alone, tilt alone, the diagonals through the centre,
then the rim (pans at the extreme tilts, tilts at the extreme pans) both ways round, the first and
the last held ``--still`` seconds and every other one ``--hold`` seconds.

Each pose is one ``angles`` look at the ``slow`` speed (the arbiter's ``slow_deg_s``) with ``hold``
and no depth frames: the door answers once the head has settled, the client waits the hold and
asks the next pose, which replaces the last (same source). A step must end within the arbiter's
``move_timeout_s`` or the arbiter gives the look up and the head goes home mid-dance, so the plan
refuses a step longer than that at the slow speed. The neck_goto driver ignores the speed (the
board's own top speed), so ``--move`` refuses to run on it. At the end one ``home`` look.

``--check``: one ustreamer snapshot (``--image`` a saved rectified left eye instead), cut and
rectified as camera_stream does, then Kalibr's own AprilGrid detector on it in pepin-kalibr
(ros/tools/kalibr_detect.py): the tags found, their size in pixels, the room left to the
picture's edges, the grid's distance and the ``--centre``/``--distance`` to dance around. The
head's pose comes from the gaze door (``--head PAN TILT`` for a saved picture). The picture is
kept as ros/maps/rec/calib_check_<UTC>Z.png (PEPIN_CALIB_REC moves it, as for ros/calib_record.sh).

The scan of the first version (``--scan``): sixteen fixed views at the saccade speed, pans +-45 deg
and tilts 18 up .. 60 down, which takes the grid out of the picture at the extremes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config"
DOOR = "http://127.0.0.1:3339"
SNAPSHOT = "http://{board}:8080/snapshot"
KALIBR_IMAGE = "pepin-kalibr"
HOME_PAN_DEG, HOME_TILT_DEG = 0.0, 23.8
FRAMES_PER_VIEW = 10  # still frames after each settle: a second at the camera's 10 Hz
OPERATOR_BAND = 0  # pepin.gaze.OPERATOR: nothing else takes the head while it dances
SOURCE = "calibration"

# The dance's largest swing either way of its centre (the wall board at 0.25-0.30 m: within these
# the grid can stay whole in the picture), and the reach's margin: a view at the reach's very
# edge is one encoder tick from a refusal.
PAN_MAX_DEG = 20.0
TILT_MAX_DEG = 15.0
REACH_MARGIN_DEG = 1.0
SEARCH_STEP_DEG = 0.5
MIN_AMPLITUDE_DEG = 5.0  # below this the dance excites too little to be worth the run
BORDER_PX = 10.0  # a tag this close to the picture's edge is not counted as whole
TAGS = 6
SPACING = 0.3  # Kalibr's tagSpacing: the gap between tags as a fraction of a tag
MIN_TAG_PX = 20.0  # the --check's floor: Kalibr found every tag down to ~19 px on clean renders
MIN_TAGS = 30  # of 36 whole in Kalibr's detection, for --check to call the picture ready
# What a step costs besides its travel at the slow speed: the write's way to the board, the
# arbiter's settle (two still encoder readings), the door's answer.
SETTLE_S = 0.4
STEP_MARGIN_S = 0.5  # a step's travel must end this long before the arbiter's move timeout
TTL_SLACK_S = 5.0
# The pattern in units of the amplitudes: pan +1 is left, tilt -1 is up (+1 down); the first
# and the last pose are the still ones.
PATTERN: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0), (0, 0), (-1, 0), (0, 0),  # pan alone
    (0, -1), (0, 0), (0, 1), (0, 0),  # tilt alone
    (1, -1), (0, 0), (-1, 1), (0, 0), (-1, -1), (0, 0), (1, 1), (0, 0),  # the diagonals
    (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1),  # the rim
    (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1),  # and back round
    (0, 0),
)  # fmt: skip

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Pinhole:
    """The rectified left eye: the picture every consumer of /camera/image sees."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def ray(self, u: float, v: float) -> Array:
        """The unit direction through pixel (u, v) in camera_link's axes (x ahead, y left, z up)."""
        d = np.array([1.0, -(u - self.cx) / self.fx, -(v - self.cy) / self.fy])
        return np.asarray(d / np.linalg.norm(d), dtype=float)


@dataclass(frozen=True)
class Grid:
    """The AprilGrid as it stands, in base_link: its centre, the unit normal of its face (toward
    the robot) and the printed tag's black square (6 tags and 5 gaps of 0.3 tag a side)."""

    centre: tuple[float, float, float]
    normal: tuple[float, float, float]
    tag_m: float

    @property
    def side_m(self) -> float:
        """The grid's side, tags and gaps."""
        return self.tag_m * (TAGS + (TAGS - 1) * SPACING)

    def tag_corners(self) -> Array:
        """The 36 tags' four corners in base_link, shape (36, 4, 3)."""
        normal = np.asarray(self.normal, dtype=float)
        normal /= np.linalg.norm(normal)
        right = np.cross([0.0, 0.0, 1.0], normal)
        right = right / np.linalg.norm(right) if np.linalg.norm(right) > 1e-6 else np.eye(3)[1]
        up = np.cross(normal, right)
        pitch, half = self.tag_m * (1.0 + SPACING), self.side_m / 2
        offsets = ((0.0, 0.0), (self.tag_m, 0.0), (self.tag_m, self.tag_m), (0.0, self.tag_m))
        centre = np.asarray(self.centre, dtype=float)
        return np.asarray(
            [
                [
                    centre + (i * pitch - half + du) * right + (j * pitch - half + dv) * up
                    for du, dv in offsets
                ]
                for i in range(TAGS)
                for j in range(TAGS)
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class Pose:
    """One held view: joint angles (pan left +, tilt below level +, degrees), how long it is
    held after settling, and the share of the grid's tags whole in the picture there."""

    pan_deg: float
    tilt_deg: float
    hold_s: float
    visible: float


def rectified_eye(config_dir: Path = CONFIG) -> Pinhole:
    """The rectified left eye of the active stereo rig (config/stereo_calibration.json)."""
    sys.path.insert(0, str(REPO / "src"))
    from pepin.camera import CameraConfig
    from pepin.stereo import Rectifier, StereoCalibration

    cfg = CameraConfig.load(config_dir / "camera.json")
    if cfg.rig is None:
        raise SystemExit(f"{cfg.name} is not a stereo rig")
    rect = Rectifier.from_calibration(StereoCalibration.load(cfg.rig.calibration_path(config_dir)))
    return Pinhole(rect.fx, rect.fy, rect.cx, rect.cy, rect.width, rect.height)


def neck_config(config_dir: Path = CONFIG) -> Any:
    """config/neck.json as pepin.neck reads it."""
    sys.path.insert(0, str(REPO / "src"))
    from pepin.neck import NeckConfig

    return NeckConfig.from_json(config_dir / "neck.json")


def lens(cfg: Any, pan_deg: float, tilt_deg: float) -> tuple[Array, Array]:
    """The lens's position and camera_link's rotation (x ahead, y left, z up) in base_link with
    the head at these angles, through pepin.neck's camera pose."""
    from pepin.neck import NeckAngles, camera_pose

    x, y, z, _roll, pitch, yaw = camera_pose(
        cfg, NeckAngles(math.radians(pan_deg), math.radians(tilt_deg))
    )
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rz = np.array([[cy_, -sy_, 0.0], [sy_, cy_, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return np.array([x, y, z]), rz @ ry


def project(
    cfg: Any, eye: Pinhole, pan_deg: float, tilt_deg: float, points: Array
) -> tuple[Array, Array, Array]:
    """Pixel u, v and the distance ahead of base_link points with the head at these angles."""
    position, rotation = lens(cfg, pan_deg, tilt_deg)
    local = (points - position) @ rotation  # rows: R^T (p - c)
    ahead = local[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = eye.cx - eye.fx * local[..., 1] / ahead
        v = eye.cy - eye.fy * local[..., 2] / ahead
    return u, v, ahead


def visible_fraction(
    cfg: Any,
    eye: Pinhole,
    grid: Grid,
    pan_deg: float,
    tilt_deg: float,
    border_px: float = BORDER_PX,
) -> float:
    """The share of the grid's tags with all four corners inside the picture, ``border_px``
    from its edge, with the head at these angles."""
    u, v, ahead = project(cfg, eye, pan_deg, tilt_deg, grid.tag_corners())
    inside = (
        (ahead > 0.05)
        & (u >= border_px)
        & (u <= eye.width - border_px)
        & (v >= border_px)
        & (v <= eye.height - border_px)
    )
    return float(np.mean(np.all(inside, axis=1)))


def centre_aim(cfg: Any, eye: Pinhole, point: Sequence[float]) -> tuple[float, float]:
    """The joint angles (degrees) that put a base_link point at the picture's centre (not on
    the optical axis: the rectified principal point sits 73 px above the centre)."""
    from pepin.gaze import aim_at_point

    aim = aim_at_point(cfg, (float(point[0]), float(point[1]), float(point[2])))
    pan, tilt = math.degrees(aim.pan_rad), math.degrees(aim.tilt_rad)
    target = np.asarray([point], dtype=float)
    for _ in range(8):
        u, v, _ahead = project(cfg, eye, pan, tilt, target)
        pan -= math.degrees(math.atan((float(u[0]) - eye.width / 2) / eye.fx))
        tilt += math.degrees(math.atan((float(v[0]) - eye.height / 2) / eye.fy))
    return pan, tilt


def _upright(centre: Array, toward: Array, tag_m: float) -> Grid:
    """A grid upright on a wall at ``centre``, its face turned to ``toward`` (level only)."""
    level = np.array([toward[0], toward[1], 0.0])
    normal = level / np.linalg.norm(level)
    return Grid(
        (float(centre[0]), float(centre[1]), float(centre[2])),
        (float(normal[0]), float(normal[1]), float(normal[2])),
        tag_m,
    )


def grid_seen(
    cfg: Any, eye: Pinhole, pan_deg: float, tilt_deg: float, distance_m: float, tag_m: float
) -> Grid:
    """The grid on a wall whose centre is at the picture's centre with the head at (pan, tilt),
    ``distance_m`` from the lens."""
    position, rotation = lens(cfg, pan_deg, tilt_deg)
    direction = rotation @ eye.ray(eye.width / 2, eye.height / 2)
    return _upright(position + distance_m * direction, -direction, tag_m)


def grid_ahead(cfg: Any, distance_m: float, height_m: float, tag_m: float) -> Grid:
    """The grid on a wall on the robot's centre line, its centre ``height_m`` above the floor
    and ``distance_m`` from the lens with the head level."""
    position, _rotation = lens(cfg, 0.0, 0.0)
    rise = height_m - float(position[2])
    ahead = math.sqrt(max(distance_m**2 - rise**2, 0.01))
    return _upright(position + np.array([ahead, 0.0, rise]), np.array([-1.0, 0.0, 0.0]), tag_m)


def envelope(cfg: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    """(pan, tilt) ranges in degrees the dance may use: the neck's reach
    (pepin.neck.angle_limits) less ``REACH_MARGIN_DEG``."""
    from pepin.neck import angle_limits

    (pan_lo, pan_hi), (tilt_lo, tilt_hi) = angle_limits(cfg)
    m = REACH_MARGIN_DEG
    return (
        (math.degrees(pan_lo) + m, math.degrees(pan_hi) - m),
        (math.degrees(tilt_lo) + m, math.degrees(tilt_hi) - m),
    )


def _largest(ok: Callable[[float], bool], limit: float, step: float = SEARCH_STEP_DEG) -> float:
    """The largest multiple of ``step`` up to ``limit`` for which ``ok`` holds, walking out from
    zero and stopping at the first failure."""
    amplitude = 0.0
    while amplitude + step <= limit + 1e-9 and ok(amplitude + step):
        amplitude += step
    return amplitude


def grid_dance(
    cfg: Any,
    eye: Pinhole,
    grid: Grid,
    *,
    min_visible: float = 1.0,
    margin_deg: float = 1.0,
    max_pan_deg: float = PAN_MAX_DEG,
    max_tilt_deg: float = TILT_MAX_DEG,
    hold_s: float = 1.0,
    still_s: float = 3.0,
) -> list[Pose]:
    """The dance's poses for this grid (the module's docstring); ``ValueError`` when the grid
    leaves less than ``MIN_AMPLITUDE_DEG`` of pan or tilt either way."""
    pan0, tilt0 = centre_aim(cfg, eye, grid.centre)
    (pan_lo, pan_hi), (tilt_lo, tilt_hi) = envelope(cfg)
    border = BORDER_PX + math.radians(margin_deg) * eye.fx

    def seen(pan: float, tilt: float) -> bool:
        return visible_fraction(cfg, eye, grid, pan, tilt, border) >= min_visible

    if not seen(pan0, tilt0):
        raise ValueError(
            f"the grid is not {min_visible:.0%} in view even centred (pan {pan0:+.1f}, tilt"
            f" {tilt0:.1f} deg, {margin_deg:g} deg of margin): too close or too big"
        )
    pan_amp = _largest(
        lambda a: seen(pan0 + a, tilt0) and seen(pan0 - a, tilt0),
        min(max_pan_deg, pan_hi - pan0, pan0 - pan_lo),
    )
    up = _largest(lambda a: seen(pan0, tilt0 - a), min(max_tilt_deg, tilt0 - tilt_lo))
    down = _largest(lambda a: seen(pan0, tilt0 + a), min(max_tilt_deg, tilt_hi - tilt0))

    # A corner (both joints at once) loses the grid sooner than either edge: it is pulled in
    # along its diagonal, both joints by the same factor.
    def corner(tilt: float) -> float:
        return _largest(
            lambda s: (
                seen(pan0 + s * pan_amp, tilt0 + s * tilt)
                and seen(pan0 - s * pan_amp, tilt0 + s * tilt)
            ),
            1.0,
            step=0.02,
        )

    corners = {-1: corner(-up), 1: corner(down)}
    smallest = min(pan_amp, up, down, *(scale * pan_amp for scale in corners.values()))
    if smallest < MIN_AMPLITUDE_DEG:
        raise ValueError(
            f"the grid leaves only {smallest:.1f} deg of motion somewhere (pan +-{pan_amp:.1f},"
            f" up {up:.1f}, down {down:.1f}, corners at {corners[-1]:.0%}/{corners[1]:.0%} of"
            " that): move it further away"
        )
    poses = []
    last = len(PATTERN) - 1
    for index, (p, t) in enumerate(PATTERN):
        scale = corners[t] if p != 0 and t != 0 else 1.0
        pan = pan0 + scale * p * pan_amp
        tilt = tilt0 + scale * (-up if t < 0 else down if t > 0 else 0.0)
        poses.append(
            Pose(
                round(pan, 2),
                round(tilt, 2),
                still_s if index in (0, last) else hold_s,
                visible_fraction(cfg, eye, grid, pan, tilt),
            )
        )
    return poses


def travel_s(a: Pose, b: Pose, slow_deg_s: float) -> float:
    """How long the step from ``a`` to ``b`` takes at the slow speed: each joint runs at it, so
    the larger of the two angles decides."""
    return max(abs(b.pan_deg - a.pan_deg), abs(b.tilt_deg - a.tilt_deg)) / slow_deg_s


def duration_s(poses: Sequence[Pose], slow_deg_s: float, start: Pose | None = None) -> float:
    """The dance's length: every step's travel and settle plus every hold, from ``start`` (home
    by default)."""
    here = start if start is not None else Pose(HOME_PAN_DEG, HOME_TILT_DEG, 0.0, 0.0)
    total = 0.0
    for pose in poses:
        total += travel_s(here, pose, slow_deg_s) + SETTLE_S + pose.hold_s
        here = pose
    return total


def too_long(
    poses: Sequence[Pose], slow_deg_s: float, move_timeout_s: float, start: Pose | None = None
) -> list[tuple[int, float]]:
    """The steps (index, travel seconds) that would not end ``STEP_MARGIN_S`` inside the
    arbiter's move timeout at the slow speed."""
    here = start if start is not None else Pose(HOME_PAN_DEG, HOME_TILT_DEG, 0.0, 0.0)
    out = []
    for index, pose in enumerate(poses):
        seconds = travel_s(here, pose, slow_deg_s)
        if seconds + STEP_MARGIN_S > move_timeout_s:
            out.append((index, seconds))
        here = pose
    return out


def look_request(pose: Pose, ttl_s: float) -> dict[str, Any]:
    """One pose as the arbiter's look: operator band, slow, held, no depth frames waited for."""
    return {
        "kind": "angles",
        "source": SOURCE,
        "band": OPERATOR_BAND,
        "speed": "slow",
        "hold": True,
        "frames": 0,
        "ttl_s": round(ttl_s, 2),
        "target": {"pan_rad": math.radians(pose.pan_deg), "tilt_rad": math.radians(pose.tilt_deg)},
    }


def home_request() -> dict[str, Any]:
    """The look that ends the dance: home, slow, then the arbiter's standing home holds it."""
    return {"kind": "home", "source": SOURCE, "band": OPERATOR_BAND, "speed": "slow", "ttl_s": 10.0}


def views() -> list[tuple[float, float]]:
    """The first version's scan: (pan, tilt) in degrees, the pan's +-45, the tilt from 18 up to
    60 down, the diagonals, home between groups."""
    home = (HOME_PAN_DEG, HOME_TILT_DEG)
    pans = [(pan, HOME_TILT_DEG) for pan in (45.0, -45.0, 20.0, -20.0)]
    tilts = [(HOME_PAN_DEG, tilt) for tilt in (-18.0, 60.0, 0.0, 45.0)]  # reach: 20 up, 63 down
    diagonals = [(30.0, 0.0), (-30.0, 45.0), (30.0, 45.0), (-30.0, 0.0)]
    return [home, *pans, home, *tilts, home, *diagonals, home]


def request() -> dict[str, Any]:
    """The first version's look: one scan of every view, held FRAMES_PER_VIEW frames each."""
    return {
        "kind": "scan",
        "source": SOURCE,
        "band": OPERATOR_BAND,
        "frames": FRAMES_PER_VIEW,
        "speed": "saccade",
        "ttl_s": 180.0,
        "target": {
            "views": [
                {"pan_rad": math.radians(pan), "tilt_rad": math.radians(tilt)}
                for pan, tilt in views()
            ]
        },
    }


def describe(poses: Sequence[Pose], grid: Grid, slow_deg_s: float) -> list[str]:
    """The plan in lines: one per pose, then the span and the length."""
    lines = [
        f"pose {i:2d}: pan {p.pan_deg:+6.1f} deg, tilt {p.tilt_deg:5.1f} deg down, hold"
        f" {p.hold_s:.1f} s, grid {p.visible:4.0%} in view"
        for i, p in enumerate(poses)
    ]
    pans = [p.pan_deg for p in poses]
    tilts = [p.tilt_deg for p in poses]
    x, y, z = grid.centre
    lines.append(
        f"dance: {len(poses)} poses around pan {poses[0].pan_deg:+.1f}, tilt"
        f" {poses[0].tilt_deg:.1f} deg for a grid at ({x:.2f}, {y:+.2f}, {z:.2f}) m in base_link,"
        f" tags {grid.tag_m * 1000:.1f} mm; pan {min(pans):+.1f}..{max(pans):+.1f} deg, tilt"
        f" {min(tilts):.1f}..{max(tilts):.1f} deg down; about {duration_s(poses, slow_deg_s):.0f} s"
        f" at {slow_deg_s:g} deg/s"
    )
    return lines


# ---- the check: does Kalibr see the grid from here? ---------------------------------------
@dataclass(frozen=True)
class Sighting:
    """What Kalibr's detector made of one picture: whether it accepts it, the grid's corners it
    found (of 144) and the tags whole among them (of 36), the tags' side in pixels, the least
    room to the picture's edge, and the grid's centre in the optical frame (metres; None when
    Kalibr refused the picture)."""

    success: bool
    corners: int
    of: int
    tags: int
    tag_px: float | None
    room_px: float | None
    centre_cam: tuple[float, float, float] | None


def sighting(found: dict[str, Any], eye: Pinhole) -> Sighting:
    """ros/tools/kalibr_detect.py's JSON as a :class:`Sighting`."""
    pixels = np.asarray(found.get("corners_px") or [], dtype=float).reshape(-1, 2)
    room = (
        float(
            min(
                pixels[:, 0].min(),
                pixels[:, 1].min(),
                eye.width - pixels[:, 0].max(),
                eye.height - pixels[:, 1].max(),
            )
        )
        if len(pixels)
        else None
    )
    centre = found.get("centre_cam")
    return Sighting(
        bool(found.get("success")),
        int(found.get("corners", 0)),
        int(found.get("of", TAGS * TAGS * 4)),
        int(found.get("tags", 0)),
        None if found.get("tag_px") is None else float(found["tag_px"]),
        room,
        None if centre is None else (float(centre[0]), float(centre[1]), float(centre[2])),
    )


def placement(
    cfg: Any, eye: Pinhole, seen: Sighting, head: tuple[float, float]
) -> tuple[float, float, float] | None:
    """(pan, tilt, distance) that centre the grid in the picture, from a sighting with the head
    at ``head`` (degrees): the optical frame's centre into camera_link, then base_link."""
    if seen.centre_cam is None:
        return None
    x, y, z = seen.centre_cam
    position, rotation = lens(cfg, *head)
    point = position + rotation @ np.array([z, -x, -y])  # optical (right, down, ahead) -> link
    pan, tilt = centre_aim(cfg, eye, point)
    return pan, tilt, float(math.sqrt(x * x + y * y + z * z))


def verdict(seen: Sighting) -> list[str]:
    """What stands between this picture and a good recording; empty when nothing does."""
    out = []
    if not seen.success:
        out.append("Kalibr does not accept the picture (fewer than 7 tags found)")
    elif seen.tags < MIN_TAGS:
        out.append(f"{seen.tags} of 36 tags, under {MIN_TAGS}: part of the grid hidden or blurred")
    if seen.tag_px is not None and seen.tag_px < MIN_TAG_PX:
        out.append(f"tags {seen.tag_px:.0f} px, under {MIN_TAG_PX:.0f}: bring the grid closer")
    if seen.room_px is not None and seen.room_px < BORDER_PX:
        out.append(f"the grid touches the picture's edge ({seen.room_px:.0f} px)")
    return out


def snapshot_left(board: str, config_dir: Path = CONFIG) -> Any:
    """One ustreamer snapshot as camera_stream would publish its left eye: cut, turned upright,
    rectified (uv's OpenCV: its focal is 0.2 % off camera_stream's, nothing here minds)."""
    import cv2

    sys.path.insert(0, str(REPO / "src"))
    from pepin.camera import CameraConfig
    from pepin.stereo import Rectifier, SideBySide, StereoCalibration

    cfg = CameraConfig.load(config_dir / "camera.json")
    if cfg.rig is None:
        raise SystemExit(f"{cfg.name} is not a stereo rig")
    with urllib.request.urlopen(SNAPSHOT.format(board=board), timeout=5.0) as response:
        jpeg = response.read()
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise SystemExit("the snapshot did not decode")
    left, right = SideBySide(cfg.rig.upside_down).eyes(frame)
    rect = Rectifier.from_calibration(StereoCalibration.load(cfg.rig.calibration_path(config_dir)))
    rect_left, _rect_right = rect.rectify(left, right)
    return rect_left


def kalibr_detect(image_path: Path, eye: Pinhole, tag_m: float) -> dict[str, Any]:
    """ros/tools/kalibr_detect.py in pepin-kalibr on one picture (cam0 = this eye)."""
    folder = image_path.parent
    (folder / "camchain.yaml").write_text(
        "cam0:\n  camera_model: pinhole\n"
        f"  intrinsics: [{eye.fx}, {eye.fy}, {eye.cx}, {eye.cy}]\n"
        "  distortion_model: radtan\n  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
        f"  resolution: [{eye.width}, {eye.height}]\n  rostopic: /camera/image\n"
    )
    (folder / "april.yaml").write_text(
        f"target_type: aprilgrid\ntagCols: 6\ntagRows: 6\ntagSize: {tag_m}\ntagSpacing: {SPACING}\n"
    )
    run = subprocess.run(
        [
            "docker", "run", "--rm", "--network", "none",
            "-v", f"{folder}:/k", "-v", f"{REPO / 'ros/tools'}:/tools:ro",
            "--entrypoint", "bash", KALIBR_IMAGE, "-c",
            "source /catkin_ws/devel/setup.bash && python3 /tools/kalibr_detect.py"
            f" /k/{image_path.name} /k/camchain.yaml /k/april.yaml",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )  # fmt: skip
    lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
    if run.returncode != 0 or not lines:
        raise SystemExit(f"kalibr_detect failed ({run.returncode}): {run.stderr.strip()[-400:]}")
    found: dict[str, Any] = json.loads(lines[-1])
    return found


def check(args: argparse.Namespace) -> int:
    """One picture through Kalibr's detector; the verdict and where to dance."""
    import cv2

    cfg, eye = neck_config(), rectified_eye()
    head = tuple(args.head) if args.head else None
    if args.image is not None:
        left = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
        if left is None:
            raise SystemExit(f"no picture at {args.image}")
    else:
        if head is None:
            try:
                from pepin.gaze_link import ask

                state = ask(args.door, "/state")
                head = (math.degrees(state["pan_rad"]), math.degrees(state["tilt_rad"]))
            except (OSError, KeyError, TypeError) as exc:
                print(f"!! the gaze door did not say where the head is ({exc}): home assumed")
        left = snapshot_left(os.environ.get("PEPIN_HOST", "10.0.0.187"))
    head = head if head is not None else (HOME_PAN_DEG, HOME_TILT_DEG)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + "Z"
    rec = Path(os.environ.get("PEPIN_CALIB_REC", str(REPO / "ros/maps/rec")))
    kept = rec / f"calib_check_{stamp}.png"
    rec.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(kept), left)
    with tempfile.TemporaryDirectory(dir=rec) as tmp:
        picture = Path(tmp) / "left.png"
        cv2.imwrite(str(picture), left)
        seen = sighting(kalibr_detect(picture, eye, args.tag), eye)
    print(f"picture: {kept} (head pan {head[0]:+.1f}, tilt {head[1]:.1f} deg)")
    print(
        f"grid: Kalibr {'accepts' if seen.success else 'REFUSES'} it; {seen.tags} of 36 tags"
        f" ({seen.corners} of {seen.of} corners); tag side"
        f" {'-' if seen.tag_px is None else f'{seen.tag_px:.1f}'} px; room to the edge"
        f" {'-' if seen.room_px is None else f'{seen.room_px:.0f}'} px"
    )
    where = placement(cfg, eye, seen, (float(head[0]), float(head[1])))
    if where is not None:
        pan, tilt, distance = where
        print(f"grid centre {distance:.2f} m from the lens; centred in the picture at pan"
              f" {pan:+.1f}, tilt {tilt:.1f} deg")  # fmt: skip
        print(f"dance with: --centre {pan:.1f} {tilt:.1f} --distance {distance:.2f}")
    problems = verdict(seen)
    for problem in problems:
        print(f"!! {problem}")
    print("OK: ready to record" if not problems else "NOT READY")
    return 0 if not problems else 1


def _log(path: Path | None, row: dict[str, Any]) -> None:
    if path is not None:
        with path.open("a") as out:
            out.write(json.dumps(row) + "\n")


def dance(
    poses: Sequence[Pose], door: str, slow_deg_s: float, move_timeout_s: float, log: Path | None
) -> int:
    """Ask the arbiter for every pose in turn, then home; 0 when every pose was reached."""
    from pepin.gaze_link import ask

    state = ask(door, "/state")
    print(f"arbiter: phase {state.get('phase')}, driver {state.get('driver')}")
    if state.get("driver") != "neck_target":
        print("the arbiter drives the neck with neck_goto, which ignores the slow speed: refused")
        return 2
    if state.get("driving"):
        print("a drive is running: refused (the dance is for a parked cart)")
        return 2
    here = Pose(HOME_PAN_DEG, HOME_TILT_DEG, 0.0, 0.0)
    code = 0
    try:
        for index, pose in enumerate(poses):
            ttl = travel_s(here, pose, slow_deg_s) + SETTLE_S + pose.hold_s + TTL_SLACK_S
            sent = time.time()
            answer = ask(
                door, "/look", look_request(pose, ttl), timeout_s=ttl + move_timeout_s + 5.0
            )
            _log(log, {"pose": index, **pose.__dict__, "sent_unix": sent, "answer": answer})
            reached = answer.get("status") == "done" and answer.get("reached")
            print(
                f"pose {index:2d}: {answer.get('status')} in {answer.get('took_ms')} ms"
                + ("" if reached else f": {answer.get('reason')}")
            )
            if not reached:
                code = 1
                break
            time.sleep(pose.hold_s)
            here = pose
    except KeyboardInterrupt:
        print("interrupted: home")
        code = 130
    finally:
        answer = ask(door, "/look", home_request(), timeout_s=30.0)
        _log(log, {"pose": "home", "sent_unix": time.time(), "answer": answer})
        print(f"home: {answer.get('status')}")
    return code


def main(argv: list[str] | None = None) -> int:
    """Print the plan; with --move, ask the arbiter for it; with --check, look first."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--move", action="store_true", help="ask the arbiter (default: print)")
    parser.add_argument("--check", action="store_true", help="one snapshot through Kalibr")
    parser.add_argument("--scan", action="store_true", help="the first version's one scan")
    parser.add_argument("--door", default=DOOR)
    parser.add_argument(
        "--centre", type=float, nargs=2, metavar=("PAN", "TILT"), help="deg: grid centred there"
    )
    parser.add_argument("--distance", type=float, default=0.28, help="m, lens to grid centre")
    parser.add_argument("--height", type=float, default=1.2, help="m, without --centre")
    parser.add_argument("--tag", type=float, default=0.025, help="m, the printed square")
    parser.add_argument("--min-visible", type=float, default=1.0, help="share of tags in view")
    parser.add_argument("--margin", type=float, default=1.0, help="deg of picture kept free")
    parser.add_argument("--max-pan", type=float, default=PAN_MAX_DEG, help="deg either way")
    parser.add_argument("--max-tilt", type=float, default=TILT_MAX_DEG, help="deg either way")
    parser.add_argument("--hold", type=float, default=1.0, help="s at each pose")
    parser.add_argument("--still", type=float, default=3.0, help="s at the first and last")
    parser.add_argument("--slow-deg-s", type=float, default=20.0, help="the arbiter's live one")
    parser.add_argument("--move-timeout-s", type=float, default=3.0, help="the arbiter's live one")
    parser.add_argument("--log", type=Path, default=None, help="JSON lines, one per answer")
    parser.add_argument("--image", type=Path, default=None, help="--check a saved left eye")
    parser.add_argument(
        "--head", type=float, nargs=2, metavar=("PAN", "TILT"), help="deg, for --image"
    )
    args = parser.parse_args(argv)
    sys.path.insert(0, str(REPO / "src"))
    if args.check:
        return check(args)
    if args.scan:
        for pan, tilt in views():
            print(f"pan {pan:+6.1f} deg, tilt {tilt:5.1f} deg down, {FRAMES_PER_VIEW} still frames")
        if not args.move:
            print("nothing moved (--move asks the gaze arbiter)")
            return 0
        from pepin.gaze_link import ask

        print(f"scan: {ask(args.door, '/look', request(), timeout_s=200.0)}")
        return 0
    cfg, eye = neck_config(), rectified_eye()
    if args.centre is not None:
        grid = grid_seen(cfg, eye, args.centre[0], args.centre[1], args.distance, args.tag)
    else:
        grid = grid_ahead(cfg, args.distance, args.height, args.tag)
    try:
        poses = grid_dance(
            cfg,
            eye,
            grid,
            min_visible=args.min_visible,
            margin_deg=args.margin,
            max_pan_deg=args.max_pan,
            max_tilt_deg=args.max_tilt,
            hold_s=args.hold,
            still_s=args.still,
        )
    except ValueError as exc:
        print(f"no dance: {exc}")
        return 2
    print("\n".join(describe(poses, grid, args.slow_deg_s)))
    slow = too_long(poses, args.slow_deg_s, args.move_timeout_s)
    if slow:
        index, seconds = slow[0]
        print(
            f"refused: step {index} takes {seconds:.1f} s at {args.slow_deg_s:g} deg/s, past the"
            f" arbiter's move timeout {args.move_timeout_s:g} s less {STEP_MARGIN_S} s"
        )
        return 2
    if not args.move:
        print("nothing moved (--move asks the gaze arbiter)")
        return 0
    return dance(poses, args.door, args.slow_deg_s, args.move_timeout_s, args.log)


if __name__ == "__main__":
    sys.exit(main())
