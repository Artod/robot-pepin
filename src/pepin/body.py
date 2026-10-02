"""The cart's own body as a few boxes in base_link, so the camera does not paint the robot.

The head turns to +-156 deg and tilts to 63 deg down, and wherever it looks back or down to a side
the cart's own top shelf, wheels and mast are in the picture: the volume then holds a wall where
the robot stands and the costmap marks the footprint itself. The standard answer renders the
robot's URDF into the depth image and subtracts it (robot_self_filter, realtime_urdf_filter,
MoveIt's shape mask); Pepin has no URDF, a rigid body of a few boxes and an integrator that already
walks every pixel's ray. So the body is ``config/body.json``: boxes in base_link, grown by one
``margin_m``, and :meth:`BodyModel.ray_depth` answers, per pixel, the optical depth at which that
pixel's ray enters the body — ``inf`` where it never does. The integrator
(:meth:`pepin.tsdf.Tsdf.integrate`, ``clip``) then writes nothing on or past that depth: a pixel
that measured the body (or something behind it) measures no room, and a ray that enters the body
carves nothing beyond it.

The answer depends on the camera's pose on the cart and the optics alone, never on the depth, so
:class:`BodyMask` keeps it until the head moves; on a ``ray_stride_px`` grid of rays (4 px, half a
degree on the stereo eye: a centimetre at the body's distance, well inside the 5 cm margin). A box
the camera stands in is skipped and named — a margin that grows a box around the lens would
otherwise blind every pixel.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pepin.depth import Intrinsics
from pepin.frame_pose import same_pose
from pepin.tsdf import RigidPose

BODY_FILE = "body.json"
Float32 = npt.NDArray[np.float32]
Index = npt.NDArray[np.intp]


@dataclass(frozen=True)
class BodyBox:
    """One axis-aligned box of the body in base_link, metres."""

    name: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not all(a < b for a, b in zip(self.lo, self.hi, strict=True)):
            raise ValueError(f"body box {self.name}: min {self.lo} is not below max {self.hi}")

    def grown(self, margin_m: float) -> BodyBox:
        """The same box ``margin_m`` larger on every side."""
        lo = tuple(v - margin_m for v in self.lo)
        hi = tuple(v + margin_m for v in self.hi)
        return replace(self, lo=(lo[0], lo[1], lo[2]), hi=(hi[0], hi[1], hi[2]))

    def holds(self, point: Sequence[float] | npt.NDArray[np.float64]) -> bool:
        """Whether ``point`` (x, y, z in base_link) is inside the box, faces included."""
        return all(a <= float(p) <= b for a, p, b in zip(self.lo, point, self.hi, strict=True))


@dataclass(frozen=True)
class RayDepth:
    """Per ray of a ``stride``-pixel grid, the optical depth (metres along the optical axis) at
    which the ray enters the body; ``inf`` where it never does. :meth:`at` reads it for any
    pixel."""

    z: Float32
    stride: int
    skipped: tuple[str, ...] = ()  # boxes the camera stands in

    def at(self, v: Index, u: Index) -> Float32:
        """The depth limit of pixels (row ``v``, column ``u``): their grid cell's ray."""
        out: Float32 = self.z[v // self.stride, u // self.stride]
        return out

    @property
    def share(self) -> float:
        """The fraction of the picture's rays that enter the body."""
        return float(np.count_nonzero(np.isfinite(self.z))) / max(self.z.size, 1)


@dataclass(frozen=True)
class BodyModel:
    """The body: boxes as measured, the margin every box is grown by, and the ray grid's step."""

    boxes: tuple[BodyBox, ...]
    margin_m: float = 0.05
    stride_px: int = 4

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BodyModel:
        """From ``config/body.json``'s object; ``ValueError`` naming what is wrong."""
        try:
            boxes = tuple(
                BodyBox(
                    str(box["name"]),
                    (float(box["min_m"][0]), float(box["min_m"][1]), float(box["min_m"][2])),
                    (float(box["max_m"][0]), float(box["max_m"][1]), float(box["max_m"][2])),
                )
                for box in data["boxes"]
            )
            margin = float(data.get("margin_m", cls.margin_m))
            stride = int(data.get("ray_stride_px", cls.stride_px))
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(f"body: a box needs a name, min_m and max_m ({exc})") from exc
        if margin < 0.0 or stride < 1:
            raise ValueError(f"body: margin_m {margin} and ray_stride_px {stride} out of range")
        return cls(boxes, margin, stride)

    @classmethod
    def load(cls, path: str | Path | None = None) -> BodyModel:
        """From ``config/body.json`` wherever this library runs, or ``path``."""
        import json

        if path is None:
            from pepin.deployment import config_file

            path = config_file(BODY_FILE)
        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def grown(self) -> tuple[BodyBox, ...]:
        """The boxes as the filter uses them: each grown by :attr:`margin_m`."""
        return tuple(box.grown(self.margin_m) for box in self.boxes)

    def ray_depth(self, intr: Intrinsics, camera: RigidPose) -> RayDepth:
        """Where each ray of the picture enters the body, ``camera`` being ``base_link <-
        camera_optical``; the rays of a :attr:`stride_px` grid, each through its cell's centre
        pixel. A box holding the camera is skipped (and named in the answer)."""
        s = self.stride_px
        rows, cols = math.ceil(intr.height / s), math.ceil(intr.width / s)
        v = np.minimum(np.arange(rows) * s + (s - 1) / 2.0, intr.height - 1)
        u = np.minimum(np.arange(cols) * s + (s - 1) / 2.0, intr.width - 1)
        uu, vv = np.meshgrid(u, v)
        rays = np.stack(
            [(uu - intr.cx) / intr.fx, (vv - intr.cy) / intr.fy, np.ones_like(uu)], axis=-1
        ).reshape(-1, 3)
        direction = rays @ np.asarray(camera.rotation, dtype=float).T  # base_link, z-unit length
        origin = np.asarray(camera.translation, dtype=float)
        entry = np.full(direction.shape[0], np.inf)
        skipped = []
        for box in self.grown:
            if box.holds(origin):
                skipped.append(box.name)
                continue
            entry = np.minimum(entry, _slab_entry(origin, direction, box))
        return RayDepth(entry.reshape(rows, cols).astype(np.float32), s, tuple(skipped))


def _slab_entry(
    origin: npt.NDArray[np.float64], direction: npt.NDArray[np.float64], box: BodyBox
) -> npt.NDArray[np.float64]:
    """Each ray's parameter at its entry into ``box`` (the slab test), ``inf`` for a miss; the
    origin is outside the box."""
    lo, hi = np.array(box.lo), np.array(box.hi)
    parallel = np.abs(direction) < 1e-12
    safe = np.where(parallel, 1.0, direction)
    t1, t2 = (lo - origin) / safe, (hi - origin) / safe
    within = (lo <= origin) & (origin <= hi)  # per axis: a parallel ray inside the slab
    near = np.where(parallel, np.where(within, -np.inf, np.inf), np.minimum(t1, t2)).max(axis=1)
    far = np.where(parallel, np.where(within, np.inf, -np.inf), np.maximum(t1, t2)).min(axis=1)
    hit = (far >= near) & (far > 0.0)
    return np.where(hit, np.maximum(near, 0.0), np.inf)


class BodyMask:
    """The body's :class:`RayDepth` for the camera's current pose and optics, rebuilt only when
    one of them, or the model, moves; ``None`` from :meth:`for_frame` while no ray meets the body
    (the working pose: then the integrator pays nothing at all)."""

    def __init__(self, model: BodyModel) -> None:
        self._model = model
        self._key: tuple[Intrinsics, RigidPose] | None = None
        self._depth: RayDepth | None = None
        self._seen = False  # whether any ray of the last answer meets the body
        self.rebuilds = 0
        self.last_ms = 0.0  # what the last rebuild cost

    @property
    def model(self) -> BodyModel:
        """The body the mask is cut from."""
        return self._model

    @model.setter
    def model(self, model: BodyModel) -> None:
        if model != self._model:
            self._model, self._key = model, None

    def for_frame(self, intr: Intrinsics, camera: RigidPose) -> RayDepth | None:
        """The ray depths for a frame whose camera sat at ``camera`` (``base_link <-
        camera_optical``), or ``None`` when the body is nowhere in its view."""
        key = self._key
        if key is None or key[0] != intr or not same_pose(key[1], camera):
            started = time.perf_counter()
            self._depth = self._model.ray_depth(intr, camera)
            self._seen = bool(np.isfinite(self._depth.z).any())
            self.last_ms = (time.perf_counter() - started) * 1e3
            self._key = (intr, camera)
            self.rebuilds += 1
        return self._depth if self._seen else None

    @property
    def last(self) -> RayDepth | None:
        """The last answer, whether or not the body was in it."""
        return self._depth


__all__ = ["BODY_FILE", "BodyBox", "BodyMask", "BodyModel", "RayDepth"]
