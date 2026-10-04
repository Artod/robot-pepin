"""The robot's own arm — an SO-101 follower on the front of the cart — posed by its joints, so the
camera does not paint it into the volume.

The head looks forward and down, and the arm is in the picture: the volume paints it as an
obstacle a few centimetres ahead of the bumper, the costmap marks it, and the stall look stares
at the robot's own arm. The cart's body is a few fixed boxes (:mod:`pepin.body`); the arm moves,
so its links are oriented boxes (:class:`pepin.body.OrientedBox`) posed by forward kinematics
from its joints, afresh whenever they move.

THE MODEL. The kinematic chain is upstream's URDF, vendored verbatim (``pepin/vendor/so101``:
TheRobotStudio/SO-ARM100 ``so101_new_calib.urdf``, Apache-2.0), parsed here with the standard
library — no URDF or KDL dependency for the laptop's container. Each link carries one or two
boxes fitted to its meshes (config/arm.json ``links``, ``scripts/arm_boxes.py``), every box grown
by ``margin_m``. The URDF's base_link sits on the cart at ``mount``: cart base_link <- arm
base_link, x/y/z and a yaw.

THE JOINTS (config/arm.json ``joints``). ``source: config`` is ``pose_deg``, the pose the arm
was read in: nothing drives the arm today (torque off, parked where it was folded).
``source: topic`` is ``/arm/joint_states`` (sensor_msgs/JointState: the URDF's joint names,
radians in the URDF's convention, stamped at the encoder read), the sample nearest each
observation's stamp; with no sample within ``max_age_s`` of it the pose falls back to
``pose_deg`` and the answer says so (:class:`ArmPose`).

THE RULE is the body's (:meth:`pepin.tsdf.Tsdf.integrate`'s clip): a ray that enters a grown
link writes nothing on or past that entry — a pixel on the arm measures no room, and a depthless
pixel does not carve through the arm into whatever stands behind it — and, because a clip only
stops new writes, every voxel inside a grown link is forgotten after every integration
(:meth:`pepin.worldmap.WorldMap.forget`), whoever painted it: a surface painted before the arm
moved there, the lidar's or a whisker's return off the arm itself.
"""

from __future__ import annotations

import math
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pepin.body import OrientedBox, RayDepth, oriented_ray_depth
from pepin.depth import Intrinsics
from pepin.frame_pose import STILL_RAD, same_pose
from pepin.mounts import rotation_from_rpy
from pepin.tsdf import RigidPose

ARM_FILE = "arm.json"
ARM_TOPIC = "/arm/joint_states"
URDF_PATH = Path(__file__).resolve().parent / "vendor" / "so101" / "so101_new_calib.urdf"
SOURCES = ("config", "topic")
Array = npt.NDArray[np.float64]
IDENTITY = RigidPose(np.eye(3), np.zeros(3))


@dataclass(frozen=True, eq=False)
class UrdfJoint:
    """One joint of the chain: parent -> child at ``origin`` (parent <- joint frame at zero), the
    axis in the joint frame, and the limits (radians or metres)."""

    name: str
    kind: str  # revolute, continuous, prismatic or fixed
    parent: str
    child: str
    origin: RigidPose
    axis: Array
    lower: float
    upper: float

    @property
    def moves(self) -> bool:
        """Whether the joint has a coordinate (everything but ``fixed``)."""
        return self.kind != "fixed"

    def pose(self, q: float) -> RigidPose:
        """parent <- child at coordinate ``q``."""
        if self.kind in ("revolute", "continuous"):
            turn = axis_rotation(self.axis, q)
            return RigidPose(self.origin.rotation @ turn, self.origin.translation)
        if self.kind == "prismatic":
            shift = self.origin.rotation @ (self.axis * q)
            return RigidPose(self.origin.rotation, self.origin.translation + shift)
        return self.origin


def _floats(text: str | None, default: str) -> tuple[float, float, float]:
    values = [float(v) for v in (text if text is not None else default).split()]
    if len(values) != 3:
        raise ValueError(f"URDF: three numbers expected, got {text!r}")
    return values[0], values[1], values[2]


def parse_urdf(text: str) -> tuple[UrdfJoint, ...]:
    """The joints of a URDF document (the ``<joint>`` children of ``<robot>``), in file order;
    ``ValueError`` for a joint without parent, child or a readable origin."""
    root = ET.fromstring(text)
    joints = []
    for node in root.findall("joint"):
        name, kind = node.get("name", ""), node.get("type", "fixed")
        parent, child = node.find("parent"), node.find("child")
        if parent is None or child is None:
            raise ValueError(f"URDF joint {name!r}: no parent or child")
        origin = node.find("origin")
        xyz = _floats(origin.get("xyz") if origin is not None else None, "0 0 0")
        rpy = _floats(origin.get("rpy") if origin is not None else None, "0 0 0")
        axis_node = node.find("axis")
        axis = np.array(_floats(axis_node.get("xyz") if axis_node is not None else None, "1 0 0"))
        norm = float(np.linalg.norm(axis))
        limit = node.find("limit")
        joints.append(
            UrdfJoint(
                name,
                kind,
                parent.get("link", ""),
                child.get("link", ""),
                RigidPose(rotation_from_rpy(*rpy), np.array(xyz)),
                axis / norm if norm > 0.0 else axis,
                float(limit.get("lower", "0")) if limit is not None else 0.0,
                float(limit.get("upper", "0")) if limit is not None else 0.0,
            )
        )
    return tuple(joints)


def axis_rotation(axis: Array, angle: float) -> Array:
    """The rotation by ``angle`` radians about the unit ``axis`` (Rodrigues)."""
    x, y, z = (float(v) for v in axis)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    out: Array = np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)
    return out


def compose(outer: RigidPose, inner: RigidPose) -> RigidPose:
    """``outer`` applied to ``inner``: the pose of ``inner``'s frame in ``outer``'s parent."""
    return RigidPose(
        outer.rotation @ inner.rotation, outer.rotation @ inner.translation + outer.translation
    )


class Chain:
    """A tree of joints from its one root link: every link's pose for a set of joint values."""

    def __init__(self, joints: Sequence[UrdfJoint]) -> None:
        children = {j.child for j in joints}
        roots = {j.parent for j in joints} - children
        if len(roots) != 1:
            raise ValueError(f"URDF: one root link expected, found {sorted(roots)}")
        self.root = roots.pop()
        self.joints = tuple(joints)
        self._below: dict[str, list[UrdfJoint]] = {}
        for joint in joints:
            self._below.setdefault(joint.parent, []).append(joint)

    @property
    def movable(self) -> tuple[str, ...]:
        """The names of the joints with a coordinate, in file order."""
        return tuple(j.name for j in self.joints if j.moves)

    def link_poses(self, angles: Mapping[str, float]) -> dict[str, RigidPose]:
        """root <- link for every link; ``angles`` must name every movable joint
        (``KeyError`` naming the first one missing)."""
        poses = {self.root: IDENTITY}
        todo = [self.root]
        while todo:
            link = todo.pop()
            for joint in self._below.get(link, []):
                q = float(angles[joint.name]) if joint.moves else 0.0
                poses[joint.child] = compose(poses[link], joint.pose(q))
                todo.append(joint.child)
        return poses


@dataclass(frozen=True, eq=False)
class LinkBox:
    """One box of a link in that link's own frame (fitted to its meshes, not yet grown)."""

    link: str
    box: OrientedBox


@dataclass(frozen=True, eq=False)
class ArmModel:
    """The arm: its chain, where it stands on the cart, its links' boxes, the margin they are
    grown by, the ray grid's step, and where its joints come from."""

    chain: Chain
    mount: RigidPose  # cart base_link <- the URDF's base_link
    links: tuple[LinkBox, ...]
    margin_m: float = 0.03
    stride_px: int = 4
    source: str = "config"
    topic: str = ARM_TOPIC
    max_age_s: float = 0.5
    pose: Mapping[str, float] | None = None  # radians, every movable joint; ``pose_deg``
    mount_measured: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], urdf_text: str | None = None) -> ArmModel:
        """From config/arm.json's object and the URDF's text (the vendored one by default);
        ``ValueError`` naming what is wrong."""
        chain = Chain(parse_urdf(urdf_text if urdf_text is not None else URDF_PATH.read_text()))
        known = {j.child for j in chain.joints} | {chain.root}
        try:
            mount = data["mount"]
            pose = RigidPose(
                rotation_from_rpy(0.0, 0.0, math.radians(float(mount["yaw_deg"]))),
                np.array([float(mount["x_m"]), float(mount["y_m"]), float(mount["z_m"])]),
            )
            links = tuple(
                LinkBox(
                    str(item["link"]),
                    OrientedBox(
                        str(item.get("name", item["link"])),
                        np.array([float(v) for v in item["centre_m"]]),
                        rotation_from_rpy(*(math.radians(float(v)) for v in item["rpy_deg"])),
                        np.array([float(v) for v in item["half_m"]]),
                    ),
                )
                for item in data["links"]
            )
            joints = data.get("joints", {})
            pose_deg = joints.get("pose_deg")
            angles = (
                {name: math.radians(float(pose_deg[name])) for name in chain.movable}
                if pose_deg is not None
                else None
            )
            model = cls(
                chain,
                pose,
                links,
                margin_m=float(data.get("margin_m", cls.margin_m)),
                stride_px=int(data.get("ray_stride_px", cls.stride_px)),
                source=str(joints.get("source", cls.source)),
                topic=str(joints.get("topic", cls.topic)),
                max_age_s=float(joints.get("max_age_s", cls.max_age_s)),
                pose=angles,
                mount_measured=bool(mount.get("measured", False)),
            )
        except (KeyError, TypeError, IndexError, AttributeError) as exc:
            raise ValueError(
                f"arm: a mount (x_m y_m z_m yaw_deg) and links are needed ({exc!r})"
            ) from exc
        for link in model.links:
            if link.link not in known:
                raise ValueError(f"arm: link {link.link!r} is not in the URDF ({sorted(known)})")
            if np.any(link.box.half <= 0.0) or link.box.half.shape != (3,):
                raise ValueError(f"arm: box {link.box.name!r} needs three positive half sides")
        if model.source not in SOURCES:
            raise ValueError(f"arm: joints.source {model.source!r} is none of {SOURCES}")
        if model.source == "config" and model.pose is None:
            raise ValueError("arm: joints.source config needs joints.pose_deg")
        if model.margin_m < 0.0 or model.stride_px < 1 or model.max_age_s <= 0.0:
            raise ValueError(
                f"arm: margin_m {model.margin_m}, ray_stride_px {model.stride_px} or max_age_s"
                f" {model.max_age_s} out of range"
            )
        return model

    @classmethod
    def load(cls, path: str | Path | None = None) -> ArmModel:
        """From config/arm.json wherever this library runs, or ``path``."""
        import json

        if path is None:
            from pepin.deployment import config_file

            path = config_file(ARM_FILE)
        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def joint_names(self) -> tuple[str, ...]:
        """The joints a pose must name: the chain's movable ones."""
        return self.chain.movable

    def link_poses(self, angles: Mapping[str, float]) -> dict[str, RigidPose]:
        """cart base_link <- link for every link of the arm at ``angles`` (radians)."""
        return {
            name: compose(self.mount, pose) for name, pose in self.chain.link_poses(angles).items()
        }

    def boxes_at(self, angles: Mapping[str, float]) -> tuple[OrientedBox, ...]:
        """The links' boxes grown by :attr:`margin_m`, in the cart's base_link, at ``angles``."""
        poses = self.link_poses(angles)
        return tuple(link.box.grown(self.margin_m).placed(poses[link.link]) for link in self.links)


@dataclass(frozen=True)
class ArmPose:
    """The joints an observation is cut with, and where they came from: ``source`` is
    ``config``, ``topic`` or ``stale`` (the topic had nothing within ``max_age_s``: the
    configured pose stands in); ``gap_s`` is the topic sample's distance from the stamp."""

    angles: tuple[tuple[str, float], ...]
    source: str
    gap_s: float = math.nan

    @property
    def mapping(self) -> dict[str, float]:
        """The joints by name, radians."""
        return dict(self.angles)

    def near(self, other: ArmPose | None, rad: float = STILL_RAD) -> bool:
        """Whether ``other`` names the same joints, each within ``rad``."""
        if other is None or [n for n, _ in other.angles] != [n for n, _ in self.angles]:
            return False
        pairs = zip(self.angles, other.angles, strict=True)
        return all(abs(a - b) <= rad for (_, a), (_, b) in pairs)


class JointHistory:
    """The arm's recent joint samples (``/arm/joint_states``), thread-safe: the subscriber adds,
    the integrators read the sample nearest their observation's stamp."""

    def __init__(self, keep_s: float = 2.0) -> None:
        self._keep_s = keep_s
        self._samples: deque[tuple[float, dict[str, float]]] = deque()
        self._lock = threading.Lock()
        self.heard = 0

    def add(self, stamp: float, names: Sequence[str], positions: Sequence[float]) -> None:
        """One message: its stamp (seconds) and its joints (radians)."""
        sample = {str(n): float(p) for n, p in zip(names, positions, strict=False)}
        with self._lock:
            self._samples.append((stamp, sample))
            self.heard += 1
            newest = max(s for s, _ in self._samples)
            while self._samples and self._samples[0][0] < newest - self._keep_s:
                self._samples.popleft()

    def newest(self) -> float | None:
        """The newest sample's stamp, or ``None`` before the first."""
        with self._lock:
            return max((s for s, _ in self._samples), default=None)

    def nearest(self, stamp: float) -> tuple[float, dict[str, float]] | None:
        """The sample whose stamp is nearest ``stamp``, or ``None`` before the first."""
        with self._lock:
            if not self._samples:
                return None
            return min(self._samples, key=lambda s: abs(s[0] - stamp))


def arm_pose(model: ArmModel, history: JointHistory | None, stamp: float) -> ArmPose | None:
    """The joints to cut an observation taken at ``stamp`` with (:class:`ArmPose`), or ``None``
    when neither the topic nor the file gives a whole pose."""
    names = model.joint_names
    if model.source == "topic" and history is not None:
        sample = history.nearest(stamp)
        if sample is not None and abs(sample[0] - stamp) <= model.max_age_s:
            joints = sample[1]
            if all(n in joints for n in names):
                return ArmPose(
                    tuple((n, joints[n]) for n in names), "topic", abs(sample[0] - stamp)
                )
    if model.pose is None:
        return None
    word = "config" if model.source == "config" else "stale"
    return ArmPose(tuple((n, float(model.pose[n])) for n in names), word)


class ArmMask:
    """The arm's boxes and ray depths for an observation: the boxes rebuilt only when the joints
    move, the ray grid only when the joints, the camera's pose on the cart or the optics do
    (the :class:`pepin.body.BodyMask` of a body that moves)."""

    def __init__(self, model: ArmModel | None) -> None:
        self._model = model
        self._boxes_key: ArmPose | None = None
        self._boxes: tuple[OrientedBox, ...] = ()
        self._key: tuple[Intrinsics, RigidPose, int, ArmPose] | None = None
        self._depth: RayDepth | None = None
        self._seen = False
        self.rebuilds = 0
        self.last_ms = 0.0  # what the last ray grid cost

    @property
    def model(self) -> ArmModel | None:
        """The arm the mask is cut from; ``None`` cuts nothing."""
        return self._model

    @model.setter
    def model(self, model: ArmModel | None) -> None:
        if model is not self._model:
            self._model, self._boxes_key, self._key = model, None, None

    def boxes(self, pose: ArmPose) -> tuple[OrientedBox, ...]:
        """The grown boxes in the cart's base_link at ``pose``."""
        if self._model is None:
            return ()
        if not pose.near(self._boxes_key):
            self._boxes = self._model.boxes_at(pose.mapping)
            self._boxes_key = pose
        return self._boxes

    def for_frame(
        self, intr: Intrinsics, camera: RigidPose, pose: ArmPose, stride_px: int | None = None
    ) -> RayDepth | None:
        """The ray depths of a frame whose camera sat at ``camera`` (cart base_link <-
        camera_optical) with the arm at ``pose``, on a ``stride_px`` grid (the model's own by
        default); ``None`` when the arm is nowhere in its view."""
        if self._model is None:
            return None
        stride = stride_px if stride_px is not None else self._model.stride_px
        key = self._key
        if (
            key is None
            or key[0] != intr
            or key[2] != stride
            or not same_pose(key[1], camera)
            or not pose.near(key[3])
        ):
            started = time.perf_counter()
            self._depth = oriented_ray_depth(self.boxes(pose), intr, camera, stride)
            self._seen = bool(np.isfinite(self._depth.z).any())
            self.last_ms = (time.perf_counter() - started) * 1e3
            self._key = (intr, camera, stride, pose)
            self.rebuilds += 1
        return self._depth if self._seen else None

    @property
    def last(self) -> RayDepth | None:
        """The last ray grid, whether or not the arm was in it."""
        return self._depth


__all__ = [
    "ARM_FILE",
    "ARM_TOPIC",
    "URDF_PATH",
    "ArmMask",
    "ArmModel",
    "ArmPose",
    "Chain",
    "JointHistory",
    "LinkBox",
    "UrdfJoint",
    "arm_pose",
    "axis_rotation",
    "compose",
    "parse_urdf",
]
