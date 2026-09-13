"""One owner of "the pose of a frame": where the cart and its camera were at a stamp.

A depth frame is judged by a scan taken tens of milliseconds earlier and placed in the map by
the tracker's pose at its own exposure; both are questions about the cart's motion in time,
and until now each node asked TF its own way (``depth_stream`` carried the scan through the
odometry with one lookup, ``depth_fusion`` placed the camera with another). :class:`FramePoser`
asks one :class:`PoseHistory` — TF in a node, a tape offline, a fake in a test — and answers
the two questions the pipeline has: :meth:`FramePoser.carry` moves base_link points from the
scan's moment to the frame's through the odometry frame (smooth, so a tracker correction
between the two stamps does not tear the scan), and :meth:`FramePoser.camera_in_map` places the
camera at the frame's stamp for the model — and a third, now that the neck moves:
:meth:`FramePoser.camera_in_base` is where the camera sat on the cart at the frame's stamp,
for the pipeline's projections. Stamps are seconds; poses are :class:`pepin.tsdf.RigidPose`
(rotation and translation, ``fixed <- frame``).

The poser is also where the cart's lean enters the pipeline. ``odom -> base_link`` comes from a
planar EKF (``ros/params/ekf.yaml``, two_d_mode: Nav2 and the tracker want it planar and it
stays planar), so a body leaning over a slipper or a threshold is placed as if it stood level —
5 degrees puts a wall 3 m ahead 26 cm out. Hand the poser a :class:`pepin.lean.LeanSource` and
switch :attr:`FramePoser.apply_lean` on, and every pose it answers carries the lean at that
stamp composed on the body's side of the planar pose (``map <- base_link_planar`` then roll and
pitch about base_link's own x and y); switched off, every answer is the single TF lookup it has
always been.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from pepin.depth import Array, carry
from pepin.lean import Lean, LeanSource
from pepin.tsdf import RigidPose

BASE_FRAME = "base_link"
CAMERA_FRAME = "camera_optical"
MAP_FRAME = "map"
ODOM_FRAME = "odom"


class PoseHistory(Protocol):
    """Where a frame was at a time, in a fixed frame: what TF knows in a node, what a tape
    knows offline."""

    def pose_at(self, stamp: float, frame: str, fixed: str) -> RigidPose | None:
        """``fixed <- frame`` at ``stamp`` (seconds), or ``None`` when the history does not
        cover that moment."""
        ...


class FramePoser:
    """The pipeline's two questions about time, answered from one history."""

    def __init__(
        self,
        history: PoseHistory,
        *,
        base: str = BASE_FRAME,
        camera: str = CAMERA_FRAME,
        map_frame: str = MAP_FRAME,
        odom_frame: str = ODOM_FRAME,
        lean: LeanSource | None = None,
        apply_lean: bool = False,
    ) -> None:
        self._history = history
        self.base = base
        self.camera = camera
        self.map_frame = map_frame
        self.odom_frame = odom_frame
        self.lean = lean
        self.apply_lean = apply_lean  # live: the owning node's imu_lean flag writes it

    def lean_at(self, stamp: float) -> Lean | None:
        """The lean this poser would apply at ``stamp``, or ``None`` when it applies none (no
        source, the switch off, or nothing known about that moment) — also what a report line
        prints to show what the switch is doing."""
        if not self.apply_lean or self.lean is None:
            return None
        return self.lean.lean_at(stamp)

    def base_in_map(self, stamp: float) -> RigidPose | None:
        """``map <- base_link`` at ``stamp``: where the cart stood, and — with ``apply_lean``
        and a lean for that moment — how its body sat on the floor."""
        return self._leaned(self._history.pose_at(stamp, self.base, self.map_frame), stamp)

    def camera_in_map(self, stamp: float) -> RigidPose | None:
        """``map <- camera_optical`` at ``stamp``: where the picture was taken from, for the
        model; ``None`` when the history cannot say.

        Level, that is one lookup. Leaning, the chain is split at base_link — the frame that
        leans — and the lean composed between the halves, so the camera swings about the wheels
        by roll and pitch as the body does; a history that cannot serve the split (no edge to
        the camera on its own) answers the one lookup unleaned rather than nothing at all."""
        if self.lean_at(stamp) is None:
            return self._history.pose_at(stamp, self.camera, self.map_frame)
        base = self._leaned(self._history.pose_at(stamp, self.base, self.map_frame), stamp)
        on_cart = self.camera_in_base(stamp)
        if base is None or on_cart is None:
            return self._history.pose_at(stamp, self.camera, self.map_frame)
        return _compose(base, on_cart)

    def camera_in_base(self, stamp: float) -> RigidPose | None:
        """``base_link <- camera_optical`` at ``stamp``: where the camera sat on the cart when
        the picture was taken — live once the neck's encoders publish the edge, so the depth
        pipeline's camera pose is asked here and not read from a file; ``None`` when the
        history has no such edge."""
        return self._history.pose_at(stamp, self.camera, self.base)

    def motion(self, from_stamp: float, to_stamp: float) -> RigidPose | None:
        """How base_link moved between two stamps, seen through the odometry frame: the
        transform taking a base_link point of ``from_stamp`` to base_link at ``to_stamp``;
        ``None`` when the odometry does not cover both moments. With ``apply_lean`` the lean
        of each moment rides along, so a scan taken while the cart leaned one way is carried
        into a frame taken while it leaned another."""
        before = self._leaned(
            self._history.pose_at(from_stamp, self.base, self.odom_frame), from_stamp
        )
        after = self._leaned(self._history.pose_at(to_stamp, self.base, self.odom_frame), to_stamp)
        if before is None or after is None:
            return None
        return _compose(after.inverse(), before)

    def carry(self, points: Array, from_stamp: float, to_stamp: float) -> Array | None:
        """(n, 3) base_link points seen at ``from_stamp`` as base_link sees them at
        ``to_stamp`` (a scan 100 ms older than the frame is 2 degrees stale at 20 deg/s);
        ``None`` when the odometry does not cover the gap — the caller decides whether the
        points may pass as they are."""
        moved = self.motion(from_stamp, to_stamp)
        if moved is None:
            return None
        return carry(np.asarray(points, dtype=float), moved.rotation, moved.translation)

    def to_map(self, points_base: Array, stamp: float) -> Array | None:
        """(n, 3) base_link points of ``stamp`` in the map, or ``None`` without a pose."""
        pose = self.base_in_map(stamp)
        if pose is None:
            return None
        return carry(np.asarray(points_base, dtype=float), pose.rotation, pose.translation)

    def _leaned(self, planar: RigidPose | None, stamp: float) -> RigidPose | None:
        """``planar`` with the lean of ``stamp`` composed on the body's side: the cart pivots
        about the wheels' contact, so the translation is untouched and only the rotation turns.
        The pose as given when no lean applies — bit for bit, not a multiplication by one."""
        lean = self.lean_at(stamp)
        if planar is None or lean is None:
            return planar
        return RigidPose(planar.rotation @ lean.rotation(), planar.translation)


def _compose(outer: RigidPose, inner: RigidPose) -> RigidPose:
    """``outer`` applied to ``inner``: the pose of ``inner``'s frame in ``outer``'s parent."""
    return RigidPose(
        outer.rotation @ inner.rotation, outer.rotation @ inner.translation + outer.translation
    )


__all__ = ["BASE_FRAME", "CAMERA_FRAME", "MAP_FRAME", "ODOM_FRAME", "FramePoser", "PoseHistory"]
