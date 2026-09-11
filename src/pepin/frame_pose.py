"""One owner of "the pose of a frame": where the cart and its camera were at a stamp.

A depth frame is judged by a scan taken tens of milliseconds earlier and placed in the map by
the tracker's pose at its own exposure; both are questions about the cart's motion in time,
and until now each node asked TF its own way (``depth_stream`` carried the scan through the
odometry with one lookup, ``depth_fusion`` placed the camera with another). :class:`FramePoser`
asks one :class:`PoseHistory` — TF in a node, a tape offline, a fake in a test — and answers
the two questions the pipeline has: :meth:`FramePoser.carry` moves base_link points from the
scan's moment to the frame's through the odometry frame (smooth, so a tracker correction
between the two stamps does not tear the scan), and :meth:`FramePoser.camera_in_map` places the
camera at the frame's stamp for the model. Stamps are seconds; poses are
:class:`pepin.tsdf.RigidPose` (rotation and translation, ``fixed <- frame``).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from pepin.depth import Array, carry
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
    ) -> None:
        self._history = history
        self.base = base
        self.camera = camera
        self.map_frame = map_frame
        self.odom_frame = odom_frame

    def base_in_map(self, stamp: float) -> RigidPose | None:
        """``map <- base_link`` at ``stamp``: where the cart stood."""
        return self._history.pose_at(stamp, self.base, self.map_frame)

    def camera_in_map(self, stamp: float) -> RigidPose | None:
        """``map <- camera_optical`` at ``stamp``: where the picture was taken from, for the
        model; ``None`` when the history cannot say."""
        return self._history.pose_at(stamp, self.camera, self.map_frame)

    def motion(self, from_stamp: float, to_stamp: float) -> RigidPose | None:
        """How base_link moved between two stamps, seen through the odometry frame: the
        transform taking a base_link point of ``from_stamp`` to base_link at ``to_stamp``;
        ``None`` when the odometry does not cover both moments."""
        before = self._history.pose_at(from_stamp, self.base, self.odom_frame)
        after = self._history.pose_at(to_stamp, self.base, self.odom_frame)
        if before is None or after is None:
            return None
        back = after.inverse()
        return RigidPose(
            back.rotation @ before.rotation, back.rotation @ before.translation + back.translation
        )

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


__all__ = ["BASE_FRAME", "CAMERA_FRAME", "MAP_FRAME", "ODOM_FRAME", "FramePoser", "PoseHistory"]
