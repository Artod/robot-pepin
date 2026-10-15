"""The picture's points as map points for the tools: :mod:`pepin.pixel_to_map` on the robot.

Kept apart from :mod:`pepin.tools.clients` because the locator is numpy's and OpenCV's: a robot's
clients load it when :meth:`pepin.tools.robot.Robot.connect` builds them, not every importer of
the tools.
"""

from __future__ import annotations

import math

from pepin.pixel_to_map import (
    Head,
    HostMono,
    HostStereo,
    LocateError,
    Locator,
    Pose2D,
    UstreamerPairs,
)
from pepin.tools.clients import GoalServer, HeadPose, Neck


class NeckHead:
    """:class:`pepin.pixel_to_map.HeadSource` over the tools' :class:`Neck`."""

    def __init__(self, neck: Neck) -> None:
        """The head the tools turn."""
        self.neck = neck

    def head(self) -> Head:
        """The encoders' newest pose, in radians."""
        pose = self.neck.pose()
        return Head(math.radians(pose.pan_deg), math.radians(pose.tilt_deg))


class GoalPose:
    """:class:`pepin.pixel_to_map.PoseSource` over the goal server's ``where``."""

    def __init__(self, goals: GoalServer) -> None:
        """The goal server the tools drive through."""
        self.goals = goals

    def pose(self) -> Pose2D:
        """``map -> base_link`` now; LocateError when the goal server has no pose."""
        answer = self.goals.where()
        if answer.get("pose") != "tf" or "x" not in answer:
            raise LocateError("the robot does not know where it is on the map (no map pose)")
        return Pose2D(
            float(answer["x"]), float(answer["y"]), math.radians(float(answer.get("yaw_deg", 0)))
        )


class StereoEyes(Locator):
    """:class:`Eyes` live: :class:`pepin.pixel_to_map.Locator` on the board's camera, the
    laptop's depth host (stereo first, the mono network where stereo has nothing) and the
    tools' own neck and goal server; the head is aimed with the same camera geometry."""

    def aim(self, x: float, y: float, z: float, pose: Pose2D) -> HeadPose:
        """Pan and tilt toward the point, the camera's height taken at the working pose."""
        rest = Head(0.0, math.radians(self.geometry.chain.neck.reference.pitch_deg))
        height = self.geometry.chain.camera_height_m(rest)
        return aim_at(x, y, z, pose, height)


def aim_at(x: float, y: float, z: float, pose: Pose2D, camera_height_m: float) -> HeadPose:
    """Pan (left +) and tilt (down +) in degrees that point a camera ``camera_height_m`` above
    ``pose``'s origin at map point ``(x, y, z)``."""
    forward, left = pose.to_base(x, y)
    pan = math.degrees(math.atan2(left, forward))
    tilt = math.degrees(math.atan2(camera_height_m - z, math.hypot(forward, left)))
    return HeadPose(pan, tilt)


def stereo_eyes(
    board: str, camera_port: int, depth_url: str, neck: Neck, goals: GoalServer
) -> StereoEyes:
    """The live :class:`StereoEyes`: the board's ustreamer, the depth host at ``depth_url``
    (RAFT-Stereo, then Depth-Anything's mono), the tools' neck and goal server."""
    return StereoEyes(
        UstreamerPairs(board, camera_port),
        [HostStereo(depth_url), HostMono(depth_url)],
        NeckHead(neck),
        GoalPose(goals),
    )
