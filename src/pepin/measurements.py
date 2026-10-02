"""Planar pose arithmetic the graph path is built of: :func:`compose` and :func:`inverse`.

The board tracker's measurement gate that lived here (a camera pose measured on the laptop and
fused on the board) is on the tag alt/tracker-2026-09-22.
"""

from __future__ import annotations

import math

from pepin.odometry import Pose2D


def compose(outer: Pose2D, inner: Pose2D) -> Pose2D:
    """``inner`` expressed in the frame ``outer`` is expressed in: the planar transform
    ``outer`` applied to the planar pose ``inner``."""
    cos, sin = math.cos(outer.theta), math.sin(outer.theta)
    return Pose2D(
        outer.x + cos * inner.x - sin * inner.y,
        outer.y + sin * inner.x + cos * inner.y,
        math.atan2(math.sin(outer.theta + inner.theta), math.cos(outer.theta + inner.theta)),
    )


def inverse(pose: Pose2D) -> Pose2D:
    """The transform the other way: the frame ``pose`` places, seen from the pose itself."""
    cos, sin = math.cos(pose.theta), math.sin(pose.theta)
    return Pose2D(
        -(cos * pose.x + sin * pose.y),
        -(-sin * pose.x + cos * pose.y),
        math.atan2(math.sin(-pose.theta), math.cos(-pose.theta)),
    )
