"""The planar pose arithmetic of pepin.measurements."""

from __future__ import annotations

import pytest

from pepin.measurements import compose, inverse
from pepin.odometry import Pose2D


def test_compose_and_inverse_are_each_other_s_undoing() -> None:
    """The two planar helpers the whole graph path is built of: a pose put through a transform
    and then through its inverse is the pose again, whatever the angles."""
    frame, pose = Pose2D(1.5, -2.5, 2.0), Pose2D(-0.25, 0.75, -1.25)
    back = compose(inverse(frame), compose(frame, pose))
    assert (back.x, back.y, back.theta) == pytest.approx((pose.x, pose.y, pose.theta))
