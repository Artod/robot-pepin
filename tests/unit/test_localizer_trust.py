"""A distrusted wheel step must not move the tracked pose."""

import math

from synthetic import raycast_room
from test_localization import PILLAR, furnished_room_map

from pepin.localization import Localizer
from pepin.odometry import Pose2D
from pepin.scanmatch import SearchWindow


def test_a_slipping_step_is_consumed_but_not_applied() -> None:
    truth = Pose2D(0.4, -0.3, math.radians(20.0))
    loc = Localizer(furnished_room_map(), truth)
    points = raycast_room(truth, pillar=PILLAR)
    loc.initialize(points, SearchWindow(0.2, 0.05, 10.0, 2.0), global_fallback=False)
    loc.update(Pose2D(0.0, 0.0, 0.0), points)  # first call only records the odometry
    loc.update(Pose2D(0.30, 0.0, 0.0), points, trust_odometry=False)  # wheels: 30 cm; scan: same
    assert math.hypot(loc.pose.x - truth.x, loc.pose.y - truth.y) < 0.06
    loc.update(Pose2D(0.30, 0.0, 0.0), points)  # the step was consumed: nothing left to apply
    assert math.hypot(loc.pose.x - truth.x, loc.pose.y - truth.y) < 0.06
