import numpy as np

from pepin.kinematics import Twist
from pepin.safety import SafetyBox, guard_forward, nearest_ahead


def test_table_leg_in_the_box_blocks_forward_only() -> None:
    leg = np.array([[0.20, 0.20], [0.21, 0.21], [0.20, 0.19], [0.22, 0.20]])
    assert nearest_ahead(leg) == 0.20
    twist, blocker = guard_forward(Twist(0.2, 0.3), leg)
    assert twist == Twist(0.0, 0.3) and blocker == 0.20
    assert guard_forward(Twist(-0.2, 0.0), leg)[0].linear == -0.2


def test_wall_beside_the_robot_does_not_block() -> None:
    side = np.array([[0.1, 0.6], [0.2, 0.6], [0.3, 0.6], [0.4, 0.61]])
    assert nearest_ahead(side) is None


def test_a_couple_of_stray_points_are_noise() -> None:
    assert nearest_ahead(np.array([[0.2, 0.0], [0.25, 0.05]])) is None
    assert nearest_ahead(np.array([[0.2, 0.0], [0.25, 0.05]]), SafetyBox(min_points=2)) == 0.2


def test_empty_scan_is_clear() -> None:
    assert nearest_ahead(np.zeros((0, 2))) is None
