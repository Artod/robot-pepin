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


# -- Reflex: hysteresis and direction -----------------------------------------


def test_reflex_side_hit_blocks_forward_and_turning_toward_it_only() -> None:
    from pepin.safety import Reflex
    from pepin.tof import ReflexConfig, TofRanges

    reflex = Reflex(ReflexConfig(side_stop_m=0.30, side_sensors_look_sideways=True))
    left_hand = TofRanges(None, 0.25, None, 0.0)
    d = reflex.step(Twist(0.15, 0.5), left_hand)  # forward and turning left
    assert d.blocked and d.twist == Twist(0.0, 0.0)
    d = reflex.step(Twist(0.15, -0.5), left_hand)  # turning right is the way out
    assert d.blocked and d.twist == Twist(0.0, -0.5)
    d = reflex.step(Twist(-0.1, 0.0), left_hand)  # backing away is always allowed
    assert not d.blocked


def test_reflex_releases_only_after_the_range_opens_by_the_margin() -> None:
    from pepin.safety import Reflex
    from pepin.tof import ReflexConfig, TofRanges

    reflex = Reflex(ReflexConfig(front_stop_m=0.22), release_margin_m=0.08)
    forward = Twist(0.15, 0.0)
    assert reflex.step(forward, TofRanges(0.21, None, None, 0.0)).blocked
    assert reflex.step(forward, TofRanges(0.26, None, None, 0.0)).blocked  # still inside the band
    assert not reflex.step(forward, TofRanges(0.31, None, None, 0.0)).blocked
    assert not reflex.step(forward, TofRanges(0.26, None, None, 0.0)).blocked  # not re-tripped


def test_reflex_does_not_release_on_a_single_no_return_frame() -> None:
    """VL53L1X says 'no return' both for an empty room and for a failure at point-blank range."""
    from pepin.kinematics import Twist
    from pepin.safety import Reflex
    from pepin.tof import TofRanges

    reflex = Reflex(release_after_none=5)
    forward = Twist(0.15, 0.0)
    assert reflex.step(forward, TofRanges(0.20, None, None, 0.0)).blocked
    assert reflex.step(forward, TofRanges(None, None, None, 0.0)).blocked  # one dropout: still held
    for _ in range(4):
        reflex.step(forward, TofRanges(None, None, None, 0.0))
    assert not reflex.step(forward, TofRanges(None, None, None, 0.0)).blocked  # gone for real
