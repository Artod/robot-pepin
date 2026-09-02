import pytest

from pepin.kinematics import Twist
from pepin.teleop import DriveState, apply_key


def test_forward_and_left_accumulate_in_steps() -> None:
    s = DriveState(linear_step=0.1, angular_step=0.5)
    s = apply_key(apply_key(s, "w"), "w")
    s = apply_key(s, "\x1b[D")
    assert s.twist == Twist(pytest.approx(0.2), pytest.approx(0.5))


def test_space_stops_and_q_quits_stopped() -> None:
    s = apply_key(DriveState(), "w")
    assert apply_key(s, " ").twist == Twist(0.0, 0.0)
    q = apply_key(s, "q")
    assert q.quit and q.twist == Twist(0.0, 0.0)


def test_unknown_keys_are_ignored() -> None:
    s = DriveState()
    assert apply_key(s, "x") == s
