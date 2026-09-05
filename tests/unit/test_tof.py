import pytest

from pepin.kinematics import Twist
from pepin.tof import ReflexConfig, TofRanges, apply_reflex

FRESH = 0.05


def test_close_obstacle_ahead_blocks_forward_but_keeps_turning() -> None:
    d = apply_reflex(Twist(0.2, 0.5), TofRanges(0.2, 1.0, 1.0, FRESH))
    assert d.blocked and d.twist == Twist(0.0, 0.5) and "front" in d.reason


def test_backing_away_is_never_blocked() -> None:
    d = apply_reflex(Twist(-0.2, 0.0), TofRanges(0.1, 0.1, 0.1, FRESH))
    assert not d.blocked and d.twist.linear == -0.2


def test_clear_path_passes_the_command_through() -> None:
    cmd = Twist(0.25, -0.1)
    assert apply_reflex(cmd, TofRanges(1.5, 0.44, 0.45, FRESH)).twist == cmd  # floor readings


def test_side_sensor_too_close_blocks() -> None:
    d = apply_reflex(Twist(0.2, 0.0), TofRanges(2.0, 0.25, 1.0, FRESH))
    assert d.blocked and "left" in d.reason


def test_no_return_means_nothing_close() -> None:
    assert not apply_reflex(Twist(0.2, 0.0), TofRanges(None, None, None, FRESH)).blocked


def test_stale_data_policy() -> None:
    stale = TofRanges(0.1, 1.0, 1.0, age_s=2.0)
    assert not apply_reflex(Twist(0.2, 0.0), stale).blocked
    assert apply_reflex(Twist(0.2, 0.0), stale, ReflexConfig(blocked_when_stale=True)).blocked


def test_client_parses_split_chunks(monkeypatch) -> None:
    from pepin.tof import TofClient

    c = TofClient("localhost")
    c._ingest({"t": 1.0, "front": 250, "left": 442, "right": -1})
    r = c.ranges()
    assert r.front == 0.25 and r.left == 0.442 and r.right is None and r.age_s < 1.0


# -- mounts: where a return lands in the robot frame ------------------------------


def test_hit_lands_along_the_beam_from_the_sensor_position() -> None:
    from pepin.tof import TofMount

    left = TofMount(x_m=0.027, y_m=0.148, yaw_deg=0.0, height_m=0.16)
    assert left.hit_xy(0.25) == pytest.approx((0.277, 0.148))
    sideways = TofMount(x_m=0.0, y_m=0.1, yaw_deg=90.0, height_m=0.16)
    assert sideways.hit_xy(0.5) == pytest.approx((0.0, 0.6))


def test_config_mounts_load_for_all_three_sensors() -> None:
    from pepin.tof import load_mounts

    mounts = load_mounts("config/tof.json")
    assert set(mounts) == {"front", "left", "right"}
    assert mounts["front"].y_m == 0.0 and mounts["front"].height_m > mounts["left"].height_m
    assert mounts["left"].y_m > 0 > mounts["right"].y_m  # left is +y in the robot frame


def test_sub_minimum_ranges_are_failure_codes_not_obstacles() -> None:
    from pepin.kinematics import Twist
    from pepin.tof import ReflexConfig, TofRanges, apply_reflex

    forward = Twist(0.15, 0.0)
    ghost = TofRanges(front=0.02, left=None, right=None, age_s=0.0)  # VL53L1X "no target" tell
    assert not apply_reflex(forward, ghost, ReflexConfig()).blocked
    real = TofRanges(front=0.15, left=None, right=None, age_s=0.0)
    assert apply_reflex(forward, real, ReflexConfig()).blocked
