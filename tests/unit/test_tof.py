import pytest


def test_client_parses_split_chunks(monkeypatch) -> None:
    from pepin.tof import TofClient

    c = TofClient("localhost")
    c.feed({"t": 1.0, "front": 250, "left": 442, "right": -1})
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
    assert mounts["left"].y_m > 0 > mounts["right"].y_m  # left is +y in the robot frame
