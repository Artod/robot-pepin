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
