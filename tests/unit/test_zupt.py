import math
import re
from pathlib import Path

import pytest

from pepin.zupt import (
    GYRO_QUIET_RAD_S,
    REST_ZUPT_VARIANCE,
    UNCLAIMED_VARIANCE,
    ZUPT_HZ,
    ZUPT_RANGES,
    RestEvidence,
    ZuptGate,
    ZuptVerdict,
    gyro_turning,
    rest_zupt_twist_covariance,
    zupt_clamped,
    zupt_setting_ok,
)

# The bridge's own numbers: imu_bias_s (the settle window), cmd_timeout_s (a command's life) and
# kStateGapMaxS (how old a witness may be).
SETTLE_S = 2.0
COMMAND_HOLD_S = 0.5
MAX_GAP_S = 1.0

# The contract table: (now, still_since, command_at, gyro_at, gyro_turn_at, settle_s, hold_s,
# verdict words). settle_s and hold_s are the bridge's live zupt_settle_s and zupt_cmd_hold_s;
# all but the last five rows hold them at their defaults (imu_bias_s 2.0, cmd_timeout_s 0.5).
CONTRACT_ROWS = [
    (100.0, 40.0, 0.0, 99.98, 0.0, 2.0, 0.5, "at rest"),
    (100.0, 0.0, 0.0, 99.98, 0.0, 2.0, 0.5, "the wheels do not witness rest"),
    (100.0, 98.5, 0.0, 99.98, 0.0, 2.0, 0.5, "settling after the last motion"),
    (100.0, 98.0, 0.0, 99.98, 0.0, 2.0, 0.5, "at rest"),  # exactly the settle window: rest
    (100.0, 40.0, 99.9, 99.98, 0.0, 2.0, 0.5, "a command is live"),
    (100.0, 40.0, 99.5, 99.98, 0.0, 2.0, 0.5, "at rest"),  # exactly the hold: over
    (100.0, 40.0, 0.0, 0.0, 0.0, 2.0, 0.5, "no gyro reading to judge by"),
    (100.0, 40.0, 0.0, 98.9, 0.0, 2.0, 0.5, "no gyro reading to judge by"),
    (100.0, 40.0, 0.0, 99.0, 0.0, 2.0, 0.5, "at rest"),  # exactly max_gap_s old: still fresh
    # the gyro: its newest sample a turn, a turn inside the settle window, and one served
    (100.0, 40.0, 0.0, 99.98, 99.98, 2.0, 0.5, "the gyro reports a turn"),
    (100.0, 40.0, 0.0, 99.98, 98.5, 2.0, 0.5, "the gyro reports a turn"),
    (100.0, 40.0, 0.0, 99.98, 98.0, 2.0, 0.5, "at rest"),
    # several vetoes at once: the command first, the wheels before the gyro, stale before turning
    (100.0, 0.0, 99.9, 0.0, 99.98, 2.0, 0.5, "a command is live"),
    (100.0, 99.0, 0.0, 0.0, 99.98, 2.0, 0.5, "settling after the last motion"),
    (100.0, 40.0, 0.0, 98.0, 97.9, 2.0, 0.5, "no gyro reading to judge by"),
    # the live windows moved: a longer settle, a shorter one, a longer hold, no hold at all, and
    # the gyro's hold following the settle window
    (100.0, 97.0, 0.0, 99.98, 0.0, 5.0, 0.5, "settling after the last motion"),
    (100.0, 99.5, 0.0, 99.98, 0.0, 0.5, 0.5, "at rest"),
    (100.0, 40.0, 98.0, 99.98, 0.0, 2.0, 3.0, "a command is live"),
    (100.0, 40.0, 99.95, 99.98, 0.0, 2.0, 0.0, "at rest"),
    (100.0, 40.0, 0.0, 99.98, 97.0, 5.0, 0.5, "the gyro reports a turn"),
]


def gate() -> ZuptGate:
    return ZuptGate(SETTLE_S, COMMAND_HOLD_S, MAX_GAP_S)


def parked(now: float, **changes: float) -> RestEvidence:
    """A cart at rest for a minute, the gyro sampled 20 ms ago, no command and no turn ever."""
    fields = {"still_since": now - 60.0, "command_at": 0.0, "gyro_at": now - 0.02}
    fields.update(changes)
    return RestEvidence(**fields)


def test_a_parked_cart_with_every_witness_agreeing_is_at_rest() -> None:
    assert gate().judge(100.0, parked(100.0)) is ZuptVerdict.AT_REST


def test_the_wheels_must_witness_rest_past_the_settle_window() -> None:
    """The same rest the gyro's bias tracker trusts: nothing while the wheels are not still, and
    nothing for ``imu_bias_s`` after they stop -- the chassis is still rocking on its tyres."""
    assert gate().judge(100.0, parked(100.0, still_since=0.0)) is ZuptVerdict.NO_REST
    assert gate().judge(100.0, parked(100.0, still_since=98.5)) is ZuptVerdict.SETTLING
    assert gate().judge(100.0, parked(100.0, still_since=98.0)) is ZuptVerdict.AT_REST


def test_a_fresh_command_stops_the_update_before_any_wheel_turns() -> None:
    """Nav2's first twist reaches the bridge before the board has applied it and long before an
    encoder ticks; the update stops on the command itself, and resumes once it is older than the
    bridge's own cmd_timeout_s and the wheels still say rest."""
    assert gate().judge(100.0, parked(100.0, command_at=99.95)) is ZuptVerdict.COMMANDED
    assert gate().judge(100.0, parked(100.0, command_at=99.6)) is ZuptVerdict.COMMANDED
    assert gate().judge(100.0, parked(100.0, command_at=99.4)) is ZuptVerdict.AT_REST


def test_without_a_gyro_reading_there_is_no_update() -> None:
    """No IMU, no bias block yet, or reads failing: a cart turned by hand on still wheels cannot
    be told from a parked one, so nothing is claimed -- silence, not a guess."""
    assert gate().judge(100.0, parked(100.0, gyro_at=0.0)) is ZuptVerdict.NO_GYRO
    assert gate().judge(100.0, parked(100.0, gyro_at=98.9)) is ZuptVerdict.NO_GYRO
    assert gate().judge(100.0, parked(100.0, gyro_at=99.0)) is ZuptVerdict.AT_REST


def test_a_hand_turn_on_still_wheels_is_not_frozen() -> None:
    """The case the wheels cannot see (the cart lifted, or pivoted on locked wheels): the gyro
    reports the turn, the update stops on the newest sample, and returns only after the settle
    window served after the last loud one -- exactly as after a wheel's move."""
    now = 100.0
    assert gate().judge(now, parked(now, gyro_turn_at=now - 0.02)) is ZuptVerdict.GYRO_TURNING
    assert gate().judge(now, parked(now, gyro_turn_at=now - 1.9)) is ZuptVerdict.GYRO_TURNING
    assert gate().judge(now, parked(now, gyro_turn_at=now - 2.0)) is ZuptVerdict.AT_REST


def test_the_newest_sample_turning_vetoes_even_with_no_settle_window() -> None:
    """``imu_bias_s`` 0 means no settle window; the gyro's check must not vanish with it."""
    no_settle = ZuptGate(0.0, COMMAND_HOLD_S, MAX_GAP_S)
    now = 100.0
    turning_now = parked(now, gyro_at=now - 0.02, gyro_turn_at=now - 0.02)
    assert no_settle.judge(now, turning_now) is ZuptVerdict.GYRO_TURNING
    quiet_since = parked(now, gyro_at=now - 0.02, gyro_turn_at=now - 0.04)
    assert no_settle.judge(now, quiet_since) is ZuptVerdict.AT_REST


def test_a_turn_is_judged_on_the_bias_corrected_rate_with_nan_as_a_turn() -> None:
    assert not gyro_turning(0.0049, GYRO_QUIET_RAD_S)
    assert not gyro_turning(-0.0049, GYRO_QUIET_RAD_S)
    assert gyro_turning(0.005, GYRO_QUIET_RAD_S), "at the threshold is a turn"
    assert gyro_turning(-0.2, GYRO_QUIET_RAD_S)
    assert gyro_turning(math.nan, GYRO_QUIET_RAD_S), "a reading nobody can judge is not quiet"


def test_the_quiet_threshold_clears_the_parked_chip_and_catches_a_slow_hand_turn() -> None:
    """The default's measurement (scratch/zupt/zupt_gyro_quiet_threshold.py over the 30 s at-rest
    trace scratch/imu_level_gyro.csv): per-sample noise 0.036 deg/s, largest |yaw - bias| 0.139
    deg/s. The threshold must sit well above both, and far below a slow hand turn (a quarter turn
    in 10 s, 9 deg/s)."""
    quiet_deg_s = math.degrees(GYRO_QUIET_RAD_S)
    assert quiet_deg_s == pytest.approx(0.286, abs=1e-3)
    assert quiet_deg_s / 0.0362 > 7.0, "sigmas of the parked per-sample noise"
    assert quiet_deg_s / 0.139 > 2.0, "times the largest parked deviation seen"
    assert 9.0 / quiet_deg_s > 30.0, "a slow hand turn is far over it"


def test_the_update_claims_the_three_fused_velocities_and_nothing_else() -> None:
    """vx, vy and vyaw at 1e-6 -- the indices ekf.yaml's odom2 fuses -- and a huge number on
    the three it does not. Tighter than every source it has to out-vote at rest: the gyro 4e-4,
    rf2o 2.5e-3 on vyaw and 9e-4 on vx, the wheels 1e-3."""
    matrix = rest_zupt_twist_covariance()
    assert len(matrix) == 36
    diagonal = [matrix[i * 6 + i] for i in range(6)]
    assert diagonal == [
        REST_ZUPT_VARIANCE,
        REST_ZUPT_VARIANCE,
        UNCLAIMED_VARIANCE,
        UNCLAIMED_VARIANCE,
        UNCLAIMED_VARIANCE,
        REST_ZUPT_VARIANCE,
    ]
    assert sum(abs(v) for v in matrix) == pytest.approx(sum(diagonal)), "diagonal only"
    for other in (4e-4, 2.5e-3, 9e-4, 1e-3):
        assert other >= REST_ZUPT_VARIANCE * 100.0, "a hundredfold under every other source"
    assert REST_ZUPT_VARIANCE > 1e-9, "robot_localization raises anything under 1e-9 to 1e-9"
    live = rest_zupt_twist_covariance(var_linear=1e-4, var_yaw=4e-4)
    assert [live[i * 6 + i] for i in range(6)] == [1e-4, 1e-4, 1e6, 1e6, 1e6, 4e-4], (
        "zupt_var_linear on vx and vy, zupt_var_yaw on vyaw"
    )


def test_every_live_setting_has_a_sane_range_and_its_default_is_inside_it() -> None:
    """The bridge refuses a ``ros2 param set`` outside these ranges (and a NaN) and keeps the
    value in force; the defaults -- imu_bias_s's 2.0 for the settle window, cmd_timeout_s's 0.5
    for the hold -- sit inside them."""
    defaults = {
        "zupt_rate_hz": ZUPT_HZ,
        "zupt_var_linear": REST_ZUPT_VARIANCE,
        "zupt_var_yaw": REST_ZUPT_VARIANCE,
        "zupt_settle_s": SETTLE_S,
        "zupt_cmd_hold_s": COMMAND_HOLD_S,
        "zupt_gyro_quiet_rad_s": GYRO_QUIET_RAD_S,
    }
    assert set(ZUPT_RANGES) == set(defaults)
    for name, (low, high) in ZUPT_RANGES.items():
        assert low < high, name
        assert zupt_setting_ok(name, defaults[name]), name
        assert zupt_setting_ok(name, low) and zupt_setting_ok(name, high), f"{name}: inclusive"
        assert not zupt_setting_ok(name, math.nan), f"{name}: a NaN is refused"
    assert ZUPT_RANGES["zupt_rate_hz"] == (1.0, 100.0)
    assert zupt_setting_ok("zupt_rate_hz", 50.0)
    assert not zupt_setting_ok("zupt_rate_hz", 0.5) and not zupt_setting_ok("zupt_rate_hz", 101.0)
    assert not zupt_setting_ok("zupt_var_yaw", 1e-10), "under robot_localization's floor"
    assert not zupt_setting_ok("zupt_settle_s", -1.0)
    with pytest.raises(KeyError):
        zupt_setting_ok("imu_bias_s", 2.0)


def test_a_borrowed_default_is_moved_into_range_not_refused() -> None:
    """zupt_settle_s defaults to imu_bias_s and zupt_cmd_hold_s to cmd_timeout_s; those two have
    no such range (imu_bias_s <= 0 means "no calibration"), so their value is clamped, never
    refused -- a launch file must not be able to stop the base from starting."""
    assert zupt_clamped("zupt_settle_s", 2.0) == 2.0
    assert zupt_clamped("zupt_settle_s", -1.0) == 0.0
    assert zupt_clamped("zupt_settle_s", 600.0) == 60.0
    assert zupt_clamped("zupt_cmd_hold_s", math.nan) == 0.0


def test_zupt_contract_the_cpp_bridge_mirrors() -> None:
    """The table the C++ port must reproduce line for line.

    The bridge that runs on the board is the C++ one
    (ros/pepin_base_cpp/include/pepin_base_cpp/zupt.hpp); that package has no ament test target,
    so this is the contract both sides implement -- and ros/pepin_base_cpp/test/zupt_contract.cpp
    replays exactly these rows against the header, verdict words included. The rows: a parked
    cart, each veto on its own and at its boundary, the order the vetoes are reported in when
    several hold at once, and the two live windows (zupt_settle_s, zupt_cmd_hold_s) moved away
    from their defaults. Change the maths here first.
    """
    for now, still_since, command_at, gyro_at, turn_at, settle_s, hold_s, words in CONTRACT_ROWS:
        evidence = RestEvidence(still_since, command_at, gyro_at, turn_at)
        verdict = ZuptGate(settle_s, hold_s, MAX_GAP_S).judge(now, evidence)
        assert verdict.value == words, (still_since, command_at, gyro_at, turn_at, settle_s, hold_s)
    assert {v.value for v in ZuptVerdict} == {row[-1] for row in CONTRACT_ROWS}, (
        "every verdict is a row"
    )


def test_the_cpp_replay_carries_exactly_these_rows() -> None:
    """ros/pepin_base_cpp/test/zupt_contract.cpp is compiled by hand (no ament test target), so
    the one thing this suite can hold is that the table it replays IS this table."""
    source = Path(__file__).resolve().parents[2] / "ros/pepin_base_cpp/test/zupt_contract.cpp"
    number = r"\s*([-\d.]+)\s*,"
    pattern = re.compile(r"\{" + number * 7 + r'\s*"([^"]+)"\}')
    cpp_rows = [
        (*(float(v) for v in match.groups()[:7]), match.group(8))
        for match in pattern.finditer(source.read_text())
    ]
    assert cpp_rows == CONTRACT_ROWS


def test_the_cpp_header_carries_the_same_words_and_numbers() -> None:
    """zupt.hpp's verdict words are what the bridge's report line prints, and its constants are
    the parameters' defaults: both must be this module's, character for character."""
    header = (
        Path(__file__).resolve().parents[2] / "ros/pepin_base_cpp/include/pepin_base_cpp/zupt.hpp"
    ).read_text()
    words = set(re.findall(r'case ZuptVerdict::k\w+: return "([^"]+)";', header))
    assert words == {v.value for v in ZuptVerdict}
    constants = dict(re.findall(r"constexpr double (k\w+) = ([-\d.e]+);", header))
    assert float(constants["kRestZuptVariance"]) == REST_ZUPT_VARIANCE
    assert float(constants["kUnclaimedVariance"]) == UNCLAIMED_VARIANCE
    assert float(constants["kGyroQuietRadS"]) == GYRO_QUIET_RAD_S
    assert float(constants["kZuptHz"]) == ZUPT_HZ
    ranges = re.findall(r'\{"(zupt_\w+)", ([-\d.e]+), ([-\d.e]+)\}', header)
    assert {name: (float(low), float(high)) for name, low, high in ranges} == ZUPT_RANGES
