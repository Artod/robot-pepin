import math
import re
from pathlib import Path

import pytest

from pepin.zupt import (
    GYRO_QUIET_RAD_S,
    REST_ZUPT_VARIANCE,
    UNCLAIMED_VARIANCE,
    RestEvidence,
    ZuptGate,
    ZuptVerdict,
    gyro_turning,
    rest_zupt_twist_covariance,
)

# The bridge's own numbers: imu_bias_s (the settle window), cmd_timeout_s (a command's life) and
# kStateGapMaxS (how old a witness may be).
SETTLE_S = 2.0
COMMAND_HOLD_S = 0.5
MAX_GAP_S = 1.0

# The contract table: (now, still_since, command_at, gyro_at, gyro_turn_at, verdict words).
CONTRACT_ROWS = [
    (100.0, 40.0, 0.0, 99.98, 0.0, "at rest"),
    (100.0, 0.0, 0.0, 99.98, 0.0, "the wheels do not witness rest"),
    (100.0, 98.5, 0.0, 99.98, 0.0, "settling after the last motion"),
    (100.0, 98.0, 0.0, 99.98, 0.0, "at rest"),  # exactly the settle window: rest
    (100.0, 40.0, 99.9, 99.98, 0.0, "a command is live"),
    (100.0, 40.0, 99.5, 99.98, 0.0, "at rest"),  # exactly the hold: over
    (100.0, 40.0, 0.0, 0.0, 0.0, "no gyro reading to judge by"),
    (100.0, 40.0, 0.0, 98.9, 0.0, "no gyro reading to judge by"),
    (100.0, 40.0, 0.0, 99.0, 0.0, "at rest"),  # exactly max_gap_s old: still fresh
    (100.0, 40.0, 0.0, 99.98, 99.98, "the gyro reports a turn"),  # the newest sample
    (100.0, 40.0, 0.0, 99.98, 98.5, "the gyro reports a turn"),  # inside the settle window
    (100.0, 40.0, 0.0, 99.98, 98.0, "at rest"),  # served
    (100.0, 0.0, 99.9, 0.0, 99.98, "a command is live"),  # every veto: the command first
    (100.0, 99.0, 0.0, 0.0, 99.98, "settling after the last motion"),  # wheels before gyro
    (100.0, 40.0, 0.0, 98.0, 97.9, "no gyro reading to judge by"),  # stale before turning
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
    """The default's measurement (scratch/zupt_gyro_quiet_threshold.py over the 30 s at-rest
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


def test_zupt_contract_the_cpp_bridge_mirrors() -> None:
    """The table the C++ port must reproduce line for line.

    The bridge that runs on the board is the C++ one
    (ros/pepin_base_cpp/include/pepin_base_cpp/zupt.hpp); that package has no ament test target,
    so this is the contract both sides implement -- and ros/pepin_base_cpp/test/zupt_contract.cpp
    replays exactly these rows against the header, verdict words included. The rows: a parked
    cart, each veto on its own and at its boundary, and the order the vetoes are reported in when
    several hold at once. Change the maths here first.
    """
    judge = gate().judge
    for now, still_since, command_at, gyro_at, gyro_turn_at, words in CONTRACT_ROWS:
        evidence = RestEvidence(still_since, command_at, gyro_at, gyro_turn_at)
        assert judge(now, evidence).value == words, (still_since, command_at, gyro_at, gyro_turn_at)
    assert {v.value for v in ZuptVerdict} == {row[-1] for row in CONTRACT_ROWS}, (
        "every verdict is a row"
    )


def test_the_cpp_replay_carries_exactly_these_rows() -> None:
    """ros/pepin_base_cpp/test/zupt_contract.cpp is compiled by hand (no ament test target), so
    the one thing this suite can hold is that the table it replays IS this table."""
    source = Path(__file__).resolve().parents[2] / "ros/pepin_base_cpp/test/zupt_contract.cpp"
    number = r"\s*([-\d.]+)\s*,"
    pattern = re.compile(r"\{" + number * 5 + r'\s*"([^"]+)"\}')
    cpp_rows = [
        (*(float(v) for v in match.groups()[:5]), match.group(6))
        for match in pattern.finditer(source.read_text())
    ]
    assert cpp_rows == CONTRACT_ROWS


def test_the_cpp_header_carries_the_same_words_and_numbers() -> None:
    """zupt.hpp's verdict words are what the bridge's report line prints, and its constants are
    the parameters' defaults: both must be this module's, character for character."""
    from pepin.zupt import ZUPT_HZ

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
