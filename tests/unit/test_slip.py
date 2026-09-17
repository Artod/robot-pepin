"""Slip seen from the lidar: same picture twice while the wheels claim a step."""

import math

import numpy as np

from pepin.odometry import Pose2D
from pepin.slip import scan_changed, slipping


def _scan(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 1.0 + 3.0 * rng.random(455)


def test_identical_scans_did_not_change() -> None:
    ranges = _scan()
    changed, delta = scan_changed(ranges, ranges.copy())
    assert not changed and delta == 0.0


def test_a_shifted_world_reads_as_changed() -> None:
    ranges = _scan()
    changed, share = scan_changed(ranges, ranges + 0.10)
    assert changed and share == 1.0


def test_a_straight_step_moves_a_quarter_of_the_beams_and_counts_as_motion() -> None:
    # 4 cm forward: beams looking along the motion shorten by ~4 cm, sideways ones barely change
    angles = np.linspace(-math.pi, math.pi, 455, endpoint=False)
    before = np.full(455, 2.0)
    after = before - 0.04 * np.cos(angles)
    changed, share = scan_changed(before, after)
    assert changed and 0.2 < share < 0.8


def test_noise_below_the_threshold_is_not_motion() -> None:
    ranges = _scan()
    jitter = ranges + np.random.default_rng(1).normal(0.0, 0.005, ranges.shape)
    changed, _ = scan_changed(ranges, jitter)
    assert not changed


def test_a_blind_scan_never_counts_as_slip() -> None:
    ranges = _scan()
    mostly_nan = np.full_like(ranges, np.nan)
    mostly_nan[:20] = ranges[:20]
    changed, share = scan_changed(ranges, mostly_nan)
    assert changed and math.isinf(share)
    assert not slipping(Pose2D(0.1, 0.0, 0.0), changed)


def test_slipping_needs_a_wheel_step_and_a_still_world() -> None:
    assert slipping(Pose2D(0.05, 0.0, 0.0), changed=False)
    assert slipping(Pose2D(0.0, 0.0, math.radians(5.0)), changed=False)
    assert not slipping(Pose2D(0.01, 0.0, math.radians(1.0)), changed=False)  # too small to judge
    assert not slipping(Pose2D(0.05, 0.0, 0.0), changed=True)  # the world moved: real motion


def test_the_watch_counts_a_streak_and_never_calls_the_first_scan_slip() -> None:
    """What the tracker used to do inline: the wheels' step since the previous scan against
    whether the picture changed, with the consecutive slipping scans counted."""
    from pepin.slip import SlipWatch

    ranges = _scan()
    watch = SlipWatch()
    assert not watch.observe(ranges, Pose2D(0.0, 0.0, 0.0)), "nothing to compare yet"
    assert watch.streak == 0
    # the wheels claim 10 cm three times over, the picture never changes: slip, said on the third
    for step in (0.10, 0.20, 0.30):
        assert watch.observe(ranges.copy(), Pose2D(step, 0.0, 0.0))
    assert watch.streak == 3
    # the world moves with the wheels again: real motion, and the streak is back to nothing
    assert not watch.observe(ranges + 0.10, Pose2D(0.40, 0.0, 0.0))
    assert watch.streak == 0


def test_the_watch_says_nothing_while_the_cart_stands_still() -> None:
    """A standing cart also shows the same picture twice; without a claimed step that is rest,
    not slip."""
    from pepin.slip import SlipWatch

    ranges, watch = _scan(), SlipWatch()
    watch.observe(ranges, Pose2D(1.0, 2.0, 0.5))
    assert not watch.observe(ranges.copy(), Pose2D(1.0, 2.0, 0.5)) and watch.streak == 0


def test_the_picture_slip_calls_the_wheels_liars_only_on_a_lasting_disagreement() -> None:
    """The wheels claiming speed while the pictures stand still is a slip once it has lasted;
    a fresh picture that agrees, quiet wheels, or no picture at all are never a slip."""
    from pepin.slip import PICTURE_SLIP_HOLD_S, PictureSlip

    watch = PictureSlip()
    assert not watch.feed(0.0, 0.13, 0.13, vo_at=0.0).slipping, "both agree: driving"
    assert not watch.feed(1.0, 0.0, 0.0, vo_at=1.0).slipping, "nobody claims anything"
    first = watch.feed(2.0, 0.13, 0.00, vo_at=2.0)
    assert not first.slipping and "not long enough" in first.reason
    late = watch.feed(2.0 + PICTURE_SLIP_HOLD_S + 0.1, 0.13, 0.00, vo_at=2.4)
    assert late.slipping and watch.slips == 1, "held long enough: the wheels are lying"
    assert not watch.feed(3.0, 0.13, 0.12, vo_at=3.0).slipping, "the picture caught up"
    watch.feed(4.0, 0.13, 0.0, vo_at=4.0)
    stale = watch.feed(4.0 + PICTURE_SLIP_HOLD_S + 0.1, 0.13, 0.0, vo_at=3.0)
    assert not stale.slipping and "no picture" in stale.reason, "a stale picture cannot testify"


def test_a_fresh_slip_mutes_the_wheels_on_their_first_word() -> None:
    """The hold is the price of the first verdict. Once the wheels have been caught, a silence
    that gives their voice back must not buy them another hold's worth of lying: while the slip
    is fresh they are muted again the moment they claim anything."""
    from pepin.slip import PICTURE_SLIP_HOLD_S, PictureSlip

    watch = PictureSlip()
    watch.feed(0.0, 0.13, 0.0, vo_at=0.0)
    assert watch.feed(PICTURE_SLIP_HOLD_S + 0.1, 0.13, 0.0, vo_at=0.4).slipping
    assert not watch.feed(0.7, 0.0, 0.0, vo_at=0.7).slipping, "a muted wheel's silence"
    again = watch.feed(0.8, 0.13, 0.0, vo_at=0.8)
    assert again.slipping, "caught a moment ago: no second hold"
    watch.feed(1.0, 0.13, 0.13, vo_at=1.0)  # the picture catches up: the slip is over
    assert not watch.feed(5.0, 0.13, 0.0, vo_at=5.0).slipping, "long past: the hold is charged"


def test_a_zero_velocity_update_is_tighter_than_the_wheels_it_answers() -> None:
    """The claim "the cart is not moving" must out-weigh the wheels that say otherwise, and it
    must claim nothing about the axes nobody measured."""
    from pepin.slip import ZUPT_SIGMA_M_S, zero_twist_covariance

    matrix = zero_twist_covariance()
    assert len(matrix) == 36
    assert math.isclose(matrix[0], ZUPT_SIGMA_M_S**2) and math.isclose(matrix[7], ZUPT_SIGMA_M_S**2)
    assert matrix[35] > 0.0 and matrix[14] > 1e3, "yaw rate claimed, z and roll left alone"
    assert ZUPT_SIGMA_M_S < 0.03, "tighter than the wheels' own claim, or the lie wins"


def test_a_duplicate_visual_odometry_pose_does_not_read_as_a_standing_picture() -> None:
    """The bridge delivered every pose twice, 2 ms apart: a copy with the same stamp must not
    turn a driving picture into a zero speed, or the slip watch mutes honest wheels mid-drive."""
    from pepin.slip import PictureSpeed

    speed = PictureSpeed()
    speed.feed(0.00, 0.0, stamp=10.0, now=100.00)
    assert math.isclose(speed.feed(0.05, 0.0, stamp=10.2, now=100.20), 0.25, rel_tol=1e-6)
    assert math.isclose(speed.feed(0.05, 0.0, stamp=10.2, now=100.202), 0.25, rel_tol=1e-6)
    assert math.isclose(speed.feed(0.10, 0.0, stamp=10.4, now=100.40), 0.25, rel_tol=1e-6)
    assert speed.at == 100.40, "freshness is the board's clock"
