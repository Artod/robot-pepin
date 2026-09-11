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
