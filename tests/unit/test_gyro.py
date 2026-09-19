import math

import pytest

from pepin.gyro import GyroBias, GyroBiasTracker, RestWitness, rest_witnessed

# The node's own numbers, small enough to write a table by hand: 2.0 s per block (imu_bias_s) at
# 2 Hz is four samples a block and a 2.0 s settle window before the first of them.
BLOCK_S = 2.0
RATE_HZ = 2.0
# TwistFromPose's max_gap_s, which the witness borrows: a state stream with a gap this long is not
# a measurement, so the twist re-primes and the rest it was witnessing is over.
MAX_GAP_S = 1.0


def feed(
    tracker: GyroBiasTracker,
    times: list[float],
    value: float,
    still_since: float,
) -> list[bool]:
    """Feed one rest spell: the same reading on all three axes at each of ``times``."""
    return [tracker.update(t, value, value, value, still_since) for t in times]


def test_nothing_is_ready_before_the_wheels_have_witnessed_a_block_of_rest() -> None:
    """The boot case the old code could not see: a node that starts while the cart is being
    pushed had no way to know, took the push as its zero and subtracted it forever. With the
    wheels' word it refuses -- and the bridge publishes nothing while ``ready`` is false."""
    tracker = GyroBiasTracker(BLOCK_S, RATE_HZ)
    assert not tracker.ready
    assert tracker.blocks == 0
    assert tracker.age_s(100.0) == math.inf
    assert not any(feed(tracker, [10.0, 10.5, 11.0, 11.5, 12.0], 0.5, 0.0)), "nobody watching"
    assert not tracker.ready
    assert tracker.bias == GyroBias(0.0, 0.0, 0.0)


def test_the_chassis_settles_for_a_block_before_the_first_sample_counts() -> None:
    """A spell that began at t=10 contributes nothing before t=12: the cart has stopped but is
    still rocking on its tyres, and those samples are not the chip's zero."""
    tracker = GyroBiasTracker(BLOCK_S, RATE_HZ)
    assert not any(feed(tracker, [10.0, 10.5, 11.0, 11.5], 0.9, 10.0)), "settling"
    assert not tracker.ready
    finished = feed(tracker, [12.0, 12.5, 13.0, 13.5], 0.1, 10.0)
    assert finished == [False, False, False, True], "four samples make the block"
    assert tracker.ready and tracker.blocks == 1
    assert tracker.bias.y == pytest.approx(0.1), "the settling samples are not in the mean"
    assert tracker.age_s(13.5) == pytest.approx(0.0)
    assert tracker.block_samples == 4


def test_a_finished_block_replaces_the_bias_and_the_next_one_replaces_that() -> None:
    """A block mean, not a filter: no gain, no time constant, and the second block owes the
    first nothing. Blocks tile back to back -- the settle window is served once per spell, not
    once per block, because the cart has not moved in between."""
    tracker = GyroBiasTracker(BLOCK_S, RATE_HZ)
    feed(tracker, [10.0, 10.5, 11.0, 11.5], 0.0, 10.0)
    for t, value in zip([12.0, 12.5, 13.0, 13.5], [0.1, 0.2, 0.3, 0.4], strict=True):
        tracker.update(t, value, value, value, 10.0)
    assert tracker.bias.z == pytest.approx(0.25), "the mean of the four"
    assert not any(feed(tracker, [14.0, 14.5, 15.0], 1.0, 10.0)), "no settle window again"
    assert tracker.update(15.5, 1.0, 1.0, 1.0, 10.0), "the second block closes here"
    assert tracker.blocks == 2
    assert tracker.bias == GyroBias(1.0, 1.0, 1.0), "replaced, not blended towards 0.25"


def test_motion_in_the_middle_of_a_block_throws_that_block_away() -> None:
    """The cart drove off three samples into a block: the spell's identity changed, the partial
    sum goes in the bin, and the settle window is served again from the new spell. Nothing that
    was measured while the cart moved can reach the bias."""
    tracker = GyroBiasTracker(BLOCK_S, RATE_HZ)
    feed(tracker, [10.0, 11.0, 12.0, 12.5, 13.0], 0.1, 10.0)  # settle, then three of four
    assert not tracker.ready
    assert not any(feed(tracker, [13.5, 14.0, 14.5, 15.0], 9.0, 13.5)), "driving: a new spell"
    assert not any(feed(tracker, [15.5, 16.0], 0.1, 15.5)), "settling again, not counting"
    assert not tracker.ready, "the three samples before the move never became a block"
    finished = feed(tracker, [17.5, 18.0, 18.5, 19.0], 0.1, 15.5)
    assert finished[-1] and tracker.blocks == 1
    assert tracker.bias.x == pytest.approx(0.1), "no trace of the 9.0 read while driving"


def test_a_spell_that_stops_being_witnessed_drops_the_block_in_progress() -> None:
    """The link went down, or /odom was muted, mid-block. Silence is not rest: the wheels stop
    answering, ``still_since`` arrives as 0, and the block starts over when they answer again."""
    tracker = GyroBiasTracker(BLOCK_S, RATE_HZ)
    feed(tracker, [10.0, 12.0, 12.5, 13.0], 0.1, 10.0)  # settle plus three of four
    assert not any(feed(tracker, [13.5, 14.0], 0.1, 0.0)), "nobody watching"
    assert not tracker.ready
    assert not any(feed(tracker, [14.5, 15.0, 15.5, 16.0], 0.1, 14.5)), "settling from scratch"
    assert feed(tracker, [16.5, 17.0, 17.5, 18.0], 0.1, 14.5)[-1], "and then a whole block"


def test_a_block_taken_while_the_cart_was_lifted_and_turned_is_healed_by_the_next() -> None:
    """The one case the wheels cannot see: the cart in the air, wheels free, turned by hand. The
    encoders do not tick, so the spell looks like rest and a block of the turn becomes the bias.
    It is wrong for exactly one block -- put down and left alone, the next block replaces it."""
    tracker = GyroBiasTracker(BLOCK_S, RATE_HZ)
    feed(tracker, [10.0, 11.0], 0.0, 10.0)
    assert feed(tracker, [12.0, 12.5, 13.0, 13.5], 0.5, 10.0)[-1], "0.5 rad/s of hand turn"
    assert tracker.bias.z == pytest.approx(0.5), "a bias that is nothing but the turn"
    assert feed(tracker, [14.0, 14.5, 15.0, 15.5], 0.001, 10.0)[-1]
    assert tracker.bias.z == pytest.approx(0.001), "one block of rest and it is gone"
    assert tracker.blocks == 2


def test_no_calibration_asked_for_publishes_at_once_with_a_zero_bias() -> None:
    """``imu_bias_s`` 0 has always meant "do not calibrate": ready immediately, nothing
    subtracted, and no block ever taken however long the cart stands."""
    tracker = GyroBiasTracker(0.0, 50.0)
    assert tracker.ready and tracker.block_samples == 0
    assert not any(feed(tracker, [1.0, 2.0, 3.0, 4.0, 5.0], 7.0, 1.0))
    assert tracker.bias == GyroBias(0.0, 0.0, 0.0)
    assert tracker.blocks == 0


def test_the_tracker_keeps_no_clock_and_no_thread_of_its_own() -> None:
    """Every time comes from the caller, so the same sequence an hour later is the same result:
    the IMU thread and the reader thread share one monotonic clock and two atomics, never a lock,
    and nothing here reads a clock that could disagree with theirs."""
    early, late = GyroBiasTracker(BLOCK_S, RATE_HZ), GyroBiasTracker(BLOCK_S, RATE_HZ)
    shift = 3600.0
    for t in [10.0, 11.0, 12.0, 12.5, 13.0, 13.5]:
        early.update(t, 0.2, 0.3, 0.4, 10.0)
        late.update(t + shift, 0.2, 0.3, 0.4, 10.0 + shift)
    assert early.bias == late.bias
    assert early.blocks == late.blocks == 1
    assert early.age_s(13.5) == pytest.approx(late.age_s(13.5 + shift))


def test_the_boot_only_bias_is_one_block_dated_before_the_start() -> None:
    """What ``imu_bias_tracking`` false does in the bridge, and why it needs no second code path:
    the node dates the last motion one block before it started, which declares the settle window
    already over, takes exactly one block on trust as it always did, and then stops feeding."""
    tracker = GyroBiasTracker(BLOCK_S, RATE_HZ)
    start = 100.0
    assumed_motion_at = start - BLOCK_S
    finished = feed(tracker, [start, start + 0.5, start + 1.0, start + 1.5], 0.3, assumed_motion_at)
    assert finished == [False, False, False, True], "no settling: the block starts at once"
    assert tracker.bias.x == pytest.approx(0.3) and tracker.blocks == 1


def test_the_wheels_witness_rest_only_while_the_stream_runs_and_nothing_is_commanded() -> None:
    """The three vetoes, one at a time: a twist that is not exactly zero, a live command the
    blocked wheels never answered, and a state line that does not continue the previous one."""
    witness = RestWitness(MAX_GAP_S)
    assert witness.judge(100.0, 10.0, moving=False, twist_is_zero=True) == 10.0, "the first line"
    assert witness.judge(100.05, 10.05, moving=False, twist_is_zero=True) == 10.0, "rest carries"
    assert witness.judge(100.1, 10.1, moving=False, twist_is_zero=False) == 10.1, "the cart moved"
    assert witness.judge(100.15, 10.15, moving=False, twist_is_zero=True) == 10.1, "rest again"
    assert witness.judge(100.2, 10.2, moving=True, twist_is_zero=True) == 10.2, "blocked wheels"
    assert witness.judge(100.25, 10.25, moving=False, twist_is_zero=True) == 10.2, (
        "the command gone: the spell is dated from the motion, not from the first still line "
        "after it, so the settle window is served from when the cart really stopped"
    )
    assert witness.at == 10.25


def test_a_zero_twist_after_a_silence_is_not_a_witness_of_rest() -> None:
    """The trap this class exists for: after a gap the twist estimator re-primes and returns a
    zero twist -- which is not a measurement, and the cart may have been pushed the whole time."""
    witness = RestWitness(MAX_GAP_S)
    witness.judge(100.0, 10.0, moving=False, twist_is_zero=True)
    witness.judge(100.05, 10.05, moving=False, twist_is_zero=True)
    assert witness.judge(103.0, 13.0, moving=False, twist_is_zero=True) == 13.0, "3 s of silence"
    assert witness.judge(103.05, 13.05, moving=False, twist_is_zero=True) == 13.0, "measuring now"
    assert witness.judge(99.0, 14.0, moving=False, twist_is_zero=True) == 14.0, "clock restarted"


def test_muting_odom_makes_the_rest_unknown_rather_than_perfect() -> None:
    """/odom muted is a sensor failure exercised in place: nothing is published, so nothing
    witnesses anything, and the spell starts over when the lines come back."""
    witness = RestWitness(MAX_GAP_S)
    witness.judge(100.0, 10.0, moving=False, twist_is_zero=True)
    witness.judge(100.05, 10.05, moving=False, twist_is_zero=True)
    witness.forget()
    assert witness.since == 0.0, "not at rest: not known to be anything"
    assert witness.judge(100.1, 10.1, moving=False, twist_is_zero=True) == 10.1, "from scratch"


def test_a_witness_nobody_has_refreshed_stops_counting_as_one() -> None:
    """The board's link went down mid-spell. The gyro thread reads the same two numbers as before
    and must not conclude the cart has been standing still for as long as the clock says."""
    assert rest_witnessed(10.0, 10.5, 11.0, MAX_GAP_S) == 10.0, "judged 0.5 s ago: fresh"
    assert rest_witnessed(10.0, 10.5, 11.6, MAX_GAP_S) == 0.0, "judged 1.1 s ago: nobody watching"
    assert rest_witnessed(0.0, 10.5, 10.6, MAX_GAP_S) == 0.0, "never at rest in the first place"


def test_gyro_bias_contract_the_cpp_bridge_mirrors() -> None:
    """The table the C++ port must reproduce line for line.

    The bridge that actually runs on the board is the C++ one
    (ros/pepin_base_cpp/include/pepin_base_cpp/gyro_bias.hpp); that package has no ament test
    target, so this is the contract both sides implement -- and
    ros/pepin_base_cpp/test/gyro_bias_contract.cpp replays exactly these rows against the header.
    The rows: a spell nobody witnesses, the settle window, the four samples of a block, the
    replacement by the block mean, the second block tiling on with no new settle, a move that
    kills a block in progress, and the settle window served again after it. Change the maths here
    first.
    """
    tracker = GyroBiasTracker(BLOCK_S, RATE_HZ)
    samples = [
        # (t, reading on all three axes, still_since, block finished, bias after, blocks after)
        (10.00, 0.90, 0.0, False, 0.00, 0),  # nobody watching: not rest, whatever the reading
        (10.50, 0.90, 10.5, False, 0.00, 0),  # the spell starts here; the chassis is settling
        (12.00, 0.90, 10.5, False, 0.00, 0),  # 1.5 s < 2.0 s: still settling
        (12.50, 0.10, 10.5, False, 0.00, 0),  # 2.0 s: the first sample of the block
        (13.00, 0.20, 10.5, False, 0.00, 0),
        (13.50, 0.30, 10.5, False, 0.00, 0),
        (14.00, 0.40, 10.5, True, 0.25, 1),  # four samples: the mean replaces the bias
        (14.50, 1.00, 10.5, False, 0.25, 1),  # the next block tiles on, no settle window again
        (15.00, 1.00, 10.5, False, 0.25, 1),
        (15.50, 1.00, 10.5, False, 0.25, 1),
        (16.00, 1.00, 10.5, True, 1.00, 2),  # replaced outright, not blended towards 0.25
        (16.50, 0.10, 10.5, False, 1.00, 2),  # a third block opens
        (17.00, 0.10, 10.5, False, 1.00, 2),
        (17.50, 9.00, 17.5, False, 1.00, 2),  # the cart moved: the two samples are thrown away
        (19.00, 0.10, 17.5, False, 1.00, 2),  # 1.5 s since the move: settling, not counting
        (19.50, 0.10, 17.5, False, 1.00, 2),  # 2.0 s: counting again, from one
        (20.00, 0.10, 17.5, False, 1.00, 2),
        (20.50, 0.10, 17.5, False, 1.00, 2),
        (21.00, 0.10, 17.5, True, 0.10, 3),  # and the block after the move closes here
    ]
    for t, reading, still_since, finished, bias, blocks in samples:
        assert tracker.update(t, reading, reading, reading, still_since) is finished, f"t={t}"
        assert tracker.bias.x == pytest.approx(bias, abs=1e-12), f"bias at t={t}"
        assert tracker.bias == GyroBias(tracker.bias.x, tracker.bias.x, tracker.bias.x)
        assert tracker.blocks == blocks, f"blocks at t={t}"
    assert tracker.age_s(26.0) == pytest.approx(5.0)
    assert tracker.block_samples == 4
    assert tracker.ready


def test_a_parked_carts_flickering_encoder_is_rest_and_a_creep_is_not() -> None:
    """Live 2026-09-19: parked on its charger the cart's right encoder flipped by one tick on
    every state line, the measured twist was never exactly zero, no rest block ever came and the
    gyro stayed silent. One tick is the encoder's own quantum: within it nothing has been seen to
    move; a creep of one tick a line in one direction leaves the band on its second line."""
    from pepin.gyro import TickDither

    tick = math.pi * 0.125 / 4096  # config/base.json, 9.587e-5 m — the literal seen on the wire
    assert tick == pytest.approx(9.587379924285257e-05)
    dither = TickDither(tick)
    assert all(dither.still(0.0, step) for step in (-tick, tick, -tick, tick, 0.0, -tick))
    creep = TickDither(tick)
    assert creep.still(tick, 0.0)
    assert not creep.still(tick, 0.0), "two ticks one way is motion"
    assert creep.still(0.0, 0.0), "and the band is re-anchored where the wheels are now"


def test_the_cpp_bridge_carries_the_same_dither_rule() -> None:
    from pathlib import Path

    REPO = Path(__file__).resolve().parents[2]  # noqa: N806
    header = (REPO / "ros/pepin_base_cpp/include/pepin_base_cpp/gyro_bias.hpp").read_text()
    bridge = (REPO / "ros/pepin_base_cpp/src/base_bridge.cpp").read_text()
    assert "class TickDither" in header and "kTickSlack = 1e-6" in header
    assert "tick_dither_.still(state.d_left_m, state.d_right_m)" in bridge
    assert "wheels.linear == 0.0" not in bridge, "exact zero never happens on a parked cart"
