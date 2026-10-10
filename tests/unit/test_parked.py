"""The parked gate on its own (pepin.parked): two witnesses still for ``after_s``, out on one
sample, and the pair rate it allows. The clock is the test's; the node's wiring is tested in
test_depth_stream.py."""

from __future__ import annotations

import math

from pepin.parked import AFTER_S, HEAD_DEG, STALE_S, TURN_DPS, ParkedGate

TICK = 0.02  # the odometry's and the neck's 50 Hz


def still_for(gate: ParkedGate, start: float, seconds: float, pan: float = 0.0) -> float:
    """Both witnesses still from ``start`` for ``seconds`` at 50 Hz; returns the clock after."""
    steps = round(seconds / TICK)
    for k in range(steps + 1):
        now = start + k * TICK
        gate.observe_base(now, 0.0, 0.0)
        gate.observe_head(now, pan, 0.3)
    return start + steps * TICK


def test_both_witnesses_still_for_after_s_is_parked_and_not_a_tick_before() -> None:
    gate = ParkedGate()
    assert not gate.parked(0.0), "never heard is not still"
    now = still_for(gate, 0.0, AFTER_S - TICK)
    assert not gate.parked(now)
    now = still_for(gate, now + TICK, 0.0)
    assert gate.parked(now)
    assert gate.text(now) == "parked (still 2 s)"


def test_one_sample_over_either_threshold_ends_parked_at_once() -> None:
    """No hysteresis on the way out: a speed, a turn or a reading nobody can judge."""
    for speed, turn in ((0.03, 0.0), (0.0, TURN_DPS + 0.1), (math.nan, 0.0)):
        gate = ParkedGate()
        now = still_for(gate, 0.0, 3.0)
        assert gate.parked(now)
        gate.observe_base(now + TICK, speed, turn)
        assert not gate.parked(now + TICK), (speed, turn)
        # ...and in again only after after_s of stillness from that sample on
        later = still_for(gate, now + 2 * TICK, AFTER_S - 2 * TICK)
        assert not gate.parked(later)
        assert gate.parked(still_for(gate, later + TICK, TICK))


def test_the_head_may_wander_within_half_a_degree_of_where_it_stood_and_no_further() -> None:
    """A drift is judged against where the stillness began, so a slow pan cannot creep through
    in steps each under the threshold."""
    gate = ParkedGate()
    now = still_for(gate, 0.0, 3.0)
    half = math.radians(HEAD_DEG)
    gate.observe_head(now + TICK, 0.8 * half, 0.3)
    assert gate.parked(now + TICK), "encoder jitter and a settling neck stay parked"
    gate.observe_head(now + 2 * TICK, 1.2 * half, 0.3)  # 0.4 of the threshold from the last
    assert not gate.parked(now + 2 * TICK), "but the drift from the start crossed it"
    gate = ParkedGate()
    now = still_for(gate, 0.0, 3.0)
    gate.observe_head(now + TICK, 0.0, 0.3 + 1.1 * half)
    assert not gate.parked(now + TICK), "the tilt counts as the pan does"


def test_a_witness_that_goes_silent_or_unreadable_is_not_still() -> None:
    gate = ParkedGate()
    now = still_for(gate, 0.0, 3.0)
    for t in (now + TICK, now + STALE_S):  # the head keeps reporting, the base does not
        gate.observe_head(t, 0.0, 0.3)
    assert gate.parked(now + STALE_S)
    later = now + STALE_S + 2 * TICK
    gate.observe_head(later, 0.0, 0.3)
    assert not gate.parked(later), "odometry silent past STALE_S: moving"
    assert "base unheard 0.5 s" in gate.text(later)
    gate.observe_base(later, 0.0, 0.0)
    assert not gate.parked(later), "heard again: its stillness starts over"
    now = still_for(gate, later, AFTER_S)
    assert gate.parked(now)
    gate.lose_head()
    assert not gate.parked(now)
    assert gate.text(now) == "moving (base still 2.0 s, head pose unknown)"


def test_parked_lets_one_pair_a_period_through_and_moving_lets_every_one() -> None:
    """At 8 pictures a second: every one until parked, then one a second, counted from the last
    that went through (1.875, the last before parking); the first after a move goes at once."""
    gate = ParkedGate(pairs_hz=1.0)
    frames = [k / 8 for k in range(8 * 6)]  # 8 a second, binary-exact stamps, six seconds
    admitted = []
    for t in frames:
        gate.observe_base(t, 0.0, 0.0)
        gate.observe_head(t, 0.0, 0.3)
        if gate.admit(t):
            admitted.append(t)
    moving = [t for t in admitted if t < AFTER_S]
    assert moving == [t for t in frames if t < AFTER_S], "every picture until parked"
    assert [t for t in admitted if t >= AFTER_S] == [2.875, 3.875, 4.875, 5.875]
    gate.observe_base(6.0, 0.1, 0.0)
    assert gate.admit(6.0) and gate.admit(6.125), "a move: full rate from the next picture"


def test_pairs_hz_zero_admits_every_pair_however_long_parked() -> None:
    gate = ParkedGate(pairs_hz=0.0)
    now = still_for(gate, 0.0, 10.0)
    assert gate.parked(now), "the state is still told"
    assert all(gate.admit(now + k * 0.1) for k in range(20))
