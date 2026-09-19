"""The snapshot packer: whose period is measured how, who drives, and who is left out.

Every expectation here is a literal — a stamp, a name, a count — never the module's own
expression evaluated twice. Two bugs reached the robot through tests that agreed with the code
instead of with reality (2026-09-18), and a packer whose pairing rule is checked against its own
arithmetic would be one of them.
"""

from __future__ import annotations

import math

import pytest

from pepin.snapshot import (
    LIVE_PERIODS,
    MEASURED_STAMPS,
    NEAREST_PERIODS,
    PAIR_PERIODS,
    Cadence,
    SnapshotPacker,
    StampRing,
)

CAMERA, LIDAR = "camera", "lidar"


def _packer(pair_periods: float = PAIR_PERIODS) -> SnapshotPacker[str]:
    """A two-source packer whose payloads are strings, so a test reads what came out."""
    return SnapshotPacker[str]((CAMERA, LIDAR), pair_periods=pair_periods)


# ---- the measured period -------------------------------------------------------------------
def test_the_period_is_measured_from_the_stamps_and_takes_two_of_them() -> None:
    """One message is no interval: until a source has delivered two it has no period, so no
    patience and no liveness, and it can neither drive nor join."""
    assert MEASURED_STAMPS == 2
    cadence = Cadence()
    assert cadence.period_s is None and cadence.patience_s is None
    assert cadence.rate_hz == 0.0 and cadence.verdict(0.0) == "absent"
    cadence.observe(1000.0)
    assert cadence.count == 1
    assert cadence.period_s is None, "one stamp is not an interval"
    assert cadence.verdict(1000.0) == "unmeasured"
    assert cadence.alive(1000.0) is False
    cadence.observe(1000.1)
    assert cadence.period_s == pytest.approx(0.1)
    assert cadence.rate_hz == pytest.approx(10.0)
    assert cadence.patience_s == pytest.approx(0.15), "1.5 periods, PAIR_PERIODS"
    assert cadence.liveness_s == pytest.approx(0.5), "5 periods, LIVE_PERIODS"
    assert cadence.verdict(1000.1) == "fresh"


def test_a_slowing_source_is_believed_within_a_couple_of_seconds() -> None:
    """The period is a running mean with pepin.sources' own two-second constant, not the mean of
    the whole run: a camera that falls from 10 Hz to 3.3 has a period near 0.3 s after a few
    seconds of it, so its patience opens with it instead of calling it silent."""
    cadence = Cadence()
    stamp = 1000.0
    for _ in range(40):  # four seconds at 10 Hz
        cadence.observe(stamp)
        stamp += 0.1
    assert cadence.period_s == pytest.approx(0.1, abs=1e-6)
    for _ in range(20):  # six seconds at 3.3 Hz
        cadence.observe(stamp)
        stamp += 0.3
    assert cadence.period_s == pytest.approx(0.288, abs=0.002), "96 % of the way there in 6 s"
    assert cadence.patience_s == pytest.approx(0.433, abs=0.003)
    assert cadence.rate_hz == pytest.approx(3.47, abs=0.02)


def test_a_source_five_of_its_own_periods_behind_has_stopped_delivering() -> None:
    """Liveness is its own question and a much looser one than pairing: the depth's worst
    measured arrival lag was 4.2 of its periods, so four periods behind is still alive and six
    is not."""
    assert LIVE_PERIODS == 5.0
    cadence = Cadence()
    cadence.observe(1000.0)
    cadence.observe(1000.1)  # period 0.1 -> liveness 0.5, patience 0.15
    assert cadence.alive(1000.5) is True and cadence.verdict(1000.5) == "fresh"
    assert cadence.alive(1000.7) is False and cadence.verdict(1000.7) == "silent"
    assert cadence.text(1000.7) == "silent 0.6 s"
    assert cadence.text(1000.1) == "fresh 10.0 Hz (period 100 ms, pairs within 150 ms)"


# ---- the ring ------------------------------------------------------------------------------
def test_the_ring_keeps_its_window_off_the_largest_stamp_ever_offered() -> None:
    """A message that arrives out of order must not empty the ring: the window hangs off the
    largest stamp seen, so an old one is simply dropped and the rest stay."""
    ring = StampRing[str](window_s=1.0)
    for k in range(5):
        ring.offer(1000.0 + 0.25 * k, f"s{k}")
    assert len(ring) == 5
    ring.offer(1001.5, "new")  # everything more than a second behind 1001.5 goes
    assert len(ring) == 4, "1000.5, 1000.75, 1001.0 and 1001.5"
    assert ring.newest() == (1001.5, "new")
    ring.offer(999.0, "ancient")
    assert len(ring) == 4 and ring.newest() == (1001.5, "new")
    assert [stamp for stamp, _ in ring.items()] == [1000.5, 1000.75, 1001.0, 1001.5]


def test_the_nearest_message_is_the_nearest_and_the_older_one_on_a_tie() -> None:
    ring = StampRing[str](window_s=1.0)
    ring.offer(1000.0, "a")
    ring.offer(1000.2, "b")
    assert ring.nearest(1000.19) == (1000.2, "b")
    assert ring.nearest(1000.1) == (1000.0, "a"), "a tie goes to the older one, so it is stable"
    assert StampRing[str]().nearest(1.0) is None


# ---- who drives, who joins -----------------------------------------------------------------
def test_the_driver_is_the_latest_moment_every_alive_source_has_spoken_for() -> None:
    """Not the newest stamp: the camera's frame reaches this machine later than the scan of the
    same moment, so driving on the newest stamp would hand the camera the whole difference. The
    OLDEST of the alive sources' newest stamps is the moment both have already covered."""
    packer = _packer()
    for stamp in (1000.00, 1000.10):
        packer.offer(LIDAR, stamp, f"scan@{stamp:.2f}")
    for stamp in (999.96, 1000.06):  # the camera lags the lidar, as it does on the wire
        packer.offer(CAMERA, stamp, f"frame@{stamp:.2f}")
    snapshot = packer.plan()
    assert snapshot is not None
    assert snapshot.driver == CAMERA
    assert snapshot.stamp == 1000.06
    assert snapshot.kind == "full" and snapshot.silent == ()
    assert snapshot.members[CAMERA] == (1000.06, "frame@1000.06")
    assert snapshot.members[LIDAR] == (1000.10, "scan@1000.10")
    assert snapshot.offset_s(LIDAR) == pytest.approx(0.04)
    assert snapshot.offset_s(CAMERA) == 0.0


def test_the_member_is_the_message_nearest_the_moment_not_that_source_s_newest() -> None:
    """The bound the whole pairing rests on: with revolutions on both sides of the frame's
    moment the one that joins is the nearest, which is at most half a period away."""
    assert NEAREST_PERIODS == 0.5
    packer = _packer()
    for stamp in (1000.00, 1000.10, 1000.20, 1000.30):
        packer.offer(LIDAR, stamp, f"scan@{stamp:.2f}")
    for stamp in (999.88, 1000.18):
        packer.offer(CAMERA, stamp, f"frame@{stamp:.2f}")
    snapshot = packer.plan()
    assert snapshot is not None
    assert snapshot.stamp == 1000.18 and snapshot.driver == CAMERA
    assert snapshot.members[LIDAR] == (1000.20, "scan@1000.20"), "nearest, not the 1000.30 newest"
    assert snapshot.offset_s(LIDAR) == pytest.approx(0.02)


def test_a_lagging_source_moves_the_moment_back_rather_than_being_dropped() -> None:
    """The rule is symmetric, and this is what it buys: when the lidar is the one behind, the
    snapshot's moment goes back to the lidar's newest instead of the lidar being left out. The
    node is then stamped a little into the past, which TF can answer for; dropping a sensor that
    is delivering is the thing a graph cannot recover from."""
    packer = _packer()
    packer.offer(LIDAR, 1000.00, "scan@1000.00")
    packer.offer(LIDAR, 1000.10, "scan@1000.10")  # period 0.1 -> patience 0.15, liveness 0.5
    packer.offer(CAMERA, 1000.20, "frame@1000.20")
    packer.offer(CAMERA, 1000.35, "frame@1000.35")  # period 0.15 -> patience 0.225
    snapshot = packer.plan()
    assert snapshot is not None
    assert snapshot.driver == LIDAR and snapshot.stamp == 1000.10
    assert snapshot.kind == "full"
    assert snapshot.members[CAMERA] == (1000.20, "frame@1000.20")
    assert snapshot.offset_s(CAMERA) == pytest.approx(0.10)


def test_a_member_past_its_own_patience_is_silent_and_the_snapshot_says_so() -> None:
    """1.5 of the source's OWN measured period. The lidar at 10 Hz pairs within 0.15 s and is
    alive for 0.5 s, so a lidar muted 0.7 s ago is out of the choice AND out of the snapshot,
    which is then camera-only — the live check for a muted sensor."""
    assert PAIR_PERIODS == 1.5
    packer = _packer()
    packer.offer(LIDAR, 1000.00, "scan@1000.00")
    packer.offer(LIDAR, 1000.10, "scan@1000.10")  # period 0.1 -> patience 0.15, liveness 0.5
    packer.offer(CAMERA, 1000.70, "frame@1000.70")
    packer.offer(CAMERA, 1000.80, "frame@1000.80")
    snapshot = packer.plan()
    assert snapshot is not None
    assert snapshot.driver == CAMERA and snapshot.stamp == 1000.80
    assert snapshot.kind == "camera-only"
    assert snapshot.silent == (LIDAR,)
    assert tuple(snapshot.members) == (CAMERA,)
    assert snapshot.offset_s(LIDAR) == math.inf


def test_a_dead_camera_hands_the_snapshots_to_the_lidar_with_nobody_deciding() -> None:
    """The whole point of the design: no timeout anyone chose, no restart, no mode. The camera's
    stamps stop advancing, the lidar's do not, and within five camera periods the camera is out of
    the choice and every snapshot is the lidar's."""
    packer = _packer()
    packer.offer(CAMERA, 1000.00, "frame@1000.00")
    packer.offer(CAMERA, 1000.10, "frame@1000.10")  # liveness 0.5 s from here
    for stamp in (1000.05, 1000.15):
        packer.offer(LIDAR, stamp, f"scan@{stamp:.2f}")
    first = packer.plan()
    assert first is not None and first.driver == CAMERA and first.kind == "full"
    for stamp in (1000.25, 1000.35, 1000.45, 1000.55, 1000.65, 1000.75):
        packer.offer(LIDAR, stamp, f"scan@{stamp:.2f}")
    later = packer.plan()
    assert later is not None
    assert later.driver == LIDAR, "the camera is more than 5 of its own periods behind"
    assert later.stamp == 1000.75 and later.kind == "lidar-only"
    assert later.silent == (CAMERA,)
    assert "camera silent 0.6 s" in packer.report()
    assert "lidar fresh 10.0 Hz (period 100 ms, pairs within 150 ms)" in packer.report()


def test_a_source_that_has_spoken_once_cannot_be_paired_yet() -> None:
    """No period, no patience: a camera one frame old is not in the snapshot, and the report line
    says ``unmeasured`` rather than pretending it is fresh or silent."""
    packer = _packer()
    packer.offer(CAMERA, 1000.00, "frame@1000.00")
    for stamp in (1000.02, 1000.12):
        packer.offer(LIDAR, stamp, f"scan@{stamp:.2f}")
    snapshot = packer.plan()
    assert snapshot is not None
    assert snapshot.driver == LIDAR and snapshot.kind == "lidar-only"
    assert snapshot.silent == (CAMERA,)
    assert "camera unmeasured (1 message)" in packer.report()


# ---- when nothing comes out ----------------------------------------------------------------
def test_nothing_at_all_before_a_period_and_nothing_twice_after_one() -> None:
    packer = _packer()
    assert packer.plan() is None, "no message at all"
    assert packer.report() == "camera no stamp yet, lidar no stamp yet"
    packer.offer(LIDAR, 1000.0, "scan")
    assert packer.plan() is None, "one message is no period"
    packer.offer(LIDAR, 1000.1, "scan2")
    assert packer.plan() is not None
    assert packer.plan() is None, "the same moment is never packed twice"
    assert packer.last_stamp == 1000.1


def test_the_minimum_gap_is_sensor_time_so_a_quiet_stack_packs_nothing() -> None:
    """One snapshot per ``min_gap_s`` of the stamps' own clock — never of this machine's — so a
    caller may ask on every arriving message, and a stack whose sensors have all gone quiet
    produces no snapshots instead of republishing the last one for ever."""
    packer = _packer()
    for stamp in (1000.0, 1000.1):
        packer.offer(LIDAR, stamp, f"scan@{stamp:.1f}")
    assert packer.plan(min_gap_s=1.0) is not None
    for stamp in (1000.2, 1000.5, 1000.9):
        packer.offer(LIDAR, stamp, f"scan@{stamp:.1f}")
        assert packer.plan(min_gap_s=1.0) is None, "under a second of sensor time"
    packer.offer(LIDAR, 1001.1, "scan@1001.1")
    second = packer.plan(min_gap_s=1.0)
    assert second is not None and second.stamp == 1001.1
    for _ in range(5):  # the stack is quiet: no new stamp, so no snapshot, however often asked
        assert packer.plan(min_gap_s=1.0) is None


def test_a_source_the_flag_switched_off_is_out_of_every_snapshot_but_still_measured() -> None:
    """``sources`` is the live A/B: off, a sensor enters no snapshot and drives none, while its
    cadence is still measured so the report line can say it is delivering."""
    packer = _packer()
    for stamp in (1000.00, 1000.10, 1000.20):
        packer.offer(LIDAR, stamp, f"scan@{stamp:.2f}")
        packer.offer(CAMERA, stamp - 0.01, f"frame@{stamp - 0.01:.2f}")
    packer.enable((LIDAR,))
    assert packer.enabled == (LIDAR,)
    snapshot = packer.plan()
    assert snapshot is not None
    assert snapshot.driver == LIDAR and snapshot.stamp == 1000.20
    assert tuple(snapshot.members) == (LIDAR,) and snapshot.silent == ()
    assert snapshot.kind == "lidar-only"
    assert "camera off" in packer.report()
    with pytest.raises(ValueError, match="unknown sources"):
        packer.enable(("tof",))
    assert packer.enabled == (LIDAR,), "a refused change leaves the roster alone"


def test_the_patience_moves_with_the_flag_and_reaches_every_source() -> None:
    packer = _packer()
    for stamp in (1000.00, 1000.10):
        packer.offer(LIDAR, stamp, f"scan@{stamp:.2f}")
    assert packer.cadence(LIDAR).patience_s == pytest.approx(0.15)
    packer.set_pair_periods(0.5)
    assert packer.cadence(LIDAR).patience_s == pytest.approx(0.05)
    assert packer.cadence(CAMERA).pair_periods == 0.5
