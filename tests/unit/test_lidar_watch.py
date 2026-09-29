"""The lidar watch: when a silent lidar's driver is restarted, and when it is left alone."""

from __future__ import annotations

from itertools import pairwise

from pepin.lidar_watch import GRACE_S, PATIENT_AFTER, PATIENT_S, SILENT_S, LidarWatch


def test_a_lidar_that_scans_is_left_alone() -> None:
    watch = LidarWatch(started_at=0.0)
    for t in range(1, 30):
        watch.scan(float(t))
        assert not watch.due(t + 0.5, port_present=True)
    assert watch.status(29.5, True) == "ok" and watch.kicks == 0


def test_no_port_means_one_kick_to_idle_however_long_the_silence() -> None:
    """The old rule was no kick at all without a port; a driver that lost its port spins at 100 %
    of a core and one started without it idles at 3 %, so one kick, then never again."""
    watch = LidarWatch(started_at=0.0)
    kicks = [t for t in range(0, 600, 5) if watch.due(float(t), port_present=False)]
    assert kicks == [5] and watch.kicks == 1 and watch.last_kick_at is None
    assert (
        watch.status(600.0, False) == "silent 600 s, no port (unplugged), driver restarted to idle"
    )


def test_a_port_lost_mid_run_is_kicked_exactly_once_however_long_the_absence() -> None:
    watch = LidarWatch(started_at=0.0)
    watch.scan(50.0)
    assert not watch.due(50.0 + SILENT_S - 0.1, port_present=False), "not silent long enough"
    assert watch.status(50.0 + SILENT_S, False) == f"silent {SILENT_S:.0f} s, no port (unplugged)"
    kicks = [t / 2 for t in range(110, 2000) if watch.due(t / 2, port_present=False)]
    assert kicks == [55.0] and watch.kicks == 1 and watch.kicks_in_a_row == 0


def test_the_port_back_after_an_absence_brings_today_s_rule_back() -> None:
    """The idle driver never opens the port that came back: kicked at once, then given its grace;
    and a later absence gets its own one kick."""
    watch = LidarWatch(started_at=0.0)
    watch.scan(50.0)
    assert watch.due(56.0, port_present=False)
    assert not watch.due(58.0, port_present=False)
    assert watch.due(60.0, port_present=True), "the port is back: kick"
    assert "with its port present, restarting its driver (1 in a row)" in watch.status(60.0, True)
    assert not watch.due(60.0 + GRACE_S - 1.0, True), "the respawned driver is given its time"
    assert watch.due(80.0, port_present=False), "a new absence: one kick to idle"
    assert not any(watch.due(float(t), port_present=False) for t in range(81, 300))
    assert watch.kicks == 3


def test_a_stack_started_without_the_lidar_kicks_once_and_never_loops() -> None:
    """Never a scan, never a port: at most one kick, whatever the silence."""
    watch = LidarWatch(started_at=0.0)
    kicks = sum(watch.due(t / 10, port_present=False) for t in range(0, 36000))
    assert kicks <= 1 and watch.kicks == kicks
    assert "driver restarted to idle" in watch.status(3600.0, False)


def test_a_lidar_plugged_in_after_the_start_is_kicked_once_its_port_is_there() -> None:
    watch = LidarWatch(started_at=0.0)
    assert watch.due(100.0, port_present=False), "the one kick to idle of this absence"
    assert not watch.due(100.5, port_present=False)
    assert watch.due(101.0, port_present=True), "the port appeared and nothing has ever scanned"
    assert not watch.due(101.0 + GRACE_S - 1.0, True), "the respawned driver is given its time"
    watch.scan(110.0)
    assert not watch.due(112.0, True) and watch.status(112.0, True) == "ok"


def test_a_lidar_unplugged_and_plugged_back_mid_run_is_kicked_after_the_silence() -> None:
    watch = LidarWatch(started_at=0.0)
    watch.scan(50.0)
    assert not watch.due(50.0 + SILENT_S - 0.1, True)
    assert watch.due(60.0, port_present=False), "unplugged: once, so it respawns idle"
    assert watch.due(61.0, port_present=True)


def test_a_dead_lidar_is_tried_three_times_then_once_a_minute() -> None:
    watch = LidarWatch(started_at=0.0)
    times = [t for t in (x * 0.5 for x in range(0, 600)) if watch.due(t, True)]
    assert len(times) >= PATIENT_AFTER + 2
    gaps = [b - a for a, b in pairwise(times)]
    assert all(abs(g - GRACE_S) < 0.6 for g in gaps[: PATIENT_AFTER - 1])
    assert all(abs(g - PATIENT_S) < 0.6 for g in gaps[PATIENT_AFTER - 1 :])
    assert "one try a minute" in watch.status(299.0, True)


def test_a_scan_ends_the_streak() -> None:
    watch = LidarWatch(started_at=0.0)
    for t in (5.0, 20.0, 35.0):
        assert watch.due(t, True)
    watch.scan(40.0)
    assert watch.kicks_in_a_row == 0 and watch.kicks == 3
    assert not watch.due(40.0 + SILENT_S, True), "still inside the last kick's grace"
    assert watch.due(35.0 + GRACE_S + 0.5, True), "silent again: the quick pace is back"
