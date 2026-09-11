"""The source roster: the ``sources`` flag, who is alive, and whose scan drives the update."""

import numpy as np
import pytest

from pepin.odometry import Pose2D
from pepin.sources import (
    CONTACT,
    DEPTH,
    LIDAR,
    ScanSource,
    SourceFeed,
    SourceHealth,
    SourceRegistry,
)
from pepin.timeline import OdomHistory, ScanGate, TimedScan


def test_the_lidar_alone_is_on_by_default_and_the_flag_picks_the_rest() -> None:
    registry = SourceRegistry()
    assert registry.enabled == (LIDAR,) and registry.names == (LIDAR, DEPTH, CONTACT)
    registry.enable([CONTACT, LIDAR])
    assert registry.enabled == (LIDAR, CONTACT)  # roster order, whatever the flag's
    assert registry.is_enabled(CONTACT) and not registry.is_enabled(DEPTH)
    registry.enable([])
    assert registry.enabled == ()
    with pytest.raises(ValueError, match="unknown sources"):
        registry.enable(["lidar", "sonar"])
    assert registry.source(DEPTH).partial and not registry.source(LIDAR).partial


def test_health_is_fresh_then_stale_and_absent_before_the_first_scan() -> None:
    health = SourceHealth(stale_after_s=0.5)
    assert health.verdict(10.0) == "absent" and health.text(10.0) == "absent"
    for k in range(20):
        health.observe(10.0 + 0.1 * k)
    assert health.verdict(12.0) == "fresh" and health.rate_hz == pytest.approx(10.0, abs=0.2)
    assert health.text(12.0).startswith("fresh 10.")
    assert health.verdict(12.6) == "stale" and health.text(12.6) == "stale 0.7 s"
    health.observe(11.0)  # a late scan does not move the clock backwards
    assert health.last_stamp == 11.9


def test_alive_is_enabled_and_fresh_and_the_report_says_who_is_off() -> None:
    registry = SourceRegistry(enabled=[LIDAR, DEPTH])
    registry.observe(LIDAR, 5.0)
    registry.observe(DEPTH, 3.0)
    registry.observe(CONTACT, 5.0)  # arriving, but the flag has it off
    assert [s.name for s in registry.alive(5.2)] == [LIDAR]
    registry.observe(DEPTH, 5.1)
    assert [s.name for s in registry.alive(5.2)] == [LIDAR, DEPTH]
    report = registry.report(5.2)
    assert report.startswith("lidar fresh") and "depth fresh" in report and "contact off" in report
    assert [s.name for s in registry.alive(9.0)] == []
    assert "lidar stale 4.0 s" in registry.report(9.0)


def test_a_custom_roster_keeps_its_own_order_and_floors() -> None:
    sonar = ScanSource("sonar", "sonar_link", 30.0, trust=0.2, stale_after_s=2.0, min_points=4)
    registry = SourceRegistry([sonar], enabled=["sonar"])
    assert registry.enabled == ("sonar",) and registry.source("sonar").min_points == 4
    assert registry.health("sonar").verdict(0.0) == "absent"


# ---- the feed: whose scan drives the update, and the riders carried to its moment ------------
def scan(stamp: float, points: list[list[float]] | None = None, scan_id: int = 0) -> TimedScan:
    """A scan taken in one instant at ``stamp`` (a camera frame, or a lidar turn standing)."""
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]] if points is None else points)
    return TimedScan(stamp, pts, np.full(len(pts), stamp), np.full(len(pts), 1.0), scan_id)


def driving(until_s: float, speed_m_s: float = 0.5) -> OdomHistory:
    """Odometry of a cart driving straight ahead at ``speed_m_s``, sampled at 20 Hz."""
    history = OdomHistory()
    for k in range(int(until_s * 20) + 1):
        history.add(0.05 * k, Pose2D(speed_m_s * 0.05 * k, 0.0, 0.0))
    return history


def test_the_lidar_anchors_while_fresh_and_a_rider_is_carried_to_its_moment() -> None:
    """The camera's frame 50 ms before the lidar's revolution saw the wall 1 m ahead; the cart
    drove 2.5 cm on since, so at the lidar's moment that wall is 97.5 cm ahead: the rider's
    returns are moved by the odometry between the two stamps, and each frame rides once."""
    feed = SourceFeed(SourceRegistry(enabled=[LIDAR, DEPTH]))
    history = driving(2.0)
    feed.offer(DEPTH, scan(0.95, [[1.0, 0.0], [1.0, 0.5]]))
    feed.offer(LIDAR, scan(1.0, scan_id=7))
    assert feed.anchor(1.02) == LIDAR
    taken = feed.take(history, 1.02)
    assert taken is not None and taken[0] == LIDAR and taken[1].scan_id == 7
    assert feed.take(history, 1.02) is None, "released once"
    riders = feed.gather(LIDAR, 1.0, history)
    assert [r.source for r in riders] == [DEPTH] and riders[0].stamp == 1.0
    np.testing.assert_allclose(riders[0].points, [[0.975, 0.0], [0.975, 0.5]], atol=1e-9)
    assert feed.gather(LIDAR, 1.0, history) == [], "a frame rides once"
    stats = feed.report()
    assert stats.gates[LIDAR].released == 1 and stats.attached[DEPTH] == 1 and stats.released == 1
    assert (
        stats.summary().startswith("lidar: scans 1, released 1") and "attached 1" in stats.summary()
    )
    alone = SourceFeed(SourceRegistry())
    alone.offer(LIDAR, scan(1.0))
    assert alone.report().summary().startswith("scans 1, released 0"), "one source: as ever"


def test_a_dead_lidar_hands_the_updates_to_the_camera_and_takes_them_back() -> None:
    feed = SourceFeed(SourceRegistry(enabled=[LIDAR, DEPTH]))
    history = driving(4.0)
    for k in range(11):  # the lidar at 10 Hz until t = 1.0, then silence
        feed.offer(LIDAR, scan(0.1 * k))
    feed.offer(DEPTH, scan(1.05))
    assert feed.take(history, 1.07) is not None and feed.anchor(1.3) == LIDAR
    assert feed.take(history, 1.3) is None, "the lidar is still fresh: the frame waits for it"
    feed.offer(DEPTH, scan(1.4))
    assert feed.anchor(1.6) == DEPTH, "0.5 s without a revolution: the camera drives"
    taken = feed.take(history, 1.6)
    assert taken is not None and taken[0] == DEPTH and taken[1].stamp == 1.4
    assert feed.gather(DEPTH, 1.4, history) == []
    assert feed.picture(1.6) is not None and feed.picture(1.6).stamp == 1.4
    assert feed.status(1.6).startswith("anchor depth; lidar stale 0.6 s, depth fresh")
    feed.offer(DEPTH, scan(1.95))
    feed.offer(LIDAR, scan(2.0))  # the lidar is back
    assert feed.anchor(2.02) == LIDAR and feed.picture(2.02).stamp == 2.0
    taken = feed.take(history, 2.02)
    assert taken is not None and taken[0] == LIDAR
    assert [r.source for r in feed.gather(LIDAR, 2.0, history)] == [DEPTH], "and the frame rides"
    assert feed.status(2.02).startswith("anchor lidar; lidar fresh")


def test_nothing_fresh_holds_a_stale_rider_is_dropped_and_an_uncovered_one_waits() -> None:
    feed = SourceFeed(SourceRegistry(enabled=[LIDAR, DEPTH]))
    history = driving(5.1)
    assert feed.anchor(0.0) is None and feed.take(history, 0.0) is None
    assert feed.picture(0.0) is None
    assert feed.status(0.0) == (
        "holding map->odom: no fresh source; lidar absent, depth absent, contact off"
    )
    feed.offer(DEPTH, scan(0.5))
    assert feed.anchor(5.0) == DEPTH, "nothing fresh but a frame waiting: its gate decides"
    taken = feed.take(history, 5.0)
    assert taken is not None and taken[0] == DEPTH, "the odometry covers it: released, late"
    assert feed.anchor(5.0) is None and feed.picture(5.0) is not None, "the last picture heard"
    assert feed.status(5.0).startswith(
        "no fresh source, last release depth 4.5 s ago; lidar absent, depth stale 4.5 s"
    )
    feed.offer(LIDAR, scan(3.0))  # the odometry never reaches it (the history is empty here)
    assert feed.anchor(3.6) == LIDAR, "called stale at 0.6 s, but its revolution waits"
    assert feed.take(OdomHistory(), 3.4) is None and feed.take(OdomHistory(), 3.6) is None
    stats = feed.report()
    assert stats.gates[LIDAR].expired == 1, "uncovered past the patience: the odometry ran late"
    assert stats.gates[DEPTH].released == 1 and stats.expired == 1
    feed.offer(DEPTH, scan(3.5))  # older than the camera's stale_after_s beside the anchor
    feed.offer(LIDAR, scan(5.0))
    assert feed.take(history, 5.02) is not None
    assert feed.gather(LIDAR, 5.0, history) == [] and feed.report().dropped[DEPTH] == 1
    feed.offer(DEPTH, scan(5.2))  # newer than the odometry: not carried yet, not dropped
    feed.offer(LIDAR, scan(5.1))
    assert feed.take(history, 5.12) is not None and feed.gather(LIDAR, 5.1, history) == []
    history.add(5.3, Pose2D(2.65, 0.0, 0.0))
    feed.offer(LIDAR, scan(5.2))
    assert feed.take(history, 5.22) is not None
    assert [r.source for r in feed.gather(LIDAR, 5.2, history)] == [DEPTH]
    feed.offer(CONTACT, scan(5.2))  # the flag has it off: heard, never anchored, never a rider
    assert feed.anchor(5.22) == LIDAR and feed.gather(LIDAR, 5.2, history) == []
    assert "contact off" in feed.status(5.22)


def test_the_sources_flag_moves_the_anchor_without_a_restart() -> None:
    feed = SourceFeed(SourceRegistry(enabled=[DEPTH]))
    feed.offer(LIDAR, scan(1.0))
    feed.offer(DEPTH, scan(1.0))
    assert feed.anchor(1.02) == DEPTH, "the lidar arrives but the flag has it off"
    feed.registry.enable([LIDAR, DEPTH])
    assert feed.anchor(1.02) == LIDAR
    feed.registry.enable([DEPTH, CONTACT])
    feed.offer(CONTACT, scan(1.01))
    assert feed.anchor(1.02) == DEPTH, "two fans alike: the roster's first while it is fresh"
    assert feed.anchor(2.005) == DEPTH, "a moment too old, but its frame still waits at the gate"
    assert feed.take(OdomHistory(), 2.005) is None, "uncovered past the patience: expired"
    assert feed.anchor(2.005) == CONTACT, "the depth fan a moment too old: the other one"
    assert feed.picture(2.005) is not None and feed.picture(2.005).stamp == 1.01


def test_a_revolution_delivered_late_is_matched_late_as_the_gate_alone_always_did() -> None:
    """The executor lag of 2026-09-06 (a scan half a second old by the time its callback
    ran): at take-time the lidar is called stale, but the odometry covers its revolution, and
    coverage — never freshness — decides a release. With the lidar alone the feed is the bare
    gate: the same offers and takes through both give the same scans and the same counters,
    and ``expired`` still means one thing, an uncovered scan past the patience."""
    history = driving(4.0)
    feed, gate = SourceFeed(SourceRegistry()), ScanGate(max_wait_s=0.5)
    for k in range(20):
        stamp = 1.0 + 0.1 * k
        revolution = scan(stamp, scan_id=k)
        feed.offer(LIDAR, revolution)
        gate.offer(revolution)
        now = stamp + 0.6  # the callback runs 0.6 s after the stamp
        assert feed.anchor(now) == LIDAR
        assert feed.status(now).startswith("anchor lidar; lidar stale 0.6 s")
        taken, released = feed.take(history, now), gate.take(history, now)
        assert taken is not None and taken[1] is released, f"scan {k}: matched late, not dropped"
    assert feed.status(3.5).startswith("no fresh source, last release lidar 0.6 s ago; lidar stale")
    assert feed.report().summary() == gate.report().summary()
    late = scan(4.0, scan_id=99)
    feed.offer(LIDAR, late)
    gate.offer(late)
    assert feed.take(OdomHistory(), 4.6) is None and gate.take(OdomHistory(), 4.6) is None
    feed_stats, gate_stats = feed.report(), gate.report()
    assert feed_stats.expired == gate_stats.expired == 1
    assert feed_stats.summary() == gate_stats.summary()
