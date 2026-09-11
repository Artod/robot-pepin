"""The source roster: the ``sources`` flag, and who is alive."""

import pytest

from pepin.sources import CONTACT, DEPTH, LIDAR, ScanSource, SourceHealth, SourceRegistry


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
