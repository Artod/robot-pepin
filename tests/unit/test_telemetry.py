from pepin.telemetry import LatencyTracker


def test_summary_reports_window_statistics() -> None:
    t = LatencyTracker("x", window=4)
    for s in (0.010, 0.020, 0.030, 0.040, 0.100):  # first sample falls out of the window
        t.add(s)
    summary = t.summary()
    assert summary.count == 5
    assert summary.max_ms == 100.0
    assert summary.median_ms == 35.0
    assert "n=5" in str(summary)


def test_measure_times_the_block() -> None:
    t = LatencyTracker("y")
    with t.measure():
        pass
    assert t.count == 1 and t.summary().max_ms >= 0.0


def test_empty_tracker_has_zero_summary() -> None:
    assert LatencyTracker("z").summary().count == 0
