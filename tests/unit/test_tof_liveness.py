"""A sensor that has left the bus is not asked forty-eight times a second."""

from pepin.tof_server import DEAD_AFTER, Liveness


def test_a_sensor_that_answers_is_always_asked() -> None:
    live = Liveness()
    for t in range(50):
        assert live.worth_asking("front", float(t))
        assert not live.observe("front", 0, float(t))


def test_ten_silences_in_a_row_declare_it_dead_once_and_park_it() -> None:
    """2026-09-09: two dead sensors produced 25 000 journal lines a minute and a weaving robot."""
    live = Liveness(retry_s=5.0)
    declared = [live.observe("left", 255, float(t)) for t in range(DEAD_AFTER + 5)]
    assert declared.count(True) == 1 and declared[DEAD_AFTER - 1]
    assert live.dead("left")
    assert not live.worth_asking("left", DEAD_AFTER + 5.0 + 1.0), "parked"
    assert live.worth_asking("left", DEAD_AFTER + 5.0 + 5.5), "asked again after the retry period"


def test_one_answer_brings_a_dead_sensor_back() -> None:
    live = Liveness()
    for t in range(DEAD_AFTER):
        live.observe("right", 255, float(t))
    assert live.dead("right")
    live.observe("right", 2, 30.0)  # any real status, even "signal too weak"
    assert not live.dead("right") and live.worth_asking("right", 30.1)


def test_a_missing_status_counts_as_silence() -> None:
    live = Liveness(dead_after=3)
    assert [live.observe("front", None, float(t)) for t in range(3)] == [False, False, True]
