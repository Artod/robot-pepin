"""RTAB-Map's statistics as the report line reads them."""

from pepin.graphtrust import HIGHEST_HYPOTHESIS, stat


def test_a_statistic_is_found_through_the_unit_segment_its_key_carries() -> None:
    """RTAB-Map's keys end in a unit ("Loop/Highest_hypothesis_value/") and a lookup that missed
    one would read a silent zero, so a report line would say the graph is nowhere near recognising
    anything while it is one frame away."""
    stats = {"Loop/Highest_hypothesis_value/": 0.04, "Memory/Distance_travelled/m": 12.5}
    assert stat(stats, HIGHEST_HYPOTHESIS) == 0.04
    assert stat(stats, "Memory/Distance_travelled") == 12.5, "the unit segment is tolerated"
    assert stat(stats, "Memory/Nothing_of_the_sort") is None, "a missing statistic is not a 0"
