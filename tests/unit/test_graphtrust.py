"""What a pose graph's word is worth (pepin.graphtrust).

The failure this module exists to stop, in one line: on 2026-09-14 a carried cart's graph
recognised nothing for 64 s and its word still claimed fit 1.00, so nothing downstream could
tell a recognition from dead reckoning.
"""

import math

import pytest

from pepin.graphtrust import (
    ACCEPTED_HYPOTHESIS_ID,
    DISTANCE_TRAVELLED_M,
    FILE_ANCHOR_TRUST,
    GRAPH_TRUST_M,
    HIGHEST_HYPOTHESIS,
    LOOP_ID,
    PROXIMITY_ICP,
    PROXIMITY_VISUAL,
    GraphTrust,
    stat,
)
from pepin.watch import ADMIT_FIT


def info(
    travelled: float,
    loop: float = 0.0,
    proximity: float = 0.0,
    hypothesis: float = 0.0,
) -> dict[str, float]:
    """One /rtabmap/info message's statistics, under the keys RTAB-Map publishes them with —
    the trailing unit segment and all."""
    return {
        "Memory/Distance_travelled/m": travelled,
        "Loop/Id/": loop,
        "Loop/Accepted_hypothesis_id/": loop,
        "Proximity/Space_detections_added_visually/": proximity,
        "Proximity/Space_detections_added_icp_multi/": 0.0,
        "Loop/Highest_hypothesis_value/": hypothesis,
    }


def test_a_statistic_is_found_through_the_unit_segment_its_key_carries() -> None:
    """RTAB-Map's keys end in a unit ("Loop/Id/", "Memory/Distance_travelled/m") and a lookup
    that missed one would read a silent zero — the trust would then never decay and never tie."""
    stats = info(12.5, loop=41.0, hypothesis=0.04)
    assert stat(stats, DISTANCE_TRAVELLED_M) == 12.5
    assert stat(stats, LOOP_ID) == 41.0
    assert stat(stats, ACCEPTED_HYPOTHESIS_ID) == 41.0
    assert stat(stats, HIGHEST_HYPOTHESIS) == 0.04
    assert stat(stats, PROXIMITY_VISUAL) == 0.0
    assert stat(stats, "Memory/Nothing_of_the_sort") is None, "a missing statistic is not a 0"


def test_the_word_is_worth_one_at_a_tie_and_decays_with_the_metres_driven_since() -> None:
    """A closure ties the present to a node the graph already had: at that instant the word can
    undo accumulated drift and is worth everything. Every metre after it is odometry again."""
    trust = GraphTrust()
    trust.update(info(10.0))  # the first message is only the baseline
    assert trust.trust() == pytest.approx(1.0)
    assert trust.since_m == 0.0

    assert trust.update(info(12.0, loop=41.0)) is True, "a closure is a tie"
    assert trust.ties == 1 and trust.since_m == 0.0
    assert trust.trust() == pytest.approx(1.0)

    trust.update(info(17.0))
    assert trust.since_m == pytest.approx(5.0)
    assert trust.trust() == pytest.approx(math.exp(-1.0))  # one decay length


def test_a_proximity_link_ties_the_graph_exactly_as_a_closure_does() -> None:
    """A proximity detection links the present node to an older one the cart has walked back
    to: the same correction by another name, and the same reset of the clock."""
    trust = GraphTrust()
    trust.update(info(0.0))
    trust.update(info(8.0))
    assert trust.trust() < 0.25, "8 m of dead reckoning is not a recognition"
    assert trust.update(info(8.0, proximity=2.0)) is True
    assert trust.trust() == pytest.approx(1.0)


def test_a_carried_cart_falls_under_every_gate_downstream_within_a_few_metres() -> None:
    """The carry test's own numbers: with the graph recognising nothing, the word must stop
    being admitted as a whole-map candidate (pepin.watch.ADMIT_FIT) and must fall under the
    file-anchor floor, so the goal server's lost ladder sees the pose for what it is."""
    trust = GraphTrust()
    trust.update(info(0.0, hypothesis=0.04))
    trust.update(info(4.0, hypothesis=0.04))
    assert trust.trust() < ADMIT_FIT, "4 m past a tie: no candidate is admitted on this"
    trust.update(info(6.1, hypothesis=0.04))
    assert trust.trust() < FILE_ANCHOR_TRUST, "6 m past a tie: under the wake-up cap too"


def test_a_file_anchor_claims_no_more_than_the_cap_until_the_graph_recognises_something() -> None:
    """The wake-up: an anchor read from a file is a relation measured in ANOTHER session, and a
    board restart moves the odom frame under it. It is a guess until the graph confirms it —
    and the moment it does, the cap is gone."""
    trust = GraphTrust()
    trust.update(info(0.0))
    assert trust.trust(anchor_from_file=False) == pytest.approx(1.0)
    assert trust.trust(anchor_from_file=True) == pytest.approx(FILE_ANCHOR_TRUST)
    assert trust.report(anchor_from_file=True).capped is True

    trust.update(info(1.0, loop=7.0))
    assert trust.trust(anchor_from_file=True) == pytest.approx(1.0), "the graph confirmed it"
    assert trust.report(anchor_from_file=True).capped is False


def test_the_cap_never_raises_a_trust_that_has_already_decayed_below_it() -> None:
    """It is a ceiling, not a value: a file anchor 20 m past anything recognised is worth less
    than the cap, not exactly it."""
    trust = GraphTrust()
    trust.update(info(0.0))
    trust.update(info(20.0))
    decayed = trust.trust(anchor_from_file=False)
    assert decayed < FILE_ANCHOR_TRUST
    assert trust.trust(anchor_from_file=True) == pytest.approx(decayed)
    assert trust.report(anchor_from_file=True).capped is False


def test_a_distance_counter_that_falls_is_a_restarted_graph_and_not_a_negative_decay() -> None:
    """RTAB-Map restarted counts from zero again; read as a difference that would be metres of
    NEGATIVE driving, and a trust above 1.0 is a lie in the other direction."""
    trust = GraphTrust()
    trust.update(info(30.0))
    trust.update(info(33.0))
    assert trust.since_m == pytest.approx(3.0)
    trust.update(info(2.0))  # a new session
    assert trust.since_m == pytest.approx(5.0)
    assert trust.trust() <= 1.0


def test_the_decay_length_is_a_live_flag_and_the_report_says_what_it_all_means() -> None:
    """graph_trust_m is written straight onto the clock by the node at every word, so a
    `ros2 param set` changes what the next word claims without a restart."""
    trust = GraphTrust()
    trust.update(info(0.0, hypothesis=0.31))
    trust.update(info(GRAPH_TRUST_M, hypothesis=0.31))
    assert trust.trust() == pytest.approx(math.exp(-1.0))
    trust.trust_m = 2 * GRAPH_TRUST_M
    assert trust.trust() == pytest.approx(math.exp(-0.5))

    report = trust.report()
    assert report.heard is True and report.ties == 0
    assert "5.0 m since the start, no tie yet" in report.text()
    assert "hypothesis 0.31" in report.text()


def test_with_no_info_at_all_the_word_claims_what_it_claimed_before_this_existed() -> None:
    """A graph whose statistics never arrive is not evidence of drift — the old behaviour
    stands, and the report line says the topic is silent so a session is never judged on a
    number nothing fed. The file-anchor cap still holds: it needs no statistic."""
    trust = GraphTrust()
    assert trust.trust() == pytest.approx(1.0)
    assert "no /rtabmap/info yet" in trust.report().text()
    assert trust.trust(anchor_from_file=True) == pytest.approx(FILE_ANCHOR_TRUST)


def test_statistics_a_message_does_not_carry_leave_the_clock_where_it_was() -> None:
    """RTAB-Map publishes the full table only with Rtabmap/PublishStats on; a message without
    the distance must not reset the clock, and one without a hypothesis must not zero it."""
    trust = GraphTrust()
    trust.update(info(0.0, hypothesis=0.5))
    trust.update(info(4.0, hypothesis=0.5))
    trust.update({PROXIMITY_ICP: 0.0})
    assert trust.since_m == pytest.approx(4.0)
    assert trust.report().hypothesis == pytest.approx(0.5)
