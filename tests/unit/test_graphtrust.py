"""What a pose graph's word is worth (pepin.graphtrust).

The failure this module exists to stop, in one line: on 2026-09-15 a graph that had recognised
nothing against the database it loaded said nineteen words about a metre and 150 degrees out, each
claiming fit 1.00 — so nothing downstream could tell a recognition from dead reckoning.
"""

import math

import pytest

from pepin.fusion import GATE
from pepin.graphtrust import (
    AGREEMENT_SCALE,
    AGREEMENT_SCALE_M,
    HIGHEST_HYPOTHESIS,
    Agreement,
    stat,
)
from pepin.watch import ADMIT_FIT


def test_a_statistic_is_found_through_the_unit_segment_its_key_carries() -> None:
    """RTAB-Map's keys end in a unit ("Loop/Highest_hypothesis_value/") and a lookup that missed
    one would read a silent zero, so a report line would say the graph is nowhere near recognising
    anything while it is one frame away."""
    stats = {"Loop/Highest_hypothesis_value/": 0.04, "Memory/Distance_travelled/m": 12.5}
    assert stat(stats, HIGHEST_HYPOTHESIS) == 0.04
    assert stat(stats, "Memory/Distance_travelled") == 12.5, "the unit segment is tolerated"
    assert stat(stats, "Memory/Nothing_of_the_sort") is None, "a missing statistic is not a 0"


def test_the_agreement_trust_follows_the_words_and_forgets_them_after_the_window() -> None:
    """What a word is worth while it is being said, rather than however many metres ago the last
    closure was: a word that tracks the tracker's odometry-propagated pose to the centimetre is
    worth everything, and one a metre out is worth nothing from its first appearance."""
    agreement = Agreement()
    assert agreement.rms() is None and agreement.trust() == pytest.approx(1.0)

    for moment in (0.0, 0.5, 1.0):
        agreement.add(moment, 0.01)
    assert agreement.rms() == pytest.approx(0.01)
    assert agreement.trust() == pytest.approx(math.exp(-0.1))

    agreement.add(1.5, 1.0)  # the frame slips a metre under the cart
    assert agreement.trust() < 0.05, "one word that far out already costs the graph its vote"

    agreement.add(20.0, 0.0)  # ...and ten seconds later nothing older is evidence
    assert agreement.rms(20.0) == pytest.approx(0.0)
    assert agreement.trust(20.0) == pytest.approx(1.0)
    assert agreement.count == 1


def test_the_agreement_window_keeps_only_its_last_words() -> None:
    """The window is the last ten words as well as the last ten seconds: a graph publishing fast
    must not be vouched for by what it said a hundred words ago."""
    agreement = Agreement()
    for step in range(30):
        agreement.add(0.01 * step, 1.0 if step < 20 else 0.0)
    assert agreement.count == 10 and agreement.rms() == pytest.approx(0.0)
    assert agreement.trust() == pytest.approx(1.0)
    assert Agreement(scale=AGREEMENT_SCALE).trust() == pytest.approx(1.0)


def test_the_agreement_residual_counts_the_heading_and_not_only_the_metres() -> None:
    """A word can be in exactly the right place facing 90 degrees the wrong way, and until
    2026-09-18 such a word kept its fit 1.00 because only the position residual was measured. The
    residual is 3-DOF now — the Mahalanobis distance the fusion's own gate judges — and a caller
    that offers no covariance keeps the curve it had, so the metres alone still read as before."""
    metres_only = Agreement()
    metres_only.add(0.0, 0.01)
    assert metres_only.trust() == pytest.approx(math.exp(-0.01 / AGREEMENT_SCALE_M))
    assert metres_only.rms() == pytest.approx(0.01)

    # the same centimetre of position, and a heading four sigmas out: a different word entirely
    turned = Agreement()
    turned.add(0.0, 0.01, math.sqrt(GATE) + 1.0)
    assert turned.rms() == pytest.approx(0.01), "the scatter is still metres, for the covariance"
    assert turned.sigmas() == pytest.approx(math.sqrt(GATE) + 1.0)
    assert turned.trust() < ADMIT_FIT, "no candidate is admitted on a word facing the wrong way"


def test_a_word_is_worth_the_wider_of_its_floor_and_the_scatter_it_shows() -> None:
    """The floor covers what a word cannot see in itself — a whole frame sitting to one side.
    The scatter is what it can: when words stop agreeing with the pose the odometry carries
    between them, the graph is coming apart and the word must say so in metres."""
    from pepin.measurements import GRAPH_FLOOR_XY_M

    agreement = Agreement()
    assert agreement.rms(0.0) is None, "nothing seen yet: the floor stands alone"
    for i in range(4):
        agreement.add(float(i), 0.01)
    quiet = agreement.rms(4.0)
    assert quiet is not None and max(GRAPH_FLOOR_XY_M, quiet) == GRAPH_FLOOR_XY_M
    for i in range(4):
        agreement.add(5.0 + i, 0.55)
    loud = agreement.rms(9.0)
    assert loud is not None and max(GRAPH_FLOOR_XY_M, loud) > GRAPH_FLOOR_XY_M, (
        "words scattering half a metre are not worth the floor"
    )
