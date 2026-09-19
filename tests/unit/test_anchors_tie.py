"""The tie as a CALIBRATION (pepin.graphtie): the pairs log, the robust fit, and the covariance
that has to be honest about a fit from one spot.

The failure this module exists to stop, in one line: a tie fitted to ONE seating took three values
metres apart in one evening, and re-fitting it whenever the lidar disagreed baked a false
recognition into every word after.
"""

import json
import math
from pathlib import Path

import pytest

from pepin.fusion import GATE
from pepin.graphtie import (
    FILE_TIE_SIGMA_M,
    PAIRS_SUFFIX,
    TiePair,
    append_pair,
    file_tie,
    fit_tie,
    identity_tie,
    load_pairs,
    pairs_path,
)
from pepin.measurements import compose, inverse
from pepin.odometry import Pose2D, wrap_angle

MAP = "239x215@-18.53,-4.38"
# A tie to measure against: the database's frame a metre out and turned 30 degrees, which is the
# scale of the thing the real flat produced ((-10.3, +1.4, +53 deg), scratch/graph_tie_fit.py).
TRUTH = Pose2D(-10.3, 1.4, math.radians(53.0))


def pair(x: float, y: float, yaw: float = 0.0, tie: Pose2D = TRUTH, **kwargs: float) -> TiePair:
    """One exact pair: a cart at (x, y, yaw) on the map, and the place in the database's frame that
    ``tie`` carries onto it — so a fit over such pairs must answer ``tie`` itself."""
    cart = Pose2D(x, y, yaw)
    return TiePair(stamp=0.0, cart=cart, place=compose(inverse(tie), cart), **kwargs)  # type: ignore[arg-type]


def test_a_map_s_identity_names_the_pairs_log_beside_its_anchor() -> None:
    """The log belongs to the PAIR (this map, this database) and is named by the map, exactly as
    the one-seating anchor file is: an owner listing ros/maps must see which map it is for."""
    assert pairs_path("/maps", MAP).name == f"239x215_-18.53_-4.38{PAIRS_SUFFIX}"


def test_pairs_are_appended_and_come_back_in_order(tmp_path: Path) -> None:
    """The log is append-only on purpose: a calibration is worth what it was measured over, and a
    log that is rewritten cannot be argued with."""
    first, second = pair(1.0, 0.0), pair(2.0, 0.5, math.radians(20.0))
    path = append_pair(tmp_path, MAP, first)
    append_pair(tmp_path, MAP, second)
    assert path == pairs_path(tmp_path, MAP)
    assert len(path.read_text().splitlines()) == 2

    back = load_pairs(tmp_path, MAP)
    assert len(back) == 2
    assert back[0].cart.x == pytest.approx(1.0) and back[1].cart.x == pytest.approx(2.0)
    assert back[1].place.x == pytest.approx(second.place.x, abs=1e-4)
    assert json.loads(path.read_text().splitlines()[0])["map"][0] == 1.0


def test_a_half_written_line_costs_one_pair_and_not_the_log(tmp_path: Path) -> None:
    """A live node appends to this file and a power cut leaves half a line, which is not a reason
    to refuse the other thousand pairs."""
    append_pair(tmp_path, MAP, pair(1.0, 0.0))
    with pairs_path(tmp_path, MAP).open("a") as log:
        log.write('{"stamp": 1.0, "map": [0.0, 0.0\n')
    append_pair(tmp_path, MAP, pair(2.0, 0.0))
    assert len(load_pairs(tmp_path, MAP)) == 2
    assert load_pairs(tmp_path, "no-such-map") == []


def test_the_fit_over_spread_pairs_answers_the_transform_they_were_made_from() -> None:
    """The whole point: pairs taken all over the flat, each exact, must give back one rigid
    transform — not the average of three seatings, and not a seating's lever arm."""
    pairs = [pair(x, y) for x in (-1.0, 0.0, 1.0, 2.0) for y in (-1.0, 0.0, 1.0)]
    tie = fit_tie(pairs)
    assert tie is not None
    assert tie.pose.x == pytest.approx(TRUTH.x, abs=1e-6)
    assert tie.pose.y == pytest.approx(TRUTH.y, abs=1e-6)
    assert tie.pose.theta == pytest.approx(TRUTH.theta, abs=1e-6)
    assert tie.pairs == 12 and tie.inliers == 12 and tie.outlier_share == 0.0
    assert tie.residual_rms_m == pytest.approx(0.0, abs=1e-6)
    assert tie.extent_m > 1.0, "the spread is what the heading is measured over"
    assert "from pairs 12/12 inliers" in tie.described()


def test_a_false_recognition_is_thrown_away_and_then_reported_as_a_rate() -> None:
    """RTAB-Map once claimed a 4.4 deg sigma while it was 93 degrees wrong. A tie re-fitted on
    that sample is wrong for every word after; a tie FITTED OVER MANY pairs throws it out, and the
    share it threw out is the measurement of how often this database does that."""
    pairs = [pair(x, y) for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)]
    liar = pair(0.5, 0.5, tie=Pose2D(-8.0, 4.0, math.radians(-40.0)))
    tie = fit_tie([*pairs, liar])
    assert tie is not None
    assert tie.pose.x == pytest.approx(TRUTH.x, abs=1e-3)
    assert tie.pose.theta == pytest.approx(TRUTH.theta, abs=1e-3)
    assert tie.pairs == 10 and tie.inliers == 9
    assert tie.outlier_share == pytest.approx(0.1)


def test_a_fit_from_one_spot_says_so_instead_of_pretending(tmp_path: Path) -> None:
    """The lever arm, which is the whole disease. Pairs crowded at one place pin the translation
    and say nothing about the rotation by geometry, so the heading may only come from the pairs'
    own headings — and the tie's position uncertainty must then GROW with the distance from that
    spot, so a word at the far wall is wide instead of confident."""
    crowded = [pair(0.0, 0.0), pair(0.01, 0.0), pair(0.0, 0.01)]
    spread = [pair(x, y) for x in (-1.5, 0.0, 1.5) for y in (-1.5, 0.0, 1.5)]
    tight, wide = fit_tie(spread), fit_tie(crowded)
    assert tight is not None and wide is not None
    assert wide.extent_m < 0.02 and tight.extent_m > 1.0
    assert wide.covariance[2][2] > tight.covariance[2][2], "no spread, no geometric heading"
    assert wide.volume > tight.volume, "and that is what 'statistically better' compares"

    here = Pose2D(*(wide.centroid or (0.0, 0.0)))
    far = Pose2D(here.x + 4.0, here.y)
    assert wide.sigma_at(far)[0] > wide.sigma_at(here)[0], "four metres of lever arm"
    assert wide.word_covariance(far)[0][0] + wide.word_covariance(far)[1][1] > (
        wide.word_covariance(here)[0][0] + wide.word_covariance(here)[1][1]
    ), "and the word carries it"


def test_one_pair_is_a_whole_tie_worth_exactly_that_pair() -> None:
    """A single pair with a heading determines a rigid transform completely — which is why it is
    the minimal sample of the robust search. Its covariance is that pair's own and no better, and
    the heading it claims is the pair's own heading difference."""
    only = pair(0.0, 0.0, math.radians(15.0), sigma=(0.03, 0.03, math.radians(1.0)))
    tie = fit_tie([only])
    assert tie is not None
    assert tie.pose.x == pytest.approx(TRUTH.x, abs=1e-6)
    assert tie.pose.theta == pytest.approx(TRUTH.theta, abs=1e-6)
    assert math.sqrt(tie.covariance[0][0]) == pytest.approx(only.sigma_xy, rel=1e-6)
    assert math.sqrt(tie.covariance[2][2]) == pytest.approx(only.sigma_yaw, rel=1e-6)
    assert only.implied().x == pytest.approx(TRUTH.x, abs=1e-6)


def test_pairs_that_disagree_with_a_rigid_model_widen_the_tie_and_do_not_hide_it() -> None:
    """The database measured on 2026-09-18 does not hold still: its sessions sit 1.6 m and 128 deg
    apart. A tie fitted over pairs that scatter more than their own error bars allow must come out
    WIDE (the Birge ratio on the misfit), not confident — silence and a lie are the two answers
    this stack is not allowed to give."""
    exact = [
        pair(x, y, sigma=(0.03, 0.03, math.radians(1.0)))
        for x in (-1.0, 0.0, 1.0)
        for y in (-1.0, 0.0, 1.0)
    ]
    bent = [
        TiePair(
            stamp=p.stamp,
            cart=Pose2D(p.cart.x + 0.10 * index, p.cart.y, p.cart.theta),
            place=p.place,
            sigma=p.sigma,
        )
        for index, p in enumerate(exact)
    ]
    clean, messy = fit_tie(exact), fit_tie(bent)
    assert clean is not None and messy is not None
    assert messy.residual_rms_m > clean.residual_rms_m
    assert messy.covariance[0][0] > clean.covariance[0][0], "the misfit inflates the error bar"


def test_the_identity_tie_needs_no_measurement_and_claims_no_error() -> None:
    """A map and a database born at the same pose in the same second are the same frame BY
    CONSTRUCTION: measuring the identity transform could only add noise to it."""
    tie = identity_tie()
    assert (tie.pose.x, tie.pose.y, tie.pose.theta) == (0.0, 0.0, 0.0)
    assert tie.volume == 0.0 and tie.origin == "identity"
    assert tie.sigma_at(Pose2D(5.0, 5.0)) == (0.0, 0.0)
    assert "identity" in tie.described()


def test_the_one_seating_file_carries_the_error_bar_it_was_measured_to_have() -> None:
    """Re-measured from another single seating across an RTAB-Map restart the same tie moved 0.40 m
    and 6.45 deg. That spread IS what such a transform is worth, so the file declares it — and
    having no reference point, it declares it flat rather than inventing a lever arm."""
    tie = file_tie(Pose2D(-11.1, 5.7, -0.485))
    assert tie.origin == "file" and tie.centroid is None
    near, far = tie.sigma_at(Pose2D()), tie.sigma_at(Pose2D(6.0, 0.0))
    assert near == far, "no centroid, no arm: the measured spread already includes one"
    assert near[0] == pytest.approx(FILE_TIE_SIGMA_M)
    assert f"{FILE_TIE_SIGMA_M * 100:.0f} cm" in tie.described()


def test_a_pair_s_residual_is_judged_in_three_degrees_of_freedom() -> None:
    """A pair can be in exactly the right place with the wrong heading, and the gate that decides
    whether the fit keeps it is the fusion's own chi-square over position AND heading."""
    pairs = [
        pair(x, y, sigma=(0.03, 0.03, math.radians(1.0)))
        for x in (-1.0, 0.0, 1.0)
        for y in (-1.0, 0.0, 1.0)
    ]
    turned = pairs[0]
    twisted = TiePair(
        stamp=turned.stamp,
        cart=Pose2D(turned.cart.x, turned.cart.y, wrap_angle(turned.cart.theta + math.radians(90))),
        place=turned.place,
        sigma=turned.sigma,
    )
    tie = fit_tie([*pairs[1:], twisted])
    assert tie is not None
    assert tie.inliers == len(pairs) - 1, "the right place facing the wrong way is not a pair"
    assert tie.pose.theta == pytest.approx(TRUTH.theta, abs=1e-3)
    assert math.sqrt(GATE) > 1.0, "the knee the Huber weight bends at is the gate's own"


def test_an_empty_log_is_no_tie_at_all() -> None:
    """A node with no pairs has nothing to fit and must say so, not answer identity: identity is a
    claim about the frames and an empty log is the absence of one."""
    assert fit_tie([]) is None
