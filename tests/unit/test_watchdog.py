"""The watchdog's judgement: what a whole-map candidate is worth, and when it moves the tracker.

Every case here is a decision the laptop's search forces on the board — the same place, another
place, a map that fits nothing, two places that fit alike — decided on numbers, with no ROS and
no map: :func:`pepin.watchdog.judge` and :class:`pepin.watchdog.CandidateGate` are pure.
"""

from __future__ import annotations

import math

import numpy as np

from pepin.fusion import sigma_from_fit
from pepin.odometry import Pose2D
from pepin.sources import TRACKER, WATCHDOG
from pepin.watchdog import (
    AMBIGUITY_MAX,
    BEAT_MARGIN,
    UNKNOWN_MAP_FIT,
    CandidateGate,
    CandidateVerdict,
    GlobalCandidate,
    ambiguity,
    judge,
    same_place,
)

MAP = "239x215@-18.53,-4.38"
HERE = Pose2D(-9.5, 2.4, math.radians(50.0))
ELSEWHERE = Pose2D(-13.5, 2.0, math.radians(-135.0))  # the corner four metres away
SURE = np.diag([0.02**2, 0.02**2, math.radians(1.0) ** 2])  # a peak as sharp as a real one


def candidate(
    pose: Pose2D = HERE, score: float = 0.65, ambiguity: float = 0.2, stamp: float = 100.0
) -> GlobalCandidate:
    """A candidate at ``pose``, as sure as a good peak on this flat's map."""
    return GlobalCandidate(pose.x, pose.y, pose.theta, SURE, score, ambiguity, stamp, MAP)


# ---- one candidate ----------------------------------------------------------------------
def test_a_candidate_on_the_tracked_pose_agrees() -> None:
    """The everyday verdict, once a second: the tracker is where the map says it is."""
    near = Pose2D(HERE.x + 0.2, HERE.y - 0.1, HERE.theta + math.radians(10.0))
    assert judge(candidate(near), HERE, 0.60) is CandidateVerdict.AGREE
    assert same_place(near, HERE) and not same_place(ELSEWHERE, HERE)


def test_a_better_place_elsewhere_disagrees() -> None:
    """A carry: the tracker still believes the old spot at 0.30, the map answers four metres
    away at 0.65 and is sure of it."""
    assert judge(candidate(ELSEWHERE), HERE, 0.30) is CandidateVerdict.DISAGREE


def test_a_place_elsewhere_that_is_no_better_claims_nothing() -> None:
    """The tracker fits 0.60, the candidate 0.65: inside the margin, so the candidate has not
    earned a teleport — and must not be counted as agreement either."""
    assert judge(candidate(ELSEWHERE, score=0.60 + BEAT_MARGIN / 2), HERE, 0.60) is (
        CandidateVerdict.NOTHING
    )


def test_a_map_nothing_fits_is_named_as_such() -> None:
    """The detector the whole feature exists to have: the best place on the whole map explains
    less than a true pose ever does here, so this is not a scan of this map at all."""
    poor = candidate(ELSEWHERE, score=UNKNOWN_MAP_FIT - 0.01)
    assert judge(poor, HERE, 0.20) is CandidateVerdict.UNKNOWN_MAP
    assert judge(poor, HERE, 0.80) is CandidateVerdict.UNKNOWN_MAP, "the tracker is not asked"


def test_two_places_that_fit_alike_are_never_acted_on() -> None:
    """A corridor, a symmetric room: the map answers "here, or just as well there"."""
    twin = candidate(ELSEWHERE, score=0.70, ambiguity=AMBIGUITY_MAX + 0.01)
    assert judge(twin, HERE, 0.20) is CandidateVerdict.UNKNOWN_MAP


# ---- how ambiguous the map's answer was -------------------------------------------------
def test_ambiguity_ignores_rivals_that_are_the_same_place() -> None:
    """Several peaks refine into one basin; only a rival somewhere ELSE makes an answer
    ambiguous. The numbers are the search's own ranking measure (Localizer.rank), so a place
    the map denies can score below zero and is simply no rival."""
    best = (HERE, 0.70)
    near = (Pose2D(HERE.x + 0.1, HERE.y, HERE.theta), 0.68)
    assert ambiguity([best, near]) == 0.0
    assert ambiguity([best]) == 0.0
    assert ambiguity([best, (ELSEWHERE, 0.35)]) == 0.5
    assert ambiguity([]) == 1.0
    assert ambiguity([(HERE, 0.0), (ELSEWHERE, 0.0)]) == 1.0, "a map nothing fits is all rivals"
    assert ambiguity([(HERE, 1.5), (ELSEWHERE, -0.2)]) == 0.0, "a denied place is no rival"


def test_a_turned_twin_in_the_same_spot_counts_as_a_rival() -> None:
    """The 180-degree twin of an empty rectangle stands in the same place, facing the other
    way: a different pose, and the one the ambiguity must see."""
    turned = Pose2D(HERE.x, HERE.y, HERE.theta + math.pi)
    assert ambiguity([(HERE, 0.70), (turned, 0.69)]) > AMBIGUITY_MAX


# ---- the streak -------------------------------------------------------------------------
def test_three_candidates_about_one_place_re_seed_the_tracker() -> None:
    """Nothing moves on the first two; the third closes the streak and carries a pose."""
    gate = CandidateGate()
    for i in range(2):
        answer = gate.observe(candidate(ELSEWHERE), HERE, 0.20, MAP)
        assert answer.verdict is CandidateVerdict.DISAGREE and answer.seed is None, i
    answer = gate.observe(candidate(ELSEWHERE), HERE, 0.20, MAP)
    assert answer.seed is not None
    seeded = answer.pending(MAP)
    assert seeded is not None and seeded[0] == MAP
    assert math.hypot(seeded[1].x - ELSEWHERE.x, seeded[1].y - ELSEWHERE.y) < 0.05, (
        "a tracker lost at 0.20 has 0.29 m of sigma: the candidate carries the fusion"
    )
    assert seeded[2] > 0.5, "the fit travels with the seed"
    assert answer.seed.source == f"{TRACKER}+{WATCHDOG}"


def test_a_tracker_that_still_fits_the_map_is_never_overruled() -> None:
    """A candidate must beat the tracker's own fit by the margin: at 0.95 the tracker is
    explaining the scan better than the candidate does, and no streak can form at all."""
    gate = CandidateGate()
    for _ in range(4):
        answer = gate.observe(candidate(ELSEWHERE), HERE, 0.95, MAP)
    assert answer.verdict is CandidateVerdict.NOTHING and answer.seed is None


def test_the_seed_is_the_two_weighed_against_each_other_not_a_teleport() -> None:
    """The tracker holds the old spot at 0.50 (sigma 20 cm), the candidate is sure to 2 cm:
    the seed lands on the candidate to within the few centimetres the tracker's own word is
    worth, and that share is exactly the ratio of their informations."""
    gate = CandidateGate()
    for _ in range(3):
        answer = gate.observe(candidate(ELSEWHERE), HERE, 0.50, MAP)
    assert answer.seed is not None
    sigma_xy, _ = sigma_from_fit(0.50)
    pull = SURE[0, 0] / (SURE[0, 0] + sigma_xy**2)  # the held pose's weight in the fusion
    apart = math.hypot(HERE.x - ELSEWHERE.x, HERE.y - ELSEWHERE.y)
    off = math.hypot(answer.seed.x - ELSEWHERE.x, answer.seed.y - ELSEWHERE.y)
    assert 0.3 * pull * apart < off < 3.0 * pull * apart < 0.2


def test_candidates_that_disagree_about_different_places_never_re_seed() -> None:
    """Three searches landing in three different rooms are noise, not evidence."""
    gate = CandidateGate()
    for x in (-13.5, -11.0, -16.0):
        answer = gate.observe(candidate(Pose2D(x, 2.0, 0.0)), HERE, 0.20, MAP)
        assert answer.seed is None, x


def test_one_agreement_ends_a_streak() -> None:
    gate = CandidateGate()
    gate.observe(candidate(ELSEWHERE), HERE, 0.20, MAP)
    gate.observe(candidate(ELSEWHERE), HERE, 0.20, MAP)
    assert gate.observe(candidate(HERE), HERE, 0.20, MAP).verdict is CandidateVerdict.AGREE
    assert gate.observe(candidate(ELSEWHERE), HERE, 0.20, MAP).seed is None


def test_the_flags_hold_the_gate_shut() -> None:
    """``accept_candidates`` off, and a goal running: judged and counted, never acted on."""
    gate = CandidateGate()
    gate.switch("accept_candidates", False)
    for _ in range(4):
        answer = gate.observe(candidate(ELSEWHERE), HERE, 0.20, MAP)
    assert answer.verdict is CandidateVerdict.DISAGREE and answer.seed is None
    assert answer.pending(MAP) is None
    gate.switch("accept_candidates", True)
    for _ in range(4):
        answer = gate.observe(candidate(ELSEWHERE), HERE, 0.20, MAP, allow=False)
    assert answer.seed is None, "no teleport while a goal runs"
    assert gate.observe(candidate(ELSEWHERE), HERE, 0.20, MAP).seed is not None


def test_the_streak_length_is_a_live_switch() -> None:
    gate = CandidateGate()
    gate.switch("candidate_streak", 1)
    assert gate.observe(candidate(ELSEWHERE), HERE, 0.20, MAP).seed is not None
    try:
        gate.switch("rest_lock", True)
    except ValueError as exc:
        assert "not a switch" in str(exc)
    else:  # pragma: no cover - the raise is the test
        raise AssertionError("a flag of another object must be refused")


def test_a_candidate_from_another_map_is_evidence_about_nothing() -> None:
    """After a map swap the laptop is a moment behind: its candidates are dropped, and they do
    not even break a streak in a way that could be confused with agreement."""
    gate = CandidateGate()
    gate.observe(candidate(ELSEWHERE), HERE, 0.20, MAP)
    gate.observe(candidate(ELSEWHERE), HERE, 0.20, "other")
    assert gate.observe(candidate(ELSEWHERE), HERE, 0.20, MAP).seed is None
    assert "elsewhere 1" in gate.report()


def test_no_tracked_pose_yet_means_no_judgement() -> None:
    gate = CandidateGate()
    answer = gate.observe(candidate(), None, 0.0, MAP)
    assert answer.verdict is CandidateVerdict.NOTHING and answer.seed is None
    assert "no_pose 1" in gate.report()


# ---- the map that does not fit ----------------------------------------------------------
def test_a_streak_of_unknown_maps_says_the_map_does_not_fit() -> None:
    """No mode switch, no re-seed: a latched verdict the report line and /localization/sources
    carry, for the owner to decide on."""
    gate = CandidateGate()
    poor = candidate(ELSEWHERE, score=0.2)
    for _ in range(2):
        assert gate.observe(poor, HERE, 0.2, MAP).verdict is CandidateVerdict.UNKNOWN_MAP
        assert gate.map_fits
    gate.observe(poor, HERE, 0.2, MAP)
    assert not gate.map_fits and gate.status()["map_fits"] is False
    assert "THE MAP DOES NOT FIT" in gate.report()
    gate.observe(candidate(HERE), HERE, 0.60, MAP)
    assert gate.map_fits, "one candidate that fits gives the map the benefit of the doubt"


# ---- the wire -----------------------------------------------------------------------------
def test_a_candidate_survives_the_trip_as_json() -> None:
    """One self-contained message: the board never joins two topics to judge one answer."""
    sent = candidate(ELSEWHERE)
    back = GlobalCandidate.from_json(sent.to_json(verdict="disagree", search_ms=142.0))
    assert back.map_id == MAP and back.stamp == sent.stamp
    assert math.hypot(back.x - sent.x, back.y - sent.y) < 1e-4
    assert abs(back.yaw - sent.yaw) < 1e-4
    assert back.score == sent.score and back.ambiguity == sent.ambiguity
    assert np.allclose(back.covariance, sent.covariance)
    assert back.measurement().source == WATCHDOG


def test_a_message_that_is_not_a_candidate_is_refused_not_obeyed() -> None:
    gate = CandidateGate()
    for text in ("", "{}", '{"x": 1}', '{"x":1,"y":1,"yaw":0,"covariance":[[1,2],[3,4]]}'):
        try:
            GlobalCandidate.from_json(text)
        except (KeyError, TypeError, ValueError) as exc:
            gate.malformed(str(exc))
        else:  # pragma: no cover - the raise is the test
            raise AssertionError(f"{text!r} parsed as a candidate")
    assert "malformed 4" in gate.report()


def test_the_report_names_every_verdict_and_resets() -> None:
    gate = CandidateGate()
    gate.observe(candidate(HERE), HERE, 0.60, MAP)
    gate.observe(candidate(ELSEWHERE), HERE, 0.20, MAP)
    line = gate.report()
    assert "candidates 2" in line and "agree 1" in line and "disagree 1" in line
    assert "re-seeds 0" in line
    assert "candidates 0 (none)" in gate.report(), "the counters are per report period"
    assert gate.status()["verdict"] == "disagree" and gate.status()["accept"] is True


def test_a_tracker_that_has_not_matched_anything_yet_does_not_silence_a_candidate() -> None:
    """The node's fit is NaN until its first check; NaN passes every ``<`` silently, so it is
    read as 0.0 exactly as pepin.watch does."""
    assert judge(candidate(ELSEWHERE), HERE, float("nan")) is CandidateVerdict.DISAGREE
