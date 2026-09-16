"""When the whole map is searched, and what may be done with the answer."""

import math

import numpy as np
import pytest

from pepin.measurements import GRAPH_FLOOR_XY_M
from pepin.odometry import Pose2D
from pepin.watch import (
    ADMIT_FIT,
    BLIND_FIT,
    BY_FIT,
    BY_SIGMA,
    BY_TF,
    CORRECTION_FRESH_S,
    DRIVE_FIT,
    DRIVE_SIGMA_M,
    GRAPH_AGREE_M,
    LOST_FIT,
    LOST_SIGMA_M,
    PAINT_EDGE_FRESH_S,
    PAINT_SIGMA_M,
    PROVISIONAL_FIT_CAP,
    SIGMA_MEDIAN_S,
    SOURCE_PATIENCE_S,
    TF_FRESH_S,
    UNKNOWN_SIGMA,
    BlindDriveWatch,
    Correction,
    GoalGate,
    LostWatch,
    PaintTrust,
    PoseSpread,
    Preflight,
    Readiness,
    Sigma,
    SigmaWindow,
    SourceSilence,
    Verdict,
    source_words,
)

BASE = Pose2D(-9.5, 2.4, 0.9)
CORNER = Pose2D(-13.5, 2.0, -2.3)  # the look-alike four metres away, 2026-09-09


def test_the_fit_scale_is_ordered_the_way_the_design_depends_on() -> None:
    """Six thresholds once lived in three files, unordered: a drive could start at a fit the
    tracker called lost, and an unconfirmed fix could pass the drive gate."""
    assert BLIND_FIT < PROVISIONAL_FIT_CAP < ADMIT_FIT <= DRIVE_FIT <= LOST_FIT


def test_a_healthy_fit_never_triggers_a_search() -> None:
    watch = LostWatch()
    assert not any(watch.observe(0.8, moving=False, navigating=False, now=t) for t in range(10))


def test_three_poor_checks_standing_still_trigger_one_search() -> None:
    watch = LostWatch()
    assert not watch.observe(0.2, moving=False, navigating=False, now=1.0)
    assert not watch.observe(0.2, moving=False, navigating=False, now=2.0)
    assert watch.observe(0.2, moving=False, navigating=False, now=3.0)
    assert not watch.observe(0.2, moving=False, navigating=False, now=4.0)  # the streak restarts


def test_a_driving_robot_is_never_re_seeded() -> None:
    watch = LostWatch()
    for t in range(1, 8):
        assert not watch.observe(0.1, moving=True, navigating=False, now=float(t))
    for t in range(8, 15):
        assert not watch.observe(0.1, moving=False, navigating=True, now=float(t))


def test_a_collapse_asks_at_once_and_forgives_the_backoff() -> None:
    watch = LostWatch()
    watch.observe(0.7, moving=False, navigating=False, now=1.0)
    for k in range(3):
        watch.answer(None, 0.0, 0.2, scan=k, now=2.0)  # three failures: the backoff is long
    assert watch.wait_s() > watch.cooldown_s
    assert watch.observe(0.1, moving=False, navigating=False, now=60.0), "a carry asks at once"
    assert watch.wait_s() == watch.cooldown_s, "a new carry is a new question"


def test_a_search_s_answer_is_a_candidate_and_moves_nothing() -> None:
    """The first version seeded every answer; the map spun between two look-alikes."""
    watch = LostWatch()
    first = watch.answer(CORNER, 0.7, 0.2, scan=1, now=1.0)
    assert first.verdict is Verdict.CANDIDATE
    assert not watch.confirmed
    assert watch.reported_fit(0.72) <= PROVISIONAL_FIT_CAP, "an unconfirmed fix must not look good"
    assert watch.observe(0.2, moving=False, navigating=False, now=1.5), "asked again at once"


def test_only_a_second_search_that_agrees_moves_the_tracker_and_only_once() -> None:
    watch = LostWatch()
    watch.answer(CORNER, 0.7, 0.2, scan=1, now=1.0)
    assert watch.answer(BASE, 0.7, 0.2, scan=2, now=3.0).verdict is Verdict.HOLD  # nothing moves
    assert watch.answer(CORNER, 0.7, 0.2, scan=3, now=5.0).verdict is Verdict.HOLD
    final = watch.answer(Pose2D(-13.4, 2.1, -2.2), 0.7, 0.2, scan=4, now=7.0)
    assert final.verdict is Verdict.APPLY and final.pose is not None and final.confidence == 0.7
    assert watch.confirmed
    assert watch.reported_fit(0.72) == 0.72


def test_the_exact_sequence_of_the_night_moves_exactly_once() -> None:
    """From the board log: corner, base, corner, base. Four seeds then; one now, and at the base."""
    watch = LostWatch()
    answers = [CORNER, BASE, CORNER, BASE]
    moves = [
        a.pose
        for k, pose in enumerate(answers)
        if (a := watch.answer(pose, 0.7, 0.2, scan=k, now=float(k))).verdict is Verdict.APPLY
    ]
    assert len(moves) <= 1


def test_an_answer_no_better_than_the_tracker_is_not_a_candidate() -> None:
    """The admission gates live in the machine now: a first answer must clear ADMIT_FIT and beat
    the tracker by the margin; a second one is judged by the same rule, not a weaker one."""
    watch = LostWatch()
    assert watch.answer(BASE, 0.44, 0.2, scan=1, now=1.0).verdict is Verdict.NOTHING
    assert watch.answer(BASE, 0.65, 0.60, scan=2, now=2.0).verdict is Verdict.NOTHING
    assert watch.confirmed
    assert watch.answer(BASE, 0.7, 0.2, scan=3, now=3.0).verdict is Verdict.CANDIDATE
    weak = watch.answer(BASE, 0.44, 0.2, scan=4, now=4.0)  # agrees, but is not admitted
    assert weak.verdict is Verdict.HOLD and not watch.confirmed


def test_a_search_replayed_on_the_same_scan_is_not_a_second_opinion() -> None:
    """The /relocalize service used to loop on the executor thread with the scan frozen, so its
    'second search' was the first one to the millimetre and every candidate was rubber-stamped."""
    watch = LostWatch()
    watch.answer(CORNER, 0.7, 0.2, scan=7, now=1.0)
    assert watch.answer(CORNER, 0.7, 0.2, scan=7, now=2.0).verdict is Verdict.REPLAY
    assert not watch.confirmed, "nothing was learned"
    assert watch.answer(CORNER, 0.7, 0.2, scan=8, now=3.0).verdict is Verdict.APPLY


def test_a_held_candidate_remembers_the_scan_it_came_from() -> None:
    watch = LostWatch()
    watch.answer(CORNER, 0.7, 0.2, scan=1, now=1.0)
    assert watch.answer(BASE, 0.7, 0.2, scan=2, now=2.0).verdict is Verdict.HOLD
    assert watch.answer(BASE, 0.7, 0.2, scan=2, now=3.0).verdict is Verdict.REPLAY
    assert watch.answer(BASE, 0.7, 0.2, scan=3, now=4.0).verdict is Verdict.APPLY


def test_a_second_first_answer_does_not_restart_the_episode() -> None:
    """proposed() twice used to reset the give-up counter; one door means one episode."""
    watch = LostWatch()
    watch.answer(CORNER, 0.7, 0.2, scan=1, now=1.0)
    for k in range(2, 6):
        watch.answer(Pose2D(5.0 * k, 0, 0), 0.7, 0.2, scan=k, now=float(k))
    assert watch.confirmed, "given up after CONFIRM_TRIES disagreements, however they arrived"


def test_a_tracker_that_recovers_by_itself_is_never_overridden() -> None:
    """Run 0052: a healthy lock (0.62 and up) is better evidence than a one-scan search."""
    watch = LostWatch()
    watch.answer(CORNER, 0.7, 0.2, scan=1, now=1.0)
    assert not watch.observe(0.65, moving=False, navigating=False, now=2.0)
    assert watch.confirmed, "the candidate is dropped, the tracker stays"
    assert watch.reported_fit(0.65) == 0.65


def test_searches_that_never_agree_are_given_up_not_looped_forever() -> None:
    watch = LostWatch()
    watch.answer(Pose2D(0, 0, 0), 0.7, 0.2, scan=0, now=0.0)
    verdicts = []
    for k in range(1, 10):  # every answer somewhere else: a scan that fits two places alike
        verdicts.append(watch.answer(Pose2D(5.0 * k, 0, 0), 0.7, 0.2, scan=k, now=float(k)).verdict)
        if verdicts[-1] is Verdict.GIVEN_UP:
            break
    assert verdicts[-1] is Verdict.GIVEN_UP and len(verdicts) <= 4 and watch.confirmed
    assert Verdict.APPLY not in verdicts
    assert not watch.observe(0.2, moving=False, navigating=False, now=10.0), "quiet after giving up"


def test_a_hand_s_seed_is_trusted_at_once() -> None:
    watch = LostWatch()
    watch.seeded(now=1.0)
    assert watch.confirmed
    assert not watch.observe(0.9, moving=False, navigating=False, now=2.0)


def test_no_match_yet_is_reported_as_zero_not_nan() -> None:
    """Every downstream gate compares with '<', and NaN passes them all: a tracker that had not
    matched yet let a drive start blind."""
    watch = LostWatch()
    assert watch.reported_fit(float("nan")) == 0.0
    watch.answer(CORNER, 0.7, 0.2, scan=1, now=1.0)
    assert watch.reported_fit(float("nan")) == 0.0


def test_a_blind_drive_is_stopped_after_the_patience_and_not_before() -> None:
    """Run 0052 drove a minute at fit 0.05-0.29 and arrived drunk."""
    blind = BlindDriveWatch()
    assert not blind.observe(0.1, now=0.0)
    assert not blind.observe(0.1, now=3.9)
    assert blind.observe(0.1, now=4.1)


def test_a_dip_in_the_fit_does_not_stop_a_drive() -> None:
    blind = BlindDriveWatch()
    for t in (0.0, 1.0, 2.0):
        assert not blind.observe(0.1, now=t)
    assert not blind.observe(0.6, now=3.0), "recovered: the clock restarts"
    assert not blind.observe(0.1, now=6.0)


def test_a_healthy_tracker_drops_a_pending_candidate_and_stops_searching() -> None:
    """The failure of 2026-09-09 18:00: a twin candidate pended, the capped reported fit (0.35)
    was fed back to observe(), and the map was searched every second at a true fit of 0.76.
    observe() judges the tracker's own fit: at 0.76 the candidate is dropped, nothing is asked."""
    w = LostWatch(lost_fit=0.55, lost_checks=3, cooldown_s=8.0)
    w.answer(Pose2D(-13.4, 2.0, -2.3), 0.75, 0.40, scan=1, now=0.0)  # a twin, admitted
    assert not w.confirmed
    assert abs(w.reported_fit(0.76) - 0.35) < 1e-9  # the outside world sees "unconfirmed"
    assert w.observe(0.76, moving=False, navigating=False, now=1.0) is False
    assert w.confirmed  # dropped: a healthy lock beats any one-scan search
    for t in range(2, 20):
        assert w.observe(0.76, moving=False, navigating=False, now=float(t)) is False


def test_an_occluded_scan_is_not_lost() -> None:
    """A person beside the cart takes a quarter of the scan and the fit falls; that is not a
    wrong pose. Occluded checks never search and never count towards being lost."""
    w = LostWatch(lost_fit=0.55, lost_checks=3, cooldown_s=8.0)
    for t in range(10):
        assert w.observe(0.30, moving=False, navigating=False, now=float(t), occluded=True) is False
    assert w.observe(float("nan"), moving=False, navigating=False, now=11.0, occluded=True) is False
    # the same three low checks unoccluded do ask for a search
    hits = [w.observe(0.30, moving=False, navigating=False, now=20.0 + t) for t in range(3)]
    assert hits == [False, False, True]


def test_a_goal_starts_on_the_tracker_s_fit_where_a_tracker_speaks() -> None:
    """The known-map stack is unchanged by the gate: at or above the drive rung the goal goes,
    under it the goal buys one whole-map search first (the node's own _find_myself)."""
    gate = GoalGate()
    assert gate.verdict(DRIVE_FIT, None) == Readiness(True, tracker=True, rule=BY_FIT)
    lost = gate.verdict(0.31, None)
    assert (lost.ready, lost.tracker, lost.search) == (False, True, True)
    assert "fit 0.31 under 0.50" in lost.reason


def test_without_a_tracker_a_fresh_transform_is_what_lets_a_goal_start() -> None:
    """Online SLAM has no tracker and therefore no fit: nobody publishes /localization_fit, and
    the gate that waited for one refused every goal on 2026-09-13. The evidence there is the
    pose's own edge — map -> base_link, re-broadcast at 10 Hz from RTAB-Map's correction."""
    gate = GoalGate()
    assert gate.verdict(None, 0.08) == Readiness(True, tracker=False, rule=BY_TF)
    assert gate.verdict(None, TF_FRESH_S) == Readiness(True, tracker=False, rule=BY_TF), (
        "the bound is allowed"
    )
    stale = gate.verdict(None, 4.2)
    assert (stale.ready, stale.tracker, stale.search) == (False, False, False)
    assert "map -> base_link is 4.2 s old" in stale.reason and "fresher than 1.0 s" in stale.reason
    missing = gate.verdict(None, None)
    assert not missing.ready and not missing.search
    assert "nothing publishes map -> base_link" in missing.reason, "which of the two it was"


def test_the_gate_never_searches_where_there_is_nothing_to_search_with() -> None:
    """A whole-map search is the tracker's own service: asking for one without a tracker is how
    a goal ended as "the tracker is not up". No fit, no search — the refusal is final until the
    transform comes back."""
    gate = GoalGate()
    assert not any(gate.verdict(None, age).search for age in (None, 0.0, 0.5, 10.0))
    assert all(gate.verdict(fit, None).tracker for fit in (0.0, 0.49, 0.5, 0.9))


def test_a_fresh_edge_is_no_evidence_that_the_slam_half_is_alive() -> None:
    """The hole this closes: slam_frame re-broadcasts the LAST correction at 10 Hz with a fresh
    stamp, so map -> base_link stays milliseconds old with the laptop shut down — the gate would
    pass, and Nav2, whose costmap reads that same fresh edge, would never time out either. The
    correction's own age is what the laptop cannot fake."""
    gate = GoalGate()
    fresh_edge = 0.08
    assert gate.verdict(None, fresh_edge, Correction(0.1)).ready
    assert gate.verdict(None, fresh_edge, Correction(CORRECTION_FRESH_S)).ready, "the bound"
    dead = gate.verdict(None, fresh_edge, Correction(9.0))
    assert (dead.ready, dead.tracker, dead.search) == (False, False, False)
    assert "the SLAM correction stopped 9.0 s ago" in dead.reason
    never = gate.verdict(None, fresh_edge, Correction(None))
    assert not never.ready and "no SLAM correction has ever arrived" in never.reason


def test_a_correction_nobody_watches_leaves_the_gate_as_it_was() -> None:
    """The default is no correction at all: the known-map stacks, where the fit decides, and
    SLAM with the watch switched off (CLAUDE.md rule 19 — the old behaviour stays reachable)."""
    gate = GoalGate()
    assert gate.verdict(None, 0.08) == Readiness(True, tracker=False, rule=BY_TF)
    assert gate.verdict(0.71, None, Correction(None)) == Readiness(True, tracker=True, rule=BY_FIT)


def test_a_missing_edge_is_named_before_the_correction_behind_it() -> None:
    """Two halves, two refusals: the operator is told which one to go and look at. The edge
    first — a board that stopped broadcasting is a board-side failure, whatever the laptop
    does."""
    gate = GoalGate()
    assert "nothing publishes map -> base_link" in gate.verdict(None, None, Correction(9.0)).reason
    assert "map -> base_link is 4.2 s old" in gate.verdict(None, 4.2, Correction(9.0)).reason


def test_the_correction_is_stale_the_moment_it_is_older_than_the_patience() -> None:
    """The pulse is 10 Hz and it crosses the bridge: the patience covers a wireless hiccup, not
    a laptop that went away. Never heard is stale too — there is no half to trust."""
    assert not Correction(0.0).stale() and not Correction(CORRECTION_FRESH_S).stale()
    assert Correction(CORRECTION_FRESH_S + 0.01).stale() and Correction(None).stale()
    assert not Correction(3.0).stale(patience_s=5.0)


def test_a_fit_no_source_earned_is_published_as_nothing() -> None:
    """2026-09-14: the tracker published 0.70 for 141 s with no source arriving and the goal
    server drove two goals on it. The number falls the moment the silence passes the patience,
    and it falls all the way to 0.0 — under DRIVE_FIT so the next goal is refused, and under
    BLIND_FIT so the drive already running is stopped by its own watch."""
    silence = SourceSilence()
    assert not silence.silent(SOURCE_PATIENCE_S) and silence.silent(SOURCE_PATIENCE_S + 0.01)
    assert silence.reported(0.70, 0.4) == 0.70
    assert silence.reported(0.70, 141.0) == 0.0 < BLIND_FIT
    assert SourceSilence(patience_s=10.0).reported(0.70, 5.0) == 0.70
    assert silence.silent(math.inf), "a source that never spoke is the loudest silence"


def test_the_silence_says_how_long_it_has_lasted() -> None:
    """The report line an operator reads to tell a quiet cart from a blind one."""
    silence = SourceSilence()
    assert silence.phrase(0.4) == "last source 0.4 s ago"
    assert silence.phrase(141.0) == "no source for 141.0 s"
    assert silence.phrase(math.inf) == "no source ever"


def test_the_switch_off_reports_the_silence_and_publishes_the_fit_anyway() -> None:
    """The flag's off state, so the old behaviour stays reachable in the field: the silence is
    still measured and still said out loud, and only the withholding stops."""
    silence = SourceSilence(zeroes_fit=False)
    assert silence.reported(0.70, 141.0) == 0.70
    assert silence.silent(141.0) and not silence.held_at_zero(141.0)
    assert silence.phrase(141.0) == "no source for 141.0 s"


def test_a_pose_is_painted_with_only_while_every_half_of_it_holds() -> None:
    """What may be written into a map frame. The volume is painted in map coordinates and a TSDF
    cannot be un-integrated, so each half is a veto: the fit, WHEN it was measured, the tracker's
    own sigma where it publishes one, and the age of the correction the pose stands on."""
    trust = PaintTrust()
    assert trust.refusal(fit=0.90, fit_age_s=0.1, edge_age_s=0.05) is None

    assert "fit 0.31" in (trust.refusal(fit=0.31, fit_age_s=0.1, edge_age_s=0.0) or "")
    # The failure of 2026-09-15: the topic stopped and the last good number stayed behind.
    stopped = trust.refusal(fit=0.90, fit_age_s=141.0, edge_age_s=0.0) or ""
    assert stopped == "the fit stopped 141.0 s ago"
    assert trust.refusal(fit=0.90, fit_age_s=SOURCE_PATIENCE_S + 0.1, edge_age_s=0.0) is not None

    wide = trust.refusal(fit=0.90, fit_age_s=0.1, sigma_xy_m=0.42, edge_age_s=0.0) or ""
    assert wide == "sigma 0.42 m over 0.10 m"
    assert trust.refusal(fit=0.90, fit_age_s=0.1, sigma_xy_m=PAINT_SIGMA_M, edge_age_s=0.0) is None


def test_an_absent_sigma_is_no_refusal_and_an_absent_correction_is() -> None:
    """Nothing publishes /localization/sigma yet, and the gate must work without it — while a
    pose cannot be placed at all on a correction TF does not hold, and one nobody has refreshed
    for a second places the cart where it WAS."""
    trust = PaintTrust()
    assert trust.refusal(fit=0.90, fit_age_s=0.1, sigma_xy_m=None, edge_age_s=0.0) is None
    assert trust.refusal(fit=0.90, fit_age_s=0.1, edge_age_s=None) == "no map -> odom edge"
    stale = trust.refusal(fit=0.90, fit_age_s=0.1, edge_age_s=PAINT_EDGE_FRESH_S + 0.5) or ""
    assert stale == "the map -> odom edge is 1.5 s from the scan"
    # An edge NEWER than the scan is just as far from it: a pose is placed by a correction that
    # covers the moment, whichever side of it the correction sits.
    assert trust.refusal(fit=0.90, fit_age_s=0.1, edge_age_s=-2.0) is not None


# -- the one uncertainty: the fusion's sigma ------------------------------------------------

SURE = np.diag([0.02**2, 0.02**2, math.radians(1.0) ** 2])  # a fused match of the good kind


def test_the_sigma_ladder_is_ordered_and_sits_inside_the_cart() -> None:
    """The two thresholds a drive lives by, read against the cart itself: the footprint is
    0.55 m wide and Nav2 calls 0.10 m arrived. Since 2026-09-16 the ladder has to admit the pose
    GRAPH, whose word is worth 0.20 m (pepin.measurements.GRAPH_FLOOR_XY_M) — so a goal starts
    under half the cart's width and a drive is cut under its whole width, and a pose nothing has
    measured is still over both."""
    assert GRAPH_FLOOR_XY_M < DRIVE_SIGMA_M < LOST_SIGMA_M < 0.55, (
        "the graph's honest word must be able to start a drive, and neither rung may exceed the"
        " cart's own width, past which its outline means nothing"
    )
    assert GRAPH_AGREE_M < DRIVE_SIGMA_M
    assert UNKNOWN_SIGMA[0] > LOST_SIGMA_M, "before the first word nothing may drive"


def test_the_sigma_grows_along_the_odometry_and_collapses_on_a_word() -> None:
    """The whole decision of the day in one test: a fused word makes the pose sure, and driving
    away from it with nothing correcting it makes it less sure, metre by metre — the filter's
    prediction step with this cart's own measured odometry error (pepin.fusion)."""
    spread = PoseSpread()
    assert spread.sigma() == UNKNOWN_SIGMA, "no word yet is not a small number"
    spread.corrected(SURE, Pose2D(1.0, 0.0, 0.0), Pose2D(0.0, 0.0, 0.0), now=100.0)
    sure_xy, sure_yaw = spread.sigma()
    assert sure_xy == pytest.approx(0.02, abs=1e-6) and sure_yaw == pytest.approx(1.0, abs=1e-6)
    assert spread.age_s(100.4) == pytest.approx(0.4)
    # Metre by metre with nothing correcting: the heading's own sigma is what a carry converts
    # into position error, so the growth is LINEAR in the distance driven (1 deg of heading is
    # 1.7 cm per metre) rather than the square root a pile of independent steps would give.
    over_drive = over_lost = 0
    for step in range(1, 31):
        spread.carried(Pose2D(float(step), 0.0, 0.0))
        over_drive = over_drive or (step if spread.sigma()[0] > DRIVE_SIGMA_M else 0)
        over_lost = over_lost or (step if spread.sigma()[0] > LOST_SIGMA_M else 0)
    assert (over_drive, over_lost) == (14, 22), (
        "14 m of dead reckoning refuses a goal, 22 m cuts one — the distances the raised ladder"
        " of 2026-09-16 buys (they were 8 m and 14 m at 0.15/0.25)"
    )
    spread.corrected(SURE, Pose2D(31.0, 0.0, 0.0), Pose2D(30.0, 0.0, 0.0), now=120.0)
    assert spread.sigma()[0] == pytest.approx(sure_xy, abs=1e-6), "one word collapses it again"


def test_a_turn_nobody_corrects_is_what_really_costs_this_cart() -> None:
    """Carpet eats 40-60 % of every in-place turn this differential drive reports (the gyro
    measured it, 2026-09-11), so the heading is where dead reckoning falls apart first — and
    the sigma has to say so while the position sigma is still small."""
    spread = PoseSpread()
    spread.corrected(SURE, Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 0.0, 0.0), now=0.0)
    spread.carried(Pose2D(0.0, 0.0, math.radians(90.0)))
    _, yaw = spread.sigma()
    assert yaw > 30.0, f"a quarter turn on nobody's word is not 1 deg of certainty: {yaw:.1f}"


def test_the_path_costs_and_not_the_displacement() -> None:
    """A cart that drives out and comes back is not suddenly sure of itself: the prediction step
    is accumulated step by step, so it is the metres travelled that widen the pose."""
    there_and_back = PoseSpread()
    there_and_back.corrected(SURE, Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 0.0, 0.0), now=0.0)
    for x in (1.0, 2.0, 1.0, 0.0):
        there_and_back.carried(Pose2D(x, 0.0, 0.0))
    stayed = PoseSpread()
    stayed.corrected(SURE, Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 0.0, 0.0), now=0.0)
    stayed.carried(Pose2D(0.0, 0.0, 0.0))
    assert there_and_back.sigma()[0] > stayed.sigma()[0]


def test_the_position_sigma_is_the_widest_direction_never_the_average() -> None:
    """A corridor pins the pose across and leaves it loose along: the number a gate reads is the
    loose direction, or a cart sliding down a corridor would report itself as certain."""
    spread = PoseSpread()
    corridor = np.diag([0.30**2, 0.01**2, math.radians(1.0) ** 2])
    spread.corrected(corridor, Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 0.0, 0.0), now=0.0)
    assert spread.sigma()[0] == pytest.approx(0.30, abs=1e-6)


def test_a_goal_is_judged_by_the_sigma_where_there_is_one_and_the_fit_where_there_is_not() -> None:
    """The camera-only drive of 2026-09-15: /localization_fit is 0.00 because no lidar scan
    scored the pose, while the fusion holds it to 6 cm. The sigma outranks the fit, and where no
    sigma is published (an older build on the board) the fit rules are untouched."""
    gate = GoalGate()
    camera_only = gate.verdict(0.0, None, sigma=Sigma(0.06, 1.2, 0.1))
    assert camera_only.ready and camera_only.rule == BY_SIGMA
    assert not gate.verdict(0.0, None).ready, "the same drive on an older board: the fit rules"
    assert gate.verdict(0.0, None).rule == BY_FIT
    wide = gate.verdict(0.95, None, sigma=Sigma(0.41, 8.0, 0.1))
    assert (wide.ready, wide.search, wide.rule) == (False, True, BY_SIGMA)
    assert "0.41 m" in wide.reason and f"{DRIVE_SIGMA_M:.2f}" in wide.reason
    bound = gate.verdict(0.0, None, sigma=Sigma(DRIVE_SIGMA_M, 3.0, 0.1))
    assert bound.ready, "the threshold itself is allowed, like every other bound here"


def test_a_sigma_that_stopped_arriving_is_not_a_sigma() -> None:
    """The tracker publishes one every check period whatever the sensors do, so silence here is
    the tracker itself — not a quiet room — and no goal starts on the last number it said."""
    gate = GoalGate()
    dead = gate.verdict(0.9, None, sigma=Sigma(0.02, 0.5, SOURCE_PATIENCE_S + 1.0))
    assert not dead.ready and dead.rule == BY_SIGMA
    assert "stopped 4.0 s ago" in dead.reason
    assert gate.verdict(0.9, None, sigma=Sigma(0.02, 0.5, SOURCE_PATIENCE_S)).ready, "the bound"


def test_the_certainty_is_the_window_s_median_and_not_the_last_flicker() -> None:
    """The nook of 2026-09-16: the lidar's match flips between two hypotheses and the published
    sigma flickers 0.01 <-> 0.31 m from one revolution to the next. One sample decides by luck —
    the median over the window decides by what the place actually is."""
    window = SigmaWindow()
    assert window.median(0.0) is None, "no word yet is not a number: the fit rules answer"
    flicker = (0.01, 0.31, 0.02, 0.31, 0.01, 0.30, 0.02)
    for i, xy in enumerate(flicker):
        window.add(Sigma(xy, 2.0, 0.0), now=100.0 + 0.25 * i)
    now = 100.0 + 0.25 * (len(flicker) - 1)
    reading = window.median(now)
    assert reading is not None
    assert reading.xy_m == pytest.approx(0.02), (
        f"the median of the nook, not whichever scan landed last: {reading.xy_m:.3f} m"
    )
    assert reading.xy_m <= DRIVE_SIGMA_M, "a place that is mostly known may be driven from"
    assert window.span() == pytest.approx(1.5)
    # ...and a pose that is really lost moves the median, because MOST of the samples move.
    for i in range(8):
        window.add(Sigma(0.45, 9.0, 0.0), now=now + 0.25 * (i + 1))
    lost = window.median(now + 2.0)
    assert lost is not None and lost.xy_m > LOST_SIGMA_M, f"a real loss lands: {lost.xy_m:.3f} m"


def test_the_window_ages_by_the_newest_sample_so_a_dead_tracker_is_seen() -> None:
    """Freshness is "is the tracker still publishing", never "how old is the middle of the
    window": a median that averaged the ages would let a tracker that stopped look alive."""
    window = SigmaWindow()
    for i in range(5):
        window.add(Sigma(0.05, 1.0, 0.0), now=100.0 + 0.4 * i)
    fresh = window.median(101.7)
    assert fresh is not None and fresh.age_s == pytest.approx(0.1), "the newest sample's age"
    assert fresh.fresh(SOURCE_PATIENCE_S)
    stopped = window.median(130.0)
    assert stopped is not None, "a tracker that stopped is a reading to refuse, not an absence"
    assert stopped.age_s == pytest.approx(28.4) and not stopped.fresh(SOURCE_PATIENCE_S)
    assert stopped.xy_m == pytest.approx(0.05), "the last thing it said, whatever its age"
    assert window.window_s == SIGMA_MEDIAN_S
    window.clear()
    assert window.median(130.0) is None, "a whole-map search seeds a pose the samples never saw"


def test_the_blind_drive_watch_reads_the_sigma_and_says_which_rule_judged() -> None:
    """The watch that cut the camera drives: with a sigma it judges the pose, with none the fit,
    and the phrase it leaves in the log names the reading, so a tape says which rule stopped the
    cart."""
    blind = BlindDriveWatch()
    assert not blind.observe(0.0, 0.0, sigma=Sigma(0.06, 1.2, 0.1)), "a camera drive is not blind"
    assert blind.rule == BY_SIGMA and "0.06 m" in blind.phrase()
    lost = False
    for t in (10.0, 11.0, 12.0, 13.0, 15.0):
        lost = blind.observe(0.99, t, sigma=Sigma(0.55, 9.0, 0.1))
    assert lost and f"over {LOST_SIGMA_M:.2f} m" in blind.phrase()
    fit_only = BlindDriveWatch()
    assert not fit_only.observe(0.9, 0.0) and fit_only.rule == BY_FIT
    by_fit = False
    for t in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        by_fit = fit_only.observe(0.1, t)
    assert by_fit and "fit 0.10 under 0.30" in fit_only.phrase()


def test_a_drive_is_cut_when_the_sigma_stops_arriving_mid_way() -> None:
    """A tracker that dies mid-drive leaves the cart following a plan on dead reckoning; the
    last sigma it published stays small for ever, so it is the AGE that has to stop the drive."""
    blind = BlindDriveWatch()
    cut = False
    for age, t in ((0.1, 0.0), (1.0, 1.0), (4.0, 4.0), (9.0, 9.0), (14.0, 14.0)):
        cut = blind.observe(0.9, t, sigma=Sigma(0.02, 0.5, age))
    assert cut and "stopped 14.0 s ago" in blind.phrase()


# -- the preflight -------------------------------------------------------------------------


def report(**sources: object) -> dict[str, object]:
    """One /localization/sources message, as pepin.localization.Localizer.sources_report writes
    it: the per-source block is what the preflight reads."""
    return {"anchor": "lidar", "fused": "lidar", "rejected": [], "fit": 0.8, "sources": sources}


LIDAR_FRESH = {"health": "fresh 9.9 Hz", "fit": 0.82, "delta": [1.2, -0.4, 0.3]}
CAMERA_FRESH = {"health": "fresh 4.8 Hz", "fit": 0.61, "delta": [3.0, 2.0, 1.1]}
GRAPH_FRESH = {"health": "fresh 1.0 Hz", "fit": 0.55, "delta": [4.0, 3.0, 0.9]}
GRAPH_FAR = {"health": "fresh 1.0 Hz", "fit": 0.55, "delta": [40.0, 30.0, 2.0]}


def test_the_sources_report_is_read_for_who_is_holding_the_pose() -> None:
    """The delta the tracker publishes is in centimetres and the preflight thinks in metres; a
    source the flag has off is not evidence, and a half-written entry costs its own fields
    only."""
    words = {w.name: w for w in source_words(report(lidar=LIDAR_FRESH, camera={"health": "off"}))}
    assert words["lidar"].spoke() and words["lidar"].delta_m == pytest.approx(0.0126, abs=1e-4)
    assert not words["camera"].enabled and not words["camera"].spoke()
    assert source_words({}) == [] and source_words({"sources": "nonsense"}) == []
    broken = source_words(report(lidar={"health": "fresh 9.9 Hz", "fit": "x", "delta": ["a", 1]}))
    assert broken[0].spoke() and broken[0].fit is None and broken[0].delta_m is None
    stale = source_words(report(lidar={"health": "stale 2.1 s"}, camera={"health": "stale 41.0 s"}))
    assert stale[0].spoke(SOURCE_PATIENCE_S) and not stale[1].spoke(SOURCE_PATIENCE_S)


def test_the_preflight_passes_a_lidar_drive_and_a_camera_drive_that_agrees() -> None:
    """Both doors of the good case: the lidar holding the pose, and the camera holding it with
    the graph recognising the room and agreeing with the tracker."""
    flight = Preflight()
    lidar = flight.checks(source_words(report(lidar=LIDAR_FRESH)), Sigma(0.03, 0.8, 0.1))
    assert Preflight.passed(lidar) and "the lidar is holding the pose" in lidar[2].detail
    camera = flight.checks(
        source_words(report(camera=CAMERA_FRESH, graph=GRAPH_FRESH)), Sigma(0.08, 2.0, 0.1)
    )
    assert Preflight.passed(camera)
    assert "judged by sigma" in camera[1].detail and "0.05 m from the tracker" in camera[2].detail


def test_the_preflight_refuses_each_case_with_the_line_that_says_why() -> None:
    """One line per check, and the refusal names the reading — never a bare "not localized"."""
    flight = Preflight()
    silent = flight.checks([], None, fit=0.9)
    assert not Preflight.passed(silent) and "no word at all" in silent[0].detail
    assert "REFUSED" in silent[0].line() and silent[0].line().startswith("preflight sources")
    asleep = flight.checks(
        source_words(report(lidar={"health": "stale 41.0 s"}, camera={"health": "absent"})),
        Sigma(0.03, 0.8, 0.1),
    )
    assert not asleep[0].ok and "no source has spoken in 3 s" in asleep[0].detail
    assert "lidar stale 41.0 s" in asleep[0].detail, "the roster is printed whole"
    wide = flight.checks(source_words(report(lidar=LIDAR_FRESH)), Sigma(0.44, 9.0, 0.2))
    assert not wide[1].ok and "0.44 m" in wide[1].detail
    assert f"needs {DRIVE_SIGMA_M:.2f} m" in wide[1].detail
    old_board = flight.checks(source_words(report(lidar=LIDAR_FRESH)), None, fit=0.31)
    assert not old_board[1].ok and "no /localization/sigma" in old_board[1].detail
    assert flight.checks(source_words(report(lidar=LIDAR_FRESH)), None, fit=0.9)[1].ok


def test_a_camera_only_drive_needs_the_graph_to_recognise_the_room() -> None:
    """Without the lidar nothing scores the scan against the map, so the evidence that the room
    under the cart is the room on the map is RTAB-Map's graph: it has recognised the place (its
    word carries a trust above zero) and it agrees with the tracker about where in it."""
    flight = Preflight()
    no_graph = flight.checks(source_words(report(camera=CAMERA_FRESH)), Sigma(0.08, 2.0, 0.1))
    assert not no_graph[2].ok and "not among the sources" in no_graph[2].detail
    blind_graph = flight.checks(
        source_words(report(camera=CAMERA_FRESH, graph={"health": "fresh 1.0 Hz", "fit": 0.0})),
        Sigma(0.08, 2.0, 0.1),
    )
    assert not blind_graph[2].ok and "recognised nothing" in blind_graph[2].detail
    far = flight.checks(
        source_words(report(camera=CAMERA_FRESH, graph=GRAPH_FAR)), Sigma(0.08, 2.0, 0.1)
    )
    assert not far[2].ok and "0.50 m from the tracker" in far[2].detail
    quiet = flight.checks(
        source_words(report(camera=CAMERA_FRESH, graph={"health": "stale 41.0 s", "fit": 0.6})),
        Sigma(0.08, 2.0, 0.1),
    )
    assert not quiet[2].ok, "a graph that stopped speaking recognises nothing now"
