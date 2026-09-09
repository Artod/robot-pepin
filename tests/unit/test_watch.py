"""When the whole map is searched, and what may be done with the answer."""

from pepin.odometry import Pose2D
from pepin.watch import (
    ADMIT_FIT,
    BLIND_FIT,
    DRIVE_FIT,
    LOST_FIT,
    PROVISIONAL_FIT_CAP,
    BlindDriveWatch,
    LostWatch,
    Verdict,
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
