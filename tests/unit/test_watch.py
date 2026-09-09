"""When the whole map is searched, and what may be done with the answer."""

from pepin.odometry import Pose2D
from pepin.watch import PROVISIONAL_FIT_CAP, BlindDriveWatch, LostWatch

BASE = Pose2D(-9.5, 2.4, 0.9)
CORNER = Pose2D(-13.5, 2.0, -2.3)  # the look-alike four metres away, 2026-09-09


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
    for _ in range(3):
        watch.searched(found=False, now=2.0)  # three failures: the backoff is long
    assert watch.wait_s() > watch.cooldown_s
    assert watch.observe(0.1, moving=False, navigating=False, now=60.0), "a carry asks at once"
    assert watch.wait_s() == watch.cooldown_s, "a new carry is a new question"


def test_a_search_s_answer_is_a_candidate_and_moves_nothing() -> None:
    """The first version seeded every answer; the map spun between two look-alikes."""
    watch = LostWatch()
    watch.proposed(CORNER, now=1.0)
    assert not watch.confirmed
    assert watch.reported_fit(0.72) <= PROVISIONAL_FIT_CAP, "an unconfirmed fix must not look good"
    assert watch.observe(0.2, moving=False, navigating=False, now=1.5), "asked again at once"


def test_only_a_second_search_that_agrees_moves_the_tracker_and_only_once() -> None:
    watch = LostWatch()
    watch.proposed(CORNER, now=1.0)
    assert watch.second_opinion(BASE, now=3.0) == "hold"  # disagrees: nothing moves
    assert watch.second_opinion(CORNER, now=5.0) == "hold"  # back to the corner: still nothing
    assert watch.second_opinion(Pose2D(-13.4, 2.1, -2.2), now=7.0) == "apply"  # agrees: one move
    assert watch.confirmed
    assert watch.reported_fit(0.72) == 0.72


def test_the_exact_sequence_of_the_night_moves_exactly_once() -> None:
    """From the board log: corner, base, corner, base. Four seeds then; one now, and at the base."""
    watch = LostWatch()
    answers = [CORNER, BASE, CORNER, BASE]
    moves = []
    watch.proposed(answers[0], now=0.0)
    for k, found in enumerate(answers[1:], start=1):
        if watch.second_opinion(found, now=float(k)) == "apply":
            moves.append(found)
            break
    assert len(moves) <= 1


def test_a_tracker_that_recovers_by_itself_is_never_overridden() -> None:
    """Run 0052: a healthy lock (0.62 and up) is better evidence than a one-scan search."""
    watch = LostWatch()
    watch.proposed(CORNER, now=1.0)
    assert not watch.observe(0.65, moving=False, navigating=False, now=2.0)
    assert watch.confirmed, "the candidate is dropped, the tracker stays"
    assert watch.reported_fit(0.65) == 0.65


def test_searches_that_never_agree_are_given_up_not_looped_forever() -> None:
    watch = LostWatch()
    watch.proposed(Pose2D(0, 0, 0), now=0.0)
    verdicts = []
    for k in range(1, 10):  # every answer somewhere else: a scan that fits two places alike
        verdicts.append(watch.second_opinion(Pose2D(5.0 * k, 0, 0), now=float(k)))
        if verdicts[-1] == "given_up":
            break
    assert verdicts[-1] == "given_up" and len(verdicts) <= 4 and watch.confirmed
    assert "apply" not in verdicts
    assert not watch.observe(0.2, moving=False, navigating=False, now=10.0), "quiet after giving up"


def test_a_hand_s_seed_is_trusted_at_once() -> None:
    watch = LostWatch()
    watch.seeded(now=1.0)
    assert watch.confirmed
    assert not watch.observe(0.9, moving=False, navigating=False, now=2.0)


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
