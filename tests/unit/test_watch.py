"""When the whole map is searched, and when it must not be."""

from pepin.odometry import Pose2D
from pepin.watch import PROVISIONAL_FIT_CAP, LostWatch


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
        watch.searched(found=False, now=1.0)
    assert watch.wait_s() > watch.cooldown_s  # failures have made it patient
    assert watch.observe(0.1, moving=False, navigating=False, now=100.0)  # carried: ask now
    assert watch.wait_s() == watch.cooldown_s  # and start counting failures afresh


def test_each_failed_search_doubles_the_wait_up_to_the_cap() -> None:
    watch = LostWatch(cooldown_s=2.0, max_backoff_s=10.0)
    watch.searched(found=False, now=0.0)
    assert watch.wait_s() == 4.0
    watch.searched(found=False, now=0.0)
    assert watch.wait_s() == 8.0
    watch.searched(found=False, now=0.0)
    assert watch.wait_s() == 10.0  # capped
    watch.searched(found=True, now=0.0)
    assert watch.wait_s() == 2.0  # a fix clears the debt


def test_nothing_is_asked_while_the_quiet_period_runs() -> None:
    watch = LostWatch(cooldown_s=5.0)
    watch.searched(found=False, now=10.0)  # quiet until 20 s (cooldown doubled once)
    for t in (10.5, 12.0, 19.9):
        assert not watch.observe(0.1, moving=False, navigating=False, now=t)
    # The quiet period ends; three poor checks are needed again before it asks.
    assert not watch.observe(0.1, moving=False, navigating=False, now=20.1)
    assert not watch.observe(0.1, moving=False, navigating=False, now=21.0)
    assert watch.observe(0.1, moving=False, navigating=False, now=22.0)


def test_a_fix_from_a_search_is_provisional_until_a_second_search_agrees() -> None:
    """One scan is a guess: on 2026-09-09 a scan at the base fitted a corner four metres away
    almost as well, the search chose the corner, and the tracker settled there for good."""
    watch = LostWatch()
    watch.seeded(now=1.0, provisional=Pose2D(-13.5, 2.0, -2.3))
    assert not watch.confirmed
    assert watch.reported_fit(0.72) <= PROVISIONAL_FIT_CAP, "an unconfirmed fix must not look good"
    assert watch.observe(0.72, moving=False, navigating=False, now=1.5), "asked again at once"
    assert watch.second_opinion(Pose2D(-9.5, 2.3, 0.9), now=3.0) == "replaced"
    assert not watch.confirmed
    assert watch.second_opinion(Pose2D(-9.6, 2.4, 0.8), now=5.0) == "confirmed"
    assert watch.confirmed
    assert watch.reported_fit(0.72) == 0.72


def test_a_hand_s_seed_is_trusted_at_once() -> None:
    watch = LostWatch()
    watch.seeded(now=1.0)
    assert watch.confirmed
    assert not watch.observe(0.9, moving=False, navigating=False, now=2.0)


def test_searches_that_never_agree_are_given_up_not_looped_forever() -> None:
    watch = LostWatch()
    watch.seeded(now=0.0, provisional=Pose2D(0, 0, 0))
    verdicts = []
    for k in range(1, 10):  # every answer somewhere else: a scan that fits two places alike
        verdicts.append(watch.second_opinion(Pose2D(5.0 * k, 0, 0), now=float(k)))
        if verdicts[-1] == "given_up":
            break
    assert verdicts[-1] == "given_up" and len(verdicts) <= 4 and watch.confirmed
    assert not watch.observe(0.7, moving=False, navigating=False, now=10.0), "quiet after giving up"
