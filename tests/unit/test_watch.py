"""When the whole map is searched, and when it must not be."""

from pepin.watch import LostWatch


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
