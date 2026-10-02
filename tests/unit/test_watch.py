"""May a goal start: the age of map -> base_link, and whether RTAB-Map's start is placed."""

import json

from pepin.watch import (
    BY_TF,
    PLACEMENT_TOPIC,
    TF_FRESH_S,
    GoalGate,
    Placement,
    Preflight,
    Readiness,
)


def test_a_fresh_transform_is_what_lets_a_goal_start() -> None:
    """The evidence is the pose's own edge — map -> base_link, RTAB-Map's correction composed
    with the board's odometry — and a refusal says whether it is missing or how stale it is."""
    gate = GoalGate()
    assert gate.verdict(0.08) == Readiness(True, rule=BY_TF)
    assert gate.verdict(TF_FRESH_S) == Readiness(True, rule=BY_TF), "the bound is allowed"
    stale = gate.verdict(4.2)
    assert not stale.ready and stale.rule == BY_TF
    assert "map -> base_link is 4.2 s old" in stale.reason and "fresher than 1.0 s" in stale.reason
    missing = gate.verdict(None)
    assert not missing.ready
    assert "nothing publishes map -> base_link" in missing.reason, "which of the two it was"


# ---- where RTAB-Map owns the frame: has this start been placed ---------------------------------
def test_rtabmap_s_saved_start_pose_is_refused_with_what_to_do() -> None:
    """2026-09-23, twice: after a restart RTAB-Map's map -> odom is the pose it saved at its last
    shutdown — "at home" at the bookshelf with 0 of 198 updates recognised, then 76 cm off inside
    the table — and a fresh transform passed the preflight. Not placed is refused, and the
    refusal says both cures: a seed, or the camera seeing a mapped place."""
    saved = Placement(updates=198, recognised=0, seeds=0, loaded=True)
    assert not saved.placed
    check = Preflight.placement(saved)
    assert not check.ok and check.name == "placed"
    assert "0 of 198 updates" in check.detail
    assert "ros/goto.sh seed X Y YAW" in check.detail, "the one-command cure"
    assert "mapped place" in check.detail, "and the other one"
    assert "start_needs_placement false" in check.detail, "and the switch back (rule 19)"
    assert check.line().startswith("preflight placed    REFUSED")


def test_a_recognition_or_a_seed_places_the_start_and_so_does_an_empty_database() -> None:
    assert Preflight.placement(Placement(12, 1, 0, loaded=True)).ok
    assert Preflight.placement(Placement(12, 0, 1, loaded=True)).ok, "an operator's seed"
    empty = Preflight.placement(Placement(3, 0, 0, loaded=False))
    assert empty.ok and "empty database" in empty.detail
    not_yet = Preflight.placement(Placement(0, 0, 0, loaded=None))
    assert not not_yet.ok and "not made one update" in not_yet.detail


def test_nothing_heard_is_refused_and_the_flag_off_is_the_old_behaviour() -> None:
    """No message is nobody saying, never "placed"; the flag off takes any pose as it is."""
    silent = Preflight.placement(None)
    assert not silent.ok and PLACEMENT_TOPIC in silent.detail and "pepin-vslam" in silent.detail
    off = Placement(198, 0, 0, loaded=True, required=False)
    assert off.placed and Preflight.placement(off).ok
    assert "start_needs_placement is off" in off.how()


def test_the_goal_server_s_switch_off_passes_even_silence_and_says_so() -> None:
    """The board-side switch (goal_server start_needs_placement): off, nothing heard and not
    placed both pass — the one way back when the laptop's node cannot say anything — and the
    line says it was not asked."""
    for heard in (None, Placement(198, 0, 0, loaded=True)):
        check = Preflight.placement(heard, asked=False)
        assert check.ok and "goal_server start_needs_placement is off" in check.detail
    assert "0 of 198 updates" in Preflight.placement(Placement(198, 0, 0, True), asked=False).detail


def test_the_placement_travels_as_json_and_a_garbled_one_is_nobody_saying() -> None:
    for sent in (
        Placement(198, 0, 0, True),
        Placement(5, 2, 1, True, required=False),
        Placement(0, 0, 0, None),
        Placement(1, 0, 0, False),
    ):
        text = sent.to_json(1790212953.5)
        assert Placement.from_json(text) == sent
        assert json.loads(text)["placed"] is sent.placed
    for garbled in ("", "{", "[]", '{"updates": "x"}', '{"placed": true}'):
        assert Placement.from_json(garbled) is None
