"""When the graph's database may LEARN and when it may only RECOGNISE (pepin.graphmode), and the
seating test that decides it: what a pose must be worth to teach from at all."""

import math

from pepin.graphmode import (
    ALWAYS_LOCALISE,
    ALWAYS_MAP,
    LOCALISING,
    MAPPING,
    ModeRule,
    describe_sigma,
    seating_refusal,
)


# ---- the seating test ------------------------------------------------------------------------
def test_a_seating_the_scan_pins_in_one_axis_only_is_no_place_to_teach_from() -> None:
    """Whatever the database is taught is baked into every word it says afterwards. A scan sliding
    along a sofa reports an honest fit and half a metre of freedom in y, so the gate reads the error
    bar the tracker publishes, not its score — and refuses with a reason a person can read."""
    sharp = (0.008, 0.012, math.radians(0.4))
    assert seating_refusal(sharp) is None

    along_a_sofa = seating_refusal((0.008, 0.40, math.radians(0.4)))
    assert along_a_sofa is not None and "soft" in along_a_sofa and "40.0 cm" in along_a_sofa

    free_to_turn = seating_refusal((0.008, 0.012, math.radians(4.0)))
    assert free_to_turn is not None and "heading" in free_to_turn

    assert seating_refusal(None) is not None, "no covariance is not a sharp seating"
    assert seating_refusal((0.008, 0.40, math.radians(4.0)), 1.0, 180.0) is None, "wide open"


def test_a_seating_reads_as_centimetres_and_degrees_in_a_report_line() -> None:
    """An operator reads the report line, not the covariance: metres and radians are neither."""
    assert describe_sigma((0.008, 0.012, math.radians(0.23))) == "0.8/1.2 cm, 0.23 deg"
    assert describe_sigma(None) == "unknown"


# ---- the rule --------------------------------------------------------------------------------
def test_the_database_learns_only_from_a_pose_worth_learning_from() -> None:
    """The rule, and it names no sensor. A lidar-held seating at 1-2 cm teaches; a mono camera-only
    pose held by graph words at ~20 cm fails the seating test BY ITSELF; a stereo matcher good to a
    few cm will teach the day it exists, with nothing here changed."""
    rule = ModeRule(hold_s=2.0, graph="graph")
    assert rule.verdict(None, "lidar").mapping is True
    assert rule.verdict(None, "depth").mapping is True, "the rule reads sharpness, not a name"
    soft = rule.verdict("the lidar's seating is soft (20.0/20.0 cm, over 3.0 cm)", "depth")
    assert soft.mapping is False and "not worth learning from" in soft.why
    pupil = rule.verdict(None, "graph")
    assert pupil.mapping is False and "pupil is not the teacher" in pupil.why
    assert rule.verdict(None, None).mapping is False, "nobody holding the pose teaches nothing"
    assert rule.verdict(None, "lidar", "1.2/0.4 cm, 0.30 deg").text().startswith(f"{MAPPING}: ")
    assert rule.verdict(None, "graph").text().startswith(f"{LOCALISING}: ")


def test_the_mode_is_asked_for_once_and_then_only_when_it_has_held() -> None:
    """A verdict is acted on only after it has survived as long as the evidence it rests on takes to
    refresh — the seating's own freshness window — so one missed /tracker_pose cannot flap the mode.
    The very first verdict is the initial mode and is asked for at once."""
    rule = ModeRule(hold_s=2.0)
    first = rule.update(0.0, None, "lidar")
    assert first is not None and first.mapping is True and rule.switches == 1
    assert rule.update(0.5, None, "lidar") is None, "no change, nothing to say"

    assert rule.update(1.0, None, "graph") is None, "the clock starts here"
    assert rule.update(2.5, None, "graph") is None, "1.5 s is not 2 s"
    assert rule.update(1.0, None, "lidar") is None, "...and a blip puts it back with no call"
    assert rule.update(3.0, None, "graph") is None
    switched = rule.update(5.5, None, "graph")
    assert switched is not None and switched.mapping is False and rule.switches == 2
    assert rule.mode == "localising" and "2 switches" in rule.text()


def test_the_mode_can_be_pinned_either_way() -> None:
    """CLAUDE.md rule 19: the old behaviour stays reachable. "map" is the arrangement of before
    2026-09-18; "localise" freezes a database for a session."""
    pinned = ModeRule(hold_s=2.0, override=ALWAYS_MAP)
    verdict = pinned.update(0.0, "the lidar is not driving the tracker (fit 0.00)", "graph")
    assert verdict is not None and verdict.mapping is True
    assert pinned.update(100.0, "still soft", "graph") is None, "pinned means pinned"

    frozen = ModeRule(hold_s=2.0, override=ALWAYS_LOCALISE)
    told = frozen.update(0.0, None, "lidar")
    assert told is not None and told.mapping is False
