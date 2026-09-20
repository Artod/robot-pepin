"""RTAB-Map's two live switches (pepin.graphmode): when the database may LEARN and when it may only
RECOGNISE — with the seating test that decides it — and which REGISTRATION the snapshots need.

Both rules are pure and both are acted on only once a change has held, so both are tested the same
way: a reading and a clock in, the verdict to act on out."""

import math

from pepin.graphmode import (
    ALWAYS_LOCALISE,
    ALWAYS_MAP,
    LOCALISING,
    MAPPING,
    REGISTRATION_PARAMETERS,
    STRATEGY_ICP,
    STRATEGY_VIS,
    ModeRule,
    StrategyRule,
    describe_sigma,
    registration_verdict,
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


# ---- the registration follows the snapshot ----------------------------------------------------
def test_the_strategy_is_chosen_by_what_the_snapshots_carry() -> None:
    """The whole rule, as a pure function: a scan in the snapshots means ICP, no scan means
    visual. The values are RTAB-Map's own (Parameters.h:677, "0=Vis, 1=Icp, 2=VisIcp")."""
    assert registration_verdict(scan=True).strategy == "1" == STRATEGY_ICP
    assert registration_verdict(scan=False).strategy == "0" == STRATEGY_VIS
    assert registration_verdict(scan=True).name == "ICP on the scans"
    assert registration_verdict(scan=False).name == "visual"
    assert registration_verdict(scan=False, kind="camera-only").why == (
        "the snapshots carry no scan (snapshots camera-only)"
    )


def test_only_reg_strategy_travels_with_the_verdict() -> None:
    """ONE parameter each, and it must be one the launch table already overrode: rtabmap inserts
    only overridden keys into the map update_parameters re-reads (CoreWrapper.cpp:362-379), so a
    companion parameter added here that the table does not carry would be accepted and ignored."""
    # The grid's sensor travels with the strategy: a scan writes the grid while there is one
    # (the mono depth beside it flattened the lidar tracker's match: sigma 0.47 m against 0.01).
    # ...and the neighbour links are refined by the scan while there is one (unrefined, the map
    # turned with the gyro's bias under a parked cart: +27 deg in 40 min).
    assert registration_verdict(scan=False).parameters == {
        "Reg/Strategy": "0",
        "RGBD/NeighborLinkRefining": "false",
    }
    assert registration_verdict(scan=True).parameters == {
        "Reg/Strategy": "1",
        "RGBD/NeighborLinkRefining": "false",
    }
    assert set(REGISTRATION_PARAMETERS) == {"0", "1"}, "VisIcp (2) is never asked for"
    for table in REGISTRATION_PARAMETERS.values():
        assert all(isinstance(value, str) for value in table.values()), (
            "rtabmap declares every parameter as a string and reads it back with as_string()"
        )


def test_the_rule_asks_for_nothing_until_it_has_a_reason_to() -> None:
    """Unlike the memory rule, whose first verdict IS the initial mode: the launch table has
    already set Reg/Strategy, so a rule that agreed with it would rebuild the pipeline for
    nothing."""
    rule = StrategyRule(STRATEGY_ICP)
    assert rule.update(now=0.0, hold_s=1.0, scan=True, kind="full") is None
    assert rule.update(now=5.0, hold_s=1.0, scan=True, kind="full") is None
    assert rule.strategy == STRATEGY_ICP and rule.switches == 0


def test_a_change_is_acted_on_only_once_it_has_held_for_the_evidences_own_refresh() -> None:
    rule = StrategyRule(STRATEGY_ICP)
    rule.update(now=0.0, hold_s=1.0, scan=True, kind="full")
    assert rule.update(now=1.0, hold_s=1.0, scan=False, kind="camera-only") is None, "just now"
    assert rule.update(now=1.5, hold_s=1.0, scan=False, kind="camera-only") is None, "not yet"
    verdict = rule.update(now=2.1, hold_s=1.0, scan=False, kind="camera-only")
    assert verdict is not None and verdict.strategy == STRATEGY_VIS
    assert rule.strategy == STRATEGY_VIS and rule.switches == 1
    assert rule.update(now=3.0, hold_s=1.0, scan=False, kind="camera-only") is None, "once"


def test_a_source_that_stutters_for_one_snapshot_cannot_rebuild_the_pipeline() -> None:
    """The hold exists for exactly this: the lidar missing from one snapshot is not the lidar
    going away, and the registration pipeline is deleted and re-created on every change."""
    rule = StrategyRule(STRATEGY_ICP)
    rule.update(now=0.0, hold_s=1.0, scan=True, kind="full")
    assert rule.update(now=0.5, hold_s=1.0, scan=False, kind="camera-only") is None
    assert rule.update(now=0.9, hold_s=1.0, scan=True, kind="full") is None, "back already"
    assert rule.update(now=2.0, hold_s=1.0, scan=True, kind="full") is None
    assert rule.switches == 0


def test_silence_is_not_evidence_that_the_lidar_is_gone() -> None:
    """No snapshot state at all — the packer has not spoken, or its last word is older than its
    own refresh — leaves the strategy in force: a node that lost its state topic must not stop
    linking scans."""
    rule = StrategyRule(STRATEGY_ICP)
    for now in (0.0, 1.0, 5.0, 60.0):
        assert rule.update(now=now, hold_s=1.0, scan=None) is None
    assert rule.strategy == STRATEGY_ICP and rule.switches == 0
    assert "nothing said about the snapshots" in rule.text()


def test_the_hold_starts_over_when_the_evidence_comes_back() -> None:
    """A gap in the state must not count towards the hold of the change that follows it."""
    rule = StrategyRule(STRATEGY_ICP)
    rule.update(now=0.0, hold_s=1.0, scan=False, kind="camera-only")
    rule.update(now=0.5, hold_s=1.0, scan=None)
    assert rule.update(now=1.2, hold_s=1.0, scan=False, kind="camera-only") is None, (
        "the hold began again when the evidence did"
    )
    assert rule.update(now=2.3, hold_s=1.0, scan=False, kind="camera-only") is not None


def test_the_report_line_says_which_strategy_and_why() -> None:
    rule = StrategyRule(STRATEGY_ICP)
    rule.update(now=0.0, hold_s=1.0, scan=True, kind="full")
    assert (
        rule.text() == "ICP on the scans (the snapshots carry a scan (snapshots full), 0 switches)"
    )
    rule.update(now=1.0, hold_s=1.0, scan=False, kind="camera-only")
    assert "asking visual" in rule.text(), "a verdict waiting out its hold is visible"
