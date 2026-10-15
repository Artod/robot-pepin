"""The body's scene for the brain: coarse buckets with hysteresis (the breath and the wheels'
wobble tell nothing, a new sector or cell does), only the changed fields told, every known one
at the first message of a conversation or a brain, a turn the brain did not keep leaving its
scene to the next, one short line, and the switch."""

from __future__ import annotations

import pytest

from pepin.expression.speaker import Attention, AttentionPhase
from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.scene import (
    Reading,
    SceneKnobs,
    SceneStatus,
    book_of,
    cell,
    pose_of,
    sector,
)

NOW = 100.0
BRAIN = object()
PRINTER = {"printer": (2.0, 0.0)}  # two metres along the map's x axis


def person(
    bearing_deg: float, *, seen_ago_s: float = 0.5, phase: AttentionPhase = "track"
) -> Attention:
    """The speaker gaze's snapshot of the person engaged."""
    return Attention(phase=phase, bearing_deg=bearing_deg, seen_s=NOW - seen_ago_s)


def told(status: SceneStatus, reading: Reading, conversation: str = "c1") -> str:
    """The line of a message whose turn the brain's history kept."""
    report = status.report(reading, conversation, BRAIN)
    status.keep(report)
    return report.line


def test_the_first_message_carries_every_field_known_on_one_short_line() -> None:
    reading = Reading((0.0, 0.0, 0.0), "home", PRINTER, person(40.0), now=NOW)
    line = told(SceneStatus(SceneKnobs()), reading)
    assert line == (
        "[scene] heading 0; position 0.0, 0.0 (at home); ahead: printer;"
        " person: left 40 deg, seen; standing still"
    )
    assert len(line) < 120 and "\n" not in line


def test_the_breath_and_the_wobble_tell_nothing_and_a_new_sector_tells_the_heading() -> None:
    status = SceneStatus(SceneKnobs())  # 15-degree sectors, 5 degrees of hysteresis
    assert told(status, Reading((0.0, 0.0, 95.0))).startswith("[scene] heading 95;")
    for wobble in (95.3, 94.6, 101.0, 102.4):  # the sector 82.5..97.5 kept to 102.5
        assert told(status, Reading((0.0, 0.0, wobble))) == ""
    assert told(status, Reading((0.0, 0.0, 112.0))) == "[scene] heading 112"
    assert told(status, Reading((0.0, 0.0, 108.0))) == ""  # back within the new sector's slack
    assert told(status, Reading((0.0, 0.0, -170.0))) == "[scene] heading -170"


def test_sectors_and_cells_hold_until_past_an_edge_by_the_hysteresis() -> None:
    assert sector(0.0, 15.0, 5.0) == 0 and sector(-7.6, 15.0, 5.0) == 23
    assert sector(12.0, 15.0, 5.0, held=0) == 0 and sector(12.6, 15.0, 5.0, held=0) == 1
    assert sector(-179.0, 15.0, 5.0, held=12) == 12  # across the wrap at 180
    assert cell(1.32, 0.25, 0.1, held=4) == 4 and cell(1.36, 0.25, 0.1, held=4) == 5
    assert cell(0.95, 0.25, 0.1, held=4) == 4 and cell(0.89, 0.25, 0.1, held=4) == 3


def test_a_position_changes_only_a_tenth_of_a_metre_past_a_cell_s_edge() -> None:
    status = SceneStatus(SceneKnobs())  # cells of 0.25 m
    told(status, Reading((1.1, 0.1, 0.0)))
    assert told(status, Reading((1.32, 0.12, 0.0))) == ""
    assert told(status, Reading((1.36, 0.12, 0.0))) == "[scene] position 1.4, 0.1"
    assert told(status, Reading((1.36, 0.12, 0.0), "printer")) == (
        "[scene] position 1.4, 0.1 (at printer)"
    )


def test_only_the_fields_that_changed_are_told() -> None:
    status = SceneStatus(SceneKnobs())
    told(status, Reading((0.0, 0.0, 180.0), attention=person(40.0), now=NOW))
    moved = Reading((0.0, 0.0, 180.0), attention=person(-30.0), now=NOW)
    assert told(status, moved) == "[scene] person: right 30 deg, seen"
    driving = Reading((0.0, 0.0, 180.0), attention=person(-30.0), driving=True, now=NOW)
    assert told(status, driving) == "[scene] driving"
    assert told(status, driving) == ""


def test_ahead_is_a_named_place_before_the_nose_else_the_person_else_nothing_named() -> None:
    book = {**PRINTER, "door": (3.0, 0.5), "home": (0.05, 0.0)}  # the door 3 m off: too far
    status = SceneStatus(SceneKnobs())
    assert "ahead: printer" in told(status, Reading((0.0, 0.0, 0.0), places=book))
    # turned 22 degrees: the printer told stays ahead within the slack (20 + 5), then leaves
    assert told(status, Reading((0.0, 0.0, -22.0), places=book)) == "[scene] heading -22"
    assert "ahead: nothing named" in told(status, Reading((0.0, 0.0, -26.0), places=book))
    # home underfoot is never ahead; the person standing before the nose is
    facing = Reading((0.0, 0.0, 180.0), places=book, attention=person(5.0), now=NOW)
    assert "ahead: the person" in told(SceneStatus(SceneKnobs()), facing)
    first = SceneStatus(SceneKnobs())
    assert "ahead: printer" in told(first, Reading((0.0, 0.0, 15.0), places=book))  # 15 deg off
    assert "ahead" not in told(first, Reading((0.0, 0.0, 15.3), places=book))


def test_the_body_turning_back_to_the_person_is_told_with_the_next_message() -> None:
    """2026-10-14: after turn(printer) the gaze turned the body back toward the person, and the
    brain said the printer was in front of it."""
    status = SceneStatus(SceneKnobs())
    turned = Reading((0.0, 0.0, 0.0), places=PRINTER, attention=person(90.0), now=NOW)
    assert "ahead: printer" in told(status, turned)
    back = Reading((0.0, 0.0, 90.0), places=PRINTER, attention=person(0.0), now=NOW)
    assert told(status, back) == "[scene] heading 90; ahead: the person; person: ahead, seen"


def test_a_turn_the_brain_did_not_keep_leaves_its_scene_to_the_next() -> None:
    status = SceneStatus(SceneKnobs())
    reading = Reading((0.0, 0.0, 0.0), places=PRINTER)
    dropped = status.report(reading, "c1", BRAIN)  # merged into the turn after it: not kept
    assert dropped.line == status.report(reading, "c1", BRAIN).line != ""
    status.keep(status.report(reading, "c1", BRAIN))
    assert status.report(Reading((0.0, 0.0, 45.0), places=PRINTER), "c1", BRAIN).line == (
        "[scene] heading 45; ahead: nothing named"
    )
    assert status.report(reading, "c1", BRAIN).line == ""  # the 45 was never kept


def test_a_new_conversation_or_a_new_brain_hears_the_whole_scene() -> None:
    status = SceneStatus(SceneKnobs())
    reading = Reading((0.0, 0.0, 0.0), places=PRINTER)
    whole = told(status, reading)
    assert told(status, reading) == ""
    assert told(status, reading, conversation="c2") == whole
    assert status.report(reading, "c2", object()).line == whole  # a brain switched in


def test_the_switch_off_and_on_again_tells_the_whole_scene() -> None:
    status = SceneStatus(SceneKnobs())
    assert status.flag == "scene_status" and status.on
    reading = Reading((0.0, 0.0, 0.0))
    whole = told(status, reading)
    assert status.switch(False) is None and not status.on
    assert status.switch(True) is None and status.on
    assert told(status, reading) == whole
    assert not SceneStatus(SceneKnobs(), on=False).on


def test_the_person_seen_not_seen_behind_and_not_tracked() -> None:
    status = SceneStatus(SceneKnobs())
    assert "person: left 40 deg, seen" in told(status, Reading(attention=person(40.0), now=NOW))
    unseen = Reading(attention=person(40.0, seen_ago_s=5.0), now=NOW)
    assert told(status, unseen) == "[scene] person: left 40 deg, not seen"
    behind = Reading(attention=person(-178.0), now=NOW)
    assert told(status, behind) == "[scene] person: behind, seen"
    paused = Reading(attention=person(40.0, phase="paused"), now=NOW)  # a drive pauses the gaze
    assert told(status, paused) == "[scene] person: not tracked"
    assert "person" not in told(SceneStatus(SceneKnobs()), Reading((0.0, 0.0, 0.0)))  # no gaze


def test_a_lost_pose_is_told_once_and_one_never_known_is_left_out() -> None:
    status = SceneStatus(SceneKnobs())
    assert told(status, Reading()) == "[scene] standing still"
    assert told(status, Reading((0.5, 0.5, 10.0))) == (
        "[scene] heading 10; position 0.5, 0.5; ahead: nothing named"
    )
    assert told(status, Reading()) == "[scene] heading unknown; position unknown; ahead: unknown"
    assert told(status, Reading()) == ""


def test_a_place_s_name_stays_on_one_line() -> None:
    reading = Reading((0.0, 0.0, 0.0), "the\nsofa", {"the\nprinter": (2.0, 0.0)})
    line = told(SceneStatus(SceneKnobs()), reading)
    assert "\n" not in line and "ahead: the printer" in line and "(at the sofa)" in line


def test_where_am_i_s_and_list_places_answers_read_as_the_pose_and_the_book() -> None:
    answer = {"ok": True, "x": 1.23, "y": -2, "heading_deg": 95, "at": "printer"}
    assert pose_of(answer) == ((1.23, -2.0, 95.0), "printer")
    assert pose_of({**answer, "at": None}) == ((1.23, -2.0, 95.0), None)
    assert pose_of({"ok": False, "why": "no pose"}) == (None, None)
    assert pose_of({"ok": True, "places": []}) == (None, None)
    rows = [{"name": "printer", "x": 2.0, "y": 0}, {"name": "door"}, "noise"]
    assert book_of(rows) == {"printer": (2.0, 0.0)} and book_of(None) == {}


def test_the_switch_and_the_knobs_come_from_the_file() -> None:
    config = CascadeConfig.load()
    assert config.scene_status is True
    knobs = (config.scene_heading_sector_deg, config.scene_hysteresis_deg, config.scene_cell_m)
    assert knobs == (15.0, 5.0, 0.25)
    off = CascadeConfig.from_dict({"conversation": {"scene_status": False}})
    assert off.scene_status is False and CascadeConfig.from_dict({}).scene_status is True
    for bad in ({"scene_heading_sector_deg": 7}, {"scene_hysteresis_deg": 15},
                {"scene_cell_m": 0}):  # fmt: skip
        with pytest.raises(ValueError, match="scene"):
            CascadeConfig.from_dict({"conversation": bad})
