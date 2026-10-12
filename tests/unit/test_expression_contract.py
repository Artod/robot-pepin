"""The expression layer's shared contract: the grammar's patterns, the stubs' refusals, the fakes'
answers, and the config files the two features read."""

import json
from pathlib import Path

from pepin.expression.animation import Clip
from pepin.expression.conversation import GESTURES, SPEAKER_GAZE
from pepin.expression.executors import NO_BODY, NoBody, NoWheels, WheelMove
from pepin.expression.fakes import FakeExecutor, FakeFlag, FakeGazeDoor, FakePlayout
from pepin.expression.markers import CLAUSE_END, MARKER, NAME, SPELLED
from pepin.expression.trajectory import JOINTS
from pepin.gaze import Aim

CONFIG = Path(__file__).resolve().parents[2] / "config"


def test_a_marker_is_one_pair_of_braces_anywhere_in_the_text() -> None:
    text = "Ну {laugh} да, {style: whispering} все уже спят. {} {face:smile}"  # noqa: RUF001
    assert MARKER.findall(text) == ["laugh", "style: whispering", "", "face:smile"]
    assert MARKER.findall("{a{b}") == ["b"]  # no nesting: the innermost pair is the marker


def test_a_spelled_out_marker_names_its_channel() -> None:
    match = SPELLED.match(" Style : whispering, slow ")
    assert match is not None and match.groups() == ("Style", "whispering, slow")
    assert SPELLED.match("happy, excited") is None  # a bare style of the older grammar
    assert NAME.match("laugh") and not NAME.match("шёпотом") and not NAME.match("Laugh")


def test_a_style_cuts_a_sentence_only_at_a_clause_boundary() -> None:
    assert CLAUSE_END.search("")
    assert CLAUSE_END.search("Знаешь, ")
    assert CLAUSE_END.search("Знаешь — ")
    assert not CLAUSE_END.search("Знаешь ")


def test_the_wheels_stubs_refuse_with_the_reason() -> None:
    body = NoBody()
    assert body.ready() == NO_BODY
    assert not body.play(Clip("body", 0.0, 0.0, 1.0)).ok
    report = NoWheels().move(WheelMove(turn_deg=20.0))
    assert not report.done and report.why == NO_BODY and report.turned_deg == 0.0


def test_the_fakes_answer_as_scripted() -> None:
    playout = FakePlayout(now=1.5, marks={(3, 0): (1.0, 2.0)})
    assert playout.played() == 1.5 and playout.mark(3, 0) == (1.0, 2.0)
    assert playout.mark(3, 1) is None
    part = FakeExecutor("arm", why_not="driving")
    assert not part.play(Clip("arm", 0.0, 0.2, 1.0)).ok and part.calls[0][0] == "play"
    flag = FakeFlag(GESTURES)
    assert flag.switch(True) is None and flag.on
    door = FakeGazeDoor()
    aim = Aim(0.3, 0.1)
    assert door.aim(aim, speed="saccade", ttl_s=6.0).head == aim


def test_the_joints_are_the_self_filters_joints() -> None:
    arm = json.loads((CONFIG / "arm.json").read_text())
    assert tuple(arm["joints"]["pose_deg"]) == JOINTS


def test_both_features_ship_off_and_the_doa_measured_at_four_places() -> None:
    config = json.loads((CONFIG / "expression.json").read_text())
    assert config["flags"][GESTURES] is False and config["flags"][SPEAKER_GAZE] is False
    doa = config["speaker"]["doa"]
    assert set(doa["readings"]) == {"front", "left", "behind", "right"}
    assert doa["front_deg"] == doa["readings"]["front"] and doa["measured"].startswith("MEASURED")
