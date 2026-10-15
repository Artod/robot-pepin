"""The turn tool against fakes of its owners (pepin.tools.fakes): the angle it resolves for the
person the speaker gaze keeps, a side, a number or a place; the ONE spin it asks the goal server
for; and its words when it cannot."""

from __future__ import annotations

import math
from typing import Any

import pytest

from pepin.expression.speaker import Attention
from pepin.tools import TOOLS, Result, Robot
from pepin.tools.fakes import FakeClock, FakeGaze, FakeGoalServer, fake_robot


def call(robot: Robot, /, **arguments: Any) -> Result:
    return TOOLS.call("turn", arguments, robot)


def goals(robot: Robot) -> FakeGoalServer:
    assert isinstance(robot.goals, FakeGoalServer)
    return robot.goals


def spins(robot: Robot) -> list[float]:
    """The degrees of every spin asked of the goal server."""
    return [float(a["deg"]) for a in goals(robot).asked if a.get("cmd") == "spin"]


def with_gaze(view: Attention, **kw: Any) -> tuple[Robot, FakeGaze]:
    """A fake robot whose speaker gaze holds ``view``."""
    clock = FakeClock()
    gaze = FakeGaze(clock, view=view, **kw)
    return fake_robot(clock=clock, gaze=gaze), gaze


def test_turn_to_me_spins_by_the_bearing_the_gaze_keeps_and_the_head_stays_on_them() -> None:
    """ "Повернись ко мне": the gaze is asked to keep the person before anything turns, its
    snapshot's bearing (40 deg left, seen just now) is the one spin; the result is the drive's,
    the turn measured, the head kept."""
    robot, gaze = with_gaze(Attention(phase="track", bearing_deg=40.0, seen_s=0.0))
    result = call(robot, target="me")
    assert result["ok"], result
    assert gaze.asked == ["turn me"] and spins(robot) == [40.0]
    assert result["found"] == {"kind": "person", "bearing_deg": 40, "by": "sight"}
    assert (result["turned"], result["asked_deg"], result["turned_deg"]) == (True, 40.0, 40.0)
    assert "arrived" not in result and result["head"] == "kept on the person"
    assert result["position"]["heading_deg"] == 43, "the fake stood at 3 deg"


def test_a_person_nearly_ahead_or_nowhere_turns_nothing_and_says_why() -> None:
    robot, _ = with_gaze(Attention(phase="voice", bearing_deg=-8.0, seen_s=None))
    result = call(robot, target="me")
    assert result["ok"] and result["turned"] is False and spins(robot) == []
    assert result["found"]["by"] == "voice" and "already facing the person" in result["detail"]
    idle, _ = with_gaze(Attention())
    assert "no conversation" in call(idle, target="me")["why"]
    refused, _ = with_gaze(Attention(phase="track", bearing_deg=40.0), refuse="the gaze is off")
    assert "the gaze is off" in call(refused, target="me")["why"]
    nobody, _ = with_gaze(Attention(phase="seek"))
    assert "nobody engaged" in call(nobody, target="me")["why"]
    alone = fake_robot()  # no speaker gaze runs (the MCP server, a chat loop)
    assert "no speaker gaze runs here" in call(alone, target="me")["why"]
    assert [spins(r) for r in (idle, refused, nobody, alone)] == [[], [], [], []]


def test_left_right_and_an_angle_spin_as_said_and_the_head_is_kept_for_them_too() -> None:
    robot, gaze = with_gaze(Attention(phase="track", bearing_deg=10.0, seen_s=0.0))
    assert call(robot, target="left")["ok"]
    assert call(robot, target="Right", degrees=-45)["ok"]
    assert call(robot, target="-30")["ok"]
    assert call(robot, target="180")["ok"]
    assert spins(robot) == [90.0, -45.0, -30.0, 180.0]
    assert gaze.asked == ["turn left", "turn Right", "turn -30", "turn 180"]
    for wrong in ({"target": "400"}, {"target": "0.5"}, {"target": "nan"}):
        assert "1 to 360 degrees" in call(robot, **wrong)["why"], wrong
    assert "degrees goes with left or right" in call(robot, target="me", degrees=30)["why"]
    assert "none of" in call(robot, target="the moon")["why"]
    assert len(spins(robot)) == 4, "nothing more was asked"


def test_a_place_is_turned_to_by_its_bearing_and_standing_there_by_its_own_heading() -> None:
    """The robot at (0.1, 0.05) facing 3 deg; the printer at (-1.2, 3.4): its bearing 111 deg,
    a turn of 108 left. Standing on the printer's mark (facing 0), its own heading: 90."""
    robot = fake_robot()
    result = call(robot, target="Printer")
    assert result["ok"], result
    bearing = math.degrees(math.atan2(3.4 - 0.05, -1.2 - 0.1))
    (deg,) = spins(robot)
    assert deg == pytest.approx(bearing - 3.0, abs=0.1)
    assert result["found"]["name"] == "printer" and result["found"]["distance_m"] == 3.59
    goals(robot).pose = {"x": -1.15, "y": 3.3, "yaw_deg": 0.0, "age_s": 0.05}
    assert call(robot, target="printer")["ok"]
    assert spins(robot)[-1] == pytest.approx(90.0)


def test_the_goal_server_s_refusal_is_said_and_an_old_one_is_named_for_a_restart() -> None:
    robot = fake_robot()
    goals(robot).refuse = "spin requests are off (flag spin_requests of the goal server)"
    result = call(robot, target="left")
    assert not result["ok"] and "spin_requests" in result["why"]
    goals(robot).refuse = "unknown command 'spin'"
    result = call(robot, target="left")
    assert not result["ok"] and "restart it" in result["why"]


def test_a_turn_is_a_person_s_ask_and_ends_at_the_goal_server_s_arrival() -> None:
    """The spin goes with the source ``tool`` (the goal server lets it cancel the speaker
    gaze's own turn, never the other way round); the call ends at ``arrival``, Nav2's word,
    not at ``done`` after the tape's close; the run it replaced is named unless it was the
    gaze's own turn."""
    robot, _ = with_gaze(Attention(phase="track", bearing_deg=40.0, seen_s=0.0))
    goals(robot).drive = [
        {"event": "replaced", "run": 40, "source": "tool", "target": "go printer",
         "cancelled_run": 40, "started": True},
        {"event": "replaced", "run": 39, "source": "gaze", "target": "spin +100 deg",
         "cancelled_run": 39, "started": True},
        {"event": "accepted", "run": 41, "deg": 40.0, "pose": "tf", "x": 0.1, "y": 0.05,
         "yaw_deg": 3.0},
        {"event": "arrival", "run": 41, "status": 4, "seconds": 1.5, "turned_deg": 39.0,
         "short_deg": 1.0, "arrival": {"x": 0.1, "y": 0.05, "yaw_deg": 42.0}},
        {"event": "done", "run": 41, "status": 6, "seconds": 9.0},
    ]  # fmt: skip
    result = call(robot, target="me")
    (asked,) = [a for a in goals(robot).asked if a.get("cmd") == "spin"]
    assert asked == {"cmd": "spin", "deg": 40.0, "source": "tool"}
    assert result["ok"] and result["turned"] is True and result["seconds"] == 1.5
    assert result["replaced"] == "go printer (run 40) cancelled for this one"


def test_the_same_turn_already_under_way_is_said_so_and_claims_no_turn() -> None:
    robot, _ = with_gaze(Attention(phase="track", bearing_deg=40.0, seen_s=0.0))
    detail = "already under way: spin +40 deg (run 41)"
    goals(robot).drive = [{"event": "already", "run": 41, "source": "tool", "detail": detail}]
    result = call(robot, target="me")
    assert result["ok"] and result["already"] is True and result["run"] == 41
    assert result["detail"] == f"{detail}, no new goal" and "turned" not in result
