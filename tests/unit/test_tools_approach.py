"""The approach tool against fakes of its owners (pepin.tools.fakes): what it asks the face
sensor, the eyes and the goal server, the ONE goal it sends, and its words when it cannot."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pytest

from pepin.pixel_to_map import Box
from pepin.tools import TOOLS, Result, Robot, ToolError
from pepin.tools.clients import FaceSensorSightings, FaceSighting, HeadPose, ServiceDownError
from pepin.tools.fakes import FakeEyes, FakeGoalServer, FakeNeck, FakeSightings, fake_robot


def call(robot: Robot, /, **arguments: Any) -> Result:
    return TOOLS.call("approach", arguments, robot)


def goals(robot: Robot) -> FakeGoalServer:
    assert isinstance(robot.goals, FakeGoalServer)
    return robot.goals


def eyes(robot: Robot) -> FakeEyes:
    assert isinstance(robot.eyes, FakeEyes)
    return robot.eyes


def sent(robot: Robot) -> list[dict[str, Any]]:
    return [a for a in goals(robot).asked if a.get("cmd") == "go"]


def planned(robot: Robot) -> list[dict[str, Any]]:
    return [a for a in goals(robot).asked if a.get("cmd") == "plan"]


def test_come_to_me_stands_on_the_line_to_the_face_and_looks_at_it() -> None:
    robot = fake_robot()  # the robot at (0.1, 0.05), the face 2 m ahead at (2.1, 0.05), 1.55 up
    result = call(robot, target="me")
    assert result["ok"], result
    assert eyes(robot).boxes == [Box(380.0, 210.0, 420.0, 258.0)]  # the face sensor's box
    (goal,) = sent(robot)
    assert (goal["x"], goal["y"]) == pytest.approx((1.3, 0.05), abs=0.01)  # 0.8 m short of it
    assert goal["yaw_deg"] == pytest.approx(0.0, abs=0.5)
    assert len(planned(robot)) == 1, "asked before it drove, once"
    assert result["found"]["kind"] == "person"
    assert result["found"]["who"] == "artem"
    assert result["found"]["depth_source"] == "stereo"
    assert result["stand"]["choice"] == "1 of 3: to"
    assert result["arrived"] is True
    look = result["look"]
    assert look["looked"] is True
    assert look["tilt_deg"] == pytest.approx(-20.0), "the face is above the neck's reach"
    assert "past the neck's reach" in look["short_by"]


def test_right_of_me_is_the_person_s_right_side_by_side() -> None:
    robot = fake_robot()
    result = call(robot, target="me", relation="right_of")
    assert result["ok"], result
    (goal,) = sent(robot)
    # The person faces the robot (west); their right is the map's +y, 0.74 m out: the side's
    # 0.7 m is under the person minimum 0.6 m plus two sigma (0.07) of the face's depth.
    assert (goal["x"], goal["y"]) == pytest.approx((2.1, 0.05 + 0.74), abs=0.01)
    assert goal["yaw_deg"] == pytest.approx(180.0, abs=0.5)
    assert result["found"]["facing_deg"] == 180


def test_the_head_s_turn_turns_the_person_s_right() -> None:
    turned = FaceSighting((380.0, 210.0, 420.0, 258.0), "artem", 90.0, 1, 0.3)  # to their left
    robot = fake_robot(sightings=FakeSightings(turned))
    call(robot, target="me", relation="right_of")
    (goal,) = sent(robot)
    # Facing south now (west turned 90 deg to their left): their right is west, toward the robot.
    assert (goal["x"], goal["y"]) == pytest.approx((2.1 - 0.74, 0.05), abs=0.01)


def test_no_face_in_view_asks_for_one_and_nothing_moves() -> None:
    robot = fake_robot(sightings=FakeSightings(None))
    result = call(robot, target="me")
    assert not result["ok"]
    assert "no face in view" in result["why"]
    assert not sent(robot) and not planned(robot)


def test_a_face_sensor_that_is_not_running_is_said_so() -> None:
    robot = fake_robot(sightings=FakeSightings(down=True))
    result = call(robot, target="me")
    assert "face sensor" in result["why"]
    assert not sent(robot)


def test_no_depth_for_the_face_is_said_and_nothing_moves() -> None:
    robot = fake_robot()
    eyes(robot).blank = "no depth for that box: stereo: no depth in that box"
    result = call(robot, target="me")
    assert "cannot tell where the person is" in result["why"]
    assert not sent(robot)


def test_a_box_of_the_last_picture_is_a_thing_the_robot_faces() -> None:
    robot = fake_robot()
    eyes(robot).point = (1.8, -1.0, 0.75)  # a board on a table, front-right
    TOOLS.call("see", {}, robot)
    result = call(robot, target="412,300,470,352")
    assert result["ok"], result
    assert eyes(robot).boxes == [Box(412, 300, 470, 352)]
    (goal,) = sent(robot)
    stand = (goal["x"], goal["y"])
    assert math.dist(stand, (1.8, -1.0)) == pytest.approx(0.8, abs=0.01)
    facing = math.degrees(math.atan2(-1.0 - stand[1], 1.8 - stand[0]))
    assert goal["yaw_deg"] == pytest.approx(facing, abs=0.5)
    assert result["found"]["kind"] == "thing"
    assert result["look"]["looked"] is True
    assert robot.seen is None, "the drive made the picture stale"


def test_a_box_without_a_picture_or_after_the_head_turned_is_refused() -> None:
    robot = fake_robot()
    result = call(robot, target="412,300,470,352")
    assert "call see()" in result["why"]
    TOOLS.call("see", {}, robot)
    assert isinstance(robot.neck, FakeNeck)
    robot.neck.pose_now = HeadPose(25.0, 23.8)  # the head went elsewhere since
    result = call(robot, target="412,300,470,352")
    assert "the head has turned 25 deg" in result["why"]
    assert not sent(robot) and not eyes(robot).boxes


def test_a_place_is_its_own_pose_and_its_sides_are_its_own() -> None:
    robot = fake_robot()
    result = call(robot, target="Printer")  # the book's names, any case
    assert result["ok"], result
    (goal,) = sent(robot)
    assert (goal["x"], goal["y"], goal["yaw_deg"]) == pytest.approx((-1.2, 3.4, 90.0))
    assert "look" not in result, "a place already faces its thing"
    robot = fake_robot()
    call(robot, target="printer", relation="right_of")
    (goal,) = sent(robot)
    assert (goal["x"], goal["y"]) == pytest.approx((-0.5, 3.4))


def test_an_unknown_target_names_what_it_may_be() -> None:
    result = call(fake_robot(), target="kitchen")
    assert not result["ok"]
    assert (
        "'me'" in result["why"] and "bookshelf" in result["why"] and "x0,y0,x1,y1" in result["why"]
    )


def test_the_planner_refusing_the_first_spot_takes_the_next() -> None:
    robot = fake_robot()
    goals(robot).unplannable = 1
    result = call(robot, target="me")
    assert result["ok"], result
    assert len(planned(robot)) == 2
    assert result["stand"]["choice"].startswith("2 of 3: to turned 30 deg left")
    (goal,) = sent(robot)
    assert math.dist((goal["x"], goal["y"]), (2.1, 0.05)) == pytest.approx(0.8, abs=0.01)


def test_nowhere_to_stand_is_said_with_every_refusal_and_nothing_moves() -> None:
    robot = fake_robot()
    goals(robot).unplannable = 3
    result = call(robot, target="me")
    assert not result["ok"]
    assert "nowhere to stand to artem" in result["why"]
    assert result["why"].count("goal occupied") == 3
    assert not sent(robot)


def test_a_goal_server_without_the_plan_check_is_told_to_restart() -> None:
    robot = fake_robot()
    goals(robot).plan_error = "unknown command 'plan'"
    result = call(robot, target="me")
    assert "restart Nav2" in result["why"]
    assert not sent(robot)


def test_a_spoken_distance_is_kept_but_never_nearer_than_the_minimum() -> None:
    robot = fake_robot()
    call(robot, target="me", distance_m=0.3)
    (goal,) = sent(robot)
    # 0.6 m plus two sigma (0.07) of the face's depth: 0.74 m short of the face.
    assert goal["x"] == pytest.approx(2.1 - 0.74, abs=0.01)
    assert "between 0.2 and 3" in call(fake_robot(), target="me", distance_m=5.0)["why"]


def test_approach_is_a_drive_of_the_registry() -> None:
    assert TOOLS["approach"].moves
    assert [p.name for p in TOOLS["approach"].params] == ["target", "relation", "distance_m"]


def test_the_face_sensor_s_line_as_the_tool_reads_it(tmp_path: Path) -> None:
    line = tmp_path / "latest.json"
    sightings = FaceSensorSightings(line)
    with pytest.raises(ServiceDownError, match="face sensor"):
        sightings.face()  # no sensor has written yet
    now = time.time()
    face = {"stamp": now, "faces": 2, "face_px": 40, "box": [1, 2, 41, 50]}
    line.write_text(json.dumps(face | {"who": {"id": "artem"}, "head_deg": {"yaw": -10.0}}))
    assert sightings.face() == FaceSighting(
        (1.0, 2.0, 41.0, 50.0), "artem", -10.0, 2, pytest.approx(0.0, abs=1.0)
    )
    line.write_text(json.dumps({"stamp": now, "faces": 0, "face_px": None, "box": None}))
    assert sightings.face() is None
    line.write_text(json.dumps(face | {"stamp": now - 10.0}))
    with pytest.raises(ServiceDownError, match="10 s old"):
        sightings.face()
    line.write_text(json.dumps({"stamp": now, "faces": 1, "face_px": 40}))  # an older sensor
    with pytest.raises(ToolError, match="no face box"):
        sightings.face()
