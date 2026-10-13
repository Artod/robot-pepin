"""The approach tool against fakes of its owners (pepin.tools.fakes): what it asks the camera
channel's sightings, the eyes and the goal server, the ONE goal it sends, and its words when it
cannot; and the speaker as the camera channel's records and the speaker gaze give it."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import pytest

from pepin.expression.speaker import Attention, AttentionPhase
from pepin.expression.tracker import SpeakerSettings
from pepin.perception.fakes import FakePerception
from pepin.perception.record import RAW_LEFT, RECTIFIED_LEFT, Face, Person, Record
from pepin.perception.source import SNAPSHOT, VisionSource
from pepin.pixel_to_map import Box
from pepin.stand_pose import RELATIONS
from pepin.tools import TOOLS, Result, Robot, ToolError
from pepin.tools.clients import (
    MATCH_DEG,
    ChannelSightings,
    HeadPose,
    PersonSighting,
    Seen,
    ServiceDownError,
)
from pepin.tools.fakes import SPEAKER as SEEN_SPEAKER
from pepin.tools.fakes import (
    FakeCamera,
    FakeEyes,
    FakeGoalServer,
    FakeNeck,
    FakePeople,
    FakeSightings,
    fake_robot,
)

REPO = Path(__file__).resolve().parents[2]


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
    # the channel's face box, measured in the rectified pixels it was found in
    assert eyes(robot).asked == [(Box(380.0, 210.0, 420.0, 258.0), RECTIFIED_LEFT)]
    assert isinstance(robot.sightings, FakeSightings)
    assert robot.sightings.heads == [HeadPose(0.0, 23.8)], "asked with the still head's pose"
    (goal,) = sent(robot)
    assert (goal["x"], goal["y"]) == pytest.approx(
        (1.71, 0.05), abs=0.01
    )  # 0.39 m short of it: the floor + 2 sigma
    assert goal["yaw_deg"] == pytest.approx(0.0, abs=0.5)
    assert len(planned(robot)) == 1, "asked before it drove, once"
    assert result["found"]["kind"] == "person"
    assert (result["found"]["seen_by"], result["found"]["frame"]) == ("face", 7)
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
    assert (goal["x"], goal["y"]) == pytest.approx((2.1, 0.05 + 0.5), abs=0.01)  # side_m 0.5
    assert goal["yaw_deg"] == pytest.approx(180.0, abs=0.5)
    assert result["found"]["facing_deg"] == 180


def test_the_head_s_turn_turns_the_person_s_right() -> None:
    turned = dataclasses.replace(SEEN_SPEAKER, yaw_deg=90.0)  # to their own left
    robot = fake_robot(sightings=FakeSightings(turned))
    call(robot, target="me", relation="right_of")
    (goal,) = sent(robot)
    # Facing south now (west turned 90 deg to their left): their right is west, toward the robot.
    assert (goal["x"], goal["y"]) == pytest.approx((2.1 - 0.5, 0.05), abs=0.01)  # side_m 0.5


def test_nobody_in_view_asks_for_them_and_nothing_moves() -> None:
    robot = fake_robot(sightings=FakeSightings(None))
    result = call(robot, target="me")
    assert not result["ok"]
    assert "nobody in view" in result["why"]
    assert not sent(robot) and not planned(robot)


def test_a_camera_channel_that_does_not_answer_is_said_so() -> None:
    robot = fake_robot(sightings=FakeSightings(down=True))
    result = call(robot, target="me")
    assert "perception service" in result["why"]
    assert not sent(robot)


def test_a_head_that_moved_while_the_person_was_measured_is_refused() -> None:
    """The box is the frame's and the depth is the pair grabbed after it: a head that turned
    in between (the gaze following someone) makes them disagree, so nothing is driven."""

    class Turning(FakeSightings):
        def speaker(self, head: HeadPose) -> PersonSighting | None:
            assert isinstance(robot.neck, FakeNeck)
            robot.neck.pose_now = HeadPose(head.pan_deg + 8.0, head.tilt_deg)
            return super().speaker(head)

    robot = fake_robot(sightings=Turning())
    result = call(robot, target="me")
    assert "the head moved 8 deg while the person was being measured" in result["why"]
    assert not sent(robot) and not planned(robot)


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
    assert eyes(robot).asked == [(Box(412, 300, 470, 352), RAW_LEFT)], "a snapshot's raw eye"
    (goal,) = sent(robot)
    stand = (goal["x"], goal["y"])
    assert math.dist(stand, (1.8, -1.0)) == pytest.approx(0.3, abs=0.01)  # front_m 0.3
    facing = math.degrees(math.atan2(-1.0 - stand[1], 1.8 - stand[0]))
    assert goal["yaw_deg"] == pytest.approx(facing, abs=0.5)
    assert result["found"]["kind"] == "thing"
    assert result["look"]["looked"] is True
    assert robot.seen is None, "the drive made the picture stale"


def test_a_box_on_the_channel_s_picture_is_placed_in_its_rectified_pixels() -> None:
    """see on the camera channel shows the rectified left eye: it records the head pose, that
    pixel space and the frame's number, and the box is measured in that space as drawn."""
    channel = FakeCamera(frame={"seq": 41, "still": True, "optics": RECTIFIED_LEFT})
    robot = fake_robot(camera=channel)
    shown = TOOLS.call("see", {}, robot)
    assert shown["frame"]["optics"] == RECTIFIED_LEFT
    assert robot.seen == Seen(HeadPose(0.0, 23.8), RECTIFIED_LEFT, 41)
    assert call(robot, target="412,300,470,352")["ok"]
    assert eyes(robot).asked == [(Box(412, 300, 470, 352), RECTIFIED_LEFT)]


def test_a_picture_taken_while_the_head_still_moved_carries_no_box() -> None:
    """A frame the channel marks not still was exposed while the head moved: no box drawn on
    it is placed, and see() is asked again."""
    moving = FakeCamera(frame={"seq": 42, "still": False, "optics": RECTIFIED_LEFT})
    robot = fake_robot(camera=moving)
    TOOLS.call("see", {}, robot)
    assert robot.seen is None
    assert "call see()" in call(robot, target="412,300,470,352")["why"]
    assert not eyes(robot).asked and not sent(robot)


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
    assert (goal["x"], goal["y"]) == pytest.approx((-0.7, 3.4))  # beside: side_m 0.5


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
    assert math.dist((goal["x"], goal["y"]), (2.1, 0.05)) == pytest.approx(
        0.39, abs=0.01
    )  # the floor + 2 sigma


def test_nowhere_to_stand_is_said_with_every_refusal_and_nothing_moves() -> None:
    robot = fake_robot()
    goals(robot).unplannable = 3
    result = call(robot, target="me")
    assert not result["ok"]
    assert "nowhere to stand to the person" in result["why"]
    assert result["why"].count("goal occupied") == 3
    assert not sent(robot)


def test_a_goal_server_without_the_plan_check_is_told_to_restart() -> None:
    robot = fake_robot()
    goals(robot).plan_error = "unknown command 'plan'"
    result = call(robot, target="me")
    assert "ros/laptop.sh kick goal_server" in result["why"]
    assert not sent(robot)


def test_a_spoken_distance_is_kept_but_never_nearer_than_the_minimum() -> None:
    robot = fake_robot()
    call(robot, target="me", distance_m=0.3)
    (goal,) = sent(robot)
    # the 0.25 m floor plus two sigma (0.07 each) of the face's depth: 0.39 m short of the face.
    assert goal["x"] == pytest.approx(2.1 - 0.39, abs=0.01)
    assert "between 0.2 and 3" in call(fake_robot(), target="me", distance_m=5.0)["why"]


def test_come_closer_when_already_near_and_facing_does_not_move() -> None:
    """The person 0.40 m ahead (within the 0.3 m stand distance plus near_m), the robot facing
    them: nothing is planned or driven, and the answer says it is there."""
    robot = fake_robot()
    eyes(robot).point = (0.5, 0.05, 1.55)
    result = call(robot, target="me")
    assert result["ok"] and result["moved"] is False and result["arrived"] is True
    assert result["stand"]["choice"].startswith("1 of 1: already 0.40 m from it")
    assert not sent(robot) and not planned(robot)


def test_turn_to_me_is_a_turn_where_the_robot_stands() -> None:
    """The person to the robot's left: one plan and one goal at the robot's own spot, turned
    toward them (the goal server's go turns in place)."""
    robot = fake_robot()
    eyes(robot).point = (0.1, 1.6, 1.55)
    result = call(robot, target="me", relation="face")
    assert result["ok"], result
    (goal,) = sent(robot)
    assert (goal["x"], goal["y"], goal["yaw_deg"]) == pytest.approx((0.1, 0.05, 90.0))
    assert len(planned(robot)) == 1
    near = fake_robot()
    eyes(near).point = (0.35, 0.325, 1.55)  # 0.37 m off at 48 deg: "come closer" only turns
    assert call(near, target="me")["stand"]["heading_deg"] == 48
    assert [(g["x"], g["y"]) for g in sent(near)] == [(0.1, 0.05)]


def test_recognition_s_door_and_the_sightings_are_two_parts_of_the_robot() -> None:
    """Robot.people names and forgets people; Robot.sightings says who stands in front: the
    approach reads the second and never touches the first."""
    robot = fake_robot()
    assert isinstance(robot.people, FakePeople) and isinstance(robot.sightings, FakeSightings)
    assert call(robot, target="me")["ok"]
    assert robot.people.names == [] and len(robot.sightings.heads) == 1
    assert TOOLS.call("remember_person", {"name": "Саша"}, robot)["ok"]
    assert robot.people.names == ["Саша"] and len(robot.sightings.heads) == 1


def test_approach_is_a_drive_of_the_registry() -> None:
    assert TOOLS["approach"].moves
    assert [p.name for p in TOOLS["approach"].params] == ["target", "relation", "distance_m"]
    relation = TOOLS["approach"].input_schema()["properties"]["relation"]
    assert relation["enum"] == list(RELATIONS) and "face" in relation["enum"]


def test_the_brain_is_told_approach_and_never_go_to_pose_for_a_person() -> None:
    """The morning of 2026-10-13: "подъедь ко мне поближе" became go_to_pose 0.7 m toward
    Artem, a goal inside his footprint, and a minute of backing up and spinning. The tool's
    description, go_to_pose's and the persona each say which tool a person's request is."""
    said = " ".join(TOOLS["approach"].description.split())
    for request in (
        "come closer",
        "stand beside me",
        "turn to me",
        "go to the printer and face it",
    ):
        assert request in said, request
    assert "Never use go_to_pose to reach or face a person" in said
    pose = " ".join(TOOLS["go_to_pose"].description.split())
    assert "Never for a person" in pose and "call approach" in pose
    persona = (REPO / "config" / "voice_cascade_persona.md").read_text()
    (line,) = [row for row in persona.splitlines() if "call approach" in row]
    assert "подъедь поближе" in line and "never call go_to_pose toward a person" in line


# -- the speaker as the camera channel's records and the speaker gaze give it --------------------

STILL_FROM = 500.0  # the board's clock: the head stood still from here on
HEAD = HeadPose(10.0, 20.0, still_from=STILL_FROM)  # the head 10 deg left of the nose


def face(pan_deg: float, width: float, yaw: float | None = 0.0) -> Face:
    """A face of a record, ``pan_deg`` left of the optical axis, ``width`` px wide."""
    u = 400.0 - pan_deg * 8.0
    return Face((u - width / 2, 200.0, u + width / 2, 200.0 + width), 0.8, (), pan_deg, 3.0,
                yaw, 50.0, 90.0)  # fmt: skip


def body(pan_deg: float, height_deg: float) -> Person:
    """A body box of a record, ``pan_deg`` left of the optical axis."""
    u = 400.0 - pan_deg * 8.0
    return Person((u - 40, 120.0, u + 40, 560.0), 0.7, pan_deg, 6.0, pan_deg, -8.0, height_deg)


def record(*found: Face | Person, optics: str = RECTIFIED_LEFT, stamp: float = 0.1) -> Record:
    """A record of frame 12, exposed ``stamp`` after the head stood still."""
    faces = tuple(f for f in found if isinstance(f, Face))
    persons = tuple(p for p in found if isinstance(p, Person))
    return Record(12, STILL_FROM + stamp, 0.0, optics=optics, faces=faces, persons=persons)


class Gaze:
    """The speaker gaze's attention: the person engaged at ``bearing`` (None: nobody)."""

    def __init__(self, bearing: float | None, phase: AttentionPhase = "track") -> None:
        self.view = Attention(phase=phase, bearing_deg=bearing)

    def attention(self) -> Attention:
        return self.view


def sightings(*records: Record, gaze: Gaze | None = None, down: bool = False) -> ChannelSightings:
    return ChannelSightings(FakePerception(list(records), down=down), None, gaze)


def test_the_speaker_is_the_person_the_gaze_holds_not_the_nearest_face() -> None:
    """Two faces: the wide near one at +15 deg of the cart, the one the gaze holds at -20 (a
    face 30 deg right of a head turned 10 left); the gaze's is taken, in the frame's pixels."""
    near, held = face(5.0, 60.0), face(-30.0, 30.0, yaw=-12.0)
    found = sightings(record(near, held), gaze=Gaze(-18.0)).speaker(HEAD)
    assert found == PersonSighting(held.box, RECTIFIED_LEFT, "face", -20.0, -12.0, 2, 12,
                                   "the person the gaze holds")  # fmt: skip


def test_the_frame_is_one_exposed_since_the_head_stood_still() -> None:
    perception = FakePerception([record(face(0.0, 40.0))])
    ChannelSightings(perception, None, None, wait_s=0.4).speaker(HEAD)
    assert perception.calls == [("latest", {"since": 0, "min_stamp": STILL_FROM,
                                            "timeout_s": 0.4})]  # fmt: skip
    stale = FakePerception([record(face(0.0, 40.0), stamp=-1.0)])
    with pytest.raises(ToolError, match="no frame taken since the head stopped"):
        ChannelSightings(stale).speaker(HEAD)


def test_a_person_turned_away_is_their_body_box() -> None:
    found = sightings(record(body(-25.0, 40.0)), gaze=Gaze(-14.0)).speaker(HEAD)
    assert found is not None and (found.kind, found.bearing_deg) == ("body", -15.0)
    assert found.yaw_deg is None


def test_without_a_gaze_the_widest_face_then_the_tallest_body() -> None:
    near, far = face(-10.0, 55.0), face(20.0, 30.0)
    found = sightings(record(far, near)).speaker(HEAD)
    assert found is not None and found.box == near.box and found.chosen.startswith("the widest")
    found = sightings(record(body(0.0, 30.0), body(15.0, 45.0))).speaker(HEAD)
    assert found is not None and found.bearing_deg == 25.0 and found.chosen.startswith("the tall")
    idle = sightings(record(near, far), gaze=Gaze(None, phase="idle")).speaker(HEAD)
    assert idle is not None and idle.box == near.box
    assert sightings(record()).speaker(HEAD) is None, "nobody in the frame"


def test_the_gaze_s_person_out_of_the_frame_is_refused_never_swapped() -> None:
    with pytest.raises(ToolError, match=r"at \+60 deg and the newest picture shows nobody there"):
        sightings(record(face(0.0, 50.0)), gaze=Gaze(60.0)).speaker(HEAD)
    assert SpeakerSettings.load().match_deg == MATCH_DEG, "the gaze's own rule"


def test_the_channel_down_snapshot_or_uncalibrated_is_said() -> None:
    with pytest.raises(ServiceDownError, match="perception service"):
        sightings(record(face(0.0, 40.0)), down=True).speaker(HEAD)
    snapshot = ChannelSightings(FakePerception([record(face(0.0, 40.0))]), VisionSource(SNAPSHOT))
    with pytest.raises(ToolError, match=r"vision\.source is snapshot"):
        snapshot.speaker(HEAD)
    with pytest.raises(ToolError, match="raw-left pixels, not the rectified left eye"):
        sightings(record(face(0.0, 40.0), optics=RAW_LEFT)).speaker(HEAD)
