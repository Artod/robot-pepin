"""The approach tool against fakes of its owners (pepin.tools.fakes): what it asks the camera
channel's sightings, the eyes and the goal server, the ONE goal it sends, and its words when it
cannot; and the speaker as the camera channel's records and the speaker gaze give it."""

from __future__ import annotations

import dataclasses
import logging
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
    TF_NOW,
    FakeCamera,
    FakeEyes,
    FakeGaze,
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
        (1.80, 0.05), abs=0.01
    )  # front_m 0.3 short of it: the floor (0.25, sigma_k 0) is under it
    assert goal["yaw_deg"] == pytest.approx(0.0, abs=0.5)
    assert len(planned(robot)) == 1, "asked before it drove, once"
    assert result["found"]["kind"] == "person"
    assert (result["found"]["seen_by"], result["found"]["frame"]) == ("face", 7)
    assert result["found"]["depth_source"] == "stereo"
    assert result["stand"]["choice"] == "1 of 3: to"
    assert result["drive_to"] == {"x": 1.8, "y": 0.05, "heading_deg": 0, "what": "the stand"}
    assert result["short_m"] == 0.0, "the plan ends at the stand"
    assert result["arrived"] is True
    look = result["look"]
    assert look["looked"] is True
    assert look["tilt_deg"] == pytest.approx(-20.0), "the face is above the neck's reach"
    assert "past the neck's reach" in look["short_by"]


def test_a_point_behind_is_looked_at_from_the_end_the_head_is_at_never_across_the_front() -> None:
    """The head at +150 (left), a point a hair behind on the RIGHT (-176 by the wrap): the
    nearest end by number is the right one, 317 deg away through the front; the look stays
    at the left end, short of the point, and says so."""
    from pepin.tools.approach import look_at, pan_within

    robot = fake_robot()
    neck = robot.neck
    assert isinstance(neck, FakeNeck)
    reach = neck.reach()
    assert pan_within(reach, -176.0, 150.0) == reach.pan_left_deg
    assert pan_within(reach, -176.0, -100.0) == reach.pan_right_deg
    assert pan_within(reach, -150.0, 150.0) == -150.0, "inside the reach: as asked"
    neck.pose_now = HeadPose(150.0, 23.8)
    # the fake goal server stands at (0.1, 0.05) facing 3 deg: a point 2 m behind, a hair right
    look = look_at(robot, (-1.9, -0.09, 1.2))
    assert look["looked"] is True and "past the neck's reach" in look["short_by"]
    assert look["pan_deg"] == pytest.approx(reach.pan_left_deg, abs=0.1), look


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


class Answers(FakeSightings):
    """The camera's speaker per ask: nobody first, then whoever ``then`` says."""

    def __init__(self, *answers: PersonSighting | None) -> None:
        super().__init__()
        self.answers = list(answers)

    def speaker(self, head: HeadPose) -> PersonSighting | None:
        self.heads.append(head)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def test_nobody_in_view_turns_the_body_to_the_voice_then_finds_the_person_and_drives() -> None:
    """2026-10-13: "подъедь ко мне" from behind: the camera sees nobody, the voice says -150
    deg. The tool asks the speaker gaze for its body turn (the whole bearing, one request),
    waits for it, reads the picture again with the head still, and drives as if the person had
    been in view from the start; no head sweep anywhere."""
    robot = fake_robot(sightings=Answers(None, SEEN_SPEAKER))
    gaze = FakeGaze(robot.clock, Attention(phase="voice", bearing_deg=-150.0))  # type: ignore[arg-type]
    robot.gaze = gaze
    result = call(robot, target="me")
    assert result["ok"], result
    assert gaze.asked == ["approach: nobody in view, the voice is the lead"]
    assert isinstance(robot.sightings, Answers) and len(robot.sightings.heads) == 2
    assert sent(robot), "the drive went after the turn"
    assert gaze.attention().phase == "track"


def test_nobody_in_view_after_the_turn_says_the_body_turned_and_nothing_else_moved() -> None:
    robot = fake_robot(sightings=FakeSightings(None))
    robot.gaze = FakeGaze(robot.clock, Attention(phase="voice", bearing_deg=120.0))  # type: ignore[arg-type]
    result = call(robot, target="me")
    assert not result["ok"]
    assert "after the body turned +120 deg toward the voice" in result["why"]
    assert "only the body turned" in result["why"]
    assert not sent(robot) and not planned(robot)


def test_no_voice_to_go_by_leaves_the_old_answer_and_a_refused_turn_is_said() -> None:
    robot = fake_robot(sightings=FakeSightings(None))
    robot.gaze = FakeGaze(robot.clock, Attention(phase="idle"))  # type: ignore[arg-type]
    result = call(robot, target="me")
    assert "nobody in view: ask the person" in result["why"] and "nothing moved" in result["why"]
    own = "the robot's own drive call (this one) has paused the gaze and the wheels"
    gaze = FakeGaze(robot.clock, Attention(phase="voice", bearing_deg=90.0), refuse=own)  # type: ignore[arg-type]
    robot.gaze = gaze
    result = call(robot, target="me")
    assert f"cannot turn to the voice: {own}. Nothing moved: ask the person" in result["why"]
    assert "call approach again only after they do" in result["why"]
    assert not sent(robot)


def test_a_turn_already_running_is_waited_for_and_a_turn_that_never_ends_is_given_up() -> None:
    robot = fake_robot(sightings=Answers(None, SEEN_SPEAKER))
    gaze = FakeGaze(robot.clock, Attention(phase="voice", bearing_deg=-150.0))  # type: ignore[arg-type]
    gaze.face_person("the gaze's own arc rule")  # the tracker started the turn before the tool ran
    robot.gaze = gaze
    result = call(robot, target="me")
    assert result["ok"] and gaze.asked == ["the gaze's own arc rule"], "not asked twice"
    robot = fake_robot(sightings=FakeSightings(None))
    robot.gaze = FakeGaze(robot.clock, Attention(phase="voice", bearing_deg=-150.0), turn_s=99.0)  # type: ignore[arg-type]
    result = call(robot, target="me")
    assert "did not end within 12 s" in result["why"] and not sent(robot)


def test_a_camera_channel_that_does_not_answer_is_said_so() -> None:
    robot = fake_robot(sightings=FakeSightings(down=True))
    result = call(robot, target="me")
    assert "perception service" in result["why"]
    assert not sent(robot)


FRAME_AT = TF_NOW - 0.2  # the fake speaker's frame, exposed 0.2 s before the goal server's now


def turning_cart(cart_deg: float, camera_deg: float) -> Any:
    """TF through the gaze's own turn (2026-10-14): when the person's frame was exposed the
    cart and its camera faced the map's x; now the cart heads ``cart_deg`` and its camera
    ``camera_deg`` (the neck holding the person through the turn keeps it at 0)."""

    def tf(frame: str, stamp: float | None) -> dict[str, float]:
        then = stamp is not None
        if frame == "base_link":
            return {"x": 0.1, "y": 0.05, "yaw_deg": 0.0 if then else cart_deg}
        return {"x": 0.1, "y": 0.05, "z": 1.2, "yaw_deg": 0.0 if then else camera_deg}

    return tf


def test_the_person_is_placed_with_the_poses_of_their_frame_s_stamp_not_of_now() -> None:
    """2026-10-14: approach(me) during the gaze's own turn placed a face of a frame exposed
    mid-turn with the cart's pose of after it, and the cart drove 4.9 m into the kitchen. The
    sighting is placed with TF's camera and cart AT THE FRAME'S STAMP; the camera held on the
    person (the neck's counter-turn) means the depth pair grabbed after it agrees."""
    robot = fake_robot()
    server = goals(robot)
    server.pose = {"x": 0.1, "y": 0.05, "yaw_deg": -40.0, "age_s": 0.05}  # where: after it
    server.tf = turning_cart(cart_deg=-40.0, camera_deg=0.0)
    result = call(robot, target="me")
    assert result["ok"], result
    asked = [
        (a["frame"], a.get("stamp"), a.get("parent", "map"))
        for a in server.asked
        if a["cmd"] == "pose_at"
    ]
    assert asked == [
        ("camera_link", FRAME_AT, "map"),  # where the camera was: places the box
        ("base_link", FRAME_AT, "map"),  # where the cart was: the person's range and bearing
        ("camera_link", FRAME_AT, "odom"),  # how the camera moved since, after the depth pair
        ("camera_link", None, "odom"),
    ]
    (at,) = eyes(robot).stamped
    assert at is not None and (at.stamp, at.age_s) == (FRAME_AT, pytest.approx(0.2))
    assert math.degrees(at.base.yaw_rad) == pytest.approx(0.0), "the frame's heading, not -40"
    found = result["found"]
    assert found["pose"] == {"x": 0.1, "y": 0.05, "heading_deg": 0, "at": "the frame's stamp"}
    assert (found["stamp"], found["age_s"], found["camera_turned_deg"]) == (FRAME_AT, 0.2, 0.0)


def test_a_camera_that_turned_between_the_frame_and_the_depth_is_refused() -> None:
    """The box is the frame's and the depth is the pair grabbed after it: a camera that turned
    in between (the cart's turn, or the head's) makes them disagree, so nothing is driven."""
    for cart, camera in ((-40.0, -40.0), (0.0, 8.0)):
        robot = fake_robot()
        goals(robot).tf = turning_cart(cart_deg=cart, camera_deg=camera)
        result = call(robot, target="me")
        said = f"the camera turned {abs(camera):.0f} deg between the person's picture and the depth"
        assert said in result["why"], result
        assert not sent(robot) and not planned(robot)


def test_a_stale_sighting_is_refused_never_placed() -> None:
    stale = dataclasses.replace(SEEN_SPEAKER, stamp=TF_NOW - 2.3)
    robot = fake_robot(sightings=FakeSightings(stale))
    result = call(robot, target="me")
    assert "the newest sighting of the person is 2.3 s old (over 1.5 s)" in result["why"]
    assert "nothing moved" in result["why"]
    assert not eyes(robot).asked and not sent(robot) and not planned(robot)


def test_without_tf_history_for_the_frame_the_poses_of_now_place_the_person_and_say_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """TF cannot answer the frame's stamp: the neck and the cart of now place the person (the
    path before 2026-10-14), logged and said in the found row; the neck's encoders then judge
    whether the head moved between the frame and the depth."""
    robot = fake_robot()
    goals(robot).tf = lambda frame, stamp: None
    with caplog.at_level(logging.WARNING, logger="pepin.tools.approach"):
        result = call(robot, target="me")
    assert result["ok"], result
    assert eyes(robot).stamped == [None]
    pose = result["found"]["pose"]
    assert pose["at"] == "now" and pose["heading_deg"] == 3
    assert pose["why"].startswith("TF has no pose at the frame's stamp (Extrapolation:")
    assert "the person is placed with the neck and the cart of now" in caplog.text
    assert result["found"]["age_s"] == 0.2 and "camera_turned_deg" not in result["found"]

    class Turning(FakeSightings):
        def speaker(self, head: HeadPose) -> PersonSighting | None:
            assert isinstance(robot.neck, FakeNeck)
            robot.neck.pose_now = HeadPose(head.pan_deg + 8.0, head.tilt_deg)
            return super().speaker(head)

    robot = fake_robot(sightings=Turning())
    goals(robot).tf = lambda frame, stamp: None
    result = call(robot, target="me")
    assert "the head moved 8 deg while the person was being measured" in result["why"]
    assert not sent(robot) and not planned(robot)


def test_a_goal_server_without_pose_at_is_told_to_restart() -> None:
    robot = fake_robot()
    goals(robot).pose_at_error = "unknown command 'pose_at'"
    result = call(robot, target="me")
    assert "started before its pose_at command" in result["why"]
    assert "ros/laptop.sh kick goal_server" in result["why"]
    assert not eyes(robot).asked and not sent(robot)


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
        0.30, abs=0.01
    )  # front_m: the floor (0.25, sigma_k 0) is under it


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


def test_a_plan_ending_within_the_goal_checker_s_tolerance_drives_the_stand(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The plan to the stand (1.80, 0.05) facing the face ends 0.08 m off and turned 10 deg:
    inside the goal checker's 0.10 m and 11.5 deg, so the stand is the goal, as before."""
    robot = fake_robot()
    goals(robot).plan_end = {"x": 1.75, "y": 0.11, "yaw_deg": 10.0}
    with caplog.at_level(logging.WARNING, logger="pepin.tools.approach"):
        result = call(robot, target="me")
    assert result["ok"], result
    (goal,) = sent(robot)
    assert (goal["x"], goal["y"], goal["yaw_deg"]) == pytest.approx((1.80, 0.05, 0.0), abs=0.01)
    assert result["drive_to"]["what"] == "the stand"
    assert result["short_m"] == 0.08
    assert result["target"] == "to the person"
    assert "the planner stops" not in caplog.text


def test_a_plan_ending_short_of_the_stand_drives_its_end_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """2026-10-14, run 66: the stand 0.30 m from a seated person's face had their legs under the
    hull; Smac Hybrid's tolerance ended every plan elsewhere and MPPI chased the end for 80 s.
    The plan asked before the drive ends 0.74 m off and turned 20 deg: that END is the goal
    driven, with its own heading, and the log, the answer and the arrival's words say so."""
    robot = fake_robot()
    goals(robot).plan_end = {"x": 1.3, "y": 0.6, "yaw_deg": -20.0}
    with caplog.at_level(logging.WARNING, logger="pepin.tools.approach"):
        result = call(robot, target="me")
    assert result["ok"], result
    assert len(planned(robot)) == 1, "the one plan asked today, no other"
    (goal,) = sent(robot)
    assert (goal["x"], goal["y"], goal["yaw_deg"]) == (1.3, 0.6, -20.0)
    assert (result["stand"]["x"], result["stand"]["y"]) == (1.8, 0.05), "the stand is kept"
    assert result["drive_to"] == {
        "x": 1.3,
        "y": 0.6,
        "heading_deg": -20,
        "what": "the plan's end, 0.74 m and 20 deg short of the stand; its own heading",
    }
    assert result["short_m"] == 0.74
    # the drive's target is what the [robot] report names on arrival
    assert result["target"] == (
        "to the person, stopping 0.74 m and 20 deg short of the stand: the nearest the planner"
        " reaches"
    )
    assert (
        "approach: the planner stops 0.74 m and 20 deg short of the stand: driving to its end"
        " (1.30, 0.60, -20 deg; its own heading)"
    ) in caplog.text
    assert result["look"]["looked"] is True, "the head still turns to the person"


def test_a_plan_turned_off_the_stand_s_heading_alone_drives_its_end() -> None:
    """At the stand's point but 30 deg off its heading (past the goal checker's 11.5): the
    planner reaches the spot only turned, and that pose is the goal."""
    robot = fake_robot()
    goals(robot).plan_end = {"x": 1.8, "y": 0.05, "yaw_deg": 30.0}
    result = call(robot, target="me")
    (goal,) = sent(robot)
    assert (goal["x"], goal["y"], goal["yaw_deg"]) == (1.8, 0.05, 30.0)
    assert result["target"] == (
        "to the person, stopping 30 deg short of the stand: the nearest the planner reaches"
    )


def test_an_end_without_a_heading_faces_the_person_from_there() -> None:
    """A last pose with no heading (an all-zero quaternion): the end is driven facing the
    person's point (2.1, 0.05) from it, and the answer says which heading and why."""
    robot = fake_robot()
    goals(robot).plan_end = {"x": 1.3, "y": 0.6, "yaw_deg": None}
    result = call(robot, target="me")
    (goal,) = sent(robot)
    facing = math.degrees(math.atan2(0.05 - 0.6, 2.1 - 1.3))
    assert (goal["x"], goal["y"]) == (1.3, 0.6)
    assert goal["yaw_deg"] == pytest.approx(facing, abs=0.1)
    assert result["drive_to"]["what"] == (
        "the plan's end, 0.74 m short of the stand; facing the person from there: the plan's"
        " last pose has no heading"
    )


def test_the_next_spot_s_plan_ending_short_drives_that_plan_s_end() -> None:
    """The candidate loop as before: the first spot refused, the second accepted; its plan's
    end, short of it, replaces it as the goal, and the stand row names the second spot."""
    robot = fake_robot()
    goals(robot).unplannable = 1
    goals(robot).plan_end = {"x": 1.5, "y": 0.5, "yaw_deg": -45.0}
    result = call(robot, target="me")
    assert result["ok"], result
    assert len(planned(robot)) == 2
    assert result["stand"]["choice"].startswith("2 of 3: to turned 30 deg left")
    (goal,) = sent(robot)
    assert (goal["x"], goal["y"], goal["yaw_deg"]) == (1.5, 0.5, -45.0)
    assert result["drive_to"]["what"].startswith("the plan's end, ")


def test_a_goal_server_whose_plan_names_no_end_is_told_to_restart() -> None:
    robot = fake_robot()
    old_build = {"event": "plan", "ok": True, "cell": "free", "length_m": 2.0, "poses": 40}
    goals(robot).plan = lambda x, y, yaw_deg: old_build
    result = call(robot, target="me")
    assert not result["ok"]
    assert "does not say where a plan ends" in result["why"]
    assert "ros/laptop.sh kick goal_server" in result["why"]
    assert not sent(robot)


def test_a_spoken_distance_is_kept_but_never_nearer_than_the_minimum() -> None:
    robot = fake_robot()
    call(robot, target="me", distance_m=0.2)
    (goal,) = sent(robot)
    # the 0.25 m floor alone (sigma_k 0: the face's 0.07 sigma adds nothing): 0.25 m short of
    # the face, not the spoken 0.2.
    assert goal["x"] == pytest.approx(2.1 - 0.25, abs=0.01)
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
    """The person to the robot's left: one goal at the robot's own spot, turned toward them
    (the goal server's go turns in place), and NO plan asked for it: the cart is already on
    that cell, and Smac answers a goal on the start's cell with a one-pose path the goal
    server's plan reads as no path (2026-10-13, "повернись ко мне" on the dock)."""
    robot = fake_robot()
    eyes(robot).point = (0.1, 1.6, 1.55)
    result = call(robot, target="me", relation="face")
    assert result["ok"], result
    (goal,) = sent(robot)
    assert (goal["x"], goal["y"], goal["yaw_deg"]) == pytest.approx((0.1, 0.05, 90.0))
    assert not planned(robot)
    near = fake_robot()
    eyes(near).point = (0.35, 0.325, 1.55)  # 0.37 m off at 48 deg: "come closer" only turns
    assert call(near, target="me")["stand"]["heading_deg"] == 48
    assert [(g["x"], g["y"]) for g in sent(near)] == [(0.1, 0.05)]
    assert not planned(near)


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
                                   "the person the gaze holds", STILL_FROM + 0.1)  # fmt: skip
    assert found is not None and found.stamp == STILL_FROM + 0.1, "the record's own stamp"


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
