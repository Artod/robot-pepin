"""Every tool of pepin.tools against fakes of the owners it speaks to (pepin.tools.fakes): what
it asks, what it answers, and the words it gives the model when something is not right."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest

from pepin.tools import TOOLS, Result, Robot
from pepin.tools.clients import HeadPose
from pepin.tools.fakes import (
    REST,
    FakeCamera,
    FakeClock,
    FakeGoalServer,
    FakeNeck,
    FakeSpeech,
    FakeThing,
    FakeWorld,
    arrival_script,
    fake_robot,
)


def call(robot: Robot, tool: str, /, **arguments: Any) -> Result:
    """One tool call, as a model makes it."""
    return TOOLS.call(tool, arguments, robot)


def goals(robot: Robot) -> FakeGoalServer:
    assert isinstance(robot.goals, FakeGoalServer)
    return robot.goals


def neck(robot: Robot) -> FakeNeck:
    assert isinstance(robot.neck, FakeNeck)
    return robot.neck


def world(robot: Robot) -> FakeWorld:
    assert isinstance(robot.world, FakeWorld)
    return robot.world


# -- where_am_i, list_places ----------------------------------------------------------------------


def test_where_am_i_names_the_place_the_robot_stands_at() -> None:
    result = call(fake_robot(), "where_am_i")
    assert result == {
        "ok": True,
        "x": 0.1,
        "y": 0.05,
        "heading_deg": 3,
        "nearest_place": "home",
        "distance_m": 0.11,
        "at": "home",
    }


def test_where_am_i_away_from_any_place_names_the_nearest_and_warns_of_a_stale_pose() -> None:
    robot = fake_robot()
    goals(robot).pose = {"x": 1.0, "y": 1.0, "yaw_deg": -90.0, "age_s": 4.2}
    result = call(robot, "where_am_i")
    assert result["at"] is None and result["nearest_place"] == "home"
    assert result["warning"] == "the pose is 4.2 s old: localisation may have stalled"


def test_where_am_i_without_a_pose_says_so() -> None:
    robot = fake_robot()
    goals(robot).pose = None
    result = call(robot, "where_am_i")
    assert not result["ok"] and "does not know where it is" in result["why"]


def test_a_goal_server_that_is_down_is_named_with_what_to_check() -> None:
    robot = fake_robot()
    goals(robot).down = True
    result = call(robot, "where_am_i")
    assert result["why"] == (
        "the goal server (127.0.0.1:3337) is not answering: connection refused. Is it up?"
    )


def test_list_places_nearest_first() -> None:
    result = call(fake_robot(), "list_places")
    assert [p["name"] for p in result["places"]] == ["home", "bookshelf", "printer"]
    assert result["places"][2] == {"name": "printer", "x": -1.2, "y": 3.4, "distance_m": 3.59}


def test_list_places_without_a_pose_or_a_book() -> None:
    robot = fake_robot()
    goals(robot).pose = None
    assert [p["name"] for p in call(robot, "list_places")["places"]] == [
        "bookshelf",
        "home",
        "printer",
    ]
    goals(robot).book = {}
    empty = call(robot, "list_places")
    assert empty["ok"] and empty["places"] == [] and "ros/goto.sh mark" in empty["note"]


# -- go_to, go_to_pose, cancel --------------------------------------------------------------------


def test_go_to_waits_for_the_arrival_and_reports_it() -> None:
    robot = fake_robot()
    result = call(robot, "go_to", place="printer")
    assert result == {
        "ok": True,
        "arrived": True,
        "target": "printer",
        "seconds": 3.0,
        "run": 7,
        "position": {"x": -1.15, "y": 3.37, "heading_deg": 92},
        "off_by_m": 0.06,
    }
    assert goals(robot).asked[-1] == {"cmd": "go", "place": "printer"}


def test_go_to_pose_sends_the_coordinates() -> None:
    robot = fake_robot()
    result = call(robot, "go_to_pose", x=1.5, y="-0.5", yaw_deg=90)
    assert result["arrived"] and result["target"] == "(1.50, -0.50)"
    assert goals(robot).asked[-1] == {"cmd": "go", "x": 1.5, "y": -0.5, "yaw_deg": 90.0}


def test_an_unknown_place_is_answered_with_the_known_ones() -> None:
    result = call(fake_robot(), "go_to", place="kitchen")
    assert result == {
        "ok": False,
        "why": "not driving: no such place: 'kitchen'. Known places: bookshelf, home, printer",
    }


def test_a_second_drive_is_refused_with_what_to_do() -> None:
    robot = fake_robot()
    goals(robot).refuse = "already driving: cancel first"
    why = call(robot, "go_to", place="printer")["why"]
    assert why == (
        "not driving: already driving: cancel first (another drive is under way: wait for it"
        " to end, or cancel it)"
    )


def test_a_blocked_way_says_how_far_short_and_why() -> None:
    robot = fake_robot()
    goals(robot).drive = [
        {"event": "accepted", "run": 8, "place": "printer", "x": -1.2, "y": 3.4, "yaw_deg": 90.0},
        {"event": "feedback", "t": 5.0, "distance": 2.44, "recoveries": 3},
        {"event": "done", "run": 8, "status": 6, "seconds": 41.0, "arrival": None},
    ]
    result = call(robot, "go_to", place="printer")
    assert not result["ok"] and not result["arrived"]
    assert result["why"] == (
        "the navigator gave up after 3 recovery attempts, 2.4 m short of printer: the way there"
        " is probably blocked, or the goal lies inside an obstacle"
    )
    assert result["distance_left_m"] == 2.44


def test_a_lost_pose_mid_drive_is_the_reason_given() -> None:
    robot = fake_robot()
    goals(robot).drive = [
        {"event": "accepted", "run": 9, "place": "printer", "x": -1.2, "y": 3.4, "yaw_deg": 90.0},
        {"event": "feedback", "t": 3.0, "distance": 1.1, "recoveries": 0},
        {"event": "lost", "t": 4.0, "reading": "no SLAM correction for 3.1 s"},
        {"event": "done", "run": 9, "status": 5, "seconds": 4.5},
    ]
    why = call(robot, "go_to", place="printer")["why"]
    assert why == (
        "localisation was lost mid-drive (no SLAM correction for 3.1 s) and the goal server"
        " stopped, 1.1 m short of printer"
    )


def test_a_drive_past_its_time_is_cancelled_by_the_tool() -> None:
    """Nobody queues motions: a drive that outlives its call's patience is stopped."""
    robot = fake_robot(drive_timeout_s=5.0)
    goals(robot).drive = [
        {"event": "accepted", "run": 10, "place": "printer", "x": -1.2, "y": 3.4, "yaw_deg": 0.0},
        *(
            {"event": "feedback", "t": float(t), "distance": 3.0, "recoveries": 0}
            for t in range(20)
        ),
    ]
    result = call(robot, "go_to", place="printer")
    assert goals(robot).cancelled == 1
    assert result["why"] == "gave up after 5 s without arriving, 3.0 m short of printer"


def test_a_drive_whose_reports_are_lost_is_halted() -> None:
    """The connection to the goal server drops mid-drive: the drive must not go on unwatched."""
    robot = fake_robot()
    script = arrival_script(-1.2, 3.4, 90.0, "printer")

    def dropping(request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        goals(robot).asked.append(request)
        yield script[0]
        yield script[1]
        raise ConnectionResetError("reset by peer")

    goals(robot).go = dropping  # type: ignore[method-assign]
    result = call(robot, "go_to", place="printer")
    assert goals(robot).cancelled == 1
    assert result["why"] == (
        "lost the drive's reports (reset by peer); cancel sent: every drive is stopping"
    )
    assert result["distance_left_m"] == 2.0


def test_a_stream_that_ends_before_the_drive_was_taken_still_halts() -> None:
    robot = fake_robot()
    goals(robot).drive = []  # the server closed without a word
    why = call(robot, "go_to", place="printer")["why"]
    assert why == (
        "the goal server went silent before confirming the drive (no word); cancel sent: every"
        " drive is stopping"
    )
    assert goals(robot).cancelled == 1


def test_a_drive_leaves_the_head_to_the_gaze_arbiter() -> None:
    """The arbiter lets every look go when a drive starts and points the head for it: the drive
    tool never touches the head, and a head nobody can reach does not stop a drive."""
    robot = fake_robot()
    neck(robot).pose_now = HeadPose(60.0, 10.0)
    neck(robot).down = True
    result = call(robot, "go_to", place="home")
    assert result["arrived"] and "head" not in result
    assert neck(robot).moves == []


def test_cancel_stops_the_drive() -> None:
    robot = fake_robot()
    assert call(robot, "cancel") == {"ok": True, "stopped": True, "detail": "stopping the drive"}
    assert goals(robot).asked == [{"cmd": "cancel"}]


def test_a_cancel_that_does_not_reach_the_robot_says_a_person_must_stop_it() -> None:
    robot = fake_robot()
    goals(robot).down = True
    why = call(robot, "cancel")["why"]
    assert why.startswith("the cancel did NOT reach the robot") and "ros/stop.sh" in why


def test_a_cancel_no_navigator_confirmed_is_a_failure() -> None:
    robot = fake_robot()

    def unconfirmed() -> dict[str, Any]:
        return {
            "event": "cancelled",
            "had_goal": False,
            "navigators": {"navigate_to_pose": {"outcome": "NOT confirmed in 30 s"}},
        }

    goals(robot).cancel = unconfirmed  # type: ignore[method-assign]
    why = call(robot, "cancel")["why"]
    assert why.startswith("no navigator confirmed the cancel (cancel — navigate_to_pose: NOT")


# -- look, look_around, see -----------------------------------------------------------------------


def test_look_turns_the_head_and_reports_where_it_points() -> None:
    robot = fake_robot()
    assert call(robot, "look", pan_deg=45) == {"ok": True, "pan_deg": 45.0, "tilt_deg": 23.8}
    assert call(robot, "look", pan_deg=-30, tilt_deg=60) == {
        "ok": True,
        "pan_deg": -30.0,
        "tilt_deg": 60.0,
    }


def test_look_past_the_reach_is_refused_before_anything_moves() -> None:
    robot = fake_robot()
    result = call(robot, "look", pan_deg=170)
    assert result["why"].startswith("pan +170 deg is past the neck's reach")
    assert "tilt -40 deg" in call(robot, "look", pan_deg=0, tilt_deg=-40)["why"]
    assert neck(robot).moves == []


def test_look_during_a_drive_says_to_wait_in_the_arbiters_words() -> None:
    robot = fake_robot()
    neck(robot).wheels_moving = True
    why = call(robot, "look", pan_deg=30)["why"]
    assert why == (
        "the head did not get there: the head does not move during a drive (the base server"
        " moves the neck only at rest): wait for the drive to end, or cancel it"
    )


def test_look_around_reports_each_direction_and_leaves_the_head_to_go_home() -> None:
    """No tool turns the head back: the last view's TTL ends and the arbiter takes it home."""
    robot = fake_robot()
    neck(robot).pose_now = HeadPose(10.0, 20.0)
    world(robot).things = [
        FakeThing("chair", pan_deg=60.0, range_m=2.0, bearing_deg=58.0),
        FakeThing("chair", pan_deg=55.0, range_m=3.0),
        FakeThing("cup", pan_deg=-120.0, range_m=1.2, bearing_deg=-118.0),
    ]
    result = call(robot, "look_around")
    seen = {view["pan_deg"]: [s["label"] for s in view["seen"]] for view in result["views"]}
    assert seen == {120.0: [], 60.0: ["chair"], 0.0: [], -60.0: [], -120.0: ["cup"]}
    chair = result["views"][1]["seen"][0]
    assert chair["count"] == 2 and chair["range_m"] == 2.0 and chair["bearing_deg"] == 58
    assert "head" not in result and neck(robot).moves[-1] == (-120.0, None)


def test_look_around_without_the_memory_does_not_move_the_head() -> None:
    robot = fake_robot()
    world(robot).down = True
    result = call(robot, "look_around")
    assert "world" in result["why"] and neck(robot).moves == []


def test_see_returns_the_picture_and_where_the_head_points_and_keeps_it_there() -> None:
    robot = fake_robot()
    result = call(robot, "see")
    assert result["image"].mime == "image/jpeg" and result["pan_deg"] == 0.0
    assert neck(robot).kept == 1


def test_see_without_the_head_still_returns_the_picture() -> None:
    robot = fake_robot()
    neck(robot).down = True
    result = call(robot, "see")
    assert result["ok"] and result["head"].startswith("unknown: the gaze arbiter")


def test_see_without_the_camera_says_what_to_check() -> None:
    robot = fake_robot(camera=FakeCamera(down=True))
    assert "camera" in call(robot, "see")["why"]


# -- find, recall, map_tree, remember -------------------------------------------------------------


def test_find_answers_from_a_sighting_in_view_now() -> None:
    robot = fake_robot()
    world(robot).things = [
        FakeThing("chair", range_m=1.8, bearing_deg=12.0, x=2.0, y=1.0, score=0.6),
        FakeThing("chair", range_m=2.5, score=0.9, x=3.0, y=0.0),
    ]
    result = call(robot, "find", thing=" Chair ")
    assert result == {
        "ok": True,
        "label": "chair",
        "score": 0.9,
        "range_m": 2.5,
        "bearing_deg": 0,
        "x": 3.0,
        "y": 0.0,
        "z": 0.4,
        "also_seen": 1,
    }


def test_find_waits_for_a_fresh_sighting() -> None:
    robot = fake_robot()
    world(robot).things = [FakeThing("cup", from_s=1.0)]
    assert call(robot, "find", thing="cup")["label"] == "cup"
    assert 1.0 <= robot.clock() < 2.5


def test_find_does_not_answer_from_a_stale_sighting() -> None:
    """A cup seen a minute ago is not in view now: recall is for that."""
    robot = fake_robot()
    robot.sleep(60.0)
    world(robot).things = [FakeThing("cup", until_s=0.0)]
    why = call(robot, "find", thing="cup")["why"]
    assert why.startswith("no cup in view now (watched 2.5 s)") and "recall('cup')" in why


def test_find_with_look_around_stays_pointed_at_the_thing() -> None:
    robot = fake_robot()
    world(robot).things = [FakeThing("backpack", pan_deg=-60.0, range_m=1.4, bearing_deg=-61.0)]
    result = call(robot, "find", thing="backpack", look_around=True)
    assert result["ok"] and result["bearing_deg"] == -61
    assert result["head"] == {"pan_deg": -60.0, "tilt_deg": 23.8}
    assert neck(robot).pose_now.pan_deg == -60.0


def test_find_with_look_around_leaves_the_head_to_go_home_when_nothing_is_found() -> None:
    robot = fake_robot()
    result = call(robot, "find", thing="umbrella", look_around=True)
    assert result["why"].startswith("no umbrella anywhere around here: the head looked in 5")
    assert neck(robot).moves[-1] == (-120.0, None) and neck(robot).pose_now != REST


def test_find_cannot_sweep_during_a_drive() -> None:
    robot = fake_robot()
    neck(robot).wheels_moving = True
    why = call(robot, "find", thing="cup", look_around=True)["why"]
    assert why.startswith("the head could not sweep: the head does not move during a drive")


def test_recall_gives_the_memory_with_distances_from_here() -> None:
    robot = fake_robot()
    world(robot).objects_known = [
        {"id": "o7", "label": "mug", "x": 3.1, "y": 0.05, "zone": "kitchen", "count": 4},
        {"id": "o9", "label": "chair", "x": 0.0, "y": 0.0},
    ]
    result = call(robot, "recall", thing="mug")
    assert result == {
        "ok": True,
        "matches": [
            {
                "id": "o7",
                "label": "mug",
                "x": 3.1,
                "y": 0.05,
                "zone": "kitchen",
                "count": 4,
                "distance_m": 3.0,
            }
        ],
        "total": 1,
    }
    assert "never seen" in call(robot, "recall", thing="sock")["why"]


def test_map_tree_is_the_memory_s_text() -> None:
    robot = fake_robot()
    assert call(robot, "map_tree")["tree"].startswith("flat\n  living room")
    world(robot).tree_text = ""
    assert not call(robot, "map_tree")["ok"]


def test_remember_a_zone_where_the_robot_stands_and_an_object_by_id() -> None:
    robot = fake_robot()
    assert call(robot, "remember", name="kitchen", what="zone")["ok"]
    assert call(robot, "remember", name="my mug", what="object", object_id="o7")["ok"]
    assert world(robot).remembered == [
        {"what": "zone", "name": "kitchen", "x": 0.1, "y": 0.05},
        {"what": "object", "name": "my mug", "id": "o7"},
    ]


@pytest.mark.parametrize(
    ("arguments", "why"),
    [
        ({"name": "desk", "what": "place"}, "a place to drive back to is named in RTAB-Map's"),
        ({"name": "mug", "what": "object"}, "which thing? call recall first"),
        ({"name": "  ", "what": "zone"}, "the name is empty"),
        ({"name": "x", "what": "room"}, "what must be one of zone, object, place"),
    ],
)
def test_remember_refusals(arguments: dict[str, Any], why: str) -> None:
    robot = fake_robot()
    assert TOOLS.call("remember", arguments, robot)["why"].startswith(why)
    assert world(robot).remembered == []


def test_the_memory_being_down_is_named() -> None:
    robot = fake_robot()
    world(robot).down = True
    for name, arguments in (("find", {"thing": "cup"}), ("recall", {"thing": "cup"})):
        assert (
            "world (http://127.0.0.1:8798) is not answering"
            in call(robot, name, **arguments)["why"]
        )


# -- say, status ----------------------------------------------------------------------------------


def test_say_speaks_through_the_voice() -> None:
    speech = FakeSpeech()
    robot = fake_robot(speech=speech)
    assert call(robot, "say", text="  Еду к принтеру ") == {"ok": True, "spoken_s": 0.8}
    assert speech.said == ["Еду к принтеру"]
    assert call(robot, "say", text=" ")["why"] == "nothing to say: the text is empty"
    assert call(robot, "say", text="a" * 601)["why"].startswith("too long to say (601 characters")


def test_status_asks_every_owner_and_lists_what_is_down() -> None:
    robot = fake_robot(speech=FakeSpeech(down=True))
    world(robot).down = True
    result = call(robot, "status")
    assert result["ok"]
    assert result["up"] == ["goal_server", "head", "camera"]
    assert result["down"] == ["world", "voice"]
    assert result["services"]["goal_server"]["detail"] == (
        "pose 0.05 s old, planner hybrid, lidar ok"
    )


# -- safety ---------------------------------------------------------------------------------------


def test_no_tool_asks_the_goal_server_for_anything_but_where_places_go_cancel() -> None:
    """The tools compose the goal server's four commands; they never mark, re-plan or reach a
    topic. Every tool is run once with fakes and the commands it sent are collected."""
    robot = fake_robot()
    world(robot).things = [FakeThing("cup")]
    world(robot).objects_known = [{"id": "o1", "label": "cup", "x": 1.0, "y": 1.0}]
    arguments = {
        "go_to": {"place": "printer"},
        "go_to_pose": {"x": 1.0, "y": 0.0, "yaw_deg": 0.0},
        "look": {"pan_deg": 10.0},
        "find": {"thing": "cup", "look_around": True},
        "recall": {"thing": "cup"},
        "remember": {"name": "hall", "what": "zone"},
        "say": {"text": "hi"},
    }
    for t in TOOLS:
        assert call(robot, t.name, **arguments.get(t.name, {}))["ok"], t.name
    assert {request["cmd"] for request in goals(robot).asked} == {"where", "places", "go", "cancel"}


def test_a_fake_robot_shares_one_clock() -> None:
    clock = FakeClock()
    robot = fake_robot(clock=clock)
    robot.sleep(1.5)
    assert clock() == robot.clock() == 1.5
    assert math.isinf(FakeThing("x").until_s)
