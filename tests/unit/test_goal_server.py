"""Who tells the goal server where the cart stands, and what a goal is judged on.

rclpy is faked (``ros_stubs``), so the node is built here exactly as on the board: its clients
answer nobody until a test says the service is there, and its TF buffer holds whatever the test
puts in it. The pose is ``map -> base_link`` (RTAB-Map owns ``map -> odom``); a goal is judged by
how fresh that edge is and by whether this start of RTAB-Map is placed (test_one_localiser holds
the placement rule itself).
"""

from __future__ import annotations

import json
import math
import types
from pathlib import Path
from typing import Any

import ros_stubs

ros_stubs.install()


from pepin_bringup.goal_server import GoalServer  # noqa: E402
from ros_stubs import (  # noqa: E402
    Header,
    Quaternion,
    TransformStamped,
    Vector3,
)

from pepin.watch import PLACEMENT_TOPIC, TF_FRESH_S, Placement  # noqa: E402

NOW = 1000.0  # the node's clock, in seconds; a transform's age is NOW minus its stamp


class Wire:
    """The operator's socket as the node writes to it: every event, decoded."""

    def __init__(self) -> None:
        self.raw = b""

    def sendall(self, data: bytes) -> None:
        self.raw += data

    def events(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.raw.decode().splitlines() if line.strip()]


def server(tmp_path: Path) -> Any:
    """A goal server on an ephemeral port, its records in ``tmp_path``: no TF, no placement,
    no Nav2 — a test adds what its stack has."""
    with ros_stubs.parameters(port=0, record_dir=str(tmp_path)):
        node = GoalServer()
    node.clock.seconds = NOW
    return node


def at(x: float, y: float, yaw_deg: float, age_s: float) -> Any:
    """map -> base_link with the cart there, stamped ``age_s`` seconds before the node's now."""
    stamp = NOW - age_s
    return TransformStamped(
        header=Header(
            stamp=ros_stubs.Time(sec=int(stamp), nanosec=round((stamp % 1.0) * 1e9)),
            frame_id="map",
        ),
        child_frame_id="base_link",
        transform=ros_stubs.Transform(
            translation=Vector3(x=x, y=y),
            rotation=Quaternion(
                z=math.sin(math.radians(yaw_deg) / 2.0), w=math.cos(math.radians(yaw_deg) / 2.0)
            ),
        ),
    )


def standing_at(node: Any, transform: Any) -> None:
    """Put map -> base_link into the node's own TF buffer. The first lookup is what starts the
    listener (the node never subscribes to /tf until something asks for a pose), so one ask
    comes first — exactly as on the robot."""
    node._tf_pose()
    node._tf.buffer.transforms[("map", "base_link")] = transform


def placed(node: Any) -> None:
    """rtabmap_frame's latched word: this start of RTAB-Map recognised the loaded map."""
    word = Placement(updates=40, recognised=1, seeds=0, loaded=True)
    node.subs[PLACEMENT_TOPIC][1](ros_stubs.String(data=word.to_json(0.0)))


class Pending:
    """Nav2's result future as the drive loop reads it: not done for ``ticks`` turns of the
    loop, then done. A callback fires at once, so the thread's own wait never hangs a test;
    ``on_tick`` runs at the top of every turn — where a test makes the world change mid-drive."""

    def __init__(self, ticks: int, on_tick: Any = None) -> None:
        self.ticks = ticks
        self._on_tick = on_tick

    def done(self) -> bool:
        self.ticks -= 1
        if self._on_tick is not None:
            self._on_tick()
        return self.ticks < 0

    def add_done_callback(self, callback: Any) -> None:
        callback(self)

    def result(self) -> Any:
        return None


class Handle:
    """Nav2's goal handle: accepted, its result pending, and it counts its cancellations."""

    def __init__(self, ticks: int, on_tick: Any = None) -> None:
        self.accepted = True
        self.cancelled = 0
        self._result = Pending(ticks, on_tick)

    def get_result_async(self) -> Pending:
        return self._result

    def cancel_goal_async(self) -> None:
        self.cancelled += 1


def nav2_answers(node: Any, ticks: int, on_tick: Any = None) -> Handle:
    """A Nav2 that accepts the goal and keeps driving for ``ticks`` turns of the drive loop."""
    node._client.server = True
    node._client.handle = Handle(ticks, on_tick)
    handle: Handle = node._client.handle
    return handle


def test_a_fresh_transform_is_the_pose_and_the_goal_goes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The pose is map -> base_link and the goal is judged by how fresh that edge is (and by the
    placement, given here)."""
    node = server(tmp_path)
    standing_at(node, at(1.25, -0.5, 90.0, age_s=0.1))
    placed(node)
    pose = node._pose_now()
    assert (round(pose["x"], 3), round(pose["y"], 3)) == (1.25, -0.5)
    assert abs(pose["yaw_deg"] - 90.0) < 1e-6
    assert abs(pose["age_s"] - 0.1) < 1e-3
    assert "fit" not in pose, "TF carries no fit, and none is invented"
    ready = node._ready()
    assert ready.ready

    wire = Wire()
    node._handle({"cmd": "go", "x": 1.0, "y": 0.3, "yaw_deg": 0.0}, wire)
    # The gate let it through: what stops it now is Nav2's own action server.
    assert [e["detail"] for e in wire.events()] == ["Nav2 is not up"]


def test_without_a_transform_the_goal_is_refused_with_the_reason(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Nothing knows where the cart is: the refusal says so."""
    node = server(tmp_path)
    ready = node._ready()
    assert not ready.ready
    assert node._pose_now() == {}

    wire = Wire()
    node._handle({"cmd": "go", "x": 1.0, "y": 0.3}, wire)
    (event,) = wire.events()
    assert event["event"] == "error"
    assert "nothing publishes map -> base_link" in event["detail"]
    assert not node._client.goals, "nothing was sent to Nav2"


def test_a_transform_that_stopped_coming_is_as_good_as_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """RTAB-Map re-broadcasts map -> odom at 20 Hz over 50 Hz odometry, so a second-old edge
    means a publisher has stopped. The refusal carries the age, because "stale" and "missing"
    are two different things to go and look at."""
    node = server(tmp_path)
    placed(node)
    standing_at(node, at(0.0, 0.0, 0.0, age_s=4.2))
    ready = node._ready()
    assert not ready.ready
    assert "map -> base_link is 4.2 s old" in ready.reason
    standing_at(node, at(0.0, 0.0, 0.0, age_s=TF_FRESH_S - 0.01))
    assert node._ready().ready


def test_a_name_is_never_answered_from_the_old_map_s_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The yaml beside the map holds coordinates of a frame that no longer exists. Until the
    graph's book has arrived a named goal is refused with THAT reason — on 2026-09-21 `printer`
    was answered from the file 2.1 s after a cold start, as (-11.38, +0.77), outside the map,
    and the behaviour tree backed the cart into a sofa. The file is never read."""
    (tmp_path / "places.yaml").write_text(
        json.dumps({"printer": {"x": -11.38, "y": 0.77, "yaw_deg": 140.0}})
    )
    node = server(tmp_path)
    assert node.places() == {}
    wire = Wire()
    node._handle({"cmd": "go", "place": "printer"}, wire)
    (event,) = wire.events()
    assert event["event"] == "error" and "has not arrived" in event["detail"]
    assert not node._client.goals, "nothing was sent to Nav2"

    node._on_places(ros_stubs.String(data=json.dumps({"places": {}})))
    wire = Wire()
    node._handle({"cmd": "go", "place": "printer"}, wire)
    assert "no such place" in wire.events()[0]["detail"], "the book arrived and has no such name"


# ---- a cancel means every goal on the board ---------------------------------------------------

CANCEL_SERVICES = {
    "navigate_to_pose": "/navigate_to_pose/_action/cancel_goal",
    "navigate_through_poses": "/navigate_through_poses/_action/cancel_goal",
}


def navigators_up(node: Any, **answers: Any) -> None:
    """Both navigators' cancel services exist; ``answers[action]`` is what each says."""
    for action, name in CANCEL_SERVICES.items():
        client = node.service_clients[name]
        client.ready = True
        client.response = answers.get(action)


def test_a_cancel_reaches_every_goal_on_both_navigators_whoever_sent_it(tmp_path: Path) -> None:
    """ros/goto.sh's drives are goto_ros.py's goals, which this node never sent: the cancel asks
    both action servers for EVERY goal (an empty request, a zero id) and answers in the words
    goto_ros.py prints, so the laptop's pepin.goal_link shows the operator the same line."""
    from pepin.goal_link import cancel_line

    node = server(tmp_path)
    navigators_up(
        node,
        navigate_to_pose=ros_stubs.CancelGoal.Response(return_code=0, goals_canceling=[1]),
        navigate_through_poses=ros_stubs.CancelGoal.Response(return_code=2, goals_canceling=[]),
    )
    wire = Wire()
    node._handle({"cmd": "cancel"}, wire)
    [answer] = wire.events()
    assert answer["event"] == "cancelled" and answer["had_goal"] is False
    assert cancel_line(answer) == (
        "cancel — navigate_to_pose: accepted, 1 cancelling;"
        " navigate_through_poses: no such goal, 0 cancelling"
    )
    for name in CANCEL_SERVICES.values():
        assert len(node.service_clients[name].calls) == 1, name


def test_this_node_s_own_goal_is_one_of_every_goal_and_is_cancelled_once(tmp_path: Path) -> None:
    """A second request for a goal already put into canceling is rejected by the navigator, and
    the operator would read 'rejected' for a drive that is stopping: the handle is let go, the
    navigators' cancel is the one that stops it, and no resume can send it again."""
    node = server(tmp_path)
    navigators_up(
        node, navigate_to_pose=ros_stubs.CancelGoal.Response(return_code=0, goals_canceling=[1])
    )
    own = Handle(ticks=5)
    node._goal_handle, node._driving = own, True
    wire = Wire()
    node._handle({"cmd": "cancel"}, wire)
    [answer] = wire.events()
    assert answer["had_goal"] is True
    assert answer["navigators"]["navigate_to_pose"] == {"outcome": "accepted", "cancelling": 1}
    assert own.cancelled == 0 and node._goal_handle is None and not node._driving


def test_a_navigator_that_is_not_up_or_does_not_answer_is_said_so(tmp_path: Path) -> None:
    node = server(tmp_path)
    navigators_up(node)  # both there, neither answers (the stub's future holds None)
    node.service_clients[CANCEL_SERVICES["navigate_through_poses"]].ready = False
    said = node.cancel_every_goal()
    assert said == {
        "navigate_to_pose": {"outcome": "NOT confirmed in 30 s — use ros/stop.sh"},
        "navigate_through_poses": {"outcome": "no server answered"},
    }
    waited = node.service_clients[CANCEL_SERVICES["navigate_through_poses"]].waits
    assert waited and max(waited) <= 15.0, "an absent navigator costs half the window, not more"


# ---- a cancel stops a drive at every stage of it ----------------------------------------------


class Finished:
    """A result future that is already done with Nav2's ``status``."""

    def __init__(self, status: int, on_wait: Any = None) -> None:
        self._status, self._on_wait = status, on_wait

    def done(self) -> bool:
        return True

    def add_done_callback(self, callback: Any) -> None:
        if self._on_wait is not None:
            self._on_wait()  # the operator's cancel lands while this is awaited
        callback(self)

    def result(self) -> Any:
        return types.SimpleNamespace(status=self._status)


class Goal:
    """A goal handle that Nav2 accepted, its result ``status``, counting its cancellations."""

    def __init__(self, status: int, on_wait: Any = None) -> None:
        self.accepted, self.cancelled = True, 0
        self._result = Finished(status, on_wait)

    def get_result_async(self) -> Finished:
        return self._result

    def cancel_goal_async(self) -> None:
        self.cancelled += 1


def ready_to_drive(tmp_path: Path) -> Any:
    """A placed node with a fresh pose, Nav2 up, and no recorder to wait for."""
    node = server(tmp_path)
    standing_at(node, at(0.0, 0.0, 0.0, age_s=0.1))
    placed(node)
    node._client.server = True
    node.start_recording = lambda name: None
    node.stop_recording = lambda: None
    return node


def test_a_cancel_before_nav2_has_the_goal_means_the_goal_is_never_sent(tmp_path: Path) -> None:
    """The recorder alone may hold a drive up to 8 s between `_driving` and the goal going to
    Nav2; a cancel in that window used to clear the flag and the goal went out anyway."""
    node = ready_to_drive(tmp_path)

    def recorder_slow_and_a_cancel_meanwhile(name: str) -> None:
        node.cancel()

    node.start_recording = recorder_slow_and_a_cancel_meanwhile
    wire = Wire()
    node._handle({"cmd": "go", "x": 1.0, "y": 0.3}, wire)
    assert node._client.goals == [], "nothing was sent to Nav2"
    assert wire.events()[-1] == {"event": "error", "detail": "cancelled before Nav2 had the goal"}
    assert not node._driving and not node.navigating()


def test_a_handle_that_arrives_after_the_cancel_is_cancelled_at_once(tmp_path: Path) -> None:
    node = ready_to_drive(tmp_path)
    goal = Goal(status=4)
    sent = node._client.send_goal_async

    def nav2_takes_it_while_the_operator_cancels(request: Any, feedback: Any = None) -> Any:
        future = sent(request, feedback)
        node.cancel()
        return future

    node._client.handle = goal
    node._client.send_goal_async = nav2_takes_it_while_the_operator_cancels
    wire = Wire()
    node._handle({"cmd": "go", "x": 1.0, "y": 0.3}, wire)
    assert goal.cancelled == 1, "the late handle is cancelled, not driven"
    assert [e["event"] for e in wire.events()] == ["error"]
    assert node._goal_handle is None and not node._driving


def test_a_cancel_during_the_pivot_stops_the_spin(tmp_path: Path) -> None:
    """The drive ends on position (status 4) and the behaviour server's Spin turns the cart to
    the mark's heading: no navigator owns that goal, so the cancel stops it by its own handle."""
    node = ready_to_drive(tmp_path)
    node._client.handle = Goal(status=4)
    node._spin.server = True
    spin = Goal(status=5, on_wait=lambda: node._handle({"cmd": "cancel"}, Wire()))
    node._spin.handle = spin
    wire = Wire()
    node._handle({"cmd": "go", "x": 1.0, "y": 0.3, "yaw_deg": 90.0}, wire)
    assert len(node._spin.goals) == 1, "the pivot started"
    assert spin.cancelled == 1, "and the cancel stopped it"
    assert node._spin_handle is None and not node.navigating()
    assert [e["event"] for e in wire.events()] == ["accepted", "pivot", "done"]


def test_where_says_whether_any_goal_runs_on_the_navigators(tmp_path: Path) -> None:
    """The measured motions (ros/goto.sh round, move) refuse unless the goal server says idle:
    a goal of anyone's on a navigator counts, from its latched status list."""
    node = server(tmp_path)
    wire = Wire()
    node._handle({"cmd": "where"}, wire)
    assert wire.events()[0]["navigating"] is False
    executing = ros_stubs.GoalStatusArray(status_list=[ros_stubs.GoalStatus(status=2)])
    node.subs["/navigate_through_poses/_action/status"][1](executing)
    wire = Wire()
    node._handle({"cmd": "where"}, wire)
    assert wire.events()[0]["navigating"] is True
    done = ros_stubs.GoalStatusArray(status_list=[ros_stubs.GoalStatus(status=4)])
    node.subs["/navigate_through_poses/_action/status"][1](done)
    assert not node.navigating()
