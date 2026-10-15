"""Who tells the goal server where the cart stands, and what a goal is judged on.

rclpy is faked (``ros_stubs``), so the node is built here exactly as on the board: its clients
answer nobody until a test says the service is there, and its TF buffer holds whatever the test
puts in it. The pose is ``map -> base_link`` (RTAB-Map owns ``map -> odom``); a goal is judged by
how fresh that edge is and by whether this start of RTAB-Map is placed (test_one_localiser holds
the placement rule itself).
"""

from __future__ import annotations

import contextlib
import itertools
import json
import math
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest
import ros_stubs

ros_stubs.install()


import pepin_bringup.goal_server as goal_server_module  # noqa: E402
from pepin_bringup.goal_server import GoalServer, Run  # noqa: E402
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


class FaceSink:
    def __init__(self) -> None:
        self.said: list[tuple[str, ...]] = []

    def event(self, name: str, *, end: bool = False) -> None:
        self.said.append(("event", name, "end") if end else ("event", name))

    def clear(self) -> None:
        self.said.append(("clear",))

    def lease(self, seconds: float) -> None:
        self.said.append(("lease", str(seconds)))

    def close(self) -> None:
        self.said.append(("close",))


def server(tmp_path: Path, face: FaceSink | None = None) -> Any:
    """A goal server on an ephemeral port, its records in ``tmp_path``: no TF, no placement,
    no Nav2 — a test adds what its stack has; its face (on by default) speaks to ``face``,
    never to the board's head server."""
    sink = face or FaceSink()
    real = goal_server_module.face_client
    goal_server_module.face_client = lambda: sink  # type: ignore[assignment]
    try:
        with ros_stubs.parameters(port=0, record_dir=str(tmp_path)):
            node = GoalServer()
    finally:
        goal_server_module.face_client = real  # type: ignore[assignment]
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
    run = Run("tool", "go", (1.0, 0.3, 0.0), "go (1.00, 0.30, 0 deg)")
    node._goal_handle, node._current = own, run
    wire = Wire()
    node._handle({"cmd": "cancel"}, wire)
    [answer] = wire.events()
    assert answer["had_goal"] is True
    assert answer["navigators"]["navigate_to_pose"] == {"outcome": "accepted", "cancelling": 1}
    assert own.cancelled == 0 and node._goal_handle is None
    assert run.stopped and run.cancelled_for is None, "the operator's: no request named"
    wire = Wire()
    node._handle({"cmd": "cancel"}, wire)
    assert wire.events()[0]["had_goal"] is False, "a run already stopping is no goal to stop"


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


def ready_to_drive(tmp_path: Path, face: FaceSink | None = None) -> Any:
    """A placed node with a fresh pose, Nav2 up, and no recorder to wait for."""
    node = server(tmp_path, face)
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
    assert node._current is None and not node.navigating()


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
    assert node._goal_handle is None and node._current is None


def test_a_cancel_during_the_pivot_stops_the_spin(tmp_path: Path) -> None:
    """The drive ends on position (status 4) and the behaviour server's Spin turns the cart to
    the mark's heading: no navigator owns that goal, so the cancel stops it by its own handle."""
    node = ready_to_drive(tmp_path)
    node._switches.set("controller", "rpp")  # the pair that ends on position alone
    node._client.handle = Goal(status=4)
    node._spin.server = True
    spin = Goal(status=5, on_wait=lambda: node._handle({"cmd": "cancel"}, Wire()))
    node._spin.handle = spin
    wire = Wire()
    node._handle({"cmd": "go", "x": 1.0, "y": 0.3, "yaw_deg": 90.0}, wire)
    assert len(node._spin.goals) == 1, "the pivot started"
    assert spin.cancelled == 1, "and the cancel stopped it"
    assert node._spin_handle is None and not node.navigating()
    assert [e["event"] for e in wire.events()] == ["accepted", "arrival", "pivot", "done"]


# ---- shim_mppi: drive on the shim, park the last metre on MPPI ---------------------------------


class Reached(Pending):
    """Nav2's result future for a goal that drives ``ticks`` turns of the loop and SUCCEEDS."""

    def result(self) -> Any:
        return types.SimpleNamespace(status=4)


def picks(node: Any) -> list[tuple[str, str]]:
    """Every (controller, goal checker) the node handed the tree, in order."""
    controllers = [m.data for m in node.pubs["controller_selector"].sent]
    checkers = [m.data for m in node.pubs["goal_checker_selector"].sent]
    return list(zip(controllers, checkers, strict=True))


def parking_drive(tmp_path: Path, monkeypatch: Any, path: list[tuple[float, float]]) -> Any:
    """A shim_mppi goal to (1.0, 0.0), heading 90 deg, with the cart (facing 0 deg) at each
    point of ``path`` on successive turns of the drive loop, then reached."""
    import pepin_bringup.goal_server as goal_server

    monkeypatch.setattr(goal_server.time, "sleep", lambda _s: None)
    node = ready_to_drive(tmp_path)
    node._spin.server = True
    node._switches.set("controller", "shim_mppi")
    points = iter(path)

    def move() -> None:
        point = next(points, None)
        if point is not None:
            node._tf.buffer.transforms[("map", "base_link")] = at(*point, 0.0, age_s=0.1)

    node._client.handle = Handle(len(path), move)
    node._client.handle._result = Reached(len(path), move)
    standing_at(node, at(*path[0], 0.0, age_s=0.1))
    return node


def test_shim_mppi_hands_the_last_metre_to_mppi_and_starts_the_next_goal_on_the_shim(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The goal starts on the shim; within park_distance_m (1.0) MPPI takes it, once and for
    good (backing out does not hand it back); MPPI ends on the heading itself, so no pivot; the
    next goal finds the shim again."""
    node = parking_drive(tmp_path, monkeypatch, [(-1.0, 0.0), (0.5, 0.0), (-0.5, 0.0), (0.9, 0)])
    before = len(picks(node))
    wire = Wire()
    node._handle({"cmd": "go", "x": 1.0, "y": 0.0, "yaw_deg": 90.0}, wire)
    events = wire.events()
    assert [e["event"] for e in events] == ["accepted", "handover", "arrival", "done"], "no pivot"
    assert events[2]["pivot"] is False
    accepted, handover = events[0], events[1]
    assert accepted["controller"] == "FollowPathShim" and accepted["mode"] == "shim_mppi"
    assert accepted["parks_within_m"] == 1.0
    assert handover == {
        "event": "handover",
        "from": "FollowPathShim",
        "to": "FollowPathMPPI",
        "distance_m": 0.5,
    }
    assert picks(node)[before:] == [
        ("FollowPathMPPI", "general_goal_checker"),
        ("FollowPathShim", "general_goal_checker"),
    ], "one handover, kept when the cart backed out to 1.5 m, and the shim back after the goal"
    assert node._spin.goals == []
    said = node.get_logger().texts("info")
    assert any("(shim_mppi: FollowPathMPPI within 1.00 m)" in line for line in said)
    assert any(
        "FollowPathShim -> FollowPathMPPI (general_goal_checker) parks it" in line for line in said
    )


def test_a_goal_that_starts_within_the_parking_distance_starts_on_mppi(
    tmp_path: Path, monkeypatch: Any
) -> None:
    node = parking_drive(tmp_path, monkeypatch, [(0.4, 0.0)])
    before = len(picks(node))
    sent = node._client.send_goal_async

    def nav2_takes_it(request: Any, feedback: Any = None) -> Any:
        assert node.controller == "FollowPathMPPI", "MPPI is in force before the goal goes"
        return sent(request, feedback)

    node._client.send_goal_async = nav2_takes_it
    wire = Wire()
    node._handle({"cmd": "go", "x": 1.0, "y": 0.0, "yaw_deg": 0.0}, wire)
    assert [e["event"] for e in wire.events()] == ["accepted", "handover", "arrival", "done"]
    assert [c for c, _ in picks(node)[before:]] == ["FollowPathMPPI", "FollowPathShim"]


def test_no_handover_outside_shim_mppi_or_with_the_knob_at_zero(
    tmp_path: Path, monkeypatch: Any
) -> None:
    node = parking_drive(tmp_path, monkeypatch, [(0.0, 0.0), (0.95, 0.0)])
    node._switches.set("park_distance_m", 0.0)
    wire = Wire()
    node._handle({"cmd": "go", "x": 1.0, "y": 0.0, "yaw_deg": 0.0}, wire)
    assert "handover" not in [e["event"] for e in wire.events()]
    node._switches.set("park_distance_m", 1.0)
    node._switches.set("controller", "mppi")
    node._client.handle = Handle(1, None)
    before = len(picks(node))
    wire = Wire()
    node._handle({"cmd": "go", "x": 1.0, "y": 0.0, "yaw_deg": 0.0}, wire)
    assert "handover" not in [e["event"] for e in wire.events()]
    assert picks(node)[before:] == [], "nothing re-published around a plain mppi goal"


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


def test_clearance_tells_the_voice_s_wheels_the_drive_the_cancel_and_the_marks_near_the_cart(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The newest /depth_marks is kept raw and decoded only when asked; its marks come back in
    base_link as published (the slice's own frame), the far ones and the NaN bearings left out,
    aged by the slice's own stamp; the raw lidar scan is never the source; a cancel (ros/stop.sh
    always sends one) is told by its age."""
    stamp = NOW - 0.25
    marks = ros_stubs.LaserScan(
        header=Header(stamp=ros_stubs.Time(sec=int(stamp), nanosec=round(stamp % 1.0 * 1e9)),
                      frame_id="base_link"),
        angle_min=-math.pi, angle_increment=math.pi / 2, range_min=0.05, range_max=3.0,
        ranges=[0.6, float("nan"), 0.5, 2.5],
    )  # fmt: skip
    decoded: list[bytes] = []

    def decode(raw: bytes) -> Any:
        decoded.append(raw)
        return marks

    monkeypatch.setattr(goal_server_module, "scan_message", decode)
    node = server(tmp_path)
    wire = Wire()
    node._handle({"cmd": "clearance"}, wire)
    (empty,) = wire.events()
    assert empty["event"] == "clearance" and empty["marks_age_s"] is None and empty["points"] == []
    assert empty["navigating"] is False and empty["cancel_age_s"] is None
    node.subs[goal_server_module.LIDAR_SCAN_TOPIC][1](b"raw scan")
    wire = Wire()
    node._handle({"cmd": "clearance"}, wire)
    assert wire.events()[0]["points"] == [] and decoded == [], "the lidar's scan is not the source"
    node.subs[goal_server_module.MARKS_TOPIC][1](b"raw marks")
    assert decoded == [], "the marks are not decoded on arrival"
    wire = Wire()
    node._handle({"cmd": "clearance", "radius_m": 1.0}, wire)
    (answer,) = wire.events()
    assert decoded == [b"raw marks"] and answer["marks"] == "ok"
    assert answer["marks_age_s"] == 0.25, "aged by the slice's own stamp"
    behind, ahead = answer["points"]
    assert abs(behind[0] + 0.6) < 1e-3 and abs(behind[1]) < 1e-3, "0.6 m behind the axle"
    assert abs(ahead[0] - 0.5) < 1e-3 and abs(ahead[1]) < 1e-3, "0.5 m ahead"
    node._handle({"cmd": "cancel"}, Wire())
    wire = Wire()
    node._handle({"cmd": "clearance", "radius_m": 3.0}, wire)
    (after,) = wire.events()
    assert after["cancel_age_s"] is not None and after["cancel_age_s"] < 1.0
    assert len(after["points"]) == 3, "the 2.5 m mark within a 3 m radius"


def test_clearance_says_depth_fusion_is_silent(tmp_path: Path) -> None:
    """No marks for MARKS_SILENT_S: the answer names depth_fusion and how long, so the wheels
    refuse with that reason."""
    node = server(tmp_path)
    now = node._started_at + goal_server_module.MARKS_SILENT_S + 1.0
    status = node.clearance(1.0, now)["marks"]
    assert status.startswith("silent 6 s, none since the start") and "depth_fusion" in status
    node._last_marks_at = now - 0.1
    assert node.clearance(1.0, now)["marks"] == "ok"


def test_face_events_follow_a_drive_behind_their_flag(tmp_path: Path, monkeypatch: Any) -> None:
    """On by default: the door is opened at the start. Focused when Nav2 takes the goal, one
    struggle per new recovery, happy at the arrival (ending the focus); the lease rides the
    node's timer; off: the door is cleared and closed."""
    import pepin_bringup.goal_server as goal_server

    sink = FaceSink()
    monkeypatch.setattr(goal_server.time, "sleep", lambda _s: None)
    node = ready_to_drive(tmp_path, sink)
    assert node._switches["face_events"] is True and node._drive_face is not None
    feedback: list[Any] = []
    sent = node._client.send_goal_async

    def takes(goal: Any, callback: Any = None) -> Any:
        feedback.append(callback)
        return sent(goal, callback)

    recoveries = iter([0, 1, 1, 2])

    def tick() -> None:
        count = next(recoveries, 2)
        message = types.SimpleNamespace(distance_remaining=1.0, number_of_recoveries=count)
        feedback[0](types.SimpleNamespace(feedback=message))

    node._client.send_goal_async = takes
    node._client.handle = Handle(4, tick)
    node._client.handle._result = Reached(4, tick)
    node._handle({"cmd": "go", "x": 1.0, "y": 0.0, "yaw_deg": 0.0}, Wire())
    assert sink.said == [
        ("event", "goal_accepted"),
        ("event", "recovery"),
        ("event", "recovery"),
        ("event", "arrived", "end"),
    ]
    lease = next(callback for period, callback in node.timers if period == 2.0)
    lease()
    assert sink.said[-1] == ("lease", "6.0")
    node._switches.set("face_events", False)
    assert sink.said[-2:] == [("clear",), ("close",)] and node._drive_face is None


def test_a_refused_goal_is_a_sad_face_and_a_dropped_report_clears_the_focus(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import pepin_bringup.goal_server as goal_server

    sink = FaceSink()
    monkeypatch.setattr(goal_server.time, "sleep", lambda _s: None)
    node = ready_to_drive(tmp_path, sink)
    refused = Handle(1)
    refused.accepted = False
    node._client.handle = refused
    node._handle({"cmd": "go", "x": 1.0, "y": 0.0, "yaw_deg": 0.0}, Wire())
    assert sink.said == [("event", "goal_failed", "end")]

    class Hangs(Wire):
        def sendall(self, data: bytes) -> None:
            if b'"feedback"' in data or b'"done"' in data:
                raise BrokenPipeError("the caller hung up")
            super().sendall(data)

    sink.said.clear()
    node._client.handle = Handle(1)
    with contextlib.suppress(BrokenPipeError):
        node._handle({"cmd": "go", "x": 1.0, "y": 0.0, "yaw_deg": 0.0}, Hangs())
    assert sink.said[0] == ("event", "goal_accepted") and sink.said[-1] == ("clear",)


# -- plan: would a goal be driven, asked without motion ----------------------------------------


def grid(node: Any, cells: dict[tuple[int, int], int]) -> None:
    """RTAB-Map's latched grid: 20 x 20 cells of 0.1 m from (-1, -1), free unless ``cells``
    (col, row) says otherwise (-1 unknown, 100 occupied)."""
    data = [0] * 400
    for (col, row), value in cells.items():
        data[row * 20 + col] = value
    info = ros_stubs.MapMetaData(resolution=0.1, width=20, height=20)
    info.origin.position.x, info.origin.position.y = -1.0, -1.0
    node.subs["/map"][1](ros_stubs.OccupancyGrid(info=info, data=data))


def costmap(node: Any, monkeypatch: Any, cells: dict[tuple[int, int], int]) -> None:
    """Nav2's global costmap as the node keeps it (raw, decoded when asked): the same 20 x 20
    cells of 0.1 m from (-1, -1), free unless ``cells`` (col, row) says otherwise (100 lethal,
    99 inscribed)."""
    data = [0] * 400
    for (col, row), value in cells.items():
        data[row * 20 + col] = value
    info = ros_stubs.MapMetaData(resolution=0.1, width=20, height=20)
    info.origin.position.x, info.origin.position.y = -1.0, -1.0
    decoded = ros_stubs.OccupancyGrid(info=info, data=data)
    monkeypatch.setattr(goal_server_module, "costmap_message", lambda raw: decoded)
    node.subs[goal_server_module.COSTMAP_TOPIC][1](b"raw costmap")


class PlanHandle:
    """The planner's goal handle: accepted, its result a path through ``poses`` or a code;
    ``last_q`` is the last pose's quaternion (x, y, z, w), the message's identity by default."""

    def __init__(
        self,
        poses: list[tuple[float, float]],
        code: int = 0,
        last_q: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.accepted = True
        path = ros_stubs.Path_(poses=[ros_stubs.PoseStamped() for _ in poses])
        for stamped, (x, y) in zip(path.poses, poses, strict=True):
            stamped.pose.position.x, stamped.pose.position.y = x, y
        if last_q is not None:
            q = path.poses[-1].pose.orientation
            q.x, q.y, q.z, q.w = last_q
        answer = types.SimpleNamespace(path=path, error_code=code, error_msg="")
        self._result = ros_stubs.Future(types.SimpleNamespace(result=answer))

    def get_result_async(self) -> Any:
        return self._result


def ask_plan(node: Any, **request: Any) -> dict[str, Any]:
    wire = Wire()
    node._handle({"cmd": "plan", **request}, wire)
    (event,) = wire.events()
    return dict(event)


def test_a_plan_on_a_free_cell_asks_the_planner_and_never_drives(
    tmp_path: Path, monkeypatch: Any
) -> None:
    node = server(tmp_path)
    grid(node, {})
    costmap(node, monkeypatch, {})
    node._planner.server = True
    node._planner.handle = PlanHandle([(0.0, 0.0), (0.3, 0.4), (0.3, 1.4)])
    answer = ask_plan(node, x=0.3, y=0.4, yaw_deg=90.0)
    assert answer == {
        "event": "plan",
        "x": 0.3,
        "y": 0.4,
        "yaw_deg": 90.0,
        "cell": "free",
        "costmap": "unchecked",
        "ok": True,
        "length_m": 1.5,
        "poses": 3,
        "end": {"x": 0.3, "y": 1.4, "yaw_deg": 0.0},
        "short_m": 1.0,
        "short_deg": -90.0,
    }
    (goal,) = node._planner.goals
    assert (goal.goal.pose.position.x, goal.goal.pose.position.y) == (0.3, 0.4)
    assert goal.goal.header.frame_id == "map"
    assert goal.planner_id == "GridBased" and goal.use_start is False
    assert not node._client.goals, "a plan is never a drive"


def test_a_plan_says_where_it_ends_and_how_far_short_of_the_goal(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Smac Hybrid's taxi (tolerance 1.0): a stand 0.30 m from a seated person's face with their
    legs under the hull is never reached, and the plan ends at the nearest pose it can, here
    0.39 m off and turned 45 deg from the goal's +65 (2026-10-14, run 66: 0.55-0.72 m off,
    -34..+34 deg). The end is the path's last pose with its own heading; a last pose whose
    quaternion is all zeros has none to give."""
    node = server(tmp_path)
    grid(node, {})
    costmap(node, monkeypatch, {})
    node._planner.server = True
    half = math.radians(20.0) / 2.0
    path = [(0.6, -0.8), (0.3, -0.2), (0.0, 0.3)]
    node._planner.handle = PlanHandle(path, last_q=(0.0, 0.0, math.sin(half), math.cos(half)))
    answer = ask_plan(node, x=0.25, y=0.6, yaw_deg=65.0)
    assert answer["ok"] is True and answer["poses"] == 3
    assert answer["end"] == {"x": 0.0, "y": 0.3, "yaw_deg": 20.0}
    assert (answer["short_m"], answer["short_deg"]) == (0.391, -45.0)
    node._planner.handle = PlanHandle(path, last_q=(0.0, 0.0, 0.0, 0.0))
    answer = ask_plan(node, x=0.25, y=0.6, yaw_deg=65.0)
    assert answer["end"] == {"x": 0.0, "y": 0.3, "yaw_deg": None}
    assert (answer["short_m"], answer["short_deg"]) == (0.391, None)
    node._planner.handle = PlanHandle([(0.6, -0.8), (0.25, 0.6)], last_q=(0.0, 0.0, 1.0, 0.0))
    answer = ask_plan(node, x=0.25, y=0.6, yaw_deg=-170.0)
    assert answer["end"] == {"x": 0.25, "y": 0.6, "yaw_deg": 180.0}
    assert (answer["short_m"], answer["short_deg"]) == (0.0, -10.0), "wrapped across 180"


def test_an_unknown_occupied_or_outside_cell_is_refused_before_any_plan(tmp_path: Path) -> None:
    node = server(tmp_path)
    grid(node, {(13, 14): -1, (5, 5): 100})
    node._planner.server = True
    cases = (((0.35, 0.45), "unknown"), ((-0.45, -0.45), "occupied"), ((5.0, 0.0), "outside"))
    for (x, y), cell in cases:
        answer = ask_plan(node, x=x, y=y)
        assert (answer["cell"], answer["ok"], answer["blames"]) == (cell, False, "goal"), cell
    assert not node._planner.goals


def test_without_a_map_no_goal_is_called_free(tmp_path: Path) -> None:
    node = server(tmp_path)
    answer = ask_plan(node, x=0.0, y=0.0)
    assert (answer["cell"], answer["ok"], answer["blames"]) == ("no map", False, "planner")


def test_a_goal_inside_a_person_s_inflated_footprint_never_reaches_the_planner(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A person's legs marked lethal at (0.55, 0.45) on Nav2's costmap: the cart at (0.45, 0.45)
    facing them would cover them with its bumper, at (0.75, 0.45) facing away with its rear
    (0.30 m behind base_link); neither reaches the planner. At (0.15, 0.45) facing them the hull
    is clear (an inscribed cell under it is the planner's to judge) and the plan is asked. No
    costmap at all is the planner's refusal, not the goal's."""
    node = server(tmp_path)
    grid(node, {})
    node._planner.server = True
    node._planner.handle = PlanHandle([(0.0, 0.0), (0.15, 0.45)])
    # the flag is OFF as shipped (2026-10-13 evening): the planner alone judges the goal
    costmap(node, monkeypatch, {(15, 14): 100, (11, 14): 99})
    answer = ask_plan(node, x=0.45, y=0.45, yaw_deg=0.0)
    assert (answer["costmap"], answer["ok"]) == ("unchecked", True)
    assert len(node._planner.goals) == 1, "the lethal hull reached the planner: its to judge"
    node._planner.goals.clear()
    node._switches.set("goal_hull_check", True)
    node._costmap = None  # the check on, no costmap heard yet: the planner's refusal
    answer = ask_plan(node, x=0.45, y=0.45, yaw_deg=0.0)
    assert (answer["costmap"], answer["ok"], answer["blames"]) == ("no costmap", False, "planner")
    costmap(node, monkeypatch, {(15, 14): 100, (11, 14): 99})
    for x, yaw in ((0.45, 0.0), (0.75, 0.0)):
        answer = ask_plan(node, x=x, y=0.45, yaw_deg=yaw)
        assert (answer["costmap"], answer["ok"], answer["blames"]) == ("lethal", False, "goal"), x
        assert "would cover 1 lethal cells" in answer["reason"]
        assert "inside an obstacle's footprint" in answer["reason"]
    assert not node._planner.goals, "neither reached the planner"
    answer = ask_plan(node, x=0.15, y=0.45, yaw_deg=0.0)
    assert (answer["costmap"], answer["ok"]) == ("clear", True)
    assert len(node._planner.goals) == 1


def test_a_planner_refusal_says_whose_it_is(tmp_path: Path, monkeypatch: Any) -> None:
    node = server(tmp_path)
    grid(node, {})
    costmap(node, monkeypatch, {})
    node._planner.server = True
    node._planner.handle = PlanHandle([], code=206)
    answer = ask_plan(node, x=0.3, y=0.4)
    assert (answer["ok"], answer["blames"], answer["reason"]) == (False, "goal", "goal occupied")
    node._planner.handle = PlanHandle([], code=205)  # start occupied: no other goal fares better
    answer = ask_plan(node, x=0.3, y=0.4)
    assert (answer["ok"], answer["blames"]) == (False, "planner")


def test_no_plan_while_driving_and_none_without_a_planner(tmp_path: Path, monkeypatch: Any) -> None:
    node = server(tmp_path)
    grid(node, {})
    costmap(node, monkeypatch, {})
    assert ask_plan(node, x=0.3, y=0.4) == {
        "event": "error",
        "detail": "the planner server is not up",
    }
    executing = ros_stubs.GoalStatusArray(status_list=[ros_stubs.GoalStatus(status=2)])
    node.subs["/navigate_to_pose/_action/status"][1](executing)
    assert "a drive is under way" in ask_plan(node, x=0.3, y=0.4)["detail"]
    assert "numbers x and y" in ask_plan(node, x="here")["detail"]


# -- spin: a turn in place, the brain's turn tool -------------------------------------------------


class Turned(Pending):
    """The Spin's result future: it runs ``ticks`` turns of the loop, then ends with ``status``."""

    def __init__(self, ticks: int, on_tick: Any, status: int) -> None:
        super().__init__(ticks, on_tick)
        self.status = status

    def result(self) -> Any:
        return types.SimpleNamespace(status=self.status)


def spinning(node: Any, to_yaw_deg: float, status: int = 4, on_tick: Any = None) -> Handle:
    """The behaviour server up, its Spin turning the cart (facing 0 deg) to ``to_yaw_deg`` over
    three turns of the loop and ending with ``status``."""
    node._spin.server = True
    steps = iter((to_yaw_deg / 3.0, 2.0 * to_yaw_deg / 3.0, to_yaw_deg))

    def turn() -> None:
        yaw = next(steps, to_yaw_deg)
        node._tf.buffer.transforms[("map", "base_link")] = at(0.0, 0.0, yaw, age_s=0.1)
        if on_tick is not None:
            on_tick()

    handle = Handle(3, turn)
    handle._result = Turned(3, turn, status)
    node._spin.handle = handle
    return handle


def test_a_spin_is_one_spin_goal_answered_like_a_drive_with_the_turn_measured(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """`spin` 90 deg left: one goal on the behaviour server (90 deg, 15 s), nothing to the
    navigators; accepted with the run and the pose it starts from, arrival at Nav2's word and
    done after the tape (both with Nav2's status, the pose after it and the turn on map ->
    base_link: 88 of 90, within the goal checker's 11.5); the face shows it as a drive;
    nothing is left running."""
    monkeypatch.setattr(goal_server_module.time, "sleep", lambda _s: None)
    sink = FaceSink()
    node = ready_to_drive(tmp_path, sink)
    spinning(node, 88.0)
    wire = Wire()
    node._handle({"cmd": "spin", "deg": 90}, wire)
    (goal,) = node._spin.goals
    assert goal.target_yaw == math.radians(90.0) and goal.time_allowance.sec == 15
    assert node._client.goals == [], "no navigator is asked"
    accepted, arrival, done = wire.events()
    assert accepted["event"] == "accepted" and accepted["deg"] == 90.0
    assert accepted["source"] == "tool", "a request naming no source is a person's"
    assert (accepted["pose"], accepted["x"], accepted["y"]) == ("tf", 0.0, 0.0)
    assert done["event"] == "done" and done["status"] == 4 and "detail" not in done
    assert (done["turned_deg"], done["short_deg"]) == (88.0, 2.0)
    assert round(done["arrival"]["yaw_deg"], 3) == 88.0
    assert arrival["event"] == "arrival" and arrival["status"] == 4
    assert (arrival["turned_deg"], arrival["arrival"]) == (88.0, done["arrival"])
    assert sink.said == [("event", "goal_accepted"), ("event", "arrived", "end")]
    assert node._spin_handle is None and node._current is None and not node.navigating()
    assert any("spin +90 deg: status 4" in line for line in node.get_logger().texts("info"))


def test_a_spin_short_of_the_goal_checker_s_tolerance_says_so_and_the_turn_unwraps() -> None:
    """Nav2 may call a spin done that the map's heading says fell short: past the goal
    checker's 11.5 deg the answer carries why. A turn past half a turn is measured toward the
    ask, not the wrap; without a heading at either end nothing is claimed."""
    measured = GoalServer.turn_measured
    short = measured(90.0, {"yaw_deg": 0.0}, {"yaw_deg": 60.0}, 4)
    assert (short["turned_deg"], short["short_deg"]) == (60.0, 30.0)
    assert "turned +60 of +90 deg" in short["detail"] and "30 deg short" in short["detail"]
    right = measured(-90.0, {"yaw_deg": 170.0}, {"yaw_deg": 85.0}, 4)
    assert (right["turned_deg"], right["short_deg"]) == (-85.0, -5.0) and "detail" not in right
    whole = measured(270.0, {"yaw_deg": 0.0}, {"yaw_deg": -90.0}, 4)
    assert (whole["turned_deg"], whole["short_deg"]) == (270.0, 0.0)
    assert "detail" not in measured(90.0, {"yaw_deg": 0.0}, {"yaw_deg": 60.0}, 6), "Nav2's word"
    assert measured(90.0, {}, {"yaw_deg": 60.0}, 4) == {}


def test_a_spin_is_refused_behind_its_flag_without_an_angle_or_while_anything_drives(
    tmp_path: Path,
) -> None:
    node = ready_to_drive(tmp_path)
    node._spin.server = True

    def ask(**request: Any) -> str:
        wire = Wire()
        node._handle({"cmd": "spin", **request}, wire)
        (event,) = wire.events()
        assert event["event"] == "error", event
        return str(event["detail"])

    node._switches.set("spin_requests", False)
    assert "spin_requests" in ask(deg=90)
    node._switches.set("spin_requests", True)
    assert "spin needs deg" in ask()
    assert "non-zero" in ask(deg=0) and "within +-360" in ask(deg=400)
    assert "source must be one of tool, gaze" in ask(deg=90, source="operator")
    executing = ros_stubs.GoalStatusArray(status_list=[ros_stubs.GoalStatus(status=2)])
    node.subs["/navigate_to_pose/_action/status"][1](executing)
    assert ask(deg=90) == "a goal runs on Nav2: cancel first", "someone else's goal"
    assert node._current is None, "the refused spin is no run"
    idle = ros_stubs.GoalStatusArray(status_list=[ros_stubs.GoalStatus(status=4)])
    node.subs["/navigate_to_pose/_action/status"][1](idle)
    node._spin.server = False
    assert ask(deg=90) == "not turning: no spin behaviour"
    assert node._spin.goals == [] and node._current is None


def test_a_cancel_during_a_spin_stops_it_and_the_spin_ends_cancelled(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """No navigator owns the Spin's goal: the operator's cancel stops it by its own handle, and
    the answer is Nav2's CANCELED with the part of the turn that was made."""
    monkeypatch.setattr(goal_server_module.time, "sleep", lambda _s: None)
    node = ready_to_drive(tmp_path)
    cancelled: list[bool] = []

    def operator_cancels() -> None:
        if not cancelled:
            cancelled.append(True)
            node._handle({"cmd": "cancel"}, Wire())

    handle = spinning(node, 30.0, status=5, on_tick=operator_cancels)
    wire = Wire()
    node._handle({"cmd": "spin", "deg": -120}, wire)
    assert handle.cancelled == 1, "the cancel reached the spin"
    accepted, done = wire.events()
    assert accepted["event"] == "accepted" and done["status"] == 5
    assert "detail" not in done, "a cancelled spin is not judged by the goal checker"
    assert node._spin_handle is None and not node.navigating()


# -- one owner of motion, one rule for a new request ------------------------------------------


class Cancelled:
    """Nav2's result future for a goal that runs until its handle is cancelled, then ends
    CANCELED; ``on_tick`` runs at the top of every turn of the loop that waits for it."""

    def __init__(self, handle: Until, on_tick: Any = None) -> None:
        self._handle, self._on_tick = handle, on_tick

    def done(self) -> bool:
        if self._on_tick is not None:
            self._on_tick()
        return self._handle.cancelled > 0

    def add_done_callback(self, callback: Any) -> None:
        callback(self)

    def result(self) -> Any:
        return types.SimpleNamespace(status=5 if self._handle.cancelled else 4)


class Until:
    """A goal handle Nav2 accepted that runs until it is cancelled."""

    def __init__(self, on_tick: Any = None) -> None:
        self.accepted, self.cancelled = True, 0
        self._result = Cancelled(self, on_tick)

    def get_result_async(self) -> Cancelled:
        return self._result

    def cancel_goal_async(self) -> None:
        self.cancelled += 1


def reaching(ticks: int, on_tick: Any = None) -> Handle:
    """A goal handle that drives ``ticks`` turns of the loop and SUCCEEDS."""
    handle = Handle(ticks, on_tick)
    handle._result = Reached(ticks, on_tick)
    return handle


def numbered(node: Any, first: int = 7) -> None:
    """Every run's tape confirmed under the next number from ``first`` (the recorder, faked)."""
    numbers = itertools.count(first)
    node._runs = types.SimpleNamespace(run=first - 1)

    def start(name: str) -> Path:
        node._runs.run = next(numbers)
        return Path(f"/maps/rec/{node._runs.run:04d}_{name}.jsonl")

    node.start_recording = start


def short_sleeps(monkeypatch: Any) -> None:
    """The loops' 50 ms sleeps as 1 ms: real enough for a second request's thread to run."""
    real = time.sleep
    monkeypatch.setattr(goal_server_module.time, "sleep", lambda _s: real(0.001))


def test_one_target_is_one_source_s_pose_within_the_goal_checker_s_tolerance() -> None:
    """The same target: what the running goal's arrival would count as reached (10 cm, 11.5
    deg, the goal checker's), asked by the same source; a turn within 11.5 deg of the other."""
    printer = Run("tool", "go", (1.0, 0.3, 90.0), "go printer")
    assert printer.same(Run("tool", "go", (1.08, 0.3, 100.0), "go (1.08, 0.30, 100 deg)"))
    assert printer.same(Run("tool", "go", (1.0, 0.3, -270.0), "go (1.00, 0.30, -270 deg)"))
    assert not printer.same(Run("tool", "go", (1.15, 0.3, 90.0), "15 cm off"))
    assert not printer.same(Run("tool", "go", (1.0, 0.3, 105.0), "15 deg off"))
    assert not printer.same(Run("gaze", "go", (1.0, 0.3, 90.0), "another source"))
    assert not printer.same(Run("tool", "spin", (90.0,), "spin +90 deg"))
    turn = Run("gaze", "spin", (120.0,), "spin +120 deg")
    assert turn.same(Run("gaze", "spin", (110.0,), "spin +110 deg"))
    assert not turn.same(Run("gaze", "spin", (100.0,), "spin +100 deg"))


def test_the_same_target_asked_while_it_runs_is_already_under_way_with_no_new_goal(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A person's go runs (run 7): the same go asked again, and then a pose 5 cm and 5 deg
    from it, are answered `already` with run 7; Nav2 has one goal, nothing is cancelled, and
    the drive's own end goes to its own requester."""
    monkeypatch.setattr(goal_server_module.time, "sleep", lambda _s: None)
    node = ready_to_drive(tmp_path)
    numbered(node)
    asks = iter(({"x": 1.0, "y": 0.3}, {"x": 1.05, "y": 0.3, "yaw_deg": 5.0, "source": "tool"}))
    answers: list[list[dict[str, Any]]] = []

    def asked_again() -> None:
        ask = next(asks, None)
        if ask is not None:
            wire = Wire()
            node._handle({"cmd": "go", **ask}, wire)
            answers.append(wire.events())

    handle = reaching(3, asked_again)
    node._client.handle = handle
    wire = Wire()
    node._handle({"cmd": "go", "x": 1.0, "y": 0.3}, wire)
    assert len(node._client.goals) == 1 and handle.cancelled == 0
    assert [e["event"] for e in wire.events()] == ["accepted", "arrival", "done"]
    for (already,) in answers:
        assert already == {
            "event": "already",
            "run": 7,
            "source": "tool",
            "target": "go (1.00, 0.30, 0 deg)",
            "detail": "already under way: go (1.00, 0.30, 0 deg) (run 7)",
        }
    assert len(answers) == 2 and node._current is None


def test_another_target_cancels_the_run_under_way_and_starts_once_it_has_ended(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A go to another point while run 7 drives: run 7's goal is cancelled by its handle and
    its own end says what it was cancelled for; the new goal goes to Nav2 only after that end,
    its answer naming the run it replaced (run 8 then drives to its arrival)."""
    short_sleeps(monkeypatch)
    node = ready_to_drive(tmp_path)
    numbered(node)
    first_wire, second_wire = Wire(), Wire()
    second = reaching(2)
    threads: list[threading.Thread] = []

    def another_asked() -> None:
        if not threads:
            node._client.handle = second
            request = {"cmd": "go", "x": -2.0, "y": 1.0, "yaw_deg": 90.0}
            threads.append(threading.Thread(target=node._handle, args=(request, second_wire)))
            threads[0].start()

    first = Until(another_asked)
    node._client.handle = first
    sent = node._client.send_goal_async
    heard_at_send: list[list[str]] = []

    def send(goal: Any, feedback: Any = None) -> Any:
        heard_at_send.append([e["event"] for e in first_wire.events()])
        return sent(goal, feedback)

    node._client.send_goal_async = send
    node._handle({"cmd": "go", "x": 1.0, "y": 0.3, "yaw_deg": 0.0}, first_wire)
    threads[0].join(5.0)
    assert not threads[0].is_alive()
    assert first.cancelled == 1 and len(node._client.goals) == 2
    assert heard_at_send == [[], ["accepted", "done"]], "the new goal after the old one's end"
    accepted, done = first_wire.events()
    assert (accepted["run"], done["status"]) == (7, 5)
    assert done["cancelled_for"] == "go (-2.00, 1.00, 90 deg) (tool)"
    replaced, *rest = second_wire.events()
    assert replaced == {
        "event": "replaced",
        "run": 7,
        "source": "tool",
        "target": "go (1.00, 0.30, 0 deg)",
        "cancelled_run": 7,
        "started": True,
    }
    assert [e["event"] for e in rest] == ["accepted", "arrival", "done"]
    assert rest[0]["run"] == 8 and "cancelled_for" not in rest[-1]
    assert node._current is None and node._next is None


def test_the_gaze_s_turn_is_refused_while_a_person_s_drive_runs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The speaker gaze asks for its own turn while run 7 (a tool's) drives: refused, "a drive
    runs"; no spin goal, the drive untouched (2026-10-14, session 185350: the other way round,
    the gaze's spin had the goal server refuse go_to(home))."""
    monkeypatch.setattr(goal_server_module.time, "sleep", lambda _s: None)
    node = ready_to_drive(tmp_path)
    numbered(node)
    node._spin.server = True
    answers: list[list[dict[str, Any]]] = []

    def gaze_asks() -> None:
        if not answers:
            wire = Wire()
            node._handle({"cmd": "spin", "deg": 120.0, "source": "gaze"}, wire)
            answers.append(wire.events())

    handle = reaching(3, gaze_asks)
    node._client.handle = handle
    wire = Wire()
    node._handle({"cmd": "go", "x": 1.0, "y": 0.3}, wire)
    assert answers == [
        [
            {
                "event": "error",
                "detail": "a drive runs (run 7: go (1.00, 0.30, 0 deg) (tool)): the gaze's turn"
                " waits",
            }
        ]
    ]
    assert node._spin.goals == [] and handle.cancelled == 0
    assert [e["event"] for e in wire.events()] == ["accepted", "arrival", "done"]


def test_a_person_s_drive_cancels_the_gaze_s_turn_whose_end_says_what_for(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The gaze's own spin (run 7) turns when a tool's go comes: the spin is cancelled by its
    handle, its end (CANCELED) names the go it gave way to, and the go starts after it, its
    answer naming the gaze's run it replaced."""
    short_sleeps(monkeypatch)
    node = ready_to_drive(tmp_path)
    numbered(node)
    node._spin.server = True
    drive_wire = Wire()
    threads: list[threading.Thread] = []

    def tool_asks() -> None:
        if not threads:
            node._client.handle = reaching(2)
            request = {"cmd": "go", "x": 1.0, "y": 0.3, "source": "tool"}
            threads.append(threading.Thread(target=node._handle, args=(request, drive_wire)))
            threads[0].start()

    spin = Until(tool_asks)
    node._spin.handle = spin
    wire = Wire()
    node._handle({"cmd": "spin", "deg": 120.0, "source": "gaze"}, wire)
    threads[0].join(5.0)
    assert spin.cancelled == 1
    accepted, done = wire.events()
    assert accepted["source"] == "gaze" and accepted["run"] == 7
    assert done["status"] == 5 and done["cancelled_for"] == "go (1.00, 0.30, 0 deg) (tool)"
    replaced, *rest = drive_wire.events()
    assert (replaced["event"], replaced["source"]) == ("replaced", "gaze")
    assert (replaced["cancelled_run"], replaced["target"]) == (7, "spin +120 deg")
    assert [e["event"] for e in rest] == ["accepted", "arrival", "done"]
    assert node._current is None and node._spin_handle is None


def test_the_arrival_is_said_the_moment_nav2_reports_it_and_the_pivot_comes_after(
    tmp_path: Path,
) -> None:
    """The RPP pair ends on position: `arrival` (status 4, the pose) goes out the moment
    Nav2's result comes, before the pivot's Spin is even awaited; the pivot and `done` follow,
    and the drive is under way until `done`."""
    node = ready_to_drive(tmp_path)
    node._switches.set("controller", "rpp")
    node._client.handle = Goal(status=4)
    node._spin.server = True
    wire = Wire()
    during_pivot: list[tuple[list[str], bool]] = []

    def pivot_turns() -> None:
        during_pivot.append(([e["event"] for e in wire.events()], node._current is not None))

    node._spin.handle = Goal(status=4, on_wait=pivot_turns)
    node._handle({"cmd": "go", "x": 1.0, "y": 0.3, "yaw_deg": 90.0}, wire)
    assert during_pivot == [(["accepted", "arrival"], True)]
    events = wire.events()
    assert [e["event"] for e in events] == ["accepted", "arrival", "pivot", "done"]
    arrival = events[1]
    assert arrival["status"] == 4 and arrival["pivot"] is True
    assert (arrival["arrival"]["x"], arrival["arrival"]["y"]) == (0.0, 0.0)


def test_a_turn_in_place_never_blocks_a_plan_and_a_drive_does(tmp_path: Path) -> None:
    """A plan races only a navigator's own planning: during the gaze's spin approach may plan
    (and its go then replaces the spin); during a go it may not."""
    node = server(tmp_path)
    node._current = Run("gaze", "spin", (120.0,), "spin +120 deg")
    assert ask_plan(node, x=0.3, y=0.4)["event"] == "plan"
    node._current = Run("tool", "go", (1.0, 0.3, 0.0), "go printer")
    assert "a drive is under way" in ask_plan(node, x=0.3, y=0.4)["detail"]


def test_a_stopped_run_nav2_never_answers_lets_go_and_frees_the_node(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A Nav2 restarted mid-goal never answers the cancel: the stopped run waits STOPPED_WAIT_S
    for it, then ends with that said, so the next request is not held by it for good."""
    monkeypatch.setattr(goal_server_module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(goal_server_module, "STOPPED_WAIT_S", 0.0)
    node = ready_to_drive(tmp_path)
    deaf = Until(lambda: node.cancel() if node._goal_handle is not None else None)
    deaf.cancel_goal_async = lambda: None  # type: ignore[method-assign]
    node._client.handle = deaf
    wire = Wire()
    node._handle({"cmd": "go", "x": 1.0, "y": 0.3}, wire)
    accepted, done = wire.events()
    assert accepted["event"] == "accepted" and done["status"] == 0
    assert done["detail"] == goal_server_module.UNANSWERED
    assert node._current is None and not node.navigating()


# -- pose_at: where a frame was at a stamp, out of the one TF history --------------------------


class ExtrapolationException(Exception):  # noqa: N818 — tf2's own name, whose kind is answered
    """tf2's: the history does not cover that moment."""


class TurningHistory:
    """A TF buffer of a cart turning in place at the map's origin, 35 deg/s to the right, its
    camera link 1.2 m up and turned with it; the history ends 20 ms before the node's now (the
    chain's newest edge) and begins 10 s before that. Every lookup is kept."""

    NEWEST = NOW - 0.02

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float, float]] = []

    def lookup_transform(self, target: str, source: str, time: Any, timeout: Any = None) -> Any:
        seconds = time.nanoseconds * 1e-9
        self.calls.append((target, source, seconds, timeout.nanoseconds * 1e-9))
        at_s = seconds or self.NEWEST  # time zero is "the newest there is"
        if at_s > self.NEWEST:
            raise ExtrapolationException("Lookup would require extrapolation into the future")
        yaw = math.radians(35.0 * (self.NEWEST - at_s))  # turning right: more left back then
        return TransformStamped(
            header=Header(stamp=_stamp(at_s), frame_id=target),
            child_frame_id=source,
            transform=ros_stubs.Transform(
                translation=Vector3(z=1.2 if source == "camera_link" else 0.0),
                rotation=Quaternion(z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0)),
            ),
        )


def _stamp(seconds: float) -> Any:
    whole = math.floor(seconds)
    return ros_stubs.Time(sec=whole, nanosec=round((seconds - whole) * 1e9))


def ask_pose_at(node: Any, **request: Any) -> dict[str, Any]:
    wire = Wire()
    node._handle({"cmd": "pose_at", **request}, wire)
    (answer,) = wire.events()
    return answer


def turning(tmp_path: Path) -> tuple[Any, TurningHistory]:
    node = server(tmp_path)
    node._tf_pose()  # the listener starts on the first ask, as on the robot
    history = TurningHistory()
    node._tf.buffer = history
    return node, history


def test_pose_at_answers_map_to_the_frame_at_the_stamp_asked(tmp_path: Path) -> None:
    """A frame exposed a second ago is answered with the camera where it was then (35 deg
    left of where it is now), aged on the node's clock; the lookup is TF's at that stamp."""
    node, history = turning(tmp_path)
    then = ask_pose_at(node, stamp=NOW - 1.0, frame="camera_link")
    assert then["ok"] and then["event"] == "pose_at" and then["frame"] == "camera_link"
    assert then["yaw_deg"] == pytest.approx(35.0 * (1.0 - 0.02))
    assert (then["x"], then["y"], then["z"]) == pytest.approx((0.0, 0.0, 1.2))
    yaw = math.radians(then["yaw_deg"])
    assert (then["qz"], then["qw"]) == pytest.approx((math.sin(yaw / 2), math.cos(yaw / 2)))
    assert (then["stamp"], then["now"], then["age_s"]) == pytest.approx((NOW - 1.0, NOW, 1.0))
    ((target, source, seconds, wait),) = history.calls
    assert (target, source, wait) == ("map", "camera_link", goal_server_module.POSE_AT_WAIT_S)
    assert seconds == pytest.approx(NOW - 1.0, abs=1e-6)
    later = ask_pose_at(node, stamp=NOW - 0.5, frame="base_link")
    assert later["yaw_deg"] == pytest.approx(35.0 * (0.5 - 0.02)) and later["z"] == 0.0
    assert then["parent"] == later["parent"] == "map"
    moved = ask_pose_at(node, stamp=NOW - 1.0, frame="camera_link", parent="odom")
    assert moved["ok"] and moved["parent"] == "odom"
    # how the camera moved is asked through odom: smooth across a localisation update
    assert history.calls[-1][:2] == ("odom", "camera_link")


def test_pose_at_without_a_stamp_is_the_newest_with_its_own_stamp_and_age(tmp_path: Path) -> None:
    node, history = turning(tmp_path)
    newest = ask_pose_at(node, frame="base_link")
    assert newest["ok"] and history.calls[0][2] == 0.0, "time zero: tf2's newest"
    assert newest["stamp"] == pytest.approx(TurningHistory.NEWEST)
    assert newest["age_s"] == pytest.approx(0.02) and newest["yaw_deg"] == pytest.approx(0.0)
    assert ask_pose_at(node)["frame"] == "base_link", "the cart's frame by default"


def test_pose_at_refuses_a_stamp_older_than_the_history_without_looking(tmp_path: Path) -> None:
    node, history = turning(tmp_path)
    old = ask_pose_at(node, stamp=NOW - 12.0, frame="camera_link")
    assert old["ok"] is False and old["kind"] == "too_old"
    assert old["reason"] == "the stamp is 12.0 s old and TF keeps 10 s of history"
    assert old["age_s"] == pytest.approx(12.0)
    assert history.calls == [], "never looked up"


def test_pose_at_says_tf2_s_refusal_a_wrong_frame_and_clocks_that_disagree(
    tmp_path: Path,
) -> None:
    node, history = turning(tmp_path)
    future = ask_pose_at(node, stamp=NOW + 0.3, frame="camera_link")
    assert (future["ok"], future["kind"]) == (False, "Extrapolation")
    assert "extrapolation into the future" in future["reason"]
    assert len(history.calls) == 1, "looked up, waited, refused by tf2"
    odom = ask_pose_at(node, stamp=NOW - 1.0, frame="odom")
    assert (odom["kind"], odom["reason"]) == ("frame", "frame is base_link or camera_link")
    parent = ask_pose_at(node, stamp=NOW - 1.0, frame="camera_link", parent="base_link")
    assert (parent["kind"], parent["reason"]) == ("parent", "parent is map or odom")
    ahead = ask_pose_at(node, stamp=NOW + 5.0, frame="camera_link")
    assert ahead["kind"] == "ahead" and "ros/time.sh offset" in ahead["reason"]
    assert len(history.calls) == 1
    assert ask_pose_at(node, stamp="soon")["event"] == "error"
