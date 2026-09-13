"""Who tells the goal server where the cart stands, and what a goal is judged on.

rclpy is faked (``ros_stubs``), so the node is built here exactly as on the board: its clients
answer nobody until a test says the service is there, and its TF buffer holds whatever the test
puts in it. That is enough to write down the two stacks — a saved map with a scan-matching
tracker, and online SLAM with no tracker at all — and to check that a goal is accepted or
refused for the right reason in each.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import ros_stubs

ros_stubs.install()

from pepin_bringup.goal_server import FLAGS, GoalServer  # noqa: E402
from ros_stubs import Float32, Header, Quaternion, TransformStamped, Vector3  # noqa: E402

from pepin.watch import TF_FRESH_S  # noqa: E402

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
    """A goal server on an ephemeral port, its places book in ``tmp_path``: no tracker, no TF,
    no Nav2 — a test adds what its stack has."""
    with ros_stubs.parameters(
        port=0, places=str(tmp_path / "places.yaml"), record_dir=str(tmp_path)
    ):
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


def tracker_says(node: Any, fit: float) -> None:
    """A stack whose tracker is up: its service answers and its fit is on the wire."""
    where = node.service_clients["where_am_i"]
    where.ready = True
    where.response = ros_stubs.Trigger.Response(
        success=True, message=f"x -1.400 m, y 0.800 m, yaw 140.0 deg, fit {fit:.2f}"
    )
    node.service_clients["relocalize"].ready = True
    node.subs["localization_fit"][1](Float32(data=fit))


def test_without_a_tracker_a_fresh_transform_is_the_pose_and_the_goal_goes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Online SLAM: nobody serves /where_am_i and nobody publishes /localization_fit, so the
    pose is map -> base_link and the goal is judged by how fresh that edge is. The first live
    session's goals were all refused here with "the tracker is not up" (2026-09-13 14:05)."""
    node = server(tmp_path)
    standing_at(node, at(1.25, -0.5, 90.0, age_s=0.1))
    pose = node._pose_now()
    assert (round(pose["x"], 3), round(pose["y"], 3)) == (1.25, -0.5)
    assert abs(pose["yaw_deg"] - 90.0) < 1e-6
    assert abs(pose["age_s"] - 0.1) < 1e-3
    assert "fit" not in pose, "there is no fit in this stack, and none is invented"
    ready = node._ready()
    assert ready.ready and not ready.tracker and not ready.search

    wire = Wire()
    node._handle({"cmd": "go", "x": 1.0, "y": 0.3, "yaw_deg": 0.0}, wire)
    # The gate let it through: what stops it now is Nav2's own action server, not the tracker.
    assert [e["detail"] for e in wire.events()] == ["Nav2 is not up"]
    assert not node.service_clients["relocalize"].calls, "no whole-map search was even asked for"


def test_with_neither_a_tracker_nor_a_transform_the_goal_is_refused_with_the_reason(  # type: ignore[no-untyped-def]
    tmp_path,
) -> None:
    """Nothing knows where the cart is: the refusal says which of the two was missing, and no
    search is attempted — a whole-map search is the tracker's own service."""
    node = server(tmp_path)
    ready = node._ready()
    assert not ready.ready and not ready.search
    assert node._pose_now() == {}

    wire = Wire()
    node._handle({"cmd": "go", "x": 1.0, "y": 0.3}, wire)
    (event,) = wire.events()
    assert event["event"] == "error"
    assert "nothing publishes map -> base_link" in event["detail"]
    assert "no tracker" in event["detail"]
    assert not node._client.goals, "nothing was sent to Nav2"


def test_a_transform_that_stopped_coming_is_as_good_as_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """slam_frame re-broadcasts map -> odom at 10 Hz over 50 Hz odometry, so a second-old edge
    means a publisher has stopped. The refusal carries the age, because "stale" and "missing"
    are two different things to go and look at."""
    node = server(tmp_path)
    standing_at(node, at(0.0, 0.0, 0.0, age_s=4.2))
    ready = node._ready()
    assert not ready.ready and not ready.search
    assert "map -> base_link is 4.2 s old" in ready.reason
    standing_at(node, at(0.0, 0.0, 0.0, age_s=TF_FRESH_S - 0.01))
    assert node._ready().ready


def test_where_a_tracker_speaks_nothing_changes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The known-map stack is untouched: the tracker's own pose is the answer, its fit decides,
    and a weak fit still buys the whole-map search instead of driving on a fresh transform."""
    node = server(tmp_path)
    standing_at(node, at(9.9, 9.9, 0.0, age_s=0.05))
    tracker_says(node, 0.71)
    pose = node._pose_now()
    assert (pose["x"], pose["fit"]) == (-1.4, 0.71), "the tracker's, not the transform's"
    assert node._ready() == node._gate.verdict(0.71, None)

    tracker_says(node, 0.31)
    lost = node._ready()
    assert lost.tracker and lost.search and not lost.ready


def test_the_flag_off_is_the_old_node_that_only_ever_asked_the_tracker(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """CLAUDE.md rule 19: the old behaviour stays reachable. With tf_pose off a stack without a
    tracker refuses goals again — the pose is not read from TF and the gate judges the fit that
    never comes."""
    assert FLAGS.flag("tf_pose").default is True and FLAGS.flag("tf_pose").live
    node = server(tmp_path)
    standing_at(node, at(1.0, 1.0, 0.0, age_s=0.05))
    node._switches.set("tf_pose", False)
    assert node._pose_now() == {}
    old = node._ready()
    assert old.tracker and old.search, "it waits for a tracker that will never answer"
    assert "tf_pose=off" in node._switches.state()


def test_a_place_marked_without_a_tracker_carries_no_fit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The session's own book fills in SLAM mode too. A mark is refused on the same evidence a
    goal is, and it writes no fit at all rather than a 0.00 that would read as "marked lost"."""
    node = server(tmp_path)
    standing_at(node, at(2.0, -1.0, -45.0, age_s=0.2))
    answer = node.mark("charger")
    assert answer["event"] == "marked"
    book = json.loads((tmp_path / "places.yaml").read_text())
    assert book["charger"] == {"x": 2.0, "y": -1.0, "yaw_deg": -45.0}

    standing_at(node, at(2.0, -1.0, -45.0, age_s=9.0))
    stale = node.mark("printer")
    assert stale["event"] == "error" and "9.0 s old" in stale["detail"]
    assert "printer" not in json.loads((tmp_path / "places.yaml").read_text())
