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

from pepin_bringup.goal_server import (  # noqa: E402
    CORRECTION_TOPIC,
    FLAGS,
    TRACKER_WAIT_S,
    GoalServer,
)
from ros_stubs import (  # noqa: E402
    Float32,
    Header,
    Quaternion,
    String,
    TransformStamped,
    Vector3,
)

from pepin.watch import (  # noqa: E402
    BY_FIT,
    BY_SIGMA,
    CORRECTION_FRESH_S,
    SIGMA_TOPIC,
    SOURCE_PATIENCE_S,
    TF_FRESH_S,
    BlindDriveWatch,
    Sigma,
)

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


def correction_landed(node: Any, age_s: float = 0.0) -> None:
    """The SLAM correction arrived ``age_s`` seconds ago: the pulse of the half of the stack
    that owns the pose in SLAM mode (rtabmap_frame publishes it at 10 Hz, graph or no graph)."""
    node.clock.seconds = NOW - age_s
    node.subs[CORRECTION_TOPIC][1](TransformStamped())
    node.clock.seconds = NOW


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
    correction_landed(node)
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
    correction_landed(node)
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
    correction_landed(node)
    answer = node.mark("charger")
    assert answer["event"] == "marked"
    book = json.loads((tmp_path / "places.yaml").read_text())
    assert book["charger"] == {"x": 2.0, "y": -1.0, "yaw_deg": -45.0}

    standing_at(node, at(2.0, -1.0, -45.0, age_s=9.0))
    stale = node.mark("printer")
    assert stale["event"] == "error" and "9.0 s old" in stale["detail"]
    assert "printer" not in json.loads((tmp_path / "places.yaml").read_text())


def test_a_goal_is_refused_when_the_correction_stopped_though_the_edge_is_fresh(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The hole the transform cannot see. slam_frame re-broadcasts the LAST correction at 10 Hz
    with a fresh stamp, so with the laptop gone map -> base_link is still 0.1 s old: the gate
    passed, Nav2's costmaps (0.3 s tolerance on that same edge) never timed out, and the cart
    would have driven a map that stopped growing. The correction's own age is the evidence."""
    node = server(tmp_path)
    standing_at(node, at(1.0, 0.5, 0.0, age_s=0.1))
    correction_landed(node, age_s=9.0)
    refused = node._ready()
    assert not refused.ready and not refused.search and not refused.tracker
    assert "the SLAM correction stopped 9.0 s ago" in refused.reason

    wire = Wire()
    node._handle({"cmd": "go", "x": 1.0, "y": 0.3}, wire)
    (event,) = wire.events()
    assert event["event"] == "error" and "SLAM correction stopped" in event["detail"]
    assert not node._client.goals, "nothing was sent to Nav2"

    correction_landed(node, age_s=CORRECTION_FRESH_S)
    assert node._ready().ready, "the bound is allowed; a WiFi hiccup is not a dead laptop"


def test_a_goal_before_the_slam_half_is_up_is_refused_by_name(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Nothing has ever arrived on /map_odom: the edge is identity, which is the honest map at
    the start of a session and tells the operator nothing. The refusal names what is missing."""
    node = server(tmp_path)
    standing_at(node, at(0.0, 0.0, 0.0, age_s=0.05))
    never = node._ready()
    assert not never.ready and "no SLAM correction has ever arrived" in never.reason


def test_the_correction_watch_off_is_the_transform_alone(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """CLAUDE.md rule 19: the old behaviour stays one flag away. Off, a goal rests on the age of
    map -> base_link, the way the first SLAM session drove."""
    assert FLAGS.flag("correction_watch").default is True
    assert FLAGS.flag("correction_watch").live
    node = server(tmp_path)
    standing_at(node, at(1.0, 0.5, 0.0, age_s=0.1))
    correction_landed(node, age_s=9.0)
    assert not node._ready().ready
    node._switches.set("correction_watch", False)
    assert node._ready().ready
    assert "correction_watch=off" in node._switches.state()


def test_a_drive_is_cut_when_the_correction_dies_under_it(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The SLAM analogue of the blind-drive watch. Nav2 is no backstop here — it reads the same
    re-broadcast edge and would keep following the plan by dead reckoning — so the drive stops
    the moment the pulse of the half that owns the pose stops, with no search to fall back on."""
    node = server(tmp_path)
    standing_at(node, at(0.0, 0.0, 0.0, age_s=0.1))
    correction_landed(node)
    # The laptop goes away with the goal already running: the gate saw a live pulse, and the
    # clock moves nine seconds past the last correction on the drive loop's first turn.
    handle = nav2_answers(node, ticks=20, on_tick=lambda: setattr(node.clock, "seconds", NOW + 9.0))
    wire = Wire()
    node._go({"cmd": "go", "x": 1.0, "y": 0.0}, wire, record=tmp_path / "run.jsonl")

    events = {event["event"]: event for event in wire.events()}
    assert events["accepted"]["pose"] == "tf"
    assert events["lost"]["correction_s"] == 9.0, "and it says how long the silence was"
    assert "the SLAM correction stopped" in events["done"]["detail"]
    assert handle.cancelled == 1, "the goal was cancelled, not left to finish"
    assert not node._driving


def test_a_live_correction_lets_the_drive_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The other half of the same rule: while the pulse keeps arriving nothing interferes."""
    node = server(tmp_path)
    standing_at(node, at(0.0, 0.0, 0.0, age_s=0.1))
    correction_landed(node)
    handle = nav2_answers(node, ticks=2)
    wire = Wire()
    node._go({"cmd": "go", "x": 1.0, "y": 0.0}, wire, record=tmp_path / "run.jsonl")

    events = {event["event"]: event for event in wire.events()}
    assert "lost" not in events and handle.cancelled == 0
    assert "detail" not in events["done"]


def test_a_stack_with_no_tracker_never_waits_on_the_tracker_s_service(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Every pose read used to open with a full second of standing still on a /where_am_i that
    has no server in this mode: about four seconds per goal, two of them between arrival and the
    corrective pivot. The probe for a tracker is paid once per process and never again."""
    node = server(tmp_path)
    standing_at(node, at(1.0, 0.0, 0.0, age_s=0.1))
    correction_landed(node)
    for _ in range(4):
        node._pose_now()
    node._ready()
    assert node.service_clients["where_am_i"].waits == [], "the service is not there to ask"
    assert node.service_clients["relocalize"].waits == [TRACKER_WAIT_S], "one probe, then the graph"


def test_a_tracker_that_comes_up_late_is_still_found(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The probe's answer is not cached for ever: a known-map stack whose tracker is still
    starting must be picked up, or the goal server would read TF for the rest of the session."""
    node = server(tmp_path)
    standing_at(node, at(1.0, 0.0, 0.0, age_s=0.1))
    assert not node._tracker_here()
    node.service_clients["relocalize"].ready = True  # the tracker finished starting
    assert node._tracker_here()
    tracker_says(node, 0.8)
    assert node._pose_now()["fit"] == 0.8, "and its pose is the answer again"


def tracker_sigma(node: Any, xy_m: float, yaw_deg: float = 1.2) -> None:
    """The tracker's fused uncertainty on the wire, as pepin_bringup.relocalizer publishes it:
    one JSON string, the shape pepin.watch.Sigma defines."""
    said = Sigma(xy_m, yaw_deg, 0.0).to_json(stamp=node.clock.seconds, word_age_s=0.1)
    node.subs[SIGMA_TOPIC][1](String(data=said))


def test_a_camera_only_drive_is_gated_on_the_sigma_and_not_on_the_lidar_s_fit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """2026-09-15: with the camera holding the pose /localization_fit is 0.00 — no lidar scan
    scores it — and this gate refused every goal. The sigma out of the fusion says the same pose
    is known to 6 cm, and it outranks the fit; the flag off puts the fit rules back (rule 19)."""
    node = server(tmp_path)
    tracker_says(node, 0.0)
    assert not node._ready().ready, "the fit alone: the failure of the day"
    tracker_sigma(node, 0.06)
    ready = node._ready()
    assert ready.ready and ready.rule == BY_SIGMA and ready.tracker
    node._switches.set("sigma_gate", False)
    assert not node._ready().ready and node._ready().rule == BY_FIT
    assert "sigma_gate=off" in node._switches.state()
    node._switches.set("sigma_gate", True)
    tracker_sigma(node, 0.44)
    wide = node._ready()
    assert not wide.ready and wide.search and "0.44 m" in wide.reason


def test_a_sigma_that_stops_arriving_stops_the_drive_under_way(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The blind-drive watch on the new reading: a healthy camera drive runs, and a tracker that
    goes quiet — its last sigma small for ever — is what cuts it, because the number's AGE is
    part of the judgment."""
    node = server(tmp_path)
    tracker_says(node, 0.0)
    tracker_sigma(node, 0.06)
    assert node._sigma() is not None and node._sigma().xy_m == 0.06, "the JSON was read"
    blind = BlindDriveWatch()
    assert not blind.observe(node.fit, 0.0, sigma=node._sigma()), "a camera drive is not blind"
    assert blind.rule == BY_SIGMA
    node.clock.seconds += SOURCE_PATIENCE_S + 1.0
    cut = False
    for tick in range(1, 8):
        cut = blind.observe(node.fit, float(tick), sigma=node._sigma())
    assert cut and "stopped" in blind.phrase(), blind.phrase()
