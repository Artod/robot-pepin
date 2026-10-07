"""The gaze node under the ROS stubs: the doors, the stall look end to end, drive starts, path
gaze, drive 306 replayed (the baseline preset against the node before it, following, glances,
the slow way home, the looks' frames), and the topic and service names (literals: they are the
contract with the behaviour tree, the frame consumers and the tools)."""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import ros_stubs
import yaml

ros_stubs.install()

from gaze_replay import T0, Tape, Write, replay, replay_tape  # noqa: E402
from pepin_bringup import gaze as gaze_node  # noqa: E402
from pepin_bringup.gaze import Gaze  # noqa: E402
from pepin_bringup.msgs import cloud_from_fields  # noqa: E402
from ros_stubs import (  # noqa: E402
    Future,
    GetPointMapROI,
    GoalStatus,
    GoalStatusArray,
    Header,
    OccupancyGrid,
    Path_,
    PoseStamped,
    Time,
)

from pepin.gaze import PERSON, Aim, HeldLook, Look, Outcome  # noqa: E402
from pepin.neck import NeckConfig  # noqa: E402
from pepin.tsdf import RigidPose  # noqa: E402

RES, SIZE, ORIGIN = 0.05, 60, -1.5


class FakeLink:
    """The base server's socket: lines sent, and a callback the test feeds lines through."""

    def __init__(self, host: str, port: int, on_message: Any, *, name: str) -> None:
        self.on_message = on_message
        self.sent: list[dict[str, Any]] = []
        self.connected = True

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def send(self, line: bytes) -> bool:
        self.sent.append(json.loads(line))
        return True


class FakeTf:
    """Every frame on top of every other: the cart at the origin of odom and map, facing +x."""

    def pose(self, target: str, source: str, stamp: Any = None, timeout_s: float = 0.0) -> Any:
        return RigidPose(np.eye(3), np.zeros(3))

    def close(self) -> None:
        pass


class FaceSink:
    """The head server's door as the stall look's face speaks to it."""

    def __init__(self) -> None:
        self.said: list[tuple[str, ...]] = []

    def event(self, name: str, *, end: bool = False) -> None:
        self.said.append(("event", name))

    def clear(self) -> None:
        self.said.append(("clear",))

    def lease(self, seconds: float) -> None:
        self.said.append(("lease",))

    def close(self) -> None:
        self.said.append(("close",))


def build(monkeypatch: pytest.MonkeyPatch) -> Gaze:
    """A gaze node on a fake link, a fake face and a TF with every frame on top of the other."""
    monkeypatch.setattr(gaze_node, "JsonLineLink", FakeLink)
    face = FaceSink()
    monkeypatch.setattr(gaze_node, "face_client", lambda host: face)  # never the board's head
    with ros_stubs.parameters(http_port=0):
        built = Gaze()
    built._tf = FakeTf()  # type: ignore[assignment]
    return built


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> Iterator[Gaze]:
    built = build(monkeypatch)
    yield built
    built.close()


def link(node: Gaze) -> FakeLink:
    return node._link  # type: ignore[return-value]


def state_line(node: Gaze, **extra: Any) -> None:
    link(node).on_message({"type": "state", "moving": False, "v": 0.0, "w": 0.0, **extra})


def costmap(*cells: tuple[float, float]) -> Any:
    msg = OccupancyGrid()
    msg.header.frame_id = "odom"
    msg.info.resolution = RES
    msg.info.width = msg.info.height = SIZE
    msg.info.origin.position.x = msg.info.origin.position.y = ORIGIN
    data = np.zeros((SIZE, SIZE), dtype=int)
    for x, y in cells:
        data[int((y - ORIGIN) / RES), int((x - ORIGIN) / RES)] = 100
    msg.data = list(data.reshape(-1))
    return msg


def plan_ahead() -> Any:
    plan = Path_()
    plan.header.frame_id = "map"
    for x in np.arange(0.0, 2.0, 0.05):
        pose = PoseStamped()
        pose.pose.position.x = float(x)
        plan.poses.append(pose)
    return plan


def column_answer(points: list[tuple[float, float, float, float]]) -> Any:
    xyz = np.array([p[:3] for p in points], dtype=float).reshape(-1, 3)
    weights = np.array([p[3] for p in points], dtype=float)
    cloud = cloud_from_fields(
        {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "weight": weights,
            "lidar": np.zeros(len(weights)),
        },
        Time(),
        "odom",
    )
    return GetPointMapROI.Response(sub_map=cloud)


def columns(node: Gaze, *answers: Any) -> list[Any]:
    """The column service is up and answers these, one per call; the requests are kept."""
    client = node.service_clients["/fusion/column"]
    client.ready = True
    queue = list(answers)
    asked: list[Any] = []

    def call_async(request: Any) -> Future:
        asked.append(request)
        return Future(queue.pop(0) if len(queue) > 1 else queue[0])

    client.call_async = call_async  # type: ignore[method-assign]
    return asked


def stall(node: Gaze) -> Any:
    srv_type, callback = node.services["/gaze/stall_look"]
    return callback(srv_type.Request(), srv_type.Response())


def blocked_ahead(node: Gaze, x: float = 0.6) -> None:
    node._on_costmap(costmap((x, 0.0)))
    node._on_plan(plan_ahead())


def drive_robot(node: Gaze, until: Callable[[], bool], timeout_s: float = 5.0) -> None:
    """Play the board and the fusion node: answer every neck move as arrived, fuse frames,
    step the arbiter, until ``until``."""
    cfg = NeckConfig.from_json(gaze_node.config_file("neck.json"))
    answered = 0
    deadline = time.monotonic() + timeout_s
    while not until() and time.monotonic() < deadline:
        state_line(node)
        sent = link(node).sent
        for message in sent[answered:]:
            if message["cmd"] == "neck_goto":
                pan, tilt = message["pan_ticks"], message["tilt_ticks"]
            elif message["cmd"] == "neck_home":
                pan, tilt = cfg.reference.pan_ticks, cfg.reference.tilt_ticks
            else:
                continue
            link(node).on_message(
                {"type": "neck_goto", "reached": True, "pan_ticks": pan, "tilt_ticks": tilt}
            )
        answered = len(sent)
        node._on_frame(Header(stamp=_stamp(time.time())))
        node._step()
        time.sleep(0.01)


def _stamp(t: float) -> Any:
    whole = math.floor(t)
    return Time(sec=whole, nanosec=round((t - whole) * 1e9))


def in_thread(work: Callable[[], Any]) -> tuple[threading.Thread, list[Any]]:
    out: list[Any] = []
    thread = threading.Thread(target=lambda: out.append(work()), daemon=True)
    thread.start()
    return thread, out


# ---- the contract --------------------------------------------------------------------------------
def test_the_topics_the_service_and_the_flags(node: Gaze) -> None:
    assert {"/gaze/state", "/gaze/stall"} <= set(node.pubs)
    assert "/gaze/stall_look" in node.services and "/gaze/stall_look" in node.service_groups
    assert "/fusion/column" in node.service_clients
    assert {
        "/neck/state",
        "/fusion/frame",
        "/plan",
        "/local_costmap/costmap",
        "/scan",
        "/depth_marks",
        "/navigate_to_pose/_action/status",
        "/navigate_through_poses/_action/status",
        "/follow_path/_action/status",
        "/backup/_action/status",
        "/drive_on_heading/_action/status",
        "/controller_selector",
    } <= set(node.subs)
    # on by default since 2026-10-06 (24 drives with them live)
    assert node._switches.on("stall_look") and node._switches.on("path_gaze")
    assert node._switches.on("reverse_gaze")
    assert int(node._switches["frames"]) == 3
    line = node.logger.texts("info")[-1]
    assert line.startswith("gaze up:") and "stall_look=on" in line


def test_the_recoveries_it_watches_are_the_behaviour_servers_and_the_trees() -> None:
    """The reverse look's recovery trigger reads these actions' status lists by name."""
    repo = Path(__file__).resolve().parents[2]
    params = yaml.safe_load((repo / "ros/params/nav2_params.yaml").read_text())
    plugins = params["behavior_server"]["ros__parameters"]["behavior_plugins"]
    assert set(gaze_node.BEHAVIOUR_ACTIONS) <= set(plugins)
    tree = (repo / "ros/params/pepin_nav_to_pose.xml").read_text()
    assert "<BackUp " in tree and "<DriveOnHeading " in tree and "server_name" not in tree
    assert gaze_node.CONTROLLER_ACTION == "follow_path"  # Nav2's controller server, no rename
    # ...and the tape keeps its status, which the replays' follow rows are made from
    assert "/follow_path/_action/status" in (repo / "src/pepin/tape_rows.py").read_text()


def test_the_main_spins_on_several_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    rclpy = ros_stubs.install()
    monkeypatch.setattr(gaze_node, "JsonLineLink", FakeLink)
    monkeypatch.setattr(gaze_node, "face_client", lambda host: FaceSink())
    with ros_stubs.parameters(http_port=0):
        gaze_node.main()
    assert "spin on MultiThreadedExecutor" in rclpy.log


def test_the_state_goes_out_on_a_change(node: Gaze) -> None:
    node._step()
    first = json.loads(node.pubs["/gaze/state"].sent[-1].data)
    assert first["phase"] == "home" and first["blind"] is False
    sent = len(node.pubs["/gaze/state"].sent)
    node._step()
    assert len(node.pubs["/gaze/state"].sent) == sent  # nothing changed
    node._publish_state()
    assert len(node.pubs["/gaze/state"].sent) == sent + 1  # ...but the 10 Hz line goes out


# ---- the stall look ------------------------------------------------------------------------------
def test_stall_look_off_answers_at_once(node: Gaze) -> None:
    node._switches.set("stall_look", False)  # on by default since 2026-10-06
    answer = stall(node)
    assert answer.success and answer.message == "stall_look off: nothing looked at"
    assert link(node).sent == []


@pytest.mark.slow
def test_a_stall_look_saccades_counts_frames_reads_the_verdict_and_comes_home(
    node: Gaze,
) -> None:
    node._switches.set("stall_look", True)
    blocked_ahead(node)
    phantom = column_answer([(0.6, 0.0, 0.3, 12.0), (0.62, 0.01, 0.4, 8.0)])
    asked = columns(node, phantom, column_answer([]))
    thread, out = in_thread(lambda: stall(node))
    drive_robot(node, lambda: not thread.is_alive())
    answer = out[0]
    assert answer.success, answer.message
    assert "carved" in answer.message and "home done" in answer.message
    sent = [m["cmd"] for m in link(node).sent]
    assert sent[0] == "neck_goto" and link(node).sent[0]["hold"] is True
    assert sent[-1] == "neck_home"
    assert len(asked) == 2  # the columns before and after the look
    # ...in the marks' own band: the floor's surface stands under every cell and blocks nothing
    box = asked[0]
    assert box.z - box.l_z / 2 == pytest.approx(0.15) and box.z + box.l_z / 2 == pytest.approx(1.3)
    record = json.loads(node.pubs["/gaze/stall"].sent[-1].data)
    assert record["verdict"] == "carved" and record["look"]["frames_seen"] >= 3
    assert record["blockers"]["unexplained"] == 1 and record["before"]["occupied"] == 1
    assert record["after"]["occupied"] == 0 and record["lidar_before"]["cells"] == 0
    assert node._stalls == {"carved": 1}
    face = node._face._sink  # type: ignore[union-attr]
    assert face.said == [("event", "stall_look"), ("event", "phantom_carved")]  # the face
    node._switches.set("face_events", False)
    assert node._stall_face() is None


def test_nothing_under_the_hull_is_nothing_to_look_at(node: Gaze) -> None:
    node._switches.set("stall_look", True)
    node._on_costmap(costmap((0.6, 1.0)))
    node._on_plan(plan_ahead())
    answer = stall(node)
    assert answer.success and "no lethal cell under the hull" in answer.message
    assert link(node).sent == []


def test_an_empty_column_is_a_stale_mark_and_no_look(node: Gaze) -> None:
    node._switches.set("stall_look", True)
    blocked_ahead(node)
    columns(node, column_answer([]))
    answer = stall(node)
    assert answer.success and "a stale mark" in answer.message and link(node).sent == []


def test_without_the_fusion_column_there_is_no_look(node: Gaze) -> None:
    node._switches.set("stall_look", True)
    blocked_ahead(node)
    answer = stall(node)
    assert answer.success and "no /fusion/column" in answer.message


def test_a_blocker_under_the_bumper_asks_the_tree_to_back_off(node: Gaze) -> None:
    node._switches.set("stall_look", True)
    # 7 cm ahead of the lens and a metre below it: 86 deg down, past stall_max_depression_deg
    blocked_ahead(node, x=0.1)
    columns(node, column_answer([(0.1, 0.0, 0.05, 10.0)]))
    answer = stall(node)
    assert not answer.success and "back off first" in answer.message, answer.message
    assert link(node).sent == []
    assert json.loads(node.pubs["/gaze/stall"].sent[-1].data)["verdict"] == "back off"


# ---- drives and doors ----------------------------------------------------------------------------
def goals(*uuids: int) -> Any:
    msg = GoalStatusArray()
    for uuid in uuids:
        status = GoalStatus(status=2)
        status.goal_info.goal_id.uuid = [uuid] * 16
        msg.status_list.append(status)
    return msg


def test_a_drive_start_lets_every_look_go_and_its_end_the_navigation_ones(node: Gaze) -> None:
    out: list[Any] = []
    node._arbiter.submit(
        Look("llm.look", (node._arbiter.home,), PERSON, 0, 0.0, 10.0), 0.0, out.append
    )
    node._on_nav_status("navigate_to_pose", goals(1))
    assert node._driving and out[0].status == "preempted"
    assert out[0].reason == "by the drive's start"
    node._arbiter.submit(Look("nav.stall", (node._arbiter.home,), 1, 0, 0.0, 3.0), 0.0, out.append)
    node._on_nav_status("navigate_to_pose", goals(1))  # the same goal: nothing new
    assert len(out) == 1
    node._on_nav_status("navigate_to_pose", goals())
    assert not node._driving and out[1].status == "preempted"


def test_a_head_left_turned_by_hand_goes_home_when_a_drive_starts(node: Gaze) -> None:
    """What the tools' face_forward did for their own drives, the arbiter does for every one."""
    state_line(node)
    joints = gaze_node.JointState()
    joints.header.stamp = _stamp(time.time())
    joints.name = ["neck_pan", "head_tilt"]
    joints.position = [math.radians(40.0), math.radians(23.8)]
    node._on_neck(joints)
    node._on_nav_status("navigate_to_pose", goals(3))
    node._step()
    assert link(node).sent == [{"cmd": "neck_home", "hold": False}]
    assert "the head found off home is sent there" in node.logger.texts("info")[-1]


def test_a_persons_look_is_refused_during_a_drive_by_todays_base_server(node: Gaze) -> None:
    node._on_nav_status("navigate_to_pose", goals(1))
    answer = node._door_look(
        {"source": "llm.look", "kind": "angles", "target": {"pan_rad": 0.5, "tilt_rad": 0.4}}
    )
    assert answer["status"] == "denied" and "during a drive" in answer["reason"]
    bad = node._door_look({"source": "llm.look", "kind": "stare"})
    assert bad["status"] == "denied" and "stare" in bad["reason"]


def test_the_door_state_names_the_driver_and_the_reach(node: Gaze) -> None:
    state = node._door_state({})
    assert state["driver"] == "neck_goto" and state["phase"] == "home"
    assert state["driving"] is False and state["pending"] == []
    assert state["reach"]["home"]["tilt_deg"] == pytest.approx(23.8)
    state_line(node, pan_ticks=2029, tilt_ticks=2311)
    assert node._door_state({})["driver"] == "neck_target"
    assert node._door_renew({"source": "nobody"}) == {"renewed": 0}


def test_path_gaze_looks_along_the_plan_with_neck_target(node: Gaze) -> None:
    node._switches.set("path_gaze", True)
    state_line(node, pan_ticks=2029, tilt_ticks=2311, v=0.3)
    turn = Path_()
    turn.header.frame_id = "map"
    for x, y in [(0.0, 0.0), (0.3, 0.0), (0.3, 0.3), (0.3, 1.5)]:
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y = x, y
        turn.poses.append(pose)
    node._on_plan(turn)
    node._on_nav_status("navigate_to_pose", goals(7))
    node._step()
    sent = link(node).sent[-1]
    assert sent["cmd"] == "neck_target" and sent["pan_rad"] > math.radians(20)
    assert [r.source for r in node._arbiter.pending()] == ["nav.path"]


def test_path_gaze_waits_for_a_base_server_that_moves_the_neck_while_driving(node: Gaze) -> None:
    node._switches.set("path_gaze", True)
    state_line(node, moving=True, v=0.3)
    node._on_plan(plan_ahead())
    node._on_nav_status("navigate_to_pose", goals(7))
    node._step()
    assert link(node).sent == [] and node._arbiter.pending() == []


def test_the_report_line(node: Gaze) -> None:
    node._report()
    line = node.logger.texts("info")[-1]
    assert line.startswith("gaze: driver neck_goto (base server connected), phase home")
    assert "stall looks: none" in line and "flags: stall_look=on" in line


# ---- following, glances, the slow way home, the looks' frames (gaze_replay: drive 306) ----------
PRESETS = json.loads((Path(__file__).resolve().parents[2] / "config/gaze_presets.json").read_text())
# The writes the node made before the follow-in-zone knobs (64bc0cd), on the same replay
# (scratch/gaze_follow/baseline_capture.py on that tree): the reverse look, the path look, then
# home and back at each short reverse, home at the drive's end.
BEFORE_FOLLOW = (
    Write(1.0, -150.0, 23.8, None),
    Write(2.4, 45.0, 48.5, None),
    Write(3.5, 0.0, 23.8, None),
    Write(3.8, 45.0, 48.5, None),
    Write(4.5, 0.0, 23.8, None),
    Write(5.2, 45.0, 48.5, None),
    Write(6.7, 0.0, 23.8, None),
)


# 2 s forward, 1.5 s back, 2 s forward, and the same node's writes (scratch/gaze_follow/leg_old.py):
# the path look lapsed home 0.5 s into the reverse, the reverse look was cut mid-swing by the path
# look when the cart went forward again.
LEG_V = (0.1,) * 20 + (-0.1,) * 15 + (0.1,) * 20
BEFORE_FOLLOW_LEG = (
    Write(0.0, 45.0, 48.5, None),
    Write(2.3, 0.0, 23.8, None),
    Write(3.0, -150.0, 23.8, None),
    Write(3.6, 45.0, 48.5, None),
    Write(5.5, 0.0, 23.8, None),
)


def test_the_baseline_preset_makes_exactly_the_writes_of_the_node_before_it(node: Gaze) -> None:
    assert tuple(replay(node, knobs=PRESETS["baseline"])) == BEFORE_FOLLOW


def test_the_baseline_preset_on_a_reverse_leg_is_the_node_before_it_too(node: Gaze) -> None:
    writes = replay(node, LEG_V, (0.5,) * len(LEG_V), knobs=PRESETS["baseline"])
    assert tuple(writes) == BEFORE_FOLLOW_LEG
    drive = node.logger.texts("info")[-1]
    assert "1 WROTE NO FRAMES: nav.reverse -150/24 deg 0.60 s preempted" in drive, drive


# The reverse look timed as before 2026-10-06: only after reverse_min_s, let go when the leg ends.
LATE_REVERSE = {"reverse_min_m": 0.0, "reverse_recovery": 0, "reverse_hold_s": 0.0}


def test_the_path_look_is_held_until_the_reverse_look_replaces_it(node: Gaze) -> None:
    writes = replay(node, LEG_V, (0.5,) * len(LEG_V), knobs=LATE_REVERSE)
    assert [(x.t, x.pan_deg, x.speed_deg_s) for x in writes] == [
        (0.0, 45.0, None),
        (3.0, -150.0, None),  # no lapse home at 2.5: the reverse look takes the held head
        (4.1, 45.0, None),  # the glance's swing and 0.6 s dwell first, then the path look
        (5.5, 0.0, 45.0),
    ]
    assert node.logger.texts("info")[-1].endswith("every one wrote frames")


def test_the_first_forward_command_lets_the_reverse_look_go_even_mid_swing(node: Gaze) -> None:
    """The same leg at the defaults: nothing announced it (the plan leads ahead, no controller
    status), so its look comes after reverse_min_s, and the cart going forward 0.5 s later takes
    it back before the head arrived; the path look turns the head in the same step."""
    writes = replay(node, LEG_V, (0.5,) * len(LEG_V))
    assert [(x.t, x.pan_deg, x.speed_deg_s) for x in writes] == [
        (0.0, 45.0, None),
        (3.0, -150.0, None),
        (3.5, 45.0, None),  # the first forward command: no dwell, no swing's end awaited
        (5.5, 0.0, 45.0),
    ]
    assert "gaze: the reverse look let go at the first forward command" in node.logger.texts("info")
    assert "nav.reverse -150/24 deg 0.50 s preempted" in node.logger.texts("info")[-1]


def test_the_follow_defaults_hold_the_path_look_through_short_reverses(node: Gaze) -> None:
    writes = replay(node, knobs=LATE_REVERSE)  # the follow preset with the late reverse look
    home_tilt = math.degrees(node._arbiter.home.tilt_rad)
    assert writes == [
        Write(1.0, -150.0, 23.8, None),  # the reverse glance
        Write(2.4, 45.0, 48.5, None),  # the path look, held through both short reverses
        Write(6.7, 0.0, 23.8, 45.0),  # home at return_deg_s once the drive ended
    ]
    assert not any(w.home(home_tilt) for w in writes[:-1])
    drive = [t for t in node.logger.texts("info") if t.startswith("gaze: the drive's head")]
    assert drive == [
        "gaze: the drive's head: 3 writes, looks nav.path 1 (35 frames), nav.reverse 1"
        " (7 frames); every one wrote frames"
    ]
    node._report()
    assert "looks nav.path 1 (35 frames), nav.reverse 1 (7 frames)" in node.logger.texts("info")[-1]


def test_a_reverse_glance_ends_whole_when_the_leg_ends_under_it(node: Gaze) -> None:
    """A 1.2 s reverse then forward, the reverse look timed as before 2026-10-06: the glance
    starts at 1.0 s, the head is still turning when the leg ends, and the path look waits until
    the glance has its frames."""
    v = (-0.1,) * 12 + (0.1,) * 20
    writes = replay(node, v, (0.5,) * len(v), knobs=LATE_REVERSE)
    assert [(x.pan_deg, x.speed_deg_s) for x in writes] == [
        (-150.0, None),
        (45.0, None),
        (0.0, 45.0),
    ]
    assert writes[1].t - writes[0].t >= 150.0 / 300.0 + 0.3  # the swing, then its dwell
    drive = node.logger.texts("info")[-1]
    assert drive.startswith("gaze: the drive's head: 3 writes") and "nav.reverse 1 (" in drive
    assert "nav.reverse 1 (0 frames)" not in drive and "every one wrote frames" in drive


# Hybrid-A*'s departure from a dock (the cart at the origin facing +x): 0.20 m backwards to a cusp,
# then forward round to the left. 0.2 s standing, 1.5 s back, 2 s forward.
BACK_THEN_LEFT = ((0.0, 0.0), (-0.1, 0.0), (-0.2, 0.0), (-0.1, 0.05), (0.2, 0.4), (0.3, 1.5))
DEPART_V = (0.0,) * 2 + (-0.1,) * 15 + (0.1,) * 20


def rear_times(writes: list[Write]) -> list[float]:
    return [w.t for w in writes if abs(w.pan_deg) == 150.0]


def test_a_reverse_leg_the_plan_announces_turns_the_head_at_its_first_command(
    node: Gaze, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = replay(node, DEPART_V, (0.5,) * len(DEPART_V), route=BACK_THEN_LEFT)
    assert rear_times(writes) == [0.2]  # the step of the leg's first reversing command
    let_go = writes[[w.t for w in writes].index(0.2) + 1]
    assert let_go.t == 1.7 and abs(let_go.pan_deg) < 90.0  # the first forward command: ahead
    info = node.logger.texts("info")
    assert "gaze: a reverse look (the plan's 0.20 m reverse leg), pan -150 deg" in info
    late = build(monkeypatch)
    before = replay(
        late, DEPART_V, (0.5,) * len(DEPART_V), route=BACK_THEN_LEFT, knobs=LATE_REVERSE
    )
    late.close()
    assert rear_times(before) == [1.2]  # reverse_min_s into the leg, as before 2026-10-06
    short_leg = build(monkeypatch)  # a 0.20 m reverse leg is no announcement at 0.25
    after = replay(
        short_leg,
        DEPART_V,
        (0.5,) * len(DEPART_V),
        route=BACK_THEN_LEFT,
        knobs={"reverse_min_m": 0.25},
    )
    short_leg.close()
    assert rear_times(after) == [1.2]


def test_a_reverse_a_recovery_drives_turns_the_head_at_its_first_command(
    node: Gaze, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan leads ahead (TURN_LEFT); the controller's status says it runs no goal: the
    tree's BackUp drives. A BackUp goal running says so whatever the controller's status."""
    node._on_driver_status("follow_path", goals())
    writes = replay(node, DEPART_V, (0.5,) * len(DEPART_V))
    assert rear_times(writes) == [0.2]
    assert "gaze: a reverse look (a recovery drives), pan -150 deg" in node.logger.texts("info")
    backup = build(monkeypatch)
    backup._on_driver_status("follow_path", goals(1))
    backup._on_driver_status("backup", goals(2))
    assert rear_times(replay(backup, DEPART_V, (0.5,) * len(DEPART_V))) == [0.2]
    backup.close()
    # the controller driving, its status never heard, or the knob off: reverse_min_s
    for status, knobs in ((goals(1), {}), (None, {}), (goals(), {"reverse_recovery": 0})):
        other = build(monkeypatch)
        if status is not None:
            other._on_driver_status("follow_path", status)
        late = replay(other, DEPART_V, (0.5,) * len(DEPART_V), knobs=knobs)
        other.close()
        assert rear_times(late) == [1.2]


def test_a_reverse_look_is_held_through_a_short_stand_and_let_go_at_its_end(node: Gaze) -> None:
    """Backing 1.5 s, standing 3 s, forward: held 1.5 s of the stand (a reverse within it would
    be the same leg), then the path look."""
    v = (0.0,) * 2 + (-0.1,) * 15 + (0.0,) * 30 + (0.1,) * 10
    writes = replay(node, v, (0.5,) * len(v), route=BACK_THEN_LEFT)
    assert rear_times(writes) == [0.2]
    let_go = writes[[w.t for w in writes].index(0.2) + 1]
    assert let_go.t == 3.2 and abs(let_go.pan_deg) < 90.0
    assert "gaze: the reverse look let go at its stand's end" in node.logger.texts("info")


def test_a_look_that_wrote_no_frames_is_named_in_the_report(node: Gaze) -> None:
    node._looks.add(HeldLook("1", "nav.path", 4, Aim(0.8, 0.85), 10.0, 10.3, "preempted", None))
    node._report()
    line = node.logger.texts("info")[-1]
    assert "looks nav.path 1 (0 frames); 1 WROTE NO FRAMES: nav.path +46/49 deg 0.30 s" in line
    node._report()
    assert "looks none" in node.logger.texts("info")[-1]  # the window starts over


def test_the_dark_frames_reach_the_arbiter_and_a_dark_look_is_named_in_the_report(
    node: Gaze,
) -> None:
    """depth_stream's /depth/dark goes to the arbiter at the frame's own stamp; the patience is
    the live knob; a look that gave up on dark frames is counted and named as such."""
    stamp = ros_stubs.Time(sec=100, nanosec=500_000_000)
    node.subs["/depth/dark"][1](ros_stubs.Header(stamp=stamp))
    assert list(node._arbiter._dark) == [100.5]
    assert node._settings().dark_patience_s == 1.5
    node._switches.set("gate_dark_patience_s", 0.0)
    assert node._settings().dark_patience_s == 0.0
    node._arbiter.counts["expired"] += 1
    node._arbiter.counts["dark"] += 1
    dark = HeldLook("1", "nav.stall", 1, Aim(0.8, 0.85), 10.0, 11.6, "dark, no frames", 10.1)
    node._looks.add(dark)
    node._report()
    line = node.logger.texts("info")[-1]
    assert "expired 1 of which dark, no frames 1)" in line
    assert "1 WROTE NO FRAMES: nav.stall +46/49 deg 1.60 s dark, no frames" in line


def test_the_stall_look_is_a_glance_capped_by_its_ttl_only_with_glances_on(node: Gaze) -> None:
    node._switches.set("stall_look", True)
    asked: list[Look] = []

    def answer(look: Look) -> Outcome:
        asked.append(look)
        return Outcome(look.id, look.source, "done", frames_seen=3)

    node._ask = answer  # type: ignore[method-assign]
    for dwell in (0.3, 0.0):
        node._switches.set("glance_dwell_s", dwell)
        blocked_ahead(node)
        columns(node, column_answer([(0.6, 0.0, 0.3, 12.0)]), column_answer([]))
        assert stall(node).success
    assert [(a.frames, a.hold_s) for a in asked if a.views] == [(3, 3.0), (3, 0.0)]


def test_no_return_look_when_the_head_is_home_or_the_knob_is_off(node: Gaze) -> None:
    node._on_nav_status("navigate_to_pose", goals(5))
    node._on_nav_status("navigate_to_pose", goals())
    assert node._arbiter.pending() == []  # nothing ever moved the head
    node._switches.set("return_deg_s", 0.0)
    node._return_home(0.0)
    assert node._arbiter.pending() == []


# ---- drives 0330 and 0358 replayed (gaze_replay.replay_tape, tests/fixtures/drive_<n>_gaze.json)
# At the live drive's frame rate (3.3-6.2 fused fps while driving: a frame every 4th tick, 5 Hz) and
# head speed (-60 -> +150 deg in 0.98-1.03 s, latency included: 210 deg/s). On the tree before the
# path_tail_m knob and the preempting reverse glance (scratch/gaze_night/old) the 0330 replay makes
# the live drive's 18 writes: the path look -60/48 expired after 0.80 s (live 0.70 s) and the tail's
# saccade at +40.8 s (live +41.5 s).
TAPE_0330 = Tape.load("0330")
TAPE_0358 = Tape.load("0358")
TAPE_0365 = Tape.load("0365")
LIVE_0330: dict[str, Any] = {"frame_every": 4, "head_deg_s": 210.0}
# Path gaze without its economy (no saccade standing or just stopped, the re-centre): the follower
# of 2026-10-06, which the tapes' reverse-timing and tail tests below were written against.
# The head of 2026-10-06 night: no ahead magnet, no parking look (the knobs' baseline values).
NO_AHEAD = {"path_bend_deg": 0.0, "path_park_ahead_m": 0.0, "reverse_park_min_m": 0.0}
NO_ECONOMY = {
    "path_still_m_s": 0.0,
    "path_stall_guard_s": 0.0,
    "path_recentre_s": 0.0,
    **NO_AHEAD,
}
FOLLOW_0330 = [
    (0.0, "nav.path", -60.0),  # the tape's first plan (live: -14/66 from the one before it)
    (0.4, "nav.stall", -3.8),
    (1.5, "nav.path", -60.0),
    (3.6, "nav.stall", -4.3),
    (4.7, "nav.path", -60.0),
    (6.55, "nav.reverse", 150.0),  # the BackUp's first command (the controller runs no goal)
    (9.35, "nav.path", -60.0),  # its first forward command
    (10.35, "nav.reverse", 150.0),  # the controller's reverse, once the plan read 0.16 m of it
    (14.65, "nav.path", -31.2),  # held through the 1.4 s stand and Rock's BackUp, to forward
    (17.65, "nav.path", -53.5),
    (18.45, "nav.reverse", 150.0),  # the plan's 0.23 m (a 1.1 s leg: none before)
    (21.05, "nav.path", -53.5),  # 1.5 s into a 3.5 s spinning stand
    (25.15, "nav.reverse", 150.0),
    (27.75, "nav.path", -53.5),
    (31.15, "nav.path", -25.0),
    (35.35, "nav.path", 0.2),
    (42.0, "nav.return", 0.0),  # no saccade in the last metres: held to the end, then home
]
# The same replay with the reverse look timed as before 2026-10-06 (LATE_REVERSE): 1.0-1.2 s into
# each leg, the 1.1 s leg at +18.4 none, the path look swung in between at the 1.4 s stand.
FOLLOW_0330_LATE = [
    (0.0, "nav.path", -60.0),
    (0.4, "nav.stall", -3.8),
    (1.5, "nav.path", -60.0),
    (3.6, "nav.stall", -4.3),
    (4.7, "nav.path", -60.0),
    (6.6, "nav.reverse", 150.0),
    (9.2, "nav.path", -60.0),
    (11.4, "nav.reverse", 150.0),
    (13.1, "nav.path", -60.0),  # the cart stood 1.4 s between two backups
    (13.6, "nav.reverse", 150.0),  # ...and backed again: the glance takes the swinging head
    (14.8, "nav.path", -31.2),
    (17.6, "nav.path", -53.5),
    (26.2, "nav.reverse", 150.0),
    (27.9, "nav.path", -53.5),
    (31.2, "nav.path", -21.4),
    (37.2, "nav.path", 10.6),
    (42.0, "nav.return", 0.0),
]


def short(writes: list[Write]) -> list[tuple[float, str, float]]:
    return [(w.t, w.source, w.pan_deg) for w in writes]


def reverse_looks(node: Gaze) -> list[tuple[float, float]]:
    """The reverse looks that held the head: when they took it and let it go, from the goal."""
    return [
        (round(b.since - T0, 2), round(b.until - T0, 2))
        for b in node._looks.looks
        if b.source == "nav.reverse"
    ]


def first_forward(tape: Tape, after: float) -> float:
    """The tape's first forward command at or after ``after``."""
    return next(t for t, v, _w in tape.cmd if t >= after and v > 0.02)


@pytest.mark.slow
def test_drive_0330_has_no_path_saccade_in_the_plans_last_metres(
    node: Gaze, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tail rule against the node of 2026-10-06 01:00, the reverse look timed as then."""
    writes = replay_tape(node, TAPE_0330, knobs={**LATE_REVERSE, **NO_ECONOMY}, **LIVE_0330)
    assert short(writes) == FOLLOW_0330_LATE
    path = [b for b in node._looks.looks if b.source == "nav.path"]
    assert path[-1].until - T0 == pytest.approx(TAPE_0330.end_s)  # held to the drive's end
    drive = [t for t in node.logger.texts("info") if t.startswith("gaze: the drive's head")]
    assert drive[0].startswith("gaze: the drive's head: 17 writes,")
    before = build(monkeypatch)  # the metres off: the seconds alone, as before
    old = replay_tape(
        before, TAPE_0330, knobs={**LATE_REVERSE, **NO_ECONOMY, "path_tail_m": 0.0}, **LIVE_0330
    )
    before.close()
    assert [w for w in old if w.t < 40.0] == [w for w in writes if w.t < 40.0]  # nothing earlier
    assert [(w.t, w.source, w.pan_deg, w.tilt_deg) for w in old if 40.0 <= w.t < 42.0] == [
        (40.8, "nav.path", -23.1, 65.6)  # 0.10 m of plan left at 0.15 m/s and slowing
    ]


@pytest.mark.slow
def test_drive_0330_a_swinging_path_look_is_replaced_by_the_reverse_look_not_expired(
    node: Gaze,
) -> None:
    replay_tape(node, TAPE_0330, knobs={**LATE_REVERSE, **NO_ECONOMY}, **LIVE_0330)
    looks = node._looks.looks
    assert [b.text() for b in looks if b.ended == "expired"] == []
    swung = next(b for b in looks if b.source == "nav.path" and abs(b.since - T0 - 13.1) < 1e-6)
    assert (round(swung.until - T0, 2), swung.ended) == (13.6, "preempted")
    taker = next(b for b in looks if abs(b.since - swung.until) < 1e-6)
    assert taker.source == "nav.reverse" and taker.frames >= 2
    # Every path look held the head until another look took it or the drive ended.
    assert {b.ended for b in looks if b.source == "nav.path"} == {"preempted"}


@pytest.mark.slow
def test_drive_0330_looks_back_at_each_legs_start_and_ahead_at_the_first_forward_command(
    node: Gaze,
) -> None:
    writes = replay_tape(node, TAPE_0330, knobs=NO_ECONOMY, **LIVE_0330)
    assert short(writes) == FOLLOW_0330
    drive = [t for t in node.logger.texts("info") if t.startswith("gaze: the drive's head")]
    assert drive[0].startswith("gaze: the drive's head: 17 writes,")  # as many as before
    path = [b for b in node._looks.looks if b.source == "nav.path"]
    assert path[-1].until - T0 == pytest.approx(TAPE_0330.end_s)  # held to the drive's end
    looks = reverse_looks(node)
    starts = (6.537, 10.237, 18.437, 25.137)  # the legs' first reversing commands (and 13.137,
    # Rock's BackUp after the 1.4 s stand, inside the second look)
    requested = [round(took - start, 2) for (took, _), start in zip(looks, starts, strict=True)]
    assert requested == [0.01, 0.11, 0.01, 0.01]  # the plan read 0.08 m at the second leg's
    # start and 0.16 m 0.1 s later; the 50 ms step is the replay's
    forward = [first_forward(TAPE_0330, start) for start in (6.6, 13.2, 25.2)]
    assert forward == [9.337, 14.637, 27.737]
    let_go = [round(looks[i][1] - f, 2) for i, f in zip((0, 1, 3), forward, strict=True)]
    assert let_go == [0.01, 0.01, 0.01]  # the step of the first forward command
    assert looks[2][1] == 21.05  # 1.5 s into a 3.5 s stand (spinning) after its 1.1 s leg
    assert {b.ended for b in node._looks.looks if b.source == "nav.reverse"} == {"preempted"}
    assert [b.text() for b in node._looks.looks if b.ended == "expired"] == []


# Drive 0358 (home from the printer): the departure's 0.23 m reverse, a 0.8 s reverse after a
# 1.3 s stand, a stall look in the next stand, a controller reverse of 1.1 s that no plan announced
# (0.08 m), forward 0.3 s, the BackUp of 2.0 s at +8.5. Live (reverse_min_s alone): the BackUp's
# look requested +1.16 s into it, settled +2.18, forward at +2.30, let go +2.88.
FOLLOW_0358 = [
    (0.0, "nav.path", -60.0),
    (0.35, "nav.reverse", 150.0),  # the plan's 0.23 m: the leg's first command (+0.32)
    (3.9, "nav.stall", 76.1),  # held through the stand and the 0.8 s leg until the stall look
    (5.1, "nav.path", -60.0),  # ...after which the head does not turn back
    (8.15, "nav.reverse", 150.0),  # the unannounced 1.1 s leg: reverse_min_s, 1.0 s into it
    (8.25, "nav.path", -60.0),  # ...and let go 0.1 s later, at its first forward command
    (8.55, "nav.reverse", 150.0),  # the BackUp's first command (+8.52): the controller runs none
    (10.85, "nav.path", -32.2),  # its first forward command (+10.82)
    (15.05, "nav.path", -6.0),
    (23.7, "nav.return", 0.0),
]


@pytest.mark.slow
def test_drive_0358_looks_back_at_each_legs_start_and_ahead_at_the_first_forward_command(
    node: Gaze, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = replay_tape(node, TAPE_0358, knobs=NO_ECONOMY, **LIVE_0330)
    assert short(writes) == FOLLOW_0358
    looks = reverse_looks(node)
    assert [took for took, _ in looks] == [0.35, 8.15, 8.55]
    assert first_forward(TAPE_0358, 8.6) == 10.821
    assert looks[-1] == (8.55, 10.85)  # the BackUp: requested +0.03, let go +0.03 after forward
    info = node.logger.texts("info")
    assert "gaze: a reverse look (the plan's 0.23 m reverse leg), pan +150 deg" in info
    assert "gaze: a reverse look (a recovery drives), pan +150 deg" in info
    drive = [t for t in info if t.startswith("gaze: the drive's head")]
    assert drive[0].startswith("gaze: the drive's head: 10 writes,")
    late = build(monkeypatch)
    before = replay_tape(late, TAPE_0358, knobs={**LATE_REVERSE, **NO_ECONOMY}, **LIVE_0330)
    late.close()
    assert [(w.t, w.source) for w in before if w.source == "nav.reverse"] == [
        (1.4, "nav.reverse"),  # +1.08 into the departure, after it had ended (+1.20)
        (8.2, "nav.reverse"),  # +1.08 into the 1.1 s leg, held through the BackUp to +10.6
    ]
    assert len(before) == 9


# ---- the head's economy on the tapes (path_still_m_s/path_still_deg, path_stall_guard_s) ---------
# Drive 0365 without the economy (the reverse look timed as on 2026-10-06 evening): the departure
# look cut 0.3 s later by the reverse look, the path look between the reverse and the stall look,
# the aim -29 set in the pivot held until the zone let it go at +10.6.
FOLLOW_0365 = [
    (0.0, "nav.path", -60.0),
    (0.3, "nav.reverse", 150.0),
    (2.4, "nav.path", -60.0),
    (3.15, "nav.stall", 44.8),
    (4.1, "nav.path", -60.0),
    (8.0, "nav.path", -29.2),
    (10.6, "nav.path", -3.4),
    (18.6, "nav.path", -60.0),
    (19.2, "nav.reverse", -150.0),
    (20.3, "nav.return", 0.0),
]
# At the defaults: no path look while the cart stands at the start (the head stays home), the
# stall looks' aims held through the stands between them, the first path look at the first
# forward command.
ECONOMY_0330 = [
    (0.4, "nav.stall", -3.8),
    (3.6, "nav.stall", -4.3),  # no -60 in the stand between the stall looks, nor home
    (6.55, "nav.reverse", 150.0),  # ...nor after it: the BackUp's first command
    (9.35, "nav.path", -60.0),  # the first forward command
    (10.35, "nav.reverse", 150.0),
    (14.65, "nav.path", -31.2),
    (18.45, "nav.reverse", 150.0),  # no -53.5 in the stand before it (+17.65)
    (21.05, "nav.path", -56.9),
    (25.15, "nav.reverse", 150.0),
    (27.75, "nav.path", -56.9),
    (30.95, "nav.path", -25.0),
    (35.35, "nav.path", 0.2),
    (42.0, "nav.return", 0.0),
]


def stall_frames(node: Gaze) -> list[tuple[float, int, str]]:
    return [
        (round(b.since - T0, 2), b.frames, b.ended)
        for b in node._looks.looks
        if b.source == "nav.stall"
    ]


@pytest.mark.slow
def test_the_economy_off_is_the_follower_before_it_on_drive_0365(node: Gaze) -> None:
    assert short(replay_tape(node, TAPE_0365, knobs=NO_ECONOMY, **LIVE_0330)) == FOLLOW_0365


@pytest.mark.slow
def test_drive_0330_no_path_saccade_while_the_cart_stands(
    node: Gaze, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = replay_tape(node, TAPE_0330, knobs=NO_AHEAD, **LIVE_0330)
    assert short(writes) == ECONOMY_0330  # 13 writes, 17 without the economy
    looks = node._looks.looks
    assert stall_frames(node) == [(0.4, 3, "done"), (3.6, 3, "done")]
    kept = [b for b in looks if b.source == "nav.path" and b.since - T0 < 6.0]
    assert [b.aim for b in kept] == [looks[0].aim, looks[2].aim]  # the stall looks' aims
    assert sum(b.frames == 0 for b in looks) == 1  # 2 without the economy
    before = build(monkeypatch)
    replay_tape(before, TAPE_0330, knobs=NO_ECONOMY, **LIVE_0330)
    assert stall_frames(before) == stall_frames(node)  # no stall look lost a frame
    before.close()


@pytest.mark.slow
def test_drives_0358_and_0365_keep_the_head_home_until_the_first_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both start with a reverse leg: no path look at the plan's first aim while the cart
    stands, the reverse look takes the head from home at the leg's first command; the rest is
    the drive without the economy, the stall look's frames too."""
    for tape, before in ((TAPE_0358, FOLLOW_0358), (TAPE_0365, FOLLOW_0365)):
        node = build(monkeypatch)
        writes = short(replay_tape(node, tape, knobs=NO_AHEAD, **LIVE_0330))
        assert writes == before[1:], tape.end_s
        assert writes[0][1] == "nav.reverse"
        frames = stall_frames(node)
        node.close()
        off = build(monkeypatch)
        replay_tape(off, tape, knobs=NO_ECONOMY, **LIVE_0330)
        assert frames == stall_frames(off) and frames
        off.close()


# ---- straight ahead by default, the parking spot (path_bend_deg, path_park_ahead_m) -------------
TAPE_0376 = Tape.load("0376")
TAPE_0379 = Tape.load("0379")
# Drive 0376 (the printer) without them: the -12.9 look in the approach, the -60/58 look after
# the parker's plan grew back past the tail's 0.35 m, held to the end as the cart pivoted 56 deg
# under it to face the printer (live: -22.0 at +19.02, -52.6/57.7 at +21.37).
FOLLOW_0376 = [
    (0.35, "nav.reverse", -150.0),
    (3.15, "nav.path", 60.0),
    (4.35, "nav.reverse", -150.0),
    (5.45, "nav.path", 60.0),
    (6.45, "nav.reverse", -150.0),
    (7.45, "nav.path", 60.0),
    (11.45, "nav.path", 24.2),
    (18.85, "nav.path", -12.9),
    (21.25, "nav.path", -60.0),
    (25.6, "nav.return", 0.0),
]
AHEAD_0376 = [
    *FOLLOW_0376[:7],  # the departure's reverses and pivots: the same looks
    (13.45, "nav.path", 0.0),  # the left arc passed: straight ahead
    (18.25, "nav.path", -41.0),  # the parker from +17.27: the spot past the plan's end
    (23.85, "nav.path", 0.0),  # the cart faces it: straight ahead, the tail notwithstanding
    (25.6, "nav.return", 0.0),
]
# Drive 0379 (bookshelf -> home) without them: -10.8 held from the arc to the end, and the parker's
# 0.16 m reverse stub turned the head 150 deg back 0.6 s before the end (live 0.41 s).
FOLLOW_0379 = [
    (0.35, "nav.reverse", -150.0),
    (2.5, "nav.stall", -99.5),
    (3.5, "nav.path", 60.0),
    (10.75, "nav.path", 25.2),
    (13.15, "nav.path", -10.8),
    (24.3, "nav.reverse", -150.0),
    (25.7, "nav.return", 0.0),
]
AHEAD_0379 = [
    *FOLLOW_0379[:4],
    (12.75, "nav.path", 0.0),  # straight ahead through the right arc (its aim -14..-20 deg)
    (24.9, "nav.return", 0.0),  # no look back at the parker's stub
]


def ahead_share(writes: list[tuple[float, str, float]], end_s: float) -> float:
    """The share of the time from the first of these writes to the drive's end with the head's
    target straight ahead (|pan| <= 10 deg)."""
    held = [(t, pan) for t, _source, pan in writes if t <= end_s]
    ends = [*held[1:], (end_s, 0.0)]
    spans = [(t, nxt, pan) for (t, pan), (nxt, _) in zip(held, ends, strict=True)]
    total = sum(b - a for a, b, _ in spans)
    return sum(b - a for a, b, pan in spans if abs(pan) <= 10.0) / total


@pytest.mark.slow
def test_drive_0376_parks_looking_at_the_printer_not_where_the_last_bend_left_the_head(
    node: Gaze, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = short(replay_tape(node, TAPE_0376, **LIVE_0330))
    assert writes == AHEAD_0376
    info = node.logger.texts("info")
    assert "gaze: parking (the controller FollowPathMPPI): the head looks at the spot" in info
    before = build(monkeypatch)
    assert short(replay_tape(before, TAPE_0376, knobs=NO_AHEAD, **LIVE_0330)) == FOLLOW_0376
    assert reverse_looks(before) == reverse_looks(node)  # the departure's looks back unchanged
    before.close()
    final = [w for w in writes if w[1] == "nav.path"][-1]
    assert final == (23.85, "nav.path", 0.0) and TAPE_0376.end_s - final[0] > 1.5


@pytest.mark.slow
def test_drive_0379_comes_back_ahead_after_the_arc_and_does_not_look_back_on_arrival(
    node: Gaze, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = short(replay_tape(node, TAPE_0379, **LIVE_0330))
    assert writes == AHEAD_0379
    before = build(monkeypatch)
    assert short(replay_tape(before, TAPE_0379, knobs=NO_AHEAD, **LIVE_0330)) == FOLLOW_0379
    assert stall_frames(before) == stall_frames(node) == [(2.5, 3, "done")]
    assert reverse_looks(before)[0] == reverse_looks(node)[0] == (0.35, 2.5)  # the departure
    before.close()
    after_stall = [w for w in writes if w[0] > 3.0]
    assert ahead_share(after_stall, TAPE_0379.end_s) > 0.5  # 0 before: +60, +25, -10.8
    assert ahead_share([w for w in FOLLOW_0379 if w[0] > 3.0], TAPE_0379.end_s) == 0.0


@pytest.mark.slow
def test_the_stall_and_reverse_looks_are_untouched_on_drives_0330_0358_0365(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Straight ahead and the parking change only path looks: every stall look writes the same
    frames, every reverse look is asked at the same moment (0365's last one, from the plan's
    0.51 m reverse leg while parking, too)."""
    for tape in (TAPE_0330, TAPE_0358, TAPE_0365):
        new, old = build(monkeypatch), build(monkeypatch)
        replay_tape(new, tape, **LIVE_0330)
        replay_tape(old, tape, knobs=NO_AHEAD, **LIVE_0330)
        assert stall_frames(new) == stall_frames(old), tape.end_s
        assert [t for t, _ in reverse_looks(new)] == [t for t, _ in reverse_looks(old)]
        new.close()
        old.close()


def test_the_selector_and_the_parker_are_the_goal_servers() -> None:
    """The parking starts on the goal server's hand-over: its latched pick's topic and the
    parker's controller id, by name."""
    source = (
        Path(__file__).resolve().parents[2] / "ros/pepin_bringup/pepin_bringup/goal_server.py"
    ).read_text()
    assert 'create_publisher(String, "controller_selector", latched)' in source
    assert gaze_node.SELECTOR_TOPIC == "/controller_selector"
    assert gaze_node.PARKERS == ("FollowPathMPPI",)
    assert 'PARKERS = {"shim_mppi": ("FollowPathMPPI", "general_goal_checker")}' in source


def test_a_hand_over_mid_drive_starts_the_parking_and_a_drive_on_the_parker_does_not(
    node: Gaze,
) -> None:
    node._on_controller(gaze_node.String(data="FollowPathShim"))
    node._on_nav_status("navigate_to_pose", goals(1))
    assert not node._parking
    node._on_controller(gaze_node.String(data="FollowPathMPPI"))
    assert node._parking
    node._on_nav_status("navigate_to_pose", goals())
    node._on_nav_status("navigate_to_pose", goals(2))  # the next goal starts on the parker
    assert not node._parking
    node._on_controller(gaze_node.String(data="FollowPathMPPI"))
    assert not node._parking  # MPPI for the whole drive is no hand-over
    node._switches.set("path_park_ahead_m", 0.0)
    node._on_controller(gaze_node.String(data="FollowPathShim"))
    node._on_nav_status("navigate_to_pose", goals())
    node._on_nav_status("navigate_to_pose", goals(3))
    node._on_controller(gaze_node.String(data="FollowPathMPPI"))
    assert not node._parking  # the knob at 0: no parking phase
