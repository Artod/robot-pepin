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
    } <= set(node.subs)
    assert not node._switches.on("stall_look") and not node._switches.on("path_gaze")
    assert not node._switches.on("reverse_gaze")
    assert int(node._switches["frames"]) == 3
    line = node.logger.texts("info")[-1]
    assert line.startswith("gaze up:") and "stall_look=off" in line


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
    assert "stall looks: none" in line and "flags: stall_look=off" in line


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


def test_the_path_look_is_held_until_the_reverse_look_replaces_it(node: Gaze) -> None:
    writes = replay(node, LEG_V, (0.5,) * len(LEG_V))
    assert [(x.t, x.pan_deg, x.speed_deg_s) for x in writes] == [
        (0.0, 45.0, None),
        (3.0, -150.0, None),  # no lapse home at 2.5: the reverse look takes the held head
        (4.1, 45.0, None),  # the glance's swing and 0.6 s dwell first, then the path look
        (5.5, 0.0, 45.0),
    ]
    assert node.logger.texts("info")[-1].endswith("every one wrote frames")


def test_the_follow_defaults_hold_the_path_look_through_short_reverses(node: Gaze) -> None:
    writes = replay(node)  # the knobs' defaults: the follow preset
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
    """A 1.2 s reverse then forward: the glance starts at 1.0 s, the head is still turning when
    the leg ends, and the path look waits until the glance has its frames."""
    v = (-0.1,) * 12 + (0.1,) * 20
    writes = replay(node, v, (0.5,) * len(v))
    assert [(x.pan_deg, x.speed_deg_s) for x in writes] == [
        (-150.0, None),
        (45.0, None),
        (0.0, 45.0),
    ]
    assert writes[1].t - writes[0].t >= 150.0 / 300.0 + 0.3  # the swing, then its dwell
    drive = node.logger.texts("info")[-1]
    assert drive.startswith("gaze: the drive's head: 3 writes") and "nav.reverse 1 (" in drive
    assert "nav.reverse 1 (0 frames)" not in drive and "every one wrote frames" in drive


def test_a_look_that_wrote_no_frames_is_named_in_the_report(node: Gaze) -> None:
    node._looks.add(HeldLook("1", "nav.path", 4, Aim(0.8, 0.85), 10.0, 10.3, "preempted", None))
    node._report()
    line = node.logger.texts("info")[-1]
    assert "looks nav.path 1 (0 frames); 1 WROTE NO FRAMES: nav.path +46/49 deg 0.30 s" in line
    node._report()
    assert "looks none" in node.logger.texts("info")[-1]  # the window starts over


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


# ---- drive 0330 replayed (gaze_replay.replay_tape, tests/fixtures/drive_0330_gaze.json) ----------
# At the live drive's frame rate (3.3-6.2 fused fps while driving: a frame every 4th tick, 5 Hz) and
# head speed (-60 -> +150 deg in 0.98-1.03 s, latency included: 210 deg/s). On the tree before the
# path_tail_m knob and the preempting reverse glance (scratch/gaze_night/old) this replay makes the
# live drive's 18 writes: the path look -60/48 expired after 0.80 s (live 0.70 s) and the tail's
# saccade at +40.8 s (live +41.5 s).
TAPE_0330 = Tape.load("0330")
LIVE_0330: dict[str, Any] = {"frame_every": 4, "head_deg_s": 210.0}
FOLLOW_0330 = [
    (0.0, "nav.path", -60.0),  # the tape's first plan (live: -14/66 from the one before it)
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
    (42.0, "nav.return", 0.0),  # no saccade in the last metres: held to the end, then home
]


def short(writes: list[Write]) -> list[tuple[float, str, float]]:
    return [(w.t, w.source, w.pan_deg) for w in writes]


@pytest.mark.slow
def test_drive_0330_has_no_path_saccade_in_the_plans_last_metres(
    node: Gaze, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes = replay_tape(node, TAPE_0330, **LIVE_0330)
    assert short(writes) == FOLLOW_0330
    path = [b for b in node._looks.looks if b.source == "nav.path"]
    assert path[-1].until - T0 == pytest.approx(TAPE_0330.end_s)  # held to the drive's end
    drive = [t for t in node.logger.texts("info") if t.startswith("gaze: the drive's head")]
    assert drive[0].startswith("gaze: the drive's head: 17 writes,")
    before = build(monkeypatch)  # the metres off: the seconds alone, as before
    old = replay_tape(before, TAPE_0330, knobs={"path_tail_m": 0.0}, **LIVE_0330)
    before.close()
    assert [w for w in old if w.t < 40.0] == [w for w in writes if w.t < 40.0]  # nothing earlier
    assert [(w.t, w.source, w.pan_deg, w.tilt_deg) for w in old if 40.0 <= w.t < 42.0] == [
        (40.8, "nav.path", -23.1, 65.6)  # 0.10 m of plan left at 0.15 m/s and slowing
    ]


@pytest.mark.slow
def test_drive_0330_a_swinging_path_look_is_replaced_by_the_reverse_look_not_expired(
    node: Gaze,
) -> None:
    replay_tape(node, TAPE_0330, **LIVE_0330)
    looks = node._looks.looks
    assert [b.text() for b in looks if b.ended == "expired"] == []
    swung = next(b for b in looks if b.source == "nav.path" and abs(b.since - T0 - 13.1) < 1e-6)
    assert (round(swung.until - T0, 2), swung.ended) == (13.6, "preempted")
    taker = next(b for b in looks if abs(b.since - swung.until) < 1e-6)
    assert taker.source == "nav.reverse" and taker.frames >= 2
    # Every path look held the head until another look took it or the drive ended.
    assert {b.ended for b in looks if b.source == "nav.path"} == {"preempted"}
