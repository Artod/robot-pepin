"""The laptop's bridge watch: a dead route is found by counting messages, not routes.

rclpy is faked (``ros_stubs``) and so are the two things the node reaches outside itself — the
bridges' REST admin (``Admin``, which answers with the routes a real admin printed) and the
repair (``FakeRepair``, which records instead of restarting a container). The node is then
driven round by round exactly as the timer drives it on the laptop.
"""

from __future__ import annotations

import contextlib
import sys
import types
from typing import Any

import ros_stubs

RCLPY = ros_stubs.install()

from pepin_bringup import bridge_watch as watch_module  # noqa: E402
from pepin_bringup.bridge_watch import FLAGS, BridgeWatch, DockerRestart  # noqa: E402
from ros_stubs import Imu  # noqa: E402

from pepin.deployment import CONTAINER_STOP_TIMEOUT_S  # noqa: E402

BOARD = "10.0.0.187"
ADMIN = "http://pepin-zenoh:8000"
BOARD_ZID = "abee59d7f052e5eeffe2098f0b8ef347"
LAPTOP_ZID = "ed8b4614af3e4d0f8250fb60d94969c5"
IMU = "sensor_msgs/msg/Imu"


def router(zid: str) -> str:
    return f'[{{"key":"@/{zid}/router","value":{{"zid":"{zid}","sessions":[]}}}}]'


def routes(local_nodes: str = '["/depth_fusion"]', publishers: str = '["/base_bridge"]') -> str:
    """Both bridges' routes for /imu/data_raw as one network-wide admin reply."""
    return (
        f'[{{"key":"@/{BOARD_ZID}/ros2/route/topic/pub/imu/data_raw","value":'
        f'{{"dds_reader":"abc","local_nodes":{publishers},'
        f'"remote_routes":["{LAPTOP_ZID}:imu/data_raw"],"ros2_name":"/imu/data_raw",'
        f'"ros2_type":"{IMU}"}}}},'
        f'{{"key":"@/{LAPTOP_ZID}/ros2/route/topic/sub/imu/data_raw","value":'
        f'{{"dds_writer":"def","is_active":true,"local_nodes":{local_nodes},'
        f'"remote_routes":["{BOARD_ZID}:imu/data_raw"],'
        f'"ros2_name":"/imu/data_raw","ros2_type":"{IMU}"}}}}]'
    )


def readerless_board_route(topic: str = "scan") -> str:
    """A pub route of the BOARD's bridge as the admin printed it on 2026-09-15: the board's own
    publisher, a remote route naming the laptop's bridge, and no dds_reader at all — thirteen of
    these carried nothing while this watch said "dead routes 0"."""
    return (
        f'{{"key":"@/{BOARD_ZID}/ros2/route/topic/pub/{topic}","value":'
        f'{{"dds_reader":"","local_nodes":["/ldlidar_node"],'
        f'"remote_routes":["{LAPTOP_ZID}:{topic}"],"ros2_name":"/{topic}",'
        f'"ros2_type":"sensor_msgs/msg/LaserScan"}}}}'
    )


def dead_vo() -> str:
    """The laptop bridge's pub route for /vo as the admin printed it on 2026-09-14: a publisher,
    the board's matching route, and no DDS reader at all."""
    return (
        f'{{"key":"@/{LAPTOP_ZID}/ros2/route/topic/pub/vo","value":'
        f'{{"dds_reader":"","local_nodes":["/visual_odometry"],'
        f'"remote_routes":["{BOARD_ZID}:vo"],"ros2_name":"/vo",'
        f'"ros2_type":"nav_msgs/msg/Odometry"}}}}'
    )


CONFIG = (
    f'[{{"key":"@/{LAPTOP_ZID}/ros2/config","value":{{"allow":{{"subscribers":'
    '"^^/(imu/data_raw|scan)$$"}}}]'
)


class Admin:
    """The REST admin both bridges answer with, as a table of URL suffix -> body."""

    def __init__(self) -> None:
        self.board_zid: str | None = BOARD_ZID
        self.routes = routes()
        self.asked: list[str] = []

    def fetch(self, url: str, timeout_s: float = 3.0) -> str | None:
        self.asked.append(url)
        if url.endswith("/@/local/router"):
            if url.startswith(ADMIN):
                return router(LAPTOP_ZID)
            return None if self.board_zid is None else router(self.board_zid)
        if url.endswith("/ros2/config"):
            return CONFIG
        if url.endswith("/ros2/route/**"):
            return self.routes
        return None


class FakeRepair:
    """A repair that records instead of restarting the bridge container."""

    def __init__(self, present: bool = True) -> None:
        self.present = present
        self.restarts = 0

    def available(self) -> bool:
        return self.present

    def restart(self) -> str:
        self.restarts += 1
        return f"restarted pepin-zenoh: HTTP 204 ok ({self.restarts})"


class ExitsError(Exception):
    """The process ending: the node calls ``exit_with`` and the test catches it."""


def build(
    monkeypatch: Any, repair: Any, admin: Any = None, grace_s: float = 0.0
) -> tuple[BridgeWatch, Admin, list[int]]:
    """A watch with its admin and repair faked, its worker thread stopped: the test calls
    ``round`` itself, as the timer would. ``grace_s`` is the startup grace, zero unless the
    test is about it."""
    admin = admin or Admin()
    codes: list[int] = []

    def exit_with(code: int) -> None:
        codes.append(code)
        raise ExitsError(str(code))

    monkeypatch.setattr(watch_module, "fetch", admin.fetch)
    monkeypatch.setattr(watch_module, "route_count", lambda board: 24)
    monkeypatch.setattr(watch_module, "wait_for_routes", lambda board, expected: 24)
    monkeypatch.setitem(
        sys.modules,
        "rosidl_runtime_py.utilities",
        types.SimpleNamespace(get_message=lambda name: Imu),
    )
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py", types.ModuleType("rosidl_runtime_py"))
    node = BridgeWatch(BOARD, ADMIN, repair=repair, exit_with=exit_with, grace_s=grace_s)
    node.close()  # the worker thread: this test is the one driving the rounds
    return node, admin, codes


def drive(node: BridgeWatch, admin: Admin, seconds: list[float], messages: int = 0) -> None:
    """Run a round at each moment, delivering ``messages`` messages in between."""
    for now in seconds:
        node.round(now)
        node.attach()
        for _ in range(messages):
            node.count("/imu/data_raw")


def test_a_flowing_topic_is_followed_once_and_never_repaired(monkeypatch: Any) -> None:
    """The watch subscribes to what both bridges say should arrive, with the QoS the topic is
    pinned to (/imu/data_raw is RELIABLE on both sides or the route's is a race), and a topic
    whose counter moves is never touched."""
    repair = FakeRepair()
    node, admin, codes = build(monkeypatch, repair)
    drive(node, admin, [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0], messages=48)
    assert "/imu/data_raw" in node.subs, "the watch follows it"
    assert node.subs["/imu/data_raw"][0] is Imu, "the type comes from the route, not a guess"
    assert repair.restarts == 0 and codes == []


def test_a_route_that_carries_nothing_restarts_the_bridge_alone(monkeypatch: Any) -> None:
    """The failure of 2026-09-12 and 2026-09-13: route on both admins, publisher on the far
    side, zero messages. The repair is the bridge container, not this half."""
    repair = FakeRepair()
    node, admin, codes = build(monkeypatch, repair)
    drive(node, admin, [0.0, 5.0, 10.0, 15.0], messages=0)
    assert repair.restarts == 0, "not yet: the patience is twenty seconds"
    drive(node, admin, [20.0], messages=0)
    assert repair.restarts == 1 and codes == [], "the bridge restarted; this half lives on"


def test_a_silence_that_survives_the_restart_escalates_to_the_half(monkeypatch: Any) -> None:
    """The ladder's last rung, with the middle one (the kick to the board) switched off."""
    repair = FakeRepair()
    node, admin, codes = build(monkeypatch, repair)
    node._switches.set("half_restart", True)  # the old escalation, asked for explicitly
    node._switches.set("bridge_kick", False)
    drive(node, admin, [0.0, 20.0], messages=0)
    assert repair.restarts == 1
    drive(node, admin, [30.0, 60.0, 100.0], messages=0)  # inside the cooldown: nothing happens
    assert repair.restarts == 1 and codes == []
    with contextlib.suppress(ExitsError):
        drive(node, admin, [130.0], messages=0)
    assert codes == [watch_module.BRIDGE_CHANGED_EXIT], "the old action, once the gentle one failed"


def test_without_the_docker_socket_the_repair_is_the_old_one(monkeypatch: Any) -> None:
    repair = FakeRepair(present=False)
    node, admin, codes = build(monkeypatch, repair)
    node._switches.set("half_restart", True)  # the old escalation, asked for explicitly
    with contextlib.suppress(ExitsError):
        drive(node, admin, [0.0, 20.0], messages=0)
    assert repair.restarts == 0 and codes == [watch_module.BRIDGE_CHANGED_EXIT]


def test_a_topic_nobody_here_reads_is_never_repaired(monkeypatch: Any) -> None:
    """The watch is not a consumer: a route whose only local node is the watch itself carries
    nothing because nobody wants it, and restarting a bridge would not change that."""
    repair = FakeRepair()
    admin = Admin()
    admin.routes = routes(local_nodes='["/bridge_watch"]')
    node, admin, codes = build(monkeypatch, repair, admin)
    drive(node, admin, [0.0, 20.0, 40.0], messages=0)
    assert repair.restarts == 0 and codes == []
    assert "/imu/data_raw" not in node.subs, "and it never subscribed: no route is kept alive"


def test_a_far_side_with_no_publisher_is_not_a_dead_route(monkeypatch: Any) -> None:
    repair = FakeRepair()
    admin = Admin()
    admin.routes = routes(publishers="[]")
    node, admin, codes = build(monkeypatch, repair, admin)
    drive(node, admin, [0.0, 20.0, 40.0], messages=0)
    assert repair.restarts == 0 and codes == []


def test_a_new_board_bridge_restarts_the_laptop_s_bridge_not_the_half(monkeypatch: Any) -> None:
    """The fault this watch already had an answer for, with the answer it earned on 2026-09-14:
    restarting the whole half on a board-bridge change left /vo with a route and no DDS reader,
    and what cured it was restarting the laptop's bridge alone with the nodes up."""
    repair = FakeRepair()
    node, admin, codes = build(monkeypatch, repair)
    drive(node, admin, [0.0], messages=48)
    admin.board_zid = "0000000000000000000000000000ffff"
    node.round(5.0)
    assert repair.restarts == 1 and codes == [], "the bridge alone; this half lives on"


def test_a_new_board_bridge_restarts_the_half_when_the_gentle_repair_is_off(
    monkeypatch: Any,
) -> None:
    """The old action stays one flag away (CLAUDE.md rule 19)."""
    repair = FakeRepair()
    node, admin, codes = build(monkeypatch, repair)
    node._switches.set("bridge_restart", False)
    node._switches.set("half_restart", True)  # the old escalation, asked for explicitly
    drive(node, admin, [0.0], messages=48)
    admin.board_zid = "0000000000000000000000000000ffff"
    with contextlib.suppress(ExitsError):
        node.round(5.0)
    assert codes == [watch_module.BRIDGE_CHANGED_EXIT] and repair.restarts == 0


def test_a_route_with_no_dds_endpoint_is_dead_and_the_bridge_is_restarted(
    monkeypatch: Any,
) -> None:
    """2026-09-14: the laptop bridge's pub route for /vo had the publisher, had the board's
    matching route, and had no dds_reader. No counter can find that — the topic is published
    here, not received here — so the endpoints are read straight off the admin."""
    repair = FakeRepair()
    admin = Admin()
    admin.routes = routes()[:-1] + "," + dead_vo() + "]"
    node, admin, codes = build(monkeypatch, repair, admin)
    drive(node, admin, [0.0, 10.0], messages=48)
    assert repair.restarts == 0, "not yet: a route caught between its creation and its endpoint"
    drive(node, admin, [20.0], messages=48)
    assert repair.restarts == 1 and codes == [], "the bridge alone, with every topic flowing"


def test_dead_routes_off_leaves_the_endpoints_unjudged(monkeypatch: Any) -> None:
    repair = FakeRepair()
    admin = Admin()
    admin.routes = routes()[:-1] + "," + dead_vo() + "]"
    node, admin, codes = build(monkeypatch, repair, admin)
    node._switches.set("dead_routes", False)
    drive(node, admin, [0.0, 20.0, 40.0], messages=48)
    assert repair.restarts == 0 and codes == []


def test_the_report_line_counts_the_dead_routes(monkeypatch: Any) -> None:
    repair = FakeRepair()
    node, admin, _codes = build(monkeypatch, repair)
    drive(node, admin, [0.0], messages=48)
    node.round(60.0)
    line = next(line for line in node.get_logger().texts("info") if "topics Hz" in line)
    assert "dead routes 0" in line and "dead_routes=on" in line


def test_flow_watch_off_leaves_the_watch_as_it_was(monkeypatch: Any) -> None:
    repair = FakeRepair()
    node, admin, codes = build(monkeypatch, repair)
    node._switches.set("flow_watch", False)
    drive(node, admin, [0.0, 20.0, 40.0, 60.0], messages=0)
    assert repair.restarts == 0 and codes == [] and not node.subs


def test_the_docker_repair_asks_the_daemon_to_restart_the_bridge_container() -> None:
    """The only handle on a container beside this one is the daemon's socket: plain HTTP over
    /var/run/docker.sock, no docker CLI in the image."""
    asked: list[tuple[str, str]] = []

    class Reply:
        status = 204

        def read(self) -> bytes:
            return b""

    class Connection:
        def request(self, method: str, path: str) -> None:
            asked.append((method, path))

        def getresponse(self) -> Reply:
            return Reply()

        def close(self) -> None:
            asked.append(("close", ""))

    repair = DockerRestart("pepin-zenoh", connect=Connection)
    assert repair.available()
    assert "restarted pepin-zenoh: HTTP 204" in repair.restart()
    assert asked[0] == (
        "POST",
        f"/containers/pepin-zenoh/restart?t={CONTAINER_STOP_TIMEOUT_S:g}",
    ), "a repair is a stop like any other: the daemon gets the repo's one stop window"
    assert asked[-1][0] == "close", "the socket is closed even when the daemon refuses"
    assert not DockerRestart("pepin-zenoh", socket_path="/no/such/socket").available()


def test_a_bridged_topic_is_subscribed_with_the_qos_both_sides_must_use() -> None:
    """The one-line fix behind the rate: the board writes /imu/data_raw RELIABLE ten deep, so
    every reader of it here must ask for the same or the route's QoS is decided by a race."""
    from pepin_bringup.node_kit import bridged_qos_profile
    from rclpy.qos import ReliabilityPolicy

    imu = bridged_qos_profile("/imu/data_raw")
    assert imu.reliability == ReliabilityPolicy.RELIABLE and imu.depth == 10
    assert bridged_qos_profile("/scan") is not imu, "no rule: the sensor-data default, as before"


def test_nothing_is_repaired_while_this_half_is_still_coming_up(monkeypatch: Any) -> None:
    """For the first ten seconds after a restart the admin still lists the previous containers'
    nodes on every route (the DDS lease), so a topic can look due and carry nothing through no
    fault of the bridge. The watch waits a minute before it touches anything."""
    repair = FakeRepair()
    node, admin, codes = build(monkeypatch, repair, grace_s=watch_module.STARTUP_GRACE_S)
    drive(node, admin, [0.0, 20.0, 40.0, 59.0], messages=0)
    assert repair.restarts == 0 and codes == [], "silent, but this half is younger than a minute"
    drive(node, admin, [61.0], messages=0)
    assert repair.restarts == 1


def test_a_type_this_image_cannot_resolve_is_reported_and_never_called_dead(
    monkeypatch: Any,
) -> None:
    """An rtabmap message on a container without rtabmap_msgs: the watch says so once and stops
    expecting it, instead of restarting a bridge for a topic it cannot read."""
    repair = FakeRepair()
    node, admin, codes = build(monkeypatch, repair)

    def no_class(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setitem(
        sys.modules, "rosidl_runtime_py.utilities", types.SimpleNamespace(get_message=no_class)
    )
    drive(node, admin, [0.0, 20.0, 40.0], messages=0)
    assert repair.restarts == 0 and codes == [] and not node.subs
    assert any("no message class" in line for line in node.get_logger().texts("warning"))


def test_the_report_line_is_a_rate_per_topic_and_the_switches(monkeypatch: Any) -> None:
    repair = FakeRepair()
    node, admin, _codes = build(monkeypatch, repair)
    drive(node, admin, [0.0], messages=48)
    node.round(60.0)
    line = next(line for line in node.get_logger().texts("info") if "topics Hz" in line)
    assert "imu/data_raw 0.8" in line, "48 messages over the minute since the last line"
    assert "flow_watch=on" in line and "bridge_restart=on" in line


def test_every_switch_is_in_the_table_and_printed() -> None:
    assert {flag.name for flag in FLAGS} == {
        "flow_watch",
        "half_restart",
        "flow_silence_s",
        "dead_routes",
        "board_routes",
        "bridge_kick",
        "bridge_restart",
    }
    assert all(flag.why and flag.on_when and flag.off_when for flag in FLAGS)


# ---- the board's own routes, and the kick that mends them ---------------------------------


def with_readerless_board() -> Admin:
    """An admin whose board bridge has a pub route with no reader beside a healthy /imu route."""
    admin = Admin()
    admin.routes = routes()[:-1] + "," + readerless_board_route() + "]"
    return admin


def test_the_board_s_own_readerless_route_is_a_fault_this_side_can_see(monkeypatch: Any) -> None:
    """2026-09-15: every topic stopped and the watch said "dead routes 0" — it judged this
    bridge's routes alone. The board's are in the same network-wide reply."""
    repair = FakeRepair()
    node, admin, codes = build(monkeypatch, repair, with_readerless_board())
    drive(node, admin, [0.0, 10.0], messages=48)
    assert repair.restarts == 0, "not yet: a route caught between its creation and its endpoint"
    drive(node, admin, [20.0], messages=48)
    assert repair.restarts == 1 and codes == [], "the gentle repair first, as for any fault"


def test_board_routes_off_leaves_the_far_side_unjudged(monkeypatch: Any) -> None:
    repair = FakeRepair()
    node, admin, codes = build(monkeypatch, repair, with_readerless_board())
    node._switches.set("board_routes", False)
    drive(node, admin, [0.0, 20.0, 40.0], messages=48)
    assert repair.restarts == 0 and codes == []


def test_the_report_line_counts_the_board_s_routes_without_a_reader(monkeypatch: Any) -> None:
    repair = FakeRepair()
    node, admin, _codes = build(monkeypatch, repair, with_readerless_board())
    drive(node, admin, [0.0], messages=48)
    node.round(60.0)
    line = next(line for line in node.get_logger().texts("info") if "topics Hz" in line)
    assert "BOARD ROUTES WITHOUT A READER 1 [/scan]" in line and "board_routes=on" in line


def test_a_fault_that_survives_the_gentle_repair_kicks_the_board(monkeypatch: Any) -> None:
    """The repair invariant: this side's bridge first, the board's after it — of two bridges the
    one that starts last is the one that gets working routes."""
    repair = FakeRepair()
    node, admin, codes = build(monkeypatch, repair, with_readerless_board())
    drive(node, admin, [0.0, 20.0], messages=48)
    assert repair.restarts == 1 and not node.pubs["/bridge/kick"].sent, "the bridge alone first"
    drive(node, admin, [25.0, 45.0], messages=48)  # the board's route is still readerless
    sent = node.pubs["/bridge/kick"].sent
    assert len(sent) == 1 and "/scan" in sent[0].data, "one kick, saying what is wrong"
    assert codes == [] and repair.restarts == 1, "this half and its bridge are left alone"


def test_the_board_bridge_that_comes_back_after_a_kick_is_not_a_new_fault(
    monkeypatch: Any,
) -> None:
    """The board's bridge restarts with a new zenoh id BY CONSTRUCTION, so a watch that repaired
    on that would restart the two bridges for ever."""
    repair = FakeRepair()
    node, admin, codes = build(monkeypatch, repair, with_readerless_board())
    drive(node, admin, [0.0, 20.0, 25.0, 45.0], messages=48)
    assert len(node.pubs["/bridge/kick"].sent) == 1
    admin.board_zid = "0000000000000000000000000000ffff"  # the kicked bridge, back
    admin.routes = routes()  # ...with its reader this time
    node.round(70.0)
    assert repair.restarts == 1 and codes == [], "no repair: this identity change was ours"
    assert any("came back after our kick" in line for line in node.get_logger().texts("info"))


def test_the_kick_is_off_when_the_switch_is_off(monkeypatch: Any) -> None:
    repair = FakeRepair()
    node, admin, codes = build(monkeypatch, repair, with_readerless_board())
    node._switches.set("bridge_kick", False)
    drive(node, admin, [0.0, 20.0, 25.0, 45.0, 65.0], messages=48)
    assert not node.pubs["/bridge/kick"].sent and repair.restarts == 1 and codes == []


# ---- the cooldown state machine ------------------------------------------------------------


def test_the_cooldown_is_armed_once_and_then_runs_out(monkeypatch: Any) -> None:
    """The bug of 2026-09-15: every failing round (five seconds apart) re-armed the 120 s
    cooldown, so the gentle repair — allowed once per cooldown — never ran a second time at all.

    The fault here is the board's readerless route, whose patience is ``flow_silence_s`` on this
    watch's own clocks, so the timeline is readable: a repair at 20 s, the ladder spent at 45 s
    (the kick is switched off), and the next gentle repair as soon as 120 s have passed since
    then — not 120 s after the last failing round.
    """
    repair = FakeRepair()
    node, admin, codes = build(monkeypatch, repair, with_readerless_board())
    node._switches.set("bridge_kick", False)  # the ladder's middle rung out of the way
    drive(node, admin, [0.0, 20.0], messages=48)
    assert repair.restarts == 1, "the gentle repair"
    drive(node, admin, [25.0, 45.0], messages=48)  # still readerless: the ladder is spent, cooling
    drive(node, admin, [50.0, 80.0, 110.0, 140.0, 160.0], messages=48)
    assert repair.restarts == 1, "still cooling down"
    assert any("cooling down" in line for line in node.get_logger().texts("error"))
    drive(node, admin, [170.0], messages=48)
    assert repair.restarts == 2, "the cooldown ran out because no failing round pushed it forward"
    assert codes == [], "and this half was never touched"
