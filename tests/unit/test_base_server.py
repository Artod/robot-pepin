"""The base server's core against a fake bus: arming, deadman, odometry, idle release."""

from test_base import CFG, FakeBus

from pepin.base import LEFT, RIGHT
from pepin.base_link import decode_state
from pepin.base_server import BaseServerCore


class PingableBus(FakeBus):
    """FakeBus plus a ping roster: odd ids answer, even ids are silent."""

    def ping(self, motor: str, num_retry: int = 0) -> int | None:
        return 0 if motor in (LEFT, "servo1") else None


def make_core() -> tuple[BaseServerCore, PingableBus]:
    bus = PingableBus()
    core = BaseServerCore(bus, CFG, servo_names=[LEFT, RIGHT, "servo1"])
    core.tick(0.0)  # priming read
    return core, bus


def test_first_twist_arms_the_wheels_and_writes_the_velocity() -> None:
    core, bus = make_core()
    assert not core.armed and not bus.torque
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=1.0)
    assert core.armed and bus.torque[-1][0] == "on"
    assert bus.writes[-1][0] == "Goal_Velocity" and core.moving


def test_deadman_stops_the_wheels_when_commands_stop_arriving() -> None:
    core, bus = make_core()
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=1.0)
    core.tick(1.3)
    assert core.moving and not core.deadman  # 0.3 s: still fine
    core.tick(1.6)
    assert not core.moving and core.deadman
    assert bus.writes[-1] == ("Goal_Velocity", {LEFT: 0, RIGHT: 0})
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=1.7)
    assert core.moving and not core.deadman  # a new command re-arms it


def test_idle_wheels_are_released_so_the_cart_can_be_pushed() -> None:
    core, bus = make_core()
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=1.0)
    core.command({"cmd": "stop"}, now=2.0)
    core.tick(10.9)
    assert core.armed  # 9.9 s after the last motion (at 1.0)
    core.tick(11.1)
    assert not core.armed and bus.torque[-1][0] == "off"


def test_ticks_integrate_odometry_and_snapshots_report_travel_once() -> None:
    core, bus = make_core()
    ticks_per_m = 4096 / (3.141592653589793 * 0.125)  # from the test geometry: 0.125 m wheels
    # Both wheels roll +10 cm forward; the left motor is mirrored (direction -1) in CFG.
    bus.positions[LEFT] = -round(0.10 * ticks_per_m)
    bus.positions[RIGHT] = round(0.10 * ticks_per_m)
    core.tick(1.0)
    state = decode_state(core.snapshot(1.0), received_at=1.0)
    assert abs(state.pose.x - 0.10) < 0.002 and abs(state.pose.y) < 1e-6
    assert abs(state.d_left_m - 0.10) < 0.002 and abs(state.d_right_m - 0.10) < 0.002
    again = decode_state(core.snapshot(1.05), received_at=1.05)
    assert again.d_left_m == 0.0  # travel is reported once, not accumulated forever
    assert again.pose.x == state.pose.x


def test_ping_reports_every_servo_in_the_roster() -> None:
    core, _ = make_core()
    reply = core.command({"cmd": "ping"}, now=1.0)
    assert reply == {"type": "pong", "servos": {LEFT: True, RIGHT: False, "servo1": True}}


def test_ping_is_refused_while_the_wheels_turn() -> None:
    core, _ = make_core()
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=1.0)
    assert core.command({"cmd": "ping"}, now=1.1) == {"type": "pong", "busy": True}
    core.command({"cmd": "stop"}, now=1.2)
    assert "servos" in (core.command({"cmd": "ping"}, now=1.3) or {})


def test_idle_release_counts_from_the_last_motion_not_the_last_message() -> None:
    core, bus = make_core()
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=1.0)
    t = 1.05
    while t < 11.5:  # a teleop loop keeps sending zero twists as its heartbeat
        core.command({"cmd": "twist", "v": 0.0, "w": 0.0}, now=t)
        core.tick(t)
        t += 0.05
    assert not core.armed and bus.torque[-1][0] == "off"  # released 10 s after the last motion


def test_release_from_a_leaving_client_stops_without_touching_the_deadman_clock() -> None:
    core, bus = make_core()
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=1.0)
    core.command({"cmd": "release"}, now=1.2)
    assert not core.moving and bus.writes[-1] == ("Goal_Velocity", {LEFT: 0, RIGHT: 0})
    assert core._last_command_at == 1.0


def test_serve_end_to_end_over_localhost() -> None:
    """A real client drives the served core: states flow, a twist moves it, leaving releases it."""
    import threading
    import time

    from pepin.base_link import BaseClient
    from pepin.base_server import serve
    from pepin.kinematics import Twist
    from pepin.streams import JsonLinesServer

    core, _ = make_core()
    server = JsonLinesServer(0, on_last_client_left={"cmd": "release"}).start()
    stop = threading.Event()
    worker = threading.Thread(target=serve, args=(core, server, 50.0, 20.0, stop), daemon=True)
    worker.start()
    client = BaseClient("127.0.0.1", server.port).start()
    first = client.wait_for_state(3.0)
    assert first is not None and not first.moving
    client.set_twist(Twist(0.1, 0.0))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not core.moving:
        time.sleep(0.01)
    assert core.moving and core.armed
    assert client.ping(timeout_s=2.0) == {}  # moving: the roster is not pinged
    client.stop()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and core.moving:
        time.sleep(0.01)
    servos = client.ping(timeout_s=2.0)
    assert servos is not None and servos.get("left") is True
    client.set_twist(Twist(0.1, 0.0))
    while time.monotonic() < deadline + 2.0 and not core.moving:
        time.sleep(0.01)
    client.close()  # the only client leaves: the core must release the wheels
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and core.moving:
        time.sleep(0.01)
    assert not core.moving
    stop.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive() and not core.armed
