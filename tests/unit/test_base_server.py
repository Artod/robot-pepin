"""The base server's core against a fake bus: arming, deadman, odometry, idle release, the neck."""

from pathlib import Path

import pytest
from test_base import CFG, FakeBus

from pepin.base import LEFT, RIGHT
from pepin.base_link import decode_state
from pepin.base_server import BaseServerCore, NeckMover, NeckReader
from pepin.neck import NeckConfig

REPO = Path(__file__).resolve().parents[2]


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
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not core.moving:
        time.sleep(0.01)
    assert core.moving
    client.close()  # the only client leaves: the core must release the wheels
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and core.moving:
        time.sleep(0.01)
    assert not core.moving
    stop.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive() and not core.armed


def test_malformed_client_lines_do_not_stop_the_wheel_loop() -> None:
    import socket
    import threading
    import time

    from pepin.base_server import serve
    from pepin.streams import JsonLinesServer

    core, _ = make_core()
    server = JsonLinesServer(0).start()
    stop = threading.Event()
    worker = threading.Thread(target=serve, args=(core, server, 50.0, 20.0, stop), daemon=True)
    worker.start()
    with socket.create_connection(("127.0.0.1", server.port), timeout=2.0) as raw:
        raw.sendall(b'[1, 2]\n"hi"\nnot json at all\n{"cmd": "twist", "v": "fast", "w": 0}\n')
        raw.sendall(b'{"cmd": "twist", "v": 0.1, "w": 0.0}\n')
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not core.moving:
            time.sleep(0.01)
        assert core.moving and worker.is_alive()
    stop.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()


class NeckBus(PingableBus):
    """FakeBus with neck servos on it: counts the reads, serves their operating mode, and can
    fall silent on the neck only."""

    def __init__(self) -> None:
        super().__init__()
        self.positions.update({"neck": 2048, "head": 2360})
        self.modes = {"neck": 0, "head": 0}  # 0: position mode, the only one a goal makes sense in
        self.neck_reads = 0
        self.neck_silent = False
        self.torque_silent = False

    def enable_torque(self, motors: list[str] | None = None) -> None:
        self._answers_torque()
        super().enable_torque(motors)

    def disable_torque(self, motors: list[str] | None = None) -> None:
        self._answers_torque()
        super().disable_torque(motors)

    def _answers_torque(self) -> None:
        if self.torque_silent:
            raise TimeoutError("no reply from ids [9] after 2 attempts")

    def sync_read(
        self, data_name: str, motors: list[str], *, normalize: bool = True
    ) -> dict[str, int]:
        if data_name == "Operating_Mode":
            return {m: self.modes[m] for m in motors}
        if "neck" in motors:
            self.neck_reads += 1
            if self.neck_silent:
                raise TimeoutError("no reply from ids [9, 10] after 2 attempts")
        return super().sync_read(data_name, motors, normalize=normalize)


def make_neck_core() -> tuple[BaseServerCore, NeckBus]:
    bus = NeckBus()
    core = BaseServerCore(
        bus,
        CFG,
        servo_names=[LEFT, RIGHT, "neck", "head"],
        neck=NeckReader(bus, ("neck", "head"), period_s=0.05, retry_s=5.0),
    )
    core.tick(0.0)
    return core, bus


def test_the_neck_command_answers_the_encoders_from_one_bus_read_per_period() -> None:
    core, bus = make_neck_core()
    first = core.command({"cmd": "neck"}, now=1.0)
    assert first is not None and first["type"] == "neck"
    assert (first["pan_ticks"], first["tilt_ticks"]) == (2048, 2360)
    assert first["age_s"] == 0.0 and "error" not in first and bus.neck_reads == 1
    bus.positions["head"] = 2400
    cached = core.command({"cmd": "neck"}, now=1.02)  # inside the 50 ms period: no bus traffic
    assert cached is not None and cached["tilt_ticks"] == 2360 and bus.neck_reads == 1
    assert cached["age_s"] == pytest.approx(0.02)
    fresh = core.command({"cmd": "neck"}, now=1.06)
    assert fresh is not None and fresh["tilt_ticks"] == 2400 and bus.neck_reads == 2
    assert "read_ms" in fresh
    # the wheel path is untouched: no neck read happens on a tick
    core.tick(1.1)
    assert bus.neck_reads == 2


def test_a_silent_neck_answers_an_error_and_is_retried_only_at_rest_after_a_pause() -> None:
    core, bus = make_neck_core()
    bus.neck_silent = True
    reply = core.command({"cmd": "neck"}, now=1.0)
    assert reply is not None and "error" in reply and "pan_ticks" not in reply
    assert bus.neck_reads == 1
    assert core.command({"cmd": "neck"}, now=2.0) == reply and bus.neck_reads == 1  # held
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=3.0)
    core.command({"cmd": "neck"}, now=9.0)
    assert bus.neck_reads == 1, "a silent servo is never retried while the wheels turn"
    core.command({"cmd": "stop"}, now=9.5)
    bus.neck_silent = False
    healed = core.command({"cmd": "neck"}, now=9.6)
    assert healed is not None and healed["pan_ticks"] == 2048 and bus.neck_reads == 2
    assert "error" not in healed


def test_stale_ticks_ride_along_with_the_error_of_a_servo_that_fell_silent() -> None:
    core, bus = make_neck_core()
    core.command({"cmd": "neck"}, now=1.0)
    bus.neck_silent = True
    reply = core.command({"cmd": "neck"}, now=2.0)
    assert reply is not None and reply["pan_ticks"] == 2048 and reply["age_s"] == pytest.approx(1.0)
    assert "error" in reply


def test_without_a_neck_the_command_answers_an_error_not_a_crash() -> None:
    core, _ = make_core()
    reply = core.command({"cmd": "neck"}, now=1.0)
    assert reply is not None and reply["type"] == "neck" and "error" in reply


def test_the_neck_ids_come_from_the_file_and_a_missing_file_costs_only_the_neck(
    tmp_path: Path,
) -> None:
    from pepin.base_server import load_neck_ids

    assert load_neck_ids(str(REPO / "config/neck.json")) == {"neck": 9, "head": 10}
    assert load_neck_ids(str(tmp_path / "absent.json")) == {}
    broken = tmp_path / "neck.json"
    broken.write_text('{"neck": {"id": "nine"}}')
    assert load_neck_ids(str(broken)) == {}


def test_a_neck_observer_leaving_does_not_release_the_wheels_but_the_driver_does() -> None:
    """The neck node is a second client of the base server: its arrival and departure must be
    invisible to the wheels, and the bridge's departure must still stop them at once."""
    import socket
    import threading
    import time

    from pepin.base_server import DRIVING_COMMANDS, serve
    from pepin.streams import JsonLinesServer

    core, _ = make_neck_core()
    server = JsonLinesServer(
        0, on_last_client_left={"cmd": "release"}, driving_commands=DRIVING_COMMANDS
    ).start()
    stop = threading.Event()
    worker = threading.Thread(target=serve, args=(core, server, 50.0, 20.0, stop), daemon=True)
    worker.start()
    driver = socket.create_connection(("127.0.0.1", server.port), timeout=2.0)
    observer = socket.create_connection(("127.0.0.1", server.port), timeout=2.0)
    observer.sendall(b'{"cmd": "neck"}\n')
    driver.sendall(b'{"cmd": "twist", "v": 0.1, "w": 0.0}\n')
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not core.moving:
        time.sleep(0.01)
    assert core.moving
    observer.close()
    time.sleep(0.2)
    assert core.moving, "an observer leaving is not the driver leaving"
    driver.close()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and core.moving:
        time.sleep(0.01)
    assert not core.moving, "the last driver left: the wheels are released"
    stop.set()
    worker.join(timeout=2.0)


def test_sigterm_sets_the_stop_event_so_the_wheels_are_released() -> None:
    """systemd stops the service with SIGTERM; serve() must get to its finally."""
    import os
    import signal

    from pepin.base_server import stop_on_sigterm

    previous = signal.getsignal(signal.SIGTERM)
    try:
        stop = stop_on_sigterm()
        assert not stop.is_set()
        os.kill(os.getpid(), signal.SIGTERM)
        assert stop.wait(2.0)
    finally:
        signal.signal(signal.SIGTERM, previous)


# -- the neck's write side ---------------------------------------------------------------------

NECK_CFG = NeckConfig.from_json(REPO / "config/neck.json")


def make_move_core() -> tuple[BaseServerCore, NeckBus]:
    """A core that can both read and move the neck, against the repo's real config/neck.json."""
    bus = NeckBus()
    reader = NeckReader(bus, ("neck", "head"), period_s=0.05, retry_s=5.0)
    core = BaseServerCore(
        bus,
        CFG,
        servo_names=[LEFT, RIGHT, "neck", "head"],
        neck=reader,
        mover=NeckMover(bus, reader, NECK_CFG),
    )
    core.tick(0.0)
    return core, bus


def goals(bus: NeckBus) -> dict[str, int]:
    """Every Goal_Position written to the bus so far, by motor name."""
    written: dict[str, int] = {}
    for name, values in bus.writes:
        if name == "Goal_Position":
            written.update(values)
    return written


def test_a_move_writes_the_goal_then_answers_when_the_head_arrives() -> None:
    """The command itself answers nothing: the reply is born when the head stops, and leaves
    through take_replies (serve broadcasts it). Torque goes on before the goal and off after."""
    core, bus = make_move_core()
    assert (
        core.command({"cmd": "neck_goto", "pan_ticks": 2021, "tilt_ticks": 2311}, now=1.0) is None
    )
    assert bus.torque == [("on", ["neck", "head"])]
    assert goals(bus) == {"neck": 2021, "head": 2311}
    assert ("Goal_Velocity", {"neck": 400}) in bus.writes, "a profile speed, not a wheel command"
    core.tick(1.1)  # the servos are still on their way
    assert core.take_replies() == []
    bus.positions.update({"neck": 2021, "head": 2309})  # arrived, the tilt two ticks off
    core.tick(1.2)
    (reply,) = core.take_replies()
    assert reply["type"] == "neck_goto" and reply["reached"] is True
    assert (reply["pan_ticks"], reply["tilt_ticks"]) == (2021, 2309)
    assert reply["ms"] == pytest.approx(200.0) and reply["hold"] is False
    assert bus.torque[-1] == ("off", ["neck", "head"]), "the head is free to be pushed again"
    assert core.take_replies() == []  # taken once


def test_a_target_outside_the_configured_limits_is_refused_not_clamped() -> None:
    core, bus = make_move_core()
    reply = core.command({"cmd": "neck_goto", "pan_ticks": 4000}, now=1.0)
    assert reply is not None and reply["reached"] is False
    assert "257..3812" in reply["error"] and "4000" in reply["error"]
    assert bus.torque == [] and goals(bus) == {}, "nothing was energised, nothing was written"
    low = core.command({"cmd": "neck_goto", "tilt_ticks": 1000}, now=1.1)
    assert low is not None and "1814..3090" in low["error"]
    assert core.command({"cmd": "neck_goto"}, now=1.2) == {
        "type": "neck_goto",
        "reached": False,
        "error": "neck_goto needs pan_ticks or tilt_ticks",
    }


def test_a_head_that_never_arrives_gives_up_and_is_released() -> None:
    core, bus = make_move_core()
    core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=1.0)
    assert bus.torque == [("on", ["neck"])], "only the addressed servo is energised"
    core.tick(2.0)
    assert core.take_replies() == []
    core.tick(4.1)  # 3 s after the start: the head never moved (a jammed or unpowered servo)
    (reply,) = core.take_replies()
    assert reply["reached"] is False and reply["pan_ticks"] == 2048
    assert reply["ms"] == pytest.approx(3100.0)
    assert bus.torque[-1] == ("off", ["neck"])


def test_hold_leaves_the_servos_energised_until_the_server_lets_go() -> None:
    core, bus = make_move_core()
    core.command({"cmd": "neck_goto", "pan_ticks": 2048, "tilt_ticks": 2360, "hold": True}, now=1.0)
    core.tick(1.1)  # already there: reached on the first step
    (reply,) = core.take_replies()
    assert reply["reached"] is True and reply["hold"] is True
    assert bus.torque == [("on", ["neck", "head"])], "held: the torque stays on"
    core.release()  # shutdown must not leave a head energised with nobody to release it
    assert bus.torque[-1] == ("off", ["neck", "head"])


def test_neck_home_goes_to_the_reference_ticks_of_the_config() -> None:
    core, bus = make_move_core()
    ref = NECK_CFG.reference
    assert core.command({"cmd": "neck_home"}, now=1.0) is None
    assert goals(bus) == {"neck": ref.pan_ticks, "head": ref.tilt_ticks}


def test_home_without_reference_ticks_says_so_instead_of_moving() -> None:
    """The reference is a hardware reading that may be missing (null in the file); nothing is
    written then, because there is no pose to go to."""
    from dataclasses import replace

    bus = NeckBus()
    reader = NeckReader(bus, ("neck", "head"))
    unread = replace(NECK_CFG, reference=replace(NECK_CFG.reference, pan_ticks=None))
    core = BaseServerCore(bus, CFG, neck=reader, mover=NeckMover(bus, reader, unread))
    reply = core.command({"cmd": "neck_home"}, now=1.0)
    assert reply is not None and "reference ticks are unread" in reply["error"]
    assert goals(bus) == {} and bus.torque == []


def test_a_servo_left_in_velocity_mode_is_never_given_a_goal() -> None:
    """scripts/jog.py wheel writes velocity mode into a servo's EEPROM; in that mode the profile
    speed is a command and the head would turn until something broke."""
    core, bus = make_move_core()
    bus.modes["head"] = 1
    reply = core.command({"cmd": "neck_goto", "tilt_ticks": 2311}, now=1.0)
    assert reply is not None and "position mode" in reply["error"] and "'head': 1" in reply["error"]
    assert bus.torque == [] and goals(bus) == {}
    assert core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=1.1) is None  # pan is fine


def test_a_move_is_refused_while_the_wheels_turn_and_never_delays_the_deadman() -> None:
    core, bus = make_move_core()
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=1.0)
    refused = core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=1.05)
    assert refused is not None and refused["error"] == "the wheels are moving"
    core.command({"cmd": "stop"}, now=1.1)
    assert core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=1.2) is None
    assert core.command({"cmd": "neck_goto", "pan_ticks": 2100}, now=1.25) == {
        "type": "neck_goto",
        "reached": False,
        "error": "a neck move is already under way",
    }
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=1.3)  # driving, the head still moving
    core.tick(1.35)
    assert core.moving and not core.deadman
    core.tick(1.95)  # 0.65 s without a command: the deadman fires on time, mid-move
    assert core.deadman and not core.moving
    assert bus.writes[-1] == ("Goal_Velocity", {LEFT: 0, RIGHT: 0})


def test_a_move_command_without_a_neck_or_with_junk_in_it_answers_an_error() -> None:
    core, _ = make_core()
    reply = core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=1.0)
    assert reply is not None and reply["error"] == "no neck configured on this base server"
    assert core.command({"cmd": "neck_home"}, now=1.0) == reply
    movable, _ = make_move_core()
    junk = movable.command({"cmd": "neck_goto", "pan_ticks": "left a bit"}, now=1.0)
    assert junk is not None and junk["error"].startswith("bad target:")


def test_the_finished_move_reaches_the_client_that_asked_over_the_socket() -> None:
    """End to end: the answer outlives the request, so serve broadcasts it as a line of its own."""
    import json
    import socket
    import threading
    import time

    from pepin.base_server import DRIVING_COMMANDS, serve
    from pepin.streams import JsonLinesServer

    core, bus = make_move_core()
    server = JsonLinesServer(0, driving_commands=DRIVING_COMMANDS).start()
    stop = threading.Event()
    worker = threading.Thread(target=serve, args=(core, server, 50.0, 20.0, stop), daemon=True)
    worker.start()
    try:
        with socket.create_connection(("127.0.0.1", server.port), timeout=2.0) as raw:
            raw.sendall(b'{"cmd": "neck_goto", "pan_ticks": 2021, "tilt_ticks": 2311}\n')
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and "neck" not in goals(bus):
                time.sleep(0.01)
            bus.positions.update({"neck": 2021, "head": 2311})
            raw.settimeout(2.0)
            buffer, answer = b"", None
            while answer is None and time.monotonic() < deadline:
                buffer += raw.recv(4096)
                *lines, buffer = buffer.split(b"\n")
                for line in lines:
                    message = json.loads(line)
                    if message.get("type") == "neck_goto":
                        answer = message
            assert answer is not None and answer["reached"] is True
            assert (answer["pan_ticks"], answer["tilt_ticks"]) == (2021, 2311)
    finally:
        stop.set()
        worker.join(timeout=2.0)


def test_a_bus_that_will_not_take_the_write_refuses_the_move_and_stays_usable() -> None:
    core, bus = make_move_core()
    bus.torque_silent = True
    reply = core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=1.0)
    assert reply is not None and reply["error"].startswith("the bus refused the move:")
    assert goals(bus) == {} and bus.torque == []
    bus.torque_silent = False
    assert core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=1.1) is None  # not stuck


def test_a_servo_that_falls_silent_mid_move_is_answered_with_its_error() -> None:
    """The encoders go quiet halfway: the move cannot be confirmed, so it is answered as not
    reached with the bus error on it — and the torque-off that fails too is only logged."""
    core, bus = make_move_core()
    core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=1.0)
    bus.neck_silent = True
    bus.torque_silent = True
    core.tick(1.1)
    assert core.take_replies() == []
    core.tick(4.2)
    (reply,) = core.take_replies()
    assert reply["reached"] is False and "no reply from ids" in reply["error"]
    again = core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=4.3)
    assert again is not None and "bus refused" in again["error"]  # the bus is still out
