"""The base server's core against a fake bus: arming, deadman, odometry, idle release, the neck."""

import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from servo_wire import ServoWire
from test_base import CFG, FakeBus

from pepin.base import LEFT, RIGHT
from pepin.base_link import decode_state
from pepin.base_server import (
    MODE_CHECK_WINDOW_S,
    NECK_ACC_CEILING,
    NECK_BLOCK,
    BaseServerCore,
    PublishGrid,
    jog_ticks_s,
    neck_parts,
)
from pepin.feetech import FeetechTcpClient
from pepin.neck import NeckAngles, NeckConfig, acc_units, speed_ticks, ticks_for
from pepin.telemetry import LatencyTracker

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


def test_the_wheel_ceiling_is_answered_set_live_and_refused_outside_its_range() -> None:
    """ros/speed.sh's base half: the ceiling in force and the config's, a set that holds until
    the server restarts, a value outside pepin.speed's range refused with the ceiling unchanged,
    and none of it a driving command (a client that only asked never releases the wheels)."""
    from pepin.base_server import DRIVING_COMMANDS

    core, bus = make_core()
    assert core.command({"cmd": "max_wheel_speed"}, now=1.0) == {
        "type": "max_wheel_speed",
        "config_m_s": 0.30,
        "m_s": 0.30,
    }
    reply = core.command({"cmd": "max_wheel_speed", "m_s": 0.2}, now=1.0)
    assert reply is not None and (reply["m_s"], reply["was_m_s"]) == (0.2, 0.30)
    core.command({"cmd": "twist", "v": 0.3, "w": 0.0}, now=1.1)
    rim = bus.writes[-1][1][RIGHT] * CFG.geometry.m_per_tick
    assert rim == pytest.approx(0.2, abs=1e-3)
    for bad in (0.5, 0.05, "fast", True, None):
        reply = core.command({"cmd": "max_wheel_speed", "m_s": bad}, now=1.2)
        assert reply is not None and "error" in reply and reply["m_s"] == 0.2, bad
    assert "max_wheel_speed" not in DRIVING_COMMANDS


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


def _published(publish_hz: float, ticks: list[float]) -> list[float]:
    """The ticks a PublishGrid on a 50 Hz loop publishes on."""
    grid = PublishGrid(publish_hz, 50.0, ticks[0])
    return [t for t in ticks if grid.due(t)]


def test_a_state_line_goes_out_on_every_tick_at_the_tick_rate() -> None:
    """50 Hz on a 50 Hz loop is every tick, a late tick included: the old rule waited one
    period after the last line and then for the next tick, which gave 20 Hz as 16.7."""
    ticks = [i * 0.0201 for i in range(500)]  # the loop's own drift: each tick a little late
    assert len(_published(50.0, ticks)) == 500
    early = [0.0, 0.0195, 0.0391, 0.0602]  # a tick that wakes up a little early still counts
    assert _published(50.0, early) == early


def test_a_lower_rate_averages_out_to_exactly_that_rate() -> None:
    """20 Hz on 20 ms ticks alternates two and three ticks apart: 20 lines a second, not 16.7."""
    ticks = [i * 0.02 for i in range(500)]  # 10 s
    assert len(_published(20.0, ticks)) == 200


def test_a_stalled_loop_resumes_on_a_fresh_grid_without_a_burst() -> None:
    """A tick that took 0.4 s (a silent servo) publishes once, and the next one is a tick later,
    not the twenty lines the stall skipped."""
    ticks = [0.0, 0.02, 0.04, 0.44, 0.46, 0.48]
    assert _published(50.0, ticks) == ticks
    assert _published(20.0, ticks) == [0.0, 0.04, 0.44, 0.48]


def test_the_bus_p95_is_resorted_once_a_second_not_on_every_state_line() -> None:
    """Sorting the 512-sample window per 50 Hz line costs more than the line; the state lines of
    one second share one p95."""
    latency = LatencyTracker("bus")
    core = BaseServerCore(FakeBus(), CFG, latency=latency)
    core.tick(0.0)
    latency.add(0.004)
    assert core.snapshot(0.0)["bus_p95_ms"] == pytest.approx(4.0)
    latency.add(0.009)
    latency.add(0.009)
    assert core.snapshot(0.5)["bus_p95_ms"] == pytest.approx(4.0), "within the second: cached"
    assert core.snapshot(1.0)["bus_p95_ms"] == pytest.approx(9.0)


class SlowBus(FakeBus):
    """FakeBus whose encoder read takes 4 ms of a fake clock."""

    def __init__(self) -> None:
        super().__init__()
        self.now = 0.0

    def sync_read(
        self, data_name: str, motors: list[str], *, normalize: bool = True, **riders: Any
    ) -> dict[str, int]:
        self.now += 0.004
        return super().sync_read(data_name, motors, normalize=normalize, **riders)


def test_a_state_line_is_stamped_with_the_middle_of_its_encoder_read() -> None:
    """The bridge differences two poses over their stamps: the stamp is when the encoders were
    read, not when the tick began (a twist written first delays the read by milliseconds that
    are a tenth of a 20 ms gap); a failed read stamps the tick."""
    bus = SlowBus()
    core = BaseServerCore(bus, CFG, clock=lambda: bus.now)
    bus.now = 10.0
    core.tick(10.0)
    bus.now = 10.003  # the twist's write before the read
    core.tick(10.0)
    assert core.snapshot(10.0)["t"] == pytest.approx(10.005)


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


def test_wheels_that_never_turn_are_released_however_hard_they_are_commanded() -> None:
    """2026-09-14: Artem found the servos locked with the cart standing and had to power-cycle
    it. A controller (a stalled Nav2 after a cancelled drive) kept sending small non-zero twists,
    so the release clock — which counted from the last non-zero TWIST — was refreshed for ever
    while the encoders showed nothing at all. The encoders now have the last word."""
    core, bus = make_core()
    for i in range(1, 12):  # a twist every second, and a cart that does not move a millimetre
        core.command({"cmd": "twist", "v": 0.01, "w": 0.0}, now=float(i))
        core.tick(float(i) + 0.5)
        if i < 10:
            assert core.armed, "still trying"
    assert not core.armed and bus.torque[-1][0] == "off"

    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=20.0)
    assert core.armed and bus.torque[-1][0] == "on", "the next twist arms it again"


def test_a_cart_that_really_travels_keeps_its_torque() -> None:
    """The other half: the same commands with the wheels turning must never release them."""
    core, bus = make_core()
    ticks = 0
    for i in range(1, 30):
        core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=float(i))
        ticks += 30  # about 3 mm a second at 0.096 mm a tick: a slow creep, but travel
        bus.positions[LEFT], bus.positions[RIGHT] = -ticks, ticks
        core.tick(float(i) + 0.5)
        assert core.armed, f"travelling at second {i}"
    assert [state for state, _ in bus.torque] == ["on"]


def test_the_travel_release_can_be_turned_off() -> None:
    """The behaviour before 2026-09-14 stays reachable (config/base.json)."""
    bus = PingableBus()
    core = BaseServerCore(bus, CFG, servo_names=[LEFT, RIGHT], disarm_without_travel=False)
    core.tick(0.0)
    for i in range(1, 30):
        core.command({"cmd": "twist", "v": 0.01, "w": 0.0}, now=float(i))
        core.tick(float(i) + 0.5)
    assert core.armed, "commanded, so armed, however little the cart moved"


def test_encoder_noise_alone_never_counts_as_travel() -> None:
    """The threshold is signed travel, not the sum of its absolute values: a reading that
    jitters by a tick each way for ten seconds is a cart standing still — and at 50 Hz the sum
    of the absolute values would have reached the threshold in a second and armed it for ever.

    It also shows the one limit of the rule as it is asked for: a controller that keeps
    commanding at 20 Hz re-arms the wheels with its very next twist, so each release lasts until
    then. The torque does come off, and it stays off as soon as the commands stop.
    """
    core, bus = make_core()
    for i in range(1, 300):
        core.command({"cmd": "twist", "v": 0.01, "w": 0.0}, now=i * 0.05)
        bus.positions[LEFT] = i % 2
        bus.positions[RIGHT] = -(i % 2)
        core.tick(i * 0.05 + 0.02)
    releases = [state for state, _ in bus.torque].count("off")
    assert releases >= 1, "ten seconds of jitter released it"
    core.tick(26.0)  # the commands stop: ten seconds later it is released for good
    assert not core.armed


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
    """FakeBus with the neck's servos on it: every read records who rode in it, either neck servo
    can fall silent, the operating mode is served, and a write can fail like a lost link."""

    def __init__(self) -> None:
        super().__init__()
        self.positions.update({"neck": 2048, "head": 2360})
        self.modes = {"neck": 0, "head": 0}  # 0: position mode, the only one a goal makes sense in
        self.ceilings = {"neck": 50, "head": 50}  # Maximum_Acceleration as the servos came
        self.keeps_ceiling = True  # False: a servo that ignores the ceiling's write
        self.silent: set[str] = set()
        self.reads: list[tuple[str, list[str], tuple[str, ...], float]] = []
        self.link_down = False

    def sync_read(
        self,
        data_name: str,
        motors: list[str],
        *,
        normalize: bool = True,
        optional: Sequence[str] = (),
        optional_window_s: float = 0.0,
    ) -> dict[str, int]:
        self.reads.append((data_name, list(motors), tuple(optional), optional_window_s))
        if data_name == "Operating_Mode":
            return {m: self.modes[m] for m in [*motors, *optional] if m not in self.silent}
        if data_name == "Maximum_Acceleration":
            return {m: self.ceilings[m] for m in [*motors, *optional] if m not in self.silent}
        read = super().sync_read(data_name, motors, normalize=normalize)
        read.update({m: self.positions[m] for m in optional if m not in self.silent})
        return read

    def sync_write(self, data_name: str, values: dict[str, int], *, normalize: bool = True) -> None:
        if self.link_down:
            raise TimeoutError("bus link lost: connection reset")
        if data_name == "Maximum_Acceleration" and self.keeps_ceiling:
            self.ceilings.update(values)
        super().sync_write(data_name, values, normalize=normalize)

    def sync_write_block(self, data_names: Sequence[str], values: dict[str, Sequence[int]]) -> None:
        if self.link_down:
            raise TimeoutError("bus link lost: connection reset")
        assert tuple(data_names) == NECK_BLOCK
        super().sync_write_block(data_names, values)


NECK_CFG = NeckConfig.from_json(REPO / "config/neck.json")
REF = NECK_CFG.reference
TOP_SPEED = speed_ticks(NECK_CFG.motion.max_speed_deg_s)  # 299 deg/s: 3402 ticks/s
TOP_ACC = acc_units(NECK_CFG.motion.max_acc_deg_s2)  # 2232 deg/s^2: 254 units
HOME_TILT_RAD = math.radians(REF.pitch_deg)


def make_neck_core(
    cfg: NeckConfig = NECK_CFG, bus: NeckBus | None = None
) -> tuple[BaseServerCore, NeckBus]:
    """A core that reads and moves the neck (the repo's own config/neck.json), after its first
    tick — which heard the pair, read their mode and let them go — with the bus's log cleared."""
    bus = bus if bus is not None else NeckBus()
    encoders, mover = neck_parts(bus, cfg)
    core = BaseServerCore(
        bus, CFG, servo_names=[LEFT, RIGHT, "neck", "head"], neck=encoders, mover=mover
    )
    core.tick(0.0)
    bus.writes.clear()
    bus.reads.clear()
    return core, bus


def neck_writes(bus: NeckBus) -> list[tuple[str, dict[str, Any]]]:
    """Every write that addressed the neck, in order."""
    return [(name, v) for name, v in bus.writes if "neck" in v or "head" in v]


def goals(bus: NeckBus) -> dict[str, int]:
    """The newest goal written to each neck servo, by a Goal_Position write or a block."""
    written: dict[str, int] = {}
    for name, values in bus.writes:
        if name == "Goal_Position":
            written.update(values)
        elif name == "block":
            written.update({motor: row[2] for motor, row in values.items()})
    return written


def energised(bus: NeckBus) -> bool:
    """Whether the newest torque the neck was given is on: a block energises, a Torque_Enable 0
    lets go."""
    for name, values in reversed(neck_writes(bus)):
        if name == "block":
            return True
        if name == "Torque_Enable":
            return bool(values["neck"])
    return False


def block(pan: int, tilt: int, speed: int = TOP_SPEED, acc: int = TOP_ACC) -> tuple[str, Any]:
    """The one packet that energises, ramps, aims and paces both servos."""
    return ("block", {"neck": (1, acc, pan, 0, speed), "head": (1, acc, tilt, 0, speed)})


RELEASE = ("Torque_Enable", {"neck": 0, "head": 0})
CEILING = ("Maximum_Acceleration", {"neck": NECK_ACC_CEILING, "head": NECK_ACC_CEILING})


def target(core: BaseServerCore, now: float, pan_rad: float = 0.0, **extra: Any) -> Any:
    """One ``neck_target`` (tilt at the reference pitch unless given); its reply."""
    message = {"cmd": "neck_target", "pan_rad": pan_rad, "tilt_rad": HOME_TILT_RAD, **extra}
    return core.command(message, now=now)


# -- the neck's encoders ride the wheels' read -------------------------------------------------


def test_the_neck_rides_in_the_wheels_read_and_the_state_line_carries_its_ticks() -> None:
    """One sync_read a tick: the wheels mandatory, the neck after them as optional ids with a
    3 ms window; the state line carries both ticks under the odometry's own stamp, and the
    ``neck`` command is answered from that read without a bus transaction of its own."""
    core, bus = make_neck_core()
    core.tick(1.0)
    assert bus.reads == [("Present_Position", [LEFT, RIGHT], ("neck", "head"), 0.003)]
    line = core.snapshot(1.0)
    assert (line["pan_ticks"], line["tilt_ticks"]) == (2048, 2360)
    decode_state(line, received_at=1.0)  # the laptop's decoder takes the line as it is
    bus.reads.clear()
    reply = core.command({"cmd": "neck"}, now=1.01)
    assert reply is not None and (reply["pan_ticks"], reply["tilt_ticks"]) == (2048, 2360)
    assert reply["age_s"] == pytest.approx(0.01) and "error" not in reply
    assert bus.reads == [], "no bus traffic for the question"


def test_the_first_read_that_hears_the_neck_checks_its_mode_and_lets_it_go() -> None:
    """The mode is read once, bounded by its own window and never retried in the tick; a pair
    in position mode has its acceleration ceiling lifted and read back the same way; a pair
    nobody has told anything yet is let go — all of it unacknowledged packets."""
    bus = NeckBus()
    encoders, mover = neck_parts(bus, NECK_CFG)
    core = BaseServerCore(bus, CFG, neck=encoders, mover=mover)
    core.tick(0.0)
    assert bus.reads == [
        ("Present_Position", [LEFT, RIGHT], ("neck", "head"), 0.003),
        ("Operating_Mode", [], ("neck", "head"), MODE_CHECK_WINDOW_S),
        ("Maximum_Acceleration", [], ("neck", "head"), MODE_CHECK_WINDOW_S),
    ]
    assert neck_writes(bus) == [CEILING, RELEASE]
    core.tick(0.02)
    assert len(bus.reads) == 4 and len(neck_writes(bus)) == 2, "then nothing: one read a tick"


def test_the_acceleration_ceiling_is_lifted_or_its_refusal_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """2026-10-02: the servos keep Acceleration under Maximum_Acceleration (50 units, 439
    deg/s^2, as delivered), so every ramp asked above it ran at it. The check lifts it to the
    register's top; a servo that will not keep it is logged, and the head still moves."""
    core, bus = make_neck_core()
    assert bus.ceilings == {"neck": NECK_ACC_CEILING, "head": NECK_ACC_CEILING}
    stubborn = NeckBus()
    stubborn.keeps_ceiling = False
    with caplog.at_level("WARNING"):
        core, stubborn = make_neck_core(bus=stubborn)
    assert "ceiling reads {'neck': 50, 'head': 50}" in caplog.text
    assert target(core, 1.0) is None
    core.tick(1.0)
    assert energised(stubborn), "a low ceiling slows the head, it does not stop it"


def test_a_silent_neck_leaves_the_read_and_is_asked_again_every_five_seconds() -> None:
    """Three reads without both answers take the pair out of the read; it rides along again
    every retry_s, driving or not (a silent optional id costs its window, not 0.4 s), and the
    odometry never notices. Back, it is mode-checked and rewritten whole."""
    core, bus = make_neck_core()
    bus.silent = {"head"}  # one dead servo is a dead pair: no pose without both
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=1.0)
    asked = []
    now = 1.0
    for _ in range(5):
        bus.reads.clear()
        bus.positions[RIGHT] += 50  # the wheels keep turning
        core.tick(now)
        core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=now)
        asked.append(bus.reads[0][2] == ("neck", "head"))
        line = core.snapshot(now)
        assert "pan_ticks" not in line, "no ticks without both answers"
        now += 0.02
    assert asked == [True, True, True, False, False]
    assert core.snapshot(now)["dr"] == 0.0 and core._odom.pose.x != 0.0, "odometry carried on"
    reply = core.command({"cmd": "neck"}, now=now)
    assert reply is not None and "['head']" in reply["error"] and reply["pan_ticks"] == 2048
    bus.reads.clear()
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=1.04 + 5.0)
    core.tick(1.04 + 5.0)  # the retry rides along, the wheels still turning
    assert bus.reads[0][2] == ("neck", "head") and core.moving
    bus.silent = set()
    bus.writes.clear()
    core.tick(1.04 + 10.1)
    assert [r[0] for r in bus.reads[-3:]] == [
        "Present_Position",
        "Operating_Mode",
        "Maximum_Acceleration",
    ]
    assert neck_writes(bus) == [CEILING, RELEASE], "back: checked and lifted again, let go"


def test_a_neck_in_velocity_mode_is_never_given_a_goal() -> None:
    """scripts/jog.py wheel writes velocity mode into a servo's EEPROM; in that mode the speed
    register is a command and the head would turn until something broke."""
    bus = NeckBus()
    bus.modes["head"] = 1
    core, bus = make_neck_core(bus=bus)
    assert bus.ceilings == {"neck": 50, "head": 50}, "nothing written to a servo in that mode"
    refused = target(core, 1.0)
    assert (
        refused is not None
        and "position mode" in refused["error"]
        and "'head': 1" in refused["error"]
    )
    assert (
        core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=1.0)["error"] == refused["error"]
    )
    assert jog(core, 1.0, pan=1) == {"type": "neck_jog", "error": refused["error"]}
    core.tick(1.0)
    assert neck_writes(bus) == []


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
    """A client that only asks about or aims the head (a tool, the gaze arbiter) is a second
    client of the base server: its arrival and departure must be invisible to the wheels, and
    the bridge's departure must still stop them at once."""
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


# -- neck_target: the arbiter's stream ------------------------------------------------------


def test_a_target_is_one_packet_that_energises_ramps_aims_and_paces_both_servos() -> None:
    """Accepted silently, written on the tick: the whole block the first time (torque, ramp,
    goal, time 0, speed); after it only the goal, and nothing at all for the same goal."""
    core, bus = make_neck_core()
    assert target(core, 1.0) is None
    assert neck_writes(bus) == [], "a command only says what to hold; the tick writes it"
    core.tick(1.0)
    assert neck_writes(bus) == [block(REF.pan_ticks, REF.tilt_ticks)]
    target(core, 1.02, pan_rad=math.radians(10.0))  # pan +10 deg is left: pan_sign -1
    core.tick(1.02)
    assert neck_writes(bus)[-1] == ("Goal_Position", {"neck": REF.pan_ticks - 114, "head": 2311})
    target(core, 1.04, pan_rad=math.radians(10.0))
    core.tick(1.04)
    assert len(neck_writes(bus)) == 2, "the same goal again costs nothing"


def test_a_target_while_the_wheels_turn_is_accepted_and_never_delays_the_deadman() -> None:
    core, bus = make_neck_core()
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=1.0)
    assert target(core, 1.05, pan_rad=0.5) is None
    core.tick(1.05)
    assert energised(bus) and core.moving and not core.deadman
    core.tick(1.6)  # 0.6 s without a twist: the deadman fires on time, the head held on
    assert core.deadman and not core.moving
    assert bus.writes[-1] == ("Goal_Velocity", {LEFT: 0, RIGHT: 0})


def test_one_neck_write_a_tick_and_the_latest_target_wins() -> None:
    core, bus = make_neck_core()
    for i, pan in enumerate((0.1, 0.2, 0.3)):
        assert target(core, 1.0 + i * 0.001, pan_rad=pan) is None
    core.tick(1.003)
    (written,) = neck_writes(bus)
    pan_ticks, _ = ticks_for(NECK_CFG, NeckAngles(0.3, HOME_TILT_RAD))
    assert written == block(pan_ticks, REF.tilt_ticks)


def test_the_lease_lapses_home_at_full_pace_and_the_head_is_let_go_there() -> None:
    """A slow target (40 deg/s), renewed at 2.0; nothing after it for lease_s: the head goes home
    at the full pace — a new pace is a block — and the torque comes off on arrival."""
    core, bus = make_neck_core()
    assert target(core, 1.0, pan_rad=0.5, speed_deg_s=40.0) is None
    core.tick(1.0)
    pan_ticks, tilt_ticks = ticks_for(NECK_CFG, NeckAngles(0.5, HOME_TILT_RAD))
    assert neck_writes(bus) == [block(pan_ticks, tilt_ticks, speed=speed_ticks(40.0))]
    bus.positions.update({"neck": pan_ticks, "head": tilt_ticks})
    target(core, 2.0, pan_rad=0.5, speed_deg_s=40.0)  # renewed: the same goal, nothing written
    core.tick(3.9)
    assert len(neck_writes(bus)) == 1 and energised(bus), "held: the lease runs to 4.0"
    core.tick(4.02)
    assert neck_writes(bus)[-1] == block(REF.pan_ticks, REF.tilt_ticks)
    core.tick(4.04)
    assert energised(bus), "on its way home"
    bus.positions.update({"neck": REF.pan_ticks + 2, "head": REF.tilt_ticks - 3})
    core.tick(4.5)
    assert neck_writes(bus)[-1] == RELEASE, "home: let go"
    assert core.take_replies() == [], "nobody asked for the way home"


def test_a_target_s_speed_and_ramp_are_ceilings_under_the_motion_settings() -> None:
    core, bus = make_neck_core()
    target(core, 1.0, speed_deg_s=500.0, acc_deg_s2=50.0)
    core.tick(1.0)
    assert neck_writes(bus)[-1] == block(REF.pan_ticks, REF.tilt_ticks, acc=acc_units(50.0))
    target(core, 1.02, speed_deg_s=30.0, acc_deg_s2=50.0)
    core.tick(1.02)
    assert neck_writes(bus)[-1] == block(
        REF.pan_ticks, REF.tilt_ticks, speed=speed_ticks(30.0), acc=acc_units(50.0)
    ), "a new pace is a new block, the goal in it"


@pytest.mark.parametrize(
    ("message", "words"),
    [
        ({"pan_rad": 3.0, "tilt_rad": 0.4}, "outside its limits 257..3812"),  # 172 deg
        ({"pan_rad": 0.0, "tilt_rad": -1.0}, "outside its limits 1814..2760"),  # 57 deg up
        ({"pan_rad": 0.0}, "bad target: tilt_rad is required"),
        ({"pan_rad": "left", "tilt_rad": 0.4}, "bad target: pan_rad must be a number"),
        ({"pan_rad": True, "tilt_rad": 0.4}, "bad target: pan_rad must be a number"),
        ({"pan_rad": float("nan"), "tilt_rad": 0.4}, "bad target: pan_rad must be a number"),
        ({"pan_rad": 0.0, "tilt_rad": 0.4, "speed_deg_s": -5}, "speed_deg_s must be a positive"),
        ({"pan_rad": 0.0, "tilt_rad": 0.4, "acc_deg_s2": "max"}, "acc_deg_s2 must be a positive"),
    ],
)
def test_a_target_outside_the_reach_or_without_numbers_is_refused_not_clamped(
    message: dict[str, Any], words: str
) -> None:
    core, bus = make_neck_core()
    reply = core.command({"cmd": "neck_target", **message}, now=1.0)
    assert reply is not None and reply["type"] == "neck_target" and words in reply["error"]
    core.tick(1.0)
    assert neck_writes(bus) == []


def test_a_target_needs_the_reference_ticks_and_a_neck() -> None:
    unread = replace(NECK_CFG, reference=replace(REF, pan_ticks=None))
    core, _ = make_neck_core(unread)
    reply = target(core, 1.0)
    assert reply is not None and "reference ticks are unread" in reply["error"]
    bare, _ = make_core()
    assert target(bare, 1.0) == {
        "type": "neck_target",
        "error": "no neck configured on this base server",
    }


# -- who holds the head ----------------------------------------------------------------------


def test_a_target_preempts_a_move_and_a_move_waits_out_the_lease() -> None:
    core, bus = make_neck_core()
    assert core.command({"cmd": "neck_goto", "pan_ticks": 2500}, now=1.0) is None
    core.tick(1.0)
    assert target(core, 1.1) is None
    (preempted,) = core.take_replies()
    assert preempted["reached"] is False and preempted["error"] == "preempted by neck_target"
    refused = core.command({"cmd": "neck_goto", "pan_ticks": 2500}, now=1.2)
    assert refused == {
        "type": "neck_goto",
        "reached": False,
        "error": "the head is leased to neck_target for another 1.9 s",
    }
    core.tick(3.2)  # lapsed: on its way home, which nobody waits for
    assert core.command({"cmd": "neck_goto", "pan_ticks": 2500}, now=3.3) is None
    core.tick(3.3)
    assert goals(bus)["neck"] == 2500


def test_an_operator_jog_takes_the_head_from_a_lease_and_holds_it_against_targets() -> None:
    core, bus = make_neck_core()
    target(core, 1.0)
    core.tick(1.0)
    assert jog(core, 1.1, pan=1) is None
    assert target(core, 1.2) == {"type": "neck_target", "error": "an operator jog holds the head"}
    core.tick(1.9)  # the jog's deadman: let go where it is
    assert neck_writes(bus)[-1] == RELEASE
    assert target(core, 2.0) is None


# -- the motion settings, live -----------------------------------------------------------------


def test_the_motion_settings_are_answered_set_live_and_refused_outside_their_range() -> None:
    """ros/neck.sh motion: the settings in force beside the file's; a set lasts until the server
    restarts and reaches the head on the next tick; a bad value refuses the whole message."""
    core, bus = make_neck_core()
    motion = NECK_CFG.motion
    filed = {
        "max_speed_deg_s": motion.max_speed_deg_s,
        "max_acc_deg_s2": motion.max_acc_deg_s2,
        "lease_s": motion.lease_s,
    }
    reply = core.command({"cmd": "neck_motion"}, now=1.0)
    assert reply == {"type": "neck_motion", **filed, "config": filed}
    target(core, 1.0)
    core.tick(1.0)
    changed = core.command({"cmd": "neck_motion", "max_speed_deg_s": 60, "lease_s": 5.0}, now=1.1)
    assert changed is not None
    assert changed["was"] == {"max_speed_deg_s": motion.max_speed_deg_s, "lease_s": motion.lease_s}
    assert changed["max_speed_deg_s"] == 60.0
    assert changed["config"]["max_speed_deg_s"] == motion.max_speed_deg_s
    core.tick(1.1)
    assert neck_writes(bus)[-1] == block(REF.pan_ticks, REF.tilt_ticks, speed=speed_ticks(60.0))
    for bad in (
        {"max_speed_deg_s": 900},
        {"max_acc_deg_s2": 0},
        {"lease_s": "long"},
        {"lease_s": True},
        {"max_speed_deg_s": 100, "lease_s": 0.0},
    ):
        refused = core.command({"cmd": "neck_motion", **bad}, now=1.2)
        assert refused is not None and "refused" in refused["error"], bad
        assert (refused["max_speed_deg_s"], refused["lease_s"]) == (60.0, 5.0), "unchanged"
    bare, _ = make_core()
    assert "error" in (bare.command({"cmd": "neck_motion"}, now=1.0) or {})


# -- neck_goto / neck_home: one-shot moves ---------------------------------------------------


def test_a_move_writes_the_goal_then_answers_when_the_head_arrives() -> None:
    """The command itself answers nothing: the reply is born when the head stops, and leaves
    through take_replies (serve broadcasts it). One block energises and aims both servos at the
    full pace; the torque comes off on arrival."""
    core, bus = make_neck_core()
    assert (
        core.command({"cmd": "neck_goto", "pan_ticks": 2021, "tilt_ticks": 2311}, now=1.0) is None
    )
    core.tick(1.0)
    assert neck_writes(bus) == [block(2021, 2311)]
    core.tick(1.1)  # the servos are still on their way
    assert core.take_replies() == []
    bus.positions.update({"neck": 2021, "head": 2309})  # arrived, the tilt two ticks off
    core.tick(1.2)
    (reply,) = core.take_replies()
    assert reply["type"] == "neck_goto" and reply["reached"] is True
    assert (reply["pan_ticks"], reply["tilt_ticks"]) == (2021, 2309)
    assert reply["ms"] == pytest.approx(200.0) and reply["hold"] is False
    assert neck_writes(bus)[-1] == RELEASE, "the head is free to be pushed again"
    assert core.take_replies() == []  # taken once


def test_a_move_leaving_one_axis_alone_holds_it_where_it_is() -> None:
    core, bus = make_neck_core()
    assert core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=1.0) is None
    core.tick(1.0)
    assert neck_writes(bus) == [block(2021, 2360)], "the tilt held at its reading"


def test_a_move_target_outside_the_configured_limits_is_refused_not_clamped() -> None:
    core, bus = make_neck_core()
    reply = core.command({"cmd": "neck_goto", "pan_ticks": 4000}, now=1.0)
    assert reply is not None and reply["reached"] is False
    assert "257..3812" in reply["error"] and "4000" in reply["error"]
    low = core.command({"cmd": "neck_goto", "tilt_ticks": 1000}, now=1.1)
    assert low is not None and "1814..2760" in low["error"]
    assert core.command({"cmd": "neck_goto"}, now=1.2) == {
        "type": "neck_goto",
        "reached": False,
        "error": "neck_goto needs pan_ticks or tilt_ticks",
    }
    core.tick(1.2)
    assert neck_writes(bus) == [], "nothing was energised, nothing was written"


def test_a_head_that_never_arrives_gives_up_and_is_released() -> None:
    """27 ticks at 1365 ticks/s is a short move: it is given the 3 s floor."""
    core, bus = make_neck_core()
    core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=1.0)
    core.tick(2.0)
    assert core.take_replies() == [] and energised(bus)
    core.tick(4.1)  # 3 s after the start: the head never moved (a jammed or unpowered servo)
    (reply,) = core.take_replies()
    assert reply["reached"] is False and reply["pan_ticks"] == 2048
    assert reply["ms"] == pytest.approx(3100.0)
    assert neck_writes(bus)[-1] == RELEASE


def test_a_long_move_at_a_slow_pace_is_given_the_time_it_needs() -> None:
    """At 20 deg/s the 155 deg from 2048 to the pan limit take 7.75 s: the move is given half
    as long again plus a second, not the 3 s floor."""
    core, _ = make_neck_core()
    core.command({"cmd": "neck_motion", "max_speed_deg_s": 20.0}, now=0.5)
    core.command({"cmd": "neck_goto", "pan_ticks": 3812}, now=1.0)
    core.tick(1.0 + 7.75)
    assert core.take_replies() == [], "still on its way"
    core.tick(1.0 + 12.7)
    (reply,) = core.take_replies()
    assert reply["reached"] is False and reply["ms"] == pytest.approx(12700.0)


def test_hold_leaves_the_servos_energised_until_the_server_lets_go() -> None:
    core, bus = make_neck_core()
    core.command({"cmd": "neck_goto", "pan_ticks": 2048, "tilt_ticks": 2360, "hold": True}, now=1.0)
    core.tick(1.1)  # already there: reached on the first step
    (reply,) = core.take_replies()
    assert reply["reached"] is True and reply["hold"] is True
    core.tick(1.2)
    assert energised(bus), "held: the torque stays on"
    core.release()  # shutdown must not leave a head energised with nobody to release it
    assert neck_writes(bus)[-1] == RELEASE


def test_neck_home_goes_to_the_reference_ticks_of_the_config() -> None:
    core, bus = make_neck_core()
    assert core.command({"cmd": "neck_home"}, now=1.0) is None
    core.tick(1.0)
    assert goals(bus) == {"neck": REF.pan_ticks, "head": REF.tilt_ticks}


def test_home_without_reference_ticks_says_so_instead_of_moving() -> None:
    """The reference is a hardware reading that may be missing (null in the file); nothing is
    written then, because there is no pose to go to."""
    unread = replace(NECK_CFG, reference=replace(REF, pan_ticks=None))
    core, bus = make_neck_core(unread)
    reply = core.command({"cmd": "neck_home"}, now=1.0)
    assert reply is not None and "reference ticks are unread" in reply["error"]
    core.tick(1.0)
    assert neck_writes(bus) == []


def test_a_move_while_the_wheels_turn_goes_out_and_never_delays_the_deadman() -> None:
    """The old refusal existed because an acknowledged write to a silent servo held this thread
    0.4 s; every neck write is unacknowledged now, so the head turns while the cart drives."""
    core, bus = make_neck_core()
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=1.0)
    assert core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=1.05) is None
    core.tick(1.05)
    assert goals(bus)["neck"] == 2021 and core.moving
    assert core.command({"cmd": "neck_goto", "pan_ticks": 2100}, now=1.25) == {
        "type": "neck_goto",
        "reached": False,
        "error": "a neck move is already under way",
    }
    core.tick(1.6)  # 0.6 s without a twist: the deadman fires on time, mid-move
    assert core.deadman and not core.moving
    assert bus.writes[-1] == ("Goal_Velocity", {LEFT: 0, RIGHT: 0})


def test_a_move_command_without_a_neck_or_with_junk_in_it_answers_an_error() -> None:
    core, _ = make_core()
    reply = core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=1.0)
    assert reply is not None and reply["error"] == "no neck configured on this base server"
    assert core.command({"cmd": "neck_home"}, now=1.0) == reply
    movable, _ = make_neck_core()
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

    core, bus = make_neck_core()
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


def test_a_write_the_link_lost_is_rewritten_whole_on_the_next_tick() -> None:
    core, bus = make_neck_core()
    target(core, 1.0)
    core.tick(1.0)
    target(core, 1.02, pan_rad=0.2)
    bus.link_down = True
    core.tick(1.02)  # logged, never raised: the wheels' tick goes on
    bus.link_down = False
    core.tick(1.04)
    pan_ticks, _ = ticks_for(NECK_CFG, NeckAngles(0.2, HOME_TILT_RAD))
    assert neck_writes(bus)[-1] == block(pan_ticks, REF.tilt_ticks), "unknown state: the block"


def test_a_pair_that_falls_silent_mid_move_ends_it_with_the_error() -> None:
    core, bus = make_neck_core()
    core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=1.0)
    core.tick(1.0)
    bus.silent = {"neck", "head"}
    for now in (1.02, 1.04):
        core.tick(now)
        assert core.take_replies() == []
    core.tick(1.06)  # the third silent read: the pair is out
    (reply,) = core.take_replies()
    assert reply["reached"] is False and "fell silent" in reply["error"]
    again = core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=1.1)
    assert again is not None and "silent" in again["error"]


# -- the neck's jog (the game-mode teleop's held keys) -------------------------------------------

TICK_S = 0.02  # the board ticks at 50 Hz
JOG_FAST = round(jog_ticks_s(52.0))  # 592 ticks/s
JOG_SLOW = round(jog_ticks_s(8.0))  # 91


def jog(core: BaseServerCore, now: float, pan: int = 0, tilt: int = 0, slow: bool = False) -> Any:
    """One ``neck_jog`` message; the reply (None when accepted)."""
    return core.command({"cmd": "neck_jog", "pan": pan, "tilt": tilt, "slow": slow}, now=now)


def run_jog(
    core: BaseServerCore,
    bus: NeckBus,
    *,
    seconds: float,
    pan: int = 0,
    tilt: int = 0,
    slow: bool = False,
    start: float = 1.0,
    follow: bool = True,
) -> float:
    """Hold the keys for ``seconds``: a message and a tick every 20 ms, the servos following the
    written goals exactly when ``follow``; returns the clock at the end."""
    now = start
    for _ in range(round(seconds / TICK_S)):
        jog(core, now, pan, tilt, slow)
        core.tick(now)
        if follow:
            bus.positions.update(goals(bus))
        now += TICK_S
    return now


def test_a_jog_energises_the_head_and_walks_the_goal_at_the_fast_rate() -> None:
    """Pan +1 is left; config/neck.json's pan_sign -1 makes that a FALLING tick count. 52 deg/s
    for one second is 592 ticks: one block seeded where the head is, then one Goal_Position a
    tick."""
    core, bus = make_neck_core()
    assert jog(core, 1.0, pan=1) is None, "accepted: silent, like a twist"
    core.tick(1.0)
    assert neck_writes(bus) == [block(2048, 2360, speed=JOG_FAST)], "seeded: no jump"
    end = run_jog(core, bus, seconds=1.0, pan=1, start=1.0 + TICK_S)
    assert goals(bus)["neck"] == pytest.approx(2048 - 592, abs=2)
    assert goals(bus)["head"] == 2360, "the axis with direction 0 stays"
    assert core.take_replies() == [], "a jog answers nothing on the way"
    assert energised(bus), "still held while the messages keep coming"
    assert end == pytest.approx(2.0 + TICK_S)


def test_shift_makes_the_jog_slow_and_the_rate_changes_live() -> None:
    core, bus = make_neck_core()
    end = run_jog(core, bus, seconds=1.0, tilt=1, slow=True)  # tilt +1 is down: ticks rise
    assert goals(bus)["head"] == pytest.approx(2360 + 91, abs=2), "8 deg/s is 91 ticks/s"
    assert neck_writes(bus)[0][1]["head"][4] == JOG_SLOW
    run_jog(core, bus, seconds=1.0, pan=-1, slow=False, start=end)  # right: ticks rise
    assert goals(bus)["neck"] == pytest.approx(2048 + 592, abs=3)
    paces = [v["neck"][4] for name, v in neck_writes(bus) if name == "block"]
    assert paces == [JOG_SLOW, JOG_FAST], "the pace follows the key, one block each"
    assert goals(bus)["head"] == pytest.approx(2360 + 91, abs=2), "tilt released: frozen"


def test_the_goal_never_passes_the_configured_limits() -> None:
    """Tilting down for ten seconds would be 5920 ticks; the head stops at head.max — 2760 since
    2026-09-30, the camera's own stop is at 2783."""
    core, bus = make_neck_core()
    run_jog(core, bus, seconds=10.0, tilt=1)
    assert goals(bus)["head"] == NECK_CFG.tilt.max_ticks == 2760
    end = run_jog(core, bus, seconds=10.0, tilt=-1, start=11.0)
    assert goals(bus)["head"] == NECK_CFG.tilt.min_ticks == 1814
    run_jog(core, bus, seconds=10.0, pan=-1, start=end)  # right: ticks rise towards pan max
    assert goals(bus)["neck"] == NECK_CFG.pan.max_ticks == 3812


def test_releasing_the_key_stops_the_head_at_once_and_the_deadman_lets_go() -> None:
    """The client sends one zero jog when the key goes up: the goal freezes that tick, the
    servos stay energised, and half a second without a message releases them."""
    core, bus = make_neck_core()
    end = run_jog(core, bus, seconds=0.5, pan=1)
    frozen = goals(bus)["neck"]
    assert jog(core, end) is None  # both zero: stop where it is
    core.tick(end)
    writes = len(neck_writes(bus))
    core.tick(end + 0.3)
    assert goals(bus)["neck"] == frozen and len(neck_writes(bus)) == writes, "frozen, quiet"
    assert energised(bus), "held for the deadman's half second"
    core.tick(end + 0.6)
    assert neck_writes(bus)[-1] == RELEASE, "the deadman: torque off where it stands"
    core.tick(end + 1.0)
    assert neck_writes(bus).count(RELEASE) == 1, "released once, then idle"


def test_a_client_that_vanishes_mid_jog_is_cut_off_by_the_deadman() -> None:
    core, bus = make_neck_core()
    end = run_jog(core, bus, seconds=0.5, pan=1)
    advanced = goals(bus)["neck"]
    core.tick(end + 0.4)  # 0.4 s without a message: still within the deadman, goal walks on
    assert goals(bus)["neck"] < advanced and energised(bus)
    core.tick(end + 0.52)
    assert neck_writes(bus)[-1] == RELEASE
    stopped = goals(bus)["neck"]
    core.tick(end + 1.0)
    assert goals(bus)["neck"] == stopped


def test_a_zero_jog_with_nothing_under_way_does_nothing() -> None:
    core, bus = make_neck_core()
    assert jog(core, 1.0) is None
    core.tick(1.0)
    assert neck_writes(bus) == []


def test_a_jog_and_a_move_refuse_each_other() -> None:
    core, bus = make_neck_core()
    assert core.command({"cmd": "neck_goto", "pan_ticks": 2021}, now=1.0) is None
    assert jog(core, 1.05, pan=1) == {"type": "neck_jog", "error": "a neck move is under way"}
    bus.positions["neck"] = 2021
    core.tick(1.1)  # arrived: the move is over
    assert core.take_replies()[0]["reached"] is True
    assert jog(core, 1.2, pan=1) is None
    assert core.command({"cmd": "neck_goto", "pan_ticks": 2100}, now=1.25) == {
        "type": "neck_goto",
        "reached": False,
        "error": "a neck jog is under way",
    }
    assert core.command({"cmd": "neck_home"}, now=1.26) is not None


def test_a_jog_carries_on_while_the_wheels_turn_and_never_delays_their_deadman() -> None:
    """The game-mode teleop drives and looks at once now: the head's writes cannot block."""
    core, bus = make_neck_core()
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now=1.0)
    end = run_jog(core, bus, seconds=0.2, pan=1, start=1.0)
    assert core.moving and goals(bus)["neck"] < 2048, "jogged while driving"
    core.tick(end + 0.3)  # no twist since 1.0: the deadman
    assert core.deadman and not core.moving


def test_a_head_that_lags_its_goal_is_waited_for_not_wound_further_ahead() -> None:
    """A jammed or slow servo: the goal stops 120 ticks ahead of the encoder, and walks on as
    soon as the head catches up."""
    core, bus = make_neck_core()
    run_jog(core, bus, seconds=2.0, pan=-1, follow=False)  # the head never moves
    assert goals(bus)["neck"] <= 2048 + 120 + 10
    bus.positions["neck"] = goals(bus)["neck"]  # it catches up
    run_jog(core, bus, seconds=0.5, pan=-1, start=3.0)
    assert goals(bus)["neck"] > 2048 + 120 + 100


def test_bad_directions_and_a_missing_neck_answer_an_error() -> None:
    core, bus = make_neck_core()
    reply = core.command({"cmd": "neck_jog", "pan": 2}, now=1.0)
    assert reply is not None and reply["error"].startswith("bad direction:")
    assert core.command({"cmd": "neck_jog", "pan": True}, now=1.0) is not None
    assert core.command({"cmd": "neck_jog", "tilt": "up"}, now=1.0) is not None
    core.tick(1.0)
    assert neck_writes(bus) == []
    bare, _ = make_core()
    assert jog(bare, 1.0, pan=1) == {
        "type": "neck_jog",
        "error": "no neck configured on this base server",
    }


def test_a_jog_is_refused_while_the_encoders_are_silent() -> None:
    core, bus = make_neck_core()
    bus.silent = {"neck"}
    for now in (0.02, 0.04, 0.06):
        core.tick(now)
    refused = jog(core, 0.1, pan=1)
    assert refused is not None and refused["error"].startswith("the neck servos are silent")


def test_a_link_lost_mid_jog_costs_one_tick_and_the_jog_goes_on() -> None:
    core, bus = make_neck_core()
    end = run_jog(core, bus, seconds=0.2, pan=1)
    bus.link_down = True
    jog(core, end, pan=1)
    core.tick(end)  # the write is lost, logged; nothing is raised into the wheels' tick
    bus.link_down = False
    jog(core, end + TICK_S, pan=1)
    core.tick(end + TICK_S)
    assert core._mover is not None and core._mover.jogging
    assert neck_writes(bus)[-1][0] == "block", "the servos' state was unknown: rewritten whole"


def test_shutdown_releases_a_jogging_head() -> None:
    core, bus = make_neck_core()
    run_jog(core, bus, seconds=0.2, tilt=1)
    core.release()
    assert neck_writes(bus)[-1] == RELEASE


def test_encoders_falling_silent_mid_jog_end_it_with_an_error_reply() -> None:
    """The pair stops answering: the goal is not walked on against stale ticks, and once the
    pair is out of the read the jog ends with an error line for the teleop."""
    core, bus = make_neck_core()
    end = run_jog(core, bus, seconds=0.2, pan=1)
    bus.silent = {"neck", "head"}
    now = end
    for _ in range(3):
        jog(core, now, pan=1)
        core.tick(now)
        now += TICK_S
    assert core._mover is not None and not core._mover.jogging
    (reply,) = core.take_replies()
    assert reply["type"] == "neck_jog" and reply["error"].startswith("the encoders failed")


def test_a_stalled_tick_advances_the_goal_by_at_most_two_ticks_worth() -> None:
    """A 0.5 s stall of the tick thread is not one 296-tick stride for the head."""
    core, bus = make_neck_core()
    jog(core, 1.0, pan=-1)  # right: ticks rise
    core.tick(1.0)
    jog(core, 1.5, pan=-1)  # the message keeps the deadman quiet; the tick thread stalled
    core.tick(1.5)
    stride = goals(bus)["neck"] - 2048
    assert 10 < stride <= round(2 * TICK_S * 592) + 1  # 24 ticks, two ticks' worth


def test_both_axes_jogging_cost_one_write_per_tick() -> None:
    core, bus = make_neck_core()
    assert jog(core, 1.0, pan=1, tilt=-1) is None  # left and up: room on both axes for 0.7 s
    core.tick(1.0)
    assert neck_writes(bus) == [block(2048, 2360, speed=JOG_FAST)], "one seed write"
    before = len(bus.writes)
    run_jog(core, bus, seconds=0.7, pan=1, tilt=-1, start=1.0 + TICK_S)
    added = bus.writes[before:]
    assert len(added) == 35, "35 ticks, 35 transactions"
    assert all(name == "Goal_Position" and set(v) == {"neck", "head"} for name, v in added)


class WarmBus(NeckBus):
    """A bus whose servos answer Present_Temperature (register 63), the neck's when asked."""

    def __init__(self) -> None:
        super().__init__()
        self.temperatures = {LEFT: 31, RIGHT: 33, "neck": 40, "head": 41}

    def sync_read(
        self,
        data_name: str,
        motors: list[str],
        *,
        normalize: bool = True,
        optional: Sequence[str] = (),
        optional_window_s: float = 0.0,
    ) -> dict[str, int]:
        if data_name == "Present_Temperature":
            self.reads.append((data_name, list(motors), tuple(optional), optional_window_s))
            return {m: self.temperatures[m] for m in [*motors, *optional]}
        return super().sync_read(
            data_name,
            motors,
            normalize=normalize,
            optional=optional,
            optional_window_s=optional_window_s,
        )


def test_the_state_line_carries_the_wheels_temperature_read_every_five_seconds() -> None:
    """The servos cut out at 70 C: the state line says how warm they are, read at most every
    5 s so the wheels' loop pays for it once in 250 ticks; a bus that does not answer the read
    leaves the field empty and nothing else."""
    bus = WarmBus()
    core = BaseServerCore(bus, CFG, servo_names=[LEFT, RIGHT, "servo1"])
    core.tick(0.0)
    assert core.snapshot(0.0)["temp_c"] == {"left": 31, "right": 33}
    bus.temperatures = {LEFT: 50, RIGHT: 52}
    assert core.snapshot(1.0)["temp_c"] == {"left": 31, "right": 33}, "not re-read within 5 s"
    assert core.snapshot(6.0)["temp_c"] == {"left": 50, "right": 52}
    silent, _ = make_core()  # FakeBus answers positions only
    assert silent.snapshot(0.0)["temp_c"] is None


def test_the_neck_s_temperature_rides_along_while_it_answers() -> None:
    """A lease holds the head's torque for as long as the arbiter wants: its servos' heat is in
    the same line, read in the same packet as the wheels', the neck as optional ids."""
    core, bus = make_neck_core(bus=WarmBus())
    assert core.snapshot(1.0)["temp_c"] == {"left": 31, "right": 33, "neck": 40, "head": 41}
    assert bus.reads[-1] == ("Present_Temperature", [LEFT, RIGHT], ("neck", "head"), 0.003)


# -- on the wire: the real client against a fake ser2net (tests/unit/servo_wire.py) -------------

MOTORS = {LEFT: 7, RIGHT: 8, "neck": 9, "head": 10}


def wired(silent: set[int] | None = None, latency_s: float = 0.002) -> tuple[ServoWire, Any]:
    """A fake ser2net with the four servos (the neck at 2048, 2048) and a connected client."""
    wire = ServoWire(list(MOTORS.values()), reply_latency_s=latency_s)
    for motor_id in (9, 10):
        wire.set_word(motor_id, 56, 2048)
    wire.silent = set(silent or ())
    bus = FeetechTcpClient("127.0.0.1", wire.start(), MOTORS, retries=1)
    bus.connect()
    return wire, bus


@pytest.mark.slow
def test_on_the_wire_a_tick_is_one_round_trip_and_at_most_one_neck_write() -> None:
    """Driving and aiming the head on every tick: one sync_read a tick with all four servos in
    it, and the twist and the neck as two unacknowledged writes; the servos end up holding what
    the core asked, torque, ramp, goal and speed — the ramp too, because the neck's acceleration
    ceiling was lifted first (the fake stores a larger Acceleration as the ceiling, as the
    STS3215 does), and only the neck's."""
    wire, bus = wired()
    try:
        encoders, mover = neck_parts(bus, NECK_CFG)
        core = BaseServerCore(bus, CFG, neck=encoders, mover=mover)
        core.tick(0.0)  # primes the wheels, hears the neck, reads its mode, lets it go
        core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, 0.5)  # arming: two acked writes
        core.tick(0.5)  # a read answered: the fake has applied every packet before it
        counts = wire.counts
        reads, writes, acked = counts.sync_reads, counts.sync_writes, counts.writes
        for i in range(20):
            now = 1.0 + i * 0.02
            core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, now)
            target(core, now, pan_rad=0.01 * i)
            core.tick(now)
        line = core.snapshot(1.4)
        core.tick(1.42)  # a read after the last write: the fake has applied everything before it
        assert counts.sync_reads - reads == 21 + 1, "one a tick, and the state line's temperature"
        assert counts.sync_writes - writes == 20 + 20, "a twist and a neck write a tick"
        assert counts.writes == acked, "no acknowledged write after arming"
        pan, tilt = ticks_for(NECK_CFG, NeckAngles(0.19, HOME_TILT_RAD))
        assert [wire.byte(9, 40), wire.byte(9, 41), wire.word(9, 42), wire.word(9, 44)] == [
            1,
            TOP_ACC,
            pan,
            0,
        ]
        assert wire.word(9, 46) == TOP_SPEED and wire.word(10, 42) == tilt
        assert wire.byte(10, 41) == TOP_ACC
        ceilings = [wire.byte(i, 85) for i in MOTORS.values()]
        assert ceilings == [50, 50, NECK_ACC_CEILING, NECK_ACC_CEILING]
        assert (line["pan_ticks"], line["tilt_ticks"]) == (2048, 2048)
    finally:
        bus.close()
        wire.stop()


@pytest.mark.slow
def test_on_the_wire_the_registers_command_reads_a_servo_s_table() -> None:
    """ros/neck.sh registers: raw bytes of one roster servo's table through the server that owns
    the bus (what found the neck's acceleration ceiling, 2026-10-02); refused while the wheels
    turn, and a servo off the roster or a read past the limit answers an error."""
    wire, bus = wired()
    try:
        encoders, mover = neck_parts(bus, NECK_CFG)
        core = BaseServerCore(bus, CFG, servo_names=list(MOTORS), neck=encoders, mover=mover)
        core.tick(0.0)

        def ask(now: float = 0.1, **message: Any) -> Any:
            return core.command({"cmd": "registers", **message}, now)

        assert ask(servo="neck", address=84, size=2) == {
            "type": "registers",
            "servo": "neck",
            "address": 84,
            "values": [0, NECK_ACC_CEILING],
        }
        assert ask(servo=LEFT, address=85)["values"] == [50]
        assert "ValueError" in ask(servo="neck", address=0, size=65)["error"]
        assert "roster" in ask(servo="tail", address=0)["error"]
        core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, 0.2)
        assert ask(0.2, servo="neck", address=85) == {
            "type": "registers",
            "servo": "neck",
            "busy": True,
        }
    finally:
        bus.close()
        wire.stop()


@pytest.mark.slow
def test_on_the_wire_a_dead_neck_costs_the_odometry_its_window_and_never_a_retry() -> None:
    """Both neck servos unpowered: the wheels' read still answers in one packet, no retry, at
    most the 3 ms window later than a read without the neck; three ticks later the neck is out
    of the read altogether."""
    wire, bus = wired(silent={9, 10})
    try:
        import time

        def timed(**riders: Any) -> float:
            started = time.perf_counter()
            assert set(
                bus.sync_read("Present_Position", [LEFT, RIGHT], normalize=False, **riders)
            ) == {
                LEFT,
                RIGHT,
            }
            return time.perf_counter() - started

        alone = sorted(timed() for _ in range(9))[4]
        reads = wire.counts.sync_reads
        riding = sorted(
            timed(optional=["neck", "head"], optional_window_s=0.003) for _ in range(9)
        )[4]
        assert wire.counts.sync_reads - reads == 9, "never a retry for a silent optional id"
        assert riding - alone < 0.003 + 0.010
        encoders, mover = neck_parts(bus, NECK_CFG)
        core = BaseServerCore(bus, CFG, neck=encoders, mover=mover)
        for i in range(5):
            core.tick(i * 0.02)
        assert encoders.riders(0.1) == [] and not encoders.live
        assert "pan_ticks" not in core.snapshot(0.1)
    finally:
        bus.close()
        wire.stop()
