"""The arm on the base server's tick, against a fake bus: it rides the wheels' read, reads its
mode and range once, energises where it stands, writes one goal a tick, refuses (never clamps),
rests and lets go by itself, yields to the wheels, and halts on silence."""

import itertools
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from servo_wire import ServoWire
from test_base import CFG, FakeBus

from pepin.arm_driver import ARM_BLOCK, interpolate, trajectory_from, trajectory_line
from pepin.arm_motion import ServoRange, load_arm_motion
from pepin.base import LEFT, RIGHT
from pepin.base_server import (
    READ_BUDGET_BYTES,
    BaseServerCore,
    NeckEncoders,
    arm_part,
    neck_parts,
)
from pepin.expression.animation import Clip, Track
from pepin.expression.fakes import FakeArmLink
from pepin.expression.parts import ArmPart
from pepin.expression.trajectory import JOINTS, Knot, Trajectory
from pepin.feetech import REGISTERS, FeetechTcpClient
from pepin.neck import NeckConfig, acc_units, speed_ticks

REPO = Path(__file__).resolve().parents[2]
MOTION = load_arm_motion(REPO / "config/arm.json")
NAMES = tuple(f"servo{i}" for i in range(1, 7))
# lerobot's calibration of 2026-08-30, as the servos' EEPROM holds it (min, max ticks)
CAL = {"servo1": (1515, 3492), "servo2": (1318, 3663), "servo3": (723, 2928),
       "servo4": (280, 2575), "servo5": (0, 4095), "servo6": (1837, 3126)}  # fmt: skip
RANGES = tuple(
    ServoRange(*CAL[n], gripper=j == "gripper") for n, j in zip(NAMES, JOINTS, strict=True)
)
REST_TICKS = tuple(r.ticks(MOTION.rest_deg[j]) for r, j in zip(RANGES, JOINTS, strict=True))
MOTORS = {LEFT: 7, RIGHT: 8, **{n: i for i, n in enumerate(NAMES, start=1)}}
NECK = ("neck", "head")
HEADER_BYTES = 6  # FF FF id length error ... checksum around a reply's register bytes
TICK = 0.02
WRIST = JOINTS.index("wrist_flex")


class ArmBus(FakeBus):
    """FakeBus with the arm on it: positions in ticks, modes, the EEPROM range, silent servos;
    the servos follow every goal at once (``follow``)."""

    def __init__(self) -> None:
        super().__init__()
        self.positions.update(dict(zip(NAMES, REST_TICKS, strict=True)))
        self.positions.update(dict.fromkeys(NECK, 2048))
        self.modes = dict.fromkeys(NAMES, 0)
        self.silent: set[str] = set()
        self.reads: list[tuple[str, tuple[str, ...]]] = []
        self.replies: list[int] = []  # the reply bytes each sync_read asked for
        self.fail_next = False  # the next position read times out (the wheels' read failed)
        self.follow = True

    def ping(self, motor: str, num_retry: int = 0) -> int | None:
        return 0

    def sync_read(
        self,
        data_name: str,
        motors: list[str],
        *,
        normalize: bool = True,
        optional: Sequence[str] = (),
        optional_window_s: float = 0.0,
    ) -> dict[str, int]:
        self.reads.append((data_name, tuple(optional)))
        size = REGISTERS[data_name].size
        self.replies.append((len(motors) + len(optional)) * (HEADER_BYTES + size))
        if data_name == "Present_Position" and self.fail_next:
            self.fail_next = False
            raise TimeoutError("no reply from ids [7, 8] after 2 attempts")
        here = [m for m in optional if m not in self.silent]
        values: dict[str, dict[str, int]] = {
            "Operating_Mode": self.modes,
            "Min_Position_Limit": {n: CAL[n][0] for n in NAMES},
            "Max_Position_Limit": {n: CAL[n][1] for n in NAMES},
            "Present_Temperature": dict.fromkeys([*NAMES, *NECK, LEFT, RIGHT], 30),
            "Present_Position": self.positions,
        }
        table = values[data_name]
        return {m: table[m] for m in [*motors, *here]}

    def sync_write(self, data_name: str, values: dict[str, int], *, normalize: bool = True) -> None:
        super().sync_write(data_name, values, normalize=normalize)
        if data_name == "Goal_Position" and self.follow:
            self.positions.update(values)

    def sync_write_block(self, data_names: Sequence[str], values: dict[str, Sequence[int]]) -> None:
        assert tuple(data_names) == ARM_BLOCK
        super().sync_write_block(data_names, values)
        if self.follow:
            self.positions.update({m: row[2] for m, row in values.items()})


def make_core(bus: ArmBus | None = None) -> tuple[BaseServerCore, ArmBus]:
    """A core with the arm, after its first tick (which heard the arm and read its mode and
    range), the bus's log cleared."""
    bus = bus if bus is not None else ArmBus()
    driver = arm_part(bus, MOTION, MOTORS)
    assert driver is not None
    core = BaseServerCore(bus, CFG, servo_names=list(MOTORS), arm=driver)
    core.tick(0.0)
    bus.writes.clear()
    bus.reads.clear()
    return core, bus


def make_board_core(bus: ArmBus | None = None) -> tuple[BaseServerCore, ArmBus]:
    """The board's read: the wheels, the neck and the arm (in two parts), after the first ride
    (two ticks), the bus's log cleared."""
    bus = bus if bus is not None else ArmBus()
    driver = arm_part(bus, MOTION, MOTORS)
    assert driver is not None
    core = BaseServerCore(bus, CFG, servo_names=list(MOTORS), neck=NeckEncoders(), arm=driver)
    core.tick(0.0)
    core.tick(TICK)
    bus.writes.clear()
    bus.reads.clear()
    bus.replies.clear()
    return core, bus


def position_reads(bus: ArmBus) -> list[set[str]]:
    """The arm's servos in each tick's position read, in order."""
    return [set(o) & set(NAMES) for name, o in bus.reads if name == "Present_Position"]


def arm_writes(bus: ArmBus) -> list[tuple[str, dict[str, Any]]]:
    return [(name, v) for name, v in bus.writes if "servo1" in v]


def wave(amp: float = 8.0, seconds: float = 2.0, speed: float = 1.0, tid: str = "t1") -> Trajectory:
    """A wrist wave from rest, planned by the laptop's own part."""
    track = Track("wrist_flex", "additive", (0.0, 0.3 * seconds, 0.7 * seconds, seconds),
                  (0.0, amp, -amp, 0.0))  # fmt: skip
    part = ArmPart(FakeArmLink(), MOTION, speed_scale=speed)
    planned = part.plan(Clip("arm", 0.0, 0.0, seconds, (track,)), "wave")
    assert isinstance(planned, Trajectory), planned
    return Trajectory(tid, planned.source, planned.knots, planned.then, planned.hold_s, "wave",
                      planned.speed_scale)  # fmt: skip


def send(core: BaseServerCore, trajectory: Trajectory, now: float) -> Any:
    return core.command(dict(trajectory_line(trajectory)), now)


def run(core: BaseServerCore, start: float, seconds: float) -> float:
    t = start
    for _ in range(round(seconds / TICK)):
        t += TICK
        core.tick(t)
    return t


# -- reading ------------------------------------------------------------------------------------


def test_at_rest_the_arm_rides_the_wheels_read_a_few_times_a_second() -> None:
    core, bus = make_core()
    lines = []
    t = 0.0
    for _ in range(50):  # one second of ticks
        t += TICK
        core.tick(t)
        lines.append(core.snapshot(t))
    rides = [optional for name, optional in bus.reads if name == "Present_Position" and optional]
    assert 4 <= len(rides) <= 6 and all(set(r) == set(NAMES) for r in rides)
    with_arm = [line["arm"] for line in lines if "arm" in line]
    assert len(with_arm) == len(rides)
    assert with_arm[0]["deg"] == pytest.approx([MOTION.rest_deg[j] for j in JOINTS], abs=0.1)
    assert with_arm[0]["phase"] == "off" and with_arm[0]["torque"] is False
    assert not arm_writes(bus)


def test_beside_the_neck_no_read_outgrows_the_adapter_s_usb_packet() -> None:
    """Wheels, neck and arm in one read are 80 bytes of replies, and the adapter's 64-byte USB
    packet held the last 16 (servos 5 and 6) back until the next request (2026-10-12): the arm
    rides in two parts, at rest and energised, and the temperature read takes one part a time,
    the other's degrees kept."""
    core, bus = make_board_core()
    t = run(core, TICK, 0.0)
    lines = []
    for _ in range(300):  # six seconds at rest: rides and two temperature reads
        t += TICK
        core.tick(t)
        lines.append(core.snapshot(t))
    assert send(core, wave(), t)["status"] == "accepted"
    for _ in range(100):  # two seconds energised: rides back to back
        t += TICK
        core.tick(t)
        lines.append(core.snapshot(t))
    assert max(bus.replies) <= READ_BUDGET_BYTES
    rides = [r for r in position_reads(bus) if r]
    assert {"servo5", "servo6"} in rides and {"servo1", "servo2", "servo3", "servo4"} in rides
    assert not any("servo1" in r and "servo5" in r for r in rides)
    assert set(lines[-1]["temp_c"]) == {LEFT, RIGHT, *NECK, *NAMES}


def test_the_line_carries_the_arm_only_when_both_parts_of_one_ride_answered() -> None:
    """Joints 5-6 are read a tick before 1-4: a line carries the six on the tick that read 1-4
    (its stamp is that read; 5-6 are one tick older, and the reading's age counts from them),
    at 5 Hz at rest and every other tick energised. A part that missed or a failed read
    between the two costs that ride's line, never a reading mixed across rides."""
    core, bus = make_board_core()
    t, lines = TICK, []
    for _ in range(50):  # one second at rest
        t += TICK
        core.tick(t)
        lines.append("arm" in core.snapshot(t))
    reads = position_reads(bus)
    closing = [i for i, r in enumerate(reads) if "servo1" in r]
    assert 4 <= len(closing) <= 5 and lines.count(True) == len(closing)
    assert all(lines[i] and reads[i - 1] == {"servo5", "servo6"} for i in closing)
    assert core.command({"cmd": "arm_read"}, t)["age_s"] <= 0.2

    def ride(between: Callable[[], object]) -> dict[str, Any]:
        """Ticks until a ride's first part is read, ``between``, its second part: the line."""
        nonlocal t
        while True:
            t += TICK
            before = len(bus.reads)
            core.tick(t)
            if any("servo5" in o for _, o in bus.reads[before:]):
                break
        between()
        t += TICK
        core.tick(t)
        return core.snapshot(t)

    def by_hand() -> None:  # the gripper and the pan moved between the two parts
        bus.positions["servo6"] += 40
        bus.positions["servo1"] += 40

    deg = ride(by_hand)["arm"]["deg"]
    gripper = RANGES[5].degrees(bus.positions["servo6"] - 40)
    assert deg[5] == pytest.approx(gripper, abs=0.01), "5-6 as read on the tick before"
    assert deg[0] == pytest.approx(RANGES[0].degrees(bus.positions["servo1"]), abs=0.01)
    assert core.command({"cmd": "arm_read"}, t)["age_s"] == pytest.approx(TICK)
    bus.silent = {"servo6"}
    assert "arm" not in ride(bus.silent.clear)  # the first part missed servo6
    assert "arm" not in ride(lambda: setattr(bus, "fail_next", True))  # the second's read failed
    assert "arm" in ride(lambda: None)
    assert send(core, wave(seconds=4.0), t)["status"] == "accepted"
    carried = []
    for _ in range(40):
        t += TICK
        core.tick(t)
        carried.append("arm" in core.snapshot(t))
    assert carried.count(True) >= 19  # every other tick: 25 Hz
    assert not any(a and b for a, b in itertools.pairwise(carried))


def test_its_mode_and_range_are_read_once_and_arm_read_answers_without_the_bus() -> None:
    bus = ArmBus()
    core, _ = make_core(bus)
    assert bus.reads == []  # cleared after the first tick, which read them
    fresh = ArmBus()
    driver = arm_part(fresh, MOTION, MOTORS)
    assert driver is not None
    BaseServerCore(fresh, CFG, arm=driver).tick(0.0)
    names = [name for name, _ in fresh.reads]
    assert names.count("Operating_Mode") == 1 and names.count("Min_Position_Limit") == 1
    reply = core.command({"cmd": "arm_read"}, 0.1)
    assert reply["type"] == "arm_read" and reply["ticks"] == list(REST_TICKS)
    assert reply["deg"] == pytest.approx([MOTION.rest_deg[j] for j in JOINTS], abs=0.1)
    assert reply["range_deg"][0] == pytest.approx([-86.9, 86.9], abs=0.05)
    assert bus.reads == []


# -- moving -------------------------------------------------------------------------------------


def test_a_trajectory_energises_where_the_arm_stands_then_one_goal_a_tick() -> None:
    core, bus = make_core()
    reply = send(core, wave(), 0.12)  # serve() hands a command and its tick the same time
    assert reply == {"type": "arm", "id": "t1", "status": "accepted", "reason": ""}
    core.tick(0.12)
    ((name, block),) = arm_writes(bus)
    assert name == "block"
    torque, acc, _goal, goal_time, speed, limit = block["servo4"]
    assert (torque, goal_time, limit) == (1, 0, 500)
    assert acc == acc_units(MOTION.max_acc_deg_s2) and speed == speed_ticks(MOTION.max_speed_deg_s)
    assert {m: row[2] for m, row in block.items()} == dict(zip(NAMES, REST_TICKS, strict=True))
    bus.writes.clear()
    t = run(core, 0.12, 2.1)
    writes = arm_writes(bus)
    assert writes and all(name == "Goal_Position" for name, _ in writes)
    assert len(writes) <= round(2.1 / TICK)  # at most one a tick
    peak = max(values["servo4"] for _, values in writes)
    assert RANGES[WRIST].degrees(peak) == pytest.approx(MOTION.rest_deg["wrist_flex"] + 8, abs=0.3)
    answers = core.take_replies()
    assert {"type": "arm", "id": "t1", "status": "done", "reason": ""} in answers
    t = run(core, t, 0.5 + MOTION.release_after_s)
    assert arm_writes(bus)[-1] == ("Torque_Enable", dict.fromkeys(NAMES, 0))
    assert core.snapshot(t).get("arm", {"phase": "off"})["phase"] == "off"


def test_the_speed_scale_slows_the_profile_and_the_rest_after_it() -> None:
    core, bus = make_core()
    assert send(core, wave(seconds=10.0, speed=0.2), 0.12)["status"] == "accepted"
    core.tick(0.12)
    ((_, block),) = arm_writes(bus)
    assert block["servo1"][4] == speed_ticks(MOTION.max_speed_deg_s * 0.2)
    assert block["servo1"][1] == acc_units(MOTION.max_acc_deg_s2 * 0.2**2)


def test_away_from_the_first_knot_the_arm_joins_it_within_the_caps() -> None:
    bus = ArmBus()
    bus.positions["servo4"] = RANGES[WRIST].ticks(40.0)  # moved by hand, torque off
    core, _ = make_core(bus)
    run(core, 0.0, 0.3)  # a reading after the move (it rides 5 times a second at rest)
    bus.writes.clear()
    assert send(core, wave(), 0.42)["status"] == "accepted"
    core.tick(0.42)
    ((_, block),) = arm_writes(bus)
    assert block["servo4"][2] == RANGES[WRIST].ticks(40.0)  # energised where it is
    goals = [40.0]
    t = 0.42
    for _ in range(40):
        t += TICK
        core.tick(t)
        goals += [RANGES[WRIST].degrees(v["servo4"]) for n, v in arm_writes(bus)[-1:]
                  if n == "Goal_Position"]  # fmt: skip
    steps = [abs(b - a) / TICK for a, b in itertools.pairwise(goals)]
    assert (
        max(steps) <= MOTION.max_speed_deg_s * 1.05 and goals[-1] > 60.0
    )  # on its way back to 76.6


@pytest.mark.parametrize(
    ("change", "why"),
    [
        (lambda t: Trajectory(t.id, t.source, (Knot(0.0, (0.0,) * 5),), "rest"), "malformed"),
        (lambda t: Trajectory(t.id, t.source, (*t.knots, Knot(t.knots[-1].t_s + 0.04,
                                                             (0, -110, 0, 0, 0, 0))), "rest"),
         "outside"),
        (lambda t: Trajectory(t.id, t.source, (*t.knots, Knot(t.knots[-1].t_s + 0.01,
                                                             (-34.3, -102.6, 96.8, 81.6, -54.1,
                                                              -0.4))), "rest"),
         "deg/s"),
        (lambda t: Trajectory(t.id, t.source, (t.knots[0], Knot(40.0, t.knots[0].deg)), "rest"),
         "max_duration_s"),
        (lambda t: Trajectory(t.id, t.source, t.knots, "rest", speed_scale=1.5), "speed"),
    ],
)  # fmt: skip
def test_a_bad_trajectory_is_refused_with_the_reason_never_clamped(change: Any, why: str) -> None:
    core, bus = make_core()
    reply = send(core, change(wave()), 0.1)
    assert reply["status"] == "refused" and why in reply["reason"]
    run(core, 0.1, 0.2)
    assert not arm_writes(bus)


def test_nothing_starts_while_the_wheels_are_commanded_to_move() -> None:
    core, _bus = make_core()
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, 0.05)
    reply = send(core, wave(), 0.1)
    assert reply["status"] == "refused" and "wheels" in reply["reason"]


def test_a_twist_sends_a_moving_arm_to_rest() -> None:
    core, _bus = make_core()
    send(core, wave(seconds=4.0), 0.1)
    t = run(core, 0.1, 0.6)
    core.command({"cmd": "twist", "v": 0.1, "w": 0.0}, t)
    core.tick(t + TICK)
    halted = [r for r in core.take_replies() if r.get("type") == "arm"]
    assert halted == [{"type": "arm", "id": "t1", "status": "halted",
                       "reason": "the wheels moved: to rest"}]  # fmt: skip
    assert core.snapshot(t + TICK).get("arm", {}).get("phase", "resting") == "resting"


def test_a_newer_trajectory_takes_the_arm_from_the_one_playing() -> None:
    core, _ = make_core()
    send(core, wave(tid="first", seconds=4.0), 0.1)
    t = run(core, 0.1, 0.5)
    assert send(core, wave(tid="second"), t)["status"] == "accepted"
    assert {"type": "arm", "id": "first", "status": "preempted",
            "reason": "preempted by second"} in core.take_replies()  # fmt: skip


def test_a_halt_holds_where_the_goal_is_then_rests_by_itself() -> None:
    core, bus = make_core()
    send(core, wave(seconds=4.0), 0.1)
    t = run(core, 0.1, 1.0)
    assert core.command({"cmd": "arm_halt"}, t) == {"type": "arm_halt", "phase": "halted"}
    bus.writes.clear()
    t = run(core, t, MOTION.hold_max_s - 0.1)
    assert not arm_writes(bus)  # held: the goal does not change
    t = run(core, t, 0.2)
    assert arm_writes(bus)  # on its way to rest
    assert core.command({"cmd": "arm_rest"}, t)["phase"] == "resting"


def test_rest_and_halt_at_rest_do_nothing_and_say_so() -> None:
    core, bus = make_core()
    assert core.command({"cmd": "arm_rest"}, 0.1) == {"type": "arm_rest", "phase": "off"}
    assert core.command({"cmd": "arm_halt"}, 0.1) == {"type": "arm_halt", "phase": "off"}
    run(core, 0.1, 0.2)
    assert not arm_writes(bus)


def test_servos_falling_silent_end_the_motion_and_are_never_written() -> None:
    core, bus = make_core()
    send(core, wave(seconds=4.0), 0.1)
    t = run(core, 0.1, 0.3)
    bus.silent = {"servo3"}
    t = run(core, t, 3 * TICK)  # three reads without it take the group out (the neck's rule)
    bus.writes.clear()
    t = run(core, t, 4 * TICK)
    assert {k: v for k, v in core.take_replies()[-1].items() if k != "reason"} == {
        "type": "arm",
        "id": "t1",
        "status": "halted",
    }
    assert not arm_writes(bus)
    refused = send(core, wave(tid="again"), t)
    assert refused["status"] == "refused" and "silent" in refused["reason"]
    bus.silent = set()
    t = run(core, t, 6.0)  # asked again after retry_s, its mode and range read again
    assert send(core, wave(tid="back"), t)["status"] == "accepted"


def test_a_servo_out_of_position_mode_refuses_every_motion() -> None:
    bus = ArmBus()
    bus.modes["servo3"] = 1
    core, _ = make_core(bus)
    reply = send(core, wave(), 0.1)
    assert reply["status"] == "refused" and "position mode" in reply["reason"]
    assert "position mode" in core.command({"cmd": "arm_read"}, 0.1)["error"]


def test_shutdown_lets_the_arm_go() -> None:
    core, bus = make_core()
    send(core, wave(), 0.1)
    run(core, 0.1, 0.2)
    core.release()
    assert arm_writes(bus)[-1] == ("Torque_Enable", dict.fromkeys(NAMES, 0))


def test_without_an_arm_the_commands_answer_an_error() -> None:
    core = BaseServerCore(ArmBus(), CFG)
    none = "no arm on this base server"
    assert core.command({"cmd": "arm_read"}, 0.0) == {"type": "arm_read", "error": none}
    assert send(core, wave(), 0.0)["reason"] == "no arm on this base server"


def test_the_line_round_trips_and_interpolation_is_linear_between_knots() -> None:
    trajectory = wave()
    assert trajectory_from(dict(trajectory_line(trajectory))) == trajectory
    knots = [(0.0, (0.0, 10.0)), (1.0, (10.0, 10.0))]
    assert interpolate(knots, 0.25) == (2.5, 10.0) and interpolate(knots, 5.0) == (10.0, 10.0)
    with pytest.raises(ValueError, match="knots"):
        trajectory_from({"cmd": "arm_trajectory", "knots": [[0.0, True, 0, 0, 0, 0, 0]]})


@pytest.mark.slow
def test_on_the_wire_the_arm_rides_the_one_read_and_its_block_lands_in_the_servos() -> None:
    """With the neck and the arm on the bus, a tick is still one sync_read; the energising
    block puts torque, goal, speed and the torque limit into the arm's control tables."""
    motors = {
        LEFT: 7,
        RIGHT: 8,
        "neck": 9,
        "head": 10,
        **dict(zip(NAMES, range(1, 7), strict=True)),
    }
    wire = ServoWire(list(motors.values()), reply_latency_s=0.002)
    for name, motor_id in motors.items():
        wire.set_word(motor_id, 56, 2048)
        if name in CAL:
            wire.set_word(motor_id, 9, CAL[name][0])
            wire.set_word(motor_id, 11, CAL[name][1])
            wire.set_word(motor_id, 56, REST_TICKS[NAMES.index(name)])
    bus = FeetechTcpClient("127.0.0.1", wire.start(), motors, retries=1)
    bus.connect()
    try:
        encoders, mover = neck_parts(bus, NeckConfig.from_json(REPO / "config/neck.json"))
        driver = arm_part(bus, MOTION, motors)
        core = BaseServerCore(bus, CFG, neck=encoders, mover=mover, arm=driver)
        core.tick(0.0)
        core.tick(TICK)  # the arm's second part: the first ride is whole
        assert send(core, wave(), 0.1)["status"] == "accepted"
        reads = wire.counts.sync_reads
        for i in range(10):
            core.tick(0.12 + i * TICK)
        assert wire.counts.sync_reads - reads == 10  # the wheels, the neck and the arm: one read
        assert wire.byte(4, 40) == 1 and wire.word(4, 48) == 500
        assert wire.word(4, 46) == speed_ticks(MOTION.max_speed_deg_s)
        replies = [
            len(params[2:]) * (HEADER_BYTES + params[1])
            for instruction, params in wire.counts.log
            if instruction == 0x82
        ]
        assert max(replies) <= READ_BUDGET_BYTES  # every read's replies in one USB packet
    finally:
        bus.close()
        wire.stop()
