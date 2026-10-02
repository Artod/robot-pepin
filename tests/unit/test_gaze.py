"""The gaze arbiter (pepin.gaze): bands, TTLs, preemption, home, the blind interval, the drivers.

The head is a fake driver that records every write and answers when a test says so; time is
passed in, so every rule is a few steps of arithmetic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from pepin.gaze import (
    NAVIGATION,
    PERSON,
    RETRY_S,
    Aim,
    Arbiter,
    BaseServerHead,
    EitherHead,
    GazeSettings,
    HeadReading,
    Look,
    NeckTargetHead,
    Outcome,
    Reach,
    Refusal,
    Speed,
    aim_at_point,
    depression_deg,
    home_aim,
    lens_at,
    look_from_json,
)
from pepin.neck import NeckAngles, NeckConfig, camera_pose

REPO = Path(__file__).resolve().parents[2]
CFG = NeckConfig.from_json(REPO / "config/neck.json")
HOME = home_aim(CFG)
LEFT = Aim(math.radians(30), math.radians(40))
RIGHT = Aim(math.radians(-30), math.radians(40))
SETTINGS = GazeSettings(frames=3, settle_tol_deg=1.0, move_timeout_s=3.0, frame_period_s=0.1)


class FakeHead:
    """A driver that records writes; a test sets what it is blocked by and what it answers."""

    moves_while_driving = False

    def __init__(self) -> None:
        self.writes: list[tuple[Aim | None, bool, float]] = []
        self.block: Refusal | None = None
        self.refusal: Refusal | None = None
        self.arrival: HeadReading | None = None
        self.kept = 0

    def blocked(self, now: float) -> Refusal | None:
        return self.block

    def write(self, aim: Aim | None, *, speed: Speed, hold: bool, now: float) -> None:
        self.writes.append((aim, hold, now))

    def take_refusal(self) -> Refusal | None:
        refusal, self.refusal = self.refusal, None
        return refusal

    def take_arrival(self) -> HeadReading | None:
        arrival, self.arrival = self.arrival, None
        return arrival

    def keep(self, now: float) -> None:
        self.kept += 1


def look(
    source: str = "llm.look",
    views: tuple[Aim, ...] = (LEFT,),
    band: int = PERSON,
    frames: int = 3,
    dwell_s: float = 0.0,
    ttl_s: float = 10.0,
    preempt: bool = False,
    kind: Any = "angles",
) -> Look:
    return Look(source, views, band, frames, dwell_s, ttl_s, preempt=preempt, kind=kind)


def arbiter() -> tuple[Arbiter, FakeHead, list[Outcome]]:
    head = FakeHead()
    return Arbiter(head, HOME, SETTINGS), head, []


def arrive(arb: Arbiter, head: FakeHead, aim: Aim, stamp: float) -> None:
    """The driver confirms the head arrived at ``aim`` at ``stamp``."""
    head.arrival = HeadReading(aim.pan_rad, aim.tilt_rad, stamp)
    arb.observe(HeadReading(aim.pan_rad, aim.tilt_rad, stamp))


def test_a_look_moves_settles_counts_its_frames_answers_and_falls_home() -> None:
    arb, head, out = arbiter()
    assert arb.submit(look(), 0.0, out.append).status == "granted"
    arb.step(0.0)
    assert head.writes == [(LEFT, True, 0.0)]
    state = arb.state(0.05)
    assert (state.phase, state.blind, state.blind_from, state.blind_until) == (
        "saccade",
        True,
        0.0,
        None,
    )
    arrive(arb, head, LEFT, 1.0)
    arb.step(1.0)
    state = arb.state(1.05)
    assert state.phase == "still" and state.blind and state.blind_until == pytest.approx(1.1)
    assert not arb.state(1.2).blind
    for stamp in (1.05, 1.2, 1.3):  # the first is inside the blur period: not a still frame
        arb.frame(stamp)
    arb.step(1.4)
    assert out == []
    arb.frame(1.4)
    arb.step(1.5)
    assert len(out) == 1 and out[0].status == "done" and out[0].reached
    assert out[0].frames_seen == 3 and out[0].settled_stamp == 1.0
    assert out[0].pan_rad == pytest.approx(LEFT.pan_rad)
    arb.step(1.6)  # nothing holds the head: home, released
    assert head.writes[-1] == (None, False, 1.6)
    assert arb.state(1.6).phase == "returning"
    arrive(arb, head, HOME, 2.5)
    arb.step(2.5)
    assert arb.state(2.7).phase == "home" and not arb.state(2.7).blind


def test_two_still_readings_settle_a_move_without_an_answer_from_the_driver() -> None:
    arb, _head, out = arbiter()
    arb.submit(look(frames=0), 0.0, out.append)
    arb.step(0.0)
    arb.observe(HeadReading(LEFT.pan_rad - 0.3, LEFT.tilt_rad, 0.5))  # still turning
    arb.observe(HeadReading(LEFT.pan_rad, LEFT.tilt_rad, 0.9))
    arb.step(0.9)
    assert arb.state(0.9).phase == "saccade"  # one reading at the target is not still yet
    arb.observe(HeadReading(LEFT.pan_rad + 0.001, LEFT.tilt_rad, 0.95))
    arb.step(1.0)
    assert out and out[0].status == "done" and out[0].settled_stamp == 0.95


def test_readings_from_before_the_write_settle_nothing() -> None:
    arb, _head, out = arbiter()
    arb.observe(HeadReading(LEFT.pan_rad, LEFT.tilt_rad, 0.1))
    arb.observe(HeadReading(LEFT.pan_rad, LEFT.tilt_rad, 0.2))
    arb.submit(look(frames=0), 1.0, out.append)
    arb.step(1.0)
    arb.step(1.1)
    assert not out and arb.state(1.1).phase == "saccade"


def test_a_request_past_its_ttl_expires_and_the_head_goes_home() -> None:
    arb, head, out = arbiter()
    arb.submit(look(ttl_s=2.0), 0.0, out.append)
    arb.step(0.0)
    arrive(arb, head, LEFT, 0.5)
    arb.step(0.5)
    arb.step(2.1)  # no frames ever came
    assert out[0].status == "expired" and "2.0 s" in out[0].reason
    assert out[0].settled_stamp == 0.5
    assert head.writes[-1] == (None, False, 2.1)


def test_a_lower_band_preempts_the_holder_and_takes_the_head() -> None:
    arb, head, out = arbiter()
    arb.submit(look(), 0.0, out.append)
    arb.step(0.0)
    stall: list[Outcome] = []
    assert arb.submit(look("nav.stall", (RIGHT,), NAVIGATION), 0.1, stall.append).status == (
        "granted"
    )
    arb.step(0.1)
    assert out[0].status == "preempted" and "nav.stall" in out[0].reason
    assert head.writes[-1] == (RIGHT, True, 0.1)


def test_within_a_band_the_holder_keeps_the_head_and_the_next_waits_its_turn() -> None:
    arb, head, out = arbiter()
    arb.submit(look("llm.look", frames=0), 0.0, out.append)
    arb.step(0.0)
    second: list[Outcome] = []
    assert arb.submit(look("llm.find", (RIGHT,), frames=0), 0.1, second.append).status == "queued"
    arb.step(0.1)
    assert head.writes == [(LEFT, True, 0.0)]
    arrive(arb, head, LEFT, 0.5)
    arb.step(0.5)
    assert out[0].status == "done"
    arb.step(0.6)
    assert head.writes[-1] == (RIGHT, True, 0.6)


def test_a_newcomer_that_preempts_takes_the_head_from_its_own_band() -> None:
    arb, head, out = arbiter()
    arb.submit(look("llm.look"), 0.0, out.append)
    arb.step(0.0)
    arb.submit(look("llm.find", (RIGHT,), preempt=True), 0.1)
    arb.step(0.1)
    assert out[0].status == "preempted" and head.writes[-1][0] == RIGHT


def test_a_source_asking_again_replaces_its_own_request() -> None:
    arb, head, out = arbiter()
    arb.submit(look("llm.look"), 0.0, out.append)
    arb.step(0.0)
    arb.submit(look("llm.look", (RIGHT,)), 0.1)
    arb.step(0.1)
    assert out[0].status == "preempted" and "newer request" in out[0].reason
    assert head.writes[-1][0] == RIGHT


def test_a_holder_that_only_dwells_yields_to_its_band() -> None:
    arb, head, out = arbiter()
    arb.submit(look("llm.look", frames=0, dwell_s=8.0), 0.0, out.append)
    arb.step(0.0)
    arrive(arb, head, LEFT, 0.5)
    arb.step(0.5)
    assert out[0].status == "done"
    arb.step(1.0)
    assert len(head.writes) == 1  # dwelling: the head stays
    arb.submit(look("llm.find", (RIGHT,)), 1.1)
    arb.step(1.1)
    assert head.writes[-1][0] == RIGHT


def test_a_dwell_holds_the_head_until_it_ends() -> None:
    arb, head, out = arbiter()
    arb.submit(look(frames=0, dwell_s=2.0), 0.0, out.append)
    arb.step(0.0)
    arrive(arb, head, LEFT, 0.5)
    arb.step(0.5)
    arb.step(2.4)
    assert len(head.writes) == 1
    arb.step(2.6)
    assert head.writes[-1][0] is None


def test_a_transient_refusal_is_tried_again_and_a_lasting_one_denies() -> None:
    arb, head, out = arbiter()
    arb.submit(look(frames=0), 0.0, out.append)
    arb.step(0.0)
    head.refusal = Refusal("the wheels are moving", True)
    arb.step(0.05)
    assert arb.state(0.05).phase == "home"  # the refused write never moved the head
    arb.step(0.05 + RETRY_S / 2)
    assert len(head.writes) == 1  # not before the retry pause
    arb.step(0.06 + RETRY_S)
    assert len(head.writes) == 2 and not out
    head.refusal = Refusal("head target 9999 is outside its limits", False)
    arb.step(0.5)
    assert out[0].status == "denied" and "outside its limits" in out[0].reason


def test_a_driver_blocked_past_the_move_timeout_denies_with_its_reason() -> None:
    arb, head, out = arbiter()
    head.block = Refusal("the wheels are moving", True)
    arb.submit(look(), 0.0, out.append)
    arb.step(0.0)
    arb.step(2.9)
    assert not out and not head.writes
    arb.step(3.1)
    assert out[0].status == "denied" and out[0].reason == "the wheels are moving"


def test_a_move_that_never_settles_ends_unreached_where_the_encoders_say() -> None:
    arb, head, out = arbiter()
    arb.submit(look(frames=0), 0.0, out.append)
    arb.step(0.0)
    arb.observe(HeadReading(0.2, 0.5, 1.0))
    arb.step(3.2)
    assert out[0].status == "done" and not out[0].reached
    assert "did not settle" in out[0].reason
    assert arb.state(3.2).target == Aim(0.2, 0.5)  # where the encoders say, not the aim
    arb.step(3.3)
    assert head.writes[-1] == (None, False, 3.3)


def test_release_drops_what_the_caller_does_not_keep() -> None:
    arb, _head, out = arbiter()
    arb.submit(look("llm.look"), 0.0, out.append)
    stall: list[Outcome] = []
    arb.submit(look("nav.stall", band=NAVIGATION), 0.0, stall.append)
    gone = arb.release(0.1, "nav.drive", keep=lambda request: request.band < PERSON)
    assert gone == 1 and out[0].status == "preempted" and out[0].reason == "by nav.drive"
    assert not stall and [request.source for request in arb.pending()] == ["nav.stall"]


def test_renew_restarts_the_ttl() -> None:
    arb, _head, out = arbiter()
    arb.submit(look(ttl_s=2.0), 0.0, out.append)
    arb.step(0.0)
    assert arb.renew("llm.look", 1.5) == 1
    arb.step(3.0)
    assert not out


def test_renewing_an_answered_request_keeps_holding_the_head() -> None:
    """Path gaze: answered at once (no frames), held for its dwell, renewed every period."""
    arb, head, out = arbiter()
    arb.submit(look("nav.path", frames=0, dwell_s=0.5, ttl_s=0.5), 0.0, out.append)
    arb.step(0.0)
    arrive(arb, head, LEFT, 0.2)
    arb.step(0.2)
    assert out[0].status == "done"
    for t in (0.4, 0.6, 0.8, 1.0, 1.2):
        arb.renew("nav.path", t)
        arb.step(t + 0.05)
    assert len(head.writes) == 1  # held all along
    arb.step(1.8)
    assert head.writes[-1][0] is None  # the renewals stopped: home


def test_a_home_request_is_answered_at_home() -> None:
    arb, head, out = arbiter()
    arb.submit(look(frames=0), 0.0)
    arb.step(0.0)
    arrive(arb, head, LEFT, 0.5)
    arb.step(0.5)
    arb.submit(look("nav.stall", (), NAVIGATION, kind="home"), 0.6, out.append)
    arb.step(0.6)
    assert head.writes[-1] == (None, False, 0.6)
    arrive(arb, head, HOME, 1.4)
    arb.step(1.4)
    assert out[0].status == "done" and out[0].reached and arb.state(1.4).phase == "home"


def test_what_the_arbiter_cannot_do_is_denied_at_once() -> None:
    arb, _head, out = arbiter()
    assert arb.submit(look(kind="track"), 0.0, out.append).reason == "track is not built yet"
    assert "band 7" in arb.submit(look(band=7), 0.0).reason
    assert arb.submit(look(views=()), 0.0).reason == "no view to look at"
    assert out[0].status == "denied"


def test_nothing_written_means_nothing_to_undo() -> None:
    arb, head, _out = arbiter()
    arb.step(0.0)
    arb.step(1.0)
    assert head.writes == [] and arb.state(1.0).phase == "home" and head.kept == 2


def test_the_state_as_json() -> None:
    arb, _head, _out = arbiter()
    arb.submit(look("nav.stall", band=NAVIGATION), 0.0)
    arb.step(0.0)
    data = json.loads(arb.state(0.1).to_json())
    assert data["phase"] == "saccade" and data["source"] == "nav.stall" and data["band"] == 1
    assert data["blind"] is True and data["blind_until"] is None and data["pan_rad"] is None
    assert data["target"]["pan_deg"] == pytest.approx(30.0)


# ---- geometry ----------------------------------------------------------------------------------
def optical_miss_m(aim: Aim, point: tuple[float, float, float]) -> float:
    """How far the point lies from the camera's optical axis with the head at ``aim``."""
    x, y, z, _r, pitch, yaw = camera_pose(CFG, NeckAngles(aim.pan_rad, aim.tilt_rad))
    axis = (
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        -math.sin(pitch),
    )
    d = (point[0] - x, point[1] - y, point[2] - z)
    along = sum(a * b for a, b in zip(d, axis, strict=True))
    return math.sqrt(max(sum(v * v for v in d) - along * along, 0.0))


@pytest.mark.parametrize(
    "point", [(1.0, 0.0, 0.0), (0.5, 0.3, 0.2), (2.0, -1.0, 1.2), (0.6, 0.0, 0.3)]
)
def test_aim_at_point_puts_the_point_on_the_optical_axis(point: tuple[float, float, float]) -> None:
    assert optical_miss_m(aim_at_point(CFG, point), point) < 1e-3


def test_a_floor_point_a_metre_ahead_is_about_fifty_degrees_down() -> None:
    aim = aim_at_point(CFG, (1.0, 0.0, 0.0))
    assert aim.pan_rad == pytest.approx(0.0, abs=1e-9)
    assert math.degrees(aim.tilt_rad) == pytest.approx(50.0, abs=1.5)
    assert depression_deg(CFG, aim, (1.0, 0.0, 0.0)) == pytest.approx(
        math.degrees(aim.tilt_rad), abs=1e-6
    )


def test_the_reach_names_what_is_past_it_and_clamps() -> None:
    reach = Reach.of(CFG)
    assert reach.refusal(HOME) is None
    assert "turn" in str(reach.refusal(Aim(math.radians(170), HOME.tilt_rad)))
    assert "down" in str(reach.refusal(Aim(0.0, math.radians(80))))
    assert reach.clamp(Aim(0.0, math.radians(80))).tilt_rad == pytest.approx(reach.tilt[1])
    assert lens_at(CFG, HOME)[2] == pytest.approx(CFG.reference.z_m, abs=1e-6)


def to_base(frame: str, xyz: dict[str, float]) -> tuple[float, float, float] | str:
    if frame != "base_link":
        return f"no {frame} -> base_link"
    return xyz["x"], xyz["y"], xyz["z"]


def test_requests_from_json() -> None:
    angles = look_from_json(
        {"source": "llm.look", "kind": "angles", "target": {"pan_rad": 0.5, "tilt_rad": 0.4}},
        SETTINGS,
        CFG,
        to_base,
    )
    assert isinstance(angles, Look) and angles.views == (Aim(0.5, 0.4),)
    assert (angles.band, angles.frames, angles.ttl_s) == (PERSON, 3, 10.0)
    point = look_from_json(
        {"source": "nav.stall", "kind": "point", "band": 1, "target": {"x": 1, "y": 0, "z": 0}},
        SETTINGS,
        CFG,
        to_base,
    )
    assert isinstance(point, Look) and point.ttl_s == 3.0 and point.kind == "point"
    scan = look_from_json(
        {
            "source": "llm.look_around",
            "kind": "scan",
            "target": {
                "views": [{"pan_rad": 1.0, "tilt_rad": 0.4}, {"pan_rad": 0, "tilt_rad": 0.4}]
            },
            "id": "abc",
        },
        SETTINGS,
        CFG,
        to_base,
    )
    assert isinstance(scan, Look) and len(scan.views) == 2 and scan.id == "abc"
    home = look_from_json({"source": "x", "kind": "home"}, SETTINGS, CFG, to_base)
    assert isinstance(home, Look) and home.views == ()


MAP_POINT = {"frame": "map", "x": 1, "y": 0, "z": 0}


@pytest.mark.parametrize(
    ("data", "why"),
    [
        ({"source": "x", "kind": "stare"}, "kind 'stare'"),
        ({"kind": "home"}, "names its source"),
        ({"source": "x", "kind": "angles", "target": {"pan_rad": 3.0, "tilt_rad": 0.4}}, "reach"),
        ({"source": "x", "kind": "angles", "target": {"pan_rad": 0.1}}, "incomplete"),
        ({"source": "x", "kind": "point", "target": MAP_POINT}, "no map"),
        ({"source": "x", "speed": "fast"}, "speed 'fast'"),
        ({"source": "x", "band": "one"}, "bad number"),
        ({"source": "x", "kind": "direction", "target": {"frame": "odom"}}, "base_link"),
    ],
)
def test_bad_requests_are_refused_in_words(data: dict[str, Any], why: str) -> None:
    answer = look_from_json(data, SETTINGS, CFG, to_base)
    assert isinstance(answer, str) and why in answer


# ---- the drivers -------------------------------------------------------------------------------
class Wire:
    """The base server's socket: what was sent, and whether it went."""

    def __init__(self, up: bool = True) -> None:
        self.lines: list[dict[str, Any]] = []
        self.up = up

    def send(self, data: bytes) -> bool:
        self.lines.append(json.loads(data))
        return self.up


def test_the_base_server_driver_speaks_neck_goto_and_reads_its_answers() -> None:
    wire = Wire()
    head = BaseServerHead(CFG, wire.send)
    assert head.blocked(0.0) == Refusal("no state line from the base server", True)
    head.on_line({"type": "state", "moving": False, "v": 0.0, "w": 0.0}, 0.0)
    assert head.blocked(0.1) is None
    head.write(LEFT, speed="saccade", hold=True, now=0.1)
    sent = wire.lines[-1]
    assert sent["cmd"] == "neck_goto" and sent["hold"] is True
    assert isinstance(sent["pan_ticks"], int) and isinstance(sent["tilt_ticks"], int)
    assert head.blocked(0.2) == Refusal("a neck move is under way", True)
    head.on_line(
        {
            "type": "neck_goto",
            "reached": True,
            "pan_ticks": sent["pan_ticks"],
            "tilt_ticks": sent["tilt_ticks"],
        },
        1.0,
    )
    arrival = head.take_arrival()
    assert arrival is not None and arrival.aim.off(LEFT) < math.radians(0.1)
    assert head.take_arrival() is None
    head.write(None, speed="saccade", hold=False, now=1.1)
    assert wire.lines[-1] == {"cmd": "neck_home", "hold": False}
    head.on_line({"type": "neck_goto", "reached": False, "error": "the wheels are moving"}, 1.2)
    assert head.take_refusal() == Refusal("the wheels are moving", True)
    head.on_line({"type": "state", "moving": True, "v": -0.1, "w": 0.2}, 1.3)
    assert head.blocked(1.3) is not None and head.twist == (-0.1, 0.2)
    assert head.blocked(5.0) == Refusal("no state line from the base server", True)


def test_a_dead_link_is_a_transient_refusal() -> None:
    head = BaseServerHead(CFG, Wire(up=False).send)
    head.on_line({"type": "state", "moving": False}, 0.0)
    head.write(LEFT, speed="saccade", hold=True, now=0.0)
    assert head.take_refusal() == Refusal("the link to the base server is down", True)
    assert head.blocked(0.1) is None


def target_head(wire: Wire) -> NeckTargetHead:
    return NeckTargetHead(CFG, wire.send, slow_deg_s=lambda: 20.0, renew_s=lambda: 0.5)


def test_the_neck_target_driver_renews_its_lease_while_it_holds() -> None:
    wire = Wire()
    head = target_head(wire)
    head.on_line({"type": "state", "moving": True}, 0.0)
    assert head.blocked(0.0) is None  # the wheels turning do not matter here
    head.write(LEFT, speed="saccade", hold=True, now=0.0)
    first = wire.lines[-1]
    assert first == {"cmd": "neck_target", "pan_rad": LEFT.pan_rad, "tilt_rad": LEFT.tilt_rad}
    head.keep(0.3)
    assert len(wire.lines) == 1
    head.keep(0.5)
    assert len(wire.lines) == 2 and wire.lines[-1] == first
    head.write(RIGHT, speed="slow", hold=True, now=0.6)
    assert wire.lines[-1]["speed_deg_s"] == 20.0
    head.write(None, speed="saccade", hold=False, now=1.2)
    assert wire.lines[-1]["pan_rad"] == 0.0
    assert wire.lines[-1]["tilt_rad"] == pytest.approx(HOME.tilt_rad)
    head.keep(5.0)
    assert len(wire.lines) == 4  # home is not renewed: the lease lapses and the board lets go
    head.on_line({"type": "neck_target", "error": "an operator jog holds the head"}, 5.0)
    assert head.take_refusal() == Refusal("an operator jog holds the head", True)
    head.on_line({"type": "neck_target", "error": "tilt target is outside its limits"}, 5.0)
    assert head.take_refusal() == Refusal("tilt target is outside its limits", False)
    assert head.take_arrival() is None


def test_either_head_picks_neck_target_once_a_state_line_carries_the_neck() -> None:
    wire = Wire()
    either = EitherHead(BaseServerHead(CFG, wire.send), target_head(wire))
    either.on_line({"type": "state", "moving": True, "v": 0.2, "w": 0.0}, 0.0)
    assert not either.moves_while_driving and either.wheels_moving and either.twist == (0.2, 0.0)
    assert either.blocked(0.0) is not None
    either.on_line({"type": "state", "moving": True, "pan_ticks": 2029, "tilt_ticks": 2311}, 0.1)
    assert either.moves_while_driving and either.blocked(0.1) is None
    either.write(LEFT, speed="saccade", hold=True, now=0.1)
    assert wire.lines[-1]["cmd"] == "neck_target"
    either.on_line({"type": "state", "moving": False}, 0.2)  # a tick the neck missed
    assert either.speaks_target and either.take_refusal() is None
    either.keep(1.0)
    assert len(wire.lines) == 2 and either.take_arrival() is None
