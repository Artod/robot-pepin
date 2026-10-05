"""The three gaze halves against each other: the arbiter's /gaze/state as the frame gate reads it,
the arbiter's neck_target as the base server takes it, the renewal inside the board's lease."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from pepin import base_server
from pepin.flags import load_knobs
from pepin.gaze import Aim, Arbiter, GazeSettings, HeadReading, Look, NeckTargetHead, home_aim
from pepin.gaze_gate import BLIND, EXPOSURE_S, SETTLE_S, FrameGate, GazeState
from pepin.neck import NeckConfig

REPO = Path(__file__).resolve().parents[2]
CFG = NeckConfig.from_json(REPO / "config/neck.json")
LEFT = Aim(math.radians(30), math.radians(40))
PERIOD = GazeSettings().frame_period_s


class StillHead:
    """A driver that takes every write and answers nothing (the encoders settle it)."""

    moves_while_driving = True

    def __init__(self) -> None:
        self.writes: list[Any] = []

    def blocked(self, now: float) -> None:
        return None

    def write(self, aim: Aim | None, *, speed: str, hold: bool, now: float) -> None:
        self.writes.append((aim, now))

    def take_refusal(self) -> None:
        return None

    def take_arrival(self) -> None:
        return None

    def keep(self, now: float) -> None:
        pass


def test_the_gate_drops_exactly_the_frames_of_a_saccade_as_the_arbiter_publishes_it() -> None:
    """The arbiter publishes at 10 Hz and on every change; the gate opens a blind interval at the
    first blind state's ``since`` (the write) and closes it at the settled phase's ``since`` (the
    settling reading) plus its own settle_s."""
    arbiter = Arbiter(StillHead(), home_aim(CFG))
    gate = FrameGate(exposure_s=EXPOSURE_S, settle_s=SETTLE_S, yaw_dps=0.0)
    states: list[GazeState] = []

    def publish(now: float) -> None:
        state = GazeState.from_json(arbiter.state(now).to_json())
        assert state is not None, "every message is a state the gate can read"
        states.append(state)
        gate.observe_state(state, now)

    arbiter.submit(Look("llm.look", (LEFT,), 2, 0, 5.0, 10.0), 10.0)
    for step in range(30):  # 1.5 s at the step rate, a state every 0.1 s and on every change
        now = 10.0 + step * 0.05
        if now >= 10.6:  # the encoders: arrived at 10.6 and still
            arbiter.observe(HeadReading(LEFT.pan_rad, LEFT.tilt_rad, now))
        arbiter.step(now)
        publish(now)
    blind = [s for s in states if s.blind]
    assert blind[0].phase == "saccade" and blind[0].since == pytest.approx(10.0)
    assert len([s for s in blind if s.phase == "saccade"]) >= 10, "10 Hz through the move"
    settled = next(s for s in states if s.phase == "still")
    assert settled.since == pytest.approx(10.65), "the settling reading's own stamp"
    now = 11.5
    assert gate.verdict(9.9, now) is None, "before the write"
    assert gate.verdict(10.3, now) == BLIND, "inside the move"
    assert gate.verdict(10.65 + SETTLE_S, now) == BLIND, "still inside the tail"
    # the stamp is the exposure's END by default (gate_stamp_end 1 since 2026-10-05): the window
    # looks back two exposures, so the first clean frame is stamped that much after the tail
    assert gate.verdict(10.65 + SETTLE_S + 2 * EXPOSURE_S + 0.02, now) is None, "after it"


def test_the_neck_target_the_arbiter_sends_is_one_the_base_server_takes() -> None:
    """Every field the arbiter writes is one the base server reads, read back by its own parsers
    as the angles and the ceiling that were meant (a saccade asks for no ceiling: the board's
    top speed)."""
    sent: list[dict[str, Any]] = []

    def send(line: bytes) -> bool:
        sent.append(json.loads(line))
        return True

    head = NeckTargetHead(CFG, send, slow_deg_s=lambda: 20.0, renew_s=lambda: 0.5)
    head.write(LEFT, speed="slow", hold=True, now=0.0)
    head.write(LEFT, speed="saccade", hold=True, now=0.1)
    slow, saccade = sent
    for message in sent:
        assert message["cmd"] == "neck_target"
        assert set(message) <= {"cmd", "pan_rad", "tilt_rad", "speed_deg_s", "acc_deg_s2"}
        assert base_server._angle(message, "pan_rad") == pytest.approx(LEFT.pan_rad)
        assert base_server._angle(message, "tilt_rad") == pytest.approx(LEFT.tilt_rad)
    assert base_server._ceiling(slow, "speed_deg_s") == 20.0
    assert base_server._ceiling(saccade, "speed_deg_s") is None


def test_a_held_target_is_renewed_well_inside_the_board_s_lease() -> None:
    renew = load_knobs("gaze", REPO / "config/knobs.json")["target_renew_s"]
    assert renew <= CFG.motion.lease_s / 2.0
