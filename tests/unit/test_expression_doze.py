"""Dozing (2026-10-15): outside a conversation the face falls asleep after the quiet time and
any wake lets it go at once (pepin.expression.doze, over a fake head server's door)."""

from __future__ import annotations

import json
from pathlib import Path

from pepin.expression.conversation import Heard
from pepin.expression.doze import FALL_ASLEEP_MS, Doze, DozeConfig, doze_config
from pepin.expression.fakes import FakeHeadFace

CONFIG = Path(__file__).resolve().parents[2] / "config"
ASLEEP = ("express", ("sleepy", 1.0, None, FALL_ASLEEP_MS))
AWAKE = ("clear", ())


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def make(**knobs: float | bool | str) -> tuple[Clock, FakeHeadFace, Doze]:
    clock, face = Clock(), FakeHeadFace()
    config = DozeConfig(**knobs)  # type: ignore[arg-type]
    return clock, face, Doze(face, lambda: config, clock=clock)


def wait(doze: Doze, clock: Clock, seconds: float) -> None:
    """``seconds`` of its thread's steps, one a second."""
    for _ in range(round(seconds)):
        clock.now += 1.0
        doze.step()


def test_the_face_dozes_after_the_quiet_time_without_a_conversation() -> None:
    clock, face, doze = make(idle_after_s=30.0)
    wait(doze, clock, 29)
    assert face.calls == [] and not doze.dozing
    wait(doze, clock, 1)
    assert face.calls == [ASLEEP] and doze.dozing  # standing: until a wake
    wait(doze, clock, 60)
    assert face.calls == [ASLEEP]  # asked once


def test_the_wake_word_wakes_it_at_once_and_it_stays_awake_until_the_close() -> None:
    clock, face, doze = make(idle_after_s=30.0)
    wait(doze, clock, 30)
    doze.opened(Heard(0.0, 1.0, "Пепин", named=True))
    assert face.calls == [ASLEEP, AWAKE] and not doze.dozing
    wait(doze, clock, 300)  # a long conversation, quiet between its turns
    assert face.calls == [ASLEEP, AWAKE]
    doze.closed("idle")
    wait(doze, clock, 29)
    doze.speech_started(clock.now)  # a moment counts as the robot's life: the quiet starts again
    wait(doze, clock, 29)
    assert face.calls == [ASLEEP, AWAKE]
    wait(doze, clock, 1)
    assert face.calls[-1] == ASLEEP


def test_another_source_on_the_face_wakes_it_and_holds_it_awake() -> None:
    """A drive from the goal server (its focus face), a tool's expression, the command line:
    the head server's status names who holds a request; the doze's own source is not one."""
    clock, face, doze = make(idle_after_s=10.0)
    wait(doze, clock, 10)
    face.last_status = {"sources": ["doze"]}
    wait(doze, clock, 5)
    assert face.calls == [ASLEEP]
    face.last_status = {"sources": ["doze", "goal"]}
    wait(doze, clock, 1)
    assert face.calls == [ASLEEP, AWAKE]
    wait(doze, clock, 60)
    assert face.calls == [ASLEEP, AWAKE]  # never asleep while the goal's face shows
    face.last_status = {"sources": []}
    wait(doze, clock, 9)
    assert face.calls == [ASLEEP, AWAKE]
    wait(doze, clock, 1)
    assert face.calls[-1] == ASLEEP


def test_a_drive_keeps_it_awake_and_the_switch_turns_it_off_and_on() -> None:
    clock, face, doze = make(idle_after_s=10.0)
    doze.drive(True)
    wait(doze, clock, 60)
    assert face.calls == []
    doze.drive(False)
    wait(doze, clock, 10)
    assert face.calls == [ASLEEP]
    assert doze.flag == "doze" and doze.on
    assert doze.switch(False) is None and not doze.on
    assert face.calls == [ASLEEP, AWAKE]
    wait(doze, clock, 60)
    assert face.calls == [ASLEEP, AWAKE]  # off: never asleep
    doze.switch(True)
    wait(doze, clock, 10)
    assert face.calls[-1] == ASLEEP
    doze.close()  # the voice loop ends: nobody listens, the face let go
    assert face.calls[-1] == AWAKE and not doze.dozing


def test_the_knobs_and_the_flag_come_from_the_expression_config() -> None:
    config = doze_config(json.loads((CONFIG / "expression.json").read_text()))
    assert config == DozeConfig(on=True, idle_after_s=30.0, expression="sleepy")
    assert doze_config({"flags": {"doze": False}}) == DozeConfig(on=False)
    _, _, doze = make(on=False)
    assert not doze.on  # the flag's default is the config's
