"""The speaker gaze (pepin.expression doa, plan, policy, tracker): the voice's bearing, the head
and the wheels' split, whom to look at, and the state machine on a fake clock against the fakes
of every seam (pepin.expression.fakes)."""
# ruff: noqa: RUF001, RUF002 -- Russian speech is the data here, not a lookalike of Latin

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from pepin.audio_link import DoaReading
from pepin.expression.conversation import SPEAKER_GAZE, Heard
from pepin.expression.doa import (
    bearing_deg,
    fit_doa_map,
    utterance_readings,
    voice_of,
    wrap_deg,
)
from pepin.expression.executors import NoWheels, WheelMove, WheelReport
from pepin.expression.fakes import FakeDoa, FakeFace, FakeFaceFinder, FakeGazeDoor, FakeWheels
from pepin.expression.head import GESTURE, IDLE_LOOK, SPEAKER
from pepin.expression.plan import BodyLimits, BoundedTurn, Step, head_after, split
from pepin.expression.policy import SpeakerPolicy, face_aim
from pepin.expression.speaker import (
    DoaMap,
    FaceFrame,
    FaceSighting,
    GazeAnswer,
    HeadView,
    Target,
)
from pepin.expression.tracker import SpeakerGaze, SpeakerGazeWorker, SpeakerSettings, flag_default
from pepin.gaze import HOME, Aim, Reach, Speed
from pepin.neck import NeckConfig

REPO = Path(__file__).resolve().parents[2]
REACH = Reach.of(NeckConfig.from_json(REPO / "config/neck.json"))
BOARD = 5000.0  # the board's clock runs this far ahead of the laptop's in these tests
MAP = DoaMap(front_deg=180.0, sign=1)  # the array reads 180 straight ahead, growing leftward
# The rules run at 3 Hz (their times were worked out at it); the rate's own tests at the shipped 10.
RULES = SpeakerSettings(doa_map=MAP, track_hz=3.0)
SEEK_TILT = math.radians(-5.0)
HOME_AIM = Aim(0.0, math.radians(23.8))


def rad(deg: float) -> float:
    return math.radians(deg)


# ---- the voice's bearing ----------
def test_angles_wrap_and_the_array_turns_into_the_cart_s_frame() -> None:
    assert [wrap_deg(d) for d in (0.0, 180.0, -180.0, 190.0, -190.0, 720.0)] == [
        0.0,
        180.0,
        180.0,
        -170.0,
        170.0,
        0.0,
    ]
    assert bearing_deg(240.0, MAP) == 60.0  # 60 deg left of the nose
    assert bearing_deg(240.0, DoaMap(180.0, -1)) == -60.0
    assert bearing_deg(10.0, DoaMap(350.0, 1)) == 20.0


def test_an_utterance_s_bearing_is_the_mean_of_its_speech_readings() -> None:
    readings = [DoaReading(0.0, 238, True), DoaReading(0.1, 242, True), DoaReading(0.2, 30, False)]
    voice = voice_of(readings, MAP)
    assert voice.doa_deg == pytest.approx(240.0) and voice.speech == 2 and voice.readings == 3
    assert voice.bearing_rad == pytest.approx(rad(60.0))
    assert "+60 deg" in voice.text()
    unmeasured = voice_of(readings, None)
    assert unmeasured.doa_deg == pytest.approx(240.0) and unmeasured.bearing_rad is None
    assert "unmeasured" in unmeasured.text()
    assert voice_of(readings, MAP, min_speech=3).doa_deg is None, "two speech readings of three"
    across = voice_of([DoaReading(0.0, 359, True), DoaReading(0.1, 1, True)], MAP).doa_deg
    assert across is not None and wrap_deg(across) == pytest.approx(0.0, abs=1e-9)


def test_the_utterance_s_window_is_mapped_onto_the_board_s_clock_by_the_newest_reading() -> None:
    """Whisper read the utterance 0.8 s after it ended: its last second ends 0.8 s before the
    newest reading, not at it."""
    readings = [DoaReading(BOARD + t / 10, 240 if t < 25 else 30, t < 25) for t in range(34)]
    doa = FakeDoa(readings)  # speech up to board +2.4 s, the room after
    heard = Heard(start_s=0.5, end_s=2.5)  # laptop clock; read at 3.3 = board +3.3
    window = utterance_readings(doa, heard, now=3.3, window_s=1.0)
    stamps = [round(r.stamp_s - BOARD, 1) for r in window]
    assert stamps[0] == pytest.approx(1.5) and stamps[-1] == pytest.approx(2.5)
    assert voice_of(window, MAP).bearing_rad == pytest.approx(rad(60.0))
    short = utterance_readings(doa, Heard(start_s=2.2, end_s=2.5), now=3.3, window_s=1.0)
    assert [round(r.stamp_s - BOARD, 1) for r in short] == [2.2, 2.3, 2.4, 2.5]
    assert utterance_readings(FakeDoa(), heard, now=3.3, window_s=1.0) == []


def test_the_calibration_gives_the_front_the_sign_and_the_quarters_and_refuses_a_lost_one() -> None:
    left_up = fit_doa_map(front=180.0, left=272.0, back=355.0, right=88.0)
    assert left_up.doa_map == DoaMap(180.0, 1, (272.0, 355.0, 88.0)) and left_up.refusal is None
    assert left_up.misses_deg[1] == pytest.approx(2.0)
    right_up = fit_doa_map(front=10.0, left=280.0, back=190.0, right=100.0)
    assert right_up.doa_map.sign == -1 and right_up.refusal is None
    skewed = fit_doa_map(front=180.0, left=230.0, back=0.0, right=90.0)
    assert skewed.refusal is None, "40 deg off: an uneven quarter the interpolation absorbs"
    lost = fit_doa_map(front=180.0, left=220.0, back=0.0, right=90.0)
    assert lost.refusal is not None and "50 deg off" in lost.refusal, "nearer the front than left"
    assert "REFUSED" in lost.text() and "front_deg 180, sign +1" in left_up.text()
    assert fit_doa_map(180.0, 270.0, 0.0, 90.0, tol_deg=60.0).refusal is None
    assert fit_doa_map(180.0, 300.0, 250.0, 90.0, tol_deg=90.0).refusal is not None, "out of turn"


TODAY = fit_doa_map(front=170.0, left=106.0, back=343.0, right=278.0)  # 2026-10-12, off-centre


def test_the_off_centre_array_s_four_readings_are_its_quarter_turns_and_between_interpolates() -> (
    None
):
    """The array at the cart's front-left read 170/106/343/278: misses of +0, -26, +7, -18 deg
    under an offset and a sign; the interpolated map puts each on its quarter exactly."""
    m = TODAY.doa_map
    assert TODAY.refusal is None and m.quarters == (106.0, 343.0, 278.0)
    assert [round(v) for v in TODAY.misses_deg] == [0, -26, 7, -18]
    assert [bearing_deg(d, m) for d in (170.0, 106.0, 343.0, 278.0)] == [0.0, 90.0, 180.0, -90.0]
    assert bearing_deg(138.0, m) == pytest.approx(45.0), "half way from the front to the left"
    assert bearing_deg(44.5, m) == pytest.approx(135.0), "half way from the left to behind"
    assert bearing_deg(310.5, m) == pytest.approx(-135.0)
    assert bearing_deg(224.0, m) == pytest.approx(-45.0)
    # the live run's voices: behind-left read 82 (+88 by the offset; the face was at +111)
    assert bearing_deg(82.0, m) == pytest.approx(107.6, abs=0.1)
    assert bearing_deg(82.0, DoaMap(170.0, -1)) == pytest.approx(88.0)
    assert bearing_deg(246.0, m) == pytest.approx(-63.3, abs=0.1), "the quiet one: -76 before"


# ---- the head and the wheels ----------
def test_the_head_alone_within_its_comfort_and_the_wheels_for_the_rest() -> None:
    near = split(rad(45), SEEK_TILT, comfort_rad=rad(60), max_turn_rad=rad(30))
    assert near.body_rad == 0.0 and near.head.pan_rad == pytest.approx(rad(45))
    far = split(rad(80), SEEK_TILT, comfort_rad=rad(60), max_turn_rad=rad(30))
    assert far.body_rad == pytest.approx(rad(20)) and far.head.pan_rad == pytest.approx(rad(60))
    back = split(rad(-150), SEEK_TILT, comfort_rad=rad(60), max_turn_rad=rad(30))
    assert back.body_rad == pytest.approx(rad(-30)) and back.head.pan_rad == pytest.approx(
        rad(-120)
    )
    assert head_after(rad(170), 0.0, SEEK_TILT, REACH).pan_rad == pytest.approx(REACH.pan[1])


class Steps:
    """The wheels' attempts: each turn answered from ``blocked`` (a stall short of the turn),
    ``stopped`` ends one for good; a back-off answered from ``backs`` (its blocked reason)."""

    def __init__(
        self,
        blocked: list[float] | None = None,
        stopped: str = "",
        backs: list[str] | None = None,
    ) -> None:
        self.blocked = deque(blocked or [])
        self.stopped = stopped
        self.backs = deque(backs or [])
        self.calls: list[tuple[str, float]] = []

    def turn(self, deg: float) -> Step:
        self.calls.append(("turn", deg))
        if self.stopped:
            return Step(math.copysign(2.0, deg), stopped=self.stopped, cut=True)
        if self.blocked:
            return Step(math.copysign(self.blocked.popleft(), deg), blocked="stalled")
        return Step(deg)

    def back_off(self, metres: float) -> Step:
        self.calls.append(("back", metres))
        blocked = self.backs.popleft() if self.backs else ""
        return Step(0.0 if blocked else metres, blocked=blocked)


def test_a_turn_goes_in_one_go_backs_off_on_a_stall_and_gives_up_within_its_bounds() -> None:
    steps = Steps()
    done = BoundedTurn(steps).move(WheelMove(turn_deg=25.0))
    assert done.done and done.turned_deg == pytest.approx(25.0) and done.attempts == 1
    assert steps.calls == [("turn", 25.0)], "one continuous turn, its steps are checkpoints"
    steps = Steps(blocked=[4.0])
    retried = BoundedTurn(steps).move(WheelMove(turn_deg=-20.0))
    assert retried.done and retried.backed_m == pytest.approx(0.03) and retried.attempts == 2
    assert steps.calls == [("turn", -20.0), ("back", 0.03), ("turn", -16.0)]
    steps = Steps(blocked=[1.0, 1.0, 1.0, 1.0])
    given_up = BoundedTurn(steps).move(WheelMove(turn_deg=20.0))
    assert not given_up.done and given_up.attempts == 3 and given_up.turned_deg == 3.0
    assert given_up.backed_m == pytest.approx(0.06) and given_up.why == "gave up: stalled"
    assert [c for c in steps.calls if c[0] == "back"] == [("back", 0.03), ("back", 0.03)]
    assert "past 30" in BoundedTurn(Steps()).move(WheelMove(turn_deg=35.0)).why
    assert "rock" in BoundedTurn(Steps()).move(WheelMove(rock_m=0.01, rocks=2)).why
    tight = BoundedTurn(Steps(blocked=[1.0, 1.0]), BodyLimits(attempts=5, max_backoff_m=0.03))
    assert tight.move(WheelMove(turn_deg=10.0)).backed_m == pytest.approx(0.03)


def test_a_stopped_turn_never_backs_off_and_a_blocked_back_off_gives_up() -> None:
    """A halt, someone else's stop, a drive: final (no back-off after the red button)."""
    steps = Steps(stopped="halted: the person speaks")
    cut = BoundedTurn(steps).move(WheelMove(turn_deg=20.0))
    assert not cut.done and cut.cut and cut.why == "halted: the person speaks"
    assert steps.calls == [("turn", 20.0)] and cut.turned_deg == 2.0
    steps = Steps(blocked=[1.0], backs=["the back-off is blocked: a return behind"])
    stuck = BoundedTurn(steps).move(WheelMove(turn_deg=20.0))
    assert not stuck.done and stuck.why == "gave up: the back-off is blocked: a return behind"
    assert stuck.attempts == 1 and stuck.backed_m == 0.0


# ---- whom to look at ----------
def face(pan: float, width: float = 60.0, tilt: float = 0.0, who: str | None = None) -> Any:
    return FaceSighting(pan, tilt, width, 0.8, who)


def frame(*faces: FaceSighting) -> FaceFrame:
    return FaceFrame(0.0, faces)


@dataclass
class Named:
    name: str | None

    def wanted(self) -> str | None:
        return self.name


def test_the_speaker_s_face_is_the_one_nearest_the_voice_else_nearest_the_centre() -> None:
    policy = SpeakerPolicy(match_deg=20.0, doa_jump_deg=30.0)
    head = Aim(rad(30), 0.0)
    two = frame(face(-10.0), face(15.0, width=40.0))  # at +20 and +45 deg in the cart's frame
    assert policy.pick(rad(40), two, head) == two.faces[1], "the face within 20 deg of the voice"
    assert policy.pick(rad(-40), two, head) == two.faces[0], "none near the voice: the centre's"
    assert policy.pick(None, two, head) == two.faces[0]
    tie = frame(face(5.0, width=30.0), face(-5.0, width=80.0))
    assert policy.pick(None, tie, head) == tie.faces[1], "the widest on a tie"
    assert policy.pick(None, frame(), head) is None and policy.pick(None, None, head) is None
    aim = policy.choose(rad(40), two, head)
    assert aim is not None and aim.pan_rad == pytest.approx(rad(45))
    named = SpeakerPolicy(match_deg=20.0, doa_jump_deg=30.0, addressee=Named("artem"))
    known = frame(face(-10.0), face(25.0, who="artem"))
    assert named.pick(rad(20), known, head) == known.faces[1], "the addressee before the voice"


def test_a_new_speaker_is_a_voice_far_from_the_person_held() -> None:
    policy = SpeakerPolicy(match_deg=20.0, doa_jump_deg=30.0)
    assert policy.new_speaker(rad(10), rad(35)) is False
    assert policy.new_speaker(rad(10), rad(45)) is True
    assert policy.new_speaker(rad(170), rad(-170)) is False, "across the back: 20 deg"
    assert policy.new_speaker(None, rad(0)) is True and policy.new_speaker(rad(0), None) is False
    assert face_aim(Aim(0.1, 0.2), face(10.0, tilt=-5.0)).tilt_rad == pytest.approx(0.2 - rad(5))


# ---- the state machine ----------
def view(head: Aim | None = HOME_AIM, holder: str = "", **kw: Any) -> HeadView:
    fields: dict[str, Any] = {
        "head": head,
        "holder": holder,
        "band": HOME,
        "blind": False,
        "blind_until": None,
        "driving": False,
    }
    fields.update(kw)
    return HeadView(**fields)


@dataclass
class Door(FakeGazeDoor):
    """The arbiter as the speaker gaze sees it: an aim that is done moves the head there and
    makes the speaker the holder."""

    answers: deque[GazeAnswer] = field(default_factory=deque)

    def aim(self, aim: Aim, *, speed: Speed, ttl_s: float) -> GazeAnswer:
        self.calls.append(("aim", (aim, speed, ttl_s)))
        answer = self.answers.popleft() if self.answers else GazeAnswer("done", head=aim)
        if answer.status == "done" and self.view is not None:
            self.view = replace(self.view, head=aim, holder=SPEAKER, band=5)
        return answer

    def aims(self) -> list[tuple[Aim, Speed]]:
        return [(c[1][0], c[1][1]) for c in self.calls if c[0] == "aim"]

    def count(self, name: str) -> int:
        return sum(1 for c in self.calls if c[0] == name)


def speech(deg: int, start: float, end: float) -> list[DoaReading]:
    """The array's readings of a voice at array angle ``deg`` from ``start`` to ``end`` (laptop
    seconds), on the board's clock."""
    steps = round((end - start) * 10)
    return [DoaReading(BOARD + start + k / 10, deg, True) for k in range(steps + 1)]


@dataclass
class Rig:
    machine: SpeakerGaze
    door: Door
    doa: FakeDoa
    finder: FakeFaceFinder
    wheels: Any
    face: FakeFace
    said: list[str]


def rig(
    *,
    settings: SpeakerSettings | None = None,
    wheels: Any = None,
    frames: list[FaceFrame | None] | None = None,
    finder: bool = True,
) -> Rig:
    door = Door(view=view())
    doa = FakeDoa()
    faces = FakeFaceFinder(deque(frames or []))
    face_channel = FakeFace()
    said: list[str] = []
    wheels = wheels if wheels is not None else NoWheels()
    machine = SpeakerGaze(
        settings if settings is not None else RULES,
        door=door,
        doa=doa,
        finder=faces if finder else None,
        wheels=wheels,
        face=face_channel,
        reach=REACH,
        report=said.append,
    )
    return Rig(machine, door, doa, faces, wheels, face_channel, said)


def wake(r: Rig, deg: int, *, start: float = 0.0, end: float = 1.2, read: float = 1.6) -> None:
    """The name said from array angle ``deg``, read by Whisper at ``read``; the room's readings
    (no speech, another angle) in between."""
    room = [replace(x, speech=False) for x in speech(30, end + 0.1, read)]
    r.doa.readings += speech(deg, start, end) + room
    r.machine.opened(Heard(start, end, "Пепин"), read)


def run(r: Rig, until: float, start: float) -> None:
    """Tick the machine at every due time up to ``until``."""
    now = start
    for _ in range(1000):
        due = r.machine.tick(now)
        if due is None or due > until:
            return
        now = max(due, now)


def test_the_wake_turns_the_head_to_the_voice_finds_the_face_centres_it_and_smiles_once() -> None:
    """G3: Artem says the name 60 deg to the left; the head turns there, a frame finds his face
    4 deg further left and 3 deg up, the head is centred on it slowly, the face smiles once."""
    r = rig(frames=[frame(face(4.0, tilt=-3.0)), frame(face(1.0)), frame(face(0.5))])
    wake(r, 240)
    ((first, speed),) = r.door.aims()
    assert first.pan_rad == pytest.approx(rad(60)) and first.tilt_rad == pytest.approx(SEEK_TILT)
    assert speed == "saccade" and r.door.calls[0][1][2] == 6.0, "held for its 6 s TTL"
    assert r.machine.phase == "seek" and "+60 deg" in r.said[0]
    r.machine.tick(1.6)  # the first frame at once
    refined, slow = r.door.aims()[-1]
    assert slow == "slow" and refined.pan_rad == pytest.approx(rad(64))
    assert refined.tilt_rad == pytest.approx(SEEK_TILT - rad(3))
    assert r.machine.phase == "track" and r.face.calls == [("express", ("smile", 1.0, 2.0, None))]
    run(r, until=4.0, start=1.6)
    assert r.finder.looks == 8, "three frames a second while the conversation lasts"
    assert len(r.door.aims()) == 2, "1 deg and 0.5 deg: in the dead band"
    assert len(r.face.calls) == 1, "one smile a conversation"
    assert r.door.count("renew") >= 1


def test_without_a_face_the_head_sweeps_the_tilt_then_stays_on_the_voice_without_a_smile() -> None:
    """G4: dark, or behind: three frames 0.4 s apart at the seek tilt, three 10 deg higher (a
    standing face far away), then the voice at the seek tilt, three frames a second."""
    r = rig(frames=[frame() for _ in range(7)] + [frame(face(-3.0))])
    wake(r, 240)
    for t in (1.6, 2.0, 2.4):
        r.machine.tick(t)
    higher, _ = r.door.aims()[-1]
    assert higher == approx_aim(60, SEEK_TILT - rad(10)) and r.machine.phase == "seek"
    for t in (2.8, 3.2, 3.6):
        r.machine.tick(t)
    assert r.machine.phase == "voice" and r.face.calls == [] and "no face" in r.said[-1]
    assert r.door.aims()[-1][0] == approx_aim(60, SEEK_TILT), "back to the seek tilt"
    assert r.machine.next_frame == pytest.approx(3.6 + 1 / 3)
    r.machine.tick(3.6 + 1 / 3)
    assert r.machine.phase == "voice" and r.finder.looks == 7
    r.machine.tick(3.6 + 2 / 3)  # he steps into the light
    assert r.machine.phase == "track" and len(r.face.calls) == 1


def test_the_face_picked_as_the_speaker_s_is_told_to_the_listener() -> None:
    """Person recognition hears every frame's speaker face; a listener that breaks is logged."""
    heard: list[tuple[FaceFrame, Any]] = []
    r = rig(frames=[frame(face(4.0)), frame(), frame(face(1.0))])
    r.machine.on_face = lambda f, s: heard.append((f, s))
    wake(r, 240)
    r.machine.tick(1.6)
    run(r, until=3.7, start=1.6)
    assert [s.pan_deg for _, s in heard] == [4.0, 1.0]  # no face, nothing told
    r.machine.on_face = lambda f, s: 1 / 0
    r.finder.frames.append(frame(face(0.5)))
    run(r, until=4.7, start=3.7)
    assert r.machine.phase == "track"


def test_a_smile_without_a_face_when_asked_for() -> None:
    settings = replace(RULES, seek_frames=1, seek_sweep_deg=0.0, smile_unseen=True)
    r = rig(settings=settings, frames=[None])
    wake(r, 240)
    r.machine.tick(1.6)
    assert r.machine.phase == "voice" and len(r.face.calls) == 1


def test_past_the_comfort_the_head_goes_first_the_wheels_turn_the_rest_and_the_head_recentres() -> (
    None
):
    """A call from 100 deg left: the head is there at once, the wheels are only started (the
    wake returns before they move), every step they report re-centres the head on the person,
    their end once more; no frame is taken while the body turns."""
    turning = rig(settings=replace(RULES, reaim_deg=20.0), wheels=FakeWheels(),
                  frames=[frame(face(0.0))])  # fmt: skip
    wake(turning, 280)
    (start,) = turning.wheels.calls
    assert start == ("start", (WheelMove(turn_deg=pytest.approx(30.0)),))
    (first, speed) = turning.door.aims()[0]
    assert first.pan_rad == pytest.approx(rad(100)) and speed == "saccade", "the head first"
    assert turning.machine.turning is not None and turning.machine.phase == "seek"
    assert turning.machine.tick(1.6) is not None and turning.finder.looks == 0, "no frame yet"
    turning.wheels.move_on(10.0)
    turning.machine.tick(1.8)
    assert turning.door.aims()[-1][0].pan_rad == pytest.approx(rad(90))
    assert turning.door.aims()[-1][1] == "slow" and turning.finder.looks == 0
    turning.wheels.finish()
    turning.machine.tick(2.4)
    recentred = [a.pan_rad for a, _ in turning.door.aims()]
    assert recentred[:3] == pytest.approx([rad(100), rad(90), rad(70)])
    assert turning.machine.voice == pytest.approx(rad(70)), "the person moved with the cart"
    assert turning.machine.turning is None and turning.finder.looks == 1, "the frame after"
    turning.machine.heard(Heard(0.0, 1.2, "Пепин"), 2.5)  # the cascade tells the wake again
    assert turning.wheels.calls == [start], "its readings are from before the turn: ignored"


def test_wheels_that_turn_less_leave_the_head_at_its_reach_and_a_cut_turn_is_finished_later() -> (
    None
):
    stub = rig()
    wake(stub, 280)
    stub.machine.tick(1.6)  # NoWheels answered at once; the answer is taken on the next tick
    assert stub.door.aims()[0][0].pan_rad == pytest.approx(rad(100)), "NoWheels: the head alone"
    assert len(stub.door.aims()) == 1 and "the wheels turned +0 of +30 deg" in stub.said[1]
    behind = rig()
    wake(behind, 0)  # straight behind: past the neck's reach
    assert behind.door.aims()[0][0].pan_rad == pytest.approx(REACH.pan[1])
    cut = rig(wheels=FakeWheels())
    wake(cut, 280)
    cut.wheels.finish(WheelReport(False, 12.0, why="halted: the person speaks", cut=True))
    cut.machine.tick(1.7)
    assert cut.door.aims()[-1][0].pan_rad == pytest.approx(rad(88)) and cut.machine.reorient
    assert "turned +12 of +30 deg (halted: the person speaks)" in cut.said[-1]
    cut.doa.readings += speech(268, 3.0, 4.0)  # he goes on talking from where he stands
    cut.machine.heard(Heard(3.0, 4.0, "ну что"), 4.2)
    assert [c[0] for c in cut.wheels.calls] == ["start", "start"], "the rest of the turn"


def test_a_drive_and_the_close_let_the_wheels_go_and_their_late_word_is_ignored() -> None:
    r = rig(wheels=FakeWheels())
    wake(r, 280)
    r.machine.drive(True, 2.0)
    assert r.wheels.calls[-1] == ("halt", ("a drive starts",)) and r.machine.turning is None
    r.wheels.finish(WheelReport(False, 5.0, why="halted: a drive starts", cut=True))
    r.machine.tick(2.1)
    assert len(r.door.aims()) == 1, "a turn let go: its end moves nothing"
    again = rig(wheels=FakeWheels())
    wake(again, 280)
    again.machine.closed("idle", 3.0)
    assert again.wheels.calls[-1][0] == "halt" and again.door.count("release") == 1


def test_a_voice_already_in_the_picture_and_an_unmeasured_map_only_tilt_the_head_up() -> None:
    near = rig()
    near.door.view = view(head=Aim(rad(50), rad(23.8)))
    wake(near, 240)  # 60 deg: 10 deg from where the head points
    assert near.door.aims()[0][0] == approx_aim(50, SEEK_TILT)
    blind = rig(settings=replace(RULES, doa_map=None))
    wake(blind, 240)
    assert blind.door.aims()[0][0] == Aim(0.0, SEEK_TILT) and "unmeasured" in blind.said[0]


def tracking(frames: list[FaceFrame | None]) -> Rig:
    r = rig(frames=[frame(face(0.0)), *frames])
    wake(r, 240)
    r.machine.tick(1.6)
    assert r.machine.phase == "track"
    return r


def test_tracking_re_aims_past_the_dead_band_slowly_and_a_long_way_at_a_saccade() -> None:
    """G5: he walks sideways talking: the head follows, still when he stands."""
    r = tracking([frame(face(3.0)), frame(face(10.0)), frame(face(-20.0)), frame()])
    aims = len(r.door.aims())
    r.machine.tick(2.6)
    assert len(r.door.aims()) == aims, "3 deg: inside the dead band"
    r.machine.tick(3.6)
    aim, speed = r.door.aims()[-1]
    assert speed == "slow" and aim.pan_rad == pytest.approx(rad(70))
    r.machine.tick(4.6)
    aim, speed = r.door.aims()[-1]
    assert speed == "saccade" and aim.pan_rad == pytest.approx(rad(50))
    r.machine.tick(5.6)
    assert r.machine.phase == "track" and r.machine.aim == aim, "no face: the aim stays"


def test_a_sentence_from_elsewhere_is_looked_at_unless_the_robot_was_speaking() -> None:
    r = tracking([])
    r.doa.readings += speech(230, 5.0, 6.0)  # 50 deg left: the same person, 10 deg off
    r.machine.heard(Heard(5.0, 6.0, "да"), 6.4)
    assert len(r.door.aims()) == 1
    r.machine.speaking(True, 6.5)
    r.doa.readings += speech(120, 7.0, 8.0)  # 60 deg right, over the robot's voice
    r.machine.heard(Heard(7.0, 8.0, "я тоже"), 8.4)
    assert len(r.door.aims()) == 1 and r.machine.counts["over_robot"] == 1
    r.machine.speaking(False, 8.6)
    r.doa.readings += speech(120, 9.0, 10.0)
    r.machine.heard(Heard(9.0, 10.0, "я тоже"), 10.4)
    turned, _ = r.door.aims()[-1]
    assert turned.pan_rad == pytest.approx(rad(-60)) and r.machine.phase == "seek"
    r.machine.heard(Heard(11.0, 12.0, "", noise=True), 12.4)
    assert r.machine.phase == "seek"


def test_a_lost_look_is_asked_for_again_and_waits_behind_the_better_one() -> None:
    """G7: "посмотри налево" took the head: the renewal finds nothing live, the aim is asked
    again and queues; frames wait while another look holds the head."""
    r = tracking([frame(face(0.0))])
    r.door.live = 0
    r.door.answers.append(GazeAnswer("queued", "no answer in 2.0 s: held behind"))
    r.door.view = view(head=Aim(rad(90), 0.0), holder="llm.look", band=2)
    r.machine.tick(3.6)  # the renewal and the frame are due together
    assert r.door.count("renew") == 1 and len(r.door.aims()) == 2
    assert r.machine.aim is not None and "waits: queued" in r.said[-1]
    assert r.finder.looks == 1 and r.machine.counts["frames_skipped"] == 1


def test_frames_wait_for_a_still_head_and_a_gesture_but_not_for_a_breath() -> None:
    r = tracking([frame(face(0.0)), frame(face(0.0))])
    r.door.view = replace(r.door.view, blind=True) if r.door.view else None
    r.machine.tick(2.6)
    assert r.finder.looks == 1 and r.machine.next_frame == pytest.approx(2.7)
    r.door.view = replace(r.door.view, blind=False, overlay=GESTURE) if r.door.view else None
    r.machine.tick(2.7)
    assert r.finder.looks == 1, "a thinking glance pulls the head away: no frame"
    r.door.view = replace(r.door.view, overlay=IDLE_LOOK) if r.door.view else None
    r.machine.tick(3.7)
    assert r.finder.looks == 2 and r.machine.phase == "track"


def test_a_drive_pauses_the_gaze_and_the_next_utterance_orients_again() -> None:
    r = tracking([])
    r.machine.drive(True, 3.0)
    assert r.machine.phase == "paused" and r.machine.tick(3.0) is None
    r.doa.readings += speech(240, 4.0, 5.0)
    r.machine.heard(Heard(4.0, 5.0, "куда"), 5.4)
    assert len(r.door.aims()) == 1, "the drive runs"
    r.machine.drive(False, 9.0)
    r.doa.readings += speech(200, 9.5, 10.5)
    r.machine.heard(Heard(9.5, 10.5, "приехали"), 10.9)
    assert r.door.aims()[-1][0].pan_rad == pytest.approx(rad(20)) and r.machine.phase == "seek"


def test_the_conversation_s_end_lets_the_head_go_and_nothing_runs_after() -> None:
    """G8: "пока": home, frames stop."""
    r = tracking([])
    r.machine.closed("goodbye", 4.0)
    assert r.door.count("release") == 1 and r.machine.phase == "idle"
    assert r.machine.tick(5.0) is None and "conversation closed (goodbye)" in r.said[-1]
    looks = r.finder.looks
    r.machine.heard(Heard(5.0, 6.0, "эй"), 6.4)
    assert r.machine.tick(7.0) is None and r.finder.looks == looks
    r.machine.closed("idle", 8.0)
    assert r.door.count("release") == 1, "nothing held: nothing to release"


def test_a_door_that_does_not_answer_is_asked_again_after_the_renewal_period() -> None:
    r = rig(frames=[frame(face(0.0))])
    r.door.view = None
    wake(r, 240)
    assert r.machine.aim is None and r.machine.phase == "voice" and "does not answer" in r.said[-1]
    assert r.machine.tick(2.0) == pytest.approx(3.6)
    r.door.view = view()
    r.machine.tick(3.6)
    assert r.machine.aim is not None and r.machine.aim.pan_rad == pytest.approx(rad(60))
    assert r.machine.phase == "seek"


def test_no_finder_holds_the_voice_without_frames() -> None:
    r = rig(finder=False)
    wake(r, 240)
    assert r.machine.phase == "voice" and r.machine.next_frame is None
    assert r.machine.tick(1.6) == pytest.approx(3.6), "the renewals go on"


class Walker:
    """The finder seeing one person at ``pan(t)`` (cart frame, degrees) from wherever the door
    says the head points; ``t`` is set before each tick."""

    def __init__(self, door: Door, pan: Callable[[float], float]) -> None:
        self.door, self.pan, self.t, self.looks = door, pan, 0.0, 0

    def look(self) -> FaceFrame | None:
        self.looks += 1
        head = self.door.view.head if self.door.view is not None else None
        held = math.degrees(head.pan_rad) if head is not None else 0.0
        return frame(face(self.pan(self.t) - held))


def minute(pan: Callable[[float], float]) -> dict[str, int]:
    """A minute of tracking at the shipped rate, the person at ``pan(t)``: what it cost."""
    r = rig(settings=SpeakerSettings(doa_map=MAP))
    walker = Walker(r.door, pan)
    r.machine.finder = walker
    wake(r, 240)  # the face at +60
    walker.t = 1.6
    now = r.machine.tick(1.6)  # found: tracking from here
    keys = ("frames", "states", "renewals", "aims")
    before = {k: r.machine.counts[k] for k in keys}
    while now is not None and now <= 61.6 + 1e-6:
        walker.t = now
        now = r.machine.tick(now)
    assert r.machine.phase == "track"
    return {k: r.machine.counts[k] - before[k] for k in keys}


def test_a_minute_of_tracking_at_10_hz_costs_600_state_reads_and_30_renewals() -> None:
    """The budget at the shipped 10 Hz: one state read before each frame, a renewal every 2 s,
    no aim while the person stands still; walking across at 15 deg/s (a person at 1.5 m and
    0.4 m/s) the head is re-aimed past the dead band about every 0.4 s, each aim a renewal too."""
    still = minute(lambda t: 60.0)
    assert still == {"frames": 600, "states": 600, "renewals": 30, "aims": 0}
    walking = minute(lambda t: 60.0 - 15.0 * ((t - 1.6) % 8.0))  # +60 to -60, again
    assert walking["frames"] == 600 and 120 <= walking["aims"] <= 180
    assert walking["renewals"] < still["renewals"]


# ---- whoever speaks: look, verify, return ----------
TODAY_SETTINGS = replace(RULES, doa_map=TODAY.doa_map)


@dataclass(frozen=True, eq=False)
class approx_aim:  # noqa: N801 -- read as a function in the asserts
    """An aim that compares equal to an :class:`Aim` within float noise."""

    pan_deg: float
    tilt_rad: float

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Aim)
            and math.isclose(other.pan_rad, rad(self.pan_deg), abs_tol=1e-9)
            and math.isclose(other.tilt_rad, self.tilt_rad, abs_tol=1e-9)
        )

    def __hash__(self) -> int:
        return hash((self.pan_deg, self.tilt_rad))


def utter(
    r: Rig,
    deg: int,
    start: float,
    end: float,
    text: str = "да",
    *,
    peak: float | None = -15.0,
    named: bool = False,
    noise: bool = False,
    flagged: int | None = None,
) -> float:
    """An utterance from array angle ``deg`` read 0.4 s after it ended (the room's readings in
    between); ``flagged``: only its last that many readings carry the speech flag. When it was
    read."""
    read = end + 0.4
    voice = speech(deg, start, end)
    if flagged is not None:
        voice = [replace(x, speech=k >= len(voice) - flagged) for k, x in enumerate(voice)]
    room = [replace(x, speech=False) for x in speech(30, end + 0.1, read)]
    r.doa.readings += voice + room
    r.machine.heard(Heard(start, end, text, noise, peak, named), read)
    return read


def test_the_quiet_question_from_the_live_run_no_longer_takes_the_head_off_the_face() -> None:
    """2026-10-12: Artem tracked at +25; "Как у тебя дела?" at -30.8 dBFS read 246 deg (-76 by
    the offset map, -63 now) and the head turned away to nobody. A quiet utterance, or one with
    two speech readings, now moves nothing: the face stays tracked."""
    frames = [frame(face(25.0, width=75.0))] + [frame(face(0.0, width=75.0)) for _ in range(30)]
    r = rig(settings=TODAY_SETTINGS, frames=frames)
    wake(r, 193)  # -19 deg: in the picture from home, the head only tilts up
    r.machine.tick(1.6)
    assert r.machine.phase == "track" and r.machine.voice == pytest.approx(rad(25))
    aims = len(r.door.aims())
    read = utter(r, 246, 3.0, 4.5, "Как у тебя дела?", peak=-30.8)
    assert len(r.door.aims()) == aims and "quiet, peak -30.8 dBFS under -28" in r.said[-1]
    read = utter(r, 246, 5.0, 6.5, "Как у тебя дела?", flagged=2)
    assert len(r.door.aims()) == aims and "2 speech readings, under 4" in r.said[-1]
    run(r, until=8.0, start=read)
    assert r.machine.phase == "track" and len(r.door.aims()) == aims
    assert r.machine.counts["sounds_ignored"] == 2


def test_a_second_speaker_is_looked_at_at_once_and_the_first_one_s_place_remembered() -> None:
    """Look, verify, return: someone says a sentence from the right while Artem is tracked at
    +60; the head looks there at once, a frame finds her and she is followed. When Artem
    answers (his voice read 20 deg off), the head goes back to his face's place, tilt and all,
    before a frame confirms him."""
    r = rig(frames=[frame(face(0.0, tilt=5.0))])
    wake(r, 240)
    r.machine.tick(1.6)
    assert r.machine.target is not None
    artem = r.machine.target.aim
    assert artem.pan_rad == pytest.approx(rad(60)) and artem.tilt_rad == pytest.approx(0.0)
    r.finder.frames.append(frame(face(5.0, width=50.0)))
    read = utter(r, 100, 3.0, 4.0, "А мне что расскажешь?")  # -80 deg
    look, speed = r.door.aims()[-1]
    assert look == approx_aim(-80, SEEK_TILT) and speed == "saccade" and r.machine.verifying
    assert "a voice elsewhere" in r.said[-1], "the wheels' answer comes on the next tick"
    r.machine.tick(read)  # the wheels' refusal, then the check's frame: she is there
    assert any("the wheels turned +0 of -20 deg" in line for line in r.said[-3:])
    assert r.machine.phase == "track" and r.machine.voice == pytest.approx(rad(-75))
    assert r.door.aims()[-1][0].pan_rad == pytest.approx(rad(-75))
    assert [t.aim for t in r.machine.targets] == [artem]
    r.finder.frames.append(frame(face(1.0, tilt=0.0)))
    read = utter(r, 220, 5.0, 6.0, "Она шутит")  # +40 by the array, 20 deg off his place
    back, _ = r.door.aims()[-1]
    assert back.pan_rad == pytest.approx(rad(60)) and back.tilt_rad == pytest.approx(0.0)
    r.machine.tick(read)
    assert r.machine.phase == "track" and r.machine.voice == pytest.approx(rad(61))
    assert [round(math.degrees(t.aim.pan_rad)) for t in r.machine.targets] == [-75]


def test_a_check_that_finds_nobody_goes_back_to_the_face_within_verify_s() -> None:
    """A sentence from the TV at -60 while Artem is tracked at +60: one frame there at the seek
    tilt, one 10 deg higher as soon as the head stands still, then back to his face."""
    r = tracking([])
    r.finder.frames.extend([frame(), frame(), frame(face(0.0))])
    read = utter(r, 120, 3.0, 4.0, "в эфире новости")
    assert r.door.aims()[-1][0] == approx_aim(-60, SEEK_TILT) and r.machine.verifying
    r.machine.tick(read)
    assert r.door.aims()[-1][0] == approx_aim(-60, SEEK_TILT - rad(10))
    assert r.machine.next_frame == pytest.approx(read + 0.1), "the next frame once it is still"
    r.machine.tick(read + 0.1)
    back, _ = r.door.aims()[-1]
    assert back == approx_aim(60, SEEK_TILT) and r.machine.phase == "track"
    assert "back to the face at +60 deg" in r.said[-1] and r.finder.looks == 3
    r.machine.tick(read + 0.1 + 1 / 3)
    assert r.machine.phase == "track" and r.machine.voice == pytest.approx(rad(60))


def test_a_check_past_verify_s_goes_back_without_its_second_frame() -> None:
    """The head stays blind after the tilt (a ringing mast): past verify_s the check gives up
    and the head goes back to the face without the higher frame."""
    r = tracking([])
    r.finder.frames.extend([frame(), frame()])
    read = utter(r, 120, 3.0, 4.0, "в эфире новости")
    r.machine.tick(read)  # the first frame; the head tilts higher
    looks = r.finder.looks
    r.door.view = replace(r.door.view, blind=True) if r.door.view else None
    for k in range(1, 8):
        r.machine.tick(read + 0.1 * k)
    assert r.finder.looks == looks and r.machine.verifying
    r.door.view = replace(r.door.view, blind=False) if r.door.view else None
    r.machine.tick(read + 0.8)
    assert r.finder.looks == looks and r.machine.phase == "track"
    assert r.door.aims()[-1][0] == approx_aim(60, SEEK_TILT)


def test_sounds_without_words_turn_the_head_only_when_they_repeat_from_one_place() -> None:
    """A chair, the TV's noise line: one sound from elsewhere moves nothing; two in a row from
    about one place make a check there (one from the person held breaks the run)."""
    r = tracking([])
    aims = len(r.door.aims())
    utter(r, 120, 3.0, 4.0, "", noise=True)  # -60
    utter(r, 240, 5.0, 6.0, "", noise=True)  # from Artem's side: the run is broken
    utter(r, 125, 7.0, 8.0, "", noise=True)  # -55
    assert len(r.door.aims()) == aims and r.machine.phase == "track"
    utter(r, 118, 9.0, 10.0, "", noise=True)  # -62: two in a row
    look, _ = r.door.aims()[-1]
    assert look.pan_rad == pytest.approx(rad(-58.5)) and r.machine.verifying
    assert "2 sounds from -58 deg" in r.said[-1]


def test_the_robot_s_name_turns_the_head_whatever_the_gates_and_seeks_in_full() -> None:
    """The wife, far and quiet, says "Привет, Пепин" from -80 while Artem is tracked: the name
    wins over the level and the speech readings; a full seek (no return) finds her 30 px face,
    aimed at."""
    r = tracking([])
    r.finder.frames.extend([frame(), frame(face(3.0, width=30.0))])
    read = utter(r, 100, 3.0, 4.0, "Привет, Пепин", peak=-31.0, named=True, flagged=3)
    assert r.door.aims()[-1][0] == approx_aim(-80, SEEK_TILT) and not r.machine.verifying
    assert "the name" in r.said[-1]
    r.machine.tick(read)
    r.machine.tick(read + 0.4)
    assert r.machine.phase == "track" and r.machine.voice == pytest.approx(rad(-77))
    utter(r, 245, 6.0, 7.0, "Пепин, а я?", named=True)  # Artem, by name, 5 deg off his place
    assert r.door.aims()[-1][0].pan_rad == pytest.approx(rad(60)), "his remembered place"


def test_the_person_held_speaking_quietly_or_by_name_moves_nothing_and_prints_nothing() -> None:
    r = tracking([])
    aims, lines = len(r.door.aims()), len(r.said)
    utter(r, 235, 3.0, 4.0, "тихо", peak=-35.0)  # +55: 5 deg off his face, a whisper
    utter(r, 245, 5.0, 6.0, "Пепин, смотри", named=True)
    assert len(r.door.aims()) == aims and len(r.said) == lines and r.machine.phase == "track"


def test_a_face_lost_for_face_lost_s_lets_the_next_voice_turn_the_head_for_good() -> None:
    r = tracking([frame() for _ in range(10)])
    run(r, until=3.7, start=1.6)
    assert r.machine.phase == "voice" and "the face is lost (2.0 s unseen)" in r.said[-1]
    assert [round(math.degrees(t.aim.pan_rad)) for t in r.machine.targets] == [60]
    utter(r, 120, 4.0, 5.0, "я здесь")
    assert r.door.aims()[-1][0] == approx_aim(-60, SEEK_TILT) and r.machine.phase == "seek"
    assert not r.machine.verifying, "nobody to go back to: a full seek"


def test_a_voice_from_behind_turns_the_head_as_far_as_it_reaches() -> None:
    """The live run's behind-left wake read 82 deg: +108 by the interpolated map (+88 by the
    offset; the face was found at +111), within the neck's reach. One from behind-left (+170)
    and the name from behind-right (315 deg, -141) turn the head to its reach, the wheels
    refusing."""
    r = rig(settings=TODAY_SETTINGS)
    wake(r, 82)
    assert r.door.aims()[0][0].pan_rad == pytest.approx(rad(107.6), abs=1e-3)
    behind = rig(settings=TODAY_SETTINGS)
    wake(behind, 357)
    assert behind.door.aims()[0][0].pan_rad == pytest.approx(REACH.pan[1])
    behind.machine.tick(1.6)  # the stub's refusal is taken on the next tick
    assert "the wheels turned +0 of +30 deg" in behind.said[1]
    r = rig(settings=TODAY_SETTINGS, frames=[frame(face(3.0))])
    wake(r, 82)
    r.machine.tick(1.6)
    r.finder.frames.append(frame(face(-20.0)))
    read = utter(r, 315, 3.0, 4.0, "Эй, пипин, привет", named=True)
    assert r.door.aims()[-1][0].pan_rad == pytest.approx(rad(-141.2), abs=1e-3)
    r.machine.tick(read)
    assert r.door.aims()[-1][0].pan_rad == pytest.approx(REACH.pan[0]), "-161: past the reach"
    assert r.machine.voice == pytest.approx(rad(-161.2), abs=1e-3)


def test_a_walking_face_is_led_by_its_speed_and_the_lead_is_capped() -> None:
    """He walks left at 15 deg/s (5 deg a frame at 3 Hz): from the second sighting the head aims
    where he will be lead_s (0.3 s) later, 4.5 deg ahead of where he was seen; at 60 deg/s the
    lead stops at lead_max_deg; sightings over half a second apart give no speed."""
    leads: list[dict[str, Any]] = []
    r = tracking([frame(face(5.0))])
    r.machine.events = lambda kind, now, **f: leads.append(f) if kind == "gaze_reaim" else None
    r.machine.tick(1.6 + 1 / 3)
    aim, speed = r.door.aims()[-1]
    assert aim.pan_rad == pytest.approx(rad(69.5)) and speed == "slow"
    assert leads[-1]["lead_deg"] == 4.5
    fast = tracking([frame(face(20.0))])
    fast.machine.tick(1.6 + 1 / 3)
    aim, speed = fast.door.aims()[-1]
    assert aim.pan_rad == pytest.approx(rad(90)) and speed == "saccade", "80 seen, 10 ahead"
    late = tracking([frame(face(5.0))])
    late.machine.tick(2.6)  # a second after the last sighting: no speed to trust
    assert late.door.aims()[-1][0].pan_rad == pytest.approx(rad(60)), "5 deg: the dead band"


def test_a_far_face_found_by_the_sweep_is_aimed_at_and_smiled_at_only_once_clear() -> None:
    """At 3 m a face is 26-30 px, under tracking's 40: the seek takes faces from 28 px and tilts
    10 deg higher before giving up; a small face is aimed at without a smile, kept while it
    stays where the person is, and smiled at once it is 40 px or more."""
    far = frame(face(-4.0, tilt=-3.0, width=30.0))
    r = rig(frames=[frame(), frame(), frame(), far, frame(face(0.0, width=34.0))])
    r.finder.frames.append(frame(face(0.0, width=45.0)))
    wake(r, 240)
    for t in (1.6, 2.0, 2.4, 2.8):
        r.machine.tick(t)
    aim, _ = r.door.aims()[-1]
    assert aim.pan_rad == pytest.approx(rad(56)) and aim.tilt_rad == pytest.approx(rad(-18))
    assert r.machine.phase == "track" and r.face.calls == [] and "no smile yet" in r.said[-1]
    r.machine.tick(3.15)
    assert r.machine.phase == "track" and r.face.calls == [], "34 px where he is: kept"
    r.machine.tick(3.5)
    assert len(r.face.calls) == 1, "45 px: a clear face, a smile"


def test_small_faces_count_while_seeking_or_near_the_person_held_and_never_under_the_floor() -> (
    None
):
    policy = SpeakerPolicy(match_deg=20.0, doa_jump_deg=35.0, min_px=40.0, near_min_px=28.0)
    head = Aim(0.0, 0.0)
    elsewhere = frame(face(-30.0, width=30.0))
    assert policy.pick(rad(10), elsewhere, head) is None, "a small face elsewhere: not followed"
    assert policy.pick(rad(10), elsewhere, head, seeking=True) == elsewhere.faces[0]
    near = frame(face(5.0, width=30.0))
    assert policy.pick(rad(10), near, head) == near.faces[0], "the person held, far away"
    assert policy.pick(rad(10), frame(face(5.0, width=20.0)), head, seeking=True) is None
    targets = [Target(Aim(rad(25), 0.0), None, 1.0), Target(Aim(rad(-60), 0.1), None, 2.0)]
    assert policy.recall(targets, rad(-40), within_rad=rad(35)) == targets[1]
    assert policy.recall(targets, rad(100), within_rad=rad(35)) is None


def test_every_decision_is_an_event_for_the_conversation_s_log() -> None:
    """The live run's quiet question replayed: the wake, the aims and why, the face, the
    utterance's verdict with its level, the release; a log that breaks stops nothing."""
    events: list[tuple[str, float, dict[str, Any]]] = []
    r = rig(settings=TODAY_SETTINGS, frames=[frame(face(25.0, tilt=-15.0, width=75.0))])
    r.machine.events = lambda kind, now, **f: events.append((kind, now, f))
    wake(r, 193)
    r.machine.tick(1.6)
    read = utter(r, 246, 3.0, 4.5, "Как у тебя дела?", peak=-30.8)
    r.machine.closed("idle", 9.0)
    kinds = [(k, t) for k, t, _ in events]
    assert kinds == [
        ("gaze_wake", 1.6),
        ("gaze_reaim", 1.6),
        ("gaze_reaim", 1.6),
        ("gaze_face", 1.6),
        ("gaze_voice", read),
        ("gaze_release", 9.0),
    ]
    woke, first, onto, found, voice, released = (f for _, _, f in events)
    assert woke["bearing_deg"] == -19.2 and woke["doa_deg"] == 193.0 and woke["speech"] == 10
    assert first == {"why": "wake", "pan_deg": 0.0, "tilt_deg": -5.0, "speed": "saccade",
                     "status": "done"}  # fmt: skip
    assert onto["why"] == "onto the face" and onto["px"] == 75 and onto["pan_deg"] == 25.0
    assert found["px"] == 75 and found["tilt_deg"] == -20.0 and found["small"] is False
    assert voice["bearing_deg"] == -63.3 and voice["peak_db"] == -30.8
    assert voice["face_deg"] == 25.0 and voice["verdict"].startswith("quiet")
    assert voice["text"] == "Как у тебя дела?" and released["why"].startswith("the conversation")
    broken = rig(frames=[frame(face(0.0))])
    broken.machine.events = lambda kind, now, **f: 1 / 0
    wake(broken, 240)
    broken.machine.tick(1.6)
    assert broken.machine.phase == "track"


def test_the_observer_hooks_return_at_once_while_the_gaze_waits_on_the_door() -> None:
    """The gaze never delays a reply: with the worker's thread held in a door call (an aim is
    answered only when the head settles), every hook the loop calls is one queue put."""
    entered, gate = threading.Event(), threading.Event()

    @dataclass
    class SlowDoor(Door):
        def aim(self, aim: Aim, *, speed: Speed, ttl_s: float) -> GazeAnswer:
            entered.set()
            gate.wait(5.0)
            return super().aim(aim, speed=speed, ttl_s=ttl_s)

    r = rig()
    r.machine.door = SlowDoor(view=view())
    worker = SpeakerGazeWorker(
        lambda settings: r.machine, on=True, load=lambda: SpeakerSettings(doa_map=MAP)
    )
    r.doa.readings += speech(240, 0.0, 1.0)
    worker.opened(Heard(-1.0, 0.0, "Пепин"))
    assert entered.wait(2.0), "the worker is in the door call"
    hooks: list[Callable[[], None]] = [
        lambda: worker.speech_started(1.0),
        lambda: worker.speech_ended(Heard(1.0, 2.0, "да")),
        lambda: worker.part_queued(1, 0, None),  # type: ignore[arg-type]
        lambda: worker.part_spoken(1, 0),
        lambda: worker.reply_done(1),
        lambda: worker.reply_cut(1),
        lambda: worker.moment("slow_tool"),
        lambda: worker.drive(False),
    ]
    longest = 0.0
    for _ in range(50):
        for hook in hooks:
            t0 = time.perf_counter()
            hook()
            longest = max(longest, time.perf_counter() - t0)
    assert longest < 0.02, f"a hook took {longest * 1000:.1f} ms"
    gate.set()
    worker.close()


# ---- the voice first ----------
def speaking_rig() -> Rig:
    """Tracking at the shipped rates: Artem at +60, a frame every 0.1 s."""
    r = rig(settings=SpeakerSettings(doa_map=MAP), frames=[frame(face(0.0)) for _ in range(100)])
    wake(r, 240)
    r.machine.tick(1.6)
    assert r.machine.phase == "track" and r.machine.next_frame == pytest.approx(1.7)
    return r


def test_no_frame_from_a_reply_s_first_part_until_its_first_chunk_then_one_a_second() -> None:
    """The voice's first chunk shares the Mac with the gaze: from the first part queued until
    the first audio no frame is taken (renewals go on); then frames come once a second while the
    robot speaks, and ten a second again the moment it stops."""
    r = speaking_rig()
    r.machine.speaking(True, 1.65)
    looks = r.finder.looks
    assert r.machine.tick(1.7) == pytest.approx(3.6), "nothing due but the renewal"
    for t in (1.8, 2.0, 2.5):
        r.machine.tick(t)
    assert r.finder.looks == looks and r.machine.voice_hold
    r.machine.voiced(2.55)
    assert r.machine.tick(2.55) == pytest.approx(3.55)
    assert r.finder.looks == looks + 1, "the frame that waited, at once"
    for t in (2.65, 3.0, 3.5):
        r.machine.tick(t)
    assert r.finder.looks == looks + 1, "one a second while the robot speaks"
    r.machine.tick(3.55)
    r.machine.tick(3.6)
    assert r.finder.looks == looks + 2 and r.door.count("renew") >= 1
    r.machine.speaking(False, 3.7)
    assert r.machine.next_frame == pytest.approx(3.8), "listening: ten a second at once"
    r.machine.speaking(True, 5.0)
    assert r.machine.voice_hold
    r.machine.speaking(False, 5.5)
    assert not r.machine.voice_hold, "a reply cut before its first chunk ends the hold"
    assert r.machine.counts["voice_holds"] == 2


def test_the_hold_is_a_switch_and_the_robot_s_later_parts_hold_nothing() -> None:
    r = speaking_rig()
    r.machine.configure(replace(r.machine.settings, hold_for_voice=False))
    r.machine.speaking(True, 1.65)
    looks = r.finder.looks
    r.machine.tick(1.75)
    assert r.finder.looks == looks + 1 and not r.machine.voice_hold
    on = speaking_rig()
    on.machine.speaking(True, 1.65)
    on.machine.voiced(1.66)
    on.machine.speaking(True, 1.8)  # the reply's second part
    assert not on.machine.voice_hold


def test_an_aim_under_aim_min_deg_from_the_last_one_is_never_written() -> None:
    """With the dead band set under it, a correction of 0.8 deg writes nothing and one of
    1.5 deg an aim; at the shipped 6 deg dead band the floor never binds."""
    settings = replace(RULES, deadband_deg=0.5, aim_min_deg=1.0)
    r = rig(settings=settings, frames=[frame(face(0.0)), frame(face(0.8)), frame(face(1.5))])
    wake(r, 240)
    r.machine.tick(1.6)
    aims = len(r.door.aims())
    r.machine.tick(2.6)
    assert len(r.door.aims()) == aims
    r.machine.tick(3.6)
    assert len(r.door.aims()) == aims + 1


def test_the_detector_runs_on_the_neural_engine_as_shipped(tmp_path: Path) -> None:
    """config/expression.json's finder.compute_units: CPUAndNeuralEngine (never the GPU the
    voice uses), CoreML's NeuralNetwork model (the Neural Engine refuses the ML Program) and a
    quiet onnxruntime pool."""
    from pepin.expression.finder import finder_device
    from pepin.face_sensor.backends import COREML, providers, session_options

    assert finder_device(REPO / "config/expression.json") == "ane"
    path = tmp_path / "expression.json"
    path.write_text('{"finder": {"compute_units": "ALL"}}')
    assert finder_device(path) == "coreml"
    path.write_text("{}")
    assert finder_device(path) == "ane"
    path.write_text('{"finder": {"compute_units": "GPU"}}')
    with pytest.raises(ValueError, match="compute_units"):
        finder_device(path)
    names, options = providers("ane", tmp_path / "cache")
    if names[0] == COREML:
        assert options[0]["MLComputeUnits"] == "CPUAndNeuralEngine"
        assert options[0]["ModelFormat"] == "NeuralNetwork"
    assert session_options("ane").intra_op_num_threads == 1
    assert session_options("coreml") is None


# ---- the knobs and the switch ----------
def test_the_shipped_knobs_load_off_and_measured_on_the_four_readings() -> None:
    settings = SpeakerSettings.load(REPO / "config/expression.json")
    assert settings.doa_map == TODAY.doa_map and settings.ttl_s == 6.0 and settings.renew_s == 2.0
    assert settings.body.max_turn_deg == 0.0 and settings.min_speech == 2  # 0 until the live test
    assert settings.refine_deg == 2.0 and settings.smile_unseen is False
    assert (settings.track_hz, settings.lead_max_deg, settings.seek_sweep_deg) == (10.0, 10.0, 10.0)
    assert (settings.track_speaking_hz, settings.hold_for_voice) == (1.0, True)
    assert (settings.lead_s, settings.aim_min_deg) == (0.3, 1.0)
    assert settings == replace(SpeakerSettings(), doa_map=settings.doa_map, body=settings.body)
    assert (settings.reaim_deg, settings.verify_s, settings.reaim_votes) == (35.0, 0.7, 2)
    assert (settings.face_lost_s, settings.doa_min_speech, settings.doa_min_dbfs) == (2.0, 4, -28.0)
    assert (settings.seek_min_face_px, settings.track_min_face_px) == (28.0, 40.0)
    assert settings.remember_targets == 3
    assert flag_default(REPO / "config/expression.json") is False
    bad_knobs = ({"track_hz": 0}, {"reaim_votes": 0}, {"lead_max_deg": -1},
                 {"track_speaking_hz": 0})  # fmt: skip
    for bad in bad_knobs:
        with pytest.raises(ValueError, match="speaker"):
            SpeakerSettings.from_config({"speaker": bad})
    measured = SpeakerSettings.from_config({"speaker": {"doa": {"front_deg": 400, "sign": -1}}})
    assert measured.doa_map == DoaMap(40.0, -1), "the front alone: the offset and the sign"
    part = {"front_deg": 170, "sign": -1, "readings": {"front": 170, "left": 106}}
    assert SpeakerSettings.from_config({"speaker": {"doa": part}}).doa_map == DoaMap(170.0, -1)
    unmeasured = SpeakerSettings.from_config({"speaker": {"doa": {"front_deg": None}}})
    assert unmeasured.doa_map is None
    with pytest.raises(ValueError, match="sign"):
        SpeakerSettings.from_config({"speaker": {"doa": {"front_deg": 0, "sign": 2}}})
    lost = {"readings": {"front": 180, "left": 220, "behind": 0, "right": 90}}
    with pytest.raises(ValueError, match="readings"):
        SpeakerSettings.from_config({"speaker": {"doa": lost}})
    with pytest.raises(ValueError, match="renew_s"):
        SpeakerSettings.from_config({"speaker": {"ttl_s": 2.0, "renew_s": 2.0}})


def test_the_worker_runs_the_machine_on_its_thread_and_sleeps_between_conversations() -> None:
    r = rig(frames=[frame(face(0.0))])
    built: list[SpeakerSettings] = []

    def build(settings: SpeakerSettings) -> SpeakerGaze:
        built.append(settings)
        return r.machine

    worker = SpeakerGazeWorker(build, on=False, load=lambda: SpeakerSettings(doa_map=MAP))
    assert worker.flag == SPEAKER_GAZE and not worker.on
    worker.opened(Heard(0.0, 1.0, "Пепин"))
    assert worker.machine is None, "off: no thread, no machine"
    assert worker.switch(True) is None and worker.on
    assert worker.idle.wait(2.0) and built and worker.machine is r.machine
    r.doa.readings += speech(240, 0.0, 1.0)
    worker.opened(Heard(-1.0, 0.0, "Пепин"))
    worker.part_queued(1, 0, None)  # type: ignore[arg-type]
    worker.reply_done(1)
    worker.drive(True)
    assert worker.idle.wait(2.0), "a drive pauses it: nothing due"
    assert r.door.count("aim") == 1 and r.machine.phase == "paused"
    assert r.machine.spoke and r.machine.spoke[-1][1] is not None
    assert worker.switch(False) is None and not worker.on
    worker.close()
    assert r.door.count("release") == 1 and r.machine.phase == "idle"


def test_the_switch_is_refused_when_the_knobs_cannot_be_read() -> None:
    def broken() -> SpeakerSettings:
        raise ValueError("speaker: renew_s must be shorter than ttl_s")

    worker = SpeakerGazeWorker(lambda s: None, on=False, load=broken)  # type: ignore[arg-type, return-value]
    assert "renew_s" in str(worker.switch(True)) and not worker.on


def test_a_wheel_report_names_what_the_stub_refuses() -> None:
    reports: list[WheelReport] = []
    NoWheels().start(WheelMove(turn_deg=10.0), reports.append)
    assert reports == [
        WheelReport(False, why="the wheels make no expressive motion here: no robot")
    ]
