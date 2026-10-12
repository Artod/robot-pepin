"""The speaker gaze (pepin.expression doa, plan, policy, tracker): the voice's bearing, the head
and the wheels' split, whom to look at, and the state machine on a fake clock against the fakes
of every seam (pepin.expression.fakes)."""

from __future__ import annotations

import math
from collections import deque
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
from pepin.expression.speaker import DoaMap, FaceFrame, FaceSighting, GazeAnswer, HeadView
from pepin.expression.tracker import SpeakerGaze, SpeakerGazeWorker, SpeakerSettings, flag_default
from pepin.gaze import HOME, Aim, Reach, Speed
from pepin.neck import NeckConfig

REPO = Path(__file__).resolve().parents[2]
REACH = Reach.of(NeckConfig.from_json(REPO / "config/neck.json"))
BOARD = 5000.0  # the board's clock runs this far ahead of the laptop's in these tests
MAP = DoaMap(front_deg=180.0, sign=1)  # the array reads 180 straight ahead, growing leftward
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


def test_the_calibration_gives_the_front_and_the_sign_and_refuses_a_skewed_circle() -> None:
    left_up = fit_doa_map(front=180.0, left=272.0, back=355.0, right=88.0)
    assert left_up.doa_map == DoaMap(180.0, 1) and left_up.refusal is None
    assert left_up.misses_deg[1] == pytest.approx(2.0)
    right_up = fit_doa_map(front=10.0, left=280.0, back=190.0, right=100.0)
    assert right_up.doa_map == DoaMap(10.0, -1) and right_up.refusal is None
    skewed = fit_doa_map(front=180.0, left=230.0, back=0.0, right=90.0)
    assert skewed.refusal is not None and "40 deg off" in skewed.refusal
    assert "REFUSED" in skewed.text() and "front_deg 180, sign +1" in left_up.text()


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
    """The wheels' steps: each turn answered from ``blocked`` (a stall short of the step)."""

    def __init__(self, blocked: list[float] | None = None) -> None:
        self.blocked = deque(blocked or [])
        self.calls: list[tuple[str, float]] = []

    def turn(self, deg: float) -> Step:
        self.calls.append(("turn", deg))
        if self.blocked:
            return Step(math.copysign(self.blocked.popleft(), deg), blocked="stalled")
        return Step(deg)

    def back_off(self, metres: float) -> float:
        self.calls.append(("back", metres))
        return metres


def test_a_turn_goes_in_steps_backs_off_on_a_stall_and_gives_up_within_its_bounds() -> None:
    steps = Steps()
    done = BoundedTurn(steps).move(WheelMove(turn_deg=25.0))
    assert done.done and done.turned_deg == pytest.approx(25.0) and done.attempts == 1
    assert steps.calls == [("turn", 10.0), ("turn", 10.0), ("turn", 5.0)]
    steps = Steps(blocked=[4.0])
    retried = BoundedTurn(steps).move(WheelMove(turn_deg=-20.0))
    assert retried.done and retried.backed_m == pytest.approx(0.03) and retried.attempts == 2
    steps = Steps(blocked=[1.0, 1.0, 1.0, 1.0])
    given_up = BoundedTurn(steps).move(WheelMove(turn_deg=20.0))
    assert not given_up.done and given_up.attempts == 3 and given_up.turned_deg == 3.0
    assert given_up.backed_m == pytest.approx(0.06) and given_up.why == "gave up: stalled"
    assert "past 30" in BoundedTurn(Steps()).move(WheelMove(turn_deg=35.0)).why
    assert "rock" in BoundedTurn(Steps()).move(WheelMove(rock_m=0.01, rocks=2)).why
    tight = BoundedTurn(Steps(blocked=[1.0, 1.0]), BodyLimits(attempts=5, max_backoff_m=0.03))
    assert tight.move(WheelMove(turn_deg=10.0)).backed_m == pytest.approx(0.03)


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
        settings if settings is not None else SpeakerSettings(doa_map=MAP),
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
    assert r.finder.looks == 3 and len(r.door.aims()) == 2, "1 deg and 0.5 deg: in the dead band"
    assert len(r.face.calls) == 1, "one smile a conversation"
    assert r.door.count("renew") >= 1


def test_without_a_face_the_head_stays_on_the_voice_and_does_not_smile() -> None:
    """G4: dark, or behind: three frames 0.4 s apart, then the voice, a frame a second."""
    r = rig(frames=[frame(), frame(), frame(), frame(), frame(face(-2.0))])
    wake(r, 240)
    r.machine.tick(1.6)
    assert r.machine.next_frame == pytest.approx(2.0)
    r.machine.tick(2.0)
    r.machine.tick(2.4)
    assert r.machine.phase == "voice" and r.face.calls == [] and "no face" in r.said[-1]
    assert r.machine.next_frame == pytest.approx(3.4)
    r.machine.tick(3.4)
    assert r.machine.phase == "voice" and r.machine.aim is not None
    assert r.machine.aim.pan_rad == pytest.approx(rad(60))
    r.machine.tick(4.4)  # he steps into the light
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
    r = rig(settings=SpeakerSettings(doa_map=MAP, seek_frames=1, smile_unseen=True), frames=[None])
    wake(r, 240)
    r.machine.tick(1.6)
    assert r.machine.phase == "voice" and len(r.face.calls) == 1


def test_past_the_comfort_the_wheels_turn_the_rest_and_the_stub_leaves_it_to_the_head() -> None:
    turning = rig(settings=SpeakerSettings(doa_map=MAP, doa_jump_deg=20.0), wheels=FakeWheels())
    wake(turning, 280)  # 100 deg left
    (move,) = [c[1][0] for c in turning.wheels.calls]
    assert move == WheelMove(turn_deg=pytest.approx(30.0))
    assert turning.door.aims()[0][0].pan_rad == pytest.approx(rad(70))
    assert turning.machine.voice == pytest.approx(rad(70)), "the person moved with the cart"
    turning.machine.heard(Heard(0.0, 1.2, "Пепин"), 1.7)  # the cascade tells the wake again
    assert len(turning.door.aims()) == 1, "its readings are from before the turn: ignored"
    stub = rig()
    wake(stub, 280)
    assert stub.door.aims()[0][0].pan_rad == pytest.approx(rad(100)), "NoWheels: the head alone"
    assert "the wheels turned +0 of +30 deg" in stub.said[1]
    behind = rig()
    wake(behind, 0)  # straight behind: past the neck's reach
    assert behind.door.aims()[0][0].pan_rad == pytest.approx(REACH.pan[1])


def test_a_voice_already_in_the_picture_and_an_unmeasured_map_only_tilt_the_head_up() -> None:
    near = rig()
    near.door.view = view(head=Aim(rad(50), rad(23.8)))
    wake(near, 240)  # 60 deg: 10 deg from where the head points
    assert near.door.aims()[0][0] == Aim(rad(50), SEEK_TILT)
    blind = rig(settings=SpeakerSettings(doa_map=None))
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


def test_a_voice_far_from_the_person_turns_the_head_unless_the_robot_was_speaking() -> None:
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


def test_a_minute_of_tracking_costs_sixty_frames_and_ninety_calls_to_the_door() -> None:
    """The budget: a frame a second (one state read before each), a renewal every 2 s, no
    aim while the person stands still."""
    r = tracking([frame(face(1.0)) for _ in range(70)])
    before = dict(r.machine.counts)
    run(r, until=61.6, start=1.6)
    spent = {k: r.machine.counts[k] - before.get(k, 0) for k in ("frames", "states", "renewals")}
    assert spent == {"frames": 60, "states": 60, "renewals": 30}
    assert r.machine.counts["aims"] == before["aims"], "still: nothing re-aimed"


# ---- the knobs and the switch ----------
def test_the_shipped_knobs_load_off_and_unmeasured() -> None:
    settings = SpeakerSettings.load(REPO / "config/expression.json")
    assert settings.doa_map is None and settings.ttl_s == 6.0 and settings.renew_s == 2.0
    assert settings.body.max_turn_deg == 30.0 and settings.min_speech == 2
    assert settings.refine_deg == 2.0 and settings.smile_unseen is False
    assert flag_default(REPO / "config/expression.json") is False
    measured = SpeakerSettings.from_config({"speaker": {"doa": {"front_deg": 400, "sign": -1}}})
    assert measured.doa_map == DoaMap(40.0, -1)
    with pytest.raises(ValueError, match="sign"):
        SpeakerSettings.from_config({"speaker": {"doa": {"front_deg": 0, "sign": 2}}})
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
    assert NoWheels().move(WheelMove(turn_deg=10.0)) == WheelReport(
        False, why="the wheels make no expressive motion: their executor is not built"
    )
