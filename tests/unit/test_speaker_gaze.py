"""The speaker gaze (pepin.expression doa, plan, policy, tracks, tracker): the voice's bearing,
the head and the wheels' split, the people tracked, whom to look at, and the engine on a fake
clock against the fakes of every seam (pepin.expression.fakes): a room of people the head camera
sees from wherever the arbiter says the head points, and the array's readings as the person
speaks. Every case the live sessions of 2026-10-12 hit is one scenario here."""
# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

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
    recent_voice,
    robot_readings,
    utterance_readings,
    utterance_voice,
    voice_of,
    wrap_deg,
)
from pepin.expression.executors import NoWheels, WheelMove, WheelReport
from pepin.expression.fakes import (
    FakeDoa,
    FakeFace,
    FakeFaceFinder,
    FakeGazeDoor,
    FakePerson,
    FakeScene,
    FakeWheels,
)
from pepin.expression.head import GESTURE, IDLE_LOOK, SPEAKER
from pepin.expression.plan import BodyLimits, BoundedTurn, Step, head_after, split, sweep
from pepin.expression.policy import decide, doubt, face_aim
from pepin.expression.speaker import (
    BodySighting,
    DoaMap,
    Ease,
    FaceFrame,
    FaceSighting,
    GazeAnswer,
    HeadView,
)
from pepin.expression.tracker import SpeakerGaze, SpeakerGazeWorker, SpeakerSettings, flag_default
from pepin.expression.tracks import Sighting, Track, Tracks, sightings
from pepin.gaze import HOME, Aim, Reach, Speed
from pepin.neck import NeckConfig

REPO = Path(__file__).resolve().parents[2]
REACH = Reach.of(NeckConfig.from_json(REPO / "config/neck.json"))
BOARD = 5000.0  # the board's clock runs this far ahead of the laptop's in these tests
MAP = DoaMap(front_deg=180.0, sign=1)  # the array reads 180 straight ahead, growing leftward
RULES = SpeakerSettings(doa_map=MAP)  # the shipped knobs, the simple map
TILT = math.radians(3.0)  # the tilt a voice is looked at before anyone is seen
HOME_AIM = Aim(0.0, math.radians(23.8))
FRAME_S = 0.1  # a frame's wall time on the robot: the snapshot, the detector, the blind settle


def rad(deg: float) -> float:
    return math.radians(deg)


def deg(rad_: float) -> float:
    return math.degrees(rad_)


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
    clean = utterance_voice(doa, heard, 3.3, window_s=1.0, doa_map=MAP, min_speech=2)
    assert clean.voice.bearing_rad == pytest.approx(rad(60.0)) and clean.voice.robot == 0
    assert len(clean.readings) == 21, "every reading of the utterance, for the log"


def test_the_robot_s_own_voice_is_cut_out_and_an_overlapping_utterance_keeps_its_clean_part() -> (
    None
):
    """2026-10-12, cascade_150557: the head stayed at -94 deg for 90 s while Artem talked at
    -15, every utterance overlapping a reply and thrown away whole. Now the readings within the
    robot's span (and 0.3 s of echo after) are the robot's; the rest are his."""
    robot = speech(300, 2.0, 6.0)  # the robot's voice reads -120
    him = speech(165, 6.0, 8.0)  # Artem at -15
    doa = FakeDoa(sorted(robot + him, key=lambda r: r.stamp_s))
    mask = robot_readings(doa.readings, [(2.0, 6.0)], BOARD)
    assert sum(mask) == len(robot) + 4, "the span and its 0.3 s of echo, both ends in"
    heard = Heard(5.0, 8.0, "я тут, слышишь")
    clean = utterance_voice(doa, heard, 8.0, window_s=1.0, doa_map=MAP, min_speech=2,
                            spans=[(2.0, 6.0)])  # fmt: skip
    assert clean.voice.bearing_rad == pytest.approx(rad(-15.0)) and clean.voice.robot == 15
    drowned = utterance_voice(doa, Heard(3.0, 5.5, "да"), 5.5, window_s=1.0, doa_map=MAP,
                              min_speech=2, spans=[(2.0, None)])  # fmt: skip
    assert drowned.voice.bearing_rad is None and drowned.voice.robot > 0
    live = recent_voice(doa, 8.0, window_s=0.5, doa_map=MAP, min_speech=3, spans=[(2.0, 6.0)])
    assert live.bearing_rad == pytest.approx(rad(-15.0)) and live.speech == 6


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
    assert bearing_deg(82.0, m) == pytest.approx(107.6, abs=0.1)
    assert bearing_deg(82.0, DoaMap(170.0, -1)) == pytest.approx(88.0)
    assert bearing_deg(246.0, m) == pytest.approx(-63.3, abs=0.1), "the quiet one: -76 before"


# ---- the head and the wheels ----------
def test_the_head_alone_within_its_comfort_and_the_wheels_for_the_rest() -> None:
    near = split(rad(45), TILT, comfort_rad=rad(60), max_turn_rad=rad(30))
    assert near.body_rad == 0.0 and near.head.pan_rad == pytest.approx(rad(45))
    far = split(rad(80), TILT, comfort_rad=rad(60), max_turn_rad=rad(30))
    assert far.body_rad == pytest.approx(rad(20)) and far.head.pan_rad == pytest.approx(rad(60))
    back = split(rad(-150), TILT, comfort_rad=rad(60), max_turn_rad=rad(30))
    assert back.body_rad == pytest.approx(rad(-30)) and back.head.pan_rad == pytest.approx(
        rad(-120)
    )
    assert head_after(rad(170), 0.0, TILT, REACH).pan_rad == pytest.approx(REACH.pan[1])


def test_the_turn_eases_in_and_out_like_a_head_and_a_seek_looks_outward_first() -> None:
    """The servo's trapezoid at 650 deg/s^2 under 200 deg/s: a triangle of velocity, the time
    growing as the root of the angle (a human head's coarse turn: 20 deg in about a third of a
    second, 60 in about 0.6); the adjustment's 400 deg/s^2 for the small moves; past
    speed^2/acc a plateau."""
    turn, adjust = RULES.ease("turn"), RULES.ease("adjust")
    assert turn is not None and adjust is not None and turn.name == "turn"
    assert turn.seconds(20.0) == pytest.approx(0.351, abs=1e-3)
    assert turn.seconds(60.0) == pytest.approx(0.608, abs=1e-3)
    assert adjust.seconds(6.0) == pytest.approx(0.245, abs=1e-3)
    assert adjust.seconds(20.0) == pytest.approx(0.447, abs=1e-3)
    assert Ease(200.0, 650.0).seconds(90.0) == pytest.approx(90 / 200 + 200 / 650)
    assert replace(RULES, turn_acc_deg_s2=0.0).ease("turn") is None, "0: the arbiter's saccade"
    stops = sweep(rad(-15), TILT, span_rad=rad(20))
    assert [round(deg(a.pan_rad), 6) for a in stops] == [-35.0, 5.0], "outward first"
    assert all(a.tilt_rad == TILT for a in stops)
    assert deg(sweep(rad(40), TILT, span_rad=rad(20))[0].pan_rad) == pytest.approx(60.0)
    assert sweep(0.0, TILT, span_rad=0.0) == []


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


# ---- the people and whom to look at ----------
def face(pan: float, width: float = 60.0, tilt: float = 0.0, who: str | None = None) -> Any:
    return FaceSighting(pan, tilt, width, 0.8, who)


def frame(*faces: FaceSighting, bodies: tuple[BodySighting, ...] = ()) -> FaceFrame:
    return FaceFrame(0.0, faces, bodies=bodies)


def body(pan: float, head_pan: float, head_tilt: float, score: float = 0.7) -> BodySighting:
    return BodySighting(pan, 12.0, head_pan, head_tilt, 40.0, score)


def tracks() -> Tracks:
    return Tracks(default_tilt_rad=TILT, min_new_px=40.0)


def test_a_frame_s_people_are_faces_and_the_bodies_no_face_speaks_for() -> None:
    head = Aim(rad(30), rad(5))
    boxes = (body(-12.0, -10.0, 0.0), body(40.0, 38.0, 12.0), body(0.0, 0.0, 0.0, 0.3))
    seen = frame(face(-10.0, tilt=2.0), face(25.0, width=20.0), bodies=boxes)
    people = sightings(seen, head, min_face_px=28.0, min_body_score=0.5)
    assert [(p.source, round(deg(p.pan)), round(deg(p.head.tilt_rad))) for p in people] == [
        ("face", 20, 7),  # its body box (-12) is its face's
        ("body", 70, 17),  # lying low on the right: the box's head estimate
    ], "a 20 px face is under the floor, a 0.3 box is nobody"
    assert people[0].distance_m == pytest.approx(80.0 / 60.0)
    assert face_aim(Aim(0.1, 0.2), face(10.0, tilt=-5.0)).tilt_rad == pytest.approx(0.2 - rad(5))


def test_freshness_decides_between_a_track_s_place_and_a_voice() -> None:
    """A person seen within a second keeps their place against the noisy array; one seen
    longer ago takes the fresh bearing and keeps the tilt their face had."""
    known = tracks()
    (artem,) = [known.heard(rad(40), 0.0)[0]]
    known.seen([Sighting(rad(40), Aim(rad(40), rad(8)), "face", face(0.0))], 1.0)
    assert artem.seen_s == 1.0 and artem.head == Aim(rad(40), rad(8))
    same, new = known.heard(rad(48), 1.5)
    assert same is artem and not new and artem.pan == pytest.approx(rad(40)), "seen 0.5 s ago"
    assert artem.voice_pan == pytest.approx(rad(48))
    known.heard(rad(55), 30.0)  # half a minute later, 15 deg off his place
    assert artem.pan == pytest.approx(rad(55)) and artem.head == Aim(rad(55), rad(8))
    stranger, new = known.heard(rad(-60), 31.0)
    assert new and stranger.head == Aim(rad(-60), TILT) and stranger.seen_s is None
    assert known.forget(400.0, keep={stranger.id}) == [artem] and known.tracks == [stranger]


def test_detections_join_the_nearest_track_and_a_sought_track_takes_the_nearest_person() -> None:
    known = tracks()
    a = known.heard(rad(-15), 0.0)[0]
    seen = known.seen([Sighting(rad(-43.5), Aim(rad(-43.5), rad(2.8)), "face", face(0.0))], 0.5)
    assert seen == {} and len(known.tracks) == 2, "28.5 deg off: someone else, unless sought"
    claimed = known.seen([Sighting(rad(-43.5), Aim(rad(-43.5), rad(2.8)), "face", face(0.0))],
                         0.6, seeking=a, seek_rad=rad(40))  # fmt: skip
    assert claimed.keys() == {a.id} and known.tracks == [a], "the earlier sighting merged in"
    assert a.pan == pytest.approx(rad(-43.5)) and a.source == "face"
    small = Sighting(rad(60), Aim(rad(60), 0.0), "face", face(0.0, width=30.0))
    assert known.seen([small], 1.0) == {} and len(known.tracks) == 1, "a small face is nobody"
    known.seen([Sighting(rad(-40), Aim(rad(-41), rad(9)), "body")], 1.2)
    assert a.head == Aim(rad(-43.5), rad(2.8)), "a body after a fresh face keeps the face's aim"
    known.seen([Sighting(rad(-40), Aim(rad(-41), rad(9)), "body")], 3.0)
    assert a.head == Aim(rad(-41), rad(9)) and a.source == "body"


def test_the_speaker_decides_stay_switch_check_or_seek_and_a_stimulus_is_judged_first() -> None:
    seen = Track(1, rad(60), Aim(rad(60), 0.0), "face", 0.0, seen_s=1.0)
    voiced = Track(2, rad(-60), Aim(rad(-60), TILT), "voice", 2.0, heard_s=2.0)
    other = Track(3, rad(-20), Aim(rad(-20), 0.0), "face", 0.0, seen_s=0.5)
    assert decide(seen, seen, named=False).act == "stay"
    assert decide(seen, voiced, named=True).act == "seek", "the name: no return"
    assert decide(seen, other, named=False).act == "switch", "a person seen there: at once"
    assert decide(seen, voiced, named=False).act == "check", "nobody seen there: look and return"
    assert decide(None, voiced, named=False).act == "seek"
    assert decide(voiced, other, named=False).act == "switch"
    loud = voice_of(speech(240, 0.0, 1.0), MAP)
    assert doubt(Heard(0, 1, "да", peak_db=-12.0), loud, min_speech=6, min_dbfs=-18.0) is None
    quiet = doubt(Heard(0, 1, "Всё правильно.", peak_db=-21.7), loud, min_speech=6, min_dbfs=-18)
    assert quiet == "quiet, peak -21.7 dBFS under -18"
    thin = voice_of(speech(240, 0.0, 0.4), MAP)
    assert doubt(Heard(0, 1, "да"), thin, min_speech=6, min_dbfs=-18.0) == (
        "5 speech readings, under 6"
    )
    assert doubt(Heard(0, 1, "", noise=True), loud, min_speech=6, min_dbfs=-18.0) == "no words"


# ---- the engine ----------
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

    def aim(self, aim: Aim, *, speed: Speed, ttl_s: float, ease: Ease | None = None) -> GazeAnswer:
        self.calls.append(("aim", (aim, speed, ttl_s, ease)))
        answer = self.answers.popleft() if self.answers else GazeAnswer("done", head=aim)
        if answer.status == "done" and self.view is not None:
            self.view = replace(self.view, head=aim, holder=SPEAKER, band=5)
        return answer

    def aims(self) -> list[tuple[Aim, Speed]]:
        return [(c[1][0], c[1][1]) for c in self.calls if c[0] == "aim"]

    def eases(self) -> list[Ease | None]:
        return [c[1][3] for c in self.calls if c[0] == "aim"]

    def count(self, name: str) -> int:
        return sum(1 for c in self.calls if c[0] == name)


def speech(deg_: int, start: float, end: float, *, flagged: bool = True) -> list[DoaReading]:
    """The array's readings of a voice at array angle ``deg_`` from ``start`` to ``end`` (laptop
    seconds), on the board's clock, ten a second."""
    steps = round((end - start) * 10)
    return [DoaReading(BOARD + start + k / 10, deg_, flagged) for k in range(steps + 1)]


def array(bearing: float) -> int:
    """The array's angle of a voice at ``bearing`` degrees (the simple map)."""
    return round(180 + bearing) % 360


@dataclass
class Rig:
    machine: SpeakerGaze
    door: Door
    doa: FakeDoa
    scene: FakeScene
    wheels: Any
    face: FakeFace
    said: list[str]
    events: list[tuple[str, float, dict[str, Any]]]

    def person(self, pan: float, **kw: Any) -> FakePerson:
        """Someone at ``pan`` degrees in the room."""
        p = FakePerson(pan, **kw)
        self.scene.people.append(p)
        return p

    def say(
        self,
        bearing: float,
        start: float,
        end: float,
        text: str = "да",
        *,
        peak: float | None = -12.0,
        named: bool = False,
        noise: bool = False,
        flagged: int | None = None,
        read_s: float = 0.1,
    ) -> float:
        """An utterance from ``bearing`` read ``read_s`` after it ended (GigaAM), the room's
        word-less readings in between; ``flagged``: only its last that many readings carry the
        speech flag. When it was read."""
        read = end + read_s
        voice = speech(array(bearing), start, end)
        if flagged is not None:
            voice = [replace(x, speech=k >= len(voice) - flagged) for k, x in enumerate(voice)]
        room = [replace(x, speech=False) for x in speech(30, end + 0.1, read)]
        self.doa.readings += voice + room
        self.machine.heard(Heard(start, end, text, noise, peak, named), read)
        return read

    def wake(self, bearing: float, *, start: float = 0.0, end: float = 1.2, **kw: Any) -> float:
        """The robot's name from ``bearing``: the conversation opens on it. When it was read."""
        read = end + 0.1
        self.doa.readings += speech(array(bearing), start, end) + speech(30, end + 0.1, read)
        self.doa.readings[-1] = replace(self.doa.readings[-1], speech=False)
        self.machine.opened(Heard(start, end, "Пепин", named=True, **kw), read)
        return read

    def run(
        self,
        until: float,
        start: float,
        *,
        voice: Callable[[float], float | None] | None = None,
        move: Callable[[float], None] | None = None,
    ) -> float:
        """Time from ``start`` to ``until`` in 0.05 s steps: the array reads ``voice(t)`` (a
        bearing while the person speaks, None for the room) ten times a second, ``move(t)``
        moves the people, the machine is ticked whenever due. The last time ticked."""
        due = self.machine.tick(start)
        now = start
        for k in range(1, round((until - start) / 0.05) + 1):
            now = round(start + k * 0.05, 6)
            if k % 2 == 0 and voice is not None:
                bearing = voice(now)
                stamp = BOARD + now
                if bearing is None:
                    self.doa.readings.append(DoaReading(stamp, 30, False))
                else:
                    self.doa.readings.append(DoaReading(stamp, array(bearing), True))
            if move is not None:
                move(now)
            if due is not None and now >= due - 1e-9:
                due = self.machine.tick(max(now, due))
        return now

    def aimed(self) -> list[tuple[float, float]]:
        """Every aim asked, as (pan, tilt) degrees rounded."""
        return [(round(deg(a.pan_rad), 1), round(deg(a.tilt_rad), 1)) for a, _ in self.door.aims()]

    def kinds(self, prefix: str = "gaze_") -> list[str]:
        return [k for k, _, _ in self.events if k.startswith(prefix)]

    def reaims(self) -> list[dict[str, Any]]:
        return [f for k, _, f in self.events if k == "gaze_reaim"]


def rig(
    *,
    settings: SpeakerSettings | None = None,
    wheels: Any = None,
    finder: bool = True,
    head: Aim = HOME_AIM,
) -> Rig:
    door = Door(view=view(head=head))
    doa = FakeDoa()
    scene = FakeScene(door)
    face_channel = FakeFace()
    said: list[str] = []
    events: list[tuple[str, float, dict[str, Any]]] = []
    wheels = wheels if wheels is not None else NoWheels()
    machine = SpeakerGaze(
        settings if settings is not None else RULES,
        door=door,
        doa=doa,
        finder=scene if finder else None,
        wheels=wheels,
        face=face_channel,
        reach=REACH,
        report=said.append,
        events=lambda kind, now, **f: events.append((kind, now, f)),
    )
    return Rig(machine, door, doa, scene, wheels, face_channel, said, events)


def tracked(pan: float = 60.0, **kw: Any) -> Rig:
    """Artem at ``pan``, found by the wake's first frame and tracked."""
    r = rig()
    r.person(pan, **kw)
    read = r.wake(pan)
    r.machine.tick(read)
    assert r.machine.phase == "track"
    return r


def test_the_wake_is_one_eased_motion_of_pan_and_tilt_and_the_face_is_found_without_a_sweep() -> (
    None
):
    """Artem: "при «Пепин» он поднимает голову, потом направляет — как терминатор". Now the
    name from 60 deg left is ONE aim, pan and tilt together, on the turn's ease (the head from
    home, 24 deg down, to the voice at +3: no -5, -15, -5 steps); the first frame finds his face
    2 deg off, close enough: no second move; one smile."""
    r = rig()
    r.person(60.0, tilt_deg=2.0)
    read = r.wake(60.0)
    assert r.aimed() == [(60.0, 3.0)] and r.door.eases()[0] == RULES.ease("turn")
    assert r.machine.phase == "seek" and "+60 deg" in r.said[0]
    (turn,) = r.reaims()
    assert turn["why"] == "wake" and turn["speed"] == "turn"
    r.machine.tick(read)
    assert r.machine.phase == "track" and r.aimed() == [(60.0, 3.0)], "within refine: no move"
    assert r.face.calls == [("express", ("smile", 1.0, 2.0, None))]
    r.run(4.0, read)
    assert r.scene.looks == 28 and len(r.door.aims()) == 1, "ten frames a second, no aim"
    assert r.kinds()[:4] == ["gaze_doa", "gaze_wake", "gaze_reaim", "gaze_face"]


def test_the_wake_turns_to_the_bearing_even_near_where_the_head_points() -> None:
    """cascade_155039: the head at -0.2, the name read at -15: it stayed at -0.2 (the face was
    at -43.5 and was found 18 s later). Now it turns to -15, where the face is in the frame (28
    deg off) and taken at once."""
    r = rig(head=Aim(rad(-0.2), rad(-5)))
    r.person(-43.5, tilt_deg=2.8, face_px=65.0)
    read = r.wake(-15.0, peak_db=-22.0)
    assert r.aimed() == [(-15.0, 3.0)]
    r.machine.tick(read)
    assert r.machine.phase == "track" and r.aimed()[-1] == (-43.5, 2.8)
    assert r.reaims()[-1]["why"] == "onto the face" and r.reaims()[-1]["speed"] == "turn"


def test_a_stale_remembered_place_15_deg_off_the_bearing_gives_the_tilt_not_the_pan() -> None:
    """Artem was found low on the sofa (+8 deg) at +40 in the last conversation; a minute later
    he calls from +55: the head turns to +55, the fresh bearing, at his remembered +8 tilt, and
    finds him there."""
    r = rig()
    artem = r.person(40.0, tilt_deg=8.0)
    read = r.wake(40.0)
    r.machine.tick(read)
    r.machine.closed("idle", 10.0)
    artem.pan_deg = 55.0
    read = r.wake(55.0, start=70.0, end=71.2)
    assert r.aimed()[-1] == (55.0, 8.0), "the bearing's pan, the remembered tilt"
    r.machine.tick(read)
    assert r.machine.phase == "track" and r.aimed()[-1] == (55.0, 8.0)


def test_a_face_35_deg_off_the_bearing_is_found_by_the_pan_sweep_within_2_s_without_a_tilt() -> (
    None
):
    """The array read a quiet name 35 deg off the face (it read 28 off on 2026-10-12): nobody at
    the bearing (the camera's usable half width taken as 30 deg), the voice has not moved, so
    the head drifts 20 deg outward, then inward; one frame a stop; the face is found within 2 s
    of the wake on the robot's clock (each move's eased time and 0.1 s a frame), the tilt never
    leaves the remembered +3 until the face is centred."""
    for face_at, stops in ((-50.0, 1), (20.0, 2)):  # outward of the bearing, then inward
        r = rig()
        r.person(face_at, tilt_deg=3.0)
        read = r.wake(-15.0, peak_db=-22.0)
        r.run(read + 2.0, read)
        aims = r.aimed()
        assert aims[: 1 + stops] == [(-15.0, 3.0), (-35.0, 3.0), (5.0, 3.0)][: 1 + stops]
        assert all(tilt == 3.0 for _, tilt in aims), "no tilt sweep"
        assert r.machine.phase == "track" and aims[-1] == (face_at, 3.0)
        reaims = r.reaims()
        found = next(i for i, f in enumerate(reaims) if f["why"] == "onto the face")
        moves = [f["move_s"] for f in reaims[:found]]  # the centring comes after the find
        frames = stops + 1
        assert sum(moves) + frames * FRAME_S <= 2.0, (moves, frames)


def test_the_head_follows_a_person_walking_while_he_speaks_and_arrives_where_he_is() -> None:
    """Artem: «я сказал «Пепин» на ходу, он повернулся туда, где я был секунду назад». He says
    the name walking left at 15 deg/s from +10, his back half turned (no face, no body seen),
    and goes on talking for 3 s: the head follows the array's stream (the last 0.5 s, steps over
    8 deg) and is within 12 deg of him when he stops at +70; he turns round and is found."""
    r = rig()
    artem = r.person(10.0, face_px=0.0)

    def where(t: float) -> float:
        return min(10.0 + 15.0 * t, 70.0)

    r.doa.readings += [DoaReading(BOARD + k / 10, array(where(k / 10)), True) for k in range(13)]
    read = 1.3
    r.machine.opened(Heard(0.0, 1.2, "Пепин, иди сюда", named=True), read)
    first = r.aimed()[0][0]
    assert 15.0 <= first <= 25.0, "where he was during the name"
    r.machine.speech_started(1.35, 1.4)

    def move(t: float) -> None:
        artem.pan_deg = where(t)

    def voice(t: float) -> float | None:
        return where(t) if t <= 4.0 else None

    now = r.run(4.1, read, voice=voice, move=move)
    pan = deg(r.machine.aim.pan_rad) if r.machine.aim is not None else 0.0
    assert abs(pan - 70.0) <= 12.0, (pan, r.aimed())
    assert r.machine.counts["follows"] >= 3 and "the voice moved" in [f["why"] for f in r.reaims()]
    read = r.say(70.0, 1.35, 4.0, "я тут, у стола")
    artem.face_px = 60.0  # he turns round
    r.run(read + 1.0, max(read, now))
    assert r.machine.phase == "track" and r.aimed()[-1][0] == 70.0


def test_a_person_lying_on_the_sofa_is_aimed_at_by_the_body_box_without_a_face() -> None:
    """Lying on the sofa at -40, the face turned away: the body detector's box puts the head
    near -36 and 12 deg down; the head goes there from the voice's +3 in one adjustment,
    tracks the box, and does not smile at a body."""
    r = rig()
    r.person(-40.0, tilt_deg=14.0, face_px=0.0, body=True, head_pan_deg=-36.0, head_tilt_deg=12.0)
    read = r.wake(-40.0)
    r.machine.tick(read)
    assert r.aimed() == [(-40.0, 3.0), (-36.0, 12.0)]
    assert r.machine.phase == "track" and r.reaims()[-1]["why"] == "onto the person"
    face_event = next(f for k, _, f in r.events if k == "gaze_face")
    assert face_event["source"] == "body" and r.face.calls == []
    r.run(read + 3.0, read)
    assert r.machine.phase == "track" and len(r.door.aims()) == 2


def test_a_second_speaker_takes_the_gaze_and_the_first_takes_it_back_at_once() -> None:
    """Artem tracked at +60; his wife, never seen, speaks from -40: a check (one frame) finds
    her and she is engaged. Artem answers from +65 by the array: his track, seen before, is
    switched to at once (the fresh bearing at his face's tilt), his face centred; she speaks
    again: back to her at once."""
    r = tracked(60.0, tilt_deg=2.0)
    r.person(-40.0, tilt_deg=6.0)
    read = r.say(-40.0, 3.0, 4.0, "А мне что расскажешь?")
    assert r.aimed()[-1] == (-40.0, 3.0) and r.machine.verifying
    assert "a voice elsewhere" in r.said[-1]
    r.machine.tick(read)
    assert r.machine.phase == "track" and r.aimed()[-1] == (-40.0, 6.0)
    wife = r.machine.track
    read = r.say(65.0, 5.0, 6.0, "Она шутит")
    assert r.aimed()[-1] == (65.0, 2.0) and not r.machine.verifying
    assert "a person seen there" in r.said[-1]
    r.machine.tick(read)
    assert r.machine.phase == "track" and r.aimed()[-1] == (60.0, 2.0)
    r.say(-38.0, 8.0, 9.0, "Ничего я не шучу")
    assert r.machine.track is wife and r.aimed()[-1] == (-38.0, 6.0), "her bearing, her tilt"


def test_a_voice_where_nobody_is_is_checked_with_one_frame_and_the_head_comes_straight_back() -> (
    None
):
    """The TV says a sentence at -60 while Artem is tracked at +60: the head looks there, ONE
    frame finds nobody, and it turns back to his face; two aims, one frame, no sweep."""
    r = tracked(60.0)
    looks, aims = r.scene.looks, len(r.door.aims())
    read = r.say(-60.0, 3.0, 4.0, "в эфире новости")
    r.machine.tick(read)
    assert r.scene.looks == looks + 1 and r.aimed()[aims:] == [(-60.0, 3.0), (60.0, 3.0)]
    assert r.machine.phase == "track" and "back to the face at +60 deg" in r.said[-1]
    assert r.reaims()[-1]["speed"] == "turn"


def test_a_check_never_takes_the_person_it_returns_to_for_the_voice() -> None:
    """A voice 40 deg from Artem (past the voice's 25) with the camera's whole 47 deg half
    width: the check's frame sees Artem at its edge, within the seek's reach of the voice; he
    is the one the check returns to, not the voice's person."""
    r = tracked(60.0)
    r.scene.half_pan_deg = 47.0
    read = r.say(20.0, 3.0, 4.0, "кто это сказал")
    r.machine.tick(read)
    assert r.aimed()[-2:] == [(20.0, 3.0), (60.0, 3.0)] and r.machine.phase == "track"
    assert len(r.machine.tracks.tracks) == 2, "Artem, and a voice nobody was seen at"


def test_quiet_mumbling_noise_and_thin_utterances_move_nothing() -> None:
    """2026-10-12, cascade_153944: "Всё правильно." at -21.7 dBFS and "Нет." at -20 read -84
    while Artem was tracked at -8: two detours. Under -18 dBFS, without words or with fewer
    than six speech readings nothing moves; the face stays tracked."""
    r = tracked(-8.0)
    aims = len(r.door.aims())
    r.say(-84.0, 3.0, 3.64, "Всё правильно.", peak=-21.7)
    r.say(-84.0, 4.0, 5.0, "Нет.", peak=-20.0)
    r.say(-84.0, 6.0, 6.5, "", noise=True, peak=-30.0)
    read = r.say(-84.0, 7.0, 8.0, "да", flagged=5)
    r.run(read + 1.0, read)
    assert len(r.door.aims()) == aims and r.machine.phase == "track"
    assert r.machine.counts["sounds_ignored"] == 4
    assert "quiet, peak -21.7 dBFS under -18" in r.said[-3]


def test_the_robot_s_name_turns_the_head_whatever_its_level_and_never_returns() -> None:
    r = tracked(60.0)
    r.person(-80.0, face_px=30.0)
    read = r.say(-80.0, 3.0, 4.0, "Привет, Пепин", peak=-31.0, named=True, flagged=3)
    assert r.aimed()[-1] == (-80.0, 3.0) and not r.machine.verifying
    assert "the robot's name" in r.said[-1]
    r.machine.tick(read)
    assert r.machine.phase == "track" and r.aimed()[-1] == (-80.0, 3.0), "a 30 px face, sought"


def test_an_utterance_over_the_robot_s_reply_turns_the_head_by_its_clean_part() -> None:
    """cascade_150557 replayed: the wake read -94 and nobody was found there; the robot answers
    from 2 s to 6 s; Artem talks from 5 s to 8 s at -15. Before, every such utterance was
    thrown away; now its readings after the reply (and its echo) turn the head to him. One
    wholly inside a reply moves nothing."""
    r = rig()
    r.person(-15.0)
    read = r.wake(-94.0)
    r.run(read + 2.0, read)
    assert r.machine.phase == "voice" and r.aimed()[-1] == (-94.0, 3.0)
    r.machine.speaking(True, 3.4)
    r.doa.readings += speech(300, 3.5, 5.0)  # the robot's own voice
    read = r.say(-15.0, 4.0, 4.8, "Как у тебя дела?")
    assert r.machine.counts["over_robot"] == 1 and r.aimed()[-1] == (-94.0, 3.0)
    r.doa.readings += speech(300, 5.0, 6.0)
    r.machine.speaking(False, 6.0)
    read = r.say(-15.0, 5.0, 8.0, "Я тут хотел спросить про Колобка")
    assert r.aimed()[-1] == (-15.0, 3.0)
    r.machine.tick(read)
    assert r.machine.phase == "track"


def test_while_the_robot_speaks_one_missed_frame_is_no_loss_and_three_and_3_s_are() -> None:
    """At 1 Hz while the robot speaks (2026-10-12: lost/found every 2 s), a missed detection
    is not a loss; three missed frames and two seconds are: the head stays where it is."""
    r = tracked(60.0)
    artem = r.scene.people[0]
    r.machine.speaking(True, 1.4)
    r.machine.voiced(1.45)
    now = r.run(3.0, 1.45)
    artem.face_px = 0.0  # one frame misses him
    now = r.run(4.0, now)
    artem.face_px = 60.0
    now = r.run(6.0, now)
    assert r.machine.phase == "track" and r.machine.misses == 0
    artem.face_px = 0.0
    now = r.run(8.0, now)
    assert r.machine.phase == "track" and r.machine.misses == 2, "two frames missed at 1 Hz"
    r.run(8.5, now)
    assert r.machine.phase == "voice" and "lost" in r.said[-1] and "stays" in r.said[-1]
    assert r.aimed()[-1] == (60.0, 3.0)


def test_a_face_lost_for_good_fades_to_where_the_voice_was_heard_since() -> None:
    """Artem tracked at +60 turns his back (no face, no box) and speaks from +78 a second and
    more after the last sighting: when he is lost (three frames and 2 s) the head adjusts to his
    voice, and the frames go on there."""
    r = tracked(60.0)
    artem = r.scene.people[0]
    artem.face_px = 0.0
    now = r.run(2.4, 1.3)
    read = r.say(78.0, 2.5, 3.5, "я тут, за спиной")
    now = r.run(4.0, max(read, now))
    assert r.machine.phase == "voice", r.said[-3:]
    assert r.aimed()[-1] == (78.0, 3.0) and r.reaims()[-1]["why"] == "to the voice"
    assert r.reaims()[-1]["speed"] == "adjust"


def test_past_the_comfort_the_head_goes_first_the_wheels_turn_the_rest_and_the_head_recentres() -> (
    None
):
    """A call from 100 deg left: the head is there at once, the wheels are only started (the
    wake returns before they move), every step they report re-centres the head on the person,
    their end once more; no frame is taken while the body turns."""
    r = rig(wheels=FakeWheels())
    artem = r.person(100.0)
    read = r.wake(100.0)
    (start,) = r.wheels.calls
    assert start == ("start", (WheelMove(turn_deg=pytest.approx(30.0)),))
    assert r.aimed()[0] == (100.0, 3.0) and r.machine.turning is not None
    assert r.machine.tick(read) is not None and r.scene.looks == 0, "no frame yet"
    r.wheels.move_on(10.0)
    artem.pan_deg = 90.0  # the room turns the other way in the cart's frame
    r.machine.tick(read + 0.2)
    assert r.aimed()[-1] == (90.0, 3.0) and r.reaims()[-1]["speed"] == "adjust"
    r.wheels.finish()
    artem.pan_deg = 70.0
    r.machine.tick(read + 0.8)
    assert [p for p, _ in r.aimed()[:3]] == [100.0, 90.0, 70.0]
    assert r.machine.voice == pytest.approx(rad(70)), "the person moved with the cart"
    assert r.machine.turning is None and r.scene.looks == 1, "the frame after"


def test_wheels_that_turn_less_leave_the_head_at_its_reach_and_a_cut_turn_is_finished_later() -> (
    None
):
    stub = rig()
    read = stub.wake(100.0)
    stub.machine.tick(read)  # NoWheels answered at once; the answer is taken on the next tick
    assert stub.aimed()[0] == (100.0, 3.0) and "the wheels turned +0 of +30 deg" in stub.said[1]
    behind = rig()
    behind.wake(180.0)  # straight behind: past the neck's reach
    assert behind.door.aims()[0][0].pan_rad == pytest.approx(REACH.pan[1])
    cut = rig(wheels=FakeWheels())
    cut.wake(100.0)
    cut.wheels.finish(WheelReport(False, 12.0, why="halted: the person speaks", cut=True))
    cut.machine.tick(1.4)
    assert cut.aimed()[1] == (88.0, 3.0) and cut.machine.reorient, "then the seek goes on"
    assert "turned +12 of +30 deg (halted: the person speaks)" in cut.said[-1]
    cut.say(88.0, 3.0, 4.0, "ну что")
    assert [c[0] for c in cut.wheels.calls] == ["start", "start"], "the rest of the turn"


def test_a_drive_and_the_close_let_the_wheels_go_and_their_late_word_is_ignored() -> None:
    r = rig(wheels=FakeWheels())
    r.wake(100.0)
    r.machine.drive(True, 2.0)
    assert r.wheels.calls[-1] == ("halt", ("a drive starts",)) and r.machine.turning is None
    r.wheels.finish(WheelReport(False, 5.0, why="halted: a drive starts", cut=True))
    r.machine.tick(2.1)
    assert len(r.door.aims()) == 1, "a turn let go: its end moves nothing"
    again = rig(wheels=FakeWheels())
    again.wake(100.0)
    again.machine.closed("idle", 3.0)
    assert again.wheels.calls[-1][0] == "halt" and again.door.count("release") == 1


def test_a_drive_pauses_the_gaze_forgets_the_places_and_the_next_utterance_orients_again() -> None:
    r = tracked(60.0)
    r.machine.drive(True, 3.0)
    assert r.machine.phase == "paused" and r.machine.tick(3.0) is None
    assert r.machine.tracks.tracks == [], "the cart moves: every bearing goes stale"
    r.say(60.0, 4.0, 5.0, "куда")
    assert len(r.door.aims()) == 1, "the drive runs"
    r.machine.drive(False, 9.0)
    r.say(20.0, 9.5, 10.5, "приехали")
    assert r.aimed()[-1] == (20.0, 3.0) and r.machine.phase == "seek"


def test_the_conversation_s_end_lets_the_head_go_and_nothing_runs_after() -> None:
    """G8: "пока": home, frames stop; the people stay known for the next conversation."""
    r = tracked(60.0)
    r.machine.closed("goodbye", 4.0)
    assert r.door.count("release") == 1 and r.machine.phase == "idle"
    assert r.machine.tick(5.0) is None and "conversation closed (goodbye)" in r.said[-1]
    looks = r.scene.looks
    r.say(60.0, 5.0, 6.0, "эй")
    assert r.machine.tick(7.0) is None and r.scene.looks == looks
    r.machine.closed("idle", 8.0)
    assert r.door.count("release") == 1, "nothing held: nothing to release"
    assert len(r.machine.tracks.tracks) == 1
    r.machine.stop(9.0, "the switch is off")
    assert r.machine.tracks.tracks == []


def test_a_door_that_does_not_answer_is_asked_again_after_the_renewal_period() -> None:
    r = rig()
    r.person(60.0)
    r.door.view = None
    read = r.wake(60.0)
    assert r.machine.aim is None and r.machine.phase == "voice" and "does not answer" in r.said[-1]
    assert r.machine.tick(1.5) == pytest.approx(read + 2.0)
    r.door.view = view()
    r.machine.tick(read + 2.0)
    assert r.machine.aim is not None and r.aimed()[-1] == (60.0, 3.0)
    assert r.machine.phase == "seek"


def test_no_finder_holds_the_voice_without_frames_and_an_unmeasured_map_looks_where_it_points() -> (
    None
):
    r = rig(finder=False)
    read = r.wake(60.0)
    assert r.machine.phase == "voice" and r.machine.next_frame is None
    assert r.machine.tick(read) == pytest.approx(read + 2.0), "the renewals go on"
    blind = rig(settings=replace(RULES, doa_map=None), head=Aim(rad(50), rad(23.8)))
    blind.wake(60.0)
    assert blind.aimed() == [(50.0, 3.0)] and "unmeasured" in blind.said[0]


def test_tracking_re_aims_past_the_dead_band_and_leads_a_walking_face() -> None:
    """He walks left at 15 deg/s (1.5 deg a frame at 10 Hz): the head re-aims once he is more
    than 6 deg off, where he will be 0.3 s later; standing, it stays still."""
    r = tracked(60.0)
    artem = r.scene.people[0]
    now = r.run(2.3, 1.3)
    assert len(r.door.aims()) == 1, "still: no aim"

    def walk(t: float) -> None:
        artem.pan_deg = 60.0 + 15.0 * (t - 2.3)

    r.run(3.6, now, move=walk)
    leads = [f["lead_deg"] for f in r.reaims()[1:]]
    assert leads and all(3.0 <= lead <= 5.0 for lead in leads), leads
    assert r.reaims()[-1]["why"] == "the person moved" and r.reaims()[-1]["speed"] == "adjust"


def test_frames_wait_for_a_still_head_and_a_gesture_but_not_for_a_breath() -> None:
    r = tracked(60.0)
    looks = r.scene.looks
    r.door.view = replace(r.door.view, blind=True) if r.door.view else None
    r.machine.tick(1.4)
    assert r.scene.looks == looks and r.machine.next_frame == pytest.approx(1.5)
    r.door.view = replace(r.door.view, blind=False, overlay=GESTURE) if r.door.view else None
    r.machine.tick(1.5)
    assert r.scene.looks == looks, "a thinking glance pulls the head away: no frame"
    r.door.view = replace(r.door.view, overlay=IDLE_LOOK) if r.door.view else None
    r.machine.tick(1.6)
    assert r.scene.looks == looks + 1 and r.machine.phase == "track"


def test_a_lost_look_is_asked_for_again_and_waits_behind_the_better_one() -> None:
    """G7: "посмотри налево" took the head: the renewal finds nothing live, the aim is asked
    again and queues; frames wait while another look holds the head."""
    r = tracked(60.0)
    r.door.live = 0
    r.door.answers.append(GazeAnswer("queued", "no answer in 2.0 s: held behind"))
    r.door.view = view(head=Aim(rad(90), 0.0), holder="llm.look", band=2)
    r.machine.next_frame = 3.3
    r.machine.tick(3.3)  # the renewal and the frame are due together
    assert r.door.count("renew") == 1 and len(r.door.aims()) == 2
    assert r.machine.aim is not None and "waits: queued" in r.said[-1]
    assert r.machine.counts["frames_skipped"] == 1


def minute(pan: Callable[[float], float]) -> dict[str, int]:
    """A minute of tracking at the shipped rate, the person at ``pan(t)``: what it cost."""
    r = tracked(60.0)
    artem = r.scene.people[0]
    keys = ("frames", "states", "renewals", "aims")
    before = {k: r.machine.counts[k] for k in keys}

    def move(t: float) -> None:
        artem.pan_deg = pan(t)

    r.run(61.3, 1.3, move=move)
    assert r.machine.phase == "track"
    return {k: r.machine.counts[k] - before[k] for k in keys}


def test_a_minute_of_tracking_at_10_hz_costs_600_state_reads_and_30_renewals() -> None:
    """The budget at the shipped 10 Hz: one state read before each frame, a renewal every 2 s,
    no aim while the person stands still; walking across at 15 deg/s (a person at 1.5 m and
    0.4 m/s) the head is re-aimed past the dead band about every half second."""
    still = minute(lambda t: 60.0)
    assert still == {"frames": 600, "states": 600, "renewals": 30, "aims": 0}
    walking = minute(lambda t: abs(15.0 * ((t - 1.3) % 16.0) - 120.0) - 60.0)  # +60..-60..+60
    assert walking["frames"] == 600 and 100 <= walking["aims"] <= 160, walking
    assert walking["renewals"] < still["renewals"]


def test_every_utterance_s_readings_and_every_decision_land_in_the_log() -> None:
    """The array's readings of each utterance (``gaze_doa``: seconds before its end, the
    angle, 1 speech / 0 not / -1 the robot's own), the aims with their profile and time, the
    verdicts; a log that breaks stops nothing."""
    r = tracked(60.0)
    r.machine.speaking(True, 2.0)
    r.doa.readings += speech(300, 2.1, 2.5)
    r.machine.speaking(False, 2.5)
    r.say(60.0, 2.4, 3.4, "как дела", peak=-14.0)
    doa = [f for k, _, f in r.events if k == "gaze_doa"][-1]
    rows = doa["readings"]
    assert rows[0][:2] == [-1.0, 300] and rows[0][2] == -1, "the robot's own voice"
    assert rows[-1] == [0.0, 240, 1] and doa["text"] == "как дела"
    voice = [f for k, _, f in r.events if k == "gaze_voice"][-1]
    assert voice["verdict"] == "the person engaged" and voice["robot"] == 7, "and its echo"
    assert voice["face_deg"] == 60.0 and voice["bearing_deg"] == 60.0
    first = r.reaims()[0]
    assert first["speed"] == "turn" and first["move_s"] == pytest.approx(0.61, abs=0.01)
    broken = rig()
    broken.person(60.0)
    broken.machine.events = lambda kind, now, **f: 1 / 0
    read = broken.wake(60.0)
    broken.machine.tick(read)
    assert broken.machine.phase == "track"


def test_the_observer_hooks_return_at_once_while_the_gaze_waits_on_the_door() -> None:
    """The gaze never delays a reply: with the worker's thread held in a door call (an aim is
    answered only when the head settles), every hook the loop calls is one queue put."""
    entered, gate = threading.Event(), threading.Event()

    @dataclass
    class SlowDoor(Door):
        def aim(
            self, aim: Aim, *, speed: Speed, ttl_s: float, ease: Ease | None = None
        ) -> GazeAnswer:
            entered.set()
            gate.wait(5.0)
            return super().aim(aim, speed=speed, ttl_s=ttl_s, ease=ease)

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
def test_no_frame_from_a_reply_s_first_part_until_its_first_chunk_then_one_a_second() -> None:
    """The voice's first chunk shares the Mac with the gaze: from the first part queued until
    the first audio no frame is taken (renewals go on); then frames come once a second while the
    robot speaks, and ten a second again the moment it stops."""
    r = tracked(60.0)
    assert r.machine.next_frame == pytest.approx(1.4)
    r.machine.speaking(True, 1.35)
    looks = r.scene.looks
    assert r.machine.tick(1.4) == pytest.approx(3.3), "nothing due but the renewal"
    for t in (1.5, 1.7, 2.2):
        r.machine.tick(t)
    assert r.scene.looks == looks and r.machine.voice_hold
    r.machine.voiced(2.25)
    assert r.machine.tick(2.25) == pytest.approx(3.25)
    assert r.scene.looks == looks + 1, "the frame that waited, at once"
    for t in (2.35, 2.7, 3.2):
        r.machine.tick(t)
    assert r.scene.looks == looks + 1, "one a second while the robot speaks"
    r.machine.tick(3.25)
    r.machine.tick(3.3)
    assert r.scene.looks == looks + 2 and r.door.count("renew") >= 1
    r.machine.speaking(False, 3.4)
    assert r.machine.next_frame == pytest.approx(3.5), "listening: ten a second at once"
    r.machine.speaking(True, 5.0)
    assert r.machine.voice_hold
    r.machine.speaking(False, 5.5)
    assert not r.machine.voice_hold, "a reply cut before its first chunk ends the hold"
    assert r.machine.counts["voice_holds"] == 2


def test_the_hold_is_a_switch_and_the_robot_s_later_parts_hold_nothing() -> None:
    r = tracked(60.0)
    r.machine.configure(replace(r.machine.settings, hold_for_voice=False))
    r.machine.speaking(True, 1.35)
    looks = r.scene.looks
    r.machine.tick(1.45)
    assert r.scene.looks == looks + 1 and not r.machine.voice_hold
    on = tracked(60.0)
    on.machine.speaking(True, 1.35)
    on.machine.voiced(1.36)
    on.machine.speaking(True, 1.5)  # the reply's second part
    assert not on.machine.voice_hold


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
    assert settings.body.max_turn_deg == 30.0 and settings.min_speech == 2
    assert settings == replace(SpeakerSettings(), doa_map=settings.doa_map, body=settings.body)
    assert (settings.seek_tilt_deg, settings.seek_pan_deg, settings.seek_frames) == (3.0, 20.0, 1)
    assert (settings.doa_min_speech, settings.doa_min_dbfs) == (6, -18.0)
    assert (settings.face_lost_frames, settings.face_lost_s) == (3, 2.0)
    assert (settings.turn_acc_deg_s2, settings.adjust_acc_deg_s2) == (650.0, 400.0)
    assert (settings.doa_follow_s, settings.doa_moved_deg, settings.follow_max_deg) == (
        0.5,
        8.0,
        35.0,
    )
    assert flag_default(REPO / "config/expression.json") is False
    bad_knobs = ({"track_hz": 0}, {"max_tracks": 0}, {"lead_max_deg": -1},
                 {"track_speaking_hz": 0}, {"face_lost_frames": 0}, {"turn_speed_deg_s": 0},
                 {"doa_follow_s": 0})  # fmt: skip
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
    r = rig()
    r.person(60.0)
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
    worker.speech_started(0.5)
    worker.part_queued(1, 0, None)  # type: ignore[arg-type]
    worker.reply_done(1)
    worker.drive(True)
    assert worker.idle.wait(2.0), "a drive pauses it: nothing due"
    assert r.door.count("aim") >= 1 and r.machine.phase == "paused"
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


def test_the_fake_finder_still_scripts_frames() -> None:
    finder = FakeFaceFinder(deque([frame(face(1.0)), None]))
    assert finder.look() == frame(face(1.0)) and finder.look() is None and finder.looks == 2
