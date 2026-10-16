"""The speaker gaze (pepin.expression doa, plan, policy, tracks, tracker): the voice's bearing,
the head and the wheels' split, the people tracked, whom to look at, and the engine on a fake
clock against the fakes of every seam (pepin.expression.fakes): a room of people the head camera
sees from wherever the arbiter says the head points, and the array's readings as the person
speaks. Every case the live sessions of 2026-10-12 hit is one scenario here."""
# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

from __future__ import annotations

import itertools
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
    CHIP,
    WHEELS_ONSET_S,
    Tagged,
    Tagger,
    VoiceRule,
    bearing_deg,
    fit_doa_map,
    recent_voice,
    robot_readings,
    utterance_readings,
    utterance_voice,
    voice_of,
    wrap_deg,
)
from pepin.expression.door import view_of
from pepin.expression.executors import NoWheels, WheelMove, WheelReport
from pepin.expression.fakes import (
    FakeDoa,
    FakeFace,
    FakeFaceFinder,
    FakeGazeDoor,
    FakePerson,
    FakeScene,
    FakeTf,
    FakeWheels,
)
from pepin.expression.head import GESTURE, IDLE_LOOK, SPEAKER
from pepin.expression.people_map import Person, Register, Sighting, VoiceRay, sightings
from pepin.expression.plan import BodyLimits, BoundedTurn, Step, head_after, split, sweep
from pepin.expression.policy import decide, doubt
from pepin.expression.poses import HeadModel, Pose, PosesAt
from pepin.expression.speaker import (
    BodySighting,
    DoaMap,
    Ease,
    FaceFrame,
    FaceSighting,
    GazeAnswer,
    HeadView,
    Point,
)
from pepin.expression.tracker import (
    MOVED,
    NO_WALL,
    SIGHTED,
    VOICED,
    SpeakerGaze,
    SpeakerGazeWorker,
    SpeakerSettings,
    flag_default,
)
from pepin.gaze import HOME, Aim, Reach, Speed
from pepin.neck import NeckConfig, NeckPivot

REPO = Path(__file__).resolve().parents[2]
NECK = NeckConfig.from_json(REPO / "config/neck.json")
REACH = Reach.of(NECK)
# The head with its lens on the neck's two axes (no lever arms): a person's angles from the
# camera are the same at any range, so these tests' degrees are exact (the real arms' own
# round trip: test_speaker_gaze_pursuit)
FLAT = HeadModel(replace(NECK, pivot=NeckPivot()))
BOARD = 5000.0  # the board's clock runs this far ahead of the laptop's in these tests
# The array's readings are stamped on the board's MONOTONIC clock (its uptime: 33,450 s on the
# evening of 2026-10-15), the link's offset carrying them onto its wall clock, TF's (BOARD here)
UPTIME = 33_450.0
LINK_OFFSET = BOARD - UPTIME
MAP = DoaMap(front_deg=180.0, sign=1)  # the array reads 180 straight ahead, growing leftward
# The shipped knobs, the simple map; the body turned by the wheels' executor (body_turn
# wheels: these tests are its; the shipped spin's are test_speaker_gaze_spin.py)
RULES = SpeakerSettings(doa_map=MAP, body_turn="wheels")
TILT = math.radians(3.0)  # the tilt a voice is looked at before anyone is seen
HOME_AIM = Aim(0.0, math.radians(23.8))
FRAME_S = 0.1  # a frame's wall time on the robot: the snapshot, the detector, the blind settle


@dataclass
class FakeSeries:
    """The ears' Silero probabilities: ``p`` everywhere (a voice; None: the model hears nothing)
    but over the scripted ``spans`` (laptop clock: from, to, p), where theirs."""

    p: float | None = 0.9
    spans: list[tuple[float, float, float]] = field(default_factory=list)

    def peak(self, t0: float, t1: float) -> float | None:
        inside = [p for a, b, p in self.spans if a <= t1 and t0 <= b]
        return max(inside) if inside else self.p


def chip(readings: list[DoaReading]) -> list[Tagged]:
    """``readings`` judged by the array's own flag alone (the rule before 2026-10-14)."""
    return Tagger(CHIP).tag(readings, 0.0)


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
    voice = voice_of(chip(readings), MAP)
    assert voice.doa_deg == pytest.approx(240.0) and voice.speech == 2 and voice.readings == 3
    assert voice.bearing_rad == pytest.approx(rad(60.0))
    assert "+60 deg" in voice.text()
    unmeasured = voice_of(chip(readings), None)
    assert unmeasured.doa_deg == pytest.approx(240.0) and unmeasured.bearing_rad is None
    assert "unmeasured" in unmeasured.text()
    assert voice_of(chip(readings), MAP, min_speech=3).doa_deg is None, "two speech of three"
    across = voice_of(chip([DoaReading(0.0, 359, True), DoaReading(0.1, 1, True)]), MAP).doa_deg
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
    assert voice_of(chip(window), MAP).bearing_rad == pytest.approx(rad(60.0))
    short = utterance_readings(doa, Heard(start_s=2.2, end_s=2.5), now=3.3, window_s=1.0)
    assert [round(r.stamp_s - BOARD, 1) for r in short] == [2.2, 2.3, 2.4, 2.5]
    assert utterance_readings(FakeDoa(), heard, now=3.3, window_s=1.0) == []
    clean = utterance_voice(doa, heard, 3.3, window_s=1.0, doa_map=MAP, min_speech=2,
                            tagger=Tagger(CHIP))  # fmt: skip
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
    mask = robot_readings(doa.readings, [(2.0, 6.0)], UPTIME)  # the readings' own clock
    assert sum(mask) == len(robot) + 4, "the span and its 0.3 s of echo, both ends in"
    heard = Heard(5.0, 8.0, "я тут, слышишь")
    clean = utterance_voice(doa, heard, 8.0, window_s=1.0, doa_map=MAP, min_speech=2,
                            tagger=Tagger(CHIP, spans=[(2.0, 6.0)]))  # fmt: skip
    assert clean.voice.bearing_rad == pytest.approx(rad(-15.0)) and clean.voice.robot == 15
    drowned = utterance_voice(doa, Heard(3.0, 5.5, "да"), 5.5, window_s=1.0, doa_map=MAP,
                              min_speech=2, tagger=Tagger(CHIP, spans=[(2.0, None)]))  # fmt: skip
    assert drowned.voice.bearing_rad is None and drowned.voice.robot > 0
    live = recent_voice(doa, 8.0, window_s=0.5, doa_map=MAP, min_speech=3,
                        tagger=Tagger(CHIP, spans=[(2.0, 6.0)]))  # fmt: skip
    assert live.bearing_rad == pytest.approx(rad(-15.0)) and live.speech == 6


def test_a_whine_the_chip_calls_speech_is_no_voice_and_the_person_s_voice_still_is() -> None:
    """2026-10-14: the neck's whine and the wheels came back "speech" 10 of 10 readings at
    -64..-124 deg. Tagged, a reading is the person's voice only with the chip's flag AND Silero
    at voice_p_min or more AND not the robot's own speech AND not its wheels turning; the chip
    switch is the rule before (its flag, the robot's speech cut out)."""
    whine = speech(array(-120.0), 2.0, 3.0)  # the neck, flagged speech by the array
    him = speech(array(20.0), 3.1, 4.0)  # Artem at +20
    doa = FakeDoa(whine + him)  # read at 4.0, the newest reading's time
    ears = FakeSeries(p=0.9, spans=[(1.9, 2.95, 0.1)])  # Silero: 0.1 on the whine, 0.9 on him
    silero = Tagger(VoiceRule("silero", 0.5), ears)
    noise = utterance_voice(doa, Heard(2.0, 3.0, ""), 4.0, window_s=1.0, doa_map=MAP,
                            min_speech=2, tagger=silero)  # fmt: skip
    assert noise.voice.bearing_rad is None and noise.voice.speech == 0 and noise.voice.chip == 11
    assert [t.tags.voice_p for t in noise.tagged] == [0.1] * 11, "every reading tagged"
    old = utterance_voice(doa, Heard(2.0, 3.0, ""), 4.0, window_s=1.0, doa_map=MAP,
                          min_speech=2, tagger=Tagger(CHIP, ears))  # fmt: skip
    assert old.voice.bearing_rad == pytest.approx(rad(-120.0)), "the chip: the whine's bearing"
    voice = utterance_voice(doa, Heard(3.1, 4.0, "я тут"), 4.0, window_s=1.0, doa_map=MAP,
                            min_speech=2, tagger=silero)  # fmt: skip
    assert voice.voice.bearing_rad == pytest.approx(rad(20.0)) and voice.voice.speech == 10
    live = recent_voice(doa, 4.0, window_s=1.45, doa_map=MAP, min_speech=3, tagger=silero)
    assert live.bearing_rad == pytest.approx(rad(20.0)), "the stream: his readings alone"
    assert live.speech == 10 and live.chip == 15
    own = utterance_voice(doa, Heard(3.1, 4.0, "я тут"), 4.0, window_s=1.0, doa_map=MAP,
                          min_speech=2, tagger=replace(silero, spans=[(3.0, None)]))  # fmt: skip
    assert own.voice.bearing_rad is None and own.voice.robot == 10, "the robot's own: not his"
    assert all(t.tags.own and not t.voice for t in own.tagged)
    deaf = utterance_voice(doa, Heard(3.1, 4.0, "я тут"), 4.0, window_s=1.0, doa_map=MAP,
                           min_speech=2, tagger=Tagger(VoiceRule()))  # fmt: skip
    assert deaf.voice.bearing_rad is None, "no speech model: the silero rule hears nobody"
    assert Tagger(CHIP).tag(him, UPTIME)[0].voice, "the chip needs none"


def test_the_wheels_first_jerk_is_no_voice_however_much_silero_likes_it() -> None:
    """2026-10-14: 0.2-0.4 s after the wheels' command Silero read 0.5 or more in 89 % of the
    cases (the cart's first jerk), later in the turn as the room does. The readings of the first
    WHEELS_ONSET_S of the gaze's own turn are tagged moving and are no voice; the chip switch
    takes them as before; the person speaking later in the turn is heard."""
    jerk = speech(array(70.0), 1.0, 1.6)  # the wheels, flagged speech
    tagger = Tagger(VoiceRule(), FakeSeries(p=0.95), motions=[(1.0, 1.0 + WHEELS_ONSET_S)])
    tagged = tagger.tag(jerk, UPTIME)  # the readings' own clock less the laptop's
    assert [t.tags.moving for t in tagged] == [True] * 6 + [False], "the turn's first 0.5 s"
    assert [t.voice for t in tagged] == [False] * 6 + [True]
    assert all(t.voice for t in replace(tagger, rule=CHIP).tag(jerk, UPTIME))
    assert tagged[0].tags.utterance is None and tagged[0].tags.who is None, "reserved, unset"
    r = rig(wheels=FakeWheels(), arc_deg=60.0)
    r.wake(120.0)  # past the arc: the wheels start at the wake's read
    assert r.machine.turning is not None and r.machine.moved == [(1.3, 1.3 + WHEELS_ONSET_S)]


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


def test_each_reading_is_taken_into_the_map_by_the_cart_s_heading_at_its_stamp() -> None:
    """A voice that stays put while the cart turns 20 deg left reads 40 -> 20 in the cart's
    frame; each reading taken into the map by the cart's heading (TF's) at its own stamp says
    40 in the map, and by the heading at the newest reading, 20 in the cart's frame then: the
    bearing it has now. A reading TF has no heading for is left out; across 180 the mean is
    circular."""
    readings = [DoaReading(BOARD + k / 10, array(40.0 - 2.0 * k), True) for k in range(11)]
    heading = lambda stamp: rad(20.0 * (stamp - BOARD))  # noqa: E731
    plain = voice_of(chip(readings), MAP)
    assert plain.bearing_rad is not None and deg(plain.bearing_rad) == pytest.approx(30.0, abs=0.6)
    assert plain.map_rad is None, "no heading: no direction in the map"
    now = voice_of(chip(readings), MAP, heading=heading)
    assert now.map_rad is not None and deg(now.map_rad) == pytest.approx(40.0, abs=0.6)
    assert now.bearing_rad is not None and deg(now.bearing_rad) == pytest.approx(20.0, abs=0.6)
    assert now.ref_s == pytest.approx(BOARD + 1.0) and now.doa_deg == plain.doa_deg
    gaps = voice_of(chip(readings), MAP, heading=lambda st: None if st > BOARD + 0.55 else 0.0)
    assert gaps.map_rad is not None and deg(gaps.map_rad) == pytest.approx(35.0, abs=0.6)
    assert gaps.ref_s == pytest.approx(BOARD + 0.5), "the newest reading with a heading"
    behind = [DoaReading(BOARD, array(175.0), True), DoaReading(BOARD + 0.1, array(-175.0), True)]
    back = voice_of(chip(behind), MAP, heading=lambda stamp: 0.0)
    assert back.bearing_rad is not None and abs(deg(back.bearing_rad)) == pytest.approx(180.0)


# ---- the head and the wheels ----------
def test_the_head_alone_within_the_forward_arc_and_past_it_the_body_turns_toward_the_person() -> (
    None
):
    """The arc's half is 80 deg: within it the head alone; past it the body turns to face the
    person, a request's worth at most (30), so one just past the edge lands well inside the arc
    and does not set the body hunting at its edge."""
    arc = {"half_arc_rad": rad(80), "max_turn_rad": rad(30)}
    near = split(rad(80), TILT, **arc)
    assert near.body_rad == 0.0 and near.head.pan_rad == pytest.approx(rad(80))
    edge = split(rad(82), TILT, **arc)
    assert edge.body_rad == pytest.approx(rad(30)) and edge.head.pan_rad == pytest.approx(rad(52))
    back = split(rad(-150), TILT, **arc)
    assert back.body_rad == pytest.approx(rad(-30)) and back.head.pan_rad == pytest.approx(
        rad(-120)
    )
    wide = split(rad(100), TILT, half_arc_rad=rad(80), max_turn_rad=rad(180))
    assert wide.body_rad == pytest.approx(rad(100)) and wide.head.pan_rad == pytest.approx(0.0)
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
    """The wheels' attempts: each turn answered from ``blocked`` (a stall short of the turn, the
    degrees it made), ``stopped`` ends one for good; a step away answered from ``aways`` (its
    blocked reason)."""

    def __init__(
        self,
        blocked: list[float] | None = None,
        stopped: str = "",
        aways: list[str] | None = None,
    ) -> None:
        self.blocked = deque(blocked or [])
        self.stopped = stopped
        self.aways = deque(aways or [])
        self.calls: list[tuple[str, float]] = []

    def turn(self, deg: float) -> Step:
        self.calls.append(("turn", deg))
        if self.stopped:
            return Step(math.copysign(2.0, deg), stopped=self.stopped, cut=True)
        if self.blocked:
            return Step(math.copysign(self.blocked.popleft(), deg), blocked="stalled")
        return Step(deg)

    def step_away(self, metres: float) -> Step:
        self.calls.append(("away", metres))
        blocked = self.aways.popleft() if self.aways else ""
        return Step(0.0 if blocked else metres, blocked=blocked)


def test_a_turn_goes_in_one_go_and_a_stall_gets_one_step_away_and_one_more_try() -> None:
    steps = Steps()
    done = BoundedTurn(steps).move(WheelMove(turn_deg=25.0))
    assert done.done and done.turned_deg == pytest.approx(25.0) and done.attempts == 1
    assert steps.calls == [("turn", 25.0)], "one continuous turn, its steps are checkpoints"
    steps = Steps(blocked=[4.0])
    retried = BoundedTurn(steps).move(WheelMove(turn_deg=-20.0))
    assert retried.done and retried.backed_m == pytest.approx(0.02) and retried.attempts == 2
    assert steps.calls == [("turn", -20.0), ("away", 0.02), ("turn", -16.0)]
    steps = Steps(blocked=[1.0, 1.0, 1.0])
    given_up = BoundedTurn(steps).move(WheelMove(turn_deg=20.0))
    assert not given_up.done and given_up.attempts == 2 and given_up.turned_deg == 2.0
    assert given_up.backed_m == pytest.approx(0.02) and given_up.why == "gave up: stalled"
    assert steps.calls == [("turn", 20.0), ("away", 0.02), ("turn", 19.0)], "one try more"
    bounded = BoundedTurn(Steps(), BodyLimits(max_turn_deg=30.0))
    assert "past 30" in bounded.move(WheelMove(turn_deg=35.0)).why
    assert "rock" in BoundedTurn(Steps()).move(WheelMove(rock_m=0.01, rocks=2)).why


def test_a_step_that_cannot_go_turns_back_the_other_way_once_then_the_rest_once_more() -> None:
    steps = Steps(blocked=[3.0], aways=["the step forward is blocked: a mark ahead"])
    unwound = BoundedTurn(steps).move(WheelMove(turn_deg=20.0))
    assert steps.calls == [("turn", 20.0), ("away", 0.02), ("turn", -10.0), ("turn", 27.0)]
    assert unwound.done and unwound.turned_deg == pytest.approx(20.0) and unwound.attempts == 2
    assert unwound.backed_m == 0.0
    steps = Steps(blocked=[3.0, 2.0], aways=["the step back is blocked: a mark behind"])
    stuck = BoundedTurn(steps, BodyLimits(backoff_deg=5.0)).move(WheelMove(turn_deg=-20.0))
    assert steps.calls == [("turn", -20.0), ("away", 0.02), ("turn", 5.0)], "never a retry"
    assert not stuck.done and stuck.attempts == 1 and stuck.turned_deg == pytest.approx(-1.0)
    assert stuck.why == ("gave up: stalled; the step back is blocked: a mark behind;"
                         " the turn back: stalled")  # fmt: skip


def test_a_stopped_turn_never_steps_away() -> None:
    """A halt, someone else's stop, a drive: final (no recovery after the red button)."""
    steps = Steps(stopped="halted: the person speaks")
    cut = BoundedTurn(steps).move(WheelMove(turn_deg=20.0))
    assert not cut.done and cut.cut and cut.why == "halted: the person speaks"
    assert steps.calls == [("turn", 20.0)] and cut.turned_deg == 2.0


# ---- the people and whom to look at ----------
def face(pan: float, width: float = 60.0, tilt: float = 0.0, who: str | None = None) -> Any:
    return FaceSighting(pan, tilt, width, 0.8, who)


def frame(*faces: FaceSighting, bodies: tuple[BodySighting, ...] = ()) -> FaceFrame:
    return FaceFrame(0.0, faces, bodies=bodies)


def body(pan: float, head_pan: float, head_tilt: float, score: float = 0.7) -> BodySighting:
    return BodySighting(pan, 12.0, head_pan, head_tilt, 40.0, score)


ORIGIN = Pose.planar(0.0, 0.0, 0.0)  # the cart at the map's origin, facing +x
LENS = FLAT.lens(Aim(0.0, 0.0))  # the FLAT head's lens: on the axes, at any aim


def register(**knobs: Any) -> Register:
    return Register(min_new_px=40.0, **knobs)


def axis_dir(pan: float, tilt: float) -> tuple[float, float, float]:
    """The head's axis at ``pan`` and ``tilt`` degrees, in the map (the cart at the origin)."""
    p, t = rad(pan), rad(tilt)
    return math.cos(t) * math.cos(p), math.cos(t) * math.sin(p), -math.sin(t)


def at_aim(pan: float, tilt: float, range_m: float = 2.0) -> Point:
    """The point the head looks at, ``pan`` and ``tilt`` degrees, ``range_m`` from the lens."""
    return ORIGIN.to_map(FLAT.axis_point(Aim(rad(pan), rad(tilt)), range_m))


def azimuth_deg(point: Point) -> float:
    return deg(math.atan2(point.y, point.x))


def elevation_deg(point: Point) -> float:
    """How far below level ``point`` is from the lens (degrees)."""
    return deg(math.atan2(LENS[2] - point.z, math.hypot(point.x, point.y)))


def voice_at(pan: float, stamp: float = 0.0) -> VoiceRay:
    """A voice from ``pan`` degrees with the cart at the origin, placed as the gaze places it
    (at a known person's range and height, else 2 m at the seek's tilt)."""

    def place(person: Person | None) -> Point:
        if person is None:
            return at_aim(pan, deg(TILT))
        local = ORIGIN.to_frame(person.point)
        toward = FLAT.aim_at(local)
        return at_aim(pan, deg(toward.tilt_rad), math.dist(FLAT.lens(toward), local))

    return VoiceRay(rad(pan), (0.0, 0.0), stamp, place)


def seen_at(
    pan: float, tilt: float, source: Any = "face", stamp: float = 0.0, width: float = 60.0
) -> Sighting:
    """One person seen from the lens at ``pan`` and ``tilt`` degrees: a face of ``width`` px (at
    the range its size says), or a body box (at the default 2 m)."""
    d = axis_dir(pan, tilt)
    if source == "face":
        range_m, how, seen = 80.0 / width, "face", face(0.0, width)
    else:
        range_m, how, seen = 2.0, "default", None
    point = Point(LENS[0] + d[0] * range_m, LENS[1] + d[1] * range_m, LENS[2] + d[2] * range_m)
    return Sighting(point, source, stamp, LENS, d, range_m, how, seen)  # type: ignore[arg-type]


def test_a_frame_s_people_are_faces_and_the_bodies_no_face_speaks_for() -> None:
    """The camera's pose at the frame's exposure places each face along the ray its angles
    name, at the range its size says (a body box at the default range, its head estimate the
    point); a 20 px face is under the floor, a 0.3 box is nobody."""
    camera = ORIGIN.then(FLAT.camera(Aim(rad(30), rad(5))))
    at = PosesAt(BOARD, camera, ORIGIN)
    boxes = (body(-12.0, -10.0, 0.0), body(40.0, 38.0, 12.0), body(0.0, 0.0, 0.0, 0.3))
    seen = frame(face(-10.0, tilt=2.0), face(25.0, width=20.0), bodies=boxes)
    people = sightings(seen, at, FLAT, min_face_px=28.0, min_body_score=0.5, default_range_m=2.0)
    found = [(p.source, round(azimuth_deg(p.point)), round(elevation_deg(p.point))) for p in people]
    assert found == [
        ("face", 20, 7),  # its body box (-12) is its face's
        ("body", 69, 16),  # lying low on the right: the box's head estimate
    ], "a 20 px face is under the floor, a 0.3 box is nobody"
    assert people[0].range_m == pytest.approx(80.0 / 60.0) and people[0].range_from == "face"
    assert math.dist(LENS, (people[0].point.x, people[0].point.y, people[0].point.z)) == (
        pytest.approx(80.0 / 60.0)
    )
    assert people[1].range_from == "default" and people[0].stamp == BOARD
    measured = frame(replace(face(-10.0, tilt=2.0), depth_m=2.5))
    (deep,) = sightings(measured, at, FLAT, min_face_px=28.0, min_body_score=0.5,
                        default_range_m=2.0)  # fmt: skip
    assert deep.range_m == 2.5 and deep.range_from == "depth", "the stereo depth first"
    assert sightings(seen, PosesAt(BOARD, None, ORIGIN), FLAT, min_face_px=28.0,
                     min_body_score=0.5, default_range_m=2.0) == []  # fmt: skip


def test_freshness_decides_between_a_person_s_point_and_a_voice() -> None:
    """A person seen within a second keeps their point against the noisy array; one seen
    longer ago takes the fresh voice, at their range and the height their face had."""
    known = register()
    (artem,) = [known.heard(voice_at(40.0, 0.0))[0]]
    known.seen([seen_at(40.0, 8.0, stamp=1.0)], 1.0)
    assert artem.seen_s == 1.0 and artem.source == "face" and artem.origin == "utterance"
    assert azimuth_deg(artem.point) == pytest.approx(40.0)
    assert elevation_deg(artem.point) == pytest.approx(8.0)
    face_point = artem.point
    same, new = known.heard(voice_at(48.0, 1.5))
    assert same is artem and not new and artem.point == face_point, "seen 0.5 s ago"
    assert artem.voice is not None and azimuth_deg(artem.voice) == pytest.approx(48.0)
    known.heard(voice_at(55.0, 30.0))  # half a minute later, 15 deg off his point
    assert azimuth_deg(artem.point) == pytest.approx(55.0) and artem.source == "voice"
    assert elevation_deg(artem.point) == pytest.approx(8.0), "his face's height, remembered"
    assert math.dist(LENS, (artem.point.x, artem.point.y, artem.point.z)) == pytest.approx(
        80.0 / 60.0
    ), "at his range"
    stranger, new = known.heard(voice_at(-60.0, 31.0))
    assert new and stranger.seen_s is None and stranger.who is None
    assert azimuth_deg(stranger.point) == pytest.approx(-60.0)
    assert elevation_deg(stranger.point) == pytest.approx(deg(TILT)), "the seek's tilt, 2 m"
    assert known.forget(400.0, keep={stranger.id}) == [artem] and known.people == [stranger]


def test_detections_join_the_nearest_person_and_a_sought_one_takes_the_nearest() -> None:
    known = register()
    a = known.heard(voice_at(-15.0, 0.0))[0]
    seen = known.seen([seen_at(-43.5, 2.8, stamp=0.5)], 0.5)
    assert seen == {} and len(known.people) == 2, "28.5 deg off: someone else, unless sought"
    claimed = known.seen([seen_at(-43.5, 2.8, stamp=0.6)], 0.6, seeking=a, seek_rad=rad(40))
    assert claimed.keys() == {a.id} and known.people == [a], "the earlier sighting merged in"
    assert azimuth_deg(a.point) == pytest.approx(-43.5) and a.source == "face"
    small = seen_at(60.0, 0.0, stamp=1.0, width=30.0)
    assert known.seen([small], 1.0) == {} and len(known.people) == 1, "a small face is nobody"
    known.seen([seen_at(-41.0, 9.0, "body", stamp=1.2)], 1.2)
    assert azimuth_deg(a.point) == pytest.approx(-43.5), "a body after a fresh face: the face's"
    assert a.seen_s == 1.2, "still there"
    known.seen([seen_at(-41.0, 9.0, "body", stamp=3.0)], 3.0)
    assert azimuth_deg(a.point) == pytest.approx(-41.0) and a.source == "body"
    assert elevation_deg(a.point) == pytest.approx(9.0) and a.range_from == "kept"
    assert a.range_m == pytest.approx(80.0 / 60.0), "the box at the range his face gave"


def test_the_height_changes_its_source_only_past_the_hysteresis() -> None:
    """The head nodded between Artem's face and his body box whenever the face came and went in
    the frames: the height changes source (face to box or back) only past tilt_hold_deg of
    elevation; the bearing follows the newest sighting either way, and one source moves its own
    height freely."""
    known = register(tilt_hold_deg=12.0)
    a = known.heard(voice_at(30.0, 0.0))[0]
    assert a.height_from == "voice"
    known.seen([seen_at(30.0, -15.0, stamp=0.1)], 0.1)
    assert elevation_deg(a.point) == pytest.approx(-15.0) and a.height_from == "face"
    known.seen([seen_at(33.0, -5.0, "body", stamp=2.0)], 2.0)  # the box's estimate, 10 lower
    assert azimuth_deg(a.point) == pytest.approx(33.0) and a.height_from == "face"
    assert elevation_deg(a.point) == pytest.approx(-15.0), "10 deg: the face's"
    known.seen([seen_at(31.0, -17.0, stamp=2.1)], 2.1)
    assert elevation_deg(a.point) == pytest.approx(-17.0), "the face itself moves freely"
    known.seen([seen_at(31.0, 8.0, "body", stamp=4.0)], 4.0)  # he sat down: 25 deg
    assert elevation_deg(a.point) == pytest.approx(8.0) and a.height_from == "body"
    known.seen([seen_at(31.0, 1.0, stamp=4.1)], 4.1)
    assert elevation_deg(a.point) == pytest.approx(8.0) and a.height_from == "body", "7: kept"
    assert a.source == "face" and azimuth_deg(a.point) == pytest.approx(31.0)
    assert RULES.tilt_hysteresis_deg == 12.0


def test_the_register_keeps_its_records_explicit_and_their_confidence_halves_with_age() -> None:
    """The live layer of a future social map: a name moves to whoever recognition says it is,
    the camera's sightings fit a velocity in the map, a record's confidence is its source's
    trust halving every half-life, and the array's peaks never move a person's point."""
    known = register(half_life_s=10.0)
    a = known.heard(voice_at(0.0, 0.0))[0]
    b = known.heard(voice_at(90.0, 0.0))[0]
    known.named(a, "Саша")
    known.named(b, "Саша")
    assert (a.who, b.who) == (None, "Саша"), "one person a name"
    for k in range(4):  # walking left at 0.5 m/s, 2 m ahead
        known.seen([seen_at(math.degrees(math.atan2(0.05 * k, 1.333)), 0.0, stamp=1.0 + 0.1 * k)],
                   1.0 + 0.1 * k)  # fmt: skip
    assert a.velocity[1] == pytest.approx(0.5, abs=0.05) and abs(a.velocity[0]) < 0.05
    assert a.confidence(a.stamp, 10.0) == pytest.approx(1.0)
    assert a.confidence(a.stamp + 10.0, 10.0) == pytest.approx(0.5)
    assert b.confidence(b.stamp + 20.0, 10.0) == pytest.approx(0.5 / 4), "a voice's: half"
    before = a.point
    known.peak(voice_at(5.0, 2.0), 0.7)
    assert a.point == before and a.peaks == 1 and a.strength == 0.7, "a peak: never the point"
    alone = known.peak(voice_at(-120.0, 2.0), 0.4)
    assert alone.origin == "peaks" and known.voices() == [a, alone]


def test_the_speaker_decides_stay_switch_check_or_seek_and_a_stimulus_is_judged_first() -> None:
    seen = Person(1, at_aim(60.0, 0.0), 1.0, "face", "camera", 0.0, seen_s=1.0)
    voiced = Person(2, at_aim(-60.0, 3.0), 2.0, "voice", "utterance", 2.0, heard_s=2.0)
    other = Person(3, at_aim(-20.0, 0.0), 0.5, "face", "camera", 0.0, seen_s=0.5)
    assert decide(seen, seen, named=False).act == "stay"
    assert decide(seen, voiced, named=True).act == "seek", "the name: no return"
    assert decide(seen, other, named=False).act == "switch", "a person seen there: at once"
    assert decide(seen, voiced, named=False).act == "check", "nobody seen there: look and return"
    assert decide(None, voiced, named=False).act == "seek"
    assert decide(voiced, other, named=False).act == "switch"
    loud = voice_of(chip(speech(240, 0.0, 1.0)), MAP)
    assert doubt(Heard(0, 1, "да", peak_db=-12.0), loud, min_speech=6, min_dbfs=-18.0) is None
    quiet = doubt(Heard(0, 1, "Всё правильно.", peak_db=-21.7), loud, min_speech=6, min_dbfs=-18)
    assert quiet == "quiet, peak -21.7 dBFS under -18"
    thin = voice_of(chip(speech(240, 0.0, 0.4)), MAP)
    assert doubt(Heard(0, 1, "да"), thin, min_speech=6, min_dbfs=-18.0) == (
        "5 speech readings, under 6"
    )
    assert doubt(Heard(0, 1, "", noise=True), loud, min_speech=6, min_dbfs=-18.0) == "no words"


def test_the_state_read_passes_the_encoder_reading_s_own_stamp_through() -> None:
    """``/state``'s ``head_stamp`` (the encoder reading's moment on the board's clock) reaches
    the gaze with the head it dates; an arbiter that sends none, a stamp that is no finite
    number, or no head: None."""
    state = {"pan_rad": 0.3, "tilt_rad": 0.1, "head_stamp": 1_760_000_000.0325}
    assert view_of(state).head_stamp == 1_760_000_000.0325
    assert view_of({"pan_rad": 0.3, "tilt_rad": 0.1}).head_stamp is None, "an older arbiter"
    for bad in (None, True, float("nan"), "1760000000"):
        assert view_of({**state, "head_stamp": bad}).head_stamp is None, bad
    assert view_of({**state, "pan_rad": None}).head_stamp is None, "no head, no stamp"


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
    """The arbiter as the speaker gaze sees it (pepin.gaze's point look that follows): an aim
    asks for a point in the map, resolved to pan and tilt from the cart's pose now (TF's) and
    clamped toward the head's own pan; one that is done puts the head there and makes the
    speaker the holder, and the state keeps the head on the held point as the cart turns, its
    reading stamped on the board's clock (TF's newest)."""

    answers: deque[GazeAnswer] = field(default_factory=deque)
    tf: FakeTf | None = None
    held: Point | None = None
    resolved: list[Aim] = field(default_factory=list)

    def look_at(self, point: Point) -> Aim:
        """The pan and tilt that look at ``point`` now (the arbiter's geometry and clamp)."""
        assert self.tf is not None
        aim = FLAT.aim_at(self.tf.base(self.tf.now()).to_frame(point))
        here = self.view.head.pan_rad if self.view is not None and self.view.head else None
        return REACH.clamp(aim, from_pan=here)

    def aim(
        self, point: Point, *, speed: Speed, ttl_s: float, ease: Ease | None = None
    ) -> GazeAnswer:
        self.calls.append(("aim", (point, speed, ttl_s, ease)))
        head = self.look_at(point)
        self.resolved.append(head)
        answer = self.answers.popleft() if self.answers else GazeAnswer("done", head=head)
        if answer.status == "done" and self.view is not None:
            self.held = point
            self.view = replace(self.view, head=head, holder=SPEAKER, band=5)
        return answer

    def state(self) -> HeadView | None:
        view = self.view
        if view is None:
            return None
        if self.held is not None and view.holder == SPEAKER:  # the head kept on its point
            view = self.view = replace(view, head=self.look_at(self.held))
        return replace(view, head_stamp=self.tf.now() if self.tf is not None else None)

    def aims(self) -> list[tuple[Aim, Speed]]:
        """Every aim asked: the pan and tilt it resolved to then, and its speed."""
        speeds = [c[1][1] for c in self.calls if c[0] == "aim"]
        return list(zip(self.resolved, speeds, strict=True))

    def points(self) -> list[Point]:
        return [c[1][0] for c in self.calls if c[0] == "aim"]

    def eases(self) -> list[Ease | None]:
        return [c[1][3] for c in self.calls if c[0] == "aim"]

    def count(self, name: str) -> int:
        return sum(1 for c in self.calls if c[0] == name)


def read_at(t: float, deg_: int, flagged: bool) -> DoaReading:
    """The array's reading at laptop second ``t``: stamped on the board's monotonic clock, with
    the link's offset onto its wall clock (``wall_s``: BOARD + t, TF's)."""
    return DoaReading(UPTIME + t, deg_, flagged, LINK_OFFSET)


def speech(deg_: int, start: float, end: float, *, flagged: bool = True) -> list[DoaReading]:
    """The array's readings of a voice at array angle ``deg_`` from ``start`` to ``end`` (laptop
    seconds), on the board's clock, ten a second."""
    steps = round((end - start) * 10)
    return [read_at(start + k / 10, deg_, flagged) for k in range(steps + 1)]


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
    tf: FakeTf
    series: FakeSeries = field(default_factory=FakeSeries)  # the ears' Silero: a voice throughout
    turns: list[tuple[float, float]] = field(default_factory=list)  # TF's (board stamp, heading)

    def person(self, pan: float, **kw: Any) -> FakePerson:
        """Someone at ``pan`` degrees in the room (the map's, from the cart's first place)."""
        p = FakePerson(pan, **kw)
        self.scene.people.append(p)
        return p

    def turn_to(self, deg_: float, at: float | None = None) -> None:
        """The cart's heading in the map is ``deg_`` from the board's stamp ``at`` (now: the
        machine's last moment) on: TF's history, it turned."""
        self.turns.append((self.tf.now() if at is None else at, deg_))

    def cart_deg(self, stamp: float) -> float:
        """The cart's heading at a board stamp, as TF's history has it."""
        return next((d for t, d in reversed(self.turns) if t <= stamp + 1e-9), 0.0)

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
        """An utterance from ``bearing`` (the cart's frame) read ``read_s`` after it ended
        (GigaAM), the room's word-less readings in between; ``flagged``: only its last that
        many readings carry the speech flag. When it was read."""
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
                if bearing is None:
                    self.doa.readings.append(read_at(now, 30, False))
                else:
                    self.doa.readings.append(read_at(now, array(bearing), True))
            if move is not None:
                move(now)
            if due is not None and now >= due - 1e-9:
                due = self.machine.tick(max(now, due))
        return now

    def aimed(self) -> list[tuple[float, float]]:
        """Every aim asked, as (pan, tilt) degrees rounded, as the arbiter resolved it then."""
        return [(round(deg(a.pan_rad), 1), round(deg(a.tilt_rad), 1)) for a, _ in self.door.aims()]

    def head(self) -> tuple[float, float]:
        """Where the head points now (the arbiter keeps it on the point held)."""
        view = self.door.state()
        assert view is not None and view.head is not None
        return round(deg(view.head.pan_rad), 1), round(deg(view.head.tilt_rad), 1)

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
    **knobs: Any,
) -> Rig:
    """The engine on the fakes; ``knobs`` are the SpeakerSettings a scenario pins over the
    shipped ones (``arc_deg=160.0``, ``seek_pan_deg=20.0``, ``body=BodyLimits(...)``). TF has
    the cart at the map's origin, heading as :meth:`Rig.turn_to` turned it, the board's clock
    BOARD ahead of the machine's own; the lens on the neck's axes (FLAT), so a person's angles
    are exact."""
    door = Door(view=view(head=head))
    doa = FakeDoa()
    holder: list[Rig] = []
    tf = FakeTf(
        FLAT,
        now=lambda: BOARD + holder[0].machine.now_s,
        cart_at=lambda t: (0.0, 0.0, holder[0].cart_deg(t)),
        head_at=lambda t: door.view.head if door.view is not None else None,
        history_s=10.0,  # the goal server's TF_HISTORY_S: an older stamp is refused, too_old
    )
    door.tf = tf
    scene = FakeScene(door, tf)
    face_channel = FakeFace()
    said: list[str] = []
    events: list[tuple[str, float, dict[str, Any]]] = []
    series = FakeSeries()
    wheels = wheels if wheels is not None else NoWheels()
    machine = SpeakerGaze(
        replace(settings if settings is not None else RULES, **knobs),
        door=door,
        doa=doa,
        finder=scene if finder else None,
        wheels=wheels,
        face=face_channel,
        reach=REACH,
        head=FLAT,
        poses=tf,
        report=said.append,
        events=lambda kind, now, **f: events.append((kind, now, f)),
        series=series,
    )
    machine.spawn = lambda call: call()  # an aim sent without waiting: its answer in the inbox
    built = Rig(machine, door, doa, scene, wheels, face_channel, said, events, tf, series)
    holder.append(built)
    return built


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
    2 deg off, close enough: no second move; the first clear face is told to the gesture
    engine (which smiles: the face's one writer), and the gaze smiles only with speaker.smile
    set. The arc of +-80: 60 deg is the head's alone, no wheels in the story."""
    r = rig(arc_deg=160.0)
    r.person(60.0, tilt_deg=2.0)
    read = r.wake(60.0)
    assert r.aimed() == [(60.0, 3.0)] and r.door.eases()[0] == RULES.ease("turn")
    assert r.machine.phase == "seek" and "+60 deg" in r.said[0]
    (turn,) = r.reaims()
    assert turn["why"] == "wake" and turn["speed"] == "turn"
    r.machine.tick(read)
    assert r.machine.phase == "track" and r.aimed() == [(60.0, 3.0)], "within refine: no move"
    assert r.face.calls == [] and r.machine.attention(read).first_face_s is not None
    r.run(4.0, read)
    assert r.scene.looks == 28 and len(r.door.aims()) == 1, "ten frames a second, no aim"
    assert r.kinds()[:4] == ["gaze_doa", "gaze_wake", "gaze_reaim", "gaze_face"]
    smiling = rig(settings=replace(RULES, smile="smile"), arc_deg=160.0)
    smiling.person(60.0, tilt_deg=2.0)
    read = smiling.wake(60.0)
    smiling.machine.tick(read)
    smiling.run(4.0, read)
    assert smiling.face.calls == [("express", ("smile", 1.0, 2.0, None))]


def test_the_wake_turns_to_the_bearing_even_near_where_the_head_points() -> None:
    """cascade_155039: the head at -0.2, the name read at -15: it stayed at -0.2 (the face was
    at -43.5 and was found 18 s later). Now it turns to -15, where the face is in the frame (28
    deg off) and taken at once: within the seek's reach (seek_pan_deg 20 + match_deg 20)."""
    r = rig(head=Aim(rad(-0.2), rad(-5)), seek_pan_deg=20.0)
    r.person(-43.5, tilt_deg=2.8, face_px=65.0)
    read = r.wake(-15.0, peak_db=-22.0)
    assert r.aimed() == [(-15.0, 3.0)]
    r.machine.tick(read)
    assert r.machine.phase == "track" and r.aimed()[-1] == (-43.5, 2.8)
    last = r.reaims()[-1]
    assert (last["why"], last["source"], last["speed"]) == (SIGHTED, "face", "turn")


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
        r = rig(seek_pan_deg=20.0)
        r.person(face_at, tilt_deg=3.0)
        read = r.wake(-15.0, peak_db=-22.0)
        r.run(read + 2.0, read)
        aims = r.aimed()
        assert aims[: 1 + stops] == [(-15.0, 3.0), (-35.0, 3.0), (5.0, 3.0)][: 1 + stops]
        assert all(tilt == 3.0 for _, tilt in aims), "no tilt sweep"
        assert r.machine.phase == "track" and aims[-1] == (face_at, 3.0)
        reaims = r.reaims()
        found = next(i for i, f in enumerate(reaims) if f["why"] == SIGHTED)
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

    r.doa.readings += [read_at(k / 10, array(where(k / 10)), True) for k in range(13)]
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
    pan = r.aimed()[-1][0]
    assert abs(pan - 70.0) <= 12.0, (pan, r.aimed())
    assert r.machine.counts["follows"] >= 3 and VOICED in [f["why"] for f in r.reaims()]
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
    assert r.machine.phase == "track" and r.reaims()[-1]["why"] == SIGHTED
    assert r.reaims()[-1]["source"] == "body"
    face_event = next(f for k, _, f in r.events if k == "gaze_face")
    assert face_event["source"] == "body" and r.face.calls == []
    r.run(read + 3.0, read)
    assert r.machine.phase == "track" and len(r.door.aims()) == 2


def test_a_second_speaker_takes_the_gaze_and_the_first_takes_it_back_at_once() -> None:
    """Artem tracked at +60; Masha, never seen, speaks from -40: a check (one frame) finds
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
    masha = r.machine.person
    read = r.say(65.0, 5.0, 6.0, "Она шутит")
    assert r.aimed()[-1] == (65.0, 2.0) and not r.machine.verifying
    assert "a person seen there" in r.said[-1]
    r.machine.tick(read)
    assert r.machine.phase == "track" and r.aimed()[-1] == (60.0, 2.0)
    r.say(-38.0, 8.0, 9.0, "Ничего я не шучу")
    assert r.machine.person is masha and r.aimed()[-1] == (-38.0, 6.0), "her bearing, her tilt"


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
    assert len(r.machine.people.people) == 2, "Artem, and a voice nobody was seen at"


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


def test_the_head_never_follows_its_own_whine_the_chip_calls_speech() -> None:
    """2026-10-14: while Artem spoke unseen, the neck's whine came back speech-flagged from 30 deg
    off him and the head followed it. Silero reads the whine 0.1: no follow; with doa_tags chip
    (the rule before) the head goes after the whine as it did."""

    def whine(r: Rig) -> list[float]:
        r.person(10.0, face_px=0.0)
        read = r.wake(10.0)
        r.machine.speech_started(1.35, 1.4)
        r.series.spans.append((1.95, 3.0, 0.1))  # Silero on the whine
        r.run(4.1, read, voice=lambda t: -20.0 if 2.0 <= t <= 3.0 else 10.0)
        return [p for p, _ in r.aimed()]

    silero = rig()
    assert all(abs(p - 10.0) < 1.0 for p in whine(silero)), silero.aimed()
    assert silero.machine.counts["follows"] == 0
    old = rig(doa_tags="chip")
    assert any(abs(p + 20.0) < 3.0 for p in whine(old)) and old.machine.counts["follows"] >= 1


def test_while_the_robot_speaks_frames_come_at_10_hz_and_only_2_s_unseen_is_a_loss() -> None:
    """Ten frames a second while the robot speaks, as while it listens (2026-10-14: at 1 Hz a
    person walking was lost; 2026-10-12 at 1 Hz: lost/found every 2 s). A second of missed
    detections is no loss; face_lost_s of them (and face_lost_frames) are: the head stays."""
    r = tracked(60.0)
    artem = r.scene.people[0]
    r.machine.speaking(True, 1.4)
    r.machine.voiced(1.45)
    looks = r.scene.looks
    now = r.run(3.0, 1.45)
    assert r.scene.looks - looks >= 14, "ten a second while the robot speaks"
    artem.face_px = 0.0  # a second of frames misses him
    now = r.run(4.0, now)
    artem.face_px = 60.0
    now = r.run(6.0, now)
    assert r.machine.phase == "track" and r.machine.misses == 0
    artem.face_px = 0.0
    now = r.run(7.8, now)
    assert r.machine.phase == "track" and r.machine.misses >= 15, "1.8 s unseen: kept"
    r.run(8.2, now)
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


def test_past_the_arc_the_wheels_start_with_the_head_and_tf_keeps_the_head_on_his_point() -> None:
    """A call from 100 deg left (past the forward arc's 80), the caller's face turned away: the
    wheels are started before the head's aim is asked (their checks take longer than its turn),
    neither waits for the other, and the aim (his voice's point in the map) goes out without
    waiting for the arbiter (status sent, the answer through the inbox). Every record is taken
    while the body turns and none shows him; as TF turns the cart at 30 deg/s the arbiter keeps
    the head on his point (the pan comes round 100, 97, 94, 91) and the gaze sends no aim at
    all (no odometry, no glide: 2026-10-15). The turn's end reads the cart's pose again: he is
    at 70, within the arc, and the seek goes on there."""
    r = rig(wheels=FakeWheels(), arc_deg=160.0, body=BodyLimits(max_turn_deg=30.0))
    r.person(100.0, face_px=0.0)
    read = r.wake(100.0)
    (start,) = r.wheels.calls
    assert start == ("start", (WheelMove(turn_deg=pytest.approx(30.0)),))
    assert r.aimed()[0] == (100.0, 3.0) and r.machine.turning is not None
    assert r.reaims()[0]["status"] == "sent" and r.machine.ours, "not waited for"
    assert r.reaims()[0]["source"] == "voice" and len(r.reaims()[0]["point"]) == 3
    due = r.machine.tick(read)
    assert due == pytest.approx(read + 0.1) and r.scene.looks == 1, "a record, nobody in it"
    assert r.machine.inbox == deque() and r.machine.counts["aims"] == 1, "its answer taken"
    heads = []
    for k in range(1, 4):  # TF: the cart at 30 deg/s
        r.turn_to(3.0 * k)
        r.machine.tick(read + 0.1 * k)
        heads.append(r.head()[0])
    assert heads == [97.0, 94.0, 91.0], "the arbiter keeps the head on his point"
    assert len(r.door.aims()) == 1 and r.machine.counts["aims"] == 1, "no aim of the gaze's"
    assert r.scene.looks == 4 and r.machine.counts["frames_turning"] == 4, "every record"
    r.turn_to(30.0)
    r.wheels.finish()
    r.machine.tick(read + 0.8)
    assert r.machine.bearing == pytest.approx(rad(70)), "the turn's end: the cart's pose read"
    assert r.machine.turning is None and r.scene.looks == 5, "the frames go on"
    assert r.aimed()[-1] == (100.0, 3.0), "nobody seen there: the seek's first stop, outward"
    assert len(r.wheels.calls) == 1, "back within the arc: no second turn"


def test_aims_sent_without_waiting_are_answered_through_the_inbox_the_newest_counting() -> None:
    """An aim sent without waiting (while the body turns, and every pursuit's) is answered
    later: done and preempted (by our own newer aim) change nothing; queued on the newest says
    another look holds the head (the wheels let go, as a waited-for queued does); denied on the
    newest loses the head; an older aim's late denied is ignored; past AIMS_IN_FLIGHT
    unanswered aims none goes out; a door that throws off the thread is a denied; the shipped
    spawn runs the call on a thread of its own."""
    from pepin.expression.tracker import AIMS_IN_FLIGHT, _on_a_thread

    r = rig(wheels=FakeWheels())  # the arc's +-30: Artem at 150 turns the body
    artem = r.person(150.0)
    read = r.wake(150.0)
    r.machine.tick(read)
    assert r.machine.phase == "track" and r.machine.turning is not None
    r.door.answers.append(GazeAnswer("preempted", "by a newer request of voice.speaker"))
    artem.pan_deg = 160.0  # he steps aside: a pursuit's aim
    r.machine.tick(read + 0.1)
    r.machine.tick(read + 0.2)
    assert r.machine.ours and r.machine.turning is not None and len(r.wheels.calls) == 1
    r.door.answers.append(GazeAnswer("queued", "held behind look"))
    artem.pan_deg = 145.0
    r.machine.tick(read + 0.3)  # the aim goes out, its queued answer lands in the inbox
    r.machine.tick(read + 0.4)
    assert r.wheels.calls[-1][0] == "halt" and r.machine.turning is None
    assert any("the head waits: queued" in line for line in r.said)
    late = rig(wheels=FakeWheels())
    read = late.wake(150.0)
    newest = late.machine.aim_id
    late.machine.wheels_event(("aimed", newest - 1, GazeAnswer("denied", "stale")), read + 0.1)
    assert late.machine.aim is not None and late.machine.phase != "voice", "an older aim's word"
    late.machine.wheels_event(("aimed", newest, GazeAnswer("denied", "no head")), read + 0.1)
    assert late.machine.aim is None and late.machine.phase == "voice", "the newest: lost"
    assert late.wheels.calls[-1] == ("halt", ("the head is not this source's",))
    stuck = rig()
    stuck.machine.spawn = lambda call: None  # the door never answers what goes without waiting
    walker = stuck.person(10.0)
    read = stuck.wake(10.0)
    stuck.machine.tick(read)
    assert stuck.machine.phase == "track"
    for k in range(1, AIMS_IN_FLIGHT + 3):
        walker.pan_deg = 18.0 if k % 2 else 10.0  # past the dead band every record
        stuck.machine.tick(read + 0.1 * k)
    sent = [f["status"] for f in stuck.reaims()]
    assert sent.count("sent") == AIMS_IN_FLIGHT and sent[-1] == "unsent"
    assert stuck.machine.counts["aims_unsent"] >= 1

    class Broken(Door):
        def aim(self, point: Point, *, speed: Speed, ttl_s: float, ease: Ease | None = None) -> Any:
            raise OSError("the door fell off")

    broken = rig(wheels=FakeWheels())
    broken.machine.door = Broken(view=view(), tf=broken.tf)
    read = broken.wake(150.0)
    broken.machine.tick(read)
    assert broken.machine.phase == "voice" and broken.machine.aim is None, "denied, off-thread"
    threaded = rig(wheels=FakeWheels())
    threaded.machine.spawn = _on_a_thread  # the shipped one
    threaded.wake(150.0)
    deadline = time.monotonic() + 2.0
    while not threaded.machine.inbox and time.monotonic() < deadline:
        time.sleep(0.005)
    assert threaded.machine.inbox and threaded.machine.inbox[0][0] == "aimed"


def test_a_face_past_straight_back_keeps_the_head_at_its_end_and_never_whips_across() -> None:
    """2026-10-13 17:51 (cascade_175115): Artem behind the robot, the head at the left end;
    the finder saw his face 20 deg left of the head, i.e. +184, which the wrap reads as -176 and
    the plain clamp sent to the RIGHT end: 320 deg through the front at full speed, for the same
    person. The end nearer the head wins: every aim stays on the left, within the reach."""
    r = rig(head=Aim(rad(153.0), TILT))
    r.person(184.0)  # a hair past straight back, on the left: past the left end (+164)
    read = r.wake(179.3)
    r.run(read + 3.0, read)
    pans = [p for p, _ in r.aimed()]
    assert pans and all(0.0 < p <= deg(REACH.pan[1]) + 0.1 for p in pans), pans
    assert r.scene.looks >= 1, "the person was looked for"
    assert r.machine.phase == "track" and r.machine.person is not None
    assert all(abs(a - b) < 30.0 for a, b in itertools.pairwise(pans)), pans


def test_a_tool_s_face_turns_the_body_the_whole_way_past_the_arc_rule_and_a_turn_that_waited() -> (
    None
):
    """The approach tool, nobody in view: face() turns the body to the person engaged in one
    request whatever the arc, even after a turn that fell short (body_waits), and answers why
    not when there is no conversation, no person, or a turn already runs; paused by a drive
    call, it says whose (2026-10-14: "a drive owns the wheels" was read as another trip, when
    the drive was approach's own call)."""
    r = rig(wheels=FakeWheels(), arc_deg=360.0)  # the arc never turns the body by itself
    assert r.machine.face_person(0.0) == "no conversation"
    read = r.wake(-150.0)
    r.machine.tick(read)
    assert r.wheels.calls == [], "within the arc: the head alone"
    r.machine.body_waits = True  # a turn that fell short waits for the next utterance
    assert r.machine.face_person(read + 0.1) is None
    (start,) = r.wheels.calls
    assert start == ("start", (WheelMove(turn_deg=pytest.approx(-150.0)),))
    assert r.machine.turning is not None and r.machine.body_waits is False
    assert r.machine.face_person(read + 0.2) is None, "one already runs: fine"
    assert len(r.wheels.calls) == 1
    r.machine.drive(True, read + 0.3)  # the loop tells it at a drive tool's call, approach's too
    own = r.machine.face_person(read + 0.4)
    assert own is not None and own.startswith("the robot's own drive call (this one)"), own
    r.machine.drive(False, read + 0.5)
    after = r.machine.face_person(read + 0.6)
    assert after is not None and after.startswith("the gaze waits for the person's next"), after


def test_an_utterance_spoken_across_a_turn_gives_the_bearing_the_person_has_now() -> None:
    """The array turns with the cart: Artem at 100 deg keeps talking while the body turns 30
    toward him, so the array reads him sweeping 100 -> 70 (the utterance's last second averages
    ~78). Each reading is taken into the map by TF's heading at its own stamp (one board clock):
    he is at 100 in the room, at 70 from the cart as it stands at his last word, and the voice
    stays his (no hunting)."""
    r = rig(wheels=FakeWheels())
    read = r.wake(100.0)
    assert r.machine.turning is not None
    start, end = read + 0.1, read + 2.0

    def yaw(stamp: float) -> float:
        return 30.0 * min(max((stamp - BOARD - start) / (end - start), 0.0), 1.0)

    r.tf.cart_at = lambda t: (0.0, 0.0, yaw(t))  # TF's heading history
    for k in range(round((end - start) * 10) + 1):
        t = start + k / 10
        r.doa.readings.append(read_at(t, array(100.0 - yaw(BOARD + t)), True))
    r.doa.readings.append(read_at(end + 0.1, 30, False))
    r.machine.heard(Heard(start, end, "я здесь", False, -12.0), end + 0.1)
    (voice,) = [f for k, _, f in r.events if k == "gaze_voice"]
    assert voice["bearing_deg"] == pytest.approx(70.0, abs=1.0), voice
    assert voice["azimuth_deg"] == pytest.approx(100.0, abs=1.0), "in the room"
    assert voice["verdict"] == "the person engaged" and voice["track"] == 1
    assert r.machine.bearing == pytest.approx(rad(70.0), abs=rad(0.5)), "from the cart now"
    doa = [f for k, _, f in r.events if k == "gaze_doa"][-1]
    rows, voiced = doa["readings"], doa["voiced"]
    first = voiced.index(1)  # the wheels' first jerk is no voice: no pose asked for it
    stamp = BOARD + end + rows[first][0]  # its seconds before the utterance's end
    assert rows[first][3] == pytest.approx(30.0 - yaw(stamp), abs=0.5), "TF's turn since"
    assert rows[first][3] > 15.0 and rows[-1][3] == pytest.approx(0.0)
    window = utterance_readings(r.doa, Heard(start, end, "я здесь"), end + 0.1, 1.0)
    unturned = voice_of(chip(window), MAP)
    assert unturned.bearing_rad is not None and deg(unturned.bearing_rad) > 75.0, "uncorrected"


def test_a_voice_stamped_on_the_board_s_uptime_is_asked_of_tf_at_its_wall_clock_moment() -> None:
    """2026-10-15 evening, 19 of 19 utterances: "no place (base_link: too_old: the stamp is
    1792080462 s old)": the array's stamps are the board's uptime (33,450 s) and TF keeps its
    wall clock. Every reading carries the audio link's offset, so the gaze asks TF (here as the
    goal server answers, 10 s of history) at each reading's ``wall_s``: Artem behind on the
    right (-120 deg, run 143's) is placed and engaged, and the board's "now" is the wall
    clock's. A reading without the offset (an audio server before 2026-10-16) is never asked at
    its monotonic stamp, the verdict says why; one truly 20 s old is still refused as too old."""
    wall = 1_792_113_912.0  # the board's wall clock (TF's) at the machine's 0
    offset = wall - UPTIME
    r = rig(finder=False, arc_deg=360.0)
    r.tf.now = lambda: wall + r.machine.now_s

    def said(bearing: float, start: float, end: float, link: float | None) -> list[DoaReading]:
        steps = round((end - start) * 10)
        voice = [DoaReading(UPTIME + start + k / 10, array(bearing), True, link)
                 for k in range(steps + 1)]  # fmt: skip
        return [*voice, DoaReading(UPTIME + end + 0.1, 30, False, link)]

    r.doa.readings += said(-120.0, 0.0, 1.2, offset)
    r.machine.opened(Heard(0.0, 1.2, "Пепин", named=True), 1.3)
    (wake,) = [f for k, _, f in r.events if k == "gaze_wake"]
    assert wake["azimuth_deg"] == pytest.approx(-120.0, abs=1.0), wake
    assert r.machine.person is not None and r.machine.person.source == "voice"
    assert r.aimed()[-1][0] == pytest.approx(-120.0, abs=1.0), "the head on him"
    asked = [s for stamps, _ in r.tf.asked for s in stamps if s is not None]
    assert asked and all(wall <= s <= wall + 1.3 for s in asked), "TF's clock, never the uptime"
    assert r.machine.person.stamp == pytest.approx(wall + 1.2), "the voice's place, its stamp"
    board = r.machine.clock.now()
    assert board is not None and wall + 1.2 <= board <= wall + 1.3, "the board's now: wall"
    r.doa.readings += said(-100.0, 2.0, 3.0, None)  # an old audio server: no offset
    before = len(r.tf.asked)
    r.machine.heard(Heard(2.0, 3.0, "я тут", False, -12.0), 3.1)
    verdict = [f for k, _, f in r.events if k == "gaze_voice"][-1]["verdict"]
    assert verdict == f"no place ({NO_WALL})" and len(r.tf.asked) == before, "nothing asked"
    r.doa.readings += said(-100.0, 4.0, 5.0, offset - 20.0)  # their moments 20 s ago
    r.machine.heard(Heard(4.0, 5.0, "я тут", False, -12.0), 5.1)
    verdict = [f for k, _, f in r.events if k == "gaze_voice"][-1]["verdict"]
    assert (
        verdict == "no place (base_link: too_old: the stamp is 21.1 s old and TF keeps 10 s of"
        " history)"
    ), verdict


def test_a_person_far_behind_turns_the_body_again_until_they_are_inside_the_arc() -> None:
    """Behind at 150 deg: a request's worth (30) at a time, the next asked when the last ends
    (the cart's pose read again), until the person is within the forward arc (+-80: 150 -> 120
    -> 90 -> 60), the head on his point all the while (the arbiter's; no eyes here: no seek's
    stop moves it)."""
    r = rig(wheels=FakeWheels(), finder=False, arc_deg=160.0, body=BodyLimits(max_turn_deg=30.0))
    read = r.wake(150.0)
    heads = [r.head()[0]]
    for k in range(3):
        assert r.wheels.calls[-1] == ("start", (WheelMove(turn_deg=pytest.approx(30.0)),))
        r.turn_to(30.0 * (k + 1))
        r.wheels.finish()
        r.machine.tick(read + 0.1 * (k + 1))
        heads.append(r.head()[0])
    assert len(r.wheels.calls) == 3 and r.machine.turning is None
    assert r.machine.bearing == pytest.approx(rad(60))
    assert heads == [150.0, 120.0, 90.0, 60.0], "the head came round"
    assert len(r.door.aims()) == 1, "the one point, kept by the arbiter"


def test_a_person_tracked_who_walks_past_the_arc_turns_the_body_toward_them() -> None:
    """One rule, wherever the person's point comes from: Artem tracked at 70 deg walks round to
    95; the frame that places him past the arc (+-80) starts the turn (a request's worth, 30),
    the head already on him."""
    r = rig(wheels=FakeWheels(), arc_deg=160.0, body=BodyLimits(max_turn_deg=30.0))
    artem = r.person(70.0)
    read = r.wake(70.0)
    now = r.run(read + 1.0, read)
    assert r.machine.phase == "track" and r.wheels.calls == [], "within the arc: the head alone"

    def walk(t: float) -> None:
        artem.pan_deg = min(95.0, 70.0 + 25.0 * (t - now))  # 25 deg/s round to 95

    r.run(now + 2.0, now, move=walk)
    (start,) = r.wheels.calls
    assert start[0] == "start" and start[1][0].turn_deg == pytest.approx(30.0)
    assert r.machine.bearing is not None and deg(r.machine.bearing) > 80.0


def test_wheels_that_turn_less_leave_the_head_alone_and_wait_for_the_next_utterance() -> None:
    """A refusal (nothing moved) or a cut turn is asked again at the person's next utterance; a
    turn that moved the cart and still fell short (stalled, blocked, the back-offs spent) is
    not asked again for that person in this conversation (no creeping back-offs). The turns
    asked are a request's worth of 30."""
    stub = rig(body=BodyLimits(max_turn_deg=30.0))
    read = stub.wake(100.0)
    stub.machine.tick(read)  # NoWheels answered at once; the answer is taken on the next tick
    assert stub.aimed()[0] == (100.0, 3.0) and "the wheels turned +0 of +30 deg" in stub.said[1]
    assert stub.machine.body_waits, "refused: until the next utterance"
    behind = rig()
    behind.wake(180.0)  # straight behind: past the neck's reach
    assert behind.door.aims()[0][0].pan_rad == pytest.approx(REACH.pan[1])
    # the seek's 20 deg sweep goes on after the cut turn's word (without it the frame that
    # finds nobody ends the seek at once, and its line is the last said)
    cut = rig(wheels=FakeWheels(), seek_pan_deg=20.0, body=BodyLimits(max_turn_deg=30.0))
    cut.wake(100.0)
    cut.turn_to(12.0)  # TF: the cart turned 12 of the 30
    cut.wheels.finish(WheelReport(False, 12.0, why="superseded by a newer motion", cut=True))
    cut.machine.tick(1.4)
    assert cut.head() == (108.0, 3.0) and cut.machine.body_waits, "then the seek goes on"
    assert cut.aimed()[1] == (108.0, 3.0), "its first stop, outward of his 88 from the cart"
    assert "turned +12 of +30 deg (superseded by a newer motion)" in cut.said[-1]
    cut.machine.tick(1.5)
    assert len(cut.wheels.calls) == 1, "it waits for the next utterance"
    cut.say(88.0, 3.0, 4.0, "ну что")
    cut.machine.tick(4.2)
    assert [c[0] for c in cut.wheels.calls] == ["start", "start"], "the rest of the turn"
    stuck = rig(wheels=FakeWheels())
    stuck.wake(100.0)
    blocked = WheelReport(False, 6.0, 0.06, 3, "gave up: the arc is blocked")
    stuck.turn_to(6.0)
    stuck.wheels.finish(blocked)
    stuck.machine.tick(1.4)
    stuck.say(94.0, 3.0, 4.0, "эй")
    stuck.machine.tick(4.2)
    assert len(stuck.wheels.calls) == 1, "gave up for this person"


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
    assert r.machine.people.people == [], "the cart moves: every bearing goes stale"
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
    assert len(r.machine.people.people) == 1
    r.machine.stop(9.0, "the switch is off")
    assert r.machine.people.people == []


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


def test_tracking_re_aims_past_the_dead_band_and_a_frame_without_its_stamp_gives_no_speed() -> None:
    """Standing, the head stays still. This scene's frames carry no exposure stamp: each is
    placed with the head's pose of its read and gives no speed, so a walk at 15 deg/s is
    re-aimed past the dead band (6 deg), without a lead, every aim sent without waiting (the
    pursuit of a walk on stamped frames: test_speaker_gaze_pursuit.py)."""
    r = tracked(60.0)
    artem = r.scene.people[0]
    now = r.run(2.3, 1.3)
    assert len(r.door.aims()) == 1, "still: no aim"

    def walk(t: float) -> None:
        artem.pan_deg = 60.0 + 15.0 * (t - 2.3)

    r.run(3.6, now, move=walk)
    rows = r.reaims()[1:]
    assert rows and {f["lead_deg"] for f in rows} == {0.0}, rows
    assert {f["speed"] for f in rows} == {"adjust"} and {f["status"] for f in rows} == {"sent"}
    assert rows[-1]["why"] == MOVED and r.machine.counts["pursuits"] == 0
    assert r.machine.counts["frames_unposed"] >= 13 and r.machine.counts["frames_posed"] == 0


def test_a_tracked_person_s_frames_go_on_while_the_head_moves_and_wait_for_a_gesture() -> None:
    """Tracking, the frames go on while the head moves (blind: each placed with the head's pose
    at its exposure); a gesture's frame is no look (the head is pulled away), the idle layer's
    breath's is; a seek's frame still waits for a still head (test_perception: the record
    exposed after it settled)."""
    r = tracked(60.0)
    looks = r.scene.looks
    r.door.view = replace(r.door.view, blind=True) if r.door.view else None
    r.machine.tick(1.4)
    assert r.scene.looks == looks + 1 and r.machine.next_frame == pytest.approx(1.5)
    r.door.view = replace(r.door.view, blind=False, overlay=GESTURE) if r.door.view else None
    r.machine.tick(1.5)
    assert r.machine.counts["frames_skipped_gesture"] == 1, "a thinking glance: no look"
    assert r.machine.next_frame == pytest.approx(1.6), "tried again right after"
    r.door.view = replace(r.door.view, overlay=IDLE_LOOK) if r.door.view else None
    frames = r.machine.counts["frames"]
    r.machine.tick(1.6)
    assert r.machine.counts["frames"] == frames + 1 and r.machine.phase == "track"
    seek = rig()
    seek.person(60.0)
    seek.door.view = replace(seek.door.view, blind=True) if seek.door.view else None
    seek.machine.tick(seek.wake(60.0))
    assert seek.scene.looks == 0 and seek.machine.phase == "seek", "a seek waits for still"


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
def test_no_frame_from_a_reply_s_first_part_until_its_first_chunk_then_ten_a_second() -> None:
    """The voice's first chunk shares the Mac with the gaze: from the first part queued until
    the first audio no frame is taken (renewals go on); then frames come ten a second while the
    robot speaks as while it listens (shipped since 2026-10-14); the knob still slows them
    (track_speaking_hz 1: once a second, ten a second again the moment it stops)."""
    r = tracked(60.0)
    assert r.machine.next_frame == pytest.approx(1.4)
    r.machine.speaking(True, 1.35)
    looks = r.scene.looks
    assert r.machine.tick(1.4) == pytest.approx(3.3), "nothing due but the renewal"
    for t in (1.5, 1.7, 2.2):
        r.machine.tick(t)
    assert r.scene.looks == looks and r.machine.voice_hold
    r.machine.voiced(2.25)
    assert r.machine.tick(2.25) == pytest.approx(2.35)
    assert r.scene.looks == looks + 1, "the frame that waited, at once"
    for t in (2.35, 2.45, 2.55):
        r.machine.tick(t)
    assert r.scene.looks == looks + 4, "ten a second while the robot speaks"
    r.machine.speaking(False, 2.6)
    r.machine.speaking(True, 5.0)
    assert r.machine.voice_hold
    r.machine.speaking(False, 5.5)
    assert not r.machine.voice_hold, "a reply cut before its first chunk ends the hold"
    assert r.machine.counts["voice_holds"] == 2
    slow = tracked(60.0)
    slow.machine.configure(replace(slow.machine.settings, track_speaking_hz=1.0))
    slow.machine.speaking(True, 1.35)
    slow.machine.voiced(1.36)
    looks = slow.scene.looks
    for t in (1.4, 1.5, 1.9):
        slow.machine.tick(t)
    assert slow.scene.looks == looks + 1, "one a second at the knob's 1 Hz"
    slow.machine.speaking(False, 1.95)
    assert slow.machine.next_frame == pytest.approx(2.05), "listening: ten a second at once"


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
def test_the_voice_rule_is_speaker_doa_s_and_each_reading_s_tags_are_logged() -> None:
    """speaker.doa ships doa_tags silero and voice_p_min 0.5 (2026-10-14); another rule or a
    probability past 0..1 is refused; gaze_doa carries every reading's Silero probability and
    the rule's verdict beside the array's own flag."""
    shipped = SpeakerSettings.load(REPO / "config/expression.json")
    assert (shipped.doa_tags, shipped.voice_p_min) == ("silero", 0.5)
    knobs = {"doa_tags": "chip", "voice_p_min": 0.3}
    other = SpeakerSettings.from_config({"speaker": {"doa": knobs}})
    assert (other.doa_tags, other.voice_p_min) == ("chip", 0.3)
    for bad in ({"doa_tags": "vad"}, {"voice_p_min": 1.5}):
        with pytest.raises(ValueError, match="doa_tags"):
            SpeakerSettings.from_config({"speaker": {"doa": bad}})
    r = rig()
    r.series.spans.append((0.0, 0.45, 0.2))  # the name's first syllables read low
    r.wake(40.0)
    (logged,) = [f for k, _, f in r.events if k == "gaze_doa"]
    assert logged["tags"] == "silero" and len(logged["voice_p"]) == len(logged["readings"])
    assert logged["voice_p"][:6] == [0.2] * 6 and logged["voice_p"][-1] == 0.9
    assert logged["voiced"] == [0] * 6 + [1] * 7, "Silero under 0.5 on the first six"
    assert [row[2] for row in logged["readings"]] == [1] * 13, "the array's flag as read"
    assert "moving" not in logged
    deaf = rig()
    deaf.machine.series = None  # the ears run no speech model (barge_in.detector level)
    deaf.wake(40.0)
    assert any("no speech model in the ears" in line for line in deaf.said)
    built = rig().machine
    built.series = None  # robot_gaze builds it without: the worker hands the ears' over
    worker = SpeakerGazeWorker(lambda s: built, on=False, load=lambda: RULES)
    ears = FakeSeries()
    worker.series = ears  # the cascade wires the ears' series once its loop is built
    assert worker.switch(True) is None and worker.idle.wait(2.0) and built.series is ears
    worker.series = None
    assert built.series is None and worker.series is None
    worker.close()


def test_the_shipped_knobs_load_off_and_measured_on_the_four_readings() -> None:
    settings = SpeakerSettings.load(REPO / "config/expression.json")
    assert settings.doa_map == TODAY.doa_map and settings.ttl_s == 6.0 and settings.renew_s == 2.0
    assert settings.body.max_turn_deg == 180.0 and settings.min_speech == 2
    assert settings.mac_doa_map is not None and settings.mac_doa_map.front_deg == 182.3
    assert settings == replace(
        SpeakerSettings(), doa_map=settings.doa_map, body=settings.body,
        mac_doa_map=settings.mac_doa_map,
    )  # fmt: skip
    assert (settings.seek_tilt_deg, settings.seek_pan_deg, settings.seek_frames) == (3.0, 30.0, 1)
    assert (settings.doa_min_speech, settings.doa_min_dbfs) == (6, -18.0)
    assert (settings.face_lost_frames, settings.face_lost_s) == (3, 2.0)
    assert (settings.turn_acc_deg_s2, settings.adjust_acc_deg_s2) == (650.0, 400.0)
    assert (settings.doa_follow_s, settings.doa_moved_deg, settings.follow_max_deg) == (
        0.5,
        8.0,
        35.0,
    )
    assert settings.glide_gain == 2.0
    assert (settings.default_range_m, settings.people_half_life_s) == (2.0, 60.0)
    assert settings.body_turn == "spin", "2026-10-14, Artem: Nav2's Spin turns the body"
    assert flag_default(REPO / "config/expression.json") is False
    bad_knobs = ({"track_hz": 0}, {"max_tracks": 0}, {"lead_max_deg": -1},
                 {"track_speaking_hz": 0}, {"face_lost_frames": 0}, {"turn_speed_deg_s": 0},
                 {"doa_follow_s": 0}, {"glide_gain": 0.5}, {"default_range_m": 0},
                 {"people_half_life_s": -1}, {"body_turn": "nav2"})  # fmt: skip
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


# ---- the gesture engine reads attention; the body's turn asks for the arm ----------
def test_a_frame_a_gesture_skips_spends_no_seek_stop_and_no_check() -> None:
    """2026-10-12 critic: a frame skipped while a gesture overlay played counted as a miss, so a
    seek swept away from the caller while the hand waved and a check went back unlooked. Now it
    is no look: counted by its cause and taken again BLIND_RETRY_S later (the seek's sweep of
    20 deg then finds the face)."""
    r = rig(seek_pan_deg=20.0)
    r.person(-50.0, tilt_deg=3.0)
    read = r.wake(-15.0, peak_db=-22.0)
    assert r.machine.phase == "seek" and r.machine.attention(read).phase == "seek"
    r.door.view = replace(r.door.view, overlay=GESTURE) if r.door.view else None
    for k in range(10):
        r.machine.tick(read + 0.1 * k)
    assert r.scene.looks == 0 and r.aimed() == [(-15.0, 3.0)], "no stop spent, no sweep"
    assert r.machine.counts["frames_skipped_gesture"] == 10
    assert r.machine.next_frame == pytest.approx(read + 0.9 + 0.1)
    r.door.view = replace(r.door.view, overlay="") if r.door.view else None
    r.run(read + 3.0, read + 1.0)
    assert r.machine.phase == "track" and r.aimed()[-1] == (-50.0, 3.0)


def test_attention_tells_the_engine_its_phase_whom_it_saw_when_and_the_arc() -> None:
    """The arc of +-80: 60 deg is within it, 150 past it."""
    r = rig(wheels=FakeWheels(), arc_deg=160.0)
    assert r.machine.attention(0.0).phase == "idle"
    r.person(60.0)
    read = r.wake(60.0)
    seek = r.machine.attention(read)
    assert seek.phase == "seek" and seek.eyes and seek.seen_s is None and not seek.past_arc
    r.machine.tick(read)
    seen = r.machine.attention(read)
    assert seen.phase == "track" and seen.seen_s == read and seen.first_face_s == read
    assert seen.bearing_deg == pytest.approx(60.0) and seen.since_s == read
    r.say(-60.0, 3.0, 4.0, "в эфире новости")
    assert r.machine.attention(4.1).phase == "check"
    behind = rig(wheels=FakeWheels(), arc_deg=160.0)
    behind.person(150.0)
    turning = behind.machine.attention(behind.wake(150.0))
    assert turning.phase == "turning" and turning.past_arc


class Arm:
    """The gesture engine as the gaze asks it: the arm rests in ``seconds``."""

    def __init__(self, seconds: float) -> None:
        self.seconds, self.asked = seconds, 0

    def release_arm(self, why: str) -> float:
        self.asked += 1
        return self.seconds


def test_a_wake_from_behind_rests_the_arm_first_then_turns_the_body() -> None:
    """«Пепин, привет!» from behind while the hand is up (a greeting the engine started): no
    arm while the body turns (attention's rule), so the turn asks for the arm first and starts
    once it has had the seconds the engine answered; the frames go on meanwhile, and find him."""
    r = rig(wheels=FakeWheels())
    arm = Arm(0.9)
    r.machine.arm = arm
    r.person(150.0)
    read = r.wake(150.0)
    assert arm.asked == 1 and r.wheels.calls == [], "the arm first"
    turning = r.machine.turning
    assert turning is not None and turning.starts_s == pytest.approx(read + 0.9)
    assert r.machine.tick(read + 0.5) == pytest.approx(read + 0.6), "the next record"
    assert r.wheels.calls == [] and r.scene.looks == 1 and r.machine.phase == "track"
    assert r.machine.tick(read + 0.8) == pytest.approx(read + 0.9), "the turn's start"
    r.machine.tick(read + 0.9)
    whole = WheelMove(turn_deg=pytest.approx(150.0))  # the whole turn in one request, as shipped
    assert r.wheels.calls == [("start", (whole,))]
    calm = rig(wheels=FakeWheels())
    calm.machine.arm = Arm(0.0)
    calm.person(150.0)
    calm.wake(150.0)
    assert len(calm.wheels.calls) == 1, "an arm at rest: the turn at once"


def test_the_engine_lets_the_turn_go_first_on_a_wake_from_behind_with_a_greeting() -> None:
    """The same wake with the engine itself: the arm of a gesture playing goes to rest when the
    turn asks, and while the body turns the reply's greeting marker plays its face but neither
    the arm nor the head."""
    from pepin.arm_motion import load_arm_motion
    from pepin.expression.animator import ARM_REST_S, EngineConfig, GestureEngine
    from pepin.expression.book import load_book, with_rest
    from pepin.expression.fakes import FakeExecutor, FakePlayout
    from pepin.expression.markers import Marker, Part

    motion = load_arm_motion(REPO / "config/arm.json")
    book = with_rest(load_book(REPO / "config/animation.json"), motion.rest_deg)
    clock = [1.0]
    r = rig(wheels=FakeWheels())
    records: list[tuple[str, dict[str, Any]]] = []

    class View:
        def attention(self) -> Any:
            return r.machine.attention(clock[0])

    for seed in range(50):  # a shrug whose variant raises the arm
        parts = {p: FakeExecutor(p) for p in ("arm", "head", "face")}  # type: ignore[arg-type]
        records.clear()
        config = EngineConfig(arm_speed_scale=1.0, seed=seed)
        engine = GestureEngine(
            parts,  # type: ignore[arg-type]
            FakePlayout(), lambda c=config: (book, c), on=True, clock=lambda: clock[0],
            record=lambda kind, fields: records.append((kind, fields)), attention=View(),
        )  # fmt: skip
        assert engine.perform("shrug", source="tool").ok
        last = engine.last()
        if last is not None and any(c.part == "arm" for c in last.clips):
            break
    for _ in range(10):  # the hand goes up
        clock[0] += 0.01
        engine.step()
    assert [c for c in parts["arm"].calls if c[0] == "play"]
    r.machine.arm = engine
    r.person(150.0)
    read = r.wake(150.0)
    clock[0] = read
    assert ("fade", ("gesture", 0.35)) in parts["arm"].calls and r.wheels.calls == []
    since = len(records)
    engine.part_queued(1, 0, Part("Привет!", None, (Marker("gesture", "greeting", 0),)))
    engine.part_spoken(1, 0)
    for _ in range(300):
        clock[0] += 0.01
        engine.step()
    cues = [f for k, f in records[since:] if k == "cue"]
    assert cues and all(not c["ok"] for c in cues if c["part"] in ("arm", "head")), cues
    assert any(c["ok"] for c in cues if c["part"] == "face")
    r.machine.tick(read + ARM_REST_S + 0.06)
    assert len(r.wheels.calls) == 1, "the turn after the arm's rest"


# ---- a turn in place the gaze did not start (the turn tool's Nav2 Spin) ----------
def test_a_turn_in_place_gets_the_drive_s_pause_back_and_tf_keeps_the_head_on_the_person() -> None:
    """Artem's "повернись ко мне": the cascade tells the turn tool's call as a drive (the
    pause, the people put aside); the tool asks the gaze to keep the person: they come back
    with the phase and the point, the snapshot says turning at their bearing, and as Nav2 turns
    the cart (TF) every record is taken while the arbiter keeps the head on his point: no aim
    of the gaze's (he stands), the gaze's own wheels never asked; the drive's end keeps him (no
    reorientation) and the frames go on."""
    r = rig(wheels=FakeWheels(), arc_deg=360.0)  # the arc never turns the body by itself
    r.person(60.0)
    read = r.wake(60.0)
    r.machine.tick(read)
    assert r.machine.phase == "track"
    r.machine.drive(True, read + 0.1)
    assert r.machine.phase == "paused" and r.machine.person is None, "a drive as ever"
    assert r.machine.keep_person(read + 0.15, "turn: me") is None
    assert r.machine.phase == "track" and r.machine.bearing == pytest.approx(rad(60.0))
    view = r.machine.attention(read + 0.15)
    assert view.phase == "turning" and view.bearing_deg == pytest.approx(60.0)
    assert view.point is not None and r.kinds()[-1] == "gaze_spin" and r.machine.spin is not None
    looks, aims, t = r.scene.looks, len(r.door.aims()), read + 0.15
    for k in range(1, 6):  # Nav2's Spin at 30 deg/s, TF's heading
        t = read + 0.15 + 0.1 * k
        r.turn_to(3.0 * k)
        due = r.machine.tick(t)
        assert due == pytest.approx(t + 0.1), "a record every 0.1 s"
    assert r.head() == (45.0, 3.0), "on him: the arbiter keeps the head on his point"
    assert len(r.door.aims()) == aims, "he stands: no aim of the gaze's"
    assert r.scene.looks == looks + 5, "every record while the body turns"
    assert r.wheels.calls == [], "the gaze's own wheels are never asked"
    assert r.machine.bearing == pytest.approx(rad(45.0), abs=rad(0.5)), "from the turned cart"
    r.machine.drive(False, t + 0.1)
    assert r.machine.spin is None and r.machine.phase == "track" and not r.machine.reorient
    r.machine.tick(t + 0.2)
    assert r.scene.looks == looks + 6, "the frames go on"


def test_a_turn_in_place_keeps_the_person_and_refusals_say_why() -> None:
    """The person is kept and the frames go on whatever wheels the gaze has. No conversation,
    or a drive that began with nobody engaged, is refused with why; a drive the tool never
    claims pauses and forgets as ever, and its end orients at the next utterance."""
    r = rig(arc_deg=360.0)  # NoWheels
    assert r.machine.keep_person(0.0, "turn: left") == "no conversation"
    r.person(60.0)
    read = r.wake(60.0)
    r.machine.tick(read)
    r.machine.drive(True, read + 0.1)
    assert r.machine.keep_person(read + 0.15, "turn: left") is None
    assert r.machine.spin is not None
    assert r.said[-1].endswith("(turn: left): the head keeps the person's point")
    looks = r.scene.looks
    r.machine.tick(read + 0.2)
    assert r.scene.looks == looks + 1, "the frames go on"
    r.machine.drive(False, read + 3.0)
    r.machine.drive(True, read + 3.1)
    r.machine.drive(False, read + 4.0)
    assert r.machine.reorient and r.machine.people.people == [], "unclaimed: a drive as ever"
    r.machine.drive(True, read + 4.1)  # paused, nobody engaged since that drive
    why = r.machine.keep_person(read + 4.2, "turn: me")
    assert why == "nobody was engaged when the drive began"


def test_a_sure_name_from_recognition_names_the_person_engaged_in_the_register() -> None:
    """Person recognition's sure verdict on the utterance just heard names the person the gaze
    holds (one person a name: it moves off anyone else), logged; nobody engaged, nobody named."""
    r = tracked(60.0)
    other = r.machine.people.heard(voice_at(-60.0, BOARD + 1.4))[0]
    r.machine.people.named(other, "Саша")
    r.machine.named("Саша", 1.5)
    person = r.machine.person
    assert person is not None and person.who == "Саша" and other.who is None
    assert [f for k, _, f in r.events if k == "gaze_named"] == [{"who": "Саша", "track": person.id}]
    idle = rig()
    idle.machine.named("Саша", 0.0)
    assert idle.machine.people.people == []


def test_the_worker_hands_the_turn_tool_the_person_kept_through_its_drive() -> None:
    from pepin.expression.speaker import Attention

    r = rig(arc_deg=360.0)
    r.person(60.0)
    worker = SpeakerGazeWorker(
        lambda s: r.machine, on=False, load=lambda: SpeakerSettings(doa_map=MAP)
    )
    assert worker.keep_person("turn: me") == "the speaker gaze is off"
    assert worker.switch(True) is None
    r.doa.readings += speech(240, 0.0, 1.0)
    worker.opened(Heard(-1.0, 0.0, "Пепин", named=True))
    worker.drive(True)  # the cascade, before the tool runs
    view = worker.keep_person("turn: me")  # the tool, after it: one queue, in order
    assert isinstance(view, Attention), view
    assert view.phase == "turning" and view.bearing_deg == pytest.approx(
        deg(r.machine.bearing or 0)
    )
    worker.drive(False)
    worker.switch(False)
    worker.close()
    assert r.machine.spin is None
