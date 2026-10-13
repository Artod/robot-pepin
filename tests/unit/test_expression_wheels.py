"""The wheels' expressive motions (pepin.expression.wheels, clearance): a turn reaches its angle
within the caps, the bounded retry and its limits, every refusal, a halt mid-turn, someone
else's command ending a motion without a fight, the rock and a body clip on their times, and
nothing on the caller's thread ever waiting for the wheels. The base server is a fake on a fake
clock (pepin.expression.fakes.FakeBase), the goal server's word a fake room."""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pepin.base_link import BaseState, StateListener
from pepin.expression.animation import Clip, Track
from pepin.expression.book import load_book
from pepin.expression.clearance import Footprint, View, drive_blocker, near_points, turn_blocker
from pepin.expression.conversation import Heard
from pepin.expression.executors import WheelMove, WheelReport, WheelsDone
from pepin.expression.fakes import FakeBase, FakeClock, FakeDoa, FakeFace, FakeFaceFinder, FakeView
from pepin.expression.mix import stretched
from pepin.expression.plan import BodyLimits
from pepin.expression.render import BookRenderer
from pepin.expression.tracker import SpeakerGaze, SpeakerGazeWorker, SpeakerSettings
from pepin.expression.wheels import (
    FIRST_VIEW_S,
    HELD_S,
    WheelsExecutor,
    WheelsGuard,
    WheelsSettings,
    legs_of,
    standing,
)
from pepin.gaze import Reach
from pepin.kinematics import Twist
from pepin.neck import NeckConfig
from pepin.odometry import Pose2D

REPO = Path(__file__).resolve().parents[2]
FULL = WheelsSettings(speed_scale=1.0, yaw_acc_deg_s2=90.0)  # the caps these tests pin
RATE, RAMP = math.radians(30.0), math.radians(90.0)  # the caps at speed 1
BLOCKER = (0.13, -0.06)  # ahead-right: a 20 deg left turn's front corner comes within 0.04 m
# A caller's thread is never held: milliseconds here, bounded far under the 0.6 s a stuck goal
# server costs the wheels' own thread (FIRST_VIEW_S), with room for a loaded parallel run.
CALLER_S = 0.2


@dataclass
class Rig:
    wheels: WheelsExecutor
    base: FakeBase
    view: FakeView
    clock: FakeClock
    said: list[str]


def rig(settings: WheelsSettings = FULL, **view: Any) -> Rig:
    clock = FakeClock()
    base = FakeBase(clock)
    room = FakeView(base, **view)
    said: list[str] = []
    wheels = WheelsExecutor(base.attach, room, load=lambda: settings, clock=clock,
                            pause=base.pause, report=said.append)  # fmt: skip
    return Rig(wheels, base, room, clock, said)


def run(start: Callable[[WheelsDone], None], wait_s: float = 5.0) -> WheelReport:
    """Start a motion and wait (real time) for its end."""
    ended = threading.Event()
    reports: list[WheelReport] = []

    def done(report: WheelReport) -> None:
        reports.append(report)
        ended.set()

    start(done)
    assert ended.wait(wait_s), "the motion never ended"
    return reports[0]


def turn(r: Rig, deg: float, moved: Callable[[float], None] | None = None) -> WheelReport:
    return run(lambda done: r.wheels.start(WheelMove(turn_deg=deg), done, moved))


def twists_after(base: FakeBase, t: float) -> list[Twist]:
    return [twist for at, kind, twist in base.sent if at > t and kind == "twist"]


# ---- the turn --------------------------------------------------------------------------------
def test_a_turn_reaches_its_angle_within_the_tolerance_and_the_caps() -> None:
    r = rig()
    progress: list[float] = []
    report = turn(r, 20.0, progress.append)
    assert report.done and report.attempts == 1 and report.backed_m == 0.0
    assert report.turned_deg == pytest.approx(20.0, abs=FULL.settle_deg)
    assert math.degrees(r.base.pose.theta) == pytest.approx(20.0, abs=FULL.settle_deg)
    rates = [t.angular for t in r.base.twists()]
    assert (
        max(rates) <= RATE + 1e-9
        and min(rates) >= 0.0
        and all(t.linear == 0 for t in r.base.twists())
    )
    steps = np.diff([0.0, *rates])
    assert max(steps) <= RAMP * 0.04 + math.radians(2.0) + 1e-9, "the ramp (the floor on the first)"
    assert r.base.twists()[-1] == Twist(0.0, 0.0), "it ends on a zero twist"
    assert progress[:1] == [pytest.approx(10.0, abs=1.5)], "told every 10 deg"
    assert "turn +20 deg (gaze): done" in r.said[-1]
    right = rig()
    assert turn(right, -15.0).turned_deg == pytest.approx(-15.0, abs=FULL.settle_deg)
    slow = rig(replace(FULL, speed_scale=0.2))
    report = turn(slow, 10.0)
    assert report.done and max(t.angular for t in slow.base.twists()) <= RATE * 0.2 + 1e-9


def test_a_stall_backs_off_and_tries_again_within_three_attempts_and_six_centimetres() -> None:
    r = rig()
    r.base.stalled_turns = True  # the wheels do not turn; they do drive straight
    report = turn(r, 20.0)
    assert not report.done and report.attempts == 3 and report.why.startswith("gave up: stalled")
    assert report.backed_m == pytest.approx(0.06, abs=0.012) and report.turned_deg == 0.0
    backs = [t.linear for t in r.base.twists() if t.linear < 0]
    assert backs and max(abs(v) for v in backs) <= FULL.speed_m_s + 1e-9
    assert r.base.pose.x == pytest.approx(-0.06, abs=0.012), "two back-offs of 3 cm, no third"


def test_a_blocked_arc_backs_off_and_the_turn_goes_once_the_arc_is_clear() -> None:
    r = rig(replace(FULL, clearance_m=0.05), world=[BLOCKER], marks_age_s=0.0)
    report = turn(r, 20.0)
    assert report.done and report.attempts == 2
    assert report.backed_m == pytest.approx(0.03, abs=0.006)
    assert report.turned_deg == pytest.approx(20.0, abs=FULL.settle_deg)
    wide = rig(replace(FULL, clearance_m=0.10), world=[BLOCKER], marks_age_s=0.0)
    given_up = turn(wide, 20.0)
    assert not given_up.done and given_up.attempts == 3 and given_up.turned_deg == 0.0
    assert given_up.why.startswith("gave up: the arc is blocked: a mark at (+0.")
    assert given_up.backed_m == pytest.approx(0.06, abs=0.012)


def test_a_return_that_walks_into_the_arc_mid_turn_stops_it() -> None:
    r = rig(replace(FULL, limits=BodyLimits(attempts=1)), marks_age_s=0.0)
    r.base.at(r.clock.now + 0.4, lambda b: r.view.world.append((-0.30, -0.40)))  # behind right
    report = turn(r, 30.0)
    assert not report.done and "the arc is blocked" in report.why
    assert 0.0 < report.turned_deg < 25.0


# ---- refusals --------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("setup", "why"),
    [
        (lambda r: setattr(r.view, "navigating", True), "Nav2 navigates"),
        (lambda r: setattr(r.view, "navigating", None), "did not say whether Nav2 navigates"),
        (lambda r: setattr(r.view, "why", "the goal server does not answer (refused)"),
         "the goal server does not answer"),
        (lambda r: setattr(r.view, "marks_age_s", 2.0), "the newest obstacle marks are 2.0 s old"),
        (lambda r: setattr(r.view, "marks_age_s", None), "no obstacle marks"),
        (lambda r: r.base.foreign(Twist(0.2, 0.0)), "the wheels are commanded (v +0.20"),
        (lambda r: setattr(r.base, "travel", (0.002, 0.002)), "the cart is rolling"),
        (lambda r: setattr(r.base, "bus_ok", False), "the servo bus did not answer"),
        (lambda r: setattr(r.base, "arm_phase", "moving"), "the arm is mid-gesture (moving)"),
    ],
)  # fmt: skip
def test_every_precondition_refuses_with_its_reason_before_any_twist(
    setup: Callable[[Rig], None], why: str
) -> None:
    r = rig()
    setup(r)
    r.base.line()  # a fresh line says it
    report = turn(r, 20.0)
    assert not report.done and report.why.startswith("refused: ") and why in report.why
    assert r.base.sent == [], "nothing was sent to the wheels"


def test_a_silent_goal_server_or_base_and_the_knobs_refuse() -> None:
    silent = rig(silent=True)
    report = turn(silent, 10.0)
    assert f"the goal server did not answer within {FIRST_VIEW_S:.1f} s" in report.why
    no_base = rig()
    no_base.base.silent = True
    no_base.base.line()  # never a state line
    assert "no word from the base server" in turn(no_base, 10.0).why
    past = rig()
    reports: list[WheelReport] = []
    past.wheels.start(WheelMove(turn_deg=35.0), reports.append)
    past.wheels.rock(0.03, 1, reports.append)
    past.wheels.start(WheelMove(turn_deg=10.0, rock_m=0.01, rocks=1), reports.append)
    assert [x.why for x in reports] == [
        "refused: a turn of +35 deg is past 30",
        "refused: a rock of 3.0 cm is past 2",
        "refused: a turn and a rock in one request",
    ], "answered at once, on the caller's thread"


# ---- halts and other drivers -----------------------------------------------------------------
def test_a_halt_mid_turn_stops_it_at_once_and_no_twist_follows() -> None:
    r = rig()
    halted_at: list[float] = []

    def speak(base: FakeBase) -> None:
        halted_at.append(r.clock.now)
        r.wheels.halt("the person speaks")

    r.base.at(r.clock.now + 0.5, speak)
    report = turn(r, 30.0)
    assert not report.done and report.cut and report.why == "halted: the person speaks"
    assert 3.0 < report.turned_deg < 30.0
    stops = [at for at, kind, _ in r.base.sent if kind == "stop"]
    assert stops and stops[0] == pytest.approx(halted_at[0]), "the stop went with the halt"
    assert twists_after(r.base, halted_at[0]) in ([], [Twist(0.0, 0.0)]), "no twist after it"


def test_someone_else_s_stop_ends_the_motion_and_holds_the_wheels_off() -> None:
    """ros/stop.sh: the first state line that shows a stop we did not send ends the motion;
    our own stop agrees with it, nothing else follows, and new motions wait stop_hold_s."""
    r = rig()
    pressed = r.clock.now + 0.5
    r.base.at(pressed, lambda b: b.foreign(Twist(0.0, 0.0)))
    report = turn(r, 30.0)
    assert not report.done and not report.cut and "someone else stopped the wheels" in report.why
    after = twists_after(r.base, pressed + 0.02)
    assert all(t == Twist(0.0, 0.0) for t in after), "no twist of ours over the red button"
    assert r.base.applied == Twist(0.0, 0.0)
    held = turn(r, 10.0)
    assert held.why.startswith("refused: someone else stopped the wheels") and "30 s" in held.why
    assert r.wheels.ready() is not None, "the gestures are held off too"
    r.clock.now += FULL.stop_hold_s
    r.base.advance(0.02)
    assert turn(r, 10.0).done


def test_someone_else_driving_ends_the_motion_without_a_stop_of_ours() -> None:
    r = rig()
    taken = r.clock.now + 0.4
    r.base.at(taken, lambda b: b.foreign(Twist(0.15, 0.0), keep=True))
    report = turn(r, 30.0)
    assert "someone else drives the wheels (v +0.15" in report.why
    assert [k for at, k, _ in r.base.sent if at > taken] == [], "nothing sent over their drive"


def test_the_red_button_s_cancel_ends_a_motion_whose_stop_line_was_overwritten() -> None:
    """ros/stop.sh sends ONE stop line; our next twist may land in the same 20 ms tick and no
    state line ever shows the stop. Its cancel through the goal server (always its second step)
    ends the motion on the next answer, with a stop of ours, and holds the wheels off."""
    r = rig()
    pressed = r.clock.now + 0.5
    r.base.at(pressed, lambda b: setattr(r.view, "cancelled_at", b.clock()))
    report = turn(r, 30.0)
    assert not report.done and "a cancel reached the goal server 0 s ago" in report.why
    assert twists_after(r.base, pressed + 0.04) in ([], [Twist(0.0, 0.0)])
    assert r.base.applied == Twist(0.0, 0.0)
    held = turn(r, 10.0)
    assert held.why.startswith("refused: a cancel reached the goal server")
    assert r.base.sent[-1][1] == "twist" and r.base.sent[-1][2] == Twist(0.0, 0.0)


def test_the_board_s_deadman_ends_a_motion_whose_twists_stop_arriving() -> None:
    r = rig()
    r.base.at(r.clock.now + 0.3, lambda b: setattr(b, "drop_ours", True))
    report = turn(r, 30.0)
    assert not report.done and "the board's deadman stopped the wheels" in report.why


def test_an_old_deadman_of_a_teleop_does_not_end_a_new_motion() -> None:
    r = rig()
    r.base.deadman = True
    r.base.advance(0.02)
    assert turn(r, 10.0).done


def test_a_state_line_is_ours_while_it_shows_a_twist_the_board_may_still_hold() -> None:
    r = rig()
    w = r.wheels
    w._first_at = 10.0
    w._sent.extend([(10.0, 0.0, 0.1), (10.04, 0.0, 0.2)])
    assert w._ours(0.0, 0.1, 10.2) and w._ours(0.0, 0.2, 10.2)
    assert w._ours(0.0, 0.0, 10.2), "the twist before ours, while ours is on its way"
    assert w._ours(0.0, 0.1, 10.0 + HELD_S - 0.01), "held to the board's deadman"
    late = 10.04 + HELD_S + 0.1
    assert w._ours(0.0, 0.2, late), "the last sent stands after it"
    assert not w._ours(0.0, 0.1, late)
    assert not w._ours(0.0, 0.0, late), "a stop we did not send"


# ---- the rock and the body's clips -------------------------------------------------------------
def test_a_rock_goes_forward_its_centimetres_and_back() -> None:
    r = rig()
    report = run(lambda done: r.wheels.rock(0.02, 2, done))
    assert report.done
    xs = [state.pose.x for _, state in r.base.lines]
    assert max(xs) == pytest.approx(0.02, abs=FULL.settle_m)
    assert max(xs) <= 0.02 + 0.003, "never past the rock's 2 cm by more than the last tick's"
    assert abs(r.base.pose.x) <= FULL.settle_m, "back where it stood"
    forward = [t.linear for t in r.base.twists() if t.linear > 0]
    assert forward and max(forward) <= FULL.speed_m_s + 1e-9
    peaks = sum(
        1 for a, b, c in zip(xs, xs[1:], xs[2:], strict=False) if b > a and b >= c and b > 0.01
    )
    assert peaks == 2, "twice"


def book_clip(atom: str, **params: float) -> Clip:
    return BookRenderer(load_book(REPO / "config/animation.json")).atom_clip(atom, params)


def test_the_rock_atom_lands_its_forward_stroke_near_its_time_and_ends_on_time() -> None:
    clip = book_clip("body_rock", amp_m=0.02, period_s=1.0)
    r = rig()
    began = r.clock.now
    outcome = r.wheels.play(clip)
    assert outcome.ok and outcome.seconds == pytest.approx(1.0)
    for _ in range(500):  # the gesture's motion has no callback: its line in the output
        if r.said:
            break
        time.sleep(0.005)
    assert r.said and "clip of 2 legs (gesture): done" in r.said[-1]
    track = [(at, state.pose.x) for at, state in r.base.lines]
    peak_at, peak = max(track, key=lambda p: p[1])
    assert peak == pytest.approx(0.02, abs=FULL.settle_m)
    assert peak_at - began == pytest.approx(clip.stroke_s, abs=0.2), "the stroke on its time"
    back_at = next(at for at, x in track if at > peak_at and abs(x) <= FULL.settle_m)
    assert back_at - began == pytest.approx(clip.duration_s, abs=0.3)


def test_every_body_atom_of_the_book_fits_the_caps_at_its_extremes_and_faster_is_refused() -> None:
    renderer = BookRenderer(load_book(REPO / "config/animation.json"))
    body = [name for name, atom in renderer.book.atoms.items() if atom.part == "body"]
    assert sorted(body) == ["body_rock", "body_turn", "body_wiggle"]
    for name in body:
        atom = renderer.book.atoms[name]
        for corner in range(2 ** len(atom.params)):
            params = {
                k: (r.hi if corner >> i & 1 else r.lo)
                for i, (k, r) in enumerate(sorted(atom.params.items()))
            }
            clip = renderer.atom_clip(name, params)
            assert not isinstance(legs_of(clip, FULL), str), (
                f"{name} {params}: {legs_of(clip, FULL)}"
            )
            slow = replace(FULL, speed_scale=0.2)
            assert not isinstance(legs_of(stretched(clip, 5.0), slow), str), "the engine stretches"
            assert isinstance(legs_of(clip, slow), str), "unstretched at speed 0.2: too fast"
    quick = book_clip("body_rock", amp_m=0.02, period_s=0.5)
    assert "faster than the wheels' caps" in str(legs_of(quick, FULL))
    far = Clip("body", 0.0, 0.0, 2.0, (Track("yaw_deg", "additive", (0.0, 2.0), (0.0, 40.0)),))
    assert "past wheels.max_turn_deg 30" in str(legs_of(far, FULL))
    yaw = Track("yaw_deg", "additive", (0.0, 2.0), (0.0, 5.0))
    both = Clip("body", 0.0, 0.0, 2.0, (yaw, Track("x_m", "additive", (0.0, 2.0), (0.0, 0.01))))
    assert "turns and drives at once" in str(legs_of(both, FULL))


def test_a_gaze_turn_wins_over_a_gesture_and_a_gesture_waits_its_turn() -> None:
    r = rig()
    started, release = threading.Event(), threading.Event()

    def mid_turn(base: FakeBase) -> None:  # the wheels' thread, held here mid-turn
        started.set()
        release.wait(2.0)

    r.base.at(r.clock.now + 0.3, mid_turn)
    reports: list[WheelReport] = []
    r.wheels.start(WheelMove(turn_deg=30.0), reports.append)
    assert started.wait(2.0)
    refused = r.wheels.play(book_clip("body_rock", amp_m=0.01, period_s=1.0))
    assert not refused.ok and refused.why == "the wheels turn toward the speaker"
    r.wheels.rest()  # the gestures rest: a gaze turn goes on
    release.set()
    for _ in range(400):
        if reports:
            break
        time.sleep(0.005)
    assert reports and reports[0].done


# ---- never on the caller's thread ----------------------------------------------------------------
class StillBase:
    """A base that always reads standing, fresh (for real-clock tests)."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def set_twist(self, twist: Twist) -> None:
        self.sent.append("twist")

    def stop(self) -> None:
        self.sent.append("stop")

    def state(self, now: float | None = None) -> BaseState:
        return BaseState(Pose2D(0, 0, 0), 0, 0, 0, 0, False, False, False, True, 1, 0, 0)


class Stuck:
    """A goal server that never answers."""

    def watch(self, on: bool) -> None:
        pass

    def latest(self) -> View | None:
        return None


def real_wheels(said: list[str]) -> tuple[WheelsExecutor, StillBase]:
    base = StillBase()
    wheels = WheelsExecutor(lambda listener: base, Stuck(), load=lambda: FULL, report=said.append)
    return wheels, base


def test_the_wheels_answer_every_call_at_once_while_their_thread_waits_on_the_goal_server() -> None:
    said: list[str] = []
    wheels, base = real_wheels(said)
    guard = WheelsGuard(wheels)
    reports: list[WheelReport] = []
    began = time.perf_counter()
    wheels.start(WheelMove(turn_deg=20.0), reports.append)
    wheels.play(book_clip("body_rock", amp_m=0.01, period_s=1.0))
    guard.opened(Heard(0.0, 1.0, "Пепин"))
    guard.speech_ended(Heard(0.0, 1.0, "Пепин"))
    took = time.perf_counter() - began
    assert took < CALLER_S and reports == [], f"{took * 1000:.1f} ms on the caller's thread"
    began = time.perf_counter()
    guard.speech_started(1.5)  # the person speaks: a gesture's motion stops, the turn goes on
    assert reports == [], "the turn toward the speaker is not the person's to stop"
    guard.drive(True)  # a drive: every motion is cut at once
    guard.closed("idle")
    assert time.perf_counter() - began < CALLER_S
    for _ in range(200):
        if reports:
            break
        time.sleep(0.005)
    assert reports and reports[0].cut and "a drive starts" in reports[0].why
    assert base.sent == [], "no twist: it never got past the goal server"
    wheels.close()


def test_the_person_speaking_stops_a_gesture_s_motion_and_never_the_turn_toward_them() -> None:
    """Artem called from behind and spoke on: the body stopped half way (2026-10-12, -14.7 of
    -30 deg, "halted: the person speaks"). The turn toward the person speaking is the speaker
    gaze's base layer and goes on; a gesture's motion stops."""

    def held(r: Rig) -> tuple[threading.Event, threading.Event]:
        started, release = threading.Event(), threading.Event()

        def mid(base: FakeBase) -> None:  # the wheels' thread, held here mid-motion
            started.set()
            release.wait(2.0)

        r.base.at(r.clock.now + 0.2, mid)
        return started, release

    gesture = rig()
    started, release = held(gesture)
    assert gesture.wheels.play(book_clip("body_rock", amp_m=0.01, period_s=1.0)).ok
    assert started.wait(2.0)
    gesture.wheels.hush("the person speaks")
    release.set()
    for _ in range(400):
        if gesture.said:
            break
        time.sleep(0.005)
    assert gesture.said and "halted: the person speaks" in gesture.said[-1]
    gaze = rig()
    started, release = held(gaze)
    reports: list[WheelReport] = []
    gaze.wheels.start(WheelMove(turn_deg=20.0), reports.append)
    assert started.wait(2.0)
    gaze.wheels.hush("the person speaks")
    release.set()
    for _ in range(400):
        if reports:
            break
        time.sleep(0.005)
    assert reports and reports[0].done, reports
    gaze.wheels.close()
    gesture.wheels.close()


def test_the_executor_keeps_every_state_line_s_heading_on_the_board_s_clock() -> None:
    """The speaker gaze's frame follows the odometry's heading whoever turns the cart: every
    state line the base link hears is kept by its board stamp."""
    r = rig()
    assert r.wheels.headings().latest() is None, "no link before a motion"
    report = turn(r, 20.0)
    assert report.done
    stamp, heading = r.wheels.headings().latest() or (0.0, 0.0)
    assert heading == pytest.approx(r.base.pose.theta) and stamp == r.base.lines[-1][0]
    first = r.base.lines[0][0]
    assert r.wheels.headings().at(first) == pytest.approx(0.0), "before the turn"
    r.wheels.close()


def test_the_gaze_hands_the_turn_to_the_wheels_and_returns_at_once() -> None:
    """The coordinator's rule: the head turns at once, the speech path gets no wait. The gaze
    machine on a fake clock, the real executor behind it stuck on a silent goal server: the wake
    returns in milliseconds of wall time with the head already aimed and the turn handed over."""
    from test_speaker_gaze import MAP, Door, speech, view

    said: list[str] = []
    wheels, _ = real_wheels(said)
    door = Door(view=view())
    doa = FakeDoa()
    machine = SpeakerGaze(SpeakerSettings(doa_map=MAP), door=door, doa=doa,
                          finder=FakeFaceFinder(), wheels=wheels, face=FakeFace(),
                          reach=Reach.of(NeckConfig.from_json(REPO / "config/neck.json")),
                          report=lambda line: None)  # fmt: skip
    doa.readings += speech(280, 0.0, 1.2)
    began = time.perf_counter()
    machine.opened(Heard(0.0, 1.2, "Пепин"), 1.6)
    took = time.perf_counter() - began
    assert took < CALLER_S, f"{took * 1000:.1f} ms"
    assert door.aims()[0][0].pan_rad == pytest.approx(math.radians(100))
    assert machine.turning is not None and machine.turning.asked_deg == pytest.approx(30.0)
    wheels.close()


def test_the_gaze_worker_s_moments_never_wait_on_the_wheels() -> None:
    said: list[str] = []
    wheels, _ = real_wheels(said)
    from test_speaker_gaze import MAP, Door, view

    machine = SpeakerGaze(SpeakerSettings(doa_map=MAP), door=Door(view=view()), doa=FakeDoa(),
                          finder=None, wheels=wheels, face=None,
                          reach=Reach.of(NeckConfig.from_json(REPO / "config/neck.json")),
                          report=lambda line: None)  # fmt: skip
    worker = SpeakerGazeWorker(
        lambda s: machine, on=True, load=lambda: SpeakerSettings(doa_map=MAP)
    )
    guard = WheelsGuard(wheels)
    calls: list[Callable[[], None]] = [
        lambda: worker.opened(Heard(0.0, 1.0, "Пепин")),
        lambda: guard.speech_started(2.0),
        lambda: worker.speech_ended(Heard(2.0, 3.0, "привет")),
        lambda: worker.drive(True),
        lambda: guard.drive(True),
        lambda: worker.closed("idle"),
        lambda: guard.closed("idle"),
    ]
    for call in calls:
        began = time.perf_counter()
        call()
        assert time.perf_counter() - began < CALLER_S / 2
    worker.close()
    wheels.close()


# ---- the pieces --------------------------------------------------------------------------------
def test_standing_names_what_keeps_the_cart_from_an_expressive_motion() -> None:
    still = BaseState(Pose2D(0, 0, 0), 0, 0, 0, 0, False, True, False, True, 1, 0, 0)
    assert standing(still) is None
    assert standing(replace(still, arm_phase="resting")) is None, "on its way to rest: fine"
    assert "holding" in str(standing(replace(still, arm_phase="holding")))


def test_the_swept_footprint_blocks_what_the_motion_approaches_not_what_it_leaves() -> None:
    fp = Footprint()
    wall = np.array([[-0.35, y] for y in np.linspace(-0.4, 0.4, 17)])  # 5 cm behind the cart
    assert turn_blocker(wall, fp, 0.0, math.radians(30), 0.10) is not None, "the rear swings"
    assert drive_blocker(wall, fp, 0.0, -0.03, 0.10) is not None
    assert drive_blocker(wall, fp, 0.0, 0.03, 0.10) is None, "away from it"
    assert drive_blocker(wall, fp, 0.02, 0.0, 0.10) is None, "back to where it stood"
    ahead_right = np.array([[0.2, -0.25]])
    hit = turn_blocker(ahead_right, fp, 0.0, math.radians(30), 0.10)
    assert hit is not None and 0 < hit.at < 30, "a left turn swings the front-right corner out"
    assert turn_blocker(ahead_right, fp, 0.0, math.radians(-30), 0.10) is None
    inside = np.array([[-0.1, 0.0]])
    assert turn_blocker(inside, fp, 0.0, math.radians(30), 0.10) is None, "the cart's own post"
    far = np.array([[1.5, 0.0], [-1.0, 0.5]])
    assert turn_blocker(far, fp, 0.0, math.radians(30), 0.10) is None


def test_the_goal_server_s_returns_are_placed_through_the_laser_s_mount() -> None:
    from pepin.mounts import load_lidar_mount

    mount = load_lidar_mount(REPO / "config")
    lidar = json.loads((REPO / "config/lidar.json").read_text())
    forward = math.radians(-lidar["yaw_offset_deg"])  # the laser's bearing of the cart's nose
    angles = np.array([forward, forward + math.pi / 2, 0.3])
    ranges = np.array([0.5, 0.4, 2.0])
    points = near_points(angles, ranges, mount.rotation(), mount.translation(), 1.0)
    assert len(points) == 2, "the 2 m return is beyond the radius"
    assert points[0] == pytest.approx([0.5 + lidar["x_m"], 0.0], abs=1e-6)
    assert points[1] == pytest.approx([lidar["x_m"], -0.4], abs=1e-6), "upside down: right"


def test_the_knobs_ship_as_the_design_bounds_them() -> None:
    settings = WheelsSettings.load(REPO / "config/expression.json")
    lim = settings.limits
    # shipped at 0 until the first supervised motion (2026-10-12); the design bounds are 30 / 0.02
    assert (lim.max_turn_deg, lim.max_rock_m, lim.attempts) == (30.0, 0.02, 3)
    assert (lim.backoff_m, lim.max_backoff_m) == (0.03, 0.06)
    assert (settings.speed_scale, settings.yaw_acc_deg_s2) == (0.4, 360.0)
    assert SpeakerSettings.load(REPO / "config/expression.json").body == lim
    with pytest.raises(ValueError, match="speed_scale"):
        WheelsSettings.from_config({"wheels": {"speed_scale": 1.5}})


def test_the_listener_hears_every_state_line_and_the_arm_s_last_word_holds() -> None:
    from pepin.base_link import BaseClient

    heard: list[tuple[BaseState, float]] = []
    client = BaseClient("unused")
    listener: StateListener = lambda state, at: heard.append((state, at))  # noqa: E731
    client.on_state = listener
    pose = {"t": 1.0, "x": 0.0, "y": 0.0, "theta": 0.0, "dl": 0.0, "dr": 0.0}
    flags = {"moving": True, "armed": True, "deadman": False, "bus_ok": True}
    line = {"type": "state", **pose, "v": 0.0, "w": 0.1, **flags}
    client._ingest({**line, "arm": {"phase": "moving"}})
    client._ingest(line)
    assert [s.arm_phase for s, _ in heard] == ["moving", "moving"]
    assert heard[1][0].w == 0.1 and client.state() is not None


# ---- the real wire ------------------------------------------------------------------------------
class RollingBus:
    """The servo bus with wheels that turn at the velocity last written (for the real base
    server core on a socket): every read advances each encoder by its ticks a second."""

    def __init__(self) -> None:
        from pepin.base import LEFT, RIGHT

        self.velocity = {LEFT: 0, RIGHT: 0}
        self.positions = {LEFT: 0.0, RIGHT: 0.0}
        self.read_at = time.monotonic()

    def sync_write(self, data_name: str, values: dict[str, int], *, normalize: bool = True) -> None:
        if data_name == "Goal_Velocity":
            self._roll()
            self.velocity.update(values)

    def sync_write_block(self, data_names: Any, values: Any) -> None:
        pass

    def sync_read(self, data_name: str, motors: list[str], **kw: Any) -> dict[str, int]:
        self._roll()
        if data_name != "Present_Position":
            return {m: 30 for m in motors}
        return {m: int(self.positions[m]) % 4096 for m in motors}

    def enable_torque(self, motors: list[str] | None = None) -> None:
        pass

    def disable_torque(self, motors: list[str] | None = None) -> None:
        pass

    def ping(self, motor: str, num_retry: int = 0) -> int | None:
        return 0

    def _roll(self) -> None:
        now = time.monotonic()
        for m, v in self.velocity.items():
            self.positions[m] += v * (now - self.read_at)
        self.read_at = now


class OpenRoom:
    """The goal server's word on the real clock: nothing navigates, a fresh scan, no return."""

    def watch(self, on: bool) -> None:
        pass

    def latest(self) -> View:
        return View(time.monotonic(), False, 0.05)


@pytest.mark.slow
def test_on_the_real_wire_the_turn_knows_its_own_twists_and_ends_on_a_teleop_s() -> None:
    """The real BaseClient against the real base server core on a socket: every state line
    carries back the twist as sent (a line misread as someone else's would end every turn at
    once), the turn lands on its angle; another client driving ends the next one."""
    import socket

    from test_base import CFG

    from pepin.base_link import BaseClient
    from pepin.base_server import DRIVING_COMMANDS, BaseServerCore, serve
    from pepin.streams import JsonLinesServer

    core = BaseServerCore(RollingBus(), CFG)  # type: ignore[arg-type]
    server = JsonLinesServer(0, driving_commands=DRIVING_COMMANDS).start()
    stop = threading.Event()
    worker = threading.Thread(target=serve, args=(core, server, 50.0, 50.0, stop), daemon=True)
    worker.start()

    def link(listener: StateListener) -> BaseClient:
        client = BaseClient("127.0.0.1", server.port)
        client.on_state = listener
        return client.start()

    said: list[str] = []
    wheels = WheelsExecutor(link, OpenRoom(), load=lambda: FULL, report=said.append)
    try:
        report = run(lambda done: wheels.start(WheelMove(turn_deg=20.0), done), wait_s=10.0)
        assert report.done, report.why
        assert math.degrees(core._odom.pose.theta) == pytest.approx(20.0, abs=3.0)
        teleop = socket.create_connection(("127.0.0.1", server.port), timeout=2.0)
        keep = threading.Event()

        def drive() -> None:  # a teleop holding its key: a twist every 50 ms
            time.sleep(0.3)
            while not keep.is_set():
                teleop.sendall(b'{"cmd": "twist", "v": 0.05, "w": 0.0}\n')
                time.sleep(0.05)

        threading.Thread(target=drive, daemon=True).start()
        taken = run(lambda done: wheels.start(WheelMove(turn_deg=-25.0), done), wait_s=10.0)
        keep.set()
        teleop.close()
        assert "someone else drives the wheels (v +0.05" in taken.why
    finally:
        wheels.close()
        stop.set()
        worker.join(timeout=2.0)


def test_the_goal_server_s_answer_becomes_a_view_and_an_older_build_refuses() -> None:
    from pepin.expression.clearance import view_of

    answer = {"event": "clearance", "navigating": False, "marks": "ok", "cancel_age_s": 12.5,
              "marks_age_s": 0.04, "points": [[0.5, 0.0], [-0.4, 0.2]]}  # fmt: skip
    view = view_of(answer, 7.0)
    assert (view.at, view.navigating, view.marks_age_s) == (7.0, False, 0.04)
    assert view.cancel_age_s == 12.5 and view.points.shape == (2, 2) and view.why == ""
    old = view_of({"event": "error", "detail": "unknown command 'clearance'"}, 7.0)
    assert "an older build" in old.why and "kick goal_server" in old.why
    silent = view_of({**answer, "marks": "silent 9 s (depth_fusion)"}, 7.0)
    assert silent.why == "the obstacle marks: silent 9 s (depth_fusion)"
    unmarked = view_of({**answer, "marks_age_s": None, "points": []}, 7.0)
    assert unmarked.why.startswith("no obstacle marks have reached the goal server")
    raw_scan = {"event": "clearance", "navigating": False, "lidar": "ok", "cancel_age_s": None,
                "scan_age_s": 0.04, "points": [[0.5, 0.0]]}  # fmt: skip
    before = view_of(raw_scan, 7.0)
    assert "kick goal_server" in before.why, "a goal server answering from the raw scan refuses"


def test_the_goal_server_view_asks_only_while_watched() -> None:
    import socket

    from pepin.expression.clearance import GoalServerView

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    asked: list[str] = []

    def serve() -> None:
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            with conn:
                asked.append(conn.makefile("r").readline())
                line = {"event": "clearance", "navigating": True, "marks": "ok",
                        "marks_age_s": 0.02, "points": [[0.3, 0.0]]}  # fmt: skip
                conn.sendall((json.dumps(line) + "\n").encode())

    threading.Thread(target=serve, daemon=True).start()
    view = GoalServerView(host="127.0.0.1", port=listener.getsockname()[1], every_s=0.02)
    assert view.latest() is None and asked == [], "nothing asked until watched"
    view.watch(True)
    latest = None
    for _ in range(100):
        latest = view.latest()
        if latest is not None:
            break
        time.sleep(0.005)
    assert latest is not None and latest.navigating is True
    assert latest.points.tolist() == [[0.3, 0.0]]
    assert json.loads(asked[0]) == {"cmd": "clearance", "radius_m": 1.0}
    view.watch(False)
    time.sleep(0.05)
    count = len(asked)
    time.sleep(0.1)
    assert len(asked) == count, "unwatched: it stops asking"
    listener.close()
    away = GoalServerView(host="127.0.0.1", port=1, timeout_s=0.1)
    assert "the goal server does not answer" in away.look().why
