"""The arbiter's overlay (pepin.gaze, kind ``clip``): keyed offsets added to whatever holds the
head, bounded, with a TTL, one at a time, the idle layer giving way; ``withdraw`` with a fade.

The head is a fake ``neck_target`` driver that records every write and its own speed ceiling; time
is passed in, so every rule is a few steps of arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pepin.gaze import (
    IDLE,
    OVERLAY_DRIVER,
    PERSON,
    Aim,
    Arbiter,
    GazeSettings,
    HeadReading,
    Look,
    Outcome,
    Overlay,
    OverlayKey,
    Reach,
    Refusal,
    Speed,
    home_aim,
    look_from_json,
    overlay_from_json,
)
from pepin.neck import NeckConfig

REPO = Path(__file__).resolve().parents[2]
CFG = NeckConfig.from_json(REPO / "config/neck.json")
HOME = home_aim(CFG)
REACH = Reach.of(CFG)
LEFT = Aim(math.radians(30), math.radians(20))
SETTINGS = GazeSettings(frames=3, settle_tol_deg=1.0, move_timeout_s=3.0, frame_period_s=0.1)
SPEAKER, GESTURE, BREATH = "voice.speaker", "voice.gesture", "voice.idle"
Key = tuple[float, float, float, float]  # at_s, pan_deg, tilt_deg, speed_deg_s
NOD: tuple[Key, ...] = ((0.0, 0.0, 5.0, 100.0), (0.25, 0.0, 0.0, 100.0))
GLANCE: tuple[Key, ...] = ((0.0, 15.0, -6.0, 200.0), (2.0, 0.0, 0.0, 200.0))
BREATHE: tuple[Key, ...] = ((0.0, 0.0, -0.6, 0.6), (2.0, 0.0, 0.6, 0.6), (4.0, 0.0, -0.6, 0.6))


class TargetHead:
    """A ``neck_target`` driver: takes targets while the head moves; records each write with the
    ceiling of its own it carried."""

    moves_while_driving = True

    def __init__(self) -> None:
        self.writes: list[tuple[Aim | None, bool, float]] = []
        self.ceilings: list[float | None] = []
        self.ramps: list[float | None] = []
        self.tilt_ramps: list[float | None] = []
        self.refusal: Refusal | None = None

    def blocked(self, now: float) -> Refusal | None:
        return None

    def write(
        self,
        aim: Aim | None,
        *,
        speed: Speed,
        hold: bool,
        now: float,
        speed_deg_s: float | None = None,
        acc_deg_s2: float | None = None,
        tilt_acc_deg_s2: float | None = None,
    ) -> None:
        self.writes.append((aim, hold, now))
        self.ceilings.append(speed_deg_s)
        self.ramps.append(acc_deg_s2)
        self.tilt_ramps.append(tilt_acc_deg_s2)

    def take_refusal(self) -> Refusal | None:
        refusal, self.refusal = self.refusal, None
        return refusal

    def take_arrival(self) -> HeadReading | None:
        return None

    def keep(self, now: float) -> None:
        pass


class GotoHead(TargetHead):
    """A base server that moves the neck one move at a time, at rest."""

    moves_while_driving = False


def arbiter(head: TargetHead | None = None) -> tuple[Arbiter, TargetHead, list[Outcome]]:
    head = head if head is not None else TargetHead()
    return Arbiter(head, HOME, SETTINGS, REACH), head, []


def clip(
    keys: tuple[Key, ...] = NOD,
    source: str = GESTURE,
    *,
    band: int = IDLE,
    ttl_s: float = 5.0,
    loop: bool = False,
    idle: bool = False,
) -> Look:
    overlay = Overlay(
        tuple(OverlayKey(at, math.radians(p), math.radians(t), v) for at, p, t, v in keys),
        loop=loop,
        idle=idle,
    )
    return Look(source, (), band, 0, 0.0, ttl_s, kind="clip", overlay=overlay)


def still_at(arb: Arbiter, aim: Aim, stamp: float) -> None:
    """Two encoder readings at ``aim`` after ``stamp``: the head has settled there."""
    arb.observe(HeadReading(aim.pan_rad, aim.tilt_rad, stamp))
    arb.observe(HeadReading(aim.pan_rad, aim.tilt_rad, stamp + 0.01))


def plus(aim: Aim, pan_deg: float, tilt_deg: float) -> Aim:
    return Aim(aim.pan_rad + math.radians(pan_deg), aim.tilt_rad + math.radians(tilt_deg))


def speaker_on(arb: Arbiter, aim: Aim = LEFT, now: float = 0.0) -> list[Outcome]:
    """The speaker gaze holds the head at ``aim``, settled."""
    out: list[Outcome] = []
    arb.submit(Look(SPEAKER, (aim,), IDLE, 0, 6.0, 6.0), now, out.append)
    arb.step(now)
    still_at(arb, aim, now + 0.3)
    arb.step(now + 0.4)
    assert out and out[0].status == "done"
    return out


def assert_aim(written: Aim | None, expected: Aim) -> None:
    assert written is not None
    assert written.pan_rad == pytest.approx(expected.pan_rad)
    assert written.tilt_rad == pytest.approx(expected.tilt_rad)


def test_an_overlay_adds_its_keys_to_the_holder_s_aim_and_the_holder_keeps_the_head() -> None:
    arb, head, out = arbiter()
    speaker_on(arb)
    assert arb.submit(clip(), 1.0, out.append).status == "granted"
    arb.step(1.0)
    assert_aim(head.writes[-1][0], plus(LEFT, 0, 5))
    assert head.ceilings[-1] == 100.0 and head.writes[-1][1] is True
    state = arb.state(1.05)
    assert state.blind and state.source == SPEAKER and arb.holds(SPEAKER)
    assert arb.overlay() is not None and [r.source for r in arb.pending()] == [SPEAKER]
    still_at(arb, plus(LEFT, 0, 5), 1.15)
    arb.step(1.2)
    held = plus(LEFT, 0, 5)
    arb.observe(HeadReading(held.pan_rad, held.tilt_rad, 1.35))  # the board's now: its newest
    assert not arb.state(1.35).blind, "a held key is a still head: frames count again"
    arb.step(1.25)
    assert_aim(head.writes[-1][0], LEFT)
    writes = len(head.writes)
    arb.step(1.3)  # the last key's move (5 deg at 100 deg/s) is over: the clip ends
    assert out[-1].status == "done" and out[-1].reached and arb.overlay() is None
    arb.step(1.5)
    assert len(head.writes) == writes, "already on the holder's aim: nothing to write back"
    assert arb.holds(SPEAKER) and arb.state(1.5).target == LEFT


def test_no_key_is_written_while_the_holder_s_move_flies_and_the_key_waits_to_settle() -> None:
    """2026-10-13: the pulse's melody and the idle breath wrote their glide keys (1-3 ramp
    units) over the speaker look's 200 deg/s turn while it flew; the base server passed the
    ramp on and the servo lost its brakes. A key due while the head has not settled waits;
    the key due once it has is the one written, with its own ceilings."""
    arb, head, out = arbiter()
    arb.submit(Look(SPEAKER, (LEFT,), IDLE, 0, 6.0, 6.0), 0.0, out.append)
    arb.step(0.0)  # the holder's turn written: the head flies, not settled
    writes = len(head.writes)
    assert arb.submit(clip(BREATHE, BREATH, idle=True, loop=True), 0.1, out.append).status == (
        "granted"
    )
    arb.step(0.1)
    arb.step(0.5)
    assert len(head.writes) == writes, "no key over a head in flight"
    still_at(arb, LEFT, 0.6)
    arb.step(0.7)
    assert len(head.writes) == writes + 1, "settled: the key due now is written"
    assert_aim(head.writes[-1][0], plus(LEFT, 0, -0.6))
    assert head.ceilings[-1] == 0.6


def test_over_nothing_an_overlay_plays_over_home_and_lets_the_head_go_after() -> None:
    arb, head, out = arbiter()
    arb.submit(clip(GLANCE), 0.0, out.append)
    arb.step(0.0)
    assert_aim(head.writes[-1][0], plus(HOME, 15, -6))
    assert head.writes[-1][1] is True, "a lifted home is held"
    still_at(arb, plus(HOME, 15, -6), 0.3)  # the key's move settled: the way back may go
    arb.step(2.0)
    assert head.writes[-1] == (None, False, 2.0), "home, let go"
    arb.step(2.2)
    assert out[-1].status == "done"


def test_an_overlay_never_plays_over_a_better_band_and_ends_when_one_takes_the_head() -> None:
    arb, head, out = arbiter()
    person = Look("llm.look", (LEFT,), PERSON, 0, 10.0, 10.0)
    arb.submit(person, 0.0)
    arb.step(0.0)
    denied = arb.submit(clip(), 0.5, out.append)
    assert denied.status == "denied" and "llm.look (person band)" in denied.reason
    arb, head, out = arbiter()
    speaker_on(arb)
    arb.submit(clip(GLANCE), 1.0, out.append)
    arb.step(1.0)
    right = Aim(math.radians(-30), math.radians(20))
    arb.submit(Look("llm.look", (right,), PERSON, 0, 10.0, 10.0), 1.5)
    arb.step(1.5)
    assert out[-1].status == "preempted" and out[-1].reason == "by llm.look"
    assert head.writes[-1][0] == right and head.ceilings[-1] is None, "no offset rides along"


def test_an_overlay_past_the_reach_the_bound_or_the_mast_s_rate_is_refused_whole() -> None:
    arb, _head, out = arbiter()
    speaker_on(arb, Aim(math.radians(150), math.radians(20)))
    far = arb.submit(clip(((0.0, 20.0, 0.0, 100.0), (0.5, 0.0, 0.0, 100.0))), 1.0, out.append)
    assert far.status == "denied" and "past the neck's reach" in far.reason
    wide = clip(((0.0, 35.0, 0.0, 100.0),))
    assert "clip_max_deg 30" in str(wide.overlay and wide.overlay.refusal(SETTINGS))
    shiver = ((0.0, 0.0, 3.0, 300.0), (0.1, 0.0, -3.0, 300.0), (0.2, 0.0, 0.0, 300.0))
    fast = clip(shiver)
    assert "clip_max_hz 2.5" in str(fast.overlay and fast.overlay.refusal(SETTINGS))
    nod = clip(((0.0, 0.0, 5.0, 100.0), (0.2, 0.0, -1.0, 100.0), (0.4, 0.0, 0.0, 100.0)))
    assert nod.overlay is not None and nod.overlay.refusal(SETTINGS) is None, "2.5 Hz is in"
    loop = clip(((0.0, 0.0, 1.0, 10.0),), loop=True)
    assert "period" in str(loop.overlay and loop.overlay.refusal(SETTINGS))


def test_a_newer_overlay_replaces_the_one_playing_and_an_idle_one_gives_way() -> None:
    arb, _head, out = arbiter()
    speaker_on(arb)
    arb.submit(clip(BREATHE, BREATH, loop=True, idle=True), 1.0, out.append)
    arb.step(1.0)
    arb.submit(clip(GLANCE), 1.5, out.append)
    assert out[-1].source == BREATH and out[-1].status == "preempted"
    assert out[-1].reason == f"by {GESTURE}"
    refused = arb.submit(clip(BREATHE, BREATH, loop=True, idle=True), 1.6, out.append)
    assert refused.status == "denied" and f"{GESTURE}'s overlay plays" in refused.reason
    arb.submit(clip(NOD), 1.7, out.append)
    assert out[-1].reason == f"by a newer request of {GESTURE}"
    overlay = arb.overlay()
    assert overlay is not None and overlay.overlay is not None and overlay.overlay.keys[0].at_s == 0


def test_a_breath_loops_without_blinding_the_head_until_its_ttl_lapses_unrenewed() -> None:
    arb, head, out = arbiter()
    speaker_on(arb)
    arb.submit(clip(BREATHE, BREATH, ttl_s=3.0, loop=True, idle=True), 1.0, out.append)
    for step in range(1, 50):
        now = 1.0 + step * 0.1
        arb.observe(HeadReading(LEFT.pan_rad, LEFT.tilt_rad, now))  # the encoders, 10 a second
        arb.step(now)
        assert not arb.state(now).blind, f"blind at {now:.1f}: a breath is no move"
        if now == pytest.approx(2.5):
            assert arb.renew(BREATH, now) == 1
    lifts = [(w[0], w[2]) for w in head.writes[1:]]
    assert [round(t, 1) for _aim, t in lifts[:3]] == [1.1, 3.0, 5.0], "a key every 2 s"
    assert_aim(lifts[0][0], plus(LEFT, 0, -0.6))
    assert_aim(lifts[1][0], plus(LEFT, 0, 0.6))
    assert_aim(lifts[2][0], plus(LEFT, 0, -0.6))
    assert out[-1].status == "expired", "renewed at 2.5, lapsed 3 s later"
    assert_aim(head.writes[-1][0], LEFT)
    assert arb.holds(SPEAKER)


def test_its_source_s_release_ends_the_overlay_and_the_head_fades_back() -> None:
    arb, head, out = arbiter()
    speaker_on(arb)
    arb.submit(clip(((0.0, 20.0, 0.0, 200.0), (3.0, 0.0, 0.0, 200.0))), 1.0, out.append)
    arb.step(1.0)
    still_at(arb, plus(LEFT, 20, 0), 1.2)  # the key's move settled: the way back may go
    assert arb.withdraw(GESTURE, 1.5, "its source's release", fade_s=0.4) == 1
    assert out[-1].status == "preempted"
    arb.step(1.55)
    assert_aim(head.writes[-1][0], LEFT)
    assert head.ceilings[-1] == pytest.approx(50.0), "20 deg over 0.4 s"
    assert arb.holds(SPEAKER), "the holder's look is not the gesture's to let go"


def test_a_drive_s_start_lets_the_overlay_go_with_the_rest() -> None:
    arb, head, out = arbiter()
    speaker_on(arb)
    arb.submit(clip(GLANCE), 1.0, out.append)
    arb.step(1.0)
    assert arb.release(1.2, "the drive's start", keep=lambda look: look.band == 0) == 2
    arb.step(1.25)
    assert out[-1].status == "preempted" and arb.overlay() is None
    assert head.writes[-1] == (None, False, 1.25), "nothing holds the head: home"


def test_a_new_aim_of_the_holder_waits_for_a_gesture_and_rides_under_an_idle_overlay() -> None:
    """2026-10-15: the speaker gaze's follow wrote the head seven times in 0.22 s under a shake
    and its middle key was never written. A gesture's overlay rides the holder's aim its first
    key was written on: a newer aim of the holder writes nothing while it plays, its keys stay
    on the aim they started on, and its end takes the head back to the newer aim; an idle
    overlay (the breath) carries on over the holder's newest aim."""
    arb, head, out = arbiter()
    speaker_on(arb)
    arb.submit(clip(GLANCE), 1.0, out.append)
    arb.step(1.0)
    assert_aim(head.writes[-1][0], plus(LEFT, 15, -6))
    nearer = Aim(math.radians(10), math.radians(20))
    arb.submit(Look(SPEAKER, (nearer,), IDLE, 0, 6.0, 6.0), 1.2)
    writes = len(head.writes)
    still_at(arb, plus(LEFT, 15, -6), 1.25)
    arb.step(1.3)
    assert len(head.writes) == writes, "the holder's move waits for the gesture's end"
    arb.step(3.0)  # the glance's last key: back on the aim it started on, not the newer one
    assert_aim(head.writes[-1][0], LEFT)
    still_at(arb, LEFT, 3.1)
    arb.step(3.2)
    assert out[-1].status == "done"
    arb.step(3.25)
    assert_aim(head.writes[-1][0], nearer)  # the way back: the holder's aim of now
    assert arb.state(3.25).target == nearer
    still_at(arb, nearer, 3.4)
    arb.step(3.5)
    arb.submit(clip(BREATHE, BREATH, idle=True, loop=True), 3.5)
    arb.step(3.5)
    assert_aim(head.writes[-1][0], plus(nearer, 0, -0.6))
    nearest = Aim(math.radians(5), math.radians(20))
    arb.submit(Look(SPEAKER, (nearest,), IDLE, 0, 6.0, 6.0), 3.7)
    still_at(arb, plus(nearer, 0, -0.6), 3.75)
    arb.step(3.8)
    assert_aim(head.writes[-1][0], plus(nearest, 0, -0.6))  # an idle overlay follows the holder


def test_an_overlay_needs_a_driver_that_takes_targets_while_the_head_moves() -> None:
    arb, _head, out = arbiter(GotoHead())
    assert arb.submit(clip(), 0.0, out.append).reason == OVERLAY_DRIVER


def test_a_refused_key_ends_the_overlay() -> None:
    arb, head, out = arbiter()
    speaker_on(arb)
    arb.submit(clip(GLANCE), 1.0, out.append)
    arb.step(1.0)
    head.refusal = Refusal("a jog is under way", True)
    arb.step(1.05)
    assert out[-1].status == "denied" and out[-1].reason == "a jog is under way"


def test_the_keys_of_a_loop_and_the_end_of_a_pass() -> None:
    breath = clip(BREATHE, BREATH, loop=True).overlay
    assert breath is not None and breath.period_s == 4.0
    assert [breath.key_at(t) for t in (0.0, 1.9, 2.0, 4.0, 6.5)] == [
        (0, 0),
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    ]
    late = clip(((0.5, 5.0, 0.0, 50.0), (1.0, 0.0, 0.0, 50.0))).overlay
    assert late is not None and late.key_at(0.2) == (-1, 0)
    assert late.end_s() == pytest.approx(1.1), "the last key plus 5 deg at 50 deg/s"


def test_a_clip_from_json() -> None:
    keys = [
        {"at_s": 0.0, "pan_rad": 0.0, "tilt_rad": 0.08, "speed_deg_s": 100.0},
        {"at_s": 0.3, "pan_rad": 0.0, "tilt_rad": 0.0, "speed_deg_s": 100.0},
    ]
    look = look_from_json(
        {"source": GESTURE, "kind": "clip", "ttl_s": 2.0, "target": {"keys": keys}},
        SETTINGS,
        CFG,
        lambda frame, xyz: "unused",
    )
    assert isinstance(look, Look) and look.band == IDLE and look.overlay is not None
    assert look.views == () and look.ttl_s == 2.0 and len(look.overlay.keys) == 2
    cycle = [
        {"at_s": at, "tilt_rad": tilt, "speed_deg_s": 0.6}
        for at, tilt in ((0.0, -0.01), (2.0, 0.01), (4.0, -0.01))
    ]
    breath = look_from_json(
        {"source": BREATH, "kind": "clip", "target": {"keys": cycle, "loop": True, "idle": True}},
        SETTINGS,
        CFG,
        lambda frame, xyz: "unused",
    )
    assert isinstance(breath, Look) and breath.overlay is not None
    assert breath.overlay.loop and breath.overlay.idle and breath.ttl_s == 20.0


@pytest.mark.parametrize(
    ("target", "why"),
    [
        ({}, "carries its keys"),
        ({"keys": [{"at_s": 0.0}]}, "incomplete"),
        ({"keys": [{"at_s": 0.5, "speed_deg_s": 9}, {"at_s": 0.1, "speed_deg_s": 9}]}, "order"),
        ({"keys": [{"at_s": 0.0, "speed_deg_s": 0.0}]}, "above 0"),
        ({"keys": [{"at_s": 0.0, "pan_rad": 1.0, "speed_deg_s": 99}]}, "clip_max_deg"),
        ({"keys": [{"at_s": float("nan"), "speed_deg_s": 9}]}, "finite"),
        ({"keys": [{"at_s": 0.0, "speed_deg_s": 9}] * 65}, "at most 64"),
    ],
)
def test_bad_clips_are_refused_in_words(target: dict[str, Any], why: str) -> None:
    answer = look_from_json(
        {"source": GESTURE, "kind": "clip", "target": target},
        SETTINGS,
        CFG,
        lambda frame, xyz: "unused",
    )
    assert isinstance(answer, str) and why in answer


def test_an_eased_look_carries_its_own_speed_and_ramp_to_the_driver() -> None:
    """The speaker gaze's turn: an angles look with ``speed_deg_s`` and ``acc_deg_s2`` is written
    with both (neck_target runs the servo's trapezoid at them: an ease in and out); a look
    without them writes none; a ceiling that is no positive number is refused in words."""

    def unused(frame: str, xyz: dict[str, float]) -> str:
        return "unused"

    request = {"source": SPEAKER, "kind": "angles", "band": IDLE, "hold": True, "frames": 0,
               "target": {"pan_rad": 0.5, "tilt_rad": 0.05}, "speed": "saccade"}  # fmt: skip
    eased = look_from_json({**request, "speed_deg_s": 200, "acc_deg_s2": 650.0}, SETTINGS, CFG,
                           unused)  # fmt: skip
    assert isinstance(eased, Look) and (eased.speed_deg_s, eased.acc_deg_s2) == (200.0, 650.0)
    arb, head, _ = arbiter()
    arb.submit(eased, 0.0)
    arb.step(0.0)
    assert head.writes[-1][0] == Aim(0.5, 0.05)
    assert (head.ceilings[-1], head.ramps[-1]) == (200.0, 650.0)
    plain = look_from_json(request, SETTINGS, CFG, unused)
    assert isinstance(plain, Look) and plain.speed_deg_s is None and plain.acc_deg_s2 is None
    arb.submit(replace(plain, source="llm.look", band=PERSON, views=(Aim(-0.5, 0.05),)), 1.0)
    arb.step(1.0)
    assert (head.ceilings[-1], head.ramps[-1]) == (None, None)
    for bad, why in (
        ({"acc_deg_s2": 0}, "above 0"),
        ({"speed_deg_s": "fast"}, "a number"),
        ({"acc_deg_s2": float("inf")}, "finite"),
        ({"speed_deg_s": True}, "a number"),
    ):
        answer = look_from_json({**request, **bad}, SETTINGS, CFG, unused)
        assert isinstance(answer, str) and why in answer, (bad, answer)


def test_an_overlay_key_s_own_ramp_is_read_checked_and_written_with_it() -> None:
    """A key's ``acc_deg_s2`` (the idle layer's glide) rides to the driver beside its speed;
    absent, none is written (the neck's own); zero or below is refused whole."""
    settings = GazeSettings()
    keys = [{"at_s": 0.0, "tilt_rad": math.radians(0.8), "speed_deg_s": 2.0, "acc_deg_s2": 9.5},
            {"at_s": 0.7, "tilt_rad": 0.0, "speed_deg_s": 2.0}]  # fmt: skip
    overlay = overlay_from_json({"keys": keys, "idle": True}, settings)
    assert not isinstance(overlay, str)
    assert [k.acc_deg_s2 for k in overlay.keys] == [9.5, None]
    bad = [{**keys[0], "acc_deg_s2": 0.0}, keys[1]]
    assert "ramps at more than 0" in str(overlay_from_json({"keys": bad}, settings))


def test_a_key_asks_the_tilt_s_ramp_up_to_clip_acc_max_and_the_neck_hears_it() -> None:
    """A paced nod's key asks the tilt for a ramp of its own (tilt_acc_deg_s2): refused whole
    above the knob (600 as shipped: no overlay tilts faster than any other move), passed to the
    driver beside its speed at or under it, and the way back carries the last key's."""
    target: dict[str, Any] = {"keys": [
        {"at_s": 0.0, "pan_rad": 0.0, "tilt_rad": math.radians(15.0), "speed_deg_s": 150.0,
         "tilt_acc_deg_s2": 1500.0},
        {"at_s": 0.3, "pan_rad": 0.0, "tilt_rad": 0.0, "speed_deg_s": 150.0,
         "tilt_acc_deg_s2": 1500.0},
    ]}  # fmt: skip
    assert GazeSettings().clip_acc_max_deg_s2 == 600.0
    refused = overlay_from_json(target, SETTINGS)
    assert isinstance(refused, str) and "past clip_acc_max_deg_s2 600" in refused
    bad = {"keys": [{**target["keys"][0], "tilt_acc_deg_s2": 0.0}]}
    assert overlay_from_json(bad, SETTINGS) == "a clip's key ramps at more than 0 deg/s^2"
    raised = replace(SETTINGS, clip_acc_max_deg_s2=1500.0)
    overlay = overlay_from_json(target, raised)
    assert isinstance(overlay, Overlay) and overlay.keys[0].tilt_acc_deg_s2 == 1500.0
    arb, head, out = arbiter()
    arb.settings = raised
    speaker_on(arb)
    nod = Look(GESTURE, (), IDLE, 0, 0.0, 5.0, kind="clip", overlay=overlay)
    assert arb.submit(nod, 1.0, out.append).status == "granted"
    arb.step(1.0)
    assert head.tilt_ramps[-1] == 1500.0 and head.ramps[-1] is None
    still_at(arb, plus(LEFT, 0, 15), 1.2)
    arb.step(1.3)
    assert_aim(head.writes[-1][0], LEFT)
    assert head.tilt_ramps[-1] == 1500.0
    still_at(arb, LEFT, 1.4)
    arb.step(1.5)
    assert out[-1].status == "done"


def test_a_followed_point_re_resolved_under_a_gesture_waits_for_its_end() -> None:
    """The speaker gaze's look on a person's point follows the cart (retarget): under a
    gesture's overlay the re-resolved aim is kept, not written; the gesture's end takes the
    head there."""
    arb, head, out = arbiter()
    arb.submit(Look(SPEAKER, (LEFT,), IDLE, 0, 6.0, 6.0, follow=True), 0.0)
    arb.step(0.0)
    still_at(arb, LEFT, 0.3)
    arb.step(0.4)
    arb.submit(clip(NOD), 1.0, out.append)
    arb.step(1.0)
    writes = len(head.writes)
    moved = plus(LEFT, -8, 0)
    assert arb.retarget(arb.pending()[0].id, moved, 1.1)
    assert len(head.writes) == writes, "no write under the gesture"
    still_at(arb, plus(LEFT, 0, 5), 1.15)
    arb.step(1.25)
    assert_aim(head.writes[-1][0], LEFT)  # the nod's way up: on the aim it started on
    still_at(arb, LEFT, 1.3)
    arb.step(1.4)
    assert out[-1].status == "done"
    arb.step(1.45)
    assert_aim(head.writes[-1][0], moved)
