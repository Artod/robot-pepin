"""The gaze arbiter (pepin.gaze): bands, TTLs, preemption, home, the blind interval, the drivers.

The head is a fake driver that records every write and answers when a test says so; time is
passed in, so every rule is a few steps of arithmetic.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import pytest

from pepin.gaze import (
    DARK_ENDED,
    DRIVING,
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
    LookTally,
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
from pepin.gaze_gate import DARK, DarkLog, FrameGate
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
        self.ceilings: list[dict[str, float]] = []

    def blocked(self, now: float) -> Refusal | None:
        return self.block

    def write(
        self, aim: Aim | None, *, speed: Speed, hold: bool, now: float, **ceilings: float
    ) -> None:
        self.writes.append((aim, hold, now))
        self.ceilings.append(ceilings)

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
    assert arb.state(1.2).blind, "no reading past the tail yet: the board's now is 1.0"
    arb.observe(HeadReading(LEFT.pan_rad, LEFT.tilt_rad, 1.12))
    assert not arb.state(1.2).blind, "the newest reading is past the tail (one clock, the board's)"
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
    arb.observe(HeadReading(HOME.pan_rad, HOME.tilt_rad, 2.7))
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


def test_a_head_moved_by_someone_else_is_adopted_and_brought_home() -> None:
    """A jog or a hand moved the head: at a drive's start the arbiter takes it as its own."""
    arb, head, _out = arbiter()
    assert not arb.adopt(0.0, math.radians(5))  # no reading: nothing to adopt
    arb.observe(HeadReading(HOME.pan_rad + 0.05, HOME.tilt_rad, 0.0))
    assert not arb.adopt(0.1, math.radians(5))  # within the tolerance of home
    arb.observe(HeadReading(math.radians(40), HOME.tilt_rad, 0.2))
    assert arb.adopt(0.3, math.radians(5))
    arb.step(0.3)
    assert head.writes == [(None, False, 0.3)]


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
    assert data["head_stamp"] is None, "no reading yet"


def test_the_state_carries_the_newest_reading_with_its_own_stamp() -> None:
    """The head in the state is the newest encoder reading, and ``head_stamp`` that reading's
    stamp (the board's clock, /neck/state's header), not the moment the state is asked for."""
    arb, _head, _out = arbiter()
    arb.observe(HeadReading(0.1, 0.4, 1_760_000_000.0123))
    arb.observe(HeadReading(0.2, 0.4, 1_760_000_000.0323))
    state = arb.state(1_760_000_000.09)
    assert state.head == Aim(0.2, 0.4) and state.head_stamp == 1_760_000_000.0323
    data = json.loads(state.to_json())
    assert data["head_stamp"] == pytest.approx(1_760_000_000.0323, abs=1e-4)
    assert data["pan_rad"] == pytest.approx(0.2)


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


def test_an_angles_look_past_the_tilt_stop_is_held_at_the_stop() -> None:
    """2026-10-14: a person standing over the robot asked the head for 21.9 deg up, 2 deg past
    the stop; the arbiter passed the ticks on and the base server refused the 1791-tick target,
    so the head never moved. Now every angles, scan and direction look keeps its tilt inside the
    reach; the pan stays as asked (the holder clamps it by travel)."""
    reach = Reach.of(CFG)
    up = reach.tilt[0]
    target = {"pan_rad": math.radians(-33.9), "tilt_rad": up - math.radians(2.0)}
    past = {"kind": "angles", "band": PERSON, "source": "t", "frames": 0, "target": target}
    look = look_from_json(past, GazeSettings(), CFG, to_base)
    assert isinstance(look, Look), look
    (view,) = look.views
    assert view.tilt_rad == pytest.approx(up)
    assert view.pan_rad == pytest.approx(math.radians(-33.9))
    views = {"views": [target, {"pan_rad": 0.0, "tilt_rad": 9.0}]}
    look = look_from_json(dict(past, kind="scan", target=views), GazeSettings(), CFG, to_base)
    assert isinstance(look, Look)
    assert [v.tilt_rad for v in look.views] == pytest.approx([up, reach.tilt[1]])


def test_a_pan_past_the_reach_goes_to_the_end_nearer_the_head_never_across_the_front() -> None:
    """2026-10-13: a face a hair past straight back on the left (+184) reads -176 by the wrap;
    by number the right end is nearer, and the head at the left end swept 320 deg through the
    front to look at the same person from the other side. With the head's own pan the clamp
    takes the end the head is at; inside the reach nothing changes."""
    reach = Reach.of(CFG)
    lo, hi = reach.pan
    behind = Aim(math.radians(-176.0), HOME.tilt_rad)
    assert reach.clamp(behind).pan_rad == pytest.approx(lo), "by number alone: the right end"
    assert reach.clamp(behind, from_pan=math.radians(153.0)).pan_rad == pytest.approx(hi)
    assert reach.clamp(behind, from_pan=math.radians(-100.0)).pan_rad == pytest.approx(lo)
    assert reach.clamp(behind, from_pan=0.0).pan_rad == pytest.approx(hi), "from home: 164 < 167"
    inside = Aim(math.radians(-150.0), HOME.tilt_rad)
    kept = reach.clamp(inside, from_pan=math.radians(153.0))
    assert kept.pan_rad == pytest.approx(inside.pan_rad), "inside the reach: as asked"
    assert kept.tilt_rad == pytest.approx(HOME.tilt_rad)


def test_a_point_past_the_reach_is_looked_at_from_the_end_nearer_the_head() -> None:
    """A person a hair past straight back on the left, asked as a point: resolved to -177 deg
    by the wrap, looked at from the left end when the head stands left (its own pan), as an
    angles look's holder clamps it; without the head's pan, by number."""
    reach = Reach.of(CFG)
    behind = {"source": "voice.speaker", "kind": "point", "frames": 0,
              "target": {"frame": "base_link", "x": -2.0, "y": -0.1, "z": 1.2}}  # fmt: skip
    left = look_from_json(behind, SETTINGS, CFG, to_base, from_pan=math.radians(150.0))
    assert isinstance(left, Look) and left.views[0].pan_rad == pytest.approx(reach.pan[1])
    plain = look_from_json(behind, SETTINGS, CFG, to_base)
    assert isinstance(plain, Look) and plain.views[0].pan_rad == pytest.approx(reach.pan[0])
    assert left.point == ("base_link", -2.0, -0.1, 1.2) and not left.follow


def test_a_point_look_that_follows_is_written_again_as_its_point_moves_under_the_head() -> None:
    """The speaker gaze's person as a point that follows: once it holds the head (answered,
    held for its TTL) a view re-resolved within the settle tolerance writes nothing (TF's
    jitter, wheels that slip and turn no TF), one past it is written at once with the look's
    own ceilings; a look that does not follow, or another's id, is never retargeted."""
    arb, head, out = arbiter()
    data = {"source": "voice.speaker", "kind": "point", "band": 5, "frames": 0, "hold": True,
            "follow": True, "speed_deg_s": 60.0, "acc_deg_s2": 400.0,
            "target": {"frame": "base_link", "x": 2.0, "y": 0.0, "z": 1.0}}  # fmt: skip
    asked = look_from_json(data, SETTINGS, CFG, to_base)
    assert isinstance(asked, Look) and asked.follow and asked.point is not None
    arb.submit(asked, 0.0, out.append)
    arb.step(0.0)
    (first,) = arb.pending()
    view = first.views[0]
    arrive(arb, head, view, 0.4)
    arb.step(0.4)
    assert out and out[0].status == "done" and arb.holds("voice.speaker"), "answered, held"
    following = arb.following()
    assert following is not None and following.id == asked.id
    writes = len(head.writes)
    nudged = Aim(view.pan_rad + math.radians(0.5), view.tilt_rad)
    assert not arb.retarget(asked.id, nudged, 0.5), "within the settle tolerance"
    assert len(head.writes) == writes
    turned = Aim(view.pan_rad - math.radians(20.0), view.tilt_rad)  # the cart turned 20 left
    assert arb.retarget(asked.id, turned, 0.6)
    assert head.writes[-1] == (turned, True, 0.6) and arb.counts["follows"] == 1
    assert head.ceilings[-1] == {"speed_deg_s": 60.0, "acc_deg_s2": 400.0}, "the look's own"
    assert arb.pending()[0].views == (turned,) and arb.state(0.6).target == turned
    assert not arb.retarget("someone else", view, 0.7)
    other, _head, _out = arbiter()
    other.submit(look(frames=0, dwell_s=5.0), 0.0)
    other.step(0.0)
    assert other.following() is None, "an angles look holds joint angles"
    refused = look_from_json({**data, "kind": "angles", "target": {"pan_rad": 0.1,
                              "tilt_rad": 0.2}}, SETTINGS, CFG, to_base)  # fmt: skip
    assert isinstance(refused, str) and "follow is a point look's" in refused


def test_a_bearing_a_turn_carries_past_the_reach_holds_the_end_never_swept_round_the_front() -> (
    None
):
    """2026-10-15, run 140: a 180 deg turn left with Artem at -20 deg carried his bearing past
    the right end (-167) and through the 29 deg behind the reach to +160, inside the reach on the
    other side; resolved afresh (by the head's pan or by number) that is a pan the head reaches
    only through the front, and it swept 288 deg after him. Carried round with the turn, his
    bearing stays past the right end: the head is written down to -167 and held there, nothing
    on the left side ever written; the turn coming back brings him in from the right and the
    follow goes on; a new look starts its own carry from its view."""
    reach = Reach.of(CFG)
    lo = reach.pan[0]
    assert reach.clamp(Aim(math.radians(160.0), 0.3), from_pan=lo).pan_rad == pytest.approx(
        math.radians(160.0)
    ), "afresh: inside the reach, the far side of the front"
    arb = Arbiter(FakeHead(), HOME, SETTINGS, reach)
    head = arb.driver
    assert isinstance(head, FakeHead)
    data: dict[str, Any] = {
        "source": "voice.speaker", "kind": "point", "band": 5, "frames": 0, "hold": True,
        "follow": True, "target": {"frame": "base_link", "x": 1.88, "y": -0.68, "z": 1.0},
    }  # fmt: skip
    asked = look_from_json(data, SETTINGS, CFG, to_base)
    assert isinstance(asked, Look)
    arb.submit(asked, 0.0)
    arb.step(0.0)
    view = arb.pending()[0].views[0]
    arrive(arb, head, view, 0.4)
    arb.step(0.4)
    start = math.degrees(view.pan_rad)
    t, pans = 0.4, []
    for turned in range(3, 181, 3):  # Nav2's Spin, 3 deg a step, his bearing falls
        t += 0.09
        bearing = math.radians(start - turned)
        arb.follow(asked.id, Aim(math.atan2(math.sin(bearing), math.cos(bearing)), 0.3), t)
        pans.append(math.degrees(head.writes[-1][0].pan_rad))
    assert all(b <= a + 1e-9 for a, b in itertools.pairwise(pans)), "never back, never across"
    assert pans[-1] == pytest.approx(math.degrees(lo), abs=1.0) and max(pans) < start
    held = arb.state(t).target
    assert held is not None and math.degrees(held.pan_rad) == pytest.approx(
        math.degrees(lo), abs=1.0
    ), "held at the end he went past (within the settle tolerance)"
    assert arb.counts["follow_past_reach"] >= 10
    for back in range(3, 61, 3):  # the cart turns back 60: he comes in from the right
        t += 0.09
        bearing = math.radians(start - 180.0 + back)
        arb.follow(asked.id, Aim(math.atan2(math.sin(bearing), math.cos(bearing)), 0.3), t)
    assert math.degrees(head.writes[-1][0].pan_rad) == pytest.approx(start - 120.0, abs=1.5)
    again = look_from_json({**data, "id": "next"}, SETTINGS, CFG, to_base,
                           from_pan=head.writes[-1][0].pan_rad)  # fmt: skip
    assert isinstance(again, Look)
    arb.submit(again, t)
    arb.step(t)
    assert arb.follow("next", Aim(math.radians(-40.0), 0.3), t + 0.1), "its own carry, afresh"
    assert arb.pending()[0].views[0].pan_rad == pytest.approx(math.radians(-40.0))
    assert not arb.follow(asked.id, Aim(0.0, 0.3), t + 0.2), "the old look holds nothing"


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
    assert (angles.band, angles.frames, angles.ttl_s, angles.dwell_s) == (PERSON, 3, 10.0, 0.0)
    held = look_from_json(
        {"source": "llm.look", "target": {"pan_rad": 0.5, "tilt_rad": 0.4}, "hold": True},
        SETTINGS,
        CFG,
        to_base,
    )
    assert isinstance(held, Look) and held.dwell_s == held.ttl_s == 10.0
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
    head.write(RIGHT, speed="saccade", hold=True, now=0.7, speed_deg_s=200.0, acc_deg_s2=650.0)
    assert (wire.lines[-1]["speed_deg_s"], wire.lines[-1]["acc_deg_s2"]) == (200.0, 650.0)
    wire.lines.pop()  # an eased look: the board's trapezoid at those ceilings
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


# ---- glances and the looks' books -------------------------------------------------------------
def glance(
    source: str = "nav.reverse",
    aim: Aim = LEFT,
    frames: int = 3,
    hold_s: float = 0.3,
    ttl_s: float = 0.5,
    band: int = DRIVING,
) -> Look:
    return Look(source, (aim,), band, frames, ttl_s, ttl_s, hold_s=hold_s)


def test_a_glance_is_answered_at_its_frames_or_at_its_hold_whichever_first() -> None:
    arb, head, out = arbiter()
    arb.submit(glance(hold_s=0.3), 0.0, out.append)
    arb.step(0.0)
    arrive(arb, head, LEFT, 0.2)
    arb.step(0.2)
    arb.frame(0.35)
    arb.frame(0.45)
    arb.step(0.45)
    assert out == []  # two frames, 0.25 s settled
    arb.step(0.5)
    assert out[0].status == "done" and out[0].frames_seen == 2  # the hold ran out first
    arb2, head2, out2 = arbiter()
    arb2.submit(glance(hold_s=5.0), 0.0, out2.append)
    arb2.step(0.0)
    arrive(arb2, head2, LEFT, 0.2)
    arb2.step(0.2)
    for stamp in (0.35, 0.45, 0.55):
        arb2.frame(stamp)
    arb2.step(0.56)
    assert out2[0].status == "done" and out2[0].frames_seen == 3  # the frames came first


def test_a_glance_under_way_outlives_its_ttl_a_release_and_its_own_band() -> None:
    arb, head, out = arbiter()
    arb.submit(glance(ttl_s=0.5), 0.0, out.append)
    arb.step(0.0)
    path: list[Outcome] = []
    arb.submit(look("nav.path", (RIGHT,), DRIVING, 0, 0.5, 0.5, preempt=True), 0.6, path.append)
    arb.step(0.8)  # its TTL is long gone, the head still turning: kept
    assert arb.glancing("nav.reverse") and out == [] and len(head.writes) == 1
    assert arb.release(0.85, "the drive's end", keep=lambda r: False) == 1  # the path look
    assert path[0].status == "preempted" and arb.glancing("nav.reverse")
    arrive(arb, head, LEFT, 1.0)
    arb.step(1.0)
    for stamp in (1.15, 1.2, 1.25):
        arb.frame(stamp)
    arb.step(1.3)
    assert out[0].status == "done" and out[0].frames_seen == 3
    assert arb.pending() == [] and not arb.glancing("nav.reverse")  # released once done
    arb.step(1.35)
    assert head.writes[-1][0] is None  # home


def test_a_better_band_takes_the_head_from_a_glance() -> None:
    arb, head, out = arbiter()
    arb.submit(glance(), 0.0, out.append)
    arb.step(0.0)
    arb.submit(look("nav.stall", (RIGHT,), NAVIGATION), 0.1)
    arb.step(0.1)
    assert out[0].status == "preempted" and head.writes[-1][0] == RIGHT


def test_a_newer_request_of_a_glancing_source_waits_for_the_glance() -> None:
    arb, head, out = arbiter()
    arb.submit(glance(), 0.0, out.append)
    arb.step(0.0)
    arb.submit(glance(aim=RIGHT), 0.1)
    arb.step(0.1)
    assert head.writes[-1][0] == LEFT and arb.glancing("nav.reverse")
    arrive(arb, head, LEFT, 0.3)
    arb.step(0.3)
    arb.renew("nav.reverse", 0.5)  # the node renews its source every period
    arb.step(0.65)  # held 0.35 s: done, and dropped as replaced
    assert out[0].status == "done"
    arb.step(0.7)
    assert head.writes[-1][0] == RIGHT


def test_its_own_source_withdraws_a_glance_under_way_and_the_next_look_takes_the_head() -> None:
    """The reverse look let go at the first forward command: mid-swing, where a release waits."""
    arb, head, out = arbiter()
    arb.submit(glance(hold_s=0.6), 0.0, out.append)
    arb.step(0.0)
    assert arb.holds("nav.reverse") and not arb.holds("nav.path")
    path: list[Outcome] = []
    arb.submit(look("nav.path", (RIGHT,), DRIVING, 0, 0.5, 0.5), 0.3, path.append)
    assert arb.withdraw("nav.path", 0.3, "nothing") == 1 and path[0].status == "preempted"
    arb.submit(look("nav.path", (RIGHT,), DRIVING, 0, 0.5, 0.5), 0.3)
    assert arb.withdraw("nav.reverse", 0.3, "the first forward command") == 1
    assert out[0].status == "preempted" and out[0].reason == "by the first forward command"
    arb.step(0.3)
    assert head.writes[-1][0] == RIGHT and arb.holds("nav.path")
    assert arb.withdraw("nav.reverse", 0.35, "again") == 0  # nothing of it left
    booked = arb.take_looks(2.0)
    assert [(b.source, b.ended, b.still_from) for b in booked] == [
        ("nav.reverse", "preempted", None)  # cut before it settled: it wrote no frames
    ]


def test_a_look_without_a_hold_is_no_glance() -> None:
    """hold_s 0 (the baseline): the TTL ends it mid-move and a release cuts it, as before."""
    arb, _head, out = arbiter()
    arb.submit(glance(hold_s=0.0), 0.0, out.append)
    arb.step(0.0)
    assert not arb.glancing("nav.reverse")
    arb.step(0.6)
    assert out[0].status == "expired"


def test_every_look_that_held_the_head_is_booked_with_the_frames_it_held_still_for() -> None:
    arb, head, _out = arbiter()
    arb.submit(look("nav.path", (LEFT,), DRIVING, 0, 0.5, 0.5), 0.0)
    arb.step(0.0)
    arrive(arb, head, LEFT, 0.2)
    arb.step(0.2)
    arb.frame(0.25)  # inside the blur period: not still
    for stamp in (0.35, 0.45, 0.55):
        arb.frame(stamp)
    arb.renew("nav.path", 0.4)
    arb.submit(look("nav.other", (RIGHT,), DRIVING, 0, 0.5, 0.5), 0.6)
    arb.step(0.6)  # the answered path look yields to its band's newcomer
    arb.frame(0.58)  # late: taken while the path look still held the head
    arb.frame(0.65)  # the other look's head is moving
    arb.submit(look("nav.stall", (HOME,), NAVIGATION), 0.8)
    arb.step(0.8)  # takes the head from the other look mid-saccade
    arb.submit(look("llm.look", (), PERSON, kind="home"), 0.9)
    assert arb.take_looks(1.0) == []  # still counting late frames
    booked = arb.take_looks(2.5)
    assert [(b.source, b.frames, b.ended) for b in booked] == [
        ("nav.path", 4, "preempted"),
        ("nav.other", 0, "preempted"),
    ]
    assert booked[0].still_from == pytest.approx(0.3) and booked[1].still_from is None
    assert arb.take_looks(3.0) == []  # once each
    tally = LookTally()
    assert tally.text() == "looks none"
    for one in booked:
        tally.add(one)
    assert tally.text() == (
        "looks nav.other 1 (0 frames), nav.path 1 (4 frames); 1 WROTE NO FRAMES:"
        " nav.other -30/40 deg 0.20 s preempted"
    )
    tally.looks = tally.looks[:1]
    assert tally.text() == "looks nav.path 1 (4 frames); every one wrote frames"


def test_the_neck_target_driver_goes_home_slowly_after_a_drive() -> None:
    wire = Wire()
    head = NeckTargetHead(
        CFG, wire.send, slow_deg_s=lambda: 20.0, renew_s=lambda: 0.5, return_deg_s=lambda: 45.0
    )
    head.write(None, speed="return", hold=False, now=0.0)
    assert wire.lines[-1]["speed_deg_s"] == 45.0
    target_head(wire).write(None, speed="return", hold=False, now=0.0)
    assert "speed_deg_s" not in wire.lines[-1]  # no return speed given: the board's own


# ---- dark frames: the depth gate's darkness rule -----------------------------------------------
def _stall_look(arb: Arbiter, head: FakeHead, out: list[Outcome]) -> None:
    """A navigation look (3 frames, TTL 3 s) that settles on LEFT at board second 0.0."""
    arb.submit(look("nav.stall", band=NAVIGATION, ttl_s=3.0), 0.0, out.append)
    arb.step(0.0)
    arrive(arb, head, LEFT, 0.0)
    arb.step(0.0)


def test_a_look_over_dark_frames_gives_up_after_the_patience_not_the_ttl() -> None:
    """Twenty frames at 10 fps that the depth gate drops as dark (a room the auto exposure cannot
    light): the look ends 1.5 s into the dark, expired as 'dark, no frames', booked so and
    counted, instead of holding the head for its 3 s TTL."""
    arb, head, out = arbiter()
    _stall_look(arb, head, out)
    ended = None
    for k in range(20):
        stamp = 0.15 + 0.1 * k
        arb.dark_frame(stamp)
        arb.step(stamp)
        if out and ended is None:
            ended = stamp
    assert ended == pytest.approx(1.65), "1.5 s from the first dark frame, not 3.0 s"
    assert out[0].status == "expired" and out[0].reason.startswith(DARK_ENDED)
    assert out[0].frames_seen == 0 and arb.counts["dark"] == 1
    arb.step(5.0)
    booked = arb.take_looks(5.0)
    assert [b.ended for b in booked] == [DARK_ENDED]
    tally = LookTally()
    tally.add(booked[0])
    assert "1 WROTE NO FRAMES: nav.stall +30/40 deg" in tally.text()
    assert tally.text().endswith(DARK_ENDED)


def test_a_look_whose_frames_turn_light_completes_as_ever() -> None:
    """A turn from a window: 1 s of dark frames, then the eyes catch up and the frames are fused
    again; the look is answered done on its three frames, as in a lit room."""
    arb, head, out = arbiter()
    _stall_look(arb, head, out)
    for k in range(9):
        arb.dark_frame(0.15 + 0.1 * k)
        arb.step(0.15 + 0.1 * k)
    assert out == []
    for stamp in (1.05, 1.15, 1.25):
        arb.frame(stamp)
        arb.step(stamp)
    assert out[0].status == "done" and out[0].frames_seen == 3
    assert arb.counts["dark"] == 0


def test_without_patience_a_dark_look_waits_for_its_ttl() -> None:
    head = FakeHead()
    settings = GazeSettings(frames=3, frame_period_s=0.1, dark_patience_s=0.0)
    arb, out = Arbiter(head, HOME, settings), []
    _stall_look(arb, head, out)
    for k in range(29):
        arb.dark_frame(0.15 + 0.1 * k)
        arb.step(0.15 + 0.1 * k)
    assert out == []
    arb.step(3.05)
    assert out[0].status == "expired" and not out[0].reason.startswith(DARK_ENDED)


def test_a_look_s_still_frames_do_not_advance_on_frames_the_gate_finds_dark() -> None:
    """The gate and the look together, as depth_stream and depth_fusion wire them: each frame's
    brightness through the darkness rule (floor 10, leaves at 12), a dark one told to the
    arbiter as dark and never fused, a passed one fused. The auto exposure ramps up after a turn
    into the room: the look's count starts only once the picture is past 12 grey."""
    arb, head, out = arbiter()
    gate = FrameGate(exposure_s=0.0, dark_on=True, dark=DarkLog(10.0, 2.0))
    _stall_look(arb, head, out)
    ramp = (4.0, 5.0, 6.0, 8.0, 9.0, 10.5, 11.5, 13.0, 25.0, 40.0, 60.0)
    fused = []
    for k, level in enumerate(ramp):
        stamp = 0.15 + 0.1 * k
        gate.observe_brightness(stamp, level)
        if gate.verdict(stamp, 0.0) == DARK:
            arb.dark_frame(stamp)
        else:
            arb.frame(stamp)
            fused.append(stamp)
        arb.step(stamp)
        if len(fused) < 3:
            assert out == [], f"answered at {stamp:.2f} on {len(fused)} still frames"
    assert fused[0] == pytest.approx(0.85), "10.5 and 11.5 are still dark: the hysteresis"
    assert out[0].status == "done" and out[0].frames_seen == 3
