"""The head's gestures as the gaze arbiter's overlay: a clip's keyframes as keys written one move
early at the speed that lands them on time, the idle and loop marks, the door's answers and
releases, every head atom and composition within the arbiter's bounds and the neck's ramps, the
engine's head through it, and the speaker gaze keeping the head while a gesture plays."""

from __future__ import annotations

import itertools
import json
import logging
import math
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from pepin.arm_motion import load_arm_motion
from pepin.expression.animation import Clip, Scene, Track
from pepin.expression.animator import EngineConfig, GestureEngine
from pepin.expression.book import load_book, with_rest
from pepin.expression.door import AWAY, OverlayDoor
from pepin.expression.fakes import FakeExecutor, FakeHeadDoor, FakePlayout
from pepin.expression.head import GESTURE, IDLE_LOOK, SPEAKER, ClipLook
from pepin.expression.listeners import head_clip
from pepin.expression.markers import Marker, Part
from pepin.expression.parts import (
    GLIDE,
    HEAD_MIN_SPEED_DEG_S,
    Absent,
    HeadPart,
    NeckLimits,
    Serial,
    clip_look,
    keyframes,
    move_speed,
    needed_acc,
)
from pepin.expression.render import BookRenderer
from pepin.gaze import (
    IDLE,
    Aim,
    Arbiter,
    GazeSettings,
    HeadReading,
    Look,
    Outcome,
    Refusal,
    Speed,
    home_aim,
    look_from_json,
    overlay_from_json,
)
from pepin.neck import ACC_UNIT_DEG_S2, NeckConfig

CONFIG = Path(__file__).resolve().parents[2] / "config"
BOOK = with_rest(
    load_book(CONFIG / "animation.json"), load_arm_motion(CONFIG / "arm.json").rest_deg
)
NECK = NeckConfig.from_json(CONFIG / "neck.json")
LIMITS = NeckLimits.of(NECK.motion)
KNOBS = {
    k: v["default"]
    for k, v in json.loads((CONFIG / "knobs.json").read_text())["gaze"].items()
    if isinstance(v, dict)
}
BOUNDS = GazeSettings(
    clip_max_deg=KNOBS["clip_max_deg"],
    clip_max_hz=KNOBS["clip_max_hz"],
    clip_still_deg=KNOBS["clip_still_deg"],
)
EVERYTHING = Scene(frozenset({"arm", "head", "face", "body"}))
RENDERER = BookRenderer(BOOK)


def nod(amp: float = 5.0, period: float = 0.4, layer: str = "gesture") -> Clip:
    params = {"amp_deg": amp, "period_s": period, "repeats": 1.0}
    return RENDERER.atom_clip("head_nod", params, layer=layer)  # type: ignore[arg-type]


def degrees(key: Mapping[str, float]) -> tuple[float, float, float, float]:
    return (
        key["at_s"],
        math.degrees(key["pan_rad"]),
        math.degrees(key["tilt_rad"]),
        key["speed_deg_s"],
    )


# -- the adapter: a clip as the arbiter's overlay -----------------------------------------------


def test_the_neck_limits_are_config_neck_json_s() -> None:
    assert LIMITS == NeckLimits(299.0, 2232.0, 600.0) == NeckLimits()


def test_move_speed_lands_a_move_on_time_at_its_ramp_or_goes_as_fast_as_the_ramp_allows() -> None:
    for travel, seconds, acc in [(5.0, 0.2, 600.0), (20.0, 0.4, 2232.0), (0.6, 1.0, 600.0)]:
        v = move_speed(travel, seconds, acc)
        assert travel / v + v / acc == pytest.approx(seconds)  # a trapezoid that ends on time
        assert needed_acc(travel, seconds) <= acc
    assert move_speed(10.0, 0.1, 600.0) == pytest.approx(math.sqrt(600.0 * 10.0))  # too short
    assert needed_acc(10.0, 0.1) > 600.0 and move_speed(0.0, 1.0, 600.0) == 0.0


def test_each_keyframe_is_written_one_move_early_with_the_speed_that_lands_it_on_time() -> None:
    """A nod's keyframes (0, 0) (0.2, 5 down) (0.4, 0): the move down is written at 0 and
    reaches 5 deg at 0.2 s, the stroke; the move back is written at 0.2."""
    look = clip_look(nod(5.0, 0.4), GESTURE)
    assert look is not None
    assert look["source"] == GESTURE and look["kind"] == "clip" and look["band"] == IDLE
    assert look["ttl_s"] == pytest.approx(0.4 + 1.0)
    assert "idle" not in look["target"] and "loop" not in look["target"]
    (at0, pan0, tilt0, v0), (at1, pan1, tilt1, v1) = map(degrees, look["target"]["keys"])
    assert (at0, pan0, tilt0) == (0.0, 0.0, pytest.approx(5.0))
    assert (at1, pan1, tilt1) == (0.2, 0.0, 0.0)
    assert v0 == v1 == pytest.approx(move_speed(5.0, 0.2, 600.0), abs=0.01)
    assert 5.0 / v0 + v0 / 600.0 == pytest.approx(0.2, abs=1e-3)  # down by the stroke


def test_a_hold_writes_nothing_and_a_merged_clip_s_steps_add_up() -> None:
    glance = RENDERER.atom_clip(
        "head_glance", {"side_deg": 15.0, "up_deg": 5.0, "period_s": 2.0}, mirrored=True
    )
    look = clip_look(glance, GESTURE)
    assert look is not None
    keys = [degrees(k) for k in look["target"]["keys"]]
    assert [k[:3] for k in keys] == [
        (0.0, pytest.approx(-15.0), pytest.approx(-5.0)),
        (1.6, 0.0, 0.0),
    ]
    assert keys[0][3] == pytest.approx(move_speed(15.0, 0.4, 2232.0), abs=0.01)  # the pan's
    two = Clip(
        "head",
        0.0,
        0.0,
        1.0,
        (
            Track("tilt_deg", "additive", (0.0, 0.5, 1.0), (0.0, 2.0, 0.0)),
            Track("tilt_deg", "additive", (0.0, 0.5, 1.0), (0.0, 3.0, 0.0)),
        ),
    )
    assert keyframes(two)[1] == (0.5, 0.0, 5.0)
    assert clip_look(Clip("head", 0.0, 0.0, 1.0), GESTURE) is None  # no head track
    still = Clip("head", 0.0, 0.0, 1.0, (Track("pan_deg", "additive", (0.0, 1.0), (0.0, 0.0)),))
    assert clip_look(still, GESTURE) is None  # moves nothing


def test_the_idle_layer_s_clip_is_marked_idle_and_a_looping_atom_loops() -> None:
    drift = clip_look(RENDERER.render_atom("head_drift", 5), IDLE_LOOK)
    assert drift is not None and drift["target"].get("idle") is True
    assert "loop" not in drift["target"]
    breath = RENDERER.render_atom("breath", 5)
    assert breath.loop and breath.layer == "idle"
    look = clip_look(breath, IDLE_LOOK)
    assert look is not None and look["target"].get("loop") is True
    keys = look["target"]["keys"]
    assert keys[-1]["at_s"] == pytest.approx(12.0)  # three breaths, then round again
    assert {**keys[-1], "at_s": 0.0} == keys[0]  # the arbiter's loop: the last key is the first
    overlay = overlay_from_json(dict(look["target"]), BOUNDS)
    assert not isinstance(overlay, str) and overlay.loop and overlay.idle
    assert math.degrees(overlay.reach_rad) <= BOUNDS.clip_still_deg  # never blinds the head


def test_the_idle_layer_s_keys_glide_and_a_gesture_s_strike_at_the_neck_s_ramp() -> None:
    """Each key of an idle clip (the melody, the breath, the drift) carries its own ramp, the
    gentlest that lands its move on time (GLIDE times the least, never under the Acceleration
    register's unit), and the speed that ends the move on time at it: the servo eases from key
    to key; a gesture's keys carry none (the neck's own ramp, a nod's stroke on its word)."""
    melody = head_clip(((0.0, 0.0, 0.0), (0.6, 0.0, -0.9), (1.3, 0.0, 0.5), (2.0, 0.0, 0.0)), 0.0,
                       layer="idle")  # fmt: skip
    for clip in (melody, RENDERER.render_atom("breath", 5), RENDERER.render_atom("head_drift", 5)):
        look = clip_look(clip, IDLE_LOOK)
        assert look is not None
        pairs = itertools.pairwise(keyframes(clip))
        moves = [(b[0] - a[0], max(abs(b[1] - a[1]), abs(b[2] - a[2])))
                 for a, b in pairs if (b[1], b[2]) != (a[1], a[2])]  # fmt: skip
        for key, (seconds, travel) in zip(look["target"]["keys"], moves, strict=False):
            ramp = key["acc_deg_s2"]
            assert ramp >= ACC_UNIT_DEG_S2 - 1e-9
            assert ramp <= max(ACC_UNIT_DEG_S2, GLIDE * needed_acc(travel, seconds)) + 0.01
            speed = move_speed(travel, seconds, ramp)
            assert key["speed_deg_s"] == pytest.approx(max(HEAD_MIN_SPEED_DEG_S, speed), abs=0.01)
        overlay = overlay_from_json(dict(look["target"]), BOUNDS)
        assert not isinstance(overlay, str) and all(k.acc_deg_s2 for k in overlay.keys)
    nod = clip_look(head_clip(((0.0, 0.0, 0.0), (0.2, 0.0, 3.0), (0.4, 0.0, 0.0)), 0.2), GESTURE)
    assert nod is not None and all("acc_deg_s2" not in k for k in nod["target"]["keys"])


def test_the_glide_s_ramp_reaches_the_neck_with_each_key() -> None:
    """The arbiter writes each key of an idle overlay with its own ramp beside its speed (the
    board's neck_target takes both); the idle layer's breath through the door, end to end."""
    clock = Clock()
    driver = TargetHead()
    arbiter = Arbiter(driver, home_aim(NECK), GazeSettings(), None)
    door = OverlayDoor(client=lambda s, a=arbiter, c=clock: ArbiterClient(s, a, c))
    head = HeadPart(door, run=inline)
    breath = RENDERER.render_atom("breath", 5)
    clock.now = 1.0
    assert head.play(breath).ok
    while clock.now < 1.0 + breath.duration_s:
        clock.now += 0.05
        arbiter.step(clock.now)
    look = clip_look(breath, IDLE_LOOK)
    assert look is not None
    ramps = {k["acc_deg_s2"] for k in look["target"]["keys"]}
    written = [r for r in driver.ramps if r is not None]
    assert written and set(written) <= ramps


# -- authoring: nothing the arbiter refuses for a reason known in advance -----------------------


def _problems(clip: Clip) -> list[str]:
    """Why the arbiter would refuse ``clip``'s overlay (its own check, the knobs' defaults), or
    which keyframe needs more than the neck's ramps to land on time."""
    look = clip_look(clip, GESTURE, LIMITS)
    if look is None:
        return []
    refusal = overlay_from_json(dict(look["target"]), BOUNDS)
    found = [refusal] if isinstance(refusal, str) else []
    for (t0, pan0, tilt0), (t1, pan1, tilt1) in itertools.pairwise(keyframes(clip)):
        for axis, travel, ramp in (
            ("pan", pan1 - pan0, LIMITS.pan_acc_deg_s2),
            ("tilt", tilt1 - tilt0, LIMITS.tilt_acc_deg_s2),
        ):
            need = needed_acc(travel, t1 - t0)
            if need > ramp * (1.0 + 1e-9):
                found.append(f"{axis} {abs(travel):.1f} deg in {t1 - t0:.3f} s needs {need:.0f}")
    return found


def _head_boxes() -> list[tuple[str, str, dict[str, Any]]]:
    """Every head atom's parameter box: its own, and each composition step's over it."""
    boxes = [
        (name, name, dict(atom.params)) for name, atom in BOOK.atoms.items() if atom.part == "head"
    ]
    for composition in BOOK.compositions.values():
        for v, variant in enumerate(composition.variants):
            for s, step in enumerate(variant.steps):
                atom = BOOK.atoms[step.atom]
                if atom.part == "head":
                    where = f"{composition.name}[{v}].steps[{s}]"
                    boxes.append((where, atom.name, {**atom.params, **step.params}))
    return boxes


def test_every_head_atom_at_the_corners_of_its_ranges_is_within_the_arbiter_and_the_ramps() -> None:
    """Largest and smallest amplitudes, shortest and longest periods, mirrored or not, as a
    gesture and as the idle layer: within clip_max_deg, never a tilt turning back faster than
    clip_max_hz, every keyframe reachable on time at 600 deg/s^2 (tilt), 2232 (pan)."""
    assert {
        "head_nod",
        "head_shake",
        "head_back",
        "head_cock",
        "head_glance",
        "recoil",
        "head_drift",
    } <= {atom for _, atom, _ in _head_boxes()}
    for where, atom, box in _head_boxes():
        names = sorted(box)
        for corner in itertools.product(*[(box[n].lo, box[n].hi) for n in names]):
            params = dict(zip(names, corner, strict=True))
            for mirrored, layer in itertools.product((False, True), ("gesture", "idle")):
                clip = RENDERER.atom_clip(atom, params, mirrored=mirrored, layer=layer)  # type: ignore[arg-type]
                assert _problems(clip) == [], f"{where} at {params}"


def test_the_breath_is_one_slow_oscillation_with_a_stop_only_at_its_turns() -> None:
    """The looping breath (2 deg at 0.25 Hz): two keys a cycle, up over half the period and
    down over the other half, so between its turns the head never stops (the four-key cycle
    before stopped at the middle every pass: a floating drift, 2026-10-13); each key a glide
    at the Acceleration register's gentlest unit, ~2.3 deg/s, still (never blinds)."""
    breath = RENDERER.render_atom("breath", 5)
    frames = keyframes(breath)
    tilts = [t for _, _, t in frames]
    assert tilts == pytest.approx([0.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0])  # three cycles
    assert [b[0] - a[0] for a, b in itertools.pairwise(frames)] == pytest.approx([2.0] * 6)
    look = clip_look(breath, IDLE_LOOK)
    assert look is not None and look["target"]["loop"] is True
    keys = look["target"]["keys"]
    assert len(keys) == 7 and keys[-1]["at_s"] == pytest.approx(12.0)  # the loop's closing key
    for key in keys[1:-1]:  # 4 deg in 2 s: the gentlest ramp, a slow glide
        assert key["acc_deg_s2"] == pytest.approx(ACC_UNIT_DEG_S2, abs=0.01)
        assert 2.0 <= key["speed_deg_s"] <= 2.6
    overlay = overlay_from_json(dict(look["target"]), BOUNDS)
    assert not isinstance(overlay, str) and overlay.idle
    assert math.degrees(overlay.reach_rad) <= BOUNDS.clip_still_deg  # still: never blinds
    assert _problems(breath) == []


def test_every_composition_s_head_clip_plays_whole_through_the_arbiter() -> None:
    """Steps of one part merge into one clip, their offsets adding up: drawn many times, every
    variant's head clip passes the same checks (the head's steps never undo each other)."""
    for composition in BOOK.compositions.values():
        for variant in range(len(composition.variants)):
            for seed in range(60):
                performance = RENDERER.render(composition.name, EVERYTHING, seed, variant=variant)
                for clip in performance.clips:
                    if clip.part == "head":
                        assert _problems(clip) == [], f"{composition.name} {variant} seed {seed}"


def test_the_head_moves_in_the_compositions_the_research_puts_it_in() -> None:
    def head_atoms(intent: str) -> set[str]:
        composition = BOOK.compositions[intent]
        return {
            s.atom
            for v in composition.variants
            for s in v.steps
            if BOOK.atoms[s.atom].part == "head"
        }

    assert "head_nod" in head_atoms("greeting") and "head_nod" in head_atoms("agreement")
    assert head_atoms("disagreement") == {"head_shake"}
    assert "head_cock" in head_atoms("doubt") and head_atoms("thinking") == {"head_glance"}
    assert {"head_back", "head_nod"} <= head_atoms("laugh") and head_atoms("surprise") == {"recoil"}
    assert head_atoms("sympathy") == {"head_drop", "head_cock"}
    for intent in (
        "greeting",
        "agreement",
        "disagreement",
        "doubt",
        "thinking",
        "laugh",
        "surprise",
        "sympathy",
    ):
        clips = RENDERER.render(intent, EVERYTHING, 11, variant=0).clips
        assert any(c.part == "head" for c in clips), intent


# -- the head part ------------------------------------------------------------------------------


def inline(fn: Any) -> None:
    fn()


def test_the_head_part_posts_each_layer_as_its_source_and_releases_it_over_the_fade() -> None:
    door = FakeHeadDoor()
    head = HeadPart(door, run=inline, fade_s=0.35)
    assert head.ready() is None and head.latency_s == 0.1
    assert head.play(nod()).ok
    assert head.play(RENDERER.render_atom("head_drift", 3)).ok
    gesture, idle = door.looks()
    assert gesture["source"] == GESTURE and "idle" not in gesture["target"]
    assert idle["source"] == IDLE_LOOK and idle["target"]["idle"] is True
    head.fade("gesture", 0.35)
    head.fade("idle", 0.2)
    head.rest()
    withdrawn = [args for name, args in door.calls if name == "withdraw"]
    assert withdrawn == [(GESTURE, 0.35), (IDLE_LOOK, 0.2), (GESTURE, 0.35), (IDLE_LOOK, 0.35)]
    refused = head.play(Clip("head", 0.0, 0.0, 1.0))
    assert not refused.ok and "no head" in refused.why


def test_a_refusal_is_logged_once_counted_and_never_retried(
    caplog: pytest.LogCaptureFixture,
) -> None:
    why = "the head is held by llm.look (person band), which an overlay never plays over"
    door = FakeHeadDoor(refusals=deque([why, why, None, why, "past clip_max_deg"]))
    heard: list[tuple[str, str | None]] = []
    head = HeadPart(door, run=inline, on_answer=lambda r, a: heard.append((r["source"], a)))
    with caplog.at_level(logging.WARNING, logger="pepin.expression.parts"):
        for _ in range(5):
            assert head.play(nod()).ok  # the door answers later: the engine is never held up
    assert len(door.looks()) == 5  # one post per clip: nothing retried
    assert head.refused == 4
    assert [a for _, a in heard] == [why, why, None, why, "past clip_max_deg"]
    logged = [r.getMessage() for r in caplog.records]
    assert len(logged) == 3  # the same refusal again in a row is not logged again
    assert all("refused" in line for line in logged)


def test_the_serial_runner_keeps_the_order_of_its_calls() -> None:
    order: list[int] = []
    done = threading.Event()
    run = Serial("test-serial")
    for i in range(20):
        run(lambda i=i: order.append(i))
    run(done.set)
    assert done.wait(2.0) and order == list(range(20))


# -- the door over the arbiter's JSON -----------------------------------------------------------


class Client:
    """One source's overlay calls, scripted: answers (or exceptions) in turn."""

    def __init__(self, source: str, answers: deque[Any]) -> None:
        self.source = source
        self.answers = answers
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def clip(
        self,
        keys: Sequence[Mapping[str, float]],
        *,
        ttl_s: float,
        band: int = IDLE,
        loop: bool = False,
        idle: bool = False,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "clip",
                {
                    "keys": list(keys),
                    "ttl_s": ttl_s,
                    "band": band,
                    "loop": loop,
                    "idle": idle,
                    "timeout_s": timeout_s,
                },
            )
        )
        answer = self.answers.popleft() if self.answers else {"status": "done"}
        if isinstance(answer, BaseException):
            raise answer
        return answer  # type: ignore[no-any-return]

    def release(self, fade_s: float = 0.0) -> int:
        self.calls.append(("release", {"fade_s": fade_s}))
        answer = self.answers.popleft() if self.answers else 1
        if isinstance(answer, BaseException):
            raise answer
        return 1


def test_the_overlay_door_sends_a_source_s_clip_and_reads_the_answer() -> None:
    answers: deque[Any] = deque(
        [
            {"status": "done"},
            {"status": "denied", "reason": "no head overlay during a drive: the wheels turn"},
            TimeoutError("still playing"),
            ConnectionRefusedError("no door"),
            {"status": "granted"},
        ]
    )
    clients: dict[str, Client] = {}

    def client(source: str) -> Client:
        clients[source] = Client(source, answers)
        return clients[source]

    door = OverlayDoor(client=client, answer_wait_s=0.2)
    look = clip_look(nod(), GESTURE)
    assert look is not None
    assert door.look(look) is None
    assert door.look(look) == "no head overlay during a drive: the wheels turn"
    assert door.look(look) is None  # no refusal within answer_wait_s: it plays
    away = door.look(look)
    assert away is not None and away.startswith(AWAY)
    breath = clip_look(RENDERER.render_atom("breath", 1), IDLE_LOOK)
    assert breath is not None and door.look(breath) is None  # a loop is answered at once
    name, sent = clients[GESTURE].calls[0]
    assert name == "clip" and sent["timeout_s"] == 0.2 and sent["band"] == IDLE
    assert sent["keys"] == look["target"]["keys"] and sent["ttl_s"] == look["ttl_s"]
    assert not sent["loop"] and not sent["idle"]
    _, looped = clients[IDLE_LOOK].calls[0]
    assert looped["loop"] and looped["idle"]
    answers.append(OSError("gone"))
    door.withdraw(GESTURE, 0.35)  # a door away: nothing raised
    door.withdraw(IDLE_LOOK, 0.1)
    assert clients[GESTURE].calls[-1] == ("release", {"fade_s": 0.35})
    assert clients[IDLE_LOOK].calls[-1] == ("release", {"fade_s": 0.1})
    assert set(clients) == {GESTURE, IDLE_LOOK} and door.calls == 7  # one client per source


# -- with the real arbiter: the gesture never takes the head from the person --------------------


class TargetHead:
    """A ``neck_target`` driver that records every write and its speed ceiling."""

    moves_while_driving = True

    def __init__(self) -> None:
        self.writes: list[tuple[Aim | None, float, float | None]] = []
        self.ramps: list[float | None] = []

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
    ) -> None:
        self.writes.append((aim, now, speed_deg_s))
        self.ramps.append(acc_deg_s2)

    def take_refusal(self) -> Refusal | None:
        return None

    def take_arrival(self) -> HeadReading | None:
        return None

    def keep(self, now: float) -> None:
        pass


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


class ArbiterClient:
    """One source's door calls straight into an in-process arbiter (as the node's routes do; a
    finite overlay's wait is cut at once: it plays)."""

    def __init__(self, source: str, arbiter: Arbiter, clock: Clock) -> None:
        self.source, self.arbiter, self.clock = source, arbiter, clock

    def clip(
        self,
        keys: Sequence[Mapping[str, float]],
        *,
        ttl_s: float,
        band: int = IDLE,
        loop: bool = False,
        idle: bool = False,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        body = {
            "source": self.source,
            "kind": "clip",
            "band": band,
            "ttl_s": ttl_s,
            "target": {"keys": [dict(k) for k in keys], "loop": loop, "idle": idle},
        }
        look = look_from_json(body, self.arbiter.settings, NECK, lambda frame, xyz: "no frames")
        if isinstance(look, str):
            return {"status": "denied", "reason": look}
        return self.arbiter.submit(look, self.clock()).as_dict()

    def renew(self) -> int:
        return self.arbiter.renew(self.source, self.clock())

    def release(self, fade_s: float = 0.0) -> int:
        return self.arbiter.withdraw(self.source, self.clock(), "its source's release", fade_s)


def test_a_looping_breath_plays_on_while_renewed_and_lapses_without_it() -> None:
    """The arbiter's TTL of a looped overlay is one period and a second: renewed once a period
    through the door it plays on; left alone it lapses."""
    for renewed in (True, False):
        clock = Clock()
        arbiter = Arbiter(TargetHead(), home_aim(NECK), GazeSettings(), None)
        door = OverlayDoor(client=lambda s, a=arbiter, c=clock: ArbiterClient(s, a, c))
        head = HeadPart(door, run=inline)
        breath = RENDERER.render_atom("breath", 5)
        clock.now = 1.0
        assert head.play(breath).ok
        due = clock.now + breath.duration_s
        while clock.now < 1.0 + 3 * breath.duration_s:
            clock.now += 0.1
            if renewed and clock.now >= due:
                head.renew("idle")
                due = clock.now + breath.duration_s
            arbiter.step(clock.now)
        playing = arbiter.overlay()
        assert (playing is not None and playing.source == IDLE_LOOK) == renewed


def _speaker_holds(arbiter: Arbiter, aim: Aim) -> list[Outcome]:
    answers: list[Outcome] = []
    arbiter.submit(Look(SPEAKER, (aim,), IDLE, 0, 6.0, 6.0), 0.0, answers.append)
    arbiter.step(0.0)
    for stamp in (0.3, 0.31):
        arbiter.observe(HeadReading(aim.pan_rad, aim.tilt_rad, stamp))
    arbiter.step(0.4)
    assert answers and answers[0].status == "done"
    return answers


def test_a_gesture_overlay_never_changes_the_holder_and_the_head_returns_to_the_person() -> None:
    """The speaker gaze holds the head on the person (an angles look of the idle band); an
    agreement nod plays on top of it: the holder stays the speaker throughout, every write is
    the person's aim plus the nod's offset, and the head ends back on the person."""
    clock = Clock()
    driver = TargetHead()
    person = Aim(math.radians(25.0), math.radians(10.0))
    arbiter = Arbiter(driver, home_aim(NECK), GazeSettings(), None)
    _speaker_holds(arbiter, person)
    door = OverlayDoor(client=lambda source: ArbiterClient(source, arbiter, clock))
    head = HeadPart(door, run=inline)
    clip = next(
        c for c in RENDERER.render("agreement", EVERYTHING, 4, variant=0).clips if c.part == "head"
    )
    clock.now = 1.0
    written = len(driver.writes)
    assert head.play(clip).ok and head.refused == 0
    overlay = arbiter.overlay()
    assert overlay is not None and overlay.source == GESTURE
    offsets = set()
    settled = len(driver.writes)
    while clock.now < 1.0 + clip.duration_s + 0.5:
        arbiter.step(clock.now)
        state = arbiter.state(clock.now)
        assert state.source == SPEAKER  # the person keeps the head
        if len(driver.writes) > settled:  # the head arrives: a key waits for the one before
            aim, _, _ = driver.writes[-1]
            at = aim if aim is not None else person
            for stamp in (clock.now + 0.005, clock.now + 0.01):
                arbiter.observe(HeadReading(at.pan_rad, at.tilt_rad, stamp))
            settled = len(driver.writes)
        clock.now += 0.02
    for aim, _, ceiling in driver.writes[written:]:
        assert aim is not None and ceiling is not None
        assert aim.pan_rad == pytest.approx(person.pan_rad)  # a nod: the tilt alone moves
        offsets.add(round(math.degrees(aim.tilt_rad - person.tilt_rad), 2))
    assert max(offsets) >= 4.0 and arbiter.overlay() is None
    last, _, _ = driver.writes[-1]
    assert last == person  # back on the person
    assert [look.source for look in arbiter.pending()] == [SPEAKER]


def test_a_released_gesture_eases_back_onto_the_person_over_the_fade() -> None:
    clock = Clock()
    driver = TargetHead()
    person = Aim(math.radians(-20.0), math.radians(5.0))
    arbiter = Arbiter(driver, home_aim(NECK), GazeSettings(), None)
    _speaker_holds(arbiter, person)
    head = HeadPart(OverlayDoor(client=lambda s: ArbiterClient(s, arbiter, clock)), run=inline)
    glance = RENDERER.atom_clip("head_glance", {"side_deg": 14.0, "up_deg": 0.0, "period_s": 2.0})
    clock.now = 1.0
    assert head.play(glance).ok
    arbiter.step(1.0)
    away, _, _ = driver.writes[-1]
    assert away is not None and math.degrees(away.pan_rad - person.pan_rad) == pytest.approx(14.0)
    for stamp in (1.1, 1.11):  # the glance's move settled: the way back may go
        arbiter.observe(HeadReading(away.pan_rad, away.tilt_rad, stamp))
    clock.now = 1.3
    head.fade("gesture", 0.35)  # the person starts speaking
    arbiter.step(1.3)
    back, _, ceiling = driver.writes[-1]
    assert back == person and ceiling == pytest.approx(14.0 / 0.35, rel=1e-3)
    assert arbiter.state(1.3).source == SPEAKER and arbiter.overlay() is None


# -- the engine's head --------------------------------------------------------------------------


def test_the_engine_s_head_goes_through_the_door_and_lets_go_on_speech_drive_and_close() -> None:
    clock = Clock()
    clock.now = 100.0
    playout = FakePlayout()
    door = FakeHeadDoor()
    head = HeadPart(door, run=inline, fade_s=0.1)
    config = EngineConfig(
        arm_speed_scale=1.0, seed=4, fade_s={"arm": 0.35, "head": 0.3, "face": 0.1, "body": 0.35}
    )
    engine = GestureEngine(
        {
            "head": head,
            "face": FakeExecutor("face"),
            "arm": Absent("arm", "-"),
            "body": Absent("body", "-"),
        },
        playout,
        lambda: (BOOK, config),
        on=True,
        clock=clock,
    )
    assert head.fade_s == 0.3  # the engine's knob for the head's ease back
    engine.part_queued(1, 0, Part("Да, конечно.", None, (Marker("gesture", "agreement", 0),)))
    playout.marks[(1, 0)] = (0.0, 1.0)
    engine.part_spoken(1, 0)
    while not door.looks() and playout.now < 1.0:
        clock.now += 0.01
        playout.now += 0.01
        engine.step()
    (look,) = door.looks()
    assert look["source"] == GESTURE and look["kind"] == "clip"
    assert max(math.degrees(k["tilt_rad"]) for k in look["target"]["keys"]) >= 4.0
    engine.speech_started(clock.now)  # while the nod plays
    engine.step()
    assert door.calls[-1] == ("withdraw", (GESTURE, 0.3))
    engine.drive(True)
    engine.step()
    assert door.calls[-2:] == [("withdraw", (GESTURE, 0.3)), ("withdraw", (IDLE_LOOK, 0.3))]


def test_a_request_from_the_adapter_is_one_the_arbiter_s_json_reader_takes() -> None:
    look: ClipLook | None = clip_look(nod(), GESTURE)
    assert look is not None
    parsed = look_from_json(dict(look), GazeSettings(), NECK, lambda frame, xyz: "no frames")
    assert isinstance(parsed, Look) and parsed.kind == "clip" and parsed.band == IDLE
    assert parsed.overlay is not None and len(parsed.overlay.keys) == 2
