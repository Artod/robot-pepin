"""The speaker gaze's body turn as Nav2's Spin (``speaker.body_turn`` spin, as shipped since
2026-10-14): one request of the whole turn to the goal server's ``spin``, the head kept on the
person by the frames meanwhile (the odometry's glide only while they show nobody), the server's
end (or refusal) ending it by the wheels' rule. The engine on a fake clock against the fakes of
test_speaker_gaze; the goal server is a fake that the test speaks for. The wheels' executor's own
turn keeps its tests in test_speaker_gaze; the turns on slipping wheels, test_speaker_gaze_pursuit.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
from test_speaker_gaze import BOARD, Arm, Rig, array, rad, rig, speech

from pepin.expression.conversation import Heard
from pepin.expression.fakes import FakeWheels
from pepin.expression.speaker import FaceFrame
from pepin.expression.tracker import BY_FRAMES, SETTLED, GoalSpins, SpeakerGazeWorker, SpinWord


@dataclass
class ListeningWheels(FakeWheels):
    """The wheels' executor as a spin finds it: :meth:`listen` opens its base link, whose
    first state line gives the odometry's heading."""

    listened: int = 0

    def listen(self) -> None:
        """Counted; the link's first line (the heading as it stands) when it was not open."""
        self.listened += 1
        if self.log.latest() is None:
            self.turn_to(self.yaw_deg)


@dataclass
class FakeSpins:
    """The goal server's spin as the gaze asks it: each request recorded; the test speaks for
    the server (:meth:`tell`) on the newest request."""

    asked: list[float] = field(default_factory=list)
    words: list[SpinWord] = field(default_factory=list)

    def spin(self, deg: float, said: SpinWord) -> None:
        self.asked.append(deg)
        self.words.append(said)

    def tell(self, event: str, **fields: Any) -> None:
        self.words[-1]({"event": event, **fields})


def spin_rig(*, listen: bool = True, **knobs: Any) -> tuple[Rig, FakeSpins]:
    """The engine with the body turned by Nav2's Spin, the wheels' link able to open."""
    r = rig(wheels=ListeningWheels() if listen else FakeWheels(), body_turn="spin", **knobs)
    spins = FakeSpins()
    r.machine.spins = spins
    return r, spins


def spin_rows(r: Rig) -> list[dict[str, Any]]:
    """The ``gaze_spin`` rows logged so far."""
    return [f for k, _, f in r.events if k == "gaze_spin"]


def test_past_the_arc_one_spin_of_the_whole_bearing_and_the_frames_keep_the_head_on_him() -> None:
    """Artem calls from 120 deg left, past the forward arc (+-30): ONE request to the goal
    server's spin of +120 (the whole bearing, within max_turn_deg), the wheels' link opened
    first for the odometry's heading, the wheels' own executor never asked, the head's aim out
    with it. Nav2's acceptance logs the start (run, asked); every record is taken while the
    body turns, and as the room comes round at 30 deg/s the head is aimed at him BY THE FRAMES
    (sent without waiting, the odometry's rate and Nav2's count beside each). Nav2's done ends
    it with the turn measured on the map; the cart went 3 deg further than its odometry says,
    and the first frame after the end re-centres him where the camera sees him; no second
    spin."""
    r, spins = spin_rig()
    artem = r.person(120.0)
    read = r.wake(120.0)
    assert spins.asked == [pytest.approx(120.0)] and r.wheels.listened == 1
    assert r.wheels.calls == [], "the wheels' own executor is never asked"
    assert r.aimed()[0] == (120.0, 3.0) and r.reaims()[0]["status"] == "sent"
    assert r.machine.attention(read).phase == "turning"
    spins.tell("accepted", run=41, deg=120.0, sent_in_ms=35)
    r.machine.tick(read)
    track = r.machine.engaged
    assert spin_rows(r) == [{"own": True, "run": 41, "asked_deg": 120.0, "track": track,
                             "glides": True, "waited_s": 0.0, "sent_in_ms": 35}]  # fmt: skip
    assert "the body turns +120 deg: Nav2's Spin, run 41" in r.said
    assert r.machine.phase == "track", "the record taken as the body turns found him"
    looks = r.scene.looks
    t = read
    for k in range(1, 4):  # Nav2's Spin at 30 deg/s, its heading on the board's clock
        t = read + 0.1 * k
        r.wheels.turn_to(3.0 * k, BOARD + t)
        artem.pan_deg = 120.0 - 3.0 * k  # the room turns the other way in the cart's frame
        assert r.machine.tick(t) == pytest.approx(t + 0.1), "a record and the heading every 0.1 s"
    assert [p for p, _ in r.aimed()] == [120.0, 117.0, 114.0, 111.0], "on him by the frames"
    frames = r.reaims()[1:]
    assert {f["speed"] for f in frames} == {"glide"} and {f["status"] for f in frames} == {"sent"}
    assert {f["why"] for f in frames} == {BY_FRAMES}
    assert all(f["rate_deg_s"] == pytest.approx(30.0, abs=0.5) for f in frames), "the odometry's"
    assert r.scene.looks == looks + 3, "every record while the body turns"
    assert r.machine.voice == pytest.approx(rad(111.0)), "the person turned with the cart"
    spins.tell("feedback", t=1.0, spun_deg=12.0)
    for k in range(4, 40):  # on round, a record each tenth
        t = read + 0.1 * k
        r.wheels.turn_to(3.0 * k, BOARD + t)
        artem.pan_deg = 120.0 - 3.0 * k
        r.machine.tick(t)
        if k == 4:
            assert r.reaims()[-1]["turned_deg"] == 12.0, "Nav2's count beside each aim"
    assert r.aimed()[-1] == (3.0, 3.0) and r.machine.counts["glides"] == 0, "no odometry's"
    r.wheels.turn_to(118.6, BOARD + read + 4.0)
    artem.pan_deg = -3.0  # the cart went 121.6 by the room's, 118.6 by its odometry
    spins.tell("done", run=41, status=4, seconds=4.1, turned_deg=118.6, short_deg=1.4,
               spun_deg=119.8, arrival={"x": 0.0, "y": 0.0, "yaw_deg": 118.6})  # fmt: skip
    r.machine.tick(read + 4.0)
    assert spin_rows(r)[-1] == {
        "own": True, "end": True, "run": 41, "asked_deg": 120.0, "turned_deg": 118.6,
        "short_deg": 1.4, "spun_deg": 119.8, "done": True, "why": "", "status": 4,
        "seconds": 4.1,
    }  # fmt: skip
    assert "the spin turned +119 of +120 deg" in r.said and r.machine.turning is None
    assert r.reaims()[-1]["why"] == SETTLED and r.aimed()[-1] == (-3.0, 3.0), "the frame after"
    assert r.machine.voice == pytest.approx(rad(-3.0), abs=rad(0.1)), "where the camera sees him"
    r.machine.tick(read + 4.2)
    assert len(spins.asked) == 1 and r.wheels.calls == [], "within the arc: no second turn"


def test_its_own_spin_is_no_drive_to_pause_for_and_a_person_on_the_right_is_turned_right() -> None:
    """The arbiter's drive flag pauses the gaze (a Nav2 drive owns the cart), but not during the
    gaze's own spin, marked before its request: Artem on the right (-100) is one spin of -100,
    and, his face turned away (no frame shows him), a voice that moves during it re-aims the
    head while the spin goes on. With the wheels' executor turning the body the same flag
    pauses as ever."""
    r, spins = spin_rig()
    r.person(-100.0, face_px=0.0)
    read = r.wake(-100.0)
    assert spins.asked == [pytest.approx(-100.0)], "a right turn: the sign of the bearing"
    spins.tell("accepted", run=7, deg=-100.0)
    r.machine.tick(read)
    r.door.view = replace(r.door.view, driving=True)
    r.say(-115.0, read + 1.0, read + 2.0, "я тут")
    assert r.machine.phase != "paused" and r.reaims()[-1]["why"] == "the voice moved"
    assert r.aimed()[-1][0] == pytest.approx(-115.0, abs=1.0) and len(spins.asked) == 1
    assert r.machine.turning is not None, "the spin goes on"
    wheels = rig(wheels=FakeWheels())  # body_turn wheels
    wheels.person(-100.0)
    read = wheels.wake(-100.0)
    wheels.door.view = replace(wheels.door.view, driving=True)
    wheels.say(-115.0, read + 1.0, read + 2.0, "я тут")
    assert wheels.machine.phase == "paused", "a drive's flag, as ever"


def test_a_refused_spin_says_why_leaves_the_head_on_the_person_and_waits_for_his_words() -> None:
    """The goal server refuses (a goal of the turn tool's runs, spin_requests off, Nav2's own
    check): the end row says why, nothing turned; the head stays on him as far as it reaches,
    and no spin is asked again before his next utterance, which asks again."""
    r, spins = spin_rig()
    r.person(120.0)
    read = r.wake(120.0)
    runs = "a drive runs (run 12: go home (tool)): the gaze's turn waits"
    spins.tell("error", detail=runs)
    r.machine.tick(read)
    assert spin_rows(r)[-1] == {
        "own": True, "end": True, "run": None, "asked_deg": 120.0, "turned_deg": 0.0,
        "short_deg": None, "spun_deg": None, "done": False,
        "why": runs, "status": None, "seconds": None,
    }  # fmt: skip
    line = f"no spin ({runs}): the head looks as far as it reaches"
    assert line in r.said
    assert r.machine.turning is None and r.machine.body_waits and not r.machine.gave_up
    assert r.aimed()[0] == (120.0, 3.0), "the head on him"
    now = r.run(read + 2.0, read)
    assert len(spins.asked) == 1 and r.machine.phase == "track", "not again before his words"
    r.say(120.0, now + 0.5, now + 1.5, "ну повернись")
    r.machine.tick(now + 1.6)
    assert spins.asked == [pytest.approx(120.0), pytest.approx(120.0)], "his words: once more"


def test_a_spin_that_moved_and_fell_short_is_not_asked_again_for_him() -> None:
    """Nav2 aborts after 40 of 120 deg (its collision check beside furniture), or calls a spin
    20 deg short a success (the goal server's detail): the turn moved the cart and fell short,
    so the body is not turned toward him again in this conversation (no creeping retries),
    whatever he says."""
    r, spins = spin_rig()
    r.person(120.0)
    read = r.wake(120.0)
    spins.tell("accepted", run=8, deg=120.0)
    r.machine.tick(read)
    r.wheels.turn_to(40.0, BOARD + read + 1.5)
    spins.tell("done", run=8, status=6, seconds=1.6, turned_deg=40.2, short_deg=79.8,
               spun_deg=41.0)  # fmt: skip
    r.machine.tick(read + 1.6)
    end = spin_rows(r)[-1]
    assert (end["done"], end["why"], end["turned_deg"], end["status"]) == (
        False, "Nav2 aborted", 40.2, 6
    )  # fmt: skip
    reach = "the head looks as far as it reaches"
    assert f"the spin turned +40 of +120 deg (Nav2 aborted): {reach}" in r.said
    assert r.machine.gave_up == {r.machine.engaged} and not r.machine.body_waits
    r.say(80.0, read + 3.0, read + 4.0, "эй, сюда")
    r.machine.tick(read + 4.1)
    assert len(spins.asked) == 1, "gave up for him"
    short, spins = spin_rig()
    short.person(120.0)
    read = short.wake(120.0)
    spins.tell("accepted", run=9, deg=120.0)
    detail = "turned +100 of +120 deg by the map's heading: 20 deg short"
    spins.tell("done", run=9, status=4, seconds=4.0, turned_deg=100.0, short_deg=20.0,
               detail=detail)  # fmt: skip
    short.machine.tick(read)
    end = spin_rows(short)[-1]
    assert end["done"] is False and end["why"] == f"Nav2 succeeded: {detail}"
    assert short.machine.gave_up == {short.machine.engaged}


def test_a_drive_s_start_never_stops_nav2_s_spin_and_its_end_is_still_taken() -> None:
    """The loop's drive pause (a tool's call) halts the wheels' own turn at once, but Nav2's
    spin is not the gaze's to stop (a cancel is every goal's, the red button's): it runs on (the
    tool's own request is the goal server's to refuse meanwhile), and its end is still logged
    and ends the turn. A spin still waiting for the arm to rest is never asked."""
    r, spins = spin_rig()
    r.person(120.0)
    read = r.wake(120.0)
    spins.tell("accepted", run=9, deg=120.0)
    r.machine.tick(read)
    r.machine.drive(True, read + 0.5)
    assert r.machine.phase == "paused" and r.machine.turning is not None
    assert r.wheels.calls == [], "nothing halted, nothing cancelled"
    spins.tell("done", run=9, status=4, seconds=4.0, turned_deg=119.0, short_deg=1.0)
    r.machine.tick(read + 4.0)
    assert spin_rows(r)[-1]["end"] and spin_rows(r)[-1]["done"]
    assert r.machine.turning is None
    waits, spins = spin_rig()
    waits.machine.arm = Arm(0.4)
    waits.person(120.0)
    read = waits.wake(120.0)
    assert spins.asked == [] and waits.machine.turning is not None, "the arm first"
    waits.machine.drive(True, read + 0.1)
    waits.machine.tick(read + 1.0)
    assert spins.asked == [] and waits.machine.turning is None


def test_no_goal_server_or_no_odometry_heading_refuses_the_spin() -> None:
    """Without a goal server there is no spin; without any odometry heading to come (no wheels'
    link to open, none there yet) the people's bearings could not follow the turn: refused,
    said why, the head alone on the person. A heading already there is enough."""
    alone, _ = spin_rig()
    alone.machine.spins = None
    alone.person(120.0)
    alone.machine.tick(alone.wake(120.0))
    assert spin_rows(alone)[-1]["why"] == "no goal server to ask for a spin here"
    assert alone.machine.body_waits
    blind, spins = spin_rig(listen=False)  # the wheels' fake: no link to open, no heading
    blind.person(120.0)
    blind.machine.tick(blind.wake(120.0))
    assert spins.asked == []
    why = "no odometry heading here to keep the people's bearings through a turn"
    assert spin_rows(blind)[-1]["why"] == why
    seen, spins = spin_rig(listen=False)
    seen.wheels.turn_to(0.0, BOARD)  # the link opened by an earlier motion
    seen.person(120.0)
    seen.wake(120.0)
    assert spins.asked == [pytest.approx(120.0)]


def test_goal_spins_sends_one_request_and_relays_the_server_s_events_to_their_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """:class:`GoalSpins` (no socket here: the goal link's events are a fake): one ``spin``
    request with the source ``gaze`` (the goal server's one rule: a person's request cancels
    it, and it never cancels theirs), every event relayed on its own thread until done or
    error; a server that cannot be reached, or one that closes without an end, is an error
    event all the same."""
    from pepin import goal_link

    asked: list[tuple[dict[str, Any], str, int]] = []
    script: list[dict[str, Any]] = []

    def events(
        request: dict[str, Any], host: str, port: int, *timeouts: float | None
    ) -> Iterator[dict[str, Any]]:
        asked.append((request, host, port))
        if not script:
            raise goal_link.GoalServerUnreachableError(f"{host}:{port}: refused")
        yield from script

    monkeypatch.setattr(goal_link, "events", events)

    def spin(deg: float) -> list[dict[str, Any]]:
        heard: list[dict[str, Any]] = []
        ended = threading.Event()

        def said(event: dict[str, Any]) -> None:
            heard.append(event)
            if event["event"] in ("done", "error"):
                ended.set()

        GoalSpins(port=1).spin(deg, said)  # port 1: nobody, should the fake ever be bypassed
        assert ended.wait(2.0)
        return heard

    defaults = GoalSpins()
    assert (defaults.host, defaults.port) == (goal_link.HOST, goal_link.PORT)
    script[:] = [{"event": "accepted", "run": 3}, {"event": "feedback", "spun_deg": 40.0},
                 {"event": "arrival", "status": 4}, {"event": "done", "status": 4},
                 {"event": "after the end"}]  # fmt: skip
    assert [e["event"] for e in spin(120.04)] == ["accepted", "feedback", "arrival", "done"]
    assert asked[-1] == ({"cmd": "spin", "deg": 120.0, "source": "gaze"}, goal_link.HOST, 1)
    script[:] = [{"event": "accepted", "run": 4}]
    closed = spin(-30.0)
    assert closed[-1] == {"event": "error",
                          "detail": "the goal server closed the spin without its end"}  # fmt: skip
    script.clear()
    (down,) = spin(30.0)
    assert down["event"] == "error" and "refused" in down["detail"]


def test_a_spin_the_goal_server_cancelled_for_a_person_s_request_gives_way_and_waits() -> None:
    """A tool's drive came while the gaze's spin turned (the loop's drive notice first, the
    paused gaze letting the spin run on): the goal server cancelled the spin (its one rule) and
    its end names the request; the end row says it gave way, and the body neither gives up on
    him nor waits for his words (the drive's end orients again). The arrival line before an end
    changes nothing; the gaze cancels nothing itself."""
    r, spins = spin_rig()
    r.person(120.0)
    read = r.wake(120.0)
    spins.tell("accepted", run=52, deg=120.0)
    r.machine.tick(read)
    r.machine.drive(True, read + 1.0)
    assert r.machine.phase == "paused" and r.machine.turning is not None
    r.wheels.turn_to(50.0, BOARD + read + 1.5)
    spins.tell("done", run=52, status=5, seconds=1.6, turned_deg=50.2, short_deg=69.8,
               spun_deg=51.0, cancelled_for="go home (tool)")  # fmt: skip
    r.machine.tick(read + 1.6)
    end = spin_rows(r)[-1]
    assert (end["done"], end["why"], end["status"]) == (False, "cancelled for go home (tool)", 5)
    assert (end["gave_way"], end["cancelled_for"]) == (True, "go home (tool)")
    assert "the spin turned +50 of +120 deg: cancelled for go home (tool)" in r.said
    assert not r.machine.gave_up and not r.machine.body_waits and r.machine.turning is None
    assert not hasattr(spins, "cancel") and len(spins.asked) == 1
    reached, spins = spin_rig()
    reached.person(120.0)
    read = reached.wake(120.0)
    spins.tell("accepted", run=53, deg=120.0)
    spins.tell("arrival", run=53, status=4, seconds=4.0, turned_deg=119.0)
    reached.machine.tick(read + 4.0)
    assert reached.machine.turning is not None, "the turn ends at done, its tape closed"


# ---- a tool's drive holds the body ------------------------------------------------------------
def hold_rows(r: Rig) -> list[dict[str, Any]]:
    """The ``gaze_body_hold`` rows logged so far."""
    return [f for k, _, f in r.events if k == "gaze_body_hold"]


def test_a_tool_s_turn_holds_the_gaze_s_body_turn_until_his_next_words() -> None:
    """Artem: «повернись на 180» (2026-10-14: the turn tool's 180, then the gaze's arc rule spun
    the body straight back to him; his rule: a turn on command waits for the first phrase). The
    loop's drive notice holds the gaze's own body turn (a row), the tool keeps him through its
    Nav2 turn, and when it ends he stands behind, past the arc: the head stays on him as far as
    it reaches, no spin is asked however long; words he began during the turn and a sound
    without words do not free it; his next words do (a row), and the body turns to him."""
    r, spins = spin_rig()
    artem = r.person(10.0)
    read = r.wake(10.0)
    r.machine.tick(read)
    assert r.machine.phase == "track" and spins.asked == []
    r.wheels.turn_to(0.0, BOARD + read)
    r.machine.drive(True, read + 0.1)
    assert r.machine.keep_person(read + 0.15, "turn: 180") is None
    assert hold_rows(r) == [{"held": True, "why": "a tool's drive"}]
    assert "the body's own turn waits for the person's next words (a tool's drive)" in r.said
    t = read + 0.15
    for k in range(1, 19):  # the tool's Nav2 Spin, 10 deg a tenth of a second
        t = read + 0.15 + 0.1 * k
        r.wheels.turn_to(10.0 * k, BOARD + t)
        artem.pan_deg = 10.0 - 10.0 * k  # the room turns the other way in the cart's frame
        r.machine.tick(t)
    r.machine.drive(False, t + 0.1)
    assert r.machine.voice == pytest.approx(rad(-170.0), abs=rad(1.0)), "behind, past the arc"
    now = r.run(t + 3.0, t + 0.1)
    assert spins.asked == [] and r.machine.phase == "track", "the head alone, three seconds"
    assert r.aimed()[-1][0] <= -160.0, "the head on him as far as it reaches"
    r.say(-170.0, t - 0.5, now, "вот так")  # begun while the cart still turned
    r.say(-170.0, now + 0.2, now + 0.6, "", noise=True)
    r.machine.tick(now + 0.8)
    assert spins.asked == [] and r.machine.body_hold is not None
    read = r.say(-170.0, now + 1.0, now + 2.0, "ну и что там")
    assert hold_rows(r)[-1]["held"] is False and hold_rows(r)[-1]["text"] == "ну и что там"
    assert any(line.startswith("the person spoke: the body may turn again") for line in r.said)
    r.machine.tick(read + 0.05)
    assert spins.asked == [pytest.approx(-170.0, abs=1.0)], "his words: the body turns to him"


def test_after_a_tool_s_drive_the_body_turns_at_his_words_not_at_the_reorientation() -> None:
    """approach or go_to: the drive's notice pauses the gaze and holds its body turn; after the
    drive's end an utterance begun during it orients the head (as ever) but leaves the body
    held; the next words, from behind, free it and the body turns to him. A conversation's
    close and the next wake begin with no hold."""
    r, spins = spin_rig()
    r.person(10.0)
    read = r.wake(10.0)
    r.machine.tick(read)
    r.machine.drive(True, read + 0.5)
    assert r.machine.phase == "paused" and r.machine.body_hold == "a tool's drive"
    r.machine.drive(False, read + 5.0)
    heard = r.say(150.0, read + 4.0, read + 5.5, "приехали")  # begun during the drive
    assert r.machine.phase == "seek" and r.aimed()[-1][0] == pytest.approx(150.0, abs=1.0)
    r.machine.tick(heard + 0.1)
    assert spins.asked == [] and r.machine.body_hold is not None, "the head turned, not the body"
    heard = r.say(150.0, heard + 1.0, heard + 2.0, "я здесь, сзади")
    r.machine.tick(heard + 0.1)
    assert spins.asked == [pytest.approx(150.0, abs=1.0)]
    assert [f["held"] for f in hold_rows(r)] == [True, False]
    r.machine.drive(True, heard + 0.5)
    r.machine.closed("bye", heard + 1.0)
    assert r.machine.body_hold is None
    again, spins = spin_rig()
    again.person(10.0)
    again.machine.drive(True, 0.0)  # no conversation: nothing held
    assert again.machine.body_hold is None and hold_rows(again) == []


def until(done: Callable[[], bool], within_s: float = 2.0) -> bool:
    """Whether ``done`` came true within ``within_s`` (the worker's thread runs meanwhile)."""
    deadline = time.monotonic() + within_s
    while not done():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def test_the_hold_counts_words_from_the_drive_s_end_as_told_not_as_the_gaze_got_to_it() -> None:
    """The turn tool's turn ends at 3.0 s (the loop's notice), but the gaze's thread is inside
    a look then (the finder waits up to 0.4 s for a record; here the test holds it until 6.0
    s). Artem's words begun at 4.0, after the turn, free the body: the hold counts from the
    drive's end as the loop told it, not from when the gaze got to the notice."""
    clock = [0.0]
    r, _ = spin_rig()
    r.person(10.0)
    entered, gate = threading.Event(), threading.Event()
    armed = [False]

    class Held:
        """The rig's finder, holding the look it is armed for until the gate opens."""

        def look(self, not_before: float | None = None) -> FaceFrame | None:
            if armed[0]:
                armed[0] = False
                entered.set()
                gate.wait(5.0)
            return r.scene.look(not_before=not_before)

    r.machine.finder = Held()
    worker = SpeakerGazeWorker(lambda s: r.machine, on=False, load=lambda: r.machine.settings,
                               clock=lambda: clock[0])  # fmt: skip
    assert worker.switch(True) is None
    r.doa.readings += speech(array(10.0), 0.0, 1.2)
    clock[0] = 1.3
    worker.opened(Heard(0.0, 1.2, "Пепин", named=True))
    assert until(lambda: r.machine.phase == "track")
    clock[0] = 2.0
    worker.drive(True)
    armed[0] = True
    clock[0] = 2.1
    assert not isinstance(worker.keep_person("turn: 90"), str)
    assert entered.wait(2.0), "the thread is inside a look"
    clock[0] = 3.0
    worker.drive(False)  # the turn's end, told at 3.0
    clock[0] = 6.0
    gate.set()
    assert until(lambda: r.machine.spin is None), "the end taken at 6.0"
    assert r.machine.hold_after_s == pytest.approx(3.0)
    clock[0] = 6.6
    worker.speech_ended(Heard(4.0, 6.5, "ну вот"))
    assert until(lambda: r.machine.body_hold is None), "his words after the turn: given back"
    assert hold_rows(r)[-1]["held"] is False
    worker.close()


def walks_past_the_arc(r: Rig, artem: Any, start: float) -> float:
    """Artem walks silently from +10 to +60 deg (past the arc's +30) at 25 deg/s from
    ``start``, then stands; the machine ticked to 3 s after. The last time ticked."""
    return r.run(start + 5.0, start, move=lambda t: setattr(
        artem, "pan_deg", 10.0 + 25.0 * min(max(t - start, 0.0), 2.0)))  # fmt: skip


def test_a_drive_that_never_ran_holds_nothing_and_gives_the_pause_back() -> None:
    """go_to(garage), refused at once (no such place): the loop's notice paused the gaze and
    held its body turn, and the drive's end comes with ``ran`` False (2026-10-14 review: a
    refusal held the body until his next words though nothing moved). The hold is lifted at
    once (a row: released, the drive never ran), the pause's people come back (the one
    engaged, tracked again at once, no reorientation waiting for words); walking past the arc
    without a word he gets the body turned to him."""
    r, spins = spin_rig()
    artem = r.person(10.0)
    read = r.wake(10.0)
    r.machine.tick(read)
    engaged = r.machine.engaged
    assert r.machine.phase == "track" and engaged is not None
    r.machine.drive(True, read + 0.2)
    assert r.machine.phase == "paused" and r.machine.body_hold == "a tool's drive"
    r.machine.drive(False, read + 0.4, ran=False)
    assert r.machine.body_hold is None and r.machine.hold_after_s is None
    assert (r.machine.phase, r.machine.engaged, r.machine.reorient) == ("track", engaged, False)
    assert hold_rows(r)[-1] == {"held": False, "why": "a tool's drive", "held_s": 0.2,
                                "ran": False}  # fmt: skip
    assert "released: the drive never ran (a tool's drive)" in r.said
    walks_past_the_arc(r, artem, read + 0.5)
    assert spins.asked and spins.asked[0] > 30.0, "past the arc, no word needed: the body turns"


def test_a_drive_that_ran_holds_until_his_next_words_and_a_refused_one_after_keeps_it() -> None:
    """approach ran (``ran`` True at its end): the body is held until his next words begun
    after it ended. A second call refused meanwhile (``ran`` False) gives back the hold as the
    first drive left it, its end the first drive's: his words then free the body, and the
    rows say held, then given back by his words (no release by the refusal)."""
    r, spins = spin_rig()
    r.person(10.0)
    read = r.wake(10.0)
    r.machine.tick(read)
    r.machine.drive(True, read + 0.5)
    r.machine.drive(False, read + 5.0, ran=True)
    assert r.machine.body_hold == "a tool's drive" and r.machine.hold_after_s == read + 5.0
    r.machine.drive(True, read + 5.2)
    assert r.machine.hold_after_s is None, "a drive runs"
    r.machine.drive(False, read + 5.3, ran=False)
    assert r.machine.body_hold == "a tool's drive" and r.machine.hold_after_s == read + 5.0
    heard = r.say(150.0, read + 6.0, read + 7.0, "я здесь, сзади")
    r.machine.tick(heard + 0.1)
    assert r.machine.body_hold is None and spins.asked == [pytest.approx(150.0, abs=1.0)]
    assert [f["held"] for f in hold_rows(r)] == [True, False]
    assert hold_rows(r)[-1]["text"] == "я здесь, сзади"


def test_the_turn_tool_already_facing_him_leaves_no_hold() -> None:
    """«Повернись ко мне» while he stands 5 deg off the nose: the loop's notice pauses the
    gaze, the tool's keep_person takes the pause back and holds the body (a turn in place
    kept through), then finds him already faced: nothing moved, and the drive's end says so
    (``ran`` False: no goal taken). The turn in place ends without a re-centre (a row: never
    ran), the hold is lifted at once, the head stays on him; walking past the arc without a
    word he gets the body turned to him."""
    r, spins = spin_rig()
    artem = r.person(5.0)
    read = r.wake(5.0)
    r.machine.tick(read)
    r.machine.drive(True, read + 0.1)
    assert r.machine.keep_person(read + 0.15, "turn me") is None
    assert r.machine.spin is not None and r.machine.body_hold == "a tool's drive"
    aims = len(r.aimed())
    r.machine.drive(False, read + 0.2, ran=False)
    assert r.machine.spin is None and r.machine.body_hold is None
    assert spin_rows(r)[-1] == {"end": True, "ran": False}
    assert "the turn in place never ran" in r.said and len(r.aimed()) == aims, "no re-centre"
    assert r.machine.phase == "track" and hold_rows(r)[-1]["ran"] is False
    walks_past_the_arc(r, artem, read + 0.3)
    assert spins.asked and spins.asked[0] > 30.0, "past the arc, no word needed: the body turns"


def test_the_worker_passes_whether_the_drive_ran() -> None:
    """The cascade's notice through the worker's queue: a drive's end that never ran lifts
    the hold on the gaze's thread as on the machine."""
    r, _ = spin_rig()
    r.person(10.0)
    clock = [0.0]
    worker = SpeakerGazeWorker(lambda s: r.machine, on=False, load=lambda: r.machine.settings,
                               clock=lambda: clock[0])  # fmt: skip
    assert worker.switch(True) is None
    r.doa.readings += speech(array(10.0), 0.0, 1.2)
    clock[0] = 1.3
    worker.opened(Heard(0.0, 1.2, "Пепин", named=True))
    assert until(lambda: r.machine.phase == "track")
    clock[0] = 2.0
    worker.drive(True)
    assert until(lambda: r.machine.body_hold is not None)
    clock[0] = 2.1
    worker.drive(False, False)
    assert until(lambda: r.machine.body_hold is None and r.machine.phase == "track")
    assert hold_rows(r)[-1]["ran"] is False
    worker.close()
