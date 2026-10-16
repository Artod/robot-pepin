"""The gesture tool and the laptop's arm link: the tool's names are the book's, it goes through
the engine; the link reads the arm's state and the board's answers off the base server's lines."""

import threading
from pathlib import Path

from pepin.arm_driver import trajectory_from
from pepin.arm_motion import load_arm_motion
from pepin.expression.animator import EngineConfig, GestureEngine
from pepin.expression.arm_link import ArmClient
from pepin.expression.book import intents, load_book, with_rest
from pepin.expression.fakes import FakeArmLink, FakeExecutor, FakePlaces, FakePlayout
from pepin.expression.parts import ArmPart
from pepin.expression.trajectory import ArmStatus, Knot, Trajectory
from pepin.tools import TOOLS
from pepin.tools.fakes import FakeAnimator, fake_robot
from pepin.tools.gesture import GESTURES

CONFIG = Path(__file__).resolve().parents[2] / "config"
MOTION = load_arm_motion(CONFIG / "arm.json")
BOOK = with_rest(load_book(CONFIG / "animation.json"), MOTION.rest_deg)


def test_the_tool_s_names_are_the_book_s_intents() -> None:
    # the book's feelings, in its order; its co-speech primitives, the snap and the atoms'
    # auto-wrappers (single basic moves) are inline marks only
    assert tuple(n for n in BOOK.compositions if n in GESTURES) == GESTURES
    singles = {n for n, c in BOOK.compositions.items() if c.wraps is not None and c.callable}
    assert set(intents(BOOK)) - set(GESTURES) - singles == {
        "accent", "point", "arc", "size", "weigh", "count", "draw", "stop", "offer", "snap",
        "dance",
        "wink", "glow", "eyes", "frown", "brows_up", "squint",
        "style",
        "nod", "shake",
    }  # fmt: skip
    schema = TOOLS["gesture"].input_schema()
    assert schema["properties"]["name"]["enum"] == list(GESTURES)
    assert not TOOLS["gesture"].moves  # never a drive: the voice loop runs it as any tool


def test_the_tool_performs_through_the_animator_and_says_why_not() -> None:
    animator = FakeAnimator()
    result = TOOLS.call("gesture", {"name": "greeting"}, fake_robot(animator=animator))
    assert result["ok"] and result["gesture"] == "greeting" and result["seconds"] == 2.0
    assert animator.performed == [("greeting", "tool")]
    refused = TOOLS.call(
        "gesture", {"name": "laugh"}, fake_robot(animator=FakeAnimator(refusal="off"))
    )
    assert not refused["ok"] and refused["why"] == "off"
    absent = TOOLS.call("gesture", {"name": "laugh"}, fake_robot(animator=None))
    assert not absent["ok"] and "voice cascade" in absent["why"]
    assert not TOOLS.call("gesture", {"name": "dance"}, fake_robot())["ok"]


def test_the_tool_on_the_real_engine_sends_the_arm_a_trajectory() -> None:
    link = FakeArmLink(state=ArmStatus(0.0, MOTION.rest(), False, "off"))
    parts = {
        "arm": ArmPart(link, MOTION),
        "face": FakeExecutor("face"),
        "head": FakeExecutor("head"),
    }
    now = [0.0]
    engine = GestureEngine(
        parts,  # type: ignore[arg-type]
        FakePlayout(),
        lambda: (BOOK, EngineConfig(arm_speed_scale=1.0, seed=2)),
        on=True,
        clock=lambda: now[0],
    )
    result = TOOLS.call("gesture", {"name": "evil_click"}, fake_robot(animator=engine))
    assert result["ok"] and "seed 2" in result["how"]
    for _ in range(100):  # five seconds: every clip has gone
        now[0] += 0.05
        engine.step()
    assert [name for name, _ in link.calls].count("send") == 1


def test_a_person_s_ask_takes_a_mark_with_its_argument_and_the_book_s_spellings() -> None:
    """``perform("point:printer")`` (the tool's and the command line's door) points at the
    room's place; ``count:second`` is ``count:2``; ``!`` the accent; a bare ``point`` its first
    argument (you); a name with no such argument is refused with the book's names."""
    rows: list[tuple[str, dict[str, object]]] = []
    engine = GestureEngine(
        {"arm": FakeExecutor("arm"), "face": FakeExecutor("face"), "head": FakeExecutor("head")},
        FakePlayout(), lambda: (BOOK, EngineConfig(arm_speed_scale=1.0, seed=2)), on=True,
        clock=lambda: 10.0, record=lambda kind, fields: rows.append((kind, fields)),
    )  # fmt: skip
    engine.resolver.room = FakePlaces({"printer": 10.0})
    for ask in ("point:printer", "count:second", "!", "point"):
        assert engine.perform(ask, source="tool").ok, ask
    shown = [(f["intent"], f["arg"]) for k, f in rows if k == "performance"]
    assert shown == [("point", "printer"), ("count", "2"), ("accent", None), ("point", None)]
    refused = engine.perform("size:enormous", source="tool")
    assert not refused.ok and "point:printer" in refused.why


def test_the_link_reads_the_arm_s_state_and_the_board_s_answers() -> None:
    heard = []
    client = ArmClient("board", on_answer=lambda answer, t: heard.append(answer))
    state = {"type": "state", "t": 12.5, "arm": {"deg": [1, 2, 3, 4, 5, 6], "torque": True,
                                                 "phase": "moving", "id": "g-1",
                                                 "progress_s": 0.4, "error": ""}}  # fmt: skip
    client._ingest(state)
    status = client.status()
    assert status is not None and status.deg == (1, 2, 3, 4, 5, 6) and status.phase == "moving"
    assert status.stamp == 12.5 and status.trajectory == "g-1" and status.torque
    client._ingest({"type": "arm", "id": "g-1", "status": "done", "reason": ""})
    assert client.answer("g-1") is not None and heard[0].status == "done"
    assert client.wait_answer("g-1", ("done",), 0.1) == heard[0]
    threading.Timer(0.05, lambda: client._ingest({"type": "arm_read", "deg": [0] * 6})).start()
    reply = client.ask("arm_read", timeout_s=2.0)  # its line is dropped: no connection
    assert reply == {"type": "arm_read", "deg": [0] * 6}
    assert client.ask("arm_rest", timeout_s=0.05) is None


def test_a_trajectory_survives_the_wire_both_ways() -> None:
    knots = (Knot(0.0, MOTION.rest()), Knot(0.5, MOTION.rest()))
    trajectory = Trajectory("g-7", "voice.gesture", knots, "hold", 1.5, "wave", 0.2)
    sent = []
    client = ArmClient("board")
    client._lines.send = sent.append  # type: ignore[method-assign]
    client.send(trajectory)
    client.rest()
    client.halt()
    assert trajectory_from(sent[0]) == trajectory
    assert sent[1:] == [{"cmd": "arm_rest"}, {"cmd": "arm_halt"}]
