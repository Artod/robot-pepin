"""Robot moments as face events (pepin.face_events): what the goal server and the voice say."""

from __future__ import annotations

from pathlib import Path

from pepin.face import load_face_table
from pepin.face_events import DRIVE_END, LEASE_S, DriveFace, VoiceFace

REPO = Path(__file__).resolve().parents[2]


class Sink:
    def __init__(self) -> None:
        self.said: list[tuple[str, ...]] = []

    def event(self, name: str, *, end: bool = False) -> None:
        self.said.append(("event", name, "end") if end else ("event", name))

    def clear(self) -> None:
        self.said.append(("clear",))

    def lease(self, seconds: float) -> None:
        self.said.append(("lease", str(seconds)))

    def close(self) -> None:
        self.said.append(("close",))


def test_every_moment_named_here_is_in_the_face_table() -> None:
    """The producers name moments; config/face.json says what they look like. A moment the
    table lacks would be refused by the head server at the worst time."""
    events = set(load_face_table(REPO / "config" / "face.json").events)
    named = {"goal_accepted", "recovery", "listening", "thinking", "speaking", "brain_lost"}
    assert named | set(DRIVE_END.values()) <= events


def test_a_drive_that_ends_without_a_known_status_just_clears() -> None:
    sink = Sink()
    face = DriveFace(sink)
    face.accepted()
    face.progress(2)  # the first report already counts two recoveries: one struggle
    face.progress(1)  # a count that went back is not a new recovery
    face.progress(2)
    face.done(0)
    face.abandoned()  # after an end: nothing more
    assert sink.said == [("event", "goal_accepted"), ("event", "recovery"), ("clear",)]
    face.lease()
    assert sink.said[-1] == ("lease", str(LEASE_S))


def test_a_voice_turn() -> None:
    sink = Sink()
    voice = VoiceFace(sink)
    voice.listening()
    voice.thinking()
    voice.speaking()
    voice.done()
    assert sink.said == [
        ("event", "listening"),
        ("event", "thinking"),
        ("event", "speaking"),
        ("clear",),
    ]
