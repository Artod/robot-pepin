"""Robot moments as face events (pepin.face_events): what the goal server, the voice loops, the
gaze's stall look and the VIO keeper say."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pepin.face import load_face_table
from pepin.face_events import (
    DRIVE_END,
    LEASE_S,
    STALL_VERDICT,
    VIO_RESTART,
    VOICE_STATES,
    DriveFace,
    StallFace,
    VoiceFace,
    VoiceStateFace,
)

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
    named |= set(DRIVE_END.values()) | set(STALL_VERDICT.values()) | set(VOICE_STATES)
    assert named | {"stall_look", VIO_RESTART} <= events


def test_the_moments_that_repeat_have_a_gap_and_the_states_do_not() -> None:
    """A moment from a loop (a recovery, a stall look, a restart) cannot flood the head; a state
    (listening, thinking, speaking, a drive's focus) is never dropped."""
    table = load_face_table(REPO / "config" / "face.json")
    for moment in ("recovery", "stall_look", "phantom_carved", "obstacle_confirmed", VIO_RESTART):
        assert table.event(moment).min_gap_s > 0.0, moment
    for state in (*VOICE_STATES, "goal_accepted", "arrived", "brain_lost"):
        assert table.event(state).min_gap_s == 0.0, state


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


@dataclass
class State:
    state: str
    level: float = 0.0


def test_the_live_voice_states_follow_on_the_face_once_per_change() -> None:
    sink = Sink()
    face = VoiceStateFace(sink)
    for state in ["thinking", "speaking", "speaking", "speaking", "acting", "listening", "idle"]:
        face(State(state, 0.5))
    face.close()
    assert sink.said == [
        ("event", "thinking"),
        ("event", "speaking"),  # once, though every level sample says speaking
        ("clear",),  # acting: a drive the voice started, the goal server's face shows it
        ("event", "listening"),
        ("clear",),  # idle: the resting face
        ("clear",),
        ("close",),
    ]


def test_a_stall_look_shows_its_turn_and_its_verdict() -> None:
    sink = Sink()
    face = StallFace(sink)
    face.looking()
    face.verdict("carved")
    face.verdict("partly carved")
    face.verdict("confirmed")
    face.verdict("unknown")  # no moment for a look that read nothing
    face.verdict("empty")
    assert sink.said == [
        ("event", "stall_look"),
        ("event", "phantom_carved"),
        ("event", "phantom_carved"),
        ("event", "obstacle_confirmed"),
    ]
