"""Whom the head looks at: the person speaking, among the people in the register
(:mod:`pepin.expression.people_map`).

The gaze ENGAGES one person and serves them until someone else speaks. An utterance is first
judged as a stimulus (:func:`doubt`): a sound without words, one with fewer than
``doa_min_speech`` of the array's speech-flagged readings or a peak under ``doa_min_dbfs`` never
moves the head (2026-10-12: every bearing at -16 dBFS or louder was within 14 deg of the face; at
-20..-22 dBFS three of four missed by 28-76 deg); the robot's name moves it whatever its level.
Then its person decides
(:func:`decide`), as in Pepper's basic awareness:

* the person engaged speaks: stay (the head follows its voice only while nobody sees it);
* the robot's name: engage the speaker and seek them, never returning;
* a person seen before (a face or a body placed them): engage them at once, their point;
* a voice with no one seen there while someone seen is engaged: a CHECK (look, one frame, back
  when nobody is there: the SemiEngaged mode);
* a voice with nobody engaged (or nobody ever seen): engage it and seek.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pepin.expression.conversation import Heard
from pepin.expression.doa import VoiceBearing
from pepin.expression.people_map import Person

__all__ = ["Act", "Decision", "decide", "doubt"]

Act = Literal["stay", "switch", "check", "seek"]


@dataclass(frozen=True)
class Decision:
    """What an utterance does to the gaze, and why in words."""

    act: Act
    why: str


def doubt(heard: Heard, voice: VoiceBearing, *, min_speech: int, min_dbfs: float) -> str | None:
    """Why an utterance is no speaker to turn to (no words, too few speech readings, too
    quiet); None when it is one."""
    if not heard.words:
        return "no words"
    if voice.speech < min_speech:
        return f"{voice.speech} speech readings, under {min_speech}"
    if heard.peak_db is not None and heard.peak_db < min_dbfs:
        return f"quiet, peak {heard.peak_db:.1f} dBFS under {min_dbfs:.0f}"
    return None


def decide(engaged: Person | None, speaker: Person, *, named: bool) -> Decision:
    """What the speaker's utterance does, given the person engaged now."""
    if engaged is not None and speaker.id == engaged.id:
        return Decision("stay", "the person engaged")
    if named:
        return Decision("seek", "the robot's name")
    if speaker.seen_s is not None:
        return Decision("switch", "a person seen there")
    if engaged is not None and engaged.seen_s is not None:
        return Decision("check", "a voice elsewhere")
    return Decision("seek", "a voice")
