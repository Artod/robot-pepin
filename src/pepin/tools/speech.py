"""The voice: text spoken through the robot's own speaker (``pepin.audio_server``)."""

from __future__ import annotations

from pepin.tools.registry import Result, fail, ok, tool
from pepin.tools.robot import Robot

MAX_CHARS = 600  # a minute of speech: anything longer is a lecture, not a reply


@tool
def say(robot: Robot, text: str) -> Result:
    """Say something out loud through the robot's speaker, in the language of the text, and
    return once it has been spoken. Keep it to a sentence or two.

    Args:
        text: the words to speak.
    """
    words = text.strip()
    if not words:
        return fail("nothing to say: the text is empty")
    if len(words) > MAX_CHARS:
        return fail(f"too long to say ({len(words)} characters, at most {MAX_CHARS}): shorten it")
    return ok(spoken_s=round(robot.speech.say(words), 1))
