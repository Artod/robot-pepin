"""The face: the mouth on the screen under the camera's two lenses, and what it can show.

The head server owns the screen and decides what shows: an expression from a tool is the
``llm`` source's and always timed, so the face goes back by itself to whatever stood before it
(a drive's focus, the voice loop's listening). An info screen covers the face for its seconds.
The expressions are config/face.json's; :data:`EMOTIONS` is held equal to them by a test.
"""

from __future__ import annotations

from typing import Literal

from pepin.tools.registry import Result, fail, ok, tool
from pepin.tools.robot import Robot

Emotion = Literal[
    "neutral",
    "smile",
    "happy",
    "grin",
    "clenched",
    "sad",
    "worried",
    "surprised",
    "thinking",
    "struggling",
    "flat",
    "sleepy",
    "focused",
    "listening",
]
EMOTIONS: tuple[str, ...] = Emotion.__args__  # type: ignore[attr-defined]
MAX_SECONDS = 60.0
SERVO_LIMIT_C = 70  # the STS servos' own cut-out (pepin.base_server's temperature note)


@tool
def express(robot: Robot, emotion: Emotion, seconds: float = 5.0) -> Result:
    """Show a facial expression on the robot's mouth (the screen under its camera eyes) for a
    few seconds; the face then goes back by itself. React with it the way a person's face
    would: happy when something worked, sad when it failed, surprised, worried, thinking while
    working something out, clenched teeth when something is awkward or hard.

    Args:
        emotion: neutral, smile, happy (an open smile), grin, clenched (a grin with clenched
            teeth), sad, worried, surprised (an O), thinking, struggling (a zigzag), flat (no
            comment), sleepy, focused, listening.
        seconds: how long to show it, 1 to 60.
    """
    if not 0.0 < seconds <= MAX_SECONDS:
        return fail(f"seconds must be between 1 and {MAX_SECONDS:.0f}, not {seconds:g}")
    answer = robot.face.express(emotion, max(seconds, 1.0))
    return ok(showing=answer.get("showing", emotion), seconds=max(seconds, 1.0))


@tool
def show(robot: Robot, text: str, seconds: float = 8.0) -> Result:
    """Show information on the robot's face screen (1.9 inch, 320x170, about 25 characters by
    6 lines) for a few seconds; then the face comes back. One item per line: the first line is
    the title; 'label: 63%' or 'label: 41/70 C' draws a bar with that text beside it;
    'key: value' a row; anything else a line of text. For example:
    'Servo temperatures\\nleft wheel: 41/70 C\\nright wheel: 39/70 C'.

    Args:
        text: the lines, separated by newlines (or ' | '); at most 8.
        seconds: how long to show it, 1 to 60.
    """
    lines = [line for line in text.replace("|", "\n").splitlines() if line.strip()]
    if not lines:
        return fail("nothing to show: the text is empty")
    if not 0.0 < seconds <= MAX_SECONDS:
        return fail(f"seconds must be between 1 and {MAX_SECONDS:.0f}, not {seconds:g}")
    answer = robot.face.show(text, max(seconds, 1.0))
    shown = int(answer.get("items", len(lines)))
    note = {} if len(lines) <= 8 else {"note": f"only the first 8 of {len(lines)} lines fit"}
    return ok(items=shown, seconds=max(seconds, 1.0), **note)


@tool
def servo_temperatures(robot: Robot) -> Result:
    """The temperature of each of the robot's servos in degrees C (the two wheels, and the
    neck's pan and tilt when they answer), as the base read them in the last 5 s; the servos
    cut out at 70 C. Put them on the face screen with show() when asked to show them."""
    temps = robot.body.temperatures()
    hottest = max(temps.values()) if temps else None
    return ok(temperatures_c=temps, limit_c=SERVO_LIMIT_C, hottest_c=hottest)
