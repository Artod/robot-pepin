"""The face: the mouth on the screen under the camera's two lenses, the eyes round them, and
what they can show.

The head server owns the screen and the eyes and decides what shows: an expression from a tool
is the ``llm`` source's and always timed, so the face goes back by itself to whatever stood
before it (a drive's focus, the voice loop's listening); so is an eyes override beside it, and
a drawing style (the brain's standing choice is its mark's: ``{style:NAME,hold}``). An info
screen covers the face for its seconds.
The expressions are config/face.json's; :data:`EMOTIONS` is held equal to them by a test. The
styles are read from it at each call (its ``styles``; the brain reads them in its marks).
"""

from __future__ import annotations

from typing import Any, Literal

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
    "sinister",
    "kiss",
    "kiss_heart",
    "pout",
    "thinking_flat",
    "moustache",
    "o_mouth",
    "vampire",
    "terminator",
    "blush",
    "tease",
    "tired",
    "cat",
    "cat_blush",
    "smile_wide",
    "happy_plain",
    "angry",
    "shocked",
]
EMOTIONS: tuple[str, ...] = Emotion.__args__  # type: ignore[attr-defined]
MAX_SECONDS = 60.0
SERVO_LIMIT_C = 70  # the STS servos' own cut-out (pepin.base_server's temperature note)


@tool
def express(
    robot: Robot,
    emotion: Emotion,
    seconds: float = 5.0,
    brows: float | None = None,
    lids: float | None = None,
    glow: float | None = None,
    wink: Literal["left", "right"] | None = None,
    effect: Literal["sparkle", "flash", "spin", "pulse", "flicker", "doze", "tear"] | None = None,
    style: str | None = None,
) -> Result:
    """Show a facial expression on the robot's mouth (the screen under its camera eyes) and
    eyes (light rings round the lenses) for a few seconds; the face then goes back by itself.
    React with it the way a person's face would: happy when something worked, sad when it
    failed, surprised, worried, thinking while working something out, clenched teeth when
    something is awkward or hard. The eyes follow the expression; brows, lids, glow and wink
    tune them for the same seconds, and style draws the mouth another way for them.

    Args:
        emotion: neutral, smile, happy (an open smile), grin, clenched (a grin with clenched
            teeth), sad, worried, surprised (an O), thinking, struggling (a zigzag), flat (no
            comment), sleepy, focused, listening; for fun and jokes: sinister (sharp teeth),
            kiss, kiss_heart (blows a heart), pout (sulking), tease (cheeky, the tongue
            straight out), tired (tongue hanging limp), cat (the anime cat's w mouth),
            cat_blush (the same, blushing), moustache, o_mouth (an oo), vampire,
            terminator, blush (embarrassed), smile_wide and happy_plain (the smile's and
            happy's variants), angry (gritted teeth), shocked (a small open mouth, the
            cheeks striped).
        seconds: how long to show it, 1 to 60.
        brows: the brows, -1 low (a frown) to 1 raised (surprise).
        lids: the upper lids, 0 wide open to 1 shut (0.5 sleepy or doubtful).
        glow: the eyes' glow, 0 dim to 1 bright.
        wink: wink the robot's own left or right eye once.
        effect: a light in the eyes: sparkle, flash (an aha), spin (thinking), pulse, flicker,
            doze, tear (a drop running down: sad).
        style: draw the mouth in another style for the same seconds, a gag: a name from your
            style:NAME marks.
    """
    if not 0.0 < seconds <= MAX_SECONDS:
        return fail(f"seconds must be between 1 and {MAX_SECONDS:.0f}, not {seconds:g}")
    tuned = {"brow_raise": brows, "lid_top": lids, "glow": glow}
    for name, value, lo in (("brows", brows, -1.0), ("lids", lids, 0.0), ("glow", glow, 0.0)):
        if value is not None and not lo <= value <= 1.0:
            return fail(f"{name} must be between {lo:g} and 1, not {value:g}")
    if style is not None:
        from pepin.face import load_face_table

        styles = [s.name for s in load_face_table().styles]
        if style not in styles:
            return fail(f"style must be one of {', '.join(styles)}, not {style!r}")
    seconds = max(seconds, 1.0)
    answer = robot.face.express(emotion, seconds)
    eyes = {k: v for k, v in tuned.items() if v is not None}
    extra: dict[str, Any] = {}
    if eyes:
        robot.face.eyes(eyes, seconds)
        extra["eyes"] = eyes
    if wink is not None:
        robot.face.wink(wink)
        extra["winked"] = wink
    if effect is not None:
        robot.face.effect(effect, seconds)
        extra["effect"] = effect
    if style is not None:
        robot.face.style(style, seconds)
        extra["style"] = style
    return ok(showing=answer.get("showing", emotion), seconds=seconds, **extra)


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
