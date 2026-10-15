"""The body's gestures as a tool: an intent of config/animation.json performed now, by name.

The brain's main door to gestures is inline in its text (``{laugh}`` before the word,
:mod:`pepin.expression.markers`), which costs no round trip; this tool is for a person who
asks to see one ("помахай ручкой") and for debugging by voice. It goes through the same engine
(:class:`pepin.expression.animator.GestureEngine`) as the markers, so the same switch
(``gestures``) and the same rule about drives hold; a person's ask is never capped. The names
are the book's compositions, held equal by a test; the brain never picks joints.
"""

from __future__ import annotations

from typing import Literal

from pepin.tools.registry import Result, fail, ok, tool
from pepin.tools.robot import Robot

Gesture = Literal[
    "greeting",
    "farewell",
    "laugh",
    "doubt",
    "thinking",
    "emphasis",
    "surprise",
    "agreement",
    "disagreement",
    "shrug",
    "evil_click",
    "sympathy",
]
GESTURES: tuple[str, ...] = Gesture.__args__  # type: ignore[attr-defined]


@tool
def gesture(robot: Robot, name: Gesture) -> Result:
    """Make a gesture with the body now (the head, the mouth on the screen, the arm and its
    gripper), when a person asks to see one ("wave", "nod"). Most of the time the robot
    gestures without this tool; call it only when asked. To laugh, do not call this: write
    {laugh} in your reply before the ha-ha, so the body laughs WITH the sound (a laugh played
    here runs before the voice and reads as a twitch).

    Args:
        name: greeting (a wave hello), farewell (a wave goodbye), laugh (silent, only when
            asked to show the gesture), doubt (so-so), thinking (a look away), emphasis (a nod
            on a word), surprise, agreement (a nod), disagreement (a shake of the head), shrug,
            evil_click (the gripper clicks mischievously), sympathy.
    """
    animator = robot.animator
    if animator is None:
        return fail("the body's gestures do not run in this process (the voice cascade has them)")
    outcome = animator.perform(name, source="tool")
    if not outcome.ok:
        return fail(outcome.why)
    return ok(gesture=name, seconds=round(outcome.seconds, 1), how=outcome.why)
