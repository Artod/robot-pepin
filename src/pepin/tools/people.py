"""Meeting people: naming the new person the robot talks with, and forgetting someone.

The robot recognises people by face and voice (:mod:`pepin.people`, its switch ``people`` in
the voice cascade); the brain learns who speaks from the tag line in front of the words
(``[говорит: Саша]``, ``[говорит: кто-то новый]``). These two tools are how a name is given and
taken back; both answer that recognition is off where it does not run.
"""

from __future__ import annotations

from pepin.tools.registry import Result, fail, ok, tool
from pepin.tools.robot import Robot


@tool
def remember_person(robot: Robot, name: str, family: bool = False) -> Result:
    """Remember a new person by name, so the robot knows them by face and voice from now on.
    Call it when someone the tag line calls "кто-то новый" tells their name, or when someone
    introduces a person standing in front of the robot ("это Саша"). It binds the name to the
    new person who spoke last (or the one new face in view). If the name is taken by someone
    else, it says so: ask how to tell them apart and call again with that name.

    Args:
        name: the name as the person said it, in its usual form ("Саша", "тётя Маша").
        family: true only for a member of Artem's family; guests and friends are false.
    """
    door = robot.people
    if door is None:
        return fail("person recognition does not run in this process (the voice cascade has it)")
    outcome = door.remember(name, family)
    if not outcome.ok:
        return fail(outcome.why, **outcome.payload)
    return ok(**outcome.payload)


@tool
def forget_person(robot: Robot, name: str) -> Result:
    """Forget a person completely: their name and every face and voice sample are deleted
    from the robot. Call it only when that person, or Artem, asks the robot to forget them
    ("забудь меня", "забудь Сашу").

    Args:
        name: the person's name as the robot knows it.
    """
    door = robot.people
    if door is None:
        return fail("person recognition does not run in this process (the voice cascade has it)")
    outcome = door.forget(name)
    if not outcome.ok:
        return fail(outcome.why, **outcome.payload)
    return ok(**outcome.payload)
