"""What the tools see of person recognition: naming the person in front of the robot and
forgetting one (:class:`pepin.people.worker.People` is the real door; a fake in
:mod:`pepin.people.fakes`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Outcome:
    """A door's answer: ok or why not (in words the brain can act on), and what was done."""

    ok: bool
    why: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


class PeopleDoor(Protocol):
    """Name and forget people."""

    def remember(self, name: str, family: bool = False) -> Outcome:
        """Bind ``name`` to the new person the robot is talking with."""
        ...

    def forget(self, name: str) -> Outcome:
        """Delete everything of the person called ``name``."""
        ...
