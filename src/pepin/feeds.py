"""What every sensor or link on the laptop shares: it runs by itself and is asked, never awaited.

A :class:`Feed` starts a background reader, keeps the newest reading, says how
old it is, and can be closed. The control loop composes feeds into one
:class:`pepin.navigator.Sense` per tick without ever blocking on the network;
a feed that has nothing to say shows up as a large ``age_s``, which the
navigator's hold rules turn into "stand still" or "carry on without it"
depending on the sensor. Adding a sensor means adding a Feed and one field to
``Sense``; switching one off is a flag in ``config/robot.json``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Feed(Protocol):
    """A background reader with a freshness clock."""

    connected: bool

    def start(self) -> Any:
        """Begin reading in a daemon thread; returns self so it chains."""
        ...

    def close(self) -> None:
        """Stop the reader and release its socket."""
        ...

    def age_s(self, now: float | None = None) -> float:
        """Seconds since the newest reading; infinite before the first."""
        ...
