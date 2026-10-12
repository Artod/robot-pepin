"""The fillers of a slow tool: what the robot says by itself while a tool runs before its reply's
first word (:class:`pepin.voice_cascade.config.FillerLines`).

A filler is chosen at random from the tool's own lines, else from the common ones, and is never
the line said just before (a tool with one line of its own then takes a common one). When the
wait goes on ``filler_again_s`` with nothing said, a second one comes from the ``again`` lines.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


class FillerPicker:
    """Picks the fillers at random (an unseeded generator: each run differs), never the line it
    picked last."""

    def __init__(self, rng: random.Random | None = None) -> None:
        """``rng``: seeded in a test; a fresh one by default."""
        self.rng = rng if rng is not None else random.Random()
        self.last: str | None = None

    def pick(self, lines: Sequence[str], spare: Sequence[str] = ()) -> str | None:
        """One of ``lines`` other than the last one picked; from ``spare`` when ``lines`` holds
        that one alone; the last one again when nothing else is there; None for no lines."""
        if not lines:
            return None
        choices = [line for line in lines if line != self.last]
        if not choices:
            choices = [line for line in spare if line != self.last] or list(lines)
        self.last = self.rng.choice(choices)
        return self.last


def again_due(said_s: float | None, now: float, again_s: float) -> bool:
    """The second filler's time: ``again_s`` (above 0) since the first was said (``said_s``,
    None while it has not been)."""
    return said_s is not None and again_s > 0 and now - said_s >= again_s
