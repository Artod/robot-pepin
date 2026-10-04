"""What the voice is doing, for whoever shows it (the face, a log): one small callback.

States: ``idle`` (no session; the wake gate listens), ``listening`` (a session is open and the
user may speak), ``thinking`` (the user finished, or a tool runs, and no answer plays yet),
``speaking`` (the answer plays; sent every ~40 ms with the amplitude of what the speaker is
playing at that moment, 0..1). A subscriber is any ``Callable[[VoiceEvent], None]``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

State = Literal["idle", "listening", "thinking", "speaking"]


@dataclass(frozen=True)
class VoiceEvent:
    """One state change, or one amplitude sample while speaking."""

    state: State
    level: float = 0.0  # 0..1, only while speaking
    t: float = 0.0  # laptop monotonic seconds


EventSink = Callable[[VoiceEvent], None]


class Events:
    """Fans events out to the subscribers; a state is sent once per change (``speaking`` every
    sample), and a subscriber that raises is logged, never allowed to stop the voice."""

    def __init__(self, *sinks: EventSink, clock: Callable[[], float] = time.monotonic) -> None:
        """Start with ``sinks`` subscribed."""
        self._sinks = list(sinks)
        self._clock = clock
        self.state: State = "idle"

    def subscribe(self, sink: EventSink) -> None:
        """Add a subscriber."""
        self._sinks.append(sink)

    def emit(self, state: State, level: float = 0.0) -> None:
        """Announce ``state`` (and the amplitude while speaking)."""
        if state == self.state and state != "speaking":
            return
        self.state = state
        event = VoiceEvent(state, round(min(1.0, max(0.0, level)), 3), self._clock())
        for sink in self._sinks:
            try:
                sink(event)
            except Exception:
                logger.exception("voice event subscriber failed")


class StatePrinter:
    """A console subscriber: state changes only, no amplitude samples."""

    def __init__(self) -> None:
        """Nothing printed yet."""
        self._last: State | None = None

    def __call__(self, event: VoiceEvent) -> None:
        """Print the state when it changed."""
        if event.state != self._last:
            self._last = event.state
            print(f"  [{event.state}]", flush=True)
