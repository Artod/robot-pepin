"""Dozing: outside a conversation the robot's standing face is sleepy, its eyes dozing.

Artem (2026-10-15): sleepy is the default when no session is active; the robot dozes, a z now
and then, so one sees it listens or dozes and waits to be called. :class:`Doze` is a
conversation observer of the voice cascade (:class:`pepin.expression.conversation.
ConversationObserver`) and its live switch ``doze`` (:class:`pepin.expression.conversation.
LiveFlag`; config/expression.json ``flags.doze``, on). Once no conversation is open and nothing
has happened for ``doze.idle_after_s`` (config/expression.json), it asks the head server for
``doze.expression`` (sleepy: the eyes' doze effect and the mouth's z run in the firmware) as the
standing request of its own source ``doze``; the head server's ``resting_expression`` shows
whenever no source holds anything. Any wake lets it go at once (the source cleared: the resting
face) and starts the quiet time again: the conversation opening on the wake word, any moment of
a conversation (the person speaking, a reply, a tool's filler, a person found), a drive of the
conversation, the flag switched off, and another source showing something on the face (the
goal server's drive, a tool's expression, the command line; the head server's status line,
once a second, names who holds a request). Its thread steps once a second and falls asleep
when due (:meth:`Doze.step`).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pepin.expression.conversation import Heard
from pepin.expression.markers import Part

logger = logging.getLogger(__name__)

DOZE = "doze"  # the live switch's name (config/expression.json flags) and the head server's source
POLL_S = 1.0  # the thread's step: the head server's status comes once a second
FALL_ASLEEP_MS = 900.0  # the face's slow way into the sleepy look (the head server's brain_lost's)


class DozeFace(Protocol):
    """The head server's door as :class:`pepin.head_link.HeadClient` has it: one source's
    expressions and the newest status line."""

    @property
    def last_status(self) -> dict[str, Any] | None:
        """The head server's newest status line (None: none yet)."""
        ...

    def express(
        self,
        name: str,
        *,
        intensity: float = 1.0,
        hold_s: float | None = None,
        transition_ms: float | None = None,
    ) -> None:
        """Show expression ``name`` (held until replaced when ``hold_s`` is None)."""
        ...

    def clear(self) -> None:
        """This source shows nothing any more."""
        ...


@dataclass(frozen=True)
class DozeConfig:
    """config/expression.json's ``doze`` block and its flag."""

    on: bool = True
    idle_after_s: float = 30.0
    expression: str = "sleepy"


def doze_config(data: Mapping[str, Any]) -> DozeConfig:
    """config/expression.json's object as the doze's knobs (defaults for what it lacks)."""
    block, d = data.get("doze", {}), DozeConfig()
    return DozeConfig(
        on=bool(data.get("flags", {}).get(DOZE, d.on)),
        idle_after_s=float(block.get("idle_after_s", d.idle_after_s)),
        expression=str(block.get("expression", d.expression)),
    )


class Doze:
    """The face asleep while nobody talks to the robot (the module's notes); a conversation
    observer and the ``doze`` switch."""

    def __init__(
        self,
        face: DozeFace,
        load: Callable[[], DozeConfig] = DozeConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``face``: the head server's door as source ``doze``; ``load`` reads the knobs and
        the flag's default, now and at every switch on."""
        self.face = face
        self.load = load
        self.clock = clock
        self.config = load()
        self._on = self.config.on
        self._lock = threading.Lock()
        self._conversation = False
        self._driving = False
        self._quiet_since = clock()
        self.dozing = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- the switch ------------------------------------------------------------------------------

    @property
    def flag(self) -> str:
        """``doze``."""
        return DOZE

    @property
    def on(self) -> bool:
        """Whether the face dozes outside a conversation."""
        return self._on

    def switch(self, on: bool) -> str | None:
        """On: the knobs read again, the quiet time from now; off: awake at once."""
        with self._lock:
            if on:
                self.config = self.load()
                self._quiet_since = self.clock()
            else:
                self._wake("the doze is switched off")
            self._on = on
        return None

    # -- the conversation ------------------------------------------------------------------------

    def opened(self, wake: Heard) -> None:
        """The wake word: awake, and awake until the conversation closes."""
        with self._lock:
            self._conversation = True
            self._wake("a conversation opened")

    def closed(self, reason: str) -> None:
        """The quiet time starts."""
        with self._lock:
            self._conversation = False
            self._quiet_since = self.clock()

    def drive(self, active: bool, ran: bool = True) -> None:
        """A drive of the conversation: awake while it runs (run or not)."""
        with self._lock:
            self._driving = active
            self._wake("a drive")

    def speech_started(self, t: float) -> None:
        """A moment of the conversation (awake)."""
        self._stir("the person speaks")

    def speech_ended(self, heard: Heard) -> None:
        """A moment of the conversation (awake)."""
        self._stir("the person spoke")

    def part_queued(self, turn: int, index: int, part: Part) -> None:
        """A moment of the conversation (awake)."""
        self._stir("a reply")

    def voice_started(self, turn: int) -> None:
        """A moment of the conversation (awake)."""
        self._stir("a reply")

    def part_started(self, turn: int, index: int, stream_s: float) -> None:
        """A moment of the conversation (awake)."""
        self._stir("a reply")

    def text_ahead(self, turn: int, sentence: int, text: str) -> None:
        """A moment of the conversation (awake)."""
        self._stir("a reply")

    def part_spoken(self, turn: int, index: int) -> None:
        """A moment of the conversation (awake)."""
        self._stir("a reply")

    def reply_cut(self, turn: int) -> None:
        """A moment of the conversation (awake)."""
        self._stir("a reply cut")

    def reply_done(self, turn: int) -> None:
        """A moment of the conversation (awake)."""
        self._stir("a reply done")

    def moment(self, name: str, text: str = "") -> None:
        """A moment of the robot (a slow tool, a person found: awake)."""
        self._stir(name)

    # -- the clock -------------------------------------------------------------------------------

    def step(self) -> None:
        """Awake while another source shows something on the face; asleep once the quiet time
        is over with no conversation, no drive and the switch on."""
        with self._lock:
            others = self._others()
            if others:
                self._wake(f"the face is {', '.join(others)}'s")
                return
            if not self._on or self._conversation or self._driving or self.dozing:
                return
            quiet = self.clock() - self._quiet_since
            if quiet < self.config.idle_after_s:
                return
            self.face.express(self.config.expression, transition_ms=FALL_ASLEEP_MS)
            self.dozing = True
            logger.info("doze: %s after %.0f s without a conversation", self.config.expression,
                        quiet)  # fmt: skip

    def start(self) -> Doze:
        """Step on a daemon thread of its own, once a second."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="doze", daemon=True)
            self._thread.start()
        return self

    def close(self) -> None:
        """The thread stopped and the face let go (nobody listens any more)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            self._wake("the voice loop ends")

    def _run(self) -> None:
        while not self._stop.wait(POLL_S):
            try:
                self.step()
            except Exception:
                logger.exception("doze: a step broke")

    def _stir(self, why: str) -> None:
        with self._lock:
            self._wake(why)

    def _wake(self, why: str) -> None:
        """Awake (the lock held): the quiet time from now, the face let go when it dozed."""
        self._quiet_since = self.clock()
        if self.dozing:
            self.face.clear()
            self.dozing = False
            logger.info("doze: awake: %s", why)

    def _others(self) -> list[str]:
        """The other sources that hold a request on the face, by the newest status line."""
        status = self.face.last_status or {}
        return [str(s) for s in status.get("sources", ()) if s != DOZE]
