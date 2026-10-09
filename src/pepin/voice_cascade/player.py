"""The speaker side: the voice's PCM paced to the speaker (:class:`pepin.voice_live.audio.Pacer`),
with a clock of what has actually played.

Every reply sentence is marked by where its audio sits in the stream fed to the pacer, and the
speaker is wrapped (:class:`PlayoutClock`) so the time each block plays is known. A barge-in
asks :meth:`Player.flush` how far the stream had played: that, against the marks, is which
sentences the person heard (:meth:`Player.heard`). The same clock times the speaker's start and
end of each turn.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Literal

from pepin.audio_link import SAMPLE_BYTES, Speaker
from pepin.voice_cascade.stages import Played

Kind = Literal["reply", "backchannel"]
Playback = Literal["silent", "reply", "backchannel"]
HISTORY_BLOCKS = 3000  # ~2 min of 40 ms blocks: enough to time any turn's end


class PlayoutClock:
    """A :class:`pepin.audio_link.Speaker` wrapper that keeps the play-out timeline: each block
    plays when the one before it ends, or now when the speaker is idle (the pacer's own model of
    the speaker). Thread-safe (the pacer plays, a barge-in flushes)."""

    def __init__(self, speaker: Speaker, rate: int, clock: Callable[[], float]) -> None:
        """``rate``: the speaker's play rate."""
        self.speaker, self.rate, self.clock = speaker, rate, clock
        self._lock = threading.Lock()
        self.sent_s = 0.0  # stream seconds sent and not flushed away
        self.cursor = 0.0  # when the last block sent ends
        self._blocks: deque[tuple[float, float, float]] = deque(maxlen=HISTORY_BLOCKS)

    def play(self, pcm: bytes) -> None:
        """Send a block; note where in the stream it is and when it plays."""
        duration = len(pcm) / SAMPLE_BYTES / self.rate
        with self._lock:
            start = max(self.cursor, self.clock())
            self._blocks.append((self.sent_s, start, duration))
            self.sent_s += duration
            self.cursor = start + duration
        self.speaker.play(pcm)

    def play_end(self) -> None:
        """The utterance is complete."""
        self.speaker.play_end()

    def flush(self) -> None:
        """The speaker drops what it has not played yet."""
        with self._lock:
            now = self.clock()
            self.sent_s -= max(0.0, self.cursor - now)
            self.cursor = min(self.cursor, now)
        self.speaker.flush()

    def played(self) -> float:
        """Stream seconds played by now."""
        with self._lock:
            return self.sent_s - max(0.0, self.cursor - self.clock())

    def when(self, offset: float, *, end: bool = False) -> float | None:
        """When the stream's second ``offset`` starts playing (``end``: finishes playing, for
        the last sample of a stretch); None when it was never sent or is too old to remember.
        A block boundary belongs to the block after it, or with ``end`` to the one before."""
        eps = 1e-6
        with self._lock:
            for start_offset, start, duration in reversed(self._blocks):
                low, high = start_offset, start_offset + duration
                inside = (
                    low + eps < offset <= high + eps if end else low - eps <= offset < high - eps
                )
                if inside:
                    return start + (offset - start_offset)
        return None


class Player:
    """The voice's PCM to the speaker at real time, with the marks and the clock above."""

    def __init__(
        self,
        speaker: Speaker,
        *,
        rate: int,
        lead_s: float = 0.3,
        block_s: float = 0.04,
        on_level: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``rate``: the speaker's play rate (the PCM fed must already be at it)."""
        from pepin.voice_live.audio import Pacer

        self.rate = rate
        self.clock = clock
        self.playout = PlayoutClock(speaker, rate, clock)
        self.pacer = Pacer(
            self.playout, rate=rate, lead_s=lead_s, block_s=block_s, on_level=on_level, clock=clock
        )
        self._lock = threading.Lock()
        self._fed = 0.0  # stream seconds fed (aligned to the played clock at every flush)
        self._kind: Kind = "reply"
        self._marks: dict[tuple[int, int], list[float]] = {}  # (turn, sentence) -> [start, end]
        self.flushed_at: float | None = None

    def start(self) -> threading.Event:
        """Run the pacer on its thread; set the returned event to stop it."""
        return self.pacer.start()

    def feed(self, pcm: bytes, kind: Kind, turn: int, index: int) -> None:
        """Queue a chunk of sentence ``index`` of reply ``turn`` (a backchannel: index -1)."""
        if not pcm:
            return
        duration = len(pcm) / SAMPLE_BYTES / self.rate
        with self._lock:
            start = self._fed
            self._fed += duration
            self._kind = kind
            if kind == "reply":
                mark = self._marks.setdefault((turn, index), [start, start])
                mark[1] = self._fed
            self.pacer.feed(pcm)

    def end(self) -> None:
        """The turn's audio is complete: play out what is queued."""
        self.pacer.end()

    def flush(self) -> float:
        """Barge-in: silence the speaker now; the stream seconds that had played."""
        with self._lock:
            played = self.playout.played()
            self.pacer.flush()
            self._fed = self.playout.sent_s
            self.flushed_at = self.clock()
        return played

    def playback(self) -> Playback:
        """What the speaker is doing: nothing, a reply, or a backchannel."""
        return self._kind if self.pacer.speaking() else "silent"

    def speaking(self) -> bool:
        """Audio is queued or playing (with the room's tail)."""
        return self.pacer.speaking()

    def span(self, turn: int) -> tuple[float, float] | None:
        """Where reply ``turn``'s audio sits in the stream: its first and last second."""
        with self._lock:
            spans = [v for (t, _), v in self._marks.items() if t == turn]
        if not spans:
            return None
        return min(s for s, _ in spans), max(e for _, e in spans)

    def heard(self, turn: int, played: float, texts: list[str]) -> Played:
        """Which of reply ``turn``'s sentences (``texts``, in order) had played by stream second
        ``played``: the whole ones, then the words of the one cut, in proportion."""
        with self._lock:
            marks = {i: tuple(v) for (t, i), v in self._marks.items() if t == turn}
        whole = 0
        partial = ""
        for index, text in enumerate(texts):
            mark = marks.get(index)
            if mark is None:
                break
            start, end = mark
            if end <= played + 0.05:
                whole = index + 1
                continue
            if start < played:
                words = text.split()
                share = (played - start) / max(end - start, 0.07 * len(text))
                partial = " ".join(words[: round(len(words) * min(1.0, share))])
            break
        return Played(whole, partial)

    def forget(self, turn: int) -> None:
        """Drop turn ``turn``'s marks (its line is written)."""
        with self._lock:
            for key in [k for k in self._marks if k[0] <= turn]:
                del self._marks[key]
