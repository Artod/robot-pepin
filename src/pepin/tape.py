"""The tape of a run: the seconds before it started, and everything until it ended.

A recorder used to be a separate process started per goal. rclpy needs about four seconds to
come up on this board, so the first seconds of every drive — the initial turn, exactly the part
worth watching — were never on tape, and the process cost about 100 MB on a 1.4 GB board. This
class holds the last seconds in memory instead: when a run starts it writes that prelude first
and then every record as it arrives, so a tape begins before the command did.

Records are `pepin.recording.SessionRecorder`'s (topics scan/pose/loc/plan/cmd); the writer is
injected, so this is a pure object a test can drive with a StringIO.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TextIO

PRELUDE_S = 15.0
MAX_RUN_S = 900.0  # a tape nobody stops is a bug, not a feature (two orphans wrote for 40 minutes)
FSYNC_EVERY_S = 2.0


def _open(path: Path) -> TextIO:
    """Line-buffered append: every record reaches the OS as it is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", buffering=1)  # closed by RunTape.stop, not here


def _sync(stream: TextIO) -> None:
    """Force the file to the card, so a power cut costs at most the last seconds."""
    stream.flush()
    with contextlib.suppress(OSError, ValueError):  # a test's StringIO has no descriptor
        os.fsync(stream.fileno())


def next_run_number(directory: Path) -> int:
    """The next run's number in ``directory``, so a drive has a name short enough to say aloud.

    Kept in a counter file next to the recordings; a lost or corrupt counter falls back to the
    numbered files already there, which only ever repeats a number after they are deleted.
    """
    counter = directory / ".run_seq"
    try:
        number = int(counter.read_text().strip()) + 1
    except (OSError, ValueError):
        number = 1 + len(list(directory.glob("[0-9][0-9][0-9][0-9]_*.jsonl")))
    with contextlib.suppress(OSError):
        directory.mkdir(parents=True, exist_ok=True)
        counter.write_text(f"{number}\n")
    return number


class RunTape:
    """Keeps the last seconds of records and writes a run's jsonl from the moment it starts."""

    def __init__(
        self,
        prelude_s: float = PRELUDE_S,
        max_run_s: float = MAX_RUN_S,
        clock: Callable[[], float] = time.monotonic,
        opener: Callable[[Path], TextIO] = _open,
        sync: Callable[[TextIO], None] = _sync,
    ) -> None:
        self._prelude_s = prelude_s
        self._max_run_s = max_run_s
        self._clock = clock
        self._opener = opener
        self._sync = sync
        self._buffer: deque[tuple[float, Mapping[str, Any]]] = deque()
        self._stream: TextIO | None = None
        self._started = 0.0
        self._last_sync = 0.0
        self.written = 0

    @property
    def recording(self) -> bool:
        """True while a run is being written."""
        return self._stream is not None

    def add(self, record: Mapping[str, Any]) -> None:
        """Take one record: written straight through during a run, remembered otherwise."""
        now = self._clock()
        if self._stream is None:
            self._buffer.append((now, record))
            horizon = now - self._prelude_s
            while self._buffer and self._buffer[0][0] < horizon:
                self._buffer.popleft()
            return
        if now - self._started > self._max_run_s:
            self.stop()
            self._buffer.append((now, record))
            return
        self._write(record)
        if now - self._last_sync > FSYNC_EVERY_S:
            self._last_sync = now
            self._sync(self._stream)

    def start(self, path: Path) -> Path:
        """Open ``path`` and put the remembered seconds in it; returns the path written to."""
        self.stop()
        self._stream = self._opener(path)
        self._started = self._last_sync = self._clock()
        self.written = 0
        for _, record in self._buffer:
            self._write(record)
        self._buffer.clear()
        return path

    def stop(self) -> None:
        """Close the run's file (flushed and synced); harmless when nothing is being recorded."""
        stream, self._stream = self._stream, None
        if stream is None:
            return
        self._sync(stream)
        stream.close()

    def _write(self, record: Mapping[str, Any]) -> None:
        assert self._stream is not None
        self._stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.written += 1
