"""A conversation's record: ``data/voice/<day>/cascade_<HHMMSS>/`` with the mic (in.wav, 16 kHz),
the robot's voice (out.wav, at the voice's rate), every event as a JSON line and the summary;
the summary also goes to the day's ``cascade.jsonl``. Utterances heard outside a conversation go
to the day's ``cascade_heard.jsonl``."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from pepin.audio_link import RATE
from pepin.voice_live.audio import WavWriter


def day_dir(root: Path, wall: float) -> Path:
    """``root/<YYYYMMDD>`` of the local day at ``wall``."""
    return root / time.strftime("%Y%m%d", time.localtime(wall))


def append_line(path: Path, row: dict[str, Any]) -> None:
    """One JSON line appended to ``path`` (its directory made)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


class CascadeLog:
    """One conversation's directory. Thread-safe: the mic, the voice and the loop write here."""

    def __init__(self, root: Path, wall: float, t0: float, voice_rate: int) -> None:
        """Under ``root/<day>/cascade_<HHMMSS>``; ``t0`` is the monotonic zero of the events."""
        stamp = time.localtime(wall)
        self.day_dir = day_dir(root, wall)
        name = f"cascade_{time.strftime('%H%M%S', stamp)}"
        self.dir = self.day_dir / name
        suffix = 1
        while self.dir.exists():
            suffix += 1
            self.dir = self.day_dir / f"{name}_{suffix}"
        self.dir.mkdir(parents=True)
        self.t0 = t0
        self.mic = WavWriter(self.dir / "in.wav", RATE)
        self.voice = WavWriter(self.dir / "out.wav", voice_rate)
        self._lock = threading.Lock()
        self._events = (self.dir / "events.jsonl").open("a")

    def event(self, kind: str, now: float, /, **fields: Any) -> None:
        """One line: what happened, seconds since the conversation opened, the details."""
        row = {"event": kind, "t": round(now - self.t0, 3), **fields}
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            if not self._events.closed:
                self._events.write(line)
                self._events.flush()

    def close(self, summary: dict[str, Any]) -> None:
        """Finish the WAVs; the summary beside the events and in the day's index."""
        self.mic.close()
        self.voice.close()
        with self._lock:
            self._events.close()
        line = json.dumps(summary, ensure_ascii=False, default=str)
        (self.dir / "summary.json").write_text(line + "\n")
        with (self.day_dir / "cascade.jsonl").open("a") as index:
            index.write(line + "\n")
