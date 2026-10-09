"""The latency bench: recorded utterances replayed into the real ears, Whisper, brain and voice,
the robot's speech written to a WAV, and the turns' timelines summed as p50/p90 per stage.

The utterances come from a folder of 16 kHz mono WAVs (one utterance each, in name order:
scratch/prosody_1008/wavs holds Artem's 55 of 2026-10-08) or are cut from a Live session's
in.wav by the server's voice-activity offsets in its events.jsonl. Between utterances the mic
plays the utterance's own quietest frame, so the gate's floor is the room the speech was
recorded in, not digital silence; the next utterance starts once the robot has answered and
gone quiet for ``gap_s``.
"""

from __future__ import annotations

import json
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.voice_live.sources import FrameSink

FRAME_BYTES = FRAME_SAMPLES * 2
PAD_S = 0.2  # around a voice-activity cut


def read_wav(path: Path) -> bytes:
    """A 16 kHz mono s16le WAV's samples; ``ValueError`` for any other format."""
    with wave.open(str(path)) as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, RATE):
            raise ValueError(f"{path}: need 16 kHz mono 16-bit")
        return bytes(w.readframes(w.getnframes()))


def _seconds(text: str) -> float:
    return float(str(text).rstrip("s"))


def session_cuts(session: Path) -> list[tuple[str, bytes]]:
    """A Live session's utterances: in.wav cut by the voice-activity START/END offsets in its
    events.jsonl (padded by 0.2 s), each with the server's transcript."""
    pcm = read_wav(session / "in.wav")
    spans: list[list[object]] = []
    for line in (session / "events.jsonl").read_text().splitlines():
        row = json.loads(line)
        message = row.get("message", {}) if row.get("event") == "server" else {}
        activity = message.get("voice_activity")
        if activity and activity.get("voice_activity_type") == "ACTIVITY_START":
            spans.append([_seconds(activity["audio_offset"]), None, ""])
        elif activity and activity.get("voice_activity_type") == "ACTIVITY_END" and spans:
            spans[-1][1] = _seconds(activity["audio_offset"])
        said = message.get("server_content", {}).get("input_transcription")
        if said and spans:
            spans[-1][2] = str(spans[-1][2]) + str(said.get("text", ""))
    out = []
    for start, end, text in spans:
        if not isinstance(start, float) or not isinstance(end, float):
            continue
        a = max(0, int((start - PAD_S) * RATE)) * 2
        b = min(len(pcm), int((end + PAD_S) * RATE) * 2)
        out.append((str(text).strip() or f"{start:.1f}s", pcm[a:b]))
    return out


def utterances(path: Path, n: int | None = None) -> list[tuple[str, bytes]]:
    """(name, PCM) of the first ``n`` utterances of a WAV folder or a Live session folder."""
    if (path / "events.jsonl").is_file() and (path / "in.wav").is_file():
        found = session_cuts(path)
    else:
        found = [(f.stem, read_wav(f)) for f in sorted(path.glob("*.wav"))]
    if not found:
        raise ValueError(f"{path}: no utterances (a folder of WAVs or a live_* session)")
    return found[:n] if n else found


def quietest_frame(pcm: bytes) -> bytes:
    """The utterance's quietest 20 ms: the room it was recorded in."""
    frames = [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES)]
    if not frames:
        return b"\x00\x00" * FRAME_SAMPLES

    def energy(frame: bytes) -> float:
        x = np.frombuffer(frame, dtype="<i2").astype(float)
        return float(np.mean(x * x))

    return min(frames, key=energy)


class ReplayMic:
    """A mic that says recorded utterances one after another, each when ``ready()`` has held
    for ``gap_s``; the room's own quiet between them. Frames at real time, from a thread."""

    def __init__(
        self, utterances: list[bytes], ready: Callable[[], bool], *, gap_s: float = 1.0
    ) -> None:
        """``utterances``: 16 kHz s16le PCM each."""
        self.utterances = list(utterances)
        self.ready = ready
        self.gap_s = gap_s
        self.done = threading.Event()  # every utterance has been said
        self.said = 0
        self._stop = threading.Event()

    def start(self, sink: FrameSink) -> None:
        """Deliver frames at real time from a thread."""
        threading.Thread(target=self._run, args=(sink,), name="replay-mic", daemon=True).start()

    def close(self) -> None:
        """Stop."""
        self._stop.set()

    def _run(self, sink: FrameSink) -> None:
        step = FRAME_SAMPLES / RATE
        start = time.monotonic()
        n = 0
        pending = list(self.utterances)
        filler = quietest_frame(pending[0]) if pending else b"\x00\x00" * FRAME_SAMPLES
        speaking = b""
        ready_since: float | None = None
        while not self._stop.is_set():
            now = time.monotonic()
            if not speaking and pending:
                ready = self.ready() and now - start > 1.0
                ready_since = (ready_since or now) if ready else None
                if ready_since is not None and now - ready_since >= self.gap_s:
                    speaking, ready_since = pending.pop(0), None
                    filler = quietest_frame(speaking)
                    self.said += 1
            if speaking:
                frame = speaking[:FRAME_BYTES].ljust(FRAME_BYTES, b"\x00")
                speaking = speaking[FRAME_BYTES:]
                if not speaking and not pending:
                    self.done.set()
            else:
                frame = filler
            sink(frame, time.monotonic())
            n += 1
            time.sleep(max(0.0, start + n * step - time.monotonic()))
