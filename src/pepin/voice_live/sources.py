"""Where the voice loop hears and speaks: the board's array over the audio link (with the
reconnect and the keepalive it needs), or phrases synthesised on the Mac and a WAV file, for a
test that needs no person and makes no sound in the flat."""

from __future__ import annotations

import contextlib
import logging
import random
import struct
import subprocess
import tempfile
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path

from pepin.audio_link import FRAME_SAMPLES, RATE, AudioClient

logger = logging.getLogger(__name__)

FrameSink = Callable[[bytes, float], None]  # (20 ms of s16le at 16 kHz, laptop monotonic)


class BoardLink:
    """One audio link to the board for both directions: frames in (:meth:`start`), speech out
    (``play``/``play_end``/``flush``, the :class:`pepin.audio_link.Speaker` protocol). A link
    that goes silent for three seconds is replaced; a status probe every 2 s keeps the board
    from closing a link that only listens."""

    def __init__(self, host: str, connect: Callable[[str], AudioClient] | None = None) -> None:
        """``connect`` opens a started, listening client (the real one by default)."""
        self.host = host
        self._connect = connect or (lambda h: AudioClient(h).start(listen=True))
        self._client = self._connect(host)
        self._stop = threading.Event()

    @property
    def play_rate(self) -> int:
        """The board's playback rate."""
        return self._client.play_rate

    def start(self, sink: FrameSink) -> None:
        """Deliver frames to ``sink`` from a reader thread; keep the link alive."""
        threading.Thread(target=self._pump, args=(sink,), name="mic", daemon=True).start()
        threading.Thread(target=self._keepalive, name="keepalive", daemon=True).start()

    def play(self, pcm: bytes) -> None:
        """Queue speech on the board's speaker."""
        self._client.play(pcm)

    def play_end(self) -> None:
        """The utterance is complete."""
        self._client.play_end()

    def flush(self) -> None:
        """Silence the speaker now."""
        self._client.flush()

    def close(self) -> None:
        """Stop the threads and drop the link."""
        self._stop.set()
        self._client.close()

    def _pump(self, sink: FrameSink) -> None:
        silent = 0
        while not self._stop.is_set():
            frame = self._client.next_frame(timeout_s=1.0)
            if frame is None:
                silent += 1
                if silent >= 3 and not self._stop.is_set():
                    print("  (audio link lost: reconnecting)", flush=True)
                    with contextlib.suppress(Exception):
                        self._client.close()
                    try:
                        self._client = self._connect(self.host)
                    except OSError as error:
                        logger.warning("audio link: %s", error)
                        time.sleep(1.0)
                    silent = 0
                continue
            silent = 0
            sink(frame.pcm, time.monotonic())

    def _keepalive(self) -> None:
        while not self._stop.wait(2.0):
            with contextlib.suppress(Exception):
                self._client._send({"cmd": "status"})


def synthesize(text: str, voice: str = "Milena", rate_wpm: int = 185) -> bytes:
    """``text`` spoken by macOS ``say`` as 16 kHz s16le mono."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "say.wav"
        subprocess.run(
            ["say", "-v", voice, "-r", str(rate_wpm), "--file-format=WAVE",
             "--data-format=LEI16@16000", "-o", str(out), text],
            check=True,
        )  # fmt: skip
        with wave.open(str(out)) as w:
            return bytes(w.readframes(w.getnframes()))


class ScriptedMic:
    """A mic that says the given utterances one after another, each when ``ready()`` has held
    for ``gap_s`` (the robot idle or listening, not speaking or thinking); room hiss between."""

    def __init__(
        self,
        utterances: list[bytes],
        ready: Callable[[], bool],
        *,
        gap_s: float = 1.2,
        lead_s: float = 1.0,
    ) -> None:
        """``utterances``: 16 kHz s16le PCM each."""
        self.utterances = list(utterances)
        self.ready = ready
        self.gap_s, self.lead_s = gap_s, lead_s
        self.done = threading.Event()  # every utterance has been said
        self._stop = threading.Event()

    def start(self, sink: FrameSink) -> None:
        """Deliver frames at real time from a thread."""
        threading.Thread(target=self._run, args=(sink,), name="scripted-mic", daemon=True).start()

    def close(self) -> None:
        """Stop."""
        self._stop.set()

    def _run(self, sink: FrameSink) -> None:
        rng = random.Random(0)
        step = FRAME_SAMPLES / RATE
        frame_bytes = FRAME_SAMPLES * 2
        start = time.monotonic()
        n = 0
        queue = list(self.utterances)
        speaking: bytes = b""
        ready_since: float | None = None
        while not self._stop.is_set():
            now = time.monotonic()
            if not speaking and queue:
                ready = self.ready() and now - start > self.lead_s
                ready_since = (ready_since or now) if ready else None
                if ready_since is not None and now - ready_since >= self.gap_s:
                    speaking, ready_since = queue.pop(0), None
                    print(f"  (scripted mic: {len(speaking) / 2 / RATE:.1f} s)", flush=True)
            if speaking:
                pcm = speaking[:frame_bytes].ljust(frame_bytes, b"\x00")
                speaking = speaking[frame_bytes:]
                if not speaking and not queue:
                    self.done.set()
            else:
                pcm = struct.pack(f"<{FRAME_SAMPLES}h", *(rng.randint(-3, 3) for _ in range(320)))
            sink(pcm, time.monotonic())
            n += 1
            time.sleep(max(0.0, start + n * step - time.monotonic()))


class WavSpeaker:
    """A speaker that writes what it is given to a WAV (no sound): the
    :class:`pepin.audio_link.Speaker` protocol at ``play_rate``."""

    def __init__(self, path: Path, play_rate: int = RATE) -> None:
        """Create ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.play_rate = play_rate
        self._wav = wave.open(str(path), "wb")  # noqa: SIM115 -- open as long as the speaker
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(play_rate)
        self._lock = threading.Lock()
        self.flushes = 0

    def play(self, pcm: bytes) -> None:
        """Append to the file."""
        with self._lock:
            self._wav.writeframes(pcm)

    def play_end(self) -> None:
        """Nothing to do."""

    def flush(self) -> None:
        """Counted."""
        self.flushes += 1

    def close(self) -> None:
        """Finish the file."""
        with self._lock:
            self._wav.close()
