"""Barge-in detectors: is the person cutting in while the robot speaks?

The ears hand every mic frame to one :class:`BargeInDetector` (``barge_in.detector``); while a
reply plays, the frame that confirms a barge-in gets back the reason, and the ears flush the
speaker. Three detectors:

- ``level`` (:class:`LevelDetector`): a frame louder than the room's floor by ``over_floor_db``
  (and at least ``min_dbfs``) is voice; ``hold_s`` of voice is a barge-in. Any loud sound will
  do: a knock, a door, a cough.
- ``vad`` (:class:`VadDetector`): Silero VAD's speech probability instead of loudness, at least
  ``vad_min_dbfs`` and clear of the robot's own predicted echo (what the speaker played
  ``echo_lag_s`` ago, less ``echo_loss_db``) by ``echo_margin_db``; ``vad_hold_s`` of it.
- ``words`` (:class:`WordsDetector`): a voice gate held ``words_listen_s`` sends the voice so
  far to Whisper; the barge-in is its first word that is not a backchannel (mm-hm, uh-huh), not one
  of Whisper's noise lines and not the robot's own sentence heard back.

A quiet frame inside a run takes back half a frame (:class:`Run`): speech dips between
syllables. Detectors see every frame, also while the robot is silent, so a model's state
follows the room.
"""

from __future__ import annotations

import concurrent.futures
import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.voice_live.wake import is_hallucination, words

if TYPE_CHECKING:
    from pepin.voice_cascade.config import CascadeConfig

FRAME_S = FRAME_SAMPLES / RATE
SILERO_REPO, SILERO_FILE = "onnx-community/silero-vad", "onnx/model.onnx"  # v5, MIT, 2.2 MB
# v6.2.3 from Silero's own repository (MIT, 2.3 MB), pinned by tag and checked by its SHA-256
SILERO_V6 = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx",
    "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
)
MODEL_CACHE = Path("~/.cache/pepin").expanduser()
# words that keep the listener's turn going rather than take it (and Whisper's spellings)
BACKCHANNELS = frozenset(
    ("угу", "ага", "мм", "ммм", "хм", "ну", "да", "ок", "окей", "ясно", "понятно", "ааа", "м")  # noqa: RUF001
)
# sounds that are not words: laughter, a growl, exclamations, hesitation (Whisper's spellings)
NOT_WORDS = re.compile("(?:х[аеиоэы]|гр+|[ауоэ]х|ой|ай|э+м*|м+|хм+|а+|о+)")  # noqa: RUF001
NO_ECHO_DB = -120.0  # the predicted echo when nothing has played

Samples = npt.NDArray[np.float32]
EchoLevel = Callable[[float], float]  # mic time -> the robot's predicted echo there, dBFS
PlayedLevel = Callable[[float, float], float]  # (t0, t1) -> the loudest block played, dBFS


@dataclass(frozen=True)
class Frame:
    """One 20 ms mic frame as a detector sees it: s16le PCM, its laptop time, its level and
    the room's floor (dBFS) the gate holds."""

    pcm: bytes
    t: float
    db: float
    floor_db: float


class BargeInDetector(Protocol):
    """Decides, frame by frame, that the person is cutting in."""

    preroll_s: float  # the speech before the trigger the gate is given back

    def feed(self, frame: Frame, speaking: bool) -> str | None:
        """One frame; while the robot speaks, the reason when this frame confirms a
        barge-in (None otherwise)."""
        ...

    def reset(self) -> None:
        """Forget the run in progress (a barge-in was acted on)."""
        ...


class VoiceTest(Protocol):
    """Whether one frame is the person's voice (called once per frame, in order)."""

    def voiced(self, frame: Frame) -> bool:
        """Whether ``frame`` counts as voice."""
        ...


class SpeechProbability(Protocol):
    """A voice-activity model: the speech probability of each ``chunk`` samples (float, 16 kHz)
    in stream order."""

    chunk: int

    def __call__(self, samples: Samples) -> float:
        """The probability of speech in the next ``chunk`` samples."""
        ...


class Pending(Protocol):
    """A transcript on its way."""

    def result(self, t: float) -> str | None:
        """The words once ready at laptop time ``t`` (None while not ready)."""
        ...

    def cancel(self) -> None:
        """Not needed any more."""
        ...


Reader = Callable[[bytes, float], Pending]  # (16 kHz s16le PCM, laptop time) -> its words


class Run:
    """Seconds of voice in the run in progress: a voiced frame adds its length, a quiet one
    takes half a frame back."""

    def __init__(self, hold_s: float) -> None:
        """``hold_s``: the voice that completes the run."""
        self.hold_s = hold_s
        self.s = 0.0

    def add(self, voiced: bool) -> bool:
        """One frame; whether the run has reached ``hold_s``."""
        self.s = self.s + FRAME_S if voiced else max(0.0, self.s - FRAME_S / 2)
        return self.s >= self.hold_s - 1e-9

    def clear(self) -> None:
        """Back to no voice."""
        self.s = 0.0


class LevelDetector:
    """Loudness over the room's floor: a frame ``over_floor_db`` above it and at least
    ``min_dbfs`` is voice, ``hold_s`` of voice a barge-in."""

    preroll_s = 0.6

    def __init__(self, *, over_floor_db: float, min_dbfs: float, hold_s: float) -> None:
        """Thresholds in dB and seconds."""
        self.over_floor_db, self.min_dbfs = over_floor_db, min_dbfs
        self._run = Run(hold_s)

    def threshold_db(self, floor_db: float) -> float:
        """The level a voice must pass over a room at ``floor_db``."""
        return max(floor_db + self.over_floor_db, self.min_dbfs)

    def voiced(self, frame: Frame) -> bool:
        """Louder than the threshold."""
        return frame.db > self.threshold_db(frame.floor_db)

    def feed(self, frame: Frame, speaking: bool) -> str | None:
        """See :class:`BargeInDetector`."""
        if not speaking:
            self._run.clear()
            return None
        if not self._run.add(self.voiced(frame)):
            return None
        self._run.clear()
        return f"level {frame.db:.0f} dBFS"

    def reset(self) -> None:
        """See :class:`BargeInDetector`."""
        self._run.clear()


class SileroVad:
    """Silero VAD v5 (MIT) through onnxruntime on one CPU thread: the speech probability of
    each 32 ms chunk, the last 64 samples of the chunk before in front of it as the model
    expects; ~0.1 ms a call on the M5. Fetched from Hugging Face once (2.2 MB)."""

    chunk = 512
    context = 64

    def __init__(self, path: Path | None = None) -> None:
        """``path``: a local ONNX file instead of the Hub's."""
        import onnxruntime as ort

        if path is None:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(SILERO_REPO, SILERO_FILE))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session: Any = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._rate = np.array(RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """A new stream."""
        self._state: Samples = np.zeros((2, 1, 128), dtype=np.float32)
        self._context: Samples = np.zeros(self.context, dtype=np.float32)

    def __call__(self, samples: Samples) -> float:
        """See :class:`SpeechProbability`."""
        x = np.concatenate([self._context, samples])[None, :].astype(np.float32)
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._rate}
        )
        self._context = samples[-self.context :]
        return float(out[0][0])


class VadDetector:
    """A voice-activity model instead of loudness: a frame is voice when the model's latest
    probability is at least ``threshold``, the frame is at least ``min_dbfs`` and it stands
    ``echo_margin_db`` over the robot's predicted echo (``echo``); ``hold_s`` of voice is a
    barge-in."""

    preroll_s = 0.6

    def __init__(
        self,
        vad: SpeechProbability,
        *,
        threshold: float = 0.5,
        hold_s: float = 0.2,
        min_dbfs: float = -50.0,
        echo: EchoLevel | None = None,
        echo_margin_db: float = 10.0,
    ) -> None:
        """``echo``: the predicted echo at a mic time (None: no echo guard)."""
        self.vad, self.threshold, self.min_dbfs = vad, threshold, min_dbfs
        self.echo, self.echo_margin_db = echo, echo_margin_db
        self._run = Run(hold_s)
        self._buffer: Samples = np.zeros(0, dtype=np.float32)
        self.p = 0.0  # the model's latest probability

    def voiced(self, frame: Frame) -> bool:
        """See :class:`VoiceTest`; runs the model on every whole chunk the frame completes."""
        samples = np.frombuffer(frame.pcm, dtype="<i2").astype(np.float32) / 32768
        self._buffer = np.concatenate([self._buffer, samples])
        n = self.vad.chunk
        while len(self._buffer) >= n:
            self.p = self.vad(self._buffer[:n])
            self._buffer = self._buffer[n:]
        if self.p < self.threshold or frame.db < self.min_dbfs:
            return False
        return self.echo is None or frame.db >= self.echo(frame.t) + self.echo_margin_db

    def feed(self, frame: Frame, speaking: bool) -> str | None:
        """See :class:`BargeInDetector`."""
        voiced = self.voiced(frame)  # the model follows the stream, speaking or not
        if not speaking:
            self._run.clear()
            return None
        if not self._run.add(voiced):
            return None
        self._run.clear()
        return f"vad p {self.p:.2f} at {frame.db:.0f} dBFS"

    def reset(self) -> None:
        """See :class:`BargeInDetector`."""
        self._run.clear()


class FuturePending:
    """A :class:`Pending` over a thread pool's future (a failed call reads as no words)."""

    def __init__(self, future: concurrent.futures.Future[str]) -> None:
        """``future``: the transcriber's call."""
        self.future = future

    def result(self, t: float) -> str | None:
        """See :class:`Pending`."""
        if not self.future.done():
            return None
        if self.future.cancelled() or self.future.exception() is not None:
            return ""
        return self.future.result()

    def cancel(self) -> None:
        """See :class:`Pending`."""
        self.future.cancel()


def own_echo(said: list[str], own: str) -> bool:
    """Whether ``said`` (two words or more) is a run of the robot's own text ``own``, word for
    word by their first five letters (Whisper's endings drift)."""
    if len(said) < 2:
        return False
    mine = [w[:5] for w in words(own)]
    heard = [w[:5] for w in said]
    n = len(heard)
    return any(mine[i : i + n] == heard for i in range(len(mine) - n + 1))


def interrupting_words(text: str, own: str) -> list[str]:
    """The words of a transcript that take the turn: none for Whisper's noise lines, the
    robot's own text heard back, or backchannels, laughter and exclamations alone."""
    said = words(text)
    if not said or is_hallucination(text) or own_echo(said, own):
        return []
    if all(w in BACKCHANNELS or NOT_WORDS.fullmatch(w) for w in said):
        return []
    return said


class WordsDetector:
    """A voice gate confirmed by Whisper: once ``gate`` has held ``listen_s`` of voice, the
    voice so far (from ``lead_s`` before the run, at most ``window_s``) goes to ``reader``; the
    barge-in is a transcript with a word that takes the turn (:func:`interrupting_words`
    against ``own_text``, the robot's sentences in play). While the voice holds, an answer
    without such a word is followed by a new call, at most one every ``every_s``."""

    def __init__(
        self,
        gate: VoiceTest,
        reader: Reader,
        own_text: Callable[[], str],
        *,
        listen_s: float = 0.4,
        window_s: float = 1.5,
        every_s: float = 0.3,
        lead_s: float = 0.3,
    ) -> None:
        """Seconds as above."""
        self.gate, self.reader, self.own_text = gate, reader, own_text
        self.window_s, self.every_s, self.lead_s = window_s, every_s, lead_s
        self.preroll_s = listen_s + 1.2  # the voice before a call, and two calls' wait
        self._run = Run(listen_s)
        self._frames: deque[bytes] = deque(maxlen=round((window_s + lead_s) / FRAME_S))
        self._since = 0  # frames since the run began
        self._pending: Pending | None = None
        self._next_s = -1e9
        self.calls = 0  # transcripts asked for

    def feed(self, frame: Frame, speaking: bool) -> str | None:
        """See :class:`BargeInDetector`."""
        self._frames.append(frame.pcm)
        voiced = self.gate.voiced(frame)
        if not speaking:
            self.reset()
            return None
        if self._pending is not None:
            text = self._pending.result(frame.t)
            if text is not None:
                self._pending = None
                said = interrupting_words(text, self.own_text())
                if said:
                    self.reset()
                    return "words " + " ".join(said)
        self._since = self._since + 1 if self._run.s > 0 or voiced else 0
        if self._run.add(voiced) and self._pending is None and frame.t >= self._next_s:
            frames = min(len(self._frames), self._since + round(self.lead_s / FRAME_S))
            frames = min(frames, round(self.window_s / FRAME_S))
            self.calls += 1
            self._pending = self.reader(b"".join(list(self._frames)[-frames:]), frame.t)
            self._next_s = frame.t + self.every_s
        return None

    def reset(self) -> None:
        """See :class:`BargeInDetector`; a transcript on its way is dropped."""
        self._run.clear()
        self._since = 0
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None


def echo_level(
    played: PlayedLevel, *, lag_s: float, loss_db: float, spread_s: float = 0.1
) -> EchoLevel:
    """The robot's predicted echo at a mic time: the loudest block the speaker played within
    ``spread_s`` of ``lag_s`` earlier, less ``loss_db`` (the room and the array's echo
    canceller together)."""

    def level(t: float) -> float:
        db = played(t - lag_s - spread_s, t - lag_s + spread_s)
        return NO_ECHO_DB if db <= NO_ECHO_DB else db - loss_db

    return level


def fetch(url: str) -> bytes:
    """The body at ``url``."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=30) as response:
        body: bytes = response.read()
    return body


def model_path(
    name: str | None, cache: Path = MODEL_CACHE, get: Callable[[str], bytes] = fetch
) -> Path | None:
    """The VAD model ``barge_in.vad_model`` names: None or ``silero-v5`` for the Hub's v5
    (None: :class:`SileroVad` fetches it), ``silero-v6`` for v6.2.3 (fetched into ``cache``
    once and checked), or a local file."""
    if name in (None, "", "silero-v5"):
        return None
    if name != "silero-v6":
        return Path(name).expanduser()
    url, sha256 = SILERO_V6
    path = cache / f"silero_vad_v6.2.3_{sha256[:12]}.onnx"
    if not path.exists():
        import hashlib

        body = get(url)
        if hashlib.sha256(body).hexdigest() != sha256:
            raise ValueError(f"{url}: not the pinned Silero VAD v6.2.3 (SHA-256 differs)")
        cache.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        partial.write_bytes(body)
        partial.replace(path)
    return path


def build_detector(
    config: CascadeConfig,
    *,
    reader: Reader | None = None,
    own_text: Callable[[], str] = str,
    played: PlayedLevel | None = None,
    vad: SpeechProbability | None = None,
) -> BargeInDetector:
    """The detector ``barge_in.detector`` names. ``reader``: Whisper for ``words``; ``own_text``:
    the robot's sentences in play; ``played``: the speaker's level history for the echo guard
    (None: no guard); ``vad``: the model (None: Silero, loaded here from ``vad_model``, a local
    ONNX file, or the Hub's v5)."""
    level = LevelDetector(
        over_floor_db=config.barge_in_over_floor_db,
        min_dbfs=config.barge_in_min_dbfs,
        hold_s=config.barge_in_hold_s,
    )
    name = config.barge_in_detector
    if name == "level":
        return level
    gate: VoiceTest = level
    if name == "vad" or config.barge_in_words_gate == "vad":
        echo = None
        if played is not None:
            echo = echo_level(
                played, lag_s=config.barge_in_echo_lag_s, loss_db=config.barge_in_echo_loss_db
            )
        voice = VadDetector(
            vad if vad is not None else SileroVad(model_path(config.barge_in_vad_model)),
            threshold=config.barge_in_vad_threshold,
            hold_s=config.barge_in_vad_hold_s,
            min_dbfs=config.barge_in_vad_min_dbfs,
            echo=echo,
            echo_margin_db=config.barge_in_echo_margin_db,
        )
        if name == "vad":
            return voice
        gate = voice
    if reader is None:
        raise ValueError("barge_in.detector words needs a transcriber")
    return WordsDetector(
        gate,
        reader,
        own_text,
        listen_s=config.barge_in_words_listen_s,
        window_s=config.barge_in_words_window_s,
        every_s=config.barge_in_words_every_s,
    )
